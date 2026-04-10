from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from hecsn.config.model_config import HECSNConfig
from hecsn.reporting.io import write_json_file
from hecsn.training.memory_consolidation_runner import (
    build_memory_consolidation_gate,
    collect_assemblies,
    mean_assembly_overlap,
    mean_reconstruction_error,
)
from hecsn.training.runner_utils import set_seed
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer


TASK_A_WINDOWS = (
    "alpha memory signal",
    "alpha plastic trace",
    "alpha stable concept",
)
TASK_B_WINDOWS = (
    "beta routing context",
    "beta semantic drift",
    "beta retrieval anchor",
)


def _pattern_examples(trainer: HECSNTrainer, windows: tuple[str, ...]) -> list[tuple[str, torch.Tensor]]:
    examples: list[tuple[str, torch.Tensor]] = []
    for raw_window in windows:
        pattern = trainer.encoder.feature_vector([ord(ch) for ch in raw_window]).to(torch.float32)
        examples.append((raw_window, pattern))
    return examples


def _finite_model_state(trainer: HECSNTrainer) -> bool:
    tensors = [
        trainer.model.competitive.prototypes,
        trainer.model.competitive.input_weights,
        trainer.model.competitive.W_project,
        trainer.model.W_assembly_project,
    ]
    local_plasticity = trainer.model.competitive.local_plasticity
    if local_plasticity is not None and local_plasticity.adex_neurons is not None:
        tensors.extend([local_plasticity.adex_neurons.V, local_plasticity.adex_neurons.w])
    return bool(all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors))


def _run_backend(backend: str, *, seed: int) -> dict[str, Any]:
    set_seed(seed)
    cfg = HECSNConfig(
        n_columns=12,
        column_latent_dim=24,
        bootstrap_tokens=0,
        memory_capacity=96,
        eta_competitive=0.05,
        eta_decay=0.0,
        input_weight_blend=0.0,
        plasticity_mode="local_stdp",
        plasticity_spike_backend=backend,
        micro_sleep_interval_tokens=10**9,
        deep_sleep_interval_tokens=10**9,
        deep_sleep_replay_steps=24,
        deep_sleep_candidate_pool=24,
        enable_learned_chunking=False,
    )
    trainer = HECSNTrainer(HECSNModelLite(cfg), cfg)
    task_a_examples = _pattern_examples(trainer, TASK_A_WINDOWS)
    task_b_examples = _pattern_examples(trainer, TASK_B_WINDOWS)
    task_a_eval = [pattern for _, pattern in task_a_examples]
    task_b_eval = [pattern for _, pattern in task_b_examples]

    post_spike_fractions: list[float] = []
    membrane_voltages: list[float] = []

    def _record_metrics(metrics: dict[str, Any]) -> None:
        post_spike_fractions.append(float(metrics.get("local_post_spike_fraction", 0.0)))
        membrane_voltages.append(float(metrics.get("local_mean_membrane_voltage", 0.0)))

    for _ in range(18):
        for raw_window, pattern in task_a_examples:
            _record_metrics(trainer.train_step(pattern, raw_window=raw_window))
    task_a_after_a = mean_reconstruction_error(trainer, task_a_eval)
    task_b_before_b = mean_reconstruction_error(trainer, task_b_eval)
    task_a_reference_assemblies = collect_assemblies(trainer, task_a_eval)
    task_b_reference_assemblies = collect_assemblies(trainer, task_b_eval)

    tagged_entries = trainer.tag_recent_memories(window_tokens=trainer.token_count, strength=3.0)
    anchored_columns = trainer.capture_recent_memory_anchors(window_tokens=trainer.token_count, strength=8.0)
    boundary_updates = trainer.run_sleep_maintenance(mode="deep", cycles=2)

    for _ in range(18):
        for raw_window, pattern in task_b_examples:
            _record_metrics(trainer.train_step(pattern, raw_window=raw_window))
    task_a_after_b = mean_reconstruction_error(trainer, task_a_eval)
    task_b_after_b = mean_reconstruction_error(trainer, task_b_eval)
    task_a_overlap_after_b = mean_assembly_overlap(task_a_reference_assemblies, collect_assemblies(trainer, task_a_eval))
    task_b_overlap_after_b = mean_assembly_overlap(task_b_reference_assemblies, collect_assemblies(trainer, task_b_eval))
    memory_before = trainer.model.memory_store.summary_stats()

    consolidation_updates = trainer.run_sleep_maintenance(mode="deep", cycles=4)
    task_a_after_consolidation = mean_reconstruction_error(trainer, task_a_eval)
    task_b_after_consolidation = mean_reconstruction_error(trainer, task_b_eval)
    task_a_overlap_after_consolidation = mean_assembly_overlap(
        task_a_reference_assemblies,
        collect_assemblies(trainer, task_a_eval),
    )
    task_b_overlap_after_consolidation = mean_assembly_overlap(
        task_b_reference_assemblies,
        collect_assemblies(trainer, task_b_eval),
    )
    memory_after = trainer.model.memory_store.summary_stats()

    memory_gate = build_memory_consolidation_gate(
        task_a_after_a=task_a_after_a,
        task_a_after_b=task_a_after_b,
        task_a_after_consolidation=task_a_after_consolidation,
        task_a_overlap_after_consolidation=task_a_overlap_after_consolidation,
    )

    mean_post_spike_fraction = float(sum(post_spike_fractions) / max(1, len(post_spike_fractions)))
    mean_membrane_voltage = float(sum(membrane_voltages) / max(1, len(membrane_voltages)))
    finite_model_state = _finite_model_state(trainer)
    local_plasticity = trainer.model.competitive.local_plasticity
    uses_adex = bool(local_plasticity is not None and local_plasticity.spike_backend == "adex")
    gate_pass = bool(
        memory_gate["pass"]
        and finite_model_state
        and (backend != "adex" or (uses_adex and mean_post_spike_fraction > 0.0))
    )
    return {
        "backend": backend,
        "uses_adex_post_spikes": uses_adex,
        "finite_model_state": finite_model_state,
        "mean_post_spike_fraction": mean_post_spike_fraction,
        "mean_membrane_voltage": mean_membrane_voltage,
        "task_a_after_a": float(task_a_after_a),
        "task_b_before_b": float(task_b_before_b),
        "task_a_after_b": float(task_a_after_b),
        "task_b_after_b": float(task_b_after_b),
        "task_a_after_consolidation": float(task_a_after_consolidation),
        "task_b_after_consolidation": float(task_b_after_consolidation),
        "task_a_overlap_after_b": float(task_a_overlap_after_b),
        "task_b_overlap_after_b": float(task_b_overlap_after_b),
        "task_a_overlap_after_consolidation": float(task_a_overlap_after_consolidation),
        "task_b_overlap_after_consolidation": float(task_b_overlap_after_consolidation),
        "tagged_entries": int(tagged_entries),
        "anchored_columns": int(anchored_columns),
        "boundary_updates": int(boundary_updates),
        "consolidation_updates": int(consolidation_updates),
        "memory_before": {
            "mean_consolidation_level": float(memory_before.get("mean_consolidation_level", 0.0)),
            "mean_fragility": float(memory_before.get("mean_fragility", 0.0)),
        },
        "memory_after": {
            "mean_consolidation_level": float(memory_after.get("mean_consolidation_level", 0.0)),
            "mean_fragility": float(memory_after.get("mean_fragility", 0.0)),
        },
        "memory_consolidation_gate": memory_gate,
        "gate_pass": gate_pass,
    }


def run_adex_consolidation_benchmark(*, output_dir: Path, seed: int = 7) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    proxy = _run_backend("proxy", seed=seed)
    adex = _run_backend("adex", seed=seed)
    summary = {
        "benchmark": "adex_consolidation_smoke",
        "seed": int(seed),
        "runtime_scope": {
            "mode": "local_plasticity_consolidation_comparison",
            "note": "This smoke validates that the optional AdEx spike backend stays healthy under the maintained replay/consolidation path.",
        },
        "backends": {
            "proxy": proxy,
            "adex": adex,
        },
        "comparison": {
            "task_a_after_consolidation_delta": float(adex["task_a_after_consolidation"] - proxy["task_a_after_consolidation"]),
            "task_a_overlap_after_consolidation_delta": float(
                adex["task_a_overlap_after_consolidation"] - proxy["task_a_overlap_after_consolidation"]
            ),
            "mean_consolidation_level_delta": float(
                adex["memory_after"]["mean_consolidation_level"] - proxy["memory_after"]["mean_consolidation_level"]
            ),
            "mean_fragility_delta": float(
                adex["memory_after"]["mean_fragility"] - proxy["memory_after"]["mean_fragility"]
            ),
        },
        "adex_consolidation_gate": {
            "pass": bool(proxy["gate_pass"] and adex["gate_pass"]),
            "thresholds": {
                "memory_consolidation_gate": True,
                "finite_model_state": True,
                "adex_mean_post_spike_fraction_gt": 0.0,
            },
        },
    }
    write_json_file(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AdEx consolidation smoke benchmark.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "phase7_adex_consolidation_smoke",
        help="Directory where the benchmark summary will be written.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Deterministic seed for the benchmark.")
    args = parser.parse_args()

    summary = run_adex_consolidation_benchmark(output_dir=args.output_dir, seed=args.seed)
    print(f"[adex_consolidation_smoke] pass={summary['adex_consolidation_gate']['pass']}")


if __name__ == "__main__":
    main()
