from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import torch

from marulho.config.model_config import MarulhoConfig
from marulho.reporting.io import write_json_file
from marulho.training.memory_consolidation_runner import (
    build_memory_consolidation_gate,
    collect_assemblies,
    mean_assembly_overlap,
)
from marulho.training.model import MarulhoModel
from marulho.training.runner_utils import set_seed
from marulho.training.trainer import MarulhoTrainer


def _pattern(config: MarulhoConfig, *indices: int) -> torch.Tensor:
    pattern = torch.zeros(config.input_dim, dtype=torch.float32)
    for index in indices:
        pattern[int(index)] = 1.0
    return pattern / pattern.sum().clamp(min=1e-8)


def _mean_reconstruction(
    trainer: MarulhoTrainer,
    items: list[tuple[str, torch.Tensor]],
) -> float:
    if not items:
        return float("nan")
    return float(
        sum(trainer.reconstruction_error(pattern) for _, pattern in items)
        / len(items)
    )


def _inject_anchor_replay_pressure(trainer: MarulhoTrainer) -> int:
    anchor_buckets = {int(bucket_id) for bucket_id in trainer.column_anchors}
    touched = 0
    store = trainer.model.memory_store
    for idx, bucket_id in enumerate(store.slow_bucket_ids):
        if bucket_id not in anchor_buckets:
            continue
        store.slow_capture_tag[idx] = max(1.0, float(store.slow_capture_tag[idx]))
        store.slow_local_prp[idx] = max(1.0, float(store.slow_local_prp[idx]))
        store.slow_consolidation_level[idx] = min(
            0.2,
            float(store.slow_consolidation_level[idx]),
        )
        touched += 1
    if touched:
        store._invalidate_summary_cache()
    return touched


def run_trial(
    *,
    seed: int,
    final_selector: str,
    n_columns: int,
    column_latent_dim: int,
    memory_capacity: int,
    task_repetitions: int,
    boundary_cycles: int,
    consolidation_cycles: int,
    replay_steps: int,
    candidate_pool: int,
) -> dict[str, Any]:
    set_seed(seed)
    config = MarulhoConfig(
        n_columns=n_columns,
        column_latent_dim=column_latent_dim,
        bootstrap_tokens=0,
        memory_capacity=memory_capacity,
        eta_competitive=0.05,
        eta_decay=0.0,
        input_weight_blend=0.0,
        micro_sleep_interval_tokens=10**9,
        deep_sleep_interval_tokens=10**9,
        deep_sleep_replay_steps=replay_steps,
        deep_sleep_candidate_pool=candidate_pool,
        enable_learned_chunking=False,
    )
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    task_a = [
        ("alpha memory signal", _pattern(config, 1, 2, 3)),
        ("alpha plastic trace", _pattern(config, 4, 5, 6)),
        ("alpha stable concept", _pattern(config, 7, 8, 9)),
    ]
    task_b = [
        ("beta routing context", _pattern(config, 20, 21, 22)),
        ("beta semantic drift", _pattern(config, 23, 24, 25)),
        ("beta retrieval anchor", _pattern(config, 26, 27, 28)),
    ]

    for _ in range(max(1, int(task_repetitions))):
        for raw_window, pattern in task_a:
            trainer.train_step(pattern, raw_window=raw_window)
    task_a_after_a = _mean_reconstruction(trainer, task_a)
    reference_assemblies = collect_assemblies(
        trainer,
        [pattern for _, pattern in task_a],
    )

    trainer.tag_recent_memories(window_tokens=trainer.token_count, strength=3.0)
    anchored_columns = trainer.capture_recent_memory_anchors(
        window_tokens=trainer.token_count,
        strength=8.0,
    )
    trainer.run_sleep_maintenance(mode="deep", cycles=boundary_cycles)

    for _ in range(max(1, int(task_repetitions))):
        for raw_window, pattern in task_b:
            trainer.train_step(pattern, raw_window=raw_window)
    task_a_after_b = _mean_reconstruction(trainer, task_a)

    pressure_entries = 0
    if final_selector == "global_control":
        trainer.column_anchors.clear()
    elif final_selector == "bounded_positive_pressure":
        pressure_entries = _inject_anchor_replay_pressure(trainer)
    elif final_selector != "bounded_zero_pressure_guard":
        raise ValueError(f"Unknown final selector: {final_selector}")

    started = time.perf_counter()
    updates = 0
    cycle_selection_reports: list[dict[str, Any]] = []
    for _ in range(max(0, int(consolidation_cycles))):
        updates += trainer.run_sleep_maintenance(mode="deep", cycles=1)
        cycle_selection_reports.append(
            dict(trainer._last_sleep_replay_selection_report)
        )
    latency_ms = (time.perf_counter() - started) * 1000.0

    task_a_after_consolidation = _mean_reconstruction(trainer, task_a)
    task_a_overlap_after_consolidation = mean_assembly_overlap(
        reference_assemblies,
        collect_assemblies(trainer, [pattern for _, pattern in task_a]),
    )
    gate = build_memory_consolidation_gate(
        task_a_after_a=task_a_after_a,
        task_a_after_b=task_a_after_b,
        task_a_after_consolidation=task_a_after_consolidation,
        task_a_overlap_after_consolidation=task_a_overlap_after_consolidation,
    )
    selection = dict(trainer._last_sleep_replay_selection_report)
    bounded_cycle_count = sum(
        1
        for report in cycle_selection_reports
        if report.get("candidate_scope") == "bucket_indexed_candidate_window"
        and bool(report.get("bounded_by_bucket_index"))
    )
    global_fallback_cycle_count = sum(
        1
        for report in cycle_selection_reports
        if report.get("candidate_scope") == "global_slow_path_score_scan"
    )
    return {
        "trial": final_selector,
        "seed": int(seed),
        "updates": int(updates),
        "latency_ms": float(latency_ms),
        "anchored_columns": int(anchored_columns),
        "pressure_entries": int(pressure_entries),
        "metrics": {
            "task_a_recon_after_a": task_a_after_a,
            "task_a_recon_after_b": task_a_after_b,
            "task_a_recon_after_consolidation": task_a_after_consolidation,
            "task_a_recovery_delta": (
                task_a_after_b - task_a_after_consolidation
            ),
            "task_a_overlap_after_consolidation": (
                task_a_overlap_after_consolidation
            ),
        },
        "memory_consolidation_gate": gate,
        "selection": selection,
        "cycle_selection_reports": cycle_selection_reports,
        "bounded_cycle_count": int(bounded_cycle_count),
        "global_fallback_cycle_count": int(global_fallback_cycle_count),
        "device": {
            "model_device": str(trainer.model.device),
            "memory_store": trainer.model.memory_store.device_report(),
        },
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    trials = [
        run_trial(
            seed=args.seed,
            final_selector=selector,
            n_columns=args.n_columns,
            column_latent_dim=args.column_latent_dim,
            memory_capacity=args.memory_capacity,
            task_repetitions=args.task_repetitions,
            boundary_cycles=args.boundary_cycles,
            consolidation_cycles=args.consolidation_cycles,
            replay_steps=args.replay_steps,
            candidate_pool=args.candidate_pool,
        )
        for selector in (
            "bounded_positive_pressure",
            "bounded_zero_pressure_guard",
            "global_control",
        )
    ]
    return {
        "surface": "bounded_replay_window_benchmark.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "parameters": {
            "seed": int(args.seed),
            "n_columns": int(args.n_columns),
            "column_latent_dim": int(args.column_latent_dim),
            "memory_capacity": int(args.memory_capacity),
            "task_repetitions": int(args.task_repetitions),
            "boundary_cycles": int(args.boundary_cycles),
            "consolidation_cycles": int(args.consolidation_cycles),
            "replay_steps": int(args.replay_steps),
            "candidate_pool": int(args.candidate_pool),
        },
        "trials": trials,
        "quality_claim": (
            "selection_evidence_only_until reconstruction gate passes under "
            "bounded replay pressure and long-run hot-path evidence"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded replay-window selection evidence.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--n-columns", type=int, default=16)
    parser.add_argument("--column-latent-dim", type=int, default=32)
    parser.add_argument("--memory-capacity", type=int, default=128)
    parser.add_argument("--task-repetitions", type=int, default=18)
    parser.add_argument("--boundary-cycles", type=int, default=2)
    parser.add_argument("--consolidation-cycles", type=int, default=4)
    parser.add_argument("--replay-steps", type=int, default=16)
    parser.add_argument("--candidate-pool", type=int, default=32)
    args = parser.parse_args()

    report = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(args.output, report)
    print(f"bounded_replay_window_benchmark={args.output}")
    for trial in report["trials"]:
        selection = trial["selection"]
        metrics = trial["metrics"]
        print(
            f"{trial['trial']}: updates={trial['updates']} "
            f"candidate_scope={selection.get('candidate_scope')} "
            f"bounded_cycles={trial['bounded_cycle_count']} "
            f"global_fallback_cycles={trial['global_fallback_cycle_count']} "
            f"score_count={selection.get('score_count')} "
            f"selected_count={selection.get('selected_count')} "
            f"gate_pass={trial['memory_consolidation_gate']['pass']} "
            f"recovery_delta={metrics['task_a_recovery_delta']:.8f}"
        )


if __name__ == "__main__":
    main()
