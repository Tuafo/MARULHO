from __future__ import annotations

import argparse
from datetime import datetime
from itertools import combinations, product
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch

from hecsn.config.model_config import HECSNConfig
from hecsn.config.presets import get_contextual_routing_preset, contextual_routing_preset_names
from hecsn.data.pattern_loader import interleave_tagged_blocks, labeled_pattern_stream, load_train_eval_examples
from hecsn.data.rtf_encoder import RTFEncoder
from hecsn.reporting.io import write_json_file
from hecsn.reporting.benchmark_plots import plot_contextual_routing_summary
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.training.runner_utils import set_seed
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer


def mean_reconstruction_error(trainer: HECSNTrainer, patterns: List[torch.Tensor]) -> float:
    if not patterns:
        return float("nan")
    values = [trainer.reconstruction_error(pattern) for pattern in patterns]
    return float(sum(values) / len(values))


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)
    return float(sim.item())


def probe_under_context(
    trainer: HECSNTrainer,
    prime_patterns: List[torch.Tensor],
    probe_patterns: List[torch.Tensor],
) -> tuple[torch.Tensor, List[int], List[torch.Tensor]]:
    trainer.prime_context(prime_patterns, update_weights=False)
    state = trainer.context_state()
    winners = [trainer.contextual_winner_for_pattern(pattern) for pattern in probe_patterns]
    assemblies = [trainer.contextual_assembly_for_pattern(pattern).float() for pattern in probe_patterns]
    return state, winners, assemblies


def mean_probe_distance(first: List[torch.Tensor], second: List[torch.Tensor]) -> float:
    if not first or not second or len(first) != len(second):
        return float("nan")
    distances = [1.0 - cosine_similarity(left, right) for left, right in zip(first, second)]
    return float(sum(distances) / len(distances))


def mean_signature(vectors: List[torch.Tensor]) -> torch.Tensor:
    if not vectors:
        return torch.empty(0, dtype=torch.float32)
    stacked = torch.stack([vector.float() for vector in vectors], dim=0)
    mean = stacked.mean(dim=0)
    norm = float(mean.norm(p=2).item())
    if norm <= 0.0:
        return mean
    return mean / norm


def mean_pairwise_distance(vectors: List[torch.Tensor]) -> float:
    if len(vectors) < 2:
        return float("nan")
    distances = [1.0 - cosine_similarity(left, right) for left, right in combinations(vectors, 2)]
    return float(sum(distances) / len(distances))


def mean_cross_distance(left: List[torch.Tensor], right: List[torch.Tensor]) -> float:
    if not left or not right:
        return float("nan")
    distances = [1.0 - cosine_similarity(first, second) for first, second in product(left, right)]
    return float(sum(distances) / len(distances))


def leave_one_out_family_accuracy(families: dict[str, List[torch.Tensor]]) -> float:
    if len(families) < 2:
        return float("nan")

    correct = 0
    total = 0
    family_items = list(families.items())
    for family_name, vectors in family_items:
        other_vectors = [vector for other_name, items in family_items if other_name != family_name for vector in items]
        if not other_vectors:
            continue
        for index, vector in enumerate(vectors):
            same_vectors = [item for item_index, item in enumerate(vectors) if item_index != index]
            if not same_vectors:
                continue
            same_centroid = mean_signature(same_vectors)
            other_centroid = mean_signature(other_vectors)
            own_similarity = cosine_similarity(vector, same_centroid)
            other_similarity = cosine_similarity(vector, other_centroid)
            correct += int(own_similarity > other_similarity)
            total += 1

    return float(correct / total) if total > 0 else float("nan")


def text_examples(text: str, encoder: RTFEncoder, window_size: int) -> list[tuple[str, torch.Tensor]]:
    return [(window, pattern.clone()) for window, pattern in labeled_pattern_stream(text, encoder, window_size)]


def bank_polysemy_probe(
    trainer: HECSNTrainer,
    encoder: RTFEncoder,
    window_size: int,
) -> dict[str, Any]:
    river_contexts = [
        "river water shore mud current ",
        "boats drift by the river shore ",
        "floodplain water along the river edge ",
        "wet stones and reeds by the river ",
    ]
    money_contexts = [
        "money loan account credit savings ",
        "cash deposit branch interest rate ",
        "finance office handles loan accounts ",
        "credit savings and bank deposits ",
    ]
    probe_examples = [(label, pattern) for label, pattern in text_examples("bank", encoder, window_size) if label.strip()]
    samples: list[dict[str, Any]] = []

    for family_name, contexts in (("river", river_contexts), ("money", money_contexts)):
        for context_text in contexts:
            prime_patterns = [pattern for _, pattern in text_examples(context_text, encoder, window_size)]
            if not prime_patterns:
                continue
            trainer.prime_context_with_signatures(
                prime_patterns,
                update_weights=False,
                blend_context_state=True,
                readout_mode="relu",
            )
            context_state = trainer.context_state().float()
            winners: list[int] = []
            assemblies: list[torch.Tensor] = []
            for _, probe_pattern in probe_examples:
                signature = trainer.contextual_signature_for_pattern(
                    probe_pattern,
                    blend_context_state=True,
                    readout_mode="relu",
                ).float()
                winners.append(int(torch.argmax(signature).item()))
                assemblies.append(signature)
            samples.append(
                {
                    "family": family_name,
                    "context_text": context_text,
                    "context_state": context_state,
                    "winner_sequence": tuple(winners),
                    "final_winner": int(winners[-1]),
                    "signature": mean_signature(assemblies),
                }
            )

    family_signatures = {
        family_name: [sample["signature"] for sample in samples if sample["family"] == family_name]
        for family_name in ("river", "money")
    }
    family_context_states = {
        family_name: [sample["context_state"] for sample in samples if sample["family"] == family_name]
        for family_name in ("river", "money")
    }

    river_samples = [sample for sample in samples if sample["family"] == "river"]
    money_samples = [sample for sample in samples if sample["family"] == "money"]
    winner_sequence_difference_rate = float(
        sum(int(left["winner_sequence"] != right["winner_sequence"]) for left, right in product(river_samples, money_samples))
        / max(1, len(river_samples) * len(money_samples))
    )
    final_winner_difference_rate = float(
        sum(int(left["final_winner"] != right["final_winner"]) for left, right in product(river_samples, money_samples))
        / max(1, len(river_samples) * len(money_samples))
    )

    within_signature_distance = float(
        np.nanmean([
            mean_pairwise_distance(vectors)
            for vectors in family_signatures.values()
            if len(vectors) >= 2
        ])
    ) if any(len(vectors) >= 2 for vectors in family_signatures.values()) else float("nan")
    cross_signature_distance = mean_cross_distance(
        family_signatures["river"],
        family_signatures["money"],
    )
    within_context_state_distance = float(
        np.nanmean([
            mean_pairwise_distance(vectors)
            for vectors in family_context_states.values()
            if len(vectors) >= 2
        ])
    ) if any(len(vectors) >= 2 for vectors in family_context_states.values()) else float("nan")
    cross_context_state_distance = mean_cross_distance(
        family_context_states["river"],
        family_context_states["money"],
    )
    signature_margin = float(cross_signature_distance - within_signature_distance)
    context_state_margin = float(cross_context_state_distance - within_context_state_distance)
    family_accuracy = leave_one_out_family_accuracy(family_signatures)

    return {
        "probe_word": "bank",
        "probe_labels": [label for label, _ in probe_examples],
        "sample_count": len(samples),
        "river_contexts": river_contexts,
        "money_contexts": money_contexts,
        "family_classification_accuracy": family_accuracy,
        "signature_within_distance": within_signature_distance,
        "signature_cross_distance": cross_signature_distance,
        "signature_separation_margin": signature_margin,
        "context_state_within_distance": within_context_state_distance,
        "context_state_cross_distance": cross_context_state_distance,
        "context_state_separation_margin": context_state_margin,
        "winner_sequence_difference_rate": winner_sequence_difference_rate,
        "final_winner_difference_rate": final_winner_difference_rate,
        "samples": [
            {
                "family": sample["family"],
                "context_text": sample["context_text"],
                "winner_sequence": list(sample["winner_sequence"]),
                "final_winner": sample["final_winner"],
            }
            for sample in samples
        ],
    }


def summarize_training_metrics(rows: List[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "mean_dopamine": float("nan"),
            "mean_serotonin": float("nan"),
            "mean_acetylcholine": float("nan"),
            "mean_norepinephrine": float("nan"),
            "mean_binding_strength": float("nan"),
            "mean_context_gain": float("nan"),
        }
    return {
        "mean_dopamine": float(np.mean([float(row["dopamine"]) for row in rows])),
        "mean_serotonin": float(np.mean([float(row.get("serotonin", float("nan"))) for row in rows])),
        "mean_acetylcholine": float(np.mean([float(row["acetylcholine"]) for row in rows])),
        "mean_norepinephrine": float(np.mean([float(row["norepinephrine"]) for row in rows])),
        "mean_binding_strength": float(np.mean([float(row["binding_strength"]) for row in rows])),
        "mean_context_gain": float(np.mean([float(row["context_gain_mean"]) for row in rows])),
    }


def run_contextual_routing(
    task_a_source: str,
    task_a_hf_config: Optional[str],
    task_a_text_field: str,
    task_a_train_tokens: int,
    task_b_source: str,
    task_b_hf_config: Optional[str],
    task_b_text_field: str,
    task_b_train_tokens: int,
    eval_tokens: int,
    context_block_tokens: int,
    prime_tokens: int,
    probe_tokens: int,
    output_dir: Path,
    seed: int,
    n_columns: int,
    column_latent_dim: int,
    memory_capacity: int,
    input_weight_blend: float,
    input_synapse_ltp: float,
    input_synapse_ltd: float,
    input_weight_row_target: float,
    homeostasis_beta: float,
    homeostasis_lr: float,
    slow_mean_decay: float,
    use_winner_local_drift: bool,
    drift_threshold: float,
    micro_sleep_interval_tokens: int,
    micro_sleep_replay_steps: int,
    micro_sleep_candidate_pool: int,
    micro_sleep_memory_blend: float,
    deep_sleep_interval_tokens: int,
    deep_sleep_replay_steps: int,
    deep_sleep_candidate_pool: int,
    deep_sleep_memory_blend: float,
    deep_sleep_cooldown_tokens: int,
    emergency_deep_sleep_cooldown_tokens: int,
    drift_floor_history_tokens: int,
    drift_floor_check_interval_tokens: int,
    drift_floor_window_tokens: int,
    drift_floor_trigger_min_tokens: int,
    drift_floor_rise_tolerance: float,
    prototype_momentum: float,
    context_decay: float,
    context_transition_lr: float,
    context_modulation_strength: float,
    binding_threshold: float,
    binding_association_lr: float,
    binding_association_decay: float,
    binding_gain_strength: float,
    checkpoint_out: Optional[Path],
    save_plots: bool,
) -> None:
    cfg = HECSNConfig(
        n_columns=n_columns,
        column_latent_dim=column_latent_dim,
        memory_capacity=memory_capacity,
        input_weight_blend=input_weight_blend,
        input_synapse_ltp=input_synapse_ltp,
        input_synapse_ltd=input_synapse_ltd,
        input_weight_row_target=input_weight_row_target,
        homeostasis_beta=homeostasis_beta,
        homeostasis_lr=homeostasis_lr,
        slow_mean_decay=slow_mean_decay,
        use_winner_local_drift=use_winner_local_drift,
        drift_threshold=drift_threshold,
        micro_sleep_interval_tokens=micro_sleep_interval_tokens,
        micro_sleep_replay_steps=micro_sleep_replay_steps,
        micro_sleep_candidate_pool=micro_sleep_candidate_pool,
        micro_sleep_memory_blend=micro_sleep_memory_blend,
        deep_sleep_interval_tokens=deep_sleep_interval_tokens,
        deep_sleep_replay_steps=deep_sleep_replay_steps,
        deep_sleep_candidate_pool=deep_sleep_candidate_pool,
        deep_sleep_memory_blend=deep_sleep_memory_blend,
        deep_sleep_cooldown_tokens=deep_sleep_cooldown_tokens,
        emergency_deep_sleep_cooldown_tokens=emergency_deep_sleep_cooldown_tokens,
        drift_floor_history_tokens=drift_floor_history_tokens,
        drift_floor_check_interval_tokens=drift_floor_check_interval_tokens,
        drift_floor_window_tokens=drift_floor_window_tokens,
        drift_floor_trigger_min_tokens=drift_floor_trigger_min_tokens,
        drift_floor_rise_tolerance=drift_floor_rise_tolerance,
        prototype_momentum=prototype_momentum,
        enable_context_layer=True,
        context_decay=context_decay,
        context_transition_lr=context_transition_lr,
        context_modulation_strength=context_modulation_strength,
        enable_binding_layer=True,
        binding_threshold=binding_threshold,
        binding_association_lr=binding_association_lr,
        binding_association_decay=binding_association_decay,
        binding_gain_strength=binding_gain_strength,
    )
    encoder = RTFEncoder.from_config(cfg)

    task_a_train_examples, task_a_eval_examples = load_train_eval_examples(
        source=task_a_source,
        source_type="hf",
        hf_config=task_a_hf_config,
        text_field=task_a_text_field,
        encoder=encoder,
        window_size=cfg.window_size,
        train_tokens=task_a_train_tokens,
        eval_tokens=eval_tokens,
    )
    task_b_train_examples, task_b_eval_examples = load_train_eval_examples(
        source=task_b_source,
        source_type="hf",
        hf_config=task_b_hf_config,
        text_field=task_b_text_field,
        encoder=encoder,
        window_size=cfg.window_size,
        train_tokens=task_b_train_tokens,
        eval_tokens=eval_tokens,
    )
    task_a_train = [pattern for _, pattern in task_a_train_examples]
    task_a_eval = [pattern for _, pattern in task_a_eval_examples]
    task_b_train = [pattern for _, pattern in task_b_train_examples]
    task_b_eval = [pattern for _, pattern in task_b_eval_examples]
    if not task_a_train or not task_a_eval:
        raise ValueError("Task A did not produce enough HuggingFace patterns")
    if not task_b_train or not task_b_eval:
        raise ValueError("Task B did not produce enough HuggingFace patterns")

    set_seed(seed)
    model = HECSNModelLite(cfg)
    trainer = HECSNTrainer(model, cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[dict[str, Any]] = []
    for block_name, example in interleave_tagged_blocks(
        task_a_train_examples,
        task_b_train_examples,
        context_block_tokens,
        first_label="task_a",
        second_label="task_b",
    ):
        raw_window, pattern = example
        row = trainer.train_step(pattern, raw_window=raw_window)
        row["block_name"] = block_name
        metrics_rows.append(row)

    task_a_recon = mean_reconstruction_error(trainer, task_a_eval)
    task_b_recon = mean_reconstruction_error(trainer, task_b_eval)

    prime_len = min(max(1, int(prime_tokens)), len(task_a_eval), len(task_b_eval))
    probe_bank = task_a_eval[prime_len : prime_len + probe_tokens] + task_b_eval[prime_len : prime_len + probe_tokens]
    if not probe_bank:
        raise ValueError("Probe bank is empty; increase eval_tokens or reduce prime/probe budget")

    state_a, winners_a, assemblies_a = probe_under_context(trainer, task_a_eval[:prime_len], probe_bank)
    state_b, winners_b, assemblies_b = probe_under_context(trainer, task_b_eval[:prime_len], probe_bank)

    switch_rate = float(sum(int(left != right) for left, right in zip(winners_a, winners_b)) / max(1, len(probe_bank)))
    context_similarity = cosine_similarity(state_a, state_b)
    context_separation = float(1.0 - context_similarity)
    probe_distance = mean_probe_distance(assemblies_a, assemblies_b)
    polysemy_probe = bank_polysemy_probe(trainer, encoder, cfg.window_size)
    polysemy_accuracy = float(polysemy_probe["family_classification_accuracy"])
    polysemy_margin = float(polysemy_probe["signature_separation_margin"])

    contextual_routing_gate = {
        "pass": bool(
            context_separation >= 0.10
            and switch_rate >= 0.05
            and polysemy_accuracy >= 0.60
            and polysemy_margin > 0.0
        ),
        "context_state_separation_gte_0_10": bool(context_separation >= 0.10),
        "probe_winner_switch_rate_gte_0_05": bool(switch_rate >= 0.05),
        "bank_polysemy_accuracy_gte_0_60": bool(polysemy_accuracy >= 0.60),
        "bank_polysemy_signature_margin_gt_0": bool(polysemy_margin > 0.0),
        "thresholds": {
            "context_state_separation_min": 0.10,
            "probe_winner_switch_rate_min": 0.05,
            "bank_polysemy_accuracy_min": 0.60,
            "bank_polysemy_signature_margin_min": 0.0,
        },
    }

    summary = {
        "protocol": "contextual_routing_hf",
        "data_setup": {
            "task_a": {
                "source": task_a_source,
                "source_type": "hf",
                "hf_config": task_a_hf_config,
                "text_field": task_a_text_field,
                "train_tokens": task_a_train_tokens,
                "eval_tokens": len(task_a_eval),
            },
            "task_b": {
                "source": task_b_source,
                "source_type": "hf",
                "hf_config": task_b_hf_config,
                "text_field": task_b_text_field,
                "train_tokens": task_b_train_tokens,
                "eval_tokens": len(task_b_eval),
            },
            "context_block_tokens": context_block_tokens,
            "prime_tokens": prime_len,
            "probe_tokens": len(probe_bank),
            "n_columns": cfg.n_columns,
            "column_latent_dim": cfg.column_latent_dim,
        },
        "runtime_scope": model.runtime_scope_report(),
        "training_diagnostics": summarize_training_metrics(metrics_rows),
        "contextual_routing_metrics": {
            "task_a_recon_error": task_a_recon,
            "task_b_recon_error": task_b_recon,
            "context_state_similarity": context_similarity,
            "context_state_separation": context_separation,
            "probe_winner_switch_rate": switch_rate,
            "probe_mean_assembly_distance": probe_distance,
            "bank_polysemy_accuracy": polysemy_accuracy,
            "bank_polysemy_signature_margin": polysemy_margin,
            "bank_polysemy_winner_sequence_difference_rate": float(polysemy_probe["winner_sequence_difference_rate"]),
            "bank_polysemy_final_winner_difference_rate": float(polysemy_probe["final_winner_difference_rate"]),
        },
        "polysemy_probe": polysemy_probe,
        "contextual_routing_gate": contextual_routing_gate,
    }

    checkpoint_path: Optional[Path] = None
    if checkpoint_out is not None:
        checkpoint_path = save_trainer_checkpoint(
            checkpoint_out,
            trainer,
            metadata={
                "protocol": "contextual_routing_hf",
                "benchmark": "contextual_routing",
                "task_a_source": task_a_source,
                "task_a_hf_config": task_a_hf_config,
                "task_a_text_field": task_a_text_field,
                "task_b_source": task_b_source,
                "task_b_hf_config": task_b_hf_config,
                "task_b_text_field": task_b_text_field,
                "task_a_train_tokens": len(task_a_train),
                "task_b_train_tokens": len(task_b_train),
                "eval_tokens": len(task_a_eval),
                "context_block_tokens": context_block_tokens,
            },
        )
        summary["checkpoint_path"] = str(checkpoint_path)

    write_json_file(output_dir / "summary.json", summary)
    if save_plots:
        plot_contextual_routing_summary(output_dir, summary, metrics_rows)

    print("Contextual-routing summary")
    print(f"task_a_recon_error={task_a_recon:.6f}")
    print(f"task_b_recon_error={task_b_recon:.6f}")
    print(f"context_state_separation={context_separation:.6f}")
    print(f"probe_winner_switch_rate={switch_rate:.6f}")
    print(f"probe_mean_assembly_distance={probe_distance:.6f}")
    print(f"bank_polysemy_accuracy={polysemy_accuracy:.6f}")
    print(f"bank_polysemy_signature_margin={polysemy_margin:.6f}")
    print(f"contextual_routing_gate_pass={contextual_routing_gate['pass']}")
    print(f"summary_json={output_dir / 'summary.json'}")
    if checkpoint_path is not None:
        print(f"checkpoint_path={checkpoint_path}")
    if save_plots:
        print(f"contextual_routing_plot={output_dir / 'contextual_routing_diagnostics.png'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the HECSN contextual-routing benchmark")
    parser.add_argument("--preset", choices=contextual_routing_preset_names(), default=None)
    parser.add_argument("--task-a-source", default="ag_news")
    parser.add_argument("--task-a-hf-config", default=None)
    parser.add_argument("--task-a-text-field", default="text")
    parser.add_argument("--task-a-train-tokens", type=int, default=4000)
    parser.add_argument("--task-b-source", default="wikitext")
    parser.add_argument("--task-b-hf-config", default="wikitext-103-raw-v1")
    parser.add_argument("--task-b-text-field", default="text")
    parser.add_argument("--task-b-train-tokens", type=int, default=4000)
    parser.add_argument("--eval-tokens", type=int, default=1000)
    parser.add_argument("--context-block-tokens", type=int, default=250)
    parser.add_argument("--prime-tokens", type=int, default=128)
    parser.add_argument("--probe-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-columns", type=int, default=100)
    parser.add_argument("--column-latent-dim", type=int, default=256)
    parser.add_argument("--memory-capacity", type=int, default=1000)
    parser.add_argument("--input-weight-blend", type=float, default=0.02)
    parser.add_argument("--input-synapse-ltp", type=float, default=0.02)
    parser.add_argument("--input-synapse-ltd", type=float, default=0.01)
    parser.add_argument("--input-weight-row-target", type=float, default=1.0)
    parser.add_argument("--homeostasis-beta", type=float, default=0.01)
    parser.add_argument("--homeostasis-lr", type=float, default=0.2)
    parser.add_argument("--slow-mean-decay", type=float, default=0.9999)
    parser.add_argument("--use-winner-local-drift", action="store_true")
    parser.add_argument("--drift-threshold", type=float, default=0.02)
    parser.add_argument("--micro-sleep-interval-tokens", type=int, default=200)
    parser.add_argument("--micro-sleep-replay-steps", type=int, default=10)
    parser.add_argument("--micro-sleep-candidate-pool", type=int, default=5)
    parser.add_argument("--micro-sleep-memory-blend", type=float, default=0.05)
    parser.add_argument("--deep-sleep-interval-tokens", type=int, default=2500)
    parser.add_argument("--deep-sleep-replay-steps", type=int, default=200)
    parser.add_argument("--deep-sleep-candidate-pool", type=int, default=100)
    parser.add_argument("--deep-sleep-memory-blend", type=float, default=0.20)
    parser.add_argument("--deep-sleep-cooldown-tokens", type=int, default=1000)
    parser.add_argument("--emergency-deep-sleep-cooldown-tokens", type=int, default=1000)
    parser.add_argument("--drift-floor-history-tokens", type=int, default=1000)
    parser.add_argument("--drift-floor-check-interval-tokens", type=int, default=200)
    parser.add_argument("--drift-floor-window-tokens", type=int, default=10000)
    parser.add_argument("--drift-floor-trigger-min-tokens", type=int, default=1000)
    parser.add_argument("--drift-floor-rise-tolerance", type=float, default=0.0)
    parser.add_argument("--prototype-momentum", type=float, default=0.85)
    parser.add_argument("--context-decay", type=float, default=0.92)
    parser.add_argument("--context-transition-lr", type=float, default=0.05)
    parser.add_argument("--context-modulation-strength", type=float, default=0.60)
    parser.add_argument("--binding-threshold", type=float, default=0.02)
    parser.add_argument("--binding-association-lr", type=float, default=0.20)
    parser.add_argument("--binding-association-decay", type=float, default=0.995)
    parser.add_argument("--binding-gain-strength", type=float, default=0.80)
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    preset = get_contextual_routing_preset(args.preset)

    merged = vars(args).copy()
    for key, value in preset.items():
        if merged.get(key) == parser.get_default(key):
            merged[key] = value

    output_dir = merged["output_dir"]
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = merged["preset"] if merged["preset"] is not None else "contextual_routing_run"
        output_dir = Path("reports") / f"{suffix}_{stamp}"

    run_contextual_routing(
        task_a_source=merged["task_a_source"],
        task_a_hf_config=merged["task_a_hf_config"],
        task_a_text_field=merged["task_a_text_field"],
        task_a_train_tokens=int(merged["task_a_train_tokens"]),
        task_b_source=merged["task_b_source"],
        task_b_hf_config=merged["task_b_hf_config"],
        task_b_text_field=merged["task_b_text_field"],
        task_b_train_tokens=int(merged["task_b_train_tokens"]),
        eval_tokens=int(merged["eval_tokens"]),
        context_block_tokens=int(merged["context_block_tokens"]),
        prime_tokens=int(merged["prime_tokens"]),
        probe_tokens=int(merged["probe_tokens"]),
        output_dir=output_dir,
        seed=int(merged["seed"]),
        n_columns=int(merged["n_columns"]),
        column_latent_dim=int(merged["column_latent_dim"]),
        memory_capacity=int(merged["memory_capacity"]),
        input_weight_blend=float(merged["input_weight_blend"]),
        input_synapse_ltp=float(merged["input_synapse_ltp"]),
        input_synapse_ltd=float(merged["input_synapse_ltd"]),
        input_weight_row_target=float(merged["input_weight_row_target"]),
        homeostasis_beta=float(merged["homeostasis_beta"]),
        homeostasis_lr=float(merged["homeostasis_lr"]),
        slow_mean_decay=float(merged["slow_mean_decay"]),
        use_winner_local_drift=bool(merged["use_winner_local_drift"]),
        drift_threshold=float(merged["drift_threshold"]),
        micro_sleep_interval_tokens=int(merged["micro_sleep_interval_tokens"]),
        micro_sleep_replay_steps=int(merged["micro_sleep_replay_steps"]),
        micro_sleep_candidate_pool=int(merged["micro_sleep_candidate_pool"]),
        micro_sleep_memory_blend=float(merged["micro_sleep_memory_blend"]),
        deep_sleep_interval_tokens=int(merged["deep_sleep_interval_tokens"]),
        deep_sleep_replay_steps=int(merged["deep_sleep_replay_steps"]),
        deep_sleep_candidate_pool=int(merged["deep_sleep_candidate_pool"]),
        deep_sleep_memory_blend=float(merged["deep_sleep_memory_blend"]),
        deep_sleep_cooldown_tokens=int(merged["deep_sleep_cooldown_tokens"]),
        emergency_deep_sleep_cooldown_tokens=int(merged["emergency_deep_sleep_cooldown_tokens"]),
        drift_floor_history_tokens=int(merged["drift_floor_history_tokens"]),
        drift_floor_check_interval_tokens=int(merged["drift_floor_check_interval_tokens"]),
        drift_floor_window_tokens=int(merged["drift_floor_window_tokens"]),
        drift_floor_trigger_min_tokens=int(merged["drift_floor_trigger_min_tokens"]),
        drift_floor_rise_tolerance=float(merged["drift_floor_rise_tolerance"]),
        prototype_momentum=float(merged["prototype_momentum"]),
        context_decay=float(merged["context_decay"]),
        context_transition_lr=float(merged["context_transition_lr"]),
        context_modulation_strength=float(merged["context_modulation_strength"]),
        binding_threshold=float(merged["binding_threshold"]),
        binding_association_lr=float(merged["binding_association_lr"]),
        binding_association_decay=float(merged["binding_association_decay"]),
        binding_gain_strength=float(merged["binding_gain_strength"]),
        checkpoint_out=merged["checkpoint_out"],
        save_plots=not bool(merged["no_plots"]),
    )


if __name__ == "__main__":
    main()
