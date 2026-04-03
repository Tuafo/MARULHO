from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import time
from typing import Any, List, Optional

import numpy as np
import torch

from hecsn.config.model_config import HECSNConfig
from hecsn.config.presets import get_hierarchical_scale_preset, hierarchical_scale_preset_names
from hecsn.data.corpus_loader import SourceType
from hecsn.data.pattern_loader import load_train_eval_examples
from hecsn.data.rtf_encoder import RTFEncoder
from hecsn.reporting.io import write_json_file
from hecsn.reporting.benchmark_plots import plot_hierarchical_scale_summary
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.training.runner_utils import set_seed
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer


def mean_reconstruction_error(trainer: HECSNTrainer, patterns: List[torch.Tensor]) -> float:
    if not patterns:
        return float("nan")
    values = [trainer.reconstruction_error(pattern) for pattern in patterns]
    return float(sum(values) / len(values))


def entropy_bits(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = np.asarray([count / total for count in counts.values() if count > 0], dtype=np.float64)
    return float(-(probs * np.log2(probs)).sum())


def shard_id_for_column(column_id: int, n_shards: int) -> int:
    return int(column_id) % max(1, int(n_shards))


def memory_budget_estimate(cfg: HECSNConfig) -> dict[str, Any]:
    estimated_neurons = int(cfg.n_columns * cfg.neurons_per_column_assumption)
    estimated_synapses = int(estimated_neurons * estimated_neurons * 0.15)
    weights_bytes = estimated_synapses * 2
    sparse_eligibility_bytes = int(estimated_synapses * 0.05) * 2
    stc_state_bytes = estimated_synapses * 4
    routing_index_bytes = int(cfg.n_columns * cfg.column_latent_dim * 4 * 2)
    prototype_bytes = int(cfg.n_columns * cfg.column_latent_dim * 4)
    memory_store_bytes = int(cfg.memory_capacity * cfg.column_latent_dim * 4)
    total_bytes = (
        weights_bytes
        + sparse_eligibility_bytes
        + stc_state_bytes
        + routing_index_bytes
        + prototype_bytes
        + memory_store_bytes
    )
    bytes_per_gb = float(1024 ** 3)
    return {
        "estimated_neurons": estimated_neurons,
        "estimated_synapses": estimated_synapses,
        "weights_gb": float(weights_bytes / bytes_per_gb),
        "sparse_eligibility_gb": float(sparse_eligibility_bytes / bytes_per_gb),
        "stc_state_gb": float(stc_state_bytes / bytes_per_gb),
        "routing_index_gb": float(routing_index_bytes / bytes_per_gb),
        "prototype_gb": float(prototype_bytes / bytes_per_gb),
        "memory_store_gb": float(memory_store_bytes / bytes_per_gb),
        "estimated_total_gpu_gb": float(total_bytes / bytes_per_gb),
    }


def exact_topk_columns(trainer: HECSNTrainer, routing_key: torch.Tensor, k: int) -> List[int]:
    sims = torch.mv(trainer.model.competitive.prototypes, routing_key.to(trainer.model.device))
    topk = min(int(k), int(sims.numel()))
    if topk <= 0:
        return []
    indices = torch.topk(sims, k=topk).indices.detach().cpu().tolist()
    return [int(idx) for idx in indices]


def evaluate_routing(
    trainer: HECSNTrainer,
    patterns: List[torch.Tensor],
    k: int,
    routing_shards: int,
    latency_queries: int,
) -> dict[str, Any]:
    recalls: List[float] = []
    top1_hits = 0
    latencies_ms: List[float] = []
    primary_shard_counts: Counter[int] = Counter()
    winner_shard_counts: Counter[int] = Counter()

    for query_idx, pattern in enumerate(patterns):
        routing_key = trainer.routing_key_for_pattern(pattern).to(trainer.model.device)
        exact_ids = exact_topk_columns(trainer, routing_key, k)

        if query_idx < max(0, int(latency_queries)):
            started = time.perf_counter()
            candidate_ids, _ = trainer.model.hnsw_index.search(routing_key.unsqueeze(0), k=k)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(float(elapsed_ms))
        else:
            candidate_ids, _ = trainer.model.hnsw_index.search(routing_key.unsqueeze(0), k=k)

        ann_ids = [int(candidate) for candidate in candidate_ids[0]] if candidate_ids else []
        if exact_ids:
            recalls.append(float(len(set(ann_ids) & set(exact_ids)) / len(exact_ids)))
            top1_hits += int(exact_ids[0] in ann_ids)
            winner_shard_counts[shard_id_for_column(exact_ids[0], routing_shards)] += 1
        if ann_ids:
            primary_shard_counts[shard_id_for_column(ann_ids[0], routing_shards)] += 1

    return {
        "recall_at_k": float(np.mean(recalls)) if recalls else float("nan"),
        "top1_recall": float(top1_hits / max(1, len(patterns))),
        "mean_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else float("nan"),
        "p95_latency_ms": float(np.percentile(np.asarray(latencies_ms, dtype=np.float32), 95.0)) if latencies_ms else float("nan"),
        "latency_samples": int(len(latencies_ms)),
        "primary_shard_counts": dict(sorted(primary_shard_counts.items())),
        "winner_shard_counts": dict(sorted(winner_shard_counts.items())),
    }


def evaluate_index_integrity(trainer: HECSNTrainer, k: int) -> dict[str, float]:
    unreachable = 0
    total = int(trainer.config.n_columns)
    prototypes = trainer.model.competitive.prototypes.detach()
    for column_id in range(total):
        candidate_ids, _ = trainer.model.hnsw_index.search(prototypes[column_id : column_id + 1], k=max(1, int(k)))
        row = candidate_ids[0] if candidate_ids else []
        unreachable += int(column_id not in row)
    unreachable_fraction = float(unreachable / max(1, total))
    return {
        "unreachable_columns": float(unreachable),
        "unreachable_fraction": unreachable_fraction,
        "self_recall": float(1.0 - unreachable_fraction),
    }


def summarize_training(
    cfg: HECSNConfig,
    metrics_samples: List[dict[str, Any]],
    winner_counts: Counter[int],
    train_seconds: float,
    eval_recon: float,
) -> dict[str, Any]:
    last_recon = float(metrics_samples[-1]["recon_error"]) if metrics_samples else float("nan")
    mean_recon = float(np.mean([float(row["recon_error"]) for row in metrics_samples])) if metrics_samples else float("nan")
    mean_drift = float(np.mean([float(row["drift"]) for row in metrics_samples])) if metrics_samples else float("nan")
    sleep_events = int(metrics_samples[-1]["sleep_events_total"]) if metrics_samples else 0
    unique_winners = int(len(winner_counts))
    winner_coverage = float(unique_winners / max(1, cfg.n_columns))
    total_tokens = int(sum(winner_counts.values()))
    tokens_per_sec = float(total_tokens / max(1e-8, train_seconds))
    return {
        "throughput_chars_per_sec": float(tokens_per_sec * cfg.window_size),
        "train_seconds": float(train_seconds),
        "tokens_per_sec": tokens_per_sec,
        "winner_entropy_bits": entropy_bits(winner_counts),
        "unique_winners": unique_winners,
        "winner_coverage": winner_coverage,
        "sampled_mean_recon_error": mean_recon,
        "sampled_last_recon_error": last_recon,
        "sampled_mean_drift": mean_drift,
        "sleep_events_total": sleep_events,
        "eval_recon_error": float(eval_recon),
    }


def build_hierarchical_scale_gate(
    cfg: HECSNConfig,
    routing_metrics: dict[str, Any],
    integrity_metrics: dict[str, float],
    memory_budget: dict[str, Any],
) -> dict[str, Any]:
    estimated_neurons = int(memory_budget["estimated_neurons"])
    recall_ok = bool(float(routing_metrics["recall_at_k"]) >= 0.95)
    top1_ok = bool(float(routing_metrics["top1_recall"]) >= 0.90)
    latency_ok = bool(float(routing_metrics["mean_latency_ms"]) <= 5.0)
    integrity_ok = bool(float(integrity_metrics["unreachable_fraction"]) <= 0.01)
    scale_ok = bool(estimated_neurons >= 100000)
    sharding_ok = bool(cfg.routing_shards >= 2)
    overall = bool(scale_ok and sharding_ok and recall_ok and top1_ok and latency_ok and integrity_ok)
    return {
        "pass": overall,
        "estimated_neurons_gte_100k": scale_ok,
        "routing_shards_gte_2": sharding_ok,
        "routing_recall_at_k_gte_0_95": recall_ok,
        "routing_top1_recall_gte_0_90": top1_ok,
        "mean_latency_ms_lte_5_0": latency_ok,
        "unreachable_fraction_lte_0_01": integrity_ok,
        "thresholds": {
            "estimated_neurons_min": 100000,
            "routing_recall_at_k_min": 0.95,
            "routing_top1_recall_min": 0.90,
            "mean_latency_ms_max": 5.0,
            "unreachable_fraction_max": 0.01,
            "routing_shards_min": 2,
        },
    }


def run_hierarchical_scale(
    source: str,
    source_type: SourceType,
    hf_config: Optional[str],
    text_field: str,
    train_tokens: int,
    eval_tokens: int,
    routing_eval_queries: int,
    latency_eval_queries: int,
    output_dir: Path,
    seed: int,
    n_columns: int,
    column_latent_dim: int,
    k_routing: int,
    index_rebuild_threshold: int,
    routing_shards: int,
    shard_candidate_factor: int,
    neurons_per_column_assumption: int,
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
    checkpoint_out: Optional[Path],
    save_plots: bool,
) -> None:
    cfg = HECSNConfig(
        n_columns=n_columns,
        column_latent_dim=column_latent_dim,
        k_routing=k_routing,
        index_rebuild_threshold=index_rebuild_threshold,
        routing_shards=routing_shards,
        shard_candidate_factor=shard_candidate_factor,
        neurons_per_column_assumption=neurons_per_column_assumption,
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
    )
    encoder = RTFEncoder.from_config(cfg)

    train_examples, eval_examples = load_train_eval_examples(
        source=source,
        source_type=source_type,
        hf_config=hf_config,
        text_field=text_field,
        encoder=encoder,
        window_size=cfg.window_size,
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
    )
    if not train_examples or not eval_examples:
        raise ValueError(f"Hierarchical-scale source did not produce enough patterns (source_type={source_type!r})")
    train_patterns = [pattern for _, pattern in train_examples]
    eval_patterns = [pattern for _, pattern in eval_examples]

    set_seed(seed)
    model = HECSNModelLite(cfg)
    trainer = HECSNTrainer(model, cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_samples: List[dict[str, Any]] = []
    winner_counts: Counter[int] = Counter()
    train_started = time.perf_counter()
    for idx, (raw_window, pattern) in enumerate(train_examples, start=1):
        row = trainer.train_step(pattern, raw_window=raw_window)
        winner_counts[int(row["winner"])] += 1
        if idx == 1 or idx % 500 == 0 or idx == len(train_patterns):
            metrics_samples.append(
                {
                    "token": int(row["token"]),
                    "recon_error": float(row["recon_error"]),
                    "drift": float(row["drift"]),
                    "sleep_events_total": int(row["sleep_events_total"]),
                }
            )
    train_seconds = time.perf_counter() - train_started

    trainer.model.hnsw_index.rebuild()

    eval_recon = mean_reconstruction_error(trainer, eval_patterns)
    routing_probe_bank = eval_patterns[: max(1, min(len(eval_patterns), int(routing_eval_queries)))]
    latency_probe_bank = routing_probe_bank[: max(1, min(len(routing_probe_bank), int(latency_eval_queries)))]
    routing_metrics = evaluate_routing(
        trainer=trainer,
        patterns=routing_probe_bank,
        k=cfg.k_routing,
        routing_shards=cfg.routing_shards,
        latency_queries=len(latency_probe_bank),
    )
    integrity_metrics = evaluate_index_integrity(trainer, k=cfg.k_routing)
    index_stats = model.hnsw_index.stats()
    memory_budget = memory_budget_estimate(cfg)
    training_summary = summarize_training(
        cfg=cfg,
        metrics_samples=metrics_samples,
        winner_counts=winner_counts,
        train_seconds=train_seconds,
        eval_recon=eval_recon,
    )

    shard_count = int(index_stats.get("n_shards", 1)) if isinstance(index_stats, dict) else 1
    shard_sizes = list(index_stats.get("per_shard_unique_vectors", [int(index_stats.get("unique_vectors", cfg.n_columns))]))
    primary_counts_map = routing_metrics["primary_shard_counts"]
    winner_counts_map = routing_metrics["winner_shard_counts"]
    primary_query_shard_counts = [int(primary_counts_map.get(shard_id, 0)) for shard_id in range(shard_count)]
    winner_shard_counts = [int(winner_counts_map.get(shard_id, 0)) for shard_id in range(shard_count)]
    winner_shard_coverage = float(sum(int(count > 0) for count in winner_shard_counts) / max(1, shard_count))

    gate = build_hierarchical_scale_gate(cfg, routing_metrics, integrity_metrics, memory_budget)

    summary = {
        "protocol": "hierarchical_scale_hf",
        "data_setup": {
            "source": source,
            "source_type": "hf",
            "hf_config": hf_config,
            "text_field": text_field,
            "train_tokens": len(train_patterns),
            "eval_tokens": len(eval_patterns),
            "routing_eval_queries": len(routing_probe_bank),
            "latency_eval_queries": len(latency_probe_bank),
        },
        "runtime_scope": model.runtime_scope_report(),
        "training_diagnostics": training_summary,
        "routing_metrics": {
            "recall_at_k": float(routing_metrics["recall_at_k"]),
            "top1_recall": float(routing_metrics["top1_recall"]),
            "mean_latency_ms": float(routing_metrics["mean_latency_ms"]),
            "p95_latency_ms": float(routing_metrics["p95_latency_ms"]),
            "latency_samples": int(routing_metrics["latency_samples"]),
            "candidate_fraction": float(cfg.k_routing / max(1, cfg.n_columns)),
            "dynamic_insertions": int(cfg.n_columns + len(train_patterns)),
            "index_unique_vectors": int(index_stats.get("unique_vectors", cfg.n_columns)),
            "index_raw_entries": int(index_stats.get("raw_entries", cfg.n_columns)),
            "rebuild_count": int(index_stats.get("rebuild_count", 0)),
        },
        "index_integrity": integrity_metrics,
        "sharding": {
            "routing_shards": int(cfg.routing_shards),
            "shard_candidate_factor": int(cfg.shard_candidate_factor),
            "index_shard_sizes": shard_sizes,
            "index_shard_balance_ratio": float(index_stats.get("shard_balance_ratio", 1.0)),
            "primary_query_shard_counts": primary_query_shard_counts,
            "winner_shard_counts": winner_shard_counts,
            "winner_shard_coverage": winner_shard_coverage,
        },
        "memory_budget_estimate": memory_budget,
        "hierarchical_scale_gate": gate,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    checkpoint_path: Optional[Path] = None
    if checkpoint_out is not None:
        checkpoint_path = save_trainer_checkpoint(
            checkpoint_out,
            trainer,
            metadata={
                "protocol": "hierarchical_scale_hf",
                "benchmark": "hierarchical_scale",
                "source": source,
                "hf_config": hf_config,
                "text_field": text_field,
                "train_tokens": len(train_patterns),
                "eval_tokens": len(eval_patterns),
            },
        )
        summary["checkpoint_path"] = str(checkpoint_path)

    write_json_file(output_dir / "summary.json", summary)
    if save_plots:
        plot_hierarchical_scale_summary(output_dir, summary, metrics_samples)

    print("Hierarchical-scale summary")
    print(f"eval_recon_error={eval_recon:.6f}")
    print(f"routing_recall_at_k={float(routing_metrics['recall_at_k']):.6f}")
    print(f"routing_top1_recall={float(routing_metrics['top1_recall']):.6f}")
    print(f"routing_mean_latency_ms={float(routing_metrics['mean_latency_ms']):.6f}")
    print(f"index_unreachable_fraction={float(integrity_metrics['unreachable_fraction']):.6f}")
    print(f"estimated_neurons={int(memory_budget['estimated_neurons'])}")
    print(f"hierarchical_scale_gate_pass={gate['pass']}")
    print(f"summary_json={output_dir / 'summary.json'}")
    if checkpoint_path is not None:
        print(f"checkpoint_path={checkpoint_path}")
    if save_plots:
        print(f"hierarchical_scale_plot={output_dir / 'hierarchical_scale_diagnostics.png'}")


def build_arg_parser() -> argparse.ArgumentParser:
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument("--preset", choices=hierarchical_scale_preset_names(), default=None)
    preset_args, _ = preset_parser.parse_known_args()
    preset_defaults = get_hierarchical_scale_preset(preset_args.preset)

    parser = argparse.ArgumentParser(description="Run the HECSN hierarchical-scale benchmark")
    parser.add_argument("--preset", choices=hierarchical_scale_preset_names(), default=preset_args.preset)
    parser.add_argument("--source", type=str, default=preset_defaults.get("source", "wikitext"))
    parser.add_argument("--source-type", type=str, default=preset_defaults.get("source_type", "hf"), choices=["hf", "file", "text"], help="Source type: 'hf' for HuggingFace, 'file' for local file, 'text' for raw string")
    parser.add_argument("--hf-config", type=str, default=preset_defaults.get("hf_config", "wikitext-103-raw-v1"))
    parser.add_argument("--text-field", type=str, default=preset_defaults.get("text_field", "text"))
    parser.add_argument("--train-tokens", type=int, default=preset_defaults.get("train_tokens", 12000))
    parser.add_argument("--eval-tokens", type=int, default=preset_defaults.get("eval_tokens", 1500))
    parser.add_argument("--routing-eval-queries", type=int, default=preset_defaults.get("routing_eval_queries", 256))
    parser.add_argument("--latency-eval-queries", type=int, default=preset_defaults.get("latency_eval_queries", 128))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=preset_defaults.get("seed", 7))
    parser.add_argument("--n-columns", type=int, default=preset_defaults.get("n_columns", 256))
    parser.add_argument("--column-latent-dim", type=int, default=preset_defaults.get("column_latent_dim", 256))
    parser.add_argument("--k-routing", type=int, default=preset_defaults.get("k_routing", 12))
    parser.add_argument("--index-rebuild-threshold", type=int, default=preset_defaults.get("index_rebuild_threshold", 128))
    parser.add_argument("--routing-shards", type=int, default=preset_defaults.get("routing_shards", 4))
    parser.add_argument("--shard-candidate-factor", type=int, default=preset_defaults.get("shard_candidate_factor", 2))
    parser.add_argument("--neurons-per-column-assumption", type=int, default=preset_defaults.get("neurons_per_column_assumption", 100))
    parser.add_argument("--memory-capacity", type=int, default=preset_defaults.get("memory_capacity", 1000))
    parser.add_argument("--input-weight-blend", type=float, default=preset_defaults.get("input_weight_blend", 0.02))
    parser.add_argument("--input-synapse-ltp", type=float, default=preset_defaults.get("input_synapse_ltp", 0.02))
    parser.add_argument("--input-synapse-ltd", type=float, default=preset_defaults.get("input_synapse_ltd", 0.01))
    parser.add_argument("--input-weight-row-target", type=float, default=preset_defaults.get("input_weight_row_target", 1.0))
    parser.add_argument("--homeostasis-beta", type=float, default=preset_defaults.get("homeostasis_beta", 0.01))
    parser.add_argument("--homeostasis-lr", type=float, default=preset_defaults.get("homeostasis_lr", 0.2))
    parser.add_argument("--slow-mean-decay", type=float, default=preset_defaults.get("slow_mean_decay", 0.9999))
    parser.add_argument("--use-winner-local-drift", action="store_true", default=bool(preset_defaults.get("use_winner_local_drift", True)))
    parser.add_argument("--no-winner-local-drift", action="store_true")
    parser.add_argument("--drift-threshold", type=float, default=preset_defaults.get("drift_threshold", 0.02))
    parser.add_argument("--micro-sleep-interval-tokens", type=int, default=preset_defaults.get("micro_sleep_interval_tokens", 200))
    parser.add_argument("--micro-sleep-replay-steps", type=int, default=preset_defaults.get("micro_sleep_replay_steps", 10))
    parser.add_argument("--micro-sleep-candidate-pool", type=int, default=preset_defaults.get("micro_sleep_candidate_pool", 5))
    parser.add_argument("--micro-sleep-memory-blend", type=float, default=preset_defaults.get("micro_sleep_memory_blend", 0.05))
    parser.add_argument("--deep-sleep-interval-tokens", type=int, default=preset_defaults.get("deep_sleep_interval_tokens", 2500))
    parser.add_argument("--deep-sleep-replay-steps", type=int, default=preset_defaults.get("deep_sleep_replay_steps", 200))
    parser.add_argument("--deep-sleep-candidate-pool", type=int, default=preset_defaults.get("deep_sleep_candidate_pool", 100))
    parser.add_argument("--deep-sleep-memory-blend", type=float, default=preset_defaults.get("deep_sleep_memory_blend", 0.20))
    parser.add_argument("--deep-sleep-cooldown-tokens", type=int, default=preset_defaults.get("deep_sleep_cooldown_tokens", 1000))
    parser.add_argument("--emergency-deep-sleep-cooldown-tokens", type=int, default=preset_defaults.get("emergency_deep_sleep_cooldown_tokens", 1000))
    parser.add_argument("--drift-floor-history-tokens", type=int, default=preset_defaults.get("drift_floor_history_tokens", 1000))
    parser.add_argument("--drift-floor-check-interval-tokens", type=int, default=preset_defaults.get("drift_floor_check_interval_tokens", 200))
    parser.add_argument("--drift-floor-window-tokens", type=int, default=preset_defaults.get("drift_floor_window_tokens", 10000))
    parser.add_argument("--drift-floor-trigger-min-tokens", type=int, default=preset_defaults.get("drift_floor_trigger_min_tokens", 1000))
    parser.add_argument("--drift-floor-rise-tolerance", type=float, default=preset_defaults.get("drift_floor_rise_tolerance", 0.0))
    parser.add_argument("--prototype-momentum", type=float, default=preset_defaults.get("prototype_momentum", 0.85))
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.use_winner_local_drift and args.no_winner_local_drift:
        raise ValueError("Choose at most one of --use-winner-local-drift or --no-winner-local-drift")

    use_winner_local_drift = True
    if args.no_winner_local_drift:
        use_winner_local_drift = False

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("reports") / (f"{args.preset}_{stamp}" if args.preset else f"hierarchical_scale_{stamp}")
    else:
        output_dir = args.output_dir

    run_hierarchical_scale(
        source=args.source,
        source_type=args.source_type,
        hf_config=args.hf_config,
        text_field=args.text_field,
        train_tokens=args.train_tokens,
        eval_tokens=args.eval_tokens,
        routing_eval_queries=args.routing_eval_queries,
        latency_eval_queries=args.latency_eval_queries,
        output_dir=output_dir,
        seed=args.seed,
        n_columns=args.n_columns,
        column_latent_dim=args.column_latent_dim,
        k_routing=args.k_routing,
        index_rebuild_threshold=args.index_rebuild_threshold,
        routing_shards=args.routing_shards,
        shard_candidate_factor=args.shard_candidate_factor,
        neurons_per_column_assumption=args.neurons_per_column_assumption,
        memory_capacity=args.memory_capacity,
        input_weight_blend=args.input_weight_blend,
        input_synapse_ltp=args.input_synapse_ltp,
        input_synapse_ltd=args.input_synapse_ltd,
        input_weight_row_target=args.input_weight_row_target,
        homeostasis_beta=args.homeostasis_beta,
        homeostasis_lr=args.homeostasis_lr,
        slow_mean_decay=args.slow_mean_decay,
        use_winner_local_drift=use_winner_local_drift,
        drift_threshold=args.drift_threshold,
        micro_sleep_interval_tokens=args.micro_sleep_interval_tokens,
        micro_sleep_replay_steps=args.micro_sleep_replay_steps,
        micro_sleep_candidate_pool=args.micro_sleep_candidate_pool,
        micro_sleep_memory_blend=args.micro_sleep_memory_blend,
        deep_sleep_interval_tokens=args.deep_sleep_interval_tokens,
        deep_sleep_replay_steps=args.deep_sleep_replay_steps,
        deep_sleep_candidate_pool=args.deep_sleep_candidate_pool,
        deep_sleep_memory_blend=args.deep_sleep_memory_blend,
        deep_sleep_cooldown_tokens=args.deep_sleep_cooldown_tokens,
        emergency_deep_sleep_cooldown_tokens=args.emergency_deep_sleep_cooldown_tokens,
        drift_floor_history_tokens=args.drift_floor_history_tokens,
        drift_floor_check_interval_tokens=args.drift_floor_check_interval_tokens,
        drift_floor_window_tokens=args.drift_floor_window_tokens,
        drift_floor_trigger_min_tokens=args.drift_floor_trigger_min_tokens,
        drift_floor_rise_tolerance=args.drift_floor_rise_tolerance,
        prototype_momentum=args.prototype_momentum,
        checkpoint_out=args.checkpoint_out,
        save_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
