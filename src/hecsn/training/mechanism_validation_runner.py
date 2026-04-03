from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, List, Literal, Optional, cast

import numpy as np
import torch

from hecsn.config.model_config import HECSNConfig
from hecsn.config.presets import get_mechanism_validation_preset, mechanism_validation_preset_names
from hecsn.data.pattern_loader import load_train_eval_examples
from hecsn.data.rtf_encoder import RTFEncoder
from hecsn.reporting.io import write_json_file
from hecsn.reporting.mechanism_validation import (
    plot_mechanism_validation_artifacts,
    write_mechanism_validation_metrics_csv,
)
from hecsn.training.baselines import CharNGramMemory
from hecsn.training.behavioral_metrics import ascii_codes, clustering_metrics, completion_coherence, cosine_similarity
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.training.runner_utils import set_seed
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer


def _linear_slope(series: List[float]) -> float:
    if len(series) < 2:
        return 0.0
    n = float(len(series))
    x_mean = (n - 1.0) / 2.0
    y_mean = sum(series) / n
    num = 0.0
    den = 0.0
    for i, y in enumerate(series):
        dx = float(i) - x_mean
        num += dx * (y - y_mean)
        den += dx * dx
    if den == 0.0:
        return 0.0
    return num / den


def _window_mins(series: List[float], window_tokens: int) -> List[float]:
    if window_tokens <= 0:
        return []
    mins: List[float] = []
    for start in range(0, len(series), window_tokens):
        chunk = series[start : start + window_tokens]
        if chunk:
            mins.append(float(min(chunk)))
    return mins


def _sleep_aggressiveness_stats(metrics_rows: List[dict[str, Any]]) -> dict[str, float]:
    total_tokens = max(1, len(metrics_rows))
    events_all = [row for row in metrics_rows if int(row.get("sleep_triggered", 0)) == 1]
    total_updates = float(sum(int(row.get("sleep_replay_updates", 0)) for row in events_all))

    def _event_stats(rows: List[dict[str, Any]]) -> tuple[float, float, float, float]:
        trigger_tokens = [int(row["token"]) for row in rows]
        if len(trigger_tokens) >= 2:
            intervals = [float(b - a) for a, b in zip(trigger_tokens[:-1], trigger_tokens[1:])]
            avg_interval = float(sum(intervals) / len(intervals))
            median_interval = float(np.median(np.asarray(intervals, dtype=np.float32)))
        else:
            avg_interval = float("inf")
            median_interval = float("inf")

        events = float(len(trigger_tokens))
        events_per_1k = events / (float(total_tokens) / 1000.0)
        updates = float(sum(int(row.get("sleep_replay_updates", 0)) for row in rows))
        return events, events_per_1k, avg_interval, updates

    events, events_per_1k, avg_interval, _ = _event_stats(events_all)
    median_interval = float("inf")
    if len(events_all) >= 2:
        trigger_tokens = [int(row["token"]) for row in events_all]
        intervals = [float(b - a) for a, b in zip(trigger_tokens[:-1], trigger_tokens[1:])]
        median_interval = float(np.median(np.asarray(intervals, dtype=np.float32)))

    micro_rows = [row for row in events_all if row.get("sleep_type") == "micro"]
    deep_rows = [row for row in events_all if row.get("sleep_type") == "deep"]
    micro_events, micro_per_1k, _, micro_updates = _event_stats(micro_rows)
    deep_events, deep_per_1k, _, deep_updates = _event_stats(deep_rows)
    updates_per_event = total_updates / events if events > 0 else 0.0

    return {
        "events": events,
        "events_per_1k_tokens": events_per_1k,
        "avg_interval_tokens": avg_interval,
        "median_interval_tokens": median_interval,
        "replay_updates_total": total_updates,
        "replay_updates_per_event": updates_per_event,
        "micro_events": micro_events,
        "micro_events_per_1k_tokens": micro_per_1k,
        "micro_replay_updates_total": micro_updates,
        "deep_events": deep_events,
        "deep_events_per_1k_tokens": deep_per_1k,
        "deep_replay_updates_total": deep_updates,
    }


def _mean_reconstruction_error(trainer: HECSNTrainer, patterns: List[torch.Tensor]) -> float:
    if not patterns:
        return float("nan")
    vals = [trainer.reconstruction_error(p) for p in patterns]
    return float(sum(vals) / len(vals))


def _clustering_metrics(
    trainer: HECSNTrainer,
    patterns: List[torch.Tensor],
) -> tuple[Optional[float], Optional[float], str]:
    X: List[np.ndarray] = []
    labels: List[int] = []
    for p in patterns:
        key = trainer.routing_key_for_pattern(p).detach().cpu().numpy()
        X.append(key)
        labels.append(trainer.winner_for_pattern(p))
    tensors = [torch.from_numpy(row) for row in X]
    return clustering_metrics(tensors, labels)


def _random_assignment_baseline_eval_error(
    cfg: HECSNConfig,
    train_patterns: List[torch.Tensor],
    eval_patterns: List[torch.Tensor],
    seed: int,
) -> float:
    set_seed(seed)
    model = HECSNModelLite(cfg)
    trainer = HECSNTrainer(model, cfg)

    for p in train_patterns:
        x = p.to(model.device)
        routing = model.routing_key_from_pattern(x)
        rand_idx = torch.randint(0, cfg.n_columns, (1,), device=model.device)
        assembly = model.competitive.process(routing, rand_idx, modulator=0.5)
        model.memory_store.update(assembly, importance=0.5)

    return _mean_reconstruction_error(trainer, eval_patterns)


def run_mechanism_validation(
    source: str,
    source_type: Literal["auto", "file", "hf"],
    hf_config: Optional[str],
    text_field: str,
    train_tokens: int,
    eval_tokens: int,
    log_every: int,
    output_dir: Path,
    save_plots: bool,
    seed: int,
    n_columns: int,
    column_latent_dim: int,
    memory_capacity: int,
    input_representation: Literal["order_weighted_ascii", "unigram_ascii", "hashed_ngram"],
    hashed_ngram_dim: int,
    hashed_ngram_min_n: int,
    hashed_ngram_max_n: int,
    behavior_probe_samples: int,
    input_weight_blend: float,
    input_synapse_ltp: float,
    input_synapse_ltd: float,
    input_weight_row_target: float,
    homeostasis_beta: float,
    homeostasis_lr: float,
    slow_mean_decay: float,
    slow_memory_start_tokens: int,
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
) -> None:
    cfg = HECSNConfig(
        n_columns=n_columns,
        column_latent_dim=column_latent_dim,
        memory_capacity=memory_capacity,
        input_representation=input_representation,
        hashed_ngram_dim=hashed_ngram_dim,
        hashed_ngram_min_n=hashed_ngram_min_n,
        hashed_ngram_max_n=hashed_ngram_max_n,
        input_weight_blend=input_weight_blend,
        input_synapse_ltp=input_synapse_ltp,
        input_synapse_ltd=input_synapse_ltd,
        input_weight_row_target=input_weight_row_target,
        homeostasis_beta=homeostasis_beta,
        homeostasis_lr=homeostasis_lr,
        slow_mean_decay=slow_mean_decay,
        slow_memory_start_tokens=slow_memory_start_tokens,
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
    train_patterns = [pattern for _, pattern in train_examples]
    eval_patterns = [pattern for _, pattern in eval_examples]
    train_raw_windows = [raw_window for raw_window, _ in train_examples]
    eval_raw_windows = [raw_window for raw_window, _ in eval_examples]
    if not train_patterns:
        raise ValueError("No training patterns produced from source")
    if not eval_patterns:
        raise ValueError("No evaluation patterns produced; increase eval tokens or source size")

    set_seed(seed)
    model = HECSNModelLite(cfg)
    trainer = HECSNTrainer(model, cfg)

    output_dir.mkdir(parents=True, exist_ok=True)

    drifts: List[float] = []
    surprises: List[float] = []
    sparsities: List[float] = []
    recon_errors: List[float] = []
    metrics_rows: List[dict[str, Any]] = []

    for idx, (raw_window, pattern) in enumerate(train_examples, start=1):
        metrics = trainer.train_step(pattern, raw_window=raw_window)

        drifts.append(float(metrics["drift"]))
        surprises.append(float(metrics["surprise"]))
        sparsities.append(float(metrics["sparsity"]))
        recon_errors.append(float(metrics["recon_error"]))
        metrics_rows.append(
            {
                "token": int(metrics["token"]),
                "drift": float(metrics["drift"]),
                "surprise": float(metrics["surprise"]),
                "sparsity": float(metrics["sparsity"]),
                "winner": int(metrics["winner"]),
                "pred_error": float(metrics["pred_error"]) if "pred_error" in metrics else None,
                "recon_error": float(metrics["recon_error"]),
                "sleep_triggered": int(metrics.get("sleep_triggered", 0)),
                "sleep_type": str(metrics.get("sleep_type", "none")),
                "sleep_replay_updates": int(metrics.get("sleep_replay_updates", 0)),
                "sleep_events_total": int(metrics.get("sleep_events_total", 0)),
                "micro_sleep_events_total": int(metrics.get("micro_sleep_events_total", 0)),
                "deep_sleep_events_total": int(metrics.get("deep_sleep_events_total", 0)),
                "deep_sleep_emergency": int(metrics.get("deep_sleep_emergency", 0)),
                "drift_floor": float(metrics.get("drift_floor", metrics["drift"])),
                "drift_floor_rising": int(metrics.get("drift_floor_rising", 0)),
            }
        )

        if idx % log_every == 0:
            print(
                f"token={idx} drift={metrics['drift']:.4f} "
                f"surprise={metrics['surprise']:.4f} sparsity={metrics['sparsity']:.4f} "
                f"winner={metrics['winner']}"
            )

    n = max(1, len(drifts))
    print("\\nMechanism-validation summary")
    print(f"tokens={len(drifts)}")
    print(f"drift_mean={sum(drifts)/n:.6f}")
    print(f"surprise_mean={sum(surprises)/n:.6f}")
    print(f"sparsity_mean={sum(sparsities)/n:.6f}")

    sleep_events = sum(int(row.get("sleep_triggered", 0)) for row in metrics_rows)
    sleep_updates = sum(int(row.get("sleep_replay_updates", 0)) for row in metrics_rows)
    micro_sleep_events = sum(1 for row in metrics_rows if row.get("sleep_type") == "micro")
    deep_sleep_events = sum(1 for row in metrics_rows if row.get("sleep_type") == "deep")
    deep_sleep_emergencies = sum(int(row.get("deep_sleep_emergency", 0)) for row in metrics_rows)
    print(
        "sleep_events="
        f"{sleep_events} micro={micro_sleep_events} deep={deep_sleep_events} "
        f"deep_emergency={deep_sleep_emergencies} sleep_replay_updates={sleep_updates}"
    )

    sleep_stats = _sleep_aggressiveness_stats(metrics_rows)
    drift_floor = _window_mins(drifts, cfg.drift_floor_window_tokens)
    drift_floor_slope = _linear_slope(drift_floor)
    drift_floor_decreasing = bool(drift_floor_slope < 0.0)
    print(
        "sleep_aggressiveness="
        f"events_per_1k={sleep_stats['events_per_1k_tokens']:.3f} "
        f"avg_interval_tokens={sleep_stats['avg_interval_tokens']:.2f} "
        f"replay_updates_per_event={sleep_stats['replay_updates_per_event']:.2f}"
    )
    print(
        "drift_floor="
        f"window={cfg.drift_floor_window_tokens} "
        f"slope={drift_floor_slope:.8f} decreasing={drift_floor_decreasing}"
    )

    trained_eval_recon_error = _mean_reconstruction_error(trainer, eval_patterns)
    baseline_eval_recon_error = _random_assignment_baseline_eval_error(
        cfg=cfg,
        train_patterns=train_patterns,
        eval_patterns=eval_patterns,
        seed=seed,
    )
    ablation_gain = baseline_eval_recon_error - trained_eval_recon_error

    def _assembly_for_window(window: str) -> torch.Tensor:
        pattern = encoder.feature_vector(ascii_codes(window))
        return trainer.assembly_for_pattern(pattern)

    hecsn_behavior_b1 = completion_coherence(
        lambda prefix, candidate: cosine_similarity(_assembly_for_window(prefix), _assembly_for_window(candidate)),
        eval_raw_windows,
        behavior_probe_samples,
    )
    ngram_baseline = CharNGramMemory(max_context=max(1, cfg.window_size - 1))
    ngram_baseline.fit(train_raw_windows)
    baseline_behavior_b1 = completion_coherence(
        ngram_baseline.completion_score,
        eval_raw_windows,
        behavior_probe_samples,
    )
    eval_assemblies = [trainer.assembly_for_pattern(pattern) for pattern in eval_patterns]
    eval_assembly_labels = [int(torch.argmax(assembly).item()) for assembly in eval_assemblies]
    behavior_b2_silhouette, behavior_b2_dbi, behavior_b2_status = clustering_metrics(
        eval_assemblies,
        eval_assembly_labels,
    )

    silhouette, dbi, clustering_status = _clustering_metrics(trainer, eval_patterns)
    recon_slope = _linear_slope(recon_errors[-min(10000, len(recon_errors)):])

    gate_clustering = bool(
        (silhouette is not None and silhouette >= cfg.silhouette_min)
        or (dbi is not None and dbi <= cfg.davies_bouldin_max)
    )
    gate_recon_trend = bool(recon_slope < cfg.recon_slope_max)
    gate_ablation = bool(trained_eval_recon_error < baseline_eval_recon_error)
    gate_behavior_b1 = bool(
        hecsn_behavior_b1["sample_count"] > 0
        and hecsn_behavior_b1["mean_margin"] is not None
        and float(hecsn_behavior_b1["mean_margin"]) > 0.0
    )
    gate_behavior_b2 = bool(
        behavior_b2_silhouette is not None
        and float(behavior_b2_silhouette) >= 0.25
    )

    winner_counts = Counter(int(row["winner"]) for row in metrics_rows)
    winner_entropy = 0.0
    total_winners = max(1, sum(winner_counts.values()))
    for count in winner_counts.values():
        p = count / total_winners
        winner_entropy -= p * np.log2(p)
    max_entropy = float(np.log2(max(1, cfg.n_columns)))
    winner_entropy_fraction_of_max = float(winner_entropy / (max_entropy + 1e-8)) if max_entropy > 0.0 else 0.0
    winner_max_share = float(max(winner_counts.values()) / total_winners) if winner_counts else 0.0
    gate_entropy = bool(winner_entropy >= cfg.winner_entropy_min_bits)
    gate_pass = bool(
        gate_clustering
        and gate_recon_trend
        and gate_ablation
        and gate_entropy
        and gate_behavior_b1
        and gate_behavior_b2
    )

    print("\nMechanism-validation gate checks")
    print(f"clustering_pass={gate_clustering} silhouette={silhouette} dbi={dbi} status={clustering_status}")
    print(f"recon_trend_pass={gate_recon_trend} recon_slope={recon_slope:.8f}")
    print(
        "ablation_pass="
        f"{gate_ablation} trained_eval_recon={trained_eval_recon_error:.6f} "
        f"baseline_eval_recon={baseline_eval_recon_error:.6f}"
    )
    print(
        "behavior_b1_pass="
        f"{gate_behavior_b1} hecsn_success_rate={hecsn_behavior_b1['success_rate']} "
        f"hecsn_mean_margin={hecsn_behavior_b1['mean_margin']} "
        f"ngram_success_rate={baseline_behavior_b1['success_rate']} "
        f"ngram_mean_margin={baseline_behavior_b1['mean_margin']}"
    )
    print(
        "behavior_b2_pass="
        f"{gate_behavior_b2} silhouette={behavior_b2_silhouette} "
        f"dbi={behavior_b2_dbi} status={behavior_b2_status}"
    )
    print(
        f"entropy_pass={gate_entropy} winner_entropy={winner_entropy:.6f} "
        f"min_required={cfg.winner_entropy_min_bits:.6f}"
    )
    print(
        "winner_usage="
        f"fraction_of_max_entropy={winner_entropy_fraction_of_max:.6f} "
        f"max_share={winner_max_share:.6f}"
    )
    runtime_scope = model.runtime_scope_report()
    weight_distribution = runtime_scope.get("weight_distribution", {})
    input_weight_stats = weight_distribution.get("column_input_weights", {})
    input_weight_log_stats = weight_distribution.get("column_input_weights_log_space", {})
    proto_stats = weight_distribution.get("prototype_components", {})
    print(
        "weight_validation="
        f"status={weight_distribution.get('status', 'unknown')} "
        f"full_target_supported={runtime_scope.get('validates_full_log_stdp_weight_target', False)}"
    )
    print(
        "weight_stats="
        f"input_excess_kurtosis={input_weight_stats.get('excess_kurtosis', float('nan')):.6f} "
        f"input_log_excess_kurtosis={input_weight_log_stats.get('excess_kurtosis', float('nan')):.6f} "
        f"prototype_excess_kurtosis={proto_stats.get('excess_kurtosis', float('nan')):.6f}"
    )
    print(f"mechanism_validation_gate_pass={gate_pass}")

    summary = {
        "protocol": "mechanism_validation_hf",
        "benchmark": "mechanism_validation",
        "tokens": len(drifts),
        "eval_tokens": len(eval_patterns),
        "data_setup": {
            "source": source,
            "source_type": source_type,
            "hf_config": hf_config,
            "text_field": text_field,
            "train_tokens_target": train_tokens,
            "eval_tokens_target": eval_tokens,
            "train_tokens_used": len(drifts),
            "eval_tokens_used": len(eval_patterns),
            "n_columns": cfg.n_columns,
            "column_latent_dim": cfg.column_latent_dim,
            "estimated_neurons": cfg.n_columns * cfg.neurons_per_column_assumption,
            "neurons_per_column_assumption": cfg.neurons_per_column_assumption,
            "input_representation": cfg.input_representation,
            "input_dim": cfg.input_dim,
            "hashed_ngram_dim": cfg.hashed_ngram_dim,
            "hashed_ngram_min_n": cfg.hashed_ngram_min_n,
            "hashed_ngram_max_n": cfg.hashed_ngram_max_n,
            "behavior_probe_samples": int(behavior_probe_samples),
            "input_weight_blend": cfg.input_weight_blend,
            "input_synapse_ltp": cfg.input_synapse_ltp,
            "input_synapse_ltd": cfg.input_synapse_ltd,
            "input_weight_row_target": cfg.input_weight_row_target,
            "homeostasis_beta": cfg.homeostasis_beta,
            "homeostasis_lr": cfg.homeostasis_lr,
            "slow_mean_decay": cfg.slow_mean_decay,
            "slow_memory_start_tokens": cfg.slow_memory_start_tokens,
            "use_winner_local_drift": cfg.use_winner_local_drift,
        },
        "drift_mean": sum(drifts) / n,
        "surprise_mean": sum(surprises) / n,
        "sparsity_mean": sum(sparsities) / n,
        "recon_error_mean": sum(recon_errors) / max(1, len(recon_errors)),
        "drift_slope": _linear_slope(drifts),
        "surprise_slope": _linear_slope(surprises),
        "sparsity_slope": _linear_slope(sparsities),
        "recon_error_slope": recon_slope,
        "winner_entropy_bits": float(winner_entropy),
        "winner_entropy_fraction_of_max": winner_entropy_fraction_of_max,
        "winner_max_share": winner_max_share,
        "winners_used": len(winner_counts),
        "runtime_scope": runtime_scope,
        "sleep_events": int(sleep_events),
        "sleep_replay_updates": int(sleep_updates),
        "sleep_aggressiveness": {
            **sleep_stats,
            "drift_threshold": cfg.drift_threshold,
            "micro_sleep_interval_tokens": cfg.micro_sleep_interval_tokens,
            "micro_sleep_replay_steps": cfg.micro_sleep_replay_steps,
            "micro_sleep_candidate_pool": cfg.micro_sleep_candidate_pool,
            "micro_sleep_memory_blend": cfg.micro_sleep_memory_blend,
            "deep_sleep_interval_tokens": cfg.deep_sleep_interval_tokens,
            "deep_sleep_replay_steps": cfg.deep_sleep_replay_steps,
            "deep_sleep_candidate_pool": cfg.deep_sleep_candidate_pool,
            "deep_sleep_memory_blend": cfg.deep_sleep_memory_blend,
            "deep_sleep_cooldown_tokens": cfg.deep_sleep_cooldown_tokens,
            "emergency_deep_sleep_cooldown_tokens": cfg.emergency_deep_sleep_cooldown_tokens,
            "drift_floor_history_tokens": cfg.drift_floor_history_tokens,
            "drift_floor_check_interval_tokens": cfg.drift_floor_check_interval_tokens,
            "drift_floor_trigger_min_tokens": cfg.drift_floor_trigger_min_tokens,
            "drift_floor_rise_tolerance": cfg.drift_floor_rise_tolerance,
            "prototype_momentum": cfg.prototype_momentum,
            "input_weight_blend": cfg.input_weight_blend,
            "input_synapse_ltp": cfg.input_synapse_ltp,
            "input_synapse_ltd": cfg.input_synapse_ltd,
            "input_weight_row_target": cfg.input_weight_row_target,
            "homeostasis_beta": cfg.homeostasis_beta,
            "homeostasis_lr": cfg.homeostasis_lr,
            "slow_mean_decay": cfg.slow_mean_decay,
            "slow_memory_start_tokens": cfg.slow_memory_start_tokens,
            "use_winner_local_drift": cfg.use_winner_local_drift,
        },
        "drift_floor": {
            "window_tokens": cfg.drift_floor_window_tokens,
            "values": drift_floor,
            "slope": drift_floor_slope,
            "decreasing": drift_floor_decreasing,
        },
        "eval_metrics": {
            "trained_eval_recon_error": trained_eval_recon_error,
            "random_assignment_eval_recon_error": baseline_eval_recon_error,
            "ablation_gain": ablation_gain,
            "routing_key_silhouette": silhouette,
            "routing_key_davies_bouldin": dbi,
            "routing_key_clustering_status": clustering_status,
        },
        "behavioral_metrics": {
            "character_ngram_recovery": {
                **hecsn_behavior_b1,
                "pass": gate_behavior_b1,
                "baseline": {
                    **baseline_behavior_b1,
                    "model": "char_ngram_memory",
                    "max_context": max(1, cfg.window_size - 1),
                },
            },
            "distributional_clustering": {
                "sample_count": int(len(eval_assemblies)),
                "cluster_count_used": int(len(set(eval_assembly_labels))),
                "silhouette": behavior_b2_silhouette,
                "davies_bouldin": behavior_b2_dbi,
                "clustering_status": behavior_b2_status,
                "pass": gate_behavior_b2,
            },
        },
        "mechanism_validation_gate": {
            "pass": gate_pass,
            "gate_clustering": gate_clustering,
            "gate_reconstruction_trend": gate_recon_trend,
            "gate_ablation_superiority": gate_ablation,
            "gate_behavior_character_ngram_recovery": gate_behavior_b1,
            "gate_behavior_distributional_clustering": gate_behavior_b2,
            "gate_winner_entropy": gate_entropy,
            "thresholds": {
                "silhouette_min": cfg.silhouette_min,
                "davies_bouldin_max": cfg.davies_bouldin_max,
                "recon_slope_max": cfg.recon_slope_max,
                "winner_entropy_min_bits": cfg.winner_entropy_min_bits,
                "behavior_b1_mean_margin_min": 0.0,
                "behavior_b2_silhouette_min": 0.25,
                "ablation_requires": "trained_eval_recon_error < random_assignment_eval_recon_error",
            },
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    write_mechanism_validation_metrics_csv(output_dir / "metrics.csv", metrics_rows)
    write_json_file(output_dir / "summary.json", summary)
    checkpoint_path: Optional[Path] = None
    if checkpoint_out is not None:
        checkpoint_path = save_trainer_checkpoint(
            checkpoint_out,
            trainer,
            metadata={
                "protocol": "mechanism_validation_hf",
                "benchmark": "mechanism_validation",
                "source": source,
                "source_type": source_type,
                "hf_config": hf_config,
                "text_field": text_field,
                "train_tokens": len(train_patterns),
                "eval_tokens": len(eval_patterns),
            },
        )
    if save_plots:
        plot_mechanism_validation_artifacts(output_dir, metrics_rows, summary)

    print("\nSaved artifacts")
    print(f"output_dir={output_dir}")
    print(f"metrics_csv={output_dir / 'metrics.csv'}")
    print(f"summary_json={output_dir / 'summary.json'}")
    if checkpoint_path is not None:
        print(f"checkpoint_path={checkpoint_path}")
    if save_plots:
        print(
            "plots="
            f"{[str(output_dir / 'mechanism_validation_metrics.png'), str(output_dir / 'bootstrap_prediction_error.png'), str(output_dir / 'mechanism_validation_scorecard.png')]}"
        )


def main() -> None:
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument("--preset", choices=mechanism_validation_preset_names(), default=None)
    preset_args, _ = preset_parser.parse_known_args()
    preset_defaults = get_mechanism_validation_preset(preset_args.preset)

    parser = argparse.ArgumentParser(description="Run the HECSN mechanism-validation benchmark with automatic gate checks")
    parser.add_argument("--preset", choices=mechanism_validation_preset_names(), default=preset_args.preset)
    parser.add_argument("--source", type=str, default=preset_defaults.get("source"), help="File path or HuggingFace dataset id")
    parser.add_argument("--source-type", choices=["auto", "file", "hf"], default=preset_defaults.get("source_type", "auto"))
    parser.add_argument("--hf-config", type=str, default=preset_defaults.get("hf_config"), help="HuggingFace dataset config name")
    parser.add_argument("--text-field", type=str, default=preset_defaults.get("text_field", "text"))

    parser.add_argument("--train-tokens", type=int, default=preset_defaults.get("train_tokens", 5000))
    parser.add_argument("--eval-tokens", type=int, default=preset_defaults.get("eval_tokens", 1000))
    parser.add_argument("--log-every", type=int, default=preset_defaults.get("log_every", 200))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=preset_defaults.get("seed", 7))
    parser.add_argument("--n-columns", type=int, default=preset_defaults.get("n_columns", 10))
    parser.add_argument("--column-latent-dim", type=int, default=preset_defaults.get("column_latent_dim", 256))
    parser.add_argument("--memory-capacity", type=int, default=preset_defaults.get("memory_capacity", 1000))
    parser.add_argument(
        "--input-representation",
        choices=["order_weighted_ascii", "unigram_ascii", "hashed_ngram"],
        default=preset_defaults.get("input_representation", "order_weighted_ascii"),
    )
    parser.add_argument("--hashed-ngram-dim", type=int, default=preset_defaults.get("hashed_ngram_dim", 2048))
    parser.add_argument("--hashed-ngram-min-n", type=int, default=preset_defaults.get("hashed_ngram_min_n", 2))
    parser.add_argument("--hashed-ngram-max-n", type=int, default=preset_defaults.get("hashed_ngram_max_n", 3))
    parser.add_argument("--behavior-probe-samples", type=int, default=preset_defaults.get("behavior_probe_samples", 128))
    parser.add_argument("--input-weight-blend", type=float, default=preset_defaults.get("input_weight_blend", 0.02))
    parser.add_argument("--input-synapse-ltp", type=float, default=preset_defaults.get("input_synapse_ltp", 0.02))
    parser.add_argument("--input-synapse-ltd", type=float, default=preset_defaults.get("input_synapse_ltd", 0.01))
    parser.add_argument("--input-weight-row-target", type=float, default=preset_defaults.get("input_weight_row_target", 1.0))
    parser.add_argument("--homeostasis-beta", type=float, default=preset_defaults.get("homeostasis_beta", 0.01))
    parser.add_argument("--homeostasis-lr", type=float, default=preset_defaults.get("homeostasis_lr", 0.2))
    parser.add_argument("--slow-mean-decay", type=float, default=preset_defaults.get("slow_mean_decay", 1.0))
    parser.add_argument("--slow-memory-start-tokens", type=int, default=preset_defaults.get("slow_memory_start_tokens", 0))
    parser.add_argument("--use-winner-local-drift", action="store_true", default=bool(preset_defaults.get("use_winner_local_drift", False)))
    parser.add_argument("--no-winner-local-drift", action="store_true")
    parser.add_argument("--drift-threshold", type=float, default=preset_defaults.get("drift_threshold", 0.02))
    parser.add_argument("--micro-sleep-interval-tokens", type=int, default=preset_defaults.get("micro_sleep_interval_tokens", 200))
    parser.add_argument("--micro-sleep-replay-steps", type=int, default=preset_defaults.get("micro_sleep_replay_steps", 10))
    parser.add_argument("--micro-sleep-candidate-pool", type=int, default=preset_defaults.get("micro_sleep_candidate_pool", 5))
    parser.add_argument("--micro-sleep-memory-blend", type=float, default=preset_defaults.get("micro_sleep_memory_blend", 0.05))
    parser.add_argument("--deep-sleep-interval-tokens", type=int, default=preset_defaults.get("deep_sleep_interval_tokens", 5000))
    parser.add_argument("--deep-sleep-replay-steps", type=int, default=preset_defaults.get("deep_sleep_replay_steps", 100))
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
    args = parser.parse_args()

    source = args.source
    if source is None:
        raise ValueError("Provide --source")

    train_tokens = int(args.train_tokens)

    if args.source_type not in {"auto", "file", "hf"}:
        raise ValueError("source_type must be one of: auto, file, hf")
    src_type = cast(Literal["auto", "file", "hf"], args.source_type)

    hf_config = args.hf_config
    if src_type == "hf" and source == "wikitext" and hf_config is None:
        hf_config = "wikitext-103-raw-v1"
        print(f"Selected default wikitext config: {hf_config}")

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("reports") / (f"{args.preset}_{stamp}" if args.preset else f"mechanism_validation_{stamp}")
    else:
        output_dir = args.output_dir

    if args.use_winner_local_drift and args.no_winner_local_drift:
        raise ValueError("Choose at most one of --use-winner-local-drift or --no-winner-local-drift")

    use_winner_local_drift = True
    if args.no_winner_local_drift:
        use_winner_local_drift = False

    run_mechanism_validation(
        source=source,
        source_type=src_type,
        hf_config=hf_config,
        text_field=args.text_field,
        train_tokens=train_tokens,
        eval_tokens=args.eval_tokens,
        log_every=args.log_every,
        output_dir=output_dir,
        save_plots=(not args.no_plots),
        seed=args.seed,
        n_columns=args.n_columns,
        column_latent_dim=args.column_latent_dim,
        memory_capacity=args.memory_capacity,
        input_representation=cast(Literal["order_weighted_ascii", "unigram_ascii", "hashed_ngram"], args.input_representation),
        hashed_ngram_dim=args.hashed_ngram_dim,
        hashed_ngram_min_n=args.hashed_ngram_min_n,
        hashed_ngram_max_n=args.hashed_ngram_max_n,
        behavior_probe_samples=args.behavior_probe_samples,
        input_weight_blend=args.input_weight_blend,
        input_synapse_ltp=args.input_synapse_ltp,
        input_synapse_ltd=args.input_synapse_ltd,
        input_weight_row_target=args.input_weight_row_target,
        homeostasis_beta=args.homeostasis_beta,
        homeostasis_lr=args.homeostasis_lr,
        slow_mean_decay=args.slow_mean_decay,
        slow_memory_start_tokens=args.slow_memory_start_tokens,
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
    )


if __name__ == "__main__":
    main()
