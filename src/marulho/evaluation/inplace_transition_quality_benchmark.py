"""Quality and throughput A/B for the evaluation-only in-place CUDA transition."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

import torch

from marulho.evaluation.inplace_hot_window_benchmark import (
    install_inplace_transition_for_benchmark,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.checkpointing import load_trainer_checkpoint


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def _winner_distribution(winners: Sequence[int], n_columns: int) -> dict[str, Any]:
    counts = Counter(int(winner) for winner in winners)
    sample_count = max(1, len(winners))
    probabilities = [count / sample_count for count in counts.values()]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    max_unique = min(sample_count, max(1, int(n_columns)))
    normalized_entropy = entropy / math.log(max_unique) if max_unique > 1 else 0.0
    return {
        "unique_winners": len(counts),
        "normalized_entropy": normalized_entropy,
        "max_winner_share": max(counts.values(), default=0) / sample_count,
        "counts": {str(key): value for key, value in sorted(counts.items())},
    }


def _state_is_finite(trainer: Any) -> bool:
    tensors = (
        trainer.model.competitive.prototypes,
        trainer.model.competitive.prototype_velocity,
        trainer.model.competitive.thresholds,
        trainer.model.competitive.win_rate_ema,
        trainer.model.predictive.location,
        trainer.model.predictive.velocity,
        trainer.model.predictive._prediction_weights,
        trainer.model.predictive.prediction_error,
        trainer.model.predictive.confidence,
    )
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)


def _run_arm(
    checkpoint: Path,
    *,
    executor: str,
    patterns: Sequence[torch.Tensor],
    warmup_steps: int,
) -> dict[str, Any]:
    trainer, metadata = load_trainer_checkpoint(checkpoint)
    trainer.config.micro_sleep_interval_tokens = 10**9
    trainer.config.deep_sleep_interval_tokens = 10**9
    if executor == "inplace_triton_runtime":
        install_inplace_transition_for_benchmark(trainer)
    elif executor != "runtime":
        raise ValueError(f"unsupported transition executor: {executor}")

    device = trainer.model.device
    device_patterns = [pattern.to(device) for pattern in patterns]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for index in range(warmup_steps):
        trainer.train_step(
            device_patterns[index],
            raw_window=f"quality warmup {index}",
            allow_sleep_maintenance=False,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    latencies_ms: list[float] = []
    reconstruction_errors: list[float] = []
    drift_values: list[float] = []
    active_columns: list[float] = []
    sparsity_values: list[float] = []
    winners: list[int] = []
    started = time.perf_counter_ns()
    for index, pattern in enumerate(device_patterns[warmup_steps:], start=warmup_steps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_started = time.perf_counter_ns()
        metrics = trainer.train_step(
            pattern,
            raw_window=f"quality measure {index}",
            allow_sleep_maintenance=False,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies_ms.append((time.perf_counter_ns() - step_started) / 1e6)
        reconstruction_errors.append(float(metrics.get("recon_error", 0.0)))
        drift_values.append(float(metrics.get("drift", 0.0)))
        active_columns.append(float(metrics.get("active_columns", 0.0)))
        sparsity_values.append(float(metrics.get("sparsity", 0.0)))
        winners.append(int(metrics["winner"]))
    total_elapsed_s = (time.perf_counter_ns() - started) / 1e9

    predictive = trainer.model.predictive
    memory = trainer.model.memory_store.summary_stats(
        current_token=trainer.token_count,
        force=True,
    )
    sample_count = len(winners)
    return {
        "executor": executor,
        "device": str(device),
        "metadata": metadata,
        "sample_count": sample_count,
        "throughput_ticks_per_second": sample_count / max(total_elapsed_s, 1e-9),
        "total_elapsed_s": total_elapsed_s,
        "latency_ms": {
            "median": statistics.median(latencies_ms),
            "p95": _percentile(latencies_ms, 0.95),
            "mean": statistics.fmean(latencies_ms),
        },
        "reconstruction_error": {
            "mean": statistics.fmean(reconstruction_errors),
            "p95": _percentile(reconstruction_errors, 0.95),
        },
        "prediction": {
            "error_mean": float(predictive.prediction_error.mean().item()),
            "error_max": float(predictive.prediction_error.max().item()),
            "confidence_mean": float(predictive.confidence.mean().item()),
            "confidence_min": float(predictive.confidence.min().item()),
        },
        "trajectory": {
            "drift_mean": statistics.fmean(drift_values),
            "active_columns_mean": statistics.fmean(active_columns),
            "sparsity_mean": statistics.fmean(sparsity_values),
            "winners": winners,
            "winner_distribution": _winner_distribution(
                winners,
                trainer.model.competitive.n_columns,
            ),
        },
        "spike_health": trainer.model.competitive.spike_health_report(),
        "memory": memory,
        "state_finite": _state_is_finite(trainer),
        "grounding": {
            "cross_modal_layer_configured": trainer.model.cross_modal is not None,
            "quality_evidence_available": False,
            "reason": "benchmark_supplies_encoded_text_tensors_without_visual_or_audio_evidence",
        },
        "cuda_memory": (
            {
                "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
            }
            if device.type == "cuda"
            else {}
        ),
    }


def compare_quality_arms(
    baseline: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_recon = float(baseline["reconstruction_error"]["mean"])
    variant_recon = float(variant["reconstruction_error"]["mean"])
    baseline_prediction = float(baseline["prediction"]["error_mean"])
    variant_prediction = float(variant["prediction"]["error_mean"])
    baseline_confidence = float(baseline["prediction"]["confidence_mean"])
    variant_confidence = float(variant["prediction"]["confidence_mean"])
    baseline_entropy = float(
        baseline["trajectory"]["winner_distribution"]["normalized_entropy"]
    )
    variant_entropy = float(
        variant["trajectory"]["winner_distribution"]["normalized_entropy"]
    )
    baseline_spike = baseline["spike_health"]
    variant_spike = variant["spike_health"]
    baseline_memory = baseline["memory"]
    variant_memory = variant["memory"]
    baseline_winners = list(baseline["trajectory"]["winners"])
    variant_winners = list(variant["trajectory"]["winners"])
    winner_pairs = min(len(baseline_winners), len(variant_winners))
    winner_agreement = (
        sum(
            baseline_winners[index] == variant_winners[index]
            for index in range(winner_pairs)
        )
        / winner_pairs
        if winner_pairs
        else 0.0
    )

    gates = {
        "speedup_at_least_1_10x": (
            float(variant["throughput_ticks_per_second"])
            / max(float(baseline["throughput_ticks_per_second"]), 1e-9)
        )
        >= 1.10,
        "reconstruction_mean_within_tolerance": variant_recon
        <= baseline_recon + max(0.01, baseline_recon * 0.05),
        "prediction_error_mean_within_tolerance": variant_prediction
        <= baseline_prediction + 0.01,
        "prediction_confidence_preserved": variant_confidence
        >= baseline_confidence - 0.02,
        "winner_diversity_preserved": variant_entropy >= baseline_entropy - 0.10,
        "no_new_spike_health_risk": (
            variant_spike["activity_state"] == "sparse_responsive"
            or variant_spike["activity_state"] == baseline_spike["activity_state"]
        ),
        "silent_fraction_not_regressed": float(variant_spike["silent_fraction"])
        <= float(baseline_spike["silent_fraction"]) + 0.05,
        "saturated_fraction_not_regressed": float(
            variant_spike["saturated_fraction"]
        )
        <= float(baseline_spike["saturated_fraction"]) + 0.05,
        "memory_capture_not_regressed": float(
            variant_memory["mean_capture_strength"]
        )
        >= float(baseline_memory["mean_capture_strength"]) - 0.05,
        "memory_fragility_not_regressed": float(variant_memory["mean_fragility"])
        <= float(baseline_memory["mean_fragility"]) + 0.10,
        "finite_state": bool(baseline["state_finite"])
        and bool(variant["state_finite"]),
    }
    quality_preserved = all(gates.values())
    grounding_available = bool(
        baseline["grounding"]["quality_evidence_available"]
        and variant["grounding"]["quality_evidence_available"]
    )
    speedup = float(variant["throughput_ticks_per_second"]) / max(
        float(baseline["throughput_ticks_per_second"]),
        1e-9,
    )
    return {
        "quality_preserved": quality_preserved,
        "grounding_quality_available": grounding_available,
        "promotion_eligible": quality_preserved and grounding_available,
        "gates": gates,
        "deltas": {
            "speedup": speedup,
            "reconstruction_error_mean": variant_recon - baseline_recon,
            "prediction_error_mean": variant_prediction - baseline_prediction,
            "prediction_confidence_mean": variant_confidence - baseline_confidence,
            "winner_entropy_normalized": variant_entropy - baseline_entropy,
            "memory_capture_strength": float(
                variant_memory["mean_capture_strength"]
            )
            - float(baseline_memory["mean_capture_strength"]),
            "memory_consolidation_level": float(
                variant_memory["mean_consolidation_level"]
            )
            - float(baseline_memory["mean_consolidation_level"]),
            "memory_fragility": float(variant_memory["mean_fragility"])
            - float(baseline_memory["mean_fragility"]),
        },
        "winner_agreement": winner_agreement,
        "winner_agreement_interpretation": (
            "diagnostic_only; divergent specialists are acceptable only when "
            "aggregate cognitive quality and grounding remain supported"
        ),
    }


def run_inplace_transition_quality_benchmark(
    *,
    checkpoint_path: str | Path,
    samples: int = 128,
    warmup_steps: int = 8,
    seed: int = 20260620,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("in-place transition quality benchmark requires CUDA")

    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    probe_trainer, _ = load_trainer_checkpoint(checkpoint)
    input_dim = int(probe_trainer.config.input_dim)
    del probe_trainer
    generator = torch.Generator(device="cpu").manual_seed(seed)
    patterns = [
        torch.rand(input_dim, generator=generator)
        for _ in range(samples + warmup_steps)
    ]

    baseline = _run_arm(
        checkpoint,
        executor="runtime",
        patterns=patterns,
        warmup_steps=warmup_steps,
    )
    torch.cuda.empty_cache()
    variant = _run_arm(
        checkpoint,
        executor="inplace_triton_runtime",
        patterns=patterns,
        warmup_steps=warmup_steps,
    )
    comparison = compare_quality_arms(baseline, variant)
    report = {
        "schema_version": 1,
        "artifact_kind": "marulho_inplace_transition_quality_benchmark",
        "surface": "inplace_transition_quality_benchmark.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "eligible_for_operator_review"
            if comparison["promotion_eligible"]
            else "quality_supported_pending_grounded_evaluation"
            if comparison["quality_preserved"]
            else "blocked_quality_regression"
        ),
        "passed": bool(comparison["promotion_eligible"]),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _checkpoint_hash(checkpoint),
        },
        "seed": int(seed),
        "samples": int(samples),
        "warmup_steps": int(warmup_steps),
        "baseline": baseline,
        "variant": variant,
        "comparison": comparison,
        "promotion_gate": {
            "runtime_default_changed": False,
            "checkpoint_opt_in_required": True,
            "compile_only_startup_warmup_available": True,
            "grounded_visual_or_audio_quality_evidence_available_elsewhere": True,
            "production_executor_available": bool(
                comparison["promotion_eligible"]
            ),
            "eligible": bool(comparison["promotion_eligible"]),
        },
        "mutates_live_runtime": False,
        "writes_live_checkpoint": False,
        "claim_boundary": (
            "same-checkpoint encoded-tensor evaluation of sequential train_step "
            "quality and throughput; no visual/audio grounding, service, source, "
            "sleep, checkpoint write, or live-runtime mutation"
        ),
    }
    if output_path is not None:
        write_json_report_with_readme(
            output_path,
            report,
            title="In-Place CUDA Transition Quality Benchmark",
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args(argv)
    report = run_inplace_transition_quality_benchmark(
        checkpoint_path=args.checkpoint,
        samples=args.samples,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] != "blocked_quality_regression" else 1


if __name__ == "__main__":
    raise SystemExit(main())
