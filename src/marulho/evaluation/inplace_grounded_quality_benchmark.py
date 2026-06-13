"""Sensory-grounded CUDA A/B for the evaluation-only in-place transition."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from marulho.config.model_config import MarulhoConfig
from marulho.evaluation.inplace_hot_window_benchmark import (
    install_inplace_transition_for_benchmark,
)
from marulho.evaluation.inplace_transition_quality_benchmark import (
    _percentile,
    _state_is_finite,
    _winner_distribution,
    compare_quality_arms,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.checkpointing import (
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


@contextmanager
def _observe_triton_compilation(context: Mapping[str, Any]) -> Any:
    try:
        import triton.knobs as triton_knobs
    except ImportError:
        yield []
        return

    events: list[dict[str, Any]] = []
    starts: dict[str, int] = {}
    previous_cache_hook = triton_knobs.runtime.jit_cache_hook
    previous_post_hook = triton_knobs.runtime.jit_post_compile_hook

    def cache_hook(**kwargs: Any) -> bool | None:
        key = str(kwargs["key"])
        starts[key] = time.perf_counter_ns()
        if previous_cache_hook is not None:
            return previous_cache_hook(**kwargs)
        return None

    def post_hook(**kwargs: Any) -> bool | None:
        key = str(kwargs["key"])
        started = starts.pop(key, time.perf_counter_ns())
        compile_data = kwargs.get("compile") or {}
        specialization_data = str(compile_data.get("specialization_data", ""))
        events.append(
            {
                "key_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                "key": key,
                "specialization_data": specialization_data,
                "duration_s": (time.perf_counter_ns() - started) / 1e9,
                "manual_warmup": bool(kwargs.get("is_manual_warmup")),
                "already_compiled": bool(kwargs.get("already_compiled")),
                "phase": str(context.get("phase", "unknown")),
                "tick_index": context.get("tick_index"),
            }
        )
        if previous_post_hook is not None:
            return previous_post_hook(**kwargs)
        return None

    triton_knobs.runtime.jit_cache_hook = cache_hook
    triton_knobs.runtime.jit_post_compile_hook = post_hook
    try:
        yield events
    finally:
        triton_knobs.runtime.jit_cache_hook = previous_cache_hook
        triton_knobs.runtime.jit_post_compile_hook = previous_post_hook


def _make_grounded_stream(
    *,
    input_dim: int,
    visual_dim: int,
    audio_dim: int,
    samples: int,
    concepts: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    text_prototypes = torch.rand(concepts, input_dim, generator=generator)
    visual_signatures = torch.zeros(concepts, visual_dim)
    audio_signatures = torch.zeros(concepts, audio_dim)
    visual_width = max(2, min(8, visual_dim // max(1, concepts)))
    audio_width = max(2, min(4, audio_dim // max(1, concepts)))
    for concept_id in range(concepts):
        visual_start = (concept_id * visual_width) % visual_dim
        audio_start = (concept_id * audio_width) % audio_dim
        visual_indices = (
            torch.arange(visual_width) + visual_start
        ) % visual_dim
        audio_indices = (
            torch.arange(audio_width) + audio_start
        ) % audio_dim
        visual_signatures[concept_id, visual_indices] = 1.0
        audio_signatures[concept_id, audio_indices] = 1.0

    stream: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
    for index in range(samples):
        concept_id = index % concepts
        noise = 0.015 * torch.rand(input_dim, generator=generator)
        pattern = torch.clamp(text_prototypes[concept_id] + noise, 0.0, 1.0)
        stream.append(
            (
                pattern,
                visual_signatures[concept_id].clone(),
                audio_signatures[concept_id].clone(),
                concept_id,
            )
        )
    return stream


def prepare_grounded_checkpoint(
    checkpoint_path: str | Path,
    *,
    n_columns: int = 1024,
    training_samples: int = 256,
    concepts: int = 8,
    seed: int = 20260621,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("grounded in-place benchmark requires CUDA")
    cfg = MarulhoConfig(
        n_columns=n_columns,
        column_latent_dim=64,
        bootstrap_tokens=0,
        memory_capacity=64,
        routing_index_mode="torch_topk",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        enable_cross_modal=True,
        cross_modal_dim_visual=64,
        cross_modal_dim_audio=32,
        micro_sleep_interval_tokens=10**9,
        deep_sleep_interval_tokens=10**9,
    )
    trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
    trainer.developmental_stage = 1
    stream = _make_grounded_stream(
        input_dim=cfg.input_dim,
        visual_dim=cfg.cross_modal_dim_visual,
        audio_dim=cfg.cross_modal_dim_audio,
        samples=training_samples,
        concepts=concepts,
        seed=seed,
    )
    for pattern, visual, audio, concept_id in stream:
        trainer.train_step(
            pattern.to(trainer.model.device),
            raw_window=f"grounded concept {concept_id}",
            visual_spikes=visual.to(trainer.model.device),
            audio_spikes=audio.to(trainer.model.device),
            allow_sleep_maintenance=False,
        )
    return save_trainer_checkpoint(
        checkpoint_path,
        trainer,
        metadata={
            "benchmark": "inplace_grounded_quality",
            "synthetic_multimodal": True,
            "concepts": concepts,
            "training_samples": training_samples,
            "seed": seed,
        },
    )


def _cosine(prediction: torch.Tensor, target: torch.Tensor) -> float:
    if float(prediction.norm().item()) <= 1e-8:
        return 0.0
    return float(
        F.cosine_similarity(
            prediction.unsqueeze(0),
            target.unsqueeze(0),
        ).item()
    )


def _run_grounded_arm(
    checkpoint: Path,
    *,
    executor: str,
    stream: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
    warmup_steps: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    trainer, metadata = load_trainer_checkpoint(checkpoint)
    trainer.config.micro_sleep_interval_tokens = 10**9
    trainer.config.deep_sleep_interval_tokens = 10**9
    trainer.developmental_stage = 1
    if trainer.model.cross_modal is None:
        raise RuntimeError("grounded checkpoint has no cross-modal layer")
    if executor == "inplace_triton_runtime":
        install_inplace_transition_for_benchmark(trainer)
    elif executor == "fused_triton_text_runtime":
        from marulho.evaluation.fused_route_vote_hot_window_benchmark import (
            install_fused_route_vote_for_benchmark,
        )

        install_fused_route_vote_for_benchmark(trainer)
    elif executor == "cuda_graph_text_runtime":
        from marulho.evaluation.fused_route_vote_hot_window_benchmark import (
            install_cuda_graph_route_transition_for_benchmark,
        )

        install_cuda_graph_route_transition_for_benchmark(trainer)
    elif executor != "runtime":
        raise ValueError(f"unsupported executor: {executor}")

    device = trainer.model.device
    device_stream = [
        (
            pattern.to(device),
            visual.to(device),
            audio.to(device),
            concept_id,
        )
        for pattern, visual, audio, concept_id in stream
    ]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    compile_context: dict[str, Any] = {
        "phase": "warmup",
        "tick_index": None,
    }
    with _observe_triton_compilation(compile_context) as triton_compile_events:
        for index in range(warmup_steps):
            compile_context["tick_index"] = index
            pattern, visual, audio, concept_id = device_stream[index]
            trainer.train_step(
                pattern,
                raw_window=f"grounded warmup concept {concept_id}",
                visual_spikes=visual,
                audio_spikes=audio,
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
        visual_cosines: list[float] = []
        audio_cosines: list[float] = []
        visual_accepts = 0
        audio_accepts = 0
        started = time.perf_counter_ns()
        compile_context["phase"] = "measure"
        for measured_index, (
            pattern,
            visual,
            audio,
            concept_id,
        ) in enumerate(device_stream[warmup_steps:]):
            compile_context["tick_index"] = measured_index
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_started = time.perf_counter_ns()
            metrics = trainer.train_step(
                pattern,
                raw_window=f"grounded measure concept {concept_id}",
                visual_spikes=visual,
                audio_spikes=audio,
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
            visual_accepts += int(bool(metrics["cross_modal_visual_accepted"]))
            audio_accepts += int(bool(metrics["cross_modal_audio_accepted"]))
            text_spike = F.normalize(pattern.unsqueeze(0), dim=1).squeeze(0)
            visual_cosines.append(
                _cosine(trainer.model.cross_modal.predict_visual(text_spike), visual)
            )
            audio_cosines.append(
                _cosine(trainer.model.cross_modal.predict_audio(text_spike), audio)
            )
        total_elapsed_s = (time.perf_counter_ns() - started) / 1e9

    predictive = trainer.model.predictive
    memory = trainer.model.memory_store.summary_stats(
        current_token=trainer.token_count,
        force=True,
    )
    cross_modal = trainer.model.cross_modal
    sample_count = len(winners)
    arm = {
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
            "max": max(latencies_ms),
            "max_index": latencies_ms.index(max(latencies_ms)),
            "over_1000ms_count": sum(value >= 1000.0 for value in latencies_ms),
        },
        "triton_compilation": {
            "event_count": len(triton_compile_events),
            "total_duration_s": sum(
                float(event["duration_s"]) for event in triton_compile_events
            ),
            "events": triton_compile_events,
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
            "cross_modal_layer_configured": True,
            "quality_evidence_available": True,
            "evidence_kind": "synthetic_correlated_visual_audio_spikes",
            "visual_prediction_cosine_mean": statistics.fmean(visual_cosines),
            "visual_prediction_cosine_p05": _percentile(visual_cosines, 0.05),
            "audio_prediction_cosine_mean": statistics.fmean(audio_cosines),
            "audio_prediction_cosine_p05": _percentile(audio_cosines, 0.05),
            "visual_acceptance_rate": visual_accepts / max(1, sample_count),
            "audio_acceptance_rate": audio_accepts / max(1, sample_count),
            "visual_confidence_mean": float(
                cross_modal.visual_confidence.mean().item()
            ),
            "audio_confidence_mean": float(
                cross_modal.audio_confidence.mean().item()
            ),
            "device_report": cross_modal.device_report(),
        },
        "column_transition_runtime": (
            trainer.column_transition_runtime_report()
        ),
        "cuda_memory": (
            {
                "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
            }
            if device.type == "cuda"
            else {}
        ),
    }
    grounding_state = {
        name: tensor.detach().cpu()
        for name, tensor in {
            "W_tv": cross_modal.W_tv,
            "W_vt": cross_modal.W_vt,
            "W_ta": cross_modal.W_ta,
            "W_at": cross_modal.W_at,
            "visual_confidence": cross_modal.visual_confidence,
            "audio_confidence": cross_modal.audio_confidence,
        }.items()
    }
    return arm, grounding_state


def compare_grounded_arms(
    baseline: Mapping[str, Any],
    variant: Mapping[str, Any],
    baseline_state: Mapping[str, torch.Tensor],
    variant_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    comparison = compare_quality_arms(baseline, variant)
    baseline_grounding = baseline["grounding"]
    variant_grounding = variant["grounding"]
    tensor_max_abs = {
        name: float(
            (
                variant_state[name].float() - baseline_state[name].float()
            ).abs().max().item()
        )
        for name in baseline_state
    }
    grounding_gates = {
        "visual_prediction_preserved": float(
            variant_grounding["visual_prediction_cosine_mean"]
        )
        >= float(baseline_grounding["visual_prediction_cosine_mean"]) - 0.01,
        "audio_prediction_preserved": float(
            variant_grounding["audio_prediction_cosine_mean"]
        )
        >= float(baseline_grounding["audio_prediction_cosine_mean"]) - 0.01,
        "visual_confidence_preserved": float(
            variant_grounding["visual_confidence_mean"]
        )
        >= float(baseline_grounding["visual_confidence_mean"]) - 0.01,
        "audio_confidence_preserved": float(
            variant_grounding["audio_confidence_mean"]
        )
        >= float(baseline_grounding["audio_confidence_mean"]) - 0.01,
        "visual_acceptance_preserved": float(
            variant_grounding["visual_acceptance_rate"]
        )
        >= float(baseline_grounding["visual_acceptance_rate"]),
        "audio_acceptance_preserved": float(
            variant_grounding["audio_acceptance_rate"]
        )
        >= float(baseline_grounding["audio_acceptance_rate"]),
        "cross_modal_state_exact": max(tensor_max_abs.values(), default=0.0) == 0.0,
    }
    comparison["grounding_gates"] = grounding_gates
    comparison["grounding_state_max_abs_difference"] = tensor_max_abs
    comparison["grounding_quality_preserved"] = all(grounding_gates.values())
    comparison["promotion_eligible"] = bool(
        comparison["quality_preserved"]
        and comparison["grounding_quality_preserved"]
    )
    return comparison


def run_inplace_grounded_quality_benchmark(
    *,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    n_columns: int = 1024,
    training_samples: int = 256,
    samples: int = 128,
    warmup_steps: int = 8,
    concepts: int = 8,
    seed: int = 20260621,
    prepare_checkpoint: bool = True,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path).resolve()
    if prepare_checkpoint:
        prepare_grounded_checkpoint(
            checkpoint,
            n_columns=n_columns,
            training_samples=training_samples,
            concepts=concepts,
            seed=seed,
        )
    elif not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    trainer, _ = load_trainer_checkpoint(checkpoint)
    stream = _make_grounded_stream(
        input_dim=trainer.config.input_dim,
        visual_dim=trainer.config.cross_modal_dim_visual,
        audio_dim=trainer.config.cross_modal_dim_audio,
        samples=samples + warmup_steps,
        concepts=concepts,
        seed=seed + 1,
    )
    del trainer

    baseline, baseline_state = _run_grounded_arm(
        checkpoint,
        executor="runtime",
        stream=stream,
        warmup_steps=warmup_steps,
    )
    torch.cuda.empty_cache()
    variant, variant_state = _run_grounded_arm(
        checkpoint,
        executor="inplace_triton_runtime",
        stream=stream,
        warmup_steps=warmup_steps,
    )
    comparison = compare_grounded_arms(
        baseline,
        variant,
        baseline_state,
        variant_state,
    )
    report = {
        "schema_version": 1,
        "artifact_kind": "marulho_inplace_grounded_quality_benchmark",
        "surface": "inplace_grounded_quality_benchmark.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "production_executor_grounded_quality_supported"
            if comparison["promotion_eligible"]
            else "blocked_grounded_quality_regression"
        ),
        "passed": bool(comparison["promotion_eligible"]),
        "checkpoint": str(checkpoint),
        "seed": seed,
        "n_columns": n_columns,
        "concepts": concepts,
        "training_samples": training_samples,
        "samples": samples,
        "warmup_steps": warmup_steps,
        "baseline": baseline,
        "variant": variant,
        "comparison": comparison,
        "promotion_gate": {
            "runtime_default_changed": False,
            "checkpoint_opt_in_required": True,
            "compile_only_startup_warmup_available": True,
            "production_executor_available": bool(
                comparison["promotion_eligible"]
            ),
        },
        "mutates_live_runtime": False,
        "writes_evaluation_checkpoint": True,
        "claim_boundary": (
            "synthetic correlated visual/audio spike associations on isolated "
            "checkpoint clones; proves cross-modal path preservation but not "
            "camera, microphone, or real-world semantic grounding"
        ),
    }
    if output_path is not None:
        write_json_report_with_readme(
            output_path,
            report,
            title="In-Place CUDA Grounded Quality Benchmark",
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-columns", type=int, default=1024)
    parser.add_argument("--training-samples", type=int, default=256)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--concepts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--reuse-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    report = run_inplace_grounded_quality_benchmark(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        n_columns=args.n_columns,
        training_samples=args.training_samples,
        samples=args.samples,
        warmup_steps=args.warmup_steps,
        concepts=args.concepts,
        seed=args.seed,
        prepare_checkpoint=not args.reuse_checkpoint,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
