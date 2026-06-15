"""Benchmark the fixed-shape dense predictive-column state transition."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time
from typing import Callable

import torch

from marulho.core.predictive_columns import (
    PredictiveColumnState,
    dense_predictive_transition,
)
from marulho.core.inplace_column_cuda import candidate_predictive_writeback_cuda
from marulho.training.checkpointing import load_trainer_checkpoint


TransitionFn = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def _ensure_windows_triton_compiler() -> str | None:
    if os.environ.get("CC"):
        return os.environ["CC"]
    if os.name != "nt":
        return None
    try:
        import triton  # type: ignore[import-not-found]
    except Exception:
        return None
    tcc = Path(triton.__file__).parent / "runtime" / "tcc" / "tcc.exe"
    if tcc.exists():
        os.environ["CC"] = str(tcc)
        return str(tcc)
    return None


def _measure(
    name: str,
    fn: TransitionFn,
    tensors: tuple[torch.Tensor, ...],
    *,
    iterations: int,
) -> dict[str, object]:
    device = tensors[0].device
    timings_ms: list[float] = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        outputs = fn(*tensors)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings_ms.append((time.perf_counter_ns() - started) / 1e6)
        if len(outputs) != 6:
            raise RuntimeError(f"{name} returned an invalid transition")
    elapsed_s = sum(timings_ms) / 1000.0
    return {
        "name": name,
        "iterations": int(iterations),
        "tokens_per_second": float(iterations / max(elapsed_s, 1e-9)),
        "latency_ms": {
            "median": statistics.median(timings_ms),
            "p95": _percentile(timings_ms, 0.95),
            "mean": statistics.mean(timings_ms),
            "min": min(timings_ms),
            "max": max(timings_ms),
        },
    }


def _clone_predictive_state(source: PredictiveColumnState) -> PredictiveColumnState:
    state = PredictiveColumnState(
        source.n_columns,
        location_dim=source.location_dim,
        device=source.device,
    )
    state.location = source.location.detach().clone()
    state.velocity = source.velocity.detach().clone()
    state._prediction_weights = source._prediction_weights.detach().clone()
    state.prediction_error = source.prediction_error.detach().clone()
    state.prediction_failure_streak = source.prediction_failure_streak.detach().clone()
    state.confidence = source.confidence.detach().clone()
    state._error_ema_alpha = source._error_ema_alpha
    state._failure_streak_threshold = source._failure_streak_threshold
    state.predictive_step_count = 0
    state.predictive_last_update_step.zero_()
    state._predictive_has_cached_columns = False
    state._last_predictive_completed_candidates = None
    state._last_predictive_completed_step = 0
    return state


def _measure_writeback(
    name: str,
    step_fn: Callable[[int], None],
    *,
    start_index: int,
    iterations: int,
    device: torch.device,
    updated_column_count: int,
    total_columns: int,
    fallback_reason: str | None,
) -> dict[str, object]:
    timings_ms: list[float] = []
    for offset in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        step_fn(start_index + offset)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings_ms.append((time.perf_counter_ns() - started) / 1e6)
    elapsed_s = sum(timings_ms) / 1000.0
    cached = max(0, int(total_columns) - int(updated_column_count))
    return {
        "name": name,
        "iterations": int(iterations),
        "tokens_per_second": float(iterations / max(elapsed_s, 1e-9)),
        "latency_ms": {
            "median": statistics.median(timings_ms),
            "p95": _percentile(timings_ms, 0.95),
            "mean": statistics.mean(timings_ms),
            "min": min(timings_ms),
            "max": max(timings_ms),
        },
        "updated_column_count": int(updated_column_count),
        "cached_state_count": int(cached),
        "runs_all_columns": int(updated_column_count) >= int(total_columns),
        "fallback_reason": fallback_reason,
    }


def _candidate_row_parity(
    dense_state: PredictiveColumnState,
    scoped_state: PredictiveColumnState,
    candidates: torch.Tensor,
) -> dict[str, object]:
    float_pairs = {
        "location": (dense_state.location, scoped_state.location),
        "velocity": (dense_state.velocity, scoped_state.velocity),
        "prediction_weights": (
            dense_state._prediction_weights,
            scoped_state._prediction_weights,
        ),
        "prediction_error": (dense_state.prediction_error, scoped_state.prediction_error),
        "confidence": (dense_state.confidence, scoped_state.confidence),
    }
    max_abs: dict[str, float] = {}
    for name, (left, right) in float_pairs.items():
        delta = (
            left.index_select(0, candidates)
            - right.index_select(0, candidates)
        ).abs()
        max_abs[name] = float(delta.max().item()) if int(delta.numel()) else 0.0
    streak_equal = bool(
        torch.equal(
            dense_state.prediction_failure_streak.index_select(0, candidates),
            scoped_state.prediction_failure_streak.index_select(0, candidates),
        )
    )
    max_float = max(max_abs.values(), default=0.0)
    return {
        "candidate_rows_match_dense": bool(streak_equal and max_float <= 1e-6),
        "max_abs": max_abs,
        "prediction_failure_streak_equal": streak_equal,
    }


def _run_writeback_experiment(
    predictive: PredictiveColumnState,
    *,
    routing_keys: torch.Tensor,
    previous_routing_key: torch.Tensor,
    candidate_count: int,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, object]:
    device = predictive.device
    total_steps = int(iterations) + int(warmup_iterations)
    k = min(max(1, int(candidate_count)), int(predictive.n_columns))
    candidates = torch.arange(k, device=device, dtype=torch.long)
    winner_sequence = candidates.index_select(
        0,
        torch.remainder(
            torch.arange(total_steps, device=device, dtype=torch.long),
            k,
        ),
    )
    winner_ids = [int(value) for value in candidates.detach().cpu().tolist()]
    dense_state = _clone_predictive_state(predictive)
    scoped_state = _clone_predictive_state(predictive)
    triton_state = _clone_predictive_state(predictive)

    def previous_for(index: int) -> torch.Tensor:
        return previous_routing_key if index <= 0 else routing_keys[index - 1]

    def winner_for(index: int) -> int:
        return winner_ids[index % k]

    def dense_step(index: int) -> None:
        dense_state.apply_dense_transition(
            winner_sequence[index : index + 1],
            routing_keys[index],
            previous_for(index),
            learning_rate=0.005,
            transition_mode="fused_eager",
        )

    def scoped_step(index: int) -> None:
        scoped_state.update_candidate_prediction_transition(
            [winner_for(index)],
            routing_keys[index],
            previous_for(index),
            learning_rate=0.005,
            candidate_indices=candidates,
        )

    def triton_step(index: int) -> None:
        candidate_predictive_writeback_cuda(
            location=triton_state.location,
            location_velocity=triton_state.velocity,
            prediction_weights=triton_state._prediction_weights,
            prediction_error=triton_state.prediction_error,
            prediction_failure_streak=triton_state.prediction_failure_streak,
            confidence=triton_state.confidence,
            routing_key=routing_keys[index],
            previous_routing_key=previous_for(index),
            winners=winner_sequence[index : index + 1],
            candidates=candidates,
            has_previous_routing_key=True,
            prediction_error_ema_alpha=triton_state._error_ema_alpha,
            prediction_failure_streak_threshold=(
                triton_state._failure_streak_threshold
            ),
            prediction_learning_rate=0.005,
        )

    triton_error: str | None = None
    triton_available = bool(device.type == "cuda")
    for step in range(int(warmup_iterations)):
        dense_step(step)
        scoped_step(step)
        if triton_available:
            try:
                triton_step(step)
            except Exception as exc:  # pragma: no cover - backend dependent
                triton_error = repr(exc)
                triton_available = False
    if device.type == "cuda":
        torch.cuda.synchronize()

    dense_arm = _measure_writeback(
        "dense_all_columns_writeback",
        dense_step,
        start_index=int(warmup_iterations),
        iterations=int(iterations),
        device=device,
        updated_column_count=int(predictive.n_columns),
        total_columns=int(predictive.n_columns),
        fallback_reason=None,
    )
    scoped_arm = _measure_writeback(
        "candidate_scoped_eager_writeback",
        scoped_step,
        start_index=int(warmup_iterations),
        iterations=int(iterations),
        device=device,
        updated_column_count=k,
        total_columns=int(predictive.n_columns),
        fallback_reason=(
            "launch_bound_candidate_indexing_experiment_not_promoted"
            if device.type == "cuda"
            else None
        ),
    )
    arms = [dense_arm, scoped_arm]
    parity_by_arm = {
        "candidate_scoped_eager_writeback": _candidate_row_parity(
            dense_state,
            scoped_state,
            candidates,
        )
    }
    if triton_available:
        try:
            triton_arm = _measure_writeback(
                "candidate_scoped_triton_writeback",
                triton_step,
                start_index=int(warmup_iterations),
                iterations=int(iterations),
                device=device,
                updated_column_count=k,
                total_columns=int(predictive.n_columns),
                fallback_reason=None,
            )
            arms.append(triton_arm)
            parity_by_arm["candidate_scoped_triton_writeback"] = (
                _candidate_row_parity(dense_state, triton_state, candidates)
            )
        except Exception as exc:  # pragma: no cover - backend dependent
            triton_error = repr(exc)
    dense_mean = float(dense_arm["latency_ms"]["mean"])
    candidate_arms = [arm for arm in arms if not bool(arm["runs_all_columns"])]
    best_candidate_arm = min(
        candidate_arms,
        key=lambda arm: float(arm["latency_ms"]["mean"]),
    )
    best_candidate_name = str(best_candidate_arm["name"])
    best_candidate_mean = float(best_candidate_arm["latency_ms"]["mean"])
    best_candidate_matches = bool(
        parity_by_arm.get(best_candidate_name, {}).get(
            "candidate_rows_match_dense",
            False,
        )
    )
    candidate_rows_match_dense = all(
        bool(parity["candidate_rows_match_dense"])
        for parity in parity_by_arm.values()
    )
    promote_candidate = bool(
        best_candidate_mean <= dense_mean * 1.02 and best_candidate_matches
    )
    return {
        "scope": "isolated_predictive_state_writeback_candidate_scope_experiment",
        "claim_boundary": (
            "compares dense all-column predictive writeback with eager "
            "candidate-indexed writeback on the same tensor device; excludes "
            "routing, competition, CUDA graph capture, service throughput, and "
            "checkpoint growth or pruning"
        ),
        "candidate_count": int(k),
        "total_columns": int(predictive.n_columns),
        "candidate_fraction": round(float(k) / float(max(1, predictive.n_columns)), 6),
        "arms": arms,
        "candidate_rows_match_dense": bool(candidate_rows_match_dense),
        "candidate_row_parity": parity_by_arm["candidate_scoped_eager_writeback"],
        "candidate_row_parity_by_arm": parity_by_arm,
        "best_candidate_arm": best_candidate_arm,
        "scoped_neutral_or_better": bool(promote_candidate),
        "promotion_decision": (
            "promote_candidate_scoped_writeback_for_further_runtime_testing"
            if promote_candidate
            else "retain_dense_cuda_predictive_update"
        ),
        "fallback_reason": (
            None
            if promote_candidate
            else "candidate_scoped_predictive_writeback_not_neutral_or_better"
        ),
        "triton_error": triton_error,
    }


def run_predictive_transition_benchmark(
    checkpoint: Path,
    *,
    iterations: int = 512,
    warmup_iterations: int = 16,
    seed: int = 20260611,
    candidate_count: int = 10,
) -> dict[str, object]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")

    trainer, _ = load_trainer_checkpoint(checkpoint)
    predictive = trainer.model.predictive
    device = trainer.model.device
    generator = torch.Generator(device=device).manual_seed(seed)
    routing_key = torch.randn(
        trainer.config.column_latent_dim,
        generator=generator,
        device=device,
    )
    previous_routing_key = torch.randn(
        trainer.config.column_latent_dim,
        generator=generator,
        device=device,
    )
    routing_keys = torch.randn(
        iterations + warmup_iterations,
        trainer.config.column_latent_dim,
        generator=generator,
        device=device,
    )
    winners = torch.tensor([7 % trainer.config.n_columns], dtype=torch.long, device=device)
    tensors = (
        predictive.location.detach(),
        predictive.velocity.detach(),
        predictive._prediction_weights.detach(),
        predictive.prediction_error.detach(),
        predictive.prediction_failure_streak.detach(),
        predictive.confidence.detach(),
        routing_key,
        previous_routing_key,
        winners,
    )

    def transition(*args: torch.Tensor):
        return dense_predictive_transition(
            *args,
            has_previous_routing_key=True,
            error_ema_alpha=predictive._error_ema_alpha,
            failure_streak_threshold=predictive._failure_streak_threshold,
            learning_rate=0.005,
        )

    for _ in range(warmup_iterations):
        transition(*tensors)
    if device.type == "cuda":
        torch.cuda.synchronize()

    compiler_path = _ensure_windows_triton_compiler()
    arms = [_measure("eager", transition, tensors, iterations=iterations)]
    compile_errors: list[dict[str, str]] = []
    if device.type == "cuda":
        for mode in ("default", "reduce-overhead"):
            try:
                compiled = torch.compile(transition, mode=mode, fullgraph=True)
                for _ in range(warmup_iterations):
                    compiled(*tensors)
                torch.cuda.synchronize()
                arms.append(
                    _measure(
                        f"torch_compile_{mode}",
                        compiled,
                        tensors,
                        iterations=iterations,
                    )
                )
            except Exception as exc:  # pragma: no cover - backend dependent
                compile_errors.append({"mode": mode, "error": repr(exc)})

    eager_outputs = transition(*tensors)
    writeback_experiment = _run_writeback_experiment(
        predictive,
        routing_keys=routing_keys,
        previous_routing_key=previous_routing_key,
        candidate_count=candidate_count,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
    )
    best = max(arms, key=lambda arm: float(arm["tokens_per_second"]))
    cuda_memory: dict[str, float] = {}
    if device.type == "cuda":
        cuda_memory = {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        }

    return {
        "surface": "predictive_transition_benchmark.v1",
        "checkpoint": str(checkpoint),
        "scope": "isolated_dense_predictive_state_transition_no_runtime_writeback",
        "claim_boundary": (
            "measures fixed-shape dense predictive error, location, confidence, "
            "failure-streak, and prediction-weight transition; excludes routing, "
            "competition, plasticity, memory, binding, context, and service throughput"
        ),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "n_columns": int(trainer.config.n_columns),
        "location_dim": int(predictive.location_dim),
        "iterations": int(iterations),
        "warmup_iterations": int(warmup_iterations),
        "candidate_count": int(min(max(1, candidate_count), predictive.n_columns)),
        "output_shapes": [list(output.shape) for output in eager_outputs],
        "arms": arms,
        "best_arm": best,
        "writeback_experiment": writeback_experiment,
        "compile_errors": compile_errors,
        "compile_environment": {"cc": compiler_path},
        "cuda_memory": cuda_memory,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=512)
    parser.add_argument("--warmup-iterations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--candidate-count", type=int, default=10)
    args = parser.parse_args()
    report = run_predictive_transition_benchmark(
        args.checkpoint,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        seed=args.seed,
        candidate_count=args.candidate_count,
    )
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
