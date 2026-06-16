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

from marulho.core.predictive_columns import dense_predictive_transition
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


def run_predictive_transition_benchmark(
    checkpoint: Path,
    *,
    iterations: int = 512,
    warmup_iterations: int = 16,
    seed: int = 20260611,
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
        "output_shapes": [list(output.shape) for output in eager_outputs],
        "arms": arms,
        "best_arm": best,
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
    args = parser.parse_args()
    report = run_predictive_transition_benchmark(
        args.checkpoint,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        seed=args.seed,
    )
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
