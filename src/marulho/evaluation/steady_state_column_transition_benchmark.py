"""Benchmark the rejected functional steady-state column transition.

This runner is an oracle and promotion gate for a future in-place CUDA/Triton
implementation. The functional transition is intentionally not imported by the
always-on trainer because full configured hot-window A/B did not improve.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

from marulho.core.column_transition import steady_state_column_transition
from marulho.training.checkpointing import load_trainer_checkpoint


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
    compiler = Path(triton.__file__).parent / "runtime" / "tcc" / "tcc.exe"
    if compiler.exists():
        os.environ["CC"] = str(compiler)
        return str(compiler)
    return None


def run_steady_state_column_transition_benchmark(
    checkpoint: Path,
    *,
    iterations: int = 256,
    warmup_iterations: int = 16,
    seed: int = 20260616,
) -> dict[str, object]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")

    trainer, _ = load_trainer_checkpoint(checkpoint)
    comp = trainer.model.competitive
    predictive = trainer.model.predictive
    device = trainer.model.device
    generator = torch.Generator(device=device).manual_seed(seed)
    routing_key = torch.nn.functional.normalize(
        torch.rand(
            trainer.config.column_latent_dim,
            generator=generator,
            device=device,
        ),
        dim=0,
    )
    previous_routing_key = torch.nn.functional.normalize(
        torch.rand(
            trainer.config.column_latent_dim,
            generator=generator,
            device=device,
        ),
        dim=0,
    )
    candidates = trainer._routing_candidates(routing_key)
    if candidates is None:
        raise RuntimeError("checkpoint routing index returned no candidates")
    context_gain = torch.ones(trainer.config.n_columns, device=device)
    consolidation = trainer.model.memory_store.bucket_consolidation_tensor(
        comp.n_columns,
        device=device,
    ).index_select(0, candidates)
    fixed_inputs = (
        routing_key,
        previous_routing_key,
        candidates,
        context_gain,
        consolidation,
        torch.tensor(0.25, device=device),
        torch.tensor(trainer.model.surprise.dopamine, device=device),
        torch.tensor(trainer.model.surprise.serotonin, device=device),
        torch.tensor(comp.get_lr(), device=device),
    )
    initial_state = (
        comp.prototypes.detach().clone(),
        comp.prototype_velocity.detach().clone(),
        comp.thresholds.detach().clone(),
        comp.win_rate_ema.detach().clone(),
        comp.steps_since_win.detach().clone(),
        predictive.location.detach().clone(),
        predictive.velocity.detach().clone(),
        predictive._prediction_weights.detach().clone(),
        predictive.prediction_error.detach().clone(),
        predictive.prediction_failure_streak.detach().clone(),
        predictive.confidence.detach().clone(),
    )

    def transition(*state: torch.Tensor):
        return steady_state_column_transition(
            *state,
            *fixed_inputs,
            prototype_momentum=comp.prototype_momentum,
            homeostasis_beta=comp.homeostasis_beta,
            homeostasis_lr=comp.homeostasis_lr,
            target_firing_rate=comp.target_firing_rate,
            threshold_min=comp.threshold_min,
            threshold_max=comp.threshold_max,
            candidate_scoped_homeostasis=False,
            prediction_error_ema_alpha=predictive._error_ema_alpha,
            prediction_failure_streak_threshold=predictive._failure_streak_threshold,
            prediction_learning_rate=0.005,
        )

    def measure(name: str, fn) -> dict[str, object]:
        state = tuple(value.clone() for value in initial_state)
        for _ in range(warmup_iterations):
            outputs = fn(*state)
            for target, source in zip(state, outputs[3:8] + outputs[9:15]):
                target.copy_(source)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings: list[float] = []
        for _ in range(iterations):
            if name.startswith("torch_compile"):
                torch.compiler.cudagraph_mark_step_begin()
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter_ns()
            outputs = fn(*state)
            for target, source in zip(state, outputs[3:8] + outputs[9:15]):
                target.copy_(source)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter_ns() - started) / 1e6)
        elapsed_s = sum(timings) / 1000.0
        return {
            "name": name,
            "transitions_per_second": iterations / max(elapsed_s, 1e-9),
            "latency_ms": {
                "median": statistics.median(timings),
                "p95": _percentile(timings, 0.95),
                "mean": statistics.mean(timings),
            },
        }

    arms = [measure("functional_eager", transition)]
    compile_errors: list[str] = []
    compiler = _ensure_windows_triton_compiler()
    if device.type == "cuda":
        try:
            compiled = torch.compile(
                transition,
                mode="reduce-overhead",
                fullgraph=True,
            )
            arms.append(measure("torch_compile_reduce_overhead", compiled))
        except Exception as exc:  # pragma: no cover - backend dependent
            compile_errors.append(repr(exc))

    best = max(arms, key=lambda arm: float(arm["transitions_per_second"]))
    cuda_memory: dict[str, float] = {}
    if device.type == "cuda":
        cuda_memory = {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        }
    return {
        "surface": "steady_state_column_transition_benchmark.v1",
        "checkpoint": str(checkpoint),
        "scope": (
            "evaluation_only_functional_competition_prediction_prototype_"
            "homeostasis_transition_with_stable_state_writeback"
        ),
        "promotion_status": "rejected_for_always_on_runtime",
        "claim_boundary": (
            "isolated transition can be fast, but full configured hot-window A/B "
            "did not improve; future promotion requires in-place device mutation "
            "without full-state functional outputs and must beat live train_step"
        ),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "n_columns": int(comp.n_columns),
        "candidate_count": int(candidates.numel()),
        "state_tensor_count": len(initial_state),
        "iterations": int(iterations),
        "warmup_iterations": int(warmup_iterations),
        "arms": arms,
        "best_arm": best,
        "compile_errors": compile_errors,
        "compiler": compiler,
        "cuda_memory": cuda_memory,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--warmup-iterations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260616)
    args = parser.parse_args()
    report = run_steady_state_column_transition_benchmark(
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
