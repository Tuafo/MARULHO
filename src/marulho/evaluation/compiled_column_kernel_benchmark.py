"""Benchmark a pure tensor column-competition kernel.

This is an isolated acceleration probe. It does not alter the live trainer,
routing index, plasticity state, checkpoint state, or Runtime Truth verdicts.
The goal is to measure whether MARULHO's candidate-scoped competition math can
reach production token rates once separated from Python control flow.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time
from typing import Callable

import torch
import torch.nn.functional as F

from marulho.training.checkpointing import load_trainer_checkpoint


KernelFn = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        float,
    ],
    tuple[torch.Tensor, torch.Tensor],
]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def column_competition_kernel(
    routing_keys: torch.Tensor,
    candidate_indices: torch.Tensor,
    prototypes: torch.Tensor,
    input_weights: torch.Tensor,
    thresholds: torch.Tensor,
    input_patterns: torch.Tensor,
    context_gain: torch.Tensor | None,
    input_weight_blend: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run candidate-scoped WTA competition over a fixed batch.

    Inputs are fixed-shape tensors:
    - routing_keys: [batch, column_dim]
    - candidate_indices: [batch, k]
    - prototypes: [n_columns, column_dim]
    - input_weights: [n_columns, input_dim]
    - thresholds: [n_columns]
    - input_patterns: [batch, input_dim]
    - context_gain: optional [batch, n_columns]

    It intentionally returns only winner ids and strengths. Prototype updates,
    STDP, homeostasis, memory, binding, cross-modal grounding, and checkpointing
    remain outside this benchmark.
    """

    x = F.normalize(torch.clamp(routing_keys.float(), min=0.0), dim=1)
    candidates = candidate_indices.long()
    candidate_prototypes = prototypes[candidates]
    similarity = (candidate_prototypes * x.unsqueeze(1)).sum(dim=2)

    normalized_inputs = torch.clamp(input_patterns.float(), min=0.0)
    normalized_inputs = normalized_inputs / normalized_inputs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    candidate_weights = input_weights[candidates]
    raw_drive = (candidate_weights * normalized_inputs.unsqueeze(1)).sum(dim=2)
    drive = raw_drive / raw_drive.max(dim=1, keepdim=True).values.clamp(min=1e-8)

    blend = float(input_weight_blend)
    combined = (1.0 - blend) * similarity + blend * drive
    if context_gain is not None:
        gathered_gain = torch.gather(context_gain, 1, candidates)
        combined = combined * torch.clamp(gathered_gain, min=0.5, max=1.5)

    inhibition = thresholds[candidates]
    activation = torch.relu(combined - inhibition)
    winner_local = torch.argmax(activation, dim=1)
    winner_ids = torch.gather(candidates, 1, winner_local.unsqueeze(1)).squeeze(1)
    winner_values = torch.gather(activation, 1, winner_local.unsqueeze(1)).squeeze(1)
    strengths = torch.where(
        winner_values > 0.0,
        winner_values / winner_values.clamp(min=1e-8),
        torch.ones_like(winner_values),
    )
    return winner_ids, strengths


def _measure_kernel(
    name: str,
    fn: KernelFn,
    *,
    routing_keys: torch.Tensor,
    candidate_indices: torch.Tensor,
    prototypes: torch.Tensor,
    input_weights: torch.Tensor,
    thresholds: torch.Tensor,
    input_patterns: torch.Tensor,
    context_gain: torch.Tensor | None,
    input_weight_blend: float,
    iterations: int,
) -> dict[str, object]:
    device = routing_keys.device
    timings_ms: list[float] = []
    total_tokens = 0
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        winners, strengths = fn(
            routing_keys,
            candidate_indices,
            prototypes,
            input_weights,
            thresholds,
            input_patterns,
            context_gain,
            input_weight_blend,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings_ms.append((time.perf_counter_ns() - started) / 1e6)
        total_tokens += int(winners.numel())
        if int(strengths.numel()) != int(winners.numel()):
            raise RuntimeError(f"{name} returned mismatched winner/strength shapes")

    total_s = sum(timings_ms) / 1000.0
    return {
        "name": name,
        "iterations": int(iterations),
        "tokens": int(total_tokens),
        "tokens_per_second": float(total_tokens / max(total_s, 1e-9)),
        "latency_ms": {
            "median": statistics.median(timings_ms),
            "p95": _percentile(timings_ms, 0.95),
            "mean": statistics.mean(timings_ms),
            "min": min(timings_ms),
            "max": max(timings_ms),
        },
    }


def _compile_kernel(mode: str) -> KernelFn:
    compiled = torch.compile(column_competition_kernel, mode=mode, fullgraph=True)
    return compiled


def _ensure_windows_triton_compiler() -> str | None:
    """Use triton-windows' bundled TinyCC when no compiler is configured.

    PyTorch Inductor/Triton needs a C compiler even for this isolated CUDA
    benchmark. On Windows, the `triton-windows` wheel commonly ships a TinyCC
    executable under `triton/runtime/tcc/tcc.exe`; using it keeps this benchmark
    local and avoids requiring a system-wide Visual Studio Build Tools install.
    """

    if os.environ.get("CC"):
        return os.environ["CC"]
    if os.name != "nt":
        return None
    try:
        import triton  # type: ignore[import-not-found]
    except Exception:
        return None

    triton_root = Path(triton.__file__).parent
    tcc = triton_root / "runtime" / "tcc" / "tcc.exe"
    if tcc.exists():
        os.environ["CC"] = str(tcc)
        return str(tcc)
    return None


def run_compiled_column_kernel_benchmark(
    checkpoint: Path,
    *,
    batch_size: int = 256,
    iterations: int = 128,
    warmup_iterations: int = 8,
    seed: int = 20260611,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")

    trainer, _ = load_trainer_checkpoint(checkpoint)
    device = trainer.model.device
    comp = trainer.model.competitive
    if trainer.config.k_routing <= 0:
        raise ValueError("k_routing must be positive")

    generator = torch.Generator(device=device).manual_seed(seed)
    routing_keys = torch.rand(
        batch_size,
        trainer.config.column_latent_dim,
        generator=generator,
        device=device,
    )
    input_patterns = torch.rand(
        batch_size,
        trainer.config.input_dim,
        generator=generator,
        device=device,
    )
    candidate_indices = torch.randint(
        low=0,
        high=trainer.config.n_columns,
        size=(batch_size, trainer.config.k_routing),
        generator=generator,
        device=device,
    )
    context_gain = torch.ones(batch_size, trainer.config.n_columns, device=device)

    prototypes = comp.prototypes.detach()
    input_weights = comp.input_weights.detach()
    thresholds = comp.thresholds.detach()
    input_weight_blend = float(comp.input_weight_blend)

    for _ in range(warmup_iterations):
        column_competition_kernel(
            routing_keys,
            candidate_indices,
            prototypes,
            input_weights,
            thresholds,
            input_patterns,
            context_gain,
            input_weight_blend,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()

    compiler_path = _ensure_windows_triton_compiler()
    arms: list[dict[str, object]] = [
        _measure_kernel(
            "eager_batch",
            column_competition_kernel,
            routing_keys=routing_keys,
            candidate_indices=candidate_indices,
            prototypes=prototypes,
            input_weights=input_weights,
            thresholds=thresholds,
            input_patterns=input_patterns,
            context_gain=context_gain,
            input_weight_blend=input_weight_blend,
            iterations=iterations,
        )
    ]

    compile_results: list[dict[str, object]] = []
    compile_errors: list[dict[str, str]] = []
    compile_modes = ("default", "reduce-overhead") if device.type == "cuda" else ()
    for mode in compile_modes:
        try:
            fn = _compile_kernel(mode)
            for _ in range(warmup_iterations):
                fn(
                    routing_keys,
                    candidate_indices,
                    prototypes,
                    input_weights,
                    thresholds,
                    input_patterns,
                    context_gain,
                    input_weight_blend,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            compile_results.append(
                _measure_kernel(
                    f"torch_compile_{mode}",
                    fn,
                    routing_keys=routing_keys,
                    candidate_indices=candidate_indices,
                    prototypes=prototypes,
                    input_weights=input_weights,
                    thresholds=thresholds,
                    input_patterns=input_patterns,
                    context_gain=context_gain,
                    input_weight_blend=input_weight_blend,
                    iterations=iterations,
                )
            )
        except Exception as exc:  # pragma: no cover - backend dependent
            compile_errors.append({"mode": mode, "error": repr(exc)})
    arms.extend(compile_results)

    best = max(arms, key=lambda arm: float(arm["tokens_per_second"]))
    cuda_memory: dict[str, float] = {}
    if device.type == "cuda":
        cuda_memory = {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        }

    return {
        "surface": "compiled_column_kernel_benchmark.v1",
        "checkpoint": str(checkpoint),
        "scope": "isolated_fixed_shape_candidate_competition_no_runtime_mutation",
        "claim_boundary": (
            "measures a pure tensor candidate-competition kernel; "
            "does not include retrieval, trainer orchestration, plasticity, "
            "memory, binding, cross-modal grounding, checkpointing, or service throughput"
        ),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": int(batch_size),
        "iterations": int(iterations),
        "warmup_iterations": int(warmup_iterations),
        "n_columns": int(trainer.config.n_columns),
        "input_dim": int(trainer.config.input_dim),
        "column_latent_dim": int(trainer.config.column_latent_dim),
        "k_routing": int(trainer.config.k_routing),
        "arms": arms,
        "best_arm": best,
        "compile_errors": compile_errors,
        "compile_environment": {
            "cc": compiler_path,
            "cc_source": (
                "environment_or_triton_windows_tcc"
                if compiler_path is not None
                else "not_configured"
            ),
        },
        "throughput_reference": {
            "reference_floor_tokens_per_second": 1000.0,
            "reference_floor_met": float(best["tokens_per_second"]) >= 1000.0,
            "best_multiple_over_reference_floor": float(best["tokens_per_second"]) / 1000.0,
            "goal": "maximize_sustainable_local_throughput_not_stop_at_reference_floor",
        },
        "cuda_memory": cuda_memory,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--warmup-iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260611)
    args = parser.parse_args()

    report = run_compiled_column_kernel_benchmark(
        args.checkpoint,
        batch_size=args.batch_size,
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
