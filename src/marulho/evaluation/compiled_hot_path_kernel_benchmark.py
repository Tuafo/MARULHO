"""Benchmark a wider pure tensor MARULHO hot-path kernel.

This benchmark isolates projection, candidate-scoped competition, and
candidate-scoped predictive-state math into a fixed-shape tensor block. It is a
promotion probe only: no trainer state, predictive state, memory, binding,
cross-modal state, replay, checkpoint, service endpoint, or Runtime Truth state
is mutated.
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


HotPathKernelFn = Callable[
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
        torch.Tensor,
        torch.Tensor,
        float,
    ],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def _normalize_input(input_patterns: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(input_patterns.float(), min=0.0)
    return x / x.sum(dim=1, keepdim=True).clamp(min=1e-8)


def _normalize_routing_keys(keys: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.clamp(keys.float(), min=0.0), dim=1)


def _project_routing_keys(input_patterns: torch.Tensor, w_project: torch.Tensor) -> torch.Tensor:
    normalized_inputs = _normalize_input(input_patterns)
    return _normalize_routing_keys(torch.matmul(normalized_inputs, w_project))


def _deterministic_fallback_candidates(
    batch_size: int,
    k_routing: int,
    n_columns: int,
    device: torch.device,
) -> torch.Tensor:
    row = torch.arange(k_routing, device=device, dtype=torch.long) % max(1, int(n_columns))
    return row.unsqueeze(0).expand(batch_size, k_routing).clone()


def _random_candidates(
    *,
    batch_size: int,
    k_routing: int,
    n_columns: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=n_columns,
        size=(batch_size, k_routing),
        generator=generator,
        device=device,
    )


def _routing_index_candidates(
    *,
    trainer,
    input_patterns: torch.Tensor,
    w_project: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, object]]:
    device = input_patterns.device
    batch_size = int(input_patterns.shape[0])
    k_routing = int(trainer.config.k_routing)
    n_columns = int(trainer.config.n_columns)

    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter_ns()
    routing_keys = _project_routing_keys(input_patterns, w_project)
    ids, distances = trainer.model.hnsw_index.search(routing_keys, k=k_routing)
    fallback_rows = 0
    candidate_rows: list[list[int]] = []
    fallback = list(range(min(k_routing, n_columns)))
    while len(fallback) < k_routing:
        fallback.extend(fallback or [0])
    fallback = fallback[:k_routing]

    for row in ids:
        clean_row = [int(candidate_id) for candidate_id in row[:k_routing] if 0 <= int(candidate_id) < n_columns]
        if len(clean_row) < k_routing:
            fallback_rows += 1
            seen = set(clean_row)
            for candidate_id in fallback:
                if candidate_id not in seen:
                    clean_row.append(candidate_id)
                    seen.add(candidate_id)
                if len(clean_row) >= k_routing:
                    break
            while len(clean_row) < k_routing:
                clean_row.append(fallback[len(clean_row) % len(fallback)])
        candidate_rows.append(clean_row[:k_routing])

    if len(candidate_rows) != batch_size:
        fallback_rows = batch_size
        candidate_indices = _deterministic_fallback_candidates(batch_size, k_routing, n_columns, device)
    else:
        candidate_indices = torch.tensor(candidate_rows, dtype=torch.long, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    prep_latency_ms = (time.perf_counter_ns() - started) / 1e6
    finite_distances = distances[distances < float("inf")]
    distance_stats: dict[str, float | None] = {
        "mean": float(finite_distances.mean()) if finite_distances.size else None,
        "min": float(finite_distances.min()) if finite_distances.size else None,
        "max": float(finite_distances.max()) if finite_distances.size else None,
    }

    return candidate_indices, {
        "candidate_source": "routing_index",
        "candidate_prep_latency_ms": prep_latency_ms,
        "candidate_prep_tokens_per_second": float(batch_size / max(prep_latency_ms / 1000.0, 1e-9)),
        "fallback_rows": int(fallback_rows),
        "distance_stats": distance_stats,
        "routing_index_stats": trainer.model.hnsw_index.stats(),
    }


def _routing_index_tensor_candidates(
    *,
    trainer,
    input_patterns: torch.Tensor,
    w_project: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, object]]:
    device = input_patterns.device
    batch_size = int(input_patterns.shape[0])
    k_routing = int(trainer.config.k_routing)
    n_columns = int(trainer.config.n_columns)

    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter_ns()
    routing_keys = _project_routing_keys(input_patterns, w_project)
    ids, distances = trainer.model.hnsw_index.search_tensors(routing_keys, k=k_routing)

    fallback_rows = 0
    if ids.shape != (batch_size, k_routing):
        fallback_rows = batch_size
        candidate_indices = _deterministic_fallback_candidates(batch_size, k_routing, n_columns, device)
    else:
        invalid = (ids < 0) | (ids >= n_columns)
        fallback_rows = int(invalid.any(dim=1).sum().item())
        candidate_indices = torch.where(
            invalid,
            _deterministic_fallback_candidates(batch_size, k_routing, n_columns, device),
            ids.to(device=device, dtype=torch.long),
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    prep_latency_ms = (time.perf_counter_ns() - started) / 1e6

    if device.type == "cuda":
        torch.cuda.synchronize()
    warm_started = time.perf_counter_ns()
    warm_routing_keys = _project_routing_keys(input_patterns, w_project)
    warm_ids, _ = trainer.model.hnsw_index.search_tensors(warm_routing_keys, k=k_routing)
    if device.type == "cuda":
        torch.cuda.synchronize()
    warm_prep_latency_ms = (time.perf_counter_ns() - warm_started) / 1e6
    finite_distances = distances[torch.isfinite(distances)]
    distance_stats: dict[str, float | None] = {
        "mean": float(finite_distances.mean().item()) if finite_distances.numel() else None,
        "min": float(finite_distances.min().item()) if finite_distances.numel() else None,
        "max": float(finite_distances.max().item()) if finite_distances.numel() else None,
    }

    return candidate_indices, {
        "candidate_source": "routing_index_tensor",
        "candidate_prep_latency_ms": prep_latency_ms,
        "candidate_prep_tokens_per_second": float(batch_size / max(prep_latency_ms / 1000.0, 1e-9)),
        "candidate_prep_warm_latency_ms": warm_prep_latency_ms,
        "candidate_prep_warm_tokens_per_second": float(batch_size / max(warm_prep_latency_ms / 1000.0, 1e-9)),
        "candidate_prep_warm_shape": list(warm_ids.shape),
        "fallback_rows": int(fallback_rows),
        "distance_stats": distance_stats,
        "routing_index_stats": trainer.model.hnsw_index.stats(),
    }


def observe_route_predict_kernel(
    input_patterns: torch.Tensor,
    candidate_indices: torch.Tensor,
    w_project: torch.Tensor,
    prototypes: torch.Tensor,
    input_weights: torch.Tensor,
    thresholds: torch.Tensor,
    context_gain: torch.Tensor,
    prediction_location: torch.Tensor,
    prediction_weights: torch.Tensor,
    prediction_error: torch.Tensor,
    prediction_confidence: torch.Tensor,
    input_weight_blend: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run fixed-shape projection, competition, and predictive-state math.

    Returned predictive tensors are candidate-local next values. The function
    deliberately avoids writing those values back into runtime tensors.
    """

    normalized_inputs = _normalize_input(input_patterns)
    routing_keys = _normalize_routing_keys(torch.matmul(normalized_inputs, w_project))
    candidates = candidate_indices.long()

    candidate_prototypes = prototypes[candidates]
    similarity = (candidate_prototypes * routing_keys.unsqueeze(1)).sum(dim=2)
    candidate_weights = input_weights[candidates]
    raw_drive = (candidate_weights * normalized_inputs.unsqueeze(1)).sum(dim=2)
    drive = raw_drive / raw_drive.max(dim=1, keepdim=True).values.clamp(min=1e-8)

    blend = float(input_weight_blend)
    combined = (1.0 - blend) * similarity + blend * drive
    gathered_gain = torch.gather(context_gain, 1, candidates)
    combined = combined * torch.clamp(gathered_gain, min=0.5, max=1.5)

    activation = torch.relu(combined - thresholds[candidates])
    winner_local = torch.argmax(activation, dim=1)
    winner_ids = torch.gather(candidates, 1, winner_local.unsqueeze(1)).squeeze(1)
    winner_values = torch.gather(activation, 1, winner_local.unsqueeze(1)).squeeze(1)
    strengths = torch.where(
        winner_values > 0.0,
        winner_values / winner_values.clamp(min=1e-8),
        torch.ones_like(winner_values),
    )

    candidate_prediction = torch.sigmoid(
        (prediction_location[candidates] * prediction_weights[candidates]).sum(dim=2)
    )
    actual_binary = (candidates == winner_ids.unsqueeze(1)).to(torch.float32)
    raw_error = (candidate_prediction - actual_binary).abs()
    next_error = 0.2 * raw_error + 0.8 * prediction_error[candidates]
    next_confidence = torch.clamp(
        0.95 * prediction_confidence[candidates] + 0.05 * (1.0 - raw_error),
        min=0.0,
        max=1.0,
    )
    repeated_failure = raw_error > 0.65
    next_streak = repeated_failure.to(torch.int32)
    return winner_ids, strengths, next_error, next_confidence, next_streak


def _ensure_windows_triton_compiler() -> str | None:
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


def _compile_kernel(mode: str) -> HotPathKernelFn:
    return torch.compile(observe_route_predict_kernel, mode=mode, fullgraph=True)


def _measure_kernel(
    name: str,
    fn: HotPathKernelFn,
    *,
    input_patterns: torch.Tensor,
    candidate_indices: torch.Tensor,
    w_project: torch.Tensor,
    prototypes: torch.Tensor,
    input_weights: torch.Tensor,
    thresholds: torch.Tensor,
    context_gain: torch.Tensor,
    prediction_location: torch.Tensor,
    prediction_weights: torch.Tensor,
    prediction_error: torch.Tensor,
    prediction_confidence: torch.Tensor,
    input_weight_blend: float,
    iterations: int,
) -> dict[str, object]:
    device = input_patterns.device
    timings_ms: list[float] = []
    total_tokens = 0
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        winners, strengths, next_error, next_confidence, next_streak = fn(
            input_patterns,
            candidate_indices,
            w_project,
            prototypes,
            input_weights,
            thresholds,
            context_gain,
            prediction_location,
            prediction_weights,
            prediction_error,
            prediction_confidence,
            input_weight_blend,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings_ms.append((time.perf_counter_ns() - started) / 1e6)
        total_tokens += int(winners.numel())
        expected_shape = candidate_indices.shape
        if (
            int(strengths.numel()) != int(winners.numel())
            or next_error.shape != expected_shape
            or next_confidence.shape != expected_shape
            or next_streak.shape != expected_shape
        ):
            raise RuntimeError(f"{name} returned inconsistent output shapes")

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


def run_compiled_hot_path_kernel_benchmark(
    checkpoint: Path,
    *,
    batch_size: int = 256,
    iterations: int = 128,
    warmup_iterations: int = 8,
    matmul_precision: str = "high",
    candidate_source: str = "random",
    merge_torch_shards: bool = True,
    seed: int = 20260611,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if matmul_precision not in {"default", "highest", "high", "medium"}:
        raise ValueError("matmul_precision must be default, highest, high, or medium")
    if candidate_source not in {"random", "routing_index", "routing_index_tensor"}:
        raise ValueError("candidate_source must be random, routing_index, or routing_index_tensor")
    if matmul_precision != "default":
        torch.set_float32_matmul_precision(matmul_precision)

    trainer, _ = load_trainer_checkpoint(checkpoint)
    device = trainer.model.device
    if hasattr(trainer.model.hnsw_index, "merge_torch_shards"):
        trainer.model.hnsw_index.merge_torch_shards = bool(merge_torch_shards)
    comp = trainer.model.competitive
    predictive = trainer.model.predictive

    generator = torch.Generator(device=device).manual_seed(seed)
    input_patterns = torch.rand(
        batch_size,
        trainer.config.input_dim,
        generator=generator,
        device=device,
    )
    context_gain = torch.ones(batch_size, trainer.config.n_columns, device=device)

    w_project = comp.W_project.detach()
    prototypes = comp.prototypes.detach()
    input_weights = comp.input_weights.detach()
    thresholds = comp.thresholds.detach()
    prediction_location = predictive.location.detach()
    prediction_weights = predictive._prediction_weights.detach()
    prediction_error = predictive.prediction_error.detach()
    prediction_confidence = predictive.confidence.detach()
    input_weight_blend = float(comp.input_weight_blend)

    candidate_prep: dict[str, object]
    if candidate_source == "routing_index":
        candidate_indices, candidate_prep = _routing_index_candidates(
            trainer=trainer,
            input_patterns=input_patterns,
            w_project=w_project,
        )
    elif candidate_source == "routing_index_tensor":
        candidate_indices, candidate_prep = _routing_index_tensor_candidates(
            trainer=trainer,
            input_patterns=input_patterns,
            w_project=w_project,
        )
    else:
        candidate_indices = _random_candidates(
            batch_size=batch_size,
            k_routing=trainer.config.k_routing,
            n_columns=trainer.config.n_columns,
            generator=generator,
            device=device,
        )
        candidate_prep = {
            "candidate_source": "random",
            "candidate_prep_latency_ms": 0.0,
            "candidate_prep_tokens_per_second": None,
            "fallback_rows": 0,
            "distance_stats": None,
            "routing_index_stats": trainer.model.hnsw_index.stats(),
        }

    for _ in range(warmup_iterations):
        observe_route_predict_kernel(
            input_patterns,
            candidate_indices,
            w_project,
            prototypes,
            input_weights,
            thresholds,
            context_gain,
            prediction_location,
            prediction_weights,
            prediction_error,
            prediction_confidence,
            input_weight_blend,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()

    compiler_path = _ensure_windows_triton_compiler()
    arms: list[dict[str, object]] = [
        _measure_kernel(
            "eager_batch",
            observe_route_predict_kernel,
            input_patterns=input_patterns,
            candidate_indices=candidate_indices,
            w_project=w_project,
            prototypes=prototypes,
            input_weights=input_weights,
            thresholds=thresholds,
            context_gain=context_gain,
            prediction_location=prediction_location,
            prediction_weights=prediction_weights,
            prediction_error=prediction_error,
            prediction_confidence=prediction_confidence,
            input_weight_blend=input_weight_blend,
            iterations=iterations,
        )
    ]

    compile_errors: list[dict[str, str]] = []
    compile_modes = ("default", "reduce-overhead") if device.type == "cuda" else ()
    for mode in compile_modes:
        try:
            fn = _compile_kernel(mode)
            for _ in range(warmup_iterations):
                fn(
                    input_patterns,
                    candidate_indices,
                    w_project,
                    prototypes,
                    input_weights,
                    thresholds,
                    context_gain,
                    prediction_location,
                    prediction_weights,
                    prediction_error,
                    prediction_confidence,
                    input_weight_blend,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            arms.append(
                _measure_kernel(
                    f"torch_compile_{mode}",
                    fn,
                    input_patterns=input_patterns,
                    candidate_indices=candidate_indices,
                    w_project=w_project,
                    prototypes=prototypes,
                    input_weights=input_weights,
                    thresholds=thresholds,
                    context_gain=context_gain,
                    prediction_location=prediction_location,
                    prediction_weights=prediction_weights,
                    prediction_error=prediction_error,
                    prediction_confidence=prediction_confidence,
                    input_weight_blend=input_weight_blend,
                    iterations=iterations,
                )
            )
        except Exception as exc:  # pragma: no cover - backend dependent
            compile_errors.append({"mode": mode, "error": repr(exc)})

    best = max(arms, key=lambda arm: float(arm["tokens_per_second"]))
    cuda_memory: dict[str, float] = {}
    if device.type == "cuda":
        cuda_memory = {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        }

    return {
        "surface": "compiled_hot_path_kernel_benchmark.v1",
        "checkpoint": str(checkpoint),
        "scope": "isolated_fixed_shape_projection_competition_predictive_no_runtime_mutation",
        "claim_boundary": (
            "measures projection, candidate competition, and candidate-local predictive math; "
            "does not include retrieval, trainer orchestration, in-place plasticity, memory, "
            "binding, cross-modal grounding, checkpointing, or service throughput"
        ),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": int(batch_size),
        "iterations": int(iterations),
        "warmup_iterations": int(warmup_iterations),
        "matmul_precision": matmul_precision,
        "candidate_source": candidate_source,
        "merge_torch_shards": bool(merge_torch_shards),
        "candidate_prep": candidate_prep,
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
    parser.add_argument(
        "--matmul-precision",
        choices=("default", "highest", "high", "medium"),
        default="high",
    )
    parser.add_argument(
        "--candidate-source",
        choices=("random", "routing_index", "routing_index_tensor"),
        default="random",
    )
    parser.add_argument("--disable-merged-torch-shards", action="store_true")
    parser.add_argument("--seed", type=int, default=20260611)
    args = parser.parse_args()

    report = run_compiled_hot_path_kernel_benchmark(
        args.checkpoint,
        batch_size=args.batch_size,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        matmul_precision=args.matmul_precision,
        candidate_source=args.candidate_source,
        merge_torch_shards=not args.disable_merged_torch_shards,
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
