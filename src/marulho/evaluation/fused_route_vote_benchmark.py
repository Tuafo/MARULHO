"""Benchmark the evaluation-only two-launch Triton route and vote probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from marulho.core.inplace_column_cuda import select_fused_vote_competition_cuda
from marulho.evaluation.compiled_hot_path_kernel_benchmark import (
    _routing_tensor_cache,
)
from marulho.evaluation.fused_route_vote_triton import fused_route_vote_cuda
from marulho.training.checkpointing import load_trainer_checkpoint


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def _measure(
    name: str,
    fn,
    routing_keys: torch.Tensor,
    *,
    iterations: int,
) -> dict[str, object]:
    timings_ms: list[float] = []
    device = routing_keys.device
    for step in range(iterations):
        key = routing_keys[step % int(routing_keys.shape[0])]
        torch.cuda.synchronize(device)
        started = time.perf_counter_ns()
        fn(key)
        torch.cuda.synchronize(device)
        timings_ms.append((time.perf_counter_ns() - started) / 1e6)
    total_seconds = sum(timings_ms) / 1000.0
    return {
        "name": name,
        "iterations": iterations,
        "ticks_per_second": iterations / max(total_seconds, 1e-9),
        "latency_ms": {
            "median": statistics.median(timings_ms),
            "p95": _percentile(timings_ms, 0.95),
            "mean": statistics.mean(timings_ms),
            "min": min(timings_ms),
            "max": max(timings_ms),
        },
    }


def run_fused_route_vote_benchmark(
    checkpoint: Path,
    *,
    samples: int = 128,
    iterations: int = 512,
    warmup_steps: int = 16,
    previous_winner: int = 0,
    seed: int = 20260612,
) -> dict[str, object]:
    if samples <= 0 or iterations <= 0:
        raise ValueError("samples and iterations must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    trainer, metadata = load_trainer_checkpoint(checkpoint)
    device = trainer.model.device
    if device.type != "cuda":
        raise RuntimeError("fused route/vote benchmark requires CUDA")
    comp = trainer.model.competitive
    if not bool(trainer.config.enable_learned_chunking):
        raise RuntimeError("fused route/vote benchmark requires learned chunking")
    if int(comp.n_winners) != 1 or float(comp.input_weight_blend) != 0.0:
        raise RuntimeError(
            "fused route/vote benchmark requires one winner and zero input blend"
        )

    generator = torch.Generator(device=device).manual_seed(seed)
    patterns = torch.rand(
        samples,
        int(trainer.config.input_dim),
        generator=generator,
        device=device,
    )
    routing_keys = torch.stack(
        [trainer.model.routing_key_from_pattern(pattern) for pattern in patterns]
    )
    routing_vectors, routing_ids, cache_report = _routing_tensor_cache(
        trainer.model.hnsw_index
    )
    k_routing = min(
        int(trainer.config.k_routing),
        int(routing_ids.numel()),
    )

    production_previous = torch.tensor(
        [previous_winner],
        dtype=torch.long,
        device=device,
    )
    production_winner = torch.empty(1, dtype=torch.long, device=device)
    production_strength = torch.empty(1, device=device)
    production_positive = torch.empty((), dtype=torch.bool, device=device)

    fused_previous = production_previous.clone()
    fused_scores = torch.empty(int(routing_ids.numel()), device=device)
    fused_candidates = torch.empty(k_routing, dtype=torch.long, device=device)
    fused_winner = torch.empty(1, dtype=torch.long, device=device)
    fused_strength = torch.empty(1, device=device)
    fused_positive = torch.empty((), dtype=torch.bool, device=device)
    fused_reconstruction_error = torch.empty(1, device=device)

    def production_step(key: torch.Tensor) -> torch.Tensor:
        candidates, _ = trainer.model.hnsw_index.search_tensors(
            key.unsqueeze(0),
            k=k_routing,
        )
        normalized = comp._cached_normalize_key(key)  # noqa: SLF001
        select_fused_vote_competition_cuda(
            routing_key=normalized,
            prototypes=comp.prototypes,
            thresholds=comp.thresholds,
            prediction_location=trainer.model.predictive.location,
            candidates=candidates[0],
            previous_winner=production_previous,
            winner_out=production_winner,
            strength_out=production_strength,
            competition_had_positive=production_positive,
        )
        return candidates[0]

    def fused_step(key: torch.Tensor) -> torch.Tensor:
        fused_route_vote_cuda(
            routing_key=key,
            routing_vectors=routing_vectors,
            routing_ids=routing_ids,
            prototypes=comp.prototypes,
            thresholds=comp.thresholds,
            prediction_location=trainer.model.predictive.location,
            previous_winner=fused_previous,
            scores_out=fused_scores,
            candidates_out=fused_candidates,
            winner_out=fused_winner,
            strength_out=fused_strength,
            competition_had_positive=fused_positive,
            reconstruction_error_out=fused_reconstruction_error,
        )
        return fused_candidates

    parity_candidate_mismatches = 0
    parity_candidate_set_mismatches = 0
    parity_winner_mismatches = 0
    parity_positive_mismatches = 0
    for key in routing_keys:
        reference_candidates = production_step(key).clone()
        got_candidates = fused_step(key).clone()
        torch.cuda.synchronize(device)
        parity_candidate_mismatches += int(
            not torch.equal(reference_candidates, got_candidates)
        )
        parity_candidate_set_mismatches += int(
            not torch.equal(
                torch.sort(reference_candidates).values,
                torch.sort(got_candidates).values,
            )
        )
        parity_winner_mismatches += int(
            int(production_winner.item()) != int(fused_winner.item())
        )
        parity_positive_mismatches += int(
            bool(production_positive.item()) != bool(fused_positive.item())
        )

    for _ in range(warmup_steps):
        key = routing_keys[_ % samples]
        production_step(key)
        fused_step(key)
    torch.cuda.synchronize(device)

    initial_previous = torch.tensor(
        [previous_winner],
        dtype=torch.long,
        device=device,
    )
    production_previous.copy_(initial_previous)
    production_arm = _measure(
        "production_tensor_route_plus_fused_vote",
        production_step,
        routing_keys,
        iterations=iterations,
    )
    fused_previous.copy_(initial_previous)
    fused_arm = _measure(
        "evaluation_two_launch_triton_route_vote",
        fused_step,
        routing_keys,
        iterations=iterations,
    )
    speedup = float(fused_arm["ticks_per_second"]) / max(
        float(production_arm["ticks_per_second"]),
        1e-9,
    )
    parity_safe = (
        parity_candidate_set_mismatches == 0
        and parity_winner_mismatches == 0
        and parity_positive_mismatches == 0
    )

    return {
        "surface": "fused_route_vote_benchmark.v1",
        "checkpoint": str(checkpoint),
        "checkpoint_metadata": metadata,
        "scope": "sequential_routing_keys_exact_cache_route_predictive_vote_competition",
        "claim_boundary": (
            "starts after production routing_key_from_pattern and excludes "
            "column mutation, memory, consolidation, cross-modal work, source "
            "orchestration, service endpoints, and checkpointing"
        ),
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        },
        "shape": {
            "samples": samples,
            "routing_vectors": list(routing_vectors.shape),
            "routing_ids": list(routing_ids.shape),
            "k_routing": k_routing,
            "column_dim": int(comp.column_dim),
            "location_dim": int(trainer.model.predictive.location.shape[1]),
        },
        "routing_cache": cache_report,
        "parity": {
            "checked_ticks": samples,
            "candidate_mismatch_count": parity_candidate_mismatches,
            "candidate_set_mismatch_count": parity_candidate_set_mismatches,
            "winner_mismatch_count": parity_winner_mismatches,
            "positive_branch_mismatch_count": parity_positive_mismatches,
            "promotion_safe": parity_safe,
        },
        "arms": [production_arm, fused_arm],
        "speedup": speedup,
        "promotion_status": (
            "eligible_for_complete_train_step_ab"
            if parity_safe and speedup > 1.0
            else "rejected_until_parity_and_speedup"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--previous-winner", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260612)
    args = parser.parse_args()
    report = run_fused_route_vote_benchmark(
        args.checkpoint,
        samples=args.samples,
        iterations=args.iterations,
        warmup_steps=args.warmup_steps,
        previous_winner=args.previous_winner,
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
