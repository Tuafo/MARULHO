"""Hot-window benchmark for already-encoded MARULHO token tensors.

This benchmark intentionally excludes service, source loading, tokenization,
UI/status reads, checkpointing, replay, and sleep maintenance. It measures the
current Subcortex training loop over fixed-shape input tensors so wider compile,
CUDA graph, or fused-kernel work has a stable target. Individual promoted
transitions may already be compiled and are reported explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Callable

import torch

from marulho.training.checkpointing import load_trainer_checkpoint


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def run_hot_window_benchmark(
    checkpoint: Path,
    *,
    samples: int = 512,
    warmup_steps: int = 64,
    routing_candidate_mode: str = "tensor",
    merge_torch_shards: bool = True,
    predictive_transition_mode: str | None = None,
    seed: int = 20260611,
    _trainer_setup: Callable[[object], None] | None = None,
    profile_trainer_stages: bool = False,
    sync_mode: str = "step",
) -> dict[str, object]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if routing_candidate_mode not in {"list", "tensor"}:
        raise ValueError("routing_candidate_mode must be list or tensor")
    if predictive_transition_mode not in {
        None,
        "legacy",
        "fused_eager",
        "compiled",
        "inplace_triton",
    }:
        raise ValueError(
            "predictive_transition_mode must be legacy, fused_eager, compiled, inplace_triton, or None"
        )
    if sync_mode not in {"step", "window"}:
        raise ValueError("sync_mode must be step or window")
    trainer, _metadata = load_trainer_checkpoint(checkpoint)
    trainer.config.micro_sleep_interval_tokens = 10**9
    trainer.config.deep_sleep_interval_tokens = 10**9
    if predictive_transition_mode is not None:
        trainer.config.predictive_dense_transition_mode = predictive_transition_mode
        if predictive_transition_mode == "inplace_triton":
            from marulho.training.column_transition_runtime import (
                ColumnTransitionRuntime,
            )

            trainer._column_transition_runtime = ColumnTransitionRuntime(trainer)
    if _trainer_setup is not None:
        _trainer_setup(trainer)
    device = trainer.model.device
    if hasattr(trainer.model.hnsw_index, "merge_torch_shards"):
        trainer.model.hnsw_index.merge_torch_shards = bool(merge_torch_shards)
    if routing_candidate_mode == "list":
        def _list_routing_candidates(routing_key: torch.Tensor) -> torch.Tensor | None:
            candidate_ids, _ = trainer.model.hnsw_index.search(
                routing_key.unsqueeze(0),
                k=trainer.config.k_routing,
            )
            if not candidate_ids or not candidate_ids[0]:
                return None
            return torch.tensor(candidate_ids[0], dtype=torch.long, device=device)

        trainer._routing_candidates = _list_routing_candidates  # type: ignore[method-assign]

    generator = torch.Generator(device=device).manual_seed(seed)
    patterns = [
        torch.rand(trainer.config.input_dim, generator=generator, device=device)
        for _ in range(samples + warmup_steps)
    ]

    warmup_started = time.perf_counter_ns()
    for index in range(warmup_steps):
        trainer.train_step(
            patterns[index],
            raw_window=f"hot-window warmup {index}",
            allow_sleep_maintenance=False,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    warmup_elapsed_s = (time.perf_counter_ns() - warmup_started) / 1e9

    if profile_trainer_stages:
        trainer.enable_train_step_profile(reset=True)
    step_latencies_ms: list[float] = []
    if device.type == "cuda" and sync_mode == "window":
        torch.cuda.synchronize()
    started_window = time.perf_counter_ns()
    for index, pattern in enumerate(patterns[warmup_steps:], start=warmup_steps):
        if device.type == "cuda" and sync_mode == "step":
            torch.cuda.synchronize()
        started_step = time.perf_counter_ns()
        trainer.train_step(
            pattern,
            raw_window=f"hot-window measure {index}",
            allow_sleep_maintenance=False,
        )
        if device.type == "cuda" and sync_mode == "step":
            torch.cuda.synchronize()
        step_latencies_ms.append((time.perf_counter_ns() - started_step) / 1e6)
    if device.type == "cuda" and sync_mode == "window":
        torch.cuda.synchronize()
    total_elapsed_s = (time.perf_counter_ns() - started_window) / 1e9
    trainer_stage_profile: dict[str, object] | None = None
    if profile_trainer_stages:
        trainer_stage_profile = dict(trainer.train_step_profile_report())
        trainer_stage_profile["scope"] = (
            "measured_hot_window_steps_only_no_service_no_source_no_sleep"
        )
        trainer.disable_train_step_profile()

    tokens_per_second = samples / max(total_elapsed_s, 1e-9)
    step_latencies_ms_sorted = sorted(step_latencies_ms)
    cuda_available = torch.cuda.is_available()
    cuda_memory: dict[str, float] = {}
    if device.type == "cuda":
        cuda_memory = {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        }

    return {
        "surface": "hot_window_benchmark.v1",
        "checkpoint": str(checkpoint),
        "scope": "already_encoded_tensor_train_step_no_service_no_source_no_sleep",
        "claim_boundary": (
            "measures current configured hot-window throughput; "
            "individual compiled transitions are reported but the full train_step is not fused; "
            "does not prove service endpoint throughput or fused-kernel capacity"
        ),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": bool(cuda_available),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": int(seed),
        "samples": int(samples),
        "warmup_steps": int(warmup_steps),
        "routing_candidate_mode": routing_candidate_mode,
        "merge_torch_shards": bool(merge_torch_shards),
        "sync_mode": sync_mode,
        "latency_sample_scope": (
            "cuda_synchronized_step_latency"
            if sync_mode == "step"
            else "host_dispatch_latency_with_single_window_cuda_sync"
        ),
        "predictive_transition_mode": str(
            trainer.config.predictive_dense_transition_mode
        ),
        "transition_executor": str(
            getattr(trainer, "_benchmark_transition_executor", "runtime")
        ),
        "warmup_elapsed_s": warmup_elapsed_s,
        "trainer_stage_profile": trainer_stage_profile,
        "n_columns": int(trainer.config.n_columns),
        "input_dim": int(trainer.config.input_dim),
        "k_routing": int(trainer.config.k_routing),
        "total_elapsed_s": total_elapsed_s,
        "tokens_per_second": tokens_per_second,
        "step_latency_ms": {
            "median": statistics.median(step_latencies_ms),
            "p95": _percentile(step_latencies_ms_sorted, 0.95),
            "mean": statistics.mean(step_latencies_ms),
            "min": min(step_latencies_ms),
            "max": max(step_latencies_ms),
        },
        "target_gap": {
            "target_tokens_per_second": 1000.0,
            "target_met": tokens_per_second >= 1000.0,
            "multiple_needed": 1000.0 / max(tokens_per_second, 1e-9),
        },
        "runtime_counters": {
            "token_count": int(trainer.token_count),
            "last_winner": None if trainer.last_winner is None else int(trainer.last_winner),
            "binding_execution_mode": (
                str(getattr(trainer.model.binding_layer, "last_runtime_execution_mode", "disabled"))
                if trainer.model.binding_layer is not None
                else "disabled"
            ),
            "cross_modal_text_execution_mode": (
                str(getattr(trainer.model.cross_modal, "last_text_runtime_execution_mode", "disabled"))
                if trainer.model.cross_modal is not None
                else "disabled"
            ),
            "cross_modal_text_idle_probe_interval_tokens": int(
                trainer.config.cross_modal_text_idle_probe_interval_tokens
            ),
            "candidate_homeostasis_start_tokens": int(
                trainer.config.candidate_homeostasis_start_tokens
            ),
            "routing_index": trainer.model.hnsw_index.stats(),
            "competitive": trainer.model.competitive.execution_report(),
            "predictive": trainer.model.predictive.device_report(),
            "column_transition_runtime": (
                trainer.column_transition_runtime_report()
            ),
        },
        "cuda_memory": cuda_memory,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument(
        "--routing-candidate-mode",
        choices=("list", "tensor"),
        default="tensor",
    )
    parser.add_argument("--disable-merged-torch-shards", action="store_true")
    parser.add_argument(
        "--predictive-transition-mode",
        choices=("legacy", "fused_eager", "compiled", "inplace_triton"),
    )
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--profile-trainer-stages", action="store_true")
    parser.add_argument(
        "--sync-mode",
        choices=("step", "window"),
        default="step",
        help=(
            "step synchronizes CUDA before/after every measured token; "
            "window synchronizes once around the measured window to estimate "
            "continuous stream throughput without per-token host barriers"
        ),
    )
    args = parser.parse_args()

    report = run_hot_window_benchmark(
        args.checkpoint,
        samples=args.samples,
        warmup_steps=args.warmup_steps,
        routing_candidate_mode=args.routing_candidate_mode,
        merge_torch_shards=not args.disable_merged_torch_shards,
        predictive_transition_mode=args.predictive_transition_mode,
        seed=args.seed,
        profile_trainer_stages=args.profile_trainer_stages,
        sync_mode=args.sync_mode,
    )
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
