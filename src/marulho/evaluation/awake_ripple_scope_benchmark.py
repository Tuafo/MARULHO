"""Compare global awake-ripple tagging with wake-plan-scoped tagging."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import torch

from marulho.consolidation.memory_store import DualMemoryStore
from marulho.reporting.readme_reports import write_json_report_with_readme


def _build_store(
    *,
    capacity: int,
    bucket_count: int,
    dim: int,
) -> DualMemoryStore:
    store = DualMemoryStore(capacity=capacity)
    template = torch.linspace(0.0, 1.0, steps=max(1, int(dim)), dtype=torch.float32)
    for index in range(int(capacity)):
        bucket_id = int(index % max(1, int(bucket_count)))
        store.update(
            template + float(index % 17) * 0.001,
            importance=0.8,
            token_count=index + 1,
            bucket_id=bucket_id,
        )
    return store


def _measure(
    store: DualMemoryStore,
    *,
    iterations: int,
    current_token: int,
    window_tokens: int,
    awake_bucket_ids: list[int] | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    tagged_total = 0
    for offset in range(int(iterations)):
        tagged_total += int(
            store.ripple_tag_awake(
                current_token=int(current_token) + offset,
                window_tokens=int(window_tokens),
                da_level=0.95,
                awake_bucket_ids=awake_bucket_ids,
            )
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "elapsed_ms": float(elapsed_ms),
        "mean_ms": float(elapsed_ms / max(1, int(iterations))),
        "tagged_total": int(tagged_total),
        "last_ripple_scan_mode": str(store.last_ripple_scan_mode),
        "ripple_scalar_scan_count": int(store.ripple_scalar_scan_count),
        "ripple_vector_scan_count": int(store.ripple_vector_scan_count),
        "ripple_awake_bucket_scan_count": int(
            store.ripple_awake_bucket_scan_count
        ),
        "ripple_awake_bucket_candidate_count": int(
            store.ripple_awake_bucket_candidate_count
        ),
        "last_ripple_awake_bucket_count": int(
            store.last_ripple_awake_bucket_count
        ),
        "last_ripple_awake_candidate_count": int(
            store.last_ripple_awake_candidate_count
        ),
    }


def run_awake_ripple_scope_benchmark(
    *,
    output_path: Path,
    capacity: int = 8192,
    bucket_count: int = 8192,
    awake_bucket_count: int = 10,
    iterations: int = 256,
    dim: int = 16,
) -> dict[str, Any]:
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    if awake_bucket_count < 0:
        raise ValueError("awake_bucket_count must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    bounded_awake_bucket_count = min(int(awake_bucket_count), int(bucket_count))
    awake_bucket_ids = list(range(bounded_awake_bucket_count))
    current_token = int(capacity) + 1
    window_tokens = int(capacity) + int(iterations) + 1

    global_store = _build_store(
        capacity=int(capacity),
        bucket_count=int(bucket_count),
        dim=int(dim),
    )
    scoped_store = _build_store(
        capacity=int(capacity),
        bucket_count=int(bucket_count),
        dim=int(dim),
    )

    global_result = _measure(
        global_store,
        iterations=int(iterations),
        current_token=current_token,
        window_tokens=window_tokens,
        awake_bucket_ids=None,
    )
    scoped_result = _measure(
        scoped_store,
        iterations=int(iterations),
        current_token=current_token,
        window_tokens=window_tokens,
        awake_bucket_ids=awake_bucket_ids,
    )
    speedup = float(
        global_result["mean_ms"] / max(float(scoped_result["mean_ms"]), 1e-12)
    )
    report = {
        "surface": "awake_ripple_scope_benchmark.v1",
        "scope": "memory_replay_tagging_uses_training_owned_wake_buckets",
        "capacity": int(capacity),
        "bucket_count": int(bucket_count),
        "awake_bucket_count": int(bounded_awake_bucket_count),
        "iterations": int(iterations),
        "dim": int(dim),
        "global_unscoped": global_result,
        "wake_bucket_scoped": scoped_result,
        "speedup": speedup,
        "gates": {
            "scoped_avoids_global_memory_scan": bool(
                scoped_result["ripple_scalar_scan_count"] == 0
                and scoped_result["ripple_vector_scan_count"] == 0
                and scoped_result["ripple_awake_bucket_scan_count"]
                == int(iterations)
            ),
            "scoped_candidate_count_bounded": bool(
                scoped_result["last_ripple_awake_candidate_count"]
                <= int(bounded_awake_bucket_count)
                * max(1, (int(capacity) + int(bucket_count) - 1) // int(bucket_count))
            ),
            "scoped_not_slower_than_global": bool(
                float(scoped_result["mean_ms"]) <= float(global_result["mean_ms"])
            ),
        },
    }
    write_json_report_with_readme(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=8192)
    parser.add_argument("--bucket-count", type=int, default=8192)
    parser.add_argument("--awake-bucket-count", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--dim", type=int, default=16)
    args = parser.parse_args()

    report = run_awake_ripple_scope_benchmark(
        output_path=args.output,
        capacity=args.capacity,
        bucket_count=args.bucket_count,
        awake_bucket_count=args.awake_bucket_count,
        iterations=args.iterations,
        dim=args.dim,
    )
    print(
        {
            "speedup": report["speedup"],
            "global_mean_ms": report["global_unscoped"]["mean_ms"],
            "scoped_mean_ms": report["wake_bucket_scoped"]["mean_ms"],
            "gates": report["gates"],
        }
    )


if __name__ == "__main__":
    main()
