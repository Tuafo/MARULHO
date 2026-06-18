from __future__ import annotations

import argparse
from array import array
from pathlib import Path
import time
from typing import Any, Callable

import torch

from marulho.consolidation.memory_store import DualMemoryStore
from marulho.reporting.io import write_json_file


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return float(ordered[index])


def _timed(
    fn: Callable[[], Any],
    *,
    iterations: int,
) -> tuple[Any, dict[str, float]]:
    durations: list[float] = []
    result: Any = None
    for _ in range(max(1, int(iterations))):
        started = time.perf_counter()
        result = fn()
        durations.append((time.perf_counter() - started) * 1000.0)
    return result, {
        "mean_ms": float(sum(durations) / len(durations)),
        "p95_ms": _p95(durations),
        "min_ms": float(min(durations)),
        "max_ms": float(max(durations)),
    }


def _build_hot_bucket_store(*, archive_size: int, bucket_id: int) -> DualMemoryStore:
    size = max(0, int(archive_size))
    store = DualMemoryStore(capacity=size)
    base = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    store.slow_buffer = [base for _ in range(size)]
    store.slow_input_patterns = [base for _ in range(size)]
    store.slow_routing_keys = [base for _ in range(size)]
    store.slow_raw_windows = [None for _ in range(size)]
    store.slow_texts = [None for _ in range(size)]
    store.slow_metadata = [None for _ in range(size)]
    store.slow_bucket_ids = [int(bucket_id) for _ in range(size)]
    store.slow_importance = [1.0 for _ in range(size)]
    store.slow_capture_tag = array("d", [1.0 for _ in range(size)])
    store.slow_tag_is_strong = array("b", [1 for _ in range(size)])
    store.slow_local_prp = array("d", [1.0 for _ in range(size)])
    store.slow_last_capture_token = [0 for _ in range(size)]
    store.slow_consolidation_level = [0.0 for _ in range(size)]
    store.slow_consolidation_events = [0 for _ in range(size)]
    store.slow_entry_timestamps = array("q", range(size))
    store.slow_last_replay_token = [0 for _ in range(size)]
    store.slow_replay_count = [0 for _ in range(size)]
    store.slow_ripple_strength = array("d", [0.0 for _ in range(size)])
    store._bucket_entry_indices[int(bucket_id)] = list(range(size))
    return store


def _legacy_materialized_candidates(
    store: DualMemoryStore,
    *,
    bucket_id: int,
    limit: int,
) -> dict[str, Any]:
    materialized = list(reversed(store._bucket_entry_indices[int(bucket_id)]))
    return {
        "surface": "legacy_hot_bucket_candidate_materialization.v1",
        "candidate_indices": [int(index) for index in materialized[: max(0, int(limit))]],
        "source_materialized_entry_count": int(len(materialized)),
        "source_materialization_count": 1,
        "candidate_source_full_bucket_materialization": True,
        "candidate_source_full_bucket_scan": True,
    }


def run_benchmark(
    *,
    output: Path,
    archive_size: int,
    candidate_limit: int,
    iterations: int,
    bucket_id: int,
) -> dict[str, Any]:
    store = _build_hot_bucket_store(archive_size=archive_size, bucket_id=bucket_id)
    limit = max(0, int(candidate_limit))
    cuda_available = bool(torch.cuda.is_available())
    cuda_before = int(torch.cuda.memory_allocated()) if cuda_available else 0

    legacy_result, legacy_latency = _timed(
        lambda: _legacy_materialized_candidates(
            store,
            bucket_id=bucket_id,
            limit=limit,
        ),
        iterations=iterations,
    )
    bounded_result, bounded_latency = _timed(
        lambda: store.collect_query_memory_match_indices(
            candidate_bucket_ids=[bucket_id],
            max_candidates=limit,
            scope="hot_bucket_candidate_source_window_benchmark",
        ),
        iterations=iterations,
    )
    replay_selection = store.select_replay_window(
        n=limit,
        current_token=archive_size,
        candidate_pool=limit,
        strategy="consolidation",
        candidate_bucket_ids=[bucket_id],
        scope="hot_bucket_replay_selection_source_window_benchmark",
    )

    cuda_after = int(torch.cuda.memory_allocated()) if cuda_available else 0
    bounded_indices = [int(index) for index in bounded_result.get("match_indices", [])]
    legacy_indices = [int(index) for index in legacy_result.get("candidate_indices", [])]
    expected_recent = list(range(max(0, int(archive_size) - limit), int(archive_size)))
    expected_recent.reverse()
    recent_hits = len(set(bounded_indices).intersection(expected_recent))
    speedup = (
        float(legacy_latency["mean_ms"] / bounded_latency["mean_ms"])
        if bounded_latency["mean_ms"] > 0.0
        else float("inf")
    )
    quality_gate = {
        "selected_index_parity": bool(bounded_indices == legacy_indices),
        "newest_candidate_hit_rate": float(recent_hits / max(1, len(expected_recent))),
        "bounded_source_read_lte_candidate_limit": bool(
            int(bounded_result.get("candidate_source_entry_read_count", 0) or 0)
            <= limit
        ),
        "no_full_bucket_materialization": not bool(
            bounded_result.get("candidate_source_full_bucket_materialization")
        ),
        "no_full_bucket_scan": not bool(
            bounded_result.get("candidate_source_full_bucket_scan")
        ),
        "no_global_scan": not bool(bounded_result.get("global_candidate_scan"))
        and not bool(bounded_result.get("global_score_scan")),
        "cpu_archival_storage": str(
            bounded_result.get("archival_storage_device")
        )
        == "cpu",
        "latency_improved": bool(
            bounded_latency["mean_ms"] <= legacy_latency["mean_ms"]
        ),
    }
    status_pass = bool(
        quality_gate["selected_index_parity"]
        and quality_gate["newest_candidate_hit_rate"] >= 1.0
        and quality_gate["bounded_source_read_lte_candidate_limit"]
        and quality_gate["no_full_bucket_materialization"]
        and quality_gate["no_full_bucket_scan"]
        and quality_gate["no_global_scan"]
        and quality_gate["cpu_archival_storage"]
    )
    payload = {
        "surface": "bucket_candidate_source_window_benchmark.v1",
        "status": "passed" if status_pass else "failed",
        "archive_size": int(archive_size),
        "candidate_limit": int(limit),
        "bucket_id": int(bucket_id),
        "iterations": int(max(1, int(iterations))),
        "legacy": {
            **legacy_latency,
            "result": legacy_result,
        },
        "bounded": {
            **bounded_latency,
            "result": bounded_result,
        },
        "replay_selection": replay_selection,
        "speedup": speedup,
        "quality_gate": quality_gate,
        "device": {
            "archival_storage_device": "cpu",
            "active_candidate_compute_device": "cpu",
            "cuda_available": cuda_available,
            "cuda_memory_before_bytes": cuda_before,
            "cuda_memory_after_bytes": cuda_after,
            "cuda_memory_delta_mib": float(
                (cuda_after - cuda_before) / (1024.0 * 1024.0)
            ),
            "memory_store_device_report": store.device_report(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded hot-bucket candidate source windows."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-size", type=int, default=65536)
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--bucket-id", type=int, default=7)
    args = parser.parse_args()
    payload = run_benchmark(
        output=args.output,
        archive_size=args.archive_size,
        candidate_limit=args.candidate_limit,
        iterations=args.iterations,
        bucket_id=args.bucket_id,
    )
    print(
        {
            "status": payload["status"],
            "archive_size": payload["archive_size"],
            "candidate_limit": payload["candidate_limit"],
            "legacy_mean_ms": payload["legacy"]["mean_ms"],
            "bounded_mean_ms": payload["bounded"]["mean_ms"],
            "speedup": payload["speedup"],
        }
    )


if __name__ == "__main__":
    main()
