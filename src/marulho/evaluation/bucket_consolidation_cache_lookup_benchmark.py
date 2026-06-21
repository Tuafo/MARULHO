from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from marulho.consolidation.memory_store import DualMemoryStore


def _store(*, entries: int, buckets: int) -> DualMemoryStore:
    store = DualMemoryStore(capacity=int(entries))
    store.slow_buffer = [torch.empty(0) for _ in range(int(entries))]
    store.slow_bucket_ids = [int(index) % int(buckets) for index in range(int(entries))]
    store.slow_importance = [1.0 + float(index % 5) for index in range(int(entries))]
    store.slow_consolidation_level = [
        float((index % 17) / 16.0) for index in range(int(entries))
    ]
    store.bucket_consolidation_tensor(int(buckets), device=torch.device("cpu"))
    return store


def _retired_scan_level(store: DualMemoryStore, bucket_id: int) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for idx, raw_bucket_id in enumerate(store.slow_bucket_ids):
        if raw_bucket_id is None or int(raw_bucket_id) != int(bucket_id):
            continue
        weight = max(1e-6, float(store.slow_importance[idx]))
        level = max(0.0, min(1.0, float(store.slow_consolidation_level[idx])))
        weighted_sum += weight * level
        total_weight += weight
    if total_weight <= 0.0:
        return 0.0
    return float(weighted_sum / total_weight)


def _measure(fn: Any, *, runs: int) -> tuple[list[float], list[float]]:
    elapsed: list[float] = []
    values: list[float] = []
    for _ in range(int(runs)):
        started = time.perf_counter()
        values.append(float(fn()))
        elapsed.append(float((time.perf_counter() - started) * 1000.0))
    return elapsed, values


def _latency(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(statistics.fmean(values)) if values else 0.0,
        "median_ms": float(statistics.median(values)) if values else 0.0,
        "min_ms": float(min(values)) if values else 0.0,
        "max_ms": float(max(values)) if values else 0.0,
    }


def run_benchmark(
    *,
    entries: int,
    buckets: int,
    bucket_id: int,
    runs: int,
) -> dict[str, Any]:
    store = _store(entries=entries, buckets=buckets)
    bucket = int(bucket_id) % max(1, int(buckets))
    retired_elapsed, retired_values = _measure(
        lambda: _retired_scan_level(store, bucket),
        runs=runs,
    )
    cached_elapsed, cached_values = _measure(
        lambda: store.bucket_consolidation_level(bucket),
        runs=runs,
    )
    report = dict(store.device_report()["last_bucket_consolidation_level_report"])
    retired_stats = _latency(retired_elapsed)
    cached_stats = _latency(cached_elapsed)
    value_delta = abs(float(retired_values[-1]) - float(cached_values[-1]))
    quality = {
        "value_delta": float(value_delta),
        "cached_value_matches_retired_scan": bool(value_delta <= 1e-6),
        "cached_lookup_no_scan": bool(
            report.get("full_memory_scan") is False
            and int(report.get("scan_entry_count", -1)) == 0
            and report.get("status") == "cache_hit"
        ),
        "latency_reduction_mean_ms": float(
            max(0.0, retired_stats["mean_ms"] - cached_stats["mean_ms"])
        ),
    }
    quality["pass"] = bool(
        quality["cached_value_matches_retired_scan"]
        and quality["cached_lookup_no_scan"]
        and cached_stats["mean_ms"] < retired_stats["mean_ms"]
    )
    cuda_available = bool(torch.cuda.is_available())
    cuda_allocated_mib = (
        float(torch.cuda.memory_allocated() / (1024 * 1024))
        if cuda_available
        else 0.0
    )
    return {
        "artifact_kind": "bucket_consolidation_cache_lookup_benchmark",
        "surface": "bucket_consolidation_level_cache_lookup.v1",
        "entries": int(entries),
        "buckets": int(buckets),
        "bucket_id": int(bucket),
        "runs": int(runs),
        "selection_criteria": [
            "single_winner_bucket_consolidation_metric",
            "bucket_consolidation_cache_ready",
        ],
        "memory_budget": {
            "cached_bucket_rows": int(buckets),
            "retired_scan_rows": int(entries),
            "archival_storage_device": "cpu",
            "cache_metadata_device": "cpu",
        },
        "device_placement": {
            "bucket_cache_device": "cpu",
            "archival_storage_device": "cpu",
            "cuda_available": cuda_available,
            "cuda_memory_allocated_after_mib": cuda_allocated_mib,
        },
        "runtime_truth": {
            "runs_live_tick": True,
            "full_memory_scan": False,
            "scan_entry_count": 0,
            "runs_every_token": False,
            "hidden_language_reasoning": False,
            "mutates_runtime_state": False,
        },
        "latency": {
            "retired_full_bucket_scan": retired_stats,
            "cached_bucket_lookup": cached_stats,
        },
        "last_cached_lookup_report": report,
        "cache_rebuild_count": int(store.bucket_consolidation_cache_rebuild_count),
        "cache_rebuild_scan_entry_count": int(
            store.bucket_consolidation_cache_rebuild_scan_entry_count
        ),
        "quality": quality,
        "pass": bool(quality["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark cached bucket consolidation lookup versus retired scan."
    )
    parser.add_argument("--entries", type=int, default=65_536)
    parser.add_argument("--buckets", type=int, default=65_536)
    parser.add_argument("--bucket-id", type=int, default=37)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(
        entries=args.entries,
        buckets=args.buckets,
        bucket_id=args.bucket_id,
        runs=args.runs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "pass={pass_gate} entries={entries} bucket={bucket} "
        "retired_mean_ms={retired_ms:.6f} cached_mean_ms={cached_ms:.6f} "
        "scan_count={scan_count}".format(
            pass_gate=report["pass"],
            entries=report["entries"],
            bucket=report["bucket_id"],
            retired_ms=report["latency"]["retired_full_bucket_scan"]["mean_ms"],
            cached_ms=report["latency"]["cached_bucket_lookup"]["mean_ms"],
            scan_count=report["last_cached_lookup_report"].get("scan_entry_count"),
        )
    )


if __name__ == "__main__":
    main()
