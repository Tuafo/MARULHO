"""Benchmark bounded query-memory episode readout.

This benchmark compares fragment-only episode readout against the reported
selected-neighbor readout. It does not use live ticks or archive scans: the
synthetic store exposes direct indexed raw windows, and the readout may only
load windows around already returned memory matches.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from marulho.reporting.io import write_json_file
from marulho.training.query_runner import build_memory_episodes_with_report


class _SyntheticWindowSequence:
    def __init__(self, size: int, cluster_start: int) -> None:
        self.size = int(size)
        self.cluster_start = int(cluster_start)
        self.cluster = {
            self.cluster_start + 0: ".\na cat pu",
            self.cluster_start + 1: "purrs when",
            self.cluster_start + 2: "en it feel",
            self.cluster_start + 3: "els safe.\n",
        }

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> str:
        idx = int(index)
        if idx < 0 or idx >= self.size:
            raise IndexError(index)
        return self.cluster.get(idx, f"unrelated memory fragment {idx}")


class _SyntheticStore:
    def __init__(self, *, capacity: int, cluster_start: int) -> None:
        self.slow_raw_windows = _SyntheticWindowSequence(capacity, cluster_start)


def _matches(cluster_start: int) -> list[dict[str, Any]]:
    return [
        {
            "memory_index": int(cluster_start + 3),
            "text": "els safe.\n",
            "raw_window": "els safe.\n",
            "similarity": 0.99,
            "importance": 1.0,
        },
        {
            "memory_index": int(cluster_start + 2),
            "text": "en it feel",
            "raw_window": "en it feel",
            "similarity": 0.98,
            "importance": 1.0,
        },
        {
            "memory_index": int(cluster_start + 1),
            "text": "purrs when",
            "raw_window": "purrs when",
            "similarity": 0.97,
            "importance": 1.0,
        },
        {
            "memory_index": int(cluster_start),
            "text": ".\na cat pu",
            "raw_window": ".\na cat pu",
            "similarity": 0.96,
            "importance": 1.0,
        },
    ]


def _target_hit(episodes: list[dict[str, Any]], target: str) -> bool:
    if not episodes:
        return False
    return target.lower() in str(episodes[0].get("text", "")).lower()


def run_benchmark(
    *,
    capacity: int,
    iterations: int,
    neighbor_radius: int,
    min_speedup: float,
) -> dict[str, Any]:
    cluster_start = max(0, min(int(capacity) - 4, int(capacity) // 2))
    store = _SyntheticStore(capacity=int(capacity), cluster_start=cluster_start)
    matches = _matches(cluster_start)
    target = "cat purrs when it feels safe"

    legacy_latencies: list[float] = []
    bounded_latencies: list[float] = []
    legacy_episodes: list[dict[str, Any]] = []
    bounded_episodes: list[dict[str, Any]] = []
    bounded_report: dict[str, Any] = {}

    for _ in range(max(1, int(iterations))):
        started = time.perf_counter()
        legacy_episodes, _legacy_report = build_memory_episodes_with_report(
            matches,
            top_k=2,
            query_terms=["purrs", "feels", "safe"],
        )
        legacy_latencies.append((time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        bounded_episodes, bounded_report = build_memory_episodes_with_report(
            matches,
            top_k=2,
            query_terms=["purrs", "feels", "safe"],
            memory_store=store,
            neighbor_radius=int(neighbor_radius),
        )
        bounded_latencies.append((time.perf_counter() - started) * 1000.0)

    legacy_hit = _target_hit(legacy_episodes, target)
    bounded_hit = _target_hit(bounded_episodes, target)
    legacy_mean = float(statistics.fmean(legacy_latencies))
    bounded_mean = float(statistics.fmean(bounded_latencies))
    speedup = float(legacy_mean / bounded_mean) if bounded_mean > 0.0 else float("inf")
    report_gate_pass = (
        bounded_report.get("surface") == "bounded_query_memory_episode_readout.v1"
        and bounded_report.get("raw_text_payload_policy") == "selected_match_neighbor_windows_only"
        and not bool(bounded_report.get("global_candidate_scan"))
        and not bool(bounded_report.get("global_score_scan"))
        and not bool(bounded_report.get("runs_live_tick"))
        and not bool(bounded_report.get("runs_every_token"))
        and not bool(bounded_report.get("language_reasoning"))
        and bounded_report.get("archival_storage_device") == "cpu"
    )
    latency_gate_pass = bool(speedup >= float(min_speedup))
    quality_gate_pass = bool((not legacy_hit) and bounded_hit)

    return {
        "surface": "bounded_query_memory_episode_readout_benchmark.v1",
        "passed": bool(quality_gate_pass and report_gate_pass and latency_gate_pass),
        "capacity": int(capacity),
        "iterations": int(max(1, int(iterations))),
        "cluster_start": int(cluster_start),
        "input_match_count": int(len(matches)),
        "quality": {
            "metric": "target_phrase_top_episode_recovery",
            "target_phrase": target,
            "legacy_target_hit": bool(legacy_hit),
            "bounded_target_hit": bool(bounded_hit),
            "legacy_top_text": "" if not legacy_episodes else str(legacy_episodes[0].get("text", "")),
            "bounded_top_text": "" if not bounded_episodes else str(bounded_episodes[0].get("text", "")),
        },
        "latency_ms": {
            "legacy_mean": legacy_mean,
            "bounded_mean": bounded_mean,
            "speedup": speedup,
            "legacy_min": float(min(legacy_latencies)),
            "bounded_min": float(min(bounded_latencies)),
        },
        "device_placement": {
            "archival_storage_device": "cpu",
            "readout_compute_device": "cpu",
        },
        "bounded_report": bounded_report,
        "gates": {
            "quality_gate_pass": bool(quality_gate_pass),
            "report_gate_pass": bool(report_gate_pass),
            "latency_gate_pass": bool(latency_gate_pass),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--capacity", type=int, default=65536)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--neighbor-radius", type=int, default=3)
    parser.add_argument("--min-speedup", type=float, default=0.1)
    args = parser.parse_args()

    report = run_benchmark(
        capacity=int(args.capacity),
        iterations=int(args.iterations),
        neighbor_radius=int(args.neighbor_radius),
        min_speedup=float(args.min_speedup),
    )
    if args.output is not None:
        write_json_file(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not bool(report.get("passed")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
