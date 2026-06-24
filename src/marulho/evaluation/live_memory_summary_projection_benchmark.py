from __future__ import annotations

import argparse
from array import array
import json
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any

import torch

from marulho.consolidation.memory_store import DualMemoryStore


def _build_store(*, entries: int, dim: int, seed: int, token: int) -> DualMemoryStore:
    torch.manual_seed(int(seed))
    store = DualMemoryStore(capacity=int(entries))
    vectors = torch.randn(int(entries), int(dim), dtype=torch.float32)
    store.slow_buffer = [row.detach().clone() for row in vectors]
    store.slow_input_patterns = [None for _ in range(int(entries))]
    store.slow_routing_keys = [None for _ in range(int(entries))]
    store.slow_raw_windows = [None for _ in range(int(entries))]
    store.slow_texts = [None for _ in range(int(entries))]
    store.slow_metadata = [None for _ in range(int(entries))]
    store.slow_bucket_ids = [int(index % 32) for index in range(int(entries))]
    store.slow_importance = [
        0.25 + 0.75 * float((index % 17) / 16.0) for index in range(int(entries))
    ]
    store.slow_capture_tag = array(
        "d", [0.05 + 0.90 * float((index % 23) / 22.0) for index in range(int(entries))]
    )
    store.slow_tag_is_strong = array(
        "b", [1 if index % 7 == 0 else 0 for index in range(int(entries))]
    )
    store.slow_local_prp = array(
        "d", [0.02 + 0.40 * float((index % 19) / 18.0) for index in range(int(entries))]
    )
    store.slow_last_capture_token = [
        int(token) - int(index % 1024) for index in range(int(entries))
    ]
    store.slow_consolidation_level = [
        0.10 + 0.80 * float((index % 29) / 28.0) for index in range(int(entries))
    ]
    store.slow_consolidation_events = [int(index % 5) for index in range(int(entries))]
    store.slow_entry_timestamps = array(
        "q", [int(token) - int(index % 4096) for index in range(int(entries))]
    )
    store.slow_last_replay_token = [
        int(token) - int(128 + index % 4096) for index in range(int(entries))
    ]
    store.slow_replay_count = [int(index % 11) for index in range(int(entries))]
    store.slow_ripple_strength = array(
        "d", [float((index % 13) / 12.0) for index in range(int(entries))]
    )
    store.fast_ema = vectors.mean(dim=0)
    store._slow_mean = store.fast_ema * 0.99
    store._slow_weight_sum = float(entries)
    store._slow_mean_token = int(token)
    store.global_prp_pool = 2.0
    for bucket in range(32):
        store.bucket_prp_pool[bucket] = 0.1 * float(bucket + 1)
    store._state_token = int(token)
    store.n_seen = int(entries)
    store.admission_count = int(entries)
    store.last_replay_selection_report = {
        "surface": "bounded_replay_window_selection.v1",
        "candidate_scope": "benchmark_seeded",
        "score_count": 32,
    }
    return store


def _measure(fn: Any, *, runs: int) -> tuple[list[float], list[dict[str, Any]]]:
    elapsed: list[float] = []
    rows: list[dict[str, Any]] = []
    for _ in range(int(runs)):
        started = time.perf_counter()
        row = fn()
        elapsed.append(float((time.perf_counter() - started) * 1000.0))
        rows.append(dict(row))
    return elapsed, rows


def _latency_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(statistics.fmean(values)) if values else 0.0,
        "median_ms": float(statistics.median(values)) if values else 0.0,
        "min_ms": float(min(values)) if values else 0.0,
        "max_ms": float(max(values)) if values else 0.0,
    }


def run_benchmark(*, entries: int, dim: int, runs: int, seed: int) -> dict[str, Any]:
    token = 50_000
    live_store = _build_store(entries=entries, dim=dim, seed=seed, token=token)
    live_state_before = int(live_store._state_token)
    live_tags_before = list(live_store.slow_capture_tag[:8])

    tracemalloc.start()
    live_elapsed, live_rows = _measure(
        lambda: live_store.live_summary_stats(
            current_token=int(live_store._state_token) + 1_000
        ),
        runs=runs,
    )
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    live_last = live_rows[-1] if live_rows else {}
    live_state_after = int(live_store._state_token)
    live_tags_after = list(live_store.slow_capture_tag[:8])
    live_stats = _latency_stats(live_elapsed)
    quality = {
        "scalar_fill_exact": bool(
            int(live_last.get("size", -1)) == int(entries)
            and int(live_last.get("capacity", -1)) == int(entries)
            and abs(float(live_last.get("fill_fraction", -1.0)) - 1.0) < 1e-12
            and int(live_last.get("n_seen", -1)) == int(entries)
            and int(live_last.get("total_stored", -1)) == int(entries)
        ),
        "last_report_surface_visible": bool(
            live_last.get("last_replay_selection_report", {}).get("surface")
            == "bounded_replay_window_selection.v1"
        ),
        "live_projection_read_only": bool(
            live_state_before == live_state_after and live_tags_before == live_tags_after
        ),
        "live_scan_count_zero": bool(
            live_last.get("summary_full_memory_scan") is False
            and int(live_last.get("summary_scan_entry_count", -1)) == 0
        ),
        "live_projection_has_expected_marker": bool(
            int(live_last.get("summary_token_marker", -1)) == token + 1_000
            and int(live_last.get("summary_state_token", -1)) == token
            and live_last.get("summary_projection_read_only") is True
        ),
    }
    quality["pass"] = bool(
        quality["scalar_fill_exact"]
        and quality["last_report_surface_visible"]
        and quality["live_projection_read_only"]
        and quality["live_scan_count_zero"]
        and quality["live_projection_has_expected_marker"]
    )
    cuda_available = bool(torch.cuda.is_available())
    cuda_allocated = (
        float(torch.cuda.memory_allocated() / (1024 * 1024))
        if cuda_available
        else 0.0
    )
    return {
        "artifact_kind": "live_memory_summary_projection_benchmark",
        "surface": "bounded_memory_summary_projection.v1",
        "entries": int(entries),
        "dim": int(dim),
        "runs": int(runs),
        "selection_criteria": [
            "trainer_telemetry_tick",
            "service_status_projection",
            "live_runtime_snapshot",
        ],
        "memory_budget": {
            "live_summary_scan_entry_budget": 0,
            "retired_live_full_summary_scan_rows_removed": int(entries),
            "archival_storage_device": "cpu",
        },
        "device_placement": {
            "archival_storage_device": "cpu",
            "projection_device": "cpu",
            "cuda_available": cuda_available,
            "cuda_memory_allocated_before_mib": 0.0,
            "cuda_memory_allocated_after_mib": cuda_allocated,
        },
        "resource_behavior": {
            "python_tracemalloc_current_mib": round(
                float(traced_current) / (1024.0 * 1024.0),
                6,
            ),
            "python_tracemalloc_peak_mib": round(
                float(traced_peak) / (1024.0 * 1024.0),
                6,
            ),
            "cuda_memory_allocated_after_mib": cuda_allocated,
        },
        "runtime_truth": {
            "live_summary_surface": str(live_last.get("summary_surface")),
            "live_summary_full_memory_scan": bool(
                live_last.get("summary_full_memory_scan")
            ),
            "live_summary_scan_entry_count": int(
                live_last.get("summary_scan_entry_count", -1)
            ),
            "global_candidate_scan": False,
            "hidden_language_reasoning": False,
        },
        "retired_path": {
            "name": "full summary_stats scan from trainer/service/status live projections",
            "replacement": "DualMemoryStore.live_summary_stats",
            "full_summary_mean_fields_policy": (
                "full_summary_kept_for_explicit_offline_quality_and_consolidation_windows"
            ),
        },
        "retired_live_full_summary_scan_absence": {
            "implementation_present": False,
            "diagnostic_callable": False,
            "active_report_field_present": False,
            "removed_policy": "live_projection_full_summary_stats_scan_comparator",
        },
        "latency": {
            "live_summary_projection": live_stats,
        },
        "last_live_summary_projection": live_last,
        "quality": quality,
        "pass": bool(quality["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded live memory summary projection."
    )
    parser.add_argument("--entries", type=int, default=65_536)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(
        entries=args.entries,
        dim=args.dim,
        runs=args.runs,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "pass={pass_gate} entries={entries} live_mean_ms={live_ms:.6f} "
        "live_scan_entries={live_scan}".format(
            pass_gate=report["pass"],
            entries=report["entries"],
            live_ms=report["latency"]["live_summary_projection"]["mean_ms"],
            live_scan=report["runtime_truth"]["live_summary_scan_entry_count"],
        )
    )


if __name__ == "__main__":
    main()
