from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
import json
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any

import torch

from marulho.consolidation.memory_store import DualMemoryStore


def _seed_store(*, entries: int, buckets: int, dim: int = 4) -> DualMemoryStore:
    store = DualMemoryStore(
        capacity=int(entries),
        ema_alpha=0.1,
        slow_mean_decay=1.0,
        capture_tag_decay=1.0,
        capture_release=0.5,
        consolidation_rate=1.0,
        prp_synthesis_rate=0.5,
    )
    count = int(entries)
    bucket_count = max(1, int(buckets))
    vector_templates = [
        torch.tensor(
            [
                float(index % 17) / 17.0,
                float(index % 31) / 31.0,
                float(index % 7) / 7.0,
                1.0,
            ][:dim],
            dtype=torch.float32,
        )
        for index in range(64)
    ]
    store.slow_buffer = [
        vector_templates[index % len(vector_templates)]
        for index in range(count)
    ]
    store.slow_input_patterns = [None] * count
    store.slow_routing_keys = [None] * count
    store.slow_raw_windows = [None] * count
    store.slow_texts = [None] * count
    store.slow_metadata = [None] * count
    store.slow_bucket_ids = [index % bucket_count for index in range(count)]
    store.slow_importance = [1.0 + float(index % 5) for index in range(count)]
    store.slow_capture_tag = array("d", (1.0 for _ in range(count)))
    store.slow_tag_is_strong = array("b", (True for _ in range(count)))
    store.slow_local_prp = array("d", (1.0 for _ in range(count)))
    store.slow_last_capture_token = [int(index) for index in range(count)]
    store.slow_consolidation_level = [
        0.10 + 0.05 * float(index % 5) for index in range(count)
    ]
    store.slow_consolidation_events = [0 for _ in range(count)]
    store.slow_entry_timestamps = array("q", (int(index) for index in range(count)))
    store.slow_last_replay_token = [0 for _ in range(count)]
    store.slow_replay_count = [0 for _ in range(count)]
    store.slow_ripple_strength = array("d", (0.0 for _ in range(count)))
    store.global_prp_pool = 10.0
    store.bucket_prp_pool = defaultdict(float)
    store._state_token = count
    store.n_seen = count
    store.update_calls = count
    store.admission_count = count
    store._invalidate_bucket_consolidation_cache()
    return store


def _selected_indices(*, entries: int, width: int) -> list[int]:
    count = max(0, min(int(entries), int(width)))
    if count <= 0:
        return []
    stride = max(1, int(entries) // count)
    return [min(int(entries) - 1, index * stride) for index in range(count)]


def _latency(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(statistics.fmean(values)) if values else 0.0,
        "median_ms": float(statistics.median(values)) if values else 0.0,
        "min_ms": float(min(values)) if values else 0.0,
        "max_ms": float(max(values)) if values else 0.0,
    }


def _selected_state(store: DualMemoryStore, indices: list[int]) -> dict[str, Any]:
    return {
        "consolidation_level": [
            float(store.slow_consolidation_level[index]) for index in indices
        ],
        "replay_count": [int(store.slow_replay_count[index]) for index in indices],
        "consolidation_events": [
            int(store.slow_consolidation_events[index]) for index in indices
        ],
        "capture_tag": [float(store.slow_capture_tag[index]) for index in indices],
    }


def _max_abs_delta(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return float("inf")
    if not left:
        return 0.0
    return float(max(abs(float(a) - float(b)) for a, b in zip(left, right)))


def _run_bounded(
    *,
    entries: int,
    buckets: int,
    indices: list[int],
) -> tuple[float, DualMemoryStore, dict[str, Any]]:
    store = _seed_store(entries=entries, buckets=buckets)
    started = time.perf_counter()
    report = store.consolidate_replay(
        indices,
        current_token=int(entries) + 1,
        blend=0.4,
        protein_synthesis_level=1.1,
    )
    elapsed_ms = float((time.perf_counter() - started) * 1000.0)
    return elapsed_ms, store, report


def _run_retired_diagnostic(
    *,
    entries: int,
    buckets: int,
    indices: list[int],
) -> tuple[float, DualMemoryStore, dict[str, Any]]:
    store = _seed_store(entries=entries, buckets=buckets)
    started = time.perf_counter()
    store._rebuild_bucket_consolidation_cache(
        n_buckets=int(buckets),
        reason="diagnostic_retired_selected_replay_window",
    )
    report = store.consolidate_replay(
        indices,
        current_token=int(entries) + 1,
        blend=0.4,
        protein_synthesis_level=1.1,
    )
    elapsed_ms = float((time.perf_counter() - started) * 1000.0)
    return elapsed_ms, store, report


def run_benchmark(
    *,
    entries: int,
    buckets: int,
    selected_count: int,
    runs: int,
) -> dict[str, Any]:
    selected = _selected_indices(entries=entries, width=selected_count)
    bounded_elapsed: list[float] = []
    retired_elapsed: list[float] = []
    bounded_store: DualMemoryStore | None = None
    retired_store: DualMemoryStore | None = None
    bounded_report: dict[str, Any] = {}
    retired_report: dict[str, Any] = {}
    cuda_available = bool(torch.cuda.is_available())
    cuda_allocated_before = (
        float(torch.cuda.memory_allocated() / (1024 * 1024))
        if cuda_available
        else 0.0
    )
    cuda_reserved_before = (
        float(torch.cuda.memory_reserved() / (1024 * 1024))
        if cuda_available
        else 0.0
    )
    for _ in range(int(runs)):
        elapsed, bounded_store, bounded_report = _run_bounded(
            entries=entries,
            buckets=buckets,
            indices=selected,
        )
        bounded_elapsed.append(elapsed)
        elapsed, retired_store, retired_report = _run_retired_diagnostic(
            entries=entries,
            buckets=buckets,
            indices=selected,
        )
        retired_elapsed.append(elapsed)

    if bounded_store is None or retired_store is None:
        raise RuntimeError("benchmark did not run")

    peak_store = _seed_store(entries=entries, buckets=buckets)
    tracemalloc.start()
    try:
        peak_store.consolidate_replay(
            selected,
            current_token=int(entries) + 1,
            blend=0.4,
            protein_synthesis_level=1.1,
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    bounded_state = _selected_state(bounded_store, selected)
    retired_state = _selected_state(retired_store, selected)
    quality = {
        "selected_consolidation_level_max_delta": _max_abs_delta(
            bounded_state["consolidation_level"],
            retired_state["consolidation_level"],
        ),
        "selected_capture_tag_max_delta": _max_abs_delta(
            bounded_state["capture_tag"],
            retired_state["capture_tag"],
        ),
        "selected_replay_counts_match": bool(
            bounded_state["replay_count"] == retired_state["replay_count"]
        ),
        "selected_consolidation_events_match": bool(
            bounded_state["consolidation_events"]
            == retired_state["consolidation_events"]
        ),
        "selected_fast_ema_matches": bool(
            isinstance(bounded_store.fast_ema, torch.Tensor)
            and isinstance(retired_store.fast_ema, torch.Tensor)
            and torch.allclose(bounded_store.fast_ema, retired_store.fast_ema)
        ),
        "bounded_path_no_cache_rebuild": bool(
            int(bounded_report.get("cache_rebuild_count_delta", -1)) == 0
            and int(bounded_report.get("cache_rebuild_scan_entry_count", -1)) == 0
        ),
        "bounded_path_no_full_memory_scan": bool(
            bounded_report.get("full_memory_scan") is False
            and int(bounded_report.get("scan_entry_count", -1)) == 0
        ),
    }
    quality["selected_state_matches_retired_diagnostic"] = bool(
        quality["selected_consolidation_level_max_delta"] <= 1e-6
        and quality["selected_capture_tag_max_delta"] <= 1e-6
        and quality["selected_replay_counts_match"]
        and quality["selected_consolidation_events_match"]
        and quality["selected_fast_ema_matches"]
    )
    bounded_stats = _latency(bounded_elapsed)
    retired_stats = _latency(retired_elapsed)
    quality["mean_latency_reduction_ms"] = float(
        max(0.0, retired_stats["mean_ms"] - bounded_stats["mean_ms"])
    )
    quality["pass"] = bool(
        quality["selected_state_matches_retired_diagnostic"]
        and quality["bounded_path_no_cache_rebuild"]
        and quality["bounded_path_no_full_memory_scan"]
        and bounded_stats["mean_ms"] < retired_stats["mean_ms"]
    )
    cuda_allocated_after = (
        float(torch.cuda.memory_allocated() / (1024 * 1024))
        if cuda_available
        else 0.0
    )
    cuda_reserved_after = (
        float(torch.cuda.memory_reserved() / (1024 * 1024))
        if cuda_available
        else 0.0
    )
    selected_valid = max(1, int(bounded_report.get("selected_valid_index_count", 0)))
    return {
        "artifact_kind": "selected_replay_consolidation_cache_benchmark",
        "surface": "bounded_selected_replay_consolidation.v1",
        "entries": int(entries),
        "buckets": int(buckets),
        "selected_count": int(selected_count),
        "selected_indices": selected,
        "runs": int(runs),
        "selection_criteria": [
            "selected_replay_window_indices",
            "sleep_or_replay_slow_path_only",
            "missing_bucket_cache_must_not_trigger_archive_rebuild",
        ],
        "memory_budget": {
            "selected_window_indices": int(selected_valid),
            "retained_entries": int(entries),
            "retired_rebuild_scan_entries": int(entries),
            "bounded_rebuild_scan_entries": 0,
            "work_reduction_vs_retired_rebuild": float(entries / selected_valid),
            "archival_storage_device": "cpu",
            "cache_metadata_device": "cpu",
        },
        "device_placement": {
            "archival_storage_device": "cpu",
            "selected_replay_compute_device": "cpu",
            "cache_metadata_device": "cpu",
            "gpu_used": False,
            "cuda_available": cuda_available,
            "cuda_memory_allocated_before_mib": cuda_allocated_before,
            "cuda_memory_allocated_after_mib": cuda_allocated_after,
            "cuda_memory_reserved_before_mib": cuda_reserved_before,
            "cuda_memory_reserved_after_mib": cuda_reserved_after,
            "python_tracemalloc_peak_mib": float(peak_bytes / (1024 * 1024)),
        },
        "runtime_truth": {
            "runs_live_tick": False,
            "runs_every_token": False,
            "full_memory_scan": False,
            "scan_entry_count": 0,
            "global_candidate_scan": False,
            "hidden_language_reasoning": False,
            "raw_text_payload_loaded": False,
            "mutates_runtime_state": True,
            "applies_plasticity": True,
            "gpu_used": False,
        },
        "latency": {
            "bounded_selected_replay_missing_cache": bounded_stats,
            "retired_diagnostic_full_cache_rebuild_then_replay": retired_stats,
        },
        "bounded_report": bounded_report,
        "retired_diagnostic_report": retired_report,
        "quality": quality,
        "pass": bool(quality["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark selected replay consolidation without full bucket-cache "
            "rebuild against the retired diagnostic rebuild path."
        )
    )
    parser.add_argument("--entries", type=int, default=65_536)
    parser.add_argument("--buckets", type=int, default=65_536)
    parser.add_argument("--selected-count", type=int, default=16)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(
        entries=args.entries,
        buckets=args.buckets,
        selected_count=args.selected_count,
        runs=args.runs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "pass={pass_gate} entries={entries} selected={selected} "
        "bounded_mean_ms={bounded_ms:.6f} retired_mean_ms={retired_ms:.6f} "
        "work_reduction={work_reduction:.1f}".format(
            pass_gate=report["pass"],
            entries=report["entries"],
            selected=report["selected_count"],
            bounded_ms=report["latency"]["bounded_selected_replay_missing_cache"][
                "mean_ms"
            ],
            retired_ms=report["latency"][
                "retired_diagnostic_full_cache_rebuild_then_replay"
            ]["mean_ms"],
            work_reduction=report["memory_budget"][
                "work_reduction_vs_retired_rebuild"
            ],
        )
    )


if __name__ == "__main__":
    main()
