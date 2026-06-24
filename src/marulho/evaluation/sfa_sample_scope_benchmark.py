from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any

import torch

from marulho.consolidation.memory_store import DualMemoryStore
from marulho.reporting.io import write_json_file


def _populate_store(*, capacity: int, vector_dim: int, candidate_count: int) -> DualMemoryStore:
    store = DualMemoryStore(capacity=max(1, int(capacity)))
    dim = max(1, int(vector_dim))
    size = max(1, int(capacity))
    bucket_mod = max(1, int(candidate_count))
    base = torch.eye(dim, dtype=torch.float32)
    store.slow_buffer = [
        base[index % dim].detach().clone()
        for index in range(size)
    ]
    store.slow_input_patterns = [item.detach().clone() for item in store.slow_buffer]
    store.slow_routing_keys = [None for _ in range(size)]
    store.slow_raw_windows = [None for _ in range(size)]
    store.slow_texts = [None for _ in range(size)]
    store.slow_metadata = [None for _ in range(size)]
    store.slow_bucket_ids = [index % bucket_mod for index in range(size)]
    store.slow_importance = [1.0 for _ in range(size)]
    store.slow_capture_tag = array("d", [0.0 for _ in range(size)])
    store.slow_tag_is_strong = array("b", [False for _ in range(size)])
    store.slow_local_prp = array("d", [0.0 for _ in range(size)])
    store.slow_last_capture_token = [0 for _ in range(size)]
    store.slow_consolidation_level = [0.0 for _ in range(size)]
    store.slow_consolidation_events = [0 for _ in range(size)]
    store.slow_entry_timestamps = array("q", [index for index in range(size)])
    store.slow_last_replay_token = [0 for _ in range(size)]
    store.slow_replay_count = [0 for _ in range(size)]
    store.slow_ripple_strength = array("d", [0.0 for _ in range(size)])
    store.n_seen = size
    store.admission_count = size
    store.update_calls = size
    bucket_index: defaultdict[int, list[int]] = defaultdict(list)
    for index, bucket_id in enumerate(store.slow_bucket_ids):
        bucket_index[int(bucket_id)].append(int(index))
    store._bucket_entry_indices = bucket_index
    store._recent_entry_indices = [(index, index) for index in range(size)]
    store._invalidate_summary_cache()
    return store


def _selected_window_purity(sample_indices: list[int], candidate_set: set[int]) -> float:
    if not sample_indices:
        return 0.0
    selected = sum(1 for index in sample_indices if int(index) in candidate_set)
    return float(selected / max(1, len(sample_indices)))


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    capacity = max(1, int(args.capacity))
    candidate_count = max(1, min(int(args.candidate_count), capacity))
    candidate_indices = list(range(candidate_count))
    candidate_set = set(candidate_indices)
    store = _populate_store(
        capacity=capacity,
        vector_dim=int(args.vector_dim),
        candidate_count=candidate_count,
    )
    cuda_available = bool(torch.cuda.is_available())
    cuda_before = int(torch.cuda.memory_allocated()) if cuda_available else 0

    bounded_latencies: list[float] = []
    bounded_purities: list[float] = []
    bounded_report: dict[str, Any] = {}
    bounded_sample_indices: list[int] = []

    for iteration in range(max(1, int(args.iterations))):
        torch.manual_seed(int(args.seed) + iteration)
        samples, report = store.sample_for_sfa_with_report(
            n=int(args.sample_count),
            candidate_indices=candidate_indices,
            scope="benchmark_selected_replay_window_sfa",
        )
        bounded_report = dict(report)
        bounded_latencies.append(float(report["latency_ms"]))
        bounded_sample_indices = [int(index) for index in report["sample_indices"]]
        bounded_purities.append(
            _selected_window_purity(bounded_sample_indices, candidate_set)
        )
        if len(samples) != int(report["sample_count"]):
            raise AssertionError("SFA report sample count drifted from returned samples")

    cuda_after = int(torch.cuda.memory_allocated()) if cuda_available else 0
    bounded_mean = statistics.fmean(bounded_latencies)
    bounded_p95 = sorted(bounded_latencies)[
        min(len(bounded_latencies) - 1, int(0.95 * (len(bounded_latencies) - 1)))
    ]
    bounded_purity = statistics.fmean(bounded_purities)
    quality_gate_pass = bool(
        bounded_purity >= float(args.min_selected_window_purity)
        and set(bounded_sample_indices).issubset(candidate_set)
    )
    report_gate_pass = bool(
        bounded_report.get("surface") == "bounded_sfa_sample.v1"
        and bounded_report.get("candidate_scope") == "selected_replay_window"
        and not bool(bounded_report.get("global_candidate_scan"))
        and not bool(bounded_report.get("runs_live_tick"))
        and not bool(bounded_report.get("runs_every_token"))
        and not bool(bounded_report.get("language_reasoning"))
        and bounded_report.get("archival_storage_device") == "cpu"
        and bounded_report.get("sample_device") == "cpu"
    )
    latency_gate_pass = bool(bounded_mean <= float(args.max_bounded_mean_ms))
    return {
        "surface": "bounded_sfa_sample_scope_benchmark.v1",
        "passed": bool(quality_gate_pass and report_gate_pass and latency_gate_pass),
        "capacity": int(capacity),
        "candidate_count": int(candidate_count),
        "sample_count": int(args.sample_count),
        "iterations": int(args.iterations),
        "selection_criteria": "selected_replay_window_indices_only",
        "memory_budget_entries": int(capacity),
        "candidate_window_entries": int(candidate_count),
        "retired_full_buffer_sfa_sample_absence": {
            "implementation_present": False,
            "diagnostic_callable": False,
            "active_report_field_present": False,
            "removed_policy": "sfa_full_buffer_sample_comparator",
        },
        "memory_budget": {
            "archival_entries": int(capacity),
            "candidate_window_entries": int(candidate_count),
            "sample_count": int(args.sample_count),
            "retired_full_buffer_sample_rows_removed": int(capacity),
        },
        "quality": {
            "metric": "selected_window_sample_purity",
            "bounded_mean": float(bounded_purity),
            "selected_sample_indices": bounded_sample_indices,
            "min_selected_window_purity": float(args.min_selected_window_purity),
        },
        "latency_ms": {
            "bounded_mean": float(bounded_mean),
            "bounded_p95": float(bounded_p95),
            "bounded_min": float(min(bounded_latencies)),
            "max_bounded_mean_ms": float(args.max_bounded_mean_ms),
        },
        "bounded_report": bounded_report,
        "device_placement": {
            "archival_storage_device": bounded_report.get("archival_storage_device"),
            "sample_device": bounded_report.get("sample_device"),
            "active_replay_cuda_required": False,
            "cuda_available": cuda_available,
            "cuda_memory_delta_mib": float(
                (cuda_after - cuda_before) / (1024.0 * 1024.0)
            ),
        },
        "gates": {
            "quality_gate_pass": quality_gate_pass,
            "report_gate_pass": report_gate_pass,
            "latency_gate_pass": latency_gate_pass,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=65536)
    parser.add_argument("--vector-dim", type=int, default=64)
    parser.add_argument("--candidate-count", type=int, default=192)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-selected-window-purity", type=float, default=1.0)
    parser.add_argument("--max-bounded-mean-ms", type=float, default=10.0)
    args = parser.parse_args()

    result = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(args.output, result)
    print(
        json.dumps(
            {
                "passed": bool(result["passed"]),
                "quality": result["quality"],
                "bounded_mean_ms": result["latency_ms"]["bounded_mean"],
                "bounded_report": {
                    key: result["bounded_report"].get(key)
                    for key in (
                        "surface",
                        "candidate_scope",
                        "candidate_index_count",
                        "sample_count",
                        "global_candidate_scan",
                        "runs_live_tick",
                    )
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
