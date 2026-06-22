from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from marulho.training import query_runner


class _RoutingIndex:
    def search_tensors(
        self,
        queries: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = max(1, int(k))
        ids = torch.zeros((1, count), dtype=torch.long)
        scores = torch.ones((1, count), dtype=torch.float32)
        return ids, scores


class _MemoryStore:
    def __init__(self, capacity: int) -> None:
        self._texts = [
            "routed bucket has no requested query term.",
            *[
                f"needle support memory outside routed bucket {index}."
                for index in range(1, capacity)
            ],
        ]
        self.slow_buffer = [
            torch.tensor([1.0, 0.0], dtype=torch.float32)
            for _ in range(capacity)
        ]
        self.slow_input_patterns = [item.clone() for item in self.slow_buffer]
        self.slow_routing_keys = [item.clone() for item in self.slow_buffer]
        self.slow_raw_windows = list(self._texts)
        self.slow_metadata = [{} for _ in self._texts]
        self.slow_bucket_ids = [0, *([1] * max(0, capacity - 1))]
        self.slow_importance = [1.0 for _ in self._texts]
        self.slow_entry_timestamps = list(range(capacity))
        self.slow_replay_count = [0 for _ in self._texts]
        self.query_match_row_calls: list[tuple[int, bool]] = []
        self.recent_collector_call_count = 0
        self.recent_collector_indices: list[int] = []
        self.last_query_memory_match_report: dict[str, Any] = {}

    def replay_scores_for_indices(
        self,
        indices: list[int],
        token_count: int,
    ) -> dict[int, float]:
        return {int(index): 0.0 for index in indices}

    def collect_query_memory_match_indices(
        self,
        *,
        candidate_bucket_ids: list[int] | torch.Tensor | None,
        max_candidates: int,
        scope: str = "query_memory_match_slow_path",
    ) -> dict[str, Any]:
        buckets = (
            []
            if candidate_bucket_ids is None
            else [int(value) for value in candidate_bucket_ids]
        )
        bucket_set = set(buckets)
        candidates = [
            index
            for index, bucket_id in enumerate(self.slow_bucket_ids)
            if int(bucket_id) in bucket_set
        ][: max(0, int(max_candidates))]
        report = {
            "surface": "bounded_query_memory_match_candidates.v1",
            "status": "collected" if candidates else "empty",
            "scope": scope,
            "memory_size": len(self.slow_buffer),
            "requested_count": int(max_candidates),
            "candidate_window_limit": int(max_candidates),
            "candidate_window_policy": "recent_bucket_round_robin_candidate_pool",
            "candidate_scope": "bucket_indexed_candidate_window",
            "candidate_bucket_ids": buckets,
            "candidate_bucket_count": len(buckets),
            "candidate_index_available_count": len(candidates),
            "candidate_index_count": len(candidates),
            "match_indices": candidates,
            "score_count": 0,
            "global_score_scan": False,
            "global_candidate_scan": False,
            "runs_live_tick": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "fallback_reason": None if candidates else "empty_query_candidate_window",
        }
        self.last_query_memory_match_report = report
        return report

    def collect_recent_entry_indices(
        self,
        *,
        current_token: int,
        window_tokens: int,
        max_entries: int = 256,
        require_bucket: bool = False,
        scope: str = "recent_memory_slow_path",
    ) -> dict[str, Any]:
        self.recent_collector_call_count += 1
        limit = max(0, int(max_entries))
        indices = list(range(len(self.slow_buffer) - 1, 0, -1))[:limit]
        self.recent_collector_indices = indices
        return {
            "surface": "bounded_recent_memory_window.v1",
            "status": "collected" if indices else "empty",
            "scope": scope,
            "current_token": int(current_token),
            "window_tokens": int(window_tokens),
            "requested_count": int(limit),
            "candidate_window_policy": "recent_entry_index_reverse_window",
            "candidate_scope": (
                "bucketed_recent_entry_index_window"
                if require_bucket
                else "recent_entry_index_window"
            ),
            "candidate_indices": indices,
            "candidate_index_count": len(indices),
            "requires_bucket": bool(require_bucket),
            "global_candidate_scan": False,
            "global_score_scan": False,
            "runs_live_tick": False,
            "runs_every_token": False,
            "archival_storage_device": "cpu",
        }

    def query_match_row(
        self,
        idx: int,
        current_token: int | None = None,
        *,
        include_text_payload: bool = False,
    ) -> dict[str, Any]:
        index = int(idx)
        self.query_match_row_calls.append((index, bool(include_text_payload)))
        return {
            "surface": "bounded_query_memory_match_row.v1",
            "memory_index": index,
            "assembly": self.slow_buffer[index],
            "input_pattern": self.slow_input_patterns[index],
            "routing_key": self.slow_routing_keys[index],
            "bucket_id": self.slow_bucket_ids[index],
            "importance": self.slow_importance[index],
            "capture_tag": 0.0,
            "tag_strength": 0.0,
            "prp_level": 0.0,
            "capture_strength": 0.0,
            "consolidation_level": 0.0,
            "age_tokens": max(
                0,
                int(current_token or 0) - int(self.slow_entry_timestamps[index]),
            ),
            "replay_count": self.slow_replay_count[index],
            "text": self._texts[index] if include_text_payload else None,
            "raw_window": self.slow_raw_windows[index] if include_text_payload else None,
            "metadata": {} if include_text_payload else None,
            "raw_text_payload_loaded": bool(include_text_payload),
            "global_candidate_scan": False,
            "global_score_scan": False,
            "runs_live_tick": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
        }


class _Trainer:
    def __init__(self, capacity: int) -> None:
        self.token_count = capacity
        self.config = SimpleNamespace(input_representation="hashed_ngram", k_routing=1)
        self.model = SimpleNamespace(
            memory_store=_MemoryStore(capacity),
            routing_index=_RoutingIndex(),
        )


def _stats(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"count": 0, "mean_ms": 0.0, "median_ms": 0.0, "max_ms": 0.0}
    return {
        "count": len(samples),
        "mean_ms": round(float(statistics.mean(samples)), 6),
        "median_ms": round(float(statistics.median(samples)), 6),
        "max_ms": round(float(max(samples)), 6),
        "samples_ms": [round(float(value), 6) for value in samples],
    }


def _cuda_report() -> dict[str, Any]:
    available = torch.cuda.is_available()
    return {
        "torch_available": True,
        "cuda_available": bool(available),
        "gpu_used": False,
        "memory_allocated_mib": (
            round(float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0), 3)
            if available
            else 0.0
        ),
        "memory_reserved_mib": (
            round(float(torch.cuda.memory_reserved()) / (1024.0 * 1024.0), 3)
            if available
            else 0.0
        ),
    }


def run_benchmark(
    *,
    capacity: int,
    candidate_limit: int,
    top_k: int,
    iterations: int,
) -> dict[str, Any]:
    latency_samples: list[float] = []
    last_matches: list[dict[str, Any]] = []
    last_report: dict[str, Any] = {}
    last_store: _MemoryStore | None = None
    pattern = torch.tensor([1.0, 0.0], dtype=torch.float32)

    tracemalloc.start()
    for _ in range(max(1, int(iterations))):
        trainer = _Trainer(max(2, int(capacity)))
        started = time.perf_counter()
        matches, report = query_runner.memory_matches_with_report(
            trainer,
            pattern,
            pattern,
            top_k=max(1, int(top_k)),
            top_chars=1,
            query_terms=["needle"],
            memory_candidate_limit=max(1, int(candidate_limit)),
            candidate_bucket_ids=[0],
        )
        latency_samples.append((time.perf_counter() - started) * 1000.0)
        last_matches = matches
        last_report = report
        last_store = trainer.model.memory_store
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    returned_indices = [
        int(item.get("memory_index", -1))
        for item in last_matches
        if isinstance(item, dict)
    ]
    report_keys = set(last_report)
    old_recent_limit = min(max(1, int(candidate_limit)), max(32, max(1, int(top_k)) * 4))
    diagnostic_old_recent_source_count = min(
        max(0, int(capacity) - 1),
        old_recent_limit,
    )
    recent_call_count = (
        0 if last_store is None else int(last_store.recent_collector_call_count)
    )
    query_text_payload_rows = (
        []
        if last_store is None
        else [
            int(index)
            for index, include_text in last_store.query_match_row_calls
            if include_text
        ]
    )

    pass_checks = {
        "surface": last_report.get("surface") == "bounded_query_memory_match.v1",
        "candidate_scope_bucket_only": last_report.get("candidate_scope")
        == "bucket_indexed_candidate_window",
        "candidate_window_policy_retained": last_report.get("candidate_window_policy")
        == "recent_bucket_round_robin_candidate_pool",
        "recent_collector_not_called": recent_call_count == 0,
        "recent_fallback_fields_absent": not any(
            str(key).startswith("recent_fallback") for key in report_keys
        ),
        "returned_inside_bucket_window": returned_indices == [0],
        "raw_text_loaded_only_for_candidate": query_text_payload_rows == [0],
        "global_candidate_scan_false": last_report.get("global_candidate_scan") is False,
        "global_score_scan_false": last_report.get("global_score_scan") is False,
        "runs_live_tick_false": last_report.get("runs_live_tick") is False,
        "language_reasoning_false": last_report.get("language_reasoning") is False,
        "cpu_archival": last_report.get("archival_storage_device") == "cpu",
    }

    return {
        "surface": "query_recent_fallback_retirement_benchmark.v1",
        "pass": all(pass_checks.values()),
        "pass_checks": pass_checks,
        "input": {
            "capacity": int(capacity),
            "candidate_limit": int(candidate_limit),
            "top_k": int(top_k),
            "iterations": int(iterations),
            "candidate_bucket_ids": [0],
            "query_terms": ["needle"],
        },
        "quality": {
            "metric": "bucket_indexed_query_window_does_not_widen_to_recent_entries",
            "returned_indices": returned_indices,
            "candidate_indices": list(last_report.get("match_indices") or []),
            "candidate_index_count": int(last_report.get("candidate_index_count", 0) or 0),
            "query_term_count": int(last_report.get("query_term_count", 0) or 0),
            "returned_query_overlap": [
                int(item.get("query_overlap", 0))
                for item in last_matches
                if isinstance(item, dict)
            ],
        },
        "retired_path": {
            "name": "query_recent_entry_text_support_fallback",
            "production_callable": False,
            "old_scope": "query_runner_memory_match_recent_text_support_fallback",
            "old_candidate_scope": "bucket_indexed_plus_recent_entry_candidate_window",
            "diagnostic_old_recent_source_count": int(diagnostic_old_recent_source_count),
            "diagnostic_old_recent_source_device": "cpu",
            "replacement": "bounded_query_memory_match bucket-indexed candidate window",
        },
        "latency_ms": {
            "active_bucket_only_query": _stats(latency_samples),
        },
        "resource_behavior": {
            "python_tracemalloc_current_mib": round(
                float(traced_current) / (1024.0 * 1024.0),
                3,
            ),
            "python_tracemalloc_peak_mib": round(
                float(traced_peak) / (1024.0 * 1024.0),
                3,
            ),
            "cuda": _cuda_report(),
        },
        "device_placement": {
            "archival_storage_device": "cpu",
            "source_window_selection_device": "cpu",
            "score_device": "cpu",
            "gpu_used": False,
            "gpu_resident_archival_metadata": False,
        },
        "report": last_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark retirement of query recent-entry fallback.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=65536)
    parser.add_argument("--candidate-limit", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=11)
    args = parser.parse_args()

    report = run_benchmark(
        capacity=args.capacity,
        candidate_limit=args.candidate_limit,
        top_k=args.top_k,
        iterations=args.iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
