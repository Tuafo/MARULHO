from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
from typing import Any, Sequence

import torch

from marulho.reporting.io import write_json_file
from marulho.training import query_runner


class _SyntheticTextSequence:
    def __init__(self, prefix: str, size: int) -> None:
        self.prefix = str(prefix)
        self.size = max(0, int(size))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> str:
        idx = int(index)
        if idx < 0 or idx >= self.size:
            raise IndexError(idx)
        return f"{self.prefix} {idx}"


class _SyntheticMemoryStore:
    def __init__(self, *, capacity: int, bucket_count: int) -> None:
        self.capacity = max(1, int(capacity))
        self.bucket_count = max(1, int(bucket_count))
        base = torch.tensor([1.0, 0.0], dtype=torch.float32)
        self.slow_buffer = [base for _ in range(self.capacity)]
        self.slow_input_patterns = [base for _ in range(self.capacity)]
        self.slow_routing_keys = [None for _ in range(self.capacity)]
        self.slow_raw_windows = _SyntheticTextSequence("query memory episode", self.capacity)
        self.slow_texts = self.slow_raw_windows
        self.slow_metadata = [{} for _ in range(self.capacity)]
        self.slow_bucket_ids = [int(index % self.bucket_count) for index in range(self.capacity)]
        self.slow_importance = [1.0 for _ in range(self.capacity)]
        self.slow_entry_timestamps = [0 for _ in range(self.capacity)]
        self.slow_replay_count = [0 for _ in range(self.capacity)]
        self.query_match_row_calls: list[tuple[int, bool]] = []
        self.last_query_memory_match_report: dict[str, Any] = {}

    def replay_scores_for_indices(
        self,
        indices: Sequence[int],
        token_count: int,
    ) -> dict[int, float]:
        return {int(index): 0.0 for index in indices}

    def collect_query_memory_match_indices(
        self,
        *,
        candidate_bucket_ids: Sequence[int] | torch.Tensor | None,
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
        report: dict[str, Any] = {
            "surface": "bounded_query_memory_match_candidates.v1",
            "status": "collected" if candidates else "empty",
            "scope": scope,
            "memory_size": int(len(self.slow_buffer)),
            "requested_count": int(max_candidates),
            "candidate_window_limit": int(max_candidates),
            "candidate_window_policy": "recent_bucket_round_robin_candidate_pool",
            "candidate_scope": "bucket_indexed_candidate_window",
            "candidate_bucket_ids": buckets,
            "candidate_bucket_count": int(len(buckets)),
            "candidate_index_available_count": int(
                sum(1 for bucket_id in self.slow_bucket_ids if int(bucket_id) in bucket_set)
            ),
            "candidate_index_count": int(len(candidates)),
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

    def query_match_row(
        self,
        idx: int,
        current_token: int | None = None,
        *,
        include_text_payload: bool = False,
    ) -> dict[str, Any]:
        index = int(idx)
        self.query_match_row_calls.append((index, bool(include_text_payload)))
        text = self.slow_raw_windows[index]
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
            "age_tokens": max(0, int(current_token or 0) - int(self.slow_entry_timestamps[index])),
            "replay_count": self.slow_replay_count[index],
            "raw_window": text if include_text_payload else None,
            "text": text if include_text_payload else None,
            "metadata": {} if include_text_payload else None,
            "raw_text_payload_loaded": bool(include_text_payload),
            "global_candidate_scan": False,
            "global_score_scan": False,
            "runs_live_tick": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
        }


def _trainer_for_store(store: _SyntheticMemoryStore) -> Any:
    return SimpleNamespace(
        token_count=1024,
        config=SimpleNamespace(input_representation="hashed_ngram", k_routing=8),
        model=SimpleNamespace(memory_store=store),
    )


def run_benchmark(
    *,
    capacity: int,
    bucket_count: int,
    candidate_limit: int,
    top_k: int,
    iterations: int,
) -> dict[str, Any]:
    store = _SyntheticMemoryStore(capacity=capacity, bucket_count=bucket_count)
    trainer = _trainer_for_store(store)
    pattern = torch.tensor([1.0, 0.0], dtype=torch.float32)
    candidate_bucket_ids = [0]

    bounded_latencies: list[float] = []
    bounded_payload_counts: list[int] = []
    bounded_selected: list[int] = []
    bounded_report: dict[str, Any] = {}

    for _ in range(max(1, int(iterations))):
        store.query_match_row_calls.clear()
        started = time.perf_counter()
        bounded_matches, bounded_report = query_runner.memory_matches_with_report(
            trainer,
            pattern,
            pattern,
            top_k=top_k,
            top_chars=1,
            memory_candidate_limit=candidate_limit,
            candidate_bucket_ids=candidate_bucket_ids,
        )
        bounded_latencies.append((time.perf_counter() - started) * 1000.0)
        bounded_payload_counts.append(
            int(sum(1 for _index, include_text in store.query_match_row_calls if include_text))
        )
        bounded_selected = [
            int(match.get("memory_index", -1))
            for match in bounded_matches
        ]

    bounded_mean = statistics.fmean(bounded_latencies)
    quality_pass = bool(
        bounded_selected
        and all(int(index) % int(bucket_count) in candidate_bucket_ids for index in bounded_selected)
        and len(bounded_selected) <= int(top_k)
    )
    payload_gate_pass = bool(max(bounded_payload_counts or [0]) <= max(1, int(top_k)))
    report_gate_pass = bool(
        bounded_report.get("raw_text_payload_policy")
        == "returned_similarity_matches_only"
        and bounded_report.get("query_row_surface")
        == "bounded_query_memory_match_row.v1"
        and bool(bounded_report.get("query_row_reader_owned_by_store"))
        and bool(bounded_report.get("direct_slow_memory_array_reads_retired"))
        and not bool(bounded_report.get("global_candidate_scan"))
        and not bool(bounded_report.get("global_score_scan"))
        and not bool(bounded_report.get("language_reasoning"))
    )
    latency_gate_pass = bool(bounded_mean <= 500.0)

    return {
        "surface": "bounded_query_memory_payload_benchmark.v1",
        "capacity": int(capacity),
        "bucket_count": int(bucket_count),
        "candidate_bucket_ids": candidate_bucket_ids,
        "candidate_window_limit": int(candidate_limit),
        "top_k": int(top_k),
        "iterations": int(iterations),
        "retired_eager_query_payload_absence": {
            "implementation_present": False,
            "diagnostic_callable": False,
            "active_report_field_present": False,
            "removed_policy": "query_memory_eager_candidate_text_payload_comparator",
        },
        "memory_budget": {
            "archival_entries": int(capacity),
            "candidate_window_limit": int(candidate_limit),
            "returned_payload_limit": int(top_k),
        },
        "quality": {
            "metric": "returned_payload_selection_inside_candidate_window",
            "bounded_selected_indices": bounded_selected,
            "selected_indices_inside_candidate_window": quality_pass,
        },
        "latency": {
            "bounded_mean_ms": float(bounded_mean),
            "max_bounded_mean_ms": 500.0,
        },
        "payload": {
            "bounded_raw_text_payload_count_mean": float(statistics.fmean(bounded_payload_counts)),
            "bounded_report_raw_text_payload_count": int(
                bounded_report.get("raw_text_payload_count", 0) or 0
            ),
            "bounded_report_raw_text_payload_policy": str(
                bounded_report.get("raw_text_payload_policy", "")
            ),
        },
        "bounded_report": bounded_report,
        "device_placement": {
            "archival_storage_device": "cpu",
            "score_device": "cpu",
            "active_replay_cuda_required": False,
        },
        "gates": {
            "quality_pass": quality_pass,
            "payload_gate_pass": payload_gate_pass,
            "report_gate_pass": report_gate_pass,
            "latency_gate_pass": latency_gate_pass,
        },
        "passed": bool(
            quality_pass and payload_gate_pass and report_gate_pass and latency_gate_pass
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=65536)
    parser.add_argument("--bucket-count", type=int, default=16)
    parser.add_argument("--candidate-limit", type=int, default=192)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=16)
    args = parser.parse_args()

    result = run_benchmark(
        capacity=int(args.capacity),
        bucket_count=int(args.bucket_count),
        candidate_limit=int(args.candidate_limit),
        top_k=int(args.top_k),
        iterations=int(args.iterations),
    )
    write_json_file(args.output, result)
    print(
        json.dumps(
            {
                "passed": bool(result["passed"]),
                "quality": result["quality"],
                "payload": result["payload"],
                "bounded_mean_ms": result["latency"]["bounded_mean_ms"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
