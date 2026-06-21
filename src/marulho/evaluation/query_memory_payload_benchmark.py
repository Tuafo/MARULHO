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
        self.slow_bucket_ids = [int(index % self.bucket_count) for index in range(self.capacity)]
        self.slow_importance = [1.0 for _ in range(self.capacity)]
        self.slow_entry_timestamps = [0 for _ in range(self.capacity)]
        self.slow_replay_count = [0 for _ in range(self.capacity)]
        self.replay_entry_calls: list[int] = []
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

    def replay_entry(
        self,
        idx: int,
        current_token: int | None = None,
        *,
        include_text_payload: bool = False,
    ) -> dict[str, Any]:
        index = int(idx)
        self.replay_entry_calls.append(index)
        text = self.slow_raw_windows[index]
        if not include_text_payload:
            return {"text": None, "raw_window": None, "metadata": None}
        return {
            "text": text,
            "raw_window": text,
            "metadata": {},
        }


def _trainer_for_store(store: _SyntheticMemoryStore) -> Any:
    return SimpleNamespace(
        token_count=1024,
        config=SimpleNamespace(input_representation="hashed_ngram", k_routing=8),
        model=SimpleNamespace(memory_store=store),
    )


def _diagnostic_eager_payload(
    store: _SyntheticMemoryStore,
    *,
    candidate_bucket_ids: Sequence[int],
    candidate_limit: int,
    top_k: int,
) -> dict[str, Any]:
    store.replay_entry_calls.clear()
    started = time.perf_counter()
    candidate_report = store.collect_query_memory_match_indices(
        candidate_bucket_ids=candidate_bucket_ids,
        max_candidates=candidate_limit,
        scope="diagnostic_eager_query_payload",
    )
    candidate_indices = [int(index) for index in list(candidate_report.get("match_indices", []))]
    replay_scores = store.replay_scores_for_indices(candidate_indices, 1024)
    query_pattern = torch.tensor([1.0, 0.0], dtype=torch.float32)
    rows: list[dict[str, Any]] = []
    for index in candidate_indices:
        evidence_pattern = store.slow_input_patterns[int(index)].float()
        similarity = query_runner.cosine_similarity(query_pattern, evidence_pattern)
        entry = store.replay_entry(
            int(index),
            current_token=1024,
            include_text_payload=True,
        )
        raw_window = str(entry.get("raw_window", ""))
        text = str(entry.get("text", ""))
        complete_sentence, clipped_overlap = query_runner.episode_quality(
            text,
            raw_window,
        )
        rows.append(
            {
                "memory_index": int(index),
                "similarity": float(similarity),
                "text": text,
                "raw_window": raw_window,
                "complete_sentence": int(complete_sentence),
                "clipped_overlap": int(clipped_overlap),
                "replay_priority": float(replay_scores.get(int(index), 0.0)),
                "top_chars": query_runner.top_feature_details(
                    evidence_pattern,
                    1,
                    "hashed_ngram",
                ),
            }
        )
    rows.sort(key=lambda item: float(item["similarity"]), reverse=True)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "selected_indices": [int(item["memory_index"]) for item in rows[:top_k]],
        "latency_ms": float(latency_ms),
        "raw_text_payload_count": int(len(store.replay_entry_calls)),
        "candidate_report": candidate_report,
    }


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

    legacy_latencies: list[float] = []
    bounded_latencies: list[float] = []
    legacy_payload_counts: list[int] = []
    bounded_payload_counts: list[int] = []
    legacy_selected: list[int] = []
    bounded_selected: list[int] = []
    bounded_report: dict[str, Any] = {}

    for _ in range(max(1, int(iterations))):
        legacy = _diagnostic_eager_payload(
            store,
            candidate_bucket_ids=candidate_bucket_ids,
            candidate_limit=candidate_limit,
            top_k=top_k,
        )
        legacy_latencies.append(float(legacy["latency_ms"]))
        legacy_payload_counts.append(int(legacy["raw_text_payload_count"]))
        legacy_selected = list(legacy["selected_indices"])

        store.replay_entry_calls.clear()
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
        bounded_payload_counts.append(int(len(store.replay_entry_calls)))
        bounded_selected = [
            int(match.get("memory_index", -1))
            for match in bounded_matches
        ]

    legacy_mean = statistics.fmean(legacy_latencies)
    bounded_mean = statistics.fmean(bounded_latencies)
    quality_pass = bool(legacy_selected == bounded_selected)
    payload_gate_pass = bool(max(bounded_payload_counts or [0]) <= max(1, int(top_k)))
    report_gate_pass = bool(
        bounded_report.get("raw_text_payload_policy")
        == "returned_similarity_matches_only"
        and not bool(bounded_report.get("global_candidate_scan"))
        and not bool(bounded_report.get("global_score_scan"))
        and not bool(bounded_report.get("language_reasoning"))
    )
    latency_gate_pass = bool(bounded_mean <= legacy_mean)

    return {
        "surface": "bounded_query_memory_payload_benchmark.v1",
        "capacity": int(capacity),
        "bucket_count": int(bucket_count),
        "candidate_bucket_ids": candidate_bucket_ids,
        "candidate_window_limit": int(candidate_limit),
        "top_k": int(top_k),
        "iterations": int(iterations),
        "quality": {
            "metric": "selected_indices_match_diagnostic_eager_payload",
            "legacy_selected_indices": legacy_selected,
            "bounded_selected_indices": bounded_selected,
            "selected_indices_match": quality_pass,
        },
        "latency": {
            "legacy_mean_ms": float(legacy_mean),
            "bounded_mean_ms": float(bounded_mean),
            "speedup": float(legacy_mean / max(1e-9, bounded_mean)),
        },
        "payload": {
            "legacy_raw_text_payload_count_mean": float(statistics.fmean(legacy_payload_counts)),
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
                "speedup": result["latency"]["speedup"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
