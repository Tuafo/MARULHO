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
    def __init__(self, prefix: str, size: int, repeats: int) -> None:
        self.prefix = str(prefix)
        self.size = max(0, int(size))
        self.repeats = max(1, int(repeats))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> str:
        idx = int(index)
        if idx < 0 or idx >= self.size:
            raise IndexError(idx)
        base = f"{self.prefix} {idx % 97} context comparison evidence"
        return " ".join(base for _ in range(self.repeats))


class _SyntheticMemoryStore:
    def __init__(self, *, capacity: int, bucket_count: int, text_repeats: int) -> None:
        self.capacity = max(1, int(capacity))
        self.bucket_count = max(1, int(bucket_count))
        base = torch.tensor([1.0, 0.0], dtype=torch.float32)
        self.slow_buffer = [base for _ in range(self.capacity)]
        self.slow_input_patterns = [base for _ in range(self.capacity)]
        self.slow_routing_keys = [None for _ in range(self.capacity)]
        self.slow_raw_windows = _SyntheticTextSequence(
            "context memory episode",
            self.capacity,
            text_repeats,
        )
        self.slow_bucket_ids = [
            int(index % self.bucket_count) for index in range(self.capacity)
        ]
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
        _ = token_count
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
                sum(
                    1
                    for bucket_id in self.slow_bucket_ids
                    if int(bucket_id) in bucket_set
                )
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
        _ = current_token
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
        token_count=2048,
        config=SimpleNamespace(input_representation="hashed_ngram", k_routing=8),
        model=SimpleNamespace(memory_store=store),
    )


def _run_report_dropping_context(
    trainer: Any,
    *,
    context_count: int,
    top_k: int,
    candidate_limit: int,
    candidate_bucket_ids: Sequence[int],
) -> dict[str, Any]:
    pattern = torch.tensor([1.0, 0.0], dtype=torch.float32)
    selected_by_context: list[list[int]] = []
    started = time.perf_counter()
    for _ in range(max(1, int(context_count))):
        matches, _report = query_runner.memory_matches_with_report(
            trainer,
            pattern,
            pattern,
            top_k=top_k,
            top_chars=1,
            memory_candidate_limit=candidate_limit,
            candidate_bucket_ids=candidate_bucket_ids,
        )
        selected_by_context.append([int(match["memory_index"]) for match in matches])
    return {
        "latency_ms": float((time.perf_counter() - started) * 1000.0),
        "selected_by_context": selected_by_context,
    }


def _run_reported_context(
    trainer: Any,
    *,
    context_count: int,
    top_k: int,
    candidate_limit: int,
    candidate_bucket_ids: Sequence[int],
) -> dict[str, Any]:
    pattern = torch.tensor([1.0, 0.0], dtype=torch.float32)
    replay_entry_cache: dict[int, dict[str, Any]] = {}
    reports: list[dict[str, Any]] = []
    selected_by_context: list[list[int]] = []
    started = time.perf_counter()
    for context_index in range(max(1, int(context_count))):
        matches, report = query_runner.memory_matches_with_report(
            trainer,
            pattern,
            pattern,
            top_k=top_k,
            top_chars=1,
            memory_candidate_limit=candidate_limit,
            candidate_bucket_ids=candidate_bucket_ids,
            replay_entry_cache=replay_entry_cache,
        )
        reports.append({**dict(report), "context_label": f"context_{context_index}"})
        selected_by_context.append([int(match["memory_index"]) for match in matches])
    aggregate = query_runner.build_context_memory_match_report(reports)
    return {
        "latency_ms": float((time.perf_counter() - started) * 1000.0),
        "selected_by_context": selected_by_context,
        "aggregate_report": aggregate,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    store = _SyntheticMemoryStore(
        capacity=args.capacity,
        bucket_count=args.bucket_count,
        text_repeats=args.text_repeats,
    )
    trainer = _trainer_for_store(store)
    candidate_bucket_ids = [0]
    legacy_latencies: list[float] = []
    bounded_latencies: list[float] = []
    legacy_payload_counts: list[int] = []
    bounded_payload_counts: list[int] = []
    legacy_selected: list[list[int]] = []
    bounded_selected: list[list[int]] = []
    aggregate_report: dict[str, Any] = {}

    for _ in range(max(1, int(args.iterations))):
        store.replay_entry_calls.clear()
        legacy = _run_report_dropping_context(
            trainer,
            context_count=args.context_count,
            top_k=args.top_k,
            candidate_limit=args.candidate_limit,
            candidate_bucket_ids=candidate_bucket_ids,
        )
        legacy_latencies.append(float(legacy["latency_ms"]))
        legacy_payload_counts.append(len(store.replay_entry_calls))
        legacy_selected = list(legacy["selected_by_context"])

        store.replay_entry_calls.clear()
        bounded = _run_reported_context(
            trainer,
            context_count=args.context_count,
            top_k=args.top_k,
            candidate_limit=args.candidate_limit,
            candidate_bucket_ids=candidate_bucket_ids,
        )
        bounded_latencies.append(float(bounded["latency_ms"]))
        bounded_payload_counts.append(len(store.replay_entry_calls))
        bounded_selected = list(bounded["selected_by_context"])
        aggregate_report = dict(bounded["aggregate_report"])

    legacy_mean = statistics.fmean(legacy_latencies)
    bounded_mean = statistics.fmean(bounded_latencies)
    speedup = legacy_mean / max(1e-9, bounded_mean)
    quality_pass = bool(legacy_selected == bounded_selected)
    payload_gate_pass = bool(
        int(aggregate_report.get("raw_text_payload_count", 0) or 0)
        <= int(args.top_k)
        and int(aggregate_report.get("raw_text_payload_cache_hits", 0) or 0)
        >= int(args.top_k) * max(0, int(args.context_count) - 1)
    )
    report_gate_pass = bool(
        aggregate_report.get("surface") == "bounded_context_comparison_memory_match.v1"
        and not bool(aggregate_report.get("global_candidate_scan"))
        and not bool(aggregate_report.get("global_score_scan"))
        and not bool(aggregate_report.get("runs_live_tick"))
        and not bool(aggregate_report.get("runs_every_token"))
        and not bool(aggregate_report.get("language_reasoning"))
    )
    latency_gate_pass = bool(speedup >= float(args.min_speedup))
    return {
        "surface": "bounded_context_comparison_memory_match_benchmark.v1",
        "passed": bool(
            quality_pass and payload_gate_pass and report_gate_pass and latency_gate_pass
        ),
        "capacity": int(args.capacity),
        "bucket_count": int(args.bucket_count),
        "candidate_bucket_ids": candidate_bucket_ids,
        "candidate_window_limit": int(args.candidate_limit),
        "context_count": int(args.context_count),
        "top_k": int(args.top_k),
        "iterations": int(args.iterations),
        "text_repeats": int(args.text_repeats),
        "selection_criteria": "per-context bounded query memory candidate window",
        "quality": {
            "metric": "selected_indices_match_report_dropping_context",
            "min": 1.0 if quality_pass else 0.0,
            "selected_indices_match": quality_pass,
            "legacy_selected_by_context": legacy_selected,
            "bounded_selected_by_context": bounded_selected,
        },
        "latency_ms": {
            "legacy_mean": float(legacy_mean),
            "bounded_mean": float(bounded_mean),
            "speedup": float(speedup),
            "legacy_min": float(min(legacy_latencies)),
            "bounded_min": float(min(bounded_latencies)),
        },
        "payload": {
            "legacy_raw_text_payload_count_mean": float(
                statistics.fmean(legacy_payload_counts)
            ),
            "bounded_raw_text_payload_count_mean": float(
                statistics.fmean(bounded_payload_counts)
            ),
            "aggregate_raw_text_payload_count": int(
                aggregate_report.get("raw_text_payload_count", 0) or 0
            ),
            "aggregate_raw_text_payload_cache_hits": int(
                aggregate_report.get("raw_text_payload_cache_hits", 0) or 0
            ),
        },
        "aggregate_report": aggregate_report,
        "device_placement": {
            "archival_storage_device": aggregate_report.get("archival_storage_device"),
            "score_device": aggregate_report.get("score_device"),
            "active_replay_cuda_required": False,
        },
        "gates": {
            "quality_gate_pass": quality_pass,
            "payload_cache_gate_pass": payload_gate_pass,
            "report_gate_pass": report_gate_pass,
            "latency_gate_pass": latency_gate_pass,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=65536)
    parser.add_argument("--bucket-count", type=int, default=16)
    parser.add_argument("--candidate-limit", type=int, default=192)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--context-count", type=int, default=2)
    parser.add_argument("--text-repeats", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--min-speedup", type=float, default=1.0)
    args = parser.parse_args()

    result = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(args.output, result)
    print(
        json.dumps(
            {
                "passed": bool(result["passed"]),
                "quality": result["quality"],
                "payload": result["payload"],
                "speedup": result["latency_ms"]["speedup"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
