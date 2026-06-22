from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from marulho.gap_planner import _frontier_gap_terms_from_entries
from marulho.gap_planner import frontier_gap_plan
from marulho.semantics.grounding_text import normalize_text as _normalize_text


class _SyntheticSequence:
    def __init__(self, size: int, value_fn: Any) -> None:
        self.size = int(size)
        self.value_fn = value_fn

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Any:
        idx = int(index)
        if idx < 0 or idx >= self.size:
            raise IndexError(idx)
        return self.value_fn(idx)


class _BenchmarkFrontierStore:
    def __init__(self, *, capacity: int, candidate_window: int) -> None:
        self.capacity = int(capacity)
        self.candidate_window = int(candidate_window)
        self.frontier_indices = {
            self.capacity - 24,
            self.capacity - 40,
            self.capacity - 56,
        }
        self.slow_raw_windows = _SyntheticSequence(self.capacity, self._raw_window)
        self.slow_importance = _SyntheticSequence(self.capacity, self._importance)
        self.slow_capture_tag = _SyntheticSequence(self.capacity, self._capture)
        self.slow_consolidation_level = _SyntheticSequence(
            self.capacity,
            self._consolidation,
        )
        self.collect_calls = 0
        self.query_match_row_calls: list[tuple[int, bool]] = []

    def _raw_window(self, index: int) -> str:
        if index in self.frontier_indices:
            return "credit loan deposit account unstable frontier memory"
        topic = index % 97
        return f"background topic {topic} stable consolidated trace"

    def _importance(self, index: int) -> float:
        return 1.0 if index in self.frontier_indices else 0.05 + 0.0001 * float(index % 13)

    def _capture(self, index: int) -> float:
        return 0.95 if index in self.frontier_indices else 0.03 + 0.001 * float(index % 7)

    def _consolidation(self, index: int) -> float:
        return 0.05 if index in self.frontier_indices else 0.90

    def query_match_row(
        self,
        index: int,
        current_token: int | None = None,
        *,
        include_text_payload: bool = False,
    ) -> dict[str, Any]:
        _ = current_token
        idx = int(index)
        self.query_match_row_calls.append((idx, bool(include_text_payload)))
        raw_window = self.slow_raw_windows[idx]
        capture = float(self.slow_capture_tag[idx])
        row: dict[str, Any] = {
            "surface": "bounded_query_memory_match_row.v1",
            "memory_index": idx,
            "read_only": True,
            "importance": float(self.slow_importance[idx]),
            "capture_tag": capture,
            "capture_strength": capture,
            "consolidation_level": float(self.slow_consolidation_level[idx]),
            "raw_window": None,
            "text": None,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
        }
        if include_text_payload:
            row.update(
                {
                    "raw_window": raw_window,
                    "text": raw_window,
                    "raw_text_payload_loaded": bool(raw_window),
                }
            )
        return row

    def collect_frontier_gap_indices(
        self,
        *,
        current_token: int,
        max_candidates: int,
        scope: str,
    ) -> dict[str, Any]:
        self.collect_calls += 1
        limit = max(0, int(max_candidates))
        start = max(0, self.capacity - limit)
        indices = list(range(self.capacity - 1, start - 1, -1))
        return {
            "surface": "bounded_frontier_gap_candidates.v1",
            "status": "collected" if indices else "empty",
            "scope": str(scope),
            "memory_size": int(self.capacity),
            "current_token": int(current_token),
            "requested_count": int(max_candidates),
            "candidate_window_limit": int(max_candidates),
            "candidate_window_policy": "recent_entry_index_candidate_window",
            "candidate_scope": "recent_entry_index_candidate_window",
            "candidate_bucket_ids": [],
            "candidate_bucket_count": 0,
            "candidate_index_available_count": int(min(self.capacity, limit + 1)),
            "candidate_index_available_count_is_lower_bound": self.capacity > limit,
            "candidate_index_count": int(len(indices)),
            "candidate_indices": indices,
            "global_score_scan": False,
            "global_candidate_scan": False,
            "runs_live_tick": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "fallback_reason": None,
        }


class _MissingCollectorFrontierStore:
    def __init__(self, *, capacity: int) -> None:
        self.capacity = int(capacity)
        self.slow_raw_windows = _SyntheticSequence(
            self.capacity,
            lambda index: f"unbounded compatibility text {index}",
        )


def _diagnostic_legacy_frontier_terms(
    store: _BenchmarkFrontierStore,
    *,
    current_token: int,
    top_entries: int,
    max_terms: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    windows = [store.slow_raw_windows[index] for index in range(store.capacity)]
    importance_values = [store.slow_importance[index] for index in range(store.capacity)]
    capture_values = [store.slow_capture_tag[index] for index in range(store.capacity)]
    consolidation_values = [
        store.slow_consolidation_level[index]
        for index in range(store.capacity)
    ]
    scored_entries: list[tuple[float, str, int]] = []
    for index, raw_window in enumerate(windows):
        text = _normalize_text(raw_window)
        if not text:
            continue
        importance = float(importance_values[index])
        capture = float(capture_values[index])
        consolidation = float(consolidation_values[index])
        frontier_pressure = max(0.0, capture - consolidation) + 0.5 * max(0.0, 1.0 - consolidation)
        score = max(1e-6, importance) * (1.0 + frontier_pressure)
        scored_entries.append((float(score), text, int(index)))
    scored_entries.sort(key=lambda item: item[0], reverse=True)
    selected = scored_entries[: max(1, int(top_entries))]
    terms = _frontier_gap_terms_from_entries(selected, limit=max_terms)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return terms, {
        "latency_ms": float(latency_ms),
        "selected_indices": [int(index) for _score, _text, index in selected],
        "score_count": int(len(scored_entries)),
        "global_candidate_scan": True,
        "global_score_scan": True,
        "archive_window_materialization_count": 1,
        "side_list_materialization_count": 3,
        "retired_inner_side_list_materialization_upper_bound": int(store.capacity * 3),
    }


def _missing_collector_gate(capacity: int) -> dict[str, Any]:
    plan = frontier_gap_plan(
        memory_store=_MissingCollectorFrontierStore(capacity=int(capacity)),
        current_token=int(capacity),
        max_terms=8,
        max_queries=4,
        max_questions=4,
        top_entries=24,
    )
    report = dict(plan.get("frontier_selection_report") or {})
    passed = bool(
        report.get("surface") == "bounded_frontier_gap_selection.v1"
        and report.get("fallback_reason")
        == "memory_store_missing_bounded_frontier_collector"
        and int(report.get("candidate_index_count", 0) or 0) == 0
        and int(report.get("raw_text_payload_count", 0) or 0) == 0
        and not bool(report.get("raw_text_payload_loaded"))
        and not bool(report.get("global_candidate_scan"))
        and not bool(report.get("global_score_scan"))
        and not list(plan.get("gap_terms") or [])
    )
    return {
        "passed": passed,
        "fallback_reason": report.get("fallback_reason"),
        "candidate_index_count": int(report.get("candidate_index_count", 0) or 0),
        "raw_text_payload_count": int(report.get("raw_text_payload_count", 0) or 0),
        "raw_text_payload_loaded": bool(report.get("raw_text_payload_loaded")),
        "global_candidate_scan": bool(report.get("global_candidate_scan")),
        "global_score_scan": bool(report.get("global_score_scan")),
        "gap_term_count": int(len(list(plan.get("gap_terms") or []))),
    }


def _term_recall(reference: list[str], candidate: list[str]) -> float:
    reference_set = {str(term) for term in reference if str(term)}
    candidate_set = {str(term) for term in candidate if str(term)}
    if not reference_set:
        return 1.0
    return float(len(reference_set & candidate_set)) / float(len(reference_set))


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate_window = max(32, int(args.top_entries) * 8)
    store = _BenchmarkFrontierStore(
        capacity=int(args.capacity),
        candidate_window=candidate_window,
    )
    legacy_latencies: list[float] = []
    bounded_latencies: list[float] = []
    legacy_terms: list[dict[str, Any]] = []
    legacy_report: dict[str, Any] = {}
    bounded_plan: dict[str, Any] = {}
    for _ in range(max(1, int(args.iterations))):
        legacy_terms, legacy_report = _diagnostic_legacy_frontier_terms(
            store,
            current_token=int(args.capacity),
            top_entries=int(args.top_entries),
            max_terms=int(args.max_terms),
        )
        legacy_latencies.append(float(legacy_report["latency_ms"]))

        bounded_started = time.perf_counter()
        bounded_plan = frontier_gap_plan(
            memory_store=store,
            current_token=int(args.capacity),
            max_terms=int(args.max_terms),
            max_queries=4,
            max_questions=4,
            top_entries=int(args.top_entries),
        )
        bounded_latencies.append((time.perf_counter() - bounded_started) * 1000.0)

    legacy_term_values = [str(item.get("term", "")) for item in legacy_terms]
    bounded_term_values = [
        str(item.get("term", ""))
        for item in list(bounded_plan.get("gap_terms") or [])
    ]
    expected_terms = ["credit", "loan", "deposit", "account"]
    expected_recall = _term_recall(expected_terms, bounded_term_values)
    legacy_recall = _term_recall(legacy_term_values[:4], bounded_term_values)
    quality_min = min(expected_recall, legacy_recall)
    legacy_mean = sum(legacy_latencies) / float(len(legacy_latencies))
    bounded_mean = sum(bounded_latencies) / float(len(bounded_latencies))
    speedup = legacy_mean / max(1e-9, bounded_mean)
    selection_report = dict(bounded_plan.get("frontier_selection_report") or {})
    passed = bool(
        quality_min >= float(args.min_quality)
        and not bool(selection_report.get("global_candidate_scan"))
        and not bool(selection_report.get("global_score_scan"))
        and bool(selection_report.get("frontier_row_reader_owned_by_store"))
        and bool(selection_report.get("direct_slow_memory_array_reads_retired"))
        and not bool(selection_report.get("effective_capture_reader_used"))
        and not bool(selection_report.get("stc_state_advance"))
        and int(selection_report.get("candidate_index_count", 0) or 0) <= candidate_window
    )
    missing_collector = _missing_collector_gate(int(args.capacity))
    passed = bool(passed and bool(missing_collector["passed"]))
    return {
        "surface": "frontier_gap_bounded_benchmark.v1",
        "passed": passed,
        "capacity": int(args.capacity),
        "iterations": int(args.iterations),
        "top_entries": int(args.top_entries),
        "candidate_window_limit": int(candidate_window),
        "quality": {
            "metric": "term_recall_against_legacy_and_expected_frontier_terms",
            "min": float(quality_min),
            "expected_term_recall": float(expected_recall),
            "legacy_top_term_recall": float(legacy_recall),
            "expected_terms": expected_terms,
            "legacy_terms": legacy_term_values,
            "bounded_terms": bounded_term_values,
        },
        "latency_ms": {
            "legacy_mean": float(legacy_mean),
            "bounded_mean": float(bounded_mean),
            "legacy_min": float(min(legacy_latencies)),
            "bounded_min": float(min(bounded_latencies)),
            "speedup": float(speedup),
        },
        "legacy": legacy_report,
        "bounded_selection_report": selection_report,
        "missing_collector_gate": missing_collector,
        "device": {
            "archival_storage_device": "cpu",
            "score_device": "cpu",
            "active_replay_cuda_required": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=65_536)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--top-entries", type=int, default=24)
    parser.add_argument("--max-terms", type=int, default=8)
    parser.add_argument("--min-quality", type=float, default=1.0)
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "speedup": report["latency_ms"]["speedup"], "quality_min": report["quality"]["min"]}, sort_keys=True))


if __name__ == "__main__":
    main()
