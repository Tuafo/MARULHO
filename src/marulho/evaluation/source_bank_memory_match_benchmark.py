from __future__ import annotations

import argparse
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from marulho.reporting.io import write_json_file
from marulho.semantics.frontier import bank_memory_matches_with_report
from marulho.training.query_runner import memory_matches_with_report


@dataclass
class _SyntheticBank:
    name: str
    probe_patterns: list[torch.Tensor]
    probe_raw_windows: list[str]
    train_raw_windows: list[str]


class _SyntheticMemoryStore:
    def __init__(self, *, capacity: int, bucket_count: int, payload_repeats: int) -> None:
        self._row_assemblies = [torch.tensor([1.0, 0.0], dtype=torch.float32) for _ in range(capacity)]
        self._row_input_patterns = [item.clone() for item in self._row_assemblies]
        self._row_routing_keys = [torch.tensor([1.0, 0.0], dtype=torch.float32) for _ in range(capacity)]
        self._row_texts = [f"source bank memory episode {index}" for index in range(capacity)]
        self._row_bucket_ids = [index % max(1, int(bucket_count)) for index in range(capacity)]
        self._row_importance = [1.0 for _ in range(capacity)]
        self._row_timestamps = [0 for _ in range(capacity)]
        self._row_replay_count = [0 for _ in range(capacity)]
        self._row_consolidation_level = [0.25 for _ in range(capacity)]
        self.payload_repeats = max(1, int(payload_repeats))
        self.query_match_row_calls: list[tuple[int, bool]] = []
        self.last_bank_memory_match_report: dict[str, Any] = {}

    def replay_scores_for_indices(self, indices: list[int], current_token: int) -> dict[int, float]:
        return {int(index): 0.0 for index in indices}

    def live_summary_stats(self, current_token: int | None = None) -> dict[str, Any]:
        _ = current_token
        return {"size": len(self._row_assemblies)}

    def collect_query_memory_match_indices(
        self,
        *,
        candidate_bucket_ids,
        max_candidates: int,
        scope: str = "query_memory_match_slow_path",
    ) -> dict[str, Any]:
        buckets = [int(value) for value in (candidate_bucket_ids or [])]
        bucket_set = set(buckets)
        candidates = [
            index
            for index, bucket_id in enumerate(self._row_bucket_ids)
            if int(bucket_id) in bucket_set
        ][: max(0, int(max_candidates))]
        return {
            "surface": "bounded_query_memory_match_candidates.v1",
            "status": "collected" if candidates else "empty",
            "scope": scope,
            "memory_size": len(self._row_assemblies),
            "requested_count": int(max_candidates),
            "candidate_window_limit": int(max_candidates),
            "candidate_window_policy": "recent_bucket_round_robin_candidate_pool",
            "candidate_scope": "bucket_indexed_candidate_window",
            "candidate_bucket_ids": buckets,
            "candidate_bucket_count": len(buckets),
            "candidate_index_available_count": sum(
                1 for bucket_id in self._row_bucket_ids if int(bucket_id) in bucket_set
            ),
            "candidate_index_count": len(candidates),
            "match_indices": candidates,
            "global_score_scan": False,
            "global_candidate_scan": False,
            "runs_live_tick": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "fallback_reason": None if candidates else "empty_query_candidate_window",
        }

    def query_match_row(
        self,
        idx: int,
        current_token: int | None = None,
        *,
        include_text_payload: bool = False,
    ) -> dict[str, Any]:
        _ = current_token
        index = int(idx)
        self.query_match_row_calls.append((index, bool(include_text_payload)))
        base = self._row_texts[index]
        text = " ".join(base for _ in range(self.payload_repeats))
        row: dict[str, Any] = {
            "surface": "bounded_query_memory_match_row.v1",
            "memory_index": index,
            "read_only": True,
            "assembly": self._row_assemblies[index],
            "input_pattern": self._row_input_patterns[index],
            "routing_key": self._row_routing_keys[index],
            "bucket_id": self._row_bucket_ids[index],
            "importance": self._row_importance[index],
            "capture_tag": 0.0,
            "capture_strength": 0.0,
            "prp_level": 0.0,
            "consolidation_level": self._row_consolidation_level[index],
            "replay_count": self._row_replay_count[index],
            "age_tokens": int(max(0, int(current_token or 0) - int(self._row_timestamps[index]))),
            "raw_window": None,
            "text": None,
            "metadata": None,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
        }
        if include_text_payload:
            row.update(
                {
                    "text": text,
                    "raw_window": base,
                    "metadata": {},
                    "raw_text_payload_loaded": True,
                }
            )
        return row

    def record_bank_memory_match_report(self, report: dict[str, Any]) -> dict[str, Any]:
        self.last_bank_memory_match_report = dict(report)
        return self.last_bank_memory_match_report


class _SyntheticRoutingIndex:
    def __init__(self, bucket_count: int) -> None:
        self.bucket_count = max(1, int(bucket_count))

    def search_tensors(self, queries: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        count = max(1, min(int(k), self.bucket_count))
        ids = torch.arange(count, dtype=torch.long).unsqueeze(0)
        return ids, torch.ones((1, count), dtype=torch.float32)


class _SyntheticTrainer:
    def __init__(self, *, capacity: int, bucket_count: int, payload_repeats: int) -> None:
        self.token_count = 100
        self.config = SimpleNamespace(input_representation="hashed_ngram", k_routing=bucket_count)
        self.model = SimpleNamespace(
            memory_store=_SyntheticMemoryStore(
                capacity=capacity,
                bucket_count=bucket_count,
                payload_repeats=payload_repeats,
            ),
            routing_index=_SyntheticRoutingIndex(bucket_count),
        )

    def routing_key_for_pattern(self, pattern: torch.Tensor) -> torch.Tensor:
        return pattern


def _diagnostic_legacy_bank_memory_matches_no_cache(
    trainer: _SyntheticTrainer,
    bank: _SyntheticBank,
    *,
    probe_samples: int,
    memories_per_probe: int,
    max_matches: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aggregated: dict[int, dict[str, Any]] = {}
    probe_indices = list(range(min(max(0, int(probe_samples)), len(bank.probe_patterns))))
    raw_text_payload_count = 0
    cache_hits = 0
    candidate_count = 0
    for probe_idx in probe_indices:
        pattern = bank.probe_patterns[probe_idx]
        routing_key = trainer.routing_key_for_pattern(pattern)
        matches, report = memory_matches_with_report(
            trainer,
            pattern,
            routing_key,
            top_k=memories_per_probe,
            top_chars=1,
        )
        raw_text_payload_count += int(report.get("raw_text_payload_count", 0) or 0)
        cache_hits += int(report.get("raw_text_payload_cache_hits", 0) or 0)
        candidate_count += int(report.get("candidate_index_count", 0) or 0)
        for match in matches:
            memory_index = int(match.get("memory_index", -1))
            if memory_index < 0:
                continue
            existing = aggregated.get(memory_index)
            if existing is None or float(match.get("similarity", 0.0)) > float(existing.get("similarity", 0.0)):
                aggregated[memory_index] = dict(match)
    ranked = sorted(
        aggregated.values(),
        key=lambda item: (
            float(item.get("similarity", 0.0)),
            float(item.get("capture_strength", 0.0)),
            float(item.get("importance", 0.0)),
        ),
        reverse=True,
    )[: max(1, int(max_matches))]
    return ranked, {
        "raw_text_payload_count": int(raw_text_payload_count),
        "raw_text_payload_cache_hits": int(cache_hits),
        "candidate_index_count": int(candidate_count),
        "match_indices": [int(item["memory_index"]) for item in ranked],
    }


def _measure(fn, iterations: int) -> tuple[list[float], list[Any]]:
    latencies: list[float] = []
    results: list[Any] = []
    for _ in range(max(1, int(iterations))):
        started = time.perf_counter()
        result = fn()
        latencies.append((time.perf_counter() - started) * 1000.0)
        results.append(result)
    return latencies, results


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    trainer = _SyntheticTrainer(
        capacity=args.capacity,
        bucket_count=args.bucket_count,
        payload_repeats=args.payload_repeats,
    )
    bank = _SyntheticBank(
        name="source-bank",
        probe_patterns=[
            torch.tensor([1.0, 0.0], dtype=torch.float32)
            for _ in range(max(1, int(args.probe_samples)))
        ],
        probe_raw_windows=["source bank probe" for _ in range(max(1, int(args.probe_samples)))],
        train_raw_windows=["source bank probe"],
    )

    def legacy_call() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        trainer.model.memory_store.query_match_row_calls = []
        return _diagnostic_legacy_bank_memory_matches_no_cache(
            trainer,
            bank,
            probe_samples=args.probe_samples,
            memories_per_probe=args.memories_per_probe,
            max_matches=args.max_matches,
        )

    def bounded_call() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        trainer.model.memory_store.query_match_row_calls = []
        return bank_memory_matches_with_report(
            trainer,
            bank,
            probe_samples=args.probe_samples,
            memories_per_probe=args.memories_per_probe,
            max_matches=args.max_matches,
        )

    cuda_available = torch.cuda.is_available()
    cuda_allocated_before = torch.cuda.memory_allocated() if cuda_available else 0
    cuda_reserved_before = torch.cuda.memory_reserved() if cuda_available else 0
    tracemalloc.start()
    legacy_latencies, legacy_results = _measure(legacy_call, args.iterations)
    bounded_latencies, bounded_results = _measure(bounded_call, args.iterations)
    _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cuda_allocated_after = torch.cuda.memory_allocated() if cuda_available else 0
    cuda_reserved_after = torch.cuda.memory_reserved() if cuda_available else 0
    legacy_matches, legacy_report = legacy_results[-1]
    bounded_matches, bounded_report = bounded_results[-1]
    legacy_indices = [int(match["memory_index"]) for match in legacy_matches]
    bounded_indices = [int(match["memory_index"]) for match in bounded_matches]
    legacy_mean = statistics.fmean(legacy_latencies)
    bounded_mean = statistics.fmean(bounded_latencies)
    speedup = legacy_mean / max(1e-9, bounded_mean)
    passed = (
        legacy_indices == bounded_indices
        and int(bounded_report.get("raw_text_payload_count", 0)) <= len(set(bounded_indices))
        and bool(bounded_report.get("source_bank_row_reader_owned_by_store"))
        and not bool(bounded_report.get("replay_entry_reader_used"))
        and bool(bounded_report.get("direct_slow_memory_array_reads_retired"))
        and not bool(bounded_report.get("stc_state_advance"))
        and not bool(bounded_report.get("global_candidate_scan"))
        and not bool(bounded_report.get("global_score_scan"))
        and not bool(bounded_report.get("runs_live_tick"))
        and not bool(bounded_report.get("language_reasoning"))
    )
    return {
        "surface": "bounded_source_bank_memory_match_benchmark.v1",
        "passed": bool(passed),
        "capacity": int(args.capacity),
        "bucket_count": int(args.bucket_count),
        "probe_samples": int(args.probe_samples),
        "memories_per_probe": int(args.memories_per_probe),
        "max_matches": int(args.max_matches),
        "iterations": int(args.iterations),
        "payload_repeats": int(args.payload_repeats),
        "quality": {
            "selected_indices_match": bool(legacy_indices == bounded_indices),
            "legacy_selected_indices": legacy_indices,
            "bounded_selected_indices": bounded_indices,
            "min": 1.0 if legacy_indices == bounded_indices else 0.0,
        },
        "latency_ms": {
            "legacy_mean": float(legacy_mean),
            "bounded_mean": float(bounded_mean),
            "speedup": float(speedup),
            "legacy_min": float(min(legacy_latencies)),
            "bounded_min": float(min(bounded_latencies)),
        },
        "legacy_report": legacy_report,
        "bounded_report": bounded_report,
        "selection_budget": bounded_report.get("selection_budget", {}),
        "device": {
            "archival_storage_device": bounded_report.get("archival_storage_device"),
            "score_device": bounded_report.get("score_device"),
            "active_replay_cuda_required": False,
            "cuda_available": bool(cuda_available),
            "cuda_memory_allocated_delta_mib": float(
                (cuda_allocated_after - cuda_allocated_before) / (1024.0 * 1024.0)
            ),
            "cuda_memory_reserved_delta_mib": float(
                (cuda_reserved_after - cuda_reserved_before) / (1024.0 * 1024.0)
            ),
            "python_traced_peak_mib": float(peak_bytes / (1024.0 * 1024.0)),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark bounded source-bank memory recall reports")
    parser.add_argument("--capacity", type=int, default=65536)
    parser.add_argument("--bucket-count", type=int, default=16)
    parser.add_argument("--probe-samples", type=int, default=8)
    parser.add_argument("--memories-per-probe", type=int, default=4)
    parser.add_argument("--max-matches", type=int, default=16)
    parser.add_argument("--payload-repeats", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    report = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(args.output, report)
    print(f"passed={report['passed']}")
    print(
        "latency_ms legacy_mean={legacy:.6f} bounded_mean={bounded:.6f} speedup={speedup:.6f}".format(
            legacy=report["latency_ms"]["legacy_mean"],
            bounded=report["latency_ms"]["bounded_mean"],
            speedup=report["latency_ms"]["speedup"],
        )
    )
    print(
        "payload legacy={legacy} bounded={bounded} cache_hits={hits}".format(
            legacy=report["legacy_report"]["raw_text_payload_count"],
            bounded=report["bounded_report"]["raw_text_payload_count"],
            hits=report["bounded_report"]["raw_text_payload_cache_hits"],
        )
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
