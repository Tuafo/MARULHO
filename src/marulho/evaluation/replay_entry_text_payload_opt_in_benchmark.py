from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch

from marulho.consolidation.memory_store import DualMemoryStore
from marulho.reporting.io import write_json_file
from marulho.training import query_runner


class _SyntheticRoutingIndex:
    def __init__(self, bucket_count: int) -> None:
        self.bucket_count = max(1, int(bucket_count))

    def search_tensors(
        self,
        queries: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = queries
        count = max(1, min(int(k), self.bucket_count))
        ids = torch.arange(count, dtype=torch.long).unsqueeze(0)
        return ids, torch.ones((1, count), dtype=torch.float32)


def _build_store(*, entries: int, buckets: int) -> DualMemoryStore:
    size = max(1, int(entries))
    bucket_count = max(1, int(buckets))
    store = DualMemoryStore(capacity=size)
    base = torch.tensor([1.0, 0.0], dtype=torch.float32)
    store.slow_buffer = [base for _ in range(size)]
    store.slow_input_patterns = [base for _ in range(size)]
    store.slow_routing_keys = [base for _ in range(size)]
    store.slow_raw_windows = [
        f"explicit replay entry raw window {index}."
        for index in range(size)
    ]
    store.slow_texts = [
        f"explicit replay entry source text {index}."
        for index in range(size)
    ]
    store.slow_metadata = [
        {"source_type": "benchmark", "source_index": int(index)}
        for index in range(size)
    ]
    store.slow_bucket_ids = [int(index % bucket_count) for index in range(size)]
    store.slow_importance = [1.0 for _ in range(size)]
    store.slow_capture_tag = array("d", (0.0 for _ in range(size)))
    store.slow_tag_is_strong = array("b", (False for _ in range(size)))
    store.slow_local_prp = array("d", (0.0 for _ in range(size)))
    store.slow_last_capture_token = [int(index) for index in range(size)]
    store.slow_consolidation_level = [0.0 for _ in range(size)]
    store.slow_consolidation_events = [0 for _ in range(size)]
    store.slow_last_replay_token = [int(index) for index in range(size)]
    store.slow_replay_count = [0 for _ in range(size)]
    store.slow_ripple_strength = array("d", (0.0 for _ in range(size)))
    store.slow_entry_timestamps = array("q", (int(index) for index in range(size)))
    bucket_entries: defaultdict[int, list[int]] = defaultdict(list)
    for index, bucket_id in enumerate(store.slow_bucket_ids):
        bucket_entries[int(bucket_id)].append(int(index))
    store._bucket_entry_indices = bucket_entries
    store._recent_entry_indices = [
        (int(index), int(index)) for index in range(size)
    ]
    return store


def _measure_entries(
    store: DualMemoryStore,
    indices: Sequence[int],
    *,
    include_text_payload: bool | None,
    current_token: int,
    runs: int,
) -> dict[str, Any]:
    elapsed: list[float] = []
    payload_counts: list[int] = []
    for _ in range(max(1, int(runs))):
        started = time.perf_counter()
        payload_count = 0
        for index in indices:
            if include_text_payload is None:
                entry = store.replay_entry(int(index), current_token=current_token)
            else:
                entry = store.replay_entry(
                    int(index),
                    current_token=current_token,
                    include_text_payload=bool(include_text_payload),
                )
            if entry.get("raw_window") is not None or entry.get("text") is not None:
                payload_count += 1
        elapsed.append((time.perf_counter() - started) * 1000.0)
        payload_counts.append(int(payload_count))
    return {
        "mean_ms": float(statistics.fmean(elapsed)),
        "min_ms": float(min(elapsed)),
        "max_ms": float(max(elapsed)),
        "payload_count_last": int(payload_counts[-1]),
        "payload_count_mean": float(statistics.fmean(payload_counts)),
    }


def run_benchmark(
    *,
    entries: int,
    buckets: int,
    read_count: int,
    candidate_limit: int,
    top_k: int,
    runs: int,
) -> dict[str, Any]:
    store = _build_store(entries=entries, buckets=buckets)
    current_token = int(entries) + 1
    indices = list(range(min(max(1, int(read_count)), int(entries))))
    retired_explicit = _measure_entries(
        store,
        indices,
        include_text_payload=True,
        current_token=current_token,
        runs=runs,
    )
    bounded_default = _measure_entries(
        store,
        indices,
        include_text_payload=None,
        current_token=current_token,
        runs=runs,
    )
    trainer = SimpleNamespace(
        token_count=current_token,
        config=SimpleNamespace(input_representation="hashed_ngram", k_routing=buckets),
        model=SimpleNamespace(
            memory_store=store,
            routing_index=_SyntheticRoutingIndex(buckets),
        ),
    )
    query_pattern = torch.tensor([1.0, 0.0], dtype=torch.float32)
    query_matches, query_report = query_runner.memory_matches_with_report(
        trainer,
        query_pattern,
        query_pattern,
        top_k=top_k,
        top_chars=1,
        memory_candidate_limit=candidate_limit,
        candidate_bucket_ids=[0],
    )
    query_text_payload_count = int(query_report.get("raw_text_payload_count", 0) or 0)
    query_match_text_count = sum(
        1
        for match in query_matches
        if match.get("raw_window") is not None or match.get("text") is not None
    )
    quality = {
        "default_replay_entry_text_payload_count": int(
            bounded_default["payload_count_last"]
        ),
        "retired_default_text_payload_count": int(
            retired_explicit["payload_count_last"]
        ),
        "query_returned_match_count": int(len(query_matches)),
        "query_match_text_payload_count": int(query_match_text_count),
        "query_report_text_payload_count": int(query_text_payload_count),
        "query_report_policy": str(query_report.get("raw_text_payload_policy", "")),
    }
    gates = {
        "default_payload_gate": bool(
            int(bounded_default["payload_count_last"]) == 0
        ),
        "explicit_payload_gate": bool(
            int(retired_explicit["payload_count_last"]) == len(indices)
        ),
        "query_opt_in_gate": bool(
            query_text_payload_count == len(query_matches)
            and query_match_text_count == len(query_matches)
            and query_report.get("raw_text_payload_policy")
            == "returned_similarity_matches_only"
        ),
        "bounded_candidate_gate": bool(
            not bool(query_report.get("global_candidate_scan"))
            and not bool(query_report.get("global_score_scan"))
        ),
        "language_boundary_gate": bool(
            not bool(query_report.get("language_reasoning"))
        ),
    }
    passed = bool(all(gates.values()))
    return {
        "surface": "explicit_replay_entry_text_payload_opt_in.v1",
        "entries": int(entries),
        "buckets": int(buckets),
        "read_count": int(len(indices)),
        "candidate_window_limit": int(candidate_limit),
        "top_k": int(top_k),
        "runs": int(runs),
        "selection_criteria": [
            "default_replay_entry_tensor_metadata_read",
            "explicit_query_readout_text_payload_opt_in",
        ],
        "memory_budget": {
            "default_read_entries": int(len(indices)),
            "query_candidate_window_entries": int(candidate_limit),
            "query_return_budget_entries": int(top_k),
            "archival_storage_device": "cpu",
        },
        "device_placement": {
            "archival_storage_device": "cpu",
            "score_device": "cpu",
            "cuda_required": False,
        },
        "runtime_truth": {
            "runs_live_tick": False,
            "runs_every_token": False,
            "full_memory_scan": False,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_text_payload_default_loaded": False,
            "raw_text_payload_explicit_opt_in_required": True,
            "hidden_language_reasoning": False,
            "language_reasoning": False,
        },
        "latency": {
            "retired_default_text_payload_simulated": retired_explicit,
            "bounded_default_tensor_only": bounded_default,
        },
        "query_report": query_report,
        "quality": quality,
        "gates": gates,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark explicit replay-entry raw text payload opt-in."
    )
    parser.add_argument("--entries", type=int, default=65_536)
    parser.add_argument("--buckets", type=int, default=16)
    parser.add_argument("--read-count", type=int, default=192)
    parser.add_argument("--candidate-limit", type=int, default=192)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(
        entries=args.entries,
        buckets=args.buckets,
        read_count=args.read_count,
        candidate_limit=args.candidate_limit,
        top_k=args.top_k,
        runs=args.runs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(args.output, report)
    print(
        "passed={passed} entries={entries} default_payload={default_payload} "
        "explicit_payload={explicit_payload} query_payload={query_payload}".format(
            passed=report["passed"],
            entries=report["entries"],
            default_payload=report["quality"][
                "default_replay_entry_text_payload_count"
            ],
            explicit_payload=report["quality"]["retired_default_text_payload_count"],
            query_payload=report["quality"]["query_report_text_payload_count"],
        )
    )


if __name__ == "__main__":
    main()
