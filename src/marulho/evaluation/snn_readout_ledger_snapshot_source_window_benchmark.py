"""Benchmark bounded SNN readout-ledger snapshot source windows."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from threading import RLock
import time
import tracemalloc
from typing import Any, Callable, Mapping

import torch

from marulho.service.runtime_state import RuntimeState
from marulho.service.snn_language_readout_ledger import (
    SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS,
    SNN_LANGUAGE_READOUT_LEDGER_SNAPSHOT_EVENT_FIELDS,
    SNN_LANGUAGE_READOUT_LEDGER_SNAPSHOT_SOURCE_WINDOW_POLICY,
    SNNLanguageReadoutEvidenceLedger,
)


class CountedRows:
    def __init__(self, field: str, count: int) -> None:
        self.field = field
        self.count = int(count)
        self.iterated = 0

    def __len__(self) -> int:
        return int(self.count)

    def __iter__(self):
        for index in range(self.count):
            self.iterated += 1
            yield _event(self.field, index)


def _event(field: str, index: int) -> dict[str, Any]:
    return {
        "surface": "snn_language_readout_ledger_benchmark_event.v1",
        "field": str(field),
        "ordinal": int(index),
        "readout_evidence_id": f"{field}:readout:{index}",
        "readout_evidence_hash": f"{field}:readout-hash:{index}",
        "rollout_evidence_hash": f"{field}:rollout-hash:{index}",
        "emission_review_hash": f"{field}:review-hash:{index}",
        "prediction_hash": f"{field}:prediction:{index}",
        "transition_memory_evaluation_hash": f"{field}:evaluation:{index}",
        "persistent_transition_weights_hash": f"{field}:weights:{index % 11}",
        "labels": [f"{field}:label:{index % 7}"],
        "label_grounding": [True],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def _ledger_with_counted_rows(
    *,
    retention_count: int,
    ledger_limit: int,
) -> tuple[SNNLanguageReadoutEvidenceLedger, dict[str, CountedRows]]:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    rows_by_field = {
        field: CountedRows(field, retention_count)
        for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }
    ledger_state: dict[str, object] = dict(rows_by_field)
    ledger_state.update(
        {
            "total_recorded_count": int(retention_count),
            "total_rollout_recorded_count": int(retention_count),
            "total_emission_review_count": int(retention_count),
            "total_dense_label_candidate_count": int(retention_count),
            "total_dense_label_calibration_update_count": int(retention_count),
            "last_recorded_at": "2026-06-20T00:00:00+00:00",
        }
    )
    return (
        SNNLanguageReadoutEvidenceLedger(
            lock=lock,
            runtime_state=runtime_state,
            ledger_state=lambda: ledger_state,
            limit=ledger_limit,
        ),
        rows_by_field,
    )


def _row_reads(rows_by_field: Mapping[str, CountedRows]) -> dict[str, int]:
    return {field: int(rows.iterated) for field, rows in rows_by_field.items()}


def _quality_from_events(
    events_by_field: Mapping[str, list[Mapping[str, Any]]],
    *,
    snapshot_limit: int,
) -> dict[str, Any]:
    expected = list(range(max(0, int(snapshot_limit))))
    ordinals_by_field = {
        field: [
            int(item.get("ordinal", -1))
            for item in list(events_by_field.get(field) or [])
        ]
        for field in SNN_LANGUAGE_READOUT_LEDGER_SNAPSHOT_EVENT_FIELDS
    }
    parity_by_field = {
        field: ordinals == expected
        for field, ordinals in ordinals_by_field.items()
    }
    return {
        "newest_first_ordinals": ordinals_by_field,
        "newest_first_parity_by_field": parity_by_field,
        "newest_first_parity": all(parity_by_field.values()),
    }


def _bounded_snapshot_once(
    *,
    retention_count: int,
    ledger_limit: int,
    snapshot_limit: int,
) -> dict[str, Any]:
    ledger, rows_by_field = _ledger_with_counted_rows(
        retention_count=retention_count,
        ledger_limit=ledger_limit,
    )
    snapshot = ledger.snapshot(limit=snapshot_limit)
    events_by_field = {
        field: [deepcopy(dict(item)) for item in list(snapshot.get(field) or [])]
        for field in SNN_LANGUAGE_READOUT_LEDGER_SNAPSHOT_EVENT_FIELDS
    }
    row_reads = _row_reads(rows_by_field)
    source_window = dict(snapshot["summary"]["snapshot_source_window"])
    snapshot_fields = set(source_window["snapshot_event_fields"])
    return {
        "row_reads": row_reads,
        "row_read_count": int(sum(row_reads.values())),
        "source_window": source_window,
        "event_count": int(snapshot["summary"]["event_count"]),
        "returned_event_count": int(snapshot["summary"]["returned_event_count"]),
        "quality": {
            **_quality_from_events(
                events_by_field,
                snapshot_limit=snapshot_limit,
            ),
            "retained_event_count_preserved": (
                int(snapshot["summary"]["event_count"])
                == min(int(retention_count), int(ledger_limit))
            ),
            "source_window_policy_match": (
                source_window["policy"]
                == SNN_LANGUAGE_READOUT_LEDGER_SNAPSHOT_SOURCE_WINDOW_POLICY
            ),
            "unreturned_fields_unread": all(
                count == 0
                for field, count in row_reads.items()
                if field not in snapshot_fields
            ),
        },
    }


def _retired_normalized_snapshot_model_once(
    *,
    retention_count: int,
    ledger_limit: int,
    snapshot_limit: int,
) -> dict[str, Any]:
    ledger, rows_by_field = _ledger_with_counted_rows(
        retention_count=retention_count,
        ledger_limit=ledger_limit,
    )
    normalized = ledger._normalized_state()  # noqa: SLF001 - benchmark-only retired model
    events_by_field = {
        field: [
            deepcopy(dict(item))
            for item in list(normalized.get(field) or [])[:snapshot_limit]
        ]
        for field in SNN_LANGUAGE_READOUT_LEDGER_SNAPSHOT_EVENT_FIELDS
    }
    row_reads = _row_reads(rows_by_field)
    return {
        "row_reads": row_reads,
        "row_read_count": int(sum(row_reads.values())),
        "source_window": dict(normalized.get("_normalization_source_window") or {}),
        "event_count": int(len(normalized.get("events") or [])),
        "returned_event_count": int(len(events_by_field["events"])),
        "quality": {
            **_quality_from_events(
                events_by_field,
                snapshot_limit=snapshot_limit,
            ),
            "retained_event_count_preserved": (
                int(len(normalized.get("events") or []))
                == min(int(retention_count), int(ledger_limit))
            ),
            "source_window_policy_match": True,
            "unreturned_fields_unread": False,
        },
    }


def _measure(
    fn: Callable[[], dict[str, Any]],
    *,
    runs: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for _ in range(max(1, int(runs))):
        tracemalloc.start()
        started = time.perf_counter()
        record = fn()
        latency_ms = (time.perf_counter() - started) * 1000.0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        records.append(
            {
                **record,
                "latency_ms": float(latency_ms),
                "python_peak_mib": float(peak_bytes / (1024.0 * 1024.0)),
            }
        )
    latencies = [float(record["latency_ms"]) for record in records]
    peaks = [float(record["python_peak_mib"]) for record in records]
    row_reads = [int(record["row_read_count"]) for record in records]
    quality = records[-1]["quality"]
    return {
        "runs": int(len(records)),
        "latency_ms": {
            "mean": float(statistics.fmean(latencies)),
            "median": float(statistics.median(latencies)),
            "min": float(min(latencies)),
            "max": float(max(latencies)),
        },
        "python_peak_mib": {
            "mean": float(statistics.fmean(peaks)),
            "max": float(max(peaks)),
        },
        "row_read_count": {
            "mean": float(statistics.fmean(row_reads)),
            "last": int(row_reads[-1]),
        },
        "source_window": records[-1]["source_window"],
        "event_count": int(records[-1]["event_count"]),
        "returned_event_count": int(records[-1]["returned_event_count"]),
        "quality": quality,
    }


def run_benchmark(
    *,
    retention_count: int,
    ledger_limit: int,
    snapshot_limit: int,
    runs: int,
) -> dict[str, Any]:
    bounded = _measure(
        lambda: _bounded_snapshot_once(
            retention_count=retention_count,
            ledger_limit=ledger_limit,
            snapshot_limit=snapshot_limit,
        ),
        runs=runs,
    )
    retired = _measure(
        lambda: _retired_normalized_snapshot_model_once(
            retention_count=retention_count,
            ledger_limit=ledger_limit,
            snapshot_limit=snapshot_limit,
        ),
        runs=runs,
    )
    bounded_rows = max(1.0, float(bounded["row_read_count"]["mean"]))
    retired_rows = float(retired["row_read_count"]["mean"])
    bounded_latency = max(1e-9, float(bounded["latency_ms"]["mean"]))
    retired_latency = float(retired["latency_ms"]["mean"])
    cuda_available = bool(torch.cuda.is_available())
    return {
        "surface": "snn_readout_ledger_snapshot_source_window_benchmark.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "retention_count": int(retention_count),
            "ledger_limit": int(ledger_limit),
            "snapshot_limit": int(snapshot_limit),
            "runs": int(runs),
        },
        "bounded_snapshot": bounded,
        "retired_normalized_snapshot_model": retired,
        "comparison": {
            "row_read_reduction_ratio": float(retired_rows / bounded_rows),
            "latency_speedup_ratio": float(retired_latency / bounded_latency),
            "bounded_reads_only_returned_snapshot_fields": bool(
                bounded["quality"]["unreturned_fields_unread"]
            ),
            "quality_preserved": bool(
                bounded["quality"]["newest_first_parity"]
                and bounded["quality"]["retained_event_count_preserved"]
                and retired["quality"]["newest_first_parity"]
                and retired["quality"]["retained_event_count_preserved"]
            ),
        },
        "runtime_truth": {
            "source_window_surface": bounded["source_window"]["surface"],
            "source_window_policy": bounded["source_window"]["policy"],
            "selection_criteria": bounded["source_window"]["selection_criteria"],
            "memory_budget": bounded["source_window"]["memory_budget"],
            "archival_storage_device": "cpu",
            "snapshot_device": "cpu",
            "gpu_used": False,
            "cuda_available": cuda_available,
            "cuda_allocated_mib": float(
                torch.cuda.memory_allocated() / (1024.0 * 1024.0)
            )
            if cuda_available
            else 0.0,
            "cuda_reserved_mib": float(
                torch.cuda.memory_reserved() / (1024.0 * 1024.0)
            )
            if cuda_available
            else 0.0,
            "gpu_resident_archival_metadata": False,
            "runs_live_tick": False,
            "runs_every_token": False,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "language_reasoning": False,
            "raw_text_scored": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the bounded SNN readout-ledger snapshot source window."
        )
    )
    parser.add_argument("--retention-count", type=int, default=2048)
    parser.add_argument("--ledger-limit", type=int, default=128)
    parser.add_argument("--snapshot-limit", type=int, default=20)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run_benchmark(
        retention_count=args.retention_count,
        ledger_limit=args.ledger_limit,
        snapshot_limit=args.snapshot_limit,
        runs=args.runs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"snn_readout_ledger_snapshot_source_window_benchmark={args.output}")


if __name__ == "__main__":
    main()
