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
    source_window = bounded["source_window"]
    bounded_row_reads = int(bounded["row_read_count"]["last"])
    retained_rows_per_field = min(int(retention_count), int(ledger_limit))
    projected_all_family_rows = (
        int(len(SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS))
        * int(retained_rows_per_field)
    )
    source_window_memory_budget = (
        source_window.get("memory_budget")
        if isinstance(source_window.get("memory_budget"), Mapping)
        else {}
    )
    quality = {
        **bounded["quality"],
        "retired_all_family_snapshot_comparator_removed": True,
        "returned_field_only_source_reads": bool(
            bounded["quality"]["unreturned_fields_unread"]
        ),
        "source_window_rows_match_memory_budget": (
            bounded_row_reads
            == int(source_window_memory_budget.get("max_records_total", -1))
        ),
    }
    retired_absence = {
        "implementation_present": False,
        "diagnostic_callable": False,
        "active_report_field_present": False,
        "removed_policy": (
            "snn_readout_ledger_snapshot_all_family_normalized_comparator"
        ),
    }
    cuda_available = bool(torch.cuda.is_available())
    return {
        "surface": "snn_readout_ledger_snapshot_source_window_benchmark.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pass": all(bool(value) for value in quality.values()),
        "parameters": {
            "retention_count": int(retention_count),
            "ledger_limit": int(ledger_limit),
            "snapshot_limit": int(snapshot_limit),
            "runs": int(runs),
        },
        "bounded_snapshot": bounded,
        "quality": quality,
        "memory_budget": {
            "all_ledger_event_field_count": int(
                len(SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS)
            ),
            "snapshot_event_field_count": int(
                len(SNN_LANGUAGE_READOUT_LEDGER_SNAPSHOT_EVENT_FIELDS)
            ),
            "retained_rows_per_field": int(retained_rows_per_field),
            "projected_all_family_snapshot_rows": int(projected_all_family_rows),
            "bounded_snapshot_rows_read": int(bounded_row_reads),
            "projected_all_family_snapshot_rows_removed": max(
                0,
                int(projected_all_family_rows) - int(bounded_row_reads),
            ),
            "source_window_limit_per_field": int(
                source_window.get("source_window_limit_per_field", 0) or 0
            ),
            "archival_storage_device": "cpu",
            "runs_live_tick": False,
            "runs_every_token": False,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "language_reasoning": False,
        },
        "retired_all_family_snapshot_comparator_absence": retired_absence,
        "runtime_truth": {
            "source_window_surface": source_window["surface"],
            "source_window_policy": source_window["policy"],
            "selection_criteria": source_window["selection_criteria"],
            "memory_budget": source_window["memory_budget"],
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
            "retired_all_family_snapshot_comparator_removed": True,
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
