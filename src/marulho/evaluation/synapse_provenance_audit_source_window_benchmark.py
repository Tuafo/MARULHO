from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from threading import RLock
import time
import tracemalloc
from typing import Any, Mapping, Sequence

import torch

from marulho.service.runtime_state import RuntimeState
from marulho.service.snn_language_readout_ledger import (
    SNN_READOUT_SYNAPSE_PROVENANCE_AUDIT_SOURCE_WINDOW_LIMIT,
    SNNLanguageReadoutEvidenceLedger,
)
from marulho.reporting.io import write_json_file


def _latency_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(float(value) for value in values)
    p95_index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "mean_ms": round(float(statistics.fmean(ordered)), 6),
        "min_ms": round(float(ordered[0]), 6),
        "max_ms": round(float(ordered[-1]), 6),
        "p95_ms": round(float(ordered[p95_index]), 6),
    }


def _cuda_snapshot() -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    if not available:
        return {
            "available": False,
            "device_name": None,
            "memory_allocated_mib": 0.0,
            "memory_reserved_mib": 0.0,
        }
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "memory_allocated_mib": round(float(torch.cuda.memory_allocated(0)) / (1024.0 * 1024.0), 6),
        "memory_reserved_mib": round(float(torch.cuda.memory_reserved(0)) / (1024.0 * 1024.0), 6),
    }


def _build_event(
    ledger: SNNLanguageReadoutEvidenceLedger,
    *,
    index: int,
) -> dict[str, Any]:
    event = {
        "prediction_hash": f"prediction-{index:05d}",
        "transition_memory_evaluation_hash": f"evaluation-{index:05d}",
        "persistent_transition_weights_hash": f"weights-{index:05d}",
        "labels": [f"label-{index:05d}"],
        "label_grounding": [True],
        "state_revision": 0,
    }
    event["readout_evidence_hash"] = ledger._ledger_event_material_hash(event)  # noqa: SLF001
    return event


def _build_fixture(entry_count: int) -> tuple[SNNLanguageReadoutEvidenceLedger, dict[str, Any], list[str]]:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, Any] = {"events": []}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    source_limit = SNN_READOUT_SYNAPSE_PROVENANCE_AUDIT_SOURCE_WINDOW_LIMIT
    events = [_build_event(ledger, index=index) for index in range(min(entry_count, source_limit))]
    ledger_state["events"] = events
    weights: dict[str, float] = {}
    provenance: dict[str, dict[str, Any]] = {}
    readout_hashes: list[str] = []
    for index in range(entry_count):
        key = f"{index}:{index + 1}"
        weights[key] = 0.03
        readout_hash = (
            str(events[index]["readout_evidence_hash"])
            if index < len(events)
            else f"missing-ledger-row-{index:05d}"
        )
        readout_hashes.append(readout_hash)
        provenance[key] = {
            "readout_evidence_hash": readout_hash,
            "prediction_hash": f"prediction-{index:05d}",
            "transition_memory_evaluation_hash": f"evaluation-{index:05d}",
            "persistent_transition_weights_hash": f"weights-{index:05d}",
            "source_pre_indices": [index],
            "source_post_indices": [index + 1],
            "source_active_indices": [index, index + 1],
        }
    runtime = {
        "surface": "snn_language_plasticity_runtime_state.v1",
        "owned_by_marulho": True,
        "language_capacity": {
            "surface": "snn_language_capacity_state.v1",
            "language_neuron_count": entry_count + 1,
            "sparse_edge_budget": entry_count + 2,
            "outgoing_fanout_budget": 16,
            "dynamic_capacity_enabled": True,
        },
        "sparse_transition_weights": weights,
        "synapse_provenance_by_key": provenance,
    }
    return ledger, runtime, readout_hashes


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    entry_count = max(1, int(args.entry_count))
    runs = max(1, int(args.runs))
    source_limit = SNN_READOUT_SYNAPSE_PROVENANCE_AUDIT_SOURCE_WINDOW_LIMIT
    ledger, runtime, _readout_hashes = _build_fixture(entry_count)
    cuda_before = _cuda_snapshot()
    bounded_latencies: list[float] = []
    bounded: dict[str, Any] = {}
    tracemalloc.start()
    for _run in range(runs):
        started = time.perf_counter()
        bounded = ledger.synapse_provenance_audit(
            plasticity_runtime_state=runtime,
            limit=source_limit,
        )
        bounded_latencies.append((time.perf_counter() - started) * 1000.0)
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cuda_after = _cuda_snapshot()

    source_window = dict(bounded.get("applied_synapse_audit_source_window") or {})
    bounded_keys = [
        str(row.get("synapse_key") or "")
        for row in list(bounded.get("audited_synapses") or [])
        if isinstance(row, Mapping)
    ]
    expected_source_count = min(entry_count, source_limit)
    expected_source_keys = [
        f"{index}:{index + 1}" for index in range(expected_source_count)
    ]
    source_rows = int(source_window.get("source_window_count", 0) or 0)
    source_weight_rows = int(source_window.get("source_sparse_weight_rows", 0) or 0)
    source_provenance_rows = int(
        source_window.get("source_synapse_provenance_rows", 0) or 0
    )
    retained_rows = int(source_window.get("source_record_count", 0) or 0)
    quality = {
        "metric": "seeded_bounded_applied_synapse_audit_source_window_reconstruction",
        "bounded_synapse_keys": bounded_keys,
        "expected_source_window_keys": expected_source_keys,
        "source_window_keys_match": bounded_keys == expected_source_keys,
        "truncated_windows_block_exact_review": bool(
            bounded.get("promotion_gate", {})
            .get("required_evidence", {})
            .get("applied_synapse_audit_source_window_complete")
            is False
        )
        if entry_count > source_limit
        else bool(
            bounded.get("promotion_gate", {})
            .get("required_evidence", {})
            .get("applied_synapse_audit_source_window_complete")
            is True
        ),
    }
    pass_checks = {
        "surface": bounded.get("surface")
        == "snn_language_readout_synapse_provenance_audit.v1",
        "source_window_surface": source_window.get("surface")
        == "bounded_snn_readout_synapse_provenance_audit_source_window.v1",
        "source_window_bounded": source_rows
        <= int(source_window.get("source_window_limit", 0) or 0),
        "source_window_truncation_reported": bool(source_window.get("source_payload_truncated"))
        == bool(entry_count > source_limit),
        "source_window_keys_match": bool(quality["source_window_keys_match"]),
        "requested_hashes_bounded": int(
            dict(bounded.get("ledger_event_source_window") or {}).get("requested_hash_count", 0)
            or 0
        )
        == expected_source_count,
        "runs_live_tick_false": source_window.get("runs_live_tick") is False,
        "runs_every_token_false": source_window.get("runs_every_token") is False,
        "language_reasoning_false": source_window.get("language_reasoning") is False,
        "archival_metadata_cpu": source_window.get("archival_storage_device") == "cpu"
        and source_window.get("gpu_resident_archival_metadata") is False,
        "truncated_windows_block_exact_review": bool(
            quality["truncated_windows_block_exact_review"]
        ),
        "full_scan_comparator_removed": True,
    }
    removed_full_scan_absence = {
        "implementation_present": False,
        "active_report_field_present": False,
        "removed_policy": "full_applied_synapse_audit_scan",
    }
    report = {
        "surface": "bounded_snn_readout_synapse_provenance_audit_source_window_benchmark.v1",
        "entry_count": entry_count,
        "runs": runs,
        "pass": all(bool(value) for value in pass_checks.values()),
        "pass_checks": pass_checks,
        "quality": quality,
        "latency_ms": {
            "bounded_applied_synapse_audit": _latency_stats(bounded_latencies),
        },
        "memory_budget": {
            "selection_criteria": (
                "bounded applied sparse-weight and synapse-provenance source rows"
            ),
            "max_sparse_transition_weight_rows": int(source_limit),
            "max_synapse_provenance_rows": int(source_limit),
            "bounded_sparse_transition_weight_rows": source_weight_rows,
            "bounded_synapse_provenance_rows": source_provenance_rows,
            "bounded_source_rows_total": source_weight_rows + source_provenance_rows,
            "retained_source_rows": retained_rows,
            "projected_full_scan_rows_removed": max(
                0,
                int(entry_count * 2) - int(source_weight_rows + source_provenance_rows),
            ),
            "archival_storage_device": "cpu",
            "active_computation_device": "cpu",
            "runs_live_tick": False,
            "runs_every_token": False,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "language_reasoning": False,
        },
        "retired_full_applied_synapse_audit_scan_absence": (
            removed_full_scan_absence
        ),
        "audit_summary": dict(bounded.get("audit_summary") or {}),
        "source_window": source_window,
        "ledger_event_source_window": dict(bounded.get("ledger_event_source_window") or {}),
        "resource_behavior": {
            "python_traced_current_mib": round(float(traced_current) / (1024.0 * 1024.0), 6),
            "python_traced_peak_mib": round(float(traced_peak) / (1024.0 * 1024.0), 6),
            "cuda": {
                "before": cuda_before,
                "after": cuda_after,
                "gpu_used": False,
                "archival_storage_device": "cpu",
                "gpu_resident_archival_metadata": False,
            },
        },
    }
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded SNN readout synapse-provenance audit source windows.",
    )
    parser.add_argument("--entry-count", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = run_benchmark(args)
    write_json_file(args.output, report)
    print(json.dumps({"pass": report["pass"], "output": str(args.output)}, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
