from __future__ import annotations

import argparse
import json
import math
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


def _diagnostic_full_applied_synapse_audit(
    runtime: Mapping[str, Any],
    *,
    source_limit: int,
) -> dict[str, Any]:
    weights = runtime.get("sparse_transition_weights")
    weight_items = list(weights.items()) if isinstance(weights, Mapping) else []
    provenance = runtime.get("synapse_provenance_by_key")
    provenance_items = list(provenance.items()) if isinstance(provenance, Mapping) else []
    requested_hashes: list[str] = []
    finite_weight_count = 0
    bounded_weight_count = 0
    materialized_rows: list[dict[str, Any]] = []
    for _key, value in weight_items:
        try:
            weight = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(weight):
            continue
        finite_weight_count += 1
        if abs(weight) <= 1.0:
            bounded_weight_count += 1
    for _key, raw in provenance_items:
        if not isinstance(raw, Mapping):
            continue
        key = str(_key)
        readout_hash = str(raw.get("readout_evidence_hash") or "").strip()
        if readout_hash:
            requested_hashes.append(readout_hash)
        raw_weight = weights.get(_key) if isinstance(weights, Mapping) else None
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            weight = float("nan") if raw_weight is not None else None
        try:
            pre_text, post_text = key.split(":", 1)
            pre_index = int(pre_text)
            post_index = int(post_text)
        except (TypeError, ValueError):
            pre_index = None
            post_index = None
        source_pre_indices = [
            int(value)
            for value in list(raw.get("source_pre_indices") or [])
            if isinstance(value, int)
        ]
        source_post_indices = [
            int(value)
            for value in list(raw.get("source_post_indices") or [])
            if isinstance(value, int)
        ]
        source_active_indices = [
            int(value)
            for value in list(raw.get("source_active_indices") or [])
            if isinstance(value, int)
        ]
        materialized_rows.append(
            {
                "synapse_key": key,
                "weight_available": raw_weight is not None,
                "weight": weight,
                "weight_finite": bool(weight is not None and math.isfinite(float(weight))),
                "weight_bounded": bool(
                    weight is not None and math.isfinite(float(weight)) and abs(float(weight)) <= 1.0
                ),
                "pre_index": pre_index,
                "post_index": post_index,
                "readout_evidence_hash": readout_hash,
                "prediction_hash": raw.get("prediction_hash"),
                "transition_memory_evaluation_hash": raw.get("transition_memory_evaluation_hash"),
                "persistent_transition_weights_hash": raw.get("persistent_transition_weights_hash"),
                "source_pre_indices": source_pre_indices,
                "source_post_indices": source_post_indices,
                "source_active_indices": source_active_indices,
                "source_indices_match_synapse": bool(
                    pre_index in source_pre_indices
                    and post_index in source_post_indices
                    and pre_index in source_active_indices
                    and post_index in source_active_indices
                )
                if pre_index is not None and post_index is not None
                else False,
            }
        )
    return {
        "surface": "diagnostic_full_applied_synapse_provenance_audit_scan.v1",
        "records_scanned": int(len(weight_items) + len(provenance_items)),
        "weight_records_scanned": int(len(weight_items)),
        "provenance_records_scanned": int(len(provenance_items)),
        "rows_materialized": int(len(materialized_rows)),
        "requested_hash_count": int(len(requested_hashes)),
        "finite_weight_count": int(finite_weight_count),
        "bounded_weight_count": int(bounded_weight_count),
        "first_source_window_keys": [
            str(key) for key, _value in provenance_items[:source_limit]
        ],
        "production_callable": False,
        "benchmark_local_only": True,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    entry_count = max(1, int(args.entry_count))
    runs = max(1, int(args.runs))
    source_limit = SNN_READOUT_SYNAPSE_PROVENANCE_AUDIT_SOURCE_WINDOW_LIMIT
    ledger, runtime, _readout_hashes = _build_fixture(entry_count)
    cuda_before = _cuda_snapshot()
    diagnostic_latencies: list[float] = []
    bounded_latencies: list[float] = []
    diagnostic: dict[str, Any] = {}
    bounded: dict[str, Any] = {}
    tracemalloc.start()
    for _run in range(runs):
        started = time.perf_counter()
        diagnostic = _diagnostic_full_applied_synapse_audit(
            runtime,
            source_limit=source_limit,
        )
        diagnostic_latencies.append((time.perf_counter() - started) * 1000.0)

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
    diagnostic_keys = list(diagnostic.get("first_source_window_keys") or [])
    source_rows = int(source_window.get("source_window_count", 0) or 0)
    retained_rows = int(source_window.get("source_record_count", 0) or 0)
    reduction = float(diagnostic.get("records_scanned", 0) or 0) / max(
        1.0,
        float(source_rows * 2),
    )
    quality = {
        "metric": "bounded_applied_synapse_audit_matches_diagnostic_source_window",
        "bounded_synapse_keys": bounded_keys,
        "diagnostic_source_window_keys": diagnostic_keys,
        "source_window_keys_match": bounded_keys == diagnostic_keys,
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
        == min(entry_count, source_limit),
        "runs_live_tick_false": source_window.get("runs_live_tick") is False,
        "runs_every_token_false": source_window.get("runs_every_token") is False,
        "language_reasoning_false": source_window.get("language_reasoning") is False,
        "archival_metadata_cpu": source_window.get("archival_storage_device") == "cpu"
        and source_window.get("gpu_resident_archival_metadata") is False,
        "truncated_windows_block_exact_review": bool(
            quality["truncated_windows_block_exact_review"]
        ),
    }
    report = {
        "surface": "bounded_snn_readout_synapse_provenance_audit_source_window_benchmark.v1",
        "entry_count": entry_count,
        "runs": runs,
        "pass": all(bool(value) for value in pass_checks.values()),
        "pass_checks": pass_checks,
        "quality": quality,
        "latency_ms": {
            "diagnostic_full_applied_synapse_scan": _latency_stats(diagnostic_latencies),
            "bounded_applied_synapse_audit": _latency_stats(bounded_latencies),
        },
        "work_reduction": {
            "diagnostic_records_scanned": int(diagnostic.get("records_scanned", 0) or 0),
            "bounded_source_rows": source_rows,
            "retained_source_rows": retained_rows,
            "record_reduction": round(float(reduction), 6),
        },
        "diagnostic": diagnostic,
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
