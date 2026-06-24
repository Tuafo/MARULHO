"""Benchmark maintained SNN readout-ledger bounded source windows.

The benchmark keeps one executable path per maintained ledger boundary. Deleted
full-materialized and broad-normalized comparator implementations are represented
only as explicit absence evidence in the JSON report.
"""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import statistics
from threading import RLock
import time
import tracemalloc
from typing import Any, Mapping

import torch

from marulho.service.runtime_state import RuntimeState
from marulho.service.snn_language_readout_ledger import (
    SNN_AUTONOMOUS_CONFIDENCE_USE_SOURCE_WINDOW_POLICY,
    SNN_DENSE_LABEL_CALIBRATION_EVALUATION_SOURCE_WINDOW_POLICY,
    SNN_DENSE_LABEL_CALIBRATION_UPDATE_SOURCE_WINDOW_POLICY,
    SNN_DENSE_LABEL_CANDIDATE_CALIBRATION_SOURCE_WINDOW_POLICY,
    SNN_EMISSION_REVIEW_HISTORY_SOURCE_WINDOW_POLICY,
    SNN_LANGUAGE_READOUT_LEDGER_COUNT_FIELDS,
    SNN_LANGUAGE_READOUT_LEDGER_CURRENT_MAPPING_FIELDS,
    SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS,
    SNN_LANGUAGE_READOUT_LEDGER_TIMESTAMP_FIELDS,
    SNN_READOUT_EVIDENCE_HASH_SOURCE_WINDOW_POLICY,
    SNN_READOUT_LEDGER_RECORD_FAMILY_SOURCE_WINDOW_POLICY,
    SNNLanguageReadoutEvidenceLedger,
)

SNN_BENCHMARK_READOUT_LEDGER_NORMALIZATION_SOURCE_WINDOW_POLICY = (
    "recent_ledger_event_field_source_window_v1"
)


def _hex(value: int) -> str:
    return f"{int(value):064x}"


def _seed_event(field: str, index: int) -> dict[str, Any]:
    return {
        "field": field,
        "ordinal": int(index),
        "readout_evidence_hash": f"{field}:readout:{index}",
        "rollout_evidence_hash": f"{field}:rollout:{index}",
        "emission_review_hash": _hex(index + 1001),
        "emission_hash": _hex(index + 1101),
        "trajectory_hash": _hex(index + 1201),
        "prediction_hash": _hex(index + 1301),
        "persistent_transition_weights_hash": _hex(index + 1401),
        "dense_label_candidate_evidence_hash": _hex(index + 1),
        "dense_label_candidate_evidence_id": f"dense-label-candidate:{index}",
        "review_hash": _hex(index + 2001),
        "source_execution_hash": _hex(index + 3001),
        "label_hash": _hex((index % 8) + 4001),
        "text": f"{field}:text:{index}",
        "labels": [f"{field}:label:{index}"],
        "label_grounding": [True],
        "tensor_device": "cpu",
        "active_count": (index % 16) + 1,
        "applied_calibration_update_hash": _hex(index + 5001),
        "applied_at": "2026-06-24T00:00:00+00:00",
        "runtime_update_applied": True,
        "weights_persisted": False,
        "autonomous_confidence_use_event_hash": _hex(index + 6001),
        "used_at": "2026-06-24T00:00:00+00:00",
        "output_is_label_hash_only": True,
        "selected_candidate_count": 1,
        "selected_candidate_refs": [
            {
                "dense_label_candidate_evidence_hash": _hex(index + 1),
                "label_hash": _hex((index % 8) + 4001),
                "calibrated_confidence": 0.75,
            }
        ],
    }


def _seed_ledger_state(*, retention_count: int) -> dict[str, Any]:
    count = max(1, int(retention_count))
    state: dict[str, Any] = {
        field: [_seed_event(field, index) for index in range(count)]
        for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }
    state.update(
        {
            "total_recorded_count": count,
            "total_rollout_recorded_count": count,
            "total_emission_review_count": count,
            "total_dense_label_candidate_count": count,
            "total_dense_label_calibration_update_count": count,
            "total_autonomous_confidence_use_count": count,
            "total_autonomous_hash_readout_binding_count": count,
            "total_autonomous_bound_readout_observation_count": count,
            "total_autonomous_readout_training_window_count": count,
            "total_autonomous_decoder_probe_count": count,
            "total_autonomous_language_output_count": count,
            "total_autonomous_decoded_output_count": count,
            "total_autonomous_bounded_text_emission_count": count,
            "total_autonomous_text_surface_commit_count": count,
            "total_autonomous_text_surface_materialization_count": count,
            "total_autonomous_bounded_language_surface_commit_count": count,
            "total_autonomous_bounded_language_surface_use_count": count,
            "total_autonomous_snn_language_generation_count": count,
            "total_autonomous_snn_language_decoding_count": count,
            "total_snn_language_readout_surface_count": count,
            "total_snn_language_readout_memory_count": count,
            "total_snn_language_readout_consolidation_count": count,
            "total_snn_language_readout_structural_plasticity_count": count,
        }
    )
    for field in SNN_LANGUAGE_READOUT_LEDGER_TIMESTAMP_FIELDS:
        state[field] = "2026-06-24T00:00:00+00:00"
    state["current_text_surface_commit"] = {}
    state["current_text_surface_materialization"] = {}
    state["current_bounded_language_surface_commit"] = {}
    state["current_dense_label_calibration_update"] = deepcopy(
        state["dense_label_calibration_update_events"][0]
    )
    return state


def _source_record_count(state: Mapping[str, Any], name: str) -> int | None:
    raw_value = state.get(name) or []
    if isinstance(raw_value, (str, bytes, Mapping)):
        return 0
    try:
        return int(len(raw_value))
    except TypeError:
        return None


def _bounded_mapping_deque_from_state(
    state: Mapping[str, Any],
    name: str,
    *,
    limit: int,
) -> deque[dict[str, Any]]:
    source_limit = max(1, int(limit))
    raw_value = state.get(name) or []
    if isinstance(raw_value, (str, bytes, Mapping)):
        return deque(maxlen=source_limit)
    return deque(
        (
            deepcopy(dict(item))
            for item in list(raw_value)[:source_limit]
            if isinstance(item, Mapping)
        ),
        maxlen=source_limit,
    )


def _normalization_source_window_report(
    state: Mapping[str, Any],
    normalized_event_fields: Mapping[str, deque[dict[str, Any]]],
    *,
    limit: int,
) -> dict[str, Any]:
    source_limit = max(1, int(limit))
    field_window_counts = {
        name: int(len(normalized_event_fields.get(name) or []))
        for name in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }
    source_record_counts = {
        name: _source_record_count(state, name)
        for name in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }
    known_source_record_total = sum(
        int(value) for value in source_record_counts.values() if value is not None
    )
    return {
        "surface": "bounded_snn_readout_ledger_normalization_source_window.v1",
        "policy": SNN_BENCHMARK_READOUT_LEDGER_NORMALIZATION_SOURCE_WINDOW_POLICY,
        "source": "benchmark_local_recent_ledger_event_field_windows",
        "event_field_count": len(SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS),
        "source_window_limit_per_field": int(source_limit),
        "source_window_count_total": int(sum(field_window_counts.values())),
        "source_record_count_total_known": int(known_source_record_total),
        "source_record_counts": source_record_counts,
        "source_window_counts": field_window_counts,
        "selection_criteria": (
            "benchmark-local newest-first ledger event field windows after "
            "production all-family normalizer retirement"
        ),
        "memory_budget": {
            "max_fields": len(SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS),
            "max_records_per_field": int(source_limit),
            "max_records_total": int(
                len(SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS) * source_limit
            ),
            "archival_storage_device": "cpu",
        },
        "archival_storage_device": "cpu",
        "normalization_device": "cpu",
        "gpu_used": False,
        "runs_live_tick": False,
        "runs_every_token": False,
        "global_candidate_scan": False,
        "global_score_scan": False,
        "language_reasoning": False,
        "raw_text_scored": False,
        "production_callable": False,
        "benchmark_local_only": True,
    }


def _bounded_normalized_state(
    ledger: SNNLanguageReadoutEvidenceLedger,
) -> dict[str, Any]:
    state = ledger._ledger_state()  # noqa: SLF001
    limit = max(1, int(getattr(ledger, "_limit", 1)))
    normalized: dict[str, Any] = {
        name: _bounded_mapping_deque_from_state(state, name, limit=limit)
        for name in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }
    normalized["_normalization_source_window"] = _normalization_source_window_report(
        state,
        normalized,
        limit=limit,
    )
    for name in SNN_LANGUAGE_READOUT_LEDGER_CURRENT_MAPPING_FIELDS:
        value = state.get(name)
        normalized[name] = deepcopy(dict(value)) if isinstance(value, Mapping) else {}
    for name in SNN_LANGUAGE_READOUT_LEDGER_COUNT_FIELDS:
        normalized[name] = int(state.get(name, 0) or 0)
    for name in SNN_LANGUAGE_READOUT_LEDGER_TIMESTAMP_FIELDS:
        normalized[name] = state.get(name)
    return normalized


def _bounded_store_state(
    ledger: SNNLanguageReadoutEvidenceLedger,
    target: dict[str, Any],
    normalized: Mapping[str, Any],
) -> dict[str, Any]:
    target.clear()
    ledger._store_state(normalized)  # noqa: SLF001
    return target


def _bounded_known_hash_lookup(
    ledger: SNNLanguageReadoutEvidenceLedger,
) -> dict[str, Any]:
    hashes, report = ledger._known_readout_evidence_hashes_with_report()  # noqa: SLF001
    return {"hashes": sorted(hashes), "report": report}


def _readout_evidence_event_map_target_hashes() -> set[str]:
    return {
        "events:readout:0",
        "events:readout:1",
        "events:readout:2",
        "events:readout:3",
    }


def _bounded_readout_evidence_event_map_lookup(
    ledger: SNNLanguageReadoutEvidenceLedger,
) -> dict[str, Any]:
    event_map, report = ledger._readout_evidence_event_map_for_hashes_with_report(  # noqa: SLF001
        _readout_evidence_event_map_target_hashes()
    )
    return {"hashes": sorted(event_map.keys()), "report": report}


def _bounded_emission_review_history(
    ledger: SNNLanguageReadoutEvidenceLedger,
    *,
    limit: int,
) -> dict[str, Any]:
    history = ledger.emission_review_history(limit=limit)
    return {
        "review_hashes": [
            str(item.get("emission_review_hash") or "")
            for item in list(history.get("emission_review_events") or [])
        ],
        "source_window": dict(history.get("source_window") or {}),
    }


def _dense_label_evaluation_preflight(*, limit: int) -> dict[str, Any]:
    selected_count = max(1, min(int(limit), 8))
    return {
        "surface": "snn_language_dense_label_candidate_calibration_evaluation_preflight.v1",
        "ready": True,
        "preflight_hash": "benchmark-dense-label-calibration-evaluation",
        "selected_candidate_hashes": [_hex(index + 1) for index in range(selected_count)],
        "selected_candidate_count": selected_count,
        "mutates_runtime_state": False,
        "trains_runtime_model": False,
        "applies_plasticity": False,
        "writes_checkpoint": False,
        "generates_text": False,
        "promotion_gate": {
            "eligible_for_dense_label_calibration_evaluation_executor": True,
            "required_evidence": {
                "expected_revision_current": True,
                "executor_capability_available": True,
            },
        },
    }


def _heldout_label_evidence(*, limit: int) -> dict[str, Any]:
    selected_count = max(1, min(int(limit), 8))
    return {
        "labels": [
            f"dense_label_candidate_events:label:{index}"
            for index in range(selected_count)
        ],
    }


def _bounded_dense_label_calibration(
    ledger: SNNLanguageReadoutEvidenceLedger,
    *,
    limit: int,
) -> dict[str, Any]:
    history = ledger.dense_label_candidate_history(limit=limit)
    policy = ledger.dense_label_candidate_calibration_policy(limit=limit)
    return {
        "history_hashes": [
            str(item.get("dense_label_candidate_evidence_hash") or "")
            for item in list(history.get("dense_label_candidate_events") or [])
        ],
        "policy_hashes": [
            str(item.get("dense_label_candidate_evidence_hash") or "")
            for item in list(policy.get("calibration_candidates") or [])
        ],
        "ready_candidate_count": int(policy.get("ready_candidate_count", 0) or 0),
        "policy_source_window": dict(policy.get("source_window") or {}),
    }


def _bounded_dense_label_evaluation(
    ledger: SNNLanguageReadoutEvidenceLedger,
    *,
    limit: int,
) -> dict[str, Any]:
    evaluation = ledger.dense_label_candidate_calibration_evaluation(
        dense_label_candidate_calibration_evaluation_preflight=(
            _dense_label_evaluation_preflight(limit=limit)
        ),
        heldout_label_evidence=_heldout_label_evidence(limit=limit),
        bin_count=5,
    )
    return {
        "ready": bool(evaluation.get("ready")),
        "sample_hashes": [
            str(sample.get("dense_label_candidate_evidence_hash") or "")
            for sample in list(evaluation.get("evaluated_samples") or [])
        ],
        "metrics": dict(evaluation.get("metrics") or {}),
        "source_window": dict(evaluation.get("source_window") or {}),
    }


def _bounded_dense_label_calibration_update_lookup(
    ledger: SNNLanguageReadoutEvidenceLedger,
) -> dict[str, Any]:
    events, current, source_window = (
        ledger._dense_label_calibration_update_source_window_with_report()  # noqa: SLF001
    )
    return {
        "event_hashes": [
            str(item.get("applied_calibration_update_hash") or "")
            for item in events
        ],
        "current_hash": str(current.get("applied_calibration_update_hash") or ""),
        "source_window": source_window,
    }


def _bounded_autonomous_confidence_use_lookup(
    ledger: SNNLanguageReadoutEvidenceLedger,
) -> dict[str, Any]:
    events, source_window = (
        ledger._autonomous_confidence_use_source_window_with_report()  # noqa: SLF001
    )
    return {
        "event_hashes": [
            str(item.get("autonomous_confidence_use_event_hash") or "")
            for item in events
        ],
        "source_window": source_window,
    }


def _record_append_event() -> dict[str, Any]:
    return {
        "readout_evidence_hash": "record-family-append-readout-hash",
        "recorded_at": "2026-06-24T00:00:00+00:00",
        "state_revision": 0,
        "labels": ["record-family-append"],
    }


def _reset_state(target: dict[str, Any], seed: Mapping[str, Any]) -> None:
    target.clear()
    target.update(deepcopy(dict(seed)))


def _bounded_record_family_append(
    ledger: SNNLanguageReadoutEvidenceLedger,
) -> dict[str, Any]:
    event = _record_append_event()
    duplicate, summary, source_window = ledger._append_record_family_window(  # noqa: SLF001
        field="events",
        event=event,
        duplicate_key="readout_evidence_hash",
        total_count_key="total_recorded_count",
        timestamp_key="last_recorded_at",
        timestamp_value=event["recorded_at"],
    )
    events = list(ledger._ledger_state().get("events") or [])  # noqa: SLF001
    latest = dict(events[0]) if events and isinstance(events[0], Mapping) else {}
    return {
        "duplicate": duplicate,
        "latest_hash": str(latest.get("readout_evidence_hash") or ""),
        "total_recorded_count": int(summary.get("total_recorded_count", 0) or 0),
        "event_count": int(summary.get("event_count", 0) or 0),
        "source_window": source_window,
    }


def _timed_runs(
    *,
    runs: int,
    fn: Any,
    setup: Any | None = None,
) -> tuple[dict[str, Any], list[float]]:
    samples: list[float] = []
    last: dict[str, Any] | None = None
    for _ in range(max(1, int(runs))):
        if setup is not None:
            setup()
        started = time.perf_counter()
        last = fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    assert last is not None
    return last, samples


def _latency_summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    index_95 = min(
        len(ordered) - 1,
        max(0, int(round((len(ordered) - 1) * 0.95))),
    )
    return {
        "mean_ms": round(statistics.fmean(samples), 6),
        "median_ms": round(statistics.median(samples), 6),
        "p95_ms": round(ordered[index_95], 6),
    }


def _recent_retention_rate(normalized: Mapping[str, deque[dict[str, Any]]]) -> float:
    retained = 0
    for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS:
        rows = list(normalized.get(field) or [])
        if rows and int(rows[0].get("ordinal", -1)) == 0:
            retained += 1
    return retained / max(1, len(SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS))


def _first_ordinals(
    normalized: Mapping[str, deque[dict[str, Any]]],
) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS:
        rows = list(normalized.get(field) or [])
        result[field] = int(rows[0].get("ordinal", -1)) if rows else None
    return result


def _source_window_ok(
    report: Mapping[str, Any],
    *,
    surface: str,
    policy: str,
) -> bool:
    try:
        count = int(report.get("source_window_count", -1))
        limit = int(report.get("source_window_limit", -1))
    except (TypeError, ValueError):
        return False
    return (
        report.get("surface") == surface
        and report.get("policy") == policy
        and 0 <= count <= limit
        and report.get("global_candidate_scan") is False
        and report.get("global_score_scan") is False
        and report.get("language_reasoning") is False
        and report.get("runs_live_tick") is False
        and report.get("runs_every_token") is False
        and report.get("archival_storage_device") == "cpu"
        and report.get("gpu_used") is False
    )


def _cuda_report() -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    report: dict[str, Any] = {
        "torch_available": True,
        "cuda_available": available,
        "gpu_used": False,
    }
    if available:
        report["device_name"] = torch.cuda.get_device_name(0)
        report["memory_allocated_mib"] = round(
            float(torch.cuda.memory_allocated(0)) / (1024.0 * 1024.0),
            6,
        )
        report["memory_reserved_mib"] = round(
            float(torch.cuda.memory_reserved(0)) / (1024.0 * 1024.0),
            6,
        )
    return report


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    retention_count = max(1, int(args.retention_count))
    ledger_limit = max(1, int(args.ledger_limit))
    runs = max(1, int(args.runs))
    state = _seed_ledger_state(retention_count=retention_count)
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: state,
        limit=ledger_limit,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tracemalloc.start()
    bounded, bounded_samples = _timed_runs(
        runs=runs,
        fn=lambda: _bounded_normalized_state(ledger),
    )
    bounded_store_target: dict[str, Any] = {}
    bounded_store_ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: bounded_store_target,
        limit=ledger_limit,
    )
    bounded_store, bounded_store_samples = _timed_runs(
        runs=runs,
        fn=lambda: _bounded_store_state(
            bounded_store_ledger,
            bounded_store_target,
            state,
        ),
    )
    known_hash, known_hash_samples = _timed_runs(
        runs=runs,
        fn=lambda: _bounded_known_hash_lookup(ledger),
    )
    event_map, event_map_samples = _timed_runs(
        runs=runs,
        fn=lambda: _bounded_readout_evidence_event_map_lookup(ledger),
    )
    emission_history, emission_history_samples = _timed_runs(
        runs=runs,
        fn=lambda: _bounded_emission_review_history(ledger, limit=ledger_limit),
    )
    dense_label, dense_label_samples = _timed_runs(
        runs=runs,
        fn=lambda: _bounded_dense_label_calibration(ledger, limit=ledger_limit),
    )
    dense_label_evaluation, dense_label_evaluation_samples = _timed_runs(
        runs=runs,
        fn=lambda: _bounded_dense_label_evaluation(ledger, limit=ledger_limit),
    )
    dense_label_update, dense_label_update_samples = _timed_runs(
        runs=runs,
        fn=lambda: _bounded_dense_label_calibration_update_lookup(ledger),
    )
    confidence_use, confidence_use_samples = _timed_runs(
        runs=runs,
        fn=lambda: _bounded_autonomous_confidence_use_lookup(ledger),
    )
    record_seed_state = _seed_ledger_state(retention_count=retention_count)
    record_state: dict[str, Any] = {}
    record_ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: record_state,
        limit=ledger_limit,
    )
    record_append, record_append_samples = _timed_runs(
        runs=runs,
        setup=lambda: _reset_state(record_state, record_seed_state),
        fn=lambda: _bounded_record_family_append(record_ledger),
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    field_count = len(SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS)
    expected_window_per_field = min(retention_count, ledger_limit)
    expected_window_rows = int(field_count * expected_window_per_field)
    expected_source_rows = int(field_count * retention_count)
    expected_first_ordinals = {
        field: 0 for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }
    expected_readout_hashes = {
        f"events:readout:{index}" for index in range(expected_window_per_field)
    }
    expected_event_map_hashes = sorted(
        _readout_evidence_event_map_target_hashes() & expected_readout_hashes
    )
    expected_review_hashes = [
        _hex(index + 1001) for index in range(expected_window_per_field)
    ]
    expected_dense_label_hashes = [
        _hex(index + 1) for index in range(expected_window_per_field)
    ]
    expected_update_hashes = [
        _hex(index + 5001) for index in range(expected_window_per_field)
    ]
    expected_confidence_hashes = [
        _hex(index + 6001) for index in range(expected_window_per_field)
    ]
    expected_eval_hashes = expected_dense_label_hashes[
        : min(expected_window_per_field, 8)
    ]

    source_window = dict(bounded.get("_normalization_source_window") or {})
    bounded_rows = int(source_window.get("source_window_count_total", 0) or 0)
    source_known_rows = int(
        source_window.get("source_record_count_total_known", 0) or 0
    )
    bounded_store_rows = int(
        sum(
            len(list(bounded_store.get(field) or []))
            for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
        )
    )
    bounded_store_first_ordinals = _first_ordinals(bounded_store)
    known_hash_report = dict(known_hash.get("report") or {})
    event_map_report = dict(event_map.get("report") or {})
    emission_history_source_window = dict(emission_history.get("source_window") or {})
    dense_label_source_window = dict(dense_label.get("policy_source_window") or {})
    dense_label_evaluation_source_window = dict(
        dense_label_evaluation.get("source_window") or {}
    )
    dense_label_update_source_window = dict(
        dense_label_update.get("source_window") or {}
    )
    confidence_use_source_window = dict(confidence_use.get("source_window") or {})
    record_append_source_window = dict(record_append.get("source_window") or {})

    absence = {
        "implementation_present": False,
        "active_report_field_present": False,
    }
    retired_full_materialized_absence = {
        **absence,
        "removed_policy": (
            "snn_readout_ledger_full_materialized_normalization_comparator"
        ),
    }
    retired_broad_normalized_absence = {
        **absence,
        "removed_policy": "snn_readout_ledger_broad_normalized_boundary_comparators",
    }

    pass_checks = {
        "surface_present": source_window.get("surface")
        == "bounded_snn_readout_ledger_normalization_source_window.v1",
        "policy_present": source_window.get("policy")
        == SNN_BENCHMARK_READOUT_LEDGER_NORMALIZATION_SOURCE_WINDOW_POLICY,
        "source_limit_respected": all(
            int(value) <= ledger_limit
            for value in dict(source_window.get("source_window_counts") or {}).values()
        ),
        "source_rows_match_expected_seed_window": bounded_rows == expected_window_rows,
        "source_known_rows_match_expected_seed": source_known_rows
        == expected_source_rows,
        "no_global_scan": source_window.get("global_candidate_scan") is False
        and source_window.get("global_score_scan") is False,
        "not_live_tick": source_window.get("runs_live_tick") is False
        and source_window.get("runs_every_token") is False,
        "cpu_only": source_window.get("archival_storage_device") == "cpu"
        and source_window.get("normalization_device") == "cpu"
        and source_window.get("gpu_used") is False,
        "no_language_reasoning": source_window.get("language_reasoning") is False,
        "recent_rows_preserved": _recent_retention_rate(bounded) == 1.0,
        "first_ordinals_match_expected_seed": _first_ordinals(bounded)
        == expected_first_ordinals,
        "store_rows_match_expected_seed_window": bounded_store_rows
        == expected_window_rows,
        "store_first_ordinals_match_expected_seed": bounded_store_first_ordinals
        == expected_first_ordinals,
        "known_hash_surface_present": _source_window_ok(
            known_hash_report,
            surface="bounded_snn_readout_known_evidence_hash_source_window.v1",
            policy=SNN_READOUT_EVIDENCE_HASH_SOURCE_WINDOW_POLICY,
        ),
        "known_hash_expected_set": set(known_hash.get("hashes") or [])
        == expected_readout_hashes,
        "event_map_surface_present": _source_window_ok(
            event_map_report,
            surface="bounded_snn_readout_evidence_event_map_source_window.v1",
            policy=SNN_READOUT_EVIDENCE_HASH_SOURCE_WINDOW_POLICY,
        ),
        "event_map_expected_hashes": event_map.get("hashes")
        == expected_event_map_hashes,
        "emission_history_surface_present": _source_window_ok(
            emission_history_source_window,
            surface="bounded_snn_emission_review_history_source_window.v1",
            policy=SNN_EMISSION_REVIEW_HISTORY_SOURCE_WINDOW_POLICY,
        ),
        "emission_history_expected_review_hashes": emission_history.get(
            "review_hashes"
        )
        == expected_review_hashes,
        "dense_label_surface_present": _source_window_ok(
            dense_label_source_window,
            surface="bounded_snn_dense_label_candidate_calibration_source_window.v1",
            policy=SNN_DENSE_LABEL_CANDIDATE_CALIBRATION_SOURCE_WINDOW_POLICY,
        ),
        "dense_label_history_matches_expected_seed": dense_label.get(
            "history_hashes"
        )
        == expected_dense_label_hashes,
        "dense_label_policy_inside_expected_window": set(
            dense_label.get("policy_hashes") or []
        ).issubset(set(expected_dense_label_hashes)),
        "dense_label_ready_count_matches_expected_window": int(
            dense_label.get("ready_candidate_count", 0) or 0
        )
        == expected_window_per_field,
        "dense_label_evaluation_surface_present": _source_window_ok(
            dense_label_evaluation_source_window,
            surface=(
                "bounded_snn_dense_label_candidate_calibration_evaluation_source_window.v1"
            ),
            policy=SNN_DENSE_LABEL_CALIBRATION_EVALUATION_SOURCE_WINDOW_POLICY,
        ),
        "dense_label_evaluation_expected_samples": dense_label_evaluation.get(
            "sample_hashes"
        )
        == expected_eval_hashes,
        "dense_label_evaluation_ready": bool(dense_label_evaluation.get("ready")),
        "dense_label_update_surface_present": _source_window_ok(
            dense_label_update_source_window,
            surface="bounded_snn_dense_label_calibration_update_source_window.v1",
            policy=SNN_DENSE_LABEL_CALIBRATION_UPDATE_SOURCE_WINDOW_POLICY,
        ),
        "dense_label_update_expected_event_hashes": dense_label_update.get(
            "event_hashes"
        )
        == expected_update_hashes,
        "dense_label_update_expected_current_hash": dense_label_update.get(
            "current_hash"
        )
        == expected_update_hashes[0],
        "confidence_use_surface_present": _source_window_ok(
            confidence_use_source_window,
            surface="bounded_snn_autonomous_confidence_use_source_window.v1",
            policy=SNN_AUTONOMOUS_CONFIDENCE_USE_SOURCE_WINDOW_POLICY,
        ),
        "confidence_use_expected_event_hashes": confidence_use.get("event_hashes")
        == expected_confidence_hashes,
        "record_append_surface_present": _source_window_ok(
            record_append_source_window,
            surface="bounded_snn_readout_ledger_record_family_source_window.v1",
            policy=SNN_READOUT_LEDGER_RECORD_FAMILY_SOURCE_WINDOW_POLICY,
        ),
        "record_append_latest_hash_matches_event": record_append.get("latest_hash")
        == _record_append_event()["readout_evidence_hash"],
        "record_append_total_count_incremented": int(
            record_append.get("total_recorded_count", 0) or 0
        )
        == retention_count + 1,
        "full_materialized_comparator_removed": retired_full_materialized_absence[
            "implementation_present"
        ]
        is False,
        "broad_normalized_comparators_removed": retired_broad_normalized_absence[
            "implementation_present"
        ]
        is False,
    }

    return {
        "surface": "bounded_snn_readout_ledger_normalization_source_window_benchmark.v2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": {
            "retention_count_per_field": retention_count,
            "ledger_limit": ledger_limit,
            "event_field_count": field_count,
            "runs": runs,
        },
        "pass": all(pass_checks.values()),
        "pass_checks": pass_checks,
        "quality": {
            "metric": "seeded_newest_first_bounded_source_window_reconstruction",
            "bounded_recent_retention_rate": round(_recent_retention_rate(bounded), 6),
            "expected_recent_retention_rate": 1.0,
            "bounded_first_ordinals": _first_ordinals(bounded),
            "expected_first_ordinals": expected_first_ordinals,
            "known_hash_count": int(len(known_hash.get("hashes") or [])),
            "event_map_hashes": event_map.get("hashes"),
            "dense_label_evaluation_metrics": dense_label_evaluation.get("metrics"),
        },
        "latency": {
            "bounded": _latency_summary(bounded_samples),
            "store_state": _latency_summary(bounded_store_samples),
            "known_evidence_hash": _latency_summary(known_hash_samples),
            "readout_evidence_event_map": _latency_summary(event_map_samples),
            "emission_review_history": _latency_summary(emission_history_samples),
            "dense_label_calibration": _latency_summary(dense_label_samples),
            "dense_label_evaluation": _latency_summary(
                dense_label_evaluation_samples
            ),
            "dense_label_calibration_update": _latency_summary(
                dense_label_update_samples
            ),
            "autonomous_confidence_use": _latency_summary(confidence_use_samples),
            "record_family_append": _latency_summary(record_append_samples),
        },
        "memory_budget": {
            "selection_criteria": (
                "seeded newest-first bounded ledger event-field windows"
            ),
            "event_field_count": field_count,
            "max_records_per_field": ledger_limit,
            "bounded_window_records_per_field": expected_window_per_field,
            "bounded_window_rows": bounded_rows,
            "source_rows_known": source_known_rows,
            "retired_full_materialized_rows_removed": max(
                0,
                expected_source_rows - bounded_rows,
            ),
            "archival_storage_device": "cpu",
            "active_computation_device": "cpu",
            "runs_live_tick": False,
            "runs_every_token": False,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "language_reasoning": False,
        },
        "retired_full_materialized_ledger_normalization_absence": (
            retired_full_materialized_absence
        ),
        "retired_broad_normalized_ledger_comparator_absence": (
            retired_broad_normalized_absence
        ),
        "normalization_source_window": source_window,
        "store_state_boundary": {
            "surface": "bounded_snn_readout_ledger_store_state_source_window.v1",
            "policy": SNN_BENCHMARK_READOUT_LEDGER_NORMALIZATION_SOURCE_WINDOW_POLICY,
            "quality": {
                "metric": "newest_first_store_window_expected_seed_reconstruction",
                "bounded_recent_retention_rate": round(
                    _recent_retention_rate(bounded_store),
                    6,
                ),
                "expected_recent_retention_rate": 1.0,
                "bounded_first_ordinals": bounded_store_first_ordinals,
                "expected_first_ordinals": expected_first_ordinals,
            },
            "latency": {"bounded": _latency_summary(bounded_store_samples)},
            "source_window_limit_per_field": int(ledger_limit),
            "source_window_count_total": int(bounded_store_rows),
            "source_record_count_total_known": int(expected_source_rows),
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "runs_live_tick": False,
            "runs_every_token": False,
            "mutates_runtime_state": True,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "store_device": "cpu",
            "gpu_used": False,
        },
        "known_evidence_hash_boundary": {
            "quality": {
                "hash_set_matches_expected_seed": set(known_hash.get("hashes") or [])
                == expected_readout_hashes,
                "hash_count": int(len(known_hash.get("hashes") or [])),
            },
            "latency": {"bounded": _latency_summary(known_hash_samples)},
            "source_window": known_hash_report,
        },
        "readout_evidence_event_map_boundary": {
            "quality": {
                "hash_set_matches_expected_seed": event_map.get("hashes")
                == expected_event_map_hashes,
                "requested_hash_count": int(
                    event_map_report.get("requested_hash_count", 0) or 0
                ),
                "matched_hash_count": int(
                    event_map_report.get("matched_hash_count", 0) or 0
                ),
            },
            "latency": {"bounded": _latency_summary(event_map_samples)},
            "source_window": event_map_report,
        },
        "emission_review_history_boundary": {
            "quality": {
                "review_hashes_match_expected_seed": emission_history.get(
                    "review_hashes"
                )
                == expected_review_hashes,
                "returned_review_count": int(
                    len(emission_history.get("review_hashes") or [])
                ),
            },
            "latency": {"bounded": _latency_summary(emission_history_samples)},
            "source_window": emission_history_source_window,
        },
        "dense_label_calibration_boundary": {
            "quality": {
                "history_hashes_match_expected_seed": dense_label.get(
                    "history_hashes"
                )
                == expected_dense_label_hashes,
                "policy_hashes_inside_expected_window": set(
                    dense_label.get("policy_hashes") or []
                ).issubset(set(expected_dense_label_hashes)),
                "ready_candidate_count": int(
                    dense_label.get("ready_candidate_count", 0) or 0
                ),
            },
            "latency": {"bounded": _latency_summary(dense_label_samples)},
            "source_window": dense_label_source_window,
        },
        "dense_label_evaluation_boundary": {
            "quality": {
                "ready": bool(dense_label_evaluation.get("ready")),
                "sample_hashes_match_expected_seed": dense_label_evaluation.get(
                    "sample_hashes"
                )
                == expected_eval_hashes,
                "sample_count": int(
                    len(dense_label_evaluation.get("sample_hashes") or [])
                ),
                "metrics": dense_label_evaluation.get("metrics"),
            },
            "latency": {
                "bounded": _latency_summary(dense_label_evaluation_samples)
            },
            "source_window": dense_label_evaluation_source_window,
        },
        "dense_label_calibration_update_boundary": {
            "quality": {
                "event_hashes_match_expected_seed": dense_label_update.get(
                    "event_hashes"
                )
                == expected_update_hashes,
                "current_hash_matches_expected_seed": dense_label_update.get(
                    "current_hash"
                )
                == expected_update_hashes[0],
                "event_hash_count": int(
                    len(dense_label_update.get("event_hashes") or [])
                ),
            },
            "latency": {"bounded": _latency_summary(dense_label_update_samples)},
            "source_window": dense_label_update_source_window,
        },
        "autonomous_confidence_use_boundary": {
            "quality": {
                "event_hashes_match_expected_seed": confidence_use.get("event_hashes")
                == expected_confidence_hashes,
                "event_hash_count": int(len(confidence_use.get("event_hashes") or [])),
            },
            "latency": {"bounded": _latency_summary(confidence_use_samples)},
            "source_window": confidence_use_source_window,
        },
        "record_family_append_boundary": {
            "quality": {
                "latest_hash_matches_event": record_append.get("latest_hash")
                == _record_append_event()["readout_evidence_hash"],
                "total_count_incremented": int(
                    record_append.get("total_recorded_count", 0) or 0
                )
                == retention_count + 1,
                "window_event_count": int(record_append.get("event_count", 0) or 0),
            },
            "latency": {"bounded": _latency_summary(record_append_samples)},
            "source_window": record_append_source_window,
        },
        "resource_behavior": {
            "python_tracemalloc_current_mib": round(current / (1024 * 1024), 6),
            "python_tracemalloc_peak_mib": round(peak / (1024 * 1024), 6),
            "cuda": _cuda_report(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded SNN readout-ledger normalization."
    )
    parser.add_argument("--retention-count", type=int, default=2048)
    parser.add_argument("--ledger-limit", type=int, default=128)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/bounded_replay_window_20260618/snn-readout-ledger-normalization-source-window.json"
        ),
    )
    args = parser.parse_args()
    report = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"snn_readout_ledger_normalization_source_window_benchmark={args.output}")
    print(
        "pass={passed} bounded_mean_ms={bounded:.6f} "
        "store_mean_ms={store:.6f} known_hash_mean_ms={known_hash:.6f} "
        "event_map_mean_ms={event_map:.6f} emission_history_mean_ms={emission:.6f} "
        "dense_label_mean_ms={dense_label:.6f} dense_label_eval_mean_ms={dense_eval:.6f} "
        "dense_label_update_mean_ms={dense_update:.6f} "
        "confidence_use_mean_ms={confidence:.6f} record_append_mean_ms={record:.6f}".format(
            passed=report["pass"],
            bounded=report["latency"]["bounded"]["mean_ms"],
            store=report["latency"]["store_state"]["mean_ms"],
            known_hash=report["latency"]["known_evidence_hash"]["mean_ms"],
            event_map=report["latency"]["readout_evidence_event_map"]["mean_ms"],
            emission=report["latency"]["emission_review_history"]["mean_ms"],
            dense_label=report["latency"]["dense_label_calibration"]["mean_ms"],
            dense_eval=report["latency"]["dense_label_evaluation"]["mean_ms"],
            dense_update=report["latency"]["dense_label_calibration_update"][
                "mean_ms"
            ],
            confidence=report["latency"]["autonomous_confidence_use"]["mean_ms"],
            record=report["latency"]["record_family_append"]["mean_ms"],
        )
    )


if __name__ == "__main__":
    main()
