"""Benchmark bounded SNN readout-ledger normalization.

The production normalizer reads each retained event family through a newest-first
source window. The diagnostic legacy path below preserves the retired
full-materialize-then-cap shape for latency, work, and recent-row retention
comparison only.
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
    SNN_DENSE_LABEL_CANDIDATE_CALIBRATION_SOURCE_WINDOW_POLICY,
    SNN_DENSE_LABEL_CALIBRATION_UPDATE_SOURCE_WINDOW_POLICY,
    SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS,
    SNN_LANGUAGE_READOUT_LEDGER_NORMALIZATION_SOURCE_WINDOW_POLICY,
    SNN_READOUT_EVIDENCE_HASH_SOURCE_WINDOW_POLICY,
    SNNLanguageReadoutEvidenceLedger,
)


def _seed_ledger_state(*, retention_count: int) -> dict[str, Any]:
    count = max(1, int(retention_count))
    state: dict[str, Any] = {}
    for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS:
        rows: list[dict[str, Any]] = []
        for index in range(count):
            rows.append(
                {
                    "field": field,
                    "ordinal": int(index),
                    "readout_evidence_hash": f"{field}:readout:{index}",
                    "rollout_evidence_hash": f"{field}:rollout:{index}",
                    "emission_review_hash": f"{field}:review:{index}",
                    "prediction_hash": f"{field}:prediction:{index}",
                    "persistent_transition_weights_hash": f"{field}:weights:{index}",
                    "dense_label_candidate_evidence_hash": f"{index + 1:064x}",
                    "dense_label_candidate_evidence_id": (
                        f"dense-label-candidate:{index}"
                    ),
                    "review_hash": f"{index + 1001:064x}",
                    "source_execution_hash": f"{index + 2001:064x}",
                    "label_hash": f"{(index % 8) + 3001:064x}",
                    "labels": [f"{field}:label:{index}"],
                    "label_grounding": [True],
                    "tensor_device": "cpu",
                    "active_count": (index % 16) + 1,
                    "applied_calibration_update_hash": f"{index + 5001:064x}",
                    "applied_at": "2026-06-18T00:00:00+00:00",
                    "runtime_update_applied": True,
                    "weights_persisted": False,
                    "autonomous_confidence_use_event_hash": f"{index + 6001:064x}",
                    "used_at": "2026-06-18T00:00:00+00:00",
                    "output_is_label_hash_only": True,
                    "selected_candidate_count": 1,
                    "selected_candidate_refs": [
                        {
                            "dense_label_candidate_evidence_hash": f"{index + 1:064x}",
                            "label_hash": f"{(index % 8) + 3001:064x}",
                            "calibrated_confidence": 0.75,
                        }
                    ],
                }
            )
        state[field] = rows
    state.update(
        {
            "total_recorded_count": count,
            "total_rollout_recorded_count": count,
            "total_emission_review_count": count,
            "total_dense_label_candidate_count": count,
            "total_dense_label_calibration_update_count": count,
            "total_autonomous_confidence_use_count": count,
            "last_recorded_at": "2026-06-18T00:00:00+00:00",
            "last_rollout_recorded_at": "2026-06-18T00:00:00+00:00",
            "last_emission_reviewed_at": "2026-06-18T00:00:00+00:00",
            "last_dense_label_candidate_recorded_at": "2026-06-18T00:00:00+00:00",
            "last_dense_label_calibration_update_applied_at": (
                "2026-06-18T00:00:00+00:00"
            ),
            "last_autonomous_confidence_used_at": (
                "2026-06-18T00:00:00+00:00"
            ),
        }
    )
    state["current_dense_label_calibration_update"] = deepcopy(
        state["dense_label_calibration_update_events"][0]
    )
    return state


def _legacy_full_materialized_normalized_state(
    state: Mapping[str, Any],
    *,
    limit: int,
) -> dict[str, deque[dict[str, Any]]]:
    normalized: dict[str, deque[dict[str, Any]]] = {}
    for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS:
        raw_rows = list(state.get(field) or [])
        normalized[field] = deque(
            (
                deepcopy(dict(item))
                for item in raw_rows
                if isinstance(item, Mapping)
            ),
            maxlen=max(1, int(limit)),
        )
    return normalized


def _legacy_full_materialized_store_state(
    normalized: Mapping[str, Any],
    *,
    limit: int,
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS:
        raw_rows = list(normalized.get(field) or [])
        state[field] = [
            deepcopy(dict(item))
            for item in raw_rows[: max(1, int(limit))]
            if isinstance(item, Mapping)
        ]
    return state


def _bounded_store_state(
    ledger: SNNLanguageReadoutEvidenceLedger,
    target: dict[str, Any],
    normalized: Mapping[str, Any],
) -> dict[str, Any]:
    target.clear()
    ledger._store_state(normalized)  # noqa: SLF001
    return {
        field: list(target.get(field) or [])
        for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }


def _bounded_known_hash_lookup(
    ledger: SNNLanguageReadoutEvidenceLedger,
) -> dict[str, Any]:
    hashes, report = ledger._known_readout_evidence_hashes_with_report()  # noqa: SLF001
    return {
        "hashes": sorted(hashes),
        "report": report,
    }


def _broad_normalized_known_hash_lookup(
    ledger: SNNLanguageReadoutEvidenceLedger,
) -> dict[str, Any]:
    normalized = ledger._normalized_state()  # noqa: SLF001
    hashes = {
        str(item.get("readout_evidence_hash") or "")
        for item in list(normalized.get("events") or [])
        if isinstance(item, Mapping) and item.get("readout_evidence_hash")
    }
    return {
        "hashes": sorted(hashes),
        "normalization_source_window": dict(
            normalized.get("_normalization_source_window") or {}
        ),
    }


def _dense_label_policy_hashes(
    events: list[dict[str, Any]],
    *,
    limit: int,
) -> list[str]:
    selected = list(events)[: max(0, int(limit))]
    label_set_counts: dict[str, int] = {}
    execution_counts: dict[str, int] = {}
    for event in events:
        label_hash = str(event.get("label_hash") or "")
        execution_hash = str(event.get("source_execution_hash") or "")
        if label_hash:
            label_set_counts[label_hash] = label_set_counts.get(label_hash, 0) + 1
        if execution_hash:
            execution_counts[execution_hash] = execution_counts.get(execution_hash, 0) + 1
    scored: list[tuple[float, str]] = []
    for index, event in enumerate(selected):
        label_hash = str(event.get("label_hash") or "")
        active_count = int(event.get("active_count", 0) or 0)
        evidence_hash = str(event.get("dense_label_candidate_evidence_hash") or "")
        recency = (
            1.0 - min(1.0, index / max(1, len(selected) - 1))
            if len(selected) > 1
            else 1.0
        )
        repetition = (
            min(1.0, label_set_counts.get(label_hash, 0) / 3.0)
            if label_hash
            else 0.0
        )
        activity = min(1.0, active_count / 16.0)
        score = 100.0 * (0.40 * repetition + 0.30 * activity + 0.30 * recency)
        scored.append((score, evidence_hash))
    scored.sort(key=lambda item: (-float(item[0]), str(item[1])))
    return [evidence_hash for _score, evidence_hash in scored]


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
        "history_source_window": dict(
            dict(history.get("summary") or {}).get("source_window") or {}
        ),
        "policy_source_window": dict(policy.get("source_window") or {}),
    }


def _broad_normalized_dense_label_calibration(
    ledger: SNNLanguageReadoutEvidenceLedger,
    *,
    limit: int,
) -> dict[str, Any]:
    normalized = ledger._normalized_state()  # noqa: SLF001
    source_window = dict(normalized.get("_normalization_source_window") or {})
    events = [
        dict(item)
        for item in list(normalized.get("dense_label_candidate_events") or [])
        if isinstance(item, Mapping)
    ]
    count = max(0, min(int(limit), len(events)))
    return {
        "history_hashes": [
            str(item.get("dense_label_candidate_evidence_hash") or "")
            for item in events[:count]
        ],
        "policy_hashes": _dense_label_policy_hashes(events, limit=count),
        "ready_candidate_count": int(count),
        "normalization_source_window": source_window,
    }


def _dense_label_evaluation_preflight(*, limit: int) -> dict[str, Any]:
    selected_count = max(1, min(int(limit), 8))
    selected_hashes = [f"{index + 1:064x}" for index in range(selected_count)]
    return {
        "surface": "snn_language_dense_label_candidate_calibration_evaluation_preflight.v1",
        "ready": True,
        "preflight_hash": "benchmark-dense-label-calibration-evaluation",
        "selected_candidate_hashes": selected_hashes,
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


def _dense_label_evaluation_metrics(
    events: list[dict[str, Any]],
    heldout_labels: set[str],
    *,
    bin_count: int,
) -> dict[str, Any]:
    bins = max(2, min(int(bin_count or 5), 16))
    samples: list[dict[str, Any]] = []
    for event in events:
        labels = [
            str(label).strip()
            for label in list(event.get("labels") or [])
            if str(label).strip()
        ][:8]
        active_count = int(event.get("active_count", 0) or 0)
        confidence = min(1.0, max(0.0, active_count / 16.0))
        matches = sum(1 for label in labels if label in heldout_labels)
        accuracy = matches / max(1, len(labels))
        bin_index = min(bins - 1, int(confidence * bins))
        samples.append(
            {
                "dense_label_candidate_evidence_hash": event.get(
                    "dense_label_candidate_evidence_hash"
                ),
                "label_hash": event.get("label_hash"),
                "labels": labels,
                "label_count": len(labels),
                "heldout_match_count": matches,
                "confidence": round(confidence, 6),
                "accuracy": round(accuracy, 6),
                "calibration_gap": round(abs(confidence - accuracy), 6),
                "bin_index": bin_index,
                "tensor_device": event.get("tensor_device"),
                "active_count": active_count,
            }
        )
    total = max(1, len(samples))
    ece = 0.0
    for bin_index in range(bins):
        row_samples = [
            sample
            for sample in samples
            if int(sample.get("bin_index", -1)) == bin_index
        ]
        if row_samples:
            avg_confidence = sum(
                float(sample.get("confidence", 0.0) or 0.0)
                for sample in row_samples
            ) / len(row_samples)
            avg_accuracy = sum(
                float(sample.get("accuracy", 0.0) or 0.0)
                for sample in row_samples
            ) / len(row_samples)
        else:
            avg_confidence = 0.0
            avg_accuracy = 0.0
        ece += (len(row_samples) / total) * abs(avg_confidence - avg_accuracy)
    coverage = len(
        {
            label
            for sample in samples
            for label in list(sample.get("labels") or [])
            if str(label) in heldout_labels
        }
    ) / max(1, len(heldout_labels))
    label_hashes = {
        str(sample.get("label_hash") or "")
        for sample in samples
        if str(sample.get("label_hash") or "")
    }
    stability = 1.0 if len(label_hashes) <= 1 and samples else 1.0 / max(1, len(label_hashes))
    return {
        "sample_hashes": [
            str(sample.get("dense_label_candidate_evidence_hash") or "")
            for sample in samples
        ],
        "metrics": {
            "expected_calibration_error": round(ece, 6),
            "coverage_gap": round(1.0 - coverage, 6),
            "label_set_stability": round(stability, 6),
            "bin_count": bins,
        },
    }


def _bounded_dense_label_evaluation(
    ledger: SNNLanguageReadoutEvidenceLedger,
    *,
    limit: int,
) -> dict[str, Any]:
    preflight = _dense_label_evaluation_preflight(limit=limit)
    heldout = _heldout_label_evidence(limit=limit)
    evaluation = ledger.dense_label_candidate_calibration_evaluation(
        dense_label_candidate_calibration_evaluation_preflight=preflight,
        heldout_label_evidence=heldout,
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


def _broad_normalized_dense_label_evaluation(
    ledger: SNNLanguageReadoutEvidenceLedger,
    *,
    limit: int,
) -> dict[str, Any]:
    preflight = _dense_label_evaluation_preflight(limit=limit)
    heldout = _heldout_label_evidence(limit=limit)
    heldout_labels = {str(label) for label in list(heldout.get("labels") or [])}
    candidate_hashes = {
        str(value)
        for value in list(preflight.get("selected_candidate_hashes") or [])
        if str(value)
    }
    normalized = ledger._normalized_state()  # noqa: SLF001
    source_window = dict(normalized.get("_normalization_source_window") or {})
    events = [
        dict(event)
        for event in list(normalized.get("dense_label_candidate_events") or [])
        if str(event.get("dense_label_candidate_evidence_hash") or "")
        in candidate_hashes
    ][: max(1, min(int(preflight.get("selected_candidate_count", 1) or 1), 8))]
    metrics = _dense_label_evaluation_metrics(
        events,
        heldout_labels,
        bin_count=5,
    )
    return {
        **metrics,
        "normalization_source_window": source_window,
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


def _broad_normalized_dense_label_calibration_update_lookup(
    ledger: SNNLanguageReadoutEvidenceLedger,
) -> dict[str, Any]:
    normalized = ledger._normalized_state()  # noqa: SLF001
    source_window = dict(normalized.get("_normalization_source_window") or {})
    current = (
        normalized.get("current_dense_label_calibration_update")
        if isinstance(normalized.get("current_dense_label_calibration_update"), Mapping)
        else {}
    )
    events = [
        dict(item)
        for item in list(normalized.get("dense_label_calibration_update_events") or [])
        if isinstance(item, Mapping)
    ]
    return {
        "event_hashes": [
            str(item.get("applied_calibration_update_hash") or "")
            for item in events
        ],
        "current_hash": str(current.get("applied_calibration_update_hash") or ""),
        "normalization_source_window": source_window,
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


def _broad_normalized_autonomous_confidence_use_lookup(
    ledger: SNNLanguageReadoutEvidenceLedger,
) -> dict[str, Any]:
    normalized = ledger._normalized_state()  # noqa: SLF001
    source_window = dict(normalized.get("_normalization_source_window") or {})
    events = [
        dict(item)
        for item in list(normalized.get("autonomous_confidence_use_events") or [])
        if isinstance(item, Mapping)
    ]
    return {
        "event_hashes": [
            str(item.get("autonomous_confidence_use_event_hash") or "")
            for item in events
        ],
        "normalization_source_window": source_window,
    }


def _timed_runs(*, runs: int, fn: Any) -> tuple[dict[str, Any], list[float]]:
    samples: list[float] = []
    last: dict[str, Any] | None = None
    for _ in range(max(1, int(runs))):
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


def _first_ordinals(normalized: Mapping[str, deque[dict[str, Any]]]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS:
        rows = list(normalized.get(field) or [])
        result[field] = int(rows[0].get("ordinal", -1)) if rows else None
    return result


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
        fn=lambda: ledger._normalized_state(),  # noqa: SLF001
    )
    legacy, legacy_samples = _timed_runs(
        runs=runs,
        fn=lambda: _legacy_full_materialized_normalized_state(
            state,
            limit=ledger_limit,
        ),
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
    legacy_store, legacy_store_samples = _timed_runs(
        runs=runs,
        fn=lambda: _legacy_full_materialized_store_state(
            state,
            limit=ledger_limit,
        ),
    )
    known_hash, known_hash_samples = _timed_runs(
        runs=runs,
        fn=lambda: _bounded_known_hash_lookup(ledger),
    )
    broad_known_hash, broad_known_hash_samples = _timed_runs(
        runs=runs,
        fn=lambda: _broad_normalized_known_hash_lookup(ledger),
    )
    dense_label, dense_label_samples = _timed_runs(
        runs=runs,
        fn=lambda: _bounded_dense_label_calibration(
            ledger,
            limit=ledger_limit,
        ),
    )
    broad_dense_label, broad_dense_label_samples = _timed_runs(
        runs=runs,
        fn=lambda: _broad_normalized_dense_label_calibration(
            ledger,
            limit=ledger_limit,
        ),
    )
    dense_label_evaluation, dense_label_evaluation_samples = _timed_runs(
        runs=runs,
        fn=lambda: _bounded_dense_label_evaluation(
            ledger,
            limit=ledger_limit,
        ),
    )
    broad_dense_label_evaluation, broad_dense_label_evaluation_samples = _timed_runs(
        runs=runs,
        fn=lambda: _broad_normalized_dense_label_evaluation(
            ledger,
            limit=ledger_limit,
        ),
    )
    dense_label_update, dense_label_update_samples = _timed_runs(
        runs=runs,
        fn=lambda: _bounded_dense_label_calibration_update_lookup(ledger),
    )
    broad_dense_label_update, broad_dense_label_update_samples = _timed_runs(
        runs=runs,
        fn=lambda: _broad_normalized_dense_label_calibration_update_lookup(ledger),
    )
    autonomous_confidence_use, autonomous_confidence_use_samples = _timed_runs(
        runs=runs,
        fn=lambda: _bounded_autonomous_confidence_use_lookup(ledger),
    )
    broad_autonomous_confidence_use, broad_autonomous_confidence_use_samples = (
        _timed_runs(
            runs=runs,
            fn=lambda: _broad_normalized_autonomous_confidence_use_lookup(ledger),
        )
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    source_window = dict(bounded.get("_normalization_source_window") or {})
    bounded_rate = _recent_retention_rate(bounded)
    legacy_rate = _recent_retention_rate(legacy)
    bounded_store_rate = _recent_retention_rate(bounded_store)
    legacy_store_rate = _recent_retention_rate(legacy_store)
    bounded_rows = int(source_window.get("source_window_count_total", 0) or 0)
    legacy_rows = int(retention_count * len(SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS))
    bounded_store_rows = int(ledger_limit * len(SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS))
    legacy_store_rows = legacy_rows
    bounded_mean = statistics.fmean(bounded_samples)
    legacy_mean = statistics.fmean(legacy_samples)
    bounded_store_mean = statistics.fmean(bounded_store_samples)
    legacy_store_mean = statistics.fmean(legacy_store_samples)
    bounded_store_first_ordinals = _first_ordinals(bounded_store)
    legacy_store_first_ordinals = _first_ordinals(legacy_store)
    known_hash_report = dict(known_hash.get("report") or {})
    known_hash_mean = statistics.fmean(known_hash_samples)
    broad_known_hash_mean = statistics.fmean(broad_known_hash_samples)
    known_hash_rows = int(known_hash_report.get("source_window_count", 0) or 0)
    broad_known_hash_rows = int(
        dict(
            broad_known_hash.get("normalization_source_window") or {}
        ).get("source_window_count_total", 0)
        or 0
    )
    dense_label_source_window = dict(dense_label.get("policy_source_window") or {})
    dense_label_mean = statistics.fmean(dense_label_samples)
    broad_dense_label_mean = statistics.fmean(broad_dense_label_samples)
    dense_label_rows = int(dense_label_source_window.get("source_window_count", 0) or 0)
    broad_dense_label_rows = int(
        dict(
            broad_dense_label.get("normalization_source_window") or {}
        ).get("source_window_count_total", 0)
        or 0
    )
    dense_label_evaluation_source_window = dict(
        dense_label_evaluation.get("source_window") or {}
    )
    dense_label_evaluation_mean = statistics.fmean(dense_label_evaluation_samples)
    broad_dense_label_evaluation_mean = statistics.fmean(
        broad_dense_label_evaluation_samples
    )
    dense_label_evaluation_rows = int(
        dense_label_evaluation_source_window.get("source_window_count", 0) or 0
    )
    broad_dense_label_evaluation_rows = int(
        dict(
            broad_dense_label_evaluation.get("normalization_source_window") or {}
        ).get("source_window_count_total", 0)
        or 0
    )
    dense_label_update_source_window = dict(
        dense_label_update.get("source_window") or {}
    )
    dense_label_update_mean = statistics.fmean(dense_label_update_samples)
    broad_dense_label_update_mean = statistics.fmean(
        broad_dense_label_update_samples
    )
    dense_label_update_rows = int(
        dense_label_update_source_window.get("source_window_count", 0) or 0
    )
    broad_dense_label_update_rows = int(
        dict(
            broad_dense_label_update.get("normalization_source_window") or {}
        ).get("source_window_count_total", 0)
        or 0
    )
    autonomous_confidence_use_source_window = dict(
        autonomous_confidence_use.get("source_window") or {}
    )
    autonomous_confidence_use_mean = statistics.fmean(
        autonomous_confidence_use_samples
    )
    broad_autonomous_confidence_use_mean = statistics.fmean(
        broad_autonomous_confidence_use_samples
    )
    autonomous_confidence_use_rows = int(
        autonomous_confidence_use_source_window.get("source_window_count", 0) or 0
    )
    broad_autonomous_confidence_use_rows = int(
        dict(
            broad_autonomous_confidence_use.get("normalization_source_window") or {}
        ).get("source_window_count_total", 0)
        or 0
    )
    pass_checks = {
        "surface_present": source_window.get("surface")
        == "bounded_snn_readout_ledger_normalization_source_window.v1",
        "policy_present": source_window.get("policy")
        == SNN_LANGUAGE_READOUT_LEDGER_NORMALIZATION_SOURCE_WINDOW_POLICY,
        "source_limit_respected": all(
            int(value) <= ledger_limit
            for value in dict(source_window.get("source_window_counts") or {}).values()
        ),
        "no_global_scan": source_window.get("global_candidate_scan") is False
        and source_window.get("global_score_scan") is False,
        "not_live_tick": source_window.get("runs_live_tick") is False
        and source_window.get("runs_every_token") is False,
        "cpu_only": source_window.get("archival_storage_device") == "cpu"
        and source_window.get("normalization_device") == "cpu"
        and source_window.get("gpu_used") is False,
        "no_language_reasoning": source_window.get("language_reasoning") is False,
        "recent_rows_preserved": bounded_rate == 1.0,
        "legacy_recent_loss_detected": legacy_rate < bounded_rate,
        "bounded_less_work": bounded_rows < legacy_rows,
        "store_source_limit_respected": all(
            len(list(value or [])) <= ledger_limit for value in bounded_store.values()
        ),
        "store_recent_rows_preserved": bounded_store_rate == 1.0,
        "store_matches_legacy_window": (
            bounded_store_first_ordinals == legacy_store_first_ordinals
        ),
        "store_bounded_less_work": bounded_store_rows < legacy_store_rows,
        "store_latency_not_slower_than_legacy": (
            bounded_store_mean <= legacy_store_mean * 1.1
        ),
        "known_hash_surface_present": known_hash_report.get("surface")
        == "bounded_snn_readout_known_evidence_hash_source_window.v1",
        "known_hash_policy_present": known_hash_report.get("policy")
        == SNN_READOUT_EVIDENCE_HASH_SOURCE_WINDOW_POLICY,
        "known_hash_set_parity": known_hash.get("hashes")
        == broad_known_hash.get("hashes"),
        "known_hash_bounded_less_work": known_hash_rows < broad_known_hash_rows,
        "known_hash_latency_not_slower_than_broad_normalization": (
            known_hash_mean <= broad_known_hash_mean * 1.1
        ),
        "dense_label_surface_present": dense_label_source_window.get("surface")
        == "bounded_snn_dense_label_candidate_calibration_source_window.v1",
        "dense_label_policy_present": dense_label_source_window.get("policy")
        == SNN_DENSE_LABEL_CANDIDATE_CALIBRATION_SOURCE_WINDOW_POLICY,
        "dense_label_history_parity": dense_label.get("history_hashes")
        == broad_dense_label.get("history_hashes"),
        "dense_label_policy_parity": dense_label.get("policy_hashes")
        == broad_dense_label.get("policy_hashes"),
        "dense_label_ready_count_parity": dense_label.get("ready_candidate_count")
        == broad_dense_label.get("ready_candidate_count"),
        "dense_label_bounded_less_work": dense_label_rows < broad_dense_label_rows,
        "dense_label_latency_not_slower_than_broad_normalization": (
            dense_label_mean <= broad_dense_label_mean * 1.1
        ),
        "dense_label_evaluation_surface_present": (
            dense_label_evaluation_source_window.get("surface")
            == (
                "bounded_snn_dense_label_candidate_calibration_evaluation_source_window.v1"
            )
        ),
        "dense_label_evaluation_policy_present": (
            dense_label_evaluation_source_window.get("policy")
            == SNN_DENSE_LABEL_CALIBRATION_EVALUATION_SOURCE_WINDOW_POLICY
        ),
        "dense_label_evaluation_sample_hash_parity": (
            dense_label_evaluation.get("sample_hashes")
            == broad_dense_label_evaluation.get("sample_hashes")
        ),
        "dense_label_evaluation_metric_parity": (
            dense_label_evaluation.get("metrics")
            == broad_dense_label_evaluation.get("metrics")
        ),
        "dense_label_evaluation_bounded_less_work": (
            dense_label_evaluation_rows < broad_dense_label_evaluation_rows
        ),
        "dense_label_evaluation_latency_not_slower_than_broad_normalization": (
            dense_label_evaluation_mean <= broad_dense_label_evaluation_mean * 1.1
        ),
        "dense_label_update_surface_present": (
            dense_label_update_source_window.get("surface")
            == "bounded_snn_dense_label_calibration_update_source_window.v1"
        ),
        "dense_label_update_policy_present": (
            dense_label_update_source_window.get("policy")
            == SNN_DENSE_LABEL_CALIBRATION_UPDATE_SOURCE_WINDOW_POLICY
        ),
        "dense_label_update_event_hash_parity": (
            dense_label_update.get("event_hashes")
            == broad_dense_label_update.get("event_hashes")
        ),
        "dense_label_update_current_hash_parity": (
            dense_label_update.get("current_hash")
            == broad_dense_label_update.get("current_hash")
        ),
        "dense_label_update_bounded_less_work": (
            dense_label_update_rows < broad_dense_label_update_rows
        ),
        "dense_label_update_latency_not_slower_than_broad_normalization": (
            dense_label_update_mean <= broad_dense_label_update_mean * 1.1
        ),
        "autonomous_confidence_use_surface_present": (
            autonomous_confidence_use_source_window.get("surface")
            == "bounded_snn_autonomous_confidence_use_source_window.v1"
        ),
        "autonomous_confidence_use_policy_present": (
            autonomous_confidence_use_source_window.get("policy")
            == SNN_AUTONOMOUS_CONFIDENCE_USE_SOURCE_WINDOW_POLICY
        ),
        "autonomous_confidence_use_event_hash_parity": (
            autonomous_confidence_use.get("event_hashes")
            == broad_autonomous_confidence_use.get("event_hashes")
        ),
        "autonomous_confidence_use_bounded_less_work": (
            autonomous_confidence_use_rows < broad_autonomous_confidence_use_rows
        ),
        "autonomous_confidence_use_latency_not_slower_than_broad_normalization": (
            autonomous_confidence_use_mean
            <= broad_autonomous_confidence_use_mean * 1.1
        ),
    }
    return {
        "surface": "bounded_snn_readout_ledger_normalization_source_window_benchmark.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": {
            "retention_count_per_field": retention_count,
            "ledger_limit": ledger_limit,
            "event_field_count": len(SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS),
            "runs": runs,
        },
        "pass": all(pass_checks.values()),
        "pass_checks": pass_checks,
        "quality": {
            "metric": "newest_first_recent_row_retention_rate",
            "bounded_recent_retention_rate": round(bounded_rate, 6),
            "legacy_recent_retention_rate": round(legacy_rate, 6),
            "bounded_first_ordinals": _first_ordinals(bounded),
            "legacy_first_ordinals": _first_ordinals(legacy),
        },
        "latency": {
            "bounded": _latency_summary(bounded_samples),
            "legacy": _latency_summary(legacy_samples),
            "bounded_speedup_vs_legacy": round(
                legacy_mean / max(bounded_mean, 1e-9),
                6,
            ),
        },
        "retired_path_comparison": {
            "old_policy": "full_materialize_each_ledger_event_field_then_cap",
            "bounded_checked_record_count": bounded_rows,
            "old_checked_record_count": legacy_rows,
            "record_work_reduction": round(legacy_rows / max(1, bounded_rows), 6),
        },
        "normalization_source_window": source_window,
        "store_state_boundary": {
            "surface": "bounded_snn_readout_ledger_store_state_source_window.v1",
            "policy": SNN_LANGUAGE_READOUT_LEDGER_NORMALIZATION_SOURCE_WINDOW_POLICY,
            "source": "checkpoint_reload_and_record_persistence_event_fields",
            "selection_criteria": [
                "newest-first ledger event field order",
                "bounded source window before persistence copy",
                "single event-field helper shared with normalization",
            ],
            "quality": {
                "metric": "newest_first_store_window_parity",
                "bounded_recent_retention_rate": round(bounded_store_rate, 6),
                "legacy_recent_retention_rate": round(legacy_store_rate, 6),
                "bounded_first_ordinals": bounded_store_first_ordinals,
                "legacy_first_ordinals": legacy_store_first_ordinals,
            },
            "latency": {
                "bounded": _latency_summary(bounded_store_samples),
                "legacy": _latency_summary(legacy_store_samples),
                "bounded_speedup_vs_legacy": round(
                    legacy_store_mean / max(bounded_store_mean, 1e-9),
                    6,
                ),
            },
            "retired_path_comparison": {
                "old_policy": "full_materialize_each_ledger_event_field_then_cap_before_store",
                "bounded_checked_record_count": bounded_store_rows,
                "old_checked_record_count": legacy_store_rows,
                "record_work_reduction": round(
                    legacy_store_rows / max(1, bounded_store_rows),
                    6,
                ),
            },
            "source_window_limit_per_field": int(ledger_limit),
            "source_window_count_total": int(bounded_store_rows),
            "source_record_count_total_known": int(legacy_store_rows),
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
            "memory_budget": {
                "max_fields": len(SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS),
                "max_records_per_field": int(ledger_limit),
                "max_records_total": int(bounded_store_rows),
            },
        },
        "known_evidence_hash_boundary": {
            "surface": "bounded_snn_readout_known_evidence_hash_source_window.v1",
            "policy": SNN_READOUT_EVIDENCE_HASH_SOURCE_WINDOW_POLICY,
            "source": "snn_readout_ledger.events",
            "selection_criteria": [
                "internal_readout_evidence_events_only",
                "bounded source window before replay provenance lookup",
            ],
            "quality": {
                "metric": "known_readout_evidence_hash_set_parity",
                "hash_set_parity": known_hash.get("hashes")
                == broad_known_hash.get("hashes"),
                "hash_count": int(len(known_hash.get("hashes") or [])),
            },
            "latency": {
                "bounded": _latency_summary(known_hash_samples),
                "broad_normalized": _latency_summary(broad_known_hash_samples),
                "bounded_speedup_vs_broad_normalized": round(
                    broad_known_hash_mean / max(known_hash_mean, 1e-9),
                    6,
                ),
            },
            "retired_path_comparison": {
                "old_policy": "normalize_all_ledger_event_fields_before_known_hash_lookup",
                "bounded_checked_record_count": known_hash_rows,
                "old_checked_record_count": broad_known_hash_rows,
                "record_work_reduction": round(
                    broad_known_hash_rows / max(1, known_hash_rows),
                    6,
                ),
            },
            "source_window": known_hash_report,
        },
        "dense_label_calibration_boundary": {
            "surface": "bounded_snn_dense_label_candidate_calibration_source_window.v1",
            "policy": SNN_DENSE_LABEL_CANDIDATE_CALIBRATION_SOURCE_WINDOW_POLICY,
            "source": "snn_readout_ledger.dense_label_candidate_events",
            "selection_criteria": [
                "operator_reviewed_dense_label_candidates_only",
                "bounded source window before history or calibration policy",
            ],
            "quality": {
                "metric": "dense_label_history_and_policy_hash_parity",
                "history_hash_parity": dense_label.get("history_hashes")
                == broad_dense_label.get("history_hashes"),
                "policy_hash_parity": dense_label.get("policy_hashes")
                == broad_dense_label.get("policy_hashes"),
                "ready_candidate_count_parity": dense_label.get(
                    "ready_candidate_count"
                )
                == broad_dense_label.get("ready_candidate_count"),
                "candidate_count": int(len(dense_label.get("policy_hashes") or [])),
            },
            "latency": {
                "bounded": _latency_summary(dense_label_samples),
                "broad_normalized": _latency_summary(broad_dense_label_samples),
                "bounded_speedup_vs_broad_normalized": round(
                    broad_dense_label_mean / max(dense_label_mean, 1e-9),
                    6,
                ),
            },
            "retired_path_comparison": {
                "old_policy": (
                    "normalize_all_ledger_event_fields_before_dense_label_history"
                    "_or_calibration_policy"
                ),
                "bounded_checked_record_count": dense_label_rows,
                "old_checked_record_count": broad_dense_label_rows,
                "record_work_reduction": round(
                    broad_dense_label_rows / max(1, dense_label_rows),
                    6,
                ),
            },
            "source_window": dense_label_source_window,
        },
        "dense_label_evaluation_boundary": {
            "surface": (
                "bounded_snn_dense_label_candidate_calibration_evaluation_source_window.v1"
            ),
            "policy": SNN_DENSE_LABEL_CALIBRATION_EVALUATION_SOURCE_WINDOW_POLICY,
            "source": "snn_readout_ledger.dense_label_candidate_events",
            "selection_criteria": [
                "preflight_selected_dense_label_candidate_hashes_only",
                "bounded source window before calibration evaluation",
            ],
            "quality": {
                "metric": "dense_label_calibration_evaluation_sample_and_metric_parity",
                "ready": bool(dense_label_evaluation.get("ready")),
                "sample_hash_parity": dense_label_evaluation.get("sample_hashes")
                == broad_dense_label_evaluation.get("sample_hashes"),
                "metric_parity": dense_label_evaluation.get("metrics")
                == broad_dense_label_evaluation.get("metrics"),
                "sample_count": int(
                    len(dense_label_evaluation.get("sample_hashes") or [])
                ),
            },
            "latency": {
                "bounded": _latency_summary(dense_label_evaluation_samples),
                "broad_normalized": _latency_summary(
                    broad_dense_label_evaluation_samples
                ),
                "bounded_speedup_vs_broad_normalized": round(
                    broad_dense_label_evaluation_mean
                    / max(dense_label_evaluation_mean, 1e-9),
                    6,
                ),
            },
            "retired_path_comparison": {
                "old_policy": (
                    "normalize_all_ledger_event_fields_before_preflight_selected"
                    "_dense_label_calibration_evaluation"
                ),
                "bounded_checked_record_count": dense_label_evaluation_rows,
                "old_checked_record_count": broad_dense_label_evaluation_rows,
                "record_work_reduction": round(
                    broad_dense_label_evaluation_rows
                    / max(1, dense_label_evaluation_rows),
                    6,
                ),
            },
            "source_window": dense_label_evaluation_source_window,
        },
        "dense_label_calibration_update_boundary": {
            "surface": "bounded_snn_dense_label_calibration_update_source_window.v1",
            "policy": SNN_DENSE_LABEL_CALIBRATION_UPDATE_SOURCE_WINDOW_POLICY,
            "source": "snn_readout_ledger.dense_label_calibration_update_events",
            "selection_criteria": [
                "applied_dense_label_calibration_updates_only",
                "bounded source window before update application or review",
            ],
            "quality": {
                "metric": "dense_label_calibration_update_event_and_current_hash_parity",
                "event_hash_parity": dense_label_update.get("event_hashes")
                == broad_dense_label_update.get("event_hashes"),
                "current_hash_parity": dense_label_update.get("current_hash")
                == broad_dense_label_update.get("current_hash"),
                "event_hash_count": int(
                    len(dense_label_update.get("event_hashes") or [])
                ),
                "current_hash": dense_label_update.get("current_hash"),
            },
            "latency": {
                "bounded": _latency_summary(dense_label_update_samples),
                "broad_normalized": _latency_summary(
                    broad_dense_label_update_samples
                ),
                "bounded_speedup_vs_broad_normalized": round(
                    broad_dense_label_update_mean
                    / max(dense_label_update_mean, 1e-9),
                    6,
                ),
            },
            "retired_path_comparison": {
                "old_policy": (
                    "normalize_all_ledger_event_fields_before_dense_label"
                    "_calibration_update_duplicate_or_current_hash_lookup"
                ),
                "bounded_checked_record_count": dense_label_update_rows,
                "old_checked_record_count": broad_dense_label_update_rows,
                "record_work_reduction": round(
                    broad_dense_label_update_rows
                    / max(1, dense_label_update_rows),
                    6,
                ),
            },
            "source_window": dense_label_update_source_window,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "runs_live_tick": False,
            "runs_every_token": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "lookup_device": "cpu",
            "write_device": "cpu",
            "gpu_used": False,
        },
        "autonomous_confidence_use_boundary": {
            "surface": "bounded_snn_autonomous_confidence_use_source_window.v1",
            "policy": SNN_AUTONOMOUS_CONFIDENCE_USE_SOURCE_WINDOW_POLICY,
            "source": "snn_readout_ledger.autonomous_confidence_use_events",
            "selection_criteria": [
                "recorded_autonomous_confidence_use_events_only",
                "bounded_source_window_before_hash_only_use_review",
            ],
            "quality": {
                "metric": "autonomous_confidence_use_event_hash_parity",
                "event_hash_parity": autonomous_confidence_use.get("event_hashes")
                == broad_autonomous_confidence_use.get("event_hashes"),
                "event_hash_count": int(
                    len(autonomous_confidence_use.get("event_hashes") or [])
                ),
            },
            "latency": {
                "bounded": _latency_summary(autonomous_confidence_use_samples),
                "broad_normalized": _latency_summary(
                    broad_autonomous_confidence_use_samples
                ),
                "bounded_speedup_vs_broad_normalized": round(
                    broad_autonomous_confidence_use_mean
                    / max(autonomous_confidence_use_mean, 1e-9),
                    6,
                ),
            },
            "retired_path_comparison": {
                "old_policy": (
                    "normalize_all_ledger_event_fields_before_autonomous"
                    "_confidence_use_duplicate_or_review_lookup"
                ),
                "bounded_checked_record_count": autonomous_confidence_use_rows,
                "old_checked_record_count": broad_autonomous_confidence_use_rows,
                "record_work_reduction": round(
                    broad_autonomous_confidence_use_rows
                    / max(1, autonomous_confidence_use_rows),
                    6,
                ),
            },
            "source_window": autonomous_confidence_use_source_window,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "runs_live_tick": False,
            "runs_every_token": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "lookup_device": "cpu",
            "write_device": "cpu",
            "gpu_used": False,
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
        default=Path("reports/bounded_replay_window_20260618/snn-readout-ledger-normalization-source-window.json"),
    )
    args = parser.parse_args()
    report = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"snn_readout_ledger_normalization_source_window_benchmark={args.output}")
    print(
        "pass={passed} bounded_mean_ms={bounded:.6f} legacy_mean_ms={legacy:.6f} "
        "work_reduction={work:.6f} bounded_recent={bounded_recent:.6f} "
        "legacy_recent={legacy_recent:.6f} store_bounded_mean_ms={store_bounded:.6f} "
        "store_legacy_mean_ms={store_legacy:.6f} known_hash_bounded_mean_ms={known_hash_bounded:.6f} "
        "known_hash_broad_mean_ms={known_hash_broad:.6f} dense_label_bounded_mean_ms={dense_label_bounded:.6f} "
        "dense_label_broad_mean_ms={dense_label_broad:.6f} dense_label_eval_bounded_mean_ms={dense_label_eval_bounded:.6f} "
        "dense_label_eval_broad_mean_ms={dense_label_eval_broad:.6f} "
        "dense_label_update_bounded_mean_ms={dense_label_update_bounded:.6f} "
        "dense_label_update_broad_mean_ms={dense_label_update_broad:.6f} "
        "autonomous_confidence_use_bounded_mean_ms={confidence_use_bounded:.6f} "
        "autonomous_confidence_use_broad_mean_ms={confidence_use_broad:.6f}".format(
            passed=report["pass"],
            bounded=report["latency"]["bounded"]["mean_ms"],
            legacy=report["latency"]["legacy"]["mean_ms"],
            work=report["retired_path_comparison"]["record_work_reduction"],
            bounded_recent=report["quality"]["bounded_recent_retention_rate"],
            legacy_recent=report["quality"]["legacy_recent_retention_rate"],
            store_bounded=report["store_state_boundary"]["latency"]["bounded"][
                "mean_ms"
            ],
            store_legacy=report["store_state_boundary"]["latency"]["legacy"][
                "mean_ms"
            ],
            known_hash_bounded=report["known_evidence_hash_boundary"]["latency"][
                "bounded"
            ]["mean_ms"],
            known_hash_broad=report["known_evidence_hash_boundary"]["latency"][
                "broad_normalized"
            ]["mean_ms"],
            dense_label_bounded=report["dense_label_calibration_boundary"][
                "latency"
            ]["bounded"]["mean_ms"],
            dense_label_broad=report["dense_label_calibration_boundary"][
                "latency"
            ]["broad_normalized"]["mean_ms"],
            dense_label_eval_bounded=report["dense_label_evaluation_boundary"][
                "latency"
            ]["bounded"]["mean_ms"],
            dense_label_eval_broad=report["dense_label_evaluation_boundary"][
                "latency"
            ]["broad_normalized"]["mean_ms"],
            dense_label_update_bounded=report[
                "dense_label_calibration_update_boundary"
            ]["latency"]["bounded"]["mean_ms"],
            dense_label_update_broad=report[
                "dense_label_calibration_update_boundary"
            ]["latency"]["broad_normalized"]["mean_ms"],
            confidence_use_bounded=report[
                "autonomous_confidence_use_boundary"
            ]["latency"]["bounded"]["mean_ms"],
            confidence_use_broad=report[
                "autonomous_confidence_use_boundary"
            ]["latency"]["broad_normalized"]["mean_ms"],
        )
    )


if __name__ == "__main__":
    main()
