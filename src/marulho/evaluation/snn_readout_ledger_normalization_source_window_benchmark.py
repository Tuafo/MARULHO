"""Benchmark benchmark-local SNN readout-ledger normalization models.

Production no longer exposes an all-family normalizer. The bounded model below
is evidence-only and the diagnostic legacy path preserves the retired
full-materialize-then-cap shape for latency, work, and recent-row retention
comparison only.
"""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from itertools import islice
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
    SNN_EMISSION_REVIEW_HISTORY_SOURCE_WINDOW_POLICY,
    SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS,
    SNN_LANGUAGE_READOUT_LEDGER_COUNT_FIELDS,
    SNN_LANGUAGE_READOUT_LEDGER_CURRENT_MAPPING_FIELDS,
    SNN_LANGUAGE_READOUT_LEDGER_TIMESTAMP_FIELDS,
    SNN_READOUT_LEDGER_RECORD_FAMILY_SOURCE_WINDOW_POLICY,
    SNN_READOUT_EVIDENCE_HASH_SOURCE_WINDOW_POLICY,
    SNNLanguageReadoutEvidenceLedger,
)

SNN_BENCHMARK_READOUT_LEDGER_NORMALIZATION_SOURCE_WINDOW_POLICY = (
    "recent_ledger_event_field_source_window_v1"
)

_AUTONOMOUS_CHAIN_COMPONENT_NAMES = (
    "binding",
    "observation",
    "training",
    "decoder",
    "language_output",
    "decoded_output",
    "text_emission",
    "text_commit",
    "materialization",
    "language_surface_commit",
    "language_surface_use",
    "language_generation",
    "language_decoding",
    "readout_surface",
    "readout_memory",
    "thought_consolidation",
    "thought_structural_plasticity",
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
                    "text": f"{field}:text:{index}",
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
            "total_autonomous_snn_language_thought_consolidation_count": count,
            "total_autonomous_snn_language_thought_structural_plasticity_count": count,
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
            "last_autonomous_hash_readout_bound_at": (
                "2026-06-18T00:00:00+00:00"
            ),
            "last_autonomous_bound_readout_observed_at": (
                "2026-06-18T00:00:00+00:00"
            ),
            "last_autonomous_readout_training_window_trained_at": (
                "2026-06-18T00:00:00+00:00"
            ),
            "last_autonomous_decoder_probed_at": (
                "2026-06-18T00:00:00+00:00"
            ),
            "last_autonomous_language_output_emitted_at": (
                "2026-06-18T00:00:00+00:00"
            ),
            "last_autonomous_decoded_output_at": (
                "2026-06-18T00:00:00+00:00"
            ),
            "last_autonomous_bounded_text_emitted_at": (
                "2026-06-18T00:00:00+00:00"
            ),
            "last_autonomous_text_surface_committed_at": (
                "2026-06-18T00:00:00+00:00"
            ),
            "last_autonomous_text_surface_materialized_at": (
                "2026-06-18T00:00:00+00:00"
            ),
            "last_autonomous_bounded_language_surface_committed_at": (
                "2026-06-18T00:00:00+00:00"
            ),
            "last_autonomous_bounded_language_surface_used_at": (
                "2026-06-18T00:00:00+00:00"
            ),
            "last_autonomous_snn_language_generated_at": (
                "2026-06-18T00:00:00+00:00"
            ),
            "last_autonomous_snn_language_decoded_at": (
                "2026-06-18T00:00:00+00:00"
            ),
            "last_snn_language_readout_surface_recorded_at": (
                "2026-06-18T00:00:00+00:00"
            ),
            "last_snn_language_readout_memory_recorded_at": (
                "2026-06-18T00:00:00+00:00"
            ),
            "last_autonomous_snn_language_thought_consolidated_at": (
                "2026-06-18T00:00:00+00:00"
            ),
            "last_autonomous_snn_language_thought_structural_plasticity_applied_at": (
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
    for field in SNN_LANGUAGE_READOUT_LEDGER_CURRENT_MAPPING_FIELDS:
        value = normalized.get(field)
        state[field] = deepcopy(dict(value)) if isinstance(value, Mapping) else {}
    for field in SNN_LANGUAGE_READOUT_LEDGER_COUNT_FIELDS:
        state[field] = int(normalized.get(field, 0) or 0)
    for field in SNN_LANGUAGE_READOUT_LEDGER_TIMESTAMP_FIELDS:
        state[field] = normalized.get(field)
    return state


def _benchmark_source_record_count(
    state: Mapping[str, Any],
    name: str,
) -> int | None:
    raw_value = state.get(name) or []
    if isinstance(raw_value, (str, bytes, Mapping)):
        return 0
    try:
        return int(len(raw_value))
    except TypeError:
        return None


def _benchmark_bounded_mapping_deque_from_state(
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
            for item in islice(raw_value, source_limit)
            if isinstance(item, Mapping)
        ),
        maxlen=source_limit,
    )


def _benchmark_normalization_source_window_report(
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
        name: _benchmark_source_record_count(state, name)
        for name in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }
    truncated_source_counts = {
        name: (
            max(0, int(source_count) - int(field_window_counts[name]))
            if source_count is not None
            else None
        )
        for name, source_count in source_record_counts.items()
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
        "truncated_source_counts": truncated_source_counts,
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


def _benchmark_bounded_normalized_state(
    ledger: SNNLanguageReadoutEvidenceLedger,
) -> dict[str, Any]:
    state = ledger._ledger_state()  # noqa: SLF001
    limit = max(1, int(getattr(ledger, "_limit", 1)))
    normalized: dict[str, Any] = {
        name: _benchmark_bounded_mapping_deque_from_state(
            state,
            name,
            limit=limit,
        )
        for name in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }
    normalized["_normalization_source_window"] = (
        _benchmark_normalization_source_window_report(
            state,
            normalized,
            limit=limit,
        )
    )
    for field in SNN_LANGUAGE_READOUT_LEDGER_CURRENT_MAPPING_FIELDS:
        value = state.get(field)
        normalized[field] = deepcopy(dict(value)) if isinstance(value, Mapping) else {}
    for field in SNN_LANGUAGE_READOUT_LEDGER_COUNT_FIELDS:
        normalized[field] = int(state.get(field, 0) or 0)
    for field in SNN_LANGUAGE_READOUT_LEDGER_TIMESTAMP_FIELDS:
        normalized[field] = state.get(field)
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
    return {
        "hashes": sorted(hashes),
        "report": report,
    }


def _broad_normalized_known_hash_lookup(
    ledger: SNNLanguageReadoutEvidenceLedger,
) -> dict[str, Any]:
    normalized = _benchmark_bounded_normalized_state(ledger)
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
    return {
        "hashes": sorted(event_map.keys()),
        "event_count": int(len(event_map)),
        "report": report,
    }


def _broad_normalized_readout_evidence_event_map_lookup(
    ledger: SNNLanguageReadoutEvidenceLedger,
) -> dict[str, Any]:
    target_hashes = _readout_evidence_event_map_target_hashes()
    normalized = _benchmark_bounded_normalized_state(ledger)
    event_map = {
        str(item.get("readout_evidence_hash") or ""): dict(item)
        for item in list(normalized.get("events") or [])
        if (
            isinstance(item, Mapping)
            and str(item.get("readout_evidence_hash") or "") in target_hashes
        )
    }
    return {
        "hashes": sorted(event_map.keys()),
        "event_count": int(len(event_map)),
        "normalization_source_window": dict(
            normalized.get("_normalization_source_window") or {}
        ),
    }


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
        "text_hashes": [
            str(item.get("text_hash") or "")
            for item in list(history.get("emission_review_events") or [])
        ],
        "summary": dict(history.get("summary") or {}),
        "source_window": dict(history.get("source_window") or {}),
    }


def _broad_normalized_emission_review_history(
    ledger: SNNLanguageReadoutEvidenceLedger,
    *,
    limit: int,
) -> dict[str, Any]:
    normalized = _benchmark_bounded_normalized_state(ledger)
    source_window = dict(normalized.get("_normalization_source_window") or {})
    normalized_events = list(normalized.get("emission_review_events") or [])
    count = max(0, min(int(limit), len(normalized_events)))
    events = [
        dict(item)
        for item in normalized_events[:count]
        if isinstance(item, Mapping)
    ]
    text_hashes = [
        ledger._sha256_json(  # noqa: SLF001
            {
                "text": str(item.get("text") or ""),
                "labels": [
                    str(value)
                    for value in list(item.get("labels") or [])
                    if str(value)
                ],
            }
        )
        for item in events
    ]
    return {
        "review_hashes": [
            str(item.get("emission_review_hash") or "")
            for item in events
        ],
        "text_hashes": text_hashes,
        "normalization_source_window": source_window,
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
    normalized = _benchmark_bounded_normalized_state(ledger)
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
    normalized = _benchmark_bounded_normalized_state(ledger)
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
    normalized = _benchmark_bounded_normalized_state(ledger)
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
    normalized = _benchmark_bounded_normalized_state(ledger)
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


def _record_append_event() -> dict[str, Any]:
    return {
        "readout_evidence_hash": "record-family-append-readout-hash",
        "recorded_at": "2026-06-19T00:00:00+00:00",
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


def _broad_normalized_record_family_append(
    ledger: SNNLanguageReadoutEvidenceLedger,
) -> dict[str, Any]:
    event = _record_append_event()
    normalized = _benchmark_bounded_normalized_state(ledger)
    source_window = dict(normalized.get("_normalization_source_window") or {})
    events = normalized["events"]
    existing_hashes = {
        str(item.get("readout_evidence_hash") or "") for item in list(events)
    }
    duplicate = event["readout_evidence_hash"] in existing_hashes
    if not duplicate:
        events.appendleft(deepcopy(event))
        normalized["total_recorded_count"] = int(
            normalized.get("total_recorded_count", 0) or 0
        ) + 1
        normalized["last_recorded_at"] = event["recorded_at"]
        ledger._store_state(normalized)  # noqa: SLF001
    stored_events = list(ledger._ledger_state().get("events") or [])  # noqa: SLF001
    latest = (
        dict(stored_events[0])
        if stored_events and isinstance(stored_events[0], Mapping)
        else {}
    )
    return {
        "duplicate": duplicate,
        "latest_hash": str(latest.get("readout_evidence_hash") or ""),
        "total_recorded_count": int(
            ledger._ledger_state().get("total_recorded_count", 0) or 0  # noqa: SLF001
        ),
        "event_count": int(len(stored_events)),
        "normalization_source_window": source_window,
    }


def _autonomous_binding_event() -> dict[str, Any]:
    return {
        "autonomous_hash_readout_binding_event_hash": "a" * 64,
        "autonomous_hash_readout_binding_event_id": "benchmark-binding",
        "bound_at": "2026-06-19T00:00:00+00:00",
        "state_revision": 0,
        "binding_count": 1,
        "bindings": [
            {
                "binding_index": 0,
                "dense_label_candidate_evidence_hash": "1" * 64,
                "readout_slot_hash": "2" * 64,
            }
        ],
        "output_is_hash_binding_only": True,
    }


def _autonomous_observation_event() -> dict[str, Any]:
    return {
        "autonomous_bound_readout_observation_event_hash": "b" * 64,
        "autonomous_bound_readout_observation_event_id": "benchmark-observation",
        "observed_at": "2026-06-19T00:00:00+00:00",
        "state_revision": 1,
        "binding_count": 1,
        "observation_cycles": 4,
        "sample_count": 4,
        "mean_activation_sparsity": 0.75,
        "max_slot_drift": 0.05,
        "mean_binding_reactivation": 0.8,
        "target_hashes": ["3" * 64],
        "sample_hashes": ["4" * 64, "5" * 64, "6" * 64, "7" * 64],
        "output_is_hash_observation_only": True,
    }


def _autonomous_training_window_event() -> dict[str, Any]:
    return {
        "autonomous_readout_training_window_event_hash": "c" * 64,
        "autonomous_readout_training_window_event_id": "benchmark-training-window",
        "trained_at": "2026-06-19T00:00:00+00:00",
        "state_revision": 2,
        "autonomous_bound_readout_observation_event_hash": "b" * 64,
        "target_hashes": ["3" * 64],
        "sample_hashes": ["4" * 64, "5" * 64, "6" * 64, "7" * 64],
        "training_window_steps": 4,
        "truncated_bptt_steps": 2,
        "micro_batch_size": 2,
        "learning_rule": "local_surrogate_gradient",
        "learning_rate": 0.0002,
        "loss_before": 0.42,
        "loss_after": 0.38,
        "mean_gradient_norm": 1.2,
        "max_weight_delta": 0.01,
        "observed_spike_sparsity": 0.74,
        "weight_update_hash": "8" * 64,
        "gradient_update_hash": "9" * 64,
        "optimizer_state_hash": "d" * 64,
        "device_trace_hash": "e" * 64,
        "runtime_weights_updated": True,
        "trains_runtime_model": True,
        "generates_text": False,
        "decodes_text": False,
        "writes_checkpoint": False,
        "runs_replay": False,
        "applies_plasticity": False,
    }


def _autonomous_decoder_probe_event() -> dict[str, Any]:
    return {
        "autonomous_decoder_probe_event_hash": "f" * 64,
        "autonomous_decoder_probe_event_id": "benchmark-decoder-probe",
        "probed_at": "2026-06-19T00:00:00+00:00",
        "state_revision": 3,
        "autonomous_readout_training_window_event_hash": "c" * 64,
        "probe_mode": "hash_rank_probe",
        "max_probe_steps": 4,
        "top_k": 1,
        "probe_result_count": 1,
        "probe_target_hashes": ["8" * 64],
        "output_hashes": ["9" * 64],
        "rank_hashes": ["a" * 64, "b" * 64],
        "mean_top_score": 0.82,
        "mean_spike_sparsity": 0.73,
        "max_slot_drift": 0.04,
        "probe_results": [
            {
                "probe_target_hash": "8" * 64,
                "output_hash": "9" * 64,
                "rank_hashes": ["a" * 64, "b" * 64],
                "top_score": 0.82,
                "spike_sparsity": 0.73,
                "slot_drift": 0.04,
            }
        ],
        "output_is_hash_probe_only": True,
        "generates_text": False,
        "decodes_text": False,
        "writes_checkpoint": False,
        "runs_replay": False,
        "applies_plasticity": False,
        "trains_runtime_model": False,
    }


def _autonomous_language_output_event() -> dict[str, Any]:
    return {
        "autonomous_language_output_event_hash": "a1" * 32,
        "autonomous_language_output_event_id": "benchmark-language-output",
        "emitted_at": "2026-06-19T00:00:00+00:00",
        "state_revision": 4,
        "preflight_hash": "a2" * 32,
        "language_output_design_hash": "a3" * 32,
        "review_hash": "a4" * 32,
        "autonomous_decoder_probe_event_hash": "f" * 64,
        "output_mode": "hash_candidate_slots",
        "max_output_tokens": 1,
        "output_slot_count": 1,
        "candidate_hashes": ["a5" * 32],
        "output_slot_hashes": ["a6" * 32],
        "emitted_hashes": ["a7" * 32],
        "rank_hashes": ["a8" * 32],
        "mean_confidence_score": 0.81,
        "mean_spike_sparsity": 0.74,
        "max_slot_drift": 0.03,
        "output_slot_results": [
            {
                "language_output_slot_hash": "a6" * 32,
                "candidate_hash": "a5" * 32,
                "emitted_hash": "a7" * 32,
                "rank_hashes": ["a8" * 32],
                "confidence_score": 0.81,
                "spike_sparsity": 0.74,
                "slot_drift": 0.03,
            }
        ],
        "source_output_slots": [
            {
                "language_output_slot_hash": "a6" * 32,
                "candidate_hash": "a5" * 32,
                "slot_index": 0,
            }
        ],
        "output_is_hash_only": True,
        "operator_approval_required": False,
        "generates_text": False,
        "decodes_text": False,
        "writes_checkpoint": False,
        "runs_replay": False,
        "applies_plasticity": False,
        "trains_runtime_model": False,
    }


def _autonomous_decoded_output_event() -> dict[str, Any]:
    return {
        "autonomous_decoded_output_event_hash": "d1" * 32,
        "autonomous_decoded_output_event_id": "benchmark-decoded-output",
        "decoded_at": "2026-06-19T00:00:00+00:00",
        "state_revision": 5,
        "preflight_hash": "d2" * 32,
        "decoded_output_design_hash": "d3" * 32,
        "review_hash": "d4" * 32,
        "autonomous_language_output_event_hash": "a1" * 32,
        "decode_mode": "constrained_token_hash_map",
        "max_decoded_tokens": 1,
        "decoded_token_count": 1,
        "decoded_output_slot_hashes": ["d5" * 32],
        "token_candidate_hashes": ["d6" * 32],
        "decoded_token_hashes": ["d7" * 32],
        "token_id_hashes": ["d8" * 32],
        "constraint_state_hashes": ["d9" * 32],
        "mean_confidence_score": 0.79,
        "mean_spike_sparsity": 0.72,
        "max_slot_drift": 0.02,
        "decoded_token_results": [
            {
                "decoded_output_slot_hash": "d5" * 32,
                "token_candidate_hash": "d6" * 32,
                "decoded_token_hash": "d7" * 32,
                "token_id_hash": "d8" * 32,
                "constraint_state_hash": "d9" * 32,
                "constraint_valid": True,
                "confidence_score": 0.79,
                "spike_sparsity": 0.72,
                "slot_drift": 0.02,
            }
        ],
        "output_is_hash_only": True,
        "operator_approval_required": False,
        "generates_text": False,
        "decodes_text": False,
        "writes_checkpoint": False,
        "runs_replay": False,
        "applies_plasticity": False,
        "trains_runtime_model": False,
    }


def _autonomous_bounded_text_emission_event() -> dict[str, Any]:
    return {
        "autonomous_bounded_text_emission_event_hash": "e1" * 32,
        "autonomous_bounded_text_emission_event_id": (
            "benchmark-bounded-text-emission"
        ),
        "emitted_at": "2026-06-19T00:00:00+00:00",
        "state_revision": 6,
        "preflight_hash": "e2" * 32,
        "bounded_text_emission_design_hash": "e3" * 32,
        "review_hash": "e4" * 32,
        "autonomous_decoded_output_event_hash": "d1" * 32,
        "emission_mode": "bounded_text_hash_sequence",
        "max_text_fragments": 1,
        "text_fragment_count": 1,
        "decoded_token_hashes": ["d7" * 32],
        "text_fragment_hashes": ["e5" * 32],
        "text_emission_slot_hashes": ["e6" * 32],
        "text_surface_schema_hash": "e7" * 32,
        "text_normalizer_hash": "e8" * 32,
        "semantic_constraint_hash": "e9" * 32,
        "mean_confidence_score": 0.79,
        "mean_spike_sparsity": 0.72,
        "max_slot_drift": 0.02,
        "text_emission_results": [
            {
                "bounded_text_emission_slot_hash": "e6" * 32,
                "decoded_token_hash": "d7" * 32,
                "text_fragment_hash": "e5" * 32,
                "text_surface_schema_hash": "e7" * 32,
                "text_normalizer_hash": "e8" * 32,
                "semantic_constraint_hash": "e9" * 32,
                "semantic_constraint_valid": True,
                "text_normalized": True,
                "confidence_score": 0.79,
                "spike_sparsity": 0.72,
                "slot_drift": 0.02,
            }
        ],
        "output_is_hash_only": True,
        "literal_text_returned": False,
        "operator_approval_required": False,
        "generates_text": False,
        "decodes_text": False,
        "writes_checkpoint": False,
        "runs_replay": False,
        "applies_plasticity": False,
        "trains_runtime_model": False,
    }


def _autonomous_text_surface_commit_event() -> dict[str, Any]:
    return {
        "autonomous_text_surface_commit_event_hash": "c1" * 32,
        "autonomous_text_surface_commit_event_id": "benchmark-text-surface-commit",
        "committed_at": "2026-06-19T00:00:00+00:00",
        "state_revision": 7,
        "preflight_hash": "c2" * 32,
        "text_surface_commit_design_hash": "c3" * 32,
        "commit_plan_hash": "c4" * 32,
        "sequence_review_hash": "c5" * 32,
        "autonomous_bounded_text_emission_event_hash": "e1" * 32,
        "commit_scope": "hash_surface_state",
        "retention_class": "ephemeral_hash_surface",
        "text_fragment_count": 1,
        "decoded_token_hashes": ["d7" * 32],
        "text_fragment_hashes": ["e5" * 32],
        "text_emission_slot_hashes": ["e6" * 32],
        "fragment_sequence_hash": "c6" * 32,
        "committed_surface_hash": "c6" * 32,
        "previous_surface_hash": None,
        "state_chain_hash": "c7" * 32,
        "text_surface_schema_hash": "e7" * 32,
        "text_normalizer_hash": "e8" * 32,
        "semantic_constraint_hash": "e9" * 32,
        "mean_confidence_score": 0.79,
        "mean_spike_sparsity": 0.72,
        "max_slot_drift": 0.02,
        "output_is_hash_only": True,
        "literal_text_returned": False,
        "operator_approval_required": False,
        "generates_text": False,
        "decodes_text": False,
        "writes_checkpoint": False,
        "runs_replay": False,
        "applies_plasticity": False,
        "trains_runtime_model": False,
    }


def _autonomous_text_surface_materialization_event() -> dict[str, Any]:
    return {
        "autonomous_text_surface_materialization_event_hash": "a1" * 32,
        "autonomous_text_surface_materialization_event_id": (
            "benchmark-text-surface-materialization"
        ),
        "materialized_at": "2026-06-19T00:00:00+00:00",
        "state_revision": 8,
        "preflight_hash": "a2" * 32,
        "materialization_design_hash": "a3" * 32,
        "materialization_plan_hash": "a4" * 32,
        "review_hash": "a5" * 32,
        "autonomous_text_surface_commit_event_hash": "c1" * 32,
        "commit_plan_hash": "c4" * 32,
        "sequence_review_hash": "c5" * 32,
        "autonomous_bounded_text_emission_event_hash": "e1" * 32,
        "fragment_sequence_hash": "c6" * 32,
        "committed_surface_hash": "c6" * 32,
        "state_chain_hash": "c7" * 32,
        "materialization_mode": "bounded_text_surface",
        "output_contract": "literal_bounded_fragments",
        "max_surface_chars": 256,
        "text_fragment_count": 1,
        "decoded_token_hashes": ["d7" * 32],
        "text_fragment_hashes": ["a7" * 32],
        "literal_fragment_hashes": ["a7" * 32],
        "text_emission_slot_hashes": ["e6" * 32],
        "text_surface_schema_hash": "e7" * 32,
        "text_normalizer_hash": "e8" * 32,
        "semantic_constraint_hash": "e9" * 32,
        "rendered_text_hash": "a6" * 32,
        "rendered_text": "bounded surface",
        "text_fragments": ["bounded surface"],
        "mean_confidence_score": 0.79,
        "mean_spike_sparsity": 0.72,
        "max_slot_drift": 0.02,
        "literal_text_returned": True,
        "output_is_bounded_text_surface": True,
        "operator_approval_required": False,
        "generates_text": False,
        "decodes_text": False,
        "writes_checkpoint": False,
        "runs_replay": False,
        "applies_plasticity": False,
        "trains_runtime_model": False,
    }


def _autonomous_bounded_language_surface_commit_event() -> dict[str, Any]:
    return {
        "autonomous_bounded_language_surface_commit_event_hash": "b1" * 32,
        "autonomous_bounded_language_surface_commit_event_id": (
            "benchmark-bounded-language-surface-commit"
        ),
        "committed_at": "2026-06-19T00:00:00+00:00",
        "state_revision": 9,
        "language_surface_commit_preflight_hash": "b2" * 32,
        "language_surface_commit_design_hash": "b3" * 32,
        "language_surface_commit_plan_hash": "b4" * 32,
        "bounded_language_surface_hash": "b5" * 32,
        "committed_language_surface_hash": "b5" * 32,
        "previous_language_surface_hash": None,
        "language_surface_state_chain_hash": "b6" * 32,
        "language_surface_review_hash": "b7" * 32,
        "materialization_review_hash": "b8" * 32,
        "autonomous_text_surface_materialization_event_hash": "a1" * 32,
        "commit_scope": "bounded_language_surface",
        "retention_class": "ephemeral_language_surface",
        "rendered_text_hash": "a6" * 32,
        "rendered_text": "bounded surface",
        "text_fragments": ["bounded surface"],
        "text_fragment_count": 1,
        "literal_fragment_hashes": ["a7" * 32],
        "text_fragment_hashes": ["a7" * 32],
        "decoded_token_hashes": ["d7" * 32],
        "text_emission_slot_hashes": ["e6" * 32],
        "text_surface_schema_hash": "e7" * 32,
        "text_normalizer_hash": "e8" * 32,
        "semantic_constraint_hash": "e9" * 32,
        "mean_confidence_score": 0.79,
        "mean_spike_sparsity": 0.72,
        "max_slot_drift": 0.02,
        "literal_text_returned": True,
        "output_is_bounded_text_surface": True,
        "operator_approval_required": False,
        "generates_text": False,
        "decodes_text": False,
        "writes_checkpoint": False,
        "runs_replay": False,
        "applies_plasticity": False,
        "trains_runtime_model": False,
        "promotes_fact": False,
        "executes_action": False,
    }


def _autonomous_bounded_language_surface_use_event() -> dict[str, Any]:
    return {
        "autonomous_bounded_language_surface_use_event_hash": "f1" * 32,
        "autonomous_bounded_language_surface_use_event_id": (
            "benchmark-bounded-language-surface-use"
        ),
        "used_at": "2026-06-19T00:00:00+00:00",
        "state_revision": 10,
        "bounded_language_surface_use_preflight_hash": "f2" * 32,
        "use_artifact_review_hash": "f3" * 32,
        "use_review_hash": "f4" * 32,
        "commit_event_review_hash": "f5" * 32,
        "autonomous_bounded_language_surface_commit_event_hash": "b1" * 32,
        "bounded_language_surface_hash": "b5" * 32,
        "used_language_surface_hash": "b5" * 32,
        "committed_language_surface_hash": "b5" * 32,
        "language_surface_state_chain_hash": "b6" * 32,
        "language_surface_use_chain_hash": "f6" * 32,
        "language_use_scope": "bounded_language_evidence",
        "use_mode": "bounded_language_evidence_observation",
        "rendered_text_hash": "a6" * 32,
        "rendered_text": "bounded surface",
        "text_fragments": ["bounded surface"],
        "text_fragment_count": 1,
        "literal_fragment_hashes": ["a7" * 32],
        "text_fragment_hashes": ["a7" * 32],
        "decoded_token_hashes": ["d7" * 32],
        "text_emission_slot_hashes": ["e6" * 32],
        "text_surface_schema_hash": "e7" * 32,
        "text_normalizer_hash": "e8" * 32,
        "semantic_constraint_hash": "e9" * 32,
        "mean_confidence_score": 0.79,
        "mean_spike_sparsity": 0.72,
        "max_slot_drift": 0.02,
        "literal_text_returned": True,
        "output_is_bounded_text_surface": True,
        "operator_approval_required": False,
        "generates_text": False,
        "decodes_text": False,
        "writes_checkpoint": False,
        "runs_replay": False,
        "applies_plasticity": False,
        "trains_runtime_model": False,
        "promotes_fact": False,
        "executes_action": False,
    }


def _autonomous_snn_language_generation_event() -> dict[str, Any]:
    return {
        "autonomous_snn_language_generation_event_hash": "9a" * 32,
        "autonomous_snn_language_generation_event_id": (
            "benchmark-snn-language-generation"
        ),
        "generated_at": "2026-06-19T00:00:00+00:00",
        "state_revision": 11,
        "language_generation_preflight_hash": "9b" * 32,
        "language_generation_design_hash": "9c" * 32,
        "generation_plan_hash": "9d" * 32,
        "generation_projection_hash": "9e" * 32,
        "use_event_review_hash": "9f" * 32,
        "autonomous_bounded_language_surface_use_event_hash": "f1" * 32,
        "bounded_language_surface_hash": "b5" * 32,
        "language_surface_use_chain_hash": "f6" * 32,
        "rendered_text_hash": "a6" * 32,
        "text_surface_schema_hash": "e7" * 32,
        "text_normalizer_hash": "e8" * 32,
        "semantic_constraint_hash": "e9" * 32,
        "generation_mode": "snn_bounded_next_token_projection",
        "decoding_strategy": "hash_projection_only",
        "requested_device": "cpu",
        "requires_cuda": False,
        "generated_token_hashes": ["11" * 32],
        "generated_token_count": 1,
        "spike_projection_hashes": ["12" * 32],
        "active_neuron_hashes": ["13" * 32],
        "membrane_state_hashes": ["14" * 32],
        "output_fragment_hashes": ["15" * 32],
        "literal_text_returned": False,
        "generated_text_returned": False,
        "operator_approval_required": False,
        "generates_text": False,
        "decodes_text": False,
        "writes_checkpoint": False,
        "runs_replay": False,
        "applies_plasticity": False,
        "trains_runtime_model": False,
        "promotes_fact": False,
        "executes_action": False,
    }


def _autonomous_snn_language_decoding_event() -> dict[str, Any]:
    return {
        "autonomous_snn_language_decoding_event_hash": "21" * 32,
        "autonomous_snn_language_decoding_event_id": (
            "benchmark-snn-language-decoding"
        ),
        "recorded_at": "2026-06-19T00:00:01+00:00",
        "state_revision": 12,
        "language_decoding_preflight_hash": "22" * 32,
        "language_decoding_design_hash": "23" * 32,
        "decoding_plan_hash": "24" * 32,
        "event_review_hash": "25" * 32,
        "autonomous_snn_language_generation_event_hash": "9a" * 32,
        "generated_token_hashes": ["11" * 32],
        "decoded_token_hashes": ["11" * 32],
        "decoded_text_fragment_hashes": ["26" * 32],
        "decoded_text_fragments": ["bounded surface"],
        "rendered_text": "bounded surface",
        "rendered_text_hash": "a6" * 32,
        "spike_projection_hashes": ["12" * 32],
        "active_neuron_hashes": ["13" * 32],
        "membrane_state_hashes": ["14" * 32],
        "output_fragment_hashes": ["15" * 32],
        "text_surface_schema_hash": "e7" * 32,
        "text_normalizer_hash": "e8" * 32,
        "semantic_constraint_hash": "e9" * 32,
        "decoding_mode": "hash_bound_surface_projection",
        "materialization_target": "bounded_local_surface",
        "requested_device": "cpu",
        "schema_valid": True,
        "text_normalized": True,
        "semantic_constraint_valid": True,
        "literal_text_returned": True,
        "generated_text_returned": True,
        "operator_approval_required": False,
        "generates_text": True,
        "decodes_text": True,
        "writes_checkpoint": False,
        "runs_replay": False,
        "applies_plasticity": False,
        "trains_runtime_model": False,
        "promotes_fact": False,
        "executes_action": False,
    }


def _snn_language_readout_surface_event() -> dict[str, Any]:
    return {
        "snn_language_readout_surface_event_hash": "31" * 32,
        "snn_language_readout_surface_event_id": (
            "benchmark-snn-language-readout-surface"
        ),
        "recorded_at": "2026-06-19T00:00:02+00:00",
        "state_revision": 13,
        "language_readout_surface_preflight_hash": "32" * 32,
        "language_readout_surface_design_hash": "33" * 32,
        "autonomous_snn_language_decoding_event_review_hash": "34" * 32,
        "autonomous_snn_language_decoding_event_hash": "21" * 32,
        "readout_surface_hash": "35" * 32,
        "readout_role": "bounded_readout_candidate",
        "binding_mode": "hash_bound_readout_language",
        "rendered_text": "bounded surface",
        "decoded_text_fragments": ["bounded surface"],
        "generated_token_hashes": ["11" * 32],
        "decoded_token_hashes": ["11" * 32],
        "spike_projection_hashes": ["12" * 32],
        "active_neuron_hashes": ["13" * 32],
        "membrane_state_hashes": ["14" * 32],
        "output_fragment_hashes": ["15" * 32],
        "text_surface_schema_hash": "e7" * 32,
        "text_normalizer_hash": "e8" * 32,
        "semantic_constraint_hash": "e9" * 32,
        "requested_device": "cpu",
        "max_readout_fragments": 1,
        "max_surface_chars": 256,
        "max_association_edges": 1,
        "schema_valid": True,
        "text_normalized": True,
        "semantic_constraint_valid": True,
        "cognition_substrate_claimed": False,
        "operator_approval_required": False,
        "generates_text": True,
        "decodes_text": True,
        "writes_checkpoint": False,
        "runs_replay": False,
        "applies_plasticity": False,
        "trains_runtime_model": False,
        "promotes_fact": False,
        "executes_action": False,
    }


def _snn_language_readout_memory_event() -> dict[str, Any]:
    return {
        "snn_language_readout_memory_event_hash": "41" * 32,
        "snn_language_readout_memory_event_id": (
            "benchmark-snn-language-readout-memory"
        ),
        "recorded_at": "2026-06-19T00:00:03+00:00",
        "state_revision": 14,
        "language_readout_memory_preflight_hash": "42" * 32,
        "language_readout_memory_design_hash": "43" * 32,
        "readout_surface_event_review_hash": "44" * 32,
        "snn_language_readout_surface_event_hash": "31" * 32,
        "readout_surface_hash": "35" * 32,
        "memory_trace_hash": "45" * 32,
        "memory_scope": "working_trace",
        "consolidation_route": "deferred_local_trace",
        "requested_device": "cpu",
        "readout_role": "bounded_readout_candidate",
        "binding_mode": "hash_bound_readout_language",
        "rendered_text": "bounded surface",
        "decoded_text_fragments": ["bounded surface"],
        "generated_token_hashes": ["11" * 32],
        "decoded_token_hashes": ["11" * 32],
        "spike_projection_hashes": ["12" * 32],
        "active_neuron_hashes": ["13" * 32],
        "membrane_state_hashes": ["14" * 32],
        "output_fragment_hashes": ["15" * 32],
        "local_learning_target_hashes": ["46" * 32],
        "text_surface_schema_hash": "e7" * 32,
        "text_normalizer_hash": "e8" * 32,
        "semantic_constraint_hash": "e9" * 32,
        "max_trace_events": 1,
        "max_local_learning_targets": 1,
        "memory_recorded": True,
        "cognition_substrate_claimed": False,
        "operator_approval_required": False,
        "generates_text": True,
        "decodes_text": True,
        "writes_checkpoint": False,
        "runs_replay": False,
        "applies_plasticity": False,
        "trains_runtime_model": False,
        "promotes_fact": False,
        "executes_action": False,
    }


def _autonomous_snn_language_thought_consolidation_event() -> dict[str, Any]:
    candidate_updates = [
        {
            "local_learning_target_hash": "46" * 32,
            "applied_to_runtime": True,
            "applied_in_ledger": True,
        }
    ]
    return {
        "autonomous_snn_language_thought_consolidation_event_hash": "51" * 32,
        "autonomous_snn_language_thought_consolidation_event_id": (
            "benchmark-snn-language-thought-consolidation"
        ),
        "recorded_at": "2026-06-19T00:00:04+00:00",
        "state_revision": 15,
        "thought_consolidation_preflight_hash": "52" * 32,
        "thought_consolidation_design_hash": "53" * 32,
        "memory_trace_hash": "45" * 32,
        "snn_language_readout_memory_event_hash": "41" * 32,
        "consolidation_scope": "local_trace_reinforcement",
        "consolidation_route": "deferred_local_trace",
        "requested_device": "cpu",
        "candidate_update_count": len(candidate_updates),
        "candidate_updates": candidate_updates,
        "local_learning_target_hashes": ["46" * 32],
        "generated_token_hashes": ["11" * 32],
        "spike_projection_hashes": ["12" * 32],
        "learning_rate": 0.02,
        "max_weight_delta": 0.04,
        "homeostatic_decay": 0.01,
        "local_only": True,
        "normalization": True,
        "plasticity_applied": True,
        "runtime_state_mutated": True,
        "cognition_substrate_claimed": False,
        "operator_approval_required": False,
        "generates_text": True,
        "decodes_text": True,
        "writes_checkpoint": False,
        "runs_replay": False,
        "applies_plasticity": True,
        "trains_runtime_model": False,
        "promotes_fact": False,
        "executes_action": False,
    }


def _autonomous_snn_language_thought_structural_plasticity_event() -> dict[str, Any]:
    growth_candidates = [
        {
            "local_learning_target_hash": "46" * 32,
            "applied_to_runtime": True,
            "applied_in_ledger": True,
        }
    ]
    return {
        "autonomous_snn_language_thought_structural_plasticity_event_hash": (
            "61" * 32
        ),
        "autonomous_snn_language_thought_structural_plasticity_event_id": (
            "benchmark-snn-language-thought-structural-plasticity"
        ),
        "recorded_at": "2026-06-19T00:00:05+00:00",
        "state_revision": 16,
        "thought_structural_plasticity_preflight_hash": "62" * 32,
        "thought_structural_plasticity_design_hash": "63" * 32,
        "consolidation_event_review_hash": "64" * 32,
        "autonomous_snn_language_thought_consolidation_event_hash": "51" * 32,
        "memory_trace_hash": "45" * 32,
        "snn_language_readout_memory_event_hash": "41" * 32,
        "structural_scope": "thought_trace_sparse_capacity",
        "structural_route": "reviewed_consolidation_to_growth_prune",
        "requested_device": "cpu",
        "growth_candidate_count": len(growth_candidates),
        "growth_candidates": growth_candidates,
        "prune_candidate_count": 0,
        "prune_candidates": [],
        "proposed_new_neuron_count": 1,
        "proposed_new_synapse_count": 1,
        "proposed_prune_synapse_count": 0,
        "max_new_neurons": 1,
        "max_new_synapses": 1,
        "max_prune_synapses": 0,
        "structural_plasticity_applied": True,
        "runtime_state_mutated": True,
        "cognition_substrate_claimed": False,
        "operator_approval_required": False,
        "generates_text": True,
        "decodes_text": True,
        "writes_checkpoint": False,
        "runs_replay": False,
        "applies_plasticity": False,
        "trains_runtime_model": False,
        "promotes_fact": False,
        "executes_action": False,
        "adds_neurons": True,
        "adds_synapses": True,
        "prunes_network": False,
    }


def _bounded_append_review_record_family(
    ledger: SNNLanguageReadoutEvidenceLedger,
    *,
    field: str,
    event: Mapping[str, Any],
    duplicate_key: str,
    total_count_key: str,
    timestamp_key: str,
    timestamp_value: str,
) -> dict[str, Any]:
    duplicate, summary, append_window = ledger._append_record_family_window(  # noqa: SLF001
        field=field,
        event=dict(event),
        duplicate_key=duplicate_key,
        total_count_key=total_count_key,
        timestamp_key=timestamp_key,
        timestamp_value=timestamp_value,
    )
    source_events, review_window = ledger._record_family_window_with_report(  # noqa: SLF001
        field=field,
        duplicate_key=duplicate_key,
    )
    event_hash = str(event.get(duplicate_key) or "")
    return {
        "duplicate": duplicate,
        "hash": event_hash,
        "review_match": any(
            str(item.get(duplicate_key) or "") == event_hash
            for item in source_events
        ),
        "total_count": int(summary.get(total_count_key, 0) or 0),
        "append_window": append_window,
        "review_window": review_window,
    }


def _broad_normalized_append_review_record_family(
    ledger: SNNLanguageReadoutEvidenceLedger,
    *,
    field: str,
    event: Mapping[str, Any],
    duplicate_key: str,
    total_count_key: str,
    timestamp_key: str,
    timestamp_value: str,
) -> dict[str, Any]:
    event_hash = str(event.get(duplicate_key) or "")
    normalized = _benchmark_bounded_normalized_state(ledger)
    append_source = dict(normalized.get("_normalization_source_window") or {})
    source_events = normalized[field]
    duplicate = event_hash in {
        str(item.get(duplicate_key) or "") for item in list(source_events)
    }
    if not duplicate:
        source_events.appendleft(deepcopy(dict(event)))
        normalized[total_count_key] = int(
            normalized.get(total_count_key, 0) or 0
        ) + 1
        normalized[timestamp_key] = timestamp_value
        ledger._store_state(normalized)  # noqa: SLF001

    normalized = _benchmark_bounded_normalized_state(ledger)
    review_source = dict(normalized.get("_normalization_source_window") or {})
    return {
        "duplicate": duplicate,
        "hash": event_hash,
        "review_match": any(
            str(item.get(duplicate_key) or "") == event_hash
            for item in list(normalized[field])
        ),
        "total_count": int(
            ledger._ledger_state().get(total_count_key, 0) or 0  # noqa: SLF001
        ),
        "append_window": append_source,
        "review_window": review_source,
    }


def _snn_language_thought_family_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "language_decoding",
            "field": "autonomous_snn_language_decoding_events",
            "event": _autonomous_snn_language_decoding_event(),
            "duplicate_key": "autonomous_snn_language_decoding_event_hash",
            "total_count_key": "total_autonomous_snn_language_decoding_count",
            "timestamp_key": "last_autonomous_snn_language_decoded_at",
            "timestamp_value_key": "recorded_at",
        },
        {
            "name": "readout_surface",
            "field": "snn_language_readout_surface_events",
            "event": _snn_language_readout_surface_event(),
            "duplicate_key": "snn_language_readout_surface_event_hash",
            "total_count_key": (
                "total_snn_language_readout_surface_count"
            ),
            "timestamp_key": (
                "last_snn_language_readout_surface_recorded_at"
            ),
            "timestamp_value_key": "recorded_at",
        },
        {
            "name": "readout_memory",
            "field": "snn_language_readout_memory_events",
            "event": _snn_language_readout_memory_event(),
            "duplicate_key": "snn_language_readout_memory_event_hash",
            "total_count_key": "total_snn_language_readout_memory_count",
            "timestamp_key": (
                "last_snn_language_readout_memory_recorded_at"
            ),
            "timestamp_value_key": "recorded_at",
        },
        {
            "name": "thought_consolidation",
            "field": "autonomous_snn_language_thought_consolidation_events",
            "event": _autonomous_snn_language_thought_consolidation_event(),
            "duplicate_key": (
                "autonomous_snn_language_thought_consolidation_event_hash"
            ),
            "total_count_key": (
                "total_autonomous_snn_language_thought_consolidation_count"
            ),
            "timestamp_key": "last_autonomous_snn_language_thought_consolidated_at",
            "timestamp_value_key": "recorded_at",
        },
        {
            "name": "thought_structural_plasticity",
            "field": (
                "autonomous_snn_language_thought_structural_plasticity_events"
            ),
            "event": _autonomous_snn_language_thought_structural_plasticity_event(),
            "duplicate_key": (
                "autonomous_snn_language_thought_structural_plasticity_event_hash"
            ),
            "total_count_key": (
                "total_autonomous_snn_language_thought_structural_plasticity_count"
            ),
            "timestamp_key": (
                "last_autonomous_snn_language_thought_structural_plasticity_applied_at"
            ),
            "timestamp_value_key": "recorded_at",
        },
    )


def _bounded_autonomous_readout_event_family_chain(
    ledger: SNNLanguageReadoutEvidenceLedger,
) -> dict[str, Any]:
    binding_event = _autonomous_binding_event()
    binding_duplicate, binding_summary, binding_append_window = (
        ledger._append_record_family_window(  # noqa: SLF001
            field="autonomous_hash_readout_binding_events",
            event=binding_event,
            duplicate_key="autonomous_hash_readout_binding_event_hash",
            total_count_key="total_autonomous_hash_readout_binding_count",
            timestamp_key="last_autonomous_hash_readout_bound_at",
            timestamp_value=binding_event["bound_at"],
        )
    )
    binding_events, binding_review_window = ledger._record_family_window_with_report(  # noqa: SLF001
        field="autonomous_hash_readout_binding_events",
        duplicate_key="autonomous_hash_readout_binding_event_hash",
    )
    binding_hash = binding_event["autonomous_hash_readout_binding_event_hash"]
    binding_review_match = any(
        str(item.get("autonomous_hash_readout_binding_event_hash") or "")
        == binding_hash
        for item in binding_events
    )

    observation_event = _autonomous_observation_event()
    observation_duplicate, observation_summary, observation_append_window = (
        ledger._append_record_family_window(  # noqa: SLF001
            field="autonomous_bound_readout_observation_events",
            event=observation_event,
            duplicate_key="autonomous_bound_readout_observation_event_hash",
            total_count_key="total_autonomous_bound_readout_observation_count",
            timestamp_key="last_autonomous_bound_readout_observed_at",
            timestamp_value=observation_event["observed_at"],
        )
    )
    observation_events, observation_review_window = (
        ledger._record_family_window_with_report(  # noqa: SLF001
            field="autonomous_bound_readout_observation_events",
            duplicate_key="autonomous_bound_readout_observation_event_hash",
        )
    )
    observation_hash = observation_event[
        "autonomous_bound_readout_observation_event_hash"
    ]
    observation_review_match = any(
        str(item.get("autonomous_bound_readout_observation_event_hash") or "")
        == observation_hash
        for item in observation_events
    )

    training_event = _autonomous_training_window_event()
    training_duplicate, training_summary, training_append_window = (
        ledger._append_record_family_window(  # noqa: SLF001
            field="autonomous_readout_training_window_events",
            event=training_event,
            duplicate_key="autonomous_readout_training_window_event_hash",
            total_count_key="total_autonomous_readout_training_window_count",
            timestamp_key="last_autonomous_readout_training_window_trained_at",
            timestamp_value=training_event["trained_at"],
        )
    )
    training_events, training_review_window = ledger._record_family_window_with_report(  # noqa: SLF001
        field="autonomous_readout_training_window_events",
        duplicate_key="autonomous_readout_training_window_event_hash",
    )
    training_hash = training_event["autonomous_readout_training_window_event_hash"]
    training_review_match = any(
        str(item.get("autonomous_readout_training_window_event_hash") or "")
        == training_hash
        for item in training_events
    )

    decoder_event = _autonomous_decoder_probe_event()
    decoder_duplicate, decoder_summary, decoder_append_window = (
        ledger._append_record_family_window(  # noqa: SLF001
            field="autonomous_decoder_probe_events",
            event=decoder_event,
            duplicate_key="autonomous_decoder_probe_event_hash",
            total_count_key="total_autonomous_decoder_probe_count",
            timestamp_key="last_autonomous_decoder_probed_at",
            timestamp_value=decoder_event["probed_at"],
        )
    )
    decoder_events, decoder_review_window = ledger._record_family_window_with_report(  # noqa: SLF001
        field="autonomous_decoder_probe_events",
        duplicate_key="autonomous_decoder_probe_event_hash",
    )
    decoder_hash = decoder_event["autonomous_decoder_probe_event_hash"]
    decoder_review_match = any(
        str(item.get("autonomous_decoder_probe_event_hash") or "") == decoder_hash
        for item in decoder_events
    )

    language_output_event = _autonomous_language_output_event()
    language_output_duplicate, language_output_summary, language_output_append_window = (
        ledger._append_record_family_window(  # noqa: SLF001
            field="autonomous_language_output_events",
            event=language_output_event,
            duplicate_key="autonomous_language_output_event_hash",
            total_count_key="total_autonomous_language_output_count",
            timestamp_key="last_autonomous_language_output_emitted_at",
            timestamp_value=language_output_event["emitted_at"],
        )
    )
    language_output_events, language_output_review_window = (
        ledger._record_family_window_with_report(  # noqa: SLF001
            field="autonomous_language_output_events",
            duplicate_key="autonomous_language_output_event_hash",
        )
    )
    language_output_hash = language_output_event[
        "autonomous_language_output_event_hash"
    ]
    language_output_review_match = any(
        str(item.get("autonomous_language_output_event_hash") or "")
        == language_output_hash
        for item in language_output_events
    )

    decoded_output_event = _autonomous_decoded_output_event()
    decoded_output_duplicate, decoded_output_summary, decoded_output_append_window = (
        ledger._append_record_family_window(  # noqa: SLF001
            field="autonomous_decoded_output_events",
            event=decoded_output_event,
            duplicate_key="autonomous_decoded_output_event_hash",
            total_count_key="total_autonomous_decoded_output_count",
            timestamp_key="last_autonomous_decoded_output_at",
            timestamp_value=decoded_output_event["decoded_at"],
        )
    )
    decoded_output_events, decoded_output_review_window = (
        ledger._record_family_window_with_report(  # noqa: SLF001
            field="autonomous_decoded_output_events",
            duplicate_key="autonomous_decoded_output_event_hash",
        )
    )
    decoded_output_hash = decoded_output_event[
        "autonomous_decoded_output_event_hash"
    ]
    decoded_output_review_match = any(
        str(item.get("autonomous_decoded_output_event_hash") or "")
        == decoded_output_hash
        for item in decoded_output_events
    )

    text_emission_event = _autonomous_bounded_text_emission_event()
    text_emission_duplicate, text_emission_summary, text_emission_append_window = (
        ledger._append_record_family_window(  # noqa: SLF001
            field="autonomous_bounded_text_emission_events",
            event=text_emission_event,
            duplicate_key="autonomous_bounded_text_emission_event_hash",
            total_count_key="total_autonomous_bounded_text_emission_count",
            timestamp_key="last_autonomous_bounded_text_emitted_at",
            timestamp_value=text_emission_event["emitted_at"],
        )
    )
    text_emission_events, text_emission_review_window = (
        ledger._record_family_window_with_report(  # noqa: SLF001
            field="autonomous_bounded_text_emission_events",
            duplicate_key="autonomous_bounded_text_emission_event_hash",
        )
    )
    text_emission_hash = text_emission_event[
        "autonomous_bounded_text_emission_event_hash"
    ]
    text_emission_review_match = any(
        str(item.get("autonomous_bounded_text_emission_event_hash") or "")
        == text_emission_hash
        for item in text_emission_events
    )

    text_commit_event = _autonomous_text_surface_commit_event()
    text_commit_duplicate, text_commit_summary, text_commit_append_window = (
        ledger._append_record_family_window(  # noqa: SLF001
            field="autonomous_text_surface_commit_events",
            event=text_commit_event,
            duplicate_key="autonomous_text_surface_commit_event_hash",
            total_count_key="total_autonomous_text_surface_commit_count",
            timestamp_key="last_autonomous_text_surface_committed_at",
            timestamp_value=text_commit_event["committed_at"],
        )
    )
    if not text_commit_duplicate:
        ledger._ledger_state()["current_text_surface_commit"] = deepcopy(  # noqa: SLF001
            text_commit_event
        )
    text_commit_events, text_commit_review_window = (
        ledger._record_family_window_with_report(  # noqa: SLF001
            field="autonomous_text_surface_commit_events",
            duplicate_key="autonomous_text_surface_commit_event_hash",
        )
    )
    text_commit_hash = text_commit_event["autonomous_text_surface_commit_event_hash"]
    text_commit_review_match = any(
        str(item.get("autonomous_text_surface_commit_event_hash") or "")
        == text_commit_hash
        for item in text_commit_events
    )
    text_commit_current_match = (
        dict(ledger._ledger_state().get("current_text_surface_commit") or {})  # noqa: SLF001
        == text_commit_event
    )

    materialization_event = _autonomous_text_surface_materialization_event()
    materialization_duplicate, materialization_summary, materialization_append_window = (
        ledger._append_record_family_window(  # noqa: SLF001
            field="autonomous_text_surface_materialization_events",
            event=materialization_event,
            duplicate_key="autonomous_text_surface_materialization_event_hash",
            total_count_key="total_autonomous_text_surface_materialization_count",
            timestamp_key="last_autonomous_text_surface_materialized_at",
            timestamp_value=materialization_event["materialized_at"],
        )
    )
    if not materialization_duplicate:
        ledger._ledger_state()[  # noqa: SLF001
            "current_text_surface_materialization"
        ] = deepcopy(materialization_event)
    materialization_events, materialization_review_window = (
        ledger._record_family_window_with_report(  # noqa: SLF001
            field="autonomous_text_surface_materialization_events",
            duplicate_key="autonomous_text_surface_materialization_event_hash",
        )
    )
    materialization_hash = materialization_event[
        "autonomous_text_surface_materialization_event_hash"
    ]
    materialization_review_match = any(
        str(item.get("autonomous_text_surface_materialization_event_hash") or "")
        == materialization_hash
        for item in materialization_events
    )
    materialization_current_match = (
        dict(
            ledger._ledger_state().get("current_text_surface_materialization")  # noqa: SLF001
            or {}
        )
        == materialization_event
    )

    language_surface_commit_event = (
        _autonomous_bounded_language_surface_commit_event()
    )
    (
        language_surface_commit_duplicate,
        language_surface_commit_summary,
        language_surface_commit_append_window,
    ) = ledger._append_record_family_window(  # noqa: SLF001
        field="autonomous_bounded_language_surface_commit_events",
        event=language_surface_commit_event,
        duplicate_key="autonomous_bounded_language_surface_commit_event_hash",
        total_count_key="total_autonomous_bounded_language_surface_commit_count",
        timestamp_key="last_autonomous_bounded_language_surface_committed_at",
        timestamp_value=language_surface_commit_event["committed_at"],
    )
    if not language_surface_commit_duplicate:
        ledger._ledger_state()[  # noqa: SLF001
            "current_bounded_language_surface_commit"
        ] = deepcopy(language_surface_commit_event)
    language_surface_commit_events, language_surface_commit_review_window = (
        ledger._record_family_window_with_report(  # noqa: SLF001
            field="autonomous_bounded_language_surface_commit_events",
            duplicate_key="autonomous_bounded_language_surface_commit_event_hash",
        )
    )
    language_surface_commit_hash = language_surface_commit_event[
        "autonomous_bounded_language_surface_commit_event_hash"
    ]
    language_surface_commit_review_match = any(
        str(item.get("autonomous_bounded_language_surface_commit_event_hash") or "")
        == language_surface_commit_hash
        for item in language_surface_commit_events
    )
    language_surface_commit_current_match = (
        dict(
            ledger._ledger_state().get("current_bounded_language_surface_commit")  # noqa: SLF001
            or {}
        )
        == language_surface_commit_event
    )

    language_surface_use_event = _autonomous_bounded_language_surface_use_event()
    (
        language_surface_use_duplicate,
        language_surface_use_summary,
        language_surface_use_append_window,
    ) = ledger._append_record_family_window(  # noqa: SLF001
        field="autonomous_bounded_language_surface_use_events",
        event=language_surface_use_event,
        duplicate_key="autonomous_bounded_language_surface_use_event_hash",
        total_count_key="total_autonomous_bounded_language_surface_use_count",
        timestamp_key="last_autonomous_bounded_language_surface_used_at",
        timestamp_value=language_surface_use_event["used_at"],
    )
    language_surface_use_events, language_surface_use_review_window = (
        ledger._record_family_window_with_report(  # noqa: SLF001
            field="autonomous_bounded_language_surface_use_events",
            duplicate_key="autonomous_bounded_language_surface_use_event_hash",
        )
    )
    language_surface_use_hash = language_surface_use_event[
        "autonomous_bounded_language_surface_use_event_hash"
    ]
    language_surface_use_review_match = any(
        str(item.get("autonomous_bounded_language_surface_use_event_hash") or "")
        == language_surface_use_hash
        for item in language_surface_use_events
    )

    language_generation_event = _autonomous_snn_language_generation_event()
    (
        language_generation_duplicate,
        language_generation_summary,
        language_generation_append_window,
    ) = ledger._append_record_family_window(  # noqa: SLF001
        field="autonomous_snn_language_generation_events",
        event=language_generation_event,
        duplicate_key="autonomous_snn_language_generation_event_hash",
        total_count_key="total_autonomous_snn_language_generation_count",
        timestamp_key="last_autonomous_snn_language_generated_at",
        timestamp_value=language_generation_event["generated_at"],
    )
    language_generation_events, language_generation_review_window = (
        ledger._record_family_window_with_report(  # noqa: SLF001
            field="autonomous_snn_language_generation_events",
            duplicate_key="autonomous_snn_language_generation_event_hash",
        )
    )
    language_generation_hash = language_generation_event[
        "autonomous_snn_language_generation_event_hash"
    ]
    language_generation_review_match = any(
        str(item.get("autonomous_snn_language_generation_event_hash") or "")
        == language_generation_hash
        for item in language_generation_events
    )
    thought_chain_results = {}
    for spec in _snn_language_thought_family_specs():
        event = dict(spec["event"])
        thought_chain_results[str(spec["name"])] = (
            _bounded_append_review_record_family(
                ledger,
                field=str(spec["field"]),
                event=event,
                duplicate_key=str(spec["duplicate_key"]),
                total_count_key=str(spec["total_count_key"]),
                timestamp_key=str(spec["timestamp_key"]),
                timestamp_value=str(event[str(spec["timestamp_value_key"])]),
            )
        )

    return {
        "binding_duplicate": binding_duplicate,
        "observation_duplicate": observation_duplicate,
        "training_duplicate": training_duplicate,
        "decoder_duplicate": decoder_duplicate,
        "language_output_duplicate": language_output_duplicate,
        "decoded_output_duplicate": decoded_output_duplicate,
        "text_emission_duplicate": text_emission_duplicate,
        "text_commit_duplicate": text_commit_duplicate,
        "materialization_duplicate": materialization_duplicate,
        "language_surface_commit_duplicate": language_surface_commit_duplicate,
        "language_surface_use_duplicate": language_surface_use_duplicate,
        "language_generation_duplicate": language_generation_duplicate,
        **{
            f"{name}_duplicate": bool(result.get("duplicate"))
            for name, result in thought_chain_results.items()
        },
        "binding_hash": binding_hash,
        "observation_hash": observation_hash,
        "training_hash": training_hash,
        "decoder_hash": decoder_hash,
        "language_output_hash": language_output_hash,
        "decoded_output_hash": decoded_output_hash,
        "text_emission_hash": text_emission_hash,
        "text_commit_hash": text_commit_hash,
        "materialization_hash": materialization_hash,
        "language_surface_commit_hash": language_surface_commit_hash,
        "language_surface_use_hash": language_surface_use_hash,
        "language_generation_hash": language_generation_hash,
        **{
            f"{name}_hash": str(result.get("hash") or "")
            for name, result in thought_chain_results.items()
        },
        "binding_review_match": binding_review_match,
        "observation_review_match": observation_review_match,
        "training_review_match": training_review_match,
        "decoder_review_match": decoder_review_match,
        "language_output_review_match": language_output_review_match,
        "decoded_output_review_match": decoded_output_review_match,
        "text_emission_review_match": text_emission_review_match,
        "text_commit_review_match": text_commit_review_match,
        "materialization_review_match": materialization_review_match,
        "language_surface_commit_review_match": (
            language_surface_commit_review_match
        ),
        "language_surface_use_review_match": language_surface_use_review_match,
        "language_generation_review_match": language_generation_review_match,
        **{
            f"{name}_review_match": bool(result.get("review_match"))
            for name, result in thought_chain_results.items()
        },
        "text_commit_current_match": text_commit_current_match,
        "materialization_current_match": materialization_current_match,
        "language_surface_commit_current_match": (
            language_surface_commit_current_match
        ),
        "binding_total_count": int(
            binding_summary.get("total_autonomous_hash_readout_binding_count", 0)
            or 0
        ),
        "observation_total_count": int(
            observation_summary.get(
                "total_autonomous_bound_readout_observation_count",
                0,
            )
            or 0
        ),
        "training_total_count": int(
            training_summary.get("total_autonomous_readout_training_window_count", 0)
            or 0
        ),
        "decoder_total_count": int(
            decoder_summary.get("total_autonomous_decoder_probe_count", 0)
            or 0
        ),
        "language_output_total_count": int(
            language_output_summary.get("total_autonomous_language_output_count", 0)
            or 0
        ),
        "decoded_output_total_count": int(
            decoded_output_summary.get("total_autonomous_decoded_output_count", 0)
            or 0
        ),
        "text_emission_total_count": int(
            text_emission_summary.get(
                "total_autonomous_bounded_text_emission_count",
                0,
            )
            or 0
        ),
        "text_commit_total_count": int(
            text_commit_summary.get("total_autonomous_text_surface_commit_count", 0)
            or 0
        ),
        "materialization_total_count": int(
            materialization_summary.get(
                "total_autonomous_text_surface_materialization_count",
                0,
            )
            or 0
        ),
        "language_surface_commit_total_count": int(
            language_surface_commit_summary.get(
                "total_autonomous_bounded_language_surface_commit_count",
                0,
            )
            or 0
        ),
        "language_surface_use_total_count": int(
            language_surface_use_summary.get(
                "total_autonomous_bounded_language_surface_use_count",
                0,
            )
            or 0
        ),
        "language_generation_total_count": int(
            language_generation_summary.get(
                "total_autonomous_snn_language_generation_count",
                0,
            )
            or 0
        ),
        **{
            f"{name}_total_count": int(result.get("total_count", 0) or 0)
            for name, result in thought_chain_results.items()
        },
        "source_windows": {
            "binding_append": binding_append_window,
            "binding_review": binding_review_window,
            "observation_append": observation_append_window,
            "observation_review": observation_review_window,
            "training_append": training_append_window,
            "training_review": training_review_window,
            "decoder_append": decoder_append_window,
            "decoder_review": decoder_review_window,
            "language_output_append": language_output_append_window,
            "language_output_review": language_output_review_window,
            "decoded_output_append": decoded_output_append_window,
            "decoded_output_review": decoded_output_review_window,
            "text_emission_append": text_emission_append_window,
            "text_emission_review": text_emission_review_window,
            "text_commit_append": text_commit_append_window,
            "text_commit_review": text_commit_review_window,
            "materialization_append": materialization_append_window,
            "materialization_review": materialization_review_window,
            "language_surface_commit_append": (
                language_surface_commit_append_window
            ),
            "language_surface_commit_review": (
                language_surface_commit_review_window
            ),
            "language_surface_use_append": language_surface_use_append_window,
            "language_surface_use_review": language_surface_use_review_window,
            "language_generation_append": language_generation_append_window,
            "language_generation_review": language_generation_review_window,
            **{
                f"{name}_append": result["append_window"]
                for name, result in thought_chain_results.items()
            },
            **{
                f"{name}_review": result["review_window"]
                for name, result in thought_chain_results.items()
            },
        },
    }


def _broad_normalized_autonomous_readout_event_family_chain(
    ledger: SNNLanguageReadoutEvidenceLedger,
) -> dict[str, Any]:
    binding_event = _autonomous_binding_event()
    binding_hash = binding_event["autonomous_hash_readout_binding_event_hash"]
    normalized = _benchmark_bounded_normalized_state(ledger)
    binding_append_source = dict(normalized.get("_normalization_source_window") or {})
    binding_events = normalized["autonomous_hash_readout_binding_events"]
    binding_duplicate = binding_hash in {
        str(item.get("autonomous_hash_readout_binding_event_hash") or "")
        for item in list(binding_events)
    }
    if not binding_duplicate:
        binding_events.appendleft(deepcopy(binding_event))
        normalized["total_autonomous_hash_readout_binding_count"] = int(
            normalized.get("total_autonomous_hash_readout_binding_count", 0) or 0
        ) + 1
        normalized["last_autonomous_hash_readout_bound_at"] = binding_event[
            "bound_at"
        ]
        ledger._store_state(normalized)  # noqa: SLF001

    normalized = _benchmark_bounded_normalized_state(ledger)
    binding_review_source = dict(normalized.get("_normalization_source_window") or {})
    binding_review_match = any(
        str(item.get("autonomous_hash_readout_binding_event_hash") or "")
        == binding_hash
        for item in list(normalized["autonomous_hash_readout_binding_events"])
    )

    observation_event = _autonomous_observation_event()
    observation_hash = observation_event[
        "autonomous_bound_readout_observation_event_hash"
    ]
    normalized = _benchmark_bounded_normalized_state(ledger)
    observation_append_source = dict(
        normalized.get("_normalization_source_window") or {}
    )
    observation_events = normalized["autonomous_bound_readout_observation_events"]
    observation_duplicate = observation_hash in {
        str(item.get("autonomous_bound_readout_observation_event_hash") or "")
        for item in list(observation_events)
    }
    if not observation_duplicate:
        observation_events.appendleft(deepcopy(observation_event))
        normalized["total_autonomous_bound_readout_observation_count"] = int(
            normalized.get("total_autonomous_bound_readout_observation_count", 0) or 0
        ) + 1
        normalized["last_autonomous_bound_readout_observed_at"] = observation_event[
            "observed_at"
        ]
        ledger._store_state(normalized)  # noqa: SLF001

    normalized = _benchmark_bounded_normalized_state(ledger)
    observation_review_source = dict(
        normalized.get("_normalization_source_window") or {}
    )
    observation_review_match = any(
        str(item.get("autonomous_bound_readout_observation_event_hash") or "")
        == observation_hash
        for item in list(normalized["autonomous_bound_readout_observation_events"])
    )

    training_event = _autonomous_training_window_event()
    training_hash = training_event["autonomous_readout_training_window_event_hash"]
    normalized = _benchmark_bounded_normalized_state(ledger)
    training_append_source = dict(normalized.get("_normalization_source_window") or {})
    training_events = normalized["autonomous_readout_training_window_events"]
    training_duplicate = training_hash in {
        str(item.get("autonomous_readout_training_window_event_hash") or "")
        for item in list(training_events)
    }
    if not training_duplicate:
        training_events.appendleft(deepcopy(training_event))
        normalized["total_autonomous_readout_training_window_count"] = int(
            normalized.get("total_autonomous_readout_training_window_count", 0) or 0
        ) + 1
        normalized["last_autonomous_readout_training_window_trained_at"] = (
            training_event["trained_at"]
        )
        ledger._store_state(normalized)  # noqa: SLF001

    normalized = _benchmark_bounded_normalized_state(ledger)
    training_review_source = dict(normalized.get("_normalization_source_window") or {})
    training_review_match = any(
        str(item.get("autonomous_readout_training_window_event_hash") or "")
        == training_hash
        for item in list(normalized["autonomous_readout_training_window_events"])
    )

    decoder_event = _autonomous_decoder_probe_event()
    decoder_hash = decoder_event["autonomous_decoder_probe_event_hash"]
    normalized = _benchmark_bounded_normalized_state(ledger)
    decoder_append_source = dict(normalized.get("_normalization_source_window") or {})
    decoder_events = normalized["autonomous_decoder_probe_events"]
    decoder_duplicate = decoder_hash in {
        str(item.get("autonomous_decoder_probe_event_hash") or "")
        for item in list(decoder_events)
    }
    if not decoder_duplicate:
        decoder_events.appendleft(deepcopy(decoder_event))
        normalized["total_autonomous_decoder_probe_count"] = int(
            normalized.get("total_autonomous_decoder_probe_count", 0) or 0
        ) + 1
        normalized["last_autonomous_decoder_probed_at"] = decoder_event["probed_at"]
        ledger._store_state(normalized)  # noqa: SLF001

    normalized = _benchmark_bounded_normalized_state(ledger)
    decoder_review_source = dict(normalized.get("_normalization_source_window") or {})
    decoder_review_match = any(
        str(item.get("autonomous_decoder_probe_event_hash") or "") == decoder_hash
        for item in list(normalized["autonomous_decoder_probe_events"])
    )

    language_output_event = _autonomous_language_output_event()
    language_output_hash = language_output_event[
        "autonomous_language_output_event_hash"
    ]
    normalized = _benchmark_bounded_normalized_state(ledger)
    language_output_append_source = dict(
        normalized.get("_normalization_source_window") or {}
    )
    language_output_events = normalized["autonomous_language_output_events"]
    language_output_duplicate = language_output_hash in {
        str(item.get("autonomous_language_output_event_hash") or "")
        for item in list(language_output_events)
    }
    if not language_output_duplicate:
        language_output_events.appendleft(deepcopy(language_output_event))
        normalized["total_autonomous_language_output_count"] = int(
            normalized.get("total_autonomous_language_output_count", 0) or 0
        ) + 1
        normalized["last_autonomous_language_output_emitted_at"] = (
            language_output_event["emitted_at"]
        )
        ledger._store_state(normalized)  # noqa: SLF001

    normalized = _benchmark_bounded_normalized_state(ledger)
    language_output_review_source = dict(
        normalized.get("_normalization_source_window") or {}
    )
    language_output_review_match = any(
        str(item.get("autonomous_language_output_event_hash") or "")
        == language_output_hash
        for item in list(normalized["autonomous_language_output_events"])
    )

    decoded_output_event = _autonomous_decoded_output_event()
    decoded_output_hash = decoded_output_event["autonomous_decoded_output_event_hash"]
    normalized = _benchmark_bounded_normalized_state(ledger)
    decoded_output_append_source = dict(
        normalized.get("_normalization_source_window") or {}
    )
    decoded_output_events = normalized["autonomous_decoded_output_events"]
    decoded_output_duplicate = decoded_output_hash in {
        str(item.get("autonomous_decoded_output_event_hash") or "")
        for item in list(decoded_output_events)
    }
    if not decoded_output_duplicate:
        decoded_output_events.appendleft(deepcopy(decoded_output_event))
        normalized["total_autonomous_decoded_output_count"] = int(
            normalized.get("total_autonomous_decoded_output_count", 0) or 0
        ) + 1
        normalized["last_autonomous_decoded_output_at"] = decoded_output_event[
            "decoded_at"
        ]
        ledger._store_state(normalized)  # noqa: SLF001

    normalized = _benchmark_bounded_normalized_state(ledger)
    decoded_output_review_source = dict(
        normalized.get("_normalization_source_window") or {}
    )
    decoded_output_review_match = any(
        str(item.get("autonomous_decoded_output_event_hash") or "")
        == decoded_output_hash
        for item in list(normalized["autonomous_decoded_output_events"])
    )

    text_emission_event = _autonomous_bounded_text_emission_event()
    text_emission_hash = text_emission_event[
        "autonomous_bounded_text_emission_event_hash"
    ]
    normalized = _benchmark_bounded_normalized_state(ledger)
    text_emission_append_source = dict(
        normalized.get("_normalization_source_window") or {}
    )
    text_emission_events = normalized["autonomous_bounded_text_emission_events"]
    text_emission_duplicate = text_emission_hash in {
        str(item.get("autonomous_bounded_text_emission_event_hash") or "")
        for item in list(text_emission_events)
    }
    if not text_emission_duplicate:
        text_emission_events.appendleft(deepcopy(text_emission_event))
        normalized["total_autonomous_bounded_text_emission_count"] = int(
            normalized.get("total_autonomous_bounded_text_emission_count", 0) or 0
        ) + 1
        normalized["last_autonomous_bounded_text_emitted_at"] = text_emission_event[
            "emitted_at"
        ]
        ledger._store_state(normalized)  # noqa: SLF001

    normalized = _benchmark_bounded_normalized_state(ledger)
    text_emission_review_source = dict(
        normalized.get("_normalization_source_window") or {}
    )
    text_emission_review_match = any(
        str(item.get("autonomous_bounded_text_emission_event_hash") or "")
        == text_emission_hash
        for item in list(normalized["autonomous_bounded_text_emission_events"])
    )

    text_commit_event = _autonomous_text_surface_commit_event()
    text_commit_hash = text_commit_event["autonomous_text_surface_commit_event_hash"]
    normalized = _benchmark_bounded_normalized_state(ledger)
    text_commit_append_source = dict(
        normalized.get("_normalization_source_window") or {}
    )
    text_commit_events = normalized["autonomous_text_surface_commit_events"]
    text_commit_duplicate = text_commit_hash in {
        str(item.get("autonomous_text_surface_commit_event_hash") or "")
        for item in list(text_commit_events)
    }
    if not text_commit_duplicate:
        text_commit_events.appendleft(deepcopy(text_commit_event))
        normalized["total_autonomous_text_surface_commit_count"] = int(
            normalized.get("total_autonomous_text_surface_commit_count", 0) or 0
        ) + 1
        normalized["last_autonomous_text_surface_committed_at"] = text_commit_event[
            "committed_at"
        ]
        normalized["current_text_surface_commit"] = deepcopy(text_commit_event)
        ledger._store_state(normalized)  # noqa: SLF001

    normalized = _benchmark_bounded_normalized_state(ledger)
    text_commit_review_source = dict(
        normalized.get("_normalization_source_window") or {}
    )
    text_commit_review_match = any(
        str(item.get("autonomous_text_surface_commit_event_hash") or "")
        == text_commit_hash
        for item in list(normalized["autonomous_text_surface_commit_events"])
    )
    text_commit_current_match = (
        dict(normalized.get("current_text_surface_commit") or {}) == text_commit_event
    )

    materialization_event = _autonomous_text_surface_materialization_event()
    materialization_hash = materialization_event[
        "autonomous_text_surface_materialization_event_hash"
    ]
    normalized = _benchmark_bounded_normalized_state(ledger)
    materialization_append_source = dict(
        normalized.get("_normalization_source_window") or {}
    )
    materialization_events = normalized[
        "autonomous_text_surface_materialization_events"
    ]
    materialization_duplicate = materialization_hash in {
        str(item.get("autonomous_text_surface_materialization_event_hash") or "")
        for item in list(materialization_events)
    }
    if not materialization_duplicate:
        materialization_events.appendleft(deepcopy(materialization_event))
        normalized["total_autonomous_text_surface_materialization_count"] = int(
            normalized.get(
                "total_autonomous_text_surface_materialization_count",
                0,
            )
            or 0
        ) + 1
        normalized["last_autonomous_text_surface_materialized_at"] = (
            materialization_event["materialized_at"]
        )
        normalized["current_text_surface_materialization"] = deepcopy(
            materialization_event
        )
        ledger._store_state(normalized)  # noqa: SLF001

    normalized = _benchmark_bounded_normalized_state(ledger)
    materialization_review_source = dict(
        normalized.get("_normalization_source_window") or {}
    )
    materialization_review_match = any(
        str(item.get("autonomous_text_surface_materialization_event_hash") or "")
        == materialization_hash
        for item in list(
            normalized["autonomous_text_surface_materialization_events"]
        )
    )
    materialization_current_match = (
        dict(normalized.get("current_text_surface_materialization") or {})
        == materialization_event
    )

    language_surface_commit_event = (
        _autonomous_bounded_language_surface_commit_event()
    )
    language_surface_commit_hash = language_surface_commit_event[
        "autonomous_bounded_language_surface_commit_event_hash"
    ]
    normalized = _benchmark_bounded_normalized_state(ledger)
    language_surface_commit_append_source = dict(
        normalized.get("_normalization_source_window") or {}
    )
    language_surface_commit_events = normalized[
        "autonomous_bounded_language_surface_commit_events"
    ]
    language_surface_commit_duplicate = language_surface_commit_hash in {
        str(item.get("autonomous_bounded_language_surface_commit_event_hash") or "")
        for item in list(language_surface_commit_events)
    }
    if not language_surface_commit_duplicate:
        language_surface_commit_events.appendleft(
            deepcopy(language_surface_commit_event)
        )
        normalized[
            "total_autonomous_bounded_language_surface_commit_count"
        ] = int(
            normalized.get(
                "total_autonomous_bounded_language_surface_commit_count",
                0,
            )
            or 0
        ) + 1
        normalized["last_autonomous_bounded_language_surface_committed_at"] = (
            language_surface_commit_event["committed_at"]
        )
        normalized["current_bounded_language_surface_commit"] = deepcopy(
            language_surface_commit_event
        )
        ledger._store_state(normalized)  # noqa: SLF001

    normalized = _benchmark_bounded_normalized_state(ledger)
    language_surface_commit_review_source = dict(
        normalized.get("_normalization_source_window") or {}
    )
    language_surface_commit_review_match = any(
        str(item.get("autonomous_bounded_language_surface_commit_event_hash") or "")
        == language_surface_commit_hash
        for item in list(
            normalized["autonomous_bounded_language_surface_commit_events"]
        )
    )
    language_surface_commit_current_match = (
        dict(normalized.get("current_bounded_language_surface_commit") or {})
        == language_surface_commit_event
    )

    language_surface_use_event = _autonomous_bounded_language_surface_use_event()
    language_surface_use_hash = language_surface_use_event[
        "autonomous_bounded_language_surface_use_event_hash"
    ]
    normalized = _benchmark_bounded_normalized_state(ledger)
    language_surface_use_append_source = dict(
        normalized.get("_normalization_source_window") or {}
    )
    language_surface_use_events = normalized[
        "autonomous_bounded_language_surface_use_events"
    ]
    language_surface_use_duplicate = language_surface_use_hash in {
        str(item.get("autonomous_bounded_language_surface_use_event_hash") or "")
        for item in list(language_surface_use_events)
    }
    if not language_surface_use_duplicate:
        language_surface_use_events.appendleft(deepcopy(language_surface_use_event))
        normalized["total_autonomous_bounded_language_surface_use_count"] = int(
            normalized.get(
                "total_autonomous_bounded_language_surface_use_count",
                0,
            )
            or 0
        ) + 1
        normalized["last_autonomous_bounded_language_surface_used_at"] = (
            language_surface_use_event["used_at"]
        )
        ledger._store_state(normalized)  # noqa: SLF001

    normalized = _benchmark_bounded_normalized_state(ledger)
    language_surface_use_review_source = dict(
        normalized.get("_normalization_source_window") or {}
    )
    language_surface_use_review_match = any(
        str(item.get("autonomous_bounded_language_surface_use_event_hash") or "")
        == language_surface_use_hash
        for item in list(normalized["autonomous_bounded_language_surface_use_events"])
    )

    language_generation_event = _autonomous_snn_language_generation_event()
    language_generation_hash = language_generation_event[
        "autonomous_snn_language_generation_event_hash"
    ]
    normalized = _benchmark_bounded_normalized_state(ledger)
    language_generation_append_source = dict(
        normalized.get("_normalization_source_window") or {}
    )
    language_generation_events = normalized[
        "autonomous_snn_language_generation_events"
    ]
    language_generation_duplicate = language_generation_hash in {
        str(item.get("autonomous_snn_language_generation_event_hash") or "")
        for item in list(language_generation_events)
    }
    if not language_generation_duplicate:
        language_generation_events.appendleft(deepcopy(language_generation_event))
        normalized["total_autonomous_snn_language_generation_count"] = int(
            normalized.get(
                "total_autonomous_snn_language_generation_count",
                0,
            )
            or 0
        ) + 1
        normalized["last_autonomous_snn_language_generated_at"] = (
            language_generation_event["generated_at"]
        )
        ledger._store_state(normalized)  # noqa: SLF001

    normalized = _benchmark_bounded_normalized_state(ledger)
    language_generation_review_source = dict(
        normalized.get("_normalization_source_window") or {}
    )
    language_generation_review_match = any(
        str(item.get("autonomous_snn_language_generation_event_hash") or "")
        == language_generation_hash
        for item in list(normalized["autonomous_snn_language_generation_events"])
    )
    thought_chain_results = {}
    for spec in _snn_language_thought_family_specs():
        event = dict(spec["event"])
        thought_chain_results[str(spec["name"])] = (
            _broad_normalized_append_review_record_family(
                ledger,
                field=str(spec["field"]),
                event=event,
                duplicate_key=str(spec["duplicate_key"]),
                total_count_key=str(spec["total_count_key"]),
                timestamp_key=str(spec["timestamp_key"]),
                timestamp_value=str(event[str(spec["timestamp_value_key"])]),
            )
        )

    return {
        "binding_duplicate": binding_duplicate,
        "observation_duplicate": observation_duplicate,
        "training_duplicate": training_duplicate,
        "decoder_duplicate": decoder_duplicate,
        "language_output_duplicate": language_output_duplicate,
        "decoded_output_duplicate": decoded_output_duplicate,
        "text_emission_duplicate": text_emission_duplicate,
        "text_commit_duplicate": text_commit_duplicate,
        "materialization_duplicate": materialization_duplicate,
        "language_surface_commit_duplicate": language_surface_commit_duplicate,
        "language_surface_use_duplicate": language_surface_use_duplicate,
        "language_generation_duplicate": language_generation_duplicate,
        **{
            f"{name}_duplicate": bool(result.get("duplicate"))
            for name, result in thought_chain_results.items()
        },
        "binding_hash": binding_hash,
        "observation_hash": observation_hash,
        "training_hash": training_hash,
        "decoder_hash": decoder_hash,
        "language_output_hash": language_output_hash,
        "decoded_output_hash": decoded_output_hash,
        "text_emission_hash": text_emission_hash,
        "text_commit_hash": text_commit_hash,
        "materialization_hash": materialization_hash,
        "language_surface_commit_hash": language_surface_commit_hash,
        "language_surface_use_hash": language_surface_use_hash,
        "language_generation_hash": language_generation_hash,
        **{
            f"{name}_hash": str(result.get("hash") or "")
            for name, result in thought_chain_results.items()
        },
        "binding_review_match": binding_review_match,
        "observation_review_match": observation_review_match,
        "training_review_match": training_review_match,
        "decoder_review_match": decoder_review_match,
        "language_output_review_match": language_output_review_match,
        "decoded_output_review_match": decoded_output_review_match,
        "text_emission_review_match": text_emission_review_match,
        "text_commit_review_match": text_commit_review_match,
        "materialization_review_match": materialization_review_match,
        "language_surface_commit_review_match": (
            language_surface_commit_review_match
        ),
        "language_surface_use_review_match": language_surface_use_review_match,
        "language_generation_review_match": language_generation_review_match,
        **{
            f"{name}_review_match": bool(result.get("review_match"))
            for name, result in thought_chain_results.items()
        },
        "text_commit_current_match": text_commit_current_match,
        "materialization_current_match": materialization_current_match,
        "language_surface_commit_current_match": (
            language_surface_commit_current_match
        ),
        "binding_total_count": int(
            ledger._ledger_state().get(  # noqa: SLF001
                "total_autonomous_hash_readout_binding_count",
                0,
            )
            or 0
        ),
        "observation_total_count": int(
            ledger._ledger_state().get(  # noqa: SLF001
                "total_autonomous_bound_readout_observation_count",
                0,
            )
            or 0
        ),
        "training_total_count": int(
            ledger._ledger_state().get(  # noqa: SLF001
                "total_autonomous_readout_training_window_count",
                0,
            )
            or 0
        ),
        "decoder_total_count": int(
            ledger._ledger_state().get(  # noqa: SLF001
                "total_autonomous_decoder_probe_count",
                0,
            )
            or 0
        ),
        "language_output_total_count": int(
            ledger._ledger_state().get(  # noqa: SLF001
                "total_autonomous_language_output_count",
                0,
            )
            or 0
        ),
        "decoded_output_total_count": int(
            ledger._ledger_state().get(  # noqa: SLF001
                "total_autonomous_decoded_output_count",
                0,
            )
            or 0
        ),
        "text_emission_total_count": int(
            ledger._ledger_state().get(  # noqa: SLF001
                "total_autonomous_bounded_text_emission_count",
                0,
            )
            or 0
        ),
        "text_commit_total_count": int(
            ledger._ledger_state().get(  # noqa: SLF001
                "total_autonomous_text_surface_commit_count",
                0,
            )
            or 0
        ),
        "materialization_total_count": int(
            ledger._ledger_state().get(  # noqa: SLF001
                "total_autonomous_text_surface_materialization_count",
                0,
            )
            or 0
        ),
        "language_surface_commit_total_count": int(
            ledger._ledger_state().get(  # noqa: SLF001
                "total_autonomous_bounded_language_surface_commit_count",
                0,
            )
            or 0
        ),
        "language_surface_use_total_count": int(
            ledger._ledger_state().get(  # noqa: SLF001
                "total_autonomous_bounded_language_surface_use_count",
                0,
            )
            or 0
        ),
        "language_generation_total_count": int(
            ledger._ledger_state().get(  # noqa: SLF001
                "total_autonomous_snn_language_generation_count",
                0,
            )
            or 0
        ),
        **{
            f"{name}_total_count": int(result.get("total_count", 0) or 0)
            for name, result in thought_chain_results.items()
        },
        "normalization_source_windows": {
            "binding_append": binding_append_source,
            "binding_review": binding_review_source,
            "observation_append": observation_append_source,
            "observation_review": observation_review_source,
            "training_append": training_append_source,
            "training_review": training_review_source,
            "decoder_append": decoder_append_source,
            "decoder_review": decoder_review_source,
            "language_output_append": language_output_append_source,
            "language_output_review": language_output_review_source,
            "decoded_output_append": decoded_output_append_source,
            "decoded_output_review": decoded_output_review_source,
            "text_emission_append": text_emission_append_source,
            "text_emission_review": text_emission_review_source,
            "text_commit_append": text_commit_append_source,
            "text_commit_review": text_commit_review_source,
            "materialization_append": materialization_append_source,
            "materialization_review": materialization_review_source,
            "language_surface_commit_append": (
                language_surface_commit_append_source
            ),
            "language_surface_commit_review": (
                language_surface_commit_review_source
            ),
            "language_surface_use_append": language_surface_use_append_source,
            "language_surface_use_review": language_surface_use_review_source,
            "language_generation_append": language_generation_append_source,
            "language_generation_review": language_generation_review_source,
            **{
                f"{name}_append": result["append_window"]
                for name, result in thought_chain_results.items()
            },
            **{
                f"{name}_review": result["review_window"]
                for name, result in thought_chain_results.items()
            },
        },
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


def _sum_source_window_counts(
    windows: Mapping[str, Mapping[str, Any]],
    *,
    key: str,
) -> int:
    return int(
        sum(int(dict(window).get(key, 0) or 0) for window in windows.values())
    )


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
        fn=lambda: _benchmark_bounded_normalized_state(ledger),
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
    event_map, event_map_samples = _timed_runs(
        runs=runs,
        fn=lambda: _bounded_readout_evidence_event_map_lookup(ledger),
    )
    broad_event_map, broad_event_map_samples = _timed_runs(
        runs=runs,
        fn=lambda: _broad_normalized_readout_evidence_event_map_lookup(ledger),
    )
    emission_history, emission_history_samples = _timed_runs(
        runs=runs,
        fn=lambda: _bounded_emission_review_history(
            ledger,
            limit=ledger_limit,
        ),
    )
    broad_emission_history, broad_emission_history_samples = _timed_runs(
        runs=runs,
        fn=lambda: _broad_normalized_emission_review_history(
            ledger,
            limit=ledger_limit,
        ),
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
    record_seed_state = _seed_ledger_state(retention_count=retention_count)
    record_state: dict[str, Any] = {}
    broad_record_state: dict[str, Any] = {}
    record_ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: record_state,
        limit=ledger_limit,
    )
    broad_record_ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: broad_record_state,
        limit=ledger_limit,
    )
    record_append, record_append_samples = _timed_runs(
        runs=runs,
        setup=lambda: _reset_state(record_state, record_seed_state),
        fn=lambda: _bounded_record_family_append(record_ledger),
    )
    broad_record_append, broad_record_append_samples = _timed_runs(
        runs=runs,
        setup=lambda: _reset_state(broad_record_state, record_seed_state),
        fn=lambda: _broad_normalized_record_family_append(broad_record_ledger),
    )
    autonomous_chain_seed_state = _seed_ledger_state(retention_count=retention_count)
    autonomous_chain_state: dict[str, Any] = {}
    broad_autonomous_chain_state: dict[str, Any] = {}
    autonomous_chain_ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: autonomous_chain_state,
        limit=ledger_limit,
    )
    broad_autonomous_chain_ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: broad_autonomous_chain_state,
        limit=ledger_limit,
    )
    autonomous_chain, autonomous_chain_samples = _timed_runs(
        runs=runs,
        setup=lambda: _reset_state(
            autonomous_chain_state,
            autonomous_chain_seed_state,
        ),
        fn=lambda: _bounded_autonomous_readout_event_family_chain(
            autonomous_chain_ledger
        ),
    )
    broad_autonomous_chain, broad_autonomous_chain_samples = _timed_runs(
        runs=runs,
        setup=lambda: _reset_state(
            broad_autonomous_chain_state,
            autonomous_chain_seed_state,
        ),
        fn=lambda: _broad_normalized_autonomous_readout_event_family_chain(
            broad_autonomous_chain_ledger
        ),
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
    event_map_report = dict(event_map.get("report") or {})
    event_map_mean = statistics.fmean(event_map_samples)
    broad_event_map_mean = statistics.fmean(broad_event_map_samples)
    event_map_rows = int(event_map_report.get("source_window_count", 0) or 0)
    broad_event_map_rows = int(
        dict(
            broad_event_map.get("normalization_source_window") or {}
        ).get("source_window_count_total", 0)
        or 0
    )
    emission_history_source_window = dict(
        emission_history.get("source_window") or {}
    )
    emission_history_mean = statistics.fmean(emission_history_samples)
    broad_emission_history_mean = statistics.fmean(
        broad_emission_history_samples
    )
    emission_history_rows = int(
        emission_history_source_window.get("source_window_count", 0) or 0
    )
    broad_emission_history_rows = int(
        dict(
            broad_emission_history.get("normalization_source_window") or {}
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
    record_append_source_window = dict(record_append.get("source_window") or {})
    record_append_mean = statistics.fmean(record_append_samples)
    broad_record_append_mean = statistics.fmean(broad_record_append_samples)
    record_append_rows = int(
        record_append_source_window.get("source_window_count", 0) or 0
    )
    broad_record_append_rows = int(
        dict(
            broad_record_append.get("normalization_source_window") or {}
        ).get("source_window_count_total", 0)
        or 0
    )
    autonomous_chain_source_windows = {
        str(key): dict(value)
        for key, value in dict(autonomous_chain.get("source_windows") or {}).items()
        if isinstance(value, Mapping)
    }
    broad_autonomous_chain_source_windows = {
        str(key): dict(value)
        for key, value in dict(
            broad_autonomous_chain.get("normalization_source_windows") or {}
        ).items()
        if isinstance(value, Mapping)
    }
    autonomous_chain_mean = statistics.fmean(autonomous_chain_samples)
    broad_autonomous_chain_mean = statistics.fmean(broad_autonomous_chain_samples)
    autonomous_chain_rows = _sum_source_window_counts(
        autonomous_chain_source_windows,
        key="source_window_count",
    )
    broad_autonomous_chain_rows = _sum_source_window_counts(
        broad_autonomous_chain_source_windows,
        key="source_window_count_total",
    )
    pass_checks = {
        "surface_present": source_window.get("surface")
        == "bounded_snn_readout_ledger_normalization_source_window.v1",
        "policy_present": source_window.get("policy")
        == SNN_BENCHMARK_READOUT_LEDGER_NORMALIZATION_SOURCE_WINDOW_POLICY,
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
            len(list(bounded_store.get(field) or [])) <= ledger_limit
            for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
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
        "event_map_surface_present": event_map_report.get("surface")
        == "bounded_snn_readout_evidence_event_map_source_window.v1",
        "event_map_policy_present": event_map_report.get("policy")
        == SNN_READOUT_EVIDENCE_HASH_SOURCE_WINDOW_POLICY,
        "event_map_hash_parity": event_map.get("hashes")
        == broad_event_map.get("hashes"),
        "event_map_requested_hash_count": int(
            event_map_report.get("requested_hash_count", 0) or 0
        )
        == len(_readout_evidence_event_map_target_hashes()),
        "event_map_bounded_less_work": event_map_rows < broad_event_map_rows,
        "event_map_latency_not_slower_than_broad_normalization": (
            event_map_mean <= broad_event_map_mean * 1.1
        ),
        "emission_history_surface_present": emission_history_source_window.get(
            "surface"
        )
        == "bounded_snn_emission_review_history_source_window.v1",
        "emission_history_policy_present": emission_history_source_window.get(
            "policy"
        )
        == SNN_EMISSION_REVIEW_HISTORY_SOURCE_WINDOW_POLICY,
        "emission_history_review_hash_parity": (
            emission_history.get("review_hashes")
            == broad_emission_history.get("review_hashes")
        ),
        "emission_history_text_hash_parity": (
            emission_history.get("text_hashes")
            == broad_emission_history.get("text_hashes")
        ),
        "emission_history_bounded_less_work": (
            emission_history_rows < broad_emission_history_rows
        ),
        "emission_history_latency_not_slower_than_broad_normalization": (
            emission_history_mean <= broad_emission_history_mean * 1.1
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
        "record_append_surface_present": (
            record_append_source_window.get("surface")
            == "bounded_snn_readout_ledger_record_family_source_window.v1"
        ),
        "record_append_policy_present": (
            record_append_source_window.get("policy")
            == SNN_READOUT_LEDGER_RECORD_FAMILY_SOURCE_WINDOW_POLICY
        ),
        "record_append_latest_hash_parity": (
            record_append.get("latest_hash") == broad_record_append.get("latest_hash")
        ),
        "record_append_total_count_parity": (
            record_append.get("total_recorded_count")
            == broad_record_append.get("total_recorded_count")
        ),
        "record_append_bounded_less_work": (
            record_append_rows < broad_record_append_rows
        ),
        "record_append_latency_not_slower_than_broad_normalization": (
            record_append_mean <= broad_record_append_mean * 1.1
        ),
        "autonomous_chain_surface_present": all(
            dict(window).get("surface")
            == "bounded_snn_readout_ledger_record_family_source_window.v1"
            for window in autonomous_chain_source_windows.values()
        ),
        "autonomous_chain_policy_present": all(
            dict(window).get("policy")
            == SNN_READOUT_LEDGER_RECORD_FAMILY_SOURCE_WINDOW_POLICY
            for window in autonomous_chain_source_windows.values()
        ),
        "autonomous_chain_hash_parity": all(
            autonomous_chain.get(f"{name}_hash")
            == broad_autonomous_chain.get(f"{name}_hash")
            for name in _AUTONOMOUS_CHAIN_COMPONENT_NAMES
        ),
        "autonomous_chain_review_match_parity": all(
            autonomous_chain.get(f"{name}_review_match")
            == broad_autonomous_chain.get(f"{name}_review_match")
            for name in _AUTONOMOUS_CHAIN_COMPONENT_NAMES
        ),
        "autonomous_chain_total_count_parity": all(
            autonomous_chain.get(f"{name}_total_count")
            == broad_autonomous_chain.get(f"{name}_total_count")
            for name in _AUTONOMOUS_CHAIN_COMPONENT_NAMES
        ),
        "autonomous_chain_current_commit_parity": (
            bool(autonomous_chain.get("text_commit_current_match"))
            and bool(broad_autonomous_chain.get("text_commit_current_match"))
            and bool(autonomous_chain.get("materialization_current_match"))
            and bool(broad_autonomous_chain.get("materialization_current_match"))
            and bool(autonomous_chain.get("language_surface_commit_current_match"))
            and bool(
                broad_autonomous_chain.get(
                    "language_surface_commit_current_match"
                )
            )
        ),
        "autonomous_chain_bounded_less_work": (
            autonomous_chain_rows < broad_autonomous_chain_rows
        ),
        "autonomous_chain_latency_not_slower_than_broad_normalization": (
            autonomous_chain_mean <= broad_autonomous_chain_mean * 1.1
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
            "policy": SNN_BENCHMARK_READOUT_LEDGER_NORMALIZATION_SOURCE_WINDOW_POLICY,
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
        "readout_evidence_event_map_boundary": {
            "surface": "bounded_snn_readout_evidence_event_map_source_window.v1",
            "policy": SNN_READOUT_EVIDENCE_HASH_SOURCE_WINDOW_POLICY,
            "source": "snn_readout_ledger.events",
            "selection_criteria": [
                "requested_readout_evidence_hashes_only",
                "bounded source window before synapse provenance audit",
            ],
            "quality": {
                "metric": "requested_readout_evidence_event_map_hash_parity",
                "hash_set_parity": event_map.get("hashes")
                == broad_event_map.get("hashes"),
                "requested_hash_count": int(
                    event_map_report.get("requested_hash_count", 0) or 0
                ),
                "matched_hash_count": int(
                    event_map_report.get("matched_hash_count", 0) or 0
                ),
                "missing_hash_count": int(
                    event_map_report.get("missing_hash_count", 0) or 0
                ),
                "event_count": int(event_map.get("event_count", 0) or 0),
            },
            "latency": {
                "bounded": _latency_summary(event_map_samples),
                "broad_normalized": _latency_summary(broad_event_map_samples),
                "bounded_speedup_vs_broad_normalized": round(
                    broad_event_map_mean / max(event_map_mean, 1e-9),
                    6,
                ),
            },
            "retired_path_comparison": {
                "old_policy": (
                    "normalize_all_ledger_event_fields_before_synapse"
                    "_provenance_readout_evidence_event_map"
                ),
                "bounded_checked_record_count": event_map_rows,
                "old_checked_record_count": broad_event_map_rows,
                "record_work_reduction": round(
                    broad_event_map_rows / max(1, event_map_rows),
                    6,
                ),
            },
            "source_window": event_map_report,
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
            "gpu_used": False,
        },
        "emission_review_history_boundary": {
            "surface": "bounded_snn_emission_review_history_source_window.v1",
            "policy": SNN_EMISSION_REVIEW_HISTORY_SOURCE_WINDOW_POLICY,
            "source": "snn_readout_ledger.emission_review_events",
            "selection_criteria": [
                "operator_reviewed_bounded_snn_emissions_only",
                "bounded source window before display history",
            ],
            "quality": {
                "metric": "emission_review_history_hash_and_text_hash_parity",
                "review_hash_parity": emission_history.get("review_hashes")
                == broad_emission_history.get("review_hashes"),
                "text_hash_parity": emission_history.get("text_hashes")
                == broad_emission_history.get("text_hashes"),
                "returned_review_count": int(
                    len(emission_history.get("review_hashes") or [])
                ),
            },
            "latency": {
                "bounded": _latency_summary(emission_history_samples),
                "broad_normalized": _latency_summary(
                    broad_emission_history_samples
                ),
                "bounded_speedup_vs_broad_normalized": round(
                    broad_emission_history_mean
                    / max(emission_history_mean, 1e-9),
                    6,
                ),
            },
            "retired_path_comparison": {
                "old_policy": (
                    "normalize_all_ledger_event_fields_before_emission"
                    "_review_display_history"
                ),
                "bounded_checked_record_count": emission_history_rows,
                "old_checked_record_count": broad_emission_history_rows,
                "record_work_reduction": round(
                    broad_emission_history_rows
                    / max(1, emission_history_rows),
                    6,
                ),
            },
            "source_window": emission_history_source_window,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_text_payload_loaded": True,
            "language_reasoning": False,
            "runs_live_tick": False,
            "runs_every_token": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "lookup_device": "cpu",
            "gpu_used": False,
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
        "record_family_append_boundary": {
            "surface": "bounded_snn_readout_ledger_record_family_source_window.v1",
            "policy": SNN_READOUT_LEDGER_RECORD_FAMILY_SOURCE_WINDOW_POLICY,
            "source": "snn_readout_ledger.events",
            "selection_criteria": [
                "single_record_family_only",
                "bounded_source_window_before_duplicate_check",
            ],
            "quality": {
                "metric": "record_family_append_latest_hash_and_total_count_parity",
                "latest_hash_parity": record_append.get("latest_hash")
                == broad_record_append.get("latest_hash"),
                "total_count_parity": record_append.get("total_recorded_count")
                == broad_record_append.get("total_recorded_count"),
                "latest_hash": record_append.get("latest_hash"),
                "total_recorded_count": int(
                    record_append.get("total_recorded_count", 0) or 0
                ),
                "window_event_count": int(record_append.get("event_count", 0) or 0),
            },
            "latency": {
                "bounded": _latency_summary(record_append_samples),
                "broad_normalized": _latency_summary(broad_record_append_samples),
                "bounded_speedup_vs_broad_normalized": round(
                    broad_record_append_mean / max(record_append_mean, 1e-9),
                    6,
                ),
            },
            "retired_path_comparison": {
                "old_policy": (
                    "normalize_all_ledger_event_fields_before_single_family"
                    "_record_append"
                ),
                "bounded_checked_record_count": record_append_rows,
                "old_checked_record_count": broad_record_append_rows,
                "record_work_reduction": round(
                    broad_record_append_rows / max(1, record_append_rows),
                    6,
                ),
            },
            "source_window": record_append_source_window,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "runs_live_tick": False,
            "runs_every_token": False,
            "mutates_runtime_state": True,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "lookup_device": "cpu",
            "write_device": "cpu",
            "gpu_used": False,
        },
        "autonomous_hash_readout_event_family_chain_boundary": {
            "surface": "bounded_snn_autonomous_hash_readout_event_family_chain_source_window.v1",
            "policy": SNN_READOUT_LEDGER_RECORD_FAMILY_SOURCE_WINDOW_POLICY,
            "source": (
                "snn_readout_ledger.autonomous_hash_readout_binding_events"
                "+autonomous_bound_readout_observation_events"
                "+autonomous_readout_training_window_events"
                "+autonomous_decoder_probe_events"
                "+autonomous_language_output_events"
                "+autonomous_decoded_output_events"
                "+autonomous_bounded_text_emission_events"
                "+autonomous_text_surface_commit_events"
                "+autonomous_text_surface_materialization_events"
                "+autonomous_bounded_language_surface_commit_events"
                "+autonomous_bounded_language_surface_use_events"
                "+autonomous_snn_language_generation_events"
                "+autonomous_snn_language_decoding_events"
                "+snn_language_readout_surface_events"
                "+snn_language_readout_memory_events"
                "+autonomous_snn_language_thought_consolidation_events"
                "+autonomous_snn_language_thought_structural_plasticity_events"
            ),
            "selection_criteria": [
                (
                    "binding_observation_training_decoder_output_text_commit_"
                    "materialization_language_surface_generation_thought_"
                    "event_families_only"
                ),
                "bounded_source_window_before_duplicate_or_review_lookup",
            ],
            "quality": {
                "metric": (
                    "autonomous_hash_readout_thought_structural_chain_hash_count"
                    "_review_current_pointer_parity"
                ),
                "component_hash_parity": {
                    name: autonomous_chain.get(f"{name}_hash")
                    == broad_autonomous_chain.get(f"{name}_hash")
                    for name in _AUTONOMOUS_CHAIN_COMPONENT_NAMES
                },
                "component_review_match_parity": {
                    name: autonomous_chain.get(f"{name}_review_match")
                    == broad_autonomous_chain.get(f"{name}_review_match")
                    for name in _AUTONOMOUS_CHAIN_COMPONENT_NAMES
                },
                "component_total_count_parity": {
                    name: autonomous_chain.get(f"{name}_total_count")
                    == broad_autonomous_chain.get(f"{name}_total_count")
                    for name in _AUTONOMOUS_CHAIN_COMPONENT_NAMES
                },
                "all_component_hash_parity": all(
                    autonomous_chain.get(f"{name}_hash")
                    == broad_autonomous_chain.get(f"{name}_hash")
                    for name in _AUTONOMOUS_CHAIN_COMPONENT_NAMES
                ),
                "all_component_review_match_parity": all(
                    autonomous_chain.get(f"{name}_review_match")
                    == broad_autonomous_chain.get(f"{name}_review_match")
                    for name in _AUTONOMOUS_CHAIN_COMPONENT_NAMES
                ),
                "all_component_total_count_parity": all(
                    autonomous_chain.get(f"{name}_total_count")
                    == broad_autonomous_chain.get(f"{name}_total_count")
                    for name in _AUTONOMOUS_CHAIN_COMPONENT_NAMES
                ),
                "text_commit_current_parity": (
                    bool(autonomous_chain.get("text_commit_current_match"))
                    and bool(broad_autonomous_chain.get("text_commit_current_match"))
                ),
                "materialization_current_parity": (
                    bool(autonomous_chain.get("materialization_current_match"))
                    and bool(
                        broad_autonomous_chain.get(
                            "materialization_current_match"
                        )
                    )
                ),
                "language_surface_commit_current_parity": (
                    bool(
                        autonomous_chain.get(
                            "language_surface_commit_current_match"
                        )
                    )
                    and bool(
                        broad_autonomous_chain.get(
                            "language_surface_commit_current_match"
                        )
                    )
                ),
            },
            "latency": {
                "bounded": _latency_summary(autonomous_chain_samples),
                "broad_normalized": _latency_summary(
                    broad_autonomous_chain_samples
                ),
                "bounded_speedup_vs_broad_normalized": round(
                    broad_autonomous_chain_mean / max(autonomous_chain_mean, 1e-9),
                    6,
                ),
            },
            "retired_path_comparison": {
                "old_policy": (
                    "normalize_all_ledger_event_fields_before_autonomous_hash"
                    "_readout_thought_structural_chain_append_or_review"
                ),
                "bounded_checked_record_count": autonomous_chain_rows,
                "old_checked_record_count": broad_autonomous_chain_rows,
                "record_work_reduction": round(
                    broad_autonomous_chain_rows / max(1, autonomous_chain_rows),
                    6,
                ),
            },
            "source_windows": autonomous_chain_source_windows,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "runs_live_tick": False,
            "runs_every_token": False,
            "mutates_runtime_state": True,
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
        "known_hash_broad_mean_ms={known_hash_broad:.6f} event_map_bounded_mean_ms={event_map_bounded:.6f} "
        "event_map_broad_mean_ms={event_map_broad:.6f} emission_history_bounded_mean_ms={emission_history_bounded:.6f} "
        "emission_history_broad_mean_ms={emission_history_broad:.6f} dense_label_bounded_mean_ms={dense_label_bounded:.6f} "
        "dense_label_broad_mean_ms={dense_label_broad:.6f} dense_label_eval_bounded_mean_ms={dense_label_eval_bounded:.6f} "
        "dense_label_eval_broad_mean_ms={dense_label_eval_broad:.6f} "
        "dense_label_update_bounded_mean_ms={dense_label_update_bounded:.6f} "
        "dense_label_update_broad_mean_ms={dense_label_update_broad:.6f} "
        "autonomous_confidence_use_bounded_mean_ms={confidence_use_bounded:.6f} "
        "autonomous_confidence_use_broad_mean_ms={confidence_use_broad:.6f} "
        "record_append_bounded_mean_ms={record_append_bounded:.6f} "
        "record_append_broad_mean_ms={record_append_broad:.6f} "
        "autonomous_chain_bounded_mean_ms={autonomous_chain_bounded:.6f} "
        "autonomous_chain_broad_mean_ms={autonomous_chain_broad:.6f}".format(
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
            event_map_bounded=report["readout_evidence_event_map_boundary"][
                "latency"
            ]["bounded"]["mean_ms"],
            event_map_broad=report["readout_evidence_event_map_boundary"][
                "latency"
            ]["broad_normalized"]["mean_ms"],
            emission_history_bounded=report["emission_review_history_boundary"][
                "latency"
            ]["bounded"]["mean_ms"],
            emission_history_broad=report["emission_review_history_boundary"][
                "latency"
            ]["broad_normalized"]["mean_ms"],
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
            record_append_bounded=report["record_family_append_boundary"][
                "latency"
            ]["bounded"]["mean_ms"],
            record_append_broad=report["record_family_append_boundary"][
                "latency"
            ]["broad_normalized"]["mean_ms"],
            autonomous_chain_bounded=report[
                "autonomous_hash_readout_event_family_chain_boundary"
            ]["latency"]["bounded"]["mean_ms"],
            autonomous_chain_broad=report[
                "autonomous_hash_readout_event_family_chain_boundary"
            ]["latency"]["broad_normalized"]["mean_ms"],
        )
    )


if __name__ == "__main__":
    main()
