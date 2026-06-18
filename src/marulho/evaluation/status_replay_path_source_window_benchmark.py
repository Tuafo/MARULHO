"""Benchmark bounded Runtime Truth replay-path status projections.

This keeps the production read model on bounded CPU source windows while the
benchmark-local diagnostic functions preserve the old retained-scan shape for
latency and work comparison only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from collections import deque
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

import torch

from marulho.config.model_config import MarulhoConfig
from marulho.service.runtime_state import RuntimeState
from marulho.service.status_read_model import (
    SNN_STATUS_REPLAY_PATH_SOURCE_WINDOW_LIMIT,
    StatusReadModel,
)
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


def _sha256_json(payload: Mapping[str, Any] | list[Any]) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _build_read_model(ledger_state: Mapping[str, Any]) -> StatusReadModel:
    config = MarulhoConfig(
        n_columns=4,
        column_latent_dim=8,
        bootstrap_tokens=0,
        memory_capacity=64,
        eta_competitive=0.05,
        eta_decay=0.0,
        input_weight_blend=0.0,
        enable_context_layer=True,
        enable_binding_layer=True,
        device="cpu",
    )
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    return StatusReadModel(
        lock=lock,
        runtime_state=runtime_state,
        trainer=trainer,
        trace_history=deque(maxlen=200),
        metadata={},
        checkpoint_path_str="benchmark-status.pt",
        trace_dir_str="benchmark-traces",
        concept_store_snapshot_fn=lambda: {"top_concepts": [], "total_concepts": 0},
        brain_runtime_snapshot_fn=lambda: {
            "configured": False,
            "running": False,
            "source_bank": [],
            "living_loop": {},
        },
        sensory_preview_history=deque(maxlen=8),
        architecture_snapshot_fn=lambda: {
            "model_name": "Terminus",
            "core_name": "GPCSN",
            "version": "current",
            "family": "subcortex_runtime",
            "layers": [],
            "config": {},
        },
        animation_snapshot_fn=lambda: {},
        readout_ledger_state_fn=lambda: ledger_state,
    )


def _seed_ledger_state(retention_count: int) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    rollouts: list[dict[str, Any]] = []
    for index in range(int(retention_count)):
        label = f"status-label-{index}"
        prediction_hash = f"prediction-{index}"
        evaluation_hash = f"evaluation-{index}"
        weights_hash = f"weights-{index}"
        readout_hash = f"readout-{index}"
        events.append(
            {
                "readout_evidence_hash": readout_hash,
                "readout_evidence_id": f"readout-id-{index}",
                "prediction_hash": prediction_hash,
                "transition_memory_evaluation_hash": evaluation_hash,
                "persistent_transition_weights_hash": weights_hash,
                "labels": [label],
                "label_grounding": [True],
            }
        )
        reviews.append(
            {
                "emission_review_hash": f"review-{index}",
                "emission_hash": f"emission-{index}",
                "prediction_hash": prediction_hash,
                "transition_memory_evaluation_hash": evaluation_hash,
                "persistent_transition_weights_hash": weights_hash,
                "reviewed_at": "2026-06-18T00:00:00+00:00",
                "text": f"diagnostic-only reviewed text {index}",
                "labels": [label],
            }
        )
        rollouts.append(
            {
                "rollout_evidence_hash": f"rollout-evidence-{index}",
                "rollout_hash": f"rollout-{index}",
                "persistent_transition_weights_hash": weights_hash,
                "recorded_at": "2026-06-18T00:00:00+00:00",
            }
        )
    return {
        "events": events,
        "emission_review_events": reviews,
        "rollout_events": rollouts,
        "total_recorded_count": len(events),
        "total_emission_review_count": len(reviews),
        "total_rollout_recorded_count": len(rollouts),
        "last_emission_reviewed_at": "2026-06-18T00:00:00+00:00",
        "last_rollout_recorded_at": "2026-06-18T00:00:00+00:00",
    }


def _legacy_emission_projection(state: Mapping[str, Any]) -> dict[str, Any]:
    readout_events = [
        dict(item) for item in list(state.get("events") or []) if isinstance(item, Mapping)
    ]
    review_events = [
        dict(item)
        for item in list(state.get("emission_review_events") or [])
        if isinstance(item, Mapping)
    ]
    readout_by_binding: dict[
        tuple[str, str, str, tuple[str, ...]],
        dict[str, Any],
    ] = {}
    for event in readout_events:
        labels = tuple(str(value) for value in list(event.get("labels") or []))
        key = (
            str(event.get("prediction_hash") or ""),
            str(event.get("transition_memory_evaluation_hash") or ""),
            str(event.get("persistent_transition_weights_hash") or ""),
            labels,
        )
        if all(key[:3]) and labels:
            readout_by_binding.setdefault(key, event)
    matched = []
    for review in review_events:
        labels = tuple(str(value) for value in list(review.get("labels") or []))
        key = (
            str(review.get("prediction_hash") or ""),
            str(review.get("transition_memory_evaluation_hash") or ""),
            str(review.get("persistent_transition_weights_hash") or ""),
            labels,
        )
        readout = readout_by_binding.get(key)
        if readout is None:
            continue
        grounding = [bool(value) for value in list(readout.get("label_grounding") or [])]
        matched.append(
            {
                "prediction_hash": review.get("prediction_hash"),
                "readout_evidence_hash": readout.get("readout_evidence_hash"),
                "grounded": bool(grounding) and all(grounding),
                "label_hash": _sha256_json(list(labels)),
            }
        )
    return {
        "policy_candidate_count": len(matched),
        "design_seed_candidate_count": sum(1 for item in matched if item["grounded"]),
        "latest_prediction_hash": matched[0].get("prediction_hash") if matched else None,
        "checked_record_count": len(readout_events) + len(review_events),
    }


def _legacy_rollout_projection(state: Mapping[str, Any]) -> dict[str, Any]:
    events = [
        dict(item) for item in list(state.get("events") or []) if isinstance(item, Mapping)
    ]
    rollout_events = [
        dict(item) for item in list(state.get("rollout_events") or []) if isinstance(item, Mapping)
    ]
    transition_hashes = {
        str(item.get("persistent_transition_weights_hash") or "")
        for item in rollout_events
        if str(item.get("persistent_transition_weights_hash") or "")
    }
    rollout_hashes = {
        str(item.get("rollout_hash") or "")
        for item in rollout_events
        if str(item.get("rollout_hash") or "")
    }
    latest = rollout_events[0] if rollout_events else {}
    return {
        "event_count": len(events),
        "rollout_event_count": len(rollout_events),
        "unique_rollout_count": len(rollout_hashes),
        "unique_transition_memory_count": len(transition_hashes),
        "latest_rollout_hash": latest.get("rollout_hash"),
        "checked_record_count": len(events) + len(rollout_events),
    }


def _legacy_emission_review_history(state: Mapping[str, Any]) -> dict[str, Any]:
    events = [
        dict(item)
        for item in list(state.get("emission_review_events") or [])
        if isinstance(item, Mapping)
    ]
    emission_hashes = {
        str(item.get("emission_hash") or "")
        for item in events
        if str(item.get("emission_hash") or "")
    }
    trajectory_hashes = {
        str(item.get("trajectory_hash") or "")
        for item in events
        if str(item.get("trajectory_hash") or "")
    }
    transition_hashes = {
        str(item.get("persistent_transition_weights_hash") or "")
        for item in events
        if str(item.get("persistent_transition_weights_hash") or "")
    }
    latest = events[0] if events else {}
    return {
        "emission_review_event_count": len(events),
        "unique_emission_count": len(emission_hashes),
        "unique_trajectory_count": len(trajectory_hashes),
        "unique_transition_memory_count": len(transition_hashes),
        "latest_emission_review_hash": latest.get("emission_review_hash"),
        "latest_emission_hash": latest.get("emission_hash"),
        "checked_record_count": len(events),
    }


def _measure(fn, runs: int) -> tuple[list[float], Any]:
    samples: list[float] = []
    last: Any = None
    for _ in range(runs):
        started = time.perf_counter()
        last = fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples, last


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    index_95 = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * 0.95))))
    return {
        "mean_ms": round(statistics.fmean(samples), 6),
        "median_ms": round(statistics.median(samples), 6),
        "p95_ms": round(ordered[index_95], 6),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    retention_count = max(
        SNN_STATUS_REPLAY_PATH_SOURCE_WINDOW_LIMIT,
        int(args.retention_count),
    )
    runs = max(1, int(args.runs))
    state = _seed_ledger_state(retention_count)
    model = _build_read_model(state)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    bounded_emission_samples, bounded_emission = _measure(
        model._snn_readout_emission_replay_design_path,  # noqa: SLF001
        runs,
    )
    bounded_rollout_samples, bounded_rollout = _measure(
        model._snn_readout_rollout_consolidation_path,  # noqa: SLF001
        runs,
    )
    bounded_history_samples, bounded_history = _measure(
        model._snn_readout_emission_review_history,  # noqa: SLF001
        runs,
    )
    legacy_emission_samples, legacy_emission = _measure(
        lambda: _legacy_emission_projection(state),
        runs,
    )
    legacy_rollout_samples, legacy_rollout = _measure(
        lambda: _legacy_rollout_projection(state),
        runs,
    )
    legacy_history_samples, legacy_history = _measure(
        lambda: _legacy_emission_review_history(state),
        runs,
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    bounded_emission_window = dict(bounded_emission.get("source_window") or {})
    bounded_rollout_window = dict(bounded_rollout.get("source_window") or {})
    bounded_history_window = dict(bounded_history.get("source_window") or {})
    pass_checks = {
        "emission_surface_present": bounded_emission_window.get("surface")
        == "bounded_snn_status_emission_replay_design_path_source_window.v1",
        "rollout_surface_present": bounded_rollout_window.get("surface")
        == "bounded_snn_status_rollout_consolidation_path_source_window.v1",
        "history_surface_present": bounded_history_window.get("surface")
        == "bounded_snn_status_emission_review_history_source_window.v1",
        "emission_source_limit_respected": int(
            bounded_emission_window.get("emission_review_event_window_count", 0) or 0
        )
        <= SNN_STATUS_REPLAY_PATH_SOURCE_WINDOW_LIMIT,
        "rollout_source_limit_respected": int(
            bounded_rollout_window.get("rollout_event_window_count", 0) or 0
        )
        <= SNN_STATUS_REPLAY_PATH_SOURCE_WINDOW_LIMIT,
        "history_source_limit_respected": int(
            bounded_history_window.get("emission_review_event_window_count", 0) or 0
        )
        <= SNN_STATUS_REPLAY_PATH_SOURCE_WINDOW_LIMIT,
        "emission_no_global_scan": bounded_emission_window.get("global_candidate_scan")
        is False
        and bounded_emission_window.get("global_score_scan") is False,
        "rollout_no_global_scan": bounded_rollout_window.get("global_candidate_scan")
        is False
        and bounded_rollout_window.get("global_score_scan") is False,
        "history_no_global_scan": bounded_history_window.get("global_candidate_scan")
        is False
        and bounded_history_window.get("global_score_scan") is False,
        "not_live_tick": bounded_emission_window.get("runs_live_tick") is False
        and bounded_rollout_window.get("runs_live_tick") is False,
        "history_not_live_tick": bounded_history_window.get("runs_live_tick") is False
        and bounded_history_window.get("runs_every_token") is False,
        "no_language_reasoning": bounded_emission_window.get("language_reasoning")
        is False
        and bounded_rollout_window.get("language_reasoning") is False,
        "history_no_language_reasoning": bounded_history_window.get(
            "language_reasoning"
        )
        is False
        and bounded_history_window.get("raw_text_payload_loaded") is False,
        "cpu_only": bounded_emission_window.get("archival_storage_device") == "cpu"
        and bounded_rollout_window.get("archival_storage_device") == "cpu"
        and bounded_history_window.get("archival_storage_device") == "cpu"
        and bounded_emission_window.get("gpu_used") is False
        and bounded_rollout_window.get("gpu_used") is False,
        "history_cpu_only": bounded_history_window.get("gpu_used") is False,
        "latest_emission_matches_legacy": bounded_emission.get("latest_prediction_hash")
        == legacy_emission.get("latest_prediction_hash"),
        "latest_history_matches_legacy": bounded_history.get(
            "latest_emission_review_hash"
        )
        == legacy_history.get("latest_emission_review_hash"),
        "latest_rollout_matches_legacy": bounded_rollout.get("latest_rollout_hash")
        == legacy_rollout.get("latest_rollout_hash"),
    }
    bounded_checked = int(
        bounded_emission_window.get("emission_review_event_window_count", 0) or 0
    ) + int(
        bounded_emission_window.get("internal_readout_event_window_count", 0) or 0
    ) + int(
        bounded_rollout_window.get("rollout_event_window_count", 0) or 0
    ) + int(
        bounded_rollout_window.get("internal_readout_event_window_count", 0) or 0
    ) + int(
        bounded_history_window.get("emission_review_event_window_count", 0) or 0
    )
    legacy_checked = int(legacy_emission["checked_record_count"]) + int(
        legacy_rollout["checked_record_count"]
    ) + int(
        legacy_history["checked_record_count"]
    )
    bounded_mean = statistics.fmean(
        bounded_emission_samples + bounded_rollout_samples + bounded_history_samples
    )
    legacy_mean = statistics.fmean(
        legacy_emission_samples + legacy_rollout_samples + legacy_history_samples
    )
    return {
        "surface": "bounded_snn_status_replay_path_source_window_benchmark.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": {
            "retention_count": retention_count,
            "requested_retention_count": int(args.retention_count),
            "runs": runs,
            "source_window_limit": SNN_STATUS_REPLAY_PATH_SOURCE_WINDOW_LIMIT,
        },
        "pass": all(pass_checks.values()),
        "pass_checks": pass_checks,
        "quality": {
            "latest_emission_matches_legacy": pass_checks["latest_emission_matches_legacy"],
            "latest_history_matches_legacy": pass_checks[
                "latest_history_matches_legacy"
            ],
            "latest_rollout_matches_legacy": pass_checks["latest_rollout_matches_legacy"],
            "bounded_emission_seed_count": bounded_emission.get("design_seed_candidate_count"),
            "legacy_emission_seed_count": legacy_emission.get("design_seed_candidate_count"),
            "bounded_history_unique_count": bounded_history.get("unique_emission_count"),
            "legacy_history_unique_count": legacy_history.get("unique_emission_count"),
            "history_unique_scope": bounded_history.get("unique_count_scope"),
            "bounded_rollout_unique_count": bounded_rollout.get("unique_rollout_count"),
            "legacy_rollout_unique_count": legacy_rollout.get("unique_rollout_count"),
            "unique_rollout_scope": bounded_rollout.get("unique_count_scope"),
        },
        "latency": {
            "bounded_emission": _summary(bounded_emission_samples),
            "bounded_rollout": _summary(bounded_rollout_samples),
            "bounded_history": _summary(bounded_history_samples),
            "legacy_emission": _summary(legacy_emission_samples),
            "legacy_rollout": _summary(legacy_rollout_samples),
            "legacy_history": _summary(legacy_history_samples),
            "bounded_combined_mean_ms": round(bounded_mean, 6),
            "legacy_combined_mean_ms": round(legacy_mean, 6),
            "bounded_speedup_vs_legacy": round(legacy_mean / max(bounded_mean, 1e-9), 6),
        },
        "retired_path_comparison": {
            "old_policy": (
                "status_read_model_materialized_all_retained_readout_review_and_rollout_rows"
            ),
            "bounded_checked_record_count": bounded_checked,
            "old_checked_record_count": legacy_checked,
            "record_work_reduction": round(legacy_checked / max(1, bounded_checked), 6),
        },
        "emission_source_window": bounded_emission_window,
        "emission_review_history_source_window": bounded_history_window,
        "rollout_source_window": bounded_rollout_window,
        "resource_behavior": {
            "python_tracemalloc_current_mib": round(current / (1024 * 1024), 6),
            "python_tracemalloc_peak_mib": round(peak / (1024 * 1024), 6),
            "cuda": {
                "torch_available": True,
                "cuda_available": bool(torch.cuda.is_available()),
                "device_name": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
                "gpu_used": False,
                "memory_allocated_mib": (
                    round(torch.cuda.memory_allocated() / (1024 * 1024), 6)
                    if torch.cuda.is_available()
                    else 0.0
                ),
                "memory_reserved_mib": (
                    round(torch.cuda.memory_reserved() / (1024 * 1024), 6)
                    if torch.cuda.is_available()
                    else 0.0
                ),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retention-count", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
