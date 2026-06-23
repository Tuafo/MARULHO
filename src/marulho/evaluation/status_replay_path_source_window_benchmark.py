"""Benchmark bounded Runtime Truth replay-path status projections.

This keeps the production read model on bounded CPU source windows and asserts
seeded source-window quality directly. Retired retained-scan projections stay
absent from the executable benchmark harness.
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
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    bounded_emission_window = dict(bounded_emission.get("source_window") or {})
    bounded_rollout_window = dict(bounded_rollout.get("source_window") or {})
    bounded_history_window = dict(bounded_history.get("source_window") or {})
    expected_emission = (state.get("emission_review_events") or [{}])[0]
    expected_rollout = (state.get("rollout_events") or [{}])[0]
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
        "latest_emission_matches_expected_seed": bounded_emission.get(
            "latest_prediction_hash"
        )
        == expected_emission.get("prediction_hash"),
        "latest_history_matches_expected_seed": bounded_history.get(
            "latest_emission_review_hash"
        )
        == expected_emission.get("emission_review_hash"),
        "latest_rollout_matches_expected_seed": bounded_rollout.get(
            "latest_rollout_hash"
        )
        == expected_rollout.get("rollout_hash"),
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
    bounded_mean = statistics.fmean(
        bounded_emission_samples + bounded_rollout_samples + bounded_history_samples
    )
    removed_retained_checked = int(
        len(state.get("events") or []) * 2
        + len(state.get("emission_review_events") or []) * 2
        + len(state.get("rollout_events") or [])
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
            "latest_emission_matches_expected_seed": pass_checks[
                "latest_emission_matches_expected_seed"
            ],
            "latest_history_matches_expected_seed": pass_checks[
                "latest_history_matches_expected_seed"
            ],
            "latest_rollout_matches_expected_seed": pass_checks[
                "latest_rollout_matches_expected_seed"
            ],
            "expected_latest_prediction_hash": expected_emission.get(
                "prediction_hash"
            ),
            "expected_latest_emission_review_hash": expected_emission.get(
                "emission_review_hash"
            ),
            "expected_latest_rollout_hash": expected_rollout.get("rollout_hash"),
            "bounded_emission_seed_count": bounded_emission.get("design_seed_candidate_count"),
            "bounded_history_unique_count": bounded_history.get("unique_emission_count"),
            "history_unique_scope": bounded_history.get("unique_count_scope"),
            "bounded_rollout_unique_count": bounded_rollout.get("unique_rollout_count"),
            "unique_rollout_scope": bounded_rollout.get("unique_count_scope"),
        },
        "latency": {
            "bounded_emission": _summary(bounded_emission_samples),
            "bounded_rollout": _summary(bounded_rollout_samples),
            "bounded_history": _summary(bounded_history_samples),
            "bounded_combined_mean_ms": round(bounded_mean, 6),
        },
        "source_budget": {
            "bounded_checked_record_count": bounded_checked,
            "removed_full_retained_checked_record_count": removed_retained_checked,
            "source_work_reduction_estimate": round(
                removed_retained_checked / max(1, bounded_checked),
                6,
            ),
        },
        "retired_full_retained_status_projection_absence": {
            "implementation_present": False,
            "diagnostic_callable": False,
            "active_report_field_present": False,
            "removed_policy": (
                "status_read_model_materialized_all_retained_readout_review_and_rollout_rows"
            ),
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
