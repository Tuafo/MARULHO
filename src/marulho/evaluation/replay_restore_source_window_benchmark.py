from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics
from threading import RLock
import time
import tracemalloc
from typing import Any, Callable, Mapping, Sequence

import torch

from marulho.service.replay_runtime import (
    DEFAULT_REPLAY_REGENERATION_PERMITS,
    DEFAULT_REPLAY_SAMPLE_HISTORY,
    DEFAULT_SNN_REPLAY_ARTIFACT_RECORDING_REVIEW_TICKETS,
    DEFAULT_SNN_REPLAY_EVALUATION_CONTEXTS,
    DEFAULT_SNN_SLEEP_PLASTICITY_REVIEW_SCHEDULER_INSTALLATIONS,
    DEFAULT_SNN_SLEEP_PLASTICITY_REVIEW_TICKETS,
    DEFAULT_SNN_SLEEP_PLASTICITY_SCHEDULER_DESIGN_REVIEW_TICKETS,
    DEFAULT_SNN_TRANSITION_MEMORY_REPLAY_ARTIFACTS,
    REPLAY_RESTORE_SOURCE_WINDOW_SURFACE,
    ReplayController,
    ReplayControllerDependencies,
)


@dataclass
class _RuntimeState:
    state_revision: int = 7

    def __post_init__(self) -> None:
        self.dirty_without_revision_calls = 0

    def mark_dirty_without_revision(self) -> None:
        self.dirty_without_revision_calls += 1

    def mutation_summary(self) -> dict[str, Any]:
        return {"dirty_state": True, "state_revision": self.state_revision}


class _ReplayHarness:
    def __init__(self) -> None:
        self.lock = RLock()
        self.runtime_state = _RuntimeState()
        self.action_history: deque[dict[str, Any]] = deque(maxlen=24)
        self.trainer = type("_Trainer", (), {"token_count": 13})()

    @staticmethod
    def normalize_action_text(value: Any) -> str:
        return " ".join(str(value).split()).strip()

    @classmethod
    def normalize_feedback_text(cls, value: Any, *, max_chars: int = 2000) -> str:
        text = cls.normalize_action_text(value)
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + "..."
        return text

    def living_loop_snapshot(self, **_: Any) -> dict[str, Any]:
        return {
            "state_revision": self.runtime_state.state_revision,
            "token_count": 13,
            "feedback_summary": {},
            "benchmark_telemetry": {},
            "memory_health": {"status": "available", "fill_ratio": 0.2},
            "subcortex_sleep_pressure": {"fatigue": 0.2},
            "policy_decision": {"action": "continue_current_policy"},
            "world_model_lite": {"uncertainty": 0.0},
        }

    @staticmethod
    def replay_plan_summary(plan: Mapping[str, Any] | None) -> dict[str, Any]:
        payload = dict(plan or {})
        return {
            "endpoint": payload.get("endpoint", "/terminus/replay-plan"),
            "count": len(payload.get("candidates", [])),
        }

    @staticmethod
    def runtime_trace_export_safe_value(value: Any) -> Any:
        return value

    @staticmethod
    def runtime_feedback_summary() -> dict[str, int]:
        return {"feedback_count": 0}


def _controller(**kwargs: Any) -> ReplayController:
    harness = _ReplayHarness()
    return ReplayController(
        ReplayControllerDependencies(
            action_history=lambda: harness.action_history,
            living_loop_snapshot=harness.living_loop_snapshot,
            lock=harness.lock,
            normalize_action_text=harness.normalize_action_text,
            normalize_feedback_text=harness.normalize_feedback_text,
            replay_plan_summary=harness.replay_plan_summary,
            runtime_feedback_summary=harness.runtime_feedback_summary,
            runtime_state=harness.runtime_state,
            runtime_trace_export_safe_value=harness.runtime_trace_export_safe_value,
            trainer=lambda: harness.trainer,
        ),
        **kwargs,
    )


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _known_readout_evidence_source_window() -> dict[str, Any]:
    return {
        "surface": "bounded_snn_readout_known_evidence_hash_source_window.v1",
        "source": "snn_readout_ledger.events",
        "source_window_limit": 8,
        "source_window_count": 1,
        "source_record_count": 1,
        "hash_count": 1,
        "global_candidate_scan": False,
        "global_score_scan": False,
        "raw_text_payload_loaded": False,
        "language_reasoning": False,
        "runs_live_tick": False,
        "runs_every_token": False,
        "mutates_runtime_state": False,
        "applies_plasticity": False,
        "archival_storage_device": "cpu",
        "gpu_used": False,
    }


def _readout_replay_priority_source_window() -> dict[str, Any]:
    return {
        "surface": "bounded_snn_readout_replay_priority_source_window.v1",
        "policy": "recent_readout_event_source_window_v1",
        "window_policy": "recent_readout_event_source_window_v1",
        "source_event_retention_count": 1,
        "source_event_window_limit": 32,
        "source_event_window_count": 1,
        "source_event_truncated_count": 0,
        "candidate_count_before_rank": 1,
        "candidate_count_returned": 1,
        "global_candidate_scan": False,
        "global_score_scan": False,
        "raw_text_payload_loaded": False,
        "language_reasoning": False,
        "runs_live_tick": False,
        "runs_every_token": False,
        "mutates_runtime_state": False,
        "applies_plasticity": False,
        "archival_storage_device": "cpu",
        "score_device": "cpu",
        "gpu_used": False,
    }


def _source_window() -> dict[str, Any]:
    return {
        "surface": "bounded_snn_replay_artifact_provenance_source_window.v1",
        "source": "replay_controller.indexed_restore_window",
        "source_window_limit": 4,
        "source_window_count": 4,
        "source_record_count": 4,
        "global_candidate_scan": False,
        "global_score_scan": False,
        "raw_text_payload_loaded": False,
        "language_reasoning": False,
        "runs_live_tick": False,
        "gpu_used": False,
    }


def _artifact(index: int) -> dict[str, Any]:
    source_window = _source_window()
    readout_window = _known_readout_evidence_source_window()
    priority_window = _readout_replay_priority_source_window()
    return {
        "surface": "snn_transition_memory_replay_artifact.v1",
        "artifact_kind": "terminus_snn_transition_memory_replay_artifact",
        "internal_ledger_backed": True,
        "replay_artifact_id": f"artifact-{index:05d}",
        "evidence_hash": f"artifact-evidence-{index:05d}",
        "artifact_proposal_hash": f"artifact-proposal-{index:05d}",
        "replay_evaluation_context_id": f"context-{index:05d}",
        "replay_evaluation_context_hash": f"context-hash-{index:05d}",
        "review_ticket_id": f"ticket-{index:05d}",
        "review_ticket_hash": f"ticket-hash-{index:05d}",
        "readout_evidence_hashes": ["readout-hash-1"],
        "source_window": source_window,
        "source_window_hash": _sha256_json(source_window),
        "readout_evidence_source_window": readout_window,
        "readout_evidence_source_window_hash": _sha256_json(readout_window),
        "replay_priority_source_window": priority_window,
        "replay_priority_source_window_hash": _sha256_json(priority_window),
    }


def _records(entry_count: int) -> dict[str, list[dict[str, Any]]]:
    count = max(1, int(entry_count))
    return {
        "replay_sample_history": [
            {
                "schema_version": 1,
                "replay_sample_id": f"replay-sample-{index:05d}",
                "mode": "sample",
                "status": "recorded",
                "selected_candidate_ids": [f"candidate-{index:05d}"],
                "selected_candidates": [],
                "safety_flags": {"audit_only": True, "operator_confirmed": True},
            }
            for index in range(count)
        ],
        "replay_regeneration_permits": [
            {"permit_id": f"permit-{index:05d}", "evidence_hash": f"permit-hash-{index:05d}"}
            for index in range(count)
        ],
        "snn_replay_evaluation_contexts": [
            {
                "replay_evaluation_context_id": f"context-{index:05d}",
                "evidence_hash": f"context-hash-{index:05d}",
            }
            for index in range(count)
        ],
        "snn_replay_artifact_recording_review_tickets": [
            {"review_ticket_id": f"ticket-{index:05d}", "evidence_hash": f"ticket-hash-{index:05d}"}
            for index in range(count)
        ],
        "snn_sleep_plasticity_review_tickets": [
            {"review_ticket_id": f"sleep-ticket-{index:05d}", "evidence_hash": f"sleep-hash-{index:05d}"}
            for index in range(count)
        ],
        "snn_sleep_plasticity_scheduler_design_review_tickets": [
            {
                "scheduler_design_review_ticket_id": f"design-ticket-{index:05d}",
                "evidence_hash": f"design-hash-{index:05d}",
            }
            for index in range(count)
        ],
        "snn_sleep_plasticity_review_scheduler_installations": [
            {"scheduler_installation_id": f"install-{index:05d}", "evidence_hash": f"install-hash-{index:05d}"}
            for index in range(count)
        ],
        "snn_transition_memory_replay_artifacts": [_artifact(index) for index in range(count)],
    }


def _controller_from_records(records: Mapping[str, Sequence[Mapping[str, Any]]]) -> ReplayController:
    return _controller(
        replay_sample_history=records["replay_sample_history"],
        regeneration_permits=records["replay_regeneration_permits"],
        snn_replay_evaluation_contexts=records["snn_replay_evaluation_contexts"],
        snn_replay_artifact_recording_review_tickets=records[
            "snn_replay_artifact_recording_review_tickets"
        ],
        snn_sleep_plasticity_review_tickets=records["snn_sleep_plasticity_review_tickets"],
        snn_sleep_plasticity_scheduler_design_review_tickets=records[
            "snn_sleep_plasticity_scheduler_design_review_tickets"
        ],
        snn_sleep_plasticity_review_scheduler_installations=records[
            "snn_sleep_plasticity_review_scheduler_installations"
        ],
        snn_transition_memory_replay_artifacts=records[
            "snn_transition_memory_replay_artifacts"
        ],
    )


def _state(controller: ReplayController) -> dict[str, Any]:
    return {
        "replay_sample_ids": [
            str(item.get("replay_sample_id") or "") for item in controller.history
        ],
        "permit_ids": [
            str(item.get("permit_id") or "") for item in controller.regeneration_permits
        ],
        "context_ids": [
            str(item.get("replay_evaluation_context_id") or "")
            for item in controller.snn_replay_evaluation_contexts
        ],
        "review_ticket_ids": [
            str(item.get("review_ticket_id") or "")
            for item in controller.snn_replay_artifact_recording_review_tickets
        ],
        "sleep_ticket_ids": [
            str(item.get("review_ticket_id") or "")
            for item in controller.snn_sleep_plasticity_review_tickets
        ],
        "design_ticket_ids": [
            str(item.get("scheduler_design_review_ticket_id") or "")
            for item in controller.snn_sleep_plasticity_scheduler_design_review_tickets
        ],
        "installation_ids": [
            str(item.get("scheduler_installation_id") or "")
            for item in controller.snn_sleep_plasticity_review_scheduler_installations
        ],
        "artifact_ids": [
            str(item.get("replay_artifact_id") or "")
            for item in controller.snn_transition_memory_replay_artifacts
        ],
    }


def _retired_full_materialized_state(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    controller = _controller()
    replay_samples = [
        item
        for item in (
            controller._normalize_replay_sample_record(raw)  # noqa: SLF001
            for raw in records["replay_sample_history"]
        )
        if item is not None
    ][:DEFAULT_REPLAY_SAMPLE_HISTORY]
    artifacts = [
        item
        for item in (
            controller._normalize_evaluated_snn_transition_memory_replay_artifact(raw)  # noqa: SLF001
            for raw in records["snn_transition_memory_replay_artifacts"]
        )
        if item is not None
    ][:DEFAULT_SNN_TRANSITION_MEMORY_REPLAY_ARTIFACTS]
    return {
        "replay_sample_ids": [
            str(item.get("replay_sample_id") or "") for item in replay_samples
        ],
        "permit_ids": [
            str(item.get("permit_id") or "")
            for item in [dict(raw) for raw in records["replay_regeneration_permits"] if isinstance(raw, Mapping)][
                :DEFAULT_REPLAY_REGENERATION_PERMITS
            ]
        ],
        "context_ids": [
            str(item.get("replay_evaluation_context_id") or "")
            for item in [dict(raw) for raw in records["snn_replay_evaluation_contexts"] if isinstance(raw, Mapping)][
                :DEFAULT_SNN_REPLAY_EVALUATION_CONTEXTS
            ]
        ],
        "review_ticket_ids": [
            str(item.get("review_ticket_id") or "")
            for item in [
                dict(raw)
                for raw in records["snn_replay_artifact_recording_review_tickets"]
                if isinstance(raw, Mapping)
            ][:DEFAULT_SNN_REPLAY_ARTIFACT_RECORDING_REVIEW_TICKETS]
        ],
        "sleep_ticket_ids": [
            str(item.get("review_ticket_id") or "")
            for item in [dict(raw) for raw in records["snn_sleep_plasticity_review_tickets"] if isinstance(raw, Mapping)][
                :DEFAULT_SNN_SLEEP_PLASTICITY_REVIEW_TICKETS
            ]
        ],
        "design_ticket_ids": [
            str(item.get("scheduler_design_review_ticket_id") or "")
            for item in [
                dict(raw)
                for raw in records["snn_sleep_plasticity_scheduler_design_review_tickets"]
                if isinstance(raw, Mapping)
            ][:DEFAULT_SNN_SLEEP_PLASTICITY_SCHEDULER_DESIGN_REVIEW_TICKETS]
        ],
        "installation_ids": [
            str(item.get("scheduler_installation_id") or "")
            for item in [
                dict(raw)
                for raw in records["snn_sleep_plasticity_review_scheduler_installations"]
                if isinstance(raw, Mapping)
            ][:DEFAULT_SNN_SLEEP_PLASTICITY_REVIEW_SCHEDULER_INSTALLATIONS]
        ],
        "artifact_ids": [
            str(item.get("replay_artifact_id") or "") for item in artifacts
        ],
    }


def _measure(fn: Callable[[], dict[str, Any]], *, runs: int) -> tuple[list[float], dict[str, Any]]:
    elapsed: list[float] = []
    latest: dict[str, Any] = {}
    for _ in range(max(1, int(runs))):
        started = time.perf_counter()
        latest = fn()
        elapsed.append(float((time.perf_counter() - started) * 1000.0))
    return elapsed, latest


def _latency(values: Sequence[float]) -> dict[str, Any]:
    return {
        "count": int(len(values)),
        "mean_ms": round(float(statistics.fmean(values)), 6) if values else 0.0,
        "median_ms": round(float(statistics.median(values)), 6) if values else 0.0,
        "max_ms": round(float(max(values)), 6) if values else 0.0,
        "samples_ms": [round(float(value), 6) for value in values],
    }


def run_benchmark(*, entry_count: int, runs: int) -> dict[str, Any]:
    records = _records(entry_count)
    cuda_available = bool(torch.cuda.is_available())
    cuda_allocated_before = (
        float(torch.cuda.memory_allocated() / (1024 * 1024))
        if cuda_available
        else 0.0
    )
    cuda_reserved_before = (
        float(torch.cuda.memory_reserved() / (1024 * 1024))
        if cuda_available
        else 0.0
    )

    def _run_bounded() -> dict[str, Any]:
        controller = _controller_from_records(records)
        return {
            "state": _state(controller),
            "restore_source_window": controller.replay_restore_source_window_report(),
        }

    bounded_elapsed, bounded_latest = _measure(_run_bounded, runs=runs)
    retired_elapsed, retired_state = _measure(
        lambda: _retired_full_materialized_state(records),
        runs=runs,
    )

    tracemalloc.start()
    try:
        peak_latest = _run_bounded()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    bounded_state = bounded_latest["state"]
    report = dict(peak_latest["restore_source_window"])
    source_work = int(report.get("total_source_window_inspected_count", 0) or 0)
    retired_work = sum(int(len(value)) for value in records.values())
    parity = bounded_state == retired_state
    required_false_flags = [
        "full_retained_materialization",
        "global_candidate_scan",
        "global_score_scan",
        "raw_text_payload_loaded",
        "language_reasoning",
        "runs_live_tick",
        "runs_every_token",
        "runs_live_replay",
        "mutates_runtime_state",
        "applies_plasticity",
        "gpu_used",
        "gpu_resident_archival_metadata",
    ]
    fields = report.get("fields") if isinstance(report.get("fields"), Mapping) else {}
    all_fields_bounded = all(
        isinstance(field, Mapping)
        and field.get("normalizes_full_retained_state") is False
        and int(field.get("source_window_inspected_count", 0) or 0)
        <= int(field.get("retention_limit", 0) or 0)
        for field in fields.values()
    )
    cuda_allocated_after = (
        float(torch.cuda.memory_allocated() / (1024 * 1024))
        if cuda_available
        else 0.0
    )
    cuda_reserved_after = (
        float(torch.cuda.memory_reserved() / (1024 * 1024))
        if cuda_available
        else 0.0
    )
    pass_gate = bool(
        report.get("surface") == REPLAY_RESTORE_SOURCE_WINDOW_SURFACE
        and parity
        and all_fields_bounded
        and all(report.get(flag) is False for flag in required_false_flags)
        and source_work > 0
        and retired_work > source_work
    )
    return {
        "surface": "bounded_replay_restore_source_window_benchmark.v1",
        "pass": pass_gate,
        "entry_count": int(entry_count),
        "runs": int(runs),
        "selection_criteria": [
            "newest_checkpoint_replay_records_first",
            "controller_retention_limit_per_replay_field",
            "normalize_and_index_only_bounded_restore_source_window",
        ],
        "memory_budget": {
            "replay_sample_history": DEFAULT_REPLAY_SAMPLE_HISTORY,
            "replay_regeneration_permits": DEFAULT_REPLAY_REGENERATION_PERMITS,
            "snn_replay_evaluation_contexts": DEFAULT_SNN_REPLAY_EVALUATION_CONTEXTS,
            "snn_replay_artifact_recording_review_tickets": (
                DEFAULT_SNN_REPLAY_ARTIFACT_RECORDING_REVIEW_TICKETS
            ),
            "snn_sleep_plasticity_review_tickets": DEFAULT_SNN_SLEEP_PLASTICITY_REVIEW_TICKETS,
            "snn_sleep_plasticity_scheduler_design_review_tickets": (
                DEFAULT_SNN_SLEEP_PLASTICITY_SCHEDULER_DESIGN_REVIEW_TICKETS
            ),
            "snn_sleep_plasticity_review_scheduler_installations": (
                DEFAULT_SNN_SLEEP_PLASTICITY_REVIEW_SCHEDULER_INSTALLATIONS
            ),
            "snn_transition_memory_replay_artifacts": DEFAULT_SNN_TRANSITION_MEMORY_REPLAY_ARTIFACTS,
        },
        "quality": {
            "latest_window_parity_with_retired_full_materialized_restore": bool(parity),
            "valid_artifact_window_parity": bool(
                bounded_state.get("artifact_ids") == retired_state.get("artifact_ids")
            ),
            "restored_artifact_count": int(len(bounded_state.get("artifact_ids", []))),
        },
        "latency": {
            "bounded_restore": _latency(bounded_elapsed),
            "retired_full_materialized_restore": _latency(retired_elapsed),
            "speedup": (
                float(statistics.fmean(retired_elapsed) / statistics.fmean(bounded_elapsed))
                if bounded_elapsed and statistics.fmean(bounded_elapsed) > 0
                else None
            ),
        },
        "source_work": {
            "bounded_source_window_inspected_count": source_work,
            "retired_full_materialized_source_count": int(retired_work),
            "work_reduction": (
                float(retired_work / source_work) if source_work else None
            ),
        },
        "device_placement": {
            "archival_metadata": "cpu",
            "source_window": "cpu",
            "normalization": "cpu",
            "cuda_available": cuda_available,
            "gpu_used": False,
            "cuda_allocated_before_mib": cuda_allocated_before,
            "cuda_allocated_after_mib": cuda_allocated_after,
            "cuda_reserved_before_mib": cuda_reserved_before,
            "cuda_reserved_after_mib": cuda_reserved_after,
        },
        "memory": {
            "python_traced_peak_mib": float(peak_bytes / (1024 * 1024)),
        },
        "runtime_truth": report,
        "retired_path": {
            "name": "full-materialized replay restore normalization",
            "production_callable": False,
            "benchmark_local_only": True,
            "retired_work_model": "normalize every retained replay checkpoint record before retention slicing",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-count", type=int, default=65536)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_benchmark(entry_count=args.entry_count, runs=args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
