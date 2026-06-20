"""Benchmark indexed SNN replay-artifact provenance source windows."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from threading import RLock
import statistics
import sys
import time
import tracemalloc
from typing import Any, Mapping

from marulho.service.replay_runtime import (
    DEFAULT_REPLAY_REGENERATION_PERMITS,
    DEFAULT_SNN_REPLAY_ARTIFACT_RECORDING_REVIEW_TICKETS,
    DEFAULT_SNN_REPLAY_EVALUATION_CONTEXTS,
    DEFAULT_SNN_TRANSITION_MEMORY_REPLAY_ARTIFACTS,
    SNN_REPLAY_PROVENANCE_SOURCE_RECORD_LIMIT,
    ReplayController,
    ReplayControllerDependencies,
)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
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
        "selection_criteria": [
            "recent_provenance_bound_readout_events",
            "label_repetition_within_source_window",
            "transition_memory_reuse_within_source_window",
            "recency_within_source_window",
        ],
        "source_limits": {
            "readout_events": 32,
            "returned_candidates": 1,
            "ledger_retention": 1,
        },
        "source_counts": {
            "retained_readout_events": 1,
            "source_readout_events": 1,
        },
        "window_counts": {
            "candidate_count_before_rank": 1,
            "candidate_count_returned": 1,
        },
        "truncated_source_counts": {
            "readout_events": 0,
        },
        "selection_budget": {
            "source_event_window_limit": 32,
            "requested_candidate_limit": 1,
            "returned_candidate_limit": 1,
            "ledger_retention_limit": 1,
        },
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
        "device_placement": {
            "archival_storage": "cpu",
            "source_window_selection": "cpu",
            "score": "cpu",
        },
        "gpu_used": False,
    }


@dataclass
class _RuntimeState:
    state_revision: int = 17

    def __post_init__(self) -> None:
        self.dirty_without_revision_calls = 0

    def mark_dirty_without_revision(self) -> None:
        self.dirty_without_revision_calls += 1


class _ReplayHarness:
    def __init__(self) -> None:
        self.lock = RLock()
        self.runtime_state = _RuntimeState()
        self.action_history: deque[dict[str, Any]] = deque(maxlen=8)
        self.trainer = type("_Trainer", (), {"token_count": 4096})()

    @staticmethod
    def normalize_text(value: Any, *, max_chars: int = 2000) -> str:
        text = " ".join(str(value).split()).strip()
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + "..."
        return text

    def living_loop_snapshot(self, **_: Any) -> dict[str, Any]:
        return {
            "state_revision": self.runtime_state.state_revision,
            "token_count": self.trainer.token_count,
            "runtime_episodes": [],
            "actions": [],
            "predictions": [],
            "uncertain_domains": [],
            "feedback_summary": {},
        }

    @staticmethod
    def replay_plan_summary(plan: Any) -> dict[str, Any]:
        payload = dict(plan or {})
        return {
            "endpoint": payload.get("endpoint", "/terminus/replay-plan"),
            "count": len(payload.get("candidates", [])),
        }


def _controller() -> ReplayController:
    harness = _ReplayHarness()
    return ReplayController(
        ReplayControllerDependencies(
            action_history=lambda: harness.action_history,
            living_loop_snapshot=harness.living_loop_snapshot,
            lock=harness.lock,
            normalize_action_text=harness.normalize_text,
            normalize_feedback_text=harness.normalize_text,
            replay_plan_summary=harness.replay_plan_summary,
            runtime_feedback_summary=lambda: {"feedback_count": 0},
            runtime_state=harness.runtime_state,
            runtime_trace_export_safe_value=lambda value: value,
            trainer=lambda: harness.trainer,
        )
    )


def _mismatch_report(score: float) -> dict[str, Any]:
    return {
        "surface": "snn_language_sequence_mismatch_probe.v1",
        "available": True,
        "owned_by_marulho": True,
        "prediction_error": {"mismatch_score": float(score)},
        "promotion_gate": {"status": "ready_for_operator_review"},
    }


def _pressure_report(score: float) -> dict[str, Any]:
    return {
        "surface": "snn_language_plasticity_pressure.v1",
        "available": True,
        "owned_by_marulho": True,
        "plasticity_pressure": {"pressure_score": float(score)},
        "promotion_gate": {"status": "ready_for_operator_review"},
    }


def _regeneration_design(index: int) -> dict[str, Any]:
    pre_index = int(index % 32)
    post_index = int(pre_index + 1)
    return {
        "locality_radius": 2,
        "initial_weight": 0.02,
        "max_new_synapses": 8,
        "mismatch_score": 0.9,
        "candidate_synapses": [
            {"pre_index": pre_index, "post_index": post_index, "initial_weight": 0.02}
        ],
    }


def _record_artifact_chain(
    controller: ReplayController,
    *,
    index: int,
    operator_id: str,
) -> dict[str, Any]:
    context = controller.record_snn_replay_evaluation_context(
        mismatch_report=_mismatch_report(0.95),
        pressure_report=_pressure_report(0.95),
        source_metadata={
            "source": "snn-replay-artifact-provenance-benchmark",
            "emission_hash": f"emission-{index}",
            "readout_evidence_hash": f"readout-{index}",
            "prediction_hash": f"prediction-{index}",
        },
    )
    context_id = str(context["replay_evaluation_context_id"])
    queue = controller.snn_replay_consolidation_priority_queue(
        readout_replay_priority_report={
            "surface": "snn_language_readout_replay_priority.v1",
            "candidates": [
                {
                    "priority_score": 95.0,
                    "all_labels_grounded": True,
                    "replay_evaluation_context_id": context_id,
                }
            ],
        },
        limit=4,
    )
    policy = controller.snn_replay_artifact_recording_policy_proposal(
        consolidation_priority_queue=queue,
        policy={"min_priority_score": 60.0},
    )
    ticket = controller.record_snn_replay_artifact_recording_review_ticket(
        policy_proposal=policy,
        operator_id=operator_id,
        confirmation=True,
    )
    context_lineage = controller._snn_replay_context_emission_lineage(  # noqa: SLF001
        context.get("source_metadata")
        if isinstance(context.get("source_metadata"), Mapping)
        else {}
    )
    readout_hash = f"readout-window-{index}"
    replay_priority_source_window = _readout_replay_priority_source_window()
    artifact = controller.record_evaluated_snn_transition_memory_replay_artifact(
        artifact_proposal={
            "surface": "snn_transition_memory_replay_artifact_proposal.v1",
            "ready": True,
            "owned_by_marulho": True,
            "source": "service.snn_language_readout_ledger.transition_memory_replay_artifact_proposal",
            "mismatch_report": context["mismatch_report"],
            "pressure_report": context["pressure_report"],
            "replay_evaluation_context_id": context_id,
            "replay_evaluation_context_hash": context["evidence_hash"],
            "source_metadata_hash": context.get("source_metadata_hash"),
            "emission_lineage": context_lineage,
            "replay_window": [{"readout_evidence_hash": readout_hash, "grounded": True}],
            "replay_priority_source_window": replay_priority_source_window,
            "replay_priority_source_window_hash": _sha256_json(
                replay_priority_source_window
            ),
            "promotion_gate": {"status": "ready_for_operator_recording_review"},
        },
        known_readout_evidence_hashes={readout_hash},
        known_readout_evidence_source_window=_known_readout_evidence_source_window(),
        replay_evaluation_context_id=context_id,
        review_ticket_id=str(ticket["review_ticket_id"]),
        operator_id=operator_id,
        confirmation=True,
    )
    design = _regeneration_design(index)
    permit = controller.issue_regeneration_permit(
        replay_artifact_id=str(artifact["replay_artifact_id"]),
        regeneration_design=design,
        operator_id=operator_id,
        confirmation=True,
    )
    return {
        "context": context,
        "ticket": ticket,
        "artifact": artifact,
        "permit": permit,
        "design": design,
    }


def _process_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    try:
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def _cuda_report() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"torch_available": False, "cuda_available": False, "gpu_used": False}
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
            3,
        )
        report["memory_reserved_mib"] = round(
            float(torch.cuda.memory_reserved(0)) / (1024.0 * 1024.0),
            3,
        )
    return report


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    controller = _controller()
    retention_count = max(
        1,
        min(
            int(args.retention_count),
            DEFAULT_SNN_REPLAY_EVALUATION_CONTEXTS,
            DEFAULT_SNN_REPLAY_ARTIFACT_RECORDING_REVIEW_TICKETS,
            DEFAULT_SNN_TRANSITION_MEMORY_REPLAY_ARTIFACTS,
            DEFAULT_REPLAY_REGENERATION_PERMITS,
        ),
    )
    operator_id = "operator-provenance-benchmark"
    old_chain = _record_artifact_chain(controller, index=0, operator_id=operator_id)
    for index in range(1, retention_count):
        _record_artifact_chain(controller, index=index, operator_id=operator_id)

    proposal = {
        "replay_evidence": old_chain["permit"],
        "regeneration_design": old_chain["design"],
    }
    full_source_window = controller._snn_replay_provenance_source_window(  # noqa: SLF001
        replay_evaluation_context_id=str(
            old_chain["context"]["replay_evaluation_context_id"]
        ),
        review_ticket_id=str(old_chain["ticket"]["review_ticket_id"]),
        replay_artifact_id=str(old_chain["artifact"]["replay_artifact_id"]),
        permit_id=str(old_chain["permit"]["permit_id"]),
    )
    readout_source_window = (
        dict(old_chain["artifact"].get("readout_evidence_source_window"))
        if isinstance(old_chain["artifact"].get("readout_evidence_source_window"), Mapping)
        else {}
    )
    replay_priority_source_window = (
        dict(old_chain["artifact"].get("replay_priority_source_window"))
        if isinstance(old_chain["artifact"].get("replay_priority_source_window"), Mapping)
        else {}
    )
    rss_before = _process_rss_mb()
    timings_ms: list[float] = []
    results: list[bool] = []
    for _ in range(max(1, int(args.runs))):
        started = time.perf_counter()
        results.append(bool(controller.verify_regeneration_permit(proposal)))
        timings_ms.append((time.perf_counter() - started) * 1000.0)
    rss_after = _process_rss_mb()
    tracemalloc.start()
    traced_verified = bool(controller.verify_regeneration_permit(proposal))
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    context_tail_id = str(controller.snn_replay_evaluation_contexts[-1]["replay_evaluation_context_id"])
    ticket_tail_id = str(controller.snn_replay_artifact_recording_review_tickets[-1]["review_ticket_id"])
    artifact_tail_id = str(controller.snn_transition_memory_replay_artifacts[-1]["replay_artifact_id"])
    permit_tail_id = str(controller.regeneration_permits[-1]["permit_id"])
    old_ids = {
        "replay_evaluation_context_id": str(
            old_chain["context"]["replay_evaluation_context_id"]
        ),
        "review_ticket_id": str(old_chain["ticket"]["review_ticket_id"]),
        "replay_artifact_id": str(old_chain["artifact"]["replay_artifact_id"]),
        "permit_id": str(old_chain["permit"]["permit_id"]),
    }
    latency = {
        "runs": int(max(1, int(args.runs))),
        "mean_ms": round(float(statistics.mean(timings_ms)), 6),
        "median_ms": round(float(statistics.median(timings_ms)), 6),
        "p95_ms": round(float(sorted(timings_ms)[-1]), 6),
        "samples_ms": [round(float(value), 6) for value in timings_ms],
    }
    retired_scan_comparisons = {
        "old_policy": "linear_scan_retained_context_ticket_artifact_permit_deques_by_id",
        "old_worst_case_retained_record_checks": int(retention_count * 4),
        "bounded_index_lookup_count": int(full_source_window.get("index_lookup_count", 0) or 0),
        "lookup_work_reduction": round(
            float(retention_count * 4)
            / max(1, int(full_source_window.get("index_lookup_count", 0) or 0)),
            6,
        ),
    }
    pass_checks = {
        "old_permit_verified_every_run": all(results) and traced_verified,
        "old_records_remain_retained_tail": (
            context_tail_id == old_ids["replay_evaluation_context_id"]
            and ticket_tail_id == old_ids["review_ticket_id"]
            and artifact_tail_id == old_ids["replay_artifact_id"]
            and permit_tail_id == old_ids["permit_id"]
        ),
        "source_window_surface_present": full_source_window.get("surface")
        == "bounded_snn_replay_artifact_provenance_source_window.v1",
        "source_record_limit_respected": int(
            full_source_window.get("source_record_count", 0) or 0
        )
        <= SNN_REPLAY_PROVENANCE_SOURCE_RECORD_LIMIT,
        "all_index_lookups_hit": int(full_source_window.get("index_hit_count", 0) or 0)
        == int(full_source_window.get("source_record_count", 0) or 0),
        "no_global_scan": full_source_window.get("global_candidate_scan") is False
        and full_source_window.get("global_score_scan") is False,
        "cpu_only": full_source_window.get("archival_storage_device") == "cpu"
        and full_source_window.get("gpu_used") is False,
        "not_live_tick": full_source_window.get("runs_live_tick") is False,
        "no_language_reasoning": full_source_window.get("language_reasoning") is False,
        "known_readout_source_window_surface_present": (
            readout_source_window.get("surface")
            == "bounded_snn_readout_known_evidence_hash_source_window.v1"
        ),
        "known_readout_source_window_bounded": int(
            readout_source_window.get("source_window_count", 0) or 0
        )
        <= int(readout_source_window.get("source_window_limit", 0) or 0),
        "known_readout_source_window_cpu_only": (
            readout_source_window.get("archival_storage_device") == "cpu"
            and readout_source_window.get("gpu_used") is False
        ),
        "known_readout_source_window_no_hidden_work": (
            readout_source_window.get("global_candidate_scan") is False
            and readout_source_window.get("global_score_scan") is False
            and readout_source_window.get("raw_text_payload_loaded") is False
            and readout_source_window.get("language_reasoning") is False
            and readout_source_window.get("runs_live_tick") is False
            and readout_source_window.get("runs_every_token") is False
        ),
        "known_readout_source_window_persisted_hash": bool(
            old_chain["artifact"].get("readout_evidence_source_window_hash")
        ),
        "replay_priority_source_window_surface_present": (
            replay_priority_source_window.get("surface")
            == "bounded_snn_readout_replay_priority_source_window.v1"
        ),
        "replay_priority_source_window_bounded": (
            0
            <= int(replay_priority_source_window.get("source_event_window_count", -1) or -1)
            <= int(replay_priority_source_window.get("source_event_window_limit", 0) or 0)
            <= 32
            and 0
            <= int(
                replay_priority_source_window.get(
                    "candidate_count_returned",
                    -1,
                )
                or -1
            )
            <= int(
                replay_priority_source_window.get(
                    "candidate_count_before_rank",
                    0,
                )
                or 0
            )
        ),
        "replay_priority_source_window_cpu_only": (
            replay_priority_source_window.get("archival_storage_device") == "cpu"
            and replay_priority_source_window.get("score_device") == "cpu"
            and replay_priority_source_window.get("gpu_used") is False
        ),
        "replay_priority_source_window_no_hidden_work": (
            replay_priority_source_window.get("global_candidate_scan") is False
            and replay_priority_source_window.get("global_score_scan") is False
            and replay_priority_source_window.get("raw_text_payload_loaded") is False
            and replay_priority_source_window.get("language_reasoning") is False
            and replay_priority_source_window.get("runs_live_tick") is False
            and replay_priority_source_window.get("runs_every_token") is False
        ),
        "replay_priority_source_window_persisted_hash": bool(
            old_chain["artifact"].get("replay_priority_source_window_hash")
        ),
    }
    return {
        "surface": "bounded_snn_replay_artifact_provenance_source_window_benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "pass": all(pass_checks.values()),
        "pass_checks": pass_checks,
        "input": {
            "retention_count": retention_count,
            "requested_retention_count": int(args.retention_count),
            "runs": int(max(1, int(args.runs))),
        },
        "quality": {
            "old_permit_verified": all(results) and traced_verified,
            "old_ids": old_ids,
            "tail_ids": {
                "replay_evaluation_context_id": context_tail_id,
                "review_ticket_id": ticket_tail_id,
                "replay_artifact_id": artifact_tail_id,
                "permit_id": permit_tail_id,
            },
            "source_window": full_source_window,
            "known_readout_evidence_source_window": readout_source_window,
            "known_readout_evidence_source_window_hash": old_chain["artifact"].get(
                "readout_evidence_source_window_hash"
            ),
            "replay_priority_source_window": replay_priority_source_window,
            "replay_priority_source_window_hash": old_chain["artifact"].get(
                "replay_priority_source_window_hash"
            ),
        },
        "latency": latency,
        "retired_path_comparison": retired_scan_comparisons,
        "resource_behavior": {
            "process_rss_before_mib": None if rss_before is None else round(rss_before, 3),
            "process_rss_after_mib": None if rss_after is None else round(rss_after, 3),
            "process_rss_delta_mib": None
            if rss_before is None or rss_after is None
            else round(rss_after - rss_before, 3),
            "python_tracemalloc_current_mib": round(
                float(traced_current) / (1024.0 * 1024.0),
                6,
            ),
            "python_tracemalloc_peak_mib": round(
                float(traced_peak) / (1024.0 * 1024.0),
                6,
            ),
            "cuda": _cuda_report(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention-count", type=int, default=64)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run_benchmark(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
