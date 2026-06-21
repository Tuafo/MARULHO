from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping, Sequence

from marulho.service.replay_runtime import (
    SNN_SLEEP_PLASTICITY_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT,
    SNN_SLEEP_PLASTICITY_SCHEDULER_DESIGN_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT,
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


def _controller() -> ReplayController:
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
        )
    )


def _sleep_policy() -> dict[str, Any]:
    return {
        "surface": "snn_language_transition_memory_sleep_policy.v1",
        "available": True,
        "owned_by_marulho": True,
        "mutates_runtime_state": False,
        "transition_memory": {
            "sparse_transition_weight_count": 4,
            "homeostatic_maintenance_count": 0,
            "regeneration_count": 1,
            "regenerated_synapse_count_total": 1,
        },
        "replay_evidence": {"available": True, "ready": True},
        "rollout_regeneration_evidence": {"available": True, "application_applied": True},
        "readout_ledger_evidence": {"available": True, "rollout_event_count": 1},
        "recommendation": {
            "action": "review_transition_memory_homeostatic_maintenance",
            "recommended": True,
            "suggested_endpoint": (
                "/terminus/snn-language-sequence/plasticity-homeostatic-maintenance"
            ),
            "requires_operator_confirmation": True,
            "executable": False,
            "reason_codes": ["post_growth_homeostatic_maintenance_due"],
        },
    }


def _latency_stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "max_ms": None}
    return {
        "count": int(len(values)),
        "mean_ms": round(float(statistics.fmean(values)), 6),
        "median_ms": round(float(statistics.median(values)), 6),
        "max_ms": round(float(max(values)), 6),
        "samples_ms": [round(float(value), 6) for value in values],
    }


def _measure(fn: Callable[[], dict[str, Any]], *, runs: int) -> tuple[list[float], dict[str, Any]]:
    elapsed: list[float] = []
    latest: dict[str, Any] = {}
    for _ in range(int(runs)):
        started = time.perf_counter()
        latest = fn()
        elapsed.append(float((time.perf_counter() - started) * 1000.0))
    return elapsed, latest


def _duplicate_records(
    record: Mapping[str, Any],
    *,
    id_field: str,
    prefix: str,
    count: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(int(count)):
        item = deepcopy(dict(record))
        item[id_field] = f"{prefix}-{index:04d}"
        records.append(item)
    return records


def _diagnostic_sleep_ticket_queue(controller: ReplayController) -> dict[str, Any]:
    tickets: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    verified_count = 0
    stale_count = 0
    tampered_count = 0
    current_revision = int(controller._runtime_state.state_revision)  # noqa: SLF001
    for source_rank, raw_ticket in enumerate(controller.snn_sleep_plasticity_review_tickets):
        if not isinstance(raw_ticket, Mapping):
            continue
        ticket = dict(raw_ticket)
        verified_ticket = controller.verified_snn_sleep_plasticity_review_ticket(
            str(ticket.get("review_ticket_id") or "")
        )
        verified = verified_ticket is not None
        revision_current = int(ticket.get("recorded_state_revision", -1)) == current_revision
        if verified:
            verified_count += 1
            action_counts[str(ticket.get("recommended_action") or "unknown")] += 1
        elif not revision_current:
            stale_count += 1
        else:
            tampered_count += 1
        tickets.append(
            {
                "review_ticket_id": ticket.get("review_ticket_id"),
                "suggested_endpoint": ticket.get("suggested_endpoint"),
                "recommended_action": ticket.get("recommended_action"),
                "verified": verified,
                "revision_current": revision_current,
                "source_rank": int(source_rank),
            }
        )
    latest_verified = next((deepcopy(item) for item in tickets if item["verified"]), None)
    return {
        "surface": "diagnostic_full_retained_sleep_plasticity_review_ticket_queue.v1",
        "records_scanned": int(len(tickets)),
        "verified_count": int(verified_count),
        "stale_count": int(stale_count),
        "tampered_count": int(tampered_count),
        "pending_action_counts": dict(action_counts),
        "latest_verified_ticket": latest_verified,
    }


def _diagnostic_scheduler_design_ticket_queue(
    controller: ReplayController,
) -> dict[str, Any]:
    tickets: list[dict[str, Any]] = []
    verified_count = 0
    stale_count = 0
    tampered_count = 0
    current_revision = int(controller._runtime_state.state_revision)  # noqa: SLF001
    for source_rank, raw_ticket in enumerate(
        controller.snn_sleep_plasticity_scheduler_design_review_tickets
    ):
        if not isinstance(raw_ticket, Mapping):
            continue
        ticket = dict(raw_ticket)
        try:
            verified_ticket = controller.verified_snn_sleep_plasticity_scheduler_design_review_ticket(
                str(ticket.get("scheduler_design_review_ticket_id") or "")
            )
        except (TypeError, ValueError):
            verified_ticket = None
        verified = verified_ticket is not None
        revision_current = int(ticket.get("recorded_state_revision", -1)) == current_revision
        if verified:
            verified_count += 1
        elif not revision_current:
            stale_count += 1
        else:
            tampered_count += 1
        tickets.append(
            {
                "scheduler_design_review_ticket_id": ticket.get(
                    "scheduler_design_review_ticket_id"
                ),
                "scheduler_design_hash": ticket.get("scheduler_design_hash"),
                "verified": verified,
                "revision_current": revision_current,
                "source_rank": int(source_rank),
            }
        )
    latest_verified = next((deepcopy(item) for item in tickets if item["verified"]), None)
    return {
        "surface": (
            "diagnostic_full_retained_sleep_plasticity_scheduler_design_"
            "review_ticket_queue.v1"
        ),
        "records_scanned": int(len(tickets)),
        "verified_count": int(verified_count),
        "stale_count": int(stale_count),
        "tampered_count": int(tampered_count),
        "latest_verified_ticket": latest_verified,
    }


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
        try:
            report["device_name"] = torch.cuda.get_device_name(0)
            report["memory_allocated_mib"] = round(
                float(torch.cuda.memory_allocated(0)) / (1024.0 * 1024.0),
                3,
            )
            report["memory_reserved_mib"] = round(
                float(torch.cuda.memory_reserved(0)) / (1024.0 * 1024.0),
                3,
            )
        except Exception as exc:
            report["cuda_query_error"] = str(exc)
    return report


def run_benchmark(*, retained_count: int, runs: int) -> dict[str, Any]:
    controller = _controller()
    seed_sleep_ticket = controller.record_snn_sleep_plasticity_review_ticket(
        sleep_policy=_sleep_policy(),
        operator_id="benchmark-sleep",
        confirmation=True,
    )
    sleep_records = _duplicate_records(
        seed_sleep_ticket,
        id_field="review_ticket_id",
        prefix="sleep-ticket",
        count=retained_count,
    )
    controller.snn_sleep_plasticity_review_tickets = sleep_records

    design = controller.snn_sleep_plasticity_scheduler_design(
        limit=64,
        cycles=3,
        min_stable_cycles=3,
        max_review_interval_seconds=120.0,
    )
    design_ticket = controller.record_snn_sleep_plasticity_scheduler_design_review_ticket(
        limit=64,
        cycles=3,
        min_stable_cycles=3,
        max_review_interval_seconds=120.0,
        expected_state_revision=controller._runtime_state.state_revision,  # noqa: SLF001
        scheduler_design_hash=design["provenance_evidence"]["scheduler_design_hash"],
        operator_id="benchmark-design",
        confirmation=True,
    )
    design_records = _duplicate_records(
        design_ticket,
        id_field="scheduler_design_review_ticket_id",
        prefix="design-ticket",
        count=retained_count,
    )
    controller.snn_sleep_plasticity_scheduler_design_review_tickets = design_records

    tracemalloc.start()
    sleep_bounded_latencies, sleep_bounded = _measure(
        lambda: controller.snn_sleep_plasticity_review_ticket_queue(limit=64),
        runs=runs,
    )
    design_bounded_latencies, design_bounded = _measure(
        lambda: controller.snn_sleep_plasticity_scheduler_design_review_ticket_queue(limit=64),
        runs=runs,
    )
    sleep_diagnostic_latencies, sleep_diagnostic = _measure(
        lambda: _diagnostic_sleep_ticket_queue(controller),
        runs=runs,
    )
    design_diagnostic_latencies, design_diagnostic = _measure(
        lambda: _diagnostic_scheduler_design_ticket_queue(controller),
        runs=runs,
    )
    current_mib, peak_mib = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    sleep_window = dict(sleep_bounded.get("source_window") or {})
    design_window = dict(design_bounded.get("source_window") or {})
    sleep_latest_bounded = (
        sleep_bounded.get("latest_verified_ticket")
        if isinstance(sleep_bounded.get("latest_verified_ticket"), Mapping)
        else {}
    )
    sleep_latest_diagnostic = (
        sleep_diagnostic.get("latest_verified_ticket")
        if isinstance(sleep_diagnostic.get("latest_verified_ticket"), Mapping)
        else {}
    )
    design_latest_bounded = (
        design_bounded.get("latest_verified_ticket")
        if isinstance(design_bounded.get("latest_verified_ticket"), Mapping)
        else {}
    )
    design_latest_diagnostic = (
        design_diagnostic.get("latest_verified_ticket")
        if isinstance(design_diagnostic.get("latest_verified_ticket"), Mapping)
        else {}
    )
    quality = {
        "sleep_latest_verified_matches_diagnostic": (
            sleep_latest_bounded.get("review_ticket_id")
            == sleep_latest_diagnostic.get("review_ticket_id")
        ),
        "design_latest_verified_matches_diagnostic": (
            design_latest_bounded.get("scheduler_design_review_ticket_id")
            == design_latest_diagnostic.get("scheduler_design_review_ticket_id")
        ),
        "sleep_window_bounded": (
            int(sleep_window.get("source_window_count", -1))
            <= SNN_SLEEP_PLASTICITY_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT
        ),
        "design_window_bounded": (
            int(design_window.get("source_window_count", -1))
            <= SNN_SLEEP_PLASTICITY_SCHEDULER_DESIGN_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT
        ),
        "sleep_work_reduced": (
            int(sleep_diagnostic.get("records_scanned", 0))
            > int(sleep_window.get("source_window_count", 0))
        ),
        "design_work_reduced": (
            int(design_diagnostic.get("records_scanned", 0))
            > int(design_window.get("source_window_count", 0))
        ),
        "no_global_scan": (
            sleep_window.get("global_candidate_scan") is False
            and sleep_window.get("global_score_scan") is False
            and design_window.get("global_candidate_scan") is False
            and design_window.get("global_score_scan") is False
        ),
        "not_live_tick": (
            sleep_window.get("runs_live_tick") is False
            and design_window.get("runs_live_tick") is False
        ),
        "not_every_token": (
            sleep_window.get("runs_every_token") is False
            and design_window.get("runs_every_token") is False
        ),
        "cpu_archival": (
            sleep_window.get("archival_storage_device") == "cpu"
            and design_window.get("archival_storage_device") == "cpu"
        ),
        "gpu_not_used": (
            sleep_window.get("gpu_used") is False
            and design_window.get("gpu_used") is False
        ),
        "no_mutation": (
            sleep_bounded.get("mutates_runtime_state") is False
            and design_bounded.get("mutates_runtime_state") is False
        ),
    }
    quality["pass"] = all(bool(value) for value in quality.values())
    sleep_source_count = max(1, int(sleep_window.get("source_window_count", 0)))
    design_source_count = max(1, int(design_window.get("source_window_count", 0)))
    return {
        "artifact_kind": "sleep_plasticity_ticket_queue_source_window_benchmark",
        "surface": "bounded_sleep_plasticity_ticket_queue_source_window.v1",
        "retained_count": int(retained_count),
        "runs": int(runs),
        "selection_criteria": [
            "newest_sleep_plasticity_review_tickets_first",
            "newest_scheduler_design_review_tickets_first",
            "current_revision_verified_non_executing_records",
            "source_window_before_scheduler_or_installation_proposal",
        ],
        "memory_budget": {
            "sleep_ticket_source_window_limit": (
                SNN_SLEEP_PLASTICITY_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT
            ),
            "scheduler_design_ticket_source_window_limit": (
                SNN_SLEEP_PLASTICITY_SCHEDULER_DESIGN_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT
            ),
            "retained_sleep_ticket_records": int(retained_count),
            "retained_scheduler_design_ticket_records": int(retained_count),
            "archival_storage_device": "cpu",
        },
        "device_placement": {
            "active_computation_device": "cpu",
            "archival_storage_device": "cpu",
            "gpu_resident_archival_metadata": False,
            "gpu_used": False,
        },
        "runtime_truth": {
            "runs_live_tick": False,
            "runs_every_token": False,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_replay_text_payload_loaded": False,
            "hidden_language_reasoning": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "records_replay_artifact": False,
            "installs_scheduler": False,
            "executes_suggested_endpoint": False,
        },
        "source_window": {
            "sleep_review_ticket_queue": sleep_window,
            "scheduler_design_review_ticket_queue": design_window,
        },
        "latency_ms": {
            "bounded_sleep_review_ticket_queue": _latency_stats(sleep_bounded_latencies),
            "diagnostic_full_sleep_review_ticket_queue": _latency_stats(
                sleep_diagnostic_latencies
            ),
            "bounded_scheduler_design_review_ticket_queue": _latency_stats(
                design_bounded_latencies
            ),
            "diagnostic_full_scheduler_design_review_ticket_queue": _latency_stats(
                design_diagnostic_latencies
            ),
        },
        "work_reduction": {
            "sleep_review_ticket_records": round(
                float(int(sleep_diagnostic.get("records_scanned", 0))) / sleep_source_count,
                6,
            ),
            "scheduler_design_review_ticket_records": round(
                float(int(design_diagnostic.get("records_scanned", 0))) / design_source_count,
                6,
            ),
        },
        "bounded": {
            "sleep_review_ticket_queue": sleep_bounded,
            "scheduler_design_review_ticket_queue": design_bounded,
        },
        "diagnostic": {
            "sleep_review_ticket_queue": sleep_diagnostic,
            "scheduler_design_review_ticket_queue": design_diagnostic,
        },
        "resource_behavior": {
            "python_tracemalloc_current_mib": round(float(current_mib) / (1024.0 * 1024.0), 3),
            "python_tracemalloc_peak_mib": round(float(peak_mib) / (1024.0 * 1024.0), 3),
            "cuda": _cuda_report(),
        },
        "quality": quality,
        "pass": bool(quality["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded sleep-plasticity ticket queue source windows."
    )
    parser.add_argument("--retained-count", type=int, default=64)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(
        retained_count=max(1, int(args.retained_count)),
        runs=max(1, int(args.runs)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "pass={passed} sleep_window={sleep_window} design_window={design_window} "
        "sleep_work_reduction={sleep_reduction} design_work_reduction={design_reduction}".format(
            passed=report["pass"],
            sleep_window=report["source_window"]["sleep_review_ticket_queue"][
                "source_window_count"
            ],
            design_window=report["source_window"][
                "scheduler_design_review_ticket_queue"
            ]["source_window_count"],
            sleep_reduction=report["work_reduction"]["sleep_review_ticket_records"],
            design_reduction=report["work_reduction"][
                "scheduler_design_review_ticket_records"
            ],
        )
    )


if __name__ == "__main__":
    main()
