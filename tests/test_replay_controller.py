from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import RLock
from typing import Mapping
import unittest
from unittest.mock import patch

from marulho.service.replay_runtime import (
    DEFAULT_REPLAY_SAMPLE_HISTORY,
    REPLAY_RESTORE_SOURCE_WINDOW_SURFACE,
    SNN_REPLAY_PRIORITY_CONTEXT_WINDOW_LIMIT,
    SNN_SLEEP_PLASTICITY_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT,
    SNN_SLEEP_PLASTICITY_SCHEDULER_DESIGN_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT,
    ReplayController,
    ReplayControllerDependencies,
    _snn_replay_priority_source_window_bounded,
)
from marulho.service.snn_language_plasticity_executor import (
    SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT,
)


@dataclass
class _FakeRuntimeState:
    state_revision: int = 7

    def __post_init__(self) -> None:
        self.dirty_without_revision_calls = 0

    def mark_dirty_without_revision(self) -> None:
        self.dirty_without_revision_calls += 1

    def mutation_summary(self) -> dict[str, int]:
        return {"dirty_state": True, "state_revision": self.state_revision}


class _FakeReplayManager:
    def __init__(self) -> None:
        self._lock = RLock()
        self._runtime_state = _FakeRuntimeState()
        self._replay_sample_history = deque(maxlen=256)
        self._trainer = type("_Trainer", (), {"token_count": 13})()
        self._action_history = deque(maxlen=24)

    @staticmethod
    def _normalize_action_text(value: object) -> str:
        return " ".join(str(value).split()).strip()

    @classmethod
    def _normalize_feedback_text(cls, value: object, *, max_chars: int = 2000) -> str:
        text = cls._normalize_action_text(value)
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + "…"
        return text

    def _living_loop_snapshot_locked(self) -> dict[str, object]:
        return {
            "state_revision": self._runtime_state.state_revision,
            "token_count": 13,
            "runtime_episode_count": 0,
            "action_count": 0,
            "prediction_count": 0,
            "runtime_episodes": [],
            "actions": [],
            "predictions": [],
            "uncertain_domains": [],
            "feedback_summary": {},
            "benchmark_telemetry": {},
            "memory_health": {"status": "available", "fill_ratio": 0.2},
            "subcortex_sleep_pressure": {"fatigue": 0.2},
            "policy_decision": {"action": "continue_current_policy"},
            "world_model_lite": {"uncertainty": 0.0},
        }

    @staticmethod
    def _replay_plan_summary(plan: dict[str, object] | None) -> dict[str, object]:
        payload = dict(plan or {})
        return {"endpoint": payload.get("endpoint", "/terminus/replay-plan"), "count": len(payload.get("candidates", []))}

    @staticmethod
    def _runtime_trace_export_safe_value(value: object) -> object:
        return value

    def _replay_sample_state_counts_locked(self) -> dict[str, int]:
        return {
            "token_count": 13,
            "state_revision": self._runtime_state.state_revision,
            "action_history_count": 0,
            "feedback_count": 0,
        }

    @staticmethod
    def _runtime_feedback_summary_locked() -> dict[str, int]:
        return {"feedback_count": 0}


class _IterationBlockedContexts:
    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("verified context lookup must use the controller index")


class _CountingIterable:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows
        self.iterated_count = 0

    def __iter__(self):  # type: ignore[no-untyped-def]
        for row in self._rows:
            self.iterated_count += 1
            yield row


def _replay_controller(manager: _FakeReplayManager, **kwargs: object) -> ReplayController:
    return ReplayController(
        ReplayControllerDependencies(
            action_history=lambda: manager._action_history,
            living_loop_snapshot=manager._living_loop_snapshot_locked,
            lock=manager._lock,
            normalize_action_text=manager._normalize_action_text,
            normalize_feedback_text=manager._normalize_feedback_text,
            replay_plan_summary=manager._replay_plan_summary,
            runtime_feedback_summary=manager._runtime_feedback_summary_locked,
            runtime_state=manager._runtime_state,
            runtime_trace_export_safe_value=manager._runtime_trace_export_safe_value,
            trainer=lambda: manager._trainer,
        ),
        **kwargs,
    )


def _known_readout_evidence_source_window() -> dict[str, object]:
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


def _readout_replay_priority_source_window() -> dict[str, object]:
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


def _with_replay_priority_source_window(
    controller: ReplayController,
    proposal: Mapping[str, object],
) -> dict[str, object]:
    source_window = _readout_replay_priority_source_window()
    payload = dict(proposal)
    payload["replay_priority_source_window"] = source_window
    payload["replay_priority_source_window_hash"] = controller._sha256_json(  # noqa: SLF001
        source_window
    )
    return payload


class ReplayControllerTests(unittest.TestCase):
    @staticmethod
    def _mismatch_report() -> dict[str, object]:
        return {
            "surface": "snn_language_sequence_mismatch_probe.v1",
            "available": True,
            "owned_by_marulho": True,
            "prediction_error": {"mismatch_score": 0.9},
            "promotion_gate": {"status": "ready_for_operator_review"},
        }

    @staticmethod
    def _pressure_report() -> dict[str, object]:
        return {
            "surface": "snn_language_plasticity_pressure.v1",
            "available": True,
            "owned_by_marulho": True,
            "promotion_gate": {"status": "ready_for_operator_review"},
        }

    @staticmethod
    def _sleep_policy() -> dict[str, object]:
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
                "suggested_endpoint": "/terminus/snn-language-sequence/plasticity-homeostatic-maintenance",
                "requires_operator_confirmation": True,
                "executable": False,
                "reason_codes": ["post_growth_homeostatic_maintenance_due"],
            },
        }

    @classmethod
    def _record_replay_evaluation_context(
        cls,
        controller: ReplayController,
        *,
        source_metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        return controller.record_snn_replay_evaluation_context(
            mismatch_report=cls._mismatch_report(),
            pressure_report=cls._pressure_report(),
            source_metadata=source_metadata,
        )

    @staticmethod
    def _record_review_ticket(
        controller: ReplayController,
        *,
        operator_id: str = "operator-1",
    ) -> dict[str, object]:
        queue = controller.snn_replay_consolidation_priority_queue(
            readout_replay_priority_report={
                "surface": "snn_language_readout_replay_priority.v1",
                "candidates": [{"priority_score": 90.0, "all_labels_grounded": True}],
            },
            limit=4,
        )
        proposal = controller.snn_replay_artifact_recording_policy_proposal(
            consolidation_priority_queue=queue,
            policy={"min_priority_score": 60.0},
        )
        return controller.record_snn_replay_artifact_recording_review_ticket(
            policy_proposal=proposal,
            operator_id=operator_id,
            confirmation=True,
        )

    @staticmethod
    def _record_regeneration_replay_artifact(
        controller: ReplayController,
        *,
        operator_id: str = "operator-1",
    ) -> dict[str, object]:
        context = ReplayControllerTests._record_replay_evaluation_context(controller)
        ticket = ReplayControllerTests._record_review_ticket(controller, operator_id=operator_id)
        return controller.record_evaluated_snn_transition_memory_replay_artifact(
            artifact_proposal=_with_replay_priority_source_window(
                controller,
                {
                    "surface": "snn_transition_memory_replay_artifact_proposal.v1",
                    "ready": True,
                    "owned_by_marulho": True,
                    "source": "service.snn_language_readout_ledger.transition_memory_replay_artifact_proposal",
                    "mismatch_report": context["mismatch_report"],
                    "pressure_report": context["pressure_report"],
                    "replay_evaluation_context_id": context["replay_evaluation_context_id"],
                    "replay_evaluation_context_hash": context["evidence_hash"],
                    "replay_window": [
                        {"readout_evidence_hash": "readout-hash-1", "grounded": True}
                    ],
                    "promotion_gate": {"status": "ready_for_operator_recording_review"},
                },
            ),
            known_readout_evidence_hashes={"readout-hash-1"},
            known_readout_evidence_source_window=_known_readout_evidence_source_window(),
            replay_evaluation_context_id=str(context["replay_evaluation_context_id"]),
            review_ticket_id=str(ticket["review_ticket_id"]),
            operator_id=operator_id,
            confirmation=True,
        )

    def test_replay_sample_history_is_controller_owned(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        controller.history.appendleft(
            {
                "schema_version": 1,
                "replay_sample_id": "replay-sample-1",
                "mode": "sample",
                "status": "recorded",
                "selected_candidate_ids": ["candidate-1"],
                "selected_candidates": [],
                "safety_flags": {"audit_only": True, "operator_confirmed": True},
            }
        )

        history = controller.replay_sample_history(limit=1)

        self.assertEqual(history["count"], 1)
        self.assertEqual(history["history"][0]["replay_sample_id"], "replay-sample-1")
        self.assertEqual(controller.history[0]["replay_sample_id"], "replay-sample-1")

    def test_history_setter_preserves_existing_deque_reference(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)

        history = controller.history
        controller.history = [
            {
                "schema_version": 1,
                "replay_sample_id": "replay-sample-1",
                "mode": "sample",
                "status": "recorded",
                "selected_candidate_ids": ["candidate-1"],
                "selected_candidates": [],
                "safety_flags": {"audit_only": True, "operator_confirmed": True},
            }
        ]

        self.assertIs(controller.history, history)
        self.assertEqual(history[0]["replay_sample_id"], "replay-sample-1")

    def test_replay_state_restore_uses_bounded_source_window_before_normalization(self) -> None:
        manager = _FakeReplayManager()
        records = [
            {
                "schema_version": 1,
                "replay_sample_id": f"replay-sample-{index:04d}",
                "mode": "sample",
                "status": "recorded",
                "selected_candidate_ids": [f"candidate-{index:04d}"],
                "selected_candidates": [],
                "safety_flags": {"audit_only": True, "operator_confirmed": True},
            }
            for index in range(DEFAULT_REPLAY_SAMPLE_HISTORY + 17)
        ]
        source = _CountingIterable(records)

        controller = _replay_controller(manager, replay_sample_history=source)

        self.assertEqual(source.iterated_count, DEFAULT_REPLAY_SAMPLE_HISTORY)
        self.assertEqual(len(controller.history), DEFAULT_REPLAY_SAMPLE_HISTORY)
        self.assertEqual(controller.history[0]["replay_sample_id"], "replay-sample-0000")
        self.assertEqual(
            controller.history[-1]["replay_sample_id"],
            f"replay-sample-{DEFAULT_REPLAY_SAMPLE_HISTORY - 1:04d}",
        )
        report = controller.replay_restore_source_window_report()
        self.assertEqual(report["surface"], REPLAY_RESTORE_SOURCE_WINDOW_SURFACE)
        self.assertFalse(report["full_retained_materialization"])
        self.assertFalse(report["runs_live_tick"])
        self.assertFalse(report["runs_every_token"])
        self.assertFalse(report["gpu_used"])
        self.assertFalse(report["language_reasoning"])
        field = report["fields"]["replay_sample_history"]
        self.assertEqual(field["source_window_inspected_count"], DEFAULT_REPLAY_SAMPLE_HISTORY)
        self.assertEqual(field["normalized_count"], DEFAULT_REPLAY_SAMPLE_HISTORY)
        self.assertFalse(field["source_record_count_known"])
        self.assertFalse(field["normalizes_full_retained_state"])
        self.assertTrue(field["indexes_only_restored_window"])

    def test_replay_state_restore_reports_truncated_checkpoint_sequences(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        records = [
            {
                "schema_version": 1,
                "replay_sample_id": f"replay-sample-{index:04d}",
                "mode": "sample",
                "status": "recorded",
                "selected_candidate_ids": [f"candidate-{index:04d}"],
                "selected_candidates": [],
                "safety_flags": {"audit_only": True, "operator_confirmed": True},
            }
            for index in range(DEFAULT_REPLAY_SAMPLE_HISTORY + 3)
        ]

        controller.history = records

        report = controller.replay_restore_source_window_report()
        field = report["fields"]["replay_sample_history"]
        self.assertTrue(field["source_record_count_known"])
        self.assertEqual(field["source_record_count"], DEFAULT_REPLAY_SAMPLE_HISTORY + 3)
        self.assertTrue(field["source_window_truncated"])
        self.assertEqual(field["source_truncated_count"], 3)
        self.assertEqual(len(controller.history), DEFAULT_REPLAY_SAMPLE_HISTORY)

    def test_snn_transition_memory_replay_artifact_setter_preserves_existing_deque_reference_and_drops_raw_artifacts(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        artifacts = controller.snn_transition_memory_replay_artifacts
        artifact = self._record_regeneration_replay_artifact(controller)

        controller.snn_transition_memory_replay_artifacts = [
            {"replay_artifact_id": "raw-artifact-1", "evidence_hash": "hash-1"},
            artifact,
        ]

        self.assertIs(controller.snn_transition_memory_replay_artifacts, artifacts)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0]["replay_artifact_id"], artifact["replay_artifact_id"])
        self.assertTrue(artifacts[0]["internal_ledger_backed"])

    def test_snn_replay_evaluation_context_setter_preserves_existing_deque_reference(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        contexts = controller.snn_replay_evaluation_contexts

        controller.snn_replay_evaluation_contexts = [
            {"replay_evaluation_context_id": "context-1", "evidence_hash": "hash-1"}
        ]

        self.assertIs(controller.snn_replay_evaluation_contexts, contexts)
        self.assertEqual(contexts[0]["replay_evaluation_context_id"], "context-1")

    def test_snn_replay_artifact_recording_review_ticket_setter_preserves_existing_deque_reference(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        tickets = controller.snn_replay_artifact_recording_review_tickets

        controller.snn_replay_artifact_recording_review_tickets = [
            {"review_ticket_id": "ticket-1", "evidence_hash": "hash-1"}
        ]

        self.assertIs(controller.snn_replay_artifact_recording_review_tickets, tickets)
        self.assertEqual(tickets[0]["review_ticket_id"], "ticket-1")

    def test_snn_sleep_plasticity_review_ticket_setter_preserves_existing_deque_reference(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        tickets = controller.snn_sleep_plasticity_review_tickets

        controller.snn_sleep_plasticity_review_tickets = [
            {"review_ticket_id": "sleep-ticket-1", "evidence_hash": "hash-1"}
        ]

        self.assertIs(controller.snn_sleep_plasticity_review_tickets, tickets)
        self.assertEqual(tickets[0]["review_ticket_id"], "sleep-ticket-1")

    def test_snn_sleep_plasticity_scheduler_design_review_ticket_setter_preserves_existing_deque_reference(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        tickets = controller.snn_sleep_plasticity_scheduler_design_review_tickets

        controller.snn_sleep_plasticity_scheduler_design_review_tickets = [
            {"scheduler_design_review_ticket_id": "design-ticket-1", "evidence_hash": "hash-1"}
        ]

        self.assertIs(controller.snn_sleep_plasticity_scheduler_design_review_tickets, tickets)
        self.assertEqual(tickets[0]["scheduler_design_review_ticket_id"], "design-ticket-1")

    def test_snn_sleep_plasticity_ticket_queues_reload_as_bounded_source_windows(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        sleep_ticket = controller.record_snn_sleep_plasticity_review_ticket(
            sleep_policy=self._sleep_policy(),
            operator_id="operator-sleep",
            confirmation=True,
        )
        sleep_records = []
        for index in range(64):
            record = deepcopy(sleep_ticket)
            record["review_ticket_id"] = f"sleep-ticket-{index:04d}"
            sleep_records.append(record)
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
            expected_state_revision=manager._runtime_state.state_revision,
            scheduler_design_hash=design["provenance_evidence"]["scheduler_design_hash"],
            operator_id="operator-scheduler-design",
            confirmation=True,
        )
        design_records = []
        for index in range(64):
            record = deepcopy(design_ticket)
            record["scheduler_design_review_ticket_id"] = f"design-ticket-{index:04d}"
            design_records.append(record)

        reloaded_manager = _FakeReplayManager()
        reloaded = _replay_controller(
            reloaded_manager,
            snn_sleep_plasticity_review_tickets=sleep_records,
            snn_sleep_plasticity_scheduler_design_review_tickets=design_records,
        )

        sleep_queue = reloaded.snn_sleep_plasticity_review_ticket_queue(limit=64)
        self.assertEqual(sleep_queue["retained_count"], 64)
        self.assertEqual(
            sleep_queue["count"],
            SNN_SLEEP_PLASTICITY_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT,
        )
        self.assertEqual(
            sleep_queue["source_window"]["source_window_count"],
            SNN_SLEEP_PLASTICITY_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT,
        )
        self.assertEqual(
            sleep_queue["source_window"]["source_window_inspected_count"],
            SNN_SLEEP_PLASTICITY_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT,
        )
        self.assertEqual(sleep_queue["source_window"]["source_truncated_count"], 48)
        self.assertEqual(
            sleep_queue["latest_verified_ticket"]["review_ticket_id"],
            "sleep-ticket-0000",
        )
        self.assertEqual(
            [ticket["source_rank"] for ticket in sleep_queue["tickets"]],
            list(range(SNN_SLEEP_PLASTICITY_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT)),
        )
        self.assertFalse(sleep_queue["source_window"]["global_candidate_scan"])
        self.assertFalse(sleep_queue["source_window"]["runs_live_tick"])
        self.assertFalse(sleep_queue["source_window"]["runs_every_token"])
        self.assertEqual(sleep_queue["source_window"]["archival_storage_device"], "cpu")

        design_queue = (
            reloaded.snn_sleep_plasticity_scheduler_design_review_ticket_queue(limit=64)
        )
        self.assertEqual(design_queue["retained_count"], 64)
        self.assertEqual(
            design_queue["count"],
            SNN_SLEEP_PLASTICITY_SCHEDULER_DESIGN_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT,
        )
        self.assertEqual(
            design_queue["source_window"]["source_window_count"],
            SNN_SLEEP_PLASTICITY_SCHEDULER_DESIGN_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT,
        )
        self.assertEqual(
            design_queue["source_window"]["source_window_inspected_count"],
            SNN_SLEEP_PLASTICITY_SCHEDULER_DESIGN_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT,
        )
        self.assertEqual(design_queue["source_window"]["source_truncated_count"], 48)
        self.assertEqual(
            design_queue["latest_verified_ticket"]["scheduler_design_review_ticket_id"],
            "design-ticket-0000",
        )
        self.assertEqual(
            [ticket["source_rank"] for ticket in design_queue["tickets"]],
            list(
                range(
                    SNN_SLEEP_PLASTICITY_SCHEDULER_DESIGN_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT
                )
            ),
        )
        self.assertFalse(design_queue["source_window"]["global_candidate_scan"])
        self.assertFalse(design_queue["source_window"]["runs_live_tick"])
        self.assertFalse(design_queue["source_window"]["runs_every_token"])
        self.assertEqual(design_queue["source_window"]["archival_storage_device"], "cpu")

    def test_snn_sleep_plasticity_review_scheduler_installation_setter_preserves_existing_deque_reference(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        installations = controller.snn_sleep_plasticity_review_scheduler_installations

        controller.snn_sleep_plasticity_review_scheduler_installations = [
            {"scheduler_installation_id": "scheduler-1", "evidence_hash": "hash-1"}
        ]

        self.assertIs(
            controller.snn_sleep_plasticity_review_scheduler_installations,
            installations,
        )
        self.assertEqual(installations[0]["scheduler_installation_id"], "scheduler-1")

    def test_snn_replay_evaluation_context_verification_fails_closed(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        context = self._record_replay_evaluation_context(controller)
        context_id = str(context["replay_evaluation_context_id"])

        self.assertIsNotNone(controller.verified_snn_replay_evaluation_context(context_id))
        controller.snn_replay_evaluation_contexts[0]["mismatch_hash"] = "tampered"
        self.assertIsNone(controller.verified_snn_replay_evaluation_context(context_id))
        controller.snn_replay_evaluation_contexts[0] = context
        manager._runtime_state.state_revision += 1
        self.assertIsNone(controller.verified_snn_replay_evaluation_context(context_id))

    def test_snn_replay_evaluation_context_verifies_source_metadata(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        context = controller.record_snn_replay_evaluation_context(
            mismatch_report=self._mismatch_report(),
            pressure_report=self._pressure_report(),
            source_metadata={
                "source": "test",
                "emission_hash": "emission-hash",
                "readout_evidence_hash": "readout-hash",
            },
        )
        context_id = str(context["replay_evaluation_context_id"])

        verified = controller.verified_snn_replay_evaluation_context(context_id)
        self.assertIsNotNone(verified)
        assert verified is not None
        self.assertEqual(verified["source_metadata"]["emission_hash"], "emission-hash")
        self.assertTrue(verified["source_metadata_hash"])
        controller.snn_replay_evaluation_contexts[0]["source_metadata"][
            "emission_hash"
        ] = "tampered"
        self.assertIsNone(controller.verified_snn_replay_evaluation_context(context_id))

    def test_snn_replay_evaluation_context_verification_uses_id_index(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        context = self._record_replay_evaluation_context(controller)
        context_id = str(context["replay_evaluation_context_id"])
        controller._snn_replay_evaluation_contexts = _IterationBlockedContexts()  # type: ignore[assignment]

        verified = controller.verified_snn_replay_evaluation_context(context_id)

        self.assertIsNotNone(verified)
        assert verified is not None
        self.assertEqual(verified["replay_evaluation_context_id"], context_id)

    def test_snn_replay_artifact_review_ticket_verification_uses_id_indexes(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        self._record_replay_evaluation_context(controller)
        ticket = self._record_review_ticket(controller)
        ticket_id = str(ticket["review_ticket_id"])
        source_window = dict(ticket["source_window"])

        self.assertEqual(
            source_window["surface"],
            "bounded_snn_replay_artifact_provenance_source_window.v1",
        )
        self.assertEqual(source_window["policy"], "indexed_context_ticket_artifact_permit_window_v1")
        self.assertFalse(source_window["global_candidate_scan"])
        self.assertFalse(source_window["runs_live_tick"])
        self.assertFalse(source_window["gpu_used"])
        controller._snn_replay_evaluation_contexts = _IterationBlockedContexts()  # type: ignore[assignment]
        controller._snn_replay_artifact_recording_review_tickets = _IterationBlockedContexts()  # type: ignore[assignment]

        verified = controller.verified_snn_replay_artifact_recording_review_ticket(
            ticket_id,
            operator_id="operator-1",
        )

        self.assertIsNotNone(verified)
        assert verified is not None
        self.assertEqual(verified["review_ticket_id"], ticket_id)

    def test_snn_replay_priority_source_window_validator_requires_explicit_flags(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        self._record_replay_evaluation_context(controller)
        queue = controller.snn_replay_consolidation_priority_queue(
            readout_replay_priority_report={
                "surface": "snn_language_readout_replay_priority.v1",
                "candidates": [{"priority_score": 90.0, "all_labels_grounded": True}],
            },
            limit=4,
        )
        source_window = dict(queue["source_window"])

        self.assertTrue(_snn_replay_priority_source_window_bounded(source_window))
        missing_flag = dict(source_window)
        missing_flag.pop("raw_text_payload_loaded")
        self.assertFalse(_snn_replay_priority_source_window_bounded(missing_flag))
        wrong_device = dict(source_window)
        wrong_device["archival_storage_device"] = "cuda"
        self.assertFalse(_snn_replay_priority_source_window_bounded(wrong_device))

    def test_evaluated_replay_artifact_rejects_stale_context(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        context = self._record_replay_evaluation_context(controller)
        manager._runtime_state.state_revision += 1

        with self.assertRaisesRegex(ValueError, "verified server-held evaluation context"):
            controller.record_evaluated_snn_transition_memory_replay_artifact(
                artifact_proposal=_with_replay_priority_source_window(
                    controller,
                    {
                        "surface": "snn_transition_memory_replay_artifact_proposal.v1",
                        "ready": True,
                        "owned_by_marulho": True,
                        "source": "service.snn_language_readout_ledger.transition_memory_replay_artifact_proposal",
                        "mismatch_report": context["mismatch_report"],
                        "pressure_report": context["pressure_report"],
                        "replay_evaluation_context_id": context["replay_evaluation_context_id"],
                        "replay_evaluation_context_hash": context["evidence_hash"],
                        "replay_window": [{"readout_evidence_hash": "readout-hash-1", "grounded": True}],
                        "promotion_gate": {"status": "ready_for_operator_recording_review"},
                    },
                ),
                known_readout_evidence_hashes={"readout-hash-1"},
                known_readout_evidence_source_window=_known_readout_evidence_source_window(),
                replay_evaluation_context_id=str(context["replay_evaluation_context_id"]),
                review_ticket_id="stale-context-ticket",
                operator_id="operator-1",
                confirmation=True,
            )

    def test_snn_replay_consolidation_priority_queue_is_advisory_and_current_revision_only(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        stale_context = self._record_replay_evaluation_context(controller)
        manager._runtime_state.state_revision += 1
        current_context = self._record_replay_evaluation_context(controller)

        priority = controller.snn_replay_consolidation_priority_queue(
            readout_replay_priority_report={
                "surface": "snn_language_readout_replay_priority.v1",
                "candidates": [
                    {
                        "priority_score": 88.0,
                        "all_labels_grounded": True,
                    }
                ],
            },
            limit=4,
        )

        self.assertEqual(priority["surface"], "snn_replay_consolidation_priority_queue.v1")
        self.assertTrue(priority["advisory"])
        self.assertFalse(priority["executable"])
        self.assertFalse(priority["mutates_runtime_state"])
        self.assertFalse(priority["eligible_for_live_replay"])
        self.assertFalse(priority["eligible_for_artifact_recording"])
        self.assertFalse(priority["promotion_gate"]["eligible_for_artifact_recording"])
        self.assertFalse(priority["promotion_gate"]["eligible_for_live_replay"])
        self.assertEqual(priority["candidate_count"], 1)
        self.assertEqual(
            priority["candidates"][0]["replay_evaluation_context_id"],
            current_context["replay_evaluation_context_id"],
        )
        self.assertNotEqual(
            priority["candidates"][0]["replay_evaluation_context_id"],
            stale_context["replay_evaluation_context_id"],
        )
        self.assertFalse(priority["candidates"][0]["eligible_for_live_replay"])
        self.assertFalse(priority["candidates"][0]["eligible_for_artifact_recording"])
        self.assertGreater(priority["candidates"][0]["priority_score"], 0.0)

    def test_snn_replay_consolidation_priority_queue_uses_bounded_source_window(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        old_context = self._record_replay_evaluation_context(
            controller,
            source_metadata={"source": "old-readout-target"},
        )
        for index in range(SNN_REPLAY_PRIORITY_CONTEXT_WINDOW_LIMIT + 2):
            self._record_replay_evaluation_context(
                controller,
                source_metadata={"source": f"recent-context-{index}"},
            )
        old_context_id = str(old_context["replay_evaluation_context_id"])

        priority = controller.snn_replay_consolidation_priority_queue(
            readout_replay_priority_report={
                "surface": "snn_language_readout_replay_priority.v1",
                "candidates": [
                    {
                        "priority_score": 99.0,
                        "all_labels_grounded": True,
                        "replay_evaluation_context_id": old_context_id,
                    }
                ],
            },
            limit=4,
        )

        source_window = priority["source_window"]
        self.assertEqual(
            source_window["surface"],
            "bounded_snn_replay_priority_source_window.v1",
        )
        self.assertEqual(
            source_window["recent_context_window_count"],
            SNN_REPLAY_PRIORITY_CONTEXT_WINDOW_LIMIT,
        )
        self.assertEqual(source_window["readout_target_context_count"], 1)
        self.assertTrue(source_window["source_context_count_is_lower_bound"])
        self.assertFalse(source_window["global_candidate_scan"])
        self.assertFalse(source_window["runs_live_tick"])
        self.assertFalse(source_window["gpu_used"])
        candidate_by_id = {
            str(item["replay_evaluation_context_id"]): item
            for item in priority["candidates"]
        }
        self.assertIn(old_context_id, candidate_by_id)
        self.assertEqual(
            candidate_by_id[old_context_id]["source_window"]["source"],
            "readout_priority_target_context_id",
        )
        self.assertIn(
            "readout_target_context_id",
            candidate_by_id[old_context_id]["reason_codes"],
        )

    def test_snn_replay_artifact_recording_policy_proposal_is_advisory(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        context = self._record_replay_evaluation_context(controller)
        queue = controller.snn_replay_consolidation_priority_queue(
            readout_replay_priority_report={
                "surface": "snn_language_readout_replay_priority.v1",
                "candidates": [
                    {
                        "priority_score": 90.0,
                        "all_labels_grounded": True,
                    }
                ],
            },
            limit=4,
        )

        proposal = controller.snn_replay_artifact_recording_policy_proposal(
            consolidation_priority_queue=queue,
            policy={"min_priority_score": 60.0},
        )

        self.assertEqual(
            proposal["surface"],
            "snn_replay_artifact_recording_policy_proposal.v1",
        )
        self.assertTrue(proposal["recommended"])
        self.assertTrue(proposal["advisory"])
        self.assertFalse(proposal["executable"])
        self.assertFalse(proposal["mutates_runtime_state"])
        self.assertFalse(proposal["eligible_for_artifact_recording"])
        self.assertFalse(proposal["promotion_gate"]["eligible_for_artifact_recording"])
        self.assertEqual(
            proposal["recommended_review"]["replay_evaluation_context_id"],
            context["replay_evaluation_context_id"],
        )
        self.assertEqual(manager._runtime_state.dirty_without_revision_calls, 1)

    def test_snn_replay_artifact_recording_review_preserves_context_lineage(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        source_metadata = {
            "source": "runtime_facade.snn_language_readout_emission_replay_context_review",
            "surface": "snn_language_readout_emission_replay_context_review.v1",
            "design_hash": "design-hash",
            "seed_hash": "seed-hash",
            "emission_review_hash": "emission-review-hash",
            "emission_hash": "emission-hash",
            "readout_evidence_hash": "readout-hash",
            "prediction_hash": "prediction-hash",
            "operator_id": "operator-lineage",
        }
        context = self._record_replay_evaluation_context(
            controller,
            source_metadata=source_metadata,
        )
        queue = controller.snn_replay_consolidation_priority_queue(
            readout_replay_priority_report={
                "surface": "snn_language_readout_replay_priority.v1",
                "candidates": [
                    {
                        "priority_score": 90.0,
                        "all_labels_grounded": True,
                    }
                ],
            },
            limit=4,
        )

        candidate = queue["candidates"][0]
        self.assertEqual(candidate["source_metadata_hash"], context["source_metadata_hash"])
        self.assertEqual(candidate["emission_lineage"]["emission_hash"], "emission-hash")
        self.assertEqual(candidate["emission_lineage"]["readout_evidence_hash"], "readout-hash")
        self.assertEqual(candidate["emission_lineage"]["prediction_hash"], "prediction-hash")
        self.assertNotIn("source_metadata", candidate)
        self.assertNotIn("operator_id", candidate["emission_lineage"])
        self.assertTrue(candidate["emission_lineage_available"])

        proposal = controller.snn_replay_artifact_recording_policy_proposal(
            consolidation_priority_queue=queue,
            policy={"min_priority_score": 60.0},
        )
        self.assertTrue(
            proposal["promotion_gate"]["required_evidence"][
                "candidate_lineage_matches_verified_context"
            ]
        )
        self.assertEqual(
            proposal["recommended_review"]["source_metadata_hash"],
            context["source_metadata_hash"],
        )
        self.assertEqual(
            proposal["recommended_review"]["emission_lineage"],
            candidate["emission_lineage"],
        )

        ticket = controller.record_snn_replay_artifact_recording_review_ticket(
            policy_proposal=proposal,
            operator_id="operator-lineage",
            confirmation=True,
        )
        self.assertEqual(ticket["source_metadata_hash"], context["source_metadata_hash"])
        self.assertEqual(ticket["emission_lineage"], candidate["emission_lineage"])
        self.assertIsNotNone(
            controller.verified_snn_replay_artifact_recording_review_ticket(
                str(ticket["review_ticket_id"]),
                operator_id="operator-lineage",
            )
        )

        controller.snn_replay_artifact_recording_review_tickets[0]["emission_lineage"][
            "emission_hash"
        ] = "tampered"
        self.assertIsNone(
            controller.verified_snn_replay_artifact_recording_review_ticket(
                str(ticket["review_ticket_id"]),
                operator_id="operator-lineage",
            )
        )

    def test_snn_replay_artifact_recording_review_ticket_verification_fails_closed(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        self._record_replay_evaluation_context(controller)
        ticket = self._record_review_ticket(controller)
        ticket_id = str(ticket["review_ticket_id"])

        self.assertIsNotNone(
            controller.verified_snn_replay_artifact_recording_review_ticket(
                ticket_id,
                operator_id="operator-1",
            )
        )
        self.assertIsNone(
            controller.verified_snn_replay_artifact_recording_review_ticket(
                ticket_id,
                operator_id="operator-2",
            )
        )
        controller.snn_replay_artifact_recording_review_tickets[0]["policy_proposal_hash"] = "tampered"
        self.assertIsNone(
            controller.verified_snn_replay_artifact_recording_review_ticket(
                ticket_id,
                operator_id="operator-1",
            )
        )
        controller.snn_replay_artifact_recording_review_tickets[0] = ticket
        manager._runtime_state.state_revision += 1
        self.assertIsNone(
            controller.verified_snn_replay_artifact_recording_review_ticket(
                ticket_id,
                operator_id="operator-1",
            )
        )

    def test_snn_sleep_plasticity_review_ticket_is_controller_owned_and_revision_bound(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)

        ticket = controller.record_snn_sleep_plasticity_review_ticket(
            sleep_policy=self._sleep_policy(),
            operator_id="operator-sleep",
            confirmation=True,
        )
        ticket_id = str(ticket["review_ticket_id"])

        self.assertEqual(ticket["surface"], "snn_sleep_plasticity_review_ticket.v1")
        self.assertEqual(ticket["recommended_action"], "review_transition_memory_homeostatic_maintenance")
        self.assertEqual(ticket["review_gate_key"], "transition_memory_homeostatic_maintenance_review")
        self.assertEqual(ticket["operator_id"], "operator-sleep")
        self.assertFalse(ticket["executable"])
        self.assertFalse(ticket["mutates_runtime_state"])
        self.assertFalse(ticket["applies_plasticity"])
        self.assertFalse(ticket["writes_checkpoint"])
        self.assertFalse(ticket["records_replay_artifact"])
        self.assertFalse(ticket["issues_regeneration_permit"])
        self.assertIsNotNone(
            controller.verified_snn_sleep_plasticity_review_ticket(
                ticket_id,
                operator_id="operator-sleep",
            )
        )
        self.assertEqual(manager._runtime_state.dirty_without_revision_calls, 1)
        queue = controller.snn_sleep_plasticity_review_ticket_queue(limit=4)
        self.assertEqual(queue["surface"], "snn_sleep_plasticity_review_ticket_queue.v1")
        self.assertTrue(queue["ready"])
        self.assertEqual(queue["verified_count"], 1)
        self.assertEqual(queue["stale_count"], 0)
        self.assertEqual(
            queue["next_gate"],
            "/terminus/snn-language-sequence/plasticity-homeostatic-maintenance",
        )
        self.assertEqual(
            queue["pending_action_counts"]["review_transition_memory_homeostatic_maintenance"],
            1,
        )
        self.assertFalse(queue["executable"])
        self.assertFalse(queue["mutates_runtime_state"])
        self.assertFalse(queue["applies_plasticity"])
        sleep_queue_window = queue["source_window"]
        self.assertEqual(
            sleep_queue_window["surface"],
            "bounded_snn_sleep_plasticity_review_ticket_queue_source_window.v1",
        )
        self.assertEqual(
            sleep_queue_window["source_window_limit"],
            SNN_SLEEP_PLASTICITY_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT,
        )
        self.assertEqual(sleep_queue_window["source_window_count"], 1)
        self.assertEqual(queue["retained_count"], 1)
        self.assertFalse(sleep_queue_window["global_candidate_scan"])
        self.assertFalse(sleep_queue_window["runs_live_tick"])
        self.assertFalse(sleep_queue_window["runs_every_token"])
        self.assertFalse(sleep_queue_window["gpu_used"])
        proposal = controller.snn_sleep_plasticity_autonomy_proposal(limit=4)
        self.assertEqual(proposal["surface"], "snn_sleep_plasticity_autonomy_proposal.v1")
        self.assertTrue(proposal["ready"])
        self.assertEqual(proposal["candidate"]["action"], "review_sleep_plasticity_next_gate")
        self.assertEqual(proposal["candidate"]["review_ticket_id"], ticket_id)
        self.assertEqual(
            proposal["promotion_gate"]["next_gate"],
            "/terminus/snn-language-sequence/plasticity-homeostatic-maintenance",
        )
        self.assertTrue(proposal["promotion_gate"]["eligible_for_autonomy_planning"])
        self.assertFalse(proposal["promotion_gate"]["eligible_for_action"])
        self.assertFalse(proposal["promotion_gate"]["eligible_for_structural_write"])
        self.assertFalse(proposal["executable"])
        self.assertFalse(proposal["mutates_runtime_state"])
        self.assertFalse(proposal["applies_plasticity"])
        revision_before_experiment = manager._runtime_state.state_revision
        dirty_calls_before_experiment = manager._runtime_state.dirty_without_revision_calls
        ticket_count_before_experiment = len(controller.snn_sleep_plasticity_review_tickets)
        experiment = controller.snn_sleep_plasticity_scheduler_experiment(
            limit=4,
            cycles=3,
        )
        repeated_experiment = controller.snn_sleep_plasticity_scheduler_experiment(
            limit=4,
            cycles=3,
        )
        self.assertEqual(experiment["surface"], "snn_sleep_plasticity_scheduler_experiment.v1")
        self.assertTrue(experiment["ready"])
        self.assertTrue(experiment["isolated"])
        self.assertEqual(experiment["experiment_summary"]["cycle_count"], 3)
        self.assertEqual(experiment["experiment_summary"]["stable_cycle_count"], 3)
        self.assertTrue(experiment["experiment_summary"]["proposal_stable"])
        self.assertEqual(experiment["experiment_summary"]["review_ticket_id"], ticket_id)
        self.assertFalse(experiment["device_evidence"]["tensor_execution_required"])
        self.assertFalse(experiment["device_evidence"]["cuda_applicable"])
        self.assertFalse(experiment["external_dependency"])
        self.assertFalse(experiment["loads_external_checkpoint"])
        self.assertFalse(experiment["returns_trained_weights"])
        self.assertFalse(experiment["eligible_for_plasticity"])
        self.assertEqual(
            repeated_experiment["provenance_evidence"]["scheduler_experiment_hash"],
            experiment["provenance_evidence"]["scheduler_experiment_hash"],
        )
        self.assertEqual(
            repeated_experiment["provenance_evidence"]["scheduler_experiment_id"],
            experiment["provenance_evidence"]["scheduler_experiment_id"],
        )
        self.assertFalse(experiment["executes_suggested_endpoint"])
        self.assertFalse(experiment["ephemeral_experiment"]["scheduler_installed"])
        self.assertFalse(experiment["ephemeral_experiment"]["suggested_endpoint_called"])
        self.assertTrue(
            experiment["promotion_gate"]["eligible_for_operator_scheduler_design_review"]
        )
        self.assertFalse(experiment["promotion_gate"]["eligible_for_scheduler_installation"])
        self.assertFalse(experiment["promotion_gate"]["eligible_for_action"])
        self.assertFalse(experiment["promotion_gate"]["eligible_for_structural_write"])
        design = controller.snn_sleep_plasticity_scheduler_design(
            limit=4,
            cycles=3,
            min_stable_cycles=3,
            max_review_interval_seconds=120.0,
        )
        repeated_design = controller.snn_sleep_plasticity_scheduler_design(
            limit=4,
            cycles=3,
            min_stable_cycles=3,
            max_review_interval_seconds=120.0,
        )
        self.assertEqual(design["surface"], "snn_sleep_plasticity_scheduler_design.v1")
        self.assertTrue(design["ready"])
        self.assertTrue(design["isolated"])
        self.assertEqual(design["scheduler_design"]["scheduler_mode"], "operator_review_only")
        self.assertEqual(design["scheduler_design"]["review_ticket_id"], ticket_id)
        self.assertEqual(
            design["scheduler_design"]["bound_state_revision"],
            manager._runtime_state.state_revision,
        )
        self.assertEqual(design["scheduler_design"]["min_stable_cycles"], 3)
        self.assertEqual(design["scheduler_design"]["observed_stable_cycles"], 3)
        self.assertEqual(design["scheduler_design"]["max_review_interval_seconds"], 120.0)
        self.assertEqual(
            design["scheduler_design"]["source_scheduler_experiment_hash"],
            experiment["provenance_evidence"]["scheduler_experiment_hash"],
        )
        self.assertFalse(design["scheduler_design"]["automatic_endpoint_execution"])
        self.assertFalse(design["scheduler_design"]["automatic_plasticity"])
        self.assertFalse(design["installs_scheduler"])
        self.assertFalse(design["executes_suggested_endpoint"])
        self.assertFalse(design["writes_checkpoint"])
        self.assertFalse(design["applies_plasticity"])
        self.assertFalse(design["mutates_runtime_state"])
        self.assertFalse(design["eligible_for_plasticity"])
        self.assertFalse(design["safety_contract"]["scheduler_installation_allowed"])
        self.assertFalse(design["safety_contract"]["suggested_endpoint_execution_allowed"])
        self.assertFalse(design["safety_contract"]["runtime_mutation_allowed"])
        self.assertFalse(design["device_evidence"]["tensor_execution_required"])
        self.assertFalse(design["device_evidence"]["cuda_applicable"])
        self.assertEqual(
            repeated_design["provenance_evidence"]["scheduler_design_hash"],
            design["provenance_evidence"]["scheduler_design_hash"],
        )
        self.assertFalse(design["promotion_gate"]["eligible_for_scheduler_installation"])
        self.assertTrue(
            design["promotion_gate"]["eligible_for_operator_scheduler_design_review"]
        )
        design_ticket = controller.record_snn_sleep_plasticity_scheduler_design_review_ticket(
            limit=4,
            cycles=3,
            min_stable_cycles=3,
            max_review_interval_seconds=120.0,
            expected_state_revision=manager._runtime_state.state_revision,
            scheduler_design_hash=design["provenance_evidence"]["scheduler_design_hash"],
            operator_id="operator-scheduler-design",
            confirmation=True,
        )
        design_ticket_id = str(design_ticket["scheduler_design_review_ticket_id"])
        self.assertEqual(
            design_ticket["surface"],
            "snn_sleep_plasticity_scheduler_design_review_ticket.v1",
        )
        self.assertEqual(
            design_ticket["scheduler_design_hash"],
            design["provenance_evidence"]["scheduler_design_hash"],
        )
        self.assertEqual(
            design_ticket["recorded_state_revision"],
            manager._runtime_state.state_revision,
        )
        self.assertFalse(design_ticket["installs_scheduler"])
        self.assertFalse(design_ticket["executes_suggested_endpoint"])
        self.assertFalse(design_ticket["records_replay_artifact"])
        self.assertFalse(design_ticket["issues_regeneration_permit"])
        self.assertFalse(design_ticket["writes_checkpoint"])
        self.assertFalse(design_ticket["applies_plasticity"])
        self.assertFalse(design_ticket["mutates_transition_memory"])
        self.assertFalse(design_ticket["mutates_runtime_state"])
        self.assertIsNotNone(
            controller.verified_snn_sleep_plasticity_scheduler_design_review_ticket(
                design_ticket_id,
                operator_id="operator-scheduler-design",
            )
        )
        design_ticket_queue = (
            controller.snn_sleep_plasticity_scheduler_design_review_ticket_queue(limit=4)
        )
        self.assertEqual(
            design_ticket_queue["surface"],
            "snn_sleep_plasticity_scheduler_design_review_ticket_queue.v1",
        )
        self.assertEqual(design_ticket_queue["verified_count"], 1)
        self.assertEqual(
            design_ticket_queue["latest_verified_ticket"][
                "scheduler_design_review_ticket_id"
            ],
            design_ticket_id,
        )
        design_ticket_queue_window = design_ticket_queue["source_window"]
        self.assertEqual(
            design_ticket_queue_window["surface"],
            "bounded_snn_sleep_plasticity_scheduler_design_review_ticket_queue_source_window.v1",
        )
        self.assertEqual(
            design_ticket_queue_window["source_window_limit"],
            SNN_SLEEP_PLASTICITY_SCHEDULER_DESIGN_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT,
        )
        self.assertEqual(design_ticket_queue_window["source_window_count"], 1)
        self.assertEqual(design_ticket_queue["retained_count"], 1)
        self.assertFalse(design_ticket_queue_window["global_candidate_scan"])
        self.assertFalse(design_ticket_queue_window["runs_live_tick"])
        self.assertFalse(design_ticket_queue_window["runs_every_token"])
        self.assertFalse(design_ticket_queue_window["gpu_used"])
        self.assertFalse(design_ticket_queue["installs_scheduler"])
        self.assertFalse(design_ticket_queue["executes_suggested_endpoint"])
        self.assertFalse(design_ticket_queue["mutates_runtime_state"])
        installation_proposal = (
            controller.snn_sleep_plasticity_scheduler_installation_autonomy_proposal(
                limit=4
            )
        )
        repeated_installation_proposal = (
            controller.snn_sleep_plasticity_scheduler_installation_autonomy_proposal(
                limit=4
            )
        )
        self.assertEqual(
            installation_proposal["surface"],
            "snn_sleep_plasticity_scheduler_installation_autonomy_proposal.v1",
        )
        self.assertTrue(installation_proposal["ready"])
        self.assertEqual(
            installation_proposal["candidate"]["scheduler_design_review_ticket_id"],
            design_ticket_id,
        )
        self.assertFalse(installation_proposal["installs_scheduler"])
        self.assertFalse(installation_proposal["registers_timer"])
        self.assertFalse(installation_proposal["starts_background_worker"])
        self.assertFalse(installation_proposal["executes_suggested_endpoint"])
        self.assertFalse(installation_proposal["mutates_runtime_state"])
        self.assertFalse(
            installation_proposal["promotion_gate"][
                "eligible_for_scheduler_installation"
            ]
        )
        self.assertEqual(
            repeated_installation_proposal["provenance_evidence"][
                "scheduler_installation_autonomy_proposal_hash"
            ],
            installation_proposal["provenance_evidence"][
                "scheduler_installation_autonomy_proposal_hash"
            ],
        )
        older_valid_ticket = deepcopy(design_ticket)
        older_valid_ticket["scheduler_design_review_ticket_id"] = "older-valid-design-ticket"
        controller.snn_sleep_plasticity_scheduler_design_review_tickets.append(
            older_valid_ticket
        )
        proposal_with_older_ticket = (
            controller.snn_sleep_plasticity_scheduler_installation_autonomy_proposal(
                limit=4
            )
        )
        self.assertEqual(
            proposal_with_older_ticket["provenance_evidence"][
                "scheduler_installation_autonomy_proposal_hash"
            ],
            installation_proposal["provenance_evidence"][
                "scheduler_installation_autonomy_proposal_hash"
            ],
        )
        malformed_ticket = deepcopy(design_ticket)
        malformed_ticket["scheduler_design_review_ticket_id"] = "malformed-design-ticket"
        malformed_ticket["design_parameters"] = {"cycles": "not-an-integer"}
        controller.snn_sleep_plasticity_scheduler_design_review_tickets.appendleft(
            malformed_ticket
        )
        limited_queue = (
            controller.snn_sleep_plasticity_scheduler_design_review_ticket_queue(
                limit=1
            )
        )
        self.assertEqual(limited_queue["tampered_count"], 1)
        self.assertIsNone(limited_queue["latest_verified_ticket"])
        self.assertFalse(limited_queue["ready"])
        self.assertEqual(limited_queue["source_window"]["source_window_count"], 1)
        self.assertEqual(
            limited_queue["source_window"]["latest_verified_scope"],
            "source_window_only",
        )
        controller.snn_sleep_plasticity_scheduler_design_review_tickets.popleft()
        controller.snn_sleep_plasticity_scheduler_design_review_tickets.pop()
        installation_preflight = (
            controller.snn_sleep_plasticity_scheduler_installation_preflight(
                limit=4
            )
        )
        repeated_installation_preflight = (
            controller.snn_sleep_plasticity_scheduler_installation_preflight(
                limit=4
            )
        )
        self.assertEqual(
            installation_preflight["surface"],
            "snn_sleep_plasticity_scheduler_installation_preflight.v1",
        )
        self.assertTrue(installation_preflight["ready"])
        self.assertEqual(
            installation_preflight["installation_review_preflight"][
                "scheduler_design_review_ticket_id"
            ],
            design_ticket_id,
        )
        self.assertEqual(
            installation_preflight["installation_review_preflight"][
                "scheduler_design_review_ticket_hash"
            ],
            design_ticket["evidence_hash"],
        )
        self.assertEqual(
            installation_preflight["installation_review_preflight"][
                "source_scheduler_experiment_hash"
            ],
            design["provenance_evidence"]["source_scheduler_experiment_hash"],
        )
        self.assertEqual(
            installation_preflight["installation_review_preflight"][
                "scheduler_review_parameters"
            ]["cycles"],
            3,
        )
        self.assertEqual(
            installation_preflight["installation_review_preflight"][
                "bound_state_revision"
            ],
            manager._runtime_state.state_revision,
        )
        self.assertFalse(installation_preflight["installs_scheduler"])
        self.assertFalse(installation_preflight["registers_timer"])
        self.assertFalse(installation_preflight["starts_background_worker"])
        self.assertFalse(installation_preflight["executes_suggested_endpoint"])
        self.assertFalse(installation_preflight["writes_checkpoint"])
        self.assertFalse(installation_preflight["applies_plasticity"])
        self.assertFalse(installation_preflight["mutates_runtime_state"])
        self.assertFalse(
            installation_preflight["promotion_gate"][
                "eligible_for_scheduler_installation"
            ]
        )
        self.assertEqual(
            repeated_installation_preflight["provenance_evidence"][
                "scheduler_installation_preflight_hash"
            ],
            installation_preflight["provenance_evidence"][
                "scheduler_installation_preflight_hash"
            ],
        )
        review_scheduler_installation = (
            controller.install_snn_sleep_plasticity_review_scheduler(
                limit=4,
                expected_state_revision=manager._runtime_state.state_revision,
                scheduler_installation_preflight_hash=(
                    installation_preflight["provenance_evidence"][
                        "scheduler_installation_preflight_hash"
                    ]
                ),
                operator_id="operator-review-scheduler",
                confirmation=True,
            )
        )
        self.assertEqual(
            review_scheduler_installation["surface"],
            "snn_sleep_plasticity_review_scheduler_installation.v1",
        )
        self.assertTrue(review_scheduler_installation["scheduler_installed"])
        self.assertEqual(review_scheduler_installation["scheduler_mode"], "review_only")
        self.assertFalse(review_scheduler_installation["registers_os_timer"])
        self.assertFalse(review_scheduler_installation["starts_background_worker"])
        self.assertFalse(review_scheduler_installation["executes_suggested_endpoint"])
        self.assertFalse(review_scheduler_installation["writes_checkpoint"])
        self.assertFalse(review_scheduler_installation["applies_plasticity"])
        self.assertFalse(review_scheduler_installation["mutates_transition_memory"])
        self.assertFalse(review_scheduler_installation["mutates_runtime_state"])
        review_scheduler_runtime = (
            controller.snn_sleep_plasticity_review_scheduler_runtime()
        )
        self.assertEqual(
            review_scheduler_runtime["surface"],
            "snn_sleep_plasticity_review_scheduler_runtime.v1",
        )
        self.assertTrue(review_scheduler_runtime["ready"])
        self.assertTrue(review_scheduler_runtime["scheduler_installed"])
        self.assertFalse(review_scheduler_runtime["review_due"])
        self.assertFalse(review_scheduler_runtime["registers_os_timer"])
        self.assertFalse(review_scheduler_runtime["starts_background_worker"])
        self.assertFalse(review_scheduler_runtime["executes_suggested_endpoint"])
        self.assertFalse(review_scheduler_runtime["applies_plasticity"])
        self.assertFalse(review_scheduler_runtime["mutates_runtime_state"])
        due_observed_at = datetime.fromisoformat(
            str(review_scheduler_installation["next_review_due_at"])
        ) + timedelta(seconds=1)
        due_cycle_inspection = (
            controller.snn_sleep_plasticity_review_scheduler_cycle_inspection(
                observed_at=due_observed_at
            )
        )
        repeated_due_cycle_inspection = (
            controller.snn_sleep_plasticity_review_scheduler_cycle_inspection(
                observed_at=due_observed_at
            )
        )
        self.assertEqual(
            due_cycle_inspection["surface"],
            "snn_sleep_plasticity_review_scheduler_cycle_inspection.v1",
        )
        self.assertTrue(due_cycle_inspection["ready"])
        self.assertTrue(due_cycle_inspection["cycle_inspection"]["review_due"])
        self.assertFalse(due_cycle_inspection["executes_suggested_endpoint"])
        self.assertFalse(due_cycle_inspection["records_replay_artifact"])
        self.assertFalse(due_cycle_inspection["writes_checkpoint"])
        self.assertFalse(due_cycle_inspection["applies_plasticity"])
        self.assertFalse(due_cycle_inspection["mutates_transition_memory"])
        self.assertFalse(due_cycle_inspection["mutates_runtime_state"])
        self.assertFalse(due_cycle_inspection["eligible_for_endpoint_execution"])
        self.assertEqual(
            repeated_due_cycle_inspection["provenance_evidence"][
                "review_scheduler_cycle_inspection_hash"
            ],
            due_cycle_inspection["provenance_evidence"][
                "review_scheduler_cycle_inspection_hash"
            ],
        )
        due_cycle_autonomy_proposal = (
            controller.snn_sleep_plasticity_review_scheduler_cycle_autonomy_proposal(
                observed_at=due_observed_at
            )
        )
        repeated_due_cycle_autonomy_proposal = (
            controller.snn_sleep_plasticity_review_scheduler_cycle_autonomy_proposal(
                observed_at=due_observed_at
            )
        )
        self.assertEqual(
            due_cycle_autonomy_proposal["surface"],
            "snn_sleep_plasticity_review_scheduler_cycle_autonomy_proposal.v1",
        )
        self.assertTrue(due_cycle_autonomy_proposal["ready"])
        self.assertEqual(
            due_cycle_autonomy_proposal["candidate"][
                "reviewed_sleep_plasticity_endpoint"
            ],
            review_scheduler_installation["reviewed_sleep_plasticity_endpoint"],
        )
        self.assertFalse(due_cycle_autonomy_proposal["executes_suggested_endpoint"])
        self.assertFalse(due_cycle_autonomy_proposal["records_replay_artifact"])
        self.assertFalse(due_cycle_autonomy_proposal["writes_checkpoint"])
        self.assertFalse(due_cycle_autonomy_proposal["applies_plasticity"])
        self.assertFalse(due_cycle_autonomy_proposal["mutates_transition_memory"])
        self.assertFalse(due_cycle_autonomy_proposal["mutates_runtime_state"])
        self.assertFalse(
            due_cycle_autonomy_proposal["candidate"]["endpoint_execution_allowed"]
        )
        self.assertEqual(
            repeated_due_cycle_autonomy_proposal["provenance_evidence"][
                "review_scheduler_cycle_autonomy_proposal_hash"
            ],
            due_cycle_autonomy_proposal["provenance_evidence"][
                "review_scheduler_cycle_autonomy_proposal_hash"
            ],
        )
        sleep_replay_context = self._record_replay_evaluation_context(controller)
        sleep_replay_priority_queue = controller.snn_replay_consolidation_priority_queue(
            readout_replay_priority_report={
                "surface": "snn_language_readout_replay_priority.v1",
                "candidates": [
                    {
                        "priority_score": 90.0,
                        "all_labels_grounded": True,
                    }
                ],
            },
            limit=4,
        )
        sleep_replay_selection_proposal = (
            controller.snn_due_cycle_bounded_replay_selection_proposal(
                consolidation_priority_queue=sleep_replay_priority_queue,
                observed_at=due_observed_at,
            )
        )
        repeated_sleep_replay_selection_proposal = (
            controller.snn_due_cycle_bounded_replay_selection_proposal(
                consolidation_priority_queue=sleep_replay_priority_queue,
                observed_at=due_observed_at,
            )
        )
        self.assertEqual(
            sleep_replay_selection_proposal["surface"],
            "snn_due_cycle_bounded_replay_selection_proposal.v1",
        )
        self.assertTrue(sleep_replay_selection_proposal["ready"])
        self.assertEqual(
            sleep_replay_selection_proposal["selection"]["candidates"][0][
                "replay_evaluation_context_id"
            ],
            sleep_replay_context["replay_evaluation_context_id"],
        )
        self.assertFalse(sleep_replay_selection_proposal["executes_suggested_endpoint"])
        self.assertFalse(sleep_replay_selection_proposal["records_replay_artifact"])
        self.assertFalse(sleep_replay_selection_proposal["runs_live_replay"])
        self.assertFalse(sleep_replay_selection_proposal["writes_checkpoint"])
        self.assertFalse(sleep_replay_selection_proposal["applies_plasticity"])
        self.assertFalse(sleep_replay_selection_proposal["mutates_transition_memory"])
        self.assertFalse(sleep_replay_selection_proposal["mutates_runtime_state"])
        self.assertEqual(
            repeated_sleep_replay_selection_proposal["provenance_evidence"][
                "due_cycle_bounded_replay_selection_proposal_hash"
            ],
            sleep_replay_selection_proposal["provenance_evidence"][
                "due_cycle_bounded_replay_selection_proposal_hash"
            ],
        )
        sleep_replay_recording_policy = (
            controller.snn_replay_artifact_recording_policy_proposal(
                consolidation_priority_queue=sleep_replay_priority_queue,
                policy={"min_priority_score": 60.0},
            )
        )
        due_cycle_recording_review_proposal = (
            controller.snn_due_cycle_replay_artifact_recording_review_proposal(
                due_cycle_selection_proposal=sleep_replay_selection_proposal,
                artifact_recording_policy_proposal=sleep_replay_recording_policy,
            )
        )
        repeated_due_cycle_recording_review_proposal = (
            controller.snn_due_cycle_replay_artifact_recording_review_proposal(
                due_cycle_selection_proposal=sleep_replay_selection_proposal,
                artifact_recording_policy_proposal=sleep_replay_recording_policy,
            )
        )
        self.assertEqual(
            due_cycle_recording_review_proposal["surface"],
            "snn_due_cycle_replay_artifact_recording_review_proposal.v1",
        )
        self.assertTrue(due_cycle_recording_review_proposal["ready"])
        self.assertEqual(
            due_cycle_recording_review_proposal["review_target"][
                "replay_evaluation_context_id"
            ],
            sleep_replay_context["replay_evaluation_context_id"],
        )
        self.assertFalse(due_cycle_recording_review_proposal["records_replay_artifact"])
        self.assertFalse(due_cycle_recording_review_proposal["runs_live_replay"])
        self.assertFalse(due_cycle_recording_review_proposal["writes_checkpoint"])
        self.assertFalse(due_cycle_recording_review_proposal["applies_plasticity"])
        self.assertFalse(due_cycle_recording_review_proposal["mutates_runtime_state"])
        self.assertEqual(
            repeated_due_cycle_recording_review_proposal["provenance_evidence"][
                "due_cycle_replay_artifact_recording_review_proposal_hash"
            ],
            due_cycle_recording_review_proposal["provenance_evidence"][
                "due_cycle_replay_artifact_recording_review_proposal_hash"
            ],
        )
        due_cycle_recording_review_ticket = (
            controller.record_snn_replay_artifact_recording_review_ticket(
                policy_proposal=sleep_replay_recording_policy,
                due_cycle_review_proposal=due_cycle_recording_review_proposal,
                operator_id="operator-due-cycle-replay-review",
                confirmation=True,
            )
        )
        self.assertEqual(
            due_cycle_recording_review_ticket["surface"],
            "snn_replay_artifact_recording_review_ticket.v1",
        )
        self.assertEqual(
            due_cycle_recording_review_ticket["due_cycle_review_proposal_hash"],
            due_cycle_recording_review_proposal["provenance_evidence"][
                "due_cycle_replay_artifact_recording_review_proposal_hash"
            ],
        )
        self.assertIsNotNone(
            controller.verified_snn_replay_artifact_recording_review_ticket(
                str(due_cycle_recording_review_ticket["review_ticket_id"]),
                operator_id="operator-due-cycle-replay-review",
            )
        )
        controller.snn_replay_artifact_recording_review_tickets[0][
            "due_cycle_review_proposal_hash"
        ] = "tampered"
        self.assertIsNone(
            controller.verified_snn_replay_artifact_recording_review_ticket(
                str(due_cycle_recording_review_ticket["review_ticket_id"]),
                operator_id="operator-due-cycle-replay-review",
            )
        )
        controller.snn_replay_artifact_recording_review_tickets[0] = (
            due_cycle_recording_review_ticket
        )
        cycle_acknowledgment_preflight = (
            controller.snn_sleep_plasticity_review_scheduler_cycle_acknowledgment_preflight(
                scheduler_installation_id=review_scheduler_installation[
                    "scheduler_installation_id"
                ],
                scheduler_installation_evidence_hash=review_scheduler_installation[
                    "evidence_hash"
                ],
                review_ticket_id=due_cycle_recording_review_ticket["review_ticket_id"],
                observed_at=due_observed_at,
            )
        )
        self.assertTrue(cycle_acknowledgment_preflight["ready"])
        self.assertFalse(cycle_acknowledgment_preflight["executes_suggested_endpoint"])
        self.assertFalse(cycle_acknowledgment_preflight["records_replay_artifact"])
        self.assertFalse(cycle_acknowledgment_preflight["applies_plasticity"])
        self.assertFalse(cycle_acknowledgment_preflight["mutates_runtime_state"])
        sleep_phase_separation = controller.snn_sleep_phase_separation_proposal(
            due_cycle_selection_proposal=sleep_replay_selection_proposal,
            cycle_acknowledgment_preflight=cycle_acknowledgment_preflight,
        )
        repeated_sleep_phase_separation = controller.snn_sleep_phase_separation_proposal(
            due_cycle_selection_proposal=sleep_replay_selection_proposal,
            cycle_acknowledgment_preflight=cycle_acknowledgment_preflight,
        )
        self.assertEqual(
            sleep_phase_separation["surface"],
            "snn_sleep_phase_separation_proposal.v1",
        )
        self.assertTrue(
            sleep_phase_separation["nrem_like_replay_nomination"]["ready"]
        )
        self.assertTrue(
            sleep_phase_separation["rem_like_stabilization_review"]["ready"]
        )
        self.assertFalse(sleep_phase_separation["records_replay_artifact"])
        self.assertFalse(sleep_phase_separation["runs_live_replay"])
        self.assertFalse(sleep_phase_separation["applies_plasticity"])
        self.assertFalse(sleep_phase_separation["mutates_runtime_state"])
        self.assertEqual(
            repeated_sleep_phase_separation["provenance_evidence"][
                "sleep_phase_separation_proposal_hash"
            ],
            sleep_phase_separation["provenance_evidence"][
                "sleep_phase_separation_proposal_hash"
            ],
        )
        nrem_only_phase_separation = controller.snn_sleep_phase_separation_proposal(
            due_cycle_selection_proposal=sleep_replay_selection_proposal,
        )
        self.assertTrue(
            nrem_only_phase_separation["nrem_like_replay_nomination"]["ready"]
        )
        self.assertFalse(
            nrem_only_phase_separation["rem_like_stabilization_review"]["ready"]
        )
        revision_before_rem_preflight = manager._runtime_state.state_revision
        dirty_calls_before_rem_preflight = (
            manager._runtime_state.dirty_without_revision_calls
        )
        rem_homeostatic_preflight = (
            controller.snn_rem_like_homeostatic_stabilization_preflight(
                sleep_phase_separation_proposal=sleep_phase_separation,
                transition_memory_state={
                    "sparse_transition_weight_count": 4,
                    "homeostatic_maintenance_count": 0,
                    "regeneration_count": 1,
                },
            )
        )
        repeated_rem_homeostatic_preflight = (
            controller.snn_rem_like_homeostatic_stabilization_preflight(
                sleep_phase_separation_proposal=sleep_phase_separation,
                transition_memory_state={
                    "sparse_transition_weight_count": 4,
                    "homeostatic_maintenance_count": 0,
                    "regeneration_count": 1,
                },
            )
        )
        self.assertEqual(
            manager._runtime_state.state_revision,
            revision_before_rem_preflight,
        )
        self.assertEqual(
            manager._runtime_state.dirty_without_revision_calls,
            dirty_calls_before_rem_preflight,
        )
        self.assertEqual(
            rem_homeostatic_preflight["surface"],
            "snn_rem_like_homeostatic_stabilization_preflight.v1",
        )
        self.assertTrue(rem_homeostatic_preflight["ready"])
        self.assertEqual(
            rem_homeostatic_preflight["review_plan"]["suggested_endpoint"],
            "/terminus/snn-language-sequence/plasticity-homeostatic-maintenance",
        )
        self.assertFalse(rem_homeostatic_preflight["executes_suggested_endpoint"])
        self.assertFalse(rem_homeostatic_preflight["writes_checkpoint"])
        self.assertFalse(rem_homeostatic_preflight["applies_plasticity"])
        self.assertFalse(rem_homeostatic_preflight["mutates_transition_memory"])
        self.assertFalse(rem_homeostatic_preflight["mutates_runtime_state"])
        self.assertEqual(
            repeated_rem_homeostatic_preflight["provenance_evidence"][
                "rem_like_homeostatic_stabilization_preflight_hash"
            ],
            rem_homeostatic_preflight["provenance_evidence"][
                "rem_like_homeostatic_stabilization_preflight_hash"
            ],
        )
        blocked_rem_homeostatic_preflight = (
            controller.snn_rem_like_homeostatic_stabilization_preflight(
                sleep_phase_separation_proposal=nrem_only_phase_separation,
                transition_memory_state={
                    "sparse_transition_weight_count": 4,
                    "homeostatic_maintenance_count": 0,
                    "regeneration_count": 1,
                },
            )
        )
        self.assertFalse(blocked_rem_homeostatic_preflight["ready"])
        missing_transition_memory_preflight = (
            controller.snn_rem_like_homeostatic_stabilization_preflight(
                sleep_phase_separation_proposal=sleep_phase_separation,
                transition_memory_state={
                    "sparse_transition_weight_count": 0,
                    "homeostatic_maintenance_count": 0,
                    "regeneration_count": 1,
                },
            )
        )
        self.assertFalse(missing_transition_memory_preflight["ready"])
        self.assertFalse(
            missing_transition_memory_preflight["promotion_gate"]["required_evidence"][
                "transition_memory_present"
            ]
        )
        stale_maintenance_pressure_preflight = (
            controller.snn_rem_like_homeostatic_stabilization_preflight(
                sleep_phase_separation_proposal=sleep_phase_separation,
                transition_memory_state={
                    "sparse_transition_weight_count": 4,
                    "homeostatic_maintenance_count": 1,
                    "regeneration_count": 1,
                },
            )
        )
        self.assertFalse(stale_maintenance_pressure_preflight["ready"])
        self.assertFalse(
            stale_maintenance_pressure_preflight["promotion_gate"][
                "required_evidence"
            ]["post_growth_homeostatic_maintenance_due"]
        )
        tampered_phase_separation = deepcopy(sleep_phase_separation)
        tampered_phase_separation["writes_checkpoint"] = True
        tampered_phase_separation["applies_plasticity"] = True
        tampered_phase_separation["mutates_transition_memory"] = True
        tampered_phase_preflight = (
            controller.snn_rem_like_homeostatic_stabilization_preflight(
                sleep_phase_separation_proposal=tampered_phase_separation,
                transition_memory_state={
                    "sparse_transition_weight_count": 4,
                    "homeostatic_maintenance_count": 0,
                    "regeneration_count": 1,
                },
            )
        )
        self.assertFalse(tampered_phase_preflight["ready"])
        self.assertFalse(
            tampered_phase_preflight["promotion_gate"]["required_evidence"][
                "checkpoint_write_blocked"
            ]
        )
        self.assertFalse(
            tampered_phase_preflight["promotion_gate"]["required_evidence"][
                "plasticity_blocked"
            ]
        )
        self.assertFalse(
            tampered_phase_preflight["promotion_gate"]["required_evidence"][
                "transition_memory_mutation_blocked"
            ]
        )
        out_of_bound_policy_preflight = (
            controller.snn_rem_like_homeostatic_stabilization_preflight(
                sleep_phase_separation_proposal=sleep_phase_separation,
                transition_memory_state={
                    "sparse_transition_weight_count": 4,
                    "homeostatic_maintenance_count": 0,
                    "regeneration_count": 1,
                },
                maintenance_policy={
                    "decay_factor": 1.2,
                    "prune_below": 0.5,
                    "max_outgoing_row_mass": 8.0,
                },
            )
        )
        self.assertFalse(out_of_bound_policy_preflight["ready"])
        self.assertFalse(
            out_of_bound_policy_preflight["promotion_gate"]["required_evidence"][
                "maintenance_parameters_bounded"
            ]
        )
        self.assertIsNone(out_of_bound_policy_preflight["review_plan"]["decay_factor"])
        self.assertEqual(
            manager._runtime_state.state_revision,
            revision_before_rem_preflight,
        )
        self.assertEqual(
            manager._runtime_state.dirty_without_revision_calls,
            dirty_calls_before_rem_preflight,
        )
        two_candidate_selection = deepcopy(sleep_replay_selection_proposal)
        two_candidate_selection["selection"]["candidate_count"] = 2
        two_candidate_selection["selection"]["candidates"].append(
            deepcopy(two_candidate_selection["selection"]["candidates"][0])
        )
        two_candidate_phase_separation = controller.snn_sleep_phase_separation_proposal(
            due_cycle_selection_proposal=two_candidate_selection,
            cycle_acknowledgment_preflight=cycle_acknowledgment_preflight,
        )
        self.assertFalse(
            two_candidate_phase_separation["nrem_like_replay_nomination"]["ready"]
        )
        self.assertTrue(
            two_candidate_phase_separation["rem_like_stabilization_review"]["ready"]
        )
        cycle_acknowledgment = (
            controller.acknowledge_snn_sleep_plasticity_review_scheduler_cycle(
                expected_state_revision=manager._runtime_state.state_revision,
                scheduler_installation_id=review_scheduler_installation[
                    "scheduler_installation_id"
                ],
                scheduler_installation_evidence_hash=review_scheduler_installation[
                    "evidence_hash"
                ],
                review_ticket_id=due_cycle_recording_review_ticket["review_ticket_id"],
                operator_id="operator-review-scheduler-cycle",
                confirmation=True,
                observed_at=due_observed_at,
            )
        )
        self.assertEqual(
            cycle_acknowledgment["surface"],
            "snn_sleep_plasticity_review_scheduler_cycle_acknowledgment.v1",
        )
        self.assertTrue(cycle_acknowledgment["scheduler_cadence_advanced"])
        self.assertTrue(cycle_acknowledgment["mutates_scheduler_cadence_state"])
        self.assertFalse(cycle_acknowledgment["executes_suggested_endpoint"])
        self.assertFalse(cycle_acknowledgment["records_replay_artifact"])
        self.assertFalse(cycle_acknowledgment["runs_live_replay"])
        self.assertFalse(cycle_acknowledgment["writes_checkpoint"])
        self.assertFalse(cycle_acknowledgment["applies_plasticity"])
        self.assertFalse(cycle_acknowledgment["mutates_transition_memory"])
        self.assertFalse(cycle_acknowledgment["mutates_runtime_state"])
        acknowledged_scheduler_runtime = (
            controller.snn_sleep_plasticity_review_scheduler_runtime(
                observed_at=due_observed_at
            )
        )
        self.assertTrue(acknowledged_scheduler_runtime["ready"])
        self.assertFalse(acknowledged_scheduler_runtime["review_due"])
        self.assertEqual(
            acknowledged_scheduler_runtime["acknowledged_cycle_count"],
            1,
        )
        self.assertEqual(
            len(controller.snn_sleep_plasticity_review_scheduler_installations),
            2,
        )
        acknowledged_scheduler_configuration = (
            controller.snn_sleep_plasticity_review_scheduler_installations[0]
        )
        self.assertEqual(
            acknowledged_scheduler_configuration[
                "previous_scheduler_configuration_evidence_hash"
            ],
            review_scheduler_installation["evidence_hash"],
        )
        self.assertEqual(
            datetime.fromisoformat(
                acknowledged_scheduler_configuration["next_review_due_at"]
            ),
            datetime.fromisoformat(review_scheduler_installation["next_review_due_at"])
            + timedelta(seconds=120.0),
        )
        consumed_cycle_acknowledgment_preflight = (
            controller.snn_sleep_plasticity_review_scheduler_cycle_acknowledgment_preflight(
                scheduler_installation_id=review_scheduler_installation[
                    "scheduler_installation_id"
                ],
                scheduler_installation_evidence_hash=review_scheduler_installation[
                    "evidence_hash"
                ],
                review_ticket_id=due_cycle_recording_review_ticket["review_ticket_id"],
                observed_at=due_observed_at,
            )
        )
        self.assertFalse(consumed_cycle_acknowledgment_preflight["ready"])
        with self.assertRaisesRegex(ValueError, "current preflight evidence"):
            controller.acknowledge_snn_sleep_plasticity_review_scheduler_cycle(
                expected_state_revision=manager._runtime_state.state_revision,
                scheduler_installation_id=review_scheduler_installation[
                    "scheduler_installation_id"
                ],
                scheduler_installation_evidence_hash=review_scheduler_installation[
                    "evidence_hash"
                ],
                review_ticket_id=due_cycle_recording_review_ticket["review_ticket_id"],
                operator_id="operator-review-scheduler-cycle",
                confirmation=True,
                observed_at=due_observed_at,
            )
        waiting_sleep_replay_selection_proposal = (
            controller.snn_due_cycle_bounded_replay_selection_proposal(
                consolidation_priority_queue=sleep_replay_priority_queue,
            )
        )
        self.assertFalse(waiting_sleep_replay_selection_proposal["ready"])
        self.assertEqual(
            waiting_sleep_replay_selection_proposal["selection"]["candidate_count"],
            0,
        )
        tampered_sleep_replay_priority_queue = deepcopy(sleep_replay_priority_queue)
        tampered_sleep_replay_priority_queue["candidates"][0][
            "replay_evaluation_context_hash"
        ] = "tampered"
        tampered_sleep_replay_selection_proposal = (
            controller.snn_due_cycle_bounded_replay_selection_proposal(
                consolidation_priority_queue=tampered_sleep_replay_priority_queue,
                observed_at=due_observed_at,
            )
        )
        self.assertFalse(tampered_sleep_replay_selection_proposal["ready"])
        self.assertEqual(
            tampered_sleep_replay_selection_proposal["selection"]["candidate_count"],
            0,
        )
        tampered_due_cycle_recording_review_proposal = (
            controller.snn_due_cycle_replay_artifact_recording_review_proposal(
                due_cycle_selection_proposal=tampered_sleep_replay_selection_proposal,
                artifact_recording_policy_proposal=sleep_replay_recording_policy,
            )
        )
        self.assertFalse(tampered_due_cycle_recording_review_proposal["ready"])
        self.assertIsNone(
            tampered_due_cycle_recording_review_proposal["review_target"][
                "replay_evaluation_context_id"
            ]
        )
        controller.snn_sleep_plasticity_scheduler_design_review_tickets[0][
            "scheduler_design_hash"
        ] = "tampered"
        self.assertIsNone(
            controller.verified_snn_sleep_plasticity_scheduler_design_review_ticket(
                design_ticket_id,
                operator_id="operator-scheduler-design",
            )
        )
        tampered_design_ticket_queue = (
            controller.snn_sleep_plasticity_scheduler_design_review_ticket_queue(limit=4)
        )
        self.assertEqual(tampered_design_ticket_queue["verified_count"], 0)
        self.assertEqual(tampered_design_ticket_queue["tampered_count"], 1)
        tampered_installation_proposal = (
            controller.snn_sleep_plasticity_scheduler_installation_autonomy_proposal(
                limit=4
            )
        )
        self.assertFalse(tampered_installation_proposal["ready"])
        tampered_installation_preflight = (
            controller.snn_sleep_plasticity_scheduler_installation_preflight(
                limit=4
            )
        )
        self.assertFalse(tampered_installation_preflight["ready"])
        tampered_review_scheduler_runtime = (
            controller.snn_sleep_plasticity_review_scheduler_runtime()
        )
        self.assertFalse(tampered_review_scheduler_runtime["ready"])
        self.assertFalse(tampered_review_scheduler_runtime["scheduler_installed"])
        controller.snn_sleep_plasticity_scheduler_design_review_tickets[0] = design_ticket
        with self.assertRaisesRegex(ValueError, "current controller evidence"):
            controller.record_snn_sleep_plasticity_scheduler_design_review_ticket(
                limit=4,
                cycles=3,
                min_stable_cycles=3,
                max_review_interval_seconds=120.0,
                expected_state_revision=manager._runtime_state.state_revision,
                scheduler_design_hash="0" * 64,
                operator_id="operator-scheduler-design",
                confirmation=True,
            )
        self.assertEqual(manager._runtime_state.state_revision, revision_before_experiment)
        self.assertEqual(
            manager._runtime_state.dirty_without_revision_calls,
            dirty_calls_before_experiment + 5,
        )
        self.assertEqual(
            len(controller.snn_sleep_plasticity_review_tickets),
            ticket_count_before_experiment,
        )
        forged_ticket = dict(ticket)
        forged_ticket["surface"] = "forged_sleep_plasticity_review_ticket.v1"
        controller.snn_sleep_plasticity_review_tickets[0] = forged_ticket
        forged_queue = controller.snn_sleep_plasticity_review_ticket_queue(limit=4)
        self.assertEqual(forged_queue["verified_count"], 0)
        self.assertEqual(forged_queue["tampered_count"], 1)
        controller.snn_sleep_plasticity_review_tickets[0] = ticket
        controller.snn_sleep_plasticity_review_tickets[0]["sleep_policy_hash"] = "tampered"
        tampered_queue = controller.snn_sleep_plasticity_review_ticket_queue(limit=4)
        self.assertEqual(tampered_queue["verified_count"], 0)
        self.assertEqual(tampered_queue["tampered_count"], 1)
        tampered_proposal = controller.snn_sleep_plasticity_autonomy_proposal(limit=4)
        self.assertFalse(tampered_proposal["ready"])
        self.assertEqual(
            tampered_proposal["candidate"]["action"],
            "collect_sleep_plasticity_review_ticket",
        )
        tampered_experiment = controller.snn_sleep_plasticity_scheduler_experiment(
            limit=4,
            cycles=3,
        )
        self.assertFalse(tampered_experiment["ready"])
        self.assertFalse(
            tampered_experiment["promotion_gate"][
                "eligible_for_operator_scheduler_design_review"
            ]
        )
        tampered_design = controller.snn_sleep_plasticity_scheduler_design(
            limit=4,
            cycles=3,
        )
        self.assertFalse(tampered_design["ready"])
        self.assertFalse(
            tampered_design["promotion_gate"][
                "eligible_for_operator_scheduler_design_review"
            ]
        )
        self.assertIsNone(
            controller.verified_snn_sleep_plasticity_review_ticket(
                ticket_id,
                operator_id="operator-sleep",
            )
        )
        controller.snn_sleep_plasticity_review_tickets[0] = ticket
        manager._runtime_state.state_revision += 1
        stale_queue = controller.snn_sleep_plasticity_review_ticket_queue(limit=4)
        self.assertEqual(stale_queue["verified_count"], 0)
        self.assertEqual(stale_queue["stale_count"], 1)
        stale_experiment = controller.snn_sleep_plasticity_scheduler_experiment(
            limit=4,
            cycles=3,
        )
        self.assertFalse(stale_experiment["ready"])
        self.assertFalse(
            stale_experiment["promotion_gate"][
                "eligible_for_operator_scheduler_design_review"
            ]
        )
        stale_design = controller.snn_sleep_plasticity_scheduler_design(
            limit=4,
            cycles=3,
        )
        self.assertFalse(stale_design["ready"])
        self.assertFalse(
            stale_design["promotion_gate"][
                "eligible_for_operator_scheduler_design_review"
            ]
        )
        self.assertIsNone(
            controller.verified_snn_sleep_plasticity_review_ticket(
                ticket_id,
                operator_id="operator-sleep",
            )
        )

    def test_snn_sleep_plasticity_review_ticket_rejects_non_recommended_policy(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        policy = self._sleep_policy()
        policy["recommendation"] = {
            "action": "continue_monitoring_transition_memory",
            "recommended": False,
            "suggested_endpoint": None,
            "executable": False,
        }

        with self.assertRaisesRegex(ValueError, "non-executable recommendation"):
            controller.record_snn_sleep_plasticity_review_ticket(
                sleep_policy=policy,
                operator_id="operator-sleep",
                confirmation=True,
            )

    def test_snn_sleep_plasticity_review_ticket_rejects_mismatched_action_endpoint(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        policy = self._sleep_policy()
        policy["recommendation"]["suggested_endpoint"] = (
            "/terminus/snn-language-sequence/transition-memory-replay-artifact/proposal"
        )

        with self.assertRaisesRegex(ValueError, "non-executable recommendation"):
            controller.record_snn_sleep_plasticity_review_ticket(
                sleep_policy=policy,
                operator_id="operator-sleep",
                confirmation=True,
            )

    def test_snn_replay_artifact_recording_policy_proposal_blocks_below_threshold(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        self._record_replay_evaluation_context(controller)
        queue = controller.snn_replay_consolidation_priority_queue(
            readout_replay_priority_report={
                "surface": "snn_language_readout_replay_priority.v1",
                "candidates": [
                    {
                        "priority_score": 10.0,
                        "all_labels_grounded": True,
                    }
                ],
            },
            limit=4,
        )

        proposal = controller.snn_replay_artifact_recording_policy_proposal(
            consolidation_priority_queue=queue,
            policy={"min_priority_score": 99.0},
        )

        self.assertFalse(proposal["recommended"])
        self.assertEqual(proposal["candidate_count"], 0)
        self.assertFalse(
            proposal["promotion_gate"]["required_evidence"]["candidate_above_policy_threshold"]
        )

    def test_replay_sample_marks_dirty_without_revision(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)

        plan_payload = {
            "endpoint": "/terminus/replay-plan",
            "candidates": [
                {
                    "candidate_id": "candidate-1",
                    "target_type": "runtime_episode",
                    "target_id": "episode-1",
                    "priority_score": 8.0,
                }
            ],
        }

        class _FakePlan:
            def to_payload(self) -> dict[str, object]:
                return plan_payload

        with patch("marulho.service.replay_runtime.build_replay_plan", return_value=_FakePlan()):
            record = controller.replay_sample(
                mode="sample",
                candidate_id="candidate-1",
                operator_id="operator-1",
                confirmation=True,
            )

        self.assertEqual(manager._runtime_state.dirty_without_revision_calls, 1)
        self.assertEqual(record["selected_candidate_ids"], ["candidate-1"])
        self.assertFalse(record["safety_flags"]["state_revision_mutated"])
        self.assertEqual(record["before"]["state_revision"], record["after"]["state_revision"])
        self.assertEqual(controller.history[0]["replay_sample_id"], record["replay_sample_id"])

    def test_regeneration_permit_is_controller_owned_and_revision_bound(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        artifact = self._record_regeneration_replay_artifact(controller)
        permit = controller.issue_regeneration_permit(
            replay_artifact_id=str(artifact["replay_artifact_id"]),
            regeneration_design={
                "locality_radius": 2,
                "initial_weight": 0.02,
                "max_new_synapses": 8,
                "mismatch_score": 0.9,
                "candidate_synapses": [{"pre_index": 1, "post_index": 2, "initial_weight": 0.02}],
            },
            operator_id="operator-1",
            confirmation=True,
        )
        proposal = {
            "replay_evidence": permit,
            "regeneration_design": {
                "locality_radius": 2,
                "initial_weight": 0.02,
                "max_new_synapses": 8,
                "mismatch_score": 0.9,
                "candidate_synapses": [{"pre_index": 1, "post_index": 2, "initial_weight": 0.02}],
            },
        }

        self.assertTrue(controller.verify_regeneration_permit(proposal))
        self.assertEqual(artifact["readout_evidence_hashes"], ["readout-hash-1"])
        self.assertEqual(permit["readout_evidence_hashes"], ["readout-hash-1"])
        self.assertTrue(permit["confirmation"])
        self.assertEqual(permit["operator_id"], "operator-1")
        self.assertTrue(permit["regeneration_design_hash"])
        self.assertEqual(controller.regeneration_permits[0]["permit_id"], permit["permit_id"])
        self.assertEqual(manager._runtime_state.dirty_without_revision_calls, 4)
        manager._runtime_state.state_revision += 1
        self.assertFalse(controller.verify_regeneration_permit(proposal))

    def test_regeneration_permit_rejects_oversized_candidate_source_window_before_mutation(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        artifact = self._record_regeneration_replay_artifact(controller)
        before_dirty_calls = manager._runtime_state.dirty_without_revision_calls
        before_permit_count = len(controller.regeneration_permits)
        oversized_candidates = [
            {
                "pre_index": index % 31,
                "post_index": (index % 31) + 1,
                "initial_weight": 0.02,
            }
            for index in range(SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT + 1)
        ]

        with self.assertRaisesRegex(ValueError, "candidate count must be bounded"):
            controller.issue_regeneration_permit(
                replay_artifact_id=str(artifact["replay_artifact_id"]),
                regeneration_design={
                    "locality_radius": 2,
                    "initial_weight": 0.02,
                    "max_new_synapses": SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT,
                    "mismatch_score": 0.9,
                    "candidate_synapses": oversized_candidates,
                },
                operator_id="operator-1",
                confirmation=True,
            )

        self.assertEqual(len(controller.regeneration_permits), before_permit_count)
        self.assertEqual(
            manager._runtime_state.dirty_without_revision_calls,
            before_dirty_calls,
        )

    def test_evaluated_replay_artifact_and_permit_preserve_emission_lineage(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        source_metadata = {
            "source": "runtime_facade.snn_language_readout_emission_replay_context_review",
            "surface": "snn_language_readout_emission_replay_context_review.v1",
            "design_hash": "design-hash",
            "seed_hash": "seed-hash",
            "emission_review_hash": "emission-review-hash",
            "emission_hash": "emission-hash",
            "readout_evidence_hash": "readout-hash-1",
            "prediction_hash": "prediction-hash",
            "operator_id": "operator-lineage",
        }
        context = self._record_replay_evaluation_context(
            controller,
            source_metadata=source_metadata,
        )
        ticket = self._record_review_ticket(controller, operator_id="operator-lineage")
        emission_lineage = {
            "source": source_metadata["source"],
            "surface": source_metadata["surface"],
            "design_hash": source_metadata["design_hash"],
            "seed_hash": source_metadata["seed_hash"],
            "emission_review_hash": source_metadata["emission_review_hash"],
            "emission_hash": source_metadata["emission_hash"],
            "readout_evidence_hash": source_metadata["readout_evidence_hash"],
            "prediction_hash": source_metadata["prediction_hash"],
        }
        proposal = _with_replay_priority_source_window(
            controller,
            {
                "surface": "snn_transition_memory_replay_artifact_proposal.v1",
                "ready": True,
                "owned_by_marulho": True,
                "source": "service.snn_language_readout_ledger.transition_memory_replay_artifact_proposal",
                "mismatch_report": context["mismatch_report"],
                "pressure_report": context["pressure_report"],
                "replay_evaluation_context_id": context["replay_evaluation_context_id"],
                "replay_evaluation_context_hash": context["evidence_hash"],
                "source_metadata_hash": context["source_metadata_hash"],
                "emission_lineage": emission_lineage,
                "replay_window": [{"readout_evidence_hash": "readout-hash-1", "grounded": True}],
                "promotion_gate": {"status": "ready_for_operator_recording_review"},
            },
        )

        artifact = controller.record_evaluated_snn_transition_memory_replay_artifact(
            artifact_proposal=proposal,
            known_readout_evidence_hashes=["readout-hash-1"],
            known_readout_evidence_source_window=_known_readout_evidence_source_window(),
            replay_evaluation_context_id=str(context["replay_evaluation_context_id"]),
            review_ticket_id=str(ticket["review_ticket_id"]),
            operator_id="operator-lineage",
            confirmation=True,
        )
        self.assertEqual(
            artifact["readout_evidence_source_window"]["surface"],
            "bounded_snn_readout_known_evidence_hash_source_window.v1",
        )
        self.assertIn("readout_evidence_source_window_hash", artifact)
        self.assertEqual(
            artifact["replay_priority_source_window"]["surface"],
            "bounded_snn_readout_replay_priority_source_window.v1",
        )
        self.assertIn("replay_priority_source_window_hash", artifact)
        permit_design = {
            "locality_radius": 2,
            "initial_weight": 0.02,
            "max_new_synapses": 8,
            "mismatch_score": 0.9,
            "candidate_synapses": [{"pre_index": 1, "post_index": 2, "initial_weight": 0.02}],
        }
        permit = controller.issue_regeneration_permit(
            replay_artifact_id=str(artifact["replay_artifact_id"]),
            regeneration_design=permit_design,
            operator_id="operator-lineage",
            confirmation=True,
        )

        self.assertEqual(artifact["source_metadata_hash"], context["source_metadata_hash"])
        self.assertEqual(artifact["emission_lineage"], emission_lineage)
        self.assertIn("replay_priority_source_window_hash", artifact)
        self.assertEqual(permit["source_metadata_hash"], context["source_metadata_hash"])
        self.assertEqual(permit["emission_lineage"], emission_lineage)
        self.assertTrue(
            controller.verify_regeneration_permit(
                {"replay_evidence": permit, "regeneration_design": permit_design}
            )
        )
        controller.regeneration_permits[0]["emission_lineage"]["emission_hash"] = "tampered"
        self.assertFalse(
            controller.verify_regeneration_permit(
                {"replay_evidence": permit, "regeneration_design": permit_design}
            )
        )

    def test_regeneration_permit_rejects_tampered_payload(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        artifact = self._record_regeneration_replay_artifact(controller)
        permit = controller.issue_regeneration_permit(
            replay_artifact_id=str(artifact["replay_artifact_id"]),
            regeneration_design={
                "locality_radius": 2,
                "initial_weight": 0.02,
                "max_new_synapses": 8,
                "mismatch_score": 0.9,
                "candidate_synapses": [{"pre_index": 1, "post_index": 2, "initial_weight": 0.02}],
            },
            operator_id="operator-1",
            confirmation=True,
        )
        permit["evidence_hash"] = "fabricated"

        self.assertFalse(
            controller.verify_regeneration_permit(
                {
                    "replay_evidence": permit,
                    "regeneration_design": {
                        "locality_radius": 2,
                        "initial_weight": 0.02,
                        "max_new_synapses": 8,
                        "mismatch_score": 0.9,
                        "candidate_synapses": [
                            {"pre_index": 1, "post_index": 2, "initial_weight": 0.02}
                        ],
                    },
                }
            )
        )

    def test_regeneration_permit_rejects_candidate_design_drift(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        artifact = self._record_regeneration_replay_artifact(controller)
        design = {
            "locality_radius": 2,
            "initial_weight": 0.02,
            "max_new_synapses": 8,
            "mismatch_score": 0.9,
            "candidate_synapses": [{"pre_index": 1, "post_index": 2, "initial_weight": 0.02}],
        }
        permit = controller.issue_regeneration_permit(
            replay_artifact_id=str(artifact["replay_artifact_id"]),
            regeneration_design=design,
            operator_id="operator-1",
            confirmation=True,
        )
        tampered_design = deepcopy(design)
        tampered_design["candidate_synapses"][0]["post_index"] = 3

        self.assertFalse(
            controller.verify_regeneration_permit(
                {"replay_evidence": permit, "regeneration_design": tampered_design}
            )
        )

    def test_regeneration_permit_rejects_replay_window_provenance_drift(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        artifact = self._record_regeneration_replay_artifact(controller)
        design = {
            "locality_radius": 2,
            "initial_weight": 0.02,
            "max_new_synapses": 8,
            "mismatch_score": 0.9,
            "candidate_synapses": [{"pre_index": 1, "post_index": 2, "initial_weight": 0.02}],
        }
        permit = controller.issue_regeneration_permit(
            replay_artifact_id=str(artifact["replay_artifact_id"]),
            regeneration_design=design,
            operator_id="operator-1",
            confirmation=True,
        )
        controller.snn_transition_memory_replay_artifacts[0]["readout_evidence_hashes"] = []

        self.assertFalse(
            controller.verify_regeneration_permit(
                {"replay_evidence": permit, "regeneration_design": design}
            )
        )

    def test_regeneration_permit_verification_uses_indexed_provenance_window(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        artifact = self._record_regeneration_replay_artifact(controller)
        design = {
            "locality_radius": 2,
            "initial_weight": 0.02,
            "max_new_synapses": 8,
            "mismatch_score": 0.9,
            "candidate_synapses": [{"pre_index": 1, "post_index": 2, "initial_weight": 0.02}],
        }
        permit = controller.issue_regeneration_permit(
            replay_artifact_id=str(artifact["replay_artifact_id"]),
            regeneration_design=design,
            operator_id="operator-1",
            confirmation=True,
        )

        artifact_window = dict(artifact["source_window"])
        permit_window = dict(permit["source_window"])
        self.assertEqual(
            artifact_window["surface"],
            "bounded_snn_replay_artifact_provenance_source_window.v1",
        )
        self.assertEqual(artifact_window["source_record_count"], 2)
        self.assertEqual(artifact_window["index_lookup_count"], 2)
        self.assertFalse(artifact_window["global_candidate_scan"])
        self.assertFalse(artifact_window["language_reasoning"])
        self.assertEqual(permit_window["source_record_count"], 3)
        self.assertEqual(permit_window["archival_storage_device"], "cpu")
        self.assertFalse(permit_window["gpu_used"])

        controller._regeneration_permits = _IterationBlockedContexts()  # type: ignore[assignment]
        controller._snn_transition_memory_replay_artifacts = _IterationBlockedContexts()  # type: ignore[assignment]
        controller._snn_replay_artifact_recording_review_tickets = _IterationBlockedContexts()  # type: ignore[assignment]
        controller._snn_replay_evaluation_contexts = _IterationBlockedContexts()  # type: ignore[assignment]

        self.assertTrue(
            controller.verify_regeneration_permit(
                {"replay_evidence": permit, "regeneration_design": design}
            )
        )

    def test_regeneration_permit_rejects_missing_server_owned_replay_artifact(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)

        with self.assertRaisesRegex(ValueError, "verified server-owned SNN replay artifact"):
            controller.issue_regeneration_permit(
                replay_artifact_id="fabricated-replay-artifact",
                regeneration_design={
                    "locality_radius": 2,
                    "initial_weight": 0.02,
                    "max_new_synapses": 8,
                    "mismatch_score": 0.9,
                    "candidate_synapses": [
                        {"pre_index": 1, "post_index": 2, "initial_weight": 0.02}
                    ],
                },
                operator_id="operator-1",
                confirmation=True,
            )

    def test_evaluated_replay_artifact_rejects_spoofed_internal_ledger_hash(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        context = self._record_replay_evaluation_context(controller)
        proposal = _with_replay_priority_source_window(
            controller,
            {
                "surface": "snn_transition_memory_replay_artifact_proposal.v1",
                "ready": True,
                "owned_by_marulho": True,
                "source": "service.snn_language_readout_ledger.transition_memory_replay_artifact_proposal",
                "mismatch_report": context["mismatch_report"],
                "pressure_report": context["pressure_report"],
                "replay_evaluation_context_id": context["replay_evaluation_context_id"],
                "replay_evaluation_context_hash": context["evidence_hash"],
                "replay_window": [
                    {"readout_evidence_hash": "fabricated-hash", "grounded": True}
                ],
                "promotion_gate": {"status": "ready_for_operator_recording_review"},
            },
        )

        with self.assertRaisesRegex(ValueError, "current internal-ledger evidence"):
            controller.record_evaluated_snn_transition_memory_replay_artifact(
                artifact_proposal=proposal,
                known_readout_evidence_hashes={"real-hash"},
                known_readout_evidence_source_window=_known_readout_evidence_source_window(),
                replay_evaluation_context_id=str(context["replay_evaluation_context_id"]),
                review_ticket_id=str(self._record_review_ticket(controller)["review_ticket_id"]),
                operator_id="operator-1",
                confirmation=True,
            )

    def test_evaluated_replay_artifact_rejects_incomplete_known_readout_source_window(
        self,
    ) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        context = self._record_replay_evaluation_context(controller)
        proposal = _with_replay_priority_source_window(
            controller,
            {
                "surface": "snn_transition_memory_replay_artifact_proposal.v1",
                "ready": True,
                "owned_by_marulho": True,
                "source": "service.snn_language_readout_ledger.transition_memory_replay_artifact_proposal",
                "mismatch_report": context["mismatch_report"],
                "pressure_report": context["pressure_report"],
                "replay_evaluation_context_id": context["replay_evaluation_context_id"],
                "replay_evaluation_context_hash": context["evidence_hash"],
                "replay_window": [
                    {"readout_evidence_hash": "readout-hash-1", "grounded": True}
                ],
                "promotion_gate": {"status": "ready_for_operator_recording_review"},
            },
        )

        with self.assertRaisesRegex(
            ValueError,
            "bounded current internal-ledger evidence source window",
        ):
            controller.record_evaluated_snn_transition_memory_replay_artifact(
                artifact_proposal=proposal,
                known_readout_evidence_hashes={"readout-hash-1"},
                known_readout_evidence_source_window={
                    "surface": "bounded_snn_readout_known_evidence_hash_source_window.v1",
                    "source_window_count": 1,
                    "source_window_limit": 8,
                    "archival_storage_device": "cpu",
                    "gpu_used": False,
                },
                replay_evaluation_context_id=str(context["replay_evaluation_context_id"]),
                review_ticket_id=str(self._record_review_ticket(controller)["review_ticket_id"]),
                operator_id="operator-1",
                confirmation=True,
            )

    def test_evaluated_replay_artifact_rejects_missing_replay_priority_source_window(
        self,
    ) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        context = self._record_replay_evaluation_context(controller)
        ticket = self._record_review_ticket(controller)
        proposal = {
            "surface": "snn_transition_memory_replay_artifact_proposal.v1",
            "ready": True,
            "owned_by_marulho": True,
            "source": "service.snn_language_readout_ledger.transition_memory_replay_artifact_proposal",
            "mismatch_report": context["mismatch_report"],
            "pressure_report": context["pressure_report"],
            "replay_evaluation_context_id": context["replay_evaluation_context_id"],
            "replay_evaluation_context_hash": context["evidence_hash"],
            "replay_window": [
                {"readout_evidence_hash": "readout-hash-1", "grounded": True}
            ],
            "promotion_gate": {"status": "ready_for_operator_recording_review"},
        }

        with self.assertRaisesRegex(ValueError, "replay-priority source window"):
            controller.record_evaluated_snn_transition_memory_replay_artifact(
                artifact_proposal=proposal,
                known_readout_evidence_hashes={"readout-hash-1"},
                known_readout_evidence_source_window=_known_readout_evidence_source_window(),
                replay_evaluation_context_id=str(context["replay_evaluation_context_id"]),
                review_ticket_id=str(ticket["review_ticket_id"]),
                operator_id="operator-1",
                confirmation=True,
            )

    def test_evaluated_replay_artifact_rejects_tampered_replay_priority_source_window_hash(
        self,
    ) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        context = self._record_replay_evaluation_context(controller)
        ticket = self._record_review_ticket(controller)
        proposal = _with_replay_priority_source_window(
            controller,
            {
                "surface": "snn_transition_memory_replay_artifact_proposal.v1",
                "ready": True,
                "owned_by_marulho": True,
                "source": "service.snn_language_readout_ledger.transition_memory_replay_artifact_proposal",
                "mismatch_report": context["mismatch_report"],
                "pressure_report": context["pressure_report"],
                "replay_evaluation_context_id": context["replay_evaluation_context_id"],
                "replay_evaluation_context_hash": context["evidence_hash"],
                "replay_window": [
                    {"readout_evidence_hash": "readout-hash-1", "grounded": True}
                ],
                "promotion_gate": {"status": "ready_for_operator_recording_review"},
            },
        )
        proposal["replay_priority_source_window_hash"] = "fabricated"

        with self.assertRaisesRegex(
            ValueError,
            "matching replay-priority source-window hash",
        ):
            controller.record_evaluated_snn_transition_memory_replay_artifact(
                artifact_proposal=proposal,
                known_readout_evidence_hashes={"readout-hash-1"},
                known_readout_evidence_source_window=_known_readout_evidence_source_window(),
                replay_evaluation_context_id=str(context["replay_evaluation_context_id"]),
                review_ticket_id=str(ticket["review_ticket_id"]),
                operator_id="operator-1",
                confirmation=True,
            )

    def test_regeneration_permit_rejects_tampered_evaluated_artifact_source_windows(
        self,
    ) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        artifact = self._record_regeneration_replay_artifact(controller)
        controller.snn_transition_memory_replay_artifacts[0][
            "replay_priority_source_window"
        ]["source_event_window_limit"] = 999

        with self.assertRaisesRegex(ValueError, "verified server-owned SNN replay artifact"):
            controller.issue_regeneration_permit(
                replay_artifact_id=str(artifact["replay_artifact_id"]),
                regeneration_design={
                    "locality_radius": 2,
                    "initial_weight": 0.02,
                    "max_new_synapses": 8,
                    "mismatch_score": 0.9,
                    "candidate_synapses": [
                        {"pre_index": 1, "post_index": 2, "initial_weight": 0.02}
                    ],
                },
                operator_id="operator-1",
                confirmation=True,
            )

    def test_regeneration_permit_rejects_raw_caller_window_artifact(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        self.assertFalse(hasattr(controller, "record_snn_transition_memory_replay_artifact"))
        controller.snn_transition_memory_replay_artifacts = [
            {
                "artifact_kind": "terminus_snn_transition_memory_replay_artifact",
                "surface": "snn_transition_memory_replay_artifact.v1",
                "replay_artifact_id": "raw-artifact-1",
                "evidence_hash": "raw-hash-1",
                "internal_ledger_backed": False,
            }
        ]

        self.assertEqual(len(controller.snn_transition_memory_replay_artifacts), 0)

        with self.assertRaisesRegex(ValueError, "verified server-owned SNN replay artifact"):
            controller.issue_regeneration_permit(
                replay_artifact_id="raw-artifact-1",
                regeneration_design={
                    "locality_radius": 2,
                    "initial_weight": 0.02,
                    "max_new_synapses": 8,
                    "mismatch_score": 0.9,
                    "candidate_synapses": [
                        {"pre_index": 1, "post_index": 2, "initial_weight": 0.02}
                    ],
                },
                operator_id="operator-1",
                confirmation=True,
            )


if __name__ == "__main__":
    unittest.main()
