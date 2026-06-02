from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from threading import RLock
import unittest
from unittest.mock import patch

from hecsn.service.replay_runtime import ReplayController, ReplayControllerDependencies


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


def _replay_controller(manager: _FakeReplayManager) -> ReplayController:
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
        )
    )


class ReplayControllerTests(unittest.TestCase):
    @staticmethod
    def _mismatch_report() -> dict[str, object]:
        return {
            "surface": "snn_language_sequence_mismatch_probe.v1",
            "available": True,
            "owned_by_hecsn": True,
            "prediction_error": {"mismatch_score": 0.9},
            "promotion_gate": {"status": "ready_for_operator_review"},
        }

    @staticmethod
    def _pressure_report() -> dict[str, object]:
        return {
            "surface": "snn_language_plasticity_pressure.v1",
            "available": True,
            "owned_by_hecsn": True,
            "promotion_gate": {"status": "ready_for_operator_review"},
        }

    @staticmethod
    def _sleep_policy() -> dict[str, object]:
        return {
            "surface": "snn_language_transition_memory_sleep_policy.v1",
            "available": True,
            "owned_by_hecsn": True,
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
    def _record_replay_evaluation_context(cls, controller: ReplayController) -> dict[str, object]:
        return controller.record_snn_replay_evaluation_context(
            mismatch_report=cls._mismatch_report(),
            pressure_report=cls._pressure_report(),
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
            artifact_proposal={
                "surface": "snn_transition_memory_replay_artifact_proposal.v1",
                "ready": True,
                "owned_by_hecsn": True,
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
            known_readout_evidence_hashes={"readout-hash-1"},
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

    def test_snn_transition_memory_replay_artifact_setter_preserves_existing_deque_reference(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        artifacts = controller.snn_transition_memory_replay_artifacts

        controller.snn_transition_memory_replay_artifacts = [
            {"replay_artifact_id": "artifact-1", "evidence_hash": "hash-1"}
        ]

        self.assertIs(controller.snn_transition_memory_replay_artifacts, artifacts)
        self.assertEqual(artifacts[0]["replay_artifact_id"], "artifact-1")

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

    def test_evaluated_replay_artifact_rejects_stale_context(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        context = self._record_replay_evaluation_context(controller)
        manager._runtime_state.state_revision += 1

        with self.assertRaisesRegex(ValueError, "verified server-held evaluation context"):
            controller.record_evaluated_snn_transition_memory_replay_artifact(
                artifact_proposal={
                    "surface": "snn_transition_memory_replay_artifact_proposal.v1",
                    "ready": True,
                    "owned_by_hecsn": True,
                    "source": "service.snn_language_readout_ledger.transition_memory_replay_artifact_proposal",
                    "mismatch_report": context["mismatch_report"],
                    "pressure_report": context["pressure_report"],
                    "replay_evaluation_context_id": context["replay_evaluation_context_id"],
                    "replay_evaluation_context_hash": context["evidence_hash"],
                    "replay_window": [{"readout_evidence_hash": "readout-hash-1", "grounded": True}],
                    "promotion_gate": {"status": "ready_for_operator_recording_review"},
                },
                known_readout_evidence_hashes={"readout-hash-1"},
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
        self.assertEqual(
            limited_queue["latest_verified_ticket"][
                "scheduler_design_review_ticket_id"
            ],
            design_ticket_id,
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
            dirty_calls_before_experiment + 1,
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

        with patch("hecsn.service.replay_runtime.build_replay_plan", return_value=_FakePlan()):
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
        proposal = {
            "surface": "snn_transition_memory_replay_artifact_proposal.v1",
            "ready": True,
            "owned_by_hecsn": True,
            "source": "service.snn_language_readout_ledger.transition_memory_replay_artifact_proposal",
            "mismatch_report": context["mismatch_report"],
            "pressure_report": context["pressure_report"],
            "replay_evaluation_context_id": context["replay_evaluation_context_id"],
            "replay_evaluation_context_hash": context["evidence_hash"],
            "replay_window": [
                {"readout_evidence_hash": "fabricated-hash", "grounded": True}
            ],
            "promotion_gate": {"status": "ready_for_operator_recording_review"},
        }

        with self.assertRaisesRegex(ValueError, "current internal-ledger evidence"):
            controller.record_evaluated_snn_transition_memory_replay_artifact(
                artifact_proposal=proposal,
                known_readout_evidence_hashes={"real-hash"},
                replay_evaluation_context_id=str(context["replay_evaluation_context_id"]),
                review_ticket_id=str(self._record_review_ticket(controller)["review_ticket_id"]),
                operator_id="operator-1",
                confirmation=True,
            )

    def test_regeneration_permit_rejects_raw_caller_window_artifact(self) -> None:
        manager = _FakeReplayManager()
        controller = _replay_controller(manager)
        raw_artifact = controller.record_snn_transition_memory_replay_artifact(
            mismatch_report={"available": True, "prediction_error": {"mismatch_score": 0.9}},
            pressure_report={"available": True},
            replay_window=[{"case_id": "caller-window", "grounded": True}],
            operator_id="operator-1",
            confirmation=True,
        )

        with self.assertRaisesRegex(ValueError, "verified server-owned SNN replay artifact"):
            controller.issue_regeneration_permit(
                replay_artifact_id=str(raw_artifact["replay_artifact_id"]),
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
