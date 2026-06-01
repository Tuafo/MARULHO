from __future__ import annotations

from typing import Any, Mapping

from hecsn.service.operator_interaction import OperatorInteractionRuntime
from hecsn.service.reporting import ServiceReporter
from hecsn.service.replay_dataset_bundle import ReplayDatasetPackager
from hecsn.service.runtime_evidence import RuntimeEvidenceReporter


class RuntimeFacade:
    """Operator-facing runtime interface over Service Manager deep modules.

    HECSNServiceManager is the composition root. This facade is the runtime
    interface used by HTTP routes, runners, and integration tests that need the
    stable operator surface without depending on manager pass-through methods.
    """

    def __init__(self, composition_root: Any) -> None:
        self._root = composition_root

    def status(self, *, fresh_wait_seconds: float | None = None) -> dict[str, Any]:
        return self._root._status_read_model.status(fresh_wait_seconds=fresh_wait_seconds)

    def terminus_status(self, *, fresh_wait_seconds: float | None = None) -> dict[str, Any]:
        return self._root._status_read_model.terminus_status(fresh_wait_seconds=fresh_wait_seconds)

    def sensory_previews(self, limit: int = 6) -> dict[str, Any]:
        return self._root._status_read_model.sensory_previews(limit=limit)

    def architecture_summary(self) -> dict[str, Any]:
        return self._root._status_read_model.architecture_summary()

    def telemetry_snapshot(self) -> dict[str, Any]:
        return self._root._status_read_model.telemetry_snapshot()

    def living_loop_status(self) -> dict[str, Any]:
        return self._root._status_read_model.living_loop_status()

    def policy_actuator_status(self) -> dict[str, Any]:
        return self._root._status_read_model.policy_actuator_status()

    def cognitive_signal_state(self) -> dict[str, Any]:
        return self._root._status_read_model.cognitive_signal_state()

    def subcortical_language_surface(self) -> dict[str, Any]:
        return self._root._status_read_model.subcortical_language_surface()

    def subcortical_deliberation_surface(self) -> dict[str, Any]:
        return self._root._status_read_model.subcortical_deliberation_surface()

    def snn_language_readiness_surface(self) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_readiness_surface()

    def snn_language_evaluation_surface(self) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_evaluation_surface()

    def snn_language_adapter_heldout_evaluation(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_adapter_heldout_evaluation(**kwargs)

    def snn_language_training_readiness(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_training_readiness(**kwargs)

    def snn_language_trainer_dry_run(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_trainer_dry_run(**kwargs)

    def snn_language_trainer_isolated_evaluation(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_trainer_isolated_evaluation(**kwargs)

    def snn_language_sequence_prediction_probe(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("persistent_transition_weights") is None:
            state = getattr(self._root, "_snn_language_plasticity_state", {})
            if isinstance(state, Mapping):
                kwargs["persistent_transition_weights"] = dict(state.get("sparse_transition_weights") or {})
        return self._root._status_read_model.snn_language_sequence_prediction_probe(**kwargs)

    def snn_language_sequence_mismatch_probe(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_sequence_mismatch_probe(**kwargs)

    def snn_language_readout_draft(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_readout_draft(**kwargs)

    def snn_language_readout_evidence_ledger(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.snapshot(**kwargs)

    def snn_language_readout_replay_priority(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.replay_priority(**kwargs)

    def snn_language_readout_rehearsal_evaluation(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.rehearsal_evaluation(**kwargs)

    def snn_language_readout_rehearsal_experiment(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.rehearsal_experiment(**kwargs)

    def snn_language_readout_replay_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.replay_design(**kwargs)

    def snn_language_readout_replay_dry_run(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.replay_dry_run(**kwargs)

    def snn_language_readout_plasticity_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.plasticity_preflight(**kwargs)

    def snn_language_readout_plasticity_replay_bridge(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.plasticity_replay_bridge(**kwargs)

    def snn_language_readout_evidence_ledger_record(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.record_readout_draft(**kwargs)

    def snn_language_transition_memory_prediction_evaluation(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("transition_memory_state") is None:
            kwargs["transition_memory_state"] = self.snn_language_plasticity_runtime_state()
        return self._root._status_read_model.snn_language_transition_memory_prediction_evaluation(**kwargs)

    def snn_language_plasticity_pressure(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_plasticity_pressure(**kwargs)

    def snn_language_plasticity_trial(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_plasticity_trial(**kwargs)

    def snn_language_plasticity_replay_evaluation(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_plasticity_replay_evaluation(**kwargs)

    def snn_language_plasticity_replay_experiment(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_plasticity_replay_experiment(**kwargs)

    def snn_language_plasticity_application_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_plasticity_application_design(**kwargs)

    def snn_language_plasticity_shadow_application(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_plasticity_shadow_application(**kwargs)

    def snn_language_plasticity_shadow_delta(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_plasticity_shadow_delta(**kwargs)

    def snn_language_plasticity_live_application_readiness(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_plasticity_live_application_readiness(**kwargs)

    def snn_language_plasticity_live_application_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_plasticity_live_application_preflight(**kwargs)

    def snn_language_plasticity_live_application(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_plasticity_executor.apply_live_application(**kwargs)

    def snn_language_plasticity_runtime_state(self) -> dict[str, Any]:
        return self._root._snn_language_plasticity_executor.snapshot()

    def snn_language_transition_memory_homeostatic_maintenance(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_plasticity_executor.maintain_transition_memory(**kwargs)

    def snn_language_transition_memory_sleep_policy(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("transition_memory_state") is None:
            kwargs["transition_memory_state"] = self.snn_language_plasticity_runtime_state()
        return self._root._status_read_model.snn_language_transition_memory_sleep_policy(**kwargs)

    def snn_language_transition_memory_regeneration_proposal(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("transition_memory_state") is None:
            kwargs["transition_memory_state"] = self.snn_language_plasticity_runtime_state()
        return self._root._status_read_model.snn_language_transition_memory_regeneration_proposal(**kwargs)

    def snn_language_transition_memory_regeneration_permit(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._replay_controller.issue_regeneration_permit(**kwargs)

    def snn_language_transition_memory_regeneration(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_plasticity_executor.regenerate_transition_memory(**kwargs)

    def subcortical_self_repair_surface(self) -> dict[str, Any]:
        return self._root._status_read_model.subcortical_self_repair_surface()

    def subcortical_self_repair_evaluation_surface(self) -> dict[str, Any]:
        return self._root._status_read_model.subcortical_self_repair_evaluation_surface()

    def subcortical_structural_plasticity_surface(self) -> dict[str, Any]:
        return self._root._status_read_model.subcortical_structural_plasticity_surface()

    def subcortical_structural_plasticity_isolated_evaluation(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.subcortical_structural_plasticity_isolated_evaluation(**kwargs)

    def checkpoint_list(self) -> list[dict[str, Any]]:
        return self._root._runtime_persistence.checkpoint_list()

    def recent_traces(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._root._runtime_persistence.recent_traces(limit=limit)

    def save_checkpoint(self, path: str | None = None) -> dict[str, Any]:
        return self._root._runtime_persistence.save_checkpoint(path)

    def restore_checkpoint(self, path: str) -> dict[str, Any]:
        return self._root._runtime_persistence.restore_checkpoint(path)

    def feed(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._interaction_pipeline.feed(**kwargs)

    def query(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._interaction_pipeline.query(**kwargs)

    def respond(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._interaction_pipeline.respond(**kwargs)

    def acquire(self, **kwargs: Any) -> dict[str, Any]:
        return OperatorInteractionRuntime.acquire(self._root, **kwargs)

    def configure_terminus(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._runtime_control.configure_terminus(**kwargs)

    def start_terminus(self) -> dict[str, Any]:
        return self._root._runtime_control.start_terminus()

    def stop_terminus(self) -> dict[str, Any]:
        return self._root._runtime_control.stop_terminus()

    def terminus_tick(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._runtime_control.terminus_tick(**kwargs)

    def quick_start_terminus(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._runtime_control.quick_start_terminus(**kwargs)

    def replay_plan_status(self, *, limit: int = 20) -> dict[str, Any]:
        return self._root._replay_controller.replay_plan_status(limit=limit)

    def replay_sample(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._replay_controller.replay_sample(**kwargs)

    def replay_sample_history(self, *, limit: int = 20) -> dict[str, Any]:
        return self._root._replay_controller.replay_sample_history(limit=limit)

    def export_runtime_trace_examples(self, **kwargs: Any) -> dict[str, Any]:
        return RuntimeEvidenceReporter.export_runtime_trace_examples(self._root, **kwargs)

    def replay_dataset_preview(self, **kwargs: Any) -> dict[str, Any]:
        return RuntimeEvidenceReporter.replay_dataset_preview(self._root, **kwargs)

    def replay_dataset_candidates(self, **kwargs: Any) -> dict[str, Any]:
        return RuntimeEvidenceReporter.replay_dataset_candidates(self._root, **kwargs)

    def replay_dataset_history(self, **kwargs: Any) -> dict[str, Any]:
        return RuntimeEvidenceReporter.replay_dataset_history(self._root, **kwargs)

    def replay_dataset_bundle(self, **kwargs: Any) -> dict[str, Any]:
        return ReplayDatasetPackager.replay_dataset_bundle(self._root, **kwargs)

    def action_history(self, limit: int = 20) -> dict[str, Any]:
        return self._root._action_executor.action_history(limit=limit)

    def execute_digital_action(self, action: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self._root._action_executor.execute_digital_action(action, **kwargs)

    def record_runtime_feedback(self, feedback: Mapping[str, Any]) -> dict[str, Any]:
        return self._root._feedback_applier.record_runtime_feedback(feedback)

    def run_grounding_probe(self) -> dict[str, Any]:
        return ServiceReporter.run_grounding_probe(self._root)
