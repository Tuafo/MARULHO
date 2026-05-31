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

    def subcortical_self_repair_surface(self) -> dict[str, Any]:
        return self._root._status_read_model.subcortical_self_repair_surface()

    def subcortical_self_repair_evaluation_surface(self) -> dict[str, Any]:
        return self._root._status_read_model.subcortical_self_repair_evaluation_surface()

    def subcortical_structural_plasticity_surface(self) -> dict[str, Any]:
        return self._root._status_read_model.subcortical_structural_plasticity_surface()

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
