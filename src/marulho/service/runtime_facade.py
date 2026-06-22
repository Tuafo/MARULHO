from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from marulho.service.operator_interaction import OperatorInteractionRuntime
from marulho.service.reporting import ServiceReporter
from marulho.service.replay_dataset_bundle import ReplayDatasetPackager
from marulho.service.runtime_evidence import RuntimeEvidenceReporter
from marulho.service.snn_language_plasticity_executor import (
    SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT,
    bounded_application_synapse_window,
)

_SNN_LANGUAGE_CAPACITY_SURFACE = "snn_language_capacity_state.v1"
_SNN_LANGUAGE_NEURON_COUNT = 64
_SNN_LANGUAGE_SPARSE_EDGE_BUDGET = 256
_SNN_LANGUAGE_OUTGOING_FANOUT_BUDGET = 16
_ROLLOUT_REGENERATION_SOURCE_WINDOW_FALSE_FLAGS = (
    "global_candidate_scan",
    "global_score_scan",
    "raw_text_payload_loaded",
    "hidden_language_reasoning",
    "language_reasoning",
    "runs_live_tick",
    "runs_every_token",
    "mutates_runtime_state",
    "applies_plasticity",
    "gpu_used",
    "gpu_resident_archival_metadata",
)
_READOUT_REPLAY_PAYLOAD_SOURCE_WINDOW_FALSE_FLAGS = (
    "global_candidate_scan",
    "global_score_scan",
    "raw_text_payload_loaded",
    "language_reasoning",
    "runs_live_tick",
    "runs_every_token",
    "mutates_runtime_state",
    "applies_plasticity",
    "gpu_resident_archival_metadata",
    "gpu_used_for_archival_metadata",
)


def _source_window_int(source_window: Mapping[str, Any], key: str) -> int | None:
    try:
        return int(source_window.get(key, -1))
    except (TypeError, ValueError):
        return None


def _source_window_counts_bounded(
    source_window: Mapping[str, Any],
    *,
    max_limit: int | None = None,
    require_mapping_count: bool = False,
) -> bool:
    source_window_count = _source_window_int(source_window, "source_window_count")
    source_window_limit = _source_window_int(source_window, "source_window_limit")
    if source_window_count is None or source_window_limit is None:
        return False
    if source_window_limit <= 0 or source_window_count < 0:
        return False
    if source_window_count > source_window_limit:
        return False
    if max_limit is not None and source_window_limit > int(max_limit):
        return False
    if require_mapping_count:
        source_mapping_count = _source_window_int(source_window, "source_mapping_count")
        if source_mapping_count is None or source_mapping_count != source_window_count:
            return False
    return True


def _source_window_flags_explicit_false(
    source_window: Mapping[str, Any],
    flags: tuple[str, ...],
) -> bool:
    return all(source_window.get(flag) is False for flag in flags)


class RuntimeFacade:
    """Operator-facing runtime interface over Service Manager deep modules.

    MarulhoServiceManager is the composition root. This facade is the runtime
    interface used by HTTP routes, runners, and integration tests that need the
    stable operator surface without depending on manager pass-through methods.
    """

    def __init__(self, composition_root: Any) -> None:
        self._root = composition_root

    @staticmethod
    def _rollout_regeneration_candidate_window(
        raw_value: Any,
        *,
        source: str,
        surface: str,
        field_name: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return bounded_application_synapse_window(
            raw_value,
            source=source,
            surface=surface,
            field_name=field_name,
        )

    @staticmethod
    def _rollout_regeneration_candidate_window_bounded(
        source_window: Mapping[str, Any],
        *,
        surface: str,
    ) -> bool:
        return (
            source_window.get("surface") == surface
            and _source_window_counts_bounded(
                source_window,
                max_limit=SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT,
                require_mapping_count=True,
            )
            and _source_window_flags_explicit_false(
                source_window,
                _ROLLOUT_REGENERATION_SOURCE_WINDOW_FALSE_FLAGS,
            )
            and source_window.get("archival_storage_device") == "cpu"
            and source_window.get("source_window_selection_device") == "cpu"
        )

    @staticmethod
    def _readout_replay_payload_window_bounded(
        source_window: Mapping[str, Any],
        *,
        surface: str,
    ) -> bool:
        return (
            source_window.get("surface") == surface
            and _source_window_counts_bounded(
                source_window,
                require_mapping_count=True,
            )
            and _source_window_flags_explicit_false(
                source_window,
                _READOUT_REPLAY_PAYLOAD_SOURCE_WINDOW_FALSE_FLAGS,
            )
            and source_window.get("archival_storage_device") == "cpu"
        )

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
        if kwargs.get("language_neuron_count") is None:
            state = getattr(self._root, "_snn_language_plasticity_state", {})
            capacity = (
                state.get("language_capacity")
                if isinstance(state, Mapping)
                and isinstance(state.get("language_capacity"), Mapping)
                else {}
            )
            kwargs["language_neuron_count"] = max(
                64,
                int(capacity.get("language_neuron_count", 64) or 64),
            )
        return self._root._status_read_model.snn_language_trainer_dry_run(**kwargs)

    def snn_language_trainer_isolated_evaluation(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_trainer_isolated_evaluation(**kwargs)

    def snn_language_sequence_prediction_probe(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("persistent_transition_weights") is None:
            state = getattr(self._root, "_snn_language_plasticity_state", {})
            if isinstance(state, Mapping):
                kwargs["persistent_transition_weights"] = dict(state.get("sparse_transition_weights") or {})
        if kwargs.get("language_neuron_count") is None:
            state = getattr(self._root, "_snn_language_plasticity_state", {})
            capacity = (
                state.get("language_capacity")
                if isinstance(state, Mapping)
                and isinstance(state.get("language_capacity"), Mapping)
                else {}
            )
            kwargs["language_neuron_count"] = max(
                64,
                int(capacity.get("language_neuron_count", 64) or 64),
            )
        return self._root._status_read_model.snn_language_sequence_prediction_probe(**kwargs)

    def snn_language_sequence_mismatch_probe(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_sequence_mismatch_probe(**kwargs)

    def snn_language_readout_draft(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_readout_draft(**kwargs)

    def snn_language_readout_emission(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_readout_emission(**kwargs)

    def snn_language_readout_emission_review_record(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.record_readout_emission_review(**kwargs)

    def snn_language_dense_readout_label_candidate_evidence_record(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.record_dense_readout_label_candidate_review(
            **kwargs
        )

    def snn_language_readout_emission_review_history(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.emission_review_history(**kwargs)

    def snn_language_dense_label_candidate_history(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.dense_label_candidate_history(
            **kwargs
        )

    def snn_language_dense_label_candidate_calibration_policy(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.dense_label_candidate_calibration_policy(
            **kwargs
        )

    def snn_language_dense_label_candidate_calibration_evaluation_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.dense_label_candidate_calibration_evaluation_design(
            **kwargs
        )

    def snn_language_dense_label_candidate_calibration_evaluation_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.dense_label_candidate_calibration_evaluation_preflight(
            **kwargs
        )

    def snn_language_dense_label_candidate_calibration_evaluation(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.dense_label_candidate_calibration_evaluation(
            **kwargs
        )

    def snn_language_dense_label_candidate_calibration_evaluation_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.dense_label_candidate_calibration_evaluation_review(
            **kwargs
        )

    def snn_language_dense_label_candidate_calibration_update_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.dense_label_candidate_calibration_update_design(
            **kwargs
        )

    def snn_language_dense_label_candidate_calibration_update_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.dense_label_candidate_calibration_update_preflight(
            **kwargs
        )

    def snn_language_dense_label_candidate_calibration_update_application(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.apply_dense_label_candidate_calibration_update(
            **kwargs
        )

    def snn_language_dense_label_candidate_calibration_update_application_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.dense_label_candidate_calibration_update_application_review(
            **kwargs
        )

    def snn_language_dense_label_candidate_post_calibration_observation_window(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.dense_label_candidate_post_calibration_observation_window(
            **kwargs
        )

    def snn_language_dense_label_candidate_post_calibration_operator_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.dense_label_candidate_post_calibration_operator_review(
            **kwargs
        )

    def snn_language_calibrated_dense_label_confidence_use_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.calibrated_dense_label_confidence_use_design(
            **kwargs
        )

    def snn_language_calibrated_dense_label_confidence_use_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.calibrated_dense_label_confidence_use_preflight(
            **kwargs
        )

    def snn_language_calibrated_dense_label_confidence_use_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_calibrated_dense_label_confidence_use(
            **kwargs
        )

    def snn_language_calibrated_dense_label_confidence_operator_display_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.calibrated_dense_label_confidence_operator_display_review(
            **kwargs
        )

    def snn_language_calibrated_dense_label_confidence_internal_stability_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.calibrated_dense_label_confidence_internal_stability_review(
            **kwargs
        )

    def snn_language_calibrated_dense_label_confidence_autonomous_replay_review_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.calibrated_dense_label_confidence_autonomous_replay_review_design(
            **kwargs
        )

    def snn_language_calibrated_dense_label_confidence_autonomous_replay_review_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.calibrated_dense_label_confidence_autonomous_replay_review_preflight(
            **kwargs
        )

    def snn_language_calibrated_dense_label_confidence_autonomous_replay_review_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_calibrated_dense_label_confidence_autonomous_replay_review(
            **kwargs
        )

    def snn_language_calibrated_dense_label_confidence_autonomous_recalibration_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.calibrated_dense_label_confidence_autonomous_recalibration_design(
            **kwargs
        )

    def snn_language_calibrated_dense_label_confidence_autonomous_recalibration_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.calibrated_dense_label_confidence_autonomous_recalibration_preflight(
            **kwargs
        )

    def snn_language_calibrated_dense_label_confidence_autonomous_recalibration_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_calibrated_dense_label_confidence_autonomous_recalibration(
            **kwargs
        )

    def snn_language_calibrated_dense_label_confidence_autonomous_recalibration_application_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.calibrated_dense_label_confidence_autonomous_recalibration_application_review(
            **kwargs
        )

    def snn_language_calibrated_dense_label_confidence_autonomous_post_calibration_observation_window(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.calibrated_dense_label_confidence_autonomous_post_calibration_observation_window(
            **kwargs
        )

    def snn_language_calibrated_dense_label_confidence_autonomous_post_calibration_stability_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.calibrated_dense_label_confidence_autonomous_post_calibration_stability_review(
            **kwargs
        )

    def snn_language_calibrated_dense_label_confidence_autonomous_use_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.calibrated_dense_label_confidence_autonomous_use_design(
            **kwargs
        )

    def snn_language_calibrated_dense_label_confidence_autonomous_use_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.calibrated_dense_label_confidence_autonomous_use_preflight(
            **kwargs
        )

    def snn_language_calibrated_dense_label_confidence_autonomous_use_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_autonomous_calibrated_dense_label_confidence_use(
            **kwargs
        )

    def snn_language_calibrated_dense_label_confidence_autonomous_use_event_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_calibrated_dense_label_confidence_use_event_review(
            **kwargs
        )

    def snn_language_autonomous_hash_readout_binding_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_hash_readout_binding_design(
            **kwargs
        )

    def snn_language_autonomous_hash_readout_binding_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_hash_readout_binding_preflight(
            **kwargs
        )

    def snn_language_autonomous_hash_readout_binding_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_autonomous_hash_readout_binding(
            **kwargs
        )

    def snn_language_autonomous_hash_readout_binding_event_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_hash_readout_binding_event_review(
            **kwargs
        )

    def snn_language_autonomous_bound_readout_observation_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_bound_readout_observation_design(
            **kwargs
        )

    def snn_language_autonomous_bound_readout_observation_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_bound_readout_observation_preflight(
            **kwargs
        )

    def snn_language_autonomous_bound_readout_observation_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_autonomous_bound_readout_observation(
            **kwargs
        )

    def snn_language_autonomous_bound_readout_observation_event_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_bound_readout_observation_event_review(
            **kwargs
        )

    def snn_language_autonomous_readout_training_window_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_readout_training_window_design(
            **kwargs
        )

    def snn_language_autonomous_readout_training_window_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_readout_training_window_preflight(
            **kwargs
        )

    def snn_language_autonomous_readout_training_window_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_autonomous_readout_training_window(
            **kwargs
        )

    def snn_language_autonomous_readout_training_window_event_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_readout_training_window_event_review(
            **kwargs
        )

    def snn_language_autonomous_decoder_probe_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_decoder_probe_design(
            **kwargs
        )

    def snn_language_autonomous_decoder_probe_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_decoder_probe_preflight(
            **kwargs
        )

    def snn_language_autonomous_decoder_probe_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_autonomous_decoder_probe(
            **kwargs
        )

    def snn_language_autonomous_decoder_probe_event_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_decoder_probe_event_review(
            **kwargs
        )

    def snn_language_autonomous_language_output_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_language_output_design(
            **kwargs
        )

    def snn_language_autonomous_language_output_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_language_output_preflight(
            **kwargs
        )

    def snn_language_autonomous_language_output_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_autonomous_language_output(
            **kwargs
        )

    def snn_language_autonomous_language_output_event_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_language_output_event_review(
            **kwargs
        )

    def snn_language_autonomous_decoded_output_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_decoded_output_design(
            **kwargs
        )

    def snn_language_autonomous_decoded_output_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_decoded_output_preflight(
            **kwargs
        )

    def snn_language_autonomous_decoded_output_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_autonomous_decoded_output(
            **kwargs
        )

    def snn_language_autonomous_decoded_output_event_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_decoded_output_event_review(
            **kwargs
        )

    def snn_language_autonomous_bounded_text_emission_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_bounded_text_emission_design(
            **kwargs
        )

    def snn_language_autonomous_bounded_text_emission_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_bounded_text_emission_preflight(
            **kwargs
        )

    def snn_language_autonomous_bounded_text_emission_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_autonomous_bounded_text_emission(
            **kwargs
        )

    def snn_language_autonomous_bounded_text_emission_event_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_bounded_text_emission_event_review(
            **kwargs
        )

    def snn_language_autonomous_text_surface_sequence_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_text_surface_sequence_review(
            **kwargs
        )

    def snn_language_autonomous_text_surface_commit_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_text_surface_commit_design(
            **kwargs
        )

    def snn_language_autonomous_text_surface_commit_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_text_surface_commit_preflight(
            **kwargs
        )

    def snn_language_autonomous_text_surface_commit_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_autonomous_text_surface_commit(
            **kwargs
        )

    def snn_language_autonomous_text_surface_commit_event_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_text_surface_commit_event_review(
            **kwargs
        )

    def snn_language_autonomous_text_surface_materialization_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_text_surface_materialization_design(
            **kwargs
        )

    def snn_language_autonomous_text_surface_materialization_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_text_surface_materialization_preflight(
            **kwargs
        )

    def snn_language_autonomous_text_surface_materialization_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_autonomous_text_surface_materialization(
            **kwargs
        )

    def snn_language_autonomous_text_surface_materialization_event_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_text_surface_materialization_event_review(
            **kwargs
        )

    def snn_language_autonomous_bounded_language_surface_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_bounded_language_surface_review(
            **kwargs
        )

    def snn_language_autonomous_bounded_language_surface_commit_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_bounded_language_surface_commit_design(
            **kwargs
        )

    def snn_language_autonomous_bounded_language_surface_commit_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_bounded_language_surface_commit_preflight(
            **kwargs
        )

    def snn_language_autonomous_bounded_language_surface_commit_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_autonomous_bounded_language_surface_commit(
            **kwargs
        )

    def snn_language_autonomous_bounded_language_surface_commit_event_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_bounded_language_surface_commit_event_review(
            **kwargs
        )

    def snn_language_autonomous_bounded_language_surface_use_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_bounded_language_surface_use_review(
            **kwargs
        )

    def snn_language_autonomous_bounded_language_surface_use_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_bounded_language_surface_use_preflight(
            **kwargs
        )

    def snn_language_autonomous_bounded_language_surface_use_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_autonomous_bounded_language_surface_use(
            **kwargs
        )

    def snn_language_autonomous_bounded_language_surface_use_event_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_bounded_language_surface_use_event_review(
            **kwargs
        )

    def snn_language_autonomous_snn_language_generation_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_snn_language_generation_design(
            **kwargs
        )

    def snn_language_autonomous_snn_language_generation_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_snn_language_generation_preflight(
            **kwargs
        )

    def snn_language_autonomous_snn_language_generation_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_autonomous_snn_language_generation(
            **kwargs
        )

    def snn_language_autonomous_snn_language_generation_event_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_snn_language_generation_event_review(
            **kwargs
        )

    def snn_language_autonomous_snn_language_decoding_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_snn_language_decoding_design(
            **kwargs
        )

    def snn_language_autonomous_snn_language_decoding_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_snn_language_decoding_preflight(
            **kwargs
        )

    def snn_language_autonomous_snn_language_decoding_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_autonomous_snn_language_decoding(
            **kwargs
        )

    def snn_language_autonomous_snn_language_decoding_event_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_snn_language_decoding_event_review(
            **kwargs
        )

    def snn_language_readout_surface_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.snn_language_readout_surface_design(
            **kwargs
        )

    def snn_language_readout_surface_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.snn_language_readout_surface_preflight(
            **kwargs
        )

    def snn_language_readout_surface_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_snn_language_readout_surface(
            **kwargs
        )

    def snn_language_readout_surface_event_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.snn_language_readout_surface_event_review(
            **kwargs
        )

    def snn_language_readout_memory_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.snn_language_readout_memory_design(
            **kwargs
        )

    def snn_language_readout_memory_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.snn_language_readout_memory_preflight(
            **kwargs
        )

    def snn_language_readout_memory_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_snn_language_readout_memory(
            **kwargs
        )

    def snn_language_readout_memory_event_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.snn_language_readout_memory_event_review(
            **kwargs
        )

    def snn_language_readout_consolidation_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.snn_language_readout_consolidation_design(
            **kwargs
        )

    def snn_language_readout_consolidation_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.snn_language_readout_consolidation_preflight(
            **kwargs
        )

    def snn_language_readout_consolidation_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_snn_language_readout_consolidation(
            **kwargs
        )

    def snn_language_readout_consolidation_event_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.snn_language_readout_consolidation_event_review(
            **kwargs
        )

    def snn_language_readout_structural_plasticity_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.snn_language_readout_structural_plasticity_design(
            **kwargs
        )

    def snn_language_readout_structural_plasticity_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.snn_language_readout_structural_plasticity_preflight(
            **kwargs
        )

    def snn_language_readout_structural_plasticity_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.execute_snn_language_readout_structural_plasticity(
            **kwargs
        )

    def snn_language_readout_structural_plasticity_event_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.snn_language_readout_structural_plasticity_event_review(
            **kwargs
        )

    def snn_language_autonomous_snn_language_thought_capacity_mutation_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_snn_language_thought_capacity_mutation_design(
            **kwargs
        )

    def snn_language_autonomous_snn_language_thought_capacity_mutation_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_snn_language_thought_capacity_mutation_preflight(
            **kwargs
        )

    def snn_language_autonomous_snn_language_thought_capacity_mutation_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_plasticity_executor.apply_autonomous_snn_language_thought_capacity_mutation(
            **kwargs
        )

    def snn_language_autonomous_snn_language_thought_capacity_mutation_event_review(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("plasticity_runtime_state") is None:
            kwargs["plasticity_runtime_state"] = (
                self.snn_language_plasticity_runtime_state()
            )
        return self._root._snn_language_readout_ledger.autonomous_snn_language_thought_capacity_mutation_event_review(
            **kwargs
        )

    def snn_language_autonomous_snn_language_thought_newborn_neuron_integration_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_snn_language_thought_newborn_neuron_integration_design(
            **kwargs
        )

    def snn_language_autonomous_snn_language_thought_newborn_neuron_integration_preflight(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("plasticity_runtime_state") is None:
            kwargs["plasticity_runtime_state"] = (
                self.snn_language_plasticity_runtime_state()
            )
        return self._root._snn_language_readout_ledger.autonomous_snn_language_thought_newborn_neuron_integration_preflight(
            **kwargs
        )

    def snn_language_autonomous_snn_language_thought_newborn_neuron_integration_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_plasticity_executor.apply_autonomous_snn_language_thought_newborn_neuron_integration(
            **kwargs
        )

    def snn_language_autonomous_snn_language_thought_newborn_neuron_integration_event_review(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("plasticity_runtime_state") is None:
            kwargs["plasticity_runtime_state"] = (
                self.snn_language_plasticity_runtime_state()
            )
        return self._root._snn_language_readout_ledger.autonomous_snn_language_thought_newborn_neuron_integration_event_review(
            **kwargs
        )

    def snn_language_autonomous_snn_language_thought_newborn_neuron_critical_period_learning_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_snn_language_thought_newborn_neuron_critical_period_learning_design(
            **kwargs
        )

    def snn_language_autonomous_snn_language_thought_newborn_neuron_critical_period_learning_preflight(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("plasticity_runtime_state") is None:
            kwargs["plasticity_runtime_state"] = (
                self.snn_language_plasticity_runtime_state()
            )
        return self._root._snn_language_readout_ledger.autonomous_snn_language_thought_newborn_neuron_critical_period_learning_preflight(
            **kwargs
        )

    def snn_language_autonomous_snn_language_thought_newborn_neuron_critical_period_learning_executor(self, **kwargs: Any) -> dict[str, Any]:
        result = self._root._snn_language_plasticity_executor.apply_autonomous_snn_language_thought_newborn_neuron_critical_period_learning(
            **kwargs
        )
        if result.get("accepted"):
            result["developmental_autonomy"] = (
                self._root._developmental_autonomy.run_after_tick(
                    learning_executor=result
                )
            )
        return result

    def snn_language_autonomous_snn_language_thought_newborn_neuron_critical_period_learning_event_review(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("plasticity_runtime_state") is None:
            kwargs["plasticity_runtime_state"] = (
                self.snn_language_plasticity_runtime_state()
            )
        return self._root._snn_language_readout_ledger.autonomous_snn_language_thought_newborn_neuron_critical_period_learning_event_review(
            **kwargs
        )

    def snn_language_autonomous_snn_language_thought_newborn_neuron_critical_period_learning_continuation_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_snn_language_thought_newborn_neuron_critical_period_learning_continuation_design(
            **kwargs
        )

    def snn_language_autonomous_snn_language_thought_newborn_neuron_maturation_outcome_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_snn_language_thought_newborn_neuron_maturation_outcome_review(
            **kwargs
        )

    def snn_language_autonomous_snn_language_thought_newborn_synapse_pruning_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.autonomous_snn_language_thought_newborn_synapse_pruning_design(
            **kwargs
        )

    def snn_language_autonomous_snn_language_thought_newborn_synapse_pruning_preflight(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("plasticity_runtime_state") is None:
            kwargs["plasticity_runtime_state"] = (
                self.snn_language_plasticity_runtime_state()
            )
        return self._root._snn_language_readout_ledger.autonomous_snn_language_thought_newborn_synapse_pruning_preflight(
            **kwargs
        )

    def snn_language_autonomous_snn_language_thought_newborn_synapse_pruning_executor(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_plasticity_executor.apply_autonomous_snn_language_thought_newborn_synapse_pruning(
            **kwargs
        )

    def snn_language_readout_emission_replay_evaluation_policy(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.emission_review_replay_evaluation_policy(**kwargs)

    def snn_language_readout_emission_replay_evaluation_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.emission_review_replay_evaluation_design(**kwargs)

    def snn_language_readout_emission_replay_context_review(self, **kwargs: Any) -> dict[str, Any]:
        design = dict(kwargs.pop("emission_replay_evaluation_design"))
        seed_source_window_surface = (
            "bounded_snn_emission_replay_context_review_seed_window.v1"
        )
        observed_slot_source_window_surface = (
            "bounded_snn_emission_replay_context_review_observed_slot_window.v1"
        )
        seeds, seed_source_window = (
            self._root._snn_language_readout_ledger._bounded_replay_payload_window(
                design.get("selected_replay_context_seeds"),
                source=(
                    "runtime_facade.snn_language_readout_emission_replay_context_review."
                    "selected_replay_context_seeds"
                ),
                surface=seed_source_window_surface,
                active_replay_computation_device="cpu",
            )
        )
        observed_slots, observed_slot_source_window = (
            self._root._snn_language_readout_ledger._bounded_replay_payload_window(
                kwargs.pop("observed_readout_slots"),
                source=(
                    "runtime_facade.snn_language_readout_emission_replay_context_review."
                    "observed_readout_slots"
                ),
                surface=observed_slot_source_window_surface,
                active_replay_computation_device="cpu",
            )
        )
        prediction_report = dict(kwargs.pop("prediction_report"))
        operator_id = str(kwargs.pop("operator_id", "") or "").strip()
        confirmation = bool(kwargs.pop("confirmation", False))
        gate = (
            design.get("promotion_gate")
            if isinstance(design.get("promotion_gate"), Mapping)
            else {}
        )
        provenance = (
            prediction_report.get("provenance_evidence")
            if isinstance(prediction_report.get("provenance_evidence"), Mapping)
            else {}
        )
        prediction_hash = str(provenance.get("prediction_hash") or "")
        design_payload = (
            design.get("emission_replay_evaluation_design")
            if isinstance(design.get("emission_replay_evaluation_design"), Mapping)
            else {}
        )
        matched_seed = next(
            (
                seed
                for seed in seeds
                if bool(seed.get("eligible_for_replay_context_review"))
                and str(seed.get("prediction_hash") or "") == prediction_hash
            ),
            None,
        )
        required = {
            "design_surface_available": design.get("surface")
            == "snn_language_readout_emission_replay_evaluation_design.v1",
            "design_gate_ready": bool(
                gate.get("eligible_for_operator_replay_context_review")
            ),
            "design_does_not_record_replay_context": (
                design.get("records_ledger_event") is False
                and bool(design_payload.get("records_replay_context")) is False
            ),
            "seed_source_window_bounded": self._readout_replay_payload_window_bounded(
                seed_source_window,
                surface=seed_source_window_surface,
            ),
            "seed_payload_not_truncated": not bool(
                seed_source_window.get("source_payload_truncated")
            ),
            "seed_payload_well_formed": int(
                seed_source_window.get("source_mapping_count", 0) or 0
            )
            == int(seed_source_window.get("source_window_count", 0) or 0),
            "observed_slot_source_window_bounded": (
                self._readout_replay_payload_window_bounded(
                    observed_slot_source_window,
                    surface=observed_slot_source_window_surface,
                )
            ),
            "observed_slot_payload_not_truncated": not bool(
                observed_slot_source_window.get("source_payload_truncated")
            ),
            "observed_slot_payload_well_formed": int(
                observed_slot_source_window.get("source_mapping_count", 0) or 0
            )
            == int(observed_slot_source_window.get("source_window_count", 0) or 0),
            "prediction_report_surface_available": prediction_report.get("surface")
            == "snn_language_sequence_prediction_probe.v1",
            "prediction_hash_available": bool(prediction_hash),
            "prediction_hash_matches_design_seed": matched_seed is not None,
            "observed_readout_slots_available": bool(observed_slots),
            "operator_id_available": bool(operator_id),
            "operator_confirmation": confirmation,
        }
        ready = all(required.values())

        def blocked() -> dict[str, Any]:
            return {
                "artifact_kind": (
                    "terminus_snn_language_readout_emission_replay_context_review"
                ),
                "surface": "snn_language_readout_emission_replay_context_review.v1",
                "accepted": False,
                "ready": False,
                "owned_by_marulho": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "records_replay_context": False,
                "records_ledger_event": False,
                "runs_replay": False,
                "writes_checkpoint": False,
                "generates_text": False,
                "decodes_text": False,
                "exposes_reviewed_bounded_text": False,
                "trains_runtime_model": False,
                "applies_plasticity": False,
                "mutates_runtime_state": False,
                "eligible_for_replay_memory": False,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "review": {
                    "operator_id": operator_id or None,
                    "prediction_hash": prediction_hash or None,
                    "matched_design_seed_hash": (
                        matched_seed.get("replay_context_seed_hash")
                        if matched_seed
                        else None
                    ),
                    "replay_evaluation_context_id": None,
                    "replay_evaluation_context_hash": None,
                },
                "seed_source_window": dict(seed_source_window),
                "observed_slot_source_window": dict(observed_slot_source_window),
                "promotion_gate": {
                    "status": "blocked_missing_emission_replay_context_review_evidence",
                    "eligible_for_replay_context_recording": False,
                    "eligible_for_replay_memory": False,
                    "eligible_for_live_replay": False,
                    "eligible_for_plasticity_application": False,
                    "eligible_for_fact_promotion": False,
                    "eligible_for_action": False,
                    "required_evidence": required,
                },
            }

        if not ready:
            return blocked()

        mismatch = self._root._status_read_model.snn_language_sequence_mismatch_probe(
            prediction_report=prediction_report,
            observed_readout_slots=observed_slots,
            device_evidence=kwargs.pop("device_evidence", None),
        )
        pressure = self._root._status_read_model.snn_language_plasticity_pressure(
            mismatch_report=mismatch,
            runtime_truth_delta=kwargs.pop("runtime_truth_delta", None),
            rollback_policy=kwargs.pop("rollback_policy", None),
        )
        source_metadata = {
            "source": "runtime_facade.snn_language_readout_emission_replay_context_review",
            "surface": design.get("surface"),
            "design_hash": design_payload.get("design_hash"),
            "seed_hash": matched_seed.get("replay_context_seed_hash")
            if matched_seed
            else None,
            "emission_review_hash": matched_seed.get("emission_review_hash")
            if matched_seed
            else None,
            "emission_hash": matched_seed.get("emission_hash") if matched_seed else None,
            "readout_evidence_hash": matched_seed.get("readout_evidence_hash")
            if matched_seed
            else None,
            "prediction_hash": prediction_hash,
            "operator_id": operator_id,
        }
        context = self._root._replay_controller.record_snn_replay_evaluation_context(
            mismatch_report=mismatch,
            pressure_report=pressure,
            source_metadata=source_metadata,
        )
        review_material = {
            "design_hash": design_payload.get("design_hash"),
            "seed_hash": matched_seed.get("replay_context_seed_hash")
            if matched_seed
            else None,
            "prediction_hash": prediction_hash,
            "replay_evaluation_context_hash": context.get("evidence_hash"),
            "operator_id": operator_id,
        }
        review_hash = self._root._snn_language_readout_ledger._sha256_json(
            review_material
        )
        return {
            "artifact_kind": (
                "terminus_snn_language_readout_emission_replay_context_review"
            ),
            "surface": "snn_language_readout_emission_replay_context_review.v1",
            "accepted": True,
            "ready": True,
            "owned_by_marulho": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "records_replay_context": True,
            "records_ledger_event": False,
            "runs_replay": False,
            "writes_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "exposes_reviewed_bounded_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": True,
            "eligible_for_replay_memory": False,
            "eligible_for_live_replay": False,
            "eligible_for_plasticity_application": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_action": False,
            "review": {
                "review_hash": review_hash,
                "operator_id": operator_id,
                "prediction_hash": prediction_hash,
                "matched_design_seed_hash": matched_seed.get("replay_context_seed_hash"),
                "readout_evidence_hash": matched_seed.get("readout_evidence_hash"),
                "emission_hash": matched_seed.get("emission_hash"),
                "replay_evaluation_context_id": context.get(
                    "replay_evaluation_context_id"
                ),
                "replay_evaluation_context_hash": context.get("evidence_hash"),
                "replay_evaluation_context_source_metadata_hash": context.get(
                    "source_metadata_hash"
                ),
                "mismatch_hash": context.get("mismatch_hash"),
                "pressure_hash": context.get("pressure_hash"),
            },
            "seed_source_window": dict(seed_source_window),
            "observed_slot_source_window": dict(observed_slot_source_window),
            "promotion_gate": {
                "status": "replay_evaluation_context_recorded_for_operator_review",
                "eligible_for_replay_context_recording": False,
                "eligible_for_replay_memory": False,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "next_gate": (
                    "/terminus/snn-language-sequence/"
                    "replay-consolidation-priority-queue"
                ),
                "required_evidence": required,
            },
        }

    def snn_language_readout_rollout_candidate(self, **kwargs: Any) -> dict[str, Any]:
        state = dict(self.snn_language_plasticity_runtime_state())
        state["transition_memory_state_source"] = (
            "service.runtime_facade.snn_language_plasticity_runtime_state"
        )
        state["current_state_revision"] = int(self._root._runtime_state.state_revision)
        kwargs["transition_memory_state"] = state
        return self._root._status_read_model.snn_language_readout_rollout_candidate(**kwargs)

    def snn_language_readout_rollout_replay_evaluation(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_readout_rollout_replay_evaluation(**kwargs)

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

    def snn_language_readout_synapse_provenance_audit(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("plasticity_runtime_state") is None:
            kwargs["plasticity_runtime_state"] = self.snn_language_plasticity_runtime_state()
        if kwargs.get("applied_replay_lineage_restore_validation") is None:
            validation = self._applied_replay_lineage_restore_validation()
            if isinstance(validation, Mapping):
                kwargs["applied_replay_lineage_restore_validation"] = dict(validation)
        return self._root._snn_language_readout_ledger.synapse_provenance_audit(**kwargs)

    def _applied_replay_lineage_restore_validation(self) -> dict[str, Any]:
        metadata = getattr(self._root, "_metadata", {})
        service_state = (
            metadata.get("service_state")
            if isinstance(metadata, Mapping)
            and isinstance(metadata.get("service_state"), Mapping)
            else {}
        )
        validation = service_state.get("snn_applied_replay_lineage_restore_validation")
        return dict(validation) if isinstance(validation, Mapping) else {}

    def _applied_replay_lineage_restore_validation_not_mismatched(self) -> bool:
        validation = self._applied_replay_lineage_restore_validation()
        available = (
            validation.get("surface")
            == "snn_applied_replay_lineage_restore_validation.v1"
        )
        return bool(not available or validation.get("summary_matches_restored_state"))

    def snn_language_readout_evidence_ledger_record(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.record_readout_draft(**kwargs)

    def snn_language_readout_rollout_evidence_ledger_record(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.record_readout_rollout_replay_evaluation(**kwargs)

    def snn_language_readout_rollout_rehearsal_promotion_policy(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.rollout_rehearsal_promotion_policy(**kwargs)

    def snn_language_readout_rollout_rehearsal_evaluation(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.rollout_rehearsal_evaluation(**kwargs)

    def snn_language_readout_rollout_rehearsal_experiment(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.rollout_rehearsal_experiment(**kwargs)

    def snn_language_readout_rollout_consolidation_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.rollout_consolidation_design(**kwargs)

    def snn_language_readout_rollout_consolidation_shadow_delta(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.rollout_consolidation_shadow_delta(**kwargs)

    def snn_language_readout_rollout_consolidation_shadow_application_preflight(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if kwargs.get("transition_memory_state") is None:
            kwargs["transition_memory_state"] = self.snn_language_plasticity_runtime_state()
        return self._root._snn_language_readout_ledger.rollout_consolidation_shadow_application_preflight(
            **kwargs
        )

    def snn_language_readout_rollout_developmental_plasticity_review(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if kwargs.get("transition_memory_state") is None:
            kwargs["transition_memory_state"] = self.snn_language_plasticity_runtime_state()
        return self._root._snn_language_readout_ledger.rollout_developmental_plasticity_review(
            **kwargs
        )

    def snn_language_readout_rollout_regeneration_proposal_adapter(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.rollout_regeneration_proposal_adapter(
            **kwargs
        )

    def snn_language_readout_rollout_regeneration_replay_artifact_review(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._root._snn_language_readout_ledger.rollout_regeneration_replay_artifact_review(
            **kwargs
        )

    def snn_language_readout_rollout_regeneration_permit_request(
        self,
        *,
        rollout_regeneration_replay_artifact_review: Mapping[str, Any],
        operator_id: str,
        confirmation: bool,
    ) -> dict[str, Any]:
        review = dict(rollout_regeneration_replay_artifact_review)
        gate = review.get("promotion_gate") if isinstance(review.get("promotion_gate"), Mapping) else {}
        preview = (
            review.get("permit_request_preview")
            if isinstance(review.get("permit_request_preview"), Mapping)
            else {}
        )
        language_capacity = self._snn_language_capacity_state(review)
        language_neuron_count = int(language_capacity["language_neuron_count"])
        regeneration_design = (
            preview.get("regeneration_design")
            if isinstance(preview.get("regeneration_design"), Mapping)
            else {}
        )
        candidate_window_surface = (
            "bounded_snn_rollout_regeneration_permit_candidate_synapse_window.v1"
        )
        candidates, candidate_source_window = self._rollout_regeneration_candidate_window(
            regeneration_design.get("candidate_synapses"),
            source=(
                "service.runtime_facade."
                "rollout_regeneration_permit_candidate_synapses"
            ),
            surface=candidate_window_surface,
            field_name=(
                "permit_request_preview.regeneration_design.candidate_synapses"
            ),
        )
        bounded_regeneration_design = {
            **dict(regeneration_design),
            "candidate_synapses": candidates,
        }
        before_revision = int(self._root._runtime_state.state_revision)
        restore_validation_not_mismatched = (
            self._applied_replay_lineage_restore_validation_not_mismatched()
        )
        required = {
            "confirmation": bool(confirmation),
            "operator_id_available": bool(str(operator_id or "").strip()),
            "review_surface_available": review.get("surface")
            == "snn_language_readout_rollout_regeneration_replay_artifact_review.v1",
            "review_owned_by_marulho": bool(review.get("owned_by_marulho")),
            "review_gate_ready": bool(
                gate.get("eligible_for_regeneration_permit_request")
            ),
            "review_does_not_apply_plasticity": not bool(review.get("applies_plasticity")),
            "review_does_not_mutate_synapses": not bool(review.get("mutates_runtime_state")),
            "review_has_no_existing_permit": not bool(
                preview.get("permit_issued")
            ),
            "replay_artifact_id_available": bool(str(preview.get("replay_artifact_id") or "")),
            "regeneration_design_available": bool(regeneration_design),
            "candidate_source_window_bounded": (
                self._rollout_regeneration_candidate_window_bounded(
                    candidate_source_window,
                    surface=candidate_window_surface,
                )
            ),
            "candidate_payload_not_truncated": not bool(
                candidate_source_window.get("source_payload_truncated")
            ),
            "regeneration_design_indices_canonical": all(
                0 <= int(item.get("pre_index", -1)) < language_neuron_count
                and 0 <= int(item.get("post_index", -1)) < language_neuron_count
                for item in candidates
            ),
            "language_capacity_state_available": bool(language_capacity["present"]),
            "language_capacity_state_dynamic_limits_applied": True,
            "applied_replay_lineage_restore_validation_not_mismatched": (
                restore_validation_not_mismatched
            ),
        }
        required_evidence = {
            **required,
            "candidate_source_window": dict(candidate_source_window),
        }
        if not all(required.values()):
            return {
                "artifact_kind": "terminus_snn_language_readout_rollout_regeneration_permit_request",
                "surface": "snn_language_readout_rollout_regeneration_permit_request.v1",
                "accepted": False,
                "available": bool(review),
                "status": "blocked_missing_rollout_regeneration_permit_request_evidence",
                "owned_by_marulho": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "generates_text": False,
                "decodes_text": False,
                "trains_runtime_model": False,
                "applies_plasticity": False,
                "mutates_runtime_state": False,
                "returns_trained_weights": False,
                "issues_regeneration_permit": False,
                "executor_ready": False,
                "language_capacity": language_capacity,
                "candidate_source_window": dict(candidate_source_window),
                "before": {"state_revision": before_revision},
                "after": {"state_revision": int(self._root._runtime_state.state_revision)},
                "promotion_gate": {
                    "status": "blocked_missing_rollout_regeneration_permit_request_evidence",
                    "eligible_for_regeneration_application": False,
                    "eligible_for_structural_write": False,
                    "eligible_for_growth": False,
                    "eligible_for_pruning": False,
                    "eligible_for_plasticity_application": False,
                    "eligible_for_action": False,
                    "required_evidence": required_evidence,
                },
            }
        try:
            permit = self._root._replay_controller.issue_regeneration_permit(
                replay_artifact_id=str(preview.get("replay_artifact_id") or ""),
                regeneration_design=dict(bounded_regeneration_design),
                operator_id=operator_id,
                confirmation=confirmation,
            )
        except ValueError as exc:
            return {
                "artifact_kind": "terminus_snn_language_readout_rollout_regeneration_permit_request",
                "surface": "snn_language_readout_rollout_regeneration_permit_request.v1",
                "accepted": False,
                "available": True,
                "status": "blocked_replay_controller_regeneration_permit_rejected",
                "reason": str(exc),
                "owned_by_marulho": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "generates_text": False,
                "decodes_text": False,
                "trains_runtime_model": False,
                "applies_plasticity": False,
                "mutates_runtime_state": False,
                "returns_trained_weights": False,
                "issues_regeneration_permit": False,
                "executor_ready": False,
                "language_capacity": language_capacity,
                "candidate_source_window": dict(candidate_source_window),
                "before": {"state_revision": before_revision},
                "after": {"state_revision": int(self._root._runtime_state.state_revision)},
                "promotion_gate": {
                    "status": "blocked_replay_controller_regeneration_permit_rejected",
                    "eligible_for_regeneration_application": False,
                    "eligible_for_structural_write": False,
                    "eligible_for_growth": False,
                    "eligible_for_pruning": False,
                    "eligible_for_plasticity_application": False,
                    "eligible_for_action": False,
                    "required_evidence": {
                        **required_evidence,
                        "replay_controller_permit_issued": False,
                    },
                },
            }
        return {
            "artifact_kind": "terminus_snn_language_readout_rollout_regeneration_permit_request",
            "surface": "snn_language_readout_rollout_regeneration_permit_request.v1",
            "accepted": True,
            "available": True,
            "status": "regeneration_permit_issued",
            "owned_by_marulho": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": True,
            "returns_trained_weights": False,
            "issues_regeneration_permit": True,
            "executor_ready": False,
            "rollout_regeneration_replay_artifact_review_hash": review.get(
                "rollout_regeneration_replay_artifact_review_hash"
            ),
            "replay_evidence": permit,
            "language_capacity": language_capacity,
            "regeneration_design": dict(bounded_regeneration_design),
            "candidate_source_window": dict(candidate_source_window),
            "before": {"state_revision": before_revision},
            "after": {
                "state_revision": int(self._root._runtime_state.state_revision),
                "dirty_state": bool(self._root._runtime_state.dirty_state),
            },
            "promotion_gate": {
                "status": "ready_for_checkpoint_backed_regeneration_application",
                "eligible_for_regeneration_application": True,
                "eligible_for_structural_write": False,
                "eligible_for_growth": False,
                "eligible_for_pruning": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_action": False,
                "next_gate": "checkpoint_backed_snn_transition_memory_regeneration",
                "required_evidence": {
                    **required_evidence,
                    "replay_controller_permit_issued": True,
                    "checkpoint_executor_still_required": True,
                },
            },
        }

    def snn_language_readout_rollout_regeneration_application_preflight(
        self,
        *,
        rollout_regeneration_permit_request: Mapping[str, Any],
        expected_state_revision: int,
        checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        request = dict(rollout_regeneration_permit_request)
        gate = request.get("promotion_gate") if isinstance(request.get("promotion_gate"), Mapping) else {}
        permit = request.get("replay_evidence") if isinstance(request.get("replay_evidence"), Mapping) else {}
        design = (
            request.get("regeneration_design")
            if isinstance(request.get("regeneration_design"), Mapping)
            else {}
        )
        language_capacity = self._snn_language_capacity_state(request)
        language_neuron_count = int(language_capacity["language_neuron_count"])
        candidate_window_surface = (
            "bounded_snn_rollout_regeneration_application_preflight_"
            "candidate_synapse_window.v1"
        )
        candidates, candidate_source_window = self._rollout_regeneration_candidate_window(
            design.get("candidate_synapses"),
            source=(
                "service.runtime_facade."
                "rollout_regeneration_application_preflight_candidate_synapses"
            ),
            surface=candidate_window_surface,
            field_name="regeneration_design.candidate_synapses",
        )
        bounded_regeneration_design = {
            **dict(design),
            "candidate_synapses": candidates,
        }
        before_revision = int(self._root._runtime_state.state_revision)
        checkpoint = str(checkpoint_path or "").strip()
        request_required = (
            gate.get("required_evidence")
            if isinstance(gate.get("required_evidence"), Mapping)
            else {}
        )
        required = {
            "permit_request_surface_available": request.get("surface")
            == "snn_language_readout_rollout_regeneration_permit_request.v1",
            "permit_request_accepted": bool(request.get("accepted")),
            "permit_request_owned_by_marulho": bool(request.get("owned_by_marulho")),
            "permit_request_gate_ready": bool(gate.get("eligible_for_regeneration_application")),
            "expected_revision_current": int(expected_state_revision) == before_revision,
            "checkpoint_path_available": bool(checkpoint),
            "permit_available": bool(permit.get("permit_id")),
            "permit_ready": bool(permit.get("ready")),
            "permit_owned_by_marulho": bool(permit.get("owned_by_marulho")),
            "regeneration_design_available": bool(design),
            "candidate_source_window_bounded": (
                self._rollout_regeneration_candidate_window_bounded(
                    candidate_source_window,
                    surface=candidate_window_surface,
                )
            ),
            "candidate_payload_not_truncated": not bool(
                candidate_source_window.get("source_payload_truncated")
            ),
            "regeneration_design_indices_canonical": all(
                0 <= int(item.get("pre_index", -1)) < language_neuron_count
                and 0 <= int(item.get("post_index", -1)) < language_neuron_count
                for item in candidates
            ),
            "language_capacity_state_available": bool(language_capacity["present"]),
            "language_capacity_state_dynamic_limits_applied": True,
            "permit_request_does_not_apply_plasticity": not bool(request.get("applies_plasticity")),
            "permit_request_does_not_checkpoint": not bool(request.get("checkpoint_written")),
            "applied_replay_lineage_restore_validation_not_mismatched": (
                bool(
                    request_required.get(
                        "applied_replay_lineage_restore_validation_not_mismatched",
                        True,
                    )
                )
                and self._applied_replay_lineage_restore_validation_not_mismatched()
            ),
        }
        required_evidence = {
            **required,
            "candidate_source_window": dict(candidate_source_window),
        }
        ready = all(required.values())
        proposal = {
            "available": ready,
            "ready": ready,
            "owned_by_marulho": True,
            "source": "service.runtime_facade.rollout_regeneration_application_preflight",
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "replay_evidence": dict(permit),
            "language_capacity": language_capacity,
            "regeneration_design": dict(bounded_regeneration_design),
            "candidate_source_window": dict(candidate_source_window),
            "promotion_gate": {
                "status": "ready_for_operator_review"
                if ready
                else "blocked_missing_rollout_regeneration_application_preflight_evidence"
            },
        }
        return {
            "artifact_kind": "terminus_snn_language_readout_rollout_regeneration_application_preflight",
            "surface": "snn_language_readout_rollout_regeneration_application_preflight.v1",
            "available": bool(request),
            "ready": ready,
            "owned_by_marulho": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "returns_trained_weights": False,
            "writes_checkpoint": False,
            "executor_called": False,
            "expected_state_revision": int(expected_state_revision),
            "checkpoint_path": checkpoint or None,
            "language_capacity": language_capacity,
            "candidate_source_window": dict(candidate_source_window),
            "regeneration_proposal": proposal,
            "before": {"state_revision": before_revision},
            "after": {"state_revision": int(self._root._runtime_state.state_revision)},
            "promotion_gate": {
                "status": "ready_for_checkpoint_backed_regeneration_executor"
                if ready
                else "blocked_missing_rollout_regeneration_application_preflight_evidence",
                "eligible_for_checkpoint_backed_regeneration_executor": ready,
                "eligible_for_regeneration_application": False,
                "eligible_for_structural_write": False,
                "eligible_for_growth": False,
                "eligible_for_pruning": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_action": False,
                "requires_operator_confirmation_at_executor": ready,
                "next_gate": "checkpoint_backed_snn_transition_memory_regeneration"
                if ready
                else "collect_rollout_regeneration_permit_and_checkpoint_evidence",
                "required_evidence": required_evidence,
            },
        }

    def snn_language_readout_rollout_regeneration_application(
        self,
        *,
        rollout_regeneration_application_preflight: Mapping[str, Any],
        expected_state_revision: int,
        operator_id: str,
        confirmation: bool,
        checkpoint_path: str | None = None,
        max_outgoing_row_mass: float = 1.0,
    ) -> dict[str, Any]:
        preflight = dict(rollout_regeneration_application_preflight)
        gate = preflight.get("promotion_gate") if isinstance(preflight.get("promotion_gate"), Mapping) else {}
        proposal = (
            preflight.get("regeneration_proposal")
            if isinstance(preflight.get("regeneration_proposal"), Mapping)
            else {}
        )
        language_capacity = self._snn_language_capacity_state(preflight)
        proposal_language_capacity = self._snn_language_capacity_state(proposal)
        language_neuron_count = int(language_capacity["language_neuron_count"])
        design = (
            proposal.get("regeneration_design")
            if isinstance(proposal.get("regeneration_design"), Mapping)
            else {}
        )
        candidate_window_surface = (
            "bounded_snn_rollout_regeneration_application_candidate_synapse_"
            "window.v1"
        )
        candidates, candidate_source_window = self._rollout_regeneration_candidate_window(
            design.get("candidate_synapses"),
            source=(
                "service.runtime_facade."
                "rollout_regeneration_application_candidate_synapses"
            ),
            surface=candidate_window_surface,
            field_name="regeneration_proposal.regeneration_design.candidate_synapses",
        )
        bounded_design = {
            **dict(design),
            "candidate_synapses": candidates,
        }
        bounded_proposal = {
            **dict(proposal),
            "regeneration_design": bounded_design,
            "candidate_source_window": dict(candidate_source_window),
        }
        preflight_checkpoint = str(preflight.get("checkpoint_path") or "").strip()
        requested_checkpoint = str(checkpoint_path or "").strip()
        effective_checkpoint = requested_checkpoint or preflight_checkpoint
        before_revision = int(self._root._runtime_state.state_revision)
        required = {
            "preflight_surface_available": preflight.get("surface")
            == "snn_language_readout_rollout_regeneration_application_preflight.v1",
            "preflight_ready": bool(preflight.get("ready")),
            "preflight_owned_by_marulho": bool(preflight.get("owned_by_marulho")),
            "preflight_gate_ready": bool(
                gate.get("eligible_for_checkpoint_backed_regeneration_executor")
            ),
            "preflight_does_not_apply_plasticity": not bool(preflight.get("applies_plasticity")),
            "preflight_does_not_mutate_runtime": not bool(preflight.get("mutates_runtime_state")),
            "preflight_did_not_call_executor": not bool(preflight.get("executor_called")),
            "expected_revision_current": int(expected_state_revision) == before_revision,
            "expected_revision_matches_preflight": int(preflight.get("expected_state_revision", -1))
            == int(expected_state_revision),
            "checkpoint_path_available": bool(effective_checkpoint),
            "checkpoint_path_matches_preflight": not bool(requested_checkpoint and preflight_checkpoint)
            or requested_checkpoint == preflight_checkpoint,
            "operator_id_available": bool(str(operator_id or "").strip()),
            "confirmation": bool(confirmation),
            "proposal_available": bool(proposal.get("available")),
            "proposal_ready": bool(proposal.get("ready")),
            "proposal_owned_by_marulho": bool(proposal.get("owned_by_marulho")),
            "proposal_does_not_generate_text": not bool(proposal.get("generates_text")),
            "proposal_does_not_load_external_checkpoint": not bool(
                proposal.get("loads_external_checkpoint")
            ),
            "regeneration_design_available": bool(design),
            "candidate_source_window_bounded": (
                self._rollout_regeneration_candidate_window_bounded(
                    candidate_source_window,
                    surface=candidate_window_surface,
                )
            ),
            "candidate_payload_not_truncated": not bool(
                candidate_source_window.get("source_payload_truncated")
            ),
            "regeneration_design_indices_canonical": all(
                0 <= int(item.get("pre_index", -1)) < language_neuron_count
                and 0 <= int(item.get("post_index", -1)) < language_neuron_count
                for item in candidates
            ),
            "language_capacity_state_available": bool(language_capacity["present"]),
            "proposal_language_capacity_state_available": bool(
                proposal_language_capacity["present"]
            ),
            "proposal_language_capacity_matches_preflight": (
                int(proposal_language_capacity["language_neuron_count"])
                == int(language_capacity["language_neuron_count"])
                and int(proposal_language_capacity["sparse_edge_budget"])
                == int(language_capacity["sparse_edge_budget"])
                and int(proposal_language_capacity["outgoing_fanout_budget"])
                == int(language_capacity["outgoing_fanout_budget"])
            ),
            "language_capacity_state_dynamic_limits_applied": True,
        }
        required_evidence = {
            **required,
            "candidate_source_window": dict(candidate_source_window),
        }
        if not all(required.values()):
            return {
                "artifact_kind": "terminus_snn_language_readout_rollout_regeneration_application",
                "surface": "snn_language_readout_rollout_regeneration_application.v1",
                "accepted": False,
                "available": bool(preflight),
                "status": "blocked_missing_rollout_regeneration_application_evidence",
                "reason": "blocked_missing_rollout_regeneration_application_evidence",
                "owned_by_marulho": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "generates_text": False,
                "decodes_text": False,
                "trains_runtime_model": False,
                "applies_plasticity": False,
                "mutates_runtime_state": False,
                "returns_trained_weights": False,
                "writes_checkpoint": False,
                "executor_called": False,
                "checkpoint_path": effective_checkpoint or None,
                "language_capacity": language_capacity,
                "proposal_language_capacity": proposal_language_capacity,
                "candidate_source_window": dict(candidate_source_window),
                "before": {"state_revision": before_revision},
                "after": {"state_revision": int(self._root._runtime_state.state_revision)},
                "promotion_gate": {
                    "status": "blocked_missing_rollout_regeneration_application_evidence",
                    "eligible_for_checkpoint_backed_regeneration_executor": False,
                    "eligible_for_structural_write": False,
                    "eligible_for_growth": False,
                    "eligible_for_pruning": False,
                    "eligible_for_plasticity_application": False,
                    "eligible_for_action": False,
                    "required_evidence": required_evidence,
                },
            }

        executor_result = self._root._snn_language_plasticity_executor.regenerate_transition_memory(
            regeneration_proposal=dict(bounded_proposal),
            expected_state_revision=expected_state_revision,
            operator_id=operator_id,
            confirmation=confirmation,
            checkpoint_path=effective_checkpoint,
            max_outgoing_row_mass=max_outgoing_row_mass,
        )
        checkpoint_transaction = executor_result.get("checkpoint_transaction")
        writes_checkpoint = bool(
            isinstance(checkpoint_transaction, Mapping)
            and checkpoint_transaction.get("pre_regeneration_checkpoint_saved")
        )
        accepted = bool(executor_result.get("accepted"))
        return {
            "artifact_kind": "terminus_snn_language_readout_rollout_regeneration_application",
            "surface": "snn_language_readout_rollout_regeneration_application.v1",
            "accepted": accepted,
            "available": True,
            "status": "regeneration_applied"
            if accepted
            else "blocked_by_checkpoint_backed_regeneration_executor",
            "reason": executor_result.get("reason"),
            "owned_by_marulho": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": bool(executor_result.get("applies_plasticity")),
            "mutates_runtime_state": bool(executor_result.get("mutates_runtime_state")),
            "returns_trained_weights": False,
            "writes_checkpoint": writes_checkpoint,
            "executor_called": True,
            "checkpoint_path": effective_checkpoint,
            "language_capacity": language_capacity,
            "proposal_language_capacity": proposal_language_capacity,
            "candidate_source_window": dict(candidate_source_window),
            "executor_result": executor_result,
            "before": {"state_revision": before_revision},
            "after": {"state_revision": int(self._root._runtime_state.state_revision)},
            "promotion_gate": {
                "status": "checkpoint_backed_regeneration_applied"
                if accepted
                else "checkpoint_backed_regeneration_executor_blocked",
                "eligible_for_checkpoint_backed_regeneration_executor": False,
                "eligible_for_structural_write": False,
                "eligible_for_growth": False,
                "eligible_for_pruning": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_action": False,
                "required_evidence": required_evidence,
            },
        }

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

    def snn_language_capacity_expansion_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_capacity_expansion_design(
            **kwargs
        )

    def snn_language_capacity_expansion_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_capacity_expansion_preflight(
            **kwargs
        )

    def snn_language_capacity_resize_compatibility_audit(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_capacity_resize_compatibility_audit(
            **kwargs
        )

    def snn_language_dense_readout_resize_plan(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_dense_readout_resize_plan(
            **kwargs
        )

    def snn_language_dense_readout_resize_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_dense_readout_resize_preflight(
            **kwargs
        )

    def snn_language_dense_readout_resize_transaction_proposal(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_dense_readout_resize_transaction_proposal(
            **kwargs
        )

    def snn_language_dense_readout_resize_executor_readiness_audit(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_dense_readout_resize_executor_readiness_audit(
            **kwargs
        )

    def snn_language_dense_readout_layout_migration(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_plasticity_executor.apply_dense_readout_layout_migration(
            **kwargs
        )

    def snn_language_dense_readout_tensor_materialization_readiness(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_dense_readout_tensor_materialization_readiness(
            **kwargs
        )

    def snn_language_dense_readout_tensor_materialization(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_plasticity_executor.apply_dense_readout_tensor_materialization(
            **kwargs
        )

    def snn_language_dense_readout_training_readiness(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_dense_readout_training_readiness(
            **kwargs
        )

    def snn_language_dense_readout_training_loop_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_dense_readout_training_loop_design(
            **kwargs
        )

    def snn_language_dense_readout_training_loop_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_dense_readout_training_loop_preflight(
            **kwargs
        )

    def snn_language_dense_readout_training(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_plasticity_executor.apply_dense_readout_training_loop(
            **kwargs
        )

    def snn_language_dense_readout_post_training_evaluation(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_dense_readout_post_training_evaluation(
            **kwargs
        )

    def snn_language_dense_readout_decoder_probe_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_dense_readout_decoder_probe_design(
            **kwargs
        )

    def snn_language_dense_readout_decoder_probe_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_dense_readout_decoder_probe_preflight(
            **kwargs
        )

    def snn_language_dense_readout_decoder_probe_execution(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_dense_readout_decoder_probe_execution(
            **kwargs
        )

    def snn_language_dense_readout_label_candidate_review(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.snn_language_dense_readout_label_candidate_review(
            **kwargs
        )

    def snn_language_plasticity_runtime_state(self) -> dict[str, Any]:
        return self._root._snn_language_plasticity_executor.snapshot()

    def snn_language_transition_memory_homeostatic_maintenance(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_plasticity_executor.maintain_transition_memory(**kwargs)

    def snn_language_transition_memory_sleep_policy(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("transition_memory_state") is None:
            kwargs["transition_memory_state"] = self.snn_language_plasticity_runtime_state()
        return self._root._status_read_model.snn_language_transition_memory_sleep_policy(**kwargs)

    def snn_sleep_plasticity_review_ticket(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._replay_controller.record_snn_sleep_plasticity_review_ticket(**kwargs)

    def snn_sleep_plasticity_review_ticket_queue(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._replay_controller.snn_sleep_plasticity_review_ticket_queue(**kwargs)

    def snn_sleep_plasticity_autonomy_proposal(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._replay_controller.snn_sleep_plasticity_autonomy_proposal(**kwargs)

    def snn_sleep_plasticity_scheduler_experiment(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._replay_controller.snn_sleep_plasticity_scheduler_experiment(**kwargs)

    def snn_sleep_plasticity_scheduler_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._replay_controller.snn_sleep_plasticity_scheduler_design(**kwargs)

    def snn_sleep_plasticity_scheduler_design_review_ticket(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._replay_controller.record_snn_sleep_plasticity_scheduler_design_review_ticket(
            **kwargs
        )

    def snn_sleep_plasticity_scheduler_design_review_ticket_queue(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._replay_controller.snn_sleep_plasticity_scheduler_design_review_ticket_queue(
            **kwargs
        )

    def snn_sleep_plasticity_scheduler_installation_autonomy_proposal(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._replay_controller.snn_sleep_plasticity_scheduler_installation_autonomy_proposal(
            **kwargs
        )

    def snn_sleep_plasticity_scheduler_installation_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._replay_controller.snn_sleep_plasticity_scheduler_installation_preflight(
            **kwargs
        )

    def snn_sleep_plasticity_review_scheduler_installation(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._replay_controller.install_snn_sleep_plasticity_review_scheduler(
            **kwargs
        )

    def snn_sleep_plasticity_review_scheduler_runtime(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._replay_controller.snn_sleep_plasticity_review_scheduler_runtime(
            **kwargs
        )

    def snn_sleep_plasticity_review_scheduler_cycle_inspection(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._replay_controller.snn_sleep_plasticity_review_scheduler_cycle_inspection(
            **kwargs
        )

    def snn_sleep_plasticity_review_scheduler_cycle_acknowledgment(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._replay_controller.acknowledge_snn_sleep_plasticity_review_scheduler_cycle(
            **kwargs
        )

    def snn_sleep_plasticity_review_scheduler_cycle_acknowledgment_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._replay_controller.snn_sleep_plasticity_review_scheduler_cycle_acknowledgment_preflight(
            **kwargs
        )

    def snn_sleep_plasticity_review_scheduler_cycle_autonomy_proposal(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._replay_controller.snn_sleep_plasticity_review_scheduler_cycle_autonomy_proposal(
            **kwargs
        )

    def snn_due_cycle_bounded_replay_selection_proposal(self, **kwargs: Any) -> dict[str, Any]:
        limit = kwargs.pop("limit", 8)
        queue = self.snn_replay_consolidation_priority_queue(limit=limit)
        return self._root._replay_controller.snn_due_cycle_bounded_replay_selection_proposal(
            consolidation_priority_queue=queue,
            max_candidates=kwargs.pop("max_candidates", 1),
            **kwargs,
        )

    def snn_due_cycle_replay_artifact_recording_review_proposal(self, **kwargs: Any) -> dict[str, Any]:
        limit = kwargs.pop("limit", 8)
        max_candidates = kwargs.pop("max_candidates", 1)
        policy = kwargs.pop("policy", None)
        selection = self.snn_due_cycle_bounded_replay_selection_proposal(
            limit=limit,
            max_candidates=max_candidates,
            **kwargs,
        )
        artifact_recording_policy = self.snn_replay_artifact_recording_policy_proposal(
            limit=limit,
            policy=policy,
        )
        return self._root._replay_controller.snn_due_cycle_replay_artifact_recording_review_proposal(
            due_cycle_selection_proposal=selection,
            artifact_recording_policy_proposal=artifact_recording_policy,
        )

    def snn_sleep_phase_separation_proposal(self, **kwargs: Any) -> dict[str, Any]:
        limit = kwargs.pop("limit", 8)
        max_candidates = kwargs.pop("max_candidates", 1)
        selection = self.snn_due_cycle_bounded_replay_selection_proposal(
            limit=limit,
            max_candidates=max_candidates,
        )
        return self._root._replay_controller.snn_sleep_phase_separation_proposal(
            due_cycle_selection_proposal=selection,
            cycle_acknowledgment_preflight=kwargs.pop(
                "cycle_acknowledgment_preflight",
                None,
            ),
        )

    def snn_rem_like_homeostatic_stabilization_preflight(self, **kwargs: Any) -> dict[str, Any]:
        phase = self.snn_sleep_phase_separation_proposal(
            limit=kwargs.pop("limit", 8),
            max_candidates=kwargs.pop("max_candidates", 1),
        )
        return self._root._replay_controller.snn_rem_like_homeostatic_stabilization_preflight(
            sleep_phase_separation_proposal=phase,
            transition_memory_state=self.snn_language_plasticity_runtime_state(),
            maintenance_policy=kwargs.pop("maintenance_policy", None),
        )

    def snn_language_transition_memory_regeneration_proposal(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("transition_memory_state") is None:
            kwargs["transition_memory_state"] = self.snn_language_plasticity_runtime_state()
        return self._root._status_read_model.snn_language_transition_memory_regeneration_proposal(**kwargs)

    def snn_language_transition_memory_regeneration_permit(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._replay_controller.issue_regeneration_permit(**kwargs)

    def snn_replay_evaluation_context(self, **kwargs: Any) -> dict[str, Any]:
        observed_slot_source_window_surface = (
            "bounded_snn_replay_evaluation_context_observed_slot_window.v1"
        )
        observed_slots, observed_slot_source_window = (
            self._root._snn_language_readout_ledger._bounded_replay_payload_window(
                kwargs.pop("observed_readout_slots"),
                source="runtime_facade.snn_replay_evaluation_context.observed_readout_slots",
                surface=observed_slot_source_window_surface,
                active_replay_computation_device="cpu",
            )
        )
        required = {
            "observed_slot_source_window_bounded": (
                self._readout_replay_payload_window_bounded(
                    observed_slot_source_window,
                    surface=observed_slot_source_window_surface,
                )
            ),
            "observed_slot_payload_not_truncated": not bool(
                observed_slot_source_window.get("source_payload_truncated")
            ),
            "observed_slot_payload_well_formed": int(
                observed_slot_source_window.get("source_mapping_count", 0) or 0
            )
            == int(observed_slot_source_window.get("source_window_count", 0) or 0),
            "observed_readout_slots_available": bool(observed_slots),
        }
        if not all(required.values()):
            return {
                "artifact_kind": "terminus_snn_replay_evaluation_context",
                "surface": "snn_replay_evaluation_context.v1",
                "available": False,
                "ready": False,
                "accepted": False,
                "owned_by_marulho": True,
                "external_dependency": False,
                "records_replay_context": False,
                "records_ledger_event": False,
                "runs_replay": False,
                "writes_checkpoint": False,
                "generates_text": False,
                "decodes_text": False,
                "trains_runtime_model": False,
                "applies_plasticity": False,
                "mutates_runtime_state": False,
                "eligible_for_replay_memory": False,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "observed_slot_source_window": dict(observed_slot_source_window),
                "promotion_gate": {
                    "status": "blocked_observed_slot_source_window",
                    "eligible_for_replay_context_recording": False,
                    "eligible_for_replay_memory": False,
                    "eligible_for_live_replay": False,
                    "eligible_for_plasticity_application": False,
                    "eligible_for_fact_promotion": False,
                    "eligible_for_action": False,
                    "required_evidence": required,
                },
            }
        mismatch = self._root._status_read_model.snn_language_sequence_mismatch_probe(
            prediction_report=kwargs.pop("prediction_report"),
            observed_readout_slots=observed_slots,
            device_evidence=kwargs.pop("device_evidence", None),
        )
        pressure = self._root._status_read_model.snn_language_plasticity_pressure(
            mismatch_report=mismatch,
            runtime_truth_delta=kwargs.pop("runtime_truth_delta", None),
            rollback_policy=kwargs.pop("rollback_policy", None),
        )
        context = self._root._replay_controller.record_snn_replay_evaluation_context(
            mismatch_report=mismatch,
            pressure_report=pressure,
            source_metadata={
                "source": "runtime_facade.snn_replay_evaluation_context",
                "observed_slot_source_window": dict(observed_slot_source_window),
            },
        )
        context = dict(context)
        context["accepted"] = True
        context["records_replay_context"] = True
        context["observed_slot_source_window"] = dict(observed_slot_source_window)
        context["promotion_gate"] = {
            "status": "replay_evaluation_context_recorded",
            "eligible_for_replay_context_recording": False,
            "eligible_for_replay_memory": False,
            "eligible_for_live_replay": False,
            "eligible_for_plasticity_application": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_action": False,
            "required_evidence": required,
        }
        return context

    def snn_replay_consolidation_priority_queue(self, **kwargs: Any) -> dict[str, Any]:
        limit = kwargs.pop("limit", 8)
        readout_priority = self._root._snn_language_readout_ledger.replay_priority(limit=limit)
        return self._root._replay_controller.snn_replay_consolidation_priority_queue(
            readout_replay_priority_report=readout_priority,
            limit=limit,
        )

    def snn_replay_artifact_recording_policy_proposal(self, **kwargs: Any) -> dict[str, Any]:
        limit = kwargs.pop("limit", 8)
        queue = self.snn_replay_consolidation_priority_queue(limit=limit)
        return self._root._replay_controller.snn_replay_artifact_recording_policy_proposal(
            consolidation_priority_queue=queue,
            policy=kwargs.pop("policy", None),
        )

    def snn_replay_artifact_recording_review_ticket(self, **kwargs: Any) -> dict[str, Any]:
        proposal = self.snn_replay_artifact_recording_policy_proposal(
            limit=kwargs.pop("limit", 8),
            policy=kwargs.pop("policy", None),
        )
        return self._root._replay_controller.record_snn_replay_artifact_recording_review_ticket(
            policy_proposal=proposal,
            **kwargs,
        )

    def snn_due_cycle_replay_artifact_recording_review_ticket(self, **kwargs: Any) -> dict[str, Any]:
        limit = kwargs.pop("limit", 8)
        max_candidates = kwargs.pop("max_candidates", 1)
        policy = kwargs.pop("policy", None)
        policy_proposal = self.snn_replay_artifact_recording_policy_proposal(
            limit=limit,
            policy=policy,
        )
        due_cycle_review_proposal = (
            self.snn_due_cycle_replay_artifact_recording_review_proposal(
                limit=limit,
                max_candidates=max_candidates,
                policy=policy,
            )
        )
        return self._root._replay_controller.record_snn_replay_artifact_recording_review_ticket(
            policy_proposal=policy_proposal,
            due_cycle_review_proposal=due_cycle_review_proposal,
            **kwargs,
        )

    def snn_transition_memory_replay_artifact_proposal(self, **kwargs: Any) -> dict[str, Any]:
        context = self._root._replay_controller.verified_snn_replay_evaluation_context(
            kwargs.pop("replay_evaluation_context_id")
        )
        if context is None:
            raise ValueError("SNN replay artifact proposal requires a verified server-held evaluation context.")
        proposal = self._root._snn_language_readout_ledger.transition_memory_replay_artifact_proposal(
            mismatch_report=context["mismatch_report"],
            pressure_report=context["pressure_report"],
            **kwargs,
        )
        proposal["replay_evaluation_context_id"] = context["replay_evaluation_context_id"]
        proposal["replay_evaluation_context_hash"] = context["evidence_hash"]
        proposal["source_metadata_hash"] = context.get("source_metadata_hash")
        proposal["emission_lineage"] = (
            self._root._replay_controller._snn_replay_context_emission_lineage(
                context.get("source_metadata")
                if isinstance(context.get("source_metadata"), Mapping)
                else {}
            )
        )
        return proposal

    def snn_transition_memory_evaluated_replay_artifact(self, **kwargs: Any) -> dict[str, Any]:
        replay_evaluation_context_id = kwargs.pop("replay_evaluation_context_id")
        proposal = self.snn_transition_memory_replay_artifact_proposal(
            replay_evaluation_context_id=replay_evaluation_context_id,
            limit=kwargs.pop("limit", 8),
        )
        kwargs["artifact_proposal"] = proposal
        kwargs["replay_evaluation_context_id"] = replay_evaluation_context_id
        (
            known_readout_evidence_hashes,
            known_readout_evidence_source_window,
        ) = self._root._snn_language_readout_ledger.known_readout_evidence_hashes_with_report()
        kwargs["known_readout_evidence_hashes"] = known_readout_evidence_hashes
        kwargs["known_readout_evidence_source_window"] = (
            known_readout_evidence_source_window
        )
        return self._root._replay_controller.record_evaluated_snn_transition_memory_replay_artifact(
            **kwargs
        )

    def snn_language_transition_memory_regeneration(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._snn_language_plasticity_executor.regenerate_transition_memory(**kwargs)

    def subcortical_self_repair_surface(self) -> dict[str, Any]:
        return self._root._status_read_model.subcortical_self_repair_surface()

    def subcortical_self_repair_evaluation_surface(self) -> dict[str, Any]:
        return self._root._status_read_model.subcortical_self_repair_evaluation_surface()

    def subcortical_structural_plasticity_surface(self) -> dict[str, Any]:
        return self._root._status_read_model.subcortical_structural_plasticity_surface()

    def binding_growth_trial_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.binding_growth_trial_design(**kwargs)

    def subcortical_structural_plasticity_isolated_evaluation(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.subcortical_structural_plasticity_isolated_evaluation(**kwargs)

    def subcortical_structural_mutation_design(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.subcortical_structural_mutation_design(**kwargs)

    def subcortical_structural_mutation_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._status_read_model.subcortical_structural_mutation_preflight(**kwargs)

    def subcortical_structural_mutation_application(self, **kwargs: Any) -> dict[str, Any]:
        return self._root._structural_mutation_executor.apply_subcortical_structural_mutation(**kwargs)

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

    @classmethod
    def _snn_language_capacity_state(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        raw = (
            state.get("language_capacity")
            if isinstance(state.get("language_capacity"), Mapping)
            else {}
        )
        present = bool(raw)
        return {
            "surface": _SNN_LANGUAGE_CAPACITY_SURFACE,
            "raw_surface": str(raw.get("surface") or "") if present else None,
            "present": present,
            "owned_by_marulho": True,
            "external_dependency": False,
            "language_neuron_count": cls._positive_capacity_int(
                raw.get("language_neuron_count"),
                default=_SNN_LANGUAGE_NEURON_COUNT,
                minimum=_SNN_LANGUAGE_NEURON_COUNT,
            ),
            "sparse_edge_budget": cls._positive_capacity_int(
                raw.get("sparse_edge_budget"),
                default=_SNN_LANGUAGE_SPARSE_EDGE_BUDGET,
                minimum=_SNN_LANGUAGE_SPARSE_EDGE_BUDGET,
            ),
            "outgoing_fanout_budget": cls._positive_capacity_int(
                raw.get("outgoing_fanout_budget"),
                default=_SNN_LANGUAGE_OUTGOING_FANOUT_BUDGET,
                minimum=_SNN_LANGUAGE_OUTGOING_FANOUT_BUDGET,
            ),
            "dynamic_capacity_enabled": bool(
                raw.get("dynamic_capacity_enabled")
            ),
            "capacity_expansion_count": cls._positive_capacity_int(
                raw.get("capacity_expansion_count"),
                default=0,
                minimum=0,
            ),
            "resizes_network": bool(raw.get("resizes_network")),
            "adds_neurons": bool(raw.get("adds_neurons")),
            "adds_layers": bool(raw.get("adds_layers")),
            "writes_checkpoint": bool(raw.get("writes_checkpoint")),
            "last_capacity_mutation": deepcopy(
                raw.get("last_capacity_mutation")
            ),
        }

    @staticmethod
    def _positive_capacity_int(
        value: Any,
        *,
        default: int,
        minimum: int,
    ) -> int:
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            normalized = int(default)
        return max(int(minimum), normalized)
