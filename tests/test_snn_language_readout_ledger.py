from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from threading import RLock

from marulho.service.runtime_state import RuntimeState
from marulho.service.runtime_facade import RuntimeFacade
from marulho.service.snn_language_plasticity_executor import (
    SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT,
    SNNLanguagePlasticityApplicationExecutor,
)
from marulho.service.snn_language_readout_ledger import (
    SNN_AUTONOMOUS_CONFIDENCE_USE_SOURCE_WINDOW_POLICY,
    SNN_EMISSION_REVIEW_REPLAY_POLICY_SOURCE_WINDOW_LIMIT,
    SNN_EMISSION_REVIEW_HISTORY_SOURCE_WINDOW_POLICY,
    SNN_DENSE_LABEL_CALIBRATION_EVALUATION_SOURCE_WINDOW_POLICY,
    SNN_DENSE_LABEL_CANDIDATE_CALIBRATION_SOURCE_WINDOW_POLICY,
    SNN_DENSE_LABEL_CALIBRATION_UPDATE_SOURCE_WINDOW_POLICY,
    SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS,
    SNN_LANGUAGE_READOUT_LEDGER_SNAPSHOT_SOURCE_WINDOW_POLICY,
    SNN_READOUT_LEDGER_RECORD_FAMILY_SOURCE_WINDOW_POLICY,
    SNN_READOUT_EVIDENCE_HASH_SOURCE_WINDOW_POLICY,
    SNN_READOUT_REPLAY_PRIORITY_SOURCE_WINDOW_LIMIT,
    SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT,
    SNN_READOUT_SYNAPSE_PROVENANCE_AUDIT_SOURCE_WINDOW_LIMIT,
    SNN_READOUT_SYNAPSE_PROVENANCE_AUDIT_SOURCE_WINDOW_POLICY,
    SNN_ROLLOUT_REHEARSAL_SOURCE_WINDOW_LIMIT,
    SNNLanguageReadoutEvidenceLedger,
    normalize_snn_language_readout_ledger_state,
)


def test_readout_ledger_does_not_expose_all_family_normalizer() -> None:
    assert not hasattr(SNNLanguageReadoutEvidenceLedger, "_normalized_state")


def test_readout_ledger_state_migrates_legacy_surface_fields_once() -> None:
    legacy_event = {"snn_language_readout_surface_event_hash": "1" * 64}
    migrated = normalize_snn_language_readout_ledger_state(
        {
            "autonomous_snn_language_thought_surface_events": [legacy_event],
            "total_autonomous_snn_language_thought_surface_count": 1,
            "last_autonomous_snn_language_thought_surface_recorded_at": (
                "2026-06-22T00:00:00+00:00"
            ),
        }
    )

    assert migrated["snn_language_readout_surface_events"] == [legacy_event]
    assert migrated["total_snn_language_readout_surface_count"] == 1
    assert (
        migrated["last_snn_language_readout_surface_recorded_at"]
        == "2026-06-22T00:00:00+00:00"
    )
    assert "autonomous_snn_language_thought_surface_events" not in migrated
    assert "total_autonomous_snn_language_thought_surface_count" not in migrated
    assert (
        "last_autonomous_snn_language_thought_surface_recorded_at"
        not in migrated
    )


def test_readout_ledger_state_migrates_legacy_memory_fields_once() -> None:
    legacy_event = {"snn_language_readout_memory_event_hash": "2" * 64}
    migrated = normalize_snn_language_readout_ledger_state(
        {
            "autonomous_snn_language_thought_memory_events": [legacy_event],
            "total_autonomous_snn_language_thought_memory_count": 1,
            "last_autonomous_snn_language_thought_memory_recorded_at": (
                "2026-06-22T00:00:00+00:00"
            ),
        }
    )

    assert migrated["snn_language_readout_memory_events"] == [legacy_event]
    assert migrated["total_snn_language_readout_memory_count"] == 1
    assert (
        migrated["last_snn_language_readout_memory_recorded_at"]
        == "2026-06-22T00:00:00+00:00"
    )
    assert "autonomous_snn_language_thought_memory_events" not in migrated
    assert "total_autonomous_snn_language_thought_memory_count" not in migrated
    assert (
        "last_autonomous_snn_language_thought_memory_recorded_at"
        not in migrated
    )


def _ready_draft() -> dict[str, object]:
    return _ready_draft_for("prediction-hash-1", "evaluation-hash-1", "weights-hash-1", ["memory pressure"])


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _assert_dense_label_calibration_update_source_window(
    window: dict[str, object],
    *,
    expected_count: int | None = None,
) -> None:
    assert window["surface"] == (
        "bounded_snn_dense_label_calibration_update_source_window.v1"
    )
    assert window["policy"] == SNN_DENSE_LABEL_CALIBRATION_UPDATE_SOURCE_WINDOW_POLICY
    assert window["selection_criteria"] == [
        "applied_dense_label_calibration_updates_only",
        "bounded_source_window_before_update_application_or_review",
    ]
    if expected_count is not None:
        assert window["source_window_count"] == expected_count
    assert window["global_candidate_scan"] is False
    assert window["global_score_scan"] is False
    assert window["raw_text_payload_loaded"] is False
    assert window["language_reasoning"] is False
    assert window["runs_live_tick"] is False
    assert window["runs_every_token"] is False
    assert window["mutates_runtime_state"] is False
    assert window["applies_plasticity"] is False
    assert window["archival_storage_device"] == "cpu"
    assert window["lookup_device"] == "cpu"
    assert window["write_device"] == "cpu"
    assert window["gpu_used"] is False


def _assert_autonomous_confidence_use_source_window(
    window: dict[str, object],
    *,
    expected_count: int | None = None,
) -> None:
    assert window["surface"] == "bounded_snn_autonomous_confidence_use_source_window.v1"
    assert window["policy"] == SNN_AUTONOMOUS_CONFIDENCE_USE_SOURCE_WINDOW_POLICY
    assert window["selection_criteria"] == [
        "recorded_autonomous_confidence_use_events_only",
        "bounded_source_window_before_hash_only_use_review",
    ]
    if expected_count is not None:
        assert window["source_window_count"] == expected_count
    assert window["global_candidate_scan"] is False
    assert window["global_score_scan"] is False
    assert window["raw_text_payload_loaded"] is False
    assert window["language_reasoning"] is False
    assert window["runs_live_tick"] is False
    assert window["runs_every_token"] is False
    assert window["mutates_runtime_state"] is False
    assert window["applies_plasticity"] is False
    assert window["archival_storage_device"] == "cpu"
    assert window["lookup_device"] == "cpu"
    assert window["write_device"] == "cpu"
    assert window["gpu_used"] is False


def _assert_record_family_source_window(
    window: dict[str, object],
    *,
    field: str,
    expected_count: int | None = None,
) -> None:
    assert window["surface"] == "bounded_snn_readout_ledger_record_family_source_window.v1"
    assert window["policy"] == SNN_READOUT_LEDGER_RECORD_FAMILY_SOURCE_WINDOW_POLICY
    assert window["event_family"] == field
    assert window["source"] == f"snn_readout_ledger.{field}"
    assert window["selection_criteria"] == [
        "single_record_family_only",
        "bounded_source_window_before_duplicate_check",
    ]
    if expected_count is not None:
        assert window["source_window_count"] == expected_count
    assert window["global_candidate_scan"] is False
    assert window["global_score_scan"] is False
    assert window["raw_text_payload_loaded"] is False
    assert window["language_reasoning"] is False
    assert window["runs_live_tick"] is False
    assert window["runs_every_token"] is False
    assert window["applies_plasticity"] is False
    assert window["archival_storage_device"] == "cpu"
    assert window["lookup_device"] == "cpu"
    assert window["write_device"] == "cpu"
    assert window["gpu_used"] is False


def _terminal_newborn_learning_review() -> dict[str, object]:
    cycle = {
        "synapse": "4:64",
        "source_neuron_index": 4,
        "target_neuron_index": 64,
        "applied_weight": 0.01,
        "critical_period_age_cycles": 64,
        "critical_period_cycles": 64,
        "critical_period_cycles_remaining": 0,
        "active_cycle_count": 0,
        "inactive_cycle_count": 64,
        "minimum_survival_active_cycles": 16,
        "critical_period_learning_application_hash": "l" * 64,
        "newborn_integration_synapse_hash": "i" * 64,
        "current_maturation_state": "prune_eligible",
        "maturation_decided": True,
        "pruning_applied": False,
    }
    return {
        "surface": (
            "snn_language_autonomous_snn_language_thought_newborn_neuron_"
            "critical_period_learning_event_review.v1"
        ),
        "accepted": True,
        "ready": True,
        "review_hash": "r" * 64,
        "requires_operator_approval": False,
        "autonomous_snn_language_thought_newborn_neuron_"
        "critical_period_learning_event_review": {
            "newborn_neuron_critical_period_learning_event_hash": "e" * 64,
            "after_state_revision": 0,
            "actual_device": "cpu",
            "tensor_is_cuda": False,
            "verified_applied_learning_cycles": [cycle],
            "mature_synapse_count": 0,
            "prune_eligible_synapse_count": 1,
            "sparse_dense_developmental_provenance_consistent": True,
            "operator_approval_required": False,
        },
        "promotion_gate": {
            "eligible_for_autonomous_snn_language_thought_newborn_neuron_"
            "maturation_outcome_review": True
        },
    }


def test_terminal_newborn_outcome_builds_verified_synapse_pruning_preflight() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: {},
    )
    outcome = ledger.autonomous_snn_language_thought_newborn_neuron_maturation_outcome_review(
        autonomous_snn_language_thought_newborn_neuron_critical_period_learning_event_review=(
            _terminal_newborn_learning_review()
        )
    )
    design = ledger.autonomous_snn_language_thought_newborn_synapse_pruning_design(
        autonomous_snn_language_thought_newborn_neuron_maturation_outcome_review=outcome
    )
    candidate = design[
        "autonomous_snn_language_thought_newborn_synapse_pruning_design"
    ]["prune_candidates"][0]
    application_hash = candidate[
        "critical_period_learning_application_hash"
    ]
    runtime = {
        "surface": "snn_language_plasticity_runtime_state.v1",
        "sparse_transition_weights": {"4:64": 0.01},
        "synapse_provenance_by_key": {
            "4:64": {
                "provenance_type": "newborn_neuron_integration",
                "current_maturation_state": "prune_eligible",
            }
        },
        "newborn_neuron_critical_period_state_by_synapse": {
            "4:64": {
                "critical_period_cycles_remaining": 0,
                "current_maturation_state": "prune_eligible",
                "maturation_decided": True,
                "critical_period_learning_application_hash": (
                    application_hash
                ),
            }
        },
        "critical_period_learning_dense_samples": [
            {
                "synapse": "4:64",
                "source_neuron_index": 4,
                "target_neuron_index": 64,
                "weight": 0.01,
            }
        ],
    }
    preflight = ledger.autonomous_snn_language_thought_newborn_synapse_pruning_preflight(
        autonomous_snn_language_thought_newborn_synapse_pruning_design=design,
        expected_state_revision=0,
        plasticity_runtime_state=runtime,
        checkpoint_transaction={
            "pre_pruning_checkpoint_saved": True,
            "pre_pruning_checkpoint_restore_verified": True,
            "checkpoint_path": "memory://prune",
        },
        executor_capabilities={
            "autonomous_snn_language_thought_newborn_"
            "synapse_pruning_executor": True
        },
    )

    assert outcome["accepted"] is True
    assert outcome[
        "autonomous_snn_language_thought_newborn_neuron_"
        "maturation_outcome_review"
    ]["retained_mature_synapse_count"] == 0
    assert design["accepted"] is True
    assert preflight["accepted"] is True
    assert preflight["mutates_runtime_state"] is False
    assert runtime_state.state_revision == 0


def _language_capacity(
    *,
    language_neuron_count: int = 64,
    sparse_edge_budget: int = 256,
    outgoing_fanout_budget: int = 16,
    capacity_expansion_count: int = 0,
) -> dict[str, object]:
    return {
        "surface": "snn_language_capacity_state.v1",
        "language_neuron_count": language_neuron_count,
        "sparse_edge_budget": sparse_edge_budget,
        "outgoing_fanout_budget": outgoing_fanout_budget,
        "capacity_expansion_count": capacity_expansion_count,
    }


def _ready_rollout_replay_evaluation() -> dict[str, object]:
    return {
        "surface": "snn_language_readout_rollout_replay_evaluation.v1",
        "owned_by_marulho": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "freeform_language_generation": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "recorded_in_ledger": False,
        "eligible_for_replay_priority": False,
        "replay_evaluation": {
            "trace_step_count": 2,
            "replay_targets": [
                {
                    "step_index": 0,
                    "selected_label": "memory pressure",
                    "grounded": True,
                    "selection_score": 0.4,
                    "transition_support": 0.6,
                    "predicted_sparse_indices": [1, 2, 3],
                    "active_indices_hash": _sha256_json([1, 2, 3]),
                    "active_indices_hash_valid": True,
                },
                {
                    "step_index": 1,
                    "selected_label": "prediction error",
                    "grounded": True,
                    "selection_score": 0.3,
                    "transition_support": 0.5,
                    "predicted_sparse_indices": [2, 3, 4],
                    "active_indices_hash": _sha256_json([2, 3, 4]),
                    "active_indices_hash_valid": True,
                }
            ],
        },
        "provenance_evidence": {
            "rollout_replay_evaluation_hash": "rollout-eval-hash-1",
            "rollout_hash": "rollout-hash-1",
            "rollout_id": "snn-readout-rollout:rollout-hash-1",
            "prediction_hash": "prediction-hash-1",
            "current_sparse_code_hash": "current-sparse-code-hash-1",
            "transition_memory_evaluation_hash": "evaluation-hash-1",
            "persistent_transition_weights_hash": "weights-hash-1",
            "server_transition_memory_hash": "weights-hash-1",
            "server_transition_memory_hash_match": True,
            "transition_memory_state_source": (
                "service.runtime_facade.snn_language_plasticity_runtime_state"
            ),
        },
        "device_evidence": {
            "requested_device": "cpu",
            "tensor_device": "cpu",
            "cuda_tensor": False,
            "device_source": "test",
        },
        "promotion_gate": {
            "eligible_for_readout_rollout_ledger_recording_review": True,
            "eligible_for_replay_priority": False,
        },
    }


def _ready_rollout_replay_evaluation_for(
    index: int,
    *,
    label: str,
    weights_hash: str,
) -> dict[str, object]:
    report = deepcopy(_ready_rollout_replay_evaluation())
    provenance = report["provenance_evidence"]
    assert isinstance(provenance, dict)
    provenance["rollout_replay_evaluation_hash"] = f"rollout-eval-hash-{index}"
    provenance["rollout_hash"] = f"rollout-hash-{index}"
    provenance["rollout_id"] = f"snn-readout-rollout:rollout-hash-{index}"
    provenance["prediction_hash"] = f"prediction-hash-{index}"
    provenance["current_sparse_code_hash"] = f"current-sparse-code-hash-{index}"
    provenance["transition_memory_evaluation_hash"] = f"evaluation-hash-{index}"
    provenance["persistent_transition_weights_hash"] = weights_hash
    provenance["server_transition_memory_hash"] = weights_hash
    replay = report["replay_evaluation"]
    assert isinstance(replay, dict)
    targets = replay["replay_targets"]
    assert isinstance(targets, list)
    for target_index, target in enumerate(targets):
        assert isinstance(target, dict)
        sparse_indices = [
            (index + target_index) % 64,
            (index + target_index + 1) % 64,
        ]
        target["selected_label"] = label
        target["predicted_sparse_indices"] = sparse_indices
        target["active_indices_hash"] = _sha256_json(sparse_indices)
    return report


def _ready_emission() -> dict[str, object]:
    emission_hash = _sha256_json({"emission": "memory pressure"})
    trajectory_hash = _sha256_json({"trajectory": "memory pressure"})
    prediction_hash = _sha256_json({"prediction": "memory pressure"})
    evaluation_hash = _sha256_json({"evaluation": "memory pressure"})
    weights_hash = _sha256_json({"weights": "memory pressure"})
    return {
        "surface": "snn_language_readout_emission.v1",
        "ready": True,
        "owned_by_marulho": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": True,
        "decodes_text": True,
        "generation_scope": "operator_visible_bounded_snn_readout_emission",
        "freeform_language_generation": False,
        "mutates_runtime_state": False,
        "applies_plasticity": False,
        "writes_checkpoint": False,
        "promotes_fact": False,
        "promotes_action": False,
        "cognition_substrate": False,
        "language_output": {
            "text": "memory pressure",
            "labels": ["memory pressure"],
            "term_count": 1,
            "max_terms": 12,
        },
        "emission_hash": emission_hash,
        "emission_binding": {
            "trajectory_hash": trajectory_hash,
            "prediction_hash": prediction_hash,
            "transition_memory_evaluation_hash": evaluation_hash,
            "persistent_transition_weights_hash": weights_hash,
        },
        "promotion_gate": {
            "eligible_for_operator_display": True,
            "eligible_for_freeform_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_action": False,
        },
    }


def _ready_dense_label_candidate_review() -> dict[str, object]:
    review_hash = _sha256_json({"review": "dense-label-candidates"})
    execution_hash = _sha256_json({"execution": "dense-decoder-probe"})
    return {
        "surface": "snn_language_dense_readout_label_candidate_review.v1",
        "ready": True,
        "review_recorded": True,
        "owned_by_marulho": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "freeform_language_generation": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "returns_trained_weights": False,
        "applies_plasticity": False,
        "records_replay_artifact": False,
        "promotes_facts": False,
        "executes_actions": False,
        "mutates_runtime_state": False,
        "writes_checkpoint": False,
        "review_hash": review_hash,
        "source_execution_hash": execution_hash,
        "grounded_label_candidates": ["prediction error", "concept focus"],
        "candidate_label_count": 2,
        "review_context": {
            "operator_id": "operator-dense-label",
            "confirmation": True,
            "tensor_device": "cpu",
            "active_count": 3,
        },
        "promotion_gate": {
            "eligible_for_bounded_label_candidate_evidence_record": True,
            "eligible_for_language_generation": False,
            "eligible_for_freeform_language_generation": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_action": False,
            "eligible_for_plasticity_application": False,
            "eligible_for_replay_artifact": False,
            "eligible_for_checkpoint_write": False,
        },
    }


def _ready_draft_for(
    prediction_hash: str,
    evaluation_hash: str,
    weights_hash: str,
    labels: list[str],
) -> dict[str, object]:
    return {
        "surface": "snn_language_readout_draft.v1",
        "generation_scope": "bounded_grounded_readout_label_draft",
        "freeform_language_generation": False,
        "mutates_runtime_state": False,
        "draft": {"labels": labels, "text": " ".join(labels)},
        "sparse_decode_evidence": {
            "candidate_matches": [
                {"label": label, "grounded": True}
                for label in labels
            ]
        },
        "transition_memory_evaluation_evidence": {
            "provenance_match": True,
            "prediction_hash": prediction_hash,
            "transition_memory_evaluation_hash": evaluation_hash,
            "persistent_transition_weights_hash": weights_hash,
        },
        "promotion_gate": {
            "eligible_for_bounded_readout_generation": True,
            "eligible_for_cognition_substrate": False,
        },
    }


def _first_rollout_synapse(design: dict[str, object]) -> str:
    rollout_design = design["rollout_consolidation_design"]
    assert isinstance(rollout_design, dict)
    candidates = rollout_design["candidate_synapses"]
    assert isinstance(candidates, list) and candidates
    first = candidates[0]
    assert isinstance(first, dict)
    return (
        f"{int(first.get('source_neuron_index', first['source_step_index']))}:"
        f"{int(first.get('target_neuron_index', first['target_step_index']))}"
    )


def _regeneration_candidate(index: int) -> dict[str, object]:
    pre_index = 65 + (index % 31)
    post_index = 66 + (index % 31)
    return {
        "pre_index": pre_index,
        "post_index": post_index,
        "synapse": f"{pre_index}:{post_index}",
        "initial_weight": 0.02,
        "locality_distance": 1,
    }


def _regeneration_candidates(count: int) -> list[dict[str, object]]:
    return [_regeneration_candidate(index) for index in range(count)]


def _rollout_sparse_transition_candidate(index: int) -> dict[str, object]:
    source_index = index % 32
    target_index = source_index + 1
    return {
        "source_index": source_index,
        "target_index": target_index,
        "source_trace_index": index,
        "source_step_index": index,
        "target_step_index": index + 1,
        "source_active_indices_hash": _sha256_json([source_index]),
        "target_active_indices_hash": _sha256_json([target_index]),
    }


def _rollout_sparse_transition_candidates(count: int) -> list[dict[str, object]]:
    return [_rollout_sparse_transition_candidate(index) for index in range(count)]


def _rollout_design_candidate(index: int) -> dict[str, object]:
    source_index = index % 32
    target_index = source_index + 1
    return {
        "synapse_id": f"snn-rollout-local:{source_index}:{target_index}:{index}",
        "source_step_index": source_index,
        "target_step_index": target_index,
        "source_neuron_index": source_index,
        "target_neuron_index": target_index,
        "source_trace_index": index,
        "source_rollout_step_index": index,
        "target_rollout_step_index": index + 1,
        "source_active_indices_hash": _sha256_json([source_index]),
        "target_active_indices_hash": _sha256_json([target_index]),
        "local_only": True,
        "proposed_weight_delta": 0.02,
        "homeostatic_decay": 0.0,
        "normalization": True,
        "applied_to_runtime": False,
    }


def _rollout_design_candidates(count: int) -> list[dict[str, object]]:
    return [_rollout_design_candidate(index) for index in range(count)]


def _rollout_growth_candidate(index: int) -> dict[str, object]:
    pre_index = index % 32
    post_index = pre_index + 1
    return {
        "synapse": f"{pre_index}:{post_index}",
        "pre_index": pre_index,
        "post_index": post_index,
        "initial_weight": 0.02,
        "locality_distance": abs(post_index - pre_index),
        "source_synapse_id": f"snn-rollout-local:{pre_index}:{post_index}:{index}",
        "source_trace_index": index,
        "source_rollout_step_index": index,
        "target_rollout_step_index": index + 1,
        "source_active_indices_hash": _sha256_json([pre_index]),
        "target_active_indices_hash": _sha256_json([post_index]),
        "local_only": True,
        "normalization": True,
        "applied_to_runtime": False,
    }


def _rollout_growth_candidates(count: int) -> list[dict[str, object]]:
    return [_rollout_growth_candidate(index) for index in range(count)]


def test_readout_ledger_records_ready_provenance_bound_draft_once() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )

    result = ledger.record_readout_draft(
        readout_draft=_ready_draft(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    duplicate = ledger.record_readout_draft(
        readout_draft=_ready_draft(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    snapshot = ledger.snapshot(limit=4)

    assert result["accepted"] is True
    assert result["mutates_runtime_state"] is True
    assert result["promotion_gate"]["eligible_for_replay_memory"] is True
    assert result["recorded_event"]["prediction_hash"] == "prediction-hash-1"
    assert duplicate["accepted"] is True
    assert duplicate["duplicate"] is True
    assert duplicate["mutates_runtime_state"] is False
    assert runtime_state.dirty_state is True
    assert runtime_state.state_revision == 0
    assert snapshot["summary"]["event_count"] == 1
    assert snapshot["events"][0]["labels"] == ["memory pressure"]
    assert snapshot["generates_text"] is False
    assert snapshot["mutates_runtime_state"] is False


def test_readout_ledger_blocks_unready_or_unconfirmed_draft() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    draft = _ready_draft()
    draft["transition_memory_evaluation_evidence"] = {
        "provenance_match": False,
        "prediction_hash": "prediction-hash-1",
    }

    result = ledger.record_readout_draft(
        readout_draft=draft,
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    unconfirmed = ledger.record_readout_draft(
        readout_draft=_ready_draft(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=False,
    )

    assert result["accepted"] is False
    assert result["promotion_gate"]["required_evidence"]["provenance_match"] is False
    assert unconfirmed["accepted"] is False
    assert unconfirmed["promotion_gate"]["required_evidence"]["confirmation"] is False
    assert ledger.snapshot()["summary"]["event_count"] == 0
    assert runtime_state.dirty_state is False


def test_readout_ledger_records_ready_emission_review_separately_from_replay_memory() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )

    result = ledger.record_readout_emission_review(
        readout_emission=_ready_emission(),
        expected_state_revision=0,
        operator_id="operator-emission",
        confirmation=True,
    )
    duplicate = ledger.record_readout_emission_review(
        readout_emission=_ready_emission(),
        expected_state_revision=0,
        operator_id="operator-emission",
        confirmation=True,
    )
    snapshot = ledger.snapshot(limit=4)

    assert result["accepted"] is True
    assert result["surface"] == "snn_language_readout_emission_review_record.v1"
    assert result["mutates_runtime_state"] is True
    assert result["promotion_gate"]["eligible_for_operator_display_history"] is True
    assert result["promotion_gate"]["eligible_for_replay_memory"] is False
    assert result["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert result["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert result["promotion_gate"]["eligible_for_action"] is False
    assert duplicate["accepted"] is True
    assert duplicate["duplicate"] is True
    assert duplicate["mutates_runtime_state"] is False
    assert snapshot["summary"]["emission_review_event_count"] == 1
    assert snapshot["summary"]["event_count"] == 0
    assert snapshot["summary"]["rollout_event_count"] == 0
    assert snapshot["emission_review_events"][0]["text"] == "memory pressure"
    assert snapshot["emission_review_events"][0]["eligible_for_replay_memory"] is False
    assert runtime_state.state_revision == 0
    assert runtime_state.dirty_state is True


def test_readout_ledger_records_dense_label_candidate_review_as_audit_only() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )

    result = ledger.record_dense_readout_label_candidate_review(
        dense_readout_label_candidate_review=_ready_dense_label_candidate_review(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-dense-label",
        confirmation=True,
    )
    duplicate = ledger.record_dense_readout_label_candidate_review(
        dense_readout_label_candidate_review=_ready_dense_label_candidate_review(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-dense-label",
        confirmation=True,
    )
    snapshot = ledger.snapshot(limit=4)

    assert result["accepted"] is True
    assert (
        result["surface"]
        == "snn_language_dense_readout_label_candidate_evidence_record.v1"
    )
    assert result["mutates_runtime_state"] is True
    assert result["records_replay_artifact"] is False
    assert result["promotes_facts"] is False
    assert result["executes_actions"] is False
    assert result["promotion_gate"]["eligible_for_dense_label_candidate_history"] is True
    assert result["promotion_gate"]["eligible_for_replay_memory"] is False
    assert result["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert result["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert result["promotion_gate"]["eligible_for_action"] is False
    assert duplicate["accepted"] is True
    assert duplicate["duplicate"] is True
    assert duplicate["mutates_runtime_state"] is False
    assert snapshot["summary"]["dense_label_candidate_event_count"] == 1
    assert snapshot["summary"]["event_count"] == 0
    assert snapshot["summary"]["rollout_event_count"] == 0
    assert snapshot["summary"]["emission_review_event_count"] == 0
    event = snapshot["dense_label_candidate_events"][0]
    assert event["labels"] == ["prediction error", "concept focus"]
    assert event["eligible_for_replay_memory"] is False
    assert event["eligible_for_plasticity_application"] is False
    assert event["eligible_for_fact_promotion"] is False
    assert event["eligible_for_action"] is False
    assert runtime_state.state_revision == 0
    assert runtime_state.dirty_state is True


def test_readout_ledger_blocks_unready_or_unconfirmed_dense_label_candidate_review() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    review = _ready_dense_label_candidate_review()
    review["ready"] = False

    result = ledger.record_dense_readout_label_candidate_review(
        dense_readout_label_candidate_review=review,
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-dense-label",
        confirmation=True,
    )
    unconfirmed = ledger.record_dense_readout_label_candidate_review(
        dense_readout_label_candidate_review=_ready_dense_label_candidate_review(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-dense-label",
        confirmation=False,
    )

    assert result["accepted"] is False
    assert result["promotion_gate"]["required_evidence"]["review_ready"] is False
    assert unconfirmed["accepted"] is False
    assert unconfirmed["promotion_gate"]["required_evidence"]["confirmation"] is False
    assert ledger.snapshot()["summary"]["dense_label_candidate_event_count"] == 0
    assert runtime_state.dirty_state is False


def test_readout_ledger_dense_label_candidate_history_is_read_only_audit_surface() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_dense_readout_label_candidate_review(
        dense_readout_label_candidate_review=_ready_dense_label_candidate_review(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-dense-label",
        confirmation=True,
    )
    runtime_state.mark_clean()
    before_revision = runtime_state.state_revision

    history = ledger.dense_label_candidate_history(limit=1)
    empty = ledger.dense_label_candidate_history(limit=0)

    assert runtime_state.state_revision == before_revision
    assert runtime_state.dirty_state is False
    assert history["surface"] == "snn_language_dense_label_candidate_history.v1"
    assert history["artifact_kind"] == "terminus_snn_language_dense_label_candidate_history"
    assert history["advisory"] is True
    assert history["executable"] is False
    assert history["records_ledger_event"] is False
    assert history["runs_replay"] is False
    assert history["writes_checkpoint"] is False
    assert history["generates_text"] is False
    assert history["decodes_text"] is False
    assert history["exposes_reviewed_bounded_labels"] is True
    assert history["freeform_language_generation"] is False
    assert history["applies_plasticity"] is False
    assert history["mutates_runtime_state"] is False
    assert history["summary"]["source_window"]["surface"] == (
        "bounded_snn_dense_label_candidate_calibration_source_window.v1"
    )
    assert history["summary"]["returned_dense_label_candidate_event_count"] == 1
    assert history["summary"]["dense_label_candidate_event_count"] == 1
    event = history["dense_label_candidate_events"][0]
    assert event["labels"] == ["prediction error", "concept focus"]
    assert event["label_count"] == 2
    assert event["tensor_device"] == "cpu"
    assert event["active_count"] == 3
    assert event["eligible_for_replay_memory"] is False
    assert event["eligible_for_live_replay"] is False
    assert event["eligible_for_plasticity_application"] is False
    assert event["eligible_for_fact_promotion"] is False
    assert event["eligible_for_action"] is False
    assert history["promotion_gate"][
        "eligible_for_operator_dense_label_candidate_history_inspection"
    ] is True
    assert history["promotion_gate"]["eligible_for_replay_memory"] is False
    assert history["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert history["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert history["promotion_gate"]["eligible_for_action"] is False
    assert empty["summary"]["returned_dense_label_candidate_event_count"] == 0
    assert empty["promotion_gate"][
        "eligible_for_operator_dense_label_candidate_history_inspection"
    ] is False
    assert "events" not in history
    assert "rollout_events" not in history
    assert "emission_review_events" not in history
    assert "replay_targets" not in history


def test_readout_ledger_dense_label_candidate_calibration_policy_is_advisory_only() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_dense_readout_label_candidate_review(
        dense_readout_label_candidate_review=_ready_dense_label_candidate_review(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-dense-label",
        confirmation=True,
    )
    runtime_state.mark_clean()
    before_revision = runtime_state.state_revision

    policy = ledger.dense_label_candidate_calibration_policy(limit=4)
    empty = ledger.dense_label_candidate_calibration_policy(limit=0)

    assert runtime_state.state_revision == before_revision
    assert runtime_state.dirty_state is False
    assert policy["surface"] == "snn_language_dense_label_candidate_calibration_policy.v1"
    assert policy["artifact_kind"] == "terminus_snn_language_dense_label_candidate_calibration_policy"
    assert policy["advisory"] is True
    assert policy["executable"] is False
    assert policy["records_ledger_event"] is False
    assert policy["runs_replay"] is False
    assert policy["writes_checkpoint"] is False
    assert policy["generates_text"] is False
    assert policy["decodes_text"] is False
    assert policy["trains_runtime_model"] is False
    assert policy["applies_plasticity"] is False
    assert policy["mutates_runtime_state"] is False
    assert policy["source_window"]["surface"] == (
        "bounded_snn_dense_label_candidate_calibration_source_window.v1"
    )
    assert policy["candidate_count"] == 1
    assert policy["ready_candidate_count"] == 1
    candidate = policy["calibration_candidates"][0]
    assert candidate["labels"] == ["prediction error", "concept focus"]
    assert candidate["eligible_for_dense_label_calibration_review"] is True
    assert candidate["eligible_for_replay_memory"] is False
    assert candidate["eligible_for_live_replay"] is False
    assert candidate["eligible_for_plasticity_application"] is False
    assert candidate["eligible_for_fact_promotion"] is False
    assert candidate["eligible_for_action"] is False
    assert policy["promotion_gate"][
        "eligible_for_operator_dense_label_calibration_review"
    ] is True
    assert policy["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert policy["promotion_gate"]["eligible_for_language_generation"] is False
    assert policy["promotion_gate"]["eligible_for_replay_memory"] is False
    assert policy["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert policy["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert policy["promotion_gate"]["eligible_for_action"] is False
    assert empty["ready_candidate_count"] == 0
    assert empty["promotion_gate"][
        "eligible_for_operator_dense_label_calibration_review"
    ] is False


def test_dense_label_candidate_history_and_policy_use_dense_source_window_only() -> None:
    class CountedRows:
        def __init__(self, field: str, count: int) -> None:
            self.field = field
            self.count = count
            self.iterated = 0

        def __iter__(self):
            for index in range(self.count):
                self.iterated += 1
                label_slot = index % 4
                yield {
                    "field": self.field,
                    "ordinal": index,
                    "dense_label_candidate_evidence_hash": f"{index + 1:064x}",
                    "dense_label_candidate_evidence_id": (
                        f"dense-label-candidate:{index}"
                    ),
                    "review_hash": f"{index + 1001:064x}",
                    "source_execution_hash": f"{index + 2001:064x}",
                    "label_hash": f"{label_slot + 3001:064x}",
                    "labels": [f"label-{label_slot}", f"focus-{index % 2}"],
                    "tensor_device": "cpu",
                    "active_count": (index % 16) + 1,
                }

        def __len__(self) -> int:
            return self.count

    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_limit = 8
    source_count = 256
    ledger_state: dict[str, object] = {
        field: CountedRows(field, source_count)
        for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }
    ledger_state["total_dense_label_candidate_count"] = source_count
    ledger_state["last_dense_label_candidate_recorded_at"] = (
        "2026-06-19T00:00:00+00:00"
    )
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
        limit=ledger_limit,
    )

    history = ledger.dense_label_candidate_history(limit=4)
    policy = ledger.dense_label_candidate_calibration_policy(limit=4)

    history_window = history["summary"]["source_window"]
    policy_window = policy["source_window"]
    assert history_window["surface"] == (
        "bounded_snn_dense_label_candidate_calibration_source_window.v1"
    )
    assert policy_window["surface"] == (
        "bounded_snn_dense_label_candidate_calibration_source_window.v1"
    )
    assert history_window["policy"] == (
        SNN_DENSE_LABEL_CANDIDATE_CALIBRATION_SOURCE_WINDOW_POLICY
    )
    assert policy_window["policy"] == (
        SNN_DENSE_LABEL_CANDIDATE_CALIBRATION_SOURCE_WINDOW_POLICY
    )
    assert history_window["source_window_count"] == ledger_limit
    assert policy_window["source_window_count"] == ledger_limit
    assert history_window["source_record_count"] == source_count
    assert policy_window["source_record_count"] == source_count
    assert history_window["source_payload_truncated"] is True
    assert policy_window["source_payload_truncated"] is True
    assert history["summary"]["returned_dense_label_candidate_event_count"] == 4
    assert history["summary"]["dense_label_candidate_event_count"] == ledger_limit
    assert policy["candidate_count"] == 4
    assert policy["ready_candidate_count"] == 4
    assert policy["promotion_gate"]["required_evidence"][
        "dense_label_candidate_source_window_bounded"
    ] is True
    assert policy_window["runs_live_tick"] is False
    assert policy_window["runs_every_token"] is False
    assert policy_window["global_candidate_scan"] is False
    assert policy_window["global_score_scan"] is False
    assert policy_window["raw_text_payload_loaded"] is False
    assert policy_window["language_reasoning"] is False
    assert policy_window["mutates_runtime_state"] is False
    assert policy_window["applies_plasticity"] is False
    assert policy_window["archival_storage_device"] == "cpu"
    assert policy_window["lookup_device"] == "cpu"
    assert policy_window["gpu_used"] is False
    for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS:
        source = ledger_state[field]
        assert isinstance(source, CountedRows)
        if field == "dense_label_candidate_events":
            assert source.iterated == ledger_limit * 2
        else:
            assert source.iterated == 0


def test_dense_label_calibration_update_application_uses_update_source_window_only() -> None:
    class CountedRows:
        def __init__(self, field: str, count: int) -> None:
            self.field = field
            self.count = count
            self.iterated = 0

        def __iter__(self):
            for index in range(self.count):
                self.iterated += 1
                yield {
                    "field": self.field,
                    "ordinal": index,
                    "applied_calibration_update_hash": f"{index + 1:064x}",
                    "applied_at": "2026-06-19T00:00:00+00:00",
                    "state_revision": index,
                    "method": "bounded_temperature_scaling",
                    "runtime_update_applied": True,
                    "weights_persisted": False,
                }

        def __len__(self) -> int:
            return self.count

    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_limit = 8
    source_count = 256
    counted_sources = {
        field: CountedRows(field, source_count)
        for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }
    ledger_state: dict[str, object] = dict(counted_sources)
    ledger_state.update(
        {
            "total_recorded_count": source_count,
            "total_dense_label_calibration_update_count": source_count,
            "last_dense_label_calibration_update_applied_at": (
                "2026-06-19T00:00:00+00:00"
            ),
            "current_text_surface_commit": {"surface": "preserve-current.v1"},
        }
    )
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
        limit=ledger_limit,
    )
    before_revision = runtime_state.state_revision
    preflight = {
        "surface": "snn_language_dense_label_candidate_calibration_update_preflight.v1",
        "ready": True,
        "observed_state_revision": before_revision,
        "expected_state_revision": before_revision,
        "preflight_hash": "1" * 64,
        "design_hash": "2" * 64,
        "review_hash": "3" * 64,
        "evaluation_hash": "4" * 64,
        "mutates_runtime_state": False,
        "trains_runtime_model": False,
        "applies_plasticity": False,
        "writes_checkpoint": False,
        "generates_text": False,
        "calibration_update_preflight": {
            "method": "bounded_temperature_scaling",
            "base_temperature": 1.0,
            "target_temperature": 1.1,
            "max_temperature_delta": 0.25,
            "checkpoint_path": "checkpoints/dense-label-calibration.json",
            "rollback_policy": {"available": True, "snapshot_id": "rollback-1"},
            "bounded_post_hoc_update": True,
            "runtime_update_applied": False,
            "weights_persisted": False,
        },
        "device_preflight": {
            "requested_device": "cpu",
            "cuda_requirement_satisfied": True,
            "executor_capability_available": True,
        },
        "promotion_gate": {
            "eligible_for_dense_label_calibration_update_executor": True
        },
    }

    applied = ledger.apply_dense_label_candidate_calibration_update(
        dense_label_candidate_calibration_update_preflight=preflight,
        expected_state_revision=before_revision,
        operator_id="operator-dense-label",
        confirmation=True,
    )
    review = ledger.dense_label_candidate_calibration_update_application_review(
        dense_label_candidate_calibration_update_application=applied,
        expected_state_revision=runtime_state.state_revision,
    )

    assert applied["accepted"] is True
    assert review["ready"] is True
    _assert_dense_label_calibration_update_source_window(
        applied["source_window"],
        expected_count=ledger_limit,
    )
    assert applied["source_window"]["source_record_count"] == source_count
    assert applied["source_window"]["source_payload_truncated"] is True
    assert applied["source_window"]["total_dense_label_calibration_update_count"] == (
        source_count
    )
    assert applied["ledger_summary"]["total_dense_label_calibration_update_count"] == (
        source_count + 1
    )
    assert applied["promotion_gate"]["required_evidence"][
        "dense_label_calibration_update_source_window_bounded"
    ] is True
    _assert_dense_label_calibration_update_source_window(
        review["source_window"],
        expected_count=ledger_limit,
    )
    assert review["promotion_gate"]["required_evidence"][
        "dense_label_calibration_update_source_window_bounded"
    ] is True
    stored_updates = ledger_state["dense_label_calibration_update_events"]
    assert isinstance(stored_updates, list)
    assert len(stored_updates) == ledger_limit
    assert stored_updates[0]["applied_calibration_update_hash"] == (
        applied["applied_calibration_update"]["applied_calibration_update_hash"]
    )
    assert ledger_state["current_text_surface_commit"] == {
        "surface": "preserve-current.v1"
    }
    assert ledger_state["total_recorded_count"] == source_count
    for field, source in counted_sources.items():
        if field == "dense_label_calibration_update_events":
            assert source.iterated == ledger_limit
        else:
            assert source.iterated == 0
            assert ledger_state[field] is source


def test_autonomous_confidence_use_uses_confidence_source_window_only() -> None:
    class CountedRows:
        def __init__(self, field: str, count: int) -> None:
            self.field = field
            self.count = count
            self.iterated = 0

        def __iter__(self):
            for index in range(self.count):
                self.iterated += 1
                yield {
                    "field": self.field,
                    "ordinal": index,
                    "autonomous_confidence_use_event_hash": f"{index + 1:064x}",
                    "used_at": "2026-06-19T00:00:00+00:00",
                    "state_revision": index,
                    "output_is_label_hash_only": True,
                }

        def __len__(self) -> int:
            return self.count

    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_limit = 8
    source_count = 256
    candidate_hash = "a" * 64
    counted_sources = {
        field: CountedRows(field, source_count)
        for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }
    ledger_state: dict[str, object] = dict(counted_sources)
    ledger_state.update(
        {
            "total_recorded_count": source_count,
            "total_autonomous_confidence_use_count": source_count,
            "last_autonomous_confidence_used_at": "2026-06-19T00:00:00+00:00",
            "current_text_surface_commit": {"surface": "preserve-current.v1"},
        }
    )
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
        limit=ledger_limit,
    )
    before_revision = runtime_state.state_revision
    preflight = {
        "surface": "snn_language_calibrated_dense_label_confidence_autonomous_use_preflight.v1",
        "ready": True,
        "requires_operator_approval": False,
        "observed_state_revision": before_revision,
        "expected_state_revision": before_revision,
        "autonomous_confidence_use_preflight_hash": "1" * 64,
        "autonomous_confidence_use_design_hash": "2" * 64,
        "mutates_runtime_state": False,
        "runs_recalibration": False,
        "runs_calibration_update": False,
        "trains_runtime_model": False,
        "runs_replay": False,
        "applies_plasticity": False,
        "writes_checkpoint": False,
        "generates_text": False,
        "decodes_text": False,
        "candidate_preflight": {
            "use_mode": "threshold_and_abstain",
            "min_confidence_threshold": 0.6,
            "max_candidates": 1,
            "candidate_count": 1,
            "candidate_hashes": [candidate_hash],
        },
        "promotion_gate": {
            "eligible_for_autonomous_calibrated_confidence_use_executor": True
        },
    }

    execution = ledger.execute_autonomous_calibrated_dense_label_confidence_use(
        calibrated_dense_label_confidence_autonomous_use_preflight=preflight,
        expected_state_revision=before_revision,
        candidate_evidence={
            "candidates": [
                {
                    "dense_label_candidate_evidence_hash": candidate_hash,
                    "label_hash": "b" * 64,
                    "calibrated_confidence": 0.75,
                    "pre_calibration_confidence": 0.8,
                }
            ]
        },
        execution_policy={"max_selected_candidates": 1},
    )
    review = ledger.autonomous_calibrated_dense_label_confidence_use_event_review(
        calibrated_dense_label_confidence_autonomous_use_executor=execution,
        expected_state_revision=runtime_state.state_revision,
        review_policy={"min_selected_candidates": 1, "max_selected_candidates": 2},
    )

    assert execution["accepted"] is True
    assert review["ready"] is True
    _assert_autonomous_confidence_use_source_window(
        execution["source_window"],
        expected_count=ledger_limit,
    )
    assert execution["source_window"]["source_record_count"] == source_count
    assert execution["source_window"]["source_payload_truncated"] is True
    assert execution["source_window"]["total_autonomous_confidence_use_count"] == (
        source_count
    )
    assert execution["ledger_summary"]["total_autonomous_confidence_use_count"] == (
        source_count + 1
    )
    assert execution["promotion_gate"]["required_evidence"][
        "autonomous_confidence_use_source_window_bounded"
    ] is True
    _assert_autonomous_confidence_use_source_window(
        review["source_window"],
        expected_count=ledger_limit,
    )
    assert review["promotion_gate"]["required_evidence"][
        "autonomous_confidence_use_source_window_bounded"
    ] is True
    stored_events = ledger_state["autonomous_confidence_use_events"]
    assert isinstance(stored_events, list)
    assert len(stored_events) == ledger_limit
    assert stored_events[0]["autonomous_confidence_use_event_hash"] == (
        execution["autonomous_confidence_use_event_hash"]
    )
    assert ledger_state["current_text_surface_commit"] == {
        "surface": "preserve-current.v1"
    }
    assert ledger_state["total_recorded_count"] == source_count
    for field, source in counted_sources.items():
        if field == "autonomous_confidence_use_events":
            assert source.iterated == ledger_limit
        else:
            assert source.iterated == 0
            assert ledger_state[field] is source


def test_readout_ledger_dense_label_calibration_evaluation_design_is_read_only() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_dense_readout_label_candidate_review(
        dense_readout_label_candidate_review=_ready_dense_label_candidate_review(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-dense-label",
        confirmation=True,
    )
    policy = ledger.dense_label_candidate_calibration_policy(limit=4)
    runtime_state.mark_clean()
    before_revision = runtime_state.state_revision

    blocked = ledger.dense_label_candidate_calibration_evaluation_design(
        dense_label_candidate_calibration_policy=policy,
        heldout_label_evidence={},
        device_evidence={"device": "cpu"},
    )
    ready = ledger.dense_label_candidate_calibration_evaluation_design(
        dense_label_candidate_calibration_policy=policy,
        heldout_label_evidence={
            "labels": ["prediction error", "concept focus"],
            "target_hash": _sha256_json(["prediction error", "concept focus"]),
        },
        design_policy={
            "metrics": ["expected_calibration_error", "coverage_gap"],
            "max_candidates": 2,
            "min_heldout_labels": 2,
        },
        device_evidence={"device": "cpu", "source": "unit"},
    )

    assert runtime_state.state_revision == before_revision
    assert runtime_state.dirty_state is False
    assert blocked["ready"] is False
    assert blocked["promotion_gate"]["required_evidence"][
        "heldout_label_evidence_available"
    ] is False
    assert ready["surface"] == "snn_language_dense_label_candidate_calibration_evaluation_design.v1"
    assert ready["ready"] is True
    assert ready["advisory"] is True
    assert ready["executable"] is False
    assert ready["records_ledger_event"] is False
    assert ready["runs_replay"] is False
    assert ready["runs_calibration_evaluation"] is False
    assert ready["writes_checkpoint"] is False
    assert ready["generates_text"] is False
    assert ready["decodes_text"] is False
    assert ready["trains_runtime_model"] is False
    assert ready["applies_plasticity"] is False
    assert ready["mutates_runtime_state"] is False
    assert ready["selected_candidate_count"] == 1
    assert ready["selected_calibration_candidates"][0]["labels"] == [
        "prediction error",
        "concept focus",
    ]
    assert ready["calibration_evaluation_design"]["requires_cross_validation"] is True
    assert ready["calibration_evaluation_design"][
        "requires_expected_calibration_error"
    ] is True
    assert ready["promotion_gate"][
        "eligible_for_dense_label_calibration_evaluation_preflight"
    ] is True
    assert ready["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert ready["promotion_gate"]["eligible_for_language_generation"] is False
    assert ready["promotion_gate"]["eligible_for_replay_memory"] is False
    assert ready["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert ready["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert ready["promotion_gate"]["eligible_for_action"] is False


def test_readout_ledger_dense_label_calibration_evaluation_preflight_requires_revision_and_executor() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_dense_readout_label_candidate_review(
        dense_readout_label_candidate_review=_ready_dense_label_candidate_review(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-dense-label",
        confirmation=True,
    )
    policy = ledger.dense_label_candidate_calibration_policy(limit=4)
    design = ledger.dense_label_candidate_calibration_evaluation_design(
        dense_label_candidate_calibration_policy=policy,
        heldout_label_evidence={
            "labels": ["prediction error", "concept focus"],
            "target_hash": _sha256_json(["prediction error", "concept focus"]),
        },
        design_policy={
            "metrics": ["expected_calibration_error", "coverage_gap"],
            "max_candidates": 2,
            "min_heldout_labels": 2,
        },
        device_evidence={"device": "cpu", "source": "unit"},
    )
    runtime_state.mark_clean()
    before_revision = runtime_state.state_revision

    blocked = ledger.dense_label_candidate_calibration_evaluation_preflight(
        dense_label_candidate_calibration_evaluation_design=design,
        expected_state_revision=runtime_state.state_revision + 1,
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"calibration_evaluation_executor": True},
    )
    missing_executor = ledger.dense_label_candidate_calibration_evaluation_preflight(
        dense_label_candidate_calibration_evaluation_design=design,
        expected_state_revision=runtime_state.state_revision,
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"calibration_evaluation_executor": False},
    )
    ready = ledger.dense_label_candidate_calibration_evaluation_preflight(
        dense_label_candidate_calibration_evaluation_design=design,
        expected_state_revision=runtime_state.state_revision,
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"calibration_evaluation_executor": True},
    )

    assert runtime_state.state_revision == before_revision
    assert runtime_state.dirty_state is False
    assert blocked["ready"] is False
    assert blocked["promotion_gate"]["required_evidence"][
        "expected_revision_current"
    ] is False
    assert missing_executor["ready"] is False
    assert missing_executor["promotion_gate"]["required_evidence"][
        "executor_capability_available"
    ] is False
    assert ready["surface"] == "snn_language_dense_label_candidate_calibration_evaluation_preflight.v1"
    assert ready["ready"] is True
    assert ready["advisory"] is True
    assert ready["executable"] is False
    assert ready["records_ledger_event"] is False
    assert ready["runs_replay"] is False
    assert ready["runs_calibration_evaluation"] is False
    assert ready["writes_checkpoint"] is False
    assert ready["generates_text"] is False
    assert ready["decodes_text"] is False
    assert ready["trains_runtime_model"] is False
    assert ready["applies_plasticity"] is False
    assert ready["mutates_runtime_state"] is False
    assert ready["device_preflight"]["requested_device"] == "cpu"
    assert ready["device_preflight"]["executor_capability_available"] is True
    assert ready["promotion_gate"][
        "eligible_for_dense_label_calibration_evaluation_executor"
    ] is True
    assert ready["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert ready["promotion_gate"]["eligible_for_language_generation"] is False
    assert ready["promotion_gate"]["eligible_for_replay_memory"] is False
    assert ready["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert ready["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert ready["promotion_gate"]["eligible_for_action"] is False


def test_readout_ledger_dense_label_calibration_evaluation_computes_metrics_without_mutation() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_dense_readout_label_candidate_review(
        dense_readout_label_candidate_review=_ready_dense_label_candidate_review(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-dense-label",
        confirmation=True,
    )
    policy = ledger.dense_label_candidate_calibration_policy(limit=4)
    design = ledger.dense_label_candidate_calibration_evaluation_design(
        dense_label_candidate_calibration_policy=policy,
        heldout_label_evidence={
            "labels": ["prediction error", "concept focus"],
            "target_hash": _sha256_json(["prediction error", "concept focus"]),
        },
        design_policy={
            "metrics": ["expected_calibration_error", "coverage_gap"],
            "max_candidates": 2,
            "min_heldout_labels": 2,
        },
        device_evidence={"device": "cpu", "source": "unit"},
    )
    preflight = ledger.dense_label_candidate_calibration_evaluation_preflight(
        dense_label_candidate_calibration_evaluation_design=design,
        expected_state_revision=runtime_state.state_revision,
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"calibration_evaluation_executor": True},
    )
    runtime_state.mark_clean()
    before_revision = runtime_state.state_revision

    blocked = ledger.dense_label_candidate_calibration_evaluation(
        dense_label_candidate_calibration_evaluation_preflight={
            **preflight,
            "ready": False,
        },
        heldout_label_evidence={"labels": ["prediction error", "concept focus"]},
    )
    ready = ledger.dense_label_candidate_calibration_evaluation(
        dense_label_candidate_calibration_evaluation_preflight=preflight,
        heldout_label_evidence={"labels": ["prediction error", "concept focus"]},
        bin_count=5,
    )

    assert runtime_state.state_revision == before_revision
    assert runtime_state.dirty_state is False
    assert blocked["ready"] is False
    assert blocked["sample_count"] == 0
    assert ready["surface"] == "snn_language_dense_label_candidate_calibration_evaluation.v1"
    assert ready["ready"] is True
    assert ready["advisory"] is True
    assert ready["executable"] is False
    assert ready["records_ledger_event"] is False
    assert ready["runs_replay"] is False
    assert ready["runs_calibration_evaluation"] is True
    assert ready["writes_checkpoint"] is False
    assert ready["generates_text"] is False
    assert ready["decodes_text"] is False
    assert ready["trains_runtime_model"] is False
    assert ready["applies_plasticity"] is False
    assert ready["mutates_runtime_state"] is False
    assert ready["sample_count"] == 1
    assert ready["source_window"]["surface"] == (
        "bounded_snn_dense_label_candidate_calibration_evaluation_source_window.v1"
    )
    assert (
        ready["source_window"]["policy"]
        == SNN_DENSE_LABEL_CALIBRATION_EVALUATION_SOURCE_WINDOW_POLICY
    )
    assert ready["source_window"]["runs_live_tick"] is False
    assert ready["source_window"]["runs_every_token"] is False
    assert ready["metrics"]["expected_calibration_error"] is not None
    assert ready["metrics"]["coverage_gap"] == 0.0
    assert len(ready["reliability_bins"]) == 5
    assert ready["evaluated_samples"][0]["heldout_match_count"] == 2
    assert ready["promotion_gate"][
        "eligible_for_dense_label_calibration_evaluation_review"
    ] is True
    assert ready["promotion_gate"]["required_evidence"][
        "dense_label_candidate_evaluation_source_window_bounded"
    ] is True
    assert ready["promotion_gate"]["required_evidence"][
        "preflight_selected_candidates_within_source_window"
    ] is True
    assert ready["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert ready["promotion_gate"]["eligible_for_language_generation"] is False
    assert ready["promotion_gate"]["eligible_for_replay_memory"] is False
    assert ready["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert ready["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert ready["promotion_gate"]["eligible_for_action"] is False


def test_dense_label_calibration_evaluation_uses_selected_source_window_only() -> None:
    class CountedRows:
        def __init__(self, field: str, count: int) -> None:
            self.field = field
            self.count = count
            self.iterated = 0

        def __iter__(self):
            for index in range(self.count):
                self.iterated += 1
                label_slot = index % 4
                yield {
                    "field": self.field,
                    "ordinal": index,
                    "dense_label_candidate_evidence_hash": f"{index + 1:064x}",
                    "dense_label_candidate_evidence_id": (
                        f"dense-label-candidate:{index}"
                    ),
                    "review_hash": f"{index + 1001:064x}",
                    "source_execution_hash": f"{index + 2001:064x}",
                    "label_hash": f"{label_slot + 3001:064x}",
                    "labels": [f"label-{label_slot}", f"focus-{index % 2}"],
                    "tensor_device": "cpu",
                    "active_count": (index % 16) + 1,
                }

        def __len__(self) -> int:
            return self.count

    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_limit = 8
    source_count = 256
    ledger_state: dict[str, object] = {
        field: CountedRows(field, source_count)
        for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }
    ledger_state["total_dense_label_candidate_count"] = source_count
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
        limit=ledger_limit,
    )
    selected_hashes = [f"{index + 1:064x}" for index in range(4)]
    preflight = {
        "surface": "snn_language_dense_label_candidate_calibration_evaluation_preflight.v1",
        "ready": True,
        "preflight_hash": _sha256_json({"preflight": "dense-label-eval-window"}),
        "selected_candidate_hashes": selected_hashes,
        "selected_candidate_count": len(selected_hashes),
        "mutates_runtime_state": False,
        "trains_runtime_model": False,
        "applies_plasticity": False,
        "writes_checkpoint": False,
        "generates_text": False,
        "promotion_gate": {
            "eligible_for_dense_label_calibration_evaluation_executor": True,
            "required_evidence": {
                "expected_revision_current": True,
                "executor_capability_available": True,
            },
        },
    }

    evaluation = ledger.dense_label_candidate_calibration_evaluation(
        dense_label_candidate_calibration_evaluation_preflight=preflight,
        heldout_label_evidence={"labels": ["label-0", "focus-0", "label-1"]},
        bin_count=5,
    )
    outside_window = ledger.dense_label_candidate_calibration_evaluation(
        dense_label_candidate_calibration_evaluation_preflight={
            **preflight,
            "selected_candidate_hashes": [f"{source_count:064x}"],
            "selected_candidate_count": 1,
        },
        heldout_label_evidence={"labels": ["label-0", "focus-0", "label-1"]},
        bin_count=5,
    )

    source_window = evaluation["source_window"]
    assert evaluation["ready"] is True
    assert evaluation["sample_count"] == 4
    assert source_window["surface"] == (
        "bounded_snn_dense_label_candidate_calibration_evaluation_source_window.v1"
    )
    assert (
        source_window["policy"]
        == SNN_DENSE_LABEL_CALIBRATION_EVALUATION_SOURCE_WINDOW_POLICY
    )
    assert source_window["source_window_count"] == ledger_limit
    assert source_window["source_record_count"] == source_count
    assert source_window["source_payload_truncated"] is True
    assert source_window["preflight_selected_hash_count"] == 4
    assert source_window["matched_candidate_event_count"] == 4
    assert source_window["selected_candidates_within_source_window"] is True
    assert source_window["global_candidate_scan"] is False
    assert source_window["global_score_scan"] is False
    assert source_window["raw_text_payload_loaded"] is False
    assert source_window["language_reasoning"] is False
    assert source_window["runs_live_tick"] is False
    assert source_window["runs_every_token"] is False
    assert source_window["archival_storage_device"] == "cpu"
    assert source_window["lookup_device"] == "cpu"
    assert source_window["evaluation_device"] == "cpu"
    assert source_window["gpu_used"] is False
    assert evaluation["promotion_gate"]["required_evidence"][
        "dense_label_candidate_evaluation_source_window_bounded"
    ] is True
    assert evaluation["promotion_gate"]["required_evidence"][
        "preflight_selected_candidates_within_source_window"
    ] is True
    assert outside_window["ready"] is False
    assert outside_window["source_window"][
        "selected_candidates_within_source_window"
    ] is False
    assert outside_window["promotion_gate"]["required_evidence"][
        "preflight_selected_candidates_within_source_window"
    ] is False
    for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS:
        source = ledger_state[field]
        assert isinstance(source, CountedRows)
        if field == "dense_label_candidate_events":
            assert source.iterated == ledger_limit * 2
        else:
            assert source.iterated == 0


def test_readout_ledger_dense_label_calibration_evaluation_review_gates_metrics_without_mutation() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_dense_readout_label_candidate_review(
        dense_readout_label_candidate_review=_ready_dense_label_candidate_review(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-dense-label",
        confirmation=True,
    )
    policy = ledger.dense_label_candidate_calibration_policy(limit=4)
    design = ledger.dense_label_candidate_calibration_evaluation_design(
        dense_label_candidate_calibration_policy=policy,
        heldout_label_evidence={
            "labels": ["prediction error", "concept focus"],
            "target_hash": _sha256_json(["prediction error", "concept focus"]),
        },
        design_policy={
            "metrics": ["expected_calibration_error", "coverage_gap"],
            "max_candidates": 2,
            "min_heldout_labels": 2,
        },
        device_evidence={"device": "cpu", "source": "unit"},
    )
    preflight = ledger.dense_label_candidate_calibration_evaluation_preflight(
        dense_label_candidate_calibration_evaluation_design=design,
        expected_state_revision=runtime_state.state_revision,
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"calibration_evaluation_executor": True},
    )
    evaluation = ledger.dense_label_candidate_calibration_evaluation(
        dense_label_candidate_calibration_evaluation_preflight=preflight,
        heldout_label_evidence={"labels": ["prediction error", "concept focus"]},
        bin_count=5,
    )
    runtime_state.mark_clean()
    before_revision = runtime_state.state_revision

    blocked = ledger.dense_label_candidate_calibration_evaluation_review(
        dense_label_candidate_calibration_evaluation=evaluation,
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-dense-label",
        confirmation=False,
    )
    ready = ledger.dense_label_candidate_calibration_evaluation_review(
        dense_label_candidate_calibration_evaluation=evaluation,
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-dense-label",
        confirmation=True,
        review_policy={
            "max_expected_calibration_error": 1.0,
            "max_coverage_gap": 0.1,
            "min_label_set_stability": 0.5,
        },
    )

    assert runtime_state.state_revision == before_revision
    assert runtime_state.dirty_state is False
    assert blocked["ready"] is False
    assert blocked["promotion_gate"]["required_evidence"]["confirmation"] is False
    assert ready["surface"] == "snn_language_dense_label_candidate_calibration_evaluation_review.v1"
    assert ready["ready"] is True
    assert ready["review_recorded"] is True
    assert ready["advisory"] is True
    assert ready["executable"] is False
    assert ready["records_ledger_event"] is False
    assert ready["runs_replay"] is False
    assert ready["runs_calibration_evaluation"] is False
    assert ready["writes_checkpoint"] is False
    assert ready["generates_text"] is False
    assert ready["decodes_text"] is False
    assert ready["trains_runtime_model"] is False
    assert ready["applies_plasticity"] is False
    assert ready["mutates_runtime_state"] is False
    assert ready["metric_review"]["metric_thresholds_met"] is True
    assert ready["promotion_gate"][
        "eligible_for_dense_label_calibration_update_design"
    ] is True
    assert ready["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert ready["promotion_gate"]["eligible_for_language_generation"] is False
    assert ready["promotion_gate"]["eligible_for_replay_memory"] is False
    assert ready["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert ready["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert ready["promotion_gate"]["eligible_for_action"] is False


def test_readout_ledger_dense_label_calibration_update_design_is_bounded_and_read_only() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_dense_readout_label_candidate_review(
        dense_readout_label_candidate_review=_ready_dense_label_candidate_review(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-dense-label",
        confirmation=True,
    )
    policy = ledger.dense_label_candidate_calibration_policy(limit=4)
    design = ledger.dense_label_candidate_calibration_evaluation_design(
        dense_label_candidate_calibration_policy=policy,
        heldout_label_evidence={
            "labels": ["prediction error", "concept focus"],
            "target_hash": _sha256_json(["prediction error", "concept focus"]),
        },
        design_policy={
            "metrics": ["expected_calibration_error", "coverage_gap"],
            "max_candidates": 2,
            "min_heldout_labels": 2,
        },
        device_evidence={"device": "cpu", "source": "unit"},
    )
    preflight = ledger.dense_label_candidate_calibration_evaluation_preflight(
        dense_label_candidate_calibration_evaluation_design=design,
        expected_state_revision=runtime_state.state_revision,
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"calibration_evaluation_executor": True},
    )
    evaluation = ledger.dense_label_candidate_calibration_evaluation(
        dense_label_candidate_calibration_evaluation_preflight=preflight,
        heldout_label_evidence={"labels": ["prediction error", "concept focus"]},
        bin_count=5,
    )
    review = ledger.dense_label_candidate_calibration_evaluation_review(
        dense_label_candidate_calibration_evaluation=evaluation,
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-dense-label",
        confirmation=True,
        review_policy={
            "max_expected_calibration_error": 1.0,
            "max_coverage_gap": 0.1,
            "min_label_set_stability": 0.5,
        },
    )
    runtime_state.mark_clean()
    before_revision = runtime_state.state_revision

    blocked = ledger.dense_label_candidate_calibration_update_design(
        dense_label_candidate_calibration_evaluation_review=review,
        update_policy={"method": "bounded_temperature_scaling"},
        rollback_policy={},
        device_evidence={"device": "cpu", "source": "unit"},
    )
    ready = ledger.dense_label_candidate_calibration_update_design(
        dense_label_candidate_calibration_evaluation_review=review,
        update_policy={
            "method": "bounded_temperature_scaling",
            "base_temperature": 1.0,
            "max_temperature_delta": 0.25,
        },
        rollback_policy={"available": True, "snapshot_id": "calibration-snapshot"},
        device_evidence={"device": "cpu", "source": "unit"},
    )

    assert runtime_state.state_revision == before_revision
    assert runtime_state.dirty_state is False
    assert blocked["ready"] is False
    assert blocked["promotion_gate"]["required_evidence"][
        "rollback_policy_available"
    ] is False
    assert ready["surface"] == "snn_language_dense_label_candidate_calibration_update_design.v1"
    assert ready["ready"] is True
    assert ready["advisory"] is True
    assert ready["executable"] is False
    assert ready["records_ledger_event"] is False
    assert ready["runs_replay"] is False
    assert ready["writes_checkpoint"] is False
    assert ready["generates_text"] is False
    assert ready["decodes_text"] is False
    assert ready["trains_runtime_model"] is False
    assert ready["applies_plasticity"] is False
    assert ready["mutates_runtime_state"] is False
    update_design = ready["calibration_update_design"]
    assert update_design["method"] == "bounded_temperature_scaling"
    assert update_design["bounded_post_hoc_update"] is True
    assert update_design["runtime_update_applied"] is False
    assert update_design["weights_persisted"] is False
    assert update_design["target_temperature"] <= 1.25
    assert ready["promotion_gate"][
        "eligible_for_dense_label_calibration_update_preflight"
    ] is True
    assert ready["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert ready["promotion_gate"]["eligible_for_language_generation"] is False
    assert ready["promotion_gate"]["eligible_for_replay_memory"] is False
    assert ready["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert ready["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert ready["promotion_gate"]["eligible_for_action"] is False

    blocked_preflight = ledger.dense_label_candidate_calibration_update_preflight(
        dense_label_candidate_calibration_update_design=ready,
        expected_state_revision=before_revision,
        checkpoint_path="",
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"calibration_update_executor": False},
    )
    ready_preflight = ledger.dense_label_candidate_calibration_update_preflight(
        dense_label_candidate_calibration_update_design=ready,
        expected_state_revision=before_revision,
        checkpoint_path="checkpoints/dense-label-calibration.json",
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"calibration_update_executor": True},
    )

    assert runtime_state.state_revision == before_revision
    assert runtime_state.dirty_state is False
    assert blocked_preflight["ready"] is False
    assert blocked_preflight["promotion_gate"]["required_evidence"][
        "checkpoint_path_available"
    ] is False
    assert blocked_preflight["promotion_gate"]["required_evidence"][
        "executor_capability_available"
    ] is False
    assert (
        ready_preflight["surface"]
        == "snn_language_dense_label_candidate_calibration_update_preflight.v1"
    )
    assert ready_preflight["ready"] is True
    assert ready_preflight["advisory"] is True
    assert ready_preflight["executable"] is False
    assert ready_preflight["records_ledger_event"] is False
    assert ready_preflight["runs_replay"] is False
    assert ready_preflight["runs_calibration_update"] is False
    assert ready_preflight["writes_checkpoint"] is False
    assert ready_preflight["generates_text"] is False
    assert ready_preflight["decodes_text"] is False
    assert ready_preflight["trains_runtime_model"] is False
    assert ready_preflight["applies_plasticity"] is False
    assert ready_preflight["mutates_runtime_state"] is False
    assert ready_preflight["design_hash"] == ready["design_hash"]
    assert ready_preflight["review_hash"] == ready["review_hash"]
    assert ready_preflight["evaluation_hash"] == ready["evaluation_hash"]
    assert (
        ready_preflight["calibration_update_preflight"]["checkpoint_path"]
        == "checkpoints/dense-label-calibration.json"
    )
    assert ready_preflight["calibration_update_preflight"][
        "runtime_update_applied"
    ] is False
    assert ready_preflight["calibration_update_preflight"]["weights_persisted"] is False
    assert ready_preflight["promotion_gate"][
        "eligible_for_dense_label_calibration_update_executor"
    ] is True
    assert ready_preflight["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert ready_preflight["promotion_gate"]["eligible_for_language_generation"] is False
    assert ready_preflight["promotion_gate"]["eligible_for_replay_memory"] is False
    assert ready_preflight["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert ready_preflight["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert ready_preflight["promotion_gate"]["eligible_for_action"] is False

    blocked_application = ledger.apply_dense_label_candidate_calibration_update(
        dense_label_candidate_calibration_update_preflight=ready_preflight,
        expected_state_revision=before_revision,
        operator_id="operator-dense-label",
        confirmation=False,
    )
    applied = ledger.apply_dense_label_candidate_calibration_update(
        dense_label_candidate_calibration_update_preflight=ready_preflight,
        expected_state_revision=before_revision,
        operator_id="operator-dense-label",
        confirmation=True,
    )
    stale_reapply = ledger.apply_dense_label_candidate_calibration_update(
        dense_label_candidate_calibration_update_preflight=ready_preflight,
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-dense-label",
        confirmation=True,
    )
    blocked_review = ledger.dense_label_candidate_calibration_update_application_review(
        dense_label_candidate_calibration_update_application={
            **applied,
            "applied_calibration_update": {
                **applied["applied_calibration_update"],
                "applied_calibration_update_hash": "0" * 64,
            },
        },
        expected_state_revision=runtime_state.state_revision,
    )
    application_review = ledger.dense_label_candidate_calibration_update_application_review(
        dense_label_candidate_calibration_update_application=applied,
        expected_state_revision=runtime_state.state_revision,
        review_policy={"max_temperature_delta": 0.25},
    )
    observation_samples = [
        {
            "sample_hash": _sha256_json(["sample", index]),
            "label_hash": _sha256_json(["label", index]),
            "pre_calibration_confidence": 0.8,
            "calibrated_confidence": 0.9 if index != 1 else 0.15,
            "correct": index != 1,
        }
        for index in range(3)
    ]
    blocked_observation = ledger.dense_label_candidate_post_calibration_observation_window(
        dense_label_candidate_calibration_update_application_review=application_review,
        observation_evidence={"samples": observation_samples[:1]},
        expected_state_revision=runtime_state.state_revision,
        window_policy={"min_samples": 3},
    )
    observation_window = ledger.dense_label_candidate_post_calibration_observation_window(
        dense_label_candidate_calibration_update_application_review=application_review,
        observation_evidence={"samples": observation_samples},
        expected_state_revision=runtime_state.state_revision,
        window_policy={
            "min_samples": 3,
            "max_expected_calibration_error": 0.2,
            "max_confidence_drift": 0.7,
        },
    )
    blocked_operator_review = ledger.dense_label_candidate_post_calibration_operator_review(
        dense_label_candidate_post_calibration_observation_window=observation_window,
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-dense-label",
        confirmation=False,
    )
    operator_review = ledger.dense_label_candidate_post_calibration_operator_review(
        dense_label_candidate_post_calibration_observation_window=observation_window,
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-dense-label",
        confirmation=True,
        review_policy={
            "min_samples": 3,
            "max_expected_calibration_error": 0.2,
            "max_confidence_drift": 0.7,
        },
    )
    blocked_confidence_use_design = ledger.calibrated_dense_label_confidence_use_design(
        dense_label_candidate_post_calibration_operator_review=operator_review,
        confidence_use_policy={"use_mode": "generate_text"},
        device_evidence={},
    )
    confidence_use_design = ledger.calibrated_dense_label_confidence_use_design(
        dense_label_candidate_post_calibration_operator_review=operator_review,
        confidence_use_policy={
            "use_mode": "threshold_and_abstain",
            "min_confidence_threshold": 0.6,
            "max_candidates": 4,
        },
        device_evidence={"device": "cpu", "source": "unit"},
    )
    blocked_confidence_use_preflight = ledger.calibrated_dense_label_confidence_use_preflight(
        dense_label_confidence_use_design=confidence_use_design,
        expected_state_revision=runtime_state.state_revision,
        candidate_evidence={
            "candidates": [
                {
                    "dense_label_candidate_evidence_hash": _sha256_json(
                        ["candidate", "blocked"]
                    ),
                    "label_hash": _sha256_json(["label", "blocked"]),
                    "calibrated_confidence": 0.4,
                    "pre_calibration_confidence": 0.8,
                }
            ]
        },
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"calibrated_confidence_use_executor": True},
    )
    confidence_use_preflight = ledger.calibrated_dense_label_confidence_use_preflight(
        dense_label_confidence_use_design=confidence_use_design,
        expected_state_revision=runtime_state.state_revision,
        candidate_evidence={
            "candidates": [
                {
                    "dense_label_candidate_evidence_hash": _sha256_json(
                        ["candidate", "ready"]
                    ),
                    "label_hash": _sha256_json(["label", "ready"]),
                    "calibrated_confidence": 0.75,
                    "pre_calibration_confidence": 0.8,
                }
            ]
        },
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"calibrated_confidence_use_executor": True},
    )
    blocked_confidence_use_executor = (
        ledger.execute_calibrated_dense_label_confidence_use(
            calibrated_dense_label_confidence_use_preflight=blocked_confidence_use_preflight,
            expected_state_revision=runtime_state.state_revision,
            candidate_evidence={
                "candidates": [
                    {
                        "dense_label_candidate_evidence_hash": _sha256_json(
                            ["candidate", "blocked"]
                        ),
                        "label_hash": _sha256_json(["label", "blocked"]),
                        "calibrated_confidence": 0.4,
                        "pre_calibration_confidence": 0.8,
                    }
                ]
            },
        )
    )
    confidence_use_executor = ledger.execute_calibrated_dense_label_confidence_use(
        calibrated_dense_label_confidence_use_preflight=confidence_use_preflight,
        expected_state_revision=runtime_state.state_revision,
        candidate_evidence={
            "candidates": [
                {
                    "dense_label_candidate_evidence_hash": _sha256_json(
                        ["candidate", "ready"]
                    ),
                    "label_hash": _sha256_json(["label", "ready"]),
                    "calibrated_confidence": 0.75,
                    "pre_calibration_confidence": 0.8,
                }
            ]
        },
        execution_policy={"max_selected_candidates": 1},
    )
    blocked_confidence_display_review = (
        ledger.calibrated_dense_label_confidence_operator_display_review(
            calibrated_dense_label_confidence_use_executor=confidence_use_executor,
            expected_state_revision=runtime_state.state_revision,
            operator_id="operator-dense-label",
            confirmation=False,
        )
    )
    confidence_display_review = (
        ledger.calibrated_dense_label_confidence_operator_display_review(
            calibrated_dense_label_confidence_use_executor=confidence_use_executor,
            expected_state_revision=runtime_state.state_revision,
            operator_id="operator-dense-label",
            confirmation=True,
            review_policy={"max_ranked_display": 4},
        )
    )
    selected_hash = _sha256_json(["candidate", "ready"])
    blocked_internal_stability_review = (
        ledger.calibrated_dense_label_confidence_internal_stability_review(
            calibrated_dense_label_confidence_use_executor=confidence_use_executor,
            expected_state_revision=runtime_state.state_revision,
            stability_evidence={
                "cycles": [
                    {
                        "selected_candidate_hashes": [selected_hash],
                        "selected_confidence": 0.75,
                    }
                ]
            },
            review_policy={"min_cycles": 3, "max_confidence_drift": 0.05},
        )
    )
    internal_stability_review = (
        ledger.calibrated_dense_label_confidence_internal_stability_review(
            calibrated_dense_label_confidence_use_executor=confidence_use_executor,
            expected_state_revision=runtime_state.state_revision,
            stability_evidence={
                "cycles": [
                    {
                        "selected_candidate_hashes": [selected_hash],
                        "selected_confidence": 0.75,
                    },
                    {
                        "selected_candidate_hashes": [selected_hash],
                        "selected_confidence": 0.74,
                    },
                    {
                        "selected_candidate_hashes": [selected_hash],
                        "selected_confidence": 0.76,
                    },
                ]
            },
            review_policy={"min_cycles": 3, "max_confidence_drift": 0.05},
        )
    )
    blocked_autonomous_replay_design = (
        ledger.calibrated_dense_label_confidence_autonomous_replay_review_design(
            calibrated_dense_label_confidence_internal_stability_review=(
                internal_stability_review
            ),
            replay_policy={"max_replay_cycles": 4},
            device_evidence={},
        )
    )
    autonomous_replay_design = (
        ledger.calibrated_dense_label_confidence_autonomous_replay_review_design(
            calibrated_dense_label_confidence_internal_stability_review=(
                internal_stability_review
            ),
            replay_policy={
                "max_replay_cycles": 4,
                "min_replay_consistency": 1.0,
                "max_confidence_drift": 0.05,
            },
            device_evidence={"device": "cpu", "source": "unit"},
        )
    )
    blocked_autonomous_replay_preflight = (
        ledger.calibrated_dense_label_confidence_autonomous_replay_review_preflight(
            calibrated_dense_label_confidence_autonomous_replay_review_design=(
                autonomous_replay_design
            ),
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cpu", "source": "unit"},
            executor_capabilities={
                "autonomous_confidence_replay_review_executor": False
            },
        )
    )
    autonomous_replay_preflight = (
        ledger.calibrated_dense_label_confidence_autonomous_replay_review_preflight(
            calibrated_dense_label_confidence_autonomous_replay_review_design=(
                autonomous_replay_design
            ),
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cpu", "source": "unit"},
            executor_capabilities={
                "autonomous_confidence_replay_review_executor": True
            },
        )
    )
    blocked_autonomous_replay_executor = (
        ledger.execute_calibrated_dense_label_confidence_autonomous_replay_review(
            calibrated_dense_label_confidence_autonomous_replay_review_preflight=(
                blocked_autonomous_replay_preflight
            ),
            expected_state_revision=runtime_state.state_revision,
            replay_cycle_evidence={
                "cycles": [
                    {
                        "cycle_index": 0,
                        "selected_candidate_hashes": [selected_hash],
                        "selected_confidence": 0.75,
                    }
                ]
            },
        )
    )
    autonomous_replay_executor = (
        ledger.execute_calibrated_dense_label_confidence_autonomous_replay_review(
            calibrated_dense_label_confidence_autonomous_replay_review_preflight=(
                autonomous_replay_preflight
            ),
            expected_state_revision=runtime_state.state_revision,
            replay_cycle_evidence={
                "cycles": [
                    {
                        "cycle_index": 0,
                        "selected_candidate_hashes": [selected_hash],
                        "selected_confidence": 0.75,
                    },
                    {
                        "cycle_index": 1,
                        "selected_candidate_hashes": [selected_hash],
                        "selected_confidence": 0.74,
                    },
                    {
                        "cycle_index": 2,
                        "selected_candidate_hashes": [selected_hash],
                        "selected_confidence": 0.76,
                    },
                    {
                        "cycle_index": 3,
                        "selected_candidate_hashes": [selected_hash],
                        "selected_confidence": 0.75,
                    },
                ]
            },
        )
    )
    blocked_autonomous_recalibration_design = (
        ledger.calibrated_dense_label_confidence_autonomous_recalibration_design(
            calibrated_dense_label_confidence_autonomous_replay_review_executor=(
                blocked_autonomous_replay_executor
            ),
            recalibration_policy={"method": "unbounded_self_training"},
            rollback_policy={},
            device_evidence={},
        )
    )
    autonomous_recalibration_design = (
        ledger.calibrated_dense_label_confidence_autonomous_recalibration_design(
            calibrated_dense_label_confidence_autonomous_replay_review_executor=(
                autonomous_replay_executor
            ),
            recalibration_policy={
                "method": "bounded_temperature_scaling",
                "max_temperature_delta": 0.05,
                "max_confidence_rescale_delta": 0.05,
                "min_replay_consistency": 1.0,
                "max_confidence_drift": 0.05,
            },
            rollback_policy={"can_restore_previous_calibration": True},
            device_evidence={"device": "cpu", "source": "unit"},
        )
    )
    blocked_autonomous_recalibration_preflight = (
        ledger.calibrated_dense_label_confidence_autonomous_recalibration_preflight(
            calibrated_dense_label_confidence_autonomous_recalibration_design=(
                autonomous_recalibration_design
            ),
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cpu", "source": "unit"},
            executor_capabilities={
                "autonomous_confidence_recalibration_executor": False
            },
        )
    )
    autonomous_recalibration_preflight = (
        ledger.calibrated_dense_label_confidence_autonomous_recalibration_preflight(
            calibrated_dense_label_confidence_autonomous_recalibration_design=(
                autonomous_recalibration_design
            ),
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cpu", "source": "unit"},
            executor_capabilities={
                "autonomous_confidence_recalibration_executor": True
            },
        )
    )
    blocked_autonomous_recalibration_executor = (
        ledger.execute_calibrated_dense_label_confidence_autonomous_recalibration(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight=(
                blocked_autonomous_recalibration_preflight
            ),
            expected_state_revision=runtime_state.state_revision,
        )
    )

    assert blocked_application["accepted"] is False
    assert blocked_application["mutates_runtime_state"] is False
    assert blocked_application["promotion_gate"]["required_evidence"][
        "confirmation"
    ] is False
    assert applied["surface"] == (
        "snn_language_dense_label_candidate_calibration_update_application.v1"
    )
    assert applied["accepted"] is True
    assert applied["duplicate"] is False
    assert applied["records_ledger_event"] is True
    assert applied["runs_calibration_update"] is True
    assert applied["writes_checkpoint"] is False
    assert applied["generates_text"] is False
    assert applied["decodes_text"] is False
    assert applied["trains_runtime_model"] is False
    assert applied["applies_plasticity"] is False
    assert applied["mutates_runtime_state"] is True
    assert applied["before"]["state_revision"] == before_revision
    assert applied["after"]["state_revision"] == before_revision + 1
    assert runtime_state.state_revision == before_revision + 1
    assert runtime_state.dirty_state is True
    applied_update = applied["applied_calibration_update"]
    assert applied_update["preflight_hash"] == ready_preflight["preflight_hash"]
    assert applied_update["design_hash"] == ready["design_hash"]
    assert applied_update["target_temperature"] <= 1.25
    assert applied_update["runtime_update_applied"] is True
    assert applied_update["weights_persisted"] is False
    assert applied["ledger_summary"]["dense_label_calibration_update_event_count"] == 1
    assert applied["ledger_summary"]["total_dense_label_calibration_update_count"] == 1
    _assert_dense_label_calibration_update_source_window(
        applied["source_window"],
        expected_count=0,
    )
    assert applied["promotion_gate"]["required_evidence"][
        "dense_label_calibration_update_source_window_bounded"
    ] is True
    assert applied["promotion_gate"][
        "eligible_for_dense_label_calibration_application_review"
    ] is True
    assert applied["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert applied["promotion_gate"]["eligible_for_language_generation"] is False
    assert applied["promotion_gate"]["eligible_for_replay_memory"] is False
    assert applied["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert applied["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert applied["promotion_gate"]["eligible_for_action"] is False
    assert stale_reapply["accepted"] is False
    assert stale_reapply["promotion_gate"]["required_evidence"][
        "preflight_revision_current"
    ] is False
    assert blocked_review["ready"] is False
    assert blocked_review["promotion_gate"]["required_evidence"][
        "current_applied_hash_matches"
    ] is False
    assert application_review["surface"] == (
        "snn_language_dense_label_candidate_calibration_update_application_review.v1"
    )
    assert application_review["ready"] is True
    assert application_review["advisory"] is True
    assert application_review["executable"] is False
    assert application_review["records_ledger_event"] is False
    assert application_review["runs_replay"] is False
    assert application_review["runs_calibration_update"] is False
    assert application_review["writes_checkpoint"] is False
    assert application_review["generates_text"] is False
    assert application_review["decodes_text"] is False
    assert application_review["trains_runtime_model"] is False
    assert application_review["applies_plasticity"] is False
    assert application_review["mutates_runtime_state"] is False
    assert application_review["applied_calibration_update_hash"] == applied_update[
        "applied_calibration_update_hash"
    ]
    assert application_review["current_dense_label_calibration_update_hash"] == (
        applied_update["applied_calibration_update_hash"]
    )
    _assert_dense_label_calibration_update_source_window(
        application_review["source_window"],
        expected_count=1,
    )
    assert application_review["source_window"][
        "current_dense_label_calibration_update_hash"
    ] == applied_update["applied_calibration_update_hash"]
    assert application_review["promotion_gate"]["required_evidence"][
        "dense_label_calibration_update_source_window_bounded"
    ] is True
    assert application_review["applied_calibration_review"][
        "runtime_update_applied"
    ] is True
    assert application_review["applied_calibration_review"]["weights_persisted"] is False
    assert application_review["promotion_gate"][
        "eligible_for_post_calibration_observation_window"
    ] is True
    assert application_review["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert application_review["promotion_gate"]["eligible_for_language_generation"] is False
    assert application_review["promotion_gate"]["eligible_for_replay_memory"] is False
    assert application_review["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert application_review["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert application_review["promotion_gate"]["eligible_for_action"] is False
    assert blocked_observation["ready"] is False
    assert blocked_observation["promotion_gate"]["required_evidence"][
        "sample_count_sufficient"
    ] is False
    assert observation_window["surface"] == (
        "snn_language_dense_label_candidate_post_calibration_observation_window.v1"
    )
    assert observation_window["ready"] is True
    assert observation_window["advisory"] is True
    assert observation_window["executable"] is False
    assert observation_window["records_ledger_event"] is False
    assert observation_window["runs_replay"] is False
    assert observation_window["runs_calibration_update"] is False
    assert observation_window["writes_checkpoint"] is False
    assert observation_window["generates_text"] is False
    assert observation_window["decodes_text"] is False
    assert observation_window["trains_runtime_model"] is False
    assert observation_window["applies_plasticity"] is False
    assert observation_window["mutates_runtime_state"] is False
    assert observation_window["sample_count"] == 3
    assert observation_window["metrics"]["expected_calibration_error"] <= 0.2
    assert observation_window["metrics"]["mean_confidence_drift"] <= 0.7
    assert observation_window["promotion_gate"][
        "eligible_for_post_calibration_operator_review"
    ] is True
    assert observation_window["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert observation_window["promotion_gate"]["eligible_for_language_generation"] is False
    assert observation_window["promotion_gate"]["eligible_for_replay_memory"] is False
    assert observation_window["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert observation_window["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert observation_window["promotion_gate"]["eligible_for_action"] is False
    assert blocked_operator_review["ready"] is False
    assert blocked_operator_review["promotion_gate"]["required_evidence"][
        "confirmation"
    ] is False
    assert operator_review["surface"] == (
        "snn_language_dense_label_candidate_post_calibration_operator_review.v1"
    )
    assert operator_review["ready"] is True
    assert operator_review["review_recorded"] is True
    assert operator_review["advisory"] is True
    assert operator_review["executable"] is False
    assert operator_review["records_ledger_event"] is False
    assert operator_review["runs_replay"] is False
    assert operator_review["runs_calibration_update"] is False
    assert operator_review["writes_checkpoint"] is False
    assert operator_review["generates_text"] is False
    assert operator_review["decodes_text"] is False
    assert operator_review["trains_runtime_model"] is False
    assert operator_review["applies_plasticity"] is False
    assert operator_review["mutates_runtime_state"] is False
    assert operator_review["observation_hash"] == observation_window["observation_hash"]
    assert operator_review["metric_review"]["metric_thresholds_met"] is True
    assert operator_review["promotion_gate"][
        "eligible_for_calibrated_dense_label_confidence_use_design"
    ] is True
    assert operator_review["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert operator_review["promotion_gate"]["eligible_for_language_generation"] is False
    assert operator_review["promotion_gate"]["eligible_for_replay_memory"] is False
    assert operator_review["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert operator_review["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert operator_review["promotion_gate"]["eligible_for_action"] is False
    assert blocked_confidence_use_design["ready"] is False
    assert blocked_confidence_use_design["promotion_gate"]["required_evidence"][
        "use_mode_supported"
    ] is False
    assert blocked_confidence_use_design["promotion_gate"]["required_evidence"][
        "device_evidence_available"
    ] is False
    assert confidence_use_design["surface"] == (
        "snn_language_calibrated_dense_label_confidence_use_design.v1"
    )
    assert confidence_use_design["ready"] is True
    assert confidence_use_design["advisory"] is True
    assert confidence_use_design["executable"] is False
    assert confidence_use_design["records_ledger_event"] is False
    assert confidence_use_design["runs_replay"] is False
    assert confidence_use_design["runs_calibration_update"] is False
    assert confidence_use_design["writes_checkpoint"] is False
    assert confidence_use_design["generates_text"] is False
    assert confidence_use_design["decodes_text"] is False
    assert confidence_use_design["trains_runtime_model"] is False
    assert confidence_use_design["applies_plasticity"] is False
    assert confidence_use_design["mutates_runtime_state"] is False
    use_design = confidence_use_design["confidence_use_design"]
    assert use_design["use_mode"] == "threshold_and_abstain"
    assert use_design["min_confidence_threshold"] == 0.6
    assert "generate_language" in use_design["disallowed_operations"]
    assert confidence_use_design["promotion_gate"][
        "eligible_for_calibrated_dense_label_confidence_use_preflight"
    ] is True
    assert confidence_use_design["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert confidence_use_design["promotion_gate"]["eligible_for_language_generation"] is False
    assert confidence_use_design["promotion_gate"]["eligible_for_replay_memory"] is False
    assert confidence_use_design["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert confidence_use_design["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert confidence_use_design["promotion_gate"]["eligible_for_action"] is False
    assert blocked_confidence_use_preflight["ready"] is False
    assert blocked_confidence_use_preflight["promotion_gate"]["required_evidence"][
        "threshold_mode_has_passing_candidate"
    ] is False
    assert confidence_use_preflight["surface"] == (
        "snn_language_calibrated_dense_label_confidence_use_preflight.v1"
    )
    assert confidence_use_preflight["ready"] is True
    assert confidence_use_preflight["advisory"] is True
    assert confidence_use_preflight["executable"] is False
    assert confidence_use_preflight["records_ledger_event"] is False
    assert confidence_use_preflight["runs_replay"] is False
    assert confidence_use_preflight["runs_calibration_update"] is False
    assert confidence_use_preflight["writes_checkpoint"] is False
    assert confidence_use_preflight["generates_text"] is False
    assert confidence_use_preflight["decodes_text"] is False
    assert confidence_use_preflight["trains_runtime_model"] is False
    assert confidence_use_preflight["applies_plasticity"] is False
    assert confidence_use_preflight["mutates_runtime_state"] is False
    assert confidence_use_preflight["design_hash"] == confidence_use_design["design_hash"]
    assert confidence_use_preflight["candidate_preflight"]["candidate_count"] == 1
    assert confidence_use_preflight["candidate_preflight"]["passing_candidate_count"] == 1
    assert confidence_use_preflight["device_preflight"][
        "executor_capability_available"
    ] is True
    assert confidence_use_preflight["promotion_gate"][
        "eligible_for_calibrated_dense_label_confidence_use_executor"
    ] is True
    assert confidence_use_preflight["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert confidence_use_preflight["promotion_gate"]["eligible_for_language_generation"] is False
    assert confidence_use_preflight["promotion_gate"]["eligible_for_replay_memory"] is False
    assert confidence_use_preflight["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert confidence_use_preflight["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert confidence_use_preflight["promotion_gate"]["eligible_for_action"] is False
    assert blocked_confidence_use_executor["ready"] is False
    assert blocked_confidence_use_executor["executable"] is False
    assert blocked_confidence_use_executor["promotion_gate"]["required_evidence"][
        "preflight_ready"
    ] is False
    assert confidence_use_executor["surface"] == (
        "snn_language_calibrated_dense_label_confidence_use_executor.v1"
    )
    assert confidence_use_executor["ready"] is True
    assert confidence_use_executor["advisory"] is False
    assert confidence_use_executor["executable"] is True
    assert confidence_use_executor["records_ledger_event"] is False
    assert confidence_use_executor["runs_replay"] is False
    assert confidence_use_executor["runs_calibration_update"] is False
    assert confidence_use_executor["writes_checkpoint"] is False
    assert confidence_use_executor["generates_text"] is False
    assert confidence_use_executor["decodes_text"] is False
    assert confidence_use_executor["trains_runtime_model"] is False
    assert confidence_use_executor["applies_plasticity"] is False
    assert confidence_use_executor["mutates_runtime_state"] is False
    result = confidence_use_executor["confidence_use_result"]
    assert result["use_mode"] == "threshold_and_abstain"
    assert result["selected_candidate_count"] == 1
    assert result["abstained"] is False
    assert result["output_is_label_hash_only"] is True
    assert result["selected_candidate_refs"][0]["calibrated_confidence"] == 0.75
    assert "label_hash" in result["selected_candidate_refs"][0]
    assert "label" not in result["selected_candidate_refs"][0]
    assert confidence_use_executor["promotion_gate"][
        "eligible_for_operator_display_confidence_result"
    ] is True
    assert confidence_use_executor["promotion_gate"]["eligible_for_language_generation"] is False
    assert confidence_use_executor["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert confidence_use_executor["promotion_gate"]["eligible_for_replay_memory"] is False
    assert confidence_use_executor["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert confidence_use_executor["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert confidence_use_executor["promotion_gate"]["eligible_for_action"] is False
    assert blocked_confidence_display_review["ready"] is False
    assert blocked_confidence_display_review["promotion_gate"]["required_evidence"][
        "confirmation"
    ] is False
    assert confidence_display_review["surface"] == (
        "snn_language_calibrated_dense_label_confidence_operator_display_review.v1"
    )
    assert confidence_display_review["ready"] is True
    assert confidence_display_review["review_recorded"] is True
    assert confidence_display_review["advisory"] is True
    assert confidence_display_review["executable"] is False
    assert confidence_display_review["records_ledger_event"] is False
    assert confidence_display_review["runs_replay"] is False
    assert confidence_display_review["runs_calibration_update"] is False
    assert confidence_display_review["writes_checkpoint"] is False
    assert confidence_display_review["generates_text"] is False
    assert confidence_display_review["decodes_text"] is False
    assert confidence_display_review["trains_runtime_model"] is False
    assert confidence_display_review["applies_plasticity"] is False
    assert confidence_display_review["mutates_runtime_state"] is False
    display = confidence_display_review["operator_review"]
    assert display["hash_only_output"] is True
    assert display["display_ranked_count"] == 1
    assert display["selected_candidate_count"] == 1
    assert display["abstained"] is False
    assert display["display_refs"][0]["calibrated_confidence"] == 0.75
    assert "label_hash" in display["display_refs"][0]
    assert "label" not in display["display_refs"][0]
    assert confidence_display_review["promotion_gate"][
        "eligible_for_operator_display_only"
    ] is True
    assert confidence_display_review["promotion_gate"]["eligible_for_language_generation"] is False
    assert confidence_display_review["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert confidence_display_review["promotion_gate"]["eligible_for_replay_memory"] is False
    assert confidence_display_review["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert confidence_display_review["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert confidence_display_review["promotion_gate"]["eligible_for_action"] is False
    assert blocked_internal_stability_review["ready"] is False
    assert blocked_internal_stability_review["requires_operator_approval"] is False
    assert blocked_internal_stability_review["promotion_gate"]["required_evidence"][
        "stability_cycles_sufficient"
    ] is False
    assert internal_stability_review["surface"] == (
        "snn_language_calibrated_dense_label_confidence_internal_stability_review.v1"
    )
    assert internal_stability_review["ready"] is True
    assert internal_stability_review["self_reviewed"] is True
    assert internal_stability_review["requires_operator_approval"] is False
    assert internal_stability_review["advisory"] is True
    assert internal_stability_review["executable"] is False
    assert internal_stability_review["records_ledger_event"] is False
    assert internal_stability_review["runs_replay"] is False
    assert internal_stability_review["runs_calibration_update"] is False
    assert internal_stability_review["writes_checkpoint"] is False
    assert internal_stability_review["generates_text"] is False
    assert internal_stability_review["decodes_text"] is False
    assert internal_stability_review["trains_runtime_model"] is False
    assert internal_stability_review["applies_plasticity"] is False
    assert internal_stability_review["mutates_runtime_state"] is False
    stability = internal_stability_review["internal_stability_review"]
    assert stability["cycle_count"] == 3
    assert stability["matching_cycle_count"] == 3
    assert stability["replay_consistency"] == 1.0
    assert stability["confidence_drift"] <= 0.05
    assert internal_stability_review["promotion_gate"][
        "eligible_for_autonomous_confidence_replay_review"
    ] is True
    assert internal_stability_review["promotion_gate"]["eligible_for_operator_display_only"] is False
    assert internal_stability_review["promotion_gate"]["eligible_for_language_generation"] is False
    assert internal_stability_review["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert internal_stability_review["promotion_gate"]["eligible_for_replay_memory"] is False
    assert internal_stability_review["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert internal_stability_review["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert internal_stability_review["promotion_gate"]["eligible_for_action"] is False
    assert blocked_autonomous_replay_design["ready"] is False
    assert blocked_autonomous_replay_design["requires_operator_approval"] is False
    assert blocked_autonomous_replay_design["promotion_gate"]["required_evidence"][
        "device_evidence_available"
    ] is False
    assert autonomous_replay_design["surface"] == (
        "snn_language_calibrated_dense_label_confidence_autonomous_replay_review_design.v1"
    )
    assert autonomous_replay_design["ready"] is True
    assert autonomous_replay_design["requires_operator_approval"] is False
    assert autonomous_replay_design["advisory"] is True
    assert autonomous_replay_design["executable"] is False
    assert autonomous_replay_design["records_ledger_event"] is False
    assert autonomous_replay_design["runs_replay"] is False
    assert autonomous_replay_design["runs_calibration_update"] is False
    assert autonomous_replay_design["writes_checkpoint"] is False
    assert autonomous_replay_design["generates_text"] is False
    assert autonomous_replay_design["decodes_text"] is False
    assert autonomous_replay_design["trains_runtime_model"] is False
    assert autonomous_replay_design["applies_plasticity"] is False
    assert autonomous_replay_design["mutates_runtime_state"] is False
    replay_design = autonomous_replay_design["autonomous_replay_review_design"]
    assert replay_design["max_replay_cycles"] == 4
    assert replay_design["observed_replay_consistency"] == 1.0
    assert replay_design["observed_confidence_drift"] <= 0.05
    assert replay_design["operator_approval_required"] is False
    assert autonomous_replay_design["promotion_gate"][
        "eligible_for_autonomous_confidence_replay_review_preflight"
    ] is True
    assert autonomous_replay_design["promotion_gate"]["eligible_for_operator_display_only"] is False
    assert autonomous_replay_design["promotion_gate"]["eligible_for_language_generation"] is False
    assert autonomous_replay_design["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert autonomous_replay_design["promotion_gate"]["eligible_for_replay_memory"] is False
    assert autonomous_replay_design["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert autonomous_replay_design["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert autonomous_replay_design["promotion_gate"]["eligible_for_action"] is False
    assert blocked_autonomous_replay_preflight["ready"] is False
    assert blocked_autonomous_replay_preflight["requires_operator_approval"] is False
    assert blocked_autonomous_replay_preflight["promotion_gate"]["required_evidence"][
        "executor_capability_available"
    ] is False
    assert autonomous_replay_preflight["surface"] == (
        "snn_language_calibrated_dense_label_confidence_autonomous_replay_review_preflight.v1"
    )
    assert autonomous_replay_preflight["ready"] is True
    assert autonomous_replay_preflight["requires_operator_approval"] is False
    assert autonomous_replay_preflight["advisory"] is True
    assert autonomous_replay_preflight["executable"] is False
    assert autonomous_replay_preflight["records_ledger_event"] is False
    assert autonomous_replay_preflight["runs_replay"] is False
    assert autonomous_replay_preflight["runs_calibration_update"] is False
    assert autonomous_replay_preflight["writes_checkpoint"] is False
    assert autonomous_replay_preflight["generates_text"] is False
    assert autonomous_replay_preflight["decodes_text"] is False
    assert autonomous_replay_preflight["trains_runtime_model"] is False
    assert autonomous_replay_preflight["applies_plasticity"] is False
    assert autonomous_replay_preflight["mutates_runtime_state"] is False
    replay_preflight = autonomous_replay_preflight[
        "autonomous_replay_review_preflight"
    ]
    assert replay_preflight["max_replay_cycles"] == 4
    assert replay_preflight["operator_approval_required"] is False
    assert replay_preflight["executor_capability_available"] is True
    assert autonomous_replay_preflight["promotion_gate"][
        "eligible_for_autonomous_confidence_replay_review_executor"
    ] is True
    assert autonomous_replay_preflight["promotion_gate"]["eligible_for_operator_display_only"] is False
    assert autonomous_replay_preflight["promotion_gate"]["eligible_for_language_generation"] is False
    assert autonomous_replay_preflight["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert autonomous_replay_preflight["promotion_gate"]["eligible_for_replay_memory"] is False
    assert autonomous_replay_preflight["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert autonomous_replay_preflight["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert autonomous_replay_preflight["promotion_gate"]["eligible_for_action"] is False
    assert blocked_autonomous_replay_executor["ready"] is False
    assert blocked_autonomous_replay_executor["requires_operator_approval"] is False
    assert blocked_autonomous_replay_executor["promotion_gate"]["required_evidence"][
        "preflight_ready"
    ] is False
    assert autonomous_replay_executor["surface"] == (
        "snn_language_calibrated_dense_label_confidence_autonomous_replay_review_executor.v1"
    )
    assert autonomous_replay_executor["ready"] is True
    assert autonomous_replay_executor["requires_operator_approval"] is False
    assert autonomous_replay_executor["advisory"] is False
    assert autonomous_replay_executor["executable"] is True
    assert autonomous_replay_executor["records_ledger_event"] is False
    assert autonomous_replay_executor["runs_replay"] is True
    assert autonomous_replay_executor["runs_live_replay"] is False
    assert autonomous_replay_executor["runs_calibration_update"] is False
    assert autonomous_replay_executor["writes_checkpoint"] is False
    assert autonomous_replay_executor["generates_text"] is False
    assert autonomous_replay_executor["decodes_text"] is False
    assert autonomous_replay_executor["trains_runtime_model"] is False
    assert autonomous_replay_executor["applies_plasticity"] is False
    assert autonomous_replay_executor["mutates_runtime_state"] is False
    replay_review = autonomous_replay_executor["autonomous_replay_review"]
    assert replay_review["cycle_count"] == 4
    assert replay_review["matching_cycle_count"] == 4
    assert replay_review["replay_consistency"] == 1.0
    assert replay_review["confidence_drift"] <= 0.05
    assert replay_review["mutation_allowed"] is False
    assert autonomous_replay_executor["promotion_gate"][
        "eligible_for_autonomous_confidence_recalibration_design"
    ] is True
    assert autonomous_replay_executor["promotion_gate"]["eligible_for_operator_display_only"] is False
    assert autonomous_replay_executor["promotion_gate"]["eligible_for_language_generation"] is False
    assert autonomous_replay_executor["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert autonomous_replay_executor["promotion_gate"]["eligible_for_replay_memory"] is False
    assert autonomous_replay_executor["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert autonomous_replay_executor["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert autonomous_replay_executor["promotion_gate"]["eligible_for_action"] is False
    assert blocked_autonomous_recalibration_design["ready"] is False
    assert blocked_autonomous_recalibration_design[
        "requires_operator_approval"
    ] is False
    assert blocked_autonomous_recalibration_design["promotion_gate"][
        "required_evidence"
    ]["replay_review_executor_ready"] is False
    assert autonomous_recalibration_design["surface"] == (
        "snn_language_calibrated_dense_label_confidence_autonomous_recalibration_design.v1"
    )
    assert autonomous_recalibration_design["ready"] is True
    assert autonomous_recalibration_design["requires_operator_approval"] is False
    assert autonomous_recalibration_design["advisory"] is True
    assert autonomous_recalibration_design["executable"] is False
    assert autonomous_recalibration_design["records_ledger_event"] is False
    assert autonomous_recalibration_design["runs_replay"] is False
    assert autonomous_recalibration_design["runs_recalibration"] is False
    assert autonomous_recalibration_design["runs_calibration_update"] is False
    assert autonomous_recalibration_design["writes_checkpoint"] is False
    assert autonomous_recalibration_design["generates_text"] is False
    assert autonomous_recalibration_design["decodes_text"] is False
    assert autonomous_recalibration_design["trains_runtime_model"] is False
    assert autonomous_recalibration_design["applies_plasticity"] is False
    assert autonomous_recalibration_design["mutates_runtime_state"] is False
    recalibration_design = autonomous_recalibration_design[
        "autonomous_recalibration_design"
    ]
    assert recalibration_design["method"] == "bounded_temperature_scaling"
    assert recalibration_design["operator_approval_required"] is False
    assert recalibration_design["mutation_allowed"] is False
    assert recalibration_design["proposed_temperature_delta"] <= 0.05
    assert autonomous_recalibration_design["promotion_gate"][
        "eligible_for_autonomous_confidence_recalibration_preflight"
    ] is True
    assert autonomous_recalibration_design["promotion_gate"][
        "eligible_for_autonomous_confidence_recalibration_executor"
    ] is False
    assert autonomous_recalibration_design["promotion_gate"]["eligible_for_operator_display_only"] is False
    assert autonomous_recalibration_design["promotion_gate"]["eligible_for_language_generation"] is False
    assert autonomous_recalibration_design["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert autonomous_recalibration_design["promotion_gate"]["eligible_for_replay_memory"] is False
    assert autonomous_recalibration_design["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert autonomous_recalibration_design["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert autonomous_recalibration_design["promotion_gate"]["eligible_for_action"] is False
    assert blocked_autonomous_recalibration_preflight["ready"] is False
    assert blocked_autonomous_recalibration_preflight[
        "requires_operator_approval"
    ] is False
    assert blocked_autonomous_recalibration_preflight["promotion_gate"][
        "required_evidence"
    ]["executor_capability_available"] is False
    assert autonomous_recalibration_preflight["surface"] == (
        "snn_language_calibrated_dense_label_confidence_autonomous_recalibration_preflight.v1"
    )
    assert autonomous_recalibration_preflight["ready"] is True
    assert autonomous_recalibration_preflight["requires_operator_approval"] is False
    assert autonomous_recalibration_preflight["advisory"] is True
    assert autonomous_recalibration_preflight["executable"] is False
    assert autonomous_recalibration_preflight["records_ledger_event"] is False
    assert autonomous_recalibration_preflight["runs_replay"] is False
    assert autonomous_recalibration_preflight["runs_recalibration"] is False
    assert autonomous_recalibration_preflight["runs_calibration_update"] is False
    assert autonomous_recalibration_preflight["writes_checkpoint"] is False
    assert autonomous_recalibration_preflight["generates_text"] is False
    assert autonomous_recalibration_preflight["decodes_text"] is False
    assert autonomous_recalibration_preflight["trains_runtime_model"] is False
    assert autonomous_recalibration_preflight["applies_plasticity"] is False
    assert autonomous_recalibration_preflight["mutates_runtime_state"] is False
    recalibration_preflight = autonomous_recalibration_preflight[
        "autonomous_recalibration_preflight"
    ]
    assert recalibration_preflight["method"] == "bounded_temperature_scaling"
    assert recalibration_preflight["executor_capability_available"] is True
    assert recalibration_preflight["operator_approval_required"] is False
    assert recalibration_preflight["mutation_allowed"] is False
    assert autonomous_recalibration_preflight["promotion_gate"][
        "eligible_for_autonomous_confidence_recalibration_executor"
    ] is True
    assert autonomous_recalibration_preflight["promotion_gate"]["eligible_for_operator_display_only"] is False
    assert autonomous_recalibration_preflight["promotion_gate"]["eligible_for_language_generation"] is False
    assert autonomous_recalibration_preflight["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert autonomous_recalibration_preflight["promotion_gate"]["eligible_for_replay_memory"] is False
    assert autonomous_recalibration_preflight["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert autonomous_recalibration_preflight["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert autonomous_recalibration_preflight["promotion_gate"]["eligible_for_action"] is False
    assert blocked_autonomous_recalibration_executor["accepted"] is False
    assert blocked_autonomous_recalibration_executor[
        "requires_operator_approval"
    ] is False
    assert blocked_autonomous_recalibration_executor["promotion_gate"][
        "required_evidence"
    ]["preflight_ready"] is False
    revision_before_autonomous_recalibration = runtime_state.state_revision
    autonomous_recalibration_executor = (
        ledger.execute_calibrated_dense_label_confidence_autonomous_recalibration(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight=(
                autonomous_recalibration_preflight
            ),
            expected_state_revision=revision_before_autonomous_recalibration,
        )
    )
    assert autonomous_recalibration_executor["surface"] == (
        "snn_language_calibrated_dense_label_confidence_autonomous_recalibration_executor.v1"
    )
    assert autonomous_recalibration_executor["accepted"] is True
    assert autonomous_recalibration_executor["duplicate"] is False
    assert autonomous_recalibration_executor["requires_operator_approval"] is False
    assert autonomous_recalibration_executor["records_ledger_event"] is True
    assert autonomous_recalibration_executor["runs_replay"] is False
    assert autonomous_recalibration_executor["runs_recalibration"] is True
    assert autonomous_recalibration_executor["runs_calibration_update"] is True
    assert autonomous_recalibration_executor["writes_checkpoint"] is False
    assert autonomous_recalibration_executor["generates_text"] is False
    assert autonomous_recalibration_executor["decodes_text"] is False
    assert autonomous_recalibration_executor["trains_runtime_model"] is False
    assert autonomous_recalibration_executor["applies_plasticity"] is False
    assert autonomous_recalibration_executor["mutates_runtime_state"] is True
    assert autonomous_recalibration_executor["before"]["state_revision"] == (
        revision_before_autonomous_recalibration
    )
    assert autonomous_recalibration_executor["after"]["state_revision"] == (
        revision_before_autonomous_recalibration + 1
    )
    assert runtime_state.state_revision == revision_before_autonomous_recalibration + 1
    autonomous_update = autonomous_recalibration_executor[
        "applied_calibration_update"
    ]
    assert autonomous_update["autonomous_recalibration"] is True
    assert autonomous_update["operator_approval_required"] is False
    assert autonomous_update["operator_id"] == "marulho-autonomous-confidence-recalibrator"
    assert autonomous_update["runtime_update_applied"] is True
    assert autonomous_update["weights_persisted"] is False
    assert autonomous_recalibration_executor["ledger_summary"][
        "dense_label_calibration_update_event_count"
    ] == 2
    _assert_dense_label_calibration_update_source_window(
        autonomous_recalibration_executor["source_window"],
        expected_count=1,
    )
    assert autonomous_recalibration_executor["source_window"][
        "current_dense_label_calibration_update_hash"
    ] == applied_update["applied_calibration_update_hash"]
    assert autonomous_recalibration_executor["promotion_gate"]["required_evidence"][
        "dense_label_calibration_update_source_window_bounded"
    ] is True
    assert autonomous_recalibration_executor["promotion_gate"][
        "eligible_for_autonomous_confidence_recalibration_application_review"
    ] is True
    assert autonomous_recalibration_executor["promotion_gate"]["eligible_for_language_generation"] is False
    assert autonomous_recalibration_executor["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert autonomous_recalibration_executor["promotion_gate"]["eligible_for_replay_memory"] is False
    assert autonomous_recalibration_executor["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert autonomous_recalibration_executor["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert autonomous_recalibration_executor["promotion_gate"]["eligible_for_action"] is False
    blocked_autonomous_application_review = (
        ledger.calibrated_dense_label_confidence_autonomous_recalibration_application_review(
            calibrated_dense_label_confidence_autonomous_recalibration_executor={
                **autonomous_recalibration_executor,
                "applied_calibration_update": {
                    **autonomous_update,
                    "applied_calibration_update_hash": "0" * 64,
                },
            },
            expected_state_revision=runtime_state.state_revision,
        )
    )
    autonomous_application_review = (
        ledger.calibrated_dense_label_confidence_autonomous_recalibration_application_review(
            calibrated_dense_label_confidence_autonomous_recalibration_executor=(
                autonomous_recalibration_executor
            ),
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "max_temperature_delta": 0.05,
                "max_confidence_rescale_delta": 0.05,
            },
        )
    )
    assert blocked_autonomous_application_review["ready"] is False
    assert blocked_autonomous_application_review[
        "requires_operator_approval"
    ] is False
    assert blocked_autonomous_application_review["promotion_gate"][
        "required_evidence"
    ]["current_applied_hash_matches"] is False
    assert autonomous_application_review["surface"] == (
        "snn_language_calibrated_dense_label_confidence_autonomous_recalibration_application_review.v1"
    )
    assert autonomous_application_review["ready"] is True
    assert autonomous_application_review["requires_operator_approval"] is False
    assert autonomous_application_review["advisory"] is True
    assert autonomous_application_review["executable"] is False
    assert autonomous_application_review["records_ledger_event"] is False
    assert autonomous_application_review["runs_replay"] is False
    assert autonomous_application_review["runs_recalibration"] is False
    assert autonomous_application_review["runs_calibration_update"] is False
    assert autonomous_application_review["writes_checkpoint"] is False
    assert autonomous_application_review["generates_text"] is False
    assert autonomous_application_review["decodes_text"] is False
    assert autonomous_application_review["trains_runtime_model"] is False
    assert autonomous_application_review["applies_plasticity"] is False
    assert autonomous_application_review["mutates_runtime_state"] is False
    assert autonomous_application_review["applied_calibration_update_hash"] == (
        autonomous_update["applied_calibration_update_hash"]
    )
    assert autonomous_application_review[
        "current_dense_label_calibration_update_hash"
    ] == autonomous_update["applied_calibration_update_hash"]
    _assert_dense_label_calibration_update_source_window(
        autonomous_application_review["source_window"],
        expected_count=2,
    )
    assert autonomous_application_review["source_window"][
        "current_dense_label_calibration_update_hash"
    ] == autonomous_update["applied_calibration_update_hash"]
    assert autonomous_application_review["promotion_gate"]["required_evidence"][
        "dense_label_calibration_update_source_window_bounded"
    ] is True
    reviewed_autonomous_update = autonomous_application_review[
        "autonomous_recalibration_application_review"
    ]
    assert reviewed_autonomous_update["runtime_update_applied"] is True
    assert reviewed_autonomous_update["weights_persisted"] is False
    assert reviewed_autonomous_update["operator_approval_required"] is False
    assert autonomous_application_review["promotion_gate"][
        "eligible_for_autonomous_post_calibration_observation_window"
    ] is True
    assert autonomous_application_review["promotion_gate"]["eligible_for_language_generation"] is False
    assert autonomous_application_review["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert autonomous_application_review["promotion_gate"]["eligible_for_replay_memory"] is False
    assert autonomous_application_review["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert autonomous_application_review["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert autonomous_application_review["promotion_gate"]["eligible_for_action"] is False
    blocked_autonomous_observation = (
        ledger.calibrated_dense_label_confidence_autonomous_post_calibration_observation_window(
            calibrated_dense_label_confidence_autonomous_recalibration_application_review=(
                autonomous_application_review
            ),
            observation_evidence={"samples": observation_samples[:1]},
            expected_state_revision=runtime_state.state_revision,
            window_policy={"min_samples": 3},
        )
    )
    autonomous_observation = (
        ledger.calibrated_dense_label_confidence_autonomous_post_calibration_observation_window(
            calibrated_dense_label_confidence_autonomous_recalibration_application_review=(
                autonomous_application_review
            ),
            observation_evidence={"samples": observation_samples},
            expected_state_revision=runtime_state.state_revision,
            window_policy={
                "min_samples": 3,
                "max_expected_calibration_error": 0.2,
                "max_confidence_drift": 0.7,
            },
        )
    )
    assert blocked_autonomous_observation["ready"] is False
    assert blocked_autonomous_observation["requires_operator_approval"] is False
    assert blocked_autonomous_observation["promotion_gate"]["required_evidence"][
        "sample_count_sufficient"
    ] is False
    assert autonomous_observation["surface"] == (
        "snn_language_calibrated_dense_label_confidence_autonomous_post_calibration_observation_window.v1"
    )
    assert autonomous_observation["ready"] is True
    assert autonomous_observation["requires_operator_approval"] is False
    assert autonomous_observation["advisory"] is True
    assert autonomous_observation["executable"] is False
    assert autonomous_observation["records_ledger_event"] is False
    assert autonomous_observation["runs_replay"] is False
    assert autonomous_observation["runs_recalibration"] is False
    assert autonomous_observation["runs_calibration_update"] is False
    assert autonomous_observation["writes_checkpoint"] is False
    assert autonomous_observation["generates_text"] is False
    assert autonomous_observation["decodes_text"] is False
    assert autonomous_observation["trains_runtime_model"] is False
    assert autonomous_observation["applies_plasticity"] is False
    assert autonomous_observation["mutates_runtime_state"] is False
    assert autonomous_observation["sample_count"] == 3
    assert autonomous_observation["metrics"]["expected_calibration_error"] <= 0.2
    assert autonomous_observation["metrics"]["mean_confidence_drift"] <= 0.7
    assert autonomous_observation["promotion_gate"][
        "eligible_for_autonomous_post_calibration_stability_review"
    ] is True
    assert autonomous_observation["promotion_gate"]["eligible_for_language_generation"] is False
    assert autonomous_observation["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert autonomous_observation["promotion_gate"]["eligible_for_replay_memory"] is False
    assert autonomous_observation["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert autonomous_observation["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert autonomous_observation["promotion_gate"]["eligible_for_action"] is False
    blocked_autonomous_stability_review = (
        ledger.calibrated_dense_label_confidence_autonomous_post_calibration_stability_review(
            calibrated_dense_label_confidence_autonomous_post_calibration_observation_window=(
                blocked_autonomous_observation
            ),
            expected_state_revision=runtime_state.state_revision,
            stability_policy={"min_samples": 3},
        )
    )
    autonomous_stability_review = (
        ledger.calibrated_dense_label_confidence_autonomous_post_calibration_stability_review(
            calibrated_dense_label_confidence_autonomous_post_calibration_observation_window=(
                autonomous_observation
            ),
            expected_state_revision=runtime_state.state_revision,
            stability_policy={
                "min_samples": 3,
                "max_expected_calibration_error": 0.2,
                "max_confidence_drift": 0.7,
            },
        )
    )
    assert blocked_autonomous_stability_review["ready"] is False
    assert blocked_autonomous_stability_review[
        "requires_operator_approval"
    ] is False
    assert blocked_autonomous_stability_review["promotion_gate"][
        "required_evidence"
    ]["observation_ready"] is False
    assert autonomous_stability_review["surface"] == (
        "snn_language_calibrated_dense_label_confidence_autonomous_post_calibration_stability_review.v1"
    )
    assert autonomous_stability_review["ready"] is True
    assert autonomous_stability_review["requires_operator_approval"] is False
    assert autonomous_stability_review["advisory"] is True
    assert autonomous_stability_review["executable"] is False
    assert autonomous_stability_review["records_ledger_event"] is False
    assert autonomous_stability_review["runs_replay"] is False
    assert autonomous_stability_review["runs_recalibration"] is False
    assert autonomous_stability_review["runs_calibration_update"] is False
    assert autonomous_stability_review["writes_checkpoint"] is False
    assert autonomous_stability_review["generates_text"] is False
    assert autonomous_stability_review["decodes_text"] is False
    assert autonomous_stability_review["trains_runtime_model"] is False
    assert autonomous_stability_review["applies_plasticity"] is False
    assert autonomous_stability_review["mutates_runtime_state"] is False
    stability_review = autonomous_stability_review[
        "autonomous_post_calibration_stability_review"
    ]
    assert stability_review["sample_count"] == 3
    assert stability_review["operator_approval_required"] is False
    assert stability_review["mutation_allowed"] is False
    assert autonomous_stability_review["promotion_gate"][
        "eligible_for_autonomous_calibrated_confidence_use_design"
    ] is True
    assert autonomous_stability_review["promotion_gate"]["eligible_for_language_generation"] is False
    assert autonomous_stability_review["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert autonomous_stability_review["promotion_gate"]["eligible_for_replay_memory"] is False
    assert autonomous_stability_review["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert autonomous_stability_review["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert autonomous_stability_review["promotion_gate"]["eligible_for_action"] is False
    blocked_autonomous_confidence_use_design = (
        ledger.calibrated_dense_label_confidence_autonomous_use_design(
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review=(
                blocked_autonomous_stability_review
            ),
            confidence_use_policy={"use_mode": "generate_text"},
            device_evidence={},
        )
    )
    autonomous_confidence_use_design = (
        ledger.calibrated_dense_label_confidence_autonomous_use_design(
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review=(
                autonomous_stability_review
            ),
            confidence_use_policy={
                "use_mode": "threshold_and_abstain",
                "min_confidence_threshold": 0.6,
                "max_candidates": 4,
            },
            device_evidence={"device": "cpu", "source": "unit"},
        )
    )
    assert blocked_autonomous_confidence_use_design["ready"] is False
    assert blocked_autonomous_confidence_use_design[
        "requires_operator_approval"
    ] is False
    assert blocked_autonomous_confidence_use_design["promotion_gate"][
        "required_evidence"
    ]["stability_review_ready"] is False
    assert autonomous_confidence_use_design["surface"] == (
        "snn_language_calibrated_dense_label_confidence_autonomous_use_design.v1"
    )
    assert autonomous_confidence_use_design["ready"] is True
    assert autonomous_confidence_use_design["requires_operator_approval"] is False
    assert autonomous_confidence_use_design["advisory"] is True
    assert autonomous_confidence_use_design["executable"] is False
    assert autonomous_confidence_use_design["records_ledger_event"] is False
    assert autonomous_confidence_use_design["runs_replay"] is False
    assert autonomous_confidence_use_design["runs_recalibration"] is False
    assert autonomous_confidence_use_design["runs_calibration_update"] is False
    assert autonomous_confidence_use_design["writes_checkpoint"] is False
    assert autonomous_confidence_use_design["generates_text"] is False
    assert autonomous_confidence_use_design["decodes_text"] is False
    assert autonomous_confidence_use_design["trains_runtime_model"] is False
    assert autonomous_confidence_use_design["applies_plasticity"] is False
    assert autonomous_confidence_use_design["mutates_runtime_state"] is False
    use_design = autonomous_confidence_use_design[
        "autonomous_confidence_use_design"
    ]
    assert use_design["use_mode"] == "threshold_and_abstain"
    assert use_design["operator_approval_required"] is False
    assert "generate_language" in use_design["disallowed_operations"]
    assert autonomous_confidence_use_design["promotion_gate"][
        "eligible_for_autonomous_calibrated_confidence_use_preflight"
    ] is True
    assert autonomous_confidence_use_design["promotion_gate"]["eligible_for_language_generation"] is False
    assert autonomous_confidence_use_design["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert autonomous_confidence_use_design["promotion_gate"]["eligible_for_replay_memory"] is False
    assert autonomous_confidence_use_design["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert autonomous_confidence_use_design["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert autonomous_confidence_use_design["promotion_gate"]["eligible_for_action"] is False
    blocked_autonomous_confidence_use_preflight = (
        ledger.calibrated_dense_label_confidence_autonomous_use_preflight(
            calibrated_dense_label_confidence_autonomous_use_design=(
                autonomous_confidence_use_design
            ),
            expected_state_revision=runtime_state.state_revision,
            candidate_evidence={
                "candidates": [
                    {
                        "dense_label_candidate_evidence_hash": _sha256_json(
                            ["candidate", "blocked-autonomous"]
                        ),
                        "label_hash": _sha256_json(["label", "blocked-autonomous"]),
                        "calibrated_confidence": 0.4,
                        "pre_calibration_confidence": 0.8,
                    }
                ]
            },
            executor_capabilities={
                "autonomous_calibrated_confidence_use_executor": False
            },
        )
    )
    autonomous_confidence_use_preflight = (
        ledger.calibrated_dense_label_confidence_autonomous_use_preflight(
            calibrated_dense_label_confidence_autonomous_use_design=(
                autonomous_confidence_use_design
            ),
            expected_state_revision=runtime_state.state_revision,
            candidate_evidence={
                "candidates": [
                    {
                        "dense_label_candidate_evidence_hash": selected_hash,
                        "label_hash": _sha256_json(["label", "ready"]),
                        "calibrated_confidence": 0.75,
                        "pre_calibration_confidence": 0.8,
                    }
                ]
            },
            executor_capabilities={
                "autonomous_calibrated_confidence_use_executor": True
            },
        )
    )
    assert blocked_autonomous_confidence_use_preflight["ready"] is False
    assert blocked_autonomous_confidence_use_preflight[
        "requires_operator_approval"
    ] is False
    assert blocked_autonomous_confidence_use_preflight["promotion_gate"][
        "required_evidence"
    ]["executor_capability_available"] is False
    assert autonomous_confidence_use_preflight["surface"] == (
        "snn_language_calibrated_dense_label_confidence_autonomous_use_preflight.v1"
    )
    assert autonomous_confidence_use_preflight["ready"] is True
    assert autonomous_confidence_use_preflight[
        "requires_operator_approval"
    ] is False
    assert autonomous_confidence_use_preflight["advisory"] is True
    assert autonomous_confidence_use_preflight["executable"] is False
    assert autonomous_confidence_use_preflight["records_ledger_event"] is False
    assert autonomous_confidence_use_preflight["runs_replay"] is False
    assert autonomous_confidence_use_preflight["runs_recalibration"] is False
    assert autonomous_confidence_use_preflight["runs_calibration_update"] is False
    assert autonomous_confidence_use_preflight["writes_checkpoint"] is False
    assert autonomous_confidence_use_preflight["generates_text"] is False
    assert autonomous_confidence_use_preflight["decodes_text"] is False
    assert autonomous_confidence_use_preflight["trains_runtime_model"] is False
    assert autonomous_confidence_use_preflight["applies_plasticity"] is False
    assert autonomous_confidence_use_preflight["mutates_runtime_state"] is False
    assert autonomous_confidence_use_preflight["candidate_preflight"][
        "candidate_count"
    ] == 1
    assert autonomous_confidence_use_preflight["candidate_preflight"][
        "passing_candidate_count"
    ] == 1
    assert autonomous_confidence_use_preflight["promotion_gate"][
        "eligible_for_autonomous_calibrated_confidence_use_executor"
    ] is True
    assert autonomous_confidence_use_preflight["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert autonomous_confidence_use_preflight["promotion_gate"][
        "eligible_for_dense_readout_training"
    ] is False
    assert autonomous_confidence_use_preflight["promotion_gate"][
        "eligible_for_replay_memory"
    ] is False
    assert autonomous_confidence_use_preflight["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert autonomous_confidence_use_preflight["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert autonomous_confidence_use_preflight["promotion_gate"][
        "eligible_for_action"
    ] is False


def test_readout_ledger_emission_review_history_is_read_only_narrow_display_surface() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_readout_emission_review(
        readout_emission=_ready_emission(),
        expected_state_revision=0,
        operator_id="operator-emission",
        confirmation=True,
    )
    runtime_state.mark_clean()
    before_revision = runtime_state.state_revision

    history = ledger.emission_review_history(limit=1)
    empty = ledger.emission_review_history(limit=0)

    assert runtime_state.state_revision == before_revision
    assert runtime_state.dirty_state is False
    assert history["surface"] == "snn_language_readout_emission_review_history.v1"
    assert history["executable"] is False
    assert history["mutates_runtime_state"] is False
    assert history["records_ledger_event"] is False
    assert history["generates_text"] is False
    assert history["decodes_text"] is False
    assert history["exposes_reviewed_bounded_text"] is True
    assert history["summary"]["emission_review_event_count"] == 1
    assert history["summary"]["returned_emission_review_event_count"] == 1
    assert history["source_window"]["surface"] == (
        "bounded_snn_emission_review_history_source_window.v1"
    )
    assert history["source_window"][
        "policy"
    ] == SNN_EMISSION_REVIEW_HISTORY_SOURCE_WINDOW_POLICY
    assert history["source_window"]["source_window_count"] == 1
    assert history["source_window"]["global_candidate_scan"] is False
    assert history["source_window"]["global_score_scan"] is False
    assert history["source_window"]["raw_text_payload_loaded"] is True
    assert history["source_window"]["language_reasoning"] is False
    assert history["source_window"]["runs_live_tick"] is False
    assert history["source_window"]["runs_every_token"] is False
    assert history["source_window"]["archival_storage_device"] == "cpu"
    assert history["source_window"]["lookup_device"] == "cpu"
    assert history["source_window"]["gpu_used"] is False
    assert history["summary"]["source_window"] == history["source_window"]
    assert len(history["emission_review_events"]) == 1
    reviewed = history["emission_review_events"][0]
    assert reviewed["text"] == "memory pressure"
    assert reviewed["labels"] == ["memory pressure"]
    assert reviewed["eligible_for_replay_memory"] is False
    assert reviewed["eligible_for_live_replay"] is False
    assert reviewed["eligible_for_plasticity_application"] is False
    assert reviewed["eligible_for_fact_promotion"] is False
    assert reviewed["eligible_for_action"] is False
    assert history["promotion_gate"]["eligible_for_operator_display_history_inspection"] is True
    assert history["promotion_gate"]["eligible_for_replay_memory"] is False
    assert history["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert history["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert history["promotion_gate"]["eligible_for_action"] is False
    assert history["promotion_gate"]["required_evidence"][
        "source_window_bounded"
    ] is True
    assert "events" not in history
    assert "rollout_events" not in history
    assert "prediction_report" not in reviewed
    assert "transition_memory_evaluation" not in reviewed
    assert empty["summary"]["returned_emission_review_event_count"] == 0
    assert empty["emission_review_events"] == []
    assert empty["promotion_gate"]["eligible_for_operator_display_history_inspection"] is False


def test_readout_ledger_emission_review_history_uses_review_source_window_only() -> None:
    class CountedRows:
        def __init__(self, field: str, count: int) -> None:
            self.field = field
            self.count = count
            self.iterated = 0

        def __iter__(self):
            for index in range(self.count):
                self.iterated += 1
                yield {
                    "field": self.field,
                    "ordinal": index,
                    "emission_review_hash": f"{self.field}:review:{index}",
                    "emission_hash": f"{self.field}:emission:{index}",
                    "trajectory_hash": f"{self.field}:trajectory:{index}",
                    "persistent_transition_weights_hash": (
                        f"{self.field}:weights:{index}"
                    ),
                    "text": f"{self.field}:text:{index}",
                    "labels": [f"{self.field}:label:{index}"],
                }

        def __len__(self) -> int:
            return self.count

    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_limit = 8
    source_count = 256
    ledger_state: dict[str, object] = {
        field: CountedRows(field, source_count)
        for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }
    ledger_state["total_emission_review_count"] = source_count
    ledger_state["last_emission_reviewed_at"] = "2026-06-19T00:00:00+00:00"
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
        limit=ledger_limit,
    )

    history = ledger.emission_review_history(limit=2)

    assert history["source_window"]["surface"] == (
        "bounded_snn_emission_review_history_source_window.v1"
    )
    assert history["source_window"][
        "policy"
    ] == SNN_EMISSION_REVIEW_HISTORY_SOURCE_WINDOW_POLICY
    assert history["source_window"]["source_window_limit"] == ledger_limit
    assert history["source_window"]["source_window_count"] == ledger_limit
    assert history["source_window"]["source_record_count"] == source_count
    assert history["source_window"]["source_payload_truncated"] is True
    assert history["source_window"]["source_truncated_count"] == (
        source_count - ledger_limit
    )
    assert history["summary"]["emission_review_event_count"] == ledger_limit
    assert history["summary"]["returned_emission_review_event_count"] == 2
    assert history["summary"]["total_emission_review_count"] == source_count
    assert history["summary"]["last_emission_reviewed_at"] == (
        "2026-06-19T00:00:00+00:00"
    )
    history_hashes = [
        item["emission_review_hash"]
        for item in history["emission_review_events"]
    ]
    assert history_hashes == [
        "emission_review_events:review:0",
        "emission_review_events:review:1",
    ]
    assert history["promotion_gate"]["required_evidence"][
        "source_window_bounded"
    ] is True
    for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS:
        source = ledger_state[field]
        assert isinstance(source, CountedRows)
        if field == "emission_review_events":
            assert source.iterated == ledger_limit
        else:
            assert source.iterated == 0


def test_readout_ledger_autonomous_confidence_use_preflight_audits_candidates_without_execution() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    candidate_hash = _sha256_json(["candidate", "autonomous-preflight"])
    design_hash = _sha256_json(["design", "autonomous-preflight"])
    design = {
        "surface": "snn_language_calibrated_dense_label_confidence_autonomous_use_design.v1",
        "ready": True,
        "requires_operator_approval": False,
        "advisory": True,
        "executable": False,
        "records_ledger_event": False,
        "runs_replay": False,
        "runs_recalibration": False,
        "runs_calibration_update": False,
        "writes_checkpoint": False,
        "generates_text": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "autonomous_confidence_use_design_hash": design_hash,
        "stability_review_hash": _sha256_json(["stability", "autonomous-preflight"]),
        "observation_hash": _sha256_json(["observation", "autonomous-preflight"]),
        "application_review_hash": _sha256_json(
            ["application-review", "autonomous-preflight"]
        ),
        "applied_calibration_update_hash": _sha256_json(
            ["applied", "autonomous-preflight"]
        ),
        "autonomous_confidence_use_design": {
            "use_mode": "threshold_and_abstain",
            "min_confidence_threshold": 0.6,
            "max_candidates": 4,
            "operator_approval_required": False,
            "device_evidence": {"device": "cpu", "source": "unit"},
        },
        "promotion_gate": {
            "eligible_for_autonomous_calibrated_confidence_use_preflight": True
        },
    }

    blocked = ledger.calibrated_dense_label_confidence_autonomous_use_preflight(
        calibrated_dense_label_confidence_autonomous_use_design=design,
        expected_state_revision=runtime_state.state_revision,
        candidate_evidence={
            "candidates": [
                {
                    "dense_label_candidate_evidence_hash": candidate_hash,
                    "label_hash": _sha256_json(["label", "autonomous-preflight"]),
                    "calibrated_confidence": 0.4,
                    "pre_calibration_confidence": 0.8,
                }
            ]
        },
        executor_capabilities={"autonomous_calibrated_confidence_use_executor": False},
    )
    ready = ledger.calibrated_dense_label_confidence_autonomous_use_preflight(
        calibrated_dense_label_confidence_autonomous_use_design=design,
        expected_state_revision=runtime_state.state_revision,
        candidate_evidence={
            "candidates": [
                {
                    "dense_label_candidate_evidence_hash": candidate_hash,
                    "label_hash": _sha256_json(["label", "autonomous-preflight"]),
                    "calibrated_confidence": 0.75,
                    "pre_calibration_confidence": 0.8,
                }
            ]
        },
        executor_capabilities={"autonomous_calibrated_confidence_use_executor": True},
    )

    assert blocked["ready"] is False
    assert blocked["promotion_gate"]["required_evidence"][
        "executor_capability_available"
    ] is False
    assert ready["surface"] == (
        "snn_language_calibrated_dense_label_confidence_autonomous_use_preflight.v1"
    )
    assert ready["ready"] is True
    assert ready["requires_operator_approval"] is False
    assert ready["advisory"] is True
    assert ready["executable"] is False
    assert ready["records_ledger_event"] is False
    assert ready["runs_replay"] is False
    assert ready["runs_recalibration"] is False
    assert ready["runs_calibration_update"] is False
    assert ready["writes_checkpoint"] is False
    assert ready["generates_text"] is False
    assert ready["decodes_text"] is False
    assert ready["trains_runtime_model"] is False
    assert ready["applies_plasticity"] is False
    assert ready["mutates_runtime_state"] is False
    assert ready["candidate_preflight"]["candidate_count"] == 1
    assert ready["candidate_preflight"]["passing_candidate_count"] == 1
    assert ready["promotion_gate"][
        "eligible_for_autonomous_calibrated_confidence_use_executor"
    ] is True
    assert ready["promotion_gate"]["eligible_for_language_generation"] is False
    assert ready["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert ready["promotion_gate"]["eligible_for_replay_memory"] is False
    assert ready["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert ready["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert ready["promotion_gate"]["eligible_for_action"] is False

    before_revision = runtime_state.state_revision
    execution = ledger.execute_autonomous_calibrated_dense_label_confidence_use(
        calibrated_dense_label_confidence_autonomous_use_preflight=ready,
        expected_state_revision=before_revision,
        candidate_evidence={
            "candidates": [
                {
                    "dense_label_candidate_evidence_hash": candidate_hash,
                    "label_hash": _sha256_json(["label", "autonomous-preflight"]),
                    "calibrated_confidence": 0.75,
                    "pre_calibration_confidence": 0.8,
                }
            ]
        },
        execution_policy={"max_selected_candidates": 1},
    )
    stale_execution = ledger.execute_autonomous_calibrated_dense_label_confidence_use(
        calibrated_dense_label_confidence_autonomous_use_preflight=ready,
        expected_state_revision=runtime_state.state_revision,
        candidate_evidence={
            "candidates": [
                {
                    "dense_label_candidate_evidence_hash": candidate_hash,
                    "label_hash": _sha256_json(["label", "autonomous-preflight"]),
                    "calibrated_confidence": 0.75,
                    "pre_calibration_confidence": 0.8,
                }
            ]
        },
        execution_policy={"max_selected_candidates": 1},
    )
    blocked_execution = ledger.execute_autonomous_calibrated_dense_label_confidence_use(
        calibrated_dense_label_confidence_autonomous_use_preflight=ready,
        expected_state_revision=runtime_state.state_revision,
        candidate_evidence={
            "candidates": [
                {
                    "dense_label_candidate_evidence_hash": candidate_hash,
                    "label_hash": _sha256_json(["label", "autonomous-preflight"]),
                    "label": "text is not allowed",
                    "calibrated_confidence": 0.75,
                    "pre_calibration_confidence": 0.8,
                }
            ]
        },
        execution_policy={"max_selected_candidates": 1},
    )

    assert execution["surface"] == (
        "snn_language_calibrated_dense_label_confidence_autonomous_use_executor.v1"
    )
    assert execution["accepted"] is True
    assert execution["ready"] is True
    assert execution["requires_operator_approval"] is False
    assert execution["records_ledger_event"] is True
    assert execution["mutates_runtime_state"] is True
    assert execution["after"]["state_revision"] == before_revision + 1
    assert execution["runs_replay"] is False
    assert execution["runs_recalibration"] is False
    assert execution["runs_calibration_update"] is False
    assert execution["writes_checkpoint"] is False
    assert execution["generates_text"] is False
    assert execution["decodes_text"] is False
    assert execution["trains_runtime_model"] is False
    assert execution["applies_plasticity"] is False
    event = execution["autonomous_confidence_use_event"]
    assert event["operator_approval_required"] is False
    assert event["output_is_label_hash_only"] is True
    assert event["selected_candidate_count"] == 1
    assert event["selected_candidate_refs"][0][
        "dense_label_candidate_evidence_hash"
    ] == candidate_hash
    assert execution["ledger_summary"]["total_autonomous_confidence_use_count"] == 1
    _assert_autonomous_confidence_use_source_window(
        execution["source_window"],
        expected_count=0,
    )
    assert execution["promotion_gate"]["required_evidence"][
        "autonomous_confidence_use_source_window_bounded"
    ] is True
    assert execution["promotion_gate"][
        "eligible_for_autonomous_confidence_use_event_review"
    ] is True
    assert execution["promotion_gate"]["eligible_for_language_generation"] is False
    assert execution["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert execution["promotion_gate"]["eligible_for_replay_memory"] is False
    assert execution["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert execution["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert execution["promotion_gate"]["eligible_for_action"] is False
    assert stale_execution["accepted"] is False
    assert stale_execution["records_ledger_event"] is False
    assert stale_execution["mutates_runtime_state"] is False
    assert stale_execution["promotion_gate"]["required_evidence"][
        "preflight_revision_current"
    ] is False
    assert blocked_execution["accepted"] is False
    assert blocked_execution["records_ledger_event"] is False
    assert blocked_execution["promotion_gate"]["required_evidence"][
        "text_payload_absent"
    ] is False
    blocked_review = (
        ledger.autonomous_calibrated_dense_label_confidence_use_event_review(
            calibrated_dense_label_confidence_autonomous_use_executor={
                **execution,
                "autonomous_confidence_use_event": {
                    **execution["autonomous_confidence_use_event"],
                    "autonomous_confidence_use_event_hash": "0" * 64,
                },
            },
            expected_state_revision=runtime_state.state_revision,
        )
    )
    review = ledger.autonomous_calibrated_dense_label_confidence_use_event_review(
        calibrated_dense_label_confidence_autonomous_use_executor=execution,
        expected_state_revision=runtime_state.state_revision,
        review_policy={"min_selected_candidates": 1, "max_selected_candidates": 2},
    )

    assert blocked_review["ready"] is False
    assert blocked_review["requires_operator_approval"] is False
    assert blocked_review["promotion_gate"]["required_evidence"][
        "event_recorded_in_ledger"
    ] is False
    assert review["surface"] == (
        "snn_language_calibrated_dense_label_confidence_autonomous_use_event_review.v1"
    )
    assert review["ready"] is True
    assert review["requires_operator_approval"] is False
    assert review["advisory"] is True
    assert review["executable"] is False
    assert review["records_ledger_event"] is False
    assert review["mutates_runtime_state"] is False
    assert review["runs_replay"] is False
    assert review["runs_recalibration"] is False
    assert review["runs_calibration_update"] is False
    assert review["writes_checkpoint"] is False
    assert review["generates_text"] is False
    assert review["decodes_text"] is False
    assert review["trains_runtime_model"] is False
    assert review["applies_plasticity"] is False
    event_review = review["autonomous_confidence_use_event_review"]
    assert event_review["event_recorded_in_ledger"] is True
    assert event_review["selected_candidate_count"] == 1
    assert event_review["selected_candidate_hashes"] == [candidate_hash]
    _assert_autonomous_confidence_use_source_window(
        review["source_window"],
        expected_count=1,
    )
    assert review["promotion_gate"]["required_evidence"][
        "autonomous_confidence_use_source_window_bounded"
    ] is True
    assert event_review["output_is_label_hash_only"] is True
    assert event_review["operator_approval_required"] is False
    assert event_review["mutation_allowed"] is False
    assert review["promotion_gate"][
        "eligible_for_autonomous_hash_readout_binding_design"
    ] is True
    assert review["promotion_gate"]["eligible_for_language_generation"] is False
    assert review["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert review["promotion_gate"]["eligible_for_replay_memory"] is False
    assert review["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert review["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert review["promotion_gate"]["eligible_for_action"] is False
    blocked_binding_design = ledger.autonomous_hash_readout_binding_design(
        calibrated_dense_label_confidence_autonomous_use_event_review=review,
        readout_vocabulary_slots=[],
        binding_policy={"max_bindings": 1},
        device_evidence={"device": "cpu", "source": "unit"},
    )
    binding_design = ledger.autonomous_hash_readout_binding_design(
        calibrated_dense_label_confidence_autonomous_use_event_review=review,
        readout_vocabulary_slots=[
            {
                "label": "autonomous concept",
                "pressure_band": "medium",
                "grounded": True,
                "slot_id": "slot-autonomous-concept",
            }
        ],
        binding_policy={"max_bindings": 1},
        device_evidence={"device": "cpu", "source": "unit"},
    )

    assert blocked_binding_design["ready"] is False
    assert blocked_binding_design["promotion_gate"]["required_evidence"][
        "readout_slots_available"
    ] is False
    assert binding_design["surface"] == (
        "snn_language_autonomous_hash_readout_binding_design.v1"
    )
    assert binding_design["ready"] is True
    assert binding_design["requires_operator_approval"] is False
    assert binding_design["advisory"] is True
    assert binding_design["executable"] is False
    assert binding_design["records_ledger_event"] is False
    assert binding_design["mutates_runtime_state"] is False
    assert binding_design["runs_replay"] is False
    assert binding_design["runs_calibration_update"] is False
    assert binding_design["writes_checkpoint"] is False
    assert binding_design["generates_text"] is False
    assert binding_design["decodes_text"] is False
    assert binding_design["trains_runtime_model"] is False
    assert binding_design["applies_plasticity"] is False
    binding_body = binding_design["autonomous_hash_readout_binding_design"]
    assert binding_body["binding_count"] == 1
    assert binding_body["selected_candidate_hashes"] == [candidate_hash]
    assert binding_body["output_is_hash_binding_only"] is True
    assert binding_body["operator_approval_required"] is False
    assert binding_body["execution_allowed"] is False
    assert binding_body["bindings"][0][
        "dense_label_candidate_evidence_hash"
    ] == candidate_hash
    assert binding_design["promotion_gate"][
        "eligible_for_autonomous_hash_readout_binding_preflight"
    ] is True
    assert binding_design["promotion_gate"]["eligible_for_language_generation"] is False
    assert binding_design["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert binding_design["promotion_gate"]["eligible_for_replay_memory"] is False
    assert binding_design["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert binding_design["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert binding_design["promotion_gate"]["eligible_for_action"] is False
    blocked_binding_preflight = ledger.autonomous_hash_readout_binding_preflight(
        autonomous_hash_readout_binding_design=binding_design,
        expected_state_revision=runtime_state.state_revision,
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"autonomous_hash_readout_binding_executor": False},
    )
    binding_preflight = ledger.autonomous_hash_readout_binding_preflight(
        autonomous_hash_readout_binding_design=binding_design,
        expected_state_revision=runtime_state.state_revision,
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"autonomous_hash_readout_binding_executor": True},
    )

    assert blocked_binding_preflight["ready"] is False
    assert blocked_binding_preflight["requires_operator_approval"] is False
    assert blocked_binding_preflight["promotion_gate"]["required_evidence"][
        "executor_capability_available"
    ] is False
    assert binding_preflight["surface"] == (
        "snn_language_autonomous_hash_readout_binding_preflight.v1"
    )
    assert binding_preflight["ready"] is True
    assert binding_preflight["requires_operator_approval"] is False
    assert binding_preflight["advisory"] is True
    assert binding_preflight["executable"] is False
    assert binding_preflight["records_ledger_event"] is False
    assert binding_preflight["mutates_runtime_state"] is False
    assert binding_preflight["runs_replay"] is False
    assert binding_preflight["runs_calibration_update"] is False
    assert binding_preflight["writes_checkpoint"] is False
    assert binding_preflight["generates_text"] is False
    assert binding_preflight["decodes_text"] is False
    assert binding_preflight["trains_runtime_model"] is False
    assert binding_preflight["applies_plasticity"] is False
    assert binding_preflight["binding_preflight"]["binding_count"] == 1
    assert binding_preflight["binding_preflight"][
        "binding_candidate_hashes"
    ] == [candidate_hash]
    assert binding_preflight["binding_preflight"][
        "output_is_hash_binding_only"
    ] is True
    assert binding_preflight["device_preflight"][
        "executor_capability_available"
    ] is True
    assert binding_preflight["promotion_gate"][
        "eligible_for_autonomous_hash_readout_binding_executor"
    ] is True
    assert binding_preflight["promotion_gate"]["eligible_for_language_generation"] is False
    assert binding_preflight["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert binding_preflight["promotion_gate"]["eligible_for_replay_memory"] is False
    assert binding_preflight["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert binding_preflight["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert binding_preflight["promotion_gate"]["eligible_for_action"] is False
    binding_before_revision = runtime_state.state_revision
    binding_execution = ledger.execute_autonomous_hash_readout_binding(
        autonomous_hash_readout_binding_preflight=binding_preflight,
        expected_state_revision=binding_before_revision,
        execution_policy={"max_commit_bindings": 1},
    )
    stale_binding_execution = ledger.execute_autonomous_hash_readout_binding(
        autonomous_hash_readout_binding_preflight=binding_preflight,
        expected_state_revision=runtime_state.state_revision,
        execution_policy={"max_commit_bindings": 1},
    )

    assert binding_execution["surface"] == (
        "snn_language_autonomous_hash_readout_binding_executor.v1"
    )
    assert binding_execution["accepted"] is True
    assert binding_execution["ready"] is True
    assert binding_execution["requires_operator_approval"] is False
    assert binding_execution["records_ledger_event"] is True
    assert binding_execution["mutates_runtime_state"] is True
    assert binding_execution["after"]["state_revision"] == binding_before_revision + 1
    assert binding_execution["runs_replay"] is False
    assert binding_execution["runs_calibration_update"] is False
    assert binding_execution["writes_checkpoint"] is False
    assert binding_execution["generates_text"] is False
    assert binding_execution["decodes_text"] is False
    assert binding_execution["trains_runtime_model"] is False
    assert binding_execution["applies_plasticity"] is False
    binding_event = binding_execution["autonomous_hash_readout_binding_event"]
    assert binding_event["operator_approval_required"] is False
    assert binding_event["output_is_hash_binding_only"] is True
    assert binding_event["binding_count"] == 1
    assert binding_event["bindings"][0][
        "dense_label_candidate_evidence_hash"
    ] == candidate_hash
    assert binding_execution["ledger_summary"][
        "total_autonomous_hash_readout_binding_count"
    ] == 1
    _assert_record_family_source_window(
        binding_execution["source_window"],
        field="autonomous_hash_readout_binding_events",
        expected_count=0,
    )
    assert binding_execution["promotion_gate"][
        "eligible_for_autonomous_hash_readout_binding_event_review"
    ] is True
    assert binding_execution["promotion_gate"]["eligible_for_language_generation"] is False
    assert binding_execution["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert binding_execution["promotion_gate"]["eligible_for_replay_memory"] is False
    assert binding_execution["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert binding_execution["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert binding_execution["promotion_gate"]["eligible_for_action"] is False
    assert stale_binding_execution["accepted"] is False
    assert stale_binding_execution["records_ledger_event"] is False
    assert stale_binding_execution["mutates_runtime_state"] is False
    assert stale_binding_execution["promotion_gate"]["required_evidence"][
        "preflight_revision_current"
    ] is False
    blocked_binding_event_review = (
        ledger.autonomous_hash_readout_binding_event_review(
            autonomous_hash_readout_binding_executor={
                **binding_execution,
                "autonomous_hash_readout_binding_event": {
                    **binding_execution["autonomous_hash_readout_binding_event"],
                    "autonomous_hash_readout_binding_event_hash": "0" * 64,
                },
            },
            expected_state_revision=runtime_state.state_revision,
        )
    )
    binding_event_review = ledger.autonomous_hash_readout_binding_event_review(
        autonomous_hash_readout_binding_executor=binding_execution,
        expected_state_revision=runtime_state.state_revision,
        review_policy={"min_bindings": 1, "max_bindings": 2},
    )

    assert blocked_binding_event_review["ready"] is False
    assert blocked_binding_event_review["requires_operator_approval"] is False
    assert blocked_binding_event_review["promotion_gate"]["required_evidence"][
        "event_recorded_in_ledger"
    ] is False
    assert binding_event_review["surface"] == (
        "snn_language_autonomous_hash_readout_binding_event_review.v1"
    )
    assert binding_event_review["ready"] is True
    assert binding_event_review["requires_operator_approval"] is False
    assert binding_event_review["advisory"] is True
    assert binding_event_review["executable"] is False
    assert binding_event_review["records_ledger_event"] is False
    assert binding_event_review["mutates_runtime_state"] is False
    assert binding_event_review["runs_replay"] is False
    assert binding_event_review["runs_calibration_update"] is False
    assert binding_event_review["writes_checkpoint"] is False
    assert binding_event_review["generates_text"] is False
    assert binding_event_review["decodes_text"] is False
    assert binding_event_review["trains_runtime_model"] is False
    assert binding_event_review["applies_plasticity"] is False
    binding_event_review_body = binding_event_review[
        "autonomous_hash_readout_binding_event_review"
    ]
    assert binding_event_review_body["event_recorded_in_ledger"] is True
    assert binding_event_review_body["binding_count"] == 1
    assert binding_event_review_body["candidate_hashes"] == [candidate_hash]
    _assert_record_family_source_window(
        binding_event_review["source_window"],
        field="autonomous_hash_readout_binding_events",
        expected_count=1,
    )
    assert binding_event_review_body["output_is_hash_binding_only"] is True
    assert binding_event_review_body["operator_approval_required"] is False
    assert binding_event_review_body["mutation_allowed"] is False
    assert binding_event_review["promotion_gate"][
        "eligible_for_autonomous_bound_readout_observation_design"
    ] is True
    assert binding_event_review["promotion_gate"]["eligible_for_language_generation"] is False
    assert binding_event_review["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert binding_event_review["promotion_gate"]["eligible_for_replay_memory"] is False
    assert binding_event_review["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert binding_event_review["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert binding_event_review["promotion_gate"]["eligible_for_action"] is False
    blocked_observation_design = ledger.autonomous_bound_readout_observation_design(
        autonomous_hash_readout_binding_event_review=blocked_binding_event_review,
        observation_policy={"observation_cycles": 4},
        device_evidence={"device": "cpu", "source": "unit"},
    )
    observation_design = ledger.autonomous_bound_readout_observation_design(
        autonomous_hash_readout_binding_event_review=binding_event_review,
        observation_policy={
            "observation_cycles": 4,
            "min_activation_sparsity": 0.5,
            "max_slot_drift": 0.15,
            "min_binding_reactivation": 0.5,
        },
        device_evidence={"device": "cpu", "source": "unit"},
    )

    assert blocked_observation_design["ready"] is False
    assert blocked_observation_design["requires_operator_approval"] is False
    assert blocked_observation_design["promotion_gate"]["required_evidence"][
        "binding_event_review_ready"
    ] is False
    assert observation_design["surface"] == (
        "snn_language_autonomous_bound_readout_observation_design.v1"
    )
    assert observation_design["ready"] is True
    assert observation_design["requires_operator_approval"] is False
    assert observation_design["advisory"] is True
    assert observation_design["executable"] is False
    assert observation_design["records_ledger_event"] is False
    assert observation_design["mutates_runtime_state"] is False
    assert observation_design["runs_replay"] is False
    assert observation_design["runs_calibration_update"] is False
    assert observation_design["writes_checkpoint"] is False
    assert observation_design["generates_text"] is False
    assert observation_design["decodes_text"] is False
    assert observation_design["trains_runtime_model"] is False
    assert observation_design["applies_plasticity"] is False
    observation_body = observation_design[
        "autonomous_bound_readout_observation_design"
    ]
    assert observation_body["binding_count"] == 1
    assert observation_body["observation_cycles"] == 4
    assert observation_body["output_is_hash_observation_only"] is True
    assert observation_body["operator_approval_required"] is False
    assert observation_body["execution_allowed"] is False
    assert observation_body["observation_targets"][0][
        "dense_label_candidate_evidence_hash"
    ] == candidate_hash
    assert observation_design["promotion_gate"][
        "eligible_for_autonomous_bound_readout_observation_preflight"
    ] is True
    assert observation_design["promotion_gate"]["eligible_for_language_generation"] is False
    assert observation_design["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert observation_design["promotion_gate"]["eligible_for_replay_memory"] is False
    assert observation_design["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert observation_design["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert observation_design["promotion_gate"]["eligible_for_action"] is False
    blocked_observation_preflight = ledger.autonomous_bound_readout_observation_preflight(
        autonomous_bound_readout_observation_design=observation_design,
        expected_state_revision=runtime_state.state_revision,
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"autonomous_bound_readout_observation_executor": False},
    )
    observation_preflight = ledger.autonomous_bound_readout_observation_preflight(
        autonomous_bound_readout_observation_design=observation_design,
        expected_state_revision=runtime_state.state_revision,
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"autonomous_bound_readout_observation_executor": True},
    )

    assert blocked_observation_preflight["ready"] is False
    assert blocked_observation_preflight["requires_operator_approval"] is False
    assert blocked_observation_preflight["promotion_gate"]["required_evidence"][
        "executor_capability_available"
    ] is False
    assert observation_preflight["surface"] == (
        "snn_language_autonomous_bound_readout_observation_preflight.v1"
    )
    assert observation_preflight["ready"] is True
    assert observation_preflight["requires_operator_approval"] is False
    assert observation_preflight["advisory"] is True
    assert observation_preflight["executable"] is False
    assert observation_preflight["records_ledger_event"] is False
    assert observation_preflight["mutates_runtime_state"] is False
    assert observation_preflight["runs_replay"] is False
    assert observation_preflight["runs_calibration_update"] is False
    assert observation_preflight["writes_checkpoint"] is False
    assert observation_preflight["generates_text"] is False
    assert observation_preflight["decodes_text"] is False
    assert observation_preflight["trains_runtime_model"] is False
    assert observation_preflight["applies_plasticity"] is False
    assert observation_preflight["observation_preflight"]["binding_count"] == 1
    assert observation_preflight["observation_preflight"]["observation_cycles"] == 4
    assert observation_preflight["observation_preflight"][
        "output_is_hash_observation_only"
    ] is True
    assert observation_preflight["device_preflight"][
        "executor_capability_available"
    ] is True
    assert observation_preflight["promotion_gate"][
        "eligible_for_autonomous_bound_readout_observation_executor"
    ] is True
    assert observation_preflight["promotion_gate"]["eligible_for_language_generation"] is False
    assert observation_preflight["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert observation_preflight["promotion_gate"]["eligible_for_replay_memory"] is False
    assert observation_preflight["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert observation_preflight["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert observation_preflight["promotion_gate"]["eligible_for_action"] is False
    observation_target_hash = observation_preflight["observation_preflight"][
        "target_hashes"
    ][0]
    observation_slot_hash = observation_preflight["observation_preflight"][
        "slot_hashes"
    ][0]
    observation_samples = [
        {
            "binding_observation_hash": observation_target_hash,
            "dense_label_candidate_evidence_hash": candidate_hash,
            "readout_slot_hash": observation_slot_hash,
            "activation_sparsity": 0.75,
            "slot_drift": 0.05,
            "binding_reactivation": 0.8,
        }
        for _ in range(4)
    ]
    low_activation_samples = [
        {
            **sample,
            "activation_sparsity": 0.1,
        }
        for sample in observation_samples
    ]
    blocked_observation_execution = ledger.execute_autonomous_bound_readout_observation(
        autonomous_bound_readout_observation_preflight=observation_preflight,
        expected_state_revision=runtime_state.state_revision,
        observation_evidence={"samples": low_activation_samples},
        execution_policy={"max_samples": 4},
    )
    observation_before_revision = runtime_state.state_revision
    observation_execution = ledger.execute_autonomous_bound_readout_observation(
        autonomous_bound_readout_observation_preflight=observation_preflight,
        expected_state_revision=observation_before_revision,
        observation_evidence={"samples": observation_samples},
        execution_policy={"max_samples": 4},
    )
    stale_observation_execution = ledger.execute_autonomous_bound_readout_observation(
        autonomous_bound_readout_observation_preflight=observation_preflight,
        expected_state_revision=runtime_state.state_revision,
        observation_evidence={"samples": observation_samples},
        execution_policy={"max_samples": 4},
    )

    assert blocked_observation_execution["accepted"] is False
    assert blocked_observation_execution["records_ledger_event"] is False
    assert blocked_observation_execution["mutates_runtime_state"] is False
    assert blocked_observation_execution["promotion_gate"]["required_evidence"][
        "activation_sparsity_sufficient"
    ] is False
    assert observation_execution["surface"] == (
        "snn_language_autonomous_bound_readout_observation_executor.v1"
    )
    assert observation_execution["accepted"] is True
    assert observation_execution["ready"] is True
    assert observation_execution["requires_operator_approval"] is False
    assert observation_execution["records_ledger_event"] is True
    assert observation_execution["mutates_runtime_state"] is True
    assert observation_execution["after"]["state_revision"] == observation_before_revision + 1
    assert observation_execution["runs_replay"] is False
    assert observation_execution["runs_calibration_update"] is False
    assert observation_execution["writes_checkpoint"] is False
    assert observation_execution["generates_text"] is False
    assert observation_execution["decodes_text"] is False
    assert observation_execution["trains_runtime_model"] is False
    assert observation_execution["applies_plasticity"] is False
    observation_event = observation_execution[
        "autonomous_bound_readout_observation_event"
    ]
    assert observation_event["operator_approval_required"] is False
    assert observation_event["output_is_hash_observation_only"] is True
    assert observation_event["sample_count"] == 4
    assert observation_event["mean_activation_sparsity"] == 0.75
    assert observation_event["max_slot_drift"] == 0.05
    assert observation_event["mean_binding_reactivation"] == 0.8
    assert observation_execution["ledger_summary"][
        "total_autonomous_bound_readout_observation_count"
    ] == 1
    _assert_record_family_source_window(
        observation_execution["source_window"],
        field="autonomous_bound_readout_observation_events",
        expected_count=0,
    )
    assert observation_execution["promotion_gate"][
        "eligible_for_autonomous_bound_readout_observation_event_review"
    ] is True
    assert observation_execution["promotion_gate"]["eligible_for_language_generation"] is False
    assert observation_execution["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert observation_execution["promotion_gate"]["eligible_for_replay_memory"] is False
    assert observation_execution["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert observation_execution["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert observation_execution["promotion_gate"]["eligible_for_action"] is False
    assert stale_observation_execution["accepted"] is False
    assert stale_observation_execution["records_ledger_event"] is False
    assert stale_observation_execution["mutates_runtime_state"] is False
    assert stale_observation_execution["promotion_gate"]["required_evidence"][
        "preflight_revision_current"
    ] is False
    blocked_observation_event_review = (
        ledger.autonomous_bound_readout_observation_event_review(
            autonomous_bound_readout_observation_executor={
                **observation_execution,
                "autonomous_bound_readout_observation_event": {
                    **observation_execution[
                        "autonomous_bound_readout_observation_event"
                    ],
                    "autonomous_bound_readout_observation_event_hash": "0" * 64,
                },
            },
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "min_samples": 4,
                "max_samples": 4,
                "min_activation_sparsity": 0.5,
                "max_slot_drift": 0.15,
                "min_binding_reactivation": 0.5,
            },
        )
    )
    observation_event_review = (
        ledger.autonomous_bound_readout_observation_event_review(
            autonomous_bound_readout_observation_executor=observation_execution,
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "min_samples": 4,
                "max_samples": 4,
                "min_activation_sparsity": 0.5,
                "max_slot_drift": 0.15,
                "min_binding_reactivation": 0.5,
            },
        )
    )

    assert blocked_observation_event_review["ready"] is False
    assert blocked_observation_event_review["requires_operator_approval"] is False
    assert blocked_observation_event_review["promotion_gate"]["required_evidence"][
        "event_recorded_in_ledger"
    ] is False
    assert observation_event_review["surface"] == (
        "snn_language_autonomous_bound_readout_observation_event_review.v1"
    )
    assert observation_event_review["ready"] is True
    assert observation_event_review["requires_operator_approval"] is False
    assert observation_event_review["advisory"] is True
    assert observation_event_review["executable"] is False
    assert observation_event_review["records_ledger_event"] is False
    assert observation_event_review["mutates_runtime_state"] is False
    assert observation_event_review["runs_replay"] is False
    assert observation_event_review["runs_calibration_update"] is False
    assert observation_event_review["writes_checkpoint"] is False
    assert observation_event_review["generates_text"] is False
    assert observation_event_review["decodes_text"] is False
    assert observation_event_review["trains_runtime_model"] is False
    assert observation_event_review["applies_plasticity"] is False
    observation_event_review_body = observation_event_review[
        "autonomous_bound_readout_observation_event_review"
    ]
    assert observation_event_review_body["event_recorded_in_ledger"] is True
    assert observation_event_review_body["binding_count"] == 1
    assert observation_event_review_body["observation_cycles"] == 4
    assert observation_event_review_body["sample_count"] == 4
    assert observation_event_review_body["mean_activation_sparsity"] == 0.75
    assert observation_event_review_body["max_slot_drift"] == 0.05
    assert observation_event_review_body["mean_binding_reactivation"] == 0.8
    _assert_record_family_source_window(
        observation_event_review["source_window"],
        field="autonomous_bound_readout_observation_events",
        expected_count=1,
    )
    assert observation_event_review_body["output_is_hash_observation_only"] is True
    assert observation_event_review_body["operator_approval_required"] is False
    assert observation_event_review_body["mutation_allowed"] is False
    assert observation_event_review["promotion_gate"][
        "eligible_for_autonomous_readout_training_window_design"
    ] is True
    assert observation_event_review["promotion_gate"]["eligible_for_language_generation"] is False
    assert observation_event_review["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert observation_event_review["promotion_gate"]["eligible_for_replay_memory"] is False
    assert observation_event_review["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert observation_event_review["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert observation_event_review["promotion_gate"]["eligible_for_action"] is False
    blocked_training_window_design = (
        ledger.autonomous_readout_training_window_design(
            autonomous_bound_readout_observation_event_review=blocked_observation_event_review,
            training_policy={"training_window_steps": 4},
            device_evidence={"device": "cpu", "source": "unit"},
        )
    )
    training_window_design = ledger.autonomous_readout_training_window_design(
        autonomous_bound_readout_observation_event_review=observation_event_review,
        training_policy={
            "training_window_steps": 4,
            "truncated_bptt_steps": 4,
            "micro_batch_size": 1,
            "max_learning_rate": 0.0003,
            "learning_rule": "surrogate_gradient",
            "min_activation_sparsity": 0.5,
            "max_slot_drift": 0.15,
            "min_binding_reactivation": 0.5,
            "use_spike_compression": True,
            "use_gradient_checkpointing": True,
        },
        device_evidence={"device": "cpu", "source": "unit"},
    )

    assert blocked_training_window_design["ready"] is False
    assert blocked_training_window_design["requires_operator_approval"] is False
    assert blocked_training_window_design["promotion_gate"]["required_evidence"][
        "observation_event_review_ready"
    ] is False
    assert training_window_design["surface"] == (
        "snn_language_autonomous_readout_training_window_design.v1"
    )
    assert training_window_design["ready"] is True
    assert training_window_design["requires_operator_approval"] is False
    assert training_window_design["advisory"] is True
    assert training_window_design["executable"] is False
    assert training_window_design["records_ledger_event"] is False
    assert training_window_design["mutates_runtime_state"] is False
    assert training_window_design["runs_replay"] is False
    assert training_window_design["runs_calibration_update"] is False
    assert training_window_design["writes_checkpoint"] is False
    assert training_window_design["generates_text"] is False
    assert training_window_design["decodes_text"] is False
    assert training_window_design["trains_runtime_model"] is False
    assert training_window_design["applies_plasticity"] is False
    training_window_body = training_window_design[
        "autonomous_readout_training_window_design"
    ]
    assert training_window_body["binding_count"] == 1
    assert training_window_body["sample_count"] == 4
    assert training_window_body["training_window_steps"] == 4
    assert training_window_body["truncated_bptt_steps"] == 4
    assert training_window_body["micro_batch_size"] == 1
    assert training_window_body["learning_rule"] == "surrogate_gradient"
    assert training_window_body["memory_plan"]["use_spike_compression"] is True
    assert training_window_body["memory_plan"]["use_gradient_checkpointing"] is True
    assert training_window_body["operator_approval_required"] is False
    assert training_window_body["execution_allowed"] is False
    assert training_window_body["output_is_training_window_plan_only"] is True
    assert training_window_design["promotion_gate"][
        "eligible_for_autonomous_readout_training_window_preflight"
    ] is True
    assert training_window_design["promotion_gate"][
        "eligible_for_autonomous_readout_training_execution"
    ] is False
    assert training_window_design["promotion_gate"]["eligible_for_language_generation"] is False
    assert training_window_design["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert training_window_design["promotion_gate"]["eligible_for_replay_memory"] is False
    assert training_window_design["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert training_window_design["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert training_window_design["promotion_gate"]["eligible_for_action"] is False
    blocked_training_window_preflight = (
        ledger.autonomous_readout_training_window_preflight(
            autonomous_readout_training_window_design=training_window_design,
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cpu", "source": "unit"},
            executor_capabilities={
                "autonomous_readout_training_window_executor": False
            },
        )
    )
    training_window_preflight = ledger.autonomous_readout_training_window_preflight(
        autonomous_readout_training_window_design=training_window_design,
        expected_state_revision=runtime_state.state_revision,
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"autonomous_readout_training_window_executor": True},
    )

    assert blocked_training_window_preflight["ready"] is False
    assert blocked_training_window_preflight["requires_operator_approval"] is False
    assert blocked_training_window_preflight["promotion_gate"]["required_evidence"][
        "executor_capability_available"
    ] is False
    assert training_window_preflight["surface"] == (
        "snn_language_autonomous_readout_training_window_preflight.v1"
    )
    assert training_window_preflight["ready"] is True
    assert training_window_preflight["requires_operator_approval"] is False
    assert training_window_preflight["advisory"] is True
    assert training_window_preflight["executable"] is False
    assert training_window_preflight["records_ledger_event"] is False
    assert training_window_preflight["mutates_runtime_state"] is False
    assert training_window_preflight["runs_replay"] is False
    assert training_window_preflight["runs_calibration_update"] is False
    assert training_window_preflight["writes_checkpoint"] is False
    assert training_window_preflight["generates_text"] is False
    assert training_window_preflight["decodes_text"] is False
    assert training_window_preflight["trains_runtime_model"] is False
    assert training_window_preflight["applies_plasticity"] is False
    training_window_preflight_body = training_window_preflight[
        "training_window_preflight"
    ]
    assert training_window_preflight_body["sample_count"] == 4
    assert training_window_preflight_body["training_window_steps"] == 4
    assert training_window_preflight_body["truncated_bptt_steps"] == 4
    assert training_window_preflight_body["learning_rule"] == "surrogate_gradient"
    assert training_window_preflight_body["output_is_training_window_plan_only"] is True
    assert training_window_preflight_body["operator_approval_required"] is False
    assert training_window_preflight_body["execution_allowed"] is False
    assert training_window_preflight["device_preflight"][
        "executor_capability_available"
    ] is True
    assert training_window_preflight["promotion_gate"][
        "eligible_for_autonomous_readout_training_window_executor"
    ] is True
    assert training_window_preflight["promotion_gate"][
        "eligible_for_autonomous_readout_training_execution"
    ] is True
    assert training_window_preflight["promotion_gate"]["eligible_for_language_generation"] is False
    assert training_window_preflight["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert training_window_preflight["promotion_gate"]["eligible_for_replay_memory"] is False
    assert training_window_preflight["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert training_window_preflight["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert training_window_preflight["promotion_gate"]["eligible_for_action"] is False
    training_evidence = {
        "sample_hashes": training_window_preflight_body["sample_hashes"],
        "weight_update_hash": ledger._sha256_json(["weight-update", "unit"]),
        "gradient_update_hash": ledger._sha256_json(["gradient-update", "unit"]),
        "optimizer_state_hash": ledger._sha256_json(["optimizer-state", "unit"]),
        "device_trace_hash": ledger._sha256_json(["device-trace", "unit"]),
        "runtime_weights_updated": True,
        "checkpoint_written": False,
        "learning_rate": 0.0002,
        "loss_before": 0.42,
        "loss_after": 0.38,
        "mean_gradient_norm": 1.2,
        "max_weight_delta": 0.01,
        "observed_spike_sparsity": 0.74,
    }
    regressed_training_evidence = {
        **training_evidence,
        "loss_after": 0.8,
    }
    blocked_training_execution = ledger.execute_autonomous_readout_training_window(
        autonomous_readout_training_window_preflight=training_window_preflight,
        expected_state_revision=runtime_state.state_revision,
        training_evidence=regressed_training_evidence,
        execution_policy={
            "max_loss_increase": 0.02,
            "max_gradient_norm": 10.0,
            "max_weight_delta": 0.05,
            "min_spike_sparsity": 0.5,
        },
    )
    training_before_revision = runtime_state.state_revision
    training_execution = ledger.execute_autonomous_readout_training_window(
        autonomous_readout_training_window_preflight=training_window_preflight,
        expected_state_revision=training_before_revision,
        training_evidence=training_evidence,
        execution_policy={
            "max_loss_increase": 0.02,
            "max_gradient_norm": 10.0,
            "max_weight_delta": 0.05,
            "min_spike_sparsity": 0.5,
        },
    )
    stale_training_execution = ledger.execute_autonomous_readout_training_window(
        autonomous_readout_training_window_preflight=training_window_preflight,
        expected_state_revision=runtime_state.state_revision,
        training_evidence=training_evidence,
    )

    assert blocked_training_execution["accepted"] is False
    assert blocked_training_execution["records_ledger_event"] is False
    assert blocked_training_execution["mutates_runtime_state"] is False
    assert blocked_training_execution["promotion_gate"]["required_evidence"][
        "loss_not_regressed_beyond_policy"
    ] is False
    assert training_execution["surface"] == (
        "snn_language_autonomous_readout_training_window_executor.v1"
    )
    assert training_execution["accepted"] is True
    assert training_execution["ready"] is True
    assert training_execution["requires_operator_approval"] is False
    assert training_execution["records_ledger_event"] is True
    assert training_execution["mutates_runtime_state"] is True
    assert training_execution["trains_runtime_model"] is True
    assert training_execution["after"]["state_revision"] == training_before_revision + 1
    assert training_execution["runs_replay"] is False
    assert training_execution["runs_calibration_update"] is False
    assert training_execution["writes_checkpoint"] is False
    assert training_execution["generates_text"] is False
    assert training_execution["decodes_text"] is False
    assert training_execution["applies_plasticity"] is False
    training_event = training_execution["autonomous_readout_training_window_event"]
    assert training_event["runtime_weights_updated"] is True
    assert training_event["trains_runtime_model"] is True
    assert training_event["operator_approval_required"] is False
    assert training_event["loss_before"] == 0.42
    assert training_event["loss_after"] == 0.38
    assert training_event["observed_spike_sparsity"] == 0.74
    assert training_execution["ledger_summary"][
        "total_autonomous_readout_training_window_count"
    ] == 1
    _assert_record_family_source_window(
        training_execution["source_window"],
        field="autonomous_readout_training_window_events",
        expected_count=0,
    )
    assert training_execution["promotion_gate"][
        "eligible_for_autonomous_readout_training_window_event_review"
    ] is True
    assert training_execution["promotion_gate"]["eligible_for_language_generation"] is False
    assert training_execution["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert training_execution["promotion_gate"]["eligible_for_replay_memory"] is False
    assert training_execution["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert training_execution["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert training_execution["promotion_gate"]["eligible_for_action"] is False
    assert stale_training_execution["accepted"] is False
    assert stale_training_execution["records_ledger_event"] is False
    assert stale_training_execution["mutates_runtime_state"] is False
    assert stale_training_execution["promotion_gate"]["required_evidence"][
        "preflight_revision_current"
    ] is False
    blocked_training_event_review = (
        ledger.autonomous_readout_training_window_event_review(
            autonomous_readout_training_window_executor={
                **training_execution,
                "autonomous_readout_training_window_event": {
                    **training_execution[
                        "autonomous_readout_training_window_event"
                    ],
                    "autonomous_readout_training_window_event_hash": "0" * 64,
                },
            },
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "max_loss_increase": 0.02,
                "max_gradient_norm": 10.0,
                "max_weight_delta": 0.05,
                "min_spike_sparsity": 0.5,
            },
        )
    )
    training_event_review = ledger.autonomous_readout_training_window_event_review(
        autonomous_readout_training_window_executor=training_execution,
        expected_state_revision=runtime_state.state_revision,
        review_policy={
            "max_loss_increase": 0.02,
            "max_gradient_norm": 10.0,
            "max_weight_delta": 0.05,
            "min_spike_sparsity": 0.5,
        },
    )

    assert blocked_training_event_review["ready"] is False
    assert blocked_training_event_review["requires_operator_approval"] is False
    assert blocked_training_event_review["promotion_gate"]["required_evidence"][
        "event_recorded_in_ledger"
    ] is False
    assert training_event_review["surface"] == (
        "snn_language_autonomous_readout_training_window_event_review.v1"
    )
    assert training_event_review["ready"] is True
    assert training_event_review["requires_operator_approval"] is False
    assert training_event_review["advisory"] is True
    assert training_event_review["executable"] is False
    assert training_event_review["records_ledger_event"] is False
    assert training_event_review["mutates_runtime_state"] is False
    assert training_event_review["runs_replay"] is False
    assert training_event_review["runs_calibration_update"] is False
    assert training_event_review["writes_checkpoint"] is False
    assert training_event_review["generates_text"] is False
    assert training_event_review["decodes_text"] is False
    assert training_event_review["trains_runtime_model"] is False
    assert training_event_review["applies_plasticity"] is False
    training_review_body = training_event_review[
        "autonomous_readout_training_window_event_review"
    ]
    assert training_review_body["event_recorded_in_ledger"] is True
    assert training_review_body["runtime_weights_updated"] is True
    assert training_review_body["loss_before"] == 0.42
    assert training_review_body["loss_after"] == 0.38
    assert training_review_body["observed_spike_sparsity"] == 0.74
    assert training_review_body["operator_approval_required"] is False
    assert training_review_body["mutation_allowed"] is False
    _assert_record_family_source_window(
        training_event_review["source_window"],
        field="autonomous_readout_training_window_events",
        expected_count=1,
    )
    assert training_event_review["promotion_gate"][
        "eligible_for_autonomous_decoder_probe_design"
    ] is True
    assert training_event_review["promotion_gate"]["eligible_for_language_generation"] is False
    assert training_event_review["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert training_event_review["promotion_gate"]["eligible_for_replay_memory"] is False
    assert training_event_review["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert training_event_review["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert training_event_review["promotion_gate"]["eligible_for_action"] is False
    blocked_decoder_probe_design = ledger.autonomous_decoder_probe_design(
        autonomous_readout_training_window_event_review=blocked_training_event_review,
        probe_policy={"max_probe_steps": 4, "top_k": 1},
        device_evidence={"device": "cpu", "source": "unit"},
    )
    decoder_probe_design = ledger.autonomous_decoder_probe_design(
        autonomous_readout_training_window_event_review=training_event_review,
        probe_policy={
            "probe_mode": "hash_rank_probe",
            "max_probe_steps": 4,
            "top_k": 1,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
        device_evidence={"device": "cpu", "source": "unit"},
    )

    assert blocked_decoder_probe_design["ready"] is False
    assert blocked_decoder_probe_design["requires_operator_approval"] is False
    assert blocked_decoder_probe_design["promotion_gate"]["required_evidence"][
        "training_event_review_ready"
    ] is False
    assert decoder_probe_design["surface"] == (
        "snn_language_autonomous_decoder_probe_design.v1"
    )
    assert decoder_probe_design["ready"] is True
    assert decoder_probe_design["requires_operator_approval"] is False
    assert decoder_probe_design["advisory"] is True
    assert decoder_probe_design["executable"] is False
    assert decoder_probe_design["records_ledger_event"] is False
    assert decoder_probe_design["mutates_runtime_state"] is False
    assert decoder_probe_design["runs_replay"] is False
    assert decoder_probe_design["runs_calibration_update"] is False
    assert decoder_probe_design["writes_checkpoint"] is False
    assert decoder_probe_design["generates_text"] is False
    assert decoder_probe_design["decodes_text"] is False
    assert decoder_probe_design["trains_runtime_model"] is False
    assert decoder_probe_design["applies_plasticity"] is False
    decoder_probe_body = decoder_probe_design["autonomous_decoder_probe_design"]
    assert decoder_probe_body["probe_mode"] == "hash_rank_probe"
    assert decoder_probe_body["max_probe_steps"] == 4
    assert decoder_probe_body["top_k"] == 1
    assert decoder_probe_body["output_is_hash_probe_only"] is True
    assert decoder_probe_body["operator_approval_required"] is False
    assert decoder_probe_body["execution_allowed"] is False
    assert decoder_probe_body["weight_update_hash"] == training_evidence[
        "weight_update_hash"
    ]
    assert len(decoder_probe_body["probe_targets"]) == 1
    assert decoder_probe_design["promotion_gate"][
        "eligible_for_autonomous_decoder_probe_preflight"
    ] is True
    assert decoder_probe_design["promotion_gate"]["eligible_for_language_generation"] is False
    assert decoder_probe_design["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert decoder_probe_design["promotion_gate"]["eligible_for_replay_memory"] is False
    assert decoder_probe_design["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert decoder_probe_design["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert decoder_probe_design["promotion_gate"]["eligible_for_action"] is False
    blocked_decoder_probe_preflight = ledger.autonomous_decoder_probe_preflight(
        autonomous_decoder_probe_design=decoder_probe_design,
        expected_state_revision=runtime_state.state_revision,
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"autonomous_decoder_probe_executor": False},
    )
    decoder_probe_preflight = ledger.autonomous_decoder_probe_preflight(
        autonomous_decoder_probe_design=decoder_probe_design,
        expected_state_revision=runtime_state.state_revision,
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"autonomous_decoder_probe_executor": True},
    )

    assert blocked_decoder_probe_preflight["ready"] is False
    assert blocked_decoder_probe_preflight["requires_operator_approval"] is False
    assert blocked_decoder_probe_preflight["promotion_gate"]["required_evidence"][
        "executor_capability_available"
    ] is False
    assert decoder_probe_preflight["surface"] == (
        "snn_language_autonomous_decoder_probe_preflight.v1"
    )
    assert decoder_probe_preflight["ready"] is True
    assert decoder_probe_preflight["requires_operator_approval"] is False
    assert decoder_probe_preflight["advisory"] is True
    assert decoder_probe_preflight["executable"] is False
    assert decoder_probe_preflight["records_ledger_event"] is False
    assert decoder_probe_preflight["mutates_runtime_state"] is False
    assert decoder_probe_preflight["runs_replay"] is False
    assert decoder_probe_preflight["runs_calibration_update"] is False
    assert decoder_probe_preflight["writes_checkpoint"] is False
    assert decoder_probe_preflight["generates_text"] is False
    assert decoder_probe_preflight["decodes_text"] is False
    assert decoder_probe_preflight["trains_runtime_model"] is False
    assert decoder_probe_preflight["applies_plasticity"] is False
    decoder_probe_preflight_body = decoder_probe_preflight[
        "decoder_probe_preflight"
    ]
    assert decoder_probe_preflight_body["probe_mode"] == "hash_rank_probe"
    assert decoder_probe_preflight_body["max_probe_steps"] == 4
    assert decoder_probe_preflight_body["top_k"] == 1
    assert decoder_probe_preflight_body["output_is_hash_probe_only"] is True
    assert decoder_probe_preflight_body["operator_approval_required"] is False
    assert decoder_probe_preflight_body["execution_allowed"] is False
    assert decoder_probe_preflight["device_preflight"][
        "executor_capability_available"
    ] is True
    assert decoder_probe_preflight["promotion_gate"][
        "eligible_for_autonomous_decoder_probe_executor"
    ] is True
    assert decoder_probe_preflight["promotion_gate"]["eligible_for_language_generation"] is False
    assert decoder_probe_preflight["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert decoder_probe_preflight["promotion_gate"]["eligible_for_replay_memory"] is False
    assert decoder_probe_preflight["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert decoder_probe_preflight["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert decoder_probe_preflight["promotion_gate"]["eligible_for_action"] is False
    probe_target_hash = decoder_probe_preflight_body["probe_target_hashes"][0]
    probe_result = {
        "probe_target_hash": probe_target_hash,
        "output_hash": ledger._sha256_json(["probe-output", "unit"]),
        "rank_hashes": [
            ledger._sha256_json(["rank", "unit", 0]),
            ledger._sha256_json(["rank", "unit", 1]),
        ],
        "top_score": 0.82,
        "spike_sparsity": 0.73,
        "slot_drift": 0.04,
    }
    blocked_probe_execution = ledger.execute_autonomous_decoder_probe(
        autonomous_decoder_probe_preflight=decoder_probe_preflight,
        expected_state_revision=runtime_state.state_revision,
        probe_evidence={
            "probe_results": [{**probe_result, "spike_sparsity": 0.1}],
            "checkpoint_written": False,
        },
        execution_policy={"min_top_score": 0.5},
    )
    probe_before_revision = runtime_state.state_revision
    probe_execution = ledger.execute_autonomous_decoder_probe(
        autonomous_decoder_probe_preflight=decoder_probe_preflight,
        expected_state_revision=probe_before_revision,
        probe_evidence={
            "probe_results": [probe_result],
            "checkpoint_written": False,
        },
        execution_policy={"min_top_score": 0.5},
    )
    stale_probe_execution = ledger.execute_autonomous_decoder_probe(
        autonomous_decoder_probe_preflight=decoder_probe_preflight,
        expected_state_revision=runtime_state.state_revision,
        probe_evidence={
            "probe_results": [probe_result],
            "checkpoint_written": False,
        },
    )

    assert blocked_probe_execution["accepted"] is False
    assert blocked_probe_execution["records_ledger_event"] is False
    assert blocked_probe_execution["mutates_runtime_state"] is False
    assert blocked_probe_execution["promotion_gate"]["required_evidence"][
        "spike_sparsity_sufficient"
    ] is False
    assert probe_execution["surface"] == (
        "snn_language_autonomous_decoder_probe_executor.v1"
    )
    assert probe_execution["accepted"] is True
    assert probe_execution["ready"] is True
    assert probe_execution["requires_operator_approval"] is False
    assert probe_execution["records_ledger_event"] is True
    assert probe_execution["mutates_runtime_state"] is True
    assert probe_execution["after"]["state_revision"] == probe_before_revision + 1
    assert probe_execution["runs_replay"] is False
    assert probe_execution["runs_calibration_update"] is False
    assert probe_execution["writes_checkpoint"] is False
    assert probe_execution["generates_text"] is False
    assert probe_execution["decodes_text"] is False
    assert probe_execution["trains_runtime_model"] is False
    assert probe_execution["applies_plasticity"] is False
    probe_event = probe_execution["autonomous_decoder_probe_event"]
    assert probe_event["operator_approval_required"] is False
    assert probe_event["output_is_hash_probe_only"] is True
    assert probe_event["probe_result_count"] == 1
    assert probe_event["mean_top_score"] == 0.82
    assert probe_event["mean_spike_sparsity"] == 0.73
    assert probe_event["max_slot_drift"] == 0.04
    assert probe_execution["ledger_summary"]["total_autonomous_decoder_probe_count"] == 1
    _assert_record_family_source_window(
        probe_execution["source_window"],
        field="autonomous_decoder_probe_events",
        expected_count=0,
    )
    assert probe_execution["promotion_gate"][
        "eligible_for_autonomous_decoder_probe_event_review"
    ] is True
    assert probe_execution["promotion_gate"]["eligible_for_language_generation"] is False
    assert probe_execution["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert probe_execution["promotion_gate"]["eligible_for_replay_memory"] is False
    assert probe_execution["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert probe_execution["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert probe_execution["promotion_gate"]["eligible_for_action"] is False
    assert stale_probe_execution["accepted"] is False
    assert stale_probe_execution["records_ledger_event"] is False
    assert stale_probe_execution["mutates_runtime_state"] is False
    assert stale_probe_execution["promotion_gate"]["required_evidence"][
        "preflight_revision_current"
    ] is False
    blocked_probe_event_review = ledger.autonomous_decoder_probe_event_review(
        autonomous_decoder_probe_executor={
            **probe_execution,
            "autonomous_decoder_probe_event": {
                **probe_execution["autonomous_decoder_probe_event"],
                "autonomous_decoder_probe_event_hash": "0" * 64,
            },
        },
        expected_state_revision=runtime_state.state_revision,
        review_policy={
            "min_top_score": 0.5,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
    )
    probe_event_review = ledger.autonomous_decoder_probe_event_review(
        autonomous_decoder_probe_executor=probe_execution,
        expected_state_revision=runtime_state.state_revision,
        review_policy={
            "min_top_score": 0.5,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
    )

    assert blocked_probe_event_review["ready"] is False
    assert blocked_probe_event_review["requires_operator_approval"] is False
    assert blocked_probe_event_review["autonomous_decoder_probe_event_review"][
        "event_recorded_in_ledger"
    ] is False
    assert probe_event_review["surface"] == (
        "snn_language_autonomous_decoder_probe_event_review.v1"
    )
    assert probe_event_review["ready"] is True
    assert probe_event_review["accepted"] is True
    assert probe_event_review["requires_operator_approval"] is False
    assert probe_event_review["advisory"] is True
    assert probe_event_review["executable"] is False
    assert probe_event_review["records_ledger_event"] is False
    assert probe_event_review["mutates_runtime_state"] is False
    assert probe_event_review["runs_replay"] is False
    assert probe_event_review["runs_calibration_update"] is False
    assert probe_event_review["writes_checkpoint"] is False
    assert probe_event_review["generates_text"] is False
    assert probe_event_review["decodes_text"] is False
    assert probe_event_review["trains_runtime_model"] is False
    assert probe_event_review["applies_plasticity"] is False
    probe_event_review_body = probe_event_review[
        "autonomous_decoder_probe_event_review"
    ]
    assert probe_event_review_body["event_recorded_in_ledger"] is True
    assert probe_event_review_body["probe_result_count"] == 1
    assert probe_event_review_body["mean_top_score"] == 0.82
    assert probe_event_review_body["mean_spike_sparsity"] == 0.73
    assert probe_event_review_body["max_slot_drift"] == 0.04
    assert probe_event_review_body["output_is_hash_probe_only"] is True
    assert probe_event_review_body["operator_approval_required"] is False
    assert probe_event_review_body["mutation_allowed"] is False
    _assert_record_family_source_window(
        probe_event_review["source_window"],
        field="autonomous_decoder_probe_events",
        expected_count=1,
    )
    assert probe_event_review["promotion_gate"][
        "eligible_for_autonomous_language_output_design"
    ] is True
    assert probe_event_review["promotion_gate"]["eligible_for_language_generation"] is False
    assert probe_event_review["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert probe_event_review["promotion_gate"]["eligible_for_replay_memory"] is False
    assert probe_event_review["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert probe_event_review["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert probe_event_review["promotion_gate"]["eligible_for_action"] is False
    blocked_language_output_design = ledger.autonomous_language_output_design(
        autonomous_decoder_probe_event_review=blocked_probe_event_review,
        output_policy={
            "output_mode": "token_hash_sequence",
            "max_output_tokens": 3,
            "min_top_score": 0.5,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
        device_evidence={"device": "cpu", "source": "unit"},
    )
    language_output_design = ledger.autonomous_language_output_design(
        autonomous_decoder_probe_event_review=probe_event_review,
        output_policy={
            "output_mode": "token_hash_sequence",
            "max_output_tokens": 3,
            "min_top_score": 0.5,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
        device_evidence={"device": "cpu", "source": "unit"},
    )

    assert blocked_language_output_design["ready"] is False
    assert blocked_language_output_design["requires_operator_approval"] is False
    assert blocked_language_output_design["promotion_gate"]["required_evidence"][
        "decoder_probe_event_review_ready"
    ] is False
    assert language_output_design["surface"] == (
        "snn_language_autonomous_language_output_design.v1"
    )
    assert language_output_design["ready"] is True
    assert language_output_design["requires_operator_approval"] is False
    assert language_output_design["advisory"] is True
    assert language_output_design["executable"] is False
    assert language_output_design["records_ledger_event"] is False
    assert language_output_design["mutates_runtime_state"] is False
    assert language_output_design["runs_replay"] is False
    assert language_output_design["runs_calibration_update"] is False
    assert language_output_design["writes_checkpoint"] is False
    assert language_output_design["generates_text"] is False
    assert language_output_design["decodes_text"] is False
    assert language_output_design["trains_runtime_model"] is False
    assert language_output_design["applies_plasticity"] is False
    language_output_body = language_output_design["autonomous_language_output_design"]
    assert language_output_body["output_mode"] == "token_hash_sequence"
    assert language_output_body["max_output_tokens"] == 3
    assert len(language_output_body["candidate_hashes"]) == 3
    assert len(language_output_body["output_slots"]) == 3
    assert language_output_body["mean_top_score"] == 0.82
    assert language_output_body["mean_spike_sparsity"] == 0.73
    assert language_output_body["max_slot_drift"] == 0.04
    assert language_output_body["output_is_hash_probe_only"] is True
    assert language_output_body["decoded_text_allowed"] is False
    assert language_output_body["generated_text_allowed"] is False
    assert language_output_body["operator_approval_required"] is False
    assert language_output_design["promotion_gate"][
        "eligible_for_autonomous_language_output_preflight"
    ] is True
    assert language_output_design["promotion_gate"]["eligible_for_language_generation"] is False
    assert language_output_design["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert language_output_design["promotion_gate"]["eligible_for_replay_memory"] is False
    assert language_output_design["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert language_output_design["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert language_output_design["promotion_gate"]["eligible_for_action"] is False
    blocked_language_output_preflight = ledger.autonomous_language_output_preflight(
        autonomous_language_output_design=language_output_design,
        expected_state_revision=runtime_state.state_revision,
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"autonomous_language_output_executor": False},
    )
    language_output_preflight = ledger.autonomous_language_output_preflight(
        autonomous_language_output_design=language_output_design,
        expected_state_revision=runtime_state.state_revision,
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"autonomous_language_output_executor": True},
    )

    assert blocked_language_output_preflight["ready"] is False
    assert blocked_language_output_preflight["requires_operator_approval"] is False
    assert blocked_language_output_preflight["promotion_gate"]["required_evidence"][
        "executor_capability_available"
    ] is False
    assert language_output_preflight["surface"] == (
        "snn_language_autonomous_language_output_preflight.v1"
    )
    assert language_output_preflight["ready"] is True
    assert language_output_preflight["requires_operator_approval"] is False
    assert language_output_preflight["advisory"] is True
    assert language_output_preflight["executable"] is False
    assert language_output_preflight["records_ledger_event"] is False
    assert language_output_preflight["mutates_runtime_state"] is False
    assert language_output_preflight["runs_replay"] is False
    assert language_output_preflight["runs_calibration_update"] is False
    assert language_output_preflight["writes_checkpoint"] is False
    assert language_output_preflight["generates_text"] is False
    assert language_output_preflight["decodes_text"] is False
    assert language_output_preflight["trains_runtime_model"] is False
    assert language_output_preflight["applies_plasticity"] is False
    language_output_preflight_body = language_output_preflight[
        "autonomous_language_output_preflight"
    ]
    assert language_output_preflight_body["output_mode"] == "token_hash_sequence"
    assert language_output_preflight_body["max_output_tokens"] == 3
    assert len(language_output_preflight_body["candidate_hashes"]) == 3
    assert len(language_output_preflight_body["output_slot_hashes"]) == 3
    assert language_output_preflight_body["mean_top_score"] == 0.82
    assert language_output_preflight_body["mean_spike_sparsity"] == 0.73
    assert language_output_preflight_body["max_slot_drift"] == 0.04
    assert language_output_preflight_body["output_is_hash_probe_only"] is True
    assert language_output_preflight_body["decoded_text_allowed"] is False
    assert language_output_preflight_body["generated_text_allowed"] is False
    assert language_output_preflight_body["operator_approval_required"] is False
    assert language_output_preflight["promotion_gate"][
        "eligible_for_autonomous_language_output_executor"
    ] is True
    assert language_output_preflight["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert language_output_preflight["promotion_gate"][
        "eligible_for_dense_readout_training"
    ] is False
    assert language_output_preflight["promotion_gate"]["eligible_for_replay_memory"] is False
    assert language_output_preflight["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert language_output_preflight["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert language_output_preflight["promotion_gate"]["eligible_for_action"] is False
    output_slot_results = [
        {
            "language_output_slot_hash": slot_hash,
            "candidate_hash": candidate_hash,
            "emitted_hash": ledger._sha256_json(
                ["language-output", candidate_hash, index]
            ),
            "rank_hashes": [ledger._sha256_json(["language-rank", index, 0])],
            "confidence_score": 0.81,
            "spike_sparsity": 0.74,
            "slot_drift": 0.03,
        }
        for index, (slot_hash, candidate_hash) in enumerate(
            zip(
                language_output_preflight_body["output_slot_hashes"],
                language_output_preflight_body["candidate_hashes"],
            )
        )
    ]
    blocked_language_output_execution = ledger.execute_autonomous_language_output(
        autonomous_language_output_preflight=language_output_preflight,
        expected_state_revision=runtime_state.state_revision,
        output_evidence={
            "output_slot_results": output_slot_results,
            "generated_text": "blocked text",
            "checkpoint_written": False,
        },
        execution_policy={
            "min_confidence_score": 0.5,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
    )
    language_output_before_revision = runtime_state.state_revision
    language_output_execution = ledger.execute_autonomous_language_output(
        autonomous_language_output_preflight=language_output_preflight,
        expected_state_revision=language_output_before_revision,
        output_evidence={
            "output_slot_results": output_slot_results,
            "checkpoint_written": False,
        },
        execution_policy={
            "min_confidence_score": 0.5,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
    )

    assert blocked_language_output_execution["accepted"] is False
    assert blocked_language_output_execution["records_ledger_event"] is False
    assert blocked_language_output_execution["mutates_runtime_state"] is False
    assert blocked_language_output_execution["promotion_gate"]["required_evidence"][
        "generated_text_absent"
    ] is False
    _assert_record_family_source_window(
        blocked_language_output_execution["source_window"],
        field="autonomous_language_output_events",
        expected_count=0,
    )
    assert language_output_execution["surface"] == (
        "snn_language_autonomous_language_output_executor.v1"
    )
    assert language_output_execution["accepted"] is True
    assert language_output_execution["ready"] is True
    assert language_output_execution["requires_operator_approval"] is False
    assert language_output_execution["records_ledger_event"] is True
    assert language_output_execution["mutates_runtime_state"] is True
    assert language_output_execution["after"]["state_revision"] == (
        language_output_before_revision + 1
    )
    assert language_output_execution["runs_replay"] is False
    assert language_output_execution["runs_calibration_update"] is False
    assert language_output_execution["writes_checkpoint"] is False
    assert language_output_execution["generates_text"] is False
    assert language_output_execution["decodes_text"] is False
    assert language_output_execution["trains_runtime_model"] is False
    assert language_output_execution["applies_plasticity"] is False
    language_output_event = language_output_execution[
        "autonomous_language_output_event"
    ]
    assert language_output_event["operator_approval_required"] is False
    assert language_output_event["output_is_hash_only"] is True
    assert language_output_event["output_slot_count"] == 3
    assert language_output_event["mean_confidence_score"] == 0.81
    assert language_output_event["mean_spike_sparsity"] == 0.74
    assert language_output_event["max_slot_drift"] == 0.03
    assert language_output_execution["ledger_summary"][
        "total_autonomous_language_output_count"
    ] == 1
    _assert_record_family_source_window(
        language_output_execution["source_window"],
        field="autonomous_language_output_events",
        expected_count=0,
    )
    assert language_output_execution["promotion_gate"][
        "eligible_for_autonomous_language_output_event_review"
    ] is True
    assert language_output_execution["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert language_output_execution["promotion_gate"][
        "eligible_for_dense_readout_training"
    ] is False
    assert language_output_execution["promotion_gate"]["eligible_for_replay_memory"] is False
    assert language_output_execution["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert language_output_execution["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert language_output_execution["promotion_gate"]["eligible_for_action"] is False
    blocked_language_output_event_review = ledger.autonomous_language_output_event_review(
        autonomous_language_output_executor={
            **language_output_execution,
            "autonomous_language_output_event": {
                **language_output_execution["autonomous_language_output_event"],
                "autonomous_language_output_event_hash": "0" * 64,
            },
        },
        expected_state_revision=runtime_state.state_revision,
        review_policy={
            "min_confidence_score": 0.5,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
    )
    language_output_event_review = ledger.autonomous_language_output_event_review(
        autonomous_language_output_executor=language_output_execution,
        expected_state_revision=runtime_state.state_revision,
        review_policy={
            "min_confidence_score": 0.5,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
    )

    assert blocked_language_output_event_review["ready"] is False
    assert blocked_language_output_event_review["requires_operator_approval"] is False
    assert blocked_language_output_event_review["autonomous_language_output_event_review"][
        "event_recorded_in_ledger"
    ] is False
    _assert_record_family_source_window(
        blocked_language_output_event_review["source_window"],
        field="autonomous_language_output_events",
        expected_count=1,
    )
    assert language_output_event_review["surface"] == (
        "snn_language_autonomous_language_output_event_review.v1"
    )
    assert language_output_event_review["ready"] is True
    assert language_output_event_review["accepted"] is True
    assert language_output_event_review["requires_operator_approval"] is False
    assert language_output_event_review["advisory"] is True
    assert language_output_event_review["executable"] is False
    assert language_output_event_review["records_ledger_event"] is False
    assert language_output_event_review["mutates_runtime_state"] is False
    assert language_output_event_review["runs_replay"] is False
    assert language_output_event_review["writes_checkpoint"] is False
    assert language_output_event_review["generates_text"] is False
    assert language_output_event_review["decodes_text"] is False
    assert language_output_event_review["trains_runtime_model"] is False
    assert language_output_event_review["applies_plasticity"] is False
    language_output_event_review_body = language_output_event_review[
        "autonomous_language_output_event_review"
    ]
    assert language_output_event_review_body["event_recorded_in_ledger"] is True
    assert language_output_event_review_body["output_slot_count"] == 3
    assert language_output_event_review_body["mean_confidence_score"] == 0.81
    assert language_output_event_review_body["mean_spike_sparsity"] == 0.74
    assert language_output_event_review_body["max_slot_drift"] == 0.03
    assert language_output_event_review_body["output_is_hash_only"] is True
    assert language_output_event_review_body["decoded_text_allowed"] is False
    assert language_output_event_review_body["generated_text_allowed"] is False
    assert language_output_event_review_body["operator_approval_required"] is False
    assert language_output_event_review_body["mutation_allowed"] is False
    _assert_record_family_source_window(
        language_output_event_review["source_window"],
        field="autonomous_language_output_events",
        expected_count=1,
    )
    assert language_output_event_review["promotion_gate"][
        "eligible_for_autonomous_decoded_output_design"
    ] is True
    assert language_output_event_review["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert language_output_event_review["promotion_gate"][
        "eligible_for_dense_readout_training"
    ] is False
    assert language_output_event_review["promotion_gate"]["eligible_for_replay_memory"] is False
    assert language_output_event_review["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert language_output_event_review["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert language_output_event_review["promotion_gate"]["eligible_for_action"] is False
    blocked_decoded_output_design = ledger.autonomous_decoded_output_design(
        autonomous_language_output_event_review=language_output_event_review,
        vocabulary_binding={
            "token_candidate_hashes": [],
            "token_vocabulary_hash": "",
            "tokenizer_hash": "",
            "decode_constraint_hash": "",
        },
        decode_policy={
            "decode_mode": "constrained_token_hash_map",
            "max_decoded_tokens": 3,
            "min_confidence_score": 0.5,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
        device_evidence={"device": "cpu", "source": "unit"},
    )
    decoded_output_design = ledger.autonomous_decoded_output_design(
        autonomous_language_output_event_review=language_output_event_review,
        vocabulary_binding={
            "token_candidate_hashes": [
                ledger._sha256_json(["token-candidate", index])
                for index in range(3)
            ],
            "token_vocabulary_hash": ledger._sha256_json(["vocabulary", "unit"]),
            "tokenizer_hash": ledger._sha256_json(["tokenizer", "unit"]),
            "decode_constraint_hash": ledger._sha256_json(["constraint", "unit"]),
        },
        decode_policy={
            "decode_mode": "constrained_token_hash_map",
            "max_decoded_tokens": 3,
            "min_confidence_score": 0.5,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
        device_evidence={"device": "cpu", "source": "unit"},
    )

    assert blocked_decoded_output_design["ready"] is False
    assert blocked_decoded_output_design["requires_operator_approval"] is False
    assert blocked_decoded_output_design["promotion_gate"]["required_evidence"][
        "token_candidate_hashes_valid"
    ] is False
    assert decoded_output_design["surface"] == (
        "snn_language_autonomous_decoded_output_design.v1"
    )
    assert decoded_output_design["ready"] is True
    assert decoded_output_design["requires_operator_approval"] is False
    assert decoded_output_design["advisory"] is True
    assert decoded_output_design["executable"] is False
    assert decoded_output_design["records_ledger_event"] is False
    assert decoded_output_design["mutates_runtime_state"] is False
    assert decoded_output_design["runs_replay"] is False
    assert decoded_output_design["writes_checkpoint"] is False
    assert decoded_output_design["generates_text"] is False
    assert decoded_output_design["decodes_text"] is False
    assert decoded_output_design["trains_runtime_model"] is False
    assert decoded_output_design["applies_plasticity"] is False
    decoded_output_body = decoded_output_design["autonomous_decoded_output_design"]
    assert decoded_output_body["decode_mode"] == "constrained_token_hash_map"
    assert decoded_output_body["max_decoded_tokens"] == 3
    assert len(decoded_output_body["emitted_hashes"]) == 3
    assert len(decoded_output_body["decode_slots"]) == 3
    assert len(decoded_output_body["token_candidate_hashes"]) == 3
    assert decoded_output_body["mean_confidence_score"] == 0.81
    assert decoded_output_body["mean_spike_sparsity"] == 0.74
    assert decoded_output_body["max_slot_drift"] == 0.03
    assert decoded_output_body["decoded_text_allowed"] is False
    assert decoded_output_body["generated_text_allowed"] is False
    assert decoded_output_body["operator_approval_required"] is False
    assert decoded_output_design["promotion_gate"][
        "eligible_for_autonomous_decoded_output_preflight"
    ] is True
    assert decoded_output_design["promotion_gate"]["eligible_for_language_generation"] is False
    assert decoded_output_design["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert decoded_output_design["promotion_gate"]["eligible_for_replay_memory"] is False
    assert decoded_output_design["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert decoded_output_design["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert decoded_output_design["promotion_gate"]["eligible_for_action"] is False
    blocked_decoded_output_preflight = ledger.autonomous_decoded_output_preflight(
        autonomous_decoded_output_design=decoded_output_design,
        expected_state_revision=runtime_state.state_revision,
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"autonomous_decoded_output_executor": False},
    )
    decoded_output_preflight = ledger.autonomous_decoded_output_preflight(
        autonomous_decoded_output_design=decoded_output_design,
        expected_state_revision=runtime_state.state_revision,
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"autonomous_decoded_output_executor": True},
    )

    assert blocked_decoded_output_preflight["ready"] is False
    assert blocked_decoded_output_preflight["requires_operator_approval"] is False
    assert blocked_decoded_output_preflight["promotion_gate"]["required_evidence"][
        "executor_capability_available"
    ] is False
    assert decoded_output_preflight["surface"] == (
        "snn_language_autonomous_decoded_output_preflight.v1"
    )
    assert decoded_output_preflight["ready"] is True
    assert decoded_output_preflight["requires_operator_approval"] is False
    assert decoded_output_preflight["advisory"] is True
    assert decoded_output_preflight["executable"] is False
    assert decoded_output_preflight["records_ledger_event"] is False
    assert decoded_output_preflight["mutates_runtime_state"] is False
    assert decoded_output_preflight["runs_replay"] is False
    assert decoded_output_preflight["writes_checkpoint"] is False
    assert decoded_output_preflight["generates_text"] is False
    assert decoded_output_preflight["decodes_text"] is False
    assert decoded_output_preflight["trains_runtime_model"] is False
    assert decoded_output_preflight["applies_plasticity"] is False
    decoded_output_preflight_body = decoded_output_preflight[
        "autonomous_decoded_output_preflight"
    ]
    assert decoded_output_preflight_body["decode_mode"] == "constrained_token_hash_map"
    assert decoded_output_preflight_body["max_decoded_tokens"] == 3
    assert len(decoded_output_preflight_body["emitted_hashes"]) == 3
    assert len(decoded_output_preflight_body["decoded_output_slot_hashes"]) == 3
    assert len(decoded_output_preflight_body["token_candidate_hashes"]) == 3
    assert decoded_output_preflight_body["mean_confidence_score"] == 0.81
    assert decoded_output_preflight_body["mean_spike_sparsity"] == 0.74
    assert decoded_output_preflight_body["max_slot_drift"] == 0.03
    assert decoded_output_preflight_body["decoded_text_allowed"] is False
    assert decoded_output_preflight_body["generated_text_allowed"] is False
    assert decoded_output_preflight_body["operator_approval_required"] is False
    assert decoded_output_preflight["promotion_gate"][
        "eligible_for_autonomous_decoded_output_executor"
    ] is True
    assert decoded_output_preflight["promotion_gate"]["eligible_for_language_generation"] is False
    assert decoded_output_preflight["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert decoded_output_preflight["promotion_gate"]["eligible_for_replay_memory"] is False
    assert decoded_output_preflight["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert decoded_output_preflight["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert decoded_output_preflight["promotion_gate"]["eligible_for_action"] is False
    decoded_token_results = [
        {
            "decoded_output_slot_hash": slot_hash,
            "token_candidate_hash": token_candidate_hash,
            "decoded_token_hash": ledger._sha256_json(
                ["decoded-token", token_candidate_hash, index]
            ),
            "token_id_hash": ledger._sha256_json(["token-id", index]),
            "constraint_state_hash": ledger._sha256_json(
                ["constraint-state", index]
            ),
            "constraint_valid": True,
            "confidence_score": 0.79,
            "spike_sparsity": 0.72,
            "slot_drift": 0.02,
        }
        for index, (slot_hash, token_candidate_hash) in enumerate(
            zip(
                decoded_output_preflight_body["decoded_output_slot_hashes"],
                decoded_output_preflight_body["token_candidate_hashes"],
            )
        )
    ]
    blocked_decoded_output_execution = ledger.execute_autonomous_decoded_output(
        autonomous_decoded_output_preflight=decoded_output_preflight,
        expected_state_revision=runtime_state.state_revision,
        decode_evidence={
            "decoded_token_results": decoded_token_results,
            "decoded_text": "blocked text",
            "checkpoint_written": False,
        },
        execution_policy={
            "min_confidence_score": 0.5,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
    )
    decoded_output_before_revision = runtime_state.state_revision
    decoded_output_execution = ledger.execute_autonomous_decoded_output(
        autonomous_decoded_output_preflight=decoded_output_preflight,
        expected_state_revision=decoded_output_before_revision,
        decode_evidence={
            "decoded_token_results": decoded_token_results,
            "checkpoint_written": False,
        },
        execution_policy={
            "min_confidence_score": 0.5,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
    )

    assert blocked_decoded_output_execution["accepted"] is False
    assert blocked_decoded_output_execution["records_ledger_event"] is False
    assert blocked_decoded_output_execution["mutates_runtime_state"] is False
    assert blocked_decoded_output_execution["promotion_gate"]["required_evidence"][
        "decoded_text_absent"
    ] is False
    _assert_record_family_source_window(
        blocked_decoded_output_execution["source_window"],
        field="autonomous_decoded_output_events",
        expected_count=0,
    )
    assert decoded_output_execution["surface"] == (
        "snn_language_autonomous_decoded_output_executor.v1"
    )
    assert decoded_output_execution["accepted"] is True
    assert decoded_output_execution["ready"] is True
    assert decoded_output_execution["requires_operator_approval"] is False
    assert decoded_output_execution["records_ledger_event"] is True
    assert decoded_output_execution["mutates_runtime_state"] is True
    assert decoded_output_execution["after"]["state_revision"] == (
        decoded_output_before_revision + 1
    )
    assert decoded_output_execution["runs_replay"] is False
    assert decoded_output_execution["runs_calibration_update"] is False
    assert decoded_output_execution["writes_checkpoint"] is False
    assert decoded_output_execution["generates_text"] is False
    assert decoded_output_execution["decodes_text"] is False
    assert decoded_output_execution["trains_runtime_model"] is False
    assert decoded_output_execution["applies_plasticity"] is False
    decoded_output_event = decoded_output_execution[
        "autonomous_decoded_output_event"
    ]
    assert decoded_output_event["operator_approval_required"] is False
    assert decoded_output_event["output_is_hash_only"] is True
    assert decoded_output_event["decoded_token_count"] == 3
    assert decoded_output_event["mean_confidence_score"] == 0.79
    assert decoded_output_event["mean_spike_sparsity"] == 0.72
    assert decoded_output_event["max_slot_drift"] == 0.02
    assert decoded_output_execution["ledger_summary"][
        "total_autonomous_decoded_output_count"
    ] == 1
    _assert_record_family_source_window(
        decoded_output_execution["source_window"],
        field="autonomous_decoded_output_events",
        expected_count=0,
    )
    assert decoded_output_execution["promotion_gate"][
        "eligible_for_autonomous_decoded_output_event_review"
    ] is True
    assert decoded_output_execution["promotion_gate"]["eligible_for_language_generation"] is False
    assert decoded_output_execution["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert decoded_output_execution["promotion_gate"]["eligible_for_replay_memory"] is False
    assert decoded_output_execution["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert decoded_output_execution["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert decoded_output_execution["promotion_gate"]["eligible_for_action"] is False
    blocked_decoded_output_event_review = ledger.autonomous_decoded_output_event_review(
        autonomous_decoded_output_executor={
            **decoded_output_execution,
            "autonomous_decoded_output_event": {
                **decoded_output_execution["autonomous_decoded_output_event"],
                "autonomous_decoded_output_event_hash": "0" * 64,
            },
        },
        expected_state_revision=runtime_state.state_revision,
        review_policy={
            "min_confidence_score": 0.5,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
    )
    decoded_output_event_review = ledger.autonomous_decoded_output_event_review(
        autonomous_decoded_output_executor=decoded_output_execution,
        expected_state_revision=runtime_state.state_revision,
        review_policy={
            "min_confidence_score": 0.5,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
    )

    assert blocked_decoded_output_event_review["ready"] is False
    assert blocked_decoded_output_event_review["requires_operator_approval"] is False
    assert blocked_decoded_output_event_review["autonomous_decoded_output_event_review"][
        "event_recorded_in_ledger"
    ] is False
    _assert_record_family_source_window(
        blocked_decoded_output_event_review["source_window"],
        field="autonomous_decoded_output_events",
        expected_count=1,
    )
    assert decoded_output_event_review["surface"] == (
        "snn_language_autonomous_decoded_output_event_review.v1"
    )
    assert decoded_output_event_review["ready"] is True
    assert decoded_output_event_review["accepted"] is True
    assert decoded_output_event_review["requires_operator_approval"] is False
    assert decoded_output_event_review["advisory"] is True
    assert decoded_output_event_review["executable"] is False
    assert decoded_output_event_review["records_ledger_event"] is False
    assert decoded_output_event_review["mutates_runtime_state"] is False
    assert decoded_output_event_review["runs_replay"] is False
    assert decoded_output_event_review["writes_checkpoint"] is False
    assert decoded_output_event_review["generates_text"] is False
    assert decoded_output_event_review["decodes_text"] is False
    assert decoded_output_event_review["trains_runtime_model"] is False
    assert decoded_output_event_review["applies_plasticity"] is False
    decoded_output_event_review_body = decoded_output_event_review[
        "autonomous_decoded_output_event_review"
    ]
    assert decoded_output_event_review_body["event_recorded_in_ledger"] is True
    assert decoded_output_event_review_body["decoded_token_count"] == 3
    assert decoded_output_event_review_body["mean_confidence_score"] == 0.79
    assert decoded_output_event_review_body["mean_spike_sparsity"] == 0.72
    assert decoded_output_event_review_body["max_slot_drift"] == 0.02
    assert decoded_output_event_review_body["output_is_hash_only"] is True
    assert decoded_output_event_review_body["decoded_text_allowed"] is False
    assert decoded_output_event_review_body["generated_text_allowed"] is False
    assert decoded_output_event_review_body["operator_approval_required"] is False
    assert decoded_output_event_review_body["mutation_allowed"] is False
    _assert_record_family_source_window(
        decoded_output_event_review["source_window"],
        field="autonomous_decoded_output_events",
        expected_count=1,
    )
    assert decoded_output_event_review["promotion_gate"][
        "eligible_for_autonomous_bounded_text_emission_design"
    ] is True
    assert decoded_output_event_review["promotion_gate"]["eligible_for_language_generation"] is False
    assert decoded_output_event_review["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert decoded_output_event_review["promotion_gate"]["eligible_for_replay_memory"] is False
    assert decoded_output_event_review["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert decoded_output_event_review["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert decoded_output_event_review["promotion_gate"]["eligible_for_action"] is False
    bounded_text_fragments = [
        "spike trace stable",
        "hash surface committed",
        "bounded output ready",
    ]
    text_surface_schema_hash = ledger._sha256_json(
        ["text-surface-schema", "unit"]
    )
    text_normalizer_hash = ledger._sha256_json(["text-normalizer", "unit"])
    semantic_constraint_hash = ledger._sha256_json(
        ["semantic-constraint", "unit"]
    )
    bounded_text_fragment_hashes = [
        ledger._sha256_json(
            {
                "surface": "snn_language_bounded_text_fragment.v1",
                "text": value,
                "text_surface_schema_hash": text_surface_schema_hash,
                "text_normalizer_hash": text_normalizer_hash,
                "semantic_constraint_hash": semantic_constraint_hash,
            }
        )
        for value in bounded_text_fragments
    ]
    blocked_text_emission_design = ledger.autonomous_bounded_text_emission_design(
        autonomous_decoded_output_event_review=decoded_output_event_review,
        text_surface_binding={
            "text_fragment_hashes": [],
            "text_surface_schema_hash": "",
            "text_normalizer_hash": "",
            "semantic_constraint_hash": "",
        },
        emission_policy={
            "emission_mode": "bounded_text_hash_sequence",
            "max_text_fragments": 3,
            "min_confidence_score": 0.5,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
        device_evidence={"device": "cpu", "source": "unit"},
    )
    text_emission_design = ledger.autonomous_bounded_text_emission_design(
        autonomous_decoded_output_event_review=decoded_output_event_review,
        text_surface_binding={
            "text_fragment_hashes": bounded_text_fragment_hashes,
            "text_surface_schema_hash": text_surface_schema_hash,
            "text_normalizer_hash": text_normalizer_hash,
            "semantic_constraint_hash": semantic_constraint_hash,
        },
        emission_policy={
            "emission_mode": "bounded_text_hash_sequence",
            "max_text_fragments": 3,
            "min_confidence_score": 0.5,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
        device_evidence={"device": "cpu", "source": "unit"},
    )

    assert blocked_text_emission_design["ready"] is False
    assert blocked_text_emission_design["requires_operator_approval"] is False
    assert blocked_text_emission_design["promotion_gate"]["required_evidence"][
        "text_fragment_hashes_valid"
    ] is False
    assert text_emission_design["surface"] == (
        "snn_language_autonomous_bounded_text_emission_design.v1"
    )
    assert text_emission_design["ready"] is True
    assert text_emission_design["requires_operator_approval"] is False
    assert text_emission_design["advisory"] is True
    assert text_emission_design["executable"] is False
    assert text_emission_design["records_ledger_event"] is False
    assert text_emission_design["mutates_runtime_state"] is False
    assert text_emission_design["runs_replay"] is False
    assert text_emission_design["writes_checkpoint"] is False
    assert text_emission_design["generates_text"] is False
    assert text_emission_design["decodes_text"] is False
    assert text_emission_design["trains_runtime_model"] is False
    assert text_emission_design["applies_plasticity"] is False
    text_emission_body = text_emission_design[
        "autonomous_bounded_text_emission_design"
    ]
    assert text_emission_body["emission_mode"] == "bounded_text_hash_sequence"
    assert text_emission_body["max_text_fragments"] == 3
    assert len(text_emission_body["decoded_token_hashes"]) == 3
    assert len(text_emission_body["text_fragment_hashes"]) == 3
    assert len(text_emission_body["text_emission_slots"]) == 3
    assert text_emission_body["mean_confidence_score"] == 0.79
    assert text_emission_body["mean_spike_sparsity"] == 0.72
    assert text_emission_body["max_slot_drift"] == 0.02
    assert text_emission_body["decoded_text_allowed"] is False
    assert text_emission_body["generated_text_allowed"] is False
    assert text_emission_body["literal_text_returned"] is False
    assert text_emission_body["operator_approval_required"] is False
    assert text_emission_design["promotion_gate"][
        "eligible_for_autonomous_bounded_text_emission_preflight"
    ] is True
    assert text_emission_design["promotion_gate"]["eligible_for_language_generation"] is False
    assert text_emission_design["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert text_emission_design["promotion_gate"]["eligible_for_replay_memory"] is False
    assert text_emission_design["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert text_emission_design["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert text_emission_design["promotion_gate"]["eligible_for_action"] is False
    blocked_text_emission_preflight = ledger.autonomous_bounded_text_emission_preflight(
        autonomous_bounded_text_emission_design=text_emission_design,
        expected_state_revision=runtime_state.state_revision,
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"autonomous_bounded_text_emission_executor": False},
    )
    text_emission_preflight = ledger.autonomous_bounded_text_emission_preflight(
        autonomous_bounded_text_emission_design=text_emission_design,
        expected_state_revision=runtime_state.state_revision,
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"autonomous_bounded_text_emission_executor": True},
    )

    assert blocked_text_emission_preflight["ready"] is False
    assert blocked_text_emission_preflight["requires_operator_approval"] is False
    assert blocked_text_emission_preflight["promotion_gate"]["required_evidence"][
        "executor_capability_available"
    ] is False
    assert text_emission_preflight["surface"] == (
        "snn_language_autonomous_bounded_text_emission_preflight.v1"
    )
    assert text_emission_preflight["ready"] is True
    assert text_emission_preflight["requires_operator_approval"] is False
    assert text_emission_preflight["advisory"] is True
    assert text_emission_preflight["executable"] is False
    assert text_emission_preflight["records_ledger_event"] is False
    assert text_emission_preflight["mutates_runtime_state"] is False
    assert text_emission_preflight["runs_replay"] is False
    assert text_emission_preflight["writes_checkpoint"] is False
    assert text_emission_preflight["generates_text"] is False
    assert text_emission_preflight["decodes_text"] is False
    assert text_emission_preflight["trains_runtime_model"] is False
    assert text_emission_preflight["applies_plasticity"] is False
    text_emission_preflight_body = text_emission_preflight[
        "autonomous_bounded_text_emission_preflight"
    ]
    assert text_emission_preflight_body["emission_mode"] == "bounded_text_hash_sequence"
    assert text_emission_preflight_body["max_text_fragments"] == 3
    assert len(text_emission_preflight_body["decoded_token_hashes"]) == 3
    assert len(text_emission_preflight_body["text_fragment_hashes"]) == 3
    assert len(text_emission_preflight_body["text_emission_slot_hashes"]) == 3
    assert text_emission_preflight_body["mean_confidence_score"] == 0.79
    assert text_emission_preflight_body["mean_spike_sparsity"] == 0.72
    assert text_emission_preflight_body["max_slot_drift"] == 0.02
    assert text_emission_preflight_body["decoded_text_allowed"] is False
    assert text_emission_preflight_body["generated_text_allowed"] is False
    assert text_emission_preflight_body["literal_text_returned"] is False
    assert text_emission_preflight_body["operator_approval_required"] is False
    assert text_emission_preflight["promotion_gate"][
        "eligible_for_autonomous_bounded_text_emission_executor"
    ] is True
    assert text_emission_preflight["promotion_gate"]["eligible_for_language_generation"] is False
    assert text_emission_preflight["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert text_emission_preflight["promotion_gate"]["eligible_for_replay_memory"] is False
    assert text_emission_preflight["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert text_emission_preflight["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert text_emission_preflight["promotion_gate"]["eligible_for_action"] is False
    blocked_text_emission_executor = ledger.execute_autonomous_bounded_text_emission(
        autonomous_bounded_text_emission_preflight=text_emission_preflight,
        expected_state_revision=runtime_state.state_revision,
        emission_evidence={
            "text_emission_results": [],
            "checkpoint_written": False,
        },
        execution_policy={
            "min_confidence_score": 0.5,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
    )
    text_emission_executor = ledger.execute_autonomous_bounded_text_emission(
        autonomous_bounded_text_emission_preflight=text_emission_preflight,
        expected_state_revision=runtime_state.state_revision,
        emission_evidence={
            "text_emission_results": [
                {
                    "bounded_text_emission_slot_hash": slot_hash,
                    "decoded_token_hash": decoded_hash,
                    "text_fragment_hash": fragment_hash,
                    "text_surface_schema_hash": text_emission_preflight_body[
                        "text_surface_schema_hash"
                    ],
                    "text_normalizer_hash": text_emission_preflight_body[
                        "text_normalizer_hash"
                    ],
                    "semantic_constraint_hash": text_emission_preflight_body[
                        "semantic_constraint_hash"
                    ],
                    "semantic_constraint_valid": True,
                    "text_normalized": True,
                    "confidence_score": 0.79,
                    "spike_sparsity": 0.72,
                    "slot_drift": 0.02,
                }
                for slot_hash, decoded_hash, fragment_hash in zip(
                    text_emission_preflight_body["text_emission_slot_hashes"],
                    text_emission_preflight_body["decoded_token_hashes"],
                    text_emission_preflight_body["text_fragment_hashes"],
                )
            ],
            "checkpoint_written": False,
        },
        execution_policy={
            "min_confidence_score": 0.5,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
    )

    assert blocked_text_emission_executor["accepted"] is False
    assert blocked_text_emission_executor["requires_operator_approval"] is False
    assert blocked_text_emission_executor["promotion_gate"]["required_evidence"][
        "text_emission_results_bounded"
    ] is False
    _assert_record_family_source_window(
        blocked_text_emission_executor["source_window"],
        field="autonomous_bounded_text_emission_events",
        expected_count=0,
    )
    assert text_emission_executor["surface"] == (
        "snn_language_autonomous_bounded_text_emission_executor.v1"
    )
    assert text_emission_executor["accepted"] is True
    assert text_emission_executor["ready"] is True
    assert text_emission_executor["requires_operator_approval"] is False
    assert text_emission_executor["advisory"] is False
    assert text_emission_executor["executable"] is True
    assert text_emission_executor["records_ledger_event"] is True
    assert text_emission_executor["mutates_runtime_state"] is True
    assert text_emission_executor["runs_replay"] is False
    assert text_emission_executor["writes_checkpoint"] is False
    assert text_emission_executor["generates_text"] is False
    assert text_emission_executor["decodes_text"] is False
    assert text_emission_executor["trains_runtime_model"] is False
    assert text_emission_executor["applies_plasticity"] is False
    text_emission_event = text_emission_executor[
        "autonomous_bounded_text_emission_event"
    ]
    assert text_emission_event["text_fragment_count"] == 3
    assert len(text_emission_event["decoded_token_hashes"]) == 3
    assert len(text_emission_event["text_fragment_hashes"]) == 3
    assert len(text_emission_event["text_emission_slot_hashes"]) == 3
    assert text_emission_event["mean_confidence_score"] == 0.79
    assert text_emission_event["mean_spike_sparsity"] == 0.72
    assert text_emission_event["max_slot_drift"] == 0.02
    assert text_emission_event["output_is_hash_only"] is True
    assert text_emission_event["literal_text_returned"] is False
    assert text_emission_event["operator_approval_required"] is False
    _assert_record_family_source_window(
        text_emission_executor["source_window"],
        field="autonomous_bounded_text_emission_events",
        expected_count=0,
    )
    assert text_emission_executor["promotion_gate"][
        "eligible_for_autonomous_bounded_text_emission_event_review"
    ] is True
    assert text_emission_executor["promotion_gate"]["eligible_for_language_generation"] is False
    assert text_emission_executor["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert text_emission_executor["promotion_gate"]["eligible_for_replay_memory"] is False
    assert text_emission_executor["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert text_emission_executor["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert text_emission_executor["promotion_gate"]["eligible_for_action"] is False
    blocked_text_emission_event_review = (
        ledger.autonomous_bounded_text_emission_event_review(
            autonomous_bounded_text_emission_executor=blocked_text_emission_executor,
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "min_text_fragments": 1,
                "max_text_fragments": 3,
                "min_confidence_score": 0.5,
                "min_spike_sparsity": 0.5,
                "max_slot_drift": 0.2,
            },
        )
    )
    text_emission_event_review = ledger.autonomous_bounded_text_emission_event_review(
        autonomous_bounded_text_emission_executor=text_emission_executor,
        expected_state_revision=runtime_state.state_revision,
        review_policy={
            "min_text_fragments": 1,
            "max_text_fragments": 3,
            "min_confidence_score": 0.5,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
    )

    assert blocked_text_emission_event_review["ready"] is False
    assert blocked_text_emission_event_review["requires_operator_approval"] is False
    assert blocked_text_emission_event_review["promotion_gate"]["required_evidence"][
        "executor_accepted"
    ] is False
    _assert_record_family_source_window(
        blocked_text_emission_event_review["source_window"],
        field="autonomous_bounded_text_emission_events",
        expected_count=1,
    )
    assert text_emission_event_review["surface"] == (
        "snn_language_autonomous_bounded_text_emission_event_review.v1"
    )
    assert text_emission_event_review["ready"] is True
    assert text_emission_event_review["accepted"] is True
    assert text_emission_event_review["requires_operator_approval"] is False
    assert text_emission_event_review["advisory"] is True
    assert text_emission_event_review["executable"] is False
    assert text_emission_event_review["records_ledger_event"] is False
    assert text_emission_event_review["mutates_runtime_state"] is False
    assert text_emission_event_review["runs_replay"] is False
    assert text_emission_event_review["writes_checkpoint"] is False
    assert text_emission_event_review["generates_text"] is False
    assert text_emission_event_review["decodes_text"] is False
    assert text_emission_event_review["trains_runtime_model"] is False
    assert text_emission_event_review["applies_plasticity"] is False
    text_emission_review_body = text_emission_event_review[
        "autonomous_bounded_text_emission_event_review"
    ]
    assert text_emission_review_body["event_recorded_in_ledger"] is True
    assert text_emission_review_body["text_fragment_count"] == 3
    assert len(text_emission_review_body["decoded_token_hashes"]) == 3
    assert len(text_emission_review_body["text_fragment_hashes"]) == 3
    assert len(text_emission_review_body["text_emission_slot_hashes"]) == 3
    assert text_emission_review_body["mean_confidence_score"] == 0.79
    assert text_emission_review_body["mean_spike_sparsity"] == 0.72
    assert text_emission_review_body["max_slot_drift"] == 0.02
    assert text_emission_review_body["output_is_hash_only"] is True
    assert text_emission_review_body["literal_text_returned"] is False
    assert text_emission_review_body["operator_approval_required"] is False
    _assert_record_family_source_window(
        text_emission_event_review["source_window"],
        field="autonomous_bounded_text_emission_events",
        expected_count=1,
    )
    assert text_emission_event_review["promotion_gate"][
        "eligible_for_autonomous_text_surface_sequence_review"
    ] is True
    assert text_emission_event_review["promotion_gate"]["eligible_for_language_generation"] is False
    assert text_emission_event_review["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert text_emission_event_review["promotion_gate"]["eligible_for_replay_memory"] is False
    assert text_emission_event_review["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert text_emission_event_review["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert text_emission_event_review["promotion_gate"]["eligible_for_action"] is False
    blocked_text_surface_sequence_review = (
        ledger.autonomous_text_surface_sequence_review(
            autonomous_bounded_text_emission_event_review=(
                blocked_text_emission_event_review
            ),
            sequence_policy={
                "sequence_mode": "bounded_hash_fragment_sequence",
                "min_text_fragments": 1,
                "max_text_fragments": 3,
                "min_confidence_score": 0.5,
                "min_spike_sparsity": 0.5,
                "max_slot_drift": 0.2,
            },
        )
    )
    text_surface_sequence_review = ledger.autonomous_text_surface_sequence_review(
        autonomous_bounded_text_emission_event_review=text_emission_event_review,
        sequence_policy={
            "sequence_mode": "bounded_hash_fragment_sequence",
            "min_text_fragments": 1,
            "max_text_fragments": 3,
            "min_confidence_score": 0.5,
            "min_spike_sparsity": 0.5,
            "max_slot_drift": 0.2,
        },
    )

    assert blocked_text_surface_sequence_review["ready"] is False
    assert blocked_text_surface_sequence_review["requires_operator_approval"] is False
    assert blocked_text_surface_sequence_review["promotion_gate"]["required_evidence"][
        "event_review_ready"
    ] is False
    assert text_surface_sequence_review["surface"] == (
        "snn_language_autonomous_text_surface_sequence_review.v1"
    )
    assert text_surface_sequence_review["ready"] is True
    assert text_surface_sequence_review["accepted"] is True
    assert text_surface_sequence_review["requires_operator_approval"] is False
    assert text_surface_sequence_review["advisory"] is True
    assert text_surface_sequence_review["executable"] is False
    assert text_surface_sequence_review["records_ledger_event"] is False
    assert text_surface_sequence_review["mutates_runtime_state"] is False
    assert text_surface_sequence_review["runs_replay"] is False
    assert text_surface_sequence_review["writes_checkpoint"] is False
    assert text_surface_sequence_review["generates_text"] is False
    assert text_surface_sequence_review["decodes_text"] is False
    assert text_surface_sequence_review["trains_runtime_model"] is False
    assert text_surface_sequence_review["applies_plasticity"] is False
    text_surface_sequence_body = text_surface_sequence_review[
        "autonomous_text_surface_sequence_review"
    ]
    assert text_surface_sequence_body["sequence_mode"] == (
        "bounded_hash_fragment_sequence"
    )
    assert text_surface_sequence_body["text_fragment_count"] == 3
    assert len(text_surface_sequence_body["decoded_token_hashes"]) == 3
    assert len(text_surface_sequence_body["text_fragment_hashes"]) == 3
    assert len(text_surface_sequence_body["text_emission_slot_hashes"]) == 3
    assert len(text_surface_sequence_body["fragment_sequence_hash"]) == 64
    assert text_surface_sequence_body["mean_confidence_score"] == 0.79
    assert text_surface_sequence_body["mean_spike_sparsity"] == 0.72
    assert text_surface_sequence_body["max_slot_drift"] == 0.02
    assert text_surface_sequence_body["output_is_hash_only"] is True
    assert text_surface_sequence_body["literal_text_returned"] is False
    assert text_surface_sequence_body["operator_approval_required"] is False
    assert text_surface_sequence_review["promotion_gate"][
        "eligible_for_autonomous_text_surface_commit_design"
    ] is True
    assert text_surface_sequence_review["promotion_gate"]["eligible_for_language_generation"] is False
    assert text_surface_sequence_review["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert text_surface_sequence_review["promotion_gate"]["eligible_for_replay_memory"] is False
    assert text_surface_sequence_review["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert text_surface_sequence_review["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert text_surface_sequence_review["promotion_gate"]["eligible_for_action"] is False
    blocked_text_surface_commit_design = (
        ledger.autonomous_text_surface_commit_design(
            autonomous_text_surface_sequence_review=blocked_text_surface_sequence_review,
            commit_policy={
                "commit_scope": "hash_surface_state",
                "retention_class": "ephemeral_hash_surface",
                "min_text_fragments": 1,
                "max_text_fragments": 3,
            },
        )
    )
    text_surface_commit_design = ledger.autonomous_text_surface_commit_design(
        autonomous_text_surface_sequence_review=text_surface_sequence_review,
        commit_policy={
            "commit_scope": "hash_surface_state",
            "retention_class": "ephemeral_hash_surface",
            "min_text_fragments": 1,
            "max_text_fragments": 3,
        },
    )

    assert blocked_text_surface_commit_design["ready"] is False
    assert blocked_text_surface_commit_design["requires_operator_approval"] is False
    assert blocked_text_surface_commit_design["promotion_gate"]["required_evidence"][
        "sequence_review_ready"
    ] is False
    assert text_surface_commit_design["surface"] == (
        "snn_language_autonomous_text_surface_commit_design.v1"
    )
    assert text_surface_commit_design["ready"] is True
    assert text_surface_commit_design["accepted"] is True
    assert text_surface_commit_design["requires_operator_approval"] is False
    assert text_surface_commit_design["advisory"] is True
    assert text_surface_commit_design["executable"] is False
    assert text_surface_commit_design["records_ledger_event"] is False
    assert text_surface_commit_design["mutates_runtime_state"] is False
    assert text_surface_commit_design["runs_replay"] is False
    assert text_surface_commit_design["writes_checkpoint"] is False
    assert text_surface_commit_design["generates_text"] is False
    assert text_surface_commit_design["decodes_text"] is False
    assert text_surface_commit_design["trains_runtime_model"] is False
    assert text_surface_commit_design["applies_plasticity"] is False
    text_surface_commit_body = text_surface_commit_design[
        "autonomous_text_surface_commit_design"
    ]
    assert text_surface_commit_body["commit_scope"] == "hash_surface_state"
    assert text_surface_commit_body["retention_class"] == "ephemeral_hash_surface"
    assert text_surface_commit_body["text_fragment_count"] == 3
    assert len(text_surface_commit_body["decoded_token_hashes"]) == 3
    assert len(text_surface_commit_body["text_fragment_hashes"]) == 3
    assert len(text_surface_commit_body["text_emission_slot_hashes"]) == 3
    assert len(text_surface_commit_body["fragment_sequence_hash"]) == 64
    assert len(text_surface_commit_body["commit_plan_hash"]) == 64
    assert text_surface_commit_body["mean_confidence_score"] == 0.79
    assert text_surface_commit_body["mean_spike_sparsity"] == 0.72
    assert text_surface_commit_body["max_slot_drift"] == 0.02
    assert text_surface_commit_body["output_is_hash_only"] is True
    assert text_surface_commit_body["literal_text_returned"] is False
    assert text_surface_commit_body["operator_approval_required"] is False
    assert text_surface_commit_design["promotion_gate"][
        "eligible_for_autonomous_text_surface_commit_preflight"
    ] is True
    assert text_surface_commit_design["promotion_gate"]["eligible_for_language_generation"] is False
    assert text_surface_commit_design["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert text_surface_commit_design["promotion_gate"]["eligible_for_replay_memory"] is False
    assert text_surface_commit_design["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert text_surface_commit_design["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert text_surface_commit_design["promotion_gate"]["eligible_for_action"] is False
    blocked_text_surface_commit_preflight = (
        ledger.autonomous_text_surface_commit_preflight(
            autonomous_text_surface_commit_design=text_surface_commit_design,
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cpu", "source": "unit"},
            executor_capabilities={"autonomous_text_surface_commit_executor": False},
        )
    )
    text_surface_commit_preflight = ledger.autonomous_text_surface_commit_preflight(
        autonomous_text_surface_commit_design=text_surface_commit_design,
        expected_state_revision=runtime_state.state_revision,
        device_evidence={"device": "cpu", "source": "unit"},
        executor_capabilities={"autonomous_text_surface_commit_executor": True},
    )

    assert blocked_text_surface_commit_preflight["ready"] is False
    assert blocked_text_surface_commit_preflight["requires_operator_approval"] is False
    assert blocked_text_surface_commit_preflight["promotion_gate"]["required_evidence"][
        "executor_capability_available"
    ] is False
    assert text_surface_commit_preflight["surface"] == (
        "snn_language_autonomous_text_surface_commit_preflight.v1"
    )
    assert text_surface_commit_preflight["ready"] is True
    assert text_surface_commit_preflight["accepted"] is True
    assert text_surface_commit_preflight["requires_operator_approval"] is False
    assert text_surface_commit_preflight["advisory"] is True
    assert text_surface_commit_preflight["executable"] is False
    assert text_surface_commit_preflight["records_ledger_event"] is False
    assert text_surface_commit_preflight["mutates_runtime_state"] is False
    assert text_surface_commit_preflight["runs_replay"] is False
    assert text_surface_commit_preflight["writes_checkpoint"] is False
    assert text_surface_commit_preflight["generates_text"] is False
    assert text_surface_commit_preflight["decodes_text"] is False
    assert text_surface_commit_preflight["trains_runtime_model"] is False
    assert text_surface_commit_preflight["applies_plasticity"] is False
    text_surface_commit_preflight_body = text_surface_commit_preflight[
        "autonomous_text_surface_commit_preflight"
    ]
    assert text_surface_commit_preflight_body["commit_scope"] == "hash_surface_state"
    assert text_surface_commit_preflight_body["retention_class"] == (
        "ephemeral_hash_surface"
    )
    assert text_surface_commit_preflight_body["text_fragment_count"] == 3
    assert len(text_surface_commit_preflight_body["decoded_token_hashes"]) == 3
    assert len(text_surface_commit_preflight_body["text_fragment_hashes"]) == 3
    assert len(text_surface_commit_preflight_body["text_emission_slot_hashes"]) == 3
    assert len(text_surface_commit_preflight_body["fragment_sequence_hash"]) == 64
    assert len(text_surface_commit_preflight["preflight_hash"]) == 64
    assert text_surface_commit_preflight_body["mean_confidence_score"] == 0.79
    assert text_surface_commit_preflight_body["mean_spike_sparsity"] == 0.72
    assert text_surface_commit_preflight_body["max_slot_drift"] == 0.02
    assert text_surface_commit_preflight_body["literal_text_returned"] is False
    assert text_surface_commit_preflight_body["operator_approval_required"] is False
    assert text_surface_commit_preflight["promotion_gate"][
        "eligible_for_autonomous_text_surface_commit_executor"
    ] is True
    assert text_surface_commit_preflight["promotion_gate"]["eligible_for_language_generation"] is False
    assert text_surface_commit_preflight["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert text_surface_commit_preflight["promotion_gate"]["eligible_for_replay_memory"] is False
    assert text_surface_commit_preflight["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert text_surface_commit_preflight["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert text_surface_commit_preflight["promotion_gate"]["eligible_for_action"] is False
    blocked_text_surface_commit_executor = (
        ledger.execute_autonomous_text_surface_commit(
            autonomous_text_surface_commit_preflight=blocked_text_surface_commit_preflight,
            expected_state_revision=runtime_state.state_revision,
            commit_evidence={
                "checkpoint_written": False,
            },
            execution_policy={"max_text_fragments": 3},
        )
    )
    text_surface_commit_executor = ledger.execute_autonomous_text_surface_commit(
        autonomous_text_surface_commit_preflight=text_surface_commit_preflight,
        expected_state_revision=runtime_state.state_revision,
        commit_evidence={
            "committed_surface_hash": text_surface_commit_preflight_body[
                "fragment_sequence_hash"
            ],
            "checkpoint_written": False,
        },
        execution_policy={"max_text_fragments": 3},
    )
    blocked_text_surface_commit_event_review = (
        ledger.autonomous_text_surface_commit_event_review(
            autonomous_text_surface_commit_executor=blocked_text_surface_commit_executor,
            expected_state_revision=runtime_state.state_revision,
            review_policy={"max_text_fragments": 3},
        )
    )
    text_surface_commit_event_review = (
        ledger.autonomous_text_surface_commit_event_review(
            autonomous_text_surface_commit_executor=text_surface_commit_executor,
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "max_text_fragments": 3,
                "min_confidence_score": 0.7,
                "min_spike_sparsity": 0.7,
                "max_slot_drift": 0.05,
            },
        )
    )
    blocked_text_surface_materialization_design = (
        ledger.autonomous_text_surface_materialization_design(
            autonomous_text_surface_commit_event_review=(
                blocked_text_surface_commit_event_review
            ),
            materialization_policy={"max_text_fragments": 3},
        )
    )
    text_surface_materialization_design = (
        ledger.autonomous_text_surface_materialization_design(
            autonomous_text_surface_commit_event_review=text_surface_commit_event_review,
            materialization_policy={
                "max_text_fragments": 3,
                "min_confidence_score": 0.7,
                "min_spike_sparsity": 0.7,
                "max_slot_drift": 0.05,
                "max_surface_chars": 256,
            },
        )
    )
    blocked_text_surface_materialization_preflight = (
        ledger.autonomous_text_surface_materialization_preflight(
            autonomous_text_surface_materialization_design=(
                blocked_text_surface_materialization_design
            ),
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cpu"},
            executor_capabilities={
                "autonomous_text_surface_materialization_executor": False,
            },
        )
    )
    text_surface_materialization_preflight = (
        ledger.autonomous_text_surface_materialization_preflight(
            autonomous_text_surface_materialization_design=(
                text_surface_materialization_design
            ),
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cpu"},
            executor_capabilities={
                "autonomous_text_surface_materialization_executor": True,
            },
        )
    )
    blocked_text_surface_materialization_executor = (
        ledger.execute_autonomous_text_surface_materialization(
            autonomous_text_surface_materialization_preflight=(
                blocked_text_surface_materialization_preflight
            ),
            expected_state_revision=runtime_state.state_revision,
            materialization_evidence={
                "text_fragments": bounded_text_fragments,
                "checkpoint_written": False,
            },
            execution_policy={
                "max_text_fragments": 3,
                "max_surface_chars": 256,
            },
        )
    )
    text_surface_materialization_executor = (
        ledger.execute_autonomous_text_surface_materialization(
            autonomous_text_surface_materialization_preflight=(
                text_surface_materialization_preflight
            ),
            expected_state_revision=runtime_state.state_revision,
            materialization_evidence={
                "text_fragments": bounded_text_fragments,
                "checkpoint_written": False,
            },
            execution_policy={
                "max_text_fragments": 3,
                "max_surface_chars": 256,
            },
        )
    )
    blocked_text_surface_materialization_event_review = (
        ledger.autonomous_text_surface_materialization_event_review(
            autonomous_text_surface_materialization_executor=(
                blocked_text_surface_materialization_executor
            ),
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "max_text_fragments": 3,
                "max_surface_chars": 256,
            },
        )
    )
    text_surface_materialization_event_review = (
        ledger.autonomous_text_surface_materialization_event_review(
            autonomous_text_surface_materialization_executor=(
                text_surface_materialization_executor
            ),
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "max_text_fragments": 3,
                "max_surface_chars": 256,
                "min_confidence_score": 0.7,
                "min_spike_sparsity": 0.7,
                "max_slot_drift": 0.05,
            },
        )
    )
    blocked_bounded_language_surface_review = (
        ledger.autonomous_bounded_language_surface_review(
            autonomous_text_surface_materialization_event_review=(
                blocked_text_surface_materialization_event_review
            ),
            language_surface_policy={
                "max_text_fragments": 3,
                "max_surface_chars": 256,
            },
        )
    )
    bounded_language_surface_review = (
        ledger.autonomous_bounded_language_surface_review(
            autonomous_text_surface_materialization_event_review=(
                text_surface_materialization_event_review
            ),
            language_surface_policy={
                "max_text_fragments": 3,
                "max_surface_chars": 256,
                "min_confidence_score": 0.7,
                "min_spike_sparsity": 0.7,
                "max_slot_drift": 0.05,
            },
        )
    )
    blocked_bounded_language_surface_commit_design = (
        ledger.autonomous_bounded_language_surface_commit_design(
            autonomous_bounded_language_surface_review=(
                blocked_bounded_language_surface_review
            ),
            commit_policy={
                "commit_scope": "bounded_language_surface",
                "retention_class": "ephemeral_language_surface",
                "max_surface_chars": 256,
            },
        )
    )
    bounded_language_surface_commit_design = (
        ledger.autonomous_bounded_language_surface_commit_design(
            autonomous_bounded_language_surface_review=bounded_language_surface_review,
            commit_policy={
                "commit_scope": "bounded_language_surface",
                "retention_class": "ephemeral_language_surface",
                "max_surface_chars": 256,
            },
        )
    )
    blocked_bounded_language_surface_commit_preflight = (
        ledger.autonomous_bounded_language_surface_commit_preflight(
            autonomous_bounded_language_surface_commit_design=(
                bounded_language_surface_commit_design
            ),
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cpu", "source": "unit"},
            executor_capabilities={
                "autonomous_bounded_language_surface_commit_executor": False
            },
        )
    )
    bounded_language_surface_commit_preflight = (
        ledger.autonomous_bounded_language_surface_commit_preflight(
            autonomous_bounded_language_surface_commit_design=(
                bounded_language_surface_commit_design
            ),
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cpu", "source": "unit"},
            executor_capabilities={
                "autonomous_bounded_language_surface_commit_executor": True
            },
        )
    )
    blocked_bounded_language_surface_commit_executor = (
        ledger.execute_autonomous_bounded_language_surface_commit(
            autonomous_bounded_language_surface_commit_preflight=(
                blocked_bounded_language_surface_commit_preflight
            ),
            expected_state_revision=runtime_state.state_revision,
            commit_evidence={"checkpoint_written": False},
            execution_policy={
                "max_text_fragments": 3,
                "max_surface_chars": 256,
            },
        )
    )
    bounded_language_surface_commit_executor = (
        ledger.execute_autonomous_bounded_language_surface_commit(
            autonomous_bounded_language_surface_commit_preflight=(
                bounded_language_surface_commit_preflight
            ),
            expected_state_revision=runtime_state.state_revision,
            commit_evidence={
                "committed_language_surface_hash": (
                    bounded_language_surface_commit_preflight[
                        "bounded_language_surface_hash"
                    ]
                ),
                "checkpoint_written": False,
            },
            execution_policy={
                "max_text_fragments": 3,
                "max_surface_chars": 256,
            },
        )
    )
    blocked_bounded_language_surface_commit_event_review = (
        ledger.autonomous_bounded_language_surface_commit_event_review(
            autonomous_bounded_language_surface_commit_executor=(
                blocked_bounded_language_surface_commit_executor
            ),
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "max_text_fragments": 3,
                "max_surface_chars": 256,
            },
        )
    )
    bounded_language_surface_commit_event_review = (
        ledger.autonomous_bounded_language_surface_commit_event_review(
            autonomous_bounded_language_surface_commit_executor=(
                bounded_language_surface_commit_executor
            ),
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "max_text_fragments": 3,
                "max_surface_chars": 256,
                "min_confidence_score": 0.7,
                "min_spike_sparsity": 0.7,
                "max_slot_drift": 0.05,
            },
        )
    )
    blocked_bounded_language_surface_use_review = (
        ledger.autonomous_bounded_language_surface_use_review(
            autonomous_bounded_language_surface_commit_event_review=(
                blocked_bounded_language_surface_commit_event_review
            ),
            use_policy={
                "language_use_scope": "bounded_language_evidence",
                "max_surface_chars": 256,
            },
        )
    )
    bounded_language_surface_use_review = (
        ledger.autonomous_bounded_language_surface_use_review(
            autonomous_bounded_language_surface_commit_event_review=(
                bounded_language_surface_commit_event_review
            ),
            use_policy={
                "language_use_scope": "bounded_language_evidence",
                "max_surface_chars": 256,
            },
        )
    )
    blocked_bounded_language_surface_use_preflight = (
        ledger.autonomous_bounded_language_surface_use_preflight(
            autonomous_bounded_language_surface_use_review=(
                bounded_language_surface_use_review
            ),
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cpu", "source": "unit"},
            executor_capabilities={
                "autonomous_bounded_language_surface_use_executor": False
            },
        )
    )
    bounded_language_surface_use_preflight = (
        ledger.autonomous_bounded_language_surface_use_preflight(
            autonomous_bounded_language_surface_use_review=(
                bounded_language_surface_use_review
            ),
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cpu", "source": "unit"},
            executor_capabilities={
                "autonomous_bounded_language_surface_use_executor": True
            },
        )
    )
    blocked_bounded_language_surface_use_executor = (
        ledger.execute_autonomous_bounded_language_surface_use(
            autonomous_bounded_language_surface_use_preflight=(
                blocked_bounded_language_surface_use_preflight
            ),
            expected_state_revision=runtime_state.state_revision,
            use_evidence={"checkpoint_written": False},
            execution_policy={
                "max_text_fragments": 3,
                "max_surface_chars": 256,
            },
        )
    )
    bounded_language_surface_use_executor = (
        ledger.execute_autonomous_bounded_language_surface_use(
            autonomous_bounded_language_surface_use_preflight=(
                bounded_language_surface_use_preflight
            ),
            expected_state_revision=runtime_state.state_revision,
            use_evidence={
                "used_language_surface_hash": bounded_language_surface_use_preflight[
                    "bounded_language_surface_hash"
                ],
                "use_mode": "bounded_language_evidence_observation",
                "checkpoint_written": False,
            },
            execution_policy={
                "max_text_fragments": 3,
                "max_surface_chars": 256,
            },
        )
    )
    blocked_bounded_language_surface_use_event_review = (
        ledger.autonomous_bounded_language_surface_use_event_review(
            autonomous_bounded_language_surface_use_executor=(
                blocked_bounded_language_surface_use_executor
            ),
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "max_text_fragments": 3,
                "max_surface_chars": 256,
            },
        )
    )
    bounded_language_surface_use_event_review = (
        ledger.autonomous_bounded_language_surface_use_event_review(
            autonomous_bounded_language_surface_use_executor=(
                bounded_language_surface_use_executor
            ),
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "max_text_fragments": 3,
                "max_surface_chars": 256,
                "min_confidence_score": 0.7,
                "min_spike_sparsity": 0.7,
                "max_slot_drift": 0.05,
            },
        )
    )
    blocked_snn_language_generation_design = (
        ledger.autonomous_snn_language_generation_design(
            autonomous_bounded_language_surface_use_event_review=(
                blocked_bounded_language_surface_use_event_review
            ),
            generation_policy={
                "generation_mode": "snn_bounded_next_token_projection",
                "decoding_strategy": "spike_sparse_top_k",
                "max_new_tokens": 16,
                "max_generated_fragments": 2,
                "target_device": "cpu",
                "requires_cuda": False,
            },
        )
    )
    snn_language_generation_design = (
        ledger.autonomous_snn_language_generation_design(
            autonomous_bounded_language_surface_use_event_review=(
                bounded_language_surface_use_event_review
            ),
            generation_policy={
                "generation_mode": "snn_bounded_next_token_projection",
                "decoding_strategy": "spike_sparse_top_k",
                "max_new_tokens": 16,
                "max_generated_fragments": 2,
                "target_device": "cpu",
                "requires_cuda": False,
                "min_confidence_score": 0.7,
                "min_spike_sparsity": 0.7,
                "max_slot_drift": 0.05,
            },
        )
    )
    blocked_snn_language_generation_preflight = (
        ledger.autonomous_snn_language_generation_preflight(
            autonomous_snn_language_generation_design=(
                blocked_snn_language_generation_design
            ),
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cpu", "source": "unit"},
            executor_capabilities={
                "autonomous_snn_language_generation_executor": False
            },
        )
    )
    snn_language_generation_preflight = (
        ledger.autonomous_snn_language_generation_preflight(
            autonomous_snn_language_generation_design=snn_language_generation_design,
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cpu", "source": "unit"},
            executor_capabilities={
                "autonomous_snn_language_generation_executor": True
            },
        )
    )
    generation_token_hashes = [
        _sha256_json({"generated_token_index": index, "slot": f"slot-{index}"})
        for index in range(2)
    ]
    generation_spike_hashes = [
        _sha256_json({"spike_projection": index, "active": [index, index + 1]})
        for index in range(2)
    ]
    generation_active_hashes = [
        _sha256_json({"active_neurons": [index, index + 2]})
        for index in range(2)
    ]
    generation_membrane_hashes = [
        _sha256_json({"membrane_state": index, "voltage": round(0.1 * index, 2)})
        for index in range(2)
    ]
    blocked_snn_language_generation_executor = (
        ledger.execute_autonomous_snn_language_generation(
            autonomous_snn_language_generation_preflight=(
                blocked_snn_language_generation_preflight
            ),
            expected_state_revision=runtime_state.state_revision,
            generation_evidence={
                "generated_token_hashes": generation_token_hashes,
                "spike_projection_hashes": generation_spike_hashes,
                "active_neuron_hashes": generation_active_hashes,
                "membrane_state_hashes": generation_membrane_hashes,
                "output_fragment_hashes": [
                    _sha256_json({"output_fragment": "hash-only"})
                ],
                "checkpoint_written": False,
            },
            execution_policy={"max_new_tokens": 2},
        )
    )
    snn_language_generation_executor = (
        ledger.execute_autonomous_snn_language_generation(
            autonomous_snn_language_generation_preflight=(
                snn_language_generation_preflight
            ),
            expected_state_revision=runtime_state.state_revision,
            generation_evidence={
                "generated_token_hashes": generation_token_hashes,
                "spike_projection_hashes": generation_spike_hashes,
                "active_neuron_hashes": generation_active_hashes,
                "membrane_state_hashes": generation_membrane_hashes,
                "output_fragment_hashes": [
                    _sha256_json({"output_fragment": "hash-only"})
                ],
                "checkpoint_written": False,
            },
            execution_policy={"max_new_tokens": 2},
        )
    )
    blocked_snn_language_generation_event_review = (
        ledger.autonomous_snn_language_generation_event_review(
            autonomous_snn_language_generation_executor=(
                blocked_snn_language_generation_executor
            ),
            expected_state_revision=runtime_state.state_revision,
            review_policy={"max_generated_tokens": 2},
        )
    )
    snn_language_generation_event_review = (
        ledger.autonomous_snn_language_generation_event_review(
            autonomous_snn_language_generation_executor=(
                snn_language_generation_executor
            ),
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "min_generated_tokens": 1,
                "max_generated_tokens": 2,
            },
        )
    )
    blocked_snn_language_decoding_design = (
        ledger.autonomous_snn_language_decoding_design(
            autonomous_snn_language_generation_event_review=(
                blocked_snn_language_generation_event_review
            ),
            decoding_policy={
                "decoding_mode": "bounded_hash_token_projection",
                "materialization_target": "bounded_text_surface",
                "max_decoded_tokens": 2,
                "max_decoded_fragments": 1,
                "max_surface_chars": 256,
            },
        )
    )
    snn_language_decoding_design = (
        ledger.autonomous_snn_language_decoding_design(
            autonomous_snn_language_generation_event_review=(
                snn_language_generation_event_review
            ),
            decoding_policy={
                "decoding_mode": "bounded_hash_token_projection",
                "materialization_target": "bounded_text_surface",
                "max_decoded_tokens": 2,
                "max_decoded_fragments": 1,
                "max_surface_chars": 256,
            },
        )
    )
    blocked_snn_language_decoding_preflight = (
        ledger.autonomous_snn_language_decoding_preflight(
            autonomous_snn_language_decoding_design=(
                blocked_snn_language_decoding_design
            ),
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cuda:0", "cuda_available": True},
            decoder_capabilities={
                "autonomous_snn_language_decoding_executor": True
            },
        )
    )
    snn_language_decoding_preflight = (
        ledger.autonomous_snn_language_decoding_preflight(
            autonomous_snn_language_decoding_design=snn_language_decoding_design,
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cuda:0", "cuda_available": True},
            decoder_capabilities={
                "autonomous_snn_language_decoding_executor": True
            },
        )
    )
    blocked_snn_language_decoding_executor = (
        ledger.execute_autonomous_snn_language_decoding(
            autonomous_snn_language_decoding_preflight=(
                blocked_snn_language_decoding_preflight
            ),
            expected_state_revision=runtime_state.state_revision,
            decoding_evidence={
                "decoded_token_hashes": generation_token_hashes,
                "decoded_text_fragments": ["spike language"],
                "rendered_text": "spike language",
                "schema_valid": True,
                "text_normalized": True,
                "semantic_constraint_valid": True,
                "checkpoint_written": False,
            },
        )
    )
    snn_language_decoding_executor = (
        ledger.execute_autonomous_snn_language_decoding(
            autonomous_snn_language_decoding_preflight=snn_language_decoding_preflight,
            expected_state_revision=runtime_state.state_revision,
            decoding_evidence={
                "decoded_token_hashes": generation_token_hashes,
                "decoded_text_fragments": ["spike language"],
                "rendered_text": "spike language",
                "schema_valid": True,
                "text_normalized": True,
                "semantic_constraint_valid": True,
                "checkpoint_written": False,
            },
        )
    )
    blocked_snn_language_decoding_event_review = (
        ledger.autonomous_snn_language_decoding_event_review(
            autonomous_snn_language_decoding_executor=(
                blocked_snn_language_decoding_executor
            ),
            expected_state_revision=runtime_state.state_revision,
            review_policy={"max_decoded_fragments": 1, "max_surface_chars": 256},
        )
    )
    snn_language_decoding_event_review = (
        ledger.autonomous_snn_language_decoding_event_review(
            autonomous_snn_language_decoding_executor=snn_language_decoding_executor,
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "max_decoded_tokens": 2,
                "max_decoded_fragments": 1,
                "max_surface_chars": 256,
            },
        )
    )
    blocked_snn_language_readout_surface_design = (
        ledger.snn_language_readout_surface_design(
            autonomous_snn_language_decoding_event_review=(
                blocked_snn_language_decoding_event_review
            ),
            surface_policy={
                "readout_role": "bounded_readout_candidate",
                "binding_mode": "hash_bound_readout_language",
                "max_readout_fragments": 1,
                "max_surface_chars": 256,
                "max_association_edges": 4,
            },
        )
    )
    snn_language_readout_surface_design = (
        ledger.snn_language_readout_surface_design(
            autonomous_snn_language_decoding_event_review=(
                snn_language_decoding_event_review
            ),
            surface_policy={
                "readout_role": "bounded_readout_candidate",
                "binding_mode": "hash_bound_readout_language",
                "max_readout_fragments": 1,
                "max_surface_chars": 256,
                "max_association_edges": 4,
            },
        )
    )
    blocked_snn_language_readout_surface_preflight = (
        ledger.snn_language_readout_surface_preflight(
            snn_language_readout_surface_design=(
                blocked_snn_language_readout_surface_design
            ),
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cuda:0", "cuda_available": True},
            executor_capabilities={
                "snn_language_readout_surface_executor": True
            },
        )
    )
    snn_language_readout_surface_preflight = (
        ledger.snn_language_readout_surface_preflight(
            snn_language_readout_surface_design=(
                snn_language_readout_surface_design
            ),
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cuda:0", "cuda_available": True},
            executor_capabilities={
                "snn_language_readout_surface_executor": True
            },
        )
    )
    blocked_snn_language_readout_surface_executor = (
        ledger.execute_snn_language_readout_surface(
            snn_language_readout_surface_preflight=(
                blocked_snn_language_readout_surface_preflight
            ),
            expected_state_revision=runtime_state.state_revision,
        )
    )
    snn_language_readout_surface_executor = (
        ledger.execute_snn_language_readout_surface(
            snn_language_readout_surface_preflight=(
                snn_language_readout_surface_preflight
            ),
            expected_state_revision=runtime_state.state_revision,
        )
    )
    blocked_snn_language_readout_surface_event_review = (
        ledger.snn_language_readout_surface_event_review(
            snn_language_readout_surface_executor=(
                blocked_snn_language_readout_surface_executor
            ),
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "max_readout_fragments": 1,
                "max_surface_chars": 256,
                "max_association_edges": 4,
            },
        )
    )
    snn_language_readout_surface_event_review = (
        ledger.snn_language_readout_surface_event_review(
            snn_language_readout_surface_executor=(
                snn_language_readout_surface_executor
            ),
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "max_readout_fragments": 1,
                "max_surface_chars": 256,
                "max_association_edges": 4,
            },
        )
    )
    blocked_snn_language_readout_memory_design = (
        ledger.snn_language_readout_memory_design(
            snn_language_readout_surface_event_review=(
                blocked_snn_language_readout_surface_event_review
            ),
            memory_policy={
                "memory_scope": "working_trace",
                "consolidation_route": "deferred_local_trace",
                "max_trace_fragments": 1,
                "max_trace_chars": 256,
                "max_local_learning_targets": 4,
            },
        )
    )
    snn_language_readout_memory_design = (
        ledger.snn_language_readout_memory_design(
            snn_language_readout_surface_event_review=(
                snn_language_readout_surface_event_review
            ),
            memory_policy={
                "memory_scope": "working_trace",
                "consolidation_route": "deferred_local_trace",
                "max_trace_fragments": 1,
                "max_trace_chars": 256,
                "max_local_learning_targets": 4,
            },
        )
    )
    blocked_snn_language_readout_memory_preflight = (
        ledger.snn_language_readout_memory_preflight(
            snn_language_readout_memory_design=(
                blocked_snn_language_readout_memory_design
            ),
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cuda:0", "cuda_available": True},
            executor_capabilities={
                "snn_language_readout_memory_executor": True
            },
        )
    )
    snn_language_readout_memory_preflight = (
        ledger.snn_language_readout_memory_preflight(
            snn_language_readout_memory_design=(
                snn_language_readout_memory_design
            ),
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cuda:0", "cuda_available": True},
            executor_capabilities={
                "snn_language_readout_memory_executor": True
            },
        )
    )
    blocked_snn_language_readout_memory_executor = (
        ledger.execute_snn_language_readout_memory(
            snn_language_readout_memory_preflight=(
                blocked_snn_language_readout_memory_preflight
            ),
            expected_state_revision=runtime_state.state_revision,
        )
    )
    snn_language_readout_memory_executor = (
        ledger.execute_snn_language_readout_memory(
            snn_language_readout_memory_preflight=(
                snn_language_readout_memory_preflight
            ),
            expected_state_revision=runtime_state.state_revision,
        )
    )
    blocked_snn_language_readout_memory_event_review = (
        ledger.snn_language_readout_memory_event_review(
            snn_language_readout_memory_executor=(
                blocked_snn_language_readout_memory_executor
            ),
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "max_trace_fragments": 1,
                "max_trace_chars": 256,
                "max_local_learning_targets": 4,
            },
        )
    )
    snn_language_readout_memory_event_review = (
        ledger.snn_language_readout_memory_event_review(
            snn_language_readout_memory_executor=(
                snn_language_readout_memory_executor
            ),
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "max_trace_fragments": 1,
                "max_trace_chars": 256,
                "max_local_learning_targets": 4,
            },
        )
    )
    blocked_snn_language_thought_consolidation_design = (
        ledger.autonomous_snn_language_thought_consolidation_design(
            snn_language_readout_memory_event_review=(
                blocked_snn_language_readout_memory_event_review
            ),
            consolidation_policy={
                "consolidation_scope": "local_trace_reinforcement",
                "consolidation_route": "deferred_local_trace",
                "learning_rate": 0.02,
                "max_weight_delta": 0.04,
                "homeostatic_decay": 0.01,
                "max_candidate_updates": 4,
            },
        )
    )
    snn_language_thought_consolidation_design = (
        ledger.autonomous_snn_language_thought_consolidation_design(
            snn_language_readout_memory_event_review=(
                snn_language_readout_memory_event_review
            ),
            consolidation_policy={
                "consolidation_scope": "local_trace_reinforcement",
                "consolidation_route": "deferred_local_trace",
                "learning_rate": 0.02,
                "max_weight_delta": 0.04,
                "homeostatic_decay": 0.01,
                "max_candidate_updates": 4,
            },
        )
    )
    blocked_snn_language_thought_consolidation_preflight = (
        ledger.autonomous_snn_language_thought_consolidation_preflight(
            autonomous_snn_language_thought_consolidation_design=(
                blocked_snn_language_thought_consolidation_design
            ),
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cuda:0", "cuda_available": True},
            executor_capabilities={
                "autonomous_snn_language_thought_consolidation_executor": True
            },
        )
    )
    snn_language_thought_consolidation_preflight = (
        ledger.autonomous_snn_language_thought_consolidation_preflight(
            autonomous_snn_language_thought_consolidation_design=(
                snn_language_thought_consolidation_design
            ),
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cuda:0", "cuda_available": True},
            executor_capabilities={
                "autonomous_snn_language_thought_consolidation_executor": True
            },
        )
    )
    blocked_snn_language_thought_consolidation_executor = (
        ledger.execute_autonomous_snn_language_thought_consolidation(
            autonomous_snn_language_thought_consolidation_preflight=(
                blocked_snn_language_thought_consolidation_preflight
            ),
            expected_state_revision=runtime_state.state_revision,
        )
    )
    snn_language_thought_consolidation_executor = (
        ledger.execute_autonomous_snn_language_thought_consolidation(
            autonomous_snn_language_thought_consolidation_preflight=(
                snn_language_thought_consolidation_preflight
            ),
            expected_state_revision=runtime_state.state_revision,
        )
    )
    blocked_snn_language_thought_consolidation_event_review = (
        ledger.autonomous_snn_language_thought_consolidation_event_review(
            autonomous_snn_language_thought_consolidation_executor=(
                blocked_snn_language_thought_consolidation_executor
            ),
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "max_candidate_updates": 4,
                "max_learning_rate": 0.02,
                "max_weight_delta": 0.04,
                "max_homeostatic_decay": 0.01,
            },
        )
    )
    snn_language_thought_consolidation_event_review = (
        ledger.autonomous_snn_language_thought_consolidation_event_review(
            autonomous_snn_language_thought_consolidation_executor=(
                snn_language_thought_consolidation_executor
            ),
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "max_candidate_updates": 4,
                "max_learning_rate": 0.02,
                "max_weight_delta": 0.04,
                "max_homeostatic_decay": 0.01,
            },
        )
    )
    blocked_snn_language_thought_structural_plasticity_design = (
        ledger.autonomous_snn_language_thought_structural_plasticity_design(
            autonomous_snn_language_thought_consolidation_event_review=(
                blocked_snn_language_thought_consolidation_event_review
            ),
            structural_policy={
                "structural_scope": "thought_trace_sparse_capacity",
                "structural_route": "reviewed_consolidation_to_growth_prune",
                "max_growth_candidates": 4,
                "max_prune_candidates": 2,
                "max_new_neurons": 2,
                "max_new_synapses": 4,
                "max_prune_synapses": 2,
            },
        )
    )
    snn_language_thought_structural_plasticity_design = (
        ledger.autonomous_snn_language_thought_structural_plasticity_design(
            autonomous_snn_language_thought_consolidation_event_review=(
                snn_language_thought_consolidation_event_review
            ),
            structural_policy={
                "structural_scope": "thought_trace_sparse_capacity",
                "structural_route": "reviewed_consolidation_to_growth_prune",
                "max_growth_candidates": 4,
                "max_prune_candidates": 2,
                "max_new_neurons": 2,
                "max_new_synapses": 4,
                "max_prune_synapses": 2,
            },
        )
    )
    blocked_snn_language_thought_structural_plasticity_preflight = (
        ledger.autonomous_snn_language_thought_structural_plasticity_preflight(
            autonomous_snn_language_thought_structural_plasticity_design=(
                blocked_snn_language_thought_structural_plasticity_design
            ),
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cuda:0", "cuda_available": True},
            executor_capabilities={
                "autonomous_snn_language_thought_structural_plasticity_executor": True
            },
        )
    )
    snn_language_thought_structural_plasticity_preflight = (
        ledger.autonomous_snn_language_thought_structural_plasticity_preflight(
            autonomous_snn_language_thought_structural_plasticity_design=(
                snn_language_thought_structural_plasticity_design
            ),
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cuda:0", "cuda_available": True},
            executor_capabilities={
                "autonomous_snn_language_thought_structural_plasticity_executor": True
            },
        )
    )
    blocked_snn_language_thought_structural_plasticity_executor = (
        ledger.execute_autonomous_snn_language_thought_structural_plasticity(
            autonomous_snn_language_thought_structural_plasticity_preflight=(
                blocked_snn_language_thought_structural_plasticity_preflight
            ),
            expected_state_revision=runtime_state.state_revision,
        )
    )
    snn_language_thought_structural_plasticity_executor = (
        ledger.execute_autonomous_snn_language_thought_structural_plasticity(
            autonomous_snn_language_thought_structural_plasticity_preflight=(
                snn_language_thought_structural_plasticity_preflight
            ),
            expected_state_revision=runtime_state.state_revision,
        )
    )
    blocked_snn_language_thought_structural_plasticity_event_review = (
        ledger.autonomous_snn_language_thought_structural_plasticity_event_review(
            autonomous_snn_language_thought_structural_plasticity_executor=(
                blocked_snn_language_thought_structural_plasticity_executor
            ),
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "max_growth_candidates": 4,
                "max_prune_candidates": 2,
                "max_new_neurons": 2,
                "max_new_synapses": 4,
                "max_prune_synapses": 2,
            },
        )
    )
    snn_language_thought_structural_plasticity_event_review = (
        ledger.autonomous_snn_language_thought_structural_plasticity_event_review(
            autonomous_snn_language_thought_structural_plasticity_executor=(
                snn_language_thought_structural_plasticity_executor
            ),
            expected_state_revision=runtime_state.state_revision,
            review_policy={
                "max_growth_candidates": 4,
                "max_prune_candidates": 2,
                "max_new_neurons": 2,
                "max_new_synapses": 4,
                "max_prune_synapses": 2,
            },
        )
    )
    blocked_snn_language_thought_capacity_mutation_design = (
        ledger.autonomous_snn_language_thought_capacity_mutation_design(
            autonomous_snn_language_thought_structural_plasticity_event_review=(
                blocked_snn_language_thought_structural_plasticity_event_review
            ),
            capacity_policy={
                "mutation_scope": "thought_driven_sparse_capacity",
                "mutation_route": "reviewed_structural_plasticity_to_capacity_resize",
                "current_neuron_capacity": 64,
                "current_sparse_synapse_budget": 256,
                "current_dense_rows": 64,
                "current_dense_cols": 64,
                "max_capacity_growth_factor": 2.0,
            },
        )
    )
    snn_language_thought_capacity_mutation_design = (
        ledger.autonomous_snn_language_thought_capacity_mutation_design(
            autonomous_snn_language_thought_structural_plasticity_event_review=(
                snn_language_thought_structural_plasticity_event_review
            ),
            capacity_policy={
                "mutation_scope": "thought_driven_sparse_capacity",
                "mutation_route": "reviewed_structural_plasticity_to_capacity_resize",
                "current_neuron_capacity": 64,
                "current_sparse_synapse_budget": 256,
                "current_dense_rows": 64,
                "current_dense_cols": 64,
                "max_capacity_growth_factor": 2.0,
            },
        )
    )
    blocked_snn_language_thought_capacity_mutation_preflight = (
        ledger.autonomous_snn_language_thought_capacity_mutation_preflight(
            autonomous_snn_language_thought_capacity_mutation_design=(
                blocked_snn_language_thought_capacity_mutation_design
            ),
            expected_state_revision=runtime_state.state_revision,
            checkpoint_transaction={
                "checkpoint_path": "memory://thought-capacity-before",
                "snapshot_id": "thought-capacity-snapshot",
                "pre_capacity_mutation_checkpoint_saved": True,
                "pre_capacity_mutation_checkpoint_restore_verified": True,
            },
            device_evidence={
                "device": "cuda:0",
                "cuda_available": True,
                "cuda_relayout_verified": True,
            },
            executor_capabilities={
                "autonomous_snn_language_thought_capacity_mutation_executor": True
            },
        )
    )
    snn_language_thought_capacity_mutation_preflight = (
        ledger.autonomous_snn_language_thought_capacity_mutation_preflight(
            autonomous_snn_language_thought_capacity_mutation_design=(
                snn_language_thought_capacity_mutation_design
            ),
            expected_state_revision=runtime_state.state_revision,
            checkpoint_transaction={
                "checkpoint_path": "memory://thought-capacity-before",
                "snapshot_id": "thought-capacity-snapshot",
                "pre_capacity_mutation_checkpoint_saved": True,
                "pre_capacity_mutation_checkpoint_restore_verified": True,
            },
            device_evidence={
                "device": "cuda:0",
                "cuda_available": True,
                "cuda_relayout_verified": True,
            },
            executor_capabilities={
                "autonomous_snn_language_thought_capacity_mutation_executor": True
            },
        )
    )

    assert blocked_text_surface_commit_executor["accepted"] is False
    assert blocked_text_surface_commit_executor["requires_operator_approval"] is False
    assert blocked_text_surface_commit_executor["promotion_gate"]["required_evidence"][
        "preflight_ready"
    ] is False
    _assert_record_family_source_window(
        blocked_text_surface_commit_executor["source_window"],
        field="autonomous_text_surface_commit_events",
        expected_count=0,
    )
    assert text_surface_commit_executor["surface"] == (
        "snn_language_autonomous_text_surface_commit_executor.v1"
    )
    assert text_surface_commit_executor["accepted"] is True
    assert text_surface_commit_executor["ready"] is True
    assert text_surface_commit_executor["requires_operator_approval"] is False
    assert text_surface_commit_executor["advisory"] is False
    assert text_surface_commit_executor["executable"] is True
    assert text_surface_commit_executor["records_ledger_event"] is True
    assert text_surface_commit_executor["mutates_runtime_state"] is True
    assert text_surface_commit_executor["runs_replay"] is False
    assert text_surface_commit_executor["writes_checkpoint"] is False
    assert text_surface_commit_executor["generates_text"] is False
    assert text_surface_commit_executor["decodes_text"] is False
    assert text_surface_commit_executor["trains_runtime_model"] is False
    assert text_surface_commit_executor["applies_plasticity"] is False
    text_surface_commit_event = text_surface_commit_executor[
        "autonomous_text_surface_commit_event"
    ]
    assert text_surface_commit_event["commit_scope"] == "hash_surface_state"
    assert text_surface_commit_event["retention_class"] == "ephemeral_hash_surface"
    assert text_surface_commit_event["text_fragment_count"] == 3
    assert len(text_surface_commit_event["decoded_token_hashes"]) == 3
    assert len(text_surface_commit_event["text_fragment_hashes"]) == 3
    assert len(text_surface_commit_event["text_emission_slot_hashes"]) == 3
    assert len(text_surface_commit_event["committed_surface_hash"]) == 64
    assert len(text_surface_commit_event["state_chain_hash"]) == 64
    assert text_surface_commit_event["output_is_hash_only"] is True
    assert text_surface_commit_event["literal_text_returned"] is False
    assert text_surface_commit_event["operator_approval_required"] is False
    _assert_record_family_source_window(
        text_surface_commit_executor["source_window"],
        field="autonomous_text_surface_commit_events",
        expected_count=0,
    )
    assert text_surface_commit_executor["promotion_gate"][
        "eligible_for_autonomous_text_surface_commit_event_review"
    ] is True
    assert text_surface_commit_executor["promotion_gate"]["eligible_for_language_generation"] is False
    assert text_surface_commit_executor["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert text_surface_commit_executor["promotion_gate"]["eligible_for_replay_memory"] is False
    assert text_surface_commit_executor["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert text_surface_commit_executor["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert text_surface_commit_executor["promotion_gate"]["eligible_for_action"] is False
    assert blocked_text_surface_commit_event_review["accepted"] is False
    assert blocked_text_surface_commit_event_review["requires_operator_approval"] is False
    assert blocked_text_surface_commit_event_review["promotion_gate"]["required_evidence"][
        "executor_accepted"
    ] is False
    _assert_record_family_source_window(
        blocked_text_surface_commit_event_review["source_window"],
        field="autonomous_text_surface_commit_events",
        expected_count=1,
    )
    assert text_surface_commit_event_review["surface"] == (
        "snn_language_autonomous_text_surface_commit_event_review.v1"
    )
    assert text_surface_commit_event_review["accepted"] is True
    assert text_surface_commit_event_review["ready"] is True
    assert len(text_surface_commit_event_review["review_hash"]) == 64
    assert text_surface_commit_event_review["requires_operator_approval"] is False
    assert text_surface_commit_event_review["advisory"] is True
    assert text_surface_commit_event_review["executable"] is False
    assert text_surface_commit_event_review["records_ledger_event"] is False
    assert text_surface_commit_event_review["mutates_runtime_state"] is False
    assert text_surface_commit_event_review["runs_replay"] is False
    assert text_surface_commit_event_review["writes_checkpoint"] is False
    assert text_surface_commit_event_review["generates_text"] is False
    assert text_surface_commit_event_review["decodes_text"] is False
    assert text_surface_commit_event_review["trains_runtime_model"] is False
    assert text_surface_commit_event_review["applies_plasticity"] is False
    text_surface_commit_review_body = text_surface_commit_event_review[
        "autonomous_text_surface_commit_event_review"
    ]
    assert text_surface_commit_review_body["event_recorded_in_ledger"] is True
    assert text_surface_commit_review_body["current_commit_matches_event"] is True
    assert text_surface_commit_review_body[
        "autonomous_text_surface_commit_event_hash"
    ] == text_surface_commit_event["autonomous_text_surface_commit_event_hash"]
    assert text_surface_commit_review_body["commit_scope"] == "hash_surface_state"
    assert text_surface_commit_review_body["retention_class"] == "ephemeral_hash_surface"
    assert text_surface_commit_review_body["text_fragment_count"] == 3
    assert text_surface_commit_review_body["committed_surface_hash"] == (
        text_surface_commit_event["fragment_sequence_hash"]
    )
    assert len(text_surface_commit_review_body["state_chain_hash"]) == 64
    assert text_surface_commit_review_body["output_is_hash_only"] is True
    assert text_surface_commit_review_body["literal_text_returned"] is False
    _assert_record_family_source_window(
        text_surface_commit_event_review["source_window"],
        field="autonomous_text_surface_commit_events",
        expected_count=1,
    )
    assert text_surface_commit_event_review["promotion_gate"][
        "eligible_for_autonomous_text_surface_materialization_design"
    ] is True
    assert text_surface_commit_event_review["promotion_gate"]["eligible_for_language_generation"] is False
    assert text_surface_commit_event_review["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert text_surface_commit_event_review["promotion_gate"]["eligible_for_replay_memory"] is False
    assert text_surface_commit_event_review["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert text_surface_commit_event_review["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert text_surface_commit_event_review["promotion_gate"]["eligible_for_action"] is False
    assert blocked_text_surface_materialization_design["accepted"] is False
    assert blocked_text_surface_materialization_design["requires_operator_approval"] is False
    assert blocked_text_surface_materialization_design["promotion_gate"]["required_evidence"][
        "commit_event_review_ready"
    ] is False
    assert text_surface_materialization_design["surface"] == (
        "snn_language_autonomous_text_surface_materialization_design.v1"
    )
    assert text_surface_materialization_design["accepted"] is True
    assert text_surface_materialization_design["ready"] is True
    assert len(text_surface_materialization_design["materialization_design_hash"]) == 64
    assert text_surface_materialization_design["requires_operator_approval"] is False
    assert text_surface_materialization_design["advisory"] is True
    assert text_surface_materialization_design["executable"] is False
    assert text_surface_materialization_design["records_ledger_event"] is False
    assert text_surface_materialization_design["mutates_runtime_state"] is False
    assert text_surface_materialization_design["runs_replay"] is False
    assert text_surface_materialization_design["writes_checkpoint"] is False
    assert text_surface_materialization_design["generates_text"] is False
    assert text_surface_materialization_design["decodes_text"] is False
    assert text_surface_materialization_design["trains_runtime_model"] is False
    assert text_surface_materialization_design["applies_plasticity"] is False
    text_surface_materialization_body = text_surface_materialization_design[
        "autonomous_text_surface_materialization_design"
    ]
    assert text_surface_materialization_body["materialization_mode"] == (
        "bounded_hash_to_text_surface"
    )
    assert text_surface_materialization_body["output_contract"] == (
        "bounded_display_surface"
    )
    assert text_surface_materialization_body["max_surface_chars"] == 256
    assert len(text_surface_materialization_body["materialization_plan_hash"]) == 64
    assert text_surface_materialization_body[
        "autonomous_text_surface_commit_event_hash"
    ] == text_surface_commit_event["autonomous_text_surface_commit_event_hash"]
    assert text_surface_materialization_body["committed_surface_hash"] == (
        text_surface_commit_event["fragment_sequence_hash"]
    )
    assert text_surface_materialization_body["output_is_hash_only"] is True
    assert text_surface_materialization_body["literal_text_returned"] is False
    assert text_surface_materialization_design["promotion_gate"][
        "eligible_for_autonomous_text_surface_materialization_preflight"
    ] is True
    assert text_surface_materialization_design["promotion_gate"]["eligible_for_language_generation"] is False
    assert text_surface_materialization_design["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert text_surface_materialization_design["promotion_gate"]["eligible_for_replay_memory"] is False
    assert text_surface_materialization_design["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert text_surface_materialization_design["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert text_surface_materialization_design["promotion_gate"]["eligible_for_action"] is False
    assert blocked_text_surface_materialization_preflight["accepted"] is False
    assert blocked_text_surface_materialization_preflight["requires_operator_approval"] is False
    assert blocked_text_surface_materialization_preflight["promotion_gate"][
        "required_evidence"
    ]["materialization_design_ready"] is False
    assert text_surface_materialization_preflight["surface"] == (
        "snn_language_autonomous_text_surface_materialization_preflight.v1"
    )
    assert text_surface_materialization_preflight["accepted"] is True
    assert text_surface_materialization_preflight["ready"] is True
    assert len(text_surface_materialization_preflight["preflight_hash"]) == 64
    assert text_surface_materialization_preflight["requires_operator_approval"] is False
    assert text_surface_materialization_preflight["advisory"] is True
    assert text_surface_materialization_preflight["executable"] is False
    assert text_surface_materialization_preflight["records_ledger_event"] is False
    assert text_surface_materialization_preflight["mutates_runtime_state"] is False
    assert text_surface_materialization_preflight["runs_replay"] is False
    assert text_surface_materialization_preflight["writes_checkpoint"] is False
    assert text_surface_materialization_preflight["generates_text"] is False
    assert text_surface_materialization_preflight["decodes_text"] is False
    assert text_surface_materialization_preflight["trains_runtime_model"] is False
    assert text_surface_materialization_preflight["applies_plasticity"] is False
    text_surface_materialization_preflight_body = text_surface_materialization_preflight[
        "autonomous_text_surface_materialization_preflight"
    ]
    assert text_surface_materialization_preflight_body["materialization_mode"] == (
        "bounded_hash_to_text_surface"
    )
    assert text_surface_materialization_preflight_body["output_contract"] == (
        "bounded_display_surface"
    )
    assert text_surface_materialization_preflight_body["max_surface_chars"] == 256
    assert text_surface_materialization_preflight_body["requires_cuda"] is False
    assert text_surface_materialization_preflight_body["literal_text_returned"] is False
    assert text_surface_materialization_preflight["promotion_gate"][
        "eligible_for_autonomous_text_surface_materialization_executor"
    ] is True
    assert text_surface_materialization_preflight["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert text_surface_materialization_preflight["promotion_gate"][
        "eligible_for_dense_readout_training"
    ] is False
    assert text_surface_materialization_preflight["promotion_gate"][
        "eligible_for_replay_memory"
    ] is False
    assert text_surface_materialization_preflight["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert text_surface_materialization_preflight["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert text_surface_materialization_preflight["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_text_surface_materialization_executor["accepted"] is False
    assert blocked_text_surface_materialization_executor["requires_operator_approval"] is False
    assert blocked_text_surface_materialization_executor["rendered_text"] is None
    assert blocked_text_surface_materialization_executor["promotion_gate"][
        "required_evidence"
    ]["preflight_ready"] is False
    _assert_record_family_source_window(
        blocked_text_surface_materialization_executor["source_window"],
        field="autonomous_text_surface_materialization_events",
        expected_count=0,
    )
    assert text_surface_materialization_executor["surface"] == (
        "snn_language_autonomous_text_surface_materialization_executor.v1"
    )
    assert text_surface_materialization_executor["accepted"] is True
    assert text_surface_materialization_executor["ready"] is True
    assert text_surface_materialization_executor["requires_operator_approval"] is False
    assert text_surface_materialization_executor["advisory"] is False
    assert text_surface_materialization_executor["executable"] is True
    assert text_surface_materialization_executor["records_ledger_event"] is True
    assert text_surface_materialization_executor["mutates_runtime_state"] is True
    assert text_surface_materialization_executor["runs_replay"] is False
    assert text_surface_materialization_executor["writes_checkpoint"] is False
    assert text_surface_materialization_executor["generates_text"] is False
    assert text_surface_materialization_executor["decodes_text"] is False
    assert text_surface_materialization_executor["trains_runtime_model"] is False
    assert text_surface_materialization_executor["applies_plasticity"] is False
    assert text_surface_materialization_executor["literal_text_returned"] is True
    assert text_surface_materialization_executor["output_is_bounded_text_surface"] is True
    assert text_surface_materialization_executor["text_fragments"] == (
        bounded_text_fragments
    )
    _assert_record_family_source_window(
        text_surface_materialization_executor["source_window"],
        field="autonomous_text_surface_materialization_events",
        expected_count=0,
    )
    assert (
        text_surface_materialization_executor["ledger_summary"][
            "total_autonomous_text_surface_materialization_count"
        ]
        == 1
    )
    assert text_surface_materialization_executor["rendered_text"] == (
        "\n".join(bounded_text_fragments)
    )
    assert len(text_surface_materialization_executor["rendered_text_hash"]) == 64
    text_surface_materialization_event = text_surface_materialization_executor[
        "autonomous_text_surface_materialization_event"
    ]
    assert text_surface_materialization_event["text_fragments"] == bounded_text_fragments
    assert text_surface_materialization_event["literal_fragment_hashes"] == (
        bounded_text_fragment_hashes
    )
    assert text_surface_materialization_event["rendered_text"] == (
        "\n".join(bounded_text_fragments)
    )
    assert len(
        text_surface_materialization_event[
            "autonomous_text_surface_materialization_event_hash"
        ]
    ) == 64
    assert text_surface_materialization_executor["promotion_gate"][
        "eligible_for_autonomous_text_surface_materialization_event_review"
    ] is True
    assert text_surface_materialization_executor["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert text_surface_materialization_executor["promotion_gate"][
        "eligible_for_dense_readout_training"
    ] is False
    assert text_surface_materialization_executor["promotion_gate"][
        "eligible_for_replay_memory"
    ] is False
    assert text_surface_materialization_executor["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert text_surface_materialization_executor["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert text_surface_materialization_executor["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_text_surface_materialization_event_review["accepted"] is False
    assert blocked_text_surface_materialization_event_review[
        "requires_operator_approval"
    ] is False
    assert blocked_text_surface_materialization_event_review["promotion_gate"][
        "required_evidence"
    ]["executor_accepted"] is False
    _assert_record_family_source_window(
        blocked_text_surface_materialization_event_review["source_window"],
        field="autonomous_text_surface_materialization_events",
        expected_count=1,
    )
    assert text_surface_materialization_event_review["surface"] == (
        "snn_language_autonomous_text_surface_materialization_event_review.v1"
    )
    assert text_surface_materialization_event_review["accepted"] is True
    assert text_surface_materialization_event_review["ready"] is True
    assert len(text_surface_materialization_event_review["review_hash"]) == 64
    assert text_surface_materialization_event_review["requires_operator_approval"] is False
    assert text_surface_materialization_event_review["advisory"] is True
    assert text_surface_materialization_event_review["executable"] is False
    assert text_surface_materialization_event_review["records_ledger_event"] is False
    assert text_surface_materialization_event_review["mutates_runtime_state"] is False
    assert text_surface_materialization_event_review["runs_replay"] is False
    assert text_surface_materialization_event_review["writes_checkpoint"] is False
    assert text_surface_materialization_event_review["generates_text"] is False
    assert text_surface_materialization_event_review["decodes_text"] is False
    assert text_surface_materialization_event_review["trains_runtime_model"] is False
    assert text_surface_materialization_event_review["applies_plasticity"] is False
    _assert_record_family_source_window(
        text_surface_materialization_event_review["source_window"],
        field="autonomous_text_surface_materialization_events",
        expected_count=1,
    )
    text_surface_materialization_review_body = (
        text_surface_materialization_event_review[
            "autonomous_text_surface_materialization_event_review"
        ]
    )
    assert text_surface_materialization_review_body["event_recorded_in_ledger"] is True
    assert text_surface_materialization_review_body[
        "current_materialization_matches_event"
    ] is True
    assert text_surface_materialization_review_body["rendered_text"] == (
        "\n".join(bounded_text_fragments)
    )
    assert text_surface_materialization_review_body["text_fragments"] == (
        bounded_text_fragments
    )
    assert text_surface_materialization_review_body["literal_fragment_hashes"] == (
        bounded_text_fragment_hashes
    )
    assert text_surface_materialization_review_body["literal_text_returned"] is True
    assert (
        text_surface_materialization_review_body["output_is_bounded_text_surface"]
        is True
    )
    assert text_surface_materialization_event_review["promotion_gate"][
        "eligible_for_autonomous_bounded_language_surface_review"
    ] is True
    assert text_surface_materialization_event_review["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert text_surface_materialization_event_review["promotion_gate"][
        "eligible_for_dense_readout_training"
    ] is False
    assert text_surface_materialization_event_review["promotion_gate"][
        "eligible_for_replay_memory"
    ] is False
    assert text_surface_materialization_event_review["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert text_surface_materialization_event_review["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert text_surface_materialization_event_review["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_bounded_language_surface_review["accepted"] is False
    assert blocked_bounded_language_surface_review["requires_operator_approval"] is False
    assert blocked_bounded_language_surface_review["promotion_gate"][
        "required_evidence"
    ]["materialization_event_review_ready"] is False
    assert bounded_language_surface_review["surface"] == (
        "snn_language_autonomous_bounded_language_surface_review.v1"
    )
    assert bounded_language_surface_review["accepted"] is True
    assert bounded_language_surface_review["ready"] is True
    assert len(bounded_language_surface_review["review_hash"]) == 64
    assert bounded_language_surface_review["requires_operator_approval"] is False
    assert bounded_language_surface_review["advisory"] is True
    assert bounded_language_surface_review["executable"] is False
    assert bounded_language_surface_review["records_ledger_event"] is False
    assert bounded_language_surface_review["mutates_runtime_state"] is False
    assert bounded_language_surface_review["runs_replay"] is False
    assert bounded_language_surface_review["writes_checkpoint"] is False
    assert bounded_language_surface_review["generates_text"] is False
    assert bounded_language_surface_review["decodes_text"] is False
    assert bounded_language_surface_review["trains_runtime_model"] is False
    assert bounded_language_surface_review["applies_plasticity"] is False
    assert bounded_language_surface_review["rendered_text"] == (
        "\n".join(bounded_text_fragments)
    )
    assert bounded_language_surface_review["text_fragments"] == bounded_text_fragments
    assert len(bounded_language_surface_review["bounded_language_surface_hash"]) == 64
    bounded_language_surface_body = bounded_language_surface_review[
        "autonomous_bounded_language_surface_review"
    ]
    assert bounded_language_surface_body["language_surface_mode"] == (
        "bounded_language_surface"
    )
    assert bounded_language_surface_body["rendered_text"] == (
        "\n".join(bounded_text_fragments)
    )
    assert bounded_language_surface_body["text_fragments"] == bounded_text_fragments
    assert bounded_language_surface_body["literal_text_returned"] is True
    assert bounded_language_surface_body["output_is_bounded_text_surface"] is True
    assert bounded_language_surface_review["promotion_gate"][
        "eligible_for_autonomous_bounded_language_surface_commit_design"
    ] is True
    assert bounded_language_surface_review["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert bounded_language_surface_review["promotion_gate"][
        "eligible_for_dense_readout_training"
    ] is False
    assert bounded_language_surface_review["promotion_gate"][
        "eligible_for_replay_memory"
    ] is False
    assert bounded_language_surface_review["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert bounded_language_surface_review["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert bounded_language_surface_review["promotion_gate"]["eligible_for_action"] is False
    assert blocked_bounded_language_surface_commit_design["accepted"] is False
    assert (
        blocked_bounded_language_surface_commit_design["requires_operator_approval"]
        is False
    )
    assert blocked_bounded_language_surface_commit_design["promotion_gate"][
        "required_evidence"
    ]["bounded_language_surface_review_ready"] is False
    assert bounded_language_surface_commit_design["surface"] == (
        "snn_language_autonomous_bounded_language_surface_commit_design.v1"
    )
    assert bounded_language_surface_commit_design["accepted"] is True
    assert bounded_language_surface_commit_design["ready"] is True
    assert (
        len(
            bounded_language_surface_commit_design[
                "language_surface_commit_design_hash"
            ]
        )
        == 64
    )
    assert bounded_language_surface_commit_design["requires_operator_approval"] is False
    assert bounded_language_surface_commit_design["advisory"] is True
    assert bounded_language_surface_commit_design["executable"] is False
    assert bounded_language_surface_commit_design["records_ledger_event"] is False
    assert bounded_language_surface_commit_design["mutates_runtime_state"] is False
    assert bounded_language_surface_commit_design["runs_replay"] is False
    assert bounded_language_surface_commit_design["writes_checkpoint"] is False
    assert bounded_language_surface_commit_design["generates_text"] is False
    assert bounded_language_surface_commit_design["decodes_text"] is False
    assert bounded_language_surface_commit_design["trains_runtime_model"] is False
    assert bounded_language_surface_commit_design["applies_plasticity"] is False
    bounded_language_surface_commit_body = bounded_language_surface_commit_design[
        "autonomous_bounded_language_surface_commit_design"
    ]
    assert (
        bounded_language_surface_commit_body["commit_scope"]
        == "bounded_language_surface"
    )
    assert (
        bounded_language_surface_commit_body["retention_class"]
        == "ephemeral_language_surface"
    )
    assert bounded_language_surface_commit_body["rendered_text"] == (
        "\n".join(bounded_text_fragments)
    )
    assert bounded_language_surface_commit_body["text_fragments"] == (
        bounded_text_fragments
    )
    assert (
        len(
            bounded_language_surface_commit_body[
                "language_surface_commit_plan_hash"
            ]
        )
        == 64
    )
    assert bounded_language_surface_commit_body["literal_text_returned"] is True
    assert (
        bounded_language_surface_commit_body["output_is_bounded_text_surface"]
        is True
    )
    assert bounded_language_surface_commit_design["promotion_gate"][
        "eligible_for_autonomous_bounded_language_surface_commit_preflight"
    ] is True
    assert bounded_language_surface_commit_design["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert bounded_language_surface_commit_design["promotion_gate"][
        "eligible_for_dense_readout_training"
    ] is False
    assert bounded_language_surface_commit_design["promotion_gate"][
        "eligible_for_replay_memory"
    ] is False
    assert bounded_language_surface_commit_design["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert bounded_language_surface_commit_design["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert bounded_language_surface_commit_design["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_bounded_language_surface_commit_preflight["accepted"] is False
    assert (
        blocked_bounded_language_surface_commit_preflight[
            "requires_operator_approval"
        ]
        is False
    )
    assert blocked_bounded_language_surface_commit_preflight["promotion_gate"][
        "required_evidence"
    ]["executor_capability_available"] is False
    assert bounded_language_surface_commit_preflight["surface"] == (
        "snn_language_autonomous_bounded_language_surface_commit_preflight.v1"
    )
    assert bounded_language_surface_commit_preflight["accepted"] is True
    assert bounded_language_surface_commit_preflight["ready"] is True
    assert (
        len(
            bounded_language_surface_commit_preflight[
                "language_surface_commit_preflight_hash"
            ]
        )
        == 64
    )
    assert (
        bounded_language_surface_commit_preflight["requires_operator_approval"]
        is False
    )
    assert bounded_language_surface_commit_preflight["advisory"] is True
    assert bounded_language_surface_commit_preflight["executable"] is False
    assert bounded_language_surface_commit_preflight["records_ledger_event"] is False
    assert bounded_language_surface_commit_preflight["mutates_runtime_state"] is False
    assert bounded_language_surface_commit_preflight["runs_replay"] is False
    assert bounded_language_surface_commit_preflight["writes_checkpoint"] is False
    assert bounded_language_surface_commit_preflight["generates_text"] is False
    assert bounded_language_surface_commit_preflight["decodes_text"] is False
    assert bounded_language_surface_commit_preflight["trains_runtime_model"] is False
    assert bounded_language_surface_commit_preflight["applies_plasticity"] is False
    bounded_language_surface_commit_preflight_body = (
        bounded_language_surface_commit_preflight[
            "autonomous_bounded_language_surface_commit_preflight"
        ]
    )
    assert (
        bounded_language_surface_commit_preflight_body["commit_scope"]
        == "bounded_language_surface"
    )
    assert (
        bounded_language_surface_commit_preflight_body["retention_class"]
        == "ephemeral_language_surface"
    )
    assert bounded_language_surface_commit_preflight_body["rendered_text"] == (
        "\n".join(bounded_text_fragments)
    )
    assert bounded_language_surface_commit_preflight_body["text_fragments"] == (
        bounded_text_fragments
    )
    assert (
        bounded_language_surface_commit_preflight_body["device_evidence"]["device"]
        == "cpu"
    )
    assert (
        bounded_language_surface_commit_preflight_body["literal_text_returned"]
        is True
    )
    assert (
        bounded_language_surface_commit_preflight_body[
            "output_is_bounded_text_surface"
        ]
        is True
    )
    assert bounded_language_surface_commit_preflight["promotion_gate"][
        "eligible_for_autonomous_bounded_language_surface_commit_executor"
    ] is True
    assert bounded_language_surface_commit_preflight["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert bounded_language_surface_commit_preflight["promotion_gate"][
        "eligible_for_dense_readout_training"
    ] is False
    assert bounded_language_surface_commit_preflight["promotion_gate"][
        "eligible_for_replay_memory"
    ] is False
    assert bounded_language_surface_commit_preflight["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert bounded_language_surface_commit_preflight["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert bounded_language_surface_commit_preflight["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_bounded_language_surface_commit_executor["accepted"] is False
    assert (
        blocked_bounded_language_surface_commit_executor["requires_operator_approval"]
        is False
    )
    assert blocked_bounded_language_surface_commit_executor["promotion_gate"][
        "required_evidence"
    ]["preflight_ready"] is False
    _assert_record_family_source_window(
        blocked_bounded_language_surface_commit_executor["source_window"],
        field="autonomous_bounded_language_surface_commit_events",
        expected_count=0,
    )
    assert bounded_language_surface_commit_executor["surface"] == (
        "snn_language_autonomous_bounded_language_surface_commit_executor.v1"
    )
    assert bounded_language_surface_commit_executor["accepted"] is True
    assert bounded_language_surface_commit_executor["ready"] is True
    assert (
        len(
            bounded_language_surface_commit_executor[
                "autonomous_bounded_language_surface_commit_event_hash"
            ]
        )
        == 64
    )
    assert (
        bounded_language_surface_commit_executor["requires_operator_approval"]
        is False
    )
    assert bounded_language_surface_commit_executor["advisory"] is False
    assert bounded_language_surface_commit_executor["executable"] is True
    assert bounded_language_surface_commit_executor["records_ledger_event"] is True
    assert bounded_language_surface_commit_executor["mutates_runtime_state"] is True
    assert bounded_language_surface_commit_executor["runs_replay"] is False
    assert bounded_language_surface_commit_executor["writes_checkpoint"] is False
    assert bounded_language_surface_commit_executor["generates_text"] is False
    assert bounded_language_surface_commit_executor["decodes_text"] is False
    assert bounded_language_surface_commit_executor["trains_runtime_model"] is False
    assert bounded_language_surface_commit_executor["applies_plasticity"] is False
    assert bounded_language_surface_commit_executor["literal_text_returned"] is True
    assert (
        bounded_language_surface_commit_executor["output_is_bounded_text_surface"]
        is True
    )
    assert bounded_language_surface_commit_executor["rendered_text"] == (
        "\n".join(bounded_text_fragments)
    )
    assert bounded_language_surface_commit_executor["text_fragments"] == (
        bounded_text_fragments
    )
    _assert_record_family_source_window(
        bounded_language_surface_commit_executor["source_window"],
        field="autonomous_bounded_language_surface_commit_events",
        expected_count=0,
    )
    assert (
        bounded_language_surface_commit_executor["ledger_summary"][
            "total_autonomous_bounded_language_surface_commit_count"
        ]
        == 1
    )
    bounded_language_surface_commit_event = bounded_language_surface_commit_executor[
        "autonomous_bounded_language_surface_commit_event"
    ]
    assert bounded_language_surface_commit_event["rendered_text"] == (
        "\n".join(bounded_text_fragments)
    )
    assert bounded_language_surface_commit_event["text_fragments"] == (
        bounded_text_fragments
    )
    assert (
        bounded_language_surface_commit_event["committed_language_surface_hash"]
        == bounded_language_surface_commit_preflight["bounded_language_surface_hash"]
    )
    assert len(
        bounded_language_surface_commit_event[
            "language_surface_state_chain_hash"
        ]
    ) == 64
    assert ledger_state["total_autonomous_bounded_language_surface_commit_count"] == 1
    assert (
        ledger_state["current_bounded_language_surface_commit"][
            "autonomous_bounded_language_surface_commit_event_hash"
        ]
        == bounded_language_surface_commit_executor[
            "autonomous_bounded_language_surface_commit_event_hash"
        ]
    )
    assert bounded_language_surface_commit_executor["promotion_gate"][
        "eligible_for_autonomous_bounded_language_surface_commit_event_review"
    ] is True
    assert bounded_language_surface_commit_executor["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert bounded_language_surface_commit_executor["promotion_gate"][
        "eligible_for_dense_readout_training"
    ] is False
    assert bounded_language_surface_commit_executor["promotion_gate"][
        "eligible_for_replay_memory"
    ] is False
    assert bounded_language_surface_commit_executor["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert bounded_language_surface_commit_executor["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert bounded_language_surface_commit_executor["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_bounded_language_surface_commit_event_review["accepted"] is False
    assert (
        blocked_bounded_language_surface_commit_event_review[
            "requires_operator_approval"
        ]
        is False
    )
    assert blocked_bounded_language_surface_commit_event_review["promotion_gate"][
        "required_evidence"
    ]["executor_accepted"] is False
    _assert_record_family_source_window(
        blocked_bounded_language_surface_commit_event_review["source_window"],
        field="autonomous_bounded_language_surface_commit_events",
        expected_count=1,
    )
    assert bounded_language_surface_commit_event_review["surface"] == (
        "snn_language_autonomous_bounded_language_surface_commit_event_review.v1"
    )
    assert bounded_language_surface_commit_event_review["accepted"] is True
    assert bounded_language_surface_commit_event_review["ready"] is True
    assert len(bounded_language_surface_commit_event_review["review_hash"]) == 64
    assert (
        bounded_language_surface_commit_event_review["requires_operator_approval"]
        is False
    )
    assert bounded_language_surface_commit_event_review["advisory"] is True
    assert bounded_language_surface_commit_event_review["executable"] is False
    assert (
        bounded_language_surface_commit_event_review["records_ledger_event"]
        is False
    )
    assert (
        bounded_language_surface_commit_event_review["mutates_runtime_state"]
        is False
    )
    assert bounded_language_surface_commit_event_review["runs_replay"] is False
    assert bounded_language_surface_commit_event_review["writes_checkpoint"] is False
    assert bounded_language_surface_commit_event_review["generates_text"] is False
    assert bounded_language_surface_commit_event_review["decodes_text"] is False
    assert (
        bounded_language_surface_commit_event_review["trains_runtime_model"]
        is False
    )
    assert bounded_language_surface_commit_event_review["applies_plasticity"] is False
    _assert_record_family_source_window(
        bounded_language_surface_commit_event_review["source_window"],
        field="autonomous_bounded_language_surface_commit_events",
        expected_count=1,
    )
    assert bounded_language_surface_commit_event_review["rendered_text"] == (
        "\n".join(bounded_text_fragments)
    )
    assert bounded_language_surface_commit_event_review["text_fragments"] == (
        bounded_text_fragments
    )
    bounded_language_surface_commit_review_body = (
        bounded_language_surface_commit_event_review[
            "autonomous_bounded_language_surface_commit_event_review"
        ]
    )
    assert (
        bounded_language_surface_commit_review_body["event_recorded_in_ledger"]
        is True
    )
    assert (
        bounded_language_surface_commit_review_body["current_commit_matches_event"]
        is True
    )
    assert bounded_language_surface_commit_review_body["rendered_text"] == (
        "\n".join(bounded_text_fragments)
    )
    assert bounded_language_surface_commit_review_body["text_fragments"] == (
        bounded_text_fragments
    )
    assert (
        bounded_language_surface_commit_review_body[
            "committed_language_surface_hash"
        ]
        == bounded_language_surface_commit_preflight["bounded_language_surface_hash"]
    )
    assert bounded_language_surface_commit_event_review["promotion_gate"][
        "eligible_for_autonomous_bounded_language_surface_use_review"
    ] is True
    assert bounded_language_surface_commit_event_review["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert bounded_language_surface_commit_event_review["promotion_gate"][
        "eligible_for_dense_readout_training"
    ] is False
    assert bounded_language_surface_commit_event_review["promotion_gate"][
        "eligible_for_replay_memory"
    ] is False
    assert bounded_language_surface_commit_event_review["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert bounded_language_surface_commit_event_review["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert bounded_language_surface_commit_event_review["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_bounded_language_surface_use_review["accepted"] is False
    assert (
        blocked_bounded_language_surface_use_review["requires_operator_approval"]
        is False
    )
    assert blocked_bounded_language_surface_use_review["promotion_gate"][
        "required_evidence"
    ]["commit_event_review_ready"] is False
    assert bounded_language_surface_use_review["surface"] == (
        "snn_language_autonomous_bounded_language_surface_use_review.v1"
    )
    assert bounded_language_surface_use_review["accepted"] is True
    assert bounded_language_surface_use_review["ready"] is True
    assert len(bounded_language_surface_use_review["review_hash"]) == 64
    assert bounded_language_surface_use_review["requires_operator_approval"] is False
    assert bounded_language_surface_use_review["advisory"] is True
    assert bounded_language_surface_use_review["executable"] is False
    assert bounded_language_surface_use_review["records_ledger_event"] is False
    assert bounded_language_surface_use_review["mutates_runtime_state"] is False
    assert bounded_language_surface_use_review["runs_replay"] is False
    assert bounded_language_surface_use_review["writes_checkpoint"] is False
    assert bounded_language_surface_use_review["generates_text"] is False
    assert bounded_language_surface_use_review["decodes_text"] is False
    assert bounded_language_surface_use_review["trains_runtime_model"] is False
    assert bounded_language_surface_use_review["applies_plasticity"] is False
    assert bounded_language_surface_use_review["rendered_text"] == (
        "\n".join(bounded_text_fragments)
    )
    assert bounded_language_surface_use_review["text_fragments"] == (
        bounded_text_fragments
    )
    bounded_language_surface_use_body = bounded_language_surface_use_review[
        "autonomous_bounded_language_surface_use_review"
    ]
    assert (
        bounded_language_surface_use_body["language_use_scope"]
        == "bounded_language_evidence"
    )
    assert bounded_language_surface_use_body["rendered_text"] == (
        "\n".join(bounded_text_fragments)
    )
    assert bounded_language_surface_use_body["text_fragments"] == (
        bounded_text_fragments
    )
    assert bounded_language_surface_use_body["replay_allowed"] is False
    assert bounded_language_surface_use_body["plasticity_allowed"] is False
    assert bounded_language_surface_use_body["fact_promotion_allowed"] is False
    assert bounded_language_surface_use_body["action_allowed"] is False
    assert bounded_language_surface_use_review["promotion_gate"][
        "eligible_for_autonomous_bounded_language_surface_use_preflight"
    ] is True
    assert bounded_language_surface_use_review["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert bounded_language_surface_use_review["promotion_gate"][
        "eligible_for_dense_readout_training"
    ] is False
    assert bounded_language_surface_use_review["promotion_gate"][
        "eligible_for_replay_memory"
    ] is False
    assert bounded_language_surface_use_review["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert bounded_language_surface_use_review["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert bounded_language_surface_use_review["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_bounded_language_surface_use_preflight["accepted"] is False
    assert (
        blocked_bounded_language_surface_use_preflight["requires_operator_approval"]
        is False
    )
    assert blocked_bounded_language_surface_use_preflight["promotion_gate"][
        "required_evidence"
    ]["executor_capability_available"] is False
    assert bounded_language_surface_use_preflight["surface"] == (
        "snn_language_autonomous_bounded_language_surface_use_preflight.v1"
    )
    assert bounded_language_surface_use_preflight["accepted"] is True
    assert bounded_language_surface_use_preflight["ready"] is True
    assert (
        len(
            bounded_language_surface_use_preflight[
                "bounded_language_surface_use_preflight_hash"
            ]
        )
        == 64
    )
    assert bounded_language_surface_use_preflight["requires_operator_approval"] is False
    assert bounded_language_surface_use_preflight["advisory"] is True
    assert bounded_language_surface_use_preflight["executable"] is False
    assert bounded_language_surface_use_preflight["records_ledger_event"] is False
    assert bounded_language_surface_use_preflight["mutates_runtime_state"] is False
    assert bounded_language_surface_use_preflight["runs_replay"] is False
    assert bounded_language_surface_use_preflight["writes_checkpoint"] is False
    assert bounded_language_surface_use_preflight["generates_text"] is False
    assert bounded_language_surface_use_preflight["decodes_text"] is False
    assert bounded_language_surface_use_preflight["trains_runtime_model"] is False
    assert bounded_language_surface_use_preflight["applies_plasticity"] is False
    bounded_language_surface_use_preflight_body = (
        bounded_language_surface_use_preflight[
            "autonomous_bounded_language_surface_use_preflight"
        ]
    )
    assert (
        bounded_language_surface_use_preflight_body["language_use_scope"]
        == "bounded_language_evidence"
    )
    assert bounded_language_surface_use_preflight_body["rendered_text"] == (
        "\n".join(bounded_text_fragments)
    )
    assert bounded_language_surface_use_preflight_body["text_fragments"] == (
        bounded_text_fragments
    )
    assert (
        bounded_language_surface_use_preflight_body["device_evidence"]["device"]
        == "cpu"
    )
    assert bounded_language_surface_use_preflight_body["replay_allowed"] is False
    assert bounded_language_surface_use_preflight_body["plasticity_allowed"] is False
    assert (
        bounded_language_surface_use_preflight_body["fact_promotion_allowed"]
        is False
    )
    assert bounded_language_surface_use_preflight_body["action_allowed"] is False
    assert bounded_language_surface_use_preflight["promotion_gate"][
        "eligible_for_autonomous_bounded_language_surface_use_executor"
    ] is True
    assert bounded_language_surface_use_preflight["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert bounded_language_surface_use_preflight["promotion_gate"][
        "eligible_for_dense_readout_training"
    ] is False
    assert bounded_language_surface_use_preflight["promotion_gate"][
        "eligible_for_replay_memory"
    ] is False
    assert bounded_language_surface_use_preflight["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert bounded_language_surface_use_preflight["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert bounded_language_surface_use_preflight["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_bounded_language_surface_use_executor["accepted"] is False
    assert (
        blocked_bounded_language_surface_use_executor["requires_operator_approval"]
        is False
    )
    assert blocked_bounded_language_surface_use_executor["promotion_gate"][
        "required_evidence"
    ]["preflight_ready"] is False
    _assert_record_family_source_window(
        blocked_bounded_language_surface_use_executor["source_window"],
        field="autonomous_bounded_language_surface_use_events",
        expected_count=0,
    )
    assert bounded_language_surface_use_executor["surface"] == (
        "snn_language_autonomous_bounded_language_surface_use_executor.v1"
    )
    assert bounded_language_surface_use_executor["accepted"] is True
    assert bounded_language_surface_use_executor["ready"] is True
    assert (
        len(
            bounded_language_surface_use_executor[
                "autonomous_bounded_language_surface_use_event_hash"
            ]
        )
        == 64
    )
    assert bounded_language_surface_use_executor["requires_operator_approval"] is False
    assert bounded_language_surface_use_executor["advisory"] is False
    assert bounded_language_surface_use_executor["executable"] is True
    assert bounded_language_surface_use_executor["records_ledger_event"] is True
    assert bounded_language_surface_use_executor["mutates_runtime_state"] is True
    assert bounded_language_surface_use_executor["runs_replay"] is False
    assert bounded_language_surface_use_executor["writes_checkpoint"] is False
    assert bounded_language_surface_use_executor["generates_text"] is False
    assert bounded_language_surface_use_executor["decodes_text"] is False
    assert bounded_language_surface_use_executor["trains_runtime_model"] is False
    assert bounded_language_surface_use_executor["applies_plasticity"] is False
    assert bounded_language_surface_use_executor["literal_text_returned"] is True
    assert (
        bounded_language_surface_use_executor["output_is_bounded_text_surface"]
        is True
    )
    assert bounded_language_surface_use_executor["rendered_text"] == (
        "\n".join(bounded_text_fragments)
    )
    assert bounded_language_surface_use_executor["text_fragments"] == (
        bounded_text_fragments
    )
    _assert_record_family_source_window(
        bounded_language_surface_use_executor["source_window"],
        field="autonomous_bounded_language_surface_use_events",
        expected_count=0,
    )
    assert (
        bounded_language_surface_use_executor["ledger_summary"][
            "total_autonomous_bounded_language_surface_use_count"
        ]
        == 1
    )
    bounded_language_surface_use_event = bounded_language_surface_use_executor[
        "autonomous_bounded_language_surface_use_event"
    ]
    assert bounded_language_surface_use_event["use_mode"] == (
        "bounded_language_evidence_observation"
    )
    assert (
        bounded_language_surface_use_event["language_use_scope"]
        == "bounded_language_evidence"
    )
    assert bounded_language_surface_use_event["rendered_text"] == (
        "\n".join(bounded_text_fragments)
    )
    assert bounded_language_surface_use_event["text_fragments"] == (
        bounded_text_fragments
    )
    assert (
        bounded_language_surface_use_event["used_language_surface_hash"]
        == bounded_language_surface_use_preflight["bounded_language_surface_hash"]
    )
    assert len(
        bounded_language_surface_use_event[
            "autonomous_bounded_language_surface_use_event_hash"
        ]
    ) == 64
    assert len(
        bounded_language_surface_use_event["language_surface_use_chain_hash"]
    ) == 64
    assert bounded_language_surface_use_event["runs_replay"] is False
    assert bounded_language_surface_use_event["writes_checkpoint"] is False
    assert bounded_language_surface_use_event["generates_text"] is False
    assert bounded_language_surface_use_event["decodes_text"] is False
    assert bounded_language_surface_use_event["trains_runtime_model"] is False
    assert bounded_language_surface_use_event["applies_plasticity"] is False
    assert bounded_language_surface_use_event["promotes_fact"] is False
    assert bounded_language_surface_use_event["executes_action"] is False
    assert ledger_state["total_autonomous_bounded_language_surface_use_count"] == 1
    assert (
        ledger_state["autonomous_bounded_language_surface_use_events"][0][
            "autonomous_bounded_language_surface_use_event_hash"
        ]
        == bounded_language_surface_use_executor[
            "autonomous_bounded_language_surface_use_event_hash"
        ]
    )
    assert bounded_language_surface_use_executor["promotion_gate"][
        "eligible_for_autonomous_bounded_language_surface_use_event_review"
    ] is True
    assert bounded_language_surface_use_executor["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert bounded_language_surface_use_executor["promotion_gate"][
        "eligible_for_dense_readout_training"
    ] is False
    assert bounded_language_surface_use_executor["promotion_gate"][
        "eligible_for_replay_memory"
    ] is False
    assert bounded_language_surface_use_executor["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert bounded_language_surface_use_executor["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert bounded_language_surface_use_executor["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_bounded_language_surface_use_event_review["accepted"] is False
    assert (
        blocked_bounded_language_surface_use_event_review[
            "requires_operator_approval"
        ]
        is False
    )
    assert blocked_bounded_language_surface_use_event_review["promotion_gate"][
        "required_evidence"
    ]["executor_accepted"] is False
    _assert_record_family_source_window(
        blocked_bounded_language_surface_use_event_review["source_window"],
        field="autonomous_bounded_language_surface_use_events",
        expected_count=1,
    )
    assert bounded_language_surface_use_event_review["surface"] == (
        "snn_language_autonomous_bounded_language_surface_use_event_review.v1"
    )
    assert bounded_language_surface_use_event_review["accepted"] is True
    assert bounded_language_surface_use_event_review["ready"] is True
    assert len(bounded_language_surface_use_event_review["review_hash"]) == 64
    assert (
        bounded_language_surface_use_event_review["requires_operator_approval"]
        is False
    )
    assert bounded_language_surface_use_event_review["advisory"] is True
    assert bounded_language_surface_use_event_review["executable"] is False
    assert (
        bounded_language_surface_use_event_review["records_ledger_event"]
        is False
    )
    assert (
        bounded_language_surface_use_event_review["mutates_runtime_state"]
        is False
    )
    assert bounded_language_surface_use_event_review["runs_replay"] is False
    assert bounded_language_surface_use_event_review["writes_checkpoint"] is False
    assert bounded_language_surface_use_event_review["generates_text"] is False
    assert bounded_language_surface_use_event_review["decodes_text"] is False
    assert (
        bounded_language_surface_use_event_review["trains_runtime_model"]
        is False
    )
    assert bounded_language_surface_use_event_review["applies_plasticity"] is False
    _assert_record_family_source_window(
        bounded_language_surface_use_event_review["source_window"],
        field="autonomous_bounded_language_surface_use_events",
        expected_count=1,
    )
    assert bounded_language_surface_use_event_review["rendered_text"] == (
        "\n".join(bounded_text_fragments)
    )
    assert bounded_language_surface_use_event_review["text_fragments"] == (
        bounded_text_fragments
    )
    bounded_language_surface_use_event_review_body = (
        bounded_language_surface_use_event_review[
            "autonomous_bounded_language_surface_use_event_review"
        ]
    )
    assert (
        bounded_language_surface_use_event_review_body["event_recorded_in_ledger"]
        is True
    )
    assert (
        bounded_language_surface_use_event_review_body[
            "autonomous_bounded_language_surface_use_event_hash"
        ]
        == bounded_language_surface_use_executor[
            "autonomous_bounded_language_surface_use_event_hash"
        ]
    )
    assert (
        bounded_language_surface_use_event_review_body["language_use_scope"]
        == "bounded_language_evidence"
    )
    assert bounded_language_surface_use_event_review_body["use_mode"] == (
        "bounded_language_evidence_observation"
    )
    assert (
        bounded_language_surface_use_event_review_body[
            "language_surface_use_chain_hash"
        ]
        == bounded_language_surface_use_event["language_surface_use_chain_hash"]
    )
    assert bounded_language_surface_use_event_review["promotion_gate"][
        "eligible_for_autonomous_snn_language_generation_design"
    ] is True
    assert bounded_language_surface_use_event_review["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert bounded_language_surface_use_event_review["promotion_gate"][
        "eligible_for_dense_readout_training"
    ] is False
    assert bounded_language_surface_use_event_review["promotion_gate"][
        "eligible_for_replay_memory"
    ] is False
    assert bounded_language_surface_use_event_review["promotion_gate"][
        "eligible_for_plasticity_application"
    ] is False
    assert bounded_language_surface_use_event_review["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert bounded_language_surface_use_event_review["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_generation_design["accepted"] is False
    assert blocked_snn_language_generation_design["requires_operator_approval"] is False
    assert blocked_snn_language_generation_design["promotion_gate"][
        "required_evidence"
    ]["use_event_review_ready"] is False
    assert snn_language_generation_design["surface"] == (
        "snn_language_autonomous_snn_language_generation_design.v1"
    )
    assert snn_language_generation_design["accepted"] is True
    assert snn_language_generation_design["ready"] is True
    assert len(snn_language_generation_design["language_generation_design_hash"]) == 64
    assert snn_language_generation_design["requires_operator_approval"] is False
    assert snn_language_generation_design["advisory"] is True
    assert snn_language_generation_design["executable"] is False
    assert snn_language_generation_design["records_ledger_event"] is False
    assert snn_language_generation_design["mutates_runtime_state"] is False
    assert snn_language_generation_design["runs_replay"] is False
    assert snn_language_generation_design["writes_checkpoint"] is False
    assert snn_language_generation_design["generates_text"] is False
    assert snn_language_generation_design["decodes_text"] is False
    assert snn_language_generation_design["trains_runtime_model"] is False
    assert snn_language_generation_design["applies_plasticity"] is False
    assert snn_language_generation_design["planned_generation"] is True
    assert snn_language_generation_design["planned_snn_native_generation"] is True
    snn_language_generation_design_body = snn_language_generation_design[
        "autonomous_snn_language_generation_design"
    ]
    assert snn_language_generation_design_body["generation_mode"] == (
        "snn_bounded_next_token_projection"
    )
    assert snn_language_generation_design_body["decoding_strategy"] == (
        "spike_sparse_top_k"
    )
    assert snn_language_generation_design_body["max_new_tokens"] == 16
    assert snn_language_generation_design_body["target_device"] == "cpu"
    assert snn_language_generation_design_body["requires_cuda"] is False
    assert snn_language_generation_design_body["literal_text_returned"] is False
    assert snn_language_generation_design_body["generated_text_returned"] is False
    assert len(snn_language_generation_design_body["generation_plan_hash"]) == 64
    assert (
        snn_language_generation_design_body[
            "autonomous_bounded_language_surface_use_event_hash"
        ]
        == bounded_language_surface_use_executor[
            "autonomous_bounded_language_surface_use_event_hash"
        ]
    )
    assert snn_language_generation_design["promotion_gate"][
        "eligible_for_autonomous_snn_language_generation_preflight"
    ] is True
    assert snn_language_generation_design["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert snn_language_generation_design["promotion_gate"][
        "eligible_for_freeform_language_generation"
    ] is False
    assert snn_language_generation_design["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_generation_design["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_generation_preflight["accepted"] is False
    assert blocked_snn_language_generation_preflight["requires_operator_approval"] is False
    assert blocked_snn_language_generation_preflight["promotion_gate"][
        "required_evidence"
    ]["generation_design_ready"] is False
    assert snn_language_generation_preflight["surface"] == (
        "snn_language_autonomous_snn_language_generation_preflight.v1"
    )
    assert snn_language_generation_preflight["accepted"] is True
    assert snn_language_generation_preflight["ready"] is True
    assert (
        len(
            snn_language_generation_preflight[
                "language_generation_preflight_hash"
            ]
        )
        == 64
    )
    assert snn_language_generation_preflight["requires_operator_approval"] is False
    assert snn_language_generation_preflight["advisory"] is True
    assert snn_language_generation_preflight["executable"] is False
    assert snn_language_generation_preflight["records_ledger_event"] is False
    assert snn_language_generation_preflight["mutates_runtime_state"] is False
    assert snn_language_generation_preflight["runs_replay"] is False
    assert snn_language_generation_preflight["writes_checkpoint"] is False
    assert snn_language_generation_preflight["generates_text"] is False
    assert snn_language_generation_preflight["decodes_text"] is False
    assert snn_language_generation_preflight["trains_runtime_model"] is False
    assert snn_language_generation_preflight["applies_plasticity"] is False
    snn_language_generation_preflight_body = snn_language_generation_preflight[
        "autonomous_snn_language_generation_preflight"
    ]
    assert snn_language_generation_preflight_body["generation_mode"] == (
        "snn_bounded_next_token_projection"
    )
    assert snn_language_generation_preflight_body["requested_device"] == "cpu"
    assert snn_language_generation_preflight_body["requires_cuda"] is False
    assert (
        snn_language_generation_preflight_body["execution_allowed"]
        is False
    )
    assert (
        snn_language_generation_preflight_body["generated_text_allowed"]
        is False
    )
    assert snn_language_generation_preflight["promotion_gate"][
        "eligible_for_autonomous_snn_language_generation_executor"
    ] is True
    assert snn_language_generation_preflight["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert snn_language_generation_preflight["promotion_gate"][
        "eligible_for_freeform_language_generation"
    ] is False
    assert snn_language_generation_preflight["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_generation_preflight["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_generation_executor["accepted"] is False
    assert blocked_snn_language_generation_executor["requires_operator_approval"] is False
    assert blocked_snn_language_generation_executor["promotion_gate"][
        "required_evidence"
    ]["preflight_ready"] is False
    _assert_record_family_source_window(
        blocked_snn_language_generation_executor["source_window"],
        field="autonomous_snn_language_generation_events",
        expected_count=0,
    )
    assert snn_language_generation_executor["surface"] == (
        "snn_language_autonomous_snn_language_generation_executor.v1"
    )
    assert snn_language_generation_executor["accepted"] is True
    assert snn_language_generation_executor["ready"] is True
    assert (
        len(
            snn_language_generation_executor[
                "autonomous_snn_language_generation_event_hash"
            ]
        )
        == 64
    )
    assert snn_language_generation_executor["requires_operator_approval"] is False
    assert snn_language_generation_executor["advisory"] is False
    assert snn_language_generation_executor["executable"] is True
    assert snn_language_generation_executor["records_ledger_event"] is True
    assert snn_language_generation_executor["mutates_runtime_state"] is True
    assert snn_language_generation_executor["runs_replay"] is False
    assert snn_language_generation_executor["writes_checkpoint"] is False
    assert snn_language_generation_executor["generates_text"] is False
    assert snn_language_generation_executor["decodes_text"] is False
    assert snn_language_generation_executor["trains_runtime_model"] is False
    assert snn_language_generation_executor["applies_plasticity"] is False
    assert snn_language_generation_executor["generated_text_returned"] is False
    assert snn_language_generation_executor["generated_token_hashes"] == (
        generation_token_hashes
    )
    _assert_record_family_source_window(
        snn_language_generation_executor["source_window"],
        field="autonomous_snn_language_generation_events",
        expected_count=0,
    )
    assert (
        snn_language_generation_executor["ledger_summary"][
            "total_autonomous_snn_language_generation_count"
        ]
        == 1
    )
    snn_language_generation_event = snn_language_generation_executor[
        "autonomous_snn_language_generation_event"
    ]
    assert snn_language_generation_event["generated_token_hashes"] == (
        generation_token_hashes
    )
    assert snn_language_generation_event["spike_projection_hashes"] == (
        generation_spike_hashes
    )
    assert snn_language_generation_event["active_neuron_hashes"] == (
        generation_active_hashes
    )
    assert len(snn_language_generation_event["generation_projection_hash"]) == 64
    assert ledger_state["total_autonomous_snn_language_generation_count"] == 1
    assert (
        ledger_state["autonomous_snn_language_generation_events"][0][
            "autonomous_snn_language_generation_event_hash"
        ]
        == snn_language_generation_executor[
            "autonomous_snn_language_generation_event_hash"
        ]
    )
    assert snn_language_generation_executor["promotion_gate"][
        "eligible_for_autonomous_snn_language_generation_event_review"
    ] is True
    assert snn_language_generation_executor["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert snn_language_generation_executor["promotion_gate"][
        "eligible_for_freeform_language_generation"
    ] is False
    assert snn_language_generation_executor["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_generation_executor["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_generation_event_review["accepted"] is False
    assert blocked_snn_language_generation_event_review["requires_operator_approval"] is False
    assert blocked_snn_language_generation_event_review["promotion_gate"][
        "required_evidence"
    ]["executor_accepted"] is False
    _assert_record_family_source_window(
        blocked_snn_language_generation_event_review["source_window"],
        field="autonomous_snn_language_generation_events",
        expected_count=1,
    )
    assert snn_language_generation_event_review["surface"] == (
        "snn_language_autonomous_snn_language_generation_event_review.v1"
    )
    assert snn_language_generation_event_review["accepted"] is True
    assert snn_language_generation_event_review["ready"] is True
    assert len(snn_language_generation_event_review["review_hash"]) == 64
    assert snn_language_generation_event_review["requires_operator_approval"] is False
    assert snn_language_generation_event_review["advisory"] is True
    assert snn_language_generation_event_review["executable"] is False
    assert snn_language_generation_event_review["records_ledger_event"] is False
    assert snn_language_generation_event_review["mutates_runtime_state"] is False
    assert snn_language_generation_event_review["runs_replay"] is False
    assert snn_language_generation_event_review["writes_checkpoint"] is False
    assert snn_language_generation_event_review["generates_text"] is False
    assert snn_language_generation_event_review["decodes_text"] is False
    assert snn_language_generation_event_review["trains_runtime_model"] is False
    assert snn_language_generation_event_review["applies_plasticity"] is False
    assert snn_language_generation_event_review["generated_text_returned"] is False
    assert snn_language_generation_event_review["generated_token_hashes"] == (
        generation_token_hashes
    )
    _assert_record_family_source_window(
        snn_language_generation_event_review["source_window"],
        field="autonomous_snn_language_generation_events",
        expected_count=1,
    )
    snn_language_generation_review_body = snn_language_generation_event_review[
        "autonomous_snn_language_generation_event_review"
    ]
    assert snn_language_generation_review_body["event_recorded_in_ledger"] is True
    assert (
        snn_language_generation_review_body[
            "autonomous_snn_language_generation_event_hash"
        ]
        == snn_language_generation_executor[
            "autonomous_snn_language_generation_event_hash"
        ]
    )
    assert snn_language_generation_review_body["generated_token_hashes"] == (
        generation_token_hashes
    )
    assert snn_language_generation_review_body["spike_projection_hashes"] == (
        generation_spike_hashes
    )
    assert snn_language_generation_event_review["promotion_gate"][
        "eligible_for_autonomous_snn_language_decoding_design"
    ] is True
    assert snn_language_generation_event_review["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert snn_language_generation_event_review["promotion_gate"][
        "eligible_for_freeform_language_generation"
    ] is False
    assert snn_language_generation_event_review["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_generation_event_review["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_decoding_design["accepted"] is False
    assert blocked_snn_language_decoding_design["requires_operator_approval"] is False
    assert blocked_snn_language_decoding_design["promotion_gate"][
        "required_evidence"
    ]["event_review_ready"] is False
    assert snn_language_decoding_design["surface"] == (
        "snn_language_autonomous_snn_language_decoding_design.v1"
    )
    assert snn_language_decoding_design["accepted"] is True
    assert snn_language_decoding_design["ready"] is True
    assert len(snn_language_decoding_design["language_decoding_design_hash"]) == 64
    assert snn_language_decoding_design["requires_operator_approval"] is False
    assert snn_language_decoding_design["advisory"] is True
    assert snn_language_decoding_design["executable"] is False
    assert snn_language_decoding_design["records_ledger_event"] is False
    assert snn_language_decoding_design["mutates_runtime_state"] is False
    assert snn_language_decoding_design["runs_replay"] is False
    assert snn_language_decoding_design["writes_checkpoint"] is False
    assert snn_language_decoding_design["generates_text"] is False
    assert snn_language_decoding_design["decodes_text"] is False
    assert snn_language_decoding_design["trains_runtime_model"] is False
    assert snn_language_decoding_design["applies_plasticity"] is False
    assert snn_language_decoding_design["literal_text_returned"] is False
    assert snn_language_decoding_design["generated_text_returned"] is False
    assert snn_language_decoding_design["generated_token_hashes"] == (
        generation_token_hashes
    )
    snn_language_decoding_body = snn_language_decoding_design[
        "autonomous_snn_language_decoding_design"
    ]
    assert snn_language_decoding_body["decoding_mode"] == (
        "bounded_hash_token_projection"
    )
    assert snn_language_decoding_body["materialization_target"] == (
        "bounded_text_surface"
    )
    assert snn_language_decoding_body["max_decoded_tokens"] == 2
    assert snn_language_decoding_body["max_decoded_fragments"] == 1
    assert snn_language_decoding_body["max_surface_chars"] == 256
    assert len(snn_language_decoding_body["decoding_plan_hash"]) == 64
    assert snn_language_decoding_body["generated_token_hashes"] == (
        generation_token_hashes
    )
    assert snn_language_decoding_body["spike_projection_hashes"] == (
        generation_spike_hashes
    )
    assert snn_language_decoding_body["active_neuron_hashes"] == (
        generation_active_hashes
    )
    assert (
        snn_language_decoding_body["autonomous_snn_language_generation_event_hash"]
        == snn_language_generation_executor[
            "autonomous_snn_language_generation_event_hash"
        ]
    )
    assert snn_language_decoding_body["generated_text_returned"] is False
    assert snn_language_decoding_body["decoding_allowed"] is False
    assert snn_language_decoding_design["promotion_gate"][
        "eligible_for_autonomous_snn_language_decoding_preflight"
    ] is True
    assert snn_language_decoding_design["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert snn_language_decoding_design["promotion_gate"][
        "eligible_for_freeform_language_generation"
    ] is False
    assert snn_language_decoding_design["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_decoding_design["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_decoding_preflight["accepted"] is False
    assert (
        blocked_snn_language_decoding_preflight["requires_operator_approval"]
        is False
    )
    assert blocked_snn_language_decoding_preflight["promotion_gate"][
        "required_evidence"
    ]["decoding_design_ready"] is False
    assert snn_language_decoding_preflight["surface"] == (
        "snn_language_autonomous_snn_language_decoding_preflight.v1"
    )
    assert snn_language_decoding_preflight["accepted"] is True
    assert snn_language_decoding_preflight["ready"] is True
    assert len(snn_language_decoding_preflight["preflight_hash"]) == 64
    assert snn_language_decoding_preflight["requires_operator_approval"] is False
    assert snn_language_decoding_preflight["advisory"] is True
    assert snn_language_decoding_preflight["executable"] is True
    assert snn_language_decoding_preflight["records_ledger_event"] is False
    assert snn_language_decoding_preflight["mutates_runtime_state"] is False
    assert snn_language_decoding_preflight["runs_replay"] is False
    assert snn_language_decoding_preflight["writes_checkpoint"] is False
    assert snn_language_decoding_preflight["generates_text"] is False
    assert snn_language_decoding_preflight["decodes_text"] is False
    assert snn_language_decoding_preflight["trains_runtime_model"] is False
    assert snn_language_decoding_preflight["applies_plasticity"] is False
    assert snn_language_decoding_preflight["literal_text_returned"] is False
    assert snn_language_decoding_preflight["generated_text_returned"] is False
    assert snn_language_decoding_preflight["generated_token_hashes"] == (
        generation_token_hashes
    )
    snn_language_decoding_preflight_body = snn_language_decoding_preflight[
        "autonomous_snn_language_decoding_preflight"
    ]
    assert snn_language_decoding_preflight_body["requested_device"] == "cuda:0"
    assert snn_language_decoding_preflight_body["requires_cuda"] is True
    assert snn_language_decoding_preflight_body["cuda_satisfied"] is True
    assert snn_language_decoding_preflight_body["decoder_ready"] is True
    assert snn_language_decoding_preflight_body["execution_allowed"] is True
    assert snn_language_decoding_preflight_body["decoding_allowed"] is False
    assert snn_language_decoding_preflight_body["generated_token_hashes"] == (
        generation_token_hashes
    )
    assert snn_language_decoding_preflight["promotion_gate"][
        "eligible_for_autonomous_snn_language_decoding_executor"
    ] is True
    assert snn_language_decoding_preflight["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert snn_language_decoding_preflight["promotion_gate"][
        "eligible_for_freeform_language_generation"
    ] is False
    assert snn_language_decoding_preflight["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_decoding_preflight["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_decoding_executor["accepted"] is False
    assert blocked_snn_language_decoding_executor["requires_operator_approval"] is False
    assert blocked_snn_language_decoding_executor["promotion_gate"][
        "required_evidence"
    ]["preflight_ready"] is False
    _assert_record_family_source_window(
        blocked_snn_language_decoding_executor["source_window"],
        field="autonomous_snn_language_decoding_events",
        expected_count=0,
    )
    assert snn_language_decoding_executor["surface"] == (
        "snn_language_autonomous_snn_language_decoding_executor.v1"
    )
    assert snn_language_decoding_executor["accepted"] is True
    assert snn_language_decoding_executor["ready"] is True
    assert len(
        snn_language_decoding_executor[
            "autonomous_snn_language_decoding_event_hash"
        ]
    ) == 64
    assert snn_language_decoding_executor["requires_operator_approval"] is False
    assert snn_language_decoding_executor["advisory"] is False
    assert snn_language_decoding_executor["executable"] is True
    assert snn_language_decoding_executor["records_ledger_event"] is True
    assert snn_language_decoding_executor["mutates_runtime_state"] is True
    assert snn_language_decoding_executor["runs_replay"] is False
    assert snn_language_decoding_executor["writes_checkpoint"] is False
    assert snn_language_decoding_executor["generates_text"] is True
    assert snn_language_decoding_executor["decodes_text"] is True
    assert snn_language_decoding_executor["trains_runtime_model"] is False
    assert snn_language_decoding_executor["applies_plasticity"] is False
    assert snn_language_decoding_executor["literal_text_returned"] is True
    assert snn_language_decoding_executor["generated_text_returned"] is True
    assert snn_language_decoding_executor["rendered_text"] == "spike language"
    assert snn_language_decoding_executor["decoded_token_hashes"] == (
        generation_token_hashes
    )
    assert len(snn_language_decoding_executor["rendered_text_hash"]) == 64
    assert len(snn_language_decoding_executor["decoded_text_fragment_hashes"]) == 1
    snn_language_decoding_event = snn_language_decoding_executor[
        "autonomous_snn_language_decoding_event"
    ]
    assert snn_language_decoding_event["decoded_text_fragments"] == [
        "spike language"
    ]
    assert snn_language_decoding_event["rendered_text"] == "spike language"
    assert snn_language_decoding_event["semantic_constraint_valid"] is True
    assert snn_language_decoding_event["text_normalized"] is True
    assert snn_language_decoding_event["schema_valid"] is True
    assert snn_language_decoding_event["promotes_fact"] is False
    assert snn_language_decoding_event["executes_action"] is False
    assert snn_language_decoding_executor["ledger_summary"][
        "total_autonomous_snn_language_decoding_count"
    ] == 1
    _assert_record_family_source_window(
        snn_language_decoding_executor["source_window"],
        field="autonomous_snn_language_decoding_events",
        expected_count=0,
    )
    assert snn_language_decoding_executor["promotion_gate"][
        "eligible_for_autonomous_snn_language_decoding_event_review"
    ] is True
    assert snn_language_decoding_executor["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert snn_language_decoding_executor["promotion_gate"][
        "eligible_for_freeform_language_generation"
    ] is False
    assert snn_language_decoding_executor["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_decoding_executor["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_decoding_event_review["accepted"] is False
    assert (
        blocked_snn_language_decoding_event_review["requires_operator_approval"]
        is False
    )
    assert blocked_snn_language_decoding_event_review["promotion_gate"][
        "required_evidence"
    ]["executor_accepted"] is False
    _assert_record_family_source_window(
        blocked_snn_language_decoding_event_review["source_window"],
        field="autonomous_snn_language_decoding_events",
        expected_count=1,
    )
    assert snn_language_decoding_event_review["surface"] == (
        "snn_language_autonomous_snn_language_decoding_event_review.v1"
    )
    assert snn_language_decoding_event_review["accepted"] is True
    assert snn_language_decoding_event_review["ready"] is True
    assert len(snn_language_decoding_event_review["review_hash"]) == 64
    assert snn_language_decoding_event_review["requires_operator_approval"] is False
    assert snn_language_decoding_event_review["advisory"] is True
    assert snn_language_decoding_event_review["executable"] is False
    assert snn_language_decoding_event_review["records_ledger_event"] is False
    assert snn_language_decoding_event_review["mutates_runtime_state"] is False
    assert snn_language_decoding_event_review["runs_replay"] is False
    assert snn_language_decoding_event_review["writes_checkpoint"] is False
    assert snn_language_decoding_event_review["generates_text"] is True
    assert snn_language_decoding_event_review["decodes_text"] is True
    assert snn_language_decoding_event_review["trains_runtime_model"] is False
    assert snn_language_decoding_event_review["applies_plasticity"] is False
    assert snn_language_decoding_event_review["literal_text_returned"] is True
    assert snn_language_decoding_event_review["generated_text_returned"] is True
    assert snn_language_decoding_event_review["rendered_text"] == "spike language"
    snn_language_decoding_review_body = snn_language_decoding_event_review[
        "autonomous_snn_language_decoding_event_review"
    ]
    assert snn_language_decoding_review_body["event_recorded_in_ledger"] is True
    assert snn_language_decoding_review_body["rendered_text"] == "spike language"
    assert snn_language_decoding_review_body["decoded_text_fragments"] == [
        "spike language"
    ]
    assert snn_language_decoding_review_body["decoded_token_hashes"] == (
        generation_token_hashes
    )
    assert snn_language_decoding_review_body["schema_valid"] is True
    assert snn_language_decoding_review_body["text_normalized"] is True
    assert snn_language_decoding_review_body["semantic_constraint_valid"] is True
    _assert_record_family_source_window(
        snn_language_decoding_event_review["source_window"],
        field="autonomous_snn_language_decoding_events",
        expected_count=1,
    )
    assert snn_language_decoding_event_review["promotion_gate"][
        "eligible_for_snn_language_readout_surface_design"
    ] is True
    assert snn_language_decoding_event_review["promotion_gate"][
        "eligible_for_language_generation"
    ] is False
    assert snn_language_decoding_event_review["promotion_gate"][
        "eligible_for_freeform_language_generation"
    ] is False
    assert snn_language_decoding_event_review["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_decoding_event_review["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_readout_surface_design["accepted"] is False
    assert (
        blocked_snn_language_readout_surface_design["requires_operator_approval"]
        is False
    )
    assert blocked_snn_language_readout_surface_design["promotion_gate"][
        "required_evidence"
    ]["decoding_event_review_ready"] is False
    assert snn_language_readout_surface_design["surface"] == (
        "snn_language_readout_surface_design.v1"
    )
    assert snn_language_readout_surface_design["accepted"] is True
    assert snn_language_readout_surface_design["ready"] is True
    assert len(
        snn_language_readout_surface_design[
            "language_readout_surface_design_hash"
        ]
    ) == 64
    assert snn_language_readout_surface_design["requires_operator_approval"] is False
    assert snn_language_readout_surface_design["advisory"] is True
    assert snn_language_readout_surface_design["executable"] is False
    assert snn_language_readout_surface_design["records_ledger_event"] is False
    assert snn_language_readout_surface_design["mutates_runtime_state"] is False
    assert snn_language_readout_surface_design["runs_replay"] is False
    assert snn_language_readout_surface_design["writes_checkpoint"] is False
    assert snn_language_readout_surface_design["generates_text"] is True
    assert snn_language_readout_surface_design["decodes_text"] is True
    assert snn_language_readout_surface_design["trains_runtime_model"] is False
    assert snn_language_readout_surface_design["applies_plasticity"] is False
    assert snn_language_readout_surface_design["literal_text_returned"] is True
    readout_body = snn_language_readout_surface_design[
        "snn_language_readout_surface_design"
    ]
    assert readout_body["readout_role"] == "bounded_readout_candidate"
    assert readout_body["binding_mode"] == "hash_bound_readout_language"
    assert readout_body["rendered_text"] == "spike language"
    assert readout_body["decoded_text_fragments"] == ["spike language"]
    assert len(readout_body["readout_surface_hash"]) == 64
    assert readout_body["fact_promotion_allowed"] is False
    assert readout_body["action_allowed"] is False
    assert readout_body["replay_allowed"] is False
    assert readout_body["plasticity_allowed"] is False
    assert snn_language_readout_surface_design["promotion_gate"][
        "eligible_for_snn_language_readout_surface_preflight"
    ] is True
    assert snn_language_readout_surface_design["promotion_gate"][
        "eligible_for_cognition_substrate"
    ] is False
    assert snn_language_readout_surface_design["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_readout_surface_design["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_readout_surface_preflight["accepted"] is False
    assert (
        blocked_snn_language_readout_surface_preflight[
            "requires_operator_approval"
        ]
        is False
    )
    assert blocked_snn_language_readout_surface_preflight["promotion_gate"][
        "required_evidence"
    ]["readout_surface_design_ready"] is False
    assert snn_language_readout_surface_preflight["surface"] == (
        "snn_language_readout_surface_preflight.v1"
    )
    assert snn_language_readout_surface_preflight["accepted"] is True
    assert snn_language_readout_surface_preflight["ready"] is True
    assert len(snn_language_readout_surface_preflight["preflight_hash"]) == 64
    assert snn_language_readout_surface_preflight["requires_operator_approval"] is False
    assert snn_language_readout_surface_preflight["advisory"] is True
    assert snn_language_readout_surface_preflight["executable"] is True
    assert (
        snn_language_readout_surface_preflight["records_ledger_event"]
        is False
    )
    assert snn_language_readout_surface_preflight["mutates_runtime_state"] is False
    assert snn_language_readout_surface_preflight["runs_replay"] is False
    assert snn_language_readout_surface_preflight["writes_checkpoint"] is False
    assert snn_language_readout_surface_preflight["generates_text"] is True
    assert snn_language_readout_surface_preflight["decodes_text"] is True
    assert snn_language_readout_surface_preflight["trains_runtime_model"] is False
    assert snn_language_readout_surface_preflight["applies_plasticity"] is False
    preflight_body = snn_language_readout_surface_preflight[
        "snn_language_readout_surface_preflight"
    ]
    assert preflight_body["readout_role"] == "bounded_readout_candidate"
    assert preflight_body["binding_mode"] == "hash_bound_readout_language"
    assert preflight_body["requested_device"] == "cuda:0"
    assert preflight_body["requires_cuda"] is True
    assert preflight_body["cuda_satisfied"] is True
    assert preflight_body["executor_ready"] is True
    assert preflight_body["execution_allowed"] is True
    assert preflight_body["rendered_text"] == "spike language"
    assert preflight_body["decoded_text_fragments"] == ["spike language"]
    assert len(preflight_body["readout_surface_hash"]) == 64
    assert preflight_body["fact_promotion_allowed"] is False
    assert preflight_body["action_allowed"] is False
    assert preflight_body["replay_allowed"] is False
    assert preflight_body["plasticity_allowed"] is False
    assert snn_language_readout_surface_preflight["promotion_gate"][
        "eligible_for_snn_language_readout_surface_executor"
    ] is True
    assert snn_language_readout_surface_preflight["promotion_gate"][
        "eligible_for_cognition_substrate"
    ] is False
    assert snn_language_readout_surface_preflight["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_readout_surface_preflight["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_readout_surface_executor["accepted"] is False
    assert (
        blocked_snn_language_readout_surface_executor["requires_operator_approval"]
        is False
    )
    assert blocked_snn_language_readout_surface_executor["promotion_gate"][
        "required_evidence"
    ]["preflight_ready"] is False
    _assert_record_family_source_window(
        blocked_snn_language_readout_surface_executor["source_window"],
        field="snn_language_readout_surface_events",
        expected_count=0,
    )
    assert snn_language_readout_surface_executor["surface"] == (
        "snn_language_readout_surface_executor.v1"
    )
    assert snn_language_readout_surface_executor["accepted"] is True
    assert snn_language_readout_surface_executor["ready"] is True
    assert len(
        snn_language_readout_surface_executor[
            "snn_language_readout_surface_event_hash"
        ]
    ) == 64
    assert snn_language_readout_surface_executor["requires_operator_approval"] is False
    assert snn_language_readout_surface_executor["advisory"] is False
    assert snn_language_readout_surface_executor["executable"] is True
    assert snn_language_readout_surface_executor["records_ledger_event"] is True
    assert snn_language_readout_surface_executor["mutates_runtime_state"] is True
    assert snn_language_readout_surface_executor["runs_replay"] is False
    assert snn_language_readout_surface_executor["writes_checkpoint"] is False
    assert snn_language_readout_surface_executor["generates_text"] is True
    assert snn_language_readout_surface_executor["decodes_text"] is True
    assert snn_language_readout_surface_executor["trains_runtime_model"] is False
    assert snn_language_readout_surface_executor["applies_plasticity"] is False
    readout_event = snn_language_readout_surface_executor[
        "snn_language_readout_surface_event"
    ]
    assert readout_event["readout_role"] == "bounded_readout_candidate"
    assert readout_event["binding_mode"] == "hash_bound_readout_language"
    assert readout_event["rendered_text"] == "spike language"
    assert readout_event["decoded_text_fragments"] == ["spike language"]
    assert len(readout_event["readout_surface_hash"]) == 64
    assert readout_event["promotes_fact"] is False
    assert readout_event["executes_action"] is False
    assert readout_event["runs_replay"] is False
    assert readout_event["applies_plasticity"] is False
    assert readout_event["cognition_substrate_claimed"] is False
    assert ledger_state["total_snn_language_readout_surface_count"] == 1
    assert (
        len(ledger_state["snn_language_readout_surface_events"])
        == 1
    )
    assert "total_autonomous_snn_language_thought_surface_count" not in ledger_state
    assert "autonomous_snn_language_thought_surface_events" not in ledger_state
    assert (
        "last_autonomous_snn_language_thought_surface_recorded_at"
        not in ledger_state
    )
    _assert_record_family_source_window(
        snn_language_readout_surface_executor["source_window"],
        field="snn_language_readout_surface_events",
        expected_count=0,
    )
    assert snn_language_readout_surface_executor["promotion_gate"][
        "eligible_for_snn_language_readout_surface_event_review"
    ] is True
    assert snn_language_readout_surface_executor["promotion_gate"][
        "eligible_for_cognition_substrate"
    ] is False
    assert snn_language_readout_surface_executor["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_readout_surface_executor["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_readout_surface_event_review["accepted"] is False
    assert (
        blocked_snn_language_readout_surface_event_review[
            "requires_operator_approval"
        ]
        is False
    )
    assert blocked_snn_language_readout_surface_event_review["promotion_gate"][
        "required_evidence"
    ]["executor_accepted"] is False
    _assert_record_family_source_window(
        blocked_snn_language_readout_surface_event_review["source_window"],
        field="snn_language_readout_surface_events",
        expected_count=1,
    )
    assert snn_language_readout_surface_event_review["surface"] == (
        "snn_language_readout_surface_event_review.v1"
    )
    assert snn_language_readout_surface_event_review["accepted"] is True
    assert snn_language_readout_surface_event_review["ready"] is True
    assert len(snn_language_readout_surface_event_review["review_hash"]) == 64
    assert snn_language_readout_surface_event_review[
        "requires_operator_approval"
    ] is False
    assert snn_language_readout_surface_event_review["advisory"] is True
    assert snn_language_readout_surface_event_review["executable"] is False
    assert snn_language_readout_surface_event_review["records_ledger_event"] is False
    assert snn_language_readout_surface_event_review["mutates_runtime_state"] is False
    assert snn_language_readout_surface_event_review["runs_replay"] is False
    assert snn_language_readout_surface_event_review["writes_checkpoint"] is False
    assert snn_language_readout_surface_event_review["generates_text"] is True
    assert snn_language_readout_surface_event_review["decodes_text"] is True
    assert snn_language_readout_surface_event_review["trains_runtime_model"] is False
    assert snn_language_readout_surface_event_review["applies_plasticity"] is False
    readout_review = snn_language_readout_surface_event_review[
        "snn_language_readout_surface_event_review"
    ]
    assert readout_review["event_recorded_in_ledger"] is True
    assert readout_review["readout_role"] == "bounded_readout_candidate"
    assert readout_review["binding_mode"] == "hash_bound_readout_language"
    assert readout_review["rendered_text"] == "spike language"
    assert readout_review["decoded_text_fragments"] == ["spike language"]
    assert len(readout_review["readout_surface_hash"]) == 64
    assert readout_review["fact_promotion_allowed"] is False
    assert readout_review["action_allowed"] is False
    assert readout_review["replay_allowed"] is False
    assert readout_review["plasticity_allowed"] is False
    assert readout_review["cognition_substrate_claimed"] is False
    _assert_record_family_source_window(
        snn_language_readout_surface_event_review["source_window"],
        field="snn_language_readout_surface_events",
        expected_count=1,
    )
    surface_chain_text = json.dumps(
        [
            snn_language_readout_surface_design,
            snn_language_readout_surface_preflight,
            snn_language_readout_surface_executor,
            snn_language_readout_surface_event_review,
        ],
        sort_keys=True,
    )
    assert "thought_surface" not in surface_chain_text
    assert "autonomous_snn_language_thought_surface" not in surface_chain_text
    assert (
        "snn_language_autonomous_snn_language_thought_surface"
        not in surface_chain_text
    )
    assert snn_language_readout_surface_event_review["promotion_gate"][
        "eligible_for_snn_language_readout_memory_design"
    ] is True
    assert snn_language_readout_surface_event_review["promotion_gate"][
        "eligible_for_cognition_substrate"
    ] is False
    assert snn_language_readout_surface_event_review["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_readout_surface_event_review["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_readout_memory_design["accepted"] is False
    assert (
        blocked_snn_language_readout_memory_design["requires_operator_approval"]
        is False
    )
    assert blocked_snn_language_readout_memory_design["promotion_gate"][
        "required_evidence"
    ]["readout_surface_event_review_ready"] is False
    assert snn_language_readout_memory_design["surface"] == (
        "snn_language_readout_memory_design.v1"
    )
    assert snn_language_readout_memory_design["accepted"] is True
    assert snn_language_readout_memory_design["ready"] is True
    assert len(
        snn_language_readout_memory_design[
            "language_readout_memory_design_hash"
        ]
    ) == 64
    assert len(snn_language_readout_memory_design["memory_trace_hash"]) == 64
    assert snn_language_readout_memory_design["requires_operator_approval"] is False
    assert snn_language_readout_memory_design["advisory"] is True
    assert snn_language_readout_memory_design["executable"] is False
    assert snn_language_readout_memory_design["records_ledger_event"] is False
    assert snn_language_readout_memory_design["mutates_runtime_state"] is False
    assert snn_language_readout_memory_design["runs_replay"] is False
    assert snn_language_readout_memory_design["writes_checkpoint"] is False
    assert snn_language_readout_memory_design["generates_text"] is True
    assert snn_language_readout_memory_design["decodes_text"] is True
    assert snn_language_readout_memory_design["trains_runtime_model"] is False
    assert snn_language_readout_memory_design["applies_plasticity"] is False
    readout_memory_body = snn_language_readout_memory_design[
        "snn_language_readout_memory_design"
    ]
    assert readout_memory_body["memory_scope"] == "working_trace"
    assert readout_memory_body["consolidation_route"] == "deferred_local_trace"
    assert readout_memory_body["rendered_text"] == "spike language"
    assert readout_memory_body["decoded_text_fragments"] == ["spike language"]
    assert len(readout_memory_body["local_learning_target_hashes"]) == 2
    assert all(
        len(value) == 64
        for value in readout_memory_body["local_learning_target_hashes"]
    )
    assert readout_memory_body["memory_recording_allowed"] is False
    assert readout_memory_body["replay_allowed"] is False
    assert readout_memory_body["plasticity_allowed"] is False
    assert readout_memory_body["training_allowed"] is False
    assert readout_memory_body["checkpoint_allowed"] is False
    assert readout_memory_body["fact_promotion_allowed"] is False
    assert readout_memory_body["action_allowed"] is False
    assert readout_memory_body["cognition_substrate_claimed"] is False
    assert snn_language_readout_memory_design["promotion_gate"][
        "eligible_for_snn_language_readout_memory_preflight"
    ] is True
    assert snn_language_readout_memory_design["promotion_gate"][
        "eligible_for_cognition_substrate"
    ] is False
    assert snn_language_readout_memory_design["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_readout_memory_design["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_readout_memory_preflight["accepted"] is False
    assert (
        blocked_snn_language_readout_memory_preflight["requires_operator_approval"]
        is False
    )
    assert blocked_snn_language_readout_memory_preflight["promotion_gate"][
        "required_evidence"
    ]["readout_memory_design_ready"] is False
    assert snn_language_readout_memory_preflight["surface"] == (
        "snn_language_readout_memory_preflight.v1"
    )
    assert snn_language_readout_memory_preflight["accepted"] is True
    assert snn_language_readout_memory_preflight["ready"] is True
    assert len(snn_language_readout_memory_preflight["preflight_hash"]) == 64
    assert snn_language_readout_memory_preflight["requires_operator_approval"] is False
    assert snn_language_readout_memory_preflight["advisory"] is True
    assert snn_language_readout_memory_preflight["executable"] is True
    assert snn_language_readout_memory_preflight["records_ledger_event"] is False
    assert snn_language_readout_memory_preflight["mutates_runtime_state"] is False
    assert snn_language_readout_memory_preflight["runs_replay"] is False
    assert snn_language_readout_memory_preflight["writes_checkpoint"] is False
    assert snn_language_readout_memory_preflight["generates_text"] is True
    assert snn_language_readout_memory_preflight["decodes_text"] is True
    assert snn_language_readout_memory_preflight["trains_runtime_model"] is False
    assert snn_language_readout_memory_preflight["applies_plasticity"] is False
    readout_memory_preflight_body = snn_language_readout_memory_preflight[
        "snn_language_readout_memory_preflight"
    ]
    assert readout_memory_preflight_body["memory_scope"] == "working_trace"
    assert (
        readout_memory_preflight_body["consolidation_route"]
        == "deferred_local_trace"
    )
    assert readout_memory_preflight_body["requested_device"] == "cuda:0"
    assert readout_memory_preflight_body["requires_cuda"] is True
    assert readout_memory_preflight_body["cuda_satisfied"] is True
    assert readout_memory_preflight_body["executor_ready"] is True
    assert readout_memory_preflight_body["execution_allowed"] is True
    assert readout_memory_preflight_body["rendered_text"] == "spike language"
    assert len(readout_memory_preflight_body["memory_trace_hash"]) == 64
    assert len(
        readout_memory_preflight_body["local_learning_target_hashes"]
    ) == 2
    assert readout_memory_preflight_body["memory_recording_allowed"] is False
    assert readout_memory_preflight_body["replay_allowed"] is False
    assert readout_memory_preflight_body["plasticity_allowed"] is False
    assert readout_memory_preflight_body["training_allowed"] is False
    assert readout_memory_preflight_body["checkpoint_allowed"] is False
    assert readout_memory_preflight_body["fact_promotion_allowed"] is False
    assert readout_memory_preflight_body["action_allowed"] is False
    assert readout_memory_preflight_body["cognition_substrate_claimed"] is False
    assert snn_language_readout_memory_preflight["promotion_gate"][
        "eligible_for_snn_language_readout_memory_executor"
    ] is True
    assert snn_language_readout_memory_preflight["promotion_gate"][
        "eligible_for_cognition_substrate"
    ] is False
    assert snn_language_readout_memory_preflight["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_readout_memory_preflight["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_readout_memory_executor["accepted"] is False
    assert (
        blocked_snn_language_readout_memory_executor["requires_operator_approval"]
        is False
    )
    assert blocked_snn_language_readout_memory_executor["promotion_gate"][
        "required_evidence"
    ]["preflight_ready"] is False
    _assert_record_family_source_window(
        blocked_snn_language_readout_memory_executor["source_window"],
        field="snn_language_readout_memory_events",
        expected_count=0,
    )
    assert snn_language_readout_memory_executor["surface"] == (
        "snn_language_readout_memory_executor.v1"
    )
    assert snn_language_readout_memory_executor["accepted"] is True
    assert snn_language_readout_memory_executor["ready"] is True
    assert len(
        snn_language_readout_memory_executor[
            "snn_language_readout_memory_event_hash"
        ]
    ) == 64
    assert len(snn_language_readout_memory_executor["memory_trace_hash"]) == 64
    assert snn_language_readout_memory_executor["requires_operator_approval"] is False
    assert snn_language_readout_memory_executor["advisory"] is False
    assert snn_language_readout_memory_executor["executable"] is True
    assert snn_language_readout_memory_executor["records_ledger_event"] is True
    assert snn_language_readout_memory_executor["mutates_runtime_state"] is True
    assert snn_language_readout_memory_executor["runs_replay"] is False
    assert snn_language_readout_memory_executor["writes_checkpoint"] is False
    assert snn_language_readout_memory_executor["generates_text"] is True
    assert snn_language_readout_memory_executor["decodes_text"] is True
    assert snn_language_readout_memory_executor["trains_runtime_model"] is False
    assert snn_language_readout_memory_executor["applies_plasticity"] is False
    assert snn_language_readout_memory_executor["resizes_network"] is False
    assert snn_language_readout_memory_executor["prunes_network"] is False
    readout_memory_event = snn_language_readout_memory_executor[
        "snn_language_readout_memory_event"
    ]
    assert readout_memory_event["memory_scope"] == "working_trace"
    assert readout_memory_event["consolidation_route"] == "deferred_local_trace"
    assert readout_memory_event["memory_recorded"] is True
    assert readout_memory_event["rendered_text"] == "spike language"
    assert len(readout_memory_event["local_learning_target_hashes"]) == 2
    assert readout_memory_event["runs_replay"] is False
    assert readout_memory_event["applies_plasticity"] is False
    assert readout_memory_event["trains_runtime_model"] is False
    assert readout_memory_event["writes_checkpoint"] is False
    assert readout_memory_event["resizes_network"] is False
    assert readout_memory_event["prunes_network"] is False
    assert readout_memory_event["promotes_fact"] is False
    assert readout_memory_event["executes_action"] is False
    assert readout_memory_event["cognition_substrate_claimed"] is False
    assert ledger_state["total_snn_language_readout_memory_count"] == 1
    assert len(ledger_state["snn_language_readout_memory_events"]) == 1
    _assert_record_family_source_window(
        snn_language_readout_memory_executor["source_window"],
        field="snn_language_readout_memory_events",
        expected_count=0,
    )
    assert snn_language_readout_memory_executor["promotion_gate"][
        "eligible_for_snn_language_readout_memory_event_review"
    ] is True
    assert snn_language_readout_memory_executor["promotion_gate"][
        "eligible_for_cognition_substrate"
    ] is False
    assert snn_language_readout_memory_executor["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_readout_memory_executor["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_readout_memory_event_review["accepted"] is False
    assert (
        blocked_snn_language_readout_memory_event_review[
            "requires_operator_approval"
        ]
        is False
    )
    assert blocked_snn_language_readout_memory_event_review["promotion_gate"][
        "required_evidence"
    ]["executor_accepted"] is False
    _assert_record_family_source_window(
        blocked_snn_language_readout_memory_event_review["source_window"],
        field="snn_language_readout_memory_events",
        expected_count=1,
    )
    assert snn_language_readout_memory_event_review["surface"] == (
        "snn_language_readout_memory_event_review.v1"
    )
    assert snn_language_readout_memory_event_review["accepted"] is True
    assert snn_language_readout_memory_event_review["ready"] is True
    assert len(snn_language_readout_memory_event_review["review_hash"]) == 64
    assert snn_language_readout_memory_event_review[
        "requires_operator_approval"
    ] is False
    assert snn_language_readout_memory_event_review["advisory"] is True
    assert snn_language_readout_memory_event_review["executable"] is False
    assert snn_language_readout_memory_event_review["records_ledger_event"] is False
    assert snn_language_readout_memory_event_review["mutates_runtime_state"] is False
    assert snn_language_readout_memory_event_review["runs_replay"] is False
    assert snn_language_readout_memory_event_review["writes_checkpoint"] is False
    assert snn_language_readout_memory_event_review["generates_text"] is True
    assert snn_language_readout_memory_event_review["decodes_text"] is True
    assert snn_language_readout_memory_event_review["trains_runtime_model"] is False
    assert snn_language_readout_memory_event_review["applies_plasticity"] is False
    assert snn_language_readout_memory_event_review["resizes_network"] is False
    assert snn_language_readout_memory_event_review["prunes_network"] is False
    readout_memory_review = snn_language_readout_memory_event_review[
        "snn_language_readout_memory_event_review"
    ]
    assert readout_memory_review["event_recorded_in_ledger"] is True
    assert readout_memory_review["memory_scope"] == "working_trace"
    assert (
        readout_memory_review["consolidation_route"] == "deferred_local_trace"
    )
    assert readout_memory_review["memory_recorded"] is True
    assert readout_memory_review["rendered_text"] == "spike language"
    assert len(readout_memory_review["memory_trace_hash"]) == 64
    assert len(readout_memory_review["local_learning_target_hashes"]) == 2
    assert readout_memory_review["replay_allowed"] is False
    assert readout_memory_review["plasticity_allowed"] is False
    assert readout_memory_review["training_allowed"] is False
    assert readout_memory_review["checkpoint_allowed"] is False
    assert readout_memory_review["resize_allowed"] is False
    assert readout_memory_review["prune_allowed"] is False
    assert readout_memory_review["fact_promotion_allowed"] is False
    assert readout_memory_review["action_allowed"] is False
    assert readout_memory_review["cognition_substrate_claimed"] is False
    _assert_record_family_source_window(
        snn_language_readout_memory_event_review["source_window"],
        field="snn_language_readout_memory_events",
        expected_count=1,
    )
    assert snn_language_readout_memory_event_review["promotion_gate"][
        "eligible_for_autonomous_snn_language_thought_consolidation_design"
    ] is True
    assert snn_language_readout_memory_event_review["promotion_gate"][
        "eligible_for_cognition_substrate"
    ] is False
    assert snn_language_readout_memory_event_review["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_readout_memory_event_review["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_thought_consolidation_design["accepted"] is False
    assert (
        blocked_snn_language_thought_consolidation_design[
            "requires_operator_approval"
        ]
        is False
    )
    assert blocked_snn_language_thought_consolidation_design["promotion_gate"][
        "required_evidence"
    ]["readout_memory_event_review_ready"] is False
    assert snn_language_thought_consolidation_design["surface"] == (
        "snn_language_autonomous_snn_language_thought_consolidation_design.v1"
    )
    assert snn_language_thought_consolidation_design["accepted"] is True
    assert snn_language_thought_consolidation_design["ready"] is True
    assert len(
        snn_language_thought_consolidation_design[
            "thought_consolidation_design_hash"
        ]
    ) == 64
    assert (
        snn_language_thought_consolidation_design["requires_operator_approval"]
        is False
    )
    assert snn_language_thought_consolidation_design["advisory"] is True
    assert snn_language_thought_consolidation_design["executable"] is False
    assert (
        snn_language_thought_consolidation_design["records_ledger_event"]
        is False
    )
    assert snn_language_thought_consolidation_design["mutates_runtime_state"] is False
    assert snn_language_thought_consolidation_design["runs_replay"] is False
    assert snn_language_thought_consolidation_design["writes_checkpoint"] is False
    assert snn_language_thought_consolidation_design["generates_text"] is True
    assert snn_language_thought_consolidation_design["decodes_text"] is True
    assert (
        snn_language_thought_consolidation_design["trains_runtime_model"]
        is False
    )
    assert snn_language_thought_consolidation_design["applies_plasticity"] is False
    assert snn_language_thought_consolidation_design["resizes_network"] is False
    assert snn_language_thought_consolidation_design["prunes_network"] is False
    thought_consolidation_body = snn_language_thought_consolidation_design[
        "autonomous_snn_language_thought_consolidation_design"
    ]
    assert (
        thought_consolidation_body["consolidation_scope"]
        == "local_trace_reinforcement"
    )
    assert (
        thought_consolidation_body["consolidation_route"]
        == "deferred_local_trace"
    )
    assert thought_consolidation_body["candidate_update_count"] == 2
    assert len(thought_consolidation_body["candidate_updates"]) == 2
    assert all(
        item["applied_to_runtime"] is False
        for item in thought_consolidation_body["candidate_updates"]
    )
    assert thought_consolidation_body["replay_allowed"] is False
    assert thought_consolidation_body["plasticity_allowed"] is False
    assert thought_consolidation_body["training_allowed"] is False
    assert thought_consolidation_body["checkpoint_allowed"] is False
    assert thought_consolidation_body["resize_allowed"] is False
    assert thought_consolidation_body["prune_allowed"] is False
    assert thought_consolidation_body["fact_promotion_allowed"] is False
    assert thought_consolidation_body["action_allowed"] is False
    assert thought_consolidation_body["cognition_substrate_claimed"] is False
    assert snn_language_thought_consolidation_design["promotion_gate"][
        "eligible_for_autonomous_snn_language_thought_consolidation_preflight"
    ] is True
    assert snn_language_thought_consolidation_design["promotion_gate"][
        "eligible_for_cognition_substrate"
    ] is False
    assert snn_language_thought_consolidation_design["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_thought_consolidation_design["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_thought_consolidation_preflight["accepted"] is False
    assert (
        blocked_snn_language_thought_consolidation_preflight[
            "requires_operator_approval"
        ]
        is False
    )
    assert blocked_snn_language_thought_consolidation_preflight["promotion_gate"][
        "required_evidence"
    ]["thought_consolidation_design_ready"] is False
    assert snn_language_thought_consolidation_preflight["surface"] == (
        "snn_language_autonomous_snn_language_thought_consolidation_preflight.v1"
    )
    assert snn_language_thought_consolidation_preflight["accepted"] is True
    assert snn_language_thought_consolidation_preflight["ready"] is True
    assert len(snn_language_thought_consolidation_preflight["preflight_hash"]) == 64
    assert (
        snn_language_thought_consolidation_preflight["requires_operator_approval"]
        is False
    )
    assert snn_language_thought_consolidation_preflight["advisory"] is True
    assert snn_language_thought_consolidation_preflight["executable"] is True
    assert (
        snn_language_thought_consolidation_preflight["records_ledger_event"]
        is False
    )
    assert snn_language_thought_consolidation_preflight["mutates_runtime_state"] is False
    assert snn_language_thought_consolidation_preflight["runs_replay"] is False
    assert snn_language_thought_consolidation_preflight["writes_checkpoint"] is False
    assert snn_language_thought_consolidation_preflight["generates_text"] is True
    assert snn_language_thought_consolidation_preflight["decodes_text"] is True
    assert (
        snn_language_thought_consolidation_preflight["trains_runtime_model"]
        is False
    )
    assert snn_language_thought_consolidation_preflight["applies_plasticity"] is False
    assert snn_language_thought_consolidation_preflight["resizes_network"] is False
    assert snn_language_thought_consolidation_preflight["prunes_network"] is False
    thought_consolidation_preflight_body = (
        snn_language_thought_consolidation_preflight[
            "autonomous_snn_language_thought_consolidation_preflight"
        ]
    )
    assert thought_consolidation_preflight_body["requested_device"] == "cuda:0"
    assert thought_consolidation_preflight_body["requires_cuda"] is True
    assert thought_consolidation_preflight_body["cuda_satisfied"] is True
    assert thought_consolidation_preflight_body["executor_ready"] is True
    assert thought_consolidation_preflight_body["execution_allowed"] is True
    assert thought_consolidation_preflight_body["candidate_update_count"] == 2
    assert len(thought_consolidation_preflight_body["candidate_updates"]) == 2
    assert all(
        item["applied_to_runtime"] is False
        for item in thought_consolidation_preflight_body["candidate_updates"]
    )
    assert thought_consolidation_preflight_body["replay_allowed"] is False
    assert thought_consolidation_preflight_body["plasticity_allowed"] is False
    assert thought_consolidation_preflight_body["training_allowed"] is False
    assert thought_consolidation_preflight_body["checkpoint_allowed"] is False
    assert thought_consolidation_preflight_body["resize_allowed"] is False
    assert thought_consolidation_preflight_body["prune_allowed"] is False
    assert thought_consolidation_preflight_body["fact_promotion_allowed"] is False
    assert thought_consolidation_preflight_body["action_allowed"] is False
    assert (
        thought_consolidation_preflight_body["cognition_substrate_claimed"]
        is False
    )
    assert snn_language_thought_consolidation_preflight["promotion_gate"][
        "eligible_for_autonomous_snn_language_thought_consolidation_executor"
    ] is True
    assert snn_language_thought_consolidation_preflight["promotion_gate"][
        "eligible_for_cognition_substrate"
    ] is False
    assert snn_language_thought_consolidation_preflight["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_thought_consolidation_preflight["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_thought_consolidation_executor["accepted"] is False
    assert (
        blocked_snn_language_thought_consolidation_executor[
            "requires_operator_approval"
        ]
        is False
    )
    assert blocked_snn_language_thought_consolidation_executor["promotion_gate"][
        "required_evidence"
    ]["preflight_ready"] is False
    _assert_record_family_source_window(
        blocked_snn_language_thought_consolidation_executor["source_window"],
        field="autonomous_snn_language_thought_consolidation_events",
        expected_count=0,
    )
    assert snn_language_thought_consolidation_executor["surface"] == (
        "snn_language_autonomous_snn_language_thought_consolidation_executor.v1"
    )
    assert snn_language_thought_consolidation_executor["accepted"] is True
    assert snn_language_thought_consolidation_executor["ready"] is True
    assert len(
        snn_language_thought_consolidation_executor[
            "autonomous_snn_language_thought_consolidation_event_hash"
        ]
    ) == 64
    assert (
        snn_language_thought_consolidation_executor["requires_operator_approval"]
        is False
    )
    assert snn_language_thought_consolidation_executor["advisory"] is False
    assert snn_language_thought_consolidation_executor["executable"] is True
    assert (
        snn_language_thought_consolidation_executor["records_ledger_event"]
        is True
    )
    assert snn_language_thought_consolidation_executor["mutates_runtime_state"] is True
    assert snn_language_thought_consolidation_executor["runs_replay"] is False
    assert snn_language_thought_consolidation_executor["writes_checkpoint"] is False
    assert snn_language_thought_consolidation_executor["generates_text"] is True
    assert snn_language_thought_consolidation_executor["decodes_text"] is True
    assert (
        snn_language_thought_consolidation_executor["trains_runtime_model"]
        is False
    )
    assert snn_language_thought_consolidation_executor["applies_plasticity"] is True
    assert snn_language_thought_consolidation_executor["resizes_network"] is False
    assert snn_language_thought_consolidation_executor["prunes_network"] is False
    thought_consolidation_event = snn_language_thought_consolidation_executor[
        "autonomous_snn_language_thought_consolidation_event"
    ]
    assert thought_consolidation_event["applies_plasticity"] is True
    assert thought_consolidation_event["candidate_update_count"] == 2
    assert len(thought_consolidation_event["candidate_updates"]) == 2
    assert all(
        item["applied_to_runtime"] is True
        for item in thought_consolidation_event["candidate_updates"]
    )
    assert thought_consolidation_event["runs_replay"] is False
    assert thought_consolidation_event["writes_checkpoint"] is False
    assert thought_consolidation_event["trains_runtime_model"] is False
    assert thought_consolidation_event["resizes_network"] is False
    assert thought_consolidation_event["prunes_network"] is False
    assert thought_consolidation_event["promotes_fact"] is False
    assert thought_consolidation_event["executes_action"] is False
    assert thought_consolidation_event["cognition_substrate_claimed"] is False
    assert (
        ledger_state["total_autonomous_snn_language_thought_consolidation_count"]
        == 1
    )
    assert (
        len(ledger_state["autonomous_snn_language_thought_consolidation_events"])
        == 1
    )
    _assert_record_family_source_window(
        snn_language_thought_consolidation_executor["source_window"],
        field="autonomous_snn_language_thought_consolidation_events",
        expected_count=0,
    )
    assert snn_language_thought_consolidation_executor["promotion_gate"][
        "eligible_for_autonomous_snn_language_thought_consolidation_event_review"
    ] is True
    assert snn_language_thought_consolidation_executor["promotion_gate"][
        "eligible_for_cognition_substrate"
    ] is False
    assert snn_language_thought_consolidation_executor["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_thought_consolidation_executor["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_thought_consolidation_event_review["accepted"] is False
    assert (
        blocked_snn_language_thought_consolidation_event_review[
            "requires_operator_approval"
        ]
        is False
    )
    assert blocked_snn_language_thought_consolidation_event_review["promotion_gate"][
        "required_evidence"
    ]["executor_accepted"] is False
    _assert_record_family_source_window(
        blocked_snn_language_thought_consolidation_event_review["source_window"],
        field="autonomous_snn_language_thought_consolidation_events",
        expected_count=1,
    )
    assert snn_language_thought_consolidation_event_review["surface"] == (
        "snn_language_autonomous_snn_language_thought_consolidation_event_review.v1"
    )
    assert snn_language_thought_consolidation_event_review["accepted"] is True
    assert snn_language_thought_consolidation_event_review["ready"] is True
    assert len(snn_language_thought_consolidation_event_review["review_hash"]) == 64
    assert (
        snn_language_thought_consolidation_event_review[
            "requires_operator_approval"
        ]
        is False
    )
    assert snn_language_thought_consolidation_event_review["advisory"] is True
    assert snn_language_thought_consolidation_event_review["executable"] is False
    assert (
        snn_language_thought_consolidation_event_review["records_ledger_event"]
        is False
    )
    assert (
        snn_language_thought_consolidation_event_review["mutates_runtime_state"]
        is False
    )
    assert snn_language_thought_consolidation_event_review["runs_replay"] is False
    assert snn_language_thought_consolidation_event_review["writes_checkpoint"] is False
    assert snn_language_thought_consolidation_event_review["generates_text"] is True
    assert snn_language_thought_consolidation_event_review["decodes_text"] is True
    assert (
        snn_language_thought_consolidation_event_review["trains_runtime_model"]
        is False
    )
    assert snn_language_thought_consolidation_event_review["applies_plasticity"] is True
    assert snn_language_thought_consolidation_event_review["resizes_network"] is False
    assert snn_language_thought_consolidation_event_review["prunes_network"] is False
    thought_consolidation_event_review = (
        snn_language_thought_consolidation_event_review[
            "autonomous_snn_language_thought_consolidation_event_review"
        ]
    )
    assert thought_consolidation_event_review["event_recorded_in_ledger"] is True
    assert (
        thought_consolidation_event_review[
            "autonomous_snn_language_thought_consolidation_event_hash"
        ]
        == thought_consolidation_event[
            "autonomous_snn_language_thought_consolidation_event_hash"
        ]
    )
    assert (
        thought_consolidation_event_review["consolidation_scope"]
        == "local_trace_reinforcement"
    )
    assert (
        thought_consolidation_event_review["consolidation_route"]
        == "deferred_local_trace"
    )
    assert thought_consolidation_event_review["requested_device"] == "cuda:0"
    assert thought_consolidation_event_review["candidate_update_count"] == 2
    assert len(thought_consolidation_event_review["candidate_updates"]) == 2
    assert all(
        item["applied_to_runtime"] is True
        and item["applied_in_ledger"] is True
        for item in thought_consolidation_event_review["candidate_updates"]
    )
    assert thought_consolidation_event_review["plasticity_applied"] is True
    assert thought_consolidation_event_review["runtime_state_mutated"] is True
    assert thought_consolidation_event_review["local_only"] is True
    assert thought_consolidation_event_review["normalization"] is True
    assert thought_consolidation_event_review["replay_allowed"] is False
    assert thought_consolidation_event_review["training_allowed"] is False
    assert thought_consolidation_event_review["checkpoint_allowed"] is False
    assert thought_consolidation_event_review["resize_allowed"] is False
    assert thought_consolidation_event_review["prune_allowed"] is False
    assert thought_consolidation_event_review["fact_promotion_allowed"] is False
    assert thought_consolidation_event_review["action_allowed"] is False
    assert (
        thought_consolidation_event_review["cognition_substrate_claimed"]
        is False
    )
    _assert_record_family_source_window(
        snn_language_thought_consolidation_event_review["source_window"],
        field="autonomous_snn_language_thought_consolidation_events",
        expected_count=1,
    )
    assert snn_language_thought_consolidation_event_review["promotion_gate"][
        "eligible_for_autonomous_snn_language_thought_structural_plasticity_design"
    ] is True
    assert snn_language_thought_consolidation_event_review["promotion_gate"][
        "eligible_for_cognition_substrate"
    ] is False
    assert snn_language_thought_consolidation_event_review["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_thought_consolidation_event_review["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_thought_structural_plasticity_design[
        "accepted"
    ] is False
    assert (
        blocked_snn_language_thought_structural_plasticity_design[
            "requires_operator_approval"
        ]
        is False
    )
    assert blocked_snn_language_thought_structural_plasticity_design[
        "promotion_gate"
    ]["required_evidence"]["consolidation_event_review_ready"] is False
    assert snn_language_thought_structural_plasticity_design["surface"] == (
        "snn_language_autonomous_snn_language_thought_structural_plasticity_design.v1"
    )
    assert snn_language_thought_structural_plasticity_design["accepted"] is True
    assert snn_language_thought_structural_plasticity_design["ready"] is True
    assert len(
        snn_language_thought_structural_plasticity_design[
            "thought_structural_plasticity_design_hash"
        ]
    ) == 64
    assert (
        snn_language_thought_structural_plasticity_design[
            "requires_operator_approval"
        ]
        is False
    )
    assert snn_language_thought_structural_plasticity_design["advisory"] is True
    assert snn_language_thought_structural_plasticity_design["executable"] is False
    assert (
        snn_language_thought_structural_plasticity_design["records_ledger_event"]
        is False
    )
    assert (
        snn_language_thought_structural_plasticity_design["mutates_runtime_state"]
        is False
    )
    assert snn_language_thought_structural_plasticity_design["runs_replay"] is False
    assert (
        snn_language_thought_structural_plasticity_design["writes_checkpoint"]
        is False
    )
    assert snn_language_thought_structural_plasticity_design["generates_text"] is True
    assert snn_language_thought_structural_plasticity_design["decodes_text"] is True
    assert (
        snn_language_thought_structural_plasticity_design["trains_runtime_model"]
        is False
    )
    assert (
        snn_language_thought_structural_plasticity_design["applies_plasticity"]
        is False
    )
    assert snn_language_thought_structural_plasticity_design["resizes_network"] is False
    assert snn_language_thought_structural_plasticity_design["prunes_network"] is False
    assert snn_language_thought_structural_plasticity_design["adds_neurons"] is False
    assert snn_language_thought_structural_plasticity_design["adds_synapses"] is False
    thought_structural_design = snn_language_thought_structural_plasticity_design[
        "autonomous_snn_language_thought_structural_plasticity_design"
    ]
    assert thought_structural_design["structural_scope"] == (
        "thought_trace_sparse_capacity"
    )
    assert thought_structural_design["structural_route"] == (
        "reviewed_consolidation_to_growth_prune"
    )
    assert thought_structural_design["growth_candidate_count"] == 2
    assert len(thought_structural_design["growth_candidates"]) == 2
    assert thought_structural_design["prune_candidate_count"] == 1
    assert len(thought_structural_design["prune_candidates"]) == 1
    assert all(
        item["applied_to_runtime"] is False
        for item in thought_structural_design["growth_candidates"]
    )
    assert all(
        item["applied_to_runtime"] is False
        for item in thought_structural_design["prune_candidates"]
    )
    assert thought_structural_design["structural_growth_designed"] is True
    assert thought_structural_design["structural_prune_designed"] is True
    assert thought_structural_design["growth_allowed"] is False
    assert thought_structural_design["prune_allowed"] is False
    assert thought_structural_design["replay_allowed"] is False
    assert thought_structural_design["plasticity_allowed"] is False
    assert thought_structural_design["training_allowed"] is False
    assert thought_structural_design["checkpoint_allowed"] is False
    assert thought_structural_design["resize_allowed"] is False
    assert thought_structural_design["fact_promotion_allowed"] is False
    assert thought_structural_design["action_allowed"] is False
    assert thought_structural_design["cognition_substrate_claimed"] is False
    assert snn_language_thought_structural_plasticity_design["promotion_gate"][
        "eligible_for_autonomous_snn_language_thought_structural_plasticity_preflight"
    ] is True
    assert snn_language_thought_structural_plasticity_design["promotion_gate"][
        "eligible_for_cognition_substrate"
    ] is False
    assert snn_language_thought_structural_plasticity_design["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_thought_structural_plasticity_design["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_thought_structural_plasticity_preflight[
        "accepted"
    ] is False
    assert (
        blocked_snn_language_thought_structural_plasticity_preflight[
            "requires_operator_approval"
        ]
        is False
    )
    assert blocked_snn_language_thought_structural_plasticity_preflight[
        "promotion_gate"
    ]["required_evidence"]["structural_plasticity_design_ready"] is False
    assert snn_language_thought_structural_plasticity_preflight["surface"] == (
        "snn_language_autonomous_snn_language_thought_structural_plasticity_preflight.v1"
    )
    assert snn_language_thought_structural_plasticity_preflight["accepted"] is True
    assert snn_language_thought_structural_plasticity_preflight["ready"] is True
    assert len(snn_language_thought_structural_plasticity_preflight["preflight_hash"]) == 64
    assert (
        snn_language_thought_structural_plasticity_preflight[
            "requires_operator_approval"
        ]
        is False
    )
    assert snn_language_thought_structural_plasticity_preflight["advisory"] is True
    assert snn_language_thought_structural_plasticity_preflight["executable"] is True
    assert (
        snn_language_thought_structural_plasticity_preflight[
            "records_ledger_event"
        ]
        is False
    )
    assert (
        snn_language_thought_structural_plasticity_preflight[
            "mutates_runtime_state"
        ]
        is False
    )
    assert snn_language_thought_structural_plasticity_preflight["runs_replay"] is False
    assert (
        snn_language_thought_structural_plasticity_preflight["writes_checkpoint"]
        is False
    )
    assert snn_language_thought_structural_plasticity_preflight["generates_text"] is True
    assert snn_language_thought_structural_plasticity_preflight["decodes_text"] is True
    assert (
        snn_language_thought_structural_plasticity_preflight["trains_runtime_model"]
        is False
    )
    assert (
        snn_language_thought_structural_plasticity_preflight["applies_plasticity"]
        is False
    )
    assert snn_language_thought_structural_plasticity_preflight["resizes_network"] is False
    assert snn_language_thought_structural_plasticity_preflight["prunes_network"] is False
    assert snn_language_thought_structural_plasticity_preflight["adds_neurons"] is False
    assert snn_language_thought_structural_plasticity_preflight["adds_synapses"] is False
    thought_structural_preflight = snn_language_thought_structural_plasticity_preflight[
        "autonomous_snn_language_thought_structural_plasticity_preflight"
    ]
    assert thought_structural_preflight["requested_device"] == "cuda:0"
    assert thought_structural_preflight["requires_cuda"] is True
    assert thought_structural_preflight["cuda_satisfied"] is True
    assert thought_structural_preflight["executor_ready"] is True
    assert thought_structural_preflight["execution_allowed"] is False
    assert thought_structural_preflight["growth_candidate_count"] == 2
    assert len(thought_structural_preflight["growth_candidates"]) == 2
    assert thought_structural_preflight["prune_candidate_count"] == 1
    assert len(thought_structural_preflight["prune_candidates"]) == 1
    assert thought_structural_preflight["proposed_new_neuron_count"] == 2
    assert thought_structural_preflight["proposed_new_synapse_count"] == 2
    assert thought_structural_preflight["proposed_prune_synapse_count"] == 1
    assert thought_structural_preflight["growth_allowed"] is False
    assert thought_structural_preflight["prune_allowed"] is False
    assert thought_structural_preflight["replay_allowed"] is False
    assert thought_structural_preflight["plasticity_allowed"] is False
    assert thought_structural_preflight["training_allowed"] is False
    assert thought_structural_preflight["checkpoint_allowed"] is False
    assert thought_structural_preflight["resize_allowed"] is False
    assert thought_structural_preflight["fact_promotion_allowed"] is False
    assert thought_structural_preflight["action_allowed"] is False
    assert thought_structural_preflight["cognition_substrate_claimed"] is False
    assert snn_language_thought_structural_plasticity_preflight["promotion_gate"][
        "eligible_for_autonomous_snn_language_thought_structural_plasticity_executor"
    ] is True
    assert snn_language_thought_structural_plasticity_preflight["promotion_gate"][
        "eligible_for_cognition_substrate"
    ] is False
    assert snn_language_thought_structural_plasticity_preflight["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_thought_structural_plasticity_preflight["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_thought_structural_plasticity_executor[
        "accepted"
    ] is False
    assert (
        blocked_snn_language_thought_structural_plasticity_executor[
            "requires_operator_approval"
        ]
        is False
    )
    assert blocked_snn_language_thought_structural_plasticity_executor[
        "promotion_gate"
    ]["required_evidence"]["preflight_ready"] is False
    _assert_record_family_source_window(
        blocked_snn_language_thought_structural_plasticity_executor[
            "source_window"
        ],
        field="autonomous_snn_language_thought_structural_plasticity_events",
        expected_count=0,
    )
    assert snn_language_thought_structural_plasticity_executor["surface"] == (
        "snn_language_autonomous_snn_language_thought_structural_plasticity_executor.v1"
    )
    assert snn_language_thought_structural_plasticity_executor["accepted"] is True
    assert snn_language_thought_structural_plasticity_executor["ready"] is True
    assert len(
        snn_language_thought_structural_plasticity_executor[
            "autonomous_snn_language_thought_structural_plasticity_event_hash"
        ]
    ) == 64
    assert (
        snn_language_thought_structural_plasticity_executor[
            "requires_operator_approval"
        ]
        is False
    )
    assert snn_language_thought_structural_plasticity_executor["advisory"] is False
    assert snn_language_thought_structural_plasticity_executor["executable"] is True
    assert (
        snn_language_thought_structural_plasticity_executor["records_ledger_event"]
        is True
    )
    assert (
        snn_language_thought_structural_plasticity_executor["mutates_runtime_state"]
        is True
    )
    assert snn_language_thought_structural_plasticity_executor["runs_replay"] is False
    assert (
        snn_language_thought_structural_plasticity_executor["writes_checkpoint"]
        is False
    )
    assert snn_language_thought_structural_plasticity_executor["generates_text"] is True
    assert snn_language_thought_structural_plasticity_executor["decodes_text"] is True
    assert (
        snn_language_thought_structural_plasticity_executor["trains_runtime_model"]
        is False
    )
    assert (
        snn_language_thought_structural_plasticity_executor["applies_plasticity"]
        is False
    )
    assert snn_language_thought_structural_plasticity_executor[
        "structural_plasticity_applied"
    ] is True
    assert snn_language_thought_structural_plasticity_executor["resizes_network"] is False
    assert snn_language_thought_structural_plasticity_executor["adds_neurons"] is True
    assert snn_language_thought_structural_plasticity_executor["adds_synapses"] is True
    assert snn_language_thought_structural_plasticity_executor["prunes_network"] is True
    thought_structural_event = snn_language_thought_structural_plasticity_executor[
        "autonomous_snn_language_thought_structural_plasticity_event"
    ]
    assert thought_structural_event["structural_plasticity_applied"] is True
    assert thought_structural_event["growth_candidate_count"] == 2
    assert len(thought_structural_event["growth_candidates"]) == 2
    assert thought_structural_event["prune_candidate_count"] == 1
    assert len(thought_structural_event["prune_candidates"]) == 1
    assert all(
        item["applied_to_runtime"] is True
        and item["applied_in_ledger"] is True
        for item in thought_structural_event["growth_candidates"]
    )
    assert all(
        item["applied_to_runtime"] is True
        and item["applied_in_ledger"] is True
        for item in thought_structural_event["prune_candidates"]
    )
    assert thought_structural_event["proposed_new_neuron_count"] == 2
    assert thought_structural_event["proposed_new_synapse_count"] == 2
    assert thought_structural_event["proposed_prune_synapse_count"] == 1
    assert thought_structural_event["runs_replay"] is False
    assert thought_structural_event["writes_checkpoint"] is False
    assert thought_structural_event["trains_runtime_model"] is False
    assert thought_structural_event["resizes_network"] is False
    assert thought_structural_event["adds_neurons"] is True
    assert thought_structural_event["adds_synapses"] is True
    assert thought_structural_event["prunes_network"] is True
    assert thought_structural_event["promotes_fact"] is False
    assert thought_structural_event["executes_action"] is False
    assert thought_structural_event["cognition_substrate_claimed"] is False
    assert (
        ledger_state[
            "total_autonomous_snn_language_thought_structural_plasticity_count"
        ]
        == 1
    )
    assert (
        len(
            ledger_state[
                "autonomous_snn_language_thought_structural_plasticity_events"
            ]
        )
        == 1
    )
    _assert_record_family_source_window(
        snn_language_thought_structural_plasticity_executor["source_window"],
        field="autonomous_snn_language_thought_structural_plasticity_events",
        expected_count=0,
    )
    assert snn_language_thought_structural_plasticity_executor["promotion_gate"][
        "eligible_for_autonomous_snn_language_thought_structural_plasticity_event_review"
    ] is True
    assert snn_language_thought_structural_plasticity_executor["promotion_gate"][
        "eligible_for_cognition_substrate"
    ] is False
    assert snn_language_thought_structural_plasticity_executor["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_thought_structural_plasticity_executor["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_thought_structural_plasticity_event_review[
        "accepted"
    ] is False
    assert (
        blocked_snn_language_thought_structural_plasticity_event_review[
            "requires_operator_approval"
        ]
        is False
    )
    assert blocked_snn_language_thought_structural_plasticity_event_review[
        "promotion_gate"
    ]["required_evidence"]["executor_accepted"] is False
    _assert_record_family_source_window(
        blocked_snn_language_thought_structural_plasticity_event_review[
            "source_window"
        ],
        field="autonomous_snn_language_thought_structural_plasticity_events",
        expected_count=1,
    )
    assert snn_language_thought_structural_plasticity_event_review["surface"] == (
        "snn_language_autonomous_snn_language_thought_structural_plasticity_event_review.v1"
    )
    assert snn_language_thought_structural_plasticity_event_review["accepted"] is True
    assert snn_language_thought_structural_plasticity_event_review["ready"] is True
    assert len(
        snn_language_thought_structural_plasticity_event_review["review_hash"]
    ) == 64
    assert (
        snn_language_thought_structural_plasticity_event_review[
            "requires_operator_approval"
        ]
        is False
    )
    assert snn_language_thought_structural_plasticity_event_review["advisory"] is True
    assert snn_language_thought_structural_plasticity_event_review["executable"] is False
    assert (
        snn_language_thought_structural_plasticity_event_review[
            "records_ledger_event"
        ]
        is False
    )
    assert (
        snn_language_thought_structural_plasticity_event_review[
            "mutates_runtime_state"
        ]
        is False
    )
    assert snn_language_thought_structural_plasticity_event_review["runs_replay"] is False
    assert (
        snn_language_thought_structural_plasticity_event_review[
            "writes_checkpoint"
        ]
        is False
    )
    assert snn_language_thought_structural_plasticity_event_review["generates_text"] is True
    assert snn_language_thought_structural_plasticity_event_review["decodes_text"] is True
    assert (
        snn_language_thought_structural_plasticity_event_review[
            "trains_runtime_model"
        ]
        is False
    )
    assert (
        snn_language_thought_structural_plasticity_event_review[
            "applies_plasticity"
        ]
        is False
    )
    assert snn_language_thought_structural_plasticity_event_review[
        "structural_plasticity_applied"
    ] is True
    assert (
        snn_language_thought_structural_plasticity_event_review["resizes_network"]
        is False
    )
    assert snn_language_thought_structural_plasticity_event_review["adds_neurons"] is True
    assert snn_language_thought_structural_plasticity_event_review["adds_synapses"] is True
    assert snn_language_thought_structural_plasticity_event_review["prunes_network"] is True
    thought_structural_event_review = (
        snn_language_thought_structural_plasticity_event_review[
            "autonomous_snn_language_thought_structural_plasticity_event_review"
        ]
    )
    assert thought_structural_event_review["event_recorded_in_ledger"] is True
    assert (
        thought_structural_event_review[
            "autonomous_snn_language_thought_structural_plasticity_event_hash"
        ]
        == thought_structural_event[
            "autonomous_snn_language_thought_structural_plasticity_event_hash"
        ]
    )
    assert thought_structural_event_review["structural_scope"] == (
        "thought_trace_sparse_capacity"
    )
    assert thought_structural_event_review["structural_route"] == (
        "reviewed_consolidation_to_growth_prune"
    )
    assert thought_structural_event_review["requested_device"] == "cuda:0"
    assert thought_structural_event_review["growth_candidate_count"] == 2
    assert len(thought_structural_event_review["growth_candidates"]) == 2
    assert thought_structural_event_review["prune_candidate_count"] == 1
    assert len(thought_structural_event_review["prune_candidates"]) == 1
    assert all(
        item["applied_to_runtime"] is True
        and item["applied_in_ledger"] is True
        for item in thought_structural_event_review["growth_candidates"]
    )
    assert all(
        item["applied_to_runtime"] is True
        and item["applied_in_ledger"] is True
        for item in thought_structural_event_review["prune_candidates"]
    )
    assert thought_structural_event_review["proposed_new_neuron_count"] == 2
    assert thought_structural_event_review["proposed_new_synapse_count"] == 2
    assert thought_structural_event_review["proposed_prune_synapse_count"] == 1
    assert thought_structural_event_review["structural_plasticity_applied"] is True
    assert thought_structural_event_review["runtime_state_mutated"] is True
    assert thought_structural_event_review["checkpoint_allowed"] is False
    assert thought_structural_event_review["replay_allowed"] is False
    assert thought_structural_event_review["plasticity_allowed"] is False
    assert thought_structural_event_review["training_allowed"] is False
    assert thought_structural_event_review["resize_allowed"] is False
    assert thought_structural_event_review["fact_promotion_allowed"] is False
    assert thought_structural_event_review["action_allowed"] is False
    assert thought_structural_event_review["cognition_substrate_claimed"] is False
    _assert_record_family_source_window(
        snn_language_thought_structural_plasticity_event_review[
            "source_window"
        ],
        field="autonomous_snn_language_thought_structural_plasticity_events",
        expected_count=1,
    )
    assert snn_language_thought_structural_plasticity_event_review["promotion_gate"][
        "eligible_for_autonomous_snn_language_thought_capacity_mutation_design"
    ] is True
    assert snn_language_thought_structural_plasticity_event_review["promotion_gate"][
        "eligible_for_cognition_substrate"
    ] is False
    assert snn_language_thought_structural_plasticity_event_review["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_thought_structural_plasticity_event_review["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_thought_capacity_mutation_design[
        "accepted"
    ] is False
    assert (
        blocked_snn_language_thought_capacity_mutation_design[
            "requires_operator_approval"
        ]
        is False
    )
    assert blocked_snn_language_thought_capacity_mutation_design["promotion_gate"][
        "required_evidence"
    ]["structural_event_review_ready"] is False
    assert snn_language_thought_capacity_mutation_design["surface"] == (
        "snn_language_autonomous_snn_language_thought_capacity_mutation_design.v1"
    )
    assert snn_language_thought_capacity_mutation_design["accepted"] is True
    assert snn_language_thought_capacity_mutation_design["ready"] is True
    assert len(
        snn_language_thought_capacity_mutation_design[
            "thought_capacity_mutation_design_hash"
        ]
    ) == 64
    assert (
        snn_language_thought_capacity_mutation_design["requires_operator_approval"]
        is False
    )
    assert snn_language_thought_capacity_mutation_design["advisory"] is True
    assert snn_language_thought_capacity_mutation_design["executable"] is False
    assert (
        snn_language_thought_capacity_mutation_design["records_ledger_event"]
        is False
    )
    assert (
        snn_language_thought_capacity_mutation_design["mutates_runtime_state"]
        is False
    )
    assert snn_language_thought_capacity_mutation_design["runs_replay"] is False
    assert snn_language_thought_capacity_mutation_design["writes_checkpoint"] is False
    assert snn_language_thought_capacity_mutation_design["generates_text"] is True
    assert snn_language_thought_capacity_mutation_design["decodes_text"] is True
    assert (
        snn_language_thought_capacity_mutation_design["trains_runtime_model"]
        is False
    )
    assert snn_language_thought_capacity_mutation_design["applies_plasticity"] is False
    assert snn_language_thought_capacity_mutation_design["resizes_network"] is False
    assert snn_language_thought_capacity_mutation_design["adds_neurons"] is False
    assert snn_language_thought_capacity_mutation_design["adds_synapses"] is False
    assert snn_language_thought_capacity_mutation_design["prunes_network"] is False
    thought_capacity_design = snn_language_thought_capacity_mutation_design[
        "autonomous_snn_language_thought_capacity_mutation_design"
    ]
    assert thought_capacity_design["mutation_scope"] == "thought_driven_sparse_capacity"
    assert thought_capacity_design["mutation_route"] == (
        "reviewed_structural_plasticity_to_capacity_resize"
    )
    assert thought_capacity_design["current_neuron_capacity"] == 64
    assert thought_capacity_design["target_neuron_capacity"] == 66
    assert thought_capacity_design["current_sparse_synapse_budget"] == 256
    assert thought_capacity_design["target_sparse_synapse_budget"] == 258
    assert thought_capacity_design["current_dense_shape"] == [64, 64]
    assert thought_capacity_design["target_dense_shape"] == [66, 66]
    assert thought_capacity_design["preserved_dense_shape"] == [64, 64]
    assert thought_capacity_design["zero_initialized_new_rows"] == 2
    assert thought_capacity_design["zero_initialized_new_cols"] == 2
    assert thought_capacity_design["proposed_new_neuron_count"] == 2
    assert thought_capacity_design["proposed_new_synapse_count"] == 2
    assert thought_capacity_design["proposed_prune_synapse_count"] == 1
    assert thought_capacity_design["requires_cuda_relayout"] is True
    assert thought_capacity_design["requires_checkpoint"] is True
    assert thought_capacity_design["requires_restore_validation"] is True
    assert thought_capacity_design["capacity_mutation_designed"] is True
    assert thought_capacity_design["resize_allowed"] is False
    assert thought_capacity_design["growth_allowed"] is False
    assert thought_capacity_design["prune_allowed"] is False
    assert thought_capacity_design["replay_allowed"] is False
    assert thought_capacity_design["plasticity_allowed"] is False
    assert thought_capacity_design["training_allowed"] is False
    assert thought_capacity_design["checkpoint_allowed"] is False
    assert thought_capacity_design["fact_promotion_allowed"] is False
    assert thought_capacity_design["action_allowed"] is False
    assert thought_capacity_design["cognition_substrate_claimed"] is False
    assert snn_language_thought_capacity_mutation_design["promotion_gate"][
        "eligible_for_autonomous_snn_language_thought_capacity_mutation_preflight"
    ] is True
    assert snn_language_thought_capacity_mutation_design["promotion_gate"][
        "eligible_for_cognition_substrate"
    ] is False
    assert snn_language_thought_capacity_mutation_design["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_thought_capacity_mutation_design["promotion_gate"][
        "eligible_for_action"
    ] is False
    assert blocked_snn_language_thought_capacity_mutation_preflight[
        "accepted"
    ] is False
    assert (
        blocked_snn_language_thought_capacity_mutation_preflight[
            "requires_operator_approval"
        ]
        is False
    )
    assert blocked_snn_language_thought_capacity_mutation_preflight[
        "promotion_gate"
    ]["required_evidence"]["capacity_mutation_design_ready"] is False
    assert snn_language_thought_capacity_mutation_preflight["surface"] == (
        "snn_language_autonomous_snn_language_thought_capacity_mutation_preflight.v1"
    )
    assert snn_language_thought_capacity_mutation_preflight["accepted"] is True
    assert snn_language_thought_capacity_mutation_preflight["ready"] is True
    assert len(snn_language_thought_capacity_mutation_preflight["preflight_hash"]) == 64
    assert (
        snn_language_thought_capacity_mutation_preflight["requires_operator_approval"]
        is False
    )
    assert snn_language_thought_capacity_mutation_preflight["advisory"] is True
    assert snn_language_thought_capacity_mutation_preflight["executable"] is True
    assert (
        snn_language_thought_capacity_mutation_preflight["records_ledger_event"]
        is False
    )
    assert (
        snn_language_thought_capacity_mutation_preflight["mutates_runtime_state"]
        is False
    )
    assert snn_language_thought_capacity_mutation_preflight["runs_replay"] is False
    assert (
        snn_language_thought_capacity_mutation_preflight["writes_checkpoint"]
        is False
    )
    assert snn_language_thought_capacity_mutation_preflight["generates_text"] is True
    assert snn_language_thought_capacity_mutation_preflight["decodes_text"] is True
    assert (
        snn_language_thought_capacity_mutation_preflight["trains_runtime_model"]
        is False
    )
    assert (
        snn_language_thought_capacity_mutation_preflight["applies_plasticity"]
        is False
    )
    assert snn_language_thought_capacity_mutation_preflight["resizes_network"] is False
    assert snn_language_thought_capacity_mutation_preflight["adds_neurons"] is False
    assert snn_language_thought_capacity_mutation_preflight["adds_synapses"] is False
    assert snn_language_thought_capacity_mutation_preflight["prunes_network"] is False
    thought_capacity_preflight = snn_language_thought_capacity_mutation_preflight[
        "autonomous_snn_language_thought_capacity_mutation_preflight"
    ]
    assert thought_capacity_preflight["requested_device"] == "cuda:0"
    assert thought_capacity_preflight["cuda_available"] is True
    assert thought_capacity_preflight["cuda_relayout_verified"] is True
    assert thought_capacity_preflight["executor_ready"] is True
    assert thought_capacity_preflight["checkpoint_path"] == (
        "memory://thought-capacity-before"
    )
    assert thought_capacity_preflight["snapshot_id"] == "thought-capacity-snapshot"
    assert thought_capacity_preflight["checkpoint_saved"] is True
    assert thought_capacity_preflight["restore_verified"] is True
    assert thought_capacity_preflight["current_neuron_capacity"] == 64
    assert thought_capacity_preflight["target_neuron_capacity"] == 66
    assert thought_capacity_preflight["current_sparse_synapse_budget"] == 256
    assert thought_capacity_preflight["target_sparse_synapse_budget"] == 258
    assert thought_capacity_preflight["current_dense_shape"] == [64, 64]
    assert thought_capacity_preflight["target_dense_shape"] == [66, 66]
    assert thought_capacity_preflight["preserved_dense_shape"] == [64, 64]
    assert thought_capacity_preflight["zero_initialized_new_rows"] == 2
    assert thought_capacity_preflight["zero_initialized_new_cols"] == 2
    assert thought_capacity_preflight["growth_candidate_count"] == 2
    assert len(thought_capacity_preflight["growth_candidates"]) == 2
    assert thought_capacity_preflight["prune_candidate_count"] == 1
    assert len(thought_capacity_preflight["prune_candidates"]) == 1
    assert thought_capacity_preflight["execution_allowed"] is False
    assert thought_capacity_preflight["resize_allowed"] is False
    assert thought_capacity_preflight["growth_allowed"] is False
    assert thought_capacity_preflight["prune_allowed"] is False
    assert thought_capacity_preflight["replay_allowed"] is False
    assert thought_capacity_preflight["plasticity_allowed"] is False
    assert thought_capacity_preflight["training_allowed"] is False
    assert thought_capacity_preflight["checkpoint_allowed"] is False
    assert thought_capacity_preflight["fact_promotion_allowed"] is False
    assert thought_capacity_preflight["action_allowed"] is False
    assert thought_capacity_preflight["cognition_substrate_claimed"] is False
    assert snn_language_thought_capacity_mutation_preflight["promotion_gate"][
        "eligible_for_autonomous_snn_language_thought_capacity_mutation_executor"
    ] is True
    assert snn_language_thought_capacity_mutation_preflight["promotion_gate"][
        "eligible_for_cognition_substrate"
    ] is False
    assert snn_language_thought_capacity_mutation_preflight["promotion_gate"][
        "eligible_for_fact_promotion"
    ] is False
    assert snn_language_thought_capacity_mutation_preflight["promotion_gate"][
        "eligible_for_action"
    ] is False


def test_readout_ledger_designs_activity_gated_newborn_neuron_integration_without_mutation() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    hashes = {
        name: _sha256_json(name)
        for name in (
            "review",
            "event",
            "preflight",
            "design",
            "structural",
            "memory",
            "target-0",
            "target-1",
            "token-0",
            "token-1",
            "projection-0",
            "projection-1",
            "active-0",
            "active-1",
            "membrane-0",
            "membrane-1",
        )
    }
    growth_candidates = [
        {
            "growth_candidate_id": f"growth-{index}",
            "local_learning_target_hash": hashes[f"target-{index}"],
            "generated_token_hash": hashes[f"token-{index}"],
            "spike_projection_hash": hashes[f"projection-{index}"],
            "active_neuron_hash": hashes[f"active-{index}"],
            "membrane_state_hash": hashes[f"membrane-{index}"],
            "proposed_new_neuron_count": 1,
            "proposed_new_synapse_count": 1,
            "applied_to_runtime": True,
            "applied_in_ledger": True,
        }
        for index in range(2)
    ]
    capacity_review = {
        "surface": (
            "snn_language_autonomous_snn_language_thought_"
            "capacity_mutation_event_review.v1"
        ),
        "ready": True,
        "accepted": True,
        "review_hash": hashes["review"],
        "requires_operator_approval": False,
        "mutates_runtime_state": False,
        "capacity_mutation_event_hash": hashes["event"],
        "autonomous_snn_language_thought_capacity_mutation_event_review": {
            "capacity_mutation_event_hash": hashes["event"],
            "preflight_hash": hashes["preflight"],
            "thought_capacity_mutation_design_hash": hashes["design"],
            "structural_event_review_hash": hashes["structural"],
            "memory_trace_hash": hashes["memory"],
            "requested_device": "cuda:0",
            "actual_device": "cuda:0",
            "tensor_is_cuda": True,
            "current_neuron_capacity": 64,
            "target_neuron_capacity": 66,
            "added_neuron_capacity": 2,
            "new_region_nonzero_count": 0,
            "new_region_zero_initialized": True,
            "growth_candidate_count": 2,
            "growth_candidates": growth_candidates,
            "newborn_neuron_slots_untrained": True,
            "newborn_neuron_slots_inactive": True,
        },
        "promotion_gate": {
            "eligible_for_autonomous_snn_language_thought_"
            "newborn_neuron_integration_design": True
        },
    }
    before = runtime_state.snapshot()

    design = (
        ledger.autonomous_snn_language_thought_newborn_neuron_integration_design(
            autonomous_snn_language_thought_capacity_mutation_event_review=(
                capacity_review
            ),
            integration_policy={
                "max_newborn_neurons": 2,
                "max_seed_synapses_per_newborn": 2,
                "critical_period_cycles": 64,
                "required_coactivation_events": 4,
                "inactivity_prune_cycles": 128,
                "max_initial_weight": 0.04,
                "target_firing_rate_hz": 4.0,
                "max_firing_rate_hz": 16.0,
            },
        )
    )
    repeat = (
        ledger.autonomous_snn_language_thought_newborn_neuron_integration_design(
            autonomous_snn_language_thought_capacity_mutation_event_review=(
                capacity_review
            ),
            integration_policy={
                "max_newborn_neurons": 2,
                "max_seed_synapses_per_newborn": 2,
                "critical_period_cycles": 64,
                "required_coactivation_events": 4,
                "inactivity_prune_cycles": 128,
                "max_initial_weight": 0.04,
                "target_firing_rate_hz": 4.0,
                "max_firing_rate_hz": 16.0,
            },
        )
    )
    blocked_review = deepcopy(capacity_review)
    blocked_review["accepted"] = False
    blocked = (
        ledger.autonomous_snn_language_thought_newborn_neuron_integration_design(
            autonomous_snn_language_thought_capacity_mutation_event_review=(
                blocked_review
            )
        )
    )
    after = runtime_state.snapshot()

    assert before == after
    assert design == repeat
    assert design["surface"] == (
        "snn_language_autonomous_snn_language_thought_"
        "newborn_neuron_integration_design.v1"
    )
    assert design["accepted"] is True
    assert design["ready"] is True
    assert design["requires_operator_approval"] is False
    assert design["mutates_runtime_state"] is False
    assert design["adds_neurons"] is False
    assert design["adds_synapses"] is False
    assert design["applies_plasticity"] is False
    assert design["trains_runtime_model"] is False
    assert design["writes_checkpoint"] is False
    assert design["generates_text"] is False
    assert len(
        design["thought_newborn_neuron_integration_design_hash"]
    ) == 64
    body = design[
        "autonomous_snn_language_thought_newborn_neuron_integration_design"
    ]
    assert body["newborn_neuron_indices"] == [64, 65]
    assert body["newborn_neuron_count"] == 2
    assert body["integration_candidate_count"] == 2
    assert body["integration_mode"] == (
        "activity_gated_critical_period_homeostatic"
    )
    assert body["critical_period_cycles"] == 64
    assert body["inactivity_prune_cycles"] == 128
    assert body["source_indices_resolved"] is False
    assert body["connections_applied"] is False
    assert body["weights_applied"] is False
    assert body["critical_period_started"] is False
    assert body["newborn_neurons_active"] is False
    assert body["newborn_neurons_trained"] is False
    assert [
        candidate["target_neuron_index"]
        for candidate in body["integration_candidates"]
    ] == [64, 65]
    assert all(
        candidate["source_neuron_index"] is None
        and candidate["source_resolution_mode"]
        == "activity_hash_to_live_spike_population"
        and candidate["initialization_mode"]
        == "zero_weight_activity_gated"
        and candidate["proposed_initial_weight"] == 0.0
        and candidate["max_initial_weight"] == 0.04
        and candidate["connection_applied"] is False
        and candidate["newborn_active"] is False
        and len(candidate["integration_candidate_hash"]) == 64
        and len(candidate["source_candidate_hash"]) == 64
        for candidate in body["integration_candidates"]
    )
    assert design["promotion_gate"][
        "eligible_for_autonomous_snn_language_thought_"
        "newborn_neuron_integration_preflight"
    ] is True
    assert blocked["accepted"] is False
    assert blocked["promotion_gate"]["required_evidence"][
        "capacity_mutation_event_review_ready"
    ] is False
    assert blocked[
        "autonomous_snn_language_thought_newborn_neuron_integration_design"
    ]["integration_candidates"] == []


def test_readout_ledger_preflights_hash_bound_live_newborn_sources_without_mutation() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: {},
    )
    active_populations = [[1, 4, 7], [2, 5, 8]]
    design_candidates = []
    for offset, active_indices in enumerate(active_populations):
        candidate_material = {
            "newborn_neuron_index": 64 + offset,
            "newborn_offset": offset,
            "source_growth_candidate_id": f"growth-{offset}",
            "source_candidate_hash": _sha256_json(
                {"growth_candidate_id": f"growth-{offset}"}
            ),
            "local_learning_target_hash": _sha256_json(f"target-{offset}"),
            "generated_token_hash": _sha256_json(f"token-{offset}"),
            "spike_projection_hash": _sha256_json(f"projection-{offset}"),
            "active_neuron_hash": _sha256_json(active_indices),
            "membrane_state_hash": _sha256_json(f"membrane-{offset}"),
            "source_resolution_mode": "activity_hash_to_live_spike_population",
            "source_neuron_index": None,
            "target_neuron_index": 64 + offset,
            "initialization_mode": "zero_weight_activity_gated",
            "proposed_initial_weight": 0.0,
            "max_initial_weight": 0.04,
            "max_seed_synapses": 2,
            "critical_period_cycles": 64,
            "required_coactivation_events": 4,
            "inactivity_prune_cycles": 128,
            "target_firing_rate_hz": 4.0,
            "max_firing_rate_hz": 16.0,
            "requested_device": "cuda:0",
            "actual_device": "cuda:0",
            "tensor_is_cuda": True,
        }
        design_candidates.append(
            {
                "integration_candidate_id": f"newborn-{64 + offset}",
                **candidate_material,
                "integration_candidate_hash": _sha256_json(candidate_material),
                "connection_applied": False,
                "weight_applied": False,
                "newborn_active": False,
                "newborn_trained": False,
                "critical_period_started": False,
            }
        )
    design = {
        "surface": (
            "snn_language_autonomous_snn_language_thought_"
            "newborn_neuron_integration_design.v1"
        ),
        "ready": True,
        "accepted": True,
        "requires_operator_approval": False,
        "mutates_runtime_state": False,
        "thought_newborn_neuron_integration_design_hash": _sha256_json(
            "newborn-design"
        ),
        "autonomous_snn_language_thought_newborn_neuron_integration_design": {
            "capacity_mutation_event_hash": _sha256_json("capacity-event"),
            "current_neuron_capacity": 64,
            "target_neuron_capacity": 66,
            "newborn_neuron_indices": [64, 65],
            "integration_candidates": design_candidates,
            "actual_device": "cuda:0",
        },
        "promotion_gate": {
            "eligible_for_autonomous_snn_language_thought_"
            "newborn_neuron_integration_preflight": True
        },
    }
    live_spike_evidence = {
        "surface": "snn_language_live_spike_population_evidence.v1",
        "state_revision": runtime_state.state_revision,
        "observation_window_id": "window-newborn-1",
        "device": "cuda:0",
        "tensor_is_cuda": True,
        "candidate_observations": [
            {
                "integration_candidate_id": candidate[
                    "integration_candidate_id"
                ],
                "active_neuron_indices": active_indices,
                "active_neuron_hash": _sha256_json(active_indices),
                "spike_projection_hash": candidate["spike_projection_hash"],
                "membrane_state_hash": candidate["membrane_state_hash"],
                "device": "cuda:0",
                "tensor_is_cuda": True,
                "source_activity": [
                    {
                        "neuron_index": active_indices[0],
                        "coactivation_event_count": 5,
                        "firing_rate_hz": 6.0,
                    },
                    {
                        "neuron_index": active_indices[1],
                        "coactivation_event_count": 7,
                        "firing_rate_hz": 8.0,
                    },
                ],
            }
            for candidate, active_indices in zip(
                design_candidates, active_populations, strict=True
            )
        ],
    }
    live_spike_evidence["observation_window_hash"] = _sha256_json(
        {
            "surface": live_spike_evidence["surface"],
            "state_revision": live_spike_evidence["state_revision"],
            "observation_window_id": live_spike_evidence[
                "observation_window_id"
            ],
            "device": live_spike_evidence["device"],
            "tensor_is_cuda": live_spike_evidence["tensor_is_cuda"],
            "candidate_observations": live_spike_evidence[
                "candidate_observations"
            ],
        }
    )
    runtime = {
        "surface": "snn_language_plasticity_runtime_state.v1",
        "language_capacity": {
            "language_neuron_count": 66,
            "sparse_edge_budget": 258,
            "dynamic_capacity_enabled": True,
        },
        "dense_readout_tensor": {
            "available": True,
            "shape": [66, 66],
            "device": "cuda:0",
            "is_cuda": True,
        },
        "sparse_transition_weights": {},
    }
    checkpoint = {
        "checkpoint_path": "memory://before-newborn-integration",
        "snapshot_id": "newborn-integration-snapshot",
        "pre_integration_checkpoint_saved": True,
        "pre_integration_checkpoint_restore_verified": True,
    }
    capabilities = {
        "autonomous_snn_language_thought_"
        "newborn_neuron_integration_executor": True
    }
    before = runtime_state.snapshot()

    preflight = ledger.autonomous_snn_language_thought_newborn_neuron_integration_preflight(
        autonomous_snn_language_thought_newborn_neuron_integration_design=design,
        expected_state_revision=runtime_state.state_revision,
        live_spike_evidence=live_spike_evidence,
        plasticity_runtime_state=runtime,
        checkpoint_transaction=checkpoint,
        executor_capabilities=capabilities,
    )
    repeat = ledger.autonomous_snn_language_thought_newborn_neuron_integration_preflight(
        autonomous_snn_language_thought_newborn_neuron_integration_design=design,
        expected_state_revision=runtime_state.state_revision,
        live_spike_evidence=live_spike_evidence,
        plasticity_runtime_state=runtime,
        checkpoint_transaction=checkpoint,
        executor_capabilities=capabilities,
    )
    tampered_evidence = deepcopy(live_spike_evidence)
    tampered_evidence["candidate_observations"][0][
        "active_neuron_indices"
    ] = [1, 4, 9]
    blocked = ledger.autonomous_snn_language_thought_newborn_neuron_integration_preflight(
        autonomous_snn_language_thought_newborn_neuron_integration_design=design,
        expected_state_revision=runtime_state.state_revision,
        live_spike_evidence=tampered_evidence,
        plasticity_runtime_state=runtime,
        checkpoint_transaction=checkpoint,
        executor_capabilities=capabilities,
    )
    after = runtime_state.snapshot()

    assert before == after
    assert preflight == repeat
    assert preflight["surface"] == (
        "snn_language_autonomous_snn_language_thought_"
        "newborn_neuron_integration_preflight.v1"
    )
    assert preflight["ready"] is True
    assert preflight["accepted"] is True
    assert preflight["requires_operator_approval"] is False
    assert preflight["executable"] is True
    assert preflight["mutates_runtime_state"] is False
    assert preflight["adds_synapses"] is False
    assert preflight["applies_plasticity"] is False
    assert preflight["writes_checkpoint"] is False
    assert len(preflight["preflight_hash"]) == 64
    body = preflight[
        "autonomous_snn_language_thought_newborn_neuron_integration_preflight"
    ]
    assert body["resolved_candidate_count"] == 2
    assert body["source_indices_resolved"] is True
    assert body["connections_applied"] is False
    assert body["weights_applied"] is False
    assert body["critical_period_started"] is False
    assert [
        item["source_neuron_index"]
        for item in body["resolved_integration_candidates"]
    ] == [4, 5]
    assert [
        item["target_neuron_index"]
        for item in body["resolved_integration_candidates"]
    ] == [64, 65]
    assert all(
        len(item["source_resolution_hash"]) == 64
        and item["connection_applied"] is False
        and item["weight_applied"] is False
        for item in body["resolved_integration_candidates"]
    )
    assert preflight["promotion_gate"][
        "eligible_for_autonomous_snn_language_thought_"
        "newborn_neuron_integration_executor"
    ] is True
    assert blocked["ready"] is False
    assert blocked["promotion_gate"]["required_evidence"][
        "all_candidate_sources_resolved"
    ] is False
    assert blocked["promotion_gate"]["candidate_evidence"][0][
        "active_population_hash_matches_lineage"
    ] is False
    assert blocked[
        "autonomous_snn_language_thought_newborn_neuron_integration_preflight"
    ]["resolved_integration_candidates"] == []


def test_readout_ledger_reviews_newborn_integration_sparse_dense_and_provenance() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    runtime_state.mark_mutated()
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: {},
    )
    integrated_synapse = {
        "synapse": "4:64",
        "source_neuron_index": 4,
        "target_neuron_index": 64,
        "seed_weight": 0.01,
        "max_initial_weight": 0.04,
        "coactivation_event_count": 7,
        "source_firing_rate_hz": 8.0,
        "target_firing_rate_hz": 4.0,
        "critical_period_cycles": 64,
        "inactivity_prune_cycles": 128,
        "max_seed_synapses": 2,
        "integration_candidate_id": "newborn-64",
        "integration_candidate_hash": "i" * 64,
        "source_candidate_hash": "s" * 64,
        "source_resolution_hash": "r" * 64,
        "active_neuron_hash": "a" * 64,
        "spike_projection_hash": "b" * 64,
        "membrane_state_hash": "m" * 64,
        "actual_device": "cpu",
        "tensor_is_cuda": False,
        "connection_applied": True,
        "weight_applied": True,
        "critical_period_started": True,
    }
    integrated_synapse["newborn_integration_synapse_hash"] = _sha256_json(
        integrated_synapse
    )
    event = {
        "completed_at": "2026-06-06T00:00:00+00:00",
        "before_state_revision": 0,
        "after_state_revision": 1,
        "preflight_hash": "p" * 64,
        "thought_newborn_neuron_integration_design_hash": "d" * 64,
        "capacity_mutation_event_hash": "c" * 64,
        "observation_window_id": "window-1",
        "observation_window_hash": "o" * 64,
        "checkpoint_path": "memory://pre-integration",
        "committed_checkpoint_path": "memory://committed-integration",
        "actual_device": "cpu",
        "tensor_is_cuda": False,
        "current_neuron_capacity": 64,
        "target_neuron_capacity": 66,
        "newborn_neuron_indices": [64, 65],
        "integrated_synapse_count": 1,
        "integrated_synapses": [integrated_synapse],
        "critical_period_started": True,
        "replay_executed": False,
        "training_executed": False,
        "plasticity_applied": True,
    }
    event["newborn_neuron_integration_event_hash"] = _sha256_json(event)
    executor = {
        "surface": (
            "snn_language_autonomous_snn_language_thought_"
            "newborn_neuron_integration_executor.v1"
        ),
        "accepted": True,
        "ready": True,
        "requires_operator_approval": False,
        "runs_replay": False,
        "trains_runtime_model": False,
        "generates_text": False,
        "decodes_text": False,
        "applies_plasticity": True,
        "resizes_network": False,
        "adds_neurons": False,
        "adds_synapses": True,
        "prunes_network": False,
        "checkpoint_transaction": {
            "pre_integration_checkpoint_saved": True,
            "restore_verified": True,
            "post_integration_checkpoint_saved": True,
            "post_integration_checkpoint_restore_verified": True,
            "committed_checkpoint_path": "memory://committed-integration",
        },
        "autonomous_snn_language_thought_newborn_neuron_integration_event": (
            event
        ),
        "promotion_gate": {
            "eligible_for_autonomous_snn_language_thought_"
            "newborn_neuron_integration_event_review": True
        },
    }
    runtime = {
        "surface": "snn_language_plasticity_runtime_state.v1",
        "language_capacity": {"language_neuron_count": 66},
        "dense_readout_tensor": {
            "available": True,
            "shape": [66, 66],
            "device": "cpu",
            "is_cuda": False,
        },
        "sparse_transition_weights": {"4:64": 0.01},
        "synapse_provenance_by_key": {
            "4:64": {
                "provenance_type": "newborn_neuron_integration",
                "preflight_hash": "p" * 64,
                **integrated_synapse,
            }
        },
        "newborn_integration_dense_samples": [
            {
                "synapse": "4:64",
                "source_neuron_index": 4,
                "target_neuron_index": 64,
                "weight": 0.01,
            }
        ],
        "last_thought_newborn_neuron_integration": event,
        "last_checkpoint_path": "memory://committed-integration",
    }
    before = runtime_state.snapshot()

    review = ledger.autonomous_snn_language_thought_newborn_neuron_integration_event_review(
        autonomous_snn_language_thought_newborn_neuron_integration_executor=(
            executor
        ),
        plasticity_runtime_state=runtime,
        expected_state_revision=1,
    )
    learning_design = ledger.autonomous_snn_language_thought_newborn_neuron_critical_period_learning_design(
        autonomous_snn_language_thought_newborn_neuron_integration_event_review=(
            review
        ),
        learning_policy={
            "max_learning_rate": 0.005,
            "depression_ratio": 0.5,
            "min_survival_activity_ratio": 0.25,
            "homeostatic_tolerance_ratio": 0.5,
        },
    )
    learning_candidate = learning_design[
        "autonomous_snn_language_thought_newborn_neuron_"
        "critical_period_learning_design"
    ]["learning_candidates"][0]
    activity_material = {
        "synapse": "4:64",
        "critical_period_learning_candidate_hash": learning_candidate[
            "critical_period_learning_candidate_hash"
        ],
        "cycle_index": 1,
        "pre_spike_times_ms": [1.0, 10.0],
        "post_spike_times_ms": [5.0, 12.0],
        "newborn_firing_rate_hz": 4.0,
        "prediction_error": 0.2,
        "device": "cpu",
        "tensor_is_cuda": False,
    }
    activity_evidence = {
        "surface": "snn_language_newborn_critical_period_activity.v1",
        "state_revision": 1,
        "observation_window_id": "critical-period-window-1",
        "device": "cpu",
        "tensor_is_cuda": False,
        "candidate_observations": [
            {
                **activity_material,
                "candidate_activity_hash": _sha256_json(
                    activity_material
                ),
            }
        ],
    }
    activity_evidence["observation_window_hash"] = _sha256_json(
        {
            "surface": activity_evidence["surface"],
            "state_revision": activity_evidence["state_revision"],
            "observation_window_id": activity_evidence[
                "observation_window_id"
            ],
            "device": activity_evidence["device"],
            "tensor_is_cuda": activity_evidence["tensor_is_cuda"],
            "candidate_observations": [activity_material],
        }
    )
    learning_preflight = ledger.autonomous_snn_language_thought_newborn_neuron_critical_period_learning_preflight(
        autonomous_snn_language_thought_newborn_neuron_critical_period_learning_design=(
            learning_design
        ),
        expected_state_revision=1,
        critical_period_activity_evidence=activity_evidence,
        plasticity_runtime_state=runtime,
        checkpoint_transaction={
            "checkpoint_path": "memory://pre-learning",
            "pre_learning_checkpoint_saved": True,
            "pre_learning_checkpoint_restore_verified": True,
        },
        executor_capabilities={
            "autonomous_snn_language_thought_newborn_neuron_"
            "critical_period_learning_executor": True
        },
    )
    tampered_runtime = deepcopy(runtime)
    tampered_runtime["newborn_integration_dense_samples"][0]["weight"] = 0.02
    blocked = ledger.autonomous_snn_language_thought_newborn_neuron_integration_event_review(
        autonomous_snn_language_thought_newborn_neuron_integration_executor=(
            executor
        ),
        plasticity_runtime_state=tampered_runtime,
        expected_state_revision=1,
    )
    after = runtime_state.snapshot()

    assert before == after
    assert review["accepted"] is True
    assert review["ready"] is True
    assert review["state_revision_unchanged"] is True
    assert review["requires_operator_approval"] is False
    assert review["mutates_runtime_state"] is False
    assert review["writes_checkpoint"] is False
    assert review["applies_plasticity"] is False
    assert len(review["review_hash"]) == 64
    body = review[
        "autonomous_snn_language_thought_newborn_neuron_"
        "integration_event_review"
    ]
    assert body["integrated_synapse_count"] == 1
    assert body["sparse_dense_provenance_consistent"] is True
    assert body["critical_period_started"] is True
    assert review["promotion_gate"][
        "eligible_for_autonomous_snn_language_thought_"
        "newborn_neuron_critical_period_learning_design"
    ] is True
    assert learning_design["accepted"] is True
    assert learning_design["requires_operator_approval"] is False
    assert learning_design["mutates_runtime_state"] is False
    assert learning_design["applies_plasticity"] is False
    assert learning_design["state_revision_unchanged"] is True
    learning_body = learning_design[
        "autonomous_snn_language_thought_newborn_neuron_"
        "critical_period_learning_design"
    ]
    assert learning_body["learning_candidate_count"] == 1
    assert learning_body["learning_candidates"][0]["learning_rule"] == (
        "local_pre_post_timing_with_homeostatic_scaling"
    )
    assert learning_body["learning_candidates"][0][
        "minimum_survival_active_cycles"
    ] == 16
    assert learning_design["promotion_gate"][
        "eligible_for_autonomous_snn_language_thought_newborn_neuron_"
        "critical_period_learning_preflight"
    ] is True
    assert learning_preflight["accepted"] is True
    assert learning_preflight["requires_operator_approval"] is False
    assert learning_preflight["mutates_runtime_state"] is False
    preflight_body = learning_preflight[
        "autonomous_snn_language_thought_newborn_neuron_"
        "critical_period_learning_preflight"
    ]
    assert preflight_body["resolved_cycle_count"] == 1
    resolved_cycle = preflight_body["resolved_learning_cycles"][0]
    assert resolved_cycle["causal_pair_count"] == 3
    assert resolved_cycle["anti_causal_pair_count"] == 1
    assert resolved_cycle["proposed_weight_delta"] == 0.003125
    assert abs(resolved_cycle["proposed_weight"] - 0.013125) < 1e-12
    assert learning_preflight["promotion_gate"][
        "eligible_for_autonomous_snn_language_thought_newborn_neuron_"
        "critical_period_learning_executor"
    ] is True
    assert blocked["accepted"] is False
    assert blocked["promotion_gate"]["required_evidence"][
        "all_integrated_synapses_verified"
    ] is False
    assert blocked["promotion_gate"]["synapse_evidence"][0]["checks"][
        "dense_sample_matches_event"
    ] is False


def test_readout_ledger_emission_review_replay_policy_requires_internal_readout_match() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_readout_emission_review(
        readout_emission=_ready_emission(),
        expected_state_revision=0,
        operator_id="operator-emission",
        confirmation=True,
    )
    runtime_state.mark_clean()
    before_revision = runtime_state.state_revision

    policy = ledger.emission_review_replay_evaluation_policy(limit=4)

    assert runtime_state.state_revision == before_revision
    assert runtime_state.dirty_state is False
    assert policy["surface"] == "snn_language_readout_emission_replay_evaluation_policy.v1"
    assert policy["candidate_count"] == 0
    assert policy["unmatched_emission_review_count"] == 1
    assert policy["internal_readout_evidence_count"] == 0
    assert policy["source_window"]["surface"] == (
        "bounded_snn_emission_review_replay_policy_source_window.v1"
    )
    assert policy["source_window"]["emission_review_event_window_count"] == 1
    assert policy["source_window"]["internal_readout_event_window_count"] == 0
    assert policy["source_window"]["global_candidate_scan"] is False
    assert policy["source_window"]["global_score_scan"] is False
    assert policy["source_window"]["archival_storage_device"] == "cpu"
    assert policy["source_window"]["score_device"] == "cpu"
    assert policy["source_window"]["gpu_used"] is False
    assert policy["source_window"]["raw_text_payload_loaded"] is False
    assert policy["source_window"]["language_reasoning"] is False
    assert policy["promotion_gate"]["eligible_for_operator_replay_evaluation_policy_review"] is False
    assert policy["promotion_gate"]["required_evidence"]["reviewed_emission_available"] is True
    assert policy["promotion_gate"]["required_evidence"]["matching_internal_readout_evidence_available"] is False
    assert policy["promotion_gate"]["required_evidence"]["source_window_bounded"] is True
    assert policy["promotion_gate"]["required_evidence"][
        "archival_metadata_cpu_resident"
    ] is True
    assert policy["promotion_gate"]["required_evidence"][
        "language_reasoning_absent"
    ] is True
    assert policy["promotion_gate"]["required_evidence"]["display_text_not_used_as_replay_source"] is True
    assert policy["generates_text"] is False
    assert policy["decodes_text"] is False
    assert policy["exposes_reviewed_bounded_text"] is False
    assert policy["records_ledger_event"] is False
    assert policy["runs_replay"] is False
    assert policy["runs_live_tick"] is False
    assert policy["runs_every_token"] is False
    assert policy["raw_text_payload_loaded"] is False
    assert policy["language_reasoning"] is False
    assert policy["applies_plasticity"] is False
    assert policy["mutates_runtime_state"] is False
    assert policy["eligible_for_replay_memory"] is False
    assert "text" not in policy["unmatched_emission_reviews"][0]
    assert "events" not in policy
    assert "rollout_events" not in policy


def test_readout_ledger_emission_review_replay_policy_selects_matched_sparse_readout_evidence() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    emission = _ready_emission()
    binding = emission["emission_binding"]  # type: ignore[index]
    assert isinstance(binding, dict)
    ledger.record_readout_draft(
        readout_draft=_ready_draft_for(
            str(binding["prediction_hash"]),
            str(binding["transition_memory_evaluation_hash"]),
            str(binding["persistent_transition_weights_hash"]),
            ["memory pressure"],
        ),
        expected_state_revision=0,
        operator_id="operator-readout",
        confirmation=True,
    )
    ledger.record_readout_emission_review(
        readout_emission=emission,
        expected_state_revision=0,
        operator_id="operator-emission",
        confirmation=True,
    )
    runtime_state.mark_clean()

    policy = ledger.emission_review_replay_evaluation_policy(limit=4)

    assert runtime_state.dirty_state is False
    assert policy["candidate_count"] == 1
    assert policy["ready_candidate_count"] == 1
    assert policy["unmatched_emission_review_count"] == 0
    assert policy["internal_readout_evidence_count"] == 1
    assert policy["source_window"]["surface"] == (
        "bounded_snn_emission_review_replay_policy_source_window.v1"
    )
    assert policy["source_window"]["emission_review_event_window_count"] == 1
    assert policy["source_window"]["internal_readout_event_window_count"] == 1
    assert policy["source_window"]["candidate_count_before_limit"] == 1
    assert policy["source_window"]["candidate_count_returned"] == 1
    assert policy["source_window"]["global_candidate_scan"] is False
    assert policy["source_window"]["global_score_scan"] is False
    assert policy["source_window"]["archival_storage_device"] == "cpu"
    assert policy["source_window"]["score_device"] == "cpu"
    assert policy["source_window"]["gpu_used"] is False
    assert policy["source_window"]["raw_text_payload_loaded"] is False
    assert policy["source_window"]["language_reasoning"] is False
    candidate = policy["candidates"][0]
    assert candidate["emission_hash"] == emission["emission_hash"]
    assert candidate["prediction_hash"] == binding["prediction_hash"]
    assert candidate["transition_memory_evaluation_hash"] == binding["transition_memory_evaluation_hash"]
    assert candidate["persistent_transition_weights_hash"] == binding["persistent_transition_weights_hash"]
    assert candidate["label_count"] == 1
    assert candidate["all_labels_grounded"] is True
    assert candidate["eligible_for_replay_evaluation_policy_review"] is True
    assert candidate["eligible_for_replay_memory"] is False
    assert candidate["eligible_for_live_replay"] is False
    assert candidate["eligible_for_plasticity_application"] is False
    assert candidate["eligible_for_fact_promotion"] is False
    assert candidate["eligible_for_action"] is False
    assert "text" not in candidate
    assert "labels" not in candidate
    assert candidate["text_hash_material"] == "hash_only_binding_no_raw_review_text"
    assert candidate["source_window"]["surface"] == (
        "bounded_snn_emission_review_replay_policy_source_window.v1"
    )
    assert policy["promotion_gate"]["eligible_for_operator_replay_evaluation_policy_review"] is True
    assert policy["promotion_gate"]["eligible_for_replay_memory"] is False
    assert policy["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert policy["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert policy["promotion_gate"]["eligible_for_action"] is False
    assert policy["promotion_gate"]["required_evidence"]["source_window_bounded"] is True
    assert policy["promotion_gate"]["required_evidence"][
        "archival_metadata_cpu_resident"
    ] is True
    assert policy["promotion_gate"]["required_evidence"][
        "language_reasoning_absent"
    ] is True


def test_readout_ledger_emission_review_replay_policy_caps_review_and_readout_source_windows_before_match() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
        limit=SNN_EMISSION_REVIEW_REPLAY_POLICY_SOURCE_WINDOW_LIMIT * 2,
    )
    outside_prediction_hash = ""
    for index in range(SNN_EMISSION_REVIEW_REPLAY_POLICY_SOURCE_WINDOW_LIMIT * 2):
        label = "outside-window" if index == 0 else f"review-window-{index}"
        prediction_hash = _sha256_json({"prediction": index})
        evaluation_hash = _sha256_json({"evaluation": index})
        weights_hash = _sha256_json({"weights": index})
        if index == 0:
            outside_prediction_hash = prediction_hash
        ledger.record_readout_draft(
            readout_draft=_ready_draft_for(
                prediction_hash,
                evaluation_hash,
                weights_hash,
                [label],
            ),
            expected_state_revision=runtime_state.state_revision,
            operator_id="operator-readout",
            confirmation=True,
        )
        emission = deepcopy(_ready_emission())
        emission["emission_hash"] = _sha256_json({"emission": index})
        emission["language_output"] = {
            "text": f"review text {index}",
            "labels": [label],
            "term_count": 1,
            "max_terms": 12,
        }
        emission["emission_binding"] = {
            "trajectory_hash": _sha256_json({"trajectory": index}),
            "prediction_hash": prediction_hash,
            "transition_memory_evaluation_hash": evaluation_hash,
            "persistent_transition_weights_hash": weights_hash,
        }
        ledger.record_readout_emission_review(
            readout_emission=emission,
            expected_state_revision=runtime_state.state_revision,
            operator_id="operator-emission",
            confirmation=True,
        )
    runtime_state.mark_clean()
    before = runtime_state.snapshot()

    policy = ledger.emission_review_replay_evaluation_policy(limit=8)
    design = ledger.emission_review_replay_evaluation_design(
        policy,
        design_policy={"max_candidates": 8, "min_ready_candidates": 1},
        device_evidence={"device": "cpu", "source": "test_emission_review_window"},
    )
    after = runtime_state.snapshot()
    selected_prediction_hashes = {
        str(candidate.get("prediction_hash") or "")
        for candidate in policy["candidates"]
    }
    selected_seed_prediction_hashes = {
        str(seed.get("prediction_hash") or "")
        for seed in design["selected_replay_context_seeds"]
    }

    assert before == after
    assert runtime_state.dirty_state is False
    assert policy["candidate_count"] == 8
    assert policy["source_window"]["emission_review_event_retention_count"] == (
        SNN_EMISSION_REVIEW_REPLAY_POLICY_SOURCE_WINDOW_LIMIT * 2
    )
    assert policy["source_window"]["internal_readout_event_retention_count"] == (
        SNN_EMISSION_REVIEW_REPLAY_POLICY_SOURCE_WINDOW_LIMIT * 2
    )
    assert policy["source_window"]["emission_review_event_window_count"] == (
        SNN_EMISSION_REVIEW_REPLAY_POLICY_SOURCE_WINDOW_LIMIT
    )
    assert policy["source_window"]["internal_readout_event_window_count"] == (
        SNN_EMISSION_REVIEW_REPLAY_POLICY_SOURCE_WINDOW_LIMIT
    )
    assert policy["source_window"]["candidate_count_before_limit"] == (
        SNN_EMISSION_REVIEW_REPLAY_POLICY_SOURCE_WINDOW_LIMIT
    )
    assert policy["source_window"]["candidate_count_returned"] == 8
    assert policy["source_window"]["global_candidate_scan"] is False
    assert policy["source_window"]["global_score_scan"] is False
    assert policy["source_window"]["runs_live_tick"] is False
    assert policy["source_window"]["runs_every_token"] is False
    assert policy["source_window"]["archival_storage_device"] == "cpu"
    assert policy["source_window"]["gpu_used"] is False
    assert policy["promotion_gate"]["required_evidence"]["source_window_bounded"] is True
    assert outside_prediction_hash not in selected_prediction_hashes
    assert design["source_window"]["internal_readout_event_retention_count"] == (
        SNN_EMISSION_REVIEW_REPLAY_POLICY_SOURCE_WINDOW_LIMIT * 2
    )
    assert design["source_window"]["internal_readout_event_window_count"] == (
        SNN_EMISSION_REVIEW_REPLAY_POLICY_SOURCE_WINDOW_LIMIT
    )
    assert design["source_window"]["global_candidate_scan"] is False
    assert design["source_window"]["global_score_scan"] is False
    assert design["source_window"]["runs_live_tick"] is False
    assert design["source_window"]["runs_every_token"] is False
    assert design["source_window"]["archival_storage_device"] == "cpu"
    assert design["source_window"]["gpu_used"] is False
    assert design["promotion_gate"]["required_evidence"]["policy_source_window_bounded"] is True
    assert design["promotion_gate"]["required_evidence"]["design_source_window_bounded"] is True
    assert design["promotion_gate"]["required_evidence"]["language_reasoning_absent"] is True
    assert design["promotion_gate"]["required_evidence"][
        "archival_metadata_cpu_resident"
    ] is True
    assert outside_prediction_hash not in selected_seed_prediction_hashes


def test_readout_ledger_emission_replay_design_requires_device_review_evidence() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    emission = _ready_emission()
    binding = emission["emission_binding"]  # type: ignore[index]
    assert isinstance(binding, dict)
    ledger.record_readout_draft(
        readout_draft=_ready_draft_for(
            str(binding["prediction_hash"]),
            str(binding["transition_memory_evaluation_hash"]),
            str(binding["persistent_transition_weights_hash"]),
            ["memory pressure"],
        ),
        expected_state_revision=0,
        operator_id="operator-readout",
        confirmation=True,
    )
    ledger.record_readout_emission_review(
        readout_emission=emission,
        expected_state_revision=0,
        operator_id="operator-emission",
        confirmation=True,
    )
    runtime_state.mark_clean()
    before_revision = runtime_state.state_revision

    policy = ledger.emission_review_replay_evaluation_policy(limit=4)
    design = ledger.emission_review_replay_evaluation_design(policy)

    assert runtime_state.state_revision == before_revision
    assert runtime_state.dirty_state is False
    assert design["surface"] == "snn_language_readout_emission_replay_evaluation_design.v1"
    assert design["selected_replay_context_seeds"][0]["internal_readout_ledger_match"] is True
    assert design["promotion_gate"]["eligible_for_operator_replay_context_review"] is False
    assert design["promotion_gate"]["required_evidence"]["device_review_evidence_available"] is False
    assert design["records_ledger_event"] is False
    assert design["runs_replay"] is False
    assert design["mutates_runtime_state"] is False
    assert design["eligible_for_replay_memory"] is False
    assert "text" not in design["selected_replay_context_seeds"][0]
    assert "labels" not in design["selected_replay_context_seeds"][0]


def test_readout_ledger_emission_replay_design_builds_hash_only_replay_context_seed() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    emission = _ready_emission()
    binding = emission["emission_binding"]  # type: ignore[index]
    assert isinstance(binding, dict)
    record = ledger.record_readout_draft(
        readout_draft=_ready_draft_for(
            str(binding["prediction_hash"]),
            str(binding["transition_memory_evaluation_hash"]),
            str(binding["persistent_transition_weights_hash"]),
            ["memory pressure"],
        ),
        expected_state_revision=0,
        operator_id="operator-readout",
        confirmation=True,
    )
    ledger.record_readout_emission_review(
        readout_emission=emission,
        expected_state_revision=0,
        operator_id="operator-emission",
        confirmation=True,
    )
    runtime_state.mark_clean()

    policy = ledger.emission_review_replay_evaluation_policy(limit=4)
    design = ledger.emission_review_replay_evaluation_design(
        policy,
        design_policy={"max_candidates": 1, "min_ready_candidates": 1},
        device_evidence={"device": "cuda:0", "source": "test_emission_replay_design"},
    )

    assert runtime_state.dirty_state is False
    assert design["emission_replay_evaluation_design"]["selected_seed_count"] == 1
    assert design["emission_replay_evaluation_design"]["ready_seed_count"] == 1
    assert design["emission_replay_evaluation_design"]["records_replay_context"] is False
    seed = design["selected_replay_context_seeds"][0]
    assert seed["readout_evidence_hash"] == record["recorded_event"]["readout_evidence_hash"]
    assert seed["emission_hash"] == emission["emission_hash"]
    assert seed["internal_readout_ledger_match"] is True
    assert seed["eligible_for_replay_context_review"] is True
    assert seed["eligible_for_replay_memory"] is False
    assert seed["eligible_for_live_replay"] is False
    assert seed["eligible_for_plasticity_application"] is False
    assert "replay_context_seed_hash" in seed
    assert "text" not in seed
    assert "labels" not in seed
    assert design["replay_context_review_requirements"]["accepts_display_text"] is False
    assert design["replay_context_review_requirements"]["accepts_labels_as_replay_window"] is False
    assert design["promotion_gate"]["eligible_for_operator_replay_context_review"] is True
    assert design["promotion_gate"]["eligible_for_replay_context_recording"] is False
    assert design["promotion_gate"]["eligible_for_replay_memory"] is False
    assert design["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert design["promotion_gate"]["next_gate"] == (
        "/terminus/snn-language-sequence/replay-evaluation-context"
    )


def test_runtime_facade_emission_replay_context_review_blocks_oversized_source_windows() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    emission = _ready_emission()
    binding = emission["emission_binding"]  # type: ignore[index]
    assert isinstance(binding, dict)
    ledger.record_readout_draft(
        readout_draft=_ready_draft_for(
            str(binding["prediction_hash"]),
            str(binding["transition_memory_evaluation_hash"]),
            str(binding["persistent_transition_weights_hash"]),
            ["memory pressure"],
        ),
        expected_state_revision=0,
        operator_id="operator-readout",
        confirmation=True,
    )
    ledger.record_readout_emission_review(
        readout_emission=emission,
        expected_state_revision=0,
        operator_id="operator-emission",
        confirmation=True,
    )
    policy = ledger.emission_review_replay_evaluation_policy(limit=4)
    design = ledger.emission_review_replay_evaluation_design(
        policy,
        design_policy={"max_candidates": 1, "min_ready_candidates": 1},
        device_evidence={"device": "cpu", "source": "test_emission_context"},
    )
    prediction_report = {
        "surface": "snn_language_sequence_prediction_probe.v1",
        "provenance_evidence": {"prediction_hash": str(binding["prediction_hash"])},
    }
    observed_slot = {"label": "memory pressure", "pressure_band": "high", "grounded": True}
    calls = {"mismatch": 0, "pressure": 0, "context": 0}

    class _StatusReadModel:
        def snn_language_sequence_mismatch_probe(self, **kwargs: object) -> dict[str, object]:
            calls["mismatch"] += 1
            return {
                "surface": "snn_language_sequence_mismatch_probe.v1",
                "mismatch_hash": "mismatch-hash",
                "prediction_report": kwargs.get("prediction_report"),
            }

        def snn_language_plasticity_pressure(self, **kwargs: object) -> dict[str, object]:
            calls["pressure"] += 1
            return {
                "surface": "snn_language_plasticity_pressure.v1",
                "pressure_hash": "pressure-hash",
                "mismatch_report": kwargs.get("mismatch_report"),
            }

    class _ReplayController:
        def record_snn_replay_evaluation_context(self, **kwargs: object) -> dict[str, object]:
            calls["context"] += 1
            runtime_state.mark_dirty_without_revision()
            source_metadata = kwargs.get("source_metadata")
            return {
                "surface": "snn_replay_evaluation_context.v1",
                "replay_evaluation_context_id": "context-1",
                "evidence_hash": "context-hash",
                "source_metadata_hash": _sha256_json(source_metadata),
                "mismatch_hash": "mismatch-hash",
                "pressure_hash": "pressure-hash",
            }

    class _Root:
        _runtime_state = runtime_state
        _snn_language_readout_ledger = ledger
        _status_read_model = _StatusReadModel()
        _replay_controller = _ReplayController()

    facade = RuntimeFacade(_Root())
    accepted = facade.snn_language_readout_emission_replay_context_review(
        emission_replay_evaluation_design=design,
        prediction_report=prediction_report,
        observed_readout_slots=[observed_slot],
        operator_id="operator-test",
        confirmation=True,
    )
    oversized_seed_design = deepcopy(design)
    seed = dict(design["selected_replay_context_seeds"][0])
    oversized_seed_design["selected_replay_context_seeds"] = [
        {**seed, "replay_context_seed_hash": f"seed-{index}"}
        for index in range(SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT + 1)
    ]
    seed_block = facade.snn_language_readout_emission_replay_context_review(
        emission_replay_evaluation_design=oversized_seed_design,
        prediction_report=prediction_report,
        observed_readout_slots=[observed_slot],
        operator_id="operator-test",
        confirmation=True,
    )
    slot_block = facade.snn_language_readout_emission_replay_context_review(
        emission_replay_evaluation_design=design,
        prediction_report=prediction_report,
        observed_readout_slots=[
            {**observed_slot, "label": f"memory pressure {index}"}
            for index in range(SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT + 1)
        ],
        operator_id="operator-test",
        confirmation=True,
    )

    assert accepted["accepted"] is True
    assert accepted["records_replay_context"] is True
    assert accepted["seed_source_window"]["surface"] == (
        "bounded_snn_emission_replay_context_review_seed_window.v1"
    )
    assert accepted["observed_slot_source_window"]["surface"] == (
        "bounded_snn_emission_replay_context_review_observed_slot_window.v1"
    )
    assert accepted["promotion_gate"]["required_evidence"]["seed_payload_not_truncated"] is True
    assert (
        accepted["promotion_gate"]["required_evidence"]["observed_slot_payload_not_truncated"]
        is True
    )
    assert seed_block["accepted"] is False
    assert seed_block["records_replay_context"] is False
    assert seed_block["promotion_gate"]["required_evidence"]["seed_source_window_bounded"] is True
    assert seed_block["promotion_gate"]["required_evidence"]["seed_payload_not_truncated"] is False
    assert seed_block["seed_source_window"]["source_window_count"] == (
        SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
    )
    assert seed_block["seed_source_window"]["source_payload_truncated"] is True
    assert slot_block["accepted"] is False
    assert slot_block["records_replay_context"] is False
    assert (
        slot_block["promotion_gate"]["required_evidence"]["observed_slot_source_window_bounded"]
        is True
    )
    assert (
        slot_block["promotion_gate"]["required_evidence"]["observed_slot_payload_not_truncated"]
        is False
    )
    assert slot_block["observed_slot_source_window"]["source_window_count"] == (
        SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
    )
    assert slot_block["observed_slot_source_window"]["source_payload_truncated"] is True
    assert calls == {"mismatch": 1, "pressure": 1, "context": 1}


def test_runtime_facade_snn_replay_evaluation_context_bounds_observed_slots() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    prediction_report = {
        "surface": "snn_language_sequence_prediction_probe.v1",
        "provenance_evidence": {"prediction_hash": "prediction-hash"},
    }
    observed_slot = {"label": "memory pressure", "pressure_band": "high", "grounded": True}
    calls = {"mismatch": 0, "pressure": 0, "context": 0}

    class _StatusReadModel:
        def snn_language_sequence_mismatch_probe(self, **kwargs: object) -> dict[str, object]:
            calls["mismatch"] += 1
            return {
                "surface": "snn_language_sequence_mismatch_probe.v1",
                "available": True,
                "owned_by_marulho": True,
                "prediction_error": {"mismatch_score": 0.9},
                "observed_slot_count": len(kwargs.get("observed_readout_slots") or []),
            }

        def snn_language_plasticity_pressure(self, **kwargs: object) -> dict[str, object]:
            calls["pressure"] += 1
            return {
                "surface": "snn_language_plasticity_pressure.v1",
                "available": True,
                "owned_by_marulho": True,
                "promotion_gate": {"status": "ready_for_operator_review"},
                "mismatch_report": kwargs.get("mismatch_report"),
            }

    class _ReplayController:
        def record_snn_replay_evaluation_context(self, **kwargs: object) -> dict[str, object]:
            calls["context"] += 1
            runtime_state.mark_dirty_without_revision()
            source_metadata = kwargs.get("source_metadata")
            assert isinstance(source_metadata, dict)
            return {
                "surface": "snn_replay_evaluation_context.v1",
                "available": True,
                "ready": True,
                "owned_by_marulho": True,
                "replay_evaluation_context_id": "context-1",
                "evidence_hash": "context-hash",
                "source_metadata": source_metadata,
                "source_metadata_hash": _sha256_json(source_metadata),
                "mismatch_hash": "mismatch-hash",
                "pressure_hash": "pressure-hash",
            }

    class _Root:
        _runtime_state = runtime_state
        _snn_language_readout_ledger = ledger
        _status_read_model = _StatusReadModel()
        _replay_controller = _ReplayController()

    facade = RuntimeFacade(_Root())
    accepted = facade.snn_replay_evaluation_context(
        prediction_report=prediction_report,
        observed_readout_slots=[
            {**observed_slot, "label": f"memory pressure {index}"}
            for index in range(SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT)
        ],
        device_evidence={"device": "cpu", "source": "test"},
    )
    oversized = facade.snn_replay_evaluation_context(
        prediction_report=prediction_report,
        observed_readout_slots=[
            {**observed_slot, "label": f"memory pressure {index}"}
            for index in range(SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT + 1)
        ],
        device_evidence={"device": "cpu", "source": "test"},
    )

    assert accepted["accepted"] is True
    assert accepted["records_replay_context"] is True
    assert accepted["observed_slot_source_window"]["surface"] == (
        "bounded_snn_replay_evaluation_context_observed_slot_window.v1"
    )
    assert accepted["observed_slot_source_window"]["source_window_count"] == (
        SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
    )
    assert accepted["source_metadata"]["observed_slot_source_window"] == (
        accepted["observed_slot_source_window"]
    )
    assert accepted["promotion_gate"]["required_evidence"][
        "observed_slot_payload_not_truncated"
    ] is True
    assert oversized["accepted"] is False
    assert oversized["records_replay_context"] is False
    assert oversized["observed_slot_source_window"]["source_window_count"] == (
        SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
    )
    assert oversized["observed_slot_source_window"]["source_payload_truncated"] is True
    assert oversized["promotion_gate"]["required_evidence"][
        "observed_slot_payload_not_truncated"
    ] is False
    assert calls == {"mismatch": 1, "pressure": 1, "context": 1}


def test_readout_ledger_blocks_unready_or_unconfirmed_emission_review() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    unready = _ready_emission()
    promotion_gate = dict(unready["promotion_gate"])  # type: ignore[index]
    promotion_gate["eligible_for_operator_display"] = False
    unready["ready"] = False
    unready["promotion_gate"] = promotion_gate

    result = ledger.record_readout_emission_review(
        readout_emission=unready,
        expected_state_revision=0,
        operator_id="operator-emission",
        confirmation=True,
    )
    unconfirmed = ledger.record_readout_emission_review(
        readout_emission=_ready_emission(),
        expected_state_revision=0,
        operator_id="operator-emission",
        confirmation=False,
    )

    assert result["accepted"] is False
    assert result["promotion_gate"]["required_evidence"]["emission_ready"] is False
    assert result["promotion_gate"]["eligible_for_operator_display_history"] is False
    assert unconfirmed["accepted"] is False
    assert unconfirmed["promotion_gate"]["required_evidence"]["confirmation"] is False
    assert runtime_state.state_revision == 0
    assert runtime_state.dirty_state is False


def test_readout_ledger_records_rollout_replay_evaluation_separately_from_replay_priority() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )

    result = ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=_ready_rollout_replay_evaluation(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    duplicate = ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=_ready_rollout_replay_evaluation(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    snapshot = ledger.snapshot(limit=4)
    priority = ledger.replay_priority(limit=4)

    assert result["accepted"] is True
    assert result["mutates_runtime_state"] is True
    assert result["promotion_gate"]["eligible_for_rollout_replay_memory"] is True
    assert result["promotion_gate"]["eligible_for_replay_priority"] is False
    assert result["recorded_event"]["rollout_replay_evaluation_hash"] == "rollout-eval-hash-1"
    assert result["recorded_event"]["server_transition_memory_hash"] == "weights-hash-1"
    assert result["recorded_event"]["server_transition_memory_hash_match"] is True
    assert (
        result["recorded_event"]["transition_memory_state_source"]
        == "service.runtime_facade.snn_language_plasticity_runtime_state"
    )
    assert result["recorded_event"]["recorded_in_ledger"] is True
    assert result["recorded_event"]["eligible_for_replay_priority"] is False
    assert duplicate["accepted"] is True
    assert duplicate["duplicate"] is True
    assert duplicate["mutates_runtime_state"] is False
    assert runtime_state.dirty_state is True
    assert runtime_state.state_revision == 0
    assert snapshot["summary"]["event_count"] == 0
    assert snapshot["summary"]["rollout_event_count"] == 1
    assert snapshot["rollout_events"][0]["replay_targets"][0]["selected_label"] == "memory pressure"
    assert priority["candidate_count"] == 0
    assert priority["promotion_gate"]["eligible_for_operator_replay_review"] is False


def test_readout_ledger_blocks_unready_or_unconfirmed_rollout_replay_evaluation() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    evaluation = _ready_rollout_replay_evaluation()
    evaluation["promotion_gate"] = {"eligible_for_readout_rollout_ledger_recording_review": False}

    result = ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=evaluation,
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    unconfirmed = ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=_ready_rollout_replay_evaluation(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=False,
    )
    stale_revision = ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=_ready_rollout_replay_evaluation(),
        expected_state_revision=runtime_state.state_revision + 1,
        operator_id="operator-test",
        confirmation=True,
    )
    missing_provenance_evaluation = _ready_rollout_replay_evaluation()
    missing_provenance_evaluation["provenance_evidence"] = {}
    missing_provenance = ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=missing_provenance_evaluation,
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    ungrounded_evaluation = _ready_rollout_replay_evaluation()
    ungrounded_evaluation["replay_evaluation"]["replay_targets"][0]["grounded"] = False
    ungrounded = ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=ungrounded_evaluation,
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    invalid_hash_evaluation = _ready_rollout_replay_evaluation()
    invalid_hash_evaluation["replay_evaluation"]["replay_targets"][0]["active_indices_hash"] = "invalid"
    invalid_hash = ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=invalid_hash_evaluation,
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    missing_server_hash_evaluation = _ready_rollout_replay_evaluation()
    missing_server_hash_evaluation["provenance_evidence"].pop("server_transition_memory_hash")
    missing_server_hash = ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=missing_server_hash_evaluation,
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    mismatched_server_hash_evaluation = _ready_rollout_replay_evaluation()
    mismatched_server_hash_evaluation["provenance_evidence"][
        "server_transition_memory_hash_match"
    ] = False
    mismatched_server_hash = ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=mismatched_server_hash_evaluation,
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )

    assert result["accepted"] is False
    assert result["promotion_gate"]["required_evidence"]["rollout_recording_review_ready"] is False
    assert unconfirmed["accepted"] is False
    assert unconfirmed["promotion_gate"]["required_evidence"]["confirmation"] is False
    assert stale_revision["accepted"] is False
    assert stale_revision["promotion_gate"]["required_evidence"]["expected_revision_current"] is False
    assert missing_provenance["accepted"] is False
    assert missing_provenance["promotion_gate"]["required_evidence"]["rollout_hash_available"] is False
    assert ungrounded["accepted"] is False
    assert ungrounded["promotion_gate"]["required_evidence"]["grounded_replay_targets_available"] is False
    assert invalid_hash["accepted"] is False
    assert invalid_hash["promotion_gate"]["required_evidence"]["replay_target_hashes_valid"] is False
    assert missing_server_hash["accepted"] is False
    assert (
        missing_server_hash["promotion_gate"]["required_evidence"][
            "server_transition_memory_hash_available"
        ]
        is False
    )
    assert mismatched_server_hash["accepted"] is False
    assert (
        mismatched_server_hash["promotion_gate"]["required_evidence"][
            "server_transition_memory_hash_match"
        ]
        is False
    )
    assert ledger.snapshot()["summary"]["rollout_event_count"] == 0
    assert runtime_state.dirty_state is False


def test_readout_ledger_rollout_rehearsal_promotion_policy_is_deterministic_read_only_advisory() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    empty = ledger.rollout_rehearsal_promotion_policy(candidate_limit=4)
    ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=_ready_rollout_replay_evaluation(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    before = runtime_state.snapshot()

    policy = ledger.rollout_rehearsal_promotion_policy(candidate_limit=4)
    repeat = ledger.rollout_rehearsal_promotion_policy(candidate_limit=4)
    after = runtime_state.snapshot()

    assert empty["candidate_count"] == 0
    assert empty["promotion_gate"]["eligible_for_operator_rollout_rehearsal_review"] is False
    assert policy == repeat
    assert before == after
    assert policy["surface"] == "snn_language_readout_rollout_rehearsal_promotion_policy.v1"
    assert policy["advisory"] is True
    assert policy["executable"] is False
    assert policy["mutates_runtime_state"] is False
    assert policy["generates_text"] is False
    assert policy["trains_runtime_model"] is False
    assert policy["applies_plasticity"] is False
    assert policy["runs_live_tick"] is False
    assert policy["runs_every_token"] is False
    assert policy["raw_text_payload_loaded"] is False
    assert policy["language_reasoning"] is False
    assert policy["archival_storage_device"] == "cpu"
    assert policy["score_device"] == "cpu"
    assert policy["gpu_used"] is False
    assert policy["global_candidate_scan"] is False
    assert policy["global_score_scan"] is False
    assert policy["source_window"]["surface"] == (
        "bounded_snn_readout_rollout_rehearsal_source_window.v1"
    )
    assert policy["source_window"]["source_event_window_count"] == 1
    assert policy["source_window"]["candidate_count_before_rank"] == 1
    assert policy["source_window"]["candidate_count_returned"] == 1
    assert policy["source_window"]["archival_storage_device"] == "cpu"
    assert policy["source_window"]["score_device"] == "cpu"
    assert policy["source_window"]["gpu_used"] is False
    assert policy["source_window"]["global_candidate_scan"] is False
    assert policy["source_window"]["global_score_scan"] is False
    assert policy["source_window"]["runs_live_tick"] is False
    assert policy["source_window"]["runs_every_token"] is False
    assert policy["source_window"]["language_reasoning"] is False
    assert policy["ledger_summary"]["unique_count_scope"] == "source_window"
    assert policy["candidate_count"] == 1
    assert policy["candidates"][0]["rank"] == 1
    assert policy["candidates"][0]["target_count"] == 2
    assert policy["candidates"][0]["sparse_index_count"] == 4
    assert policy["candidates"][0]["sparse_occupancy"] > 0.0
    assert policy["candidates"][0]["device_evidence"]["tensor_device"] == "cpu"
    assert policy["candidates"][0]["server_transition_memory_hash"] == "weights-hash-1"
    assert policy["candidates"][0]["server_transition_memory_hash_match"] is True
    assert (
        policy["candidates"][0]["transition_memory_state_source"]
        == "service.runtime_facade.snn_language_plasticity_runtime_state"
    )
    assert policy["candidates"][0]["priority_components"]["provenance"] == 1.0
    assert policy["candidates"][0]["priority_components"]["grounding"] == 1.0
    assert policy["candidates"][0]["priority_components"]["trace_integrity"] == 1.0
    assert policy["promotion_gate"]["eligible_for_operator_rollout_rehearsal_review"] is True
    assert policy["promotion_gate"]["eligible_for_replay_priority"] is False
    assert policy["promotion_gate"]["eligible_for_live_replay"] is False
    assert policy["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert policy["promotion_gate"]["required_evidence"]["source_window_bounded"] is True
    assert policy["promotion_gate"]["required_evidence"][
        "archival_metadata_cpu_resident"
    ] is True
    assert policy["promotion_gate"]["required_evidence"][
        "language_reasoning_absent"
    ] is True


def test_readout_ledger_rollout_rehearsal_policy_caps_source_events_before_rank() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
        limit=SNN_ROLLOUT_REHEARSAL_SOURCE_WINDOW_LIMIT * 2,
    )
    for index in range(SNN_ROLLOUT_REHEARSAL_SOURCE_WINDOW_LIMIT * 2):
        label = "outside-rollout-window" if index == 0 else f"rollout-window-{index}"
        ledger.record_readout_rollout_replay_evaluation(
            readout_rollout_replay_evaluation=_ready_rollout_replay_evaluation_for(
                index,
                label=label,
                weights_hash=f"weights-hash-{index}",
            ),
            expected_state_revision=runtime_state.state_revision,
            operator_id="operator-test",
            confirmation=True,
        )

    policy = ledger.rollout_rehearsal_promotion_policy(candidate_limit=8)
    candidate_labels = {
        str(target.get("selected_label") or "")
        for candidate in policy["candidates"]
        for target in list(candidate.get("replay_targets") or [])
    }

    assert policy["candidate_count"] == 8
    assert policy["source_window"]["source_event_retention_count"] == (
        SNN_ROLLOUT_REHEARSAL_SOURCE_WINDOW_LIMIT * 2
    )
    assert policy["source_window"]["source_event_window_count"] == (
        SNN_ROLLOUT_REHEARSAL_SOURCE_WINDOW_LIMIT
    )
    assert policy["source_window"]["source_event_truncated_count"] == (
        SNN_ROLLOUT_REHEARSAL_SOURCE_WINDOW_LIMIT
    )
    assert policy["source_window"]["candidate_count_before_rank"] == (
        SNN_ROLLOUT_REHEARSAL_SOURCE_WINDOW_LIMIT
    )
    assert policy["source_window"]["candidate_count_returned"] == 8
    assert policy["source_window"]["global_candidate_scan"] is False
    assert policy["source_window"]["global_score_scan"] is False
    assert policy["source_window"]["runs_live_tick"] is False
    assert policy["source_window"]["gpu_used"] is False
    assert "outside-rollout-window" not in candidate_labels


def test_readout_ledger_rollout_rehearsal_promotion_policy_blocks_tampered_or_fallback_records() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=_ready_rollout_replay_evaluation(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    baseline_state = deepcopy(ledger_state)

    ledger_state["rollout_events"][0]["replay_targets"][0]["predicted_sparse_indices"] = [4]
    tampered = ledger.rollout_rehearsal_promotion_policy(candidate_limit=4)
    ledger_state.clear()
    ledger_state.update(deepcopy(baseline_state))
    ledger_state["rollout_events"][0]["server_transition_memory_hash"] = "tampered"
    tampered_server_hash = ledger.rollout_rehearsal_promotion_policy(candidate_limit=4)
    ledger_state.clear()
    ledger_state.update(deepcopy(baseline_state))
    ledger_state["rollout_events"][0]["server_transition_memory_hash_match"] = False
    mismatched_server_hash = ledger.rollout_rehearsal_promotion_policy(candidate_limit=4)
    ledger_state.clear()
    ledger_state.update(deepcopy(baseline_state))
    ledger_state["rollout_events"][0]["device_evidence"] = {}
    missing_device = ledger.rollout_rehearsal_promotion_policy(candidate_limit=4)
    ledger_state.clear()
    ledger_state.update(deepcopy(baseline_state))
    ledger_state["rollout_events"][0]["device_evidence"] = {
        "requested_device": "cuda:0",
        "tensor_device": "cpu",
        "cuda_tensor": False,
        "device_source": "fallback-test",
    }
    cuda_fallback = ledger.rollout_rehearsal_promotion_policy(candidate_limit=4)

    assert tampered["candidate_count"] == 0
    assert tampered["promotion_gate"]["eligible_for_operator_rollout_rehearsal_review"] is False
    assert tampered_server_hash["candidate_count"] == 0
    assert tampered_server_hash["promotion_gate"]["eligible_for_operator_rollout_rehearsal_review"] is False
    assert mismatched_server_hash["candidate_count"] == 0
    assert mismatched_server_hash["promotion_gate"]["eligible_for_operator_rollout_rehearsal_review"] is False
    assert missing_device["candidate_count"] == 0
    assert missing_device["promotion_gate"]["eligible_for_operator_rollout_rehearsal_review"] is False
    assert cuda_fallback["candidate_count"] == 0
    assert cuda_fallback["promotion_gate"]["eligible_for_operator_rollout_rehearsal_review"] is False


def test_readout_ledger_rollout_rehearsal_evaluation_runs_ephemeral_sparse_temporal_review() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=_ready_rollout_replay_evaluation(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    policy = ledger.rollout_rehearsal_promotion_policy(candidate_limit=4)
    before = runtime_state.snapshot()

    evaluation = ledger.rollout_rehearsal_evaluation(policy, candidate_limit=4)
    empty = ledger.rollout_rehearsal_evaluation(
        ledger.rollout_rehearsal_promotion_policy(candidate_limit=0),
        candidate_limit=4,
    )
    after = runtime_state.snapshot()

    assert before == after
    assert evaluation["surface"] == "snn_language_readout_rollout_rehearsal_evaluation.v1"
    assert evaluation["generates_text"] is False
    assert evaluation["decodes_text"] is False
    assert evaluation["trains_runtime_model"] is False
    assert evaluation["applies_plasticity"] is False
    assert evaluation["mutates_runtime_state"] is False
    assert evaluation["returns_trained_weights"] is False
    assert evaluation["rehearsal_summary"]["candidate_count"] == 1
    assert evaluation["rehearsal_summary"]["step_count"] == 2
    assert evaluation["rehearsal_summary"]["activation_sparsity"] > 0.9
    assert evaluation["rehearsal_summary"]["sparse_temporal_rehearsal_stable"] is True
    assert evaluation["ephemeral_rehearsal"]["runtime_update_applied"] is False
    assert evaluation["ephemeral_rehearsal"]["weights_persisted"] is False
    assert evaluation["ephemeral_rehearsal"]["checkpoint_written"] is False
    trace = evaluation["ephemeral_rehearsal"]["trace"][0]
    assert trace["device_evidence"]["tensor_device"] == "cpu"
    assert trace["device_evidence"]["actual_device_match"] is True
    assert trace["device_evidence"]["requested_cuda_honored"] is True
    assert trace["step_trace"][0]["active_indices_hash_valid"] is True
    assert evaluation["promotion_gate"]["eligible_for_operator_rollout_rehearsal_review"] is True
    assert evaluation["promotion_gate"]["eligible_for_live_replay"] is False
    assert evaluation["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert empty["promotion_gate"]["eligible_for_operator_rollout_rehearsal_review"] is False


def test_readout_ledger_rollout_rehearsal_experiment_repeats_ephemeral_cycles_without_mutation() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=_ready_rollout_replay_evaluation(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    evaluation = ledger.rollout_rehearsal_evaluation(
        ledger.rollout_rehearsal_promotion_policy(candidate_limit=4),
        candidate_limit=4,
    )
    before = runtime_state.snapshot()

    experiment = ledger.rollout_rehearsal_experiment(
        evaluation,
        replay_cycles=4,
        stability_floor=0.95,
    )
    repeat = ledger.rollout_rehearsal_experiment(
        evaluation,
        replay_cycles=4,
        stability_floor=0.95,
    )
    blocked = ledger.rollout_rehearsal_experiment({"surface": "wrong.v1"})
    after = runtime_state.snapshot()

    assert before == after
    assert experiment == repeat
    assert experiment["surface"] == "snn_language_readout_rollout_rehearsal_experiment.v1"
    assert experiment["generates_text"] is False
    assert experiment["decodes_text"] is False
    assert experiment["trains_runtime_model"] is False
    assert experiment["applies_plasticity"] is False
    assert experiment["mutates_runtime_state"] is False
    assert experiment["returns_trained_weights"] is False
    assert experiment["experiment_summary"]["replay_cycles"] == 4
    assert experiment["experiment_summary"]["minimum_cycle_stability"] == 1.0
    assert experiment["experiment_summary"]["mean_cycle_stability"] == 1.0
    assert experiment["experiment_summary"]["cycle_drift"] == 0.0
    assert experiment["experiment_summary"]["stable"] is True
    assert len(experiment["ephemeral_experiment"]["trace"]) == 4
    assert experiment["ephemeral_experiment"]["runtime_update_applied"] is False
    assert experiment["ephemeral_experiment"]["weights_persisted"] is False
    assert experiment["ephemeral_experiment"]["checkpoint_written"] is False
    assert experiment["ephemeral_experiment"]["plasticity_applied"] is False
    assert experiment["promotion_gate"]["eligible_for_operator_rollout_rehearsal_experiment_review"] is True
    assert experiment["promotion_gate"]["eligible_for_live_replay"] is False
    assert experiment["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert blocked["promotion_gate"]["eligible_for_operator_rollout_rehearsal_experiment_review"] is False


def test_readout_ledger_rollout_consolidation_design_proposes_bounded_local_updates_without_writes() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=_ready_rollout_replay_evaluation(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    evaluation = ledger.rollout_rehearsal_evaluation(
        ledger.rollout_rehearsal_promotion_policy(candidate_limit=4),
        candidate_limit=4,
    )
    experiment = ledger.rollout_rehearsal_experiment(evaluation, replay_cycles=4)
    assert experiment["experiment_summary"]["sparse_transition_candidate_count"] == 1
    before = runtime_state.snapshot()

    design = ledger.rollout_consolidation_design(
        experiment,
        consolidation_policy={
            "learning_rate": 0.02,
            "max_weight_delta": 0.04,
            "homeostatic_decay": 0.01,
            "local_only": True,
            "normalization": True,
        },
        rollback_policy={"available": True, "snapshot_id": "rollout-snapshot-1"},
    )
    repeat = ledger.rollout_consolidation_design(
        experiment,
        consolidation_policy={
            "learning_rate": 0.02,
            "max_weight_delta": 0.04,
            "homeostatic_decay": 0.01,
            "local_only": True,
            "normalization": True,
        },
        rollback_policy={"available": True, "snapshot_id": "rollout-snapshot-1"},
    )
    blocked = ledger.rollout_consolidation_design(experiment)
    oversized_experiment = deepcopy(experiment)
    oversized_experiment["ephemeral_experiment"]["sparse_transition_candidates"] = (
        _rollout_sparse_transition_candidates(
            SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT + 1
        )
    )
    oversized_design = ledger.rollout_consolidation_design(
        oversized_experiment,
        consolidation_policy={
            "learning_rate": 0.02,
            "max_weight_delta": 0.04,
            "homeostatic_decay": 0.01,
            "local_only": True,
            "normalization": True,
        },
        rollback_policy={"available": True, "snapshot_id": "rollout-snapshot-1"},
    )
    after = runtime_state.snapshot()

    assert before == after
    assert design == repeat
    assert design["surface"] == "snn_language_readout_rollout_consolidation_design.v1"
    assert design["generates_text"] is False
    assert design["decodes_text"] is False
    assert design["trains_runtime_model"] is False
    assert design["applies_plasticity"] is False
    assert design["mutates_runtime_state"] is False
    assert design["returns_trained_weights"] is False
    proposal = design["rollout_consolidation_design"]
    assert proposal["candidate_synapse_count"] == 1
    assert proposal["candidate_synapses"][0]["local_only"] is True
    assert proposal["candidate_synapses"][0]["source_neuron_index"] == 1
    assert proposal["candidate_synapses"][0]["target_neuron_index"] == 4
    assert proposal["candidate_synapses"][0]["source_step_index"] == 1
    assert proposal["candidate_synapses"][0]["target_step_index"] == 4
    assert proposal["candidate_synapses"][0]["source_rollout_step_index"] == 0
    assert proposal["candidate_synapses"][0]["target_rollout_step_index"] == 1
    assert proposal["candidate_synapses"][0]["source_active_indices_hash"] == _sha256_json([1, 2, 3])
    assert proposal["candidate_synapses"][0]["target_active_indices_hash"] == _sha256_json([2, 3, 4])
    assert proposal["candidate_synapses"][0]["proposed_weight_delta"] == 0.02
    assert proposal["candidate_synapses"][0]["homeostatic_decay"] == 0.01
    assert proposal["runtime_update_applied"] is False
    assert proposal["weights_persisted"] is False
    assert proposal["checkpoint_written"] is False
    assert proposal["structural_write_applied"] is False
    assert design["sparse_candidate_source_window"]["surface"] == (
        "bounded_snn_rollout_consolidation_design_sparse_candidate_window.v1"
    )
    assert design["sparse_candidate_source_window"]["source_window_count"] == 1
    assert design["promotion_gate"]["required_evidence"]["sparse_candidate_source_window_bounded"] is True
    assert design["promotion_gate"]["required_evidence"]["sparse_candidate_payload_not_truncated"] is True
    assert design["promotion_gate"]["eligible_for_operator_rollout_consolidation_design_review"] is True
    assert design["promotion_gate"]["eligible_for_structural_write"] is False
    assert design["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert blocked["promotion_gate"]["required_evidence"]["rollback_policy_available"] is False
    assert blocked["promotion_gate"]["eligible_for_operator_rollout_consolidation_design_review"] is False
    assert oversized_design["promotion_gate"]["required_evidence"][
        "sparse_candidate_source_window_bounded"
    ] is True
    assert oversized_design["promotion_gate"]["required_evidence"][
        "sparse_candidate_payload_not_truncated"
    ] is False
    assert oversized_design["sparse_candidate_source_window"]["source_window_count"] == (
        SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
    )
    assert oversized_design["sparse_candidate_source_window"]["source_payload_truncated"] is True
    assert oversized_design["promotion_gate"][
        "eligible_for_operator_rollout_consolidation_design_review"
    ] is False


def test_rollout_sparse_transition_candidates_use_neuron_indices_not_step_numbers() -> None:
    candidates = SNNLanguageReadoutEvidenceLedger._rollout_sparse_transition_candidates(
        [
            {
                "step_trace": [
                    {
                        "step_index": 10,
                        "sparse_active_indices": [1, 2, 3],
                        "active_indices_hash": _sha256_json([1, 2, 3]),
                    },
                    {
                        "step_index": 20,
                        "sparse_active_indices": [2, 3, 4],
                        "active_indices_hash": _sha256_json([2, 3, 4]),
                    },
                ]
            }
        ]
    )

    assert len(candidates) == 1
    assert candidates[0]["source_index"] == 1
    assert candidates[0]["target_index"] == 4
    assert candidates[0]["source_step_index"] == 10
    assert candidates[0]["target_step_index"] == 20


def test_readout_ledger_rollout_consolidation_shadow_delta_materializes_ephemeral_tensor_without_writes() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=_ready_rollout_replay_evaluation(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    evaluation = ledger.rollout_rehearsal_evaluation(
        ledger.rollout_rehearsal_promotion_policy(candidate_limit=4),
        candidate_limit=4,
    )
    experiment = ledger.rollout_rehearsal_experiment(evaluation, replay_cycles=4)
    design = ledger.rollout_consolidation_design(
        experiment,
        rollback_policy={"available": True, "snapshot_id": "rollout-snapshot-1"},
    )
    before = runtime_state.snapshot()

    shadow = ledger.rollout_consolidation_shadow_delta(
        design,
        device_evidence={"device": "cpu", "source": "test"},
    )
    first_synapse = _first_rollout_synapse(design)
    repeat = ledger.rollout_consolidation_shadow_delta(
        design,
        device_evidence={"device": "cpu", "source": "test"},
    )
    cuda_fallback = ledger.rollout_consolidation_shadow_delta(
        design,
        device_evidence={"device": "cuda:0", "source": "test"},
    )
    invalid_design = deepcopy(design)
    invalid_design["rollout_consolidation_design"]["candidate_synapses"][0][
        "source_neuron_index"
    ] = -1
    invalid_coordinate = ledger.rollout_consolidation_shadow_delta(
        invalid_design,
        device_evidence={"device": "cpu", "source": "test"},
    )
    oversized_design = deepcopy(design)
    oversized_design["rollout_consolidation_design"]["candidate_synapses"] = (
        _rollout_design_candidates(SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT + 1)
    )
    oversized_design["rollout_consolidation_design"]["candidate_synapse_count"] = (
        SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT + 1
    )
    oversized_shadow = ledger.rollout_consolidation_shadow_delta(
        oversized_design,
        device_evidence={"device": "cpu", "source": "test"},
    )
    after = runtime_state.snapshot()

    assert before == after
    assert shadow == repeat
    assert shadow["surface"] == "snn_language_readout_rollout_consolidation_shadow_delta.v1"
    assert shadow["generates_text"] is False
    assert shadow["decodes_text"] is False
    assert shadow["trains_runtime_model"] is False
    assert shadow["applies_plasticity"] is False
    assert shadow["mutates_runtime_state"] is False
    assert shadow["returns_trained_weights"] is False
    assert shadow["affected_synapse_count"] == 1
    assert abs(shadow["max_abs_delta"] - 0.02) < 1e-6
    assert shadow["device_evidence"]["tensor_device"] == "cpu"
    assert shadow["shadow_delta"]["tensor_shape"] == [64, 64]
    assert shadow["shadow_delta"]["nonzero_count"] == 1
    assert shadow["shadow_delta"]["runtime_update_applied"] is False
    assert shadow["shadow_delta"]["weights_persisted"] is False
    assert shadow["shadow_delta"]["checkpoint_written"] is False
    assert shadow["shadow_delta"]["structural_write_applied"] is False
    assert shadow["candidate_source_window"]["surface"] == (
        "bounded_snn_rollout_consolidation_shadow_delta_candidate_window.v1"
    )
    assert shadow["candidate_source_window"]["source_window_count"] == 1
    assert shadow["promotion_gate"]["required_evidence"]["candidate_source_window_bounded"] is True
    assert shadow["promotion_gate"]["required_evidence"]["candidate_payload_not_truncated"] is True
    assert shadow["promotion_gate"]["eligible_for_operator_rollout_consolidation_shadow_review"] is True
    assert shadow["promotion_gate"]["eligible_for_shadow_application"] is False
    assert shadow["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert invalid_coordinate["promotion_gate"]["required_evidence"][
        "candidate_coordinates_canonical"
    ] is False
    assert invalid_coordinate["promotion_gate"][
        "eligible_for_operator_rollout_consolidation_shadow_review"
    ] is False
    assert oversized_shadow["affected_synapse_count"] == (
        SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
    )
    assert oversized_shadow["promotion_gate"]["required_evidence"][
        "candidate_source_window_bounded"
    ] is True
    assert oversized_shadow["promotion_gate"]["required_evidence"][
        "candidate_payload_not_truncated"
    ] is False
    assert oversized_shadow["candidate_source_window"]["source_payload_truncated"] is True
    assert oversized_shadow["promotion_gate"][
        "eligible_for_operator_rollout_consolidation_shadow_review"
    ] is False
    if cuda_fallback["device_evidence"]["tensor_device"] == "cpu":
        assert cuda_fallback["device_evidence"]["requested_cuda_honored"] is False
        assert cuda_fallback["promotion_gate"]["eligible_for_operator_rollout_consolidation_shadow_review"] is False


def test_readout_ledger_rollout_consolidation_shadow_application_preflight_verifies_integrity_without_writes() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=_ready_rollout_replay_evaluation(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    evaluation = ledger.rollout_rehearsal_evaluation(
        ledger.rollout_rehearsal_promotion_policy(candidate_limit=4),
        candidate_limit=4,
    )
    experiment = ledger.rollout_rehearsal_experiment(evaluation, replay_cycles=4)
    design = ledger.rollout_consolidation_design(
        experiment,
        rollback_policy={"available": True, "snapshot_id": "rollout-snapshot-1"},
    )
    shadow = ledger.rollout_consolidation_shadow_delta(
        design,
        device_evidence={"device": "cpu", "source": "test"},
    )
    first_synapse = _first_rollout_synapse(design)
    before = runtime_state.snapshot()

    preflight = ledger.rollout_consolidation_shadow_application_preflight(
        design,
        shadow,
        transition_memory_state={
            "sparse_transition_weights": {first_synapse: 0.1},
            "language_capacity": _language_capacity(),
        },
    )
    repeat = ledger.rollout_consolidation_shadow_application_preflight(
        design,
        shadow,
        transition_memory_state={
            "sparse_transition_weights": {first_synapse: 0.1},
            "language_capacity": _language_capacity(),
        },
    )
    growth = ledger.rollout_consolidation_shadow_application_preflight(
        design,
        shadow,
        transition_memory_state={
            "sparse_transition_weights": {},
            "language_capacity": _language_capacity(sparse_edge_budget=512, capacity_expansion_count=1),
        },
    )
    tampered_shadow = deepcopy(shadow)
    tampered_shadow["bounded_synapses"][0]["target_index"] = 63
    tampered = ledger.rollout_consolidation_shadow_application_preflight(
        design,
        tampered_shadow,
        transition_memory_state={
            "sparse_transition_weights": {first_synapse: 0.1},
            "language_capacity": _language_capacity(),
        },
    )
    tampered_design = deepcopy(design)
    tampered_design["rollout_consolidation_design_material"]["max_weight_delta"] = 0.25
    design_tampered = ledger.rollout_consolidation_shadow_application_preflight(
        tampered_design,
        shadow,
        transition_memory_state={
            "sparse_transition_weights": {first_synapse: 0.1},
            "language_capacity": _language_capacity(),
        },
    )
    duplicate_shadow = deepcopy(shadow)
    duplicate_shadow["bounded_synapses"].append(deepcopy(duplicate_shadow["bounded_synapses"][0]))
    duplicate_shadow["affected_synapse_count"] = 2
    duplicate_shadow["shadow_delta"]["nonzero_count"] = 1
    duplicate = ledger.rollout_consolidation_shadow_application_preflight(
        design,
        duplicate_shadow,
        transition_memory_state={
            "sparse_transition_weights": {first_synapse: 0.1},
            "language_capacity": _language_capacity(),
        },
    )
    after = runtime_state.snapshot()

    assert before == after
    assert preflight == repeat
    assert (
        preflight["surface"]
        == "snn_language_readout_rollout_consolidation_shadow_application_preflight.v1"
    )
    assert preflight["generates_text"] is False
    assert preflight["decodes_text"] is False
    assert preflight["trains_runtime_model"] is False
    assert preflight["applies_plasticity"] is False
    assert preflight["mutates_runtime_state"] is False
    assert preflight["returns_trained_weights"] is False
    assert preflight["preflight_summary"]["affected_synapse_count"] == 1
    assert abs(preflight["preflight_summary"]["max_abs_delta"] - 0.02) < 1e-6
    assert preflight["preflight_summary"]["functional_weight_write_applied"] is False
    assert preflight["preflight_summary"]["structural_growth_applied"] is False
    assert preflight["preflight_summary"]["structural_pruning_applied"] is False
    assert preflight["promotion_gate"][
        "eligible_for_operator_rollout_consolidation_shadow_application_preflight_review"
    ] is True
    assert preflight["promotion_gate"]["eligible_for_shadow_application"] is False
    assert preflight["promotion_gate"]["eligible_for_structural_write"] is False
    assert preflight["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert preflight["topology_evidence"]["topology_keyset_unchanged"] is True
    assert preflight["topology_evidence"]["growth_candidate_count"] == 0
    assert preflight["topology_evidence"]["sparse_edge_budget"] == 256
    assert preflight["topology_evidence"]["language_capacity"]["sparse_edge_budget"] == 256
    assert (
        preflight["promotion_gate"]["required_evidence"][
            "language_capacity_state_available"
        ]
        is True
    )
    assert (
        preflight["promotion_gate"]["required_evidence"][
            "language_capacity_state_dynamic_limits_applied"
        ]
        is True
    )
    assert preflight["rollback_evidence"]["checkpoint_restore_verified"] is False
    assert growth["topology_evidence"]["sparse_edge_budget"] == 512
    assert growth["topology_evidence"]["language_capacity"]["sparse_edge_budget"] == 512
    assert growth["promotion_gate"]["required_evidence"]["topology_keyset_unchanged"] is False
    assert growth["promotion_gate"]["eligible_for_structural_growth_review"] is True
    assert growth["promotion_gate"][
        "eligible_for_operator_rollout_consolidation_shadow_application_preflight_review"
    ] is False
    assert tampered["promotion_gate"]["required_evidence"][
        "rollout_consolidation_shadow_delta_hash_matches"
    ] is False
    assert design_tampered["promotion_gate"]["required_evidence"][
        "rollout_consolidation_design_hash_matches"
    ] is False
    assert duplicate["promotion_gate"]["required_evidence"]["synapse_coordinates_unique"] is False


def test_readout_ledger_rollout_developmental_plasticity_review_routes_growth_without_writes() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=_ready_rollout_replay_evaluation(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    evaluation = ledger.rollout_rehearsal_evaluation(
        ledger.rollout_rehearsal_promotion_policy(candidate_limit=4),
        candidate_limit=4,
    )
    experiment = ledger.rollout_rehearsal_experiment(evaluation, replay_cycles=4)
    design = ledger.rollout_consolidation_design(
        experiment,
        rollback_policy={"available": True, "snapshot_id": "rollout-snapshot-1"},
    )
    shadow = ledger.rollout_consolidation_shadow_delta(
        design,
        device_evidence={"device": "cpu", "source": "test"},
    )
    first_synapse = _first_rollout_synapse(design)
    expanded_capacity = _language_capacity(
        sparse_edge_budget=512,
        capacity_expansion_count=1,
    )
    growth_preflight = ledger.rollout_consolidation_shadow_application_preflight(
        design,
        shadow,
        transition_memory_state={
            "sparse_transition_weights": {},
            "language_capacity": expanded_capacity,
        },
    )
    existing_preflight = ledger.rollout_consolidation_shadow_application_preflight(
        design,
        shadow,
        transition_memory_state={
            "sparse_transition_weights": {first_synapse: 0.1},
            "language_capacity": expanded_capacity,
        },
    )
    before = runtime_state.snapshot()

    review = ledger.rollout_developmental_plasticity_review(
        design,
        growth_preflight,
        transition_memory_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "sparse_transition_weights": {},
            "language_capacity": expanded_capacity,
        },
    )
    repeat = ledger.rollout_developmental_plasticity_review(
        design,
        growth_preflight,
        transition_memory_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "sparse_transition_weights": {},
            "language_capacity": expanded_capacity,
        },
    )
    no_growth = ledger.rollout_developmental_plasticity_review(
        design,
        existing_preflight,
        transition_memory_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "sparse_transition_weights": {first_synapse: 0.1},
            "language_capacity": expanded_capacity,
        },
    )
    tampered_design = deepcopy(design)
    tampered_design["rollout_consolidation_design_material"]["learning_rate"] = 0.25
    design_tampered = ledger.rollout_developmental_plasticity_review(
        tampered_design,
        growth_preflight,
        transition_memory_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "sparse_transition_weights": {},
            "language_capacity": expanded_capacity,
        },
    )
    tampered_preflight = deepcopy(growth_preflight)
    tampered_preflight["topology_evidence"]["growth_candidate_count"] = 9
    preflight_tampered = ledger.rollout_developmental_plasticity_review(
        design,
        tampered_preflight,
        transition_memory_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "sparse_transition_weights": {},
            "language_capacity": expanded_capacity,
        },
    )
    signed_preflight_tampered = deepcopy(growth_preflight)
    signed_preflight_tampered["promotion_gate"]["required_evidence"][
        "topology_keyset_unchanged"
    ] = True
    signed_tampered = ledger.rollout_developmental_plasticity_review(
        design,
        signed_preflight_tampered,
        transition_memory_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "sparse_transition_weights": {},
            "language_capacity": expanded_capacity,
        },
    )
    missing_hash_design = deepcopy(design)
    missing_hash_design["rollout_consolidation_design"]["candidate_synapses"][0][
        "source_active_indices_hash"
    ] = ""
    missing_hash_design["rollout_consolidation_design_material"][
        "candidate_synapses"
    ] = missing_hash_design["rollout_consolidation_design"]["candidate_synapses"]
    missing_hash_design["rollout_consolidation_design_material"][
        "sparse_transition_candidates"
    ][0]["source_active_indices_hash"] = ""
    missing_hash = ledger.rollout_developmental_plasticity_review(
        missing_hash_design,
        growth_preflight,
        transition_memory_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "sparse_transition_weights": {},
            "language_capacity": expanded_capacity,
        },
    )
    oversized_design = deepcopy(design)
    oversized_design["rollout_consolidation_design"]["candidate_synapses"] = (
        _rollout_design_candidates(SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT + 1)
    )
    oversized_review = ledger.rollout_developmental_plasticity_review(
        oversized_design,
        growth_preflight,
        transition_memory_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "sparse_transition_weights": {},
            "language_capacity": expanded_capacity,
        },
    )
    after = runtime_state.snapshot()

    assert before == after
    assert review == repeat
    assert review["surface"] == "snn_language_readout_rollout_developmental_plasticity_review.v1"
    assert review["generates_text"] is False
    assert review["decodes_text"] is False
    assert review["trains_runtime_model"] is False
    assert review["applies_plasticity"] is False
    assert review["mutates_runtime_state"] is False
    assert review["returns_trained_weights"] is False
    assert review["runtime_memory_evidence"]["language_capacity"]["sparse_edge_budget"] == 512
    assert review["topology_budget_evidence"]["sparse_edge_budget"] == 512
    assert review["promotion_gate"]["required_evidence"]["language_capacity_state_available"] is True
    assert review["promotion_gate"]["required_evidence"]["language_capacity_state_dynamic_limits_applied"] is True
    assert review["promotion_gate"]["required_evidence"]["topology_sparse_edge_budget_matches_capacity"] is True
    assert review["developmental_plasticity_review"]["growth_candidate_count"] == 1
    assert review["developmental_plasticity_review"]["growth_candidates"][0]["synapse"] == first_synapse
    growth_candidate = review["developmental_plasticity_review"]["growth_candidates"][0]
    assert growth_candidate["source_rollout_step_index"] == 0
    assert growth_candidate["target_rollout_step_index"] == 1
    assert growth_candidate["source_active_indices_hash"] == _sha256_json([1, 2, 3])
    assert growth_candidate["target_active_indices_hash"] == _sha256_json([2, 3, 4])
    assert review["developmental_plasticity_review"]["structural_growth_applied"] is False
    assert review["developmental_plasticity_review"]["structural_pruning_applied"] is False
    assert review["integrity_evidence"]["design_hash_recomputed_match"] is True
    assert review["integrity_evidence"]["preflight_hash_recomputed_match"] is True
    assert review["candidate_source_window"]["surface"] == (
        "bounded_snn_rollout_developmental_plasticity_candidate_window.v1"
    )
    assert review["candidate_source_window"]["source_window_count"] == 1
    assert review["integrity_evidence"]["candidate_source_window_bounded"] is True
    assert review["integrity_evidence"]["candidate_payload_not_truncated"] is True
    assert review["integrity_evidence"]["candidate_rollout_step_provenance_available"] is True
    assert review["integrity_evidence"]["candidate_active_hash_provenance_available"] is True
    assert review["runtime_memory_evidence"]["candidate_synapses_absent_from_runtime"] is True
    assert review["growth_classification"]["all_candidates_are_growth"] is True
    assert review["topology_budget_evidence"]["candidate_count_bounded"] is True
    assert review["topology_budget_evidence"]["structural_mutation_ledger_write_applied"] is False
    assert "permit_id" not in review["developmental_plasticity_review"]
    assert review["promotion_gate"]["eligible_for_operator_rollout_developmental_plasticity_review"] is True
    assert review["promotion_gate"]["eligible_for_transition_memory_regeneration_proposal"] is True
    assert review["promotion_gate"]["eligible_for_structural_write"] is False
    assert review["promotion_gate"]["eligible_for_growth"] is False
    assert review["promotion_gate"]["eligible_for_pruning"] is False
    assert review["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert no_growth["promotion_gate"]["required_evidence"]["growth_pressure_available"] is False
    assert no_growth["promotion_gate"]["eligible_for_operator_rollout_developmental_plasticity_review"] is False
    assert design_tampered["integrity_evidence"]["design_hash_recomputed_match"] is False
    assert design_tampered["promotion_gate"][
        "eligible_for_operator_rollout_developmental_plasticity_review"
    ] is False
    assert preflight_tampered["promotion_gate"]["required_evidence"][
        "growth_candidate_count_matches_preflight"
    ] is False
    assert signed_tampered["integrity_evidence"]["preflight_hash_recomputed_match"] is False
    assert missing_hash["integrity_evidence"][
        "candidate_active_hash_provenance_available"
    ] is False
    assert missing_hash["promotion_gate"][
        "eligible_for_operator_rollout_developmental_plasticity_review"
    ] is False
    assert oversized_review["integrity_evidence"]["candidate_source_window_bounded"] is True
    assert oversized_review["integrity_evidence"]["candidate_payload_not_truncated"] is False
    assert oversized_review["candidate_source_window"]["source_window_count"] == (
        SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
    )
    assert oversized_review["candidate_source_window"]["source_payload_truncated"] is True
    assert oversized_review["promotion_gate"][
        "eligible_for_operator_rollout_developmental_plasticity_review"
    ] is False


def test_readout_ledger_rollout_regeneration_proposal_adapter_exports_design_without_permit() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=_ready_rollout_replay_evaluation(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    evaluation = ledger.rollout_rehearsal_evaluation(
        ledger.rollout_rehearsal_promotion_policy(candidate_limit=4),
        candidate_limit=4,
    )
    experiment = ledger.rollout_rehearsal_experiment(evaluation, replay_cycles=4)
    design = ledger.rollout_consolidation_design(
        experiment,
        rollback_policy={"available": True, "snapshot_id": "rollout-snapshot-1"},
    )
    shadow = ledger.rollout_consolidation_shadow_delta(
        design,
        device_evidence={"device": "cpu", "source": "test"},
    )
    growth_preflight = ledger.rollout_consolidation_shadow_application_preflight(
        design,
        shadow,
        transition_memory_state={
            "sparse_transition_weights": {},
            "language_capacity": _language_capacity(),
        },
    )
    review = ledger.rollout_developmental_plasticity_review(
        design,
        growth_preflight,
        transition_memory_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "sparse_transition_weights": {},
            "language_capacity": _language_capacity(),
        },
    )
    blocked_review = deepcopy(review)
    blocked_review["promotion_gate"][
        "eligible_for_operator_rollout_developmental_plasticity_review"
    ] = False
    before = runtime_state.snapshot()

    adapter = ledger.rollout_regeneration_proposal_adapter(review)
    repeat = ledger.rollout_regeneration_proposal_adapter(review)
    blocked = ledger.rollout_regeneration_proposal_adapter(blocked_review)
    oversized_review = deepcopy(review)
    oversized_review["developmental_plasticity_review"]["growth_candidate_count"] = (
        SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT + 1
    )
    oversized_review["developmental_plasticity_review"]["growth_candidates"] = (
        _rollout_growth_candidates(SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT + 1)
    )
    oversized_adapter = ledger.rollout_regeneration_proposal_adapter(oversized_review)
    checkpoint_calls = []
    language_state = {"sparse_transition_weights": {}}
    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=lambda path: checkpoint_calls.append(path) or {"path": "should-not-write.pt"},
        checkpoint_path=lambda: "should-not-write.pt",
        verify_checkpoint=lambda _path: False,
    )
    direct_executor_result = executor.regenerate_transition_memory(
        regeneration_proposal=adapter,
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    after = runtime_state.snapshot()

    assert before == after
    assert checkpoint_calls == []
    assert direct_executor_result["accepted"] is False
    assert direct_executor_result["promotion_gate"]["required_evidence"][
        "replay_permit_server_verified"
    ] is False
    assert adapter == repeat
    assert adapter["surface"] == "snn_language_readout_rollout_regeneration_proposal_adapter.v1"
    assert adapter["generates_text"] is False
    assert adapter["applies_plasticity"] is False
    assert adapter["mutates_runtime_state"] is False
    assert adapter["issues_regeneration_permit"] is False
    assert adapter["executor_ready"] is False
    assert adapter["regeneration_design"]["mismatch_score"] == 0.0
    assert adapter["regeneration_design"]["candidate_count"] == 1
    assert adapter["regeneration_design"]["candidate_synapses"][0]["synapse"] == _first_rollout_synapse(design)
    assert adapter["growth_candidate_source_window"]["surface"] == (
        "bounded_snn_rollout_regeneration_adapter_growth_candidate_window.v1"
    )
    assert adapter["growth_candidate_source_window"]["source_window_count"] == 1
    assert adapter["regeneration_design"]["growth_candidate_source_window"] == (
        adapter["growth_candidate_source_window"]
    )
    adapter_candidate = adapter["regeneration_design"]["candidate_synapses"][0]
    assert adapter_candidate["source_rollout_step_index"] == 0
    assert adapter_candidate["target_rollout_step_index"] == 1
    assert adapter_candidate["source_active_indices_hash"] == _sha256_json([1, 2, 3])
    assert adapter_candidate["target_active_indices_hash"] == _sha256_json([2, 3, 4])
    assert adapter["integrity_evidence"]["candidate_rollout_step_provenance_available"] is True
    assert adapter["integrity_evidence"]["candidate_active_hash_provenance_available"] is True
    assert adapter["integrity_evidence"]["growth_candidate_source_window_bounded"] is True
    assert adapter["integrity_evidence"]["growth_candidate_payload_not_truncated"] is True
    assert adapter["integrity_evidence"]["candidate_payload_not_truncated"] is True
    assert adapter["blocked_replay_evidence"]["ready"] is False
    assert adapter["blocked_replay_evidence"]["permit_id"] is None
    assert adapter["executor_bypass_evidence"]["is_transition_memory_regeneration_proposal"] is False
    assert adapter["executor_bypass_evidence"]["is_regeneration_permit"] is False
    assert adapter["executor_bypass_evidence"]["direct_executor_submission_expected_to_block"] is True
    assert adapter["promotion_gate"]["eligible_for_operator_rollout_regeneration_adapter_review"] is True
    assert adapter["promotion_gate"]["eligible_for_replay_artifact_recording_review"] is True
    assert adapter["promotion_gate"]["eligible_for_regeneration_permit_request"] is False
    assert adapter["promotion_gate"]["eligible_for_regeneration_application"] is False
    assert adapter["promotion_gate"]["eligible_for_structural_write"] is False
    assert "replay_evidence" not in adapter
    assert "permit_id" not in adapter
    assert blocked["promotion_gate"]["required_evidence"]["review_gate_ready"] is False
    assert blocked["promotion_gate"]["eligible_for_operator_rollout_regeneration_adapter_review"] is False
    assert oversized_adapter["integrity_evidence"]["growth_candidate_source_window_bounded"] is True
    assert oversized_adapter["integrity_evidence"]["growth_candidate_payload_not_truncated"] is False
    assert oversized_adapter["integrity_evidence"]["candidate_payload_not_truncated"] is False
    assert oversized_adapter["growth_candidate_source_window"]["source_window_count"] == (
        SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
    )
    assert oversized_adapter["growth_candidate_source_window"]["source_payload_truncated"] is True
    assert oversized_adapter["promotion_gate"][
        "eligible_for_operator_rollout_regeneration_adapter_review"
    ] is False


def test_readout_ledger_rollout_regeneration_adapter_uses_capacity_state_indices() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    growth_candidate = {
        "synapse": "65:66",
        "pre_index": 65,
        "post_index": 66,
        "initial_weight": 0.1,
        "locality_distance": 1,
        "source_synapse_id": "snn-rollout-local:65:66:0",
        "source_trace_index": 0,
        "source_rollout_step_index": 10,
        "target_rollout_step_index": 11,
        "source_active_indices_hash": "source-active-hash-65",
        "target_active_indices_hash": "target-active-hash-66",
        "local_only": True,
        "normalization": True,
        "applied_to_runtime": False,
    }
    transition_memory_snapshot_hash = _sha256_json({})
    review_hash = _sha256_json(
        {
            "rollout_consolidation_design_hash": "design-hash-1",
            "rollout_consolidation_shadow_application_preflight_hash": (
                "preflight-hash-1"
            ),
            "transition_memory_snapshot_hash": transition_memory_snapshot_hash,
            "growth_candidates": [growth_candidate],
            "candidate_source_window": {},
        }
    )
    review = {
        "surface": "snn_language_readout_rollout_developmental_plasticity_review.v1",
        "artifact_kind": "terminus_snn_language_readout_rollout_developmental_plasticity_review",
        "owned_by_marulho": True,
        "mutates_runtime_state": False,
        "applies_plasticity": False,
        "rollout_consolidation_design_hash": "design-hash-1",
        "rollout_consolidation_shadow_application_preflight_hash": (
            "preflight-hash-1"
        ),
        "rollout_developmental_plasticity_review_hash": review_hash,
        "candidate_source_window": {},
        "developmental_plasticity_review": {
            "growth_candidate_count": 1,
            "structural_growth_applied": False,
            "structural_pruning_applied": False,
            "growth_candidates": [growth_candidate],
        },
        "topology_budget_evidence": {
            "candidate_count_bounded": True,
            "outgoing_row_mass_bounded": True,
            "outgoing_fanout_bounded": True,
            "global_sparse_edge_budget_bounded": True,
        },
        "growth_classification": {"all_candidates_are_growth": True},
        "runtime_memory_evidence": {
            "transition_memory_snapshot_hash": transition_memory_snapshot_hash,
            "language_capacity": {
                "surface": "snn_language_capacity_state.v1",
                "language_neuron_count": 128,
                "sparse_edge_budget": 512,
                "outgoing_fanout_budget": 32,
                "capacity_expansion_count": 1,
            },
        },
        "promotion_gate": {
            "eligible_for_operator_rollout_developmental_plasticity_review": True,
            "eligible_for_transition_memory_regeneration_proposal": True,
            "eligible_for_growth": False,
            "eligible_for_pruning": False,
        },
    }

    adapter = ledger.rollout_regeneration_proposal_adapter(review)

    assert adapter["promotion_gate"]["eligible_for_operator_rollout_regeneration_adapter_review"] is True
    assert adapter["integrity_evidence"]["candidate_indices_canonical"] is True
    assert adapter["integrity_evidence"]["language_capacity_state_available"] is True
    assert adapter["integrity_evidence"]["language_capacity_state_dynamic_limits_applied"] is True
    assert adapter["language_capacity"]["language_neuron_count"] == 128
    assert adapter["regeneration_design"]["candidate_synapses"][0]["synapse"] == "65:66"
    replay_material = {
        "recorded_state_revision": runtime_state.state_revision,
        "operator_id": "operator-test",
        "confirmation": True,
        "mismatch_hash": "mismatch-hash-1",
        "mismatch_score": 0.9,
        "pressure_hash": "pressure-hash-1",
        "pressure_score": 0.7,
        "replay_window_hash": "window-hash-1",
        "replay_window_size": 4,
        "internal_ledger_backed": True,
        "artifact_proposal_hash": "artifact-proposal-hash-1",
        "replay_evaluation_context_id": "context-1",
        "replay_evaluation_context_hash": "context-hash-1",
        "review_ticket_id": "ticket-1",
        "review_ticket_hash": "ticket-hash-1",
        "readout_evidence_hashes": ["readout-hash-1"],
    }
    replay = {
        **replay_material,
        "surface": "snn_transition_memory_replay_artifact.v1",
        "artifact_kind": "terminus_snn_transition_memory_replay_artifact",
        "owned_by_marulho": True,
        "ready": True,
        "evidence_hash": _sha256_json(replay_material),
        "replay_artifact_id": "artifact-1",
        "replay_window_id": "window-1",
    }

    replay_review = ledger.rollout_regeneration_replay_artifact_review(
        adapter,
        replay,
    )

    assert replay_review["promotion_gate"]["eligible_for_operator_rollout_regeneration_replay_artifact_review"] is True
    replay_required = replay_review["promotion_gate"]["required_evidence"]
    assert replay_required["regeneration_design_indices_canonical"] is True
    assert replay_required["candidate_source_window_bounded"] is True
    assert replay_required["candidate_payload_not_truncated"] is True
    assert replay_required["language_capacity_state_available"] is True
    assert replay_required["language_capacity_state_dynamic_limits_applied"] is True
    assert replay_review["language_capacity"]["language_neuron_count"] == 128
    assert replay_review["regeneration_design"]["candidate_synapses"][0]["synapse"] == "65:66"


def test_readout_ledger_rollout_regeneration_replay_artifact_review_binds_replay_without_permit() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=_ready_rollout_replay_evaluation(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    evaluation = ledger.rollout_rehearsal_evaluation(
        ledger.rollout_rehearsal_promotion_policy(candidate_limit=4),
        candidate_limit=4,
    )
    experiment = ledger.rollout_rehearsal_experiment(evaluation, replay_cycles=4)
    design = ledger.rollout_consolidation_design(
        experiment,
        rollback_policy={"available": True, "snapshot_id": "rollout-snapshot-1"},
    )
    shadow = ledger.rollout_consolidation_shadow_delta(
        design,
        device_evidence={"device": "cpu", "source": "test"},
    )
    growth_preflight = ledger.rollout_consolidation_shadow_application_preflight(
        design,
        shadow,
        transition_memory_state={
            "sparse_transition_weights": {},
            "language_capacity": _language_capacity(),
        },
    )
    developmental = ledger.rollout_developmental_plasticity_review(
        design,
        growth_preflight,
        transition_memory_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "sparse_transition_weights": {},
            "language_capacity": _language_capacity(),
        },
    )
    adapter = ledger.rollout_regeneration_proposal_adapter(developmental)
    replay_material = {
        "recorded_state_revision": runtime_state.state_revision,
        "operator_id": "operator-test",
        "confirmation": True,
        "mismatch_hash": "mismatch-hash-1",
        "mismatch_score": 0.9,
        "pressure_hash": "pressure-hash-1",
        "pressure_score": 0.9,
        "replay_window_hash": "window-hash-1",
        "replay_window_size": 1,
        "internal_ledger_backed": True,
        "artifact_proposal_hash": "proposal-hash-1",
        "replay_evaluation_context_id": "context-1",
        "replay_evaluation_context_hash": "context-hash-1",
        "review_ticket_id": "ticket-1",
        "review_ticket_hash": "ticket-hash-1",
        "readout_evidence_hashes": ["readout-hash-1"],
    }
    replay_artifact = {
        "artifact_kind": "terminus_snn_transition_memory_replay_artifact",
        "surface": "snn_transition_memory_replay_artifact.v1",
        "available": True,
        "ready": True,
        "owned_by_marulho": True,
        "source": "replay_controller.snn_transition_memory_replay_artifact",
        "replay_artifact_id": "artifact-1",
        "replay_window_id": "replay-window-1",
        "evidence_hash": _sha256_json(replay_material),
        **replay_material,
    }
    tampered = deepcopy(replay_artifact)
    tampered["readout_evidence_hashes"] = []
    before = runtime_state.snapshot()

    review = ledger.rollout_regeneration_replay_artifact_review(adapter, replay_artifact)
    repeat = ledger.rollout_regeneration_replay_artifact_review(adapter, replay_artifact)
    blocked = ledger.rollout_regeneration_replay_artifact_review(adapter, tampered)
    oversized_adapter = deepcopy(adapter)
    oversized_adapter["regeneration_design"]["candidate_count"] = (
        SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT + 1
    )
    oversized_adapter["regeneration_design"]["max_new_synapses"] = (
        SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT + 1
    )
    oversized_adapter["regeneration_design"]["candidate_synapses"] = (
        _rollout_growth_candidates(SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT + 1)
    )
    oversized = ledger.rollout_regeneration_replay_artifact_review(
        oversized_adapter,
        replay_artifact,
    )
    after = runtime_state.snapshot()

    assert before == after
    assert review == repeat
    assert review["surface"] == "snn_language_readout_rollout_regeneration_replay_artifact_review.v1"
    assert review["generates_text"] is False
    assert review["applies_plasticity"] is False
    assert review["mutates_runtime_state"] is False
    assert review["issues_regeneration_permit"] is False
    assert review["executor_ready"] is False
    assert review["regeneration_design"]["mismatch_score"] == 0.9
    assert review["replay_mismatch_evidence"]["mismatch_score"] == 0.9
    assert review["candidate_source_window"]["surface"] == (
        "bounded_snn_rollout_regeneration_replay_artifact_review_candidate_window.v1"
    )
    assert review["candidate_source_window"]["source_window_count"] == 1
    assert review["promotion_gate"]["required_evidence"]["candidate_source_window_bounded"] is True
    assert review["promotion_gate"]["required_evidence"]["candidate_payload_not_truncated"] is True
    assert review["promotion_gate"]["required_evidence"]["replay_mismatch_score_high"] is True
    assert review["permit_request_preview"]["replay_artifact_id"] == "artifact-1"
    assert review["permit_request_preview"]["regeneration_design"]["mismatch_score"] == 0.9
    assert review["permit_request_preview"]["permit_issued"] is False
    assert review["promotion_gate"]["eligible_for_regeneration_permit_request"] is True
    assert review["promotion_gate"]["eligible_for_regeneration_application"] is False
    assert review["promotion_gate"]["eligible_for_structural_write"] is False
    assert blocked["promotion_gate"]["required_evidence"]["replay_artifact_hash_recomputed_match"] is False
    assert blocked["promotion_gate"]["required_evidence"]["replay_readout_evidence_available"] is False
    assert blocked["promotion_gate"]["eligible_for_regeneration_permit_request"] is False
    assert oversized["promotion_gate"]["required_evidence"]["candidate_source_window_bounded"] is True
    assert oversized["promotion_gate"]["required_evidence"]["candidate_payload_not_truncated"] is False
    assert oversized["candidate_source_window"]["source_window_count"] == (
        SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
    )
    assert oversized["candidate_source_window"]["source_payload_truncated"] is True
    assert oversized["permit_request_preview"]["regeneration_design"] is None
    assert oversized["promotion_gate"]["eligible_for_regeneration_permit_request"] is False


def test_rollout_regeneration_permit_request_uses_replay_controller_without_synapse_mutation() -> None:
    runtime_state = RuntimeState()
    calls: list[dict[str, object]] = []

    class _ReplayController:
        def issue_regeneration_permit(self, **kwargs: object) -> dict[str, object]:
            calls.append(dict(kwargs))
            runtime_state.mark_dirty_without_revision()
            return {
                "artifact_kind": "terminus_snn_language_transition_memory_regeneration_permit",
                "surface": "snn_language_transition_memory_regeneration_permit.v1",
                "ready": True,
                "owned_by_marulho": True,
                "permit_id": "permit-1",
                "replay_artifact_id": kwargs["replay_artifact_id"],
                "regeneration_design_hash": "design-hash-1",
            }

    class _Root:
        _runtime_state = runtime_state
        _replay_controller = _ReplayController()

    class _MismatchedRestoreRoot:
        _runtime_state = runtime_state
        _replay_controller = _ReplayController()
        _metadata = {
            "service_state": {
                "snn_applied_replay_lineage_restore_validation": {
                    "surface": "snn_applied_replay_lineage_restore_validation.v1",
                    "summary_matches_restored_state": False,
                }
            }
        }

    facade = RuntimeFacade(_Root())
    mismatched_facade = RuntimeFacade(_MismatchedRestoreRoot())
    review = {
        "surface": "snn_language_readout_rollout_regeneration_replay_artifact_review.v1",
        "owned_by_marulho": True,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "rollout_regeneration_replay_artifact_review_hash": "review-hash-1",
        "language_capacity": {
            "surface": "snn_language_capacity_state.v1",
            "language_neuron_count": 128,
            "sparse_edge_budget": 512,
            "outgoing_fanout_budget": 32,
            "capacity_expansion_count": 1,
        },
        "permit_request_preview": {
            "replay_artifact_id": "artifact-1",
            "regeneration_design": {
                "locality_radius": 1,
                "initial_weight": 0.02,
                "max_new_synapses": 1,
                "mismatch_score": 0.9,
                "candidate_count": 1,
                "candidate_synapses": [
                    {
                        "pre_index": 65,
                        "post_index": 66,
                        "synapse": "65:66",
                        "initial_weight": 0.02,
                        "locality_distance": 1,
                    }
                ],
            },
            "permit_issued": False,
        },
        "promotion_gate": {"eligible_for_regeneration_permit_request": True},
    }

    blocked = facade.snn_language_readout_rollout_regeneration_permit_request(
        rollout_regeneration_replay_artifact_review=review,
        operator_id="operator-test",
        confirmation=False,
    )
    accepted = facade.snn_language_readout_rollout_regeneration_permit_request(
        rollout_regeneration_replay_artifact_review=review,
        operator_id="operator-test",
        confirmation=True,
    )
    mismatched_restore = mismatched_facade.snn_language_readout_rollout_regeneration_permit_request(
        rollout_regeneration_replay_artifact_review=review,
        operator_id="operator-test",
        confirmation=True,
    )

    assert blocked["accepted"] is False
    assert blocked["issues_regeneration_permit"] is False
    assert mismatched_restore["accepted"] is False
    assert mismatched_restore["issues_regeneration_permit"] is False
    assert mismatched_restore["promotion_gate"]["required_evidence"][
        "applied_replay_lineage_restore_validation_not_mismatched"
    ] is False
    assert calls == [
        {
            "replay_artifact_id": "artifact-1",
            "regeneration_design": review["permit_request_preview"]["regeneration_design"],
            "operator_id": "operator-test",
            "confirmation": True,
        }
    ]
    assert accepted["accepted"] is True
    assert accepted["issues_regeneration_permit"] is True
    assert accepted["executor_ready"] is False
    assert accepted["applies_plasticity"] is False
    assert accepted["language_capacity"]["language_neuron_count"] == 128
    assert accepted["promotion_gate"]["required_evidence"]["regeneration_design_indices_canonical"] is True
    assert accepted["promotion_gate"]["required_evidence"]["candidate_source_window_bounded"] is True
    assert accepted["promotion_gate"]["required_evidence"]["candidate_payload_not_truncated"] is True
    assert (
        accepted["candidate_source_window"]["surface"]
        == "bounded_snn_rollout_regeneration_permit_candidate_synapse_window.v1"
    )
    assert accepted["candidate_source_window"]["source_window_limit"] == (
        SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
    )
    assert accepted["candidate_source_window"]["source_window_count"] == 1
    assert accepted["promotion_gate"]["required_evidence"]["language_capacity_state_available"] is True
    assert accepted["promotion_gate"]["required_evidence"]["language_capacity_state_dynamic_limits_applied"] is True
    assert accepted["replay_evidence"]["permit_id"] == "permit-1"
    assert accepted["before"]["state_revision"] == accepted["after"]["state_revision"]
    assert accepted["after"]["dirty_state"] is True
    assert accepted["promotion_gate"]["eligible_for_regeneration_application"] is True
    assert accepted["promotion_gate"]["eligible_for_structural_write"] is False


def test_rollout_regeneration_permit_request_blocks_oversized_candidate_window_before_replay_controller() -> None:
    runtime_state = RuntimeState()
    calls: list[dict[str, object]] = []

    class _ReplayController:
        def issue_regeneration_permit(self, **kwargs: object) -> dict[str, object]:
            calls.append(dict(kwargs))
            raise AssertionError("oversized permit must not reach replay controller")

    class _Root:
        _runtime_state = runtime_state
        _replay_controller = _ReplayController()

    facade = RuntimeFacade(_Root())
    review = {
        "surface": "snn_language_readout_rollout_regeneration_replay_artifact_review.v1",
        "owned_by_marulho": True,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "language_capacity": {
            "surface": "snn_language_capacity_state.v1",
            "language_neuron_count": 128,
            "sparse_edge_budget": 512,
            "outgoing_fanout_budget": 32,
            "capacity_expansion_count": 1,
        },
        "permit_request_preview": {
            "replay_artifact_id": "artifact-1",
            "regeneration_design": {
                "locality_radius": 1,
                "initial_weight": 0.02,
                "max_new_synapses": SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT + 1,
                "mismatch_score": 0.9,
                "candidate_count": SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT + 1,
                "candidate_synapses": _regeneration_candidates(
                    SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT + 1
                ),
            },
            "permit_issued": False,
        },
        "promotion_gate": {"eligible_for_regeneration_permit_request": True},
    }

    result = facade.snn_language_readout_rollout_regeneration_permit_request(
        rollout_regeneration_replay_artifact_review=review,
        operator_id="operator-test",
        confirmation=True,
    )

    required = result["promotion_gate"]["required_evidence"]
    assert result["accepted"] is False
    assert result["issues_regeneration_permit"] is False
    assert calls == []
    assert required["candidate_source_window_bounded"] is True
    assert required["candidate_payload_not_truncated"] is False
    assert required["candidate_source_window"]["source_window_count"] == (
        SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
    )
    assert required["candidate_source_window"]["source_payload_truncated"] is True
    assert runtime_state.state_revision == 0


def test_rollout_regeneration_application_preflight_requires_revision_and_checkpoint() -> None:
    runtime_state = RuntimeState()

    class _Root:
        _runtime_state = runtime_state

    class _MismatchedRestoreRoot:
        _runtime_state = runtime_state
        _metadata = {
            "service_state": {
                "snn_applied_replay_lineage_restore_validation": {
                    "surface": "snn_applied_replay_lineage_restore_validation.v1",
                    "summary_matches_restored_state": False,
                }
            }
        }

    facade = RuntimeFacade(_Root())
    mismatched_facade = RuntimeFacade(_MismatchedRestoreRoot())
    permit_request = {
        "surface": "snn_language_readout_rollout_regeneration_permit_request.v1",
        "accepted": True,
        "owned_by_marulho": True,
        "applies_plasticity": False,
        "mutates_runtime_state": True,
        "checkpoint_written": False,
        "language_capacity": {
            "surface": "snn_language_capacity_state.v1",
            "language_neuron_count": 128,
            "sparse_edge_budget": 512,
            "outgoing_fanout_budget": 16,
            "capacity_expansion_count": 1,
        },
        "replay_evidence": {
            "permit_id": "permit-1",
            "ready": True,
            "owned_by_marulho": True,
        },
        "regeneration_design": {
            "locality_radius": 1,
            "initial_weight": 0.02,
            "max_new_synapses": 1,
            "mismatch_score": 0.9,
            "candidate_count": 1,
            "candidate_synapses": [
                {
                    "pre_index": 65,
                    "post_index": 66,
                    "synapse": "65:66",
                    "initial_weight": 0.02,
                    "locality_distance": 1,
                }
            ],
        },
        "promotion_gate": {
            "eligible_for_regeneration_application": True,
            "required_evidence": {
                "applied_replay_lineage_restore_validation_not_mismatched": True
            },
        },
    }

    blocked = facade.snn_language_readout_rollout_regeneration_application_preflight(
        rollout_regeneration_permit_request=permit_request,
        expected_state_revision=runtime_state.state_revision + 1,
        checkpoint_path=None,
    )
    ready = facade.snn_language_readout_rollout_regeneration_application_preflight(
        rollout_regeneration_permit_request=permit_request,
        expected_state_revision=runtime_state.state_revision,
        checkpoint_path="checkpoint://rollout-regeneration",
    )
    mismatched_restore = mismatched_facade.snn_language_readout_rollout_regeneration_application_preflight(
        rollout_regeneration_permit_request=permit_request,
        expected_state_revision=runtime_state.state_revision,
        checkpoint_path="checkpoint://rollout-regeneration",
    )

    assert blocked["ready"] is False
    assert blocked["promotion_gate"]["required_evidence"]["expected_revision_current"] is False
    assert blocked["promotion_gate"]["required_evidence"]["checkpoint_path_available"] is False
    assert ready["ready"] is True
    assert ready["executor_called"] is False
    assert ready["writes_checkpoint"] is False
    assert ready["applies_plasticity"] is False
    assert ready["mutates_runtime_state"] is False
    assert ready["language_capacity"]["language_neuron_count"] == 128
    assert ready["regeneration_proposal"]["available"] is True
    assert ready["regeneration_proposal"]["language_capacity"]["language_neuron_count"] == 128
    assert ready["regeneration_proposal"]["promotion_gate"]["status"] == "ready_for_operator_review"
    assert ready["promotion_gate"]["eligible_for_checkpoint_backed_regeneration_executor"] is True
    assert ready["promotion_gate"]["eligible_for_regeneration_application"] is False
    assert (
        ready["promotion_gate"]["required_evidence"][
            "regeneration_design_indices_canonical"
        ]
        is True
    )
    assert ready["promotion_gate"]["required_evidence"]["candidate_source_window_bounded"] is True
    assert ready["promotion_gate"]["required_evidence"]["candidate_payload_not_truncated"] is True
    assert (
        ready["candidate_source_window"]["surface"]
        == (
            "bounded_snn_rollout_regeneration_application_preflight_"
            "candidate_synapse_window.v1"
        )
    )
    assert ready["candidate_source_window"]["source_window_count"] == 1
    assert (
        ready["promotion_gate"]["required_evidence"][
            "language_capacity_state_available"
        ]
        is True
    )
    assert (
        ready["promotion_gate"]["required_evidence"][
            "language_capacity_state_dynamic_limits_applied"
        ]
        is True
    )
    assert mismatched_restore["ready"] is False
    assert mismatched_restore["promotion_gate"]["required_evidence"][
        "applied_replay_lineage_restore_validation_not_mismatched"
    ] is False


def test_rollout_regeneration_application_preflight_blocks_oversized_candidate_window() -> None:
    runtime_state = RuntimeState()

    class _Root:
        _runtime_state = runtime_state

    facade = RuntimeFacade(_Root())
    permit_request = {
        "surface": "snn_language_readout_rollout_regeneration_permit_request.v1",
        "accepted": True,
        "owned_by_marulho": True,
        "applies_plasticity": False,
        "mutates_runtime_state": True,
        "checkpoint_written": False,
        "language_capacity": {
            "surface": "snn_language_capacity_state.v1",
            "language_neuron_count": 128,
            "sparse_edge_budget": 512,
            "outgoing_fanout_budget": 16,
            "capacity_expansion_count": 1,
        },
        "replay_evidence": {
            "permit_id": "permit-1",
            "ready": True,
            "owned_by_marulho": True,
        },
        "regeneration_design": {
            "locality_radius": 1,
            "initial_weight": 0.02,
            "max_new_synapses": SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT + 1,
            "mismatch_score": 0.9,
            "candidate_count": SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT + 1,
            "candidate_synapses": _regeneration_candidates(
                SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT + 1
            ),
        },
        "promotion_gate": {
            "eligible_for_regeneration_application": True,
            "required_evidence": {
                "applied_replay_lineage_restore_validation_not_mismatched": True
            },
        },
    }

    result = facade.snn_language_readout_rollout_regeneration_application_preflight(
        rollout_regeneration_permit_request=permit_request,
        expected_state_revision=runtime_state.state_revision,
        checkpoint_path="checkpoint://rollout-regeneration",
    )

    required = result["promotion_gate"]["required_evidence"]
    assert result["ready"] is False
    assert result["executor_called"] is False
    assert result["regeneration_proposal"]["available"] is False
    assert required["candidate_source_window_bounded"] is True
    assert required["candidate_payload_not_truncated"] is False
    assert required["candidate_source_window"]["source_window_count"] == (
        SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
    )
    assert len(result["regeneration_proposal"]["regeneration_design"]["candidate_synapses"]) == (
        SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
    )
    assert runtime_state.state_revision == 0


def test_rollout_regeneration_application_delegates_to_checkpoint_backed_executor(
    tmp_path: Path,
) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {
        "sparse_transition_weights": {"1:2": 0.9},
        "language_capacity": {
            "surface": "snn_language_capacity_state.v1",
            "language_neuron_count": 128,
            "sparse_edge_budget": 512,
            "outgoing_fanout_budget": 16,
            "capacity_expansion_count": 1,
        },
    }

    def save_checkpoint(path: str | None) -> dict[str, str]:
        target = Path(path or tmp_path / "rollout-regeneration.pt")
        target.write_bytes(b"checkpoint")
        return {"path": str(target)}

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / "rollout-regeneration.pt",
        verify_checkpoint=lambda path: path.exists(),
        verify_regeneration_permit=lambda proposal: True,
    )

    class _Root:
        _runtime_state = runtime_state
        _snn_language_plasticity_executor = executor

    facade = RuntimeFacade(_Root())
    checkpoint_path = str(tmp_path / "rollout-regeneration.pt")
    preflight = {
        "surface": "snn_language_readout_rollout_regeneration_application_preflight.v1",
        "ready": True,
        "owned_by_marulho": True,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "executor_called": False,
        "expected_state_revision": runtime_state.state_revision,
        "checkpoint_path": checkpoint_path,
        "language_capacity": {
            "surface": "snn_language_capacity_state.v1",
            "language_neuron_count": 128,
            "sparse_edge_budget": 512,
            "outgoing_fanout_budget": 16,
            "capacity_expansion_count": 1,
        },
        "regeneration_proposal": {
            "available": True,
            "ready": True,
            "owned_by_marulho": True,
            "generates_text": False,
            "loads_external_checkpoint": False,
            "language_capacity": {
                "surface": "snn_language_capacity_state.v1",
                "language_neuron_count": 128,
                "sparse_edge_budget": 512,
                "outgoing_fanout_budget": 16,
                "capacity_expansion_count": 1,
            },
            "promotion_gate": {"status": "ready_for_operator_review"},
            "replay_evidence": {
                "available": True,
                "ready": True,
                "owned_by_marulho": True,
                "source": "replay_controller.regeneration_permit",
                "permit_id": "permit-1",
                "replay_window_id": "replay-window-1",
                "replay_artifact_id": "artifact-1",
                "replay_artifact_hash": "artifact-hash-1",
                "replay_window_hash": "window-hash-1",
                "readout_evidence_hashes": ["readout-hash-1"],
                "evidence_hash": "sha256:rollout-replay-window-1",
            },
            "regeneration_design": {
                "locality_radius": 2,
                "mismatch_score": 0.9,
                "candidate_synapses": [
                    {
                        "pre_index": 65,
                        "post_index": 66,
                        "initial_weight": 0.1,
                        "locality_distance": 1,
                        "source_synapse_id": "snn-rollout-local:65:66:0",
                        "source_trace_index": 0,
                        "source_rollout_step_index": 10,
                        "target_rollout_step_index": 20,
                        "source_active_indices_hash": "source-active-hash-1",
                        "target_active_indices_hash": "target-active-hash-1",
                    }
                ],
            },
        },
        "promotion_gate": {
            "eligible_for_checkpoint_backed_regeneration_executor": True,
        },
    }

    result = facade.snn_language_readout_rollout_regeneration_application(
        rollout_regeneration_application_preflight=preflight,
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
        checkpoint_path=checkpoint_path,
    )

    assert result["accepted"] is True
    assert result["executor_called"] is True
    assert result["writes_checkpoint"] is True
    assert result["applies_plasticity"] is True
    assert result["mutates_runtime_state"] is True
    assert result["language_capacity"]["language_neuron_count"] == 128
    assert result["proposal_language_capacity"]["language_neuron_count"] == 128
    assert result["promotion_gate"]["required_evidence"]["regeneration_design_indices_canonical"] is True
    assert result["promotion_gate"]["required_evidence"]["candidate_source_window_bounded"] is True
    assert result["promotion_gate"]["required_evidence"]["candidate_payload_not_truncated"] is True
    assert (
        result["candidate_source_window"]["surface"]
        == "bounded_snn_rollout_regeneration_application_candidate_synapse_window.v1"
    )
    assert result["candidate_source_window"]["source_window_count"] == 1
    assert result["promotion_gate"]["required_evidence"]["language_capacity_state_available"] is True
    assert result["promotion_gate"]["required_evidence"]["proposal_language_capacity_state_available"] is True
    assert result["promotion_gate"]["required_evidence"]["proposal_language_capacity_matches_preflight"] is True
    assert result["promotion_gate"]["required_evidence"]["language_capacity_state_dynamic_limits_applied"] is True
    assert result["executor_result"]["surface"] == "snn_language_transition_memory_regeneration.v1"
    assert abs(language_state["sparse_transition_weights"]["65:66"] - 0.1) < 1e-9
    local_edge = result["executor_result"]["regeneration"]["regenerated_synapses"][0][
        "local_edge_provenance"
    ]
    assert local_edge["source_synapse_id"] == "snn-rollout-local:65:66:0"
    assert local_edge["source_rollout_step_index"] == 10
    assert local_edge["target_rollout_step_index"] == 20
    assert local_edge["source_active_indices_hash"] == "source-active-hash-1"
    assert language_state["synapse_provenance_by_key"]["65:66"][
        "local_edge_provenance"
    ] == local_edge
    assert runtime_state.state_revision == 1
    assert result["promotion_gate"]["status"] == "checkpoint_backed_regeneration_applied"


def test_rollout_regeneration_application_blocks_oversized_candidate_window_before_executor(
    tmp_path: Path,
) -> None:
    runtime_state = RuntimeState()
    calls: list[dict[str, object]] = []

    class _Executor:
        def regenerate_transition_memory(self, **kwargs: object) -> dict[str, object]:
            calls.append(dict(kwargs))
            raise AssertionError("oversized application must not reach executor")

    class _Root:
        _runtime_state = runtime_state
        _snn_language_plasticity_executor = _Executor()

    facade = RuntimeFacade(_Root())
    checkpoint_path = str(tmp_path / "rollout-regeneration.pt")
    preflight = {
        "surface": "snn_language_readout_rollout_regeneration_application_preflight.v1",
        "ready": True,
        "owned_by_marulho": True,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "executor_called": False,
        "expected_state_revision": runtime_state.state_revision,
        "checkpoint_path": checkpoint_path,
        "language_capacity": {
            "surface": "snn_language_capacity_state.v1",
            "language_neuron_count": 128,
            "sparse_edge_budget": 512,
            "outgoing_fanout_budget": 16,
            "capacity_expansion_count": 1,
        },
        "regeneration_proposal": {
            "available": True,
            "ready": True,
            "owned_by_marulho": True,
            "generates_text": False,
            "loads_external_checkpoint": False,
            "language_capacity": {
                "surface": "snn_language_capacity_state.v1",
                "language_neuron_count": 128,
                "sparse_edge_budget": 512,
                "outgoing_fanout_budget": 16,
                "capacity_expansion_count": 1,
            },
            "promotion_gate": {"status": "ready_for_operator_review"},
            "replay_evidence": {
                "available": True,
                "ready": True,
                "owned_by_marulho": True,
                "source": "replay_controller.regeneration_permit",
                "permit_id": "permit-1",
                "replay_window_id": "replay-window-1",
                "replay_artifact_id": "artifact-1",
                "replay_artifact_hash": "artifact-hash-1",
                "replay_window_hash": "window-hash-1",
                "readout_evidence_hashes": ["readout-hash-1"],
                "evidence_hash": "sha256:rollout-replay-window-1",
            },
            "regeneration_design": {
                "locality_radius": 1,
                "mismatch_score": 0.9,
                "candidate_synapses": _regeneration_candidates(
                    SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT + 1
                ),
            },
        },
        "promotion_gate": {
            "eligible_for_checkpoint_backed_regeneration_executor": True,
        },
    }

    result = facade.snn_language_readout_rollout_regeneration_application(
        rollout_regeneration_application_preflight=preflight,
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
        checkpoint_path=checkpoint_path,
    )

    required = result["promotion_gate"]["required_evidence"]
    assert result["accepted"] is False
    assert result["executor_called"] is False
    assert result["writes_checkpoint"] is False
    assert calls == []
    assert required["candidate_source_window_bounded"] is True
    assert required["candidate_payload_not_truncated"] is False
    assert required["candidate_source_window"]["source_window_count"] == (
        SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
    )
    assert not Path(checkpoint_path).exists()
    assert runtime_state.state_revision == 0


def test_readout_ledger_replay_priority_is_deterministic_read_only_advisory() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    drafts = [
        _ready_draft_for("prediction-a", "evaluation-a", "weights-a", ["memory pressure"]),
        _ready_draft_for("prediction-b", "evaluation-b", "weights-a", ["memory pressure"]),
        _ready_draft_for("prediction-c", "evaluation-c", "weights-c", ["prediction error"]),
    ]
    for draft in drafts:
        ledger.record_readout_draft(
            readout_draft=draft,
            expected_state_revision=runtime_state.state_revision,
            operator_id="operator-test",
            confirmation=True,
        )
    before = runtime_state.snapshot()

    priority = ledger.replay_priority(limit=2)
    repeat = ledger.replay_priority(limit=2)
    empty_limit = ledger.replay_priority(limit=0)
    after = runtime_state.snapshot()

    assert priority == repeat
    assert before == after
    assert priority["surface"] == "snn_language_readout_replay_priority.v1"
    assert priority["advisory"] is True
    assert priority["executable"] is False
    assert priority["mutates_runtime_state"] is False
    assert priority["generates_text"] is False
    assert priority["runs_live_tick"] is False
    assert priority["runs_every_token"] is False
    assert priority["raw_text_payload_loaded"] is False
    assert priority["language_reasoning"] is False
    assert priority["archival_storage_device"] == "cpu"
    assert priority["score_device"] == "cpu"
    assert priority["gpu_used"] is False
    assert priority["global_candidate_scan"] is False
    assert priority["global_score_scan"] is False
    assert priority["source_window"]["surface"] == (
        "bounded_snn_readout_replay_priority_source_window.v1"
    )
    assert priority["source_window"]["source_event_window_count"] == 3
    assert priority["source_window"]["candidate_count_before_rank"] == 3
    assert priority["source_window"]["candidate_count_returned"] == 2
    assert priority["source_window"]["archival_storage_device"] == "cpu"
    assert priority["source_window"]["score_device"] == "cpu"
    assert priority["source_window"]["gpu_used"] is False
    assert priority["source_window"]["global_candidate_scan"] is False
    assert priority["source_window"]["global_score_scan"] is False
    assert priority["source_window"]["runs_live_tick"] is False
    assert priority["source_window"]["language_reasoning"] is False
    assert priority["ledger_summary"]["unique_count_scope"] == "source_window"
    assert priority["promotion_gate"]["eligible_for_operator_replay_review"] is True
    assert priority["promotion_gate"]["eligible_for_live_replay"] is False
    assert priority["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert priority["promotion_gate"]["required_evidence"]["source_window_bounded"] is True
    assert priority["promotion_gate"]["required_evidence"][
        "archival_metadata_cpu_resident"
    ] is True
    assert priority["promotion_gate"]["required_evidence"][
        "language_reasoning_absent"
    ] is True
    assert priority["candidate_count"] == 2
    assert priority["candidates"][0]["rank"] == 1
    assert priority["candidates"][0]["readout_evidence_hash"]
    assert priority["candidates"][0]["prediction_hash"]
    assert priority["candidates"][0]["priority_components"]["provenance"] == 1.0
    assert priority["candidates"][0]["executable"] is False
    assert priority["candidates"][0]["generates_text"] is False
    assert priority["candidates"][0]["eligible_for_action"] is False
    assert empty_limit["candidate_count"] == 0
    assert empty_limit["source_window"]["candidate_count_before_rank"] == 3
    assert empty_limit["source_window"]["candidate_count_returned"] == 0
    assert empty_limit["promotion_gate"]["status"] == "collect_readout_evidence"


def test_readout_ledger_replay_priority_caps_source_events_before_rank() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
        limit=SNN_READOUT_REPLAY_PRIORITY_SOURCE_WINDOW_LIMIT * 2,
    )
    for index in range(SNN_READOUT_REPLAY_PRIORITY_SOURCE_WINDOW_LIMIT * 2):
        labels = ["outside-window"] if index == 0 else [f"readout-window-{index}"]
        ledger.record_readout_draft(
            readout_draft=_ready_draft_for(
                f"prediction-{index}",
                f"evaluation-{index}",
                f"weights-{index}",
                labels,
            ),
            expected_state_revision=runtime_state.state_revision,
            operator_id="operator-test",
            confirmation=True,
        )

    priority = ledger.replay_priority(limit=8)
    candidate_labels = {
        label
        for candidate in priority["candidates"]
        for label in list(candidate.get("labels") or [])
    }

    assert priority["candidate_count"] == 8
    assert priority["source_window"]["source_event_retention_count"] == (
        SNN_READOUT_REPLAY_PRIORITY_SOURCE_WINDOW_LIMIT * 2
    )
    assert priority["source_window"]["source_event_window_count"] == (
        SNN_READOUT_REPLAY_PRIORITY_SOURCE_WINDOW_LIMIT
    )
    assert priority["source_window"]["source_event_truncated_count"] == (
        SNN_READOUT_REPLAY_PRIORITY_SOURCE_WINDOW_LIMIT
    )
    assert priority["source_window"]["candidate_count_before_rank"] == (
        SNN_READOUT_REPLAY_PRIORITY_SOURCE_WINDOW_LIMIT
    )
    assert priority["source_window"]["candidate_count_returned"] == 8
    assert priority["source_window"]["global_candidate_scan"] is False
    assert priority["source_window"]["global_score_scan"] is False
    assert priority["source_window"]["runs_live_tick"] is False
    assert priority["source_window"]["gpu_used"] is False
    assert "outside-window" not in candidate_labels


def test_readout_ledger_replay_priority_source_window_validator_requires_explicit_flags() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_readout_draft(
        readout_draft=_ready_draft(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    priority = ledger.replay_priority(limit=1)
    source_window = dict(priority["source_window"])

    assert ledger._readout_replay_priority_source_window_bounded(source_window) is True
    missing_flag = dict(source_window)
    missing_flag.pop("raw_text_payload_loaded")
    assert ledger._readout_replay_priority_source_window_bounded(missing_flag) is False
    oversized = dict(source_window)
    oversized["source_event_window_limit"] = (
        SNN_READOUT_REPLAY_PRIORITY_SOURCE_WINDOW_LIMIT + 1
    )
    assert ledger._readout_replay_priority_source_window_bounded(oversized) is False


def test_runtime_facade_source_window_validators_require_explicit_flags() -> None:
    rollout_surface = "bounded_snn_rollout_regeneration_replay_artifact_review_candidate_window.v1"
    rollout_window = {
        "surface": rollout_surface,
        "source_window_count": 1,
        "source_window_limit": SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT,
        "source_mapping_count": 1,
        "global_candidate_scan": False,
        "global_score_scan": False,
        "raw_text_payload_loaded": False,
        "hidden_language_reasoning": False,
        "language_reasoning": False,
        "runs_live_tick": False,
        "runs_every_token": False,
        "mutates_runtime_state": False,
        "applies_plasticity": False,
        "gpu_used": False,
        "gpu_resident_archival_metadata": False,
        "archival_storage_device": "cpu",
        "source_window_selection_device": "cpu",
    }
    assert RuntimeFacade._rollout_regeneration_candidate_window_bounded(
        rollout_window,
        surface=rollout_surface,
    ) is True
    missing_rollout_flag = dict(rollout_window)
    missing_rollout_flag.pop("hidden_language_reasoning")
    assert RuntimeFacade._rollout_regeneration_candidate_window_bounded(
        missing_rollout_flag,
        surface=rollout_surface,
    ) is False

    readout_surface = "bounded_snn_readout_replay_payload_source_window.v1"
    readout_window = {
        "surface": readout_surface,
        "source_window_count": 1,
        "source_window_limit": 8,
        "source_mapping_count": 1,
        "global_candidate_scan": False,
        "global_score_scan": False,
        "raw_text_payload_loaded": False,
        "language_reasoning": False,
        "runs_live_tick": False,
        "runs_every_token": False,
        "mutates_runtime_state": False,
        "applies_plasticity": False,
        "gpu_resident_archival_metadata": False,
        "gpu_used_for_archival_metadata": False,
        "archival_storage_device": "cpu",
    }
    assert RuntimeFacade._readout_replay_payload_window_bounded(
        readout_window,
        surface=readout_surface,
    ) is True
    wrong_count = dict(readout_window)
    wrong_count["source_mapping_count"] = 0
    assert RuntimeFacade._readout_replay_payload_window_bounded(
        wrong_count,
        surface=readout_surface,
    ) is False


def test_readout_ledger_snapshot_normalizes_retained_histories_from_source_window() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    source_count = 10
    ledger_limit = 4
    ledger_state: dict[str, object] = {
        field: [
            {
                "field": field,
                "ordinal": index,
                "readout_evidence_hash": f"{field}:readout:{index}",
                "rollout_evidence_hash": f"{field}:rollout:{index}",
                "emission_review_hash": f"{field}:review:{index}",
            }
            for index in range(source_count)
        ]
        for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }
    ledger_state["total_recorded_count"] = source_count
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
        limit=ledger_limit,
    )

    snapshot = ledger.snapshot(limit=2)
    source_window = snapshot["summary"]["snapshot_source_window"]

    assert snapshot["summary"]["event_count"] == ledger_limit
    assert snapshot["summary"]["returned_event_count"] == 2
    assert snapshot["summary"]["unique_count_scope"] == "snapshot_source_window"
    assert len(snapshot["events"]) == 2
    assert snapshot["events"][0]["ordinal"] == 0
    assert snapshot["events"][1]["ordinal"] == 1
    assert source_window["surface"] == (
        "bounded_snn_readout_ledger_snapshot_source_window.v1"
    )
    assert source_window["policy"] == (
        SNN_LANGUAGE_READOUT_LEDGER_SNAPSHOT_SOURCE_WINDOW_POLICY
    )
    assert snapshot["summary"]["normalization_source_window"] == source_window
    assert source_window["event_field_count"] == len(source_window["snapshot_event_fields"])
    assert source_window["requested_source_window_limit_per_field"] == 2
    assert source_window["source_window_limit_per_field"] == 2
    assert source_window["retention_limit_per_field"] == ledger_limit
    assert source_window["source_record_counts"]["events"] == source_count
    assert source_window["source_window_counts"]["events"] == 2
    assert source_window["retained_window_counts"]["events"] == ledger_limit
    assert source_window["truncated_source_counts"]["events"] == (
        source_count - 2
    )
    assert source_window["retained_truncated_source_counts"]["events"] == (
        source_count - ledger_limit
    )
    assert source_window["memory_budget"]["max_records_total"] == (
        len(source_window["snapshot_event_fields"]) * 2
    )
    assert source_window["archival_storage_device"] == "cpu"
    assert source_window["snapshot_device"] == "cpu"
    assert source_window["gpu_used"] is False
    assert source_window["runs_live_tick"] is False
    assert source_window["runs_every_token"] is False
    assert source_window["global_candidate_scan"] is False
    assert source_window["global_score_scan"] is False
    assert source_window["language_reasoning"] is False


def test_readout_ledger_snapshot_reads_only_requested_event_windows() -> None:
    class CountedRows:
        def __init__(self, field: str, count: int) -> None:
            self.field = field
            self.count = count
            self.iterated = 0

        def __iter__(self):
            for index in range(self.count):
                self.iterated += 1
                yield {
                    "field": self.field,
                    "ordinal": index,
                    "readout_evidence_hash": f"{self.field}:readout:{index}",
                }

    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_limit = 8
    snapshot_limit = 3
    rows_by_field = {
        field: CountedRows(field, 64)
        for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }
    ledger_state: dict[str, object] = dict(rows_by_field)
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
        limit=ledger_limit,
    )

    snapshot = ledger.snapshot(limit=snapshot_limit)
    source_window = snapshot["summary"]["snapshot_source_window"]
    snapshot_fields = set(source_window["snapshot_event_fields"])

    assert snapshot["summary"]["event_count"] == snapshot_limit
    assert snapshot["summary"]["returned_event_count"] == snapshot_limit
    assert len(snapshot["events"]) == snapshot_limit
    assert source_window["source_window_limit_per_field"] == snapshot_limit
    assert source_window["memory_budget"]["max_records_per_field"] == snapshot_limit
    assert source_window["source_record_counts"]["events"] is None
    assert source_window["retained_window_counts"]["events"] is None
    for field, rows in rows_by_field.items():
        if field in snapshot_fields:
            assert rows.iterated == snapshot_limit
        else:
            assert rows.iterated == 0


def test_readout_ledger_store_state_uses_bounded_event_field_windows() -> None:
    class CountedRows:
        def __init__(self, field: str, count: int) -> None:
            self.field = field
            self.count = count
            self.iterated = 0

        def __iter__(self):
            for index in range(self.count):
                self.iterated += 1
                yield {
                    "field": self.field,
                    "ordinal": index,
                    "readout_evidence_hash": f"{self.field}:readout:{index}",
                }

    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger_limit = 8
    source_count = 256
    normalized = {
        field: CountedRows(field, source_count)
        for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }
    normalized.update(
        {
            "current_text_surface_commit": {"surface": "current.v1"},
            "total_recorded_count": source_count,
            "last_recorded_at": "2026-06-19T00:00:00+00:00",
        }
    )
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
        limit=ledger_limit,
    )

    ledger._store_state(normalized)  # noqa: SLF001

    for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS:
        stored = ledger_state[field]
        source = normalized[field]
        assert isinstance(stored, list)
        assert len(stored) == ledger_limit
        assert stored[0]["ordinal"] == 0
        assert stored[-1]["ordinal"] == ledger_limit - 1
        assert source.iterated == ledger_limit
    assert ledger_state["current_text_surface_commit"] == {"surface": "current.v1"}
    assert ledger_state["total_recorded_count"] == source_count
    assert ledger_state["last_recorded_at"] == "2026-06-19T00:00:00+00:00"


def test_readout_ledger_recorders_use_single_family_append_windows() -> None:
    class CountedRows:
        def __init__(self, field: str, count: int) -> None:
            self.field = field
            self.count = count
            self.iterated = 0

        def __iter__(self):
            for index in range(self.count):
                self.iterated += 1
                yield {
                    "field": self.field,
                    "ordinal": index,
                    "readout_evidence_hash": f"{self.field}:readout:{index}",
                    "rollout_evidence_hash": f"{self.field}:rollout:{index}",
                    "emission_review_hash": f"{self.field}:review:{index}",
                    "dense_label_candidate_evidence_hash": (
                        f"{self.field}:dense-label:{index}"
                    ),
                    "recorded_at": "2026-06-19T00:00:00+00:00",
                    "reviewed_at": "2026-06-19T00:00:00+00:00",
                }

        def __len__(self) -> int:
            return self.count

    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_limit = 8
    source_count = 256
    counted_sources = {
        field: CountedRows(field, source_count)
        for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }
    ledger_state: dict[str, object] = dict(counted_sources)
    ledger_state.update(
        {
            "total_recorded_count": source_count,
            "total_rollout_recorded_count": source_count,
            "total_emission_review_count": source_count,
            "total_dense_label_candidate_count": source_count,
            "current_text_surface_commit": {"surface": "preserve-current.v1"},
        }
    )
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
        limit=ledger_limit,
    )

    draft_record = ledger.record_readout_draft(
        readout_draft=_ready_draft(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    rollout_record = ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=_ready_rollout_replay_evaluation(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    emission_record = ledger.record_readout_emission_review(
        readout_emission=_ready_emission(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-emission",
        confirmation=True,
    )
    dense_label_record = ledger.record_dense_readout_label_candidate_review(
        dense_readout_label_candidate_review=_ready_dense_label_candidate_review(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-dense-label",
        confirmation=True,
    )

    assert draft_record["accepted"] is True
    assert rollout_record["accepted"] is True
    assert emission_record["accepted"] is True
    assert dense_label_record["accepted"] is True
    _assert_record_family_source_window(
        draft_record["source_window"],
        field="events",
        expected_count=ledger_limit,
    )
    _assert_record_family_source_window(
        rollout_record["source_window"],
        field="rollout_events",
        expected_count=ledger_limit,
    )
    _assert_record_family_source_window(
        emission_record["source_window"],
        field="emission_review_events",
        expected_count=ledger_limit,
    )
    _assert_record_family_source_window(
        dense_label_record["source_window"],
        field="dense_label_candidate_events",
        expected_count=ledger_limit,
    )
    assert draft_record["source_window"]["source_record_count"] == source_count
    assert rollout_record["source_window"]["source_record_count"] == source_count
    assert emission_record["source_window"]["source_record_count"] == source_count
    assert dense_label_record["source_window"]["source_record_count"] == source_count
    assert draft_record["ledger_summary"]["total_recorded_count"] == source_count + 1
    assert rollout_record["ledger_summary"]["total_rollout_recorded_count"] == (
        source_count + 1
    )
    assert emission_record["ledger_summary"]["total_emission_review_count"] == (
        source_count + 1
    )
    assert dense_label_record["ledger_summary"][
        "total_dense_label_candidate_count"
    ] == source_count + 1
    assert ledger_state["current_text_surface_commit"] == {
        "surface": "preserve-current.v1"
    }
    target_fields = {
        "events",
        "rollout_events",
        "emission_review_events",
        "dense_label_candidate_events",
    }
    for field, source in counted_sources.items():
        if field in target_fields:
            assert source.iterated == ledger_limit
            stored = ledger_state[field]
            assert isinstance(stored, list)
            assert len(stored) == ledger_limit
        else:
            assert source.iterated == 0
            assert ledger_state[field] is source


def test_known_readout_evidence_hashes_uses_events_only_source_window() -> None:
    class CountedRows:
        def __init__(self, field: str, count: int) -> None:
            self.field = field
            self.count = count
            self.iterated = 0

        def __iter__(self):
            for index in range(self.count):
                self.iterated += 1
                yield {
                    "field": self.field,
                    "ordinal": index,
                    "readout_evidence_hash": f"{self.field}:readout:{index}",
                }

        def __len__(self) -> int:
            return self.count

    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_limit = 8
    source_count = 256
    ledger_state: dict[str, object] = {
        field: CountedRows(field, source_count)
        for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
        limit=ledger_limit,
    )

    hashes, report = ledger._known_readout_evidence_hashes_with_report()  # noqa: SLF001

    assert hashes == {
        f"events:readout:{index}"
        for index in range(ledger_limit)
    }
    assert not hasattr(ledger, "_known_readout_evidence_hashes")
    assert not hasattr(ledger, "known_readout_evidence_hashes")
    assert report["surface"] == (
        "bounded_snn_readout_known_evidence_hash_source_window.v1"
    )
    assert report["policy"] == SNN_READOUT_EVIDENCE_HASH_SOURCE_WINDOW_POLICY
    assert report["source"] == "snn_readout_ledger.events"
    assert report["source_window_limit"] == ledger_limit
    assert report["source_window_count"] == ledger_limit
    assert report["source_record_count"] == source_count
    assert report["source_payload_truncated"] is True
    assert report["source_truncated_count"] == source_count - ledger_limit
    assert report["hash_count"] == ledger_limit
    assert report["global_candidate_scan"] is False
    assert report["global_score_scan"] is False
    assert report["raw_text_payload_loaded"] is False
    assert report["language_reasoning"] is False
    assert report["runs_live_tick"] is False
    assert report["runs_every_token"] is False
    assert report["mutates_runtime_state"] is False
    assert report["applies_plasticity"] is False
    assert report["archival_storage_device"] == "cpu"
    assert report["lookup_device"] == "cpu"
    assert report["gpu_used"] is False
    assert report["memory_budget"]["max_source_records"] == ledger_limit
    for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS:
        source = ledger_state[field]
        assert isinstance(source, CountedRows)
        if field == "events":
            assert source.iterated == ledger_limit
        else:
            assert source.iterated == 0

    public_hashes, public_report = ledger.known_readout_evidence_hashes_with_report()
    assert public_hashes == hashes
    assert public_report == report


def test_readout_evidence_event_map_uses_requested_hash_source_window() -> None:
    class CountedRows:
        def __init__(self, field: str, count: int) -> None:
            self.field = field
            self.count = count
            self.iterated = 0

        def __iter__(self):
            for index in range(self.count):
                self.iterated += 1
                yield {
                    "field": self.field,
                    "ordinal": index,
                    "readout_evidence_hash": f"{self.field}:readout:{index}",
                }

        def __len__(self) -> int:
            return self.count

    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_limit = 8
    source_count = 256
    ledger_state: dict[str, object] = {
        field: CountedRows(field, source_count)
        for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
        limit=ledger_limit,
    )

    event_map, report = ledger._readout_evidence_event_map_for_hashes_with_report(  # noqa: SLF001
        {
            "events:readout:0",
            "events:readout:7",
            "events:readout:9",
            "missing-readout-hash",
        }
    )

    assert set(event_map) == {"events:readout:0", "events:readout:7"}
    assert report["surface"] == (
        "bounded_snn_readout_evidence_event_map_source_window.v1"
    )
    assert report["policy"] == SNN_READOUT_EVIDENCE_HASH_SOURCE_WINDOW_POLICY
    assert report["source"] == "snn_readout_ledger.events"
    assert report["selection_criteria"] == [
        "requested_readout_evidence_hashes_only",
        "bounded_source_window_before_synapse_provenance_audit",
    ]
    assert report["source_window_limit"] == ledger_limit
    assert report["source_window_count"] == ledger_limit
    assert report["source_record_count"] == source_count
    assert report["source_payload_truncated"] is True
    assert report["source_truncated_count"] == source_count - ledger_limit
    assert report["requested_hash_count"] == 4
    assert report["matched_hash_count"] == 2
    assert report["missing_hash_count"] == 2
    assert set(report["missing_hashes"]) == {
        "events:readout:9",
        "missing-readout-hash",
    }
    assert report["global_candidate_scan"] is False
    assert report["global_score_scan"] is False
    assert report["raw_text_payload_loaded"] is False
    assert report["language_reasoning"] is False
    assert report["runs_live_tick"] is False
    assert report["runs_every_token"] is False
    assert report["mutates_runtime_state"] is False
    assert report["applies_plasticity"] is False
    assert report["archival_storage_device"] == "cpu"
    assert report["lookup_device"] == "cpu"
    assert report["gpu_used"] is False
    assert report["memory_budget"]["max_source_records"] == ledger_limit
    assert report["memory_budget"]["max_requested_hashes"] == 4
    assert report["memory_budget"]["max_returned_events"] == 4
    for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS:
        source = ledger_state[field]
        assert isinstance(source, CountedRows)
        if field == "events":
            assert source.iterated == ledger_limit
        else:
            assert source.iterated == 0


def test_transition_memory_replay_artifact_proposal_uses_internal_readout_evidence() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    record = ledger.record_readout_draft(
        readout_draft=_ready_draft(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )

    proposal = ledger.transition_memory_replay_artifact_proposal(
        mismatch_report={
            "available": True,
            "owned_by_marulho": True,
            "prediction_error": {"mismatch_score": 0.9},
        },
        pressure_report={
            "available": True,
            "owned_by_marulho": True,
            "promotion_gate": {"status": "ready_for_operator_review"},
        },
    )

    assert proposal["surface"] == "snn_transition_memory_replay_artifact_proposal.v1"
    assert proposal["ready"] is True
    assert proposal["mutates_runtime_state"] is False
    assert proposal["replay_window"][0]["readout_evidence_hash"] == record["recorded_event"][
        "readout_evidence_hash"
    ]
    assert proposal["replay_window"][0]["grounded"] is True
    assert proposal["replay_priority_source_window"]["surface"] == (
        "bounded_snn_readout_replay_priority_source_window.v1"
    )
    assert ledger._readout_replay_priority_source_window_bounded(
        proposal["replay_priority_source_window"]
    ) is True
    assert proposal["replay_priority_source_window_hash"] == _sha256_json(
        proposal["replay_priority_source_window"]
    )
    assert proposal["promotion_gate"]["required_evidence"][
        "replay_priority_source_window_bounded"
    ] is True
    assert proposal["promotion_gate"]["eligible_for_operator_recording_review"] is True


def test_transition_memory_replay_artifact_proposal_rejects_partially_grounded_evidence() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    readout_draft = _ready_draft()
    readout_draft["sparse_decode_evidence"]["candidate_matches"][0]["grounded"] = False
    ledger.record_readout_draft(
        readout_draft=readout_draft,
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )

    proposal = ledger.transition_memory_replay_artifact_proposal(
        mismatch_report={
            "available": True,
            "owned_by_marulho": True,
            "prediction_error": {"mismatch_score": 0.9},
        },
        pressure_report={
            "available": True,
            "owned_by_marulho": True,
            "promotion_gate": {"status": "ready_for_operator_review"},
        },
    )

    assert proposal["ready"] is False
    assert proposal["replay_window"] == []
    assert proposal["promotion_gate"]["eligible_for_operator_recording_review"] is False


def test_readout_ledger_rehearsal_evaluation_is_isolated_sparse_snn_review() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    for index, labels in enumerate((["memory pressure"], ["prediction error"])):
        ledger.record_readout_draft(
            readout_draft=_ready_draft_for(
                f"prediction-{index}",
                f"evaluation-{index}",
                "weights-shared",
                labels,
            ),
            expected_state_revision=runtime_state.state_revision,
            operator_id="operator-test",
            confirmation=True,
        )
    priority = ledger.replay_priority(limit=2)
    before = runtime_state.snapshot()

    evaluation = ledger.rehearsal_evaluation(
        priority,
        candidate_limit=2,
        device_evidence={"device": "cpu", "source": "readout_rehearsal_fixture"},
    )
    after = runtime_state.snapshot()

    assert before == after
    assert evaluation["surface"] == "snn_language_readout_rehearsal_evaluation.v1"
    assert evaluation["owned_by_marulho"] is True
    assert evaluation["external_dependency"] is False
    assert evaluation["generates_text"] is False
    assert evaluation["decodes_text"] is False
    assert evaluation["trains_runtime_model"] is False
    assert evaluation["applies_plasticity"] is False
    assert evaluation["mutates_runtime_state"] is False
    assert evaluation["device_evidence"]["tensor_device"] == "cpu"
    assert evaluation["rehearsal_summary"]["candidate_count"] == 2
    assert evaluation["rehearsal_summary"]["activation_sparsity"] >= 0.85
    assert evaluation["promotion_gate"]["eligible_for_operator_rehearsal_review"] is True
    assert evaluation["promotion_gate"]["eligible_for_live_replay"] is False
    assert evaluation["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert evaluation["promotion_gate"]["eligible_for_cognition_substrate"] is False
    assert evaluation["ephemeral_rehearsal"]["runtime_update_applied"] is False
    assert evaluation["ephemeral_rehearsal"]["weights_persisted"] is False


def test_readout_rehearsal_experiment_simulates_pressure_without_mutation() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    for index, labels in enumerate((["memory pressure"], ["prediction error"])):
        ledger.record_readout_draft(
            readout_draft=_ready_draft_for(
                f"prediction-{index}",
                f"evaluation-{index}",
                "weights-shared",
                labels,
            ),
            expected_state_revision=runtime_state.state_revision,
            operator_id="operator-test",
            confirmation=True,
        )
    evaluation = ledger.rehearsal_evaluation(ledger.replay_priority(limit=2), candidate_limit=2)
    before = runtime_state.snapshot()

    experiment = ledger.rehearsal_experiment(evaluation, replay_cycles=4)
    repeated = ledger.rehearsal_experiment(evaluation, replay_cycles=4)
    after = runtime_state.snapshot()

    assert experiment == repeated
    assert before == after
    assert experiment["surface"] == "snn_language_readout_rehearsal_experiment.v1"
    assert experiment["owned_by_marulho"] is True
    assert experiment["external_dependency"] is False
    assert experiment["generates_text"] is False
    assert experiment["decodes_text"] is False
    assert experiment["trains_runtime_model"] is False
    assert experiment["applies_plasticity"] is False
    assert experiment["mutates_runtime_state"] is False
    assert experiment["returns_trained_weights"] is False
    assert experiment["experiment_summary"]["candidate_count"] == 2
    assert experiment["experiment_summary"]["pressure_non_worsening"] is True
    assert experiment["promotion_gate"]["eligible_for_operator_rehearsal_experiment_review"] is True
    assert experiment["promotion_gate"]["eligible_for_live_replay"] is False
    assert experiment["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert experiment["promotion_gate"]["eligible_for_fact_promotion"] is False
    first_trace = experiment["ephemeral_experiment"]["trace"][0]
    assert first_trace["readout_evidence_hash"]
    assert first_trace["prediction_hash"]
    assert first_trace["applied_to_runtime"] is False
    assert first_trace["weights_persisted"] is False
    assert first_trace["generated_text"] is False


def test_readout_replay_design_bounds_future_replay_without_execution() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    for index, labels in enumerate((["memory pressure"], ["prediction error"])):
        ledger.record_readout_draft(
            readout_draft=_ready_draft_for(
                f"prediction-{index}",
                f"evaluation-{index}",
                "weights-shared",
                labels,
            ),
            expected_state_revision=runtime_state.state_revision,
            operator_id="operator-test",
            confirmation=True,
        )
    evaluation = ledger.rehearsal_evaluation(ledger.replay_priority(limit=2), candidate_limit=2)
    experiment = ledger.rehearsal_experiment(evaluation, replay_cycles=4)
    before = runtime_state.snapshot()

    design = ledger.replay_design(
        experiment,
        replay_policy={"max_candidates": 1, "max_replay_cycles": 3, "min_pressure_gain": 0.01},
        rollback_policy={"available": True, "snapshot_id": "snapshot-1"},
    )
    repeated = ledger.replay_design(
        experiment,
        replay_policy={"max_candidates": 1, "max_replay_cycles": 3, "min_pressure_gain": 0.01},
        rollback_policy={"available": True, "snapshot_id": "snapshot-1"},
    )
    after = runtime_state.snapshot()

    assert design == repeated
    assert before == after
    assert design["surface"] == "snn_language_readout_replay_design.v1"
    assert design["owned_by_marulho"] is True
    assert design["generates_text"] is False
    assert design["decodes_text"] is False
    assert design["trains_runtime_model"] is False
    assert design["applies_plasticity"] is False
    assert design["mutates_runtime_state"] is False
    assert design["returns_trained_weights"] is False
    assert design["readout_replay_design"]["selected_candidate_count"] == 1
    assert design["readout_replay_design"]["max_replay_cycles"] == 3
    assert design["readout_replay_design"]["execution_allowed"] is False
    assert design["selected_replay_targets"][0]["readout_evidence_hash"]
    assert design["selected_replay_targets"][0]["transition_memory_evaluation_hash"]
    assert design["selected_replay_targets"][0]["persistent_transition_weights_hash"]
    assert design["selected_replay_targets"][0]["executable"] is False
    assert design["promotion_gate"]["eligible_for_operator_replay_design_review"] is True
    assert design["promotion_gate"]["eligible_for_live_replay"] is False
    assert design["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert design["promotion_gate"]["eligible_for_action"] is False


def test_readout_replay_dry_run_executes_only_isolated_sparse_tensors() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_readout_draft(
        readout_draft=_ready_draft_for(
            "prediction-dry-run",
            "evaluation-dry-run",
            "weights-dry-run",
            ["memory pressure"],
        ),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    evaluation = ledger.rehearsal_evaluation(
        ledger.replay_priority(limit=1),
        candidate_limit=1,
        device_evidence={"device": "cpu", "source": "unit-test"},
    )
    experiment = ledger.rehearsal_experiment(evaluation, replay_cycles=4)
    design = ledger.replay_design(
        experiment,
        replay_policy={"max_candidates": 1, "max_replay_cycles": 3, "min_pressure_gain": 0.01},
        rollback_policy={"available": True, "snapshot_id": "snapshot-1"},
    )
    before = runtime_state.snapshot()

    dry_run = ledger.replay_dry_run(
        design,
        operator_approval=True,
        operator_id="operator-test",
        device_evidence={"device": "cpu", "source": "unit-test"},
    )
    repeated = ledger.replay_dry_run(
        design,
        operator_approval=True,
        operator_id="operator-test",
        device_evidence={"device": "cpu", "source": "unit-test"},
    )
    after = runtime_state.snapshot()

    assert dry_run == repeated
    assert before == after
    assert dry_run["surface"] == "snn_language_readout_replay_dry_run.v1"
    assert dry_run["readout_replay_dry_run_hash"]
    assert dry_run["owned_by_marulho"] is True
    assert dry_run["generates_text"] is False
    assert dry_run["decodes_text"] is False
    assert dry_run["trains_runtime_model"] is False
    assert dry_run["applies_plasticity"] is False
    assert dry_run["mutates_runtime_state"] is False
    assert dry_run["returns_trained_weights"] is False
    assert dry_run["device_evidence"]["tensor_device"] == "cpu"
    assert dry_run["device_evidence"]["cuda_fallback_blocked"] is False
    assert dry_run["replay_target_window"]["surface"] == (
        "bounded_snn_readout_replay_dry_run_target_window.v1"
    )
    assert dry_run["replay_target_window"]["source_window_limit"] == (
        SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
    )
    assert dry_run["replay_target_window"]["source_window_count"] == 1
    assert dry_run["replay_target_window"]["global_candidate_scan"] is False
    assert dry_run["replay_target_window"]["global_score_scan"] is False
    assert dry_run["replay_target_window"]["raw_text_payload_loaded"] is False
    assert dry_run["replay_target_window"]["language_reasoning"] is False
    assert dry_run["replay_target_window"]["archival_storage_device"] == "cpu"
    assert dry_run["replay_target_window"]["active_replay_computation_device"] == "cpu"
    assert dry_run["promotion_gate"]["required_evidence"]["replay_target_window_bounded"] is True
    assert dry_run["isolated_replay_summary"]["target_count"] == 1
    assert dry_run["isolated_replay_summary"]["pressure_non_worsening"] is True
    assert dry_run["ephemeral_replay"]["runtime_update_applied"] is False
    assert dry_run["ephemeral_replay"]["weights_persisted"] is False
    assert dry_run["ephemeral_replay"]["checkpoint_written"] is False
    assert dry_run["ephemeral_replay"]["trace"][0]["checkpoint_written"] is False
    assert dry_run["promotion_gate"]["eligible_for_operator_replay_dry_run_review"] is True
    assert dry_run["promotion_gate"]["eligible_for_live_replay"] is False
    assert dry_run["promotion_gate"]["eligible_for_plasticity_application"] is False


def test_readout_replay_dry_run_bounds_untrusted_target_payload() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_readout_draft(
        readout_draft=_ready_draft_for(
            "prediction-wide-dry-run",
            "evaluation-wide-dry-run",
            "weights-wide-dry-run",
            ["memory pressure"],
        ),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    evaluation = ledger.rehearsal_evaluation(ledger.replay_priority(limit=1), candidate_limit=1)
    experiment = ledger.rehearsal_experiment(evaluation, replay_cycles=4)
    design = ledger.replay_design(
        experiment,
        replay_policy={"max_candidates": 1, "max_replay_cycles": 3, "min_pressure_gain": 0.01},
        rollback_policy={"available": True, "snapshot_id": "snapshot-1"},
    )
    oversized = deepcopy(design)
    oversized["selected_replay_targets"] = list(design["selected_replay_targets"]) * (
        SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT * 4
    )
    before = runtime_state.snapshot()

    dry_run = ledger.replay_dry_run(
        oversized,
        operator_approval=True,
        operator_id="operator-test",
        device_evidence={"device": "cpu", "source": "unit-test"},
    )
    after = runtime_state.snapshot()

    assert before == after
    assert dry_run["isolated_replay_summary"]["target_count"] == (
        SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
    )
    assert len(dry_run["ephemeral_replay"]["trace"]) == SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
    assert dry_run["replay_target_source_count"] == SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT * 4
    assert dry_run["replay_target_window"]["source_truncated_count"] == (
        SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT * 3
    )
    assert dry_run["replay_target_window"]["source_window_count"] == (
        SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
    )
    assert dry_run["replay_target_window"]["global_candidate_scan"] is False
    assert dry_run["replay_target_window"]["runs_live_tick"] is False
    assert dry_run["replay_target_window"]["runs_every_token"] is False
    assert dry_run["replay_target_window"]["language_reasoning"] is False
    assert dry_run["promotion_gate"]["required_evidence"]["replay_target_window_bounded"] is True
    assert dry_run["promotion_gate"]["eligible_for_operator_replay_dry_run_review"] is True


def test_readout_replay_dry_run_blocks_without_operator_or_device_evidence() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )

    dry_run = ledger.replay_dry_run({"surface": "wrong.v1"})

    assert dry_run["promotion_gate"]["status"] == "blocked_missing_readout_replay_dry_run_evidence"
    assert dry_run["promotion_gate"]["eligible_for_operator_replay_dry_run_review"] is False
    assert dry_run["promotion_gate"]["required_evidence"]["operator_approval"] is False
    assert dry_run["promotion_gate"]["required_evidence"]["device_evidence_available"] is False
    assert dry_run["mutates_runtime_state"] is False


def test_readout_plasticity_preflight_reviews_dry_run_without_applying_weights() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_readout_draft(
        readout_draft=_ready_draft_for(
            "prediction-preflight",
            "evaluation-preflight",
            "weights-preflight",
            ["memory pressure", "prediction error", "transition support"],
        ),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    evaluation = ledger.rehearsal_evaluation(
        ledger.replay_priority(limit=1),
        candidate_limit=1,
        device_evidence={"device": "cpu", "source": "unit-test"},
    )
    experiment = ledger.rehearsal_experiment(evaluation, replay_cycles=6)
    design = ledger.replay_design(
        experiment,
        replay_policy={"max_candidates": 1, "max_replay_cycles": 6, "min_pressure_gain": 0.01},
        rollback_policy={"available": True, "snapshot_id": "snapshot-1"},
    )
    dry_run = ledger.replay_dry_run(
        design,
        operator_approval=True,
        operator_id="operator-test",
        device_evidence={"device": "cpu", "source": "unit-test"},
    )
    before = runtime_state.snapshot()

    preflight = ledger.plasticity_preflight(
        dry_run,
        plasticity_policy={
            "learning_rate": 0.02,
            "max_weight_delta": 0.03,
            "locality_radius": 8,
            "max_candidate_synapses": 16,
            "normalization": True,
            "local_only": True,
        },
        runtime_truth_delta={"improved_or_stable": True},
        rollback_policy={"available": True, "snapshot_id": "snapshot-1"},
    )
    repeated = ledger.plasticity_preflight(
        dry_run,
        plasticity_policy={
            "learning_rate": 0.02,
            "max_weight_delta": 0.03,
            "locality_radius": 8,
            "max_candidate_synapses": 16,
            "normalization": True,
            "local_only": True,
        },
        runtime_truth_delta={"improved_or_stable": True},
        rollback_policy={"available": True, "snapshot_id": "snapshot-1"},
    )
    after = runtime_state.snapshot()

    assert preflight == repeated
    assert before == after
    assert preflight["surface"] == "snn_language_readout_plasticity_preflight.v1"
    assert preflight["readout_replay_dry_run_hash"] == dry_run["readout_replay_dry_run_hash"]
    assert preflight["device_evidence"]["tensor_device"] == "cpu"
    assert preflight["replay_trace_window"]["surface"] == (
        "bounded_snn_readout_plasticity_preflight_trace_window.v1"
    )
    assert preflight["replay_trace_window"]["source_window_count"] == 1
    assert preflight["replay_trace_window"]["archival_storage_device"] == "cpu"
    assert preflight["replay_trace_window"]["global_candidate_scan"] is False
    assert preflight["promotion_gate"]["required_evidence"]["replay_trace_window_bounded"] is True
    assert preflight["readout_plasticity_preflight_hash"]
    assert preflight["owned_by_marulho"] is True
    assert preflight["generates_text"] is False
    assert preflight["decodes_text"] is False
    assert preflight["trains_runtime_model"] is False
    assert preflight["applies_plasticity"] is False
    assert preflight["mutates_runtime_state"] is False
    assert preflight["returns_trained_weights"] is False
    assert preflight["plasticity_preflight"]["candidate_synapse_count"] > 0
    assert preflight["plasticity_preflight"]["runtime_update_applied"] is False
    assert preflight["plasticity_preflight"]["weights_persisted"] is False
    assert preflight["plasticity_preflight"]["checkpoint_written"] is False
    assert preflight["candidate_replay_sequences"][0]["runtime_update_applied"] is False
    assert preflight["promotion_gate"]["eligible_for_operator_readout_plasticity_review"] is True
    assert preflight["promotion_gate"]["eligible_for_plasticity_application_design"] is True
    assert preflight["promotion_gate"]["next_gate"] == "operator_review_readout_to_plasticity_replay_bridge"
    assert preflight["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert preflight["promotion_gate"]["eligible_for_fact_promotion"] is False


def test_readout_plasticity_preflight_blocks_missing_dry_run_evidence() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )

    preflight = ledger.plasticity_preflight({"surface": "wrong.v1"})

    assert preflight["promotion_gate"]["status"] == "blocked_missing_readout_plasticity_preflight_evidence"
    assert preflight["promotion_gate"]["eligible_for_operator_readout_plasticity_review"] is False
    assert preflight["promotion_gate"]["eligible_for_plasticity_application_design"] is False
    assert preflight["promotion_gate"]["required_evidence"]["dry_run_gate_ready"] is False
    assert preflight["promotion_gate"]["required_evidence"]["rollback_policy_available"] is False
    assert preflight["mutates_runtime_state"] is False


def test_readout_plasticity_replay_bridge_emits_existing_replay_experiment_contract() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_readout_draft(
        readout_draft=_ready_draft_for(
            "prediction-bridge",
            "evaluation-bridge",
            "weights-bridge",
            ["memory pressure", "prediction error", "transition support"],
        ),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    evaluation = ledger.rehearsal_evaluation(
        ledger.replay_priority(limit=1),
        candidate_limit=1,
        device_evidence={"device": "cpu", "source": "unit-test"},
    )
    experiment = ledger.rehearsal_experiment(evaluation, replay_cycles=6)
    design = ledger.replay_design(
        experiment,
        replay_policy={"max_candidates": 1, "max_replay_cycles": 6, "min_pressure_gain": 0.01},
        rollback_policy={"available": True, "snapshot_id": "snapshot-1"},
    )
    dry_run = ledger.replay_dry_run(
        design,
        operator_approval=True,
        operator_id="operator-test",
        device_evidence={"device": "cpu", "source": "unit-test"},
    )
    preflight = ledger.plasticity_preflight(
        dry_run,
        plasticity_policy={"locality_radius": 8},
        runtime_truth_delta={"improved_or_stable": True},
        rollback_policy={"available": True, "snapshot_id": "snapshot-1"},
    )
    before = runtime_state.snapshot()

    bridge = ledger.plasticity_replay_bridge(
        preflight,
        runtime_truth_delta={"improved_or_stable": True},
        rollback_policy={"available": True, "snapshot_id": "snapshot-1"},
    )
    repeated = ledger.plasticity_replay_bridge(
        preflight,
        runtime_truth_delta={"improved_or_stable": True},
        rollback_policy={"available": True, "snapshot_id": "snapshot-1"},
    )
    after = runtime_state.snapshot()

    assert bridge == repeated
    assert before == after
    assert bridge["artifact_kind"] == "terminus_snn_language_plasticity_replay_experiment"
    assert bridge["surface"] == "snn_language_plasticity_replay_experiment.v1"
    assert bridge["source"] == "service.snn_language_readout_ledger.plasticity_replay_bridge"
    assert bridge["readout_replay_dry_run_hash"] == dry_run["readout_replay_dry_run_hash"]
    assert bridge["readout_plasticity_replay_bridge_hash"]
    assert bridge["device_evidence"]["tensor_device"] == "cpu"
    assert bridge["device_evidence"]["device_report_available"] is True
    assert bridge["replay_sequence_window"]["surface"] == (
        "bounded_snn_readout_plasticity_bridge_sequence_window.v1"
    )
    assert bridge["replay_sequence_window"]["source_window_count"] == bridge["replay_experiment"]["replay_sequence_count"]
    assert bridge["replay_sequence_window"]["global_candidate_scan"] is False
    assert bridge["replay_sequence_window"]["global_score_scan"] is False
    assert bridge["replay_sequence_window"]["archival_storage_device"] == "cpu"
    assert bridge["promotion_gate"]["required_evidence"]["replay_sequence_window_bounded"] is True
    assert bridge["generates_text"] is False
    assert bridge["decodes_text"] is False
    assert bridge["trains_runtime_model"] is False
    assert bridge["applies_plasticity"] is False
    assert bridge["mutates_runtime_state"] is False
    assert bridge["returns_trained_weights"] is False
    assert bridge["replay_experiment"]["replay_sequence_count"] > 0
    assert bridge["replay_experiment"]["grounded_replay_coverage"] >= 0.5
    assert bridge["replay_experiment"]["pressure_stable_after_replay"] is True
    assert bridge["application_design"]["learning_rate"] == preflight["plasticity_preflight"]["learning_rate"]
    assert bridge["application_design"]["max_weight_delta"] == preflight["plasticity_preflight"]["max_weight_delta"]
    assert bridge["application_design"]["locality_radius"] == preflight["plasticity_preflight"]["locality_radius"]
    assert bridge["application_design"]["normalization"] is True
    assert bridge["application_design"]["local_only"] is True
    assert bridge["application_design"]["runtime_update_applied"] is False
    assert bridge["application_design"]["weights_persisted"] is False
    assert bridge["application_target_hint"]["target_id"] == "marulho.snn_language.sparse_transition_weights"
    assert bridge["application_target_hint"]["checkpointed"] is True
    assert bridge["checkpoint_transaction_requirements"]["restore_verification_required"] is True
    assert bridge["canonical_replay_sequences"][0]["pre_indices"]
    assert bridge["canonical_replay_sequences"][0]["post_indices"]
    assert bridge["canonical_replay_sequences"][0]["runtime_update_applied"] is False
    assert bridge["ephemeral_replay"]["runtime_update_applied"] is False
    assert bridge["ephemeral_replay"]["weights_persisted"] is False
    assert bridge["ephemeral_replay"]["checkpoint_written"] is False
    assert bridge["promotion_gate"]["eligible_for_operator_application_review"] is True
    assert bridge["promotion_gate"]["eligible_for_plasticity_application"] is False


def test_readout_plasticity_replay_bridge_bounds_untrusted_sequence_payload() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_readout_draft(
        readout_draft=_ready_draft_for(
            "prediction-wide-bridge",
            "evaluation-wide-bridge",
            "weights-wide-bridge",
            ["memory pressure", "prediction error", "transition support"],
        ),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    evaluation = ledger.rehearsal_evaluation(ledger.replay_priority(limit=1), candidate_limit=1)
    experiment = ledger.rehearsal_experiment(evaluation, replay_cycles=6)
    design = ledger.replay_design(
        experiment,
        replay_policy={"max_candidates": 1, "max_replay_cycles": 6, "min_pressure_gain": 0.01},
        rollback_policy={"available": True, "snapshot_id": "snapshot-1"},
    )
    dry_run = ledger.replay_dry_run(
        design,
        operator_approval=True,
        operator_id="operator-test",
        device_evidence={"device": "cpu", "source": "unit-test"},
    )
    preflight = ledger.plasticity_preflight(
        dry_run,
        plasticity_policy={"locality_radius": 8},
        runtime_truth_delta={"improved_or_stable": True},
        rollback_policy={"available": True, "snapshot_id": "snapshot-1"},
    )
    oversized = deepcopy(preflight)
    oversized["candidate_replay_sequences"] = list(preflight["candidate_replay_sequences"]) * (
        SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT * 4
    )
    before = runtime_state.snapshot()

    bridge = ledger.plasticity_replay_bridge(
        oversized,
        runtime_truth_delta={"improved_or_stable": True},
        rollback_policy={"available": True, "snapshot_id": "snapshot-1"},
    )
    after = runtime_state.snapshot()

    assert before == after
    assert bridge["replay_experiment"]["replay_sequence_count"] == (
        SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
    )
    assert len(bridge["canonical_replay_sequences"]) == SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
    assert bridge["replay_sequence_source_count"] == (
        len(preflight["candidate_replay_sequences"])
        * SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
        * 4
    )
    assert bridge["replay_sequence_window"]["source_window_count"] == (
        SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
    )
    assert bridge["replay_sequence_window"]["source_truncated_count"] == (
        bridge["replay_sequence_source_count"] - SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
    )
    assert bridge["replay_sequence_window"]["runs_live_tick"] is False
    assert bridge["replay_sequence_window"]["runs_every_token"] is False
    assert bridge["replay_sequence_window"]["language_reasoning"] is False
    assert bridge["promotion_gate"]["required_evidence"]["replay_sequence_window_bounded"] is True
    assert bridge["promotion_gate"]["eligible_for_operator_application_review"] is True


def test_readout_plasticity_replay_bridge_blocks_missing_preflight() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )

    bridge = ledger.plasticity_replay_bridge({"surface": "wrong.v1"})

    assert bridge["surface"] == "snn_language_plasticity_replay_experiment.v1"
    assert bridge["promotion_gate"]["status"] == "blocked_missing_replay_experiment_evidence"
    assert bridge["promotion_gate"]["eligible_for_operator_application_review"] is False
    assert bridge["promotion_gate"]["required_evidence"]["preflight_gate_ready"] is False
    assert bridge["mutates_runtime_state"] is False


def test_readout_synapse_provenance_audit_uses_dynamic_neuron_capacity() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: {},
    )

    audit = ledger.synapse_provenance_audit(
        plasticity_runtime_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "language_capacity": {
                "surface": "snn_language_capacity_state.v1",
                "language_neuron_count": 66,
                "sparse_edge_budget": 258,
                "outgoing_fanout_budget": 16,
                "dynamic_capacity_enabled": True,
            },
            "sparse_transition_weights": {"64:65": 0.03},
            "synapse_provenance_by_key": {
                "64:65": {
                    "readout_evidence_hash": "missing-ledger-row",
                    "prediction_hash": "prediction-dynamic",
                    "transition_memory_evaluation_hash": "evaluation-dynamic",
                    "persistent_transition_weights_hash": "weights-dynamic",
                    "source_pre_indices": [64],
                    "source_post_indices": [65],
                    "source_active_indices": [64, 65],
                }
            },
        }
    )

    row = audit["audited_synapses"][0]
    assert row["synapse_key"] == "64:65"
    assert row["canonical_synapse_key"] is True
    assert row["synapse_indices_in_range"] is True
    assert row["source_indices_in_range"] is True
    assert row["source_indices_match_synapse"] is True
    assert audit["ledger_event_source_window"]["surface"] == (
        "bounded_snn_readout_evidence_event_map_source_window.v1"
    )
    assert audit["ledger_event_source_window"][
        "policy"
    ] == SNN_READOUT_EVIDENCE_HASH_SOURCE_WINDOW_POLICY
    assert audit["ledger_event_source_window"]["requested_hash_count"] == 1
    assert audit["ledger_event_source_window"]["matched_hash_count"] == 0
    assert audit["ledger_event_source_window"]["missing_hash_count"] == 1
    assert audit["ledger_event_source_window"]["global_candidate_scan"] is False
    assert audit["ledger_event_source_window"]["runs_live_tick"] is False
    assert audit["ledger_event_source_window"]["archival_storage_device"] == "cpu"
    assert audit["promotion_gate"]["required_evidence"][
        "ledger_event_source_window_bounded"
    ] is True


def test_readout_synapse_provenance_audit_uses_bounded_applied_synapse_source_window() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: {},
    )
    source_limit = SNN_READOUT_SYNAPSE_PROVENANCE_AUDIT_SOURCE_WINDOW_LIMIT
    retained_count = source_limit + 16
    weights: dict[str, float] = {}
    provenance: dict[str, dict[str, object]] = {}
    for index in range(retained_count):
        key = f"{index}:{index + 1}"
        weights[key] = 0.03
        provenance[key] = {
            "readout_evidence_hash": f"missing-ledger-row-{index:03d}",
            "prediction_hash": f"prediction-{index:03d}",
            "transition_memory_evaluation_hash": f"evaluation-{index:03d}",
            "persistent_transition_weights_hash": f"weights-{index:03d}",
            "source_pre_indices": [index],
            "source_post_indices": [index + 1],
            "source_active_indices": [index, index + 1],
        }

    audit = ledger.synapse_provenance_audit(
        plasticity_runtime_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "language_capacity": {
                "surface": "snn_language_capacity_state.v1",
                "language_neuron_count": retained_count + 1,
                "sparse_edge_budget": retained_count + 2,
                "outgoing_fanout_budget": 16,
                "dynamic_capacity_enabled": True,
            },
            "sparse_transition_weights": weights,
            "synapse_provenance_by_key": provenance,
        },
        limit=source_limit,
    )

    source_window = audit["applied_synapse_audit_source_window"]
    assert source_window["surface"] == (
        "bounded_snn_readout_synapse_provenance_audit_source_window.v1"
    )
    assert source_window["policy"] == SNN_READOUT_SYNAPSE_PROVENANCE_AUDIT_SOURCE_WINDOW_POLICY
    assert source_window["source_window_limit"] == source_limit
    assert source_window["source_sparse_weight_rows"] == source_limit
    assert source_window["source_synapse_provenance_rows"] == source_limit
    assert source_window["retained_sparse_weight_rows"] == retained_count
    assert source_window["retained_synapse_provenance_rows"] == retained_count
    assert source_window["source_payload_truncated"] is True
    assert source_window["source_window_complete"] is False
    assert source_window["source_truncated_counts"]["sparse_transition_weights"] == 16
    assert source_window["source_truncated_counts"]["synapse_provenance_by_key"] == 16
    assert source_window["global_candidate_scan"] is False
    assert source_window["global_score_scan"] is False
    assert source_window["runs_live_tick"] is False
    assert source_window["runs_every_token"] is False
    assert source_window["language_reasoning"] is False
    assert source_window["archival_storage_device"] == "cpu"
    assert source_window["gpu_resident_archival_metadata"] is False
    assert audit["audit_summary"]["audited_synapse_count"] == source_limit
    assert audit["audit_summary"]["returned_synapse_count"] == source_limit
    assert audit["audit_summary"]["provenanced_synapse_count"] == retained_count
    assert audit["audit_summary"]["sparse_transition_weight_count"] == retained_count
    assert audit["ledger_event_source_window"]["requested_hash_count"] == source_limit
    assert audit["ledger_event_source_window"]["matched_hash_count"] == 0
    assert audit["audited_synapses"][0]["synapse_key"] == "0:1"
    assert audit["audited_synapses"][-1]["synapse_key"] == f"{source_limit - 1}:{source_limit}"
    assert audit["promotion_gate"]["eligible_for_readout_synapse_audit_review"] is False
    assert audit["promotion_gate"]["required_evidence"][
        "applied_synapse_audit_source_window_bounded"
    ] is True
    assert audit["promotion_gate"]["required_evidence"][
        "applied_synapse_audit_source_window_complete"
    ] is False


def test_readout_synapse_provenance_audit_uses_runtime_source_window_retained_counts() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: {},
    )
    source_limit = SNN_READOUT_SYNAPSE_PROVENANCE_AUDIT_SOURCE_WINDOW_LIMIT
    retained_count = source_limit + 23
    weights = {f"{index}:{index + 1}": 0.03 for index in range(source_limit)}
    provenance = {
        key: {
            "readout_evidence_hash": f"missing-ledger-row-{index:03d}",
            "prediction_hash": f"prediction-{index:03d}",
            "transition_memory_evaluation_hash": f"evaluation-{index:03d}",
            "persistent_transition_weights_hash": f"weights-{index:03d}",
            "source_pre_indices": [index],
            "source_post_indices": [index + 1],
            "source_active_indices": [index, index + 1],
        }
        for index, key in enumerate(weights)
    }

    audit = ledger.synapse_provenance_audit(
        plasticity_runtime_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "language_capacity": {
                "surface": "snn_language_capacity_state.v1",
                "language_neuron_count": retained_count + 1,
                "sparse_edge_budget": retained_count + 2,
                "outgoing_fanout_budget": 16,
                "dynamic_capacity_enabled": True,
            },
            "sparse_transition_weights": weights,
            "synapse_provenance_by_key": provenance,
            "transition_memory_source_window": {
                "surface": (
                    "bounded_snn_language_plasticity_runtime_"
                    "transition_memory_source_window.v1"
                ),
                "source_counts": {
                    "retained_sparse_transition_weights": retained_count,
                    "retained_synapse_provenance_rows": retained_count,
                    "source_sparse_transition_weights": source_limit,
                    "source_synapse_provenance_rows": source_limit,
                },
                "retained_sparse_transition_weight_rows": retained_count,
                "retained_synapse_provenance_rows": retained_count,
            },
        },
        limit=source_limit,
    )

    source_window = audit["applied_synapse_audit_source_window"]
    assert source_window["retained_sparse_weight_rows"] == retained_count
    assert source_window["retained_synapse_provenance_rows"] == retained_count
    assert source_window["source_sparse_weight_rows"] == source_limit
    assert source_window["source_synapse_provenance_rows"] == source_limit
    assert source_window["source_payload_truncated"] is True
    assert source_window["source_window_complete"] is False
    assert source_window["source_truncated_counts"]["sparse_transition_weights"] == 23
    assert source_window["source_truncated_counts"]["synapse_provenance_by_key"] == 23
    assert audit["audit_summary"]["audited_synapse_count"] == source_limit
    assert audit["audit_summary"]["sparse_transition_weight_count"] == retained_count
    assert audit["audit_summary"]["provenanced_synapse_count"] == retained_count
    assert audit["promotion_gate"]["eligible_for_readout_synapse_audit_review"] is False


def test_readout_synapse_provenance_audit_checks_runtime_weights_against_ledger() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    record = ledger.record_readout_draft(
        readout_draft=_ready_draft_for(
            "prediction-audit",
            "evaluation-audit",
            "weights-audit",
            ["memory pressure"],
        ),
        expected_state_revision=runtime_state.state_revision,
        operator_id="operator-test",
        confirmation=True,
    )
    evidence_hash = record["recorded_event"]["readout_evidence_hash"]

    audit = ledger.synapse_provenance_audit(
        plasticity_runtime_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "sparse_transition_weights": {"1:2": 0.03},
            "synapse_provenance_by_key": {
                "1:2": {
                    "readout_evidence_hash": evidence_hash,
                    "prediction_hash": "prediction-audit",
                    "transition_memory_evaluation_hash": "evaluation-audit",
                    "persistent_transition_weights_hash": "weights-audit",
                    "source_pre_indices": [1],
                    "source_post_indices": [2],
                    "source_active_indices": [1, 2],
                }
            },
        }
    )
    mismatched = ledger.synapse_provenance_audit(
        plasticity_runtime_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "sparse_transition_weights": {"1:2": 0.03},
            "synapse_provenance_by_key": {
                "1:2": {
                    "readout_evidence_hash": evidence_hash,
                    "prediction_hash": "wrong-prediction",
                    "transition_memory_evaluation_hash": "evaluation-audit",
                    "persistent_transition_weights_hash": "weights-audit",
                    "source_pre_indices": [1],
                    "source_post_indices": [2],
                    "source_active_indices": [1, 2],
                }
            },
        }
    )
    blocked = ledger.synapse_provenance_audit(
        plasticity_runtime_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "sparse_transition_weights": {"1:2": 0.03},
            "synapse_provenance_by_key": {},
        }
    )
    malformed = ledger.synapse_provenance_audit(
        plasticity_runtime_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "sparse_transition_weights": {"01:65": 1.5},
            "synapse_provenance_by_key": {
                "01:65": {
                    "readout_evidence_hash": evidence_hash,
                    "prediction_hash": "prediction-audit",
                    "transition_memory_evaluation_hash": "evaluation-audit",
                    "persistent_transition_weights_hash": "weights-audit",
                    "source_pre_indices": [1],
                    "source_post_indices": [65],
                    "source_active_indices": [1, 65],
                }
            },
        }
    )
    non_numeric = ledger.synapse_provenance_audit(
        plasticity_runtime_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "sparse_transition_weights": {"1:2": None},
            "synapse_provenance_by_key": {
                "1:2": {
                    "readout_evidence_hash": evidence_hash,
                    "prediction_hash": "prediction-audit",
                    "transition_memory_evaluation_hash": "evaluation-audit",
                    "persistent_transition_weights_hash": "weights-audit",
                    "source_pre_indices": [1],
                    "source_post_indices": [2],
                    "source_active_indices": [1, 2],
                }
            },
        }
    )
    replay_regeneration = ledger.synapse_provenance_audit(
        plasticity_runtime_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "sparse_transition_weights": {"1:3": 0.03},
            "synapse_provenance_by_key": {
                "1:3": {
                    "provenance_type": "replay_regeneration",
                    "permit_id": "permit-1",
                    "replay_artifact_id": "artifact-1",
                    "replay_artifact_hash": "artifact-hash-1",
                    "replay_window_hash": "window-hash-1",
                    "readout_evidence_hashes": [evidence_hash],
                    "source_metadata_hash": "source-metadata-hash-1",
                    "emission_lineage": {
                        "emission_hash": "emission-hash-1",
                        "readout_evidence_hash": evidence_hash,
                        "prediction_hash": "prediction-audit",
                        "design_hash": "design-hash-1",
                    },
                    "local_edge_provenance": {
                        "source_synapse_id": "snn-rollout-local:1:3:0",
                        "source_trace_index": 0,
                        "source_rollout_step_index": 10,
                        "target_rollout_step_index": 20,
                        "source_active_indices_hash": "source-active-hash-1",
                        "target_active_indices_hash": "target-active-hash-1",
                    },
                }
            },
        }
    )
    replay_regeneration_matching_restore = ledger.synapse_provenance_audit(
        plasticity_runtime_state=deepcopy(
            {
                "surface": "snn_language_plasticity_runtime_state.v1",
                "owned_by_marulho": True,
                "sparse_transition_weights": {"1:3": 0.03},
                "synapse_provenance_by_key": {
                    "1:3": {
                        "provenance_type": "replay_regeneration",
                        "permit_id": "permit-1",
                        "replay_artifact_id": "artifact-1",
                        "replay_artifact_hash": "artifact-hash-1",
                        "replay_window_hash": "window-hash-1",
                        "readout_evidence_hashes": [evidence_hash],
                        "source_metadata_hash": "source-metadata-hash-1",
                        "emission_lineage": {
                            "emission_hash": "emission-hash-1",
                            "readout_evidence_hash": evidence_hash,
                            "prediction_hash": "prediction-audit",
                        },
                        "local_edge_provenance": {
                            "source_synapse_id": "snn-rollout-local:1:3:0",
                            "source_rollout_step_index": 10,
                            "target_rollout_step_index": 20,
                            "source_active_indices_hash": "source-active-hash-1",
                            "target_active_indices_hash": "target-active-hash-1",
                        },
                    }
                },
            }
        ),
        applied_replay_lineage_restore_validation={
            "surface": "snn_applied_replay_lineage_restore_validation.v1",
            "summary_matches_restored_state": True,
        },
    )
    replay_regeneration_mismatched_restore = ledger.synapse_provenance_audit(
        plasticity_runtime_state=deepcopy(
            {
                "surface": "snn_language_plasticity_runtime_state.v1",
                "owned_by_marulho": True,
                "sparse_transition_weights": {"1:3": 0.03},
                "synapse_provenance_by_key": {
                    "1:3": {
                        "provenance_type": "replay_regeneration",
                        "permit_id": "permit-1",
                        "replay_artifact_id": "artifact-1",
                        "replay_artifact_hash": "artifact-hash-1",
                        "replay_window_hash": "window-hash-1",
                        "readout_evidence_hashes": [evidence_hash],
                        "source_metadata_hash": "source-metadata-hash-1",
                        "emission_lineage": {
                            "emission_hash": "emission-hash-1",
                            "readout_evidence_hash": evidence_hash,
                            "prediction_hash": "prediction-audit",
                        },
                        "local_edge_provenance": {
                            "source_synapse_id": "snn-rollout-local:1:3:0",
                            "source_rollout_step_index": 10,
                            "target_rollout_step_index": 20,
                            "source_active_indices_hash": "source-active-hash-1",
                            "target_active_indices_hash": "target-active-hash-1",
                        },
                    }
                },
            }
        ),
        applied_replay_lineage_restore_validation={
            "surface": "snn_applied_replay_lineage_restore_validation.v1",
            "summary_matches_restored_state": False,
        },
    )
    replay_incomplete_lineage = ledger.synapse_provenance_audit(
        plasticity_runtime_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "sparse_transition_weights": {"1:3": 0.03},
            "synapse_provenance_by_key": {
                "1:3": {
                    "provenance_type": "replay_regeneration",
                    "permit_id": "permit-1",
                    "replay_artifact_id": "artifact-1",
                    "replay_artifact_hash": "artifact-hash-1",
                    "replay_window_hash": "window-hash-1",
                    "readout_evidence_hashes": [evidence_hash],
                    "source_metadata_hash": "source-metadata-hash-1",
                    "emission_lineage": {"emission_hash": "emission-hash-1"},
                    "local_edge_provenance": {
                        "source_synapse_id": "snn-rollout-local:1:3:0",
                        "source_trace_index": 0,
                        "source_rollout_step_index": 10,
                        "target_rollout_step_index": 20,
                        "source_active_indices_hash": "source-active-hash-1",
                        "target_active_indices_hash": "target-active-hash-1",
                    },
                }
            },
        }
    )
    replay_missing_local_edge = ledger.synapse_provenance_audit(
        plasticity_runtime_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "sparse_transition_weights": {"1:3": 0.03},
            "synapse_provenance_by_key": {
                "1:3": {
                    "provenance_type": "replay_regeneration",
                    "permit_id": "permit-1",
                    "replay_artifact_id": "artifact-1",
                    "replay_artifact_hash": "artifact-hash-1",
                    "replay_window_hash": "window-hash-1",
                    "readout_evidence_hashes": [evidence_hash],
                }
            },
        }
    )
    ledger_state["events"][0]["labels"] = ["tampered material"]
    tampered_ledger = ledger.synapse_provenance_audit(
        plasticity_runtime_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_marulho": True,
            "sparse_transition_weights": {"1:2": 0.03},
            "synapse_provenance_by_key": {
                "1:2": {
                    "readout_evidence_hash": evidence_hash,
                    "prediction_hash": "prediction-audit",
                    "transition_memory_evaluation_hash": "evaluation-audit",
                    "persistent_transition_weights_hash": "weights-audit",
                    "source_pre_indices": [1],
                    "source_post_indices": [2],
                    "source_active_indices": [1, 2],
                }
            },
        }
    )

    assert audit["surface"] == "snn_language_readout_synapse_provenance_audit.v1"
    assert audit["mutates_runtime_state"] is False
    assert audit["applies_plasticity"] is False
    assert audit["audit_summary"]["audited_synapse_count"] == 1
    assert audit["audit_summary"]["unbounded_weight_count"] == 0
    assert audit["audited_synapses"][0]["ledger_evidence_present"] is True
    assert audit["audited_synapses"][0]["ledger_field_match"] is True
    assert audit["audited_synapses"][0]["ledger_hash_valid"] is True
    assert audit["audited_synapses"][0]["canonical_synapse_key"] is True
    assert audit["audited_synapses"][0]["synapse_indices_in_range"] is True
    assert audit["audited_synapses"][0]["weight_finite"] is True
    assert audit["audited_synapses"][0]["weight_bounded"] is True
    assert audit["audited_synapses"][0]["source_indices_match_synapse"] is True
    assert audit["ledger_event_source_window"]["requested_hash_count"] == 1
    assert audit["ledger_event_source_window"]["matched_hash_count"] == 1
    assert audit["ledger_event_source_window"]["missing_hash_count"] == 0
    assert audit["audit_summary"]["ledger_event_source_window_count"] <= audit[
        "ledger_event_source_window"
    ]["source_window_limit"]
    assert audit["audit_summary"]["ledger_event_requested_hash_count"] == 1
    assert audit["audit_summary"]["ledger_event_matched_hash_count"] == 1
    assert audit["audit_summary"]["ledger_event_missing_hash_count"] == 0
    assert audit["promotion_gate"]["required_evidence"][
        "ledger_event_source_window_bounded"
    ] is True
    assert audit["applied_synapse_audit_source_window"]["source_window_complete"] is True
    assert audit["promotion_gate"]["required_evidence"][
        "applied_synapse_audit_source_window_bounded"
    ] is True
    assert audit["promotion_gate"]["required_evidence"][
        "applied_synapse_audit_source_window_complete"
    ] is True
    assert audit["promotion_gate"]["eligible_for_readout_synapse_audit_review"] is True
    assert blocked["promotion_gate"]["eligible_for_readout_synapse_audit_review"] is False
    assert blocked["promotion_gate"]["required_evidence"]["synapse_provenance_available"] is False
    assert blocked["ledger_event_source_window"]["requested_hash_count"] == 0
    assert blocked["promotion_gate"]["required_evidence"][
        "ledger_event_source_window_bounded"
    ] is True
    assert mismatched["promotion_gate"]["eligible_for_readout_synapse_audit_review"] is False
    assert mismatched["promotion_gate"]["required_evidence"]["audited_synapses_match_ledger_fields"] is False
    assert malformed["promotion_gate"]["eligible_for_readout_synapse_audit_review"] is False
    assert malformed["promotion_gate"]["required_evidence"]["audited_synapses_have_bounded_weights"] is False
    assert malformed["promotion_gate"]["required_evidence"]["audited_synapses_have_canonical_keys"] is False
    assert malformed["promotion_gate"]["required_evidence"]["audited_synapse_indices_in_range"] is False
    assert malformed["promotion_gate"]["required_evidence"]["audited_source_indices_in_range"] is False
    assert non_numeric["promotion_gate"]["eligible_for_readout_synapse_audit_review"] is False
    assert non_numeric["promotion_gate"]["required_evidence"]["audited_synapses_have_finite_weights"] is False
    assert replay_regeneration["promotion_gate"]["eligible_for_readout_synapse_audit_review"] is True
    assert replay_regeneration["ledger_event_source_window"][
        "requested_hash_count"
    ] == 1
    assert replay_regeneration["ledger_event_source_window"][
        "matched_hash_count"
    ] == 1
    assert replay_regeneration["audit_summary"]["replay_regeneration_synapse_count"] == 1
    assert replay_regeneration["audit_summary"]["local_edge_provenance_count"] == 1
    assert replay_regeneration["audit_summary"]["complete_local_edge_provenance_count"] == 1
    assert replay_regeneration["audit_summary"]["replay_artifact_lineage_count"] == 1
    assert replay_regeneration["audit_summary"]["complete_replay_artifact_lineage_count"] == 1
    assert replay_regeneration["audit_summary"]["restore_validation_available"] is False
    assert replay_regeneration["audit_summary"]["restore_validation_blocks_audit"] is False
    assert replay_regeneration["audited_synapses"][0]["source_metadata_hash"] == "source-metadata-hash-1"
    assert replay_regeneration["audited_synapses"][0]["emission_lineage"]["emission_hash"] == "emission-hash-1"
    assert replay_regeneration["audited_synapses"][0]["replay_artifact_lineage_complete"] is True
    assert replay_regeneration["audited_synapses"][0]["local_edge_provenance_complete"] is True
    assert replay_regeneration["audited_synapses"][0][
        "local_edge_rollout_step_order_valid"
    ] is True
    assert replay_regeneration["audited_synapses"][0]["source_rollout_step_index"] == 10
    assert replay_regeneration["audited_synapses"][0]["target_rollout_step_index"] == 20
    assert replay_regeneration["audited_synapses"][0][
        "source_active_indices_hash"
    ] == "source-active-hash-1"
    assert replay_regeneration_matching_restore["promotion_gate"][
        "eligible_for_readout_synapse_audit_review"
    ] is True
    assert replay_regeneration_matching_restore["promotion_gate"]["required_evidence"][
        "applied_replay_lineage_restore_validation_not_mismatched"
    ] is True
    assert replay_regeneration_matching_restore["audit_summary"][
        "restore_validation_available"
    ] is True
    assert replay_regeneration_matching_restore["audit_summary"][
        "restore_validation_blocks_audit"
    ] is False
    assert replay_regeneration_mismatched_restore["promotion_gate"][
        "eligible_for_readout_synapse_audit_review"
    ] is False
    assert replay_regeneration_mismatched_restore["promotion_gate"]["required_evidence"][
        "applied_replay_lineage_restore_validation_not_mismatched"
    ] is False
    assert replay_regeneration_mismatched_restore["audit_summary"][
        "restore_validation_available"
    ] is True
    assert replay_regeneration_mismatched_restore["audit_summary"][
        "restore_validation_blocks_audit"
    ] is True
    assert replay_missing_local_edge["promotion_gate"][
        "eligible_for_readout_synapse_audit_review"
    ] is False
    assert replay_missing_local_edge["promotion_gate"]["required_evidence"][
        "audited_replay_regeneration_local_edge_provenance_complete"
    ] is False
    assert replay_incomplete_lineage["promotion_gate"][
        "eligible_for_readout_synapse_audit_review"
    ] is False
    assert replay_incomplete_lineage["promotion_gate"]["required_evidence"][
        "audited_replay_regeneration_artifact_lineage_complete"
    ] is False
    assert tampered_ledger["promotion_gate"]["eligible_for_readout_synapse_audit_review"] is False
    assert tampered_ledger["promotion_gate"]["required_evidence"]["audited_synapses_match_ledger_fields"] is True
    assert tampered_ledger["promotion_gate"]["required_evidence"]["audited_ledger_hashes_valid"] is False


def test_readout_rehearsal_evaluation_blocks_empty_priority_report() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )

    evaluation = ledger.rehearsal_evaluation(ledger.replay_priority(limit=4))

    assert evaluation["promotion_gate"]["status"] == "blocked_missing_rehearsal_evidence"
    assert evaluation["promotion_gate"]["eligible_for_operator_rehearsal_review"] is False
    assert evaluation["rehearsal_summary"]["candidate_count"] == 0
    assert evaluation["mutates_runtime_state"] is False


def test_readout_rehearsal_experiment_blocks_bad_or_empty_evaluation() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )

    blocked = ledger.rehearsal_experiment({"surface": "wrong.v1"})
    empty = ledger.rehearsal_experiment(ledger.rehearsal_evaluation(ledger.replay_priority(limit=4)))

    assert blocked["promotion_gate"]["status"] == "blocked_missing_rehearsal_experiment_evidence"
    assert blocked["promotion_gate"]["required_evidence"]["rehearsal_gate_ready"] is False
    assert empty["promotion_gate"]["eligible_for_operator_rehearsal_experiment_review"] is False
    assert empty["mutates_runtime_state"] is False


def test_readout_replay_design_blocks_missing_experiment_evidence() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )

    design = ledger.replay_design({"surface": "wrong.v1"})

    assert design["promotion_gate"]["status"] == "blocked_missing_readout_replay_design_evidence"
    assert design["promotion_gate"]["eligible_for_operator_replay_design_review"] is False
    assert design["promotion_gate"]["required_evidence"]["rehearsal_experiment_gate_ready"] is False
    assert design["mutates_runtime_state"] is False
