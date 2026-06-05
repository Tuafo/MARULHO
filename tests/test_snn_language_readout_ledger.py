from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from threading import RLock

from hecsn.service.runtime_state import RuntimeState
from hecsn.service.runtime_facade import RuntimeFacade
from hecsn.service.snn_language_plasticity_executor import SNNLanguagePlasticityApplicationExecutor
from hecsn.service.snn_language_readout_ledger import SNNLanguageReadoutEvidenceLedger


def _ready_draft() -> dict[str, object]:
    return _ready_draft_for("prediction-hash-1", "evaluation-hash-1", "weights-hash-1", ["memory pressure"])


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
        "owned_by_hecsn": True,
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


def _ready_emission() -> dict[str, object]:
    emission_hash = _sha256_json({"emission": "memory pressure"})
    trajectory_hash = _sha256_json({"trajectory": "memory pressure"})
    prediction_hash = _sha256_json({"prediction": "memory pressure"})
    evaluation_hash = _sha256_json({"evaluation": "memory pressure"})
    weights_hash = _sha256_json({"weights": "memory pressure"})
    return {
        "surface": "snn_language_readout_emission.v1",
        "ready": True,
        "owned_by_hecsn": True,
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
        "owned_by_hecsn": True,
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
    assert ready["metrics"]["expected_calibration_error"] is not None
    assert ready["metrics"]["coverage_gap"] == 0.0
    assert len(ready["reliability_bins"]) == 5
    assert ready["evaluated_samples"][0]["heldout_match_count"] == 2
    assert ready["promotion_gate"][
        "eligible_for_dense_label_calibration_evaluation_review"
    ] is True
    assert ready["promotion_gate"]["eligible_for_dense_readout_training"] is False
    assert ready["promotion_gate"]["eligible_for_language_generation"] is False
    assert ready["promotion_gate"]["eligible_for_replay_memory"] is False
    assert ready["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert ready["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert ready["promotion_gate"]["eligible_for_action"] is False


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
    assert autonomous_update["operator_id"] == "hecsn-autonomous-confidence-recalibrator"
    assert autonomous_update["runtime_update_applied"] is True
    assert autonomous_update["weights_persisted"] is False
    assert autonomous_recalibration_executor["ledger_summary"][
        "dense_label_calibration_update_event_count"
    ] == 2
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
    assert "events" not in history
    assert "rollout_events" not in history
    assert "prediction_report" not in reviewed
    assert "transition_memory_evaluation" not in reviewed
    assert empty["summary"]["returned_emission_review_event_count"] == 0
    assert empty["emission_review_events"] == []
    assert empty["promotion_gate"]["eligible_for_operator_display_history_inspection"] is False


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
    assert policy["promotion_gate"]["eligible_for_operator_replay_evaluation_policy_review"] is False
    assert policy["promotion_gate"]["required_evidence"]["reviewed_emission_available"] is True
    assert policy["promotion_gate"]["required_evidence"]["matching_internal_readout_evidence_available"] is False
    assert policy["promotion_gate"]["required_evidence"]["display_text_not_used_as_replay_source"] is True
    assert policy["generates_text"] is False
    assert policy["decodes_text"] is False
    assert policy["exposes_reviewed_bounded_text"] is False
    assert policy["records_ledger_event"] is False
    assert policy["runs_replay"] is False
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
    assert policy["promotion_gate"]["eligible_for_operator_replay_evaluation_policy_review"] is True
    assert policy["promotion_gate"]["eligible_for_replay_memory"] is False
    assert policy["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert policy["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert policy["promotion_gate"]["eligible_for_action"] is False


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
    assert design["promotion_gate"]["eligible_for_operator_rollout_consolidation_design_review"] is True
    assert design["promotion_gate"]["eligible_for_structural_write"] is False
    assert design["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert blocked["promotion_gate"]["required_evidence"]["rollback_policy_available"] is False
    assert blocked["promotion_gate"]["eligible_for_operator_rollout_consolidation_design_review"] is False


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
    assert shadow["promotion_gate"]["eligible_for_operator_rollout_consolidation_shadow_review"] is True
    assert shadow["promotion_gate"]["eligible_for_shadow_application"] is False
    assert shadow["promotion_gate"]["eligible_for_plasticity_application"] is False
    assert invalid_coordinate["promotion_gate"]["required_evidence"][
        "candidate_coordinates_canonical"
    ] is False
    assert invalid_coordinate["promotion_gate"][
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
            "owned_by_hecsn": True,
            "sparse_transition_weights": {},
            "language_capacity": expanded_capacity,
        },
    )
    repeat = ledger.rollout_developmental_plasticity_review(
        design,
        growth_preflight,
        transition_memory_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_hecsn": True,
            "sparse_transition_weights": {},
            "language_capacity": expanded_capacity,
        },
    )
    no_growth = ledger.rollout_developmental_plasticity_review(
        design,
        existing_preflight,
        transition_memory_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_hecsn": True,
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
            "owned_by_hecsn": True,
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
            "owned_by_hecsn": True,
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
            "owned_by_hecsn": True,
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
            "owned_by_hecsn": True,
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
            "owned_by_hecsn": True,
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
    adapter_candidate = adapter["regeneration_design"]["candidate_synapses"][0]
    assert adapter_candidate["source_rollout_step_index"] == 0
    assert adapter_candidate["target_rollout_step_index"] == 1
    assert adapter_candidate["source_active_indices_hash"] == _sha256_json([1, 2, 3])
    assert adapter_candidate["target_active_indices_hash"] == _sha256_json([2, 3, 4])
    assert adapter["integrity_evidence"]["candidate_rollout_step_provenance_available"] is True
    assert adapter["integrity_evidence"]["candidate_active_hash_provenance_available"] is True
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
        }
    )
    review = {
        "surface": "snn_language_readout_rollout_developmental_plasticity_review.v1",
        "artifact_kind": "terminus_snn_language_readout_rollout_developmental_plasticity_review",
        "owned_by_hecsn": True,
        "mutates_runtime_state": False,
        "applies_plasticity": False,
        "rollout_consolidation_design_hash": "design-hash-1",
        "rollout_consolidation_shadow_application_preflight_hash": (
            "preflight-hash-1"
        ),
        "rollout_developmental_plasticity_review_hash": review_hash,
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
        "owned_by_hecsn": True,
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
            "owned_by_hecsn": True,
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
        "owned_by_hecsn": True,
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
                "owned_by_hecsn": True,
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
        "owned_by_hecsn": True,
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
    assert accepted["promotion_gate"]["required_evidence"]["language_capacity_state_available"] is True
    assert accepted["promotion_gate"]["required_evidence"]["language_capacity_state_dynamic_limits_applied"] is True
    assert accepted["replay_evidence"]["permit_id"] == "permit-1"
    assert accepted["before"]["state_revision"] == accepted["after"]["state_revision"]
    assert accepted["after"]["dirty_state"] is True
    assert accepted["promotion_gate"]["eligible_for_regeneration_application"] is True
    assert accepted["promotion_gate"]["eligible_for_structural_write"] is False


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
        "owned_by_hecsn": True,
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
            "owned_by_hecsn": True,
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
        "owned_by_hecsn": True,
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
            "owned_by_hecsn": True,
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
                "owned_by_hecsn": True,
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
    assert priority["promotion_gate"]["eligible_for_operator_replay_review"] is True
    assert priority["promotion_gate"]["eligible_for_live_replay"] is False
    assert priority["promotion_gate"]["eligible_for_fact_promotion"] is False
    assert priority["candidate_count"] == 2
    assert priority["candidates"][0]["rank"] == 1
    assert priority["candidates"][0]["readout_evidence_hash"]
    assert priority["candidates"][0]["prediction_hash"]
    assert priority["candidates"][0]["priority_components"]["provenance"] == 1.0
    assert priority["candidates"][0]["executable"] is False
    assert priority["candidates"][0]["generates_text"] is False
    assert priority["candidates"][0]["eligible_for_action"] is False
    assert empty_limit["candidate_count"] == 0
    assert empty_limit["promotion_gate"]["status"] == "collect_readout_evidence"


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
            "owned_by_hecsn": True,
            "prediction_error": {"mismatch_score": 0.9},
        },
        pressure_report={
            "available": True,
            "owned_by_hecsn": True,
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
            "owned_by_hecsn": True,
            "prediction_error": {"mismatch_score": 0.9},
        },
        pressure_report={
            "available": True,
            "owned_by_hecsn": True,
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
    assert evaluation["owned_by_hecsn"] is True
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
    assert experiment["owned_by_hecsn"] is True
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
    assert design["owned_by_hecsn"] is True
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
    assert dry_run["owned_by_hecsn"] is True
    assert dry_run["generates_text"] is False
    assert dry_run["decodes_text"] is False
    assert dry_run["trains_runtime_model"] is False
    assert dry_run["applies_plasticity"] is False
    assert dry_run["mutates_runtime_state"] is False
    assert dry_run["returns_trained_weights"] is False
    assert dry_run["device_evidence"]["tensor_device"] == "cpu"
    assert dry_run["device_evidence"]["cuda_fallback_blocked"] is False
    assert dry_run["isolated_replay_summary"]["target_count"] == 1
    assert dry_run["isolated_replay_summary"]["pressure_non_worsening"] is True
    assert dry_run["ephemeral_replay"]["runtime_update_applied"] is False
    assert dry_run["ephemeral_replay"]["weights_persisted"] is False
    assert dry_run["ephemeral_replay"]["checkpoint_written"] is False
    assert dry_run["ephemeral_replay"]["trace"][0]["checkpoint_written"] is False
    assert dry_run["promotion_gate"]["eligible_for_operator_replay_dry_run_review"] is True
    assert dry_run["promotion_gate"]["eligible_for_live_replay"] is False
    assert dry_run["promotion_gate"]["eligible_for_plasticity_application"] is False


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
    assert preflight["readout_plasticity_preflight_hash"]
    assert preflight["owned_by_hecsn"] is True
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
    assert bridge["application_target_hint"]["target_id"] == "hecsn.snn_language.sparse_transition_weights"
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
            "owned_by_hecsn": True,
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
            "owned_by_hecsn": True,
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
            "owned_by_hecsn": True,
            "sparse_transition_weights": {"1:2": 0.03},
            "synapse_provenance_by_key": {},
        }
    )
    malformed = ledger.synapse_provenance_audit(
        plasticity_runtime_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_hecsn": True,
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
            "owned_by_hecsn": True,
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
            "owned_by_hecsn": True,
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
                "owned_by_hecsn": True,
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
                "owned_by_hecsn": True,
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
            "owned_by_hecsn": True,
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
            "owned_by_hecsn": True,
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
            "owned_by_hecsn": True,
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
    assert audit["promotion_gate"]["eligible_for_readout_synapse_audit_review"] is True
    assert blocked["promotion_gate"]["eligible_for_readout_synapse_audit_review"] is False
    assert blocked["promotion_gate"]["required_evidence"]["synapse_provenance_available"] is False
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
