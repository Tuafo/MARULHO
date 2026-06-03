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
        transition_memory_state={"sparse_transition_weights": {first_synapse: 0.1}},
    )
    repeat = ledger.rollout_consolidation_shadow_application_preflight(
        design,
        shadow,
        transition_memory_state={"sparse_transition_weights": {first_synapse: 0.1}},
    )
    growth = ledger.rollout_consolidation_shadow_application_preflight(
        design,
        shadow,
        transition_memory_state={"sparse_transition_weights": {}},
    )
    tampered_shadow = deepcopy(shadow)
    tampered_shadow["bounded_synapses"][0]["target_index"] = 63
    tampered = ledger.rollout_consolidation_shadow_application_preflight(
        design,
        tampered_shadow,
        transition_memory_state={"sparse_transition_weights": {first_synapse: 0.1}},
    )
    tampered_design = deepcopy(design)
    tampered_design["rollout_consolidation_design_material"]["max_weight_delta"] = 0.25
    design_tampered = ledger.rollout_consolidation_shadow_application_preflight(
        tampered_design,
        shadow,
        transition_memory_state={"sparse_transition_weights": {first_synapse: 0.1}},
    )
    duplicate_shadow = deepcopy(shadow)
    duplicate_shadow["bounded_synapses"].append(deepcopy(duplicate_shadow["bounded_synapses"][0]))
    duplicate_shadow["affected_synapse_count"] = 2
    duplicate_shadow["shadow_delta"]["nonzero_count"] = 1
    duplicate = ledger.rollout_consolidation_shadow_application_preflight(
        design,
        duplicate_shadow,
        transition_memory_state={"sparse_transition_weights": {first_synapse: 0.1}},
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
    assert preflight["rollback_evidence"]["checkpoint_restore_verified"] is False
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
    growth_preflight = ledger.rollout_consolidation_shadow_application_preflight(
        design,
        shadow,
        transition_memory_state={"sparse_transition_weights": {}},
    )
    existing_preflight = ledger.rollout_consolidation_shadow_application_preflight(
        design,
        shadow,
        transition_memory_state={"sparse_transition_weights": {first_synapse: 0.1}},
    )
    before = runtime_state.snapshot()

    review = ledger.rollout_developmental_plasticity_review(
        design,
        growth_preflight,
        transition_memory_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_hecsn": True,
            "sparse_transition_weights": {},
        },
    )
    repeat = ledger.rollout_developmental_plasticity_review(
        design,
        growth_preflight,
        transition_memory_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_hecsn": True,
            "sparse_transition_weights": {},
        },
    )
    no_growth = ledger.rollout_developmental_plasticity_review(
        design,
        existing_preflight,
        transition_memory_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_hecsn": True,
            "sparse_transition_weights": {first_synapse: 0.1},
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
        transition_memory_state={"sparse_transition_weights": {}},
    )
    review = ledger.rollout_developmental_plasticity_review(
        design,
        growth_preflight,
        transition_memory_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_hecsn": True,
            "sparse_transition_weights": {},
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
        transition_memory_state={"sparse_transition_weights": {}},
    )
    developmental = ledger.rollout_developmental_plasticity_review(
        design,
        growth_preflight,
        transition_memory_state={
            "surface": "snn_language_plasticity_runtime_state.v1",
            "owned_by_hecsn": True,
            "sparse_transition_weights": {},
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

    facade = RuntimeFacade(_Root())
    review = {
        "surface": "snn_language_readout_rollout_regeneration_replay_artifact_review.v1",
        "owned_by_hecsn": True,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "rollout_regeneration_replay_artifact_review_hash": "review-hash-1",
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
                        "pre_index": 0,
                        "post_index": 1,
                        "synapse": "0:1",
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

    assert blocked["accepted"] is False
    assert blocked["issues_regeneration_permit"] is False
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
    assert accepted["replay_evidence"]["permit_id"] == "permit-1"
    assert accepted["before"]["state_revision"] == accepted["after"]["state_revision"]
    assert accepted["after"]["dirty_state"] is True
    assert accepted["promotion_gate"]["eligible_for_regeneration_application"] is True
    assert accepted["promotion_gate"]["eligible_for_structural_write"] is False


def test_rollout_regeneration_application_preflight_requires_revision_and_checkpoint() -> None:
    runtime_state = RuntimeState()

    class _Root:
        _runtime_state = runtime_state

    facade = RuntimeFacade(_Root())
    permit_request = {
        "surface": "snn_language_readout_rollout_regeneration_permit_request.v1",
        "accepted": True,
        "owned_by_hecsn": True,
        "applies_plasticity": False,
        "mutates_runtime_state": True,
        "checkpoint_written": False,
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
                    "pre_index": 0,
                    "post_index": 1,
                    "synapse": "0:1",
                    "initial_weight": 0.02,
                    "locality_distance": 1,
                }
            ],
        },
        "promotion_gate": {"eligible_for_regeneration_application": True},
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

    assert blocked["ready"] is False
    assert blocked["promotion_gate"]["required_evidence"]["expected_revision_current"] is False
    assert blocked["promotion_gate"]["required_evidence"]["checkpoint_path_available"] is False
    assert ready["ready"] is True
    assert ready["executor_called"] is False
    assert ready["writes_checkpoint"] is False
    assert ready["applies_plasticity"] is False
    assert ready["mutates_runtime_state"] is False
    assert ready["regeneration_proposal"]["available"] is True
    assert ready["regeneration_proposal"]["promotion_gate"]["status"] == "ready_for_operator_review"
    assert ready["promotion_gate"]["eligible_for_checkpoint_backed_regeneration_executor"] is True
    assert ready["promotion_gate"]["eligible_for_regeneration_application"] is False


def test_rollout_regeneration_application_delegates_to_checkpoint_backed_executor(
    tmp_path: Path,
) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {"sparse_transition_weights": {"1:2": 0.9}}

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
        "regeneration_proposal": {
            "available": True,
            "ready": True,
            "owned_by_hecsn": True,
            "generates_text": False,
            "loads_external_checkpoint": False,
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
                        "pre_index": 1,
                        "post_index": 3,
                        "initial_weight": 0.1,
                        "locality_distance": 2,
                        "source_synapse_id": "snn-rollout-local:1:3:0",
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
    assert result["executor_result"]["surface"] == "snn_language_transition_memory_regeneration.v1"
    assert abs(language_state["sparse_transition_weights"]["1:3"] - 0.1) < 1e-9
    local_edge = result["executor_result"]["regeneration"]["regenerated_synapses"][0][
        "local_edge_provenance"
    ]
    assert local_edge["source_synapse_id"] == "snn-rollout-local:1:3:0"
    assert local_edge["source_rollout_step_index"] == 10
    assert local_edge["target_rollout_step_index"] == 20
    assert local_edge["source_active_indices_hash"] == "source-active-hash-1"
    assert language_state["synapse_provenance_by_key"]["1:3"][
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
    assert replay_regeneration["audited_synapses"][0]["local_edge_provenance_complete"] is True
    assert replay_regeneration["audited_synapses"][0][
        "local_edge_rollout_step_order_valid"
    ] is True
    assert replay_regeneration["audited_synapses"][0]["source_rollout_step_index"] == 10
    assert replay_regeneration["audited_synapses"][0]["target_rollout_step_index"] == 20
    assert replay_regeneration["audited_synapses"][0][
        "source_active_indices_hash"
    ] == "source-active-hash-1"
    assert replay_missing_local_edge["promotion_gate"][
        "eligible_for_readout_synapse_audit_review"
    ] is False
    assert replay_missing_local_edge["promotion_gate"]["required_evidence"][
        "audited_replay_regeneration_local_edge_provenance_complete"
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
