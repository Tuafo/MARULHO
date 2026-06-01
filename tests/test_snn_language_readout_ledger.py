from __future__ import annotations

from threading import RLock

from hecsn.service.runtime_state import RuntimeState
from hecsn.service.snn_language_readout_ledger import SNNLanguageReadoutEvidenceLedger


def _ready_draft() -> dict[str, object]:
    return _ready_draft_for("prediction-hash-1", "evaluation-hash-1", "weights-hash-1", ["memory pressure"])


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
