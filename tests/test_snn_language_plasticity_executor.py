from __future__ import annotations

from pathlib import Path
from threading import RLock

from hecsn.service.runtime_state import RuntimeState
from hecsn.service.snn_language_plasticity_executor import SNNLanguagePlasticityApplicationExecutor


def _regeneration_proposal(*candidates: dict[str, object]) -> dict[str, object]:
    return {
        "available": True,
        "owned_by_hecsn": True,
        "generates_text": False,
        "loads_external_checkpoint": False,
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
            "readout_evidence_hashes": ["readout-hash-1", "readout-hash-2"],
            "source_metadata_hash": "source-metadata-hash-1",
            "emission_lineage": {
                "emission_hash": "emission-hash-1",
                "readout_evidence_hash": "readout-hash-1",
                "prediction_hash": "prediction-hash-1",
                "design_hash": "design-hash-1",
            },
            "evidence_hash": "sha256:replay-window-1",
        },
        "promotion_gate": {"status": "ready_for_operator_review"},
        "regeneration_design": {
            "locality_radius": 2,
            "mismatch_score": 0.9,
            "candidate_synapses": list(candidates),
        },
    }


def test_snapshot_exposes_read_only_language_capacity_state(tmp_path: Path) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {
        "language_capacity": {
            "language_neuron_count": 128,
            "sparse_edge_budget": 512,
            "outgoing_fanout_budget": 32,
            "capacity_expansion_count": 1,
        },
        "sparse_transition_weights": {"1:2": 0.5},
    }

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=lambda path: {"path": str(path or tmp_path / "checkpoint.pt")},
        checkpoint_path=lambda: tmp_path / "checkpoint.pt",
        verify_checkpoint=lambda path: path.exists(),
    )

    snapshot = executor.snapshot()

    assert snapshot["language_capacity"]["surface"] == "snn_language_capacity_state.v1"
    assert snapshot["language_neuron_count"] == 128
    assert snapshot["sparse_edge_budget"] == 512
    assert snapshot["outgoing_fanout_budget"] == 32
    assert snapshot["language_capacity"]["dynamic_capacity_enabled"] is False
    assert snapshot["language_capacity"]["resizes_network"] is False
    assert snapshot["language_capacity"]["adds_neurons"] is False


def test_homeostatic_maintenance_normalizes_rows_and_persists_prune_ledger(tmp_path: Path) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {
        "sparse_transition_weights": {
            "1:2": 0.8,
            "1:3": 0.8,
            "2:4": 0.001,
        }
    }

    def save_checkpoint(path: str | None) -> dict[str, str]:
        target = Path(path or tmp_path / "maintenance.pt")
        target.write_bytes(b"checkpoint")
        return {"path": str(target)}

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / "maintenance.pt",
        verify_checkpoint=lambda path: path.exists(),
        verify_regeneration_permit=lambda proposal: True,
    )
    result = executor.maintain_transition_memory(
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=True,
        checkpoint_path=str(tmp_path / "maintenance.pt"),
        decay_factor=1.0,
        prune_below=0.01,
        max_outgoing_row_mass=1.0,
    )

    assert result["accepted"] is True
    assert result["checkpoint_transaction"]["restore_verified"] is True
    assert result["homeostatic_maintenance"]["normalized_row_count"] == 1
    assert result["homeostatic_maintenance"]["max_outgoing_row_mass_after"] <= 1.0
    assert result["homeostatic_maintenance"]["pruned_synapse_count"] == 1
    assert result["pruned_synapses"][0]["synapse"] == "2:4"
    assert result["pruned_synapses"][0]["previous_weight"] == 0.001
    assert language_state["homeostatic_maintenance"]["last_maintenance"]["pruned_synapses"][0]["synapse"] == "2:4"
    assert language_state["homeostatic_maintenance"]["recent_events"][0]["committed_checkpoint_path"].endswith(
        ".homeostatic_maintenance.committed.pt"
    )
    assert sum(
        value
        for key, value in language_state["sparse_transition_weights"].items()
        if key.startswith("1:")
    ) <= 1.0
    assert runtime_state.state_revision == 1


def test_homeostatic_maintenance_blocks_when_checkpoint_restore_verification_fails(tmp_path: Path) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {"sparse_transition_weights": {"1:2": 0.8}}

    def save_checkpoint(path: str | None) -> dict[str, str]:
        target = Path(path or tmp_path / "maintenance.pt")
        target.write_bytes(b"checkpoint")
        return {"path": str(target)}

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / "maintenance.pt",
        verify_checkpoint=lambda path: False,
    )
    result = executor.maintain_transition_memory(
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=True,
        checkpoint_path=str(tmp_path / "maintenance.pt"),
    )

    assert result["accepted"] is False
    assert result["promotion_gate"]["required_evidence"]["pre_maintenance_checkpoint_saved"] is True
    assert result["promotion_gate"]["required_evidence"]["pre_maintenance_checkpoint_restore_verified"] is False
    assert language_state["sparse_transition_weights"] == {"1:2": 0.8}
    assert runtime_state.state_revision == 0


def test_regeneration_applies_bounded_local_edges_and_persists_ledger(tmp_path: Path) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {"sparse_transition_weights": {"1:2": 0.9}}

    def save_checkpoint(path: str | None) -> dict[str, str]:
        target = Path(path or tmp_path / "regeneration.pt")
        target.write_bytes(b"checkpoint")
        return {"path": str(target)}

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / "regeneration.pt",
        verify_checkpoint=lambda path: path.exists(),
        verify_regeneration_permit=lambda proposal: True,
    )
    result = executor.regenerate_transition_memory(
        regeneration_proposal=_regeneration_proposal(
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
            },
            {"pre_index": 4, "post_index": 5, "initial_weight": 0.2, "locality_distance": 1},
        ),
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=True,
        checkpoint_path=str(tmp_path / "regeneration.pt"),
        max_outgoing_row_mass=1.0,
    )

    assert result["accepted"] is True
    assert result["checkpoint_transaction"]["restore_verified"] is True
    assert abs(language_state["sparse_transition_weights"]["1:3"] - 0.1) < 1e-9
    assert language_state["sparse_transition_weights"]["4:5"] == 0.2
    assert sum(value for key, value in language_state["sparse_transition_weights"].items() if key.startswith("1:")) <= 1.0
    assert language_state["synapse_regeneration"]["last_regeneration"]["regenerated_synapse_count"] == 2
    assert language_state["synapse_regeneration"]["last_regeneration"]["replay_regeneration_permit"]["permit_id"] == "permit-1"
    assert (
        language_state["synapse_regeneration"]["last_regeneration"]["replay_regeneration_permit"][
            "readout_evidence_hashes"
        ]
        == ["readout-hash-1", "readout-hash-2"]
    )
    assert (
        language_state["synapse_regeneration"]["last_regeneration"]["replay_regeneration_permit"][
            "source_metadata_hash"
        ]
        == "source-metadata-hash-1"
    )
    assert (
        language_state["synapse_regeneration"]["last_regeneration"]["replay_regeneration_permit"][
            "emission_lineage"
        ]["emission_hash"]
        == "emission-hash-1"
    )
    assert language_state["synapse_regeneration"]["recent_events"][0]["regenerated_synapses"][0]["synapse"] == "1:3"
    local_edge = language_state["synapse_regeneration"]["recent_events"][0][
        "regenerated_synapses"
    ][0]["local_edge_provenance"]
    assert local_edge["source_synapse_id"] == "snn-rollout-local:1:3:0"
    assert local_edge["source_rollout_step_index"] == 10
    assert local_edge["target_rollout_step_index"] == 20
    assert local_edge["source_active_indices_hash"] == "source-active-hash-1"
    assert local_edge["target_active_indices_hash"] == "target-active-hash-1"
    provenance = language_state["synapse_provenance_by_key"]["1:3"]
    assert provenance["provenance_type"] == "replay_regeneration"
    assert provenance["permit_id"] == "permit-1"
    assert provenance["replay_artifact_id"] == "artifact-1"
    assert provenance["replay_window_hash"] == "window-hash-1"
    assert provenance["readout_evidence_hashes"] == ["readout-hash-1", "readout-hash-2"]
    assert provenance["source_metadata_hash"] == "source-metadata-hash-1"
    assert provenance["emission_lineage"]["emission_hash"] == "emission-hash-1"
    assert provenance["local_edge_provenance"] == local_edge
    assert language_state["synapse_regeneration"]["recent_events"][0]["committed_checkpoint_path"].endswith(
        ".regeneration.committed.pt"
    )
    assert runtime_state.state_revision == 1


def test_regeneration_uses_capacity_state_for_sparse_indices_above_default(
    tmp_path: Path,
) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {
        "language_capacity": {
            "surface": "snn_language_capacity_state.v1",
            "language_neuron_count": 128,
            "sparse_edge_budget": 512,
            "outgoing_fanout_budget": 32,
        },
        "sparse_transition_weights": {},
    }

    def save_checkpoint(path: str | None) -> dict[str, str]:
        target = Path(path or tmp_path / "regeneration.pt")
        target.write_bytes(b"checkpoint")
        return {"path": str(target)}

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / "regeneration.pt",
        verify_checkpoint=lambda path: path.exists(),
        verify_regeneration_permit=lambda proposal: True,
    )

    result = executor.regenerate_transition_memory(
        regeneration_proposal=_regeneration_proposal(
            {
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
            },
        ),
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=True,
        checkpoint_path=str(tmp_path / "regeneration.pt"),
    )

    assert result["accepted"] is True
    assert "65:66" in language_state["sparse_transition_weights"]
    assert runtime_state.state_revision == 1


def test_regeneration_blocks_without_operator_confirmation(tmp_path: Path) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {"sparse_transition_weights": {}}
    checkpoint_calls = []

    def save_checkpoint(path: str | None) -> dict[str, str]:
        checkpoint_calls.append(path)
        return {"path": str(tmp_path / "regeneration.pt")}

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / "regeneration.pt",
        verify_checkpoint=lambda path: path.exists(),
        verify_regeneration_permit=lambda proposal: True,
    )
    result = executor.regenerate_transition_memory(
        regeneration_proposal=_regeneration_proposal(
            {"pre_index": 1, "post_index": 2, "initial_weight": 0.1, "locality_distance": 1},
        ),
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=False,
    )

    assert result["accepted"] is False
    assert result["promotion_gate"]["required_evidence"]["confirmation"] is False
    assert checkpoint_calls == []
    assert language_state["sparse_transition_weights"] == {}
    assert runtime_state.state_revision == 0


def test_regeneration_blocks_nonlocal_and_modulo_alias_candidates_before_checkpoint(tmp_path: Path) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {"sparse_transition_weights": {"1:2": 0.8}}
    checkpoint_calls = []

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=lambda path: checkpoint_calls.append(path) or {"path": str(tmp_path / "regeneration.pt")},
        checkpoint_path=lambda: tmp_path / "regeneration.pt",
        verify_checkpoint=lambda path: path.exists(),
        verify_regeneration_permit=lambda proposal: True,
    )
    result = executor.regenerate_transition_memory(
        regeneration_proposal=_regeneration_proposal(
            {"pre_index": 65, "post_index": 4, "initial_weight": 0.1, "locality_distance": 1},
            {"pre_index": 1, "post_index": 20, "initial_weight": 0.1, "locality_distance": 1},
        ),
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=True,
    )

    assert result["accepted"] is False
    evidence = result["promotion_gate"]["required_evidence"]
    assert evidence["candidate_indices_canonical"] is False
    assert evidence["candidate_synapses_local"] is False
    assert checkpoint_calls == []
    assert language_state["sparse_transition_weights"] == {"1:2": 0.8}
    assert runtime_state.state_revision == 0


def test_regeneration_blocks_fabricated_permit_before_checkpoint(tmp_path: Path) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {"sparse_transition_weights": {}}
    checkpoint_calls = []
    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=lambda path: checkpoint_calls.append(path) or {"path": str(tmp_path / "regeneration.pt")},
        checkpoint_path=lambda: tmp_path / "regeneration.pt",
        verify_checkpoint=lambda path: path.exists(),
    )
    result = executor.regenerate_transition_memory(
        regeneration_proposal=_regeneration_proposal(
            {"pre_index": 1, "post_index": 2, "initial_weight": 0.1, "locality_distance": 1},
        ),
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=True,
    )

    assert result["accepted"] is False
    assert result["promotion_gate"]["required_evidence"]["replay_permit_server_verified"] is False
    assert checkpoint_calls == []
    assert runtime_state.state_revision == 0


def test_regeneration_blocks_when_checkpoint_snapshot_does_not_round_trip(tmp_path: Path) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {"sparse_transition_weights": {}}

    def save_checkpoint(path: str | None) -> dict[str, str]:
        target = Path(path or tmp_path / "regeneration.pt")
        target.write_bytes(b"checkpoint")
        return {"path": str(target)}

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / "regeneration.pt",
        verify_checkpoint=lambda path: path.exists(),
        verify_regeneration_permit=lambda proposal: True,
        verify_checkpoint_snapshot=lambda path, state, revision: False,
    )
    result = executor.regenerate_transition_memory(
        regeneration_proposal=_regeneration_proposal(
            {"pre_index": 1, "post_index": 2, "initial_weight": 0.1, "locality_distance": 1},
        ),
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=True,
    )

    assert result["accepted"] is False
    assert result["promotion_gate"]["required_evidence"]["pre_regeneration_checkpoint_restore_verified"] is False
    assert language_state["sparse_transition_weights"] == {}
    assert runtime_state.state_revision == 0


def test_regeneration_recovers_memory_when_post_mutation_checkpoint_commit_fails(tmp_path: Path) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {"sparse_transition_weights": {}}
    verification_calls = []

    def save_checkpoint(path: str | None) -> dict[str, str]:
        target = Path(path or tmp_path / "regeneration.pt")
        target.write_bytes(b"checkpoint")
        return {"path": str(target)}

    def verify_snapshot(path: Path, state: object, revision: int) -> bool:
        verification_calls.append((path.name, revision))
        return len(verification_calls) != 2

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / "regeneration.pt",
        verify_checkpoint=lambda path: path.exists(),
        verify_regeneration_permit=lambda proposal: True,
        verify_checkpoint_snapshot=verify_snapshot,
    )
    result = executor.regenerate_transition_memory(
        regeneration_proposal=_regeneration_proposal(
            {"pre_index": 1, "post_index": 2, "initial_weight": 0.1, "locality_distance": 1},
        ),
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=True,
    )

    assert result["accepted"] is False
    assert result["reason"] == "post_regeneration_checkpoint_commit_failed"
    evidence = result["promotion_gate"]["required_evidence"]
    assert evidence["rollback_recovered_in_memory"] is True
    assert evidence["rollback_checkpoint_rewritten_verified"] is True
    assert language_state == {"sparse_transition_weights": {}}
    assert runtime_state.state_revision == 0


def test_regeneration_recovers_memory_when_current_checkpoint_publication_fails(tmp_path: Path) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {"sparse_transition_weights": {}}

    def save_checkpoint(path: str | None) -> dict[str, str]:
        target = Path(path or tmp_path / "regeneration.pt")
        target.write_bytes(b"checkpoint")
        return {"path": str(target)}

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / "regeneration.pt",
        verify_checkpoint=lambda path: path.exists(),
        verify_regeneration_permit=lambda proposal: True,
        publish_committed_checkpoint=lambda path, operation: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    result = executor.regenerate_transition_memory(
        regeneration_proposal=_regeneration_proposal(
            {"pre_index": 1, "post_index": 2, "initial_weight": 0.1, "locality_distance": 1},
        ),
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=True,
    )

    assert result["accepted"] is False
    assert result["reason"] == "post_regeneration_checkpoint_commit_failed"
    assert result["promotion_gate"]["required_evidence"]["rollback_recovered_in_memory"] is True
    assert language_state == {"sparse_transition_weights": {}}
    assert runtime_state.state_revision == 0


def test_regeneration_duplicate_only_proposal_is_a_noop(tmp_path: Path) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {"sparse_transition_weights": {"1:2": 0.1}}

    def save_checkpoint(path: str | None) -> dict[str, str]:
        target = Path(path or tmp_path / "regeneration.pt")
        target.write_bytes(b"checkpoint")
        return {"path": str(target)}

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / "regeneration.pt",
        verify_checkpoint=lambda path: path.exists(),
        verify_regeneration_permit=lambda proposal: True,
    )
    result = executor.regenerate_transition_memory(
        regeneration_proposal=_regeneration_proposal(
            {"pre_index": 1, "post_index": 2, "initial_weight": 0.1, "locality_distance": 1},
        ),
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=True,
    )

    assert result["accepted"] is False
    assert result["reason"] == "blocked_no_regenerable_synapses"
    assert "synapse_regeneration" not in language_state
    assert runtime_state.state_revision == 0


def test_live_application_blocks_modulo_alias_before_checkpoint(tmp_path: Path) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {"sparse_transition_weights": {}}
    checkpoint_calls = []
    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=lambda path: checkpoint_calls.append(path) or {"path": str(tmp_path / "live.pt")},
        checkpoint_path=lambda: tmp_path / "live.pt",
        verify_checkpoint=lambda path: path.exists(),
    )
    result = executor.apply_live_application(
        live_application_readiness={
            "available": True,
            "promotion_gate": {"status": "ready_for_operator_review"},
            "rollback_readiness": {"checkpoint_available": True, "restore_endpoint_available": True},
            "operator_approval": {"approved": True},
        },
        shadow_delta={
            "available": True,
            "max_abs_weight_delta": 0.1,
            "pressure_before": 0.9,
            "pressure_after": 0.8,
            "bounded_synapses": [{"pre_index": 65, "post_index": 2}],
        },
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=True,
    )

    assert result["accepted"] is False
    assert result["promotion_gate"]["required_evidence"]["candidate_indices_canonical"] is False
    assert checkpoint_calls == []
    assert language_state["sparse_transition_weights"] == {}
    assert runtime_state.state_revision == 0
