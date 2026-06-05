from __future__ import annotations

from pathlib import Path
from threading import RLock

import torch

from marulho.service.runtime_state import RuntimeState
from marulho.service.snn_language_plasticity_executor import SNNLanguagePlasticityApplicationExecutor


def _regeneration_proposal(*candidates: dict[str, object]) -> dict[str, object]:
    return {
        "available": True,
        "owned_by_marulho": True,
        "generates_text": False,
        "loads_external_checkpoint": False,
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


def _dense_readout_transaction() -> dict[str, object]:
    return {
        "surface": "snn_language_dense_readout_resize_transaction_proposal.v1",
        "owned_by_marulho": True,
        "generates_text": False,
        "loads_external_checkpoint": False,
        "dense_readout_resize_transaction_proposal_hash": "sha256:dense-transaction",
        "dense_readout_resize_plan_hash": "sha256:dense-plan",
        "transaction_recipe": {
            "current_dense_readout_shape": [64, 64],
            "target_dense_readout_shape": [128, 128],
            "preserved_dense_window": [64, 64],
            "zero_initialized_new_dense_cell_count": 12288,
        },
    }


def _dense_readout_readiness_audit() -> dict[str, object]:
    return {
        "surface": "snn_language_dense_readout_resize_executor_readiness_audit.v1",
        "owned_by_marulho": True,
        "generates_text": False,
        "loads_external_checkpoint": False,
        "promotion_gate": {
            "required_evidence": {
                "dense_readout_layout_state_available": True,
                "dense_readout_layout_matches_transaction": True,
                "dense_readout_layout_metadata_not_applied": True,
                "dense_readout_tensor_owner_available": True,
                "transaction_checkpoint_restore_verified": True,
                "transaction_cuda_relayout_verified": True,
                "transaction_shape_invariants_available": True,
                "dense_readout_tensor_weight_owner_available": False,
            }
        },
    }


def _dense_readout_tensor_materialization_readiness() -> dict[str, object]:
    return {
        "surface": "snn_language_dense_readout_tensor_materialization_readiness.v1",
        "ready": True,
        "owned_by_marulho": True,
        "executable": False,
        "mutates_runtime_state": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "trains_runtime_model": False,
        "target_dense_readout_shape": [128, 128],
        "preserved_dense_window": [64, 64],
        "zero_initialized_new_dense_cell_count": 12288,
        "promotion_gate": {
            "required_evidence": {
                "layout_migration_checkpoint_committed": True,
                "layout_state_matches_migration": True,
                "dense_resize_not_yet_applied": True,
            }
        },
    }


def _dense_readout_training_loop_preflight() -> dict[str, object]:
    return {
        "surface": "snn_language_dense_readout_training_loop_preflight.v1",
        "ready": True,
        "owned_by_marulho": True,
        "executable": False,
        "mutates_runtime_state": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "checkpoint_path": "dense-training.pt",
        "preflight_hash": "sha256:dense-training-preflight",
        "tensor_summary": {
            "shape": [128, 128],
            "device": "cpu",
            "dtype": "torch.float32",
            "nonzero_count": 0,
        },
        "training_design": {
            "training_transition_count": 4,
            "validation_transition_count": 2,
            "learning_rate": 0.02,
            "max_delta_norm": 0.05,
            "transition_budget": 128,
            "requires_cuda": False,
        },
        "promotion_gate": {
            "status": "ready_for_checkpoint_backed_dense_readout_training_executor",
            "required_evidence": {
                "expected_state_revision_current": True,
                "checkpoint_path_available": True,
                "bounded_delta_application_capability_available": True,
            },
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
    assert (
        snapshot["dense_readout_layout"]["surface"]
        == "snn_language_dense_readout_layout_state.v1"
    )
    assert snapshot["dense_readout_layout"]["target_dense_readout_shape"] == [128, 128]
    assert snapshot["dense_readout_layout"]["preserved_dense_window"] == [64, 64]
    assert snapshot["dense_readout_layout"]["requires_cuda_relayout"] is True
    assert snapshot["dense_readout_layout"]["layout_migration_applied"] is False
    assert snapshot["dense_readout_layout"]["dense_resize_applied"] is False
    assert snapshot["dense_readout_layout"]["resizes_network"] is False
    assert snapshot["dense_readout_tensor"]["available"] is False


def test_dense_readout_layout_migration_persists_checkpointed_resize_evidence(
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
        "sparse_transition_weights": {"1:2": 0.5},
    }

    def save_checkpoint(path: str | None) -> dict[str, str]:
        target = Path(path or tmp_path / "dense-layout.pt")
        target.write_bytes(b"checkpoint")
        return {"path": str(target)}

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / "dense-layout.pt",
        verify_checkpoint=lambda path: path.exists(),
    )

    result = executor.apply_dense_readout_layout_migration(
        dense_readout_resize_transaction_proposal=_dense_readout_transaction(),
        dense_readout_resize_executor_readiness_audit=_dense_readout_readiness_audit(),
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=True,
        checkpoint_path=str(tmp_path / "dense-layout.pt"),
    )
    snapshot = executor.snapshot()

    assert result["accepted"] is True
    assert result["surface"] == "snn_language_dense_readout_layout_migration.v1"
    assert result["mutates_runtime_state"] is True
    assert result["writes_checkpoint"] is True
    assert result["resizes_network"] is False
    assert result["materializes_dense_tensor_weights"] is False
    assert result["checkpoint_transaction"]["restore_verified"] is True
    assert result["dense_readout_layout_migration"]["target_dense_readout_shape"] == [
        128,
        128,
    ]
    assert result["dense_readout_layout_migration"][
        "materializes_dense_tensor_weights"
    ] is False
    assert language_state["sparse_transition_weights"] == {"1:2": 0.5}
    assert language_state["dense_readout_layout"]["layout_migration"]["applied"] is True
    assert language_state["dense_readout_layout"]["dense_resize_applied"] is False
    assert (
        language_state["dense_readout_layout"]["migration_status"]
        == "layout_migration_applied_tensor_resize_pending"
    )
    assert snapshot["dense_readout_layout"]["layout_migration_applied"] is True
    assert snapshot["dense_readout_layout"]["dense_resize_applied"] is False
    assert runtime_state.state_revision == 1


def test_dense_readout_layout_migration_blocks_when_tensor_materialization_claimed(
    tmp_path: Path,
) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {
        "language_capacity": {"language_neuron_count": 128},
        "sparse_transition_weights": {"1:2": 0.5},
    }
    checkpoint_calls = []
    audit = _dense_readout_readiness_audit()
    audit["promotion_gate"]["required_evidence"][
        "dense_readout_tensor_weight_owner_available"
    ] = True
    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=lambda path: checkpoint_calls.append(path)
        or {"path": str(tmp_path / "dense-layout.pt")},
        checkpoint_path=lambda: tmp_path / "dense-layout.pt",
        verify_checkpoint=lambda path: path.exists(),
    )

    result = executor.apply_dense_readout_layout_migration(
        dense_readout_resize_transaction_proposal=_dense_readout_transaction(),
        dense_readout_resize_executor_readiness_audit=audit,
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=True,
    )

    assert result["accepted"] is False
    assert result["promotion_gate"]["required_evidence"][
        "tensor_weight_materialization_absent"
    ] is False
    assert checkpoint_calls == []
    assert "dense_readout_layout" not in language_state
    assert runtime_state.state_revision == 0


def test_dense_readout_tensor_materialization_projects_sparse_weights_to_dense_tensor(
    tmp_path: Path,
) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    layout_migration = {
        "applied": True,
        "target_dense_readout_shape": [128, 128],
        "preserved_dense_window": [64, 64],
        "zero_initialized_new_dense_cell_count": 12288,
    }
    language_state = {
        "language_capacity": {"language_neuron_count": 128},
        "dense_readout_layout": {
            "surface": "snn_language_dense_readout_layout_state.v1",
            "target_language_neuron_count": 128,
            "layout_migration": layout_migration,
            "dense_resize_applied": False,
        },
        "sparse_transition_weights": {"1:2": 0.5, "65:66": 0.25},
    }

    def save_checkpoint(path: str | None) -> dict[str, str]:
        target = Path(path or tmp_path / "dense-tensor.pt")
        target.write_bytes(b"checkpoint")
        return {"path": str(target)}

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / "dense-tensor.pt",
        verify_checkpoint=lambda path: path.exists(),
    )

    result = executor.apply_dense_readout_tensor_materialization(
        dense_readout_tensor_materialization_readiness=(
            _dense_readout_tensor_materialization_readiness()
        ),
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=True,
        checkpoint_path=str(tmp_path / "dense-tensor.pt"),
        requested_device="cpu",
    )
    tensor = language_state["dense_readout_weights"]
    snapshot = executor.snapshot()

    assert result["accepted"] is True
    assert result["surface"] == "snn_language_dense_readout_tensor_materialization.v1"
    assert result["materializes_dense_tensor_weights"] is True
    assert result["resizes_network"] is True
    assert result["generates_text"] is False
    assert isinstance(tensor, torch.Tensor)
    assert list(tensor.shape) == [128, 128]
    assert str(tensor.device) == "cpu"
    assert float(tensor[1, 2].item()) == 0.5
    assert float(tensor[65, 66].item()) == 0.25
    assert language_state["dense_readout_layout"]["dense_resize_applied"] is True
    assert (
        language_state["dense_readout_layout"]["migration_status"]
        == "dense_readout_tensor_materialized"
    )
    assert snapshot["dense_readout_tensor"]["available"] is True
    assert snapshot["dense_readout_tensor"]["shape"] == [128, 128]
    assert snapshot["dense_readout_tensor"]["nonzero_count"] == 2
    assert runtime_state.state_revision == 1


def test_dense_readout_tensor_materialization_blocks_unavailable_cuda_before_checkpoint(
    tmp_path: Path,
) -> None:
    if torch.cuda.is_available():
        return
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {"sparse_transition_weights": {"1:2": 0.5}}
    checkpoint_calls = []
    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=lambda path: checkpoint_calls.append(path)
        or {"path": str(tmp_path / "dense-tensor.pt")},
        checkpoint_path=lambda: tmp_path / "dense-tensor.pt",
        verify_checkpoint=lambda path: path.exists(),
    )

    result = executor.apply_dense_readout_tensor_materialization(
        dense_readout_tensor_materialization_readiness=(
            _dense_readout_tensor_materialization_readiness()
        ),
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=True,
        requested_device="cuda",
    )

    assert result["accepted"] is False
    assert result["promotion_gate"]["required_evidence"][
        "requested_cuda_available_when_requested"
    ] is False
    assert checkpoint_calls == []
    assert "dense_readout_weights" not in language_state
    assert runtime_state.state_revision == 0


def test_dense_readout_training_loop_updates_dense_and_sparse_checkpointed_state(
    tmp_path: Path,
) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {
        "dense_readout_weights": torch.zeros((128, 128), dtype=torch.float32),
        "sparse_transition_weights": {},
    }

    def save_checkpoint(path: str | None) -> dict[str, str]:
        target = Path(path or tmp_path / "dense-training.pt")
        target.write_bytes(b"checkpoint")
        return {"path": str(target)}

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / "dense-training.pt",
        verify_checkpoint=lambda path: path.exists(),
    )

    result = executor.apply_dense_readout_training_loop(
        dense_readout_training_loop_preflight=_dense_readout_training_loop_preflight(),
        training_transitions=[
            {"transition_id": "t1", "pre_indices": [1, 2], "post_indices": [3, 4]},
        ],
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=True,
        checkpoint_path=str(tmp_path / "dense-training.pt"),
    )
    snapshot = executor.snapshot()

    assert result["accepted"] is True
    assert result["surface"] == "snn_language_dense_readout_training.v1"
    assert result["trains_runtime_model"] is True
    assert result["generates_text"] is False
    assert result["returns_trained_weights"] is False
    assert result["writes_checkpoint"] is True
    assert runtime_state.state_revision == 1
    assert snapshot["dense_readout_tensor"]["available"] is True
    assert snapshot["dense_readout_tensor"]["nonzero_count"] == 4
    assert set(language_state["sparse_transition_weights"]) == {
        "1:3",
        "1:4",
        "2:3",
        "2:4",
    }
    assert snapshot["dense_readout_training"]["training_count"] == 1


def test_dense_readout_training_loop_blocks_stale_revision_before_checkpoint(
    tmp_path: Path,
) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {
        "dense_readout_weights": torch.zeros((128, 128), dtype=torch.float32),
        "sparse_transition_weights": {},
    }
    checkpoint_calls = []
    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=lambda path: checkpoint_calls.append(path)
        or {"path": str(tmp_path / "dense-training.pt")},
        checkpoint_path=lambda: tmp_path / "dense-training.pt",
        verify_checkpoint=lambda path: path.exists(),
    )

    result = executor.apply_dense_readout_training_loop(
        dense_readout_training_loop_preflight=_dense_readout_training_loop_preflight(),
        training_transitions=[
            {"transition_id": "t1", "pre_indices": [1], "post_indices": [3]},
        ],
        expected_state_revision=1,
        operator_id="operator-test",
        confirmation=True,
        checkpoint_path=str(tmp_path / "dense-training.pt"),
    )

    assert result["accepted"] is False
    assert result["promotion_gate"]["required_evidence"]["expected_revision_current"] is False
    assert checkpoint_calls == []
    assert torch.count_nonzero(language_state["dense_readout_weights"]).item() == 0
    assert language_state["sparse_transition_weights"] == {}
    assert runtime_state.state_revision == 0


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
