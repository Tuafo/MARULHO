from __future__ import annotations

from collections.abc import Mapping as CollectionsMapping
from pathlib import Path
from threading import RLock
from typing import Any

import pytest
import torch

from marulho.service.runtime_state import RuntimeState
from marulho.service.snn_language_plasticity_executor import (
    SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT,
    SNN_LANGUAGE_DENSE_READOUT_TRAINING_INDEX_WINDOW_LIMIT,
    SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT,
    SNNLanguagePlasticityApplicationExecutor,
)
from marulho.service.transition_memory_source_window import (
    SNN_LANGUAGE_PLASTICITY_RUNTIME_TRANSITION_MEMORY_SOURCE_WINDOW_LIMIT,
)


class _CountingMapping(CollectionsMapping):
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.items_yield_count = 0

    def __getitem__(self, key: str) -> Any:
        return self._payload[key]

    def __iter__(self):
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)

    def items(self):
        for item in self._payload.items():
            self.items_yield_count += 1
            yield item


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
            "transition_budget": (
                SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT
            ),
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


def _thought_capacity_mutation_preflight(
    *,
    expected_state_revision: int = 0,
) -> dict[str, object]:
    return {
        "surface": (
            "snn_language_autonomous_snn_language_thought_"
            "capacity_mutation_preflight.v1"
        ),
        "accepted": True,
        "ready": True,
        "preflight_hash": "a" * 64,
        "requires_operator_approval": False,
        "loads_external_checkpoint": False,
        "runs_replay": False,
        "trains_runtime_model": False,
        "autonomous_snn_language_thought_capacity_mutation_preflight": {
            "thought_capacity_mutation_design_hash": "b" * 64,
            "structural_event_review_hash": "c" * 64,
            "memory_trace_hash": "d" * 64,
            "expected_state_revision": expected_state_revision,
            "requested_device": "cpu",
            "cuda_relayout_verified": True,
            "executor_ready": True,
            "checkpoint_saved": True,
            "restore_verified": True,
            "current_neuron_capacity": 64,
            "target_neuron_capacity": 66,
            "current_sparse_synapse_budget": 256,
            "target_sparse_synapse_budget": 258,
            "current_dense_shape": [64, 64],
            "target_dense_shape": [66, 66],
            "preserved_dense_shape": [64, 64],
            "zero_initialized_new_rows": 2,
            "zero_initialized_new_cols": 2,
            "growth_candidate_count": 2,
            "growth_candidates": [
                {"candidate_id": "growth-1", "applied_to_runtime": True},
                {"candidate_id": "growth-2", "applied_to_runtime": True},
            ],
            "prune_candidates": [
                {"candidate_id": "prune-1", "applied_to_runtime": True}
            ],
            "operator_approval_required": False,
            "execution_allowed": False,
        },
        "promotion_gate": {
            "eligible_for_autonomous_snn_language_thought_capacity_mutation_executor": True,
            "required_evidence": {
                "cuda_relayout_verified": True,
                "checkpoint_saved": True,
                "restore_verified": True,
                "executor_capability_available": True,
            },
        },
    }


def _thought_newborn_neuron_integration_preflight(
    *,
    expected_state_revision: int = 0,
) -> dict[str, object]:
    candidates = [
        {
            "integration_candidate_id": f"newborn-{target}",
            "integration_candidate_hash": chr(97 + offset) * 64,
            "source_candidate_hash": chr(99 + offset) * 64,
            "source_resolution_hash": chr(101 + offset) * 64,
            "source_neuron_index": source,
            "target_neuron_index": target,
            "synapse": f"{source}:{target}",
            "active_neuron_hash": chr(103 + offset) * 64,
            "spike_projection_hash": chr(105 + offset) * 64,
            "membrane_state_hash": chr(107 + offset) * 64,
            "coactivation_event_count": 7,
            "source_firing_rate_hz": 8.0,
            "target_firing_rate_hz": 4.0,
            "max_firing_rate_hz": 16.0,
            "critical_period_cycles": 64,
            "inactivity_prune_cycles": 128,
            "max_seed_synapses": 2,
            "max_initial_weight": 0.04,
            "connection_applied": False,
            "weight_applied": False,
            "critical_period_started": False,
        }
        for offset, (source, target) in enumerate(((4, 64), (5, 65)))
    ]
    return {
        "surface": (
            "snn_language_autonomous_snn_language_thought_"
            "newborn_neuron_integration_preflight.v1"
        ),
        "accepted": True,
        "ready": True,
        "preflight_hash": "p" * 64,
        "requires_operator_approval": False,
        "loads_external_checkpoint": False,
        "runs_replay": False,
        "trains_runtime_model": False,
        "autonomous_snn_language_thought_newborn_neuron_integration_preflight": {
            "thought_newborn_neuron_integration_design_hash": "d" * 64,
            "capacity_mutation_event_hash": "e" * 64,
            "observation_window_id": "window-newborn-1",
            "observation_window_hash": "w" * 64,
            "expected_state_revision": expected_state_revision,
            "current_neuron_capacity": 64,
            "target_neuron_capacity": 66,
            "newborn_neuron_indices": [64, 65],
            "resolved_candidate_count": 2,
            "resolved_integration_candidates": candidates,
            "source_indices_resolved": True,
            "operator_approval_required": False,
        },
        "promotion_gate": {
            "eligible_for_autonomous_snn_language_thought_"
            "newborn_neuron_integration_executor": True,
            "required_evidence": {
                "all_candidate_sources_resolved": True,
                "checkpoint_saved": True,
                "checkpoint_restore_verified": True,
                "executor_capability_available": True,
            },
        },
    }


def _thought_newborn_neuron_critical_period_learning_preflight(
    *,
    expected_state_revision: int = 0,
) -> dict[str, object]:
    cycle = {
        "synapse": "4:64",
        "source_neuron_index": 4,
        "target_neuron_index": 64,
        "newborn_integration_synapse_hash": "i" * 64,
        "initial_weight": 0.01,
        "min_weight": 0.0,
        "max_weight": 0.04,
        "max_learning_rate": 0.005,
        "depression_ratio": 0.5,
        "stdp_window_ms": 20.0,
        "target_firing_rate_hz": 4.0,
        "homeostatic_min_firing_rate_hz": 2.0,
        "homeostatic_max_firing_rate_hz": 6.0,
        "critical_period_cycles": 64,
        "critical_period_age_cycles": 0,
        "critical_period_cycles_remaining": 64,
        "minimum_survival_active_cycles": 16,
        "inactivity_prune_cycles": 128,
        "learning_rule": (
            "local_pre_post_timing_with_homeostatic_scaling"
        ),
        "survival_rule": (
            "activity_and_prediction_contribution_competition"
        ),
        "maturation_states": [
            "critical_period",
            "mature",
            "prune_eligible",
        ],
        "current_maturation_state": "critical_period",
        "critical_period_learning_candidate_hash": "c" * 64,
        "cycle_index": 1,
        "pre_spike_count": 2,
        "post_spike_count": 2,
        "causal_pair_count": 3,
        "anti_causal_pair_count": 1,
        "active_cycle": True,
        "newborn_firing_rate_hz": 4.0,
        "prediction_error": 0.2,
        "current_weight": 0.01,
        "proposed_weight_delta": 0.003125,
        "proposed_weight": 0.013125,
        "candidate_activity_hash": "a" * 64,
        "weight_update_applied": False,
        "critical_period_age_advanced": False,
        "maturation_decided": False,
        "pruning_applied": False,
    }
    cycle["critical_period_learning_cycle_hash"] = (
        SNNLanguagePlasticityApplicationExecutor._sha256_json(cycle)
    )
    return {
        "surface": (
            "snn_language_autonomous_snn_language_thought_newborn_neuron_"
            "critical_period_learning_preflight.v1"
        ),
        "accepted": True,
        "ready": True,
        "preflight_hash": "p" * 64,
        "requires_operator_approval": False,
        "loads_external_checkpoint": False,
        "runs_replay": False,
        "trains_runtime_model": False,
        "autonomous_snn_language_thought_newborn_neuron_"
        "critical_period_learning_preflight": {
            "thought_newborn_neuron_critical_period_learning_design_hash": (
                "d" * 64
            ),
            "newborn_neuron_integration_event_hash": "e" * 64,
            "expected_state_revision": expected_state_revision,
            "observation_window_id": "critical-window-1",
            "observation_window_hash": "o" * 64,
            "actual_device": "cpu",
            "tensor_is_cuda": False,
            "checkpoint_path": "memory://pre-learning",
            "resolved_cycle_count": 1,
            "resolved_learning_cycles": [cycle],
            "weight_updates_applied": False,
            "critical_period_age_advanced": False,
            "maturation_decided": False,
            "pruning_applied": False,
            "operator_approval_required": False,
        },
        "promotion_gate": {
            "eligible_for_autonomous_snn_language_thought_newborn_neuron_"
            "critical_period_learning_executor": True
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


def test_snapshot_bounds_transition_memory_source_window_without_full_mapping_scan(
    tmp_path: Path,
) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    source_limit = SNN_LANGUAGE_PLASTICITY_RUNTIME_TRANSITION_MEMORY_SOURCE_WINDOW_LIMIT
    entry_count = source_limit + 19
    weights = _CountingMapping(
        {f"{index}:{index + 1}": float(index) / 100.0 for index in range(entry_count)}
    )
    provenance = _CountingMapping(
        {
            f"{index}:{index + 1}": {"source": "unit", "index": index}
            for index in range(entry_count)
        }
    )
    language_state = {
        "sparse_transition_weights": weights,
        "synapse_provenance_by_key": provenance,
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
    source_window = snapshot["transition_memory_source_window"]

    assert weights.items_yield_count == source_limit
    assert provenance.items_yield_count == source_limit
    assert len(snapshot["sparse_transition_weights"]) == source_limit
    assert len(snapshot["synapse_provenance_by_key"]) == source_limit
    assert snapshot["sparse_transition_weight_count"] == entry_count
    assert snapshot["synapse_provenance_count"] == entry_count
    assert snapshot["source_sparse_transition_weight_count"] == source_limit
    assert snapshot["source_synapse_provenance_count"] == source_limit
    assert snapshot["transition_memory_count_scope"] == "bounded_source_window"
    assert source_window["surface"] == (
        "bounded_snn_language_plasticity_runtime_transition_memory_source_window.v1"
    )
    assert source_window["source_payload_truncated"] is True
    assert source_window["source_truncated_counts"]["sparse_transition_weights"] == 19
    assert source_window["source_truncated_counts"]["synapse_provenance_by_key"] == 19
    assert source_window["global_candidate_scan"] is False
    assert source_window["runs_live_tick"] is False
    assert source_window["runs_every_token"] is False
    assert source_window["language_reasoning"] is False
    assert source_window["archival_storage_device"] == "cpu"
    assert source_window["gpu_resident_archival_metadata"] is False


def test_thought_capacity_mutation_grows_tensor_and_dynamic_capacity(
    tmp_path: Path,
) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    existing = torch.zeros((64, 64), dtype=torch.float32)
    existing[1, 2] = 0.5
    language_state = {
        "dense_readout_weights": existing,
        "sparse_transition_weights": {"1:2": 0.5},
    }

    def save_checkpoint(path: str | None) -> dict[str, str]:
        target = Path(path or tmp_path / "thought-capacity.pt")
        target.write_bytes(b"checkpoint")
        return {"path": str(target)}

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / "thought-capacity.pt",
        verify_checkpoint=lambda path: path.exists(),
        publish_committed_checkpoint=lambda path, operation: {
            "checkpoint_path": str(path),
            "operation": operation,
        },
    )

    result = executor.apply_autonomous_snn_language_thought_capacity_mutation(
        autonomous_snn_language_thought_capacity_mutation_preflight=(
            _thought_capacity_mutation_preflight()
        ),
        expected_state_revision=0,
        checkpoint_path=str(tmp_path / "thought-capacity.pt"),
        requested_device="cpu",
    )
    tensor = language_state["dense_readout_weights"]
    capacity = language_state["language_capacity"]
    event = result[
        "autonomous_snn_language_thought_capacity_mutation_event"
    ]
    snapshot = executor.snapshot()

    assert result["accepted"] is True
    assert result["surface"] == (
        "snn_language_autonomous_snn_language_thought_capacity_mutation_executor.v1"
    )
    assert result["requires_operator_approval"] is False
    assert result["mutates_runtime_state"] is True
    assert result["writes_checkpoint"] is True
    assert result["resizes_network"] is True
    assert result["adds_neurons"] is True
    assert result["adds_synapses"] is False
    assert result["prunes_network"] is False
    assert result["runs_replay"] is False
    assert result["trains_runtime_model"] is False
    assert result["applies_plasticity"] is False
    assert result["checkpoint_transaction"]["restore_verified"] is True
    assert isinstance(tensor, torch.Tensor)
    assert list(tensor.shape) == [66, 66]
    assert str(tensor.device) == "cpu"
    assert float(tensor[1, 2].item()) == 0.5
    assert int(torch.count_nonzero(tensor[64:, :]).item()) == 0
    assert int(torch.count_nonzero(tensor[:64, 64:]).item()) == 0
    assert capacity["language_neuron_count"] == 66
    assert capacity["sparse_edge_budget"] == 258
    assert capacity["outgoing_fanout_budget"] == 16
    assert capacity["dynamic_capacity_enabled"] is True
    assert capacity["capacity_expansion_count"] == 1
    assert capacity["resizes_network"] is True
    assert capacity["adds_neurons"] is True
    assert event["added_neuron_capacity"] == 2
    assert event["added_sparse_synapse_budget"] == 2
    assert event["new_region_zero_initialized"] is True
    assert len(event["capacity_mutation_event_hash"]) == 64
    assert snapshot["language_neuron_count"] == 66
    assert snapshot["sparse_edge_budget"] == 258
    assert snapshot["language_capacity"]["dynamic_capacity_enabled"] is True
    assert snapshot["language_capacity_mutation_count"] == 1
    assert snapshot["thought_capacity_mutation_count"] == 1
    assert (
        snapshot["last_language_capacity_mutation"]
        == snapshot["last_thought_capacity_mutation"]
    )
    assert (
        snapshot["recent_language_capacity_mutations"]
        == snapshot["recent_thought_capacity_mutations"]
    )
    assert snapshot["canonical_field_names"][
        "language_capacity_mutation_count"
    ] == "canonical_alias_for_legacy_thought_capacity_mutation_count"
    assert runtime_state.state_revision == 1


def test_thought_capacity_mutation_blocks_stale_revision_before_checkpoint(
    tmp_path: Path,
) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {
        "dense_readout_weights": torch.zeros((64, 64), dtype=torch.float32),
        "sparse_transition_weights": {},
    }
    checkpoint_calls: list[str | None] = []
    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=lambda path: checkpoint_calls.append(path)
        or {"path": str(tmp_path / "thought-capacity.pt")},
        checkpoint_path=lambda: tmp_path / "thought-capacity.pt",
        verify_checkpoint=lambda path: path.exists(),
    )

    result = executor.apply_autonomous_snn_language_thought_capacity_mutation(
        autonomous_snn_language_thought_capacity_mutation_preflight=(
            _thought_capacity_mutation_preflight(expected_state_revision=1)
        ),
        expected_state_revision=1,
        requested_device="cpu",
    )

    assert result["accepted"] is False
    assert result["requires_operator_approval"] is False
    assert result["promotion_gate"]["required_evidence"][
        "expected_revision_current"
    ] is False
    assert checkpoint_calls == []
    assert list(language_state["dense_readout_weights"].shape) == [64, 64]
    assert "language_capacity" not in language_state
    assert runtime_state.state_revision == 0


def test_thought_newborn_neuron_integration_adds_checkpointed_seed_edges(
    tmp_path: Path,
) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    dense = torch.zeros((66, 66), dtype=torch.float32)
    dense[1, 2] = 0.5
    language_state = {
        "dense_readout_weights": dense,
        "sparse_transition_weights": {"1:2": 0.5},
        "language_capacity": {
            "surface": "snn_language_capacity_state.v1",
            "language_neuron_count": 66,
            "sparse_edge_budget": 258,
            "outgoing_fanout_budget": 16,
            "dynamic_capacity_enabled": True,
            "capacity_expansion_count": 1,
        },
    }

    def save_checkpoint(path: str | None) -> dict[str, str]:
        target = Path(path or tmp_path / "newborn-integration.pt")
        target.write_bytes(b"checkpoint")
        return {"path": str(target)}

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / "newborn-integration.pt",
        verify_checkpoint=lambda path: path.exists(),
        publish_committed_checkpoint=lambda path, operation: {
            "checkpoint_path": str(path),
            "operation": operation,
        },
    )

    result = executor.apply_autonomous_snn_language_thought_newborn_neuron_integration(
        autonomous_snn_language_thought_newborn_neuron_integration_preflight=(
            _thought_newborn_neuron_integration_preflight()
        ),
        expected_state_revision=0,
        checkpoint_path=str(tmp_path / "newborn-integration.pt"),
    )
    event = result[
        "autonomous_snn_language_thought_newborn_neuron_integration_event"
    ]
    snapshot = executor.snapshot()

    assert result["accepted"] is True
    assert result["surface"] == (
        "snn_language_autonomous_snn_language_thought_"
        "newborn_neuron_integration_executor.v1"
    )
    assert result["requires_operator_approval"] is False
    assert result["mutates_runtime_state"] is True
    assert result["writes_checkpoint"] is True
    assert result["resizes_network"] is False
    assert result["adds_neurons"] is False
    assert result["adds_synapses"] is True
    assert result["applies_plasticity"] is True
    assert result["runs_replay"] is False
    assert result["trains_runtime_model"] is False
    assert result["checkpoint_transaction"]["restore_verified"] is True
    assert language_state["sparse_transition_weights"]["4:64"] == 0.01
    assert language_state["sparse_transition_weights"]["5:65"] == 0.01
    integrated = language_state["dense_readout_weights"]
    assert isinstance(integrated, torch.Tensor)
    assert float(integrated[4, 64].item()) == pytest.approx(0.01)
    assert float(integrated[5, 65].item()) == pytest.approx(0.01)
    assert float(integrated[1, 2].item()) == 0.5
    assert event["integrated_synapse_count"] == 2
    assert event["critical_period_started"] is True
    assert len(event["newborn_neuron_integration_event_hash"]) == 64
    assert all(
        item["connection_applied"] is True
        and item["weight_applied"] is True
        and item["critical_period_started"] is True
        and len(item["newborn_integration_synapse_hash"]) == 64
        for item in event["integrated_synapses"]
    )
    assert language_state["synapse_provenance_by_key"]["4:64"][
        "provenance_type"
    ] == "newborn_neuron_integration"
    assert snapshot["thought_newborn_neuron_integration_count"] == 1
    assert snapshot["last_thought_newborn_neuron_integration"] == event
    assert snapshot["newborn_integration_dense_samples"] == [
        {
            "synapse": "4:64",
            "source_neuron_index": 4,
            "target_neuron_index": 64,
            "weight": pytest.approx(0.01),
        },
        {
            "synapse": "5:65",
            "source_neuron_index": 5,
            "target_neuron_index": 65,
            "weight": pytest.approx(0.01),
        },
    ]
    assert runtime_state.state_revision == 1


def test_thought_newborn_neuron_integration_blocks_stale_revision_before_checkpoint(
    tmp_path: Path,
) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {
        "dense_readout_weights": torch.zeros((66, 66), dtype=torch.float32),
        "sparse_transition_weights": {},
        "language_capacity": {
            "language_neuron_count": 66,
            "sparse_edge_budget": 258,
            "outgoing_fanout_budget": 16,
        },
    }
    checkpoint_calls: list[str | None] = []
    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=lambda path: checkpoint_calls.append(path)
        or {"path": str(tmp_path / "newborn-integration.pt")},
        checkpoint_path=lambda: tmp_path / "newborn-integration.pt",
        verify_checkpoint=lambda path: path.exists(),
    )

    result = executor.apply_autonomous_snn_language_thought_newborn_neuron_integration(
        autonomous_snn_language_thought_newborn_neuron_integration_preflight=(
            _thought_newborn_neuron_integration_preflight(
                expected_state_revision=1
            )
        ),
        expected_state_revision=1,
    )

    assert result["accepted"] is False
    assert result["requires_operator_approval"] is False
    assert result["promotion_gate"]["required_evidence"][
        "expected_revision_current"
    ] is False
    assert checkpoint_calls == []
    assert language_state["sparse_transition_weights"] == {}
    assert int(torch.count_nonzero(language_state["dense_readout_weights"])) == 0
    assert runtime_state.state_revision == 0


def test_thought_newborn_neuron_critical_period_learning_applies_local_cycle(
    tmp_path: Path,
) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    dense = torch.zeros((66, 66), dtype=torch.float32)
    dense[4, 64] = 0.01
    language_state = {
        "dense_readout_weights": dense,
        "sparse_transition_weights": {"4:64": 0.01},
        "synapse_provenance_by_key": {
            "4:64": {
                "provenance_type": "newborn_neuron_integration",
                "newborn_integration_synapse_hash": "i" * 64,
            }
        },
        "language_capacity": {
            "language_neuron_count": 66,
            "sparse_edge_budget": 258,
            "outgoing_fanout_budget": 16,
        },
    }

    def save_checkpoint(path: str | None) -> dict[str, str]:
        target = Path(path or tmp_path / "critical-period.pt")
        target.write_bytes(b"checkpoint")
        return {"path": str(target)}

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / "critical-period.pt",
        verify_checkpoint=lambda path: path.exists(),
        publish_committed_checkpoint=lambda path, operation: {
            "checkpoint_path": str(path),
            "operation": operation,
        },
    )

    result = executor.apply_autonomous_snn_language_thought_newborn_neuron_critical_period_learning(
        autonomous_snn_language_thought_newborn_neuron_critical_period_learning_preflight=(
            _thought_newborn_neuron_critical_period_learning_preflight()
        ),
        expected_state_revision=0,
        checkpoint_path=str(tmp_path / "critical-period.pt"),
    )
    event = result[
        "autonomous_snn_language_thought_newborn_neuron_"
        "critical_period_learning_event"
    ]
    snapshot = executor.snapshot()

    assert result["accepted"] is True
    assert result["requires_operator_approval"] is False
    assert result["mutates_runtime_state"] is True
    assert result["writes_checkpoint"] is True
    assert result["applies_plasticity"] is True
    assert result["adds_synapses"] is False
    assert result["prunes_network"] is False
    assert result["checkpoint_transaction"]["restore_verified"] is True
    assert language_state["sparse_transition_weights"]["4:64"] == pytest.approx(
        0.013125
    )
    assert float(
        language_state["dense_readout_weights"][4, 64].item()
    ) == pytest.approx(0.013125)
    applied = event["applied_learning_cycles"][0]
    assert applied["cycle_index"] == 1
    assert applied["critical_period_age_cycles"] == 1
    assert applied["critical_period_cycles_remaining"] == 63
    assert applied["active_cycle_count"] == 1
    assert applied["current_maturation_state"] == "critical_period"
    assert applied["weight_update_applied"] is True
    assert applied["maturation_decided"] is False
    assert event["critical_period_synapse_count"] == 1
    assert event["mature_synapse_count"] == 0
    assert event["prune_eligible_synapse_count"] == 0
    assert snapshot[
        "thought_newborn_neuron_critical_period_learning_cycle_count"
    ] == 1
    assert snapshot[
        "newborn_neuron_critical_period_state_by_synapse"
    ]["4:64"]["critical_period_age_cycles"] == 1
    assert snapshot["critical_period_learning_dense_samples"][0][
        "weight"
    ] == pytest.approx(0.013125)
    assert language_state["synapse_provenance_by_key"]["4:64"][
        "current_maturation_state"
    ] == "critical_period"
    assert runtime_state.state_revision == 1


@pytest.mark.parametrize(
    ("active_cycle", "prior_active_cycles", "expected_state"),
    (
        (True, 15, "mature"),
        (False, 0, "prune_eligible"),
    ),
)
def test_thought_newborn_neuron_critical_period_learning_decides_terminal_state_without_pruning(
    tmp_path: Path,
    active_cycle: bool,
    prior_active_cycles: int,
    expected_state: str,
) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    dense = torch.zeros((66, 66), dtype=torch.float32)
    dense[4, 64] = 0.01
    language_state = {
        "dense_readout_weights": dense,
        "sparse_transition_weights": {"4:64": 0.01},
        "synapse_provenance_by_key": {
            "4:64": {
                "provenance_type": "newborn_neuron_integration",
                "newborn_integration_synapse_hash": "i" * 64,
            }
        },
        "thought_newborn_neuron_critical_period_learning": {
            "learning_cycle_count": 63,
            "by_synapse": {
                "4:64": {
                    "critical_period_age_cycles": 63,
                    "active_cycle_count": prior_active_cycles,
                    "inactive_cycle_count": 0,
                    "current_maturation_state": "critical_period",
                }
            },
        },
        "language_capacity": {
            "language_neuron_count": 66,
            "sparse_edge_budget": 258,
            "outgoing_fanout_budget": 16,
        },
    }
    preflight = (
        _thought_newborn_neuron_critical_period_learning_preflight()
    )
    body = preflight[
        "autonomous_snn_language_thought_newborn_neuron_"
        "critical_period_learning_preflight"
    ]
    assert isinstance(body, dict)
    cycle = body["resolved_learning_cycles"][0]
    assert isinstance(cycle, dict)
    cycle.update(
        {
            "cycle_index": 64,
            "critical_period_age_cycles": 63,
            "critical_period_cycles_remaining": 1,
            "active_cycle": active_cycle,
            "pre_spike_count": 2 if active_cycle else 0,
            "post_spike_count": 2 if active_cycle else 0,
            "causal_pair_count": 3 if active_cycle else 0,
            "anti_causal_pair_count": 1 if active_cycle else 0,
            "proposed_weight_delta": (
                0.003125 if active_cycle else 0.0
            ),
            "proposed_weight": 0.013125 if active_cycle else 0.01,
        }
    )
    cycle["critical_period_learning_cycle_hash"] = (
        SNNLanguagePlasticityApplicationExecutor._sha256_json(
            {
                key: value
                for key, value in cycle.items()
                if key != "critical_period_learning_cycle_hash"
            }
        )
    )

    def save_checkpoint(path: str | None) -> dict[str, str]:
        target = Path(path or tmp_path / f"{expected_state}.pt")
        target.write_bytes(b"checkpoint")
        return {"path": str(target)}

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / f"{expected_state}.pt",
        verify_checkpoint=lambda path: path.exists(),
        publish_committed_checkpoint=lambda path, operation: {
            "checkpoint_path": str(path),
            "operation": operation,
        },
    )

    result = executor.apply_autonomous_snn_language_thought_newborn_neuron_critical_period_learning(
        autonomous_snn_language_thought_newborn_neuron_critical_period_learning_preflight=(
            preflight
        ),
        expected_state_revision=0,
        checkpoint_path=str(tmp_path / f"{expected_state}.pt"),
    )
    event = result[
        "autonomous_snn_language_thought_newborn_neuron_"
        "critical_period_learning_event"
    ]
    applied = event["applied_learning_cycles"][0]

    assert result["accepted"] is True
    assert applied["critical_period_age_cycles"] == 64
    assert applied["critical_period_cycles_remaining"] == 0
    assert applied["maturation_decided"] is True
    assert applied["current_maturation_state"] == expected_state
    assert applied["pruning_applied"] is False
    assert "4:64" in language_state["sparse_transition_weights"]
    assert result["prunes_network"] is False
    if expected_state == "mature":
        assert event["mature_synapse_count"] == 1
        assert event["prune_eligible_synapse_count"] == 0
    else:
        assert event["mature_synapse_count"] == 0
        assert event["prune_eligible_synapse_count"] == 1


def test_thought_newborn_synapse_pruning_removes_edge_and_preserves_tombstone(
    tmp_path: Path,
) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    dense = torch.zeros((66, 66), dtype=torch.float32)
    dense[4, 64] = 0.01
    application_hash = "l" * 64
    language_state = {
        "dense_readout_weights": dense,
        "sparse_transition_weights": {"4:64": 0.01},
        "synapse_provenance_by_key": {
            "4:64": {
                "provenance_type": "newborn_neuron_integration",
                "current_weight": 0.01,
                "current_maturation_state": "prune_eligible",
            }
        },
        "thought_newborn_neuron_critical_period_learning": {
            "by_synapse": {
                "4:64": {
                    "synapse": "4:64",
                    "critical_period_cycles_remaining": 0,
                    "current_maturation_state": "prune_eligible",
                    "maturation_decided": True,
                    "critical_period_learning_application_hash": (
                        application_hash
                    ),
                }
            }
        },
        "language_capacity": {
            "language_neuron_count": 66,
            "sparse_edge_budget": 258,
            "outgoing_fanout_budget": 16,
        },
    }
    candidate = {
        "synapse": "4:64",
        "source_neuron_index": 4,
        "target_neuron_index": 64,
        "current_weight": 0.01,
        "critical_period_age_cycles": 64,
        "critical_period_cycles": 64,
        "active_cycle_count": 0,
        "inactive_cycle_count": 64,
        "minimum_survival_active_cycles": 16,
        "critical_period_learning_application_hash": application_hash,
        "newborn_integration_synapse_hash": "i" * 64,
        "terminal_maturation_state": "prune_eligible",
        "pruning_applied": False,
    }
    candidate["newborn_synapse_pruning_candidate_hash"] = (
        SNNLanguagePlasticityApplicationExecutor._sha256_json(candidate)
    )
    preflight = {
        "surface": (
            "snn_language_autonomous_snn_language_thought_newborn_"
            "synapse_pruning_preflight.v1"
        ),
        "accepted": True,
        "ready": True,
        "preflight_hash": "p" * 64,
        "requires_operator_approval": False,
        "autonomous_snn_language_thought_newborn_synapse_"
        "pruning_preflight": {
            "newborn_synapse_pruning_design_hash": "d" * 64,
            "maturation_outcome_review_hash": "r" * 64,
            "expected_state_revision": 0,
            "checkpoint_path": str(tmp_path / "prune.pt"),
            "resolved_prune_count": 1,
            "resolved_prune_candidates": [candidate],
            "operator_approval_required": False,
        },
        "promotion_gate": {
            "eligible_for_autonomous_snn_language_thought_newborn_"
            "synapse_pruning_executor": True
        },
    }

    def save_checkpoint(path: str | None) -> dict[str, str]:
        target = Path(path or tmp_path / "prune.pt")
        target.write_bytes(b"checkpoint")
        return {"path": str(target)}

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / "prune.pt",
        verify_checkpoint=lambda path: path.exists(),
        publish_committed_checkpoint=lambda path, operation: {
            "checkpoint_path": str(path),
            "operation": operation,
        },
    )

    result = executor.apply_autonomous_snn_language_thought_newborn_synapse_pruning(
        autonomous_snn_language_thought_newborn_synapse_pruning_preflight=preflight,
        expected_state_revision=0,
    )

    assert result["accepted"] is True
    assert result["prunes_network"] is True
    assert runtime_state.state_revision == 1
    assert "4:64" not in language_state["sparse_transition_weights"]
    assert float(language_state["dense_readout_weights"][4, 64]) == 0.0
    assert "4:64" not in language_state["synapse_provenance_by_key"]
    tombstone = language_state["pruned_synapse_provenance_by_key"]["4:64"]
    assert tombstone["live_synapse"] is False
    assert tombstone["final_weight"] == pytest.approx(0.01)
    assert (
        language_state["language_capacity"]["language_neuron_count"]
        == 66
    )
    snapshot = executor.snapshot()
    assert snapshot["language_newborn_synapse_pruning_count"] == 1
    assert snapshot["language_newborn_synapse_pruned_count_total"] == 1
    assert snapshot["thought_newborn_synapse_pruning_count"] == 1
    assert snapshot["thought_newborn_synapse_pruned_count_total"] == 1
    assert (
        snapshot["last_language_newborn_synapse_pruning"]
        == snapshot["last_thought_newborn_synapse_pruning"]
    )
    assert (
        snapshot["recent_language_newborn_synapse_pruning"]
        == snapshot["recent_thought_newborn_synapse_pruning"]
    )


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
    assert result["memory_budget"] == {
        "max_training_transition_records": (
            SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT
        ),
        "max_pre_indices_per_transition": (
            SNN_LANGUAGE_DENSE_READOUT_TRAINING_INDEX_WINDOW_LIMIT
        ),
        "max_post_indices_per_transition": (
            SNN_LANGUAGE_DENSE_READOUT_TRAINING_INDEX_WINDOW_LIMIT
        ),
    }
    assert result["training_transition_source_window"]["source_window_count"] == 1
    assert (
        result["training_transition_source_window"]["source_payload_truncated"]
        is False
    )
    assert (
        result["training_transition_index_source_window"][
            "max_pre_index_window_count"
        ]
        == 2
    )
    assert (
        result["training_transition_index_source_window"][
            "source_payload_truncated"
        ]
        is False
    )
    assert result["dense_readout_training"][
        "training_transition_source_window"
    ] == result["training_transition_source_window"]
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


def test_dense_readout_training_loop_blocks_oversized_transition_window_before_checkpoint(
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
    transitions = [
        {"transition_id": f"t{index}", "pre_indices": [1], "post_indices": [2]}
        for index in range(
            SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT + 1
        )
    ]

    result = executor.apply_dense_readout_training_loop(
        dense_readout_training_loop_preflight=_dense_readout_training_loop_preflight(),
        training_transitions=transitions,
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=True,
        checkpoint_path=str(tmp_path / "dense-training.pt"),
    )

    required = result["promotion_gate"]["required_evidence"]
    source_window = required["training_transition_source_window"]
    assert result["accepted"] is False
    assert required["training_transition_payload_not_truncated"] is False
    assert source_window["source_window_count"] == (
        SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT
    )
    assert source_window["source_payload_truncated"] is True
    assert checkpoint_calls == []
    assert torch.count_nonzero(language_state["dense_readout_weights"]).item() == 0
    assert language_state["sparse_transition_weights"] == {}
    assert runtime_state.state_revision == 0


def test_dense_readout_training_loop_blocks_oversized_index_window_before_checkpoint(
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
            {
                "transition_id": "oversized-index-window",
                "pre_indices": list(
                    range(
                        SNN_LANGUAGE_DENSE_READOUT_TRAINING_INDEX_WINDOW_LIMIT
                        + 1
                    )
                ),
                "post_indices": [2],
            }
        ],
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=True,
        checkpoint_path=str(tmp_path / "dense-training.pt"),
    )

    required = result["promotion_gate"]["required_evidence"]
    index_window = required["training_transition_index_source_window"]
    first_transition = index_window["per_transition_windows"][0]
    assert result["accepted"] is False
    assert required["training_transition_index_payload_not_truncated"] is False
    assert index_window["source_payload_truncated"] is True
    assert first_transition["pre_indices"]["source_window_count"] == (
        SNN_LANGUAGE_DENSE_READOUT_TRAINING_INDEX_WINDOW_LIMIT
    )
    assert first_transition["pre_indices"]["source_payload_truncated"] is True
    assert checkpoint_calls == []
    assert torch.count_nonzero(language_state["dense_readout_weights"]).item() == 0
    assert language_state["sparse_transition_weights"] == {}
    assert runtime_state.state_revision == 0


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
    lineage_summary = language_state["applied_replay_lineage_incremental_summary"]
    assert (
        lineage_summary["surface"]
        == "snn_applied_replay_lineage_incremental_summary.v1"
    )
    assert lineage_summary["applied_replay_lineage_count"] == 2
    assert lineage_summary["complete_applied_replay_lineage_count"] == 2
    assert lineage_summary["incomplete_applied_replay_lineage_count"] == 0
    assert lineage_summary["lineage_material_hash"]
    assert lineage_summary["full_provenance_scan"] is False
    assert lineage_summary["source_record_scan_count"] == 0
    assert lineage_summary["archival_metadata_device"] == "cpu"
    assert lineage_summary["gpu_used"] is False
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


def test_regeneration_blocks_truncated_candidate_payload_before_checkpoint(tmp_path: Path) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {"sparse_transition_weights": {}}
    checkpoint_calls = []
    oversized_count = SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT + 19
    candidates = [
        {
            "pre_index": index,
            "post_index": index + 1,
            "initial_weight": 0.01,
            "locality_distance": 1,
        }
        for index in range(oversized_count)
    ]

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
        regeneration_proposal=_regeneration_proposal(*candidates),
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=True,
    )

    assert result["accepted"] is False
    evidence = result["promotion_gate"]["required_evidence"]
    source_window = evidence["candidate_source_window"]
    assert evidence["candidate_source_window_bounded"] is True
    assert evidence["candidate_payload_not_truncated"] is False
    assert source_window["source_window_count"] == SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
    assert source_window["source_total_count"] == oversized_count
    assert source_window["source_payload_truncated"] is True
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


def test_live_application_blocks_truncated_synapse_payload_before_checkpoint(tmp_path: Path) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state = {"sparse_transition_weights": {}}
    checkpoint_calls = []
    oversized_count = SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT + 17
    bounded_synapses = [
        {"pre_index": index, "post_index": index + 1}
        for index in range(oversized_count)
    ]
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
            "max_abs_weight_delta": 0.01,
            "pressure_before": 0.9,
            "pressure_after": 0.8,
            "bounded_synapses": bounded_synapses,
        },
        expected_state_revision=0,
        operator_id="operator-test",
        confirmation=True,
    )

    assert result["accepted"] is False
    evidence = result["promotion_gate"]["required_evidence"]
    source_window = evidence["synapse_source_window"]
    assert evidence["synapse_source_window_bounded"] is True
    assert evidence["synapse_payload_not_truncated"] is False
    assert source_window["source_window_count"] == SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
    assert source_window["source_total_count"] == oversized_count
    assert source_window["source_payload_truncated"] is True
    assert checkpoint_calls == []
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
