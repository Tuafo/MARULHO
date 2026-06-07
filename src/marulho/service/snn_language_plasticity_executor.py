from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from marulho.service.runtime_state import RuntimeState

_LANGUAGE_NEURON_COUNT = 64
_MAX_STRUCTURAL_EDGES_PER_EVENT = 32
_MAX_OUTGOING_FANOUT = 16
_MAX_SPARSE_TRANSITION_EDGES = 256
_MAX_OUTGOING_ROW_MASS = 1.0
_LANGUAGE_CAPACITY_SURFACE = "snn_language_capacity_state.v1"
_DENSE_READOUT_LAYOUT_SURFACE = "snn_language_dense_readout_layout_state.v1"


class SNNLanguagePlasticityApplicationExecutor:
    """Checkpoint-backed mutation boundary for MARULHO-owned SNN language plasticity."""

    def __init__(
        self,
        *,
        lock: Any,
        runtime_state: RuntimeState,
        language_plasticity_state: Callable[[], dict[str, Any]],
        save_checkpoint: Callable[[str | None], dict[str, Any]],
        checkpoint_path: Callable[[], Path],
        verify_checkpoint: Callable[[Path], bool],
        verify_regeneration_permit: Callable[[Mapping[str, Any]], bool] | None = None,
        verify_checkpoint_snapshot: Callable[[Path, Mapping[str, Any], int], bool] | None = None,
        publish_committed_checkpoint: Callable[[Path, str], Mapping[str, Any]] | None = None,
    ) -> None:
        self._lock = lock
        self._runtime_state = runtime_state
        self._language_plasticity_state = language_plasticity_state
        self._save_checkpoint = save_checkpoint
        self._checkpoint_path = checkpoint_path
        self._verify_checkpoint = verify_checkpoint
        self._verify_regeneration_permit = verify_regeneration_permit or (lambda _proposal: False)
        self._verify_checkpoint_snapshot = verify_checkpoint_snapshot or (lambda _path, _state, _revision: True)
        self._publish_committed_checkpoint = publish_committed_checkpoint or (lambda path, operation: {"path": str(path), "operation": operation})

    def apply_live_application(
        self,
        *,
        live_application_readiness: Mapping[str, Any],
        shadow_delta: Mapping[str, Any],
        expected_state_revision: int,
        operator_id: str,
        confirmation: bool,
        checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        """Apply a bounded sparse transition update after fail-closed preflight."""

        with self._lock:
            before_revision = int(self._runtime_state.state_revision)
            readiness = dict(live_application_readiness)
            delta = dict(shadow_delta)
            preflight = self._preflight(
                readiness=readiness,
                delta=delta,
                before_revision=before_revision,
                expected_state_revision=expected_state_revision,
                operator_id=operator_id,
                confirmation=confirmation,
            )
            if not preflight["accepted"]:
                return preflight

            checkpoint_state = deepcopy(self._language_plasticity_state())
            before_dirty_state = bool(self._runtime_state.dirty_state)
            checkpoint = self._save_checkpoint(checkpoint_path)
            checkpoint_file = Path(str(checkpoint.get("path") or self._checkpoint_path()))
            checkpoint_verified = self._verify_checkpoint_transaction(checkpoint_file, checkpoint_state, before_revision)
            if not checkpoint_verified:
                return self._blocked(
                    reason="checkpoint_save_missing",
                    before_revision=before_revision,
                    required_evidence={
                        **preflight["required_evidence"],
                        "pre_update_checkpoint_saved": checkpoint_file.exists(),
                        "pre_update_checkpoint_restore_verified": checkpoint_verified,
                    },
                )

            state = self._language_plasticity_state()
            weights = state.setdefault("sparse_transition_weights", {})
            applied_synapses = []
            delta_value = float(delta.get("max_abs_weight_delta", 0.0) or 0.0)
            for synapse in list(delta.get("bounded_synapses") or []):
                if not isinstance(synapse, Mapping):
                    continue
                pre_index = int(synapse.get("pre_index", 0) or 0)
                post_index = int(synapse.get("post_index", 0) or 0)
                key = f"{pre_index}:{post_index}"
                previous = float(weights.get(key, 0.0) or 0.0)
                updated = previous + delta_value
                weights[key] = updated
                provenance = {
                    "sequence_id": synapse.get("sequence_id"),
                    "grounded": bool(synapse.get("grounded", True)),
                    "readout_evidence_hash": synapse.get("readout_evidence_hash"),
                    "prediction_hash": synapse.get("prediction_hash"),
                    "transition_memory_evaluation_hash": synapse.get(
                        "transition_memory_evaluation_hash"
                    ),
                    "persistent_transition_weights_hash": synapse.get(
                        "persistent_transition_weights_hash"
                    ),
                    "source_pre_indices": deepcopy(synapse.get("source_pre_indices")),
                    "source_post_indices": deepcopy(synapse.get("source_post_indices")),
                    "source_active_indices": deepcopy(synapse.get("source_active_indices")),
                }
                applied_synapses.append(
                    {
                        "pre_index": pre_index,
                        "post_index": post_index,
                        **{
                            field: value
                            for field, value in provenance.items()
                            if value is not None
                        },
                        "previous_weight": previous,
                        "updated_weight": updated,
                        "delta": delta_value,
                    }
                )

            state["applied_update_count"] = int(state.get("applied_update_count", 0) or 0) + 1
            state["last_applied_at"] = datetime.now(timezone.utc).isoformat()
            state["last_operator_id"] = operator_id
            committed_checkpoint_file = self._committed_checkpoint_path(checkpoint_file, "live_application")
            state["last_checkpoint_path"] = str(committed_checkpoint_file)
            provenance_by_key = state.setdefault("synapse_provenance_by_key", {})
            for applied in applied_synapses:
                key = f"{int(applied['pre_index'])}:{int(applied['post_index'])}"
                provenance_by_key[key] = {
                    field: deepcopy(applied.get(field))
                    for field in (
                        "sequence_id",
                        "grounded",
                        "readout_evidence_hash",
                        "prediction_hash",
                        "transition_memory_evaluation_hash",
                        "persistent_transition_weights_hash",
                        "source_pre_indices",
                        "source_post_indices",
                        "source_active_indices",
                    )
                    if applied.get(field) is not None
                }
            live_application = state.setdefault("live_application", {})
            recent_events = list(live_application.get("recent_events") or [])
            event = {
                "operator_id": operator_id,
                "applied_at": state["last_applied_at"],
                "before_state_revision": before_revision,
                "after_state_revision": before_revision + 1,
                "checkpoint_path": str(checkpoint_file),
                "staged_committed_checkpoint_path": str(committed_checkpoint_file),
                "applied_synapses": deepcopy(applied_synapses),
            }
            live_application["last_application"] = deepcopy(event)
            live_application["recent_events"] = [*recent_events, deepcopy(event)][-8:]
            self._runtime_state.mark_mutated()
            after_revision = int(self._runtime_state.state_revision)
            commit = self._commit_mutation(
                committed_checkpoint_file=committed_checkpoint_file,
                operation="live_application",
                before_state=checkpoint_state,
                before_revision=before_revision,
                before_dirty_state=before_dirty_state,
            )
            if not commit["committed"]:
                return self._blocked(
                    reason="post_update_checkpoint_commit_failed",
                    before_revision=before_revision,
                    required_evidence={**preflight["required_evidence"], **commit},
                )
            return {
                "artifact_kind": "terminus_snn_language_plasticity_live_application",
                "surface": "snn_language_plasticity_live_application.v1",
                "accepted": True,
                "status": "applied",
                "reason": None,
                "owned_by_marulho": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "generates_text": False,
                "decodes_text": False,
                "trains_runtime_model": False,
                "applies_plasticity": True,
                "mutates_runtime_state": True,
                "returns_trained_weights": False,
                "operator_id": operator_id,
                "checkpoint_transaction": {
                    "pre_update_checkpoint_saved": True,
                    "checkpoint_path": str(checkpoint_file),
                    "committed_checkpoint_path": commit["committed_checkpoint_path"],
                    "staged_committed_checkpoint_path": str(committed_checkpoint_file),
                    "post_update_checkpoint_saved": True,
                    "post_update_checkpoint_restore_verified": True,
                    "current_checkpoint_manifest": commit["current_checkpoint_manifest"],
                    "restore_endpoint_available": True,
                    "restore_verified": checkpoint_verified,
                },
                "application_target": {
                    "target_id": "marulho.snn_language.sparse_transition_weights",
                    "owned_by_marulho": True,
                    "sparse": True,
                    "checkpointed": True,
                    "applied_synapse_count": len(applied_synapses),
                    "total_synapse_count": len(weights),
                },
                "live_application_event": deepcopy(event),
                "applied_synapses": applied_synapses,
                "before": {"state_revision": before_revision},
                "after": {"state_revision": after_revision, **self._runtime_state.mutation_summary()},
            }

    def apply_dense_readout_layout_migration(
        self,
        *,
        dense_readout_resize_transaction_proposal: Mapping[str, Any],
        dense_readout_resize_executor_readiness_audit: Mapping[str, Any],
        expected_state_revision: int,
        operator_id: str,
        confirmation: bool,
        checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        """Persist dense readout layout migration evidence without tensor allocation."""

        with self._lock:
            before_revision = int(self._runtime_state.state_revision)
            proposal = dict(dense_readout_resize_transaction_proposal)
            audit = dict(dense_readout_resize_executor_readiness_audit)
            recipe = (
                proposal.get("transaction_recipe")
                if isinstance(proposal.get("transaction_recipe"), Mapping)
                else {}
            )
            gate = (
                audit.get("promotion_gate")
                if isinstance(audit.get("promotion_gate"), Mapping)
                else {}
            )
            required_audit = (
                gate.get("required_evidence")
                if isinstance(gate.get("required_evidence"), Mapping)
                else {}
            )
            current_shape = self._shape_pair(recipe.get("current_dense_readout_shape"))
            target_shape = self._shape_pair(recipe.get("target_dense_readout_shape"))
            preserved_window = self._shape_pair(recipe.get("preserved_dense_window"))
            target_neurons = int(target_shape[0]) if target_shape else _LANGUAGE_NEURON_COUNT
            zero_fill_cells = int(
                recipe.get("zero_initialized_new_dense_cell_count", 0) or 0
            )
            shape_growth_requested = bool(
                current_shape
                and target_shape
                and target_shape[0] >= current_shape[0]
                and target_shape[1] >= current_shape[1]
                and target_shape != current_shape
            )
            required = {
                "confirmation": bool(confirmation),
                "operator_id_available": bool(str(operator_id or "").strip()),
                "expected_revision_current": int(expected_state_revision)
                == before_revision,
                "transaction_surface_available": proposal.get("surface")
                == "snn_language_dense_readout_resize_transaction_proposal.v1",
                "transaction_owned_by_marulho": bool(proposal.get("owned_by_marulho")),
                "transaction_hash_available": bool(
                    proposal.get("dense_readout_resize_transaction_proposal_hash")
                ),
                "audit_surface_available": audit.get("surface")
                == "snn_language_dense_readout_resize_executor_readiness_audit.v1",
                "audit_owned_by_marulho": bool(audit.get("owned_by_marulho")),
                "layout_state_available": bool(
                    required_audit.get("dense_readout_layout_state_available")
                ),
                "layout_matches_transaction": bool(
                    required_audit.get("dense_readout_layout_matches_transaction")
                ),
                "layout_metadata_not_applied": bool(
                    required_audit.get("dense_readout_layout_metadata_not_applied")
                ),
                "layout_owner_available": bool(
                    required_audit.get("dense_readout_tensor_owner_available")
                ),
                "checkpoint_restore_verified": bool(
                    required_audit.get("transaction_checkpoint_restore_verified")
                ),
                "cuda_relayout_evidence_available": bool(
                    required_audit.get("transaction_cuda_relayout_verified")
                ),
                "shape_invariants_available": bool(
                    required_audit.get("transaction_shape_invariants_available")
                ),
                "shape_growth_requested": shape_growth_requested,
                "preserved_window_matches_current_shape": preserved_window
                == current_shape,
                "zero_fill_region_available": zero_fill_cells > 0,
                "tensor_weight_materialization_absent": not bool(
                    required_audit.get("dense_readout_tensor_weight_owner_available")
                ),
                "no_text_generation": not bool(proposal.get("generates_text"))
                and not bool(audit.get("generates_text")),
                "no_external_checkpoint": not bool(
                    proposal.get("loads_external_checkpoint")
                )
                and not bool(audit.get("loads_external_checkpoint")),
            }
            if not all(required.values()):
                return self._blocked_dense_layout_migration(
                    reason="blocked_missing_dense_readout_layout_migration_evidence",
                    before_revision=before_revision,
                    required_evidence=required,
                )

            checkpoint_state = deepcopy(self._language_plasticity_state())
            before_dirty_state = bool(self._runtime_state.dirty_state)
            checkpoint = self._save_checkpoint(checkpoint_path)
            checkpoint_file = Path(str(checkpoint.get("path") or self._checkpoint_path()))
            checkpoint_verified = self._verify_checkpoint_transaction(
                checkpoint_file,
                checkpoint_state,
                before_revision,
            )
            if not checkpoint_verified:
                return self._blocked_dense_layout_migration(
                    reason="checkpoint_save_missing",
                    before_revision=before_revision,
                    required_evidence={
                        **required,
                        "pre_layout_migration_checkpoint_saved": checkpoint_file.exists(),
                        "pre_layout_migration_checkpoint_restore_verified": checkpoint_verified,
                    },
                )

            state = self._language_plasticity_state()
            layout = state.setdefault("dense_readout_layout", {})
            migration = {
                "applied": True,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "operator_id": str(operator_id or "").strip(),
                "checkpoint_path": str(checkpoint_file),
                "current_dense_readout_shape": current_shape,
                "target_dense_readout_shape": target_shape,
                "preserved_dense_window": preserved_window,
                "zero_initialized_new_dense_cell_count": zero_fill_cells,
                "target_language_neuron_count": target_neurons,
                "transaction_hash": proposal.get(
                    "dense_readout_resize_transaction_proposal_hash"
                ),
                "plan_hash": proposal.get("dense_readout_resize_plan_hash"),
                "materializes_dense_tensor_weights": False,
                "requires_tensor_weight_executor": True,
            }
            layout.update(
                {
                    "surface": _DENSE_READOUT_LAYOUT_SURFACE,
                    "target_language_neuron_count": target_neurons,
                    "target_dense_readout_shape": target_shape,
                    "preserved_dense_window": preserved_window,
                    "zero_initialized_new_dense_cell_count": zero_fill_cells,
                    "requires_cuda_relayout": True,
                    "checkpoint_required_before_resize": True,
                    "layout_migration": deepcopy(migration),
                    "layout_migration_count": int(
                        layout.get("layout_migration_count", 0) or 0
                    )
                    + 1,
                    "dense_resize_applied": False,
                    "dynamic_dense_readout_enabled": False,
                    "migration_status": "layout_migration_applied_tensor_resize_pending",
                }
            )
            recent = list(layout.get("recent_layout_migrations") or [])
            layout["recent_layout_migrations"] = [*recent, deepcopy(migration)][-8:]
            committed_checkpoint_file = self._committed_checkpoint_path(
                checkpoint_file,
                "dense_readout_layout_migration",
            )
            migration["committed_checkpoint_path"] = str(committed_checkpoint_file)
            layout["layout_migration"] = deepcopy(migration)
            layout["recent_layout_migrations"][-1] = deepcopy(migration)
            state["last_checkpoint_path"] = str(committed_checkpoint_file)
            self._runtime_state.mark_mutated()
            commit = self._commit_mutation(
                committed_checkpoint_file=committed_checkpoint_file,
                operation="dense_readout_layout_migration",
                before_state=checkpoint_state,
                before_revision=before_revision,
                before_dirty_state=before_dirty_state,
            )
            if not commit["committed"]:
                return self._blocked_dense_layout_migration(
                    reason="post_layout_migration_checkpoint_commit_failed",
                    before_revision=before_revision,
                    required_evidence={**required, **commit},
                )
            return {
                "artifact_kind": "terminus_snn_language_dense_readout_layout_migration",
                "surface": "snn_language_dense_readout_layout_migration.v1",
                "accepted": True,
                "status": "layout_migration_applied_tensor_resize_pending",
                "owned_by_marulho": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "generates_text": False,
                "decodes_text": False,
                "trains_runtime_model": False,
                "applies_plasticity": False,
                "mutates_runtime_state": True,
                "writes_checkpoint": True,
                "resizes_network": False,
                "materializes_dense_tensor_weights": False,
                "checkpoint_transaction": {
                    "pre_layout_migration_checkpoint_saved": True,
                    "checkpoint_path": str(checkpoint_file),
                    "committed_checkpoint_path": commit["committed_checkpoint_path"],
                    "staged_committed_checkpoint_path": str(committed_checkpoint_file),
                    "post_layout_migration_checkpoint_saved": True,
                    "post_layout_migration_checkpoint_restore_verified": True,
                    "current_checkpoint_manifest": commit["current_checkpoint_manifest"],
                    "restore_verified": checkpoint_verified,
                },
                "dense_readout_layout_migration": deepcopy(migration),
                "before": {"state_revision": before_revision},
                "after": self._runtime_state.mutation_summary(),
            }

    def apply_dense_readout_tensor_materialization(
        self,
        *,
        dense_readout_tensor_materialization_readiness: Mapping[str, Any],
        expected_state_revision: int,
        operator_id: str,
        confirmation: bool,
        checkpoint_path: str | None = None,
        requested_device: str | None = None,
    ) -> dict[str, Any]:
        """Materialize a dense readout tensor from owned SNN language weights."""

        with self._lock:
            before_revision = int(self._runtime_state.state_revision)
            readiness = dict(dense_readout_tensor_materialization_readiness)
            gate = (
                readiness.get("promotion_gate")
                if isinstance(readiness.get("promotion_gate"), Mapping)
                else {}
            )
            required_readiness = (
                gate.get("required_evidence")
                if isinstance(gate.get("required_evidence"), Mapping)
                else {}
            )
            target_shape = self._shape_pair(readiness.get("target_dense_readout_shape"))
            preserved_window = self._shape_pair(readiness.get("preserved_dense_window"))
            zero_fill_cells = int(
                readiness.get("zero_initialized_new_dense_cell_count", 0) or 0
            )
            device = self._resolve_tensor_device(requested_device)
            required = {
                "confirmation": bool(confirmation),
                "operator_id_available": bool(str(operator_id or "").strip()),
                "expected_revision_current": int(expected_state_revision) == before_revision,
                "readiness_surface_available": readiness.get("surface")
                == "snn_language_dense_readout_tensor_materialization_readiness.v1",
                "readiness_owned_by_marulho": bool(readiness.get("owned_by_marulho")),
                "readiness_gate_ready": bool(readiness.get("ready")),
                "readiness_not_executable": not bool(readiness.get("executable")),
                "readiness_does_not_mutate": not bool(readiness.get("mutates_runtime_state")),
                "layout_migration_committed": bool(
                    required_readiness.get("layout_migration_checkpoint_committed")
                ),
                "layout_state_matches_migration": bool(
                    required_readiness.get("layout_state_matches_migration")
                ),
                "dense_resize_not_yet_applied": bool(
                    required_readiness.get("dense_resize_not_yet_applied")
                ),
                "target_shape_available": len(target_shape) == 2,
                "preserved_window_available": len(preserved_window) == 2,
                "zero_fill_region_available": zero_fill_cells > 0,
                "requested_device_available": bool(str(device)),
                "requested_cuda_available_when_requested": (
                    device.type != "cuda" or torch.cuda.is_available()
                ),
                "no_text_generation": not bool(readiness.get("generates_text")),
                "no_external_checkpoint": not bool(readiness.get("loads_external_checkpoint")),
                "no_training": not bool(readiness.get("trains_runtime_model")),
            }
            if not all(required.values()):
                return self._blocked_dense_tensor_materialization(
                    reason="blocked_missing_dense_readout_tensor_materialization_evidence",
                    before_revision=before_revision,
                    required_evidence=required,
                )

            checkpoint_state = deepcopy(self._language_plasticity_state())
            before_dirty_state = bool(self._runtime_state.dirty_state)
            checkpoint = self._save_checkpoint(checkpoint_path)
            checkpoint_file = Path(str(checkpoint.get("path") or self._checkpoint_path()))
            checkpoint_verified = self._verify_checkpoint_transaction(
                checkpoint_file,
                checkpoint_state,
                before_revision,
            )
            if not checkpoint_verified:
                return self._blocked_dense_tensor_materialization(
                    reason="checkpoint_save_missing",
                    before_revision=before_revision,
                    required_evidence={
                        **required,
                        "pre_tensor_materialization_checkpoint_saved": checkpoint_file.exists(),
                        "pre_tensor_materialization_checkpoint_restore_verified": checkpoint_verified,
                    },
                )

            state = self._language_plasticity_state()
            previous_tensor = state.get("dense_readout_weights")
            target = torch.zeros(tuple(target_shape), dtype=torch.float32, device=device)
            preserved_source = "sparse_transition_weights"
            copied_nonzero_count = 0
            if isinstance(previous_tensor, torch.Tensor):
                source = previous_tensor.detach().to(device=device, dtype=torch.float32)
                rows = min(int(preserved_window[0]), int(source.shape[0]), target.shape[0])
                cols = min(int(preserved_window[1]), int(source.shape[1]), target.shape[1])
                if rows > 0 and cols > 0:
                    target[:rows, :cols] = source[:rows, :cols]
                    copied_nonzero_count = int(torch.count_nonzero(target[:rows, :cols]).item())
                preserved_source = "existing_dense_readout_weights"
            else:
                for key, value in dict(state.get("sparse_transition_weights") or {}).items():
                    try:
                        pre_text, post_text = str(key).split(":", maxsplit=1)
                        pre_index = int(pre_text)
                        post_index = int(post_text)
                        weight = float(value)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= pre_index < target.shape[0] and 0 <= post_index < target.shape[1]:
                        target[pre_index, post_index] = weight
                copied_nonzero_count = int(torch.count_nonzero(target).item())

            layout = state.setdefault("dense_readout_layout", {})
            committed_checkpoint_file = self._committed_checkpoint_path(
                checkpoint_file,
                "dense_readout_tensor_materialization",
            )
            materialization = {
                "applied": True,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "operator_id": str(operator_id or "").strip(),
                "checkpoint_path": str(checkpoint_file),
                "committed_checkpoint_path": str(committed_checkpoint_file),
                "requested_device": str(requested_device or ""),
                "actual_device": str(target.device),
                "tensor_is_cuda": bool(target.is_cuda),
                "target_dense_readout_shape": target_shape,
                "preserved_dense_window": preserved_window,
                "zero_initialized_new_dense_cell_count": zero_fill_cells,
                "preserved_source": preserved_source,
                "copied_nonzero_weight_count": copied_nonzero_count,
                "materializes_dense_tensor_weights": True,
                "generates_text": False,
                "trains_runtime_model": False,
            }
            state["dense_readout_weights"] = target
            layout.update(
                {
                    "surface": _DENSE_READOUT_LAYOUT_SURFACE,
                    "target_language_neuron_count": int(target_shape[0]),
                    "target_dense_readout_shape": target_shape,
                    "preserved_dense_window": preserved_window,
                    "zero_initialized_new_dense_cell_count": zero_fill_cells,
                    "requires_cuda_relayout": False,
                    "checkpoint_required_before_resize": False,
                    "tensor_materialization": deepcopy(materialization),
                    "tensor_materialization_count": int(
                        layout.get("tensor_materialization_count", 0) or 0
                    )
                    + 1,
                    "dense_resize_applied": True,
                    "dynamic_dense_readout_enabled": True,
                    "migration_status": "dense_readout_tensor_materialized",
                }
            )
            recent = list(layout.get("recent_tensor_materializations") or [])
            layout["recent_tensor_materializations"] = [
                *recent,
                deepcopy(materialization),
            ][-8:]
            state["last_checkpoint_path"] = str(committed_checkpoint_file)
            self._runtime_state.mark_mutated()
            commit = self._commit_mutation(
                committed_checkpoint_file=committed_checkpoint_file,
                operation="dense_readout_tensor_materialization",
                before_state=checkpoint_state,
                before_revision=before_revision,
                before_dirty_state=before_dirty_state,
            )
            if not commit["committed"]:
                return self._blocked_dense_tensor_materialization(
                    reason="post_tensor_materialization_checkpoint_commit_failed",
                    before_revision=before_revision,
                    required_evidence={**required, **commit},
                )
            return {
                "artifact_kind": "terminus_snn_language_dense_readout_tensor_materialization",
                "surface": "snn_language_dense_readout_tensor_materialization.v1",
                "accepted": True,
                "status": "dense_readout_tensor_materialized",
                "owned_by_marulho": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "generates_text": False,
                "decodes_text": False,
                "trains_runtime_model": False,
                "applies_plasticity": False,
                "mutates_runtime_state": True,
                "writes_checkpoint": True,
                "resizes_network": True,
                "materializes_dense_tensor_weights": True,
                "checkpoint_transaction": {
                    "pre_tensor_materialization_checkpoint_saved": True,
                    "checkpoint_path": str(checkpoint_file),
                    "committed_checkpoint_path": commit["committed_checkpoint_path"],
                    "staged_committed_checkpoint_path": str(committed_checkpoint_file),
                    "post_tensor_materialization_checkpoint_saved": True,
                    "post_tensor_materialization_checkpoint_restore_verified": True,
                    "current_checkpoint_manifest": commit["current_checkpoint_manifest"],
                    "restore_verified": checkpoint_verified,
                },
                "dense_readout_tensor_materialization": deepcopy(materialization),
                "dense_readout_tensor": self._dense_tensor_summary(target),
                "before": {"state_revision": before_revision},
                "after": self._runtime_state.mutation_summary(),
            }

    def apply_autonomous_snn_language_thought_capacity_mutation(
        self,
        *,
        autonomous_snn_language_thought_capacity_mutation_preflight: Mapping[
            str, Any
        ],
        expected_state_revision: int,
        checkpoint_path: str | None = None,
        requested_device: str | None = None,
    ) -> dict[str, Any]:
        """Grow owned SNN language capacity from a ready machine preflight."""

        with self._lock:
            before_revision = int(self._runtime_state.state_revision)
            artifact = dict(
                autonomous_snn_language_thought_capacity_mutation_preflight or {}
            )
            gate = (
                artifact.get("promotion_gate")
                if isinstance(artifact.get("promotion_gate"), Mapping)
                else {}
            )
            required_preflight = (
                gate.get("required_evidence")
                if isinstance(gate.get("required_evidence"), Mapping)
                else {}
            )
            preflight = (
                artifact.get(
                    "autonomous_snn_language_thought_capacity_mutation_preflight"
                )
                if isinstance(
                    artifact.get(
                        "autonomous_snn_language_thought_capacity_mutation_preflight"
                    ),
                    Mapping,
                )
                else {}
            )
            current_shape = self._shape_pair(preflight.get("current_dense_shape"))
            target_shape = self._shape_pair(preflight.get("target_dense_shape"))
            preserved_shape = self._shape_pair(
                preflight.get("preserved_dense_shape")
            )
            current_neurons = int(
                preflight.get("current_neuron_capacity", 0) or 0
            )
            target_neurons = int(
                preflight.get("target_neuron_capacity", 0) or 0
            )
            current_sparse_budget = int(
                preflight.get("current_sparse_synapse_budget", 0) or 0
            )
            target_sparse_budget = int(
                preflight.get("target_sparse_synapse_budget", 0) or 0
            )
            growth_candidates = [
                dict(item)
                for item in list(preflight.get("growth_candidates") or [])
                if isinstance(item, Mapping)
            ]
            device_name = str(
                requested_device or preflight.get("requested_device") or ""
            ).strip()
            try:
                device = self._resolve_tensor_device(device_name or None)
                device_valid = True
            except (RuntimeError, TypeError):
                device = torch.device("cpu")
                device_valid = False
            try:
                declared_growth_candidate_count = int(
                    preflight.get("growth_candidate_count", -1)
                )
            except (TypeError, ValueError):
                declared_growth_candidate_count = -1

            state = self._language_plasticity_state()
            capacity_before = self._language_capacity_state(state)
            existing_tensor = state.get("dense_readout_weights")
            existing_shape = (
                [int(item) for item in list(existing_tensor.shape)]
                if isinstance(existing_tensor, torch.Tensor)
                else current_shape
            )
            preflight_hash = str(artifact.get("preflight_hash") or "")
            preflight_expected_revision = preflight.get(
                "expected_state_revision"
            )
            try:
                normalized_preflight_expected_revision = int(
                    preflight_expected_revision
                )
            except (TypeError, ValueError):
                normalized_preflight_expected_revision = -1
            required = {
                "preflight_surface_available": artifact.get("surface")
                == (
                    "snn_language_autonomous_snn_language_thought_"
                    "capacity_mutation_preflight.v1"
                ),
                "preflight_ready": bool(artifact.get("accepted"))
                and bool(artifact.get("ready")),
                "executor_gate_ready": bool(
                    gate.get(
                        "eligible_for_autonomous_snn_language_thought_"
                        "capacity_mutation_executor"
                    )
                ),
                "operator_approval_not_required": not bool(
                    artifact.get("requires_operator_approval")
                )
                and not bool(preflight.get("operator_approval_required")),
                "expected_revision_current": int(expected_state_revision)
                == before_revision,
                "expected_revision_matches_preflight": (
                    normalized_preflight_expected_revision
                    == int(expected_state_revision)
                ),
                "preflight_hash_available": len(preflight_hash) == 64,
                "design_hash_available": len(
                    str(
                        preflight.get(
                            "thought_capacity_mutation_design_hash"
                        )
                        or ""
                    )
                )
                == 64,
                "structural_event_review_hash_available": len(
                    str(preflight.get("structural_event_review_hash") or "")
                )
                == 64,
                "memory_trace_hash_available": len(
                    str(preflight.get("memory_trace_hash") or "")
                )
                == 64,
                "cuda_relayout_preflight_verified": bool(
                    required_preflight.get("cuda_relayout_verified")
                )
                and bool(preflight.get("cuda_relayout_verified")),
                "checkpoint_preflight_verified": bool(
                    required_preflight.get("checkpoint_saved")
                )
                and bool(required_preflight.get("restore_verified"))
                and bool(preflight.get("checkpoint_saved"))
                and bool(preflight.get("restore_verified")),
                "executor_capability_preflight_verified": bool(
                    required_preflight.get("executor_capability_available")
                )
                and bool(preflight.get("executor_ready")),
                "execution_was_preflight_blocked": not bool(
                    preflight.get("execution_allowed")
                ),
                "current_capacity_matches_runtime": current_neurons
                == int(capacity_before["language_neuron_count"]),
                "current_sparse_budget_matches_runtime": current_sparse_budget
                == int(capacity_before["sparse_edge_budget"]),
                "target_capacity_grows": target_neurons > current_neurons >= 1,
                "target_sparse_budget_grows": target_sparse_budget
                > current_sparse_budget
                >= 1,
                "growth_candidates_available": bool(growth_candidates),
                "growth_candidate_count_matches": (
                    declared_growth_candidate_count == len(growth_candidates)
                ),
                "target_shape_matches_capacity": target_shape
                == [target_neurons, target_neurons],
                "preserved_shape_matches_current": preserved_shape
                == current_shape,
                "existing_tensor_shape_matches_current": existing_shape
                == current_shape,
                "zero_initialized_rows_match_growth": int(
                    preflight.get("zero_initialized_new_rows", -1)
                )
                == target_shape[0] - current_shape[0]
                if target_shape and current_shape
                else False,
                "zero_initialized_cols_match_growth": int(
                    preflight.get("zero_initialized_new_cols", -1)
                )
                == target_shape[1] - current_shape[1]
                if target_shape and current_shape
                else False,
                "device_available": device_valid and bool(device_name),
                "requested_cuda_available_when_requested": device.type != "cuda"
                or torch.cuda.is_available(),
                "no_external_checkpoint": not bool(
                    artifact.get("loads_external_checkpoint")
                ),
                "no_replay": not bool(artifact.get("runs_replay")),
                "no_training": not bool(artifact.get("trains_runtime_model")),
            }
            if not all(required.values()):
                return self._blocked_thought_capacity_mutation(
                    reason="blocked_missing_thought_capacity_mutation_evidence",
                    before_revision=before_revision,
                    required_evidence=required,
                )

            checkpoint_state = deepcopy(state)
            before_dirty_state = bool(self._runtime_state.dirty_state)
            checkpoint = self._save_checkpoint(checkpoint_path)
            checkpoint_file = Path(
                str(checkpoint.get("path") or self._checkpoint_path())
            )
            checkpoint_verified = self._verify_checkpoint_transaction(
                checkpoint_file,
                checkpoint_state,
                before_revision,
            )
            if not checkpoint_verified:
                return self._blocked_thought_capacity_mutation(
                    reason="checkpoint_save_missing",
                    before_revision=before_revision,
                    required_evidence={
                        **required,
                        "pre_capacity_mutation_checkpoint_saved": (
                            checkpoint_file.exists()
                        ),
                        "pre_capacity_mutation_checkpoint_restore_verified": (
                            checkpoint_verified
                        ),
                    },
                )

            target = torch.zeros(
                tuple(target_shape),
                dtype=(
                    existing_tensor.dtype
                    if isinstance(existing_tensor, torch.Tensor)
                    else torch.float32
                ),
                device=device,
            )
            preserved_source = "sparse_transition_weights"
            if isinstance(existing_tensor, torch.Tensor):
                source = existing_tensor.detach().to(
                    device=device,
                    dtype=target.dtype,
                )
                target[: current_shape[0], : current_shape[1]] = source[
                    : current_shape[0], : current_shape[1]
                ]
                preserved_source = "existing_dense_readout_weights"
            else:
                for key, value in dict(
                    state.get("sparse_transition_weights") or {}
                ).items():
                    try:
                        pre_text, post_text = str(key).split(":", maxsplit=1)
                        pre_index = int(pre_text)
                        post_index = int(post_text)
                        weight = float(value)
                    except (TypeError, ValueError):
                        continue
                    if (
                        0 <= pre_index < current_shape[0]
                        and 0 <= post_index < current_shape[1]
                    ):
                        target[pre_index, post_index] = weight

            preserved_nonzero_count = int(
                torch.count_nonzero(
                    target[: current_shape[0], : current_shape[1]]
                ).item()
            )
            new_region_nonzero_count = int(
                torch.count_nonzero(target[current_shape[0] :, :]).item()
                + torch.count_nonzero(
                    target[: current_shape[0], current_shape[1] :]
                ).item()
            )
            completed_at = datetime.now(timezone.utc).isoformat()
            committed_checkpoint_file = self._committed_checkpoint_path(
                checkpoint_file,
                "thought_capacity_mutation",
            )
            event = {
                "completed_at": completed_at,
                "before_state_revision": before_revision,
                "after_state_revision": before_revision + 1,
                "preflight_hash": preflight_hash,
                "thought_capacity_mutation_design_hash": preflight.get(
                    "thought_capacity_mutation_design_hash"
                ),
                "structural_event_review_hash": preflight.get(
                    "structural_event_review_hash"
                ),
                "memory_trace_hash": preflight.get("memory_trace_hash"),
                "checkpoint_path": str(checkpoint_file),
                "committed_checkpoint_path": str(committed_checkpoint_file),
                "requested_device": device_name,
                "actual_device": str(target.device),
                "tensor_is_cuda": bool(target.is_cuda),
                "current_neuron_capacity": current_neurons,
                "target_neuron_capacity": target_neurons,
                "added_neuron_capacity": target_neurons - current_neurons,
                "current_sparse_synapse_budget": current_sparse_budget,
                "target_sparse_synapse_budget": target_sparse_budget,
                "added_sparse_synapse_budget": (
                    target_sparse_budget - current_sparse_budget
                ),
                "current_dense_shape": current_shape,
                "target_dense_shape": target_shape,
                "preserved_dense_shape": preserved_shape,
                "preserved_source": preserved_source,
                "preserved_nonzero_weight_count": preserved_nonzero_count,
                "new_region_nonzero_count": new_region_nonzero_count,
                "new_region_zero_initialized": new_region_nonzero_count == 0,
                "growth_candidates": deepcopy(
                    growth_candidates
                ),
                "prune_candidates": deepcopy(
                    list(preflight.get("prune_candidates") or [])
                ),
                "replay_executed": False,
                "training_executed": False,
                "plasticity_applied": False,
            }
            event["capacity_mutation_event_hash"] = self._sha256_json(event)

            state["dense_readout_weights"] = target
            capacity = state.setdefault("language_capacity", {})
            capacity.update(
                {
                    "surface": _LANGUAGE_CAPACITY_SURFACE,
                    "owned_by_marulho": True,
                    "external_dependency": False,
                    "language_neuron_count": target_neurons,
                    "sparse_edge_budget": target_sparse_budget,
                    "outgoing_fanout_budget": int(
                        capacity_before["outgoing_fanout_budget"]
                    ),
                    "dynamic_capacity_enabled": True,
                    "capacity_expansion_count": int(
                        capacity_before["capacity_expansion_count"]
                    )
                    + 1,
                    "resizes_network": True,
                    "adds_neurons": True,
                    "adds_layers": False,
                    "writes_checkpoint": True,
                    "last_capacity_mutation": deepcopy(event),
                }
            )
            layout = state.setdefault("dense_readout_layout", {})
            layout.update(
                {
                    "surface": _DENSE_READOUT_LAYOUT_SURFACE,
                    "target_language_neuron_count": target_neurons,
                    "target_dense_readout_shape": target_shape,
                    "preserved_dense_window": preserved_shape,
                    "zero_initialized_new_dense_cell_count": (
                        target_shape[0] * target_shape[1]
                        - current_shape[0] * current_shape[1]
                    ),
                    "requires_cuda_relayout": False,
                    "checkpoint_required_before_resize": False,
                    "dense_resize_applied": True,
                    "dynamic_dense_readout_enabled": True,
                    "migration_status": "thought_capacity_mutation_applied",
                    "thought_capacity_mutation": deepcopy(event),
                }
            )
            mutation_state = state.setdefault("thought_capacity_mutation", {})
            mutation_state["mutation_count"] = int(
                mutation_state.get("mutation_count", 0) or 0
            ) + 1
            mutation_state["last_mutation"] = deepcopy(event)
            mutation_state["recent_events"] = [
                *list(mutation_state.get("recent_events") or []),
                deepcopy(event),
            ][-8:]
            state["last_checkpoint_path"] = str(committed_checkpoint_file)
            self._runtime_state.mark_mutated()
            commit = self._commit_mutation(
                committed_checkpoint_file=committed_checkpoint_file,
                operation="thought_capacity_mutation",
                before_state=checkpoint_state,
                before_revision=before_revision,
                before_dirty_state=before_dirty_state,
            )
            if not commit["committed"]:
                return self._blocked_thought_capacity_mutation(
                    reason="post_capacity_mutation_checkpoint_commit_failed",
                    before_revision=before_revision,
                    required_evidence={**required, **commit},
                )
            return {
                "artifact_kind": (
                    "terminus_snn_language_autonomous_snn_language_thought_"
                    "capacity_mutation_executor"
                ),
                "surface": (
                    "snn_language_autonomous_snn_language_thought_"
                    "capacity_mutation_executor.v1"
                ),
                "accepted": True,
                "ready": True,
                "status": "thought_capacity_mutation_applied",
                "requires_operator_approval": False,
                "owned_by_marulho": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "advisory": False,
                "executable": True,
                "records_ledger_event": True,
                "generates_text": False,
                "decodes_text": False,
                "runs_replay": False,
                "trains_runtime_model": False,
                "applies_plasticity": False,
                "mutates_runtime_state": True,
                "writes_checkpoint": True,
                "resizes_network": True,
                "adds_neurons": True,
                "adds_synapses": False,
                "prunes_network": False,
                "checkpoint_transaction": {
                    "pre_capacity_mutation_checkpoint_saved": True,
                    "checkpoint_path": str(checkpoint_file),
                    "committed_checkpoint_path": commit[
                        "committed_checkpoint_path"
                    ],
                    "staged_committed_checkpoint_path": str(
                        committed_checkpoint_file
                    ),
                    "post_capacity_mutation_checkpoint_saved": True,
                    "post_capacity_mutation_checkpoint_restore_verified": True,
                    "current_checkpoint_manifest": commit[
                        "current_checkpoint_manifest"
                    ],
                    "restore_verified": checkpoint_verified,
                },
                "autonomous_snn_language_thought_capacity_mutation_event": deepcopy(
                    event
                ),
                "dense_readout_tensor": self._dense_tensor_summary(target),
                "before": {
                    "state_revision": before_revision,
                    "language_capacity": deepcopy(capacity_before),
                },
                "after": {
                    "state_revision": int(self._runtime_state.state_revision),
                    "language_capacity": deepcopy(
                        self._language_capacity_state(state)
                    ),
                    **self._runtime_state.mutation_summary(),
                },
                "promotion_gate": {
                    "status": (
                        "ready_for_autonomous_snn_language_thought_"
                        "capacity_mutation_event_review"
                    ),
                    "eligible_for_autonomous_snn_language_thought_"
                    "capacity_mutation_event_review": True,
                    "eligible_for_language_generation": False,
                    "eligible_for_replay_memory": False,
                    "eligible_for_plasticity_application": False,
                    "eligible_for_fact_promotion": False,
                    "eligible_for_action": False,
                    "eligible_for_cognition_substrate": False,
                },
            }

    def apply_autonomous_snn_language_thought_newborn_neuron_integration(
        self,
        *,
        autonomous_snn_language_thought_newborn_neuron_integration_preflight: Mapping[
            str, Any
        ],
        expected_state_revision: int,
        checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        """Connect mature sources to newborn slots under a bounded critical period."""

        with self._lock:
            before_revision = int(self._runtime_state.state_revision)
            artifact = dict(
                autonomous_snn_language_thought_newborn_neuron_integration_preflight
                or {}
            )
            preflight = (
                dict(
                    artifact.get(
                        "autonomous_snn_language_thought_newborn_neuron_"
                        "integration_preflight"
                    )
                )
                if isinstance(
                    artifact.get(
                        "autonomous_snn_language_thought_newborn_neuron_"
                        "integration_preflight"
                    ),
                    Mapping,
                )
                else {}
            )
            gate = (
                artifact.get("promotion_gate")
                if isinstance(artifact.get("promotion_gate"), Mapping)
                else {}
            )
            required_preflight = (
                dict(gate.get("required_evidence"))
                if isinstance(gate.get("required_evidence"), Mapping)
                else {}
            )
            candidates = [
                dict(item)
                for item in list(
                    preflight.get("resolved_integration_candidates") or []
                )
                if isinstance(item, Mapping)
            ]
            state = self._language_plasticity_state()
            capacity = self._language_capacity_state(state)
            neuron_count = int(capacity["language_neuron_count"])
            sparse_edge_budget = int(capacity["sparse_edge_budget"])
            outgoing_fanout_budget = int(
                capacity["outgoing_fanout_budget"]
            )
            weights = dict(state.get("sparse_transition_weights") or {})
            dense_tensor = state.get("dense_readout_weights")
            seed_candidates: list[dict[str, Any]] = []
            for candidate in candidates:
                max_initial_weight = float(
                    candidate.get("max_initial_weight", 0.0) or 0.0
                )
                seed_weight = min(
                    max_initial_weight,
                    max(0.001, max_initial_weight * 0.25),
                )
                seed_candidates.append(
                    {
                        **candidate,
                        "pre_index": int(
                            candidate.get("source_neuron_index", -1)
                        ),
                        "post_index": int(
                            candidate.get("target_neuron_index", -1)
                        ),
                        "seed_weight": seed_weight,
                    }
                )
            topology = self._validate_structural_candidates(
                candidates=seed_candidates,
                weights=weights,
                candidate_weight=lambda item: float(
                    item.get("seed_weight", 0.0) or 0.0
                ),
                row_mass_limit=_MAX_OUTGOING_ROW_MASS,
                language_neuron_count=neuron_count,
                sparse_edge_budget=sparse_edge_budget,
                outgoing_fanout_budget=outgoing_fanout_budget,
            )
            preflight_expected_revision = int(
                preflight.get("expected_state_revision", -1)
                if preflight.get("expected_state_revision") is not None
                else -1
            )
            preflight_hash = str(artifact.get("preflight_hash") or "")
            required = {
                "preflight_surface_available": artifact.get("surface")
                == (
                    "snn_language_autonomous_snn_language_thought_"
                    "newborn_neuron_integration_preflight.v1"
                ),
                "preflight_ready": bool(artifact.get("accepted"))
                and bool(artifact.get("ready"))
                and bool(
                    gate.get(
                        "eligible_for_autonomous_snn_language_thought_"
                        "newborn_neuron_integration_executor"
                    )
                ),
                "operator_approval_not_required": not bool(
                    artifact.get("requires_operator_approval")
                )
                and not bool(preflight.get("operator_approval_required")),
                "expected_revision_current": int(expected_state_revision)
                == before_revision,
                "expected_revision_matches_preflight": (
                    preflight_expected_revision == int(expected_state_revision)
                ),
                "preflight_hash_available": len(preflight_hash) == 64,
                "design_hash_available": len(
                    str(
                        preflight.get(
                            "thought_newborn_neuron_integration_design_hash"
                        )
                        or ""
                    )
                )
                == 64,
                "source_resolution_preflight_verified": bool(
                    required_preflight.get("all_candidate_sources_resolved")
                )
                and bool(preflight.get("source_indices_resolved")),
                "checkpoint_preflight_verified": bool(
                    required_preflight.get("checkpoint_saved")
                )
                and bool(
                    required_preflight.get("checkpoint_restore_verified")
                ),
                "executor_capability_preflight_verified": bool(
                    required_preflight.get("executor_capability_available")
                ),
                "resolved_candidates_available": bool(seed_candidates)
                and len(seed_candidates)
                == int(preflight.get("resolved_candidate_count", 0) or 0),
                "candidate_resolution_hashes_valid": all(
                    len(str(item.get("source_resolution_hash") or "")) == 64
                    for item in seed_candidates
                ),
                "candidate_edges_absent": all(
                    str(item.get("synapse") or "") not in weights
                    for item in seed_candidates
                ),
                "candidate_sources_mature": all(
                    0
                    <= int(item.get("pre_index", -1))
                    < int(preflight.get("current_neuron_capacity", 0) or 0)
                    for item in seed_candidates
                ),
                "candidate_targets_newborn": all(
                    int(item.get("post_index", -1))
                    in list(preflight.get("newborn_neuron_indices") or [])
                    for item in seed_candidates
                ),
                "dense_tensor_available": isinstance(
                    dense_tensor, torch.Tensor
                ),
                "dense_tensor_shape_matches_capacity": isinstance(
                    dense_tensor, torch.Tensor
                )
                and list(dense_tensor.shape) == [neuron_count, neuron_count],
                "no_external_checkpoint": not bool(
                    artifact.get("loads_external_checkpoint")
                ),
                "no_replay": not bool(artifact.get("runs_replay")),
                "no_training": not bool(
                    artifact.get("trains_runtime_model")
                ),
                **topology,
            }
            if not all(required.values()):
                return self._blocked_thought_newborn_neuron_integration(
                    reason=(
                        "blocked_missing_thought_newborn_neuron_"
                        "integration_evidence"
                    ),
                    before_revision=before_revision,
                    required_evidence=required,
                )

            checkpoint_state = deepcopy(state)
            before_dirty_state = bool(self._runtime_state.dirty_state)
            checkpoint = self._save_checkpoint(checkpoint_path)
            checkpoint_file = Path(
                str(checkpoint.get("path") or self._checkpoint_path())
            )
            checkpoint_verified = self._verify_checkpoint_transaction(
                checkpoint_file,
                checkpoint_state,
                before_revision,
            )
            if not checkpoint_verified:
                return self._blocked_thought_newborn_neuron_integration(
                    reason="checkpoint_save_missing",
                    before_revision=before_revision,
                    required_evidence={
                        **required,
                        "pre_integration_checkpoint_saved": (
                            checkpoint_file.exists()
                        ),
                        "pre_integration_checkpoint_restore_verified": (
                            checkpoint_verified
                        ),
                    },
                )

            integrated_tensor = dense_tensor.detach().clone()
            applied_synapses: list[dict[str, Any]] = []
            provenance_by_key = state.setdefault(
                "synapse_provenance_by_key", {}
            )
            live_weights = state.setdefault("sparse_transition_weights", {})
            completed_at = datetime.now(timezone.utc).isoformat()
            for candidate in seed_candidates:
                source_index = int(candidate["pre_index"])
                target_index = int(candidate["post_index"])
                synapse_key = f"{source_index}:{target_index}"
                seed_weight = float(candidate["seed_weight"])
                integrated_tensor[source_index, target_index] = seed_weight
                live_weights[synapse_key] = seed_weight
                applied = {
                    "synapse": synapse_key,
                    "source_neuron_index": source_index,
                    "target_neuron_index": target_index,
                    "seed_weight": seed_weight,
                    "max_initial_weight": float(
                        candidate.get("max_initial_weight", 0.0) or 0.0
                    ),
                    "coactivation_event_count": int(
                        candidate.get("coactivation_event_count", 0) or 0
                    ),
                    "source_firing_rate_hz": float(
                        candidate.get("source_firing_rate_hz", 0.0) or 0.0
                    ),
                    "target_firing_rate_hz": float(
                        candidate.get("target_firing_rate_hz", 0.0) or 0.0
                    ),
                    "critical_period_cycles": int(
                        candidate.get("critical_period_cycles", 0) or 0
                    ),
                    "inactivity_prune_cycles": int(
                        candidate.get("inactivity_prune_cycles", 0) or 0
                    ),
                    "max_seed_synapses": int(
                        candidate.get("max_seed_synapses", 0) or 0
                    ),
                    "integration_candidate_id": str(
                        candidate.get("integration_candidate_id") or ""
                    ),
                    "integration_candidate_hash": str(
                        candidate.get("integration_candidate_hash") or ""
                    ),
                    "source_candidate_hash": str(
                        candidate.get("source_candidate_hash") or ""
                    ),
                    "source_resolution_hash": str(
                        candidate.get("source_resolution_hash") or ""
                    ),
                    "active_neuron_hash": str(
                        candidate.get("active_neuron_hash") or ""
                    ),
                    "spike_projection_hash": str(
                        candidate.get("spike_projection_hash") or ""
                    ),
                    "membrane_state_hash": str(
                        candidate.get("membrane_state_hash") or ""
                    ),
                    "actual_device": str(integrated_tensor.device),
                    "tensor_is_cuda": bool(integrated_tensor.is_cuda),
                    "connection_applied": True,
                    "weight_applied": True,
                    "critical_period_started": True,
                }
                applied["newborn_integration_synapse_hash"] = self._sha256_json(
                    applied
                )
                applied_synapses.append(applied)
                provenance_by_key[synapse_key] = {
                    "provenance_type": "newborn_neuron_integration",
                    "preflight_hash": preflight_hash,
                    **deepcopy(applied),
                }
            state["dense_readout_weights"] = integrated_tensor
            committed_checkpoint_file = self._committed_checkpoint_path(
                checkpoint_file,
                "thought_newborn_neuron_integration",
            )
            event = {
                "completed_at": completed_at,
                "before_state_revision": before_revision,
                "after_state_revision": before_revision + 1,
                "preflight_hash": preflight_hash,
                "thought_newborn_neuron_integration_design_hash": preflight.get(
                    "thought_newborn_neuron_integration_design_hash"
                ),
                "capacity_mutation_event_hash": preflight.get(
                    "capacity_mutation_event_hash"
                ),
                "observation_window_id": preflight.get(
                    "observation_window_id"
                ),
                "observation_window_hash": preflight.get(
                    "observation_window_hash"
                ),
                "checkpoint_path": str(checkpoint_file),
                "committed_checkpoint_path": str(
                    committed_checkpoint_file
                ),
                "actual_device": str(integrated_tensor.device),
                "tensor_is_cuda": bool(integrated_tensor.is_cuda),
                "current_neuron_capacity": int(
                    preflight.get("current_neuron_capacity", 0) or 0
                ),
                "target_neuron_capacity": int(
                    preflight.get("target_neuron_capacity", 0) or 0
                ),
                "newborn_neuron_indices": deepcopy(
                    list(preflight.get("newborn_neuron_indices") or [])
                ),
                "integrated_synapse_count": len(applied_synapses),
                "integrated_synapses": deepcopy(applied_synapses),
                "critical_period_started": True,
                "replay_executed": False,
                "training_executed": False,
                "plasticity_applied": True,
            }
            event["newborn_neuron_integration_event_hash"] = self._sha256_json(
                event
            )
            integration_state = state.setdefault(
                "thought_newborn_neuron_integration", {}
            )
            integration_state["integration_count"] = int(
                integration_state.get("integration_count", 0) or 0
            ) + 1
            integration_state["last_integration"] = deepcopy(event)
            integration_state["recent_events"] = [
                *list(integration_state.get("recent_events") or []),
                deepcopy(event),
            ][-8:]
            state["last_checkpoint_path"] = str(committed_checkpoint_file)
            self._runtime_state.mark_mutated()
            commit = self._commit_mutation(
                committed_checkpoint_file=committed_checkpoint_file,
                operation="thought_newborn_neuron_integration",
                before_state=checkpoint_state,
                before_revision=before_revision,
                before_dirty_state=before_dirty_state,
            )
            if not commit["committed"]:
                return self._blocked_thought_newborn_neuron_integration(
                    reason=(
                        "post_newborn_neuron_integration_checkpoint_"
                        "commit_failed"
                    ),
                    before_revision=before_revision,
                    required_evidence={**required, **commit},
                )
            return {
                "artifact_kind": (
                    "terminus_snn_language_autonomous_snn_language_thought_"
                    "newborn_neuron_integration_executor"
                ),
                "surface": (
                    "snn_language_autonomous_snn_language_thought_"
                    "newborn_neuron_integration_executor.v1"
                ),
                "accepted": True,
                "ready": True,
                "status": "thought_newborn_neuron_integration_applied",
                "requires_operator_approval": False,
                "owned_by_marulho": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "advisory": False,
                "executable": True,
                "records_ledger_event": True,
                "generates_text": False,
                "decodes_text": False,
                "runs_replay": False,
                "trains_runtime_model": False,
                "applies_plasticity": True,
                "mutates_runtime_state": True,
                "writes_checkpoint": True,
                "resizes_network": False,
                "adds_neurons": False,
                "adds_synapses": True,
                "prunes_network": False,
                "checkpoint_transaction": {
                    "pre_integration_checkpoint_saved": True,
                    "checkpoint_path": str(checkpoint_file),
                    "committed_checkpoint_path": commit[
                        "committed_checkpoint_path"
                    ],
                    "staged_committed_checkpoint_path": str(
                        committed_checkpoint_file
                    ),
                    "post_integration_checkpoint_saved": True,
                    "post_integration_checkpoint_restore_verified": True,
                    "current_checkpoint_manifest": commit[
                        "current_checkpoint_manifest"
                    ],
                    "restore_verified": checkpoint_verified,
                },
                "autonomous_snn_language_thought_newborn_neuron_"
                "integration_event": deepcopy(event),
                "dense_readout_tensor": self._dense_tensor_summary(
                    integrated_tensor
                ),
                "newborn_integration_dense_samples": (
                    self._newborn_integration_dense_samples(
                        integrated_tensor,
                        event,
                    )
                ),
                "before": {"state_revision": before_revision},
                "after": {
                    "state_revision": int(self._runtime_state.state_revision),
                    **self._runtime_state.mutation_summary(),
                },
                "promotion_gate": {
                    "status": (
                        "ready_for_autonomous_snn_language_thought_"
                        "newborn_neuron_integration_event_review"
                    ),
                    "eligible_for_autonomous_snn_language_thought_"
                    "newborn_neuron_integration_event_review": True,
                    "eligible_for_language_generation": False,
                    "eligible_for_replay_memory": False,
                    "eligible_for_plasticity_application": False,
                    "eligible_for_fact_promotion": False,
                    "eligible_for_action": False,
                    "eligible_for_cognition_substrate": False,
                },
            }

    def apply_autonomous_snn_language_thought_newborn_neuron_critical_period_learning(
        self,
        *,
        autonomous_snn_language_thought_newborn_neuron_critical_period_learning_preflight: Mapping[
            str, Any
        ],
        expected_state_revision: int,
        checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        """Apply one checkpointed local learning cycle to newborn synapses."""

        with self._lock:
            before_revision = int(self._runtime_state.state_revision)
            artifact = dict(
                autonomous_snn_language_thought_newborn_neuron_critical_period_learning_preflight
                or {}
            )
            preflight = (
                dict(
                    artifact.get(
                        "autonomous_snn_language_thought_newborn_neuron_"
                        "critical_period_learning_preflight"
                    )
                )
                if isinstance(
                    artifact.get(
                        "autonomous_snn_language_thought_newborn_neuron_"
                        "critical_period_learning_preflight"
                    ),
                    Mapping,
                )
                else {}
            )
            gate = (
                dict(artifact.get("promotion_gate"))
                if isinstance(artifact.get("promotion_gate"), Mapping)
                else {}
            )
            cycles = [
                dict(item)
                for item in list(
                    preflight.get("resolved_learning_cycles") or []
                )
                if isinstance(item, Mapping)
            ]
            state = self._language_plasticity_state()
            weights = dict(state.get("sparse_transition_weights") or {})
            dense_tensor = state.get("dense_readout_weights")
            provenance = dict(state.get("synapse_provenance_by_key") or {})
            developmental_state = (
                dict(
                    state.get(
                        "thought_newborn_neuron_critical_period_learning"
                    )
                )
                if isinstance(
                    state.get(
                        "thought_newborn_neuron_critical_period_learning"
                    ),
                    Mapping,
                )
                else {}
            )
            developmental_by_synapse = dict(
                developmental_state.get("by_synapse") or {}
            )
            cycle_checks: list[dict[str, Any]] = []
            for cycle in cycles:
                synapse = str(cycle.get("synapse") or "")
                cycle_hash = str(
                    cycle.get("critical_period_learning_cycle_hash") or ""
                )
                recomputed_cycle_hash = self._sha256_json(
                    {
                        key: value
                        for key, value in cycle.items()
                        if key != "critical_period_learning_cycle_hash"
                    }
                )
                current_weight = weights.get(synapse)
                source_index = int(
                    cycle.get("source_neuron_index", -1)
                    if cycle.get("source_neuron_index") is not None
                    else -1
                )
                target_index = int(
                    cycle.get("target_neuron_index", -1)
                    if cycle.get("target_neuron_index") is not None
                    else -1
                )
                previous = (
                    dict(developmental_by_synapse.get(synapse))
                    if isinstance(
                        developmental_by_synapse.get(synapse), Mapping
                    )
                    else {}
                )
                expected_cycle_index = (
                    int(previous.get("critical_period_age_cycles", 0) or 0)
                    + 1
                )
                proposed_weight = float(
                    cycle.get("proposed_weight", 0.0) or 0.0
                )
                proposed_delta = float(
                    cycle.get("proposed_weight_delta", 0.0) or 0.0
                )
                checks = {
                    "canonical_synapse": synapse
                    == f"{source_index}:{target_index}",
                    "cycle_hash_available": len(cycle_hash) == 64,
                    "cycle_hash_recomputed_match": bool(cycle_hash)
                    and cycle_hash == recomputed_cycle_hash,
                    "activity_hash_available": len(
                        str(cycle.get("candidate_activity_hash") or "")
                    )
                    == 64,
                    "runtime_weight_matches_preflight": isinstance(
                        current_weight, (int, float)
                    )
                    and math.isclose(
                        float(current_weight),
                        float(cycle.get("current_weight", 0.0) or 0.0),
                        rel_tol=1e-7,
                        abs_tol=1e-9,
                    ),
                    "dense_tensor_available": isinstance(
                        dense_tensor, torch.Tensor
                    ),
                    "indices_in_dense_tensor": isinstance(
                        dense_tensor, torch.Tensor
                    )
                    and 0 <= source_index < int(dense_tensor.shape[0])
                    and 0 <= target_index < int(dense_tensor.shape[1]),
                    "provenance_is_newborn_integration": isinstance(
                        provenance.get(synapse), Mapping
                    )
                    and provenance[synapse].get("provenance_type")
                    == "newborn_neuron_integration",
                    "cycle_index_is_next": int(
                        cycle.get("cycle_index", 0) or 0
                    )
                    == expected_cycle_index,
                    "critical_period_open": int(
                        cycle.get(
                            "critical_period_cycles_remaining", 0
                        )
                        or 0
                    )
                    > 0
                    and str(
                        previous.get(
                            "current_maturation_state",
                            cycle.get(
                                "current_maturation_state",
                                "critical_period",
                            ),
                        )
                    )
                    == "critical_period",
                    "delta_bounded": abs(proposed_delta)
                    <= float(
                        cycle.get("max_learning_rate", 0.0) or 0.0
                    )
                    + 1e-12,
                    "proposed_weight_bounded": float(
                        cycle.get("min_weight", 0.0) or 0.0
                    )
                    <= proposed_weight
                    <= float(cycle.get("max_weight", 0.0) or 0.0),
                    "proposed_weight_matches_delta": isinstance(
                        current_weight, (int, float)
                    )
                    and math.isclose(
                        proposed_weight,
                        max(
                            float(
                                cycle.get("min_weight", 0.0) or 0.0
                            ),
                            min(
                                float(
                                    cycle.get("max_weight", 0.0) or 0.0
                                ),
                                float(current_weight) + proposed_delta,
                            ),
                        ),
                        rel_tol=1e-7,
                        abs_tol=1e-9,
                    ),
                    "learning_not_preapplied": not bool(
                        cycle.get("weight_update_applied")
                    )
                    and not bool(
                        cycle.get("critical_period_age_advanced")
                    ),
                    "maturation_not_predecided": not bool(
                        cycle.get("maturation_decided")
                    )
                    and not bool(cycle.get("pruning_applied")),
                }
                cycle_checks.append(
                    {
                        "synapse": synapse,
                        "checks": checks,
                        "verified": all(checks.values()),
                    }
                )
            preflight_expected_revision = int(
                preflight.get("expected_state_revision", -1)
                if preflight.get("expected_state_revision") is not None
                else -1
            )
            required = {
                "preflight_surface_available": artifact.get("surface")
                == (
                    "snn_language_autonomous_snn_language_thought_"
                    "newborn_neuron_critical_period_learning_preflight.v1"
                ),
                "preflight_ready": bool(artifact.get("accepted"))
                and bool(artifact.get("ready"))
                and bool(
                    gate.get(
                        "eligible_for_autonomous_snn_language_thought_"
                        "newborn_neuron_critical_period_learning_executor"
                    )
                ),
                "operator_approval_not_required": not bool(
                    artifact.get("requires_operator_approval")
                )
                and not bool(preflight.get("operator_approval_required")),
                "expected_revision_current": int(expected_state_revision)
                == before_revision,
                "expected_revision_matches_preflight": (
                    preflight_expected_revision == int(expected_state_revision)
                ),
                "preflight_hash_available": len(
                    str(artifact.get("preflight_hash") or "")
                )
                == 64,
                "design_hash_available": len(
                    str(
                        preflight.get(
                            "thought_newborn_neuron_critical_period_"
                            "learning_design_hash"
                        )
                        or ""
                    )
                )
                == 64,
                "resolved_cycles_available": bool(cycles)
                and len(cycles)
                == int(preflight.get("resolved_cycle_count", 0) or 0),
                "all_learning_cycles_verified": bool(cycle_checks)
                and all(item["verified"] for item in cycle_checks),
                "checkpoint_path_available": bool(
                    str(preflight.get("checkpoint_path") or "").strip()
                    or checkpoint_path
                ),
                "no_external_checkpoint": not bool(
                    artifact.get("loads_external_checkpoint")
                ),
                "no_replay": not bool(artifact.get("runs_replay")),
                "no_training": not bool(
                    artifact.get("trains_runtime_model")
                ),
            }
            if not all(required.values()):
                return self._blocked_thought_newborn_neuron_critical_period_learning(
                    reason=(
                        "blocked_missing_thought_newborn_neuron_"
                        "critical_period_learning_evidence"
                    ),
                    before_revision=before_revision,
                    required_evidence=required,
                )

            checkpoint_state = deepcopy(state)
            before_dirty_state = bool(self._runtime_state.dirty_state)
            checkpoint = self._save_checkpoint(
                checkpoint_path
                or str(preflight.get("checkpoint_path") or "")
                or None
            )
            checkpoint_file = Path(
                str(checkpoint.get("path") or self._checkpoint_path())
            )
            checkpoint_verified = self._verify_checkpoint_transaction(
                checkpoint_file,
                checkpoint_state,
                before_revision,
            )
            if not checkpoint_verified:
                return self._blocked_thought_newborn_neuron_critical_period_learning(
                    reason="checkpoint_save_missing",
                    before_revision=before_revision,
                    required_evidence={
                        **required,
                        "pre_learning_checkpoint_saved": (
                            checkpoint_file.exists()
                        ),
                        "pre_learning_checkpoint_restore_verified": (
                            checkpoint_verified
                        ),
                    },
                )

            learned_tensor = dense_tensor.detach().clone()
            live_weights = state.setdefault("sparse_transition_weights", {})
            live_provenance = state.setdefault(
                "synapse_provenance_by_key", {}
            )
            learning_state = state.setdefault(
                "thought_newborn_neuron_critical_period_learning", {}
            )
            live_by_synapse = learning_state.setdefault("by_synapse", {})
            applied_cycles: list[dict[str, Any]] = []
            completed_at = datetime.now(timezone.utc).isoformat()
            for cycle in cycles:
                synapse = str(cycle["synapse"])
                source_index = int(cycle["source_neuron_index"])
                target_index = int(cycle["target_neuron_index"])
                proposed_weight = float(cycle["proposed_weight"])
                previous = (
                    dict(live_by_synapse.get(synapse))
                    if isinstance(live_by_synapse.get(synapse), Mapping)
                    else {}
                )
                cycle_index = int(cycle["cycle_index"])
                active_cycle_count = int(
                    previous.get("active_cycle_count", 0) or 0
                ) + int(bool(cycle.get("active_cycle")))
                inactive_cycle_count = (
                    0
                    if bool(cycle.get("active_cycle"))
                    else int(
                        previous.get("inactive_cycle_count", 0) or 0
                    )
                    + 1
                )
                total_cycles = int(
                    cycle.get("critical_period_cycles", 0) or 0
                )
                cycles_remaining = max(0, total_cycles - cycle_index)
                minimum_survival_active_cycles = int(
                    cycle.get("minimum_survival_active_cycles", 0) or 0
                )
                maturation_state = "critical_period"
                maturation_decided = False
                if cycles_remaining == 0:
                    maturation_decided = True
                    maturation_state = (
                        "mature"
                        if active_cycle_count
                        >= minimum_survival_active_cycles
                        and proposed_weight > 0.0
                        else "prune_eligible"
                    )
                learned_tensor[source_index, target_index] = proposed_weight
                live_weights[synapse] = proposed_weight
                applied = {
                    "synapse": synapse,
                    "source_neuron_index": source_index,
                    "target_neuron_index": target_index,
                    "cycle_index": cycle_index,
                    "candidate_activity_hash": str(
                        cycle.get("candidate_activity_hash") or ""
                    ),
                    "critical_period_learning_candidate_hash": str(
                        cycle.get(
                            "critical_period_learning_candidate_hash"
                        )
                        or ""
                    ),
                    "newborn_integration_synapse_hash": str(
                        cycle.get("newborn_integration_synapse_hash") or ""
                    ),
                    "critical_period_learning_cycle_hash": str(
                        cycle.get("critical_period_learning_cycle_hash") or ""
                    ),
                    "previous_weight": float(cycle["current_weight"]),
                    "applied_weight_delta": float(
                        cycle["proposed_weight_delta"]
                    ),
                    "applied_weight": proposed_weight,
                    "pre_spike_count": int(
                        cycle.get("pre_spike_count", 0) or 0
                    ),
                    "post_spike_count": int(
                        cycle.get("post_spike_count", 0) or 0
                    ),
                    "causal_pair_count": int(
                        cycle.get("causal_pair_count", 0) or 0
                    ),
                    "anti_causal_pair_count": int(
                        cycle.get("anti_causal_pair_count", 0) or 0
                    ),
                    "active_cycle": bool(cycle.get("active_cycle")),
                    "newborn_firing_rate_hz": float(
                        cycle.get("newborn_firing_rate_hz", 0.0) or 0.0
                    ),
                    "prediction_error": float(
                        cycle.get("prediction_error", 0.0) or 0.0
                    ),
                    "min_weight": float(
                        cycle.get("min_weight", 0.0) or 0.0
                    ),
                    "max_weight": float(
                        cycle.get("max_weight", 0.0) or 0.0
                    ),
                    "max_learning_rate": float(
                        cycle.get("max_learning_rate", 0.0) or 0.0
                    ),
                    "depression_ratio": float(
                        cycle.get("depression_ratio", 0.0) or 0.0
                    ),
                    "stdp_window_ms": float(
                        cycle.get("stdp_window_ms", 0.0) or 0.0
                    ),
                    "target_firing_rate_hz": float(
                        cycle.get("target_firing_rate_hz", 0.0) or 0.0
                    ),
                    "homeostatic_min_firing_rate_hz": float(
                        cycle.get(
                            "homeostatic_min_firing_rate_hz", 0.0
                        )
                        or 0.0
                    ),
                    "homeostatic_max_firing_rate_hz": float(
                        cycle.get(
                            "homeostatic_max_firing_rate_hz", 0.0
                        )
                        or 0.0
                    ),
                    "inactivity_prune_cycles": int(
                        cycle.get("inactivity_prune_cycles", 0) or 0
                    ),
                    "learning_rule": str(
                        cycle.get("learning_rule") or ""
                    ),
                    "survival_rule": str(
                        cycle.get("survival_rule") or ""
                    ),
                    "maturation_states": deepcopy(
                        list(cycle.get("maturation_states") or [])
                    ),
                    "critical_period_cycles": total_cycles,
                    "critical_period_age_cycles": cycle_index,
                    "critical_period_cycles_remaining": cycles_remaining,
                    "active_cycle_count": active_cycle_count,
                    "inactive_cycle_count": inactive_cycle_count,
                    "minimum_survival_active_cycles": (
                        minimum_survival_active_cycles
                    ),
                    "current_maturation_state": maturation_state,
                    "weight_update_applied": True,
                    "critical_period_age_advanced": True,
                    "maturation_decided": maturation_decided,
                    "pruning_applied": False,
                    "actual_device": str(learned_tensor.device),
                    "tensor_is_cuda": bool(learned_tensor.is_cuda),
                }
                applied["critical_period_learning_application_hash"] = (
                    self._sha256_json(applied)
                )
                applied_cycles.append(applied)
                live_by_synapse[synapse] = deepcopy(applied)
                provenance_item = dict(live_provenance[synapse])
                provenance_item.update(
                    {
                        "current_weight": proposed_weight,
                        "critical_period_age_cycles": cycle_index,
                        "critical_period_cycles_remaining": (
                            cycles_remaining
                        ),
                        "active_cycle_count": active_cycle_count,
                        "inactive_cycle_count": inactive_cycle_count,
                        "current_maturation_state": maturation_state,
                        "last_critical_period_learning_application_hash": (
                            applied[
                                "critical_period_learning_application_hash"
                            ]
                        ),
                    }
                )
                provenance_item["critical_period_learning_history"] = [
                    *list(
                        provenance_item.get(
                            "critical_period_learning_history"
                        )
                        or []
                    ),
                    applied["critical_period_learning_application_hash"],
                ][-64:]
                live_provenance[synapse] = provenance_item
            state["dense_readout_weights"] = learned_tensor
            committed_checkpoint_file = self._committed_checkpoint_path(
                checkpoint_file,
                "thought_newborn_neuron_critical_period_learning",
            )
            event = {
                "completed_at": completed_at,
                "before_state_revision": before_revision,
                "after_state_revision": before_revision + 1,
                "preflight_hash": str(artifact.get("preflight_hash") or ""),
                "thought_newborn_neuron_critical_period_learning_design_hash": (
                    preflight.get(
                        "thought_newborn_neuron_critical_period_learning_"
                        "design_hash"
                    )
                ),
                "newborn_neuron_integration_event_hash": preflight.get(
                    "newborn_neuron_integration_event_hash"
                ),
                "observation_window_id": preflight.get(
                    "observation_window_id"
                ),
                "observation_window_hash": preflight.get(
                    "observation_window_hash"
                ),
                "checkpoint_path": str(checkpoint_file),
                "committed_checkpoint_path": str(
                    committed_checkpoint_file
                ),
                "actual_device": str(learned_tensor.device),
                "tensor_is_cuda": bool(learned_tensor.is_cuda),
                "applied_cycle_count": len(applied_cycles),
                "applied_learning_cycles": deepcopy(applied_cycles),
                "mature_synapse_count": sum(
                    item["current_maturation_state"] == "mature"
                    for item in applied_cycles
                ),
                "prune_eligible_synapse_count": sum(
                    item["current_maturation_state"] == "prune_eligible"
                    for item in applied_cycles
                ),
                "critical_period_synapse_count": sum(
                    item["current_maturation_state"] == "critical_period"
                    for item in applied_cycles
                ),
                "replay_executed": False,
                "training_executed": False,
                "plasticity_applied": True,
                "pruning_applied": False,
            }
            event[
                "newborn_neuron_critical_period_learning_event_hash"
            ] = self._sha256_json(event)
            learning_state["learning_cycle_count"] = int(
                learning_state.get("learning_cycle_count", 0) or 0
            ) + 1
            learning_state["last_learning_cycle"] = deepcopy(event)
            learning_state["recent_events"] = [
                *list(learning_state.get("recent_events") or []),
                deepcopy(event),
            ][-16:]
            state["last_checkpoint_path"] = str(committed_checkpoint_file)
            self._runtime_state.mark_mutated()
            commit = self._commit_mutation(
                committed_checkpoint_file=committed_checkpoint_file,
                operation=(
                    "thought_newborn_neuron_critical_period_learning"
                ),
                before_state=checkpoint_state,
                before_revision=before_revision,
                before_dirty_state=before_dirty_state,
            )
            if not commit["committed"]:
                return self._blocked_thought_newborn_neuron_critical_period_learning(
                    reason=(
                        "post_newborn_neuron_critical_period_learning_"
                        "checkpoint_commit_failed"
                    ),
                    before_revision=before_revision,
                    required_evidence={**required, **commit},
                )
            return {
                "artifact_kind": (
                    "terminus_snn_language_autonomous_snn_language_thought_"
                    "newborn_neuron_critical_period_learning_executor"
                ),
                "surface": (
                    "snn_language_autonomous_snn_language_thought_"
                    "newborn_neuron_critical_period_learning_executor.v1"
                ),
                "accepted": True,
                "ready": True,
                "status": (
                    "thought_newborn_neuron_critical_period_learning_applied"
                ),
                "requires_operator_approval": False,
                "owned_by_marulho": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "advisory": False,
                "executable": True,
                "records_ledger_event": True,
                "generates_text": False,
                "decodes_text": False,
                "runs_replay": False,
                "trains_runtime_model": False,
                "applies_plasticity": True,
                "mutates_runtime_state": True,
                "writes_checkpoint": True,
                "resizes_network": False,
                "adds_neurons": False,
                "adds_synapses": False,
                "prunes_network": False,
                "checkpoint_transaction": {
                    "pre_learning_checkpoint_saved": True,
                    "checkpoint_path": str(checkpoint_file),
                    "committed_checkpoint_path": commit[
                        "committed_checkpoint_path"
                    ],
                    "staged_committed_checkpoint_path": str(
                        committed_checkpoint_file
                    ),
                    "post_learning_checkpoint_saved": True,
                    "post_learning_checkpoint_restore_verified": True,
                    "current_checkpoint_manifest": commit[
                        "current_checkpoint_manifest"
                    ],
                    "restore_verified": checkpoint_verified,
                },
                "autonomous_snn_language_thought_newborn_neuron_"
                "critical_period_learning_event": deepcopy(event),
                "dense_readout_tensor": self._dense_tensor_summary(
                    learned_tensor
                ),
                "critical_period_learning_dense_samples": [
                    {
                        "synapse": item["synapse"],
                        "source_neuron_index": item[
                            "source_neuron_index"
                        ],
                        "target_neuron_index": item[
                            "target_neuron_index"
                        ],
                        "weight": float(
                            learned_tensor[
                                item["source_neuron_index"],
                                item["target_neuron_index"],
                            ].item()
                        ),
                    }
                    for item in applied_cycles
                ],
                "before": {"state_revision": before_revision},
                "after": {
                    "state_revision": int(self._runtime_state.state_revision),
                    **self._runtime_state.mutation_summary(),
                },
                "promotion_gate": {
                    "status": (
                        "ready_for_autonomous_snn_language_thought_newborn_"
                        "neuron_critical_period_learning_event_review"
                    ),
                    "eligible_for_autonomous_snn_language_thought_newborn_"
                    "neuron_critical_period_learning_event_review": True,
                    "eligible_for_language_generation": False,
                    "eligible_for_replay_memory": False,
                    "eligible_for_plasticity_application": False,
                    "eligible_for_fact_promotion": False,
                    "eligible_for_action": False,
                    "eligible_for_cognition_substrate": False,
                },
            }

    def apply_autonomous_snn_language_thought_newborn_synapse_pruning(
        self,
        *,
        autonomous_snn_language_thought_newborn_synapse_pruning_preflight: Mapping[
            str, Any
        ],
        expected_state_revision: int,
        checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        """Remove reviewed prune-eligible newborn synapses transactionally."""

        with self._lock:
            before_revision = int(self._runtime_state.state_revision)
            artifact = dict(
                autonomous_snn_language_thought_newborn_synapse_pruning_preflight
                or {}
            )
            preflight = dict(
                artifact.get(
                    "autonomous_snn_language_thought_newborn_synapse_"
                    "pruning_preflight"
                )
                or {}
            )
            gate = dict(artifact.get("promotion_gate") or {})
            candidates = [
                dict(item)
                for item in list(
                    preflight.get("resolved_prune_candidates") or []
                )
                if isinstance(item, Mapping)
            ]
            state = self._language_plasticity_state()
            weights = dict(state.get("sparse_transition_weights") or {})
            dense_tensor = state.get("dense_readout_weights")
            provenance = dict(state.get("synapse_provenance_by_key") or {})
            developmental_state = dict(
                state.get(
                    "thought_newborn_neuron_critical_period_learning"
                )
                or {}
            )
            developmental = dict(
                developmental_state.get("by_synapse") or {}
            )
            candidate_checks: list[dict[str, Any]] = []
            for item in candidates:
                synapse = str(item.get("synapse") or "")
                source_index = int(item.get("source_neuron_index", -1))
                target_index = int(item.get("target_neuron_index", -1))
                current_weight = float(
                    item.get("current_weight", 0.0) or 0.0
                )
                candidate_hash = str(
                    item.get("newborn_synapse_pruning_candidate_hash")
                    or ""
                )
                recomputed_hash = self._sha256_json(
                    {
                        key: value
                        for key, value in item.items()
                        if key != "newborn_synapse_pruning_candidate_hash"
                    }
                )
                live_development = dict(developmental.get(synapse) or {})
                live_provenance = dict(provenance.get(synapse) or {})
                checks = {
                    "canonical_synapse": synapse
                    == f"{source_index}:{target_index}",
                    "candidate_hash_matches": len(candidate_hash) == 64
                    and candidate_hash == recomputed_hash,
                    "sparse_weight_matches": synapse in weights
                    and math.isclose(
                        float(weights[synapse]),
                        current_weight,
                        rel_tol=1e-7,
                        abs_tol=1e-9,
                    ),
                    "dense_tensor_available": isinstance(
                        dense_tensor, torch.Tensor
                    ),
                    "dense_indices_valid": isinstance(
                        dense_tensor, torch.Tensor
                    )
                    and 0 <= source_index < int(dense_tensor.shape[0])
                    and 0 <= target_index < int(dense_tensor.shape[1]),
                    "dense_weight_matches": isinstance(
                        dense_tensor, torch.Tensor
                    )
                    and 0 <= source_index < int(dense_tensor.shape[0])
                    and 0 <= target_index < int(dense_tensor.shape[1])
                    and math.isclose(
                        float(
                            dense_tensor[source_index, target_index].item()
                        ),
                        current_weight,
                        rel_tol=1e-7,
                        abs_tol=1e-9,
                    ),
                    "developmental_state_terminal": (
                        live_development.get("current_maturation_state")
                        == "prune_eligible"
                        and bool(live_development.get("maturation_decided"))
                        and int(
                            live_development.get(
                                "critical_period_cycles_remaining", -1
                            )
                        )
                        == 0
                    ),
                    "developmental_application_matches": str(
                        live_development.get(
                            "critical_period_learning_application_hash"
                        )
                        or ""
                    )
                    == str(
                        item.get(
                            "critical_period_learning_application_hash"
                        )
                        or ""
                    ),
                    "provenance_is_newborn": live_provenance.get(
                        "provenance_type"
                    )
                    == "newborn_neuron_integration",
                    "not_preapplied": not bool(item.get("pruning_applied")),
                }
                candidate_checks.append(
                    {
                        "synapse": synapse,
                        "checks": checks,
                        "verified": all(checks.values()),
                    }
                )
            preflight_revision = int(
                preflight.get("expected_state_revision", -1)
            )
            required = {
                "preflight_surface_available": artifact.get("surface")
                == (
                    "snn_language_autonomous_snn_language_thought_"
                    "newborn_synapse_pruning_preflight.v1"
                ),
                "preflight_ready": bool(artifact.get("accepted"))
                and bool(artifact.get("ready"))
                and bool(
                    gate.get(
                        "eligible_for_autonomous_snn_language_thought_"
                        "newborn_synapse_pruning_executor"
                    )
                ),
                "expected_revision_current": int(expected_state_revision)
                == before_revision,
                "expected_revision_matches_preflight": preflight_revision
                == int(expected_state_revision),
                "preflight_hash_available": len(
                    str(artifact.get("preflight_hash") or "")
                )
                == 64,
                "design_hash_available": len(
                    str(
                        preflight.get(
                            "newborn_synapse_pruning_design_hash"
                        )
                        or ""
                    )
                )
                == 64,
                "candidates_available": bool(candidates)
                and len(candidates)
                == int(preflight.get("resolved_prune_count", 0) or 0),
                "all_candidates_verified": bool(candidate_checks)
                and all(item["verified"] for item in candidate_checks),
                "checkpoint_path_available": bool(
                    str(preflight.get("checkpoint_path") or "").strip()
                    or checkpoint_path
                ),
                "operator_approval_not_required": not bool(
                    artifact.get("requires_operator_approval")
                )
                and not bool(preflight.get("operator_approval_required")),
            }
            if not all(required.values()):
                return self._blocked_thought_newborn_synapse_pruning(
                    reason="blocked_missing_reviewed_newborn_synapse_evidence",
                    before_revision=before_revision,
                    required_evidence=required,
                )

            checkpoint_state = deepcopy(state)
            before_dirty_state = bool(self._runtime_state.dirty_state)
            checkpoint = self._save_checkpoint(
                checkpoint_path
                or str(preflight.get("checkpoint_path") or "")
                or None
            )
            checkpoint_file = Path(
                str(checkpoint.get("path") or self._checkpoint_path())
            )
            checkpoint_verified = self._verify_checkpoint_transaction(
                checkpoint_file,
                checkpoint_state,
                before_revision,
            )
            if not checkpoint_verified:
                return self._blocked_thought_newborn_synapse_pruning(
                    reason="checkpoint_save_missing",
                    before_revision=before_revision,
                    required_evidence={
                        **required,
                        "pre_pruning_checkpoint_saved": (
                            checkpoint_file.exists()
                        ),
                        "pre_pruning_checkpoint_restore_verified": (
                            checkpoint_verified
                        ),
                    },
                )

            pruned_tensor = dense_tensor.detach().clone()
            live_weights = state.setdefault(
                "sparse_transition_weights", {}
            )
            live_provenance = state.setdefault(
                "synapse_provenance_by_key", {}
            )
            live_developmental_state = state.setdefault(
                "thought_newborn_neuron_critical_period_learning", {}
            )
            live_developmental = live_developmental_state.setdefault(
                "by_synapse", {}
            )
            tombstones = state.setdefault(
                "pruned_synapse_provenance_by_key", {}
            )
            completed_at = datetime.now(timezone.utc).isoformat()
            applied_prunes: list[dict[str, Any]] = []
            for item in candidates:
                synapse = str(item["synapse"])
                source_index = int(item["source_neuron_index"])
                target_index = int(item["target_neuron_index"])
                removed_weight = float(live_weights.pop(synapse))
                pruned_tensor[source_index, target_index] = 0.0
                previous_development = dict(
                    live_developmental.get(synapse) or {}
                )
                previous_provenance = dict(
                    live_provenance.pop(synapse)
                )
                applied = {
                    **deepcopy(item),
                    "removed_weight": removed_weight,
                    "pruned_at": completed_at,
                    "current_maturation_state": "pruned",
                    "pruning_applied": True,
                    "sparse_edge_removed": True,
                    "dense_weight_zeroed": True,
                    "neuron_capacity_reduced": False,
                }
                applied["newborn_synapse_pruning_application_hash"] = (
                    self._sha256_json(applied)
                )
                live_developmental[synapse] = {
                    **previous_development,
                    "current_weight": 0.0,
                    "applied_weight": 0.0,
                    "current_maturation_state": "pruned",
                    "pruning_applied": True,
                    "pruned_at": completed_at,
                    "newborn_synapse_pruning_application_hash": applied[
                        "newborn_synapse_pruning_application_hash"
                    ],
                }
                tombstones[synapse] = {
                    **previous_provenance,
                    "live_synapse": False,
                    "final_weight": removed_weight,
                    "current_weight": 0.0,
                    "current_maturation_state": "pruned",
                    "pruned_at": completed_at,
                    "newborn_synapse_pruning_candidate_hash": item[
                        "newborn_synapse_pruning_candidate_hash"
                    ],
                    "newborn_synapse_pruning_application_hash": applied[
                        "newborn_synapse_pruning_application_hash"
                    ],
                }
                applied_prunes.append(applied)
            state["dense_readout_weights"] = pruned_tensor
            committed_checkpoint_file = self._committed_checkpoint_path(
                checkpoint_file,
                "thought_newborn_synapse_pruning",
            )
            event = {
                "completed_at": completed_at,
                "before_state_revision": before_revision,
                "after_state_revision": before_revision + 1,
                "preflight_hash": str(
                    artifact.get("preflight_hash") or ""
                ),
                "newborn_synapse_pruning_design_hash": str(
                    preflight.get("newborn_synapse_pruning_design_hash")
                    or ""
                ),
                "maturation_outcome_review_hash": str(
                    preflight.get("maturation_outcome_review_hash") or ""
                ),
                "checkpoint_path": str(checkpoint_file),
                "committed_checkpoint_path": str(
                    committed_checkpoint_file
                ),
                "actual_device": str(pruned_tensor.device),
                "tensor_is_cuda": bool(pruned_tensor.is_cuda),
                "pruned_synapse_count": len(applied_prunes),
                "pruned_synapses": deepcopy(applied_prunes),
                "neuron_capacity_reduction_applied": False,
                "plasticity_applied": True,
                "pruning_applied": True,
            }
            event["newborn_synapse_pruning_event_hash"] = (
                self._sha256_json(event)
            )
            pruning_state = state.setdefault(
                "thought_newborn_synapse_pruning", {}
            )
            pruning_state["pruning_count"] = int(
                pruning_state.get("pruning_count", 0) or 0
            ) + 1
            pruning_state["pruned_synapse_count_total"] = int(
                pruning_state.get("pruned_synapse_count_total", 0) or 0
            ) + len(applied_prunes)
            pruning_state["last_pruning"] = deepcopy(event)
            pruning_state["recent_events"] = [
                *list(pruning_state.get("recent_events") or []),
                deepcopy(event),
            ][-16:]
            state["last_checkpoint_path"] = str(
                committed_checkpoint_file
            )
            self._runtime_state.mark_mutated()
            commit = self._commit_mutation(
                committed_checkpoint_file=committed_checkpoint_file,
                operation="thought_newborn_synapse_pruning",
                before_state=checkpoint_state,
                before_revision=before_revision,
                before_dirty_state=before_dirty_state,
            )
            if not commit["committed"]:
                return self._blocked_thought_newborn_synapse_pruning(
                    reason="post_newborn_synapse_pruning_checkpoint_commit_failed",
                    before_revision=before_revision,
                    required_evidence={**required, **commit},
                )
            return {
                "artifact_kind": (
                    "terminus_snn_language_autonomous_snn_language_thought_"
                    "newborn_synapse_pruning_executor"
                ),
                "surface": (
                    "snn_language_autonomous_snn_language_thought_"
                    "newborn_synapse_pruning_executor.v1"
                ),
                "accepted": True,
                "ready": True,
                "status": "thought_newborn_synapse_pruning_applied",
                "requires_operator_approval": False,
                "advisory": False,
                "executable": True,
                "records_ledger_event": True,
                "applies_plasticity": True,
                "mutates_runtime_state": True,
                "writes_checkpoint": True,
                "resizes_network": False,
                "adds_neurons": False,
                "adds_synapses": False,
                "prunes_network": True,
                "checkpoint_transaction": {
                    "pre_pruning_checkpoint_saved": True,
                    "checkpoint_path": str(checkpoint_file),
                    "committed_checkpoint_path": commit[
                        "committed_checkpoint_path"
                    ],
                    "staged_committed_checkpoint_path": str(
                        committed_checkpoint_file
                    ),
                    "post_pruning_checkpoint_saved": True,
                    "post_pruning_checkpoint_restore_verified": True,
                    "current_checkpoint_manifest": commit[
                        "current_checkpoint_manifest"
                    ],
                    "restore_verified": checkpoint_verified,
                },
                "autonomous_snn_language_thought_newborn_synapse_"
                "pruning_event": deepcopy(event),
                "dense_readout_tensor": self._dense_tensor_summary(
                    pruned_tensor
                ),
                "newborn_synapse_pruning_dense_samples": [
                    {
                        "synapse": item["synapse"],
                        "source_neuron_index": item[
                            "source_neuron_index"
                        ],
                        "target_neuron_index": item[
                            "target_neuron_index"
                        ],
                        "weight": float(
                            pruned_tensor[
                                item["source_neuron_index"],
                                item["target_neuron_index"],
                            ].item()
                        ),
                    }
                    for item in applied_prunes
                ],
                "before": {"state_revision": before_revision},
                "after": {
                    "state_revision": int(
                        self._runtime_state.state_revision
                    ),
                    **self._runtime_state.mutation_summary(),
                },
                "promotion_gate": {
                    "status": (
                        "ready_for_autonomous_snn_language_thought_newborn_"
                        "synapse_pruning_event_review"
                    ),
                    "eligible_for_autonomous_snn_language_thought_newborn_"
                    "synapse_pruning_event_review": True,
                },
            }

    def apply_dense_readout_training_loop(
        self,
        *,
        dense_readout_training_loop_preflight: Mapping[str, Any],
        training_transitions: list[Mapping[str, Any]],
        expected_state_revision: int,
        operator_id: str,
        confirmation: bool,
        checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        """Apply bounded local dense-readout training without text generation."""

        with self._lock:
            before_revision = int(self._runtime_state.state_revision)
            preflight = dict(dense_readout_training_loop_preflight)
            gate = (
                preflight.get("promotion_gate")
                if isinstance(preflight.get("promotion_gate"), Mapping)
                else {}
            )
            required_preflight = (
                gate.get("required_evidence")
                if isinstance(gate.get("required_evidence"), Mapping)
                else {}
            )
            design = (
                preflight.get("training_design")
                if isinstance(preflight.get("training_design"), Mapping)
                else {}
            )
            tensor_summary = (
                preflight.get("tensor_summary")
                if isinstance(preflight.get("tensor_summary"), Mapping)
                else {}
            )
            expected_shape = self._shape_pair(tensor_summary.get("shape"))
            learning_rate = float(design.get("learning_rate", 0.0) or 0.0)
            max_delta_norm = float(design.get("max_delta_norm", 0.0) or 0.0)
            transition_budget = int(design.get("transition_budget", 0) or 0)
            transitions = [
                dict(item) for item in list(training_transitions or []) if isinstance(item, Mapping)
            ]
            state = self._language_plasticity_state()
            tensor = state.get("dense_readout_weights")
            tensor_available = isinstance(tensor, torch.Tensor)
            tensor_shape = [int(item) for item in list(tensor.shape)] if tensor_available else []
            parsed: list[dict[str, Any]] = []
            for index, transition in enumerate(transitions):
                try:
                    pre_indices = [
                        int(value)
                        for value in list(transition.get("pre_indices") or [])
                    ]
                    post_indices = [
                        int(value)
                        for value in list(transition.get("post_indices") or [])
                    ]
                except (TypeError, ValueError):
                    pre_indices = []
                    post_indices = []
                parsed.append(
                    {
                        "transition_id": str(
                            transition.get("transition_id") or f"dense_training_{index}"
                        ),
                        "pre_indices": pre_indices[:32],
                        "post_indices": post_indices[:32],
                    }
                )
            canonical_indices = bool(
                tensor_available
                and parsed
                and all(
                    bool(item["pre_indices"])
                    and bool(item["post_indices"])
                    and all(0 <= pre < tensor.shape[0] for pre in item["pre_indices"])
                    and all(0 <= post < tensor.shape[1] for post in item["post_indices"])
                    for item in parsed
                )
            )
            required = {
                "confirmation": bool(confirmation),
                "operator_id_available": bool(str(operator_id or "").strip()),
                "expected_revision_current": int(expected_state_revision)
                == before_revision,
                "preflight_surface_available": preflight.get("surface")
                == "snn_language_dense_readout_training_loop_preflight.v1",
                "preflight_ready": bool(preflight.get("ready"))
                and gate.get("status")
                == "ready_for_checkpoint_backed_dense_readout_training_executor",
                "preflight_not_executable": not bool(preflight.get("executable")),
                "preflight_does_not_mutate": not bool(
                    preflight.get("mutates_runtime_state")
                ),
                "preflight_does_not_generate": not bool(preflight.get("generates_text")),
                "preflight_checkpoint_current": bool(
                    required_preflight.get("expected_state_revision_current")
                ),
                "preflight_checkpoint_path_available": bool(
                    required_preflight.get("checkpoint_path_available")
                ),
                "preflight_bounded_delta_capability_available": bool(
                    required_preflight.get(
                        "bounded_delta_application_capability_available"
                    )
                ),
                "dense_tensor_available": tensor_available,
                "dense_tensor_shape_matches_preflight": tensor_shape == expected_shape,
                "training_transitions_available": bool(parsed),
                "training_transition_count_bounded": bool(transition_budget)
                and 0 < len(parsed) <= transition_budget,
                "training_transition_indices_canonical": canonical_indices,
                "learning_rate_bounded": 0.0 < learning_rate <= 0.25,
                "delta_norm_bounded": 0.0 < max_delta_norm <= 0.25,
                "no_text_generation": not bool(preflight.get("generates_text")),
                "no_external_checkpoint": not bool(preflight.get("loads_external_checkpoint")),
            }
            if not all(required.values()):
                return self._blocked_dense_readout_training(
                    reason="blocked_missing_dense_readout_training_evidence",
                    before_revision=before_revision,
                    required_evidence=required,
                )

            checkpoint_state = deepcopy(state)
            before_dirty_state = bool(self._runtime_state.dirty_state)
            checkpoint = self._save_checkpoint(
                checkpoint_path or str(preflight.get("checkpoint_path") or "")
            )
            checkpoint_file = Path(str(checkpoint.get("path") or self._checkpoint_path()))
            checkpoint_verified = self._verify_checkpoint_transaction(
                checkpoint_file,
                checkpoint_state,
                before_revision,
            )
            if not checkpoint_verified:
                return self._blocked_dense_readout_training(
                    reason="checkpoint_save_missing",
                    before_revision=before_revision,
                    required_evidence={
                        **required,
                        "pre_training_checkpoint_saved": checkpoint_file.exists(),
                        "pre_training_checkpoint_restore_verified": checkpoint_verified,
                    },
                )

            trained = tensor.detach().clone().to(dtype=torch.float32)
            applied_cells: dict[str, dict[str, Any]] = {}
            base_delta = min(learning_rate, max_delta_norm)
            for transition in parsed:
                cell_count = max(
                    1,
                    len(transition["pre_indices"]) * len(transition["post_indices"]),
                )
                cell_delta = base_delta / float(cell_count)
                for pre_index in transition["pre_indices"]:
                    for post_index in transition["post_indices"]:
                        previous = float(trained[pre_index, post_index].item())
                        updated = max(-1.0, min(1.0, previous + cell_delta))
                        trained[pre_index, post_index] = updated
                        key = f"{pre_index}:{post_index}"
                        applied = applied_cells.setdefault(
                            key,
                            {
                                "pre_index": pre_index,
                                "post_index": post_index,
                                "previous_weight": previous,
                                "updated_weight": updated,
                                "delta": 0.0,
                                "transition_ids": [],
                            },
                        )
                        applied["updated_weight"] = updated
                        applied["delta"] = float(applied["delta"]) + (
                            updated - previous
                        )
                        applied["transition_ids"].append(transition["transition_id"])

            weights = state.setdefault("sparse_transition_weights", {})
            provenance_by_key = state.setdefault("synapse_provenance_by_key", {})
            for key, applied in sorted(applied_cells.items()):
                updated = float(applied["updated_weight"])
                if abs(updated) > 0.0:
                    weights[key] = updated
                elif key in weights:
                    del weights[key]
                provenance_by_key[key] = {
                    "source": "dense_readout_training_loop",
                    "operator_id": operator_id,
                    "transition_ids": list(applied["transition_ids"])[:16],
                    "preflight_hash": preflight.get("preflight_hash"),
                    "checkpoint_path": str(checkpoint_file),
                }

            training_state = state.setdefault("dense_readout_training", {})
            committed_checkpoint_file = self._committed_checkpoint_path(
                checkpoint_file,
                "dense_readout_training",
            )
            now = datetime.now(timezone.utc).isoformat()
            event = {
                "completed_at": now,
                "operator_id": str(operator_id or "").strip(),
                "before_state_revision": before_revision,
                "after_state_revision": before_revision + 1,
                "checkpoint_path": str(checkpoint_file),
                "committed_checkpoint_path": str(committed_checkpoint_file),
                "preflight_hash": preflight.get("preflight_hash"),
                "training_transition_count": len(parsed),
                "updated_cell_count": len(applied_cells),
                "learning_rate": learning_rate,
                "max_delta_norm": max_delta_norm,
                "returns_trained_weights": False,
                "generates_text": False,
            }
            state["dense_readout_weights"] = trained
            training_state["training_count"] = int(training_state.get("training_count", 0) or 0) + 1
            training_state["last_training"] = deepcopy(event)
            recent_events = list(training_state.get("recent_events") or [])
            training_state["recent_events"] = [*recent_events, deepcopy(event)][-8:]
            state["last_checkpoint_path"] = str(committed_checkpoint_file)
            self._runtime_state.mark_mutated()
            commit = self._commit_mutation(
                committed_checkpoint_file=committed_checkpoint_file,
                operation="dense_readout_training",
                before_state=checkpoint_state,
                before_revision=before_revision,
                before_dirty_state=before_dirty_state,
            )
            if not commit["committed"]:
                return self._blocked_dense_readout_training(
                    reason="post_training_checkpoint_commit_failed",
                    before_revision=before_revision,
                    required_evidence={**required, **commit},
                )
            return {
                "artifact_kind": "terminus_snn_language_dense_readout_training",
                "surface": "snn_language_dense_readout_training.v1",
                "accepted": True,
                "status": "dense_readout_training_applied",
                "owned_by_marulho": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "generates_text": False,
                "decodes_text": False,
                "trains_runtime_model": True,
                "applies_plasticity": True,
                "mutates_runtime_state": True,
                "writes_checkpoint": True,
                "resizes_network": False,
                "returns_trained_weights": False,
                "checkpoint_transaction": {
                    "pre_training_checkpoint_saved": True,
                    "checkpoint_path": str(checkpoint_file),
                    "committed_checkpoint_path": commit["committed_checkpoint_path"],
                    "staged_committed_checkpoint_path": str(committed_checkpoint_file),
                    "post_training_checkpoint_saved": True,
                    "post_training_checkpoint_restore_verified": True,
                    "current_checkpoint_manifest": commit["current_checkpoint_manifest"],
                    "restore_verified": checkpoint_verified,
                },
                "dense_readout_training": deepcopy(event),
                "dense_readout_tensor": self._dense_tensor_summary(trained),
                "updated_cell_count": len(applied_cells),
                "applied_cell_summaries": list(applied_cells.values())[:16],
                "before": {"state_revision": before_revision},
                "after": self._runtime_state.mutation_summary(),
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = deepcopy(self._language_plasticity_state())
            weights = dict(state.get("sparse_transition_weights") or {})
            maintenance = dict(state.get("homeostatic_maintenance") or {})
            regeneration = dict(state.get("synapse_regeneration") or {})
            live_application = dict(state.get("live_application") or {})
            dense_training = dict(state.get("dense_readout_training") or {})
            capacity_mutation = dict(state.get("thought_capacity_mutation") or {})
            newborn_integration = dict(
                state.get("thought_newborn_neuron_integration") or {}
            )
            critical_period_learning = dict(
                state.get(
                    "thought_newborn_neuron_critical_period_learning"
                )
                or {}
            )
            newborn_synapse_pruning = dict(
                state.get("thought_newborn_synapse_pruning") or {}
            )
            capacity = self._language_capacity_state(state)
            dense_layout = self._dense_readout_layout_state(state, capacity)
            dense_tensor = state.get("dense_readout_weights")
            last_newborn_integration = (
                dict(newborn_integration.get("last_integration"))
                if isinstance(
                    newborn_integration.get("last_integration"), Mapping
                )
                else {}
            )
            return {
                "surface": "snn_language_plasticity_runtime_state.v1",
                "owned_by_marulho": True,
                "external_dependency": False,
                "language_capacity": deepcopy(capacity),
                "dense_readout_layout": deepcopy(dense_layout),
                "dense_readout_tensor": self._dense_tensor_summary(dense_tensor),
                "dense_readout_training": deepcopy(dense_training),
                "language_neuron_count": capacity["language_neuron_count"],
                "sparse_edge_budget": capacity["sparse_edge_budget"],
                "outgoing_fanout_budget": capacity["outgoing_fanout_budget"],
                "sparse_transition_weights": deepcopy(weights),
                "sparse_transition_weight_count": len(weights),
                "applied_update_count": int(state.get("applied_update_count", 0) or 0),
                "homeostatic_maintenance_count": int(maintenance.get("maintenance_count", 0) or 0),
                "pruned_synapse_count_total": int(maintenance.get("pruned_synapse_count_total", 0) or 0),
                "last_homeostatic_maintenance": deepcopy(maintenance.get("last_maintenance")),
                "recent_homeostatic_maintenance": deepcopy(list(maintenance.get("recent_events") or [])),
                "regeneration_count": int(regeneration.get("regeneration_count", 0) or 0),
                "regenerated_synapse_count_total": int(regeneration.get("regenerated_synapse_count_total", 0) or 0),
                "last_synapse_regeneration": deepcopy(regeneration.get("last_regeneration")),
                "recent_synapse_regeneration": deepcopy(list(regeneration.get("recent_events") or [])),
                "synapse_provenance_by_key": deepcopy(dict(state.get("synapse_provenance_by_key") or {})),
                "last_live_application": deepcopy(live_application.get("last_application")),
                "recent_live_applications": deepcopy(list(live_application.get("recent_events") or [])),
                "thought_capacity_mutation_count": int(
                    capacity_mutation.get("mutation_count", 0) or 0
                ),
                "last_thought_capacity_mutation": deepcopy(
                    capacity_mutation.get("last_mutation")
                ),
                "recent_thought_capacity_mutations": deepcopy(
                    list(capacity_mutation.get("recent_events") or [])
                ),
                "thought_newborn_neuron_integration_count": int(
                    newborn_integration.get("integration_count", 0) or 0
                ),
                "last_thought_newborn_neuron_integration": deepcopy(
                    newborn_integration.get("last_integration")
                ),
                "recent_thought_newborn_neuron_integrations": deepcopy(
                    list(newborn_integration.get("recent_events") or [])
                ),
                "newborn_integration_dense_samples": (
                    self._newborn_integration_dense_samples(
                        dense_tensor,
                        last_newborn_integration,
                    )
                ),
                "thought_newborn_neuron_critical_period_learning_cycle_count": int(
                    critical_period_learning.get(
                        "learning_cycle_count", 0
                    )
                    or 0
                ),
                "last_thought_newborn_neuron_critical_period_learning": deepcopy(
                    critical_period_learning.get("last_learning_cycle")
                ),
                "recent_thought_newborn_neuron_critical_period_learning": deepcopy(
                    list(
                        critical_period_learning.get("recent_events") or []
                    )
                ),
                "newborn_neuron_critical_period_state_by_synapse": deepcopy(
                    dict(critical_period_learning.get("by_synapse") or {})
                ),
                "critical_period_learning_dense_samples": (
                    self._critical_period_learning_dense_samples(
                        dense_tensor,
                        (
                            critical_period_learning.get(
                                "last_learning_cycle"
                            )
                            if isinstance(
                                critical_period_learning.get(
                                    "last_learning_cycle"
                                ),
                                Mapping,
                            )
                            else {}
                        ),
                    )
                ),
                "thought_newborn_synapse_pruning_count": int(
                    newborn_synapse_pruning.get("pruning_count", 0) or 0
                ),
                "thought_newborn_synapse_pruned_count_total": int(
                    newborn_synapse_pruning.get(
                        "pruned_synapse_count_total", 0
                    )
                    or 0
                ),
                "last_thought_newborn_synapse_pruning": deepcopy(
                    newborn_synapse_pruning.get("last_pruning")
                ),
                "recent_thought_newborn_synapse_pruning": deepcopy(
                    list(
                        newborn_synapse_pruning.get("recent_events") or []
                    )
                ),
                "pruned_synapse_provenance_by_key": deepcopy(
                    dict(
                        state.get("pruned_synapse_provenance_by_key") or {}
                    )
                ),
                "last_applied_at": state.get("last_applied_at"),
                "last_operator_id": state.get("last_operator_id"),
                "last_checkpoint_path": state.get("last_checkpoint_path"),
            }

    @classmethod
    def _language_capacity_state(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        raw = (
            state.get("language_capacity")
            if isinstance(state.get("language_capacity"), Mapping)
            else {}
        )
        return {
            "surface": _LANGUAGE_CAPACITY_SURFACE,
            "owned_by_marulho": True,
            "external_dependency": False,
            "language_neuron_count": cls._positive_capacity_int(
                raw.get("language_neuron_count"),
                default=_LANGUAGE_NEURON_COUNT,
                minimum=_LANGUAGE_NEURON_COUNT,
            ),
            "sparse_edge_budget": cls._positive_capacity_int(
                raw.get("sparse_edge_budget"),
                default=_MAX_SPARSE_TRANSITION_EDGES,
                minimum=_MAX_SPARSE_TRANSITION_EDGES,
            ),
            "outgoing_fanout_budget": cls._positive_capacity_int(
                raw.get("outgoing_fanout_budget"),
                default=_MAX_OUTGOING_FANOUT,
                minimum=_MAX_OUTGOING_FANOUT,
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

    @classmethod
    def _dense_readout_layout_state(
        cls,
        state: Mapping[str, Any],
        capacity: Mapping[str, Any],
    ) -> dict[str, Any]:
        raw = (
            state.get("dense_readout_layout")
            if isinstance(state.get("dense_readout_layout"), Mapping)
            else {}
        )
        target_neurons = cls._positive_capacity_int(
            raw.get("target_language_neuron_count"),
            default=int(capacity.get("language_neuron_count", _LANGUAGE_NEURON_COUNT)),
            minimum=_LANGUAGE_NEURON_COUNT,
        )
        layout_migration = (
            raw.get("layout_migration")
            if isinstance(raw.get("layout_migration"), Mapping)
            else {}
        )
        tensor_materialization = (
            raw.get("tensor_materialization")
            if isinstance(raw.get("tensor_materialization"), Mapping)
            else {}
        )
        current_shape = [
            _LANGUAGE_NEURON_COUNT,
            _LANGUAGE_NEURON_COUNT,
        ]
        target_shape = [target_neurons, target_neurons]
        dense_resize_applied = bool(raw.get("dense_resize_applied"))
        layout_migration_applied = bool(layout_migration.get("applied"))
        tensor_materialization_applied = bool(tensor_materialization.get("applied"))
        return {
            "surface": _DENSE_READOUT_LAYOUT_SURFACE,
            "raw_surface": str(raw.get("surface") or "") if raw else None,
            "present": bool(raw),
            "owned_by_marulho": True,
            "external_dependency": False,
            "current_dense_readout_shape": current_shape,
            "target_dense_readout_shape": target_shape,
            "preserved_dense_window": current_shape,
            "zero_initialized_new_dense_cell_count": max(
                0,
                int(target_neurons * target_neurons)
                - int(_LANGUAGE_NEURON_COUNT * _LANGUAGE_NEURON_COUNT),
            ),
            "target_language_neuron_count": target_neurons,
            "requires_cuda_relayout": target_neurons > _LANGUAGE_NEURON_COUNT
            and not tensor_materialization_applied,
            "checkpoint_required_before_resize": not dense_resize_applied,
            "layout_migration_applied": layout_migration_applied,
            "tensor_materialization_applied": tensor_materialization_applied,
            "dense_resize_applied": dense_resize_applied,
            "dynamic_dense_readout_enabled": dense_resize_applied,
            "migration_status": "layout_metadata_only_resize_pending"
            if target_neurons > _LANGUAGE_NEURON_COUNT and not layout_migration_applied
            else str(
                raw.get(
                    "migration_status",
                    "dense_readout_tensor_materialized"
                    if tensor_materialization_applied
                    else "layout_migration_applied_tensor_resize_pending"
                    if layout_migration_applied
                    else "fixed_dense_layout",
                )
            ),
            "layout_migration": deepcopy(dict(layout_migration)),
            "tensor_materialization": deepcopy(dict(tensor_materialization)),
            "resizes_network": False,
            "writes_checkpoint": False,
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

    def maintain_transition_memory(
        self,
        *,
        expected_state_revision: int,
        operator_id: str,
        confirmation: bool,
        checkpoint_path: str | None = None,
        decay_factor: float = 0.98,
        prune_below: float = 0.005,
        max_outgoing_row_mass: float = 1.0,
    ) -> dict[str, Any]:
        """Decay, normalize, and prune persistent sparse transition memory."""

        with self._lock:
            before_revision = int(self._runtime_state.state_revision)
            decay = float(decay_factor)
            threshold = float(prune_below)
            row_mass_limit = float(max_outgoing_row_mass)
            required = {
                "confirmation": bool(confirmation),
                "operator_id_available": bool(str(operator_id or "").strip()),
                "expected_revision_current": int(expected_state_revision) == before_revision,
                "decay_factor_bounded": 0.0 < decay <= 1.0,
                "prune_threshold_bounded": 0.0 <= threshold <= 0.25,
                "outgoing_row_mass_bounded": 0.0 < row_mass_limit <= _MAX_OUTGOING_ROW_MASS,
            }
            if not all(required.values()):
                return self._blocked_maintenance(
                    reason="blocked_missing_homeostatic_maintenance_evidence",
                    before_revision=before_revision,
                    required_evidence=required,
                )

            checkpoint_state = deepcopy(self._language_plasticity_state())
            before_dirty_state = bool(self._runtime_state.dirty_state)
            checkpoint = self._save_checkpoint(checkpoint_path)
            checkpoint_file = Path(str(checkpoint.get("path") or self._checkpoint_path()))
            checkpoint_verified = self._verify_checkpoint_transaction(checkpoint_file, checkpoint_state, before_revision)
            if not checkpoint_verified:
                return self._blocked_maintenance(
                    reason="checkpoint_save_missing",
                    before_revision=before_revision,
                    required_evidence={
                        **required,
                        "pre_maintenance_checkpoint_saved": checkpoint_file.exists(),
                        "pre_maintenance_checkpoint_restore_verified": checkpoint_verified,
                    },
                )

            state = self._language_plasticity_state()
            weights = state.setdefault("sparse_transition_weights", {})
            before_weights = {str(key): float(value) for key, value in dict(weights).items()}
            decayed = {key: value * decay for key, value in before_weights.items()}
            row_mass_before = self._row_mass_by_pre_index(decayed)
            normalized: dict[str, float] = {}
            for key, value in decayed.items():
                pre_index = key.split(":", maxsplit=1)[0]
                row_mass = float(row_mass_before.get(pre_index, 0.0))
                scale = min(1.0, row_mass_limit / row_mass) if row_mass > 0.0 else 1.0
                normalized[key] = value * scale
            retained = {key: value for key, value in normalized.items() if abs(value) >= threshold}
            pruned_keys = sorted(set(normalized).difference(retained))
            pruned_synapses = [
                {
                    "synapse": key,
                    "previous_weight": before_weights[key],
                    "normalized_weight": normalized[key],
                }
                for key in pruned_keys
            ]
            weights.clear()
            weights.update(retained)
            now = datetime.now(timezone.utc).isoformat()
            maintenance = state.setdefault("homeostatic_maintenance", {})
            maintenance["maintenance_count"] = int(maintenance.get("maintenance_count", 0) or 0) + 1
            maintenance["pruned_synapse_count_total"] = (
                int(maintenance.get("pruned_synapse_count_total", 0) or 0) + len(pruned_keys)
            )
            maintenance["last_maintenance"] = {
                "completed_at": now,
                "operator_id": operator_id,
                "checkpoint_path": str(checkpoint_file),
                "decay_factor": decay,
                "prune_below": threshold,
                "max_outgoing_row_mass": row_mass_limit,
                "before_synapse_count": len(before_weights),
                "after_synapse_count": len(retained),
                "pruned_synapse_count": len(pruned_keys),
                "normalized_row_count": len(set(key.split(":", maxsplit=1)[0] for key in decayed if row_mass_before.get(key.split(":", maxsplit=1)[0], 0.0) > row_mass_limit)),
                "max_outgoing_row_mass_after": max(self._row_mass_by_pre_index(retained).values(), default=0.0),
                "pruned_synapses": deepcopy(pruned_synapses),
            }
            committed_checkpoint_file = self._committed_checkpoint_path(checkpoint_file, "homeostatic_maintenance")
            maintenance["last_maintenance"]["committed_checkpoint_path"] = str(committed_checkpoint_file)
            recent_events = list(maintenance.get("recent_events") or [])
            recent_events.append(
                {
                    "event_index": int(maintenance["maintenance_count"]),
                    **deepcopy(maintenance["last_maintenance"]),
                }
            )
            maintenance["recent_events"] = recent_events[-8:]
            state["last_checkpoint_path"] = str(committed_checkpoint_file)
            self._runtime_state.mark_mutated()
            after_revision = int(self._runtime_state.state_revision)
            commit = self._commit_mutation(
                committed_checkpoint_file=committed_checkpoint_file,
                operation="homeostatic_maintenance",
                before_state=checkpoint_state,
                before_revision=before_revision,
                before_dirty_state=before_dirty_state,
            )
            if not commit["committed"]:
                return self._blocked_maintenance(
                    reason="post_maintenance_checkpoint_commit_failed",
                    before_revision=before_revision,
                    required_evidence={**required, **commit},
                )
            return {
                "artifact_kind": "terminus_snn_language_transition_memory_homeostatic_maintenance",
                "surface": "snn_language_transition_memory_homeostatic_maintenance.v1",
                "accepted": True,
                "status": "maintained",
                "owned_by_marulho": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "generates_text": False,
                "decodes_text": False,
                "trains_runtime_model": False,
                "applies_plasticity": True,
                "mutates_runtime_state": True,
                "checkpoint_transaction": {
                    "pre_maintenance_checkpoint_saved": True,
                    "checkpoint_path": str(checkpoint_file),
                    "committed_checkpoint_path": commit["committed_checkpoint_path"],
                    "staged_committed_checkpoint_path": str(committed_checkpoint_file),
                    "post_maintenance_checkpoint_saved": True,
                    "post_maintenance_checkpoint_restore_verified": True,
                    "current_checkpoint_manifest": commit["current_checkpoint_manifest"],
                    "restore_endpoint_available": True,
                    "restore_verified": checkpoint_verified,
                },
                "homeostatic_maintenance": deepcopy(maintenance["last_maintenance"]),
                "pruned_synapses": pruned_synapses,
                "before": {"state_revision": before_revision},
                "after": {"state_revision": after_revision, **self._runtime_state.mutation_summary()},
            }

    def regenerate_transition_memory(
        self,
        *,
        regeneration_proposal: Mapping[str, Any],
        expected_state_revision: int,
        operator_id: str,
        confirmation: bool,
        checkpoint_path: str | None = None,
        max_outgoing_row_mass: float = 1.0,
    ) -> dict[str, Any]:
        """Apply bounded replay-backed local transition regrowth."""

        with self._lock:
            before_revision = int(self._runtime_state.state_revision)
            proposal = dict(regeneration_proposal)
            gate = proposal.get("promotion_gate") if isinstance(proposal.get("promotion_gate"), Mapping) else {}
            design = proposal.get("regeneration_design") if isinstance(proposal.get("regeneration_design"), Mapping) else {}
            replay = proposal.get("replay_evidence") if isinstance(proposal.get("replay_evidence"), Mapping) else {}
            candidates = [dict(item) for item in list(design.get("candidate_synapses") or []) if isinstance(item, Mapping)]
            row_mass_limit = float(max_outgoing_row_mass)
            state = self._language_plasticity_state()
            capacity = self._language_capacity_state(state)
            weights = state.setdefault("sparse_transition_weights", {})
            topology = self._validate_structural_candidates(
                candidates=candidates,
                weights=weights,
                candidate_weight=lambda item: float(item.get("initial_weight", 0.0) or 0.0),
                row_mass_limit=row_mass_limit,
                language_neuron_count=int(capacity["language_neuron_count"]),
                sparse_edge_budget=int(capacity["sparse_edge_budget"]),
                outgoing_fanout_budget=int(capacity["outgoing_fanout_budget"]),
                locality_radius=int(design.get("locality_radius", 0) or 0),
                require_locality=True,
            )
            required = {
                "confirmation": bool(confirmation),
                "operator_id_available": bool(str(operator_id or "").strip()),
                "expected_revision_current": int(expected_state_revision) == before_revision,
                "proposal_available": bool(proposal.get("available")),
                "proposal_owned_by_marulho": bool(proposal.get("owned_by_marulho")),
                "proposal_gate_ready": str(gate.get("status") or "") == "ready_for_operator_review",
                "replay_evidence_available": bool(replay.get("available")),
                "replay_evidence_ready": bool(replay.get("ready")),
                "replay_evidence_source_available": bool(str(replay.get("source") or "").strip()),
                "replay_window_id_available": bool(str(replay.get("replay_window_id") or "").strip()),
                "replay_evidence_hash_available": bool(str(replay.get("evidence_hash") or "").strip()),
                "replay_permit_id_available": bool(str(replay.get("permit_id") or "").strip()),
                "replay_permit_server_verified": bool(self._verify_regeneration_permit(proposal)),
                "mismatch_score_high": float(design.get("mismatch_score", 0.0) or 0.0) >= 0.66,
                "candidate_synapses_available": bool(candidates),
                "no_text_generation": not bool(proposal.get("generates_text")),
                "no_external_checkpoint": not bool(proposal.get("loads_external_checkpoint")),
                **topology,
            }
            if not all(required.values()):
                return self._blocked_regeneration(
                    reason="blocked_missing_regeneration_evidence",
                    before_revision=before_revision,
                    required_evidence=required,
                )
            checkpoint_state = deepcopy(state)
            before_dirty_state = bool(self._runtime_state.dirty_state)
            checkpoint = self._save_checkpoint(checkpoint_path)
            checkpoint_file = Path(str(checkpoint.get("path") or self._checkpoint_path()))
            checkpoint_verified = self._verify_checkpoint_transaction(checkpoint_file, checkpoint_state, before_revision)
            if not checkpoint_verified:
                return self._blocked_regeneration(
                    reason="checkpoint_save_missing",
                    before_revision=before_revision,
                    required_evidence={
                        **required,
                        "pre_regeneration_checkpoint_saved": checkpoint_file.exists(),
                        "pre_regeneration_checkpoint_restore_verified": checkpoint_verified,
                    },
                )
            regenerated = []
            readout_evidence_hashes = [
                str(value)
                for value in list(replay.get("readout_evidence_hashes") or [])
                if str(value)
            ][:64]
            source_metadata_hash = replay.get("source_metadata_hash")
            emission_lineage = (
                dict(replay.get("emission_lineage"))
                if isinstance(replay.get("emission_lineage"), Mapping)
                else {}
            )
            for candidate in candidates:
                pre_index = int(candidate.get("pre_index", 0) or 0)
                post_index = int(candidate.get("post_index", 0) or 0)
                key = f"{pre_index}:{post_index}"
                if key in weights:
                    continue
                row_mass = self._row_mass_by_pre_index({str(k): float(v) for k, v in weights.items()}).get(str(pre_index), 0.0)
                weight = min(max(float(candidate.get("initial_weight", 0.0) or 0.0), 0.0), max(0.0, row_mass_limit - row_mass))
                if weight <= 0.0:
                    continue
                local_edge_provenance = {
                    "source_synapse_id": candidate.get("source_synapse_id"),
                    "source_trace_index": candidate.get("source_trace_index"),
                    "source_rollout_step_index": candidate.get("source_rollout_step_index"),
                    "target_rollout_step_index": candidate.get("target_rollout_step_index"),
                    "source_active_indices_hash": candidate.get("source_active_indices_hash"),
                    "target_active_indices_hash": candidate.get("target_active_indices_hash"),
                }
                weights[key] = weight
                regenerated.append(
                    {
                        "synapse": key,
                        "initial_weight": weight,
                        "locality_distance": candidate.get("locality_distance"),
                        "local_edge_provenance": local_edge_provenance,
                        "replay_provenance": {
                            "permit_id": replay.get("permit_id"),
                            "replay_artifact_id": replay.get("replay_artifact_id"),
                            "replay_artifact_hash": replay.get("replay_artifact_hash"),
                            "replay_window_hash": replay.get("replay_window_hash"),
                            "readout_evidence_hashes": deepcopy(readout_evidence_hashes),
                            "source_metadata_hash": source_metadata_hash,
                            "emission_lineage": deepcopy(emission_lineage),
                        },
                    }
                )
            if not regenerated:
                return self._blocked_regeneration(
                    reason="blocked_no_regenerable_synapses",
                    before_revision=before_revision,
                    required_evidence={**required, "regenerable_synapses_available": False},
                )
            ledger = state.setdefault("synapse_regeneration", {})
            ledger["regeneration_count"] = int(ledger.get("regeneration_count", 0) or 0) + 1
            ledger["regenerated_synapse_count_total"] = int(ledger.get("regenerated_synapse_count_total", 0) or 0) + len(regenerated)
            provenance_by_key = state.setdefault("synapse_provenance_by_key", {})
            for regenerated_synapse in regenerated:
                key = str(regenerated_synapse.get("synapse") or "")
                provenance_by_key[key] = {
                    "provenance_type": "replay_regeneration",
                    "permit_id": replay.get("permit_id"),
                    "replay_artifact_id": replay.get("replay_artifact_id"),
                    "replay_artifact_hash": replay.get("replay_artifact_hash"),
                    "replay_window_hash": replay.get("replay_window_hash"),
                    "readout_evidence_hashes": deepcopy(readout_evidence_hashes),
                    "source_metadata_hash": source_metadata_hash,
                    "emission_lineage": deepcopy(emission_lineage),
                    "local_edge_provenance": deepcopy(
                        regenerated_synapse.get("local_edge_provenance")
                        if isinstance(regenerated_synapse.get("local_edge_provenance"), Mapping)
                        else {}
                    ),
                }
            event = {
                "event_index": int(ledger["regeneration_count"]),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "operator_id": operator_id,
                "checkpoint_path": str(checkpoint_file),
                "replay_regeneration_permit": {
                    "permit_id": replay.get("permit_id"),
                    "evidence_hash": replay.get("evidence_hash"),
                    "issued_state_revision": replay.get("issued_state_revision"),
                    "mismatch_hash": replay.get("mismatch_hash"),
                    "mismatch_score": replay.get("mismatch_score"),
                    "pressure_hash": replay.get("pressure_hash"),
                    "pressure_score": replay.get("pressure_score"),
                    "replay_window_hash": replay.get("replay_window_hash"),
                    "readout_evidence_hashes": deepcopy(readout_evidence_hashes),
                    "replay_artifact_id": replay.get("replay_artifact_id"),
                    "replay_artifact_hash": replay.get("replay_artifact_hash"),
                    "source_metadata_hash": source_metadata_hash,
                    "emission_lineage": deepcopy(emission_lineage),
                    "regeneration_design_hash": replay.get("regeneration_design_hash"),
                    "regeneration_design_candidate_count": replay.get(
                        "regeneration_design_candidate_count"
                    ),
                },
                "regenerated_synapse_count": len(regenerated),
                "regenerated_synapses": deepcopy(regenerated),
                "max_outgoing_row_mass": row_mass_limit,
            }
            committed_checkpoint_file = self._committed_checkpoint_path(checkpoint_file, "regeneration")
            event["committed_checkpoint_path"] = str(committed_checkpoint_file)
            ledger["last_regeneration"] = event
            ledger["recent_events"] = [*list(ledger.get("recent_events") or []), deepcopy(event)][-8:]
            state["last_checkpoint_path"] = str(committed_checkpoint_file)
            self._runtime_state.mark_mutated()
            commit = self._commit_mutation(
                committed_checkpoint_file=committed_checkpoint_file,
                operation="regeneration",
                before_state=checkpoint_state,
                before_revision=before_revision,
                before_dirty_state=before_dirty_state,
            )
            if not commit["committed"]:
                return self._blocked_regeneration(
                    reason="post_regeneration_checkpoint_commit_failed",
                    before_revision=before_revision,
                    required_evidence={**required, **commit},
                )
            return {
                "artifact_kind": "terminus_snn_language_transition_memory_regeneration",
                "surface": "snn_language_transition_memory_regeneration.v1",
                "accepted": True,
                "status": "regenerated",
                "owned_by_marulho": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "generates_text": False,
                "decodes_text": False,
                "trains_runtime_model": False,
                "applies_plasticity": True,
                "mutates_runtime_state": True,
                "checkpoint_transaction": {
                    "pre_regeneration_checkpoint_saved": True,
                    "checkpoint_path": str(checkpoint_file),
                    "committed_checkpoint_path": commit["committed_checkpoint_path"],
                    "staged_committed_checkpoint_path": str(committed_checkpoint_file),
                    "post_regeneration_checkpoint_saved": True,
                    "post_regeneration_checkpoint_restore_verified": True,
                    "current_checkpoint_manifest": commit["current_checkpoint_manifest"],
                    "restore_verified": checkpoint_verified,
                },
                "regeneration": deepcopy(event),
                "before": {"state_revision": before_revision},
                "after": self._runtime_state.mutation_summary(),
            }

    def _preflight(
        self,
        *,
        readiness: Mapping[str, Any],
        delta: Mapping[str, Any],
        before_revision: int,
        expected_state_revision: int,
        operator_id: str,
        confirmation: bool,
    ) -> dict[str, Any]:
        gate = readiness.get("promotion_gate") if isinstance(readiness.get("promotion_gate"), Mapping) else {}
        rollback = readiness.get("rollback_readiness") if isinstance(readiness.get("rollback_readiness"), Mapping) else {}
        approval = readiness.get("operator_approval") if isinstance(readiness.get("operator_approval"), Mapping) else {}
        synapses = [item for item in list(delta.get("bounded_synapses") or []) if isinstance(item, Mapping)]
        max_delta = abs(float(delta.get("max_abs_weight_delta", 0.0) or 0.0))
        pressure_before = float(delta.get("pressure_before", 1.0) or 1.0)
        pressure_after = float(delta.get("pressure_after", pressure_before) or pressure_before)
        state = self._language_plasticity_state()
        capacity = self._language_capacity_state(state)
        topology = self._validate_structural_candidates(
            candidates=synapses,
            weights=state.setdefault("sparse_transition_weights", {}),
            candidate_weight=lambda _item: max_delta,
            row_mass_limit=_MAX_OUTGOING_ROW_MASS,
            language_neuron_count=int(capacity["language_neuron_count"]),
            sparse_edge_budget=int(capacity["sparse_edge_budget"]),
            outgoing_fanout_budget=int(capacity["outgoing_fanout_budget"]),
        )
        required = {
            "confirmation": bool(confirmation),
            "operator_id_available": bool(str(operator_id or "").strip()),
            "expected_revision_current": int(expected_state_revision) == before_revision,
            "readiness_available": bool(readiness.get("available")),
            "readiness_gate_ready": str(gate.get("status") or "") == "ready_for_operator_review",
            "operator_approval_available": bool(approval.get("approved")),
            "checkpoint_available": bool(rollback.get("checkpoint_available")),
            "restore_endpoint_available": bool(rollback.get("restore_endpoint_available")),
            "shadow_delta_available": bool(delta.get("available")),
            "bounded_synapses_available": bool(synapses),
            "max_delta_bounded": 0.0 < max_delta <= 0.25,
            "pressure_non_worsening": pressure_after <= pressure_before,
            "no_text_generation": not bool(readiness.get("generates_text")) and not bool(delta.get("generates_text")),
            "no_external_checkpoint": not bool(readiness.get("loads_external_checkpoint"))
            and not bool(delta.get("loads_external_checkpoint")),
            **topology,
        }
        if not all(required.values()):
            return self._blocked(
                reason="blocked_missing_live_application_evidence",
                before_revision=before_revision,
                required_evidence=required,
            )
        return {"accepted": True, "required_evidence": required}

    def _blocked(
        self,
        *,
        reason: str,
        before_revision: int,
        required_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "artifact_kind": "terminus_snn_language_plasticity_live_application",
            "surface": "snn_language_plasticity_live_application.v1",
            "accepted": False,
            "status": "blocked",
            "reason": reason,
            "owned_by_marulho": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "returns_trained_weights": False,
            "promotion_gate": {
                "status": "blocked_missing_live_application_evidence",
                "eligible_for_live_application": False,
                "required_evidence": dict(required_evidence),
            },
            "before": {"state_revision": before_revision},
            "after": {"state_revision": before_revision, **self._runtime_state.mutation_summary()},
        }

    @staticmethod
    def _row_mass_by_pre_index(weights: Mapping[str, float]) -> dict[str, float]:
        row_mass: dict[str, float] = {}
        for key, value in weights.items():
            pre_index = str(key).split(":", maxsplit=1)[0]
            row_mass[pre_index] = row_mass.get(pre_index, 0.0) + abs(float(value))
        return row_mass

    def _verify_checkpoint_transaction(
        self,
        path: Path,
        expected_language_state: Mapping[str, Any],
        expected_revision: int,
    ) -> bool:
        return bool(
            path.exists()
            and self._verify_checkpoint(path)
            and self._verify_checkpoint_snapshot(path, expected_language_state, expected_revision)
        )

    def _commit_mutation(
        self,
        *,
        committed_checkpoint_file: Path,
        operation: str,
        before_state: Mapping[str, Any],
        before_revision: int,
        before_dirty_state: bool,
    ) -> dict[str, Any]:
        after_state = deepcopy(self._language_plasticity_state())
        after_revision = int(self._runtime_state.state_revision)
        try:
            checkpoint = self._save_checkpoint(str(committed_checkpoint_file))
            saved_file = Path(str(checkpoint.get("path") or committed_checkpoint_file))
            verified = self._verify_checkpoint_transaction(saved_file, after_state, after_revision)
        except Exception:
            saved_file = committed_checkpoint_file
            verified = False
        if verified:
            try:
                manifest = dict(self._publish_committed_checkpoint(saved_file, operation))
            except Exception:
                manifest = {}
                verified = False
        if verified:
            return {
                "committed": True,
                "post_mutation_checkpoint_saved": True,
                "post_mutation_checkpoint_restore_verified": True,
                "committed_checkpoint_path": str(manifest.get("checkpoint_path") or saved_file),
                "staged_committed_checkpoint_path": str(saved_file),
                "current_checkpoint_manifest": manifest,
            }
        state = self._language_plasticity_state()
        state.clear()
        state.update(deepcopy(dict(before_state)))
        self._runtime_state.state_revision = before_revision
        self._runtime_state.dirty_state = before_dirty_state
        rollback_checkpoint_rewritten_verified = False
        try:
            rollback = self._save_checkpoint(str(saved_file))
            rollback_file = Path(str(rollback.get("path") or saved_file))
            rollback_checkpoint_rewritten_verified = self._verify_checkpoint_transaction(
                rollback_file,
                before_state,
                before_revision,
            )
        except Exception:
            rollback_checkpoint_rewritten_verified = False
        self._runtime_state.dirty_state = before_dirty_state
        return {
            "committed": False,
            "post_mutation_checkpoint_saved": saved_file.exists(),
            "post_mutation_checkpoint_restore_verified": False,
            "rollback_recovered_in_memory": True,
            "rollback_checkpoint_rewritten_verified": rollback_checkpoint_rewritten_verified,
            "committed_checkpoint_path": str(saved_file),
        }

    @staticmethod
    def _committed_checkpoint_path(rollback_checkpoint_file: Path, operation: str) -> Path:
        suffix = rollback_checkpoint_file.suffix or ".pt"
        return rollback_checkpoint_file.with_name(
            f"{rollback_checkpoint_file.stem}.{operation}.committed{suffix}"
        )

    @staticmethod
    def _shape_pair(value: Any) -> list[int]:
        try:
            raw = list(value or [])
            first = int(raw[0])
            second = int(raw[1])
        except (TypeError, ValueError, IndexError):
            return []
        if first <= 0 or second <= 0:
            return []
        return [first, second]

    @staticmethod
    def _sha256_json(value: Mapping[str, Any]) -> str:
        payload = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _resolve_tensor_device(requested_device: str | None) -> torch.device:
        if requested_device:
            return torch.device(str(requested_device))
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _dense_tensor_summary(value: Any) -> dict[str, Any]:
        if not isinstance(value, torch.Tensor):
            return {
                "available": False,
                "shape": [],
                "device": None,
                "is_cuda": False,
                "dtype": None,
                "nonzero_count": 0,
            }
        return {
            "available": True,
            "shape": [int(item) for item in list(value.shape)],
            "device": str(value.device),
            "is_cuda": bool(value.is_cuda),
            "dtype": str(value.dtype),
            "nonzero_count": int(torch.count_nonzero(value).item()),
        }

    @staticmethod
    def _newborn_integration_dense_samples(
        value: Any,
        event: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(value, torch.Tensor):
            return []
        samples: list[dict[str, Any]] = []
        for item in list(event.get("integrated_synapses") or [])[:64]:
            if not isinstance(item, Mapping):
                continue
            try:
                source_index = int(item.get("source_neuron_index", -1))
                target_index = int(item.get("target_neuron_index", -1))
                if (
                    source_index < 0
                    or target_index < 0
                    or source_index >= int(value.shape[0])
                    or target_index >= int(value.shape[1])
                ):
                    continue
                weight = float(value[source_index, target_index].item())
            except (IndexError, TypeError, ValueError):
                continue
            samples.append(
                {
                    "synapse": f"{source_index}:{target_index}",
                    "source_neuron_index": source_index,
                    "target_neuron_index": target_index,
                    "weight": weight,
                }
            )
        return samples

    @staticmethod
    def _critical_period_learning_dense_samples(
        value: Any,
        event: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(value, torch.Tensor):
            return []
        samples: list[dict[str, Any]] = []
        for item in list(event.get("applied_learning_cycles") or [])[:64]:
            if not isinstance(item, Mapping):
                continue
            try:
                source_index = int(item.get("source_neuron_index", -1))
                target_index = int(item.get("target_neuron_index", -1))
                if (
                    source_index < 0
                    or target_index < 0
                    or source_index >= int(value.shape[0])
                    or target_index >= int(value.shape[1])
                ):
                    continue
                weight = float(value[source_index, target_index].item())
            except (IndexError, TypeError, ValueError):
                continue
            samples.append(
                {
                    "synapse": f"{source_index}:{target_index}",
                    "source_neuron_index": source_index,
                    "target_neuron_index": target_index,
                    "weight": weight,
                }
            )
        return samples

    def _validate_structural_candidates(
        self,
        *,
        candidates: list[dict[str, Any]],
        weights: Mapping[str, Any],
        candidate_weight: Callable[[dict[str, Any]], float],
        row_mass_limit: float,
        language_neuron_count: int | None = None,
        sparse_edge_budget: int | None = None,
        outgoing_fanout_budget: int | None = None,
        locality_radius: int | None = None,
        require_locality: bool = False,
    ) -> dict[str, bool]:
        parsed = []
        for candidate in candidates:
            try:
                pre_index = int(candidate.get("pre_index"))
                post_index = int(candidate.get("post_index"))
                weight = float(candidate_weight(candidate))
            except (TypeError, ValueError):
                continue
            parsed.append((candidate, pre_index, post_index, weight))
        existing = {str(key): float(value) for key, value in dict(weights).items()}
        new_edges = {
            f"{pre_index}:{post_index}"
            for _candidate, pre_index, post_index, _weight in parsed
            if f"{pre_index}:{post_index}" not in existing
        }
        resulting = dict(existing)
        for _candidate, pre_index, post_index, weight in parsed:
            key = f"{pre_index}:{post_index}"
            resulting[key] = float(resulting.get(key, 0.0)) + weight
        fanout: dict[str, int] = {}
        for key in resulting:
            pre_index = key.split(":", maxsplit=1)[0]
            fanout[pre_index] = fanout.get(pre_index, 0) + 1
        radius = int(locality_radius or 0)
        neuron_count = max(
            _LANGUAGE_NEURON_COUNT,
            int(language_neuron_count or _LANGUAGE_NEURON_COUNT),
        )
        edge_budget = max(
            _MAX_SPARSE_TRANSITION_EDGES,
            int(sparse_edge_budget or _MAX_SPARSE_TRANSITION_EDGES),
        )
        fanout_budget = max(
            _MAX_OUTGOING_FANOUT,
            int(outgoing_fanout_budget or _MAX_OUTGOING_FANOUT),
        )
        return {
            "candidate_payload_well_formed": len(parsed) == len(candidates),
            "candidate_count_bounded": 0 < len(candidates) <= _MAX_STRUCTURAL_EDGES_PER_EVENT,
            "candidate_indices_canonical": all(
                0 <= pre_index < neuron_count and 0 <= post_index < neuron_count
                for _candidate, pre_index, post_index, _weight in parsed
            ),
            "candidate_weights_bounded": all(0.0 < weight <= 0.25 for _candidate, _pre, _post, weight in parsed),
            "locality_radius_bounded": not require_locality or 0 < radius <= 8,
            "candidate_synapses_local": not require_locality
            or all(
                abs(post_index - pre_index) <= radius
                and int(candidate.get("locality_distance", -1)) == abs(post_index - pre_index)
                for candidate, pre_index, post_index, _weight in parsed
            ),
            "outgoing_row_mass_bounded": 0.0 < row_mass_limit <= _MAX_OUTGOING_ROW_MASS
            and max(self._row_mass_by_pre_index(resulting).values(), default=0.0) <= row_mass_limit,
            "outgoing_fanout_bounded": max(fanout.values(), default=0) <= fanout_budget,
            "global_sparse_edge_budget_bounded": len(existing) + len(new_edges) <= edge_budget,
            "language_capacity_state_available": True,
            "language_capacity_state_dynamic_limits_applied": True,
        }

    def _blocked_maintenance(
        self,
        *,
        reason: str,
        before_revision: int,
        required_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "artifact_kind": "terminus_snn_language_transition_memory_homeostatic_maintenance",
            "surface": "snn_language_transition_memory_homeostatic_maintenance.v1",
            "accepted": False,
            "status": "blocked",
            "reason": reason,
            "owned_by_marulho": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "promotion_gate": {
                "status": "blocked_missing_homeostatic_maintenance_evidence",
                "required_evidence": dict(required_evidence),
            },
            "before": {"state_revision": before_revision},
            "after": {"state_revision": before_revision, **self._runtime_state.mutation_summary()},
        }

    def _blocked_regeneration(
        self,
        *,
        reason: str,
        before_revision: int,
        required_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "artifact_kind": "terminus_snn_language_transition_memory_regeneration",
            "surface": "snn_language_transition_memory_regeneration.v1",
            "accepted": False,
            "status": "blocked",
            "reason": reason,
            "owned_by_marulho": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "promotion_gate": {
                "status": "blocked_missing_regeneration_evidence",
                "required_evidence": dict(required_evidence),
            },
            "before": {"state_revision": before_revision},
            "after": {"state_revision": before_revision, **self._runtime_state.mutation_summary()},
        }

    def _blocked_dense_layout_migration(
        self,
        *,
        reason: str,
        before_revision: int,
        required_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "artifact_kind": "terminus_snn_language_dense_readout_layout_migration",
            "surface": "snn_language_dense_readout_layout_migration.v1",
            "accepted": False,
            "status": "blocked",
            "reason": reason,
            "owned_by_marulho": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "writes_checkpoint": False,
            "resizes_network": False,
            "materializes_dense_tensor_weights": False,
            "promotion_gate": {
                "status": "blocked_missing_dense_readout_layout_migration_evidence",
                "eligible_for_dense_readout_layout_migration": False,
                "eligible_for_dense_readout_tensor_materialization": False,
                "required_evidence": dict(required_evidence),
            },
            "before": {"state_revision": before_revision},
            "after": {
                "state_revision": before_revision,
                **self._runtime_state.mutation_summary(),
            },
        }

    def _blocked_dense_tensor_materialization(
        self,
        *,
        reason: str,
        before_revision: int,
        required_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "artifact_kind": "terminus_snn_language_dense_readout_tensor_materialization",
            "surface": "snn_language_dense_readout_tensor_materialization.v1",
            "accepted": False,
            "status": "blocked",
            "reason": reason,
            "owned_by_marulho": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "writes_checkpoint": False,
            "resizes_network": False,
            "materializes_dense_tensor_weights": False,
            "promotion_gate": {
                "status": "blocked_missing_dense_readout_tensor_materialization_evidence",
                "eligible_for_dense_readout_tensor_materialization": False,
                "eligible_for_network_resize": False,
                "eligible_for_language_generation": False,
                "required_evidence": dict(required_evidence),
            },
            "before": {"state_revision": before_revision},
            "after": {
                "state_revision": before_revision,
                **self._runtime_state.mutation_summary(),
            },
        }

    def _blocked_thought_capacity_mutation(
        self,
        *,
        reason: str,
        before_revision: int,
        required_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "artifact_kind": (
                "terminus_snn_language_autonomous_snn_language_thought_"
                "capacity_mutation_executor"
            ),
            "surface": (
                "snn_language_autonomous_snn_language_thought_"
                "capacity_mutation_executor.v1"
            ),
            "accepted": False,
            "ready": False,
            "status": "blocked",
            "reason": reason,
            "requires_operator_approval": False,
            "owned_by_marulho": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "advisory": False,
            "executable": True,
            "records_ledger_event": False,
            "generates_text": False,
            "decodes_text": False,
            "runs_replay": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "writes_checkpoint": False,
            "resizes_network": False,
            "adds_neurons": False,
            "adds_synapses": False,
            "prunes_network": False,
            "promotion_gate": {
                "status": (
                    "blocked_missing_autonomous_snn_language_thought_"
                    "capacity_mutation_executor_evidence"
                ),
                "eligible_for_autonomous_snn_language_thought_"
                "capacity_mutation_event_review": False,
                "eligible_for_language_generation": False,
                "eligible_for_replay_memory": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "eligible_for_cognition_substrate": False,
                "required_evidence": dict(required_evidence),
            },
            "before": {"state_revision": before_revision},
            "after": {
                "state_revision": before_revision,
                **self._runtime_state.mutation_summary(),
            },
        }

    def _blocked_thought_newborn_neuron_integration(
        self,
        *,
        reason: str,
        before_revision: int,
        required_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "artifact_kind": (
                "terminus_snn_language_autonomous_snn_language_thought_"
                "newborn_neuron_integration_executor"
            ),
            "surface": (
                "snn_language_autonomous_snn_language_thought_"
                "newborn_neuron_integration_executor.v1"
            ),
            "accepted": False,
            "ready": False,
            "status": "blocked",
            "reason": reason,
            "requires_operator_approval": False,
            "owned_by_marulho": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "advisory": False,
            "executable": True,
            "records_ledger_event": False,
            "generates_text": False,
            "decodes_text": False,
            "runs_replay": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "writes_checkpoint": False,
            "resizes_network": False,
            "adds_neurons": False,
            "adds_synapses": False,
            "prunes_network": False,
            "promotion_gate": {
                "status": (
                    "blocked_missing_autonomous_snn_language_thought_"
                    "newborn_neuron_integration_executor_evidence"
                ),
                "eligible_for_autonomous_snn_language_thought_"
                "newborn_neuron_integration_event_review": False,
                "eligible_for_language_generation": False,
                "eligible_for_replay_memory": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "eligible_for_cognition_substrate": False,
                "required_evidence": dict(required_evidence),
            },
            "before": {"state_revision": before_revision},
            "after": {
                "state_revision": before_revision,
                **self._runtime_state.mutation_summary(),
            },
        }

    def _blocked_thought_newborn_neuron_critical_period_learning(
        self,
        *,
        reason: str,
        before_revision: int,
        required_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "artifact_kind": (
                "terminus_snn_language_autonomous_snn_language_thought_"
                "newborn_neuron_critical_period_learning_executor"
            ),
            "surface": (
                "snn_language_autonomous_snn_language_thought_"
                "newborn_neuron_critical_period_learning_executor.v1"
            ),
            "accepted": False,
            "ready": False,
            "status": "blocked",
            "reason": reason,
            "requires_operator_approval": False,
            "owned_by_marulho": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "advisory": False,
            "executable": True,
            "records_ledger_event": False,
            "generates_text": False,
            "decodes_text": False,
            "runs_replay": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "writes_checkpoint": False,
            "resizes_network": False,
            "adds_neurons": False,
            "adds_synapses": False,
            "prunes_network": False,
            "promotion_gate": {
                "status": (
                    "blocked_missing_autonomous_snn_language_thought_"
                    "newborn_neuron_critical_period_learning_executor_"
                    "evidence"
                ),
                "eligible_for_autonomous_snn_language_thought_newborn_"
                "neuron_critical_period_learning_event_review": False,
                "eligible_for_language_generation": False,
                "eligible_for_replay_memory": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "eligible_for_cognition_substrate": False,
                "required_evidence": dict(required_evidence),
            },
            "before": {"state_revision": before_revision},
            "after": {
                "state_revision": before_revision,
                **self._runtime_state.mutation_summary(),
            },
        }

    def _blocked_thought_newborn_synapse_pruning(
        self,
        *,
        reason: str,
        before_revision: int,
        required_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "artifact_kind": (
                "terminus_snn_language_autonomous_snn_language_thought_"
                "newborn_synapse_pruning_executor"
            ),
            "surface": (
                "snn_language_autonomous_snn_language_thought_"
                "newborn_synapse_pruning_executor.v1"
            ),
            "accepted": False,
            "ready": False,
            "status": "blocked",
            "reason": reason,
            "requires_operator_approval": False,
            "advisory": False,
            "executable": True,
            "records_ledger_event": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "writes_checkpoint": False,
            "resizes_network": False,
            "adds_neurons": False,
            "adds_synapses": False,
            "prunes_network": False,
            "promotion_gate": {
                "status": (
                    "blocked_missing_autonomous_snn_language_thought_"
                    "newborn_synapse_pruning_executor_evidence"
                ),
                "eligible_for_autonomous_snn_language_thought_newborn_"
                "synapse_pruning_event_review": False,
                "required_evidence": dict(required_evidence),
            },
            "before": {"state_revision": before_revision},
            "after": {
                "state_revision": before_revision,
                **self._runtime_state.mutation_summary(),
            },
        }

    def _blocked_dense_readout_training(
        self,
        *,
        reason: str,
        before_revision: int,
        required_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "artifact_kind": "terminus_snn_language_dense_readout_training",
            "surface": "snn_language_dense_readout_training.v1",
            "accepted": False,
            "status": "blocked",
            "reason": reason,
            "owned_by_marulho": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "writes_checkpoint": False,
            "resizes_network": False,
            "returns_trained_weights": False,
            "promotion_gate": {
                "status": "blocked_missing_dense_readout_training_evidence",
                "eligible_for_dense_readout_training": False,
                "eligible_for_language_generation": False,
                "required_evidence": dict(required_evidence),
            },
            "before": {"state_revision": before_revision},
            "after": {
                "state_revision": before_revision,
                **self._runtime_state.mutation_summary(),
            },
        }
