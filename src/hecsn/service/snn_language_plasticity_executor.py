from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from hecsn.service.runtime_state import RuntimeState

_LANGUAGE_NEURON_COUNT = 64
_MAX_STRUCTURAL_EDGES_PER_EVENT = 32
_MAX_OUTGOING_FANOUT = 16
_MAX_SPARSE_TRANSITION_EDGES = 256
_MAX_OUTGOING_ROW_MASS = 1.0


class SNNLanguagePlasticityApplicationExecutor:
    """Checkpoint-backed mutation boundary for HECSN-owned SNN language plasticity."""

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
                "owned_by_hecsn": True,
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
                    "target_id": "hecsn.snn_language.sparse_transition_weights",
                    "owned_by_hecsn": True,
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

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = deepcopy(self._language_plasticity_state())
            weights = dict(state.get("sparse_transition_weights") or {})
            maintenance = dict(state.get("homeostatic_maintenance") or {})
            regeneration = dict(state.get("synapse_regeneration") or {})
            live_application = dict(state.get("live_application") or {})
            return {
                "surface": "snn_language_plasticity_runtime_state.v1",
                "owned_by_hecsn": True,
                "external_dependency": False,
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
                "last_applied_at": state.get("last_applied_at"),
                "last_operator_id": state.get("last_operator_id"),
                "last_checkpoint_path": state.get("last_checkpoint_path"),
            }

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
                "owned_by_hecsn": True,
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
            weights = state.setdefault("sparse_transition_weights", {})
            topology = self._validate_structural_candidates(
                candidates=candidates,
                weights=weights,
                candidate_weight=lambda item: float(item.get("initial_weight", 0.0) or 0.0),
                row_mass_limit=row_mass_limit,
                locality_radius=int(design.get("locality_radius", 0) or 0),
                require_locality=True,
            )
            required = {
                "confirmation": bool(confirmation),
                "operator_id_available": bool(str(operator_id or "").strip()),
                "expected_revision_current": int(expected_state_revision) == before_revision,
                "proposal_available": bool(proposal.get("available")),
                "proposal_owned_by_hecsn": bool(proposal.get("owned_by_hecsn")),
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
                weights[key] = weight
                regenerated.append(
                    {
                        "synapse": key,
                        "initial_weight": weight,
                        "locality_distance": candidate.get("locality_distance"),
                        "replay_provenance": {
                            "permit_id": replay.get("permit_id"),
                            "replay_artifact_id": replay.get("replay_artifact_id"),
                            "replay_artifact_hash": replay.get("replay_artifact_hash"),
                            "replay_window_hash": replay.get("replay_window_hash"),
                            "readout_evidence_hashes": deepcopy(readout_evidence_hashes),
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
                    "pressure_hash": replay.get("pressure_hash"),
                    "replay_window_hash": replay.get("replay_window_hash"),
                    "readout_evidence_hashes": deepcopy(readout_evidence_hashes),
                    "replay_artifact_id": replay.get("replay_artifact_id"),
                    "replay_artifact_hash": replay.get("replay_artifact_hash"),
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
                "owned_by_hecsn": True,
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
        topology = self._validate_structural_candidates(
            candidates=synapses,
            weights=self._language_plasticity_state().setdefault("sparse_transition_weights", {}),
            candidate_weight=lambda _item: max_delta,
            row_mass_limit=_MAX_OUTGOING_ROW_MASS,
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
            "owned_by_hecsn": True,
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

    def _validate_structural_candidates(
        self,
        *,
        candidates: list[dict[str, Any]],
        weights: Mapping[str, Any],
        candidate_weight: Callable[[dict[str, Any]], float],
        row_mass_limit: float,
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
        return {
            "candidate_payload_well_formed": len(parsed) == len(candidates),
            "candidate_count_bounded": 0 < len(candidates) <= _MAX_STRUCTURAL_EDGES_PER_EVENT,
            "candidate_indices_canonical": all(
                0 <= pre_index < _LANGUAGE_NEURON_COUNT and 0 <= post_index < _LANGUAGE_NEURON_COUNT
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
            "outgoing_fanout_bounded": max(fanout.values(), default=0) <= _MAX_OUTGOING_FANOUT,
            "global_sparse_edge_budget_bounded": len(existing) + len(new_edges) <= _MAX_SPARSE_TRANSITION_EDGES,
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
            "owned_by_hecsn": True,
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
            "owned_by_hecsn": True,
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
