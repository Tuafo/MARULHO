from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from hecsn.semantics import ConceptStore
from hecsn.service.runtime_state import RuntimeState


class StructuralMutationExecutor:
    """Checkpoint-backed mutation boundary for HECSN-owned structural plasticity."""

    def __init__(
        self,
        *,
        lock: Any,
        runtime_state: RuntimeState,
        concept_store: Callable[[], ConceptStore],
        save_checkpoint: Callable[[str | None], dict[str, Any]],
        checkpoint_path: Callable[[], Path],
        verify_checkpoint: Callable[[Path], bool],
        verify_checkpoint_snapshot: Callable[[Path, Mapping[str, Any], int], bool] | None = None,
        publish_committed_checkpoint: Callable[[Path, str], Mapping[str, Any]] | None = None,
    ) -> None:
        self._lock = lock
        self._runtime_state = runtime_state
        self._concept_store = concept_store
        self._save_checkpoint = save_checkpoint
        self._checkpoint_path = checkpoint_path
        self._verify_checkpoint = verify_checkpoint
        self._verify_checkpoint_snapshot = verify_checkpoint_snapshot or (lambda _path, _state, _revision: True)
        self._publish_committed_checkpoint = publish_committed_checkpoint or (lambda path, operation: {"path": str(path), "operation": operation})

    def apply_subcortical_structural_mutation(
        self,
        *,
        structural_mutation_preflight: Mapping[str, Any],
        expected_state_revision: int,
        operator_id: str,
        confirmation: bool,
        checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            before_revision = int(self._runtime_state.state_revision)
            preflight = dict(structural_mutation_preflight)
            gate = preflight.get("promotion_gate") if isinstance(preflight.get("promotion_gate"), Mapping) else {}
            design = preflight.get("design_binding") if isinstance(preflight.get("design_binding"), Mapping) else {}
            preflight_checkpoint = str(preflight.get("checkpoint_path") or "").strip()
            requested_checkpoint = str(checkpoint_path or "").strip()
            effective_checkpoint = requested_checkpoint or preflight_checkpoint
            required = {
                "preflight_surface_available": preflight.get("surface")
                == "subcortical_structural_mutation_preflight.v1",
                "preflight_ready": bool(preflight.get("ready")),
                "preflight_gate_ready": bool(gate.get("eligible_for_operator_execution_review")),
                "preflight_blocks_direct_mutation": not bool(gate.get("eligible_for_structural_mutation")),
                "preflight_did_not_write_checkpoint": not bool(preflight.get("writes_checkpoint")),
                "preflight_did_not_mutate_runtime": not bool(preflight.get("mutates_runtime_state")),
                "preflight_did_not_call_growth_or_prune": not bool(preflight.get("calls_growth_or_prune")),
                "design_hash_bound": bool(design.get("design_hash_available"))
                and bool(design.get("design_hash_recomputed_match")),
                "expected_revision_current": int(expected_state_revision) == before_revision,
                "expected_revision_matches_preflight": int(preflight.get("expected_state_revision", -1))
                == int(expected_state_revision),
                "checkpoint_path_available": bool(effective_checkpoint),
                "checkpoint_path_matches_preflight": not bool(requested_checkpoint and preflight_checkpoint)
                or requested_checkpoint == preflight_checkpoint,
                "operator_id_available": bool(str(operator_id or "").strip()),
                "confirmation": bool(confirmation),
            }
            if not all(required.values()):
                return self._blocked(
                    reason="blocked_missing_structural_mutation_application_evidence",
                    before_revision=before_revision,
                    required_evidence=required,
                    checkpoint_path=effective_checkpoint or None,
                )

            store = self._concept_store()
            before_state = deepcopy(store.state_dict())
            before_dirty_state = bool(self._runtime_state.dirty_state)
            checkpoint = self._save_checkpoint(effective_checkpoint)
            checkpoint_file = Path(str(checkpoint.get("path") or self._checkpoint_path()))
            checkpoint_verified = self._verify_checkpoint_transaction(checkpoint_file, before_state, before_revision)
            if not checkpoint_verified:
                return self._blocked(
                    reason="pre_structural_mutation_checkpoint_unverified",
                    before_revision=before_revision,
                    required_evidence={
                        **required,
                        "pre_structural_mutation_checkpoint_saved": checkpoint_file.exists(),
                        "pre_structural_mutation_checkpoint_restore_verified": checkpoint_verified,
                    },
                    checkpoint_path=str(checkpoint_file),
                )

            report = store.refresh_structural_capacity()
            after_refresh_state = deepcopy(store.state_dict())
            if dict(after_refresh_state) == dict(before_state):
                store.load_state_dict(dict(before_state))
                return self._blocked(
                    reason="blocked_no_structural_capacity_delta",
                    before_revision=before_revision,
                    required_evidence={
                        **required,
                        "pre_structural_mutation_checkpoint_saved": True,
                        "pre_structural_mutation_checkpoint_restore_verified": checkpoint_verified,
                        "structural_capacity_delta_observed": False,
                    },
                    checkpoint_path=str(checkpoint_file),
                )
            committed_checkpoint_file = self._committed_checkpoint_path(checkpoint_file, "subcortical_structural_mutation")
            event = {
                "operator_id": str(operator_id),
                "applied_at": datetime.now(timezone.utc).isoformat(),
                "before_state_revision": before_revision,
                "after_state_revision": before_revision + 1,
                "checkpoint_path": str(checkpoint_file),
                "staged_committed_checkpoint_path": str(committed_checkpoint_file),
                "structural_mutation_design_hash": design.get("structural_mutation_design_hash"),
                "structural_report": deepcopy(report),
            }
            self._runtime_state.mark_mutated()
            after_revision = int(self._runtime_state.state_revision)
            commit = self._commit_mutation(
                committed_checkpoint_file=committed_checkpoint_file,
                operation="subcortical_structural_mutation",
                before_state=before_state,
                before_revision=before_revision,
                before_dirty_state=before_dirty_state,
            )
            if not commit["committed"]:
                return self._blocked(
                    reason="post_structural_mutation_checkpoint_commit_failed",
                    before_revision=before_revision,
                    required_evidence={**required, **commit},
                    checkpoint_path=str(checkpoint_file),
                )
            return {
                "artifact_kind": "terminus_subcortical_structural_mutation_application",
                "surface": "subcortical_structural_mutation_application.v1",
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
                "writes_checkpoint": True,
                "calls_growth_or_prune": True,
                "applies_structural_mutation": True,
                "operator_id": str(operator_id),
                "checkpoint_transaction": {
                    "pre_structural_mutation_checkpoint_saved": True,
                    "checkpoint_path": str(checkpoint_file),
                    "committed_checkpoint_path": commit["committed_checkpoint_path"],
                    "staged_committed_checkpoint_path": str(committed_checkpoint_file),
                    "post_structural_mutation_checkpoint_saved": True,
                    "post_structural_mutation_checkpoint_restore_verified": True,
                    "current_checkpoint_manifest": commit["current_checkpoint_manifest"],
                    "restore_endpoint_available": True,
                    "restore_verified": checkpoint_verified,
                },
                "application_target": {
                    "target_id": "hecsn.subcortex.concept_store.structural_capacity",
                    "owned_by_hecsn": True,
                    "checkpointed": True,
                    "mutation_method": "ConceptStore.refresh_structural_capacity",
                },
                "structural_mutation_event": event,
                "structural_report": deepcopy(report),
                "before": {"state_revision": before_revision},
                "after": {"state_revision": after_revision, **self._runtime_state.mutation_summary()},
                "promotion_gate": {
                    "status": "checkpoint_backed_structural_mutation_applied",
                    "eligible_for_structural_mutation": False,
                    "eligible_for_action": False,
                    "required_evidence": required,
                },
            }

    def _blocked(
        self,
        *,
        reason: str,
        before_revision: int,
        required_evidence: Mapping[str, Any],
        checkpoint_path: str | None,
    ) -> dict[str, Any]:
        return {
            "artifact_kind": "terminus_subcortical_structural_mutation_application",
            "surface": "subcortical_structural_mutation_application.v1",
            "accepted": False,
            "available": True,
            "status": reason,
            "reason": reason,
            "owned_by_hecsn": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "writes_checkpoint": False,
            "calls_growth_or_prune": False,
            "applies_structural_mutation": False,
            "checkpoint_path": checkpoint_path,
            "before": {"state_revision": before_revision},
            "after": {"state_revision": int(self._runtime_state.state_revision), **self._runtime_state.mutation_summary()},
            "promotion_gate": {
                "status": reason,
                "eligible_for_structural_mutation": False,
                "eligible_for_action": False,
                "required_evidence": dict(required_evidence),
            },
        }

    def _verify_checkpoint_transaction(
        self,
        path: Path,
        expected_concept_state: Mapping[str, Any],
        expected_revision: int,
    ) -> bool:
        return bool(
            path.exists()
            and self._verify_checkpoint(path)
            and self._verify_checkpoint_snapshot(path, expected_concept_state, expected_revision)
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
        after_state = deepcopy(self._concept_store().state_dict())
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
        self._concept_store().load_state_dict(dict(before_state))
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
