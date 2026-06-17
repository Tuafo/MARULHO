from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from marulho.service.runtime_state import RuntimeState


class StructuralMutationExecutor:
    """Checkpoint-backed mutation boundary for MARULHO-owned structural plasticity."""

    def __init__(
        self,
        *,
        lock: Any,
        runtime_state: RuntimeState,
        binding_layer: Callable[[], Any | None],
        save_checkpoint: Callable[[str | None], dict[str, Any]],
        checkpoint_path: Callable[[], Path],
        verify_checkpoint: Callable[[Path], bool],
        verify_checkpoint_snapshot: Callable[[Path, Mapping[str, Any], int], bool] | None = None,
        publish_committed_checkpoint: Callable[[Path, str], Mapping[str, Any]] | None = None,
    ) -> None:
        self._lock = lock
        self._runtime_state = runtime_state
        self._binding_layer = binding_layer
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
            transaction = (
                preflight.get("checkpoint_transaction_requirements")
                if isinstance(preflight.get("checkpoint_transaction_requirements"), Mapping)
                else {}
            )
            gate_evidence = (
                gate.get("required_evidence")
                if isinstance(gate.get("required_evidence"), Mapping)
                else {}
            )
            preflight_checkpoint = str(transaction.get("checkpoint_path") or "").strip()
            requested_checkpoint = str(checkpoint_path or "").strip()
            effective_checkpoint = requested_checkpoint or preflight_checkpoint
            preflight_material = {
                "structural_mutation_design_hash": design.get("structural_mutation_design_hash"),
                "mutation_target": design.get("mutation_target"),
                "mutation_method": design.get("mutation_method"),
                "mutation_reason": str(design.get("mutation_reason") or "").strip(),
                "max_total_edge_delta": int(design.get("max_total_edge_delta", 0)),
                "checkpointed_candidate_gate_status": design.get(
                    "checkpointed_candidate_gate_status"
                ),
                "candidate_evidence_hash": design.get("candidate_evidence_hash"),
                "candidate_baseline_hash": design.get("candidate_baseline_hash"),
                "candidate_reason": design.get("candidate_reason"),
                "cost_impact_summary_hash": design.get("cost_impact_summary_hash"),
                "expected_state_revision": int(transaction.get("expected_state_revision", -1)),
                "current_state_revision": int(transaction.get("current_state_revision", -1)),
                "checkpoint_path": preflight_checkpoint or None,
                "required_evidence": dict(gate_evidence),
            }
            recomputed_preflight_hash = self._sha256_json(preflight_material)
            candidate_provenance = self._candidate_provenance(
                preflight=preflight,
                design=design,
                recomputed_preflight_hash=recomputed_preflight_hash,
            )
            required = {
                "preflight_surface_available": preflight.get("surface")
                == "subcortical_structural_mutation_preflight.v1",
                "preflight_hash_available": len(
                    str(preflight.get("structural_mutation_preflight_hash") or "")
                )
                == 64,
                "preflight_hash_recomputed_match": recomputed_preflight_hash
                == str(preflight.get("structural_mutation_preflight_hash") or ""),
                "preflight_gate_ready": bool(gate.get("eligible_for_operator_execution_review")),
                "preflight_blocks_direct_mutation": not bool(gate.get("eligible_for_structural_mutation")),
                "preflight_did_not_write_checkpoint": not bool(preflight.get("writes_checkpoint")),
                "preflight_did_not_mutate_runtime": not bool(preflight.get("mutates_runtime_state")),
                "preflight_did_not_call_growth_or_prune": not bool(preflight.get("calls_growth_or_prune")),
                "design_hash_bound": bool(design.get("design_hash_available"))
                and bool(design.get("design_hash_recomputed_match")),
                "binding_hub_topology_target_bound": design.get("mutation_target")
                == "marulho.subcortex.binding.hub_topology",
                "binding_hub_refresh_method_bound": design.get("mutation_method")
                == "HypercubeBindingLayer.refresh_hub_topology",
                "mutation_reason_available": bool(str(design.get("mutation_reason") or "").strip()),
                "positive_edge_delta_budget": int(design.get("max_total_edge_delta", 0)) > 0,
                "expected_revision_current": int(expected_state_revision) == before_revision,
                "expected_revision_matches_preflight": int(
                    transaction.get("expected_state_revision", -1)
                )
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
                    candidate_provenance=candidate_provenance,
                )

            binding = self._binding_layer()
            refresh = getattr(binding, "refresh_hub_topology", None)
            if binding is None or not callable(refresh):
                return self._blocked(
                    reason="blocked_binding_hub_topology_unavailable",
                    before_revision=before_revision,
                    required_evidence={
                        **required,
                        "binding_layer_available": binding is not None,
                        "binding_hub_refresh_available": callable(refresh),
                    },
                    checkpoint_path=effective_checkpoint or None,
                    candidate_provenance=candidate_provenance,
                )

            before_state = deepcopy(binding.state_dict())
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
                    candidate_provenance=candidate_provenance,
                )

            report = refresh(reason=str(design["mutation_reason"]))
            if not bool(report.get("topology_changed")):
                binding.load_state_dict(dict(before_state))
                return self._blocked(
                    reason="blocked_no_binding_hub_topology_delta",
                    before_revision=before_revision,
                    required_evidence={
                        **required,
                        "pre_structural_mutation_checkpoint_saved": True,
                        "pre_structural_mutation_checkpoint_restore_verified": checkpoint_verified,
                        "binding_hub_topology_delta_observed": False,
                    },
                    checkpoint_path=str(checkpoint_file),
                    candidate_provenance=candidate_provenance,
                    rollback_artifact=self._binding_rollback_artifact(
                        scope="binding_hub_topology_no_delta_rejection",
                        checkpoint_path=str(checkpoint_file),
                        before_revision=before_revision,
                        before_state=before_state,
                        current_state=binding.state_dict(),
                        checkpoint_verified=checkpoint_verified,
                    ),
                )
            structural_delta = self._structural_delta(report)
            total_edge_delta = abs(structural_delta["edges_added_delta"]) + abs(
                structural_delta["edges_removed_delta"]
            )
            max_total_edge_delta = int(design["max_total_edge_delta"])
            if total_edge_delta > max_total_edge_delta:
                binding.load_state_dict(dict(before_state))
                return self._blocked(
                    reason="blocked_binding_hub_topology_delta_over_budget",
                    before_revision=before_revision,
                    required_evidence={
                        **required,
                        "binding_hub_topology_delta_observed": True,
                        "actual_total_edge_delta": total_edge_delta,
                        "max_total_edge_delta": max_total_edge_delta,
                        "actual_edge_delta_within_budget": False,
                    },
                    checkpoint_path=str(checkpoint_file),
                    candidate_provenance=candidate_provenance,
                    rollback_artifact=self._binding_rollback_artifact(
                        scope="binding_hub_topology_over_budget_rejection",
                        checkpoint_path=str(checkpoint_file),
                        before_revision=before_revision,
                        before_state=before_state,
                        current_state=binding.state_dict(),
                        checkpoint_verified=checkpoint_verified,
                    ),
                )
            committed_checkpoint_file = self._committed_checkpoint_path(
                checkpoint_file,
                "binding_hub_topology_mutation",
            )
            event = {
                "operator_id": str(operator_id),
                "applied_at": datetime.now(timezone.utc).isoformat(),
                "before_state_revision": before_revision,
                "after_state_revision": before_revision + 1,
                "checkpoint_path": str(checkpoint_file),
                "staged_committed_checkpoint_path": str(committed_checkpoint_file),
                "structural_mutation_design_hash": design.get("structural_mutation_design_hash"),
                "mutation_reason": design.get("mutation_reason"),
                "structural_delta": structural_delta,
                "structural_report": deepcopy(report),
            }
            self._runtime_state.mark_mutated()
            after_revision = int(self._runtime_state.state_revision)
            commit = self._commit_mutation(
                committed_checkpoint_file=committed_checkpoint_file,
                operation="binding_hub_topology_mutation",
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
                    candidate_provenance=candidate_provenance,
                    rollback_artifact=self._commit_failure_rollback_artifact(
                        commit=commit,
                        checkpoint_path=str(checkpoint_file),
                        before_revision=before_revision,
                    ),
                )
            return {
                "artifact_kind": "terminus_subcortical_structural_mutation_application",
                "surface": "subcortical_structural_mutation_application.v1",
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
                    "target_id": "marulho.subcortex.binding.hub_topology",
                    "owned_by_marulho": True,
                    "checkpointed": True,
                    "mutation_method": "HypercubeBindingLayer.refresh_hub_topology",
                },
                "candidate_provenance": candidate_provenance,
                "structural_mutation_event": event,
                "structural_delta": structural_delta,
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
        candidate_provenance: Mapping[str, Any] | None = None,
        rollback_artifact: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        provenance = dict(candidate_provenance or {})
        rollback = self._normalize_rollback_artifact(
            rollback_artifact,
            checkpoint_path=checkpoint_path,
            before_revision=before_revision,
        )
        tombstone = self._candidate_tombstone_manifest(
            reason=reason,
            before_revision=before_revision,
            checkpoint_path=checkpoint_path,
            candidate_provenance=provenance,
            required_evidence=required_evidence,
            rollback_artifact=rollback,
        )
        return {
            "artifact_kind": "terminus_subcortical_structural_mutation_application",
            "surface": "subcortical_structural_mutation_application.v1",
            "accepted": False,
            "available": True,
            "status": reason,
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
            "calls_growth_or_prune": False,
            "applies_structural_mutation": False,
            "checkpoint_path": checkpoint_path,
            "candidate_provenance": provenance,
            "rollback_artifact": rollback,
            "tombstone_manifest": tombstone,
            "before": {"state_revision": before_revision},
            "after": {"state_revision": int(self._runtime_state.state_revision), **self._runtime_state.mutation_summary()},
            "promotion_gate": {
                "status": reason,
                "eligible_for_structural_mutation": False,
                "eligible_for_action": False,
                "required_evidence": dict(required_evidence),
            },
        }

    def _candidate_provenance(
        self,
        *,
        preflight: Mapping[str, Any],
        design: Mapping[str, Any],
        recomputed_preflight_hash: str,
    ) -> dict[str, Any]:
        supplied_preflight_hash = str(preflight.get("structural_mutation_preflight_hash") or "")
        return {
            "surface": "subcortical_structural_candidate_provenance.v1",
            "source": "service.structural_mutation_executor",
            "preflight_hash": supplied_preflight_hash or None,
            "recomputed_preflight_hash": recomputed_preflight_hash,
            "preflight_hash_recomputed_match": bool(
                supplied_preflight_hash and supplied_preflight_hash == recomputed_preflight_hash
            ),
            "structural_mutation_design_hash": design.get("structural_mutation_design_hash"),
            "checkpointed_candidate_gate_status": design.get(
                "checkpointed_candidate_gate_status"
            ),
            "candidate_evidence_hash": design.get("candidate_evidence_hash"),
            "candidate_baseline_hash": design.get("candidate_baseline_hash"),
            "candidate_reason": design.get("candidate_reason"),
            "cost_impact_summary_hash": design.get("cost_impact_summary_hash"),
            "hash_algorithm": "sha256_canonical_json",
        }

    def _binding_rollback_artifact(
        self,
        *,
        scope: str,
        checkpoint_path: str,
        before_revision: int,
        before_state: Mapping[str, Any],
        current_state: Mapping[str, Any],
        checkpoint_verified: bool,
    ) -> dict[str, Any]:
        state_restored = dict(before_state) == dict(current_state)
        payload = {
            "surface": "subcortical_structural_mutation_rollback_artifact.v1",
            "available": True,
            "scope": scope,
            "checkpoint_path": checkpoint_path,
            "checkpoint_restore_verified": bool(checkpoint_verified),
            "state_revision_after_rollback": int(self._runtime_state.state_revision),
            "state_revision_restored": int(self._runtime_state.state_revision) == int(before_revision),
            "binding_state_restored": state_restored,
        }
        payload["rollback_artifact_hash"] = self._sha256_json(payload)
        return payload

    def _commit_failure_rollback_artifact(
        self,
        *,
        commit: Mapping[str, Any],
        checkpoint_path: str,
        before_revision: int,
    ) -> dict[str, Any]:
        payload = {
            "surface": "subcortical_structural_mutation_rollback_artifact.v1",
            "available": bool(commit.get("rollback_recovered_in_memory")),
            "scope": "binding_hub_topology_commit_failure",
            "checkpoint_path": checkpoint_path,
            "committed_checkpoint_path": commit.get("committed_checkpoint_path"),
            "rollback_checkpoint_rewritten_verified": bool(
                commit.get("rollback_checkpoint_rewritten_verified")
            ),
            "state_revision_after_rollback": int(self._runtime_state.state_revision),
            "state_revision_restored": int(self._runtime_state.state_revision) == int(before_revision),
            "binding_state_restored": bool(commit.get("rollback_recovered_in_memory")),
        }
        payload["rollback_artifact_hash"] = self._sha256_json(payload)
        return payload

    def _normalize_rollback_artifact(
        self,
        rollback_artifact: Mapping[str, Any] | None,
        *,
        checkpoint_path: str | None,
        before_revision: int,
    ) -> dict[str, Any]:
        payload = dict(rollback_artifact or {})
        payload.setdefault("surface", "subcortical_structural_mutation_rollback_artifact.v1")
        payload.setdefault("available", False)
        payload.setdefault("scope", "blocked_before_structural_mutation")
        payload.setdefault("checkpoint_path", checkpoint_path)
        payload.setdefault("state_revision_after_rollback", int(self._runtime_state.state_revision))
        payload.setdefault(
            "state_revision_restored",
            int(self._runtime_state.state_revision) == int(before_revision),
        )
        payload.setdefault("binding_state_restored", False)
        payload.setdefault("rollback_artifact_hash", self._sha256_json(payload))
        return payload

    def _candidate_tombstone_manifest(
        self,
        *,
        reason: str,
        before_revision: int,
        checkpoint_path: str | None,
        candidate_provenance: Mapping[str, Any],
        required_evidence: Mapping[str, Any],
        rollback_artifact: Mapping[str, Any],
    ) -> dict[str, Any]:
        disposition = self._candidate_disposition(reason)
        manifest = {
            "surface": "subcortical_structural_candidate_tombstone.v1",
            "artifact_kind": "marulho_subcortical_structural_candidate_tombstone",
            "status": disposition,
            "reason": reason,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "before_state_revision": int(before_revision),
            "checkpoint_path": checkpoint_path,
            "candidate_evidence_hash": candidate_provenance.get("candidate_evidence_hash"),
            "candidate_baseline_hash": candidate_provenance.get("candidate_baseline_hash"),
            "candidate_reason": candidate_provenance.get("candidate_reason"),
            "structural_mutation_design_hash": candidate_provenance.get(
                "structural_mutation_design_hash"
            ),
            "structural_mutation_preflight_hash": candidate_provenance.get("preflight_hash"),
            "preflight_hash_recomputed_match": bool(
                candidate_provenance.get("preflight_hash_recomputed_match")
            ),
            "rollback_artifact_hash": rollback_artifact.get("rollback_artifact_hash"),
            "required_evidence_hash": self._sha256_json(dict(required_evidence)),
            "mutates_runtime_state": False,
            "calls_growth_or_prune": False,
            "writes_checkpoint": False,
            "applies_structural_mutation": False,
        }
        manifest["tombstone_manifest_hash"] = self._sha256_json(manifest)
        return manifest

    @staticmethod
    def _candidate_disposition(reason: str) -> str:
        if reason in {
            "blocked_no_binding_hub_topology_delta",
            "blocked_binding_hub_topology_delta_over_budget",
        }:
            return "rejected_with_rollback"
        if reason == "post_structural_mutation_checkpoint_commit_failed":
            return "retired_after_rollback"
        return "blocked_before_mutation"

    def _verify_checkpoint_transaction(
        self,
        path: Path,
        expected_binding_state: Mapping[str, Any],
        expected_revision: int,
    ) -> bool:
        return bool(
            path.exists()
            and self._verify_checkpoint(path)
            and self._verify_checkpoint_snapshot(path, expected_binding_state, expected_revision)
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
        binding = self._binding_layer()
        after_state = deepcopy(binding.state_dict())
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
        binding.load_state_dict(dict(before_state))
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
    def _structural_delta(report: Mapping[str, Any]) -> dict[str, int]:
        before = report.get("before") if isinstance(report.get("before"), Mapping) else {}
        after = report.get("after") if isinstance(report.get("after"), Mapping) else {}
        before_mutations = (
            before.get("structural_mutations")
            if isinstance(before.get("structural_mutations"), Mapping)
            else {}
        )
        after_mutations = (
            after.get("structural_mutations")
            if isinstance(after.get("structural_mutations"), Mapping)
            else {}
        )
        return {
            "edges_added_delta": int(after_mutations.get("edges_added_total", 0))
            - int(before_mutations.get("edges_added_total", 0)),
            "edges_removed_delta": int(after_mutations.get("edges_removed_total", 0))
            - int(before_mutations.get("edges_removed_total", 0)),
            "growth_events_delta": int(after_mutations.get("growth_events", 0))
            - int(before_mutations.get("growth_events", 0)),
            "prune_events_delta": int(after_mutations.get("prune_events", 0))
            - int(before_mutations.get("prune_events", 0)),
        }

    @staticmethod
    def _sha256_json(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            dict(payload),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _committed_checkpoint_path(rollback_checkpoint_file: Path, operation: str) -> Path:
        suffix = rollback_checkpoint_file.suffix or ".pt"
        return rollback_checkpoint_file.with_name(
            f"{rollback_checkpoint_file.stem}.{operation}.committed{suffix}"
        )
