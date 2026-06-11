from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from threading import RLock
from typing import Any

from marulho.semantics import (
    build_subcortical_structural_mutation_design,
    build_subcortical_structural_mutation_preflight,
)
from marulho.service.runtime_state import RuntimeState
from marulho.service.structural_mutation_executor import StructuralMutationExecutor


class _FakeBindingLayer:
    def __init__(self, *, changes: bool, edge_delta: int = 2) -> None:
        self._state = {
            "version": 0,
            "structural_growth_events": 0,
            "structural_prune_events": 0,
            "structural_edges_added_total": 0,
            "structural_edges_removed_total": 0,
            "hub_topology_refresh_count": 0,
            "last_hub_topology_refresh_reason": None,
        }
        self._changes = changes
        self._edge_delta = int(edge_delta)

    def state_dict(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self._state = deepcopy(payload)

    def refresh_hub_topology(self, *, reason: str) -> dict[str, Any]:
        before = self._stats()
        self._state["hub_topology_refresh_count"] += 1
        self._state["last_hub_topology_refresh_reason"] = reason
        if self._changes:
            self._state["version"] += 1
            self._state["structural_growth_events"] += 1
            self._state["structural_edges_added_total"] += self._edge_delta
        after = self._stats()
        return {
            "reason": reason,
            "refresh_count": self._state["hub_topology_refresh_count"],
            "before": before,
            "after": after,
            "topology_changed": self._changes,
        }

    def _stats(self) -> dict[str, Any]:
        return {
            "structural_mutations": {
                "growth_events": self._state["structural_growth_events"],
                "prune_events": self._state["structural_prune_events"],
                "edges_added_total": self._state["structural_edges_added_total"],
                "edges_removed_total": self._state["structural_edges_removed_total"],
            }
        }


def _ready_preflight(checkpoint_path: Path, expected_revision: int = 0) -> dict[str, Any]:
    design_binding = {
        "design_hash_available": True,
        "design_hash_recomputed_match": True,
        "structural_mutation_design_hash": "a" * 64,
        "mutation_target": "marulho.subcortex.binding.hub_topology",
        "mutation_method": "HypercubeBindingLayer.refresh_hub_topology",
        "mutation_reason": "repeated isolated prediction failure",
        "max_total_edge_delta": 16,
    }
    required_evidence = {"design_hash_recomputed_match": True}
    preflight_material = {
        "structural_mutation_design_hash": design_binding["structural_mutation_design_hash"],
        "mutation_target": design_binding["mutation_target"],
        "mutation_method": design_binding["mutation_method"],
        "mutation_reason": design_binding["mutation_reason"],
        "max_total_edge_delta": design_binding["max_total_edge_delta"],
        "expected_state_revision": expected_revision,
        "current_state_revision": expected_revision,
        "checkpoint_path": str(checkpoint_path),
        "required_evidence": required_evidence,
    }
    preflight_hash = hashlib.sha256(
        json.dumps(
            preflight_material,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "surface": "subcortical_structural_mutation_preflight.v1",
        "writes_checkpoint": False,
        "mutates_runtime_state": False,
        "calls_growth_or_prune": False,
        "structural_mutation_preflight_hash": preflight_hash,
        "design_binding": design_binding,
        "checkpoint_transaction_requirements": {
            "expected_state_revision": expected_revision,
            "current_state_revision": expected_revision,
            "checkpoint_path": str(checkpoint_path),
        },
        "promotion_gate": {
            "eligible_for_operator_execution_review": True,
            "eligible_for_structural_mutation": False,
            "required_evidence": required_evidence,
        },
    }


def _executor(
    tmp_path: Path,
    binding: _FakeBindingLayer,
    runtime_state: RuntimeState,
    *,
    fail_committed_verification: bool = False,
) -> StructuralMutationExecutor:
    def save_checkpoint(path: str | None) -> dict[str, Any]:
        checkpoint = Path(path or tmp_path / "checkpoint.pt")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return {"path": str(checkpoint)}

    def verify_snapshot(path: Path, state: dict[str, Any], revision: int) -> bool:
        if fail_committed_verification and ".committed" in path.name:
            return False
        return path.is_file() and int(revision) == int(runtime_state.state_revision)

    return StructuralMutationExecutor(
        lock=RLock(),
        runtime_state=runtime_state,
        binding_layer=lambda: binding,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / "checkpoint.pt",
        verify_checkpoint=lambda path: path.is_file(),
        verify_checkpoint_snapshot=verify_snapshot,
        publish_committed_checkpoint=lambda path, operation: {
            "checkpoint_path": str(path),
            "operation": operation,
        },
    )


def test_structural_mutation_application_commits_real_binding_edge_delta(tmp_path: Path) -> None:
    runtime_state = RuntimeState(lock=RLock())
    binding = _FakeBindingLayer(changes=True)
    executor = _executor(tmp_path, binding, runtime_state)

    result = executor.apply_subcortical_structural_mutation(
        structural_mutation_preflight=_ready_preflight(tmp_path / "pre.pt"),
        expected_state_revision=0,
        operator_id="operator-structural",
        confirmation=True,
    )

    assert result["accepted"] is True
    assert result["checkpoint_transaction"]["restore_verified"] is True
    assert result["application_target"]["target_id"] == "marulho.subcortex.binding.hub_topology"
    assert result["structural_delta"] == {
        "edges_added_delta": 2,
        "edges_removed_delta": 0,
        "growth_events_delta": 1,
        "prune_events_delta": 0,
    }
    assert result["before"]["state_revision"] == 0
    assert result["after"]["state_revision"] == 1
    assert runtime_state.state_revision == 1
    assert binding.state_dict()["version"] == 1


def test_structural_mutation_application_restores_when_refresh_has_no_topology_delta(
    tmp_path: Path,
) -> None:
    runtime_state = RuntimeState(lock=RLock())
    binding = _FakeBindingLayer(changes=False)
    before = binding.state_dict()
    executor = _executor(tmp_path, binding, runtime_state)

    result = executor.apply_subcortical_structural_mutation(
        structural_mutation_preflight=_ready_preflight(tmp_path / "pre.pt"),
        expected_state_revision=0,
        operator_id="operator-structural",
        confirmation=True,
    )

    assert result["accepted"] is False
    assert result["reason"] == "blocked_no_binding_hub_topology_delta"
    assert result["promotion_gate"]["required_evidence"]["binding_hub_topology_delta_observed"] is False
    assert runtime_state.state_revision == 0
    assert binding.state_dict() == before


def test_structural_mutation_application_rolls_back_binding_and_revision_on_commit_failure(
    tmp_path: Path,
) -> None:
    runtime_state = RuntimeState(lock=RLock())
    binding = _FakeBindingLayer(changes=True)
    before = binding.state_dict()
    executor = _executor(
        tmp_path,
        binding,
        runtime_state,
        fail_committed_verification=True,
    )

    result = executor.apply_subcortical_structural_mutation(
        structural_mutation_preflight=_ready_preflight(tmp_path / "pre.pt"),
        expected_state_revision=0,
        operator_id="operator-structural",
        confirmation=True,
    )

    assert result["accepted"] is False
    assert result["reason"] == "post_structural_mutation_checkpoint_commit_failed"
    assert result["promotion_gate"]["required_evidence"]["rollback_recovered_in_memory"] is True
    assert runtime_state.state_revision == 0
    assert binding.state_dict() == before


def test_structural_mutation_application_requires_bound_preflight_revision_and_operator(
    tmp_path: Path,
) -> None:
    runtime_state = RuntimeState(lock=RLock())
    binding = _FakeBindingLayer(changes=True)
    executor = _executor(tmp_path, binding, runtime_state)

    result = executor.apply_subcortical_structural_mutation(
        structural_mutation_preflight=_ready_preflight(tmp_path / "pre.pt"),
        expected_state_revision=1,
        operator_id="",
        confirmation=False,
    )

    evidence = result["promotion_gate"]["required_evidence"]
    assert result["accepted"] is False
    assert evidence["expected_revision_current"] is False
    assert evidence["expected_revision_matches_preflight"] is False
    assert evidence["operator_id_available"] is False
    assert evidence["confirmation"] is False
    assert runtime_state.state_revision == 0


def test_structural_mutation_application_rejects_tampered_target_reason(tmp_path: Path) -> None:
    runtime_state = RuntimeState(lock=RLock())
    binding = _FakeBindingLayer(changes=True)
    executor = _executor(tmp_path, binding, runtime_state)
    preflight = _ready_preflight(tmp_path / "pre.pt")
    preflight["design_binding"]["mutation_reason"] = "tampered after preflight"

    result = executor.apply_subcortical_structural_mutation(
        structural_mutation_preflight=preflight,
        expected_state_revision=0,
        operator_id="operator-structural",
        confirmation=True,
    )

    assert result["accepted"] is False
    assert result["promotion_gate"]["required_evidence"]["preflight_hash_recomputed_match"] is False
    assert runtime_state.state_revision == 0
    assert binding.state_dict()["version"] == 0


def test_structural_mutation_application_rolls_back_over_budget_edge_delta(
    tmp_path: Path,
) -> None:
    runtime_state = RuntimeState(lock=RLock())
    binding = _FakeBindingLayer(changes=True, edge_delta=17)
    before = binding.state_dict()
    executor = _executor(tmp_path, binding, runtime_state)

    result = executor.apply_subcortical_structural_mutation(
        structural_mutation_preflight=_ready_preflight(tmp_path / "pre.pt"),
        expected_state_revision=0,
        operator_id="operator-structural",
        confirmation=True,
    )

    assert result["accepted"] is False
    assert result["reason"] == "blocked_binding_hub_topology_delta_over_budget"
    evidence = result["promotion_gate"]["required_evidence"]
    assert evidence["actual_total_edge_delta"] == 17
    assert evidence["max_total_edge_delta"] == 16
    assert evidence["actual_edge_delta_within_budget"] is False
    assert runtime_state.state_revision == 0
    assert binding.state_dict() == before


def test_real_design_and_preflight_artifacts_execute_without_test_only_fields(
    tmp_path: Path,
) -> None:
    runtime_state = RuntimeState(lock=RLock())
    binding = _FakeBindingLayer(changes=True)
    executor = _executor(tmp_path, binding, runtime_state)
    evaluation = {
        "surface": "subcortical_structural_plasticity_isolated_evaluation.v1",
        "artifact_kind": "terminus_subcortical_structural_plasticity_isolated_evaluation",
        "structural_delta": {
            "edges_added_delta": 2,
            "edges_removed_delta": 0,
            "growth_events_delta": 1,
            "prune_events_delta": 0,
            "total_edge_delta": 2,
            "bounded_edge_delta_limit": 16,
            "bounded": True,
        },
        "snapshot_binding": {
            "pre_snapshot_hash": "b" * 64,
            "post_snapshot_hash": "c" * 64,
            "pre_state_revision": 0,
            "post_state_revision": 1,
            "snapshot_hashes_distinct": True,
            "structural_delta_present": True,
        },
        "rollback_evidence": {
            "snapshot_id": "isolated-pre",
            "pre_snapshot_hash": "b" * 64,
            "bound_to_pre_snapshot": True,
        },
        "device_evidence": {"consistent": True},
        "spike_health_delta": {"improved_or_stable": True},
        "runtime_truth_delta": {"improved_or_stable": True},
        "promotion_gate": {"status": "ready_for_operator_review"},
    }
    design = build_subcortical_structural_mutation_design(
        evaluation,
        operator_id="operator-structural",
        confirmation=True,
        mutation_reason="repeated isolated prediction failure",
    )
    preflight = build_subcortical_structural_mutation_preflight(
        design,
        expected_state_revision=0,
        current_state_revision=0,
        checkpoint_path=str(tmp_path / "pre.pt"),
    )

    result = executor.apply_subcortical_structural_mutation(
        structural_mutation_preflight=preflight,
        expected_state_revision=0,
        operator_id="operator-structural",
        confirmation=True,
    )

    assert result["accepted"] is True
    assert result["structural_mutation_event"]["mutation_reason"] == (
        "repeated isolated prediction failure"
    )
