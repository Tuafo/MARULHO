from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

from hecsn.service.runtime_state import RuntimeState
from hecsn.service.structural_mutation_executor import StructuralMutationExecutor


class _FakeConceptStore:
    def __init__(self, *, changes: bool) -> None:
        self._state = {"version": 0}
        self._changes = changes

    def state_dict(self) -> dict[str, Any]:
        return dict(self._state)

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self._state = dict(payload)

    def refresh_structural_capacity(self) -> dict[str, Any]:
        if self._changes:
            self._state["version"] = int(self._state["version"]) + 1
            return {"growth_ready": True, "expansion_events": 1, "prune_events": 0}
        return {"growth_ready": False, "expansion_events": 0, "prune_events": 0}


def _ready_preflight(checkpoint_path: Path, expected_revision: int = 0) -> dict[str, Any]:
    return {
        "surface": "subcortical_structural_mutation_preflight.v1",
        "ready": True,
        "writes_checkpoint": False,
        "mutates_runtime_state": False,
        "calls_growth_or_prune": False,
        "expected_state_revision": expected_revision,
        "checkpoint_path": str(checkpoint_path),
        "design_binding": {
            "design_hash_available": True,
            "design_hash_recomputed_match": True,
            "structural_mutation_design_hash": "a" * 64,
        },
        "promotion_gate": {
            "eligible_for_operator_execution_review": True,
            "eligible_for_structural_mutation": False,
        },
    }


def _executor(tmp_path: Path, store: _FakeConceptStore, runtime_state: RuntimeState) -> StructuralMutationExecutor:
    def save_checkpoint(path: str | None) -> dict[str, Any]:
        checkpoint = Path(path or tmp_path / "checkpoint.pt")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return {"path": str(checkpoint)}

    return StructuralMutationExecutor(
        lock=RLock(),
        runtime_state=runtime_state,
        concept_store=lambda: store,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / "checkpoint.pt",
        verify_checkpoint=lambda path: path.is_file(),
        verify_checkpoint_snapshot=lambda path, state, revision: path.is_file()
        and int(revision) == int(runtime_state.state_revision),
        publish_committed_checkpoint=lambda path, operation: {
            "checkpoint_path": str(path),
            "operation": operation,
        },
    )


def test_structural_mutation_application_commits_real_concept_store_delta(tmp_path: Path) -> None:
    runtime_state = RuntimeState(lock=RLock())
    store = _FakeConceptStore(changes=True)
    executor = _executor(tmp_path, store, runtime_state)

    result = executor.apply_subcortical_structural_mutation(
        structural_mutation_preflight=_ready_preflight(tmp_path / "pre.pt"),
        expected_state_revision=0,
        operator_id="operator-structural",
        confirmation=True,
    )

    assert result["accepted"] is True
    assert result["surface"] == "subcortical_structural_mutation_application.v1"
    assert result["checkpoint_transaction"]["restore_verified"] is True
    assert result["calls_growth_or_prune"] is True
    assert result["applies_structural_mutation"] is True
    assert result["before"]["state_revision"] == 0
    assert result["after"]["state_revision"] == 1
    assert runtime_state.state_revision == 1
    assert store.state_dict()["version"] == 1


def test_structural_mutation_application_blocks_when_refresh_has_no_delta(tmp_path: Path) -> None:
    runtime_state = RuntimeState(lock=RLock())
    store = _FakeConceptStore(changes=False)
    executor = _executor(tmp_path, store, runtime_state)

    result = executor.apply_subcortical_structural_mutation(
        structural_mutation_preflight=_ready_preflight(tmp_path / "pre.pt"),
        expected_state_revision=0,
        operator_id="operator-structural",
        confirmation=True,
    )

    assert result["accepted"] is False
    assert result["reason"] == "blocked_no_structural_capacity_delta"
    assert result["promotion_gate"]["required_evidence"]["structural_capacity_delta_observed"] is False
    assert result["mutates_runtime_state"] is False
    assert runtime_state.state_revision == 0
    assert store.state_dict()["version"] == 0


def test_structural_mutation_application_requires_bound_preflight_revision_and_operator(
    tmp_path: Path,
) -> None:
    runtime_state = RuntimeState(lock=RLock())
    store = _FakeConceptStore(changes=True)
    executor = _executor(tmp_path, store, runtime_state)

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
