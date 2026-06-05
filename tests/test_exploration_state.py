"""Tests for Subcortex active-exploration state."""

from __future__ import annotations

from marulho.semantics.exploration_state import ExplorationState, normalize_exploration_target


def test_normalize_exploration_target_removes_separators_and_bounds_length() -> None:
    target = "reef / chemistry | thermal   stress " + ("x" * 160)
    normalized = normalize_exploration_target(target)

    assert "/" not in normalized
    assert "|" not in normalized
    assert normalized.startswith("reef chemistry thermal stress")
    assert len(normalized) == 120


def test_exploration_state_from_target_clamps_score_and_metadata() -> None:
    state = ExplorationState.from_target(
        " reef / chemistry ",
        reason="prediction error " * 20,
        source="subcortex.signal.provider.with.long.name",
        score=2.5,
        updated_at=-5.0,
    )

    assert state.target == "reef chemistry"
    assert len(state.reason) == 160
    assert state.source == "subcortex.signal.provider.with.long.name"[:40]
    assert state.score == 1.0
    assert state.updated_at == 0.0
    assert state.to_dict() == {
        "target": "reef chemistry",
        "reason": state.reason,
        "source": state.source,
        "score": 1.0,
        "updated_at": 0.0,
    }


def test_exploration_state_empty_target_returns_empty_state() -> None:
    assert ExplorationState.from_target("   ").to_dict() == {
        "target": "",
        "reason": "",
        "source": "",
        "score": 0.0,
        "updated_at": 0.0,
    }
