from __future__ import annotations

import pytest

from marulho.evaluation.language_grouped_recurrent_experiment import (
    ADVANCE_DECISION,
    ARM_NAMES,
    _validate_one_billion_parent,
    grouped_recurrent_decision,
)


def _rows(
    losses: dict[str, float],
    *,
    relations: dict[str, float] | None = None,
    tokens: int = 100,
) -> dict:
    relation_values = relations or {name: 0.50 for name in ARM_NAMES}
    return {
        name: {
            "processed_tokens": tokens,
            "heldout": {"heldout_loss": losses[name]},
            "relation": {"accuracy": relation_values[name]},
        }
        for name in ARM_NAMES
    }


def test_grouped_decision_requires_isolated_loss_and_relation_guard() -> None:
    losses = {"off": 3.10, "local": 3.09, "dense": 3.08, "grouped": 3.05}
    assert grouped_recurrent_decision(
        _rows(losses),
        requested_tokens=100,
        minimum_screen_tokens=100,
    ) == ADVANCE_DECISION
    relations = {"off": 0.50, "local": 0.51, "dense": 0.52, "grouped": 0.48}
    assert grouped_recurrent_decision(
        _rows(losses, relations=relations),
        requested_tokens=100,
        minimum_screen_tokens=100,
    ) == "redesign_v17_grouped_loss_gain_relation_regression"


def test_grouped_decision_distinguishes_dense_and_local_gains() -> None:
    assert grouped_recurrent_decision(
        _rows({"off": 3.10, "local": 3.09, "dense": 3.05, "grouped": 3.08}),
        requested_tokens=100,
        minimum_screen_tokens=100,
    ) == "redesign_v17_recurrence_gain_not_grouping"
    assert grouped_recurrent_decision(
        _rows({"off": 3.10, "local": 3.07, "dense": 3.08, "grouped": 3.08}),
        requested_tokens=100,
        minimum_screen_tokens=100,
    ) == "redesign_v17_local_capacity_gain_no_recurrence_gain"


def test_grouped_decision_rejects_incomplete_evidence() -> None:
    rows = _rows({name: 3.0 for name in ARM_NAMES}, tokens=99)
    assert grouped_recurrent_decision(
        rows,
        requested_tokens=100,
        minimum_screen_tokens=100,
    ) == "incomplete_v17_token_budget"
    rows.pop("dense")
    assert grouped_recurrent_decision(
        rows,
        requested_tokens=100,
        minimum_screen_tokens=100,
    ) == "incomplete_v17_missing_control_arm"


def test_grouped_decision_cannot_promote_a_smoke_run() -> None:
    rows = _rows({name: 3.0 for name in ARM_NAMES}, tokens=100)
    assert grouped_recurrent_decision(
        rows,
        requested_tokens=100,
        minimum_screen_tokens=101,
    ) == "diagnostic_v17_below_screen_budget"


def test_v17_parent_must_be_owned_one_billion_checkpoint() -> None:
    metadata = {"processed_tokens": 1_000_001_664, "external_llm_used": False}
    assert _validate_one_billion_parent(metadata) == 1_000_001_664
    with pytest.raises(ValueError, match="one-billion-token"):
        _validate_one_billion_parent({**metadata, "processed_tokens": 999_999_999})
    with pytest.raises(ValueError, match="MARULHO-owned"):
        _validate_one_billion_parent({**metadata, "external_llm_used": True})
