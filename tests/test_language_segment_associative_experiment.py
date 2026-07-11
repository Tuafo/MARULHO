from __future__ import annotations

import pytest

from marulho.evaluation.language_segment_associative_experiment import (
    SAVE_DECISION,
    _validate_one_billion_parent,
    segment_associative_decision,
)
from marulho.training.language_segment_associative_state import (
    SEGMENT_ASSOCIATIVE_MODES,
)


def _rows(losses: dict[str, float], *, tokens: int = 100) -> dict:
    return {
        mode: {
            "processed_tokens": tokens,
            "heldout": {"heldout_loss": losses[mode]},
        }
        for mode in SEGMENT_ASSOCIATIVE_MODES
    }


def test_segment_associative_decision_requires_joint_gated_win() -> None:
    assert segment_associative_decision(
        _rows(
            {
                "off": 3.10,
                "local": 3.09,
                "delta": 3.055,
                "gated_delta": 3.05,
            }
        ),
        requested_tokens=100,
    ) == SAVE_DECISION
    assert segment_associative_decision(
        _rows(
            {
                "off": 3.10,
                "local": 3.09,
                "delta": 3.04,
                "gated_delta": 3.06,
            }
        ),
        requested_tokens=100,
    ) == "redesign_v14_gate_keep_ungated_delta_evidence"
    assert segment_associative_decision(
        _rows(
            {
                "off": 3.10,
                "local": 3.09,
                "delta": 3.08,
                "gated_delta": 3.08,
            }
        ),
        requested_tokens=100,
    ) == "retire_v14_weak_segment_state_gain"
    assert segment_associative_decision(
        _rows(
            {
                "off": 3.08,
                "local": 3.07,
                "delta": 3.10,
                "gated_delta": 3.09,
            }
        ),
        requested_tokens=100,
    ) == "retire_v14_no_segment_state_gain"


def test_segment_associative_decision_rejects_incomplete_evidence() -> None:
    incomplete = _rows(
        {mode: 3.0 for mode in SEGMENT_ASSOCIATIVE_MODES},
        tokens=99,
    )
    assert segment_associative_decision(
        incomplete,
        requested_tokens=100,
    ) == "incomplete_v14_token_budget"
    incomplete.pop("local")
    assert segment_associative_decision(
        incomplete,
        requested_tokens=100,
    ) == "incomplete_v14_missing_control_arm"


def test_v14_parent_must_be_owned_one_billion_checkpoint() -> None:
    metadata = {
        "processed_tokens": 1_000_001_664,
        "external_llm_used": False,
    }
    assert _validate_one_billion_parent(metadata) == 1_000_001_664
    with pytest.raises(ValueError, match="one-billion-token"):
        _validate_one_billion_parent({**metadata, "processed_tokens": 999_999_999})
    with pytest.raises(ValueError, match="MARULHO-owned"):
        _validate_one_billion_parent({**metadata, "external_llm_used": True})
