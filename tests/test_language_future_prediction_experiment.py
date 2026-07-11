from __future__ import annotations

from dataclasses import asdict

import pytest

from marulho.evaluation.language_future_prediction_experiment import (
    SAVE_DECISION,
    FuturePredictionExperimentConfig,
    _validate_matched_control,
    future_prediction_decision,
)


def _control(config: FuturePredictionExperimentConfig) -> dict:
    return {
        "artifact_kind": "marulho_hashed_micro_expert_general_continuation",
        "decision": "save_v11_general_continuation_for_unseen_generation",
        "configuration": asdict(config),
        "parent": {"sha256": "parent"},
        "schedule": {"sha256": "schedule"},
        "after": {
            "cumulative_processed_tokens": 318_775_424,
            "arm": {
                "processed_tokens": 67_112_960,
                "heldout": {
                    "heldout_loss": 3.3243,
                    "heldout_perplexity": 27.78,
                },
            },
        },
    }


def test_future_prediction_decision_requires_matched_control_gain() -> None:
    assert future_prediction_decision(
        control_heldout_loss=3.3243,
        candidate_heldout_loss=3.30,
        processed_tokens=200,
        requested_tokens=200,
    ) == SAVE_DECISION
    assert future_prediction_decision(
        control_heldout_loss=3.3243,
        candidate_heldout_loss=3.32,
        processed_tokens=200,
        requested_tokens=200,
    ) == "retire_v13_future_prediction_weak_control_gain"
    assert future_prediction_decision(
        control_heldout_loss=3.3243,
        candidate_heldout_loss=3.34,
        processed_tokens=200,
        requested_tokens=200,
    ) == "retire_v13_future_prediction_no_control_gain"
    assert future_prediction_decision(
        control_heldout_loss=3.3243,
        candidate_heldout_loss=3.0,
        processed_tokens=199,
        requested_tokens=200,
    ) == "incomplete_v13_future_prediction"


def test_future_prediction_control_must_match_parent_schedule_and_recipe() -> None:
    config = FuturePredictionExperimentConfig()
    record = _validate_matched_control(
        _control(config),
        parent_sha256="parent",
        schedule_sha256="schedule",
        config=config,
    )
    assert record["heldout_loss"] == pytest.approx(3.3243)
    assert record["processed_tokens"] == 67_112_960

    with pytest.raises(ValueError, match="different parent"):
        _validate_matched_control(
            _control(config),
            parent_sha256="other",
            schedule_sha256="schedule",
            config=config,
        )
    with pytest.raises(ValueError, match="different data schedule"):
        _validate_matched_control(
            _control(config),
            parent_sha256="parent",
            schedule_sha256="other",
            config=config,
        )
    changed = FuturePredictionExperimentConfig(batch_size=41)
    with pytest.raises(ValueError, match="configuration mismatch"):
        _validate_matched_control(
            _control(config),
            parent_sha256="parent",
            schedule_sha256="schedule",
            config=changed,
        )
