from __future__ import annotations

import pytest

from marulho.evaluation.language_dclm_replacement import (
    PHASE_TOKENS,
    TARGET_CUMULATIVE_TOKENS,
    TRAIN_STEPS,
    WARMUP_STEPS,
    _candidate_quality_checks,
    _learning_rate,
)


def test_v79_budget_and_learning_rate_are_frozen() -> None:
    assert PHASE_TOKENS == 31_457_280
    assert TARGET_CUMULATIVE_TOKENS == 288_887_040
    assert _learning_rate(0) == pytest.approx(3.0e-5 + 2.7e-4 / WARMUP_STEPS)
    assert _learning_rate(WARMUP_STEPS - 1) == pytest.approx(3.0e-4)
    assert _learning_rate(TRAIN_STEPS - 1) == pytest.approx(3.0e-5)


def _report(mean: float, old: float, dclm: float, throughput: float) -> dict:
    return {
        "final_evaluation": {
            "later_segment_loss": mean,
            "old_source_mean_later_loss": old,
            "later_loss_by_source": {"dclm_edu": dclm},
        },
        "training": {
            "positions_per_second_including_curve_evaluations": throughput,
        },
    }


def test_v79_candidate_gate_requires_joint_gain_and_matched_speed() -> None:
    control = _report(3.00, 2.80, 3.40, 20_000.0)
    candidate_final = {
        "later_segment_loss": 2.96,
        "old_source_mean_later_loss": 2.81,
        "later_loss_by_source": {"dclm_edu": 3.30},
    }
    candidate_training = {
        "positions_per_second_including_curve_evaluations": 19_500.0,
    }
    checks, deltas = _candidate_quality_checks(
        control=control,
        candidate_final=candidate_final,
        candidate_training=candidate_training,
    )
    assert all(checks.values())
    assert deltas["three_source_gain_control_minus_candidate"] == pytest.approx(0.04)

    candidate_final["later_loss_by_source"]["dclm_edu"] = 3.33
    checks, _ = _candidate_quality_checks(
        control=control,
        candidate_final=candidate_final,
        candidate_training=candidate_training,
    )
    assert not checks["dclm_gain_at_least_0_08"]


def test_v79_candidate_gate_rejects_unmatched_throughput() -> None:
    control = _report(3.00, 2.80, 3.40, 20_000.0)
    candidate_final = {
        "later_segment_loss": 2.96,
        "old_source_mean_later_loss": 2.80,
        "later_loss_by_source": {"dclm_edu": 3.30},
    }
    checks, _ = _candidate_quality_checks(
        control=control,
        candidate_final=candidate_final,
        candidate_training={
            "positions_per_second_including_curve_evaluations": 18_000.0,
        },
    )
    assert not checks["throughput_within_5_percent"]
