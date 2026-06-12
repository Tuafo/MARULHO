from __future__ import annotations

from copy import deepcopy

from marulho.evaluation.inplace_transition_quality_benchmark import (
    compare_quality_arms,
)


def _arm() -> dict[str, object]:
    return {
        "throughput_ticks_per_second": 40.0,
        "reconstruction_error": {"mean": 0.20},
        "prediction": {"error_mean": 0.10, "confidence_mean": 0.80},
        "trajectory": {
            "winners": [1, 2, 1, 3],
            "winner_distribution": {"normalized_entropy": 0.75},
        },
        "spike_health": {
            "activity_state": "sparse_responsive",
            "silent_fraction": 0.10,
            "saturated_fraction": 0.02,
        },
        "memory": {
            "mean_capture_strength": 0.40,
            "mean_consolidation_level": 0.30,
            "mean_fragility": 0.20,
        },
        "state_finite": True,
        "grounding": {"quality_evidence_available": False},
    }


def test_quality_comparison_preserves_quality_but_blocks_ungrounded_promotion() -> None:
    baseline = _arm()
    variant = deepcopy(baseline)
    variant["throughput_ticks_per_second"] = 80.0

    result = compare_quality_arms(baseline, variant)

    assert result["quality_preserved"] is True
    assert result["grounding_quality_available"] is False
    assert result["promotion_eligible"] is False
    assert result["deltas"]["speedup"] == 2.0
    assert result["winner_agreement"] == 1.0


def test_quality_comparison_rejects_speed_that_degrades_prediction() -> None:
    baseline = _arm()
    variant = deepcopy(baseline)
    variant["throughput_ticks_per_second"] = 100.0
    variant["prediction"]["error_mean"] = 0.25
    variant["trajectory"]["winners"] = [3, 3, 3, 3]

    result = compare_quality_arms(baseline, variant)

    assert result["quality_preserved"] is False
    assert result["promotion_eligible"] is False
    assert result["gates"]["prediction_error_mean_within_tolerance"] is False
    assert result["winner_agreement"] == 0.25
