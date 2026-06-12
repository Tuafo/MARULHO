from __future__ import annotations

from copy import deepcopy

import torch

from marulho.evaluation.inplace_grounded_quality_benchmark import (
    _make_grounded_stream,
    compare_grounded_arms,
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
        "grounding": {
            "quality_evidence_available": True,
            "visual_prediction_cosine_mean": 0.70,
            "audio_prediction_cosine_mean": 0.65,
            "visual_confidence_mean": 0.50,
            "audio_confidence_mean": 0.45,
            "visual_acceptance_rate": 1.0,
            "audio_acceptance_rate": 1.0,
        },
    }


def test_grounded_stream_is_deterministic_and_multimodal() -> None:
    left = _make_grounded_stream(
        input_dim=16,
        visual_dim=8,
        audio_dim=4,
        samples=4,
        concepts=2,
        seed=17,
    )
    right = _make_grounded_stream(
        input_dim=16,
        visual_dim=8,
        audio_dim=4,
        samples=4,
        concepts=2,
        seed=17,
    )

    assert len(left) == 4
    for left_item, right_item in zip(left, right):
        assert left_item[3] == right_item[3]
        assert all(
            torch.equal(left_tensor, right_tensor)
            for left_tensor, right_tensor in zip(left_item[:3], right_item[:3])
        )
        assert float(left_item[1].sum().item()) > 0.0
        assert float(left_item[2].sum().item()) > 0.0


def test_grounded_comparison_accepts_exact_cross_modal_preservation() -> None:
    baseline = _arm()
    variant = deepcopy(baseline)
    variant["throughput_ticks_per_second"] = 60.0
    state = {
        "W_tv": torch.ones(2, 2),
        "visual_confidence": torch.ones(2),
    }

    result = compare_grounded_arms(baseline, variant, state, state)

    assert result["quality_preserved"] is True
    assert result["grounding_quality_preserved"] is True
    assert result["promotion_eligible"] is True
    assert result["grounding_gates"]["cross_modal_state_exact"] is True


def test_grounded_comparison_rejects_cross_modal_state_drift() -> None:
    baseline = _arm()
    variant = deepcopy(baseline)
    variant["throughput_ticks_per_second"] = 60.0
    baseline_state = {"W_tv": torch.zeros(2, 2)}
    variant_state = {"W_tv": torch.ones(2, 2)}

    result = compare_grounded_arms(
        baseline,
        variant,
        baseline_state,
        variant_state,
    )

    assert result["grounding_quality_preserved"] is False
    assert result["promotion_eligible"] is False
    assert result["grounding_gates"]["cross_modal_state_exact"] is False
