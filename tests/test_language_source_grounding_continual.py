from marulho.evaluation.language_source_grounding_continual import (
    SourceGroundingContinualConfig,
    select_v48_candidate,
)


def _row(*, intact: float, gain: float, relation: float = 0.88, loss: float = 3.6):
    return {
        "source_grounding": {
            "valid": True,
            "intact_source": {"exact_answer_accuracy": intact},
            "intact_gain_over_stronger_control": gain,
        },
        "all_parameters_received_final_gradient": True,
        "processed_tokens": 4_193_280,
        "relation": {"generation_exact_accuracy": relation},
        "heldout": {"heldout_loss": loss},
    }


def test_v48_selects_answer_weight_only_for_matched_superiority() -> None:
    candidate, decision, gates = select_v48_candidate(
        {
            "ordinary_causal": _row(intact=0.28, gain=0.20),
            "answer_weight4": _row(intact=0.36, gain=0.28),
        },
        baseline_general_loss=3.6,
        baseline_relation_accuracy=0.88,
        config=SourceGroundingContinualConfig(),
    )

    assert candidate == "answer_weight4"
    assert decision == "scale_v48_answer_objective_to_confirmation"
    assert gates["answer_weight4"]["checks"]["gain_over_ordinary_arm"]


def test_v48_prefers_ordinary_when_weighting_is_unnecessary() -> None:
    candidate, decision, _gates = select_v48_candidate(
        {
            "ordinary_causal": _row(intact=0.30, gain=0.20),
            "answer_weight4": _row(intact=0.32, gain=0.22),
        },
        baseline_general_loss=3.6,
        baseline_relation_accuracy=0.88,
        config=SourceGroundingContinualConfig(),
    )

    assert candidate == "ordinary_causal"
    assert decision == "scale_v48_ordinary_objective_answer_weighting_unnecessary"


def test_v48_retires_objective_only_repair_when_retention_fails() -> None:
    candidate, decision, gates = select_v48_candidate(
        {
            "ordinary_causal": _row(intact=0.05, gain=0.04),
            "answer_weight4": _row(intact=0.40, gain=0.30, relation=0.70),
        },
        baseline_general_loss=3.6,
        baseline_relation_accuracy=0.88,
        config=SourceGroundingContinualConfig(),
    )

    assert candidate is None
    assert decision == "retire_v48_objective_only_grounding_repair"
    assert not gates["answer_weight4"]["checks"]["relation_retention"]
