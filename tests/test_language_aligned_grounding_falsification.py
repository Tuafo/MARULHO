from marulho.evaluation.language_aligned_grounding_falsification import (
    AlignedGroundingGateConfig,
    aligned_grounding_gate,
)


def _row(*, intact: float, relation: float, loss: float):
    return {
        "source_grounding": {
            "valid": True,
            "intact_source": {"exact_answer_accuracy": intact},
            "intact_gain_over_stronger_control": 0.20,
        },
        "relation": {"generation_exact_accuracy": relation},
        "heldout": {"heldout_loss": loss},
        "processed_tokens": 4_193_280,
        "all_parameters_received_final_gradient": True,
    }


def test_v52_gate_separates_capability_from_retention() -> None:
    gate = aligned_grounding_gate(
        _row(intact=0.30, relation=0.70, loss=3.30),
        baseline_general_loss=3.15,
        baseline_relation_accuracy=0.89,
        config=AlignedGroundingGateConfig(),
    )
    assert gate["capability_passed"]
    assert not gate["retention_passed"]
    assert not gate["passed"]


def test_v52_gate_requires_gain_over_v48() -> None:
    gate = aligned_grounding_gate(
        _row(intact=0.25, relation=0.89, loss=3.15),
        baseline_general_loss=3.15,
        baseline_relation_accuracy=0.89,
        config=AlignedGroundingGateConfig(),
    )
    assert not gate["capability_passed"]
    assert not gate["checks"]["minimum_gain_over_v48"]
