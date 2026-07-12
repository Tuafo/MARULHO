from __future__ import annotations

from marulho.evaluation.language_general_context_falsification import (
    ADVANCE_DECISION,
    INVALID_DECISION,
    RETIRE_DECISION,
    GeneralContextFalsificationConfig,
    arm_shape,
    build_model,
    select_v30_candidate,
    v30_decision,
)


def _row(loss: float, *, gradients: bool = True) -> dict:
    return {
        "all_parameters_received_final_gradient": gradients,
        "common_context_heldout": {"heldout_loss": loss},
    }


def test_v30_arms_match_parameters_tokens_per_step_and_initial_shape() -> None:
    config = GeneralContextFalsificationConfig()
    short_shape = arm_shape("general72", config)
    long_shape = arm_shape("general256", config)
    assert short_shape[0] * short_shape[1] == 2_304
    assert long_shape[0] * long_shape[1] == 2_304
    short = build_model(vocab_size=8192, context_length=72, config=config)
    long = build_model(vocab_size=8192, context_length=256, config=config)
    assert sum(parameter.numel() for parameter in short.parameters()) == 20_976_128
    assert sum(parameter.numel() for parameter in long.parameters()) == 20_976_128
    assert set(short.state_dict()) == set(long.state_dict())
    assert all(
        short.state_dict()[name].shape == long.state_dict()[name].shape
        for name in short.state_dict()
    )


def test_v30_selection_prefers_short_unless_long_has_material_gain() -> None:
    config = GeneralContextFalsificationConfig()
    baseline = 4.10
    assert select_v30_candidate(
        {"general72": _row(4.00), "general256": _row(3.99)},
        baseline_common_loss=baseline,
        config=config,
    ) == "general72"
    assert select_v30_candidate(
        {"general72": _row(4.00), "general256": _row(3.97)},
        baseline_common_loss=baseline,
        config=config,
    ) == "general256"
    assert select_v30_candidate(
        {"general72": _row(4.08), "general256": _row(4.04)},
        baseline_common_loss=baseline,
        config=config,
    ) == "general256"
    assert select_v30_candidate(
        {"general72": _row(4.08), "general256": _row(4.07)},
        baseline_common_loss=baseline,
        config=config,
    ) is None


def test_v30_decision_requires_quality_gradient_and_checkpoint_truth() -> None:
    config = GeneralContextFalsificationConfig()
    winning = {"general72": _row(4.00), "general256": _row(3.99)}
    assert v30_decision(
        winning,
        baseline_common_loss=4.10,
        config=config,
        checkpoint_fidelity_passed=True,
    ) == ADVANCE_DECISION
    assert v30_decision(
        winning,
        baseline_common_loss=4.10,
        config=config,
        checkpoint_fidelity_passed=False,
    ) == INVALID_DECISION
    assert v30_decision(
        {"general72": _row(4.08), "general256": _row(4.07)},
        baseline_common_loss=4.10,
        config=config,
        checkpoint_fidelity_passed=True,
    ) == RETIRE_DECISION
    assert v30_decision(
        {"general72": _row(4.00, gradients=False), "general256": _row(3.99)},
        baseline_common_loss=4.10,
        config=config,
        checkpoint_fidelity_passed=True,
    ) == INVALID_DECISION
