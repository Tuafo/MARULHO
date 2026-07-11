from __future__ import annotations

from marulho.evaluation.language_muon_falsification import (
    ARM_NAMES,
    MuonFalsificationConfig,
    arm_learning_rate,
    arm_optimizer_kind,
    build_model,
    muon_decision,
    optimizer_comparison,
)


def _row(loss: float, free: float, *, gradients: bool = True) -> dict:
    return {
        "heldout": {"heldout_loss": loss},
        "relation": {"generation_exact_accuracy": free},
        "all_parameters_received_final_gradient": gradients,
    }


def test_v29_uses_one_exact_model_across_optimizer_arms() -> None:
    config = MuonFalsificationConfig()
    model = build_model(vocab_size=8192, config=config)
    assert sum(parameter.numel() for parameter in model.parameters()) == 20_976_128
    assert {arm_optimizer_kind(arm) for arm in ARM_NAMES} == {"adamw", "muon"}
    assert arm_learning_rate("adamw_3e4", config) == 3.0e-4
    assert arm_learning_rate("muon_3e4", config) == 3.0e-4
    assert arm_learning_rate("adamw_1e3", config) == 1.0e-3
    assert arm_learning_rate("muon_1e3", config) == 1.0e-3


def test_v29_comparison_selects_best_learning_rate_per_optimizer() -> None:
    arms = {
        "adamw_3e4": _row(4.10, 0.30),
        "adamw_1e3": _row(4.20, 0.40),
        "muon_3e4": _row(4.05, 0.31),
        "muon_1e3": _row(4.00, 0.33),
    }
    comparison = optimizer_comparison(arms)
    assert comparison is not None
    assert comparison["best_adamw_arm"] == "adamw_3e4"
    assert comparison["best_muon_arm"] == "muon_1e3"
    assert abs(comparison["muon_loss_gain"] - 0.10) < 1.0e-9
    assert abs(comparison["muon_free_relation_gain"] - 0.03) < 1.0e-9


def test_v29_decision_requires_joint_loss_and_free_generation() -> None:
    config = MuonFalsificationConfig()
    winning = {
        "adamw_3e4": _row(4.10, 0.30),
        "adamw_1e3": _row(4.20, 0.29),
        "muon_3e4": _row(4.08, 0.33),
        "muon_1e3": _row(4.00, 0.32),
    }
    assert muon_decision(
        winning,
        processed_tokens=16_777_216,
        parameter_delta_fraction=0.0,
        config=config,
    ) == "advance_v29_muon_to_unseen_generation"
    loss_only = {
        **winning,
        "muon_3e4": _row(4.08, 0.30),
        "muon_1e3": _row(4.00, 0.29),
    }
    assert muon_decision(
        loss_only,
        processed_tokens=16_777_216,
        parameter_delta_fraction=0.0,
        config=config,
    ) == "redesign_v29_disjoint_optimizer_signal"
    dominated = {
        **winning,
        "muon_3e4": _row(4.30, 0.20),
        "muon_1e3": _row(4.40, 0.20),
    }
    assert muon_decision(
        dominated,
        processed_tokens=16_777_216,
        parameter_delta_fraction=0.0,
        config=config,
    ) == "retire_v29_muon_no_joint_language_win"


def test_v29_decision_rejects_incomplete_evidence() -> None:
    config = MuonFalsificationConfig()
    rows = {
        "adamw_3e4": _row(4.1, 0.3),
        "adamw_1e3": _row(4.2, 0.3),
        "muon_3e4": _row(4.0, 0.4),
        "muon_1e3": _row(3.9, 0.4, gradients=False),
    }
    assert muon_decision(
        rows,
        processed_tokens=16_777_216,
        parameter_delta_fraction=0.0,
        config=config,
    ) == "invalid_v29_incomplete_gradient_coverage"
    assert muon_decision(
        {"adamw_3e4": rows["adamw_3e4"]},
        processed_tokens=16_777_216,
        parameter_delta_fraction=0.0,
        config=config,
    ) == "incomplete_v29_missing_optimizer_arm"
    assert muon_decision(
        {name: _row(4.0, 0.4) for name in ARM_NAMES},
        processed_tokens=1_000_000,
        parameter_delta_fraction=0.0,
        config=config,
    ) == "diagnostic_v29_below_durable_token_floor"
