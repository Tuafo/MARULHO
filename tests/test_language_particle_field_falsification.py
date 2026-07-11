from __future__ import annotations

from marulho.evaluation.language_particle_field_falsification import (
    ARM_NAMES,
    ParticleFieldFalsificationConfig,
    build_arm_model,
    particle_field_decision,
)


def _row(loss: float, free: float, *, gradients: bool = True) -> dict:
    return {
        "heldout": {"heldout_loss": loss},
        "relation": {"generation_exact_accuracy": free},
        "all_parameters_received_final_gradient": gradients,
    }


def test_v28_default_models_are_parameter_matched() -> None:
    config = ParticleFieldFalsificationConfig()
    models = {
        arm: build_arm_model(arm, vocab_size=8192, config=config)
        for arm in ARM_NAMES
    }
    counts = {
        arm: sum(parameter.numel() for parameter in model.parameters())
        for arm, model in models.items()
    }
    assert counts == {
        "transformer": 20_976_128,
        "particle_field": 20_971_520,
    }
    assert abs(counts["particle_field"] - counts["transformer"]) / counts[
        "transformer"
    ] < config.maximum_parameter_delta_fraction


def test_v28_decision_requires_joint_loss_and_free_generation() -> None:
    config = ParticleFieldFalsificationConfig()
    winning = {
        "transformer": _row(4.0, 0.20),
        "particle_field": _row(3.99, 0.23),
    }
    assert particle_field_decision(
        winning,
        processed_tokens=16_777_216,
        parameter_delta_fraction=0.00022,
        config=config,
    ) == "advance_v28_particle_field_to_unseen_generation"
    loss_only = {
        "transformer": _row(4.0, 0.20),
        "particle_field": _row(3.99, 0.20),
    }
    assert particle_field_decision(
        loss_only,
        processed_tokens=16_777_216,
        parameter_delta_fraction=0.00022,
        config=config,
    ) == "redesign_v28_disjoint_loss_and_generation_signal"
    dominated = {
        "transformer": _row(4.0, 0.20),
        "particle_field": _row(4.1, 0.15),
    }
    assert particle_field_decision(
        dominated,
        processed_tokens=16_777_216,
        parameter_delta_fraction=0.00022,
        config=config,
    ) == "retire_v28_particle_field_no_joint_language_win"


def test_v28_decision_rejects_incomplete_evidence() -> None:
    config = ParticleFieldFalsificationConfig()
    rows = {
        "transformer": _row(4.0, 0.20),
        "particle_field": _row(3.9, 0.25, gradients=False),
    }
    assert particle_field_decision(
        rows,
        processed_tokens=16_777_216,
        parameter_delta_fraction=0.00022,
        config=config,
    ) == "invalid_v28_incomplete_gradient_coverage"
    assert particle_field_decision(
        {"transformer": rows["transformer"]},
        processed_tokens=16_777_216,
        parameter_delta_fraction=0.00022,
        config=config,
    ) == "incomplete_v28_missing_matched_arm"
    assert particle_field_decision(
        {
            "transformer": _row(4.0, 0.20),
            "particle_field": _row(3.9, 0.25),
        },
        processed_tokens=1_000_000,
        parameter_delta_fraction=0.00022,
        config=config,
    ) == "diagnostic_v28_below_durable_token_floor"
