from __future__ import annotations

import torch

from marulho.evaluation.language_modular_society_falsification import (
    ModularSocietyFalsificationConfig,
    _build_model,
    build_matched_schedule,
    modular_society_decision,
)


def test_v3_schedule_is_exact_and_reproducible() -> None:
    first = build_matched_schedule(
        step_count=20,
        relation_fraction=0.2,
        relation_batch_count=5,
        general_batch_counts=(9, 9),
        seed=17,
    )
    assert first == build_matched_schedule(
        step_count=20,
        relation_fraction=0.2,
        relation_batch_count=5,
        general_batch_counts=(9, 9),
        seed=17,
    )
    assert sum(kind == "relation" for kind, _ in first) == 4
    assert sum(kind == "general_0" for kind, _ in first) == 8
    assert sum(kind == "general_1" for kind, _ in first) == 8


def test_v3_society_controls_start_from_identical_parameters() -> None:
    config = ModularSocietyFalsificationConfig(
        sequence_length=16,
        cell_width=32,
        cell_layers=1,
        attention_heads=4,
        event_interval=4,
        message_dim=8,
    )
    torch.manual_seed(23)
    no_message = _build_model(
        "learned_no_message", vocab_size=96, config=config
    ).state_dict()
    torch.manual_seed(23)
    real = _build_model("learned_real", vocab_size=96, config=config).state_dict()
    assert no_message.keys() == real.keys()
    for key in no_message:
        torch.testing.assert_close(no_message[key], real[key])


def _arm(name: str, *, loss: float, free: float, tokens: int = 16_785_792) -> dict:
    return {
        "name": name,
        "processed_tokens": tokens,
        "heldout": {"heldout_loss": loss},
        "relation": {"generation_exact_accuracy": free},
    }


def test_v3_decision_scales_only_real_communication_win() -> None:
    arms = (
        _arm("monolith", loss=4.00, free=0.20),
        _arm("average_no_message", loss=4.01, free=0.20),
        _arm("learned_no_message", loss=4.02, free=0.20),
        _arm("learned_shuffled", loss=4.03, free=0.21),
        _arm("learned_real", loss=4.00, free=0.24),
    )
    assert modular_society_decision(arms) == (
        "scale_v3_real_communication_to_64m_and_unseen_generation"
    )


def test_v3_decision_preserves_independent_cell_result_without_bus_claim() -> None:
    arms = (
        _arm("monolith", loss=4.00, free=0.20),
        _arm("average_no_message", loss=3.99, free=0.23),
        _arm("learned_no_message", loss=4.02, free=0.20),
        _arm("learned_shuffled", loss=4.03, free=0.20),
        _arm("learned_real", loss=4.02, free=0.20),
    )
    assert modular_society_decision(arms) == (
        "redesign_v3_message_bus_keep_independent_cell_result"
    )


def test_v3_decision_rejects_communication_that_harms_base_quality() -> None:
    arms = (
        _arm("monolith", loss=4.00, free=0.40),
        _arm("average_no_message", loss=4.04, free=0.20),
        _arm("learned_no_message", loss=4.05, free=0.20),
        _arm("learned_shuffled", loss=4.06, free=0.20),
        _arm("learned_real", loss=4.03, free=0.25),
    )
    assert modular_society_decision(arms) == (
        "redesign_v3_cell_capacity_before_scaling_communication"
    )


def test_v3_decision_retires_no_gain_and_labels_short_run_smoke() -> None:
    arms = (
        _arm("monolith", loss=4.00, free=0.20),
        _arm("average_no_message", loss=4.01, free=0.20),
        _arm("learned_no_message", loss=4.02, free=0.20),
        _arm("learned_shuffled", loss=4.02, free=0.20),
        _arm("learned_real", loss=4.01, free=0.21),
    )
    assert modular_society_decision(arms) == (
        "retire_v3_modular_society_no_coordination_or_quality_gain"
    )
    smoke = tuple({**row, "processed_tokens": 82_944} for row in arms)
    assert modular_society_decision(smoke) == "incomplete_v3_mechanism_smoke"
