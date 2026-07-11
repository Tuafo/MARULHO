from __future__ import annotations

import torch

from marulho.evaluation.language_modular_workspace_falsification import (
    ModularWorkspaceFalsificationConfig,
    _build_model,
    build_matched_schedule,
    modular_workspace_decision,
)


def test_v4_schedule_is_exact_and_reproducible() -> None:
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


def test_v4_workspace_controls_start_from_identical_parameters() -> None:
    config = ModularWorkspaceFalsificationConfig(
        sequence_length=16,
        shared_width=32,
        shared_layers_per_stage=1,
        shared_attention_heads=4,
        cell_count=3,
        cell_width=24,
        cell_layers_per_stage=1,
        cell_attention_heads=4,
        workspace_width=8,
    )
    torch.manual_seed(23)
    no_exchange = _build_model(
        "no_exchange", vocab_size=96, config=config
    ).state_dict()
    torch.manual_seed(23)
    real = _build_model("real", vocab_size=96, config=config).state_dict()
    assert no_exchange.keys() == real.keys()
    for key in no_exchange:
        torch.testing.assert_close(no_exchange[key], real[key])


def _arm(name: str, *, loss: float, free: float, tokens: int = 16_785_792) -> dict:
    return {
        "name": name,
        "processed_tokens": tokens,
        "heldout": {"heldout_loss": loss},
        "relation": {"generation_exact_accuracy": free},
    }


def test_v4_decision_scales_only_real_communication_win() -> None:
    arms = (
        _arm("monolith", loss=4.00, free=0.20),
        _arm("no_exchange", loss=4.02, free=0.20),
        _arm("shuffled", loss=4.03, free=0.21),
        _arm("real", loss=4.00, free=0.24),
    )
    assert modular_workspace_decision(arms) == (
        "scale_v4_real_workspace_to_64m_and_unseen_generation"
    )


def test_v4_decision_preserves_independent_cell_result_without_bus_claim() -> None:
    arms = (
        _arm("monolith", loss=4.00, free=0.20),
        _arm("no_exchange", loss=3.99, free=0.23),
        _arm("shuffled", loss=4.03, free=0.20),
        _arm("real", loss=4.02, free=0.20),
    )
    assert modular_workspace_decision(arms) == (
        "redesign_v4_exchange_keep_parallel_cell_result"
    )


def test_v4_decision_rejects_communication_that_harms_base_quality() -> None:
    arms = (
        _arm("monolith", loss=4.00, free=0.40),
        _arm("no_exchange", loss=4.05, free=0.20),
        _arm("shuffled", loss=4.06, free=0.20),
        _arm("real", loss=4.03, free=0.25),
    )
    assert modular_workspace_decision(arms) == (
        "redesign_v4_shared_capacity_before_scaling_workspace"
    )


def test_v4_decision_retires_no_gain_and_labels_short_run_smoke() -> None:
    arms = (
        _arm("monolith", loss=4.00, free=0.20),
        _arm("no_exchange", loss=4.02, free=0.20),
        _arm("shuffled", loss=4.02, free=0.20),
        _arm("real", loss=4.01, free=0.21),
    )
    assert modular_workspace_decision(arms) == (
        "retire_v4_modular_workspace_no_coordination_or_quality_gain"
    )
    smoke = tuple({**row, "processed_tokens": 82_944} for row in arms)
    assert modular_workspace_decision(smoke) == "incomplete_v4_mechanism_smoke"
