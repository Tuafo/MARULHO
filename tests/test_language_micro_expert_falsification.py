from __future__ import annotations

import torch

from marulho.evaluation.language_micro_expert_falsification import (
    ARM_NAMES,
    MicroExpertFalsificationConfig,
    _build_model,
    _common_parameter_hash,
    micro_expert_decision,
)


def _arms(
    values: dict[str, tuple[float, float]] | None = None,
    *,
    tokens: int = 16_777_216,
    candidate_tps: float = 90.0,
) -> list[dict]:
    selected = {
        "transformer": (5.0, 0.10),
        "shared_only": (5.0, 0.10),
        "fixed_random": (5.0, 0.10),
        "token_hash": (5.0, 0.10),
        "learned_router": (5.0, 0.10),
    }
    selected.update(values or {})
    return [
        {
            "name": name,
            "processed_tokens": tokens,
            "heldout": {"heldout_loss": selected[name][0]},
            "relation": {"generation_exact_accuracy": selected[name][1]},
            "training": {
                "tokens_per_second": (
                    100.0 if name == "transformer" else candidate_tps
                )
            },
        }
        for name in ARM_NAMES
    ]


def test_decision_replicates_a_learned_router_that_beats_every_control() -> None:
    arms = _arms(
        {
            "shared_only": (4.98, 0.12),
            "fixed_random": (4.96, 0.13),
            "token_hash": (4.97, 0.14),
            "learned_router": (4.90, 0.20),
        }
    )
    assert micro_expert_decision(arms) == (
        "replicate_v10_learned_router_before_scale"
    )


def test_decision_rejects_a_quality_win_with_local_throughput_collapse() -> None:
    arms = _arms(
        {
            "shared_only": (4.98, 0.12),
            "fixed_random": (4.96, 0.13),
            "token_hash": (4.97, 0.14),
            "learned_router": (4.90, 0.20),
        },
        candidate_tps=40.0,
    )
    assert micro_expert_decision(arms) == (
        "redesign_v10_quality_gain_but_local_throughput_collapse"
    )


def test_decision_can_replicate_a_fixed_route_without_learning_claim() -> None:
    arms = _arms(
        {
            "shared_only": (4.99, 0.11),
            "fixed_random": (4.90, 0.20),
            "token_hash": (5.05, 0.08),
            "learned_router": (4.95, 0.11),
        }
    )
    assert micro_expert_decision(arms) == (
        "replicate_v10_fixed_random_without_learned_router_claim"
    )


def test_decision_separates_shared_path_from_micro_expert_claim() -> None:
    arms = _arms({"shared_only": (4.90, 0.20)})
    assert micro_expert_decision(arms) == (
        "replicate_v10_shared_path_without_micro_expert_claim"
    )


def test_decision_reports_disjoint_loss_and_behavior_signals() -> None:
    arms = _arms(
        {
            "fixed_random": (4.90, 0.10),
            "token_hash": (5.00, 0.20),
        }
    )
    assert micro_expert_decision(arms) == (
        "redesign_v10_disjoint_loss_and_behavior_signals"
    )


def test_decision_handles_smoke_incomplete_and_no_gain() -> None:
    arms = _arms()
    assert micro_expert_decision(arms) == (
        "retire_v10_micro_experts_no_quality_gain"
    )
    short = _arms(tokens=20_736)
    assert micro_expert_decision(short) == "incomplete_v10_mechanism_smoke"
    assert micro_expert_decision(short[:-1]) == (
        "incomplete_v10_micro_expert_comparison"
    )


def test_runner_common_hash_matches_dense_and_candidate_initialization() -> None:
    config = MicroExpertFalsificationConfig(
        sequence_length=16,
        batch_size=4,
        width=32,
        layers=2,
        attention_heads=4,
        baseline_hidden_width=64,
        shared_hidden_width=32,
        expert_layer_index=1,
        expert_pool_size=64,
        retrieval_heads=2,
        experts_per_head=2,
    )
    torch.manual_seed(43)
    dense = _build_model("transformer", vocab_size=96, config=config)
    torch.manual_seed(43)
    candidate = _build_model("micro_experts", vocab_size=96, config=config)
    assert _common_parameter_hash(
        dense,
        architecture="transformer",
        expert_layer_index=1,
    ) == _common_parameter_hash(
        candidate,
        architecture="micro_experts",
        expert_layer_index=1,
    )
