from __future__ import annotations

import pytest
import torch

from marulho.evaluation.language_hashed_micro_expert_durability import (
    ARM_NAMES,
    HashedMicroExpertDurabilityConfig,
    _build_model,
    _common_parameter_hash,
    checkpoint_reproduction_audit,
    hashed_micro_expert_durability_decision,
)
from marulho.evaluation.language_matched_support import build_matched_schedule


def _arms(
    values: dict[str, tuple[float, float]] | None = None,
    *,
    tokens: int = 67_108_864,
    candidate_tps: float = 90.0,
) -> list[dict]:
    selected = {
        "transformer": (4.0, 0.20),
        "shared_only": (4.0, 0.20),
        "token_hash": (4.0, 0.20),
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


def test_durability_promotes_only_a_joint_hash_win() -> None:
    arms = _arms(
        {
            "shared_only": (3.99, 0.21),
            "token_hash": (3.90, 0.30),
        }
    )
    assert hashed_micro_expert_durability_decision(arms) == (
        "promote_v11_hash_for_checkpoint_and_unseen_generation"
    )


def test_durability_rejects_local_throughput_collapse() -> None:
    arms = _arms(
        {
            "shared_only": (3.99, 0.21),
            "token_hash": (3.90, 0.30),
        },
        candidate_tps=40.0,
    )
    assert hashed_micro_expert_durability_decision(arms) == (
        "redesign_v11_durable_quality_but_local_throughput_collapse"
    )


def test_durability_keeps_shared_behavior_margin_unresolved() -> None:
    arms = _arms(
        {
            "shared_only": (4.02, 0.29),
            "token_hash": (3.90, 0.30),
        }
    )
    assert hashed_micro_expert_durability_decision(arms) == (
        "redesign_v11_durable_transformer_win_shared_behavior_unresolved"
    )


def test_durability_separates_loss_and_behavior_signals() -> None:
    loss_only = _arms({"token_hash": (3.90, 0.20)})
    assert hashed_micro_expert_durability_decision(loss_only) == (
        "redesign_v11_durable_loss_without_joint_behavior"
    )
    behavior_only = _arms({"token_hash": (4.00, 0.30)})
    assert hashed_micro_expert_durability_decision(behavior_only) == (
        "redesign_v11_durable_behavior_without_loss"
    )


def test_durability_handles_retire_smoke_and_incomplete() -> None:
    arms = _arms()
    assert hashed_micro_expert_durability_decision(arms) == (
        "retire_v11_hashed_micro_experts_not_durable"
    )
    short = _arms(tokens=20_736)
    assert hashed_micro_expert_durability_decision(short) == (
        "incomplete_v11_mechanism_smoke"
    )
    assert hashed_micro_expert_durability_decision(short[:-1]) == (
        "incomplete_v11_hashed_micro_expert_comparison"
    )


def test_durability_runner_common_initialization_hash_matches() -> None:
    config = HashedMicroExpertDurabilityConfig(
        sequence_length=16,
        batch_size=4,
        width=32,
        layers=2,
        attention_heads=4,
        baseline_hidden_width=64,
        shared_hidden_width=32,
        expert_layer_index=1,
        expert_pool_size=64,
        routing_heads=2,
        experts_per_head=2,
    )
    torch.manual_seed(83)
    dense = _build_model("transformer", vocab_size=96, config=config)
    torch.manual_seed(83)
    candidate = _build_model(
        "hashed_micro_experts",
        vocab_size=96,
        config=config,
    )
    assert _common_parameter_hash(
        dense,
        architecture="transformer",
        expert_layer_index=1,
    ) == _common_parameter_hash(
        candidate,
        architecture="hashed_micro_experts",
        expert_layer_index=1,
    )


def test_checkpoint_reproduction_reapplies_joint_gate_not_bit_exact_metrics() -> None:
    qualified = _arms(
        {
            "transformer": (3.895067, 0.191406),
            "shared_only": (3.908809, 0.257812),
            "token_hash": (3.874665, 0.359375),
        },
        candidate_tps=125_210.0,
    )
    for row in qualified:
        row["common_initialization_sha256"] = "stable-common-initialization"
        if row["name"] == "transformer":
            row["training"]["tokens_per_second"] = 130_415.0
    reproduced = dict(qualified[-1])
    reproduced["heldout"] = {"heldout_loss": 3.8748}
    reproduced["relation"] = {"generation_exact_accuracy": 0.292969}
    audit = checkpoint_reproduction_audit(
        reproduced,
        {
            "qualified_arms": {row["name"]: row for row in qualified},
        },
    )
    assert audit["fixed_joint_gate_reapplied"] is True
    assert audit["exact_gpu_trajectory_required"] is False
    assert audit["decision"] == (
        "promote_v11_hash_for_checkpoint_and_unseen_generation"
    )
    assert audit["heldout_loss_delta"] > 0.0
    assert audit["free_relation_accuracy_delta"] < 0.0


def test_checkpoint_reproduction_rejects_lost_behavior_margin() -> None:
    qualified = _arms(
        {
            "transformer": (3.895067, 0.191406),
            "shared_only": (3.908809, 0.257812),
            "token_hash": (3.874665, 0.359375),
        }
    )
    for row in qualified:
        row["common_initialization_sha256"] = "stable-common-initialization"
    reproduced = dict(qualified[-1])
    reproduced["heldout"] = {"heldout_loss": 3.8748}
    reproduced["relation"] = {"generation_exact_accuracy": 0.269531}
    try:
        checkpoint_reproduction_audit(
            reproduced,
            {"qualified_arms": {row["name"]: row for row in qualified}},
        )
    except RuntimeError as exc:
        assert "did not retain" in str(exc)
    else:
        raise AssertionError("Lost behavior margin should reject checkpoint")


def test_checkpoint_reproduction_rejects_token_or_initialization_drift() -> None:
    qualified = _arms(
        {
            "transformer": (3.895067, 0.191406),
            "shared_only": (3.908809, 0.257812),
            "token_hash": (3.874665, 0.359375),
        }
    )
    for row in qualified:
        row["common_initialization_sha256"] = "stable-common-initialization"
    record = {"qualified_arms": {row["name"]: row for row in qualified}}
    token_drift = {**qualified[-1], "processed_tokens": 67_108_863}
    initialization_drift = {
        **qualified[-1],
        "common_initialization_sha256": "different-initialization",
    }
    for reproduced, message in (
        (token_drift, "token count mismatch"),
        (initialization_drift, "initialization mismatch"),
    ):
        try:
            checkpoint_reproduction_audit(reproduced, record)
        except RuntimeError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("Qualification identity drift should be rejected")


def test_general_only_schedule_never_reads_relation_batches() -> None:
    schedule = build_matched_schedule(
        step_count=12,
        relation_fraction=0.0,
        relation_batch_count=0,
        general_batch_counts=(3, 4),
        seed=97,
    )
    assert len(schedule) == 12
    assert all(kind.startswith("general_") for kind, _index in schedule)


def test_schedule_requires_relation_batches_only_when_scheduled() -> None:
    with pytest.raises(ValueError, match="Scheduled relation"):
        build_matched_schedule(
            step_count=4,
            relation_fraction=0.25,
            relation_batch_count=0,
            general_batch_counts=(2, 2),
            seed=101,
        )
