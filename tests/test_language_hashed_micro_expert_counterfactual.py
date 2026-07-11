from __future__ import annotations

import torch

from marulho.evaluation.language_hashed_micro_expert_counterfactual import (
    TRAIN_GATE_DECISION,
    _audit_batches,
    counterfactual_route_decision,
)
from marulho.training.language_hashed_micro_experts import (
    HashedMicroExpertConfig,
    MarulhoHashedMicroExpertLanguageModel,
)
from marulho.training.language_model import LanguageBatch


def _model() -> MarulhoHashedMicroExpertLanguageModel:
    return MarulhoHashedMicroExpertLanguageModel(
        HashedMicroExpertConfig(
            vocab_size=96,
            width=32,
            layers=2,
            attention_heads=4,
            context_length=16,
            baseline_hidden_width=64,
            shared_hidden_width=32,
            expert_layer_index=1,
            expert_pool_size=64,
            routing_heads=2,
            experts_per_head=2,
            mode="token_hash",
        )
    )


def test_counterfactual_decision_only_admits_broad_read_only_opportunity() -> None:
    assert counterfactual_route_decision(
        mean_oracle_loss_improvement=0.03,
        fraction_regret_005=0.20,
        parameters_unchanged=True,
        forced_baseline_max_logit_delta=0.0,
    ) == TRAIN_GATE_DECISION
    assert counterfactual_route_decision(
        mean_oracle_loss_improvement=0.03,
        fraction_regret_005=0.05,
        parameters_unchanged=True,
        forced_baseline_max_logit_delta=0.0,
    ) == "redesign_v12_route_bank_no_broad_opportunity"
    assert counterfactual_route_decision(
        mean_oracle_loss_improvement=0.03,
        fraction_regret_005=0.20,
        parameters_unchanged=False,
        forced_baseline_max_logit_delta=0.0,
    ) == "invalid_counterfactual_audit_mutated_model"
    assert counterfactual_route_decision(
        mean_oracle_loss_improvement=0.03,
        fraction_regret_005=0.20,
        parameters_unchanged=True,
        forced_baseline_max_logit_delta=0.1,
    ) == "invalid_counterfactual_audit_forced_route_mismatch"


def test_counterfactual_batch_audit_is_read_only_equal_compute_and_label_safe() -> None:
    torch.manual_seed(107)
    model = _model().eval()
    batches = tuple(
        LanguageBatch(
            torch.randint(0, 96, (4, 8)),
            torch.randint(0, 96, (4, 8)),
        )
        for _ in range(2)
    )
    report = _audit_batches(
        model,
        batches,
        alternative_seed_offsets=(7, 13),
        precision="float32",
    )
    assert report["evaluated_token_count"] == 8
    assert report["forced_baseline_max_logit_delta"] == 0.0
    assert report["parameters_unchanged"] is True
    assert report["routing"]["active_experts_per_token_per_policy"] == 4
    assert report["routing"]["equal_active_compute"] is True
    assert report["routing"]["mean_duplicate_experts_within_policy_per_token"] == 0.0
    assert len(report["fixed_route_policies"]) == 3
    assert sum(
        row["selected_token_count"]
        for row in report["oracle_route_selection"]
    ) == 8
    assert report["anti_cheat"]["prediction_routes_use_targets"] is False
    assert report["anti_cheat"]["oracle_selection_uses_labels"] is True
    assert report["anti_cheat"]["oracle_selection_promotable"] is False
