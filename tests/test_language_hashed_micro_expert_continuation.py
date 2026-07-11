from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from marulho.evaluation.language_hashed_micro_expert_continuation import (
    SAVE_DECISION,
    _validate_parent,
    general_continuation_decision,
)
from marulho.training.language_hashed_micro_experts import (
    HashedMicroExpertConfig,
    MarulhoHashedMicroExpertLanguageModel,
)


def _model(mode: str = "token_hash") -> MarulhoHashedMicroExpertLanguageModel:
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
            mode=mode,
        )
    )


def test_general_continuation_saves_only_material_complete_loss_gain() -> None:
    assert general_continuation_decision(
        heldout_loss_before=3.9,
        heldout_loss_after=3.7,
        processed_tokens=200,
        requested_tokens=200,
    ) == SAVE_DECISION
    assert general_continuation_decision(
        heldout_loss_before=3.9,
        heldout_loss_after=3.85,
        processed_tokens=200,
        requested_tokens=200,
    ) == "redesign_v11_general_continuation_weak_loss_gain"
    assert general_continuation_decision(
        heldout_loss_before=3.9,
        heldout_loss_after=3.91,
        processed_tokens=200,
        requested_tokens=200,
    ) == "retire_v11_general_continuation_no_loss_gain"
    assert general_continuation_decision(
        heldout_loss_before=3.9,
        heldout_loss_after=3.0,
        processed_tokens=199,
        requested_tokens=200,
    ) == "incomplete_v11_general_continuation"


def test_general_continuation_parent_must_be_qualified_owned_hash() -> None:
    metadata = {
        "decision": "promote_v11_hash_for_checkpoint_and_unseen_generation",
        "processed_tokens": 67_112_064,
        "external_llm_used": False,
    }
    assert _validate_parent(_model(), metadata) == 67_112_064
    with pytest.raises(ValueError, match="token_hash"):
        _validate_parent(_model(mode="shared_only"), metadata)
    with pytest.raises(ValueError, match="durability qualification"):
        _validate_parent(_model(), {**metadata, "decision": "unqualified"})
    with pytest.raises(ValueError, match="token count"):
        _validate_parent(_model(), {**metadata, "processed_tokens": 100})
    with pytest.raises(ValueError, match="MARULHO-owned"):
        _validate_parent(_model(), {**metadata, "external_llm_used": True})


def test_general_continuation_model_is_trainable_after_parent_validation() -> None:
    torch.manual_seed(103)
    model = _model().train()
    input_ids = torch.randint(0, 96, (2, 8))
    logits = model(input_ids, collect_telemetry=False)["logits"]
    loss = F.cross_entropy(logits.reshape(-1, 96), input_ids.reshape(-1))
    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.parameters()
    )
