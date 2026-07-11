from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from marulho.training.language_hashed_micro_experts import (
    HASHED_MICRO_EXPERT_MODES,
    HashedMicroExpertConfig,
    MarulhoHashedMicroExpertLanguageModel,
)
from marulho.training.language_micro_experts import (
    MarulhoProductKeyMicroExpertLanguageModel,
    ProductKeyMicroExpertConfig,
)


def _config(**overrides) -> HashedMicroExpertConfig:
    values = {
        "vocab_size": 96,
        "width": 32,
        "layers": 2,
        "attention_heads": 4,
        "context_length": 16,
        "baseline_hidden_width": 64,
        "shared_hidden_width": 32,
        "expert_layer_index": 1,
        "expert_pool_size": 64,
        "routing_heads": 2,
        "experts_per_head": 2,
        "mode": "token_hash",
    }
    values.update(overrides)
    return HashedMicroExpertConfig(**values)


def _model(**overrides) -> MarulhoHashedMicroExpertLanguageModel:
    return MarulhoHashedMicroExpertLanguageModel(_config(**overrides))


@pytest.mark.parametrize("mode", HASHED_MICRO_EXPERT_MODES)
def test_hashed_micro_experts_are_causal_and_streaming_equivalent(mode: str) -> None:
    torch.manual_seed(59)
    model = _model(mode=mode).eval()
    first = torch.tensor([[1, 8, 9, 10, 11, 12, 13, 14]], dtype=torch.long)
    second = torch.tensor([[1, 8, 9, 44, 45, 46, 47, 48]], dtype=torch.long)
    with torch.no_grad():
        first_logits = model(first, collect_telemetry=False)["logits"]
        second_logits = model(second, collect_telemetry=False)["logits"]
        prompt = model(first[:, :3], collect_telemetry=False)
        state = prompt["state"]
        incremental = [prompt["logits"][:, -1]]
        for index in range(3, int(first.shape[1])):
            step = model.forward_step(
                first[:, index : index + 1],
                state,
                collect_telemetry=False,
            )
            state = step["state"]
            incremental.append(step["logits"][:, -1])
    assert torch.allclose(first_logits[:, :3], second_logits[:, :3], atol=1e-6)
    assert torch.allclose(
        torch.stack(incremental, dim=1),
        first_logits[:, 2:],
        atol=2e-5,
        rtol=1e-5,
    )


def test_pruned_hash_path_matches_v10_after_copying_surviving_tensors() -> None:
    torch.manual_seed(61)
    old = MarulhoProductKeyMicroExpertLanguageModel(
        ProductKeyMicroExpertConfig(
            vocab_size=96,
            width=32,
            layers=2,
            attention_heads=4,
            context_length=16,
            baseline_hidden_width=64,
            shared_hidden_width=32,
            expert_layer_index=1,
            expert_pool_size=64,
            retrieval_heads=2,
            experts_per_head=2,
            mode="token_hash",
        )
    ).eval()
    torch.manual_seed(67)
    pruned = _model().eval()
    old_state = old.state_dict()
    pruned_state = pruned.state_dict()
    for name, value in tuple(pruned_state.items()):
        if name in old_state and old_state[name].shape == value.shape:
            pruned_state[name] = old_state[name].detach().clone()
    pruned.load_state_dict(pruned_state, strict=True)
    pruned.set_hashed_micro_expert_mode("token_hash")
    input_ids = torch.randint(0, 96, (3, 12))
    with torch.no_grad():
        expected = old(input_ids, collect_telemetry=False)["logits"]
        actual = pruned(input_ids, collect_telemetry=False)["logits"]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("mode", HASHED_MICRO_EXPERT_MODES)
def test_hashed_modes_have_expected_expert_gradients(mode: str) -> None:
    torch.manual_seed(71)
    model = _model(mode=mode).train()
    input_ids = torch.randint(0, 96, (3, 9))
    target_ids = torch.randint(0, 96, (3, 9))
    logits = model(input_ids, collect_telemetry=False)["logits"]
    loss = F.cross_entropy(logits.reshape(-1, 96), target_ids.reshape(-1))
    loss.backward()
    report = model.final_gradient_report()
    if mode == "shared_only":
        assert report["expert_rows_with_nonzero_gradient"] == 0
    else:
        assert report["expert_rows_with_nonzero_gradient"] > 0


@pytest.mark.parametrize("mode", HASHED_MICRO_EXPERT_MODES)
def test_hashed_routing_is_unique_label_free_and_fixed_granularity(mode: str) -> None:
    torch.manual_seed(73)
    model = _model(mode=mode).eval()
    input_ids = torch.randint(0, 96, (4, 12))
    report = model.routing_report(input_ids)
    assert report["mode"] == mode
    assert report["route_assignment_count"] == 4 * 12 * 2 * 2
    assert report["active_experts_per_token"] == 4
    assert report["mean_duplicate_experts_per_token"] == 0.0
    assert report["router_uses_labels"] is False
    assert report["promotion_metric"] is False


def test_hashed_active_parameter_report_reflects_pruning() -> None:
    model = _model()
    report = model.active_parameter_report()
    assert report["shared_path_parameters"] == 3 * 32 * 32
    assert report["expert_pool_parameters"] == 2 * 64 * 32
    assert report["active_experts_per_token"] == 4
    assert report["candidate_to_baseline_multiply_ratio"] == pytest.approx(
        (3 * 32 * 32 + 4 * 2 * 32) / (3 * 32 * 64)
    )


def test_hashed_owned_generation_uses_candidate_path() -> None:
    torch.manual_seed(79)
    model = _model(context_length=24).eval()
    generated = model.generate(
        torch.tensor([1, 3, 5, 7], dtype=torch.long),
        max_new_tokens=4,
        eos_id=95,
    )
    assert generated["new_token_count"] >= 1
    assert generated["external_llm_used"] is False


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"width": 31}, "width must be even"),
        ({"expert_layer_index": 2}, "existing layer"),
        ({"expert_pool_size": 1}, "expert_pool_size"),
        ({"routing_heads": 0}, "routing_heads"),
        ({"experts_per_head": 65}, "experts_per_head"),
        ({"mode": "learned_router"}, "mode must be"),
    ],
)
def test_hashed_config_rejects_invalid_values(overrides, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _model(**overrides)
