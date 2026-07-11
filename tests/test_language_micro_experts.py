from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from marulho.training.language_micro_experts import (
    MICRO_EXPERT_MODES,
    MarulhoProductKeyMicroExpertLanguageModel,
    ProductKeyMicroExpertConfig,
)
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


def _config(**overrides) -> ProductKeyMicroExpertConfig:
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
        "retrieval_heads": 2,
        "experts_per_head": 2,
        "dropout": 0.0,
        "mode": "learned_router",
    }
    values.update(overrides)
    return ProductKeyMicroExpertConfig(**values)


def _model(**overrides) -> MarulhoProductKeyMicroExpertLanguageModel:
    return MarulhoProductKeyMicroExpertLanguageModel(_config(**overrides))


@pytest.mark.parametrize("mode", MICRO_EXPERT_MODES)
def test_micro_experts_are_causal_and_streaming_equivalent(mode: str) -> None:
    torch.manual_seed(11)
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
    actual = torch.stack(incremental, dim=1)
    assert torch.allclose(actual, first_logits[:, 2:], atol=2e-5, rtol=1e-5)
    assert int(state["position"].item()) == int(first.shape[1])


def test_micro_expert_modes_share_parameter_objects_and_are_deterministic() -> None:
    torch.manual_seed(17)
    model = _model().eval()
    parameter_ids = {name: id(value) for name, value in model.named_parameters()}
    input_ids = torch.tensor([[2, 7, 11, 3, 19, 5]], dtype=torch.long)
    rows = {}
    with torch.no_grad():
        for mode in MICRO_EXPERT_MODES:
            model.set_micro_expert_mode(mode)
            first = model(input_ids, collect_telemetry=False)["logits"]
            second = model(input_ids, collect_telemetry=False)["logits"]
            torch.testing.assert_close(first, second)
            rows[mode] = first
            assert parameter_ids == {
                name: id(value) for name, value in model.named_parameters()
            }
    assert not torch.equal(rows["shared_only"], rows["fixed_random"])
    assert not torch.equal(rows["fixed_random"], rows["token_hash"])


def test_fixed_random_router_uses_immutable_initialization_buffers() -> None:
    torch.manual_seed(19)
    model = _model(mode="fixed_random").eval()
    input_ids = torch.tensor([[2, 7, 11, 3, 19, 5]], dtype=torch.long)
    with torch.no_grad():
        expected = model(input_ids, collect_telemetry=False)["logits"]
        layer = model.state_block.expert_layer
        layer.query_projection.weight.add_(100.0)
        layer.first_subkeys.sub_(50.0)
        layer.second_subkeys.mul_(20.0)
        actual = model(input_ids, collect_telemetry=False)["logits"]
        model.set_micro_expert_mode("learned_router")
        learned = model(input_ids, collect_telemetry=False)["logits"]
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(learned, expected)


@pytest.mark.parametrize("mode", MICRO_EXPERT_MODES)
def test_only_learned_mode_updates_router_and_routed_modes_update_experts(
    mode: str,
) -> None:
    torch.manual_seed(23)
    model = _model(mode=mode).train()
    input_ids = torch.randint(0, model.config.vocab_size, (3, 9))
    target_ids = torch.randint(0, model.config.vocab_size, (3, 9))
    output = model(input_ids, collect_telemetry=False)
    loss = F.cross_entropy(
        output["logits"].reshape(-1, model.config.vocab_size),
        target_ids.reshape(-1),
    )
    loss.backward()
    layer = model.state_block.expert_layer
    router_grad = sum(
        float(value.grad.abs().sum())
        for value in (
            layer.query_projection.weight,
            layer.first_subkeys,
            layer.second_subkeys,
        )
        if value.grad is not None
    )
    expert_grad = sum(
        float(value.grad.abs().sum())
        for value in (layer.expert_input.weight, layer.expert_output.weight)
        if value.grad is not None
    )
    shared_grad = sum(
        float(value.grad.abs().sum())
        for value in (layer.shared_gate_up.weight, layer.shared_down.weight)
        if value.grad is not None
    )
    assert torch.isfinite(loss)
    assert shared_grad > 0.0
    if mode == "learned_router":
        assert router_grad > 0.0
    else:
        assert router_grad == 0.0
    if mode == "shared_only":
        assert expert_grad == 0.0
    else:
        assert expert_grad > 0.0


@pytest.mark.parametrize("mode", MICRO_EXPERT_MODES)
def test_routing_report_is_label_free_and_has_fixed_granularity(mode: str) -> None:
    torch.manual_seed(29)
    model = _model(mode=mode).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (4, 12))
    report = model.routing_report(input_ids)
    assert report["mode"] == mode
    assert report["route_assignment_count"] == 4 * 12 * 2 * 2
    assert report["active_experts_per_token"] == 4
    assert 0 < report["used_expert_count"] <= 64
    assert 0.0 < report["used_expert_fraction"] <= 1.0
    assert report["routing_unevenness_kl_uniform"] >= 0.0
    assert report["mean_route_weight_entropy"] >= 0.0
    if mode == "token_hash":
        assert report["mean_duplicate_experts_per_token"] == 0.0
    assert report["router_uses_labels"] is False
    assert report["promotion_metric"] is False
    assert report["external_llm_used"] is False


def test_active_parameter_report_matches_the_configured_work() -> None:
    torch.manual_seed(31)
    model = _model()
    report = model.active_parameter_report()
    assert report["expert_pool_size"] == 64
    assert report["active_experts_per_token"] == 4
    assert report["shared_path_parameters"] == 3 * 32 * 32
    assert report["query_projection_parameters"] == 32 * 2 * 32
    assert report["product_subkey_parameters"] == 8 * 32
    assert report["expert_pool_parameters"] == 2 * 64 * 32
    assert report["candidate_to_baseline_multiply_ratio"] == pytest.approx(
        (3 * 32 * 32 + 32 * 2 * 32 + 2 * 8 * 32 + 4 * 2 * 32)
        / (3 * 32 * 64)
    )


def test_common_transformer_parameters_match_the_dense_initialization() -> None:
    micro = _config()
    dense_config = LanguageModelConfig(
        vocab_size=micro.vocab_size,
        embedding_dim=micro.width,
        state_dim=micro.width,
        state_layers=micro.layers,
        attention_heads=micro.attention_heads,
        transformer_context_length=micro.context_length,
        transformer_mlp_ratio=micro.baseline_hidden_width / micro.width,
        transformer_dropout=micro.dropout,
        tie_embeddings=True,
    )
    torch.manual_seed(37)
    dense = MarulhoLanguageModel(dense_config)
    torch.manual_seed(37)
    candidate = MarulhoProductKeyMicroExpertLanguageModel(micro)
    dense_parameters = dict(dense.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    common = set(dense_parameters) & set(candidate_parameters)
    assert common
    for name in common:
        torch.testing.assert_close(dense_parameters[name], candidate_parameters[name])


def test_micro_expert_owned_generation_uses_the_candidate_path() -> None:
    torch.manual_seed(41)
    model = _model(context_length=24).eval()
    generated = model.generate(
        torch.tensor([1, 3, 5, 7], dtype=torch.long),
        max_new_tokens=4,
        eos_id=95,
    )
    assert generated["new_token_count"] >= 1
    assert generated["external_llm_used"] is False
    assert generated["generation_decode"]["decode_strategy"] == "greedy_argmax"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"width": 31}, "width must be even"),
        ({"expert_layer_index": 2}, "existing layer"),
        ({"expert_pool_size": 63}, "perfect square"),
        ({"retrieval_heads": 0}, "retrieval_heads"),
        ({"experts_per_head": 9}, "experts_per_head"),
        ({"mode": "oracle"}, "mode must be"),
    ],
)
def test_micro_expert_config_rejects_invalid_values(overrides, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _model(**overrides)
