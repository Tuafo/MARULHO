from __future__ import annotations

import math

import torch

from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel
from marulho.training.language_ngpt import (
    MarulhoNormalizedLanguageModel,
    NormalizedTransformerConfig,
)


def _config(**overrides) -> NormalizedTransformerConfig:
    values = {
        "vocab_size": 96,
        "width": 32,
        "layers": 2,
        "attention_heads": 4,
        "hidden_width": 64,
        "context_length": 16,
    }
    values.update(overrides)
    return NormalizedTransformerConfig(**values)


def test_normalized_candidate_is_parameter_matched_to_frozen_baseline() -> None:
    baseline = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=8192,
            embedding_dim=512,
            state_dim=512,
            state_layers=4,
            attention_heads=8,
            transformer_context_length=72,
            transformer_mlp_ratio=4.0,
            tie_embeddings=True,
        )
    )
    candidate = MarulhoNormalizedLanguageModel(
        NormalizedTransformerConfig(vocab_size=8192)
    )
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())

    assert baseline_parameters == 20_976_128
    assert candidate_parameters == 20_988_288
    assert abs(candidate_parameters - baseline_parameters) / baseline_parameters < 0.001
    assert candidate.lm_head.weight.data_ptr() != candidate.token_embedding.weight.data_ptr()


def test_normalized_scalars_have_paper_initial_values() -> None:
    model = MarulhoNormalizedLanguageModel(_config())
    base_scale = 1.0 / math.sqrt(float(model.normalized_config.width))

    assert torch.allclose(model.logit_scale / base_scale, torch.ones(96))
    for block in model.state_block.layers:
        assert torch.allclose(
            block.attention_alpha * (block.alpha_init / block.base_scale),
            torch.full((32,), 0.05),
        )
        assert torch.allclose(
            block.mlp_alpha * (block.alpha_init / block.base_scale),
            torch.full((32,), 0.05),
        )
        assert torch.allclose(
            block.attention.sqk / block.attention.base_scale,
            torch.ones(32),
        )
        assert torch.allclose(
            block.suv * math.sqrt(float(block.width)),
            torch.full((128,), math.sqrt(32.0)),
        )


def test_normalized_projection_restores_every_matrix_direction() -> None:
    torch.manual_seed(5)
    model = MarulhoNormalizedLanguageModel(_config())
    with torch.no_grad():
        for parameter, _dimension in model._hyperspherical_projection_spec():
            parameter.mul_(torch.rand((), dtype=parameter.dtype) + 0.2)

    model.post_optimizer_step()
    evidence = model.hyperspherical_weight_evidence()

    assert evidence["projected_matrix_count"] == 14.0
    assert evidence["maximum_unit_norm_error"] < 1.0e-5


def test_normalized_candidate_is_causal_streaming_equivalent_and_differentiable() -> None:
    torch.manual_seed(7)
    model = MarulhoNormalizedLanguageModel(_config()).eval()
    first = torch.tensor([[1, 8, 9, 10, 11, 12]], dtype=torch.long)
    second = torch.tensor([[1, 8, 9, 44, 45, 46]], dtype=torch.long)

    first_logits = model(first, collect_telemetry=False)["logits"]
    second_logits = model(second, collect_telemetry=False)["logits"]
    assert torch.allclose(first_logits[:, :3], second_logits[:, :3], atol=1.0e-6)

    with torch.no_grad():
        full = model(first, collect_telemetry=False)["logits"]
        state = None
        steps = []
        for index in range(int(first.shape[1])):
            result = model.forward_step(
                first[:, index],
                state,
                collect_telemetry=False,
            )
            state = result["state"]
            steps.append(result["logits"])
        streamed = torch.cat(steps, dim=1)
        hidden, _state, telemetry = model.state_block(
            model.token_embedding(first),
            collect_telemetry=False,
        )

    assert torch.allclose(streamed, full, atol=2.0e-5, rtol=1.0e-5)
    assert torch.allclose(
        hidden.float().norm(p=2, dim=-1),
        torch.ones_like(hidden[..., 0]),
        atol=1.0e-5,
    )
    assert telemetry["state_core"] == "hyperspherical_transformer"

    model.train()
    loss = model.next_token_loss(first, torch.roll(first, -1, dims=1))["loss"]
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert torch.isfinite(loss)
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_normalized_candidate_uses_marulho_generation_protocol() -> None:
    torch.manual_seed(11)
    model = MarulhoNormalizedLanguageModel(_config()).eval()
    prompt = torch.tensor([1, 4, 7, 9], dtype=torch.long)

    first = model.generate(
        prompt,
        max_new_tokens=6,
        temperature=0.8,
        top_p=0.9,
        seed=23,
    )
    second = model.generate(
        prompt,
        max_new_tokens=6,
        temperature=0.8,
        top_p=0.9,
        seed=23,
    )

    assert torch.equal(first["generated_ids"], second["generated_ids"])
    assert first["external_llm_used"] is False
    assert first["generation_decode"]["surface"] == "marulho_normalized_decode_policy.v1"
    assert first["generation_decode"]["decode_strategy"] == "nucleus_sampling"
    assert first["generation_decode"]["kv_cache"] == "bounded_hyperspherical_layers"
