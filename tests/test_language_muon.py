from __future__ import annotations

import pytest
import torch
from torch import nn

from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel
from marulho.training.language_muon import (
    MarulhoMuon,
    build_language_muon,
    newton_schulz_zeroth_power,
    warm_language_muon_orthogonalizer_shapes,
)


def test_newton_schulz_supports_batched_tall_and_wide_matrices() -> None:
    torch.manual_seed(41)
    for shape in ((3, 12, 5), (3, 5, 12)):
        update = torch.randn(shape)
        result = newton_schulz_zeroth_power(update, steps=5)
        assert result.shape == update.shape
        assert result.dtype == torch.bfloat16
        assert torch.isfinite(result).all()
        singular_values = torch.linalg.svdvals(result.float())
        assert float(singular_values.min()) > 0.35
        assert float(singular_values.max()) < 1.7


def test_newton_schulz_batch_matches_individual_matrices() -> None:
    torch.manual_seed(43)
    update = torch.randn(4, 8, 6)
    batched = newton_schulz_zeroth_power(update, steps=5)
    individual = torch.stack(
        [newton_schulz_zeroth_power(row, steps=5) for row in update]
    )
    torch.testing.assert_close(batched.float(), individual.float(), atol=0.0, rtol=0.0)


def test_language_muon_assigns_only_hidden_matrices_to_muon() -> None:
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=64,
            embedding_dim=32,
            state_dim=32,
            state_layers=2,
            attention_heads=4,
            transformer_context_length=16,
        )
    )
    optimizer, report = build_language_muon(
        model,
        learning_rate=1.0e-3,
        weight_decay=0.1,
        compile_orthogonalizer=False,
    )
    assert isinstance(optimizer, MarulhoMuon)
    muon_names = set(report["muon_parameter_names"])
    assert muon_names == {
        "state_block.layers.0.attention.qkv.weight",
        "state_block.layers.0.attention.output.weight",
        "state_block.layers.0.gate_up.weight",
        "state_block.layers.0.down.weight",
        "state_block.layers.1.attention.qkv.weight",
        "state_block.layers.1.attention.output.weight",
        "state_block.layers.1.gate_up.weight",
        "state_block.layers.1.down.weight",
    }
    assert "token_embedding.weight" in report["adamw_fallback_parameter_names"]
    assert report["muon_parameter_count"] + report[
        "adamw_fallback_parameter_count"
    ] == sum(parameter.numel() for parameter in model.parameters())


def test_language_muon_updates_every_trainable_parameter() -> None:
    torch.manual_seed(47)
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=64,
            embedding_dim=32,
            state_dim=32,
            state_layers=1,
            attention_heads=4,
            transformer_context_length=8,
        )
    )
    optimizer, _ = build_language_muon(
        model,
        learning_rate=1.0e-3,
        weight_decay=0.1,
        compile_orthogonalizer=False,
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    input_ids = torch.randint(0, 64, (4, 8))
    targets = torch.randint(0, 64, (4, 8))
    loss = model.next_token_loss(input_ids, targets)["loss"]
    loss.backward()
    optimizer.step()
    assert all(
        not torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
    )
    hidden = model.state_block.layers[0].attention.qkv.weight
    embedding = model.token_embedding.weight
    assert set(optimizer.state[hidden]) == {"momentum_buffer"}
    assert set(optimizer.state[embedding]) == {
        "step",
        "first_moment",
        "second_moment",
    }


def test_per_head_muon_partitions_combined_qkv_without_changing_state_shape() -> None:
    torch.manual_seed(53)
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=64,
            embedding_dim=32,
            state_dim=32,
            state_layers=2,
            attention_heads=4,
            transformer_context_length=8,
        )
    )
    optimizer, report = build_language_muon(
        model,
        learning_rate=1.0e-3,
        weight_decay=0.1,
        compile_orthogonalizer=False,
        per_head_attention_qkv=True,
    )
    expected_names = {
        "state_block.layers.0.attention.qkv.weight",
        "state_block.layers.1.attention.qkv.weight",
    }
    assert report["per_head_attention_qkv"] is True
    assert set(report["row_partitioned_parameter_names"]) == expected_names
    assert set(report["row_partition_count_by_parameter"].values()) == {12}

    qkv = model.state_block.layers[0].attention.qkv.weight
    original_shape = qkv.shape
    input_ids = torch.randint(0, 64, (4, 8))
    targets = torch.randint(0, 64, (4, 8))
    model.next_token_loss(input_ids, targets)["loss"].backward()
    optimizer.step()
    assert qkv.shape == original_shape
    assert torch.isfinite(qkv).all()
    assert optimizer._matrix_row_partitions[id(qkv)] == 12


def test_per_head_muon_matches_independent_partition_orthogonalization() -> None:
    torch.manual_seed(59)
    parameter = nn.Parameter(torch.randn(12, 4))
    optimizer = MarulhoMuon(
        muon_parameters=[parameter],
        adamw_parameters=[],
        learning_rate=1.0e-3,
        weight_decay=0.0,
        momentum=0.0,
        nesterov=False,
        compile_orthogonalizer=False,
        matrix_row_partitions={parameter: 3},
    )
    gradient = torch.randn_like(parameter)
    parameter.grad = gradient.clone()
    before = parameter.detach().clone()
    expected_direction = torch.cat(
        [
            newton_schulz_zeroth_power(part, steps=5)
            for part in gradient.reshape(3, 4, 4)
        ],
        dim=0,
    ).to(dtype=parameter.dtype)
    expected = before - (1.0e-3 * 0.2 * (12.0**0.5)) * expected_direction
    optimizer.step()
    torch.testing.assert_close(parameter, expected, atol=0.0, rtol=0.0)


def test_language_muon_rejects_overlapping_groups() -> None:
    parameter = nn.Parameter(torch.randn(4, 4))
    with pytest.raises(ValueError, match="disjoint"):
        MarulhoMuon(
            muon_parameters=[parameter],
            adamw_parameters=[parameter],
            learning_rate=1.0e-3,
        )


def test_muon_orthogonalizer_warmup_requires_cuda() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        warm_language_muon_orthogonalizer_shapes(
            [(8, 8)], device=torch.device("cpu")
        )
