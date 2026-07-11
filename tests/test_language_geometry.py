from __future__ import annotations

import math

import pytest
import torch

from marulho.evaluation.language_geometry import transformer_depth_geometry_report
from marulho.training.language_micro_experts import (
    MarulhoProductKeyMicroExpertLanguageModel,
    ProductKeyMicroExpertConfig,
)
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


def _dense() -> MarulhoLanguageModel:
    return MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=96,
            embedding_dim=32,
            state_dim=32,
            state_layers=2,
            attention_heads=4,
            transformer_context_length=16,
            transformer_mlp_ratio=2.0,
            tie_embeddings=True,
        )
    )


def _micro() -> MarulhoProductKeyMicroExpertLanguageModel:
    return MarulhoProductKeyMicroExpertLanguageModel(
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
        )
    )


@pytest.mark.parametrize("builder", (_dense, _micro))
def test_depth_geometry_is_finite_read_only_and_restores_mode(builder) -> None:
    torch.manual_seed(47)
    model = builder().train()
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    input_ids = torch.randint(0, 96, (4, 12))
    report = transformer_depth_geometry_report(model, input_ids, max_samples=32)
    assert model.training is True
    assert report["promotion_metric"] is False
    assert report["external_llm_used"] is False
    assert len(report["rows"]) == 3
    for index, row in enumerate(report["rows"]):
        assert row["depth"] == index
        assert row["sample_count"] == 32
        assert math.isfinite(row["participation_ratio"])
        assert math.isfinite(row["effective_rank"])
        assert math.isfinite(row["rms"])
        assert math.isfinite(row["mean_vector_norm"])
        if index == 0:
            assert row["adjacent_cosine"] is None
        else:
            assert math.isfinite(row["adjacent_cosine"])
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_depth_geometry_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="at least two"):
        transformer_depth_geometry_report(
            _dense(),
            torch.randint(0, 96, (1, 4)),
            max_samples=1,
        )


def test_depth_geometry_keeps_eigenspectrum_in_float32_under_autocast() -> None:
    torch.manual_seed(53)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        report = transformer_depth_geometry_report(
            _dense(),
            torch.randint(0, 96, (2, 8)),
            max_samples=16,
        )
    assert all(math.isfinite(row["effective_rank"]) for row in report["rows"])
