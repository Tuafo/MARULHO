from __future__ import annotations

import pytest
import torch

from marulho.training.language_future_prediction import (
    FuturePredictionConfig,
    build_future_prediction_model,
    strip_future_prediction_heads,
)
from marulho.training.language_hashed_micro_experts import (
    HashedMicroExpertConfig,
    MarulhoHashedMicroExpertLanguageModel,
)


def _base() -> MarulhoHashedMicroExpertLanguageModel:
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


def test_future_prediction_heads_preserve_and_strip_from_inference() -> None:
    torch.manual_seed(109)
    base = _base().eval()
    input_ids = torch.randint(0, 96, (3, 12))
    expected = base(input_ids, collect_telemetry=False)["logits"].detach()
    model = build_future_prediction_model(base).eval()
    actual = model(input_ids, collect_telemetry=False)["logits"].detach()
    assert torch.equal(actual, expected)
    report = model.training_parameter_report()
    assert report["temporary_future_head_parameters"] == 3 * (32 + 32 * 32)
    assert report["inference_parameters"] == sum(
        value.numel() for value in base.parameters()
    )
    assert model.config.active_language_path == (
        "marulho_hashed_micro_experts_future_prediction_v13"
    )

    model.train()
    target_ids = torch.randint(0, 96, (3, 12))
    result = model.next_token_loss(
        input_ids,
        target_ids,
        collect_telemetry=False,
    )
    result["loss"].backward()
    assert result["loss_evidence"]["horizons"] == [2, 4, 8]
    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for name, parameter in model.named_parameters()
        if name.startswith("future_")
    )
    stripped = strip_future_prediction_heads(model).eval()
    model.eval()
    with torch.no_grad():
        expected_stripped = model(input_ids, collect_telemetry=False)["logits"]
        actual_stripped = stripped(input_ids, collect_telemetry=False)["logits"]
    assert torch.equal(actual_stripped, expected_stripped)
    assert all(not name.startswith("future_") for name in stripped.state_dict())


def test_future_prediction_loss_matches_declared_weighting() -> None:
    torch.manual_seed(113)
    model = build_future_prediction_model(_base()).eval()
    input_ids = torch.randint(0, 96, (2, 12))
    target_ids = torch.randint(0, 96, (2, 12))
    components = model.future_loss_components(input_ids, target_ids)
    expected = components["base_loss"] + 0.25 * torch.stack(
        components["future_losses"]
    ).mean()
    torch.testing.assert_close(components["loss"], expected)
    assert len(components["future_losses"]) == 3


@pytest.mark.parametrize(
    ("configuration", "match"),
    [
        (FuturePredictionConfig(horizons=()), "at least one"),
        (FuturePredictionConfig(horizons=(1, 2)), "at least two"),
        (FuturePredictionConfig(horizons=(4, 2)), "unique and sorted"),
        (FuturePredictionConfig(horizons=(2, 2)), "unique and sorted"),
        (FuturePredictionConfig(horizons=(2, 32)), "exceeds model context"),
        (FuturePredictionConfig(auxiliary_weight=0.0), "auxiliary_weight"),
        (FuturePredictionConfig(auxiliary_weight=1.1), "auxiliary_weight"),
    ],
)
def test_future_prediction_rejects_invalid_configuration(
    configuration: FuturePredictionConfig,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_future_prediction_model(_base(), configuration)
