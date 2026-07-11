import pytest
import torch

from marulho.training.language_depth_allocation import (
    DepthAllocationConfig,
    MarulhoDepthAllocatedLanguageModel,
    matching_common_parameter_names,
    mlp_parameter_count,
    total_parameter_count,
)


PROFILES = {
    "uniform": (128, 128, 128, 128),
    "early_heavy": (192, 160, 96, 64),
    "late_heavy": (64, 96, 160, 192),
}


def _model(profile: str, *, seed: int = 17) -> MarulhoDepthAllocatedLanguageModel:
    torch.manual_seed(seed)
    return MarulhoDepthAllocatedLanguageModel(
        DepthAllocationConfig(
            vocab_size=97,
            width=32,
            attention_heads=4,
            context_length=12,
            mlp_hidden_widths=PROFILES[profile],
            initialization_seed=991,
        )
    )


def test_depth_profiles_match_parameters_and_mlp_compute() -> None:
    models = {name: _model(name) for name in PROFILES}
    assert len({total_parameter_count(model) for model in models.values()}) == 1
    assert {
        mlp_parameter_count(32, profile) for profile in PROFILES.values()
    } == {49_152}


def test_depth_profiles_share_every_non_mlp_initial_tensor() -> None:
    models = {name: _model(name) for name in PROFILES}
    names = matching_common_parameter_names(models["uniform"])
    assert names
    reference = dict(models["uniform"].named_parameters())
    for model in (models["early_heavy"], models["late_heavy"]):
        candidate = dict(model.named_parameters())
        assert names == matching_common_parameter_names(model)
        for name in names:
            torch.testing.assert_close(candidate[name], reference[name], rtol=0, atol=0)


def test_depth_allocated_model_is_causal_and_streaming_equivalent() -> None:
    model = _model("late_heavy").eval()
    prefix = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    changed = prefix.clone()
    changed[:, -1] = 19
    with torch.no_grad():
        original = model(prefix)["logits"]
        altered = model(changed)["logits"]
    torch.testing.assert_close(original[:, :-1], altered[:, :-1], rtol=0, atol=0)

    state = None
    pieces = []
    with torch.no_grad():
        for token in prefix.unbind(dim=1):
            step = model.forward_step(token, state)
            pieces.append(step["logits"])
            state = step["state"]
    streamed = torch.cat(pieces, dim=1)
    torch.testing.assert_close(original, streamed, rtol=1e-5, atol=1e-5)


def test_depth_allocated_model_backpropagates_to_every_parameter() -> None:
    model = _model("early_heavy")
    inputs = torch.randint(0, 97, (3, 8))
    targets = torch.randint(0, 97, (3, 8))
    model.next_token_loss(inputs, targets)["loss"].backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_depth_allocation_telemetry_and_generation_are_owned() -> None:
    model = _model("late_heavy").eval()
    output = model(torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert output["telemetry"]["mlp_hidden_widths"] == [64, 96, 160, 192]
    assert output["telemetry"]["mlp_width_monotonic_direction"] == "nondecreasing"
    assert output["telemetry"]["external_llm_used"] is False
    generated = model.generate(
        torch.tensor([1, 2, 3], dtype=torch.long),
        max_new_tokens=3,
    )
    assert generated["generated_ids"].shape == (1, 6)
    assert generated["external_llm_used"] is False


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"mlp_hidden_widths": ()}, "at least one layer"),
        ({"mlp_hidden_widths": (16,)}, "at least the model width"),
        ({"attention_heads": 3}, "divisible"),
    ],
)
def test_depth_allocation_rejects_invalid_shapes(kwargs, match: str) -> None:
    values = {
        "vocab_size": 97,
        "width": 32,
        "attention_heads": 4,
        "context_length": 12,
        "mlp_hidden_widths": (128, 128),
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        MarulhoDepthAllocatedLanguageModel(DepthAllocationConfig(**values))
