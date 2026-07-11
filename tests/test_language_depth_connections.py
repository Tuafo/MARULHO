import pytest
import torch

from marulho.training.language_depth_connections import (
    DEPTH_CONNECTION_MODES,
    DepthConnectionConfig,
    MarulhoDepthConnectedLanguageModel,
    depth_geometry_report,
)
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


def _config(*, mode: str = "identity") -> DepthConnectionConfig:
    return DepthConnectionConfig(
        vocab_size=97,
        width=32,
        layers=4,
        attention_heads=4,
        hidden_width=128,
        context_length=12,
        mode=mode,
        random_seed=991,
    )


def _model(*, mode: str = "identity", seed: int = 17):
    torch.manual_seed(seed)
    return MarulhoDepthConnectedLanguageModel(_config(mode=mode))


def _baseline(*, seed: int = 17):
    torch.manual_seed(seed)
    return MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=97,
            embedding_dim=32,
            state_dim=32,
            state_layers=4,
            attention_heads=4,
            transformer_context_length=12,
            transformer_mlp_ratio=4.0,
            tie_embeddings=True,
        )
    )


def test_identity_mode_exactly_reproduces_transformer_logits() -> None:
    baseline = _baseline().eval()
    candidate = _model(mode="identity").eval()
    inputs = torch.randint(0, 97, (3, 10))
    with torch.no_grad():
        expected = baseline(inputs)["logits"]
        actual = candidate(inputs)["logits"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert sum(parameter.numel() for parameter in candidate.parameters()) == (
        sum(parameter.numel() for parameter in baseline.parameters()) + 14
    )


def test_all_modes_share_one_parameter_graph_and_produce_controls() -> None:
    model = _model()
    state_names = tuple(model.state_dict())
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    inputs = torch.randint(0, 97, (2, 8))
    outputs = {}
    with torch.no_grad():
        for mode in DEPTH_CONNECTION_MODES:
            model.set_depth_connection_mode(mode)
            outputs[mode] = model(inputs)["logits"]
            assert tuple(model.state_dict()) == state_names
            assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
    torch.testing.assert_close(outputs["identity"], outputs["learned_unconstrained"])
    assert not torch.equal(outputs["identity"], outputs["fixed_mean"])
    assert not torch.equal(outputs["fixed_mean"], outputs["fixed_random"])


def test_learned_depth_connections_are_causal_and_streaming_equivalent() -> None:
    model = _model(mode="learned_unconstrained").eval()
    with torch.no_grad():
        model.state_block.raw_depth_weights.copy_(
            torch.linspace(-0.08, 0.08, steps=14)
        )
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    changed = inputs.clone()
    changed[:, -1] = 19
    with torch.no_grad():
        full = model(inputs)["logits"]
        altered = model(changed)["logits"]
    torch.testing.assert_close(full[:, :-1], altered[:, :-1], rtol=0, atol=0)

    state = None
    pieces = []
    with torch.no_grad():
        for token in inputs.unbind(dim=1):
            result = model.forward_step(token, state)
            pieces.append(result["logits"])
            state = result["state"]
    streamed = torch.cat(pieces, dim=1)
    torch.testing.assert_close(full, streamed, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mode", ("learned_unconstrained", "learned_simplex"))
def test_learned_modes_backpropagate_to_every_parameter(mode: str) -> None:
    model = _model(mode=mode)
    inputs = torch.randint(0, 97, (3, 8))
    targets = torch.randint(0, 97, (3, 8))
    model.next_token_loss(inputs, targets)["loss"].backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert torch.count_nonzero(model.state_block.raw_depth_weights.grad) == 14


def test_depth_weight_and_geometry_reports_are_finite_diagnostics() -> None:
    model = _model(mode="learned_simplex").eval()
    inputs = torch.randint(0, 97, (6, 10))
    weights = model.depth_weight_report()
    assert len(weights["rows"]) == 4
    assert all(row["negative_fraction"] == 0.0 for row in weights["rows"])
    geometry = depth_geometry_report(model, inputs, max_samples=32)
    assert geometry["promotion_metric"] is False
    assert len(geometry["rows"]) == 5
    assert all(row["effective_rank"] > 0.0 for row in geometry["rows"])
    assert all(row["participation_ratio"] > 0.0 for row in geometry["rows"])


def test_depth_connected_model_uses_owned_generation_protocol() -> None:
    model = _model(mode="fixed_random").eval()
    generated = model.generate(
        torch.tensor([1, 2, 3], dtype=torch.long),
        max_new_tokens=3,
    )
    assert generated["generated_ids"].shape == (1, 6)
    assert generated["external_llm_used"] is False


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"mode": "unknown"}, "Unknown"),
        ({"simplex_identity_logit": 0.0}, "positive"),
        ({"hidden_width": 16}, "at least width"),
    ],
)
def test_depth_connection_config_rejects_invalid_values(kwargs, match: str) -> None:
    values = {
        "vocab_size": 97,
        "width": 32,
        "layers": 4,
        "attention_heads": 4,
        "hidden_width": 128,
        "context_length": 12,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        MarulhoDepthConnectedLanguageModel(DepthConnectionConfig(**values))
