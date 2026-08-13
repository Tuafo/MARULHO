import torch

from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel
from marulho.training.language_periodic_hierarchy import (
    MarulhoPeriodicHierarchyLanguageModel,
    PeriodicLocalAttention,
    transfer_periodic_common_state,
)


def _config() -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=4096,
        embedding_dim=32,
        state_dim=32,
        state_layers=10,
        attention_heads=4,
        transformer_context_length=128,
        transformer_mlp_ratio=2.0,
        transformer_dropout=0.0,
        tie_embeddings=True,
        active_language_path="marulho_periodic_hierarchy_v71",
    )


def _model(macro: bool) -> MarulhoPeriodicHierarchyLanguageModel:
    torch.manual_seed(711)
    control = MarulhoLanguageModel(_config()).double()
    torch.manual_seed(712)
    model = MarulhoPeriodicHierarchyLanguageModel(
        _config(), macro_enabled=macro
    ).double()
    transfer_periodic_common_state(control, model)
    return model


def test_periodic_hierarchy_layer_topology_and_parameter_ratios() -> None:
    torch.manual_seed(711)
    control = MarulhoLanguageModel(_config()).double()
    local = _model(False)
    macro = _model(True)
    assert [index for index in range(10) if index in {4, 9}] == [4, 9]
    control_count = sum(parameter.numel() for parameter in control.parameters())
    assert sum(parameter.numel() for parameter in local.parameters()) == control_count
    assert 0.99 <= sum(p.numel() for p in macro.parameters()) / control_count <= 1.01
    for model in (local, macro):
        for name, value in control.state_dict().items():
            candidate = model.state_dict().get(name)
            if candidate is not None and candidate.shape == value.shape:
                torch.testing.assert_close(candidate, value, atol=0, rtol=0)


def test_periodic_hierarchy_macro_off_has_no_macro_parameters() -> None:
    local = _model(False)
    assert not any("macro" in name or "summary" in name for name, _ in local.named_parameters())


def test_periodic_hierarchy_is_causal() -> None:
    model = _model(True)
    torch.manual_seed(713)
    tokens = torch.randint(0, 1024, (2, 128))
    expected = model(tokens)["logits"]
    future = tokens.clone()
    future[:, 64:] = torch.randint(0, 1024, future[:, 64:].shape)
    actual = model(future)["logits"]
    torch.testing.assert_close(actual[:, :64], expected[:, :64], atol=0, rtol=0)


def test_periodic_macro_completed_block_changes_following_block() -> None:
    torch.manual_seed(715)
    attention = PeriodicLocalAttention(32, heads=4, macro_enabled=True).double()
    hidden = torch.randn((2, 128, 32), dtype=torch.float64)
    expected = attention(hidden)
    earlier = hidden.clone()
    earlier[:, :64] += 2.0
    actual = attention(earlier)
    assert not torch.equal(actual[:, 64:], expected[:, 64:])


def test_periodic_hierarchy_all_parameters_receive_gradients() -> None:
    for macro in (False, True):
        model = _model(macro)
        torch.manual_seed(714)
        inputs = torch.randint(0, 1024, (2, 128))
        targets = torch.randint(0, 1024, (2, 128))
        model.next_token_loss(inputs, targets)["loss"].backward()
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert bool(torch.isfinite(parameter.grad).all())
            assert bool(torch.count_nonzero(parameter.grad))
