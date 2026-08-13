import torch

from marulho.training.language_macro_cortex import (
    MarulhoMacroCortexLanguageModel,
    transfer_transformer_common_state,
)
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


def _config() -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=1024,
        embedding_dim=32,
        state_dim=32,
        state_layers=2,
        attention_heads=4,
        transformer_context_length=128,
        transformer_mlp_ratio=2.0,
        transformer_dropout=0.0,
        tie_embeddings=True,
        active_language_path="marulho_macro_cortex_v70",
    )


def _models() -> tuple[MarulhoLanguageModel, MarulhoMacroCortexLanguageModel]:
    torch.manual_seed(701)
    control = MarulhoLanguageModel(_config()).double()
    torch.manual_seed(702)
    candidate = MarulhoMacroCortexLanguageModel(_config()).double()
    transfer_transformer_common_state(control, candidate)
    return control, candidate


def test_macro_cortex_common_state_is_exact_and_ratio_is_matched() -> None:
    control, candidate = _models()
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    for name in set(control_state) & set(candidate_state):
        if control_state[name].shape == candidate_state[name].shape:
            torch.testing.assert_close(
                candidate_state[name], control_state[name], atol=0, rtol=0
            )
    ratio = sum(p.numel() for p in candidate.parameters()) / sum(
        p.numel() for p in control.parameters()
    )
    assert 0.99 <= ratio <= 1.01


def test_macro_cortex_future_block_cannot_change_earlier_logits() -> None:
    _, candidate = _models()
    torch.manual_seed(703)
    tokens = torch.randint(0, 64, (2, 128))
    expected = candidate(tokens)["logits"]
    future = tokens.clone()
    future[:, 64:] = torch.randint(0, 64, future[:, 64:].shape)
    actual = candidate(future)["logits"]
    torch.testing.assert_close(actual[:, :64], expected[:, :64], atol=0, rtol=0)


def test_macro_cortex_completed_block_changes_following_logits() -> None:
    _, candidate = _models()
    torch.manual_seed(704)
    tokens = torch.randint(0, 64, (2, 128))
    expected = candidate(tokens)["logits"]
    earlier = tokens.clone()
    earlier[:, :64] = (earlier[:, :64] + 1) % 64
    actual = candidate(earlier)["logits"]
    assert not torch.equal(actual[:, 64:], expected[:, 64:])


def test_macro_cortex_all_parameters_receive_finite_nonzero_gradients() -> None:
    _, candidate = _models()
    torch.manual_seed(705)
    inputs = torch.randint(0, 64, (2, 128))
    targets = torch.randint(0, 64, (2, 128))
    candidate.next_token_loss(inputs, targets)["loss"].backward()
    for parameter in candidate.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert bool(torch.count_nonzero(parameter.grad))
