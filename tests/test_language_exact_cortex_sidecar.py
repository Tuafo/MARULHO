from __future__ import annotations

import torch
import torch.nn.functional as F

from marulho.training.language_exact_cortex_sidecar import (
    V73ExactCortexSidecarLanguageModel,
    transfer_v73_transformer_state,
)
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


def _config() -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=96,
        embedding_dim=32,
        state_dim=32,
        state_layers=10,
        attention_heads=4,
        transformer_context_length=16,
        transformer_mlp_ratio=4.0,
        transformer_dropout=0.0,
        tie_embeddings=True,
        active_language_path="v73_test",
    )


def _models():
    torch.manual_seed(31)
    control = MarulhoLanguageModel(_config()).eval()
    torch.manual_seed(32)
    candidate = V73ExactCortexSidecarLanguageModel(
        _config(), state_tokens=4, state_width=16, state_heads=2
    ).eval()
    transfer_v73_transformer_state(control, candidate)
    return control, candidate


def test_v73_disabled_sidecar_is_bit_exact_to_transformer() -> None:
    control, candidate = _models()
    inputs = torch.randint(0, 96, (3, 16))
    with torch.no_grad():
        expected = control(inputs, collect_telemetry=False)["logits"]
        actual = candidate.forward_segment(
            inputs, candidate.initial_state(3), sidecar_enabled=False
        )["logits"]
    assert torch.equal(actual, expected)


def test_v73_zero_initialized_read_is_bit_exact_and_causal() -> None:
    _, candidate = _models()
    first = torch.randint(0, 96, (3, 16))
    changed = first.clone()
    changed[:, 9:] = torch.randint(0, 96, (3, 7))
    state = candidate.initial_state(3)
    with torch.no_grad():
        first_result = candidate.forward_segment(first, state)
        changed_result = candidate.forward_segment(changed, state)
        disabled = candidate.forward_segment(first, state, sidecar_enabled=False)
    assert torch.equal(first_result["logits"], disabled["logits"])
    assert torch.equal(first_result["logits"][:, :9], changed_result["logits"][:, :9])


def test_v73_boundary_controls_change_only_state_identity() -> None:
    _, candidate = _models()
    state = torch.randn(4, 4, 16)
    assert torch.equal(candidate.boundary_state(state, "persistent"), state)
    assert torch.equal(candidate.boundary_state(state, "reset"), candidate.initial_state(4))
    assert torch.equal(candidate.boundary_state(state, "shuffled"), state.roll(1, 0))


def test_v73_writer_auxiliary_cannot_change_transformer_gradients() -> None:
    _, candidate = _models()
    candidate.train()
    inputs = torch.randint(0, 96, (3, 16))
    targets = inputs[:, torch.tensor([3, 7, 11, 15])]
    result = candidate.forward_segment(inputs, candidate.initial_state(3))
    auxiliary = F.cross_entropy(
        result["workspace_logits"].reshape(-1, 96), targets.reshape(-1)
    )
    auxiliary.backward()
    assert candidate.sidecar_write.key.weight.grad is not None
    assert torch.count_nonzero(candidate.sidecar_write.key.weight.grad)
    assert candidate.sidecar_decode.weight.grad is not None
    assert candidate.token_embedding.weight.grad is None
    assert all(
        parameter.grad is None
        for name, parameter in candidate.named_parameters()
        if name.startswith("state_block.layers")
    )
