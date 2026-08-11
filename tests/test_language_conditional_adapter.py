import torch

from marulho.training.language_conditional_adapter import (
    ADAPTER_KEY,
    ADAPTER_VALUE,
    MarulhoConditionalAdapterLanguageModel,
)
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


def _config() -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=64,
        embedding_dim=16,
        state_dim=16,
        state_layers=2,
        attention_heads=4,
        transformer_context_length=16,
        transformer_mlp_ratio=2.0,
        transformer_dropout=0.0,
        tie_embeddings=True,
    )


def test_conditional_adapter_inactive_path_is_bit_exact() -> None:
    torch.manual_seed(7)
    parent = MarulhoLanguageModel(_config()).eval()
    candidate = MarulhoConditionalAdapterLanguageModel.from_parent(parent).eval()
    inputs = torch.randint(0, 64, (3, 9))

    expected = parent(inputs, collect_telemetry=False)
    observed = candidate(inputs, collect_telemetry=False)

    assert torch.equal(observed["logits"], expected["logits"])
    assert observed["state"].keys() == expected["state"].keys()
    assert all(
        torch.equal(observed["state"][key], expected["state"][key])
        for key in expected["state"]
    )
    assert candidate.parent_state_sha256()
    assert not any(
        parameter.requires_grad
        for name, parameter in candidate.named_parameters()
        if not name.startswith("conditional_adapter.")
    )


def test_conditional_adapter_active_path_streams_and_trains() -> None:
    torch.manual_seed(11)
    parent = MarulhoLanguageModel(_config()).eval()
    candidate = MarulhoConditionalAdapterLanguageModel.from_parent(parent)
    candidate.set_conditional_adapter_enabled(True)
    inputs = torch.randint(0, 64, (2, 7))
    targets = torch.randint(0, 64, (2, 7))

    candidate.eval()
    full = candidate(inputs, collect_telemetry=False)
    state = None
    logits = []
    for position in range(inputs.shape[1]):
        step = candidate.forward_step(
            inputs[:, position],
            state,
            collect_telemetry=False,
        )
        state = step["state"]
        logits.append(step["logits"])
    streamed = torch.cat(logits, dim=1)

    assert torch.allclose(streamed, full["logits"], atol=1e-5, rtol=1e-5)
    assert ADAPTER_KEY in state and ADAPTER_VALUE in state
    assert int(state[ADAPTER_KEY].shape[2]) == inputs.shape[1]

    candidate.train()
    loss = candidate.next_token_loss(
        inputs,
        targets,
        collect_telemetry=False,
        return_evidence=False,
    )["loss"]
    loss.backward()
    adapter_parameters = tuple(candidate.conditional_adapter.parameters())
    assert all(parameter.grad is not None for parameter in adapter_parameters)
    assert all(torch.count_nonzero(parameter.grad) > 0 for parameter in adapter_parameters)
    assert all(
        parameter.grad is None
        for name, parameter in candidate.named_parameters()
        if not name.startswith("conditional_adapter.")
    )
