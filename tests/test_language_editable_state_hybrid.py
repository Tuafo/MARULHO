from __future__ import annotations

import torch
import torch.nn.functional as F

from marulho.training.language_editable_state_hybrid import (
    MarulhoEditableStateHybridLanguageModel,
)
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


def _config(**overrides) -> LanguageModelConfig:
    values = {
        "vocab_size": 96,
        "embedding_dim": 32,
        "state_dim": 32,
        "state_core": "transformer",
        "state_layers": 4,
        "attention_heads": 4,
        "transformer_context_length": 16,
        "transformer_mlp_ratio": 2.0,
        "transformer_dropout": 0.0,
        "tie_embeddings": True,
        "active_language_path": "marulho_editable_state_hybrid_v33",
    }
    values.update(overrides)
    return LanguageModelConfig(**values)


def _hybrid(**overrides) -> MarulhoEditableStateHybridLanguageModel:
    return MarulhoEditableStateHybridLanguageModel(
        _config(**overrides),
        local_attention_window=8,
        matrix_chunk_size=4,
    )


def test_hybrid_exactly_matches_transformer_parameter_budget() -> None:
    torch.manual_seed(3)
    baseline = MarulhoLanguageModel(_config(active_language_path="baseline"))
    torch.manual_seed(3)
    candidate = _hybrid()
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    assert candidate_parameters == baseline_parameters
    assert torch.equal(candidate.token_embedding.weight, baseline.token_embedding.weight)


def test_hybrid_is_causal_and_every_parameter_backpropagates() -> None:
    torch.manual_seed(5)
    model = _hybrid().eval()
    first = torch.tensor([[1, 8, 9, 10, 11, 12]], dtype=torch.long)
    second = torch.tensor([[1, 8, 9, 44, 45, 46]], dtype=torch.long)
    first_logits = model(first, collect_telemetry=False)["logits"]
    second_logits = model(second, collect_telemetry=False)["logits"]
    assert torch.allclose(first_logits[:, :3], second_logits[:, :3], atol=2e-5)

    model.train()
    output = model(first)
    loss = F.cross_entropy(
        output["logits"][:, :-1].reshape(-1, model.config.vocab_size),
        first[:, 1:].reshape(-1),
    )
    loss.backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    assert missing == []
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert output["telemetry"]["state_core"] == "editable_state_local_attention_hybrid"
    assert output["telemetry"]["event_control_enabled"] is False


def test_hybrid_chunk_parallel_and_recurrent_decode_match() -> None:
    torch.manual_seed(7)
    model = _hybrid().eval()
    token_ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]], dtype=torch.long)
    with torch.no_grad():
        full = model(token_ids, collect_telemetry=False)["logits"]
        state = None
        steps = []
        for index in range(int(token_ids.shape[1])):
            step = model.forward_step(
                token_ids[:, index : index + 1],
                state,
                collect_telemetry=False,
            )
            state = step["state"]
            steps.append(step["logits"][:, -1])
    recurrent = torch.stack(steps, dim=1)
    assert torch.allclose(recurrent, full, atol=3e-5, rtol=2e-5)
    assert state is not None
    assert int(state["position"].item()) == int(token_ids.shape[1])


def test_hybrid_runtime_state_is_bounded() -> None:
    torch.manual_seed(11)
    model = _hybrid(transformer_context_length=32).eval()
    state = None
    with torch.no_grad():
        for index in range(24):
            step = model.forward_step(
                torch.tensor([[index % 32]], dtype=torch.long),
                state,
                collect_telemetry=index == 23,
            )
            state = step["state"]
    assert state is not None
    assert int(state["layer_0_key"].shape[2]) == 8
    assert tuple(state["layer_1_matrix"].shape[-2:]) == (4, 8)
    assert step["telemetry"]["kv_cache_tokens"] == 8
    assert step["telemetry"]["matrix_state_elements"] == 2 * 4 * 4 * 8

