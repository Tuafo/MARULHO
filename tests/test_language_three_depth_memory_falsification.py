from __future__ import annotations

import torch

from marulho.data.language_tokenizer import BytePairLanguageTokenizer
from marulho.evaluation.language_three_depth_memory_falsification import (
    ProtectedThreeDepthMemory,
    V62Config,
    _prepare_query,
    _prepare_source,
    _schedule_indices,
)
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


def _tokenizer() -> BytePairLanguageTokenizer:
    return BytePairLanguageTokenizer.train(
        [
            "Context: Alpha moved to the silver room.",
            "Question: Where did Alpha move? Answer: silver room",
        ],
        vocab_size=512,
    )


def _parent() -> MarulhoLanguageModel:
    torch.manual_seed(3)
    return MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=128,
            embedding_dim=32,
            state_dim=32,
            state_layers=4,
            attention_heads=4,
            transformer_context_length=24,
            transformer_mlp_ratio=2.0,
        )
    ).eval()


def _controller() -> ProtectedThreeDepthMemory:
    return ProtectedThreeDepthMemory(
        width=32,
        state_layers=4,
        memory_heads=4,
        key_width_per_head=4,
        value_width_per_head=8,
        injection_layers=(1, 2, 3),
        model_seed=9,
    )


def test_source_write_excludes_question_and_has_frozen_shape() -> None:
    tokenizer = _tokenizer()
    config = V62Config()
    inputs, targets, mask, positions = _prepare_source(
        "Alpha moved to the silver room.", tokenizer, config
    )
    assert inputs.shape == targets.shape == mask.shape == (5, 64)
    assert positions == int(mask.sum())
    decoded = tokenizer.decode(inputs[mask].tolist())
    assert "Alpha" in decoded
    assert "Question:" not in decoded
    assert "Answer:" not in decoded


def test_query_answer_mask_selects_only_answer_and_eos_targets() -> None:
    tokenizer = _tokenizer()
    row = {
        "case_id": "one",
        "question": "Where did Alpha move?",
        "answers": ["silver room"],
    }
    config = V62Config()
    inputs, targets, mask, positions = _prepare_query(row, tokenizer, config)
    assert inputs.shape == targets.shape == mask.shape == (config.context_length,)
    assert positions == int(mask.sum())
    assert tokenizer.decode(targets[mask].tolist()) == "silver room"


def test_inactive_custom_forward_matches_parent_hidden_and_kv_exactly() -> None:
    parent = _parent()
    controller = _controller()
    inputs = torch.randint(0, parent.config.vocab_size, (2, 9))
    ordinary = parent._forward_hidden(inputs, collect_telemetry=False)
    hidden, state = controller.forward_parent(parent, inputs, active=False)
    assert torch.equal(ordinary["hidden"], hidden)
    assert set(ordinary["state"]) == set(state)
    assert all(torch.equal(ordinary["state"][name], state[name]) for name in state)


def test_three_depth_memory_has_exact_shape_and_all_outer_gradients() -> None:
    torch.manual_seed(5)
    parent = _parent()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    controller = _controller()
    source_hidden = torch.randn(2, 11, 32)
    source_values = torch.randn(2, 11, 32)
    source_mask = torch.ones(2, 11, dtype=torch.bool)
    memory = controller.write(source_hidden, source_values, source_mask)
    assert memory.shape == (2, 4, 4, 8)
    inputs = torch.randint(0, parent.config.vocab_size, (2, 7))
    hidden, _ = controller.forward_parent(parent, inputs, memory, active=True)
    parent.lm_head(hidden).square().mean().backward()
    assert all(parameter.grad is not None for parameter in controller.parameters())
    assert all(torch.count_nonzero(parameter.grad) > 0 for parameter in controller.parameters())
    assert controller.fast_state_values_per_document == 128


def test_streaming_active_forward_matches_full_active_forward() -> None:
    parent = _parent()
    controller = _controller()
    source_hidden = torch.randn(1, 8, 32)
    source_values = torch.randn(1, 8, 32)
    source_mask = torch.ones(1, 8, dtype=torch.bool)
    memory = controller.write(source_hidden, source_values, source_mask)
    inputs = torch.randint(0, parent.config.vocab_size, (1, 7))
    full, _ = controller.forward_parent(parent, inputs, memory, active=True)
    pieces: list[torch.Tensor] = []
    state = None
    for index in range(inputs.shape[1]):
        hidden, state = controller.forward_parent(
            parent, inputs[:, index : index + 1], memory, state, active=True
        )
        pieces.append(hidden)
    streamed = torch.cat(pieces, dim=1)
    assert torch.allclose(full, streamed, atol=1.0e-5, rtol=1.0e-5)


def test_schedule_exactly_covers_eight_epochs() -> None:
    config = V62Config()
    schedule, digest = _schedule_indices(8192, config)
    assert schedule.numel() == config.optimizer_steps * config.batch_size
    assert schedule.numel() * config.source_memory_positions == (
        config.padded_source_position_budget
    )
    assert len(digest) == 64
    for epoch in range(config.epochs):
        chunk = schedule[epoch * 8192 : (epoch + 1) * 8192]
        assert torch.equal(torch.sort(chunk).values, torch.arange(8192))
