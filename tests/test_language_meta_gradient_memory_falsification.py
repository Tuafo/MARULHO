from __future__ import annotations

import torch

from marulho.data.language_tokenizer import BytePairLanguageTokenizer
from marulho.evaluation.language_meta_gradient_memory_falsification import (
    MetaGradientEpisodicMatrix,
    V60Config,
    _prepare_query,
    _prepare_source,
    _schedule_indices,
)


def _tokenizer() -> BytePairLanguageTokenizer:
    return BytePairLanguageTokenizer.train(
        [
            "Context: Alpha moved to the silver room.",
            "Question: Where did Alpha move? Answer: silver room",
        ],
        vocab_size=512,
    )


def test_source_write_excludes_question_and_has_frozen_shape() -> None:
    tokenizer = _tokenizer()
    config = V60Config()
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
    config = V60Config()
    inputs, targets, mask, positions = _prepare_query(row, tokenizer, config)
    assert inputs.shape == targets.shape == mask.shape == (config.context_length,)
    assert positions == int(mask.sum())
    decoded_targets = tokenizer.decode(targets[mask].tolist())
    assert decoded_targets == "silver room"


def test_meta_gradient_memory_shapes_and_gradients() -> None:
    torch.manual_seed(5)
    module = MetaGradientEpisodicMatrix(
        width=32,
        memory_heads=4,
        key_width_per_head=4,
        value_width_per_head=8,
        model_seed=9,
    )
    source = torch.randn(2, 11, 32)
    values = torch.randn(2, 11, 32)
    mask = torch.ones(2, 11, dtype=torch.bool)
    query = torch.randn(2, 5, 32)
    output, memory = module(source, values, mask, query)
    assert output.shape == query.shape
    assert memory.shape == (2, 4, 4, 8)
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in module.parameters())
    assert module.fast_state_values_per_document == 128


def test_schedule_exactly_covers_eight_epochs() -> None:
    config = V60Config()
    schedule, digest = _schedule_indices(8192, config)
    assert schedule.numel() == config.optimizer_steps * config.batch_size
    assert schedule.numel() * config.source_memory_positions == (
        config.padded_source_position_budget
    )
    assert len(digest) == 64
    for epoch in range(config.epochs):
        chunk = schedule[epoch * 8192 : (epoch + 1) * 8192]
        assert torch.equal(torch.sort(chunk).values, torch.arange(8192))
