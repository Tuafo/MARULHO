from __future__ import annotations

import torch

from marulho.data.language_tokenizer import BytePairLanguageTokenizer
from marulho.evaluation.language_nonlinear_fast_learner_falsification import (
    IterativeNonlinearFastLearner,
    V61Config,
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


def _module() -> IterativeNonlinearFastLearner:
    return IterativeNonlinearFastLearner(
        width=32,
        memory_heads=4,
        key_width_per_head=4,
        hidden_width_per_head=6,
        value_width_per_head=8,
        inner_steps=2,
        model_seed=9,
    )


def test_source_write_excludes_question_and_has_frozen_shape() -> None:
    tokenizer = _tokenizer()
    config = V61Config()
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
    config = V61Config()
    inputs, targets, mask, positions = _prepare_query(row, tokenizer, config)
    assert inputs.shape == targets.shape == mask.shape == (config.context_length,)
    assert positions == int(mask.sum())
    assert tokenizer.decode(targets[mask].tolist()) == "silver room"


def test_manual_per_example_gradient_matches_autograd() -> None:
    torch.manual_seed(5)
    module = _module()
    keys = torch.nn.functional.normalize(torch.randn(2, 7, 4, 4), dim=-1)
    targets = torch.randn(2, 7, 4, 8)
    mask = torch.ones(2, 7, dtype=torch.bool)
    w1 = module.initial_w1.detach().clone().unsqueeze(0).repeat(2, 1, 1, 1)
    w2 = module.initial_w2.detach().clone().unsqueeze(0).repeat(2, 1, 1, 1)
    w1.requires_grad_(True)
    w2.requires_grad_(True)
    loss, manual_w1, manual_w2, _ = module._reconstruction(
        keys, targets, mask, (w1, w2)
    )
    exact_w1, exact_w2 = torch.autograd.grad(loss.sum(), (w1, w2))
    assert torch.allclose(manual_w1, exact_w1, atol=1.0e-5, rtol=1.0e-5)
    assert torch.allclose(manual_w2, exact_w2, atol=1.0e-5, rtol=1.0e-5)


def test_two_step_fast_state_shapes_is_finite_and_has_outer_gradients() -> None:
    torch.manual_seed(7)
    module = _module()
    source = torch.randn(2, 11, 32)
    targets = torch.randn(2, 11, 32)
    mask = torch.ones(2, 11, dtype=torch.bool)
    query = torch.randn(2, 5, 32)
    output, state, inner_losses = module(source, targets, mask, query)
    assert output.shape == query.shape
    assert state[0].shape == (2, 4, 4, 6)
    assert state[1].shape == (2, 4, 6, 8)
    assert len(inner_losses) == 3
    assert all(torch.isfinite(value).all() for value in (*state, *inner_losses))
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in module.parameters())
    assert module.fast_state_values_per_document == 288


def test_schedule_exactly_covers_eight_epochs() -> None:
    config = V61Config()
    schedule, digest = _schedule_indices(8192, config)
    assert schedule.numel() == config.optimizer_steps * config.batch_size
    assert schedule.numel() * config.source_memory_positions == (
        config.padded_source_position_budget
    )
    assert len(digest) == 64
    for epoch in range(config.epochs):
        chunk = schedule[epoch * 8192 : (epoch + 1) * 8192]
        assert torch.equal(torch.sort(chunk).values, torch.arange(8192))
