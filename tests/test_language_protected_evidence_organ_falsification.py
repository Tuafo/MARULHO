from __future__ import annotations

import torch
import torch.nn.functional as F

from marulho.data.language_tokenizer import BytePairLanguageTokenizer
from marulho.evaluation.language_protected_evidence_organ_falsification import (
    ProtectedBidirectionalEvidenceOrgan,
    V58Config,
    _best_contiguous_span,
    _schedule_indices,
    prepare_evidence_cases,
)
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


def _tokenizer() -> BytePairLanguageTokenizer:
    return BytePairLanguageTokenizer.train(
        [
            "Context: Alpha lives in São Paulo. Question: Where does Alpha live?",
            "São Paulo Alpha Context Question lives where",
        ],
        vocab_size=512,
    )


def test_prepare_evidence_cases_preserves_exact_character_oracle_span() -> None:
    tokenizer = _tokenizer()
    source = "Alpha lives in São Paulo."
    rows = [
        {
            "case_id": "one",
            "source_text": source,
            "question": "Where does Alpha live?",
            "answers": ["São Paulo"],
            "answer_source_character_start": source.index("São Paulo"),
        }
    ]
    prepared = prepare_evidence_cases(
        rows,
        tokenizer,
        context_length=64,
        require_gold=True,
    )
    assert prepared.start_positions is not None
    assert prepared.end_positions is not None
    start = int(prepared.start_positions[0])
    end = int(prepared.end_positions[0])
    copied = prepared.cases[0]["source_text"][start : end + 1]
    assert copied == "São Paulo"
    assert bool(prepared.character_mask[0, start : end + 1].all())


def test_best_span_is_contiguous_and_length_bounded() -> None:
    starts = torch.tensor([0.0, 1.0, 9.0, 0.0, 0.0, 0.0])
    ends = torch.tensor([0.0, 0.0, 0.0, 2.0, 8.0, 0.0])
    source = torch.tensor([False, True, True, True, True, False])
    assert _best_contiguous_span(
        starts, ends, source, maximum_answer_characters=2
    ) == (2, 3)


def test_frozen_schedule_has_exact_budget_and_hash() -> None:
    config = V58Config()
    schedule, digest = _schedule_indices(8192, config)
    assert schedule.numel() == config.optimizer_steps * config.batch_size
    assert schedule.numel() * config.context_length == config.padded_position_budget
    assert len(digest) == 64
    for epoch in range(config.epochs):
        chunk = schedule[epoch * 8192 : (epoch + 1) * 8192]
        assert torch.equal(torch.sort(chunk).values, torch.arange(8192))


def test_bidirectional_organ_masks_non_source_logits_and_backpropagates() -> None:
    torch.manual_seed(7)
    parent = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=64,
            embedding_dim=32,
            state_dim=32,
            state_layers=2,
            attention_heads=4,
            transformer_context_length=16,
            transformer_mlp_ratio=2.0,
        )
    )
    organ = ProtectedBidirectionalEvidenceOrgan(
        parent,
        context_length=16,
        initialized_from_parent=True,
        model_seed=11,
    )
    ids = torch.randint(0, 64, (2, 16))
    attention = torch.ones(2, 16, dtype=torch.bool)
    character_tokens = torch.tensor([[2, 3, 4, 5, 6, 7, 0, 0]] * 2)
    character_offsets = torch.zeros(2, 8, dtype=torch.long)
    character_ids = torch.randint(0, 256, (2, 8))
    characters = torch.tensor([[True] * 6 + [False] * 2] * 2)
    start, end = organ(
        ids,
        attention,
        character_tokens,
        character_offsets,
        character_ids,
        characters,
    )
    assert torch.isfinite(start[:, :6]).all()
    assert (start[:, 6:] == torch.finfo(start.dtype).min).all()
    loss = F.cross_entropy(start, torch.tensor([1, 2])) + F.cross_entropy(
        end, torch.tensor([3, 4])
    )
    loss.backward()
    assert all(parameter.grad is not None for parameter in organ.parameters())
