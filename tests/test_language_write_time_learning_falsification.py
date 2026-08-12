from __future__ import annotations

import json

import torch

from marulho.data.language_tokenizer import BytePairLanguageTokenizer
from marulho.evaluation.language_write_time_learning_falsification import (
    _normalized,
    select_panel,
    source_windows,
)


def _tokenizer() -> BytePairLanguageTokenizer:
    return BytePairLanguageTokenizer.train(
        [
            "Context: Alpha moved to the silver room.",
            "Question: Where did Alpha move? Answer:",
        ],
        vocab_size=512,
    )


def test_round_robin_panel_covers_titles_before_repeating_depth() -> None:
    rows = [
        {"title": title, "case_id": f"{title}-{index}"}
        for title in ("b", "a", "c")
        for index in range(3)
    ]
    panel, digest = select_panel(rows, case_count=7)
    assert [row["case_id"] for row in panel] == [
        "a-0",
        "b-0",
        "c-0",
        "a-1",
        "b-1",
        "c-1",
        "a-2",
    ]
    expected = json.dumps(
        [row["case_id"] for row in panel], separators=(",", ":")
    ).encode("utf-8")
    assert len(digest) == 64
    assert expected


def test_source_windows_are_contiguous_next_token_pairs() -> None:
    tokenizer = _tokenizer()
    windows = source_windows(
        "Alpha moved to the silver room.",
        tokenizer,
        context_length=8,
    )
    assert windows
    for inputs, targets in windows:
        assert inputs.ndim == targets.ndim == 2
        assert inputs.shape == targets.shape
        assert inputs.shape[1] <= 8
        if inputs.shape[1] > 1:
            assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_normalized_exact_contract_ignores_surface_punctuation() -> None:
    assert _normalized(" New-York! ") == "new york"
    assert _normalized("February 7, 2016") == "february 7 2016"
