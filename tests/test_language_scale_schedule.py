from __future__ import annotations

import torch

from marulho.evaluation.language_scale_schedule import (
    SOURCE_SLOT_COUNTS,
    TOTAL_POSITIONS,
    TOTAL_SLOTS,
    _balanced_rows,
    _schedule_sha256,
)


def test_v80_total_budget_and_mix_are_exact() -> None:
    assert sum(SOURCE_SLOT_COUNTS.values()) == TOTAL_SLOTS == 1_048_576
    assert TOTAL_POSITIONS == 1_006_632_960


def test_balanced_rows_exposes_every_document_before_extra_repeats() -> None:
    rows = _balanced_rows(documents=7, slots=24, seed=1)
    counts = torch.bincount(rows.long(), minlength=7)
    assert int(rows.numel()) == 24
    assert int(counts.min().item()) == 3
    assert int(counts.max().item()) == 4
    assert int(rows.min().item()) == 0
    assert int(rows.max().item()) == 6


def test_schedule_hash_covers_sources_and_rows() -> None:
    sources = torch.tensor([0, 1, 2], dtype=torch.int8)
    rows = torch.tensor([4, 5, 6], dtype=torch.int32)
    assert _schedule_sha256(sources, rows) != _schedule_sha256(
        sources,
        torch.tensor([4, 6, 5], dtype=torch.int32),
    )
