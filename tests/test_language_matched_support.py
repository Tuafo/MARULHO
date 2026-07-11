import pytest
import torch

from marulho.evaluation.language_matched_support import (
    build_matched_schedule,
    full_sized_batches,
    schedule_sha256,
)
from marulho.training.language_model import LanguageBatch


def test_shared_schedule_is_reproducible_and_source_balanced() -> None:
    schedule = build_matched_schedule(
        step_count=1619,
        relation_fraction=0.20,
        relation_batch_count=400,
        general_batch_counts=(700, 600),
        seed=1337,
    )
    repeated = build_matched_schedule(
        step_count=1619,
        relation_fraction=0.20,
        relation_batch_count=400,
        general_batch_counts=(700, 600),
        seed=1337,
    )
    assert schedule == repeated
    assert schedule_sha256(schedule) == schedule_sha256(repeated)
    assert sum(kind == "relation" for kind, _index in schedule) == 323
    general = [kind for kind, _index in schedule if kind != "relation"]
    assert abs(general.count("general_0") - general.count("general_1")) <= 1


def test_shared_schedule_changes_with_seed() -> None:
    kwargs = {
        "step_count": 100,
        "relation_fraction": 0.20,
        "relation_batch_count": 30,
        "general_batch_counts": (40, 40),
    }
    assert build_matched_schedule(**kwargs, seed=1) != build_matched_schedule(
        **kwargs,
        seed=2,
    )


def test_full_batch_filter_excludes_partial_tails() -> None:
    full = LanguageBatch(
        input_ids=torch.zeros((4, 8), dtype=torch.long),
        target_ids=torch.zeros((4, 8), dtype=torch.long),
    )
    partial = LanguageBatch(
        input_ids=torch.zeros((2, 8), dtype=torch.long),
        target_ids=torch.zeros((2, 8), dtype=torch.long),
    )
    assert full_sized_batches((full, partial), batch_size=4) == (full,)
    with pytest.raises(ValueError, match="no full-sized batches"):
        full_sized_batches((partial,), batch_size=4)
