import pytest
import torch

from marulho.evaluation.language_matched_support import (
    build_matched_schedule,
    full_sized_batches,
    schedule_sha256,
    stage_schedule,
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


def test_indexed_host_schedule_matches_expanded_values_without_duplication() -> None:
    def batch(value: int) -> LanguageBatch:
        inputs = torch.full((2, 4), value, dtype=torch.long)
        return LanguageBatch(inputs, inputs + 1)

    relation = (batch(10), batch(20))
    general = ((batch(30), batch(40)), (batch(50), batch(60)))
    schedule = (
        ("general_0", 1),
        ("general_1", 0),
        ("relation", 1),
    ) * 5
    expanded = stage_schedule(
        schedule,
        relation_batches=relation,
        general_batches=general,
        device=torch.device("cpu"),
        mode="expanded_device",
    )
    indexed = stage_schedule(
        schedule,
        relation_batches=relation,
        general_batches=general,
        device=torch.device("cpu"),
        mode="indexed_host",
    )
    assert expanded.step_count == indexed.step_count == len(schedule)
    assert expanded.tokens_per_step == indexed.tokens_per_step == 8
    for index in range(len(schedule)):
        expected = expanded.batch(index, "cpu")
        actual = indexed.batch(index, "cpu")
        assert torch.equal(actual.input_ids, expected.input_ids)
        assert torch.equal(actual.target_ids, expected.target_ids)
    assert indexed.input_ids is None
    assert indexed.target_ids is None
    assert indexed.device_storage_bytes == 0
    assert indexed.storage_bytes < indexed.expanded_storage_bytes
    assert expanded.storage_bytes == expanded.expanded_storage_bytes


def test_schedule_storage_mode_is_strict() -> None:
    batch = LanguageBatch(
        torch.zeros((2, 4), dtype=torch.long),
        torch.ones((2, 4), dtype=torch.long),
    )
    with pytest.raises(ValueError, match="schedule_mode"):
        stage_schedule(
            (("general_0", 0),),
            relation_batches=(),
            general_batches=((batch,),),
            device=torch.device("cpu"),
            mode="unknown",
        )
