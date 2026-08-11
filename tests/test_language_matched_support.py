import pytest
import torch

from marulho.evaluation.language_matched_support import (
    load_matched_arm_artifact,
    project_matched_arm_runtime,
    run_matched_training_arm,
    save_matched_arm_artifact,
    build_matched_schedule,
    full_sized_batches,
    grouped_staged_batch,
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


def test_shared_schedule_balances_five_sources_without_repeats() -> None:
    schedule = build_matched_schedule(
        step_count=50,
        relation_fraction=0.0,
        relation_batch_count=0,
        general_batch_counts=(10, 10, 10, 10, 10),
        seed=42,
    )
    for source_index in range(5):
        indices = [
            index
            for kind, index in schedule
            if kind == f"general_{source_index}"
        ]
        assert len(indices) == 10
        assert len(set(indices)) == 10


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


def test_grouped_staged_batch_concatenates_consecutive_schedule_entries() -> None:
    batches = tuple(
        LanguageBatch(
            torch.full((2, 4), value, dtype=torch.long),
            torch.full((2, 4), value + 1, dtype=torch.long),
        )
        for value in (10, 20, 30)
    )
    staged = stage_schedule(
        (("general_0", 0), ("general_0", 1), ("general_0", 2)),
        relation_batches=(),
        general_batches=(batches,),
        device=torch.device("cpu"),
        mode="indexed_host",
    )
    grouped = grouped_staged_batch(staged, start=1, count=2, device="cpu")
    assert grouped.input_ids.shape == (4, 4)
    assert grouped.target_ids.shape == (4, 4)
    assert grouped.input_ids[:, 0].tolist() == [20, 20, 30, 30]
    assert grouped.target_ids[:, 0].tolist() == [21, 21, 31, 31]
    with pytest.raises(IndexError, match="out of bounds"):
        grouped_staged_batch(staged, start=2, count=2, device="cpu")


def test_runtime_projection_keeps_startup_and_uses_late_step_median() -> None:
    report = project_matched_arm_runtime(
        (30.0, 2.0, 4.0), total_optimizer_steps=100, setup_seconds=10.0
    )
    assert report["projected_steady_optimizer_step_seconds"] == 3.0
    assert report["paid_warmup_seconds"] == 36.0
    assert report["projected_counted_training_seconds"] == 300.0
    assert report["projected_total_seconds"] == 346.0


def test_runtime_projection_rejects_weak_preflight() -> None:
    with pytest.raises(ValueError, match="at least two"):
        project_matched_arm_runtime((1.0,), total_optimizer_steps=10)
    with pytest.raises(ValueError, match="positive optimizer"):
        project_matched_arm_runtime((1.0, 1.0), total_optimizer_steps=0)


def test_matched_arm_artifact_is_atomic_and_contract_strict(tmp_path) -> None:
    output = tmp_path / "arm.pt"
    state = {"weight": torch.arange(4, dtype=torch.float32)}
    save_matched_arm_artifact(
        output,
        arm_name="candidate",
        contract_sha256="abc123",
        row={"heldout": {"loss": 1.25}},
        model_state=state,
    )
    assert output.exists()
    assert not (tmp_path / ".arm.pt.tmp").exists()
    row, restored = load_matched_arm_artifact(
        output,
        expected_arm_name="candidate",
        expected_contract_sha256="abc123",
    )
    assert row == {"heldout": {"loss": 1.25}}
    assert restored is not None
    assert torch.equal(restored["weight"], state["weight"])
    with pytest.raises(ValueError, match="contract differs"):
        load_matched_arm_artifact(
            output,
            expected_arm_name="candidate",
            expected_contract_sha256="different",
        )


def test_matched_runner_reuses_completed_arm_and_restores_model(tmp_path) -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    restored_weight = torch.full_like(model.weight, 3.0)
    output = tmp_path / "completed.pt"
    save_matched_arm_artifact(
        output,
        arm_name="candidate",
        contract_sha256="frozen",
        row={"name": "candidate", "heldout": {"loss": 0.75}},
        model_state={"weight": restored_weight},
    )
    row = run_matched_training_arm(
        "candidate",
        architecture="unused-on-resume",
        model=model,  # type: ignore[arg-type]
        initial_state=model.state_dict(),
        training_loss=lambda _inputs, _targets: torch.tensor(0.0),
        execution={},
        allocated_compile_seconds=0.0,
        prepared=None,  # type: ignore[arg-type]
        training_config=None,  # type: ignore[arg-type]
        gradient_clip=1.0,
        precision="float32",
        relation_eval_batch_size=1,
        model_seed=1,
        device=torch.device("cpu"),
        progress_prefix="unused",
        arm_artifact_path=output,
        arm_contract_sha256="frozen",
    )
    assert row == {"name": "candidate", "heldout": {"loss": 0.75}}
    assert torch.equal(model.weight, restored_weight)
