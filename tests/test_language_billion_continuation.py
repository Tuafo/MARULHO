from __future__ import annotations

import pytest
import torch

from marulho.evaluation.language_billion_continuation import (
    COOLDOWN_START_STEP,
    COOLDOWN_STEPS,
    INITIAL_LEARNING_RATE,
    MINIMUM_LEARNING_RATE,
    PEAK_LEARNING_RATE,
    TARGET_CUMULATIVE_POSITIONS,
    TRAIN_STEPS,
    WARMUP_STEPS,
    _learning_rate,
    _numerically_equivalent,
    _prune_snapshots,
    _progress_payload,
    _quality_checks,
    _scheduled_documents,
    _snapshot_output_path,
)
from marulho.evaluation.language_scale_schedule import TOTAL_POSITIONS


def test_v80_budget_and_stable_cooldown_schedule_are_exact() -> None:
    assert TRAIN_STEPS == 32_768
    assert TOTAL_POSITIONS == 1_006_632_960
    assert TARGET_CUMULATIVE_POSITIONS == 1_264_062_720
    assert COOLDOWN_START_STEP + COOLDOWN_STEPS == TRAIN_STEPS
    assert _learning_rate(0) == pytest.approx(
        INITIAL_LEARNING_RATE
        + (PEAK_LEARNING_RATE - INITIAL_LEARNING_RATE) / WARMUP_STEPS
    )
    assert _learning_rate(WARMUP_STEPS - 1) == pytest.approx(PEAK_LEARNING_RATE)
    assert _learning_rate(COOLDOWN_START_STEP - 1) == pytest.approx(
        PEAK_LEARNING_RATE
    )
    assert _learning_rate(TRAIN_STEPS - 1) == pytest.approx(MINIMUM_LEARNING_RATE)


def test_v80_scheduled_documents_preserve_mixed_slot_order(monkeypatch) -> None:
    import marulho.evaluation.language_billion_continuation as module

    monkeypatch.setattr(module, "TOTAL_SLOTS", 4)
    data = {
        "source_ids": torch.tensor([2, 0, 1, 2], dtype=torch.int8),
        "row_ids": torch.tensor([1, 0, 1, 0], dtype=torch.int32),
        "sources": {
            "fineweb_edu": torch.full((2, 961), 10, dtype=torch.int32),
            "cosmopedia_v2": torch.stack(
                [
                    torch.full((961,), 20, dtype=torch.int32),
                    torch.full((961,), 21, dtype=torch.int32),
                ]
            ),
            "dclm_edu": torch.stack(
                [
                    torch.full((961,), 30, dtype=torch.int32),
                    torch.full((961,), 31, dtype=torch.int32),
                ]
            ),
        },
    }
    selected = _scheduled_documents(data, offset=0, count=4)
    assert selected[:, 0].tolist() == [31, 10, 21, 30]


def test_v80_resume_numerical_tolerance_is_strict() -> None:
    admitted = {
        "differing_fraction": 9.0e-7,
        "maximum_absolute": 9.0e-7,
        "relative_l2": 9.0e-8,
    }
    assert _numerically_equivalent(
        admitted,
        maximum_absolute=1.0e-6,
        maximum_relative_l2=1.0e-7,
    )
    for field, value in (
        ("differing_fraction", 1.1e-6),
        ("maximum_absolute", 1.1e-6),
        ("relative_l2", 1.1e-7),
    ):
        rejected = dict(admitted)
        rejected[field] = value
        assert not _numerically_equivalent(
            rejected,
            maximum_absolute=1.0e-6,
            maximum_relative_l2=1.0e-7,
        )


def test_v80_quality_gate_preserves_all_three_sources() -> None:
    initial = {
        "later_segment_loss": 2.9828618367513022,
        "later_loss_by_source": {
            "fineweb_edu": 3.1576766967773438,
            "cosmopedia_v2": 2.438720703125,
            "dclm_edu": 3.3521881103515625,
        },
    }
    candidate = {
        "later_segment_loss": 2.70,
        "later_loss_by_source": {
            "fineweb_edu": 3.17,
            "cosmopedia_v2": 2.45,
            "dclm_edu": 3.10,
        },
    }
    checks = _quality_checks(
        initial=initial,
        candidate=candidate,
        completed_steps=TRAIN_STEPS,
        run_peak_cuda_allocated_bytes=4 * 1024**3,
        gradient_audit={"passed": True},
    )
    assert all(checks.values())
    candidate["later_loss_by_source"]["cosmopedia_v2"] = 2.47
    assert not _quality_checks(
        initial=initial,
        candidate=candidate,
        completed_steps=TRAIN_STEPS,
        run_peak_cuda_allocated_bytes=4 * 1024**3,
        gradient_audit={"passed": True},
    )["cosmopedia_retained"]


def test_v80_progress_and_snapshot_paths_preserve_resume_counters(tmp_path) -> None:
    prefix = tmp_path / "v80-training"
    assert _snapshot_output_path(prefix, 1024).name == "v80-training-step-01024.pt"
    progress = _progress_payload(
        completed_steps=2,
        training_seconds=4.0,
        run_peak_cuda_allocated_bytes=123,
        curve=[],
        last_step_result={"loss": 2.5},
        latest_snapshot=None,
        decision="training",
    )
    assert progress["processed_positions"] == 61_440
    assert progress["positions_per_second"] == 15_360.0


def test_v80_snapshot_pruning_retains_only_the_two_newest(tmp_path) -> None:
    prefix = tmp_path / "v80-training"
    snapshots = [
        _snapshot_output_path(prefix, completed_steps)
        for completed_steps in (1024, 2048, 3072)
    ]
    for index, snapshot in enumerate(snapshots):
        snapshot.write_bytes(bytes([index]))

    deleted = _prune_snapshots(prefix)

    assert deleted == [str(snapshots[0])]
    assert not snapshots[0].exists()
    assert snapshots[1].read_bytes() == b"\x01"
    assert snapshots[2].read_bytes() == b"\x02"
