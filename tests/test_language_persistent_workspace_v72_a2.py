from __future__ import annotations

import json

from marulho.evaluation.language_persistent_workspace_v72_a2 import aggregate


def _row(arm: str, loss: float, *, speed: float = 100.0, swap: float | None = None):
    return {
        "arm": arm,
        "contract_sha256": "same",
        "parameter_ratio": 1.0,
        "training": {
            "positions_per_second": speed,
            "peak_cuda_allocated_bytes": 1_000_000,
            "gradient_audit": {"passed": True},
        },
        "evaluation": {
            "later_segment_loss": loss,
            "state_swap": {"wrong_minus_correct": swap},
        },
    }


def test_v72_a2_aggregate_requires_joint_language_and_state_win(tmp_path) -> None:
    rows = [
        _row("transformer", 4.10),
        _row("persistent", 4.00, speed=75.0, swap=0.03),
        _row("reset", 4.05),
        _row("shuffled", 4.06),
    ]
    inputs = []
    for row in rows:
        path = tmp_path / f"{row['arm']}.json"
        path.write_text(json.dumps(row), encoding="utf-8")
        inputs.append(path)
    report = aggregate(inputs, tmp_path / "decision.json")
    assert report["passed"] is True
    assert report["decision"] == "advance_v72_persistent_workspace_to_stage_b"


def test_v72_a2_aggregate_retires_state_without_loss_utility(tmp_path) -> None:
    rows = [
        _row("transformer", 4.00),
        _row("persistent", 4.01, speed=90.0, swap=0.03),
        _row("reset", 4.02),
        _row("shuffled", 4.03),
    ]
    inputs = []
    for row in rows:
        path = tmp_path / f"{row['arm']}.json"
        path.write_text(json.dumps(row), encoding="utf-8")
        inputs.append(path)
    report = aggregate(inputs, tmp_path / "decision.json")
    assert report["passed"] is False
    assert report["decision"] == "retire_v72_persistent_workspace_real_language_failure"
