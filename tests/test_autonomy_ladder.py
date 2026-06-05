from __future__ import annotations

import json
from pathlib import Path
import tempfile

from marulho.evaluation.autonomy_ladder import evaluate_autonomy_ladder, evaluate_autonomy_ladder_file


def _base_payload(level: int = 2) -> dict[str, object]:
    return {
        "requested_level": level,
        "permission_model": {
            "max_autonomy_level": level,
            "permitted_actions": ["workspace_search"],
            "execute_actions": level >= 2,
            "recurring_limits": {
                "max_runs": 3,
                "min_interval_seconds": 60,
                "stop_condition": "operator_stop_or_error",
            },
        },
        "expected_outcome": "Reduce uncertainty by gathering grounded workspace evidence.",
        "rollback_plan": "Disable the autonomy policy and discard pending actions.",
        "operator_approval": {"approved": True, "operator_id": "operator-a", "scope": f"autonomy_level_{level}"},
        "action_audit": {
            "passed": True,
            "safety_flags": {
                "digital_action_executed": False,
                "external_calls_made": False,
                "feedback_posted": False,
                "memory_mutated": False,
                "sleep_started": False,
                "training_started": False,
            },
        },
        "delayed_consequence_tracking": {"tracking_enabled": True, "rollback_on_failure": True},
        "trace_replay": {"passed": True, "status": "replayed_no_side_effects"},
        "evaluation_report": {
            "passed": True,
            "status": "evidence_supported",
            "benchmark_report": {"status": "passed"},
            "rollback_metadata": {"production_runtime_changed": False},
        },
    }


def test_level_zero_observe_only_passes_without_operator_approval() -> None:
    payload = _base_payload(0)
    payload["operator_approval"] = {}
    payload["permission_model"]["execute_actions"] = False  # type: ignore[index]

    report = evaluate_autonomy_ladder(
        requested_level=0,
        permission_model=payload["permission_model"],  # type: ignore[arg-type]
        expected_outcome=str(payload["expected_outcome"]),
        rollback_plan=str(payload["rollback_plan"]),
        operator_approval=payload["operator_approval"],  # type: ignore[arg-type]
        action_audit=payload["action_audit"],  # type: ignore[arg-type]
        delayed_consequence=payload["delayed_consequence_tracking"],  # type: ignore[arg-type]
        trace_replay=payload["trace_replay"],  # type: ignore[arg-type]
    )

    assert report["approved"] is True
    assert report["level_name"] == "observe_only"
    assert report["safety_flags"]["production_model_switch"] is False


def test_level_two_refuses_missing_operator_approval() -> None:
    payload = _base_payload(2)
    payload["operator_approval"] = {"approved": False, "operator_id": "", "scope": "autonomy_level_2"}

    report = evaluate_autonomy_ladder(
        requested_level=2,
        permission_model=payload["permission_model"],  # type: ignore[arg-type]
        expected_outcome=str(payload["expected_outcome"]),
        rollback_plan=str(payload["rollback_plan"]),
        operator_approval=payload["operator_approval"],  # type: ignore[arg-type]
        action_audit=payload["action_audit"],  # type: ignore[arg-type]
        delayed_consequence=payload["delayed_consequence_tracking"],  # type: ignore[arg-type]
        trace_replay=payload["trace_replay"],  # type: ignore[arg-type]
    )

    assert report["approved"] is False
    assert report["checks"]["operator_approval"] is False
    assert report["safety_flags"]["action_execution_without_approval"] is True


def test_level_three_requires_recurring_limits() -> None:
    payload = _base_payload(3)
    payload["permission_model"]["recurring_limits"] = {"max_runs": 0}  # type: ignore[index]

    report = evaluate_autonomy_ladder(
        requested_level=3,
        permission_model=payload["permission_model"],  # type: ignore[arg-type]
        expected_outcome=str(payload["expected_outcome"]),
        rollback_plan=str(payload["rollback_plan"]),
        operator_approval=payload["operator_approval"],  # type: ignore[arg-type]
        action_audit=payload["action_audit"],  # type: ignore[arg-type]
        delayed_consequence=payload["delayed_consequence_tracking"],  # type: ignore[arg-type]
        trace_replay=payload["trace_replay"],  # type: ignore[arg-type]
    )

    assert report["approved"] is False
    assert report["checks"]["recurring_limits"] is False


def test_level_five_requires_benchmark_and_rollback_evidence() -> None:
    payload = _base_payload(5)
    payload["evaluation_report"] = {"passed": True, "status": "evidence_supported"}

    report = evaluate_autonomy_ladder(
        requested_level=5,
        permission_model=payload["permission_model"],  # type: ignore[arg-type]
        expected_outcome=str(payload["expected_outcome"]),
        rollback_plan=str(payload["rollback_plan"]),
        operator_approval=payload["operator_approval"],  # type: ignore[arg-type]
        action_audit=payload["action_audit"],  # type: ignore[arg-type]
        delayed_consequence=payload["delayed_consequence_tracking"],  # type: ignore[arg-type]
        trace_replay=payload["trace_replay"],  # type: ignore[arg-type]
        evaluation_report=payload["evaluation_report"],  # type: ignore[arg-type]
    )

    assert report["approved"] is False
    assert report["checks"]["self_improvement_evaluation"] is False


def test_autonomy_ladder_file_writes_operator_visible_report() -> None:
    payload = _base_payload(4)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "autonomy.json"
        output_path = Path(tmpdir) / "report.json"
        input_path.write_text(json.dumps(payload), encoding="utf-8")
        report = evaluate_autonomy_ladder_file(input_path, output_path=output_path)
        loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["approved"] is True
    assert loaded["operator_visible_report"]["checks"]["trace_replay"] is True
    assert loaded["operator_visible_report"]["required_controls"]["policy_update_evidence"] is True
