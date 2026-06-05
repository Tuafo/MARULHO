from __future__ import annotations

from pathlib import Path
import tempfile

from marulho.evaluation.approved_action_level2 import (
    evaluate_approved_workspace_action_level2,
    replay_action_audit_without_execution,
)


def _approval() -> dict[str, object]:
    return {"approved": True, "operator_id": "operator-a", "scope": "autonomy_level_2"}


def test_approved_workspace_action_level2_executes_and_writes_audit_report() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "notes.md").write_text("Cats chase mice at night.\n", encoding="utf-8")
        output = root / "phase13.json"
        report = evaluate_approved_workspace_action_level2(
            workspace_root=root,
            action={
                "action_type": "workspace_read",
                "path": "notes.md",
                "query_text": "cats chase night",
                "predicted_outcome": "I expect notes.md to mention what cats chase at night.",
            },
            operator_approval=_approval(),
            expected_outcome="Gather grounded workspace evidence.",
            rollback_plan="Delete the action audit report if the operator rejects it.",
            output_path=output,
        )

        assert output.exists()
        assert (root / "README.md").exists()

    assert report["accepted"] is True
    assert report["status"] == "executed_approved_workspace_action"
    assert report["action_audit"]["passed"] is True
    assert report["delayed_consequence_tracking"]["tracked_across_later_runs"] is True
    assert report["trace_replay"]["reexecuted_effects"] is False
    assert report["autonomy_ladder"]["approved"] is True


def test_approved_workspace_action_level2_denies_web_actions_without_mutation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        before = sorted(path.name for path in root.iterdir())
        report = evaluate_approved_workspace_action_level2(
            workspace_root=root,
            action={
                "action_type": "web_fetch",
                "url": "http://127.0.0.1/example",
                "query_text": "example",
                "predicted_outcome": "I expect a web page.",
            },
            operator_approval=_approval(),
            expected_outcome="Gather web evidence.",
            rollback_plan="No rollback because denied actions do not execute.",
        )
        after = sorted(path.name for path in root.iterdir())

    assert report["accepted"] is False
    assert report["denied_reason"] == "non_workspace_action_requires_separate_approval"
    assert report["checks"]["denied_action_non_mutating"] is True
    assert before == after


def test_replay_action_audit_does_not_reexecute_effects() -> None:
    replay = replay_action_audit_without_execution(
        {
            "action_audit": {"passed": True, "action_id": "action-1", "action_type": "workspace_search"},
            "result": {"action_id": "action-1", "verification": {"status": "verified"}},
        }
    )

    assert replay["passed"] is True
    assert replay["status"] == "replayed_no_side_effects"
    assert replay["reexecuted_effects"] is False
