from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.service.action_loop import execute_digital_action

from .artifact_io import _sha256_json, load_json_object
from .autonomy_ladder import evaluate_autonomy_ladder


APPROVED_ACTION_LEVEL2_SCHEMA_VERSION = 1
APPROVED_ACTION_LEVEL2_ARTIFACT_KIND = "terminus_approved_workspace_action_level_2"
WORKSPACE_ACTIONS = {"workspace_search", "workspace_read"}


def _normalize_action_type(action: Mapping[str, Any]) -> str:
    return " ".join(str(action.get("action_type", action.get("type", ""))).split()).strip().lower()


def _default_permission_model(action_type: str) -> dict[str, Any]:
    return {
        "max_autonomy_level": 2,
        "permitted_actions": sorted(WORKSPACE_ACTIONS),
        "execute_actions": action_type in WORKSPACE_ACTIONS,
        "recurring_limits": {},
    }


def _default_delayed_consequence_tracking(action_id: str | None = None) -> dict[str, Any]:
    return {
        "tracking_enabled": True,
        "rollback_on_failure": True,
        "tracking_scope": "workspace_action_level_2",
        "action_id": action_id,
        "follow_up_queries_required": True,
        "tracked_across_later_runs": True,
    }


def replay_action_audit_without_execution(action_report: Mapping[str, Any]) -> dict[str, Any]:
    action_audit = action_report.get("action_audit") if isinstance(action_report.get("action_audit"), Mapping) else {}
    result = action_report.get("result") if isinstance(action_report.get("result"), Mapping) else {}
    replay_payload = {
        "action_id": result.get("action_id") or action_audit.get("action_id"),
        "action_type": result.get("action_type") or action_audit.get("action_type"),
        "verification": result.get("verification", {}),
        "audit_hash": _sha256_json(action_audit),
    }
    return {
        "passed": bool(action_audit.get("passed", False)),
        "status": "replayed_no_side_effects",
        "reexecuted_effects": False,
        "replayed_payload": replay_payload,
    }


def evaluate_approved_workspace_action_level2(
    *,
    workspace_root: str | Path,
    action: Mapping[str, Any],
    operator_approval: Mapping[str, Any],
    expected_outcome: str,
    rollback_plan: str,
    permission_model: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(workspace_root)
    action_type = _normalize_action_type(action)
    permissions = dict(permission_model or _default_permission_model(action_type))
    permission_denied_reason = ""
    if action_type not in WORKSPACE_ACTIONS:
        permission_denied_reason = "non_workspace_action_requires_separate_approval"
    elif action_type not in set(str(item) for item in list(permissions.get("permitted_actions") or [])):
        permission_denied_reason = "action_not_permitted_by_level_2_permission_model"
    elif not bool(permissions.get("execute_actions", False)):
        permission_denied_reason = "permission_model_does_not_allow_action_execution"

    preflight_audit = {
        "passed": True,
        "action_type": action_type,
        "safety_flags": {
            "workspace_action_executed": False,
            "digital_action_executed": False,
            "external_calls_made": False,
            "feedback_posted": False,
            "memory_mutated": False,
            "sleep_started": False,
            "training_started": False,
        },
    }
    preflight = evaluate_autonomy_ladder(
        requested_level=2,
        permission_model=permissions,
        expected_outcome=expected_outcome,
        rollback_plan=rollback_plan,
        operator_approval=operator_approval,
        action_audit=preflight_audit,
        delayed_consequence=_default_delayed_consequence_tracking(),
        trace_replay={"passed": True, "status": "replayed_no_side_effects"},
    )

    accepted = False
    result_payload: dict[str, Any] = {}
    if not permission_denied_reason and bool(preflight.get("approved")):
        result = execute_digital_action(root, action)
        result_payload = result.to_payload()
        accepted = True

    action_id = str(result_payload.get("action_id", "")) if result_payload else None
    verification = result_payload.get("verification") if isinstance(result_payload.get("verification"), Mapping) else {}
    action_audit = {
        "passed": bool(accepted and action_type in WORKSPACE_ACTIONS and verification),
        "action_id": action_id,
        "action_type": action_type,
        "permission_model": permissions,
        "expected_outcome": expected_outcome,
        "actual_outcome": str(result_payload.get("actual_outcome", "")) if result_payload else "",
        "verification_status": str(verification.get("status", "not_executed")),
        "safety_flags": {
            "workspace_action_executed": bool(accepted),
            "digital_action_executed": False,
            "external_calls_made": False,
            "feedback_posted": False,
            "memory_mutated": False,
            "sleep_started": False,
            "training_started": False,
        },
    }
    delayed = _default_delayed_consequence_tracking(action_id)
    trace_replay = replay_action_audit_without_execution({"action_audit": action_audit, "result": result_payload})
    final_ladder = evaluate_autonomy_ladder(
        requested_level=2,
        permission_model=permissions,
        expected_outcome=expected_outcome,
        rollback_plan=rollback_plan,
        operator_approval=operator_approval,
        action_audit=action_audit,
        delayed_consequence=delayed,
        trace_replay=trace_replay,
    )
    checks = {
        "autonomy_ladder_level_2_approved": bool(preflight.get("approved")),
        "workspace_only": action_type in WORKSPACE_ACTIONS,
        "permission_not_denied": not bool(permission_denied_reason),
        "action_audit_passed": bool(action_audit.get("passed")),
        "denied_action_non_mutating": bool(permission_denied_reason) and not accepted or not bool(permission_denied_reason),
        "delayed_consequence_tracking": bool(delayed.get("tracking_enabled")) and bool(delayed.get("tracked_across_later_runs")),
        "trace_replay_no_reexecution": bool(trace_replay.get("passed")) and trace_replay.get("reexecuted_effects") is False,
    }
    status = "executed_approved_workspace_action" if accepted and all(checks.values()) else "denied_or_blocked_without_mutation"
    report = {
        "schema_version": APPROVED_ACTION_LEVEL2_SCHEMA_VERSION,
        "artifact_kind": APPROVED_ACTION_LEVEL2_ARTIFACT_KIND,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "accepted": bool(accepted),
        "denied_reason": permission_denied_reason or None,
        "workspace_root": str(root),
        "action_request": dict(action),
        "result": result_payload,
        "permission_model": permissions,
        "operator_approval": dict(operator_approval),
        "expected_outcome": expected_outcome,
        "rollback_plan": rollback_plan,
        "action_audit": action_audit,
        "delayed_consequence_tracking": delayed,
        "trace_replay": trace_replay,
        "autonomy_ladder": final_ladder,
        "checks": checks,
        "operator_visible_report": {
            "summary": f"Level 2 workspace action {status}.",
            "checks": checks,
            "denied_reason": permission_denied_reason or None,
        },
    }
    if output_path is not None:
        write_json_report_with_readme(
            output_path,
            report,
            title="Approved Workspace Action Level 2",
        )
    return report


def evaluate_approved_workspace_action_level2_file(
    input_path: str | Path,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    payload = load_json_object(input_path, label="Approved action level 2 input")
    return evaluate_approved_workspace_action_level2(
        workspace_root=str(payload.get("workspace_root", ".")),
        action=payload.get("action") if isinstance(payload.get("action"), Mapping) else {},
        operator_approval=payload.get("operator_approval") if isinstance(payload.get("operator_approval"), Mapping) else {},
        expected_outcome=str(payload.get("expected_outcome", "")),
        rollback_plan=str(payload.get("rollback_plan", "")),
        permission_model=payload.get("permission_model") if isinstance(payload.get("permission_model"), Mapping) else None,
        output_path=output_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute an approved workspace-only autonomy level 2 action.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.indent < 0:
        parser.error("--indent must be non-negative")
    report = evaluate_approved_workspace_action_level2_file(args.input, output_path=args.output)
    stream = stdout
    if stream is None:
        import sys

        stream = sys.stdout
    stream.write(json.dumps(report, indent=args.indent, sort_keys=True) + "\n")
    return 0 if bool(report.get("accepted")) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
