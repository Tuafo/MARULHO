from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from marulho.reporting.readme_reports import write_json_report_with_readme

from .artifact_io import load_json_object


AUTONOMY_LADDER_SCHEMA_VERSION = 1
AUTONOMY_LADDER_ARTIFACT_KIND = "terminus_autonomy_ladder_evaluation"

AUTONOMY_LEVELS: dict[int, dict[str, Any]] = {
    0: {
        "name": "observe_only",
        "description": "Observe runtime state only.",
        "may_propose_actions": False,
        "may_execute_actions": False,
        "requires_operator_approval": False,
        "requires_recurring_limits": False,
        "requires_policy_update_evidence": False,
        "requires_self_improvement_evidence": False,
    },
    1: {
        "name": "propose_actions_only",
        "description": "Generate action proposals without execution.",
        "may_propose_actions": True,
        "may_execute_actions": False,
        "requires_operator_approval": False,
        "requires_recurring_limits": False,
        "requires_policy_update_evidence": False,
        "requires_self_improvement_evidence": False,
    },
    2: {
        "name": "execute_approved_actions",
        "description": "Execute only individually approved actions.",
        "may_propose_actions": True,
        "may_execute_actions": True,
        "requires_operator_approval": True,
        "requires_recurring_limits": False,
        "requires_policy_update_evidence": False,
        "requires_self_improvement_evidence": False,
    },
    3: {
        "name": "constrained_recurring_actions",
        "description": "Execute approved recurring actions with strict limits.",
        "may_propose_actions": True,
        "may_execute_actions": True,
        "requires_operator_approval": True,
        "requires_recurring_limits": True,
        "requires_policy_update_evidence": False,
        "requires_self_improvement_evidence": False,
    },
    4: {
        "name": "adaptive_policy_updates_after_evaluation",
        "description": "Update constrained policies only after saved evaluation evidence.",
        "may_propose_actions": True,
        "may_execute_actions": True,
        "requires_operator_approval": True,
        "requires_recurring_limits": True,
        "requires_policy_update_evidence": True,
        "requires_self_improvement_evidence": False,
    },
    5: {
        "name": "bounded_self_improvement_loop",
        "description": "Run bounded self-improvement only with approval, benchmark, rollback, and audit trail.",
        "may_propose_actions": True,
        "may_execute_actions": True,
        "requires_operator_approval": True,
        "requires_recurring_limits": True,
        "requires_policy_update_evidence": True,
        "requires_self_improvement_evidence": True,
    },
}

UNSAFE_ACTION_FLAGS = (
    "digital_action_executed",
    "external_calls_made",
    "feedback_posted",
    "memory_mutated",
    "sleep_started",
    "training_started",
)


def _bool_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _approval_passed(approval: Mapping[str, Any], level: int) -> bool:
    expected_scope = f"autonomy_level_{level}"
    return (
        approval.get("approved") is True
        and str(approval.get("operator_id", "")).strip() != ""
        and approval.get("scope") == expected_scope
    )


def _action_audit_passed(action_audit: Mapping[str, Any]) -> bool:
    if not action_audit:
        return False
    if "passed" in action_audit:
        return bool(action_audit.get("passed"))
    flags = _bool_mapping(action_audit.get("safety_flags"))
    return all(flags.get(flag) is not True for flag in UNSAFE_ACTION_FLAGS)


def _trace_replay_passed(trace_replay: Mapping[str, Any]) -> bool:
    if not trace_replay:
        return False
    if "passed" in trace_replay:
        return bool(trace_replay.get("passed"))
    return trace_replay.get("status") in {"passed", "replayed_no_side_effects"}


def _delayed_consequence_passed(delayed_consequence: Mapping[str, Any]) -> bool:
    if not delayed_consequence:
        return False
    return delayed_consequence.get("tracking_enabled") is True and delayed_consequence.get("rollback_on_failure") is True


def _recurring_limits_passed(recurring_limits: Mapping[str, Any]) -> bool:
    if not recurring_limits:
        return False
    return (
        int(recurring_limits.get("max_runs", 0) or 0) > 0
        and int(recurring_limits.get("max_runs", 0) or 0) <= 100
        and float(recurring_limits.get("min_interval_seconds", 0.0) or 0.0) > 0.0
        and bool(str(recurring_limits.get("stop_condition", "")).strip())
    )


def _evaluation_passed(evaluation: Mapping[str, Any]) -> bool:
    if not evaluation:
        return False
    if evaluation.get("passed") is True:
        return True
    return evaluation.get("status") in {
        "passed",
        "passed_experimental_promotion_allowed",
        "evidence_supported",
    }


def _int_value(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def evaluate_autonomy_ladder(
    *,
    requested_level: int,
    permission_model: Mapping[str, Any],
    expected_outcome: str,
    rollback_plan: str,
    operator_approval: Mapping[str, Any] | None = None,
    action_audit: Mapping[str, Any] | None = None,
    delayed_consequence: Mapping[str, Any] | None = None,
    trace_replay: Mapping[str, Any] | None = None,
    evaluation_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if requested_level not in AUTONOMY_LEVELS:
        raise ValueError("requested autonomy level must be between 0 and 5.")
    level = AUTONOMY_LEVELS[requested_level]
    approval = dict(operator_approval or {})
    audit = dict(action_audit or {})
    delayed = dict(delayed_consequence or {})
    replay = dict(trace_replay or {})
    evaluation = dict(evaluation_report or {})
    recurring_limits = _bool_mapping(permission_model.get("recurring_limits"))
    permitted_actions = permission_model.get("permitted_actions")
    if not isinstance(permitted_actions, list):
        permitted_actions = []

    checks = {
        "permission_model_present": bool(permission_model),
        "permission_model_matches_level": _int_value(permission_model.get("max_autonomy_level"), -1) >= requested_level,
        "expected_outcome_recorded": bool(expected_outcome.strip()),
        "rollback_plan_recorded": bool(rollback_plan.strip()),
        "operator_approval": (not bool(level["requires_operator_approval"])) or _approval_passed(approval, requested_level),
        "action_execution_permission": bool(level["may_execute_actions"]) or not bool(permission_model.get("execute_actions", False)),
        "recurring_limits": (not bool(level["requires_recurring_limits"])) or _recurring_limits_passed(recurring_limits),
        "delayed_consequence_tracking": _delayed_consequence_passed(delayed),
        "trace_replay": _trace_replay_passed(replay),
        "action_audit": _action_audit_passed(audit),
        "policy_update_evaluation": (not bool(level["requires_policy_update_evidence"])) or _evaluation_passed(evaluation),
        "self_improvement_evaluation": (not bool(level["requires_self_improvement_evidence"]))
        or (
            _evaluation_passed(evaluation)
            and bool(evaluation.get("benchmark_report"))
            and bool(evaluation.get("rollback_metadata"))
        ),
    }
    status = "approved_for_level" if all(checks.values()) else "blocked"
    return {
        "schema_version": AUTONOMY_LADDER_SCHEMA_VERSION,
        "artifact_kind": AUTONOMY_LADDER_ARTIFACT_KIND,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested_level": requested_level,
        "level_name": level["name"],
        "status": status,
        "approved": status == "approved_for_level",
        "permission_model": {
            "max_autonomy_level": permission_model.get("max_autonomy_level"),
            "permitted_actions": permitted_actions,
            "execute_actions": bool(permission_model.get("execute_actions", False)),
            "recurring_limits": dict(recurring_limits),
        },
        "expected_outcome": expected_outcome,
        "rollback_plan": rollback_plan,
        "delayed_consequence_tracking": delayed,
        "trace_replay": replay,
        "action_audit": audit,
        "operator_visible_report": {
            "summary": f"Autonomy level {requested_level} ({level['name']}) is {status}.",
            "checks": checks,
            "required_controls": {
                "operator_approval": bool(level["requires_operator_approval"]),
                "recurring_limits": bool(level["requires_recurring_limits"]),
                "policy_update_evidence": bool(level["requires_policy_update_evidence"]),
                "self_improvement_evidence": bool(level["requires_self_improvement_evidence"]),
            },
        },
        "safety_flags": {
            "autonomous_adapter_training": False,
            "memory_promotion_from_replay": False,
            "action_execution_without_approval": not checks["operator_approval"] and bool(level["may_execute_actions"]),
            "production_model_switch": False,
        },
        "checks": checks,
    }


def evaluate_autonomy_ladder_file(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = load_json_object(input_path, label="Autonomy ladder input")
    report = evaluate_autonomy_ladder(
        requested_level=_int_value(payload.get("requested_level"), -1),
        permission_model=_bool_mapping(payload.get("permission_model")),
        expected_outcome=str(payload.get("expected_outcome", "")),
        rollback_plan=str(payload.get("rollback_plan", "")),
        operator_approval=_bool_mapping(payload.get("operator_approval")),
        action_audit=_bool_mapping(payload.get("action_audit")),
        delayed_consequence=_bool_mapping(payload.get("delayed_consequence_tracking")),
        trace_replay=_bool_mapping(payload.get("trace_replay")),
        evaluation_report=_bool_mapping(payload.get("evaluation_report")),
    )
    if output_path is not None:
        write_json_report_with_readme(
            output_path,
            report,
            title="Terminus Autonomy Ladder Evaluation",
        )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a proposed Terminus autonomy ladder level.")
    parser.add_argument("--input", type=Path, required=True, help="Autonomy ladder input JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output report JSON.")
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.indent < 0:
        parser.error("--indent must be non-negative")
    report = evaluate_autonomy_ladder_file(args.input, output_path=args.output)
    encoded = json.dumps(report, indent=args.indent, sort_keys=True) + "\n"
    stream = stdout
    if stream is None:
        import sys

        stream = sys.stdout
    stream.write(encoded)
    return 0 if bool(report.get("approved")) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
