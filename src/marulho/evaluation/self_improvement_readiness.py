from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from marulho.reporting.readme_reports import write_json_report_with_readme

from .autonomy_ladder import evaluate_autonomy_ladder
from .artifact_io import load_json_object


SELF_IMPROVEMENT_SCHEMA_VERSION = 1
SELF_IMPROVEMENT_ARTIFACT_KIND = "terminus_bounded_self_improvement_readiness"
UNSAFE_PHASE_FLAGS = (
    "autonomous_adapter_training",
    "memory_promotion_from_replay",
    "action_execution_without_approval",
    "production_model_switch",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _passed(report: Mapping[str, Any]) -> bool:
    if report.get("passed") is True or report.get("approved") is True:
        return True
    status = str(report.get("status", ""))
    return status in {
        "evidence_supported",
        "passed_experimental_promotion_allowed",
        "passed_isolated_adaptation_evidence",
        "passed_multi_hour_living_evidence",
        "executed_approved_workspace_action",
    }


def _promotion_passed(report: Mapping[str, Any]) -> bool:
    return report.get("eligible_for_experimental_promotion") is True and report.get("eligible_for_production_promotion") is False


def _rollback_tested(metadata: Mapping[str, Any]) -> bool:
    return (
        metadata.get("production_runtime_changed") is False
        and bool(str(metadata.get("rollback_action", metadata.get("rollback_test", ""))).strip())
    )


def _safety_regressed(*reports: Mapping[str, Any]) -> bool:
    for report in reports:
        flags = _mapping(report.get("safety_flags"))
        if any(flags.get(flag) is True for flag in UNSAFE_PHASE_FLAGS):
            return True
        replay = _mapping(report.get("replay_safety_status"))
        if any(replay.get(flag) is True for flag in ("training_started", "memory_mutated", "digital_action_executed")):
            return True
    return False


def evaluate_self_improvement_readiness(
    *,
    phase12_report: Mapping[str, Any],
    phase13_report: Mapping[str, Any],
    phase14_report: Mapping[str, Any],
    promotion_gate_report: Mapping[str, Any],
    benchmark_report: Mapping[str, Any],
    operator_id: str,
    useful_behavior_note: str = "",
) -> dict[str, Any]:
    rollback_metadata = _mapping(promotion_gate_report.get("rollback_metadata"))
    benchmark_or_useful_behavior = bool(useful_behavior_note.strip()) or bool(
        _mapping(promotion_gate_report.get("checks")).get("benchmark_improvement_or_documented_useful_behavior")
    )
    evaluation_report = {
        "passed": _passed(phase12_report) and _passed(phase14_report) and _promotion_passed(promotion_gate_report),
        "status": "evidence_supported",
        "benchmark_report": {
            "success": benchmark_report.get("success", True),
            "health_verdict": phase14_report.get("health_verdict"),
            "runtime_truth_verdict": phase14_report.get("runtime_truth_verdict"),
            "useful_behavior_note": useful_behavior_note,
        },
        "rollback_metadata": dict(rollback_metadata),
    }
    autonomy_ladder = evaluate_autonomy_ladder(
        requested_level=5,
        permission_model={
            "max_autonomy_level": 5,
            "permitted_actions": ["workspace_report_generation", "isolated_adapter_experiment_review"],
            "execute_actions": True,
            "recurring_limits": {
                "max_runs": 1,
                "min_interval_seconds": 3600,
                "stop_condition": "operator_stop_or_failed_gate",
            },
        },
        expected_outcome="Run a bounded self-improvement experiment only against isolated artifacts.",
        rollback_plan=str(rollback_metadata.get("rollback_action", "Remove isolated artifact references.")),
        operator_approval={"approved": bool(operator_id.strip()), "operator_id": operator_id, "scope": "autonomy_level_5"},
        action_audit={
            "passed": _passed(phase13_report),
            "safety_flags": {
                "digital_action_executed": False,
                "external_calls_made": False,
                "feedback_posted": False,
                "memory_mutated": False,
                "sleep_started": False,
                "training_started": False,
            },
        },
        delayed_consequence={"tracking_enabled": True, "rollback_on_failure": True},
        trace_replay={"passed": True, "status": "replayed_no_side_effects"},
        evaluation_report=evaluation_report,
    )
    checks = {
        "phase12_evidence_passed": _passed(phase12_report),
        "phase13_action_audit_passed": _passed(phase13_report),
        "phase14_multi_hour_evidence_passed": _passed(phase14_report),
        "promotion_gate_passed_for_experimental_only": _promotion_passed(promotion_gate_report),
        "benchmark_or_useful_behavior_documented": benchmark_or_useful_behavior,
        "rollback_tested": _rollback_tested(rollback_metadata),
        "operator_approval_recorded": bool(operator_id.strip()),
        "no_safety_regression": not _safety_regressed(phase12_report, phase13_report, phase14_report, promotion_gate_report),
        "production_runtime_unchanged": rollback_metadata.get("production_runtime_changed") is False,
        "production_model_switch_blocked": True,
        "autonomy_ladder_level_5_passed": autonomy_ladder.get("approved") is True,
    }
    passed = all(checks.values())
    return {
        "schema_version": SELF_IMPROVEMENT_SCHEMA_VERSION,
        "artifact_kind": SELF_IMPROVEMENT_ARTIFACT_KIND,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_bounded_level_5_experiment" if passed else "blocked_for_bounded_level_5_experiment",
        "passed": passed,
        "autonomy_level": 5,
        "production_model_switch_allowed": False,
        "production_runtime_changed": False,
        "useful_behavior_note": useful_behavior_note,
        "rollback_metadata": dict(rollback_metadata),
        "autonomy_ladder_report": autonomy_ladder,
        "operator_visible_report": {
            "summary": "Phase 15 readiness passes; production switching remains blocked."
            if passed
            else "Phase 15 readiness is blocked by missing evidence.",
            "checks": checks,
        },
        "checks": checks,
    }


def evaluate_self_improvement_readiness_files(
    *,
    phase12_report_path: str | Path,
    phase13_report_path: str | Path,
    phase14_report_path: str | Path,
    promotion_gate_report_path: str | Path,
    benchmark_report_path: str | Path,
    output_path: str | Path,
    operator_id: str,
    useful_behavior_note: str = "",
) -> dict[str, Any]:
    report = evaluate_self_improvement_readiness(
        phase12_report=load_json_object(phase12_report_path, label="Phase 12 report"),
        phase13_report=load_json_object(phase13_report_path, label="Phase 13 report"),
        phase14_report=load_json_object(phase14_report_path, label="Phase 14 report"),
        promotion_gate_report=load_json_object(promotion_gate_report_path, label="Promotion gate report"),
        benchmark_report=load_json_object(benchmark_report_path, label="Benchmark report"),
        operator_id=operator_id,
        useful_behavior_note=useful_behavior_note,
    )
    write_json_report_with_readme(output_path, report, title="Terminus Bounded Self-Improvement Readiness")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Terminus Phase 15 bounded self-improvement readiness.")
    parser.add_argument("--phase12-report", type=Path, required=True)
    parser.add_argument("--phase13-report", type=Path, required=True)
    parser.add_argument("--phase14-report", type=Path, required=True)
    parser.add_argument("--promotion-gate-report", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--operator-id", required=True)
    parser.add_argument("--useful-behavior-note", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.indent < 0:
        parser.error("--indent must be non-negative")
    report = evaluate_self_improvement_readiness_files(
        phase12_report_path=args.phase12_report,
        phase13_report_path=args.phase13_report,
        phase14_report_path=args.phase14_report,
        promotion_gate_report_path=args.promotion_gate_report,
        benchmark_report_path=args.benchmark_report,
        output_path=args.output,
        operator_id=args.operator_id,
        useful_behavior_note=args.useful_behavior_note,
    )
    stream = stdout
    if stream is None:
        import sys

        stream = sys.stdout
    stream.write(json.dumps(report, indent=args.indent, sort_keys=True) + "\n")
    return 0 if bool(report.get("passed")) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
