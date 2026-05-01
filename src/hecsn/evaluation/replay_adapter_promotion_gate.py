from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from .replay_training_approval import (
    APPROVAL_ARTIFACT_KIND,
    EXPERIMENTAL_ADAPTER_PROMOTION_SCOPE,
    REQUIRED_SAFETY_ACKNOWLEDGEMENTS,
    _sha256_json,
    load_json_object,
)
from hecsn.training.replay_adapter_experiment import (
    COMPARISON_REPORT_NAME,
    ISOLATED_EXPERIMENT_ARTIFACT_KIND,
    _replay_safety_flags,
    _runtime_truth_verdict,
    _unsafe_flags_increased,
)


PROMOTION_GATE_SCHEMA_VERSION = 1
PROMOTION_GATE_ARTIFACT_KIND = "terminus_replay_adapter_promotion_gate"
RUNTIME_TRUTH_ORDER = {"failed": 0, "degraded": 1, "partial": 2, "alive": 3}


def _load_required_report(path: str | Path | None, *, label: str) -> dict[str, Any]:
    if path is None:
        raise ValueError(f"{label} is required.")
    return load_json_object(path, label=label)


def _validate_promotion_approval(approval: Mapping[str, Any], gate_report: Mapping[str, Any]) -> dict[str, Any]:
    if approval.get("artifact_kind") != APPROVAL_ARTIFACT_KIND:
        raise ValueError("approval artifact kind is invalid.")
    if approval.get("scope") != EXPERIMENTAL_ADAPTER_PROMOTION_SCOPE:
        raise ValueError(f"approval scope must be {EXPERIMENTAL_ADAPTER_PROMOTION_SCOPE!r}.")
    if str(approval.get("operator_id", "")).strip() == "":
        raise ValueError("approval operator_id is required.")
    if approval.get("intended_target") != "experimental_replay_adapter_promotion_gate":
        raise ValueError("approval intended_target is invalid for promotion.")
    if approval.get("gate_report_hash") != _sha256_json(gate_report):
        raise ValueError("approval gate report hash does not match gate report.")
    acknowledgements = approval.get("safety_acknowledgements")
    if not isinstance(acknowledgements, Mapping) or any(
        acknowledgements.get(name) is not True for name in REQUIRED_SAFETY_ACKNOWLEDGEMENTS
    ):
        raise ValueError("approval safety acknowledgements are incomplete.")
    for field in (
        "training_started",
        "memory_mutated",
        "adapter_created",
        "feedback_posted",
        "digital_action_executed",
        "external_calls_made",
        "sleep_started",
    ):
        if approval.get(field) is not False:
            raise ValueError(f"approval side-effect flag {field} must be false.")
    expires_at = datetime.fromisoformat(str(approval.get("expires_at")).replace("Z", "+00:00"))
    if expires_at <= datetime.now(timezone.utc):
        raise ValueError("promotion approval has expired.")
    return {
        "approval_id": approval.get("approval_id"),
        "operator_id": approval.get("operator_id"),
        "scope": approval.get("scope"),
        "expires_at": expires_at.isoformat(),
    }


def evaluate_replay_adapter_promotion_gate(
    *,
    adapter_manifest: Mapping[str, Any],
    comparison_report: Mapping[str, Any],
    before_benchmark: Mapping[str, Any],
    after_benchmark: Mapping[str, Any],
    gate_report: Mapping[str, Any],
    approval: Mapping[str, Any],
    useful_behavior_note: str = "",
) -> dict[str, Any]:
    if adapter_manifest.get("artifact_kind") != ISOLATED_EXPERIMENT_ARTIFACT_KIND:
        raise ValueError("adapter manifest is not an isolated replay adapter experiment.")
    approval_summary = _validate_promotion_approval(approval, gate_report)

    before_verdict = _runtime_truth_verdict(before_benchmark)
    after_verdict = _runtime_truth_verdict(after_benchmark)
    runtime_truth_regressed = (
        before_verdict in RUNTIME_TRUTH_ORDER
        and after_verdict in RUNTIME_TRUTH_ORDER
        and RUNTIME_TRUTH_ORDER[after_verdict] < RUNTIME_TRUTH_ORDER[before_verdict]
    )
    contamination_failed = any(
        isinstance(check, Mapping) and check.get("name") == "decontamination" and check.get("passed") is not True
        for check in gate_report.get("checks", [])
        if isinstance(gate_report.get("checks"), list)
    )
    unsafe_increase = _unsafe_flags_increased(before_benchmark, after_benchmark)
    comparison_checks = comparison_report.get("checks") if isinstance(comparison_report.get("checks"), Mapping) else {}
    documented_useful_behavior = bool(useful_behavior_note.strip())
    benchmark_improvement = bool(comparison_report.get("claimed_improvement", {}).get("claimed")) if isinstance(
        comparison_report.get("claimed_improvement"), Mapping
    ) else False

    checks = {
        "adapter_isolated": adapter_manifest.get("adapter", {}).get("production_runtime_switched") is False
        if isinstance(adapter_manifest.get("adapter"), Mapping)
        else False,
        "operator_promotion_approval": True,
        "benchmark_improvement_or_documented_useful_behavior": benchmark_improvement or documented_useful_behavior,
        "runtime_truth_no_regression": not runtime_truth_regressed,
        "contamination_passed": not contamination_failed,
        "unsafe_action_or_replay_not_increased": not unsafe_increase,
        "comparison_report_passed": all(bool(value) for value in comparison_checks.values()) if comparison_checks else False,
    }
    passed = all(checks.values())
    experimental_path = str(adapter_manifest.get("adapter", {}).get("path", "")) if isinstance(
        adapter_manifest.get("adapter"), Mapping
    ) else ""
    return {
        "schema_version": PROMOTION_GATE_SCHEMA_VERSION,
        "artifact_kind": PROMOTION_GATE_ARTIFACT_KIND,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_experimental_promotion_allowed" if passed else "failed_experimental_promotion_blocked",
        "eligible_for_experimental_promotion": passed,
        "eligible_for_production_promotion": False,
        "checks": checks,
        "approval": approval_summary,
        "adapter": {
            "manifest_hash": _sha256_json(adapter_manifest),
            "comparison_report_hash": _sha256_json(comparison_report),
            "experimental_path": experimental_path,
            "production_runtime_switch_allowed": False,
        },
        "evidence": {
            "runtime_truth": {"before": before_verdict, "after": after_verdict, "regressed": runtime_truth_regressed},
            "unsafe_flag_increases": unsafe_increase,
            "before_replay_safety_flags": _replay_safety_flags(before_benchmark),
            "after_replay_safety_flags": _replay_safety_flags(after_benchmark),
            "useful_behavior_note": useful_behavior_note,
        },
        "rollback_metadata": {
            "production_runtime_changed": False,
            "configured_path_kind": "non_default_experimental_path_only",
            "rollback_action": "Remove the experimental adapter path reference and delete the isolated artifact directory.",
            "adapter_artifact_path": experimental_path,
        },
    }


def evaluate_replay_adapter_promotion_gate_files(
    *,
    adapter_manifest_path: str | Path,
    comparison_report_path: str | Path | None,
    before_benchmark_path: str | Path | None,
    after_benchmark_path: str | Path | None,
    gate_report_path: str | Path | None,
    approval_path: str | Path | None,
    output_path: str | Path,
    useful_behavior_note: str = "",
) -> dict[str, Any]:
    manifest = load_json_object(adapter_manifest_path, label="Adapter manifest")
    comparison_path = comparison_report_path
    if comparison_path is None and isinstance(manifest.get("adapter"), Mapping):
        adapter_path = Path(str(manifest["adapter"].get("path", "")))
        comparison_path = adapter_path / COMPARISON_REPORT_NAME
    report = evaluate_replay_adapter_promotion_gate(
        adapter_manifest=manifest,
        comparison_report=_load_required_report(comparison_path, label="Comparison report"),
        before_benchmark=_load_required_report(before_benchmark_path, label="Before benchmark report"),
        after_benchmark=_load_required_report(after_benchmark_path, label="After benchmark report"),
        gate_report=_load_required_report(gate_report_path, label="Replay training gate report"),
        approval=_load_required_report(approval_path, label="Promotion approval"),
        useful_behavior_note=useful_behavior_note,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an isolated replay adapter for experimental promotion.")
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--comparison-report", type=Path, default=None)
    parser.add_argument("--before-benchmark", type=Path, required=True)
    parser.add_argument("--after-benchmark", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--useful-behavior-note", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.indent < 0:
        parser.error("--indent must be non-negative")
    report = evaluate_replay_adapter_promotion_gate_files(
        adapter_manifest_path=args.adapter_manifest,
        comparison_report_path=args.comparison_report,
        before_benchmark_path=args.before_benchmark,
        after_benchmark_path=args.after_benchmark,
        gate_report_path=args.gate_report,
        approval_path=args.approval,
        output_path=args.output,
        useful_behavior_note=args.useful_behavior_note,
    )
    encoded = json.dumps(report, indent=args.indent, sort_keys=True) + "\n"
    stream = stdout
    if stream is None:
        import sys

        stream = sys.stdout
    stream.write(encoded)
    return 0 if bool(report.get("eligible_for_experimental_promotion")) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
