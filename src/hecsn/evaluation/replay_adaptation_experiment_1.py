from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from hecsn.reporting.readme_reports import write_json_report_with_readme
from hecsn.training.replay_adapter_experiment import COMPARISON_REPORT_NAME, _long_test_health

from .replay_training_approval import _sha256_json, load_json_object


REPLAY_ADAPTATION_EXPERIMENT_1_SCHEMA_VERSION = 1
REPLAY_ADAPTATION_EXPERIMENT_1_ARTIFACT_KIND = "terminus_replay_to_adaptation_experiment_1_evidence"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _path_exists(path_value: Any, *, child: str | None = None) -> bool:
    path_text = str(path_value or "").strip()
    if not path_text:
        return False
    path = Path(path_text)
    if child is not None:
        path = path / child
    return path.exists()


def _comparison_passed(comparison_report: Mapping[str, Any]) -> bool:
    checks = comparison_report.get("checks")
    return comparison_report.get("status") == "passed" and isinstance(checks, Mapping) and all(
        bool(value) for value in checks.values()
    )


def evaluate_replay_adaptation_experiment_1(
    *,
    adapter_manifest: Mapping[str, Any],
    comparison_report: Mapping[str, Any],
    promotion_gate_report: Mapping[str, Any],
    before_benchmark: Mapping[str, Any],
    after_benchmark: Mapping[str, Any],
    before_long_test: Mapping[str, Any],
    after_long_test: Mapping[str, Any],
    holdout_report: Mapping[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    adapter = _mapping(adapter_manifest.get("adapter"))
    side_effects = _mapping(adapter_manifest.get("side_effects"))
    rollback = _mapping(adapter_manifest.get("rollback"))
    adapter_path = str(adapter.get("path", ""))
    holdout_checks = _mapping(holdout_report.get("checks"))
    promotion_rollback = _mapping(promotion_gate_report.get("rollback_metadata"))
    promotion_checks = _mapping(promotion_gate_report.get("checks"))

    checks = {
        "isolated_artifact_exists": _path_exists(adapter_path)
        and _path_exists(adapter_path, child=str(adapter.get("delta_file", "adapter_delta.json"))),
        "artifact_outside_production_runtime": bool(adapter.get("production_runtime_target")) is False
        and bool(adapter.get("production_runtime_switched")) is False,
        "production_runtime_unchanged": bool(side_effects.get("production_runtime_switched")) is False
        and bool(promotion_rollback.get("production_runtime_changed")) is False,
        "comparison_passed": _comparison_passed(comparison_report),
        "holdout_evidence_saved": bool(holdout_report) and (
            holdout_report.get("status") in {"passed", "evidence_supported"}
            or (bool(holdout_checks) and all(bool(value) for value in holdout_checks.values()))
        ),
        "before_after_benchmark_saved": bool(before_benchmark) and bool(after_benchmark),
        "long_test_health_no_regression": _long_test_health(after_long_test) != "failed"
        and (_long_test_health(before_long_test) != "alive" or _long_test_health(after_long_test) == "alive"),
        "promotion_gate_blocks_production": bool(promotion_gate_report.get("eligible_for_production_promotion")) is False,
        "no_safety_regression": not _mapping(comparison_report.get("unsafe_flag_increases"))
        and bool(promotion_checks.get("unsafe_action_or_replay_not_increased", True)),
        "rollback_trivial": bool(str(rollback.get("rollback_path") or promotion_rollback.get("rollback_action") or "").strip()),
    }
    status = "passed_isolated_adaptation_evidence" if all(checks.values()) else "blocked_missing_or_regressed_evidence"
    report = {
        "schema_version": REPLAY_ADAPTATION_EXPERIMENT_1_SCHEMA_VERSION,
        "artifact_kind": REPLAY_ADAPTATION_EXPERIMENT_1_ARTIFACT_KIND,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "passed": status.startswith("passed"),
        "checks": checks,
        "evidence_hashes": {
            "adapter_manifest": _sha256_json(adapter_manifest),
            "comparison_report": _sha256_json(comparison_report),
            "promotion_gate_report": _sha256_json(promotion_gate_report),
            "before_benchmark": _sha256_json(before_benchmark),
            "after_benchmark": _sha256_json(after_benchmark),
            "before_long_test": _sha256_json(before_long_test),
            "after_long_test": _sha256_json(after_long_test),
            "holdout_report": _sha256_json(holdout_report),
        },
        "adapter": {
            "path": adapter_path,
            "comparison_report": str(Path(adapter_path) / COMPARISON_REPORT_NAME) if adapter_path else "",
            "production_runtime_target": bool(adapter.get("production_runtime_target")),
            "production_runtime_switched": bool(adapter.get("production_runtime_switched")),
        },
        "before_after": {
            "runtime_truth": comparison_report.get("runtime_truth", {}),
            "long_test_health": comparison_report.get("long_test_health", {}),
            "claimed_improvement": comparison_report.get("claimed_improvement", {}),
        },
        "safety_flags": {
            "production_model_switch": False,
            "automatic_runtime_switch": False,
            "memory_mutated": bool(side_effects.get("memory_mutated", False)),
            "feedback_posted": bool(side_effects.get("feedback_posted", False)),
            "external_calls_made": bool(side_effects.get("external_calls_made", False)),
            "sleep_started": bool(side_effects.get("sleep_started", False)),
        },
        "rollback_metadata": {
            "production_runtime_changed": False,
            "rollback_action": str(rollback.get("rollback_path") or promotion_rollback.get("rollback_action") or ""),
        },
        "operator_visible_report": {
            "summary": f"Replay-to-adaptation experiment 1 is {status}.",
            "checks": checks,
        },
    }
    if output_path is not None:
        write_json_report_with_readme(
            output_path,
            report,
            title="Replay-To-Adaptation Experiment 1",
        )
    return report


def evaluate_replay_adaptation_experiment_1_files(
    *,
    adapter_manifest_path: str | Path,
    comparison_report_path: str | Path | None,
    promotion_gate_report_path: str | Path,
    before_benchmark_path: str | Path,
    after_benchmark_path: str | Path,
    before_long_test_path: str | Path,
    after_long_test_path: str | Path,
    holdout_report_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    manifest = load_json_object(adapter_manifest_path, label="Adapter manifest")
    comparison_path = comparison_report_path
    if comparison_path is None:
        comparison_path = Path(str(_mapping(manifest.get("adapter")).get("path", ""))) / COMPARISON_REPORT_NAME
    return evaluate_replay_adaptation_experiment_1(
        adapter_manifest=manifest,
        comparison_report=load_json_object(comparison_path, label="Comparison report"),
        promotion_gate_report=load_json_object(promotion_gate_report_path, label="Promotion gate report"),
        before_benchmark=load_json_object(before_benchmark_path, label="Before benchmark"),
        after_benchmark=load_json_object(after_benchmark_path, label="After benchmark"),
        before_long_test=load_json_object(before_long_test_path, label="Before long-test"),
        after_long_test=load_json_object(after_long_test_path, label="After long-test"),
        holdout_report=load_json_object(holdout_report_path, label="Holdout report"),
        output_path=output_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Phase 12 replay-to-adaptation experiment evidence.")
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--comparison-report", type=Path, default=None)
    parser.add_argument("--promotion-gate-report", type=Path, required=True)
    parser.add_argument("--before-benchmark", type=Path, required=True)
    parser.add_argument("--after-benchmark", type=Path, required=True)
    parser.add_argument("--before-long-test", type=Path, required=True)
    parser.add_argument("--after-long-test", type=Path, required=True)
    parser.add_argument("--holdout-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.indent < 0:
        parser.error("--indent must be non-negative")
    report = evaluate_replay_adaptation_experiment_1_files(
        adapter_manifest_path=args.adapter_manifest,
        comparison_report_path=args.comparison_report,
        promotion_gate_report_path=args.promotion_gate_report,
        before_benchmark_path=args.before_benchmark,
        after_benchmark_path=args.after_benchmark,
        before_long_test_path=args.before_long_test,
        after_long_test_path=args.after_long_test,
        holdout_report_path=args.holdout_report,
        output_path=args.output,
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
