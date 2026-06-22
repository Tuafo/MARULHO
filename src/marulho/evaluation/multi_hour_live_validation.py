from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from marulho.reporting.readme_reports import write_json_report_with_readme

from .live_long_run_validation import SAFE_REPLAY_FLAGS
from .artifact_io import load_json_object


MULTI_HOUR_SCHEMA_VERSION = 1
MULTI_HOUR_ARTIFACT_KIND = "terminus_multi_hour_live_validation"
MIN_MULTI_HOUR_DURATION_MINUTES = 120.0
MAX_MEMORY_FILL_FOR_STABLE_RUN = 0.85


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _runtime_truth(report: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("final_runtime_truth", "status_runtime_truth_summary", "terminus_runtime_truth_summary"):
        value = report.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _replay_safety(report: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime_truth = _runtime_truth(report)
    safety = _mapping(runtime_truth.get("safety_flags"))
    replay = safety.get("replay_safety")
    if isinstance(replay, Mapping):
        return replay
    return {}


def _action_audit(long_test_report: Mapping[str, Any], benchmark_report: Mapping[str, Any]) -> dict[str, Any]:
    actuator = _mapping(benchmark_report.get("policy_actuator_summary"))
    action_count = int(_number(long_test_report.get("action_count"), 0.0))
    return {
        "actions_recorded": action_count,
        "policy_actuator_advisory": actuator.get("advisory") is not False,
        "policy_actuator_executable": actuator.get("executable") is True,
        "passed": action_count >= 0 and actuator.get("executable") is not True,
    }


def _memory_bounded(long_test_report: Mapping[str, Any]) -> bool:
    pressure = _mapping(long_test_report.get("memory_pressure_report"))
    if pressure.get("unrecovered_high_pressure") is True:
        return False
    final_fill = _number(pressure.get("final_fill", long_test_report.get("final_memory_fill")), 1.0)
    max_fill = _number(pressure.get("max_fill", final_fill), final_fill)
    recovered = pressure.get("recovered_from_high") is True
    return final_fill <= MAX_MEMORY_FILL_FOR_STABLE_RUN or (max_fill > MAX_MEMORY_FILL_FOR_STABLE_RUN and recovered)


def _bottlenecks(
    *,
    multi_hour_report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    live_validation_report: Mapping[str, Any],
) -> list[str]:
    bottlenecks: list[str] = []
    if _number(multi_hour_report.get("p95_latency_ms")) > max(5000.0, _number(baseline_report.get("p95_latency_ms")) * 2.0):
        bottlenecks.append("p95_latency_increased")
    if _number(multi_hour_report.get("final_memory_fill")) > MAX_MEMORY_FILL_FOR_STABLE_RUN:
        bottlenecks.append("memory_fill_near_limit")
    if _number(multi_hour_report.get("total_errors")) > 0:
        bottlenecks.append("runtime_errors_recorded")
    if _mapping(multi_hour_report.get("final_embedder")).get("degraded") is True:
        bottlenecks.append("embedder_degraded")
    if live_validation_report.get("passed") is not True:
        bottlenecks.append("live_validator_not_passed")
    if not bottlenecks:
        bottlenecks.append("no_blocking_bottleneck_detected")
    return bottlenecks


def validate_multi_hour_live_run(
    *,
    multi_hour_long_test_report: Mapping[str, Any],
    baseline_long_test_report: Mapping[str, Any],
    live_validation_report: Mapping[str, Any],
    benchmark_report: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_truth = _runtime_truth(multi_hour_long_test_report)
    replay_safety = _replay_safety(multi_hour_long_test_report) or _replay_safety(benchmark_report)
    action_audit = _action_audit(multi_hour_long_test_report, benchmark_report)
    duration_minutes = _number(multi_hour_long_test_report.get("duration_minutes"))
    baseline_duration_minutes = _number(baseline_long_test_report.get("duration_minutes"))
    runtime_progress = max(
        _number(multi_hour_long_test_report.get("max_background_tokens_processed")),
        _number(multi_hour_long_test_report.get("final_tick_count")),
        _number(multi_hour_long_test_report.get("final_token_count"))
        - _number(multi_hour_long_test_report.get("initial_token_count")),
    )
    baseline_runtime_progress = max(
        _number(baseline_long_test_report.get("max_background_tokens_processed")),
        _number(baseline_long_test_report.get("final_tick_count")),
        _number(baseline_long_test_report.get("final_token_count"))
        - _number(baseline_long_test_report.get("initial_token_count")),
    )
    checks = {
        "duration_is_multi_hour": duration_minutes >= MIN_MULTI_HOUR_DURATION_MINUTES,
        "baseline_was_alive_30_minute_run": baseline_duration_minutes >= 30.0
        and baseline_long_test_report.get("health_verdict") == "alive",
        "live_validation_evidence_supported": live_validation_report.get("status") == "evidence_supported"
        and live_validation_report.get("passed") is True,
        "long_test_health_alive": multi_hour_long_test_report.get("health_verdict") == "alive",
        "runtime_truth_alive": runtime_truth.get("verdict") == "alive",
        "runtime_progress_present": runtime_progress > 0,
        "runtime_progress_not_regressed_to_zero": runtime_progress > 0 and baseline_runtime_progress > 0,
        "memory_pressure_bounded_or_recovered": _memory_bounded(multi_hour_long_test_report),
        "embedding_health_available": bool(_mapping(multi_hour_long_test_report.get("final_embedder")).get("available")),
        "embedding_health_not_degraded": _mapping(multi_hour_long_test_report.get("final_embedder")).get("degraded")
        is not True,
        "replay_training_action_safety": all(replay_safety.get(flag) is not True for flag in SAFE_REPLAY_FLAGS),
        "action_audit_passed": action_audit["passed"],
        "traces_or_snapshots_present": bool(multi_hour_long_test_report.get("snapshots"))
        or bool(live_validation_report.get("trace_evidence", {}).get("trace_count", 0))
        if isinstance(live_validation_report.get("trace_evidence"), Mapping)
        else bool(multi_hour_long_test_report.get("snapshots")),
        "recommended_operator_action_present": bool(str(runtime_truth.get("recommended_action", "")).strip()),
    }
    passed = all(checks.values())
    bottlenecks = _bottlenecks(
        multi_hour_report=multi_hour_long_test_report,
        baseline_report=baseline_long_test_report,
        live_validation_report=live_validation_report,
    )
    return {
        "schema_version": MULTI_HOUR_SCHEMA_VERSION,
        "artifact_kind": MULTI_HOUR_ARTIFACT_KIND,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_multi_hour_living_evidence" if passed else "blocked_multi_hour_living_claim",
        "passed": passed,
        "duration_minutes": duration_minutes,
        "baseline_duration_minutes": baseline_duration_minutes,
        "runtime_truth_verdict": runtime_truth.get("verdict", "unknown"),
        "health_verdict": multi_hour_long_test_report.get("health_verdict", "unknown"),
        "acceptance_verdict": multi_hour_long_test_report.get("acceptance_verdict", "unknown"),
        "baseline_health_verdict": baseline_long_test_report.get("health_verdict", "unknown"),
        "readouts": {
            "multi_hour_total": multi_hour_long_test_report.get("total_readouts", 0),
            "baseline_total": baseline_long_test_report.get("total_readouts", 0),
            "unique_topics": multi_hour_long_test_report.get("unique_topics", 0),
            "topic_diversity_ratio": multi_hour_long_test_report.get("topic_diversity_ratio", 0),
        },
        "runtime_progress": {
            "multi_hour": runtime_progress,
            "baseline": baseline_runtime_progress,
        },
        "latency_and_cost": {
            "avg_latency_ms": multi_hour_long_test_report.get("avg_latency_ms"),
            "p95_latency_ms": multi_hour_long_test_report.get("p95_latency_ms"),
            "benchmark_total_latency_ms": benchmark_report.get("total_latency_ms"),
            "cost_usd": benchmark_report.get("cost_usd", 0),
        },
        "memory_pressure": dict(_mapping(multi_hour_long_test_report.get("memory_pressure_report")))
        or {"final_fill": multi_hour_long_test_report.get("final_memory_fill")},
        "embedding_health": dict(_mapping(multi_hour_long_test_report.get("final_embedder"))),
        "replay_safety_status": dict(replay_safety),
        "action_audit_status": action_audit,
        "recommended_operator_action": runtime_truth.get("recommended_action", ""),
        "operator_visible_report": {
            "summary": "Phase 14 multi-hour evidence passes." if passed else "Phase 14 multi-hour evidence is incomplete.",
            "remaining_bottlenecks": bottlenecks,
            "checks": checks,
        },
        "checks": checks,
    }


def validate_multi_hour_live_run_files(
    *,
    multi_hour_long_test_report_path: str | Path,
    baseline_long_test_report_path: str | Path,
    live_validation_report_path: str | Path,
    benchmark_report_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    report = validate_multi_hour_live_run(
        multi_hour_long_test_report=load_json_object(multi_hour_long_test_report_path, label="Multi-hour long-test report"),
        baseline_long_test_report=load_json_object(baseline_long_test_report_path, label="Baseline long-test report"),
        live_validation_report=load_json_object(live_validation_report_path, label="Live validation report"),
        benchmark_report=load_json_object(benchmark_report_path, label="Service benchmark report"),
    )
    write_json_report_with_readme(output_path, report, title="Terminus Multi-Hour Live Validation")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Terminus Phase 14 multi-hour living evidence.")
    parser.add_argument("--multi-hour-long-test", type=Path, required=True)
    parser.add_argument("--baseline-long-test", type=Path, required=True)
    parser.add_argument("--live-validation", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.indent < 0:
        parser.error("--indent must be non-negative")
    report = validate_multi_hour_live_run_files(
        multi_hour_long_test_report_path=args.multi_hour_long_test,
        baseline_long_test_report_path=args.baseline_long_test,
        live_validation_report_path=args.live_validation,
        benchmark_report_path=args.benchmark,
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
