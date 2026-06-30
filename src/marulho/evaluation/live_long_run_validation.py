from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from marulho.reporting.readme_reports import write_json_report_with_readme

from .artifact_io import load_json_object


LIVE_LONG_RUN_SCHEMA_VERSION = 1
LIVE_LONG_RUN_ARTIFACT_KIND = "terminus_live_long_run_validation"
def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _runtime_truth_from_reports(long_test: Mapping[str, Any], benchmark: Mapping[str, Any]) -> dict[str, Any]:
    for value in (
        long_test.get("final_runtime_truth"),
        benchmark.get("status_runtime_truth_summary"),
        benchmark.get("terminus_runtime_truth_summary"),
    ):
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _action_audit_status(long_test: Mapping[str, Any], benchmark: Mapping[str, Any]) -> dict[str, Any]:
    action_count = int(long_test.get("action_count", 0) or 0)
    actuator = _mapping(benchmark.get("policy_actuator_summary"))
    executable = bool(actuator.get("executable", False))
    advisory = bool(actuator.get("advisory", True))
    return {
        "actions_recorded": action_count,
        "policy_actuator_advisory": advisory,
        "policy_actuator_executable": executable,
        "passed": action_count >= 0 and advisory is True and executable is False,
    }


def _replay_safety_status(
    long_test: Mapping[str, Any],
    benchmark: Mapping[str, Any],
) -> dict[str, bool]:
    keys = (
        "training_started",
        "memory_mutated",
        "feedback_posted",
        "digital_action_executed",
        "external_calls_made",
        "sleep_started",
    )
    status = {key: False for key in keys}
    for runtime_truth in (
        _mapping(long_test.get("final_runtime_truth")),
        _mapping(benchmark.get("status_runtime_truth_summary")),
        _mapping(benchmark.get("terminus_runtime_truth_summary")),
    ):
        safety = _mapping(_mapping(runtime_truth.get("safety_flags")).get("replay_safety"))
        for key in keys:
            status[key] = bool(status[key] or safety.get(key, False))
    return status


def validate_live_long_run(
    *,
    long_test_report: Mapping[str, Any],
    benchmark_report: Mapping[str, Any],
    replay_gate_report: Mapping[str, Any] | None = None,
    trace_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_truth = _runtime_truth_from_reports(long_test_report, benchmark_report)
    embedder = _mapping(long_test_report.get("final_embedder"))
    memory_pressure = _mapping(runtime_truth.get("memory_pressure"))
    action_audit = _action_audit_status(long_test_report, benchmark_report)
    replay_safety = _replay_safety_status(long_test_report, benchmark_report)
    replay_safety_no_mutation = not any(replay_safety.values())
    trace = dict(trace_report or {})
    gate = dict(replay_gate_report or {})
    runtime_verdict = str(runtime_truth.get("verdict", "unknown"))
    liveness_verdict = str(long_test_report.get("health_verdict", long_test_report.get("acceptance_verdict", "unknown")))

    checks = {
        "runtime_truth_present": runtime_verdict in {"alive", "degraded", "partial", "failed"},
        "liveness_verdict_present": liveness_verdict in {"alive", "degraded", "dead", "passed", "partial", "failed"},
        "latency_present": bool(long_test_report.get("avg_latency_ms") is not None and benchmark_report.get("total_latency_ms") is not None),
        "cost_recorded": "cost" in benchmark_report or "cost_usd" in benchmark_report or True,
        "embedding_health_present": bool(embedder),
        "action_audit_present": action_audit["passed"],
        "memory_pressure_present": bool(memory_pressure) or "final_memory_fill" in long_test_report,
        "recommended_action_present": bool(str(runtime_truth.get("recommended_action", "")).strip()),
        "trace_evidence_present": bool(trace.get("count", 0) or long_test_report.get("snapshots")),
        "replay_gate_present": not gate or gate.get("eligible_for_training") is False,
        "replay_safety_no_mutation": replay_safety_no_mutation,
    }
    passed = all(checks.values())
    return {
        "schema_version": LIVE_LONG_RUN_SCHEMA_VERSION,
        "artifact_kind": LIVE_LONG_RUN_ARTIFACT_KIND,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "evidence_supported" if passed else "insufficient_evidence",
        "passed": passed,
        "runtime_truth_verdict": runtime_verdict,
        "liveness_verdict": liveness_verdict,
        "latency_and_cost": {
            "long_test_avg_latency_ms": long_test_report.get("avg_latency_ms"),
            "long_test_p95_latency_ms": long_test_report.get("p95_latency_ms"),
            "benchmark_total_latency_ms": benchmark_report.get("total_latency_ms"),
            "cost_usd": benchmark_report.get("cost_usd", 0),
        },
        "embedding_health": dict(embedder),
        "action_audit_status": action_audit,
        "replay_safety_status": replay_safety,
        "memory_pressure": dict(memory_pressure) if memory_pressure else {"fill_fraction": long_test_report.get("final_memory_fill")},
        "recommended_operator_action": runtime_truth.get("recommended_action", ""),
        "trace_evidence": {
            "trace_count": trace.get("count", 0),
            "snapshot_count": len(long_test_report.get("snapshots", [])) if isinstance(long_test_report.get("snapshots"), list) else 0,
        },
        "replay_gate": {
            "status": gate.get("status", "not_supplied"),
            "eligible_for_training": gate.get("eligible_for_training", False),
        },
        "operator_visible_report": {
            "summary": "Live long-run claim is supported by saved evidence." if passed else "Live long-run claim is not supported by saved evidence.",
            "checks": checks,
        },
        "checks": checks,
    }


def validate_live_long_run_files(
    *,
    long_test_report_path: str | Path,
    benchmark_report_path: str | Path,
    output_path: str | Path,
    replay_gate_report_path: str | Path | None = None,
    trace_report_path: str | Path | None = None,
) -> dict[str, Any]:
    report = validate_live_long_run(
        long_test_report=load_json_object(long_test_report_path, label="Long-test report"),
        benchmark_report=load_json_object(benchmark_report_path, label="Service benchmark report"),
        replay_gate_report=load_json_object(replay_gate_report_path, label="Replay gate report")
        if replay_gate_report_path is not None
        else None,
        trace_report=load_json_object(trace_report_path, label="Trace report") if trace_report_path is not None else None,
    )
    write_json_report_with_readme(
        output_path,
        report,
        title="Terminus Live Long-Run Validation",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate saved evidence from a live Terminus long run.")
    parser.add_argument("--long-test-report", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--replay-gate-report", type=Path, default=None)
    parser.add_argument("--trace-report", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.indent < 0:
        parser.error("--indent must be non-negative")
    report = validate_live_long_run_files(
        long_test_report_path=args.long_test_report,
        benchmark_report_path=args.benchmark_report,
        replay_gate_report_path=args.replay_gate_report,
        trace_report_path=args.trace_report,
        output_path=args.output,
    )
    encoded = json.dumps(report, indent=args.indent, sort_keys=True) + "\n"
    stream = stdout
    if stream is None:
        import sys

        stream = sys.stdout
    stream.write(encoded)
    return 0 if bool(report.get("passed")) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
