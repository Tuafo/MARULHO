from __future__ import annotations

import json
from pathlib import Path
import tempfile

from marulho.evaluation.live_long_run_validation import validate_live_long_run, validate_live_long_run_files


def _long_test() -> dict[str, object]:
    return {
        "health_verdict": "alive",
        "avg_latency_ms": 12.0,
        "p95_latency_ms": 20.0,
        "final_embedder": {"kind": "SimpleEmbedder", "available": False, "degraded": False, "fallback_calls": 0},
        "final_memory_fill": 0.1,
        "action_count": 0,
        "snapshots": [{"runtime_truth": {"verdict": "alive"}}],
        "final_runtime_truth": {
            "schema_version": 1,
            "verdict": "alive",
            "recommended_action": "continue_monitoring",
            "memory_pressure": {"fill_fraction": 0.1, "pressure": "low"},
            "safety_flags": {
                "replay_safety": {
                    "training_started": False,
                    "memory_mutated": False,
                    "feedback_posted": False,
                    "digital_action_executed": False,
                    "external_calls_made": False,
                    "sleep_started": False,
                }
            },
        },
    }


def _benchmark(unsafe: bool = False) -> dict[str, object]:
    return {
        "schema_version": 1,
        "success": True,
        "total_latency_ms": 100.0,
        "cost_usd": 0,
        "policy_actuator_summary": {"advisory": True, "executable": False},
        "status_runtime_truth_summary": {
            "schema_version": 1,
            "verdict": "alive",
            "recommended_action": "continue_monitoring",
            "memory_pressure": {"fill_fraction": 0.1, "pressure": "low"},
            "safety_flags": {
                "replay_safety": {
                    "training_started": False,
                    "memory_mutated": False,
                    "feedback_posted": False,
                    "digital_action_executed": unsafe,
                    "external_calls_made": False,
                    "sleep_started": False,
                }
            },
        },
    }


def test_live_long_run_validation_passes_when_saved_reports_support_claim() -> None:
    report = validate_live_long_run(
        long_test_report=_long_test(),
        benchmark_report=_benchmark(),
        replay_gate_report={"status": "passed_pending_operator_training_approval", "eligible_for_training": False},
        trace_report={"count": 3},
    )

    assert report["passed"] is True
    assert report["runtime_truth_verdict"] == "alive"
    assert report["liveness_verdict"] == "alive"
    assert report["action_audit_status"]["passed"] is True
    assert report["operator_visible_report"]["checks"]["replay_safety_no_mutation"] is True


def test_live_long_run_validation_blocks_missing_runtime_truth() -> None:
    long_test = _long_test()
    long_test.pop("final_runtime_truth")
    benchmark = _benchmark()
    benchmark.pop("status_runtime_truth_summary")

    report = validate_live_long_run(long_test_report=long_test, benchmark_report=benchmark)

    assert report["passed"] is False
    assert report["checks"]["runtime_truth_present"] is False
    assert report["status"] == "insufficient_evidence"


def test_live_long_run_validation_blocks_unsafe_replay_flag() -> None:
    report = validate_live_long_run(long_test_report=_long_test(), benchmark_report=_benchmark(unsafe=True))

    assert report["passed"] is False
    assert report["checks"]["replay_safety_no_mutation"] is False
    assert report["replay_safety_status"]["digital_action_executed"] is True


def test_live_long_run_validation_file_writes_evidence_report() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        long_path = root / "long.json"
        benchmark_path = root / "benchmark.json"
        output_path = root / "validation.json"
        long_path.write_text(json.dumps(_long_test()), encoding="utf-8")
        benchmark_path.write_text(json.dumps(_benchmark()), encoding="utf-8")

        report = validate_live_long_run_files(
            long_test_report_path=long_path,
            benchmark_report_path=benchmark_path,
            output_path=output_path,
        )
        loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert loaded["latency_and_cost"]["cost_usd"] == 0
    assert "retired_runtime_path" not in loaded
    assert loaded["recommended_operator_action"] == "continue_monitoring"
