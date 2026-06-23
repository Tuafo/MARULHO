from __future__ import annotations

import json
from pathlib import Path
import tempfile

from marulho.evaluation.multi_hour_live_validation import (
    validate_multi_hour_live_run,
    validate_multi_hour_live_run_files,
)


def _baseline() -> dict[str, object]:
    return {
        "duration_minutes": 30.0,
        "health_verdict": "alive",
        "initial_token_count": 100,
        "final_token_count": 180,
        "max_background_tokens_processed": 80,
        "final_tick_count": 4,
        "total_readouts": 0,
        "p95_latency_ms": 1000.0,
    }


def _multi_hour(memory_fill: float = 0.2, health: str = "alive") -> dict[str, object]:
    return {
        "duration_minutes": 120.0,
        "health_verdict": health,
        "acceptance_verdict": "passed",
        "initial_token_count": 100,
        "final_token_count": 400,
        "max_background_tokens_processed": 300,
        "final_tick_count": 12,
        "total_readouts": 0,
        "unique_topics": 20,
        "topic_diversity_ratio": 0.47,
        "avg_latency_ms": 750.0,
        "p95_latency_ms": 1800.0,
        "final_memory_fill": memory_fill,
        "memory_pressure_report": {
            "final_fill": memory_fill,
            "max_fill": memory_fill,
            "unrecovered_high_pressure": False,
            "recovered_from_high": False,
        },
        "final_embedder": {"available": True, "degraded": False, "fallback_calls": 0},
        "snapshots": [{"elapsed_s": 60.0}],
        "final_runtime_truth": {
            "verdict": "alive",
            "recommended_action": "continue_monitoring",
            "safety_flags": {},
        },
    }


def _live_validation() -> dict[str, object]:
    return {
        "status": "evidence_supported",
        "passed": True,
        "trace_evidence": {"trace_count": 3},
    }


def _benchmark() -> dict[str, object]:
    return {
        "success": True,
        "total_latency_ms": 100.0,
        "cost_usd": 0,
        "policy_actuator_summary": {"advisory": True, "executable": False},
    }


def test_phase14_multi_hour_validation_passes_alive_bounded_run() -> None:
    report = validate_multi_hour_live_run(
        multi_hour_long_test_report=_multi_hour(),
        baseline_long_test_report=_baseline(),
        live_validation_report=_live_validation(),
        benchmark_report=_benchmark(),
    )

    assert report["passed"] is True
    assert report["status"] == "passed_multi_hour_living_evidence"
    assert report["checks"]["duration_is_multi_hour"] is True
    assert report["checks"]["runtime_progress_present"] is True
    assert report["operator_visible_report"]["remaining_bottlenecks"] == ["no_blocking_bottleneck_detected"]


def test_phase14_blocks_degraded_multi_hour_health() -> None:
    report = validate_multi_hour_live_run(
        multi_hour_long_test_report=_multi_hour(health="degraded"),
        baseline_long_test_report=_baseline(),
        live_validation_report=_live_validation(),
        benchmark_report=_benchmark(),
    )

    assert report["passed"] is False
    assert report["checks"]["long_test_health_alive"] is False
    assert report["status"] == "blocked_multi_hour_living_claim"


def test_phase14_blocks_unrecovered_memory_pressure() -> None:
    multi_hour = _multi_hour(memory_fill=1.0)
    multi_hour["memory_pressure_report"] = {
        "final_fill": 1.0,
        "max_fill": 1.0,
        "unrecovered_high_pressure": True,
        "recovered_from_high": False,
    }

    report = validate_multi_hour_live_run(
        multi_hour_long_test_report=multi_hour,
        baseline_long_test_report=_baseline(),
        live_validation_report=_live_validation(),
        benchmark_report=_benchmark(),
    )

    assert report["passed"] is False
    assert report["checks"]["memory_pressure_bounded_or_recovered"] is False
    assert "memory_fill_near_limit" in report["operator_visible_report"]["remaining_bottlenecks"]


def test_phase14_file_writes_readme_report() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        multi_path = root / "multi.json"
        baseline_path = root / "baseline.json"
        live_path = root / "live.json"
        benchmark_path = root / "benchmark.json"
        output_path = root / "phase14.json"
        multi_path.write_text(json.dumps(_multi_hour()), encoding="utf-8")
        baseline_path.write_text(json.dumps(_baseline()), encoding="utf-8")
        live_path.write_text(json.dumps(_live_validation()), encoding="utf-8")
        benchmark_path.write_text(json.dumps(_benchmark()), encoding="utf-8")

        report = validate_multi_hour_live_run_files(
            multi_hour_long_test_report_path=multi_path,
            baseline_long_test_report_path=baseline_path,
            live_validation_report_path=live_path,
            benchmark_report_path=benchmark_path,
            output_path=output_path,
        )

        assert report["passed"] is True
        assert output_path.exists()
        assert (output_path.parent / "README.md").exists()
