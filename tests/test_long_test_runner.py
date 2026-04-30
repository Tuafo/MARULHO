from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from hecsn.cortex.core import MockCortex
from hecsn.cortex.episodic_memory import SimpleEmbedder
from hecsn.training.long_test_runner import (
    TestReport as LongTestReport,
    classify_test_report,
    health_exit_code,
    run_acceptance_harness,
    write_report,
)


def test_classify_test_report_marks_dead_empty_run() -> None:
    report = LongTestReport(
        cortex_available=True,
        samples_collected=3,
        initial_token_count=100,
        final_token_count=100,
        max_background_tokens_processed=0,
        final_tick_count=0,
        total_thoughts=0,
        acceptance_verdict="failed",
    )

    classify_test_report(report)

    assert report.health_verdict == "dead"
    assert any("No observable runtime progress" in reason for reason in report.health_reasons)
    assert any("Acceptance harness failed" in reason for reason in report.health_reasons)
    assert health_exit_code(report) == 2


def test_classify_test_report_marks_degraded_when_runtime_progresses_without_thoughts() -> None:
    report = LongTestReport(
        cortex_available=True,
        samples_collected=3,
        initial_token_count=100,
        final_token_count=160,
        max_background_tokens_processed=60,
        final_tick_count=3,
        total_thoughts=0,
        acceptance_verdict="passed",
    )

    classify_test_report(report)

    assert report.health_verdict == "degraded"
    assert any("produced no thoughts" in reason for reason in report.health_reasons)
    assert health_exit_code(report) == 1


def test_classify_test_report_uses_runtime_truth_contract_warnings() -> None:
    report = LongTestReport(
        cortex_available=True,
        samples_collected=3,
        initial_token_count=100,
        final_token_count=180,
        max_background_tokens_processed=80,
        final_tick_count=4,
        total_thoughts=5,
        acceptance_verdict="passed",
        final_runtime_truth={
            "verdict": "degraded",
            "recommended_action": "run_tick_or_start_runtime",
        },
    )

    classify_test_report(report)

    assert report.health_verdict == "degraded"
    assert any("Runtime truth contract reported degraded" in reason for reason in report.health_reasons)
    assert health_exit_code(report) == 1


def test_classify_test_report_marks_alive_run() -> None:
    report = LongTestReport(
        cortex_available=True,
        samples_collected=3,
        initial_token_count=100,
        final_token_count=180,
        max_background_tokens_processed=80,
        final_tick_count=4,
        total_thoughts=5,
        acceptance_verdict="passed",
    )

    classify_test_report(report)

    assert report.health_verdict == "alive"
    assert report.health_reasons == ["Run met the minimum activity and acceptance thresholds."]
    assert health_exit_code(report) == 0


def test_run_acceptance_harness_passes_with_mock_cortex() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("hecsn.cortex.multi_cortex.create_cortex_from_env", return_value=MockCortex()), patch(
            "hecsn.cortex.multi_cortex.create_embedder_from_env",
            return_value=SimpleEmbedder(),
        ):
            result = run_acceptance_harness(output_dir=tmpdir, env_root=Path.cwd())

    assert result["verdict"] == "passed"
    assert result["failed"] == 0
    check_names = {item["name"] for item in result["checks"]}
    assert check_names == {"idle_gating", "query_answer", "grounded_source_influence", "runtime_progress"}


def test_run_acceptance_harness_reports_partial_when_cortex_initialization_fails() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch(
            "hecsn.cortex.multi_cortex.create_cortex_from_env",
            side_effect=RuntimeError("mock cortex unavailable"),
        ), patch(
            "hecsn.cortex.multi_cortex.create_embedder_from_env",
            return_value=SimpleEmbedder(),
        ):
            result = run_acceptance_harness(output_dir=tmpdir, env_root=Path.cwd())

    idle_check = next(item for item in result["checks"] if item["name"] == "idle_gating")
    assert result["verdict"] == "partial"
    assert result["passed"] > 0
    assert result["failed"] == 1
    assert idle_check["passed"] is False
    assert idle_check["details"]["cortex_enabled"] is False


def test_write_report_handles_unicode_text_and_health_sections() -> None:
    report = LongTestReport(
        start_time="2026-04-21T00:00:00+00:00",
        end_time="2026-04-21T00:01:00+00:00",
        duration_minutes=1.0,
        sample_interval_s=5.0,
        preset="curriculum",
        cortex_model="multi(test-fast,test-deep)",
        cortex_available=True,
        terminus_configured=True,
        terminus_running=True,
        initial_token_count=100,
        final_token_count=180,
        max_background_tokens_processed=80,
        final_tick_count=4,
        total_thoughts=2,
        unique_topics=4,
        topic_diversity_ratio=2.0,
        avg_latency_ms=1234.0,
        health_verdict="alive",
        health_reasons=["Run met the minimum activity and acceptance thresholds."],
        acceptance_verdict="passed",
        acceptance_checks=[
            {"name": "idle_gating", "passed": True, "summary": "Idle cortex stayed quiet.", "details": {}},
            {"name": "query_answer", "passed": True, "summary": "Grounded answer returned.", "details": {}},
        ],
        acceptance_passed=2,
        acceptance_failed=0,
        final_runtime_truth={
            "schema_version": 1,
            "verdict": "alive",
            "recommended_action": "continue_monitoring",
        },
        final_narrative_summary="Coral reefs balance calcium carbonate growth with ocean chemistry — a fragile equilibrium.",
        sample_thoughts=[
            "Aurora Borealis occurs when charged solar particles strike Earth's atmosphere.",
            "Zonal reef growth depends on carbonate saturation — a delicate équilibre.",
        ],
        snapshots=[
            {
                "elapsed_s": 5.0,
                "token_count": 140,
                "thoughts": 1,
                "thoughts_delta": 1,
                "background_tokens_processed": 40,
                "tick_count": 2,
                "runtime_running": True,
                "latency_ms": 1200.0,
                "memory_fill": 0.25,
                "memory_size": 8,
                "consolidation": 0.1,
                "ripple_tagged": 1,
                "topic_diversity": 2,
                "prediction_error_mean": 0.2,
                "prediction_error_max": 0.4,
                "dream_verification_rate": 0.0,
                "depth_counts": {"quick": 1, "standard": 0, "deep": 0},
                "exploration_target": "aurora",
                "exploration_reason": "novelty",
                "embedder": {},
                "ingestion_state": "warm",
                "action_count": 1,
                "da": 0.5,
                "errors": 0,
            }
        ],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path, md_path = write_report(report, output_dir=tmpdir)
        json_data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        md_text = Path(md_path).read_text(encoding="utf-8")

    assert json_data["preset"] == "curriculum"
    assert json_data["health_verdict"] == "alive"
    assert json_data["final_runtime_truth"]["verdict"] == "alive"
    assert "## Health Verdict" in md_text
    assert "## Acceptance Harness" in md_text
    assert "Runtime truth verdict" in md_text
    assert "continue_monitoring" in md_text
    assert "équilibre" in md_text
    assert "Aurora Borealis" in md_text
