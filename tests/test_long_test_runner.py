from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from hecsn.training.long_test_runner import (
    MetricSnapshot,
    TestReport as LongTestReport,
    classify_test_report,
    health_exit_code,
    run_acceptance_harness,
    _acceptance_failure_details,
    _summarize_global_workspace,
    _summarize_memory_pressure,
    _summarize_thought_lifecycle,
    run_long_test,
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


def test_classify_test_report_marks_alive_when_subcortex_progresses_without_thoughts() -> None:
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

    assert report.health_verdict == "alive"
    assert report.health_reasons == ["Run met the minimum activity and acceptance thresholds."]
    assert health_exit_code(report) == 0


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


def test_classify_test_report_warns_on_unrecovered_high_memory_pressure() -> None:
    report = LongTestReport(
        cortex_available=True,
        samples_collected=3,
        initial_token_count=100,
        final_token_count=180,
        max_background_tokens_processed=80,
        final_tick_count=4,
        total_thoughts=3,
        acceptance_verdict="passed",
        memory_pressure_report={
            "unrecovered_high_pressure": True,
            "recommended_action": "reduce_memory_fill_before_extending_runtime",
        },
    )

    classify_test_report(report)

    assert report.health_verdict == "degraded"
    assert any("Memory pressure reached the high band" in reason for reason in report.health_reasons)
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


def test_run_acceptance_harness_passes_without_cortex() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_acceptance_harness(output_dir=tmpdir, env_root=Path.cwd())

    assert result["verdict"] == "passed"
    assert result["failed"] == 0
    check_names = {item["name"] for item in result["checks"]}
    assert check_names == {"query_answer", "grounded_source_influence", "runtime_progress"}


def test_diagnostic_summaries_capture_phase_8_to_10_evidence() -> None:
    snapshots = [
        MetricSnapshot(
            memory_fill=0.40,
            memory_pressure={
                "fill_fraction": 0.40,
                "pressure": "low",
                "working_set_policy": {"decision": "continue_monitoring"},
            },
            thought_lifecycle={
                "attempts": 0,
                "successful": 0,
                "dreams": 0,
                "blocked_ticks": 2,
                "rejected_or_blocked_reason": "idle_no_trigger",
                "wake_triggers": {"last_gate_reason": "idle_no_trigger"},
            },
            global_workspace={
                "size": 0,
                "capacity": 5,
                "selected_context_items": [],
                "broadcast": "",
                "active_exploration": {},
                "evidence_boundary": {"hypotheses_promoted_to_fact": 0},
            },
        ),
        MetricSnapshot(
            memory_fill=1.0,
            thoughts_total=0,
            memory_pressure={
                "fill_fraction": 1.0,
                "pressure": "high",
                "working_set_policy": {
                    "decision": "throttle_ingestion_and_prioritize_consolidation",
                    "capacity_increase_recommended": False,
                    "replay_fact_promotion_allowed": False,
                },
            },
            thought_lifecycle={
                "attempts": 0,
                "successful": 0,
                "dreams": 0,
                "blocked_ticks": 10,
                "last_blocked": {"reason": "startup_quiet"},
                "rejected_or_blocked_reason": "startup_quiet",
                "wake_triggers": {"startup_quiet": True},
            },
            global_workspace={
                "size": 1,
                "capacity": 5,
                "selected_context_items": [
                    {"content": "possible explanation", "type": "hypothesis", "strength": 0.6}
                ],
                "broadcast": "Currently thinking about: possible explanation",
                "active_exploration": {"target": "memory pressure"},
                "evidence_boundary": {"hypothesis_items": 1, "hypotheses_promoted_to_fact": 0},
            },
        ),
    ]

    memory = _summarize_memory_pressure(snapshots)
    lifecycle = _summarize_thought_lifecycle(snapshots)
    workspace = _summarize_global_workspace(snapshots)

    assert memory["unrecovered_high_pressure"] is True
    assert memory["final_policy"]["capacity_increase_recommended"] is False
    assert lifecycle["successful"] == 0
    assert "startup_quiet" in lifecycle["rejected_or_blocked_reasons"]
    assert workspace["capacity"] == 5
    assert workspace["evidence_boundary"]["hypotheses_promoted_to_fact"] == 0


def test_run_long_test_skips_missed_samples_after_slow_snapshot() -> None:
    clock = {"now": 1_000.0}

    class Manager:
        snapshot_status_calls = 0

        def __init__(self, *args, **kwargs) -> None:
            self.closed = False

        def status(self, *, fresh_wait_seconds: float | None = None) -> dict:
            if fresh_wait_seconds is not None:
                Manager.snapshot_status_calls += 1
                clock["now"] += 130.0
            return {
                "token_count": 200,
                "memory_store": {"fill_fraction": 0.10, "size": 2, "capacity": 100},
                "runtime_truth": {"verdict": "alive", "recommended_action": "continue_monitoring"},
                "terminus_runtime": {
                    "configured": True,
                    "running": True,
                    "background_tokens_processed": 100,
                    "tick_count": 10,
                    "cortex": {
                        "enabled": True,
                        "running": True,
                        "current_mode": "idle",
                        "thoughts_generated": 1,
                        "thought_lifecycle": {
                            "attempts": 1,
                            "successful": 1,
                            "blocked_ticks": 0,
                        },
                    },
                },
            }

        def quick_start_terminus(self, *, preset: str) -> dict:
            return {"status": "ok", "terminus_runtime": {"configured": True, "running": True}}

        def close(self) -> None:
            self.closed = True

    def fake_sleep(seconds: float) -> None:
        clock["now"] += max(0.0, float(seconds))

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("hecsn.training.long_test_runner.run_acceptance_harness", return_value={"verdict": "passed"}), patch(
            "hecsn.training.long_test_runner._build_checkpoint",
            return_value=Path(tmpdir) / "checkpoint.pt",
        ), patch("hecsn.service.manager.HECSNServiceManager", Manager), patch(
            "hecsn.training.long_test_runner.time.time", side_effect=lambda: clock["now"]
        ), patch("hecsn.training.long_test_runner.time.sleep", side_effect=fake_sleep):
            report = run_long_test(
                duration_minutes=3.0,
                sample_interval_s=60.0,
                output_dir=tmpdir,
            )

    assert Manager.snapshot_status_calls == 1
    assert report.samples_collected == 1
    assert report.snapshots[0]["elapsed_s"] >= 180.0


def test_acceptance_failure_details_preserve_failed_check_context() -> None:
    failures = _acceptance_failure_details(
        [
            {"name": "idle_gating", "passed": True, "summary": "ok", "details": {}},
            {
                "name": "query_answer",
                "passed": False,
                "summary": "missing grounded answer",
                "details": {"response_text": ""},
            },
        ]
    )

    assert failures == [
        {
            "name": "query_answer",
            "summary": "missing grounded answer",
            "details": {"response_text": ""},
            "recommended_action": "inspect_acceptance_path",
        }
    ]


def test_write_report_handles_unicode_text_and_health_sections() -> None:
    report = LongTestReport(
        start_time="2026-04-21T00:00:00+00:00",
        end_time="2026-04-21T00:01:00+00:00",
        duration_minutes=1.0,
        sample_interval_s=5.0,
        preset="curriculum",
        memory_capacity=16384,
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
        thought_lifecycle_summary={
            "attempts": 2,
            "successful": 2,
            "dreams": 0,
            "blocked_ticks": 4,
            "wake_triggers": {"last_gate_reason": "prediction_error"},
            "rejected_or_blocked_reasons": ["interval_cooldown"],
        },
        memory_pressure_report={
            "first_fill": 0.25,
            "final_fill": 0.50,
            "max_fill": 0.50,
            "high_samples": 0,
            "recovered_from_high": False,
            "unrecovered_high_pressure": False,
            "recommended_action": "continue_monitoring",
            "final_policy": {"capacity_increase_recommended": False},
        },
        global_workspace_report={
            "final_size": 1,
            "capacity": 5,
            "max_size": 2,
            "active_exploration": {"target": "aurora"},
            "evidence_boundary": {"hypotheses_promoted_to_fact": 0},
            "final_broadcast": "Currently thinking about: aurora",
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
                "thought_lifecycle": {
                    "attempts": 2,
                    "successful": 1,
                    "blocked_ticks": 4,
                    "rejected_or_blocked_reason": "interval_cooldown",
                },
                "memory_pressure": {
                    "fill_fraction": 0.25,
                    "pressure": "low",
                    "working_set_policy": {"capacity_increase_recommended": False},
                },
                "global_workspace": {
                    "size": 1,
                    "capacity": 5,
                    "selected_context_items": [],
                    "broadcast": "Currently thinking about: aurora",
                    "evidence_boundary": {"hypotheses_promoted_to_fact": 0},
                },
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
        readme_text = Path(tmpdir, "README.md").read_text(encoding="utf-8")

    assert json_data["preset"] == "curriculum"
    assert json_data["memory_capacity"] == 16384
    assert json_data["health_verdict"] == "alive"
    assert json_data["final_runtime_truth"]["verdict"] == "alive"
    assert "## Health Verdict" in md_text
    assert "Memory capacity" in md_text
    assert "## Acceptance Harness" in md_text
    assert "## Source Configuration" in md_text
    assert "## Liveness Diagnosis" in md_text
    assert "## Memory Pressure" in md_text
    assert "## Global Workspace" in md_text
    assert "Runtime truth verdict" in md_text
    assert "continue_monitoring" in md_text
    assert "équilibre" in md_text
    assert "Aurora Borealis" in md_text
    assert readme_text == md_text
