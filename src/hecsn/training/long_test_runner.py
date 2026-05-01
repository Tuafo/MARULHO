"""Long-Test Runner -- exercises the full Terminus runtime and classifies run health.

Usage:
    python -m hecsn.training.long_test_runner --duration 20 --output reports/long_test_report.md

This runs Terminus with the curriculum preset for N minutes (default 20),
collects runtime metrics, executes a small deterministic acceptance harness,
and outputs a JSON + markdown report.

The report explicitly classifies the run as:
- alive
- degraded
- dead

so empty or stalled runs no longer read as clean success.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import logging
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

DEFAULT_HEALTH_THRESHOLDS: dict[str, int] = {
    "min_samples": 1,
    "min_runtime_progress_tokens": 1,
    "min_total_thoughts_alive": 1,
}
DEFAULT_CORTEX_INIT_WAIT_S = 15.0
DEFAULT_FINAL_THOUGHT_WAIT_S = 90.0
DEFAULT_LONG_TEST_MEMORY_CAPACITY = 16_384


@dataclass
class MetricSnapshot:
    """One measurement point during the test run."""

    timestamp: float = 0.0
    elapsed_s: float = 0.0
    token_count: int = 0
    thoughts_total: int = 0
    thoughts_delta: int = 0
    background_tokens_processed: int = 0
    tick_count: int = 0
    runtime_running: bool = False
    memory_fill: float = 0.0
    memory_size: int = 0
    consolidation_mean: float = 0.0
    ripple_tagged: int = 0
    cortex_latency_ms: float = 0.0
    topic_diversity: int = 0
    da_level: float = 0.0
    serotonin_level: float = 0.0
    ach_level: float = 0.0
    ne_level: float = 0.0
    prediction_error_mean: float = 0.0
    prediction_error_max: float = 0.0
    dream_verification_rate: float = 0.0
    depth_counts: dict[str, int] = field(default_factory=dict)
    cortex_model: str = ""
    narrative_summary: str = ""
    exploration_target: str = ""
    exploration_reason: str = ""
    embedder: dict[str, Any] = field(default_factory=dict)
    runtime_truth: dict[str, Any] = field(default_factory=dict)
    thought_lifecycle: dict[str, Any] = field(default_factory=dict)
    memory_pressure: dict[str, Any] = field(default_factory=dict)
    global_workspace: dict[str, Any] = field(default_factory=dict)
    ingestion_state: str = ""
    action_count: int = 0
    errors: int = 0


@dataclass
class TestReport:
    """Final report from a long test run."""

    start_time: str = ""
    end_time: str = ""
    duration_minutes: float = 0.0
    sample_interval_s: float = 0.0
    preset: str = ""
    memory_capacity: int = DEFAULT_LONG_TEST_MEMORY_CAPACITY
    cortex_model: str = ""
    cortex_available: bool = False
    terminus_configured: bool = False
    terminus_running: bool = False
    initial_token_count: int = 0
    final_token_count: int = 0
    total_thoughts: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    final_memory_fill: float = 0.0
    final_consolidation: float = 0.0
    final_ripple_tagged: int = 0
    unique_topics: int = 0
    topic_diversity_ratio: float = 0.0
    final_prediction_error_mean: float = 0.0
    final_prediction_error_max: float = 0.0
    final_dream_verification_rate: float = 0.0
    depth_counts: dict[str, int] = field(default_factory=dict)
    all_topics: list[str] = field(default_factory=list)
    final_narrative_summary: str = ""
    final_exploration_target: str = ""
    final_exploration_reason: str = ""
    final_embedder: dict[str, Any] = field(default_factory=dict)
    final_runtime_truth: dict[str, Any] = field(default_factory=dict)
    samples_collected: int = 0
    max_background_tokens_processed: int = 0
    final_background_tokens_processed: int = 0
    final_tick_count: int = 0
    action_count: int = 0
    health_verdict: str = "dead"
    health_reasons: list[str] = field(default_factory=list)
    health_thresholds: dict[str, Any] = field(default_factory=dict)
    acceptance_verdict: str = "not_run"
    acceptance_checks: list[dict[str, Any]] = field(default_factory=list)
    acceptance_passed: int = 0
    acceptance_failed: int = 0
    acceptance_skipped: int = 0
    acceptance_failure_details: list[dict[str, Any]] = field(default_factory=list)
    thought_lifecycle_summary: dict[str, Any] = field(default_factory=dict)
    memory_pressure_report: dict[str, Any] = field(default_factory=dict)
    global_workspace_report: dict[str, Any] = field(default_factory=dict)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    sample_thoughts: list[str] = field(default_factory=list)


def _build_checkpoint(
    root: Path,
    *,
    test_name: str,
    n_columns: int,
    column_latent_dim: int,
    memory_capacity: int,
) -> Path:
    from hecsn.config.model_config import HECSNConfig
    from hecsn.training.checkpointing import save_trainer_checkpoint
    from hecsn.training.trainer import HECSNModel, HECSNTrainer

    cfg = HECSNConfig(
        n_columns=n_columns,
        column_latent_dim=column_latent_dim,
        bootstrap_tokens=0,
        memory_capacity=memory_capacity,
        eta_competitive=0.05,
        enable_context_layer=True,
        enable_binding_layer=True,
    )
    trainer = HECSNTrainer(HECSNModel(cfg), cfg)
    return save_trainer_checkpoint(
        root / "test_checkpoint.pt",
        trainer,
        metadata={"test": test_name},
    )


def _acceptance_check(name: str, passed: bool, summary: str, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": str(name),
        "passed": bool(passed),
        "summary": str(summary),
        "details": dict(details or {}),
    }


def _summarize_acceptance_checks(checks: list[dict[str, Any]]) -> tuple[str, int, int, int]:
    passed = sum(1 for item in checks if bool(item.get("passed", False)))
    failed = sum(1 for item in checks if not bool(item.get("passed", False)))
    skipped = sum(1 for item in checks if str(item.get("summary", "")).lower().startswith("skipped:"))
    if checks and failed == 0:
        verdict = "passed"
    elif passed > 0:
        verdict = "partial"
    else:
        verdict = "failed"
    return verdict, passed, failed, skipped


def _acceptance_failure_details(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for item in checks:
        if bool(item.get("passed", False)):
            continue
        failures.append(
            {
                "name": str(item.get("name", "")),
                "summary": str(item.get("summary", "")),
                "details": dict(item.get("details") or {}) if isinstance(item.get("details"), Mapping) else {},
                "recommended_action": "inspect_acceptance_path",
            }
        )
    return failures


def _memory_pressure_band(fill_fraction: float) -> str:
    fill = max(0.0, min(1.0, float(fill_fraction)))
    if fill >= 0.85:
        return "high"
    if fill >= 0.50:
        return "medium"
    return "low"


def _memory_pressure_snapshot(
    memory_store: Mapping[str, Any],
    runtime_truth: Mapping[str, Any],
) -> dict[str, Any]:
    fill = float(memory_store.get("fill_fraction", 0.0) or 0.0)
    size = int(memory_store.get("size", 0) or 0)
    capacity = int(memory_store.get("capacity", 0) or 0)
    pressure = _memory_pressure_band(fill)
    runtime_pressure = runtime_truth.get("memory_pressure") if isinstance(runtime_truth.get("memory_pressure"), Mapping) else {}
    action = "continue_monitoring"
    if pressure == "high":
        action = "throttle_ingestion_and_prioritize_consolidation"
    elif pressure == "medium":
        action = "watch_working_set_growth"
    return {
        "fill_fraction": fill,
        "size": size,
        "capacity": capacity,
        "pressure": pressure,
        "runtime_pressure": dict(runtime_pressure),
        "working_set_policy": {
            "high_threshold": 0.85,
            "target_fill": 0.70,
            "capacity_increase_recommended": False,
            "replay_fact_promotion_allowed": False,
            "decision": action,
        },
        "consolidation_mean": float(memory_store.get("mean_consolidation_level", 0.0) or 0.0),
        "mean_fragility": float(memory_store.get("mean_fragility", 0.0) or 0.0),
        "max_fragility": float(memory_store.get("max_fragility", 0.0) or 0.0),
        "n_seen": int(memory_store.get("n_seen", 0) or 0),
    }


def _thought_lifecycle_snapshot(
    cortex_snapshot: Mapping[str, Any],
    thoughts_data: Mapping[str, Any],
) -> dict[str, Any]:
    gating = cortex_snapshot.get("gating") if isinstance(cortex_snapshot.get("gating"), Mapping) else {}
    lifecycle = cortex_snapshot.get("thought_lifecycle") if isinstance(cortex_snapshot.get("thought_lifecycle"), Mapping) else {}
    recent = list(thoughts_data.get("thoughts") or []) if isinstance(thoughts_data, Mapping) else []
    successful = int(thoughts_data.get("thoughts_generated", cortex_snapshot.get("thoughts_generated", 0)) or 0)
    attempts = int(lifecycle.get("attempts", successful) or 0)
    blocked_ticks = int(lifecycle.get("blocked_ticks", 0) or 0)
    last_blocked = dict(lifecycle.get("last_blocked") or {}) if isinstance(lifecycle.get("last_blocked"), Mapping) else {}
    if successful <= 0 and attempts <= 0:
        rejected_reason = str(last_blocked.get("reason") or gating.get("inhibition_reason") or gating.get("last_gate_reason") or "no_deliberation_attempt")
    elif attempts > successful:
        rejected_reason = str(last_blocked.get("reason") or "some_attempts_blocked_or_rejected")
    else:
        rejected_reason = ""
    return {
        "enabled": bool(cortex_snapshot.get("enabled", False)),
        "running": bool(cortex_snapshot.get("running", False)),
        "mode": str(cortex_snapshot.get("current_mode", "")),
        "attempts": attempts,
        "successful": successful,
        "dreams": int(cortex_snapshot.get("dreams_generated", 0) or 0),
        "blocked_ticks": blocked_ticks,
        "recent_thought_count": len(recent),
        "wake_triggers": {
            "pending_grounded_observations": int(gating.get("pending_grounded_observations", 0) or 0),
            "pending_substrate_wakes": int(gating.get("pending_substrate_wakes", 0) or 0),
            "active_tension_count": int(gating.get("active_tension_count", 0) or 0),
            "last_gate_reason": str(gating.get("last_gate_reason", "")),
            "inhibition_reason": str(gating.get("inhibition_reason", "")),
            "startup_quiet": bool(gating.get("startup_quiet", False)),
        },
        "depth_policy": dict(cortex_snapshot.get("depth_policy") or {}) if isinstance(cortex_snapshot.get("depth_policy"), Mapping) else {},
        "last_attempt": dict(lifecycle.get("last_attempt") or {}) if isinstance(lifecycle.get("last_attempt"), Mapping) else {},
        "last_blocked": last_blocked,
        "rejected_or_blocked_reason": rejected_reason,
    }


def _global_workspace_snapshot(cortex_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    working_memory = cortex_snapshot.get("working_memory") if isinstance(cortex_snapshot.get("working_memory"), Mapping) else {}
    items = [
        dict(item)
        for item in list(working_memory.get("items") or [])[:5]
        if isinstance(item, Mapping)
    ]
    active_exploration = cortex_snapshot.get("active_exploration") if isinstance(cortex_snapshot.get("active_exploration"), Mapping) else {}
    narrative = cortex_snapshot.get("narrative_self") if isinstance(cortex_snapshot.get("narrative_self"), Mapping) else {}
    return {
        "capacity": int(working_memory.get("capacity", 0) or 0),
        "size": int(working_memory.get("size", 0) or 0),
        "selected_context_items": items,
        "broadcast": str(working_memory.get("broadcast", ""))[:300],
        "has_tension": bool(working_memory.get("has_tension", False)),
        "has_question": bool(working_memory.get("has_question", False)),
        "active_exploration": dict(active_exploration),
        "narrative_summary": str(narrative.get("summary", "")),
        "evidence_boundary": {
            "grounded_items": sum(1 for item in items if str(item.get("type", "")) in {"observation", "insight"}),
            "hypothesis_items": sum(1 for item in items if str(item.get("type", "")) == "hypothesis"),
            "hypotheses_promoted_to_fact": 0,
        },
    }


def _summarize_memory_pressure(snapshots: list[MetricSnapshot]) -> dict[str, Any]:
    if not snapshots:
        return {}
    fills = [float(item.memory_pressure.get("fill_fraction", item.memory_fill) or 0.0) for item in snapshots]
    pressures = [str(item.memory_pressure.get("pressure") or _memory_pressure_band(item.memory_fill)) for item in snapshots]
    high_indices = [idx for idx, pressure in enumerate(pressures) if pressure == "high"]
    recovered = bool(high_indices and pressures[-1] != "high")
    unrecovered = bool(high_indices and pressures[-1] == "high")
    return {
        "first_fill": fills[0],
        "final_fill": fills[-1],
        "max_fill": max(fills),
        "high_samples": len(high_indices),
        "medium_samples": sum(1 for pressure in pressures if pressure == "medium"),
        "low_samples": sum(1 for pressure in pressures if pressure == "low"),
        "recovered_from_high": recovered,
        "unrecovered_high_pressure": unrecovered,
        "final_policy": dict(snapshots[-1].memory_pressure.get("working_set_policy") or {}),
        "recommended_action": (
            "reduce_memory_fill_before_extending_runtime"
            if unrecovered
            else "continue_monitoring" if not high_indices else "confirm_pressure_recovery"
        ),
    }


def _summarize_thought_lifecycle(snapshots: list[MetricSnapshot]) -> dict[str, Any]:
    if not snapshots:
        return {}
    final = snapshots[-1].thought_lifecycle
    blocked_reasons = [
        str(item.thought_lifecycle.get("rejected_or_blocked_reason", ""))
        for item in snapshots
        if str(item.thought_lifecycle.get("rejected_or_blocked_reason", "")).strip()
    ]
    return {
        "attempts": int(final.get("attempts", 0) or 0),
        "successful": int(final.get("successful", snapshots[-1].thoughts_total) or 0),
        "dreams": int(final.get("dreams", 0) or 0),
        "blocked_ticks": int(final.get("blocked_ticks", 0) or 0),
        "last_blocked": dict(final.get("last_blocked") or {}),
        "wake_triggers": dict(final.get("wake_triggers") or {}),
        "rejected_or_blocked_reasons": sorted(set(blocked_reasons))[:8],
    }


def _summarize_global_workspace(snapshots: list[MetricSnapshot]) -> dict[str, Any]:
    if not snapshots:
        return {}
    final = snapshots[-1].global_workspace
    max_size = max(int(item.global_workspace.get("size", 0) or 0) for item in snapshots)
    return {
        "final_size": int(final.get("size", 0) or 0),
        "capacity": int(final.get("capacity", 0) or 0),
        "max_size": max_size,
        "final_selected_context_items": list(final.get("selected_context_items") or []),
        "final_broadcast": str(final.get("broadcast", "")),
        "active_exploration": dict(final.get("active_exploration") or {}),
        "evidence_boundary": dict(final.get("evidence_boundary") or {}),
    }


def run_acceptance_harness(
    *,
    output_dir: str = "reports",
    env_root: str | Path | None = None,
    idle_wait_s: float = 0.35,
    tick_steps: int = 2,
    cortex_wait_s: float = DEFAULT_CORTEX_INIT_WAIT_S,
) -> dict[str, Any]:
    """Run a small deterministic acceptance harness on maintained runtime paths."""

    from hecsn.config.runtime_env import load_runtime_env
    from hecsn.service.manager import HECSNServiceManager

    env_anchor = Path(env_root) if env_root is not None else Path.cwd()
    load_runtime_env(anchor_paths=(env_anchor,))

    acceptance_tmp_root = Path(output_dir) / ".acceptance_tmp"
    acceptance_tmp_root.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="acceptance_", dir=str(acceptance_tmp_root)) as tmpdir:
        root = Path(tmpdir)
        notes_path = root / "notes.md"
        stream_path = root / "stream.txt"
        notes_path.write_text(
            "Cats rest indoors during the day.\nCats chase mice at night.\n",
            encoding="utf-8",
        )
        stream_path.write_text("character stream learning " * 32, encoding="utf-8")
        checkpoint_path = _build_checkpoint(
            root,
            test_name="acceptance_harness",
            n_columns=4,
            column_latent_dim=8,
            memory_capacity=64,
        )

        manager = HECSNServiceManager(
            checkpoint_path,
            trace_dir=root / "traces",
            env_root=root,
        )
        try:
            manager._ensure_cortex_initialized(wait_seconds=max(0.0, float(cortex_wait_s)))
            cortex_snapshot = manager.cortex_snapshot()
            if not bool(cortex_snapshot.get("enabled", False)) or getattr(manager, "_thought_loop", None) is None:
                checks.append(
                    _acceptance_check(
                        "idle_gating",
                        False,
                        "Cortex was unavailable, so idle gating could not be exercised.",
                        {"cortex_enabled": bool(cortex_snapshot.get("enabled", False))},
                    )
                )
            else:
                manager._thought_loop.start()
                before = manager.cortex_snapshot()
                time.sleep(max(0.0, float(idle_wait_s)))
                after = manager.cortex_snapshot()
                thoughts_before = int(before.get("thoughts_generated", 0) or 0)
                thoughts_after = int(after.get("thoughts_generated", 0) or 0)
                passed = thoughts_before == 0 and thoughts_after == 0 and not list(after.get("recent_thoughts", []))
                checks.append(
                    _acceptance_check(
                        "idle_gating",
                        passed,
                        "Idle cortex stayed quiet before any explicit query or grounded wake." if passed else "Idle cortex generated unexpected thoughts without an explicit wake.",
                        {
                            "thoughts_before": thoughts_before,
                            "thoughts_after": thoughts_after,
                            "recent_thought_count": len(list(after.get("recent_thoughts", []))),
                        },
                    )
                )

            response_bundle = manager.respond(
                query_text="What does notes.md say cats chase at night?",
                max_evidence_items=3,
                learn_mode="none",
            )
            response = response_bundle.get("response") if isinstance(response_bundle.get("response"), Mapping) else {}
            query_result = response_bundle.get("query_result") if isinstance(response_bundle.get("query_result"), Mapping) else {}
            action_assist = query_result.get("action_assist") if isinstance(query_result.get("action_assist"), Mapping) else {}
            assist_result = action_assist.get("result") if isinstance(action_assist.get("result"), Mapping) else {}
            verification = assist_result.get("verification") if isinstance(assist_result.get("verification"), Mapping) else {}
            evidence = [
                dict(item)
                for item in list(verification.get("evidence") or [])
                if isinstance(item, Mapping)
            ]
            response_text = str(response.get("response_text", ""))
            query_answer_passed = bool(response_text.strip()) and "mice" in response_text.lower()
            grounded_source_passed = (
                str(assist_result.get("action_type", "")) == "workspace_read"
                and str(verification.get("status", "")) == "verified"
                and any(str(item.get("path", "")) == "notes.md" for item in evidence)
            )
            checks.append(
                _acceptance_check(
                    "query_answer",
                    query_answer_passed,
                    "Query response returned a grounded answer mentioning the expected fact." if query_answer_passed else "Query response did not return the expected grounded answer text.",
                    {
                        "response_text": response_text,
                        "action_reason": str(action_assist.get("reason", "")),
                    },
                )
            )
            checks.append(
                _acceptance_check(
                    "grounded_source_influence",
                    grounded_source_passed,
                    "Grounded workspace evidence from notes.md influenced the query response." if grounded_source_passed else "Expected grounded workspace evidence from notes.md was not preserved through the response path.",
                    {
                        "action_type": str(assist_result.get("action_type", "")),
                        "verification_status": str(verification.get("status", "")),
                        "evidence_paths": [str(item.get("path", "")) for item in evidence],
                    },
                )
            )

            config_result = manager.configure_terminus(
                source_bank=[
                    {
                        "name": "acceptance_stream",
                        "source": str(stream_path),
                        "source_type": "file",
                    }
                ],
                tick_tokens=20,
                sleep_interval_seconds=0.01,
                repeat_sources=False,
                ingestion={
                    "enabled": True,
                    "queue_target_tokens": 40,
                    "prewarm_on_startup": False,
                    "prewarm_max_seconds": 0.2,
                },
            )
            token_before = int(config_result.get("token_count", 0) or 0)
            tick_result = manager.terminus_tick(steps=max(1, int(tick_steps)))
            runtime = tick_result.get("terminus_runtime") if isinstance(tick_result.get("terminus_runtime"), Mapping) else {}
            runtime_progress_passed = (
                int(tick_result.get("token_count", 0) or 0) > token_before
                and int(runtime.get("background_tokens_processed", 0) or 0) > 0
                and int(runtime.get("last_tick_token_delta", 0) or 0) > 0
            )
            checks.append(
                _acceptance_check(
                    "runtime_progress",
                    runtime_progress_passed,
                    "Configured Terminus runtime made observable training progress on the maintained tick path." if runtime_progress_passed else "Configured Terminus runtime did not make observable progress on the maintained tick path.",
                    {
                        "token_before": token_before,
                        "token_after": int(tick_result.get("token_count", 0) or 0),
                        "background_tokens_processed": int(runtime.get("background_tokens_processed", 0) or 0),
                        "last_tick_token_delta": int(runtime.get("last_tick_token_delta", 0) or 0),
                    },
                )
            )
        except Exception as exc:  # pragma: no cover - defensive harness guard
            checks.append(
                _acceptance_check(
                    "acceptance_harness_runtime",
                    False,
                    f"Acceptance harness raised {type(exc).__name__}: {exc}",
                    {},
                )
            )
        finally:
            manager.close()

    verdict, passed, failed, skipped = _summarize_acceptance_checks(checks)
    return {
        "verdict": verdict,
        "passed": int(passed),
        "failed": int(failed),
        "skipped": int(skipped),
        "checks": checks,
    }


def classify_test_report(
    report: TestReport,
    *,
    thresholds: Mapping[str, Any] | None = None,
) -> TestReport:
    """Classify a long-test report as alive, degraded, or dead."""

    merged_thresholds = dict(DEFAULT_HEALTH_THRESHOLDS)
    if thresholds is not None:
        merged_thresholds.update(dict(thresholds))
    report.health_thresholds = dict(merged_thresholds)

    fatal_reasons: list[str] = []
    warning_reasons: list[str] = []
    runtime_progress = max(
        int(report.max_background_tokens_processed or 0),
        max(0, int(report.final_token_count or 0) - int(report.initial_token_count or 0)),
        int(report.final_tick_count or 0),
    )

    if not bool(report.cortex_available):
        fatal_reasons.append("Cortex was unavailable for the long run.")
    if int(report.samples_collected or 0) < int(merged_thresholds["min_samples"]):
        fatal_reasons.append("No long-test metric samples were collected.")
    if runtime_progress < int(merged_thresholds["min_runtime_progress_tokens"]):
        fatal_reasons.append("No observable runtime progress was recorded.")
    if int(report.total_thoughts or 0) < int(merged_thresholds["min_total_thoughts_alive"]):
        if runtime_progress >= int(merged_thresholds["min_runtime_progress_tokens"]):
            warning_reasons.append("Runtime progressed, but the cortex produced no thoughts.")
        else:
            fatal_reasons.append("The cortex produced no thoughts during the run.")
    if int(report.total_errors or 0) > 0:
        warning_reasons.append(f"{int(report.total_errors)} snapshot or reporting errors were recorded.")

    runtime_truth = report.final_runtime_truth if isinstance(report.final_runtime_truth, Mapping) else {}
    runtime_truth_verdict = str(runtime_truth.get("verdict", "")).lower()
    runtime_truth_action = str(runtime_truth.get("recommended_action", "")).strip()
    runtime_truth_action_suffix = f" with action {runtime_truth_action}" if runtime_truth_action else ""
    if runtime_truth_verdict == "failed":
        fatal_reasons.append(f"Runtime truth contract reported failed{runtime_truth_action_suffix}.")
    elif runtime_truth_verdict in {"partial", "degraded"}:
        warning_reasons.append(f"Runtime truth contract reported {runtime_truth_verdict}{runtime_truth_action_suffix}.")

    memory_pressure_report = report.memory_pressure_report if isinstance(report.memory_pressure_report, Mapping) else {}
    if bool(memory_pressure_report.get("unrecovered_high_pressure", False)):
        warning_reasons.append("Memory pressure reached the high band and did not recover before the run ended.")

    acceptance_verdict = str(report.acceptance_verdict or "not_run")
    if acceptance_verdict == "failed":
        fatal_reasons.append("Acceptance harness failed.")
    elif acceptance_verdict == "partial":
        warning_reasons.append("Acceptance harness only partially passed.")
    elif acceptance_verdict == "not_run":
        warning_reasons.append("Acceptance harness was not executed.")

    if fatal_reasons:
        report.health_verdict = "dead"
        report.health_reasons = fatal_reasons + warning_reasons
    elif warning_reasons:
        report.health_verdict = "degraded"
        report.health_reasons = warning_reasons
    else:
        report.health_verdict = "alive"
        report.health_reasons = ["Run met the minimum activity and acceptance thresholds."]
    return report


def _collect_snapshot(
    manager: Any,
    *,
    start_perf: float,
    last_thoughts_count: int,
    all_topics: set[str],
    thoughts_seen: list[str],
    seen_thought_texts: set[str],
    fresh_wait_seconds: float,
) -> tuple[MetricSnapshot, int]:
    status = manager.status(fresh_wait_seconds=fresh_wait_seconds)
    snapshot = MetricSnapshot(timestamp=time.time())
    snapshot.elapsed_s = max(0.0, float(snapshot.timestamp - start_perf))
    terminus_runtime = status.get("terminus_runtime") if isinstance(status.get("terminus_runtime"), Mapping) else {}
    cortex_snapshot = terminus_runtime.get("cortex") if isinstance(terminus_runtime.get("cortex"), Mapping) else {}
    memory_store = status.get("memory_store") if isinstance(status.get("memory_store"), Mapping) else {}
    action_loop = terminus_runtime.get("action_loop") if isinstance(terminus_runtime.get("action_loop"), Mapping) else {}
    ingestion = terminus_runtime.get("ingestion") if isinstance(terminus_runtime.get("ingestion"), Mapping) else {}
    runtime_truth = status.get("runtime_truth") if isinstance(status.get("runtime_truth"), Mapping) else {}
    thoughts_data = manager.cortex_thoughts(limit=10)

    snapshot.token_count = int(status.get("token_count", 0) or 0)
    snapshot.thoughts_total = int(thoughts_data.get("thoughts_generated", 0) or 0)
    snapshot.thoughts_delta = int(snapshot.thoughts_total - last_thoughts_count)
    last_thoughts_count = snapshot.thoughts_total
    snapshot.background_tokens_processed = int(terminus_runtime.get("background_tokens_processed", 0) or 0)
    snapshot.tick_count = int(terminus_runtime.get("tick_count", 0) or 0)
    snapshot.runtime_running = bool(terminus_runtime.get("running", False))
    snapshot.memory_fill = float(memory_store.get("fill_fraction", 0.0) or 0.0)
    snapshot.memory_size = int(memory_store.get("size", 0) or 0)
    snapshot.consolidation_mean = float(memory_store.get("mean_consolidation_level", 0.0) or 0.0)
    snapshot.ripple_tagged = int(memory_store.get("ripple_tagged", 0) or 0)
    snapshot.cortex_model = str(cortex_snapshot.get("model", ""))
    snapshot.cortex_latency_ms = float(cortex_snapshot.get("avg_inference_ms", 0.0) or 0.0)
    signal_state = cortex_snapshot.get("cognitive_signals") if isinstance(cortex_snapshot.get("cognitive_signals"), Mapping) else {}
    snapshot.prediction_error_mean = float(signal_state.get("prediction_error_mean", 0.0) or 0.0)
    snapshot.prediction_error_max = float(signal_state.get("prediction_error_max", 0.0) or 0.0)
    drives = cortex_snapshot.get("drives") if isinstance(cortex_snapshot.get("drives"), Mapping) else {}
    snapshot.da_level = float(drives.get("dopamine", 0.0) or 0.0)
    snapshot.serotonin_level = float(drives.get("serotonin", 0.0) or 0.0)
    snapshot.ach_level = float(drives.get("acetylcholine", 0.0) or 0.0)
    snapshot.ne_level = float(drives.get("norepinephrine", 0.0) or 0.0)
    quality = cortex_snapshot.get("quality") if isinstance(cortex_snapshot.get("quality"), Mapping) else {}
    snapshot.dream_verification_rate = float(quality.get("dream_verification_rate", 0.0) or 0.0)
    depth_policy = cortex_snapshot.get("depth_policy") if isinstance(cortex_snapshot.get("depth_policy"), Mapping) else {}
    snapshot.depth_counts = dict(depth_policy.get("counts", {}))
    narrative = cortex_snapshot.get("narrative_self") if isinstance(cortex_snapshot.get("narrative_self"), Mapping) else {}
    snapshot.narrative_summary = str(narrative.get("summary", ""))
    active_exploration = cortex_snapshot.get("active_exploration") if isinstance(cortex_snapshot.get("active_exploration"), Mapping) else {}
    snapshot.exploration_target = str(active_exploration.get("target", ""))
    snapshot.exploration_reason = str(active_exploration.get("reason", ""))
    episodic_memory = cortex_snapshot.get("episodic_memory") if isinstance(cortex_snapshot.get("episodic_memory"), Mapping) else {}
    snapshot.embedder = dict(episodic_memory.get("embedder", {}))
    snapshot.runtime_truth = dict(runtime_truth)
    snapshot.thought_lifecycle = _thought_lifecycle_snapshot(cortex_snapshot, thoughts_data)
    snapshot.memory_pressure = _memory_pressure_snapshot(memory_store, runtime_truth)
    snapshot.global_workspace = _global_workspace_snapshot(cortex_snapshot)
    snapshot.ingestion_state = str(ingestion.get("startup_state", ""))
    snapshot.action_count = int(action_loop.get("actions_recorded", 0) or 0)

    for thought in thoughts_data.get("thoughts", []):
        if not isinstance(thought, Mapping):
            continue
        topics = thought.get("topics", [])
        if isinstance(topics, (list, tuple)):
            all_topics.update(str(item) for item in topics if str(item).strip())
        thought_text = str(thought.get("thought", "")).strip()
        if thought_text and thought_text not in seen_thought_texts and len(thoughts_seen) < 50:
            thoughts_seen.append(thought_text)
            seen_thought_texts.add(thought_text)

    snapshot.topic_diversity = int(len(all_topics))
    return snapshot, last_thoughts_count


def _snapshot_has_inflight_thought(snapshot: MetricSnapshot) -> bool:
    lifecycle = snapshot.thought_lifecycle if isinstance(snapshot.thought_lifecycle, Mapping) else {}
    attempts = int(lifecycle.get("attempts", 0) or 0)
    successful = int(lifecycle.get("successful", snapshot.thoughts_total) or 0)
    mode = str(lifecycle.get("mode", "")).lower()
    return attempts > successful and mode == "thinking"


def _await_final_thought_settle(
    manager: Any,
    *,
    start_perf: float,
    last_thoughts_count: int,
    all_topics: set[str],
    thoughts_seen: list[str],
    seen_thought_texts: set[str],
    wait_seconds: float,
) -> tuple[MetricSnapshot | None, int]:
    deadline = time.time() + max(0.0, float(wait_seconds))
    last_snapshot: MetricSnapshot | None = None
    while time.time() < deadline:
        snapshot, last_thoughts_count = _collect_snapshot(
            manager,
            start_perf=start_perf,
            last_thoughts_count=last_thoughts_count,
            all_topics=all_topics,
            thoughts_seen=thoughts_seen,
            seen_thought_texts=seen_thought_texts,
            fresh_wait_seconds=min(10.0, max(1.0, float(wait_seconds))),
        )
        last_snapshot = snapshot
        if not _snapshot_has_inflight_thought(snapshot):
            return snapshot, last_thoughts_count
        remaining = max(0.0, deadline - time.time())
        time.sleep(min(1.0, max(0.1, remaining / 2.0)))
    return last_snapshot, last_thoughts_count


def run_long_test(
    duration_minutes: float = 20.0,
    sample_interval_s: float = 30.0,
    preset: str = "curriculum",
    output_dir: str = "reports",
    final_thought_wait_s: float = DEFAULT_FINAL_THOUGHT_WAIT_S,
    memory_capacity: int = DEFAULT_LONG_TEST_MEMORY_CAPACITY,
) -> TestReport:
    """Run a full Terminus brain test for the specified duration."""

    from hecsn.config.runtime_env import load_runtime_env
    from hecsn.service.manager import HECSNServiceManager

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    load_runtime_env(anchor_paths=(Path.cwd(),))

    report = TestReport(
        start_time=datetime.now(timezone.utc).isoformat(),
        duration_minutes=float(duration_minutes),
        sample_interval_s=float(sample_interval_s),
        preset=str(preset),
        memory_capacity=max(1, int(memory_capacity)),
    )

    acceptance = run_acceptance_harness(output_dir=output_dir, env_root=Path.cwd())
    report.acceptance_verdict = str(acceptance.get("verdict", "not_run"))
    report.acceptance_checks = [dict(item) for item in list(acceptance.get("checks") or [])]
    report.acceptance_passed = int(acceptance.get("passed", 0) or 0)
    report.acceptance_failed = int(acceptance.get("failed", 0) or 0)
    report.acceptance_skipped = int(acceptance.get("skipped", 0) or 0)
    report.acceptance_failure_details = _acceptance_failure_details(report.acceptance_checks)

    tmpdir = output_root / ".long_test_tmp" / datetime.now().strftime("%Y%m%d_%H%M%S")
    tmpdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _build_checkpoint(
        tmpdir,
        test_name="long_run",
        n_columns=64,
        column_latent_dim=32,
        memory_capacity=report.memory_capacity,
    )

    manager: Any | None = None
    snapshots: list[MetricSnapshot] = []
    latencies: list[float] = []
    all_topics: set[str] = set()
    thoughts_seen: list[str] = []
    seen_thought_texts: set[str] = set()
    long_run_error: str | None = None

    try:
        manager = HECSNServiceManager(
            checkpoint_path,
            trace_dir=tmpdir / "traces",
            env_root=Path.cwd(),
        )
        initial_status = manager.status()
        report.initial_token_count = int(initial_status.get("token_count", 0) or 0)

        try:
            config_result = manager.quick_start_terminus(preset=preset)
            logger.info("Quick start result: %s", config_result.get("status"))
        except Exception as exc:
            long_run_error = f"Quick start failed: {exc}"
            logger.error(long_run_error)
        else:
            terminus_runtime = config_result.get("terminus_runtime") if isinstance(config_result.get("terminus_runtime"), Mapping) else {}
            report.terminus_configured = bool(terminus_runtime.get("configured", False))
            report.terminus_running = bool(terminus_runtime.get("running", False))

        time.sleep(2.0)
        cortex_snapshot = manager.cortex_snapshot()
        report.cortex_available = bool(cortex_snapshot.get("enabled", False))
        report.cortex_model = str(cortex_snapshot.get("model", ""))

        can_sample = long_run_error is None and report.cortex_available
        if not can_sample:
            if long_run_error is not None:
                logger.error(long_run_error)
            if not report.cortex_available:
                logger.error("Cortex is unavailable — long run will be classified as dead.")
        else:
            duration_s = max(0.0, float(duration_minutes) * 60.0)
            logger.info(
                "Starting long test: %.1f minutes, sampling every %.1fs",
                duration_minutes,
                sample_interval_s,
            )
            start_perf = time.time()
            last_thoughts_count = 0
            interval_s = max(0.1, float(sample_interval_s))
            next_sample_at = min(interval_s, duration_s)
            while time.time() - start_perf < duration_s - 1e-6:
                sleep_time = max(0.0, start_perf + next_sample_at - time.time())
                if sleep_time > 0:
                    time.sleep(sleep_time)
                try:
                    snapshot, last_thoughts_count = _collect_snapshot(
                        manager,
                        start_perf=start_perf,
                        last_thoughts_count=last_thoughts_count,
                        all_topics=all_topics,
                        thoughts_seen=thoughts_seen,
                        seen_thought_texts=seen_thought_texts,
                        fresh_wait_seconds=max(5.0, float(sample_interval_s)),
                    )
                    for thought in manager.cortex_thoughts(limit=10).get("thoughts", []):
                        if not isinstance(thought, Mapping):
                            continue
                        latency_ms = float(thought.get("latency_ms", 0.0) or 0.0)
                        if latency_ms > 0:
                            latencies.append(latency_ms)
                except Exception as exc:
                    snapshot = MetricSnapshot(timestamp=time.time(), errors=1)
                    snapshot.elapsed_s = max(0.0, float(snapshot.timestamp - start_perf))
                    logger.warning("Snapshot error: %s", exc)
                snapshots.append(snapshot)
                logger.info(
                    "[%3.0fs/%3.0fs] thoughts=%d (+%d) tokens=%d bg=%d latency=%.0fms mem=%.0f%% topics=%d",
                    snapshot.elapsed_s,
                    duration_s,
                    snapshot.thoughts_total,
                    snapshot.thoughts_delta,
                    snapshot.token_count,
                    snapshot.background_tokens_processed,
                    snapshot.cortex_latency_ms,
                    snapshot.memory_fill * 100.0,
                    snapshot.topic_diversity,
                )
                if next_sample_at >= duration_s:
                    break
                elapsed_after_sample = max(0.0, time.time() - start_perf)
                if elapsed_after_sample >= duration_s:
                    break
                # If a snapshot overruns its scheduled slot, skip missed slots instead
                # of trying to catch up with back-to-back expensive snapshots.
                next_sample_at = min(duration_s, elapsed_after_sample + interval_s)
            if snapshots and _snapshot_has_inflight_thought(snapshots[-1]) and final_thought_wait_s > 0:
                logger.info(
                    "Final sample caught an in-flight cortex thought; waiting up to %.1fs for post-processing",
                    float(final_thought_wait_s),
                )
                try:
                    final_snapshot, last_thoughts_count = _await_final_thought_settle(
                        manager,
                        start_perf=start_perf,
                        last_thoughts_count=last_thoughts_count,
                        all_topics=all_topics,
                        thoughts_seen=thoughts_seen,
                        seen_thought_texts=seen_thought_texts,
                        wait_seconds=float(final_thought_wait_s),
                    )
                    if final_snapshot is not None:
                        snapshots.append(final_snapshot)
                except Exception as exc:
                    snapshot = MetricSnapshot(timestamp=time.time(), errors=1)
                    snapshot.elapsed_s = max(0.0, float(snapshot.timestamp - start_perf))
                    snapshots.append(snapshot)
                    logger.warning("Final thought-settle snapshot error: %s", exc)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    finally:
        if manager is not None:
            manager.close()

    report.end_time = datetime.now(timezone.utc).isoformat()
    report.samples_collected = int(len(snapshots))
    report.total_errors = int(sum(item.errors for item in snapshots))
    report.total_thoughts = int(snapshots[-1].thoughts_total) if snapshots else 0
    report.final_token_count = int(snapshots[-1].token_count) if snapshots else report.initial_token_count
    report.max_background_tokens_processed = int(max((item.background_tokens_processed for item in snapshots), default=0))
    report.final_background_tokens_processed = int(snapshots[-1].background_tokens_processed) if snapshots else 0
    report.final_tick_count = int(snapshots[-1].tick_count) if snapshots else 0
    report.terminus_running = bool(snapshots[-1].runtime_running) if snapshots else report.terminus_running
    report.final_memory_fill = float(snapshots[-1].memory_fill) if snapshots else 0.0
    report.final_consolidation = float(snapshots[-1].consolidation_mean) if snapshots else 0.0
    report.final_ripple_tagged = int(snapshots[-1].ripple_tagged) if snapshots else 0
    report.unique_topics = int(len(all_topics))
    report.all_topics = sorted(all_topics)
    report.topic_diversity_ratio = float(len(all_topics)) / float(max(1, report.total_thoughts))
    if not report.cortex_model and snapshots:
        report.cortex_model = str(snapshots[0].cortex_model)
    report.final_prediction_error_mean = float(snapshots[-1].prediction_error_mean) if snapshots else 0.0
    report.final_prediction_error_max = float(snapshots[-1].prediction_error_max) if snapshots else 0.0
    report.final_dream_verification_rate = float(snapshots[-1].dream_verification_rate) if snapshots else 0.0
    report.depth_counts = dict(snapshots[-1].depth_counts) if snapshots else {}
    report.final_narrative_summary = str(snapshots[-1].narrative_summary) if snapshots else ""
    report.final_exploration_target = str(snapshots[-1].exploration_target) if snapshots else ""
    report.final_exploration_reason = str(snapshots[-1].exploration_reason) if snapshots else ""
    report.final_embedder = dict(snapshots[-1].embedder) if snapshots else {}
    report.final_runtime_truth = dict(snapshots[-1].runtime_truth) if snapshots else {}
    report.action_count = int(snapshots[-1].action_count) if snapshots else 0
    report.thought_lifecycle_summary = _summarize_thought_lifecycle(snapshots)
    report.memory_pressure_report = _summarize_memory_pressure(snapshots)
    report.global_workspace_report = _summarize_global_workspace(snapshots)
    report.sample_thoughts = thoughts_seen[:20]

    if latencies:
        latencies.sort()
        report.avg_latency_ms = float(sum(latencies) / len(latencies))
        p95_index = min(len(latencies) - 1, max(0, int(0.95 * len(latencies)) - 1))
        report.p95_latency_ms = float(latencies[p95_index])
        report.max_latency_ms = float(latencies[-1])

    report.snapshots = [
        {
            "elapsed_s": item.elapsed_s,
            "token_count": item.token_count,
            "thoughts": item.thoughts_total,
            "thoughts_delta": item.thoughts_delta,
            "background_tokens_processed": item.background_tokens_processed,
            "tick_count": item.tick_count,
            "runtime_running": item.runtime_running,
            "latency_ms": item.cortex_latency_ms,
            "memory_fill": item.memory_fill,
            "memory_size": item.memory_size,
            "consolidation": item.consolidation_mean,
            "ripple_tagged": item.ripple_tagged,
            "topic_diversity": item.topic_diversity,
            "prediction_error_mean": item.prediction_error_mean,
            "prediction_error_max": item.prediction_error_max,
            "dream_verification_rate": item.dream_verification_rate,
            "depth_counts": item.depth_counts,
            "exploration_target": item.exploration_target,
            "exploration_reason": item.exploration_reason,
            "embedder": item.embedder,
            "runtime_truth": item.runtime_truth,
            "thought_lifecycle": item.thought_lifecycle,
            "memory_pressure": item.memory_pressure,
            "global_workspace": item.global_workspace,
            "ingestion_state": item.ingestion_state,
            "action_count": item.action_count,
            "da": item.da_level,
            "errors": item.errors,
        }
        for item in snapshots
    ]

    if long_run_error:
        report.health_reasons.append(long_run_error)
    classify_test_report(report)
    if long_run_error and long_run_error not in report.health_reasons:
        report.health_reasons.insert(0, long_run_error)
    return report


def write_report(report: TestReport, output_dir: str = "reports") -> tuple[str, str]:
    """Write report as JSON and markdown. Returns (json_path, md_path)."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out / f"long_test_{timestamp}.json"
    md_path = out / f"long_test_{timestamp}.md"
    readme_path = out / "README.md"

    json_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Long Test Report",
        "",
        f"**Start:** {report.start_time}",
        f"**End:** {report.end_time}",
        f"**Duration:** {report.duration_minutes:.1f} minutes",
        f"**Sampling interval:** {report.sample_interval_s:.1f} s",
            f"**Preset:** {report.preset}",
            f"**Memory capacity:** {report.memory_capacity}",
            f"**Cortex:** {report.cortex_model or '-'}",
        "",
        "## Health Verdict",
        "",
        f"**Verdict:** {report.health_verdict.upper()}",
        "",
        "### Reasons",
        "",
    ]
    if report.health_reasons:
        lines.extend(f"- {reason}" for reason in report.health_reasons)
    else:
        lines.append("- No explicit health reasons were recorded.")

    lines.extend(
        [
            "",
            "## Acceptance Harness",
            "",
            f"**Verdict:** {report.acceptance_verdict}",
            f"**Passed / Failed / Skipped:** {report.acceptance_passed} / {report.acceptance_failed} / {report.acceptance_skipped}",
            "",
            "| Check | Passed | Summary |",
            "|-------|--------|---------|",
        ]
    )
    for check in report.acceptance_checks:
        lines.append(
            f"| {check.get('name', '-')} | {'yes' if check.get('passed', False) else 'no'} | {str(check.get('summary', '')).replace('|', '/')} |"
        )

    if report.acceptance_failure_details:
        lines.extend(
            [
                "",
                "### Acceptance Failures",
                "",
            ]
        )
        for failure in report.acceptance_failure_details:
            lines.append(
                f"- {failure.get('name', '-')}: {str(failure.get('summary', '')).replace('|', '/')}"
            )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Cortex available | {report.cortex_available} |",
            f"| Terminus configured | {report.terminus_configured} |",
            f"| Final runtime running | {report.terminus_running} |",
            f"| Samples collected | {report.samples_collected} |",
            f"| Initial token count | {report.initial_token_count} |",
            f"| Final token count | {report.final_token_count} |",
            f"| Max background tokens processed | {report.max_background_tokens_processed} |",
            f"| Final tick count | {report.final_tick_count} |",
            f"| Total thoughts | {report.total_thoughts} |",
            f"| Unique topics | {report.unique_topics} |",
            f"| Topic diversity | {report.topic_diversity_ratio:.2f} topics/thought |",
            f"| Avg latency | {report.avg_latency_ms:.0f} ms |",
            f"| P95 latency | {report.p95_latency_ms:.0f} ms |",
            f"| Max latency | {report.max_latency_ms:.0f} ms |",
            f"| Final memory fill | {report.final_memory_fill:.1%} |",
            f"| Memory capacity | {report.memory_capacity} |",
            f"| Final consolidation | {report.final_consolidation:.3f} |",
            f"| Ripple-tagged memories | {report.final_ripple_tagged} |",
            f"| Prediction error (mean/max) | {report.final_prediction_error_mean:.3f} / {report.final_prediction_error_max:.3f} |",
            f"| Dream verification rate | {report.final_dream_verification_rate:.1%} |",
            f"| Depth counts | quick={report.depth_counts.get('quick', 0)}, standard={report.depth_counts.get('standard', 0)}, deep={report.depth_counts.get('deep', 0)} |",
            f"| Active exploration target | {report.final_exploration_target or '-'} |",
            f"| Active exploration reason | {report.final_exploration_reason or '-'} |",
            f"| Embedder | {report.final_embedder.get('kind', '-')}: {report.final_embedder.get('model', '-')} |",
            f"| Embedder degraded | {report.final_embedder.get('degraded', False)} |",
            f"| Embedder fallback calls | {report.final_embedder.get('fallback_calls', 0)} |",
            f"| Embedder rate-limit hits | {report.final_embedder.get('rate_limit_hits', 0)} |",
            f"| Runtime truth verdict | {report.final_runtime_truth.get('verdict', '-') if report.final_runtime_truth else '-'} |",
            f"| Runtime truth action | {report.final_runtime_truth.get('recommended_action', '-') if report.final_runtime_truth else '-'} |",
            f"| Recorded actions | {report.action_count} |",
            f"| Errors | {report.total_errors} |",
        ]
    )

    thought_summary = report.thought_lifecycle_summary if isinstance(report.thought_lifecycle_summary, Mapping) else {}
    memory_summary = report.memory_pressure_report if isinstance(report.memory_pressure_report, Mapping) else {}
    workspace_summary = report.global_workspace_report if isinstance(report.global_workspace_report, Mapping) else {}
    lines.extend(
        [
            "",
            "## Liveness Diagnosis",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Thought attempts | {thought_summary.get('attempts', 0)} |",
            f"| Successful thoughts | {thought_summary.get('successful', report.total_thoughts)} |",
            f"| Dreams | {thought_summary.get('dreams', 0)} |",
            f"| Blocked ticks | {thought_summary.get('blocked_ticks', 0)} |",
            f"| Wake triggers | {thought_summary.get('wake_triggers', {})} |",
            f"| Rejected / blocked reasons | {', '.join(str(item) for item in list(thought_summary.get('rejected_or_blocked_reasons') or [])) or '-'} |",
            "",
            "## Memory Pressure",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| First fill | {float(memory_summary.get('first_fill', 0.0) or 0.0):.1%} |",
            f"| Final fill | {float(memory_summary.get('final_fill', report.final_memory_fill) or 0.0):.1%} |",
            f"| Max fill | {float(memory_summary.get('max_fill', report.final_memory_fill) or 0.0):.1%} |",
            f"| High-pressure samples | {memory_summary.get('high_samples', 0)} |",
            f"| Recovered from high | {memory_summary.get('recovered_from_high', False)} |",
            f"| Unrecovered high pressure | {memory_summary.get('unrecovered_high_pressure', False)} |",
            f"| Recommended action | {memory_summary.get('recommended_action', '-')} |",
            f"| Working-set policy | {memory_summary.get('final_policy', {})} |",
            "",
            "## Global Workspace",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Final size / capacity | {workspace_summary.get('final_size', 0)} / {workspace_summary.get('capacity', 0)} |",
            f"| Max size | {workspace_summary.get('max_size', 0)} |",
            f"| Active exploration | {workspace_summary.get('active_exploration', {})} |",
            f"| Evidence boundary | {workspace_summary.get('evidence_boundary', {})} |",
            f"| Final broadcast | {str(workspace_summary.get('final_broadcast', '')).replace('|', '/')} |",
        ]
    )

    lines.extend(
        [
            "",
            "## Timeline",
            "",
            "| Time (s) | Tokens | Thoughts | ΔThoughts | Bg tokens | Tick count | Latency (ms) | Memory | Topics | Ingestion |",
            "|----------|--------|----------|-----------|-----------|------------|--------------|--------|--------|-----------|",
        ]
    )
    for snapshot in report.snapshots:
        lines.append(
            f"| {snapshot['elapsed_s']:.0f} | {snapshot['token_count']} | {snapshot['thoughts']} | {snapshot['thoughts_delta']} | "
            f"{snapshot['background_tokens_processed']} | {snapshot['tick_count']} | {snapshot['latency_ms']:.0f} | "
            f"{snapshot['memory_fill']:.1%} | {snapshot['topic_diversity']} | {snapshot['ingestion_state'] or '-'} |"
        )

    if report.final_narrative_summary:
        lines.extend([
            "",
            "## Narrative Self",
            "",
            report.final_narrative_summary,
        ])

    if report.sample_thoughts:
        lines.extend([
            "",
            "## Sample Thoughts",
            "",
        ])
        for index, thought in enumerate(report.sample_thoughts[:10], 1):
            lines.append(f"{index}. {thought}")

    markdown = "\n".join(lines)
    md_path.write_text(markdown, encoding="utf-8")
    readme_path.write_text(markdown, encoding="utf-8")
    return str(json_path), str(md_path)


def health_exit_code(report: TestReport) -> int:
    """Map health verdicts to CLI exit codes."""

    verdict = str(report.health_verdict or "dead").lower()
    if verdict == "alive":
        return 0
    if verdict == "degraded":
        return 1
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a long Terminus brain test")
    parser.add_argument("--duration", type=float, default=20.0, help="Test duration in minutes (default: 20)")
    parser.add_argument("--interval", type=float, default=30.0, help="Sampling interval in seconds (default: 30)")
    parser.add_argument("--preset", type=str, default="curriculum", help="Terminus preset to use (default: curriculum)")
    parser.add_argument("--output", type=str, default="reports", help="Output directory for report (default: reports)")
    parser.add_argument(
        "--memory-capacity",
        type=int,
        default=DEFAULT_LONG_TEST_MEMORY_CAPACITY,
        help=f"Checkpoint memory capacity for the validation run (default: {DEFAULT_LONG_TEST_MEMORY_CAPACITY})",
    )
    parser.add_argument(
        "--final-thought-wait",
        type=float,
        default=DEFAULT_FINAL_THOUGHT_WAIT_S,
        help="Seconds to wait when the final sample catches an in-flight thought (default: 90)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    print(f"Starting long test: {args.duration} min, preset={args.preset}")
    print(f"Sampling every {args.interval}s, output to {args.output}/")
    print("Press Ctrl+C to stop early (report still generated)")
    print()

    report = run_long_test(
        duration_minutes=args.duration,
        sample_interval_s=args.interval,
        preset=args.preset,
        output_dir=args.output,
        final_thought_wait_s=args.final_thought_wait,
        memory_capacity=args.memory_capacity,
    )
    json_path, md_path = write_report(report, output_dir=args.output)

    print("\nTest complete!")
    print(f"  Health: {report.health_verdict}")
    print(f"  Acceptance: {report.acceptance_verdict}")
    print(f"  Thoughts: {report.total_thoughts}")
    print(f"  Topics: {report.unique_topics}")
    print(f"  Avg latency: {report.avg_latency_ms:.0f}ms")
    print(f"  Report: {md_path}")
    print(f"  Data: {json_path}")
    return health_exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
