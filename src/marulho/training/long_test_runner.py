"""Long-Test Runner -- exercises the MarulhoBrain runtime and classifies run health.

Usage:
    python -m marulho.training.long_test_runner --duration 20 --output reports/long_test_report.md

This runs MarulhoBrain with the curriculum preset for N minutes (default 20),
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
import hashlib
import json
import logging
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from marulho.brain import MarulhoBrain
from marulho.brain.runtime import DEFAULT_BRAIN_QUANTUM_TOKENS, DEFAULT_BRAIN_TICK_TOKENS

logger = logging.getLogger(__name__)

DEFAULT_HEALTH_THRESHOLDS: dict[str, int] = {
    "min_samples": 1,
    "min_runtime_progress_tokens": 1,
}
DEFAULT_LONG_TEST_MEMORY_CAPACITY = 16_384


@dataclass
class MetricSnapshot:
    """One measurement point during the test run."""

    timestamp: float = 0.0
    elapsed_s: float = 0.0
    token_count: int = 0
    readouts_total: int = 0
    readouts_delta: int = 0
    background_tokens_processed: int = 0
    tick_count: int = 0
    runtime_running: bool = False
    memory_fill: float = 0.0
    memory_size: int = 0
    consolidation_mean: float = 0.0
    ripple_tagged: int = 0
    runtime_latency_ms: float = 0.0
    topic_diversity: int = 0
    da_level: float = 0.0
    serotonin_level: float = 0.0
    ach_level: float = 0.0
    ne_level: float = 0.0
    prediction_error_mean: float = 0.0
    prediction_error_max: float = 0.0
    depth_counts: dict[str, int] = field(default_factory=dict)
    language_surface_summary: str = ""
    exploration_target: str = ""
    exploration_reason: str = ""
    embedder: dict[str, Any] = field(default_factory=dict)
    runtime_truth: dict[str, Any] = field(default_factory=dict)
    readout_lifecycle: dict[str, Any] = field(default_factory=dict)
    memory_pressure: dict[str, Any] = field(default_factory=dict)
    subcortex_workspace: dict[str, Any] = field(default_factory=dict)
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
    terminus_configured: bool = False
    terminus_running: bool = False
    initial_token_count: int = 0
    final_token_count: int = 0
    total_readouts: int = 0
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
    depth_counts: dict[str, int] = field(default_factory=dict)
    all_topics: list[str] = field(default_factory=list)
    final_language_surface_summary: str = ""
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
    readout_lifecycle_summary: dict[str, Any] = field(default_factory=dict)
    memory_pressure_report: dict[str, Any] = field(default_factory=dict)
    subcortex_workspace_report: dict[str, Any] = field(default_factory=dict)
    source_configuration: dict[str, Any] = field(default_factory=dict)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    sample_readouts: list[str] = field(default_factory=list)


def _build_checkpoint(
    root: Path,
    *,
    test_name: str,
    n_columns: int,
    column_latent_dim: int,
    memory_capacity: int,
) -> Path:
    from marulho.config.model_config import MarulhoConfig
    from marulho.training.checkpointing import save_trainer_checkpoint
    from marulho.training.model import MarulhoModel
    from marulho.training.trainer import MarulhoTrainer

    cfg = MarulhoConfig(
        n_columns=n_columns,
        column_latent_dim=column_latent_dim,
        bootstrap_tokens=0,
        memory_capacity=memory_capacity,
        eta_competitive=0.05,
        enable_context_layer=True,
        enable_binding_layer=True,
    )
    trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
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
    actions = {
        "grounded_source_influence": "verify_workspace_action_assist_and_preserved_source_evidence",
        "brain_feed": "check_marulho_brain_feed_source_text",
        "brain_generation": "check_marulho_brain_local_readout_transitions",
        "runtime_progress": "check_marulho_brain_source_buffer_and_tick_path",
    }
    failures: list[dict[str, Any]] = []
    for item in checks:
        if bool(item.get("passed", False)):
            continue
        failures.append(
            {
                "name": str(item.get("name", "")),
                "summary": str(item.get("summary", "")),
                "details": dict(item.get("details") or {}) if isinstance(item.get("details"), Mapping) else {},
                "recommended_action": actions.get(str(item.get("name", "")), "inspect_acceptance_path"),
            }
        )
    return failures


def _source_configuration_evidence(
    terminus_runtime: Mapping[str, Any],
    *,
    preset: str,
    configuration_surface: str,
) -> dict[str, Any]:
    source_bank = [
        dict(item)
        for item in list(terminus_runtime.get("source_bank") or [])
        if isinstance(item, Mapping)
    ]
    sensory = terminus_runtime.get("sensory") if isinstance(terminus_runtime.get("sensory"), Mapping) else {}
    sensory_source_bank = [
        dict(item)
        for item in list(sensory.get("source_bank") or [])
        if isinstance(item, Mapping)
    ]
    ingestion = terminus_runtime.get("ingestion") if isinstance(terminus_runtime.get("ingestion"), Mapping) else {}
    payload = {
        "preset": str(preset),
        "configuration_surface": str(configuration_surface),
        "source_bank": source_bank,
        "sensory_source_bank": sensory_source_bank,
        "tick_tokens": int(terminus_runtime.get("tick_tokens", 0) or 0),
        "sleep_interval_seconds": float(terminus_runtime.get("sleep_interval_seconds", 0.0) or 0.0),
        "repeat_sources": bool(terminus_runtime.get("repeat_sources", True)),
        "ingestion": dict(ingestion),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return {
        "configured": bool(terminus_runtime.get("configured")),
        "preset": str(preset),
        "configuration_surface": str(configuration_surface),
        "source_count": int(len(source_bank)),
        "source_names": [str(item.get("name", "")) for item in source_bank],
        "source_types": [str(item.get("source_type", "auto")) for item in source_bank],
        "sensory_source_count": int(len(sensory_source_bank)),
        "sensory_source_names": [str(item.get("name", "")) for item in sensory_source_bank],
        "configuration_hash": hashlib.sha256(encoded).hexdigest(),
        "configuration_payload": payload,
        "reproduction_hint": "Use the configuration_payload with MarulhoBrain.feed/start/tick using the recorded preset.",
    }


def _brain_source_text(preset: str) -> str:
    preset_name = str(preset or "curriculum").strip().lower() or "curriculum"
    passages = {
        "curriculum": (
            "MARULHO learns from compact local source streams. "
            "Sparse columns route prediction error into replay. "
            "BrainTrace records tick, generation, replay, and checkpoint evidence. "
        ),
        "stress": (
            "Adaptive memory plasticity stabilizes sparse spike routing. "
            "Grounded local observations support prediction error and replay readiness. "
        ),
        "language": (
            "Local transition readout emits bounded text from MARULHO-owned sparse state. "
            "No external LLM, ThoughtLoop, or Cortex surface owns generation. "
        ),
    }
    return passages.get(preset_name, passages["curriculum"]) * 32


def _brain_source_configuration_evidence(
    *,
    preset: str,
    source_name: str,
    source_text: str,
    tick_tokens: int,
    quantum_tokens: int,
    interval_seconds: float,
) -> dict[str, Any]:
    payload = {
        "preset": str(preset),
        "configuration_surface": "MarulhoBrain.feed/start",
        "source_bank": [
            {
                "name": str(source_name),
                "source_type": "inline_text",
                "token_estimate": len(str(source_text)),
            }
        ],
        "tick_tokens": int(tick_tokens),
        "quantum_tokens": int(quantum_tokens),
        "interval_seconds": float(interval_seconds),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return {
        "configured": True,
        "preset": str(preset),
        "configuration_surface": "MarulhoBrain.feed/start",
        "source_count": 1,
        "source_names": [str(source_name)],
        "source_types": ["inline_text"],
        "sensory_source_count": 0,
        "configuration_hash": hashlib.sha256(encoded).hexdigest(),
        "configuration_payload": payload,
        "reproduction_hint": "Load checkpoint, call MarulhoBrain.feed(...), then MarulhoBrain.start/tick with these token settings.",
    }


def _runtime_status(runtime: Any, *, fresh_wait_seconds: float | None = None) -> dict[str, Any]:
    status = getattr(runtime, "status", None)
    if not callable(status):
        return {}
    try:
        if fresh_wait_seconds is None:
            result = status()
        else:
            result = status(fresh_wait_seconds=fresh_wait_seconds)
    except TypeError:
        result = status()
    return dict(result) if isinstance(result, Mapping) else {}


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


def _summarize_readout_lifecycle(snapshots: list[MetricSnapshot]) -> dict[str, Any]:
    if not snapshots:
        return {}
    final = snapshots[-1].readout_lifecycle
    blocked_reasons = [
        str(item.readout_lifecycle.get("rejected_or_blocked_reason", ""))
        for item in snapshots
        if str(item.readout_lifecycle.get("rejected_or_blocked_reason", "")).strip()
    ]
    return {
        "attempts": int(final.get("attempts", 0) or 0),
        "successful": int(final.get("successful", snapshots[-1].readouts_total) or 0),
        "blocked_ticks": int(final.get("blocked_ticks", 0) or 0),
        "last_blocked": dict(final.get("last_blocked") or {}),
        "wake_triggers": dict(final.get("wake_triggers") or {}),
        "rejected_or_blocked_reasons": sorted(set(blocked_reasons))[:8],
    }


def _summarize_subcortex_workspace(snapshots: list[MetricSnapshot]) -> dict[str, Any]:
    if not snapshots:
        return {}
    final = snapshots[-1].subcortex_workspace
    max_size = max(int(item.subcortex_workspace.get("size", 0) or 0) for item in snapshots)
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
) -> dict[str, Any]:
    """Run a small deterministic acceptance harness on maintained runtime paths."""

    from marulho.config.runtime_env import load_runtime_env

    env_anchor = Path(env_root) if env_root is not None else Path.cwd()
    load_runtime_env(anchor_paths=(env_anchor,))
    del idle_wait_s

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

        brain = MarulhoBrain.load(checkpoint_path)
        try:
            feed_text = (
                notes_path.read_text(encoding="utf-8")
                + "\n"
                + stream_path.read_text(encoding="utf-8")
            )
            feed = brain.feed(feed_text, source="acceptance_stream", learn=False)
            accepted_tokens = int(feed.get("accepted_tokens", 0) or 0)
            checks.append(
                _acceptance_check(
                    "brain_feed",
                    accepted_tokens > 0,
                    "MarulhoBrain accepted local source text into the brain-owned buffer." if accepted_tokens > 0 else "MarulhoBrain did not accept source text.",
                    {
                        "accepted_tokens": accepted_tokens,
                        "queued_tokens": int(feed.get("queued_tokens", 0) or 0),
                        "source": str(feed.get("source", "")),
                    },
                )
            )
            token_before = int(brain.trainer.token_count)
            tick_result: dict[str, Any] = {}
            for _ in range(max(1, int(tick_steps))):
                tick_result = brain.tick(
                    tokens=20,
                    quantum_tokens=DEFAULT_BRAIN_QUANTUM_TOKENS,
                    source="acceptance_stream",
                )
            generation = brain.generate(prompt="Cats", max_tokens=12)
            readout = brain.status().get("readout")
            readout_summary = dict(readout) if isinstance(readout, Mapping) else {}
            generation_passed = (
                bool(generation.get("owned_by_marulho"))
                and not bool(generation.get("external_llm_used"))
                and not bool(generation.get("thought_loop_used"))
                and not bool(generation.get("cortex_used"))
                and int(readout_summary.get("observed_transition_count", 0) or 0) > 0
            )
            checks.append(
                _acceptance_check(
                    "brain_generation",
                    generation_passed,
                    "Local readout used MARULHO-owned transition evidence." if generation_passed else "Local readout did not accumulate transition evidence.",
                    {
                        "generation_text": str(generation.get("text", "")),
                        "available": bool(generation.get("available")),
                        "observed_transition_count": int(
                            readout_summary.get("observed_transition_count", 0) or 0
                        ),
                        "transition_state_count": int(
                            readout_summary.get("transition_state_count", 0) or 0
                        ),
                    },
                )
            )
            runtime_progress_passed = (
                int(brain.trainer.token_count) > token_before
                and int(tick_result.get("trained_tokens", 0) or 0) > 0
            )
            checks.append(
                _acceptance_check(
                    "runtime_progress",
                    runtime_progress_passed,
                    "MarulhoBrain made observable training progress on the maintained tick path." if runtime_progress_passed else "MarulhoBrain did not make observable progress on the maintained tick path.",
                    {
                        "token_before": token_before,
                        "token_after": int(brain.trainer.token_count),
                        "background_tokens_processed": max(
                            0,
                            int(brain.trainer.token_count) - token_before,
                        ),
                        "last_tick_token_delta": int(tick_result.get("trained_tokens", 0) or 0),
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
            brain.stop(timeout_seconds=1.0)

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

    if int(report.samples_collected or 0) < int(merged_thresholds["min_samples"]):
        fatal_reasons.append("No long-test metric samples were collected.")
    if runtime_progress < int(merged_thresholds["min_runtime_progress_tokens"]):
        fatal_reasons.append("No observable runtime progress was recorded.")
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
    runtime: Any,
    *,
    start_perf: float,
    all_topics: set[str],
    fresh_wait_seconds: float,
) -> MetricSnapshot:
    status = _runtime_status(runtime, fresh_wait_seconds=fresh_wait_seconds)
    snapshot = MetricSnapshot(timestamp=time.time())
    snapshot.elapsed_s = max(0.0, float(snapshot.timestamp - start_perf))
    brain_status = str(status.get("surface", "")) == "marulho_brain_runtime.v1"
    loop = status.get("loop") if isinstance(status.get("loop"), Mapping) else {}
    terminus_runtime = status.get("terminus_runtime") if isinstance(status.get("terminus_runtime"), Mapping) else {}
    memory_store = status.get("memory_store") if isinstance(status.get("memory_store"), Mapping) else {}
    if brain_status and not memory_store:
        memory_store = _brain_memory_store_snapshot(runtime)
    action_loop = terminus_runtime.get("action_loop") if isinstance(terminus_runtime.get("action_loop"), Mapping) else {}
    ingestion = terminus_runtime.get("ingestion") if isinstance(terminus_runtime.get("ingestion"), Mapping) else {}
    runtime_truth = status.get("runtime_truth") if isinstance(status.get("runtime_truth"), Mapping) else {}
    if brain_status and not runtime_truth:
        runtime_truth = _brain_runtime_truth(status)
    readout = status.get("readout") if isinstance(status.get("readout"), Mapping) else {}

    snapshot.token_count = int(status.get("token_count", 0) or 0)
    snapshot.readouts_total = int(readout.get("observed_transition_count", 0) or 0)
    snapshot.readouts_delta = 0
    snapshot.background_tokens_processed = (
        int(loop.get("tick_count", 0) or 0) * int(loop.get("tick_tokens", 0) or 0)
        if brain_status
        else int(terminus_runtime.get("background_tokens_processed", 0) or 0)
    )
    snapshot.tick_count = (
        int(loop.get("tick_count", 0) or 0)
        if brain_status
        else int(terminus_runtime.get("tick_count", 0) or 0)
    )
    snapshot.runtime_running = (
        bool(loop.get("running", False))
        if brain_status
        else bool(terminus_runtime.get("running", False))
    )
    snapshot.memory_fill = float(memory_store.get("fill_fraction", 0.0) or 0.0)
    snapshot.memory_size = int(memory_store.get("size", 0) or 0)
    snapshot.consolidation_mean = float(memory_store.get("mean_consolidation_level", 0.0) or 0.0)
    snapshot.ripple_tagged = int(memory_store.get("ripple_tagged", 0) or 0)
    snapshot.runtime_latency_ms = 0.0
    snapshot.embedder = {"kind": "subcortex_local_encoder", "available": False}
    snapshot.runtime_truth = dict(runtime_truth)
    snapshot.readout_lifecycle = {
        "source": "marulho_brain_local_transition_readout",
        "attempts": int(readout.get("observed_transition_count", 0) or 0),
        "successful": snapshot.readouts_total,
        "blocked_ticks": 0,
    }
    snapshot.memory_pressure = _memory_pressure_snapshot(memory_store, runtime_truth)
    snapshot.subcortex_workspace = {"source": "marulho_brain_runtime_state", "size": 0, "capacity": 0}
    snapshot.ingestion_state = (
        "brain_source_buffer_ready"
        if brain_status and int(status.get("queued_tokens", 0) or 0) > 0
        else str(ingestion.get("startup_state", ""))
    )
    snapshot.action_count = int(action_loop.get("actions_recorded", 0) or 0)

    snapshot.topic_diversity = int(len(all_topics))
    return snapshot


def _brain_memory_store_snapshot(runtime: Any) -> dict[str, Any]:
    trainer = getattr(runtime, "trainer", None)
    model = getattr(trainer, "model", None)
    store = getattr(model, "memory_store", None)
    capacity = int(getattr(store, "capacity", 0) or 0)
    try:
        summary = store.live_summary_stats() if store is not None else {}
    except Exception:
        summary = {}
    if not isinstance(summary, Mapping):
        summary = {}
    size = int(
        summary.get("size", getattr(store, "size", 0) if store is not None else 0)
        or 0
    )
    fill_fraction = float(size / max(1, capacity)) if capacity else 0.0
    return {
        "fill_fraction": float(summary.get("fill_fraction", fill_fraction) or fill_fraction),
        "size": size,
        "capacity": capacity,
        "mean_consolidation_level": float(summary.get("mean_consolidation_level", 0.0) or 0.0),
        "mean_fragility": float(summary.get("mean_fragility", 0.0) or 0.0),
        "max_fragility": float(summary.get("max_fragility", 0.0) or 0.0),
        "ripple_tagged": int(summary.get("ripple_tagged", 0) or 0),
        "n_seen": int(summary.get("n_seen", size) or 0),
    }


def _brain_runtime_truth(status: Mapping[str, Any]) -> dict[str, Any]:
    loop = status.get("loop") if isinstance(status.get("loop"), Mapping) else {}
    queued_tokens = int(status.get("queued_tokens", 0) or 0)
    token_count = int(status.get("token_count", 0) or 0)
    running = bool(loop.get("running", False))
    verdict = "alive" if running or token_count > 0 else "partial" if queued_tokens > 0 else "degraded"
    return {
        "surface": "marulho_brain_runtime_truth.v1",
        "verdict": verdict,
        "recommended_action": "continue_monitoring" if verdict == "alive" else "feed_and_tick_brain",
        "owner": "MarulhoBrain",
        "source_configuration": {
            "queued_tokens": queued_tokens,
            "loop": dict(loop),
        },
        "retired_brain_surfaces": dict(status.get("retired_brain_surfaces") or {}),
    }


def run_long_test(
    duration_minutes: float = 20.0,
    sample_interval_s: float = 30.0,
    preset: str = "curriculum",
    output_dir: str = "reports",
    memory_capacity: int = DEFAULT_LONG_TEST_MEMORY_CAPACITY,
) -> TestReport:
    """Run a full MarulhoBrain test for the specified duration."""

    from marulho.config.runtime_env import load_runtime_env

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

    brain: MarulhoBrain | None = None
    snapshots: list[MetricSnapshot] = []
    latencies: list[float] = []
    all_topics: set[str] = set()
    long_run_error: str | None = None

    try:
        brain = MarulhoBrain.load(checkpoint_path)
        runtime = brain
        initial_status = runtime.status()
        report.initial_token_count = int(initial_status.get("token_count", 0) or 0)

        try:
            source_name = f"long_test_{preset}"
            source_text = _brain_source_text(preset)
            feed = brain.feed(source_text, source=source_name, learn=False)
            if int(feed.get("accepted_tokens", 0) or 0) <= 0:
                raise RuntimeError("MarulhoBrain accepted no long-test source tokens")
            start_result = brain.start(
                tick_tokens=DEFAULT_BRAIN_TICK_TOKENS,
                quantum_tokens=DEFAULT_BRAIN_QUANTUM_TOKENS,
                interval_seconds=0.01,
                source=source_name,
            )
            logger.info("Brain start result: %s", start_result.get("started"))
        except Exception as exc:
            long_run_error = f"MarulhoBrain start failed: {exc}"
            logger.error(long_run_error)
        else:
            report.terminus_configured = True
            report.terminus_running = bool(brain.status().get("loop", {}).get("running", False))
            report.source_configuration = _brain_source_configuration_evidence(
                preset=preset,
                source_name=source_name,
                source_text=source_text,
                tick_tokens=DEFAULT_BRAIN_TICK_TOKENS,
                quantum_tokens=DEFAULT_BRAIN_QUANTUM_TOKENS,
                interval_seconds=0.01,
            )

        time.sleep(min(0.25, max(0.0, float(sample_interval_s))))
        can_sample = long_run_error is None
        if not can_sample:
            if long_run_error is not None:
                logger.error(long_run_error)
        else:
            duration_s = max(0.0, float(duration_minutes) * 60.0)
            logger.info(
                "Starting long test: %.1f minutes, sampling every %.1fs",
                duration_minutes,
                sample_interval_s,
            )
            start_perf = time.time()
            interval_s = max(0.1, float(sample_interval_s))
            next_sample_at = min(interval_s, duration_s)
            while time.time() - start_perf < duration_s - 1e-6:
                sleep_time = max(0.0, start_perf + next_sample_at - time.time())
                if sleep_time > 0:
                    time.sleep(sleep_time)
                try:
                    snapshot = _collect_snapshot(
                        runtime,
                        start_perf=start_perf,
                        all_topics=all_topics,
                        fresh_wait_seconds=max(5.0, float(sample_interval_s)),
                    )
                except Exception as exc:
                    snapshot = MetricSnapshot(timestamp=time.time(), errors=1)
                    snapshot.elapsed_s = max(0.0, float(snapshot.timestamp - start_perf))
                    logger.warning("Snapshot error: %s", exc)
                snapshots.append(snapshot)
                logger.info(
                    "[%3.0fs/%3.0fs] readouts=%d (+%d) tokens=%d bg=%d runtime_latency=%.0fms mem=%.0f%% topics=%d",
                    snapshot.elapsed_s,
                    duration_s,
                    snapshot.readouts_total,
                    snapshot.readouts_delta,
                    snapshot.token_count,
                    snapshot.background_tokens_processed,
                    snapshot.runtime_latency_ms,
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
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    finally:
        if brain is not None:
            brain.stop(timeout_seconds=2.0)

    report.end_time = datetime.now(timezone.utc).isoformat()
    report.samples_collected = int(len(snapshots))
    report.total_errors = int(sum(item.errors for item in snapshots))
    report.total_readouts = int(snapshots[-1].readouts_total) if snapshots else 0
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
    report.topic_diversity_ratio = float(len(all_topics)) / float(max(1, report.total_readouts))
    report.final_prediction_error_mean = float(snapshots[-1].prediction_error_mean) if snapshots else 0.0
    report.final_prediction_error_max = float(snapshots[-1].prediction_error_max) if snapshots else 0.0
    report.depth_counts = dict(snapshots[-1].depth_counts) if snapshots else {}
    report.final_language_surface_summary = str(snapshots[-1].language_surface_summary) if snapshots else ""
    report.final_exploration_target = str(snapshots[-1].exploration_target) if snapshots else ""
    report.final_exploration_reason = str(snapshots[-1].exploration_reason) if snapshots else ""
    report.final_embedder = dict(snapshots[-1].embedder) if snapshots else {}
    report.final_runtime_truth = dict(snapshots[-1].runtime_truth) if snapshots else {}
    if snapshots:
        final_source_config = snapshots[-1].runtime_truth.get("source_configuration") if isinstance(snapshots[-1].runtime_truth, Mapping) else None
        if isinstance(final_source_config, Mapping) and final_source_config:
            report.source_configuration = {
                **dict(report.source_configuration),
                "final_runtime_truth_source_configuration": dict(final_source_config),
                "benchmark_semantics_note": (
                    "Long-test configures MarulhoBrain directly through feed/start. "
                    "The active service is only a /brain adapter, so no /status, /terminus, "
                    "or service-owned source configuration is part of this result."
                ),
            }
    report.action_count = int(snapshots[-1].action_count) if snapshots else 0
    report.readout_lifecycle_summary = _summarize_readout_lifecycle(snapshots)
    report.memory_pressure_report = _summarize_memory_pressure(snapshots)
    report.subcortex_workspace_report = _summarize_subcortex_workspace(snapshots)
    report.sample_readouts = []

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
            "readouts": item.readouts_total,
            "readouts_delta": item.readouts_delta,
            "background_tokens_processed": item.background_tokens_processed,
            "tick_count": item.tick_count,
            "runtime_running": item.runtime_running,
            "latency_ms": item.runtime_latency_ms,
            "memory_fill": item.memory_fill,
            "memory_size": item.memory_size,
            "consolidation": item.consolidation_mean,
            "ripple_tagged": item.ripple_tagged,
            "topic_diversity": item.topic_diversity,
            "prediction_error_mean": item.prediction_error_mean,
            "prediction_error_max": item.prediction_error_max,
            "depth_counts": item.depth_counts,
            "exploration_target": item.exploration_target,
            "exploration_reason": item.exploration_reason,
            "embedder": item.embedder,
            "runtime_truth": item.runtime_truth,
            "readout_lifecycle": item.readout_lifecycle,
            "memory_pressure": item.memory_pressure,
            "subcortex_workspace": item.subcortex_workspace,
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
                f"- {failure.get('name', '-')}: {str(failure.get('summary', '')).replace('|', '/')} "
                f"(action: {failure.get('recommended_action', '-')})"
            )

    source_config = report.source_configuration if isinstance(report.source_configuration, Mapping) else {}
    lines.extend(
        [
            "",
            "## Source Configuration",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Configured | {source_config.get('configured', report.terminus_configured)} |",
            f"| Preset | {source_config.get('preset', report.preset)} |",
            f"| Surface | {source_config.get('configuration_surface', '-')} |",
            f"| Source count | {source_config.get('source_count', 0)} |",
            f"| Source names | {', '.join(str(item) for item in list(source_config.get('source_names') or [])) or '-'} |",
            f"| Source types | {', '.join(str(item) for item in list(source_config.get('source_types') or [])) or '-'} |",
            f"| Sensory source count | {source_config.get('sensory_source_count', 0)} |",
            f"| Config hash | {source_config.get('configuration_hash', '-')} |",
            f"| Reproduction hint | {str(source_config.get('reproduction_hint', '-')).replace('|', '/')} |",
        ]
    )
    if source_config.get("benchmark_semantics_note"):
        lines.extend(["", str(source_config.get("benchmark_semantics_note"))])

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Brain configured | {report.terminus_configured} |",
            f"| Final brain loop running | {report.terminus_running} |",
            f"| Samples collected | {report.samples_collected} |",
            f"| Initial token count | {report.initial_token_count} |",
            f"| Final token count | {report.final_token_count} |",
            f"| Max background tokens processed | {report.max_background_tokens_processed} |",
            f"| Final tick count | {report.final_tick_count} |",
            f"| Total language/readouts | {report.total_readouts} |",
            f"| Unique topics | {report.unique_topics} |",
            f"| Topic diversity | {report.topic_diversity_ratio:.2f} topics/readout |",
            f"| Avg latency | {report.avg_latency_ms:.0f} ms |",
            f"| P95 latency | {report.p95_latency_ms:.0f} ms |",
            f"| Max latency | {report.max_latency_ms:.0f} ms |",
            f"| Final memory fill | {report.final_memory_fill:.1%} |",
            f"| Memory capacity | {report.memory_capacity} |",
            f"| Final consolidation | {report.final_consolidation:.3f} |",
            f"| Ripple-tagged memories | {report.final_ripple_tagged} |",
            f"| Prediction error (mean/max) | {report.final_prediction_error_mean:.3f} / {report.final_prediction_error_max:.3f} |",
            f"| Depth counts | quick={report.depth_counts.get('quick', 0)}, standard={report.depth_counts.get('standard', 0)}, deep={report.depth_counts.get('deep', 0)} |",
            f"| Active exploration target | {report.final_exploration_target or '-'} |",
            f"| Active exploration reason | {report.final_exploration_reason or '-'} |",
            f"| Embedder | {report.final_embedder.get('kind', '-')}: {report.final_embedder.get('model', '-')} |",
            f"| Embedder degraded | {report.final_embedder.get('degraded', False)} |",
            f"| Embedder fallback calls | {report.final_embedder.get('fallback_calls', 0)} |",
            f"| Embedder external throttle hits | {report.final_embedder.get('external_throttle_hits', 0)} |",
            f"| Runtime truth verdict | {report.final_runtime_truth.get('verdict', '-') if report.final_runtime_truth else '-'} |",
            f"| Runtime truth action | {report.final_runtime_truth.get('recommended_action', '-') if report.final_runtime_truth else '-'} |",
            f"| Recorded actions | {report.action_count} |",
            f"| Errors | {report.total_errors} |",
        ]
    )

    readout_summary = report.readout_lifecycle_summary if isinstance(report.readout_lifecycle_summary, Mapping) else {}
    memory_summary = report.memory_pressure_report if isinstance(report.memory_pressure_report, Mapping) else {}
    subcortex_workspace_summary = report.subcortex_workspace_report if isinstance(report.subcortex_workspace_report, Mapping) else {}
    lines.extend(
        [
            "",
            "## Liveness Diagnosis",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Readout attempts | {readout_summary.get('attempts', 0)} |",
            f"| Successful readouts | {readout_summary.get('successful', report.total_readouts)} |",
            f"| Blocked ticks | {readout_summary.get('blocked_ticks', 0)} |",
            f"| Wake triggers | {readout_summary.get('wake_triggers', {})} |",
            f"| Rejected / blocked reasons | {', '.join(str(item) for item in list(readout_summary.get('rejected_or_blocked_reasons') or [])) or '-'} |",
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
            "## Subcortex Workspace",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Final size / capacity | {subcortex_workspace_summary.get('final_size', 0)} / {subcortex_workspace_summary.get('capacity', 0)} |",
            f"| Max size | {subcortex_workspace_summary.get('max_size', 0)} |",
            f"| Active exploration | {subcortex_workspace_summary.get('active_exploration', {})} |",
            f"| Evidence boundary | {subcortex_workspace_summary.get('evidence_boundary', {})} |",
            f"| Final broadcast | {str(subcortex_workspace_summary.get('final_broadcast', '')).replace('|', '/')} |",
        ]
    )

    lines.extend(
        [
            "",
            "## Timeline",
            "",
            "| Time (s) | Tokens | Readouts | ΔReadouts | Bg tokens | Tick count | Latency (ms) | Memory | Topics | Ingestion |",
            "|----------|--------|----------|-----------|-----------|------------|--------------|--------|--------|-----------|",
        ]
    )
    for snapshot in report.snapshots:
        lines.append(
            f"| {snapshot['elapsed_s']:.0f} | {snapshot['token_count']} | {snapshot['readouts']} | {snapshot['readouts_delta']} | "
            f"{snapshot['background_tokens_processed']} | {snapshot['tick_count']} | {snapshot['latency_ms']:.0f} | "
            f"{snapshot['memory_fill']:.1%} | {snapshot['topic_diversity']} | {snapshot['ingestion_state'] or '-'} |"
        )

    if report.final_language_surface_summary:
        lines.extend([
            "",
            "## Language Surface",
            "",
            report.final_language_surface_summary,
        ])

    if report.sample_readouts:
        lines.extend([
            "",
            "## Sample Language Readouts",
            "",
        ])
        for index, readout in enumerate(report.sample_readouts[:10], 1):
            lines.append(f"{index}. {readout}")

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
    parser = argparse.ArgumentParser(description="Run a long MarulhoBrain test")
    parser.add_argument("--duration", type=float, default=20.0, help="Test duration in minutes (default: 20)")
    parser.add_argument("--interval", type=float, default=30.0, help="Sampling interval in seconds (default: 30)")
    parser.add_argument("--preset", type=str, default="curriculum", help="MarulhoBrain preset to use (default: curriculum)")
    parser.add_argument("--output", type=str, default="reports", help="Output directory for report (default: reports)")
    parser.add_argument(
        "--memory-capacity",
        type=int,
        default=DEFAULT_LONG_TEST_MEMORY_CAPACITY,
        help=f"Checkpoint memory capacity for the validation run (default: {DEFAULT_LONG_TEST_MEMORY_CAPACITY})",
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
        memory_capacity=args.memory_capacity,
    )
    json_path, md_path = write_report(report, output_dir=args.output)

    print("\nTest complete!")
    print(f"  Health: {report.health_verdict}")
    print(f"  Acceptance: {report.acceptance_verdict}")
    print(f"  Language/readouts: {report.total_readouts}")
    print(f"  Topics: {report.unique_topics}")
    print(f"  Avg latency: {report.avg_latency_ms:.0f}ms")
    print(f"  Report: {md_path}")
    print(f"  Data: {json_path}")
    return health_exit_code(report)


if __name__ == "__main__":
    sys.exit(main())


