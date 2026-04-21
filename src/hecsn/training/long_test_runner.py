"""Long-Test Runner -- exercises the full Terminus brain loop and generates a report.

Usage:
    python -m hecsn.training.long_test_runner --duration 20 --output reports/long_test_report.md

This runs Terminus with the curriculum preset for N minutes (default 20),
collecting metrics every 30s, then outputs a JSON + markdown report.

Metrics tracked:
- Thoughts generated, topics per thought, topic diversity
- Cortex latency (p50, p95, p99)
- Memory fill ratio, consolidation level, ripple-tagged count
- Neuromodulator levels (DA, 5-HT, ACh, NE)
- Predictive-error, active-exploration, and depth-policy telemetry
- Embedder state / degradation / fallback visibility
- Grounding confidence, SNN column utilization
- Errors and fallbacks
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """One measurement point during the test run."""
    timestamp: float = 0.0
    elapsed_s: float = 0.0
    thoughts_total: int = 0
    thoughts_delta: int = 0
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
    errors: int = 0


@dataclass
class TestReport:
    """Final report from a long test run."""
    start_time: str = ""
    end_time: str = ""
    duration_minutes: float = 0.0
    preset: str = ""
    cortex_model: str = ""
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
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    sample_thoughts: list[str] = field(default_factory=list)


def run_long_test(
    duration_minutes: float = 20.0,
    sample_interval_s: float = 30.0,
    preset: str = "curriculum",
    output_dir: str = "reports",
) -> TestReport:
    """Run a full Terminus brain test for the specified duration.

    Returns a TestReport with all collected metrics.
    """
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()

    from hecsn.service.manager import HECSNServiceManager
    from hecsn.training.checkpointing import save_trainer_checkpoint
    from hecsn.config.model_config import HECSNConfig
    from hecsn.training.model import HECSNModel
    from hecsn.training.trainer import HECSNTrainer

    report = TestReport()
    report.start_time = datetime.now(timezone.utc).isoformat()
    report.preset = preset
    report.duration_minutes = duration_minutes

    # Create a fresh model + checkpoint
    cfg = HECSNConfig(
        n_columns=64,
        column_latent_dim=32,
        bootstrap_tokens=0,
        memory_capacity=256,
        eta_competitive=0.05,
        enable_context_layer=True,
        enable_binding_layer=True,
    )
    model = HECSNModel(cfg)
    trainer = HECSNTrainer(model, cfg)

    tmpdir = Path(output_dir) / ".long_test_tmp" / datetime.now().strftime("%Y%m%d_%H%M%S")
    tmpdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_trainer_checkpoint(
        tmpdir / "test_checkpoint.pt",
        trainer,
        metadata={"test": "long_run"},
    )

    # Create manager
    manager = HECSNServiceManager(
        ckpt_path,
        trace_dir=tmpdir / "traces",
    )

    # Configure and start
    try:
        config_result = manager.quick_start_terminus(preset=preset)
        logger.info("Quick start result: %s", config_result.get("status"))
    except Exception as e:
        logger.error("Quick start failed: %s", e)
        report.total_errors += 1
        return report

    # Wait for brain to start
    time.sleep(2.0)

    # Verify cortex is using NIM (not disabled)
    if not manager._cortex_available:
        logger.error("Cortex is disabled — NVIDIA_API_KEY missing or NIM unreachable. "
                     "Set NVIDIA_API_KEY in .env and ensure network connectivity.")
        report.total_errors += 1
        return report

    # Collect metrics
    snapshots: list[MetricSnapshot] = []
    latencies: list[float] = []
    all_topics: set[str] = set()
    thoughts_seen: list[str] = []
    seen_thought_texts: set[str] = set()  # Dedup across sampling intervals
    start_time = time.time()
    last_thoughts_count = 0
    sample_count = 0

    duration_s = duration_minutes * 60.0
    logger.info("Starting long test: %.1f minutes, sampling every %.0fs",
                duration_minutes, sample_interval_s)

    try:
        next_sample_at = min(sample_interval_s, duration_s)
        while next_sample_at <= duration_s + 1e-6:
            sleep_time = max(0.0, start_time + next_sample_at - time.time())
            if sleep_time > 0:
                time.sleep(sleep_time)
            elapsed = time.time() - start_time
            sample_count += 1

            # Get cortex snapshot
            snap = MetricSnapshot()
            snap.timestamp = time.time()
            snap.elapsed_s = elapsed

            try:
                # Get thought count and recent thoughts
                thoughts_data = manager.cortex_thoughts(limit=10)
                snap.thoughts_total = thoughts_data.get("thoughts_generated", 0)
                snap.thoughts_delta = snap.thoughts_total - last_thoughts_count
                last_thoughts_count = snap.thoughts_total

                cortex_snap = manager.cortex_snapshot()
                snap.cortex_model = cortex_snap.get("model", "")
                snap.cortex_latency_ms = cortex_snap.get("avg_inference_ms", 0.0)
                signal_state = cortex_snap.get("cognitive_signals", {})
                snap.prediction_error_mean = float(signal_state.get("prediction_error_mean", 0.0))
                snap.prediction_error_max = float(signal_state.get("prediction_error_max", 0.0))
                drives = cortex_snap.get("drives", {})
                snap.da_level = float(drives.get("dopamine", 0.0))
                snap.serotonin_level = float(drives.get("serotonin", 0.0))
                snap.ach_level = float(drives.get("acetylcholine", 0.0))
                snap.ne_level = float(drives.get("norepinephrine", 0.0))
                quality = cortex_snap.get("quality", {})
                snap.dream_verification_rate = float(quality.get("dream_verification_rate", 0.0))
                depth_policy = cortex_snap.get("depth_policy", {})
                snap.depth_counts = dict(depth_policy.get("counts", {}))
                narrative = cortex_snap.get("narrative_self", {})
                snap.narrative_summary = str(narrative.get("summary", ""))
                active_exploration = cortex_snap.get("active_exploration", {})
                snap.exploration_target = str(active_exploration.get("target", ""))
                snap.exploration_reason = str(active_exploration.get("reason", ""))
                episodic_memory = cortex_snap.get("episodic_memory", {})
                snap.embedder = dict(episodic_memory.get("embedder", {}))

                for t in thoughts_data.get("thoughts", []):
                    lat = t.get("latency_ms", 0.0)
                    if lat > 0:
                        latencies.append(lat)
                    topics = t.get("topics", [])
                    if isinstance(topics, (list, tuple)):
                        all_topics.update(topics)
                    thought_text = t.get("thought", "")
                    # Deduplicate: only add thoughts not already seen
                    if thought_text and thought_text not in seen_thought_texts and len(thoughts_seen) < 50:
                        thoughts_seen.append(thought_text)
                        seen_thought_texts.add(thought_text)

            except Exception as e:
                snap.errors += 1
                logger.warning("Snapshot error: %s", e)

            # Memory/brain stats from terminus_status
            try:
                status = manager.terminus_status()
                runtime = status.get("terminus_runtime", {})
                memory_info = runtime.get("memory", {})
                snap.memory_fill = memory_info.get("fill_fraction", 0.0)
                snap.memory_size = memory_info.get("size", 0)
                snap.consolidation_mean = memory_info.get("mean_consolidation_level", 0.0)
                snap.ripple_tagged = memory_info.get("ripple_tagged", 0)
            except Exception:
                pass

            snap.topic_diversity = len(all_topics)
            snapshots.append(snap)

            # Progress log
            logger.info(
                "[%3.0fs/%3.0fs] thoughts=%d (+%d) latency=%.0fms "
                "mem=%.0f%% topics=%d",
                elapsed, duration_s, snap.thoughts_total, snap.thoughts_delta,
                snap.cortex_latency_ms, snap.memory_fill * 100, snap.topic_diversity,
            )

            if next_sample_at >= duration_s:
                break
            next_sample_at = min(duration_s, next_sample_at + sample_interval_s)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    finally:
        manager.close()

    # Compile report
    report.end_time = datetime.now(timezone.utc).isoformat()
    report.total_thoughts = last_thoughts_count
    report.total_errors = sum(s.errors for s in snapshots)
    report.final_memory_fill = snapshots[-1].memory_fill if snapshots else 0.0
    report.final_consolidation = snapshots[-1].consolidation_mean if snapshots else 0.0
    report.final_ripple_tagged = snapshots[-1].ripple_tagged if snapshots else 0
    report.unique_topics = len(all_topics)
    report.all_topics = sorted(all_topics)
    report.topic_diversity_ratio = len(all_topics) / max(1, last_thoughts_count)
    report.cortex_model = snapshots[0].cortex_model if snapshots else ""
    report.final_prediction_error_mean = snapshots[-1].prediction_error_mean if snapshots else 0.0
    report.final_prediction_error_max = snapshots[-1].prediction_error_max if snapshots else 0.0
    report.final_dream_verification_rate = snapshots[-1].dream_verification_rate if snapshots else 0.0
    report.depth_counts = dict(snapshots[-1].depth_counts) if snapshots else {}
    report.final_narrative_summary = snapshots[-1].narrative_summary if snapshots else ""
    report.final_exploration_target = snapshots[-1].exploration_target if snapshots else ""
    report.final_exploration_reason = snapshots[-1].exploration_reason if snapshots else ""
    report.final_embedder = dict(snapshots[-1].embedder) if snapshots else {}
    report.sample_thoughts = thoughts_seen[:20]

    if latencies:
        latencies.sort()
        report.avg_latency_ms = sum(latencies) / len(latencies)
        report.p95_latency_ms = latencies[int(0.95 * len(latencies))] if len(latencies) > 1 else latencies[0]
        report.max_latency_ms = latencies[-1]

    report.snapshots = [
        {
            "elapsed_s": s.elapsed_s,
            "thoughts": s.thoughts_total,
            "latency_ms": s.cortex_latency_ms,
            "memory_fill": s.memory_fill,
            "consolidation": s.consolidation_mean,
            "ripple_tagged": s.ripple_tagged,
            "topic_diversity": s.topic_diversity,
            "prediction_error_mean": s.prediction_error_mean,
            "prediction_error_max": s.prediction_error_max,
            "dream_verification_rate": s.dream_verification_rate,
            "depth_counts": s.depth_counts,
            "exploration_target": s.exploration_target,
            "exploration_reason": s.exploration_reason,
            "embedder": s.embedder,
            "da": s.da_level,
        }
        for s in snapshots
    ]

    return report


def write_report(report: TestReport, output_dir: str = "reports") -> tuple[str, str]:
    """Write report as JSON and markdown. Returns (json_path, md_path)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out / f"long_test_{timestamp}.json"
    md_path = out / f"long_test_{timestamp}.md"

    # JSON
    json_path.write_text(json.dumps(report.__dict__, indent=2, default=str), encoding="utf-8")

    # Markdown
    lines = [
        f"# Long Test Report",
        f"",
        f"**Start:** {report.start_time}",
        f"**End:** {report.end_time}",
        f"**Duration:** {report.duration_minutes:.1f} minutes",
        f"**Preset:** {report.preset}",
        f"**Cortex:** {report.cortex_model}",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total thoughts | {report.total_thoughts} |",
        f"| Unique topics | {report.unique_topics} |",
        f"| Topic diversity | {report.topic_diversity_ratio:.2f} topics/thought |",
        f"| Avg latency | {report.avg_latency_ms:.0f} ms |",
        f"| P95 latency | {report.p95_latency_ms:.0f} ms |",
        f"| Max latency | {report.max_latency_ms:.0f} ms |",
        f"| Final memory fill | {report.final_memory_fill:.1%} |",
        f"| Final consolidation | {report.final_consolidation:.3f} |",
        f"| Ripple-tagged memories | {report.final_ripple_tagged} |",
        f"| Prediction error (mean/max) | {report.final_prediction_error_mean:.3f} / {report.final_prediction_error_max:.3f} |",
        f"| Dream verification rate | {report.final_dream_verification_rate:.1%} |",
        f"| Depth counts | quick={report.depth_counts.get('quick', 0)}, standard={report.depth_counts.get('standard', 0)}, deep={report.depth_counts.get('deep', 0)} |",
        f"| Active exploration target | {report.final_exploration_target or '-'} |",
        f"| Active exploration reason | {report.final_exploration_reason or '-'} |",
        f"| Embedder | {report.final_embedder.get('kind', '-')}: {report.final_embedder.get('model', '-') } |",
        f"| Embedder degraded | {report.final_embedder.get('degraded', False)} |",
        f"| Embedder fallback calls | {report.final_embedder.get('fallback_calls', 0)} |",
        f"| Embedder rate-limit hits | {report.final_embedder.get('rate_limit_hits', 0)} |",
        f"| Errors | {report.total_errors} |",
        f"",
        f"## Timeline",
        f"",
        f"| Time (s) | Thoughts | Latency (ms) | Memory | Topics |",
        f"|----------|----------|--------------|--------|--------|",
    ]
    for s in report.snapshots:
        lines.append(
            f"| {s['elapsed_s']:.0f} | {s['thoughts']} | {s['latency_ms']:.0f} "
            f"| {s['memory_fill']:.1%} | {s['topic_diversity']} |"
        )

    if report.final_narrative_summary:
        lines.extend([
            f"",
            f"## Narrative Self",
            f"",
            report.final_narrative_summary,
        ])

    if report.sample_thoughts:
        lines.extend([
            f"",
            f"## Sample Thoughts",
            f"",
        ])
        for i, thought in enumerate(report.sample_thoughts[:10], 1):
            lines.append(f"{i}. {thought}")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return str(json_path), str(md_path)


def main():
    parser = argparse.ArgumentParser(description="Run a long Terminus brain test")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Test duration in minutes (default: 20)")
    parser.add_argument("--interval", type=float, default=30.0,
                        help="Sampling interval in seconds (default: 30)")
    parser.add_argument("--preset", type=str, default="curriculum",
                        help="Terminus preset to use (default: curriculum)")
    parser.add_argument("--output", type=str, default="reports",
                        help="Output directory for report (default: reports)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print(f"Starting long test: {args.duration} min, preset={args.preset}")
    print(f"Sampling every {args.interval}s, output to {args.output}/")
    print("Press Ctrl+C to stop early (report still generated)")
    print()

    report = run_long_test(
        duration_minutes=args.duration,
        sample_interval_s=args.interval,
        preset=args.preset,
        output_dir=args.output,
    )

    json_path, md_path = write_report(report, output_dir=args.output)
    print(f"\nTest complete!")
    print(f"  Thoughts: {report.total_thoughts}")
    print(f"  Topics: {report.unique_topics}")
    print(f"  Avg latency: {report.avg_latency_ms:.0f}ms")
    print(f"  Report: {md_path}")
    print(f"  Data: {json_path}")


if __name__ == "__main__":
    main()
