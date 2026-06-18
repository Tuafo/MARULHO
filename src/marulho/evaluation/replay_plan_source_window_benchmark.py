"""Benchmark bounded service replay-plan source windows.

This evaluation stresses the replay-plan endpoint with large recorded histories
and verifies that candidate construction stays bounded while preserving a
high-signal feedback target outside the recent source tail.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time
import tracemalloc
from typing import Any

from marulho.service.living_loop_replay import build_replay_plan


def _process_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    try:
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def _cuda_report() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"torch_available": False, "cuda_available": False, "gpu_used": False}
    available = bool(torch.cuda.is_available())
    report: dict[str, Any] = {
        "torch_available": True,
        "cuda_available": available,
        "gpu_used": False,
    }
    if available:
        try:
            report["device_name"] = torch.cuda.get_device_name(0)
            report["memory_allocated_mib"] = round(
                float(torch.cuda.memory_allocated(0)) / (1024.0 * 1024.0),
                3,
            )
            report["memory_reserved_mib"] = round(
                float(torch.cuda.memory_reserved(0)) / (1024.0 * 1024.0),
                3,
            )
        except Exception as exc:
            report["cuda_query_error"] = str(exc)
    return report


def _make_payload(source_size: int, feedback_size: int, domain_size: int) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for index in range(source_size):
        created_second = index % 60
        created_minute = (index // 60) % 60
        episodes.append(
            {
                "episode_id": f"ep-{index}",
                "operation": "query",
                "status": "succeeded",
                "created_at": f"2026-01-01T00:{created_minute:02d}:{created_second:02d}+00:00",
                "verification": {
                    "status": "verified" if index % 3 else "unverified",
                    "confidence": 0.7,
                },
                "prediction": {"status": "complete", "confidence": 0.7},
            }
        )
        actions.append(
            {
                "action_id": f"act-{index}",
                "action_type": "search",
                "recorded_at": f"2026-01-01T01:{created_minute:02d}:{created_second:02d}+00:00",
                "verification": {"status": "verified", "confidence": 0.8},
            }
        )
        predictions.append(
            {
                "prediction_id": f"pred-{index}",
                "source_kind": "runtime_episode",
                "source_id": f"ep-{index}",
                "created_at": f"2026-01-01T02:{created_minute:02d}:{created_second:02d}+00:00",
                "status": "complete",
                "confidence": 0.8,
            }
        )

    feedback: list[dict[str, Any]] = []
    for index in range(feedback_size):
        target = 42 if index == 0 else max(0, source_size - 1 - index)
        feedback.append(
            {
                "feedback_id": f"fb-{index}",
                "target_type": "runtime_episode",
                "target_id": f"ep-{target}",
                "created_at": f"2026-01-02T00:{(index // 60) % 60:02d}:{index % 60:02d}+00:00",
                "verdict": "contradicted" if index == 0 else "unverified",
                "applied_status": "contradicted" if index == 0 else "unverified",
                "confidence": 0.95,
                "summary": "Old high-signal contradiction."
                if index == 0
                else "Recent unverified note.",
                "has_corrected_output": index == 0,
            }
        )

    return {
        "state_revision": 1,
        "token_count": 123,
        "runtime_episode_count": source_size,
        "action_count": source_size,
        "prediction_count": source_size,
        "runtime_episodes": episodes,
        "actions": actions,
        "predictions": predictions,
        "uncertain_domains": [
            {"domain": f"domain-{index}", "total_uncertain_signals": index % 5}
            for index in range(domain_size)
        ],
        "feedback_summary": {
            "feedback_count": len(feedback),
            "contradicted_count": 1,
            "unverified_count": max(0, len(feedback) - 1),
            "recent_feedback": feedback,
        },
        "world_model_lite": {"uncertainty": 0.3},
        "memory_health": {"status": "available", "fill_ratio": 0.2},
        "subcortex_sleep_pressure": {"fatigue": 0.1},
        "policy_decision": {"action": "continue_current_policy"},
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    payload = _make_payload(
        source_size=int(args.source_size),
        feedback_size=int(args.feedback_size),
        domain_size=int(args.domain_size),
    )
    rss_before = _process_rss_mb()
    timings_ms: list[float] = []
    last: dict[str, Any] | None = None
    for _ in range(int(args.runs)):
        started = time.perf_counter()
        last = build_replay_plan(
            payload,
            limit=int(args.limit),
            created_at="2026-01-02T01:00:00+00:00",
        ).to_payload()
        timings_ms.append((time.perf_counter() - started) * 1000.0)
    rss_after = _process_rss_mb()
    tracemalloc.start()
    _ = build_replay_plan(
        payload,
        limit=int(args.limit),
        created_at="2026-01-02T01:00:00+00:00",
    ).to_payload()
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert last is not None
    candidates = list(last.get("candidates") or [])
    candidate_ids = [
        str(candidate.get("target_id"))
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    top = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    source_window = dict(last.get("source_window") or {})
    source_counts = dict(source_window.get("source_counts") or {})
    window_counts = dict(source_window.get("window_counts") or {})
    device_placement = dict(source_window.get("device_placement") or {})
    latency = {
        "runs": int(args.runs),
        "mean_ms": round(float(statistics.mean(timings_ms)), 3),
        "median_ms": round(float(statistics.median(timings_ms)), 3),
        "p95_ms": round(float(sorted(timings_ms)[-1]), 3),
        "samples_ms": [round(float(value), 3) for value in timings_ms],
    }
    if args.baseline_unbounded_mean_ms is not None:
        baseline = float(args.baseline_unbounded_mean_ms)
        latency["baseline_unbounded_mean_ms"] = round(baseline, 3)
        latency["speedup_vs_unbounded_baseline"] = round(
            baseline / max(1.0e-9, float(statistics.mean(timings_ms))),
            3,
        )
    quality = {
        "top_candidate_target_id": top.get("target_id"),
        "top_candidate_operation": top.get("operation"),
        "top_candidate_reason_codes": list(top.get("reason_codes") or []),
        "old_feedback_target_recalled": top.get("target_id") == "ep-42",
        "old_feedback_target_in_top_k": "ep-42" in candidate_ids,
        "candidate_ids": candidate_ids,
    }
    pass_checks = {
        "surface_present": source_window.get("surface")
        == "bounded_replay_plan_source_window.v1",
        "runs_live_tick_false": source_window.get("runs_live_tick") is False,
        "gpu_not_used": device_placement.get("gpu_used") is False,
        "old_feedback_target_recalled": bool(quality["old_feedback_target_recalled"]),
        "source_windows_bounded": all(
            int(window_counts.get(key, 0) or 0)
            <= int(source_window.get("source_limits", {}).get(key, 0) or 0)
            + int(source_window.get("source_limits", {}).get("feedback_target_stubs", 0) or 0)
            for key in ("runtime_episodes", "actions", "predictions", "uncertain_domains")
        ),
        "large_sources_truncated": all(
            bool(dict(source_window.get("truncated_source_counts") or {}).get(key))
            for key in ("runtime_episodes", "actions", "predictions", "uncertain_domains")
        ),
    }
    return {
        "surface": "bounded_replay_plan_source_window_benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "pass": all(pass_checks.values()),
        "pass_checks": pass_checks,
        "input": {
            "source_size_per_stream": int(args.source_size),
            "feedback_size": int(args.feedback_size),
            "domain_size": int(args.domain_size),
            "limit": int(args.limit),
        },
        "quality": quality,
        "latency": latency,
        "source_window": source_window,
        "source_counts": source_counts,
        "window_counts": window_counts,
        "device_placement": device_placement,
        "resource_behavior": {
            "process_rss_before_mib": None if rss_before is None else round(rss_before, 3),
            "process_rss_after_mib": None if rss_after is None else round(rss_after, 3),
            "process_rss_delta_mib": None
            if rss_before is None or rss_after is None
            else round(rss_after - rss_before, 3),
            "python_tracemalloc_current_mib": round(
                float(traced_current) / (1024.0 * 1024.0),
                3,
            ),
            "python_tracemalloc_peak_mib": round(
                float(traced_peak) / (1024.0 * 1024.0),
                3,
            ),
            "cuda": _cuda_report(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-size", type=int, default=20_000)
    parser.add_argument("--feedback-size", type=int, default=128)
    parser.add_argument("--domain-size", type=int, default=2_000)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--baseline-unbounded-mean-ms", type=float)
    args = parser.parse_args()

    report = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
