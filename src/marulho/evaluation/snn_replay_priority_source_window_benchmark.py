"""Benchmark bounded SNN replay-priority source windows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock
import statistics
import sys
import time
import tracemalloc
from typing import Any, Mapping, Sequence

from marulho.service.replay_runtime import (
    DEFAULT_SNN_REPLAY_EVALUATION_CONTEXTS,
    SNN_REPLAY_PRIORITY_CONTEXT_WINDOW_LIMIT,
    ReplayController,
    ReplayControllerDependencies,
)


@dataclass
class _RuntimeState:
    state_revision: int = 11

    def __post_init__(self) -> None:
        self.dirty_without_revision_calls = 0

    def mark_dirty_without_revision(self) -> None:
        self.dirty_without_revision_calls += 1


class _ReplayHarness:
    def __init__(self) -> None:
        self.lock = RLock()
        self.runtime_state = _RuntimeState()
        self.trainer = type("_Trainer", (), {"token_count": 2048})()

    @staticmethod
    def normalize_text(value: Any, *, max_chars: int = 2000) -> str:
        text = " ".join(str(value).split()).strip()
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + "..."
        return text


def _controller() -> ReplayController:
    harness = _ReplayHarness()
    return ReplayController(
        ReplayControllerDependencies(
            lock=harness.lock,
            normalize_feedback_text=harness.normalize_text,
            runtime_state=harness.runtime_state,
            trainer=lambda: harness.trainer,
        )
    )


def _mismatch_report(score: float) -> dict[str, Any]:
    return {
        "surface": "snn_language_sequence_mismatch_probe.v1",
        "available": True,
        "owned_by_marulho": True,
        "prediction_error": {"mismatch_score": float(score)},
        "promotion_gate": {"status": "ready_for_operator_review"},
    }


def _pressure_report(score: float) -> dict[str, Any]:
    return {
        "surface": "snn_language_plasticity_pressure.v1",
        "available": True,
        "owned_by_marulho": True,
        "plasticity_pressure": {"pressure_score": float(score)},
        "promotion_gate": {"status": "ready_for_operator_review"},
    }


def _seed_contexts(controller: ReplayController, *, retention_count: int) -> dict[str, Any]:
    bounded_retention = max(1, min(int(retention_count), DEFAULT_SNN_REPLAY_EVALUATION_CONTEXTS))
    old_target = controller.record_snn_replay_evaluation_context(
        mismatch_report=_mismatch_report(0.95),
        pressure_report=_pressure_report(0.95),
        source_metadata={"source": "benchmark", "label": "old-high-signal-target"},
    )
    for index in range(bounded_retention - 1):
        controller.record_snn_replay_evaluation_context(
            mismatch_report=_mismatch_report(0.9),
            pressure_report=_pressure_report(0.9),
            source_metadata={"source": "benchmark", "label": f"recent-no-readout-{index}"},
        )
    return old_target


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
        report["device_name"] = torch.cuda.get_device_name(0)
        report["memory_allocated_mib"] = round(
            float(torch.cuda.memory_allocated(0)) / (1024.0 * 1024.0),
            3,
        )
        report["memory_reserved_mib"] = round(
            float(torch.cuda.memory_reserved(0)) / (1024.0 * 1024.0),
            3,
        )
    return report


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    controller = _controller()
    old_target = _seed_contexts(
        controller,
        retention_count=int(args.retention_count),
    )
    old_context_id = str(old_target["replay_evaluation_context_id"])
    readout_priority = {
        "surface": "snn_language_readout_replay_priority.v1",
        "candidates": [
            {
                "priority_score": 99.0,
                "all_labels_grounded": True,
                "replay_evaluation_context_id": old_context_id,
            }
        ],
    }
    rss_before = _process_rss_mb()
    timings_ms: list[float] = []
    last: dict[str, Any] | None = None
    for _ in range(max(1, int(args.runs))):
        started = time.perf_counter()
        last = controller.snn_replay_consolidation_priority_queue(
            readout_replay_priority_report=readout_priority,
            limit=int(args.limit),
        )
        timings_ms.append((time.perf_counter() - started) * 1000.0)
    rss_after = _process_rss_mb()
    tracemalloc.start()
    _ = controller.snn_replay_consolidation_priority_queue(
        readout_replay_priority_report=readout_priority,
        limit=int(args.limit),
    )
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert last is not None
    source_window = dict(last.get("source_window") or {})
    candidates = [dict(item) for item in list(last.get("candidates") or []) if isinstance(item, Mapping)]
    selected_ids = [str(item.get("replay_evaluation_context_id") or "") for item in candidates]
    old_candidate = next(
        (
            item
            for item in candidates
            if str(item.get("replay_evaluation_context_id") or "") == old_context_id
        ),
        {},
    )
    bounded_verified = int(source_window.get("verified_context_count", 0) or 0)
    retained = int(source_window.get("context_retention_count", 0) or 0)
    quality = {
        "old_readout_target_selected": old_context_id in selected_ids,
        "old_readout_target_source": (
            dict(old_candidate.get("source_window") or {}).get("source")
            if old_candidate
            else None
        ),
        "old_readout_target_reason_codes": list(old_candidate.get("reason_codes") or [])
        if old_candidate
        else [],
        "selected_candidate_count": int(len(candidates)),
        "selected_context_ids": selected_ids,
    }
    latency = {
        "runs": int(max(1, int(args.runs))),
        "mean_ms": round(float(statistics.mean(timings_ms)), 6),
        "median_ms": round(float(statistics.median(timings_ms)), 6),
        "p95_ms": round(float(sorted(timings_ms)[-1]), 6),
        "samples_ms": [round(float(value), 6) for value in timings_ms],
    }
    pass_checks = {
        "surface_present": source_window.get("surface")
        == "bounded_snn_replay_priority_source_window.v1",
        "bounded_below_retention": 0 < bounded_verified < retained,
        "source_window_limit_respected": int(
            source_window.get("recent_context_window_count", 0) or 0
        )
        <= SNN_REPLAY_PRIORITY_CONTEXT_WINDOW_LIMIT,
        "old_target_selected": bool(quality["old_readout_target_selected"]),
        "old_target_from_readout_stub": quality["old_readout_target_source"]
        == "readout_priority_target_context_id",
        "no_global_scan": source_window.get("global_candidate_scan") is False,
        "cpu_only": source_window.get("archival_storage_device") == "cpu"
        and source_window.get("gpu_used") is False,
        "not_live_tick": source_window.get("runs_live_tick") is False,
        "no_language_reasoning": source_window.get("language_reasoning") is False,
    }
    return {
        "surface": "bounded_snn_replay_priority_source_window_benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "pass": all(pass_checks.values()),
        "pass_checks": pass_checks,
        "input": {
            "retention_count": retained,
            "requested_retention_count": int(args.retention_count),
            "limit": int(args.limit),
            "runs": int(max(1, int(args.runs))),
        },
        "quality": quality,
        "latency": latency,
        "source_window": source_window,
        "selection_budget": dict(source_window.get("selection_budget") or {}),
        "memory_budget": {
            "selection_criteria": (
                "recent replay-evaluation contexts plus explicit readout-target ids"
            ),
            "retained_context_count": retained,
            "bounded_verified_context_count": bounded_verified,
            "recent_context_window_limit": int(
                source_window.get("recent_context_window_limit", 0) or 0
            ),
            "readout_target_context_count": int(
                source_window.get("readout_target_context_count", 0) or 0
            ),
            "archival_storage_device": "cpu",
            "active_computation_device": "cpu",
            "runs_live_tick": False,
            "runs_every_token": False,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "language_reasoning": False,
        },
        "resource_behavior": {
            "process_rss_before_mib": None if rss_before is None else round(rss_before, 3),
            "process_rss_after_mib": None if rss_after is None else round(rss_after, 3),
            "process_rss_delta_mib": None
            if rss_before is None or rss_after is None
            else round(rss_after - rss_before, 3),
            "python_tracemalloc_current_mib": round(
                float(traced_current) / (1024.0 * 1024.0),
                6,
            ),
            "python_tracemalloc_peak_mib": round(
                float(traced_peak) / (1024.0 * 1024.0),
                6,
            ),
            "cuda": _cuda_report(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention-count", type=int, default=64)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run_benchmark(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
