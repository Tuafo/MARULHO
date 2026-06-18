"""Benchmark bounded SNN rollout rehearsal source windows."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from threading import RLock
import statistics
import sys
import time
import tracemalloc
from typing import Any, Mapping, Sequence

import torch

from marulho.service.runtime_state import RuntimeState
from marulho.service.snn_language_readout_ledger import (
    SNN_ROLLOUT_REHEARSAL_SOURCE_WINDOW_LIMIT,
    SNNLanguageReadoutEvidenceLedger,
)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _target(step_index: int, *, label: str, base_index: int) -> dict[str, Any]:
    sparse_indices = [
        int((base_index + step_index) % 64),
        int((base_index + step_index + 1) % 64),
    ]
    return {
        "step_index": int(step_index),
        "selected_label": str(label),
        "grounded": True,
        "selection_score": 0.4,
        "transition_support": 0.6,
        "predicted_sparse_indices": sparse_indices,
        "active_indices_hash": _sha256_json(sparse_indices),
    }


def _rollout_event(
    ledger: SNNLanguageReadoutEvidenceLedger,
    index: int,
    *,
    label: str,
    weights_hash: str,
) -> dict[str, Any]:
    event = {
        "surface": "snn_language_readout_rollout_evidence_ledger_event.v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "state_revision": 11,
        "operator_id": "benchmark",
        "rollout_replay_evaluation_hash": f"rollout-eval-hash-{index}",
        "rollout_hash": f"rollout-hash-{index}",
        "rollout_id": f"snn-readout-rollout:rollout-hash-{index}",
        "prediction_hash": f"prediction-hash-{index}",
        "current_sparse_code_hash": f"current-sparse-code-hash-{index}",
        "transition_memory_evaluation_hash": f"evaluation-hash-{index}",
        "persistent_transition_weights_hash": weights_hash,
        "server_transition_memory_hash": weights_hash,
        "server_transition_memory_hash_match": True,
        "transition_memory_state_source": (
            "service.runtime_facade.snn_language_plasticity_runtime_state"
        ),
        "device_evidence": {
            "requested_device": "cpu",
            "tensor_device": "cpu",
            "cuda_tensor": False,
            "device_source": "benchmark",
        },
        "target_count": 2,
        "trace_step_count": 2,
        "replay_targets": [
            _target(0, label=label, base_index=index),
            _target(1, label=label, base_index=index),
        ],
        "recorded_in_ledger": True,
        "eligible_for_replay_priority": False,
        "eligible_for_cognition_substrate": False,
        "freeform_language_generation": False,
        "material_hash_algorithm": "sha256_canonical_json",
    }
    evidence_hash = ledger._rollout_ledger_event_material_hash(event)  # noqa: SLF001
    event["rollout_evidence_hash"] = evidence_hash
    event["rollout_evidence_id"] = (
        f"snn-readout-rollout-evidence:{evidence_hash[:16]}"
    )
    return event


def _seed_events(
    ledger: SNNLanguageReadoutEvidenceLedger,
    *,
    retention_count: int,
) -> list[dict[str, Any]]:
    count = max(SNN_ROLLOUT_REHEARSAL_SOURCE_WINDOW_LIMIT + 1, int(retention_count))
    events: list[dict[str, Any]] = []
    for source_index in range(count):
        if source_index < 4:
            label = "recent-high-signal-rollout"
            weights_hash = "recent-high-signal-rollout-weights"
        elif source_index >= count - 4:
            label = "old-retained-rollout"
            weights_hash = "old-retained-rollout-weights"
        else:
            label = f"rollout-distractor-{source_index}"
            weights_hash = f"rollout-weights-{source_index % 29}"
        events.append(
            _rollout_event(
                ledger,
                source_index,
                label=label,
                weights_hash=weights_hash,
            )
        )
    return events


def _legacy_full_retained_policy(
    ledger: SNNLanguageReadoutEvidenceLedger,
    events: Sequence[Mapping[str, Any]],
    *,
    candidate_limit: int,
) -> dict[str, Any]:
    """Diagnostic-only copy of the retired all-retained rollout policy shape."""

    retained_events = [deepcopy(dict(item)) for item in events if isinstance(item, Mapping)]
    rollout_counts: dict[str, int] = {}
    transition_counts: dict[str, int] = {}
    for event in retained_events:
        rollout_key = str(event.get("rollout_hash") or "")
        transition_key = str(event.get("persistent_transition_weights_hash") or "")
        if rollout_key:
            rollout_counts[rollout_key] = rollout_counts.get(rollout_key, 0) + 1
        if transition_key:
            transition_counts[transition_key] = transition_counts.get(transition_key, 0) + 1
    candidates: list[dict[str, Any]] = []
    total = max(1, len(retained_events))
    for index, event in enumerate(retained_events):
        targets = [
            ledger._normalized_rollout_replay_target(item, index=target_index)  # noqa: SLF001
            for target_index, item in enumerate(list(event.get("replay_targets") or []))
            if isinstance(item, Mapping)
        ][:32]
        observed_device = (
            event.get("device_evidence")
            if isinstance(event.get("device_evidence"), Mapping)
            else {}
        )
        requested_device = str(observed_device.get("requested_device") or "")
        tensor_device = str(observed_device.get("tensor_device") or "")
        cuda_tensor = bool(observed_device.get("cuda_tensor"))
        requested_cuda_honored = (
            not requested_device.startswith("cuda")
            or (tensor_device.startswith("cuda") and cuda_tensor)
        )
        provenance_complete = bool(
            event.get("rollout_replay_evaluation_hash")
            and event.get("rollout_hash")
            and event.get("prediction_hash")
            and event.get("current_sparse_code_hash")
            and event.get("transition_memory_evaluation_hash")
            and event.get("persistent_transition_weights_hash")
            and event.get("server_transition_memory_hash")
            and event.get("server_transition_memory_hash_match")
            and str(event.get("transition_memory_state_source") or "")
            == "service.runtime_facade.snn_language_plasticity_runtime_state"
        )
        grounding_complete = bool(targets) and all(
            bool(item.get("grounded")) for item in targets
        )
        trace_integrity_complete = bool(targets) and all(
            bool(item.get("active_indices_hash_valid")) for item in targets
        )
        device_evidence_complete = bool(tensor_device) and requested_cuda_honored
        evidence_hash_valid = str(event.get("rollout_evidence_hash") or "") == (
            ledger._rollout_ledger_event_material_hash(event)  # noqa: SLF001
        )
        if not (
            provenance_complete
            and grounding_complete
            and trace_integrity_complete
            and device_evidence_complete
            and evidence_hash_valid
        ):
            continue
        rollout_key = str(event.get("rollout_hash") or "")
        transition_key = str(event.get("persistent_transition_weights_hash") or "")
        recency = (
            1.0 - min(1.0, index / max(1, total - 1)) if total > 1 else 1.0
        )
        repetition = (
            min(1.0, rollout_counts.get(rollout_key, 0) / 3.0)
            if rollout_key
            else 0.0
        )
        transition_reuse = (
            min(1.0, transition_counts.get(transition_key, 0) / 3.0)
            if transition_key
            else 0.0
        )
        score = 100.0 * (
            0.35
            + 0.20
            + 0.15
            + 0.15 * recency
            + 0.10 * transition_reuse
            + 0.05 * repetition
        )
        candidates.append(
            {
                "rollout_evidence_hash": event.get("rollout_evidence_hash"),
                "rollout_hash": event.get("rollout_hash"),
                "replay_targets": targets,
                "target_count": len(targets),
                "priority_score": float(score),
            }
        )
    candidates.sort(
        key=lambda item: (
            -float(item["priority_score"]),
            -int(item["target_count"]),
            str(item.get("rollout_evidence_hash") or ""),
        )
    )
    selected = [
        {**candidate, "rank": rank}
        for rank, candidate in enumerate(
            candidates[: max(0, min(int(candidate_limit), 32))],
            start=1,
        )
    ]
    return {
        "surface": "diagnostic_legacy_snn_rollout_full_retained_policy.v1",
        "candidate_count_before_rank": int(len(candidates)),
        "candidate_count": int(len(selected)),
        "candidates": selected,
    }


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


def _timed_runs(*, runs: int, fn: Any) -> tuple[dict[str, Any], list[float]]:
    samples: list[float] = []
    last: dict[str, Any] | None = None
    for _ in range(max(1, int(runs))):
        started = time.perf_counter()
        last = fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    assert last is not None
    return last, samples


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, Any] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
        limit=max(1, int(args.retention_count)),
    )
    events = _seed_events(ledger, retention_count=int(args.retention_count))
    ledger_state.update(
        {
            "rollout_events": deepcopy(events),
            "total_rollout_recorded_count": int(len(events)),
            "last_rollout_recorded_at": events[0]["recorded_at"] if events else None,
        }
    )

    rss_before = _process_rss_mb()
    bounded, bounded_samples = _timed_runs(
        runs=int(args.runs),
        fn=lambda: ledger.rollout_rehearsal_promotion_policy(
            candidate_limit=int(args.limit)
        ),
    )
    legacy, legacy_samples = _timed_runs(
        runs=int(args.runs),
        fn=lambda: _legacy_full_retained_policy(
            ledger,
            ledger_state["rollout_events"],
            candidate_limit=int(args.limit),
        ),
    )
    rss_after = _process_rss_mb()
    tracemalloc.start()
    _ = ledger.rollout_rehearsal_promotion_policy(candidate_limit=int(args.limit))
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    source_window = dict(bounded.get("source_window") or {})
    bounded_candidates = [
        dict(item)
        for item in list(bounded.get("candidates") or [])
        if isinstance(item, Mapping)
    ]
    legacy_candidates = [
        dict(item)
        for item in list(legacy.get("candidates") or [])
        if isinstance(item, Mapping)
    ]
    bounded_top_hash = (
        bounded_candidates[0].get("rollout_evidence_hash")
        if bounded_candidates
        else None
    )
    legacy_top_hash = (
        legacy_candidates[0].get("rollout_evidence_hash")
        if legacy_candidates
        else None
    )
    bounded_top_labels = (
        [
            str(target.get("selected_label") or "")
            for target in list(bounded_candidates[0].get("replay_targets") or [])
        ]
        if bounded_candidates
        else []
    )
    bounded_mean = float(statistics.mean(bounded_samples))
    legacy_mean = float(statistics.mean(legacy_samples))
    quality = {
        "bounded_top_matches_legacy_top": bounded_top_hash == legacy_top_hash,
        "bounded_top_labels": bounded_top_labels,
        "recent_high_signal_selected": "recent-high-signal-rollout"
        in bounded_top_labels,
        "selected_candidate_count": int(len(bounded_candidates)),
        "bounded_selected_hashes": [
            str(item.get("rollout_evidence_hash") or "")
            for item in bounded_candidates
        ],
        "legacy_selected_hashes": [
            str(item.get("rollout_evidence_hash") or "")
            for item in legacy_candidates
        ],
    }
    pass_checks = {
        "surface_present": bounded.get("surface")
        == "snn_language_readout_rollout_rehearsal_promotion_policy.v1",
        "source_window_present": source_window.get("surface")
        == "bounded_snn_readout_rollout_rehearsal_source_window.v1",
        "source_window_limit_respected": int(
            source_window.get("source_event_window_count", 0) or 0
        )
        <= SNN_ROLLOUT_REHEARSAL_SOURCE_WINDOW_LIMIT,
        "bounded_below_retention": int(
            source_window.get("source_event_window_count", 0) or 0
        )
        < int(source_window.get("source_event_retention_count", 0) or 0),
        "scored_only_source_window": int(
            source_window.get("candidate_count_before_rank", 0) or 0
        )
        == int(source_window.get("source_event_window_count", 0) or 0),
        "top_quality_matches_legacy": bool(quality["bounded_top_matches_legacy_top"]),
        "recent_high_signal_selected": bool(quality["recent_high_signal_selected"]),
        "no_global_scan": source_window.get("global_candidate_scan") is False
        and source_window.get("global_score_scan") is False,
        "cpu_only": source_window.get("archival_storage_device") == "cpu"
        and source_window.get("score_device") == "cpu"
        and source_window.get("gpu_used") is False,
        "not_live_tick": source_window.get("runs_live_tick") is False
        and source_window.get("runs_every_token") is False,
        "no_language_reasoning": source_window.get("language_reasoning") is False
        and source_window.get("raw_text_payload_loaded") is False,
    }
    return {
        "surface": "bounded_snn_readout_rollout_rehearsal_source_window_benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "pass": all(pass_checks.values()),
        "pass_checks": pass_checks,
        "input": {
            "retention_count": int(len(events)),
            "requested_retention_count": int(args.retention_count),
            "limit": int(args.limit),
            "runs": int(max(1, int(args.runs))),
        },
        "quality": quality,
        "latency": {
            "bounded_mean_ms": round(bounded_mean, 6),
            "bounded_median_ms": round(float(statistics.median(bounded_samples)), 6),
            "bounded_p95_ms": round(float(sorted(bounded_samples)[-1]), 6),
            "legacy_mean_ms": round(legacy_mean, 6),
            "legacy_median_ms": round(float(statistics.median(legacy_samples)), 6),
            "legacy_p95_ms": round(float(sorted(legacy_samples)[-1]), 6),
            "bounded_speedup_vs_legacy": round(
                legacy_mean / max(bounded_mean, 1e-12),
                6,
            ),
            "bounded_samples_ms": [round(float(value), 6) for value in bounded_samples],
            "legacy_samples_ms": [round(float(value), 6) for value in legacy_samples],
        },
        "source_window": source_window,
        "selection_budget": dict(source_window.get("selection_budget") or {}),
        "retired_path_comparison": {
            "old_policy": "score_all_retained_rollout_events_before_limit",
            "old_scored_event_count": int(
                legacy.get("candidate_count_before_rank", 0) or 0
            ),
            "bounded_scored_event_count": int(
                source_window.get("candidate_count_before_rank", 0) or 0
            ),
            "score_work_reduction": round(
                float(legacy.get("candidate_count_before_rank", 0) or 0)
                / max(1, int(source_window.get("candidate_count_before_rank", 0) or 0)),
                6,
            ),
        },
        "resource_behavior": {
            "process_rss_before_mib": None
            if rss_before is None
            else round(rss_before, 3),
            "process_rss_after_mib": None
            if rss_after is None
            else round(rss_after, 3),
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
    parser.add_argument("--retention-count", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=8)
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
