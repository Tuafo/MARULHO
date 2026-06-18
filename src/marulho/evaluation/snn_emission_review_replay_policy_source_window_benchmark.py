"""Benchmark bounded SNN emission-review replay policy source windows."""

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
    SNN_EMISSION_REVIEW_REPLAY_POLICY_SOURCE_WINDOW_LIMIT,
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


def _readout_event(index: int, *, label: str, weights_hash: str) -> dict[str, Any]:
    evidence_hash = _sha256_json(
        {
            "readout": int(index),
            "label": str(label),
            "weights_hash": str(weights_hash),
        }
    )
    return {
        "surface": "snn_language_readout_evidence_ledger_event.v1",
        "readout_evidence_id": f"snn-readout-evidence:{evidence_hash[:16]}",
        "readout_evidence_hash": evidence_hash,
        "prediction_hash": f"prediction-{index}",
        "transition_memory_evaluation_hash": f"evaluation-{index}",
        "persistent_transition_weights_hash": weights_hash,
        "labels": [label],
        "label_grounding": [True],
        "state_revision": 11,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def _review_event(index: int, *, label: str, weights_hash: str) -> dict[str, Any]:
    review_hash = _sha256_json(
        {
            "review": int(index),
            "label": str(label),
            "weights_hash": str(weights_hash),
        }
    )
    return {
        "surface": "snn_language_readout_emission_review_event.v1",
        "emission_review_id": f"snn-readout-emission-review:{review_hash[:16]}",
        "emission_review_hash": review_hash,
        "emission_hash": _sha256_json({"emission": int(index)}),
        "trajectory_hash": _sha256_json({"trajectory": int(index)}),
        "prediction_hash": f"prediction-{index}",
        "transition_memory_evaluation_hash": f"evaluation-{index}",
        "persistent_transition_weights_hash": weights_hash,
        "text": f"benchmark reviewed display text {index}",
        "labels": [label],
        "term_count": 1,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "state_revision": 11,
    }


def _seed_events(
    *,
    retention_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    count = max(
        SNN_EMISSION_REVIEW_REPLAY_POLICY_SOURCE_WINDOW_LIMIT + 1,
        int(retention_count),
    )
    readouts: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    for source_index in range(count):
        if source_index < 4:
            label = "recent-high-signal-emission-review"
            weights_hash = "recent-high-signal-emission-review-weights"
        elif source_index >= count - 4:
            label = "old-retained-emission-review"
            weights_hash = "old-retained-emission-review-weights"
        else:
            label = f"emission-review-distractor-{source_index}"
            weights_hash = f"emission-review-weights-{source_index % 31}"
        readouts.append(
            _readout_event(source_index, label=label, weights_hash=weights_hash)
        )
        reviews.append(
            _review_event(source_index, label=label, weights_hash=weights_hash)
        )
    return readouts, reviews


def _legacy_full_retained_policy(
    review_events: Sequence[Mapping[str, Any]],
    readout_events: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    """Diagnostic-only copy of the retired all-retained match policy shape."""

    retained_reviews = [
        deepcopy(dict(item)) for item in review_events if isinstance(item, Mapping)
    ]
    retained_readouts = [
        deepcopy(dict(item)) for item in readout_events if isinstance(item, Mapping)
    ]
    readout_by_binding: dict[
        tuple[str, str, str, tuple[str, ...]],
        dict[str, Any],
    ] = {}
    for event in retained_readouts:
        labels = tuple(str(value) for value in list(event.get("labels") or []))
        key = (
            str(event.get("prediction_hash") or ""),
            str(event.get("transition_memory_evaluation_hash") or ""),
            str(event.get("persistent_transition_weights_hash") or ""),
            labels,
        )
        if all(key[:3]) and labels:
            readout_by_binding.setdefault(key, dict(event))

    candidates: list[dict[str, Any]] = []
    unmatched_reviews: list[dict[str, Any]] = []
    for index, review in enumerate(retained_reviews):
        labels = tuple(str(value) for value in list(review.get("labels") or []))
        key = (
            str(review.get("prediction_hash") or ""),
            str(review.get("transition_memory_evaluation_hash") or ""),
            str(review.get("persistent_transition_weights_hash") or ""),
            labels,
        )
        readout = readout_by_binding.get(key)
        if readout is None:
            unmatched_reviews.append(
                {
                    "history_index": index,
                    "emission_review_hash": review.get("emission_review_hash"),
                    "reason": "missing_matching_internal_readout_evidence",
                }
            )
            continue
        label_grounding = [
            bool(value) for value in list(readout.get("label_grounding") or [])
        ]
        grounded = bool(label_grounding) and all(label_grounding)
        candidate_material = {
            "emission_review_hash": review.get("emission_review_hash"),
            "emission_hash": review.get("emission_hash"),
            "readout_evidence_hash": readout.get("readout_evidence_hash"),
            "prediction_hash": review.get("prediction_hash"),
            "transition_memory_evaluation_hash": review.get(
                "transition_memory_evaluation_hash"
            ),
            "persistent_transition_weights_hash": review.get(
                "persistent_transition_weights_hash"
            ),
            "labels": list(labels),
            "grounded": grounded,
        }
        candidates.append(
            {
                "rank": len(candidates) + 1,
                "emission_review_hash": review.get("emission_review_hash"),
                "emission_hash": review.get("emission_hash"),
                "readout_evidence_hash": readout.get("readout_evidence_hash"),
                "prediction_hash": review.get("prediction_hash"),
                "transition_memory_evaluation_hash": review.get(
                    "transition_memory_evaluation_hash"
                ),
                "persistent_transition_weights_hash": review.get(
                    "persistent_transition_weights_hash"
                ),
                "label_hash": _sha256_json(list(labels)),
                "all_labels_grounded": grounded,
                "candidate_hash": _sha256_json(candidate_material),
                "eligible_for_replay_evaluation_policy_review": grounded,
            }
        )
    selected = candidates[: max(0, int(limit))]
    readout_by_hash = {
        str(event.get("readout_evidence_hash") or ""): dict(event)
        for event in retained_readouts
        if str(event.get("readout_evidence_hash") or "")
    }
    seeds: list[dict[str, Any]] = []
    for candidate in selected:
        readout_hash = str(candidate.get("readout_evidence_hash") or "")
        readout = readout_by_hash.get(readout_hash, {})
        ledger_match = bool(readout) and all(
            str(candidate.get(key) or "") == str(readout.get(key) or "")
            for key in (
                "prediction_hash",
                "transition_memory_evaluation_hash",
                "persistent_transition_weights_hash",
            )
        )
        seeds.append(
            {
                "readout_evidence_hash": readout_hash,
                "prediction_hash": candidate.get("prediction_hash"),
                "internal_readout_ledger_match": ledger_match,
            }
        )
    return {
        "surface": "diagnostic_legacy_snn_emission_review_full_retained_policy_and_design.v1",
        "candidate_count_before_limit": int(len(candidates)),
        "unmatched_emission_review_count": int(len(unmatched_reviews)),
        "candidate_count": int(len(selected)),
        "selected_seed_count": int(len(seeds)),
        "candidates": selected,
        "selected_replay_context_seeds": seeds,
        "retained_review_event_count": int(len(retained_reviews)),
        "retained_readout_event_count": int(len(retained_readouts)),
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


def _timed_runs(
    *,
    runs: int,
    fn: Any,
) -> tuple[dict[str, Any], list[float]]:
    samples: list[float] = []
    last: dict[str, Any] | None = None
    for _ in range(max(1, int(runs))):
        started = time.perf_counter()
        last = fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    assert last is not None
    return last, samples


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    readouts, reviews = _seed_events(retention_count=int(args.retention_count))
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, Any] = {
        "events": deepcopy(readouts),
        "emission_review_events": deepcopy(reviews),
        "total_recorded_count": int(len(readouts)),
        "total_emission_review_count": int(len(reviews)),
        "last_recorded_at": readouts[0]["recorded_at"] if readouts else None,
        "last_emission_reviewed_at": reviews[0]["reviewed_at"] if reviews else None,
    }
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
        limit=int(len(reviews)),
    )

    def _bounded_path() -> dict[str, Any]:
        policy = ledger.emission_review_replay_evaluation_policy(limit=int(args.limit))
        design = ledger.emission_review_replay_evaluation_design(
            policy,
            design_policy={
                "max_candidates": int(args.limit),
                "min_ready_candidates": 1,
            },
            device_evidence={
                "device": "cpu",
                "source": "snn_emission_review_replay_policy_source_window_benchmark",
            },
        )
        return {"policy": policy, "design": design}

    rss_before = _process_rss_mb()
    bounded, bounded_samples = _timed_runs(
        runs=int(args.runs),
        fn=_bounded_path,
    )
    legacy, legacy_samples = _timed_runs(
        runs=int(args.runs),
        fn=lambda: _legacy_full_retained_policy(
            ledger_state["emission_review_events"],
            ledger_state["events"],
            limit=int(args.limit),
        ),
    )
    rss_after = _process_rss_mb()
    tracemalloc.start()
    _ = _bounded_path()
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    policy = dict(bounded.get("policy") or {})
    design = dict(bounded.get("design") or {})
    policy_source_window = dict(policy.get("source_window") or {})
    design_source_window = dict(design.get("source_window") or {})
    bounded_candidates = [
        dict(item)
        for item in list(policy.get("candidates") or [])
        if isinstance(item, Mapping)
    ]
    legacy_candidates = [
        dict(item)
        for item in list(legacy.get("candidates") or [])
        if isinstance(item, Mapping)
    ]
    bounded_top_hash = (
        bounded_candidates[0].get("emission_review_hash")
        if bounded_candidates
        else None
    )
    legacy_top_hash = (
        legacy_candidates[0].get("emission_review_hash")
        if legacy_candidates
        else None
    )
    selected_prediction_hashes = [
        str(item.get("prediction_hash") or "") for item in bounded_candidates
    ]
    bounded_mean = float(statistics.mean(bounded_samples))
    legacy_mean = float(statistics.mean(legacy_samples))
    quality = {
        "bounded_top_matches_legacy_top": bounded_top_hash == legacy_top_hash,
        "selected_candidate_count": int(len(bounded_candidates)),
        "recent_high_signal_selected": "prediction-0" in selected_prediction_hashes,
        "old_retained_outside_window_absent": f"prediction-{len(readouts) - 1}"
        not in selected_prediction_hashes,
        "bounded_selected_prediction_hashes": selected_prediction_hashes,
        "legacy_selected_prediction_hashes": [
            str(item.get("prediction_hash") or "") for item in legacy_candidates
        ],
    }
    policy_required = (
        policy.get("promotion_gate", {}).get("required_evidence", {})
        if isinstance(policy.get("promotion_gate"), Mapping)
        else {}
    )
    design_required = (
        design.get("promotion_gate", {}).get("required_evidence", {})
        if isinstance(design.get("promotion_gate"), Mapping)
        else {}
    )
    pass_checks = {
        "policy_surface_present": policy.get("surface")
        == "snn_language_readout_emission_replay_evaluation_policy.v1",
        "design_surface_present": design.get("surface")
        == "snn_language_readout_emission_replay_evaluation_design.v1",
        "policy_source_window_present": policy_source_window.get("surface")
        == "bounded_snn_emission_review_replay_policy_source_window.v1",
        "design_source_window_present": design_source_window.get("surface")
        == "bounded_snn_emission_review_replay_policy_source_window.v1",
        "source_window_limit_respected": int(
            policy_source_window.get("emission_review_event_window_count", 0) or 0
        )
        <= SNN_EMISSION_REVIEW_REPLAY_POLICY_SOURCE_WINDOW_LIMIT
        and int(
            policy_source_window.get("internal_readout_event_window_count", 0) or 0
        )
        <= SNN_EMISSION_REVIEW_REPLAY_POLICY_SOURCE_WINDOW_LIMIT,
        "bounded_below_retention": int(
            policy_source_window.get("emission_review_event_window_count", 0) or 0
        )
        < int(
            policy_source_window.get("emission_review_event_retention_count", 0) or 0
        )
        and int(
            policy_source_window.get("internal_readout_event_window_count", 0) or 0
        )
        < int(
            policy_source_window.get("internal_readout_event_retention_count", 0) or 0
        ),
        "matched_only_source_window": int(
            policy_source_window.get("candidate_count_before_limit", 0) or 0
        )
        == int(
            policy_source_window.get("emission_review_event_window_count", 0) or 0
        ),
        "top_quality_matches_legacy": bool(quality["bounded_top_matches_legacy_top"]),
        "recent_high_signal_selected": bool(quality["recent_high_signal_selected"]),
        "old_retained_outside_window_absent": bool(
            quality["old_retained_outside_window_absent"]
        ),
        "policy_no_global_scan": policy_source_window.get("global_candidate_scan")
        is False
        and policy_source_window.get("global_score_scan") is False,
        "design_no_global_scan": design_source_window.get("global_candidate_scan")
        is False
        and design_source_window.get("global_score_scan") is False,
        "cpu_only": policy_source_window.get("archival_storage_device") == "cpu"
        and policy_source_window.get("score_device") == "cpu"
        and policy_source_window.get("gpu_used") is False
        and design_source_window.get("archival_storage_device") == "cpu"
        and design_source_window.get("score_device") == "cpu"
        and design_source_window.get("gpu_used") is False,
        "not_live_tick": policy_source_window.get("runs_live_tick") is False
        and policy_source_window.get("runs_every_token") is False
        and design_source_window.get("runs_live_tick") is False
        and design_source_window.get("runs_every_token") is False,
        "no_language_reasoning": policy_source_window.get("language_reasoning")
        is False
        and policy_source_window.get("raw_text_payload_loaded") is False
        and design_source_window.get("language_reasoning") is False
        and design_source_window.get("raw_text_payload_loaded") is False,
        "promotion_gate_policy_window_bounded": bool(
            policy_required.get("source_window_bounded")
        ),
        "promotion_gate_design_window_bounded": bool(
            design_required.get("design_source_window_bounded")
        ),
    }
    return {
        "surface": "bounded_snn_emission_review_replay_policy_source_window_benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "pass": all(pass_checks.values()),
        "pass_checks": pass_checks,
        "input": {
            "retention_count": int(len(reviews)),
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
        "policy_source_window": policy_source_window,
        "design_source_window": design_source_window,
        "selection_budget": dict(policy_source_window.get("selection_budget") or {}),
        "retired_path_comparison": {
            "old_policy": "match_all_retained_emission_reviews_and_readouts_then_verify_all_readouts_in_design",
            "old_review_event_count": int(
                legacy.get("retained_review_event_count", 0) or 0
            ),
            "old_readout_event_count": int(
                legacy.get("retained_readout_event_count", 0) or 0
            ),
            "old_matched_event_count": int(
                legacy.get("retained_review_event_count", 0) or 0
            )
            + int(legacy.get("retained_readout_event_count", 0) or 0),
            "bounded_review_event_count": int(
                policy_source_window.get("emission_review_event_window_count", 0) or 0
            ),
            "bounded_readout_event_count": int(
                policy_source_window.get("internal_readout_event_window_count", 0) or 0
            ),
            "bounded_matched_event_count": int(
                policy_source_window.get("emission_review_event_window_count", 0) or 0
            )
            + int(
                policy_source_window.get("internal_readout_event_window_count", 0)
                or 0
            ),
            "match_work_reduction": round(
                (
                    float(legacy.get("retained_review_event_count", 0) or 0)
                    + float(legacy.get("retained_readout_event_count", 0) or 0)
                )
                / max(
                    1,
                    int(
                        policy_source_window.get(
                            "emission_review_event_window_count",
                            0,
                        )
                        or 0
                    )
                    + int(
                        policy_source_window.get(
                            "internal_readout_event_window_count",
                            0,
                        )
                        or 0
                    ),
                ),
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
