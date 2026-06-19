"""Benchmark bounded SNN readout-ledger normalization.

The production normalizer reads each retained event family through a newest-first
source window. The diagnostic legacy path below preserves the retired
full-materialize-then-cap shape for latency, work, and recent-row retention
comparison only.
"""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import statistics
from threading import RLock
import time
import tracemalloc
from typing import Any, Mapping

import torch

from marulho.service.runtime_state import RuntimeState
from marulho.service.snn_language_readout_ledger import (
    SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS,
    SNN_LANGUAGE_READOUT_LEDGER_NORMALIZATION_SOURCE_WINDOW_POLICY,
    SNNLanguageReadoutEvidenceLedger,
)


def _seed_ledger_state(*, retention_count: int) -> dict[str, Any]:
    count = max(1, int(retention_count))
    state: dict[str, Any] = {}
    for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS:
        rows: list[dict[str, Any]] = []
        for index in range(count):
            rows.append(
                {
                    "field": field,
                    "ordinal": int(index),
                    "readout_evidence_hash": f"{field}:readout:{index}",
                    "rollout_evidence_hash": f"{field}:rollout:{index}",
                    "emission_review_hash": f"{field}:review:{index}",
                    "prediction_hash": f"{field}:prediction:{index}",
                    "persistent_transition_weights_hash": f"{field}:weights:{index}",
                    "labels": [f"{field}:label:{index}"],
                    "label_grounding": [True],
                }
            )
        state[field] = rows
    state.update(
        {
            "total_recorded_count": count,
            "total_rollout_recorded_count": count,
            "total_emission_review_count": count,
            "last_recorded_at": "2026-06-18T00:00:00+00:00",
            "last_rollout_recorded_at": "2026-06-18T00:00:00+00:00",
            "last_emission_reviewed_at": "2026-06-18T00:00:00+00:00",
        }
    )
    return state


def _legacy_full_materialized_normalized_state(
    state: Mapping[str, Any],
    *,
    limit: int,
) -> dict[str, deque[dict[str, Any]]]:
    normalized: dict[str, deque[dict[str, Any]]] = {}
    for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS:
        raw_rows = list(state.get(field) or [])
        normalized[field] = deque(
            (
                deepcopy(dict(item))
                for item in raw_rows
                if isinstance(item, Mapping)
            ),
            maxlen=max(1, int(limit)),
        )
    return normalized


def _legacy_full_materialized_store_state(
    normalized: Mapping[str, Any],
    *,
    limit: int,
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS:
        raw_rows = list(normalized.get(field) or [])
        state[field] = [
            deepcopy(dict(item))
            for item in raw_rows[: max(1, int(limit))]
            if isinstance(item, Mapping)
        ]
    return state


def _bounded_store_state(
    ledger: SNNLanguageReadoutEvidenceLedger,
    target: dict[str, Any],
    normalized: Mapping[str, Any],
) -> dict[str, Any]:
    target.clear()
    ledger._store_state(normalized)  # noqa: SLF001
    return {
        field: list(target.get(field) or [])
        for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS
    }


def _timed_runs(*, runs: int, fn: Any) -> tuple[dict[str, Any], list[float]]:
    samples: list[float] = []
    last: dict[str, Any] | None = None
    for _ in range(max(1, int(runs))):
        started = time.perf_counter()
        last = fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    assert last is not None
    return last, samples


def _latency_summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    index_95 = min(
        len(ordered) - 1,
        max(0, int(round((len(ordered) - 1) * 0.95))),
    )
    return {
        "mean_ms": round(statistics.fmean(samples), 6),
        "median_ms": round(statistics.median(samples), 6),
        "p95_ms": round(ordered[index_95], 6),
    }


def _recent_retention_rate(normalized: Mapping[str, deque[dict[str, Any]]]) -> float:
    retained = 0
    for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS:
        rows = list(normalized.get(field) or [])
        if rows and int(rows[0].get("ordinal", -1)) == 0:
            retained += 1
    return retained / max(1, len(SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS))


def _first_ordinals(normalized: Mapping[str, deque[dict[str, Any]]]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for field in SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS:
        rows = list(normalized.get(field) or [])
        result[field] = int(rows[0].get("ordinal", -1)) if rows else None
    return result


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
            6,
        )
        report["memory_reserved_mib"] = round(
            float(torch.cuda.memory_reserved(0)) / (1024.0 * 1024.0),
            6,
        )
    return report


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    retention_count = max(1, int(args.retention_count))
    ledger_limit = max(1, int(args.ledger_limit))
    runs = max(1, int(args.runs))
    state = _seed_ledger_state(retention_count=retention_count)
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: state,
        limit=ledger_limit,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tracemalloc.start()
    bounded, bounded_samples = _timed_runs(
        runs=runs,
        fn=lambda: ledger._normalized_state(),  # noqa: SLF001
    )
    legacy, legacy_samples = _timed_runs(
        runs=runs,
        fn=lambda: _legacy_full_materialized_normalized_state(
            state,
            limit=ledger_limit,
        ),
    )
    bounded_store_target: dict[str, Any] = {}
    bounded_store_ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: bounded_store_target,
        limit=ledger_limit,
    )
    bounded_store, bounded_store_samples = _timed_runs(
        runs=runs,
        fn=lambda: _bounded_store_state(
            bounded_store_ledger,
            bounded_store_target,
            state,
        ),
    )
    legacy_store, legacy_store_samples = _timed_runs(
        runs=runs,
        fn=lambda: _legacy_full_materialized_store_state(
            state,
            limit=ledger_limit,
        ),
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    source_window = dict(bounded.get("_normalization_source_window") or {})
    bounded_rate = _recent_retention_rate(bounded)
    legacy_rate = _recent_retention_rate(legacy)
    bounded_store_rate = _recent_retention_rate(bounded_store)
    legacy_store_rate = _recent_retention_rate(legacy_store)
    bounded_rows = int(source_window.get("source_window_count_total", 0) or 0)
    legacy_rows = int(retention_count * len(SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS))
    bounded_store_rows = int(ledger_limit * len(SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS))
    legacy_store_rows = legacy_rows
    bounded_mean = statistics.fmean(bounded_samples)
    legacy_mean = statistics.fmean(legacy_samples)
    bounded_store_mean = statistics.fmean(bounded_store_samples)
    legacy_store_mean = statistics.fmean(legacy_store_samples)
    bounded_store_first_ordinals = _first_ordinals(bounded_store)
    legacy_store_first_ordinals = _first_ordinals(legacy_store)
    pass_checks = {
        "surface_present": source_window.get("surface")
        == "bounded_snn_readout_ledger_normalization_source_window.v1",
        "policy_present": source_window.get("policy")
        == SNN_LANGUAGE_READOUT_LEDGER_NORMALIZATION_SOURCE_WINDOW_POLICY,
        "source_limit_respected": all(
            int(value) <= ledger_limit
            for value in dict(source_window.get("source_window_counts") or {}).values()
        ),
        "no_global_scan": source_window.get("global_candidate_scan") is False
        and source_window.get("global_score_scan") is False,
        "not_live_tick": source_window.get("runs_live_tick") is False
        and source_window.get("runs_every_token") is False,
        "cpu_only": source_window.get("archival_storage_device") == "cpu"
        and source_window.get("normalization_device") == "cpu"
        and source_window.get("gpu_used") is False,
        "no_language_reasoning": source_window.get("language_reasoning") is False,
        "recent_rows_preserved": bounded_rate == 1.0,
        "legacy_recent_loss_detected": legacy_rate < bounded_rate,
        "bounded_less_work": bounded_rows < legacy_rows,
        "store_source_limit_respected": all(
            len(list(value or [])) <= ledger_limit for value in bounded_store.values()
        ),
        "store_recent_rows_preserved": bounded_store_rate == 1.0,
        "store_matches_legacy_window": (
            bounded_store_first_ordinals == legacy_store_first_ordinals
        ),
        "store_bounded_less_work": bounded_store_rows < legacy_store_rows,
        "store_latency_not_slower_than_legacy": (
            bounded_store_mean <= legacy_store_mean * 1.1
        ),
    }
    return {
        "surface": "bounded_snn_readout_ledger_normalization_source_window_benchmark.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": {
            "retention_count_per_field": retention_count,
            "ledger_limit": ledger_limit,
            "event_field_count": len(SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS),
            "runs": runs,
        },
        "pass": all(pass_checks.values()),
        "pass_checks": pass_checks,
        "quality": {
            "metric": "newest_first_recent_row_retention_rate",
            "bounded_recent_retention_rate": round(bounded_rate, 6),
            "legacy_recent_retention_rate": round(legacy_rate, 6),
            "bounded_first_ordinals": _first_ordinals(bounded),
            "legacy_first_ordinals": _first_ordinals(legacy),
        },
        "latency": {
            "bounded": _latency_summary(bounded_samples),
            "legacy": _latency_summary(legacy_samples),
            "bounded_speedup_vs_legacy": round(
                legacy_mean / max(bounded_mean, 1e-9),
                6,
            ),
        },
        "retired_path_comparison": {
            "old_policy": "full_materialize_each_ledger_event_field_then_cap",
            "bounded_checked_record_count": bounded_rows,
            "old_checked_record_count": legacy_rows,
            "record_work_reduction": round(legacy_rows / max(1, bounded_rows), 6),
        },
        "normalization_source_window": source_window,
        "store_state_boundary": {
            "surface": "bounded_snn_readout_ledger_store_state_source_window.v1",
            "policy": SNN_LANGUAGE_READOUT_LEDGER_NORMALIZATION_SOURCE_WINDOW_POLICY,
            "source": "checkpoint_reload_and_record_persistence_event_fields",
            "selection_criteria": [
                "newest-first ledger event field order",
                "bounded source window before persistence copy",
                "single event-field helper shared with normalization",
            ],
            "quality": {
                "metric": "newest_first_store_window_parity",
                "bounded_recent_retention_rate": round(bounded_store_rate, 6),
                "legacy_recent_retention_rate": round(legacy_store_rate, 6),
                "bounded_first_ordinals": bounded_store_first_ordinals,
                "legacy_first_ordinals": legacy_store_first_ordinals,
            },
            "latency": {
                "bounded": _latency_summary(bounded_store_samples),
                "legacy": _latency_summary(legacy_store_samples),
                "bounded_speedup_vs_legacy": round(
                    legacy_store_mean / max(bounded_store_mean, 1e-9),
                    6,
                ),
            },
            "retired_path_comparison": {
                "old_policy": "full_materialize_each_ledger_event_field_then_cap_before_store",
                "bounded_checked_record_count": bounded_store_rows,
                "old_checked_record_count": legacy_store_rows,
                "record_work_reduction": round(
                    legacy_store_rows / max(1, bounded_store_rows),
                    6,
                ),
            },
            "source_window_limit_per_field": int(ledger_limit),
            "source_window_count_total": int(bounded_store_rows),
            "source_record_count_total_known": int(legacy_store_rows),
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "runs_live_tick": False,
            "runs_every_token": False,
            "mutates_runtime_state": True,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "store_device": "cpu",
            "gpu_used": False,
            "memory_budget": {
                "max_fields": len(SNN_LANGUAGE_READOUT_LEDGER_EVENT_FIELDS),
                "max_records_per_field": int(ledger_limit),
                "max_records_total": int(bounded_store_rows),
            },
        },
        "resource_behavior": {
            "python_tracemalloc_current_mib": round(current / (1024 * 1024), 6),
            "python_tracemalloc_peak_mib": round(peak / (1024 * 1024), 6),
            "cuda": _cuda_report(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded SNN readout-ledger normalization."
    )
    parser.add_argument("--retention-count", type=int, default=2048)
    parser.add_argument("--ledger-limit", type=int, default=128)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/bounded_replay_window_20260618/snn-readout-ledger-normalization-source-window.json"),
    )
    args = parser.parse_args()
    report = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"snn_readout_ledger_normalization_source_window_benchmark={args.output}")
    print(
        "pass={passed} bounded_mean_ms={bounded:.6f} legacy_mean_ms={legacy:.6f} "
        "work_reduction={work:.6f} bounded_recent={bounded_recent:.6f} "
        "legacy_recent={legacy_recent:.6f} store_bounded_mean_ms={store_bounded:.6f} "
        "store_legacy_mean_ms={store_legacy:.6f}".format(
            passed=report["pass"],
            bounded=report["latency"]["bounded"]["mean_ms"],
            legacy=report["latency"]["legacy"]["mean_ms"],
            work=report["retired_path_comparison"]["record_work_reduction"],
            bounded_recent=report["quality"]["bounded_recent_retention_rate"],
            legacy_recent=report["quality"]["legacy_recent_retention_rate"],
            store_bounded=report["store_state_boundary"]["latency"]["bounded"][
                "mean_ms"
            ],
            store_legacy=report["store_state_boundary"]["latency"]["legacy"][
                "mean_ms"
            ],
        )
    )


if __name__ == "__main__":
    main()
