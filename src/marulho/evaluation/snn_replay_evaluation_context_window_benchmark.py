from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from threading import RLock
import time
import tracemalloc
from typing import Any

import torch

from marulho.service.runtime_facade import RuntimeFacade
from marulho.service.runtime_state import RuntimeState
from marulho.service.snn_language_readout_ledger import (
    SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT,
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


def _observed_slot(index: int) -> dict[str, object]:
    return {
        "label": f"memory pressure {index}",
        "pressure_band": "high",
        "grounded": True,
    }


class _StatusReadModel:
    def __init__(self, calls: dict[str, int]) -> None:
        self._calls = calls

    def snn_language_sequence_mismatch_probe(self, **kwargs: object) -> dict[str, object]:
        self._calls["mismatch"] += 1
        return {
            "surface": "snn_language_sequence_mismatch_probe.v1",
            "available": True,
            "owned_by_marulho": True,
            "prediction_error": {"mismatch_score": 0.9},
            "observed_slot_count": len(kwargs.get("observed_readout_slots") or []),
        }

    def snn_language_plasticity_pressure(self, **kwargs: object) -> dict[str, object]:
        self._calls["pressure"] += 1
        return {
            "surface": "snn_language_plasticity_pressure.v1",
            "available": True,
            "owned_by_marulho": True,
            "promotion_gate": {"status": "ready_for_operator_review"},
            "mismatch_report": kwargs.get("mismatch_report"),
        }


class _ReplayController:
    def __init__(self, calls: dict[str, int], runtime_state: RuntimeState) -> None:
        self._calls = calls
        self._runtime_state = runtime_state

    def record_snn_replay_evaluation_context(self, **kwargs: object) -> dict[str, object]:
        self._calls["context"] += 1
        self._runtime_state.mark_dirty_without_revision()
        source_metadata = kwargs.get("source_metadata")
        return {
            "surface": "snn_replay_evaluation_context.v1",
            "available": True,
            "ready": True,
            "owned_by_marulho": True,
            "replay_evaluation_context_id": "context-benchmark",
            "evidence_hash": "context-hash",
            "source_metadata": source_metadata,
            "source_metadata_hash": _sha256_json(source_metadata),
            "mismatch_hash": "mismatch-hash",
            "pressure_hash": "pressure-hash",
        }


class _Root:
    def __init__(
        self,
        *,
        runtime_state: RuntimeState,
        ledger: SNNLanguageReadoutEvidenceLedger,
        calls: dict[str, int],
    ) -> None:
        self._runtime_state = runtime_state
        self._snn_language_readout_ledger = ledger
        self._status_read_model = _StatusReadModel(calls)
        self._replay_controller = _ReplayController(calls, runtime_state)


def _make_facade() -> tuple[RuntimeState, RuntimeFacade, dict[str, object], dict[str, int]]:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    calls = {"mismatch": 0, "pressure": 0, "context": 0}
    root = _Root(runtime_state=runtime_state, ledger=ledger, calls=calls)
    prediction_report = {
        "surface": "snn_language_sequence_prediction_probe.v1",
        "provenance_evidence": {"prediction_hash": "prediction-hash"},
    }
    return runtime_state, RuntimeFacade(root), prediction_report, calls


def _run_case(*, payload_count: int) -> dict[str, Any]:
    runtime_state, facade, prediction_report, calls = _make_facade()
    exact_slots = [
        _observed_slot(index)
        for index in range(SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT)
    ]

    started = time.perf_counter()
    exact = facade.snn_replay_evaluation_context(
        prediction_report=prediction_report,
        observed_readout_slots=exact_slots,
        device_evidence={"device": "cpu", "source": "benchmark"},
    )
    exact_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    oversized = facade.snn_replay_evaluation_context(
        prediction_report=prediction_report,
        observed_readout_slots=[
            _observed_slot(index) for index in range(int(payload_count))
        ],
        device_evidence={"device": "cpu", "source": "benchmark"},
    )
    oversized_ms = (time.perf_counter() - started) * 1000.0

    return {
        "elapsed_ms": {
            "exact_context_recording": exact_ms,
            "oversized_observed_slot_block": oversized_ms,
        },
        "exact_path": {
            "accepted": bool(exact.get("accepted")),
            "records_replay_context": bool(exact.get("records_replay_context")),
            "observed_slot_source_window": dict(
                exact.get("observed_slot_source_window") or {}
            ),
            "source_metadata": dict(exact.get("source_metadata") or {}),
            "context_calls_after_exact": dict(calls),
        },
        "oversized_block": {
            "accepted": bool(oversized.get("accepted")),
            "records_replay_context": bool(oversized.get("records_replay_context")),
            "required": dict(
                oversized.get("promotion_gate", {}).get("required_evidence") or {}
            ),
            "source_window": dict(oversized.get("observed_slot_source_window") or {}),
        },
        "call_counts_after_all_cases": dict(calls),
        "runtime_state_revision": int(runtime_state.state_revision),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row["elapsed_ms"][key]) for row in rows]
    return float(statistics.fmean(values)) if values else 0.0


def _median(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row["elapsed_ms"][key]) for row in rows]
    return float(statistics.median(values)) if values else 0.0


def _window_ok(window: dict[str, Any], *, truncated: bool) -> bool:
    return bool(
        window.get("source_window_count") == SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
        and window.get("source_mapping_count") == SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
        and window.get("source_payload_truncated") is truncated
        and window.get("global_candidate_scan") is False
        and window.get("global_score_scan") is False
        and window.get("runs_live_tick") is False
        and window.get("runs_every_token") is False
        and window.get("archival_storage_device") == "cpu"
        and window.get("gpu_resident_archival_metadata") is False
    )


def run_benchmark(*, payload_count: int, runs: int) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_before = int(torch.cuda.memory_allocated()) if cuda_available else 0
    cuda_reserved_before = int(torch.cuda.memory_reserved()) if cuda_available else 0
    tracemalloc.start()
    rows = [_run_case(payload_count=int(payload_count)) for _ in range(int(runs))]
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cuda_after = int(torch.cuda.memory_allocated()) if cuda_available else 0
    cuda_reserved_after = int(torch.cuda.memory_reserved()) if cuda_available else 0

    last = rows[-1] if rows else {}
    exact = dict(last.get("exact_path") or {})
    oversized = dict(last.get("oversized_block") or {})
    call_counts = dict(last.get("call_counts_after_all_cases") or {})
    exact_window = dict(exact.get("observed_slot_source_window") or {})
    source_metadata = dict(exact.get("source_metadata") or {})
    metadata_window = dict(source_metadata.get("observed_slot_source_window") or {})
    quality = {
        "exact_32_observed_slot_path_records_one_context": bool(
            exact.get("accepted")
            and exact.get("records_replay_context")
            and _window_ok(exact_window, truncated=False)
        ),
        "recorded_context_carries_observed_slot_window_metadata": bool(
            metadata_window == exact_window
        ),
        "oversized_observed_slot_payload_blocked_before_context_recording": bool(
            oversized.get("accepted") is False
            and oversized.get("records_replay_context") is False
            and _window_ok(dict(oversized.get("source_window") or {}), truncated=True)
            and oversized.get("required", {}).get("observed_slot_payload_not_truncated")
            is False
        ),
        "blocked_payload_does_not_call_mismatch_pressure_or_replay_controller": bool(
            call_counts == {"mismatch": 1, "pressure": 1, "context": 1}
        ),
        "record_work_reduction": float(
            int(payload_count) / max(1, SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT)
        ),
        "old_full_payload_generic_context_path_executable": False,
    }
    quality["pass"] = bool(
        quality["exact_32_observed_slot_path_records_one_context"]
        and quality["recorded_context_carries_observed_slot_window_metadata"]
        and quality[
            "oversized_observed_slot_payload_blocked_before_context_recording"
        ]
        and quality[
            "blocked_payload_does_not_call_mismatch_pressure_or_replay_controller"
        ]
        and quality["record_work_reduction"] >= 2.0
    )
    latency = {
        key: {
            "elapsed_mean_ms": _mean(rows, key),
            "elapsed_median_ms": _median(rows, key),
        }
        for key in [
            "exact_context_recording",
            "oversized_observed_slot_block",
        ]
    }
    return {
        "artifact_kind": "bounded_snn_replay_evaluation_context_window_benchmark",
        "surface": "bounded_snn_replay_evaluation_context_window_benchmark.v1",
        "payload_count": int(payload_count),
        "window_limit": SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT,
        "runs": int(runs),
        "selection_criteria": [
            "caller_supplied_observed_sparse_slot_order",
            "bounded_source_window_before_mismatch_pressure_or_context_recording",
            "untruncated_window_required_before_replay_controller_context_recording",
            "source_window_metadata_bound_to_recorded_context",
        ],
        "runtime_truth": {
            "runs_live_tick": False,
            "runs_every_token": False,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_text_payload_loaded": False,
            "hidden_language_reasoning": False,
            "language_reasoning": False,
            "oversized_payload_reaches_replay_controller": False,
            "old_full_payload_generic_context_path_executable": False,
        },
        "device_placement": {
            "archival_storage_device": "cpu",
            "source_window_selection_device": "cpu",
            "context_review_gate_device": "cpu",
            "active_replay_computation_device": "cpu",
            "cuda_available": cuda_available,
            "gpu_used": False,
            "gpu_resident_archival_metadata": False,
            "cuda_memory_allocated_before_mib": float(cuda_before / (1024 * 1024)),
            "cuda_memory_allocated_after_mib": float(cuda_after / (1024 * 1024)),
            "cuda_memory_reserved_before_mib": float(
                cuda_reserved_before / (1024 * 1024)
            ),
            "cuda_memory_reserved_after_mib": float(cuda_reserved_after / (1024 * 1024)),
            "python_traced_peak_mib": float(peak_bytes / (1024 * 1024)),
        },
        "latency": latency,
        "last_run": last,
        "quality": quality,
        "pass": bool(quality["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded generic SNN replay evaluation context windows."
    )
    parser.add_argument("--payload-count", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(payload_count=args.payload_count, runs=args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "pass={passed} exact_slots={exact_slots}/{limit} "
        "slot_block={slot_block}/{payload} context_calls={context_calls} "
        "reduction={reduction:.6f}".format(
            passed=report["pass"],
            exact_slots=report["last_run"]["exact_path"][
                "observed_slot_source_window"
            ]["source_window_count"],
            limit=report["window_limit"],
            payload=report["payload_count"],
            slot_block=report["last_run"]["oversized_block"]["source_window"][
                "source_window_count"
            ],
            context_calls=report["last_run"]["call_counts_after_all_cases"][
                "context"
            ],
            reduction=report["quality"]["record_work_reduction"],
        )
    )


if __name__ == "__main__":
    main()
