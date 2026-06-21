"""Benchmark bounded SNN language plasticity runtime-state snapshots.

The retired comparison in this file is diagnostic only. Production runtime
state must expose bounded CPU source windows and retained counts instead of
copying full transition-memory maps.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping as CollectionsMapping
import json
from pathlib import Path
import statistics
from threading import RLock
import time
import tracemalloc
from typing import Any, Callable, Mapping

import torch

from marulho.service.runtime_state import RuntimeState
from marulho.service.snn_language_plasticity_executor import (
    SNNLanguagePlasticityApplicationExecutor,
)
from marulho.service.transition_memory_source_window import (
    SNN_LANGUAGE_PLASTICITY_RUNTIME_TRANSITION_MEMORY_SOURCE_WINDOW_LIMIT,
)


class CountedMapping(CollectionsMapping):
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.items_iterated = 0

    def __getitem__(self, key: str) -> Any:
        return self._payload[key]

    def __iter__(self):
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)

    def items(self):
        for item in self._payload.items():
            self.items_iterated += 1
            yield item


def _entry_key(index: int) -> str:
    return f"{index}:{index + 1}"


def _seed_language_state(entry_count: int) -> dict[str, Any]:
    sparse_weights: dict[str, float] = {}
    provenance: dict[str, dict[str, Any]] = {}
    critical_period: dict[str, dict[str, Any]] = {}
    pruned_provenance: dict[str, dict[str, Any]] = {}
    for index in range(max(0, int(entry_count))):
        key = _entry_key(index)
        sparse_weights[key] = float((index % 17) + 1) / 100.0
        provenance[key] = {
            "source": "benchmark",
            "index": index,
            "readout_evidence_hash": f"readout-hash-{index}",
        }
        critical_period[key] = {
            "age": index % 64,
            "mature": False,
        }
        pruned_provenance[key] = {
            "source": "benchmark",
            "pruned_at": index,
        }
    return {
        "language_capacity": {
            "surface": "snn_language_capacity_state.v1",
            "language_neuron_count": max(128, int(entry_count) + 1),
            "sparse_edge_budget": max(256, int(entry_count)),
            "outgoing_fanout_budget": 32,
            "dynamic_capacity_enabled": True,
        },
        "sparse_transition_weights": CountedMapping(sparse_weights),
        "synapse_provenance_by_key": CountedMapping(provenance),
        "thought_newborn_neuron_critical_period_learning": {
            "learning_cycle_count": 1,
            "by_synapse": CountedMapping(critical_period),
        },
        "pruned_synapse_provenance_by_key": CountedMapping(pruned_provenance),
    }


def _source_reads(state: Mapping[str, Any]) -> int:
    critical = state.get("thought_newborn_neuron_critical_period_learning")
    critical_mapping = (
        critical.get("by_synapse")
        if isinstance(critical, Mapping)
        and isinstance(critical.get("by_synapse"), CountedMapping)
        else None
    )
    mappings = [
        state.get("sparse_transition_weights"),
        state.get("synapse_provenance_by_key"),
        critical_mapping,
        state.get("pruned_synapse_provenance_by_key"),
    ]
    return int(
        sum(
            int(item.items_iterated)
            for item in mappings
            if isinstance(item, CountedMapping)
        )
    )


def _active_snapshot(state: Mapping[str, Any]) -> dict[str, Any]:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: state,  # type: ignore[arg-type]
        save_checkpoint=lambda path: {"path": str(path or "benchmark.pt")},
        checkpoint_path=lambda: Path("benchmark.pt"),
        verify_checkpoint=lambda path: True,
    )
    snapshot = executor.snapshot()
    return {
        "snapshot": snapshot,
        "source_reads": _source_reads(state),
    }


def _mapping_items(source: Any) -> list[tuple[str, Any]]:
    if not isinstance(source, Mapping):
        return []
    return [(str(key), value) for key, value in source.items()]


def _retired_full_snapshot(state: Mapping[str, Any]) -> dict[str, Any]:
    critical = state.get("thought_newborn_neuron_critical_period_learning")
    critical_by_synapse = (
        critical.get("by_synapse")
        if isinstance(critical, Mapping)
        else {}
    )
    snapshot = {
        "surface": "retired_full_snn_language_plasticity_runtime_state_snapshot.v1",
        "sparse_transition_weights": dict(
            _mapping_items(state.get("sparse_transition_weights"))
        ),
        "synapse_provenance_by_key": dict(
            _mapping_items(state.get("synapse_provenance_by_key"))
        ),
        "newborn_neuron_critical_period_state_by_synapse": dict(
            _mapping_items(critical_by_synapse)
        ),
        "pruned_synapse_provenance_by_key": dict(
            _mapping_items(state.get("pruned_synapse_provenance_by_key"))
        ),
    }
    return {
        "snapshot": snapshot,
        "source_reads": _source_reads(state),
    }


def _cuda_memory() -> dict[str, int | bool | str | None]:
    available = bool(torch.cuda.is_available())
    if not available:
        return {
            "available": False,
            "device_name": None,
            "allocated_bytes": 0,
            "reserved_bytes": 0,
        }
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def _measure(
    factory: Callable[[], dict[str, Any]],
    fn: Callable[[Mapping[str, Any]], dict[str, Any]],
    *,
    runs: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    last_result: dict[str, Any] = {}
    for _ in range(max(1, int(runs))):
        state = factory()
        before_cuda = _cuda_memory()
        tracemalloc.start()
        started = time.perf_counter()
        result = fn(state)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        after_cuda = _cuda_memory()
        record = {
            "latency_ms": elapsed_ms,
            "python_traced_peak_mib": float(peak) / (1024.0 * 1024.0),
            "source_reads": int(result.get("source_reads", 0) or 0),
            "cuda_allocated_delta_bytes": int(after_cuda["allocated_bytes"])
            - int(before_cuda["allocated_bytes"]),
            "cuda_reserved_delta_bytes": int(after_cuda["reserved_bytes"])
            - int(before_cuda["reserved_bytes"]),
        }
        records.append(record)
        last_result = result
    return {
        "records": records,
        "latency_ms_mean": statistics.mean(item["latency_ms"] for item in records),
        "latency_ms_p95": sorted(item["latency_ms"] for item in records)[
            max(0, min(len(records) - 1, int(len(records) * 0.95) - 1))
        ],
        "python_traced_peak_mib_max": max(
            item["python_traced_peak_mib"] for item in records
        ),
        "source_reads_mean": statistics.mean(item["source_reads"] for item in records),
        "cuda_allocated_delta_bytes_max": max(
            item["cuda_allocated_delta_bytes"] for item in records
        ),
        "cuda_reserved_delta_bytes_max": max(
            item["cuda_reserved_delta_bytes"] for item in records
        ),
        "last_result": last_result,
    }


def run_benchmark(*, entry_count: int, runs: int) -> dict[str, Any]:
    factory = lambda: _seed_language_state(entry_count)
    active = _measure(factory, _active_snapshot, runs=runs)
    retired = _measure(factory, _retired_full_snapshot, runs=runs)
    active_snapshot = active["last_result"]["snapshot"]
    retired_snapshot = retired["last_result"]["snapshot"]
    source_window = active_snapshot["transition_memory_source_window"]
    source_limit = SNN_LANGUAGE_PLASTICITY_RUNTIME_TRANSITION_MEMORY_SOURCE_WINDOW_LIMIT
    quality_checks = {
        "retained_sparse_count_preserved": (
            active_snapshot["sparse_transition_weight_count"] == entry_count
        ),
        "retained_provenance_count_preserved": (
            active_snapshot["synapse_provenance_count"] == entry_count
        ),
        "bounded_sparse_payload": (
            len(active_snapshot["sparse_transition_weights"]) <= source_limit
        ),
        "bounded_provenance_payload": (
            len(active_snapshot["synapse_provenance_by_key"]) <= source_limit
        ),
        "source_window_reports_truncation": (
            source_window["source_payload_truncated"] is True
            and source_window["source_window_complete"] is False
        ),
        "retired_full_snapshot_materialized_all_rows": (
            len(retired_snapshot["sparse_transition_weights"]) == entry_count
            and len(retired_snapshot["synapse_provenance_by_key"]) == entry_count
        ),
        "cpu_archival_metadata": (
            source_window["archival_storage_device"] == "cpu"
            and source_window["gpu_resident_archival_metadata"] is False
        ),
        "no_live_tick_or_language_reasoning": (
            source_window["runs_live_tick"] is False
            and source_window["runs_every_token"] is False
            and source_window["language_reasoning"] is False
            and source_window["raw_text_payload_loaded"] is False
        ),
    }
    cuda = _cuda_memory()
    return {
        "artifact_kind": "marulho_plasticity_runtime_state_source_window_benchmark",
        "surface": "plasticity_runtime_state_source_window_benchmark.v1",
        "entry_count": int(entry_count),
        "runs": int(runs),
        "pass": all(quality_checks.values()),
        "quality_checks": quality_checks,
        "active_surface": active_snapshot["surface"],
        "active_source_window": source_window,
        "active": {
            key: value
            for key, value in active.items()
            if key not in {"records", "last_result"}
        },
        "retired_diagnostic": {
            key: value
            for key, value in retired.items()
            if key not in {"records", "last_result"}
        },
        "source_read_reduction": {
            "active_mean": active["source_reads_mean"],
            "retired_mean": retired["source_reads_mean"],
            "factor": (
                float(retired["source_reads_mean"]) / float(active["source_reads_mean"])
                if float(active["source_reads_mean"]) > 0.0
                else None
            ),
        },
        "latency_reduction": {
            "active_mean_ms": active["latency_ms_mean"],
            "retired_mean_ms": retired["latency_ms_mean"],
            "factor": (
                float(retired["latency_ms_mean"]) / float(active["latency_ms_mean"])
                if float(active["latency_ms_mean"]) > 0.0
                else None
            ),
        },
        "device_behavior": {
            "archival_storage_device": "cpu",
            "source_window_selection_device": "cpu",
            "gpu_used_for_archival_metadata": False,
            "cuda_available": bool(cuda["available"]),
            "cuda_device_name": cuda["device_name"],
            "active_cuda_allocated_delta_bytes_max": active[
                "cuda_allocated_delta_bytes_max"
            ],
            "active_cuda_reserved_delta_bytes_max": active[
                "cuda_reserved_delta_bytes_max"
            ],
        },
        "runtime_truth": {
            "runs_live_tick": False,
            "runs_every_token": False,
            "raw_text_payload_loaded": False,
            "hidden_language_reasoning": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "gpu_resident_archival_metadata": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry-count", type=int, default=65536)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run_benchmark(entry_count=args.entry_count, runs=args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
