"""Benchmark bounded transition-memory status source windows.

The retired comparison in this file is diagnostic only. Production status
projection must use bounded CPU source windows and must block exact readiness
claims when those windows are truncated.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Mapping as CollectionsMapping
import hashlib
import json
from pathlib import Path
import statistics
from threading import RLock
import time
import tracemalloc
from typing import Any, Callable, Mapping

import torch

from marulho.config.model_config import MarulhoConfig
from marulho.service.runtime_state import RuntimeState
from marulho.service.status_read_model import (
    SNN_STATUS_TRANSITION_MEMORY_SOURCE_WINDOW_LIMIT,
    SNN_STATUS_TRANSITION_MEMORY_SOURCE_WINDOW_POLICY,
    StatusReadModel,
)
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


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


def _build_read_model(
    language_state_fn: Callable[[], Mapping[str, Any]],
) -> StatusReadModel:
    config = MarulhoConfig(
        n_columns=4,
        column_latent_dim=8,
        bootstrap_tokens=0,
        memory_capacity=64,
        eta_competitive=0.05,
        eta_decay=0.0,
        input_weight_blend=0.0,
        enable_context_layer=True,
        enable_binding_layer=True,
        device="cpu",
    )
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    return StatusReadModel(
        lock=lock,
        runtime_state=runtime_state,
        trainer=trainer,
        trace_history=deque(maxlen=200),
        metadata={},
        checkpoint_path_str="benchmark-status.pt",
        trace_dir_str="benchmark-traces",
        concept_store_snapshot_fn=lambda: {"top_concepts": [], "total_concepts": 0},
        brain_runtime_snapshot_fn=lambda: {
            "configured": False,
            "running": False,
            "source_bank": [],
            "living_loop": {},
        },
        sensory_preview_history=deque(maxlen=8),
        architecture_snapshot_fn=lambda: {
            "model_name": "Terminus",
            "core_name": "GPCSN",
            "version": "current",
            "family": "subcortex_runtime",
            "layers": [],
            "config": {},
        },
        animation_snapshot_fn=lambda: {},
        language_plasticity_state_fn=language_state_fn,
    )


def _transition_key(index: int) -> str:
    source = int(index // 32)
    target = int(index % 32)
    return f"{source}:{target}"


def _seed_language_state(entry_count: int) -> dict[str, Any]:
    sparse_weights: dict[str, float] = {}
    provenance: dict[str, dict[str, Any]] = {}
    tensor = torch.zeros((128, 128), dtype=torch.float32)
    for index in range(max(0, int(entry_count))):
        key = _transition_key(index)
        value = float((index % 17) + 1) / 100.0
        sparse_weights[key] = value
        source, target = (int(part) for part in key.split(":", maxsplit=1))
        tensor[source, target] = value
        provenance[key] = {
            "provenance_type": "replay_regeneration",
            "source_metadata_hash": f"source-metadata-hash-{index}",
            "emission_lineage": {
                "emission_hash": f"emission-hash-{index}",
                "readout_evidence_hash": f"readout-hash-{index}",
                "prediction_hash": f"prediction-hash-{index}",
            },
            "local_edge_provenance": {
                "source_synapse_id": f"snn-rollout-local:{key}:0",
                "source_rollout_step_index": index,
                "target_rollout_step_index": index + 1,
            },
        }
    return {
        "language_capacity": {
            "surface": "snn_language_capacity_state.v1",
            "language_neuron_count": 128,
            "sparse_edge_budget": max(512, int(entry_count)),
            "outgoing_fanout_budget": 32,
            "dynamic_capacity_enabled": True,
        },
        "dense_readout_layout": {
            "surface": "snn_language_dense_readout_layout_state.v1",
            "target_language_neuron_count": 128,
            "tensor_materialization": {
                "applied": True,
                "actual_device": "cpu",
                "target_dense_readout_shape": [128, 128],
                "materializes_dense_tensor_weights": True,
            },
            "dense_resize_applied": True,
            "dynamic_dense_readout_enabled": True,
        },
        "dense_readout_weights": tensor,
        "sparse_transition_weights": CountedMapping(sparse_weights),
        "synapse_provenance_by_key": CountedMapping(provenance),
    }


def _bounded_projection(model: StatusReadModel) -> dict[str, Any]:
    capacity = model._snn_language_capacity_pressure()  # noqa: SLF001
    layout = model._snn_language_dense_readout_layout_state(capacity)  # noqa: SLF001
    dense = model._snn_language_dense_readout_tensor_integrity(layout)  # noqa: SLF001
    applied = model._snn_readout_applied_synapse_provenance()  # noqa: SLF001
    binding = model._snn_readout_rollout_server_state_binding()  # noqa: SLF001
    return {
        "capacity": capacity,
        "dense": dense,
        "applied": applied,
        "binding": binding,
    }


def _mapping_items(source: Any) -> list[tuple[str, Any]]:
    if not isinstance(source, Mapping):
        return []
    return [(str(key), value) for key, value in source.items()]


def _retired_broad_projection(state: Mapping[str, Any]) -> dict[str, Any]:
    capacity_sparse_items = _mapping_items(state.get("sparse_transition_weights"))
    capacity_provenance_items = _mapping_items(state.get("synapse_provenance_by_key"))
    dense_sparse_items = _mapping_items(state.get("sparse_transition_weights"))
    applied_provenance_items = _mapping_items(state.get("synapse_provenance_by_key"))
    binding_sparse_items = _mapping_items(state.get("sparse_transition_weights"))
    sparse_keys = {key for key, _value in capacity_sparse_items}
    provenance_keys = {key for key, _value in capacity_provenance_items}

    active_neurons: set[int] = set()
    outgoing: dict[int, int] = {}
    invalid = 0
    for key, _value in capacity_sparse_items:
        try:
            source, target = (int(part) for part in key.split(":", maxsplit=1))
        except (TypeError, ValueError):
            invalid += 1
            continue
        active_neurons.update((source, target))
        outgoing[source] = int(outgoing.get(source, 0)) + 1
    tensor = state.get("dense_readout_weights")
    tensor_available = isinstance(tensor, torch.Tensor)
    nonzero_count = int(torch.count_nonzero(tensor).item()) if tensor_available else 0
    transition_hash = (
        hashlib.sha256(
            json.dumps(
                dict(binding_sparse_items),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if binding_sparse_items
        else None
    )
    applied_rows = [
        dict(value)
        for _key, value in applied_provenance_items
        if isinstance(value, Mapping)
    ]
    return {
        "capacity": {
            "sparse_transition_weight_count": len(capacity_sparse_items),
            "synapse_provenance_count": len(capacity_provenance_items),
            "active_language_neuron_count": len(active_neurons),
            "max_outgoing_fanout": max(outgoing.values(), default=0),
            "invalid_synapse_key_count": invalid,
            "orphan_weight_count": len(sparse_keys - provenance_keys),
            "dangling_provenance_count": len(provenance_keys - sparse_keys),
            "eligible_for_capacity_expansion_design_review": True,
        },
        "dense": {
            "ready": bool(
                tensor_available and nonzero_count == len(dense_sparse_items)
            ),
            "nonzero_count": nonzero_count,
            "sparse_transition_weight_count": len(dense_sparse_items),
        },
        "applied": {
            "synapse_provenance_count": len(applied_rows),
            "eligible_for_readout_synapse_audit_review": True,
        },
        "binding": {
            "server_transition_memory_hash": transition_hash,
            "eligible_for_bounded_snn_readout_rollout_review": bool(transition_hash),
        },
    }


def _measure(
    factory: Callable[[], dict[str, Any]],
    fn: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    runs: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for _ in range(max(1, int(runs))):
        state = factory()
        sparse = state["sparse_transition_weights"]
        provenance = state["synapse_provenance_by_key"]
        tracemalloc.start()
        started = time.perf_counter()
        evidence = fn(state)
        latency_ms = (time.perf_counter() - started) * 1000.0
        _current, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        records.append(
            {
                "evidence": evidence,
                "latency_ms": latency_ms,
                "python_peak_mib": peak_bytes / (1024.0 * 1024.0),
                "sparse_reads": int(sparse.items_iterated),
                "provenance_reads": int(provenance.items_iterated),
            }
        )
    latencies = [float(record["latency_ms"]) for record in records]
    peaks = [float(record["python_peak_mib"]) for record in records]
    sparse_reads = [int(record["sparse_reads"]) for record in records]
    provenance_reads = [int(record["provenance_reads"]) for record in records]
    return {
        "runs": len(records),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 6),
            "median": round(statistics.median(latencies), 6),
            "min": round(min(latencies), 6),
            "max": round(max(latencies), 6),
        },
        "python_peak_mib": {
            "mean": round(statistics.fmean(peaks), 6),
            "max": round(max(peaks), 6),
        },
        "row_reads": {
            "sparse_transition_weights_last": sparse_reads[-1],
            "synapse_provenance_by_key_last": provenance_reads[-1],
            "total_last": sparse_reads[-1] + provenance_reads[-1],
            "total_mean": round(
                statistics.fmean(
                    [
                        sparse + provenance
                        for sparse, provenance in zip(sparse_reads, provenance_reads)
                    ]
                ),
                6,
            ),
        },
        "last_evidence": records[-1]["evidence"],
    }


def _cuda_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "torch_available": True,
            "cuda_available": False,
            "device_name": None,
            "memory_allocated_mib": 0.0,
            "memory_reserved_mib": 0.0,
        }
    return {
        "torch_available": True,
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "memory_allocated_mib": round(
            torch.cuda.memory_allocated() / (1024.0 * 1024.0),
            6,
        ),
        "memory_reserved_mib": round(
            torch.cuda.memory_reserved() / (1024.0 * 1024.0),
            6,
        ),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    entry_count = max(SNN_STATUS_TRANSITION_MEMORY_SOURCE_WINDOW_LIMIT, args.entry_count)
    runs = max(1, int(args.runs))
    state_holder: dict[str, Mapping[str, Any]] = {"state": {}}
    model = _build_read_model(lambda: state_holder["state"])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    cuda_before = _cuda_snapshot()

    def bounded_factory() -> dict[str, Any]:
        state = _seed_language_state(entry_count)
        state_holder["state"] = state
        return state

    bounded = _measure(
        bounded_factory,
        lambda _state: _bounded_projection(model),
        runs=runs,
    )
    retired = _measure(
        lambda: _seed_language_state(entry_count),
        _retired_broad_projection,
        runs=runs,
    )
    cuda_after = _cuda_snapshot()

    last = bounded["last_evidence"]
    windows = [
        last["capacity"]["source_window"],
        last["dense"]["source_window"],
        last["applied"]["source_window"],
        last["binding"]["source_window"],
    ]
    bounded_rows = int(bounded["row_reads"]["total_last"])
    retired_rows = int(retired["row_reads"]["total_last"])
    bounded_mean = float(bounded["latency_ms"]["mean"])
    retired_mean = float(retired["latency_ms"]["mean"])
    quality_checks = {
        "source_policy_present": all(
            window.get("policy") in {
                SNN_STATUS_TRANSITION_MEMORY_SOURCE_WINDOW_POLICY,
                "recent_status_applied_synapse_provenance_source_window_v1",
            }
            for window in windows
        ),
        "source_limit_respected": bounded_rows
        <= SNN_STATUS_TRANSITION_MEMORY_SOURCE_WINDOW_LIMIT * 2 * len(windows),
        "retained_counts_preserved": int(
            last["capacity"]["sparse_transition_weight_count"]
        )
        == entry_count
        and int(last["applied"]["synapse_provenance_count"]) == entry_count,
        "truncated_status_blocks_exact_reviews": (
            last["capacity"]["eligible_for_capacity_expansion_design_review"] is False
            and last["dense"]["ready"] is False
            and last["applied"]["eligible_for_readout_synapse_audit_review"] is False
            and last["binding"]["server_transition_memory_hash"] is None
        ),
        "no_global_scan": all(
            window.get("global_candidate_scan") is False
            and window.get("global_score_scan") is False
            for window in windows
        ),
        "not_live_tick": all(
            window.get("runs_live_tick") is False
            and window.get("runs_every_token") is False
            for window in windows
        ),
        "cpu_archival_metadata": all(
            window.get("archival_storage_device") == "cpu"
            and window.get("gpu_used") is False
            for window in windows
        ),
        "no_hidden_language_reasoning": all(
            window.get("language_reasoning") is False
            and window.get("raw_text_payload_loaded") is False
            for window in windows
        ),
    }
    return {
        "surface": "status_transition_memory_source_window_benchmark.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": {
            "entry_count": int(entry_count),
            "requested_entry_count": int(args.entry_count),
            "runs": int(runs),
            "source_window_limit": int(
                SNN_STATUS_TRANSITION_MEMORY_SOURCE_WINDOW_LIMIT
            ),
            "projection_count": len(windows),
        },
        "pass": all(quality_checks.values()),
        "quality_checks": quality_checks,
        "quality": {
            "truncated_status_blocks_exact_reviews": quality_checks[
                "truncated_status_blocks_exact_reviews"
            ],
            "retained_counts_preserved": quality_checks["retained_counts_preserved"],
            "quality_gate_passed": all(quality_checks.values()),
        },
        "latency": {
            "bounded": bounded["latency_ms"],
            "retired_broad_scan": retired["latency_ms"],
            "bounded_mean_ms": bounded_mean,
            "retired_broad_scan_mean_ms": retired_mean,
            "bounded_speedup_vs_retired": round(
                retired_mean / max(bounded_mean, 1e-9),
                6,
            ),
        },
        "work": {
            "bounded_rows_read_total": bounded_rows,
            "retired_rows_read_total": retired_rows,
            "row_reduction": round(retired_rows / max(1, bounded_rows), 6),
            "bounded_row_reads": bounded["row_reads"],
            "retired_row_reads": retired["row_reads"],
        },
        "source_windows": {
            "capacity": last["capacity"]["source_window"],
            "dense": last["dense"]["source_window"],
            "applied": last["applied"]["source_window"],
            "binding": last["binding"]["source_window"],
        },
        "bounded_evidence": {
            "capacity_promotion_status": last["capacity"]["promotion_status"],
            "dense_ready": last["dense"]["ready"],
            "applied_promotion_status": last["applied"]["promotion_status"],
            "binding_promotion_status": last["binding"]["promotion_status"],
        },
        "retired_path_comparison": {
            "old_policy": "status_read_model_materialized_transition_memory_maps_per_projection",
            "production_callable": False,
            "benchmark_local_only": True,
        },
        "resource_behavior": {
            "bounded_python_peak_mib": bounded["python_peak_mib"],
            "retired_python_peak_mib": retired["python_peak_mib"],
            "cuda_before": cuda_before,
            "cuda_after": cuda_after,
            "cuda_allocated_delta_mib": round(
                float(cuda_after["memory_allocated_mib"])
                - float(cuda_before["memory_allocated_mib"]),
                6,
            ),
            "cuda_reserved_delta_mib": round(
                float(cuda_after["memory_reserved_mib"])
                - float(cuda_before["memory_reserved_mib"]),
                6,
            ),
            "gpu_used": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry-count", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
