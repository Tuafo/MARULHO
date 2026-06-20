"""Benchmark bounded applied-synapse provenance status source windows.

Production status evidence must not scan every applied synapse provenance row
before an explicit audit window. The broad projection below is benchmark-local
retired behavior for latency and work comparison only.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Mapping as CollectionsMapping
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
    SNN_STATUS_APPLIED_SYNAPSE_PROVENANCE_SOURCE_WINDOW_LIMIT,
    SNN_STATUS_APPLIED_SYNAPSE_PROVENANCE_SOURCE_WINDOW_POLICY,
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


def _seed_language_state(entry_count: int) -> dict[str, Any]:
    sparse_weights: dict[str, float] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for index in range(max(0, int(entry_count))):
        key = f"{index}:{index + 1}"
        sparse_weights[key] = 0.1
        provenance[key] = {
            "provenance_type": "replay_regeneration",
            "permit_id": f"permit-{index}",
            "replay_artifact_id": f"artifact-{index}",
            "source_metadata_hash": f"source-metadata-hash-{index}",
            "emission_lineage": {
                "emission_hash": f"emission-hash-{index}",
                "readout_evidence_hash": f"readout-hash-{index}",
                "prediction_hash": f"prediction-hash-{index}",
            },
            "local_edge_provenance": {
                "source_synapse_id": f"snn-rollout-local:{key}:0",
                "source_trace_index": index,
                "source_rollout_step_index": index,
                "target_rollout_step_index": index + 1,
                "source_active_indices_hash": f"source-active-hash-{index}",
                "target_active_indices_hash": f"target-active-hash-{index}",
            },
        }
    return {
        "sparse_transition_weights": CountedMapping(sparse_weights),
        "synapse_provenance_by_key": CountedMapping(provenance),
    }


def _counted_items(source: Any) -> list[tuple[str, Any]]:
    if not isinstance(source, Mapping):
        return []
    return [(str(key), value) for key, value in source.items()]


def _retired_broad_applied_synapse_provenance_status(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    sparse_items = _counted_items(state.get("sparse_transition_weights"))
    provenance_items = _counted_items(state.get("synapse_provenance_by_key"))
    sparse_keys = {key for key, _value in sparse_items}
    provenance_by_key = {
        key: dict(value)
        for key, value in provenance_items
        if isinstance(value, Mapping)
    }
    rows = list(provenance_by_key.values())
    replay_rows = [
        row
        for row in rows
        if str(row.get("provenance_type") or "") == "replay_regeneration"
    ]
    missing_local_edge_rows = sum(
        1
        for row in replay_rows
        if not isinstance(row.get("local_edge_provenance"), Mapping)
    )
    invalid_rollout_step_rows = 0
    replay_artifact_lineage_rows = 0
    complete_replay_artifact_lineage_rows = 0
    for row in replay_rows:
        local_edge = row.get("local_edge_provenance")
        if isinstance(local_edge, Mapping):
            source_step = local_edge.get("source_rollout_step_index")
            target_step = local_edge.get("target_rollout_step_index")
            if (
                isinstance(source_step, int)
                and isinstance(target_step, int)
                and target_step < source_step
            ):
                invalid_rollout_step_rows += 1
        replay_artifact_lineage_rows += 1
        emission_lineage = row.get("emission_lineage")
        if (
            isinstance(emission_lineage, Mapping)
            and row.get("source_metadata_hash")
            and emission_lineage.get("emission_hash")
            and emission_lineage.get("readout_evidence_hash")
            and emission_lineage.get("prediction_hash")
        ):
            complete_replay_artifact_lineage_rows += 1
    incomplete_lineage_rows = max(
        0,
        replay_artifact_lineage_rows - complete_replay_artifact_lineage_rows,
    )
    dangling_provenance_count = len(set(provenance_by_key) - sparse_keys)
    orphan_weight_count = len(sparse_keys - set(provenance_by_key))
    ready = bool(
        rows
        and orphan_weight_count == 0
        and dangling_provenance_count == 0
        and missing_local_edge_rows == 0
        and invalid_rollout_step_rows == 0
        and incomplete_lineage_rows == 0
    )
    return {
        "surface": "retired_broad_applied_synapse_provenance_status_scan.v1",
        "sparse_transition_weight_count": len(sparse_items),
        "synapse_provenance_count": len(provenance_items),
        "replay_regeneration_synapse_count": len(replay_rows),
        "complete_local_edge_provenance_count": len(replay_rows)
        - missing_local_edge_rows,
        "missing_local_edge_provenance_count": missing_local_edge_rows,
        "invalid_rollout_step_order_count": invalid_rollout_step_rows,
        "replay_artifact_lineage_count": replay_artifact_lineage_rows,
        "complete_replay_artifact_lineage_count": complete_replay_artifact_lineage_rows,
        "incomplete_replay_artifact_lineage_count": incomplete_lineage_rows,
        "orphan_weight_count": orphan_weight_count,
        "dangling_provenance_count": dangling_provenance_count,
        "eligible_for_readout_synapse_audit_review": ready,
    }


def _measurement_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    index_95 = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * 0.95))))
    return {
        "mean_ms": round(statistics.fmean(values), 6),
        "median_ms": round(statistics.median(values), 6),
        "p95_ms": round(ordered[index_95], 6),
        "min_ms": round(min(values), 6),
        "max_ms": round(max(values), 6),
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
                "latency_ms": float(latency_ms),
                "python_peak_mib": float(peak_bytes / (1024.0 * 1024.0)),
                "sparse_items_iterated": int(sparse.items_iterated),
                "provenance_items_iterated": int(provenance.items_iterated),
            }
        )
    latencies = [float(record["latency_ms"]) for record in records]
    peaks = [float(record["python_peak_mib"]) for record in records]
    sparse_reads = [int(record["sparse_items_iterated"]) for record in records]
    provenance_reads = [int(record["provenance_items_iterated"]) for record in records]
    return {
        "runs": int(len(records)),
        "latency_ms": _measurement_summary(latencies),
        "python_peak_mib": {
            "mean": round(statistics.fmean(peaks), 6),
            "max": round(max(peaks), 6),
        },
        "row_reads": {
            "sparse_transition_weights_mean": round(statistics.fmean(sparse_reads), 6),
            "synapse_provenance_by_key_mean": round(
                statistics.fmean(provenance_reads),
                6,
            ),
            "total_mean": round(
                statistics.fmean(
                    [
                        sparse + provenance
                        for sparse, provenance in zip(sparse_reads, provenance_reads)
                    ]
                ),
                6,
            ),
            "sparse_transition_weights_last": int(sparse_reads[-1]),
            "synapse_provenance_by_key_last": int(provenance_reads[-1]),
            "total_last": int(sparse_reads[-1] + provenance_reads[-1]),
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
            "max_memory_allocated_mib": 0.0,
            "max_memory_reserved_mib": 0.0,
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
        "max_memory_allocated_mib": round(
            torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
            6,
        ),
        "max_memory_reserved_mib": round(
            torch.cuda.max_memory_reserved() / (1024.0 * 1024.0),
            6,
        ),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    entry_count = max(
        SNN_STATUS_APPLIED_SYNAPSE_PROVENANCE_SOURCE_WINDOW_LIMIT,
        int(args.entry_count),
    )
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
        lambda _state: model._snn_readout_applied_synapse_provenance(),  # noqa: SLF001
        runs=runs,
    )
    retired = _measure(
        lambda: _seed_language_state(entry_count),
        _retired_broad_applied_synapse_provenance_status,
        runs=runs,
    )
    cuda_after = _cuda_snapshot()

    evidence = bounded["last_evidence"]
    source_window = dict(evidence.get("source_window") or {})
    required = dict(
        (evidence.get("promotion_gate") or {}).get("required_evidence") or {}
    )
    bounded_rows = int(bounded["row_reads"]["total_last"])
    retired_rows = int(retired["row_reads"]["total_last"])
    bounded_mean = float(bounded["latency_ms"]["mean_ms"])
    retired_mean = float(retired["latency_ms"]["mean_ms"])
    quality_checks = {
        "surface_present": source_window.get("surface")
        == "bounded_snn_status_applied_synapse_provenance_source_window.v1",
        "policy_present": source_window.get("policy")
        == SNN_STATUS_APPLIED_SYNAPSE_PROVENANCE_SOURCE_WINDOW_POLICY,
        "source_limit_respected": bounded_rows
        <= SNN_STATUS_APPLIED_SYNAPSE_PROVENANCE_SOURCE_WINDOW_LIMIT * 2,
        "retained_counts_preserved": int(evidence["sparse_transition_weight_count"])
        == entry_count
        and int(evidence["synapse_provenance_count"]) == entry_count,
        "bounded_health_preserved": int(evidence["source_synapse_provenance_count"])
        == SNN_STATUS_APPLIED_SYNAPSE_PROVENANCE_SOURCE_WINDOW_LIMIT
        and int(evidence["missing_local_edge_provenance_count"]) == 0
        and int(evidence["invalid_rollout_step_order_count"]) == 0
        and int(evidence["incomplete_replay_artifact_lineage_count"]) == 0,
        "truncated_status_blocks_exact_audit_readiness": (
            evidence["source_window_complete"] is False
            and evidence["eligible_for_readout_synapse_audit_review"] is False
            and required.get("source_window_complete_for_exact_status") is False
        ),
        "no_global_scan": source_window.get("global_candidate_scan") is False
        and source_window.get("global_score_scan") is False,
        "not_live_tick": source_window.get("runs_live_tick") is False
        and source_window.get("runs_every_token") is False,
        "cpu_archival_metadata": source_window.get("archival_storage_device") == "cpu"
        and source_window.get("lookup_device") == "cpu"
        and source_window.get("gpu_used") is False,
        "no_hidden_language_reasoning": source_window.get("language_reasoning") is False
        and source_window.get("raw_text_payload_loaded") is False,
    }
    return {
        "surface": "status_applied_synapse_provenance_source_window_benchmark.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": {
            "entry_count": int(entry_count),
            "requested_entry_count": int(args.entry_count),
            "runs": int(runs),
            "source_window_limit": int(
                SNN_STATUS_APPLIED_SYNAPSE_PROVENANCE_SOURCE_WINDOW_LIMIT
            ),
            "requested_source_window_limit": int(args.source_window_limit),
        },
        "pass": all(quality_checks.values()),
        "quality_checks": quality_checks,
        "quality": {
            "bounded_window_preserves_recent_provenance_health": quality_checks[
                "bounded_health_preserved"
            ],
            "truncated_status_blocks_exact_audit_readiness": quality_checks[
                "truncated_status_blocks_exact_audit_readiness"
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
        "source_window": source_window,
        "bounded_evidence": {
            "sparse_transition_weight_count": evidence["sparse_transition_weight_count"],
            "synapse_provenance_count": evidence["synapse_provenance_count"],
            "source_sparse_transition_weight_count": evidence[
                "source_sparse_transition_weight_count"
            ],
            "source_synapse_provenance_count": evidence[
                "source_synapse_provenance_count"
            ],
            "source_window_complete": evidence["source_window_complete"],
            "integrity_count_scope": evidence["integrity_count_scope"],
            "eligible_for_readout_synapse_audit_review": evidence[
                "eligible_for_readout_synapse_audit_review"
            ],
            "promotion_status": evidence["promotion_status"],
        },
        "retired_path_comparison": {
            "old_policy": "status_read_model_scanned_all_applied_synapse_provenance_keys",
            "retired_surface": retired["last_evidence"]["surface"],
            "production_callable": False,
            "benchmark_local_only": True,
        },
        "resource_behavior": {
            "python_tracemalloc_peak_mib": max(
                float(bounded["python_peak_mib"]["max"]),
                float(retired["python_peak_mib"]["max"]),
            ),
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
    parser.add_argument(
        "--source-window-limit",
        type=int,
        default=SNN_STATUS_APPLIED_SYNAPSE_PROVENANCE_SOURCE_WINDOW_LIMIT,
    )
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
