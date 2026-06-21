"""Benchmark incremental applied-replay lineage checkpoint summaries.

The retired comparison in this file is diagnostic only. Production checkpoint
save/restore must read the mutation-maintained CPU summary and must not scan
``synapse_provenance_by_key`` to derive replay lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping as CollectionsMapping
from pathlib import Path
import statistics
import time
import tracemalloc
from typing import Any, Callable, Mapping

import torch

from marulho.service.applied_replay_lineage import (
    applied_replay_lineage_checkpoint_summary,
    record_applied_replay_lineage_provenance,
)

_DIGEST_MODULUS = 1 << 256
_ZERO_DIGEST = "0" * 64


class CountedMapping(CollectionsMapping):
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.keys_iterated = 0
        self.items_iterated = 0
        self.getitem_count = 0

    def __getitem__(self, key: str) -> Any:
        self.getitem_count += 1
        return self._payload[key]

    def __iter__(self):
        for key in self._payload:
            self.keys_iterated += 1
            yield key

    def __len__(self) -> int:
        return len(self._payload)

    def items(self):
        for item in self._payload.items():
            self.items_iterated += 1
            yield item

    @property
    def source_reads(self) -> int:
        return int(self.keys_iterated + self.items_iterated + self.getitem_count)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _digest_int(value: Any) -> int:
    try:
        return int(str(value), 16)
    except (TypeError, ValueError):
        return 0


def _digest_hex(value: int) -> str:
    return f"{int(value) % _DIGEST_MODULUS:064x}"


def _provenance_row(index: int, key: str) -> dict[str, Any]:
    return {
        "provenance_type": "replay_regeneration",
        "permit_id": f"permit-{index}",
        "replay_artifact_id": f"artifact-{index}",
        "replay_artifact_hash": f"artifact-hash-{index}",
        "replay_window_hash": f"window-hash-{index}",
        "readout_evidence_hashes": [
            f"readout-hash-{index}",
            f"readout-hash-{index + 1}",
        ],
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


def _transition_key(index: int) -> str:
    return f"{int(index // 512)}:{int(index % 512)}"


def _seed_state(entry_count: int) -> dict[str, Any]:
    provenance: dict[str, dict[str, Any]] = {}
    state: dict[str, Any] = {"synapse_provenance_by_key": {}}
    for index in range(max(0, int(entry_count))):
        key = _transition_key(index)
        row = _provenance_row(index, key)
        provenance[key] = row
        record_applied_replay_lineage_provenance(state, key, row)
    state["synapse_provenance_by_key"] = CountedMapping(provenance)
    return state


def _row_material(key: str, provenance: Mapping[str, Any]) -> dict[str, Any] | None:
    if str(provenance.get("provenance_type") or "") != "replay_regeneration":
        return None
    source_metadata_hash = str(provenance.get("source_metadata_hash") or "")
    emission_lineage = (
        dict(provenance.get("emission_lineage"))
        if isinstance(provenance.get("emission_lineage"), Mapping)
        else {}
    )
    if not source_metadata_hash and not emission_lineage:
        return None
    return {
        "synapse_key": str(key),
        "source_metadata_hash": source_metadata_hash,
        "emission_lineage": emission_lineage,
    }


def _row_complete(material: Mapping[str, Any]) -> bool:
    emission_lineage = (
        material.get("emission_lineage")
        if isinstance(material.get("emission_lineage"), Mapping)
        else {}
    )
    return bool(
        material.get("source_metadata_hash")
        and emission_lineage.get("emission_hash")
        and emission_lineage.get("readout_evidence_hash")
        and emission_lineage.get("prediction_hash")
    )


def _retired_full_scan_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    provenance_by_key = (
        state.get("synapse_provenance_by_key")
        if isinstance(state.get("synapse_provenance_by_key"), Mapping)
        else {}
    )
    material: list[dict[str, Any]] = []
    complete_rows = 0
    for key in sorted(dict(provenance_by_key).keys()):
        provenance = provenance_by_key.get(key)
        if not isinstance(provenance, Mapping):
            continue
        row = _row_material(str(key), provenance)
        if row is None:
            continue
        material.append(row)
        complete_rows += int(_row_complete(row))
    xor_value = 0
    sum_value = 0
    for row in material:
        row_hash = _sha256_json(row)
        row_int = _digest_int(row_hash)
        xor_value ^= row_int
        sum_value += row_int
    count = len(material)
    digest_xor = _digest_hex(xor_value)
    digest_sum = _digest_hex(sum_value)
    material_hash = (
        _sha256_json(
            {
                "applied_replay_lineage_count": count,
                "complete_applied_replay_lineage_count": complete_rows,
                "incomplete_applied_replay_lineage_count": max(
                    0,
                    count - complete_rows,
                ),
                "lineage_digest_xor": digest_xor,
                "lineage_digest_sum_mod_2_256": digest_sum,
            }
        )
        if count
        else None
    )
    return {
        "surface": "retired_applied_replay_lineage_full_scan_summary.v1",
        "production_callable": False,
        "benchmark_local_only": True,
        "applied_replay_lineage_count": count,
        "complete_applied_replay_lineage_count": complete_rows,
        "incomplete_applied_replay_lineage_count": max(0, count - complete_rows),
        "lineage_digest_xor": digest_xor,
        "lineage_digest_sum_mod_2_256": digest_sum,
        "lineage_material_hash": material_hash,
        "full_provenance_scan": True,
    }


def _active_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    return applied_replay_lineage_checkpoint_summary(
        state,
        source="applied_replay_lineage_checkpoint_summary_benchmark",
    )


def _measure(
    factory: Callable[[], dict[str, Any]],
    fn: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    runs: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for _ in range(max(1, int(runs))):
        state = factory()
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
                "source_reads": int(provenance.source_reads)
                if isinstance(provenance, CountedMapping)
                else -1,
            }
        )
    latencies = [float(record["latency_ms"]) for record in records]
    peaks = [float(record["python_peak_mib"]) for record in records]
    source_reads = [int(record["source_reads"]) for record in records]
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
        "source_reads": {
            "last": source_reads[-1],
            "mean": round(statistics.fmean(source_reads), 6),
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
    entry_count = max(1, int(args.entry_count))
    runs = max(1, int(args.runs))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    cuda_before = _cuda_snapshot()
    active = _measure(
        lambda: _seed_state(entry_count),
        _active_summary,
        runs=runs,
    )
    retired = _measure(
        lambda: _seed_state(entry_count),
        _retired_full_scan_summary,
        runs=runs,
    )
    cuda_after = _cuda_snapshot()
    active_last = active["last_evidence"]
    retired_last = retired["last_evidence"]
    active_reads = int(active["source_reads"]["last"])
    retired_reads = int(retired["source_reads"]["last"])
    active_mean = float(active["latency_ms"]["mean"])
    retired_mean = float(retired["latency_ms"]["mean"])
    quality_checks = {
        "counts_match_retired_diagnostic": all(
            active_last.get(key) == retired_last.get(key)
            for key in (
                "applied_replay_lineage_count",
                "complete_applied_replay_lineage_count",
                "incomplete_applied_replay_lineage_count",
                "lineage_digest_xor",
                "lineage_digest_sum_mod_2_256",
                "lineage_material_hash",
            )
        ),
        "active_summary_source_available": bool(
            active_last.get("summary_source_available")
        ),
        "active_zero_source_reads": active_reads == 0,
        "retired_scans_source_records": retired_reads >= entry_count,
        "no_production_full_scan": active_last.get("full_provenance_scan") is False
        and int(active_last.get("source_record_scan_count", -1)) == 0,
        "cpu_archival_metadata": active_last.get("archival_metadata_device") == "cpu"
        and active_last.get("gpu_used") is False,
        "not_live_tick": active_last.get("runs_replay") is False
        and active_last.get("applies_plasticity") is False
        and active_last.get("issues_regeneration_permit") is False,
        "no_hidden_language_reasoning": active_last.get("language_reasoning") is False
        and active_last.get("raw_text_absent") is True,
    }
    return {
        "surface": "applied_replay_lineage_checkpoint_summary_benchmark.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": {
            "entry_count": entry_count,
            "runs": runs,
        },
        "pass": all(quality_checks.values()),
        "quality_checks": quality_checks,
        "quality": {
            "quality_gate_passed": all(quality_checks.values()),
            "lineage_reconstruction_parity": quality_checks[
                "counts_match_retired_diagnostic"
            ],
            "source_reads_eliminated": quality_checks["active_zero_source_reads"],
        },
        "latency": {
            "active_incremental": active["latency_ms"],
            "retired_full_scan": retired["latency_ms"],
            "active_incremental_mean_ms": active_mean,
            "retired_full_scan_mean_ms": retired_mean,
            "active_speedup_vs_retired": round(
                retired_mean / max(active_mean, 1e-9),
                6,
            ),
        },
        "work": {
            "active_source_reads": active_reads,
            "retired_source_reads": retired_reads,
            "source_reads_removed": max(0, retired_reads - active_reads),
            "source_read_reduction": (
                "all_provenance_reads_eliminated"
                if active_reads == 0
                else round(retired_reads / max(1, active_reads), 6)
            ),
        },
        "active_evidence": active_last,
        "retired_path_comparison": {
            "old_policy": "runtime_persistence_checkpoint_summary_full_synapse_provenance_scan",
            "production_callable": False,
            "benchmark_local_only": True,
            "evidence": retired_last,
        },
        "resource_behavior": {
            "active_python_peak_mib": active["python_peak_mib"],
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
    parser.add_argument("--entry-count", type=int, default=65536)
    parser.add_argument("--runs", type=int, default=7)
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
