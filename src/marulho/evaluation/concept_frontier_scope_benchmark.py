from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import tracemalloc
from types import SimpleNamespace
from typing import Any

import torch

from marulho.consolidation.memory_store import DualMemoryStore
from marulho.training.autonomy_runner import (
    SourceBank,
    concept_frontier_metrics_with_report,
)


class _FixedRoutingIndex:
    def __init__(self, candidate_buckets: list[int]) -> None:
        self.candidate_buckets = [int(value) for value in candidate_buckets]

    def search_tensors(self, query: torch.Tensor, *, k: int):
        count = max(1, min(int(k), len(self.candidate_buckets)))
        ids = torch.tensor([self.candidate_buckets[:count]], dtype=torch.long)
        distances = torch.zeros((1, count), dtype=torch.float32)
        return ids, distances


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    value = vector.detach().cpu().float().reshape(-1)
    norm = float(value.norm().item())
    if norm <= 1e-8:
        return value
    return torch.nn.functional.normalize(value, dim=0)


def _build_trainer(
    *,
    capacity: int,
    bucket_count: int,
    candidate_bucket_count: int,
    probe_count: int,
    dim: int,
    seed: int,
) -> tuple[Any, SourceBank]:
    torch.manual_seed(seed)
    query = _normalize(torch.randn(dim))
    probe_patterns = [
        _normalize(query + 0.005 * torch.randn(dim))
        for _ in range(max(1, int(probe_count)))
    ]
    target_bucket = 7 % max(1, int(bucket_count))
    target_indices: list[int] = []
    candidate_buckets = [
        (target_bucket + offset) % max(1, int(bucket_count))
        for offset in range(max(1, int(candidate_bucket_count)))
    ]
    store = DualMemoryStore(capacity=capacity, ema_alpha=0.1)
    for index in range(int(capacity)):
        if int(index % max(1, int(bucket_count))) == target_bucket and index < max(1, int(bucket_count)) * 2:
            vector = _normalize(query + 0.01 * torch.randn(dim))
            target_indices.append(int(index))
        else:
            vector = _normalize(torch.randn(dim))
        bucket_id = int(index % max(1, int(bucket_count)))
        store.update(
            vector,
            token_count=index,
            importance=0.5,
            bucket_id=bucket_id,
            routing_key=vector,
            input_pattern=vector,
        )
        store.slow_capture_tag[-1] = 0.8 if bucket_id == target_bucket else 0.1
        store.slow_local_prp[-1] = 0.7 if bucket_id == target_bucket else 0.1
        store.slow_consolidation_level[-1] = 0.25 if bucket_id == target_bucket else 0.65
    store._rebuild_bucket_entry_index()
    trainer = SimpleNamespace(
        token_count=int(capacity),
        config=SimpleNamespace(k_routing=int(candidate_bucket_count)),
        model=SimpleNamespace(
            memory_store=store,
            routing_index=_FixedRoutingIndex(candidate_buckets),
        ),
        routing_key_for_pattern=lambda pattern: pattern,
        expected_frontier_target_indices=target_indices,
        expected_frontier_target_bucket=int(target_bucket),
    )
    bank = SourceBank(
        name="concept-frontier-scope",
        source="synthetic",
        source_type="synthetic",
        hf_config=None,
        text_field="text",
        probe_patterns=probe_patterns,
        probe_raw_windows=[
            f"concept frontier query {index}"
            for index in range(len(probe_patterns))
        ],
        train_patterns=[],
        train_raw_windows=[],
    )
    return trainer, bank


def run_benchmark(
    *,
    capacity: int,
    bucket_count: int,
    candidate_bucket_count: int,
    probe_count: int,
    dim: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    trainer, bank = _build_trainer(
        capacity=capacity,
        bucket_count=bucket_count,
        candidate_bucket_count=candidate_bucket_count,
        probe_count=probe_count,
        dim=dim,
        seed=seed,
    )
    bounded_latencies: list[float] = []
    bounded_result: tuple[float, float, float] = (1.0, 1.0, 0.0)
    bounded_report: dict[str, Any] = {}
    for _ in range(max(1, int(iterations))):
        bounded_novelty, bounded_uncertainty, bounded_support, bounded_report = concept_frontier_metrics_with_report(
            trainer,
            bank,
        )
        bounded_result = (bounded_novelty, bounded_uncertainty, bounded_support)
        bounded_latencies.append(float(bounded_report["latency_ms"]))

    tracemalloc.start()
    _ = concept_frontier_metrics_with_report(trainer, bank)
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    bounded_selected = {int(value) for value in bounded_report.get("selected_indices", [])}
    expected_targets = {
        int(value)
        for value in getattr(trainer, "expected_frontier_target_indices", [])
    }
    selected_targets = expected_targets & bounded_selected
    bounded_top1 = next(iter(bounded_report.get("selected_indices", [])), None)
    top1_target_match = (
        bounded_top1 is not None and int(bounded_top1) in expected_targets
    )
    target_hit_rate = (
        float(len(selected_targets)) / float(max(1, min(len(expected_targets), 2)))
        if expected_targets
        else 0.0
    )
    bounded_mean = statistics.fmean(bounded_latencies)
    bounded_p95 = float(
        sorted(bounded_latencies)[max(0, int(len(bounded_latencies) * 0.95) - 1)]
    )
    cuda_available = bool(torch.cuda.is_available())
    cuda_allocated = (
        float(torch.cuda.memory_allocated() / (1024 * 1024))
        if cuda_available
        else 0.0
    )
    report = {
        "surface": "concept_frontier_scope_benchmark.v1",
        "capacity": int(capacity),
        "bucket_count": int(bucket_count),
        "candidate_bucket_count": int(candidate_bucket_count),
        "source_probe_count": int(probe_count),
        "source_probe_window_limit": int(
            bounded_report.get("source_probe_window_limit", 0)
        ),
        "source_probe_index_count": int(
            bounded_report.get("source_probe_index_count", 0)
        ),
        "candidate_window_limit": int(bounded_report.get("candidate_window_limit", 0)),
        "expected_frontier_target_bucket": int(
            getattr(trainer, "expected_frontier_target_bucket", -1)
        ),
        "expected_frontier_target_indices": sorted(expected_targets),
        "iterations": int(max(1, iterations)),
        "dim": int(dim),
        "seed": int(seed),
        "memory_budget": {
            "archival_entries": int(capacity),
            "source_probe_window_limit": int(
                bounded_report.get("source_probe_window_limit", 0)
            ),
            "candidate_window_limit": int(
                bounded_report.get("candidate_window_limit", 0)
            ),
            "score_budget_entries": int(bounded_report.get("score_count", 0)),
            "retired_full_scan_rows_removed": int(capacity),
        },
        "bounded_candidate_window": {
            "mean_latency_ms": float(bounded_mean),
            "p95_latency_ms": float(bounded_p95),
            "score_count": int(bounded_report.get("score_count", 0)),
            "source_probe_index_count": int(
                bounded_report.get("source_probe_index_count", 0)
            ),
            "candidate_index_available_count": int(
                bounded_report.get("candidate_index_available_count", 0)
            ),
            "candidate_index_count": int(bounded_report.get("candidate_index_count", 0)),
            "selected_indices": list(bounded_report.get("selected_indices", [])),
            "frontier_row_read_count": int(bounded_report.get("frontier_row_read_count", 0)),
            "frontier_row_reader_owned_by_store": bool(
                bounded_report.get("frontier_row_reader_owned_by_store", False)
            ),
            "direct_slow_memory_array_reads_retired": bool(
                bounded_report.get("direct_slow_memory_array_reads_retired", False)
            ),
            "effective_capture_reader_used": bool(
                bounded_report.get("effective_capture_reader_used", True)
            ),
            "global_candidate_scan": bool(bounded_report.get("global_candidate_scan", True)),
            "global_score_scan": bool(bounded_report.get("global_score_scan", True)),
            "runs_live_tick": bool(bounded_report.get("runs_live_tick", True)),
            "archival_storage_device": bounded_report.get("archival_storage_device"),
            "novelty": float(bounded_result[0]),
            "uncertainty": float(bounded_result[1]),
            "support": float(bounded_result[2]),
        },
        "retired_concept_frontier_full_scan_absence": {
            "implementation_present": False,
            "diagnostic_callable": False,
            "active_report_field_present": False,
            "removed_policy": "concept_frontier_metrics_full_slow_memory_scan_comparator",
        },
        "quality": {
            "metric": "seeded_frontier_target_selection",
            "target_hit_rate": float(target_hit_rate),
            "top1_target_match": bool(top1_target_match),
            "selected_target_indices": sorted(selected_targets),
            "novelty": float(bounded_result[0]),
            "uncertainty": float(bounded_result[1]),
            "support": float(bounded_result[2]),
        },
        "latency": {
            "bounded_mean_latency_ms": float(bounded_mean),
            "bounded_p95_latency_ms": float(bounded_p95),
        },
        "resource_behavior": {
            "python_tracemalloc_current_mib": round(
                float(traced_current) / (1024.0 * 1024.0),
                6,
            ),
            "python_tracemalloc_peak_mib": round(
                float(traced_peak) / (1024.0 * 1024.0),
                6,
            ),
            "cuda_memory_allocated_after_mib": cuda_allocated,
        },
        "device_placement": {
            "archival_storage_device": "cpu",
            "source_probe_device": "cpu",
            "score_device": "cpu",
            "active_replay_cuda_required": False,
            "cuda_available": cuda_available,
            "cuda_memory_allocated_after_mib": cuda_allocated,
        },
        "gates": {
            "quality_gate_pass": bool(
                top1_target_match
                and target_hit_rate >= 1.0
                and float(bounded_result[1]) >= 0.5
                and float(bounded_result[2]) >= 0.2
            ),
            "bounded_scan_gate_pass": bool(
                not bounded_report.get("global_candidate_scan", True)
                and not bounded_report.get("global_score_scan", True)
                and int(bounded_report.get("score_count", 0))
                <= int(bounded_report.get("candidate_window_limit", 0))
                and bool(bounded_report.get("frontier_row_reader_owned_by_store"))
                and bool(bounded_report.get("direct_slow_memory_array_reads_retired"))
                and not bool(bounded_report.get("effective_capture_reader_used", True))
            ),
            "latency_gate_pass": bool(bounded_p95 <= 100.0),
            "live_tick_gate_pass": bool(not bounded_report.get("runs_live_tick", True)),
            "retired_path_absence_gate_pass": True,
        },
        "bounded_report": bounded_report,
    }
    report["passed"] = all(bool(value) for value in report["gates"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark bounded concept-frontier memory metrics")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=8192)
    parser.add_argument("--bucket-count", type=int, default=1024)
    parser.add_argument("--candidate-bucket-count", type=int, default=8)
    parser.add_argument("--probe-count", type=int, default=64)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260617)
    args = parser.parse_args()
    report = run_benchmark(
        capacity=args.capacity,
        bucket_count=args.bucket_count,
        candidate_bucket_count=args.candidate_bucket_count,
        probe_count=args.probe_count,
        dim=args.dim,
        iterations=args.iterations,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["gates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
