from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
from typing import Any

import torch

from marulho.consolidation.memory_store import DualMemoryStore
from marulho.training.autonomy_runner import (
    SourceBank,
    _mean_signature,
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


def _legacy_full_scan_metrics(trainer: Any, bank: SourceBank) -> tuple[float, float, float, dict[str, Any]]:
    started = time.perf_counter()
    bank_signature = _mean_signature(
        [trainer.routing_key_for_pattern(pattern).detach().cpu() for pattern in bank.probe_patterns]
    )
    store = trainer.model.memory_store
    if bank_signature is None:
        return 1.0, 1.0, 0.0, {
            "status": "empty",
            "fallback_reason": "empty_frontier_probe_signature",
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }

    memory_keys: list[tuple[int, torch.Tensor]] = []
    for index, key in enumerate(store.slow_routing_keys):
        if not isinstance(key, torch.Tensor):
            continue
        vector = _normalize(key)
        if int(vector.numel()) <= 0 or float(vector.norm().item()) <= 1e-8:
            continue
        memory_keys.append((int(index), vector))
    if not memory_keys:
        return 1.0, 1.0, 0.0, {
            "status": "empty",
            "fallback_reason": "empty_frontier_memory_keys",
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }

    similarities = torch.tensor(
        [float(torch.dot(bank_signature, key).item()) for _, key in memory_keys],
        dtype=torch.float32,
    )
    top_k = min(8, int(similarities.numel()))
    top_values, top_local_indices = torch.topk(similarities, k=top_k)
    selected_indices = [int(memory_keys[int(local_idx.item())][0]) for local_idx in top_local_indices]
    shifted = torch.clamp(top_values + 1.0, min=1e-6)
    weights = shifted / (shifted.sum() + 1e-8)
    effective_captures = torch.tensor(
        [
            float(
                store.query_match_row(
                    int(index),
                    current_token=int(trainer.token_count),
                    include_text_payload=False,
                ).get("capture_strength", 0.0)
                or 0.0
            )
            for index in selected_indices
        ],
        dtype=torch.float32,
    )
    consolidations = torch.tensor(
        [
            float(
                store.query_match_row(
                    int(index),
                    current_token=int(trainer.token_count),
                    include_text_payload=False,
                ).get("consolidation_level", 0.0)
                or 0.0
            )
            for index in selected_indices
        ],
        dtype=torch.float32,
    )
    novelty = max(0.0, min(1.0, 1.0 - max(0.0, float(top_values.max().item()))))
    uncertainty_pressure = torch.clamp(effective_captures - consolidations, min=0.0) + 0.5 * torch.clamp(
        1.0 - consolidations,
        min=0.0,
    )
    uncertainty = max(0.0, min(1.0, float(torch.dot(weights, uncertainty_pressure).item())))
    support = max(0.0, min(1.0, float(torch.dot(weights, consolidations).item())))
    latency_ms = (time.perf_counter() - started) * 1000.0
    return novelty, uncertainty, support, {
        "status": "measured",
        "candidate_scope": "diagnostic_full_slow_memory_scan",
        "memory_size": int(len(store.slow_buffer)),
        "score_count": int(len(memory_keys)),
        "selected_indices": selected_indices,
        "selected_count": int(len(selected_indices)),
        "global_candidate_scan": True,
        "global_score_scan": True,
        "latency_ms": float(latency_ms),
    }


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
    candidate_buckets = [
        (target_bucket + offset) % max(1, int(bucket_count))
        for offset in range(max(1, int(candidate_bucket_count)))
    ]
    store = DualMemoryStore(capacity=capacity, ema_alpha=0.1)
    for index in range(int(capacity)):
        if int(index % max(1, int(bucket_count))) == target_bucket and index < max(1, int(bucket_count)) * 2:
            vector = _normalize(query + 0.01 * torch.randn(dim))
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
    legacy_latencies: list[float] = []
    bounded_latencies: list[float] = []
    legacy_result: tuple[float, float, float] = (1.0, 1.0, 0.0)
    bounded_result: tuple[float, float, float] = (1.0, 1.0, 0.0)
    legacy_report: dict[str, Any] = {}
    bounded_report: dict[str, Any] = {}
    for _ in range(max(1, int(iterations))):
        legacy_novelty, legacy_uncertainty, legacy_support, legacy_report = _legacy_full_scan_metrics(
            trainer,
            bank,
        )
        bounded_novelty, bounded_uncertainty, bounded_support, bounded_report = concept_frontier_metrics_with_report(
            trainer,
            bank,
        )
        legacy_result = (legacy_novelty, legacy_uncertainty, legacy_support)
        bounded_result = (bounded_novelty, bounded_uncertainty, bounded_support)
        legacy_latencies.append(float(legacy_report["latency_ms"]))
        bounded_latencies.append(float(bounded_report["latency_ms"]))

    novelty_delta = abs(float(legacy_result[0]) - float(bounded_result[0]))
    uncertainty_delta = abs(float(legacy_result[1]) - float(bounded_result[1]))
    support_delta = abs(float(legacy_result[2]) - float(bounded_result[2]))
    legacy_selected = {int(value) for value in legacy_report.get("selected_indices", [])}
    bounded_selected = {int(value) for value in bounded_report.get("selected_indices", [])}
    top_overlap = len(legacy_selected & bounded_selected) / max(1, min(len(legacy_selected), len(bounded_selected)))
    legacy_top1 = next(iter(legacy_report.get("selected_indices", [])), None)
    bounded_top1 = next(iter(bounded_report.get("selected_indices", [])), None)
    top1_match = legacy_top1 is not None and int(legacy_top1) == int(bounded_top1)
    legacy_mean = statistics.fmean(legacy_latencies)
    bounded_mean = statistics.fmean(bounded_latencies)
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
        "iterations": int(max(1, iterations)),
        "dim": int(dim),
        "seed": int(seed),
        "legacy_full_scan": {
            "mean_latency_ms": float(legacy_mean),
            "score_count": int(legacy_report.get("score_count", 0)),
            "selected_indices": list(legacy_report.get("selected_indices", [])),
            "global_candidate_scan": True,
            "global_score_scan": True,
            "novelty": float(legacy_result[0]),
            "uncertainty": float(legacy_result[1]),
            "support": float(legacy_result[2]),
        },
        "bounded_candidate_window": {
            "mean_latency_ms": float(bounded_mean),
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
        "quality": {
            "novelty_delta": float(novelty_delta),
            "uncertainty_delta": float(uncertainty_delta),
            "support_delta": float(support_delta),
            "top_overlap": float(top_overlap),
            "top1_match": bool(top1_match),
        },
        "latency": {
            "speedup": float(legacy_mean / max(1e-9, bounded_mean)),
        },
        "gates": {
            "quality_gate_pass": bool(
                novelty_delta <= 0.02
                and uncertainty_delta <= 0.2
                and support_delta <= 0.2
                and top1_match
            ),
            "bounded_scan_gate_pass": bool(
                not bounded_report.get("global_candidate_scan", True)
                and not bounded_report.get("global_score_scan", True)
                and int(bounded_report.get("score_count", 0)) < int(legacy_report.get("score_count", 0))
                and bool(bounded_report.get("frontier_row_reader_owned_by_store"))
                and bool(bounded_report.get("direct_slow_memory_array_reads_retired"))
                and not bool(bounded_report.get("effective_capture_reader_used", True))
            ),
            "latency_gate_pass": bool(bounded_mean <= legacy_mean),
            "live_tick_gate_pass": bool(not bounded_report.get("runs_live_tick", True)),
        },
        "bounded_report": bounded_report,
    }
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
