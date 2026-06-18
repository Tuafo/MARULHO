from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from marulho.config.model_config import MarulhoConfig
from marulho.reporting.io import write_json_file
from marulho.training.memory_consolidation_runner import (
    REPLAY_QUERY_ANCHOR_BUCKET_WINDOW_LIMIT,
    _bounded_replay_recall_evaluation,
    _collect_anchor_replay_queries,
)
from marulho.training.model import MarulhoModel
from marulho.training.runner_utils import set_seed
from marulho.training.trainer import MarulhoTrainer


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    sorted_values = sorted(float(value) for value in values)
    p95_index = min(len(sorted_values) - 1, int(0.95 * (len(sorted_values) - 1)))
    return {
        "mean_ms": float(statistics.fmean(sorted_values)),
        "p95_ms": float(sorted_values[p95_index]),
        "min_ms": float(sorted_values[0]),
        "max_ms": float(sorted_values[-1]),
    }


def _pattern(config: MarulhoConfig, index: int) -> torch.Tensor:
    pattern = torch.zeros(config.input_dim, dtype=torch.float32)
    pattern[int(index) % int(config.input_dim)] = 1.0
    return pattern


def _build_trainer(*, anchor_count: int, column_latent_dim: int) -> MarulhoTrainer:
    cfg = MarulhoConfig(
        n_columns=max(1, int(anchor_count)),
        column_latent_dim=max(1, int(column_latent_dim)),
        bootstrap_tokens=0,
        memory_capacity=max(1, int(anchor_count)),
        micro_sleep_interval_tokens=10**9,
        deep_sleep_interval_tokens=10**9,
        device="cpu",
    )
    trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
    trainer.memory_warm_started = True
    trainer.token_count = int(anchor_count)
    for bucket in range(int(anchor_count)):
        pattern = _pattern(cfg, bucket)
        routing_key = trainer.model.routing_key_from_pattern(pattern)
        assembly = torch.zeros(cfg.column_latent_dim, dtype=torch.float32)
        assembly[bucket % cfg.column_latent_dim] = 1.0
        trainer.model.memory_store.update(
            assembly,
            importance=1.0,
            token_count=bucket + 1,
            bucket_id=bucket,
            input_pattern=pattern,
            routing_key=routing_key.detach().cpu(),
            capture_tag=1.0,
        )
        trainer.column_anchors[bucket] = {
            "prototype": torch.empty(0, dtype=torch.float32, device=trainer.model.device),
            "input_weights": torch.empty(0, dtype=torch.float32, device=trainer.model.device),
            "strength": 2.0,
            "captured_at_token": bucket + 1,
            "captured_source_index": bucket,
            "capture_sequence": bucket,
        }
    return trainer


def _time_legacy_collection(
    trainer: MarulhoTrainer,
    *,
    max_queries: int,
    iterations: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    all_anchor_buckets = list(range(len(trainer.column_anchors)))
    latencies: list[float] = []
    report: dict[str, Any] = {}
    store = trainer.model.memory_store
    for _iteration in range(max(1, int(iterations))):
        started = time.perf_counter()
        report = store.collect_replay_query_indices(
            candidate_bucket_ids=all_anchor_buckets,
            max_queries=max_queries,
            scope="legacy_all_anchor_query_collection_baseline",
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
    return report, _latency_summary(latencies)


def _time_bounded_collection(
    trainer: MarulhoTrainer,
    *,
    max_queries: int,
    iterations: int,
) -> tuple[list[tuple[int, torch.Tensor]], dict[str, Any], dict[str, float]]:
    latencies: list[float] = []
    queries: list[tuple[int, torch.Tensor]] = []
    report: dict[str, Any] = {}
    for _iteration in range(max(1, int(iterations))):
        started = time.perf_counter()
        queries, report = _collect_anchor_replay_queries(
            trainer,
            max_queries=max_queries,
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
    return queries, report, _latency_summary(latencies)


def run_benchmark(
    *,
    anchor_count: int,
    column_latent_dim: int,
    max_queries: int,
    max_candidates: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    cuda_available = torch.cuda.is_available()
    cuda_before = torch.cuda.memory_allocated() if cuda_available else 0
    trainer = _build_trainer(
        anchor_count=anchor_count,
        column_latent_dim=column_latent_dim,
    )

    legacy_report, legacy_latency = _time_legacy_collection(
        trainer,
        max_queries=max_queries,
        iterations=iterations,
    )
    bounded_queries, bounded_report, bounded_latency = _time_bounded_collection(
        trainer,
        max_queries=max_queries,
        iterations=iterations,
    )
    recall_report = _bounded_replay_recall_evaluation(
        trainer,
        bounded_queries,
        max_candidates=max_candidates,
        scope="benchmark_hf_anchor_replay_after_bounded_source_window",
        query_collection_report=bounded_report,
    )
    cuda_after = torch.cuda.memory_allocated() if cuda_available else 0

    expected_recent_queries = list(
        range(
            int(anchor_count) - 1,
            max(-1, int(anchor_count) - int(max_queries) - 1),
            -1,
        )
    )
    legacy_query_indices = [int(index) for index in legacy_report.get("query_indices", [])]
    bounded_query_indices = [int(index) for index in bounded_report.get("query_indices", [])]
    expected_set = set(expected_recent_queries)
    legacy_recent_hits = len(expected_set.intersection(legacy_query_indices))
    bounded_recent_hits = len(expected_set.intersection(bounded_query_indices))
    denominator = max(1, min(int(max_queries), len(expected_recent_queries)))
    legacy_recent_hit_rate = float(legacy_recent_hits / denominator)
    bounded_recent_hit_rate = float(bounded_recent_hits / denominator)
    legacy_mean = float(legacy_latency["mean_ms"])
    bounded_mean = float(bounded_latency["mean_ms"])
    speedup = float(legacy_mean / bounded_mean) if bounded_mean > 0.0 else float("inf")

    return {
        "surface": "bounded_replay_query_anchor_source_window_benchmark.v1",
        "status": "passed"
        if bool(recall_report.get("gate", {}).get("pass"))
        and bounded_recent_hit_rate >= 1.0
        and not bool(bounded_report.get("anchor_source_full_scan"))
        else "failed",
        "parameters": {
            "anchor_count": int(anchor_count),
            "column_latent_dim": int(column_latent_dim),
            "max_queries": int(max_queries),
            "max_candidates": int(max_candidates),
            "iterations": int(iterations),
            "seed": int(seed),
            "anchor_bucket_window_limit": REPLAY_QUERY_ANCHOR_BUCKET_WINDOW_LIMIT,
        },
        "legacy_all_anchor_source": {
            "anchor_bucket_count": int(anchor_count),
            "candidate_bucket_count": int(legacy_report.get("candidate_bucket_count", 0) or 0),
            "candidate_index_available_count": int(
                legacy_report.get("candidate_index_available_count", 0) or 0
            ),
            "candidate_index_count": int(legacy_report.get("candidate_index_count", 0) or 0),
            "query_indices": legacy_query_indices,
            "recent_anchor_query_hit_rate": legacy_recent_hit_rate,
            "anchor_source_full_scan": True,
            "latency": legacy_latency,
        },
        "bounded_anchor_source": {
            "source_window": dict(bounded_report.get("source_window") or {}),
            "candidate_bucket_count": int(bounded_report.get("candidate_bucket_count", 0) or 0),
            "candidate_index_available_count": int(
                bounded_report.get("candidate_index_available_count", 0) or 0
            ),
            "candidate_index_count": int(bounded_report.get("candidate_index_count", 0) or 0),
            "query_indices": bounded_query_indices,
            "recent_anchor_query_hit_rate": bounded_recent_hit_rate,
            "anchor_source_full_scan": bool(bounded_report.get("anchor_source_full_scan")),
            "latency": bounded_latency,
            "speedup_vs_legacy_mean": speedup,
        },
        "quality": {
            "metric": "recent_anchor_query_hit_rate_and_exact_input_recall",
            "expected_recent_query_indices": expected_recent_queries,
            "legacy_recent_anchor_query_hit_rate": legacy_recent_hit_rate,
            "bounded_recent_anchor_query_hit_rate": bounded_recent_hit_rate,
            "bounded_mean_input_pattern_distance": float(
                recall_report.get("mean_input_pattern_distance", float("inf"))
            ),
            "recall_gate_pass": bool(recall_report.get("gate", {}).get("pass")),
            "pass": bool(
                bounded_recent_hit_rate >= 1.0
                and recall_report.get("gate", {}).get("pass")
            ),
        },
        "recall_report": recall_report,
        "device_placement": {
            "archival_storage_device": "cpu",
            "active_replay_compute_device": "cpu",
            "cuda_available": bool(cuda_available),
            "cuda_memory_before_mib": float(cuda_before / (1024 * 1024)),
            "cuda_memory_after_mib": float(cuda_after / (1024 * 1024)),
            "cuda_memory_delta_mib": float((cuda_after - cuda_before) / (1024 * 1024)),
            "gpu_resident_archival_metadata": False,
        },
        "runtime_truth": {
            "runs_live_tick": False,
            "runs_every_token": False,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded replay-query anchor source windows."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-count", type=int, default=4096)
    parser.add_argument("--column-latent-dim", type=int, default=32)
    parser.add_argument("--max-queries", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    report = run_benchmark(
        anchor_count=args.anchor_count,
        column_latent_dim=args.column_latent_dim,
        max_queries=args.max_queries,
        max_candidates=args.max_candidates,
        iterations=args.iterations,
        seed=args.seed,
    )
    write_json_file(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
