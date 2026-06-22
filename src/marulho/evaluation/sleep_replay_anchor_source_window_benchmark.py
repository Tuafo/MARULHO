from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from marulho.config.model_config import MarulhoConfig
from marulho.reporting.io import write_json_file
from marulho.training.model import MarulhoModel
from marulho.training.replay_anchor_window import (
    SLEEP_REPLAY_ANCHOR_BUCKET_WINDOW_LIMIT,
    sleep_replay_anchor_bucket_source_window,
)
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
        n_columns=32,
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
        trainer.model.memory_store.slow_local_prp[-1] = 1.0
        trainer.column_anchors[bucket] = {
            "prototype": torch.empty(0, dtype=torch.float32),
            "input_weights": torch.empty(0, dtype=torch.float32),
            "strength": 2.0,
            "captured_at_token": bucket + 1,
            "captured_source_index": bucket,
            "capture_sequence": bucket,
        }
    return trainer


def _selected_bucket_ids(
    trainer: MarulhoTrainer,
    selection_report: dict[str, Any],
) -> list[int]:
    bucket_ids: list[int] = []
    store = trainer.model.memory_store
    for raw_index in selection_report.get("selected_indices", []):
        index = int(raw_index)
        if index < 0 or index >= len(store.slow_bucket_ids):
            continue
        bucket_id = store.slow_bucket_ids[index]
        if bucket_id is not None:
            bucket_ids.append(int(bucket_id))
    return bucket_ids


def _hit_rate(values: list[int], expected: set[int]) -> float:
    if not values:
        return 0.0
    hits = sum(1 for value in values if int(value) in expected)
    return float(hits / len(values))


def _time_legacy_source(
    trainer: MarulhoTrainer,
    *,
    iterations: int,
) -> tuple[list[int], dict[str, float]]:
    source_ids: list[int] = []
    latencies: list[float] = []
    for _iteration in range(max(1, int(iterations))):
        started = time.perf_counter()
        source_ids = sorted(int(bucket_id) for bucket_id in trainer.column_anchors)
        latencies.append((time.perf_counter() - started) * 1000.0)
    return source_ids, _latency_summary(latencies)


def _time_bounded_source(
    trainer: MarulhoTrainer,
    *,
    iterations: int,
) -> tuple[list[int], dict[str, Any], dict[str, float]]:
    source_ids: list[int] = []
    source_report: dict[str, Any] = {}
    latencies: list[float] = []
    for _iteration in range(max(1, int(iterations))):
        started = time.perf_counter()
        source_ids, source_report = sleep_replay_anchor_bucket_source_window(
            trainer,
            mode="deep",
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
    return list(source_ids or []), dict(source_report), _latency_summary(latencies)


def _time_sleep_selection(
    trainer: MarulhoTrainer,
    *,
    bucket_ids: list[int],
    replay_steps: int,
    candidate_pool: int,
    iterations: int,
    scope: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    selection_report: dict[str, Any] = {}
    latencies: list[float] = []
    store = trainer.model.memory_store
    for _iteration in range(max(1, int(iterations))):
        started = time.perf_counter()
        selection_report = store.select_replay_window(
            n=replay_steps,
            current_token=trainer.token_count,
            candidate_pool=candidate_pool,
            strategy="consolidation",
            candidate_bucket_ids=bucket_ids,
            scope=scope,
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
    return dict(selection_report), _latency_summary(latencies)


def run_benchmark(
    *,
    anchor_count: int,
    column_latent_dim: int,
    replay_steps: int,
    candidate_pool: int,
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
    legacy_ids, legacy_source_latency = _time_legacy_source(
        trainer,
        iterations=iterations,
    )
    bounded_ids, bounded_source_report, bounded_source_latency = _time_bounded_source(
        trainer,
        iterations=iterations,
    )
    legacy_selection_report, legacy_selection_latency = _time_sleep_selection(
        trainer,
        bucket_ids=legacy_ids,
        replay_steps=replay_steps,
        candidate_pool=candidate_pool,
        iterations=iterations,
        scope="legacy_all_anchor_sleep_replay_source_baseline",
    )
    bounded_selection_report, bounded_selection_latency = _time_sleep_selection(
        trainer,
        bucket_ids=bounded_ids,
        replay_steps=replay_steps,
        candidate_pool=candidate_pool,
        iterations=iterations,
        scope="bounded_sleep_replay_anchor_source_window",
    )
    cuda_after = torch.cuda.memory_allocated() if cuda_available else 0

    expected_recent_buckets = list(
        range(
            int(anchor_count) - 1,
            max(-1, int(anchor_count) - SLEEP_REPLAY_ANCHOR_BUCKET_WINDOW_LIMIT - 1),
            -1,
        )
    )
    expected_recent_set = set(expected_recent_buckets)
    bounded_selected_buckets = _selected_bucket_ids(trainer, bounded_selection_report)
    legacy_selected_buckets = _selected_bucket_ids(trainer, legacy_selection_report)
    bounded_source_hit_rate = _hit_rate(bounded_ids, expected_recent_set)
    legacy_first_window_hit_rate = _hit_rate(
        legacy_ids[:SLEEP_REPLAY_ANCHOR_BUCKET_WINDOW_LIMIT],
        expected_recent_set,
    )
    bounded_selected_hit_rate = _hit_rate(
        bounded_selected_buckets,
        expected_recent_set,
    )
    legacy_source_mean = float(legacy_source_latency["mean_ms"])
    bounded_source_mean = float(bounded_source_latency["mean_ms"])
    source_speedup = (
        float(legacy_source_mean / bounded_source_mean)
        if bounded_source_mean > 0.0
        else float("inf")
    )
    bounded_pass = bool(
        bounded_ids == expected_recent_buckets
        and bounded_source_hit_rate >= 1.0
        and not bool(bounded_source_report.get("anchor_source_full_scan"))
        and int(bounded_selection_report.get("candidate_bucket_count", 0) or 0)
        <= SLEEP_REPLAY_ANCHOR_BUCKET_WINDOW_LIMIT
        and int(bounded_selection_report.get("selected_count", 0) or 0) > 0
        and bounded_selected_hit_rate >= 1.0
        and not bool(bounded_selection_report.get("global_score_scan"))
        and not bool(bounded_selection_report.get("global_candidate_scan"))
    )

    return {
        "surface": "bounded_sleep_replay_anchor_source_window_benchmark.v1",
        "status": "passed" if bounded_pass else "failed",
        "parameters": {
            "anchor_count": int(anchor_count),
            "column_latent_dim": int(column_latent_dim),
            "replay_steps": int(replay_steps),
            "candidate_pool": int(candidate_pool),
            "iterations": int(iterations),
            "seed": int(seed),
            "anchor_bucket_window_limit": SLEEP_REPLAY_ANCHOR_BUCKET_WINDOW_LIMIT,
        },
        "retired_all_anchor_source": {
            "source_policy": "sorted_all_column_anchors",
            "anchor_bucket_count": int(len(legacy_ids)),
            "first_window_hit_rate": legacy_first_window_hit_rate,
            "anchor_source_full_scan": True,
            "latency": legacy_source_latency,
            "selection_latency": legacy_selection_latency,
            "selection_report": legacy_selection_report,
            "selected_bucket_ids": legacy_selected_buckets,
        },
        "bounded_sleep_anchor_source": {
            "source_window": bounded_source_report,
            "source_bucket_ids": bounded_ids,
            "source_hit_rate": bounded_source_hit_rate,
            "selected_bucket_ids": bounded_selected_buckets,
            "selected_recent_anchor_hit_rate": bounded_selected_hit_rate,
            "anchor_source_full_scan": bool(
                bounded_source_report.get("anchor_source_full_scan")
            ),
            "source_latency": bounded_source_latency,
            "selection_latency": bounded_selection_latency,
            "selection_report": bounded_selection_report,
            "source_speedup_vs_retired_mean": source_speedup,
        },
        "quality": {
            "metric": "newest_anchor_window_hit_rate_and_positive_sleep_selection",
            "expected_recent_bucket_ids": expected_recent_buckets,
            "bounded_source_hit_rate": bounded_source_hit_rate,
            "bounded_selected_recent_anchor_hit_rate": bounded_selected_hit_rate,
            "bounded_selected_count": int(
                bounded_selection_report.get("selected_count", 0) or 0
            ),
            "pass": bounded_pass,
        },
        "device_placement": {
            "archival_storage_device": "cpu",
            "source_window_selection_device": "cpu",
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
        description="Benchmark bounded sleep-replay anchor source windows."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-count", type=int, default=8192)
    parser.add_argument("--column-latent-dim", type=int, default=32)
    parser.add_argument("--replay-steps", type=int, default=16)
    parser.add_argument("--candidate-pool", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    report = run_benchmark(
        anchor_count=args.anchor_count,
        column_latent_dim=args.column_latent_dim,
        replay_steps=args.replay_steps,
        candidate_pool=args.candidate_pool,
        iterations=args.iterations,
        seed=args.seed,
    )
    write_json_file(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
