"""Quality and cost gate for bounded repair replay input preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import torch

from marulho.config.model_config import MarulhoConfig
from marulho.reporting.io import write_json_file
from marulho.training.model import MarulhoModel
from marulho.training.runner_utils import set_seed
from marulho.training.trainer import MarulhoTrainer


def _pattern(config: MarulhoConfig, index: int) -> torch.Tensor:
    pattern = torch.zeros(config.input_dim, dtype=torch.float32)
    width = min(8, int(config.input_dim))
    start = (int(index) * 7) % max(1, int(config.input_dim) - width + 1)
    pattern[start : start + width] = torch.linspace(1.0, 2.0, steps=width)
    return pattern / (pattern.sum() + 1e-8)


def _one_hot_assembly(n_columns: int, bucket_id: int) -> torch.Tensor:
    assembly = torch.zeros(int(n_columns), dtype=torch.float32)
    assembly[int(bucket_id) % int(n_columns)] = 1.0
    return assembly


def _setup_trainer(args: argparse.Namespace) -> tuple[MarulhoTrainer, list[int]]:
    cfg = MarulhoConfig(
        n_columns=int(args.n_columns),
        column_latent_dim=int(args.column_latent_dim),
        bootstrap_tokens=0,
        memory_capacity=max(int(args.entry_count) + 8, int(args.entry_count) * 2),
        deep_sleep_replay_steps=int(args.entry_count),
        deep_sleep_candidate_pool=int(args.candidate_pool),
        micro_sleep_interval_tokens=10**9,
        deep_sleep_interval_tokens=10**9,
        enable_learned_chunking=bool(args.enable_learned_chunking),
    )
    trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
    trainer.memory_warm_started = True
    trainer.token_count = 100
    bucket_ids: list[int] = []
    for index in range(max(1, int(args.entry_count))):
        pattern = _pattern(cfg, index)
        routing_key = trainer.model.routing_key_from_pattern(pattern).detach().cpu()
        bucket_id = int(index % int(cfg.n_columns))
        bucket_ids.append(bucket_id)
        trainer.model.memory_store.update(
            _one_hot_assembly(cfg.n_columns, bucket_id),
            importance=1.0,
            token_count=trainer.token_count + index,
            bucket_id=bucket_id,
            input_pattern=pattern,
            routing_key=routing_key,
            raw_window=f"repair replay trace {index}",
            text=f"repair replay trace {index}",
            capture_tag=1.0,
        )
        trainer.column_anchors[bucket_id] = {
            "prototype": trainer.model.competitive.prototypes[bucket_id].detach().clone(),
            "input_weights": trainer.model.competitive.input_weights[bucket_id].detach().clone(),
            "strength": 1.0,
        }
    trainer.token_count = 100 + max(1, int(args.entry_count))
    trainer.model.memory_store._invalidate_summary_cache()
    return trainer, bucket_ids


def _selected_replay_indices(
    trainer: MarulhoTrainer,
    *,
    bucket_ids: list[int],
    count: int,
    candidate_pool: int,
) -> tuple[list[int], dict[str, Any]]:
    report = trainer.model.memory_store.select_replay_window(
        n=int(count),
        current_token=trainer.token_count,
        candidate_pool=int(candidate_pool),
        strategy="repair",
        candidate_bucket_ids=sorted(set(int(bucket) for bucket in bucket_ids)),
        scope="repair_replay_input_prepare_benchmark",
    )
    return [int(index) for index in report.get("selected_indices", [])], dict(report)


def _measure_input_prepare(
    trainer: MarulhoTrainer,
    indices: list[int],
    prepare: Callable[[torch.Tensor], torch.Tensor],
) -> float:
    started = time.perf_counter()
    for index in indices:
        entry = trainer.model.memory_store.replay_entry(
            int(index),
            current_token=trainer.token_count,
            include_text_payload=False,
        )
        input_pattern = entry.get("input_pattern")
        if not isinstance(input_pattern, torch.Tensor):
            continue
        prepare(input_pattern.to(trainer.model.device))
    return float((time.perf_counter() - started) * 1000.0)


def _mean_anchor_distance(trainer: MarulhoTrainer, bucket_ids: list[int]) -> float:
    distances: list[float] = []
    for index, bucket_id in enumerate(bucket_ids):
        entry = trainer.model.memory_store.replay_entry(
            index,
            current_token=trainer.token_count,
            include_text_payload=False,
        )
        routing_key = entry.get("routing_key")
        if not isinstance(routing_key, torch.Tensor):
            continue
        target = torch.nn.functional.normalize(
            routing_key.to(trainer.model.device),
            dim=0,
        )
        current = torch.nn.functional.normalize(
            trainer.model.competitive.prototypes[int(bucket_id)].detach(),
            dim=0,
        )
        distances.append(float(torch.norm(current - target).item()))
    return float(statistics.fmean(distances)) if distances else float("inf")


def _disturb_anchor_prototypes(trainer: MarulhoTrainer, bucket_ids: list[int]) -> None:
    for offset, bucket_id in enumerate(bucket_ids):
        row = trainer.model.competitive.prototypes[int(bucket_id)].detach()
        trainer.model.competitive.prototypes[int(bucket_id)] = torch.roll(
            row,
            shifts=1 + (offset % 3),
            dims=0,
        )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer, bucket_ids = _setup_trainer(args)
    selected_indices, selection_report = _selected_replay_indices(
        trainer,
        bucket_ids=bucket_ids,
        count=int(args.entry_count),
        candidate_pool=int(args.candidate_pool),
    )

    legacy_latencies: list[float] = []
    bounded_latencies: list[float] = []
    for _ in range(max(1, int(args.prepare_iterations))):
        legacy_latencies.append(
            _measure_input_prepare(
                trainer,
                selected_indices,
                trainer.model.competitive.assembly_from_input,
            )
        )
        bounded_latencies.append(
            _measure_input_prepare(
                trainer,
                selected_indices,
                trainer.model.competitive.prepare_input_for_candidate_routing,
            )
        )

    _disturb_anchor_prototypes(trainer, bucket_ids)
    quality_before = _mean_anchor_distance(trainer, bucket_ids)
    dense_call_count = 0
    original_dense = trainer.model.competitive.assembly_from_input

    def _counted_dense(input_vec: torch.Tensor) -> torch.Tensor:
        nonlocal dense_call_count
        dense_call_count += 1
        return original_dense(input_vec)

    trainer.model.competitive.assembly_from_input = _counted_dense  # type: ignore[method-assign]
    repair_started = time.perf_counter()
    try:
        updates = trainer._sleep_replay("repair")
    finally:
        trainer.model.competitive.assembly_from_input = original_dense  # type: ignore[method-assign]
    repair_latency_ms = float((time.perf_counter() - repair_started) * 1000.0)
    quality_after = _mean_anchor_distance(trainer, bucket_ids)
    repair_report = dict(trainer._last_sleep_replay_selection_report)

    legacy_mean = float(statistics.fmean(legacy_latencies)) if legacy_latencies else 0.0
    bounded_mean = float(statistics.fmean(bounded_latencies)) if bounded_latencies else 0.0
    speedup = legacy_mean / max(1e-9, bounded_mean)
    quality_delta = float(quality_before - quality_after)
    quality_pass = bool(updates > 0 and quality_delta > 0.0)
    report_pass = bool(
        repair_report.get("sleep_replay_unconditional_dense_input_assembly_retired")
        and int(repair_report.get("sleep_replay_dense_input_assembly_fallback_count", -1)) == 0
        and int(repair_report.get("sleep_replay_bounded_input_prepare_count", 0)) >= int(updates)
        and int(dense_call_count) == 0
        and not bool(repair_report.get("sleep_replay_text_payload_loaded"))
        and not bool(repair_report.get("sleep_replay_language_reasoning"))
        and not bool(repair_report.get("global_score_scan"))
    )
    latency_pass = bool(speedup >= float(args.min_prepare_speedup))
    return {
        "surface": "bounded_sleep_repair_replay_input_prepare_benchmark.v1",
        "pass": bool(quality_pass and report_pass and latency_pass),
        "selection_criteria": "anchored_repair_replay_window_with_stored_routing_keys",
        "quality_metric": "mean_anchor_prototype_distance_to_stored_routing_key",
        "latency_metric": "input_prepare_ms_for_selected_replay_entries",
        "n_columns": int(args.n_columns),
        "entry_count": int(args.entry_count),
        "candidate_pool": int(args.candidate_pool),
        "selected_count": int(len(selected_indices)),
        "updates": int(updates),
        "quality": {
            "before": float(quality_before),
            "after": float(quality_after),
            "delta": float(quality_delta),
            "pass": bool(quality_pass),
        },
        "latency_ms": {
            "legacy_dense_prepare_mean": float(legacy_mean),
            "bounded_candidate_prepare_mean": float(bounded_mean),
            "prepare_speedup": float(speedup),
            "repair_replay_latency": float(repair_latency_ms),
        },
        "runtime_truth": {
            "runs_live_tick": False,
            "runs_every_token": False,
            "global_candidate_scan": bool(repair_report.get("global_candidate_scan")),
            "global_score_scan": bool(repair_report.get("global_score_scan")),
            "language_reasoning": bool(repair_report.get("sleep_replay_language_reasoning")),
            "raw_text_payload_loaded": bool(repair_report.get("sleep_replay_text_payload_loaded")),
            "archival_storage_device": "cpu",
            "active_computation_device": str(trainer.model.device),
            "dense_input_assembly_call_count": int(dense_call_count),
            "dense_input_assembly_fallback_count": int(
                repair_report.get("sleep_replay_dense_input_assembly_fallback_count", 0)
            ),
            "bounded_input_prepare_count": int(
                repair_report.get("sleep_replay_bounded_input_prepare_count", 0)
            ),
            "unconditional_dense_input_assembly_retired": bool(
                repair_report.get("sleep_replay_unconditional_dense_input_assembly_retired")
            ),
        },
        "device_placement": {
            "active_computation_device": str(trainer.model.device),
            "memory_store": trainer.model.memory_store.device_report(),
            "cuda_memory_mb": (
                {
                    "allocated": float(torch.cuda.memory_allocated() / (1024.0 * 1024.0)),
                    "max_allocated": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)),
                    "reserved": float(torch.cuda.memory_reserved() / (1024.0 * 1024.0)),
                }
                if torch.cuda.is_available()
                else None
            ),
        },
        "selection_report": selection_report,
        "repair_report": repair_report,
        "gates": {
            "quality_pass": bool(quality_pass),
            "report_pass": bool(report_pass),
            "latency_pass": bool(latency_pass),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-columns", type=int, default=65536)
    parser.add_argument("--column-latent-dim", type=int, default=64)
    parser.add_argument("--entry-count", type=int, default=32)
    parser.add_argument("--candidate-pool", type=int, default=64)
    parser.add_argument("--prepare-iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--min-prepare-speedup", type=float, default=1.0)
    parser.add_argument("--enable-learned-chunking", action="store_true")
    args = parser.parse_args()

    result = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(args.output, result)
    print(
        json.dumps(
            {
                "pass": result["pass"],
                "quality_delta": result["quality"]["delta"],
                "prepare_speedup": result["latency_ms"]["prepare_speedup"],
                "dense_input_assembly_call_count": result["runtime_truth"][
                    "dense_input_assembly_call_count"
                ],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
