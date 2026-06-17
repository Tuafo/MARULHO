from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import torch

from marulho.config.model_config import MarulhoConfig
from marulho.reporting.io import write_json_file
from marulho.training.memory_consolidation_runner import (
    build_memory_consolidation_gate,
    collect_assemblies,
    mean_assembly_overlap,
)
from marulho.training.model import MarulhoModel
from marulho.training.runner_utils import set_seed
from marulho.training.trainer import MarulhoTrainer


def _pattern(config: MarulhoConfig, *indices: int) -> torch.Tensor:
    pattern = torch.zeros(config.input_dim, dtype=torch.float32)
    for index in indices:
        pattern[int(index)] = 1.0
    return pattern / pattern.sum().clamp(min=1e-8)


def _mean_reconstruction(
    trainer: MarulhoTrainer,
    items: list[tuple[str, torch.Tensor]],
) -> float:
    if not items:
        return float("nan")
    return float(
        sum(trainer.reconstruction_error(pattern) for _, pattern in items)
        / len(items)
    )


def _bounded_replay_recall_summary(
    trainer: MarulhoTrainer,
    items: list[tuple[str, torch.Tensor]],
    *,
    max_candidates: int,
) -> dict[str, Any]:
    candidate_bucket_ids = sorted(int(bucket_id) for bucket_id in trainer.column_anchors)
    reports: list[dict[str, Any]] = []
    routing_distances: list[float] = []
    input_distances: list[float] = []
    for _raw_window, pattern in items:
        report = trainer.model.memory_store.recall_replay_window(
            query_routing_key=trainer.model.routing_key_from_pattern(pattern),
            query_input_pattern=pattern,
            current_token=trainer.token_count,
            candidate_bucket_ids=candidate_bucket_ids,
            max_candidates=max_candidates,
            strategy="consolidation",
        )
        reports.append(report)
        routing_distance = report.get("best_distance")
        input_distance = report.get("best_input_distance")
        routing_distances.append(
            float(routing_distance)
            if isinstance(routing_distance, (float, int))
            else float("inf")
        )
        input_distances.append(
            float(input_distance)
            if isinstance(input_distance, (float, int))
            else float("inf")
        )

    mean_routing_distance = float(sum(routing_distances) / max(1, len(routing_distances)))
    mean_input_distance = float(sum(input_distances) / max(1, len(input_distances)))
    all_bucket_scoped = all(
        report.get("candidate_scope") == "bucket_indexed_candidate_window"
        and bool(report.get("selection_report", {}).get("bounded_by_bucket_index"))
        for report in reports
    )
    any_live_tick = any(bool(report.get("runs_live_tick")) for report in reports)
    any_mutation = any(bool(report.get("mutates_runtime_state")) for report in reports)
    has_input_recall = all(
        int(report.get("input_pattern_count", 0) or 0) > 0
        for report in reports
    )
    return {
        "surface": "bounded_replay_window_recall_benchmark.v1",
        "candidate_bucket_ids": candidate_bucket_ids,
        "candidate_bucket_count": int(len(candidate_bucket_ids)),
        "mean_routing_key_distance": mean_routing_distance,
        "mean_input_pattern_distance": mean_input_distance,
        "reports": reports,
        "gate": {
            "pass": bool(
                all_bucket_scoped
                and has_input_recall
                and not any_live_tick
                and not any_mutation
                and mean_input_distance <= 0.01
            ),
            "bounded_bucket_scoped": bool(all_bucket_scoped),
            "has_input_recall": bool(has_input_recall),
            "runs_live_tick": bool(any_live_tick),
            "mutates_runtime_state": bool(any_mutation),
            "mean_input_pattern_distance_lte_0_01": bool(
                mean_input_distance <= 0.01
            ),
            "thresholds": {
                "mean_input_pattern_distance_max": 0.01,
            },
        },
    }


def _inject_anchor_replay_pressure(trainer: MarulhoTrainer) -> int:
    anchor_buckets = {int(bucket_id) for bucket_id in trainer.column_anchors}
    touched = 0
    store = trainer.model.memory_store
    for idx, bucket_id in enumerate(store.slow_bucket_ids):
        if bucket_id not in anchor_buckets:
            continue
        store.slow_capture_tag[idx] = max(1.0, float(store.slow_capture_tag[idx]))
        store.slow_local_prp[idx] = max(1.0, float(store.slow_local_prp[idx]))
        store.slow_consolidation_level[idx] = min(
            0.2,
            float(store.slow_consolidation_level[idx]),
        )
        touched += 1
    if touched:
        store._invalidate_summary_cache()
    return touched


def _clear_replay_pressure(trainer: MarulhoTrainer) -> int:
    store = trainer.model.memory_store
    touched = 0
    for idx in range(len(store.slow_buffer)):
        store.slow_capture_tag[idx] = 0.0
        store.slow_local_prp[idx] = 0.0
        store.slow_consolidation_level[idx] = min(
            0.2,
            float(store.slow_consolidation_level[idx]),
        )
        touched += 1
    if touched:
        store._invalidate_summary_cache()
    return touched


def run_trial(
    *,
    seed: int,
    final_selector: str,
    n_columns: int,
    column_latent_dim: int,
    memory_capacity: int,
    slow_memory_archive_interval_tokens: int,
    task_repetitions: int,
    boundary_cycles: int,
    consolidation_cycles: int,
    replay_steps: int,
    candidate_pool: int,
) -> dict[str, Any]:
    set_seed(seed)
    config = MarulhoConfig(
        n_columns=n_columns,
        column_latent_dim=column_latent_dim,
        bootstrap_tokens=0,
        memory_capacity=memory_capacity,
        slow_memory_archive_interval_tokens=slow_memory_archive_interval_tokens,
        eta_competitive=0.05,
        eta_decay=0.0,
        input_weight_blend=0.0,
        micro_sleep_interval_tokens=10**9,
        deep_sleep_interval_tokens=10**9,
        deep_sleep_replay_steps=replay_steps,
        deep_sleep_candidate_pool=candidate_pool,
        enable_learned_chunking=False,
    )
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    task_a = [
        ("alpha memory signal", _pattern(config, 1, 2, 3)),
        ("alpha plastic trace", _pattern(config, 4, 5, 6)),
        ("alpha stable concept", _pattern(config, 7, 8, 9)),
    ]
    task_b = [
        ("beta routing context", _pattern(config, 20, 21, 22)),
        ("beta semantic drift", _pattern(config, 23, 24, 25)),
        ("beta retrieval anchor", _pattern(config, 26, 27, 28)),
    ]

    for _ in range(max(1, int(task_repetitions))):
        for raw_window, pattern in task_a:
            trainer.train_step(pattern, raw_window=raw_window)
    task_a_after_a = _mean_reconstruction(trainer, task_a)
    reference_assemblies = collect_assemblies(
        trainer,
        [pattern for _, pattern in task_a],
    )

    trainer.tag_recent_memories(window_tokens=trainer.token_count, strength=3.0)
    anchored_columns = trainer.capture_recent_memory_anchors(
        window_tokens=trainer.token_count,
        strength=8.0,
    )
    trainer.run_sleep_maintenance(mode="deep", cycles=boundary_cycles)

    for _ in range(max(1, int(task_repetitions))):
        for raw_window, pattern in task_b:
            trainer.train_step(pattern, raw_window=raw_window)
    task_a_after_b = _mean_reconstruction(trainer, task_a)

    pressure_entries = 0
    if final_selector == "global_control":
        trainer.column_anchors.clear()
    elif final_selector == "bounded_positive_pressure":
        pressure_entries = _inject_anchor_replay_pressure(trainer)
    elif final_selector == "bounded_zero_pressure_guard":
        pressure_entries = _clear_replay_pressure(trainer)
    else:
        raise ValueError(f"Unknown final selector: {final_selector}")

    recall_summary = _bounded_replay_recall_summary(
        trainer,
        task_a,
        max_candidates=candidate_pool,
    )

    started = time.perf_counter()
    updates = 0
    cycle_selection_reports: list[dict[str, Any]] = []
    for _ in range(max(0, int(consolidation_cycles))):
        updates += trainer.run_sleep_maintenance(mode="deep", cycles=1)
        cycle_selection_reports.append(
            dict(trainer._last_sleep_replay_selection_report)
        )
    latency_ms = (time.perf_counter() - started) * 1000.0

    task_a_after_consolidation = _mean_reconstruction(trainer, task_a)
    task_a_overlap_after_consolidation = mean_assembly_overlap(
        reference_assemblies,
        collect_assemblies(trainer, [pattern for _, pattern in task_a]),
    )
    gate = build_memory_consolidation_gate(
        task_a_after_a=task_a_after_a,
        task_a_after_b=task_a_after_b,
        task_a_after_consolidation=task_a_after_consolidation,
        task_a_overlap_after_consolidation=task_a_overlap_after_consolidation,
    )
    selection = dict(trainer._last_sleep_replay_selection_report)
    bounded_cycle_count = sum(
        1
        for report in cycle_selection_reports
        if report.get("candidate_scope") == "bucket_indexed_candidate_window"
        and bool(report.get("bounded_by_bucket_index"))
    )
    global_fallback_cycle_count = sum(
        1
        for report in cycle_selection_reports
        if report.get("candidate_scope") == "global_slow_path_score_scan"
    )
    replay_commit_summary = _replay_commit_summary(cycle_selection_reports)
    return {
        "trial": final_selector,
        "seed": int(seed),
        "updates": int(updates),
        "latency_ms": float(latency_ms),
        "anchored_columns": int(anchored_columns),
        "pressure_entries": int(pressure_entries),
        "metrics": {
            "task_a_recon_after_a": task_a_after_a,
            "task_a_recon_after_b": task_a_after_b,
            "task_a_recon_after_consolidation": task_a_after_consolidation,
            "task_a_bounded_replay_recall_routing_key_distance": (
                recall_summary["mean_routing_key_distance"]
            ),
            "task_a_bounded_replay_recall_input_pattern_distance": (
                recall_summary["mean_input_pattern_distance"]
            ),
            "task_a_recovery_delta": (
                task_a_after_b - task_a_after_consolidation
            ),
            "task_a_overlap_after_consolidation": (
                task_a_overlap_after_consolidation
            ),
        },
        "memory_consolidation_gate": gate,
        "bounded_replay_recall": recall_summary,
        "selection": selection,
        "cycle_selection_reports": cycle_selection_reports,
        "replay_commit_summary": replay_commit_summary,
        "bounded_cycle_count": int(bounded_cycle_count),
        "global_fallback_cycle_count": int(global_fallback_cycle_count),
        "device": {
            "model_device": str(trainer.model.device),
            "memory_store": trainer.model.memory_store.device_report(),
        },
    }


def _sum_report_int(reports: list[dict[str, Any]], key: str) -> int:
    return int(sum(int(report.get(key, 0) or 0) for report in reports))


def _sum_report_float(reports: list[dict[str, Any]], key: str) -> float:
    return float(
        sum(
            float(value)
            for report in reports
            for value in [report.get(key)]
            if isinstance(value, (int, float))
        )
    )


def _replay_commit_summary(
    cycle_selection_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    strategies = [
        str(report.get("sleep_replay_commit_strategy"))
        for report in cycle_selection_reports
        if report.get("sleep_replay_commit_strategy")
        and report.get("sleep_replay_commit_strategy") != "not_run"
    ]
    mutating_cycles = [
        report
        for report in cycle_selection_reports
        if bool(report.get("sleep_replay_mutates_runtime_state"))
    ]
    quality_metrics = [
        str(report.get("sleep_replay_quality_metric"))
        for report in cycle_selection_reports
        if report.get("sleep_replay_quality_metric")
    ]
    quality_scopes = [
        str(report.get("sleep_replay_quality_scope"))
        for report in cycle_selection_reports
        if report.get("sleep_replay_quality_scope")
    ]
    return {
        "surface": "bounded_replay_window_commit_summary.v1",
        "commit_strategy": strategies[0] if strategies else "not_run",
        "cycle_count": int(len(cycle_selection_reports)),
        "mutating_cycle_count": int(len(mutating_cycles)),
        "total_applied_count": _sum_report_int(
            cycle_selection_reports,
            "sleep_replay_applied_count",
        ),
        "total_rejected_commit_count": _sum_report_int(
            cycle_selection_reports,
            "sleep_replay_rejected_commit_count",
        ),
        "total_candidate_column_trial_count": _sum_report_int(
            cycle_selection_reports,
            "sleep_replay_candidate_column_trial_count",
        ),
        "max_candidate_column_union_count": int(
            max(
                [
                    int(report.get("sleep_replay_candidate_column_union_count", 0) or 0)
                    for report in cycle_selection_reports
                ]
                or [0]
            )
        ),
        "max_unique_trace_count": int(
            max(
                [
                    int(report.get("sleep_replay_unique_trace_count", 0) or 0)
                    for report in cycle_selection_reports
                ]
                or [0]
            )
        ),
        "total_quality_delta": _sum_report_float(
            cycle_selection_reports,
            "sleep_replay_quality_delta",
        ),
        "quality_metric": quality_metrics[0] if quality_metrics else None,
        "quality_scope": quality_scopes[0] if quality_scopes else None,
        "score_device": (
            cycle_selection_reports[-1].get("score_device")
            if cycle_selection_reports
            else None
        ),
        "archival_storage_device": (
            cycle_selection_reports[-1].get("archival_storage_device")
            if cycle_selection_reports
            else None
        ),
        "runs_live_tick": any(
            bool(report.get("runs_live_tick"))
            for report in cycle_selection_reports
        ),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    trials = [
        run_trial(
            seed=args.seed,
            final_selector=selector,
            n_columns=args.n_columns,
            column_latent_dim=args.column_latent_dim,
            memory_capacity=args.memory_capacity,
            slow_memory_archive_interval_tokens=(
                args.slow_memory_archive_interval_tokens
            ),
            task_repetitions=args.task_repetitions,
            boundary_cycles=args.boundary_cycles,
            consolidation_cycles=args.consolidation_cycles,
            replay_steps=args.replay_steps,
            candidate_pool=args.candidate_pool,
        )
        for selector in (
            "bounded_positive_pressure",
            "bounded_zero_pressure_guard",
            "global_control",
        )
    ]
    return {
        "surface": "bounded_replay_window_benchmark.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "parameters": {
            "seed": int(args.seed),
            "n_columns": int(args.n_columns),
            "column_latent_dim": int(args.column_latent_dim),
            "memory_capacity": int(args.memory_capacity),
            "slow_memory_archive_interval_tokens": int(
                args.slow_memory_archive_interval_tokens
            ),
            "task_repetitions": int(args.task_repetitions),
            "boundary_cycles": int(args.boundary_cycles),
            "consolidation_cycles": int(args.consolidation_cycles),
            "replay_steps": int(args.replay_steps),
            "candidate_pool": int(args.candidate_pool),
        },
        "trials": trials,
        "quality_claim": (
            "bounded input-pattern recall is measured separately from prototype "
            "repair; unanchored deep replay blocks global mutation; prototype "
            "reconstruction remains open until its gate passes"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded replay-window selection evidence.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--n-columns", type=int, default=16)
    parser.add_argument("--column-latent-dim", type=int, default=32)
    parser.add_argument("--memory-capacity", type=int, default=128)
    parser.add_argument("--slow-memory-archive-interval-tokens", type=int, default=4)
    parser.add_argument("--task-repetitions", type=int, default=18)
    parser.add_argument("--boundary-cycles", type=int, default=2)
    parser.add_argument("--consolidation-cycles", type=int, default=4)
    parser.add_argument("--replay-steps", type=int, default=16)
    parser.add_argument("--candidate-pool", type=int, default=32)
    args = parser.parse_args()

    report = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(args.output, report)
    print(f"bounded_replay_window_benchmark={args.output}")
    for trial in report["trials"]:
        selection = trial["selection"]
        metrics = trial["metrics"]
        recall_gate = trial["bounded_replay_recall"]["gate"]
        commit_summary = trial["replay_commit_summary"]
        print(
            f"{trial['trial']}: updates={trial['updates']} "
            f"candidate_scope={selection.get('candidate_scope')} "
            f"bounded_cycles={trial['bounded_cycle_count']} "
            f"global_fallback_cycles={trial['global_fallback_cycle_count']} "
            f"score_count={selection.get('score_count')} "
            f"selected_count={selection.get('selected_count')} "
            f"blocked_fallback={selection.get('global_fallback_blocked_reason')} "
            f"recall_gate_pass={recall_gate['pass']} "
            f"prototype_gate_pass={trial['memory_consolidation_gate']['pass']} "
            f"commit_strategy={commit_summary['commit_strategy']} "
            f"commit_updates={commit_summary['total_applied_count']} "
            f"input_distance={metrics['task_a_bounded_replay_recall_input_pattern_distance']:.8f} "
            f"recovery_delta={metrics['task_a_recovery_delta']:.8f}"
        )


if __name__ == "__main__":
    main()
