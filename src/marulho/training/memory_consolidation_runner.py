from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time
from typing import Any, List, Optional

import torch

from marulho.config.model_config import MarulhoConfig
from marulho.config.presets import get_memory_consolidation_preset, memory_consolidation_preset_names
from marulho.data.pattern_loader import load_train_eval_examples
from marulho.data.rtf_encoder import RTFEncoder
from marulho.reporting.io import write_json_file
from marulho.training.checkpointing import (
    _model_snapshot,
    _restore_model,
    save_trainer_checkpoint,
)
from marulho.training.runner_utils import set_seed
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


def mean_reconstruction_error(trainer: MarulhoTrainer, patterns: List[torch.Tensor]) -> float:
    if not patterns:
        return float("nan")
    values = [trainer.reconstruction_error(pattern) for pattern in patterns]
    return float(sum(values) / len(values))


def collect_assemblies(trainer: MarulhoTrainer, patterns: List[torch.Tensor]) -> List[torch.Tensor]:
    return [trainer.assembly_for_pattern(pattern).float() for pattern in patterns]


def mean_assembly_overlap(
    reference_assemblies: List[torch.Tensor],
    current_assemblies: List[torch.Tensor],
) -> float:
    if not reference_assemblies or not current_assemblies or len(reference_assemblies) != len(current_assemblies):
        return float("nan")

    overlaps: List[float] = []
    for ref_assembly, cmp_assembly in zip(reference_assemblies, current_assemblies):
        overlap = torch.nn.functional.cosine_similarity(
            ref_assembly.unsqueeze(0),
            cmp_assembly.unsqueeze(0),
            dim=1,
        )
        overlaps.append(float(overlap.item()))
    return float(sum(overlaps) / len(overlaps))


def _collect_anchor_replay_queries(
    trainer: MarulhoTrainer,
    *,
    max_queries: int = 16,
) -> list[tuple[int, torch.Tensor]]:
    candidate_bucket_ids = {int(bucket_id) for bucket_id in trainer.column_anchors}
    if not candidate_bucket_ids:
        return []

    queries: list[tuple[int, torch.Tensor]] = []
    store = trainer.model.memory_store
    for index, bucket_id in enumerate(store.slow_bucket_ids):
        if len(queries) >= max(0, int(max_queries)):
            break
        if bucket_id is None or int(bucket_id) not in candidate_bucket_ids:
            continue
        if index >= len(store.slow_input_patterns):
            continue
        pattern = store.slow_input_patterns[index]
        if not isinstance(pattern, torch.Tensor) or int(pattern.numel()) <= 0:
            continue
        queries.append((int(index), pattern.detach().clone().cpu()))
    return queries


def _bounded_replay_recall_evaluation(
    trainer: MarulhoTrainer,
    queries: list[tuple[int, torch.Tensor]],
    *,
    max_candidates: int,
    scope: str,
) -> dict[str, Any]:
    candidate_bucket_ids = sorted(int(bucket_id) for bucket_id in trainer.column_anchors)
    if not queries:
        return {
            "surface": "bounded_replay_window_hf_recall.v1",
            "status": "empty",
            "scope": str(scope),
            "candidate_bucket_ids": candidate_bucket_ids,
            "candidate_bucket_count": int(len(candidate_bucket_ids)),
            "candidate_scope": "bucket_indexed_candidate_window",
            "query_count": 0,
            "max_candidates": int(max_candidates),
            "score_device": "cpu",
            "archival_storage_device": "cpu",
            "runs_live_tick": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "gate": {
                "pass": False,
                "fallback_reason": "no_anchor_replay_queries",
            },
        }

    reports: list[dict[str, Any]] = []
    routing_distances: list[float] = []
    input_distances: list[float] = []
    for _source_index, pattern in queries:
        report = trainer.model.memory_store.recall_replay_window(
            query_routing_key=trainer.model.routing_key_from_pattern(pattern),
            query_input_pattern=pattern,
            current_token=trainer.token_count,
            candidate_bucket_ids=candidate_bucket_ids,
            max_candidates=max_candidates,
            strategy="consolidation",
            scope=scope,
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
        "surface": "bounded_replay_window_hf_recall.v1",
        "status": "recalled" if reports else "empty",
        "scope": str(scope),
        "candidate_bucket_ids": candidate_bucket_ids,
        "candidate_bucket_count": int(len(candidate_bucket_ids)),
        "candidate_scope": (
            "bucket_indexed_candidate_window"
            if all_bucket_scoped
            else str(reports[0].get("candidate_scope") or "unknown")
        ),
        "query_source_indices": [int(index) for index, _pattern in queries],
        "query_count": int(len(queries)),
        "max_candidates": int(max_candidates),
        "mean_routing_key_distance": mean_routing_distance,
        "mean_input_pattern_distance": mean_input_distance,
        "reports": reports,
        "score_device": "cpu",
        "archival_storage_device": "cpu",
        "runs_live_tick": bool(any_live_tick),
        "mutates_runtime_state": bool(any_mutation),
        "applies_plasticity": False,
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


def _run_reconstruction_guarded_sleep_maintenance(
    trainer: MarulhoTrainer,
    eval_patterns: List[torch.Tensor],
    *,
    mode: str,
    cycles: int,
    quality_scope: str,
    tolerance: float = 1e-8,
) -> tuple[int, dict[str, Any]]:
    """Run sleep replay one cycle at a time and roll back harmful reconstruction."""

    started = time.perf_counter()
    requested_cycles = max(0, int(cycles))
    current_best = mean_reconstruction_error(trainer, eval_patterns)
    initial_score = float(current_best)
    accepted_updates = 0
    attempted_updates = 0
    rejected_attempted_updates = 0
    accepted_cycles = 0
    rejected_cycles = 0
    no_update_cycles = 0
    cycle_reports: list[dict[str, Any]] = []
    final_selection_report: dict[str, Any] = dict(
        getattr(trainer, "_last_sleep_replay_selection_report", {})
    )

    for cycle_index in range(requested_cycles):
        before_score = float(current_best)
        snapshot = _model_snapshot(trainer)
        attempted = trainer.run_sleep_maintenance(mode=mode, cycles=1)
        attempted_updates += int(attempted)
        after_score = mean_reconstruction_error(trainer, eval_patterns)
        raw_report = dict(getattr(trainer, "_last_sleep_replay_selection_report", {}))
        quality_accepted = bool(after_score <= before_score + tolerance)
        accepted = quality_accepted
        rollback_reason: str | None = None
        effective_updates = int(attempted)

        if int(attempted) <= 0:
            if accepted:
                no_update_cycles += 1
                current_best = float(after_score)
            else:
                rejected_cycles += 1
                effective_updates = 0
                rollback_reason = "task_a_reconstruction_regression_no_updates_reported"
                _restore_model(trainer, snapshot)
        elif accepted:
            accepted_cycles += 1
            accepted_updates += int(attempted)
            current_best = float(after_score)
        else:
            rejected_cycles += 1
            rejected_attempted_updates += int(attempted)
            effective_updates = 0
            rollback_reason = "task_a_reconstruction_regression"
            _restore_model(trainer, snapshot)

        cycle_report = {
            **raw_report,
            "guard_cycle_index": int(cycle_index),
            "sleep_replay_guard_strategy": "bounded_reconstruction_acceptance",
            "sleep_replay_guard_quality_metric": "mean_reconstruction_error",
            "sleep_replay_guard_quality_scope": str(quality_scope),
            "sleep_replay_guard_before": float(before_score),
            "sleep_replay_guard_after": float(after_score),
            "sleep_replay_guard_delta": float(before_score - after_score),
            "sleep_replay_guard_tolerance": float(tolerance),
            "sleep_replay_commit_accepted": bool(accepted),
            "sleep_replay_rollback_reason": rollback_reason,
            "sleep_replay_attempted_applied_count": int(attempted),
            "sleep_replay_effective_applied_count": int(effective_updates),
            "runs_live_tick": False,
        }
        if not accepted:
            cycle_report = {
                **cycle_report,
                "sleep_replay_applied_count": 0,
                "sleep_replay_updated_column_count": 0,
                "sleep_replay_rejected_by_guard_count": 1,
                "sleep_replay_mutates_runtime_state": False,
                "sleep_replay_applies_plasticity": False,
            }
            trainer._last_sleep_replay_selection_report = dict(cycle_report)
            trainer.model.memory_store.last_replay_selection_report = dict(cycle_report)
            trainer.model.memory_store._invalidate_summary_cache()

        cycle_reports.append(cycle_report)
        final_selection_report = dict(cycle_report)

    latency_ms = (time.perf_counter() - started) * 1000.0
    report = {
        "surface": "reconstruction_guarded_replay_consolidation.v1",
        "mode": str(mode),
        "cycle_count": int(requested_cycles),
        "accepted_cycle_count": int(accepted_cycles),
        "rejected_cycle_count": int(rejected_cycles),
        "no_update_cycle_count": int(no_update_cycles),
        "attempted_update_count": int(attempted_updates),
        "accepted_update_count": int(accepted_updates),
        "rejected_attempted_update_count": int(rejected_attempted_updates),
        "quality_metric": "mean_reconstruction_error",
        "quality_scope": str(quality_scope),
        "quality_initial": float(initial_score),
        "quality_final": float(current_best),
        "quality_delta": float(initial_score - current_best),
        "tolerance": float(tolerance),
        "runs_live_tick": False,
        "score_device": str(trainer.model.device),
        "archival_storage_device": "cpu",
        "latency_ms": float(latency_ms),
        "cycle_reports": cycle_reports,
        "final_selection_report": final_selection_report,
    }
    return accepted_updates, report


def build_memory_consolidation_gate(
    *,
    task_a_after_a: float,
    task_a_after_b: float,
    task_a_after_consolidation: float,
    task_a_overlap_after_consolidation: float,
) -> dict[str, Any]:
    relative_floor = 1e-3
    absolute_degradation_max = 0.01
    raw_relative_degradation = (task_a_after_consolidation - task_a_after_a) / max(1e-8, task_a_after_a)
    floor_adjusted_relative_degradation = (task_a_after_consolidation - task_a_after_a) / max(relative_floor, task_a_after_a)
    absolute_degradation = task_a_after_consolidation - task_a_after_a
    overlap_ok = bool(task_a_overlap_after_consolidation >= 0.50)
    recovery_ok = bool(task_a_after_consolidation <= task_a_after_b + 1e-8)
    use_absolute_gate = bool(task_a_after_a < relative_floor)
    degradation_ok = (
        bool(absolute_degradation <= absolute_degradation_max)
        if use_absolute_gate
        else bool(floor_adjusted_relative_degradation <= 0.05)
    )
    return {
        "pass": bool(degradation_ok and recovery_ok and overlap_ok),
        "task_a_overlap_gte_0_50": overlap_ok,
        "task_a_recovery_nonnegative": recovery_ok,
        "task_a_degradation_ok": degradation_ok,
        "uses_absolute_degradation_gate": use_absolute_gate,
        "thresholds": {
            "task_a_relative_degradation_max": 0.05,
            "task_a_absolute_degradation_max": absolute_degradation_max,
            "task_a_relative_degradation_floor": relative_floor,
            "task_a_overlap_min": 0.50,
        },
        "metrics": {
            "task_a_absolute_degradation_after_consolidation": absolute_degradation,
            "task_a_relative_degradation_after_consolidation": floor_adjusted_relative_degradation,
            "task_a_raw_relative_degradation_after_consolidation": raw_relative_degradation,
        },
    }


def run_memory_consolidation(
    task_a_source: str,
    task_a_hf_config: Optional[str],
    task_a_text_field: str,
    task_a_train_tokens: int,
    task_b_source: str,
    task_b_hf_config: Optional[str],
    task_b_text_field: str,
    task_b_train_tokens: int,
    eval_tokens: int,
    output_dir: Path,
    seed: int,
    n_columns: int,
    column_latent_dim: int,
    memory_capacity: int,
    input_weight_blend: float,
    input_synapse_ltp: float,
    input_synapse_ltd: float,
    input_weight_row_target: float,
    homeostasis_beta: float,
    homeostasis_lr: float,
    slow_mean_decay: float,
    use_winner_local_drift: bool,
    drift_threshold: float,
    micro_sleep_interval_tokens: int,
    micro_sleep_replay_steps: int,
    micro_sleep_candidate_pool: int,
    micro_sleep_memory_blend: float,
    deep_sleep_interval_tokens: int,
    deep_sleep_replay_steps: int,
    deep_sleep_candidate_pool: int,
    deep_sleep_memory_blend: float,
    deep_sleep_cooldown_tokens: int,
    emergency_deep_sleep_cooldown_tokens: int,
    drift_floor_history_tokens: int,
    drift_floor_check_interval_tokens: int,
    drift_floor_window_tokens: int,
    drift_floor_trigger_min_tokens: int,
    drift_floor_rise_tolerance: float,
    prototype_momentum: float,
    task_boundary_tag_strength: float,
    task_boundary_anchor_strength: float,
    task_boundary_consolidation_cycles: int,
    consolidation_mode: str,
    consolidation_cycles: int,
    checkpoint_out: Optional[Path],
    save_plots: bool,
) -> None:
    cfg = MarulhoConfig(
        n_columns=n_columns,
        column_latent_dim=column_latent_dim,
        memory_capacity=memory_capacity,
        input_weight_blend=input_weight_blend,
        input_synapse_ltp=input_synapse_ltp,
        input_synapse_ltd=input_synapse_ltd,
        input_weight_row_target=input_weight_row_target,
        homeostasis_beta=homeostasis_beta,
        homeostasis_lr=homeostasis_lr,
        slow_mean_decay=slow_mean_decay,
        use_winner_local_drift=use_winner_local_drift,
        drift_threshold=drift_threshold,
        micro_sleep_interval_tokens=micro_sleep_interval_tokens,
        micro_sleep_replay_steps=micro_sleep_replay_steps,
        micro_sleep_candidate_pool=micro_sleep_candidate_pool,
        micro_sleep_memory_blend=micro_sleep_memory_blend,
        deep_sleep_interval_tokens=deep_sleep_interval_tokens,
        deep_sleep_replay_steps=deep_sleep_replay_steps,
        deep_sleep_candidate_pool=deep_sleep_candidate_pool,
        deep_sleep_memory_blend=deep_sleep_memory_blend,
        deep_sleep_cooldown_tokens=deep_sleep_cooldown_tokens,
        emergency_deep_sleep_cooldown_tokens=emergency_deep_sleep_cooldown_tokens,
        drift_floor_history_tokens=drift_floor_history_tokens,
        drift_floor_check_interval_tokens=drift_floor_check_interval_tokens,
        drift_floor_window_tokens=drift_floor_window_tokens,
        drift_floor_trigger_min_tokens=drift_floor_trigger_min_tokens,
        drift_floor_rise_tolerance=drift_floor_rise_tolerance,
        prototype_momentum=prototype_momentum,
    )
    encoder = RTFEncoder.from_config(cfg)

    task_a_train_examples, task_a_eval_examples = load_train_eval_examples(
        source=task_a_source,
        source_type="hf",
        hf_config=task_a_hf_config,
        text_field=task_a_text_field,
        encoder=encoder,
        window_size=cfg.window_size,
        train_tokens=task_a_train_tokens,
        eval_tokens=eval_tokens,
    )
    task_b_train_examples, task_b_eval_examples = load_train_eval_examples(
        source=task_b_source,
        source_type="hf",
        hf_config=task_b_hf_config,
        text_field=task_b_text_field,
        encoder=encoder,
        window_size=cfg.window_size,
        train_tokens=task_b_train_tokens,
        eval_tokens=eval_tokens,
    )
    task_a_train = [pattern for _, pattern in task_a_train_examples]
    task_a_eval = [pattern for _, pattern in task_a_eval_examples]
    task_b_train = [pattern for _, pattern in task_b_train_examples]
    task_b_eval = [pattern for _, pattern in task_b_eval_examples]
    if not task_a_train or not task_a_eval:
        raise ValueError("Task A did not produce enough HuggingFace patterns")
    if not task_b_train or not task_b_eval:
        raise ValueError("Task B did not produce enough HuggingFace patterns")

    set_seed(seed)
    model = MarulhoModel(cfg)
    trainer = MarulhoTrainer(model, cfg)

    output_dir.mkdir(parents=True, exist_ok=True)

    for raw_window, pattern in task_a_train_examples:
        trainer.train_step(pattern, raw_window=raw_window)
    task_a_after_a = mean_reconstruction_error(trainer, task_a_eval)
    task_b_before_b = mean_reconstruction_error(trainer, task_b_eval)
    task_a_reference_assemblies = collect_assemblies(trainer, task_a_eval)
    task_b_reference_assemblies = collect_assemblies(trainer, task_b_eval)

    tagged_entries = trainer.tag_recent_memories(
        window_tokens=task_a_train_tokens,
        strength=task_boundary_tag_strength,
    )
    anchored_columns = trainer.capture_recent_memory_anchors(
        window_tokens=task_a_train_tokens,
        strength=task_boundary_anchor_strength,
    )
    boundary_started = time.perf_counter()
    boundary_updates, boundary_guard_report = _run_reconstruction_guarded_sleep_maintenance(
        trainer,
        task_a_eval,
        mode="deep",
        cycles=task_boundary_consolidation_cycles,
        quality_scope="task_a_boundary_eval_reconstruction",
    )
    boundary_latency_ms = (time.perf_counter() - boundary_started) * 1000.0
    task_a_replay_queries = _collect_anchor_replay_queries(
        trainer,
        max_queries=min(16, max(1, eval_tokens)),
    )

    for raw_window, pattern in task_b_train_examples:
        trainer.train_step(pattern, raw_window=raw_window)
    task_a_after_b = mean_reconstruction_error(trainer, task_a_eval)
    task_b_after_b = mean_reconstruction_error(trainer, task_b_eval)
    task_a_overlap_after_b = mean_assembly_overlap(task_a_reference_assemblies, collect_assemblies(trainer, task_a_eval))
    task_b_overlap_after_b = mean_assembly_overlap(task_b_reference_assemblies, collect_assemblies(trainer, task_b_eval))
    memory_before = model.memory_store.summary_stats()
    task_a_replay_recall_after_b = _bounded_replay_recall_evaluation(
        trainer,
        task_a_replay_queries,
        max_candidates=deep_sleep_candidate_pool,
        scope="hf_task_a_anchor_replay_after_task_b",
    )

    consolidation_started = time.perf_counter()
    consolidation_updates, consolidation_guard_report = _run_reconstruction_guarded_sleep_maintenance(
        trainer,
        task_a_eval,
        mode=consolidation_mode,
        cycles=consolidation_cycles,
        quality_scope="task_a_after_task_b_eval_reconstruction",
    )
    consolidation_latency_ms = (time.perf_counter() - consolidation_started) * 1000.0
    task_a_after_consolidation = mean_reconstruction_error(trainer, task_a_eval)
    task_b_after_consolidation = mean_reconstruction_error(trainer, task_b_eval)
    task_a_overlap_after_consolidation = mean_assembly_overlap(task_a_reference_assemblies, collect_assemblies(trainer, task_a_eval))
    task_b_overlap_after_consolidation = mean_assembly_overlap(task_b_reference_assemblies, collect_assemblies(trainer, task_b_eval))
    memory_after = model.memory_store.summary_stats()
    task_a_replay_recall_after_consolidation = _bounded_replay_recall_evaluation(
        trainer,
        task_a_replay_queries,
        max_candidates=deep_sleep_candidate_pool,
        scope="hf_task_a_anchor_replay_after_consolidation",
    )

    gate = build_memory_consolidation_gate(
        task_a_after_a=task_a_after_a,
        task_a_after_b=task_a_after_b,
        task_a_after_consolidation=task_a_after_consolidation,
        task_a_overlap_after_consolidation=task_a_overlap_after_consolidation,
    )
    gate_metrics = gate["metrics"]
    task_a_relative_degradation = float(gate_metrics["task_a_relative_degradation_after_consolidation"])
    task_a_raw_relative_degradation = float(gate_metrics["task_a_raw_relative_degradation_after_consolidation"])
    task_a_absolute_degradation = float(gate_metrics["task_a_absolute_degradation_after_consolidation"])
    memory_consolidation_success = bool(gate["pass"])

    summary = {
        "protocol": "memory_consolidation_sequential_ab_hf",
        "data_setup": {
            "task_a": {
                "source": task_a_source,
                "source_type": "hf",
                "hf_config": task_a_hf_config,
                "text_field": task_a_text_field,
                "train_tokens": task_a_train_tokens,
                "eval_tokens": len(task_a_eval),
            },
            "task_b": {
                "source": task_b_source,
                "source_type": "hf",
                "hf_config": task_b_hf_config,
                "text_field": task_b_text_field,
                "train_tokens": task_b_train_tokens,
                "eval_tokens": len(task_b_eval),
            },
            "n_columns": cfg.n_columns,
            "column_latent_dim": cfg.column_latent_dim,
            "memory_capacity": cfg.memory_capacity,
            "input_weight_blend": cfg.input_weight_blend,
            "slow_mean_decay": cfg.slow_mean_decay,
            "use_winner_local_drift": cfg.use_winner_local_drift,
        },
        "runtime_scope": model.runtime_scope_report(),
        "memory_stats_before_consolidation": memory_before,
        "memory_stats_after_consolidation": memory_after,
        "task_boundary": {
            "tagged_entries": tagged_entries,
            "tag_strength": task_boundary_tag_strength,
            "anchored_columns": anchored_columns,
            "anchor_strength": task_boundary_anchor_strength,
            "boundary_consolidation_cycles": task_boundary_consolidation_cycles,
            "boundary_replay_updates": boundary_updates,
            "boundary_replay_latency_ms": boundary_latency_ms,
            "reconstruction_guard": boundary_guard_report,
        },
        "consolidation": {
            "mode": consolidation_mode,
            "cycles": consolidation_cycles,
            "replay_updates": consolidation_updates,
            "replay_latency_ms": consolidation_latency_ms,
            "reconstruction_guard": consolidation_guard_report,
            "bounded_replay_window_selection": dict(
                consolidation_guard_report.get("final_selection_report", {})
            ),
            "mean_capture_tag_before": float(memory_before.get("mean_capture_tag", 0.0)),
            "mean_capture_tag_after": float(memory_after.get("mean_capture_tag", 0.0)),
            "mean_prp_level_before": float(memory_before.get("mean_prp_level", 0.0)),
            "mean_prp_level_after": float(memory_after.get("mean_prp_level", 0.0)),
            "mean_capture_strength_before": float(memory_before.get("mean_capture_strength", 0.0)),
            "mean_capture_strength_after": float(memory_after.get("mean_capture_strength", 0.0)),
            "mean_consolidation_level_before": float(memory_before.get("mean_consolidation_level", 0.0)),
            "mean_consolidation_level_after": float(memory_after.get("mean_consolidation_level", 0.0)),
        },
        "bounded_replay_recall": {
            "surface": "bounded_replay_window_hf_recall_summary.v1",
            "task_a_anchor_query_count": int(len(task_a_replay_queries)),
            "after_task_b": task_a_replay_recall_after_b,
            "after_consolidation": task_a_replay_recall_after_consolidation,
        },
        "metrics": {
            "task_a_recon_after_a": task_a_after_a,
            "task_b_recon_before_b": task_b_before_b,
            "task_a_recon_after_b": task_a_after_b,
            "task_b_recon_after_b": task_b_after_b,
            "task_a_recon_after_consolidation": task_a_after_consolidation,
            "task_b_recon_after_consolidation": task_b_after_consolidation,
            "task_a_forgetting_delta": task_a_after_b - task_a_after_a,
            "task_a_recovery_delta": task_a_after_b - task_a_after_consolidation,
            "task_b_consolidation_shift": task_b_after_consolidation - task_b_after_b,
            "task_a_overlap_after_b": task_a_overlap_after_b,
            "task_b_overlap_after_b": task_b_overlap_after_b,
            "task_a_overlap_after_consolidation": task_a_overlap_after_consolidation,
            "task_b_overlap_after_consolidation": task_b_overlap_after_consolidation,
            "task_a_absolute_degradation_after_consolidation": task_a_absolute_degradation,
            "task_a_relative_degradation_after_consolidation": task_a_relative_degradation,
            "task_a_raw_relative_degradation_after_consolidation": task_a_raw_relative_degradation,
        },
        "memory_consolidation_gate": gate,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    checkpoint_path: Optional[Path] = None
    if checkpoint_out is not None:
        checkpoint_path = save_trainer_checkpoint(
            checkpoint_out,
            trainer,
            metadata={
                "protocol": "memory_consolidation_sequential_ab_hf",
                "benchmark": "memory_consolidation",
                "task_a_source": task_a_source,
                "task_a_hf_config": task_a_hf_config,
                "task_a_text_field": task_a_text_field,
                "task_b_source": task_b_source,
                "task_b_hf_config": task_b_hf_config,
                "task_b_text_field": task_b_text_field,
                "task_a_train_tokens": len(task_a_train),
                "task_b_train_tokens": len(task_b_train),
                "eval_tokens": len(task_a_eval),
                "consolidation_mode": consolidation_mode,
                "consolidation_cycles": consolidation_cycles,
            },
        )
        summary["checkpoint_path"] = str(checkpoint_path)

    write_json_file(output_dir / "summary.json", summary)
    if save_plots:
        from marulho.reporting.benchmark_plots import plot_memory_consolidation_summary
        plot_memory_consolidation_summary(output_dir, summary)
    print("Memory-consolidation sequential AB summary")
    print(f"task_a_recon_after_a={task_a_after_a:.6f}")
    print(f"task_a_recon_after_b={task_a_after_b:.6f}")
    print(f"task_a_recon_after_consolidation={task_a_after_consolidation:.6f}")
    print(f"task_b_recon_before_b={task_b_before_b:.6f}")
    print(f"task_b_recon_after_b={task_b_after_b:.6f}")
    print(f"task_b_recon_after_consolidation={task_b_after_consolidation:.6f}")
    print(f"task_a_overlap_after_consolidation={task_a_overlap_after_consolidation:.6f}")
    print(f"task_a_relative_degradation_after_consolidation={task_a_relative_degradation:.6f}")
    print(f"consolidation_replay_updates={consolidation_updates}")
    print(f"memory_consolidation_gate_pass={memory_consolidation_success}")
    print(
        "consolidation_guard_accepted_cycles="
        f"{int(consolidation_guard_report.get('accepted_cycle_count', 0))}"
    )
    print(
        "consolidation_guard_rejected_cycles="
        f"{int(consolidation_guard_report.get('rejected_cycle_count', 0))}"
    )
    print(
        "consolidation_guard_accepted_updates="
        f"{int(consolidation_guard_report.get('accepted_update_count', 0))}"
    )
    print(
        "bounded_replay_recall_after_b_gate_pass="
        f"{bool(task_a_replay_recall_after_b.get('gate', {}).get('pass'))}"
    )
    print(
        "bounded_replay_recall_after_consolidation_gate_pass="
        f"{bool(task_a_replay_recall_after_consolidation.get('gate', {}).get('pass'))}"
    )
    print(
        "bounded_replay_recall_after_consolidation_mean_input_distance="
        f"{float(task_a_replay_recall_after_consolidation.get('mean_input_pattern_distance', float('inf'))):.8f}"
    )
    print(f"summary_json={output_dir / 'summary.json'}")
    if checkpoint_path is not None:
        print(f"checkpoint_path={checkpoint_path}")
    if save_plots:
        print(f"memory_consolidation_plot={output_dir / 'memory_consolidation_diagnostics.png'}")


def main() -> None:
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument("--preset", choices=memory_consolidation_preset_names(), default=None)
    preset_args, _ = preset_parser.parse_known_args()
    preset_defaults = get_memory_consolidation_preset(preset_args.preset)

    parser = argparse.ArgumentParser(description="Run the MARULHO memory-consolidation benchmark on HuggingFace datasets")
    parser.add_argument("--preset", choices=memory_consolidation_preset_names(), default=preset_args.preset)
    parser.add_argument("--task-a-source", type=str, default=preset_defaults.get("task_a_source", "ag_news"))
    parser.add_argument("--task-a-hf-config", type=str, default=preset_defaults.get("task_a_hf_config"))
    parser.add_argument("--task-a-text-field", type=str, default=preset_defaults.get("task_a_text_field", "text"))
    parser.add_argument("--task-a-train-tokens", type=int, default=preset_defaults.get("task_a_train_tokens", 2000))
    parser.add_argument("--task-b-source", type=str, default=preset_defaults.get("task_b_source", "wikitext"))
    parser.add_argument("--task-b-hf-config", type=str, default=preset_defaults.get("task_b_hf_config", "wikitext-103-raw-v1"))
    parser.add_argument("--task-b-text-field", type=str, default=preset_defaults.get("task_b_text_field", "text"))
    parser.add_argument("--task-b-train-tokens", type=int, default=preset_defaults.get("task_b_train_tokens", 2000))
    parser.add_argument("--eval-tokens", type=int, default=preset_defaults.get("eval_tokens", 500))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=preset_defaults.get("seed", 7))
    parser.add_argument("--n-columns", type=int, default=preset_defaults.get("n_columns", 100))
    parser.add_argument("--column-latent-dim", type=int, default=preset_defaults.get("column_latent_dim", 256))
    parser.add_argument("--memory-capacity", type=int, default=preset_defaults.get("memory_capacity", 1000))
    parser.add_argument("--input-weight-blend", type=float, default=preset_defaults.get("input_weight_blend", 0.02))
    parser.add_argument("--input-synapse-ltp", type=float, default=preset_defaults.get("input_synapse_ltp", 0.02))
    parser.add_argument("--input-synapse-ltd", type=float, default=preset_defaults.get("input_synapse_ltd", 0.01))
    parser.add_argument("--input-weight-row-target", type=float, default=preset_defaults.get("input_weight_row_target", 1.0))
    parser.add_argument("--homeostasis-beta", type=float, default=preset_defaults.get("homeostasis_beta", 0.01))
    parser.add_argument("--homeostasis-lr", type=float, default=preset_defaults.get("homeostasis_lr", 0.2))
    parser.add_argument("--slow-mean-decay", type=float, default=preset_defaults.get("slow_mean_decay", 0.9999))
    parser.add_argument("--use-winner-local-drift", action="store_true", default=bool(preset_defaults.get("use_winner_local_drift", True)))
    parser.add_argument("--no-winner-local-drift", action="store_true")
    parser.add_argument("--drift-threshold", type=float, default=preset_defaults.get("drift_threshold", 0.02))
    parser.add_argument("--micro-sleep-interval-tokens", type=int, default=preset_defaults.get("micro_sleep_interval_tokens", 200))
    parser.add_argument("--micro-sleep-replay-steps", type=int, default=preset_defaults.get("micro_sleep_replay_steps", 10))
    parser.add_argument("--micro-sleep-candidate-pool", type=int, default=preset_defaults.get("micro_sleep_candidate_pool", 5))
    parser.add_argument("--micro-sleep-memory-blend", type=float, default=preset_defaults.get("micro_sleep_memory_blend", 0.05))
    parser.add_argument("--deep-sleep-interval-tokens", type=int, default=preset_defaults.get("deep_sleep_interval_tokens", 2500))
    parser.add_argument("--deep-sleep-replay-steps", type=int, default=preset_defaults.get("deep_sleep_replay_steps", 200))
    parser.add_argument("--deep-sleep-candidate-pool", type=int, default=preset_defaults.get("deep_sleep_candidate_pool", 100))
    parser.add_argument("--deep-sleep-memory-blend", type=float, default=preset_defaults.get("deep_sleep_memory_blend", 0.20))
    parser.add_argument("--deep-sleep-cooldown-tokens", type=int, default=preset_defaults.get("deep_sleep_cooldown_tokens", 1000))
    parser.add_argument("--emergency-deep-sleep-cooldown-tokens", type=int, default=preset_defaults.get("emergency_deep_sleep_cooldown_tokens", 1000))
    parser.add_argument("--drift-floor-history-tokens", type=int, default=preset_defaults.get("drift_floor_history_tokens", 1000))
    parser.add_argument("--drift-floor-check-interval-tokens", type=int, default=preset_defaults.get("drift_floor_check_interval_tokens", 200))
    parser.add_argument("--drift-floor-window-tokens", type=int, default=preset_defaults.get("drift_floor_window_tokens", 10000))
    parser.add_argument("--drift-floor-trigger-min-tokens", type=int, default=preset_defaults.get("drift_floor_trigger_min_tokens", 1000))
    parser.add_argument("--drift-floor-rise-tolerance", type=float, default=preset_defaults.get("drift_floor_rise_tolerance", 0.0))
    parser.add_argument("--prototype-momentum", type=float, default=preset_defaults.get("prototype_momentum", 0.85))
    parser.add_argument("--task-boundary-tag-strength", type=float, default=preset_defaults.get("task_boundary_tag_strength", 1.5))
    parser.add_argument("--task-boundary-anchor-strength", type=float, default=preset_defaults.get("task_boundary_anchor_strength", 2.0))
    parser.add_argument("--task-boundary-consolidation-cycles", type=int, default=preset_defaults.get("task_boundary_consolidation_cycles", 4))
    parser.add_argument("--consolidation-mode", choices=["micro", "deep"], default=preset_defaults.get("consolidation_mode", "deep"))
    parser.add_argument("--consolidation-cycles", type=int, default=preset_defaults.get("consolidation_cycles", 5))
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    args = parser.parse_args()

    if args.use_winner_local_drift and args.no_winner_local_drift:
        raise ValueError("Choose at most one of --use-winner-local-drift or --no-winner-local-drift")

    use_winner_local_drift = True
    if args.no_winner_local_drift:
        use_winner_local_drift = False

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("reports") / (f"{args.preset}_{stamp}" if args.preset else f"memory_consolidation_{stamp}")
    else:
        output_dir = args.output_dir

    run_memory_consolidation(
        task_a_source=args.task_a_source,
        task_a_hf_config=args.task_a_hf_config,
        task_a_text_field=args.task_a_text_field,
        task_a_train_tokens=args.task_a_train_tokens,
        task_b_source=args.task_b_source,
        task_b_hf_config=args.task_b_hf_config,
        task_b_text_field=args.task_b_text_field,
        task_b_train_tokens=args.task_b_train_tokens,
        eval_tokens=args.eval_tokens,
        output_dir=output_dir,
        seed=args.seed,
        n_columns=args.n_columns,
        column_latent_dim=args.column_latent_dim,
        memory_capacity=args.memory_capacity,
        input_weight_blend=args.input_weight_blend,
        input_synapse_ltp=args.input_synapse_ltp,
        input_synapse_ltd=args.input_synapse_ltd,
        input_weight_row_target=args.input_weight_row_target,
        homeostasis_beta=args.homeostasis_beta,
        homeostasis_lr=args.homeostasis_lr,
        slow_mean_decay=args.slow_mean_decay,
        use_winner_local_drift=use_winner_local_drift,
        drift_threshold=args.drift_threshold,
        micro_sleep_interval_tokens=args.micro_sleep_interval_tokens,
        micro_sleep_replay_steps=args.micro_sleep_replay_steps,
        micro_sleep_candidate_pool=args.micro_sleep_candidate_pool,
        micro_sleep_memory_blend=args.micro_sleep_memory_blend,
        deep_sleep_interval_tokens=args.deep_sleep_interval_tokens,
        deep_sleep_replay_steps=args.deep_sleep_replay_steps,
        deep_sleep_candidate_pool=args.deep_sleep_candidate_pool,
        deep_sleep_memory_blend=args.deep_sleep_memory_blend,
        deep_sleep_cooldown_tokens=args.deep_sleep_cooldown_tokens,
        emergency_deep_sleep_cooldown_tokens=args.emergency_deep_sleep_cooldown_tokens,
        drift_floor_history_tokens=args.drift_floor_history_tokens,
        drift_floor_check_interval_tokens=args.drift_floor_check_interval_tokens,
        drift_floor_window_tokens=args.drift_floor_window_tokens,
        drift_floor_trigger_min_tokens=args.drift_floor_trigger_min_tokens,
        drift_floor_rise_tolerance=args.drift_floor_rise_tolerance,
        prototype_momentum=args.prototype_momentum,
        task_boundary_tag_strength=args.task_boundary_tag_strength,
        task_boundary_anchor_strength=args.task_boundary_anchor_strength,
        task_boundary_consolidation_cycles=args.task_boundary_consolidation_cycles,
        consolidation_mode=args.consolidation_mode,
        consolidation_cycles=args.consolidation_cycles,
        checkpoint_out=args.checkpoint_out,
        save_plots=(not args.no_plots),
    )


if __name__ == "__main__":
    main()
