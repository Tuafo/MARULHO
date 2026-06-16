"""Shared read-only projection for Column Runtime truth surfaces."""

from __future__ import annotations

from typing import Any, Mapping


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def build_column_runtime_evidence(
    *,
    runtime_scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project training/core-owned Column Runtime evidence without deciding it."""

    scope = runtime_scope if isinstance(runtime_scope, Mapping) else {}
    report = _as_mapping(scope.get("column_runtime"))
    column_transition_runtime = _as_mapping(scope.get("column_transition_runtime"))
    route_vote_scoring = _as_mapping(
        column_transition_runtime.get("route_vote_scoring")
    )
    route_candidate_bank = _as_mapping(
        column_transition_runtime.get("route_candidate_bank")
    )
    metabolism = _as_mapping(report.get("metabolism"))
    execution = _as_mapping(report.get("execution"))
    candidate_sleep_filter_execution = _as_mapping(
        report.get("candidate_sleep_filter_execution")
    )
    column_wake_plan = _as_mapping(report.get("column_wake_plan"))
    predictive_vote_execution = _as_mapping(report.get("predictive_vote_execution"))
    predictive_update_execution = _as_mapping(
        report.get("predictive_update_execution")
    )
    column_metabolism_execution = _as_mapping(
        report.get("column_metabolism_execution")
    )
    registry = _as_mapping(report.get("registry"))
    scheduler = _as_mapping(report.get("scheduler"))
    recall = _as_mapping(report.get("local_associative_recall"))
    structural_review_queue = _as_mapping(report.get("structural_review_queue"))
    votes = _as_list(report.get("votes"))
    wake_ids = _as_list(column_wake_plan.get("awake_column_ids_sample"))
    scheduler_consumers = _as_list(scheduler.get("execution_consumers"))
    wake_consumers = _as_list(column_wake_plan.get("execution_consumers"))
    return {
        "surface": report.get("surface", "column_runtime_metabolism.v1"),
        "summary_role": "compact_runtime_truth_column_metabolism_and_training_scheduler",
        "total_columns": int(report.get("total_columns", 0) or 0),
        "awake_budget": int(report.get("awake_budget", 0) or 0),
        "awake_count": int(report.get("awake_count", 0) or 0),
        "active_count": int(
            report.get("active_count", report.get("awake_count", 0)) or 0
        ),
        "candidate_count": int(
            report.get("candidate_count", report.get("awake_count", 0)) or 0
        ),
        "idle_count": int(report.get("idle_count", 0) or 0),
        "cached_vote_count": int(report.get("cached_vote_count", 0) or 0),
        "sleeping_count": int(report.get("sleeping_count", 0) or 0),
        "deep_sleeping_count": int(report.get("deep_sleeping_count", 0) or 0),
        "retired_count": int(report.get("retired_count", 0) or 0),
        "metabolism": {
            "source_tensor_device": metabolism.get("source_tensor_device"),
            "report_compute_device": metabolism.get("report_compute_device"),
            "snapshot_tensor_count": int(
                metabolism.get("snapshot_tensor_count", 0) or 0
            ),
            "source_tensor_count": int(
                metabolism.get("source_tensor_count", 0) or 0
            ),
            "materialized_column_state_count": int(
                metabolism.get("materialized_column_state_count", 0) or 0
            ),
            "snapshot_bytes": int(metabolism.get("snapshot_bytes", 0) or 0),
            "device_transfer_count": int(
                metabolism.get("device_transfer_count", 0) or 0
            ),
            "report_latency_ms": metabolism.get("report_latency_ms"),
            "hot_path_effect": metabolism.get("hot_path_effect"),
            "claim_boundary": metabolism.get("claim_boundary"),
        },
        "runs_all_columns": bool(report.get("runs_all_columns", False)),
        "scheduler_mode": scheduler.get("mode"),
        "route_vote_scoring": {
            "surface": route_vote_scoring.get(
                "surface",
                "route_vote_scoring_scope.v1",
            ),
            "mode": route_vote_scoring.get("mode"),
            "kernel_variant": route_vote_scoring.get("kernel_variant"),
            "total_columns": int(route_vote_scoring.get("total_columns", 0) or 0),
            "route_input_rows_scored": int(
                route_vote_scoring.get("route_input_rows_scored", 0) or 0
            ),
            "route_output_candidate_count": int(
                route_vote_scoring.get("route_output_candidate_count", 0) or 0
            ),
            "route_input_fraction": float(
                route_vote_scoring.get("route_input_fraction", 0.0) or 0.0
            ),
            "route_output_fraction": float(
                route_vote_scoring.get("route_output_fraction", 0.0) or 0.0
            ),
            "route_rows_run_all_columns": bool(
                route_vote_scoring.get("route_rows_run_all_columns", False)
            ),
            "bounded_route_scoring": bool(
                route_vote_scoring.get("bounded_route_scoring", False)
            ),
            "candidate_boundary": route_vote_scoring.get("candidate_boundary"),
            "route_input_source": route_vote_scoring.get("route_input_source"),
            "route_scoring_unbounded_reason": route_vote_scoring.get(
                "route_scoring_unbounded_reason"
            ),
            "claim_boundary": route_vote_scoring.get("claim_boundary"),
        },
        "route_candidate_bank": {
            "enabled": bool(route_candidate_bank.get("enabled", False)),
            "ready": bool(route_candidate_bank.get("ready", False)),
            "bank_size": int(route_candidate_bank.get("bank_size", 0) or 0),
            "probe_rows": int(route_candidate_bank.get("probe_rows", 0) or 0),
            "score_rows": int(route_candidate_bank.get("score_rows", 0) or 0),
            "probe_cursor": int(
                route_candidate_bank.get("probe_cursor", 0) or 0
            ),
            "refresh_interval_tokens": int(
                route_candidate_bank.get("refresh_interval_tokens", 0) or 0
            ),
            "refresh_owner": route_candidate_bank.get("refresh_owner"),
            "scored_since_refresh": int(
                route_candidate_bank.get("scored_since_refresh", 0) or 0
            ),
            "seed_count": int(route_candidate_bank.get("seed_count", 0) or 0),
            "refresh_count": int(
                route_candidate_bank.get("refresh_count", 0) or 0
            ),
            "host_refresh_count": int(
                route_candidate_bank.get("host_refresh_count", 0) or 0
            ),
            "device_refresh_count": int(
                route_candidate_bank.get("device_refresh_count", 0) or 0
            ),
            "probe_refresh_count": int(
                route_candidate_bank.get("probe_refresh_count", 0) or 0
            ),
            "probe_device_refresh_count": int(
                route_candidate_bank.get("probe_device_refresh_count", 0) or 0
            ),
            "graph_bypass_count": int(
                route_candidate_bank.get("graph_bypass_count", 0) or 0
            ),
            "fallback_count": int(
                route_candidate_bank.get("fallback_count", 0) or 0
            ),
            "checkpoint_restore_count": int(
                route_candidate_bank.get("checkpoint_restore_count", 0) or 0
            ),
            "last_reason": route_candidate_bank.get("last_reason"),
            "restore_reason": route_candidate_bank.get("restore_reason"),
            "probe_last_reason": route_candidate_bank.get("probe_last_reason"),
            "claim_boundary": route_candidate_bank.get("claim_boundary"),
        },
        "scheduler": {
            "mode": scheduler.get("mode"),
            "wake_plan_mode": scheduler.get("wake_plan_mode"),
            "projected_from_wake_plan": bool(
                scheduler.get("projected_from_wake_plan", False)
            ),
            "promoted_to_execution": bool(
                scheduler.get("promoted_to_execution", False)
            ),
            "execution_scope": scheduler.get("execution_scope"),
            "active_column_fraction": float(
                scheduler.get("active_column_fraction", 0.0) or 0.0
            ),
            "cached_state_policy": scheduler.get("cached_state_policy"),
            "fallback_reason": scheduler.get("fallback_reason"),
            "execution_consumers": [str(value) for value in scheduler_consumers],
        },
        "candidate_sleep_filter_execution": {
            "surface": candidate_sleep_filter_execution.get(
                "surface",
                "column_candidate_sleep_scheduler.v1",
            ),
            "mode": candidate_sleep_filter_execution.get("mode"),
            "total_columns": int(
                candidate_sleep_filter_execution.get("total_columns", 0) or 0
            ),
            "awake_budget": int(
                candidate_sleep_filter_execution.get("awake_budget", 0) or 0
            ),
            "input_candidate_count": int(
                candidate_sleep_filter_execution.get("input_candidate_count", 0)
                or 0
            ),
            "output_candidate_count": int(
                candidate_sleep_filter_execution.get("output_candidate_count", 0)
                or 0
            ),
            "filtered_deep_sleep_count": int(
                candidate_sleep_filter_execution.get("filtered_deep_sleep_count", 0)
                or 0
            ),
            "filtered_memory_pressure_count": int(
                candidate_sleep_filter_execution.get(
                    "filtered_memory_pressure_count",
                    0,
                )
                or 0
            ),
            "filtered_low_usefulness_count": int(
                candidate_sleep_filter_execution.get(
                    "filtered_low_usefulness_count",
                    0,
                )
                or 0
            ),
            "backfill_candidate_count": int(
                candidate_sleep_filter_execution.get("backfill_candidate_count", 0)
                or 0
            ),
            "deep_sleep_threshold_steps": int(
                candidate_sleep_filter_execution.get(
                    "deep_sleep_threshold_steps",
                    0,
                )
                or 0
            ),
            "start_token": int(
                candidate_sleep_filter_execution.get("start_token", 0) or 0
            ),
            "backfill_factor": int(
                candidate_sleep_filter_execution.get("backfill_factor", 0) or 0
            ),
            "memory_pressure_threshold": candidate_sleep_filter_execution.get(
                "memory_pressure_threshold"
            ),
            "memory_pressure_source": candidate_sleep_filter_execution.get(
                "memory_pressure_source"
            ),
            "usefulness_threshold": candidate_sleep_filter_execution.get(
                "usefulness_threshold"
            ),
            "usefulness_source": candidate_sleep_filter_execution.get(
                "usefulness_source"
            ),
            "runs_all_columns": bool(
                candidate_sleep_filter_execution.get("runs_all_columns", False)
            ),
            "fallback_reason": candidate_sleep_filter_execution.get(
                "fallback_reason"
            ),
            "tensor_device": candidate_sleep_filter_execution.get("tensor_device"),
            "claim_boundary": candidate_sleep_filter_execution.get("claim_boundary"),
        },
        "column_wake_plan": {
            "surface": column_wake_plan.get("surface", "column_wake_plan.v1"),
            "mode": column_wake_plan.get("mode"),
            "total_columns": int(column_wake_plan.get("total_columns", 0) or 0),
            "awake_budget": int(column_wake_plan.get("awake_budget", 0) or 0),
            "awake_count": int(column_wake_plan.get("awake_count", 0) or 0),
            "input_candidate_count": int(
                column_wake_plan.get("input_candidate_count", 0) or 0
            ),
            "filtered_deep_sleep_count": int(
                column_wake_plan.get("filtered_deep_sleep_count", 0) or 0
            ),
            "filtered_memory_pressure_count": int(
                column_wake_plan.get("filtered_memory_pressure_count", 0) or 0
            ),
            "filtered_low_usefulness_count": int(
                column_wake_plan.get("filtered_low_usefulness_count", 0) or 0
            ),
            "backfill_candidate_count": int(
                column_wake_plan.get("backfill_candidate_count", 0) or 0
            ),
            "bounded": bool(column_wake_plan.get("bounded", False)),
            "runs_all_columns": bool(
                column_wake_plan.get("runs_all_columns", False)
            ),
            "wake_reason": column_wake_plan.get("wake_reason"),
            "sleep_reason": column_wake_plan.get("sleep_reason"),
            "fallback_reason": column_wake_plan.get("fallback_reason"),
            "memory_pressure_threshold": column_wake_plan.get(
                "memory_pressure_threshold"
            ),
            "memory_pressure_source": column_wake_plan.get(
                "memory_pressure_source"
            ),
            "usefulness_threshold": column_wake_plan.get(
                "usefulness_threshold"
            ),
            "usefulness_source": column_wake_plan.get("usefulness_source"),
            "tensor_device": column_wake_plan.get("tensor_device"),
            "awake_column_ids_sample": [int(value) for value in wake_ids],
            "execution_consumers": [str(value) for value in wake_consumers],
            "claim_boundary": column_wake_plan.get("claim_boundary"),
        },
        "vote_count": len(votes),
        "wake_reasons_sample": [
            str(item.get("wake_reason"))
            for item in votes[:8]
            if isinstance(item, Mapping) and item.get("wake_reason") is not None
        ],
        "registry": {
            "surface": registry.get("surface"),
            "sample_count": len(_as_list(registry.get("columns_sample"))),
            "memory_budget_per_column": registry.get("memory_budget_per_column"),
            "mutates_runtime_state": bool(
                registry.get("mutates_runtime_state", False)
            ),
        },
        "disagreement": dict(report.get("disagreement", {}))
        if isinstance(report.get("disagreement"), Mapping)
        else {},
        "growth_gate": dict(report.get("growth_gate", {}))
        if isinstance(report.get("growth_gate"), Mapping)
        else {},
        "pruning_homeostasis": dict(report.get("pruning_homeostasis", {}))
        if isinstance(report.get("pruning_homeostasis"), Mapping)
        else {},
        "structural_review_queue": {
            "surface": structural_review_queue.get(
                "surface",
                "column_structural_review_queue.v1",
            ),
            "pending_count": int(
                structural_review_queue.get("pending_count", 0) or 0
            ),
            "growth_ticket_count": int(
                structural_review_queue.get("growth_ticket_count", 0) or 0
            ),
            "prune_or_sleep_ticket_count": int(
                structural_review_queue.get("prune_or_sleep_ticket_count", 0) or 0
            ),
            "last_update_token": structural_review_queue.get("last_update_token"),
            "last_update_mode": structural_review_queue.get("last_update_mode"),
            "last_evaluated_column_count": int(
                structural_review_queue.get("last_evaluated_column_count", 0) or 0
            ),
            "last_cached_column_count": int(
                structural_review_queue.get("last_cached_column_count", 0) or 0
            ),
            "update_count": int(
                structural_review_queue.get("update_count", 0) or 0
            ),
            "deferred_update_count": int(
                structural_review_queue.get("deferred_update_count", 0) or 0
            ),
            "last_deferred_reason": structural_review_queue.get(
                "last_deferred_reason"
            ),
            "last_reason": structural_review_queue.get("last_reason"),
            "checkpoint_backed": bool(
                structural_review_queue.get("checkpoint_backed", False)
            ),
            "requires_operator_review": bool(
                structural_review_queue.get("requires_operator_review", False)
            ),
            "mutates_runtime_state": bool(
                structural_review_queue.get("mutates_runtime_state", False)
            ),
            "runs_all_columns": bool(
                structural_review_queue.get("runs_all_columns", False)
            ),
            "next_gate": structural_review_queue.get("next_gate"),
            "claim_boundary": structural_review_queue.get("claim_boundary"),
        },
        "local_associative_recall": {
            "surface": recall.get("surface"),
            "available": bool(recall.get("available", False)),
            "enabled_in_runtime_tick": bool(
                recall.get("enabled_in_runtime_tick", False)
            ),
            "scope": recall.get("scope"),
            "claim_boundary": recall.get("claim_boundary"),
        },
        "execution": {
            "mode": execution.get("mode"),
            "total_columns": int(execution.get("total_columns", 0) or 0),
            "candidate_count": int(execution.get("candidate_count", 0) or 0),
            "scored_column_count": int(
                execution.get("scored_column_count", 0) or 0
            ),
            "runs_all_columns": bool(execution.get("runs_all_columns", False)),
            "route_vote_input_rows_scored": int(
                route_vote_scoring.get("route_input_rows_scored", 0) or 0
            ),
            "route_vote_output_candidate_count": int(
                route_vote_scoring.get("route_output_candidate_count", 0) or 0
            ),
            "route_vote_rows_run_all_columns": bool(
                route_vote_scoring.get("route_rows_run_all_columns", False)
            ),
            "route_vote_bounded_route_scoring": bool(
                route_vote_scoring.get("bounded_route_scoring", False)
            ),
            "state_transition_mode": execution.get("state_transition_mode"),
            "state_transition_column_count": int(
                execution.get("state_transition_column_count", 0) or 0
            ),
            "state_transition_cached_count": int(
                execution.get("state_transition_cached_count", 0) or 0
            ),
            "state_transition_cached_fraction": float(
                execution.get("state_transition_cached_fraction", 0.0) or 0.0
            ),
            "state_transition_runs_all_columns": bool(
                execution.get("state_transition_runs_all_columns", False)
            ),
            "state_transition_step_count": int(
                execution.get("state_transition_step_count", 0) or 0
            ),
            "state_transition_materialize_mode": execution.get(
                "state_transition_materialize_mode"
            ),
            "state_transition_materialize_count": int(
                execution.get("state_transition_materialize_count", 0) or 0
            ),
            "state_transition_materialize_max_age": int(
                execution.get("state_transition_materialize_max_age", 0) or 0
            ),
            "scored_column_fraction": float(
                execution.get("scored_column_fraction", 0.0) or 0.0
            ),
            "homeostasis_update_mode": execution.get("homeostasis_update_mode"),
            "homeostasis_update_count": int(
                execution.get("homeostasis_update_count", 0) or 0
            ),
            "homeostasis_update_fraction": float(
                execution.get("homeostasis_update_fraction", 0.0) or 0.0
            ),
            "input_weight_blend": float(
                execution.get("input_weight_blend", 0.0) or 0.0
            ),
            "input_plasticity_mode": execution.get("input_plasticity_mode"),
            "input_plasticity_update_count": int(
                execution.get("input_plasticity_update_count", 0) or 0
            ),
            "input_plasticity_skip_count": int(
                execution.get("input_plasticity_skip_count", 0) or 0
            ),
            "dormant_input_plasticity_skipped": bool(
                execution.get("dormant_input_plasticity_skipped", False)
            ),
            "sparse_candidate_execution_observed": bool(
                execution.get("sparse_candidate_execution_observed", False)
            ),
            "tensor_device": execution.get("tensor_device"),
            "fallback_reason": execution.get("fallback_reason"),
            "claim_boundary": execution.get("claim_boundary"),
        },
        "predictive_update_execution": {
            "surface": predictive_update_execution.get(
                "surface",
                "predictive_column_update_scheduler.v1",
            ),
            "mode": predictive_update_execution.get("mode"),
            "total_columns": int(
                predictive_update_execution.get("total_columns", 0) or 0
            ),
            "updated_column_count": int(
                predictive_update_execution.get("updated_column_count", 0) or 0
            ),
            "updated_column_fraction": float(
                predictive_update_execution.get("updated_column_fraction", 0.0)
                or 0.0
            ),
            "cached_state_count": int(
                predictive_update_execution.get("cached_state_count", 0) or 0
            ),
            "cached_state_fraction": float(
                predictive_update_execution.get("cached_state_fraction", 0.0)
                or 0.0
            ),
            "location_update_mode": predictive_update_execution.get(
                "location_update_mode"
            ),
            "location_update_count": int(
                predictive_update_execution.get("location_update_count", 0) or 0
            ),
            "location_cached_count": int(
                predictive_update_execution.get("location_cached_count", 0) or 0
            ),
            "location_update_runs_all_columns": bool(
                predictive_update_execution.get(
                    "location_update_runs_all_columns",
                    False,
                )
            ),
            "runs_all_columns": bool(
                predictive_update_execution.get("runs_all_columns", False)
            ),
            "fallback_reason": predictive_update_execution.get("fallback_reason"),
            "tensor_device": predictive_update_execution.get("tensor_device"),
            "claim_boundary": predictive_update_execution.get("claim_boundary"),
        },
        "predictive_vote_execution": {
            "surface": predictive_vote_execution.get(
                "surface",
                "predictive_column_vote_scheduler.v1",
            ),
            "mode": predictive_vote_execution.get("mode"),
            "total_columns": int(
                predictive_vote_execution.get("total_columns", 0) or 0
            ),
            "updated_column_count": int(
                predictive_vote_execution.get("updated_column_count", 0) or 0
            ),
            "updated_column_fraction": float(
                predictive_vote_execution.get("updated_column_fraction", 0.0)
                or 0.0
            ),
            "cached_vote_use_count": int(
                predictive_vote_execution.get("cached_vote_use_count", 0) or 0
            ),
            "cached_vote_fraction": float(
                predictive_vote_execution.get("cached_vote_fraction", 0.0) or 0.0
            ),
            "runs_all_columns": bool(
                predictive_vote_execution.get("runs_all_columns", False)
            ),
            "fallback_reason": predictive_vote_execution.get("fallback_reason"),
            "tensor_device": predictive_vote_execution.get("tensor_device"),
            "claim_boundary": predictive_vote_execution.get("claim_boundary"),
        },
        "column_metabolism_execution": {
            "surface": column_metabolism_execution.get(
                "surface",
                "column_metabolism_state.v1",
            ),
            "total_columns": int(
                column_metabolism_execution.get("total_columns", 0) or 0
            ),
            "updated_column_count": int(
                column_metabolism_execution.get("updated_column_count", 0) or 0
            ),
            "cached_column_count": int(
                column_metabolism_execution.get("cached_column_count", 0) or 0
            ),
            "memory_pressure_source": column_metabolism_execution.get(
                "memory_pressure_source"
            ),
            "usefulness_source": column_metabolism_execution.get(
                "usefulness_source"
            ),
            "runs_all_columns": bool(
                column_metabolism_execution.get("runs_all_columns", False)
            ),
            "tensor_device": column_metabolism_execution.get("tensor_device"),
            "filter": dict(column_metabolism_execution.get("filter", {}))
            if isinstance(column_metabolism_execution.get("filter"), Mapping)
            else {},
            "claim_boundary": column_metabolism_execution.get("claim_boundary"),
        },
        "claim_boundary": (
            "candidate_deep_sleep_memory_pressure_usefulness_filter_scoring_"
            "homeostasis_predictive_update_and_vote_cache_promoted_growth_pruning_remain_reviewed"
        ),
    }
