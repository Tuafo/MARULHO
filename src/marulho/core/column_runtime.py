"""Column registry, scheduling, and voting evidence.

This module is the first control-plane slice for a many-column runtime. It
never mutates topology. It summarizes existing column tensors into
awake/cached/sleep states, while the training/model layer attaches the live
execution-scope evidence that proves whether specialist work actually stayed
bounded on the last tick.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F


def _safe_tensor(
    value: torch.Tensor | None,
    *,
    n_columns: int,
    fill: float,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor) and int(value.numel()) == int(n_columns):
        return value.detach().to(device=device, dtype=torch.float32).flatten()
    return torch.full(
        (int(n_columns),),
        float(fill),
        dtype=torch.float32,
        device=device,
    )


def _column_state_tensors(
    *,
    n_columns: int,
    prediction_error: torch.Tensor | None,
    confidence: torch.Tensor | None,
    steps_since_win: torch.Tensor | None,
    win_rate_ema: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.device]:
    values = (prediction_error, confidence, steps_since_win, win_rate_ema)
    source_device = next(
        (
            value.device
            for value in values
            if isinstance(value, torch.Tensor) and int(value.numel()) == int(n_columns)
        ),
        torch.device("cpu"),
    )
    return (
        _safe_tensor(
            prediction_error,
            n_columns=n_columns,
            fill=0.0,
            device=source_device,
        ),
        _safe_tensor(
            confidence,
            n_columns=n_columns,
            fill=0.5,
            device=source_device,
        ),
        _safe_tensor(
            steps_since_win,
            n_columns=n_columns,
            fill=0.0,
            device=source_device,
        ),
        _safe_tensor(
            win_rate_ema,
            n_columns=n_columns,
            fill=0.0,
            device=source_device,
        ),
        source_device,
    )


def _tensor_to_float_list(value: torch.Tensor, limit: int) -> list[float]:
    if value.is_cuda:
        value = value.detach().cpu()
    return [round(float(item), 6) for item in value[: int(limit)].tolist()]


def _tensor_to_int_list(value: torch.Tensor, limit: int) -> list[int]:
    if value.is_cuda:
        value = value.detach().cpu()
    return [int(item) for item in value[: int(limit)].tolist()]


def _ids_to_list(value: Iterable[int] | torch.Tensor | None, *, n_columns: int) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().flatten().tolist()
    else:
        raw = list(value)
    ids: list[int] = []
    seen: set[int] = set()
    for item in raw:
        idx = int(item)
        if 0 <= idx < int(n_columns) and idx not in seen:
            ids.append(idx)
            seen.add(idx)
    return ids


def bounded_column_associative_recall(
    *,
    query: torch.Tensor,
    memory: torch.Tensor,
    top_k: int = 4,
    beta: float = 8.0,
    max_memory: int = 64,
) -> dict[str, Any]:
    """Run bounded modern-Hopfield-style recall inside one column.

    This is deliberately a local helper, not a global cognition loop. It never
    mutates memory, caps the addressable memory rows, and keeps tensor work on
    the caller's device so CUDA evidence remains observable by the caller.
    """
    if query.dim() != 1:
        raise ValueError("query must be a 1D tensor")
    if memory.dim() != 2:
        raise ValueError("memory must be a 2D tensor")
    if int(memory.shape[1]) != int(query.shape[0]):
        raise ValueError("memory width must match query width")

    memory_budget = max(0, int(max_memory))
    used_memory = memory[:memory_budget].detach()
    if int(used_memory.shape[0]) == 0:
        return {
            "surface": "bounded_column_associative_recall.v1",
            "scope": "single_column_bounded_recall",
            "used_memory_count": 0,
            "top_k": 0,
            "indices": torch.empty(0, dtype=torch.long, device=query.device),
            "weights": torch.empty(0, dtype=query.dtype, device=query.device),
            "recalled": torch.zeros_like(query),
            "entropy": 0.0,
            "mutates_runtime_state": False,
        }

    recall_k = min(max(1, int(top_k)), int(used_memory.shape[0]))
    query_on_memory_device = query.detach().to(device=used_memory.device, dtype=used_memory.dtype)
    query_norm = F.normalize(query_on_memory_device, dim=0)
    memory_norm = F.normalize(used_memory, dim=1)
    scores = torch.mv(memory_norm, query_norm)
    top = torch.topk(scores, k=recall_k)
    weights = torch.softmax(top.values * float(beta), dim=0)
    recalled = torch.sum(used_memory[top.indices] * weights.unsqueeze(1), dim=0)
    entropy = -torch.sum(weights * torch.log(weights.clamp_min(1e-12))).item()
    return {
        "surface": "bounded_column_associative_recall.v1",
        "scope": "single_column_bounded_recall",
        "used_memory_count": int(used_memory.shape[0]),
        "top_k": int(recall_k),
        "indices": top.indices,
        "weights": weights,
        "recalled": recalled.to(device=query.device, dtype=query.dtype),
        "entropy": round(float(entropy), 6),
        "mutates_runtime_state": False,
    }


def _column_role_from_export(
    *,
    column_id: int,
    awake: bool,
    cached_vote: bool,
    sleeping: bool,
    deep_sleep: bool,
    high_surprise: bool,
    last_ids: set[int],
) -> str:
    if deep_sleep:
        return "deep_sleep"
    if sleeping:
        return "sleeping"
    if awake and high_surprise:
        return "surprise_detector"
    if awake and int(column_id) in last_ids:
        return "recent_winner"
    if cached_vote:
        return "stable_cached_vote"
    if awake:
        return "awake_predictor"
    return "idle_column"


def build_column_runtime_report(
    *,
    n_columns: int,
    prediction_error: torch.Tensor | None,
    confidence: torch.Tensor | None,
    steps_since_win: torch.Tensor | None,
    win_rate_ema: torch.Tensor | None,
    last_winner_ids: Sequence[int] | torch.Tensor | None = None,
    prediction_failure_streak: torch.Tensor | None = None,
    estimated_cost: torch.Tensor | None = None,
    memory_pressure: torch.Tensor | None = None,
    memory_pressure_source: str | None = None,
    awake_limit: int = 8,
    sleep_after_steps: int = 64,
    deep_sleep_after_steps: int = 512,
    growth_streak_threshold: int = 3,
    memory_budget_per_column: int | None = None,
    registry_sample_limit: int = 16,
    token_count: int | None = None,
    device: str | None = None,
    execution_awake_indices: Sequence[int] | torch.Tensor | None = None,
    execution_wake_reason: str | None = None,
    execution_sleep_reason: str | None = None,
    execution_scheduler_mode: str | None = None,
    execution_fallback_reason: str | None = None,
    execution_consumers: Sequence[str] | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    total_columns = max(0, int(n_columns))
    awake_budget = max(0, min(int(awake_limit), total_columns))
    if total_columns <= 0:
        return {
            "surface": "column_runtime_metabolism.v1",
            "artifact_kind": "marulho_column_runtime_metabolism",
            "summary_role": "training_owned_scheduler_evidence_without_columns",
            "total_columns": 0,
            "awake_budget": 0,
            "awake_count": 0,
            "cached_vote_count": 0,
            "sleeping_count": 0,
            "deep_sleeping_count": 0,
            "metabolism": {
                "source_tensor_device": str(device or "cpu"),
                "report_compute_device": "cpu",
                "source_tensor_count": 0,
                "snapshot_tensor_count": 0,
                "materialized_column_state_count": 0,
                "snapshot_bytes": 0,
                "device_transfer_count": 0,
                "hot_path_effect": "none_latency_first_runtime_truth_snapshot",
                "claim_boundary": "latency_first_column_status_snapshot_not_hot_path_execution",
            },
            "votes": [],
            "scheduler": {"mode": "no_columns", "runs_all_columns": False},
            "growth_gate": {"ready": False, "reason": "no_columns"},
            "pruning_homeostasis": {"ready": False, "reason": "no_columns"},
            "local_associative_recall": {
                "surface": "bounded_column_associative_recall.v1",
                "scope": "single_column_bounded_recall",
                "available": True,
                "enabled_in_runtime_tick": False,
            },
        }

    pred_error_raw, conf_raw, steps_raw, win_rate_raw, source_device = _column_state_tensors(
        n_columns=total_columns,
        prediction_error=prediction_error,
        confidence=confidence,
        steps_since_win=steps_since_win,
        win_rate_ema=win_rate_ema,
    )
    original_source_device = source_device
    pred_error = pred_error_raw.clamp(min=0.0)
    conf = conf_raw.clamp(0.0, 1.0)
    steps = steps_raw.clamp(min=0.0)
    win_rate = win_rate_raw.clamp(min=0.0)
    cost = _safe_tensor(
        estimated_cost,
        n_columns=total_columns,
        fill=1.0,
        device=source_device,
    ).clamp(0.0, 1.0)
    memory_pressure_tensor = _safe_tensor(
        memory_pressure,
        n_columns=total_columns,
        fill=0.0,
        device=source_device,
    ).clamp(0.0, 1.0)
    memory_pressure_source_value = (
        str(memory_pressure_source)
        if isinstance(memory_pressure_source, str) and memory_pressure_source
        else "training_owned_column_metabolism_state"
        if isinstance(memory_pressure, torch.Tensor)
        and int(memory_pressure.numel()) == total_columns
        else "not_tracked_per_column"
    )
    last_ids = _ids_to_list(last_winner_ids, n_columns=total_columns)
    last_id_set = set(last_ids)
    execution_awake_mask_provided = execution_awake_indices is not None
    execution_awake_ids = (
        _ids_to_list(execution_awake_indices, n_columns=total_columns)
        if execution_awake_mask_provided
        else []
    )

    streak_source_device = None
    streak_tensor = None
    if isinstance(prediction_failure_streak, torch.Tensor) and int(prediction_failure_streak.numel()) == total_columns:
        streak_source_device = prediction_failure_streak.device
        streak_tensor = prediction_failure_streak.detach().to(
            device=source_device,
            dtype=torch.float32,
        ).flatten()
    if source_device.type != "cpu":
        snapshot_rows = [pred_error, conf, steps, win_rate, cost, memory_pressure_tensor]
        if streak_tensor is not None:
            snapshot_rows.append(streak_tensor)
        snapshot = torch.stack(snapshot_rows, dim=0).detach().cpu()
        pred_error = snapshot[0]
        conf = snapshot[1]
        steps = snapshot[2]
        win_rate = snapshot[3]
        cost = snapshot[4]
        memory_pressure_tensor = snapshot[5]
        if streak_tensor is not None:
            streak_tensor = snapshot[6]
        source_device = torch.device("cpu")

    surprise_norm = pred_error.clamp(min=0.0, max=1.0)
    confidence_gap = 1.0 - conf
    usefulness = torch.clamp(0.65 * conf + 0.35 * win_rate, min=0.0, max=1.0)
    recent = torch.zeros(total_columns, dtype=torch.float32, device=source_device)
    for idx in last_ids:
        recent[idx] = 1.0

    scheduler_score = (
        0.45 * surprise_norm
        + 0.25 * confidence_gap
        + 0.20 * usefulness
        + 0.10 * recent
    )
    top_count = min(max(1, awake_budget), total_columns)
    if execution_awake_mask_provided:
        awake_indices = (
            torch.as_tensor(
                execution_awake_ids[:awake_budget],
                dtype=torch.long,
                device=source_device,
            )
            if execution_awake_ids
            else torch.empty(0, dtype=torch.long, device=source_device)
        )
    else:
        awake_indices = (
            torch.topk(scheduler_score, k=top_count).indices
            if awake_budget > 0
            else torch.empty(0, dtype=torch.long, device=source_device)
        )
    awake_set = {int(idx) for idx in awake_indices.detach().cpu().tolist()}

    deep_sleep_mask = steps >= float(deep_sleep_after_steps)
    sleep_mask = (steps >= float(sleep_after_steps)) & ~deep_sleep_mask
    cached_vote_mask = (~deep_sleep_mask) & (~sleep_mask) & (surprise_norm < 0.20) & (conf >= 0.55)
    for idx in awake_set:
        if 0 <= idx < total_columns:
            cached_vote_mask[idx] = False

    score_mean_tensor = scheduler_score.mean()
    score_mean = float(score_mean_tensor.item())
    disagreement = torch.abs(scheduler_score - score_mean)
    vote_ids = list(awake_indices.detach().cpu().tolist())
    cached_ids = torch.nonzero(cached_vote_mask, as_tuple=False).flatten()
    emit_cached_vote_samples = not (
        execution_awake_mask_provided
        and str(execution_scheduler_mode or "") in {"not_run"}
    )
    if int(cached_ids.numel()) > 0 and emit_cached_vote_samples:
        cached_order = cached_ids[torch.argsort(conf[cached_ids], descending=True)]
        vote_ids.extend(int(idx) for idx in cached_order[: max(0, awake_budget - len(vote_ids))].detach().cpu().tolist())
    vote_ids_tensor = (
        torch.as_tensor(vote_ids[: max(awake_budget, len(awake_set))], dtype=torch.long, device=source_device)
        if vote_ids
        else torch.empty(0, dtype=torch.long, device=source_device)
    )

    high_surprise = pred_error > 0.65
    if streak_tensor is not None:
        repeated_surprise = high_surprise & (streak_tensor >= float(max(1, int(growth_streak_threshold))))
        repeated_surprise_available = True
    else:
        repeated_surprise = torch.zeros_like(high_surprise, dtype=torch.bool)
        repeated_surprise_available = False
    weak_columns = (conf < 0.20) | deep_sleep_mask
    redundant_columns = (win_rate < 1.0 / max(1, total_columns * 4)) & (steps > float(sleep_after_steps))
    growth_ready = bool(int(repeated_surprise.sum().item()) >= max(2, min(awake_budget, total_columns)))
    prune_ready = bool(int((weak_columns | redundant_columns).sum().item()) > max(1, total_columns // 4))
    sample_candidates = torch.unique(
        torch.cat(
            (
                awake_indices.detach().to(device=source_device),
                cached_ids[:awake_budget].detach().to(device=source_device),
                torch.nonzero(sleep_mask | deep_sleep_mask, as_tuple=False).flatten()[:awake_budget].detach().to(device=source_device),
            )
        )
    )[: max(0, int(registry_sample_limit))]
    export_ids = torch.unique(
        torch.cat(
            (
                vote_ids_tensor.detach().to(device=source_device),
                sample_candidates.detach().to(device=source_device),
            )
        )
    )
    export_ids_cpu = export_ids.detach().cpu().to(dtype=torch.long)
    export_rows = [
        pred_error,
        conf,
        surprise_norm,
        usefulness,
        cost,
        memory_pressure_tensor,
        win_rate,
        steps,
        confidence_gap,
        disagreement,
        sleep_mask.to(dtype=torch.float32),
        deep_sleep_mask.to(dtype=torch.float32),
        cached_vote_mask.to(dtype=torch.float32),
        high_surprise.to(dtype=torch.float32),
    ]
    if streak_tensor is not None:
        export_rows.append(streak_tensor)
    export_table = (
        torch.stack(export_rows, dim=0)
        .index_select(1, export_ids.to(device=source_device, dtype=torch.long))
        .detach()
        .cpu()
    )
    export_offset_by_id = {
        int(column_id): offset for offset, column_id in enumerate(export_ids_cpu.tolist())
    }
    sample_ids_cpu = sample_candidates.detach().cpu().to(dtype=torch.long)
    vote_ids_cpu = vote_ids_tensor.detach().cpu().to(dtype=torch.long)
    memory_budget = None if memory_budget_per_column is None else max(0, int(memory_budget_per_column))
    if memory_budget is not None and memory_pressure_source_value == "not_tracked_per_column":
        memory_pressure_source_value = "memory_budget_available_usage_not_tracked_per_column"
    votes: list[dict[str, Any]] = []
    for idx in vote_ids_cpu.tolist():
        column_id = int(idx)
        export_offset = export_offset_by_id[column_id]
        awake = column_id in awake_set
        cached_vote = bool(export_table[12, export_offset].item())
        prediction_error_value = round(float(export_table[0, export_offset].item()), 6)
        confidence_value = round(float(export_table[1, export_offset].item()), 6)
        surprise_value = round(float(export_table[2, export_offset].item()), 6)
        sleep_state = "awake" if awake else "cached_vote"
        wake_reason = (
            str(execution_wake_reason or "training_wake_plan_awake")
            if awake and execution_awake_mask_provided
            else "recent_or_surprising_column"
            if awake
            else "cached_stable_vote"
        )
        role = _column_role_from_export(
            column_id=column_id,
            awake=awake,
            cached_vote=cached_vote,
            sleeping=bool(export_table[10, export_offset].item()),
            deep_sleep=bool(export_table[11, export_offset].item()),
            high_surprise=bool(export_table[13, export_offset].item()),
            last_ids=last_id_set,
        )
        votes.append(
            {
                "column_id": column_id,
                "state": sleep_state,
                "role": role,
                "confidence": confidence_value,
                "prediction": {
                    "confidence": confidence_value,
                    "error": prediction_error_value,
                },
                "prediction_error": prediction_error_value,
                "surprise": surprise_value,
                "usefulness": round(float(export_table[3, export_offset].item()), 6),
                "estimated_cost": round(float(export_table[4, export_offset].item()), 6),
                "memory_pressure": round(float(export_table[5, export_offset].item()), 6),
                "memory_pressure_source": memory_pressure_source_value,
                "cached_vote": cached_vote,
                "disagreement": round(float(export_table[9, export_offset].item()), 6),
                "evidence": {
                    "prediction_error": prediction_error_value,
                    "confidence_gap": round(float(export_table[8, export_offset].item()), 6),
                    "win_rate_ema": round(float(export_table[6, export_offset].item()), 6),
                    "steps_since_win": int(export_table[7, export_offset].item()),
                },
                "wake_reason": wake_reason,
                "sleep_reason": (
                    None
                    if awake
                    else str(execution_sleep_reason)
                    if execution_sleep_reason is not None
                    else "stable_low_surprise_cached_vote"
                ),
                "mutates_column": False,
            }
        )
    registry_sample: list[dict[str, Any]] = []
    for sample_offset, column_id in enumerate(_tensor_to_int_list(sample_ids_cpu, max(0, int(registry_sample_limit)))):
        export_offset = export_offset_by_id[column_id]
        sleeping = bool(export_table[10, export_offset].item())
        deep_sleep = bool(export_table[11, export_offset].item())
        cached_vote = bool(export_table[12, export_offset].item())
        high_surprise_state = bool(export_table[13, export_offset].item())
        prediction_error_value = round(float(export_table[0, export_offset].item()), 6)
        confidence_value = round(float(export_table[1, export_offset].item()), 6)
        surprise_value = round(float(export_table[2, export_offset].item()), 6)
        registry_sample.append(
            {
                "column_id": column_id,
                "role": _column_role_from_export(
                    column_id=column_id,
                    awake=column_id in awake_set,
                    cached_vote=cached_vote,
                    sleeping=sleeping,
                    deep_sleep=deep_sleep,
                    high_surprise=high_surprise_state,
                    last_ids=last_id_set,
                ),
                "state": (
                    "deep_sleep"
                    if deep_sleep
                    else "sleeping"
                    if sleeping
                    else "awake"
                    if column_id in awake_set
                    else "cached_vote"
                    if cached_vote
                    else "idle"
                ),
                "local_state": {
                    "prediction": {
                        "confidence": confidence_value,
                        "error": prediction_error_value,
                    },
                    "prediction_error": prediction_error_value,
                    "confidence": confidence_value,
                    "surprise": surprise_value,
                    "usefulness": round(float(export_table[3, export_offset].item()), 6),
                    "estimated_cost": round(float(export_table[4, export_offset].item()), 6),
                    "memory_pressure": round(float(export_table[5, export_offset].item()), 6),
                    "memory_pressure_source": memory_pressure_source_value,
                    "cached_vote": cached_vote,
                    "win_rate_ema": round(float(export_table[6, export_offset].item()), 6),
                    "steps_since_win": int(export_table[7, export_offset].item()),
                    "last_run_token": (
                        None
                        if token_count is None
                        else max(0, int(token_count) - int(export_table[7, export_offset].item()))
                    ),
                    "memory_budget": memory_budget,
                    "prediction_failure_streak": (
                        None
                        if streak_tensor is None
                        else int(export_table[14, export_offset].item())
                    ),
                },
                "mutates_runtime_state": False,
            }
        )
    report_latency_ms = round(float((time.perf_counter() - started_at) * 1000.0), 6)
    source_tensor_count = 6 + (1 if streak_tensor is not None else 0)
    materialized_column_state_count = (
        total_columns if original_source_device.type != "cpu" else int(
            len({int(idx) for idx in sample_ids_cpu.tolist()} | {int(idx) for idx in vote_ids_cpu.tolist()})
        )
    )
    snapshot_bytes = int(
        materialized_column_state_count
        * source_tensor_count
        * torch.empty((), dtype=torch.float32).element_size()
    )
    device_transfer_count = int(original_source_device.type != "cpu")
    if streak_source_device is not None and streak_source_device.type != "cpu":
        device_transfer_count = max(device_transfer_count, 1)
    active_count = int(len(awake_set))
    cached_vote_count = int(cached_vote_mask.sum().item())
    sleeping_count = int(sleep_mask.sum().item())
    deep_sleeping_count = int(deep_sleep_mask.sum().item())
    idle_count = max(
        0,
        total_columns
        - active_count
        - cached_vote_count
        - sleeping_count
        - deep_sleeping_count,
    )

    return {
        "surface": "column_runtime_metabolism.v1",
        "artifact_kind": "marulho_column_runtime_metabolism",
        "summary_role": "training_owned_scheduler_evidence_with_cached_vote_and_deep_sleep_filter_execution",
        "source": "core.column_runtime",
        "device": str(device or source_device),
        "token_count": None if token_count is None else int(token_count),
        "metabolism": {
            "source_tensor_device": str(original_source_device),
            "report_compute_device": str(source_device),
            "source_tensor_count": source_tensor_count,
            "snapshot_tensor_count": source_tensor_count,
            "materialized_column_state_count": materialized_column_state_count,
            "snapshot_bytes": snapshot_bytes,
            "device_transfer_count": device_transfer_count,
            "report_latency_ms": report_latency_ms,
            "hot_path_effect": "none_latency_first_runtime_truth_snapshot",
            "claim_boundary": "latency_first_column_status_snapshot_not_hot_path_execution",
        },
        "total_columns": total_columns,
        "awake_budget": awake_budget,
        "awake_count": active_count,
        "active_count": active_count,
        "candidate_count": active_count,
        "idle_count": idle_count,
        "retired_count": 0,
        "awake_fraction": round(float(active_count) / float(max(1, total_columns)), 6),
        "cached_vote_count": cached_vote_count,
        "sleeping_count": sleeping_count,
        "deep_sleeping_count": deep_sleeping_count,
        "cached_votes_allowed": True,
        "runs_all_columns": False,
        "scheduler": {
            "mode": (
                "training_wake_plan_projection"
                if execution_awake_mask_provided
                else "top_k_surprise_usefulness_cost_latency_first_gate"
            ),
            "wake_plan_mode": (
                None
                if execution_scheduler_mode is None
                else str(execution_scheduler_mode)
            ),
            "projected_from_wake_plan": bool(execution_awake_mask_provided),
            "runs_all_columns": False,
            "promoted_to_execution": True,
            "execution_scope": (
                "candidate_deep_sleep_and_memory_pressure_filter_scoring_homeostasis_"
                "predictive_update_and_vote_cache"
            ),
            "selection_inputs": (
                ["training_owned_column_wake_plan"]
                if execution_awake_mask_provided
                else [
                    "prediction_error",
                    "confidence_gap",
                    "usefulness",
                    "recent_winner",
                ]
            ),
            "awake_column_ids": [int(idx) for idx in awake_indices.detach().cpu().tolist()],
            "cached_vote_column_ids": [int(idx) for idx in cached_ids[:awake_budget].detach().cpu().tolist()],
            "execution_consumers": [
                str(value) for value in (execution_consumers or [])
            ],
            "sleeping_column_ids_sample": _tensor_to_int_list(
                torch.nonzero(sleep_mask, as_tuple=False).flatten(),
                min(awake_budget, 16),
            ),
            "deep_sleeping_column_ids_sample": _tensor_to_int_list(
                torch.nonzero(deep_sleep_mask, as_tuple=False).flatten(),
                min(awake_budget, 16),
            ),
            "active_column_fraction": round(float(len(awake_set)) / float(max(1, total_columns)), 6),
            "cached_state_policy": "stable_low_surprise_columns_may_vote_without_wake",
            "fallback_reason": (
                execution_fallback_reason
                if execution_awake_mask_provided
                else None
                if awake_budget < total_columns
                else "awake_budget_reaches_total_columns"
            ),
        },
        "registry": {
            "surface": "column_registry.v1",
            "role": "report_only_column_registry_snapshot",
            "sample_limit": int(registry_sample_limit),
            "memory_budget_per_column": memory_budget,
            "columns_sample": registry_sample,
            "mutates_runtime_state": False,
        },
        "votes": votes,
        "disagreement": {
            "mean": round(float(disagreement.mean().item()), 6),
            "max": round(float(disagreement.max().item()), 6),
            "sample": _tensor_to_float_list(disagreement, min(awake_budget, 8)),
        },
        "growth_gate": {
            "ready": growth_ready,
            "candidate_column_count": int(repeated_surprise.sum().item()) if growth_ready else 0,
            "one_shot_surprise_count": int(high_surprise.sum().item()),
            "repeated_surprise_count": int(repeated_surprise.sum().item()),
            "repeated_surprise_available": repeated_surprise_available,
            "streak_threshold": int(growth_streak_threshold),
            "candidate_column_ids_sample": _tensor_to_int_list(
                torch.nonzero(repeated_surprise, as_tuple=False).flatten(),
                min(awake_budget, 16),
            ),
            "evidence": (
                "repeated_prediction_failure_streak_columns"
                if growth_ready
                else "missing_prediction_failure_streak"
                if not repeated_surprise_available
                else "insufficient_repeated_surprise"
            ),
            "requires_operator_review": True,
            "requires_checkpoint": True,
            "binding_growth_trial_available": growth_ready,
            "next_gate": (
                "explicit_binding_growth_trial_design"
                if growth_ready
                else "collect_repeated_prediction_failure"
            ),
            "reversible_path": "isolated_binding_trial_then_operator_checkpoint_transaction",
            "mutates_runtime_state": False,
        },
        "pruning_homeostasis": {
            "ready": prune_ready,
            "weak_or_redundant_column_count": int((weak_columns | redundant_columns).sum().item()),
            "actions_allowed_from_report": [],
            "suggested_actions_sample": [
                {
                    "column_id": column_id,
                    "action": "archive_after_review"
                    if bool(deep_sleep_mask[column_id].item())
                    else "sleep_or_weaken_after_review",
                    "requires_checkpoint": True,
                    "mutates_runtime_state": False,
                }
                for column_id in _tensor_to_int_list(
                    torch.nonzero(weak_columns | redundant_columns, as_tuple=False).flatten(),
                    min(awake_budget, 16),
                )
            ],
            "suggested_slow_path": "isolated_column_prune_or_sleep_review" if prune_ready else "continue_monitoring",
            "mutates_runtime_state": False,
        },
        "local_associative_recall": {
            "surface": "bounded_column_associative_recall.v1",
            "scope": "single_column_bounded_recall",
            "available": True,
            "enabled_in_runtime_tick": False,
            "max_memory_per_call": 64,
            "claim_boundary": "bounded_column_memory_recall_helper_not_global_mind_or_language_model",
        },
    }
