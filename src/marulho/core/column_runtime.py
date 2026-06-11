"""Report-only column registry, scheduling, and voting evidence.

This module is the first control-plane slice for a many-column runtime. It
does not change which tensors execute in the hot path and it never mutates
topology. It summarizes existing column tensors into awake/cached/sleep states
so Runtime Truth can show whether MARULHO is moving toward sparse column
metabolism before execution scheduling is promoted.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import torch


def _safe_tensor(value: torch.Tensor | None, *, n_columns: int, fill: float) -> torch.Tensor:
    if isinstance(value, torch.Tensor) and int(value.numel()) == int(n_columns):
        return value.detach().float().flatten()
    return torch.full((int(n_columns),), float(fill), dtype=torch.float32)


def _tensor_to_float_list(value: torch.Tensor, limit: int) -> list[float]:
    if value.is_cuda:
        value = value.detach().cpu()
    return [round(float(item), 6) for item in value[: int(limit)].tolist()]


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


def build_column_runtime_report(
    *,
    n_columns: int,
    prediction_error: torch.Tensor | None,
    confidence: torch.Tensor | None,
    steps_since_win: torch.Tensor | None,
    win_rate_ema: torch.Tensor | None,
    last_winner_ids: Sequence[int] | torch.Tensor | None = None,
    awake_limit: int = 8,
    sleep_after_steps: int = 64,
    deep_sleep_after_steps: int = 512,
    token_count: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    total_columns = max(0, int(n_columns))
    awake_budget = max(0, min(int(awake_limit), total_columns))
    if total_columns <= 0:
        return {
            "surface": "column_runtime_metabolism.v1",
            "artifact_kind": "marulho_column_runtime_metabolism",
            "summary_role": "report_only_scheduler_evidence_not_execution_scheduler",
            "total_columns": 0,
            "awake_budget": 0,
            "awake_count": 0,
            "cached_vote_count": 0,
            "sleeping_count": 0,
            "deep_sleeping_count": 0,
            "votes": [],
            "scheduler": {"mode": "no_columns", "runs_all_columns": False},
            "growth_gate": {"ready": False, "reason": "no_columns"},
            "pruning_homeostasis": {"ready": False, "reason": "no_columns"},
        }

    pred_error = _safe_tensor(prediction_error, n_columns=total_columns, fill=0.0)
    conf = _safe_tensor(confidence, n_columns=total_columns, fill=0.5).clamp(0.0, 1.0)
    steps = _safe_tensor(steps_since_win, n_columns=total_columns, fill=0.0)
    win_rate = _safe_tensor(win_rate_ema, n_columns=total_columns, fill=0.0).clamp(min=0.0)
    last_ids = _ids_to_list(last_winner_ids, n_columns=total_columns)

    surprise_norm = pred_error.clamp(min=0.0, max=1.0)
    confidence_gap = 1.0 - conf
    usefulness = torch.clamp(0.65 * conf + 0.35 * win_rate, min=0.0, max=1.0)
    cost = torch.ones(total_columns, dtype=torch.float32, device=pred_error.device)
    recent = torch.zeros(total_columns, dtype=torch.float32, device=pred_error.device)
    for idx in last_ids:
        recent[idx] = 1.0

    scheduler_score = (
        0.45 * surprise_norm
        + 0.25 * confidence_gap
        + 0.20 * usefulness
        + 0.10 * recent
    )
    top_count = min(max(1, awake_budget), total_columns)
    awake_indices = torch.topk(scheduler_score, k=top_count).indices if awake_budget > 0 else torch.empty(0, dtype=torch.long)
    awake_set = {int(idx) for idx in awake_indices.detach().cpu().tolist()}

    deep_sleep_mask = steps >= float(deep_sleep_after_steps)
    sleep_mask = (steps >= float(sleep_after_steps)) & ~deep_sleep_mask
    cached_vote_mask = (~deep_sleep_mask) & (~sleep_mask) & (surprise_norm < 0.20) & (conf >= 0.55)
    for idx in awake_set:
        if 0 <= idx < total_columns:
            cached_vote_mask[idx] = False

    score_mean = float(scheduler_score.mean().item())
    disagreement = torch.abs(scheduler_score - score_mean)
    vote_ids = list(awake_indices.detach().cpu().tolist())
    cached_ids = torch.nonzero(cached_vote_mask, as_tuple=False).flatten()
    if int(cached_ids.numel()) > 0:
        cached_order = cached_ids[torch.argsort(conf[cached_ids], descending=True)]
        vote_ids.extend(int(idx) for idx in cached_order[: max(0, awake_budget - len(vote_ids))].detach().cpu().tolist())

    votes: list[dict[str, Any]] = []
    for idx in vote_ids[: max(awake_budget, len(awake_set))]:
        column_id = int(idx)
        awake = column_id in awake_set
        sleep_state = "awake" if awake else "cached_vote"
        wake_reason = "recent_or_surprising_column" if awake else "cached_stable_vote"
        votes.append(
            {
                "column_id": column_id,
                "state": sleep_state,
                "confidence": round(float(conf[column_id].item()), 6),
                "prediction_error": round(float(pred_error[column_id].item()), 6),
                "usefulness": round(float(usefulness[column_id].item()), 6),
                "estimated_cost": round(float(cost[column_id].item()), 6),
                "disagreement": round(float(disagreement[column_id].item()), 6),
                "wake_reason": wake_reason,
                "mutates_column": False,
            }
        )

    high_surprise = pred_error > 0.65
    weak_columns = (conf < 0.20) | deep_sleep_mask
    redundant_columns = (win_rate < 1.0 / max(1, total_columns * 4)) & (steps > float(sleep_after_steps))
    growth_ready = bool(int(high_surprise.sum().item()) >= max(2, min(awake_budget, total_columns)))
    prune_ready = bool(int((weak_columns | redundant_columns).sum().item()) > max(1, total_columns // 4))

    return {
        "surface": "column_runtime_metabolism.v1",
        "artifact_kind": "marulho_column_runtime_metabolism",
        "summary_role": "report_only_scheduler_evidence_not_execution_scheduler",
        "source": "core.column_runtime",
        "device": str(device or pred_error.device),
        "token_count": None if token_count is None else int(token_count),
        "total_columns": total_columns,
        "awake_budget": awake_budget,
        "awake_count": int(len(awake_set)),
        "awake_fraction": round(float(len(awake_set)) / float(max(1, total_columns)), 6),
        "cached_vote_count": int(cached_vote_mask.sum().item()),
        "sleeping_count": int(sleep_mask.sum().item()),
        "deep_sleeping_count": int(deep_sleep_mask.sum().item()),
        "cached_votes_allowed": True,
        "runs_all_columns": False,
        "scheduler": {
            "mode": "top_k_surprise_usefulness_cost_scheduler_report",
            "runs_all_columns": False,
            "selection_inputs": [
                "prediction_error",
                "confidence_gap",
                "usefulness",
                "recent_winner",
            ],
            "awake_column_ids": [int(idx) for idx in awake_indices.detach().cpu().tolist()],
            "cached_vote_column_ids": [int(idx) for idx in cached_ids[:awake_budget].detach().cpu().tolist()],
            "fallback_reason": None if awake_budget < total_columns else "awake_budget_reaches_total_columns",
        },
        "votes": votes,
        "disagreement": {
            "mean": round(float(disagreement.mean().item()), 6),
            "max": round(float(disagreement.max().item()), 6),
            "sample": _tensor_to_float_list(disagreement, min(awake_budget, 8)),
        },
        "growth_gate": {
            "ready": growth_ready,
            "candidate_column_count": int(high_surprise.sum().item()) if growth_ready else 0,
            "evidence": "repeated_high_prediction_error_columns" if growth_ready else "insufficient_repeated_surprise",
            "requires_operator_review": True,
            "requires_checkpoint": True,
            "mutates_runtime_state": False,
        },
        "pruning_homeostasis": {
            "ready": prune_ready,
            "weak_or_redundant_column_count": int((weak_columns | redundant_columns).sum().item()),
            "actions_allowed_from_report": [],
            "suggested_slow_path": "isolated_column_prune_or_sleep_review" if prune_ready else "continue_monitoring",
            "mutates_runtime_state": False,
        },
    }
