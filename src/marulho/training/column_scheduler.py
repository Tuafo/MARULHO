"""Training-owned wake-plan contract for bounded column execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


WAKE_PLAN_CLAIM_BOUNDARY = (
    "training_owned_column_wake_plan_bounds_specialist_execution_without_all_column_sleep_scan"
)
WAKE_PLAN_EXECUTION_CONSUMERS = (
    "predictive_vote",
    "competitive_scoring",
    "predictive_update",
    "predictive_location_update",
    "competitive_homeostasis",
)


def _bounded_id_sample(indices: torch.Tensor, limit: int = 16) -> list[int]:
    if not isinstance(indices, torch.Tensor) or int(indices.numel()) <= 0:
        return []
    return [
        int(value)
        for value in indices.detach().flatten()[: max(0, int(limit))].cpu().tolist()
    ]


@dataclass(frozen=True, slots=True)
class ColumnWakePlan:
    """One scheduler-owned awake mask reused by retained column specialists."""

    mode: str
    total_columns: int
    awake_budget: int
    awake_indices: torch.Tensor
    input_candidate_count: int
    filtered_deep_sleep_count: int
    backfill_candidate_count: int
    deep_sleep_threshold_steps: int
    start_token: int
    backfill_factor: int
    wake_reason: str
    sleep_reason: str | None
    fallback_reason: str | None
    tensor_device: str
    runs_all_columns: bool = False
    claim_boundary: str = WAKE_PLAN_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        indices = self.awake_indices
        if not isinstance(indices, torch.Tensor):
            indices = torch.empty(0, dtype=torch.long)
        indices = indices.detach().flatten().to(dtype=torch.long)
        budget = max(0, min(int(self.awake_budget), int(self.total_columns)))
        if int(indices.numel()) > budget:
            indices = indices[:budget]
        object.__setattr__(self, "awake_indices", indices)
        object.__setattr__(self, "total_columns", max(0, int(self.total_columns)))
        object.__setattr__(self, "awake_budget", budget)
        object.__setattr__(
            self,
            "input_candidate_count",
            max(0, int(self.input_candidate_count)),
        )
        object.__setattr__(
            self,
            "filtered_deep_sleep_count",
            max(0, int(self.filtered_deep_sleep_count)),
        )
        object.__setattr__(
            self,
            "backfill_candidate_count",
            max(0, int(self.backfill_candidate_count)),
        )

    @property
    def awake_count(self) -> int:
        return int(self.awake_indices.numel())

    @property
    def bounded(self) -> bool:
        return self.awake_count <= int(self.awake_budget)

    def candidates(self) -> torch.Tensor:
        return self.awake_indices

    def to_execution_report(self) -> dict[str, Any]:
        return {
            "surface": "column_candidate_sleep_scheduler.v1",
            "mode": str(self.mode),
            "total_columns": int(self.total_columns),
            "awake_budget": int(self.awake_budget),
            "input_candidate_count": int(self.input_candidate_count),
            "output_candidate_count": int(self.awake_count),
            "filtered_deep_sleep_count": int(self.filtered_deep_sleep_count),
            "backfill_candidate_count": int(self.backfill_candidate_count),
            "deep_sleep_threshold_steps": int(self.deep_sleep_threshold_steps),
            "start_token": int(self.start_token),
            "backfill_factor": int(self.backfill_factor),
            "runs_all_columns": bool(self.runs_all_columns),
            "fallback_reason": self.fallback_reason,
            "tensor_device": str(self.tensor_device),
            "claim_boundary": (
                "training_owned_candidate_deep_sleep_filter_skips_deep_sleep_candidates_without_all_column_scan"
            ),
        }

    def to_report(self) -> dict[str, Any]:
        return {
            "surface": "column_wake_plan.v1",
            "mode": str(self.mode),
            "total_columns": int(self.total_columns),
            "awake_budget": int(self.awake_budget),
            "awake_count": int(self.awake_count),
            "input_candidate_count": int(self.input_candidate_count),
            "filtered_deep_sleep_count": int(self.filtered_deep_sleep_count),
            "backfill_candidate_count": int(self.backfill_candidate_count),
            "bounded": bool(self.bounded),
            "runs_all_columns": bool(self.runs_all_columns),
            "wake_reason": str(self.wake_reason),
            "sleep_reason": self.sleep_reason,
            "fallback_reason": self.fallback_reason,
            "tensor_device": str(self.tensor_device),
            "awake_column_ids_sample": _bounded_id_sample(self.awake_indices),
            "execution_consumers": list(WAKE_PLAN_EXECUTION_CONSUMERS),
            "claim_boundary": self.claim_boundary,
        }
