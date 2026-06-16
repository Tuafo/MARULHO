from __future__ import annotations

from typing import Any

import torch


class ColumnMetabolismState:
    """Per-column scheduler cost and memory-pressure state.

    The trainer updates this on the awake mask only. Non-awake columns keep
    cached values until they are routed again, so this state can feed scheduler
    decisions without an all-column hot-path scan.
    """

    def __init__(
        self,
        *,
        n_columns: int,
        device: torch.device | str,
        ema_decay: float = 0.90,
    ) -> None:
        self.n_columns = max(0, int(n_columns))
        self.device = torch.device(device)
        self.ema_decay = max(0.0, min(0.999, float(ema_decay)))
        self.estimated_cost = torch.zeros(
            self.n_columns,
            dtype=torch.float32,
            device=self.device,
        )
        self.memory_pressure = torch.zeros(
            self.n_columns,
            dtype=torch.float32,
            device=self.device,
        )
        self.last_update_step = torch.zeros(
            self.n_columns,
            dtype=torch.long,
            device=self.device,
        )
        self.last_update_count = 0
        self.last_cached_count = self.n_columns
        self.last_memory_pressure_source = "not_run"
        self.last_filter_report: dict[str, Any] = self._empty_filter_report()

    def _candidate_indices(self, candidates: torch.Tensor | None) -> torch.Tensor:
        if candidates is None or int(candidates.numel()) <= 0 or self.n_columns <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        ids = candidates.detach().flatten().to(device=self.device, dtype=torch.long)
        ids = ids[(ids >= 0) & (ids < int(self.n_columns))]
        return ids

    def _empty_filter_report(self) -> dict[str, Any]:
        return {
            "surface": "column_memory_pressure_filter.v1",
            "mode": "not_run",
            "input_candidate_count": 0,
            "output_candidate_count": 0,
            "filtered_memory_pressure_count": 0,
            "threshold": None,
            "memory_pressure_source": str(self.last_memory_pressure_source),
            "fallback_reason": None,
            "runs_all_columns": False,
            "tensor_device": str(self.device),
            "claim_boundary": (
                "training_owned_candidate_memory_pressure_filter_uses_cached_candidate_state_without_all_column_scan"
            ),
        }

    def record_awake(
        self,
        candidates: torch.Tensor | None,
        *,
        token_count: int,
        awake_budget: int,
        input_candidate_count: int,
        memory_consolidation: torch.Tensor | None = None,
    ) -> None:
        ids = self._candidate_indices(candidates)
        count = int(ids.numel())
        self.last_update_count = count
        self.last_cached_count = max(0, int(self.n_columns) - count)
        if count <= 0:
            self.last_memory_pressure_source = "no_awake_candidates"
            return

        route_pressure = min(
            1.0,
            max(0.0, float(input_candidate_count) / float(max(1, self.n_columns))),
        )
        local_pressure = min(
            1.0,
            max(0.0, float(count) / float(max(1, int(awake_budget)))),
        )
        cost_value = torch.full(
            (count,),
            0.5 * route_pressure + 0.5 * local_pressure,
            dtype=torch.float32,
            device=self.device,
        )
        old_cost = self.estimated_cost.index_select(0, ids)
        self.estimated_cost[ids] = (
            self.ema_decay * old_cost + (1.0 - self.ema_decay) * cost_value
        ).clamp(0.0, 1.0)

        if (
            isinstance(memory_consolidation, torch.Tensor)
            and int(memory_consolidation.numel()) == int(self.n_columns)
        ):
            consolidation = memory_consolidation.detach().to(
                device=self.device,
                dtype=torch.float32,
            )
            pressure_value = 1.0 - consolidation.index_select(0, ids).clamp(0.0, 1.0)
            old_pressure = self.memory_pressure.index_select(0, ids)
            self.memory_pressure[ids] = (
                self.ema_decay * old_pressure
                + (1.0 - self.ema_decay) * pressure_value
            ).clamp(0.0, 1.0)
            self.last_memory_pressure_source = "memory_store_bucket_consolidation_gap"
        else:
            self.last_memory_pressure_source = "no_memory_store_bucket_evidence"
        self.last_update_step[ids] = int(token_count)

    def filter_candidates(
        self,
        candidates: torch.Tensor,
        *,
        target_count: int,
        threshold: float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        ids = self._candidate_indices(candidates)
        candidate_count = int(ids.numel())
        target = max(0, min(int(target_count), int(self.n_columns)))
        threshold_value = max(0.0, min(1.0, float(threshold)))
        if candidate_count <= 0 or target <= 0:
            report = {
                **self._empty_filter_report(),
                "mode": "candidate_memory_pressure_filter_empty",
                "threshold": threshold_value,
                "fallback_reason": "no_candidates",
            }
            self.last_filter_report = report
            return ids[:0], report

        pressure = self.memory_pressure.index_select(0, ids)
        keep_mask = pressure <= threshold_value
        kept = ids[keep_mask]
        filtered_count = candidate_count - int(kept.numel())
        if int(kept.numel()) <= 0:
            output = ids[:target]
            fallback_reason = "all_candidates_over_memory_pressure_threshold"
            mode = "candidate_memory_pressure_filter_fallback"
        else:
            output = kept[:target]
            fallback_reason = (
                None
                if int(output.numel()) >= min(target, candidate_count)
                else "insufficient_low_pressure_candidates"
            )
            mode = "candidate_memory_pressure_filter"
        report = {
            **self._empty_filter_report(),
            "mode": mode,
            "input_candidate_count": candidate_count,
            "output_candidate_count": int(output.numel()),
            "filtered_memory_pressure_count": max(0, int(filtered_count)),
            "threshold": threshold_value,
            "memory_pressure_source": str(self.last_memory_pressure_source),
            "fallback_reason": fallback_reason,
        }
        self.last_filter_report = report
        return output, report

    def report(self) -> dict[str, Any]:
        return {
            "surface": "column_metabolism_state.v1",
            "total_columns": int(self.n_columns),
            "updated_column_count": int(self.last_update_count),
            "cached_column_count": int(self.last_cached_count),
            "memory_pressure_source": str(self.last_memory_pressure_source),
            "runs_all_columns": False,
            "tensor_device": str(self.device),
            "filter": dict(self.last_filter_report),
            "claim_boundary": (
                "training_owned_awake_mask_cost_and_memory_pressure_updates_without_all_column_scan"
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "estimated_cost": self.estimated_cost.detach().clone().cpu(),
            "memory_pressure": self.memory_pressure.detach().clone().cpu(),
            "last_update_step": self.last_update_step.detach().clone().cpu(),
            "last_update_count": int(self.last_update_count),
            "last_cached_count": int(self.last_cached_count),
            "last_memory_pressure_source": str(self.last_memory_pressure_source),
            "last_filter_report": dict(self.last_filter_report),
            "ema_decay": float(self.ema_decay),
        }

    def load_state_dict(self, snapshot: dict[str, Any]) -> None:
        for name in ("estimated_cost", "memory_pressure"):
            value = snapshot.get(name)
            if isinstance(value, torch.Tensor) and int(value.numel()) == self.n_columns:
                setattr(
                    self,
                    name,
                    value.detach().to(device=self.device, dtype=torch.float32).flatten(),
                )
        value = snapshot.get("last_update_step")
        if isinstance(value, torch.Tensor) and int(value.numel()) == self.n_columns:
            self.last_update_step = value.detach().to(
                device=self.device,
                dtype=torch.long,
            ).flatten()
        self.last_update_count = int(snapshot.get("last_update_count", 0) or 0)
        self.last_cached_count = int(
            snapshot.get("last_cached_count", self.n_columns) or 0
        )
        self.last_memory_pressure_source = str(
            snapshot.get("last_memory_pressure_source", "not_run")
        )
        filter_report = snapshot.get("last_filter_report")
        self.last_filter_report = (
            dict(filter_report)
            if isinstance(filter_report, dict)
            else self._empty_filter_report()
        )
        self.ema_decay = max(
            0.0,
            min(0.999, float(snapshot.get("ema_decay", self.ema_decay))),
        )
