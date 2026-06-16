"""Checkpoint-backed structural review queue fed by bounded column candidates."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import torch


STRUCTURAL_REVIEW_CLAIM_BOUNDARY = (
    "training_owned_checkpoint_backed_structural_review_queue_uses_bounded_awake_candidates"
)


def _index_float_tensor(
    value: torch.Tensor | None,
    *,
    ids: torch.Tensor,
    n_columns: int,
    fill: float,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor) and int(value.numel()) == int(n_columns):
        return (
            value.detach()
            .to(device=device, dtype=torch.float32)
            .flatten()
            .index_select(0, ids)
        )
    return torch.full(
        (int(ids.numel()),),
        float(fill),
        dtype=torch.float32,
        device=device,
    )


def _index_int_tensor(
    value: torch.Tensor | None,
    *,
    ids: torch.Tensor,
    n_columns: int,
    fill: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor) and int(value.numel()) == int(n_columns):
        return (
            value.detach()
            .to(device=device, dtype=torch.long)
            .flatten()
            .index_select(0, ids)
        )
    return torch.full(
        (int(ids.numel()),),
        int(fill),
        dtype=torch.long,
        device=device,
    )


def _ticket_hash(parts: list[str]) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


class ColumnStructuralReviewQueue:
    """Durable review queue for growth/prune continuations.

    The queue is deliberately advisory. It persists operator-review tickets, but
    it never mutates topology and never scans all columns to discover work.
    """

    def __init__(
        self,
        *,
        n_columns: int,
        device: torch.device | str,
        max_tickets: int = 64,
        max_candidates_per_update: int = 16,
        growth_streak_threshold: int = 3,
    ) -> None:
        self.n_columns = max(0, int(n_columns))
        self.device = torch.device(device)
        self.max_tickets = max(1, int(max_tickets))
        self.max_candidates_per_update = max(1, int(max_candidates_per_update))
        self.growth_streak_threshold = max(1, int(growth_streak_threshold))
        self._tickets: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self.update_count = 0
        self.deferred_update_count = 0
        self.last_update_token: int | None = None
        self.last_update_mode = "not_run"
        self.last_deferred_reason: str | None = None
        self.last_evaluated_column_count = 0
        self.last_cached_column_count = self.n_columns
        self.last_growth_candidate_count = 0
        self.last_prune_candidate_count = 0
        self.last_reason = "not_run"

    def _candidate_indices(self, candidates: torch.Tensor | None) -> torch.Tensor:
        if candidates is None or int(candidates.numel()) <= 0 or self.n_columns <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        ids = candidates.detach().flatten().to(device=self.device, dtype=torch.long)
        ids = ids[(ids >= 0) & (ids < int(self.n_columns))]
        if int(ids.numel()) > int(self.max_candidates_per_update):
            ids = ids[: int(self.max_candidates_per_update)]
        return ids

    def _append_or_merge_ticket(
        self,
        *,
        kind: str,
        column_id: int,
        token_count: int,
        reason: str,
        evidence: Mapping[str, Any],
    ) -> None:
        key = f"{kind}:{int(column_id)}"
        if key in self._tickets:
            ticket = self._tickets[key]
            ticket["last_seen_token"] = int(token_count)
            ticket["observation_count"] = int(ticket.get("observation_count", 1)) + 1
            ticket["reason"] = str(reason)
            ticket["evidence"] = dict(evidence)
            return

        next_gate = (
            "explicit_binding_growth_trial_design"
            if kind == "growth_review"
            else "isolated_column_prune_or_sleep_review"
        )
        ticket_id = _ticket_hash(
            [
                "column_structural_review_ticket.v1",
                str(kind),
                str(int(column_id)),
                str(next_gate),
            ]
        )
        ticket = {
            "surface": "column_structural_review_ticket.v1",
            "ticket_id": ticket_id,
            "kind": str(kind),
            "column_id": int(column_id),
            "first_seen_token": int(token_count),
            "last_seen_token": int(token_count),
            "observation_count": 1,
            "status": "pending_operator_review",
            "reason": str(reason),
            "next_gate": next_gate,
            "requires_operator_review": True,
            "requires_checkpoint_transaction": True,
            "mutates_runtime_state": False,
            "source": "training.column_structural_review_queue",
            "evidence": dict(evidence),
        }
        self._tickets[key] = ticket
        self._order.append(key)
        while len(self._order) > int(self.max_tickets):
            stale = self._order.pop(0)
            self._tickets.pop(stale, None)

    def record_deferred(
        self,
        *,
        token_count: int,
        mode: str,
        reason: str,
    ) -> None:
        self.deferred_update_count += 1
        self.last_update_token = int(token_count)
        self.last_update_mode = str(mode)
        self.last_deferred_reason = str(reason)
        self.last_evaluated_column_count = 0
        self.last_cached_column_count = int(self.n_columns)
        self.last_growth_candidate_count = 0
        self.last_prune_candidate_count = 0
        self.last_reason = "deferred"

    def record_candidates(
        self,
        candidates: torch.Tensor | None,
        *,
        token_count: int,
        mode: str,
        prediction_error: torch.Tensor | None,
        confidence: torch.Tensor | None,
        prediction_failure_streak: torch.Tensor | None,
        estimated_cost: torch.Tensor | None,
        memory_pressure: torch.Tensor | None,
        wake_reason: str | None,
        sleep_reason: str | None,
    ) -> None:
        ids = self._candidate_indices(candidates)
        count = int(ids.numel())
        self.update_count += 1
        self.last_update_token = int(token_count)
        self.last_update_mode = str(mode)
        self.last_deferred_reason = None
        self.last_evaluated_column_count = count
        self.last_cached_column_count = max(0, int(self.n_columns) - count)
        self.last_growth_candidate_count = 0
        self.last_prune_candidate_count = 0
        if count <= 0:
            self.last_reason = "no_awake_candidates"
            return

        source_device = ids.device
        pred = _index_float_tensor(
            prediction_error,
            ids=ids,
            n_columns=self.n_columns,
            fill=0.0,
            device=source_device,
        )
        conf = _index_float_tensor(
            confidence,
            ids=ids,
            n_columns=self.n_columns,
            fill=1.0,
            device=source_device,
        )
        streak = _index_int_tensor(
            prediction_failure_streak,
            ids=ids,
            n_columns=self.n_columns,
            fill=0,
            device=source_device,
        )
        cost = _index_float_tensor(
            estimated_cost,
            ids=ids,
            n_columns=self.n_columns,
            fill=0.0,
            device=source_device,
        )
        pressure = _index_float_tensor(
            memory_pressure,
            ids=ids,
            n_columns=self.n_columns,
            fill=0.0,
            device=source_device,
        )

        rows = torch.stack(
            [
                ids.to(dtype=torch.float32),
                pred.clamp(0.0, 1.0),
                conf.clamp(0.0, 1.0),
                streak.to(dtype=torch.float32),
                cost.clamp(0.0, 1.0),
                pressure.clamp(0.0, 1.0),
            ],
            dim=1,
        ).detach().cpu()

        growth_count = 0
        prune_count = 0
        for row in rows.tolist():
            column_id = int(row[0])
            pred_error = float(row[1])
            conf_value = float(row[2])
            streak_value = int(row[3])
            cost_value = float(row[4])
            pressure_value = float(row[5])
            evidence = {
                "prediction_error": round(pred_error, 6),
                "confidence": round(conf_value, 6),
                "prediction_failure_streak": int(streak_value),
                "estimated_cost": round(cost_value, 6),
                "memory_pressure": round(pressure_value, 6),
                "wake_reason": wake_reason,
                "sleep_reason": sleep_reason,
                "candidate_boundary": "bounded_awake_or_route_candidate_set",
            }
            if (
                streak_value >= int(self.growth_streak_threshold)
                and pred_error >= 0.60
                and conf_value <= 0.45
            ):
                growth_count += 1
                self._append_or_merge_ticket(
                    kind="growth_review",
                    column_id=column_id,
                    token_count=int(token_count),
                    reason="repeated_prediction_failure_on_awake_candidate",
                    evidence=evidence,
                )
            if pressure_value >= 0.95 or (cost_value >= 0.85 and conf_value <= 0.25):
                prune_count += 1
                self._append_or_merge_ticket(
                    kind="prune_or_sleep_review",
                    column_id=column_id,
                    token_count=int(token_count),
                    reason="high_memory_or_cost_pressure_on_awake_candidate",
                    evidence=evidence,
                )

        self.last_growth_candidate_count = int(growth_count)
        self.last_prune_candidate_count = int(prune_count)
        self.last_reason = (
            "queued_structural_review"
            if growth_count or prune_count
            else "no_structural_review_candidate_in_awake_set"
        )

    def report(self) -> dict[str, Any]:
        tickets = [self._tickets[key] for key in self._order if key in self._tickets]
        growth_count = sum(1 for ticket in tickets if ticket.get("kind") == "growth_review")
        prune_count = sum(
            1 for ticket in tickets if ticket.get("kind") == "prune_or_sleep_review"
        )
        next_gate = (
            "operator_review_column_structural_ticket"
            if tickets
            else "continue_bounded_candidate_evidence_collection"
        )
        return {
            "surface": "column_structural_review_queue.v1",
            "artifact_kind": "marulho_column_structural_review_queue",
            "source": "training.column_structural_review_queue",
            "total_columns": int(self.n_columns),
            "pending_count": int(len(tickets)),
            "growth_ticket_count": int(growth_count),
            "prune_or_sleep_ticket_count": int(prune_count),
            "last_update_token": self.last_update_token,
            "last_update_mode": str(self.last_update_mode),
            "last_evaluated_column_count": int(self.last_evaluated_column_count),
            "last_cached_column_count": int(self.last_cached_column_count),
            "last_growth_candidate_count": int(self.last_growth_candidate_count),
            "last_prune_or_sleep_candidate_count": int(self.last_prune_candidate_count),
            "update_count": int(self.update_count),
            "deferred_update_count": int(self.deferred_update_count),
            "last_deferred_reason": self.last_deferred_reason,
            "last_reason": str(self.last_reason),
            "max_candidates_per_update": int(self.max_candidates_per_update),
            "checkpoint_backed": True,
            "requires_operator_review": True,
            "mutates_runtime_state": False,
            "runs_all_columns": False,
            "next_gate": next_gate,
            "tickets_sample": [dict(ticket) for ticket in tickets[-8:]],
            "claim_boundary": STRUCTURAL_REVIEW_CLAIM_BOUNDARY,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "tickets": [dict(self._tickets[key]) for key in self._order if key in self._tickets],
            "update_count": int(self.update_count),
            "deferred_update_count": int(self.deferred_update_count),
            "last_update_token": self.last_update_token,
            "last_update_mode": str(self.last_update_mode),
            "last_deferred_reason": self.last_deferred_reason,
            "last_evaluated_column_count": int(self.last_evaluated_column_count),
            "last_cached_column_count": int(self.last_cached_column_count),
            "last_growth_candidate_count": int(self.last_growth_candidate_count),
            "last_prune_candidate_count": int(self.last_prune_candidate_count),
            "last_reason": str(self.last_reason),
            "max_tickets": int(self.max_tickets),
            "max_candidates_per_update": int(self.max_candidates_per_update),
            "growth_streak_threshold": int(self.growth_streak_threshold),
        }

    def load_state_dict(self, snapshot: Mapping[str, Any] | None) -> None:
        if not isinstance(snapshot, Mapping):
            return
        self._tickets.clear()
        self._order.clear()
        for raw_ticket in list(snapshot.get("tickets") or []):
            if not isinstance(raw_ticket, Mapping):
                continue
            kind = str(raw_ticket.get("kind") or "")
            column_id = raw_ticket.get("column_id")
            try:
                key = f"{kind}:{int(column_id)}"
            except (TypeError, ValueError):
                continue
            ticket = dict(raw_ticket)
            ticket["mutates_runtime_state"] = False
            ticket["requires_operator_review"] = True
            ticket["requires_checkpoint_transaction"] = True
            self._tickets[key] = ticket
            self._order.append(key)
            if len(self._order) >= int(self.max_tickets):
                break
        self.update_count = int(snapshot.get("update_count", 0) or 0)
        self.deferred_update_count = int(
            snapshot.get("deferred_update_count", 0) or 0
        )
        token = snapshot.get("last_update_token")
        self.last_update_token = None if token is None else int(token)
        self.last_update_mode = str(snapshot.get("last_update_mode", "not_run"))
        reason = snapshot.get("last_deferred_reason")
        self.last_deferred_reason = None if reason is None else str(reason)
        self.last_evaluated_column_count = int(
            snapshot.get("last_evaluated_column_count", 0) or 0
        )
        self.last_cached_column_count = int(
            snapshot.get("last_cached_column_count", self.n_columns) or 0
        )
        self.last_growth_candidate_count = int(
            snapshot.get("last_growth_candidate_count", 0) or 0
        )
        self.last_prune_candidate_count = int(
            snapshot.get("last_prune_candidate_count", 0) or 0
        )
        self.last_reason = str(snapshot.get("last_reason", "checkpoint_restored"))
        self.max_tickets = max(1, int(snapshot.get("max_tickets", self.max_tickets)))
        self.max_candidates_per_update = max(
            1,
            int(
                snapshot.get(
                    "max_candidates_per_update",
                    self.max_candidates_per_update,
                )
            ),
        )
        self.growth_streak_threshold = max(
            1,
            int(
                snapshot.get(
                    "growth_streak_threshold",
                    self.growth_streak_threshold,
                )
            ),
        )
