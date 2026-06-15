"""Predictive Columns -- Thousand Brains Theory extension for CompetitiveColumnLayer.

Implements Phase 1 of the Thousand Brains improvement:
- Each column maintains a small "location" state vector (reference frame)
- Columns predict what they'll sense next based on location state
- Prediction error drives additional STDP modulation
- Inter-column voting produces consensus about active concepts

This is a WRAPPER around CompetitiveColumnLayer -- it doesn't replace
the existing WTA mechanism, it augments it with predictive capabilities.

References:
- Hawkins et al. (2019): "A Framework for Intelligence and Cortical Function"
- Monty (tbp.monty v0.29.0): Official Thousand Brains implementation
- Lewis et al. (2019): "Locations in the Neocortex" (grid cell analog)
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


def _ensure_windows_triton_compiler() -> str | None:
    if os.environ.get("CC"):
        return os.environ["CC"]
    if os.name != "nt":
        return None
    try:
        import triton  # type: ignore[import-not-found]
    except Exception:
        return None
    tcc = Path(triton.__file__).parent / "runtime" / "tcc" / "tcc.exe"
    if tcc.exists():
        os.environ["CC"] = str(tcc)
        return str(tcc)
    return None


def dense_predictive_transition(
    location: torch.Tensor,
    velocity: torch.Tensor,
    prediction_weights: torch.Tensor,
    prediction_error: torch.Tensor,
    prediction_failure_streak: torch.Tensor,
    confidence: torch.Tensor,
    routing_key: torch.Tensor,
    previous_routing_key: torch.Tensor,
    winners: torch.Tensor,
    *,
    has_previous_routing_key: bool,
    error_ema_alpha: float,
    failure_streak_threshold: float,
    learning_rate: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Compute one dense predictive-column transition without side effects."""

    winner_ids = winners.to(device=location.device, dtype=torch.long).flatten()
    prediction = torch.sigmoid((location * prediction_weights).sum(dim=1))
    actual_binary = torch.zeros_like(prediction).scatter(
        0,
        winner_ids,
        torch.ones_like(winner_ids, dtype=prediction.dtype),
    )
    raw_error = (prediction - actual_binary).abs()
    next_failure_streak = torch.where(
        raw_error > float(failure_streak_threshold),
        prediction_failure_streak + 1,
        torch.zeros_like(prediction_failure_streak),
    )
    next_prediction_error = (
        float(error_ema_alpha) * raw_error
        + (1.0 - float(error_ema_alpha)) * prediction_error
    )
    next_confidence = torch.clamp(
        0.95 * confidence + 0.05 * (1.0 - raw_error),
        min=0.0,
        max=1.0,
    )

    next_velocity = velocity
    next_location = location
    if has_previous_routing_key:
        movement = routing_key - previous_routing_key
        location_dim = int(location.shape[1])
        if int(movement.shape[0]) >= location_dim:
            movement = movement[:location_dim]
        else:
            movement = F.pad(movement, (0, location_dim - int(movement.shape[0])))
        next_velocity = velocity * 0.9
        winner_velocity = next_velocity.index_select(0, winner_ids) + 0.1 * movement
        next_velocity = next_velocity.index_copy(0, winner_ids, winner_velocity)
        next_location = location.index_copy(
            0,
            winner_ids,
            location.index_select(0, winner_ids) + winner_velocity,
        )

    location_norm = next_location.norm(dim=1, keepdim=True).clamp(min=1e-8)
    location_scale = torch.where(
        location_norm > 5.0,
        5.0 / location_norm,
        torch.ones_like(location_norm),
    )
    next_location = next_location * location_scale

    lr = float(learning_rate)
    winner_weights = (
        prediction_weights.index_select(0, winner_ids)
        + lr * next_location.index_select(0, winner_ids)
    )
    next_prediction_weights = prediction_weights.index_copy(
        0,
        winner_ids,
        winner_weights,
    )
    updated_prediction = torch.sigmoid(
        (next_location * next_prediction_weights).sum(dim=1)
    )
    non_winner_mask = torch.ones(
        prediction_weights.shape[0],
        dtype=torch.bool,
        device=prediction_weights.device,
    ).scatter(0, winner_ids, False)
    decay_mask = non_winner_mask & (updated_prediction > 0.5)
    next_prediction_weights = torch.where(
        decay_mask.unsqueeze(1),
        next_prediction_weights * (1.0 - 0.5 * lr),
        next_prediction_weights,
    )
    return (
        next_location,
        next_velocity,
        next_prediction_weights,
        next_prediction_error,
        next_failure_streak,
        next_confidence,
    )


class PredictiveColumnState:
    """Per-column predictive state -- location, prediction, and confidence.

    Each column acts as a miniature modeling system that:
    1. Maintains a location vector (path integration analog)
    2. Predicts what sensory input it expects at this location
    3. Computes prediction error to modulate learning
    4. Votes with other columns about what object/concept is active
    """

    def __init__(
        self,
        n_columns: int,
        location_dim: int = 8,
        device: torch.device | None = None,
    ) -> None:
        self.n_columns = n_columns
        self.location_dim = location_dim
        self.device = device or torch.device("cpu")

        # Per-column location state (reference frame position)
        # Initialized near origin with small random offset
        self.location = torch.randn(n_columns, location_dim, device=self.device) * 0.1

        # Location velocity (path integration -- updated by sensory transitions)
        self.velocity = torch.zeros(n_columns, location_dim, device=self.device)

        # Per-column prediction of next sensory input
        # Learned mapping: location -> expected routing key
        self._prediction_weights = torch.randn(
            n_columns, location_dim, device=self.device
        ) * 0.01

        # Prediction error history (EMA for stability)
        self.prediction_error = torch.zeros(n_columns, device=self.device)
        self._error_ema_alpha = 0.2
        self._failure_streak_threshold = 0.65
        self.prediction_failure_streak = torch.zeros(
            n_columns,
            dtype=torch.int32,
            device=self.device,
        )

        # Column confidence (how well predictions have matched reality)
        self.confidence = torch.ones(n_columns, device=self.device) * 0.5

        self.last_prediction_update_mode = "not_run"
        self.last_prediction_update_count = 0
        self.last_prediction_update_fraction = 0.0
        self.last_prediction_cached_count = 0
        self.last_prediction_update_runs_all_columns = False
        self.last_prediction_update_fallback_reason: str | None = None
        self.last_location_update_mode = "not_run"
        self.last_location_update_count = 0
        self.last_location_cached_count = 0
        self.last_location_update_runs_all_columns = False
        self.last_location_update_fallback_reason: str | None = None
        self.cached_consensus_gain = torch.ones(n_columns, device=self.device)
        self.last_vote_update_mode = "not_run"
        self.last_vote_update_count = 0
        self.last_vote_update_fraction = 0.0
        self.last_vote_cached_count = 0
        self.last_vote_runs_all_columns = False
        self.last_vote_fallback_reason: str | None = None
        self.last_predictive_materialize_mode = "not_run"
        self.last_predictive_materialize_count = 0
        self.last_predictive_materialize_max_age = 0
        self.predictive_step_count = 0
        self.predictive_last_update_step = torch.zeros(
            n_columns,
            dtype=torch.long,
            device=self.device,
        )
        self._predictive_has_cached_columns = False
        self._predictive_materialize_learning_rate = 0.005
        self.last_dense_transition_mode = "legacy"
        self.last_dense_transition_fallback_reason: str | None = None
        self.dense_transition_compile_count = 0
        self._compiled_dense_transition_with_previous = None
        self._compiled_dense_transition_without_previous = None

    def _compiled_dense_transition(self, *, has_previous_routing_key: bool):
        attribute = (
            "_compiled_dense_transition_with_previous"
            if has_previous_routing_key
            else "_compiled_dense_transition_without_previous"
        )
        compiled = getattr(self, attribute)
        if compiled is not None:
            return compiled

        error_ema_alpha = float(self._error_ema_alpha)
        failure_streak_threshold = float(self._failure_streak_threshold)

        def transition(
            location: torch.Tensor,
            velocity: torch.Tensor,
            prediction_weights: torch.Tensor,
            prediction_error: torch.Tensor,
            prediction_failure_streak: torch.Tensor,
            confidence: torch.Tensor,
            routing_key: torch.Tensor,
            previous_routing_key: torch.Tensor,
            winners: torch.Tensor,
        ):
            return dense_predictive_transition(
                location,
                velocity,
                prediction_weights,
                prediction_error,
                prediction_failure_streak,
                confidence,
                routing_key,
                previous_routing_key,
                winners,
                has_previous_routing_key=has_previous_routing_key,
                error_ema_alpha=error_ema_alpha,
                failure_streak_threshold=failure_streak_threshold,
                learning_rate=0.005,
            )

        _ensure_windows_triton_compiler()
        compiled = torch.compile(transition, mode="reduce-overhead", fullgraph=True)
        setattr(self, attribute, compiled)
        self.dense_transition_compile_count += 1
        return compiled

    def apply_dense_transition(
        self,
        winners: torch.Tensor,
        routing_key: torch.Tensor,
        previous_routing_key: torch.Tensor | None,
        *,
        learning_rate: float = 0.005,
        transition_mode: str = "fused_eager",
    ) -> torch.Tensor:
        """Apply one all-column predictive transition with explicit writeback."""

        previous = (
            torch.zeros_like(routing_key)
            if previous_routing_key is None
            else previous_routing_key
        )
        has_previous = previous_routing_key is not None
        transition_fn = dense_predictive_transition
        transition_kwargs = {
            "has_previous_routing_key": has_previous,
            "error_ema_alpha": self._error_ema_alpha,
            "failure_streak_threshold": self._failure_streak_threshold,
            "learning_rate": learning_rate,
        }
        self.last_dense_transition_fallback_reason = None
        if transition_mode == "compiled":
            if abs(float(learning_rate) - 0.005) > 1e-12:
                self.last_dense_transition_mode = "fused_eager"
                self.last_dense_transition_fallback_reason = (
                    "compiled_learning_rate_requires_0.005"
                )
            else:
                try:
                    transition_fn = self._compiled_dense_transition(
                        has_previous_routing_key=has_previous,
                    )
                    transition_kwargs = {}
                    self.last_dense_transition_mode = "compiled"
                except Exception as exc:  # pragma: no cover - backend dependent
                    self.last_dense_transition_mode = "fused_eager"
                    self.last_dense_transition_fallback_reason = (
                        f"compile_failed:{type(exc).__name__}"
                    )
        elif transition_mode == "fused_eager":
            self.last_dense_transition_mode = "fused_eager"
        else:
            raise ValueError("transition_mode must be fused_eager or compiled")

        try:
            if self.last_dense_transition_mode == "compiled":
                torch.compiler.cudagraph_mark_step_begin()
            outputs = transition_fn(
                self.location,
                self.velocity,
                self._prediction_weights,
                self.prediction_error,
                self.prediction_failure_streak,
                self.confidence,
                routing_key,
                previous,
                winners,
                **transition_kwargs,
            )
        except Exception as exc:  # pragma: no cover - backend dependent
            if self.last_dense_transition_mode != "compiled":
                raise
            self.last_dense_transition_mode = "fused_eager"
            self.last_dense_transition_fallback_reason = (
                f"compiled_execution_failed:{type(exc).__name__}"
            )
            outputs = dense_predictive_transition(
                self.location,
                self.velocity,
                self._prediction_weights,
                self.prediction_error,
                self.prediction_failure_streak,
                self.confidence,
                routing_key,
                previous,
                winners,
                has_previous_routing_key=has_previous,
                error_ema_alpha=self._error_ema_alpha,
                failure_streak_threshold=self._failure_streak_threshold,
                learning_rate=learning_rate,
            )

        if self.last_dense_transition_mode == "compiled":
            self.location.copy_(outputs[0])
            self.velocity.copy_(outputs[1])
            self._prediction_weights.copy_(outputs[2])
            self.prediction_error.copy_(outputs[3])
            self.prediction_failure_streak.copy_(outputs[4])
            self.confidence.copy_(outputs[5])
        else:
            (
                self.location,
                self.velocity,
                self._prediction_weights,
                self.prediction_error,
                self.prediction_failure_streak,
                self.confidence,
            ) = outputs
        self._record_prediction_update_scope(None)
        self._mark_predictive_update_complete(None)
        return self.prediction_error

    def _candidate_update_indices(
        self,
        candidate_indices: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if candidate_indices is None or int(candidate_indices.numel()) == 0:
            return None
        candidates = candidate_indices.to(device=self.device, dtype=torch.long).flatten()
        if int(candidates.numel()) <= 0:
            return None
        candidates = candidates[
            (candidates >= 0) & (candidates < int(self.n_columns))
        ]
        if int(candidates.numel()) <= 0:
            return None
        return torch.unique(candidates, sorted=True)

    def materialize_predictive_state(
        self,
        candidate_indices: torch.Tensor | None,
        *,
        record_noop: bool = True,
    ) -> None:
        """Advance cached non-awake predictive columns to the current tick.

        Candidate-scoped prediction updates intentionally skip idle columns.
        When a cached column wakes as a routed candidate, this method replays
        only the zero-actual/non-winner predictive updates that column missed:
        prediction error/confidence EMA, velocity decay, location clamping, and
        high-prediction weight decay. The work is bounded to the provided
        candidate set unless an explicit all-column fallback has already chosen
        to run all columns.
        """

        if self.predictive_last_update_step.device != self.device:
            self.predictive_last_update_step = self.predictive_last_update_step.to(
                self.device
            )

        candidates = self._candidate_update_indices(candidate_indices)
        if candidates is None:
            requested_count = int(self.n_columns)
            mode = "all_columns"
        else:
            requested_count = int(candidates.numel())
            mode = (
                "all_columns_candidate_set"
                if requested_count >= int(self.n_columns)
                else "candidate_subset"
            )

        current_step = int(self.predictive_step_count)
        if (
            current_step <= 0
            or not bool(self._predictive_has_cached_columns)
        ):
            if record_noop:
                self.last_predictive_materialize_mode = f"{mode}_noop"
                self.last_predictive_materialize_count = 0
                self.last_predictive_materialize_max_age = 0
            return

        if candidates is None:
            indices = torch.arange(self.n_columns, device=self.device, dtype=torch.long)
        else:
            indices = candidates

        if int(indices.numel()) <= 0:
            if record_noop:
                self.last_predictive_materialize_mode = "empty_candidate_set"
                self.last_predictive_materialize_count = 0
                self.last_predictive_materialize_max_age = 0
            return

        last_steps = self.predictive_last_update_step.index_select(0, indices)
        ages = torch.clamp(current_step - last_steps, min=0)
        pending_mask = ages > 0
        if not bool(pending_mask.any().item()):
            if record_noop:
                self.last_predictive_materialize_mode = f"{mode}_noop"
                self.last_predictive_materialize_count = 0
                self.last_predictive_materialize_max_age = 0
            return

        pending_indices = indices[pending_mask]
        pending_ages = ages[pending_mask]
        pending_last_steps = last_steps[pending_mask]
        max_age = int(pending_ages.max().item())
        lr = float(self._predictive_materialize_learning_rate)
        decay = 1.0 - 0.5 * lr

        for offset in range(max_age):
            active = pending_ages > offset
            if not bool(active.any().item()):
                continue
            active_indices = pending_indices[active]
            active_steps = pending_last_steps[active] + int(offset)

            loc = self.location.index_select(0, active_indices)
            weights = self._prediction_weights.index_select(0, active_indices)
            prediction = torch.sigmoid((loc * weights).sum(dim=1))
            raw_error = prediction.abs()
            failure_mask = raw_error > float(self._failure_streak_threshold)

            current_streak = self.prediction_failure_streak.index_select(
                0,
                active_indices,
            )
            self.prediction_failure_streak[active_indices] = torch.where(
                failure_mask,
                current_streak + 1,
                torch.zeros_like(current_streak),
            )

            current_error = self.prediction_error.index_select(0, active_indices)
            self.prediction_error[active_indices] = (
                self._error_ema_alpha * raw_error
                + (1 - self._error_ema_alpha) * current_error
            )

            current_confidence = self.confidence.index_select(0, active_indices)
            next_confidence = 0.95 * current_confidence + 0.05 * (1.0 - raw_error)
            self.confidence[active_indices] = next_confidence.clamp(0.0, 1.0)

            decayed_velocity_mask = active_steps > 0
            if bool(decayed_velocity_mask.any().item()):
                velocity_indices = active_indices[decayed_velocity_mask]
                self.velocity[velocity_indices] *= 0.9

            loc = self.location.index_select(0, active_indices)
            loc_norm = loc.norm(dim=1, keepdim=True).clamp(min=1e-8)
            scale = torch.where(
                loc_norm > 5.0,
                5.0 / loc_norm,
                torch.ones_like(loc_norm),
            )
            self.location[active_indices] = loc * scale

            loc = self.location.index_select(0, active_indices)
            weights = self._prediction_weights.index_select(0, active_indices)
            updated_prediction = torch.sigmoid((loc * weights).sum(dim=1))
            high_pred = updated_prediction > 0.5
            if bool(high_pred.any().item()):
                high_indices = active_indices[high_pred]
                self._prediction_weights[high_indices] *= decay

        self.predictive_last_update_step[pending_indices] = current_step
        if candidates is None or requested_count >= int(self.n_columns):
            self._predictive_has_cached_columns = False
        if record_noop:
            self.last_predictive_materialize_mode = mode
            self.last_predictive_materialize_count = int(pending_indices.numel())
            self.last_predictive_materialize_max_age = max_age

    def _mark_predictive_update_complete(
        self,
        candidate_indices: torch.Tensor | None,
        *,
        step_count: int = 1,
    ) -> None:
        steps = max(1, int(step_count))
        next_step = int(self.predictive_step_count) + steps
        candidates = self._candidate_update_indices(candidate_indices)
        if candidates is None or int(candidates.numel()) >= int(self.n_columns):
            self.predictive_last_update_step.fill_(next_step)
            self._predictive_has_cached_columns = False
        else:
            self.predictive_last_update_step[candidates] = next_step
            self._predictive_has_cached_columns = True
        self.predictive_step_count = next_step

    def _record_prediction_update_scope(
        self,
        candidate_indices: torch.Tensor | None,
        *,
        fallback_reason: str | None = None,
    ) -> None:
        if candidate_indices is None:
            count = int(self.n_columns)
            mode = "all_columns"
        else:
            count = int(candidate_indices.numel())
            mode = "candidate_subset"
        self.last_prediction_update_mode = mode
        self.last_prediction_update_count = count
        self.last_prediction_update_fraction = (
            float(count) / float(self.n_columns) if self.n_columns > 0 else 0.0
        )
        self.last_prediction_cached_count = max(0, int(self.n_columns) - count)
        self.last_prediction_update_runs_all_columns = count >= int(self.n_columns)
        self.last_prediction_update_fallback_reason = fallback_reason

    def _record_location_update_scope(
        self,
        candidate_indices: torch.Tensor | None,
        *,
        fallback_reason: str | None = None,
    ) -> None:
        if candidate_indices is None:
            count = int(self.n_columns)
            mode = "all_columns"
        else:
            count = int(candidate_indices.numel())
            mode = "candidate_subset"
        self.last_location_update_mode = mode
        self.last_location_update_count = count
        self.last_location_cached_count = max(0, int(self.n_columns) - count)
        self.last_location_update_runs_all_columns = count >= int(self.n_columns)
        self.last_location_update_fallback_reason = fallback_reason

    def predict(
        self,
        column_dim: int,
        candidate_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate prediction of next input based on location state.

        Returns a [n_columns] vector where each entry is the predicted
        activation level for that column (based on its location).
        """
        # Simple prediction: dot product of location with learned weights
        # This gives each column's confidence that IT should be the winner
        candidates = self._candidate_update_indices(candidate_indices)
        if candidates is None:
            pred = (self.location * self._prediction_weights).sum(dim=1)
        else:
            pred = (
                self.location.index_select(0, candidates)
                * self._prediction_weights.index_select(0, candidates)
            ).sum(dim=1)
        return torch.sigmoid(pred)

    def update_location(
        self,
        winners: list[int],
        routing_key: torch.Tensor,
        prev_routing_key: Optional[torch.Tensor] = None,
        candidate_indices: torch.Tensor | None = None,
    ) -> None:
        """Update location state via path integration.

        The "movement" signal is the transition between consecutive inputs.
        Winner columns integrate this into their location state.
        """
        candidates = self._candidate_update_indices(candidate_indices)
        if candidates is not None:
            winner_tensor = torch.as_tensor(winners, device=self.device, dtype=torch.long)
            if int(winner_tensor.numel()) > 0:
                candidates = torch.unique(torch.cat((candidates, winner_tensor)))
                candidates = candidates[
                    (candidates >= 0) & (candidates < int(self.n_columns))
                ]
        self.materialize_predictive_state(candidates)
        self._record_location_update_scope(candidates)
        if prev_routing_key is not None:
            # Compute movement as difference between consecutive inputs
            # Project to location_dim via simple hash
            diff = routing_key - prev_routing_key
            # Use first location_dim elements as movement signal
            movement = diff[:self.location_dim] if diff.shape[0] >= self.location_dim else \
                F.pad(diff, (0, self.location_dim - diff.shape[0]))

            # Path integration: update velocity and position for winners
            decay = 0.9
            if candidates is None:
                self.velocity *= decay
            elif int(candidates.numel()) > 0:
                self.velocity[candidates] *= decay
            for w in winners:
                self.velocity[w] += 0.1 * movement.to(self.device)
                self.location[w] += self.velocity[w]

        # Normalize locations to prevent drift to infinity
        if candidates is None:
            loc_norm = self.location.norm(dim=1, keepdim=True).clamp(min=1e-8)
            scale = torch.where(loc_norm > 5.0, 5.0 / loc_norm, torch.ones_like(loc_norm))
            self.location *= scale
        elif int(candidates.numel()) > 0:
            loc = self.location.index_select(0, candidates)
            loc_norm = loc.norm(dim=1, keepdim=True).clamp(min=1e-8)
            scale = torch.where(loc_norm > 5.0, 5.0 / loc_norm, torch.ones_like(loc_norm))
            self.location[candidates] = loc * scale

    def compute_prediction_error(
        self,
        winners: list[int],
        actual_activation: torch.Tensor,
        candidate_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute prediction error: difference between predicted and actual.

        Returns per-column prediction error (higher = more surprised).
        This signal modulates STDP learning rate.
        """
        candidates = self._candidate_update_indices(candidate_indices)
        self.materialize_predictive_state(candidates)
        self._record_prediction_update_scope(candidates)
        prediction = self.predict(actual_activation.shape[0], candidates)
        # Error: columns that predicted they'd win but didn't (or vice versa)
        if candidates is None:
            actual_binary = torch.zeros(self.n_columns, device=self.device)
            for w in winners:
                actual_binary[w] = 1.0
        else:
            winner_tensor = torch.as_tensor(winners, device=self.device, dtype=torch.long)
            if int(winner_tensor.numel()) == 0:
                actual_binary = torch.zeros(candidates.shape[0], device=self.device)
            else:
                actual_binary = (candidates[:, None] == winner_tensor[None, :]).any(dim=1).to(
                    dtype=torch.float32
                )

        raw_error = (prediction - actual_binary).abs()
        failure_mask = raw_error > float(self._failure_streak_threshold)
        if candidates is None:
            self.prediction_failure_streak = torch.where(
                failure_mask,
                self.prediction_failure_streak + 1,
                torch.zeros_like(self.prediction_failure_streak),
            )

            # EMA smoothing
            self.prediction_error = (
                self._error_ema_alpha * raw_error
                + (1 - self._error_ema_alpha) * self.prediction_error
            )

            # Update confidence (columns with low error gain confidence)
            self.confidence = 0.95 * self.confidence + 0.05 * (1.0 - raw_error)
            self.confidence.clamp_(0.0, 1.0)
        else:
            current_streak = self.prediction_failure_streak.index_select(0, candidates)
            next_streak = torch.where(
                failure_mask,
                current_streak + 1,
                torch.zeros_like(current_streak),
            )
            self.prediction_failure_streak[candidates] = next_streak

            current_error = self.prediction_error.index_select(0, candidates)
            self.prediction_error[candidates] = (
                self._error_ema_alpha * raw_error
                + (1 - self._error_ema_alpha) * current_error
            )

            current_confidence = self.confidence.index_select(0, candidates)
            next_confidence = 0.95 * current_confidence + 0.05 * (1.0 - raw_error)
            self.confidence[candidates] = next_confidence.clamp(0.0, 1.0)

        return self.prediction_error

    def update_predictions(
        self,
        winners: list[int],
        learning_rate: float = 0.01,
        candidate_indices: torch.Tensor | None = None,
    ) -> None:
        """Update prediction weights for winner columns.

        Winners that correctly predicted their activation strengthen;
        non-winners that incorrectly predicted weaken.
        """
        candidates = self._candidate_update_indices(candidate_indices)
        self.materialize_predictive_state(candidates)
        self._record_prediction_update_scope(candidates)
        lr = float(learning_rate)
        self._predictive_materialize_learning_rate = lr
        for w in winners:
            # Strengthen prediction weights for winners
            self._prediction_weights[w] += lr * self.location[w]
        # Slight decay for non-winners that predicted high
        prediction = self.predict(self.n_columns, candidates)
        if candidates is None:
            non_winner_mask = torch.ones(self.n_columns, dtype=torch.bool, device=self.device)
            for w in winners:
                non_winner_mask[w] = False
            high_pred_non_winners = non_winner_mask & (prediction > 0.5)
            self._prediction_weights[high_pred_non_winners] *= (1.0 - 0.5 * lr)
        else:
            winner_tensor = torch.as_tensor(winners, device=self.device, dtype=torch.long)
            if int(winner_tensor.numel()) == 0:
                non_winner_mask = torch.ones(candidates.shape[0], dtype=torch.bool, device=self.device)
            else:
                non_winner_mask = ~(candidates[:, None] == winner_tensor[None, :]).any(dim=1)
            high_pred_non_winners = candidates[non_winner_mask & (prediction > 0.5)]
            self._prediction_weights[high_pred_non_winners] *= (1.0 - 0.5 * lr)
        self._mark_predictive_update_complete(candidates)

    def vote(
        self,
        winners: list[int],
        top_k_activations: torch.Tensor,
        candidate_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inter-column voting to reach consensus.

        Winner columns broadcast their "hypothesis" (confidence-weighted
        activation). Other columns with compatible hypotheses get boosted.

        Returns a consensus gain vector [n_columns] that modulates the next
        competitive step. When candidate_indices is provided, only those awake
        columns recompute their vote; the remaining entries are cached state.
        """
        if self.cached_consensus_gain.device != self.device:
            self.cached_consensus_gain = self.cached_consensus_gain.to(self.device)

        candidates = self._candidate_update_indices(candidate_indices)

        if candidate_indices is not None and (candidates is None or int(candidates.numel()) == 0):
            self.last_vote_update_mode = "cached_vote_no_awake_candidates"
            self.last_vote_update_count = 0
            self.last_vote_update_fraction = 0.0
            self.last_vote_cached_count = int(self.n_columns)
            self.last_vote_runs_all_columns = False
            self.last_vote_fallback_reason = "no_awake_candidates_cached_vote"
            return self.cached_consensus_gain

        self.materialize_predictive_state(candidates)

        if not winners:
            if candidates is None:
                self.cached_consensus_gain.fill_(1.0)
                count = int(self.n_columns)
                self.last_vote_update_mode = "identity_all_columns_no_winners"
                self.last_vote_runs_all_columns = True
                self.last_vote_fallback_reason = "no_previous_winner"
            else:
                self.cached_consensus_gain[candidates] = 1.0
                count = int(candidates.numel())
                self.last_vote_update_mode = "identity_awake_mask_no_winners"
                self.last_vote_runs_all_columns = False
                self.last_vote_fallback_reason = "no_previous_winner"
            self.last_vote_update_count = count
            self.last_vote_update_fraction = float(count) / float(max(1, self.n_columns))
            self.last_vote_cached_count = max(0, int(self.n_columns) - count)
            return self.cached_consensus_gain

        winner_locs = self.location[winners]
        # Cosine similarity between each column's location and winner centroid
        centroid = winner_locs.mean(dim=0)  # [location_dim]
        centroid_norm = centroid.norm().clamp(min=1e-8)

        locations = self.location if candidates is None else self.location.index_select(0, candidates)
        loc_norms = locations.norm(dim=1, keepdim=True).clamp(min=1e-8)
        similarities = (locations @ centroid) / (loc_norms.squeeze() * centroid_norm)

        # Convert to gain: similar locations get boost, dissimilar get suppression
        consensus_gain = 1.0 + 0.3 * similarities.clamp(-1, 1)
        if candidates is None:
            self.cached_consensus_gain = consensus_gain
            count = int(self.n_columns)
            self.last_vote_update_mode = "all_columns"
            self.last_vote_runs_all_columns = True
            self.last_vote_fallback_reason = None
        else:
            self.cached_consensus_gain[candidates] = consensus_gain
            count = int(candidates.numel())
            self.last_vote_update_mode = (
                "awake_mask_cached_vote"
                if count < int(self.n_columns)
                else "all_columns_candidate_set"
            )
            self.last_vote_runs_all_columns = count >= int(self.n_columns)
            self.last_vote_fallback_reason = (
                "candidate_set_covers_all_columns"
                if count >= int(self.n_columns)
                else None
            )
        self.last_vote_update_count = count
        self.last_vote_update_fraction = float(count) / float(max(1, self.n_columns))
        self.last_vote_cached_count = max(0, int(self.n_columns) - count)
        return self.cached_consensus_gain

    def vote_execution_report(self) -> dict[str, object]:
        """Return the last observed predictive-vote scheduler boundary."""
        updated = max(0, min(int(self.last_vote_update_count), int(self.n_columns)))
        cached = max(0, min(int(self.last_vote_cached_count), int(self.n_columns)))
        return {
            "surface": "predictive_column_vote_scheduler.v1",
            "mode": str(self.last_vote_update_mode),
            "total_columns": int(self.n_columns),
            "updated_column_count": updated,
            "updated_column_fraction": round(
                float(updated) / float(max(1, int(self.n_columns))),
                6,
            ),
            "cached_vote_use_count": cached,
            "cached_vote_fraction": round(
                float(cached) / float(max(1, int(self.n_columns))),
                6,
            ),
            "runs_all_columns": bool(self.last_vote_runs_all_columns),
            "fallback_reason": self.last_vote_fallback_reason,
            "tensor_device": str(self.cached_consensus_gain.device),
            "claim_boundary": (
                "training_owned_awake_mask_predictive_vote_cache_skips_non_awake_columns"
            ),
        }

    def prediction_update_execution_report(self) -> dict[str, object]:
        """Return the last observed predictive-state update scheduler boundary."""

        updated = max(0, min(int(self.last_prediction_update_count), int(self.n_columns)))
        cached = (
            0
            if self.last_prediction_update_mode == "not_run"
            else max(0, min(int(self.last_prediction_cached_count), int(self.n_columns)))
        )
        return {
            "surface": "predictive_column_update_scheduler.v1",
            "mode": str(self.last_prediction_update_mode),
            "total_columns": int(self.n_columns),
            "updated_column_count": updated,
            "updated_column_fraction": round(
                float(updated) / float(max(1, int(self.n_columns))),
                6,
            ),
            "cached_state_count": cached,
            "cached_state_fraction": round(
                float(cached) / float(max(1, int(self.n_columns))),
                6,
            ),
            "location_update_mode": str(self.last_location_update_mode),
            "location_update_count": int(self.last_location_update_count),
            "location_cached_count": int(self.last_location_cached_count),
            "location_update_runs_all_columns": bool(
                self.last_location_update_runs_all_columns
            ),
            "predictive_materialize_mode": str(
                self.last_predictive_materialize_mode
            ),
            "predictive_materialize_count": int(
                self.last_predictive_materialize_count
            ),
            "predictive_materialize_max_age": int(
                self.last_predictive_materialize_max_age
            ),
            "predictive_step_count": int(self.predictive_step_count),
            "runs_all_columns": bool(
                self.last_prediction_update_runs_all_columns
                or self.last_location_update_runs_all_columns
            ),
            "fallback_reason": self.last_prediction_update_fallback_reason,
            "tensor_device": str(self.prediction_error.device),
            "claim_boundary": (
                "training_owned_awake_mask_predictive_update_cache_skips_non_awake_columns"
            ),
        }

    def prediction_error_modulation(self) -> torch.Tensor:
        """Get STDP learning rate modulation from prediction error.

        High prediction error -> higher learning rate (surprising things
        should be learned more aggressively).
        """
        # Scale: base 1.0 + up to 2x boost for high-error columns
        return 1.0 + 2.0 * self.prediction_error

    def device_report(self) -> dict[str, object]:
        """Return runtime-visible device placement for predictive state."""
        return {
            "module": "predictive_columns",
            "device": str(self.device),
            "location_device": str(self.location.device),
            "velocity_device": str(self.velocity.device),
            "prediction_weights_device": str(self._prediction_weights.device),
            "prediction_error_device": str(self.prediction_error.device),
            "prediction_failure_streak_device": str(self.prediction_failure_streak.device),
            "prediction_failure_streak_available": True,
            "confidence_device": str(self.confidence.device),
            "n_columns": int(self.n_columns),
            "location_dim": int(self.location_dim),
            "last_prediction_update_mode": self.last_prediction_update_mode,
            "last_prediction_update_count": int(self.last_prediction_update_count),
            "last_prediction_update_fraction": float(self.last_prediction_update_fraction),
            "last_prediction_cached_count": int(self.last_prediction_cached_count),
            "last_prediction_update_runs_all_columns": bool(
                self.last_prediction_update_runs_all_columns
            ),
            "last_prediction_update_fallback_reason": self.last_prediction_update_fallback_reason,
            "last_location_update_mode": self.last_location_update_mode,
            "last_location_update_count": int(self.last_location_update_count),
            "last_location_cached_count": int(self.last_location_cached_count),
            "last_location_update_runs_all_columns": bool(
                self.last_location_update_runs_all_columns
            ),
            "last_location_update_fallback_reason": self.last_location_update_fallback_reason,
            "cached_consensus_gain_device": str(self.cached_consensus_gain.device),
            "last_vote_update_mode": self.last_vote_update_mode,
            "last_vote_update_count": int(self.last_vote_update_count),
            "last_vote_update_fraction": float(self.last_vote_update_fraction),
            "last_vote_cached_count": int(self.last_vote_cached_count),
            "last_vote_runs_all_columns": bool(self.last_vote_runs_all_columns),
            "last_vote_fallback_reason": self.last_vote_fallback_reason,
            "last_predictive_materialize_mode": self.last_predictive_materialize_mode,
            "last_predictive_materialize_count": int(
                self.last_predictive_materialize_count
            ),
            "last_predictive_materialize_max_age": int(
                self.last_predictive_materialize_max_age
            ),
            "predictive_step_count": int(self.predictive_step_count),
            "predictive_last_update_step_device": str(
                self.predictive_last_update_step.device
            ),
            "predictive_has_cached_columns": bool(
                self._predictive_has_cached_columns
            ),
            "last_dense_transition_mode": self.last_dense_transition_mode,
            "last_dense_transition_fallback_reason": self.last_dense_transition_fallback_reason,
            "dense_transition_compile_count": int(self.dense_transition_compile_count),
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Serialize predictive state for checkpointing."""
        return {
            "location": self.location.detach().clone().cpu(),
            "velocity": self.velocity.detach().clone().cpu(),
            "prediction_weights": self._prediction_weights.detach().clone().cpu(),
            "prediction_error": self.prediction_error.detach().clone().cpu(),
            "prediction_failure_streak": self.prediction_failure_streak.detach().clone().cpu(),
            "confidence": self.confidence.detach().clone().cpu(),
            "predictive_step_count": torch.tensor(
                int(self.predictive_step_count),
                dtype=torch.long,
            ),
            "predictive_last_update_step": (
                self.predictive_last_update_step.detach().clone().cpu()
            ),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore predictive state from checkpoint."""
        for key in (
            "location",
            "velocity",
            "prediction_weights",
            "prediction_error",
            "prediction_failure_streak",
            "confidence",
            "predictive_last_update_step",
        ):
            tensor_key = key if key != "prediction_weights" else "_prediction_weights"
            value = state.get(key)
            if isinstance(value, torch.Tensor):
                target = getattr(self, tensor_key if key != "prediction_weights" else "_prediction_weights")
                if value.shape == target.shape:
                    setattr(
                        self,
                        tensor_key if key != "prediction_weights" else "_prediction_weights",
                        value.to(self.device),
                    )
        step_value = state.get("predictive_step_count")
        if isinstance(step_value, torch.Tensor):
            self.predictive_step_count = int(step_value.item())
        elif isinstance(step_value, (int, float)):
            self.predictive_step_count = int(step_value)
        else:
            self.predictive_step_count = 0
        if "predictive_last_update_step" not in state:
            self.predictive_last_update_step.fill_(int(self.predictive_step_count))
        self._predictive_has_cached_columns = False

    def reset(self) -> None:
        """Reset all predictive state."""
        self.location = torch.randn(
            self.n_columns, self.location_dim, device=self.device
        ) * 0.1
        self.velocity.zero_()
        self._prediction_weights = torch.randn(
            self.n_columns, self.location_dim, device=self.device
        ) * 0.01
        self.prediction_error.zero_()
        self.prediction_failure_streak.zero_()
        self.confidence.fill_(0.5)
        self.last_prediction_update_mode = "not_run"
        self.last_prediction_update_count = 0
        self.last_prediction_update_fraction = 0.0
        self.last_prediction_cached_count = 0
        self.last_prediction_update_runs_all_columns = False
        self.last_prediction_update_fallback_reason = None
        self.last_location_update_mode = "not_run"
        self.last_location_update_count = 0
        self.last_location_cached_count = 0
        self.last_location_update_runs_all_columns = False
        self.last_location_update_fallback_reason = None
        self.cached_consensus_gain = torch.ones(self.n_columns, device=self.device)
        self.last_vote_update_mode = "not_run"
        self.last_vote_update_count = 0
        self.last_vote_update_fraction = 0.0
        self.last_vote_cached_count = 0
        self.last_vote_runs_all_columns = False
        self.last_vote_fallback_reason = None
        self.last_predictive_materialize_mode = "not_run"
        self.last_predictive_materialize_count = 0
        self.last_predictive_materialize_max_age = 0
        self.predictive_step_count = 0
        self.predictive_last_update_step = torch.zeros(
            self.n_columns,
            dtype=torch.long,
            device=self.device,
        )
        self._predictive_has_cached_columns = False
        self._predictive_materialize_learning_rate = 0.005
