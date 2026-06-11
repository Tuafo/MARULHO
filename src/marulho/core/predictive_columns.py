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
from typing import Optional

import torch
import torch.nn.functional as F


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

        # Voting state -- each column's hypothesis about active concept
        self.hypothesis = torch.zeros(n_columns, device=self.device)

        self.last_prediction_update_mode = "all_columns"
        self.last_prediction_update_count = int(n_columns)
        self.last_prediction_update_fraction = 1.0
        self.last_prediction_update_fallback_reason: str | None = None

    def _candidate_update_indices(
        self,
        candidate_indices: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if candidate_indices is None or int(candidate_indices.numel()) == 0:
            return None
        return candidate_indices.to(device=self.device, dtype=torch.long).flatten()

    def _record_prediction_update_scope(
        self,
        candidate_indices: torch.Tensor | None,
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
        self.last_prediction_update_fallback_reason = None

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
    ) -> None:
        """Update location state via path integration.

        The "movement" signal is the transition between consecutive inputs.
        Winner columns integrate this into their location state.
        """
        if prev_routing_key is not None:
            # Compute movement as difference between consecutive inputs
            # Project to location_dim via simple hash
            diff = routing_key - prev_routing_key
            # Use first location_dim elements as movement signal
            movement = diff[:self.location_dim] if diff.shape[0] >= self.location_dim else \
                F.pad(diff, (0, self.location_dim - diff.shape[0]))

            # Path integration: update velocity and position for winners
            decay = 0.9
            self.velocity *= decay
            for w in winners:
                self.velocity[w] += 0.1 * movement.to(self.device)
                self.location[w] += self.velocity[w]

        # Normalize locations to prevent drift to infinity
        loc_norm = self.location.norm(dim=1, keepdim=True).clamp(min=1e-8)
        scale = torch.where(loc_norm > 5.0, 5.0 / loc_norm, torch.ones_like(loc_norm))
        self.location *= scale

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
        self._record_prediction_update_scope(candidates)
        lr = float(learning_rate)
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

    def vote(self, winners: list[int], top_k_activations: torch.Tensor) -> torch.Tensor:
        """Inter-column voting to reach consensus.

        Winner columns broadcast their "hypothesis" (confidence-weighted
        activation). Other columns with compatible hypotheses get boosted.

        Returns a consensus gain vector [n_columns] that modulates
        the next competitive step.
        """
        # Each winner votes with its confidence
        self.hypothesis.zero_()
        for w in winners:
            self.hypothesis[w] = self.confidence[w]

        # Compute agreement: columns whose location is similar to winners
        # get a consensus boost (they're "in the same reference frame")
        if not winners:
            return torch.ones(self.n_columns, device=self.device)

        winner_locs = self.location[winners]  # [k, location_dim]
        # Cosine similarity between each column's location and winner centroid
        centroid = winner_locs.mean(dim=0)  # [location_dim]
        centroid_norm = centroid.norm().clamp(min=1e-8)

        loc_norms = self.location.norm(dim=1, keepdim=True).clamp(min=1e-8)
        similarities = (self.location @ centroid) / (loc_norms.squeeze() * centroid_norm)

        # Convert to gain: similar locations get boost, dissimilar get suppression
        consensus_gain = 1.0 + 0.3 * similarities.clamp(-1, 1)
        return consensus_gain

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
            "hypothesis_device": str(self.hypothesis.device),
            "n_columns": int(self.n_columns),
            "location_dim": int(self.location_dim),
            "last_prediction_update_mode": self.last_prediction_update_mode,
            "last_prediction_update_count": int(self.last_prediction_update_count),
            "last_prediction_update_fraction": float(self.last_prediction_update_fraction),
            "last_prediction_update_fallback_reason": self.last_prediction_update_fallback_reason,
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
            "hypothesis": self.hypothesis.detach().clone().cpu(),
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
            "hypothesis",
        ):
            tensor_key = key if key != "prediction_weights" else "_prediction_weights"
            attr_name = tensor_key if not key.startswith("_") else key
            value = state.get(key)
            if isinstance(value, torch.Tensor):
                target = getattr(self, tensor_key if key != "prediction_weights" else "_prediction_weights")
                if value.shape == target.shape:
                    setattr(
                        self,
                        tensor_key if key != "prediction_weights" else "_prediction_weights",
                        value.to(self.device),
                    )

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
        self.hypothesis.zero_()
