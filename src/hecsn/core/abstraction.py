from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _normalize_nonnegative(signal: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    values = torch.clamp(signal.float(), min=0.0)
    total = torch.clamp(values.sum(), min=eps)
    return values / total


class AbstractionLayer:
    """First-class slow abstraction layer with top-down routing feedback."""

    def __init__(
        self,
        *,
        n_columns: int,
        n_concepts: int,
        device: torch.device,
        slow_rate: float = 0.05,
        fast_rate: float = 0.30,
        learning_rate: float = 0.02,
        feedback_lr: float = 0.05,
        feedback_strength: float = 0.15,
        eps: float = 1e-6,
    ) -> None:
        self.n_columns = int(n_columns)
        self.n_concepts = int(max(1, n_concepts))
        self.device = device
        self.slow_rate = float(max(0.0, min(1.0, slow_rate)))
        self.fast_rate = float(max(0.0, min(1.0, fast_rate)))
        self.learning_rate = float(max(0.0, learning_rate))
        self.feedback_lr = float(max(0.0, feedback_lr))
        self.feedback_strength = float(max(0.0, feedback_strength))
        self.eps = float(max(1e-8, eps))

        generator = torch.Generator()
        generator.manual_seed(1879 + self.n_columns * 17 + self.n_concepts * 31)
        feedforward = torch.rand(self.n_concepts, self.n_columns, dtype=torch.float32, generator=generator)
        feedback = feedforward.t().contiguous()
        self.feedforward = F.normalize(torch.clamp(feedforward, min=1e-6), dim=1).to(self.device)
        self.feedback = F.normalize(torch.clamp(feedback, min=1e-6), dim=1).to(self.device)

        self.fast_state = torch.zeros(self.n_concepts, dtype=torch.float32, device=self.device)
        self.slow_state = torch.zeros(self.n_concepts, dtype=torch.float32, device=self.device)
        self.slow_var = torch.ones(self.n_concepts, dtype=torch.float32, device=self.device)
        self.concept_stability = torch.zeros(self.n_concepts, dtype=torch.float32, device=self.device)
        self.concept_certainty = torch.full((self.n_concepts,), 0.5, dtype=torch.float32, device=self.device)
        self.last_activations: torch.Tensor | None = None
        self.last_input: torch.Tensor | None = None
        self.updates = 0

    def _normalized_assembly(self, assembly: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(assembly, dtype=torch.float32, device=self.device).flatten()
        if int(values.numel()) != self.n_columns:
            raise ValueError(f"assembly must have {self.n_columns} values, got {int(values.numel())}")
        return _normalize_nonnegative(values, eps=self.eps)

    def _stable_signal(self) -> torch.Tensor:
        stable = torch.clamp(self.slow_state, min=0.0) * self.concept_stability * self.concept_certainty
        if float(stable.sum().item()) <= self.eps:
            return torch.zeros_like(stable)
        return stable / (stable.sum() + self.eps)

    def observe(
        self,
        assembly: torch.Tensor,
        *,
        update_weights: bool,
        precision_weight: float = 1.0,
    ) -> torch.Tensor:
        x = self._normalized_assembly(assembly)
        precision = float(max(0.0, min(1.0, precision_weight)))
        raw = torch.mv(self.feedforward, x)

        fast_rate = self.fast_rate * precision
        slow_rate = self.slow_rate * precision
        self.fast_state = (1.0 - fast_rate) * self.fast_state + fast_rate * raw
        self.slow_state = (1.0 - slow_rate) * self.slow_state + slow_rate * raw

        fast_deviation = (self.fast_state - self.slow_state).pow(2)
        self.slow_var = (1.0 - slow_rate) * self.slow_var + slow_rate * fast_deviation
        self.concept_stability = torch.clamp(1.0 / (1.0 + self.slow_var), min=0.0, max=1.0)

        certainty_target = torch.sigmoid(4.0 * raw)
        self.concept_certainty = (1.0 - slow_rate) * self.concept_certainty + slow_rate * certainty_target

        if update_weights and precision > 0.0:
            stable_signal = self._stable_signal()
            if self.last_activations is None:
                instability = torch.zeros_like(raw)
            else:
                instability = torch.abs(raw - self.last_activations)
            feedforward_delta = torch.outer(stable_signal + 0.1 * certainty_target, x) - 0.5 * instability.unsqueeze(1) * self.feedforward
            self.feedforward = F.normalize(
                torch.clamp(self.feedforward + self.learning_rate * precision * feedforward_delta, min=1e-6),
                dim=1,
            )
            feedback_delta = torch.outer(x, stable_signal + 0.25 * certainty_target)
            self.feedback = F.normalize(
                torch.clamp(self.feedback + self.feedback_lr * precision * feedback_delta, min=1e-6),
                dim=1,
            )
            self.updates += 1

        self.last_activations = raw.clone()
        self.last_input = x.clone()
        return self._stable_signal()

    def routing_gain(self) -> torch.Tensor:
        stable_signal = self._stable_signal()
        if float(stable_signal.sum().item()) <= self.eps:
            return torch.ones(self.n_columns, dtype=torch.float32, device=self.device)

        bias = torch.mv(self.feedback, stable_signal)
        bias = bias - bias.mean()
        max_abs = float(bias.abs().max().item())
        if max_abs <= self.eps:
            return torch.ones(self.n_columns, dtype=torch.float32, device=self.device)
        bias = bias / max_abs
        return torch.clamp(
            1.0 + self.feedback_strength * bias,
            min=1.0 - self.feedback_strength,
            max=1.0 + self.feedback_strength,
        )

    def curiosity_gaps(self, top_n: int = 4) -> list[dict[str, float]]:
        gap_scores = self.slow_var * (1.0 - self.concept_certainty)
        if int(gap_scores.numel()) <= 0:
            return []
        k = min(max(1, int(top_n)), int(gap_scores.numel()))
        values, indices = torch.topk(gap_scores, k=k)
        return [
            {
                "concept_idx": float(indices[i].item()),
                "gap_score": float(values[i].item()),
                "stability": float(self.concept_stability[indices[i]].item()),
                "certainty": float(self.concept_certainty[indices[i]].item()),
            }
            for i in range(k)
        ]

    def reset_state(self) -> None:
        self.fast_state.zero_()
        self.last_activations = None
        self.last_input = None

    def summary(self) -> dict[str, Any]:
        return {
            "mode": "first_class_sfa_feedback_layer",
            "n_concepts": int(self.n_concepts),
            "n_columns": int(self.n_columns),
            "updates": int(self.updates),
            "mean_stability": float(self.concept_stability.mean().item()),
            "mean_certainty": float(self.concept_certainty.mean().item()),
            "feedback_strength": float(self.feedback_strength),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "feedforward": self.feedforward.detach().clone().cpu(),
            "feedback": self.feedback.detach().clone().cpu(),
            "fast_state": self.fast_state.detach().clone().cpu(),
            "slow_state": self.slow_state.detach().clone().cpu(),
            "slow_var": self.slow_var.detach().clone().cpu(),
            "concept_stability": self.concept_stability.detach().clone().cpu(),
            "concept_certainty": self.concept_certainty.detach().clone().cpu(),
            "last_activations": None if self.last_activations is None else self.last_activations.detach().clone().cpu(),
            "last_input": None if self.last_input is None else self.last_input.detach().clone().cpu(),
            "updates": int(self.updates),
        }

    def load_state_dict(self, snapshot: dict[str, Any] | None) -> None:
        if not snapshot:
            return
        for attr in (
            "feedforward",
            "feedback",
            "fast_state",
            "slow_state",
            "slow_var",
            "concept_stability",
            "concept_certainty",
        ):
            value = snapshot.get(attr)
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(self.device))
        for attr in ("last_activations", "last_input"):
            value = snapshot.get(attr)
            setattr(self, attr, None if value is None else value.to(self.device))
        self.updates = int(snapshot.get("updates", self.updates))
