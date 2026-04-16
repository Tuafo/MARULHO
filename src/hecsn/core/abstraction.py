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
        self._stable_cache: torch.Tensor | None = None
        self._stable_cache_version: int = -1
        self._state_version: int = 0

    def _normalized_assembly(self, assembly: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(assembly, dtype=torch.float32, device=self.device).flatten()
        if int(values.numel()) != self.n_columns:
            raise ValueError(f"assembly must have {self.n_columns} values, got {int(values.numel())}")
        return _normalize_nonnegative(values, eps=self.eps)

    def _stable_signal(self) -> torch.Tensor:
        if self._stable_cache is not None and self._stable_cache_version == self._state_version:
            return self._stable_cache
        stable = torch.clamp(self.slow_state, min=0.0) * self.concept_stability * self.concept_certainty
        total = stable.sum()
        if float(total) <= self.eps:
            result = torch.zeros_like(stable)
        else:
            result = stable / (total + self.eps)
        self._stable_cache = result
        self._stable_cache_version = self._state_version
        return result

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

        # Invalidate _stable_signal cache (state was mutated)
        self._state_version += 1

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
        total = stable_signal.sum()
        if float(total) <= self.eps:
            return torch.ones(self.n_columns, dtype=torch.float32, device=self.device)

        bias = torch.mv(self.feedback, stable_signal)
        bias = bias - bias.mean()
        max_abs = bias.abs().max()
        if float(max_abs) <= self.eps:
            return torch.ones(self.n_columns, dtype=torch.float32, device=self.device)
        bias = bias / max_abs
        return torch.clamp(
            1.0 + self.feedback_strength * bias,
            min=1.0 - self.feedback_strength,
            max=1.0 + self.feedback_strength,
        )

    def max_curiosity_gap_score(self) -> float:
        """Fast path: return max gap score without building dicts."""
        gap_scores = self.slow_var * (1.0 - self.concept_certainty)
        if int(gap_scores.numel()) <= 0:
            return 0.0
        return float(gap_scores.max())

    def curiosity_gaps(self, top_n: int = 4) -> list[dict[str, float]]:
        gap_scores = self.slow_var * (1.0 - self.concept_certainty)
        if int(gap_scores.numel()) <= 0:
            return []
        k = min(max(1, int(top_n)), int(gap_scores.numel()))
        values, indices = torch.topk(gap_scores, k=k)
        # Batch extract via .tolist() instead of per-element .item()
        idx_list = indices.tolist()
        val_list = values.tolist()
        stab_list = self.concept_stability[indices].tolist()
        cert_list = self.concept_certainty[indices].tolist()
        return [
            {
                "concept_idx": float(idx_list[i]),
                "gap_score": float(val_list[i]),
                "stability": float(stab_list[i]),
                "certainty": float(cert_list[i]),
            }
            for i in range(k)
        ]

    def curiosity_routing_gain(
        self,
        *,
        top_n: int = 4,
        strength: float = 0.05,
        warmup_steps: int = 50,
        gap_threshold: float = 0.01,
    ) -> torch.Tensor | None:
        """Multiplicative routing gain biased toward high-curiosity-gap columns.

        Returns None if insufficient data to produce a reliable signal.
        Uses feedforward[concept_idx] to map concepts → columns.
        """
        if self.updates < warmup_steps:
            return None
        gap_scores = self.slow_var * (1.0 - self.concept_certainty)
        max_gap = float(gap_scores.max())
        if max_gap < gap_threshold:
            return None
        k = min(max(1, int(top_n)), int(gap_scores.numel()))
        values, indices = torch.topk(gap_scores, k=k)
        # Vectorized: weighted sum of feedforward rows
        bonus = (values.unsqueeze(1) * self.feedforward[indices]).sum(0)
        # Mean-center then normalize (same pattern as routing_gain)
        bonus = bonus - bonus.mean()
        max_abs = float(bonus.abs().max())
        if max_abs <= self.eps:
            return None
        bonus = bonus / max_abs
        return torch.clamp(
            1.0 + strength * bonus,
            min=1.0 - strength,
            max=1.0 + strength,
        )

    def reset_state(self) -> None:
        self.fast_state.zero_()
        self.last_activations = None
        self.last_input = None
        self._state_version += 1

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
        self._state_version += 1

    def sfa_correction_step(
        self,
        samples: list[torch.Tensor],
        lr: float = 0.01,
    ) -> dict[str, float]:
        """Mini-batch SFA correction during deep sleep (§4.8).

        Takes a batch of assembly snapshots from slow memory, computes
        the true covariance and temporal-derivative covariance, and
        performs one gradient step toward the exact SFA solution.

        Args:
            samples: List of assembly vectors [n_columns] from slow memory.
            lr: Learning rate for the correction step.

        Returns:
            Metrics: pre/post variance and decorrelation improvement.
        """
        if len(samples) < 2:
            return {"variance_reduction": 0.0, "decorrelation": 0.0, "n_samples": 0}

        # Stack and project through feedforward
        X = torch.stack([
            _normalize_nonnegative(s.to(self.device), eps=self.eps)
            for s in samples
        ])  # [N, n_columns]
        Y = X @ self.feedforward.t()  # [N, n_concepts]

        # Temporal derivative: Y[t] - Y[t-1]
        dY = Y[1:] - Y[:-1]  # [N-1, n_concepts]

        # Covariance of outputs (want whitened)
        Y_centered = Y - Y.mean(dim=0, keepdim=True)
        C_y = (Y_centered.t() @ Y_centered) / max(1, Y_centered.shape[0] - 1)

        # Covariance of temporal derivative (want minimized)
        dY_centered = dY - dY.mean(dim=0, keepdim=True)
        C_dy = (dY_centered.t() @ dY_centered) / max(1, dY_centered.shape[0] - 1)

        # Pre-correction metrics
        pre_output_var = float(torch.diag(C_y).sum().item())
        pre_deriv_var = float(torch.diag(C_dy).sum().item())

        # Off-diagonal magnitude (correlation between concepts)
        mask = 1.0 - torch.eye(self.n_concepts, device=self.device)
        pre_offdiag = float((C_y * mask).abs().sum().item())

        # SFA gradient: minimize E[||dY/dt||²] subject to whitening
        # Approximate: push feedforward toward directions that minimize
        # temporal derivative variance while decorrelating outputs
        #
        # Gradient on W: dL/dW = 2 * C_dy @ W @ X^T X / N
        #                      - λ * (C_y - I) @ W @ X^T X / N
        # Simplified: update W to reduce C_dy diagonal and push C_y → I

        # Decorrelation: push off-diagonal of C_y toward zero
        decorr_gradient = (C_y - torch.eye(self.n_concepts, device=self.device))
        decorr_update = decorr_gradient @ self.feedforward  # [n_concepts, n_columns]

        # Temporal slowness: reduce variance of temporal derivative
        slowness_gradient = C_dy  # [n_concepts, n_concepts]
        slowness_update = slowness_gradient @ self.feedforward  # [n_concepts, n_columns]

        # Combined update
        total_update = 0.5 * decorr_update + 0.5 * slowness_update
        self.feedforward = F.normalize(
            torch.clamp(self.feedforward - lr * total_update, min=1e-6),
            dim=1,
        )

        # Recompute post-correction metrics
        Y_post = X @ self.feedforward.t()
        dY_post = Y_post[1:] - Y_post[:-1]
        Y_post_c = Y_post - Y_post.mean(dim=0, keepdim=True)
        C_y_post = (Y_post_c.t() @ Y_post_c) / max(1, Y_post_c.shape[0] - 1)
        C_dy_post = (dY_post.t() @ dY_post) / max(1, dY_post.shape[0] - 1)

        post_output_var = float(torch.diag(C_y_post).sum().item())
        post_deriv_var = float(torch.diag(C_dy_post).sum().item())
        post_offdiag = float((C_y_post * mask).abs().sum().item())

        return {
            "n_samples": len(samples),
            "pre_output_var": pre_output_var,
            "post_output_var": post_output_var,
            "pre_deriv_var": pre_deriv_var,
            "post_deriv_var": post_deriv_var,
            "variance_reduction": max(0.0, pre_deriv_var - post_deriv_var),
            "pre_offdiag": pre_offdiag,
            "post_offdiag": post_offdiag,
            "decorrelation": max(0.0, pre_offdiag - post_offdiag),
        }
