"""Context layers -- multi-timescale temporal integration for competitive routing.

Two implementations:
- ContextLayer: Fixed 3-trace (fast/medium/slow) context with SST+ inhibition.
- AdaptiveContextLayer: Per-neuron learnable timescales (tau_min to tau_max),
  inspired by DH-SNN heterogeneous temporal dynamics.

The context layer provides routing gain to the CompetitiveColumnLayer, modulating
which columns are more likely to win based on recent temporal patterns.

Factory: use create_context_layer(mode, n_columns, device) to instantiate.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch


def _normalize(vec: torch.Tensor) -> torch.Tensor:
    vec = torch.clamp(vec.float(), min=0.0)
    total = torch.clamp(vec.sum(), min=1e-8)
    return vec / total


def _row_normalize(matrix: torch.Tensor) -> torch.Tensor:
    row_sums = torch.clamp(matrix.sum(dim=1, keepdim=True), min=1e-8)
    return matrix / row_sums


class ContextLayer:
    """Multi-timescale approximate attractor context over column assemblies.

    The maintained runtime still operates over column assemblies rather than a
    neuron-level AdEx microcircuit, but this layer now matches the paper's
    intended structure much more closely:
    - fast / medium / slow context traces for heterogeneous integration,
    - learned recurrent connectivity over a dense local manifold,
    - fast inhibitory stabilization to keep the manifold approximate rather
      than unstable.

    This is motivated by the paper's own Nair/Sagodi discussion and the later
    heterogeneous-timescale direction highlighted by HetSyn.
    """

    def __init__(
        self,
        n_columns: int,
        device: torch.device,
        decay: float = 0.92,
        transition_lr: float = 0.05,
        modulation_strength: float = 0.30,
        fast_rate: float = 0.55,
        medium_rate: float = 0.25,
        slow_rate: float = 0.08,
        recurrent_density: float = 0.35,
        recurrent_scale: float = 0.85,
        inhibition_strength: float = 0.25,
    ) -> None:
        self.n_columns = int(n_columns)
        self.device = device
        self.decay = float(decay)
        self.transition_lr = float(transition_lr)
        self.modulation_strength = float(modulation_strength)
        self.fast_rate = float(fast_rate)
        self.medium_rate = float(medium_rate)
        self.slow_rate = float(slow_rate)
        self.recurrent_density = float(recurrent_density)
        self.recurrent_scale = float(recurrent_scale)
        self.inhibition_strength = float(inhibition_strength)

        self.fast_state = torch.zeros(self.n_columns, device=self.device)
        self.medium_state = torch.zeros(self.n_columns, device=self.device)
        self.slow_state = torch.zeros(self.n_columns, device=self.device)
        self.inhibitory_state = torch.zeros(self.n_columns, device=self.device)
        self.state = torch.zeros(self.n_columns, device=self.device)
        self.last_precision_weight = 1.0

        self.feedforward = torch.eye(self.n_columns, device=self.device)
        self.no_self_mask = 1.0 - torch.eye(self.n_columns, device=self.device)
        self.recurrent_mask = (
            torch.rand(self.n_columns, self.n_columns, device=self.device) < self.recurrent_density
        ).float()
        raw_recurrent = torch.rand(self.n_columns, self.n_columns, device=self.device) * self.recurrent_mask
        raw_recurrent.fill_diagonal_(1.0)
        self.recurrent = _row_normalize(raw_recurrent)
        self.recurrent = self.recurrent * self.recurrent_scale

    def reset_state(self) -> None:
        self.fast_state.zero_()
        self.medium_state.zero_()
        self.slow_state.zero_()
        self.inhibitory_state.zero_()
        self.state.zero_()

    def context_prediction(self) -> torch.Tensor:
        if float(self.state.sum().item()) <= 0.0:
            return torch.zeros(self.n_columns, device=self.device)

        transition_recurrent = self.recurrent * self.no_self_mask
        fast_prediction = torch.mv(transition_recurrent.t(), self.fast_state)
        medium_prediction = torch.mv(transition_recurrent.t(), self.medium_state)
        slow_prediction = torch.mv(transition_recurrent.t(), self.slow_state)
        prediction = (
            0.20 * fast_prediction
            + 0.35 * medium_prediction
            + 0.45 * slow_prediction
            - 0.15 * self.inhibitory_state
        )
        return _normalize(torch.relu(prediction))

    def modulation_gain(
        self,
        norepinephrine: float = 0.5,
        acetylcholine: float = 0.5,
    ) -> torch.Tensor:
        prediction = self.context_prediction()
        return self.modulation_gain_for_signal(
            prediction,
            norepinephrine=norepinephrine,
            acetylcholine=acetylcholine,
        )

    def modulation_gain_for_signal(
        self,
        signal: torch.Tensor,
        norepinephrine: float = 0.5,
        acetylcholine: float = 0.5,
    ) -> torch.Tensor:
        prediction = _normalize(signal.to(self.device))
        if prediction.sum() <= 0.0:
            return torch.ones(self.n_columns, device=self.device)

        centered = prediction - prediction.mean()
        scale = torch.clamp(centered.abs().max(), min=1e-8)
        effective_strength = self.modulation_strength * (0.5 + 0.5 * norepinephrine)
        effective_strength *= 0.5 + 0.5 * float(acetylcholine)
        gain = 1.0 + effective_strength * (centered / scale)
        return torch.clamp(gain, min=0.65, max=1.35)

    def _integration_scale(self, precision_weight: float | None) -> float:
        if precision_weight is None:
            self.last_precision_weight = 1.0
            return 1.0
        weight = max(0.0, min(1.0, float(precision_weight)))
        self.last_precision_weight = weight
        return 0.25 + 0.75 * weight

    def observe(
        self,
        assembly: torch.Tensor,
        update_weights: bool = True,
        *,
        precision_weight: float | None = None,
    ) -> torch.Tensor:
        current = _normalize(assembly.to(self.device))
        previous_state = self.state.clone()
        if current.sum() <= 0.0:
            self.fast_state.mul_(1.0 - self.fast_rate)
            self.medium_state.mul_(self.decay)
            self.slow_state.mul_(self.decay)
            self.inhibitory_state.mul_(self.decay)
            self.state = _normalize(self.fast_state + self.medium_state + self.slow_state)
            return self.state

        recurrent_fast = torch.mv(self.recurrent, self.fast_state)
        recurrent_medium = torch.mv(self.recurrent, self.medium_state)
        recurrent_slow = torch.mv(self.recurrent, self.slow_state)
        feedforward_drive = torch.mv(self.feedforward, current)
        integration_scale = self._integration_scale(precision_weight)
        fast_rate = min(1.0, self.fast_rate * integration_scale)
        medium_rate = min(1.0, self.medium_rate * integration_scale)
        slow_rate = min(1.0, self.slow_rate * integration_scale)
        transition_lr = self.transition_lr * integration_scale

        mean_activity = (
            0.20 * self.fast_state.mean()
            + 0.30 * self.medium_state.mean()
            + 0.50 * self.slow_state.mean()
        )
        self.inhibitory_state = torch.relu(
            self.decay * self.inhibitory_state
            + self.inhibition_strength * mean_activity
        )

        fast_drive = torch.relu(feedforward_drive + 0.35 * recurrent_fast - self.inhibitory_state)
        medium_drive = torch.relu(
            0.65 * feedforward_drive + 0.35 * recurrent_medium + 0.15 * recurrent_fast - 0.60 * self.inhibitory_state
        )
        slow_drive = torch.relu(
            0.35 * feedforward_drive + 0.65 * recurrent_slow + 0.20 * recurrent_medium - 0.35 * self.inhibitory_state
        )

        self.fast_state = _normalize((1.0 - fast_rate) * self.fast_state + fast_rate * fast_drive)
        self.medium_state = _normalize(self.decay * self.medium_state + medium_rate * medium_drive)
        self.slow_state = _normalize(self.decay * self.slow_state + slow_rate * slow_drive)
        self.state = _normalize(0.20 * self.fast_state + 0.35 * self.medium_state + 0.45 * self.slow_state)

        if not update_weights:
            return self.state
        if previous_state.sum() <= 0.0:
            return self.state

        hebbian_update = torch.outer(previous_state, self.state) * self.recurrent_mask
        updated = self.decay * self.recurrent + transition_lr * hebbian_update
        self.recurrent = _row_normalize(torch.clamp(updated, min=1e-6)) * self.recurrent_scale
        return self.state

    def state_dict(self) -> dict[str, Any]:
        return {
            "fast_state": self.fast_state.detach().clone().cpu(),
            "medium_state": self.medium_state.detach().clone().cpu(),
            "slow_state": self.slow_state.detach().clone().cpu(),
            "inhibitory_state": self.inhibitory_state.detach().clone().cpu(),
            "state": self.state.detach().clone().cpu(),
            "feedforward": self.feedforward.detach().clone().cpu(),
            "recurrent_mask": self.recurrent_mask.detach().clone().cpu(),
            "recurrent": self.recurrent.detach().clone().cpu(),
            "last_precision_weight": float(self.last_precision_weight),
        }

    def load_state_dict(self, snapshot: dict[str, Any]) -> None:
        for attr in (
            "fast_state",
            "medium_state",
            "slow_state",
            "inhibitory_state",
            "state",
            "feedforward",
            "recurrent_mask",
            "recurrent",
        ):
            value = snapshot.get(attr)
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(self.device))
        self.last_precision_weight = float(snapshot.get("last_precision_weight", 1.0))


class AdaptiveContextLayer:
    """Context layer with learnable per-neuron time constants (§4.3).

    Replaces the fixed fast/medium/slow three-trace design with a continuous
    distribution of learned timescales, initialized log-uniformly over
    [tau_min, tau_max].

    Based on DH-SNN (Li et al. 2023): heterogeneous learned tau outperforms
    fixed single-timescale on every temporal benchmark tested.

    API-compatible with ContextLayer: same observe(), context_prediction(),
    modulation_gain(), state_dict(), load_state_dict().
    """

    def __init__(
        self,
        n_columns: int,
        device: torch.device,
        n_neurons: int | None = None,
        tau_min: float = 2.0,
        tau_max: float = 500.0,
        inhibition_strength: float = 0.30,
        modulation_strength: float = 0.30,
        transition_lr: float = 0.05,
    ) -> None:
        self.n_columns = int(n_columns)
        self.device = device
        self.n_neurons = int(n_neurons or n_columns)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.inhibition_strength = float(inhibition_strength)
        self.modulation_strength = float(modulation_strength)
        self.transition_lr = float(transition_lr)
        self.last_precision_weight = 1.0

        # Learnable time constants, initialized log-uniformly
        self.log_tau = torch.linspace(
            math.log(tau_min), math.log(tau_max), self.n_neurons,
            device=device,
        )

        # Context state per neuron
        self.neuron_state = torch.zeros(self.n_neurons, device=device)

        # Column-space state (for API compatibility with ContextLayer)
        self.state = torch.zeros(self.n_columns, device=device)

        # Projection: column assemblies → context neurons
        w_in = torch.randn(self.n_neurons, self.n_columns, device=device) * 0.1
        self.w_in = w_in / (torch.norm(w_in, dim=1, keepdim=True) + 1e-8)

        # Projection: context state → routing gain per column
        w_out = torch.randn(self.n_columns, self.n_neurons, device=device) * 0.1
        self.w_out = w_out / (torch.norm(w_out, dim=1, keepdim=True) + 1e-8)

        # Context-specificity buffer for routing_differentiation (§4.3).
        # Each entry is (input_signature, neuron_state_snapshot).
        # input_signature groups observations by "what input was presented"
        # so we can measure "how much did different contexts change the response."
        self._context_observations: list[tuple[tuple[int, ...], torch.Tensor]] = []
        self._context_observations_maxlen = 200

    def reset_state(self) -> None:
        self.neuron_state.zero_()
        self.state.zero_()
        self._context_observations.clear()

    def _tau(self) -> torch.Tensor:
        """Effective time constants, clamped to [tau_min, tau_max]."""
        return torch.exp(self.log_tau).clamp(min=self.tau_min, max=self.tau_max)

    def _decay_factors(self, dt: float = 1.0) -> torch.Tensor:
        """Per-neuron exponential decay factors."""
        tau = self._tau()
        return torch.exp(-dt / tau)

    def context_prediction(self) -> torch.Tensor:
        if self.neuron_state.sum() <= 0.0:
            return torch.zeros(self.n_columns, device=self.device)

        # SST+ inhibition
        mean_activity = self.neuron_state.mean()
        inhibited = self.neuron_state - self.inhibition_strength * mean_activity

        # Project to column space
        prediction = torch.mv(self.w_out, torch.relu(inhibited))
        return _normalize(torch.relu(prediction))

    def modulation_gain(
        self,
        norepinephrine: float = 0.5,
        acetylcholine: float = 0.5,
    ) -> torch.Tensor:
        prediction = self.context_prediction()
        return self.modulation_gain_for_signal(
            prediction,
            norepinephrine=norepinephrine,
            acetylcholine=acetylcholine,
        )

    def modulation_gain_for_signal(
        self,
        signal: torch.Tensor,
        norepinephrine: float = 0.5,
        acetylcholine: float = 0.5,
    ) -> torch.Tensor:
        prediction = _normalize(signal.to(self.device))
        if prediction.sum() <= 0.0:
            return torch.ones(self.n_columns, device=self.device)

        centered = prediction - prediction.mean()
        scale = torch.clamp(centered.abs().max(), min=1e-8)
        effective_strength = self.modulation_strength * (0.5 + 0.5 * norepinephrine)
        effective_strength *= 0.5 + 0.5 * float(acetylcholine)
        gain = 1.0 + effective_strength * (centered / scale)
        return torch.clamp(gain, min=0.65, max=1.35)

    def observe(
        self,
        assembly: torch.Tensor,
        update_weights: bool = True,
        *,
        precision_weight: float | None = None,
    ) -> torch.Tensor:
        raw = assembly.to(self.device)
        if raw.dim() > 1:
            raw = raw.squeeze(0)
        current = _normalize(raw)

        if precision_weight is not None:
            self.last_precision_weight = max(0.0, min(1.0, float(precision_weight)))
        else:
            self.last_precision_weight = 1.0

        if current.sum() <= 0.0:
            alpha = self._decay_factors()
            self.neuron_state = alpha * self.neuron_state
            self.state = _normalize(torch.mv(self.w_out, torch.relu(self.neuron_state)))
            return self.state

        # Drive from assembly into neuron space
        drive = torch.sigmoid(torch.mv(self.w_in, current))

        # Leaky integration with per-neuron timescale
        alpha = self._decay_factors()
        self.neuron_state = alpha * self.neuron_state + (1.0 - alpha) * drive

        # Project back to column space for state
        inhibited = self.neuron_state - self.inhibition_strength * self.neuron_state.mean()
        self.state = _normalize(torch.relu(torch.mv(self.w_out, torch.relu(inhibited))))

        # Hebbian weight update (slow)
        if update_weights:
            integration_scale = 0.25 + 0.75 * self.last_precision_weight
            lr = self.transition_lr * integration_scale
            # W_in update: outer product of neuron state and assembly
            delta_in = lr * torch.outer(torch.relu(self.neuron_state), current)
            self.w_in = self.w_in + delta_in
            # Re-normalize rows
            norms = torch.norm(self.w_in, dim=1, keepdim=True)
            self.w_in = self.w_in / (norms + 1e-8)

        # Record (input_signature, neuron_state) for routing_differentiation (wake only)
        if update_weights:
            sig = self._compute_input_signature(current)
            self._context_observations.append((sig, self.neuron_state.detach().clone()))
            if len(self._context_observations) > self._context_observations_maxlen:
                self._context_observations.pop(0)

        return self.state

    @staticmethod
    def _compute_input_signature(assembly: torch.Tensor, k: int = 8) -> tuple[int, ...]:
        """Compact hash of assembly for grouping same-input observations.

        Uses top-k indices + coarse quantized values (3 bins: low/mid/high)
        to group observations by input identity while tolerating minor noise.
        """
        k_actual = min(k, assembly.numel())
        vals, idxs = torch.topk(assembly.abs(), k_actual)
        # Quantize values into 3 bins for noise tolerance
        vmax = vals[0]
        if vmax > 0:
            bins = (vals * (2.99 / vmax)).long().clamp(0, 2)
        else:
            bins = torch.zeros_like(vals, dtype=torch.long)
        # Batch convert to Python lists (single device→host sync)
        idx_list = idxs.tolist()
        bin_list = bins.tolist()
        return tuple(v for pair in zip(idx_list, bin_list) for v in pair)

    def compute_routing_differentiation(self) -> torch.Tensor:
        """Per-neuron context-specificity over recent observations (§4.3).

        Groups observations by input signature (what assembly was presented),
        then for each input seen multiple times under different preceding
        contexts, computes the variance of neuron states.  High variance for
        the *same* input means the neuron is context-sensitive — its state
        depends on what came before, not just the current stimulus.

        Returns a tensor of shape [n_neurons].  Returns zeros if fewer than
        3 input signatures have been observed at least twice.
        """
        if len(self._context_observations) < 10:
            return torch.zeros(self.n_neurons, device=self.device)

        # Group neuron states by input signature
        groups: dict[tuple[int, ...], list[torch.Tensor]] = {}
        for sig, state in self._context_observations:
            groups.setdefault(sig, []).append(state)

        # Compute per-neuron variance within each group that has ≥2 observations
        variances: list[torch.Tensor] = []
        for states in groups.values():
            if len(states) >= 2:
                stacked = torch.stack(states)  # [n_repeats, n_neurons]
                variances.append(stacked.var(dim=0))  # [n_neurons]

        if len(variances) < 3:
            return torch.zeros(self.n_neurons, device=self.device)

        return torch.stack(variances).mean(dim=0)  # [n_neurons]

    def update_timescales(
        self,
        routing_differentiation: torch.Tensor,
        lr_tau: float = 0.001,
    ) -> None:
        """Adapt tau based on routing differentiation (§4.3).

        Neurons with high differentiation → increase tau (slower, more context).
        Neurons with low differentiation → decrease tau (faster, less overhead).
        """
        diff = routing_differentiation.to(self.device)
        if diff.shape[0] != self.n_neurons:
            # Truncate or pad
            if diff.shape[0] > self.n_neurons:
                diff = diff[: self.n_neurons]
            else:
                padded = torch.zeros(self.n_neurons, device=self.device)
                padded[: diff.shape[0]] = diff
                diff = padded

        self.log_tau += lr_tau * (diff - diff.mean())
        self.log_tau.clamp_(math.log(self.tau_min), math.log(self.tau_max))

    def tau_distribution(self) -> dict[str, float]:
        """Report current tau distribution statistics."""
        tau = self._tau()
        return {
            "tau_min": float(tau.min().item()),
            "tau_max": float(tau.max().item()),
            "tau_mean": float(tau.mean().item()),
            "tau_std": float(tau.std().item()),
            "tau_median": float(tau.median().item()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "log_tau": self.log_tau.detach().clone().cpu(),
            "neuron_state": self.neuron_state.detach().clone().cpu(),
            "state": self.state.detach().clone().cpu(),
            "w_in": self.w_in.detach().clone().cpu(),
            "w_out": self.w_out.detach().clone().cpu(),
            "last_precision_weight": float(self.last_precision_weight),
            "context_mode": "adaptive",
        }

    def load_state_dict(self, snapshot: dict[str, Any]) -> None:
        for attr in ("log_tau", "neuron_state", "state", "w_in", "w_out"):
            value = snapshot.get(attr)
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(self.device))
        self.last_precision_weight = float(snapshot.get("last_precision_weight", 1.0))


def create_context_layer(
    mode: Literal["fixed", "adaptive"],
    n_columns: int,
    device: torch.device,
    **kwargs: Any,
) -> ContextLayer | AdaptiveContextLayer:
    """Factory function for context layers (§4.3).

    Args:
        mode: "fixed" for the original 3-trace ContextLayer,
              "adaptive" for the new learnable-tau AdaptiveContextLayer.
        n_columns: Number of columns.
        device: Torch device.
        **kwargs: Extra keyword arguments passed to the constructor.

    Returns:
        A ContextLayer or AdaptiveContextLayer instance.
    """
    if mode == "adaptive":
        return AdaptiveContextLayer(n_columns=n_columns, device=device, **kwargs)
    return ContextLayer(n_columns=n_columns, device=device, **kwargs)



# Re-export BindingLayer for backwards compatibility
from hecsn.core.binding import BindingLayer  # noqa: E402, F401
