from __future__ import annotations

from typing import Any

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
        if float(prediction.sum().item()) <= 0.0:
            return torch.ones(self.n_columns, device=self.device)

        centered = prediction - prediction.mean()
        scale = torch.clamp(centered.abs().max(), min=1e-8)
        effective_strength = self.modulation_strength * (0.5 + 0.5 * float(norepinephrine))
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
        if float(current.sum().item()) <= 0.0:
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
        if float(previous_state.sum().item()) <= 0.0:
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


class BindingLayer:
    """Coincidence binding with short-term facilitation, depression, and PV inhibition."""

    def __init__(
        self,
        n_columns: int,
        device: torch.device,
        threshold: float = 0.02,
        association_lr: float = 0.10,
        association_decay: float = 0.995,
        gain_strength: float = 0.35,
        tau_binding: float = 6.0,
        stp_u_inc: float = 0.15,
        stp_tau_f: float = 12.0,
        stp_tau_d: float = 4.0,
        pv_threshold: float = 0.12,
        pv_gain: float = 0.60,
    ) -> None:
        self.n_columns = int(n_columns)
        self.device = device
        self.threshold = float(threshold)
        self.association_lr = float(association_lr)
        self.association_decay = float(association_decay)
        self.gain_strength = float(gain_strength)
        self.tau_binding = float(tau_binding)
        self.stp_u_inc = float(stp_u_inc)
        self.stp_tau_f = float(stp_tau_f)
        self.stp_tau_d = float(stp_tau_d)
        self.pv_threshold = float(pv_threshold)
        self.pv_gain = float(pv_gain)

        self.binding_state = torch.zeros(self.n_columns, device=self.device)
        self.coincidence_trace = torch.zeros(self.n_columns, device=self.device)
        self.facilitation = torch.zeros(self.n_columns, device=self.device)
        self.resources = torch.ones(self.n_columns, device=self.device)
        self.pv_inhibition = torch.tensor(0.0, device=self.device)
        self.coincidence_weights = torch.full(
            (self.n_columns, self.n_columns),
            1.0 / max(1, self.n_columns),
            device=self.device,
        )

    def reset_state(self) -> None:
        self.binding_state.zero_()
        self.coincidence_trace.zero_()
        self.facilitation.zero_()
        self.resources.fill_(1.0)
        self.pv_inhibition.zero_()

    def _binding_prediction(self, context_prediction: torch.Tensor) -> torch.Tensor:
        context = _normalize(context_prediction.to(self.device))
        if float(context.sum().item()) <= 0.0:
            return torch.zeros(self.n_columns, device=self.device)
        predicted = torch.mv(self.coincidence_weights.t(), context)
        return _normalize(torch.relu(predicted))

    def binding_prediction(self, context_prediction: torch.Tensor) -> torch.Tensor:
        return self._binding_prediction(context_prediction)

    def modulation_gain(self, context_prediction: torch.Tensor) -> torch.Tensor:
        return self.modulation_gain_for_context(context_prediction)

    def modulation_gain_for_context(self, context_prediction: torch.Tensor) -> torch.Tensor:
        prediction = self._binding_prediction(context_prediction)
        if float(prediction.sum().item()) <= 0.0:
            return torch.ones(self.n_columns, device=self.device)

        centered = prediction - prediction.mean()
        scale = torch.clamp(centered.abs().max(), min=1e-8)
        gain = 1.0 + self.gain_strength * (centered / scale)
        return torch.clamp(gain, min=0.70, max=1.35)

    def _update_stp(self, signal: torch.Tensor) -> torch.Tensor:
        drive = _normalize(signal.to(self.device))
        self.facilitation = torch.clamp(
            self.facilitation * (1.0 - 1.0 / self.stp_tau_f)
            + self.stp_u_inc * drive * (1.0 - self.facilitation),
            min=0.0,
            max=1.0,
        )
        release = torch.clamp(self.facilitation * self.resources * drive, min=0.0)
        self.resources = torch.clamp(
            self.resources + (1.0 - self.resources) / self.stp_tau_d - release,
            min=0.0,
            max=1.0,
        )
        return release

    def bind(
        self,
        context_prediction: torch.Tensor,
        assembly: torch.Tensor,
        update_weights: bool = True,
    ) -> tuple[torch.Tensor, float]:
        context = _normalize(context_prediction.to(self.device))
        current = _normalize(assembly.to(self.device))
        if float(context.sum().item()) <= 0.0 or float(current.sum().item()) <= 0.0:
            self.coincidence_trace *= max(0.0, 1.0 - 1.0 / self.tau_binding)
            self.binding_state.zero_()
            return torch.zeros_like(current), 0.0

        context_gate = 0.5 + 0.5 * float(context.max().item())
        release = self._update_stp(current) * context_gate
        self.coincidence_trace = torch.clamp(
            self.coincidence_trace * max(0.0, 1.0 - 1.0 / self.tau_binding) + release,
            min=0.0,
        )

        learned_prediction = self._binding_prediction(context)
        activity_sum = float(release.sum().item())
        pv_excess = max(0.0, activity_sum - self.pv_threshold)
        self.pv_inhibition = torch.tensor(
            0.85 * float(self.pv_inhibition.item()) + self.pv_gain * pv_excess,
            device=self.device,
        )

        bound = torch.relu(
            current
            + self.gain_strength * self.coincidence_trace
            + 0.50 * self.gain_strength * learned_prediction
            - float(self.pv_inhibition.item()) * current.mean()
            - self.threshold
        )
        strength = float(torch.minimum(current, self.coincidence_trace + learned_prediction).sum().item())

        if update_weights:
            updated = self.association_decay * self.coincidence_weights + self.association_lr * torch.outer(context, current)
            self.coincidence_weights = _row_normalize(torch.clamp(updated, min=1e-6))

        if float(bound.sum().item()) <= 0.0:
            self.binding_state.zero_()
            return torch.zeros_like(bound), strength

        self.binding_state = _normalize(bound)
        return self.binding_state, strength

    def state_dict(self) -> dict[str, Any]:
        return {
            "binding_state": self.binding_state.detach().clone().cpu(),
            "coincidence_trace": self.coincidence_trace.detach().clone().cpu(),
            "facilitation": self.facilitation.detach().clone().cpu(),
            "resources": self.resources.detach().clone().cpu(),
            "pv_inhibition": self.pv_inhibition.detach().clone().cpu(),
            "coincidence_weights": self.coincidence_weights.detach().clone().cpu(),
        }

    def load_state_dict(self, snapshot: dict[str, Any]) -> None:
        for attr in (
            "binding_state",
            "coincidence_trace",
            "facilitation",
            "resources",
            "pv_inhibition",
            "coincidence_weights",
        ):
            value = snapshot.get(attr)
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(self.device))
