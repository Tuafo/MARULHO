from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from hecsn.core.adex import AdExNeuron


def _normalize_nonnegative(signal: torch.Tensor) -> torch.Tensor:
    signal = torch.clamp(signal.float(), min=0.0)
    total = torch.clamp(signal.sum(), min=1e-8)
    return signal / total


def _normalize_columns(weights: torch.Tensor, target_norm: float) -> torch.Tensor:
    return F.normalize(torch.clamp(weights, min=1e-6), dim=0) * float(target_norm)


class LocalPlasticityCircuit:
    """Local three-factor plasticity for the maintained competitive scaffold.

    This circuit replaces the older ad hoc "spike eligibility" branch with one
    stateful subsystem that owns:
    - excitatory log-STDP-style eligibility traces,
    - iSTDP-style inhibitory balancing via adaptive inhibitory tone,
    - firing-rate-dependent synaptic scaling,
    - plastic input and projection synapses.

    It is still a columnar proxy rather than the paper's full recurrent AdEx
    microcircuit, but it now exposes the paper-backed local learning motifs in
    the executable path instead of only describing them in `hecsn.md`.
    """

    def __init__(
        self,
        *,
        n_columns: int,
        input_dim: int,
        column_dim: int,
        device: torch.device,
        input_stdp_ltp: float,
        input_stdp_ltd: float,
        trace_tau: float,
        eligibility_tau: float,
        stdp_mu_plus: float,
        stdp_mu_minus: float,
        synaptic_scaling_alpha: float,
        inhibitory_plasticity_lr: float,
        inhibitory_decay: float,
        target_firing_rate: float,
        input_row_target: float,
        projection_norm_target: float,
        projection_plasticity_scale: float,
        assembly_projection_plasticity_scale: float,
        spike_backend: str = "proxy",
        plasticity_rule: str = "pair",
        triplet_tau_plus: float = 16.8,
        triplet_tau_minus: float = 33.7,
        triplet_tau_x: float = 101.0,
        triplet_tau_y: float = 114.0,
        triplet_A2_plus: float = 5e-10,
        triplet_A2_minus: float = 7e-3,
        triplet_A3_plus: float = 6.2e-3,
        triplet_A3_minus: float = 2.3e-4,
    ) -> None:
        self.n_columns = int(n_columns)
        self.input_dim = int(input_dim)
        self.column_dim = int(column_dim)
        self.device = device

        self.input_stdp_ltp = float(input_stdp_ltp)
        self.input_stdp_ltd = float(input_stdp_ltd)
        self.trace_tau = float(trace_tau)
        self.eligibility_tau = float(eligibility_tau)
        self.stdp_mu_plus = float(stdp_mu_plus)
        self.stdp_mu_minus = float(stdp_mu_minus)
        self.synaptic_scaling_alpha = float(synaptic_scaling_alpha)
        self.inhibitory_plasticity_lr = float(inhibitory_plasticity_lr)
        self.inhibitory_decay = float(inhibitory_decay)
        self.target_firing_rate = float(target_firing_rate)
        self.input_row_target = float(input_row_target)
        self.projection_norm_target = float(projection_norm_target)
        self.projection_plasticity_scale = float(projection_plasticity_scale)
        self.assembly_projection_plasticity_scale = float(assembly_projection_plasticity_scale)
        self.spike_backend = str(spike_backend)
        if self.spike_backend not in {"proxy", "adex"}:
            raise ValueError("spike_backend must be 'proxy' or 'adex'")
        self.plasticity_rule = str(plasticity_rule)
        if self.plasticity_rule not in {"pair", "triplet"}:
            raise ValueError("plasticity_rule must be 'pair' or 'triplet'")

        # Triplet STDP parameters (Pfister & Gerstner 2006, all-to-all model)
        self.triplet_tau_plus = float(triplet_tau_plus)
        self.triplet_tau_minus = float(triplet_tau_minus)
        self.triplet_tau_x = float(triplet_tau_x)
        self.triplet_tau_y = float(triplet_tau_y)
        self.triplet_A2_plus = float(triplet_A2_plus)
        self.triplet_A2_minus = float(triplet_A2_minus)
        self.triplet_A3_plus = float(triplet_A3_plus)
        self.triplet_A3_minus = float(triplet_A3_minus)

        self.pre_trace = torch.zeros(self.input_dim, device=self.device)
        self.post_trace = torch.zeros(self.n_columns, device=self.device)
        self.projected_trace = torch.zeros(self.column_dim, device=self.device)
        self.assembly_trace = torch.zeros(self.n_columns, device=self.device)

        # Triplet STDP traces (Pfister & Gerstner 2006, all-to-all model):
        # r1 (pre-fast, τ+), r2 (pre-slow, τx), o1 (post-fast, τ-), o2 (post-slow, τy)
        self.r1_trace = torch.zeros(self.input_dim, device=self.device)
        self.o1_trace = torch.zeros(self.n_columns, device=self.device)
        self.o2_trace = torch.zeros(self.n_columns, device=self.device)
        self.r2_trace = torch.zeros(self.input_dim, device=self.device)

        self.input_eligibility = torch.zeros(self.n_columns, self.input_dim, device=self.device)
        self.projection_eligibility = torch.zeros(self.input_dim, self.column_dim, device=self.device)
        self.assembly_projection_eligibility = torch.zeros(self.n_columns, self.column_dim, device=self.device)

        self.firing_rate_ema = torch.full(
            (self.n_columns,),
            self.target_firing_rate,
            device=self.device,
        )
        self.synaptic_scale = torch.ones(self.n_columns, device=self.device)
        self.inhibitory_trace = torch.zeros(self.n_columns, device=self.device)
        self.inhibitory_tone = torch.zeros(self.n_columns, device=self.device)
        self.adex_neurons = (
            AdExNeuron(n_neurons=self.n_columns, dt=0.5, device=self.device, burst_mode=True)
            if self.spike_backend == "adex"
            else None
        )
        self.adex_step = 0
        self.last_post_spike_fraction = 0.0
        self.last_mean_membrane_voltage = 0.0
        self._renorm_counter = 0

        # Pre-compute constant decay factors (they never change)
        self._trace_decay_val = math.exp(-1.0 / max(self.trace_tau, 1e-6))
        self._eligibility_decay_val = math.exp(-1.0 / max(self.eligibility_tau, 1e-6))
        self._triplet_decays_val = (
            math.exp(-1.0 / max(self.triplet_tau_plus, 1e-6)),
            math.exp(-1.0 / max(self.triplet_tau_minus, 1e-6)),
            math.exp(-1.0 / max(self.triplet_tau_y, 1e-6)),
            math.exp(-1.0 / max(self.triplet_tau_x, 1e-6)),
        )

    def _trace_decay(self) -> float:
        return self._trace_decay_val

    def _eligibility_decay(self) -> float:
        return self._eligibility_decay_val

    def _triplet_decays(self) -> tuple[float, float, float, float]:
        """Decay constants for the four triplet traces: r1 (τ+), o1 (τ-), o2 (τy), r2 (τx).

        Per Pfister & Gerstner 2006 all-to-all model: r2 uses its own time
        constant τx (slow pre-synaptic trace), distinct from r1's τ+.
        """
        return self._triplet_decays_val

    def inhibition(self, candidates: torch.Tensor | None = None) -> torch.Tensor:
        if candidates is None:
            return self.inhibitory_tone
        return self.inhibitory_tone[candidates]

    def _proxy_winner_activity(
        self,
        winner_indices: torch.Tensor,
        winner_strengths: torch.Tensor | None,
    ) -> torch.Tensor:
        post_spikes = torch.zeros(self.n_columns, device=self.device)
        winners = winner_indices.long()
        if int(winners.numel()) == 0:
            return post_spikes

        if winner_strengths is None or int(winner_strengths.numel()) != int(winners.numel()):
            strengths = torch.full(
                (int(winners.numel()),),
                1.0 / max(1, int(winners.numel())),
                device=self.device,
            )
        else:
            strengths = _normalize_nonnegative(winner_strengths.to(self.device))
        post_spikes[winners] = strengths
        return post_spikes

    def _winner_activity(
        self,
        winner_indices: torch.Tensor,
        winner_strengths: torch.Tensor | None,
        assembly_signal: torch.Tensor,
        compute_metrics: bool = True,
    ) -> torch.Tensor:
        proxy_post = self._proxy_winner_activity(winner_indices, winner_strengths)
        if self.spike_backend != "adex" or self.adex_neurons is None:
            if compute_metrics:
                self.last_post_spike_fraction = float((proxy_post > 0).float().mean().item())
                self.last_mean_membrane_voltage = 0.0
            return proxy_post

        current = 8.0 * torch.clamp(assembly_signal.to(self.device), min=0.0)
        if proxy_post.any():
            current = current + 24.0 * proxy_post
        spikes = self.adex_neurons.step(current, t=float(self.adex_step) * float(self.adex_neurons.dt))
        self.adex_step += 1

        adex_post = spikes.to(torch.float32)
        if compute_metrics:
            self.last_post_spike_fraction = float(adex_post.mean().item())
            self.last_mean_membrane_voltage = float(self.adex_neurons.V.mean().item())
        if not adex_post.any():
            return proxy_post
        return _normalize_nonnegative(adex_post + 0.25 * proxy_post)

    def _log_stdp_delta(
        self,
        *,
        weights: torch.Tensor,
        pre_signal: torch.Tensor,
        post_signal: torch.Tensor,
        pre_trace: torch.Tensor,
        post_trace: torch.Tensor,
    ) -> torch.Tensor:
        # Sparse LTP: post_signal is only non-zero at winner columns
        win_idx = post_signal.nonzero(as_tuple=True)[0]
        delta = torch.zeros_like(weights)
        if win_idx.numel() > 0:
            w_win = weights[win_idx]
            ltp_win = (
                self.input_stdp_ltp
                * torch.pow(torch.clamp(w_win, min=1e-6), self.stdp_mu_plus)
                * post_signal[win_idx].unsqueeze(1)
                * pre_trace.unsqueeze(0)
            )
            delta[win_idx] = ltp_win
        # LTD: uses post_trace (dense) — sparse active-row optimization
        # Only compute for columns where post_trace has decayed above threshold
        active = post_trace > 1e-5
        if active.any():
            active_idx = active.nonzero(as_tuple=True)[0]
            w_active = weights[active_idx]
            ltd = (
                self.input_stdp_ltd
                * torch.pow(torch.clamp(w_active, min=1e-6), self.stdp_mu_minus)
                * post_trace[active_idx].unsqueeze(1)
                * pre_signal.unsqueeze(0)
            )
            delta[active_idx] -= ltd
        return delta

    def _triplet_stdp_delta(
        self,
        *,
        weights: torch.Tensor,
        pre_signal: torch.Tensor,
        post_signal: torch.Tensor,
    ) -> torch.Tensor:
        """Triplet STDP (Pfister & Gerstner 2006) combined with log-STDP.

        LTP at post-spike: A2+ * r1 + A3+ * r1 * o2(t-ε)
        LTD at pre-spike:  -(A2- + A3- * r2(t-ε)) * o1 * f_sublinear(w)

        Sparse optimization: LTP only computed for winner columns (post_signal
        is non-zero only at winners), reducing [n_col × input_dim] to [k × input_dim].
        """
        delta = torch.zeros_like(weights)

        # Sparse LTP: only winner columns have non-zero post_signal
        win_idx = post_signal.nonzero(as_tuple=True)[0]
        if win_idx.numel() > 0:
            r1 = self.r1_trace  # [input_dim]
            o2_win = self.o2_trace[win_idx]  # [k]
            w_win = weights[win_idx]  # [k, input_dim]
            w_pow = torch.pow(torch.clamp(w_win, min=1e-6), self.stdp_mu_plus)
            combined = (self.triplet_A2_plus + self.triplet_A3_plus * o2_win.unsqueeze(1)) * r1.unsqueeze(0)
            delta[win_idx] = w_pow * post_signal[win_idx].unsqueeze(1) * combined

        # LTD: uses o1_trace — sparse active-row optimization
        active = self.o1_trace > 1e-5
        if active.any():
            active_idx = active.nonzero(as_tuple=True)[0]
            f_sub = 1.0 / (1.0 + torch.clamp(weights[active_idx], min=1e-6))
            ltd_coeff = self.triplet_A2_minus + self.triplet_A3_minus * self.r2_trace.unsqueeze(0)
            delta[active_idx] -= (
                f_sub
                * self.o1_trace[active_idx].unsqueeze(1)
                * pre_signal.unsqueeze(0)
                * ltd_coeff
            )

        return delta

    def _update_triplet_traces(
        self,
        pre_signal: torch.Tensor,
        post_signal: torch.Tensor,
    ) -> None:
        """Update the four triplet traces: r1, o1, o2, r2."""
        dr1, do1, do2, dr2 = self._triplet_decays()
        # Decay then add spikes (traces are "just before" the spike)
        self.r1_trace = self.r1_trace * dr1 + pre_signal
        self.o1_trace = self.o1_trace * do1 + post_signal
        self.o2_trace = self.o2_trace * do2 + post_signal
        self.r2_trace = self.r2_trace * dr2 + pre_signal

    def _renormalize_input_rows(self, input_weights: torch.Tensor) -> torch.Tensor:
        target = self.input_row_target * self.synaptic_scale.unsqueeze(1)
        row_sums = torch.clamp(input_weights.sum(dim=1, keepdim=True), min=1e-8)
        return torch.clamp(input_weights, min=1e-6) * (target / row_sums)

    def apply(
        self,
        *,
        input_weights: torch.Tensor,
        projection_weights: torch.Tensor,
        assembly_projection_weights: torch.Tensor,
        input_pattern: torch.Tensor | None,
        pre_synaptic_trace: torch.Tensor | None,
        projected_input: torch.Tensor | None,
        assembly: torch.Tensor,
        routing_key: torch.Tensor,
        winner_indices: torch.Tensor,
        winner_strengths: torch.Tensor | None,
        modulator: float,
        lr: float,
        compute_metrics: bool = True,
    ) -> dict[str, float]:
        if input_pattern is None:
            return {
                "modulated_update_norm": 0.0,
                "mean_inhibitory_tone": float(self.inhibitory_tone.mean().item()),
                "mean_synaptic_scale": float(self.synaptic_scale.mean().item()),
            }

        # Inline normalize_nonnegative for performance (avoids 4 function calls + .to() overhead)
        _raw_pre = pre_synaptic_trace if pre_synaptic_trace is not None else input_pattern
        pre_signal = torch.clamp(_raw_pre.to(self.device).float(), min=0.0)
        pre_signal = pre_signal / torch.clamp(pre_signal.sum(), min=1e-8)

        _raw_proj = projected_input if projected_input is not None else routing_key
        projected_signal = torch.clamp(_raw_proj.to(self.device).float(), min=0.0)
        projected_signal = projected_signal / torch.clamp(projected_signal.sum(), min=1e-8)

        assembly_signal = torch.clamp(assembly.to(self.device).float(), min=0.0)
        assembly_signal = assembly_signal / torch.clamp(assembly_signal.sum(), min=1e-8)

        routing_signal = torch.clamp(routing_key.to(self.device).float(), min=0.0)
        routing_signal = routing_signal / torch.clamp(routing_signal.sum(), min=1e-8)

        post_signal = self._winner_activity(winner_indices, winner_strengths, assembly_signal, compute_metrics=compute_metrics)

        trace_decay = self._trace_decay()
        eligibility_decay = self._eligibility_decay()
        self.pre_trace = self.pre_trace * trace_decay + pre_signal
        self.post_trace = self.post_trace * trace_decay + post_signal
        self.projected_trace = self.projected_trace * trace_decay + projected_signal
        self.assembly_trace = self.assembly_trace * trace_decay + assembly_signal

        # Update triplet traces (always maintained; only used when rule == "triplet")
        self._update_triplet_traces(pre_signal, post_signal)

        if self.plasticity_rule == "triplet":
            input_delta = self._triplet_stdp_delta(
                weights=input_weights,
                pre_signal=pre_signal,
                post_signal=post_signal,
            )
        else:
            input_delta = self._log_stdp_delta(
                weights=input_weights,
                pre_signal=pre_signal,
                post_signal=post_signal,
                pre_trace=self.pre_trace,
                post_trace=self.post_trace,
            )
        self.input_eligibility = self.input_eligibility * eligibility_decay + input_delta

        projection_delta = (
            self.input_stdp_ltp
            * torch.pow(torch.clamp(projection_weights, min=1e-6), self.stdp_mu_plus)
            * self.pre_trace.unsqueeze(1)
            * self.projected_trace.unsqueeze(0)
            - self.input_stdp_ltd
            * torch.pow(torch.clamp(projection_weights, min=1e-6), self.stdp_mu_minus)
            * pre_signal.unsqueeze(1)
            * projected_signal.unsqueeze(0)
        )
        self.projection_eligibility = self.projection_eligibility * eligibility_decay + projection_delta

        assembly_projection_delta = (
            self.input_stdp_ltp
            * torch.pow(torch.clamp(assembly_projection_weights, min=1e-6), self.stdp_mu_plus)
            * self.assembly_trace.unsqueeze(1)
            * self.projected_trace.unsqueeze(0)
            - self.input_stdp_ltd
            * torch.pow(torch.clamp(assembly_projection_weights, min=1e-6), self.stdp_mu_minus)
            * assembly_signal.unsqueeze(1)
            * routing_signal.unsqueeze(0)
        )
        self.assembly_projection_eligibility = (
            self.assembly_projection_eligibility * eligibility_decay + assembly_projection_delta
        )

        learning_signal = float(max(-1.0, min(1.0, modulator)))
        input_weights.add_(lr * learning_signal * self.input_eligibility)
        projection_weights.add_(lr * learning_signal * self.projection_plasticity_scale * self.projection_eligibility)
        assembly_projection_weights.add_(
            lr * learning_signal * self.assembly_projection_plasticity_scale * self.assembly_projection_eligibility
        )

        self.firing_rate_ema = 0.99 * self.firing_rate_ema + 0.01 * post_signal
        scaling_ratio = torch.pow(
            (self.target_firing_rate + 1e-6) / (self.firing_rate_ema + 1e-6),
            self.synaptic_scaling_alpha,
        )
        self.synaptic_scale = torch.clamp(self.synaptic_scale * scaling_ratio, min=0.25, max=4.0)

        self.inhibitory_trace = self.inhibitory_trace * self.inhibitory_decay + post_signal
        rate_error = self.firing_rate_ema - self.target_firing_rate
        self.inhibitory_tone = torch.clamp(
            self.inhibitory_tone + self.inhibitory_plasticity_lr * rate_error * (1.0 + self.inhibitory_trace),
            min=0.0,
            max=1.5,
        )

        input_weights.copy_(self._renormalize_input_rows(input_weights))
        projection_weights.copy_(_normalize_columns(projection_weights, self.projection_norm_target))
        assembly_projection_weights.copy_(_normalize_columns(assembly_projection_weights, self.projection_norm_target))

        if compute_metrics:
            return {
                "modulated_update_norm": float((lr * learning_signal * self.input_eligibility).norm().item()),
                "mean_inhibitory_tone": float(self.inhibitory_tone.mean().item()),
                "mean_synaptic_scale": float(self.synaptic_scale.mean().item()),
                "post_spike_fraction": float(self.last_post_spike_fraction),
                "mean_membrane_voltage": float(self.last_mean_membrane_voltage),
            }
        return {}

    def revive_columns(self, column_indices: torch.Tensor) -> None:
        if int(column_indices.numel()) == 0:
            return
        indices = column_indices.long().to(self.device)
        self.post_trace[indices] = 0.0
        self.assembly_trace[indices] = 0.0
        self.input_eligibility[indices] = 0.0
        self.assembly_projection_eligibility[indices] = 0.0
        self.firing_rate_ema[indices] = self.target_firing_rate
        self.synaptic_scale[indices] = 1.0
        self.inhibitory_trace[indices] = 0.0
        self.inhibitory_tone[indices] = 0.0
        self.o1_trace[indices] = 0.0
        self.o2_trace[indices] = 0.0

    def state_dict(self) -> dict[str, Any]:
        return {
            "pre_trace": self.pre_trace.detach().clone().cpu(),
            "post_trace": self.post_trace.detach().clone().cpu(),
            "projected_trace": self.projected_trace.detach().clone().cpu(),
            "assembly_trace": self.assembly_trace.detach().clone().cpu(),
            "r1_trace": self.r1_trace.detach().clone().cpu(),
            "o1_trace": self.o1_trace.detach().clone().cpu(),
            "o2_trace": self.o2_trace.detach().clone().cpu(),
            "r2_trace": self.r2_trace.detach().clone().cpu(),
            "input_eligibility": self.input_eligibility.detach().clone().cpu(),
            "projection_eligibility": self.projection_eligibility.detach().clone().cpu(),
            "assembly_projection_eligibility": self.assembly_projection_eligibility.detach().clone().cpu(),
            "firing_rate_ema": self.firing_rate_ema.detach().clone().cpu(),
            "synaptic_scale": self.synaptic_scale.detach().clone().cpu(),
            "inhibitory_trace": self.inhibitory_trace.detach().clone().cpu(),
            "inhibitory_tone": self.inhibitory_tone.detach().clone().cpu(),
            "plasticity_rule": self.plasticity_rule,
        }

    def load_state_dict(self, snapshot: dict[str, Any]) -> None:
        for attr in (
            "pre_trace",
            "post_trace",
            "projected_trace",
            "assembly_trace",
            "r1_trace",
            "o1_trace",
            "o2_trace",
            "r2_trace",
            "input_eligibility",
            "projection_eligibility",
            "assembly_projection_eligibility",
            "firing_rate_ema",
            "synaptic_scale",
            "inhibitory_trace",
            "inhibitory_tone",
        ):
            value = snapshot.get(attr)
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(self.device))
