from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from marulho.core.plasticity import LocalPlasticityCircuit


def _normalize_positive_vector(values: torch.Tensor, dim: int) -> torch.Tensor:
    values = values.clamp(min=1e-6)
    norms = values.norm(dim=dim, keepdim=True).clamp(min=1e-8)
    return values / norms


def _normalize_routing_key(key: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Normalize a 1-D routing key (clamp-positive then L2-normalize)."""
    x = key.to(device).clamp(min=1e-6)
    return x / x.norm().clamp(min=1e-8)


class CompetitiveColumnLayer:
    """Competitive layer with explicit local plasticity on active synapses."""

    def __init__(
        self,
        n_columns: int,
        column_dim: int,
        input_dim: int,
        device: torch.device,
        k_routing: int = 10,
        n_winners: int = 1,
        lr_initial: float = 0.01,
        lr_decay: float = 1e-6,
        input_weight_blend: float = 0.10,
        input_synapse_ltp: float = 0.02,
        input_synapse_ltd: float = 0.01,
        input_weight_row_target: float = 1.0,
        plasticity_mode: str = "lite",
        plasticity_spike_backend: str = "proxy",
        homeostasis_beta: float = 0.01,
        homeostasis_lr: float = 0.2,
        threshold_min: float = 0.05,
        threshold_max: float = 1.5,
        dead_column_steps: int = 2000,
        dead_column_noise: float = 0.05,
        prototype_momentum: float = 0.85,
        stdp_trace_tau: float = 20.0,
        stdp_eligibility_tau: float = 200.0,
        stdp_mu_plus: float = 0.0,
        stdp_mu_minus: float = 1.0,
        synaptic_scaling_alpha: float = 0.1,
        inhibitory_plasticity_lr: float = 0.05,
        inhibitory_decay: float = 0.95,
        projection_plasticity_scale: float = 0.35,
        assembly_projection_plasticity_scale: float = 0.25,
        projection_norm_target: float = 0.1,
        plasticity_rule: str = "pair",
        triplet_tau_plus: float = 16.8,
        triplet_tau_minus: float = 33.7,
        triplet_tau_x: float = 101.0,
        triplet_tau_y: float = 114.0,
        triplet_A2_plus: float = 5e-10,
        triplet_A2_minus: float = 7e-3,
        triplet_A3_plus: float = 6.2e-3,
        triplet_A3_minus: float = 2.3e-4,
        prototype_init_mode: str = "random",
        bootstrap_prototypes: Optional[torch.Tensor] = None,
    ) -> None:
        self.n_columns = int(n_columns)
        self.column_dim = int(column_dim)
        self.input_dim = int(input_dim)
        self.k_routing = int(k_routing)
        self.n_winners = int(n_winners)
        self.device = device

        self.lr_initial = float(lr_initial)
        self.lr_decay = float(lr_decay)
        self.update_count = 0
        self.input_weight_blend = float(input_weight_blend)
        self.input_synapse_ltp = float(input_synapse_ltp)
        self.input_synapse_ltd = float(input_synapse_ltd)
        self.input_weight_row_target = float(input_weight_row_target)
        self.plasticity_mode = str(plasticity_mode)
        self.plasticity_spike_backend = str(plasticity_spike_backend)
        self.homeostasis_beta = float(homeostasis_beta)
        self.homeostasis_lr = float(homeostasis_lr)
        self.threshold_min = float(threshold_min)
        self.threshold_max = float(threshold_max)
        self.dead_column_steps = int(dead_column_steps)
        self.dead_column_noise = float(dead_column_noise)
        self.prototype_momentum = float(prototype_momentum)
        self.projection_norm_target = float(projection_norm_target)

        if self.plasticity_mode not in {"lite", "local_stdp"}:
            raise ValueError("plasticity_mode must be 'lite' or 'local_stdp'")

        self.W_project = torch.empty(self.input_dim, self.column_dim, device=self.device)
        self.W_project.log_normal_(mean=-2.3, std=0.5)
        self.W_project = F.normalize(self.W_project, dim=0) * self.projection_norm_target

        self.input_weights = torch.empty(self.n_columns, self.input_dim, device=self.device)
        self.input_weights.log_normal_(mean=-2.3, std=0.5)
        self.input_weights = torch.clamp(self.input_weights, min=1e-6)
        self.input_weights = self.input_weights * (
            self.input_weight_row_target / (self.input_weights.sum(dim=1, keepdim=True) + 1e-8)
        )

        self.prototype_init_mode = str(prototype_init_mode)
        if self.prototype_init_mode == "teacher" and bootstrap_prototypes is not None:
            bp = bootstrap_prototypes.to(self.device).float()
            if bp.shape != (self.n_columns, self.column_dim):
                raise ValueError(
                    f"bootstrap_prototypes shape {tuple(bp.shape)} != "
                    f"({self.n_columns}, {self.column_dim})"
                )
            self.prototypes = bp.clamp(min=1e-6)
            self.prototypes = _normalize_positive_vector(self.prototypes, dim=1)
        else:
            self.prototypes = torch.rand(self.n_columns, self.column_dim, device=self.device)
            self.prototypes = _normalize_positive_vector(self.prototypes, dim=1)
        self.prototype_velocity = torch.zeros(self.n_columns, self.column_dim, device=self.device)
        self.last_input_pattern: Optional[torch.Tensor] = None
        self.last_projected_input: Optional[torch.Tensor] = None
        self._last_norm_key_id: int = -1
        self._last_norm_key_val: Optional[torch.Tensor] = None

        # Step-level caches: populated by assembly_from_input, consumed by compete/process
        self._cached_proto_sim: Optional[torch.Tensor] = None
        self._cached_raw_drive: Optional[torch.Tensor] = None
        self.last_execution_mode = "not_run"
        self.last_scored_column_count = 0
        self.last_candidate_count = 0
        self.last_homeostasis_update_count = 0
        self.last_homeostasis_update_mode = "not_run"
        self.last_homeostasis_materialize_count = 0
        self.last_homeostasis_materialize_max_age = 0
        self.last_homeostasis_materialize_mode = "not_run"
        self.last_state_transition_mode = "not_run"
        self.last_state_transition_column_count = 0
        self.last_input_plasticity_mode = "not_run"
        self.input_plasticity_update_count = 0
        self.input_plasticity_skip_count = 0

        self.thresholds = torch.full((self.n_columns,), 0.5, device=self.device)
        self.target_firing_rate = 1.0 / max(1, self.n_columns)
        self.win_rate_ema = torch.full(
            (self.n_columns,),
            self.target_firing_rate,
            device=self.device,
        )
        self.homeostasis_step_count = 0
        self.homeostasis_last_update_step = torch.zeros(
            self.n_columns,
            device=self.device,
            dtype=torch.long,
        )
        self.state_transition_step_count = 0
        self.state_transition_all_materialized_step = 0
        self.state_transition_last_update_tensor_materialized_step = 0
        self.steps_since_win_last_update_step = torch.zeros(
            self.n_columns,
            device=self.device,
            dtype=torch.long,
        )
        self.last_state_transition_cached_count = 0
        self.last_state_transition_materialize_count = 0
        self.last_state_transition_materialize_max_age = 0
        self.last_state_transition_materialize_mode = "not_run"
        self.threshold_relaxation_history: list[bool] = []
        self.threshold_relaxation_last_applied_step = torch.zeros(
            self.n_columns,
            device=self.device,
            dtype=torch.long,
        )
        self.steps_since_win = torch.zeros(self.n_columns, device=self.device, dtype=torch.long)
        self.last_revived_indices = torch.empty(0, device=self.device, dtype=torch.long)
        self.spike_history_window = 32
        self.recent_spike_window = torch.zeros(
            self.spike_history_window,
            self.n_columns,
            device=self.device,
            dtype=torch.float32,
        )
        self.recent_spike_window_active_ids = torch.full(
            (self.spike_history_window, max(1, self.n_winners)),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        self.recent_spike_window_cursor = 0
        self.recent_spike_window_count = 0

        self.local_plasticity: LocalPlasticityCircuit | None = None
        if self.plasticity_mode == "local_stdp":
            self.local_plasticity = LocalPlasticityCircuit(
                n_columns=self.n_columns,
                input_dim=self.input_dim,
                column_dim=self.column_dim,
                device=self.device,
                input_stdp_ltp=self.input_synapse_ltp,
                input_stdp_ltd=self.input_synapse_ltd,
                trace_tau=stdp_trace_tau,
                eligibility_tau=stdp_eligibility_tau,
                stdp_mu_plus=stdp_mu_plus,
                stdp_mu_minus=stdp_mu_minus,
                synaptic_scaling_alpha=synaptic_scaling_alpha,
                inhibitory_plasticity_lr=inhibitory_plasticity_lr,
                inhibitory_decay=inhibitory_decay,
                target_firing_rate=self.target_firing_rate,
                input_row_target=self.input_weight_row_target,
                projection_norm_target=self.projection_norm_target,
                projection_plasticity_scale=projection_plasticity_scale,
                assembly_projection_plasticity_scale=assembly_projection_plasticity_scale,
                spike_backend=self.plasticity_spike_backend,
                plasticity_rule=plasticity_rule,
                triplet_tau_plus=triplet_tau_plus,
                triplet_tau_minus=triplet_tau_minus,
                triplet_tau_x=triplet_tau_x,
                triplet_tau_y=triplet_tau_y,
                triplet_A2_plus=triplet_A2_plus,
                triplet_A2_minus=triplet_A2_minus,
                triplet_A3_plus=triplet_A3_plus,
                triplet_A3_minus=triplet_A3_minus,
            )

    def get_lr(self) -> float:
        return self.lr_initial / (1.0 + self.lr_decay * self.update_count)

    def _moment_stats(self, values: torch.Tensor) -> dict[str, float]:
        flat = values.detach().float().reshape(-1)
        if flat.numel() == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "skewness": 0.0,
                "excess_kurtosis": 0.0,
            }

        mean = flat.mean()
        centered = flat - mean
        variance = centered.pow(2).mean().item()
        std = math.sqrt(max(variance, 1e-12))
        skewness = float(centered.pow(3).mean().item() / (std ** 3))
        excess_kurtosis = float(centered.pow(4).mean().item() / (std ** 4) - 3.0)
        return {
            "mean": float(mean.item()),
            "std": float(std),
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
            "skewness": skewness,
            "excess_kurtosis": excess_kurtosis,
        }

    def validate_synaptic_health(self) -> dict[str, Any]:
        """Validate synaptic weight distributions against paper targets.

        Checks:
        - Log-space normality (kurtosis in [-1, 6], skewness in [-2, 2])
        - Row-sum stability (all rows within 20% of target)
        - Synaptic scale bounds (all scales in [0.25, 4.0])

        Returns a structured payload with pass/fail and per-check details.
        """
        log_stats = self._moment_stats(torch.log(self.input_weights.clamp_min(1e-8)))
        row_sums = self.input_weights.sum(dim=1)
        target = float(self.input_weight_row_target)
        row_ratio = (row_sums / max(target, 1e-8)).detach().float()

        checks: dict[str, dict[str, Any]] = {}

        # Log-space shape: paper targets excess kurtosis 0–6 for log-normal
        log_kurt = log_stats["excess_kurtosis"]
        log_skew = log_stats["skewness"]
        checks["log_space_shape"] = {
            "pass": -1.0 <= log_kurt <= 6.0 and -2.0 <= log_skew <= 2.0,
            "excess_kurtosis": log_kurt,
            "skewness": log_skew,
            "kurtosis_range": [-1.0, 6.0],
            "skewness_range": [-2.0, 2.0],
        }

        # Row-sum stability: each row within 20% of target after renormalization
        row_min = float(row_ratio.min().item())
        row_max = float(row_ratio.max().item())
        checks["row_sum_stability"] = {
            "pass": row_min >= 0.8 and row_max <= 1.2,
            "row_ratio_min": row_min,
            "row_ratio_max": row_max,
            "tolerance": 0.2,
            "target": target,
        }

        # Synaptic scale bounds (only if plasticity active)
        if self.local_plasticity is not None:
            scale = self.local_plasticity.synaptic_scale
            s_min = float(scale.min().item())
            s_max = float(scale.max().item())
            checks["synaptic_scale_bounds"] = {
                "pass": s_min >= 0.24 and s_max <= 4.01,
                "min": s_min,
                "max": s_max,
            }

        all_pass = all(c["pass"] for c in checks.values())
        return {
            "validates": all_pass,
            "n_checks": len(checks),
            "n_columns": self.n_columns,
            "sample_size": int(self.input_weights.numel()),
            "checks": checks,
        }

    def distribution_proxy_stats(self) -> dict[str, Any]:
        """Return active parameter stats for the maintained scaffold."""
        prototype_components = self._moment_stats(self.prototypes)
        projection_components = self._moment_stats(self.W_project)
        input_weight_components = self._moment_stats(self.input_weights)
        log_input_weight_components = self._moment_stats(torch.log(self.input_weights.clamp_min(1e-8)))
        prototype_norms = self._moment_stats(torch.norm(self.prototypes.detach().float(), dim=1))
        velocity_norms = self._moment_stats(torch.norm(self.prototype_velocity.detach().float(), dim=1))

        validation = self.validate_synaptic_health()
        uses_adex = self.local_plasticity is not None and self.local_plasticity.spike_backend == "adex"

        payload = {
            "status": "active_parameter_weights",
            "plasticity_mode": self.plasticity_mode,
            "plasticity_spike_backend": self.plasticity_spike_backend if self.local_plasticity is not None else None,
            "supports_full_synaptic_weight_validation": True,
            "validates_full_log_stdp_weight_target": validation["validates"],
            "paper_target_directly_measured": True,
            "synaptic_validation": validation,
            "reason": (
                "MarulhoModel exposes active local synapses with log-STDP-style updates, "
                "iSTDP-style inhibitory balancing, synaptic scaling, and validated log-normal "
                "weight targets over the maintained competitive scaffold."
                + (" AdEx-backed postsynaptic spikes provide biologically faithful STDP timing."
                   if uses_adex else "")
            ),
            "column_input_weights": input_weight_components,
            "column_input_weights_log_space": log_input_weight_components,
            "prototype_components": prototype_components,
            "projection_components": projection_components,
            "prototype_row_norms": prototype_norms,
            "prototype_velocity_norms": velocity_norms,
        }
        if self.local_plasticity is not None:
            payload["inhibitory_tone"] = self._moment_stats(self.local_plasticity.inhibitory_tone)
            payload["synaptic_scale"] = self._moment_stats(self.local_plasticity.synaptic_scale)
            payload["uses_adex_post_spikes"] = bool(uses_adex)
        return payload

    def _cached_normalize_key(self, routing_key: torch.Tensor) -> torch.Tensor:
        """Normalize routing key with single-entry cache (same key reused within a step)."""
        key_id = id(routing_key)
        if key_id == self._last_norm_key_id and self._last_norm_key_val is not None:
            return self._last_norm_key_val
        x = _normalize_routing_key(routing_key, self.device)
        self._last_norm_key_id = key_id
        self._last_norm_key_val = x
        return x

    def _normalized_input_pattern(self, input_vec: torch.Tensor) -> torch.Tensor:
        pattern = input_vec.to(self.device).float().clamp(min=0.0)
        total = torch.clamp(pattern.sum(), min=1e-8)
        return pattern / total

    def _normalized_eligibility_trace(self, eligibility_trace: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if eligibility_trace is None:
            return None

        trace = eligibility_trace.to(self.device).float().clamp(min=0.0)
        if trace.dim() != 1 or int(trace.numel()) != self.input_dim:
            raise ValueError("eligibility_trace must be a 1D tensor with input_dim entries")

        total = float(trace.sum().item())
        if total <= 0.0:
            return None
        return trace / (trace.sum() + 1e-8)

    def project_input(self, input_vec: torch.Tensor) -> torch.Tensor:
        if input_vec.dim() != 1:
            raise ValueError("input_vec must be 1D")
        latent = torch.mv(self.W_project.t(), input_vec.to(self.device).clamp(min=0.0))
        latent = _normalize_routing_key(latent, self.device)
        self.last_projected_input = latent
        return latent

    def _input_drive(
        self,
        input_pattern: Optional[torch.Tensor],
        candidates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_pattern is None:
            n = self.n_columns if candidates is None else int(candidates.numel())
            return torch.zeros(n, device=self.device)

        if candidates is None:
            # Full computation — cache raw values for reuse by compete()
            drive = torch.mv(self.input_weights, input_pattern)
            self._cached_raw_drive = drive
            scale = torch.clamp(drive.max(), min=1e-8)
            return drive / scale

        # Candidate subset — reuse cached raw values if available
        if self._cached_raw_drive is not None:
            raw = self._cached_raw_drive[candidates]
        else:
            raw = torch.mv(self.input_weights[candidates], input_pattern)
        scale = torch.clamp(raw.max(), min=1e-8)
        return raw / scale

    def _inhibition(self, candidates: Optional[torch.Tensor] = None) -> torch.Tensor:
        thresholds = self.thresholds if candidates is None else self.thresholds[candidates]
        if self.local_plasticity is None:
            return thresholds
        return thresholds + self.local_plasticity.inhibition(candidates)

    def _canonical_column_indices(
        self,
        candidate_indices: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if candidate_indices is None:
            return None
        indices = candidate_indices.to(self.device, dtype=torch.long).flatten()
        if int(indices.numel()) > 0:
            indices = indices[(indices >= 0) & (indices < self.n_columns)]
            indices = torch.unique(indices)
        return indices

    def _state_transition_steps_snapshot(self) -> torch.Tensor:
        if (
            int(self.state_transition_all_materialized_step)
            >= int(self.state_transition_step_count)
        ):
            return self.steps_since_win
        if self.steps_since_win_last_update_step.device != self.device:
            self.steps_since_win_last_update_step = (
                self.steps_since_win_last_update_step.to(self.device)
            )
        self._ensure_state_transition_last_update_tensor_current()
        elapsed = (
            int(self.state_transition_step_count)
            - self.steps_since_win_last_update_step
        ).clamp(min=0)
        return self.steps_since_win + elapsed

    def _mark_all_state_transition_materialized(
        self,
        step: int,
        *,
        sync_last_update_tensor: bool,
    ) -> None:
        materialized_step = int(step)
        self.state_transition_all_materialized_step = materialized_step
        if sync_last_update_tensor:
            self.steps_since_win_last_update_step.fill_(materialized_step)
            self.state_transition_last_update_tensor_materialized_step = materialized_step

    def _ensure_state_transition_last_update_tensor_current(self) -> None:
        materialized_step = int(self.state_transition_all_materialized_step)
        if (
            materialized_step > 0
            and int(self.state_transition_last_update_tensor_materialized_step)
            < materialized_step
        ):
            self.steps_since_win_last_update_step.fill_(materialized_step)
            self.state_transition_last_update_tensor_materialized_step = materialized_step

    def _materialize_state_transition_indices(
        self,
        indices: torch.Tensor,
        *,
        mode: str,
        record_noop: bool,
    ) -> None:
        self._ensure_state_transition_last_update_tensor_current()
        if int(indices.numel()) <= 0:
            if record_noop:
                self.last_state_transition_materialize_mode = "empty"
                self.last_state_transition_materialize_count = 0
                self.last_state_transition_materialize_max_age = 0
            return

        last_steps = self.steps_since_win_last_update_step.index_select(0, indices)
        elapsed = (int(self.state_transition_step_count) - last_steps).clamp(min=0)
        pending_mask = elapsed > 0
        if not bool(pending_mask.any().item()):
            if record_noop:
                self.last_state_transition_materialize_mode = f"{mode}_noop"
                self.last_state_transition_materialize_count = 0
                self.last_state_transition_materialize_max_age = 0
            return

        pending_indices = indices[pending_mask]
        pending_elapsed = elapsed[pending_mask].to(
            device=self.device,
            dtype=self.steps_since_win.dtype,
        )
        self.steps_since_win[pending_indices] += pending_elapsed
        self.steps_since_win_last_update_step[pending_indices] = int(
            self.state_transition_step_count
        )
        if mode == "all_columns" and int(indices.numel()) >= int(self.n_columns):
            self.state_transition_all_materialized_step = int(
                self.state_transition_step_count
            )
            self.state_transition_last_update_tensor_materialized_step = int(
                self.state_transition_step_count
            )
        if record_noop:
            self.last_state_transition_materialize_mode = mode
            self.last_state_transition_materialize_count = int(pending_indices.numel())
            self.last_state_transition_materialize_max_age = int(
                pending_elapsed.max().item()
            )

    def _process_state_transition_indices(
        self,
        homeostasis_indices: torch.Tensor,
        winners: torch.Tensor,
    ) -> torch.Tensor:
        winner_ids = winners.to(self.device, dtype=torch.long).flatten()
        winner_ids = winner_ids[
            (winner_ids >= 0) & (winner_ids < int(self.n_columns))
        ]
        if int(winner_ids.numel()) <= 0:
            return homeostasis_indices
        if int(homeostasis_indices.numel()) <= 0:
            return winner_ids
        missing_mask = ~(
            winner_ids[:, None] == homeostasis_indices[None, :]
        ).any(dim=1)
        if not bool(missing_mask.any().item()):
            return homeostasis_indices
        return torch.cat((homeostasis_indices, winner_ids[missing_mask]))

    def state_transition_steps_snapshot(self) -> torch.Tensor:
        """Return logical ``steps_since_win`` without materializing all columns."""

        return self._state_transition_steps_snapshot().detach().clone()

    def materialize_state_transition(
        self,
        candidate_indices: Optional[torch.Tensor],
        *,
        record_noop: bool = True,
    ) -> None:
        """Advance cached stale-age counters for a bounded candidate set."""

        if self.steps_since_win_last_update_step.device != self.device:
            self.steps_since_win_last_update_step = (
                self.steps_since_win_last_update_step.to(self.device)
            )
        indices = self._canonical_column_indices(candidate_indices)
        if indices is None:
            indices = torch.arange(self.n_columns, device=self.device)
            mode = "all_columns"
        else:
            mode = (
                "candidate_subset"
                if int(indices.numel()) < self.n_columns
                else "all_columns"
            )
        self._materialize_state_transition_indices(
            indices,
            mode=mode,
            record_noop=record_noop,
        )

    def _record_recent_spike_sample(
        self,
        winners: torch.Tensor,
        *,
        sparse: bool,
    ) -> None:
        row = int(self.recent_spike_window_cursor)
        winner_ids = winners.to(self.device, dtype=torch.long).flatten()
        winner_ids = winner_ids[
            (winner_ids >= 0) & (winner_ids < int(self.n_columns))
        ]
        if int(winner_ids.numel()) <= 0:
            return
        if (
            not sparse
            or int(winner_ids.numel())
            > int(self.recent_spike_window_active_ids.shape[1])
        ):
            spike_sample = torch.zeros(self.n_columns, device=self.device)
            spike_sample[winner_ids] = 1.0
            self.recent_spike_window[row] = spike_sample
            self.recent_spike_window_active_ids[row].fill_(-1)
            remembered = winner_ids[: self.recent_spike_window_active_ids.shape[1]]
            self.recent_spike_window_active_ids[
                row,
                : int(remembered.numel()),
            ] = remembered
        else:
            previous = self.recent_spike_window_active_ids[row]
            previous = previous[(previous >= 0) & (previous < int(self.n_columns))]
            if int(previous.numel()) > 0:
                self.recent_spike_window[row, previous] = 0.0
            self.recent_spike_window[row, winner_ids] = 1.0
            self.recent_spike_window_active_ids[row].fill_(-1)
            self.recent_spike_window_active_ids[row, : int(winner_ids.numel())] = (
                winner_ids[: self.recent_spike_window_active_ids.shape[1]]
            )
        self.recent_spike_window_cursor = (
            self.recent_spike_window_cursor + 1
        ) % self.spike_history_window
        self.recent_spike_window_count = min(
            self.spike_history_window,
            self.recent_spike_window_count + 1,
        )

    def materialize_homeostasis(
        self,
        candidate_indices: Optional[torch.Tensor],
        *,
        record_noop: bool = True,
    ) -> None:
        """Apply deferred idle homeostasis to the requested columns.

        Candidate-scoped homeostasis keeps non-awake columns cached. When a
        cached column wakes as a routed candidate, this method advances only
        that bounded set through the zero-activity homeostasis updates it would
        have received on the all-column path.
        """

        if self.threshold_relaxation_last_applied_step.device != self.device:
            self.threshold_relaxation_last_applied_step = (
                self.threshold_relaxation_last_applied_step.to(self.device)
            )
        if candidate_indices is None:
            indices = torch.arange(self.n_columns, device=self.device)
            mode = "all_columns"
        else:
            indices = candidate_indices.to(self.device, dtype=torch.long).flatten()
            if int(indices.numel()) > 0:
                indices = indices[(indices >= 0) & (indices < self.n_columns)]
                indices = torch.unique(indices)
            mode = (
                "candidate_subset"
                if int(indices.numel()) < self.n_columns
                else "all_columns"
            )

        if int(indices.numel()) <= 0:
            if record_noop:
                self.last_homeostasis_materialize_count = 0
                self.last_homeostasis_materialize_max_age = 0
                self.last_homeostasis_materialize_mode = "empty"
            return

        current_step = int(self.homeostasis_step_count)
        last_steps = self.homeostasis_last_update_step.index_select(0, indices)
        elapsed = (current_step - last_steps).clamp(min=0)
        pending_mask = elapsed > 0
        if not bool(pending_mask.any()):
            if record_noop:
                self.last_homeostasis_materialize_count = 0
                self.last_homeostasis_materialize_max_age = 0
                self.last_homeostasis_materialize_mode = mode
            return

        pending_indices = indices[pending_mask]
        remaining = elapsed[pending_mask].detach().clone()
        pending_last_steps = last_steps[pending_mask].detach().clone()
        relaxation_last = self.threshold_relaxation_last_applied_step.index_select(
            0,
            pending_indices,
        )
        win_rate = self.win_rate_ema.index_select(0, pending_indices)
        thresholds = self.thresholds.index_select(0, pending_indices)
        decay = 1.0 - float(self.homeostasis_beta)
        max_age = int(remaining.max().item())

        has_relaxation_event = False
        if max_age > 0 and any(self.threshold_relaxation_history):
            start_steps = pending_last_steps.detach().cpu().tolist()
            end_steps = (pending_last_steps + remaining).detach().cpu().tolist()
            history_len = len(self.threshold_relaxation_history)
            for start, end in zip(start_steps, end_steps):
                start_i = max(0, int(start))
                end_i = min(history_len, int(end))
                if start_i < end_i and any(
                    self.threshold_relaxation_history[start_i:end_i]
                ):
                    has_relaxation_event = True
                    break

        if max_age > 0 and not has_relaxation_event:
            ages = remaining.to(device=self.device, dtype=win_rate.dtype)
            if abs(decay - 1.0) <= 1e-12:
                decayed_win_rate = win_rate
                win_rate_sum = win_rate * ages
            else:
                decay_tensor = torch.as_tensor(
                    decay,
                    dtype=win_rate.dtype,
                    device=self.device,
                )
                decay_pow = torch.pow(decay_tensor, ages)
                decayed_win_rate = win_rate * decay_pow
                win_rate_sum = (
                    win_rate
                    * decay_tensor
                    * (1.0 - decay_pow)
                    / (1.0 - decay_tensor)
                )
            unclamped_thresholds = thresholds + self.homeostasis_lr * (
                win_rate_sum - ages * float(self.target_firing_rate)
            )
            clamp_safe = (
                (unclamped_thresholds >= float(self.threshold_min))
                & (unclamped_thresholds <= float(self.threshold_max))
            )
            if bool(clamp_safe.all().item()):
                self.win_rate_ema[pending_indices] = decayed_win_rate
                self.thresholds[pending_indices] = unclamped_thresholds
                self.threshold_relaxation_last_applied_step[pending_indices] = (
                    relaxation_last
                )
                self.homeostasis_last_update_step[pending_indices] = current_step
                self.last_homeostasis_materialize_count = int(pending_indices.numel())
                self.last_homeostasis_materialize_max_age = max_age
                self.last_homeostasis_materialize_mode = mode
                return

        for offset in range(max_age):
            active = remaining > 0
            if not bool(active.any()):
                break
            active_steps = pending_last_steps[active] + int(offset)
            relax_flags = torch.tensor(
                [
                    bool(self.threshold_relaxation_history[int(step.item())])
                    if int(step.item()) < len(self.threshold_relaxation_history)
                    else False
                    for step in active_steps
                ],
                dtype=torch.bool,
                device=self.device,
            )
            active_relaxation_last = relaxation_last[active]
            relax_due = relax_flags & (active_relaxation_last <= active_steps)
            if bool(relax_due.any().item()):
                active_thresholds = thresholds[active]
                active_thresholds[relax_due] = torch.clamp(
                    active_thresholds[relax_due] * 0.995,
                    min=self.threshold_min,
                    max=self.threshold_max,
                )
                thresholds[active] = active_thresholds
                active_relaxation_last[relax_due] = active_steps[relax_due] + 1
                relaxation_last[active] = active_relaxation_last
            active_win = win_rate[active] * decay
            active_thresholds = torch.clamp(
                thresholds[active]
                + self.homeostasis_lr
                * (active_win - self.target_firing_rate),
                min=self.threshold_min,
                max=self.threshold_max,
            )
            win_rate[active] = active_win
            thresholds[active] = active_thresholds
            remaining[active] -= 1

        self.win_rate_ema[pending_indices] = win_rate
        self.thresholds[pending_indices] = thresholds
        self.threshold_relaxation_last_applied_step[pending_indices] = relaxation_last
        self.homeostasis_last_update_step[pending_indices] = current_step
        self.last_homeostasis_materialize_count = int(pending_indices.numel())
        self.last_homeostasis_materialize_max_age = max_age
        self.last_homeostasis_materialize_mode = mode

    def _record_threshold_relaxation(self) -> int:
        step = int(self.homeostasis_step_count)
        while len(self.threshold_relaxation_history) <= step:
            self.threshold_relaxation_history.append(False)
        self.threshold_relaxation_history[step] = True
        return step

    def _apply_threshold_relaxation(
        self,
        indices: torch.Tensor,
        *,
        step: int,
    ) -> None:
        if int(indices.numel()) <= 0:
            return
        selected = indices.to(self.device, dtype=torch.long).flatten()
        selected = selected[(selected >= 0) & (selected < int(self.n_columns))]
        if int(selected.numel()) <= 0:
            return
        selected = torch.unique(selected)
        if self.threshold_relaxation_last_applied_step.device != self.device:
            self.threshold_relaxation_last_applied_step = (
                self.threshold_relaxation_last_applied_step.to(self.device)
            )
        due_mask = self.threshold_relaxation_last_applied_step[selected] <= int(step)
        if not bool(due_mask.any().item()):
            return
        due = selected[due_mask]
        self.thresholds[due] = torch.clamp(
            self.thresholds[due] * 0.995,
            min=self.threshold_min,
            max=self.threshold_max,
        )
        self.threshold_relaxation_last_applied_step[due] = int(step) + 1

    def _combine_similarity_and_input_drive(
        self,
        similarity: torch.Tensor,
        *,
        candidates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.input_weight_blend == 0.0:
            self._cached_raw_drive = None
            return similarity
        drive = self._input_drive(self.last_input_pattern, candidates)
        if self.input_weight_blend == 1.0:
            return drive
        return (
            (1.0 - self.input_weight_blend) * similarity
            + self.input_weight_blend * drive
        )

    def assembly_from_input(self, input_vec: torch.Tensor) -> torch.Tensor:
        """Compute dense column activations from input before winner selection."""
        self.last_input_pattern = self._normalized_input_pattern(input_vec)
        x = self.project_input(input_vec)
        sim = torch.mv(self.prototypes, x)
        self._cached_proto_sim = sim  # reused by compete() and winner_assembly()
        self._cached_raw_drive = None
        self.last_execution_mode = "dense_assembly"
        self.last_scored_column_count = self.n_columns
        self.last_candidate_count = self.n_columns
        combined = self._combine_similarity_and_input_drive(sim)
        assembly = torch.relu(combined - self._inhibition())
        total = float(assembly.sum().item())
        if total <= 0.0:
            best = int(torch.argmax(combined).item())
            assembly = torch.zeros_like(sim)
            assembly[best] = 1.0
            return assembly
        return assembly / (assembly.sum() + 1e-8)

    def prepare_input_for_candidate_routing(self, input_vec: torch.Tensor) -> torch.Tensor:
        """Prepare one input without scoring columns before retrieval narrows candidates."""
        self.last_input_pattern = self._normalized_input_pattern(input_vec)
        projected = self.project_input(input_vec)
        self._cached_proto_sim = None
        self._cached_raw_drive = None
        self.last_execution_mode = "candidate_routing_pending"
        self.last_scored_column_count = 0
        self.last_candidate_count = 0
        return projected

    def winner_assembly(
        self,
        routing_key: torch.Tensor,
        winner_indices: torch.Tensor,
        *,
        _pre_normalized: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        winners = winner_indices.long()
        assembly = torch.zeros(self.n_columns, device=self.device)

        # Reuse cached prototype similarities when available
        if self._cached_proto_sim is not None:
            assembly[winners] = self._cached_proto_sim[winners]
        else:
            x = _pre_normalized if _pre_normalized is not None else self._cached_normalize_key(routing_key)
            winner_proto = self.prototypes[winners]
            assembly[winners] = torch.cosine_similarity(x.unsqueeze(0), winner_proto, dim=1)
        return assembly

    def compete(
        self,
        routing_key: torch.Tensor,
        candidate_indices: Optional[torch.Tensor] = None,
        fallback_allowed: bool = False,
        context_gain: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._cached_normalize_key(routing_key)

        if candidate_indices is not None and len(candidate_indices) > 0:
            candidates = candidate_indices.to(self.device).long()
        elif fallback_allowed:
            candidates = torch.arange(self.n_columns, device=self.device)
        else:
            raise RuntimeError(
                "No candidates available and fallback disabled; initialize routing index first."
            )

        candidate_count = int(candidates.numel())
        self.last_candidate_count = candidate_count
        self.last_scored_column_count = (
            self.n_columns if self._cached_proto_sim is not None else candidate_count
        )
        self.last_execution_mode = (
            "candidate_subset"
            if self._cached_proto_sim is None and candidate_count < self.n_columns
            else "dense_cached"
            if self._cached_proto_sim is not None
            else "all_columns_candidate_set"
        )

        # Reuse cached prototype similarities when available
        if self._cached_proto_sim is not None:
            sim = self._cached_proto_sim[candidates]
        else:
            proto = self.prototypes[candidates]
            sim = torch.mv(proto, x)
        combined = self._combine_similarity_and_input_drive(
            sim,
            candidates=candidates,
        )
        if context_gain is not None:
            gain = context_gain.to(self.device)
            if gain.dim() != 1 or int(gain.numel()) != self.n_columns:
                raise ValueError("context_gain must be a 1D tensor with n_columns entries")
            combined = combined * torch.clamp(gain[candidates], min=0.5, max=1.5)
        self.materialize_homeostasis(candidates)
        act = torch.relu(combined - self._inhibition(candidates))

        topk = min(self.n_winners, act.numel())
        values, local_idx = torch.topk(act, k=topk)

        if values.max() <= 0:
            relaxation_step = self._record_threshold_relaxation()
            self._apply_threshold_relaxation(candidates, step=relaxation_step)
            best_idx = torch.argmax(combined)
            winner_idx = candidates[best_idx : best_idx + 1]
            strengths = torch.ones(1, device=self.device)
            return winner_idx, strengths, candidates

        winner_idx = candidates[local_idx]
        strengths = values / (values.sum() + 1e-8)
        return winner_idx, strengths, candidates

    def process(
        self,
        routing_key: torch.Tensor,
        winner_indices: torch.Tensor,
        modulator: float,
        *,
        winner_strengths: Optional[torch.Tensor] = None,
        eligibility_trace: Optional[torch.Tensor] = None,
        assembly_projection: Optional[torch.Tensor] = None,
        prototype_lr_scale: float = 1.0,
        input_lr_scale: float = 1.0,
        update_global_state: bool = True,
        homeostasis_update_indices: Optional[torch.Tensor] = None,
        compute_metrics: bool = True,
    ) -> torch.Tensor:
        x = self._cached_normalize_key(routing_key)
        winners = winner_indices.long()

        assembly = self.winner_assembly(routing_key, winners, _pre_normalized=x)
        winner_proto = self.prototypes[winners]
        delta = x.unsqueeze(0) - winner_proto
        velocity = self.prototype_velocity[winners]

        lr = self.get_lr()
        if self.plasticity_mode == "local_stdp":
            strength_scale = 1.0
            if winner_strengths is not None and int(winner_strengths.numel()) == int(winners.numel()):
                strength_scale = float(
                    torch.clamp(
                        winner_strengths.to(self.device).float().mean(),
                        min=0.0,
                        max=1.0,
                    ).item()
                )
            learning_signal = max(-1.0, min(1.0, float(modulator)))
            prototype_lr = lr * max(0.0, float(prototype_lr_scale)) * abs(learning_signal) * max(0.2, strength_scale)
            direction = 0.0 if abs(learning_signal) <= 1e-8 else (1.0 if learning_signal >= 0.0 else -1.0)
            velocity = self.prototype_momentum * velocity + direction * prototype_lr * delta
        else:
            prototype_lr = lr * max(0.0, min(1.0, float(abs(modulator)) + 0.1)) * max(0.0, float(prototype_lr_scale))
            velocity = self.prototype_momentum * velocity + prototype_lr * delta

        self.prototype_velocity[winners] = velocity
        if prototype_lr > 0.0:
            updated_proto = torch.clamp(winner_proto + velocity, min=1e-6)
            self.prototypes[winners] = _normalize_positive_vector(updated_proto, dim=1)
        else:
            self.prototypes[winners] = winner_proto

        if self.last_input_pattern is None:
            self.last_input_plasticity_mode = "skipped_no_input"
            self.input_plasticity_skip_count += 1
        elif (
            self.plasticity_mode == "local_stdp"
            and self.local_plasticity is not None
            and assembly_projection is not None
        ):
            self.last_input_plasticity_mode = "local_stdp"
            self.input_plasticity_update_count += 1
            local_trace = self._normalized_eligibility_trace(eligibility_trace)
            self.local_plasticity.apply(
                input_weights=self.input_weights,
                projection_weights=self.W_project,
                assembly_projection_weights=assembly_projection,
                input_pattern=self.last_input_pattern,
                pre_synaptic_trace=local_trace,
                projected_input=self.last_projected_input,
                assembly=assembly,
                routing_key=x,
                winner_indices=winners,
                winner_strengths=winner_strengths,
                modulator=modulator,
                lr=lr * max(0.0, float(input_lr_scale)),
                compute_metrics=compute_metrics,
            )
        elif self.plasticity_mode == "lite" and self.input_weight_blend == 0.0:
            self.last_input_plasticity_mode = "skipped_zero_blend"
            self.input_plasticity_skip_count += 1
        else:
            self.last_input_plasticity_mode = (
                "local_stdp_fallback_lite"
                if self.plasticity_mode == "local_stdp"
                else "lite_active"
            )
            self.input_plasticity_update_count += 1
            winner_weights = self.input_weights[winners]
            input_pattern = self.last_input_pattern.unsqueeze(0)
            input_lr = lr * max(0.0, float(input_lr_scale))
            ltp = self.input_synapse_ltp * input_lr * input_pattern
            ltd = self.input_synapse_ltd * input_lr * (1.0 - input_pattern) * winner_weights
            winner_weights = torch.clamp(winner_weights + ltp - ltd, min=1e-6)
            winner_weights = winner_weights * (
                self.input_weight_row_target / (winner_weights.sum(dim=1, keepdim=True) + 1e-8)
            )
            self.input_weights[winners] = winner_weights

        self.last_revived_indices = torch.empty(0, device=self.device, dtype=torch.long)
        if update_global_state:
            if homeostasis_update_indices is None:
                homeostasis_indices = torch.arange(self.n_columns, device=self.device)
                self.last_homeostasis_update_mode = "all_columns"
            else:
                # Trainer passes the competition candidate set, which already
                # contains the winners. Avoid unique/cat overhead in the hot path.
                homeostasis_indices = homeostasis_update_indices.to(self.device).long().flatten()
                self.last_homeostasis_update_mode = (
                    "candidate_subset"
                    if int(homeostasis_indices.numel()) < self.n_columns
                    else "all_columns"
                )
            self.last_homeostasis_update_count = int(homeostasis_indices.numel())

            state_transition_next_step = int(self.state_transition_step_count) + 1
            if int(homeostasis_indices.numel()) >= int(self.n_columns):
                self.materialize_state_transition(None, record_noop=False)
                self.steps_since_win += 1
                self.steps_since_win[winners] = 0
                self._mark_all_state_transition_materialized(
                    state_transition_next_step,
                    sync_last_update_tensor=True,
                )
                self.last_state_transition_mode = "dense_all_columns_process"
                self.last_state_transition_column_count = int(self.n_columns)
                self.last_state_transition_cached_count = 0
                sparse_state_transition = False
            else:
                state_indices = self._process_state_transition_indices(
                    homeostasis_indices,
                    winners,
                )
                self._materialize_state_transition_indices(
                    state_indices,
                    mode="candidate_subset",
                    record_noop=False,
                )
                if int(state_indices.numel()) > 0:
                    self.steps_since_win[state_indices] += 1
                    self.steps_since_win[winners] = 0
                    self.steps_since_win_last_update_step[state_indices] = (
                        state_transition_next_step
                    )
                self.last_state_transition_mode = "candidate_subset_lazy_state_transition"
                self.last_state_transition_column_count = int(state_indices.numel())
                self.last_state_transition_cached_count = max(
                    0,
                    int(self.n_columns) - int(state_indices.numel()),
                )
                sparse_state_transition = True
            self.state_transition_step_count = state_transition_next_step
            self.materialize_homeostasis(homeostasis_indices, record_noop=False)
            current_homeostasis_step = int(self.homeostasis_step_count)
            if (
                current_homeostasis_step < len(self.threshold_relaxation_history)
                and self.threshold_relaxation_history[current_homeostasis_step]
            ):
                self._apply_threshold_relaxation(
                    homeostasis_indices,
                    step=current_homeostasis_step,
                )

            activity = torch.zeros(self.n_columns, device=self.device)
            activity[winners] = 1.0 / max(1, winners.numel())
            self._record_recent_spike_sample(winners, sparse=sparse_state_transition)
            if int(homeostasis_indices.numel()) > 0:
                self.win_rate_ema[homeostasis_indices] = (
                    (1.0 - self.homeostasis_beta) * self.win_rate_ema[homeostasis_indices]
                    + self.homeostasis_beta * activity[homeostasis_indices]
                )
                self.thresholds[homeostasis_indices] = torch.clamp(
                    self.thresholds[homeostasis_indices]
                    + self.homeostasis_lr
                    * (self.win_rate_ema[homeostasis_indices] - self.target_firing_rate),
                    min=self.threshold_min,
                    max=self.threshold_max,
                )
                self.homeostasis_last_update_step[homeostasis_indices] = (
                    int(self.homeostasis_step_count) + 1
                )

            self.update_count += int(winners.numel())
            self.homeostasis_step_count += 1
        # Invalidate step-level caches (prototypes/weights were updated)
        self._cached_proto_sim = None
        self._cached_raw_drive = None
        return assembly

    def force_revive_dead_columns(self, routing_key: Optional[torch.Tensor] = None) -> int:
        """Forcibly revive all columns whose ``steps_since_win`` meets or exceeds the
        dead-column threshold, regardless of whether a ``process()`` call is pending.

        Called by the trainer's norepinephrine-based self-repair reflex when sustained
        prediction failure indicates the network needs new routing capacity.

        Returns the number of columns revived.
        """
        self.materialize_state_transition(None, record_noop=False)
        dead_mask = self.steps_since_win >= self.dead_column_steps
        if not bool(dead_mask.any()):
            return 0

        n_dead = int(dead_mask.sum().item())

        if routing_key is not None:
            x = _normalize_positive_vector(routing_key.to(self.device).unsqueeze(0), dim=1).squeeze(0)
        else:
            x = _normalize_positive_vector(torch.rand(1, self.column_dim, device=self.device), dim=1).squeeze(0)

        noise = torch.rand(n_dead, self.column_dim, device=self.device) * self.dead_column_noise
        revived = _normalize_positive_vector(x.unsqueeze(0) + noise, dim=1)
        self.prototypes[dead_mask] = revived
        self.prototype_velocity[dead_mask] = 0.0

        if self.last_input_pattern is not None:
            revived_weights = self.last_input_pattern.unsqueeze(0).repeat(n_dead, 1)
        else:
            revived_weights = torch.full((n_dead, self.input_dim), 1.0 / self.input_dim, device=self.device)
        revived_weights = revived_weights + torch.rand_like(revived_weights) * self.dead_column_noise
        revived_weights = torch.clamp(revived_weights, min=1e-6)
        revived_weights = revived_weights * (
            self.input_weight_row_target / (revived_weights.sum(dim=1, keepdim=True) + 1e-8)
        )
        self.input_weights[dead_mask] = revived_weights
        self.thresholds[dead_mask] = self.threshold_min
        self.win_rate_ema[dead_mask] = self.target_firing_rate
        self.homeostasis_last_update_step[dead_mask] = int(
            self.homeostasis_step_count
        )
        self.threshold_relaxation_last_applied_step[dead_mask] = int(
            self.homeostasis_step_count
        )
        self.steps_since_win[dead_mask] = 0
        self.steps_since_win_last_update_step[dead_mask] = int(
            self.state_transition_step_count
        )
        self.last_revived_indices = torch.nonzero(dead_mask, as_tuple=False).squeeze(1).detach().clone()
        if self.local_plasticity is not None:
            self.local_plasticity.revive_columns(self.last_revived_indices)
        return n_dead

    def nearest_prototype_distance(self, routing_key: torch.Tensor) -> float:
        x = _normalize_positive_vector(routing_key.to(self.device).unsqueeze(0), dim=1).squeeze(0)
        sim = torch.mv(self.prototypes, x)
        return max(0.0, float(1.0 - sim.max().item()))

    def get_current_assemblies(self) -> dict:
        return {
            "prototypes": self.prototypes.detach().clone(),
            "n_columns": self.n_columns,
            "thresholds": self.thresholds.detach().clone(),
            "win_rate_ema": self.win_rate_ema.detach().clone(),
            "input_weights": self.input_weights.detach().clone(),
            "last_projected_input": None if self.last_projected_input is None else self.last_projected_input.detach().clone(),
        }

    def device_report(self) -> dict[str, Any]:
        """Return runtime-visible device placement for column state."""
        return {
            "module": "competitive_columns",
            "device": str(self.device),
            "W_project_device": str(self.W_project.device),
            "input_weights_device": str(self.input_weights.device),
            "prototypes_device": str(self.prototypes.device),
            "prototype_velocity_device": str(self.prototype_velocity.device),
            "thresholds_device": str(self.thresholds.device),
            "win_rate_ema_device": str(self.win_rate_ema.device),
            "homeostasis_step_count": int(self.homeostasis_step_count),
            "homeostasis_last_update_step_device": str(
                self.homeostasis_last_update_step.device
            ),
            "state_transition_step_count": int(self.state_transition_step_count),
            "steps_since_win_last_update_step_device": str(
                self.steps_since_win_last_update_step.device
            ),
            "last_state_transition_mode": str(self.last_state_transition_mode),
            "last_state_transition_column_count": int(
                self.last_state_transition_column_count
            ),
            "last_state_transition_cached_count": int(
                self.last_state_transition_cached_count
            ),
            "last_state_transition_materialize_mode": str(
                self.last_state_transition_materialize_mode
            ),
            "last_state_transition_materialize_count": int(
                self.last_state_transition_materialize_count
            ),
            "last_state_transition_materialize_max_age": int(
                self.last_state_transition_materialize_max_age
            ),
            "threshold_relaxation_event_count": int(
                sum(1 for item in self.threshold_relaxation_history if item)
            ),
            "threshold_relaxation_history_length": int(
                len(self.threshold_relaxation_history)
            ),
            "threshold_relaxation_last_applied_step_device": str(
                self.threshold_relaxation_last_applied_step.device
            ),
            "last_homeostasis_materialize_mode": str(
                self.last_homeostasis_materialize_mode
            ),
            "last_homeostasis_materialize_count": int(
                self.last_homeostasis_materialize_count
            ),
            "last_homeostasis_materialize_max_age": int(
                self.last_homeostasis_materialize_max_age
            ),
            "steps_since_win_device": str(self.steps_since_win.device),
            "last_revived_indices_device": str(self.last_revived_indices.device),
            "recent_spike_window_device": str(self.recent_spike_window.device),
            "input_weight_blend": float(self.input_weight_blend),
            "input_plasticity_mode": str(self.last_input_plasticity_mode),
            "input_plasticity_update_count": int(self.input_plasticity_update_count),
            "input_plasticity_skip_count": int(self.input_plasticity_skip_count),
            "last_input_pattern_device": (
                None if self.last_input_pattern is None else str(self.last_input_pattern.device)
            ),
            "last_projected_input_device": (
                None if self.last_projected_input is None else str(self.last_projected_input.device)
            ),
            "local_plasticity": (
                None if self.local_plasticity is None else self.local_plasticity.device_report()
            ),
        }

    def execution_report(self) -> dict[str, Any]:
        """Return the last observed competitive-column execution boundary."""
        scored = max(0, min(int(self.last_scored_column_count), self.n_columns))
        candidates = max(0, min(int(self.last_candidate_count), self.n_columns))
        homeostasis_count = max(
            0,
            min(int(self.last_homeostasis_update_count), self.n_columns),
        )
        state_transition_count = max(
            0,
            min(int(self.last_state_transition_column_count), self.n_columns),
        )
        state_transition_cached_count = max(
            0,
            min(int(self.last_state_transition_cached_count), self.n_columns),
        )
        state_transition_runs_all_columns = bool(
            self.last_state_transition_mode != "not_run"
            and state_transition_count >= self.n_columns
        )
        runs_all_columns = bool(
            (
                self.last_execution_mode != "not_run"
                and (
                    scored >= self.n_columns
                    or (
                        self.last_homeostasis_update_mode != "not_run"
                        and homeostasis_count >= self.n_columns
                    )
                )
            )
            or state_transition_runs_all_columns
        )
        fallback_reason = None
        if self.last_execution_mode == "all_columns_candidate_set":
            fallback_reason = "candidate_set_covers_all_columns"
        elif state_transition_runs_all_columns and scored < self.n_columns:
            fallback_reason = "state_transition_dense_all_columns_retained"
        return {
            "mode": str(self.last_execution_mode),
            "total_columns": int(self.n_columns),
            "candidate_count": candidates,
            "scored_column_count": scored,
            "runs_all_columns": runs_all_columns,
            "state_transition_mode": str(self.last_state_transition_mode),
            "state_transition_column_count": state_transition_count,
            "state_transition_cached_count": state_transition_cached_count,
            "state_transition_cached_fraction": round(
                float(state_transition_cached_count) / float(max(1, self.n_columns)),
                6,
            ),
            "state_transition_runs_all_columns": state_transition_runs_all_columns,
            "state_transition_step_count": int(self.state_transition_step_count),
            "state_transition_materialize_mode": str(
                self.last_state_transition_materialize_mode
            ),
            "state_transition_materialize_count": max(
                0,
                min(int(self.last_state_transition_materialize_count), self.n_columns),
            ),
            "state_transition_materialize_max_age": int(
                max(0, self.last_state_transition_materialize_max_age)
            ),
            "scored_column_fraction": round(
                float(scored) / float(max(1, self.n_columns)),
                6,
            ),
            "homeostasis_update_mode": str(self.last_homeostasis_update_mode),
            "homeostasis_update_count": homeostasis_count,
            "homeostasis_update_fraction": round(
                float(homeostasis_count)
                / float(max(1, self.n_columns)),
                6,
            ),
            "homeostasis_materialize_mode": str(
                self.last_homeostasis_materialize_mode
            ),
            "homeostasis_materialize_count": max(
                0,
                min(int(self.last_homeostasis_materialize_count), self.n_columns),
            ),
            "homeostasis_materialize_max_age": int(
                max(0, self.last_homeostasis_materialize_max_age)
            ),
            "homeostasis_step_count": int(self.homeostasis_step_count),
            "threshold_relaxation_event_count": int(
                sum(1 for item in self.threshold_relaxation_history if item)
            ),
            "threshold_relaxation_history_length": int(
                len(self.threshold_relaxation_history)
            ),
            "input_weight_blend": float(self.input_weight_blend),
            "input_plasticity_mode": str(self.last_input_plasticity_mode),
            "input_plasticity_update_count": int(self.input_plasticity_update_count),
            "input_plasticity_skip_count": int(self.input_plasticity_skip_count),
            "dormant_input_plasticity_skipped": bool(
                self.last_input_plasticity_mode == "skipped_zero_blend"
            ),
            "sparse_candidate_execution_observed": bool(
                self.last_execution_mode.startswith("candidate_subset")
                and 0 < scored < self.n_columns
                and not state_transition_runs_all_columns
            ),
            "tensor_device": str(self.device),
            "fallback_reason": fallback_reason,
            "claim_boundary": (
                "observed_competitive_scoring_and_state_transition_scope"
            ),
        }

    def _recent_spike_samples(self) -> torch.Tensor:
        count = int(self.recent_spike_window_count)
        if count <= 0:
            return self.recent_spike_window[:0]
        if count < self.spike_history_window:
            return self.recent_spike_window[:count]
        cursor = int(self.recent_spike_window_cursor)
        return torch.cat(
            (
                self.recent_spike_window[cursor:],
                self.recent_spike_window[:cursor],
            ),
            dim=0,
        )

    def _spike_correlation_report(self) -> dict[str, Any]:
        samples = self._recent_spike_samples().detach().float()
        sample_count = int(samples.shape[0])
        min_samples = 4
        if sample_count < min_samples:
            return {
                "available": False,
                "status": "insufficient_window",
                "sample_count": sample_count,
                "window_size": int(self.spike_history_window),
                "min_samples": min_samples,
                "active_columns": 0,
                "valid_pairs": 0,
                "mean_abs_offdiag_correlation": None,
                "max_abs_offdiag_correlation": None,
            }

        active_counts = samples.sum(dim=0)
        variances = samples.var(dim=0, unbiased=False)
        active_mask = (active_counts > 0.0) & (variances > 1e-8)
        active_samples = samples[:, active_mask]
        active_columns = int(active_samples.shape[1])
        if active_columns < 2:
            return {
                "available": False,
                "status": "insufficient_active_columns",
                "sample_count": sample_count,
                "window_size": int(self.spike_history_window),
                "min_samples": min_samples,
                "active_columns": active_columns,
                "valid_pairs": 0,
                "mean_abs_offdiag_correlation": None,
                "max_abs_offdiag_correlation": None,
            }

        centered = active_samples - active_samples.mean(dim=0, keepdim=True)
        norms = torch.linalg.norm(centered, dim=0).clamp(min=1e-8)
        corr = (centered.t() @ centered) / torch.outer(norms, norms)
        offdiag_mask = ~torch.eye(active_columns, device=self.device, dtype=torch.bool)
        offdiag_abs = corr[offdiag_mask].abs()
        valid_pairs = int(offdiag_abs.numel())
        mean_abs = float(offdiag_abs.mean().item()) if valid_pairs else 0.0
        max_abs = float(offdiag_abs.max().item()) if valid_pairs else 0.0
        overcorrelated_threshold = 0.85
        return {
            "available": True,
            "status": (
                "overcorrelated_risk"
                if mean_abs >= overcorrelated_threshold
                else "measured"
            ),
            "sample_count": sample_count,
            "window_size": int(self.spike_history_window),
            "min_samples": min_samples,
            "active_columns": active_columns,
            "valid_pairs": valid_pairs,
            "mean_abs_offdiag_correlation": mean_abs,
            "max_abs_offdiag_correlation": max_abs,
            "overcorrelated_threshold": overcorrelated_threshold,
        }

    def spike_health_report(self) -> dict[str, Any]:
        """Return read-only spike activity evidence from live column tensors."""
        win_rate = self.win_rate_ema.detach().float()
        target = float(self.target_firing_rate)
        silent_threshold = max(0.0, target * 0.10)
        saturation_threshold = min(1.0, max(target * 5.0, 0.25))
        dead_threshold = int(self.dead_column_steps)
        silent_mask = win_rate <= silent_threshold
        saturated_mask = win_rate >= saturation_threshold
        stale_mask = self._state_transition_steps_snapshot().detach() >= dead_threshold

        local_plasticity = self.local_plasticity
        last_post_spike_fraction = (
            None
            if local_plasticity is None
            else float(local_plasticity.last_post_spike_fraction)
        )
        mean_membrane_voltage = (
            None
            if local_plasticity is None
            else float(local_plasticity.last_mean_membrane_voltage)
        )
        spike_backend = (
            None
            if local_plasticity is None
            else str(local_plasticity.spike_backend)
        )
        plasticity_mode = "local_stdp" if local_plasticity is not None else "competitive_proxy"

        win_mean = float(win_rate.mean().item()) if int(win_rate.numel()) else 0.0
        silent_fraction = float(silent_mask.float().mean().item()) if int(win_rate.numel()) else 0.0
        saturated_fraction = float(saturated_mask.float().mean().item()) if int(win_rate.numel()) else 0.0
        stale_fraction = float(stale_mask.float().mean().item()) if int(stale_mask.numel()) else 0.0
        correlation = self._spike_correlation_report()

        if silent_fraction >= 0.80 or (
            last_post_spike_fraction is not None and last_post_spike_fraction <= silent_threshold
        ):
            activity_state = "silent_risk"
        elif saturated_fraction >= 0.20 or (
            last_post_spike_fraction is not None and last_post_spike_fraction >= saturation_threshold
        ):
            activity_state = "saturation_risk"
        elif stale_fraction >= 0.20:
            activity_state = "stale_routing_risk"
        else:
            activity_state = "sparse_responsive"

        return {
            "schema_version": 1,
            "source": "competitive_columns",
            "available": True,
            "n_columns": int(self.n_columns),
            "plasticity_mode": plasticity_mode,
            "spike_backend": spike_backend,
            "activity_state": activity_state,
            "last_post_spike_fraction": last_post_spike_fraction,
            "mean_membrane_voltage": mean_membrane_voltage,
            "target_firing_rate": target,
            "thresholds": {
                "silent_win_rate_ema": silent_threshold,
                "saturated_win_rate_ema": saturation_threshold,
                "stale_steps_since_win": dead_threshold,
            },
            "win_rate_ema_mean": win_mean,
            "win_rate_ema_min": float(win_rate.min().item()) if int(win_rate.numel()) else 0.0,
            "win_rate_ema_max": float(win_rate.max().item()) if int(win_rate.numel()) else 0.0,
            "silent_fraction": silent_fraction,
            "saturated_fraction": saturated_fraction,
            "stale_fraction": stale_fraction,
            "correlation_evidence_available": bool(correlation["available"]),
            "correlation": correlation,
            "missing_evidence": (
                []
                if bool(correlation["available"])
                else ["spike_train_window_correlation"]
            ),
            "device": str(self.device),
            "win_rate_ema_device": str(self.win_rate_ema.device),
            "recent_spike_window_device": str(self.recent_spike_window.device),
            "not_liveness_claim": True,
        }

    def state_dict(self) -> dict[str, Any]:
        """Serialize all learned state for checkpoint persistence."""
        def _clone_opt(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if t is None else t.detach().clone().cpu()

        logical_steps_since_win = self._state_transition_steps_snapshot()
        materialized_state_step = torch.full(
            (self.n_columns,),
            int(self.state_transition_step_count),
            device=self.device,
            dtype=torch.long,
        )

        return {
            "W_project": self.W_project.detach().clone().cpu(),
            "input_weights": self.input_weights.detach().clone().cpu(),
            "prototypes": self.prototypes.detach().clone().cpu(),
            "prototype_velocity": self.prototype_velocity.detach().clone().cpu(),
            "last_input_pattern": _clone_opt(self.last_input_pattern),
            "last_projected_input": _clone_opt(self.last_projected_input),
            "thresholds": self.thresholds.detach().clone().cpu(),
            "target_firing_rate": float(self.target_firing_rate),
            "win_rate_ema": self.win_rate_ema.detach().clone().cpu(),
            "homeostasis_step_count": int(self.homeostasis_step_count),
            "homeostasis_last_update_step": (
                self.homeostasis_last_update_step.detach().clone().cpu()
            ),
            "threshold_relaxation_history": torch.tensor(
                [1 if item else 0 for item in self.threshold_relaxation_history],
                dtype=torch.uint8,
            ),
            "threshold_relaxation_last_applied_step": (
                self.threshold_relaxation_last_applied_step.detach().clone().cpu()
            ),
            "state_transition_step_count": torch.tensor(
                int(self.state_transition_step_count),
                dtype=torch.long,
            ),
            "state_transition_all_materialized_step": torch.tensor(
                int(self.state_transition_step_count),
                dtype=torch.long,
            ),
            "state_transition_last_update_tensor_materialized_step": torch.tensor(
                int(self.state_transition_step_count),
                dtype=torch.long,
            ),
            "steps_since_win": logical_steps_since_win.detach().clone().cpu(),
            "steps_since_win_last_update_step": (
                materialized_state_step.detach().clone().cpu()
            ),
            "recent_spike_window": self.recent_spike_window.detach().clone().cpu(),
            "recent_spike_window_active_ids": (
                self.recent_spike_window_active_ids.detach().clone().cpu()
            ),
            "recent_spike_window_cursor": int(self.recent_spike_window_cursor),
            "recent_spike_window_count": int(self.recent_spike_window_count),
            "update_count": int(self.update_count),
            "last_revived_indices": self.last_revived_indices.detach().clone().cpu(),
            "local_plasticity": (
                None if self.local_plasticity is None
                else self.local_plasticity.state_dict()
            ),
        }

    def load_state_dict(self, snapshot: dict[str, Any]) -> None:
        """Restore learned state from a checkpoint snapshot."""
        for attr in (
            "W_project", "input_weights", "prototypes",
            "prototype_velocity", "thresholds",
            "win_rate_ema", "steps_since_win", "recent_spike_window",
        ):
            value = snapshot.get(attr)
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(self.device))

        for attr in ("last_input_pattern", "last_projected_input"):
            value = snapshot.get(attr)
            if value is None:
                setattr(self, attr, None)
            elif isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(self.device))

        self.target_firing_rate = float(
            snapshot.get("target_firing_rate", 1.0 / max(1, self.n_columns))
        )
        self.update_count = int(snapshot.get("update_count", 0))
        self.homeostasis_step_count = int(
            snapshot.get("homeostasis_step_count", self.update_count)
        )
        state_step = snapshot.get("state_transition_step_count")
        if isinstance(state_step, torch.Tensor):
            self.state_transition_step_count = int(state_step.item())
        else:
            self.state_transition_step_count = int(
                snapshot.get("state_transition_step_count", self.update_count)
            )
        self.state_transition_all_materialized_step = 0
        self.state_transition_last_update_tensor_materialized_step = 0
        last_homeostasis_step = snapshot.get("homeostasis_last_update_step")
        if (
            isinstance(last_homeostasis_step, torch.Tensor)
            and tuple(last_homeostasis_step.shape) == (self.n_columns,)
        ):
            self.homeostasis_last_update_step = last_homeostasis_step.to(
                self.device,
                dtype=torch.long,
            )
        else:
            self.homeostasis_last_update_step = torch.full(
                (self.n_columns,),
                int(self.homeostasis_step_count),
                device=self.device,
                dtype=torch.long,
            )
        last_state_step = snapshot.get("steps_since_win_last_update_step")
        state_last_is_uniform_current = False
        if (
            isinstance(last_state_step, torch.Tensor)
            and tuple(last_state_step.shape) == (self.n_columns,)
        ):
            self.steps_since_win_last_update_step = last_state_step.to(
                self.device,
                dtype=torch.long,
            )
            state_last_is_uniform_current = bool(
                torch.all(
                    self.steps_since_win_last_update_step
                    == int(self.state_transition_step_count)
                ).item()
            )
        else:
            self.steps_since_win_last_update_step = torch.full(
                (self.n_columns,),
                int(self.state_transition_step_count),
                device=self.device,
                dtype=torch.long,
            )
            state_last_is_uniform_current = True
        all_materialized_step = snapshot.get("state_transition_all_materialized_step")
        tensor_materialized_step = snapshot.get(
            "state_transition_last_update_tensor_materialized_step"
        )
        if isinstance(all_materialized_step, torch.Tensor):
            self.state_transition_all_materialized_step = int(
                all_materialized_step.item()
            )
        elif all_materialized_step is not None:
            self.state_transition_all_materialized_step = int(all_materialized_step)
        elif state_last_is_uniform_current:
            self.state_transition_all_materialized_step = int(
                self.state_transition_step_count
            )
        if isinstance(tensor_materialized_step, torch.Tensor):
            self.state_transition_last_update_tensor_materialized_step = int(
                tensor_materialized_step.item()
            )
        elif tensor_materialized_step is not None:
            self.state_transition_last_update_tensor_materialized_step = int(
                tensor_materialized_step
            )
        elif state_last_is_uniform_current:
            self.state_transition_last_update_tensor_materialized_step = int(
                self.state_transition_step_count
            )
        relaxation_history = snapshot.get("threshold_relaxation_history")
        if isinstance(relaxation_history, torch.Tensor):
            self.threshold_relaxation_history = [
                bool(item)
                for item in relaxation_history.detach().cpu().flatten().tolist()
            ]
        elif isinstance(relaxation_history, list):
            self.threshold_relaxation_history = [bool(item) for item in relaxation_history]
        else:
            self.threshold_relaxation_history = []
        relaxation_last = snapshot.get("threshold_relaxation_last_applied_step")
        if (
            isinstance(relaxation_last, torch.Tensor)
            and tuple(relaxation_last.shape) == (self.n_columns,)
        ):
            self.threshold_relaxation_last_applied_step = relaxation_last.to(
                self.device,
                dtype=torch.long,
            )
        else:
            self.threshold_relaxation_last_applied_step = torch.full(
                (self.n_columns,),
                int(self.homeostasis_step_count),
                device=self.device,
                dtype=torch.long,
            )
        self.recent_spike_window_cursor = int(
            snapshot.get("recent_spike_window_cursor", 0)
        ) % self.spike_history_window
        self.recent_spike_window_count = min(
            self.spike_history_window,
            max(0, int(snapshot.get("recent_spike_window_count", 0))),
        )
        active_ids = snapshot.get("recent_spike_window_active_ids")
        if (
            isinstance(active_ids, torch.Tensor)
            and tuple(active_ids.shape)
            == (self.spike_history_window, max(1, self.n_winners))
        ):
            self.recent_spike_window_active_ids = active_ids.to(
                self.device,
                dtype=torch.long,
            )
        else:
            self.recent_spike_window_active_ids = torch.full(
                (self.spike_history_window, max(1, self.n_winners)),
                -1,
                device=self.device,
                dtype=torch.long,
            )
            dense_rows = self.recent_spike_window.detach()
            for row in range(min(self.spike_history_window, int(dense_rows.shape[0]))):
                active = torch.nonzero(dense_rows[row] > 0.0, as_tuple=False).flatten()
                active = active[: max(1, self.n_winners)].to(
                    self.device,
                    dtype=torch.long,
                )
                if int(active.numel()) > 0:
                    self.recent_spike_window_active_ids[row, : int(active.numel())] = (
                        active
                    )
        revived = snapshot.get("last_revived_indices")
        if isinstance(revived, torch.Tensor):
            self.last_revived_indices = revived.to(self.device)
        else:
            self.last_revived_indices = torch.empty(0, device=self.device, dtype=torch.long)

        if (
            self.local_plasticity is not None
            and snapshot.get("local_plasticity") is not None
        ):
            self.local_plasticity.load_state_dict(snapshot["local_plasticity"])
