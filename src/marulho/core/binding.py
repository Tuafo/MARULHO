"""Coincidence binding layer with short-term plasticity (STP).

Implements a Tsodyks-Markram synapse model with facilitation, depression,
and PV+ fast feedforward inhibition. Binds co-occurring column activations
into composite assemblies.

This is the dense binding mode. For O(N*d) scaling at >1K columns,
use HypercubeBindingLayer (core/hypercube.py) instead.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def _row_normalize(matrix: torch.Tensor) -> torch.Tensor:
    """Normalize each row to sum to 1 (used for output weight matrices)."""
    row_sums = torch.clamp(matrix.sum(dim=1, keepdim=True), min=1e-8)
    return matrix / row_sums


def _normalize(vec: torch.Tensor) -> torch.Tensor:
    """Clamp-positive then normalize a vector to sum to 1."""
    vec = torch.clamp(vec.float(), min=0.0)
    total = torch.clamp(vec.sum(), min=1e-8)
    return vec / total

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
        n_bindings: int | None = None,
        fan_in: int = 4,
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
        self.n_bindings = self._resolve_n_bindings(n_bindings)
        self.fan_in = self._resolve_fan_in(fan_in)
        self.tau_binding = float(tau_binding)
        self.stp_u_inc = float(stp_u_inc)
        self.stp_tau_f = float(stp_tau_f)
        self.stp_tau_d = float(stp_tau_d)
        self.pv_threshold = float(pv_threshold)
        self.pv_gain = float(pv_gain)

        self.binding_state = torch.zeros(self.n_columns, device=self.device)
        self.binding_outputs = torch.zeros(self.n_bindings, device=self.device)
        self.coincidence_trace = torch.zeros(self.n_bindings, device=self.device)
        self.facilitation = torch.zeros(self.n_bindings, device=self.device)
        self.resources = torch.ones(self.n_bindings, device=self.device)
        self.binding_usage = torch.zeros(self.n_bindings, device=self.device)
        self.pv_inhibition = torch.tensor(0.0, device=self.device)
        self.connectivity = self._initial_connectivity()
        self.output_weights = _row_normalize(self.connectivity + 1e-6)

    def _resolve_n_bindings(self, requested: int | None) -> int:
        if requested is not None and int(requested) > 0:
            return int(requested)
        if self.n_columns <= 16:
            return self.n_columns
        extra = min(32, max(4, self.n_columns // 4))
        return max(8, self.n_columns + extra)

    def _resolve_fan_in(self, requested: int) -> int:
        if self.n_columns <= 2:
            return self.n_columns
        return max(2, min(int(requested), self.n_columns - 1))

    def _row_with_sources(self, source_ids: list[int]) -> torch.Tensor:
        row = torch.zeros(self.n_columns, device=self.device)
        row[source_ids] = 1.0
        return row

    def _sample_sources(self, preferred: tuple[int, int] | None = None) -> list[int]:
        selected: set[int] = set()
        if preferred is not None:
            selected.update(int(idx) for idx in preferred if 0 <= int(idx) < self.n_columns)
        while len(selected) < self.fan_in:
            selected.add(int(torch.randint(0, self.n_columns, (1,), device=self.device).item()))
        return sorted(selected)

    def _initial_connectivity(self) -> torch.Tensor:
        rows: list[torch.Tensor] = []
        anchor_rows = min(self.n_bindings, self.n_columns)
        for start in range(anchor_rows):
            sources = [int((start + offset) % self.n_columns) for offset in range(self.fan_in)]
            rows.append(self._row_with_sources(sorted(set(sources))))
        while len(rows) < self.n_bindings:
            rows.append(self._row_with_sources(self._sample_sources()))
        return torch.stack(rows, dim=0)

    def _context_drive(self, signal: torch.Tensor) -> torch.Tensor:
        return torch.mv(self.connectivity, _normalize(signal.to(self.device))) / float(max(1, self.fan_in))

    def _context_drive_fast(self, normed: torch.Tensor) -> torch.Tensor:
        """Like _context_drive but caller guarantees input is already normalized."""
        return torch.mv(self.connectivity, normed) / float(max(1, self.fan_in))

    def _column_prediction_from_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        if outputs.numel() == 0 or outputs.sum() <= 0.0:
            return torch.zeros(self.n_columns, device=self.device)
        predicted = torch.mv(self.output_weights.t(), outputs)
        source_support = torch.mv(self.connectivity.t(), outputs) / float(max(1, self.fan_in))
        return _normalize(torch.relu(0.70 * predicted + 0.30 * source_support))

    def _append_binding_rows(self, new_rows: torch.Tensor) -> int:
        if int(new_rows.numel()) == 0:
            return 0
        count = int(new_rows.shape[0])
        self.connectivity = torch.cat([self.connectivity, new_rows], dim=0)
        self.output_weights = _row_normalize(torch.cat([self.output_weights, new_rows + 1e-6], dim=0))
        self.binding_outputs = torch.cat([self.binding_outputs, torch.zeros(count, device=self.device)], dim=0)
        self.coincidence_trace = torch.cat([self.coincidence_trace, torch.zeros(count, device=self.device)], dim=0)
        self.facilitation = torch.cat([self.facilitation, torch.zeros(count, device=self.device)], dim=0)
        self.resources = torch.cat([self.resources, torch.ones(count, device=self.device)], dim=0)
        self.binding_usage = torch.cat([self.binding_usage, torch.zeros(count, device=self.device)], dim=0)
        self.n_bindings = int(self.connectivity.shape[0])
        return count

    def reset_state(self) -> None:
        self.binding_state.zero_()
        self.binding_outputs.zero_()
        self.coincidence_trace.zero_()
        self.facilitation.zero_()
        self.resources.fill_(1.0)
        self.pv_inhibition.zero_()

    def _binding_prediction(self, context_prediction: torch.Tensor) -> torch.Tensor:
        context = _normalize(context_prediction.to(self.device))
        if context.sum() <= 0.0:
            return torch.zeros(self.n_columns, device=self.device)
        if self.binding_usage.max() <= 1e-6:
            return torch.zeros(self.n_columns, device=self.device)
        predicted_outputs = self._context_drive(context) * self.binding_usage
        return self._column_prediction_from_outputs(predicted_outputs)

    def binding_prediction(self, context_prediction: torch.Tensor) -> torch.Tensor:
        return self._binding_prediction(context_prediction)

    def modulation_gain(self, context_prediction: torch.Tensor) -> torch.Tensor:
        return self.modulation_gain_for_context(context_prediction)

    def modulation_gain_for_context(self, context_prediction: torch.Tensor) -> torch.Tensor:
        prediction = self._binding_prediction(context_prediction)
        if prediction.sum() <= 0.0:
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
        # Normalize inputs once at entry — all sub-calls use pre-normalized
        context = _normalize(context_prediction.to(self.device))
        current = _normalize(assembly.to(self.device))
        ctx_sum = context.sum()
        cur_sum = current.sum()
        if ctx_sum <= 0.0 or cur_sum <= 0.0:
            self.coincidence_trace *= max(0.0, 1.0 - 1.0 / self.tau_binding)
            self.binding_state.zero_()
            self.binding_outputs.zero_()
            return torch.zeros_like(current), 0.0

        # Compute context drive once — reused in _binding_prediction below
        context_drive = self._context_drive_fast(context)
        current_drive = self._context_drive_fast(current)
        joint_drive = torch.minimum(context_drive, current_drive)
        context_gate = 0.5 + 0.5 * context.max()

        # STP update with pre-normalized joint_drive
        jd_total = joint_drive.sum()
        if jd_total > 1e-8:
            jd_norm = joint_drive / jd_total
        else:
            jd_norm = joint_drive
        self.facilitation = torch.clamp(
            self.facilitation * (1.0 - 1.0 / self.stp_tau_f)
            + self.stp_u_inc * jd_norm * (1.0 - self.facilitation),
            min=0.0,
            max=1.0,
        )
        release_raw = torch.clamp(self.facilitation * self.resources * jd_norm, min=0.0)
        self.resources = torch.clamp(
            self.resources + (1.0 - self.resources) / self.stp_tau_d - release_raw,
            min=0.0,
            max=1.0,
        )
        release = release_raw * context_gate

        self.coincidence_trace = torch.clamp(
            self.coincidence_trace * max(0.0, 1.0 - 1.0 / self.tau_binding) + release,
            min=0.0,
        )

        # Binding prediction reuses cached context_drive
        if self.binding_usage.max() > 1e-6:
            predicted_outputs = context_drive * self.binding_usage
            po_sum = predicted_outputs.sum()
            if po_sum > 0.0:
                predicted = torch.mv(self.output_weights.t(), predicted_outputs)
                source_support = torch.mv(self.connectivity.t(), predicted_outputs) / float(max(1, self.fan_in))
                combined = torch.relu(0.70 * predicted + 0.30 * source_support)
                c_total = combined.sum()
                learned_prediction = combined / torch.clamp(c_total, min=1e-8) if c_total > 1e-8 else combined
            else:
                learned_prediction = torch.zeros(self.n_columns, device=self.device)
        else:
            learned_prediction = torch.zeros(self.n_columns, device=self.device)

        activity_sum = release.sum()
        pv_excess = torch.clamp(activity_sum - self.pv_threshold, min=0.0)
        self.pv_inhibition = 0.85 * self.pv_inhibition + self.pv_gain * pv_excess

        self.binding_outputs = torch.relu(
            self.coincidence_trace + 0.50 * joint_drive - self.threshold - self.pv_inhibition
        )

        # Column prediction from binding outputs
        bo_sum = self.binding_outputs.sum()
        if self.binding_outputs.numel() > 0 and bo_sum > 0.0:
            pred_bo = torch.mv(self.output_weights.t(), self.binding_outputs)
            ss_bo = torch.mv(self.connectivity.t(), self.binding_outputs) / float(max(1, self.fan_in))
            cs_raw = torch.relu(0.70 * pred_bo + 0.30 * ss_bo)
            cs_total = cs_raw.sum()
            column_support = cs_raw / torch.clamp(cs_total, min=1e-8) if cs_total > 1e-8 else cs_raw
        else:
            column_support = torch.zeros(self.n_columns, device=self.device)

        bound = torch.relu(
            current
            + self.gain_strength * column_support
            + 0.50 * self.gain_strength * learned_prediction
            - self.pv_inhibition * current.mean()
        )
        strength_signal = column_support + 0.50 * learned_prediction
        strength = float(torch.minimum(current, strength_signal).sum().item())

        if update_weights:
            learning_drive = torch.clamp(self.binding_outputs + joint_drive, min=0.0, max=1.0)
            target_raw = 0.35 * context + 0.65 * current
            t_total = torch.clamp(target_raw.sum(), min=1e-8)
            target = target_raw / t_total
            updated = (
                self.association_decay * self.output_weights
                + self.association_lr * torch.outer(learning_drive, target)
                + 0.05 * self.connectivity
            )
            self.output_weights = _row_normalize(torch.clamp(updated, min=1e-6))
            self.binding_usage = torch.clamp(
                self.association_decay * self.binding_usage + learning_drive,
                min=0.0,
                max=1.0,
            )

        if bound.sum() <= 0.0:
            self.binding_state.zero_()
            return torch.zeros_like(bound), strength

        self.binding_state = _normalize(bound)
        return self.binding_state, strength

    def grow_binding(self, high_correlation_columns: list[tuple[int, int, float]]) -> int:
        new_rows: list[torch.Tensor] = []
        for col_a, col_b, corr in high_correlation_columns:
            a = int(col_a)
            b = int(col_b)
            if corr <= 0.7 or a == b:
                continue
            if not (0 <= a < self.n_columns and 0 <= b < self.n_columns):
                continue
            covered = bool(((self.connectivity[:, a] > 0.0) & (self.connectivity[:, b] > 0.0)).any().item())
            if covered:
                continue
            new_rows.append(self._row_with_sources(self._sample_sources((a, b))))
        if not new_rows:
            return 0
        return self._append_binding_rows(torch.stack(new_rows, dim=0))

    def device_report(self) -> dict[str, object]:
        """Return runtime-visible device placement for binding state."""
        return {
            "module": "binding",
            "device": str(self.device),
            "binding_state_device": str(self.binding_state.device),
            "binding_outputs_device": str(self.binding_outputs.device),
            "coincidence_trace_device": str(self.coincidence_trace.device),
            "facilitation_device": str(self.facilitation.device),
            "resources_device": str(self.resources.device),
            "binding_usage_device": str(self.binding_usage.device),
            "pv_inhibition_device": str(self.pv_inhibition.device),
            "connectivity_device": str(self.connectivity.device),
            "output_weights_device": str(self.output_weights.device),
            "n_bindings": int(self.n_bindings),
            "fan_in": int(self.fan_in),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "n_bindings": int(self.n_bindings),
            "fan_in": int(self.fan_in),
            "binding_state": self.binding_state.detach().clone().cpu(),
            "binding_outputs": self.binding_outputs.detach().clone().cpu(),
            "coincidence_trace": self.coincidence_trace.detach().clone().cpu(),
            "facilitation": self.facilitation.detach().clone().cpu(),
            "resources": self.resources.detach().clone().cpu(),
            "binding_usage": self.binding_usage.detach().clone().cpu(),
            "pv_inhibition": self.pv_inhibition.detach().clone().cpu(),
            "connectivity": self.connectivity.detach().clone().cpu(),
            "output_weights": self.output_weights.detach().clone().cpu(),
            "coincidence_weights": self.output_weights.detach().clone().cpu(),
        }

    def load_state_dict(self, snapshot: dict[str, Any]) -> None:
        connectivity = snapshot.get("connectivity")
        if isinstance(connectivity, torch.Tensor):
            self.connectivity = connectivity.to(self.device)
            self.n_bindings = int(self.connectivity.shape[0])
        else:
            legacy_weights = snapshot.get("coincidence_weights")
            if isinstance(legacy_weights, torch.Tensor):
                self.connectivity = torch.eye(self.n_columns, device=self.device)
                self.n_bindings = self.n_columns
            else:
                self.connectivity = self._initial_connectivity()
                self.n_bindings = int(self.connectivity.shape[0])
        self.fan_in = int(snapshot.get("fan_in", self.fan_in))

        defaults = {
            "binding_outputs": torch.zeros(self.n_bindings, device=self.device),
            "coincidence_trace": torch.zeros(self.n_bindings, device=self.device),
            "facilitation": torch.zeros(self.n_bindings, device=self.device),
            "resources": torch.ones(self.n_bindings, device=self.device),
            "binding_usage": torch.zeros(self.n_bindings, device=self.device),
            "binding_state": torch.zeros(self.n_columns, device=self.device),
        }
        self.output_weights = _row_normalize(self.connectivity + 1e-6)
        stored_output = snapshot.get("output_weights")
        if isinstance(stored_output, torch.Tensor):
            self.output_weights = _row_normalize(torch.clamp(stored_output.to(self.device), min=1e-6))
        else:
            legacy_weights = snapshot.get("coincidence_weights")
            if isinstance(legacy_weights, torch.Tensor):
                self.output_weights = _row_normalize(torch.clamp(legacy_weights.to(self.device), min=1e-6))

        for attr in (
            "binding_state",
            "binding_outputs",
            "coincidence_trace",
            "facilitation",
            "resources",
            "binding_usage",
            "pv_inhibition",
        ):
            value = snapshot.get(attr)
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(self.device))
            elif attr in defaults:
                setattr(self, attr, defaults[attr])
