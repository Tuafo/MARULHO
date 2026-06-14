from __future__ import annotations

import math
from collections import deque
from typing import Dict, List
import torch


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


class SurpriseMonitor:
    """Layer-wise surprise with internally-derived neuromodulators."""

    def __init__(self, layer_names: List[str], history_len: int = 100) -> None:
        self.layers: Dict[str, dict] = {
            name: {"errors": deque(maxlen=history_len), "precision": 1.0}
            for name in layer_names
        }
        self.predicted_error = 0.5
        self.dopamine = 0.5
        self.acetylcholine = 0.5
        self.norepinephrine = 0.5
        self.serotonin = 0.5
        self.modulator_revision = 0

    def mark_modulator_state_changed(self) -> None:
        self.modulator_revision += 1

    def update(self, layer_name: str, prediction: torch.Tensor, actual: torch.Tensor) -> None:
        self.record_error(layer_name, torch.norm(prediction - actual).item())

    def record_error(self, layer_name: str, error: float) -> None:
        err = float(error)
        buf = self.layers[layer_name]["errors"]
        buf.append(err)
        if len(buf) >= 10:
            errors = list(buf)
            mean = sum(errors) / len(errors)
            var = sum((e - mean) ** 2 for e in errors) / (len(errors) - 1)
            self.layers[layer_name]["precision"] = 1.0 / (var + 1e-6)
        self.mark_modulator_state_changed()

    def update_predicted_error(self, actual_error: float, alpha: float = 0.01) -> None:
        self.predicted_error = alpha * actual_error + (1.0 - alpha) * self.predicted_error
        self.mark_modulator_state_changed()

    def compute_dopamine_rpe(self, current_error: float) -> float:
        baseline = self.predicted_error + 1e-6
        frac = (self.predicted_error - current_error) / baseline
        return math.tanh(frac * 3.0)

    def compute_serotonin_punishment(self, current_error: float) -> float:
        baseline = self.predicted_error + 1e-6
        frac = max(0.0, (current_error - self.predicted_error) / baseline)
        return math.tanh(frac * 3.0)

    def compute_unexpected_uncertainty(self, current_error: float) -> float:
        baseline = self.predicted_error + 1e-6
        frac = abs(current_error - self.predicted_error) / baseline
        return _clamp01(math.tanh(frac * 2.0))

    def valence_balance(self) -> float:
        return max(-1.0, min(1.0, 2.0 * (self.dopamine - self.serotonin)))

    def update_neuromodulators(self, current_error: float, novelty: float) -> None:
        rpe = self.compute_dopamine_rpe(current_error)
        serotonin_drive = self.compute_serotonin_punishment(current_error)
        unexpected_uncertainty = self.compute_unexpected_uncertainty(current_error)
        novelty_drive = _clamp01(novelty)

        self.dopamine = _clamp01(0.85 * self.dopamine + 0.15 * max(0.0, rpe))
        self.serotonin = _clamp01(0.85 * self.serotonin + 0.15 * serotonin_drive)
        self.acetylcholine = _clamp01(0.90 * self.acetylcholine + 0.10 * novelty_drive)
        self.norepinephrine = _clamp01(0.85 * self.norepinephrine + 0.15 * unexpected_uncertainty)
        if serotonin_drive > 0.4:
            self.norepinephrine = _clamp01(self.norepinephrine + 0.10 * serotonin_drive)
        self.update_predicted_error(current_error)

    def precision_weight(self, layer_name: str) -> float:
        layer = self.layers.get(layer_name)
        if layer is None:
            return 1.0
        errors = list(layer.get("errors") or [])
        if len(errors) < 10:
            return 1.0
        raw_precision = float(layer.get("precision", 1.0))
        x = 0.1 * (raw_precision - 10.0)
        weight = 1.0 / (1.0 + math.exp(-x))
        return max(0.0, min(1.0, weight))

    def get_modulator(self, layer_name: str) -> float:
        errors = list(self.layers[layer_name]["errors"])
        if len(errors) < 10:
            return 0.5

        recent = errors[-1]
        mean = sum(errors) / len(errors)
        surprise = recent - mean

        precision_weight = self.precision_weight(layer_name)
        valence = self.valence_balance()
        attention = 0.5 + 0.5 * self.acetylcholine
        adaptation = 0.5 + 0.5 * self.norepinephrine
        mod = surprise * precision_weight * valence * attention * adaptation
        return max(-1.0, min(1.0, float(mod)))
