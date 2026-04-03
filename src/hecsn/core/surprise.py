from __future__ import annotations

from collections import deque
from typing import Dict, List
import torch


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

    def update(self, layer_name: str, prediction: torch.Tensor, actual: torch.Tensor) -> None:
        err = torch.norm(prediction - actual).item()
        buf = self.layers[layer_name]["errors"]
        buf.append(err)
        if len(buf) >= 10:
            var = torch.var(torch.tensor(list(buf), dtype=torch.float32)).item()
            self.layers[layer_name]["precision"] = 1.0 / (var + 1e-6)

    def update_predicted_error(self, actual_error: float, alpha: float = 0.01) -> None:
        self.predicted_error = alpha * actual_error + (1.0 - alpha) * self.predicted_error

    def compute_dopamine_rpe(self, current_error: float) -> float:
        baseline = self.predicted_error + 1e-6
        frac = (self.predicted_error - current_error) / baseline
        return float(torch.tanh(torch.tensor(frac * 3.0)).item())

    def update_neuromodulators(self, current_error: float, novelty: float) -> None:
        rpe = self.compute_dopamine_rpe(current_error)
        self.dopamine = (rpe + 1.0) / 2.0
        self.update_predicted_error(current_error)

        self.acetylcholine = 0.9 * self.acetylcholine + 0.1 * max(0.0, min(1.0, novelty))
        if abs(rpe) > 0.4:
            self.norepinephrine = min(1.0, self.norepinephrine + 0.1)
        else:
            self.norepinephrine *= 0.95

    def should_reset_network(self, threshold: float = 0.85) -> bool:
        """Return True when sustained norepinephrine exceeds *threshold*.

        High norepinephrine indicates persistent, unresolved surprise — the
        network is consistently failing to predict its input.  This is the
        biological signal for adaptive network remodelling (e.g. reviving dead
        columns, boosting plasticity).  Callers should act on this sparingly
        and only after the cooldown window has elapsed.

        The threshold is deliberately higher than the RPE trigger (0.4) so that
        transient surprises do not cause premature resets.
        """
        return bool(self.norepinephrine > float(threshold))

    def get_modulator(self, layer_name: str) -> float:
        errors = list(self.layers[layer_name]["errors"])
        if len(errors) < 10:
            return 0.5

        recent = errors[-1]
        mean = sum(errors) / len(errors)
        surprise = recent - mean

        raw_precision = float(self.layers[layer_name]["precision"])
        precision_weight = torch.sigmoid(torch.tensor(0.1 * (raw_precision - 10.0))).item()

        dopamine_factor = self.dopamine * 2.0 - 1.0
        mod = surprise * precision_weight * dopamine_factor * self.acetylcholine
        return max(-1.0, min(1.0, float(mod)))
