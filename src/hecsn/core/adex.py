from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if str(device).strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class AdExNeuron:
    """Adaptive exponential integrate-and-fire neuron population.

    This is an executable surface for the paper's reference AdEx circuit.
    It now backs the standalone stability runner and an optional local-plasticity
    postsynaptic spike backend inside HECSNModel, but the full recurrent
    AdEx / molecular-STC runtime described in the paper is still unfinished.
    """

    n_neurons: int
    dt: float = 0.5
    device: str | torch.device = "auto"
    burst_mode: bool = True

    def __post_init__(self) -> None:
        self.n = int(self.n_neurons)
        if self.n <= 0:
            raise ValueError("n_neurons must be positive")
        self.dt = float(self.dt)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        self.device = _resolve_device(self.device)

        self.C_m = 200e-3
        self.g_L = 10e-3
        self.E_L = -70.0
        self.V_T = -50.0
        self.delta_T = 2.0
        self.tau_w = 100.0
        self.a = 2e-3
        self.b = 80e-3 if bool(self.burst_mode) else 0.0
        self.V_reset = -58.0
        self.V_peak = 20.0

        self.V = torch.full((self.n,), self.E_L, device=self.device, dtype=torch.float32)
        self.w = torch.zeros(self.n, device=self.device, dtype=torch.float32)
        self.spike_times = torch.full((self.n,), -1.0, device=self.device, dtype=torch.float32)

    def device_report(self) -> dict[str, object]:
        """Return runtime-visible tensor placement for this neuron population."""
        return {
            "module": "adex_neuron",
            "device": str(self.device),
            "n_neurons": int(self.n),
            "voltage_device": str(self.V.device),
            "adaptation_device": str(self.w.device),
            "spike_times_device": str(self.spike_times.device),
            "cuda": self.device.type == "cuda",
        }

    def _current(self, I_syn: torch.Tensor) -> torch.Tensor:
        current = torch.as_tensor(I_syn, dtype=torch.float32, device=self.device).flatten()
        if int(current.numel()) != self.n:
            raise ValueError(f"I_syn must have {self.n} values, got {int(current.numel())}")
        return current

    def _voltage_derivative(
        self,
        voltage: torch.Tensor,
        adaptation: torch.Tensor,
        current: torch.Tensor,
    ) -> torch.Tensor:
        exp_arg = torch.clamp((voltage - self.V_T) / self.delta_T, min=-10.0, max=5.0)
        exp_term = self.delta_T * torch.exp(exp_arg)
        return ((-self.g_L * (voltage - self.E_L) + self.g_L * exp_term - adaptation + current) / self.C_m)

    def step(self, I_syn: torch.Tensor, t: float) -> torch.Tensor:
        """Advance one timestep with a stable Heun-style AdEx update."""

        current = self._current(I_syn)

        dV1 = self._voltage_derivative(self.V, self.w, current)
        V_pred = torch.clamp(self.V + self.dt * dV1, min=self.E_L - 20.0, max=self.V_peak)

        dw1 = (self.a * (self.V - self.E_L) - self.w) / self.tau_w
        w_pred = self.w + self.dt * dw1

        dV2 = self._voltage_derivative(V_pred, w_pred, current)
        self.V = torch.clamp(self.V + 0.5 * self.dt * (dV1 + dV2), min=self.E_L - 20.0, max=self.V_peak)

        dw = (self.a * (self.V - self.E_L) - self.w) / self.tau_w
        self.w = self.w + self.dt * dw

        spikes = self.V >= self.V_peak
        spikes_float = spikes.to(torch.float32)

        self.V = torch.where(spikes, torch.full_like(self.V, self.V_reset), self.V)
        self.w = self.w + spikes_float * self.b

        nan_mask = torch.isnan(self.V) | torch.isinf(self.V) | torch.isnan(self.w) | torch.isinf(self.w)
        if bool(nan_mask.any().item()):
            self.V = torch.where(nan_mask, torch.full_like(self.V, self.E_L), self.V)
            self.w = torch.where(nan_mask, torch.zeros_like(self.w), self.w)

        time_value = torch.full_like(self.spike_times, float(t))
        self.spike_times = torch.where(spikes, time_value, self.spike_times)
        return spikes

    def reset_state(self) -> None:
        self.V.fill_(self.E_L)
        self.w.zero_()
        self.spike_times.fill_(-1.0)

    def state_dict(self) -> dict[str, Any]:
        """Serialize neuron state with CPU tensors for portable checkpoints."""
        return {
            "n_neurons": int(self.n),
            "dt": float(self.dt),
            "burst_mode": bool(self.burst_mode),
            "C_m": float(self.C_m),
            "g_L": float(self.g_L),
            "E_L": float(self.E_L),
            "V_T": float(self.V_T),
            "delta_T": float(self.delta_T),
            "tau_w": float(self.tau_w),
            "a": float(self.a),
            "b": float(self.b),
            "V_reset": float(self.V_reset),
            "V_peak": float(self.V_peak),
            "V": self.V.detach().clone().cpu(),
            "w": self.w.detach().clone().cpu(),
            "spike_times": self.spike_times.detach().clone().cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore neuron state onto this instance's configured device."""
        for attr in (
            "C_m",
            "g_L",
            "E_L",
            "V_T",
            "delta_T",
            "tau_w",
            "a",
            "b",
            "V_reset",
            "V_peak",
        ):
            if attr in state:
                setattr(self, attr, float(state[attr]))
        if "dt" in state:
            self.dt = float(state["dt"])
        if "burst_mode" in state:
            self.burst_mode = bool(state["burst_mode"])
        for attr in ("V", "w", "spike_times"):
            value = state.get(attr)
            current = getattr(self, attr)
            if isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(current.shape):
                setattr(self, attr, value.detach().clone().to(self.device).float())

    @classmethod
    def inhibitory(
        cls,
        n_neurons: int,
        dt: float = 0.5,
        device: str | torch.device = "auto",
    ) -> "AdExNeuron":
        neuron = cls(n_neurons=n_neurons, dt=dt, device=device, burst_mode=False)
        neuron.C_m = 100e-3
        neuron.g_L = 10e-3
        neuron.V_T = -45.0
        neuron.tau_w = 20.0
        neuron.a = 0.0
        neuron.b = 0.0
        neuron.V_reset = -65.0
        neuron.reset_state()
        return neuron
