"""Event-camera temporal contrast encoder (§5.1 / Layer 0).

Simulates an event-driven visual sensor from conventional video frames.
An event is generated at pixel (x, y) whenever the log-intensity change
exceeds a contrast threshold C:

    |log I(x, y, t) - log I(x, y, t_prev)| > C

Output is a binary spike map of shape (H // pool, W // pool) flattened to
a 1-D tensor.  Sparsity is controlled by the contrast threshold and the
pooling stride — target operating range is 5–25 % active pixels per frame.
"""

from __future__ import annotations

from typing import Any

import torch


class EventCameraEncoder:
    """Biologically-inspired event-camera encoder.

    Converts greyscale video frames into sparse spike patterns by detecting
    per-pixel log-intensity changes above a contrast threshold.

    Args:
        height: Frame height in pixels.
        width: Frame width in pixels.
        pool: Spatial pooling factor (default 4).
        contrast_threshold: Minimum |Δ log I| to fire (default 0.3).
        device: Torch device.
    """

    def __init__(
        self,
        height: int = 64,
        width: int = 64,
        pool: int = 4,
        contrast_threshold: float = 0.3,
        device: torch.device | None = None,
    ) -> None:
        self.height = int(height)
        self.width = int(width)
        self.pool = int(pool)
        self.contrast_threshold = float(contrast_threshold)
        self.device = device or torch.device("cpu")

        self.out_h = self.height // self.pool
        self.out_w = self.width // self.pool
        self._output_dim = self.out_h * self.out_w

        # Previous log-intensity reference surface
        self._ref_log_intensity: torch.Tensor | None = None
        # Exponential trace for cross-modal grounding
        self._trace: torch.Tensor = torch.zeros(self._output_dim, device=self.device)
        self._trace_tau: float = 10.0  # decay time-constant in functional ticks

    # -- public API ---------------------------------------------------------

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def device_report(self) -> dict[str, Any]:
        return {
            "encoder": "event_camera",
            "device": str(self.device),
            "output_dim": int(self._output_dim),
            "height": int(self.height),
            "width": int(self.width),
            "pool": int(self.pool),
            "trace_device": str(self._trace.device),
            "ref_device": None if self._ref_log_intensity is None else str(self._ref_log_intensity.device),
        }

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode a greyscale frame into a binary spike pattern.

        Args:
            frame: Tensor of shape (H, W) with values in [0, 1].

        Returns:
            Binary spike tensor of shape (output_dim,).
        """
        frame = frame.to(self.device).float()
        if frame.dim() == 3:
            # Average RGB channels to greyscale
            frame = frame.mean(dim=0)
        if frame.shape != (self.height, self.width):
            frame = torch.nn.functional.interpolate(
                frame.unsqueeze(0).unsqueeze(0),
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        log_I = torch.log(frame.clamp(min=1e-6))

        if self._ref_log_intensity is None:
            self._ref_log_intensity = log_I.clone()
            return torch.zeros(self._output_dim, device=self.device)

        # Per-pixel log-intensity change
        delta = (log_I - self._ref_log_intensity).abs()
        events = (delta > self.contrast_threshold).float()

        # Update reference where events fired
        fired_mask = events.bool()
        self._ref_log_intensity = torch.where(fired_mask, log_I, self._ref_log_intensity)

        # Spatial pooling via max-pool
        pooled = torch.nn.functional.max_pool2d(
            events.unsqueeze(0).unsqueeze(0),
            kernel_size=self.pool,
            stride=self.pool,
        ).squeeze(0).squeeze(0)

        spikes = (pooled > 0).float().reshape(-1)

        # Update trace
        decay = (-1.0 / self._trace_tau) if self._trace_tau > 0 else -1.0
        self._trace = self._trace * torch.exp(torch.tensor(decay, device=self.device)) + spikes

        return spikes

    @property
    def trace(self) -> torch.Tensor:
        """Current exponential trace (for cross-modal STDP)."""
        return self._trace.detach()

    def sparsity(self, spikes: torch.Tensor) -> float:
        """Fraction of active pixels in a spike pattern."""
        if spikes.numel() == 0:
            return 0.0
        return float(spikes.sum().item() / spikes.numel())

    def reset(self) -> None:
        """Clear reference frame and trace."""
        self._ref_log_intensity = None
        self._trace = torch.zeros(self._output_dim, device=self.device)

    # -- serialization ------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "pool": self.pool,
            "contrast_threshold": self.contrast_threshold,
            "ref_log_intensity": (
                self._ref_log_intensity.cpu() if self._ref_log_intensity is not None else None
            ),
            "trace": self._trace.cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        ref = state.get("ref_log_intensity")
        self._ref_log_intensity = ref.to(self.device) if ref is not None else None
        tr = state.get("trace")
        if tr is not None:
            self._trace = tr.to(self.device)
