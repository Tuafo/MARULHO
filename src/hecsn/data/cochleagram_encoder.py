"""Cochleagram (mel-filterbank) encoder (§5.1 / Layer 0).

Converts raw audio waveform chunks into sparse spike patterns using a
mel-filterbank decomposition followed by log compression and thresholding.
Output is a binary spike vector of shape (n_bands,) — target sparsity
10–40 % active bands per frame.

The encoder mimics basilar-membrane frequency decomposition: each band
responds to a narrow frequency range, the log compression mirrors the
Weber–Fechner law, and the threshold mimics inner-hair-cell firing.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def _mel(f: float) -> float:
    return 2595.0 * math.log10(1.0 + f / 700.0)


def _inv_mel(m: float) -> float:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(n_bands: int, n_fft: int, sample_rate: int) -> torch.Tensor:
    """Build a mel-filterbank matrix of shape (n_bands, n_fft // 2 + 1)."""
    low_mel = _mel(0.0)
    high_mel = _mel(sample_rate / 2.0)
    mels = torch.linspace(low_mel, high_mel, n_bands + 2)
    freqs = torch.tensor([_inv_mel(float(m)) for m in mels])
    bins = torch.floor((n_fft + 1) * freqs / sample_rate).long()

    n_freqs = n_fft // 2 + 1
    fb = torch.zeros(n_bands, n_freqs)
    for i in range(n_bands):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        for j in range(left, center):
            if center > left:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                fb[i, j] = (right - j) / (right - center)
    return fb


class CochleagramEncoder:
    """Biologically-inspired cochleagram audio encoder.

    Converts audio waveform chunks into sparse binary spike patterns
    via mel-filterbank → log-compression → thresholding.

    Args:
        n_bands: Number of mel-frequency bands (default 64).
        n_fft: FFT window size (default 512).
        sample_rate: Audio sample rate in Hz (default 16000).
        spike_threshold: Log-power threshold for spike generation (default 0.3).
        device: Torch device.
    """

    def __init__(
        self,
        n_bands: int = 64,
        n_fft: int = 512,
        sample_rate: int = 16000,
        spike_threshold: float = 0.3,
        device: torch.device | None = None,
    ) -> None:
        self.n_bands = int(n_bands)
        self.n_fft = int(n_fft)
        self.sample_rate = int(sample_rate)
        self.spike_threshold = float(spike_threshold)
        self.device = device or torch.device("cpu")

        self._filterbank = _mel_filterbank(self.n_bands, self.n_fft, self.sample_rate).to(self.device)
        # Running baseline for adaptive thresholding
        self._baseline: torch.Tensor = torch.zeros(self.n_bands, device=self.device)
        self._baseline_alpha: float = 0.05
        # Exponential trace for cross-modal STDP
        self._trace: torch.Tensor = torch.zeros(self.n_bands, device=self.device)
        self._trace_tau: float = 10.0

    @property
    def output_dim(self) -> int:
        return self.n_bands

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode an audio chunk into a binary spike pattern.

        Args:
            waveform: 1-D tensor of audio samples (mono, any length ≥ n_fft).

        Returns:
            Binary spike tensor of shape (n_bands,).
        """
        wav = waveform.to(self.device).float()
        if wav.dim() > 1:
            wav = wav.squeeze()

        # Pad if shorter than n_fft
        if wav.numel() < self.n_fft:
            wav = torch.nn.functional.pad(wav, (0, self.n_fft - wav.numel()))

        # Windowed FFT (Hann window)
        window = torch.hann_window(self.n_fft, device=self.device)
        # Take last n_fft samples
        chunk = wav[-self.n_fft:]
        spectrum = torch.fft.rfft(chunk * window)
        power = spectrum.abs() ** 2

        # Mel-filterbank application
        mel_power = torch.mv(self._filterbank, power)

        # Log compression (Weber-Fechner)
        log_power = torch.log1p(mel_power)

        # Adaptive baseline update
        self._baseline = (
            (1.0 - self._baseline_alpha) * self._baseline
            + self._baseline_alpha * log_power
        )

        # Spike when log-power exceeds baseline + threshold
        spikes = ((log_power - self._baseline) > self.spike_threshold).float()

        # Update trace
        decay = (-1.0 / self._trace_tau) if self._trace_tau > 0 else -1.0
        self._trace = self._trace * torch.exp(torch.tensor(decay, device=self.device)) + spikes

        return spikes

    @property
    def trace(self) -> torch.Tensor:
        """Current exponential trace (for cross-modal STDP)."""
        return self._trace.detach()

    def sparsity(self, spikes: torch.Tensor) -> float:
        """Fraction of active bands in a spike pattern."""
        if spikes.numel() == 0:
            return 0.0
        return float(spikes.sum().item() / spikes.numel())

    def reset(self) -> None:
        """Clear baseline and trace."""
        self._baseline = torch.zeros(self.n_bands, device=self.device)
        self._trace = torch.zeros(self.n_bands, device=self.device)

    def state_dict(self) -> dict[str, Any]:
        return {
            "n_bands": self.n_bands,
            "n_fft": self.n_fft,
            "sample_rate": self.sample_rate,
            "spike_threshold": self.spike_threshold,
            "baseline": self._baseline.cpu(),
            "trace": self._trace.cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        bl = state.get("baseline")
        if bl is not None:
            self._baseline = bl.to(self.device)
        tr = state.get("trace")
        if tr is not None:
            self._trace = tr.to(self.device)
