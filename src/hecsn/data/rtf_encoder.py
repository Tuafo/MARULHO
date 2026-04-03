from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Literal
import torch

if TYPE_CHECKING:
    from hecsn.config.model_config import HECSNConfig


RepresentationMode = Literal["order_weighted_ascii", "unigram_ascii", "hashed_ngram"]


class RTFEncoder:
    """Rate-Temporal Fusion encoder.

    Produces:
    - order-sensitive routing vector [128] derived from latency coding
    - spike-time tensor [128, n_bursts_max] for temporal simulation
    """

    def __init__(
        self,
        t_max: float = 20.0,
        n_bursts_max: int = 5,
        window_size: int = 10,
        representation: RepresentationMode = "order_weighted_ascii",
        hashed_ngram_dim: int = 2048,
        hashed_ngram_min_n: int = 2,
        hashed_ngram_max_n: int = 3,
    ) -> None:
        self.t_max = float(t_max)
        self.n_bursts_max = int(n_bursts_max)
        self.window_size = int(window_size)
        self.t_spacing = self.t_max / max(1, self.window_size + 1)
        self.representation = representation
        self.hashed_ngram_dim = int(hashed_ngram_dim)
        self.hashed_ngram_min_n = int(hashed_ngram_min_n)
        self.hashed_ngram_max_n = int(hashed_ngram_max_n)

    @classmethod
    def from_config(cls, config: "HECSNConfig") -> "RTFEncoder":
        return cls(
            window_size=config.window_size,
            representation=config.input_representation,
            hashed_ngram_dim=config.hashed_ngram_dim,
            hashed_ngram_min_n=config.hashed_ngram_min_n,
            hashed_ngram_max_n=config.hashed_ngram_max_n,
        )

    @property
    def output_dim(self) -> int:
        return self.hashed_ngram_dim if self.representation == "hashed_ngram" else 128

    def character_window_to_pattern(self, chars: Iterable[int]) -> torch.Tensor:
        window: List[int] = list(chars)[-self.window_size :]
        pattern = torch.zeros(128, dtype=torch.float32)
        if not window:
            return pattern

        for c in window:
            if 0 <= c < 128:
                pattern[c] += 1.0

        return pattern / float(len(window))

    def _hash_ngram(self, ngram: Iterable[int]) -> int:
        hash_value = 2166136261
        for code in ngram:
            hash_value ^= int(code) + 1
            hash_value = (hash_value * 16777619) & 0xFFFFFFFF
        return int(hash_value % max(1, self.hashed_ngram_dim))

    def hashed_ngram_vector(self, chars: Iterable[int]) -> torch.Tensor:
        window: List[int] = [c for c in list(chars)[-self.window_size :] if 0 <= c < 128]
        vector = torch.zeros(self.hashed_ngram_dim, dtype=torch.float32)
        if not window:
            return vector

        min_n = self.hashed_ngram_min_n if len(window) >= self.hashed_ngram_min_n else 1
        max_n = min(self.hashed_ngram_max_n, len(window))
        for n in range(min_n, max_n + 1):
            for start in range(0, len(window) - n + 1):
                bucket = self._hash_ngram(window[start : start + n])
                vector[bucket] += 1.0

        return vector / (torch.norm(vector, p=2) + 1e-8)

    def routing_vector(self, chars: Iterable[int]) -> torch.Tensor:
        """Canonical routing representation used by Stage-0.

        This keeps dimensionality fixed at 128 while injecting order via
        latency-derived position weighting, so anagrams no longer collide.
        """
        window: List[int] = list(chars)[-self.window_size :]
        while len(window) < self.window_size:
            window.insert(0, 0)

        route = torch.zeros(128, dtype=torch.float32)
        for pos, c in enumerate(window):
            if 0 <= c < 128:
                latency = pos * self.t_spacing
                weight = max(0.0, (self.t_max - latency) / max(1e-8, self.t_max))
                route[c] += float(weight)

        return route / (torch.norm(route, p=2) + 1e-8)

    def feature_vector(self, chars: Iterable[int]) -> torch.Tensor:
        if self.representation == "order_weighted_ascii":
            return self.routing_vector(chars)
        if self.representation == "unigram_ascii":
            return self.character_window_to_pattern(chars)
        if self.representation == "hashed_ngram":
            return self.hashed_ngram_vector(chars)
        raise ValueError(f"Unsupported representation: {self.representation}")

    def encode(self, chars: Iterable[int], context_confidence: float) -> torch.Tensor:
        window: List[int] = list(chars)[-self.window_size :]
        while len(window) < self.window_size:
            window.insert(0, 0)

        spike_times = torch.full((128, self.n_bursts_max), -1.0, dtype=torch.float32)
        n_spikes = max(1, int(self.n_bursts_max * max(0.0, min(1.0, context_confidence))))

        for pos, c in enumerate(window):
            if 0 <= c < 128:
                first = pos * self.t_spacing
                for i in range(n_spikes):
                    spike_times[c, i] = first + i * 3.0

        return spike_times

    def spike_trace(
        self,
        chars: Iterable[int],
        context_confidence: float,
        *,
        tau: float | None = None,
        burst_decay: float = 0.85,
    ) -> torch.Tensor:
        spike_times = self.encode(chars, context_confidence=context_confidence)
        trace_tau = float(self.t_spacing if tau is None else tau)
        if trace_tau <= 0.0:
            raise ValueError("tau must be positive")
        if not 0.0 < float(burst_decay) <= 1.0:
            raise ValueError("burst_decay must be in (0, 1]")

        valid = spike_times >= 0.0
        latency_weights = torch.exp(-torch.clamp(spike_times, min=0.0) / trace_tau)
        burst_indices = torch.arange(self.n_bursts_max, dtype=torch.float32).unsqueeze(0)
        burst_weights = torch.pow(torch.full_like(burst_indices, float(burst_decay)), burst_indices)
        weighted = torch.where(valid, latency_weights * burst_weights, torch.zeros_like(spike_times))
        collapsed = weighted.sum(dim=1)
        total = float(collapsed.sum().item())
        if total <= 0.0:
            return collapsed
        return collapsed / (collapsed.sum() + 1e-8)
