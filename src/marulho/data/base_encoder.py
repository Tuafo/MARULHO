"""Base encoder protocol for MARULHO (Stage 1A).

Defines the interface that all encoders (RTFEncoder, future EventCameraEncoder,
CochleagramEncoder) must implement. The trainer and evaluation code depend only
on this protocol, never on a concrete encoder class.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Protocol, runtime_checkable

import torch


@runtime_checkable
class BaseEncoder(Protocol):
    """Protocol defining the encoder interface for MARULHO.

    All encoders must produce:
    - A fixed-dimension feature vector from raw input
    - An output_dim property declaring the vector dimensionality
    - Serialization via state_dict/load_state_dict
    """

    @property
    def output_dim(self) -> int:
        """Dimensionality of the feature vectors produced by this encoder."""
        ...

    def feature_vector(self, chars: Iterable[int]) -> torch.Tensor:
        """Produce a normalized feature vector from raw character codes.

        Args:
            chars: Iterable of integer character codes.

        Returns:
            Normalized torch.Tensor of shape (output_dim,).
        """
        ...

    def iter_char_patterns(
        self,
        chars: Iterable[str],
        window_size: int,
        *,
        learn: bool = False,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Iterate over character windows, yielding (display_text, feature_vector) pairs.

        Args:
            chars: Iterable of single characters.
            window_size: Size of the sliding window.
            learn: Whether to update internal learnable state (e.g., chunking).

        Yields:
            Tuples of (window_text, feature_vector).
        """
        ...

    def segment_text(self, text: str, *, learn: bool = False) -> list[str]:
        """Segment text into meaningful units.

        Args:
            text: Input text string.
            learn: Whether to update internal learnable state.

        Returns:
            List of text segments.
        """
        ...

    def spike_trace(
        self,
        chars: Iterable[int],
        context_confidence: float,
        *,
        tau: float | None = None,
        burst_decay: float = 0.85,
    ) -> torch.Tensor:
        """Produce a spike-trace vector for temporal processing.

        Args:
            chars: Iterable of character codes.
            context_confidence: Confidence level affecting burst count.
            tau: Time constant for exponential decay (default: encoder-specific).
            burst_decay: Decay factor per burst (default: 0.85).

        Returns:
            Normalized spike trace tensor of shape (output_dim,) or similar.
        """
        ...

    def state_dict(self) -> dict[str, Any]:
        """Serialize encoder state for checkpointing."""
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore encoder state from a checkpoint."""
        ...
