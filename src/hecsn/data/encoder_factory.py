"""Encoder factory for HECSN.

Centralizes encoder construction so all code paths use the same logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from hecsn.config.model_config import HECSNConfig
    from hecsn.data.base_encoder import BaseEncoder

logger = logging.getLogger(__name__)


def build_encoder(config: "HECSNConfig", device: torch.device | str | None = None) -> "BaseEncoder":
    """Build the appropriate encoder based on config.input_representation."""
    resolved_device = config.resolve_device() if device is None else torch.device(device)
    if config.input_representation == "semantic":
        from hecsn.data.semantic_encoder import SemanticEncoder

        encoder = SemanticEncoder.from_config(config, device=resolved_device)
        result = encoder.initialize_from_glove(
            source=config.semantic_glove_source,
            vocab_limit=config.semantic_glove_vocab_limit,
            ridge_alpha=config.semantic_ridge_alpha,
        )
        logger.info("Semantic encoder initialized: %s", result.get("source", "unknown"))
        return encoder

    from hecsn.data.rtf_encoder import RTFEncoder

    return RTFEncoder.from_config(config, device=resolved_device)
