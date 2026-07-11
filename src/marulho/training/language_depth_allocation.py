"""Experimental Transformer with an explicit per-layer feed-forward budget."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel
from marulho.training.language_transformer import MarulhoTransformerBlock


@dataclass(frozen=True)
class DepthAllocationConfig:
    """Configuration for an uninstalled depth-allocation falsification model."""

    vocab_size: int
    width: int = 512
    attention_heads: int = 8
    context_length: int = 72
    mlp_hidden_widths: tuple[int, ...] = (2048, 2048, 2048, 2048)
    dropout: float = 0.0
    initialization_seed: int = 1337
    active_language_path: str = "marulho_depth_allocation_v8"


def _validate_depth_config(config: DepthAllocationConfig) -> None:
    if int(config.vocab_size) <= 1:
        raise ValueError("vocab_size must be greater than one")
    if int(config.width) <= 0:
        raise ValueError("width must be positive")
    if int(config.attention_heads) <= 0:
        raise ValueError("attention_heads must be positive")
    if int(config.width) % int(config.attention_heads) != 0:
        raise ValueError("width must be divisible by attention_heads")
    if (int(config.width) // int(config.attention_heads)) % 2 != 0:
        raise ValueError("attention head width must be even for rotary positions")
    if int(config.context_length) < 2:
        raise ValueError("context_length must be at least two")
    if not config.mlp_hidden_widths:
        raise ValueError("mlp_hidden_widths must contain at least one layer")
    if any(int(hidden) < int(config.width) for hidden in config.mlp_hidden_widths):
        raise ValueError("every MLP hidden width must be at least the model width")
    if not math.isfinite(float(config.dropout)) or not 0.0 <= float(config.dropout) < 1.0:
        raise ValueError("dropout must be finite and in [0, 1)")


def _normal_parameter_(parameter: torch.Tensor, *, seed: int) -> None:
    generator = torch.Generator(device=parameter.device).manual_seed(int(seed))
    nn.init.normal_(parameter, mean=0.0, std=0.02, generator=generator)


class MarulhoDepthAllocatedLanguageModel(MarulhoLanguageModel):
    """Causal Transformer whose total MLP budget is redistributed across depth."""

    surface = "marulho_depth_allocated_language_model.v1"

    def __init__(self, depth_config: DepthAllocationConfig) -> None:
        _validate_depth_config(depth_config)
        self.depth_config = depth_config
        average_hidden = sum(depth_config.mlp_hidden_widths) / len(
            depth_config.mlp_hidden_widths
        )
        super().__init__(
            LanguageModelConfig(
                vocab_size=int(depth_config.vocab_size),
                embedding_dim=int(depth_config.width),
                state_dim=int(depth_config.width),
                state_layers=len(depth_config.mlp_hidden_widths),
                attention_heads=int(depth_config.attention_heads),
                transformer_context_length=int(depth_config.context_length),
                transformer_mlp_ratio=(average_hidden / float(depth_config.width)),
                transformer_dropout=float(depth_config.dropout),
                tie_embeddings=True,
                active_language_path=str(depth_config.active_language_path),
            )
        )

        original_layers = tuple(self.state_block.layers)
        replacement_layers: list[MarulhoTransformerBlock] = []
        for layer_index, (original, hidden_width) in enumerate(
            zip(original_layers, depth_config.mlp_hidden_widths, strict=True)
        ):
            replacement = MarulhoTransformerBlock(
                int(depth_config.width),
                attention_heads=int(depth_config.attention_heads),
                context_length=int(depth_config.context_length),
                mlp_ratio=(int(hidden_width) / float(depth_config.width)),
                dropout=float(depth_config.dropout),
            )
            replacement.attention_norm.load_state_dict(
                original.attention_norm.state_dict(), strict=True
            )
            replacement.attention.load_state_dict(
                original.attention.state_dict(), strict=True
            )
            replacement.mlp_norm.load_state_dict(
                original.mlp_norm.state_dict(), strict=True
            )
            seed = int(depth_config.initialization_seed) + (layer_index * 10_000)
            _normal_parameter_(replacement.gate_up.weight, seed=seed + 1)
            _normal_parameter_(replacement.down.weight, seed=seed + 2)
            replacement_layers.append(replacement)
        self.state_block.layers = nn.ModuleList(replacement_layers)

    @property
    def mlp_hidden_widths(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.depth_config.mlp_hidden_widths)

    def _allocation_telemetry(self) -> dict[str, Any]:
        widths = self.mlp_hidden_widths
        return {
            "surface": self.surface,
            "architecture": "depth_allocated_transformer",
            "mlp_hidden_widths": list(widths),
            "mlp_hidden_width_sum": sum(widths),
            "mlp_width_monotonic_direction": (
                "uniform"
                if len(set(widths)) == 1
                else "nondecreasing"
                if all(left <= right for left, right in zip(widths, widths[1:]))
                else "nonincreasing"
                if all(left >= right for left, right in zip(widths, widths[1:]))
                else "mixed"
            ),
            "external_llm_used": False,
            "owned_by_marulho": True,
        }

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> dict[str, Any]:
        result = super()._forward_hidden(
            input_ids,
            state,
            collect_telemetry=collect_telemetry,
        )
        result["telemetry"] = {
            **result["telemetry"],
            **self._allocation_telemetry(),
        }
        return result

    def forward_step(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        result = super().forward_step(
            input_ids,
            state,
            collect_telemetry=collect_telemetry,
            decode_vocab_only=decode_vocab_only,
        )
        result["telemetry"] = {
            **result["telemetry"],
            **self._allocation_telemetry(),
        }
        return result


def matching_common_parameter_names(
    model: MarulhoDepthAllocatedLanguageModel,
) -> tuple[str, ...]:
    """Names whose shape and role are invariant across allocation profiles."""

    excluded = (".gate_up.weight", ".down.weight")
    return tuple(
        name
        for name, _parameter in model.named_parameters()
        if not name.endswith(excluded)
    )


def total_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def mlp_parameter_count(width: int, hidden_widths: Sequence[int]) -> int:
    return sum(3 * int(width) * int(hidden) for hidden in hidden_widths)
