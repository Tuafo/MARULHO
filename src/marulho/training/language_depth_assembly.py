"""Identity-initialized learned routing across Transformer depth."""

from __future__ import annotations

import math
from typing import Any, Mapping, TYPE_CHECKING

import torch
from torch import nn

from marulho.training.language_transformer import (
    MarulhoCausalTransformerStateBlock,
)

if TYPE_CHECKING:
    from marulho.training.language_model import MarulhoLanguageModel


class MarulhoDepthAssemblyStateBlock(MarulhoCausalTransformerStateBlock):
    """Let each block learn bounded corrections from earlier depth states.

    A zero route vector is exactly the ordinary residual chain. For block ``i``,
    the candidate can move its input toward any representation preceding the
    current one. The triangular parameterization gives every stored coefficient
    a live path and avoids unused upper-triangular parameters.
    """

    surface = "marulho_depth_assembly_state_block.v1"

    def __init__(self, base: MarulhoCausalTransformerStateBlock) -> None:
        nn.Module.__init__(self)
        self.input_dim = int(base.input_dim)
        self.state_dim = int(base.state_dim)
        self.state_layers = int(base.state_layers)
        self.attention_heads = int(base.attention_heads)
        self.context_length = int(base.context_length)
        self.mlp_ratio = float(base.mlp_ratio)
        self.dropout = float(base.dropout)
        self.input_projection = base.input_projection
        self.layers = base.layers
        self.output_norm = base.output_norm
        route_count = self.state_layers * (self.state_layers - 1) // 2
        self.depth_routes = nn.Parameter(torch.zeros(route_count))

    def _assembled_input(
        self,
        histories: list[torch.Tensor],
        *,
        layer_index: int,
    ) -> torch.Tensor:
        if int(layer_index) == 0:
            return histories[-1]
        current = histories[-1]
        start = int(layer_index) * (int(layer_index) - 1) // 2
        end = start + int(layer_index)
        coefficients = torch.tanh(self.depth_routes[start:end]).to(
            dtype=current.dtype
        ) / math.sqrt(float(layer_index))
        correction = torch.zeros_like(current)
        for coefficient, earlier in zip(coefficients, histories[:-1], strict=True):
            correction = correction + coefficient * (earlier - current)
        return current + correction

    def _telemetry(
        self,
        *,
        device: torch.device,
        time_steps: int,
        cache_tokens: int,
        collected: bool,
    ) -> dict[str, Any]:
        telemetry = super()._telemetry(
            device=device,
            time_steps=time_steps,
            cache_tokens=cache_tokens,
            collected=collected,
        )
        telemetry.update(
            {
                "surface": self.surface,
                "state_core": "depth_assembly_transformer",
                "depth_route_parameter_count": int(self.depth_routes.numel()),
                "depth_route_parameterization": "bounded_triangular_state_corrections",
                "depth_route_identity_initialized": True,
            }
        )
        return telemetry

    def forward(
        self,
        inputs: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        if inputs.ndim != 3:
            raise ValueError("Depth assembly expects [batch, time, input_dim]")
        batch_size, time_steps, _ = inputs.shape
        if int(time_steps) > self.context_length and state is None:
            raise ValueError("Input sequence exceeds transformer_context_length")
        current_state = (
            self.initial_state(
                int(batch_size),
                device=inputs.device,
                dtype=inputs.dtype,
            )
            if state is None
            else state
        )
        position_value = current_state.get("position")
        position_offset = (
            position_value.to(device=inputs.device, dtype=torch.long)
            if isinstance(position_value, torch.Tensor)
            else torch.zeros((), device=inputs.device, dtype=torch.long)
        )
        hidden = self.input_projection(inputs)
        histories = [hidden]
        next_state: dict[str, torch.Tensor] = {
            "position": position_offset + int(time_steps)
        }
        cache_tokens = 0
        for layer_index, layer in enumerate(self.layers):
            layer_input = self._assembled_input(
                histories,
                layer_index=layer_index,
            )
            past_key = current_state.get(f"layer_{layer_index}_key")
            past_value = current_state.get(f"layer_{layer_index}_value")
            hidden, next_key, next_value = layer(
                layer_input,
                past_key=past_key,
                past_value=past_value,
                position_offset=position_offset,
            )
            histories.append(hidden)
            next_state[f"layer_{layer_index}_key"] = next_key.detach()
            next_state[f"layer_{layer_index}_value"] = next_value.detach()
            cache_tokens = int(next_key.shape[2])
        hidden = self.output_norm(hidden)
        return (
            hidden,
            next_state,
            self._telemetry(
                device=inputs.device,
                time_steps=int(time_steps),
                cache_tokens=cache_tokens,
                collected=collect_telemetry,
            ),
        )

    def route_report(self) -> dict[str, Any]:
        values = torch.tanh(self.depth_routes.detach().float().cpu())
        return {
            "surface": "marulho_depth_assembly_routes.v1",
            "parameter_count": int(values.numel()),
            "nonzero_parameter_count": int(torch.count_nonzero(values)),
            "maximum_absolute_route": (
                float(values.abs().max()) if int(values.numel()) else 0.0
            ),
            "mean_absolute_route": (
                float(values.abs().mean()) if int(values.numel()) else 0.0
            ),
            "route_values": values.tolist(),
            "external_llm_used": False,
            "owned_by_marulho": True,
        }


def install_depth_assembly(
    model: MarulhoLanguageModel,
) -> MarulhoDepthAssemblyStateBlock:
    """Replace the live state block while preserving its exact base tensors."""

    if isinstance(model.state_block, MarulhoDepthAssemblyStateBlock):
        return model.state_block
    if not isinstance(model.state_block, MarulhoCausalTransformerStateBlock):
        raise TypeError("Depth assembly requires the causal Transformer state block")
    assembled = MarulhoDepthAssemblyStateBlock(model.state_block)
    model.state_block = assembled
    return assembled
