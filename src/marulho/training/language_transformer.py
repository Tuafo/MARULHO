"""MARULHO-owned causal Transformer state core with checkpointed KV state."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F


class TransformerRMSNorm(nn.Module):
    def __init__(self, width: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(width)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = value.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return value * scale * self.weight


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    even = value[..., ::2]
    odd = value[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def _apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = int(query.shape[-1])
    inverse_frequency = 1.0 / (
        10000.0
        ** (
            torch.arange(0, head_dim, 2, device=query.device, dtype=torch.float32)
            / float(head_dim)
        )
    )
    angles = positions.to(device=query.device, dtype=torch.float32).unsqueeze(-1)
    angles = angles * inverse_frequency.unsqueeze(0)
    cosine = torch.repeat_interleave(torch.cos(angles), 2, dim=-1).to(query.dtype)
    sine = torch.repeat_interleave(torch.sin(angles), 2, dim=-1).to(query.dtype)
    cosine = cosine.unsqueeze(0).unsqueeze(0)
    sine = sine.unsqueeze(0).unsqueeze(0)
    return (
        (query * cosine) + (_rotate_half(query) * sine),
        (key * cosine) + (_rotate_half(key) * sine),
    )


class MarulhoCausalSelfAttention(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        attention_heads: int,
        context_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.attention_heads = int(attention_heads)
        self.context_length = int(context_length)
        self.dropout = float(dropout)
        if self.width % self.attention_heads != 0:
            raise ValueError("Transformer width must be divisible by attention_heads")
        self.head_dim = self.width // self.attention_heads
        if self.head_dim % 2 != 0:
            raise ValueError("Transformer attention head dimension must be even for RoPE")
        self.qkv = nn.Linear(self.width, self.width * 3, bias=False)
        self.output = nn.Linear(self.width, self.width, bias=False)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, time, _ = value.shape
        return value.view(
            int(batch),
            int(time),
            self.attention_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        value: torch.Tensor,
        *,
        past_key: torch.Tensor | None,
        past_value: torch.Tensor | None,
        position_offset: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, time_steps, _ = value.shape
        query, key, current_value = self.qkv(value).chunk(3, dim=-1)
        query = self._heads(query)
        key = self._heads(key)
        current_value = self._heads(current_value)
        positions = torch.arange(int(time_steps), device=value.device)
        positions = positions + torch.as_tensor(position_offset, device=value.device)
        query, key = _apply_rotary(query, key, positions)

        usable_past_key: torch.Tensor | None = None
        usable_past_value: torch.Tensor | None = None
        if past_key is not None and past_value is not None and int(past_key.shape[2]) > 0:
            keep_past = max(0, self.context_length - int(time_steps))
            if keep_past > 0:
                usable_past_key = past_key[:, :, -keep_past:].to(
                    device=value.device,
                    dtype=value.dtype,
                )
                usable_past_value = past_value[:, :, -keep_past:].to(
                    device=value.device,
                    dtype=value.dtype,
                )
        if usable_past_key is None:
            full_key = key
            full_value = current_value
            past_length = 0
        else:
            full_key = torch.cat((usable_past_key, key), dim=2)
            full_value = torch.cat((usable_past_value, current_value), dim=2)
            past_length = int(usable_past_key.shape[2])

        if past_length == 0:
            attention = F.scaled_dot_product_attention(
                query,
                full_key,
                full_value,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        elif int(time_steps) == 1:
            attention = F.scaled_dot_product_attention(
                query,
                full_key,
                full_value,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            key_positions = torch.arange(
                int(full_key.shape[2]),
                device=value.device,
            ).unsqueeze(0)
            query_limits = (
                past_length
                + torch.arange(int(time_steps), device=value.device).unsqueeze(1)
            )
            causal_mask = key_positions <= query_limits
            attention = F.scaled_dot_product_attention(
                query,
                full_key,
                full_value,
                attn_mask=causal_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        attention = attention.transpose(1, 2).contiguous().view(
            int(batch_size),
            int(time_steps),
            self.width,
        )
        return self.output(attention), full_key, full_value


class MarulhoTransformerBlock(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        attention_heads: int,
        context_length: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden_width = max(int(width), int(round(float(width) * float(mlp_ratio))))
        self.attention_norm = TransformerRMSNorm(width)
        self.attention = MarulhoCausalSelfAttention(
            width,
            attention_heads=attention_heads,
            context_length=context_length,
            dropout=dropout,
        )
        self.mlp_norm = TransformerRMSNorm(width)
        self.gate_up = nn.Linear(width, hidden_width * 2, bias=False)
        self.down = nn.Linear(hidden_width, width, bias=False)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        value: torch.Tensor,
        *,
        past_key: torch.Tensor | None,
        past_value: torch.Tensor | None,
        position_offset: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention, next_key, next_value = self.attention(
            self.attention_norm(value),
            past_key=past_key,
            past_value=past_value,
            position_offset=position_offset,
        )
        value = value + self.dropout(attention)
        gate, up = self.gate_up(self.mlp_norm(value)).chunk(2, dim=-1)
        value = value + self.dropout(self.down(F.silu(gate) * up))
        return value, next_key, next_value


class MarulhoCausalTransformerStateBlock(nn.Module):
    """Decoder-only Transformer that satisfies MARULHO's streaming state contract."""

    surface = "marulho_causal_transformer_state_block.v1"

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        *,
        state_layers: int,
        attention_heads: int,
        context_length: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.state_layers = max(1, int(state_layers))
        self.attention_heads = max(1, int(attention_heads))
        self.context_length = max(2, int(context_length))
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        if not math.isfinite(self.mlp_ratio) or self.mlp_ratio < 1.0:
            raise ValueError("transformer_mlp_ratio must be finite and at least one")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("transformer_dropout must be in [0, 1)")
        self.input_projection: nn.Module = (
            nn.Identity()
            if self.input_dim == self.state_dim
            else nn.Linear(self.input_dim, self.state_dim, bias=False)
        )
        self.layers = nn.ModuleList(
            MarulhoTransformerBlock(
                self.state_dim,
                attention_heads=self.attention_heads,
                context_length=self.context_length,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
            )
            for _ in range(self.state_layers)
        )
        self.output_norm = TransformerRMSNorm(self.state_dim)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        head_dim = self.state_dim // self.attention_heads
        state: dict[str, torch.Tensor] = {
            "position": torch.zeros((), device=device, dtype=torch.long)
        }
        for layer_index in range(self.state_layers):
            state[f"layer_{layer_index}_key"] = torch.empty(
                int(batch_size),
                self.attention_heads,
                0,
                head_dim,
                device=device,
                dtype=dtype,
            )
            state[f"layer_{layer_index}_value"] = torch.empty_like(
                state[f"layer_{layer_index}_key"]
            )
        return state

    def _telemetry(
        self,
        *,
        device: torch.device,
        time_steps: int,
        cache_tokens: int,
        collected: bool,
    ) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "state_core": "transformer",
            "telemetry_collected": bool(collected),
            "spike_telemetry_available": False,
            "spike_rate": 0.0,
            "dead_neuron_fraction": 0.0,
            "over_firing_fraction": 0.0,
            "adaptive_timestep_budget": 1,
            "adaptive_step_count": int(time_steps),
            "state_dim": self.state_dim,
            "state_layers": self.state_layers,
            "attention_heads": self.attention_heads,
            "context_length": self.context_length,
            "kv_cache_tokens": int(cache_tokens),
            "time_steps": int(time_steps),
            "normalization": "rmsnorm",
            "position_encoding": "rotary",
            "attention_backend": "torch_scaled_dot_product_attention",
            "state_cache_keys": [
                "position",
                *[
                    key
                    for layer_index in range(self.state_layers)
                    for key in (
                        f"layer_{layer_index}_key",
                        f"layer_{layer_index}_value",
                    )
                ],
            ],
            "recurrent_gradient_horizon": 0,
            "truncated_bptt_applied": False,
            "truncated_bptt_boundary_count": 0,
            "gradient_horizon_policy": "causal_attention_context",
            "plif_state": "not_applicable_transformer",
            "plif_forward_backend": "not_applicable_transformer",
            "device": str(device),
        }

    def forward(
        self,
        inputs: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        if inputs.ndim != 3:
            raise ValueError("Transformer state block expects [batch, time, input_dim]")
        batch_size, time_steps, _ = inputs.shape
        if int(time_steps) > self.context_length and state is None:
            raise ValueError(
                "Transformer input sequence exceeds transformer_context_length"
            )
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
        next_state: dict[str, torch.Tensor] = {
            "position": position_offset + int(time_steps)
        }
        cache_tokens = 0
        for layer_index, layer in enumerate(self.layers):
            past_key = current_state.get(f"layer_{layer_index}_key")
            past_value = current_state.get(f"layer_{layer_index}_value")
            hidden, next_key, next_value = layer(
                hidden,
                past_key=past_key,
                past_value=past_value,
                position_offset=position_offset,
            )
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

    def step(
        self,
        token_input: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        if token_input.ndim != 2:
            raise ValueError("Transformer language state step expects [batch, input_dim]")
        hidden, next_state, telemetry = self.forward(
            token_input.unsqueeze(1),
            state,
            collect_telemetry=collect_telemetry,
        )
        return hidden[:, 0], next_state, telemetry
