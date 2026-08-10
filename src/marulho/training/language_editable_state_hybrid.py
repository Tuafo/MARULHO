"""Parameter-matched local-attention and editable-state language candidate.

This module is deliberately isolated from the installed Transformer checkpoint
surface.  It tests a continuous hybrid state before any event controller is
allowed to decide whether that state executes.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel
from marulho.training.language_transformer import TransformerRMSNorm, _apply_rotary


class MarulhoLocalCausalAttention(nn.Module):
    """Causal attention with an exact bounded local window in train and decode."""

    def __init__(
        self,
        width: int,
        *,
        attention_heads: int,
        window: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.attention_heads = int(attention_heads)
        self.window = max(1, int(window))
        self.dropout = float(dropout)
        if self.width % self.attention_heads != 0:
            raise ValueError("hybrid width must be divisible by attention_heads")
        self.head_dim = self.width // self.attention_heads
        if self.head_dim % 2 != 0:
            raise ValueError("hybrid attention head dimension must be even")
        self.qkv = nn.Linear(self.width, self.width * 3, bias=False)
        self.output = nn.Linear(self.width, self.width, bias=False)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, time, _ = value.shape
        return value.view(
            int(batch), int(time), self.attention_heads, self.head_dim
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
        offset = torch.as_tensor(position_offset, device=value.device, dtype=torch.long)
        positions = torch.arange(int(time_steps), device=value.device) + offset
        query, key = _apply_rotary(query, key, positions)

        usable_past_key: torch.Tensor | None = None
        usable_past_value: torch.Tensor | None = None
        if past_key is not None and past_value is not None and int(past_key.shape[2]):
            usable_past_key = past_key[:, :, -self.window :].to(
                device=value.device, dtype=value.dtype
            )
            usable_past_value = past_value[:, :, -self.window :].to(
                device=value.device, dtype=value.dtype
            )
        if usable_past_key is None:
            full_key = key
            full_value = current_value
            past_length = 0
        else:
            full_key = torch.cat((usable_past_key, key), dim=2)
            full_value = torch.cat((usable_past_value, current_value), dim=2)
            past_length = int(usable_past_key.shape[2])

        key_positions = torch.arange(int(full_key.shape[2]), device=value.device)
        query_positions = past_length + torch.arange(
            int(time_steps), device=value.device
        )
        distance = query_positions.unsqueeze(1) - key_positions.unsqueeze(0)
        local_causal_mask = (distance >= 0) & (distance < self.window)
        attention = F.scaled_dot_product_attention(
            query,
            full_key,
            full_value,
            attn_mask=local_causal_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attention = attention.transpose(1, 2).contiguous().view(
            int(batch_size), int(time_steps), self.width
        )
        return (
            self.output(attention),
            full_key[:, :, -self.window :],
            full_value[:, :, -self.window :],
        )


class MarulhoLocalAttentionBlock(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        attention_heads: int,
        window: int,
        hidden_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = TransformerRMSNorm(width)
        self.attention = MarulhoLocalCausalAttention(
            width,
            attention_heads=attention_heads,
            window=window,
            dropout=dropout,
        )
        self.mlp_norm = TransformerRMSNorm(width)
        self.gate_up = nn.Linear(width, int(hidden_width) * 2, bias=False)
        self.down = nn.Linear(int(hidden_width), width, bias=False)
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


class MarulhoEditableMatrixStateBlock(nn.Module):
    """Chunk-parallel gated linear state with recurrent one-token decoding.

    The state is a per-head key/value matrix.  Key-channel decay and
    value-channel write gates are deliberately separate: erasing an address and
    committing a payload are not the same decision.  Training evaluates chunks
    in parallel; decoding performs the mathematically identical recurrent step.
    """

    def __init__(
        self,
        width: int,
        *,
        attention_heads: int,
        hidden_width: int,
        chunk_size: int,
        decay_scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.attention_heads = int(attention_heads)
        self.chunk_size = max(1, int(chunk_size))
        self.decay_scale = float(decay_scale)
        self.dropout = nn.Dropout(float(dropout))
        if self.width % self.attention_heads != 0:
            raise ValueError("hybrid width must be divisible by attention_heads")
        if self.width % 2 != 0:
            raise ValueError("hybrid width must be even")
        if not math.isfinite(self.decay_scale) or self.decay_scale <= 0.0:
            raise ValueError("matrix decay_scale must be positive and finite")
        self.value_head_dim = self.width // self.attention_heads
        self.key_width = self.width // 2
        if self.key_width % self.attention_heads != 0:
            raise ValueError("half hybrid width must be divisible by attention_heads")
        self.key_head_dim = self.key_width // self.attention_heads

        self.state_norm = TransformerRMSNorm(self.width)
        self.qk = nn.Linear(self.width, self.key_width * 2, bias=False)
        self.value = nn.Linear(self.width, self.width, bias=False)
        self.decay_write = nn.Linear(
            self.width, self.key_width + self.width, bias=False
        )
        self.output = nn.Linear(self.width, self.width, bias=False)
        self.write_bias = nn.Parameter(torch.zeros(self.width))
        self.residual_scale = nn.Parameter(torch.full((self.width,), 0.5))

        self.mlp_norm = TransformerRMSNorm(self.width)
        self.gate_up = nn.Linear(self.width, int(hidden_width) * 2, bias=False)
        self.down = nn.Linear(int(hidden_width), self.width, bias=False)

    def _split_key_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, time, _ = value.shape
        return value.view(
            int(batch), int(time), self.attention_heads, self.key_head_dim
        )

    def _split_value_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, time, _ = value.shape
        return value.view(
            int(batch), int(time), self.attention_heads, self.value_head_dim
        )

    @staticmethod
    def _positive_features(value: torch.Tensor) -> torch.Tensor:
        positive = F.elu(value.float()) + 1.0
        return positive / positive.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)

    def _parallel_chunk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        decay: torch.Tensor,
        write: torch.Tensor,
        matrix_state: torch.Tensor,
        mass_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [batch, time, heads, channels] -> [batch, heads, time, channels]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        decay = decay.transpose(1, 2)
        write = write.transpose(1, 2)
        # For y_t = a_t * y_(t-1) + b_t, the exact parallel form is
        # P_t * (y_0 + cumsum(b_t / P_t)), where P_t = cumprod(a_t).
        # Chunks keep P_t far from underflow while avoiding a quadratic
        # token-pair tensor or a Python loop over individual tokens.
        prefix_decay = torch.cumprod(decay.float(), dim=2).clamp_min(1.0e-12)
        updates = key.unsqueeze(-1) * (write * value).unsqueeze(-2)
        states = matrix_state.unsqueeze(2) + torch.cumsum(
            updates / prefix_decay.unsqueeze(-1), dim=2
        )
        states = prefix_decay.unsqueeze(-1) * states
        masses = mass_state.unsqueeze(2) + torch.cumsum(
            key / prefix_decay, dim=2
        )
        masses = prefix_decay * masses
        numerator = torch.einsum("bhtk,bhtkv->bhtv", query, states)
        denominator = torch.einsum("bhtk,bhtk->bht", query, masses)
        output = numerator / denominator.clamp_min(1.0e-4).unsqueeze(-1)
        return output.transpose(1, 2), states[:, :, -1], masses[:, :, -1]

    def forward(
        self,
        value: torch.Tensor,
        *,
        matrix_state: torch.Tensor,
        mass_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        normalized = self.state_norm(value)
        query, key = self.qk(normalized).chunk(2, dim=-1)
        projected_value = self.value(normalized)
        decay_logits, write_logits = torch.split(
            self.decay_write(normalized), [self.key_width, self.width], dim=-1
        )
        query = self._positive_features(self._split_key_heads(query))
        key = self._positive_features(self._split_key_heads(key))
        projected_value = self._split_value_heads(projected_value).float()
        decay = torch.exp(
            -F.softplus(self._split_key_heads(decay_logits).float())
            / self.decay_scale
        ).clamp(min=1.0e-4, max=1.0)
        write = torch.sigmoid(
            self._split_value_heads(write_logits)
            + self.write_bias.view(
                1, 1, self.attention_heads, self.value_head_dim
            )
        ).float()

        outputs: list[torch.Tensor] = []
        next_matrix = matrix_state.float()
        next_mass = mass_state.float()
        for start in range(0, int(value.shape[1]), self.chunk_size):
            stop = min(int(value.shape[1]), start + self.chunk_size)
            chunk_output, next_matrix, next_mass = self._parallel_chunk(
                query[:, start:stop],
                key[:, start:stop],
                projected_value[:, start:stop],
                decay[:, start:stop],
                write[:, start:stop],
                next_matrix,
                next_mass,
            )
            outputs.append(chunk_output)
        state_output = torch.cat(outputs, dim=1).reshape(
            int(value.shape[0]), int(value.shape[1]), self.width
        )
        state_output = self.output(state_output.to(dtype=value.dtype))
        value = value + self.dropout(state_output * self.residual_scale)
        gate, up = self.gate_up(self.mlp_norm(value)).chunk(2, dim=-1)
        value = value + self.dropout(self.down(F.silu(gate) * up))
        diagnostics = {
            "mean_decay": decay.detach().mean(),
            "mean_write": write.detach().mean(),
            "matrix_norm": next_matrix.detach().norm(dim=(-2, -1)).mean(),
        }
        return value, next_matrix, next_mass, diagnostics


class MarulhoEditableStateHybridCore(nn.Module):
    """Alternating exact local attention and bounded editable matrix state."""

    surface = "marulho_editable_state_hybrid_core.v1"

    def __init__(
        self,
        width: int,
        *,
        layers: int,
        attention_heads: int,
        context_length: int,
        local_attention_window: int,
        matrix_chunk_size: int,
        mlp_ratio: float,
        decay_scale: float,
        dropout: float,
        matrix_layer_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.layers_count = int(layers)
        self.attention_heads = int(attention_heads)
        self.context_length = int(context_length)
        self.local_attention_window = int(local_attention_window)
        self.matrix_chunk_size = int(matrix_chunk_size)
        self.dropout = float(dropout)
        if self.layers_count < 2:
            raise ValueError("editable hybrid requires at least two layers")
        if not 1 <= self.local_attention_window <= self.context_length:
            raise ValueError("local attention window must be inside context length")
        if (self.width + 4) % 6 != 0:
            raise ValueError("hybrid width must satisfy exact parameter-match arithmetic")
        dense_hidden = max(self.width, int(round(self.width * float(mlp_ratio))))
        matrix_hidden = dense_hidden - ((self.width + 4) // 6)
        if matrix_hidden < self.width:
            raise ValueError("hybrid matrix MLP hidden width would be too small")
        selected = (
            tuple(int(index) for index in matrix_layer_indices)
            if matrix_layer_indices is not None
            else tuple(index for index in range(self.layers_count) if index % 2 == 1)
        )
        if not selected or any(index < 0 or index >= self.layers_count for index in selected):
            raise ValueError("matrix_layer_indices must select valid hybrid layers")
        self.matrix_layer_indices = selected
        blocks: list[nn.Module] = []
        for index in range(self.layers_count):
            if index in self.matrix_layer_indices:
                blocks.append(
                    MarulhoEditableMatrixStateBlock(
                        self.width,
                        attention_heads=self.attention_heads,
                        hidden_width=matrix_hidden,
                        chunk_size=self.matrix_chunk_size,
                        decay_scale=decay_scale,
                        dropout=self.dropout,
                    )
                )
            else:
                blocks.append(
                    MarulhoLocalAttentionBlock(
                        self.width,
                        attention_heads=self.attention_heads,
                        window=self.local_attention_window,
                        hidden_width=dense_hidden,
                        dropout=self.dropout,
                    )
                )
        self.layers = nn.ModuleList(blocks)
        self.output_norm = TransformerRMSNorm(self.width)

    def initial_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {
            "position": torch.zeros((), device=device, dtype=torch.long)
        }
        value_head_dim = self.width // self.attention_heads
        key_head_dim = (self.width // 2) // self.attention_heads
        for index, layer in enumerate(self.layers):
            if isinstance(layer, MarulhoEditableMatrixStateBlock):
                state[f"layer_{index}_matrix"] = torch.zeros(
                    int(batch_size),
                    self.attention_heads,
                    key_head_dim,
                    value_head_dim,
                    device=device,
                    dtype=torch.float32,
                )
                state[f"layer_{index}_mass"] = torch.zeros(
                    int(batch_size),
                    self.attention_heads,
                    key_head_dim,
                    device=device,
                    dtype=torch.float32,
                )
            else:
                state[f"layer_{index}_key"] = torch.empty(
                    int(batch_size),
                    self.attention_heads,
                    0,
                    value_head_dim,
                    device=device,
                    dtype=dtype,
                )
                state[f"layer_{index}_value"] = torch.empty_like(
                    state[f"layer_{index}_key"]
                )
        return state

    def forward(
        self,
        inputs: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        if inputs.ndim != 3:
            raise ValueError("editable hybrid expects [batch, time, width]")
        if int(inputs.shape[-1]) != self.width:
            raise ValueError("editable hybrid input width is incorrect")
        if int(inputs.shape[1]) > self.context_length and state is None:
            raise ValueError("editable hybrid input exceeds its training context")
        current = (
            self.initial_state(
                int(inputs.shape[0]), device=inputs.device, dtype=inputs.dtype
            )
            if state is None
            else state
        )
        position = current.get("position")
        position_offset = (
            position.to(device=inputs.device, dtype=torch.long)
            if isinstance(position, torch.Tensor)
            else torch.zeros((), device=inputs.device, dtype=torch.long)
        )
        hidden = inputs
        next_state: dict[str, torch.Tensor] = {
            "position": position_offset + int(inputs.shape[1])
        }
        cache_tokens = 0
        diagnostic_rows: list[dict[str, torch.Tensor]] = []
        for index, layer in enumerate(self.layers):
            if isinstance(layer, MarulhoEditableMatrixStateBlock):
                hidden, matrix, mass, diagnostics = layer(
                    hidden,
                    matrix_state=current[f"layer_{index}_matrix"].to(inputs.device),
                    mass_state=current[f"layer_{index}_mass"].to(inputs.device),
                )
                next_state[f"layer_{index}_matrix"] = matrix.detach()
                next_state[f"layer_{index}_mass"] = mass.detach()
                diagnostic_rows.append(diagnostics)
            else:
                hidden, key, value = layer(
                    hidden,
                    past_key=current.get(f"layer_{index}_key"),
                    past_value=current.get(f"layer_{index}_value"),
                    position_offset=position_offset,
                )
                next_state[f"layer_{index}_key"] = key.detach()
                next_state[f"layer_{index}_value"] = value.detach()
                cache_tokens = max(cache_tokens, int(key.shape[2]))
        hidden = self.output_norm(hidden)
        matrix_elements = sum(
            int(value.numel())
            for key, value in next_state.items()
            if key.endswith("_matrix")
        )
        telemetry: dict[str, Any] = {
            "surface": self.surface,
            "state_core": "editable_state_local_attention_hybrid",
            "telemetry_collected": bool(collect_telemetry),
            "state_dim": self.width,
            "state_layers": self.layers_count,
            "attention_heads": self.attention_heads,
            "context_length": self.context_length,
            "local_attention_window": self.local_attention_window,
            "local_attention_layers": self.layers_count
            - len(self.matrix_layer_indices),
            "matrix_state_layers": len(self.matrix_layer_indices),
            "matrix_layer_indices": list(self.matrix_layer_indices),
            "matrix_chunk_size": self.matrix_chunk_size,
            "matrix_state_elements": matrix_elements,
            "kv_cache_tokens": cache_tokens,
            "parallel_training_form": "chunkwise_diagonal_affine_scan",
            "recurrent_decode_form": "constant_matrix_state_step",
            "event_control_enabled": False,
            "active_compute_fraction": 1.0,
            "external_llm_used": False,
        }
        if diagnostic_rows and collect_telemetry:
            telemetry["mean_decay"] = torch.stack(
                [row["mean_decay"] for row in diagnostic_rows]
            ).mean()
            telemetry["mean_write"] = torch.stack(
                [row["mean_write"] for row in diagnostic_rows]
            ).mean()
            telemetry["matrix_norm"] = torch.stack(
                [row["matrix_norm"] for row in diagnostic_rows]
            ).mean()
        return hidden, next_state, telemetry

    def step(
        self,
        token_input: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        if token_input.ndim != 2:
            raise ValueError("editable hybrid step expects [batch, width]")
        hidden, next_state, telemetry = self.forward(
            token_input.unsqueeze(1), state, collect_telemetry=collect_telemetry
        )
        return hidden[:, 0], next_state, telemetry


class MarulhoEditableStateHybridLanguageModel(MarulhoLanguageModel):
    """Experimental protocol adapter; never accepted by the runtime loader."""

    surface = "marulho_editable_state_hybrid_language_model.v1"
    generation_surface = "marulho_editable_state_hybrid_generation.v1"

    def __init__(
        self,
        config: LanguageModelConfig,
        *,
        local_attention_window: int = 24,
        matrix_chunk_size: int = 18,
        matrix_decay_scale: float = 64.0,
        matrix_layer_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__(config)
        self.local_attention_window = int(local_attention_window)
        self.matrix_chunk_size = int(matrix_chunk_size)
        self.matrix_decay_scale = float(matrix_decay_scale)
        self.state_block = MarulhoEditableStateHybridCore(
            int(config.state_dim),
            layers=int(config.state_layers),
            attention_heads=int(config.attention_heads),
            context_length=int(config.transformer_context_length),
            local_attention_window=self.local_attention_window,
            matrix_chunk_size=self.matrix_chunk_size,
            mlp_ratio=float(config.transformer_mlp_ratio),
            decay_scale=self.matrix_decay_scale,
            dropout=float(config.transformer_dropout),
            matrix_layer_indices=matrix_layer_indices,
        )
        for module in self.state_block.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generation_decode_policy(self, **kwargs: Any) -> dict[str, Any]:
        policy = super().generation_decode_policy(**kwargs)
        return {
            **policy,
            "surface": "marulho_editable_state_hybrid_decode_policy.v1",
            "kv_cache": "bounded_local_attention_plus_constant_matrix_state",
            "local_attention_window": self.local_attention_window,
            "matrix_chunk_size": self.matrix_chunk_size,
        }

    def next_token_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        collect_telemetry: bool = True,
        return_evidence: bool = True,
    ) -> dict[str, Any]:
        result = super().next_token_loss(
            input_ids,
            target_ids,
            collect_telemetry=collect_telemetry,
            return_evidence=return_evidence,
        )
        if return_evidence:
            result["loss_evidence"] = {
                **result["loss_evidence"],
                "surface": "marulho_editable_state_hybrid_cross_entropy.v1",
            }
        return result
