"""Experimental MARULHO delta-state/local-attention causal language model.

The recurrent operator is owned PyTorch code.  Its chunkwise training form and
tokenwise streaming form implement the same asymmetric delta-state equation;
no external model, weights, runtime kernel, or language generator is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from marulho.training.language_model import MarulhoLanguageModel
from marulho.training.language_transformer import TransformerRMSNorm, _apply_rotary


@dataclass(frozen=True)
class DeltaStateLanguageModelConfig:
    vocab_size: int = 8192
    embedding_dim: int = 640
    state_dim: int = 640
    state_layers: int = 12
    attention_heads: int = 10
    transformer_context_length: int = 320
    transformer_mlp_ratio: float = 4.1
    transformer_dropout: float = 0.0
    tie_embeddings: bool = True
    active_language_path: str = "marulho_delta_state_cortex_v64"
    state_core: str = "delta_state_cortex"
    local_attention_window: int = 64
    delta_chunk_size: int = 32
    delta_layers_per_cell: int = 3


def _validate_delta_config(config: DeltaStateLanguageModelConfig) -> None:
    if int(config.vocab_size) <= 1:
        raise ValueError("vocab_size must be greater than one")
    if int(config.embedding_dim) != int(config.state_dim):
        raise ValueError("V64 requires equal embedding and state dimensions")
    if int(config.state_dim) % int(config.attention_heads) != 0:
        raise ValueError("state_dim must be divisible by attention_heads")
    head_dim = int(config.state_dim) // int(config.attention_heads)
    if head_dim % 2:
        raise ValueError("head dimension must be even")
    if int(config.state_layers) < 4:
        raise ValueError("V64 requires at least one complete hybrid cell")
    if int(config.delta_layers_per_cell) < 1:
        raise ValueError("delta_layers_per_cell must be positive")
    if int(config.local_attention_window) < 2:
        raise ValueError("local_attention_window must be at least two")
    if int(config.delta_chunk_size) < 1:
        raise ValueError("delta_chunk_size must be positive")
    if not math.isfinite(float(config.transformer_mlp_ratio)):
        raise ValueError("MLP ratio must be finite")


def gated_delta_state_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    log_decay: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenwise reference for Gated Delta Rule-2 in state [key,value] form."""

    tensors = (query, key, value, erase, write, log_decay)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("delta-state tensors must be [batch,heads,time,channels]")
    if any(tensor.shape != query.shape for tensor in tensors[1:]):
        raise ValueError("delta-state q/k/v/gate/decay shapes must match")
    batch, heads, time, channels = query.shape
    state = (
        torch.zeros(
            int(batch),
            int(heads),
            int(channels),
            int(channels),
            device=query.device,
            dtype=torch.float32,
        )
        if initial_state is None
        else initial_state.to(device=query.device, dtype=torch.float32)
    )
    outputs: list[torch.Tensor] = []
    for index in range(int(time)):
        q = query[:, :, index].float()
        k = key[:, :, index].float()
        v = value[:, :, index].float()
        e = erase[:, :, index].float() * k
        z = write[:, :, index].float() * v
        state = torch.exp(log_decay[:, :, index].float()).unsqueeze(-1) * state
        old = torch.einsum("bhkv,bhk->bhv", state, e)
        state = state + torch.einsum("bhk,bhv->bhkv", k, z - old)
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, q))
    output = torch.stack(outputs, dim=2) if outputs else value.float()
    return output, state


def gated_delta_state_chunkwise(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    log_decay: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autograd-capable compact-WY chunk form with FP32 state and solve."""

    tensors = (query, key, value, erase, write, log_decay)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("delta-state tensors must be [batch,heads,time,channels]")
    if any(tensor.shape != query.shape for tensor in tensors[1:]):
        raise ValueError("delta-state q/k/v/gate/decay shapes must match")
    batch, heads, time, channels = query.shape
    size = max(1, int(chunk_size))
    state = (
        torch.zeros(
            int(batch),
            int(heads),
            int(channels),
            int(channels),
            device=query.device,
            dtype=torch.float32,
        )
        if initial_state is None
        else initial_state.to(device=query.device, dtype=torch.float32)
    )
    output_chunks: list[torch.Tensor] = []
    for start in range(0, int(time), size):
        stop = min(int(time), start + size)
        q = query[:, :, start:stop].float()
        k = key[:, :, start:stop].float()
        v = value[:, :, start:stop].float()
        b = erase[:, :, start:stop].float()
        w = write[:, :, start:stop].float()
        g = log_decay[:, :, start:stop].float()

        gamma = torch.exp(torch.cumsum(g, dim=2))
        normalized_key = k / gamma
        normalized_erase = gamma * b * k
        gated_value = w * v
        lower = torch.matmul(normalized_erase, normalized_key.transpose(-1, -2))
        lower = torch.tril(lower, diagonal=-1)
        identity = torch.eye(
            stop - start,
            device=query.device,
            dtype=torch.float32,
        )
        rhs = gated_value - torch.matmul(normalized_erase, state)
        residual = torch.linalg.solve_triangular(
            identity + lower,
            rhs,
            upper=False,
            unitriangular=True,
        )

        query_gamma = gamma * q
        causal_scores = torch.matmul(query_gamma, normalized_key.transpose(-1, -2))
        causal_scores = torch.tril(causal_scores)
        output_chunks.append(
            torch.matmul(query_gamma, state) + torch.matmul(causal_scores, residual)
        )

        final_gamma = gamma[:, :, -1]
        tail_key = (final_gamma.unsqueeze(2) / gamma) * k
        state = final_gamma.unsqueeze(-1) * state + torch.matmul(
            tail_key.transpose(-1, -2), residual
        )
    output = torch.cat(output_chunks, dim=2) if output_chunks else value.float()
    return output, state


class MarulhoLocalCausalAttention(nn.Module):
    def __init__(self, width: int, *, heads: int, window: int) -> None:
        super().__init__()
        self.width = int(width)
        self.heads = int(heads)
        self.window = int(window)
        self.head_dim = self.width // self.heads
        self.qkv = nn.Linear(self.width, self.width * 3, bias=False)
        self.output = nn.Linear(self.width, self.width, bias=False)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, time, _ = value.shape
        return value.view(batch, time, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        value: torch.Tensor,
        *,
        past_key: torch.Tensor | None,
        past_value: torch.Tensor | None,
        position_offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, time, _ = value.shape
        query, key, current_value = self.qkv(value).chunk(3, dim=-1)
        query, key, current_value = map(self._heads, (query, key, current_value))
        positions = torch.arange(int(time), device=value.device) + position_offset
        query, key = _apply_rotary(query, key, positions)
        if past_key is not None and int(past_key.shape[2]) > 0:
            keep = max(0, self.window - 1)
            usable_key = past_key[:, :, -keep:].to(
                device=value.device, dtype=value.dtype
            )
            usable_value = past_value[:, :, -keep:].to(
                device=value.device, dtype=value.dtype
            )
            full_key = torch.cat((usable_key, key), dim=2)
            full_value = torch.cat((usable_value, current_value), dim=2)
            past = int(usable_key.shape[2])
        else:
            full_key = key
            full_value = current_value
            past = 0
        key_positions = torch.arange(int(full_key.shape[2]), device=value.device)
        query_positions = past + torch.arange(int(time), device=value.device)
        difference = query_positions.unsqueeze(1) - key_positions.unsqueeze(0)
        mask = (difference >= 0) & (difference < self.window)
        attention = F.scaled_dot_product_attention(
            query,
            full_key,
            full_value,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attention = attention.transpose(1, 2).contiguous().view(batch, time, self.width)
        return (
            self.output(attention),
            full_key[:, :, -self.window :],
            full_value[:, :, -self.window :],
        )


class MarulhoDeltaStateMixer(nn.Module):
    def __init__(self, width: int, *, heads: int, chunk_size: int) -> None:
        super().__init__()
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = self.width // self.heads
        self.chunk_size = int(chunk_size)
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.qkv_conv = nn.Conv1d(
            width * 3,
            width * 3,
            kernel_size=4,
            groups=width * 3,
            bias=False,
        )
        self.erase = nn.Linear(width, width, bias=False)
        self.write = nn.Linear(width, width, bias=False)
        self.decay = nn.Linear(width, width, bias=False)
        self.output_gate = nn.Linear(width, width, bias=False)
        self.output_norm = TransformerRMSNorm(width)
        self.output = nn.Linear(width, width, bias=False)
        self.log_decay_rate = nn.Parameter(torch.zeros(heads))
        self.decay_bias = nn.Parameter(torch.zeros(width))

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, time, _ = value.shape
        return value.view(batch, time, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        value: torch.Tensor,
        *,
        state: torch.Tensor | None,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self.qkv(value)
        if conv_state is None or int(conv_state.shape[-1]) == 0:
            convolution_input = F.pad(projected.transpose(1, 2), (3, 0))
        else:
            history = conv_state.to(device=value.device, dtype=value.dtype)
            convolution_input = torch.cat((history, projected.transpose(1, 2)), dim=2)
        convolved = self.qkv_conv(convolution_input).transpose(1, 2)
        query, key, candidate_value = F.silu(convolved).chunk(3, dim=-1)
        query = F.normalize(self._heads(query).float(), dim=-1)
        key = F.normalize(self._heads(key).float(), dim=-1)
        candidate_value = self._heads(candidate_value)
        erase = torch.sigmoid(self._heads(self.erase(value)))
        write = torch.sigmoid(self._heads(self.write(value)))
        decay_logits = self._heads(self.decay(value)).float()
        rate = torch.exp(self.log_decay_rate.float()).view(1, self.heads, 1, 1)
        bias = self.decay_bias.float().view(self.heads, self.head_dim)
        log_decay = -rate * F.softplus(decay_logits + bias.view(1, self.heads, 1, -1))
        mixed, next_state = gated_delta_state_chunkwise(
            query,
            key,
            candidate_value,
            erase,
            write,
            log_decay,
            state,
            chunk_size=self.chunk_size,
        )
        mixed = mixed.transpose(1, 2).contiguous().view(value.shape)
        gated = self.output_norm(mixed.to(dtype=value.dtype)) * F.silu(
            self.output_gate(value)
        )
        next_conv = convolution_input[:, :, -3:]
        return self.output(gated), next_state, next_conv


class MarulhoDeltaStateBlock(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        heads: int,
        chunk_size: int,
        hidden_width: int,
    ) -> None:
        super().__init__()
        self.mixer_norm = TransformerRMSNorm(width)
        self.mixer = MarulhoDeltaStateMixer(width, heads=heads, chunk_size=chunk_size)
        self.mlp_norm = TransformerRMSNorm(width)
        self.gate_up = nn.Linear(width, hidden_width * 2, bias=False)
        self.down = nn.Linear(hidden_width, width, bias=False)

    def forward(
        self,
        value: torch.Tensor,
        *,
        state: torch.Tensor | None,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed, next_state, next_conv = self.mixer(
            self.mixer_norm(value), state=state, conv_state=conv_state
        )
        value = value + mixed
        gate, up = self.gate_up(self.mlp_norm(value)).chunk(2, dim=-1)
        return value + self.down(F.silu(gate) * up), next_state, next_conv


class MarulhoLocalAttentionBlock(nn.Module):
    def __init__(
        self, width: int, *, heads: int, window: int, hidden_width: int
    ) -> None:
        super().__init__()
        self.mixer_norm = TransformerRMSNorm(width)
        self.mixer = MarulhoLocalCausalAttention(width, heads=heads, window=window)
        self.mlp_norm = TransformerRMSNorm(width)
        self.gate_up = nn.Linear(width, hidden_width * 2, bias=False)
        self.down = nn.Linear(hidden_width, width, bias=False)

    def forward(
        self,
        value: torch.Tensor,
        *,
        past_key: torch.Tensor | None,
        past_value: torch.Tensor | None,
        position_offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed, next_key, next_value = self.mixer(
            self.mixer_norm(value),
            past_key=past_key,
            past_value=past_value,
            position_offset=position_offset,
        )
        value = value + mixed
        gate, up = self.gate_up(self.mlp_norm(value)).chunk(2, dim=-1)
        return value + self.down(F.silu(gate) * up), next_key, next_value


class MarulhoDeltaStateCortex(nn.Module):
    surface = "marulho_delta_state_cortex.v1"

    def __init__(self, config: DeltaStateLanguageModelConfig) -> None:
        super().__init__()
        self.config = config
        width = int(config.state_dim)
        hidden_width = int(round(width * float(config.transformer_mlp_ratio)))
        cell = int(config.delta_layers_per_cell) + 1
        layers: list[nn.Module] = []
        self.layer_kinds: list[str] = []
        for index in range(int(config.state_layers)):
            if index % cell == cell - 1:
                layers.append(
                    MarulhoLocalAttentionBlock(
                        width,
                        heads=int(config.attention_heads),
                        window=int(config.local_attention_window),
                        hidden_width=hidden_width,
                    )
                )
                self.layer_kinds.append("local_attention")
            else:
                layers.append(
                    MarulhoDeltaStateBlock(
                        width,
                        heads=int(config.attention_heads),
                        chunk_size=int(config.delta_chunk_size),
                        hidden_width=hidden_width,
                    )
                )
                self.layer_kinds.append("delta_state")
        self.layers = nn.ModuleList(layers)
        self.output_norm = TransformerRMSNorm(width)

    def initial_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {
            "position": torch.zeros((), device=device, dtype=torch.long)
        }
        head_dim = int(self.config.state_dim) // int(self.config.attention_heads)
        for index, kind in enumerate(self.layer_kinds):
            if kind == "delta_state":
                state[f"layer_{index}_matrix"] = torch.zeros(
                    batch_size,
                    int(self.config.attention_heads),
                    head_dim,
                    head_dim,
                    device=device,
                    dtype=torch.float32,
                )
                state[f"layer_{index}_conv"] = torch.empty(
                    batch_size,
                    int(self.config.state_dim) * 3,
                    0,
                    device=device,
                    dtype=dtype,
                )
            else:
                state[f"layer_{index}_key"] = torch.empty(
                    batch_size,
                    int(self.config.attention_heads),
                    0,
                    head_dim,
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
            raise ValueError("delta-state cortex expects [batch,time,width]")
        batch, time, _ = inputs.shape
        current = (
            self.initial_state(batch, device=inputs.device, dtype=inputs.dtype)
            if state is None
            else state
        )
        position = current["position"].to(device=inputs.device, dtype=torch.long)
        next_state: dict[str, torch.Tensor] = {"position": position + int(time)}
        hidden = inputs
        for index, (kind, layer) in enumerate(zip(self.layer_kinds, self.layers)):
            if kind == "delta_state":
                hidden, matrix, conv = layer(
                    hidden,
                    state=current.get(f"layer_{index}_matrix"),
                    conv_state=current.get(f"layer_{index}_conv"),
                )
                next_state[f"layer_{index}_matrix"] = matrix.detach()
                next_state[f"layer_{index}_conv"] = conv.detach()
            else:
                hidden, key, value = layer(
                    hidden,
                    past_key=current.get(f"layer_{index}_key"),
                    past_value=current.get(f"layer_{index}_value"),
                    position_offset=position,
                )
                next_state[f"layer_{index}_key"] = key.detach()
                next_state[f"layer_{index}_value"] = value.detach()
        telemetry = {
            "surface": self.surface,
            "state_core": "delta_state_cortex",
            "telemetry_collected": bool(collect_telemetry),
            "state_layers": len(self.layers),
            "delta_state_layers": self.layer_kinds.count("delta_state"),
            "local_attention_layers": self.layer_kinds.count("local_attention"),
            "attention_heads": int(self.config.attention_heads),
            "head_dim": int(self.config.state_dim) // int(self.config.attention_heads),
            "delta_chunk_size": int(self.config.delta_chunk_size),
            "local_attention_window": int(self.config.local_attention_window),
            "recurrent_state_dtype": "float32",
            "external_llm_used": False,
            "owned_by_marulho": True,
        }
        return self.output_norm(hidden), next_state, telemetry

    def step(
        self,
        token_input: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        hidden, next_state, telemetry = self.forward(
            token_input.unsqueeze(1), state, collect_telemetry=collect_telemetry
        )
        return hidden[:, 0], next_state, telemetry


class MarulhoDeltaStateLanguageModel(MarulhoLanguageModel):
    """Uninstalled V64 model reusing MARULHO's owned LM/generation protocol."""

    surface = "marulho_delta_state_language_model.v1"
    generation_surface = "marulho_delta_state_generation.v1"

    def __init__(self, config: DeltaStateLanguageModelConfig) -> None:
        nn.Module.__init__(self)
        _validate_delta_config(config)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.state_block = MarulhoDeltaStateCortex(config)
        self.lm_head = nn.Linear(config.state_dim, config.vocab_size, bias=False)
        if bool(config.tie_embeddings):
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        gain = 2.0**-2.5
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight, gain=gain)

    def generation_decode_policy(self, **kwargs: Any) -> dict[str, Any]:
        policy = super().generation_decode_policy(**kwargs)
        policy.update(
            {
                "surface": "marulho_delta_state_decode_policy.v1",
                "kv_cache": "three_bounded_local_kv_plus_nine_fixed_matrix_states",
                "delta_state_recurrent_streaming": True,
                "external_llm_used": False,
            }
        )
        return policy


def delta_state_parameter_report(
    model: MarulhoDeltaStateLanguageModel,
) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    recurrent = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".mixer." in name
        and "layers.3." not in name
        and "layers.7." not in name
        and "layers.11." not in name
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
        "recurrent_mixer_parameters": int(recurrent),
        "external_llm_used": False,
    }
