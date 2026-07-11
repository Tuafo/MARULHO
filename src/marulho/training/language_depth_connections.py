"""Causal Transformer experiment with controlled depth-weighted reuse."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel
from marulho.training.language_transformer import (
    MarulhoCausalTransformerStateBlock,
)


DEPTH_CONNECTION_MODES = (
    "identity",
    "fixed_mean",
    "fixed_random",
    "learned_unconstrained",
    "learned_simplex",
)


@dataclass(frozen=True)
class DepthConnectionConfig:
    vocab_size: int
    width: int = 512
    layers: int = 4
    attention_heads: int = 8
    hidden_width: int = 2048
    context_length: int = 72
    dropout: float = 0.0
    mode: str = "learned_unconstrained"
    random_seed: int = 1337
    simplex_identity_logit: float = 8.0
    active_language_path: str = "marulho_depth_connections_v9"


def _validate_config(config: DepthConnectionConfig) -> None:
    if int(config.vocab_size) <= 1:
        raise ValueError("vocab_size must be greater than one")
    if int(config.width) <= 0 or int(config.layers) <= 0:
        raise ValueError("width and layers must be positive")
    if int(config.attention_heads) <= 0:
        raise ValueError("attention_heads must be positive")
    if int(config.width) % int(config.attention_heads) != 0:
        raise ValueError("width must be divisible by attention_heads")
    if (int(config.width) // int(config.attention_heads)) % 2 != 0:
        raise ValueError("attention head width must be even for rotary positions")
    if int(config.hidden_width) < int(config.width):
        raise ValueError("hidden_width must be at least width")
    if int(config.context_length) < 2:
        raise ValueError("context_length must be at least two")
    if not math.isfinite(float(config.dropout)) or not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be finite and in [0, 1)")
    if config.mode not in DEPTH_CONNECTION_MODES:
        raise ValueError("Unknown depth connection mode")
    if (
        not math.isfinite(float(config.simplex_identity_logit))
        or float(config.simplex_identity_logit) <= 0.0
    ):
        raise ValueError("simplex_identity_logit must be finite and positive")


def _depth_parameter_count(layers: int) -> int:
    return sum(layer_index + 2 for layer_index in range(int(layers)))


class MarulhoDepthWeightedStateBlock(nn.Module):
    """Wrap maintained Transformer blocks with branchless depth mixing controls."""

    surface = "marulho_depth_weighted_state_block.v1"

    def __init__(
        self,
        base: MarulhoCausalTransformerStateBlock,
        *,
        mode: str,
        random_seed: int,
        simplex_identity_logit: float,
    ) -> None:
        super().__init__()
        if mode not in DEPTH_CONNECTION_MODES:
            raise ValueError("Unknown depth connection mode")
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
        self.simplex_identity_logit = float(simplex_identity_logit)
        self.raw_depth_weights = nn.Parameter(
            torch.zeros(_depth_parameter_count(self.state_layers))
        )
        generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
        random_rows: list[torch.Tensor] = []
        for layer_index in range(self.state_layers):
            random_rows.append(
                torch.softmax(torch.randn(layer_index + 2, generator=generator), dim=0)
            )
        self.register_buffer(
            "fixed_random_depth_weights",
            torch.cat(random_rows),
            persistent=True,
        )
        for name in DEPTH_CONNECTION_MODES:
            self.register_buffer(
                f"mode_{name}",
                torch.tensor(1.0 if name == mode else 0.0),
                persistent=True,
            )
        self._mode_name = str(mode)

    def set_mode(self, mode: str) -> None:
        if mode not in DEPTH_CONNECTION_MODES:
            raise ValueError("Unknown depth connection mode")
        for name in DEPTH_CONNECTION_MODES:
            getattr(self, f"mode_{name}").fill_(1.0 if name == mode else 0.0)
        self._mode_name = str(mode)

    def _row_slice(self, layer_index: int) -> slice:
        start = sum(index + 2 for index in range(int(layer_index)))
        return slice(start, start + int(layer_index) + 2)

    def _row_weights(
        self,
        layer_index: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        row_slice = self._row_slice(layer_index)
        raw = self.raw_depth_weights[row_slice].to(device=device, dtype=dtype)
        identity = torch.zeros_like(raw)
        identity[-1] = 1.0
        fixed_mean = torch.ones_like(raw) / float(int(raw.numel()))
        fixed_random = self.fixed_random_depth_weights[row_slice].to(
            device=device,
            dtype=dtype,
        )
        learned_unconstrained = identity + raw
        simplex_logits = raw + (identity * self.simplex_identity_logit)
        learned_simplex = torch.softmax(simplex_logits.float(), dim=0).to(dtype)
        return (
            self.mode_identity.to(device=device, dtype=dtype) * identity
            + self.mode_fixed_mean.to(device=device, dtype=dtype) * fixed_mean
            + self.mode_fixed_random.to(device=device, dtype=dtype) * fixed_random
            + self.mode_learned_unconstrained.to(device=device, dtype=dtype)
            * learned_unconstrained
            + self.mode_learned_simplex.to(device=device, dtype=dtype)
            * learned_simplex
        )

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
            "state_core": "transformer_with_depth_weighted_reuse",
            "architecture": "depth_weighted_transformer",
            "depth_connection_mode": self._mode_name,
            "depth_connection_parameter_count": int(
                self.raw_depth_weights.numel()
            ),
            "depth_connection_rows": self.state_layers,
            "telemetry_collected": bool(collected),
            "state_dim": self.state_dim,
            "state_layers": self.state_layers,
            "attention_heads": self.attention_heads,
            "context_length": self.context_length,
            "kv_cache_tokens": int(cache_tokens),
            "time_steps": int(time_steps),
            "normalization": "rmsnorm",
            "position_encoding": "rotary",
            "attention_backend": "torch_scaled_dot_product_attention",
            "gradient_horizon_policy": "causal_attention_context",
            "device": str(device),
            "external_llm_used": False,
            "owned_by_marulho": True,
        }

    def _forward_core(
        self,
        inputs: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None,
        *,
        collect_telemetry: bool,
        return_depth_history: bool,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, Any],
        tuple[torch.Tensor, ...],
    ]:
        if inputs.ndim != 3:
            raise ValueError("Depth-weighted state block expects [batch, time, input_dim]")
        batch_size, time_steps, _ = inputs.shape
        if int(time_steps) > self.context_length and state is None:
            raise ValueError("Input sequence exceeds depth-weighted context length")
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
        history: list[torch.Tensor] = [hidden]
        next_state: dict[str, torch.Tensor] = {
            "position": position_offset + int(time_steps)
        }
        cache_tokens = 0
        for layer_index, layer in enumerate(self.layers):
            hidden, next_key, next_value = layer(
                hidden,
                past_key=current_state.get(f"layer_{layer_index}_key"),
                past_value=current_state.get(f"layer_{layer_index}_value"),
                position_offset=position_offset,
            )
            candidates = (*history, hidden)
            weights = self._row_weights(
                layer_index,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            hidden = torch.stack(candidates, dim=0).mul(
                weights.view(-1, 1, 1, 1)
            ).sum(dim=0)
            history.append(hidden)
            next_state[f"layer_{layer_index}_key"] = next_key.detach()
            next_state[f"layer_{layer_index}_value"] = next_value.detach()
            cache_tokens = int(next_key.shape[2])
        hidden = self.output_norm(hidden)
        depth_history = tuple(history) if return_depth_history else ()
        return (
            hidden,
            next_state,
            self._telemetry(
                device=inputs.device,
                time_steps=int(time_steps),
                cache_tokens=cache_tokens,
                collected=collect_telemetry,
            ),
            depth_history,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        hidden, next_state, telemetry, _history = self._forward_core(
            inputs,
            state,
            collect_telemetry=collect_telemetry,
            return_depth_history=False,
        )
        return hidden, next_state, telemetry

    def forward_with_depth_history(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        hidden, _state, _telemetry, history = self._forward_core(
            inputs,
            None,
            collect_telemetry=False,
            return_depth_history=True,
        )
        return hidden, history

    def step(
        self,
        token_input: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        if token_input.ndim != 2:
            raise ValueError("Depth-weighted step expects [batch, input_dim]")
        hidden, next_state, telemetry = self.forward(
            token_input.unsqueeze(1),
            state,
            collect_telemetry=collect_telemetry,
        )
        return hidden[:, 0], next_state, telemetry

    @torch.no_grad()
    def depth_weight_report(self) -> dict[str, Any]:
        rows = []
        for layer_index in range(self.state_layers):
            weights = self._row_weights(
                layer_index,
                device=self.raw_depth_weights.device,
                dtype=torch.float32,
            )
            probabilities = weights.abs() / weights.abs().sum().clamp_min(1.0e-12)
            entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
            rows.append(
                {
                    "layer": layer_index,
                    "weights": [float(value) for value in weights.cpu()],
                    "weight_sum": float(weights.sum().cpu()),
                    "absolute_weight_sum": float(weights.abs().sum().cpu()),
                    "negative_fraction": float((weights < 0).float().mean().cpu()),
                    "absolute_weight_entropy": float(entropy.cpu()),
                    "embedding_weight": float(weights[0].cpu()),
                    "diagonal_weight": float(weights[-1].cpu()),
                }
            )
        return {
            "surface": "marulho_depth_weight_report.v1",
            "mode": self._mode_name,
            "rows": rows,
            "external_llm_used": False,
        }


class MarulhoDepthConnectedLanguageModel(MarulhoLanguageModel):
    """MARULHO-owned language model used only by the V9 falsifier."""

    surface = "marulho_depth_connected_language_model.v1"

    def __init__(self, depth_config: DepthConnectionConfig) -> None:
        _validate_config(depth_config)
        self.depth_config = depth_config
        super().__init__(
            LanguageModelConfig(
                vocab_size=int(depth_config.vocab_size),
                embedding_dim=int(depth_config.width),
                state_dim=int(depth_config.width),
                state_layers=int(depth_config.layers),
                attention_heads=int(depth_config.attention_heads),
                transformer_context_length=int(depth_config.context_length),
                transformer_mlp_ratio=(
                    float(depth_config.hidden_width) / float(depth_config.width)
                ),
                transformer_dropout=float(depth_config.dropout),
                tie_embeddings=True,
                active_language_path=str(depth_config.active_language_path),
            )
        )
        self.state_block = MarulhoDepthWeightedStateBlock(
            self.state_block,
            mode=depth_config.mode,
            random_seed=depth_config.random_seed,
            simplex_identity_logit=depth_config.simplex_identity_logit,
        )

    def set_depth_connection_mode(self, mode: str) -> None:
        self.state_block.set_mode(mode)

    def depth_weight_report(self) -> dict[str, Any]:
        return self.state_block.depth_weight_report()

    @torch.no_grad()
    def depth_hidden_states(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        runtime_ids = input_ids.to(device=self.device, dtype=torch.long)
        _hidden, history = self.state_block.forward_with_depth_history(
            self.token_embedding(runtime_ids)
        )
        return history


@torch.no_grad()
def depth_geometry_report(
    model: MarulhoDepthConnectedLanguageModel,
    input_ids: torch.Tensor,
    *,
    max_samples: int = 4096,
) -> dict[str, Any]:
    """Read-only representation geometry; never a quality promotion metric."""

    histories = model.depth_hidden_states(input_ids)
    rows: list[dict[str, Any]] = []
    previous: torch.Tensor | None = None
    for depth, hidden in enumerate(histories):
        flat = hidden.detach().float().reshape(-1, hidden.shape[-1])
        if int(flat.shape[0]) > int(max_samples):
            indices = torch.linspace(
                0,
                int(flat.shape[0]) - 1,
                steps=int(max_samples),
                device=flat.device,
            ).long()
            flat = flat.index_select(0, indices)
        centered = flat - flat.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / float(max(1, int(centered.shape[0]) - 1))
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        total = eigenvalues.sum().clamp_min(1.0e-12)
        probabilities = eigenvalues / total
        participation = total.square() / eigenvalues.square().sum().clamp_min(1.0e-12)
        effective_rank = torch.exp(
            -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
        )
        rms = flat.square().mean().sqrt()
        mean_norm = flat.mean(dim=0).norm()
        adjacent_cosine = None
        if previous is not None:
            adjacent_cosine = float(
                F.cosine_similarity(flat, previous, dim=-1).mean().cpu()
            )
        rows.append(
            {
                "depth": depth,
                "sample_count": int(flat.shape[0]),
                "participation_ratio": float(participation.cpu()),
                "effective_rank": float(effective_rank.cpu()),
                "rms": float(rms.cpu()),
                "mean_vector_norm": float(mean_norm.cpu()),
                "adjacent_cosine": adjacent_cosine,
            }
        )
        previous = flat
    return {
        "surface": "marulho_depth_geometry.v1",
        "promotion_metric": False,
        "rows": rows,
        "external_llm_used": False,
    }
