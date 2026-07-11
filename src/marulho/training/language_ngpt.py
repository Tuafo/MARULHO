"""Hyperspherical normalized Transformer language candidate."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel
from marulho.training.language_transformer import _apply_rotary


def _unit_norm(
    value: torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    dtype = value.dtype
    runtime = value.float()
    return (runtime / runtime.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)).to(
        dtype=dtype
    )


@dataclass(frozen=True)
class NormalizedTransformerConfig:
    vocab_size: int
    width: int = 448
    layers: int = 4
    attention_heads: int = 8
    hidden_width: int = 1936
    context_length: int = 72
    alpha_init: float = 0.05
    normalization_eps: float = 1.0e-6
    active_language_path: str = "marulho_hyperspherical_transformer_v6"


def _validate_config(config: NormalizedTransformerConfig) -> None:
    if int(config.vocab_size) <= 1:
        raise ValueError("vocab_size must exceed one")
    if int(config.width) <= 0 or int(config.layers) < 1:
        raise ValueError("width and layers must be positive")
    if int(config.attention_heads) < 1 or int(config.width) % int(
        config.attention_heads
    ) != 0:
        raise ValueError("width must be divisible by attention_heads")
    if (int(config.width) // int(config.attention_heads)) % 2 != 0:
        raise ValueError("attention head dimension must be even")
    if int(config.hidden_width) < int(config.width):
        raise ValueError("hidden_width must be at least width")
    if int(config.context_length) < 2:
        raise ValueError("context_length must be at least two")
    if not math.isfinite(float(config.alpha_init)) or not 0.0 < float(
        config.alpha_init
    ) < 1.0:
        raise ValueError("alpha_init must be in (0, 1)")


class MarulhoNormalizedSelfAttention(nn.Module):
    def __init__(self, config: NormalizedTransformerConfig) -> None:
        super().__init__()
        self.width = int(config.width)
        self.attention_heads = int(config.attention_heads)
        self.head_dim = self.width // self.attention_heads
        self.context_length = int(config.context_length)
        self.normalization_eps = float(config.normalization_eps)
        self.base_scale = 1.0 / math.sqrt(float(self.width))
        self.query = nn.Linear(self.width, self.width, bias=False)
        self.key = nn.Linear(self.width, self.width, bias=False)
        self.value = nn.Linear(self.width, self.width, bias=False)
        self.output = nn.Linear(self.width, self.width, bias=False)
        self.sqk = nn.Parameter(
            torch.full((self.width,), self.base_scale, dtype=torch.float32)
        )

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
        hidden: torch.Tensor,
        *,
        past_key: torch.Tensor | None,
        past_value: torch.Tensor | None,
        position_offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, time_steps, _ = hidden.shape
        query = self._heads(self.query(hidden))
        key = self._heads(self.key(hidden))
        current_value = self._heads(self.value(hidden))
        positions = torch.arange(int(time_steps), device=hidden.device)
        positions = positions + position_offset.to(hidden.device)
        query, key = _apply_rotary(query, key, positions)
        scale = (self.sqk / self.base_scale).view(
            1,
            self.attention_heads,
            1,
            self.head_dim,
        )
        scale = scale.to(device=hidden.device, dtype=query.dtype)
        query = _unit_norm(
            query,
            dim=-1,
            eps=self.normalization_eps,
        ) * scale
        key = _unit_norm(
            key,
            dim=-1,
            eps=self.normalization_eps,
        ) * scale

        usable_past_key: torch.Tensor | None = None
        usable_past_value: torch.Tensor | None = None
        if past_key is not None and past_value is not None and int(past_key.shape[2]) > 0:
            keep_past = max(0, self.context_length - int(time_steps))
            if keep_past > 0:
                usable_past_key = past_key[:, :, -keep_past:].to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                usable_past_value = past_value[:, :, -keep_past:].to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
        if usable_past_key is None:
            full_key = key
            full_value = current_value
            past_length = 0
        else:
            full_key = torch.cat((usable_past_key, key), dim=2)
            full_value = torch.cat((usable_past_value, current_value), dim=2)
            past_length = int(usable_past_key.shape[2])

        attention_scale = math.sqrt(float(self.head_dim))
        if past_length == 0:
            attended = F.scaled_dot_product_attention(
                query,
                full_key,
                full_value,
                dropout_p=0.0,
                is_causal=True,
                scale=attention_scale,
            )
        elif int(time_steps) == 1:
            attended = F.scaled_dot_product_attention(
                query,
                full_key,
                full_value,
                dropout_p=0.0,
                is_causal=False,
                scale=attention_scale,
            )
        else:
            key_positions = torch.arange(
                int(full_key.shape[2]),
                device=hidden.device,
            ).unsqueeze(0)
            query_limits = past_length + torch.arange(
                int(time_steps),
                device=hidden.device,
            ).unsqueeze(1)
            causal_mask = key_positions <= query_limits
            attended = F.scaled_dot_product_attention(
                query,
                full_key,
                full_value,
                attn_mask=causal_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=attention_scale,
            )
        attended = attended.transpose(1, 2).contiguous().view(
            int(batch_size),
            int(time_steps),
            self.width,
        )
        return self.output(attended), full_key, full_value


class MarulhoNormalizedTransformerBlock(nn.Module):
    def __init__(self, config: NormalizedTransformerConfig) -> None:
        super().__init__()
        self.width = int(config.width)
        self.hidden_width = int(config.hidden_width)
        self.normalization_eps = float(config.normalization_eps)
        self.base_scale = 1.0 / math.sqrt(float(self.width))
        self.alpha_init = float(config.alpha_init)
        self.attention = MarulhoNormalizedSelfAttention(config)
        self.gate_up = nn.Linear(
            self.width,
            2 * self.hidden_width,
            bias=False,
        )
        self.down = nn.Linear(self.hidden_width, self.width, bias=False)
        self.attention_alpha = nn.Parameter(
            torch.full((self.width,), self.base_scale, dtype=torch.float32)
        )
        self.mlp_alpha = nn.Parameter(
            torch.full((self.width,), self.base_scale, dtype=torch.float32)
        )
        self.suv = nn.Parameter(
            torch.ones(2 * self.hidden_width, dtype=torch.float32)
        )

    def _update(
        self,
        hidden: torch.Tensor,
        proposal: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        current = _unit_norm(
            hidden,
            dim=-1,
            eps=self.normalization_eps,
        )
        target = _unit_norm(
            proposal,
            dim=-1,
            eps=self.normalization_eps,
        )
        rate = (alpha * (self.alpha_init / self.base_scale)).abs()
        rate = rate.to(device=hidden.device, dtype=hidden.dtype)
        return _unit_norm(
            current + rate * (target - current),
            dim=-1,
            eps=self.normalization_eps,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        past_key: torch.Tensor | None,
        past_value: torch.Tensor | None,
        position_offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention, next_key, next_value = self.attention(
            hidden,
            past_key=past_key,
            past_value=past_value,
            position_offset=position_offset,
        )
        hidden = self._update(hidden, attention, self.attention_alpha)
        uv = self.gate_up(hidden)
        suv = self.suv * math.sqrt(float(self.width))
        uv = uv * suv.to(device=hidden.device, dtype=uv.dtype)
        up, gate = uv.chunk(2, dim=-1)
        proposal = self.down(up * F.silu(gate))
        hidden = self._update(hidden, proposal, self.mlp_alpha)
        return hidden, next_key, next_value


class MarulhoNormalizedTransformerStateBlock(nn.Module):
    surface = "marulho_normalized_transformer_state_block.v1"

    def __init__(self, config: NormalizedTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            MarulhoNormalizedTransformerBlock(config)
            for _ in range(config.layers)
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        head_dim = self.config.width // self.config.attention_heads
        state: dict[str, torch.Tensor] = {
            "position": torch.zeros((), device=device, dtype=torch.long)
        }
        for index in range(self.config.layers):
            state[f"layer_{index}_key"] = torch.empty(
                batch_size,
                self.config.attention_heads,
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
            raise ValueError("Normalized Transformer expects [batch, time, width]")
        batch_size, time_steps, _ = inputs.shape
        if int(time_steps) > self.config.context_length and state is None:
            raise ValueError("Input exceeds normalized Transformer context")
        current = (
            self.initial_state(
                int(batch_size),
                device=inputs.device,
                dtype=inputs.dtype,
            )
            if state is None
            else state
        )
        position = current["position"].to(device=inputs.device, dtype=torch.long)
        hidden = _unit_norm(
            inputs,
            dim=-1,
            eps=self.config.normalization_eps,
        )
        next_state: dict[str, torch.Tensor] = {
            "position": position + int(time_steps)
        }
        cache_tokens = 0
        for index, layer in enumerate(self.layers):
            hidden, key, value = layer(
                hidden,
                past_key=current.get(f"layer_{index}_key"),
                past_value=current.get(f"layer_{index}_value"),
                position_offset=position,
            )
            next_state[f"layer_{index}_key"] = key.detach()
            next_state[f"layer_{index}_value"] = value.detach()
            cache_tokens = int(key.shape[2])
        telemetry = {
            "surface": self.surface,
            "state_core": "hyperspherical_transformer",
            "telemetry_collected": bool(collect_telemetry),
            "state_dim": int(self.config.width),
            "state_layers": int(self.config.layers),
            "attention_heads": int(self.config.attention_heads),
            "context_length": int(self.config.context_length),
            "kv_cache_tokens": cache_tokens,
            "normalization": "l2_hypersphere",
            "position_encoding": "rotary",
            "attention_backend": "torch_scaled_dot_product_attention",
            "post_optimizer_weight_projection_required": True,
            "external_llm_used": False,
        }
        return hidden, next_state, telemetry

    def step(
        self,
        inputs: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        hidden, next_state, telemetry = self.forward(
            inputs.unsqueeze(1),
            state,
            collect_telemetry=collect_telemetry,
        )
        return hidden[:, 0], next_state, telemetry


class MarulhoNormalizedLanguageModel(MarulhoLanguageModel):
    """MARULHO-owned nGPT-style language candidate."""

    surface = "marulho_normalized_language_model.v1"

    def __init__(self, normalized_config: NormalizedTransformerConfig) -> None:
        nn.Module.__init__(self)
        _validate_config(normalized_config)
        self.normalized_config = normalized_config
        self.base_scale = 1.0 / math.sqrt(float(normalized_config.width))
        self.config = LanguageModelConfig(
            vocab_size=normalized_config.vocab_size,
            embedding_dim=normalized_config.width,
            state_dim=normalized_config.width,
            state_layers=normalized_config.layers,
            attention_heads=normalized_config.attention_heads,
            transformer_context_length=normalized_config.context_length,
            transformer_mlp_ratio=(
                float(normalized_config.hidden_width)
                / float(normalized_config.width)
            ),
            tie_embeddings=False,
            active_language_path=normalized_config.active_language_path,
        )
        self.token_embedding = nn.Embedding(
            normalized_config.vocab_size,
            normalized_config.width,
        )
        self.state_block = MarulhoNormalizedTransformerStateBlock(normalized_config)
        self.lm_head = nn.Linear(
            normalized_config.width,
            normalized_config.vocab_size,
            bias=False,
        )
        self.logit_scale = nn.Parameter(
            torch.full(
                (normalized_config.vocab_size,),
                self.base_scale,
                dtype=torch.float32,
            )
        )
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=self.base_scale,
                )
        self.project_hyperspherical_weights_()

    def _hyperspherical_projection_spec(self) -> tuple[tuple[torch.Tensor, int], ...]:
        rows: list[tuple[torch.Tensor, int]] = [
            (self.token_embedding.weight, 1),
            (self.lm_head.weight, 1),
        ]
        for block in self.state_block.layers:
            rows.extend(
                (
                    (block.attention.query.weight, 1),
                    (block.attention.key.weight, 1),
                    (block.attention.value.weight, 1),
                    (block.attention.output.weight, 0),
                    (block.gate_up.weight, 1),
                    (block.down.weight, 0),
                )
            )
        return tuple(rows)

    @torch.no_grad()
    def project_hyperspherical_weights_(self) -> None:
        """Restore matrix directions without synchronizing metrics to the host."""

        for parameter, dimension in self._hyperspherical_projection_spec():
            parameter.copy_(
                _unit_norm(
                    parameter,
                    dim=dimension,
                    eps=self.normalized_config.normalization_eps,
                )
            )

    @torch.no_grad()
    def hyperspherical_weight_evidence(self) -> dict[str, float]:
        """Audit projection error at explicit validation boundaries only."""

        rows = self._hyperspherical_projection_spec()
        errors = [
            parameter.float().norm(p=2, dim=dimension).sub(1.0).abs().max()
            for parameter, dimension in rows
        ]
        maximum_error = float(torch.stack(errors).max().cpu()) if errors else 0.0
        return {
            "projected_matrix_count": float(len(rows)),
            "maximum_unit_norm_error": maximum_error,
        }

    def post_optimizer_step(self) -> None:
        self.project_hyperspherical_weights_()

    def _effective_logit_scale(self, dtype: torch.dtype) -> torch.Tensor:
        return (self.logit_scale / self.base_scale).to(
            device=self.device,
            dtype=dtype,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        del decode_vocab_only
        if input_ids.ndim != 2:
            raise ValueError("Normalized language model expects [batch, time]")
        ids = input_ids.to(device=self.device, dtype=torch.long)
        hidden, next_state, telemetry = self.state_block(
            self.token_embedding(ids),
            state,
            collect_telemetry=collect_telemetry,
        )
        logits = self.lm_head(hidden)
        logits = logits * self._effective_logit_scale(logits.dtype)
        return {
            "logits": logits,
            "state": next_state,
            "telemetry": {
                **telemetry,
                "active_language_path": self.config.active_language_path,
                "owned_by_marulho": True,
                "external_llm_used": False,
                "vocab_size": int(self.config.vocab_size),
                "weight_tying": False,
            },
        }

    def forward_step(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        ids = input_ids.unsqueeze(1) if input_ids.ndim == 1 else input_ids
        if ids.ndim != 2 or int(ids.shape[1]) != 1:
            raise ValueError("forward_step expects [batch] or [batch, 1]")
        return self.forward(
            ids,
            state,
            collect_telemetry=collect_telemetry,
            decode_vocab_only=decode_vocab_only,
        )

    def next_token_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        collect_telemetry: bool = True,
        return_evidence: bool = True,
    ) -> dict[str, Any]:
        output = self.forward(input_ids, collect_telemetry=collect_telemetry)
        targets = target_ids.to(device=self.device, dtype=torch.long)
        logits = output["logits"]
        if logits.shape[:2] != targets.shape:
            raise ValueError("target_ids must match input batch/time dimensions")
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        evidence = {
            "surface": "marulho_normalized_cross_entropy.v1",
            "full_vocab_logits_materialized": True,
            "hyperspherical_weight_projection_required": True,
            "target_token_count": int(targets.numel()),
            "external_llm_used": False,
        }
        return {
            "loss": loss,
            "loss_kind": "full_vocab_cross_entropy",
            "loss_evidence": evidence if return_evidence else {},
            "state": output["state"],
            "telemetry": output["telemetry"],
        }

    def generation_decode_policy(
        self,
        *,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        policy = super().generation_decode_policy(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        return {
            **policy,
            "surface": "marulho_normalized_decode_policy.v1",
            "kv_cache": "bounded_hyperspherical_layers",
        }
