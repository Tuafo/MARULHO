"""MARULHO-owned positive particle-field recurrent language candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn

from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


@dataclass(frozen=True)
class ParticleFieldConfig:
    vocab_size: int
    width: int = 256
    particle_count: int = 24_576
    recurrences: int = 8
    heads: int = 4
    context_length: int = 256
    dropout: float = 0.10
    rope_theta: float = float(2**16)
    materialized_state_batch_limit: int = 8
    active_language_path: str = "marulho_particle_field_v28"


def _validate_config(config: ParticleFieldConfig) -> None:
    if int(config.vocab_size) <= 1:
        raise ValueError("particle-field vocab_size must be greater than one")
    if int(config.width) < 2 or int(config.width) % 2:
        raise ValueError("particle-field width must be positive and even")
    if int(config.particle_count) < 2:
        raise ValueError("particle-field particle_count must be at least two")
    if int(config.heads) < 1 or int(config.particle_count) % int(config.heads):
        raise ValueError("particle_count must be divisible by heads")
    if (int(config.particle_count) // int(config.heads)) % 2:
        raise ValueError("particle count per head must be even for rotary pairs")
    if int(config.recurrences) < 1:
        raise ValueError("particle-field recurrences must be positive")
    if int(config.context_length) < 2:
        raise ValueError("particle-field context_length must be at least two")
    if not math.isfinite(float(config.dropout)) or not 0.0 <= float(
        config.dropout
    ) < 1.0:
        raise ValueError("particle-field dropout must be finite and in [0, 1)")
    if not math.isfinite(float(config.rope_theta)) or float(config.rope_theta) <= 1.0:
        raise ValueError("particle-field rope_theta must be finite and greater than one")
    if int(config.materialized_state_batch_limit) < 1:
        raise ValueError("materialized_state_batch_limit must be positive")


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    even = value[..., ::2]
    odd = value[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


class MarulhoParticleFieldStateBlock(nn.Module):
    """Shared-depth positive particle dynamics with causal Hebbian state."""

    surface = "marulho_particle_field_state_block.v1"

    def __init__(self, config: ParticleFieldConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.width = int(config.width)
        self.particle_count = int(config.particle_count)
        self.recurrences = int(config.recurrences)
        self.heads = int(config.heads)
        self.particles_per_head = self.particle_count // self.heads
        self.encoder = nn.Parameter(
            torch.empty(self.heads, self.width, self.particles_per_head)
        )
        self.value_encoder = nn.Parameter(
            torch.empty(self.heads, self.width, self.particles_per_head)
        )
        self.decoder = nn.Parameter(torch.empty(self.particle_count, self.width))
        self.norm = nn.LayerNorm(
            self.width,
            elementwise_affine=False,
            bias=False,
        )
        self.dropout = nn.Dropout(float(config.dropout))
        inverse_frequency = 1.0 / (
            float(config.rope_theta)
            ** (
                torch.arange(0, self.particles_per_head, 2, dtype=torch.float32)
                / float(self.particles_per_head)
            )
        )
        self.register_buffer(
            "inverse_frequency",
            inverse_frequency,
            persistent=False,
        )
        nn.init.normal_(self.encoder, mean=0.0, std=0.02)
        nn.init.normal_(self.value_encoder, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder, mean=0.0, std=0.02)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {
            "position": torch.zeros((), device=device, dtype=torch.long)
        }
        shape = (
            int(batch_size),
            self.heads,
            self.particles_per_head,
            self.width,
        )
        for recurrence in range(self.recurrences):
            state[f"recurrence_{recurrence}_fast_weight"] = torch.zeros(
                shape,
                device=device,
                dtype=dtype,
            )
        return state

    def _rotate(
        self,
        value: torch.Tensor,
        *,
        position_offset: torch.Tensor,
    ) -> torch.Tensor:
        time_steps = int(value.shape[2])
        positions = torch.arange(
            time_steps,
            device=value.device,
            dtype=torch.float32,
        )
        positions = positions + position_offset.to(
            device=value.device,
            dtype=torch.float32,
        )
        angles = positions.unsqueeze(-1) * self.inverse_frequency.unsqueeze(0)
        cosine = torch.repeat_interleave(torch.cos(angles), 2, dim=-1).to(
            value.dtype
        )
        sine = torch.repeat_interleave(torch.sin(angles), 2, dim=-1).to(
            value.dtype
        )
        cosine = cosine.unsqueeze(0).unsqueeze(0)
        sine = sine.unsqueeze(0).unsqueeze(0)
        return value * cosine + _rotate_half(value) * sine

    def _past_state(
        self,
        state: Mapping[str, torch.Tensor] | None,
        *,
        recurrence: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if state is None:
            return None
        key = f"recurrence_{recurrence}_fast_weight"
        value = state.get(key)
        if value is None:
            raise ValueError("particle-field streaming state is incomplete")
        expected = (
            int(batch_size),
            self.heads,
            self.particles_per_head,
            self.width,
        )
        if tuple(value.shape) != expected:
            raise ValueError(f"particle-field state shape mismatch for {key}")
        return value.to(device=device, dtype=dtype)

    def forward(
        self,
        inputs: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        if inputs.ndim != 3 or int(inputs.shape[-1]) != self.width:
            raise ValueError(
                "particle-field state block expects [batch, time, width]"
            )
        batch_size, time_steps, _ = inputs.shape
        if state is None and int(time_steps) > int(self.config.context_length):
            raise ValueError("particle-field input exceeds configured training context")
        position_value = None if state is None else state.get("position")
        position_offset = (
            position_value.to(device=inputs.device, dtype=torch.long)
            if isinstance(position_value, torch.Tensor)
            else torch.zeros((), device=inputs.device, dtype=torch.long)
        )
        materialize_state = state is not None or int(batch_size) <= int(
            self.config.materialized_state_batch_limit
        )
        hidden = self.norm(inputs)
        next_state: dict[str, torch.Tensor] = {
            "position": position_offset + int(time_steps)
        }
        sparse_fractions: list[torch.Tensor] = []
        for recurrence in range(self.recurrences):
            value_stream = hidden
            particle_input = torch.einsum(
                "btd,hdn->bhtn",
                value_stream,
                self.encoder,
            )
            x_sparse = torch.relu(particle_input)
            rotated = self._rotate(x_sparse, position_offset=position_offset)
            past = self._past_state(
                state,
                recurrence=recurrence,
                batch_size=int(batch_size),
                device=inputs.device,
                dtype=inputs.dtype,
            )
            read = (
                torch.zeros(
                    int(batch_size),
                    self.heads,
                    int(time_steps),
                    self.width,
                    device=inputs.device,
                    dtype=inputs.dtype,
                )
                if past is None
                else torch.einsum("bhtn,bhnd->bhtd", rotated, past)
            )
            if int(time_steps) > 1:
                scores = torch.matmul(rotated, rotated.transpose(-1, -2))
                scores = torch.tril(scores, diagonal=-1)
                read = read + torch.matmul(scores, value_stream.unsqueeze(1))
            read = self.norm(read)
            y_sparse = torch.relu(
                torch.einsum("bhtd,hdn->bhtn", read, self.value_encoder)
            )
            product = self.dropout(x_sparse * y_sparse)
            decoded = torch.matmul(
                product.transpose(1, 2).reshape(
                    int(batch_size),
                    int(time_steps),
                    self.particle_count,
                ),
                self.decoder,
            )
            hidden = self.norm(hidden + self.norm(decoded))
            if materialize_state:
                addition = torch.einsum(
                    "bhtn,btd->bhnd",
                    rotated,
                    value_stream,
                )
                next_fast_weight = addition if past is None else past + addition
                next_state[
                    f"recurrence_{recurrence}_fast_weight"
                ] = next_fast_weight.detach()
            if collect_telemetry:
                sparse_fractions.extend(
                    (
                        (x_sparse == 0).float().mean(),
                        (y_sparse == 0).float().mean(),
                    )
                )
        state_elements_per_sample = (
            self.recurrences
            * self.heads
            * self.particles_per_head
            * self.width
        )
        telemetry = {
            "surface": self.surface,
            "state_core": "positive_particle_field",
            "telemetry_collected": bool(collect_telemetry),
            "width": self.width,
            "particle_count": self.particle_count,
            "particles_per_head": self.particles_per_head,
            "heads": self.heads,
            "shared_recurrences": self.recurrences,
            "parameter_sharing_over_recurrence": True,
            "attention_kind": "strict_causal_hebbian_linear_attention",
            "training_attention_backend": "quadratic_parallel_reference",
            "streaming_state_materialized": bool(materialize_state),
            "streaming_state_elements_per_sample": state_elements_per_sample,
            "streaming_state_bytes_per_sample_at_input_dtype": (
                state_elements_per_sample * inputs.element_size()
            ),
            "mean_zero_activation_fraction": (
                float(torch.stack(sparse_fractions).mean().detach().cpu())
                if sparse_fractions
                else None
            ),
            "positive_activation_by_construction": True,
            "external_llm_used": False,
            "owned_by_marulho": True,
        }
        return hidden, next_state, telemetry

    def step(
        self,
        token_input: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        if token_input.ndim != 2 or int(token_input.shape[-1]) != self.width:
            raise ValueError("particle-field step expects [batch, width]")
        hidden, next_state, telemetry = self.forward(
            token_input.unsqueeze(1),
            state,
            collect_telemetry=collect_telemetry,
        )
        return hidden[:, 0], next_state, telemetry


class MarulhoParticleFieldLanguageModel(MarulhoLanguageModel):
    """Uninstalled V28 particle-field causal language experiment."""

    surface = "marulho_particle_field_language_model.v1"
    generation_surface = "marulho_particle_field_generation.v1"

    def __init__(self, particle_config: ParticleFieldConfig) -> None:
        _validate_config(particle_config)
        self.particle_config = particle_config
        super().__init__(
            LanguageModelConfig(
                vocab_size=int(particle_config.vocab_size),
                embedding_dim=int(particle_config.width),
                state_dim=int(particle_config.width),
                state_layers=int(particle_config.recurrences),
                attention_heads=int(particle_config.heads),
                transformer_context_length=int(particle_config.context_length),
                transformer_mlp_ratio=1.0,
                transformer_dropout=float(particle_config.dropout),
                tie_embeddings=True,
                active_language_path=str(particle_config.active_language_path),
            )
        )
        self.state_block = MarulhoParticleFieldStateBlock(particle_config)

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
                "surface": "marulho_particle_field_cross_entropy.v1",
            }
        return result

    def generation_decode_policy(self, **kwargs) -> dict[str, Any]:
        policy = super().generation_decode_policy(**kwargs)
        return {
            **policy,
            "surface": "marulho_particle_field_decode_policy.v1",
            "kv_cache": "not_applicable",
            "state_cache": "per_recurrence_hebbian_fast_weight",
        }

    def parameter_report(self) -> dict[str, Any]:
        config = self.particle_config
        matrix_parameters = 3 * int(config.particle_count) * int(config.width)
        embedding_parameters = int(config.vocab_size) * int(config.width)
        total = sum(int(parameter.numel()) for parameter in self.parameters())
        recurrent_multiplies = (
            5
            * int(config.particle_count)
            * int(config.width)
            * int(config.recurrences)
        )
        return {
            "surface": "marulho_particle_field_parameter_report.v1",
            "config": asdict(config),
            "total_parameters": total,
            "particle_matrix_parameters": matrix_parameters,
            "tied_embedding_head_parameters": embedding_parameters,
            "parameter_accounting_exact": total
            == matrix_parameters + embedding_parameters,
            "estimated_recurrent_multiplies_per_token": recurrent_multiplies,
            "positive_particles": int(config.particle_count),
            "external_llm_used": False,
            "owned_by_marulho": True,
        }

    @torch.no_grad()
    def recurrent_scan(
        self,
        input_ids: torch.Tensor,
    ) -> dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("particle-field recurrent scan expects [batch, time]")
        state: Mapping[str, torch.Tensor] | None = None
        logits: list[torch.Tensor] = []
        for index in range(int(input_ids.shape[1])):
            row = self.forward_step(
                input_ids[:, index],
                state,
                collect_telemetry=False,
            )
            logits.append(row["logits"][:, 0])
            state = row["state"]
        return {
            "logits": torch.stack(logits, dim=1),
            "state": state,
            "external_llm_used": False,
            "owned_by_marulho": True,
        }
