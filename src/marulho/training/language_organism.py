"""Distributed predictive-organism language candidate.

The candidate keeps exact local attention and a population of small recurrent
predictive units in parallel. A bounded latent episodic store and delayed
counterfactual utility targets provide separate memory and credit timescales.
It is an experiment surface, not an installed brain runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import torch
from torch import nn
import torch.nn.functional as F

from marulho.data.language_tokenizer import (
    LanguageTokenizer,
    load_language_tokenizer_state,
)
from marulho.training.language_model import _apply_decode_controls
from marulho.training.language_transformer import (
    MarulhoCausalSelfAttention,
    TransformerRMSNorm,
)


CHECKPOINT_SURFACE = "marulho_distributed_language_checkpoint.v1"
MODEL_SURFACE = "marulho_distributed_predictive_language_model.v1"


@dataclass(frozen=True)
class DistributedLanguageConfig:
    vocab_size: int
    width: int = 512
    layers: int = 4
    attention_heads: int = 8
    context_length: int = 72
    unit_groups: int = 8
    workspace_slots: int = 2
    episodic_slots: int = 16
    state_update_interval: int = 24
    mlp_dim: int = 1592
    dropout: float = 0.0
    minimum_utility_gate: float = 0.1
    utility_target_scale: float = 10.0
    utility_loss_weight: float = 0.05
    counterfactual_rate: float = 0.125
    counterfactual_horizon: int = 8
    episode_counterfactual_fraction: float = 0.25
    episode_usage_decay: float = 0.995
    tie_embeddings: bool = True


@dataclass(frozen=True)
class _Intervention:
    layer_index: int
    token_index: int
    kind: str
    unit_group: int | None = None


def _validate_config(config: DistributedLanguageConfig) -> None:
    if int(config.vocab_size) <= 1:
        raise ValueError("vocab_size must be greater than one")
    if int(config.width) <= 0 or int(config.layers) <= 0:
        raise ValueError("width and layers must be positive")
    if int(config.attention_heads) <= 0:
        raise ValueError("attention_heads must be positive")
    if int(config.width) % int(config.attention_heads) != 0:
        raise ValueError("width must be divisible by attention_heads")
    attention_head_dim = int(config.width) // int(config.attention_heads)
    if attention_head_dim % 2 != 0:
        raise ValueError("attention head dimension must be even for rotary positions")
    if int(config.unit_groups) <= 1 or int(config.width) % int(config.unit_groups) != 0:
        raise ValueError("width must be divisible by at least two unit_groups")
    if (
        int(config.workspace_slots) <= 0
        or int(config.episodic_slots) <= 0
        or int(config.state_update_interval) <= 0
    ):
        raise ValueError(
            "workspace_slots, episodic_slots, and state_update_interval must be positive"
        )
    if int(config.context_length) < 2 or int(config.mlp_dim) < int(config.width):
        raise ValueError("context_length and mlp_dim are too small")
    for name, value in (
        ("dropout", config.dropout),
        ("minimum_utility_gate", config.minimum_utility_gate),
        ("counterfactual_rate", config.counterfactual_rate),
        ("episode_counterfactual_fraction", config.episode_counterfactual_fraction),
        ("episode_usage_decay", config.episode_usage_decay),
    ):
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if float(config.dropout) > 0.0 and float(config.counterfactual_rate) > 0.0:
        raise ValueError("counterfactual credit requires deterministic dropout=0")
    if not math.isfinite(float(config.utility_target_scale)):
        raise ValueError("utility_target_scale must be finite")
    if not math.isfinite(float(config.utility_loss_weight)):
        raise ValueError("utility_loss_weight must be finite")
    if float(config.utility_loss_weight) < 0.0:
        raise ValueError("utility_loss_weight must be non-negative")
    if int(config.counterfactual_horizon) <= 0:
        raise ValueError("counterfactual_horizon must be positive")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _PredictiveUnitPopulation(nn.Module):
    """Vectorized units, shared workspace, and latent episodic memory."""

    def __init__(self, config: DistributedLanguageConfig) -> None:
        super().__init__()
        self.width = int(config.width)
        self.unit_groups = int(config.unit_groups)
        self.unit_dim = self.width // self.unit_groups
        self.workspace_slots = int(config.workspace_slots)
        self.episodic_slots = int(config.episodic_slots)
        self.state_update_interval = int(config.state_update_interval)
        self.minimum_utility_gate = float(config.minimum_utility_gate)
        self.episode_usage_decay = float(config.episode_usage_decay)

        self.unit_input = nn.Linear(self.width, self.width, bias=False)
        self.unit_write = nn.Linear(self.width, self.unit_groups, bias=True)
        self.unit_relevance = nn.Linear(self.width, self.unit_groups, bias=False)
        self.recurrent_weight = nn.Parameter(
            torch.empty(self.unit_groups, self.unit_dim, self.unit_dim)
        )
        self.recurrent_bias = nn.Parameter(
            torch.zeros(self.unit_groups, self.unit_dim)
        )
        self.retention_logits = nn.Parameter(torch.empty(self.unit_groups, 1))
        self.workspace_queries = nn.Parameter(
            torch.empty(self.workspace_slots, self.unit_dim)
        )
        self.unit_output = nn.Parameter(
            torch.empty(self.unit_groups, self.unit_dim, self.width)
        )
        self.utility_head = nn.Linear(self.unit_dim, 1, bias=False)
        self.utility_bias = nn.Parameter(torch.zeros(self.unit_groups))

        self.episode_query = nn.Linear(self.width, self.unit_dim, bias=False)
        self.episode_key = nn.Linear(self.width, self.unit_dim, bias=False)
        self.episode_value = nn.Linear(self.width, self.unit_dim, bias=False)
        self.episode_output = nn.Linear(self.unit_dim, self.width, bias=False)
        self.episode_write = nn.Linear(self.width, 1, bias=True)
        self.episode_utility = nn.Linear(self.width + self.unit_dim, 1, bias=True)

        self.reset_special_parameters()

    def reset_special_parameters(self) -> None:
        nn.init.normal_(self.recurrent_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.workspace_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.unit_output, mean=0.0, std=0.02)
        retention = torch.linspace(0.50, 0.995, self.unit_groups).unsqueeze(1)
        with torch.no_grad():
            self.retention_logits.copy_(torch.logit(retention.clamp(0.001, 0.999)))
            self.unit_write.bias.fill_(-1.0)
            self.episode_write.bias.fill_(-1.0)
            self.utility_head.weight.zero_()
            self.utility_bias.zero_()
            self.episode_utility.weight.zero_()
            self.episode_utility.bias.zero_()

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        return {
            "units": torch.zeros(
                int(batch_size),
                self.unit_groups,
                self.unit_dim,
                device=device,
                dtype=dtype,
            ),
            "episode_keys": torch.zeros(
                int(batch_size),
                self.episodic_slots,
                self.unit_dim,
                device=device,
                dtype=dtype,
            ),
            "episode_values": torch.zeros(
                int(batch_size),
                self.episodic_slots,
                self.unit_dim,
                device=device,
                dtype=dtype,
            ),
            "episode_usage": torch.zeros(
                int(batch_size),
                self.episodic_slots,
                device=device,
                dtype=dtype,
            ),
            "pending_unit_sum": torch.zeros(
                int(batch_size),
                self.unit_groups,
                self.unit_dim,
                device=device,
                dtype=dtype,
            ),
            "pending_unit_weight": torch.zeros(
                int(batch_size),
                self.unit_groups,
                device=device,
                dtype=dtype,
            ),
            "pending_episode_sum": torch.zeros(
                int(batch_size),
                self.width,
                device=device,
                dtype=dtype,
            ),
            "pending_episode_weight": torch.zeros(
                int(batch_size),
                device=device,
                dtype=dtype,
            ),
            "pending_count": torch.zeros((), device=device, dtype=torch.long),
        }

    def _episode_read(
        self,
        current: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        usage: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = F.normalize(self.episode_query(current), dim=-1, eps=1.0e-6)
        normalized_keys = F.normalize(keys, dim=-1, eps=1.0e-6)
        similarity = torch.einsum("btd,bsd->bts", query, normalized_keys)
        availability = torch.log(usage.clamp_min(1.0e-4)).unsqueeze(1)
        weights = torch.softmax(similarity + availability, dim=-1)
        read = torch.einsum("bts,bsd->btd", weights, values)
        return read, weights, similarity

    def _episode_update(
        self,
        current: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        usage: torch.Tensor,
        strength: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        new_key = F.normalize(self.episode_key(current), dim=-1, eps=1.0e-6)
        new_value = self.episode_value(current)
        normalized_keys = F.normalize(keys, dim=-1, eps=1.0e-6)
        similarity = torch.einsum("bd,bsd->bs", new_key, normalized_keys)
        content_weights = torch.softmax(similarity / 0.25, dim=-1)
        allocation_weights = torch.softmax(-usage / 0.25, dim=-1)
        novelty = torch.sigmoid(4.0 * (0.30 - similarity.max(dim=-1).values))
        write_weights = (
            novelty.unsqueeze(-1) * allocation_weights
            + (1.0 - novelty.unsqueeze(-1)) * content_weights
        )
        erase = write_weights * strength.unsqueeze(-1)
        keys = keys * (1.0 - erase.unsqueeze(-1))
        keys = keys + erase.unsqueeze(-1) * new_key.unsqueeze(1)
        values = values * (1.0 - erase.unsqueeze(-1))
        values = values + erase.unsqueeze(-1) * new_value.unsqueeze(1)
        usage = (
            usage * self.episode_usage_decay + erase
        ).clamp(min=0.0, max=1.0)
        return keys, values, usage, strength

    def forward(
        self,
        inputs: torch.Tensor,
        state: Mapping[str, torch.Tensor],
        *,
        layer_index: int,
        intervention: _Intervention | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch_size, time_steps, _ = inputs.shape
        units = state["units"].to(device=inputs.device, dtype=inputs.dtype)
        episode_keys = state["episode_keys"].to(
            device=inputs.device, dtype=inputs.dtype
        )
        episode_values = state["episode_values"].to(
            device=inputs.device, dtype=inputs.dtype
        )
        episode_usage = state["episode_usage"].to(
            device=inputs.device, dtype=inputs.dtype
        )
        pending_unit_sum = state["pending_unit_sum"].to(
            device=inputs.device, dtype=inputs.dtype
        )
        pending_unit_weight = state["pending_unit_weight"].to(
            device=inputs.device, dtype=inputs.dtype
        )
        pending_episode_sum = state["pending_episode_sum"].to(
            device=inputs.device, dtype=inputs.dtype
        )
        pending_episode_weight = state["pending_episode_weight"].to(
            device=inputs.device, dtype=inputs.dtype
        )
        pending_count = int(state["pending_count"].detach().cpu().item())
        retention = torch.sigmoid(self.retention_logits).to(dtype=inputs.dtype)

        outputs: list[torch.Tensor] = []
        unit_utilities: list[torch.Tensor] = []
        episode_utilities: list[torch.Tensor] = []
        unit_gates: list[torch.Tensor] = []
        episode_gates: list[torch.Tensor] = []
        episode_writes: list[torch.Tensor] = []
        relevance_rows: list[torch.Tensor] = []

        start = 0
        while start < int(time_steps):
            segment_length = min(
                self.state_update_interval - pending_count,
                int(time_steps) - start,
            )
            end = start + segment_length
            current = inputs[:, start:end]
            projected = self.unit_input(current).view(
                int(batch_size), segment_length, self.unit_groups, self.unit_dim
            )
            recurrent = torch.einsum(
                "bgd,gde->bge", units, self.recurrent_weight
            ).unsqueeze(1)
            broadcast_scores = torch.einsum(
                "bgd,kd->bkg", units, self.workspace_queries
            ) / math.sqrt(float(self.unit_dim))
            broadcast_weights = torch.softmax(broadcast_scores, dim=-1)
            workspace = torch.einsum("bkg,bgd->bkd", broadcast_weights, units)
            read_scores = torch.einsum(
                "btgd,bkd->btgk", projected, workspace
            ) / math.sqrt(float(self.unit_dim))
            workspace_message = torch.einsum(
                "btgk,bkd->btgd", torch.softmax(read_scores, dim=-1), workspace
            )

            episode_read, _read_weights, _similarity = self._episode_read(
                current,
                episode_keys,
                episode_values,
                episode_usage,
            )
            candidate = torch.tanh(
                projected
                + recurrent
                + workspace_message
                + episode_read.unsqueeze(2)
                + self.recurrent_bias
            )
            utility_prediction = (
                self.utility_head(candidate).squeeze(-1) + self.utility_bias
            )
            utility_gate = self.minimum_utility_gate + (
                1.0 - self.minimum_utility_gate
            ) * torch.sigmoid(utility_prediction)
            write_gate = torch.sigmoid(self.unit_write(current)) * utility_gate

            selected_unit: int | None = None
            selected_here = bool(
                intervention is not None
                and int(intervention.layer_index) == int(layer_index)
                and start <= int(intervention.token_index) < end
            )
            if selected_here and intervention.kind == "unit":
                selected_unit = int(intervention.unit_group or 0)
                unit_mask = torch.ones_like(write_gate)
                local_token = int(intervention.token_index) - start
                unit_mask[:, local_token, selected_unit] = 0.0
                write_gate = write_gate * unit_mask

            unit_values = torch.einsum("btgd,gdw->btgw", candidate, self.unit_output)
            relevance = torch.softmax(
                self.unit_relevance(current) + utility_prediction,
                dim=-1,
            )
            contributions = relevance.unsqueeze(-1) * unit_values
            if selected_unit is not None:
                contribution_mask = torch.ones_like(relevance)
                local_token = int(intervention.token_index) - start
                contribution_mask[:, local_token, selected_unit] = 0.0
                contributions = contributions * contribution_mask.unsqueeze(-1)
            population_output = contributions.sum(dim=2)

            episode_prediction = self.episode_utility(
                torch.cat((current, episode_read), dim=-1)
            ).squeeze(-1)
            episode_gate = self.minimum_utility_gate + (
                1.0 - self.minimum_utility_gate
            ) * torch.sigmoid(episode_prediction)
            population_output = population_output + (
                episode_gate.unsqueeze(-1) * self.episode_output(episode_read)
            )
            episode_write = (
                torch.sigmoid(self.episode_write(current)).squeeze(-1)
                * episode_gate
            )
            if selected_here and intervention.kind == "episode":
                episode_mask = torch.ones_like(episode_write)
                episode_mask[:, int(intervention.token_index) - start] = 0.0
                episode_write = episode_write * episode_mask

            pending_unit_sum = pending_unit_sum + (
                write_gate.unsqueeze(-1) * candidate
            ).sum(dim=1)
            pending_unit_weight = pending_unit_weight + write_gate.sum(dim=1)
            pending_episode_sum = pending_episode_sum + (
                episode_write.unsqueeze(-1) * current
            ).sum(dim=1)
            pending_episode_weight = (
                pending_episode_weight + episode_write.sum(dim=1)
            )
            pending_count += segment_length

            if pending_count == self.state_update_interval:
                target_units = pending_unit_sum / pending_unit_weight.clamp_min(
                    1.0e-6
                ).unsqueeze(-1)
                mean_unit_write = (
                    pending_unit_weight / float(self.state_update_interval)
                ).clamp(min=0.0, max=1.0)
                units = units + (
                    (1.0 - retention.unsqueeze(0))
                    * mean_unit_write.unsqueeze(-1)
                    * (target_units - units)
                )
                episode_summary = pending_episode_sum / (
                    pending_episode_weight.clamp_min(1.0e-6).unsqueeze(-1)
                )
                episode_strength = (
                    pending_episode_weight / float(self.state_update_interval)
                ).clamp(min=0.0, max=1.0)
                (
                    episode_keys,
                    episode_values,
                    episode_usage,
                    _summary_strength,
                ) = self._episode_update(
                    episode_summary,
                    episode_keys,
                    episode_values,
                    episode_usage,
                    episode_strength,
                )
                pending_unit_sum = torch.zeros_like(pending_unit_sum)
                pending_unit_weight = torch.zeros_like(pending_unit_weight)
                pending_episode_sum = torch.zeros_like(pending_episode_sum)
                pending_episode_weight = torch.zeros_like(pending_episode_weight)
                pending_count = 0

            outputs.append(population_output)
            unit_utilities.append(utility_prediction)
            episode_utilities.append(episode_prediction)
            unit_gates.append(utility_gate)
            episode_gates.append(episode_gate)
            episode_writes.append(episode_write)
            relevance_rows.append(relevance)
            start = end

        output = torch.cat(outputs, dim=1)
        evidence = {
            "unit_utility": torch.cat(unit_utilities, dim=1),
            "episode_utility": torch.cat(episode_utilities, dim=1),
            "unit_gate": torch.cat(unit_gates, dim=1),
            "episode_gate": torch.cat(episode_gates, dim=1),
            "episode_write": torch.cat(episode_writes, dim=1),
            "unit_relevance": torch.cat(relevance_rows, dim=1),
        }
        next_state = {
            "units": units,
            "episode_keys": episode_keys,
            "episode_values": episode_values,
            "episode_usage": episode_usage,
            "pending_unit_sum": pending_unit_sum,
            "pending_unit_weight": pending_unit_weight,
            "pending_episode_sum": pending_episode_sum,
            "pending_episode_weight": pending_episode_weight,
            "pending_count": torch.as_tensor(
                pending_count, device=inputs.device, dtype=torch.long
            ),
        }
        return output, next_state, evidence


class _DistributedPredictiveBlock(nn.Module):
    def __init__(self, config: DistributedLanguageConfig) -> None:
        super().__init__()
        self.width = int(config.width)
        self.parallel_norm = TransformerRMSNorm(self.width)
        self.attention = MarulhoCausalSelfAttention(
            self.width,
            attention_heads=int(config.attention_heads),
            context_length=int(config.context_length),
            dropout=float(config.dropout),
        )
        self.population = _PredictiveUnitPopulation(config)
        self.parallel_mixer = nn.Linear(self.width, 2, bias=True)
        self.mlp_norm = TransformerRMSNorm(self.width)
        self.gate_up = nn.Linear(self.width, int(config.mlp_dim) * 2, bias=False)
        self.down = nn.Linear(int(config.mlp_dim), self.width, bias=False)
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(
        self,
        value: torch.Tensor,
        state: Mapping[str, torch.Tensor],
        *,
        layer_index: int,
        position_offset: torch.Tensor,
        intervention: _Intervention | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        normalized = self.parallel_norm(value)
        attention, next_key, next_value = self.attention(
            normalized,
            past_key=state.get("key"),
            past_value=state.get("value"),
            position_offset=position_offset,
        )
        population, population_state, evidence = self.population(
            normalized,
            state,
            layer_index=layer_index,
            intervention=intervention,
        )
        mix = torch.softmax(self.parallel_mixer(normalized), dim=-1)
        value = value + self.dropout(
            mix[..., :1] * attention + mix[..., 1:] * population
        )
        gate, up = self.gate_up(self.mlp_norm(value)).chunk(2, dim=-1)
        value = value + self.dropout(self.down(F.silu(gate) * up))
        return (
            value,
            {
                "key": next_key,
                "value": next_value,
                **population_state,
            },
            {**evidence, "parallel_mix": mix},
        )


class _DistributedPredictiveStateBlock(nn.Module):
    def __init__(self, config: DistributedLanguageConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            _DistributedPredictiveBlock(config) for _ in range(int(config.layers))
        )
        self.output_norm = TransformerRMSNorm(int(config.width))

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        head_dim = int(self.config.width) // int(self.config.attention_heads)
        state: dict[str, torch.Tensor] = {
            "position": torch.zeros((), device=device, dtype=torch.long)
        }
        for layer_index, layer in enumerate(self.layers):
            prefix = f"layer_{layer_index}_"
            state[f"{prefix}key"] = torch.empty(
                int(batch_size),
                int(self.config.attention_heads),
                0,
                head_dim,
                device=device,
                dtype=dtype,
            )
            state[f"{prefix}value"] = torch.empty_like(state[f"{prefix}key"])
            population = layer.population.initial_state(
                int(batch_size), device=device, dtype=dtype
            )
            for key, tensor in population.items():
                state[f"{prefix}{key}"] = tensor
        return state

    def forward(
        self,
        inputs: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        intervention: _Intervention | None = None,
        collect_telemetry: bool = True,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, Any],
        dict[str, torch.Tensor],
    ]:
        if inputs.ndim != 3:
            raise ValueError("Distributed state block expects [batch, time, width]")
        batch_size, time_steps, _ = inputs.shape
        if int(time_steps) > int(self.config.context_length):
            raise ValueError("input chunk exceeds bounded exact context_length")
        current_state = (
            self.initial_state(
                int(batch_size), device=inputs.device, dtype=inputs.dtype
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
        hidden = inputs
        next_state: dict[str, torch.Tensor] = {
            "position": position_offset + int(time_steps)
        }
        unit_utility_rows: list[torch.Tensor] = []
        episode_utility_rows: list[torch.Tensor] = []
        unit_gate_rows: list[torch.Tensor] = []
        episode_gate_rows: list[torch.Tensor] = []
        write_rows: list[torch.Tensor] = []
        mix_rows: list[torch.Tensor] = []
        relevance_rows: list[torch.Tensor] = []
        cache_tokens = 0
        final_usage: list[torch.Tensor] = []
        for layer_index, layer in enumerate(self.layers):
            prefix = f"layer_{layer_index}_"
            layer_state = {
                key: current_state[f"{prefix}{key}"]
                for key in (
                    "key",
                    "value",
                    "units",
                    "episode_keys",
                    "episode_values",
                    "episode_usage",
                    "pending_unit_sum",
                    "pending_unit_weight",
                    "pending_episode_sum",
                    "pending_episode_weight",
                    "pending_count",
                )
            }
            hidden, layer_next, evidence = layer(
                hidden,
                layer_state,
                layer_index=layer_index,
                position_offset=position_offset,
                intervention=intervention,
            )
            for key, tensor in layer_next.items():
                next_state[f"{prefix}{key}"] = tensor.detach()
            cache_tokens = int(layer_next["key"].shape[2])
            final_usage.append(layer_next["episode_usage"])
            unit_utility_rows.append(evidence["unit_utility"])
            episode_utility_rows.append(evidence["episode_utility"])
            unit_gate_rows.append(evidence["unit_gate"])
            episode_gate_rows.append(evidence["episode_gate"])
            write_rows.append(evidence["episode_write"])
            mix_rows.append(evidence["parallel_mix"])
            relevance_rows.append(evidence["unit_relevance"])
        hidden = self.output_norm(hidden)
        auxiliary = {
            "unit_utility": torch.stack(unit_utility_rows, dim=0),
            "episode_utility": torch.stack(episode_utility_rows, dim=0),
            "unit_gate": torch.stack(unit_gate_rows, dim=0),
            "episode_gate": torch.stack(episode_gate_rows, dim=0),
            "episode_write": torch.stack(write_rows, dim=0),
            "parallel_mix": torch.stack(mix_rows, dim=0),
            "unit_relevance": torch.stack(relevance_rows, dim=0),
        }
        telemetry: dict[str, Any] = {
            "surface": "marulho_distributed_predictive_state_block.v1",
            "state_core": "distributed_predictive_organism",
            "telemetry_collected": bool(collect_telemetry),
            "time_steps": int(time_steps),
            "state_layers": int(self.config.layers),
            "state_dim": int(self.config.width),
            "unit_groups": int(self.config.unit_groups),
            "unit_dim": int(self.config.width) // int(self.config.unit_groups),
            "workspace_slots": int(self.config.workspace_slots),
            "episodic_slots_per_layer": int(self.config.episodic_slots),
            "state_update_interval": int(self.config.state_update_interval),
            "context_length": int(self.config.context_length),
            "kv_cache_tokens": cache_tokens,
            "attention_backend": "torch_scaled_dot_product_attention",
            "exact_and_recurrent_paths": "parallel",
            "external_llm_used": False,
        }
        if collect_telemetry:
            telemetry.update(
                {
                    "mean_unit_utility_gate": float(
                        auxiliary["unit_gate"].detach().float().mean().cpu().item()
                    ),
                    "active_unit_fraction": float(
                        (auxiliary["unit_gate"].detach() >= 0.5)
                        .float()
                        .mean()
                        .cpu()
                        .item()
                    ),
                    "mean_episode_utility_gate": float(
                        auxiliary["episode_gate"]
                        .detach()
                        .float()
                        .mean()
                        .cpu()
                        .item()
                    ),
                    "mean_episode_write_strength": float(
                        auxiliary["episode_write"]
                        .detach()
                        .float()
                        .mean()
                        .cpu()
                        .item()
                    ),
                    "mean_exact_mix": float(
                        auxiliary["parallel_mix"][..., 0]
                        .detach()
                        .float()
                        .mean()
                        .cpu()
                        .item()
                    ),
                    "mean_population_mix": float(
                        auxiliary["parallel_mix"][..., 1]
                        .detach()
                        .float()
                        .mean()
                        .cpu()
                        .item()
                    ),
                    "mean_episode_usage": float(
                        torch.stack(final_usage)
                        .detach()
                        .float()
                        .mean()
                        .cpu()
                        .item()
                    ),
                }
            )
        return hidden, next_state, telemetry, auxiliary


class MarulhoDistributedLanguageModel(nn.Module):
    """MARULHO-owned experimental multi-timescale causal language model."""

    surface = MODEL_SURFACE

    def __init__(self, config: DistributedLanguageConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.token_embedding = nn.Embedding(int(config.vocab_size), int(config.width))
        self.state_block = _DistributedPredictiveStateBlock(config)
        self.lm_head = nn.Linear(int(config.width), int(config.vocab_size), bias=False)
        if bool(config.tie_embeddings):
            self.lm_head.weight = self.token_embedding.weight
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                bias = getattr(module, "bias", None)
                if isinstance(bias, torch.Tensor):
                    nn.init.zeros_(bias)
        for layer in self.state_block.layers:
            layer.population.reset_special_parameters()
            with torch.no_grad():
                layer.parallel_mixer.weight.zero_()
                layer.parallel_mixer.bias.zero_()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def context_length(self) -> int:
        return int(self.config.context_length)

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        intervention: _Intervention | None = None,
        collect_telemetry: bool = True,
    ) -> dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("Language model expects input_ids shaped [batch, time]")
        runtime_ids = input_ids.to(device=self.device, dtype=torch.long)
        hidden, next_state, telemetry, auxiliary = self.state_block(
            self.token_embedding(runtime_ids),
            state,
            intervention=intervention,
            collect_telemetry=collect_telemetry,
        )
        return {
            "hidden": hidden,
            "state": next_state,
            "telemetry": {
                **telemetry,
                "active_language_path": "marulho_distributed_predictive_candidate",
                "owned_by_marulho": True,
                "external_llm_used": False,
                "vocab_size": int(self.config.vocab_size),
            },
            "auxiliary": auxiliary,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        del decode_vocab_only
        result = self._forward_hidden(
            input_ids, state, collect_telemetry=collect_telemetry
        )
        return {
            "logits": self.lm_head(result["hidden"]),
            "state": result["state"],
            "telemetry": result["telemetry"],
        }

    def forward_step(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        del decode_vocab_only
        if input_ids.ndim == 1:
            step_ids = input_ids.unsqueeze(1)
        elif input_ids.ndim == 2 and int(input_ids.shape[1]) == 1:
            step_ids = input_ids
        else:
            raise ValueError("forward_step expects [batch] or [batch, 1] token ids")
        return self.forward(
            step_ids,
            state,
            collect_telemetry=collect_telemetry,
        )

    def next_token_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        collect_telemetry: bool = True,
        return_evidence: bool = True,
    ) -> dict[str, Any]:
        result = self._forward_hidden(
            input_ids,
            collect_telemetry=collect_telemetry,
        )
        logits = self.lm_head(result["hidden"])
        targets = target_ids.to(device=self.device, dtype=torch.long)
        if logits.shape[:2] != targets.shape:
            raise ValueError("target_ids must match input batch/time dimensions")
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        language_loss = token_losses.mean()
        utility_loss = language_loss.new_zeros(())
        counterfactual: dict[str, Any] = {
            "ran": False,
            "kind": None,
            "layer_index": None,
            "token_index": None,
            "unit_group": None,
            "horizon": 0,
            "mean_target": 0.0,
        }
        should_probe = bool(
            self.training
            and float(self.config.utility_loss_weight) > 0.0
            and float(self.config.counterfactual_rate) > 0.0
            and int(targets.shape[1]) > 1
            and float(torch.rand((), device=self.device).detach().cpu().item())
            < float(self.config.counterfactual_rate)
        )
        if should_probe:
            layer_index = int(
                torch.randint(
                    int(self.config.layers), (), device=self.device
                ).detach().cpu().item()
            )
            token_index = int(
                torch.randint(
                    int(targets.shape[1]) - 1, (), device=self.device
                ).detach().cpu().item()
            )
            episode_probe = bool(
                float(torch.rand((), device=self.device).detach().cpu().item())
                < float(self.config.episode_counterfactual_fraction)
            )
            unit_group = None
            kind = "episode" if episode_probe else "unit"
            if not episode_probe:
                unit_group = int(
                    torch.randint(
                        int(self.config.unit_groups), (), device=self.device
                    ).detach().cpu().item()
                )
            intervention = _Intervention(
                layer_index=layer_index,
                token_index=token_index,
                kind=kind,
                unit_group=unit_group,
            )
            with torch.no_grad():
                altered = self._forward_hidden(
                    input_ids,
                    intervention=intervention,
                    collect_telemetry=False,
                )
                altered_logits = self.lm_head(altered["hidden"])
                altered_losses = F.cross_entropy(
                    altered_logits.reshape(-1, altered_logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="none",
                ).view_as(targets)
                horizon_end = min(
                    int(targets.shape[1]),
                    token_index + int(self.config.counterfactual_horizon),
                )
                utility_target = (
                    altered_losses[:, token_index:horizon_end]
                    - token_losses.detach()[:, token_index:horizon_end]
                ).mean(dim=1)
                utility_target = utility_target * float(
                    self.config.utility_target_scale
                )
            auxiliary = result["auxiliary"]
            if episode_probe:
                prediction = auxiliary["episode_utility"][
                    layer_index, :, token_index
                ]
            else:
                prediction = auxiliary["unit_utility"][
                    layer_index, :, token_index, int(unit_group or 0)
                ]
            utility_loss = F.smooth_l1_loss(prediction, utility_target)
            counterfactual = {
                "ran": True,
                "kind": kind,
                "layer_index": layer_index,
                "token_index": token_index,
                "unit_group": unit_group,
                "horizon": horizon_end - token_index,
                "mean_target": float(utility_target.detach().mean().cpu().item()),
            }
        loss = language_loss + float(self.config.utility_loss_weight) * utility_loss
        evidence = {
            "surface": "marulho_distributed_cross_entropy_utility.v1",
            "sampled_vocab_training": False,
            "full_vocab_logits_materialized": True,
            "target_token_count": int(targets.numel()),
            "language_loss": float(language_loss.detach().cpu().item()),
            "utility_loss": float(utility_loss.detach().cpu().item()),
            "utility_loss_weight": float(self.config.utility_loss_weight),
            "counterfactual": counterfactual,
            "external_llm_used": False,
        }
        return {
            "loss": loss,
            "loss_kind": "full_vocab_cross_entropy_plus_counterfactual_utility",
            "loss_evidence": evidence if return_evidence else {},
            "state": result["state"],
            "telemetry": result["telemetry"],
        }

    def generation_decode_policy(
        self,
        *,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        temperature: float,
        top_p: float,
        seed: int | None,
    ) -> dict[str, Any]:
        return {
            "surface": "marulho_distributed_decode_policy.v1",
            "decode_strategy": (
                "nucleus_sampling" if float(temperature) > 0.0 else "greedy_argmax"
            ),
            "model_vocab_size": int(self.config.vocab_size),
            "generation_vocab_size": int(self.config.vocab_size),
            "full_model_vocab_logits_materialized": True,
            "repetition_penalty": max(1.0, float(repetition_penalty)),
            "no_repeat_ngram_size": max(0, int(no_repeat_ngram_size)),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "sampling_seed": None if seed is None else int(seed),
            "state_cache": "bounded_local_kv_plus_units_plus_latent_episodes",
            "external_llm_used": False,
        }

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_id: int | None = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        if not math.isfinite(float(temperature)) or float(temperature) < 0.0:
            raise ValueError("temperature must be finite and non-negative")
        if not math.isfinite(float(top_p)) or not 0.0 < float(top_p) <= 1.0:
            raise ValueError("top_p must be finite and in (0, 1]")
        if prompt_ids.ndim == 1:
            generated = prompt_ids.unsqueeze(0)
        elif prompt_ids.ndim == 2:
            generated = prompt_ids
        else:
            raise ValueError("prompt_ids must be [time] or [batch, time]")
        generated = generated.to(device=self.device, dtype=torch.long)
        if int(generated.shape[1]) <= 0:
            raise ValueError("prompt_ids must contain at least one token")
        sampling = float(temperature) > 0.0
        sampling_generator = None
        if sampling and seed is not None:
            sampling_generator = torch.Generator(device=self.device)
            sampling_generator.manual_seed(int(seed))
        was_training = bool(self.training)
        self.eval()
        try:
            state: Mapping[str, torch.Tensor] | None = None
            result: dict[str, Any] | None = None
            for start in range(0, int(generated.shape[1]), self.context_length):
                chunk = generated[:, start : start + self.context_length]
                result = self.forward(chunk, state, collect_telemetry=False)
                state = result["state"]
            if result is None:
                raise RuntimeError("prompt execution produced no result")
            next_logits = result["logits"][:, -1]
            finished = torch.zeros(
                int(generated.shape[0]), device=self.device, dtype=torch.bool
            )
            new_token_count = 0
            repetition_count = 0
            banned_count = 0
            fallback_count = 0
            for _ in range(max(0, int(max_new_tokens))):
                controlled, control = _apply_decode_controls(
                    next_logits,
                    generated,
                    repetition_penalty=max(1.0, float(repetition_penalty)),
                    no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
                )
                repetition_count += control["repetition_penalty_adjusted_token_count"]
                banned_count += control["no_repeat_ngram_banned_token_count"]
                fallback_count += control["decode_control_fallback_count"]
                if sampling:
                    probabilities = torch.softmax(
                        controlled / float(temperature), dim=-1
                    )
                    if float(top_p) < 1.0:
                        sorted_probabilities, sorted_indices = torch.sort(
                            probabilities, dim=-1, descending=True
                        )
                        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
                        remove = cumulative > float(top_p)
                        remove[..., 1:] = remove[..., :-1].clone()
                        remove[..., 0] = False
                        sorted_probabilities = sorted_probabilities.masked_fill(
                            remove, 0.0
                        )
                        sorted_probabilities = sorted_probabilities / (
                            sorted_probabilities.sum(dim=-1, keepdim=True)
                        ).clamp_min(torch.finfo(sorted_probabilities.dtype).tiny)
                        sampled_rank = torch.multinomial(
                            sorted_probabilities,
                            1,
                            generator=sampling_generator,
                        )
                        next_id = sorted_indices.gather(-1, sampled_rank)
                    else:
                        next_id = torch.multinomial(
                            probabilities, 1, generator=sampling_generator
                        )
                else:
                    next_id = torch.argmax(controlled, dim=-1, keepdim=True)
                if eos_id is not None:
                    next_id = torch.where(
                        finished.unsqueeze(1),
                        torch.full_like(next_id, int(eos_id)),
                        next_id,
                    )
                    finished = finished | (next_id[:, 0] == int(eos_id))
                generated = torch.cat((generated, next_id), dim=1)
                new_token_count += 1
                step = self.forward_step(next_id, state, collect_telemetry=False)
                state = step["state"]
                next_logits = step["logits"][:, -1]
                if eos_id is not None and bool(finished.all().cpu().item()):
                    break
            return {
                "generated_ids": generated,
                "state": state,
                "generated_token_count": new_token_count,
                "owned_by_marulho": True,
                "external_llm_used": False,
                "generation_decode": {
                    **self.generation_decode_policy(
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed,
                    ),
                    "repetition_penalty_adjusted_token_count": repetition_count,
                    "no_repeat_ngram_banned_token_count": banned_count,
                    "decode_control_fallback_count": fallback_count,
                },
            }
        finally:
            self.train(was_training)


def distributed_language_checkpoint_payload(
    model: MarulhoDistributedLanguageModel,
    tokenizer: LanguageTokenizer,
    *,
    metadata: Mapping[str, Any] | None = None,
    runtime_state: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    if int(model.config.vocab_size) != int(tokenizer.vocab_size):
        raise ValueError("checkpoint tokenizer/model vocabularies must match")
    return {
        "artifact_kind": "marulho_distributed_language_checkpoint",
        "surface": CHECKPOINT_SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "active_language_path": "candidate_not_installed",
        "config": asdict(model.config),
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "tokenizer": tokenizer.state_dict(),
        "tokenizer_hash": tokenizer.vocabulary_hash(),
        "metadata": dict(metadata or {}),
        "runtime_state": None
        if runtime_state is None
        else {key: value.detach().cpu() for key, value in runtime_state.items()},
    }


def save_distributed_language_checkpoint(
    path: str | Path,
    model: MarulhoDistributedLanguageModel,
    tokenizer: LanguageTokenizer,
    *,
    metadata: Mapping[str, Any] | None = None,
    runtime_state: Mapping[str, torch.Tensor] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = distributed_language_checkpoint_payload(
        model,
        tokenizer,
        metadata=metadata,
        runtime_state=runtime_state,
    )
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def load_distributed_language_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> tuple[
    MarulhoDistributedLanguageModel,
    LanguageTokenizer,
    dict[str, Any],
    dict[str, torch.Tensor] | None,
]:
    checkpoint = Path(path)
    payload = torch.load(checkpoint, map_location=map_location or "cpu")
    if payload.get("surface") != CHECKPOINT_SURFACE:
        raise ValueError("Rejected non-distributed candidate checkpoint")
    tokenizer = load_language_tokenizer_state(payload["tokenizer"])
    if str(payload.get("tokenizer_hash")) != tokenizer.vocabulary_hash():
        raise ValueError("Checkpoint tokenizer hash does not match its tokenizer state")
    config = DistributedLanguageConfig(**dict(payload["config"]))
    if int(config.vocab_size) != int(tokenizer.vocab_size):
        raise ValueError("Checkpoint tokenizer/model vocabularies do not match")
    model = MarulhoDistributedLanguageModel(config)
    model.load_state_dict(dict(payload["model_state"]), strict=True)
    runtime_payload = payload.get("runtime_state")
    runtime_state = (
        None
        if runtime_payload is None
        else {str(key): value for key, value in dict(runtime_payload).items()}
    )
    metadata = dict(payload.get("metadata") or {})
    metadata["checkpoint_sha256"] = _sha256_file(checkpoint)
    metadata["checkpoint_size_bytes"] = checkpoint.stat().st_size
    return model, tokenizer, metadata, runtime_state
