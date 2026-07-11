"""Causal segment-level associative state for the V14 language candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
from marulho.training.language_hashed_micro_experts import (
    HashedMicroExpertConfig,
    MarulhoHashedMicroExpertBlock,
    MarulhoHashedMicroExpertLanguageModel,
    MarulhoHashedMicroExpertStateBlock,
)
from marulho.training.language_transformer import TransformerRMSNorm


SEGMENT_ASSOCIATIVE_MODES = ("off", "local", "delta", "gated_delta")
_MODE_IDS = {name: index for index, name in enumerate(SEGMENT_ASSOCIATIVE_MODES)}
SEGMENT_ASSOCIATIVE_CHECKPOINT_SURFACE = (
    "marulho_segment_associative_language_checkpoint.v1"
)


@dataclass(frozen=True)
class SegmentAssociativeConfig:
    segment_length: int = 32
    memory_layer_index: int = 1
    memory_heads: int = 4
    key_width: int = 8
    value_width: int = 16
    mode: str = "gated_delta"
    retention_logit_bias: float = 4.0
    active_language_path: str = "marulho_segment_associative_state_v14"


def _validate_segment_config(
    config: SegmentAssociativeConfig,
    *,
    model_width: int,
    model_layers: int,
) -> None:
    if int(config.segment_length) < 2:
        raise ValueError("segment_length must be at least two")
    if not 0 <= int(config.memory_layer_index) < int(model_layers) - 1:
        raise ValueError("memory_layer_index must precede a later model layer")
    if int(config.memory_heads) < 1:
        raise ValueError("memory_heads must be positive")
    if int(config.key_width) < 1 or int(config.value_width) < 1:
        raise ValueError("associative key/value widths must be positive")
    if int(config.memory_heads) * int(config.value_width) > 2 * int(model_width):
        raise ValueError("associative read width is unreasonably large")
    if config.mode not in SEGMENT_ASSOCIATIVE_MODES:
        raise ValueError(f"mode must be one of {SEGMENT_ASSOCIATIVE_MODES}")
    if not math.isfinite(float(config.retention_logit_bias)):
        raise ValueError("retention_logit_bias must be finite")
    if not str(config.active_language_path).strip():
        raise ValueError("active_language_path is required")


class MarulhoSegmentAssociativeMemory(nn.Module):
    """Small content-addressed matrices updated from completed causal segments."""

    def __init__(self, width: int, config: SegmentAssociativeConfig) -> None:
        super().__init__()
        self.width = int(width)
        self.segment_length = int(config.segment_length)
        self.memory_heads = int(config.memory_heads)
        self.key_width = int(config.key_width)
        self.value_width = int(config.value_width)
        self.retention_logit_bias = float(config.retention_logit_bias)
        self.norm = TransformerRMSNorm(self.width)
        self.query = nn.Linear(
            self.width,
            self.memory_heads * self.key_width,
            bias=False,
        )
        self.key = nn.Linear(
            self.width,
            self.memory_heads * self.key_width,
            bias=False,
        )
        self.value = nn.Linear(
            self.width,
            self.memory_heads * self.value_width,
            bias=False,
        )
        self.write_gate = nn.Linear(self.width, self.memory_heads, bias=False)
        self.retention_gate = nn.Linear(self.width, self.memory_heads, bias=False)
        self.output = nn.Linear(
            self.memory_heads * self.value_width,
            self.width,
            bias=False,
        )
        self.register_buffer(
            "mode_id",
            torch.tensor(_MODE_IDS[config.mode], dtype=torch.long),
        )
        self._mode_name = str(config.mode)
        for projection in (self.query, self.key, self.value):
            nn.init.normal_(projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.write_gate.weight)
        nn.init.zeros_(self.retention_gate.weight)
        nn.init.zeros_(self.output.weight)

    def set_mode(self, mode: str) -> None:
        if mode not in SEGMENT_ASSOCIATIVE_MODES:
            raise ValueError(f"mode must be one of {SEGMENT_ASSOCIATIVE_MODES}")
        self.mode_id.fill_(_MODE_IDS[mode])
        self._mode_name = str(mode)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        return {
            "segment_memory": torch.zeros(
                int(batch_size),
                self.memory_heads,
                self.key_width,
                self.value_width,
                device=device,
                dtype=dtype,
            ),
            "segment_accumulator": torch.zeros(
                int(batch_size),
                self.width,
                device=device,
                dtype=dtype,
            ),
            "segment_count": torch.zeros((), device=device, dtype=torch.long),
        }

    def _token_query(
        self,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm(hidden)
        return F.normalize(
            self.query(normalized).view(
                *hidden.shape[:-1],
                self.memory_heads,
                self.key_width,
            ),
            dim=-1,
        )

    def _local_read(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden)
        query = F.normalize(
            self.query(normalized).view(
                *hidden.shape[:-1],
                self.memory_heads,
                self.key_width,
            ),
            dim=-1,
        )
        token_key = F.normalize(
            self.key(normalized).view(
                *hidden.shape[:-1],
                self.memory_heads,
                self.key_width,
            ),
            dim=-1,
        )
        token_value = self.value(normalized).view(
            *hidden.shape[:-1],
            self.memory_heads,
            self.value_width,
        )
        write = torch.sigmoid(self.write_gate(normalized)).unsqueeze(-1)
        retention = torch.sigmoid(
            self.retention_gate(normalized) + self.retention_logit_bias
        ).unsqueeze(-1)
        affinity = torch.sigmoid(
            (query * token_key).sum(dim=-1, keepdim=True)
            / math.sqrt(float(self.key_width))
        )
        return token_value * affinity * write * retention

    def _persistent_read(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim == 4:
            return torch.einsum("bthk,bhkv->bthv", query, memory)
        if query.ndim == 3:
            return torch.einsum("bhk,bhkv->bhv", query, memory)
        raise ValueError("associative query must be [batch,(time),head,key]")

    def _updated_memory(
        self,
        memory: torch.Tensor,
        summary: torch.Tensor,
        *,
        gated: bool,
    ) -> torch.Tensor:
        normalized = self.norm(summary)
        key = F.normalize(
            self.key(normalized).view(
                int(summary.shape[0]),
                self.memory_heads,
                self.key_width,
            ),
            dim=-1,
        )
        value = self.value(normalized).view(
            int(summary.shape[0]),
            self.memory_heads,
            self.value_width,
        )
        prediction = torch.einsum("bhk,bhkv->bhv", key, memory)
        correction = torch.einsum("bhk,bhv->bhkv", key, value - prediction)
        if not gated:
            return memory + correction
        write = torch.sigmoid(self.write_gate(normalized)).view(
            int(summary.shape[0]), self.memory_heads, 1, 1
        )
        retention = torch.sigmoid(
            self.retention_gate(normalized) + self.retention_logit_bias
        ).view(int(summary.shape[0]), self.memory_heads, 1, 1)
        return retention * memory + write * correction

    def _read_output(self, read: torch.Tensor) -> torch.Tensor:
        return self.output(read.flatten(start_dim=-2))

    def _fresh_forward(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, time_steps, _ = hidden.shape
        query = self._token_query(hidden)
        memory = self.initial_state(
            int(batch_size),
            device=hidden.device,
            dtype=hidden.dtype,
        )["segment_memory"]
        reads: list[torch.Tensor] = []
        accumulator = torch.zeros(
            int(batch_size), self.width, device=hidden.device, dtype=hidden.dtype
        )
        count = 0
        for start in range(0, int(time_steps), self.segment_length):
            end = min(int(time_steps), start + self.segment_length)
            reads.append(self._persistent_read(query[:, start:end], memory))
            segment = hidden[:, start:end]
            segment_count = end - start
            if segment_count == self.segment_length:
                memory = self._updated_memory(
                    memory,
                    segment.mean(dim=1),
                    gated=self._mode_name == "gated_delta",
                )
                accumulator = torch.zeros_like(accumulator)
                count = 0
            else:
                accumulator = segment.sum(dim=1)
                count = segment_count
        persistent_read = torch.cat(reads, dim=1)
        output = self._read_output(persistent_read)
        return hidden + output, {
            "segment_memory": memory.detach(),
            "segment_accumulator": accumulator.detach(),
            "segment_count": torch.tensor(
                count, device=hidden.device, dtype=torch.long
            ),
        }

    def _step_forward(
        self,
        hidden: torch.Tensor,
        state: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = int(hidden.shape[0])
        memory = state.get("segment_memory")
        accumulator = state.get("segment_accumulator")
        count = state.get("segment_count")
        if not isinstance(memory, torch.Tensor):
            initial = self.initial_state(
                batch_size,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            memory = initial["segment_memory"]
            accumulator = initial["segment_accumulator"]
            count = initial["segment_count"]
        assert isinstance(accumulator, torch.Tensor)
        assert isinstance(count, torch.Tensor)
        query = self._token_query(hidden)
        persistent_read = self._persistent_read(query, memory)
        output = self._read_output(persistent_read)
        next_accumulator = accumulator + hidden
        next_count = count + 1
        summary = next_accumulator / float(self.segment_length)
        updated = self._updated_memory(
            memory,
            summary,
            gated=self._mode_name == "gated_delta",
        )
        boundary = (next_count >= self.segment_length).to(hidden.dtype)
        memory_mask = boundary.view(1, 1, 1, 1)
        vector_mask = boundary.view(1, 1)
        next_memory = memory_mask * updated + (1.0 - memory_mask) * memory
        next_accumulator = (1.0 - vector_mask) * next_accumulator
        next_count = torch.where(
            next_count >= self.segment_length,
            torch.zeros_like(next_count),
            next_count,
        )
        return hidden + output, {
            "segment_memory": next_memory.detach(),
            "segment_accumulator": next_accumulator.detach(),
            "segment_count": next_count.detach(),
        }

    def forward(
        self,
        hidden: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if hidden.ndim != 3:
            raise ValueError("segment associative memory expects [batch,time,width]")
        if self._mode_name == "off":
            passthrough = (
                self.initial_state(
                    int(hidden.shape[0]),
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                if state is None
                else {
                    "segment_memory": state["segment_memory"].detach(),
                    "segment_accumulator": state["segment_accumulator"].detach(),
                    "segment_count": state["segment_count"].detach(),
                }
            )
            return hidden, passthrough
        if self._mode_name == "local":
            output = self._read_output(self._local_read(hidden))
            passthrough = (
                self.initial_state(
                    int(hidden.shape[0]),
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                if state is None
                else {
                    "segment_memory": state["segment_memory"].detach(),
                    "segment_accumulator": state["segment_accumulator"].detach(),
                    "segment_count": state["segment_count"].detach(),
                }
            )
            return hidden + output, passthrough
        if state is None:
            return self._fresh_forward(hidden)
        outputs: list[torch.Tensor] = []
        current = state
        for index in range(int(hidden.shape[1])):
            step, next_state = self._step_forward(hidden[:, index], current)
            outputs.append(step.unsqueeze(1))
            current = {**current, **next_state}
        return torch.cat(outputs, dim=1), dict(next_state)

    def state_bytes(self, batch_size: int, *, element_size: int = 4) -> int:
        return int(batch_size) * int(element_size) * (
            self.memory_heads * self.key_width * self.value_width + self.width
        ) + 8

    def theoretical_active_multiplies_per_token(self) -> int:
        """Return the mode-specific matrix multiply count, excluding norms."""

        width = self.width
        heads = self.memory_heads
        key = self.key_width
        value = self.value_width
        if self._mode_name == "off":
            return 0
        if self._mode_name == "local":
            return (
                2 * width * heads * key
                + 2 * width * heads * value
                + 2 * width * heads
                + heads * key
            )
        token_read = (
            width * heads * key
            + heads * key * value
            + width * heads * value
        )
        segment_update = (
            width * heads * key
            + width * heads * value
            + 2 * heads * key * value
        )
        if self._mode_name == "gated_delta":
            segment_update += 2 * width * heads + 2 * heads * key * value
        return token_read + math.ceil(segment_update / self.segment_length)

    @staticmethod
    def _binary_entropy(probability: torch.Tensor) -> torch.Tensor:
        bounded = probability.float().clamp(1.0e-6, 1.0 - 1.0e-6)
        return -(
            bounded * bounded.log()
            + (1.0 - bounded) * torch.log1p(-bounded)
        )

    @torch.no_grad()
    def diagnostic_report(self, hidden: torch.Tensor) -> dict[str, Any]:
        """Explain an already-trained state organ without reading targets."""

        if hidden.ndim != 3:
            raise ValueError("V14 diagnostic expects [batch,time,width] hidden state")
        batch_size, time_steps, _ = hidden.shape
        completed_segments = int(time_steps) // self.segment_length
        transformed, forward_state = self.forward(hidden, None)
        residual = transformed - hidden
        common: dict[str, Any] = {
            "surface": "marulho_segment_associative_state_diagnostic.v1",
            "mode": self._mode_name,
            "batch_size": int(batch_size),
            "time_steps": int(time_steps),
            "segment_length": self.segment_length,
            "completed_segment_count_per_sequence": completed_segments,
            "memory_state_bytes": self.state_bytes(
                int(batch_size), element_size=hidden.element_size()
            ),
            "theoretical_active_multiplies_per_token": (
                self.theoretical_active_multiplies_per_token()
            ),
            "residual_root_mean_square": float(
                residual.float().square().mean().sqrt().cpu()
            ),
            "write_policy_uses_labels": False,
            "promotion_metric": False,
            "external_llm_used": False,
        }
        if self._mode_name == "off":
            return {
                **common,
                "memory_update_count": 0,
                "mean_write_gate": None,
                "write_gate_binary_entropy": None,
                "write_frequency_at_or_above_half": None,
                "mean_retention_gate": None,
                "retention_gate_binary_entropy": None,
                "final_memory_frobenius_norm": 0.0,
                "final_memory_matrix_rank_mean": 0.0,
                "final_memory_matrix_effective_rank_mean": 0.0,
                "state_trajectory_effective_rank": 0.0,
                "state_perturbation_gain": 0.0,
            }
        normalized = self.norm(hidden)
        if self._mode_name == "local":
            write = torch.sigmoid(self.write_gate(normalized))
            retention = torch.sigmoid(
                self.retention_gate(normalized) + self.retention_logit_bias
            )
            return {
                **common,
                "memory_update_count": 0,
                "mean_write_gate": float(write.float().mean().cpu()),
                "write_gate_binary_entropy": float(
                    self._binary_entropy(write).mean().cpu()
                ),
                "write_frequency_at_or_above_half": float(
                    (write >= 0.5).float().mean().cpu()
                ),
                "mean_retention_gate": float(retention.float().mean().cpu()),
                "retention_gate_binary_entropy": float(
                    self._binary_entropy(retention).mean().cpu()
                ),
                "final_memory_frobenius_norm": 0.0,
                "final_memory_matrix_rank_mean": 0.0,
                "final_memory_matrix_effective_rank_mean": 0.0,
                "state_trajectory_effective_rank": 0.0,
                "state_perturbation_gain": 0.0,
            }

        def rollout(
            values: torch.Tensor,
        ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
            memory = self.initial_state(
                int(values.shape[0]),
                device=values.device,
                dtype=values.dtype,
            )["segment_memory"]
            memories: list[torch.Tensor] = []
            writes: list[torch.Tensor] = []
            retentions: list[torch.Tensor] = []
            for start in range(
                0,
                completed_segments * self.segment_length,
                self.segment_length,
            ):
                summary = values[:, start : start + self.segment_length].mean(
                    dim=1
                )
                summary_normalized = self.norm(summary)
                if self._mode_name == "gated_delta":
                    writes.append(torch.sigmoid(self.write_gate(summary_normalized)))
                    retentions.append(
                        torch.sigmoid(
                            self.retention_gate(summary_normalized)
                            + self.retention_logit_bias
                        )
                    )
                else:
                    writes.append(
                        torch.ones(
                            int(values.shape[0]),
                            self.memory_heads,
                            device=values.device,
                            dtype=values.dtype,
                        )
                    )
                    retentions.append(torch.ones_like(writes[-1]))
                memory = self._updated_memory(
                    memory,
                    summary,
                    gated=self._mode_name == "gated_delta",
                )
                memories.append(memory)
            return memory, memories, writes, retentions

        memory, memories, writes, retentions = rollout(hidden)
        if not memories:
            memory = forward_state["segment_memory"]
            memory_stack = memory.unsqueeze(0)
            write = torch.empty(0, device=hidden.device)
            retention = torch.empty(0, device=hidden.device)
        else:
            memory_stack = torch.stack(memories, dim=0)
            write = torch.stack(writes, dim=1)
            retention = torch.stack(retentions, dim=1)
        singular_values = torch.linalg.svdvals(memory.float())
        singular_probability = singular_values / singular_values.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-12)
        effective_rank = torch.exp(
            -(
                singular_probability
                * singular_probability.clamp_min(1.0e-12).log()
            ).sum(dim=-1)
        )
        trajectory = memory_stack.permute(1, 0, 2, 3, 4).reshape(
            int(batch_size), int(memory_stack.shape[0]), -1
        )
        trajectory_rank = torch.linalg.matrix_rank(trajectory.float())
        perturbed = hidden.clone()
        perturbation_size = 1.0e-3
        perturbed[0, 0, 0] += perturbation_size
        perturbed_memory, _history, _writes, _retentions = rollout(perturbed)
        perturbation_gain = (
            (perturbed_memory - memory).float().norm() / perturbation_size
        )
        return {
            **common,
            "memory_update_count": completed_segments * int(batch_size),
            "mean_write_gate": (
                None if write.numel() == 0 else float(write.float().mean().cpu())
            ),
            "write_gate_binary_entropy": (
                None
                if write.numel() == 0
                else float(self._binary_entropy(write).mean().cpu())
            ),
            "write_frequency_at_or_above_half": (
                None
                if write.numel() == 0
                else float((write >= 0.5).float().mean().cpu())
            ),
            "mean_retention_gate": (
                None
                if retention.numel() == 0
                else float(retention.float().mean().cpu())
            ),
            "retention_gate_binary_entropy": (
                None
                if retention.numel() == 0
                else float(self._binary_entropy(retention).mean().cpu())
            ),
            "final_memory_frobenius_norm": float(memory.float().norm().cpu()),
            "final_memory_matrix_rank_mean": float(
                torch.linalg.matrix_rank(memory.float()).float().mean().cpu()
            ),
            "final_memory_matrix_effective_rank_mean": float(
                effective_rank.mean().cpu()
            ),
            "state_trajectory_effective_rank": float(
                trajectory_rank.float().mean().cpu()
            ),
            "state_perturbation_gain": float(perturbation_gain.cpu()),
        }


class MarulhoSegmentAssociativeStateBlock(nn.Module):
    surface = "marulho_segment_associative_state_block.v1"

    def __init__(
        self,
        base: MarulhoHashedMicroExpertStateBlock,
        config: SegmentAssociativeConfig,
    ) -> None:
        super().__init__()
        _validate_segment_config(
            config,
            model_width=int(base.state_dim),
            model_layers=int(base.state_layers),
        )
        self.input_dim = int(base.input_dim)
        self.state_dim = int(base.state_dim)
        self.state_layers = int(base.state_layers)
        self.attention_heads = int(base.attention_heads)
        self.context_length = int(base.context_length)
        self.input_projection = base.input_projection
        self.layers = base.layers
        self.output_norm = base.output_norm
        self.expert_layer_index = int(base.expert_layer_index)
        self.memory_layer_index = int(config.memory_layer_index)
        self.associative = MarulhoSegmentAssociativeMemory(self.state_dim, config)

    @property
    def expert_layer(self) -> MarulhoHashedMicroExpertBlock:
        layer = self.layers[self.expert_layer_index]
        if not isinstance(layer, MarulhoHashedMicroExpertBlock):
            raise RuntimeError("Configured V14 layer is not a hashed expert block")
        return layer

    def set_mode(self, mode: str) -> None:
        self.expert_layer.set_mode(mode)

    def set_segment_mode(self, mode: str) -> None:
        self.associative.set_mode(mode)

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
        state.update(
            self.associative.initial_state(
                int(batch_size),
                device=device,
                dtype=dtype,
            )
        )
        return state

    def forward(
        self,
        inputs: torch.Tensor,
        route_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        forced_expert_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        if inputs.ndim != 3:
            raise ValueError("V14 state block expects [batch,time,input_dim]")
        if route_ids.shape != inputs.shape[:2]:
            raise ValueError("route_ids must match input batch/time dimensions")
        batch_size, time_steps, _ = inputs.shape
        if int(time_steps) > self.context_length and state is None:
            raise ValueError("V14 input exceeds Transformer context length")
        current_state = (
            self.initial_state(
                int(batch_size), device=inputs.device, dtype=inputs.dtype
            )
            if state is None
            else state
        )
        position = current_state.get("position")
        position_offset = (
            position.to(device=inputs.device, dtype=torch.long)
            if isinstance(position, torch.Tensor)
            else torch.zeros((), device=inputs.device, dtype=torch.long)
        )
        hidden = self.input_projection(inputs)
        next_state: dict[str, torch.Tensor] = {
            "position": position_offset + int(time_steps)
        }
        cache_tokens = 0
        memory_state: dict[str, torch.Tensor] | None = None
        for layer_index, layer in enumerate(self.layers):
            kwargs = {
                "past_key": current_state.get(f"layer_{layer_index}_key"),
                "past_value": current_state.get(f"layer_{layer_index}_value"),
                "position_offset": position_offset,
            }
            if isinstance(layer, MarulhoHashedMicroExpertBlock):
                hidden, next_key, next_value = layer(
                    hidden,
                    route_ids=route_ids,
                    forced_expert_ids=forced_expert_ids,
                    **kwargs,
                )
            else:
                hidden, next_key, next_value = layer(hidden, **kwargs)
            next_state[f"layer_{layer_index}_key"] = next_key.detach()
            next_state[f"layer_{layer_index}_value"] = next_value.detach()
            cache_tokens = int(next_key.shape[2])
            if layer_index == self.memory_layer_index:
                hidden, memory_state = self.associative(
                    hidden,
                    None if state is None else current_state,
                )
                next_state.update(memory_state)
        if memory_state is None:
            raise RuntimeError("V14 associative memory was not executed")
        hidden = self.output_norm(hidden)
        telemetry = {
            "surface": self.surface,
            "state_core": "transformer_hashed_micro_experts_segment_associative",
            "telemetry_collected": bool(collect_telemetry),
            "state_dim": self.state_dim,
            "state_layers": self.state_layers,
            "attention_heads": self.attention_heads,
            "context_length": self.context_length,
            "kv_cache_tokens": cache_tokens,
            "time_steps": int(time_steps),
            "normalization": "rmsnorm",
            "position_encoding": "rotary",
            "attention_backend": "torch_scaled_dot_product_attention",
            "expert_layer_index": self.expert_layer_index,
            "hashed_micro_expert_mode": self.expert_layer._mode_name,
            "segment_associative_mode": self.associative._mode_name,
            "segment_length": self.associative.segment_length,
            "memory_layer_index": self.memory_layer_index,
            "memory_heads": self.associative.memory_heads,
            "memory_key_width": self.associative.key_width,
            "memory_value_width": self.associative.value_width,
            "memory_state_bytes": self.associative.state_bytes(
                int(batch_size), element_size=inputs.element_size()
            ),
            "write_policy_uses_labels": False,
            "external_llm_used": False,
            "device": str(inputs.device),
        }
        return hidden, next_state, telemetry

    def step(
        self,
        token_input: torch.Tensor,
        route_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        if token_input.ndim != 2:
            raise ValueError("V14 step expects [batch,input_dim]")
        hidden, next_state, telemetry = self.forward(
            token_input.unsqueeze(1),
            route_ids.unsqueeze(1),
            state,
            collect_telemetry=collect_telemetry,
        )
        return hidden[:, 0], next_state, telemetry


class MarulhoSegmentAssociativeLanguageModel(
    MarulhoHashedMicroExpertLanguageModel
):
    surface = "marulho_segment_associative_language_model.v1"
    generation_surface = "marulho_segment_associative_generation.v1"

    def __init__(
        self,
        hashed_config: HashedMicroExpertConfig,
        segment_config: SegmentAssociativeConfig = SegmentAssociativeConfig(),
    ) -> None:
        self.segment_config = segment_config
        super().__init__(hashed_config)
        self.state_block = MarulhoSegmentAssociativeStateBlock(
            self.state_block,
            segment_config,
        )

    def set_segment_associative_mode(self, mode: str) -> None:
        self.state_block.set_segment_mode(mode)

    def segment_parameter_report(self) -> dict[str, Any]:
        parameters = sum(
            int(value.numel()) for value in self.state_block.associative.parameters()
        )
        return {
            "surface": "marulho_segment_associative_parameter_report.v1",
            "total_model_parameters": sum(
                int(value.numel()) for value in self.parameters()
            ),
            "segment_associative_parameters": parameters,
            "segment_length": self.state_block.associative.segment_length,
            "memory_heads": self.state_block.associative.memory_heads,
            "key_width": self.state_block.associative.key_width,
            "value_width": self.state_block.associative.value_width,
            "external_llm_used": False,
        }

    @torch.no_grad()
    def final_segment_gradient_report(self) -> dict[str, Any]:
        rows = []
        for name, parameter in self.state_block.associative.named_parameters():
            gradient = parameter.grad
            rows.append(
                {
                    "name": name,
                    "received_gradient": gradient is not None,
                    "nonzero_gradient_elements": (
                        0
                        if gradient is None
                        else int(torch.count_nonzero(gradient).cpu())
                    ),
                }
            )
        return {
            "surface": "marulho_segment_associative_gradient_report.v1",
            "mode": self.state_block.associative._mode_name,
            "parameters": rows,
            "all_parameters_received_gradient": all(
                row["received_gradient"] for row in rows
            ),
            "external_llm_used": False,
        }

    @torch.no_grad()
    def segment_diagnostic_report(self, input_ids: torch.Tensor) -> dict[str, Any]:
        captured: dict[str, torch.Tensor] = {}

        def capture(_module, inputs) -> None:
            captured["hidden"] = inputs[0].detach()

        handle = self.state_block.associative.register_forward_pre_hook(capture)
        was_training = self.training
        try:
            self.eval()
            self.forward(input_ids, collect_telemetry=False)
        finally:
            handle.remove()
            self.train(was_training)
        if "hidden" not in captured:
            raise RuntimeError("V14 diagnostic did not observe associative input")
        return {
            "surface": "marulho_segment_associative_diagnostic.v1",
            "mode": self.state_block.associative._mode_name,
            "gradient": self.final_segment_gradient_report(),
            "state": self.state_block.associative.diagnostic_report(
                captured["hidden"]
            ),
            "external_llm_used": False,
        }


def build_segment_associative_model(
    base_model: MarulhoHashedMicroExpertLanguageModel,
    config: SegmentAssociativeConfig = SegmentAssociativeConfig(),
) -> MarulhoSegmentAssociativeLanguageModel:
    if base_model.hashed_config.mode != "token_hash":
        raise ValueError("V14 requires the token_hash V11 base")
    hashed_config = replace(
        base_model.hashed_config,
        active_language_path=str(config.active_language_path),
    )
    model = MarulhoSegmentAssociativeLanguageModel(hashed_config, config)
    incompatible = model.load_state_dict(base_model.state_dict(), strict=False)
    expected_missing = {
        name
        for name in model.state_dict()
        if name.startswith("state_block.associative.")
    }
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(
            "V14 base load has unexpected missing tensors: "
            f"{incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"V14 base load has unexpected tensors: {incompatible.unexpected_keys}"
        )
    model.train(base_model.training)
    return model


def segment_associative_checkpoint_payload(
    model: MarulhoSegmentAssociativeLanguageModel,
    tokenizer: LanguageTokenizer,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if int(model.hashed_config.vocab_size) != int(tokenizer.vocab_size):
        raise ValueError("V14 checkpoint vocab must match its owned tokenizer")
    if model.state_block.expert_layer._mode_name != "token_hash":
        raise ValueError("V14 checkpoint requires token_hash mode")
    if model.state_block.associative._mode_name != "gated_delta":
        raise ValueError("Only gated_delta V14 can be checkpointed")
    return {
        "artifact_kind": "marulho_segment_associative_language_checkpoint",
        "surface": SEGMENT_ASSOCIATIVE_CHECKPOINT_SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "active_language_path": model.config.active_language_path,
        "hashed_config": asdict(model.hashed_config),
        "segment_config": asdict(model.segment_config),
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "tokenizer": tokenizer.state_dict(),
        "tokenizer_hash": tokenizer.vocabulary_hash(),
        "metadata": dict(metadata or {}),
    }


def save_segment_associative_checkpoint(
    path: str | Path,
    model: MarulhoSegmentAssociativeLanguageModel,
    tokenizer: LanguageTokenizer,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = segment_associative_checkpoint_payload(model, tokenizer, metadata)
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


def load_segment_associative_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> tuple[
    MarulhoSegmentAssociativeLanguageModel,
    LanguageTokenizer,
    dict[str, Any],
]:
    payload = torch.load(
        Path(path),
        map_location=map_location or "cpu",
        weights_only=False,
    )
    if payload.get("surface") != SEGMENT_ASSOCIATIVE_CHECKPOINT_SURFACE:
        raise ValueError("Rejected non-V14 segment associative checkpoint")
    if payload.get("owned_by_marulho") is not True:
        raise ValueError("V14 checkpoint must be MARULHO-owned")
    if payload.get("external_llm_used") is not False:
        raise ValueError("V14 checkpoint cannot use an external language model")
    tokenizer = load_language_tokenizer_state(payload["tokenizer"])
    if tokenizer.vocabulary_hash() != str(payload.get("tokenizer_hash")):
        raise ValueError("V14 checkpoint tokenizer hash mismatch")
    hashed_config = HashedMicroExpertConfig(**dict(payload["hashed_config"]))
    segment_config = SegmentAssociativeConfig(**dict(payload["segment_config"]))
    if hashed_config.mode != "token_hash" or segment_config.mode != "gated_delta":
        raise ValueError("V14 checkpoint modes are not promotable")
    model = MarulhoSegmentAssociativeLanguageModel(
        hashed_config,
        segment_config,
    )
    model.load_state_dict(dict(payload["model_state"]), strict=True)
    if model.lm_head.weight.data_ptr() != model.token_embedding.weight.data_ptr():
        raise RuntimeError("V14 checkpoint failed tied-embedding restoration")
    return model, tokenizer, dict(payload.get("metadata") or {})
