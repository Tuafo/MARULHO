"""Integrated persistent modular recurrent memory language candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import torch
from torch import nn
import torch.nn.functional as F

from marulho.data.language_tokenizer import LanguageTokenizer, load_language_tokenizer_state
from marulho.training.language_model import _apply_decode_controls


PMRM_FUSION_KINDS = (
    "temporal_only",
    "associative_only",
    "dual_parallel",
    "temporal_then_associative",
    "associative_then_temporal",
)
PMRM_EPISODIC_POLICIES = ("none", "surprise", "random", "recency", "full")
PMRM_CHECKPOINT_SURFACE = "marulho_pmrm_language_checkpoint.v1"


@dataclass(frozen=True)
class PMRMLanguageConfig:
    vocab_size: int
    embedding_dim: int = 256
    state_dim: int = 256
    column_count: int = 8
    active_columns: int = 2
    associative_dim: int = 32
    fusion_kind: str = "dual_parallel"
    relation_messages: bool = True
    episodic_policy: str = "surprise"
    episodic_slots: int = 16
    episodic_reads: int = 2
    workspace_registers: int = 2
    workspace_layers: int = 3
    workspace_iterations: int = 2
    workspace_mlp_dim: int = 1024
    context_length: int = 128
    router_load_balance: float = 0.01
    tie_embeddings: bool = True
    active_language_path: str = "marulho_pmrm_v0"


def _validate_config(config: PMRMLanguageConfig) -> None:
    if int(config.vocab_size) <= 1:
        raise ValueError("vocab_size must be greater than one")
    if int(config.embedding_dim) <= 0 or int(config.state_dim) <= 0:
        raise ValueError("embedding_dim and state_dim must be positive")
    if int(config.column_count) < 1:
        raise ValueError("column_count must be positive")
    if not 1 <= int(config.active_columns) <= int(config.column_count):
        raise ValueError("active_columns must be within the fixed column pool")
    if int(config.associative_dim) < 1:
        raise ValueError("associative_dim must be positive")
    if str(config.fusion_kind) not in PMRM_FUSION_KINDS:
        raise ValueError(f"fusion_kind must be one of {PMRM_FUSION_KINDS}")
    if str(config.episodic_policy) not in PMRM_EPISODIC_POLICIES:
        raise ValueError(
            f"episodic_policy must be one of {PMRM_EPISODIC_POLICIES}"
        )
    if int(config.episodic_slots) < 1:
        raise ValueError("episodic_slots must be positive")
    if not 1 <= int(config.episodic_reads) <= int(config.episodic_slots):
        raise ValueError("episodic_reads must be within the episodic slot budget")
    if (
        int(config.workspace_registers) < 1
        or int(config.workspace_layers) < 1
        or int(config.workspace_iterations) < 1
        or int(config.workspace_mlp_dim) < 1
    ):
        raise ValueError("workspace shape and iterations must be positive")
    if int(config.context_length) < 2:
        raise ValueError("context_length must be at least two")
    if not math.isfinite(float(config.router_load_balance)):
        raise ValueError("router_load_balance must be finite")
    if float(config.router_load_balance) < 0.0:
        raise ValueError("router_load_balance must be non-negative")
    if bool(config.tie_embeddings) and int(config.embedding_dim) != int(
        config.state_dim
    ):
        raise ValueError("tie_embeddings requires embedding_dim == state_dim")


class _PMRMWorkspaceLayer(nn.Module):
    def __init__(self, dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim * 2, dim)
        self.update = nn.Linear(dim * 2, dim)
        self.attention_norm = nn.RMSNorm(dim)
        self.mlp_norm = nn.RMSNorm(dim)
        self.mlp_in = nn.Linear(dim, mlp_dim * 2, bias=False)
        self.mlp_out = nn.Linear(mlp_dim, dim, bias=False)

    def forward(
        self,
        workspace: torch.Tensor,
        sources: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.einsum(
            "brd,bsd->brs",
            self.query(workspace),
            self.key(sources),
        ) / math.sqrt(int(workspace.shape[-1]))
        context = torch.einsum(
            "brs,bsd->brd",
            torch.softmax(scores, dim=-1),
            self.value(sources),
        )
        joined = torch.cat((workspace, context), dim=-1)
        gate = torch.sigmoid(self.gate(joined))
        proposal = torch.tanh(self.update(joined))
        workspace = self.attention_norm(
            workspace + gate * (proposal - workspace)
        )
        value, multiplier = self.mlp_in(self.mlp_norm(workspace)).chunk(2, dim=-1)
        return workspace + self.mlp_out(F.silu(multiplier) * value)


class MarulhoPMRMLanguageModel(nn.Module):
    """One coherent PMRM model with internal ablation switches."""

    surface = "marulho_pmrm_language_model.v1"

    def __init__(self, config: PMRMLanguageConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        dim = int(config.state_dim)
        assoc = int(config.associative_dim)
        columns = int(config.column_count)

        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.event_projection = nn.Linear(config.embedding_dim, dim, bias=False)
        self.event_norm = nn.RMSNorm(dim)
        self.column_keys = nn.Parameter(torch.empty(columns, dim))

        self.temporal_candidate = nn.Linear(dim * 2, dim)
        self.temporal_gate = nn.Linear(dim * 2, dim)
        self.associative_key = nn.Linear(dim, assoc, bias=False)
        self.associative_value = nn.Linear(dim, assoc, bias=False)
        self.associative_write = nn.Linear(dim, assoc)
        self.associative_erase = nn.Linear(dim, assoc)
        self.associative_decay_logits = nn.Parameter(torch.zeros(columns, assoc))
        self.associative_readout = nn.Linear(assoc, dim, bias=False)
        self.temporal_bridge = nn.Linear(dim, dim, bias=False)
        self.associative_bridge = nn.Linear(dim, dim, bias=False)
        self.relation_projection = nn.Linear(dim, dim, bias=False)
        self.dual_fusion = nn.Linear(dim * 2, dim, bias=False)

        self.episodic_key = nn.Linear(dim, assoc, bias=False)
        self.episodic_query = nn.Linear(dim, assoc, bias=False)
        self.episodic_gate = nn.Linear(dim * 2, dim)

        registers = int(config.workspace_registers)
        self.workspace_initial = nn.Parameter(torch.empty(registers, dim))
        self.workspace_layers = nn.ModuleList(
            _PMRMWorkspaceLayer(dim, int(config.workspace_mlp_dim))
            for _ in range(int(config.workspace_layers))
        )
        self.output_norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, config.vocab_size, bias=False)
        if bool(config.tie_embeddings):
            self.lm_head.weight = self.token_embedding.weight

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(getattr(module, "bias", None), torch.Tensor):
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.column_keys, mean=0.0, std=0.02)
        nn.init.normal_(self.workspace_initial, mean=0.0, std=0.02)
        with torch.no_grad():
            self.associative_decay_logits.fill_(4.0)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def context_length(self) -> int:
        return int(self.config.context_length)

    @property
    def generation_vocab_size(self) -> int:
        return int(self.config.vocab_size)

    def init_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = max(1, int(batch_size))
        target_device = torch.device(device) if device is not None else self.device
        target_dtype = dtype or self.token_embedding.weight.dtype
        columns = int(self.config.column_count)
        dim = int(self.config.state_dim)
        assoc = int(self.config.associative_dim)
        slots = int(self.config.episodic_slots)
        registers = int(self.config.workspace_registers)
        zeros = lambda *shape: torch.zeros(
            (batch, *shape), device=target_device, dtype=target_dtype
        )
        return {
            "temporal": zeros(columns, dim),
            "associative": zeros(columns, assoc, assoc),
            "column_usage": zeros(columns),
            "episodes_key": zeros(slots, assoc),
            "episodes_value": zeros(slots, dim),
            "episodes_score": torch.full(
                (batch, slots),
                float("-inf"),
                device=target_device,
                dtype=target_dtype,
            ),
            "episodes_valid": torch.zeros(
                (batch, slots), device=target_device, dtype=torch.bool
            ),
            "write_pointer": torch.zeros(
                (batch,), device=target_device, dtype=torch.long
            ),
            "pending_key": zeros(assoc),
            "pending_value": zeros(dim),
            "pending_prediction": zeros(dim),
            "pending_valid": torch.zeros(
                (batch,), device=target_device, dtype=torch.bool
            ),
            "step_index": torch.zeros(
                (batch,), device=target_device, dtype=torch.long
            ),
            "episodic_considered": torch.zeros(
                (batch,), device=target_device, dtype=torch.long
            ),
            "episodic_writes": torch.zeros(
                (batch,), device=target_device, dtype=torch.long
            ),
            "episodic_reads": torch.zeros(
                (batch,), device=target_device, dtype=torch.long
            ),
            "router_scores": torch.zeros(
                (batch,), device=target_device, dtype=torch.long
            ),
            "active_column_updates": torch.zeros(
                (batch,), device=target_device, dtype=torch.long
            ),
            "relation_messages": torch.zeros(
                (batch,), device=target_device, dtype=torch.long
            ),
            "workspace_updates": torch.zeros(
                (batch,), device=target_device, dtype=torch.long
            ),
        }

    def reset_state(
        self,
        state: Mapping[str, torch.Tensor],
        reset_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask = reset_mask.to(device=self.device, dtype=torch.bool).reshape(-1)
        if not state:
            return self.init_state(int(mask.numel()))
        fresh = self.init_state(
            int(mask.numel()),
            device=self.device,
            dtype=state["temporal"].dtype,
        )
        reset: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if int(value.shape[0]) != int(mask.numel()):
                raise ValueError("PMRM state batch dimension does not match reset mask")
            view = mask.reshape(mask.shape[0], *([1] * (value.ndim - 1)))
            reset[key] = torch.where(view, fresh[key], value)
        return reset

    def serialize_state(
        self, state: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in state.items()}

    def load_state(
        self, serialized: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if "temporal" not in serialized:
            raise ValueError("Serialized PMRM state is missing temporal state")
        batch = int(serialized["temporal"].shape[0])
        expected = self.init_state(batch)
        if set(serialized) != set(expected):
            raise ValueError("Serialized PMRM state keys do not match the state contract")
        loaded: dict[str, torch.Tensor] = {}
        for key, target in expected.items():
            value = serialized[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(f"Serialized PMRM state shape mismatch for {key}")
            loaded[key] = value.to(device=self.device, dtype=target.dtype)
        return loaded

    def _write_pending(
        self,
        state: Mapping[str, torch.Tensor],
        observed_token_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        next_state = dict(state)
        pending = state["pending_valid"]
        next_state["episodic_considered"] = (
            state["episodic_considered"] + pending.to(dtype=torch.long)
        )
        policy = str(self.config.episodic_policy)
        if policy == "none":
            return next_state

        with torch.no_grad():
            if policy == "surprise":
                logits = self.lm_head(state["pending_prediction"])
                priority = F.cross_entropy(
                    logits,
                    observed_token_ids,
                    reduction="none",
                )
            elif policy == "random":
                phase = (
                    (state["step_index"].to(dtype=torch.float32) + 1.0) * 12.9898
                    + observed_token_ids.to(dtype=torch.float32) * 78.233
                )
                priority = torch.remainder(torch.sin(phase) * 43758.5453, 1.0)
            else:
                priority = state["step_index"].to(dtype=torch.float32)
            priority = priority.to(dtype=state["episodes_score"].dtype)

            valid = state["episodes_valid"]
            has_free = ~valid.all(dim=1)
            first_free = (~valid).to(dtype=torch.long).argmax(dim=1)
            minimum_scores = torch.where(
                valid,
                state["episodes_score"],
                torch.full_like(state["episodes_score"], float("inf")),
            )
            minimum_index = minimum_scores.argmin(dim=1)
            minimum_value = minimum_scores.gather(1, minimum_index.unsqueeze(1)).squeeze(1)
            if policy in {"recency", "full"}:
                destination = torch.remainder(
                    state["write_pointer"], int(self.config.episodic_slots)
                )
                should_write = pending
            else:
                destination = torch.where(has_free, first_free, minimum_index)
                should_write = pending & (has_free | (priority > minimum_value))

        slot_mask = F.one_hot(
            destination,
            num_classes=int(self.config.episodic_slots),
        ).to(dtype=torch.bool)
        slot_mask = slot_mask & should_write.unsqueeze(1)
        next_state["episodes_key"] = torch.where(
            slot_mask.unsqueeze(-1),
            state["pending_key"].unsqueeze(1),
            state["episodes_key"],
        )
        next_state["episodes_value"] = torch.where(
            slot_mask.unsqueeze(-1),
            state["pending_value"].unsqueeze(1),
            state["episodes_value"],
        )
        next_state["episodes_score"] = torch.where(
            slot_mask,
            priority.unsqueeze(1),
            state["episodes_score"],
        )
        next_state["episodes_valid"] = state["episodes_valid"] | slot_mask
        next_state["episodic_writes"] = (
            state["episodic_writes"] + should_write.to(dtype=torch.long)
        )
        if policy in {"recency", "full"}:
            next_state["write_pointer"] = state["write_pointer"] + should_write.to(
                dtype=torch.long
            )
        return next_state

    def _retrieve(
        self,
        event: torch.Tensor,
        state: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(event.shape[0])
        if str(self.config.episodic_policy) == "none":
            return torch.zeros_like(event), torch.zeros(
                batch, device=event.device, dtype=torch.long
            )
        query = F.normalize(self.episodic_query(event), dim=-1)
        keys = F.normalize(state["episodes_key"], dim=-1)
        similarities = torch.einsum("bsa,ba->bs", keys, query)
        valid = state["episodes_valid"]
        has_any = valid.any(dim=1)
        similarities = similarities.masked_fill(~valid, float("-inf"))
        similarities = torch.where(
            has_any.unsqueeze(1), similarities, torch.zeros_like(similarities)
        )
        read_count = min(
            int(self.config.episodic_reads), int(self.config.episodic_slots)
        )
        selected_scores, selected_indices = torch.topk(
            similarities, k=read_count, dim=1
        )
        selected_valid = valid.gather(1, selected_indices)
        weights = torch.softmax(selected_scores, dim=1) * selected_valid.to(
            dtype=event.dtype
        )
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        values = state["episodes_value"].gather(
            1,
            selected_indices.unsqueeze(-1).expand(-1, -1, int(self.config.state_dim)),
        )
        retrieved = torch.einsum("br,brd->bd", weights, values)
        return retrieved, selected_valid.sum(dim=1).to(dtype=torch.long)

    def _route(
        self,
        event: torch.Tensor,
        state: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        event_direction = F.normalize(event, dim=-1)
        key_direction = F.normalize(self.column_keys, dim=-1)
        scores = torch.einsum("bd,cd->bc", event_direction, key_direction)
        if float(self.config.router_load_balance) > 0.0:
            denominator = state["step_index"].to(dtype=event.dtype).unsqueeze(1) + 1.0
            scores = scores - float(self.config.router_load_balance) * (
                state["column_usage"].detach() / denominator
            )
        selected_scores, selected_indices = torch.topk(
            scores, k=int(self.config.active_columns), dim=1
        )
        selected_weights = torch.softmax(selected_scores, dim=1)
        route_weights = torch.zeros_like(scores).scatter(
            1, selected_indices, selected_weights
        )
        active = torch.zeros_like(scores, dtype=torch.bool).scatter(
            1, selected_indices, True
        )
        return route_weights, active

    def _temporal_update(
        self,
        source: torch.Tensor,
        temporal: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        columns = int(self.config.column_count)
        expanded = source.unsqueeze(1).expand(-1, columns, -1)
        joined = torch.cat((expanded, temporal), dim=-1)
        candidate = torch.tanh(self.temporal_candidate(joined))
        gate = torch.sigmoid(self.temporal_gate(joined))
        proposed = temporal + gate * (candidate - temporal)
        updated = torch.where(active.unsqueeze(-1), proposed, temporal)
        return updated, updated

    def _associative_update(
        self,
        source: torch.Tensor,
        associative: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        column_event = source.unsqueeze(1) + self.column_keys.unsqueeze(0)
        key = F.normalize(self.associative_key(column_event), dim=-1)
        value = self.associative_value(column_event)
        previous_read = torch.einsum("bcij,bcj->bci", associative, key)
        residual = value - previous_read
        write = torch.sigmoid(self.associative_write(column_event))
        erase = torch.sigmoid(self.associative_erase(column_event))
        delta = torch.einsum("bci,bcj->bcij", write * residual, erase * key)
        decay = torch.sigmoid(self.associative_decay_logits).unsqueeze(0).unsqueeze(-1)
        proposed = associative * decay + delta
        updated = torch.where(active.unsqueeze(-1).unsqueeze(-1), proposed, associative)
        read = torch.einsum("bcij,bcj->bci", updated, key)
        return updated, self.associative_readout(read)

    @staticmethod
    def _aggregate(values: torch.Tensor, route_weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bc,bcd->bd", route_weights, values)

    def _update_columns(
        self,
        event: torch.Tensor,
        state: Mapping[str, torch.Tensor],
        route_weights: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temporal = state["temporal"]
        associative = state["associative"]
        relation_source = self._aggregate(temporal, route_weights)
        if bool(self.config.relation_messages) and int(self.config.active_columns) > 1:
            event = event + self.relation_projection(relation_source)
        kind = str(self.config.fusion_kind)

        if kind == "associative_only":
            associative, associative_columns = self._associative_update(
                event, associative, active
            )
            fused = self._aggregate(associative_columns, route_weights)
        elif kind == "temporal_only":
            temporal, temporal_columns = self._temporal_update(
                event, temporal, active
            )
            fused = self._aggregate(temporal_columns, route_weights)
        elif kind == "associative_then_temporal":
            associative, associative_columns = self._associative_update(
                event, associative, active
            )
            associative_read = self._aggregate(associative_columns, route_weights)
            temporal, temporal_columns = self._temporal_update(
                event + self.associative_bridge(associative_read),
                temporal,
                active,
            )
            temporal_read = self._aggregate(temporal_columns, route_weights)
            fused = self.dual_fusion(torch.cat((temporal_read, associative_read), dim=-1))
        else:
            temporal, temporal_columns = self._temporal_update(event, temporal, active)
            temporal_read = self._aggregate(temporal_columns, route_weights)
            associative_source = event
            if kind == "temporal_then_associative":
                associative_source = event + self.temporal_bridge(temporal_read)
            associative, associative_columns = self._associative_update(
                associative_source, associative, active
            )
            associative_read = self._aggregate(associative_columns, route_weights)
            if kind == "dual_parallel":
                temporal_read = self.temporal_bridge(temporal_read)
                associative_read = self.associative_bridge(associative_read)
            fused = self.dual_fusion(torch.cat((temporal_read, associative_read), dim=-1))
        return temporal, associative, event + fused

    def _workspace_step(
        self,
        fused: torch.Tensor,
        retrieved: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sources = torch.stack((fused, retrieved), dim=1)
        workspace = self.workspace_initial.to(
            device=fused.device, dtype=fused.dtype
        ).unsqueeze(0).expand(int(fused.shape[0]), -1, -1)
        workspace = workspace + fused.unsqueeze(1)
        for _ in range(int(self.config.workspace_iterations)):
            for layer in self.workspace_layers:
                workspace = layer(workspace, sources)
        hidden = self.output_norm(fused + workspace.mean(dim=1))
        return workspace, hidden

    def _telemetry(self, state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        total_steps = int(state["step_index"].sum().detach().cpu().item())
        writes = int(state["episodic_writes"].sum().detach().cpu().item())
        considered = int(
            state["episodic_considered"].sum().detach().cpu().item()
        )
        state_bytes = {
            key: int(value.numel()) * int(value.element_size())
            for key, value in state.items()
        }
        episode_keys = {
            "episodes_key",
            "episodes_value",
            "episodes_score",
            "episodes_valid",
            "write_pointer",
        }
        column_keys = {"temporal", "associative", "column_usage"}
        return {
            "surface": "marulho_pmrm_execution_telemetry.v1",
            "active_language_path": self.config.active_language_path,
            "external_llm_used": False,
            "owned_by_marulho": True,
            "fusion_kind": self.config.fusion_kind,
            "episodic_policy": self.config.episodic_policy,
            "fixed_column_count": int(self.config.column_count),
            "active_columns_per_event": int(self.config.active_columns),
            "dense_router_scoring_reported": True,
            "router_score_count": int(
                state["router_scores"].sum().detach().cpu().item()
            ),
            "active_column_update_count": int(
                state["active_column_updates"].sum().detach().cpu().item()
            ),
            "relation_message_count": int(
                state["relation_messages"].sum().detach().cpu().item()
            ),
            "episodic_capacity_entries_per_stream": int(
                self.config.episodic_slots
            ),
            "episodic_capacity_entries_total": int(
                state["episodes_valid"].shape[0]
            )
            * int(self.config.episodic_slots),
            "episodic_valid_entries": int(
                state["episodes_valid"].sum().detach().cpu().item()
            ),
            "episodic_considered_count": considered,
            "episodic_write_count": writes,
            "episodic_write_fraction": float(writes) / max(1, considered),
            "episodic_read_count": int(
                state["episodic_reads"].sum().detach().cpu().item()
            ),
            "workspace_registers": int(self.config.workspace_registers),
            "workspace_layers": int(self.config.workspace_layers),
            "workspace_iterations_per_event": int(
                self.config.workspace_iterations
            ),
            "workspace_update_count": int(
                state["workspace_updates"].sum().detach().cpu().item()
            ),
            "processed_event_count": total_steps,
            "column_state_bytes_per_batch": sum(
                state_bytes[key] for key in column_keys
            ),
            "episodic_state_bytes_per_batch": sum(
                state_bytes[key] for key in episode_keys
            ),
            "total_runtime_state_bytes_per_batch": sum(state_bytes.values()),
        }

    def _core_step(
        self,
        token_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        tokens = token_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        batch = int(tokens.shape[0])
        current = (
            self.init_state(batch, device=self.device)
            if state is None
            else dict(state)
        )
        if int(current["temporal"].shape[0]) != batch:
            raise ValueError("PMRM state batch size must match token_ids")
        current = self._write_pending(current, tokens)

        event = self.event_norm(self.event_projection(self.token_embedding(tokens)))
        route_weights, active = self._route(event, current)
        temporal, associative, fused = self._update_columns(
            event,
            current,
            route_weights,
            active,
        )
        retrieved, read_count = self._retrieve(fused, current)
        episode_gate = torch.sigmoid(
            self.episodic_gate(torch.cat((fused, retrieved), dim=-1))
        )
        fused_with_memory = fused + episode_gate * retrieved

        next_state = dict(current)
        next_state.update(
            {
                "temporal": temporal,
                "associative": associative,
                "column_usage": current["column_usage"]
                + active.to(dtype=current["column_usage"].dtype),
                "pending_key": F.normalize(self.episodic_key(fused), dim=-1).to(
                    dtype=current["pending_key"].dtype
                ),
                "pending_value": fused.to(dtype=current["pending_value"].dtype),
                "pending_prediction": fused.to(
                    dtype=current["pending_prediction"].dtype
                ),
                "pending_valid": torch.ones(
                    batch, device=self.device, dtype=torch.bool
                ),
                "step_index": current["step_index"] + 1,
                "episodic_reads": current["episodic_reads"] + read_count,
                "router_scores": current["router_scores"]
                + int(self.config.column_count),
                "active_column_updates": current["active_column_updates"]
                + int(self.config.active_columns),
                "relation_messages": current["relation_messages"]
                + (
                    int(self.config.active_columns)
                    * max(0, int(self.config.active_columns) - 1)
                    if bool(self.config.relation_messages)
                    else 0
                ),
                "workspace_updates": current["workspace_updates"]
                + int(self.config.workspace_registers)
                * int(self.config.workspace_layers)
                * int(self.config.workspace_iterations),
            }
        )
        return {
            "workspace_input": fused_with_memory,
            "retrieved": retrieved,
            "state": next_state,
        }

    def step(
        self,
        token_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> dict[str, Any]:
        core = self._core_step(token_ids, state)
        _, hidden = self._workspace_step(
            core["workspace_input"], core["retrieved"]
        )
        next_state = core["state"]
        return {
            "logits": self.lm_head(hidden),
            "hidden": hidden,
            "state": next_state,
            "telemetry": self._telemetry(next_state) if collect_telemetry else {},
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
        if input_ids.ndim == 2:
            if int(input_ids.shape[1]) != 1:
                raise ValueError("forward_step expects [batch] or [batch, 1]")
            tokens = input_ids[:, 0]
        elif input_ids.ndim == 1:
            tokens = input_ids
        else:
            raise ValueError("forward_step expects [batch] or [batch, 1]")
        result = self.step(tokens, state, collect_telemetry=collect_telemetry)
        return {
            "logits": result["logits"].unsqueeze(1),
            "state": result["state"],
            "telemetry": result["telemetry"],
        }

    def scan(
        self,
        input_ids: torch.Tensor,
        initial_state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("PMRM scan expects input_ids shaped [batch, time]")
        if int(input_ids.shape[1]) > int(self.config.context_length):
            raise ValueError("PMRM scan exceeds the configured training context")
        runtime_ids = input_ids.to(device=self.device, dtype=torch.long)
        state = initial_state
        workspace_inputs: list[torch.Tensor] = []
        retrieved_values: list[torch.Tensor] = []
        for time_index in range(int(runtime_ids.shape[1])):
            core = self._core_step(runtime_ids[:, time_index], state)
            state = core["state"]
            workspace_inputs.append(core["workspace_input"])
            retrieved_values.append(core["retrieved"])
        assert state is not None
        batch, time = runtime_ids.shape
        workspace_input = torch.stack(workspace_inputs, dim=1).reshape(
            int(batch) * int(time), int(self.config.state_dim)
        )
        retrieved = torch.stack(retrieved_values, dim=1).reshape(
            int(batch) * int(time), int(self.config.state_dim)
        )
        _, flat_hidden = self._workspace_step(workspace_input, retrieved)
        hidden = flat_hidden.reshape(
            int(batch), int(time), int(self.config.state_dim)
        )
        return {
            "logits": self.lm_head(hidden),
            "hidden": hidden,
            "state": state,
            "telemetry": self._telemetry(state) if collect_telemetry else {},
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
        return self.scan(
            input_ids,
            initial_state=state,
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
        output = self.forward(input_ids, collect_telemetry=collect_telemetry)
        targets = target_ids.to(device=self.device, dtype=torch.long)
        logits = output["logits"]
        if logits.shape[:2] != targets.shape:
            raise ValueError("target_ids must match input batch/time dimensions")
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        evidence = {
            "surface": "marulho_pmrm_cross_entropy.v1",
            "sampled_vocab_training": False,
            "full_vocab_logits_materialized": True,
            "target_token_count": int(targets.numel()),
            "surprise_write_uses_only_observed_next_token": True,
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
        sampling = float(temperature) > 0.0
        return {
            "surface": "marulho_pmrm_decode_policy.v1",
            "decode_strategy": "nucleus_sampling" if sampling else "greedy_argmax",
            "model_vocab_size": int(self.config.vocab_size),
            "generation_vocab_size": int(self.config.vocab_size),
            "full_model_vocab_logits_materialized": True,
            "repetition_penalty": max(1.0, float(repetition_penalty)),
            "no_repeat_ngram_size": max(0, int(no_repeat_ngram_size)),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "sampling_seed": None if seed is None else int(seed),
            "top_p_applied": bool(sampling and float(top_p) < 1.0),
            "state_cache": "persistent_columns_associative_episodes_workspace",
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
        temperature = float(temperature)
        top_p = float(top_p)
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be finite and non-negative")
        if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be finite and in (0, 1]")
        if prompt_ids.ndim == 1:
            generated = prompt_ids.unsqueeze(0)
        elif prompt_ids.ndim == 2:
            generated = prompt_ids
        else:
            raise ValueError("prompt_ids must be [time] or [batch, time]")
        generated = generated.to(device=self.device, dtype=torch.long)
        if int(generated.shape[1]) < 1:
            raise ValueError("generation requires at least one prompt token")
        if int(generated.shape[1]) > int(self.config.context_length):
            raise ValueError("generation prompt exceeds the configured context")

        sample = temperature > 0.0
        sampling_generator = None
        if sample and seed is not None:
            sampling_generator = torch.Generator(device=self.device)
            sampling_generator.manual_seed(int(seed))
        was_training = bool(self.training)
        self.eval()
        try:
            prompt_result = self.forward(generated, collect_telemetry=False)
            state = prompt_result["state"]
            next_logits = prompt_result["logits"][:, -1]
            new_token_count = 0
            repetition_count = 0
            banned_count = 0
            fallback_count = 0
            finished = torch.zeros(
                generated.shape[0], device=self.device, dtype=torch.bool
            )
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
                if sample:
                    probabilities = torch.softmax(controlled / temperature, dim=-1)
                    if top_p < 1.0:
                        sorted_probabilities, sorted_indices = torch.sort(
                            probabilities, dim=-1, descending=True
                        )
                        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
                        remove = cumulative > top_p
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
                if eos_id is not None and bool(finished.all().item()):
                    break
                step_result = self.forward_step(
                    next_id, state, collect_telemetry=False
                )
                state = step_result["state"]
                next_logits = step_result["logits"][:, -1]
            return {
                "surface": "marulho_pmrm_generation.v1",
                "generated_ids": generated,
                "new_token_count": int(new_token_count),
                "state": state,
                "active_language_path": self.config.active_language_path,
                "external_llm_used": False,
                "owned_by_marulho": True,
                "generation_decode": self.generation_decode_policy(
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                ),
                "decode_control_totals": {
                    "repetition_penalty_adjusted_token_count": repetition_count,
                    "no_repeat_ngram_banned_token_count": banned_count,
                    "decode_control_fallback_count": fallback_count,
                },
                "telemetry": self._telemetry(state),
            }
        finally:
            self.train(was_training)


def pmrm_language_checkpoint_payload(
    model: MarulhoPMRMLanguageModel,
    tokenizer: LanguageTokenizer,
    *,
    metadata: Mapping[str, Any] | None = None,
    runtime_state: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    if int(model.config.vocab_size) != int(tokenizer.vocab_size):
        raise ValueError("PMRM checkpoint vocabulary must match its tokenizer")
    return {
        "artifact_kind": "marulho_pmrm_language_checkpoint",
        "surface": PMRM_CHECKPOINT_SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "active_language_path": model.config.active_language_path,
        "config": asdict(model.config),
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "tokenizer": tokenizer.state_dict(),
        "tokenizer_hash": tokenizer.vocabulary_hash(),
        "runtime_state": (
            None if runtime_state is None else model.serialize_state(runtime_state)
        ),
        "metadata": dict(metadata or {}),
    }


def save_pmrm_language_checkpoint(
    path: str | Path,
    model: MarulhoPMRMLanguageModel,
    tokenizer: LanguageTokenizer,
    *,
    metadata: Mapping[str, Any] | None = None,
    runtime_state: Mapping[str, torch.Tensor] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = pmrm_language_checkpoint_payload(
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


def load_pmrm_language_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> tuple[
    MarulhoPMRMLanguageModel,
    LanguageTokenizer,
    dict[str, Any],
    dict[str, torch.Tensor] | None,
]:
    payload = torch.load(Path(path), map_location=map_location or "cpu")
    if payload.get("surface") != PMRM_CHECKPOINT_SURFACE:
        raise ValueError("Checkpoint is not a PMRM language checkpoint")
    tokenizer = load_language_tokenizer_state(payload["tokenizer"])
    config = PMRMLanguageConfig(**dict(payload["config"]))
    if int(config.vocab_size) != int(tokenizer.vocab_size):
        raise ValueError("PMRM checkpoint vocabulary does not match its tokenizer")
    model = MarulhoPMRMLanguageModel(config)
    model.load_state_dict(dict(payload["model_state"]), strict=True)
    runtime_payload = payload.get("runtime_state")
    runtime_state = None
    if isinstance(runtime_payload, Mapping):
        runtime_state = model.load_state(runtime_payload)
    return model, tokenizer, dict(payload.get("metadata") or {}), runtime_state
