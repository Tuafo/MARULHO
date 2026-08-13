"""V72 training-only persistent cross-segment workspace falsifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from .language_transformer import TransformerRMSNorm


WorkspaceMode = Literal[
    "persistent",
    "reset_each_segment",
    "shuffled_document_state",
    "nonpersistent_same_compute",
]


@dataclass(frozen=True)
class V72RecallConfig:
    segment_length: int = 64
    segments: int = 3
    width: int = 64
    attention_heads: int = 2
    workspace_tokens: int = 8
    bindings: int = 4
    key_count: int = 16
    value_count: int = 16
    distractor_count: int = 32

    @property
    def write_token(self) -> int:
        return 2

    @property
    def no_write_token(self) -> int:
        return 3

    @property
    def distractor_start(self) -> int:
        return 4

    @property
    def key_start(self) -> int:
        return self.distractor_start + self.distractor_count

    @property
    def value_start(self) -> int:
        return self.key_start + self.key_count

    @property
    def vocab_size(self) -> int:
        return self.value_start + self.value_count


@dataclass(frozen=True)
class V72RecallBatch:
    segments: torch.Tensor
    query_values: torch.Tensor
    binding_keys: torch.Tensor
    binding_values: torch.Tensor


def make_v72_recall_batch(
    config: V72RecallConfig,
    *,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str,
) -> V72RecallBatch:
    """Create documents whose answer is absent after segment zero."""

    shape = (int(batch_size), config.segments, config.segment_length)
    tokens = torch.randint(
        config.distractor_start,
        config.key_start,
        shape,
        generator=generator,
        dtype=torch.long,
    )
    tokens[:, 0, 0] = config.write_token
    tokens[:, 1:, 0] = config.no_write_token

    key_scores = torch.rand(
        int(batch_size), config.key_count, generator=generator
    )
    value_scores = torch.rand(
        int(batch_size), config.value_count, generator=generator
    )
    keys = key_scores.argsort(dim=1)[:, : config.bindings]
    values = value_scores.argsort(dim=1)[:, : config.bindings]
    for binding in range(config.bindings):
        tokens[:, 0, 1 + (binding * 2)] = config.key_start + keys[:, binding]
        tokens[:, 0, 2 + (binding * 2)] = config.value_start + values[:, binding]

    query_binding = torch.randint(
        0,
        config.bindings,
        (int(batch_size),),
        generator=generator,
    )
    batch_indices = torch.arange(int(batch_size))
    query_keys = keys[batch_indices, query_binding]
    query_values = values[batch_indices, query_binding]
    tokens[:, -1, -2] = 1
    tokens[:, -1, -1] = config.key_start + query_keys
    return V72RecallBatch(
        segments=tokens.to(device),
        query_values=query_values.to(device),
        binding_keys=keys.to(device),
        binding_values=values.to(device),
    )


class _V72TokenBlock(nn.Module):
    def __init__(self, config: V72RecallConfig) -> None:
        super().__init__()
        self.norm_attention = TransformerRMSNorm(config.width)
        self.attention = nn.MultiheadAttention(
            config.width,
            config.attention_heads,
            batch_first=True,
            dropout=0.0,
            bias=False,
        )
        self.norm_mlp = TransformerRMSNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, config.width * 2, bias=False),
            nn.SiLU(),
            nn.Linear(config.width * 2, config.width, bias=False),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = int(tokens.shape[1])
        causal_mask = torch.ones(
            length,
            length,
            dtype=torch.bool,
            device=tokens.device,
        ).triu(1)
        normalized = self.norm_attention(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.mlp(self.norm_mlp(tokens))


class V72PersistentWorkspaceRecall(nn.Module):
    """Small owned A1 model; this is not an installed language runtime."""

    def __init__(self, config: V72RecallConfig | None = None) -> None:
        super().__init__()
        self.config = config or V72RecallConfig()
        config = self.config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Parameter(
            torch.empty(config.segment_length, config.width)
        )
        self.initial_workspace = nn.Parameter(
            torch.empty(config.workspace_tokens, config.width)
        )
        self.write_queries = nn.Parameter(
            torch.empty(config.workspace_tokens, config.width)
        )
        self.token_blocks = nn.ModuleList(
            [_V72TokenBlock(config), _V72TokenBlock(config)]
        )
        self.read_norms = nn.ModuleList(
            [TransformerRMSNorm(config.width), TransformerRMSNorm(config.width)]
        )
        self.reads = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    config.width,
                    config.attention_heads,
                    batch_first=True,
                    dropout=0.0,
                    bias=False,
                )
                for _ in range(2)
            ]
        )
        self.write_norm = TransformerRMSNorm(config.width)
        self.write_attention = nn.MultiheadAttention(
            config.width,
            config.attention_heads,
            batch_first=True,
            dropout=0.0,
            bias=False,
        )
        self.workspace_norm = TransformerRMSNorm(config.width)
        self.write_gate = nn.Linear(config.width, 1)
        self.answer_head = nn.Linear(config.width, config.value_count)
        self.key_reconstruction = nn.Linear(config.width, config.key_count)
        self.value_reconstruction = nn.Linear(config.width, config.value_count)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.initial_workspace, std=0.02)
        nn.init.normal_(self.write_queries, std=0.02)

    def initial_state(self, batch_size: int) -> torch.Tensor:
        return self.initial_workspace.unsqueeze(0).expand(
            int(batch_size), -1, -1
        )

    def boundary_state(
        self,
        state: torch.Tensor,
        mode: WorkspaceMode,
    ) -> torch.Tensor:
        if mode == "persistent":
            return state
        if mode == "reset_each_segment":
            return self.initial_state(int(state.shape[0]))
        if mode == "shuffled_document_state":
            return state.roll(shifts=1, dims=0)
        if mode == "nonpersistent_same_compute":
            return state.mean(dim=0, keepdim=True).expand_as(state)
        raise ValueError(f"Unsupported workspace mode: {mode}")

    def _read(
        self,
        tokens: torch.Tensor,
        workspace: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        normalized = self.read_norms[index](tokens)
        read, _ = self.reads[index](
            normalized,
            workspace,
            workspace,
            need_weights=False,
        )
        return tokens + read

    def process_segment(
        self,
        token_ids: torch.Tensor,
        workspace: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tokens = self.token_embedding(token_ids) + self.position_embedding.unsqueeze(0)
        tokens = self.token_blocks[0](tokens)
        tokens = self._read(tokens, workspace, 0)
        tokens = self.token_blocks[1](tokens)

        queries = self.write_queries.unsqueeze(0).expand(int(tokens.shape[0]), -1, -1)
        queries = queries + workspace
        candidate, _ = self.write_attention(
            queries,
            self.write_norm(tokens),
            self.write_norm(tokens),
            need_weights=False,
        )
        gate_logits = self.write_gate(tokens[:, 0]).squeeze(-1)
        gate = torch.sigmoid(gate_logits).view(-1, 1, 1)
        updated_workspace = self.workspace_norm(workspace + (gate * candidate))
        tokens = self._read(tokens, updated_workspace, 1)
        return {
            "logits": self.answer_head(tokens[:, -1]),
            "workspace": updated_workspace,
            "gate_logits": gate_logits,
            "key_logits": self.key_reconstruction(updated_workspace),
            "value_logits": self.value_reconstruction(updated_workspace),
        }

    def forward(
        self,
        segments: torch.Tensor,
        *,
        mode: WorkspaceMode,
    ) -> dict[str, torch.Tensor]:
        if segments.ndim != 3 or int(segments.shape[1]) != self.config.segments:
            raise ValueError("segments must have shape [batch, 3, segment_length]")
        workspace = self.initial_state(int(segments.shape[0]))
        segment_logits: list[torch.Tensor] = []
        gate_logits: list[torch.Tensor] = []
        first_key_logits: torch.Tensor | None = None
        first_value_logits: torch.Tensor | None = None
        states: list[torch.Tensor] = []
        for segment_index in range(self.config.segments):
            if segment_index > 0:
                workspace = self.boundary_state(workspace, mode)
            result = self.process_segment(segments[:, segment_index], workspace)
            segment_logits.append(result["logits"])
            gate_logits.append(result["gate_logits"])
            states.append(result["workspace"])
            if segment_index == 0:
                first_key_logits = result["key_logits"]
                first_value_logits = result["value_logits"]
            workspace = result["workspace"].detach()
        assert first_key_logits is not None and first_value_logits is not None
        return {
            "segment_logits": torch.stack(segment_logits, dim=1),
            "gate_logits": torch.stack(gate_logits, dim=1),
            "first_key_logits": first_key_logits,
            "first_value_logits": first_value_logits,
            "states": torch.stack(states, dim=1),
        }


def v72_recall_loss(
    result: dict[str, torch.Tensor],
    batch: V72RecallBatch,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    config_bindings = int(batch.binding_keys.shape[1])
    workspace_tokens = int(result["first_key_logits"].shape[1])
    repeats = workspace_tokens // config_bindings
    if repeats * config_bindings != workspace_tokens:
        raise ValueError("workspace token count must be divisible by binding count")
    key_targets = batch.binding_keys.repeat(1, repeats)
    value_targets = batch.binding_values.repeat(1, repeats)
    answer = F.cross_entropy(result["segment_logits"][:, -1], batch.query_values)
    key = F.cross_entropy(
        result["first_key_logits"].reshape(-1, result["first_key_logits"].shape[-1]),
        key_targets.reshape(-1),
    )
    value = F.cross_entropy(
        result["first_value_logits"].reshape(
            -1, result["first_value_logits"].shape[-1]
        ),
        value_targets.reshape(-1),
    )
    gate_targets = torch.zeros_like(result["gate_logits"])
    gate_targets[:, 0] = 1.0
    gate = F.binary_cross_entropy_with_logits(result["gate_logits"], gate_targets)
    total = answer + (0.5 * key) + (0.5 * value) + (0.1 * gate)
    return total, {"answer": answer, "key": key, "value": value, "gate": gate}

