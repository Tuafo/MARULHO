"""V73 training-only exact-Transformer adaptive document sidecar."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn
import torch.nn.functional as F

from .language_model import LanguageModelConfig, MarulhoLanguageModel
from .language_transformer import MarulhoTransformerBlock, TransformerRMSNorm


SidecarMode = Literal["persistent", "reset", "shuffled"]


class _V73Read(nn.Module):
    def __init__(self, width: int, state_width: int, heads: int) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(state_width) // int(heads)
        self.query = nn.Linear(width, state_width, bias=False)
        self.key = nn.Linear(state_width, state_width, bias=False)
        self.value = nn.Linear(state_width, state_width, bias=False)
        self.output = nn.Linear(state_width, width, bias=False)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, time, width = value.shape
        return value.reshape(batch, time, self.heads, width // self.heads).transpose(1, 2)

    def forward(self, hidden: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        query = self._heads(self.query(hidden))
        key = self._heads(self.key(state))
        value = self._heads(self.value(state))
        read = F.scaled_dot_product_attention(query, key, value, is_causal=False)
        read = read.transpose(1, 2).contiguous().flatten(2)
        return self.output(read)


class _V73Write(nn.Module):
    def __init__(self, width: int, state_width: int, heads: int) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(state_width) // int(heads)
        self.query = nn.Linear(state_width, state_width, bias=False)
        self.key = nn.Linear(width, state_width, bias=False)
        self.value = nn.Linear(width, state_width, bias=False)
        self.output = nn.Linear(state_width, state_width, bias=False)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, time, width = value.shape
        return value.reshape(batch, time, self.heads, width // self.heads).transpose(1, 2)

    def forward(self, query_state: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        query = self._heads(self.query(query_state))
        key = self._heads(self.key(hidden))
        value = self._heads(self.value(hidden))
        written = F.scaled_dot_product_attention(query, key, value, is_causal=False)
        written = written.transpose(1, 2).contiguous().flatten(2)
        return self.output(written)


class V73ExactCortexSidecarLanguageModel(nn.Module):
    """Full Transformer plus temporary state; no runtime/checkpoint surface."""

    surface = "marulho_exact_cortex_sidecar_language_candidate.v1"
    read_layer_indices = (2, 6)

    def __init__(
        self,
        config: LanguageModelConfig,
        *,
        state_tokens: int = 8,
        state_width: int = 256,
        state_heads: int = 4,
    ) -> None:
        super().__init__()
        if int(config.state_layers) != 10:
            raise ValueError("V73 requires exactly ten Transformer layers")
        if int(config.embedding_dim) != int(config.state_dim):
            raise ValueError("V73 requires equal embedding and state widths")
        if not bool(config.tie_embeddings):
            raise ValueError("V73 requires tied embeddings")
        if int(state_width) % int(state_heads):
            raise ValueError("V73 state width must divide into state heads")
        self.config = config
        self.state_tokens = int(state_tokens)
        self.state_width = int(state_width)
        width = int(config.state_dim)
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.state_block = nn.Module()
        self.state_block.layers = nn.ModuleList(
            MarulhoTransformerBlock(
                width,
                attention_heads=config.attention_heads,
                context_length=config.transformer_context_length,
                mlp_ratio=config.transformer_mlp_ratio,
                dropout=config.transformer_dropout,
            )
            for _ in range(config.state_layers)
        )
        self.state_block.output_norm = TransformerRMSNorm(width)
        self.lm_head = nn.Linear(width, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.sidecar_initial = nn.Parameter(torch.empty(self.state_tokens, state_width))
        self.sidecar_write_queries = nn.Parameter(
            torch.empty(self.state_tokens, state_width)
        )
        self.sidecar_read_norms = nn.ModuleList(
            TransformerRMSNorm(width) for _ in self.read_layer_indices
        )
        self.sidecar_read = _V73Read(width, state_width, state_heads)
        self.sidecar_read_gates = nn.Parameter(torch.zeros(len(self.read_layer_indices)))
        self.sidecar_write_norm = TransformerRMSNorm(width)
        self.sidecar_write = _V73Write(width, state_width, state_heads)
        self.sidecar_content_gate = nn.Linear(width, 1)
        self.sidecar_state_norm = TransformerRMSNorm(state_width)
        self.sidecar_decode = nn.Linear(state_width, width, bias=False)

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                bias = getattr(module, "bias", None)
                if isinstance(bias, torch.Tensor):
                    nn.init.zeros_(bias)
        nn.init.normal_(self.sidecar_initial, mean=0.0, std=0.02)
        nn.init.normal_(self.sidecar_write_queries, mean=0.0, std=0.02)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def initial_state(self, batch_size: int) -> torch.Tensor:
        return self.sidecar_initial.unsqueeze(0).expand(int(batch_size), -1, -1)

    def boundary_state(self, state: torch.Tensor, mode: SidecarMode) -> torch.Tensor:
        if mode == "persistent":
            return state
        if mode == "reset":
            return self.initial_state(int(state.shape[0]))
        if mode == "shuffled":
            return state.roll(1, dims=0)
        raise ValueError(f"Unsupported V73 sidecar mode: {mode}")

    def forward_segment(
        self,
        input_ids: torch.Tensor,
        state: torch.Tensor,
        *,
        sidecar_enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("V73 inputs must be [batch,time]")
        hidden = self.token_embedding(input_ids.to(self.device, dtype=torch.long))
        read_index = 0
        for layer_index, layer in enumerate(self.state_block.layers):
            hidden, _, _ = layer(
                hidden,
                past_key=None,
                past_value=None,
                position_offset=0,
            )
            if layer_index in self.read_layer_indices:
                read = self.sidecar_read(
                    self.sidecar_read_norms[read_index](hidden), state
                )
                if sidecar_enabled:
                    hidden = hidden + self.sidecar_read_gates[read_index] * read
                read_index += 1
        hidden = self.state_block.output_norm(hidden)
        logits = self.lm_head(hidden)

        write_hidden = self.sidecar_write_norm(hidden.detach())
        queries = self.sidecar_write_queries.unsqueeze(0).expand(
            int(hidden.shape[0]), -1, -1
        )
        candidate = self.sidecar_write(queries + state, write_hidden)
        gate = torch.sigmoid(
            self.sidecar_content_gate(write_hidden.mean(dim=1))
        ).unsqueeze(-1)
        next_state = self.sidecar_state_norm(state + gate * candidate)
        decoded = self.sidecar_decode(next_state)
        workspace_logits = F.linear(decoded, self.lm_head.weight.detach())
        return {
            "logits": logits,
            "next_state": next_state,
            "workspace_logits": workspace_logits,
            "content_gate": gate.squeeze(-1).squeeze(-1),
        }


def transfer_v73_transformer_state(
    control: MarulhoLanguageModel,
    candidate: V73ExactCortexSidecarLanguageModel,
) -> dict[str, Any]:
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    copied: list[str] = []
    sidecar: list[str] = []
    for name, value in candidate_state.items():
        source = control_state.get(name)
        if source is not None and source.shape == value.shape:
            candidate_state[name] = source.detach().clone()
            copied.append(name)
        else:
            sidecar.append(name)
    candidate.load_state_dict(candidate_state, strict=True)
    return {
        "copied_names": copied,
        "sidecar_names": sidecar,
        "copied_tensor_count": len(copied),
        "sidecar_tensor_count": len(sidecar),
    }

