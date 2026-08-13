"""V70 training-only macro-conditioned causal language candidate."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel
from marulho.training.language_transformer import TransformerRMSNorm, _apply_rotary


class MacroCausalSelfAttention(nn.Module):
    def __init__(self, width: int, *, heads: int, summaries: int = 4) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.summaries = summaries
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.summary_queries = nn.Parameter(
            torch.empty(heads, summaries, self.head_dim)
        )
        self.start_macro = nn.Parameter(torch.empty(heads, self.head_dim))
        self.query_macro_scale = nn.Parameter(torch.ones(heads, self.head_dim))
        self.output_macro_scale = nn.Parameter(torch.ones(heads, self.head_dim))
        nn.init.normal_(self.summary_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.start_macro, mean=0.0, std=0.02)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if value.ndim != 3 or value.shape[1] % 64:
            raise ValueError("macro attention expects [batch,time,width], time % 64 == 0")
        batch, time, _ = value.shape
        blocks = time // 64
        query, key, current_value = self.qkv(value).chunk(3, dim=-1)

        def local_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch, blocks, 64, self.heads, self.head_dim
            ).permute(0, 1, 3, 2, 4).reshape(
                batch * blocks, self.heads, 64, self.head_dim
            )

        query = local_heads(query)
        key = local_heads(key)
        current_value = local_heads(current_value)
        summary_query = self.summary_queries.unsqueeze(0).expand(
            batch * blocks, -1, -1, -1
        )
        block_summaries = F.scaled_dot_product_attention(
            summary_query, key, current_value, is_causal=False
        ).reshape(batch, blocks, self.heads, self.summaries, self.head_dim)
        summary_stream = block_summaries.permute(0, 2, 1, 3, 4).reshape(
            batch, self.heads, blocks * self.summaries, self.head_dim
        )
        summary_positions = torch.arange(
            blocks * self.summaries, device=value.device
        )
        summary_query_stream, summary_key_stream = _apply_rotary(
            summary_stream, summary_stream, summary_positions
        )
        global_summaries = F.scaled_dot_product_attention(
            summary_query_stream,
            summary_key_stream,
            summary_stream,
            is_causal=True,
        ).reshape(batch, self.heads, blocks, self.summaries, self.head_dim)

        completed_macro = global_summaries.mean(dim=3).permute(0, 2, 1, 3)
        initial = self.start_macro.unsqueeze(0).unsqueeze(1).expand(
            batch, 1, -1, -1
        )
        shifted_macro = torch.cat(
            (initial, completed_macro[:, : blocks - 1]), dim=1
        ).reshape(batch * blocks, self.heads, self.head_dim)
        query = query + (
            shifted_macro.unsqueeze(2)
            * self.query_macro_scale.unsqueeze(0).unsqueeze(2)
        )
        local_positions = torch.arange(64, device=value.device)
        query, key = _apply_rotary(query, key, local_positions)
        attended = F.scaled_dot_product_attention(
            query, key, current_value, is_causal=True
        )
        attended = attended.transpose(1, 2).reshape(batch, time, self.width)
        output = self.output(attended)
        scaled_macro = shifted_macro * self.output_macro_scale.unsqueeze(0)
        projected_macro = F.linear(
            scaled_macro.reshape(batch, blocks, self.width), self.output.weight
        )
        output_blocks = output.reshape(batch, blocks, 64, self.width)
        output_blocks.add_(projected_macro.unsqueeze(2))
        return output_blocks.reshape(batch, time, self.width), global_summaries


class MacroTransformerBlock(nn.Module):
    def __init__(self, config: LanguageModelConfig) -> None:
        super().__init__()
        width = int(config.state_dim)
        hidden_width = max(
            width, int(round(width * float(config.transformer_mlp_ratio)))
        )
        self.attention_norm = TransformerRMSNorm(width)
        self.attention = MacroCausalSelfAttention(
            width, heads=int(config.attention_heads)
        )
        self.mlp_norm = TransformerRMSNorm(width)
        self.gate_up = nn.Linear(width, 2 * hidden_width, bias=False)
        self.down = nn.Linear(hidden_width, width, bias=False)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention, summaries = self.attention(self.attention_norm(value))
        value = value + attention
        gate, up = self.gate_up(self.mlp_norm(value)).chunk(2, dim=-1)
        return value + self.down(F.silu(gate) * up), summaries


class MarulhoMacroCortexLanguageModel(nn.Module):
    """Training-only V70 model; no generation or checkpoint promotion surface."""

    surface = "marulho_macro_cortex_language_candidate.v1"

    def __init__(self, config: LanguageModelConfig) -> None:
        super().__init__()
        if int(config.embedding_dim) != int(config.state_dim):
            raise ValueError("V70 requires equal embedding and state width")
        if int(config.transformer_context_length) % 64:
            raise ValueError("V70 context must be divisible by 64")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.state_block = nn.Module()
        self.state_block.input_projection = nn.Identity()
        self.state_block.layers = nn.ModuleList(
            MacroTransformerBlock(config) for _ in range(int(config.state_layers))
        )
        self.state_block.output_norm = TransformerRMSNorm(config.state_dim)
        self.lm_head = nn.Linear(config.state_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        del collect_telemetry, decode_vocab_only
        if state:
            raise ValueError("V70 Phase 1 has no incremental state surface")
        hidden = self.token_embedding(input_ids.to(self.device, dtype=torch.long))
        last_summaries: torch.Tensor | None = None
        for layer in self.state_block.layers:
            hidden, last_summaries = layer(hidden)
        hidden = self.state_block.output_norm(hidden)
        return {
            "logits": self.lm_head(hidden),
            "state": {},
            "telemetry": {
                "surface": self.surface,
                "owned_by_marulho": True,
                "external_llm_used": False,
                "macro_summary_shape": tuple(last_summaries.shape),
            },
        }

    def next_token_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        collect_telemetry: bool = True,
        return_evidence: bool = True,
    ) -> dict[str, Any]:
        output = self.forward(input_ids, collect_telemetry=collect_telemetry)
        targets = target_ids.to(self.device, dtype=torch.long)
        logits = output["logits"]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return {
            "loss": loss,
            "loss_kind": "full_vocab_cross_entropy",
            "loss_evidence": (
                {"owned_by_marulho": True, "external_llm_used": False}
                if return_evidence
                else {}
            ),
            "state": {},
            "telemetry": output["telemetry"],
        }


def transfer_transformer_common_state(
    control: MarulhoLanguageModel,
    candidate: MarulhoMacroCortexLanguageModel,
) -> dict[str, Any]:
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    copied: list[str] = []
    candidate_only: list[str] = []
    for name, tensor in candidate_state.items():
        source = control_state.get(name)
        if source is not None and source.shape == tensor.shape:
            candidate_state[name] = source.detach().clone()
            copied.append(name)
        else:
            candidate_only.append(name)
    candidate.load_state_dict(candidate_state, strict=True)
    return {
        "copied_names": copied,
        "candidate_only_names": candidate_only,
        "copied_elements": sum(candidate_state[name].numel() for name in copied),
    }
