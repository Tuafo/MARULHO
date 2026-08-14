"""V75 adaptive-retention test-time-learning falsifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from .language_transformer import MarulhoTransformerBlock, TransformerRMSNorm


RetentionMode = Literal[
    "adaptive_own",
    "forced_open_own",
    "discard_same_compute",
    "adaptive_shuffled",
]


@dataclass(frozen=True)
class V75Config:
    context_length: int = 64
    segments: int = 3
    width: int = 128
    layers: int = 4
    attention_heads: int = 4
    mlp_width: int = 512
    rank: int = 8
    gate_width: int = 16
    key_count: int = 16
    value_count: int = 16
    distractor_count: int = 32

    @property
    def query_token(self) -> int:
        return 1

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
class V75Batch:
    tokens: torch.Tensor
    query_positions: torch.Tensor
    query_values: torch.Tensor


def make_v75_batch(
    config: V75Config,
    *,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str,
) -> V75Batch:
    """Build the frozen V74 three-segment contract without future leakage."""

    length = config.context_length + 1
    tokens = torch.randint(
        config.distractor_start,
        config.key_start,
        (int(batch_size), config.segments, length),
        generator=generator,
        dtype=torch.long,
    )
    key_order = torch.rand(int(batch_size), config.key_count, generator=generator)
    value_order = torch.rand(int(batch_size), config.value_count, generator=generator)
    keys = key_order.argsort(dim=1)[:, :4]
    values = value_order.argsort(dim=1)[:, :4]
    for repeat in range(8):
        for binding in range(4):
            position = 1 + (repeat * 8) + (binding * 2)
            tokens[:, 0, position] = config.key_start + keys[:, binding]
            tokens[:, 0, position + 1] = config.value_start + values[:, binding]
    query_positions: list[int] = []
    for binding in range(4):
        position = 1 + binding * 3
        tokens[:, 2, position] = config.query_token
        tokens[:, 2, position + 1] = config.key_start + keys[:, binding]
        tokens[:, 2, position + 2] = config.value_start + values[:, binding]
        query_positions.append(position + 1)
    return V75Batch(
        tokens=tokens.to(device),
        query_positions=torch.tensor(query_positions, device=device),
        query_values=(config.value_start + values).to(device),
    )


class V75AdaptiveTTT(nn.Module):
    """Small Transformer whose local gradient acceptance is meta-learned."""

    surface = "marulho_adaptive_ttt_v75.stage_a0.v1"

    def __init__(self, config: V75Config | None = None) -> None:
        super().__init__()
        self.config = config or V75Config()
        config = self.config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.layers = nn.ModuleList(
            MarulhoTransformerBlock(
                config.width,
                attention_heads=config.attention_heads,
                context_length=config.context_length,
                mlp_ratio=config.mlp_width / config.width,
                dropout=0.0,
            )
            for _ in range(config.layers)
        )
        self.output_norm = TransformerRMSNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.fast_a0 = nn.Parameter(torch.empty(config.mlp_width, config.rank))
        self.fast_b0 = nn.Parameter(torch.zeros(config.rank, config.width))
        initial_rate = torch.log(torch.expm1(torch.tensor(0.1)))
        self.inner_log_rate = nn.Parameter(initial_rate.reshape(()))
        self.retention_gate = nn.Sequential(
            nn.Linear(4, config.gate_width),
            nn.SiLU(),
            nn.Linear(config.gate_width, 1),
        )
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.fast_a0, mean=0.0, std=0.02)
        nn.init.zeros_(self.fast_b0)
        final_gate = self.retention_gate[-1]
        assert isinstance(final_gate, nn.Linear)
        nn.init.zeros_(final_gate.weight)
        nn.init.constant_(final_gate.bias, 2.0)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def initial_fast_weights(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.fast_a0.unsqueeze(0).expand(int(batch_size), -1, -1),
            self.fast_b0.unsqueeze(0).expand(int(batch_size), -1, -1),
        )

    def forward_segment(
        self,
        input_ids: torch.Tensor,
        fast_a: torch.Tensor,
        fast_b: torch.Tensor,
        *,
        fast_enabled: bool = True,
    ) -> torch.Tensor:
        hidden = self.token_embedding(input_ids.to(self.device, dtype=torch.long))
        last = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            attended, _, _ = layer.attention(
                layer.attention_norm(hidden),
                past_key=None,
                past_value=None,
                position_offset=0,
            )
            hidden = hidden + layer.dropout(attended)
            gate, up = layer.gate_up(layer.mlp_norm(hidden)).chunk(2, dim=-1)
            activated = F.silu(gate) * up
            update = layer.down(activated)
            if index == last and fast_enabled:
                low_rank = torch.einsum("bth,bhr->btr", activated, fast_a)
                update = update + torch.einsum("btr,brw->btw", low_rank, fast_b)
            hidden = hidden + layer.dropout(update)
        return self.lm_head(self.output_norm(hidden))

    def _gate_features(
        self,
        per_document_loss: torch.Tensor,
        fast_a: torch.Tensor,
        fast_b: torch.Tensor,
        candidate_a: torch.Tensor,
        candidate_b: torch.Tensor,
    ) -> torch.Tensor:
        base_a = self.fast_a0.detach().unsqueeze(0)
        base_b = self.fast_b0.detach().unsqueeze(0)
        state_a = fast_a.detach() - base_a
        state_b = fast_b.detach() - base_b
        direction_a = -candidate_a.detach()
        direction_b = -candidate_b.detach()
        gradient_energy = direction_a.square().mean((1, 2)) + direction_b.square().mean((1, 2))
        state_energy = state_a.square().mean((1, 2)) + state_b.square().mean((1, 2))
        dot = (direction_a * state_a).sum((1, 2)) + (direction_b * state_b).sum((1, 2))
        direction_norm = torch.sqrt(
            direction_a.square().sum((1, 2)) + direction_b.square().sum((1, 2))
        )
        state_norm = torch.sqrt(
            state_a.square().sum((1, 2)) + state_b.square().sum((1, 2))
        )
        cosine = dot / (direction_norm * state_norm).clamp_min(1.0e-12)
        cosine = torch.where(state_norm > 0.0, cosine, torch.zeros_like(cosine))
        return torch.stack(
            (
                torch.log1p(per_document_loss.detach()),
                torch.log1p(100.0 * torch.sqrt(gradient_energy + 1.0e-12)),
                cosine,
                torch.log1p(100.0 * torch.sqrt(state_energy + 1.0e-12)),
            ),
            dim=-1,
        )

    def _advance(
        self,
        fast_a: torch.Tensor,
        fast_b: torch.Tensor,
        grad_a: torch.Tensor,
        grad_b: torch.Tensor,
        per_document_loss: torch.Tensor,
        *,
        mode: RetentionMode,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode == "adaptive_shuffled":
            candidate_a, candidate_b = grad_a.roll(1, 0), grad_b.roll(1, 0)
        elif mode in {"adaptive_own", "forced_open_own", "discard_same_compute"}:
            candidate_a, candidate_b = grad_a, grad_b
        else:
            raise ValueError(f"Unsupported V75 mode: {mode}")
        features = self._gate_features(
            per_document_loss, fast_a, fast_b, candidate_a, candidate_b
        )
        raw_gate = torch.sigmoid(self.retention_gate(features).squeeze(-1))
        if mode == "forced_open_own":
            accepted_gate = torch.ones_like(raw_gate)
        elif mode == "discard_same_compute":
            accepted_gate = torch.zeros_like(raw_gate)
        else:
            accepted_gate = raw_gate
        rate = F.softplus(self.inner_log_rate)
        scale = accepted_gate[:, None, None]
        value_a = (
            fast_a.detach() - rate.detach() * scale.detach() * candidate_a.detach()
        ).detach()
        value_b = (
            fast_b.detach() - rate.detach() * scale.detach() * candidate_b.detach()
        ).detach()
        anchor_a = (self.fast_a0 - self.fast_a0.detach()).unsqueeze(0)
        anchor_b = (self.fast_b0 - self.fast_b0.detach()).unsqueeze(0)
        if mode == "discard_same_compute":
            return value_a + anchor_a, value_b + anchor_b, raw_gate, features
        rate_link = rate - rate.detach()
        gate_link = (accepted_gate - accepted_gate.detach())[:, None, None]
        return (
            value_a
            + anchor_a
            - rate_link * scale.detach() * candidate_a.detach()
            - rate.detach() * gate_link * candidate_a.detach(),
            value_b
            + anchor_b
            - rate_link * scale.detach() * candidate_b.detach()
            - rate.detach() * gate_link * candidate_b.detach(),
            raw_gate,
            features,
        )

    def episode(
        self,
        batch: V75Batch,
        *,
        mode: RetentionMode,
    ) -> dict[str, torch.Tensor]:
        batch_size = int(batch.tokens.shape[0])
        fast_a, fast_b = self.initial_fast_weights(batch_size)
        losses: list[torch.Tensor] = []
        update_norms: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        gate_features: list[torch.Tensor] = []
        query_logits: torch.Tensor | None = None
        for segment in range(self.config.segments):
            inputs = batch.tokens[:, segment, :-1]
            targets = batch.tokens[:, segment, 1:]
            logits = self.forward_segment(inputs, fast_a, fast_b)
            per_token = F.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), reduction="none"
            ).reshape(batch_size, self.config.context_length)
            per_document = per_token.mean(dim=1)
            loss = per_document.mean()
            grad_a, grad_b = torch.autograd.grad(
                per_document.sum(),
                (fast_a, fast_b),
                retain_graph=True,
                create_graph=False,
            )
            losses.append(loss)
            update_norms.append(
                torch.sqrt(
                    grad_a.detach().square().mean() + grad_b.detach().square().mean()
                )
            )
            if segment == self.config.segments - 1:
                query_logits = logits.index_select(1, batch.query_positions)
            fast_a, fast_b, raw_gate, features = self._advance(
                fast_a,
                fast_b,
                grad_a,
                grad_b,
                per_document,
                mode=mode,
            )
            gates.append(raw_gate)
            gate_features.append(features)
        assert query_logits is not None
        return {
            "loss": torch.stack(losses).mean(),
            "segment_losses": torch.stack([value.detach() for value in losses]),
            "query_logits": query_logits,
            "update_norms": torch.stack(update_norms),
            "gates": torch.stack(gates, dim=1),
            "gate_features": torch.stack(gate_features, dim=1),
            "final_fast_a": fast_a,
            "final_fast_b": fast_b,
            "inner_rate": F.softplus(self.inner_log_rate),
        }
