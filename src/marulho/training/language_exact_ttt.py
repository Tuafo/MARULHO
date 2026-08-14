"""V76 exact versus first-order end-to-end TTT mechanism."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from .language_transformer import MarulhoTransformerBlock, TransformerRMSNorm


MetaGradient = Literal["exact", "first_order"]
UpdateMode = Literal["own", "discard", "shuffled"]


@dataclass(frozen=True)
class V76Config:
    context_length: int = 64
    segments: int = 3
    width: int = 128
    layers: int = 4
    attention_heads: int = 4
    mlp_width: int = 512
    rank: int = 8
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
class V76Batch:
    tokens: torch.Tensor
    query_positions: torch.Tensor
    query_values: torch.Tensor


def make_v76_batch(
    config: V76Config,
    *,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str,
) -> V76Batch:
    """Build V74's frozen fact/interference/query contract."""

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
    return V76Batch(
        tokens=tokens.to(device),
        query_positions=torch.tensor(query_positions, device=device),
        query_values=(config.value_start + values).to(device),
    )


class V76ExactTTT(nn.Module):
    """A Transformer whose inner next-token update can retain its exact graph."""

    surface = "marulho_exact_ttt_v76.stage_a0.v1"

    def __init__(self, config: V76Config | None = None) -> None:
        super().__init__()
        self.config = config or V76Config()
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
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fast_a0, mean=0.0, std=0.02)
        nn.init.zeros_(self.fast_b0)

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

    def _advance(
        self,
        fast_a: torch.Tensor,
        fast_b: torch.Tensor,
        grad_a: torch.Tensor,
        grad_b: torch.Tensor,
        *,
        meta_gradient: MetaGradient,
        update_mode: UpdateMode,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if update_mode == "own":
            used_a, used_b = grad_a, grad_b
        elif update_mode == "shuffled":
            used_a, used_b = grad_a.roll(1, 0), grad_b.roll(1, 0)
        elif update_mode == "discard":
            used_a, used_b = torch.zeros_like(grad_a), torch.zeros_like(grad_b)
        else:
            raise ValueError(f"Unsupported V76 update mode: {update_mode}")
        rate = F.softplus(self.inner_log_rate)
        if meta_gradient == "exact":
            return fast_a - rate * used_a, fast_b - rate * used_b
        if meta_gradient != "first_order":
            raise ValueError(f"Unsupported V76 meta-gradient: {meta_gradient}")
        value_a = (fast_a.detach() - rate.detach() * used_a.detach()).detach()
        value_b = (fast_b.detach() - rate.detach() * used_b.detach()).detach()
        anchor_a = (self.fast_a0 - self.fast_a0.detach()).unsqueeze(0)
        anchor_b = (self.fast_b0 - self.fast_b0.detach()).unsqueeze(0)
        if update_mode == "discard":
            return value_a + anchor_a, value_b + anchor_b
        rate_link = rate - rate.detach()
        return (
            value_a + anchor_a - rate_link * used_a.detach(),
            value_b + anchor_b - rate_link * used_b.detach(),
        )

    def episode(
        self,
        batch: V76Batch,
        *,
        meta_gradient: MetaGradient,
        update_mode: UpdateMode = "own",
    ) -> dict[str, torch.Tensor]:
        batch_size = int(batch.tokens.shape[0])
        fast_a, fast_b = self.initial_fast_weights(batch_size)
        losses: list[torch.Tensor] = []
        update_norms: list[torch.Tensor] = []
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
            losses.append(loss)
            if segment == self.config.segments - 1:
                query_logits = logits.index_select(1, batch.query_positions)
                break
            grad_a, grad_b = torch.autograd.grad(
                per_document.sum(),
                (fast_a, fast_b),
                retain_graph=True,
                create_graph=meta_gradient == "exact",
            )
            update_norms.append(
                torch.sqrt(
                    grad_a.detach().square().mean() + grad_b.detach().square().mean()
                )
            )
            fast_a, fast_b = self._advance(
                fast_a,
                fast_b,
                grad_a,
                grad_b,
                meta_gradient=meta_gradient,
                update_mode=update_mode,
            )
        assert query_logits is not None
        return {
            "loss": torch.stack(losses).mean(),
            "segment_losses": torch.stack([value.detach() for value in losses]),
            "query_logits": query_logits,
            "update_norms": torch.stack(update_norms),
            "final_fast_a": fast_a,
            "final_fast_b": fast_b,
            "inner_rate": F.softplus(self.inner_log_rate),
        }
