"""V76 Stage-A1 exact TTT over the retained 100M Transformer."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
import torch.nn.functional as F

from marulho.training.language_model import (
    MarulhoLanguageModel,
    language_model_state_sha256,
    load_language_model_checkpoint,
    load_language_model_state,
)


MetaGradient = Literal["exact", "first_order"]
UpdateMode = Literal["own", "discard", "shuffled"]
PARENT_CHECKPOINT_SHA256 = (
    "6caf97be17d49cd3fc70501b50cadd39897fd85000b121e107f13a5417a1068d"
)
PARENT_TOKENIZER_SHA256 = (
    "faca1e26aa29e897bef4e4335a0300f90e3996723d556a681b4495240f660715"
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_v76_language_parent(
    checkpoint: str | Path,
    *,
    context_length: int = 320,
) -> tuple[MarulhoLanguageModel, Any, dict[str, Any]]:
    """Strictly load V39 and extend only its shape-independent RoPE context."""

    checkpoint_path = Path(checkpoint)
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError(f"V76 parent checkpoint hash changed: {checkpoint_hash}")
    source, tokenizer, metadata = load_language_model_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    tokenizer_hash = tokenizer.vocabulary_hash()
    if tokenizer_hash != PARENT_TOKENIZER_SHA256:
        raise RuntimeError(f"V76 parent tokenizer hash changed: {tokenizer_hash}")
    source_hash = language_model_state_sha256(source)
    config = replace(
        source.config,
        transformer_context_length=int(context_length),
        active_language_path="marulho_transformer_v76_exact_ttt_100m",
    )
    extended = MarulhoLanguageModel(config)
    load_language_model_state(extended, source.state_dict())
    extended_hash = language_model_state_sha256(extended)
    if source_hash != extended_hash:
        raise RuntimeError("V76 context extension changed V39 tensor state")
    return extended, tokenizer, {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "tokenizer_sha256": tokenizer_hash,
        "source_model_state_sha256": source_hash,
        "extended_model_state_sha256": extended_hash,
        "source_context_length": int(source.config.transformer_context_length),
        "extended_context_length": int(context_length),
        "parameter_count": sum(parameter.numel() for parameter in extended.parameters()),
        "metadata": metadata,
    }


class V76ExactTTTLanguage(nn.Module):
    """Full V39 cortex plus per-document fast LoRA in its final quarter."""

    surface = "marulho_exact_ttt_v76.stage_a1_100m.v1"

    def __init__(
        self,
        base: MarulhoLanguageModel,
        *,
        rank: int = 8,
        fast_layer_indices: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        if self.rank < 1:
            raise ValueError("V76 rank must be positive")
        layer_count = len(base.state_block.layers)
        self.fast_layer_indices = fast_layer_indices or tuple(
            range(max(0, layer_count - 3), layer_count)
        )
        if not self.fast_layer_indices:
            raise ValueError("V76 requires at least one fast layer")
        if tuple(sorted(set(self.fast_layer_indices))) != self.fast_layer_indices:
            raise ValueError("V76 fast layer indices must be unique and sorted")
        if self.fast_layer_indices[0] < 0 or self.fast_layer_indices[-1] >= layer_count:
            raise ValueError("V76 fast layer index is outside the Transformer")
        self.fast_a0 = nn.ParameterList()
        self.fast_b0 = nn.ParameterList()
        for index in self.fast_layer_indices:
            layer = base.state_block.layers[index]
            hidden_width = int(layer.down.in_features)
            width = int(layer.down.out_features)
            self.fast_a0.append(nn.Parameter(torch.empty(hidden_width, self.rank)))
            self.fast_b0.append(nn.Parameter(torch.zeros(self.rank, width)))
        initial_rate = torch.log(torch.expm1(torch.tensor(0.1)))
        self.inner_log_rates = nn.Parameter(
            initial_rate.repeat(len(self.fast_layer_indices))
        )
        for value in self.fast_a0:
            nn.init.normal_(value, mean=0.0, std=0.02)
        for value in self.fast_b0:
            nn.init.zeros_(value)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def context_length(self) -> int:
        return int(self.base.config.transformer_context_length)

    def initial_fast_weights(
        self, batch_size: int
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        return (
            tuple(value.unsqueeze(0).expand(int(batch_size), -1, -1) for value in self.fast_a0),
            tuple(value.unsqueeze(0).expand(int(batch_size), -1, -1) for value in self.fast_b0),
        )

    def forward_segment(
        self,
        input_ids: torch.Tensor,
        fast_a: tuple[torch.Tensor, ...],
        fast_b: tuple[torch.Tensor, ...],
        *,
        fast_enabled: bool = True,
    ) -> torch.Tensor:
        if int(input_ids.shape[1]) > self.context_length:
            raise ValueError("V76 segment exceeds extended Transformer context")
        hidden = self.base.token_embedding(input_ids.to(self.device, dtype=torch.long))
        hidden = self.base.state_block.input_projection(hidden)
        fast_lookup = {index: slot for slot, index in enumerate(self.fast_layer_indices)}
        for index, layer in enumerate(self.base.state_block.layers):
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
            slot = fast_lookup.get(index)
            if slot is not None and fast_enabled:
                low_rank = torch.einsum("bth,bhr->btr", activated, fast_a[slot])
                update = update + torch.einsum(
                    "btr,brw->btw", low_rank, fast_b[slot]
                )
            hidden = hidden + layer.dropout(update)
        hidden = self.base.state_block.output_norm(hidden)
        return self.base.lm_head(hidden)

    def _advance(
        self,
        fast_a: tuple[torch.Tensor, ...],
        fast_b: tuple[torch.Tensor, ...],
        grad_a: tuple[torch.Tensor, ...],
        grad_b: tuple[torch.Tensor, ...],
        *,
        meta_gradient: MetaGradient,
        update_mode: UpdateMode,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        if update_mode == "own":
            used_a, used_b = grad_a, grad_b
        elif update_mode == "shuffled":
            used_a = tuple(value.roll(1, 0) for value in grad_a)
            used_b = tuple(value.roll(1, 0) for value in grad_b)
        elif update_mode == "discard":
            used_a = tuple(torch.zeros_like(value) for value in grad_a)
            used_b = tuple(torch.zeros_like(value) for value in grad_b)
        else:
            raise ValueError(f"Unsupported V76 update mode: {update_mode}")
        rates = F.softplus(self.inner_log_rates)
        if meta_gradient == "exact":
            return (
                tuple(
                    value - rates[index] * gradient
                    for index, (value, gradient) in enumerate(zip(fast_a, used_a, strict=True))
                ),
                tuple(
                    value - rates[index] * gradient
                    for index, (value, gradient) in enumerate(zip(fast_b, used_b, strict=True))
                ),
            )
        if meta_gradient != "first_order":
            raise ValueError(f"Unsupported V76 meta-gradient: {meta_gradient}")
        next_a: list[torch.Tensor] = []
        next_b: list[torch.Tensor] = []
        for index, (value_a, value_b, gradient_a, gradient_b) in enumerate(
            zip(fast_a, fast_b, used_a, used_b, strict=True)
        ):
            rate = rates[index]
            actual_a = (
                value_a.detach() - rate.detach() * gradient_a.detach()
            ).detach()
            actual_b = (
                value_b.detach() - rate.detach() * gradient_b.detach()
            ).detach()
            anchor_a = (self.fast_a0[index] - self.fast_a0[index].detach()).unsqueeze(0)
            anchor_b = (self.fast_b0[index] - self.fast_b0[index].detach()).unsqueeze(0)
            if update_mode == "discard":
                next_a.append(actual_a + anchor_a)
                next_b.append(actual_b + anchor_b)
            else:
                rate_link = rate - rate.detach()
                next_a.append(actual_a + anchor_a - rate_link * gradient_a.detach())
                next_b.append(actual_b + anchor_b - rate_link * gradient_b.detach())
        return tuple(next_a), tuple(next_b)

    def episode_documents(
        self,
        documents: torch.Tensor,
        *,
        meta_gradient: MetaGradient,
        update_mode: UpdateMode = "own",
        segment_length: int = 320,
    ) -> dict[str, torch.Tensor]:
        if documents.ndim != 2:
            raise ValueError("V76 documents must be [batch,tokens]")
        segment_length = int(segment_length)
        segment_count = (int(documents.shape[1]) - 1) // segment_length
        if segment_count != 3 or int(documents.shape[1]) != 3 * segment_length + 1:
            raise ValueError("V76 Stage A1 requires exactly three segments plus target")
        batch_size = int(documents.shape[0])
        fast_a, fast_b = self.initial_fast_weights(batch_size)
        losses: list[torch.Tensor] = []
        per_document_losses: list[torch.Tensor] = []
        update_norms: list[torch.Tensor] = []
        for segment in range(segment_count):
            start = segment * segment_length
            inputs = documents[:, start : start + segment_length]
            targets = documents[:, start + 1 : start + segment_length + 1]
            logits = self.forward_segment(inputs, fast_a, fast_b)
            per_token = F.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), reduction="none"
            ).reshape(batch_size, segment_length)
            per_document = per_token.mean(dim=1)
            losses.append(per_document.mean())
            per_document_losses.append(per_document.detach())
            if segment == segment_count - 1:
                break
            gradients = torch.autograd.grad(
                per_document.sum(),
                (*fast_a, *fast_b),
                retain_graph=True,
                create_graph=meta_gradient == "exact",
            )
            split = len(fast_a)
            grad_a = tuple(gradients[:split])
            grad_b = tuple(gradients[split:])
            update_norms.append(
                torch.sqrt(
                    sum(value.detach().square().mean() for value in gradients)
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
        return {
            "loss": torch.stack(losses).mean(),
            "segment_losses": torch.stack([value.detach() for value in losses]),
            "per_document_segment_losses": torch.stack(
                per_document_losses, dim=1
            ),
            "update_norms": torch.stack(update_norms),
            "final_fast_a": fast_a,
            "final_fast_b": fast_b,
            "inner_rates": F.softplus(self.inner_log_rates),
        }

    def static_documents(
        self,
        documents: torch.Tensor,
        *,
        segment_length: int = 320,
    ) -> dict[str, torch.Tensor]:
        fast_a, fast_b = self.initial_fast_weights(int(documents.shape[0]))
        losses: list[torch.Tensor] = []
        per_document_losses: list[torch.Tensor] = []
        for segment in range(3):
            start = segment * int(segment_length)
            inputs = documents[:, start : start + int(segment_length)]
            targets = documents[:, start + 1 : start + int(segment_length) + 1]
            logits = self.forward_segment(
                inputs, fast_a, fast_b, fast_enabled=False
            )
            per_token = F.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), reduction="none"
            ).reshape(int(documents.shape[0]), int(segment_length))
            per_document = per_token.mean(dim=1)
            losses.append(per_document.mean())
            per_document_losses.append(
                per_document.detach()
            )
        return {
            "loss": torch.stack(losses).mean(),
            "segment_losses": torch.stack([value.detach() for value in losses]),
            "per_document_segment_losses": torch.stack(
                per_document_losses, dim=1
            ),
        }
