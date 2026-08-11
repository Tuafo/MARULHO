"""Frozen-cortex source-copy path for MARULHO extractive grounding."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import torch
import torch.nn.functional as F
from torch import nn

from marulho.training.language_answer_objective import answer_target_mask
from marulho.training.language_model import MarulhoLanguageModel, _apply_decode_controls


SOURCE_POINTER_CHECKPOINT_SURFACE = "marulho_source_pointer_checkpoint.v1"


def _first_marker_start(input_ids: torch.Tensor, marker_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    size = int(marker_ids.numel())
    matches = input_ids.unfold(1, size, 1).eq(marker_ids).all(dim=-1)
    found = matches.any(dim=1)
    starts = matches.to(dtype=torch.int64).argmax(dim=1)
    return starts, found


def structural_source_mask(
    input_ids: torch.Tensor,
    *,
    context_marker_ids: torch.Tensor,
    question_marker_ids: torch.Tensor,
) -> torch.Tensor:
    """Select tokens inside the first explicit Context field."""

    context_start, has_context = _first_marker_start(input_ids, context_marker_ids)
    question_start, has_question = _first_marker_start(input_ids, question_marker_ids)
    context_end = context_start + int(context_marker_ids.numel())
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    valid = has_context & has_question & question_start.gt(context_end)
    return (
        positions.ge(context_end.unsqueeze(1))
        & positions.lt(question_start.unsqueeze(1))
        & valid.unsqueeze(1)
    )


class FrozenSourcePointerLanguageModel(nn.Module):
    """Mix frozen vocabulary probability with a learned source-token pointer."""

    surface = "marulho_frozen_source_pointer_language_model.v1"

    def __init__(
        self,
        base: MarulhoLanguageModel,
        *,
        context_marker_ids: torch.Tensor,
        question_marker_ids: torch.Tensor,
        pointer_rank: int = 64,
    ) -> None:
        super().__init__()
        if int(pointer_rank) < 1:
            raise ValueError("pointer_rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        width = int(base.config.state_dim)
        self.pointer_rank = int(pointer_rank)
        self.query = nn.Linear(width, self.pointer_rank, bias=False)
        self.key = nn.Linear(width, self.pointer_rank, bias=False)
        self.copy_gate = nn.Linear(width, 1, bias=True)
        nn.init.normal_(self.query.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.key.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.copy_gate.weight)
        nn.init.constant_(self.copy_gate.bias, -1.0)
        self.register_buffer(
            "context_marker_ids",
            context_marker_ids.detach().to(dtype=torch.long),
        )
        self.register_buffer(
            "question_marker_ids",
            question_marker_ids.detach().to(dtype=torch.long),
        )

    @property
    def device(self) -> torch.device:
        return self.query.weight.device

    @property
    def context_length(self) -> int:
        return self.base.context_length

    @property
    def generation_vocab_size(self) -> int:
        return self.base.generation_vocab_size

    def pointer_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("base.")
        )

    def pointer_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in self.state_dict().items()
            if not name.startswith("base.")
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        del state, decode_vocab_only
        runtime_ids = input_ids.to(device=self.device, dtype=torch.long)
        with torch.no_grad():
            base_result = self.base._forward_hidden(
                runtime_ids,
                collect_telemetry=False,
            )
            hidden = base_result["hidden"].detach()
            base_logits = self.base.lm_head(hidden).detach()
        query = self.query(hidden)
        key = self.key(hidden)
        scores = torch.einsum("btr,bsr->bts", query, key) / math.sqrt(
            float(self.pointer_rank)
        )
        source = structural_source_mask(
            runtime_ids,
            context_marker_ids=self.context_marker_ids,
            question_marker_ids=self.question_marker_ids,
        )
        positions = torch.arange(runtime_ids.shape[1], device=self.device)
        causal = positions.unsqueeze(0).le(positions.unsqueeze(1))
        allowed = source.unsqueeze(1) & causal.unsqueeze(0)
        attention = torch.softmax(scores.float().masked_fill(~allowed, -1.0e4), dim=-1)
        attention = attention * allowed.to(dtype=attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        copy_probability = torch.zeros(
            runtime_ids.shape[0],
            runtime_ids.shape[1],
            self.generation_vocab_size,
            device=self.device,
            dtype=attention.dtype,
        )
        copy_probability.scatter_add_(
            2,
            runtime_ids.unsqueeze(1).expand(-1, runtime_ids.shape[1], -1),
            attention,
        )
        has_source = source.any(dim=1).view(-1, 1, 1)
        gate = torch.sigmoid(self.copy_gate(hidden).float()) * has_source
        base_probability = torch.softmax(base_logits.float(), dim=-1)
        mixed = (1.0 - gate) * base_probability + gate * copy_probability
        log_probability = mixed.clamp_min(1.0e-12).log()
        output_logits = torch.where(has_source, log_probability, base_logits.float())
        return {
            "logits": output_logits,
            "state": {},
            "telemetry": {
                "surface": self.surface,
                "owned_by_marulho": True,
                "external_llm_used": False,
                "pointer_rank": self.pointer_rank,
                "pointer_parameter_count": self.pointer_parameter_count(),
                "source_token_count": source.sum(),
                "mean_copy_gate": gate.mean(),
                "base_frozen": True,
                "collect_telemetry": bool(collect_telemetry),
            },
        }

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
        if float(temperature) != 0.0 or float(top_p) != 1.0 or seed is not None:
            raise ValueError("V53 evidence generation supports deterministic greedy decode")
        prompt = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        if prompt.ndim != 2 or int(prompt.shape[1]) < 1:
            raise ValueError("prompt_ids must be nonempty [batch,time]")
        generated = prompt.to(device=self.device, dtype=torch.long)
        prompt_width = int(generated.shape[1])
        finished = torch.zeros(generated.shape[0], device=self.device, dtype=torch.bool)
        for _ in range(max(0, int(max_new_tokens))):
            context = generated[:, -self.context_length :]
            next_logits = self.forward(context, collect_telemetry=False)["logits"][:, -1]
            controlled, _control = _apply_decode_controls(
                next_logits,
                generated[:, prompt_width:],
                repetition_penalty=max(1.0, float(repetition_penalty)),
                no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
            )
            next_ids = controlled.argmax(dim=-1)
            if eos_id is not None:
                next_ids = torch.where(
                    finished,
                    torch.full_like(next_ids, int(eos_id)),
                    next_ids,
                )
            generated = torch.cat((generated, next_ids.unsqueeze(1)), dim=1)
            if eos_id is not None:
                finished |= next_ids.eq(int(eos_id))
                if bool(finished.all().item()):
                    break
        return {
            "generated_ids": generated,
            "generated_token_count": int(generated.shape[1]) - prompt_width,
            "surface": self.surface,
            "owned_by_marulho": True,
            "external_llm_used": False,
        }


def source_pointer_answer_loss(
    model: FrozenSourcePointerLanguageModel,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    answer_marker_ids: torch.Tensor,
    eos_id: int,
    pad_id: int,
) -> torch.Tensor:
    log_probability = model(input_ids, collect_telemetry=False)["logits"]
    losses = F.nll_loss(
        log_probability.reshape(-1, log_probability.shape[-1]),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape(target_ids.shape)
    answer = answer_target_mask(
        input_ids,
        marker_ids=answer_marker_ids,
        eos_id=int(eos_id),
    )
    valid = answer & target_ids.ne(int(pad_id))
    weights = valid.to(dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum()


def save_source_pointer_checkpoint(
    path: str | Path,
    model: FrozenSourcePointerLanguageModel,
    *,
    parent_checkpoint_sha256: str,
    metadata: Mapping[str, Any],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    payload = {
        "surface": SOURCE_POINTER_CHECKPOINT_SURFACE,
        "parent_checkpoint_sha256": str(parent_checkpoint_sha256),
        "pointer_rank": int(model.pointer_rank),
        "context_marker_ids": model.context_marker_ids.detach().cpu(),
        "question_marker_ids": model.question_marker_ids.detach().cpu(),
        "pointer_state": model.pointer_state_dict(),
        "metadata": dict(metadata),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_source_pointer_checkpoint(
    path: str | Path,
    base: MarulhoLanguageModel,
    *,
    expected_parent_checkpoint_sha256: str,
) -> tuple[FrozenSourcePointerLanguageModel, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if str(payload.get("surface")) != SOURCE_POINTER_CHECKPOINT_SURFACE:
        raise ValueError("source pointer checkpoint surface is incompatible")
    if str(payload.get("parent_checkpoint_sha256")) != str(
        expected_parent_checkpoint_sha256
    ):
        raise ValueError("source pointer parent checkpoint differs")
    model = FrozenSourcePointerLanguageModel(
        base,
        context_marker_ids=payload["context_marker_ids"],
        question_marker_ids=payload["question_marker_ids"],
        pointer_rank=int(payload["pointer_rank"]),
    )
    missing, unexpected = model.load_state_dict(
        {f"base.{name}": value for name, value in base.state_dict().items()}
        | dict(payload["pointer_state"]),
        strict=False,
    )
    if missing or unexpected:
        raise ValueError(
            f"source pointer state mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model, dict(payload.get("metadata") or {})
