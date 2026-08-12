"""Isolated trainable source/question span encoder for MARULHO."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import torch
import torch.nn.functional as F
from torch import nn

from marulho.data.language_tokenizer import LanguageTokenizer
from marulho.training.language_model import MarulhoLanguageModel


SPAN_ENCODER_CHECKPOINT_SURFACE = "marulho_span_encoder_checkpoint.v1"


def _marker_start(
    input_ids: torch.Tensor,
    marker_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    size = int(marker_ids.numel())
    matches = input_ids.unfold(1, size, 1).eq(marker_ids).all(dim=-1)
    found = matches.any(dim=1)
    return matches.to(dtype=torch.int64).argmax(dim=1), found


def span_encoder_type_ids(
    input_ids: torch.Tensor,
    *,
    context_marker_ids: torch.Tensor,
    question_marker_ids: torch.Tensor,
    answer_marker_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return other/source/question type IDs and the legal source-token mask."""

    context_start, has_context = _marker_start(input_ids, context_marker_ids)
    question_start, has_question = _marker_start(input_ids, question_marker_ids)
    answer_start, has_answer = _marker_start(input_ids, answer_marker_ids)
    context_end = context_start + int(context_marker_ids.numel())
    question_end = question_start + int(question_marker_ids.numel())
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    valid = (
        has_context
        & has_question
        & has_answer
        & question_start.gt(context_end)
        & answer_start.gt(question_end)
    )
    source = (
        positions.ge(context_end.unsqueeze(1))
        & positions.lt(question_start.unsqueeze(1))
        & valid.unsqueeze(1)
    )
    question = (
        positions.ge(question_end.unsqueeze(1))
        & positions.lt(answer_start.unsqueeze(1))
        & valid.unsqueeze(1)
    )
    types = source.to(dtype=torch.long) + question.to(dtype=torch.long) * 2
    return types, source


@dataclass(frozen=True)
class SpanSupervisionBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    type_ids: torch.Tensor
    source_mask: torch.Tensor
    start_positions: torch.Tensor
    end_positions: torch.Tensor

    def to(self, device: torch.device | str) -> SpanSupervisionBatch:
        return SpanSupervisionBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            type_ids=self.type_ids.to(device),
            source_mask=self.source_mask.to(device),
            start_positions=self.start_positions.to(device),
            end_positions=self.end_positions.to(device),
        )


def build_span_supervision_batches(
    manifest: Mapping[str, Any],
    tokenizer: LanguageTokenizer,
    *,
    sequence_length: int,
    batch_size: int,
) -> tuple[tuple[SpanSupervisionBatch, ...], dict[str, Any]]:
    """Build fixed aligned rows with exact character-to-token span labels."""

    length = int(sequence_length)
    size = int(batch_size)
    cases = tuple(dict(value) for value in manifest["cases"])
    if not cases or length < 2 or size < 1:
        raise ValueError("span supervision requires cases and positive shapes")
    if len(cases) % size:
        raise ValueError("span supervision cases must divide into full batches")
    rows: list[list[int]] = []
    attention_rows: list[list[bool]] = []
    type_rows: list[list[int]] = []
    source_rows: list[list[bool]] = []
    starts: list[int] = []
    ends: list[int] = []
    answer_token_lengths: list[int] = []
    for case in cases:
        prompt = str(case["prompt"])
        source = str(case["source_text"])
        answers = tuple(str(value) for value in case["answers"])
        source_prefix = "Context: "
        if not prompt.startswith(source_prefix + source):
            raise ValueError("span prompt/source boundary is inconsistent")
        selected_answer = next(
            (
                answer
                for answer in answers
                if answer and source.casefold().find(answer.casefold()) >= 0
            ),
            None,
        )
        if selected_answer is None:
            raise ValueError("span supervision answer is absent from source")
        local_start = source.casefold().find(selected_answer.casefold())
        answer_start = len(source_prefix) + local_start
        answer_end = answer_start + len(selected_answer)
        ids, offsets = tokenizer.encode_with_offsets(
            prompt,
            add_bos=True,
            add_eos=False,
        )
        if len(ids) > length:
            raise ValueError("span supervision prompt exceeds the causal context")
        span = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > answer_start and start < answer_end
        ]
        if not span:
            raise ValueError("answer characters map to no tokenizer span")
        question_char = prompt.index("\nQuestion:")
        answer_char = prompt.index("\nAnswer:")
        source_type = [
            end > len(source_prefix) and start < question_char
            for start, end in offsets
        ]
        question_type = [
            end > question_char + len("\nQuestion:") and start < answer_char
            for start, end in offsets
        ]
        source_mask = [bool(value) for value in source_type]
        if not all(source_mask[index] for index in span):
            raise ValueError("gold answer span escapes the source field")
        padding = length - len(ids)
        rows.append(ids + [int(tokenizer.pad_id)] * padding)
        attention_rows.append([True] * len(ids) + [False] * padding)
        type_rows.append(
            [1 if source_value else 2 if question_value else 0 for source_value, question_value in zip(source_type, question_type, strict=True)]
            + [0] * padding
        )
        source_rows.append(source_mask + [False] * padding)
        starts.append(span[0])
        ends.append(span[-1])
        answer_token_lengths.append(len(span))
    tensor_ids = torch.tensor(rows, dtype=torch.long)
    tensor_attention = torch.tensor(attention_rows, dtype=torch.bool)
    tensor_types = torch.tensor(type_rows, dtype=torch.long)
    tensor_source = torch.tensor(source_rows, dtype=torch.bool)
    tensor_starts = torch.tensor(starts, dtype=torch.long)
    tensor_ends = torch.tensor(ends, dtype=torch.long)
    batches = tuple(
        SpanSupervisionBatch(
            input_ids=tensor_ids[index : index + size],
            attention_mask=tensor_attention[index : index + size],
            type_ids=tensor_types[index : index + size],
            source_mask=tensor_source[index : index + size],
            start_positions=tensor_starts[index : index + size],
            end_positions=tensor_ends[index : index + size],
        )
        for index in range(0, len(cases), size)
    )
    return batches, {
        "surface": "marulho_span_supervision_batches.v1",
        "case_count": len(cases),
        "batch_count": len(batches),
        "batch_size": size,
        "sequence_length": length,
        "all_gold_spans_source_contained": True,
        "minimum_answer_tokens": min(answer_token_lengths),
        "maximum_answer_tokens": max(answer_token_lengths),
        "all_prompts_fit": True,
    }


class FrozenBaseSpanEncoder(nn.Module):
    """Train source/question representation while keeping the language base exact."""

    surface = "marulho_frozen_base_span_encoder.v1"

    def __init__(
        self,
        base: MarulhoLanguageModel,
        *,
        context_marker_ids: torch.Tensor,
        question_marker_ids: torch.Tensor,
        answer_marker_ids: torch.Tensor,
        width: int = 128,
        layers: int = 2,
        heads: int = 4,
        maximum_answer_tokens: int = 8,
    ) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.width = int(width)
        self.layers = int(layers)
        self.heads = int(heads)
        self.maximum_answer_tokens = int(maximum_answer_tokens)
        self.input_projection = nn.Linear(int(base.config.embedding_dim), self.width)
        self.position_embedding = nn.Embedding(base.context_length, self.width)
        self.type_embedding = nn.Embedding(3, self.width)
        layer = nn.TransformerEncoderLayer(
            d_model=self.width,
            nhead=self.heads,
            dim_feedforward=self.width * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.layers,
            norm=nn.LayerNorm(self.width),
            enable_nested_tensor=False,
        )
        self.start_head = nn.Linear(self.width, 1)
        self.end_head = nn.Linear(self.width, 1)
        self.register_buffer("context_marker_ids", context_marker_ids.long())
        self.register_buffer("question_marker_ids", question_marker_ids.long())
        self.register_buffer("answer_marker_ids", answer_marker_ids.long())

    @property
    def device(self) -> torch.device:
        return self.input_projection.weight.device

    def encoder_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("base.")
        )

    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in self.state_dict().items()
            if not name.startswith("base.")
        }

    def span_logits(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        type_ids: torch.Tensor | None = None,
        source_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        runtime_ids = input_ids.to(device=self.device, dtype=torch.long)
        if attention_mask is None:
            attention_mask = runtime_ids.ne(0)
        else:
            attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)
        if type_ids is None or source_mask is None:
            inferred_types, inferred_source = span_encoder_type_ids(
                runtime_ids,
                context_marker_ids=self.context_marker_ids,
                question_marker_ids=self.question_marker_ids,
                answer_marker_ids=self.answer_marker_ids,
            )
            type_ids = inferred_types if type_ids is None else type_ids
            source_mask = inferred_source if source_mask is None else source_mask
        type_ids = type_ids.to(device=self.device, dtype=torch.long)
        source_mask = source_mask.to(device=self.device, dtype=torch.bool)
        with torch.no_grad():
            embeddings = F.embedding(runtime_ids, self.base.token_embedding.weight).detach()
        positions = torch.arange(runtime_ids.shape[1], device=self.device).unsqueeze(0)
        hidden = (
            self.input_projection(embeddings)
            + self.position_embedding(positions)
            + self.type_embedding(type_ids)
        )
        encoded = self.encoder(hidden, src_key_padding_mask=~attention_mask)
        start = self.start_head(encoded).squeeze(-1).float()
        end = self.end_head(encoded).squeeze(-1).float()
        start = start.masked_fill(~source_mask, -1.0e4)
        end = end.masked_fill(~source_mask, -1.0e4)
        return start, end, source_mask

    def loss(self, batch: SpanSupervisionBatch) -> torch.Tensor:
        start, end, _source = self.span_logits(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            type_ids=batch.type_ids,
            source_mask=batch.source_mask,
        )
        return 0.5 * (
            F.cross_entropy(start, batch.start_positions.to(self.device))
            + F.cross_entropy(end, batch.end_positions.to(self.device))
        )

    def select_spans(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        start, end, source = self.span_logits(input_ids)
        joint = start.unsqueeze(2) + end.unsqueeze(1)
        positions = torch.arange(input_ids.shape[1], device=self.device)
        valid_order = positions.unsqueeze(1).le(positions.unsqueeze(0))
        valid_length = (positions.unsqueeze(0) - positions.unsqueeze(1)).lt(
            self.maximum_answer_tokens
        )
        valid = (
            source.unsqueeze(2)
            & source.unsqueeze(1)
            & valid_order.unsqueeze(0)
            & valid_length.unsqueeze(0)
        )
        flat = joint.masked_fill(~valid, -1.0e4).flatten(1).argmax(dim=1)
        width = int(input_ids.shape[1])
        return flat // width, flat % width

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
        prompt = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        prompt = prompt.to(device=self.device, dtype=torch.long)
        _types, source = span_encoder_type_ids(
            prompt,
            context_marker_ids=self.context_marker_ids,
            question_marker_ids=self.question_marker_ids,
            answer_marker_ids=self.answer_marker_ids,
        )
        if not bool(source.any(dim=1).all().item()):
            return self.base.generate(
                prompt,
                max_new_tokens=int(max_new_tokens),
                eos_id=eos_id,
                repetition_penalty=float(repetition_penalty),
                no_repeat_ngram_size=int(no_repeat_ngram_size),
                temperature=float(temperature),
                top_p=float(top_p),
                seed=seed,
            )
        starts, ends = self.select_spans(prompt)
        requested = max(0, int(max_new_tokens))
        fill = int(self.base.config.vocab_size - 1 if eos_id is None else eos_id)
        continuation = torch.full(
            (prompt.shape[0], requested),
            fill,
            dtype=torch.long,
            device=self.device,
        )
        for index, (start, end) in enumerate(zip(starts.tolist(), ends.tolist(), strict=True)):
            selected = prompt[index, int(start) : int(end) + 1][:requested]
            continuation[index, : selected.numel()] = selected
        return {
            "generated_ids": torch.cat((prompt, continuation), dim=1),
            "generated_token_count": requested,
            "surface": self.surface,
            "owned_by_marulho": True,
            "external_llm_used": False,
            "decode_kind": "source_span_copy",
        }


def save_span_encoder_checkpoint(
    path: str | Path,
    model: FrozenBaseSpanEncoder,
    *,
    parent_checkpoint_sha256: str,
    metadata: Mapping[str, Any],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    payload = {
        "surface": SPAN_ENCODER_CHECKPOINT_SURFACE,
        "parent_checkpoint_sha256": str(parent_checkpoint_sha256),
        "configuration": {
            "width": model.width,
            "layers": model.layers,
            "heads": model.heads,
            "maximum_answer_tokens": model.maximum_answer_tokens,
        },
        "markers": {
            "context": model.context_marker_ids.detach().cpu(),
            "question": model.question_marker_ids.detach().cpu(),
            "answer": model.answer_marker_ids.detach().cpu(),
        },
        "encoder_state": model.encoder_state_dict(),
        "metadata": dict(metadata),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_span_encoder_checkpoint(
    path: str | Path,
    base: MarulhoLanguageModel,
    *,
    expected_parent_checkpoint_sha256: str,
) -> tuple[FrozenBaseSpanEncoder, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if str(payload.get("surface")) != SPAN_ENCODER_CHECKPOINT_SURFACE:
        raise ValueError("span encoder checkpoint surface is incompatible")
    if str(payload.get("parent_checkpoint_sha256")) != str(expected_parent_checkpoint_sha256):
        raise ValueError("span encoder parent checkpoint differs")
    configuration = dict(payload["configuration"])
    markers = dict(payload["markers"])
    model = FrozenBaseSpanEncoder(
        base,
        context_marker_ids=markers["context"],
        question_marker_ids=markers["question"],
        answer_marker_ids=markers["answer"],
        width=int(configuration["width"]),
        layers=int(configuration["layers"]),
        heads=int(configuration["heads"]),
        maximum_answer_tokens=int(configuration["maximum_answer_tokens"]),
    )
    merged = {f"base.{name}": value for name, value in base.state_dict().items()}
    merged.update(dict(payload["encoder_state"]))
    model.load_state_dict(merged, strict=True)
    return model, dict(payload.get("metadata") or {})
