"""Frozen-parent multi-view autoregressive source transducer for MARULHO."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

import torch
import torch.nn.functional as F
from torch import nn

from marulho.data.language_tokenizer import LanguageTokenizer
from marulho.training.language_model import MarulhoLanguageModel


MULTIVIEW_TRANSDUCER_CHECKPOINT_SURFACE = (
    "marulho_multiview_answer_transducer_checkpoint.v1"
)
VIEW_MODES = ("both", "bidirectional_only", "causal_only")


def _marker_start(
    input_ids: torch.Tensor,
    marker_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    size = int(marker_ids.numel())
    matches = input_ids.unfold(1, size, 1).eq(marker_ids).all(dim=-1)
    found = matches.any(dim=1)
    return matches.to(dtype=torch.int64).argmax(dim=1), found


def multiview_type_ids(
    input_ids: torch.Tensor,
    *,
    context_marker_ids: torch.Tensor,
    question_marker_ids: torch.Tensor,
    answer_marker_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return other/source/question type IDs and legal source positions."""

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
class MultiViewSupervisionBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    type_ids: torch.Tensor
    source_mask: torch.Tensor
    start_positions: torch.Tensor
    end_positions: torch.Tensor
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor
    pointer_targets: torch.Tensor
    causal_hidden: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> MultiViewSupervisionBatch:
        return MultiViewSupervisionBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            type_ids=self.type_ids.to(device),
            source_mask=self.source_mask.to(device),
            start_positions=self.start_positions.to(device),
            end_positions=self.end_positions.to(device),
            decoder_input_ids=self.decoder_input_ids.to(device),
            decoder_attention_mask=self.decoder_attention_mask.to(device),
            pointer_targets=self.pointer_targets.to(device),
            causal_hidden=(
                None if self.causal_hidden is None else self.causal_hidden.to(device)
            ),
        )


def build_multiview_supervision_batches(
    manifest: Mapping[str, Any],
    tokenizer: LanguageTokenizer,
    *,
    sequence_length: int,
    batch_size: int,
    maximum_answer_tokens: int,
) -> tuple[tuple[MultiViewSupervisionBatch, ...], dict[str, Any]]:
    """Build aligned source rows and autoregressive source-position targets."""

    length = int(sequence_length)
    size = int(batch_size)
    answer_limit = int(maximum_answer_tokens)
    cases = tuple(dict(value) for value in manifest["cases"])
    if not cases or length < 2 or size < 1 or answer_limit < 1:
        raise ValueError("multi-view supervision requires positive shapes and cases")
    if len(cases) % size:
        raise ValueError("multi-view cases must divide into full batches")
    decoder_length = answer_limit + 1
    rows: list[list[int]] = []
    attention_rows: list[list[bool]] = []
    type_rows: list[list[int]] = []
    source_rows: list[list[bool]] = []
    starts: list[int] = []
    ends: list[int] = []
    decoder_rows: list[list[int]] = []
    decoder_mask_rows: list[list[bool]] = []
    pointer_rows: list[list[int]] = []
    answer_token_lengths: list[int] = []
    for case in cases:
        prompt = str(case["prompt"])
        source = str(case["source_text"])
        answers = tuple(str(value) for value in case["answers"])
        source_prefix = "Context: "
        if not prompt.startswith(source_prefix + source):
            raise ValueError("multi-view prompt/source boundary is inconsistent")
        selected_answer = next(
            (
                answer
                for answer in answers
                if answer and source.casefold().find(answer.casefold()) >= 0
            ),
            None,
        )
        if selected_answer is None:
            raise ValueError("multi-view answer is absent from source")
        local_start = source.casefold().find(selected_answer.casefold())
        answer_start = len(source_prefix) + local_start
        answer_end = answer_start + len(selected_answer)
        ids, offsets = tokenizer.encode_with_offsets(
            prompt, add_bos=True, add_eos=False
        )
        if len(ids) > length:
            raise ValueError("multi-view prompt exceeds the causal context")
        span = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > answer_start and start < answer_end
        ]
        if not 1 <= len(span) <= answer_limit:
            raise ValueError("answer span exceeds the transducer answer limit")
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
            raise ValueError("gold pointer target escapes the source field")
        padding = length - len(ids)
        rows.append(ids + [int(tokenizer.pad_id)] * padding)
        attention_rows.append([True] * len(ids) + [False] * padding)
        type_rows.append(
            [
                1 if source_value else 2 if question_value else 0
                for source_value, question_value in zip(
                    source_type, question_type, strict=True
                )
            ]
            + [0] * padding
        )
        source_rows.append(source_mask + [False] * padding)
        starts.append(span[0])
        ends.append(span[-1])
        answer_ids = [ids[index] for index in span]
        decoder_inputs = [int(tokenizer.bos_id), *answer_ids]
        pointer_targets = [*span, length]
        decoder_padding = decoder_length - len(pointer_targets)
        decoder_rows.append(
            decoder_inputs + [int(tokenizer.pad_id)] * decoder_padding
        )
        decoder_mask_rows.append(
            [True] * len(pointer_targets) + [False] * decoder_padding
        )
        pointer_rows.append(pointer_targets + [-100] * decoder_padding)
        answer_token_lengths.append(len(span))
    tensor_ids = torch.tensor(rows, dtype=torch.long)
    tensor_attention = torch.tensor(attention_rows, dtype=torch.bool)
    tensor_types = torch.tensor(type_rows, dtype=torch.long)
    tensor_source = torch.tensor(source_rows, dtype=torch.bool)
    tensor_starts = torch.tensor(starts, dtype=torch.long)
    tensor_ends = torch.tensor(ends, dtype=torch.long)
    tensor_decoder = torch.tensor(decoder_rows, dtype=torch.long)
    tensor_decoder_mask = torch.tensor(decoder_mask_rows, dtype=torch.bool)
    tensor_pointer = torch.tensor(pointer_rows, dtype=torch.long)
    batches = tuple(
        MultiViewSupervisionBatch(
            input_ids=tensor_ids[index : index + size],
            attention_mask=tensor_attention[index : index + size],
            type_ids=tensor_types[index : index + size],
            source_mask=tensor_source[index : index + size],
            start_positions=tensor_starts[index : index + size],
            end_positions=tensor_ends[index : index + size],
            decoder_input_ids=tensor_decoder[index : index + size],
            decoder_attention_mask=tensor_decoder_mask[index : index + size],
            pointer_targets=tensor_pointer[index : index + size],
        )
        for index in range(0, len(cases), size)
    )
    return batches, {
        "surface": "marulho_multiview_supervision_batches.v1",
        "case_count": len(cases),
        "batch_count": len(batches),
        "batch_size": size,
        "sequence_length": length,
        "decoder_length": decoder_length,
        "all_gold_spans_source_contained": True,
        "minimum_answer_tokens": min(answer_token_lengths),
        "maximum_answer_tokens": max(answer_token_lengths),
        "all_prompts_fit": True,
    }


@torch.no_grad()
def cache_frozen_causal_hidden(
    base: MarulhoLanguageModel,
    batches: Sequence[MultiViewSupervisionBatch],
    *,
    device: torch.device,
) -> tuple[tuple[MultiViewSupervisionBatch, ...], dict[str, Any]]:
    """Compute one immutable host-BF16 V39 hidden cache for repeated training."""

    base.eval()
    digest = hashlib.sha256()
    cached = []
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for index, batch in enumerate(batches):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = base._forward_hidden(
                batch.input_ids.to(device), collect_telemetry=False
            )["hidden"]
        host = hidden.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        digest.update(host.view(torch.uint16).numpy().tobytes())
        cached.append(replace(batch, causal_hidden=host))
        if (index + 1) % 16 == 0 or index + 1 == len(batches):
            print(
                f"[multiview-cache] batch {index + 1}/{len(batches)}",
                flush=True,
            )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    storage_bytes = sum(
        int(batch.causal_hidden.numel() * batch.causal_hidden.element_size())
        for batch in cached
        if batch.causal_hidden is not None
    )
    return tuple(cached), {
        "surface": "marulho_frozen_causal_hidden_cache.v1",
        "batch_count": len(cached),
        "case_count": sum(int(batch.input_ids.shape[0]) for batch in cached),
        "token_count": sum(int(batch.input_ids.numel()) for batch in cached),
        "hidden_width": int(base.config.state_dim),
        "dtype": "torch.bfloat16",
        "host_storage_bytes": storage_bytes,
        "elapsed_seconds": elapsed,
        "tokens_per_second": sum(
            int(batch.input_ids.numel()) for batch in cached
        )
        / elapsed,
        "content_sha256": digest.hexdigest(),
        "durable": False,
    }


class FrozenBaseMultiViewAnswerTransducer(nn.Module):
    """Fuse frozen causal and learned bidirectional views to emit answer tokens."""

    surface = "marulho_frozen_base_multiview_answer_transducer.v1"

    def __init__(
        self,
        base: MarulhoLanguageModel,
        *,
        context_marker_ids: torch.Tensor,
        question_marker_ids: torch.Tensor,
        answer_marker_ids: torch.Tensor,
        bos_id: int,
        pad_id: int,
        eos_id: int,
        width: int = 192,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        heads: int = 6,
        maximum_answer_tokens: int = 8,
        span_loss_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.width = int(width)
        self.encoder_layers = int(encoder_layers)
        self.decoder_layers = int(decoder_layers)
        self.heads = int(heads)
        self.maximum_answer_tokens = int(maximum_answer_tokens)
        self.span_loss_weight = float(span_loss_weight)
        self.bos_id = int(bos_id)
        self.pad_id = int(pad_id)
        self.eos_id = int(eos_id)
        self.embedding_projection = nn.Linear(
            int(base.config.embedding_dim), self.width
        )
        self.causal_projection = nn.Linear(int(base.config.state_dim), self.width)
        self.position_embedding = nn.Embedding(base.context_length, self.width)
        self.type_embedding = nn.Embedding(3, self.width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.width,
            nhead=self.heads,
            dim_feedforward=self.width * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.bidirectional_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.encoder_layers,
            norm=nn.LayerNorm(self.width),
            enable_nested_tensor=False,
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.width * 2, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
        )
        self.fusion_norm = nn.LayerNorm(self.width)
        self.answer_embedding_projection = nn.Linear(
            int(base.config.embedding_dim), self.width
        )
        self.answer_position_embedding = nn.Embedding(
            self.maximum_answer_tokens + 1, self.width
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.width,
            nhead=self.heads,
            dim_feedforward=self.width * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.answer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.decoder_layers,
            norm=nn.LayerNorm(self.width),
        )
        self.pointer_query = nn.Linear(self.width, self.width, bias=False)
        self.pointer_key = nn.Linear(self.width, self.width, bias=False)
        self.eos_head = nn.Linear(self.width, 1)
        self.start_head = nn.Linear(self.width, 1)
        self.end_head = nn.Linear(self.width, 1)
        self.register_buffer(
            "context_marker_ids", context_marker_ids.detach().to(dtype=torch.long)
        )
        self.register_buffer(
            "question_marker_ids", question_marker_ids.detach().to(dtype=torch.long)
        )
        self.register_buffer(
            "answer_marker_ids", answer_marker_ids.detach().to(dtype=torch.long)
        )
        self.inference_view_mode = "both"

    @property
    def device(self) -> torch.device:
        return self.embedding_projection.weight.device

    @property
    def context_length(self) -> int:
        return self.base.context_length

    def transducer_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("base.")
        )

    def transducer_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in self.state_dict().items()
            if not name.startswith("base.")
        }

    def set_inference_view_mode(self, mode: str) -> None:
        value = str(mode)
        if value not in VIEW_MODES:
            raise ValueError(f"unknown multi-view mode: {value}")
        self.inference_view_mode = value

    def _causal_hidden(
        self,
        runtime_ids: torch.Tensor,
        supplied: torch.Tensor | None,
    ) -> torch.Tensor:
        if supplied is not None:
            return supplied.to(device=self.device)
        with torch.no_grad():
            return self.base._forward_hidden(
                runtime_ids, collect_telemetry=False
            )["hidden"].detach()

    def encode_memory(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        type_ids: torch.Tensor | None = None,
        source_mask: torch.Tensor | None = None,
        causal_hidden: torch.Tensor | None = None,
        view_mode: str = "both",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mode = str(view_mode)
        if mode not in VIEW_MODES:
            raise ValueError(f"unknown multi-view mode: {mode}")
        runtime_ids = input_ids.to(device=self.device, dtype=torch.long)
        if attention_mask is None:
            attention_mask = runtime_ids.ne(self.pad_id)
        else:
            attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)
        if type_ids is None or source_mask is None:
            inferred_types, inferred_source = multiview_type_ids(
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
            embeddings = F.embedding(
                runtime_ids, self.base.token_embedding.weight
            ).detach()
        positions = torch.arange(runtime_ids.shape[1], device=self.device).unsqueeze(0)
        bidirectional_input = (
            self.embedding_projection(embeddings)
            + self.position_embedding(positions)
            + self.type_embedding(type_ids)
        )
        if mode != "causal_only":
            bidirectional = self.bidirectional_encoder(
                bidirectional_input, src_key_padding_mask=~attention_mask
            )
        else:
            bidirectional = torch.zeros_like(bidirectional_input)
        if mode != "bidirectional_only":
            causal = self.causal_projection(
                self._causal_hidden(runtime_ids, causal_hidden)
            )
        else:
            causal = torch.zeros_like(bidirectional_input)
        fused = self.fusion(torch.cat((bidirectional, causal), dim=-1))
        memory = self.fusion_norm(bidirectional + causal + fused)
        return memory, attention_mask, source_mask

    def pointer_logits(
        self,
        memory: torch.Tensor,
        source_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        decoder_ids = decoder_input_ids.to(device=self.device, dtype=torch.long)
        decoder_mask = decoder_attention_mask.to(
            device=self.device, dtype=torch.bool
        )
        with torch.no_grad():
            answer_embeddings = F.embedding(
                decoder_ids, self.base.token_embedding.weight
            ).detach()
        positions = torch.arange(decoder_ids.shape[1], device=self.device).unsqueeze(0)
        target = self.answer_embedding_projection(
            answer_embeddings
        ) + self.answer_position_embedding(positions)
        causal_mask = torch.ones(
            decoder_ids.shape[1],
            decoder_ids.shape[1],
            device=self.device,
            dtype=torch.bool,
        ).triu(diagonal=1)
        decoded = self.answer_decoder(
            target,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~decoder_mask,
        )
        query = self.pointer_query(decoded)
        key = self.pointer_key(memory)
        positions_logits = torch.einsum("blw,btw->blt", query, key) / math.sqrt(
            float(self.width)
        )
        positions_logits = positions_logits.float().masked_fill(
            ~source_mask.unsqueeze(1), -1.0e4
        )
        eos_logits = self.eos_head(decoded).float()
        return torch.cat((positions_logits, eos_logits), dim=-1)

    def loss(
        self,
        batch: MultiViewSupervisionBatch,
        *,
        view_mode: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        memory, _attention, source = self.encode_memory(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            type_ids=batch.type_ids,
            source_mask=batch.source_mask,
            causal_hidden=batch.causal_hidden,
            view_mode=view_mode,
        )
        logits = self.pointer_logits(
            memory,
            source,
            batch.decoder_input_ids,
            batch.decoder_attention_mask,
        )
        pointer_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch.pointer_targets.to(self.device).reshape(-1),
            ignore_index=-100,
        )
        start = self.start_head(memory).squeeze(-1).float().masked_fill(
            ~source, -1.0e4
        )
        end = self.end_head(memory).squeeze(-1).float().masked_fill(
            ~source, -1.0e4
        )
        span_loss = 0.5 * (
            F.cross_entropy(start, batch.start_positions.to(self.device))
            + F.cross_entropy(end, batch.end_positions.to(self.device))
        )
        total = pointer_loss + self.span_loss_weight * span_loss
        return total, {
            "pointer_loss": pointer_loss.detach(),
            "span_loss": span_loss.detach(),
        }

    @torch.no_grad()
    def _generate_from_memory(
        self,
        prompt: torch.Tensor,
        memory: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        max_new_tokens: int,
    ) -> torch.Tensor:
        batch_size = int(prompt.shape[0])
        requested = max(0, int(max_new_tokens))
        continuation = torch.full(
            (batch_size, requested),
            self.eos_id,
            dtype=torch.long,
            device=self.device,
        )
        decoder_ids = torch.full(
            (batch_size, 1), self.bos_id, dtype=torch.long, device=self.device
        )
        decoder_mask = torch.ones_like(decoder_ids, dtype=torch.bool)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        for step in range(min(requested, self.maximum_answer_tokens + 1)):
            logits = self.pointer_logits(
                memory, source_mask, decoder_ids, decoder_mask
            )[:, -1]
            selected = logits.argmax(dim=-1)
            is_eos = selected.eq(prompt.shape[1])
            selected_position = selected.clamp_max(prompt.shape[1] - 1)
            selected_token = prompt.gather(1, selected_position.unsqueeze(1))[:, 0]
            selected_token = torch.where(
                finished | is_eos,
                torch.full_like(selected_token, self.eos_id),
                selected_token,
            )
            continuation[:, step] = selected_token
            finished |= is_eos
            decoder_ids = torch.cat((decoder_ids, selected_token.unsqueeze(1)), dim=1)
            decoder_mask = torch.ones_like(decoder_ids, dtype=torch.bool)
            if bool(finished.all().item()):
                break
        return continuation

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
        del repetition_penalty, no_repeat_ngram_size
        if eos_id is not None and int(eos_id) != self.eos_id:
            raise ValueError("multi-view transducer EOS differs from tokenizer")
        if float(temperature) != 0.0 or float(top_p) != 1.0 or seed is not None:
            raise ValueError("V55 evidence generation is deterministic greedy")
        prompt = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        prompt = prompt.to(device=self.device, dtype=torch.long)
        _types, source = multiview_type_ids(
            prompt,
            context_marker_ids=self.context_marker_ids,
            question_marker_ids=self.question_marker_ids,
            answer_marker_ids=self.answer_marker_ids,
        )
        if not bool(source.any(dim=1).all().item()):
            return self.base.generate(
                prompt,
                max_new_tokens=int(max_new_tokens),
                eos_id=self.eos_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
            )
        memory, _attention, source = self.encode_memory(
            prompt, view_mode=self.inference_view_mode
        )
        continuation = self._generate_from_memory(
            prompt,
            memory,
            source,
            max_new_tokens=int(max_new_tokens),
        )
        return {
            "generated_ids": torch.cat((prompt, continuation), dim=1),
            "generated_token_count": int(continuation.shape[1]),
            "surface": self.surface,
            "view_mode": self.inference_view_mode,
            "decode_kind": "autoregressive_source_position_pointer",
            "owned_by_marulho": True,
            "external_llm_used": False,
        }


def save_multiview_transducer_checkpoint(
    path: str | Path,
    model: FrozenBaseMultiViewAnswerTransducer,
    *,
    parent_checkpoint_sha256: str,
    metadata: Mapping[str, Any],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    payload = {
        "surface": MULTIVIEW_TRANSDUCER_CHECKPOINT_SURFACE,
        "parent_checkpoint_sha256": str(parent_checkpoint_sha256),
        "configuration": {
            "width": model.width,
            "encoder_layers": model.encoder_layers,
            "decoder_layers": model.decoder_layers,
            "heads": model.heads,
            "maximum_answer_tokens": model.maximum_answer_tokens,
            "span_loss_weight": model.span_loss_weight,
            "bos_id": model.bos_id,
            "pad_id": model.pad_id,
            "eos_id": model.eos_id,
        },
        "markers": {
            "context": model.context_marker_ids.detach().cpu(),
            "question": model.question_marker_ids.detach().cpu(),
            "answer": model.answer_marker_ids.detach().cpu(),
        },
        "transducer_state": model.transducer_state_dict(),
        "metadata": dict(metadata),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_multiview_transducer_checkpoint(
    path: str | Path,
    base: MarulhoLanguageModel,
    *,
    expected_parent_checkpoint_sha256: str,
) -> tuple[FrozenBaseMultiViewAnswerTransducer, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if str(payload.get("surface")) != MULTIVIEW_TRANSDUCER_CHECKPOINT_SURFACE:
        raise ValueError("multi-view checkpoint surface is incompatible")
    if str(payload.get("parent_checkpoint_sha256")) != str(
        expected_parent_checkpoint_sha256
    ):
        raise ValueError("multi-view parent checkpoint differs")
    configuration = dict(payload["configuration"])
    markers = dict(payload["markers"])
    model = FrozenBaseMultiViewAnswerTransducer(
        base,
        context_marker_ids=markers["context"],
        question_marker_ids=markers["question"],
        answer_marker_ids=markers["answer"],
        bos_id=int(configuration["bos_id"]),
        pad_id=int(configuration["pad_id"]),
        eos_id=int(configuration["eos_id"]),
        width=int(configuration["width"]),
        encoder_layers=int(configuration["encoder_layers"]),
        decoder_layers=int(configuration["decoder_layers"]),
        heads=int(configuration["heads"]),
        maximum_answer_tokens=int(configuration["maximum_answer_tokens"]),
        span_loss_weight=float(configuration["span_loss_weight"]),
    )
    merged = {f"base.{name}": value for name, value in base.state_dict().items()}
    merged.update(dict(payload["transducer_state"]))
    model.load_state_dict(merged, strict=True)
    return model, dict(payload.get("metadata") or {})
