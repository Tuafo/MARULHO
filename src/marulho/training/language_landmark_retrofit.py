"""Frozen-parent landmark retrieval and causal evidence retrofit for MARULHO."""

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
from marulho.training.language_model import _apply_decode_controls


LANDMARK_RETROFIT_CHECKPOINT_SURFACE = "marulho_landmark_retrofit_checkpoint.v1"


@dataclass(frozen=True)
class LandmarkRetrofitBatch:
    source_ids: torch.Tensor
    source_attention_mask: torch.Tensor
    block_valid_mask: torch.Tensor
    gold_block_mask: torch.Tensor
    gold_evidence_indices: torch.Tensor
    retrieval_query_ids: torch.Tensor
    retrieval_query_attention_mask: torch.Tensor
    generator_input_ids: torch.Tensor
    generator_attention_mask: torch.Tensor
    generator_target_ids: torch.Tensor
    generator_loss_mask: torch.Tensor
    source_hidden: torch.Tensor | None = None
    retrieval_query_hidden: torch.Tensor | None = None
    generator_hidden: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> LandmarkRetrofitBatch:
        return LandmarkRetrofitBatch(
            source_ids=self.source_ids.to(device),
            source_attention_mask=self.source_attention_mask.to(device),
            block_valid_mask=self.block_valid_mask.to(device),
            gold_block_mask=self.gold_block_mask.to(device),
            gold_evidence_indices=self.gold_evidence_indices.to(device),
            retrieval_query_ids=self.retrieval_query_ids.to(device),
            retrieval_query_attention_mask=self.retrieval_query_attention_mask.to(
                device
            ),
            generator_input_ids=self.generator_input_ids.to(device),
            generator_attention_mask=self.generator_attention_mask.to(device),
            generator_target_ids=self.generator_target_ids.to(device),
            generator_loss_mask=self.generator_loss_mask.to(device),
            source_hidden=(
                None if self.source_hidden is None else self.source_hidden.to(device)
            ),
            retrieval_query_hidden=(
                None
                if self.retrieval_query_hidden is None
                else self.retrieval_query_hidden.to(device)
            ),
            generator_hidden=(
                None
                if self.generator_hidden is None
                else self.generator_hidden.to(device)
            ),
        )


def _answer_span_indices(
    tokenizer: LanguageTokenizer,
    source: str,
    answer: str,
) -> tuple[list[int], list[int]]:
    local_start = source.casefold().find(answer.casefold())
    if local_start < 0:
        raise ValueError("answer is absent from source")
    local_end = local_start + len(answer)
    source_ids, offsets = tokenizer.encode_with_offsets(
        source, add_bos=False, add_eos=False
    )
    span = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > local_start and start < local_end
    ]
    if not span:
        raise ValueError("answer maps to no source tokens")
    return source_ids, span


def _gold_evidence_indices(gold_mask: Sequence[bool], valid_count: int) -> list[int]:
    positives = [index for index, value in enumerate(gold_mask) if bool(value)]
    if not positives or len(positives) > 2:
        raise ValueError("answer must overlap one or two evidence blocks")
    if len(positives) == 2:
        return positives
    first = positives[0]
    if int(valid_count) == 1:
        return [first, first]
    adjacent = first + 1 if first + 1 < int(valid_count) else first - 1
    return [first, adjacent]


def _normalized_words(text: str) -> str:
    return " ".join(
        "".join(
            character if character.isalnum() else " "
            for character in str(text).casefold()
        ).split()
    )


def build_landmark_retrofit_batches(
    manifest: Mapping[str, Any],
    tokenizer: LanguageTokenizer,
    *,
    batch_size: int,
    block_tokens: int = 48,
    maximum_blocks: int = 5,
    query_length: int = 72,
) -> tuple[tuple[LandmarkRetrofitBatch, ...], dict[str, Any]]:
    """Build fixed long-source blocks and answer-only causal targets."""

    cases = tuple(dict(value) for value in manifest["cases"])
    size = int(batch_size)
    block = int(block_tokens)
    blocks = int(maximum_blocks)
    query = int(query_length)
    if not cases or size < 1 or block < 1 or blocks < 2 or query < 2:
        raise ValueError("landmark retrofit requires positive shapes and cases")
    if len(cases) % size:
        raise ValueError("landmark cases must divide into full batches")
    source_rows = []
    source_mask_rows = []
    valid_rows = []
    gold_rows = []
    evidence_rows = []
    retrieval_rows = []
    retrieval_mask_rows = []
    generator_rows = []
    generator_attention_rows = []
    target_rows = []
    loss_mask_rows = []
    answer_lengths = []
    block_counts = []
    boundary_spanning_count = 0
    for case in cases:
        source = str(case["source_text"])
        answers = tuple(str(value) for value in case["answers"])
        answer = next(
            (
                value
                for value in answers
                if value and source.casefold().find(value.casefold()) >= 0
            ),
            None,
        )
        if answer is None:
            raise ValueError("landmark case lacks a visible answer")
        source_ids, answer_span = _answer_span_indices(tokenizer, source, answer)
        block_count = math.ceil(len(source_ids) / block)
        if not 2 <= block_count <= blocks:
            raise ValueError("landmark source block count is outside the contract")
        padded_source = torch.full(
            (blocks, block), int(tokenizer.pad_id), dtype=torch.long
        )
        padded_source_mask = torch.zeros((blocks, block), dtype=torch.bool)
        for block_index in range(block_count):
            values = source_ids[block_index * block : (block_index + 1) * block]
            padded_source[block_index, : len(values)] = torch.tensor(values)
            padded_source_mask[block_index, : len(values)] = True
        gold_mask = [
            any(index // block == block_index for index in answer_span)
            for block_index in range(blocks)
        ]
        gold_indices = _gold_evidence_indices(gold_mask, block_count)
        if sum(gold_mask) == 2:
            boundary_spanning_count += 1
        question = str(case["question"])
        question_prompt = str(case["question_only_prompt"])
        normalized_question = _normalized_words(question)
        normalized_answer = _normalized_words(answer)
        if not normalized_answer or normalized_answer in normalized_question:
            raise ValueError("retrieval query leaks the normalized answer")
        if not set(index // block for index in answer_span).issubset(gold_indices):
            raise ValueError("gold evidence union does not contain the answer span")
        retrieval_ids = tokenizer.encode(question, add_bos=True, add_eos=False)
        if len(retrieval_ids) > query:
            raise ValueError("retrieval query exceeds the V39 context")
        full_text = f"{question_prompt} {answer}"
        full_ids, full_offsets = tokenizer.encode_with_offsets(
            full_text, add_bos=True, add_eos=True
        )
        if len(full_ids) > query + 1:
            raise ValueError("generator query and answer exceed the V39 context")
        generator_input = full_ids[:-1]
        generator_targets = full_ids[1:]
        answer_character_start = len(question_prompt) + 1
        target_mask = [
            bool(
                target_index == len(full_ids) - 1
                or full_offsets[target_index][1] > answer_character_start
            )
            for target_index in range(1, len(full_ids))
        ]
        if not any(target_mask):
            raise ValueError("generator has no answer/EOS targets")
        source_rows.append(padded_source)
        source_mask_rows.append(padded_source_mask)
        valid_rows.append([index < block_count for index in range(blocks)])
        gold_rows.append(gold_mask)
        evidence_rows.append(gold_indices)
        retrieval_rows.append(
            retrieval_ids + [int(tokenizer.pad_id)] * (query - len(retrieval_ids))
        )
        retrieval_mask_rows.append(
            [True] * len(retrieval_ids) + [False] * (query - len(retrieval_ids))
        )
        generator_rows.append(
            generator_input + [int(tokenizer.pad_id)] * (query - len(generator_input))
        )
        generator_attention_rows.append(
            [True] * len(generator_input) + [False] * (query - len(generator_input))
        )
        target_rows.append(
            generator_targets
            + [int(tokenizer.pad_id)] * (query - len(generator_targets))
        )
        loss_mask_rows.append(target_mask + [False] * (query - len(target_mask)))
        answer_lengths.append(len(answer_span))
        block_counts.append(block_count)
    tensors = {
        "source_ids": torch.stack(source_rows),
        "source_attention_mask": torch.stack(source_mask_rows),
        "block_valid_mask": torch.tensor(valid_rows, dtype=torch.bool),
        "gold_block_mask": torch.tensor(gold_rows, dtype=torch.bool),
        "gold_evidence_indices": torch.tensor(evidence_rows, dtype=torch.long),
        "retrieval_query_ids": torch.tensor(retrieval_rows, dtype=torch.long),
        "retrieval_query_attention_mask": torch.tensor(
            retrieval_mask_rows, dtype=torch.bool
        ),
        "generator_input_ids": torch.tensor(generator_rows, dtype=torch.long),
        "generator_attention_mask": torch.tensor(
            generator_attention_rows, dtype=torch.bool
        ),
        "generator_target_ids": torch.tensor(target_rows, dtype=torch.long),
        "generator_loss_mask": torch.tensor(loss_mask_rows, dtype=torch.bool),
    }
    batches = tuple(
        LandmarkRetrofitBatch(
            **{name: value[index : index + size] for name, value in tensors.items()}
        )
        for index in range(0, len(cases), size)
    )
    return batches, {
        "surface": "marulho_landmark_retrofit_batches.v1",
        "case_count": len(cases),
        "batch_count": len(batches),
        "batch_size": size,
        "block_tokens": block,
        "maximum_blocks": blocks,
        "query_length": query,
        "minimum_block_count": min(block_counts),
        "maximum_block_count": max(block_counts),
        "boundary_spanning_answer_count": boundary_spanning_count,
        "minimum_answer_tokens": min(answer_lengths),
        "maximum_answer_tokens": max(answer_lengths),
        "all_gold_evidence_contains_answer_union": True,
        "all_retrieval_queries_answer_free": True,
        "all_generator_sequences_fit": True,
    }


@torch.no_grad()
def cache_landmark_retrofit_hidden(
    base: MarulhoLanguageModel,
    batches: Sequence[LandmarkRetrofitBatch],
    *,
    device: torch.device,
) -> tuple[tuple[LandmarkRetrofitBatch, ...], dict[str, Any]]:
    """Cache frozen V39 block/query states once in host BF16 memory."""

    base.eval()
    digest = hashlib.sha256()
    cached = []
    token_count = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for index, batch in enumerate(batches):
        source_ids = batch.source_ids.to(device)
        source_mask = batch.source_attention_mask.to(device)
        flat_ids = source_ids.flatten(0, 1)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            flat_hidden = base._forward_hidden(flat_ids, collect_telemetry=False)[
                "hidden"
            ]
            retrieval_hidden = base._forward_hidden(
                batch.retrieval_query_ids.to(device), collect_telemetry=False
            )["hidden"]
            generator_hidden = base._forward_hidden(
                batch.generator_input_ids.to(device), collect_telemetry=False
            )["hidden"]
        source_hidden = flat_hidden.reshape(
            *source_ids.shape, int(base.config.state_dim)
        )
        source_hidden = source_hidden.masked_fill(~source_mask.unsqueeze(-1), 0.0)
        host_values = (
            source_hidden.detach().to("cpu", dtype=torch.bfloat16).contiguous(),
            retrieval_hidden.detach().to("cpu", dtype=torch.bfloat16).contiguous(),
            generator_hidden.detach().to("cpu", dtype=torch.bfloat16).contiguous(),
        )
        for value in host_values:
            digest.update(value.view(torch.uint16).numpy().tobytes())
        cached.append(
            replace(
                batch,
                source_hidden=host_values[0],
                retrieval_query_hidden=host_values[1],
                generator_hidden=host_values[2],
            )
        )
        token_count += int(flat_ids.numel())
        token_count += int(batch.retrieval_query_ids.numel())
        token_count += int(batch.generator_input_ids.numel())
        if (index + 1) % 16 == 0 or index + 1 == len(batches):
            print(f"[landmark-cache] batch {index + 1}/{len(batches)}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    storage_bytes = sum(
        int(value.numel() * value.element_size())
        for batch in cached
        for value in (
            batch.source_hidden,
            batch.retrieval_query_hidden,
            batch.generator_hidden,
        )
        if value is not None
    )
    return tuple(cached), {
        "surface": "marulho_landmark_retrofit_hidden_cache.v1",
        "batch_count": len(cached),
        "case_count": sum(int(batch.source_ids.shape[0]) for batch in cached),
        "encoded_padded_token_count": token_count,
        "hidden_width": int(base.config.state_dim),
        "dtype": "torch.bfloat16",
        "host_storage_bytes": storage_bytes,
        "elapsed_seconds": elapsed,
        "tokens_per_second": token_count / elapsed,
        "content_sha256": digest.hexdigest(),
        "durable": False,
    }


class FrozenBaseLandmarkRetrofit(nn.Module):
    """Retrieve frozen source blocks and inject them into V39 generation."""

    surface = "marulho_frozen_base_landmark_retrofit.v1"

    def __init__(
        self,
        base: MarulhoLanguageModel,
        *,
        tokenizer: LanguageTokenizer,
        pad_id: int,
        eos_id: int,
        block_tokens: int = 48,
        maximum_blocks: int = 5,
        retrieval_width: int = 128,
        adapter_width: int = 256,
        adapter_layers: int = 2,
        adapter_heads: int = 8,
    ) -> None:
        super().__init__()
        self.base = base
        self.tokenizer = tokenizer
        self.base.requires_grad_(False)
        self.block_tokens = int(block_tokens)
        self.maximum_blocks = int(maximum_blocks)
        self.retrieval_width = int(retrieval_width)
        self.adapter_width = int(adapter_width)
        self.adapter_layers = int(adapter_layers)
        self.adapter_heads = int(adapter_heads)
        self.pad_id = int(pad_id)
        self.eos_id = int(eos_id)
        hidden = int(base.config.state_dim)
        self.query_projection = nn.Linear(hidden, self.retrieval_width)
        self.landmark_projection = nn.Linear(hidden, self.retrieval_width)
        self.query_norm = nn.LayerNorm(self.retrieval_width)
        self.landmark_norm = nn.LayerNorm(self.retrieval_width)
        self.causal_projection = nn.Linear(hidden, self.adapter_width)
        self.evidence_projection = nn.Linear(hidden, self.adapter_width)
        self.evidence_block_embedding = nn.Embedding(2, self.adapter_width)
        self.evidence_position_embedding = nn.Embedding(
            self.block_tokens, self.adapter_width
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.adapter_width,
            nhead=self.adapter_heads,
            dim_feedforward=self.adapter_width * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.evidence_adapter = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.adapter_layers,
            norm=nn.LayerNorm(self.adapter_width),
        )
        self.residual_projection = nn.Linear(self.adapter_width, hidden)
        self.residual_gate = nn.Parameter(torch.tensor(-2.0))

    @property
    def device(self) -> torch.device:
        return self.query_projection.weight.device

    @property
    def context_length(self) -> int:
        return self.base.context_length

    def retrofit_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("base.")
        )

    def retrofit_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in self.state_dict().items()
            if not name.startswith("base.")
        }

    def retrieval_scores(
        self,
        source_hidden: torch.Tensor,
        source_attention_mask: torch.Tensor,
        block_valid_mask: torch.Tensor,
        retrieval_query_hidden: torch.Tensor,
        retrieval_query_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        source_mask = source_attention_mask.to(self.device, dtype=torch.bool)
        source = source_hidden.to(
            self.device, dtype=self.landmark_projection.weight.dtype
        )
        denominator = source_mask.sum(dim=2, keepdim=True).clamp_min(1)
        landmarks = (source * source_mask.unsqueeze(-1)).sum(dim=2) / denominator
        query_mask = retrieval_query_attention_mask.to(self.device, dtype=torch.bool)
        last = query_mask.to(dtype=torch.long).sum(dim=1).sub(1).clamp_min(0)
        query_hidden = retrieval_query_hidden.to(
            self.device, dtype=self.query_projection.weight.dtype
        )
        query = query_hidden[
            torch.arange(query_hidden.shape[0], device=self.device), last
        ]
        query = self.query_norm(self.query_projection(query))
        landmarks = self.landmark_norm(self.landmark_projection(landmarks))
        scores = torch.einsum("br,bkr->bk", query, landmarks) / math.sqrt(
            float(self.retrieval_width)
        )
        return scores.float().masked_fill(
            ~block_valid_mask.to(self.device, dtype=torch.bool), -1.0e4
        )

    def select_evidence(
        self,
        source_hidden: torch.Tensor,
        source_attention_mask: torch.Tensor,
        scores: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected_indices = indices.to(self.device, dtype=torch.long)
        batch = torch.arange(source_hidden.shape[0], device=self.device).unsqueeze(1)
        source = source_hidden.to(
            self.device, dtype=self.evidence_projection.weight.dtype
        )
        source_mask = source_attention_mask.to(self.device, dtype=torch.bool)
        evidence = source[batch, selected_indices]
        evidence_mask = source_mask[batch, selected_indices]
        selected_scores = scores.gather(1, selected_indices)
        score_gate = 0.25 + 0.75 * torch.sigmoid(selected_scores)
        return evidence, evidence_mask, score_gate

    def adapter_logits(
        self,
        generator_hidden: torch.Tensor,
        generator_attention_mask: torch.Tensor,
        evidence_hidden: torch.Tensor,
        evidence_attention_mask: torch.Tensor,
        evidence_score_gate: torch.Tensor,
    ) -> torch.Tensor:
        causal_hidden = generator_hidden.to(
            self.device, dtype=self.causal_projection.weight.dtype
        )
        causal_mask = generator_attention_mask.to(self.device, dtype=torch.bool)
        evidence = self.evidence_projection(evidence_hidden.to(self.device))
        batch_size, selected_blocks, block_tokens, _width = evidence.shape
        positions = torch.arange(block_tokens, device=self.device)
        block_ids = torch.arange(selected_blocks, device=self.device)
        evidence = (
            evidence
            + self.evidence_position_embedding(positions)[None, None, :, :]
            + self.evidence_block_embedding(block_ids)[None, :, None, :]
        )
        evidence = evidence * evidence_score_gate[:, :, None, None]
        evidence = evidence.reshape(batch_size, selected_blocks * block_tokens, -1)
        evidence_mask = evidence_attention_mask.to(
            self.device, dtype=torch.bool
        ).reshape(batch_size, selected_blocks * block_tokens)
        query = self.causal_projection(causal_hidden)
        length = int(query.shape[1])
        attention = torch.ones(
            length, length, device=self.device, dtype=torch.bool
        ).triu(diagonal=1)
        adapted = self.evidence_adapter(
            query,
            evidence,
            tgt_mask=attention,
            tgt_key_padding_mask=~causal_mask,
            memory_key_padding_mask=~evidence_mask,
        )
        hidden = causal_hidden + torch.sigmoid(
            self.residual_gate
        ) * self.residual_projection(adapted)
        return self.base.lm_head(hidden).float()

    def loss(
        self,
        batch: LandmarkRetrofitBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if (
            batch.source_hidden is None
            or batch.retrieval_query_hidden is None
            or batch.generator_hidden is None
        ):
            raise ValueError("landmark training requires frozen hidden caches")
        scores = self.retrieval_scores(
            batch.source_hidden,
            batch.source_attention_mask,
            batch.block_valid_mask,
            batch.retrieval_query_hidden,
            batch.retrieval_query_attention_mask,
        )
        valid = batch.block_valid_mask.to(self.device, dtype=torch.bool)
        gold = batch.gold_block_mask.to(self.device, dtype=torch.float32)
        retrieval_loss = F.binary_cross_entropy_with_logits(scores[valid], gold[valid])
        evidence, evidence_mask, score_gate = self.select_evidence(
            batch.source_hidden,
            batch.source_attention_mask,
            scores,
            batch.gold_evidence_indices,
        )
        logits = self.adapter_logits(
            batch.generator_hidden,
            batch.generator_attention_mask,
            evidence,
            evidence_mask,
            score_gate,
        )
        targets = batch.generator_target_ids.to(self.device, dtype=torch.long)
        loss_mask = batch.generator_loss_mask.to(self.device, dtype=torch.bool)
        token_losses = F.cross_entropy(
            logits.flatten(0, 1), targets.flatten(), reduction="none"
        ).reshape_as(targets)
        generator_loss = (token_losses * loss_mask).sum() / loss_mask.sum().clamp_min(1)
        total = generator_loss + retrieval_loss
        return total, {
            "generator_loss": generator_loss.detach(),
            "retrieval_loss": retrieval_loss.detach(),
        }

    def _split_long_prompt(
        self, prompt: torch.Tensor
    ) -> (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None
    ):
        if int(prompt.shape[0]) != 1:
            raise ValueError("long-prompt splitting currently requires one row")
        prompt_ids = [int(value) for value in prompt[0].detach().cpu().tolist()]
        text = self.tokenizer.decode(prompt_ids)
        context_prefix = "Context: "
        question_marker = "\nQuestion: "
        answer_marker = "\nAnswer:"
        if not text.startswith(context_prefix) or answer_marker not in text:
            return None
        question_start = text.rfind(question_marker)
        answer_start = text.rfind(answer_marker)
        if question_start < len(context_prefix) or answer_start < question_start:
            return None
        source_text = text[len(context_prefix) : question_start]
        question_text = text[question_start + len(question_marker) : answer_start]
        canonical = (
            f"{context_prefix}{source_text}{question_marker}"
            f"{question_text}{answer_marker}"
        )
        canonical_ids = self.tokenizer.encode(canonical, add_bos=True, add_eos=False)
        if canonical_ids != prompt_ids:
            raise ValueError("V56 prompt does not round-trip through its tokenizer")
        source = torch.tensor(
            self.tokenizer.encode(source_text, add_bos=False, add_eos=False),
            device=self.device,
            dtype=torch.long,
        )
        generator_query = torch.tensor(
            self.tokenizer.encode(
                f"Question: {question_text}{answer_marker}",
                add_bos=True,
                add_eos=False,
            ),
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0)
        retrieval_query = torch.tensor(
            self.tokenizer.encode(question_text, add_bos=True, add_eos=False),
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0)
        block_count = math.ceil(int(source.numel()) / self.block_tokens)
        if not 1 <= block_count <= self.maximum_blocks:
            raise ValueError("runtime source block count exceeds V56 contract")
        source_ids = torch.full(
            (1, self.maximum_blocks, self.block_tokens),
            self.pad_id,
            device=self.device,
            dtype=torch.long,
        )
        source_mask = torch.zeros_like(source_ids, dtype=torch.bool)
        for index in range(block_count):
            values = source[index * self.block_tokens : (index + 1) * self.block_tokens]
            source_ids[0, index, : values.numel()] = values
            source_mask[0, index, : values.numel()] = True
        valid = torch.tensor(
            [[index < block_count for index in range(self.maximum_blocks)]],
            device=self.device,
            dtype=torch.bool,
        )
        return source_ids, source_mask, valid, generator_query, retrieval_query

    @torch.no_grad()
    def _runtime_evidence(
        self,
        prompt: torch.Tensor,
        *,
        evidence_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        split = self._split_long_prompt(prompt)
        if split is None:
            return None
        source_ids, source_mask, valid, generator_query, retrieval_query = split
        flat = source_ids.flatten(0, 1)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            source_hidden = self.base._forward_hidden(flat, collect_telemetry=False)[
                "hidden"
            ].reshape(
                1,
                self.maximum_blocks,
                self.block_tokens,
                int(self.base.config.state_dim),
            )
            retrieval_query_hidden = self.base._forward_hidden(
                retrieval_query, collect_telemetry=False
            )["hidden"]
        scores = self.retrieval_scores(
            source_hidden,
            source_mask,
            valid,
            retrieval_query_hidden,
            torch.ones_like(retrieval_query, dtype=torch.bool),
        )
        indices = (
            scores.topk(k=min(2, int(valid.sum().item())), dim=1).indices
            if evidence_indices is None
            else evidence_indices.to(self.device).reshape(1, 2)
        )
        if int(indices.shape[1]) == 1:
            indices = indices.repeat(1, 2)
        evidence, evidence_mask, score_gate = self.select_evidence(
            source_hidden, source_mask, scores, indices
        )
        return generator_query, evidence, evidence_mask, score_gate

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
        evidence_indices: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if eos_id is not None and int(eos_id) != self.eos_id:
            raise ValueError("landmark retrofit EOS differs from tokenizer")
        if float(temperature) != 0.0 or float(top_p) != 1.0 or seed is not None:
            raise ValueError("V56 evidence generation is deterministic greedy")
        prompt = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        prompt = prompt.to(self.device, dtype=torch.long)
        runtime = self._runtime_evidence(prompt, evidence_indices=evidence_indices)
        if runtime is None:
            return self.base.generate(
                prompt,
                max_new_tokens=int(max_new_tokens),
                eos_id=self.eos_id,
                repetition_penalty=float(repetition_penalty),
                no_repeat_ngram_size=int(no_repeat_ngram_size),
            )
        if int(prompt.shape[0]) != 1:
            raise ValueError("V56 evidence generation currently requires one row")
        query, evidence, evidence_mask, score_gate = runtime
        continuation = []
        finished = False
        for _step in range(max(0, int(max_new_tokens))):
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                query_hidden = self.base._forward_hidden(
                    query, collect_telemetry=False
                )["hidden"]
                logits = self.adapter_logits(
                    query_hidden,
                    torch.ones_like(query, dtype=torch.bool),
                    evidence,
                    evidence_mask,
                    score_gate,
                )[:, -1]
            history = (
                torch.stack(continuation, dim=1)
                if continuation
                else torch.empty(1, 0, device=self.device, dtype=torch.long)
            )
            controlled, _control = _apply_decode_controls(
                logits,
                history,
                repetition_penalty=max(1.0, float(repetition_penalty)),
                no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
            )
            next_id = controlled.argmax(dim=-1)
            if finished:
                next_id.fill_(self.eos_id)
            continuation.append(next_id)
            finished = finished or bool(next_id.eq(self.eos_id).all().item())
            query = torch.cat((query, next_id.unsqueeze(1)), dim=1)
            if int(query.shape[1]) > self.context_length:
                raise ValueError("V56 generated query exceeded the V39 context")
        generated = torch.cat(
            (
                prompt,
                torch.stack(continuation, dim=1)
                if continuation
                else torch.empty(1, 0, device=self.device, dtype=torch.long),
            ),
            dim=1,
        )
        return {
            "generated_ids": generated,
            "generated_token_count": len(continuation),
            "surface": self.surface,
            "decode_kind": "landmark_retrieved_full_vocabulary",
            "owned_by_marulho": True,
            "external_llm_used": False,
        }


def save_landmark_retrofit_checkpoint(
    path: str | Path,
    model: FrozenBaseLandmarkRetrofit,
    *,
    parent_checkpoint_sha256: str,
    metadata: Mapping[str, Any],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    payload = {
        "surface": LANDMARK_RETROFIT_CHECKPOINT_SURFACE,
        "parent_checkpoint_sha256": str(parent_checkpoint_sha256),
        "configuration": {
            "block_tokens": model.block_tokens,
            "maximum_blocks": model.maximum_blocks,
            "retrieval_width": model.retrieval_width,
            "adapter_width": model.adapter_width,
            "adapter_layers": model.adapter_layers,
            "adapter_heads": model.adapter_heads,
            "pad_id": model.pad_id,
            "eos_id": model.eos_id,
        },
        "retrofit_state": model.retrofit_state_dict(),
        "metadata": dict(metadata),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_landmark_retrofit_checkpoint(
    path: str | Path,
    base: MarulhoLanguageModel,
    tokenizer: LanguageTokenizer,
    *,
    expected_parent_checkpoint_sha256: str,
) -> tuple[FrozenBaseLandmarkRetrofit, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if str(payload.get("surface")) != LANDMARK_RETROFIT_CHECKPOINT_SURFACE:
        raise ValueError("landmark retrofit checkpoint surface is incompatible")
    if str(payload.get("parent_checkpoint_sha256")) != str(
        expected_parent_checkpoint_sha256
    ):
        raise ValueError("landmark retrofit parent checkpoint differs")
    configuration = dict(payload["configuration"])
    model = FrozenBaseLandmarkRetrofit(
        base,
        tokenizer=tokenizer,
        block_tokens=int(configuration["block_tokens"]),
        maximum_blocks=int(configuration["maximum_blocks"]),
        retrieval_width=int(configuration["retrieval_width"]),
        adapter_width=int(configuration["adapter_width"]),
        adapter_layers=int(configuration["adapter_layers"]),
        adapter_heads=int(configuration["adapter_heads"]),
        pad_id=int(configuration["pad_id"]),
        eos_id=int(configuration["eos_id"]),
    )
    merged = {f"base.{name}": value for name, value in base.state_dict().items()}
    merged.update(dict(payload["retrofit_state"]))
    model.load_state_dict(merged, strict=True)
    return model, dict(payload.get("metadata") or {})
