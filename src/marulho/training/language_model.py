"""Transformer-only MARULHO language model, training batches, and checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

import torch
from torch import nn
import torch.nn.functional as F

from marulho.data.language_tokenizer import (
    LANGUAGE_DOCUMENT_ENCODE_BATCH_SIZE,
    LANGUAGE_DOCUMENT_SEPARATOR,
    LanguageTokenizer,
    iter_language_corpus_documents,
    load_language_tokenizer_state,
)
from marulho.training.language_transformer import MarulhoCausalTransformerStateBlock


LANGUAGE_STATE_CORE_KINDS = ("transformer",)
CHECKPOINT_SURFACE = "marulho_transformer_language_checkpoint.v2"


@dataclass(frozen=True)
class LanguageModelConfig:
    vocab_size: int
    embedding_dim: int = 256
    state_dim: int = 256
    state_core: str = "transformer"
    state_layers: int = 4
    attention_heads: int = 8
    transformer_context_length: int = 512
    transformer_mlp_ratio: float = 4.0
    transformer_dropout: float = 0.0
    tie_embeddings: bool = True
    output_adapter_rank: int = 0
    output_adapter_scale: float = 1.0
    active_language_path: str = "marulho_transformer"


@dataclass(frozen=True)
class LanguageBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor

    def to(self, device: torch.device | str) -> "LanguageBatch":
        return LanguageBatch(
            input_ids=self.input_ids.to(device),
            target_ids=self.target_ids.to(device),
        )


@dataclass(frozen=True)
class LanguageSplit:
    train: tuple[LanguageBatch, ...]
    eval: tuple[LanguageBatch, ...]
    report: dict[str, Any]


def _validate_config(config: LanguageModelConfig) -> None:
    if int(config.vocab_size) <= 1:
        raise ValueError("vocab_size must be greater than one")
    if str(config.state_core).strip().lower() != "transformer":
        raise ValueError("MARULHO language checkpoints now support only state_core='transformer'")
    if int(config.embedding_dim) <= 0 or int(config.state_dim) <= 0:
        raise ValueError("embedding_dim and state_dim must be positive")
    if int(config.state_layers) < 1:
        raise ValueError("state_layers must be at least one")
    if int(config.attention_heads) < 1:
        raise ValueError("attention_heads must be at least one")
    if int(config.state_dim) % int(config.attention_heads) != 0:
        raise ValueError("state_dim must be divisible by attention_heads")
    head_dim = int(config.state_dim) // int(config.attention_heads)
    if head_dim % 2 != 0:
        raise ValueError("attention head dimension must be even for rotary positions")
    if int(config.transformer_context_length) < 2:
        raise ValueError("transformer_context_length must be at least two")
    if not math.isfinite(float(config.transformer_mlp_ratio)):
        raise ValueError("transformer_mlp_ratio must be finite")
    if float(config.transformer_mlp_ratio) < 1.0:
        raise ValueError("transformer_mlp_ratio must be at least one")
    if not 0.0 <= float(config.transformer_dropout) < 1.0:
        raise ValueError("transformer_dropout must be in [0, 1)")
    if bool(config.tie_embeddings) and int(config.embedding_dim) != int(config.state_dim):
        raise ValueError("tie_embeddings requires embedding_dim == state_dim")
    if int(config.output_adapter_rank) < 0:
        raise ValueError("output_adapter_rank must be non-negative")
    if not math.isfinite(float(config.output_adapter_scale)):
        raise ValueError("output_adapter_scale must be finite")
    if float(config.output_adapter_scale) < 0.0:
        raise ValueError("output_adapter_scale must be non-negative")


def _valid_generated_tokens(token_ids: torch.Tensor, *, vocab_size: int) -> torch.Tensor:
    flat = token_ids.reshape(-1).to(dtype=torch.long)
    return flat[(flat >= 0) & (flat < int(vocab_size))]


def _banned_ngram_tokens(
    token_ids: torch.Tensor,
    *,
    ngram_size: int,
    vocab_size: int,
) -> torch.Tensor:
    size = max(0, int(ngram_size))
    tokens = _valid_generated_tokens(token_ids, vocab_size=vocab_size)
    if size <= 0 or int(tokens.numel()) < size - 1:
        return tokens.new_empty(0)
    if size == 1:
        return torch.unique(tokens)
    prefix = tokens[-(size - 1) :]
    windows = tokens.unfold(0, size, 1)
    if int(windows.shape[0]) <= 0:
        return tokens.new_empty(0)
    matches = (windows[:, :-1] == prefix.unsqueeze(0)).all(dim=1)
    return torch.unique(windows[matches, -1])


def _apply_decode_controls(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    *,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    adjusted = logits.clone()
    vocab_size = int(adjusted.shape[-1])
    generated = _valid_generated_tokens(generated_ids, vocab_size=vocab_size)
    penalty = max(1.0, float(repetition_penalty))
    repetition_count = 0
    if penalty > 1.0 and int(generated.numel()) > 0:
        seen = torch.unique(generated)
        values = adjusted.index_select(-1, seen)
        values = torch.where(values < 0, values * penalty, values / penalty)
        adjusted.index_copy_(-1, seen, values)
        repetition_count = int(seen.numel())
    banned = _banned_ngram_tokens(
        generated,
        ngram_size=no_repeat_ngram_size,
        vocab_size=vocab_size,
    )
    fallback = 0
    if int(banned.numel()) > 0:
        if int(banned.numel()) >= vocab_size:
            fallback = 1
        else:
            adjusted.index_fill_(-1, banned, float("-inf"))
    return adjusted, {
        "repetition_penalty_adjusted_token_count": repetition_count,
        "no_repeat_ngram_banned_token_count": int(banned.numel()),
        "decode_control_fallback_count": fallback,
    }


class MarulhoLanguageModel(nn.Module):
    """MARULHO-owned decoder-only Transformer language model."""

    surface = "marulho_transformer_language_model.v3"

    def __init__(self, config: LanguageModelConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.state_block = MarulhoCausalTransformerStateBlock(
            config.embedding_dim,
            config.state_dim,
            state_layers=config.state_layers,
            attention_heads=config.attention_heads,
            context_length=config.transformer_context_length,
            mlp_ratio=config.transformer_mlp_ratio,
            dropout=config.transformer_dropout,
        )
        self.lm_head = nn.Linear(config.state_dim, config.vocab_size, bias=False)
        adapter_rank = int(config.output_adapter_rank)
        self.output_adapter_down = (
            nn.Linear(config.state_dim, adapter_rank, bias=False)
            if adapter_rank > 0
            else None
        )
        self.output_adapter_up = (
            nn.Linear(adapter_rank, config.state_dim, bias=False)
            if adapter_rank > 0
            else None
        )
        if bool(config.tie_embeddings):
            self.lm_head.weight = self.token_embedding.weight
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                bias = getattr(module, "bias", None)
                if isinstance(bias, torch.Tensor):
                    nn.init.zeros_(bias)
        if self.output_adapter_up is not None:
            nn.init.zeros_(self.output_adapter_up.weight)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def generation_vocab_size(self) -> int:
        return int(self.config.vocab_size)

    def _adapt_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.output_adapter_down is None or self.output_adapter_up is None:
            return hidden
        adapted = self.output_adapter_up(
            F.silu(self.output_adapter_down(hidden))
        )
        return hidden + float(self.config.output_adapter_scale) * adapted

    def generation_decode_policy(
        self,
        *,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        sampling = float(temperature) > 0.0
        return {
            "surface": "marulho_transformer_decode_policy.v3",
            "decode_strategy": (
                "nucleus_sampling" if sampling else "greedy_argmax"
            ),
            "model_vocab_size": int(self.config.vocab_size),
            "generation_vocab_size": int(self.config.vocab_size),
            "full_model_vocab_logits_materialized": True,
            "repetition_penalty": max(1.0, float(repetition_penalty)),
            "no_repeat_ngram_size": max(0, int(no_repeat_ngram_size)),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "sampling_seed": None if seed is None else int(seed),
            "top_p_applied": bool(sampling and float(top_p) < 1.0),
            "kv_cache": "bounded_per_layer",
            "external_llm_used": False,
        }

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("Language model expects input_ids shaped [batch, time]")
        runtime_ids = input_ids.to(device=self.device, dtype=torch.long)
        hidden, next_state, transformer = self.state_block(
            self.token_embedding(runtime_ids),
            state,
            collect_telemetry=collect_telemetry,
        )
        telemetry = {
            **transformer,
            "active_language_path": self.config.active_language_path,
            "external_llm_used": False,
            "owned_by_marulho": True,
            "vocab_size": int(self.config.vocab_size),
            "output_adapter_rank": int(self.config.output_adapter_rank),
        }
        return {"hidden": hidden, "state": next_state, "telemetry": telemetry}

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        del decode_vocab_only
        result = self._forward_hidden(
            input_ids,
            state,
            collect_telemetry=collect_telemetry,
        )
        return {
            "logits": self.lm_head(self._adapt_hidden(result["hidden"])),
            "state": result["state"],
            "telemetry": result["telemetry"],
        }

    def forward_step(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        del decode_vocab_only
        if input_ids.ndim == 1:
            step_ids = input_ids.unsqueeze(1)
        elif input_ids.ndim == 2 and int(input_ids.shape[1]) == 1:
            step_ids = input_ids
        else:
            raise ValueError("forward_step expects [batch] or [batch, 1] token ids")
        runtime_ids = step_ids.to(device=self.device, dtype=torch.long)
        hidden, next_state, transformer = self.state_block.step(
            self.token_embedding(runtime_ids[:, 0]),
            state,
            collect_telemetry=collect_telemetry,
        )
        return {
            "logits": self.lm_head(self._adapt_hidden(hidden)).unsqueeze(1),
            "state": next_state,
            "telemetry": {
                **transformer,
                "active_language_path": self.config.active_language_path,
                "external_llm_used": False,
                "owned_by_marulho": True,
                "vocab_size": int(self.config.vocab_size),
                "output_adapter_rank": int(self.config.output_adapter_rank),
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
        targets = target_ids.to(device=self.device, dtype=torch.long)
        logits = output["logits"]
        if logits.shape[:2] != targets.shape:
            raise ValueError("target_ids must match input batch/time dimensions")
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        evidence = {
            "surface": "marulho_transformer_cross_entropy.v2",
            "sampled_vocab_training": False,
            "full_vocab_logits_materialized": True,
            "target_token_count": int(targets.numel()),
            "external_llm_used": False,
        }
        return {
            "loss": loss,
            "loss_kind": "full_vocab_cross_entropy",
            "loss_evidence": evidence if return_evidence else {},
            "state": output["state"],
            "telemetry": output["telemetry"],
        }

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
        temperature = float(temperature)
        top_p = float(top_p)
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be finite and non-negative")
        if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be finite and in (0, 1]")
        sample = temperature > 0.0
        sampling_generator = None
        if sample and seed is not None:
            sampling_generator = torch.Generator(device=self.device)
            sampling_generator.manual_seed(int(seed))
        was_training = bool(self.training)
        self.eval()
        try:
            if prompt_ids.ndim == 1:
                generated = prompt_ids.unsqueeze(0)
            elif prompt_ids.ndim == 2:
                generated = prompt_ids
            else:
                raise ValueError("prompt_ids must be [time] or [batch, time]")
            generated = generated.to(device=self.device, dtype=torch.long)
            if int(generated.shape[1]) > int(self.config.transformer_context_length):
                generated = generated[:, -int(self.config.transformer_context_length) :]
            result = self.forward(generated, collect_telemetry=False)
            state = result["state"]
            next_logits = result["logits"][:, -1]
            new_token_count = 0
            repetition_count = 0
            banned_count = 0
            fallback_count = 0
            for _ in range(max(0, int(max_new_tokens))):
                controlled, control = _apply_decode_controls(
                    next_logits,
                    generated,
                    repetition_penalty=max(1.0, float(repetition_penalty)),
                    no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
                )
                repetition_count += control["repetition_penalty_adjusted_token_count"]
                banned_count += control["no_repeat_ngram_banned_token_count"]
                fallback_count += control["decode_control_fallback_count"]
                if sample:
                    probabilities = torch.softmax(
                        controlled / temperature,
                        dim=-1,
                    )
                    if top_p < 1.0:
                        sorted_probabilities, sorted_indices = torch.sort(
                            probabilities,
                            dim=-1,
                            descending=True,
                        )
                        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
                        remove = cumulative > top_p
                        remove[..., 1:] = remove[..., :-1].clone()
                        remove[..., 0] = False
                        sorted_probabilities = sorted_probabilities.masked_fill(
                            remove,
                            0.0,
                        )
                        sorted_probabilities = sorted_probabilities / (
                            sorted_probabilities.sum(dim=-1, keepdim=True)
                        ).clamp_min(torch.finfo(sorted_probabilities.dtype).tiny)
                        sampled_rank = torch.multinomial(
                            sorted_probabilities,
                            1,
                            generator=sampling_generator,
                        )
                        next_id = sorted_indices.gather(-1, sampled_rank)
                    else:
                        next_id = torch.multinomial(
                            probabilities,
                            1,
                            generator=sampling_generator,
                        )
                else:
                    next_id = torch.argmax(controlled, dim=-1, keepdim=True)
                generated = torch.cat((generated, next_id), dim=1)
                new_token_count += 1
                if eos_id is not None and bool(torch.all(next_id == int(eos_id)).item()):
                    break
                result = self.forward_step(next_id, state, collect_telemetry=False)
                state = result["state"]
                next_logits = result["logits"][:, -1]
            return {
                "surface": "marulho_transformer_generation.v3",
                "generated_ids": generated,
                "new_token_count": int(new_token_count),
                "state": state,
                "active_language_path": self.config.active_language_path,
                "external_llm_used": False,
                "owned_by_marulho": True,
                "generation_decode": self.generation_decode_policy(
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                ),
                "decode_control_totals": {
                    "repetition_penalty_adjusted_token_count": repetition_count,
                    "no_repeat_ngram_banned_token_count": banned_count,
                    "decode_control_fallback_count": fallback_count,
                },
            }
        finally:
            self.train(was_training)


def build_language_model_splits(
    texts: Sequence[str],
    tokenizer: LanguageTokenizer,
    *,
    eval_texts: Sequence[str] | None = None,
    sequence_length: int,
    eval_fraction: float = 0.2,
    stride: int | None = None,
    batch_size: int = 1,
    device: torch.device | str | None = None,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
    window_selection: str = "stratified",
) -> LanguageSplit:
    if int(sequence_length) < 2:
        raise ValueError("sequence_length must be at least two")
    batch_size = max(1, int(batch_size))
    window_length = int(sequence_length) + 1
    step = int(stride or sequence_length)
    if step <= 0:
        raise ValueError("stride must be positive")

    def _token_stream(
        source_texts: Sequence[str],
        *,
        label: str,
    ) -> tuple[torch.Tensor, int, int, int, int]:
        token_chunks: list[torch.Tensor] = []
        text_token_count = 0
        document_count = 0
        explicit_separator_count = sum(
            str(text).count(LANGUAGE_DOCUMENT_SEPARATOR)
            for text in source_texts
        )
        pending_documents: list[str] = []

        def _flush_documents() -> None:
            nonlocal document_count, text_token_count
            if not pending_documents:
                return
            rows = tokenizer.encode_batch(
                pending_documents,
                add_bos=True,
                add_eos=True,
            )
            packed_ids: list[int] = []
            for encoded in rows:
                text_token_count += max(0, len(encoded) - 2)
                packed_ids.extend(encoded)
            token_chunks.append(
                torch.tensor(packed_ids, dtype=torch.long, device="cpu")
            )
            document_count += len(rows)
            pending_documents.clear()

        for document in iter_language_corpus_documents(source_texts):
            pending_documents.append(document)
            if len(pending_documents) >= LANGUAGE_DOCUMENT_ENCODE_BATCH_SIZE:
                _flush_documents()
        _flush_documents()
        if not token_chunks:
            raise ValueError(f"No {label} texts were provided")
        token_ids = (
            token_chunks[0]
            if len(token_chunks) == 1
            else torch.cat(token_chunks)
        )
        if int(token_ids.numel()) < window_length:
            raise ValueError(
                f"Not enough {label} tokens to build a next-token language split"
            )
        window_count = 1 + (int(token_ids.numel()) - window_length) // step
        return (
            token_ids,
            window_count,
            text_token_count,
            document_count,
            explicit_separator_count,
        )

    (
        train_token_ids,
        train_source_window_count,
        train_text_token_count,
        train_document_count,
        train_explicit_separator_count,
    ) = _token_stream(
        texts,
        label="training",
    )
    if eval_texts is not None:
        (
            eval_token_ids,
            eval_source_window_count,
            eval_text_token_count,
            eval_document_count,
            eval_explicit_separator_count,
        ) = _token_stream(
            eval_texts,
            label="evaluation",
        )
        train_window_count = train_source_window_count
        eval_window_count = eval_source_window_count
        train_window_offset = 0
        eval_window_offset = 0
        window_count = train_window_count + eval_window_count
        split_strategy = "explicit_text_sets"
    elif train_source_window_count == 1:
        eval_token_ids = train_token_ids
        eval_text_token_count = train_text_token_count
        eval_document_count = train_document_count
        eval_explicit_separator_count = train_explicit_separator_count
        train_window_count = 1
        eval_window_count = 1
        train_window_offset = 0
        eval_window_offset = 0
        window_count = 1
        split_strategy = "shared_single_window"
    else:
        eval_text_token_count = train_text_token_count
        eval_document_count = train_document_count
        eval_explicit_separator_count = train_explicit_separator_count
        eval_count = max(
            1,
            min(
                train_source_window_count - 1,
                math.ceil(train_source_window_count * eval_fraction),
            ),
        )
        eval_token_ids = train_token_ids
        train_window_count = train_source_window_count - eval_count
        eval_window_count = eval_count
        train_window_offset = 0
        eval_window_offset = train_window_count
        window_count = train_source_window_count
        split_strategy = "contiguous_tail_fraction"
    selection = str(window_selection).strip().lower()
    if selection not in {"stratified", "prefix"}:
        raise ValueError("window_selection must be 'stratified' or 'prefix'")

    def _limit(
        source_count: int,
        max_batches: int | None,
    ) -> tuple[list[int], dict[str, Any]]:
        maximum = source_count if not max_batches or int(max_batches) <= 0 else min(
            source_count,
            int(max_batches) * batch_size,
        )
        if maximum >= source_count:
            indices = list(range(source_count))
        elif selection == "prefix":
            indices = list(range(maximum))
        elif maximum == 1:
            indices = [source_count // 2]
        else:
            indices = [
                round(index * (source_count - 1) / float(maximum - 1))
                for index in range(maximum)
            ]
        return indices, {
            "source_window_count": source_count,
            "selected_window_count": len(indices),
            "first_selected_index": indices[0] if indices else None,
            "last_selected_index": indices[-1] if indices else None,
            "spans_full_source_window": bool(
                indices and indices[0] == 0 and indices[-1] == source_count - 1
            ),
        }

    train_before = train_window_count
    eval_before = eval_window_count
    train_selected, train_selection = _limit(
        train_window_count,
        max_train_batches,
    )
    eval_selected, eval_selection = _limit(
        eval_window_count,
        max_eval_batches,
    )
    target_device = torch.device(device or "cpu")

    def _pack(
        token_ids: torch.Tensor,
        relative_window_indices: Sequence[int],
        *,
        window_offset: int,
    ) -> tuple[tuple[LanguageBatch, ...], str]:
        batches: list[LanguageBatch] = []
        digest = hashlib.sha256()
        token_offsets = torch.arange(window_length, dtype=torch.long)
        transfer_window_count = batch_size * 256
        for chunk_offset in range(
            0,
            len(relative_window_indices),
            transfer_window_count,
        ):
            relative_indices = relative_window_indices[
                chunk_offset : chunk_offset + transfer_window_count
            ]
            starts = torch.tensor(
                [window_offset + int(index) for index in relative_indices],
                dtype=torch.long,
            ) * step
            windows = token_ids[starts.unsqueeze(1) + token_offsets.unsqueeze(0)]
            digest.update(windows.contiguous().numpy().tobytes())
            input_ids = windows[:, :-1].contiguous()
            target_ids = windows[:, 1:].contiguous()
            device_inputs = input_ids.to(target_device)
            device_targets = target_ids.to(target_device)
            for batch_offset in range(0, len(relative_indices), batch_size):
                batches.append(
                    LanguageBatch(
                        input_ids=device_inputs[
                            batch_offset : batch_offset + batch_size
                        ],
                        target_ids=device_targets[
                            batch_offset : batch_offset + batch_size
                        ],
                    )
                )
        return tuple(batches), digest.hexdigest()

    train_batches, train_split_hash = _pack(
        train_token_ids,
        train_selected,
        window_offset=train_window_offset,
    )
    eval_batches, eval_split_hash = _pack(
        eval_token_ids,
        eval_selected,
        window_offset=eval_window_offset,
    )
    report = {
        "surface": "marulho_transformer_train_eval_split.v8",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "sequence_length": int(sequence_length),
        "stride": step,
        "batch_size": batch_size,
        "source_text_count": len(texts),
        "eval_source_text_count": (
            0 if eval_texts is None else len(eval_texts)
        ),
        "split_strategy": split_strategy,
        "explicit_eval_texts": eval_texts is not None,
        "train_text_token_count": int(train_text_token_count),
        "train_token_stream_count": int(train_token_ids.numel()),
        "eval_text_token_count": int(eval_text_token_count),
        "eval_token_stream_count": int(eval_token_ids.numel()),
        "train_document_count": int(train_document_count),
        "eval_document_count": int(eval_document_count),
        "train_explicit_document_separator_count": int(
            train_explicit_separator_count
        ),
        "eval_explicit_document_separator_count": int(
            eval_explicit_separator_count
        ),
        "document_boundary_policy": (
            "bos_eos_per_explicit_marulho_record"
            if train_explicit_separator_count > 0
            else "bos_eos_per_legacy_blank_line_document"
        ),
        "split_hash_format": "selected_windows_int64_row_major.v1",
        "storage_device": str(target_device),
        "document_encode_batch_size": LANGUAGE_DOCUMENT_ENCODE_BATCH_SIZE,
        "window_count": window_count,
        "train_window_count_before_limit": train_before,
        "eval_window_count_before_limit": eval_before,
        "window_selection": selection,
        "train_window_selection": train_selection,
        "eval_window_selection": eval_selection,
        "train_window_count": len(train_selected),
        "eval_window_count": len(eval_selected),
        "train_batch_count": len(train_batches),
        "eval_batch_count": len(eval_batches),
        "tokenizer_hash": tokenizer.vocabulary_hash(),
        "train_split_hash": train_split_hash,
        "eval_split_hash": eval_split_hash,
    }
    return LanguageSplit(train=train_batches, eval=eval_batches, report=report)


@torch.no_grad()
def evaluate_language_model(
    model: MarulhoLanguageModel,
    batches: Sequence[LanguageBatch],
) -> dict[str, Any]:
    if not batches:
        raise ValueError("At least one evaluation batch is required")
    was_training = bool(model.training)
    model.eval()
    losses: list[torch.Tensor] = []
    token_count = 0
    started = time.perf_counter()
    try:
        for batch in batches:
            device_batch = batch.to(model.device)
            result = model.next_token_loss(
                device_batch.input_ids,
                device_batch.target_ids,
                collect_telemetry=False,
                return_evidence=False,
            )
            count = int(device_batch.target_ids.numel())
            losses.append(result["loss"].detach() * count)
            token_count += count
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        elapsed = max(time.perf_counter() - started, 1.0e-9)
        mean_loss = float((torch.stack(losses).sum() / max(1, token_count)).cpu().item())
        return {
            "surface": "marulho_transformer_heldout_evaluation.v3",
            "heldout_loss": mean_loss,
            "heldout_perplexity": float(math.exp(min(mean_loss, 20.0))),
            "token_count": token_count,
            "batch_count": len(batches),
            "batch_transfer_policy": "cpu_split_per_batch_to_model_device",
            "tokens_per_second": float(token_count) / elapsed,
            "external_llm_used": False,
            "owned_by_marulho": True,
        }
    finally:
        model.train(was_training)


def _validate_checkpoint_vocab(
    config: LanguageModelConfig,
    tokenizer: LanguageTokenizer,
) -> None:
    if int(config.vocab_size) != int(tokenizer.vocab_size):
        raise ValueError(
            "Transformer checkpoint vocab must exactly match its checkpoint-owned tokenizer"
        )


def language_model_checkpoint_payload(
    model: MarulhoLanguageModel,
    tokenizer: LanguageTokenizer,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_checkpoint_vocab(model.config, tokenizer)
    return {
        "artifact_kind": "marulho_transformer_language_checkpoint",
        "surface": CHECKPOINT_SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "active_language_path": model.config.active_language_path,
        "config": asdict(model.config),
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "tokenizer": tokenizer.state_dict(),
        "tokenizer_hash": tokenizer.vocabulary_hash(),
        "metadata": dict(metadata or {}),
    }


def load_language_model_state(
    model: MarulhoLanguageModel,
    state: Mapping[str, torch.Tensor],
) -> None:
    model.load_state_dict(dict(state), strict=True)


def save_language_model_checkpoint(
    path: str | Path,
    model: MarulhoLanguageModel,
    tokenizer: LanguageTokenizer,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = language_model_checkpoint_payload(model, tokenizer, metadata)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def load_language_model_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> tuple[MarulhoLanguageModel, LanguageTokenizer, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location or "cpu")
    if payload.get("surface") != CHECKPOINT_SURFACE:
        raise ValueError(
            "Rejected legacy language checkpoint; regenerate it with the Transformer-only path"
        )
    tokenizer = load_language_tokenizer_state(payload["tokenizer"])
    config = LanguageModelConfig(**dict(payload["config"]))
    _validate_checkpoint_vocab(config, tokenizer)
    model = MarulhoLanguageModel(config)
    load_language_model_state(model, payload["model_state"])
    return model, tokenizer, dict(payload.get("metadata") or {})
