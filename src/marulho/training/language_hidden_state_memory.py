"""Frozen-cortex hidden-state episodic memory for MARULHO language models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping
from uuid import uuid4

import torch
from torch import nn
import torch.nn.functional as F

from marulho.training.language_answer_objective import answer_target_mask
from marulho.training.language_model import MarulhoLanguageModel, _apply_decode_controls


MEMORY_SURFACE = "marulho_hidden_state_episodic_memory.v1"


@dataclass(frozen=True)
class HiddenStateMemoryConfig:
    top_k: int = 8
    similarity_threshold: float = 0.75
    interpolation_weight: float = 0.5
    temperature: float = 0.1

    def validate(self) -> None:
        if int(self.top_k) < 1:
            raise ValueError("top_k must be positive")
        if not -1.0 <= float(self.similarity_threshold) <= 1.0:
            raise ValueError("similarity_threshold must be in [-1, 1]")
        if not 0.0 <= float(self.interpolation_weight) <= 1.0:
            raise ValueError("interpolation_weight must be in [0, 1]")
        if float(self.temperature) <= 0.0:
            raise ValueError("temperature must be positive")


class HiddenStateEpisodicMemory(nn.Module):
    """Normalized hidden keys with next-token values and bounded top-k reads."""

    def __init__(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        config: HiddenStateMemoryConfig,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        if keys.ndim != 2 or values.ndim != 1:
            raise ValueError("memory expects [entries,width] keys and [entries] values")
        if int(keys.shape[0]) != int(values.shape[0]) or int(keys.shape[0]) < 1:
            raise ValueError("memory keys and values must have the same nonzero size")
        normalized = F.normalize(keys.detach().float(), dim=-1).to(torch.float16)
        self.register_buffer("keys", normalized.contiguous())
        self.register_buffer("values", values.detach().long().contiguous())
        self.config = config
        self.metadata = dict(metadata or {})
        self.reset_metrics()

    @property
    def entry_count(self) -> int:
        return int(self.keys.shape[0])

    @property
    def key_width(self) -> int:
        return int(self.keys.shape[1])

    def reset_metrics(self) -> None:
        self._query_count = 0
        self._active_query_count = 0
        self._full_key_comparisons = 0
        self._retrieved_value_count = 0
        self._retrieval_seconds = 0.0
        self._top_similarity_sum = 0.0

    def metrics(self) -> dict[str, Any]:
        return {
            "query_count": int(self._query_count),
            "active_query_count": int(self._active_query_count),
            "active_query_fraction": float(self._active_query_count)
            / float(max(1, self._query_count)),
            "full_key_comparisons": int(self._full_key_comparisons),
            "retrieved_value_count": int(self._retrieved_value_count),
            "retrieved_entry_fraction_per_active_query": float(
                min(int(self.config.top_k), self.entry_count)
            )
            / float(self.entry_count),
            "full_search_entry_fraction_per_query": 1.0,
            "retrieval_seconds": float(self._retrieval_seconds),
            "mean_top_similarity": float(self._top_similarity_sum)
            / float(max(1, self._query_count)),
            "search_is_dense": True,
            "read_is_top_k": True,
            "sparse_gpu_compute_claim": False,
        }

    def fuse_logits(
        self,
        logits: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if logits.ndim != 2 or hidden.ndim != 2:
            raise ValueError("memory fusion expects [queries,vocab] and [queries,width]")
        if int(logits.shape[0]) != int(hidden.shape[0]):
            raise ValueError("memory query batches must match")
        if int(hidden.shape[1]) != self.key_width:
            raise ValueError("memory key width does not match the cortex hidden width")
        started = time.perf_counter()
        query = F.normalize(hidden.float(), dim=-1).to(self.keys.dtype)
        similarities = query @ self.keys.transpose(0, 1)
        selected_k = min(int(self.config.top_k), self.entry_count)
        top_scores, top_indices = torch.topk(similarities, selected_k, dim=-1)
        selected_values = self.values[top_indices]
        active = top_scores[:, 0] >= float(self.config.similarity_threshold)
        neighbor_weights = torch.softmax(
            top_scores.float() / float(self.config.temperature),
            dim=-1,
        )
        memory_probabilities = torch.zeros(
            int(logits.shape[0]),
            int(logits.shape[1]),
            device=logits.device,
            dtype=torch.float32,
        )
        memory_probabilities.scatter_add_(1, selected_values, neighbor_weights)
        mix = active.to(torch.float32) * float(self.config.interpolation_weight)
        base_probabilities = torch.softmax(logits.float(), dim=-1)
        fused_probabilities = (
            base_probabilities * (1.0 - mix.unsqueeze(1))
            + memory_probabilities * mix.unsqueeze(1)
        )
        fused = torch.log(fused_probabilities.clamp_min(torch.finfo(torch.float32).tiny))
        if logits.device.type == "cuda":
            torch.cuda.synchronize(logits.device)
        elapsed = time.perf_counter() - started
        query_count = int(logits.shape[0])
        active_count = int(active.sum().item())
        self._query_count += query_count
        self._active_query_count += active_count
        self._full_key_comparisons += query_count * self.entry_count
        self._retrieved_value_count += active_count * selected_k
        self._retrieval_seconds += elapsed
        self._top_similarity_sum += float(top_scores[:, 0].sum().item())
        return fused, {
            "query_count": query_count,
            "active_query_count": active_count,
            "selected_k": selected_k,
            "entry_count": self.entry_count,
            "full_key_comparisons": query_count * self.entry_count,
            "retrieval_seconds": elapsed,
            "top_similarity_mean": float(top_scores[:, 0].mean().item()),
        }


class HiddenStateMemoryLanguageModel(nn.Module):
    """Read-only V39 cortex with answer-gated hidden-state memory fusion."""

    def __init__(
        self,
        model: MarulhoLanguageModel,
        memory: HiddenStateEpisodicMemory,
        *,
        answer_marker_ids: torch.Tensor,
        eos_id: int,
    ) -> None:
        super().__init__()
        if memory.key_width != int(model.config.state_dim):
            raise ValueError("memory and cortex widths must match")
        self.model = model
        self.memory = memory
        self.register_buffer(
            "answer_marker_ids",
            answer_marker_ids.detach().long().flatten().contiguous(),
        )
        self.eos_id = int(eos_id)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.eval()

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def context_length(self) -> int:
        return self.model.context_length

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> dict[str, Any]:
        result = self.model._forward_hidden(  # noqa: SLF001 - explicit experiment seam
            input_ids,
            state,
            collect_telemetry=collect_telemetry,
        )
        logits = self.model.lm_head(result["hidden"])
        mask = answer_target_mask(
            input_ids.to(self.device),
            marker_ids=self.answer_marker_ids,
            eos_id=self.eos_id,
        )
        active_positions = mask.nonzero(as_tuple=False)
        retrieval: dict[str, Any] | None = None
        if int(active_positions.shape[0]) > 0:
            rows = active_positions[:, 0]
            columns = active_positions[:, 1]
            fused, retrieval = self.memory.fuse_logits(
                logits[rows, columns],
                result["hidden"][rows, columns],
            )
            logits = logits.clone()
            logits[rows, columns] = fused.to(logits.dtype)
        return {
            "logits": logits,
            "state": result["state"],
            "telemetry": {
                **result["telemetry"],
                "hidden_state_memory": retrieval,
                "answer_memory_active_position_count": int(active_positions.shape[0]),
            },
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
    ) -> dict[str, Any]:
        prompt_batch = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        if prompt_batch.ndim != 2 or int(prompt_batch.shape[1]) < 1:
            raise ValueError("prompt_ids must be [time] or [batch,time]")
        prompt_batch = prompt_batch.to(self.device, dtype=torch.long)
        if int(prompt_batch.shape[1]) > self.context_length:
            prompt_batch = prompt_batch[:, -self.context_length :]
        requested = max(0, int(max_new_tokens))
        prompt_width = int(prompt_batch.shape[1])
        generated = torch.empty(
            int(prompt_batch.shape[0]),
            prompt_width + requested,
            device=self.device,
            dtype=torch.long,
        )
        generated[:, :prompt_width].copy_(prompt_batch)
        initial = self.model._forward_hidden(  # noqa: SLF001
            prompt_batch,
            collect_telemetry=False,
        )
        state = initial["state"]
        next_hidden = initial["hidden"][:, -1]
        next_logits = self.model.lm_head(next_hidden)
        active = answer_target_mask(
            prompt_batch,
            marker_ids=self.answer_marker_ids,
            eos_id=self.eos_id,
        )[:, -1]
        finished = torch.zeros(
            int(prompt_batch.shape[0]),
            device=self.device,
            dtype=torch.bool,
        )
        new_token_count = 0
        for _ in range(requested):
            if bool(active.any().item()):
                fused, _evidence = self.memory.fuse_logits(
                    next_logits[active],
                    next_hidden[active],
                )
                next_logits = next_logits.clone()
                next_logits[active] = fused.to(next_logits.dtype)
            write_index = prompt_width + new_token_count
            history_start = max(0, write_index - self.context_length)
            controlled, _control = _apply_decode_controls(
                next_logits,
                generated[:, history_start:write_index],
                repetition_penalty=max(1.0, float(repetition_penalty)),
                no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
            )
            next_id = torch.argmax(controlled, dim=-1, keepdim=True)
            effective_eos = self.eos_id if eos_id is None else int(eos_id)
            next_id = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_id, effective_eos),
                next_id,
            )
            generated[:, write_index : write_index + 1].copy_(next_id)
            new_token_count += 1
            emitted_eos = next_id[:, 0] == effective_eos
            active = active & ~emitted_eos
            finished = finished | emitted_eos
            if bool(finished.all().item()):
                break
            hidden, state, _telemetry = self.model.state_block.step(
                self.model.token_embedding(next_id[:, 0]),
                state,
                collect_telemetry=False,
            )
            next_hidden = hidden
            next_logits = self.model.lm_head(hidden)
        generated = generated[:, : prompt_width + new_token_count]
        return {
            "surface": "marulho_hidden_state_memory_generation.v1",
            "generated_ids": generated,
            "new_token_count": new_token_count,
            "state": state,
            "generation_decode": self.model.generation_decode_policy(
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                decode_control_window=self.context_length,
            ),
            "owned_by_marulho": True,
            "external_llm_used": False,
        }


def memory_state_sha256(memory: HiddenStateEpisodicMemory) -> str:
    digest = hashlib.sha256()
    digest.update(memory.keys.detach().cpu().contiguous().numpy().tobytes())
    digest.update(memory.values.detach().cpu().contiguous().numpy().tobytes())
    digest.update(
        json.dumps(asdict(memory.config), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


def save_hidden_state_memory(
    path: str | Path,
    memory: HiddenStateEpisodicMemory,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "surface": MEMORY_SURFACE,
        "keys": memory.keys.detach().cpu(),
        "values": memory.values.detach().cpu(),
        "config": asdict(memory.config),
        "metadata": dict(memory.metadata),
        "state_sha256": memory_state_sha256(memory),
        "owned_by_marulho": True,
        "external_llm_used": False,
    }
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


def load_hidden_state_memory(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> HiddenStateEpisodicMemory:
    payload = torch.load(Path(path), map_location="cpu")
    if payload.get("surface") != MEMORY_SURFACE:
        raise ValueError("rejected hidden-state memory surface")
    memory = HiddenStateEpisodicMemory(
        payload["keys"],
        payload["values"],
        config=HiddenStateMemoryConfig(**dict(payload["config"])),
        metadata=payload.get("metadata"),
    ).to(device)
    if memory_state_sha256(memory) != str(payload.get("state_sha256")):
        raise ValueError("hidden-state memory failed exact state verification")
    return memory
