"""Parameter-matched society of independent causal language cells."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from marulho.training.language_model import _apply_decode_controls
from marulho.training.language_transformer import MarulhoCausalTransformerStateBlock


@dataclass(frozen=True)
class ModularSocietyConfig:
    vocab_size: int
    cell_count: int = 4
    cell_width: int = 296
    cell_layers: int = 2
    attention_heads: int = 4
    context_length: int = 72
    mlp_ratio: float = 4.0
    event_interval: int = 24
    message_dim: int = 32
    mode: str = "learned_real"
    tie_embeddings: bool = True
    active_language_path: str = "marulho_modular_predictive_society_v3"


def _validate_config(config: ModularSocietyConfig) -> None:
    if int(config.vocab_size) <= 1:
        raise ValueError("vocab_size must exceed one")
    if int(config.cell_count) < 2:
        raise ValueError("cell_count must be at least two")
    if int(config.cell_width) % int(config.attention_heads) != 0:
        raise ValueError("cell_width must be divisible by attention_heads")
    if (int(config.cell_width) // int(config.attention_heads)) % 2 != 0:
        raise ValueError("cell attention head dimension must be even")
    if int(config.cell_layers) < 1 or int(config.context_length) < 2:
        raise ValueError("cell_layers and context_length are too small")
    if int(config.event_interval) < 1 or int(config.message_dim) < 1:
        raise ValueError("event_interval and message_dim must be positive")
    if str(config.mode) not in {
        "average_no_message",
        "learned_no_message",
        "learned_shuffled",
        "learned_real",
    }:
        raise ValueError("unknown modular society mode")


class _LanguageCell(nn.Module):
    def __init__(self, config: ModularSocietyConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.cell_width)
        self.core = MarulhoCausalTransformerStateBlock(
            config.cell_width,
            config.cell_width,
            state_layers=config.cell_layers,
            attention_heads=config.attention_heads,
            context_length=config.context_length,
            mlp_ratio=config.mlp_ratio,
            dropout=0.0,
        )
        self.lm_head = nn.Linear(config.cell_width, config.vocab_size, bias=False)
        if bool(config.tie_embeddings):
            self.lm_head.weight = self.embedding.weight


class MarulhoModularSocietyLanguageModel(nn.Module):
    """Several small deep models coordinated through a narrow causal event bus."""

    surface = "marulho_modular_society_language_model.v1"

    def __init__(self, config: ModularSocietyConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.cells = nn.ModuleList(_LanguageCell(config) for _ in range(config.cell_count))
        self.message_out = nn.ModuleList(
            nn.Linear(config.cell_width, config.message_dim, bias=False)
            for _ in range(config.cell_count)
        )
        self.message_in = nn.ModuleList(
            nn.Linear(config.message_dim, config.cell_width, bias=False)
            for _ in range(config.cell_count)
        )
        self.coordinator = nn.Linear(
            config.cell_count * config.cell_width, config.cell_count, bias=False
        )
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def context_length(self) -> int:
        return int(self.config.context_length)

    @property
    def generation_vocab_size(self) -> int:
        return int(self.config.vocab_size)

    def _cell_state(
        self, state: Mapping[str, torch.Tensor] | None, cell_index: int
    ) -> dict[str, torch.Tensor] | None:
        if state is None:
            return None
        prefix = f"cell_{cell_index}_"
        rows = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        return rows or None

    def _initial_bus(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
        bus = [
            torch.zeros(batch_size, self.config.cell_width, device=device, dtype=dtype)
            for _ in range(self.config.cell_count)
        ]
        pending = [torch.zeros_like(row) for row in bus]
        return bus, pending, 0

    def _next_bus(self, summaries: list[torch.Tensor]) -> list[torch.Tensor]:
        messages = [
            projection(summary)
            for projection, summary in zip(self.message_out, summaries, strict=True)
        ]
        usable = (
            [torch.roll(message, shifts=1, dims=0) for message in messages]
            if self.config.mode == "learned_shuffled"
            else messages
        )
        total = torch.stack(usable, dim=0).sum(dim=0)
        incoming = [
            (total - usable[index]) / float(self.config.cell_count - 1)
            for index in range(self.config.cell_count)
        ]
        if self.config.mode in {"average_no_message", "learned_no_message"}:
            # Keep the same projection/reduction graph while removing information.
            # This makes the no-message arms honest compute controls.
            incoming = [message * 0.0 for message in incoming]
        return [
            projection(message)
            for projection, message in zip(self.message_in, incoming, strict=True)
        ]

    def _combine(
        self, hidden_rows: list[torch.Tensor], logits: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        learned_weights = torch.softmax(
            self.coordinator(torch.cat(hidden_rows, dim=-1)), dim=-1
        )
        if self.config.mode == "average_no_message":
            uniform = logits[0].new_full(
                (*logits[0].shape[:2], self.config.cell_count),
                1.0 / float(self.config.cell_count),
            )
            # Preserve coordinator forward/backward work without exposing its signal.
            weights = uniform + (learned_weights * 0.0)
        else:
            weights = learned_weights
        stacked = torch.stack(logits, dim=-2)
        return (stacked * weights.unsqueeze(-1)).sum(dim=-2), weights

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        del decode_vocab_only
        if input_ids.ndim != 2:
            raise ValueError("Modular language model expects [batch, time]")
        ids = input_ids.to(device=self.device, dtype=torch.long)
        batch_size, time_steps = int(ids.shape[0]), int(ids.shape[1])
        if time_steps < 1:
            raise ValueError("Modular language model expects at least one token")
        if state is None:
            bus, pending, pending_count = self._initial_bus(
                batch_size,
                device=ids.device,
                dtype=self.cells[0].embedding.weight.dtype,
            )
        else:
            assert state is not None
            bus = [
                state[f"bus_{index}"].to(self.device)
                for index in range(self.config.cell_count)
            ]
            pending = [
                state[f"pending_{index}"].to(self.device)
                for index in range(self.config.cell_count)
            ]
            pending_count = int(state["pending_count"].detach().cpu().item())
        cell_states = [
            self._cell_state(state, index) for index in range(self.config.cell_count)
        ]
        output_chunks: list[torch.Tensor] = []
        weight_chunks: list[torch.Tensor] = []
        message_events = 0
        start = 0
        while start < time_steps:
            take = min(self.config.event_interval - pending_count, time_steps - start)
            end = start + take
            token_chunk = ids[:, start:end]
            hidden_rows: list[torch.Tensor] = []
            logits_rows: list[torch.Tensor] = []
            next_cell_states: list[dict[str, torch.Tensor]] = []
            for index, cell in enumerate(self.cells):
                embedded = cell.embedding(token_chunk) + bus[index].unsqueeze(1)
                hidden, next_state, _telemetry = cell.core(
                    embedded,
                    cell_states[index],
                    collect_telemetry=False,
                )
                hidden_rows.append(hidden)
                logits_rows.append(cell.lm_head(hidden))
                next_cell_states.append(next_state)
                pending[index] = pending[index] + hidden.sum(dim=1)
            combined, weights = self._combine(hidden_rows, logits_rows)
            output_chunks.append(combined)
            weight_chunks.append(weights)
            pending_count += take
            cell_states = next_cell_states
            if pending_count == self.config.event_interval:
                summaries = [row / float(self.config.event_interval) for row in pending]
                bus = self._next_bus(summaries)
                pending = [torch.zeros_like(row) for row in pending]
                pending_count = 0
                message_events += batch_size
            start = end
        next_state: dict[str, torch.Tensor] = {
            "position": cell_states[0]["position"].detach(),
            "pending_count": torch.tensor(
                pending_count, device=self.device, dtype=torch.long
            ),
        }
        for index, cell_state in enumerate(cell_states):
            for key, value in cell_state.items():
                next_state[f"cell_{index}_{key}"] = value.detach()
            next_state[f"bus_{index}"] = bus[index].detach()
            next_state[f"pending_{index}"] = pending[index].detach()
        weights = torch.cat(weight_chunks, dim=1)
        telemetry = {
            "surface": self.surface,
            "active_language_path": self.config.active_language_path,
            "owned_by_marulho": True,
            "external_llm_used": False,
            "cell_count": self.config.cell_count,
            "cell_width": self.config.cell_width,
            "cell_layers": self.config.cell_layers,
            "message_dim": self.config.message_dim,
            "mode": self.config.mode,
            "message_event_batch_count": message_events,
            "communication_active": self.config.mode in {
                "learned_shuffled", "learned_real"
            },
            "mean_coordinator_entropy": (
                float(
                    (-(weights.float() * weights.float().clamp_min(1e-9).log()).sum(-1))
                    .mean()
                    .detach()
                    .cpu()
                )
                if collect_telemetry
                else None
            ),
        }
        return {
            "logits": torch.cat(output_chunks, dim=1),
            "state": next_state,
            "telemetry": telemetry,
        }

    def forward_step(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        ids = input_ids.unsqueeze(1) if input_ids.ndim == 1 else input_ids
        if ids.ndim != 2 or int(ids.shape[1]) != 1:
            raise ValueError("forward_step expects [batch] or [batch, 1]")
        return self.forward(
            ids,
            state,
            collect_telemetry=collect_telemetry,
            decode_vocab_only=decode_vocab_only,
        )

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
            "surface": "marulho_modular_society_loss.v1",
            "full_vocab_logits_materialized": True,
            "external_llm_used": False,
        }
        return {
            "loss": loss,
            "loss_kind": "full_vocab_cross_entropy",
            "loss_evidence": evidence if return_evidence else {},
            "state": output["state"],
            "telemetry": output["telemetry"],
        }

    def generation_decode_policy(
        self,
        *,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        return {
            "surface": "marulho_modular_society_decode_policy.v1",
            "decode_strategy": "nucleus_sampling" if temperature > 0 else "greedy_argmax",
            "model_vocab_size": int(self.config.vocab_size),
            "generation_vocab_size": int(self.config.vocab_size),
            "full_model_vocab_logits_materialized": True,
            "repetition_penalty": max(1.0, float(repetition_penalty)),
            "no_repeat_ngram_size": max(0, int(no_repeat_ngram_size)),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "sampling_seed": None if seed is None else int(seed),
            "top_p_applied": bool(float(temperature) > 0.0 and float(top_p) < 1.0),
            "kv_cache": "bounded_per_cell_per_layer",
            "external_llm_used": False,
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
        if prompt_ids.ndim == 1:
            generated = prompt_ids.unsqueeze(0)
        elif prompt_ids.ndim == 2:
            generated = prompt_ids
        else:
            raise ValueError("prompt_ids must be [time] or [batch, time]")
        generated = generated.to(device=self.device, dtype=torch.long)
        if int(generated.shape[1]) < 1:
            raise ValueError("prompt must contain at least one token")

        state = None
        result = None
        for start in range(0, int(generated.shape[1]), self.context_length):
            result = self.forward(
                generated[:, start : start + self.context_length],
                state,
                collect_telemetry=False,
            )
            state = result["state"]
        assert result is not None
        next_logits = result["logits"][:, -1]
        generator = None
        if temperature > 0 and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        new_count = 0
        for _ in range(max(0, int(max_new_tokens))):
            controlled, _control = _apply_decode_controls(
                next_logits,
                generated,
                repetition_penalty=max(1.0, float(repetition_penalty)),
                no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
            )
            if temperature > 0:
                probabilities = torch.softmax(controlled / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probabilities, sorted_indices = torch.sort(
                        probabilities, descending=True, dim=-1
                    )
                    cumulative = torch.cumsum(sorted_probabilities, dim=-1)
                    remove = cumulative > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
                    sorted_probabilities /= sorted_probabilities.sum(-1, keepdim=True)
                    rank = torch.multinomial(sorted_probabilities, 1, generator=generator)
                    next_id = sorted_indices.gather(-1, rank)
                else:
                    next_id = torch.multinomial(probabilities, 1, generator=generator)
            else:
                next_id = torch.argmax(controlled, dim=-1, keepdim=True)
            generated = torch.cat((generated, next_id), dim=1)
            new_count += 1
            if eos_id is not None and bool((next_id == int(eos_id)).all()):
                break
            step = self.forward_step(next_id, state, collect_telemetry=False)
            state = step["state"]
            next_logits = step["logits"][:, -1]
        return {
            "generated_ids": generated,
            "new_token_count": new_count,
            "generation_decode": self.generation_decode_policy(
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            ),
            "owned_by_marulho": True,
            "external_llm_used": False,
        }
