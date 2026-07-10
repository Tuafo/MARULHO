"""MARULHO-owned editable delta-memory causal language model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from marulho.training.language_model import _apply_decode_controls
from marulho.training.language_protocol import LanguageRuntimeState
from marulho.training.language_transformer import (
    MarulhoCausalSelfAttention,
    TransformerRMSNorm,
)


@dataclass(frozen=True)
class DeltaLanguageConfig:
    vocab_size: int
    width: int = 512
    layers: int = 4
    memory_heads: int = 8
    memory_head_dim: int = 32
    attention_heads: int = 8
    local_attention_every: int = 0
    context_length: int = 256
    mlp_dim: int = 2048
    dropout: float = 0.0
    tie_embeddings: bool = True
    active_language_path: str = "marulho_delta_memory_v0"


def _validate_config(config: DeltaLanguageConfig) -> None:
    if int(config.vocab_size) <= 1:
        raise ValueError("vocab_size must be greater than one")
    if int(config.width) <= 0 or int(config.layers) < 1:
        raise ValueError("width and layers must be positive")
    if int(config.memory_heads) < 1 or int(config.memory_head_dim) < 1:
        raise ValueError("memory_heads and memory_head_dim must be positive")
    if int(config.attention_heads) < 1 or int(config.width) % int(config.attention_heads):
        raise ValueError("width must be divisible by attention_heads")
    if (int(config.width) // int(config.attention_heads)) % 2:
        raise ValueError("attention head dimension must be even")
    if int(config.local_attention_every) < 0:
        raise ValueError("local_attention_every must be non-negative")
    if int(config.context_length) < 2 or int(config.mlp_dim) < int(config.width):
        raise ValueError("context_length or mlp_dim is too small")
    if not math.isfinite(float(config.dropout)) or not 0.0 <= float(config.dropout) < 1.0:
        raise ValueError("dropout must be finite and in [0, 1)")


class EditableDeltaMemory(nn.Module):
    """Fixed-state recurrent memory with independent decay, erase, and write."""

    def __init__(self, width: int, *, heads: int, head_dim: int) -> None:
        super().__init__()
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.memory_width = self.heads * self.head_dim
        self.qkv = nn.Linear(self.width, self.memory_width * 3, bias=False)
        self.gates = nn.Linear(self.width, self.memory_width * 4, bias=False)
        self.read_norm = TransformerRMSNorm(self.memory_width)
        self.output = nn.Linear(self.memory_width, self.width, bias=False)
        decay = torch.linspace(-3.0, -6.0, self.head_dim)
        self.decay_bias = nn.Parameter(decay.repeat(self.heads, 1))

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(
            int(batch_size),
            self.heads,
            self.head_dim,
            self.head_dim,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        value: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _ = value.shape
        q, k, v = self.qkv(value).chunk(3, dim=-1)
        decay_logits, erase_logits, write_logits, output_gate = self.gates(value).chunk(
            4, dim=-1
        )

        def _heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, steps, self.heads, self.head_dim)

        q = F.normalize(_heads(q).float(), dim=-1).to(dtype=value.dtype)
        k = F.normalize(_heads(k).float(), dim=-1).to(dtype=value.dtype)
        v = _heads(v)
        decay = torch.exp(
            -F.softplus(_heads(decay_logits).float() + self.decay_bias.float())
        ).to(dtype=value.dtype)
        erase = torch.sigmoid(_heads(erase_logits))
        write = torch.sigmoid(_heads(write_logits))
        output_gate = _heads(output_gate)
        current = (
            self.initial_state(batch, device=value.device, dtype=value.dtype)
            if state is None
            else state.to(device=value.device, dtype=value.dtype)
        )
        outputs: list[torch.Tensor] = []
        for index in range(int(steps)):
            key = k[:, index]
            decayed = decay[:, index].unsqueeze(-1) * current
            erase_read = torch.einsum(
                "bhk,bhkv->bhv", erase[:, index] * key, decayed
            )
            current = decayed - key.unsqueeze(-1) * erase_read.unsqueeze(-2)
            current = current + key.unsqueeze(-1) * (
                write[:, index] * v[:, index]
            ).unsqueeze(-2)
            read = torch.einsum("bhk,bhkv->bhv", q[:, index], current)
            read = read.reshape(batch, self.memory_width)
            gate = F.silu(output_gate[:, index].reshape(batch, self.memory_width))
            outputs.append(self.output(self.read_norm(read) * gate))
        return torch.stack(outputs, dim=1), current


class DeltaLanguageBlock(nn.Module):
    def __init__(
        self,
        config: DeltaLanguageConfig,
        *,
        layer_index: int,
    ) -> None:
        super().__init__()
        every = int(config.local_attention_every)
        self.kind = (
            "local_attention"
            if every > 0 and (int(layer_index) + 1) % every == 0
            else "editable_delta"
        )
        self.mix_norm = TransformerRMSNorm(config.width)
        if self.kind == "local_attention":
            self.mixer: nn.Module = MarulhoCausalSelfAttention(
                config.width,
                attention_heads=config.attention_heads,
                context_length=config.context_length,
                dropout=config.dropout,
            )
        else:
            self.mixer = EditableDeltaMemory(
                config.width,
                heads=config.memory_heads,
                head_dim=config.memory_head_dim,
            )
        self.mlp_norm = TransformerRMSNorm(config.width)
        self.gate_up = nn.Linear(config.width, config.mlp_dim * 2, bias=False)
        self.down = nn.Linear(config.mlp_dim, config.width, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        value: torch.Tensor,
        state: Mapping[str, torch.Tensor],
        *,
        layer_index: int,
        position_offset: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        normalized = self.mix_norm(value)
        next_state: dict[str, torch.Tensor] = {}
        if self.kind == "local_attention":
            assert isinstance(self.mixer, MarulhoCausalSelfAttention)
            mixed, key, memory_value = self.mixer(
                normalized,
                past_key=state.get(f"layer_{layer_index}_key"),
                past_value=state.get(f"layer_{layer_index}_value"),
                position_offset=position_offset,
            )
            next_state[f"layer_{layer_index}_key"] = key
            next_state[f"layer_{layer_index}_value"] = memory_value
        else:
            assert isinstance(self.mixer, EditableDeltaMemory)
            mixed, memory = self.mixer(
                normalized,
                state.get(f"layer_{layer_index}_memory"),
            )
            next_state[f"layer_{layer_index}_memory"] = memory
        value = value + self.dropout(mixed)
        gate, up = self.gate_up(self.mlp_norm(value)).chunk(2, dim=-1)
        value = value + self.dropout(self.down(F.silu(gate) * up))
        return value, next_state


class MarulhoDeltaLanguageModel(nn.Module):
    """Causal LM using editable recurrent state and optional local attention."""

    surface = "marulho_delta_language_model.v1"

    def __init__(self, config: DeltaLanguageConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.layers = nn.ModuleList(
            DeltaLanguageBlock(config, layer_index=index)
            for index in range(config.layers)
        )
        self.output_norm = TransformerRMSNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
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

    def init_state(
        self,
        batch_size: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        runtime_dtype = dtype or self.token_embedding.weight.dtype
        state: dict[str, torch.Tensor] = {
            "position": torch.zeros((), device=self.device, dtype=torch.long)
        }
        for index, layer in enumerate(self.layers):
            if layer.kind == "local_attention":
                head_dim = self.config.width // self.config.attention_heads
                state[f"layer_{index}_key"] = torch.empty(
                    batch_size,
                    self.config.attention_heads,
                    0,
                    head_dim,
                    device=self.device,
                    dtype=runtime_dtype,
                )
                state[f"layer_{index}_value"] = torch.empty_like(
                    state[f"layer_{index}_key"]
                )
            else:
                mixer = layer.mixer
                assert isinstance(mixer, EditableDeltaMemory)
                state[f"layer_{index}_memory"] = mixer.initial_state(
                    batch_size, device=self.device, dtype=runtime_dtype
                )
        return state

    def serialize_state(
        self, state: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in state.items()}

    def load_state(
        self, serialized: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if "position" not in serialized:
            raise ValueError("Delta runtime state is missing position")
        batch = next(
            int(value.shape[0])
            for key, value in serialized.items()
            if key != "position"
        )
        expected = self.init_state(batch)
        if set(serialized) != set(expected):
            raise ValueError("Delta runtime state keys do not match the model")
        loaded: dict[str, torch.Tensor] = {}
        for key, target in expected.items():
            value = serialized[key]
            if key.endswith(("_key", "_value")):
                valid_shape = (
                    value.ndim == 4
                    and int(value.shape[0]) == batch
                    and int(value.shape[1]) == int(target.shape[1])
                    and int(value.shape[2]) <= int(self.config.context_length)
                    and int(value.shape[3]) == int(target.shape[3])
                )
            else:
                valid_shape = tuple(value.shape) == tuple(target.shape)
            if not valid_shape:
                raise ValueError(f"Delta runtime state shape mismatch for {key}")
            loaded[key] = value.to(device=self.device, dtype=target.dtype)
        return loaded

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
            "surface": "marulho_delta_decode_policy.v1",
            "decode_strategy": "nucleus_sampling" if temperature > 0.0 else "greedy_argmax",
            "model_vocab_size": int(self.config.vocab_size),
            "generation_vocab_size": int(self.config.vocab_size),
            "full_model_vocab_logits_materialized": True,
            "repetition_penalty": max(1.0, float(repetition_penalty)),
            "no_repeat_ngram_size": max(0, int(no_repeat_ngram_size)),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "sampling_seed": None if seed is None else int(seed),
            "state_cache": "fixed_fast_weight_plus_bounded_local_kv",
            "external_llm_used": False,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        state: LanguageRuntimeState | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        del decode_vocab_only
        if input_ids.ndim != 2:
            raise ValueError("Delta language model expects [batch, time] token ids")
        runtime_ids = input_ids.to(device=self.device, dtype=torch.long)
        hidden = self.token_embedding(runtime_ids)
        current = self.init_state(int(runtime_ids.shape[0]), dtype=hidden.dtype) if state is None else dict(state)
        position = current["position"].to(device=self.device, dtype=torch.long)
        next_state: dict[str, torch.Tensor] = {"position": position + int(runtime_ids.shape[1])}
        layer_kinds: list[str] = []
        for index, layer in enumerate(self.layers):
            hidden, layer_state = layer(
                hidden,
                current,
                layer_index=index,
                position_offset=position,
            )
            next_state.update(layer_state)
            layer_kinds.append(layer.kind)
        hidden = self.output_norm(hidden)
        state_bytes = sum(
            value.numel() * value.element_size()
            for key, value in next_state.items()
            if key != "position"
        )
        telemetry = {
            "surface": self.surface,
            "state_core": "editable_delta_memory",
            "telemetry_collected": bool(collect_telemetry),
            "active_language_path": self.config.active_language_path,
            "layer_kinds": layer_kinds,
            "delta_layer_count": layer_kinds.count("editable_delta"),
            "local_attention_layer_count": layer_kinds.count("local_attention"),
            "memory_heads": int(self.config.memory_heads),
            "memory_head_dim": int(self.config.memory_head_dim),
            "runtime_state_bytes_per_batch": int(state_bytes),
            "external_llm_used": False,
            "owned_by_marulho": True,
            "vocab_size": int(self.config.vocab_size),
        }
        return {
            "logits": self.lm_head(hidden),
            "state": next_state,
            "telemetry": telemetry,
        }

    def forward_step(
        self,
        input_ids: torch.Tensor,
        state: LanguageRuntimeState | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)
        if input_ids.ndim != 2 or int(input_ids.shape[1]) != 1:
            raise ValueError("forward_step expects [batch] or [batch, 1]")
        return self.forward(
            input_ids,
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
        if output["logits"].shape[:2] != targets.shape:
            raise ValueError("target_ids must match input batch/time dimensions")
        loss = F.cross_entropy(
            output["logits"].reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
        )
        evidence = {
            "surface": "marulho_delta_cross_entropy.v1",
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
        if not math.isfinite(float(temperature)) or float(temperature) < 0.0:
            raise ValueError("temperature must be finite and non-negative")
        if not math.isfinite(float(top_p)) or not 0.0 < float(top_p) <= 1.0:
            raise ValueError("top_p must be finite and in (0, 1]")
        generated = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        if generated.ndim != 2:
            raise ValueError("prompt_ids must be [time] or [batch, time]")
        generated = generated.to(device=self.device, dtype=torch.long)
        sample = float(temperature) > 0.0
        generator = None
        if sample and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        was_training = bool(self.training)
        self.eval()
        try:
            output = self.forward(generated, collect_telemetry=False)
            state = output["state"]
            logits = output["logits"][:, -1]
            finished = torch.zeros(generated.shape[0], device=self.device, dtype=torch.bool)
            totals = {
                "repetition_penalty_adjusted_token_count": 0,
                "no_repeat_ngram_banned_token_count": 0,
                "decode_control_fallback_count": 0,
            }
            new_tokens = 0
            for _ in range(max(0, int(max_new_tokens))):
                controlled, control = _apply_decode_controls(
                    logits,
                    generated,
                    repetition_penalty=max(1.0, float(repetition_penalty)),
                    no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
                )
                for key in totals:
                    totals[key] += int(control[key])
                if sample:
                    probabilities = torch.softmax(controlled / float(temperature), dim=-1)
                    if float(top_p) < 1.0:
                        sorted_probabilities, sorted_indices = torch.sort(probabilities, dim=-1, descending=True)
                        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
                        remove = cumulative > float(top_p)
                        remove[..., 1:] = remove[..., :-1].clone()
                        remove[..., 0] = False
                        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
                        sorted_probabilities = sorted_probabilities / sorted_probabilities.sum(dim=-1, keepdim=True).clamp_min(
                            torch.finfo(sorted_probabilities.dtype).tiny
                        )
                        rank = torch.multinomial(sorted_probabilities, 1, generator=generator)
                        next_id = sorted_indices.gather(-1, rank)
                    else:
                        next_id = torch.multinomial(probabilities, 1, generator=generator)
                else:
                    next_id = torch.argmax(controlled, dim=-1, keepdim=True)
                if eos_id is not None:
                    next_id = torch.where(
                        finished.unsqueeze(1), torch.full_like(next_id, int(eos_id)), next_id
                    )
                    finished = finished | next_id[:, 0].eq(int(eos_id))
                generated = torch.cat((generated, next_id), dim=1)
                new_tokens += 1
                if eos_id is not None and bool(finished.all().item()):
                    break
                output = self.forward_step(next_id, state, collect_telemetry=False)
                state = output["state"]
                logits = output["logits"][:, -1]
            return {
                "surface": "marulho_delta_generation.v1",
                "generated_ids": generated,
                "new_token_count": int(new_tokens),
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
                "decode_control_totals": totals,
            }
        finally:
            self.train(was_training)


def delta_parameter_inventory(config: DeltaLanguageConfig) -> dict[str, int]:
    model = MarulhoDeltaLanguageModel(config)
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "embedding_parameters": int(model.token_embedding.weight.numel()),
        "non_embedding_parameters": sum(parameter.numel() for parameter in model.parameters())
        - int(model.token_embedding.weight.numel()),
        "config": asdict(config),
    }
