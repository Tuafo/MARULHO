from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import torch
from torch import nn
from torch.nn import functional as F

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer


@dataclass(frozen=True)
class LanguageModelConfig:
    vocab_size: int
    embedding_dim: int = 64
    state_dim: int = 128
    spike_slope: float = 5.0
    adaptive_timestep_budget: int = 1
    active_language_path: str = "marulho_lm_head"


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


def _surrogate_spike(voltage_delta: torch.Tensor, slope: float) -> torch.Tensor:
    soft = torch.sigmoid(float(slope) * voltage_delta)
    hard = (voltage_delta >= 0).to(voltage_delta.dtype)
    return hard.detach() - soft.detach() + soft


def _tensor_hash(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    payload = {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "values": cpu.reshape(-1).tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _split_hash(batches: Sequence[LanguageBatch]) -> str:
    payload = [
        {
            "input": _tensor_hash(batch.input_ids),
            "target": _tensor_hash(batch.target_ids),
        }
        for batch in batches
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RMSNorm(nn.Module):
    """RMSNorm without sequence-shape assumptions."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(dim)))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        rms = value.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return value * rms * self.weight.to(device=value.device, dtype=value.dtype)


class MarulhoSelectiveSpikingStateBlock(nn.Module):
    """Causal selective recurrent spike block for the MARULHO LM foundation."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        spike_slope: float = 5.0,
        adaptive_timestep_budget: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.spike_slope = float(spike_slope)
        self.adaptive_timestep_budget = max(1, int(adaptive_timestep_budget))
        self.input_norm = RMSNorm(self.input_dim)
        self.input_proj = nn.Linear(self.input_dim, self.state_dim)
        self.current_proj = nn.Linear(self.input_dim, self.state_dim)
        self.beta_input_proj = nn.Linear(self.input_dim, self.state_dim)
        self.beta_state_proj = nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.threshold_input_proj = nn.Linear(self.input_dim, self.state_dim)
        self.select_proj = nn.Linear(self.input_dim, self.state_dim * 3)
        self.recurrent_proj = nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.residual_proj = nn.Linear(self.input_dim, self.state_dim)
        self.state_output_proj = nn.Linear(self.state_dim, self.state_dim)
        self.output_norm = RMSNorm(self.state_dim)
        self.raw_leak = nn.Parameter(torch.full((self.state_dim,), 1.75))
        self.current_gain = nn.Parameter(torch.ones(self.state_dim))
        self.threshold = nn.Parameter(torch.ones(self.state_dim))

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        zeros = torch.zeros(batch_size, self.state_dim, device=device, dtype=dtype)
        return {
            "membrane": zeros,
            "spikes": zeros.clone(),
            "selective_state": zeros.clone(),
            "eligibility_trace": zeros.clone(),
        }

    def forward(
        self,
        inputs: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        if inputs.ndim != 3:
            raise ValueError("Language state block expects [batch, time, input_dim]")
        batch_size, time_steps, _ = inputs.shape
        if state is None:
            current_state = self.initial_state(
                batch_size,
                device=inputs.device,
                dtype=inputs.dtype,
            )
        else:
            current_state = {
                "membrane": state["membrane"].to(device=inputs.device, dtype=inputs.dtype),
                "spikes": state["spikes"].to(device=inputs.device, dtype=inputs.dtype),
                "selective_state": state["selective_state"].to(
                    device=inputs.device,
                    dtype=inputs.dtype,
                ),
                "eligibility_trace": state.get(
                    "eligibility_trace",
                    torch.zeros_like(state["spikes"]),
                ).to(device=inputs.device, dtype=inputs.dtype),
            }

        membrane = current_state["membrane"]
        spikes = current_state["spikes"]
        selective_state = current_state["selective_state"]
        eligibility_trace = current_state["eligibility_trace"]
        raw_leak = self.raw_leak.to(device=inputs.device, dtype=inputs.dtype)
        base_threshold = self.threshold.to(device=inputs.device, dtype=inputs.dtype)
        current_gain = self.current_gain.to(device=inputs.device, dtype=inputs.dtype)
        outputs: list[torch.Tensor] = []
        spike_sum = inputs.new_tensor(0.0)
        active_neuron_counts = inputs.new_zeros(self.state_dim)

        for step in range(time_steps):
            token_input = self.input_norm(inputs[:, step, :])
            select_logits = self.select_proj(token_input)
            state_decay_logits, state_input_logits, state_output_logits = select_logits.chunk(
                3,
                dim=-1,
            )
            state_decay = torch.sigmoid(state_decay_logits)
            state_input = torch.sigmoid(state_input_logits)
            state_output = torch.sigmoid(state_output_logits)
            leak = torch.sigmoid(
                raw_leak
                + self.beta_input_proj(token_input)
                + self.beta_state_proj(selective_state)
            )
            threshold = F.softplus(
                base_threshold + self.threshold_input_proj(token_input)
            )
            current = current_gain * self.current_proj(token_input)
            drive = self.input_proj(token_input) + current + self.recurrent_proj(spikes)
            for _substep in range(self.adaptive_timestep_budget):
                membrane = leak * membrane + (1.0 - leak) * drive - spikes * threshold
                spikes = _surrogate_spike(membrane - threshold, self.spike_slope)
                selective_state = state_decay * selective_state + state_input * spikes
                eligibility_trace = (
                    0.95 * eligibility_trace + spikes
                )
            residual = self.residual_proj(token_input)
            mixed_state = state_output * selective_state + spikes
            outputs.append(self.output_norm(residual + self.state_output_proj(mixed_state)))
            spike_sum = spike_sum + spikes.sum()
            active_neuron_counts = active_neuron_counts + (spikes > 0).sum(dim=0)

        hidden = torch.stack(outputs, dim=1)
        denominator = max(1, batch_size * time_steps * self.state_dim)
        per_neuron_denominator = max(1, batch_size * time_steps)
        firing_fraction = active_neuron_counts / float(per_neuron_denominator)
        dead_fraction = (firing_fraction <= 0).to(inputs.dtype).mean()
        over_firing_fraction = (firing_fraction >= 0.8).to(inputs.dtype).mean()
        telemetry = {
            "surface": "marulho_selective_spiking_state_block.v1",
            "spike_rate": float((spike_sum / float(denominator)).detach().cpu().item()),
            "dead_neuron_fraction": float(dead_fraction.detach().cpu().item()),
            "over_firing_fraction": float(over_firing_fraction.detach().cpu().item()),
            "adaptive_timestep_budget": int(self.adaptive_timestep_budget),
            "adaptive_step_count": int(time_steps * self.adaptive_timestep_budget),
            "state_dim": self.state_dim,
            "time_steps": int(time_steps),
            "normalization": "rmsnorm",
            "plif_state": "membrane_spikes_selective_state",
            "state_cache_keys": [
                "membrane",
                "spikes",
                "selective_state",
                "eligibility_trace",
            ],
            "input_dependent_leak": True,
            "input_dependent_threshold": True,
            "trainable_current_terms": True,
            "surrogate_gradient": "straight_through_sigmoid",
            "device": str(inputs.device),
        }
        return (
            hidden,
            {
                "membrane": membrane,
                "spikes": spikes,
                "selective_state": selective_state,
                "eligibility_trace": eligibility_trace,
            },
            telemetry,
        )


class MarulhoLanguageModel(nn.Module):
    """MARULHO-owned next-token language model foundation."""

    def __init__(self, config: LanguageModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.state_block = MarulhoSelectiveSpikingStateBlock(
            config.embedding_dim,
            config.state_dim,
            spike_slope=config.spike_slope,
            adaptive_timestep_budget=config.adaptive_timestep_budget,
        )
        self.lm_head = nn.Linear(config.state_dim, config.vocab_size)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("Language model expects input_ids shaped [batch, time]")
        embeddings = self.token_embedding(input_ids.to(self.device))
        hidden, next_state, telemetry = self.state_block(embeddings, state)
        logits = self.lm_head(hidden)
        telemetry = {
            **telemetry,
            "active_language_path": self.config.active_language_path,
            "external_llm_used": False,
            "owned_by_marulho": True,
            "vocab_size": self.config.vocab_size,
        }
        return {
            "logits": logits,
            "state": next_state,
            "telemetry": telemetry,
        }

    def next_token_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> dict[str, Any]:
        result = self.forward(input_ids)
        logits = result["logits"]
        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            target_ids.to(logits.device).reshape(-1),
        )
        return {
            **result,
            "loss": loss,
            "loss_kind": "causal_next_token_cross_entropy",
        }

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_id: int | None = None,
    ) -> dict[str, Any]:
        self.eval()
        if prompt_ids.ndim == 1:
            prompt = prompt_ids.unsqueeze(0)
        elif prompt_ids.ndim == 2:
            prompt = prompt_ids
        else:
            raise ValueError("prompt_ids must be [time] or [batch, time]")
        generated = prompt.to(self.device)
        state: Mapping[str, torch.Tensor] | None = None
        result = self.forward(generated, state)
        state = result["state"]
        next_logits = result["logits"][:, -1, :]
        new_token_count = 0
        for _ in range(max(0, int(max_new_tokens))):
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=1)
            new_token_count += 1
            if eos_id is not None and bool(torch.all(next_id == int(eos_id)).item()):
                break
            result = self.forward(next_id, state)
            state = result["state"]
            next_logits = result["logits"][:, -1, :]
        return {
            "surface": "marulho_language_generation.v1",
            "generated_ids": generated.detach().cpu(),
            "new_token_count": new_token_count,
            "active_language_path": self.config.active_language_path,
            "external_llm_used": False,
            "owned_by_marulho": True,
            "loads_external_checkpoint": False,
        }


def build_language_model_splits(
    texts: Sequence[str],
    tokenizer: ByteLevelLanguageTokenizer,
    *,
    sequence_length: int,
    eval_fraction: float = 0.2,
    stride: int | None = None,
    device: torch.device | str | None = None,
) -> LanguageSplit:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    token_ids: list[int] = []
    for text in texts:
        encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
        token_ids.extend(encoded)
    window_length = int(sequence_length) + 1
    if len(token_ids) < window_length:
        raise ValueError("Not enough tokens to build a next-token language split")
    step = int(stride or sequence_length)
    if step <= 0:
        raise ValueError("stride must be positive")
    windows = [
        token_ids[offset : offset + window_length]
        for offset in range(0, len(token_ids) - window_length + 1, step)
    ]
    if not windows:
        raise ValueError("No language windows were produced")
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    batches = tuple(
        LanguageBatch(
            input_ids=torch.tensor([window[:-1]], dtype=torch.long, device=target_device),
            target_ids=torch.tensor([window[1:]], dtype=torch.long, device=target_device),
        )
        for window in windows
    )
    if len(batches) == 1:
        train_batches = batches
        eval_batches = batches
    else:
        eval_count = max(1, min(len(batches) - 1, math.ceil(len(batches) * float(eval_fraction))))
        split_index = len(batches) - eval_count
        train_batches = batches[:split_index]
        eval_batches = batches[split_index:]
    report = {
        "surface": "marulho_language_train_eval_split.v1",
        "owned_by_marulho": True,
        "external_dependency": False,
        "sequence_length": int(sequence_length),
        "stride": int(step),
        "source_text_count": len(texts),
        "window_count": len(batches),
        "train_batch_count": len(train_batches),
        "eval_batch_count": len(eval_batches),
        "tokenizer_hash": tokenizer.vocabulary_hash(),
        "train_split_hash": _split_hash(train_batches),
        "eval_split_hash": _split_hash(eval_batches),
    }
    return LanguageSplit(train=train_batches, eval=eval_batches, report=report)


@torch.no_grad()
def evaluate_language_model(
    model: MarulhoLanguageModel,
    batches: Sequence[LanguageBatch],
) -> dict[str, Any]:
    if not batches:
        raise ValueError("At least one evaluation batch is required")
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    last_telemetry: dict[str, Any] = {}
    for batch in batches:
        result = model.next_token_loss(batch.input_ids.to(model.device), batch.target_ids.to(model.device))
        token_count = int(batch.target_ids.numel())
        total_loss += float(result["loss"].detach().cpu().item()) * token_count
        total_tokens += token_count
        last_telemetry = dict(result["telemetry"])
    heldout_loss = total_loss / max(1, total_tokens)
    return {
        "artifact_kind": "marulho_language_model_heldout_evaluation",
        "surface": "marulho_language_model_heldout_evaluation.v1",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "active_language_path": model.config.active_language_path,
        "heldout_loss": heldout_loss,
        "heldout_perplexity": float(math.exp(min(heldout_loss, 30.0))),
        "eval_batch_count": len(batches),
        "eval_token_count": total_tokens,
        "device": str(model.device),
        "spike_telemetry": last_telemetry,
    }


def language_model_checkpoint_payload(
    model: MarulhoLanguageModel,
    tokenizer: ByteLevelLanguageTokenizer,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "artifact_kind": "marulho_language_model_checkpoint",
        "surface": "marulho_language_model_checkpoint.v1",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "active_language_path": model.config.active_language_path,
        "config": asdict(model.config),
        "model_state": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "tokenizer": tokenizer.state_dict(),
        "tokenizer_hash": tokenizer.vocabulary_hash(),
        "metadata": dict(metadata or {}),
    }


def save_language_model_checkpoint(
    path: str | Path,
    model: MarulhoLanguageModel,
    tokenizer: ByteLevelLanguageTokenizer,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = language_model_checkpoint_payload(model, tokenizer, metadata)
    temporary_path = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def load_language_model_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> tuple[MarulhoLanguageModel, ByteLevelLanguageTokenizer, dict[str, Any]]:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=map_location or "cpu")
    tokenizer = ByteLevelLanguageTokenizer.load_state_dict(payload["tokenizer"])
    config = LanguageModelConfig(**dict(payload["config"]))
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "Language model checkpoint vocab size does not match tokenizer state"
        )
    model = MarulhoLanguageModel(config)
    model.load_state_dict(payload["model_state"])
    return model, tokenizer, dict(payload.get("metadata") or {})
