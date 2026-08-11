"""Experimental conditional hierarchical low-rank plasticity for V50."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import os
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import torch
from torch import nn

from marulho.data.language_tokenizer import (
    LanguageTokenizer,
    load_language_tokenizer_state,
)
from marulho.training.language_model import LanguageModelConfig
from marulho.training.language_model import MarulhoLanguageModel


CHECKPOINT_SURFACE = "marulho_conditional_hierarchical_lora_checkpoint.v1"


class ConditionalLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int) -> None:
        super().__init__()
        if int(rank) < 1:
            raise ValueError("conditional LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.lora_a = nn.Linear(int(base.in_features), self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, int(base.out_features), bias=False)
        nn.init.normal_(self.lora_a.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_b.weight)
        self.enabled = False

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = self.base(value)
        if not self.enabled:
            return result
        return result + self.lora_b(self.lora_a(value))


def _canonical_parent_name(name: str) -> str:
    return name.replace(".base.weight", ".weight").replace(
        ".base.bias",
        ".bias",
    )


def parent_state_sha256(model: nn.Module) -> str:
    rows = []
    for name, value in model.state_dict().items():
        if ".lora_a." in name or ".lora_b." in name:
            continue
        rows.append((_canonical_parent_name(name), value))
    digest = hashlib.sha256()
    for name, value in sorted(rows, key=lambda row: row[0]):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class MarulhoConditionalLoRALanguageModel(MarulhoLanguageModel):
    """V39-compatible model with conditionally skipped low-rank layer deltas."""

    adapter_surface = "marulho_conditional_hierarchical_lora.v1"

    def __init__(self, config, *, rank: int = 16, install: bool = True) -> None:
        super().__init__(config)
        self.conditional_lora_rank = int(rank)
        self._conditional_lora_installed = False
        if install:
            self.install_conditional_lora()

    @classmethod
    def from_parent(
        cls,
        parent: MarulhoLanguageModel,
        *,
        rank: int = 16,
    ) -> "MarulhoConditionalLoRALanguageModel":
        model = cls(parent.config, rank=int(rank), install=False)
        model.load_state_dict(parent.state_dict(), strict=True)
        model.install_conditional_lora()
        model.freeze_parent()
        if parent_state_sha256(model) != parent_state_sha256(parent):
            raise ValueError("conditional LoRA parent transfer is not exact")
        return model

    def install_conditional_lora(self) -> None:
        if self._conditional_lora_installed:
            raise RuntimeError("conditional LoRA is already installed")
        for layer in self.state_block.layers:
            layer.attention.qkv = ConditionalLoRALinear(
                layer.attention.qkv,
                rank=self.conditional_lora_rank,
            )
            layer.attention.output = ConditionalLoRALinear(
                layer.attention.output,
                rank=self.conditional_lora_rank,
            )
            layer.gate_up = ConditionalLoRALinear(
                layer.gate_up,
                rank=self.conditional_lora_rank,
            )
            layer.down = ConditionalLoRALinear(
                layer.down,
                rank=self.conditional_lora_rank,
            )
        self._conditional_lora_installed = True

    def conditional_lora_modules(self) -> tuple[ConditionalLoRALinear, ...]:
        return tuple(
            module
            for module in self.modules()
            if isinstance(module, ConditionalLoRALinear)
        )

    def conditional_lora_parameters(self) -> Iterable[nn.Parameter]:
        for module in self.conditional_lora_modules():
            yield from module.lora_a.parameters()
            yield from module.lora_b.parameters()

    def conditional_lora_named_parameters(self):
        for name, parameter in self.named_parameters():
            if ".lora_a." in name or ".lora_b." in name:
                yield name, parameter

    def freeze_parent(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.conditional_lora_parameters():
            parameter.requires_grad_(True)

    def set_conditional_lora_enabled(self, enabled: bool) -> None:
        for module in self.conditional_lora_modules():
            module.enabled = bool(enabled)

    @property
    def conditional_lora_enabled(self) -> bool:
        modules = self.conditional_lora_modules()
        return bool(modules and all(module.enabled for module in modules))

    def conditional_lora_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.conditional_lora_parameters())

    def parent_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if ".lora_a." not in name and ".lora_b." not in name
        )


def save_conditional_lora_checkpoint(
    path: str | Path,
    model: MarulhoConditionalLoRALanguageModel,
    tokenizer: LanguageTokenizer,
    metadata: dict,
) -> Path:
    if int(model.config.vocab_size) != int(tokenizer.vocab_size):
        raise ValueError("conditional LoRA checkpoint vocabulary differs")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "surface": CHECKPOINT_SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "config": asdict(model.config),
        "rank": int(model.conditional_lora_rank),
        "model_state": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "tokenizer": tokenizer.state_dict(),
        "tokenizer_hash": tokenizer.vocabulary_hash(),
        "conditional_lora_enabled_at_rest": False,
        "metadata": dict(metadata),
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


def load_conditional_lora_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[MarulhoConditionalLoRALanguageModel, LanguageTokenizer, dict]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("surface") != CHECKPOINT_SURFACE:
        raise ValueError("conditional LoRA checkpoint surface differs")
    if payload.get("conditional_lora_enabled_at_rest") is not False:
        raise ValueError("conditional LoRA checkpoint must be inactive at rest")
    tokenizer = load_language_tokenizer_state(payload["tokenizer"])
    if tokenizer.vocabulary_hash() != str(payload["tokenizer_hash"]):
        raise ValueError("conditional LoRA checkpoint tokenizer hash differs")
    model = MarulhoConditionalLoRALanguageModel(
        LanguageModelConfig(**dict(payload["config"])),
        rank=int(payload["rank"]),
    )
    if int(model.config.vocab_size) != int(tokenizer.vocab_size):
        raise ValueError("conditional LoRA checkpoint vocabulary differs")
    model.load_state_dict(dict(payload["model_state"]), strict=True)
    model.freeze_parent()
    model.set_conditional_lora_enabled(False)
    return model, tokenizer, dict(payload.get("metadata") or {})
