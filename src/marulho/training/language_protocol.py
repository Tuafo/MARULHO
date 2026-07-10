"""Shared causal-language model interface for matched architecture experiments."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

import torch


LanguageRuntimeState = Mapping[str, torch.Tensor]


@runtime_checkable
class CausalLanguageModel(Protocol):
    """Deep interface shared by MARULHO language architecture adapters."""

    @property
    def device(self) -> torch.device: ...

    @property
    def context_length(self) -> int: ...

    def forward(
        self,
        input_ids: torch.Tensor,
        state: LanguageRuntimeState | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]: ...

    def next_token_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        collect_telemetry: bool = True,
        return_evidence: bool = True,
    ) -> dict[str, Any]: ...

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
    ) -> dict[str, Any]: ...
