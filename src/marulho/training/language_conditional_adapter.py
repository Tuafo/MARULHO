"""Experimental frozen-base conditional residual plasticity for V49."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import torch
from torch import nn

from marulho.training.language_model import MarulhoLanguageModel
from marulho.training.language_transformer import MarulhoTransformerBlock


ADAPTER_KEY = "conditional_adapter_key"
ADAPTER_VALUE = "conditional_adapter_value"


class MarulhoConditionalAdapterLanguageModel(MarulhoLanguageModel):
    """V39-compatible model with one explicitly gated causal residual sidecar."""

    adapter_surface = "marulho_conditional_residual_adapter.v1"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.conditional_adapter = MarulhoTransformerBlock(
            int(config.state_dim),
            attention_heads=int(config.attention_heads),
            context_length=int(config.transformer_context_length),
            mlp_ratio=1.0,
            dropout=0.0,
        )
        for module in self.conditional_adapter.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        self._conditional_adapter_enabled = False

    @classmethod
    def from_parent(
        cls,
        parent: MarulhoLanguageModel,
    ) -> "MarulhoConditionalAdapterLanguageModel":
        model = cls(parent.config)
        result = model.load_state_dict(parent.state_dict(), strict=False)
        expected_missing = {
            name
            for name in model.state_dict()
            if name.startswith("conditional_adapter.")
        }
        if set(result.missing_keys) != expected_missing or result.unexpected_keys:
            raise ValueError("conditional adapter parent state transfer is incomplete")
        model.freeze_parent()
        return model

    def freeze_parent(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("conditional_adapter."))

    def set_conditional_adapter_enabled(self, enabled: bool) -> None:
        self._conditional_adapter_enabled = bool(enabled)

    @property
    def conditional_adapter_enabled(self) -> bool:
        return bool(self._conditional_adapter_enabled)

    def conditional_adapter_parameter_count(self) -> int:
        return sum(
            parameter.numel() for parameter in self.conditional_adapter.parameters()
        )

    def parent_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("conditional_adapter.")
        )

    def parent_state_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, value in self.state_dict().items():
            if name.startswith("conditional_adapter."):
                continue
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def _active_forward(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None,
        *,
        collect_telemetry: bool,
    ) -> dict[str, Any]:
        base = self._forward_hidden(
            input_ids,
            state,
            collect_telemetry=collect_telemetry,
        )
        position = (
            torch.zeros((), device=self.device, dtype=torch.long)
            if state is None
            else state.get(
                "position",
                torch.zeros((), device=self.device, dtype=torch.long),
            )
        )
        past_key = None if state is None else state.get(ADAPTER_KEY)
        past_value = None if state is None else state.get(ADAPTER_VALUE)
        hidden, next_key, next_value = self.conditional_adapter(
            base["hidden"],
            past_key=past_key,
            past_value=past_value,
            position_offset=position,
        )
        next_state = dict(base["state"])
        next_state[ADAPTER_KEY] = next_key.detach()
        next_state[ADAPTER_VALUE] = next_value.detach()
        telemetry = {
            **base["telemetry"],
            "conditional_adapter_surface": self.adapter_surface,
            "conditional_adapter_enabled": True,
            "conditional_adapter_parameters": self.conditional_adapter_parameter_count(),
            "conditional_adapter_cache_tokens": int(next_key.shape[2]),
        }
        return {
            "logits": self.lm_head(hidden),
            "state": next_state,
            "telemetry": telemetry,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        if not self.conditional_adapter_enabled:
            return super().forward(
                input_ids,
                state,
                collect_telemetry=collect_telemetry,
                decode_vocab_only=decode_vocab_only,
            )
        del decode_vocab_only
        return self._active_forward(
            input_ids,
            state,
            collect_telemetry=collect_telemetry,
        )

    def forward_step(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        if not self.conditional_adapter_enabled:
            return super().forward_step(
                input_ids,
                state,
                collect_telemetry=collect_telemetry,
                decode_vocab_only=decode_vocab_only,
            )
        del decode_vocab_only
        if input_ids.ndim == 1:
            step_ids = input_ids.unsqueeze(1)
        elif input_ids.ndim == 2 and int(input_ids.shape[1]) == 1:
            step_ids = input_ids
        else:
            raise ValueError("forward_step expects [batch] or [batch, 1] token ids")
        return self._active_forward(
            step_ids,
            state,
            collect_telemetry=collect_telemetry,
        )
