"""Bounded cross-attention reader for exact episodic evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from marulho.training.language_hashed_micro_experts import (
    MarulhoHashedMicroExpertBlock,
    MarulhoHashedMicroExpertLanguageModel,
)
from marulho.training.language_model import _apply_decode_controls


EVIDENCE_INTERFACES = ("gate_zero", "raw_context", "separate_reader")


@dataclass(frozen=True)
class EvidenceReaderConfig:
    width: int
    attention_heads: int
    dropout: float = 0.0
    gate_logit_initial: float = -2.0
    injection_layer_indices: tuple[int, ...] = (0,)


def _validate_reader_config(
    cortex: MarulhoHashedMicroExpertLanguageModel,
    config: EvidenceReaderConfig,
) -> None:
    if int(config.width) != int(cortex.hashed_config.width):
        raise ValueError("evidence reader width must match the V11 cortex")
    if int(config.attention_heads) < 1:
        raise ValueError("evidence reader attention_heads must be positive")
    if int(config.width) % int(config.attention_heads) != 0:
        raise ValueError("evidence reader width must divide attention_heads")
    if not 0.0 <= float(config.dropout) < 1.0:
        raise ValueError("evidence reader dropout must be in [0, 1)")
    if not math.isfinite(float(config.gate_logit_initial)):
        raise ValueError("evidence reader gate initialization must be finite")
    if not config.injection_layer_indices:
        raise ValueError("evidence reader requires at least one injection layer")
    normalized = tuple(int(value) for value in config.injection_layer_indices)
    if len(set(normalized)) != len(normalized):
        raise ValueError("evidence reader injection layers must be unique")
    if tuple(sorted(normalized)) != normalized:
        raise ValueError("evidence reader injection layers must be sorted")
    if min(normalized) < 0 or max(normalized) >= int(cortex.hashed_config.layers):
        raise ValueError("evidence reader injection layer is outside the cortex")


class MarulhoEvidenceReaderLanguageModel(nn.Module):
    """V11 dual streams with bounded evidence reads between cortex blocks."""

    surface = "marulho_interleaved_evidence_reader_language_model.v2"
    generation_surface = "marulho_interleaved_evidence_reader_generation.v2"

    def __init__(
        self,
        cortex: MarulhoHashedMicroExpertLanguageModel,
        config: EvidenceReaderConfig,
    ) -> None:
        super().__init__()
        _validate_reader_config(cortex, config)
        self.cortex = cortex
        self.reader_config = config
        self.query_norm = nn.RMSNorm(int(config.width))
        self.evidence_norm = nn.RMSNorm(int(config.width))
        self.cross_attention = nn.MultiheadAttention(
            int(config.width),
            int(config.attention_heads),
            dropout=float(config.dropout),
            bias=False,
            batch_first=True,
        )
        self.reader_gate_logits = nn.Parameter(
            torch.full(
                (len(config.injection_layer_indices),),
                float(config.gate_logit_initial),
                dtype=torch.float32,
            )
        )

    @property
    def device(self) -> torch.device:
        return self.cortex.device

    @property
    def context_length(self) -> int:
        return int(self.cortex.hashed_config.context_length)

    def _validate_inputs(
        self,
        query_ids: torch.Tensor,
        evidence_ids: torch.Tensor | None,
        interface: str,
    ) -> None:
        if str(interface) not in EVIDENCE_INTERFACES:
            raise ValueError(f"unknown evidence interface: {interface}")
        if query_ids.ndim != 2:
            raise ValueError("evidence reader query_ids must be [batch,time]")
        if int(query_ids.shape[1]) > self.context_length:
            raise ValueError("evidence reader query exceeds local cortex context")
        if interface != "gate_zero":
            if evidence_ids is None or evidence_ids.ndim != 2:
                raise ValueError(f"{interface} requires [batch,time] evidence_ids")
            if int(evidence_ids.shape[0]) != int(query_ids.shape[0]):
                raise ValueError("evidence and query batch sizes must match")
        if interface == "raw_context" and evidence_ids is not None:
            if int(evidence_ids.shape[1] + query_ids.shape[1]) > self.context_length:
                raise ValueError("raw evidence plus query exceeds cortex context")

    def _interleaved_hidden(
        self,
        query_ids: torch.Tensor,
        evidence_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block = self.cortex.state_block
        runtime_query_ids = query_ids.to(device=self.device, dtype=torch.long)
        runtime_evidence_ids = evidence_ids.to(device=self.device, dtype=torch.long)
        query = block.input_projection(self.cortex.token_embedding(runtime_query_ids))
        evidence = block.input_projection(
            self.cortex.token_embedding(runtime_evidence_ids)
        )
        gates = torch.sigmoid(self.reader_gate_logits)
        injection_to_gate = {
            int(layer_index): gate_index
            for gate_index, layer_index in enumerate(
                self.reader_config.injection_layer_indices
            )
        }
        position_offset = torch.zeros((), device=self.device, dtype=torch.long)
        for layer_index, layer in enumerate(block.layers):
            kwargs = {
                "past_key": None,
                "past_value": None,
                "position_offset": position_offset,
            }
            if isinstance(layer, MarulhoHashedMicroExpertBlock):
                query, _query_key, _query_value = layer(
                    query,
                    route_ids=runtime_query_ids,
                    forced_expert_ids=None,
                    **kwargs,
                )
                evidence, _evidence_key, _evidence_value = layer(
                    evidence,
                    route_ids=runtime_evidence_ids,
                    forced_expert_ids=None,
                    **kwargs,
                )
            else:
                query, _query_key, _query_value = layer(query, **kwargs)
                evidence, _evidence_key, _evidence_value = layer(
                    evidence, **kwargs
                )
            gate_index = injection_to_gate.get(layer_index)
            if gate_index is not None:
                normalized_evidence = self.evidence_norm(evidence)
                cross, _weights = self.cross_attention(
                    self.query_norm(query),
                    normalized_evidence,
                    normalized_evidence,
                    need_weights=False,
                )
                gate = gates[gate_index].to(dtype=cross.dtype)
                query = query + gate * cross
        return block.output_norm(query), gates

    def forward_hidden(
        self,
        query_ids: torch.Tensor,
        evidence_ids: torch.Tensor | None = None,
        *,
        interface: str = "separate_reader",
        collect_telemetry: bool = False,
    ) -> dict[str, Any]:
        self._validate_inputs(query_ids, evidence_ids, str(interface))
        if interface == "gate_zero":
            local = self.cortex._forward_hidden(
                query_ids, collect_telemetry=collect_telemetry
            )
            return {
                "hidden": local["hidden"],
                "telemetry": {
                    "surface": self.surface,
                    "interface": "gate_zero",
                    "reader_active": False,
                    "reader_gate": 0.0,
                    "local_query_positions": int(query_ids.shape[1]),
                    "evidence_positions": 0,
                    "external_llm_used": False,
                    "owned_by_marulho": True,
                },
            }
        assert evidence_ids is not None
        if interface == "raw_context":
            combined = torch.cat((evidence_ids, query_ids), dim=1)
            raw = self.cortex._forward_hidden(
                combined, collect_telemetry=collect_telemetry
            )
            return {
                "hidden": raw["hidden"][:, -int(query_ids.shape[1]) :],
                "telemetry": {
                    "surface": self.surface,
                    "interface": "raw_context",
                    "reader_active": False,
                    "reader_gate": 0.0,
                    "local_query_positions": int(query_ids.shape[1]),
                    "evidence_positions": int(evidence_ids.shape[1]),
                    "external_llm_used": False,
                    "owned_by_marulho": True,
                },
            }
        hidden, gates = self._interleaved_hidden(query_ids, evidence_ids)
        return {
            "hidden": hidden,
            "telemetry": {
                "surface": self.surface,
                "interface": "separate_reader",
                "reader_active": True,
                "reader_gates": (
                    [float(value) for value in gates.detach().float().cpu()]
                    if collect_telemetry
                    else None
                ),
                "injection_layer_indices": list(
                    self.reader_config.injection_layer_indices
                ),
                "local_query_positions": int(query_ids.shape[1]),
                "evidence_positions": int(evidence_ids.shape[1]),
                "external_llm_used": False,
                "owned_by_marulho": True,
            },
        }

    def forward(
        self,
        query_ids: torch.Tensor,
        evidence_ids: torch.Tensor | None = None,
        *,
        interface: str = "separate_reader",
        collect_telemetry: bool = False,
    ) -> dict[str, Any]:
        result = self.forward_hidden(
            query_ids,
            evidence_ids,
            interface=str(interface),
            collect_telemetry=collect_telemetry,
        )
        return {
            "logits": self.cortex.lm_head(result["hidden"]),
            "telemetry": result["telemetry"],
        }

    def masked_next_token_loss(
        self,
        query_ids: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
        evidence_ids: torch.Tensor | None = None,
        *,
        interface: str = "separate_reader",
    ) -> torch.Tensor:
        output = self.forward(
            query_ids,
            evidence_ids,
            interface=str(interface),
            collect_telemetry=False,
        )
        runtime_targets = targets.to(device=self.device, dtype=torch.long)
        runtime_mask = loss_mask.to(device=self.device, dtype=torch.bool)
        if output["logits"].shape[:2] != runtime_targets.shape:
            raise ValueError("evidence reader targets must match query shape")
        return F.cross_entropy(
            output["logits"][runtime_mask], runtime_targets[runtime_mask]
        )

    @torch.no_grad()
    def generate_with_evidence(
        self,
        query_ids: torch.Tensor,
        evidence_ids: torch.Tensor | None,
        *,
        interface: str,
        max_new_tokens: int,
        eos_id: int | None,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 3,
    ) -> dict[str, Any]:
        self._validate_inputs(query_ids, evidence_ids, str(interface))
        was_training = bool(self.training)
        self.eval()
        try:
            generated = query_ids.to(device=self.device, dtype=torch.long)
            evidence = (
                None
                if evidence_ids is None
                else evidence_ids.to(device=self.device, dtype=torch.long)
            )
            finished = torch.zeros(
                generated.shape[0], device=self.device, dtype=torch.bool
            )
            new_token_count = 0
            for _ in range(max(0, int(max_new_tokens))):
                local_query = generated[:, -self.context_length :]
                if interface == "raw_context" and evidence is not None:
                    available = self.context_length - int(evidence.shape[1])
                    local_query = generated[:, -available:]
                logits = self.forward(
                    local_query,
                    evidence,
                    interface=str(interface),
                    collect_telemetry=False,
                )["logits"][:, -1]
                controlled, _control = _apply_decode_controls(
                    logits,
                    generated,
                    repetition_penalty=max(1.0, float(repetition_penalty)),
                    no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
                )
                next_id = controlled.argmax(dim=-1, keepdim=True)
                if eos_id is not None:
                    next_id = torch.where(
                        finished.unsqueeze(1),
                        torch.full_like(next_id, int(eos_id)),
                        next_id,
                    )
                    finished = finished | (next_id[:, 0] == int(eos_id))
                generated = torch.cat((generated, next_id), dim=1)
                new_token_count += 1
                if eos_id is not None and bool(finished.all()):
                    break
            return {
                "surface": self.generation_surface,
                "generated_ids": generated,
                "new_token_count": new_token_count,
                "interface": str(interface),
                "external_llm_used": False,
                "owned_by_marulho": True,
            }
        finally:
            self.train(was_training)

    def reader_parameter_report(self) -> dict[str, Any]:
        reader_parameters = {
            name: parameter
            for name, parameter in self.named_parameters()
            if not name.startswith("cortex.")
        }
        return {
            "surface": "marulho_evidence_reader_parameter_report.v1",
            "config": asdict(self.reader_config),
            "cortex_parameters": sum(
                int(parameter.numel()) for parameter in self.cortex.parameters()
            ),
            "reader_parameters": sum(
                int(parameter.numel()) for parameter in reader_parameters.values()
            ),
            "reader_parameter_tensors": len(reader_parameters),
            "reader_gates": [
                float(value)
                for value in torch.sigmoid(self.reader_gate_logits)
                .detach()
                .float()
                .cpu()
            ],
            "injection_layer_indices": list(
                self.reader_config.injection_layer_indices
            ),
            "external_llm_used": False,
            "owned_by_marulho": True,
        }
