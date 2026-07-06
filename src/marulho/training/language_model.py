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
from torch.nn import functional as F

from marulho.core.language_expert_dispatch_triton import (
    language_expert_dispatch,
    language_expert_dispatch_triton_stats,
    language_expert_dispatch_triton_stats_delta,
)
from marulho.core.language_eligibility_trace_triton import (
    language_eligibility_trace_final,
    language_eligibility_trace_triton_stats,
    language_eligibility_trace_triton_stats_delta,
)
from marulho.core.language_memory_slots_triton import (
    language_memory_slots,
    language_memory_slots_triton_stats,
    language_memory_slots_triton_stats_delta,
)
from marulho.core.language_plif_triton import (
    language_plif_forward,
    language_plif_forward_no_eligibility,
    language_plif_surrogate_update,
    language_plif_triton_stats,
    language_plif_triton_stats_delta,
)
from marulho.core.language_rmsnorm_triton import language_rmsnorm
from marulho.core.language_route_topk_triton import (
    language_route_topk,
    language_route_topk_triton_stats,
    language_route_topk_triton_stats_delta,
)
from marulho.core.language_sampled_vocab_ce_triton import (
    build_sampled_target_positions,
    build_sampled_vocab_ids,
    language_sampled_vocab_ce_triton_stats,
    language_sampled_vocab_ce_triton_stats_delta,
    language_sampled_vocab_cross_entropy,
    language_sampled_vocab_cross_entropy_pair,
)
from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer


@dataclass(frozen=True)
class LanguageModelConfig:
    vocab_size: int
    embedding_dim: int = 64
    state_dim: int = 128
    spike_slope: float = 5.0
    adaptive_timestep_budget: int = 1
    expert_count: int = 0
    active_expert_count: int = 1
    route_candidate_count: int = 0
    expert_hidden_dim: int = 0
    sampled_vocab_size: int = 0
    sampled_vocab_sparse_lm_head_gradient: bool = False
    sparse_token_embedding_gradients: bool = False
    generation_vocab_size: int = 0
    recurrent_gradient_horizon: int = 0
    memory_slot_count: int = 0
    memory_slot_candidate_count: int = 0
    active_memory_slot_count: int = 1
    memory_slot_init_std: float = 0.02
    active_language_path: str = "marulho_lm_head"


@dataclass(frozen=True)
class LanguageBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    sampled_vocab_ids: torch.Tensor | None = None
    sampled_target_positions: torch.Tensor | None = None
    memory_candidate_ids: torch.Tensor | None = None
    route_candidate_ids: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "LanguageBatch":
        return LanguageBatch(
            input_ids=self.input_ids.to(device),
            target_ids=self.target_ids.to(device),
            sampled_vocab_ids=(
                None
                if self.sampled_vocab_ids is None
                else self.sampled_vocab_ids.to(device)
            ),
            sampled_target_positions=(
                None
                if self.sampled_target_positions is None
                else self.sampled_target_positions.to(device)
            ),
            memory_candidate_ids=(
                None
                if self.memory_candidate_ids is None
                else self.memory_candidate_ids.to(device)
            ),
            route_candidate_ids=(
                None
                if self.route_candidate_ids is None
                else self.route_candidate_ids.to(device)
            ),
        )


def _cat_optional_batch_tensor(
    left: torch.Tensor | None,
    right: torch.Tensor | None,
    *,
    device: torch.device,
    name: str,
) -> torch.Tensor | None:
    if left is None and right is None:
        return None
    if left is None or right is None:
        raise ValueError(f"{name} must be present on both paired batches")
    if left.ndim != right.ndim or tuple(left.shape[1:]) != tuple(right.shape[1:]):
        raise ValueError(f"{name} paired batches must share non-batch dimensions")
    return torch.cat((left.to(device), right.to(device)), dim=0)


@dataclass(frozen=True)
class LanguageSplit:
    train: tuple[LanguageBatch, ...]
    eval: tuple[LanguageBatch, ...]
    report: dict[str, Any]


def precompute_sampled_vocab_batches(
    model: "MarulhoLanguageModel",
    batches: Sequence[LanguageBatch],
    *,
    assume_no_sleeping_experts: bool = False,
) -> tuple[tuple[LanguageBatch, ...], dict[str, Any]]:
    configured_sample_count = int(model.config.sampled_vocab_size)
    use_sampled_vocab = (
        configured_sample_count > 0
        and configured_sample_count < int(model.config.vocab_size)
    )
    use_memory_candidates = (
        max(0, int(model.config.memory_slot_count)) > 0
        and model._memory_slot_candidate_count() > 0
    )
    use_route_candidates = bool(model.routed_experts.enabled)
    if (
        not bool(use_sampled_vocab)
        and not bool(use_memory_candidates)
        and not bool(use_route_candidates)
    ):
        return tuple(batches), {
            "surface": "marulho_language_sampled_vocab_batch_precompute.v1",
            "enabled": False,
            "reason": "sampled_vocab_training_disabled",
            "batch_count": 0,
            "device": str(model.device),
            "memory_candidate_precompute": {
                "surface": "marulho_language_memory_candidate_batch_precompute.v1",
                "enabled": False,
                "reason": "language_memory_slots_disabled",
                "batch_count": 0,
                "device": str(model.device),
            },
            "route_candidate_precompute": {
                "surface": "marulho_language_route_candidate_batch_precompute.v1",
                "enabled": False,
                "reason": "language_expert_routing_disabled",
                "batch_count": 0,
                "device": str(model.device),
            },
        }

    cached: list[LanguageBatch] = []
    sampled_row_counts: list[int] = []
    target_position_counts: list[int] = []
    memory_candidate_counts: list[int] = []
    route_candidate_counts: list[int] = []
    for batch in batches:
        input_ids = batch.input_ids.to(model.device)
        target_ids = batch.target_ids.to(model.device)
        sampled_vocab_ids: torch.Tensor | None = None
        sampled_target_positions: torch.Tensor | None = None
        if bool(use_sampled_vocab):
            flat_targets = target_ids.reshape(-1).to(
                device=model.device,
                dtype=torch.long,
            )
            sampled_vocab_ids = build_sampled_vocab_ids(
                flat_targets,
                vocab_size=int(model.config.vocab_size),
                sample_count=configured_sample_count,
                device=model.device,
                validate_ids=False,
            )
            sampled_target_positions = build_sampled_target_positions(
                flat_targets,
                sampled_vocab_ids,
                device=model.device,
                validate_targets=True,
            )
            sampled_row_counts.append(int(sampled_vocab_ids.numel()))
            target_position_counts.append(int(sampled_target_positions.numel()))
        memory_candidate_ids = (
            model._language_memory_candidates(input_ids)
            if bool(use_memory_candidates)
            else None
        )
        if memory_candidate_ids is not None:
            memory_candidate_counts.append(int(memory_candidate_ids.shape[-1]))
        route_candidate_ids = (
            model._language_route_candidates(
                input_ids,
                assume_no_sleeping_experts=assume_no_sleeping_experts,
            )
            if bool(use_route_candidates)
            else None
        )
        if route_candidate_ids is not None:
            route_candidate_counts.append(int(route_candidate_ids.shape[-1]))
        cached.append(
            LanguageBatch(
                input_ids=input_ids,
                target_ids=target_ids,
                sampled_vocab_ids=sampled_vocab_ids,
                sampled_target_positions=sampled_target_positions,
                memory_candidate_ids=memory_candidate_ids,
                route_candidate_ids=route_candidate_ids,
            )
        )
    route_candidate_precompute_enabled = bool(route_candidate_counts)
    return tuple(cached), {
        "surface": "marulho_language_sampled_vocab_batch_precompute.v1",
        "enabled": bool(use_sampled_vocab),
        "reason": None if bool(use_sampled_vocab) else "sampled_vocab_training_disabled",
        "batch_count": len(cached) if bool(use_sampled_vocab) else 0,
        "device": str(model.device),
        "sampled_vocab_size": int(configured_sample_count),
        "min_sampled_rows": min(sampled_row_counts, default=0),
        "max_sampled_rows": max(sampled_row_counts, default=0),
        "min_target_positions": min(target_position_counts, default=0),
        "max_target_positions": max(target_position_counts, default=0),
        "hot_update_window_precomputed": True,
        "memory_candidate_precompute": {
            "surface": "marulho_language_memory_candidate_batch_precompute.v1",
            "enabled": bool(use_memory_candidates),
            "reason": (
                None
                if bool(use_memory_candidates)
                else "language_memory_slots_disabled"
            ),
            "batch_count": len(cached) if bool(use_memory_candidates) else 0,
            "device": str(model.device),
            "memory_slot_count": int(max(0, int(model.config.memory_slot_count))),
            "memory_slot_candidate_count": int(model._memory_slot_candidate_count()),
            "min_candidate_count": min(memory_candidate_counts, default=0),
            "max_candidate_count": max(memory_candidate_counts, default=0),
            "candidate_id_source": (
                "precomputed_batch_memory_candidate_ids"
                if bool(use_memory_candidates)
                else None
            ),
            "hot_update_window_precomputed": bool(use_memory_candidates),
        },
        "route_candidate_precompute": {
            "surface": "marulho_language_route_candidate_batch_precompute.v1",
            "enabled": bool(route_candidate_precompute_enabled),
            "reason": (
                None
                if bool(route_candidate_precompute_enabled)
                else (
                    "route_candidate_plan_dense_or_empty"
                    if bool(use_route_candidates)
                    else "language_expert_routing_disabled"
                )
            ),
            "batch_count": len(cached) if bool(route_candidate_precompute_enabled) else 0,
            "device": str(model.device),
            "expert_count": int(model.routed_experts.expert_count),
            "active_expert_count": int(model.routed_experts.active_expert_count),
            "route_candidate_count": int(model.routed_experts.route_candidate_count),
            "min_candidate_count": min(route_candidate_counts, default=0),
            "max_candidate_count": max(route_candidate_counts, default=0),
            "assume_no_sleeping_experts": bool(assume_no_sleeping_experts),
            "candidate_id_source": (
                "precomputed_batch_route_candidate_ids"
                if bool(route_candidate_precompute_enabled)
                else None
            ),
            "hot_update_window_precomputed": bool(route_candidate_precompute_enabled),
        },
    }


def _tensor_hash(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    payload = {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "values": cpu.reshape(-1).tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _uncollected_language_memory_slots_triton_delta(
    hidden: torch.Tensor,
) -> dict[str, Any]:
    return {
        "surface": "marulho_language_memory_slots_triton_stats_delta.v1",
        "triton_available": False,
        "triton_forward_calls": 0,
        "triton_autograd_forward_calls": 0,
        "torch_autograd_backward_calls": 0,
        "triton_forward_elements": 0,
        "torch_fallback_calls": 0,
        "torch_fallback_elements": 0,
        "triton_failure_count": 0,
        "last_failure": None,
        "last_device": str(hidden.device),
        "last_dtype": str(hidden.dtype),
        "triton_kernel_used": False,
        "triton_autograd_used": False,
        "telemetry_collected": False,
    }


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
        return language_rmsnorm(value, self.weight, eps=self.eps)


def _state_block_preallocate_no_grad_enabled() -> bool:
    raw = os.environ.get("MARULHO_LANGUAGE_STATE_BLOCK_PREALLOCATE_NO_GRAD")
    if raw is None:
        return False
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _state_block_defer_eligibility_no_grad_enabled() -> bool:
    raw = os.environ.get("MARULHO_LANGUAGE_STATE_BLOCK_DEFER_ELIGIBILITY_NO_GRAD")
    if raw is None:
        return False
    return raw.strip().lower() not in {"0", "false", "no", "off"}


class MarulhoSelectiveSpikingStateBlock(nn.Module):
    """Causal selective recurrent spike block for the MARULHO LM foundation."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        spike_slope: float = 5.0,
        adaptive_timestep_budget: int = 1,
        recurrent_gradient_horizon: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.spike_slope = float(spike_slope)
        self.adaptive_timestep_budget = max(1, int(adaptive_timestep_budget))
        self.recurrent_gradient_horizon = max(0, int(recurrent_gradient_horizon))
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

    def step(
        self,
        token_input: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        if token_input.ndim != 2:
            raise ValueError("Language state block step expects [batch, input_dim]")
        batch_size = int(token_input.shape[0])
        if state is None:
            current_state = self.initial_state(
                batch_size,
                device=token_input.device,
                dtype=token_input.dtype,
            )
        else:
            current_state = {
                "membrane": state["membrane"].to(
                    device=token_input.device,
                    dtype=token_input.dtype,
                ),
                "spikes": state["spikes"].to(
                    device=token_input.device,
                    dtype=token_input.dtype,
                ),
                "selective_state": state["selective_state"].to(
                    device=token_input.device,
                    dtype=token_input.dtype,
                ),
                "eligibility_trace": state.get(
                    "eligibility_trace",
                    torch.zeros_like(state["spikes"]),
                ).to(device=token_input.device, dtype=token_input.dtype),
            }

        membrane = current_state["membrane"]
        spikes = current_state["spikes"]
        selective_state = current_state["selective_state"]
        eligibility_trace = current_state["eligibility_trace"]
        raw_leak = self.raw_leak.to(device=token_input.device, dtype=token_input.dtype)
        base_threshold = self.threshold.to(device=token_input.device, dtype=token_input.dtype)
        current_gain = self.current_gain.to(device=token_input.device, dtype=token_input.dtype)
        token_input = self.input_norm(token_input)
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
        threshold = F.softplus(base_threshold + self.threshold_input_proj(token_input))
        current = current_gain * self.current_proj(token_input)
        drive = self.input_proj(token_input) + current + self.recurrent_proj(spikes)
        plif_stats_before = (
            language_plif_triton_stats()
            if bool(collect_telemetry)
            else None
        )
        mixed_state: torch.Tensor
        for _substep in range(self.adaptive_timestep_budget):
            if not torch.is_grad_enabled():
                (
                    membrane,
                    spikes,
                    selective_state,
                    eligibility_trace,
                    mixed_state,
                ) = language_plif_forward(
                    membrane=membrane,
                    spikes=spikes,
                    selective_state=selective_state,
                    eligibility_trace=eligibility_trace,
                    leak=leak,
                    threshold=threshold,
                    drive=drive,
                    state_decay=state_decay,
                    state_input=state_input,
                    state_output=state_output,
                )
            else:
                (
                    membrane,
                    spikes,
                    selective_state,
                    eligibility_trace,
                    mixed_state,
                ) = language_plif_surrogate_update(
                    membrane=membrane,
                    spikes=spikes,
                    selective_state=selective_state,
                    eligibility_trace=eligibility_trace,
                    leak=leak,
                    threshold=threshold,
                    drive=drive,
                    state_decay=state_decay,
                    state_input=state_input,
                    state_output=state_output,
                    spike_slope=self.spike_slope,
                )
        residual = self.residual_proj(token_input)
        hidden = self.output_norm(residual + self.state_output_proj(mixed_state))
        if bool(collect_telemetry):
            plif_delta = (
                language_plif_triton_stats_delta(
                    plif_stats_before,
                    language_plif_triton_stats(),
                )
                if plif_stats_before is not None
                else None
            )
            denominator = max(1, batch_size * self.state_dim)
            firing_fraction = (spikes > 0).sum(dim=0) / float(max(1, batch_size))
            dead_fraction = (firing_fraction <= 0).to(token_input.dtype).mean()
            over_firing_fraction = (firing_fraction >= 0.8).to(token_input.dtype).mean()
            telemetry = {
                "surface": "marulho_selective_spiking_state_block.v1",
                "spike_rate": float((spikes.sum() / float(denominator)).detach().cpu().item()),
                "dead_neuron_fraction": float(dead_fraction.detach().cpu().item()),
                "over_firing_fraction": float(over_firing_fraction.detach().cpu().item()),
                "adaptive_timestep_budget": int(self.adaptive_timestep_budget),
                "adaptive_step_count": int(self.adaptive_timestep_budget),
                "state_dim": self.state_dim,
                "time_steps": 1,
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
                "recurrent_gradient_horizon": int(self.recurrent_gradient_horizon),
                "truncated_bptt_applied": False,
                "truncated_bptt_boundary_count": 0,
                "gradient_horizon_policy": "single_step_streaming_state",
                "eligibility_trace_update_mode": "inline_plif_update",
                "eligibility_trace_scan_backend": "not_applicable_single_step",
                "plif_forward_backend": (
                    (
                        "triton_surrogate_forward_backward"
                        if torch.is_grad_enabled()
                        else "triton_forward_no_grad"
                    )
                    if plif_delta is not None
                    and bool(plif_delta.get("triton_kernel_used", False))
                    else (
                        (
                            "torch_surrogate_forward"
                            if torch.is_grad_enabled()
                            else "torch_forward_no_grad_fallback"
                        )
                        if plif_delta is not None
                        else "torch_surrogate_forward"
                    )
                ),
                "device": str(token_input.device),
            }
        else:
            telemetry = {
                "surface": "marulho_selective_spiking_state_block.v1",
                "telemetry_collected": False,
                "recurrent_gradient_horizon": int(self.recurrent_gradient_horizon),
                "truncated_bptt_applied": False,
                "truncated_bptt_boundary_count": 0,
                "gradient_horizon_policy": "single_step_streaming_state",
                "eligibility_trace_update_mode": "inline_plif_update",
                "eligibility_trace_scan_backend": "not_applicable_single_step",
                "device": str(token_input.device),
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

    def forward(
        self,
        inputs: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
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
        preallocate_no_grad_sequence = bool(
            not torch.is_grad_enabled()
            and _state_block_preallocate_no_grad_enabled()
            and int(time_steps) > 0
        )
        defer_eligibility_no_grad_sequence = bool(
            not torch.is_grad_enabled()
            and _state_block_defer_eligibility_no_grad_enabled()
            and self.adaptive_timestep_budget == 1
            and int(time_steps) > 1
        )
        initial_eligibility_trace = eligibility_trace
        mixed_states: list[torch.Tensor] = []
        mixed_state_sequence_buffer: torch.Tensor | None = (
            inputs.new_empty((batch_size, time_steps, self.state_dim))
            if preallocate_no_grad_sequence
            else None
        )
        spike_sequence_buffer: torch.Tensor | None = (
            inputs.new_empty((batch_size, time_steps, self.state_dim))
            if defer_eligibility_no_grad_sequence
            else None
        )
        spike_sum = inputs.new_tensor(0.0)
        active_neuron_counts = (
            inputs.new_zeros(self.state_dim) if bool(collect_telemetry) else None
        )
        normalized_inputs = self.input_norm(inputs)
        select_logits_all = self.select_proj(normalized_inputs)
        state_decay_all, state_input_all, state_output_all = select_logits_all.chunk(
            3,
            dim=-1,
        )
        state_decay_all = torch.sigmoid(state_decay_all)
        state_input_all = torch.sigmoid(state_input_all)
        state_output_all = torch.sigmoid(state_output_all)
        beta_input_all = self.beta_input_proj(normalized_inputs)
        threshold_input_all = self.threshold_input_proj(normalized_inputs)
        current_all = current_gain.view(1, 1, -1) * self.current_proj(normalized_inputs)
        input_drive_all = self.input_proj(normalized_inputs)
        residual_all = self.residual_proj(normalized_inputs)
        plif_stats_before = (
            language_plif_triton_stats()
            if bool(collect_telemetry)
            else None
        )
        eligibility_stats_before = (
            language_eligibility_trace_triton_stats()
            if bool(collect_telemetry)
            else None
        )
        gradient_horizon = int(self.recurrent_gradient_horizon)
        detach_recurrent_state = bool(
            torch.is_grad_enabled()
            and gradient_horizon > 0
            and self.adaptive_timestep_budget == 1
        )
        truncated_bptt_boundary_count = 0

        for step in range(time_steps):
            leak = torch.sigmoid(
                raw_leak
                + beta_input_all[:, step, :]
                + self.beta_state_proj(selective_state)
            )
            threshold = F.softplus(
                base_threshold + threshold_input_all[:, step, :]
            )
            drive = (
                input_drive_all[:, step, :]
                + current_all[:, step, :]
                + self.recurrent_proj(spikes)
            )
            mixed_state: torch.Tensor
            for _substep in range(self.adaptive_timestep_budget):
                if defer_eligibility_no_grad_sequence:
                    (
                        membrane,
                        spikes,
                        selective_state,
                        mixed_state,
                    ) = language_plif_forward_no_eligibility(
                        membrane=membrane,
                        spikes=spikes,
                        selective_state=selective_state,
                        leak=leak,
                        threshold=threshold,
                        drive=drive,
                        state_decay=state_decay_all[:, step, :],
                        state_input=state_input_all[:, step, :],
                        state_output=state_output_all[:, step, :],
                    )
                elif not torch.is_grad_enabled():
                    (
                        membrane,
                        spikes,
                        selective_state,
                        eligibility_trace,
                        mixed_state,
                    ) = language_plif_forward(
                        membrane=membrane,
                        spikes=spikes,
                        selective_state=selective_state,
                        eligibility_trace=eligibility_trace,
                        leak=leak,
                        threshold=threshold,
                        drive=drive,
                        state_decay=state_decay_all[:, step, :],
                        state_input=state_input_all[:, step, :],
                        state_output=state_output_all[:, step, :],
                    )
                else:
                    (
                        membrane,
                        spikes,
                        selective_state,
                        eligibility_trace,
                        mixed_state,
                    ) = language_plif_surrogate_update(
                        membrane=membrane,
                        spikes=spikes,
                        selective_state=selective_state,
                        eligibility_trace=eligibility_trace,
                        leak=leak,
                        threshold=threshold,
                        drive=drive,
                        state_decay=state_decay_all[:, step, :],
                        state_input=state_input_all[:, step, :],
                        state_output=state_output_all[:, step, :],
                        spike_slope=self.spike_slope,
                    )
            if spike_sequence_buffer is not None:
                spike_sequence_buffer[:, step, :].copy_(spikes)
            if mixed_state_sequence_buffer is not None:
                mixed_state_sequence_buffer[:, step, :].copy_(mixed_state)
            else:
                mixed_states.append(mixed_state)
            if bool(collect_telemetry):
                spike_sum = spike_sum + spikes.sum()
                assert active_neuron_counts is not None
                active_neuron_counts = active_neuron_counts + (spikes > 0).sum(dim=0)
            if (
                detach_recurrent_state
                and step + 1 < time_steps
                and (step + 1) % gradient_horizon == 0
            ):
                membrane = membrane.detach()
                spikes = spikes.detach()
                selective_state = selective_state.detach()
                eligibility_trace = eligibility_trace.detach()
                truncated_bptt_boundary_count += 1

        if spike_sequence_buffer is not None:
            eligibility_trace = language_eligibility_trace_final(
                initial_eligibility_trace,
                spike_sequence_buffer,
            )

        mixed_state_sequence = (
            mixed_state_sequence_buffer
            if mixed_state_sequence_buffer is not None
            else torch.stack(mixed_states, dim=1)
        )
        mixed_state_sequence_buffer_mode = (
            "preallocated_no_grad_mixed_state_sequence"
            if mixed_state_sequence_buffer is not None
            else "stacked_mixed_state_list"
        )
        hidden = self.output_norm(
            residual_all + self.state_output_proj(mixed_state_sequence)
        )
        if bool(collect_telemetry):
            plif_delta = (
                language_plif_triton_stats_delta(
                    plif_stats_before,
                    language_plif_triton_stats(),
                )
                if plif_stats_before is not None
                else None
            )
            eligibility_delta = (
                language_eligibility_trace_triton_stats_delta(
                    eligibility_stats_before,
                    language_eligibility_trace_triton_stats(),
                )
                if eligibility_stats_before is not None
                else None
            )
            denominator = max(1, batch_size * time_steps * self.state_dim)
            per_neuron_denominator = max(1, batch_size * time_steps)
            assert active_neuron_counts is not None
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
                "state_block_projection_mode": (
                    "batched_token_and_state_output_projection_recurrent_loop"
                ),
                "mixed_state_sequence_buffer_mode": mixed_state_sequence_buffer_mode,
                "recurrent_gradient_horizon": int(gradient_horizon),
                "truncated_bptt_applied": bool(truncated_bptt_boundary_count > 0),
                "truncated_bptt_boundary_count": int(truncated_bptt_boundary_count),
                "gradient_horizon_policy": (
                    "bounded_recurrent_state_detach"
                    if truncated_bptt_boundary_count > 0
                    else "full_sequence_bptt"
                ),
                "eligibility_trace_update_mode": (
                    "deferred_sequence_scan_no_grad"
                    if spike_sequence_buffer is not None
                    else "inline_plif_update"
                ),
                "eligibility_trace_sequence_buffer_mode": (
                    "spike_sequence_buffer"
                    if spike_sequence_buffer is not None
                    else "none"
                ),
                "eligibility_trace_scan_backend": (
                    "triton_final_scan"
                    if eligibility_delta is not None
                    and bool(eligibility_delta.get("triton_kernel_used", False))
                    else (
                        "torch_final_scan_fallback"
                        if spike_sequence_buffer is not None
                        else "not_applicable_inline_plif"
                    )
                ),
                "plif_forward_backend": (
                    (
                        (
                            "triton_surrogate_forward_backward"
                            if torch.is_grad_enabled()
                            else (
                                "triton_forward_no_grad_no_eligibility"
                                if spike_sequence_buffer is not None
                                else "triton_forward_no_grad"
                            )
                        )
                    )
                    if plif_delta is not None
                    and bool(plif_delta.get("triton_kernel_used", False))
                    else (
                        (
                            "torch_surrogate_forward"
                            if torch.is_grad_enabled()
                            else "torch_forward_no_grad_fallback"
                        )
                        if plif_delta is not None
                        else "torch_surrogate_forward"
                    )
                ),
                "device": str(inputs.device),
            }
        else:
            telemetry = {
                "surface": "marulho_selective_spiking_state_block.v1",
                "telemetry_collected": False,
                "state_block_projection_mode": (
                    "batched_token_and_state_output_projection_recurrent_loop"
                ),
                "mixed_state_sequence_buffer_mode": mixed_state_sequence_buffer_mode,
                "recurrent_gradient_horizon": int(gradient_horizon),
                "truncated_bptt_applied": bool(truncated_bptt_boundary_count > 0),
                "truncated_bptt_boundary_count": int(truncated_bptt_boundary_count),
                "gradient_horizon_policy": (
                    "bounded_recurrent_state_detach"
                    if truncated_bptt_boundary_count > 0
                    else "full_sequence_bptt"
                ),
                "eligibility_trace_update_mode": (
                    "deferred_sequence_scan_no_grad"
                    if spike_sequence_buffer is not None
                    else "inline_plif_update"
                ),
                "eligibility_trace_sequence_buffer_mode": (
                    "spike_sequence_buffer"
                    if spike_sequence_buffer is not None
                    else "none"
                ),
                "eligibility_trace_scan_backend": (
                    "unknown_deferred_no_telemetry"
                    if spike_sequence_buffer is not None
                    else "not_applicable_inline_plif"
                ),
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


class RoutedLanguageExpertLayer(nn.Module):
    """Sparse routed expert layer for MARULHO LM hidden states."""

    surface = "marulho_routed_language_experts.v1"

    def __init__(
        self,
        state_dim: int,
        *,
        expert_count: int,
        active_expert_count: int,
        route_candidate_count: int,
        expert_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.expert_count = max(0, int(expert_count))
        self.active_expert_count = max(1, int(active_expert_count))
        self.route_candidate_count = max(0, int(route_candidate_count))
        hidden_dim = int(expert_hidden_dim) if int(expert_hidden_dim) > 0 else self.state_dim * 2
        self.expert_hidden_dim = hidden_dim
        if self.expert_count > 0:
            self.route_keys = nn.Parameter(torch.empty(self.expert_count, self.state_dim))
            self.route_bias = nn.Parameter(torch.zeros(self.expert_count))
            self.register_buffer(
                "sleeping_expert_mask",
                torch.zeros(self.expert_count, dtype=torch.bool),
            )
            nn.init.normal_(self.route_keys, mean=0.0, std=self.state_dim**-0.5)
        else:
            self.register_parameter("route_keys", None)
            self.register_parameter("route_bias", None)
            self.register_buffer(
                "sleeping_expert_mask",
                torch.zeros(0, dtype=torch.bool),
            )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.state_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, self.state_dim),
                )
                for _ in range(self.expert_count)
            ]
        )
        self._inference_expert_stack_cache: dict[str, Any] | None = None

    @property
    def enabled(self) -> bool:
        return self.expert_count > 0

    def awake_expert_ids(self, device: torch.device | None = None) -> torch.Tensor:
        target_device = device or self.sleeping_expert_mask.device
        if not self.enabled:
            return torch.empty(0, dtype=torch.long, device=target_device)
        mask = ~self.sleeping_expert_mask.to(device=target_device)
        return torch.arange(
            self.expert_count,
            device=target_device,
            dtype=torch.long,
        )[mask]

    def sleeping_expert_ids(self) -> list[int]:
        if not self.enabled:
            return []
        return [
            int(value)
            for value in torch.nonzero(
                self.sleeping_expert_mask.detach().cpu(),
                as_tuple=False,
            ).flatten().tolist()
        ]

    def candidate_count(self) -> int:
        if not self.enabled:
            return 0
        awake_count = int((~self.sleeping_expert_mask).sum().detach().cpu().item())
        if awake_count <= 0:
            return 0
        if self.route_candidate_count <= 0:
            return awake_count
        return min(awake_count, self.route_candidate_count)

    def _expert_parameter_count(self) -> int:
        if not self.experts:
            return 0
        return sum(int(parameter.numel()) for parameter in self.experts[0].parameters())

    def _expert_parameter_version(self) -> tuple[int, ...]:
        versions: list[int] = []
        for expert in self.experts:
            first = expert[0]
            second = expert[2]
            versions.extend(
                [
                    int(first.weight._version),
                    int(first.bias._version),
                    int(second.weight._version),
                    int(second.bias._version),
                ]
            )
        return tuple(versions)

    def _stacked_expert_parameters(
        self,
        *,
        use_inference_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        version = self._expert_parameter_version() if use_inference_cache else ()
        cache = self._inference_expert_stack_cache
        if (
            use_inference_cache
            and cache is not None
            and cache.get("version") == version
        ):
            return (
                cache["first_weights"],
                cache["first_biases"],
                cache["second_weights"],
                cache["second_biases"],
            )

        first_weights = torch.stack(
            [expert[0].weight for expert in self.experts],
            dim=0,
        )
        first_biases = torch.stack(
            [expert[0].bias for expert in self.experts],
            dim=0,
        )
        second_weights = torch.stack(
            [expert[2].weight for expert in self.experts],
            dim=0,
        )
        second_biases = torch.stack(
            [expert[2].bias for expert in self.experts],
            dim=0,
        )
        if use_inference_cache:
            self._inference_expert_stack_cache = {
                "version": version,
                "first_weights": first_weights,
                "first_biases": first_biases,
                "second_weights": second_weights,
                "second_biases": second_biases,
            }
        return first_weights, first_biases, second_weights, second_biases

    def _dense_candidate_ids(
        self,
        *,
        batch_size: int,
        time_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        awake_ids = self.awake_expert_ids(device)
        return awake_ids.view(1, 1, int(awake_ids.numel())).expand(
            batch_size,
            time_steps,
            int(awake_ids.numel()),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        candidate_ids: torch.Tensor | None = None,
        *,
        collect_telemetry: bool = True,
        assume_no_sleeping_experts: bool = False,
        precomputed_candidate_ids: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if hidden.ndim != 3:
            raise ValueError("Routed language experts expect [batch, time, state_dim]")
        batch_size, time_steps, _state_dim = hidden.shape
        if not self.enabled:
            return hidden, {
                "surface": self.surface,
                "enabled": False,
                "total_columns": 0,
                "active_columns": 0,
                "candidate_rows_scored": 0,
                "output_candidate_count": 0,
                "runs_all_columns": False,
                "fallback_reason": "language_expert_routing_disabled",
                "route_device": str(hidden.device),
                "route_latency_ms": 0.0,
                "active_parameters_per_token": 0,
                "awake_columns": 0,
                "sleeping_columns": 0,
                "sleeping_expert_ids": [],
                "sleep_filter_applied": False,
                "sleeping_candidate_filtered_count": 0,
                "precomputed_candidate_ids_used": False,
            }

        started = time.perf_counter()
        if bool(assume_no_sleeping_experts):
            sleeping_mask = None
            sleeping_count = 0
            awake_ids = None
            awake_count = int(self.expert_count)
        else:
            sleeping_mask = self.sleeping_expert_mask.to(device=hidden.device)
            sleeping_count = int(sleeping_mask.sum().detach().cpu().item())
            awake_ids = self.awake_expert_ids(hidden.device)
            awake_count = int(awake_ids.numel())
        if awake_count <= 0:
            return hidden, {
                "surface": self.surface,
                "enabled": True,
                "route_plan_source": "all_language_experts_sleeping",
                "total_columns": int(self.expert_count),
                "active_columns": 0,
                "active_expert_count_per_token": 0,
                "candidate_rows_scored": 0,
                "route_candidate_count": 0,
                "output_candidate_count": 0,
                "runs_all_columns": False,
                "fallback_reason": "all_language_experts_sleeping",
                "route_device": str(hidden.device),
                "route_latency_ms": 0.0,
                "active_parameters_per_token": 0,
                "expert_parameters_per_column": int(self._expert_parameter_count()),
                "awake_columns": 0,
                "sleeping_columns": int(sleeping_count),
                "sleeping_expert_ids": self.sleeping_expert_ids(),
                "sleep_filter_applied": bool(sleeping_count > 0),
                "sleeping_candidate_filtered_count": 0,
                "precomputed_candidate_ids_used": bool(precomputed_candidate_ids),
            }
        if candidate_ids is None:
            candidate_ids = self._dense_candidate_ids(
                batch_size=batch_size,
                time_steps=time_steps,
                device=hidden.device,
            )
            runs_all_columns = sleeping_count == 0
            fallback_reason: str | None = (
                "route_candidate_plan_missing_dense_score"
                if runs_all_columns
                else None
            )
            route_plan_source = (
                "dense_all_experts" if runs_all_columns else "awake_expert_mask"
            )
            candidate_id_source = route_plan_source
        else:
            if candidate_ids.ndim == 1:
                candidate_ids = candidate_ids.view(1, 1, -1).expand(
                    batch_size,
                    time_steps,
                    int(candidate_ids.numel()),
                )
            if candidate_ids.ndim != 3:
                raise ValueError("candidate_ids must be [batch, time, candidate]")
            candidate_ids = candidate_ids.to(device=hidden.device, dtype=torch.long)
            runs_all_columns = (
                int(candidate_ids.shape[-1]) >= self.expert_count
                and sleeping_count == 0
            )
            fallback_reason = "route_candidate_plan_unbounded" if runs_all_columns else None
            route_plan_source = (
                "token_hash_candidate_bank_all_awake_direct_modulo"
                if bool(assume_no_sleeping_experts)
                else "token_hash_candidate_bank"
            )
            candidate_id_source = (
                "all_awake_direct_expert_ids"
                if bool(assume_no_sleeping_experts)
                else "awake_index_select"
            )

        candidate_ids = candidate_ids.remainder(self.expert_count)
        sleeping_candidate_filtered_count = 0
        sleeping_candidates = None
        if sleeping_mask is not None:
            sleeping_candidates = sleeping_mask.index_select(0, candidate_ids.reshape(-1)).reshape_as(
                candidate_ids
            )
            sleeping_candidate_filtered_count = int(
                sleeping_candidates.sum().detach().cpu().item()
            )
        if sleeping_candidate_filtered_count > 0:
            assert sleeping_candidates is not None
            assert awake_ids is not None
            replacement_positions = (
                torch.arange(
                    int(candidate_ids.shape[-1]),
                    device=hidden.device,
                    dtype=torch.long,
                )
                % max(1, awake_count)
            )
            replacements = awake_ids.index_select(0, replacement_positions).view(
                1,
                1,
                int(candidate_ids.shape[-1]),
            ).expand_as(candidate_ids)
            candidate_ids = torch.where(
                sleeping_candidates,
                replacements,
                candidate_ids,
            )
            route_plan_source = f"{route_plan_source}_sleep_filtered"
            candidate_id_source = f"{candidate_id_source}_sleep_filtered"
        candidate_count = int(candidate_ids.shape[-1])
        active_count = min(self.active_expert_count, candidate_count)
        route_topk_stats_before = (
            language_route_topk_triton_stats()
            if bool(collect_telemetry)
            else None
        )
        if not torch.is_grad_enabled():
            selected_flat_ids, _top_scores_flat, top_weights_flat = language_route_topk(
                hidden.reshape(-1, self.state_dim),
                candidate_ids.reshape(-1, candidate_count),
                self.route_keys,
                self.route_bias,
                active_count,
            )
            selected_expert_ids = selected_flat_ids.reshape(
                batch_size,
                time_steps,
                active_count,
            )
            top_weights = top_weights_flat.reshape(batch_size, time_steps, active_count)
        else:
            route_keys = self.route_keys[candidate_ids]
            route_bias = self.route_bias[candidate_ids]
            route_logits = (hidden.unsqueeze(-2) * route_keys).sum(dim=-1) + route_bias
            top_scores, top_positions = torch.topk(route_logits, k=active_count, dim=-1)
            top_weights = torch.softmax(top_scores, dim=-1)
            selected_expert_ids = candidate_ids.gather(dim=-1, index=top_positions)
        route_topk_delta = (
            language_route_topk_triton_stats_delta(
                route_topk_stats_before,
                language_route_topk_triton_stats(),
            )
            if route_topk_stats_before is not None
            else None
        )

        flat_hidden = hidden.reshape(-1, self.state_dim)
        flat_ids = selected_expert_ids.reshape(-1, active_count)
        flat_weights = top_weights.reshape(-1, active_count)
        (
            first_weights,
            first_biases,
            second_weights,
            second_biases,
        ) = self._stacked_expert_parameters(
            use_inference_cache=not torch.is_grad_enabled(),
        )
        dispatch_stats_before = (
            language_expert_dispatch_triton_stats()
            if bool(collect_telemetry)
            else None
        )
        if not torch.is_grad_enabled():
            expert_delta = language_expert_dispatch(
                flat_hidden,
                flat_ids,
                flat_weights,
                first_weights,
                first_biases,
                second_weights,
                second_biases,
            )
        else:
            selected_flat = flat_ids.reshape(-1)
            selected_first_weights = first_weights.index_select(0, selected_flat).reshape(
                flat_ids.shape[0],
                active_count,
                self.expert_hidden_dim,
                self.state_dim,
            )
            selected_first_biases = first_biases.index_select(0, selected_flat).reshape(
                flat_ids.shape[0],
                active_count,
                self.expert_hidden_dim,
            )
            selected_second_weights = second_weights.index_select(
                0,
                selected_flat,
            ).reshape(
                flat_ids.shape[0],
                active_count,
                self.state_dim,
                self.expert_hidden_dim,
            )
            selected_second_biases = second_biases.index_select(0, selected_flat).reshape(
                flat_ids.shape[0],
                active_count,
                self.state_dim,
            )
            hidden_column = flat_hidden.view(flat_hidden.shape[0], 1, self.state_dim, 1)
            expert_hidden = torch.matmul(
                selected_first_weights,
                hidden_column,
            ).squeeze(-1)
            expert_hidden = F.silu(expert_hidden + selected_first_biases)
            expert_outputs = (
                torch.matmul(
                    selected_second_weights,
                    expert_hidden.unsqueeze(-1),
                ).squeeze(-1)
                + selected_second_biases
            )
            expert_delta = (expert_outputs * flat_weights.unsqueeze(-1)).sum(dim=1)
        dispatch_delta = (
            language_expert_dispatch_triton_stats_delta(
                dispatch_stats_before,
                language_expert_dispatch_triton_stats(),
            )
            if dispatch_stats_before is not None
            else None
        )

        routed = hidden + expert_delta.reshape_as(hidden)
        route_latency_ms = (time.perf_counter() - started) * 1000.0
        if bool(collect_telemetry):
            active_columns = int(torch.unique(flat_ids.detach()).numel())
            sleeping_expert_ids = self.sleeping_expert_ids()
        else:
            active_columns = 0
            sleeping_expert_ids = []
        expert_parameters = self._expert_parameter_count()
        active_parameters_per_token = int(active_count * expert_parameters)
        if torch.is_grad_enabled():
            route_selection_backend = "torch_grad_route_topk"
        elif route_topk_delta is not None and bool(
            route_topk_delta.get("triton_kernel_used", False)
        ):
            route_selection_backend = "triton_route_vote_topk"
        else:
            route_selection_backend = "torch_route_topk"
        if torch.is_grad_enabled():
            expert_dispatch_backend = "torch_selected_expert_batched_matmul_dispatch"
        elif dispatch_delta is not None and bool(
            dispatch_delta.get("triton_kernel_used", False)
        ):
            expert_dispatch_backend = "triton_block_sparse_dispatch"
        else:
            expert_dispatch_backend = "torch_selected_expert_dispatch"
        return routed, {
            "surface": self.surface,
            "enabled": True,
            "route_plan_source": route_plan_source,
            "total_columns": int(self.expert_count),
            "active_columns": int(active_columns),
            "active_expert_count_per_token": int(active_count),
            "candidate_rows_scored": int(batch_size * time_steps * candidate_count),
            "route_candidate_count": int(candidate_count),
            "output_candidate_count": int(active_count),
            "runs_all_columns": bool(runs_all_columns),
            "fallback_reason": fallback_reason,
            "route_device": str(hidden.device),
            "route_latency_ms": float(route_latency_ms),
            "active_parameters_per_token": active_parameters_per_token,
            "expert_parameters_per_column": int(expert_parameters),
            "awake_columns": int(awake_count),
            "sleeping_columns": int(sleeping_count),
            "sleeping_expert_ids": sleeping_expert_ids,
            "sleep_filter_applied": bool(sleeping_count > 0),
            "sleeping_candidate_filtered_count": int(
                sleeping_candidate_filtered_count
            ),
            "candidate_id_source": candidate_id_source,
            "all_awake_candidate_fastpath": bool(
                candidate_id_source == "all_awake_direct_expert_ids"
            ),
            "precomputed_candidate_ids_used": bool(precomputed_candidate_ids),
            "route_selection_backend": route_selection_backend,
            "route_topk_triton_stats_delta": route_topk_delta,
            "expert_dispatch_backend": expert_dispatch_backend,
            "expert_dispatch_triton_stats_delta": dispatch_delta,
        }


def _valid_generated_token_tensor(
    token_ids: torch.Tensor,
    *,
    vocab_size: int,
) -> torch.Tensor:
    vocab_size = max(0, int(vocab_size))
    if token_ids.numel() <= 0 or vocab_size <= 0:
        return token_ids.new_empty((0,), dtype=torch.long)
    token_ids = token_ids.to(dtype=torch.long)
    return token_ids[(token_ids >= 0) & (token_ids < vocab_size)]


def _no_repeat_ngram_banned_token_tensor(
    token_ids: torch.Tensor,
    *,
    ngram_size: int,
    vocab_size: int,
) -> torch.Tensor:
    ngram_size = max(0, int(ngram_size))
    vocab_size = max(0, int(vocab_size))
    if ngram_size <= 0 or vocab_size <= 0:
        return token_ids.new_empty((0,), dtype=torch.long)
    tokens = _valid_generated_token_tensor(token_ids, vocab_size=vocab_size)
    if ngram_size == 1:
        return torch.unique(tokens, sorted=True)
    if int(tokens.numel()) < ngram_size:
        return token_ids.new_empty((0,), dtype=torch.long)
    prefix_length = ngram_size - 1
    prefix = tokens[-prefix_length:]
    windows = tokens.unfold(0, ngram_size, 1)
    matches = (windows[:, :prefix_length] == prefix).all(dim=1)
    return torch.unique(windows[matches, -1], sorted=True)


def _apply_generation_decode_controls(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    *,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    repetition_penalty = max(1.0, float(repetition_penalty))
    no_repeat_ngram_size = max(0, int(no_repeat_ngram_size))
    if repetition_penalty <= 1.0 and no_repeat_ngram_size <= 0:
        return logits, {
            "repetition_penalty_adjusted_token_count": 0,
            "no_repeat_ngram_banned_token_count": 0,
            "decode_control_fallback_count": 0,
        }

    adjusted = logits.clone()
    vocab_size = int(adjusted.shape[-1])
    repetition_penalty_adjusted_token_count = 0
    no_repeat_ngram_banned_token_count = 0
    decode_control_fallback_count = 0
    for batch_index, row in enumerate(generated_ids.detach()):
        previous_tokens = _valid_generated_token_tensor(
            row,
            vocab_size=vocab_size,
        )
        if repetition_penalty > 1.0 and int(previous_tokens.numel()) > 0:
            unique_tokens = torch.unique(previous_tokens, sorted=True)
            selected = adjusted[batch_index].index_select(0, unique_tokens)
            penalized = torch.where(
                selected > 0,
                selected / repetition_penalty,
                selected * repetition_penalty,
            )
            adjusted[batch_index].index_copy_(0, unique_tokens, penalized)
            repetition_penalty_adjusted_token_count += int(unique_tokens.numel())
        banned_tokens = _no_repeat_ngram_banned_token_tensor(
            previous_tokens,
            ngram_size=no_repeat_ngram_size,
            vocab_size=vocab_size,
        )
        if int(banned_tokens.numel()) > 0:
            if int(banned_tokens.numel()) >= vocab_size:
                decode_control_fallback_count += 1
            else:
                adjusted[batch_index, banned_tokens] = -torch.inf
                no_repeat_ngram_banned_token_count += int(banned_tokens.numel())
    return adjusted, {
        "repetition_penalty_adjusted_token_count": int(
            repetition_penalty_adjusted_token_count
        ),
        "no_repeat_ngram_banned_token_count": int(
            no_repeat_ngram_banned_token_count
        ),
        "decode_control_fallback_count": int(decode_control_fallback_count),
    }


class MarulhoLanguageModel(nn.Module):
    """MARULHO-owned next-token language model foundation."""

    def __init__(self, config: LanguageModelConfig) -> None:
        super().__init__()
        if int(config.vocab_size) <= 1:
            raise ValueError("LanguageModelConfig.vocab_size must be greater than one")
        if int(config.generation_vocab_size) < 0:
            raise ValueError("generation_vocab_size must be non-negative")
        if int(config.generation_vocab_size) > int(config.vocab_size):
            raise ValueError("generation_vocab_size cannot exceed vocab_size")
        if int(config.recurrent_gradient_horizon) < 0:
            raise ValueError("recurrent_gradient_horizon must be non-negative")
        if int(config.memory_slot_count) < 0:
            raise ValueError("memory_slot_count must be non-negative")
        if int(config.memory_slot_candidate_count) < 0:
            raise ValueError("memory_slot_candidate_count must be non-negative")
        if int(config.active_memory_slot_count) < 1:
            raise ValueError("active_memory_slot_count must be at least one")
        if not math.isfinite(float(config.memory_slot_init_std)):
            raise ValueError("memory_slot_init_std must be finite")
        if float(config.memory_slot_init_std) < 0.0:
            raise ValueError("memory_slot_init_std must be non-negative")
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            sparse=bool(config.sparse_token_embedding_gradients),
        )
        self.state_block = MarulhoSelectiveSpikingStateBlock(
            config.embedding_dim,
            config.state_dim,
            spike_slope=config.spike_slope,
            adaptive_timestep_budget=config.adaptive_timestep_budget,
            recurrent_gradient_horizon=config.recurrent_gradient_horizon,
        )
        self.routed_experts = RoutedLanguageExpertLayer(
            config.state_dim,
            expert_count=config.expert_count,
            active_expert_count=config.active_expert_count,
            route_candidate_count=config.route_candidate_count,
            expert_hidden_dim=config.expert_hidden_dim,
        )
        self.lm_head = nn.Linear(config.state_dim, config.vocab_size)
        memory_slot_count = max(0, int(config.memory_slot_count))
        if memory_slot_count > 0:
            self.memory_slots = nn.Parameter(
                torch.empty(memory_slot_count, config.state_dim)
            )
            init_std = float(config.memory_slot_init_std)
            if init_std > 0.0:
                nn.init.normal_(self.memory_slots, mean=0.0, std=init_std)
            else:
                nn.init.zeros_(self.memory_slots)
            self.memory_slot_gate = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("memory_slots", None)
            self.register_parameter("memory_slot_gate", None)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def generation_vocab_size(self) -> int:
        configured = int(self.config.generation_vocab_size)
        if configured <= 0:
            return int(self.config.vocab_size)
        return max(2, min(configured, int(self.config.vocab_size)))

    def generation_decode_policy(
        self,
        *,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> dict[str, Any]:
        repetition_penalty = max(1.0, float(repetition_penalty))
        no_repeat_ngram_size = max(0, int(no_repeat_ngram_size))
        generation_vocab_size = int(self.generation_vocab_size)
        model_vocab_size = int(self.config.vocab_size)
        return {
            "surface": "marulho_language_generation_decode_policy.v1",
            "decode_strategy": "greedy_argmax",
            "decode_controls_backend": "torch_device_tensor",
            "decode_controls_cpu_token_copy": False,
            "model_vocab_size": model_vocab_size,
            "generation_vocab_size": generation_vocab_size,
            "full_model_vocab_logits_materialized": bool(
                generation_vocab_size >= model_vocab_size
            ),
            "padded_vocab_rows_masked": max(0, model_vocab_size - generation_vocab_size),
            "repetition_penalty": float(repetition_penalty),
            "repetition_penalty_applied": bool(repetition_penalty > 1.0),
            "no_repeat_ngram_size": int(no_repeat_ngram_size),
            "no_repeat_ngram_applied": bool(no_repeat_ngram_size > 0),
            "policy": (
                "limit_generation_to_configured_vocab_rows"
                if generation_vocab_size < model_vocab_size
                else "full_model_vocab_generation"
            ),
        }

    def _lm_head_logits(
        self,
        hidden: torch.Tensor,
        *,
        decode_vocab_only: bool = False,
    ) -> torch.Tensor:
        if not bool(decode_vocab_only):
            return self.lm_head(hidden)
        generation_vocab_size = int(self.generation_vocab_size)
        if generation_vocab_size >= int(self.config.vocab_size):
            return self.lm_head(hidden)
        return F.linear(
            hidden,
            self.lm_head.weight[:generation_vocab_size],
            self.lm_head.bias[:generation_vocab_size],
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        assume_no_sleeping_experts: bool = False,
        decode_vocab_only: bool = False,
        memory_candidate_ids: torch.Tensor | None = None,
        route_candidate_ids: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        result = self._forward_hidden(
            input_ids,
            state,
            collect_telemetry=collect_telemetry,
            assume_no_sleeping_experts=assume_no_sleeping_experts,
            memory_candidate_ids=memory_candidate_ids,
            route_candidate_ids=route_candidate_ids,
        )
        logits = self._lm_head_logits(
            result["hidden"],
            decode_vocab_only=decode_vocab_only,
        )
        telemetry = dict(result["telemetry"])
        if bool(decode_vocab_only):
            telemetry["generation_decode"] = self.generation_decode_policy()
        return {
            "logits": logits,
            "state": result["state"],
            "telemetry": telemetry,
        }

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        collect_memory_evidence: bool | None = None,
        assume_no_sleeping_experts: bool = False,
        memory_candidate_ids: torch.Tensor | None = None,
        route_candidate_ids: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("Language model expects input_ids shaped [batch, time]")
        runtime_input_ids = input_ids.to(self.device)
        embeddings = self.token_embedding(runtime_input_ids)
        hidden, next_state, telemetry = self.state_block(
            embeddings,
            state,
            collect_telemetry=collect_telemetry,
        )
        hidden, memory_telemetry = self._apply_memory_slots(
            hidden,
            runtime_input_ids,
            collect_telemetry=collect_telemetry,
            collect_evidence=(
                collect_telemetry
                if collect_memory_evidence is None
                else bool(collect_memory_evidence)
            ),
            candidate_ids=memory_candidate_ids,
        )
        precomputed_route_candidates = route_candidate_ids is not None
        if route_candidate_ids is None:
            route_candidates = self._language_route_candidates(
                runtime_input_ids,
                assume_no_sleeping_experts=assume_no_sleeping_experts,
            )
        else:
            route_candidates = route_candidate_ids.to(
                device=runtime_input_ids.device,
                dtype=torch.long,
            )
            expected_shape = (*tuple(runtime_input_ids.shape), route_candidates.shape[-1])
            if tuple(route_candidates.shape) != expected_shape:
                raise ValueError(
                    "route_candidate_ids must be shaped [batch, time, candidates]"
                )
        hidden, routing_telemetry = self.routed_experts(
            hidden,
            route_candidates,
            collect_telemetry=collect_telemetry,
            assume_no_sleeping_experts=assume_no_sleeping_experts,
            precomputed_candidate_ids=precomputed_route_candidates,
        )
        telemetry = {
            **telemetry,
            "active_language_path": self.config.active_language_path,
            "external_llm_used": False,
            "owned_by_marulho": True,
            "vocab_size": self.config.vocab_size,
            "memory": memory_telemetry,
            "routing": routing_telemetry,
        }
        return {
            "hidden": hidden,
            "state": next_state,
            "telemetry": telemetry,
        }

    def _memory_slot_candidate_count(self) -> int:
        slot_count = max(0, int(self.config.memory_slot_count))
        if slot_count <= 0:
            return 0
        configured = int(self.config.memory_slot_candidate_count)
        if configured <= 0:
            return min(slot_count, max(1, int(self.config.active_memory_slot_count)))
        return min(slot_count, configured)

    def _language_memory_candidates(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        slot_count = max(0, int(self.config.memory_slot_count))
        candidate_count = self._memory_slot_candidate_count()
        if slot_count <= 0 or candidate_count <= 0:
            return None
        offsets = torch.arange(candidate_count, device=input_ids.device, dtype=torch.long)
        return (input_ids.to(torch.long).unsqueeze(-1) + offsets) % int(slot_count)

    def _apply_memory_slots(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        collect_telemetry: bool,
        collect_evidence: bool | None = None,
        candidate_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        collect_evidence = (
            bool(collect_telemetry)
            if collect_evidence is None
            else bool(collect_evidence)
        )
        slot_count = max(0, int(self.config.memory_slot_count))
        if slot_count <= 0 or self.memory_slots is None:
            return hidden, {
                "surface": "marulho_language_memory_slots.v1",
                "enabled": False,
                "total_slots": 0,
                "candidate_slot_count": 0,
                "active_slots_per_token": 0,
                "candidate_slots_scored": 0,
                "runs_all_slots": False,
                "fallback_reason": "language_memory_slots_disabled",
                "memory_device": str(hidden.device),
                "active_parameters_per_token": 0,
            }
        precomputed_candidate_ids = candidate_ids is not None
        if candidate_ids is None:
            candidate_ids = self._language_memory_candidates(input_ids)
            candidate_id_source = "token_hash_memory_slot_bank"
        else:
            candidate_ids = candidate_ids.to(
                device=hidden.device,
                dtype=torch.long,
            )
            expected_shape = (*tuple(input_ids.shape), candidate_ids.shape[-1])
            if tuple(candidate_ids.shape) != expected_shape:
                raise ValueError(
                    "memory_candidate_ids must be shaped [batch, time, candidates]"
                )
            candidate_id_source = "precomputed_batch_memory_candidate_ids"
        if candidate_ids is None:
            return hidden, {
                "surface": "marulho_language_memory_slots.v1",
                "enabled": True,
                "total_slots": int(slot_count),
                "candidate_slot_count": 0,
                "active_slots_per_token": 0,
                "candidate_slots_scored": 0,
                "runs_all_slots": False,
                "fallback_reason": "memory_slot_candidate_plan_empty",
                "memory_device": str(hidden.device),
                "active_parameters_per_token": 0,
                "candidate_id_source": "memory_slot_candidate_plan_empty",
                "precomputed_candidate_ids_used": False,
            }
        candidate_count = int(candidate_ids.shape[-1])
        if candidate_count <= 0:
            return hidden, {
                "surface": "marulho_language_memory_slots.v1",
                "enabled": True,
                "total_slots": int(slot_count),
                "candidate_slot_count": 0,
                "active_slots_per_token": 0,
                "candidate_slots_scored": 0,
                "runs_all_slots": False,
                "fallback_reason": "memory_slot_candidate_plan_empty",
                "memory_device": str(hidden.device),
                "active_parameters_per_token": 0,
                "candidate_id_source": "memory_slot_candidate_plan_empty",
                "precomputed_candidate_ids_used": bool(precomputed_candidate_ids),
            }
        active_count = min(max(1, int(self.config.active_memory_slot_count)), candidate_count)
        memory_slots_triton_before = (
            language_memory_slots_triton_stats() if bool(collect_evidence) else None
        )
        if not torch.is_grad_enabled():
            flat_hidden = hidden.reshape(-1, int(self.config.state_dim))
            flat_candidate_ids = candidate_ids.reshape(-1, candidate_count)
            routed = language_memory_slots(
                flat_hidden,
                flat_candidate_ids,
                self.memory_slots,
                self.memory_slot_gate,
                active_count,
                prefer_triton=True,
            ).reshape_as(hidden)
            memory_slots_triton_delta = (
                language_memory_slots_triton_stats_delta(
                    memory_slots_triton_before,
                    language_memory_slots_triton_stats(),
                )
                if memory_slots_triton_before is not None
                else _uncollected_language_memory_slots_triton_delta(hidden)
            )
            retrieval_backend = (
                "triton_no_grad_bounded_memory_slots"
                if bool(memory_slots_triton_delta.get("triton_kernel_used", False))
                else "torch_no_grad_bounded_memory_slots"
                if bool(collect_evidence)
                else "not_collected_no_grad_bounded_memory_slots"
            )
        else:
            flat_hidden = hidden.reshape(-1, int(self.config.state_dim))
            flat_candidate_ids = candidate_ids.reshape(-1, candidate_count)
            routed = language_memory_slots(
                flat_hidden,
                flat_candidate_ids,
                self.memory_slots,
                self.memory_slot_gate,
                active_count,
                prefer_triton=True,
            ).reshape_as(hidden)
            memory_slots_triton_delta = (
                language_memory_slots_triton_stats_delta(
                    memory_slots_triton_before,
                    language_memory_slots_triton_stats(),
                )
                if memory_slots_triton_before is not None
                else _uncollected_language_memory_slots_triton_delta(hidden)
            )
            retrieval_backend = (
                "triton_forward_torch_backward_bounded_memory_slots"
                if bool(memory_slots_triton_delta.get("triton_autograd_used", False))
                else "torch_autograd_bounded_memory_slots"
                if bool(collect_evidence)
                else "not_collected_autograd_bounded_memory_slots"
            )
        runs_all_slots = candidate_count >= slot_count
        return routed, {
            "surface": "marulho_language_memory_slots.v1",
            "enabled": True,
            "total_slots": int(slot_count),
            "candidate_slot_count": int(candidate_count),
            "active_slots_per_token": int(active_count),
            "candidate_slots_scored": int(hidden.shape[0] * hidden.shape[1] * candidate_count),
            "runs_all_slots": bool(runs_all_slots),
            "fallback_reason": "memory_slot_candidate_plan_unbounded"
            if bool(runs_all_slots)
            else None,
            "memory_device": str(hidden.device),
            "active_parameters_per_token": int(active_count * int(self.config.state_dim)),
            "candidate_id_source": candidate_id_source,
            "precomputed_candidate_ids_used": bool(precomputed_candidate_ids),
            "memory_gate_readback": False,
            "memory_slot_initialization": "nonzero_slots_zero_gate",
            "memory_slot_init_std": float(self.config.memory_slot_init_std),
            "memory_slot_retrieval_backend": retrieval_backend,
            "memory_slot_triton_stats_delta": memory_slots_triton_delta,
            "collect_telemetry": bool(collect_telemetry),
            "evidence_collected": bool(collect_evidence),
        }

    def _language_route_candidates(
        self,
        input_ids: torch.Tensor,
        *,
        assume_no_sleeping_experts: bool = False,
    ) -> torch.Tensor | None:
        if not self.routed_experts.enabled:
            return None
        if bool(assume_no_sleeping_experts):
            awake_count = int(self.routed_experts.expert_count)
            awake_ids = None
        else:
            awake_ids = self.routed_experts.awake_expert_ids(input_ids.device)
            awake_count = int(awake_ids.numel())
        if awake_count <= 0:
            return None
        if bool(assume_no_sleeping_experts):
            candidate_count = (
                awake_count
                if self.routed_experts.route_candidate_count <= 0
                else min(awake_count, self.routed_experts.route_candidate_count)
            )
        else:
            candidate_count = self.routed_experts.candidate_count()
        if candidate_count <= 0:
            return None
        if (
            awake_count == self.routed_experts.expert_count
            and candidate_count >= awake_count
        ):
            return None
        offsets = torch.arange(candidate_count, device=input_ids.device, dtype=torch.long)
        candidate_positions = (input_ids.to(torch.long).unsqueeze(-1) + offsets) % int(
            awake_count
        )
        if bool(assume_no_sleeping_experts):
            return candidate_positions
        assert awake_ids is not None
        return awake_ids.index_select(0, candidate_positions.reshape(-1)).reshape(
            candidate_positions.shape
        )

    def _next_token_loss_from_hidden(
        self,
        hidden: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        state: Mapping[str, torch.Tensor] | None,
        telemetry: Mapping[str, Any],
        collect_telemetry: bool,
        sampled_vocab_ids: torch.Tensor | None = None,
        sampled_target_positions: torch.Tensor | None = None,
        return_evidence: bool = True,
    ) -> dict[str, Any]:
        flat_hidden = hidden.reshape(-1, int(self.config.state_dim))
        flat_targets = target_ids.to(
            device=flat_hidden.device,
            dtype=torch.long,
        ).reshape(-1)
        configured_sample_count = int(self.config.sampled_vocab_size)
        use_sampled_vocab = (
            configured_sample_count > 0
            and configured_sample_count < int(self.config.vocab_size)
        )
        if use_sampled_vocab:
            precomputed_sampled_vocab = sampled_vocab_ids is not None
            precomputed_target_positions = sampled_target_positions is not None
            if sampled_vocab_ids is None:
                sampled_vocab_ids = build_sampled_vocab_ids(
                    flat_targets,
                    vocab_size=int(self.config.vocab_size),
                    sample_count=configured_sample_count,
                    device=flat_hidden.device,
                    validate_ids=False,
                )
                sampled_vocab_id_source = "built_per_batch_targets"
            else:
                sampled_vocab_ids = sampled_vocab_ids.to(
                    device=flat_hidden.device,
                    dtype=torch.long,
                ).reshape(-1)
                sampled_vocab_id_source = "precomputed_batch_sampled_vocab_ids"
            runtime_target_positions = (
                None
                if sampled_target_positions is None
                else sampled_target_positions.to(
                    device=flat_hidden.device,
                    dtype=torch.long,
                ).reshape(-1)
            )
            sampled_target_position_source = (
                "precomputed_batch_target_positions"
                if runtime_target_positions is not None
                else "loss_runtime_target_match"
            )
            validate_sampled_targets = bool(
                precomputed_sampled_vocab and not precomputed_target_positions
            )
            sampled_vocab_stats_before = (
                language_sampled_vocab_ce_triton_stats()
                if bool(return_evidence)
                else None
            )
            loss = language_sampled_vocab_cross_entropy(
                flat_hidden,
                flat_targets,
                sampled_vocab_ids,
                self.lm_head.weight,
                self.lm_head.bias,
                prefer_triton=True,
                validate_targets=validate_sampled_targets,
                sparse_weight_gradient=bool(
                    self.config.sampled_vocab_sparse_lm_head_gradient
                ),
                sampled_target_positions=runtime_target_positions,
            )
            if not bool(return_evidence):
                return {
                    "logits": None,
                    "state": state,
                    "loss": loss,
                    "loss_kind": "sampled_adaptive_vocab_cross_entropy",
                }
            sampled_vocab_stats_delta = (
                language_sampled_vocab_ce_triton_stats_delta(
                    sampled_vocab_stats_before,
                    language_sampled_vocab_ce_triton_stats(),
                )
                if sampled_vocab_stats_before is not None
                else {
                    "surface": (
                        "marulho_language_sampled_vocab_ce_triton_stats_delta.v1"
                    ),
                    "triton_available": False,
                    "triton_forward_calls": 0,
                    "triton_forward_elements": 0,
                    "triton_autograd_forward_calls": 0,
                    "triton_autograd_backward_calls": 0,
                    "triton_autograd_backward_elements": 0,
                    "torch_fallback_calls": 0,
                    "torch_fallback_elements": 0,
                    "triton_failure_count": 0,
                    "last_failure": None,
                    "last_device": str(flat_hidden.device),
                    "last_dtype": str(flat_hidden.dtype),
                    "triton_kernel_used": False,
                    "telemetry_collected": False,
                }
            )
            sampled_vocab_hash = (
                _tensor_hash(sampled_vocab_ids)
                if bool(collect_telemetry)
                else None
            )
            triton_forward_training = bool(
                sampled_vocab_stats_delta.get("triton_kernel_used")
                and int(
                    sampled_vocab_stats_delta.get(
                        "triton_autograd_forward_calls",
                        0,
                    )
                    or 0
                )
                > 0
            )
            loss_evidence = {
                "surface": "marulho_language_vocab_loss_evidence.v1",
                "loss_kind": "sampled_adaptive_vocab_cross_entropy",
                "full_vocab_logits_materialized": False,
                "sampled_vocab_training": True,
                "configured_sampled_vocab_size": configured_sample_count,
                "actual_sampled_vocab_size": int(sampled_vocab_ids.numel()),
                "model_vocab_size": int(self.config.vocab_size),
                "target_token_count": int(flat_targets.numel()),
                "sampled_vocab_ids_device": str(sampled_vocab_ids.device),
                "sampled_vocab_id_source": sampled_vocab_id_source,
                "sampled_target_position_source": sampled_target_position_source,
                "precomputed_sampled_vocab_used": bool(precomputed_sampled_vocab),
                "precomputed_target_positions_used": bool(
                    precomputed_target_positions
                ),
                "loss_backend": (
                    "triton_forward_torch_backward_selected_lm_head_rows"
                    if triton_forward_training
                    else "torch_autograd_selected_lm_head_rows"
                ),
                "lm_head_weight_gradient_sparse": bool(
                    self.config.sampled_vocab_sparse_lm_head_gradient
                ),
                "token_embedding_gradient_sparse": bool(
                    self.config.sparse_token_embedding_gradients
                ),
                "target_contract": "builder_includes_all_targets",
                "per_batch_target_membership_cpu_sync": bool(validate_sampled_targets),
                "triton_forward_kernel_used_for_training": triton_forward_training,
                "triton_autograd_backward": triton_forward_training,
                "triton_stats_delta": sampled_vocab_stats_delta,
                "sampled_vocab_hash": sampled_vocab_hash,
            }
            merged_telemetry = {
                **telemetry,
                "vocab_loss": loss_evidence,
            }
            return {
                "logits": None,
                "state": state,
                "telemetry": merged_telemetry,
                "loss": loss,
                "loss_kind": "sampled_adaptive_vocab_cross_entropy",
                "loss_evidence": loss_evidence,
            }

        logits = self.lm_head(hidden)
        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            flat_targets,
        )
        if not bool(return_evidence):
            return {
                "logits": None,
                "state": state,
                "loss": loss,
                "loss_kind": "causal_next_token_cross_entropy",
            }
        loss_evidence = {
            "surface": "marulho_language_vocab_loss_evidence.v1",
            "loss_kind": "causal_next_token_cross_entropy",
            "full_vocab_logits_materialized": True,
            "sampled_vocab_training": False,
            "configured_sampled_vocab_size": configured_sample_count,
            "actual_sampled_vocab_size": 0,
            "model_vocab_size": int(self.config.vocab_size),
            "target_token_count": int(flat_targets.numel()),
            "loss_backend": "torch_dense_full_vocab_cross_entropy",
        }
        merged_telemetry = {
            **telemetry,
            "vocab_loss": loss_evidence,
        }
        return {
            "logits": logits,
            "state": state,
            "telemetry": merged_telemetry,
            "loss": loss,
            "loss_kind": "causal_next_token_cross_entropy",
            "loss_evidence": loss_evidence,
        }

    def next_token_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        collect_telemetry: bool = True,
        assume_no_sleeping_experts: bool = False,
        sampled_vocab_ids: torch.Tensor | None = None,
        sampled_target_positions: torch.Tensor | None = None,
        memory_candidate_ids: torch.Tensor | None = None,
        route_candidate_ids: torch.Tensor | None = None,
        return_evidence: bool = True,
    ) -> dict[str, Any]:
        result = self._forward_hidden(
            input_ids,
            collect_telemetry=collect_telemetry,
            collect_memory_evidence=return_evidence,
            assume_no_sleeping_experts=assume_no_sleeping_experts,
            memory_candidate_ids=memory_candidate_ids,
            route_candidate_ids=route_candidate_ids,
        )
        return self._next_token_loss_from_hidden(
            result["hidden"],
            target_ids,
            state=result["state"],
            telemetry=result["telemetry"],
            collect_telemetry=collect_telemetry,
            sampled_vocab_ids=sampled_vocab_ids,
            sampled_target_positions=sampled_target_positions,
            return_evidence=return_evidence,
        )

    def next_token_loss_pair(
        self,
        update_batch: LanguageBatch,
        replay_batch: LanguageBatch,
        *,
        replay_loss_weight: float,
        collect_telemetry: bool = False,
        assume_no_sleeping_experts: bool = False,
        paired_sampled_vocab_ids: torch.Tensor | None = None,
        paired_sampled_target_ids: torch.Tensor | None = None,
        paired_sampled_target_positions: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if update_batch.input_ids.ndim != replay_batch.input_ids.ndim:
            raise ValueError("paired input_ids must have the same rank")
        if tuple(update_batch.input_ids.shape[1:]) != tuple(
            replay_batch.input_ids.shape[1:]
        ):
            raise ValueError("paired input_ids must share non-batch dimensions")
        if tuple(update_batch.target_ids.shape[1:]) != tuple(
            replay_batch.target_ids.shape[1:]
        ):
            raise ValueError("paired target_ids must share non-batch dimensions")
        update_batch_size = int(update_batch.input_ids.shape[0])
        combined_input_ids = torch.cat(
            (
                update_batch.input_ids.to(self.device),
                replay_batch.input_ids.to(self.device),
            ),
            dim=0,
        )
        combined_memory_candidate_ids = _cat_optional_batch_tensor(
            update_batch.memory_candidate_ids,
            replay_batch.memory_candidate_ids,
            device=self.device,
            name="memory_candidate_ids",
        )
        combined_route_candidate_ids = _cat_optional_batch_tensor(
            update_batch.route_candidate_ids,
            replay_batch.route_candidate_ids,
            device=self.device,
            name="route_candidate_ids",
        )
        result = self._forward_hidden(
            combined_input_ids,
            collect_telemetry=collect_telemetry,
            collect_memory_evidence=False,
            assume_no_sleeping_experts=assume_no_sleeping_experts,
            memory_candidate_ids=combined_memory_candidate_ids,
            route_candidate_ids=combined_route_candidate_ids,
        )
        hidden = result["hidden"]
        update_hidden = hidden[:update_batch_size]
        replay_hidden = hidden[update_batch_size:]
        configured_sample_count = int(self.config.sampled_vocab_size)
        use_paired_sampled_vocab_loss = (
            configured_sample_count > 0
            and configured_sample_count < int(self.config.vocab_size)
            and paired_sampled_vocab_ids is not None
        )
        if bool(use_paired_sampled_vocab_loss):
            if paired_sampled_target_positions is None:
                raise ValueError(
                    "paired sampled-vocab loss requires sampled target positions"
                )
            flat_hidden = hidden.reshape(-1, int(self.config.state_dim))
            update_token_count = int(update_batch.target_ids.numel())
            if paired_sampled_target_ids is None:
                update_targets = update_batch.target_ids.to(
                    device=flat_hidden.device,
                    dtype=torch.long,
                ).reshape(-1)
                replay_targets = replay_batch.target_ids.to(
                    device=flat_hidden.device,
                    dtype=torch.long,
                ).reshape(-1)
                combined_targets = torch.cat((update_targets, replay_targets), dim=0)
            else:
                combined_targets = paired_sampled_target_ids.to(
                    device=flat_hidden.device,
                    dtype=torch.long,
                ).reshape(-1)
            loss, update_loss, replay_loss = language_sampled_vocab_cross_entropy_pair(
                flat_hidden,
                combined_targets,
                paired_sampled_vocab_ids.to(
                    device=flat_hidden.device,
                    dtype=torch.long,
                ).reshape(-1),
                self.lm_head.weight,
                self.lm_head.bias,
                update_token_count=update_token_count,
                replay_loss_weight=float(replay_loss_weight),
                prefer_triton=True,
                validate_targets=True,
                sparse_weight_gradient=bool(
                    self.config.sampled_vocab_sparse_lm_head_gradient
                ),
                sampled_target_positions=paired_sampled_target_positions.to(
                    device=flat_hidden.device,
                    dtype=torch.long,
                ).reshape(-1),
            )
            loss_kind = "paired_sampled_adaptive_vocab_cross_entropy"
        else:
            update_result = self._next_token_loss_from_hidden(
                update_hidden,
                update_batch.target_ids,
                state=None,
                telemetry={},
                collect_telemetry=False,
                sampled_vocab_ids=update_batch.sampled_vocab_ids,
                sampled_target_positions=update_batch.sampled_target_positions,
                return_evidence=False,
            )
            replay_result = self._next_token_loss_from_hidden(
                replay_hidden,
                replay_batch.target_ids,
                state=None,
                telemetry={},
                collect_telemetry=False,
                sampled_vocab_ids=replay_batch.sampled_vocab_ids,
                sampled_target_positions=replay_batch.sampled_target_positions,
                return_evidence=False,
            )
            update_loss = update_result["loss"]
            replay_loss = replay_result["loss"]
            loss = update_loss + float(replay_loss_weight) * replay_loss
            loss_kind = "paired_update_replay_next_token_loss"
        return {
            "logits": None,
            "state": result["state"],
            "loss": loss,
            "update_loss": update_loss,
            "replay_loss": replay_loss,
            "loss_kind": loss_kind,
            "paired_forward": True,
            "paired_sampled_vocab_loss_fused": bool(use_paired_sampled_vocab_loss),
        }

    def forward_step(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        assume_no_sleeping_experts: bool = False,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        if input_ids.ndim == 1:
            token_ids = input_ids.view(-1)
        elif input_ids.ndim == 2 and input_ids.shape[1] == 1:
            token_ids = input_ids[:, 0]
        else:
            raise ValueError("forward_step expects token ids shaped [batch] or [batch, 1]")
        token_ids = token_ids.to(self.device)
        embeddings = self.token_embedding(token_ids)
        hidden, next_state, telemetry = self.state_block.step(
            embeddings,
            state,
            collect_telemetry=collect_telemetry,
        )
        hidden_sequence, memory_telemetry = self._apply_memory_slots(
            hidden.unsqueeze(1),
            token_ids.view(-1, 1),
            collect_telemetry=collect_telemetry,
        )
        route_candidates = self._language_route_candidates(
            token_ids.view(-1, 1),
            assume_no_sleeping_experts=assume_no_sleeping_experts,
        )
        hidden, routing_telemetry = self.routed_experts(
            hidden_sequence,
            route_candidates,
            collect_telemetry=collect_telemetry,
            assume_no_sleeping_experts=assume_no_sleeping_experts,
        )
        logits = self._lm_head_logits(
            hidden[:, 0, :],
            decode_vocab_only=decode_vocab_only,
        )
        telemetry = {
            **telemetry,
            "active_language_path": self.config.active_language_path,
            "external_llm_used": False,
            "owned_by_marulho": True,
            "vocab_size": self.config.vocab_size,
            "memory": memory_telemetry,
            "routing": routing_telemetry,
        }
        if bool(decode_vocab_only):
            telemetry["generation_decode"] = self.generation_decode_policy()
        return {
            "logits": logits.unsqueeze(1),
            "state": next_state,
            "telemetry": telemetry,
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
        repetition_penalty = max(1.0, float(repetition_penalty))
        no_repeat_ngram_size = max(0, int(no_repeat_ngram_size))
        was_training = self.training
        try:
            self.eval()
            if prompt_ids.ndim == 1:
                prompt = prompt_ids.unsqueeze(0)
            elif prompt_ids.ndim == 2:
                prompt = prompt_ids
            else:
                raise ValueError("prompt_ids must be [time] or [batch, time]")
            generated = prompt.to(self.device)
            state: Mapping[str, torch.Tensor] | None = None
            assume_no_sleeping = (
                self.routed_experts.enabled
                and not bool(
                    self.routed_experts.sleeping_expert_mask.detach().any().cpu().item()
                )
            )
            result = self.forward(
                generated,
                state,
                collect_telemetry=False,
                assume_no_sleeping_experts=assume_no_sleeping,
                decode_vocab_only=True,
            )
            state = result["state"]
            next_logits = result["logits"][:, -1, :]
            new_token_count = 0
            repetition_penalty_adjusted_token_count = 0
            no_repeat_ngram_banned_token_count = 0
            decode_control_fallback_count = 0
            for _ in range(max(0, int(max_new_tokens))):
                decode_logits, decode_control = _apply_generation_decode_controls(
                    next_logits,
                    generated,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )
                repetition_penalty_adjusted_token_count += int(
                    decode_control["repetition_penalty_adjusted_token_count"]
                )
                no_repeat_ngram_banned_token_count += int(
                    decode_control["no_repeat_ngram_banned_token_count"]
                )
                decode_control_fallback_count += int(
                    decode_control["decode_control_fallback_count"]
                )
                next_id = torch.argmax(decode_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_id], dim=1)
                new_token_count += 1
                if eos_id is not None and bool(torch.all(next_id == int(eos_id)).item()):
                    break
                result = self.forward_step(
                    next_id,
                    state,
                    collect_telemetry=False,
                    assume_no_sleeping_experts=assume_no_sleeping,
                    decode_vocab_only=True,
                )
                state = result["state"]
                next_logits = result["logits"][:, -1, :]
            generation_decode = self.generation_decode_policy(
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            generation_decode.update(
                {
                    "repetition_penalty_adjusted_token_count": int(
                        repetition_penalty_adjusted_token_count
                    ),
                    "no_repeat_ngram_banned_token_count": int(
                        no_repeat_ngram_banned_token_count
                    ),
                    "decode_control_fallback_count": int(
                        decode_control_fallback_count
                    ),
                }
            )
            return {
                "surface": "marulho_language_generation.v1",
                "generated_ids": generated.detach().cpu(),
                "new_token_count": new_token_count,
                "active_language_path": self.config.active_language_path,
                "external_llm_used": False,
                "owned_by_marulho": True,
                "loads_external_checkpoint": False,
                "generation_decode": generation_decode,
            }
        finally:
            if was_training:
                self.train()
            else:
                self.eval()


def build_language_model_splits(
    texts: Sequence[str],
    tokenizer: ByteLevelLanguageTokenizer,
    *,
    sequence_length: int,
    eval_fraction: float = 0.2,
    stride: int | None = None,
    batch_size: int = 1,
    device: torch.device | str | None = None,
) -> LanguageSplit:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    batch_size = max(1, int(batch_size))
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
    if len(windows) == 1:
        train_windows = windows
        eval_windows = windows
    else:
        eval_count = max(
            1,
            min(len(windows) - 1, math.ceil(len(windows) * float(eval_fraction))),
        )
        split_index = len(windows) - eval_count
        train_windows = windows[:split_index]
        eval_windows = windows[split_index:]

    def _pack(window_rows: Sequence[Sequence[int]]) -> tuple[LanguageBatch, ...]:
        packed: list[LanguageBatch] = []
        for offset in range(0, len(window_rows), batch_size):
            chunk = window_rows[offset : offset + batch_size]
            packed.append(
                LanguageBatch(
                    input_ids=torch.tensor(
                        [window[:-1] for window in chunk],
                        dtype=torch.long,
                        device=target_device,
                    ),
                    target_ids=torch.tensor(
                        [window[1:] for window in chunk],
                        dtype=torch.long,
                        device=target_device,
                    ),
                )
            )
        return tuple(packed)

    train_batches = _pack(train_windows)
    eval_batches = _pack(eval_windows)
    report = {
        "surface": "marulho_language_train_eval_split.v1",
        "owned_by_marulho": True,
        "external_dependency": False,
        "sequence_length": int(sequence_length),
        "stride": int(step),
        "batch_size": int(batch_size),
        "source_text_count": len(texts),
        "window_count": len(windows),
        "train_window_count": len(train_windows),
        "eval_window_count": len(eval_windows),
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
    was_training = model.training
    try:
        model.eval()
        cuda_synchronized_before_evaluation_start = False
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
            cuda_synchronized_before_evaluation_start = True
        started = time.perf_counter()
        total_loss_tensor: torch.Tensor | None = None
        total_tokens = 0
        last_telemetry: dict[str, Any] = {}
        caller_device_transfer_calls = 0
        evidence_probe_batch_tokens = 0
        assume_no_sleeping = (
            model.routed_experts.enabled
            and not bool(
                model.routed_experts.sleeping_expert_mask.detach().any().cpu().item()
            )
        )
        for batch_index, batch in enumerate(batches):
            collect_batch_evidence = batch_index == len(batches) - 1
            input_ids = batch.input_ids
            if input_ids.device != model.device:
                input_ids = input_ids.to(model.device)
                caller_device_transfer_calls += 1
            target_ids = batch.target_ids
            if target_ids.device != model.device:
                target_ids = target_ids.to(model.device)
                caller_device_transfer_calls += 1
            result = model.next_token_loss(
                input_ids,
                target_ids,
                collect_telemetry=collect_batch_evidence,
                assume_no_sleeping_experts=assume_no_sleeping,
                sampled_vocab_ids=batch.sampled_vocab_ids,
                sampled_target_positions=batch.sampled_target_positions,
                memory_candidate_ids=batch.memory_candidate_ids,
                route_candidate_ids=batch.route_candidate_ids,
                return_evidence=collect_batch_evidence,
            )
            token_count = int(batch.target_ids.numel())
            detached_weighted_loss = result["loss"].detach() * token_count
            total_loss_tensor = (
                detached_weighted_loss
                if total_loss_tensor is None
                else total_loss_tensor + detached_weighted_loss
            )
            total_tokens += token_count
            if collect_batch_evidence:
                evidence_probe_batch_tokens = token_count
                last_telemetry = dict(result.get("telemetry") or {})
        cuda_synchronized_before_evaluation_stop = False
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
            cuda_synchronized_before_evaluation_stop = True
        elapsed_seconds = max(0.0, time.perf_counter() - started)
        if total_loss_tensor is None:
            heldout_loss = 0.0
        else:
            heldout_loss = float(
                (total_loss_tensor / max(1, total_tokens)).detach().cpu().item()
            )
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
            "elapsed_seconds": float(elapsed_seconds),
            "tokens_per_second": (
                float(total_tokens) / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
            ),
            "metric_readback_mode": "deferred_gpu_scalar_aggregation",
            "per_batch_metric_cpu_sync": False,
            "evidence_collection_mode": "last_batch_only",
            "per_batch_evidence_dict_build": False,
            "evidence_probe_batch_tokens": int(evidence_probe_batch_tokens),
            "caller_device_transfer_calls": int(caller_device_transfer_calls),
            "cuda_synchronized_before_evaluation_start": bool(
                cuda_synchronized_before_evaluation_start
            ),
            "cuda_synchronized_before_evaluation_stop": bool(
                cuda_synchronized_before_evaluation_stop
            ),
            "device": str(model.device),
            "spike_telemetry": last_telemetry,
        }
    finally:
        if was_training:
            model.train()
        else:
            model.eval()


def language_model_checkpoint_payload(
    model: MarulhoLanguageModel,
    tokenizer: ByteLevelLanguageTokenizer,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    vocab_policy = _validate_language_checkpoint_vocab_policy(
        model.config,
        tokenizer,
    )
    return {
        "artifact_kind": "marulho_language_model_checkpoint",
        "surface": "marulho_language_model_checkpoint.v1",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "active_language_path": model.config.active_language_path,
        "config": asdict(model.config),
        "vocab_policy": vocab_policy,
        "model_state": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "tokenizer": tokenizer.state_dict(),
        "tokenizer_hash": tokenizer.vocabulary_hash(),
        "metadata": dict(metadata or {}),
    }


def _validate_language_checkpoint_vocab_policy(
    config: LanguageModelConfig,
    tokenizer: ByteLevelLanguageTokenizer,
) -> dict[str, Any]:
    model_vocab_size = int(config.vocab_size)
    tokenizer_vocab_size = int(tokenizer.vocab_size)
    generation_vocab_size = (
        int(config.generation_vocab_size)
        if int(config.generation_vocab_size) > 0
        else model_vocab_size
    )
    if model_vocab_size < tokenizer_vocab_size:
        raise ValueError("Language model checkpoint vocab size is smaller than tokenizer state")
    if generation_vocab_size > model_vocab_size:
        raise ValueError("Language model generation vocab size exceeds model vocab size")
    padded_vocab_rows = max(0, model_vocab_size - tokenizer_vocab_size)
    if padded_vocab_rows > 0 and generation_vocab_size != tokenizer_vocab_size:
        raise ValueError(
            "Padded-vocab language checkpoints must set generation_vocab_size "
            "to the tokenizer vocab size"
        )
    return {
        "surface": "marulho_language_checkpoint_vocab_policy.v1",
        "model_vocab_size": model_vocab_size,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "generation_vocab_size": generation_vocab_size,
        "padded_vocab_rows": padded_vocab_rows,
        "padded_vocab_decode_policy": (
            "limit_generation_to_tokenizer_vocab_rows"
            if padded_vocab_rows > 0
            else "full_tokenizer_vocab_generation"
        ),
        "checkpoint_load_requires_decode_policy": bool(padded_vocab_rows > 0),
    }


def load_language_model_state(
    model: MarulhoLanguageModel,
    model_state: Mapping[str, torch.Tensor],
) -> None:
    result = model.load_state_dict(dict(model_state), strict=False)
    allowed_missing = {"routed_experts.sleeping_expert_mask"}
    unexpected = list(result.unexpected_keys)
    missing = [
        key
        for key in result.missing_keys
        if key not in allowed_missing
    ]
    if unexpected or missing:
        raise RuntimeError(
            "Language model checkpoint state does not match current model: "
            f"missing={missing}, unexpected={unexpected}"
        )


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
    _validate_language_checkpoint_vocab_policy(config, tokenizer)
    model = MarulhoLanguageModel(config)
    load_language_model_state(model, payload["model_state"])
    return model, tokenizer, dict(payload.get("metadata") or {})
