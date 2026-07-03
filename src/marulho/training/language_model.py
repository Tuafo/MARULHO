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

from marulho.core.language_plif_triton import (
    language_plif_forward,
    language_plif_surrogate_update,
    language_plif_triton_stats,
    language_plif_triton_stats_delta,
)
from marulho.core.language_rmsnorm_triton import language_rmsnorm
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
        return language_rmsnorm(value, self.weight, eps=self.eps)


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
        outputs: list[torch.Tensor] = []
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
            outputs.append(residual_all[:, step, :] + self.state_output_proj(mixed_state))
            if bool(collect_telemetry):
                spike_sum = spike_sum + spikes.sum()
                assert active_neuron_counts is not None
                active_neuron_counts = active_neuron_counts + (spikes > 0).sum(dim=0)

        hidden = self.output_norm(torch.stack(outputs, dim=1))
        if bool(collect_telemetry):
            plif_delta = (
                language_plif_triton_stats_delta(
                    plif_stats_before,
                    language_plif_triton_stats(),
                )
                if plif_stats_before is not None
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
                "state_block_projection_mode": "batched_token_projection_recurrent_loop",
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
                "device": str(inputs.device),
            }
        else:
            telemetry = {
                "surface": "marulho_selective_spiking_state_block.v1",
                "telemetry_collected": False,
                "state_block_projection_mode": "batched_token_projection_recurrent_loop",
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
            }

        started = time.perf_counter()
        if bool(assume_no_sleeping_experts):
            sleeping_mask = None
            sleeping_count = 0
            awake_ids = torch.arange(
                self.expert_count,
                device=hidden.device,
                dtype=torch.long,
            )
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
            route_plan_source = "token_hash_candidate_bank"

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
        candidate_count = int(candidate_ids.shape[-1])
        active_count = min(self.active_expert_count, candidate_count)
        route_keys = self.route_keys[candidate_ids]
        route_bias = self.route_bias[candidate_ids]
        route_logits = (hidden.unsqueeze(-2) * route_keys).sum(dim=-1) + route_bias
        top_scores, top_positions = torch.topk(route_logits, k=active_count, dim=-1)
        top_weights = torch.softmax(top_scores, dim=-1)
        selected_expert_ids = candidate_ids.gather(dim=-1, index=top_positions)

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
        selected_second_weights = second_weights.index_select(0, selected_flat).reshape(
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
        expanded_hidden = flat_hidden.unsqueeze(1).expand(
            -1,
            active_count,
            -1,
        )
        expert_hidden = F.silu(
            torch.einsum("nkd,nkhd->nkh", expanded_hidden, selected_first_weights)
            + selected_first_biases
        )
        expert_outputs = (
            torch.einsum("nkh,nkdh->nkd", expert_hidden, selected_second_weights)
            + selected_second_biases
        )
        expert_delta = (expert_outputs * flat_weights.unsqueeze(-1)).sum(dim=1)

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
        }


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
        self.routed_experts = RoutedLanguageExpertLayer(
            config.state_dim,
            expert_count=config.expert_count,
            active_expert_count=config.active_expert_count,
            route_candidate_count=config.route_candidate_count,
            expert_hidden_dim=config.expert_hidden_dim,
        )
        self.lm_head = nn.Linear(config.state_dim, config.vocab_size)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        assume_no_sleeping_experts: bool = False,
    ) -> dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("Language model expects input_ids shaped [batch, time]")
        embeddings = self.token_embedding(input_ids.to(self.device))
        hidden, next_state, telemetry = self.state_block(
            embeddings,
            state,
            collect_telemetry=collect_telemetry,
        )
        route_candidates = self._language_route_candidates(
            input_ids.to(self.device),
            assume_no_sleeping_experts=assume_no_sleeping_experts,
        )
        hidden, routing_telemetry = self.routed_experts(
            hidden,
            route_candidates,
            collect_telemetry=collect_telemetry,
            assume_no_sleeping_experts=assume_no_sleeping_experts,
        )
        logits = self.lm_head(hidden)
        telemetry = {
            **telemetry,
            "active_language_path": self.config.active_language_path,
            "external_llm_used": False,
            "owned_by_marulho": True,
            "vocab_size": self.config.vocab_size,
            "routing": routing_telemetry,
        }
        return {
            "logits": logits,
            "state": next_state,
            "telemetry": telemetry,
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
            awake_ids = torch.arange(
                awake_count,
                device=input_ids.device,
                dtype=torch.long,
            )
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
        return awake_ids.index_select(0, candidate_positions.reshape(-1)).reshape(
            candidate_positions.shape
        )

    def next_token_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        collect_telemetry: bool = True,
        assume_no_sleeping_experts: bool = False,
    ) -> dict[str, Any]:
        result = self.forward(
            input_ids,
            collect_telemetry=collect_telemetry,
            assume_no_sleeping_experts=assume_no_sleeping_experts,
        )
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

    def forward_step(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        assume_no_sleeping_experts: bool = False,
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
        route_candidates = self._language_route_candidates(
            token_ids.view(-1, 1),
            assume_no_sleeping_experts=assume_no_sleeping_experts,
        )
        hidden, routing_telemetry = self.routed_experts(
            hidden.unsqueeze(1),
            route_candidates,
            collect_telemetry=collect_telemetry,
            assume_no_sleeping_experts=assume_no_sleeping_experts,
        )
        logits = self.lm_head(hidden[:, 0, :])
        telemetry = {
            **telemetry,
            "active_language_path": self.config.active_language_path,
            "external_llm_used": False,
            "owned_by_marulho": True,
            "vocab_size": self.config.vocab_size,
            "routing": routing_telemetry,
        }
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
        )
        state = result["state"]
        next_logits = result["logits"][:, -1, :]
        new_token_count = 0
        for _ in range(max(0, int(max_new_tokens))):
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=1)
            new_token_count += 1
            if eos_id is not None and bool(torch.all(next_id == int(eos_id)).item()):
                break
            result = self.forward_step(
                next_id,
                state,
                collect_telemetry=False,
                assume_no_sleeping_experts=assume_no_sleeping,
            )
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
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    last_telemetry: dict[str, Any] = {}
    assume_no_sleeping = (
        model.routed_experts.enabled
        and not bool(model.routed_experts.sleeping_expert_mask.detach().any().cpu().item())
    )
    for batch_index, batch in enumerate(batches):
        result = model.next_token_loss(
            batch.input_ids.to(model.device),
            batch.target_ids.to(model.device),
            collect_telemetry=batch_index == len(batches) - 1,
            assume_no_sleeping_experts=assume_no_sleeping,
        )
        token_count = int(batch.target_ids.numel())
        total_loss += float(result["loss"].detach().cpu().item()) * token_count
        total_tokens += token_count
        if batch_index == len(batches) - 1:
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
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "Language model checkpoint vocab size does not match tokenizer state"
        )
    model = MarulhoLanguageModel(config)
    load_language_model_state(model, payload["model_state"])
    return model, tokenizer, dict(payload.get("metadata") or {})
