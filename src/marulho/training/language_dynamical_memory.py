"""Causal Transformer hybrid with gated multiscale dynamical memory."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel
from marulho.training.language_transformer import (
    MarulhoTransformerBlock,
    TransformerRMSNorm,
)


DYNAMICAL_MEMORY_MODES = (
    "memory_off",
    "single_scale",
    "multiscale_always",
    "multiscale_random",
    "multiscale_learned",
)


@dataclass(frozen=True)
class DynamicalMemoryConfig:
    vocab_size: int
    width: int = 512
    layers: int = 4
    attention_heads: int = 8
    hidden_width: int = 1920
    context_length: int = 72
    memory_after_layer: int = 2
    memory_bank_count: int = 4
    memory_bank_width: int = 128
    memory_decays: tuple[float, ...] = (0.50, 0.875, 0.96875, 0.9921875)
    single_scale_decay: float = 0.875
    residual_alpha_init: float = 0.05
    random_control_seed: int = 2718
    mode: str = "multiscale_learned"
    active_language_path: str = "marulho_multiscale_dynamical_memory_v7"


def _validate_config(config: DynamicalMemoryConfig) -> None:
    if int(config.vocab_size) <= 1:
        raise ValueError("vocab_size must exceed one")
    if int(config.width) <= 0 or int(config.layers) < 1:
        raise ValueError("width and layers must be positive")
    if int(config.attention_heads) < 1 or int(config.width) % int(
        config.attention_heads
    ) != 0:
        raise ValueError("width must be divisible by attention_heads")
    if (int(config.width) // int(config.attention_heads)) % 2 != 0:
        raise ValueError("attention head dimension must be even")
    if int(config.hidden_width) < int(config.width):
        raise ValueError("hidden_width must be at least width")
    if int(config.context_length) < 2:
        raise ValueError("context_length must be at least two")
    if not 1 <= int(config.memory_after_layer) <= int(config.layers):
        raise ValueError("memory_after_layer must select an existing layer boundary")
    if int(config.memory_bank_count) < 1 or int(config.memory_bank_width) < 2:
        raise ValueError("memory bank dimensions must be positive")
    if int(config.memory_bank_width) % 2 != 0:
        raise ValueError("memory_bank_width must be even for planar rotations")
    if int(config.memory_bank_count) * int(config.memory_bank_width) != int(
        config.width
    ):
        raise ValueError("total memory width must equal Transformer width")
    if len(config.memory_decays) != int(config.memory_bank_count):
        raise ValueError("memory_decays must provide one decay per bank")
    if any(not 0.0 < float(value) < 1.0 for value in config.memory_decays):
        raise ValueError("memory decays must lie in (0, 1)")
    if not 0.0 < float(config.single_scale_decay) < 1.0:
        raise ValueError("single_scale_decay must lie in (0, 1)")
    if not 0.0 < float(config.residual_alpha_init) < 1.0:
        raise ValueError("residual_alpha_init must lie in (0, 1)")
    if str(config.mode) not in DYNAMICAL_MEMORY_MODES:
        raise ValueError(f"mode must be one of {DYNAMICAL_MEMORY_MODES}")


class MarulhoGatedDynamicalMemory(nn.Module):
    """Stable rotating reservoirs with content-controlled writes."""

    surface = "marulho_gated_dynamical_memory.v1"

    def __init__(self, config: DynamicalMemoryConfig) -> None:
        super().__init__()
        self.config = config
        self.width = int(config.width)
        self.bank_count = int(config.memory_bank_count)
        self.bank_width = int(config.memory_bank_width)
        self.context_length = int(config.context_length)
        self.base_scale = 1.0 / math.sqrt(float(self.width))
        self.input_norm = TransformerRMSNorm(self.width)
        self.candidate = nn.Linear(self.width, self.width, bias=False)
        self.write_gate = nn.Linear(self.width, self.width, bias=False)
        self.output = nn.Linear(self.width, self.width, bias=False)
        self.residual_alpha = nn.Parameter(
            torch.full((self.width,), self.base_scale, dtype=torch.float32)
        )

        decays = torch.tensor(config.memory_decays, dtype=torch.float32).view(
            1,
            self.bank_count,
            1,
        )
        single = torch.full_like(decays, float(config.single_scale_decay))
        input_scales = torch.sqrt((1.0 - decays.square()).clamp_min(1.0e-8))
        single_input_scales = torch.sqrt(
            (1.0 - single.square()).clamp_min(1.0e-8)
        )
        generator = torch.Generator(device="cpu").manual_seed(
            int(config.random_control_seed)
        )
        angle = torch.rand(
            self.bank_count,
            self.bank_width // 2,
            generator=generator,
            dtype=torch.float32,
        )
        angle = angle * math.pi
        random_pattern = torch.rand(
            self.context_length,
            self.bank_count,
            self.bank_width,
            generator=generator,
            dtype=torch.float32,
        )
        self.register_buffer("multiscale_decay", decays, persistent=True)
        self.register_buffer("single_decay", single, persistent=True)
        self.register_buffer("multiscale_input_scale", input_scales, persistent=True)
        self.register_buffer("single_input_scale", single_input_scales, persistent=True)
        self.register_buffer("rotation_cos", torch.cos(angle), persistent=True)
        self.register_buffer("rotation_sin", torch.sin(angle), persistent=True)
        lags = torch.arange(self.context_length + 1, dtype=torch.float32)
        phase = angle.unsqueeze(-1) * lags.view(1, 1, -1)
        multiscale_magnitude = decays.view(self.bank_count, 1, 1).pow(
            lags.view(1, 1, -1)
        )
        single_magnitude = single.view(self.bank_count, 1, 1).pow(
            lags.view(1, 1, -1)
        )
        self.register_buffer(
            "multiscale_power_real",
            multiscale_magnitude * torch.cos(phase),
            persistent=True,
        )
        self.register_buffer(
            "multiscale_power_imag",
            multiscale_magnitude * torch.sin(phase),
            persistent=True,
        )
        self.register_buffer(
            "single_power_real",
            single_magnitude * torch.cos(phase),
            persistent=True,
        )
        self.register_buffer(
            "single_power_imag",
            single_magnitude * torch.sin(phase),
            persistent=True,
        )
        self.register_buffer("random_gate_pattern", random_pattern, persistent=True)
        for name in DYNAMICAL_MEMORY_MODES:
            self.register_buffer(
                f"mode_{name}",
                torch.zeros((), dtype=torch.float32),
                persistent=True,
            )
        self._mode_name = ""
        self.set_mode(config.mode)

    @torch.no_grad()
    def set_mode(self, mode: str) -> None:
        selected = str(mode)
        if selected not in DYNAMICAL_MEMORY_MODES:
            raise ValueError(f"mode must be one of {DYNAMICAL_MEMORY_MODES}")
        for name in DYNAMICAL_MEMORY_MODES:
            getattr(self, f"mode_{name}").fill_(float(name == selected))
        self._mode_name = selected

    def _rotate(self, state: torch.Tensor) -> torch.Tensor:
        pairs = state.view(
            int(state.shape[0]),
            self.bank_count,
            self.bank_width // 2,
            2,
        )
        first = pairs[..., 0]
        second = pairs[..., 1]
        cosine = self.rotation_cos.to(device=state.device, dtype=state.dtype)
        sine = self.rotation_sin.to(device=state.device, dtype=state.dtype)
        rotated_first = cosine * first - sine * second
        rotated_second = sine * first + cosine * second
        return torch.stack((rotated_first, rotated_second), dim=-1).reshape_as(state)

    def _parallel_recurrence(
        self,
        update: torch.Tensor,
        initial_state: torch.Tensor,
        *,
        single_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the fixed complex-diagonal recurrence as causal convolutions."""

        batch_size, time_steps, _banks, _width = update.shape
        if int(time_steps) == 1:
            decay = (
                (1.0 - single_weight) * self.multiscale_decay
                + single_weight * self.single_decay
            ).to(device=update.device, dtype=update.dtype)
            current = decay * self._rotate(initial_state) + update[:, 0]
            return current.unsqueeze(1)

        multiscale_weight = 1.0 - single_weight
        power_real = (
            multiscale_weight * self.multiscale_power_real
            + single_weight * self.single_power_real
        ).to(device=update.device, dtype=update.dtype)
        power_imag = (
            multiscale_weight * self.multiscale_power_imag
            + single_weight * self.single_power_imag
        ).to(device=update.device, dtype=update.dtype)
        channels = self.bank_count * (self.bank_width // 2)
        pairs = update.view(
            int(batch_size),
            int(time_steps),
            self.bank_count,
            self.bank_width // 2,
            2,
        )
        update_real = pairs[..., 0].permute(0, 2, 3, 1).reshape(
            int(batch_size), channels, int(time_steps)
        )
        update_imag = pairs[..., 1].permute(0, 2, 3, 1).reshape(
            int(batch_size), channels, int(time_steps)
        )
        kernel_real = power_real[..., : int(time_steps)].flip(-1).reshape(
            channels,
            1,
            int(time_steps),
        )
        kernel_imag = power_imag[..., : int(time_steps)].flip(-1).reshape(
            channels,
            1,
            int(time_steps),
        )
        padding = int(time_steps) - 1
        real_real = F.conv1d(
            update_real,
            kernel_real,
            padding=padding,
            groups=channels,
        )[..., : int(time_steps)]
        imag_imag = F.conv1d(
            update_imag,
            kernel_imag,
            padding=padding,
            groups=channels,
        )[..., : int(time_steps)]
        real_imag = F.conv1d(
            update_real,
            kernel_imag,
            padding=padding,
            groups=channels,
        )[..., : int(time_steps)]
        imag_real = F.conv1d(
            update_imag,
            kernel_real,
            padding=padding,
            groups=channels,
        )[..., : int(time_steps)]
        output_real = real_real - imag_imag
        output_imag = real_imag + imag_real

        initial_pairs = initial_state.view(
            int(batch_size),
            self.bank_count,
            self.bank_width // 2,
            2,
        )
        initial_real = initial_pairs[..., 0].unsqueeze(-1)
        initial_imag = initial_pairs[..., 1].unsqueeze(-1)
        initial_power_real = power_real[..., 1 : int(time_steps) + 1].unsqueeze(0)
        initial_power_imag = power_imag[..., 1 : int(time_steps) + 1].unsqueeze(0)
        initial_output_real = (
            initial_real * initial_power_real - initial_imag * initial_power_imag
        )
        initial_output_imag = (
            initial_real * initial_power_imag + initial_imag * initial_power_real
        )
        output_real = output_real + initial_output_real.reshape(
            int(batch_size), channels, int(time_steps)
        )
        output_imag = output_imag + initial_output_imag.reshape(
            int(batch_size), channels, int(time_steps)
        )
        output_pairs = torch.stack((output_real, output_imag), dim=-1)
        return output_pairs.view(
            int(batch_size),
            self.bank_count,
            self.bank_width // 2,
            int(time_steps),
            2,
        ).permute(0, 3, 1, 2, 4).reshape(
            int(batch_size),
            int(time_steps),
            self.bank_count,
            self.bank_width,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        state: torch.Tensor,
        *,
        position_offset: torch.Tensor,
        collect_telemetry: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if hidden.ndim != 3:
            raise ValueError("Dynamical memory expects [batch, time, width]")
        batch_size, time_steps, _width = hidden.shape
        runtime = self.input_norm(hidden)
        candidate = torch.tanh(self.candidate(runtime)).view(
            int(batch_size),
            int(time_steps),
            self.bank_count,
            self.bank_width,
        )
        learned_gate = torch.sigmoid(self.write_gate(runtime)).view_as(candidate)
        positions = torch.arange(int(time_steps), device=hidden.device)
        positions = positions + position_offset.to(device=hidden.device)
        random_gate = self.random_gate_pattern.index_select(
            0,
            positions.remainder(self.context_length).to(dtype=torch.long),
        )
        random_gate = random_gate.to(device=hidden.device, dtype=hidden.dtype)
        random_gate = random_gate.unsqueeze(0).expand(int(batch_size), -1, -1, -1)
        learned_weight = (
            self.mode_multiscale_learned + self.mode_single_scale
        ).to(device=hidden.device, dtype=hidden.dtype)
        always_weight = self.mode_multiscale_always.to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        random_weight = self.mode_multiscale_random.to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        gate = (
            learned_weight * learned_gate
            + always_weight * torch.ones_like(learned_gate)
            + random_weight * random_gate
        )
        single_weight = self.mode_single_scale.to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        multiscale_weight = 1.0 - single_weight
        input_scale = (
            multiscale_weight * self.multiscale_input_scale
            + single_weight * self.single_input_scale
        ).to(device=hidden.device, dtype=hidden.dtype)
        initial_state = state.view(int(batch_size), self.bank_count, self.bank_width)
        memory_states = self._parallel_recurrence(
            input_scale * gate * candidate,
            initial_state,
            single_weight=single_weight,
        )
        current = memory_states[:, -1]
        memory_hidden = memory_states.reshape(
            int(batch_size),
            int(time_steps),
            self.width,
        )
        branch = self.output(memory_hidden)
        alpha = (
            self.residual_alpha * (self.config.residual_alpha_init / self.base_scale)
        ).abs()
        alpha = alpha.to(device=hidden.device, dtype=hidden.dtype)
        output_enabled = (1.0 - self.mode_memory_off).to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        result = hidden + output_enabled * alpha * branch
        telemetry: dict[str, Any] = {
            "surface": self.surface,
            "telemetry_collected": bool(collect_telemetry),
            "mode": self._mode_name if collect_telemetry else "not_collected",
            "memory_bank_count": self.bank_count,
            "memory_bank_width": self.bank_width,
            "memory_decays": [float(value) for value in self.config.memory_decays],
            "single_scale_decay": float(self.config.single_scale_decay),
            "transition": "fixed_stable_planar_rotation",
            "write_gate": "content_controlled_sigmoid",
            "external_llm_used": False,
        }
        if collect_telemetry:
            gate_runtime = gate.detach().float().clamp(1.0e-6, 1.0 - 1.0e-6)
            entropy = -(
                gate_runtime * gate_runtime.log()
                + (1.0 - gate_runtime) * (1.0 - gate_runtime).log()
            )
            bank_norms = current.detach().float().norm(p=2, dim=-1).mean(dim=0)
            telemetry.update(
                {
                    "write_gate_mean": float(gate_runtime.mean().cpu()),
                    "write_gate_entropy": float(entropy.mean().cpu()),
                    "final_bank_norm_means": [
                        float(value) for value in bank_norms.cpu().tolist()
                    ],
                    "residual_alpha_mean": float(alpha.detach().float().mean().cpu()),
                }
            )
        return result, current.reshape(int(batch_size), self.width), telemetry


class MarulhoDynamicalMemoryStateBlock(nn.Module):
    """Four attention layers with a recurrent memory seam in the middle."""

    surface = "marulho_dynamical_memory_state_block.v1"

    def __init__(self, config: DynamicalMemoryConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            MarulhoTransformerBlock(
                config.width,
                attention_heads=config.attention_heads,
                context_length=config.context_length,
                mlp_ratio=float(config.hidden_width) / float(config.width),
                dropout=0.0,
            )
            for _ in range(config.layers)
        )
        self.memory = MarulhoGatedDynamicalMemory(config)
        self.output_norm = TransformerRMSNorm(config.width)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        head_dim = self.config.width // self.config.attention_heads
        state: dict[str, torch.Tensor] = {
            "position": torch.zeros((), device=device, dtype=torch.long),
            "memory_state": torch.zeros(
                int(batch_size),
                self.config.width,
                device=device,
                dtype=dtype,
            ),
        }
        for index in range(self.config.layers):
            state[f"layer_{index}_key"] = torch.empty(
                int(batch_size),
                self.config.attention_heads,
                0,
                head_dim,
                device=device,
                dtype=dtype,
            )
            state[f"layer_{index}_value"] = torch.empty_like(
                state[f"layer_{index}_key"]
            )
        return state

    def forward(
        self,
        inputs: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        if inputs.ndim != 3:
            raise ValueError("Dynamical memory state block expects [batch, time, width]")
        batch_size, time_steps, _width = inputs.shape
        if int(time_steps) > self.config.context_length and state is None:
            raise ValueError("Input exceeds dynamical memory context")
        current = (
            self.initial_state(
                int(batch_size),
                device=inputs.device,
                dtype=inputs.dtype,
            )
            if state is None
            else state
        )
        position = current["position"].to(device=inputs.device, dtype=torch.long)
        hidden = inputs
        next_state: dict[str, torch.Tensor] = {
            "position": position + int(time_steps),
        }
        memory_telemetry: dict[str, Any] = {}
        cache_tokens = 0
        for index, layer in enumerate(self.layers):
            hidden, key, value = layer(
                hidden,
                past_key=current.get(f"layer_{index}_key"),
                past_value=current.get(f"layer_{index}_value"),
                position_offset=position,
            )
            next_state[f"layer_{index}_key"] = key.detach()
            next_state[f"layer_{index}_value"] = value.detach()
            cache_tokens = int(key.shape[2])
            if index + 1 == int(self.config.memory_after_layer):
                hidden, memory_state, memory_telemetry = self.memory(
                    hidden,
                    current["memory_state"].to(
                        device=hidden.device,
                        dtype=hidden.dtype,
                    ),
                    position_offset=position,
                    collect_telemetry=collect_telemetry,
                )
                next_state["memory_state"] = memory_state.detach()
        hidden = self.output_norm(hidden)
        telemetry = {
            "surface": self.surface,
            "state_core": "transformer_with_gated_dynamical_memory",
            "telemetry_collected": bool(collect_telemetry),
            "state_dim": int(self.config.width),
            "state_layers": int(self.config.layers),
            "attention_heads": int(self.config.attention_heads),
            "context_length": int(self.config.context_length),
            "kv_cache_tokens": cache_tokens,
            "memory_after_layer": int(self.config.memory_after_layer),
            "memory": memory_telemetry,
            "normalization": "rmsnorm",
            "position_encoding": "rotary",
            "attention_backend": "torch_scaled_dot_product_attention",
            "external_llm_used": False,
        }
        return hidden, next_state, telemetry

    def step(
        self,
        inputs: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        if inputs.ndim != 2:
            raise ValueError("Dynamical memory step expects [batch, width]")
        hidden, next_state, telemetry = self.forward(
            inputs.unsqueeze(1),
            state,
            collect_telemetry=collect_telemetry,
        )
        return hidden[:, 0], next_state, telemetry


class MarulhoDynamicalMemoryLanguageModel(MarulhoLanguageModel):
    """MARULHO-owned Transformer/dynamical-memory hybrid candidate."""

    surface = "marulho_dynamical_memory_language_model.v1"

    def __init__(self, memory_config: DynamicalMemoryConfig) -> None:
        nn.Module.__init__(self)
        _validate_config(memory_config)
        self.memory_config = memory_config
        self.config = LanguageModelConfig(
            vocab_size=memory_config.vocab_size,
            embedding_dim=memory_config.width,
            state_dim=memory_config.width,
            state_layers=memory_config.layers,
            attention_heads=memory_config.attention_heads,
            transformer_context_length=memory_config.context_length,
            transformer_mlp_ratio=(
                float(memory_config.hidden_width) / float(memory_config.width)
            ),
            transformer_dropout=0.0,
            tie_embeddings=True,
            active_language_path=memory_config.active_language_path,
        )
        self.token_embedding = nn.Embedding(
            memory_config.vocab_size,
            memory_config.width,
        )
        self.state_block = MarulhoDynamicalMemoryStateBlock(memory_config)
        self.lm_head = nn.Linear(
            memory_config.width,
            memory_config.vocab_size,
            bias=False,
        )
        self.lm_head.weight = self.token_embedding.weight
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                bias = getattr(module, "bias", None)
                if isinstance(bias, torch.Tensor):
                    nn.init.zeros_(bias)

    def set_memory_mode(self, mode: str) -> None:
        self.state_block.memory.set_mode(mode)

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        result = super().forward(
            input_ids,
            state,
            collect_telemetry=collect_telemetry,
            decode_vocab_only=decode_vocab_only,
        )
        result["telemetry"] = {
            **result["telemetry"],
            "architecture": "transformer_multiscale_gated_dynamical_memory",
            "parameter_graph_shared_across_memory_controls": True,
        }
        return result
