"""All-active grouped recurrent language state for the V17 screen."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from marulho.training.language_hashed_micro_experts import (
    HashedMicroExpertConfig,
    MarulhoHashedMicroExpertBlock,
    MarulhoHashedMicroExpertLanguageModel,
    MarulhoHashedMicroExpertStateBlock,
)
from marulho.training.language_transformer import TransformerRMSNorm


RECURRENT_ARCHITECTURES = ("grouped", "dense")
RECURRENT_MODES = ("off", "local", "recurrent")


@dataclass(frozen=True)
class GroupedRecurrentConfig:
    architecture: str = "grouped"
    mode: str = "recurrent"
    memory_layer_index: int = 1
    group_count: int = 8
    group_width: int = 32
    active_language_path: str = "marulho_grouped_recurrent_state_v17"

    @property
    def total_state_width(self) -> int:
        return int(self.group_count) * int(self.group_width)


def _validate_recurrent_config(
    config: GroupedRecurrentConfig,
    *,
    model_width: int,
    model_layers: int,
) -> None:
    if config.architecture not in RECURRENT_ARCHITECTURES:
        raise ValueError(f"architecture must be one of {RECURRENT_ARCHITECTURES}")
    if config.mode not in RECURRENT_MODES:
        raise ValueError(f"mode must be one of {RECURRENT_MODES}")
    if not 0 <= int(config.memory_layer_index) < int(model_layers) - 1:
        raise ValueError("memory_layer_index must precede a later model layer")
    if int(config.group_count) < 2:
        raise ValueError("group_count must be at least two")
    if int(config.group_width) < 1:
        raise ValueError("group_width must be positive")
    if int(config.total_state_width) > 2 * int(model_width):
        raise ValueError("recurrent state width is unreasonably large")
    if not str(config.active_language_path).strip():
        raise ValueError("active_language_path is required")


class MarulhoGroupedRecurrentOrgan(nn.Module):
    """Block-diagonal or dense GRU state with a shared residual readout."""

    def __init__(self, width: int, config: GroupedRecurrentConfig) -> None:
        super().__init__()
        self.width = int(width)
        self.architecture = str(config.architecture)
        self.group_count = int(config.group_count)
        self.group_width = int(config.group_width)
        self.total_state_width = int(config.total_state_width)
        self._mode_name = str(config.mode)
        self.norm = TransformerRMSNorm(self.width)
        if self.architecture == "grouped":
            self.groups = nn.ModuleList(
                nn.GRU(
                    input_size=self.width,
                    hidden_size=self.group_width,
                    batch_first=True,
                )
                for _ in range(self.group_count)
            )
            self.dense: nn.GRU | None = None
        else:
            self.groups = nn.ModuleList()
            self.dense = nn.GRU(
                input_size=self.width,
                hidden_size=self.total_state_width,
                batch_first=True,
            )
        self.output = nn.Linear(self.total_state_width, self.width, bias=False)
        nn.init.zeros_(self.output.weight)

    def set_mode(self, mode: str) -> None:
        if mode not in RECURRENT_MODES:
            raise ValueError(f"mode must be one of {RECURRENT_MODES}")
        self._mode_name = str(mode)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        if self.architecture == "grouped":
            return {
                f"recurrent_group_{index}": torch.zeros(
                    int(batch_size),
                    self.group_width,
                    device=device,
                    dtype=dtype,
                )
                for index in range(self.group_count)
            }
        return {
            "dense_recurrent_state": torch.zeros(
                int(batch_size),
                self.total_state_width,
                device=device,
                dtype=dtype,
            )
        }

    def _passthrough_state(
        self,
        hidden: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        if state is None:
            return self.initial_state(
                int(hidden.shape[0]),
                device=hidden.device,
                dtype=hidden.dtype,
            )
        keys = (
            [f"recurrent_group_{index}" for index in range(self.group_count)]
            if self.architecture == "grouped"
            else ["dense_recurrent_state"]
        )
        return {key: state[key].detach() for key in keys}

    def _grouped_features(
        self,
        normalized: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None,
        *,
        local: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, time_steps, _ = normalized.shape
        outputs: list[torch.Tensor] = []
        next_state: dict[str, torch.Tensor] = {}
        for index, group in enumerate(self.groups):
            if local:
                flattened = normalized.reshape(
                    int(batch_size) * int(time_steps), 1, self.width
                )
                observed, _final = group(flattened)
                outputs.append(
                    observed[:, 0].reshape(
                        int(batch_size), int(time_steps), self.group_width
                    )
                )
                next_state[f"recurrent_group_{index}"] = normalized.new_zeros(
                    int(batch_size), self.group_width
                )
            else:
                previous = (
                    None
                    if state is None
                    else state.get(f"recurrent_group_{index}")
                )
                initial = None if previous is None else previous.unsqueeze(0)
                observed, final = group(normalized, initial)
                outputs.append(observed)
                next_state[f"recurrent_group_{index}"] = final[0].detach()
        return torch.cat(outputs, dim=-1), next_state

    def _dense_features(
        self,
        normalized: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None,
        *,
        local: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert self.dense is not None
        batch_size, time_steps, _ = normalized.shape
        if local:
            flattened = normalized.reshape(
                int(batch_size) * int(time_steps), 1, self.width
            )
            observed, _final = self.dense(flattened)
            features = observed[:, 0].reshape(
                int(batch_size), int(time_steps), self.total_state_width
            )
            final_state = normalized.new_zeros(
                int(batch_size), self.total_state_width
            )
        else:
            previous = None if state is None else state.get("dense_recurrent_state")
            initial = None if previous is None else previous.unsqueeze(0)
            features, final = self.dense(normalized, initial)
            final_state = final[0].detach()
        return features, {"dense_recurrent_state": final_state}

    def forward(
        self,
        hidden: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if hidden.ndim != 3:
            raise ValueError("grouped recurrent organ expects [batch,time,width]")
        if self._mode_name == "off":
            return hidden, self._passthrough_state(hidden, state)
        normalized = self.norm(hidden)
        local = self._mode_name == "local"
        if self.architecture == "grouped":
            features, next_state = self._grouped_features(
                normalized, state, local=local
            )
        else:
            features, next_state = self._dense_features(
                normalized, state, local=local
            )
        return hidden + self.output(features), next_state

    def state_bytes(self, batch_size: int, *, element_size: int = 4) -> int:
        return (
            int(batch_size) * self.total_state_width * int(element_size)
        )

    def theoretical_recurrent_multiplies_per_token(self) -> int:
        if self._mode_name == "off":
            return 0
        if self.architecture == "grouped":
            recurrence = self.group_count * 3 * (
                self.width * self.group_width
                + self.group_width * self.group_width
            )
        else:
            recurrence = 3 * (
                self.width * self.total_state_width
                + self.total_state_width * self.total_state_width
            )
        return recurrence + self.total_state_width * self.width

    @staticmethod
    def _geometry(features: torch.Tensor) -> dict[str, Any]:
        matrix = features.detach().float().reshape(-1, int(features.shape[-1])).cpu()
        if int(matrix.shape[0]) > 4096:
            positions = torch.linspace(
                0, int(matrix.shape[0]) - 1, steps=4096
            ).long()
            matrix = matrix.index_select(0, positions)
        centered = matrix - matrix.mean(dim=0, keepdim=True)
        singular = torch.linalg.svdvals(centered)
        variance = singular.square()
        probability = variance / variance.sum().clamp_min(1.0e-12)
        participation = variance.sum().square() / variance.square().sum().clamp_min(
            1.0e-12
        )
        effective = torch.exp(
            -(probability * probability.clamp_min(1.0e-12).log()).sum()
        )
        return {
            "sample_count": int(matrix.shape[0]),
            "ambient_dimension": int(matrix.shape[1]),
            "matrix_rank": int(torch.linalg.matrix_rank(centered)),
            "participation_ratio": float(participation),
            "effective_rank": float(effective),
            "mean_state_norm": float(matrix.norm(dim=-1).mean()),
            "promotion_metric": False,
        }

    @torch.no_grad()
    def diagnostic_report(self, hidden: torch.Tensor) -> dict[str, Any]:
        if hidden.ndim != 3:
            raise ValueError("recurrent diagnostic expects [batch,time,width]")
        captured: dict[str, torch.Tensor] = {}

        def capture(_module, inputs) -> None:
            captured["features"] = inputs[0].detach()

        handle = self.output.register_forward_pre_hook(capture)
        try:
            transformed, final_state = self.forward(hidden, None)
        finally:
            handle.remove()
        common: dict[str, Any] = {
            "surface": "marulho_grouped_recurrent_diagnostic.v1",
            "architecture": self.architecture,
            "mode": self._mode_name,
            "state_bytes": self.state_bytes(
                int(hidden.shape[0]), element_size=hidden.element_size()
            ),
            "theoretical_recurrent_multiplies_per_token": (
                self.theoretical_recurrent_multiplies_per_token()
            ),
            "residual_root_mean_square": float(
                (transformed - hidden).float().square().mean().sqrt().cpu()
            ),
            "write_policy_uses_labels": False,
            "promotion_metric": False,
            "external_llm_used": False,
        }
        if self._mode_name == "off":
            return {
                **common,
                "state_geometry": None,
                "mean_absolute_inter_group_trajectory_cosine": None,
                "group_residual_root_mean_squares": [],
                "final_state_norm": 0.0,
                "state_perturbation_gain": 0.0,
            }
        features = captured["features"]
        geometry = self._geometry(features)
        group_cosine: float | None = None
        contributions: list[float] = []
        if self.architecture == "grouped":
            grouped = features.reshape(
                int(features.shape[0]),
                int(features.shape[1]),
                self.group_count,
                self.group_width,
            )
            trajectories = grouped.permute(2, 0, 1, 3).reshape(
                self.group_count, -1
            )
            normalized = F.normalize(trajectories.float(), dim=-1)
            gram = normalized @ normalized.T
            group_cosine = float(
                (
                    (gram.abs().sum() - gram.diagonal().abs().sum())
                    / (self.group_count * (self.group_count - 1))
                ).cpu()
            )
            for index in range(self.group_count):
                start = index * self.group_width
                end = start + self.group_width
                contribution = F.linear(
                    grouped[:, :, index],
                    self.output.weight[:, start:end],
                )
                contributions.append(
                    float(contribution.float().square().mean().sqrt().cpu())
                )
        else:
            contributions.append(
                float(
                    self.output(features).float().square().mean().sqrt().cpu()
                )
            )
        final_vector = torch.cat(list(final_state.values()), dim=-1)
        perturbation_gain = 0.0
        if self._mode_name == "recurrent":
            perturbed = hidden.clone()
            perturbation_size = 1.0e-3
            perturbed[0, 0, 0] += perturbation_size
            _changed, perturbed_state = self.forward(perturbed, None)
            changed_vector = torch.cat(list(perturbed_state.values()), dim=-1)
            perturbation_gain = float(
                ((changed_vector - final_vector).float().norm() / perturbation_size).cpu()
            )
        return {
            **common,
            "state_geometry": geometry,
            "mean_absolute_inter_group_trajectory_cosine": group_cosine,
            "group_residual_root_mean_squares": contributions,
            "final_state_norm": float(final_vector.float().norm().cpu()),
            "state_perturbation_gain": perturbation_gain,
        }


class MarulhoGroupedRecurrentStateBlock(nn.Module):
    surface = "marulho_grouped_recurrent_state_block.v1"

    def __init__(
        self,
        base: MarulhoHashedMicroExpertStateBlock,
        config: GroupedRecurrentConfig,
    ) -> None:
        super().__init__()
        _validate_recurrent_config(
            config,
            model_width=int(base.state_dim),
            model_layers=int(base.state_layers),
        )
        self.input_dim = int(base.input_dim)
        self.state_dim = int(base.state_dim)
        self.state_layers = int(base.state_layers)
        self.attention_heads = int(base.attention_heads)
        self.context_length = int(base.context_length)
        self.input_projection = base.input_projection
        self.layers = base.layers
        self.output_norm = base.output_norm
        self.expert_layer_index = int(base.expert_layer_index)
        self.memory_layer_index = int(config.memory_layer_index)
        self.recurrent = MarulhoGroupedRecurrentOrgan(self.state_dim, config)

    @property
    def expert_layer(self) -> MarulhoHashedMicroExpertBlock:
        layer = self.layers[self.expert_layer_index]
        if not isinstance(layer, MarulhoHashedMicroExpertBlock):
            raise RuntimeError("Configured V17 layer is not a hashed expert block")
        return layer

    def set_mode(self, mode: str) -> None:
        self.expert_layer.set_mode(mode)

    def set_recurrent_mode(self, mode: str) -> None:
        self.recurrent.set_mode(mode)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        head_dim = self.state_dim // self.attention_heads
        state: dict[str, torch.Tensor] = {
            "position": torch.zeros((), device=device, dtype=torch.long)
        }
        for layer_index in range(self.state_layers):
            state[f"layer_{layer_index}_key"] = torch.empty(
                int(batch_size),
                self.attention_heads,
                0,
                head_dim,
                device=device,
                dtype=dtype,
            )
            state[f"layer_{layer_index}_value"] = torch.empty_like(
                state[f"layer_{layer_index}_key"]
            )
        state.update(
            self.recurrent.initial_state(
                int(batch_size), device=device, dtype=dtype
            )
        )
        return state

    def forward(
        self,
        inputs: torch.Tensor,
        route_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        forced_expert_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        if inputs.ndim != 3:
            raise ValueError("V17 state block expects [batch,time,input_dim]")
        if route_ids.shape != inputs.shape[:2]:
            raise ValueError("route_ids must match input batch/time dimensions")
        batch_size, time_steps, _ = inputs.shape
        if int(time_steps) > self.context_length and state is None:
            raise ValueError("V17 input exceeds Transformer context length")
        current_state = (
            self.initial_state(
                int(batch_size), device=inputs.device, dtype=inputs.dtype
            )
            if state is None
            else state
        )
        position = current_state.get("position")
        position_offset = (
            position.to(device=inputs.device, dtype=torch.long)
            if isinstance(position, torch.Tensor)
            else torch.zeros((), device=inputs.device, dtype=torch.long)
        )
        hidden = self.input_projection(inputs)
        next_state: dict[str, torch.Tensor] = {
            "position": position_offset + int(time_steps)
        }
        cache_tokens = 0
        recurrent_state: dict[str, torch.Tensor] | None = None
        for layer_index, layer in enumerate(self.layers):
            kwargs = {
                "past_key": current_state.get(f"layer_{layer_index}_key"),
                "past_value": current_state.get(f"layer_{layer_index}_value"),
                "position_offset": position_offset,
            }
            if isinstance(layer, MarulhoHashedMicroExpertBlock):
                hidden, next_key, next_value = layer(
                    hidden,
                    route_ids=route_ids,
                    forced_expert_ids=forced_expert_ids,
                    **kwargs,
                )
            else:
                hidden, next_key, next_value = layer(hidden, **kwargs)
            next_state[f"layer_{layer_index}_key"] = next_key.detach()
            next_state[f"layer_{layer_index}_value"] = next_value.detach()
            cache_tokens = int(next_key.shape[2])
            if layer_index == self.memory_layer_index:
                hidden, recurrent_state = self.recurrent(
                    hidden,
                    None if state is None else current_state,
                )
                next_state.update(recurrent_state)
        if recurrent_state is None:
            raise RuntimeError("V17 recurrent organ was not executed")
        hidden = self.output_norm(hidden)
        telemetry = {
            "surface": self.surface,
            "state_core": "transformer_hashed_micro_experts_grouped_recurrent",
            "telemetry_collected": bool(collect_telemetry),
            "state_dim": self.state_dim,
            "state_layers": self.state_layers,
            "attention_heads": self.attention_heads,
            "context_length": self.context_length,
            "kv_cache_tokens": cache_tokens,
            "time_steps": int(time_steps),
            "normalization": "rmsnorm",
            "position_encoding": "rotary",
            "attention_backend": "torch_scaled_dot_product_attention",
            "expert_layer_index": self.expert_layer_index,
            "hashed_micro_expert_mode": self.expert_layer._mode_name,
            "recurrent_architecture": self.recurrent.architecture,
            "recurrent_mode": self.recurrent._mode_name,
            "recurrent_group_count": self.recurrent.group_count,
            "recurrent_group_width": self.recurrent.group_width,
            "recurrent_total_state_width": self.recurrent.total_state_width,
            "recurrent_state_bytes": self.recurrent.state_bytes(
                int(batch_size), element_size=inputs.element_size()
            ),
            "write_policy_uses_labels": False,
            "external_llm_used": False,
            "device": str(inputs.device),
        }
        return hidden, next_state, telemetry

    def step(
        self,
        token_input: torch.Tensor,
        route_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        if token_input.ndim != 2:
            raise ValueError("V17 step expects [batch,input_dim]")
        hidden, next_state, telemetry = self.forward(
            token_input.unsqueeze(1),
            route_ids.unsqueeze(1),
            state,
            collect_telemetry=collect_telemetry,
        )
        return hidden[:, 0], next_state, telemetry


class MarulhoGroupedRecurrentLanguageModel(
    MarulhoHashedMicroExpertLanguageModel
):
    surface = "marulho_grouped_recurrent_language_model.v1"
    generation_surface = "marulho_grouped_recurrent_generation.v1"

    def __init__(
        self,
        hashed_config: HashedMicroExpertConfig,
        recurrent_config: GroupedRecurrentConfig = GroupedRecurrentConfig(),
    ) -> None:
        self.recurrent_config = recurrent_config
        super().__init__(hashed_config)
        self.state_block = MarulhoGroupedRecurrentStateBlock(
            self.state_block,
            recurrent_config,
        )

    def set_grouped_recurrent_mode(self, mode: str) -> None:
        self.state_block.set_recurrent_mode(mode)

    def recurrent_parameter_report(self) -> dict[str, Any]:
        organ = self.state_block.recurrent
        organ_parameters = sum(int(value.numel()) for value in organ.parameters())
        return {
            "surface": "marulho_grouped_recurrent_parameter_report.v1",
            "architecture": organ.architecture,
            "total_model_parameters": sum(
                int(value.numel()) for value in self.parameters()
            ),
            "recurrent_organ_parameters": organ_parameters,
            "group_count": organ.group_count,
            "group_width": organ.group_width,
            "total_state_width": organ.total_state_width,
            "theoretical_recurrent_multiplies_per_token": (
                organ.theoretical_recurrent_multiplies_per_token()
            ),
            "external_llm_used": False,
        }

    @torch.no_grad()
    def final_recurrent_gradient_report(self) -> dict[str, Any]:
        rows = []
        for name, parameter in self.state_block.recurrent.named_parameters():
            gradient = parameter.grad
            rows.append(
                {
                    "name": name,
                    "received_gradient": gradient is not None,
                    "nonzero_gradient_elements": (
                        0
                        if gradient is None
                        else int(torch.count_nonzero(gradient).cpu())
                    ),
                }
            )
        return {
            "surface": "marulho_grouped_recurrent_gradient_report.v1",
            "architecture": self.state_block.recurrent.architecture,
            "mode": self.state_block.recurrent._mode_name,
            "parameters": rows,
            "all_parameters_received_gradient": all(
                row["received_gradient"] for row in rows
            ),
            "external_llm_used": False,
        }

    @torch.no_grad()
    def recurrent_diagnostic_report(self, input_ids: torch.Tensor) -> dict[str, Any]:
        captured: dict[str, torch.Tensor] = {}

        def capture(_module, inputs) -> None:
            captured["hidden"] = inputs[0].detach()

        handle = self.state_block.recurrent.register_forward_pre_hook(capture)
        was_training = self.training
        try:
            self.eval()
            self.forward(input_ids, collect_telemetry=False)
        finally:
            handle.remove()
            self.train(was_training)
        if "hidden" not in captured:
            raise RuntimeError("V17 diagnostic did not observe recurrent input")
        return {
            "surface": "marulho_grouped_recurrent_full_diagnostic.v1",
            "gradient": self.final_recurrent_gradient_report(),
            "state": self.state_block.recurrent.diagnostic_report(
                captured["hidden"]
            ),
            "external_llm_used": False,
        }


def build_grouped_recurrent_model(
    base_model: MarulhoHashedMicroExpertLanguageModel,
    config: GroupedRecurrentConfig = GroupedRecurrentConfig(),
) -> MarulhoGroupedRecurrentLanguageModel:
    if base_model.hashed_config.mode != "token_hash":
        raise ValueError("V17 requires the token_hash V11 base")
    hashed_config = replace(
        base_model.hashed_config,
        active_language_path=str(config.active_language_path),
    )
    model = MarulhoGroupedRecurrentLanguageModel(hashed_config, config)
    incompatible = model.load_state_dict(base_model.state_dict(), strict=False)
    expected_missing = {
        name
        for name in model.state_dict()
        if name.startswith("state_block.recurrent.")
    }
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(
            "V17 base load has unexpected missing tensors: "
            f"{incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"V17 base load has unexpected tensors: {incompatible.unexpected_keys}"
        )
    model.train(base_model.training)
    return model
