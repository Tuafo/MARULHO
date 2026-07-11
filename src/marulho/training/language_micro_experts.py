"""Causal product-key singleton micro-experts for the V10 falsifier."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel
from marulho.training.language_transformer import (
    MarulhoCausalTransformerStateBlock,
    MarulhoTransformerBlock,
    TransformerRMSNorm,
)


MICRO_EXPERT_MODES = (
    "shared_only",
    "fixed_random",
    "token_hash",
    "learned_router",
)
_MODE_IDS = {name: index for index, name in enumerate(MICRO_EXPERT_MODES)}


@dataclass(frozen=True)
class ProductKeyMicroExpertConfig:
    vocab_size: int
    width: int = 512
    layers: int = 4
    attention_heads: int = 8
    context_length: int = 72
    baseline_hidden_width: int = 2048
    shared_hidden_width: int = 1024
    expert_layer_index: int = 2
    expert_pool_size: int = 16_384
    retrieval_heads: int = 4
    experts_per_head: int = 2
    dropout: float = 0.0
    mode: str = "learned_router"
    hash_seed: int = 10_729
    active_language_path: str = "marulho_product_key_micro_experts_v10"


def _validate_config(config: ProductKeyMicroExpertConfig) -> None:
    if int(config.vocab_size) < 2:
        raise ValueError("vocab_size must be at least two")
    if int(config.width) < 8 or int(config.width) % 2 != 0:
        raise ValueError("width must be even and at least eight")
    if int(config.layers) < 1:
        raise ValueError("layers must be positive")
    if int(config.width) % int(config.attention_heads) != 0:
        raise ValueError("width must be divisible by attention_heads")
    if not 0 <= int(config.expert_layer_index) < int(config.layers):
        raise ValueError("expert_layer_index must identify an existing layer")
    if int(config.baseline_hidden_width) < int(config.width):
        raise ValueError("baseline_hidden_width must be at least width")
    if int(config.shared_hidden_width) < 1:
        raise ValueError("shared_hidden_width must be positive")
    subkey_count = math.isqrt(int(config.expert_pool_size))
    if subkey_count * subkey_count != int(config.expert_pool_size):
        raise ValueError("expert_pool_size must be a perfect square")
    if int(config.retrieval_heads) < 1:
        raise ValueError("retrieval_heads must be positive")
    if not 1 <= int(config.experts_per_head) <= subkey_count:
        raise ValueError("experts_per_head must be between one and sqrt(pool)")
    if config.mode not in MICRO_EXPERT_MODES:
        raise ValueError(f"mode must be one of {MICRO_EXPERT_MODES}")
    if not math.isfinite(float(config.dropout)) or not 0.0 <= float(
        config.dropout
    ) < 1.0:
        raise ValueError("dropout must be in [0, 1)")


class _RouteObserver(nn.Module):
    """Hook point for read-only routing diagnostics."""

    def forward(self, expert_ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        del expert_ids
        return weights


class MarulhoProductKeyMicroExpertBlock(nn.Module):
    """Transformer block with a shared MLP and retrieved singleton experts."""

    def __init__(
        self,
        base: MarulhoTransformerBlock,
        *,
        config: ProductKeyMicroExpertConfig,
    ) -> None:
        super().__init__()
        self.width = int(config.width)
        self.shared_hidden_width = int(config.shared_hidden_width)
        self.expert_pool_size = int(config.expert_pool_size)
        self.subkey_count = math.isqrt(self.expert_pool_size)
        self.retrieval_heads = int(config.retrieval_heads)
        self.experts_per_head = int(config.experts_per_head)
        self.hash_seed = int(config.hash_seed)
        self.expert_output_scale = 1.0 / math.sqrt(float(self.retrieval_heads))

        self.attention_norm = base.attention_norm
        self.attention = base.attention
        self.mlp_norm = base.mlp_norm
        self.dropout = base.dropout

        self.shared_gate_up = nn.Linear(
            self.width,
            self.shared_hidden_width * 2,
            bias=False,
        )
        self.shared_down = nn.Linear(
            self.shared_hidden_width,
            self.width,
            bias=False,
        )
        self.query_projection = nn.Linear(
            self.width,
            self.retrieval_heads * self.width,
            bias=False,
        )
        half_width = self.width // 2
        self.first_subkeys = nn.Parameter(
            torch.empty(self.subkey_count, half_width)
        )
        self.second_subkeys = nn.Parameter(
            torch.empty(self.subkey_count, half_width)
        )
        self.expert_input = nn.Embedding(self.expert_pool_size, self.width)
        self.expert_output = nn.Embedding(self.expert_pool_size, self.width)
        self.route_observer = _RouteObserver()
        self.register_buffer(
            "mode_id",
            torch.tensor(_MODE_IDS[config.mode], dtype=torch.long),
        )
        self._mode_name = str(config.mode)
        self._reset_new_parameters()

    def _reset_new_parameters(self) -> None:
        for value in (
            self.shared_gate_up.weight,
            self.shared_down.weight,
            self.query_projection.weight,
            self.first_subkeys,
            self.second_subkeys,
            self.expert_input.weight,
            self.expert_output.weight,
        ):
            nn.init.normal_(value, mean=0.0, std=0.02)

    def set_mode(self, mode: str) -> None:
        if mode not in MICRO_EXPERT_MODES:
            raise ValueError(f"mode must be one of {MICRO_EXPERT_MODES}")
        self.mode_id.fill_(_MODE_IDS[mode])
        self._mode_name = str(mode)

    def _router_tensors(
        self,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        learned = (self.mode_id == _MODE_IDS["learned_router"]).to(value.dtype)
        query = self.query_projection(value)
        query = query.detach() + learned * (query - query.detach())
        first = self.first_subkeys.detach() + learned * (
            self.first_subkeys - self.first_subkeys.detach()
        )
        second = self.second_subkeys.detach() + learned * (
            self.second_subkeys - self.second_subkeys.detach()
        )
        batch, time, _ = value.shape
        query = query.view(batch, time, self.retrieval_heads, self.width)
        query = query * query.pow(2).mean(dim=-1, keepdim=True).add(1.0e-6).rsqrt()
        return query, first, second

    def _retrieved_routes(
        self,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query, first, second = self._router_tensors(value)
        first_query, second_query = query.chunk(2, dim=-1)
        scale = 1.0 / math.sqrt(float(self.width // 2))
        first_scores = torch.einsum("bthd,sd->bths", first_query, first) * scale
        second_scores = torch.einsum("bthd,sd->bths", second_query, second) * scale
        first_values, first_indices = torch.topk(
            first_scores,
            k=self.experts_per_head,
            dim=-1,
        )
        second_values, second_indices = torch.topk(
            second_scores,
            k=self.experts_per_head,
            dim=-1,
        )
        pair_scores = first_values.unsqueeze(-1) + second_values.unsqueeze(-2)
        pair_ids = (
            first_indices.unsqueeze(-1) * self.subkey_count
            + second_indices.unsqueeze(-2)
        )
        flat_scores = pair_scores.flatten(-2)
        flat_ids = pair_ids.flatten(-2)
        selected_scores, selected_positions = torch.topk(
            flat_scores,
            k=self.experts_per_head,
            dim=-1,
        )
        selected_ids = flat_ids.gather(-1, selected_positions)
        return selected_ids, torch.softmax(selected_scores, dim=-1)

    def _hash_routes(self, route_ids: torch.Tensor) -> torch.Tensor:
        token = route_ids.to(dtype=torch.long).unsqueeze(-1).unsqueeze(-1)
        heads = torch.arange(
            self.retrieval_heads,
            device=route_ids.device,
            dtype=torch.long,
        ).view(1, 1, -1, 1)
        slots = torch.arange(
            self.experts_per_head,
            device=route_ids.device,
            dtype=torch.long,
        ).view(1, 1, 1, -1)
        return torch.remainder(
            token * 1_315_423_911
            + (heads + 1) * 2_654_435_761
            + (slots + 1) * 2_246_822_519
            + self.hash_seed,
            self.expert_pool_size,
        )

    def _micro_expert_output(
        self,
        value: torch.Tensor,
        route_ids: torch.Tensor,
    ) -> torch.Tensor:
        retrieved_ids, retrieved_weights = self._retrieved_routes(value)
        hashed_ids = self._hash_routes(route_ids)
        use_hash = self.mode_id == _MODE_IDS["token_hash"]
        selected_ids = torch.where(use_hash, hashed_ids, retrieved_ids)
        uniform_weights = torch.full_like(
            retrieved_weights,
            1.0 / float(self.experts_per_head),
        )
        route_weights = torch.where(use_hash, uniform_weights, retrieved_weights)
        route_weights = self.route_observer(selected_ids, route_weights)

        input_vectors = self.expert_input(selected_ids)
        activations = torch.einsum("btd,bthkd->bthk", value, input_vectors)
        activations = F.silu(activations) * route_weights
        output_vectors = self.expert_output(selected_ids)
        output = torch.einsum("bthk,bthkd->btd", activations, output_vectors)
        enabled = (self.mode_id != _MODE_IDS["shared_only"]).to(value.dtype)
        return output * enabled * self.expert_output_scale

    def forward(
        self,
        value: torch.Tensor,
        *,
        route_ids: torch.Tensor,
        past_key: torch.Tensor | None,
        past_value: torch.Tensor | None,
        position_offset: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention, next_key, next_value = self.attention(
            self.attention_norm(value),
            past_key=past_key,
            past_value=past_value,
            position_offset=position_offset,
        )
        value = value + self.dropout(attention)
        mlp_input = self.mlp_norm(value)
        gate, up = self.shared_gate_up(mlp_input).chunk(2, dim=-1)
        shared = self.shared_down(F.silu(gate) * up)
        experts = self._micro_expert_output(mlp_input, route_ids)
        value = value + self.dropout(shared + experts)
        return value, next_key, next_value


class MarulhoProductKeyMicroExpertStateBlock(nn.Module):
    """Transformer state block with one causally routed micro-expert layer."""

    surface = "marulho_product_key_micro_expert_state_block.v1"

    def __init__(
        self,
        base: MarulhoCausalTransformerStateBlock,
        *,
        config: ProductKeyMicroExpertConfig,
    ) -> None:
        super().__init__()
        self.input_dim = int(base.input_dim)
        self.state_dim = int(base.state_dim)
        self.state_layers = int(base.state_layers)
        self.attention_heads = int(base.attention_heads)
        self.context_length = int(base.context_length)
        self.dropout = float(base.dropout)
        self.input_projection = base.input_projection
        layers: list[nn.Module] = []
        for index, layer in enumerate(base.layers):
            if index == int(config.expert_layer_index):
                layers.append(MarulhoProductKeyMicroExpertBlock(layer, config=config))
            else:
                layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.output_norm = base.output_norm
        self.expert_layer_index = int(config.expert_layer_index)

    @property
    def expert_layer(self) -> MarulhoProductKeyMicroExpertBlock:
        layer = self.layers[self.expert_layer_index]
        if not isinstance(layer, MarulhoProductKeyMicroExpertBlock):
            raise RuntimeError("Configured V10 layer is not a micro-expert block")
        return layer

    def set_mode(self, mode: str) -> None:
        self.expert_layer.set_mode(mode)

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
        return state

    def _telemetry(
        self,
        *,
        device: torch.device,
        time_steps: int,
        cache_tokens: int,
        collected: bool,
    ) -> dict[str, Any]:
        layer = self.expert_layer
        return {
            "surface": self.surface,
            "state_core": "transformer_product_key_micro_experts",
            "telemetry_collected": bool(collected),
            "spike_telemetry_available": False,
            "spike_rate": 0.0,
            "dead_neuron_fraction": 0.0,
            "over_firing_fraction": 0.0,
            "adaptive_timestep_budget": 1,
            "adaptive_step_count": int(time_steps),
            "state_dim": self.state_dim,
            "state_layers": self.state_layers,
            "attention_heads": self.attention_heads,
            "context_length": self.context_length,
            "kv_cache_tokens": int(cache_tokens),
            "time_steps": int(time_steps),
            "normalization": "rmsnorm_with_per_token_query_rms",
            "position_encoding": "rotary",
            "attention_backend": "torch_scaled_dot_product_attention",
            "expert_layer_index": self.expert_layer_index,
            "expert_pool_size": layer.expert_pool_size,
            "retrieval_heads": layer.retrieval_heads,
            "experts_per_head": layer.experts_per_head,
            "active_experts_per_token": (
                layer.retrieval_heads * layer.experts_per_head
            ),
            "micro_expert_mode": layer._mode_name,
            "mode_selected_by_tensor_buffer": True,
            "recurrent_gradient_horizon": 0,
            "truncated_bptt_applied": False,
            "truncated_bptt_boundary_count": 0,
            "gradient_horizon_policy": "causal_attention_context",
            "device": str(device),
        }

    def forward(
        self,
        inputs: torch.Tensor,
        route_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        if inputs.ndim != 3:
            raise ValueError("V10 state block expects [batch, time, input_dim]")
        if route_ids.shape != inputs.shape[:2]:
            raise ValueError("route_ids must match input batch/time dimensions")
        batch_size, time_steps, _ = inputs.shape
        if int(time_steps) > self.context_length and state is None:
            raise ValueError("V10 input exceeds transformer context length")
        current_state = (
            self.initial_state(
                int(batch_size),
                device=inputs.device,
                dtype=inputs.dtype,
            )
            if state is None
            else state
        )
        position_value = current_state.get("position")
        position_offset = (
            position_value.to(device=inputs.device, dtype=torch.long)
            if isinstance(position_value, torch.Tensor)
            else torch.zeros((), device=inputs.device, dtype=torch.long)
        )
        hidden = self.input_projection(inputs)
        next_state: dict[str, torch.Tensor] = {
            "position": position_offset + int(time_steps)
        }
        cache_tokens = 0
        for layer_index, layer in enumerate(self.layers):
            kwargs = {
                "past_key": current_state.get(f"layer_{layer_index}_key"),
                "past_value": current_state.get(f"layer_{layer_index}_value"),
                "position_offset": position_offset,
            }
            if isinstance(layer, MarulhoProductKeyMicroExpertBlock):
                hidden, next_key, next_value = layer(
                    hidden,
                    route_ids=route_ids,
                    **kwargs,
                )
            else:
                hidden, next_key, next_value = layer(hidden, **kwargs)
            next_state[f"layer_{layer_index}_key"] = next_key.detach()
            next_state[f"layer_{layer_index}_value"] = next_value.detach()
            cache_tokens = int(next_key.shape[2])
        hidden = self.output_norm(hidden)
        return (
            hidden,
            next_state,
            self._telemetry(
                device=inputs.device,
                time_steps=int(time_steps),
                cache_tokens=cache_tokens,
                collected=collect_telemetry,
            ),
        )

    def step(
        self,
        token_input: torch.Tensor,
        route_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        if token_input.ndim != 2:
            raise ValueError("V10 step expects [batch, input_dim]")
        if route_ids.ndim != 1 or int(route_ids.shape[0]) != int(token_input.shape[0]):
            raise ValueError("V10 step route_ids must be [batch]")
        hidden, next_state, telemetry = self.forward(
            token_input.unsqueeze(1),
            route_ids.unsqueeze(1),
            state,
            collect_telemetry=collect_telemetry,
        )
        return hidden[:, 0], next_state, telemetry


class MarulhoProductKeyMicroExpertLanguageModel(MarulhoLanguageModel):
    """MARULHO-owned V10 model used only by matched falsification."""

    surface = "marulho_product_key_micro_expert_language_model.v1"

    def __init__(self, micro_config: ProductKeyMicroExpertConfig) -> None:
        _validate_config(micro_config)
        self.micro_config = micro_config
        super().__init__(
            LanguageModelConfig(
                vocab_size=int(micro_config.vocab_size),
                embedding_dim=int(micro_config.width),
                state_dim=int(micro_config.width),
                state_layers=int(micro_config.layers),
                attention_heads=int(micro_config.attention_heads),
                transformer_context_length=int(micro_config.context_length),
                transformer_mlp_ratio=(
                    float(micro_config.baseline_hidden_width)
                    / float(micro_config.width)
                ),
                transformer_dropout=float(micro_config.dropout),
                tie_embeddings=True,
                active_language_path=str(micro_config.active_language_path),
            )
        )
        self.state_block = MarulhoProductKeyMicroExpertStateBlock(
            self.state_block,
            config=micro_config,
        )

    def set_micro_expert_mode(self, mode: str) -> None:
        self.state_block.set_mode(mode)

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("V10 model expects input_ids shaped [batch, time]")
        runtime_ids = input_ids.to(device=self.device, dtype=torch.long)
        hidden, next_state, transformer = self.state_block(
            self.token_embedding(runtime_ids),
            runtime_ids,
            state,
            collect_telemetry=collect_telemetry,
        )
        return {
            "hidden": hidden,
            "state": next_state,
            "telemetry": {
                **transformer,
                "active_language_path": self.config.active_language_path,
                "external_llm_used": False,
                "owned_by_marulho": True,
                "vocab_size": int(self.config.vocab_size),
            },
        }

    def forward_step(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        del decode_vocab_only
        if input_ids.ndim == 1:
            step_ids = input_ids.unsqueeze(1)
        elif input_ids.ndim == 2 and int(input_ids.shape[1]) == 1:
            step_ids = input_ids
        else:
            raise ValueError("forward_step expects [batch] or [batch, 1] token ids")
        runtime_ids = step_ids.to(device=self.device, dtype=torch.long)
        hidden, next_state, transformer = self.state_block.step(
            self.token_embedding(runtime_ids[:, 0]),
            runtime_ids[:, 0],
            state,
            collect_telemetry=collect_telemetry,
        )
        return {
            "logits": self.lm_head(hidden).unsqueeze(1),
            "state": next_state,
            "telemetry": {
                **transformer,
                "active_language_path": self.config.active_language_path,
                "external_llm_used": False,
                "owned_by_marulho": True,
                "vocab_size": int(self.config.vocab_size),
            },
        }

    def active_parameter_report(self) -> dict[str, Any]:
        config = self.micro_config
        width = int(config.width)
        shared = 3 * width * int(config.shared_hidden_width)
        query = width * int(config.retrieval_heads) * width
        key_search = int(config.retrieval_heads) * math.isqrt(
            int(config.expert_pool_size)
        ) * width
        active_experts = int(config.retrieval_heads) * int(config.experts_per_head)
        expert_work = active_experts * 2 * width
        candidate_work = shared + query + key_search + expert_work
        baseline_work = 3 * width * int(config.baseline_hidden_width)
        stored_micro = (
            shared
            + query
            + math.isqrt(int(config.expert_pool_size)) * width
            + 2 * int(config.expert_pool_size) * width
        )
        return {
            "surface": "marulho_micro_expert_active_parameters.v1",
            "total_model_parameters": sum(
                int(value.numel()) for value in self.parameters()
            ),
            "stored_replacement_parameters": stored_micro,
            "baseline_replaced_mlp_parameters": baseline_work,
            "shared_path_parameters": shared,
            "query_projection_parameters": query,
            "product_subkey_parameters": (
                math.isqrt(int(config.expert_pool_size)) * width
            ),
            "expert_pool_parameters": 2 * int(config.expert_pool_size) * width,
            "expert_pool_size": int(config.expert_pool_size),
            "active_experts_per_token": active_experts,
            "theoretical_candidate_multiplies_per_token": candidate_work,
            "theoretical_baseline_mlp_multiplies_per_token": baseline_work,
            "candidate_to_baseline_multiply_ratio": candidate_work / baseline_work,
            "external_llm_used": False,
        }

    @torch.no_grad()
    def routing_report(
        self,
        input_ids: torch.Tensor,
        *,
        max_vector_samples: int = 256,
    ) -> dict[str, Any]:
        captured: dict[str, torch.Tensor] = {}

        def capture(_module, inputs, output) -> None:
            captured["ids"] = inputs[0].detach()
            captured["weights"] = output.detach()

        layer = self.state_block.expert_layer
        handle = layer.route_observer.register_forward_hook(capture)
        was_training = self.training
        try:
            self.eval()
            self.forward(input_ids, collect_telemetry=False)
        finally:
            handle.remove()
            self.train(was_training)
        expert_ids = captured["ids"]
        weights = captured["weights"]
        flat_ids = expert_ids.reshape(-1)
        counts = torch.bincount(flat_ids, minlength=layer.expert_pool_size).float()
        mass = torch.zeros(
            layer.expert_pool_size,
            device=weights.device,
            dtype=torch.float32,
        )
        mass.scatter_add_(0, flat_ids, weights.reshape(-1).float())
        probabilities = mass / mass.sum().clamp_min(1.0e-12)
        nonzero = probabilities > 0
        unevenness = (
            probabilities[nonzero]
            * (probabilities[nonzero] * layer.expert_pool_size).log()
        ).sum()
        route_entropy = -(
            weights.float() * weights.float().clamp_min(1.0e-12).log()
        ).sum(dim=-1).mean()
        per_token = expert_ids.flatten(-2)
        sorted_ids = per_token.sort(dim=-1).values
        duplicate_count = (sorted_ids[..., 1:] == sorted_ids[..., :-1]).sum(dim=-1)
        used_ids = torch.nonzero(counts > 0, as_tuple=False).flatten()
        if int(used_ids.numel()) > int(max_vector_samples):
            positions = torch.linspace(
                0,
                int(used_ids.numel()) - 1,
                steps=int(max_vector_samples),
                device=used_ids.device,
            ).long()
            sampled_ids = used_ids.index_select(0, positions)
        else:
            sampled_ids = used_ids

        def mean_absolute_off_diagonal_cosine(table: torch.Tensor) -> float:
            if int(sampled_ids.numel()) < 2:
                return 0.0
            vectors = F.normalize(table.index_select(0, sampled_ids).float(), dim=-1)
            gram = vectors @ vectors.T
            count = int(gram.numel()) - int(gram.shape[0])
            return float(
                (gram.abs().sum() - gram.diagonal().abs().sum()).div(count).cpu()
            )

        return {
            "surface": "marulho_micro_expert_routing_report.v1",
            "mode": layer._mode_name,
            "expert_pool_size": layer.expert_pool_size,
            "route_assignment_count": int(flat_ids.numel()),
            "active_experts_per_token": int(per_token.shape[-1]),
            "used_expert_count": int((counts > 0).sum().cpu()),
            "used_expert_fraction": float((counts > 0).float().mean().cpu()),
            "maximum_assignment_count": int(counts.max().cpu()),
            "routing_unevenness_kl_uniform": float(unevenness.cpu()),
            "mean_route_weight_entropy": float(route_entropy.cpu()),
            "mean_duplicate_experts_per_token": float(
                duplicate_count.float().mean().cpu()
            ),
            "expert_input_mean_absolute_cosine": (
                mean_absolute_off_diagonal_cosine(layer.expert_input.weight)
            ),
            "expert_output_mean_absolute_cosine": (
                mean_absolute_off_diagonal_cosine(layer.expert_output.weight)
            ),
            "router_uses_labels": False,
            "promotion_metric": False,
            "external_llm_used": False,
        }
