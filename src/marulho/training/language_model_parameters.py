from __future__ import annotations

from dataclasses import asdict
from typing import Any

from marulho.training.language_model import LanguageModelConfig


def _state_block_parameter_count(config: LanguageModelConfig) -> int:
    embedding_dim = int(config.embedding_dim)
    state_dim = int(config.state_dim)
    return int(
        (8 * embedding_dim * state_dim)
        + (3 * state_dim * state_dim)
        + (13 * state_dim)
        + embedding_dim
    )


def _expert_hidden_dim(config: LanguageModelConfig) -> int:
    hidden = int(config.expert_hidden_dim)
    return hidden if hidden > 0 else int(config.state_dim) * 2


def _expert_parameter_count_per_column(config: LanguageModelConfig) -> int:
    if int(config.expert_count) <= 0:
        return 0
    state_dim = int(config.state_dim)
    hidden_dim = _expert_hidden_dim(config)
    return int((2 * state_dim * hidden_dim) + hidden_dim + state_dim)


def estimate_language_model_parameters(
    config: LanguageModelConfig,
    *,
    dtype_bytes: int = 2,
) -> dict[str, Any]:
    vocab_size = int(config.vocab_size)
    embedding_dim = int(config.embedding_dim)
    state_dim = int(config.state_dim)
    expert_count = max(0, int(config.expert_count))
    active_expert_count = max(1, int(config.active_expert_count))
    route_candidate_count = (
        expert_count
        if int(config.route_candidate_count) <= 0
        else min(expert_count, int(config.route_candidate_count))
    )
    token_embedding = int(vocab_size * embedding_dim)
    state_block = _state_block_parameter_count(config)
    route_bank = int((expert_count * state_dim) + expert_count) if expert_count else 0
    per_expert = _expert_parameter_count_per_column(config)
    expert_total = int(expert_count * per_expert)
    memory_slots = int(max(0, int(config.memory_slot_count)) * state_dim)
    memory_slot_gate = 1 if int(config.memory_slot_count) > 0 else 0
    lm_head = int((state_dim * vocab_size) + vocab_size)
    total = int(
        token_embedding
        + state_block
        + route_bank
        + expert_total
        + memory_slots
        + memory_slot_gate
        + lm_head
    )
    active_experts = min(active_expert_count, max(0, route_candidate_count))
    route_candidate_scored_parameters = (
        int(route_candidate_count * (state_dim + 1)) if expert_count else 0
    )
    memory_candidate_count = (
        0
        if int(config.memory_slot_count) <= 0
        else (
            min(
                int(config.memory_slot_count),
                max(1, int(config.active_memory_slot_count)),
            )
            if int(config.memory_slot_candidate_count) <= 0
            else min(int(config.memory_slot_count), int(config.memory_slot_candidate_count))
        )
    )
    active_memory_slots = min(
        max(1, int(config.active_memory_slot_count)),
        max(0, int(memory_candidate_count)),
    )
    memory_active_parameters = int(active_memory_slots * state_dim)
    active_parameters_per_token = int(
        embedding_dim
        + state_block
        + lm_head
        + (active_experts * per_expert)
        + route_candidate_scored_parameters
        + memory_active_parameters
    )
    dtype_size = max(1, int(dtype_bytes))
    return {
        "surface": "marulho_language_model_parameter_estimate.v1",
        "config": asdict(config),
        "total_parameters": total,
        "parameter_breakdown": {
            "token_embedding": token_embedding,
            "selective_spiking_state_block": state_block,
            "route_bank": route_bank,
            "routed_experts": expert_total,
            "memory_slots": memory_slots,
            "memory_slot_gate": memory_slot_gate,
            "lm_head_dense_vocab": lm_head,
        },
        "expert_parameters_per_column": int(per_expert),
        "memory_slot_count": int(max(0, int(config.memory_slot_count))),
        "memory_slot_candidate_count": int(memory_candidate_count),
        "active_memory_slot_count_per_token": int(active_memory_slots),
        "active_parameters_per_token_estimate": active_parameters_per_token,
        "active_parameter_fraction_estimate": (
            float(active_parameters_per_token) / float(total) if total > 0 else 0.0
        ),
        "active_expert_count_per_token": int(active_experts),
        "route_candidate_count": int(route_candidate_count),
        "route_candidate_rows_scored_per_token": int(route_candidate_count),
        "dense_vocab_head_active": True,
        "sampled_or_adaptive_vocab_xent_present": False,
        "parameter_memory_mib": float(total * dtype_size / (1024 * 1024)),
        "parameter_memory_mib_fp16": float(total * 2 / (1024 * 1024)),
        "parameter_memory_mib_fp32": float(total * 4 / (1024 * 1024)),
        "adamw_train_state_mib_fp32_estimate": float(total * 16 / (1024 * 1024)),
    }
