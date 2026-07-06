from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import time
from typing import Any, Mapping, Sequence

import torch

from marulho.core.language_expert_dispatch_triton import (
    language_expert_dispatch_triton_stats,
    language_expert_dispatch_triton_stats_delta,
)
from marulho.core.language_memory_slots_triton import (
    language_memory_slots_triton_stats,
    language_memory_slots_triton_stats_delta,
)
from marulho.core.language_plif_triton import (
    language_plif_triton_stats,
    language_plif_triton_stats_delta,
)
from marulho.core.language_rmsnorm_triton import (
    language_rmsnorm_triton_stats,
    language_rmsnorm_triton_stats_delta,
)
from marulho.core.language_route_topk_triton import (
    language_route_topk_triton_stats,
    language_route_topk_triton_stats_delta,
)
from marulho.core.language_sampled_vocab_ce_triton import (
    build_sampled_target_positions,
    build_sampled_vocab_ids,
    language_sampled_vocab_ce_triton_stats,
    language_sampled_vocab_ce_triton_stats_delta,
)
from marulho.training.language_model import (
    LanguageBatch,
    MarulhoLanguageModel,
    evaluate_language_model,
    precompute_sampled_vocab_batches,
)
from marulho.training.language_model_parameters import estimate_language_model_parameters


_TRAINING_WINDOW_TRITON_TRACKERS = (
    (
        "language_rmsnorm_triton",
        language_rmsnorm_triton_stats,
        language_rmsnorm_triton_stats_delta,
    ),
    (
        "language_plif_triton",
        language_plif_triton_stats,
        language_plif_triton_stats_delta,
    ),
    (
        "language_route_topk_triton",
        language_route_topk_triton_stats,
        language_route_topk_triton_stats_delta,
    ),
    (
        "language_expert_dispatch_triton",
        language_expert_dispatch_triton_stats,
        language_expert_dispatch_triton_stats_delta,
    ),
    (
        "language_memory_slots_triton",
        language_memory_slots_triton_stats,
        language_memory_slots_triton_stats_delta,
    ),
    (
        "language_sampled_vocab_ce_triton",
        language_sampled_vocab_ce_triton_stats,
        language_sampled_vocab_ce_triton_stats_delta,
    ),
)


def _language_training_triton_stats_snapshot() -> dict[str, Mapping[str, Any]]:
    return {
        name: stats_fn()
        for name, stats_fn, _delta_fn in _TRAINING_WINDOW_TRITON_TRACKERS
    }


def _language_training_triton_stats_delta(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        name: delta_fn(before.get(name, {}), after.get(name, {}))
        for name, _stats_fn, delta_fn in _TRAINING_WINDOW_TRITON_TRACKERS
    }


def _sum_triton_delta_field(
    deltas: Mapping[str, Mapping[str, Any]],
    *fields: str,
) -> int:
    return int(
        sum(
            int(delta.get(field, 0) or 0)
            for delta in deltas.values()
            for field in fields
        )
    )


def _training_window_triton_accounting(
    deltas: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    names = tuple(
        name for name, _stats_fn, _delta_fn in _TRAINING_WINDOW_TRITON_TRACKERS
    )
    used_names = tuple(
        name
        for name in names
        if bool(deltas.get(name, {}).get("triton_kernel_used", False))
        or bool(deltas.get(name, {}).get("triton_autograd_used", False))
    )
    used_name_set = set(used_names)
    return {
        "surface": "marulho_language_continual_training_window_triton_accounting.v1",
        "scope": "measured_update_window_only",
        "tracked_kernel_names": list(names),
        "tracked_kernel_used_names": list(used_names),
        "tracked_kernel_unused_names": [
            name for name in names if name not in used_name_set
        ],
        "tracked_triton_forward_calls": _sum_triton_delta_field(
            deltas,
            "triton_forward_calls",
            "triton_forward_no_eligibility_calls",
        ),
        "tracked_triton_backward_calls": _sum_triton_delta_field(
            deltas,
            "triton_backward_calls",
        ),
        "tracked_triton_autograd_forward_calls": _sum_triton_delta_field(
            deltas,
            "triton_autograd_forward_calls",
        ),
        "tracked_torch_autograd_backward_calls": _sum_triton_delta_field(
            deltas,
            "torch_autograd_backward_calls",
        ),
        "tracked_torch_fallback_calls": _sum_triton_delta_field(
            deltas,
            "torch_fallback_calls",
        ),
        "tracked_torch_fallback_elements": _sum_triton_delta_field(
            deltas,
            "torch_fallback_elements",
        ),
        "tracked_triton_failure_count": _sum_triton_delta_field(
            deltas,
            "triton_failure_count",
        ),
        **{name: dict(deltas.get(name, {})) for name in names},
    }


@dataclass(frozen=True)
class LanguageContinualLearningConfig:
    learning_rate: float = 1e-3
    max_steps: int = 1
    replay_loss_weight: float = 0.25
    max_grad_norm: float = 1.0
    gradient_clip_interval: int = 1
    sparse_vocab_optimizer: bool = True
    dense_adamw_backend: str = "default"
    paired_sampled_vocab_loss: bool = False
    collect_training_telemetry: bool = False
    forgetting_tolerance: float = 0.10
    replay_retention_tolerance: float = 0.10
    min_new_loss_improvement: float = 0.0
    rollback_on_forgetting: bool = True


def _clone_state_dict(model: MarulhoLanguageModel) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone().cpu()
        for key, value in model.state_dict().items()
    }


def _state_dict_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _parameter_delta_l2(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
) -> float:
    total = 0.0
    for key, before_tensor in before.items():
        after_tensor = after[key].detach().cpu()
        if not (
            torch.is_floating_point(before_tensor)
            and torch.is_floating_point(after_tensor)
        ):
            continue
        delta = after_tensor - before_tensor.detach().cpu()
        total += float(delta.pow(2).sum().item())
    return float(math.sqrt(total))


def _metric_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    key: str,
) -> float:
    return float(after.get(key, 0.0) or 0.0) - float(before.get(key, 0.0) or 0.0)


def _spike_rate(report: Mapping[str, Any]) -> float:
    telemetry = report.get("spike_telemetry")
    if not isinstance(telemetry, Mapping):
        return 0.0
    return float(telemetry.get("spike_rate", 0.0) or 0.0)


def _precomputed_memory_candidate_slots_scored(batch: LanguageBatch) -> int:
    if batch.memory_candidate_ids is None:
        return 0
    return int(batch.memory_candidate_ids.numel())


def _optional_batch_tensor_pair_fusable(
    left: torch.Tensor | None,
    right: torch.Tensor | None,
) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return left.ndim == right.ndim and tuple(left.shape[1:]) == tuple(right.shape[1:])


def _update_replay_pair_fusable(update_batch: LanguageBatch, replay_batch: LanguageBatch) -> bool:
    return (
        update_batch.input_ids.ndim == replay_batch.input_ids.ndim
        and tuple(update_batch.input_ids.shape[1:])
        == tuple(replay_batch.input_ids.shape[1:])
        and tuple(update_batch.target_ids.shape[1:])
        == tuple(replay_batch.target_ids.shape[1:])
        and _optional_batch_tensor_pair_fusable(
            update_batch.memory_candidate_ids,
            replay_batch.memory_candidate_ids,
        )
        and _optional_batch_tensor_pair_fusable(
            update_batch.route_candidate_ids,
            replay_batch.route_candidate_ids,
        )
    )


def _paired_update_replay_fusion_plan(
    *,
    new_update_batches: Sequence[LanguageBatch],
    replay_update_batches: Sequence[LanguageBatch],
    step_count: int,
    replay_loss_weight: float,
) -> dict[str, Any]:
    if not replay_update_batches:
        return {
            "surface": "marulho_language_continual_paired_update_replay_fusion.v1",
            "enabled": False,
            "reason": "no_replay_batches",
            "replay_loss_weight": float(replay_loss_weight),
            "planned_optimizer_steps": int(max(1, step_count) * len(new_update_batches)),
            "planned_fused_steps": 0,
        }
    replay_count = len(replay_update_batches)
    planned_steps = int(max(1, step_count) * len(new_update_batches))
    unique_pairs: set[tuple[int, int]] = set()
    for step in range(max(1, step_count)):
        for index, batch in enumerate(new_update_batches):
            replay_index = (step * len(new_update_batches) + index) % replay_count
            replay_batch = replay_update_batches[replay_index]
            unique_pairs.add((index, replay_index))
            if not _update_replay_pair_fusable(batch, replay_batch):
                return {
                    "surface": (
                        "marulho_language_continual_paired_update_replay_fusion.v1"
                    ),
                    "enabled": False,
                    "reason": "paired_batch_shape_or_candidate_mismatch",
                    "replay_loss_weight": float(replay_loss_weight),
                    "planned_optimizer_steps": planned_steps,
                    "planned_fused_steps": 0,
                    "unique_pair_count": len(unique_pairs),
                }
    return {
        "surface": "marulho_language_continual_paired_update_replay_fusion.v1",
        "enabled": True,
        "reason": None,
        "mode": "single_hidden_forward_split_update_replay_losses",
        "weighted_replay_loss_preserved": True,
        "replay_loss_weight": float(replay_loss_weight),
        "planned_optimizer_steps": planned_steps,
        "planned_fused_steps": planned_steps,
        "unique_pair_count": len(unique_pairs),
    }


def _precompute_paired_update_replay_sampled_vocab(
    model: MarulhoLanguageModel,
    *,
    new_update_batches: Sequence[LanguageBatch],
    replay_update_batches: Sequence[LanguageBatch],
    step_count: int,
) -> tuple[
    dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    dict[str, Any],
]:
    configured_sample_count = int(model.config.sampled_vocab_size)
    use_sampled_vocab = (
        configured_sample_count > 0
        and configured_sample_count < int(model.config.vocab_size)
    )
    planned_steps = int(max(1, step_count) * len(new_update_batches))
    if not bool(use_sampled_vocab):
        return {}, {
            "surface": (
                "marulho_language_continual_paired_sampled_vocab_precompute.v1"
            ),
            "enabled": False,
            "reason": "sampled_vocab_training_disabled",
            "planned_optimizer_steps": planned_steps,
            "pair_count": 0,
            "device": str(model.device),
        }
    if not replay_update_batches:
        return {}, {
            "surface": (
                "marulho_language_continual_paired_sampled_vocab_precompute.v1"
            ),
            "enabled": False,
            "reason": "no_replay_batches",
            "planned_optimizer_steps": planned_steps,
            "pair_count": 0,
            "device": str(model.device),
        }
    pair_inputs: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    sampled_row_counts: list[int] = []
    update_position_counts: list[int] = []
    replay_position_counts: list[int] = []
    replay_count = len(replay_update_batches)
    for step in range(max(1, step_count)):
        for index, update_batch in enumerate(new_update_batches):
            replay_index = (step * len(new_update_batches) + index) % replay_count
            pair_key = (index, replay_index)
            if pair_key in pair_inputs:
                continue
            replay_batch = replay_update_batches[replay_index]
            update_targets = update_batch.target_ids.to(
                device=model.device,
                dtype=torch.long,
            ).reshape(-1)
            replay_targets = replay_batch.target_ids.to(
                device=model.device,
                dtype=torch.long,
            ).reshape(-1)
            combined_targets = torch.cat((update_targets, replay_targets), dim=0)
            sampled_vocab_ids = build_sampled_vocab_ids(
                combined_targets,
                vocab_size=int(model.config.vocab_size),
                sample_count=configured_sample_count,
                device=model.device,
                validate_ids=False,
            )
            combined_positions = build_sampled_target_positions(
                combined_targets,
                sampled_vocab_ids,
                device=model.device,
                validate_targets=True,
            )
            pair_inputs[pair_key] = (
                sampled_vocab_ids,
                combined_targets,
                combined_positions,
            )
            sampled_row_counts.append(int(sampled_vocab_ids.numel()))
            update_position_count = int(update_targets.numel())
            update_position_counts.append(update_position_count)
            replay_position_counts.append(int(replay_targets.numel()))
    return pair_inputs, {
        "surface": "marulho_language_continual_paired_sampled_vocab_precompute.v1",
        "enabled": True,
        "reason": None,
        "planned_optimizer_steps": planned_steps,
        "pair_count": len(pair_inputs),
        "device": str(model.device),
        "sampled_vocab_size": int(configured_sample_count),
        "min_sampled_rows": min(sampled_row_counts, default=0),
        "max_sampled_rows": max(sampled_row_counts, default=0),
        "min_update_target_positions": min(update_position_counts, default=0),
        "max_update_target_positions": max(update_position_counts, default=0),
        "min_replay_target_positions": min(replay_position_counts, default=0),
        "max_replay_target_positions": max(replay_position_counts, default=0),
        "hot_update_window_precomputed": True,
    }


def _batch_tensors_on_device(batch: LanguageBatch, device: torch.device) -> bool:
    tensors = (
        batch.input_ids,
        batch.target_ids,
        batch.sampled_vocab_ids,
        batch.sampled_target_positions,
        batch.memory_candidate_ids,
        batch.route_candidate_ids,
    )
    return all(tensor is None or tensor.device == device for tensor in tensors)


def _stage_batch_on_device(batch: LanguageBatch, device: torch.device) -> LanguageBatch:
    if _batch_tensors_on_device(batch, device):
        return batch
    return batch.to(device)


def _stage_batches_on_device(
    batches: Sequence[LanguageBatch],
    device: torch.device,
) -> tuple[LanguageBatch, ...]:
    return tuple(_stage_batch_on_device(batch, device) for batch in batches)


def _batch_device_staging_report(
    *,
    device: torch.device,
    old_eval_batches: Sequence[LanguageBatch],
    new_eval_batches: Sequence[LanguageBatch],
    new_update_batches: Sequence[LanguageBatch],
    replay_update_batches: Sequence[LanguageBatch],
    elapsed_seconds: float,
) -> dict[str, Any]:
    update_batches_on_device = all(
        _batch_tensors_on_device(batch, device)
        for batch in (*tuple(new_update_batches), *tuple(replay_update_batches))
    )
    return {
        "surface": "marulho_language_continual_batch_device_staging.v1",
        "device": str(device),
        "old_eval_batch_count": int(len(old_eval_batches)),
        "new_eval_batch_count": int(len(new_eval_batches)),
        "new_update_batch_count": int(len(new_update_batches)),
        "replay_update_batch_count": int(len(replay_update_batches)),
        "elapsed_seconds": float(elapsed_seconds),
        "staged_before_pre_update_evaluation": True,
        "staged_before_measured_update_window": True,
        "all_update_batches_on_device_before_timing": bool(update_batches_on_device),
        "measured_update_loop_caller_device_transfer_calls": 0,
    }


def _memory_slot_learning_summary(
    model: MarulhoLanguageModel,
    *,
    update_memory: Mapping[str, Any],
    replay_memory: Mapping[str, Any],
    training_window_memory_slot_triton_stats_delta: Mapping[str, Any] | None,
    training_window_memory_slot_triton_training_autograd_enabled: bool,
    update_candidate_slots_scored: int,
    replay_candidate_slots_scored: int,
    update_observation_count: int,
    replay_observation_count: int,
) -> dict[str, Any]:
    source = update_memory if update_memory else replay_memory
    configured_slots = max(0, int(model.config.memory_slot_count))
    total_slots = int(source.get("total_slots", configured_slots) or 0)
    candidate_slot_count = int(
        source.get("candidate_slot_count", model.config.memory_slot_candidate_count)
        or 0
    )
    active_slots = int(
        source.get("active_slots_per_token", model.config.active_memory_slot_count)
        or 0
    )
    fallback_reasons = tuple(
        str(reason)
        for reason in (
            update_memory.get("fallback_reason"),
            replay_memory.get("fallback_reason"),
        )
        if reason
    )
    training_triton_delta = dict(
        training_window_memory_slot_triton_stats_delta or {}
    )
    return {
        "surface": "marulho_language_continual_memory_slots.v1",
        "enabled": bool(source.get("enabled", configured_slots > 0)),
        "total_slots": total_slots,
        "candidate_slot_count": candidate_slot_count,
        "active_slots_per_token": active_slots,
        "candidate_slots_scored": int(
            update_candidate_slots_scored + replay_candidate_slots_scored
        ),
        "update_candidate_slots_scored": int(update_candidate_slots_scored),
        "replay_candidate_slots_scored": int(replay_candidate_slots_scored),
        "update_observation_count": int(update_observation_count),
        "replay_observation_count": int(replay_observation_count),
        "runs_all_slots": bool(
            update_memory.get("runs_all_slots", False)
            or replay_memory.get("runs_all_slots", False)
        ),
        "fallback_reason": fallback_reasons[0] if fallback_reasons else None,
        "candidate_id_source": source.get("candidate_id_source"),
        "precomputed_candidate_ids_used": bool(
            source.get("precomputed_candidate_ids_used", False)
        ),
        "memory_slot_retrieval_backend": source.get("memory_slot_retrieval_backend"),
        "memory_slot_triton_stats_delta": source.get(
            "memory_slot_triton_stats_delta"
        ),
        "training_window_memory_slot_triton_stats_delta": training_triton_delta,
        "training_window_memory_slot_triton_training_autograd_enabled": bool(
            training_window_memory_slot_triton_training_autograd_enabled
        ),
        "training_window_memory_slot_triton_autograd_used": bool(
            training_triton_delta.get("triton_autograd_used", False)
        ),
        "training_window_memory_slot_triton_forward_calls": int(
            training_triton_delta.get("triton_forward_calls", 0) or 0
        ),
        "training_window_memory_slot_triton_autograd_forward_calls": int(
            training_triton_delta.get("triton_autograd_forward_calls", 0) or 0
        ),
        "training_window_memory_slot_torch_autograd_backward_calls": int(
            training_triton_delta.get("torch_autograd_backward_calls", 0) or 0
        ),
        "training_window_memory_slot_torch_fallback_calls": int(
            training_triton_delta.get("torch_fallback_calls", 0) or 0
        ),
        "memory_gate_readback": bool(source.get("memory_gate_readback", False)),
        "memory_slot_initialization": source.get("memory_slot_initialization"),
        "memory_slot_init_std": float(
            source.get("memory_slot_init_std", model.config.memory_slot_init_std)
        ),
        "memory_device": source.get("memory_device", str(model.device)),
        "active_parameters_per_token": int(
            source.get(
                "active_parameters_per_token",
                active_slots * int(model.config.state_dim),
            )
            or 0
        ),
        "online_update_path": True,
        "bounded_memory_slot_path": bool(
            source.get("enabled", configured_slots > 0)
            and candidate_slot_count > 0
            and candidate_slot_count < max(1, total_slots)
        ),
    }


def _active_compute_learning_summary(
    model: MarulhoLanguageModel,
    *,
    loss_evidence: Mapping[str, Any],
    update_routing: Mapping[str, Any],
    replay_routing: Mapping[str, Any],
    memory_slots: Mapping[str, Any],
    update_token_count: int,
    measured_update_loop_model_loss_calls: int,
) -> dict[str, Any]:
    parameter_estimate = estimate_language_model_parameters(model.config)
    update_route = dict(update_routing)
    replay_route = dict(replay_routing)
    route_source = update_route if update_route else replay_route
    expert_count = int(
        route_source.get("total_columns", int(model.config.expert_count)) or 0
    )
    active_columns_per_token = int(
        route_source.get(
            "active_expert_count_per_token",
            parameter_estimate.get("active_expert_count_per_token", 0),
        )
        or 0
    )
    route_candidate_rows_per_token = int(
        route_source.get(
            "route_candidate_count",
            parameter_estimate.get("route_candidate_rows_scored_per_token", 0),
        )
        or 0
    )
    route_candidate_rows_scored_estimate = int(
        max(0, int(update_token_count)) * max(0, route_candidate_rows_per_token)
    )
    observed_probe_route_candidate_rows = int(
        (update_route.get("candidate_rows_scored", 0) or 0)
        + (replay_route.get("candidate_rows_scored", 0) or 0)
    )
    sampled_vocab_training = bool(loss_evidence.get("sampled_vocab_training", False))
    configured_sampled_vocab_size = int(
        loss_evidence.get(
            "configured_sampled_vocab_size",
            int(model.config.sampled_vocab_size),
        )
        or 0
    )
    sampled_vocab_rows_per_loss_call = int(
        loss_evidence.get("actual_sampled_vocab_size", 0) or 0
    )
    loss_target_tokens_per_call = int(loss_evidence.get("target_token_count", 0) or 0)
    state_dim = int(model.config.state_dim)
    vocab_size = int(model.config.vocab_size)
    dense_lm_head_parameters = int(
        parameter_estimate.get("parameter_breakdown", {}).get(
            "lm_head_dense_vocab",
            (state_dim * vocab_size) + vocab_size,
        )
    )
    active_lm_head_rows_per_loss_call = (
        sampled_vocab_rows_per_loss_call if sampled_vocab_training else vocab_size
    )
    active_lm_head_parameters_per_loss_call = int(
        max(0, active_lm_head_rows_per_loss_call) * (state_dim + 1)
    )
    dense_active_parameters_per_token = int(
        parameter_estimate.get("active_parameters_per_token_estimate", 0) or 0
    )
    sampled_loss_active_parameters_per_loss_call = int(
        dense_active_parameters_per_token
        - dense_lm_head_parameters
        + active_lm_head_parameters_per_loss_call
    )
    total_slots = int(memory_slots.get("total_slots", 0) or 0)
    active_slots = int(memory_slots.get("active_slots_per_token", 0) or 0)
    candidate_slots = int(memory_slots.get("candidate_slot_count", 0) or 0)
    runs_all_columns = bool(
        update_route.get("runs_all_columns", False)
        or replay_route.get("runs_all_columns", False)
    )
    runs_all_slots = bool(memory_slots.get("runs_all_slots", False))
    return {
        "surface": "marulho_language_continual_active_compute.v1",
        "scope": "post_window_report_for_measured_update_shape",
        "collected_outside_measured_update_window": True,
        "per_step_active_compute_dict_build": False,
        "device": str(model.device),
        "active_language_path": str(model.config.active_language_path),
        "model_vocab_size": vocab_size,
        "generation_vocab_size": int(model.generation_vocab_size),
        "sampled_vocab_size": int(model.config.sampled_vocab_size),
        "configured_sampled_vocab_size": configured_sampled_vocab_size,
        "sampled_vocab_training": sampled_vocab_training,
        "full_vocab_logits_materialized": bool(
            loss_evidence.get("full_vocab_logits_materialized", True)
        ),
        "sampled_vocab_rows_per_loss_call": sampled_vocab_rows_per_loss_call,
        "loss_target_tokens_per_loss_call": loss_target_tokens_per_call,
        "measured_update_loop_model_loss_calls": int(
            measured_update_loop_model_loss_calls
        ),
        "update_token_count": int(update_token_count),
        "total_parameters": int(parameter_estimate.get("total_parameters", 0) or 0),
        "parameter_breakdown": dict(parameter_estimate.get("parameter_breakdown", {})),
        "active_parameters_per_token_estimate": dense_active_parameters_per_token,
        "active_parameter_fraction_estimate": float(
            parameter_estimate.get("active_parameter_fraction_estimate", 0.0) or 0.0
        ),
        "active_parameter_estimate_counts_dense_lm_head": True,
        "dense_lm_head_parameters": dense_lm_head_parameters,
        "active_lm_head_rows_per_loss_call": int(active_lm_head_rows_per_loss_call),
        "active_lm_head_parameters_per_loss_call": int(
            active_lm_head_parameters_per_loss_call
        ),
        "sampled_loss_active_parameters_per_loss_call_estimate": int(
            sampled_loss_active_parameters_per_loss_call
        ),
        "sampled_loss_active_parameter_fraction_estimate": (
            float(sampled_loss_active_parameters_per_loss_call)
            / float(parameter_estimate.get("total_parameters", 1) or 1)
        ),
        "total_columns": expert_count,
        "active_columns_per_token": active_columns_per_token,
        "route_candidate_rows_scored_per_token": route_candidate_rows_per_token,
        "route_candidate_rows_scored_in_measured_update_window_estimate": (
            route_candidate_rows_scored_estimate
        ),
        "observed_probe_route_candidate_rows_scored": (
            observed_probe_route_candidate_rows
        ),
        "runs_all_columns": runs_all_columns,
        "route_fallback_reason": (
            update_route.get("fallback_reason") or replay_route.get("fallback_reason")
        ),
        "route_device": route_source.get("route_device", str(model.device)),
        "route_candidate_id_source": route_source.get("candidate_id_source"),
        "precomputed_route_candidate_ids_used": bool(
            update_route.get("precomputed_candidate_ids_used", False)
            or replay_route.get("precomputed_candidate_ids_used", False)
        ),
        "total_memory_slots": total_slots,
        "candidate_memory_slots_per_token": candidate_slots,
        "active_memory_slots_per_token": active_slots,
        "memory_slot_candidate_rows_scored_in_measured_update_window": int(
            memory_slots.get("candidate_slots_scored", 0) or 0
        ),
        "active_memory_parameters_per_token": int(
            memory_slots.get("active_parameters_per_token", 0) or 0
        ),
        "runs_all_memory_slots": runs_all_slots,
        "memory_slot_fallback_reason": memory_slots.get("fallback_reason"),
        "memory_slot_candidate_id_source": memory_slots.get("candidate_id_source"),
        "precomputed_memory_candidate_ids_used": bool(
            memory_slots.get("precomputed_candidate_ids_used", False)
        ),
        "bounded_active_compute_path": bool(
            not runs_all_columns
            and not runs_all_slots
            and (expert_count <= 0 or route_candidate_rows_per_token < expert_count)
            and (total_slots <= 0 or candidate_slots < total_slots)
        ),
        "parameter_estimate": parameter_estimate,
    }


def _continual_optimizer_policy(
    model: MarulhoLanguageModel,
    config: LanguageContinualLearningConfig,
) -> tuple[list[torch.optim.Optimizer], str]:
    dense_backend = str(config.dense_adamw_backend).strip().lower()
    if dense_backend not in {"default", "foreach", "fused"}:
        raise ValueError(
            "dense_adamw_backend must be one of: default, foreach, fused"
        )
    if dense_backend == "fused" and model.device.type != "cuda":
        raise ValueError("dense_adamw_backend=fused requires a CUDA model")
    adamw_kwargs: dict[str, bool] = {}
    adamw_policy_suffix = ""
    if dense_backend == "foreach":
        adamw_kwargs["foreach"] = True
        adamw_policy_suffix = "_foreach"
    elif dense_backend == "fused":
        adamw_kwargs["fused"] = True
        adamw_policy_suffix = "_fused"
    sparse_names: set[str] = set()
    if bool(model.config.sparse_token_embedding_gradients):
        sparse_names.add("token_embedding.weight")
    if bool(model.config.sampled_vocab_sparse_lm_head_gradient):
        sparse_names.add("lm_head.weight")
    if not sparse_names:
        return [
            torch.optim.AdamW(
                model.parameters(),
                lr=float(config.learning_rate),
                **adamw_kwargs,
            )
        ], f"AdamW{adamw_policy_suffix}_all_parameters"
    if not bool(config.sparse_vocab_optimizer):
        raise ValueError(
            "sparse_vocab_optimizer must be enabled for models with sparse vocab gradients"
        )
    sparse_params: list[torch.nn.Parameter] = []
    dense_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in sparse_names:
            sparse_params.append(parameter)
        else:
            dense_params.append(parameter)
    optimizers: list[torch.optim.Optimizer] = []
    if dense_params:
        optimizers.append(
            torch.optim.AdamW(
                dense_params,
                lr=float(config.learning_rate),
                **adamw_kwargs,
            )
        )
    if sparse_params:
        optimizers.append(
            torch.optim.SparseAdam(
                sparse_params,
                lr=float(config.learning_rate),
            )
        )
    if not optimizers:
        raise ValueError("Language continual learning model has no trainable parameters")
    return (
        optimizers,
        f"AdamW{adamw_policy_suffix}_dense_core_plus_SparseAdam_vocab_rows",
    )


def _clip_grad_norm_sparse_aware(
    parameters: Sequence[torch.nn.Parameter],
    *,
    max_norm: float,
    device: torch.device,
) -> torch.Tensor:
    total_sq = torch.zeros((), device=device, dtype=torch.float32)
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        values = grad.coalesce().values() if grad.is_sparse else grad
        total_sq = total_sq + values.detach().float().pow(2).sum()
    total_norm = torch.sqrt(total_sq)
    limit = float(max_norm)
    if limit > 0.0:
        clip_coef = torch.clamp(
            torch.tensor(limit, device=device, dtype=torch.float32)
            / (total_norm + 1e-6),
            max=1.0,
        )
        for parameter in parameters:
            grad = parameter.grad
            if grad is None:
                continue
            if grad.is_sparse:
                grad = grad.coalesce()
                grad.values().mul_(clip_coef)
                parameter.grad = grad
            else:
                grad.mul_(clip_coef)
    return total_norm


def run_language_continual_learning_window(
    model: MarulhoLanguageModel,
    *,
    new_batches: Sequence[LanguageBatch],
    old_eval_batches: Sequence[LanguageBatch],
    new_eval_batches: Sequence[LanguageBatch],
    replay_batches: Sequence[LanguageBatch] = (),
    config: LanguageContinualLearningConfig | None = None,
) -> dict[str, Any]:
    """Apply a bounded online LM update and report anti-forgetting evidence."""

    if not new_batches:
        raise ValueError("new_batches must not be empty")
    if not old_eval_batches:
        raise ValueError("old_eval_batches must not be empty")
    if not new_eval_batches:
        raise ValueError("new_eval_batches must not be empty")
    cfg = config or LanguageContinualLearningConfig()
    total_started = time.perf_counter()
    was_training = model.training
    snapshot_started = time.perf_counter()
    snapshot = _clone_state_dict(model)
    snapshot_hash = _state_dict_hash(snapshot)
    assume_no_sleeping = (
        model.routed_experts.enabled
        and not bool(model.routed_experts.sleeping_expert_mask.detach().any().cpu().item())
    )
    snapshot_elapsed_seconds = max(0.0, time.perf_counter() - snapshot_started)
    precompute_started = time.perf_counter()
    old_eval_runtime_batches, old_eval_sampled_vocab_precompute = (
        precompute_sampled_vocab_batches(
            model,
            old_eval_batches,
            assume_no_sleeping_experts=assume_no_sleeping,
        )
    )
    new_eval_runtime_batches, new_eval_sampled_vocab_precompute = (
        precompute_sampled_vocab_batches(
            model,
            new_eval_batches,
            assume_no_sleeping_experts=assume_no_sleeping,
        )
    )
    new_update_batches, new_sampled_vocab_precompute = precompute_sampled_vocab_batches(
        model,
        new_batches,
        assume_no_sleeping_experts=assume_no_sleeping,
    )
    replay_update_batches, replay_sampled_vocab_precompute = (
        precompute_sampled_vocab_batches(
            model,
            replay_batches,
            assume_no_sleeping_experts=assume_no_sleeping,
        )
        if replay_batches
        else (
            tuple(),
            {
                "surface": "marulho_language_sampled_vocab_batch_precompute.v1",
                "enabled": False,
                "reason": "no_replay_batches",
                "batch_count": 0,
                "device": str(model.device),
                "route_candidate_precompute": {
                    "surface": "marulho_language_route_candidate_batch_precompute.v1",
                    "enabled": False,
                    "reason": "no_replay_batches",
                    "batch_count": 0,
                    "device": str(model.device),
                },
            },
        )
    )
    sampled_vocab_precompute_elapsed_seconds = max(
        0.0,
        time.perf_counter() - precompute_started,
    )
    batch_device_staging_started = time.perf_counter()
    old_eval_runtime_batches = _stage_batches_on_device(
        old_eval_runtime_batches,
        model.device,
    )
    new_eval_runtime_batches = _stage_batches_on_device(
        new_eval_runtime_batches,
        model.device,
    )
    new_update_batches = _stage_batches_on_device(new_update_batches, model.device)
    replay_update_batches = _stage_batches_on_device(
        replay_update_batches,
        model.device,
    )
    batch_device_staging_elapsed_seconds = max(
        0.0,
        time.perf_counter() - batch_device_staging_started,
    )
    batch_device_staging = _batch_device_staging_report(
        device=model.device,
        old_eval_batches=old_eval_runtime_batches,
        new_eval_batches=new_eval_runtime_batches,
        new_update_batches=new_update_batches,
        replay_update_batches=replay_update_batches,
        elapsed_seconds=batch_device_staging_elapsed_seconds,
    )
    step_count = max(1, int(cfg.max_steps))
    replay_count = len(replay_update_batches)
    paired_update_replay_plan_started = time.perf_counter()
    paired_update_replay_fusion = _paired_update_replay_fusion_plan(
        new_update_batches=new_update_batches,
        replay_update_batches=replay_update_batches,
        step_count=step_count,
        replay_loss_weight=float(cfg.replay_loss_weight),
    )
    paired_update_replay_plan_elapsed_seconds = max(
        0.0,
        time.perf_counter() - paired_update_replay_plan_started,
    )
    paired_update_replay_fusion_enabled = bool(
        paired_update_replay_fusion.get("enabled", False)
    )
    paired_sampled_vocab_precompute_started = time.perf_counter()
    if bool(cfg.paired_sampled_vocab_loss):
        (
            paired_sampled_vocab_inputs,
            paired_sampled_vocab_precompute,
        ) = _precompute_paired_update_replay_sampled_vocab(
            model,
            new_update_batches=new_update_batches,
            replay_update_batches=replay_update_batches,
            step_count=step_count,
        )
    else:
        paired_sampled_vocab_inputs = {}
        paired_sampled_vocab_precompute = {
            "surface": (
                "marulho_language_continual_paired_sampled_vocab_precompute.v1"
            ),
            "enabled": False,
            "reason": "paired_sampled_vocab_loss_disabled",
            "planned_optimizer_steps": int(step_count * len(new_update_batches)),
            "pair_count": 0,
            "device": str(model.device),
        }
    paired_sampled_vocab_precompute_elapsed_seconds = max(
        0.0,
        time.perf_counter() - paired_sampled_vocab_precompute_started,
    )
    pre_update_evaluation_started = time.perf_counter()
    old_before = evaluate_language_model(model, old_eval_runtime_batches)
    new_before = evaluate_language_model(model, new_eval_runtime_batches)
    replay_before = (
        evaluate_language_model(model, replay_update_batches)
        if replay_update_batches
        else None
    )
    pre_update_evaluation_elapsed_seconds = max(
        0.0,
        time.perf_counter() - pre_update_evaluation_started,
    )

    optimizer_setup_started = time.perf_counter()
    model.train()
    optimizers, optimizer_policy = _continual_optimizer_policy(model, cfg)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer_setup_elapsed_seconds = max(
        0.0,
        time.perf_counter() - optimizer_setup_started,
    )
    training_triton_stats_before = _language_training_triton_stats_snapshot()
    cuda_synchronized_before_timing_start = False
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
        cuda_synchronized_before_timing_start = True
    started = time.perf_counter()
    update_loss_sum: torch.Tensor | None = None
    update_loss_count = 0
    replay_loss_sum: torch.Tensor | None = None
    replay_loss_count = 0
    max_gradient_norm_tensor: torch.Tensor | None = None
    last_loss_evidence: dict[str, Any] = {}
    last_replay_loss_evidence: dict[str, Any] = {}
    last_update_memory_evidence: dict[str, Any] = {}
    last_replay_memory_evidence: dict[str, Any] = {}
    last_update_routing_evidence: dict[str, Any] = {}
    last_replay_routing_evidence: dict[str, Any] = {}
    update_memory_candidate_slots_scored = 0
    replay_memory_candidate_slots_scored = 0
    update_memory_observation_count = 0
    replay_memory_observation_count = 0
    last_update_batch_for_probe: LanguageBatch | None = None
    last_replay_batch_for_probe: LanguageBatch | None = None
    update_token_count = 0
    optimizer_step_count = 0
    gradient_clip_applied_step_count = 0
    gradient_clip_skipped_step_count = 0
    gradient_clip_interval = max(0, int(cfg.gradient_clip_interval))
    measured_update_loop_model_loss_calls = 0
    fused_update_replay_step_count = 0
    separate_replay_forward_loss_calls_avoided = 0
    paired_sampled_vocab_loss_fused_step_count = 0
    for step in range(step_count):
        for index, batch in enumerate(new_update_batches):
            optimizer_step_count += 1
            last_update_batch_for_probe = batch
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            replay_index = (
                (step * len(new_update_batches) + index) % replay_count
                if replay_update_batches
                else -1
            )
            replay_batch = (
                replay_update_batches[replay_index] if replay_update_batches else None
            )
            if (
                replay_batch is not None
                and bool(paired_update_replay_fusion_enabled)
            ):
                last_replay_batch_for_probe = replay_batch
                paired_sampled_vocab = paired_sampled_vocab_inputs.get(
                    (index, replay_index)
                )
                paired_result = model.next_token_loss_pair(
                    batch,
                    replay_batch,
                    replay_loss_weight=float(cfg.replay_loss_weight),
                    collect_telemetry=bool(cfg.collect_training_telemetry),
                    assume_no_sleeping_experts=assume_no_sleeping,
                    paired_sampled_vocab_ids=(
                        None if paired_sampled_vocab is None else paired_sampled_vocab[0]
                    ),
                    paired_sampled_target_ids=(
                        None if paired_sampled_vocab is None else paired_sampled_vocab[1]
                    ),
                    paired_sampled_target_positions=(
                        None if paired_sampled_vocab is None else paired_sampled_vocab[2]
                    ),
                )
                measured_update_loop_model_loss_calls += 1
                fused_update_replay_step_count += 1
                separate_replay_forward_loss_calls_avoided += 1
                if bool(paired_result.get("paired_sampled_vocab_loss_fused", False)):
                    paired_sampled_vocab_loss_fused_step_count += 1
                loss = paired_result["loss"]
                replay_loss = paired_result["replay_loss"]
            else:
                update_result = model.next_token_loss(
                    batch.input_ids,
                    batch.target_ids,
                    collect_telemetry=bool(cfg.collect_training_telemetry),
                    assume_no_sleeping_experts=assume_no_sleeping,
                    sampled_vocab_ids=batch.sampled_vocab_ids,
                    sampled_target_positions=batch.sampled_target_positions,
                    memory_candidate_ids=batch.memory_candidate_ids,
                    route_candidate_ids=batch.route_candidate_ids,
                    return_evidence=False,
                )
                measured_update_loop_model_loss_calls += 1
                loss = update_result["loss"]
                replay_loss = None
            update_candidate_slots_scored = _precomputed_memory_candidate_slots_scored(
                batch
            )
            if update_candidate_slots_scored > 0:
                update_memory_candidate_slots_scored += update_candidate_slots_scored
                update_memory_observation_count += 1
            update_token_count += int(batch.target_ids.numel())
            if replay_batch is not None:
                last_replay_batch_for_probe = replay_batch
                if replay_loss is None:
                    replay_result = model.next_token_loss(
                        replay_batch.input_ids,
                        replay_batch.target_ids,
                        collect_telemetry=bool(cfg.collect_training_telemetry),
                        assume_no_sleeping_experts=assume_no_sleeping,
                        sampled_vocab_ids=replay_batch.sampled_vocab_ids,
                        sampled_target_positions=replay_batch.sampled_target_positions,
                        memory_candidate_ids=replay_batch.memory_candidate_ids,
                        route_candidate_ids=replay_batch.route_candidate_ids,
                        return_evidence=False,
                    )
                    measured_update_loop_model_loss_calls += 1
                    replay_loss = replay_result["loss"]
                replay_candidate_slots_scored = (
                    _precomputed_memory_candidate_slots_scored(replay_batch)
                )
                if replay_candidate_slots_scored > 0:
                    replay_memory_candidate_slots_scored += replay_candidate_slots_scored
                    replay_memory_observation_count += 1
                detached_replay_loss = replay_loss.detach()
                replay_loss_sum = (
                    detached_replay_loss
                    if replay_loss_sum is None
                    else replay_loss_sum + detached_replay_loss
                )
                replay_loss_count += 1
                if not bool(paired_update_replay_fusion_enabled):
                    loss = loss + float(cfg.replay_loss_weight) * replay_loss
                update_token_count += int(replay_batch.target_ids.numel())
            loss.backward()
            should_clip_gradients = bool(
                float(cfg.max_grad_norm) > 0.0
                and gradient_clip_interval > 0
                and optimizer_step_count % gradient_clip_interval == 0
            )
            if should_clip_gradients:
                grad_norm = _clip_grad_norm_sparse_aware(
                    trainable_parameters,
                    max_norm=float(cfg.max_grad_norm),
                    device=model.device,
                )
                detached_grad_norm = grad_norm.detach()
                max_gradient_norm_tensor = (
                    detached_grad_norm
                    if max_gradient_norm_tensor is None
                    else torch.maximum(max_gradient_norm_tensor, detached_grad_norm)
                )
                gradient_clip_applied_step_count += 1
            else:
                gradient_clip_skipped_step_count += 1
            for optimizer in optimizers:
                optimizer.step()
            detached_update_loss = loss.detach()
            update_loss_sum = (
                detached_update_loss
                if update_loss_sum is None
                else update_loss_sum + detached_update_loss
            )
            update_loss_count += 1
    cuda_synchronized_before_timing_stop = False
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
        cuda_synchronized_before_timing_stop = True
    elapsed_seconds = max(0.0, time.perf_counter() - started)
    paired_update_replay_fusion = {
        **paired_update_replay_fusion,
        "actual_fused_steps": int(fused_update_replay_step_count),
        "measured_update_loop_model_loss_calls": int(
            measured_update_loop_model_loss_calls
        ),
        "separate_replay_forward_loss_calls_avoided": int(
            separate_replay_forward_loss_calls_avoided
        ),
        "paired_sampled_vocab_loss_fused_steps": int(
            paired_sampled_vocab_loss_fused_step_count
        ),
        "sampled_vocab_ce_loss_calls_avoided": int(
            paired_sampled_vocab_loss_fused_step_count
        ),
    }
    training_triton_stats_after = _language_training_triton_stats_snapshot()
    training_triton_stats_delta = _language_training_triton_stats_delta(
        training_triton_stats_before,
        training_triton_stats_after,
    )
    training_window_triton_accounting = _training_window_triton_accounting(
        training_triton_stats_delta
    )
    training_memory_slots_triton_after = training_triton_stats_after[
        "language_memory_slots_triton"
    ]
    training_window_memory_slot_triton_stats_delta = (
        training_triton_stats_delta["language_memory_slots_triton"]
    )
    training_window_memory_slot_triton_training_autograd_enabled = bool(
        training_memory_slots_triton_after.get(
            "triton_training_autograd_enabled",
            False,
        )
    )
    telemetry_probe_outside_measured_window = False
    post_window_update_probe_batch_tokens = 0
    post_window_replay_probe_batch_tokens = 0
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    if last_update_batch_for_probe is not None:
        update_probe_result = model.next_token_loss(
            last_update_batch_for_probe.input_ids,
            last_update_batch_for_probe.target_ids,
            collect_telemetry=bool(cfg.collect_training_telemetry),
            assume_no_sleeping_experts=assume_no_sleeping,
            sampled_vocab_ids=last_update_batch_for_probe.sampled_vocab_ids,
            sampled_target_positions=last_update_batch_for_probe.sampled_target_positions,
            memory_candidate_ids=last_update_batch_for_probe.memory_candidate_ids,
            route_candidate_ids=last_update_batch_for_probe.route_candidate_ids,
            return_evidence=True,
        )
        telemetry_probe_outside_measured_window = True
        post_window_update_probe_batch_tokens = int(
            last_update_batch_for_probe.target_ids.numel()
        )
        last_loss_evidence = dict(update_probe_result.get("loss_evidence") or {})
        last_update_memory_evidence = dict(
            (update_probe_result.get("telemetry") or {}).get("memory") or {}
        )
        last_update_routing_evidence = dict(
            (update_probe_result.get("telemetry") or {}).get("routing") or {}
        )
        del update_probe_result
    if last_replay_batch_for_probe is not None:
        replay_probe_result = model.next_token_loss(
            last_replay_batch_for_probe.input_ids,
            last_replay_batch_for_probe.target_ids,
            collect_telemetry=bool(cfg.collect_training_telemetry),
            assume_no_sleeping_experts=assume_no_sleeping,
            sampled_vocab_ids=last_replay_batch_for_probe.sampled_vocab_ids,
            sampled_target_positions=last_replay_batch_for_probe.sampled_target_positions,
            memory_candidate_ids=last_replay_batch_for_probe.memory_candidate_ids,
            route_candidate_ids=last_replay_batch_for_probe.route_candidate_ids,
            return_evidence=True,
        )
        telemetry_probe_outside_measured_window = True
        post_window_replay_probe_batch_tokens = int(
            last_replay_batch_for_probe.target_ids.numel()
        )
        last_replay_loss_evidence = dict(
            replay_probe_result.get("loss_evidence") or {}
        )
        last_replay_memory_evidence = dict(
            (replay_probe_result.get("telemetry") or {}).get("memory") or {}
        )
        last_replay_routing_evidence = dict(
            (replay_probe_result.get("telemetry") or {}).get("routing") or {}
        )
        del replay_probe_result
    mean_update_loss = (
        float((update_loss_sum / max(1, update_loss_count)).detach().cpu().item())
        if update_loss_sum is not None
        else 0.0
    )
    mean_replay_loss = (
        float((replay_loss_sum / max(1, replay_loss_count)).detach().cpu().item())
        if replay_loss_sum is not None
        else None
    )
    max_gradient_norm = (
        float(max_gradient_norm_tensor.detach().cpu().item())
        if max_gradient_norm_tensor is not None
        else 0.0
    )

    candidate_state = _clone_state_dict(model)
    candidate_hash = _state_dict_hash(candidate_state)
    post_update_evaluation_started = time.perf_counter()
    old_after = evaluate_language_model(model, old_eval_runtime_batches)
    new_after = evaluate_language_model(model, new_eval_runtime_batches)
    replay_after = (
        evaluate_language_model(model, replay_update_batches)
        if replay_update_batches
        else None
    )
    post_update_evaluation_elapsed_seconds = max(
        0.0,
        time.perf_counter() - post_update_evaluation_started,
    )
    new_domain_loss_delta = -_metric_delta(new_before, new_after, "heldout_loss")
    old_domain_forgetting = _metric_delta(old_before, old_after, "heldout_loss")
    replay_retention_delta = (
        _metric_delta(replay_before, replay_after, "heldout_loss")
        if replay_before is not None and replay_after is not None
        else 0.0
    )
    new_domain_improved = new_domain_loss_delta >= float(cfg.min_new_loss_improvement)
    forgetting_within_gate = old_domain_forgetting <= float(cfg.forgetting_tolerance)
    replay_within_gate = replay_retention_delta <= float(cfg.replay_retention_tolerance)
    rollback_required = (
        bool(cfg.rollback_on_forgetting)
        and (not forgetting_within_gate or not replay_within_gate)
    )
    rollback_applied = False
    rollback_elapsed_seconds = 0.0
    if rollback_required:
        rollback_started = time.perf_counter()
        model.load_state_dict(snapshot)
        rollback_elapsed_seconds = max(0.0, time.perf_counter() - rollback_started)
        rollback_applied = True

    final_state = _clone_state_dict(model)
    final_hash = _state_dict_hash(final_state)
    restore_verified = not rollback_applied or final_hash == snapshot_hash
    if was_training:
        model.train()
    else:
        model.eval()

    status = (
        "accepted_online_update"
        if new_domain_improved and forgetting_within_gate and replay_within_gate and not rollback_applied
        else "rolled_back_for_replay_or_forgetting"
        if rollback_applied
        else "needs_more_learning_evidence"
    )
    total_elapsed_seconds = max(0.0, time.perf_counter() - total_started)
    memory_slot_evidence = _memory_slot_learning_summary(
        model,
        update_memory=last_update_memory_evidence,
        replay_memory=last_replay_memory_evidence,
        training_window_memory_slot_triton_stats_delta=(
            training_window_memory_slot_triton_stats_delta
        ),
        training_window_memory_slot_triton_training_autograd_enabled=(
            training_window_memory_slot_triton_training_autograd_enabled
        ),
        update_candidate_slots_scored=update_memory_candidate_slots_scored,
        replay_candidate_slots_scored=replay_memory_candidate_slots_scored,
        update_observation_count=update_memory_observation_count,
        replay_observation_count=replay_memory_observation_count,
    )
    active_compute_evidence = _active_compute_learning_summary(
        model,
        loss_evidence=last_loss_evidence,
        update_routing=last_update_routing_evidence,
        replay_routing=last_replay_routing_evidence,
        memory_slots=memory_slot_evidence,
        update_token_count=update_token_count,
        measured_update_loop_model_loss_calls=measured_update_loop_model_loss_calls,
    )
    return {
        "artifact_kind": "marulho_language_continual_learning_window",
        "surface": "marulho_language_continual_learning_window.v1",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "active_language_path": model.config.active_language_path,
        "model_vocab_size": int(model.config.vocab_size),
        "generation_vocab_size": int(model.generation_vocab_size),
            "sampled_vocab_size": int(model.config.sampled_vocab_size),
            "paired_sampled_vocab_loss_enabled": bool(
                cfg.paired_sampled_vocab_loss
            ),
            "sparse_vocab_optimizer": bool(cfg.sparse_vocab_optimizer),
        "mutates_language_model_weights": not rollback_applied,
        "runtime_training": True,
        "status": status,
        "config": asdict(cfg),
        "memory_slots": memory_slot_evidence,
        "active_compute": active_compute_evidence,
        "old_domain_before": old_before,
        "old_domain_after": old_after,
        "new_domain_before": new_before,
        "new_domain_after": new_after,
        "replay_before": replay_before,
        "replay_after": replay_after,
        "learning_evidence": {
            "new_domain_loss_delta": float(new_domain_loss_delta),
            "new_domain_perplexity_delta": _metric_delta(
                new_before,
                new_after,
                "heldout_perplexity",
            ),
            "old_domain_forgetting": float(old_domain_forgetting),
            "old_domain_perplexity_delta": _metric_delta(
                old_before,
                old_after,
                "heldout_perplexity",
            ),
            "general_replay_retention_delta": float(replay_retention_delta),
            "spike_rate_delta": float(_spike_rate(new_after) - _spike_rate(new_before)),
            "memory_slots": memory_slot_evidence,
            "active_compute": active_compute_evidence,
            "training_window_memory_slot_triton_stats_delta": (
                training_window_memory_slot_triton_stats_delta
            ),
            "training_window_triton_accounting": training_window_triton_accounting,
            "training_window_memory_slot_triton_training_autograd_enabled": bool(
                training_window_memory_slot_triton_training_autograd_enabled
            ),
            "training_window_memory_slot_triton_autograd_used": bool(
                training_window_memory_slot_triton_stats_delta.get(
                    "triton_autograd_used",
                    False,
                )
            ),
            "update_batch_count": int(len(new_batches) * step_count),
            "replay_batch_count": int(len(replay_batches)),
            "sampled_vocab_precompute": {
                "surface": "marulho_language_continual_sampled_vocab_precompute.v1",
                "old_eval_batches": old_eval_sampled_vocab_precompute,
                "new_eval_batches": new_eval_sampled_vocab_precompute,
                "new_batches": new_sampled_vocab_precompute,
                "replay_batches": replay_sampled_vocab_precompute,
                "paired_update_replay_batches": paired_sampled_vocab_precompute,
            },
            "batch_device_staging": batch_device_staging,
            "optimizer_step_count": int(optimizer_step_count),
            "optimizer_policy": optimizer_policy,
            "dense_adamw_backend": str(cfg.dense_adamw_backend),
            "gradient_clip_mode": (
                "disabled"
                if float(cfg.max_grad_norm) <= 0.0 or gradient_clip_interval <= 0
                else (
                    "sparse_aware_device_norm_every_step"
                    if gradient_clip_interval == 1
                    else "sparse_aware_device_norm_every_n_steps"
                )
            ),
            "gradient_clip_interval": int(gradient_clip_interval),
            "gradient_clip_applied_step_count": int(gradient_clip_applied_step_count),
            "gradient_clip_skipped_step_count": int(gradient_clip_skipped_step_count),
            "gradient_norm_observed_step_count": int(gradient_clip_applied_step_count),
            "update_token_count": int(update_token_count),
            "elapsed_seconds": float(elapsed_seconds),
            "window_phase_timings": {
                "surface": "marulho_language_continual_window_phase_timings.v1",
                "state_snapshot_seconds": float(snapshot_elapsed_seconds),
                "sampled_vocab_precompute_seconds": float(
                    sampled_vocab_precompute_elapsed_seconds
                ),
                "paired_update_replay_plan_seconds": float(
                    paired_update_replay_plan_elapsed_seconds
                ),
                "paired_sampled_vocab_precompute_seconds": float(
                    paired_sampled_vocab_precompute_elapsed_seconds
                ),
                "batch_device_staging_seconds": float(
                    batch_device_staging_elapsed_seconds
                ),
                "pre_update_evaluation_seconds": float(
                    pre_update_evaluation_elapsed_seconds
                ),
                "optimizer_setup_seconds": float(optimizer_setup_elapsed_seconds),
                "update_seconds": float(elapsed_seconds),
                "post_update_evaluation_seconds": float(
                    post_update_evaluation_elapsed_seconds
                ),
                "rollback_seconds": float(rollback_elapsed_seconds),
                "total_window_seconds": float(total_elapsed_seconds),
            },
            "total_window_elapsed_seconds": float(total_elapsed_seconds),
            "tokens_per_second": (
                float(update_token_count) / elapsed_seconds
                if elapsed_seconds > 0.0
                else 0.0
            ),
            "total_window_tokens_per_second": (
                float(update_token_count) / total_elapsed_seconds
                if total_elapsed_seconds > 0.0
                else 0.0
            ),
            "mean_update_loss": mean_update_loss,
            "mean_replay_loss": mean_replay_loss,
            "max_gradient_norm": max_gradient_norm,
            "metric_readback_mode": "deferred_gpu_scalar_aggregation",
            "per_step_metric_cpu_sync": False,
            "measured_update_loop_caller_device_transfer_calls": 0,
            "hot_update_evidence_mode": "post_window_telemetry_probe",
            "per_step_evidence_dict_build": False,
            "paired_update_replay_fusion": paired_update_replay_fusion,
            "measured_update_loop_model_loss_calls": int(
                measured_update_loop_model_loss_calls
            ),
            "memory_slot_hot_update_evidence_mode": (
                "training_window_counter_delta_plus_post_window_probe"
            ),
            "per_step_memory_slot_stats_delta": False,
            "telemetry_probe_outside_measured_window": bool(
                telemetry_probe_outside_measured_window
            ),
            "post_window_update_probe_batch_tokens": int(
                post_window_update_probe_batch_tokens
            ),
            "post_window_replay_probe_batch_tokens": int(
                post_window_replay_probe_batch_tokens
            ),
            "cuda_synchronized_before_timing_start": bool(
                cuda_synchronized_before_timing_start
            ),
            "cuda_synchronized_before_timing_stop": bool(
                cuda_synchronized_before_timing_stop
            ),
            "sampled_vocab_training": bool(
                last_loss_evidence.get("sampled_vocab_training", False)
            ),
            "full_vocab_logits_materialized": bool(
                last_loss_evidence.get("full_vocab_logits_materialized", True)
            ),
            "loss_evidence": last_loss_evidence,
            "replay_loss_evidence": last_replay_loss_evidence,
            "training_telemetry_collected": bool(cfg.collect_training_telemetry),
            "candidate_parameter_delta_l2": _parameter_delta_l2(
                snapshot,
                candidate_state,
            ),
            "final_parameter_delta_l2": _parameter_delta_l2(snapshot, final_state),
            "device": str(model.device),
        },
        "rollback_evidence": {
            "snapshot_hash": snapshot_hash,
            "candidate_state_hash": candidate_hash,
            "final_state_hash": final_hash,
            "rollback_required": bool(rollback_required),
            "rollback_applied": bool(rollback_applied),
            "restore_verified": bool(restore_verified),
        },
        "promotion_gate": {
            "status": status,
            "eligible_for_online_learning_review": (
                bool(new_domain_improved)
                and bool(forgetting_within_gate)
                and bool(replay_within_gate)
                and not bool(rollback_applied)
            ),
            "new_domain_improved": bool(new_domain_improved),
            "old_domain_forgetting_within_tolerance": bool(forgetting_within_gate),
            "general_replay_retention_within_tolerance": bool(replay_within_gate),
            "forgetting_tolerance": float(cfg.forgetting_tolerance),
            "replay_retention_tolerance": float(cfg.replay_retention_tolerance),
            "rollback_available": True,
        },
    }
