"""Measure complete training-step impact from bounded LM memory slots."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from marulho.core.language_memory_slots_triton import (
    language_memory_slots_triton_stats,
    language_memory_slots_triton_stats_delta,
)
from marulho.core.language_sampled_vocab_ce_triton import (
    language_sampled_vocab_ce_triton_stats,
    language_sampled_vocab_ce_triton_stats_delta,
)
from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_training_experiment import DEFAULT_CORPUS
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    LanguageBatch,
    LanguageModelConfig,
    MarulhoLanguageModel,
    build_language_model_splits,
    precompute_sampled_vocab_batches,
)


SURFACE = "marulho_language_memory_slot_training_impact.v1"
ARTIFACT_KIND = "marulho_language_memory_slot_training_impact"


@dataclass(frozen=True)
class MemorySlotTrainingImpactConfig:
    vocab_size: int = 524288
    sampled_vocab_size: int = 1024
    embedding_dim: int = 64
    state_dim: int = 128
    expert_count: int = 16
    active_expert_count: int = 4
    route_candidate_count: int = 8
    expert_hidden_dim: int = 192
    adaptive_timestep_budget: int = 1
    recurrent_gradient_horizon: int = 8
    memory_slot_count: int = 1024
    bounded_memory_slot_candidate_count: int = 8
    active_memory_slot_count: int = 2
    sequence_length: int = 64
    batch_size: int = 16
    warmup_steps: int = 2
    repeats: int = 8
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    gradient_clip_interval: int = 8
    cuda_allow_tf32: bool = True
    cuda_float32_matmul_precision: str = "high"
    device: str = "auto"
    seed: int = 20260705


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device was requested but torch.cuda.is_available() is false")
    return resolved


def _sync_if_cuda(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    torch.cuda.synchronize(device)
    return True


def _cuda_memory(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(device),
            "allocated_mib": 0.0,
            "reserved_mib": 0.0,
            "peak_allocated_mib": 0.0,
            "peak_reserved_mib": 0.0,
        }
    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "allocated_mib": float(torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)),
        "reserved_mib": float(torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)),
        "peak_allocated_mib": float(
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        ),
        "peak_reserved_mib": float(
            torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
        ),
    }


def _cuda_math_policy_snapshot() -> dict[str, Any]:
    precision = (
        torch.get_float32_matmul_precision()
        if hasattr(torch, "get_float32_matmul_precision")
        else "unavailable"
    )
    return {
        "surface": "marulho_cuda_math_policy.v1",
        "cuda_available": bool(torch.cuda.is_available()),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": str(precision),
    }


def _apply_cuda_math_policy(
    device: torch.device,
    config: MemorySlotTrainingImpactConfig,
) -> dict[str, Any]:
    requested_precision = str(config.cuda_float32_matmul_precision)
    if requested_precision not in {"highest", "high", "medium"}:
        raise ValueError(
            "cuda_float32_matmul_precision must be one of: highest, high, medium"
        )
    before = _cuda_math_policy_snapshot()
    applied = False
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(config.cuda_allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(config.cuda_allow_tf32)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(requested_precision)
        applied = True
    active = _cuda_math_policy_snapshot()
    return {
        "surface": "marulho_cuda_math_policy_application.v1",
        "device": str(device),
        "applied": bool(applied),
        "requested_matmul_allow_tf32": bool(config.cuda_allow_tf32),
        "requested_cudnn_allow_tf32": bool(config.cuda_allow_tf32),
        "requested_float32_matmul_precision": requested_precision,
        "before": before,
        "active": active,
    }


def _restore_cuda_math_policy(snapshot: Mapping[str, Any]) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(
        snapshot.get("matmul_allow_tf32", False)
    )
    torch.backends.cudnn.allow_tf32 = bool(snapshot.get("cudnn_allow_tf32", True))
    precision = snapshot.get("float32_matmul_precision")
    if isinstance(precision, str) and hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision(precision)
        except RuntimeError:
            pass


def _model_config(
    config: MemorySlotTrainingImpactConfig,
    tokenizer: ByteLevelLanguageTokenizer,
    *,
    memory_slot_count: int,
    memory_slot_candidate_count: int,
) -> LanguageModelConfig:
    sparse_vocab_gradients = int(config.sampled_vocab_size) > 0
    return LanguageModelConfig(
        vocab_size=int(config.vocab_size),
        embedding_dim=int(config.embedding_dim),
        state_dim=int(config.state_dim),
        adaptive_timestep_budget=int(config.adaptive_timestep_budget),
        recurrent_gradient_horizon=max(0, int(config.recurrent_gradient_horizon)),
        expert_count=int(config.expert_count),
        active_expert_count=int(config.active_expert_count),
        route_candidate_count=int(config.route_candidate_count),
        expert_hidden_dim=int(config.expert_hidden_dim),
        sampled_vocab_size=int(config.sampled_vocab_size),
        sampled_vocab_sparse_lm_head_gradient=bool(sparse_vocab_gradients),
        sparse_token_embedding_gradients=bool(sparse_vocab_gradients),
        generation_vocab_size=int(tokenizer.vocab_size),
        memory_slot_count=max(0, int(memory_slot_count)),
        memory_slot_candidate_count=max(0, int(memory_slot_candidate_count)),
        active_memory_slot_count=max(1, int(config.active_memory_slot_count)),
    )


def _build_batch(
    tokenizer: ByteLevelLanguageTokenizer,
    config: MemorySlotTrainingImpactConfig,
    *,
    device: torch.device,
) -> tuple[LanguageBatch, dict[str, Any]]:
    split = build_language_model_splits(
        [DEFAULT_CORPUS],
        tokenizer,
        sequence_length=int(config.sequence_length),
        eval_fraction=0.20,
        stride=int(config.sequence_length),
        batch_size=int(config.batch_size),
        device=device,
    )
    return split.train[0], split.report


def _clone_base_state(
    config: MemorySlotTrainingImpactConfig,
    tokenizer: ByteLevelLanguageTokenizer,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(int(config.seed))
    base_model = MarulhoLanguageModel(
        _model_config(
            config,
            tokenizer,
            memory_slot_count=0,
            memory_slot_candidate_count=0,
        )
    )
    return {
        key: value.detach().clone()
        for key, value in base_model.state_dict().items()
    }


def _load_matching_state(
    model: MarulhoLanguageModel,
    base_state: Mapping[str, torch.Tensor],
) -> None:
    target_state = model.state_dict()
    for key, source_value in base_state.items():
        if key in target_state and target_state[key].shape == source_value.shape:
            target_state[key] = source_value.detach().clone().to(target_state[key].device)
    model.load_state_dict(target_state)


def _optimizer_policy(
    model: MarulhoLanguageModel,
    *,
    config: MemorySlotTrainingImpactConfig,
) -> tuple[list[torch.optim.Optimizer], str]:
    sparse_names = {"token_embedding.weight", "lm_head.weight"}
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
        optimizers.append(torch.optim.AdamW(dense_params, lr=float(config.learning_rate)))
    if sparse_params:
        optimizers.append(torch.optim.SparseAdam(sparse_params, lr=float(config.learning_rate)))
    return optimizers, "AdamW_dense_core_plus_SparseAdam_vocab_rows"


def _all_trainable_parameters(model: MarulhoLanguageModel) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _clip_grad_norm_sparse_aware(
    parameters: list[torch.nn.Parameter],
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


def _tensor_scalar(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(value.detach().cpu().item())


def _gradient_evidence(parameter: torch.nn.Parameter | None) -> dict[str, Any]:
    if parameter is None or parameter.grad is None:
        return {
            "present": False,
            "nonzero": False,
            "nonzero_count": 0,
            "max_abs": 0.0,
            "is_sparse": False,
        }
    grad = parameter.grad.detach()
    values = grad.coalesce().values() if grad.is_sparse else grad
    nonzero_count = int(torch.count_nonzero(values).detach().cpu().item())
    max_abs = float(values.float().abs().max().detach().cpu().item()) if values.numel() else 0.0
    return {
        "present": True,
        "nonzero": bool(nonzero_count > 0),
        "nonzero_count": nonzero_count,
        "max_abs": max_abs,
        "is_sparse": bool(grad.is_sparse),
    }


def _run_training_steps(
    model: MarulhoLanguageModel,
    optimizers: list[torch.optim.Optimizer],
    batch: LanguageBatch,
    *,
    config: MemorySlotTrainingImpactConfig,
    step_count: int,
) -> tuple[int, torch.Tensor | None, torch.Tensor | None, int, int, Mapping[str, Any]]:
    token_count = 0
    last_loss: torch.Tensor | None = None
    last_grad_norm: torch.Tensor | None = None
    last_result: Mapping[str, Any] = {}
    gradient_clip_applied = 0
    gradient_clip_skipped = 0
    assume_no_sleeping = bool(model.routed_experts.enabled)
    parameters = _all_trainable_parameters(model)
    gradient_clip_interval = max(0, int(config.gradient_clip_interval))
    for step_index in range(max(0, int(step_count))):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        result = model.next_token_loss(
            batch.input_ids.to(model.device),
            batch.target_ids.to(model.device),
            collect_telemetry=False,
            assume_no_sleeping_experts=assume_no_sleeping,
            sampled_vocab_ids=batch.sampled_vocab_ids,
            sampled_target_positions=batch.sampled_target_positions,
            memory_candidate_ids=batch.memory_candidate_ids,
            route_candidate_ids=batch.route_candidate_ids,
        )
        loss = result["loss"]
        loss.backward()
        should_clip = bool(
            float(config.max_grad_norm) > 0.0
            and gradient_clip_interval > 0
            and (step_index + 1) % gradient_clip_interval == 0
        )
        if should_clip:
            grad_norm = _clip_grad_norm_sparse_aware(
                parameters,
                max_norm=float(config.max_grad_norm),
                device=model.device,
            )
            gradient_clip_applied += 1
        else:
            grad_norm = torch.zeros((), device=model.device, dtype=torch.float32)
            gradient_clip_skipped += 1
        for optimizer in optimizers:
            optimizer.step()
        token_count += int(batch.target_ids.numel())
        last_loss = loss.detach()
        last_grad_norm = grad_norm.detach()
        last_result = result
    return (
        token_count,
        last_loss,
        last_grad_norm,
        gradient_clip_applied,
        gradient_clip_skipped,
        last_result,
    )


def _memory_summary(
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    memory = telemetry.get("memory") if isinstance(telemetry.get("memory"), Mapping) else {}
    return {
        "enabled": bool(memory.get("enabled", False)),
        "total_slots": int(memory.get("total_slots", 0) or 0),
        "candidate_slot_count": int(memory.get("candidate_slot_count", 0) or 0),
        "active_slots_per_token": int(memory.get("active_slots_per_token", 0) or 0),
        "candidate_slots_scored": int(memory.get("candidate_slots_scored", 0) or 0),
        "runs_all_slots": bool(memory.get("runs_all_slots", False)),
        "fallback_reason": memory.get("fallback_reason"),
        "candidate_id_source": memory.get("candidate_id_source"),
        "memory_gate_readback": bool(memory.get("memory_gate_readback", False)),
        "memory_device": memory.get("memory_device"),
        "active_parameters_per_token": int(
            memory.get("active_parameters_per_token", 0) or 0
        ),
        "memory_slot_initialization": memory.get("memory_slot_initialization"),
        "memory_slot_init_std": memory.get("memory_slot_init_std"),
        "memory_slot_retrieval_backend": memory.get("memory_slot_retrieval_backend"),
        "memory_slot_triton_stats_delta": (
            dict(memory.get("memory_slot_triton_stats_delta"))
            if isinstance(memory.get("memory_slot_triton_stats_delta"), Mapping)
            else {}
        ),
    }


def _run_arm(
    name: str,
    *,
    memory_slot_count: int,
    memory_slot_candidate_count: int,
    base_state: Mapping[str, torch.Tensor],
    batch: LanguageBatch,
    config: MemorySlotTrainingImpactConfig,
    tokenizer: ByteLevelLanguageTokenizer,
    device: torch.device,
    memory_triton_training_autograd: bool,
) -> dict[str, Any]:
    previous_memory_triton_training = os.environ.get(
        "MARULHO_LANGUAGE_MEMORY_SLOTS_TRITON_TRAINING"
    )
    os.environ["MARULHO_LANGUAGE_MEMORY_SLOTS_TRITON_TRAINING"] = (
        "1" if bool(memory_triton_training_autograd) else "0"
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(config.seed) + (1 if memory_slot_count > 0 else 0))
    model_config = _model_config(
        config,
        tokenizer,
        memory_slot_count=memory_slot_count,
        memory_slot_candidate_count=memory_slot_candidate_count,
    )
    model: MarulhoLanguageModel | None = None
    optimizers: list[torch.optim.Optimizer] = []
    try:
        model = MarulhoLanguageModel(model_config).to(device)
        _load_matching_state(model, base_state)
        cached_batches, sampled_vocab_precompute = precompute_sampled_vocab_batches(
            model,
            (batch,),
            assume_no_sleeping_experts=bool(model.routed_experts.enabled),
        )
        cached_batch = cached_batches[0]
        optimizers, optimizer_policy = _optimizer_policy(model, config=config)
        model.train()
        with torch.no_grad():
            telemetry_probe = model.next_token_loss(
                cached_batch.input_ids.to(model.device),
                cached_batch.target_ids.to(model.device),
                collect_telemetry=True,
                assume_no_sleeping_experts=bool(model.routed_experts.enabled),
                sampled_vocab_ids=cached_batch.sampled_vocab_ids,
                sampled_target_positions=cached_batch.sampled_target_positions,
                memory_candidate_ids=cached_batch.memory_candidate_ids,
                route_candidate_ids=cached_batch.route_candidate_ids,
            )
        initial_memory = _memory_summary(telemetry_probe.get("telemetry", {}))
        memory_slot_nonzero_count = (
            0
            if model.memory_slots is None
            else int(torch.count_nonzero(model.memory_slots.detach()).cpu().item())
        )
        initial_memory_slot_gate_value = (
            None
            if model.memory_slot_gate is None
            else float(model.memory_slot_gate.detach().cpu().item())
        )
        warmup = _run_training_steps(
            model,
            optimizers,
            cached_batch,
            config=config,
            step_count=int(config.warmup_steps),
        )
        warmup_tokens = warmup[0]
        sampled_vocab_ce_stats_before = language_sampled_vocab_ce_triton_stats()
        memory_slots_stats_before = language_memory_slots_triton_stats()
        cuda_synchronized_before_timing_start = _sync_if_cuda(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        (
            token_count,
            last_loss,
            last_grad_norm,
            gradient_clip_applied,
            gradient_clip_skipped,
            last_result,
        ) = _run_training_steps(
            model,
            optimizers,
            cached_batch,
            config=config,
            step_count=int(config.repeats),
        )
        cuda_synchronized_before_timing_stop = _sync_if_cuda(device)
        sampled_vocab_ce_stats_delta = language_sampled_vocab_ce_triton_stats_delta(
            sampled_vocab_ce_stats_before,
            language_sampled_vocab_ce_triton_stats(),
        )
        memory_slots_stats_delta = language_memory_slots_triton_stats_delta(
            memory_slots_stats_before,
            language_memory_slots_triton_stats(),
        )
        elapsed = max(0.0, time.perf_counter() - started)
        telemetry = (
            last_result.get("telemetry")
            if isinstance(last_result.get("telemetry"), Mapping)
            else {}
        )
        memory = _memory_summary(telemetry)
        loss_evidence = dict(last_result.get("loss_evidence", {}))
        memory_slot_gate_gradient = _gradient_evidence(model.memory_slot_gate)
        memory_slots_gradient = _gradient_evidence(model.memory_slots)
        memory_slot_gate_value_after_training = (
            None
            if model.memory_slot_gate is None
            else float(model.memory_slot_gate.detach().cpu().item())
        )
        return {
            "surface": "marulho_language_memory_slot_training_arm.v1",
            "name": name,
            "success": True,
            "failure_reason": None,
            "model_config": asdict(model_config),
            "optimizer_policy": optimizer_policy,
            "sampled_vocab_precompute": sampled_vocab_precompute,
            "warmup_steps": int(config.warmup_steps),
            "warmup_tokens": int(warmup_tokens),
            "measured_steps": int(config.repeats),
            "token_count": int(token_count),
            "tokens_per_optimizer_step": int(cached_batch.target_ids.numel()),
            "elapsed_seconds": elapsed,
            "tokens_per_second": (
                float(token_count) / elapsed if elapsed > 0.0 else 0.0
            ),
            "loss": _tensor_scalar(last_loss),
            "gradient_norm": _tensor_scalar(last_grad_norm),
            "gradient_clip_interval": int(config.gradient_clip_interval),
            "gradient_clip_applied_steps": int(gradient_clip_applied),
            "gradient_clip_skipped_steps": int(gradient_clip_skipped),
            "loss_kind": str(last_result.get("loss_kind")),
            "loss_evidence": loss_evidence,
            "sampled_vocab_ce_triton_stats_delta": sampled_vocab_ce_stats_delta,
            "training_window_memory_slot_triton_stats_delta": (
                memory_slots_stats_delta
            ),
            "memory_triton_training_autograd_requested": bool(
                memory_triton_training_autograd
            ),
            "full_vocab_logits_materialized": bool(
                loss_evidence.get("full_vocab_logits_materialized", True)
            ),
            "sampled_vocab_training": bool(
                loss_evidence.get("sampled_vocab_training", False)
            ),
            "initial_memory": initial_memory,
            "memory": memory,
            "memory_enabled": bool(memory.get("enabled", False)),
            "total_slots": int(memory.get("total_slots", 0) or 0),
            "candidate_slot_count": int(memory.get("candidate_slot_count", 0) or 0),
            "active_slots_per_token": int(memory.get("active_slots_per_token", 0) or 0),
            "candidate_slots_scored": int(memory.get("candidate_slots_scored", 0) or 0),
            "runs_all_slots": bool(memory.get("runs_all_slots", False)),
            "memory_fallback_reason": memory.get("fallback_reason"),
            "candidate_id_source": memory.get("candidate_id_source"),
            "memory_gate_readback": bool(memory.get("memory_gate_readback", False)),
            "memory_slot_retrieval_backend": memory.get(
                "memory_slot_retrieval_backend"
            ),
            "memory_slot_triton_stats_delta": memory.get(
                "memory_slot_triton_stats_delta",
                {},
            ),
            "memory_slot_nonzero_count": int(memory_slot_nonzero_count),
            "initial_memory_slot_gate_value": initial_memory_slot_gate_value,
            "memory_slot_trainable_neutral_initialization": bool(
                memory_slot_nonzero_count > 0
                and initial_memory_slot_gate_value == 0.0
            ),
            "memory_slot_gate_value_after_training": memory_slot_gate_value_after_training,
            "memory_slot_gate_gradient": memory_slot_gate_gradient,
            "memory_slots_gradient": memory_slots_gradient,
            "cuda_synchronized_before_timing_start": bool(
                cuda_synchronized_before_timing_start
            ),
            "cuda_synchronized_before_timing_stop": bool(
                cuda_synchronized_before_timing_stop
            ),
            "device": str(device),
            "cuda_memory": _cuda_memory(device),
        }
    except RuntimeError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "surface": "marulho_language_memory_slot_training_arm.v1",
            "name": name,
            "success": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "model_config": asdict(model_config),
            "optimizer_policy": "AdamW_dense_core_plus_SparseAdam_vocab_rows",
            "warmup_steps": int(config.warmup_steps),
            "measured_steps": int(config.repeats),
            "token_count": 0,
            "elapsed_seconds": 0.0,
            "tokens_per_second": 0.0,
            "loss": None,
            "gradient_norm": None,
            "loss_kind": None,
            "loss_evidence": {},
            "sampled_vocab_ce_triton_stats_delta": {},
            "training_window_memory_slot_triton_stats_delta": {},
            "memory_triton_training_autograd_requested": bool(
                memory_triton_training_autograd
            ),
            "full_vocab_logits_materialized": False,
            "sampled_vocab_training": True,
            "memory_enabled": False,
            "total_slots": 0,
            "candidate_slot_count": 0,
            "active_slots_per_token": 0,
            "candidate_slots_scored": 0,
            "runs_all_slots": False,
            "memory_fallback_reason": f"{type(exc).__name__}: {exc}",
            "candidate_id_source": None,
            "memory_gate_readback": False,
            "memory_slot_retrieval_backend": None,
            "memory_slot_triton_stats_delta": {},
            "memory_slot_nonzero_count": 0,
            "initial_memory_slot_gate_value": None,
            "memory_slot_trainable_neutral_initialization": False,
            "memory_slot_gate_value_after_training": None,
            "memory_slot_gate_gradient": _gradient_evidence(None),
            "memory_slots_gradient": _gradient_evidence(None),
            "device": str(device),
            "cuda_memory": _cuda_memory(device),
        }
    finally:
        if previous_memory_triton_training is None:
            os.environ.pop("MARULHO_LANGUAGE_MEMORY_SLOTS_TRITON_TRAINING", None)
        else:
            os.environ[
                "MARULHO_LANGUAGE_MEMORY_SLOTS_TRITON_TRAINING"
            ] = previous_memory_triton_training
        del optimizers
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _comparison(
    control: Mapping[str, Any],
    bounded: Mapping[str, Any],
    triton_training: Mapping[str, Any],
) -> dict[str, Any]:
    control_success = bool(control.get("success"))
    bounded_success = bool(bounded.get("success"))
    control_tps = float(control.get("tokens_per_second", 0.0) or 0.0)
    bounded_tps = float(bounded.get("tokens_per_second", 0.0) or 0.0)
    triton_training_success = bool(triton_training.get("success"))
    triton_training_tps = float(
        triton_training.get("tokens_per_second", 0.0) or 0.0
    )
    bounded_peak = float(
        (bounded.get("cuda_memory") or {}).get("peak_allocated_mib", 0.0)
        if isinstance(bounded.get("cuda_memory"), Mapping)
        else 0.0
    )
    control_peak = float(
        (control.get("cuda_memory") or {}).get("peak_allocated_mib", 0.0)
        if isinstance(control.get("cuda_memory"), Mapping)
        else 0.0
    )
    bounded_avoids_all_slot_scan = bool(
        bounded_success
        and bounded.get("memory_enabled")
        and int(bounded.get("candidate_slot_count", 0) or 0)
        < int(bounded.get("total_slots", 0) or 0)
        and not bool(bounded.get("runs_all_slots", False))
    )
    gate_gradient = (
        bounded.get("memory_slot_gate_gradient")
        if isinstance(bounded.get("memory_slot_gate_gradient"), Mapping)
        else {}
    )
    slot_gradient = (
        bounded.get("memory_slots_gradient")
        if isinstance(bounded.get("memory_slots_gradient"), Mapping)
        else {}
    )
    sampled_loss = (
        bounded_success
        and bool(bounded.get("sampled_vocab_training", False))
        and not bool(bounded.get("full_vocab_logits_materialized", True))
    )
    triton_training_stats = (
        triton_training.get("training_window_memory_slot_triton_stats_delta")
        if isinstance(
            triton_training.get("training_window_memory_slot_triton_stats_delta"),
            Mapping,
        )
        else {}
    )
    triton_training_used = bool(
        triton_training_stats.get("triton_autograd_used", False)
    )
    if control_success and bounded_success and bounded_avoids_all_slot_scan:
        evidence_status = "measured_bounded_memory_slot_training_impact"
    elif control_success and bounded_success:
        evidence_status = "measured_memory_slot_training_without_bounded_retrieval"
    else:
        evidence_status = "memory_slot_training_impact_measurement_failed"
    return {
        "surface": "marulho_language_memory_slot_training_comparison.v1",
        "control_success": control_success,
        "bounded_success": bounded_success,
        "control_tokens_per_second": control_tps,
        "bounded_tokens_per_second": bounded_tps,
        "triton_training_success": triton_training_success,
        "triton_training_tokens_per_second": triton_training_tps,
        "bounded_vs_control_tokens_per_second_ratio": _ratio(bounded_tps, control_tps),
        "triton_training_vs_control_tokens_per_second_ratio": _ratio(
            triton_training_tps,
            control_tps,
        ),
        "triton_training_vs_bounded_tokens_per_second_ratio": _ratio(
            triton_training_tps,
            bounded_tps,
        ),
        "control_peak_cuda_allocated_mib": control_peak,
        "bounded_peak_cuda_allocated_mib": bounded_peak,
        "bounded_vs_control_peak_cuda_allocated_ratio": _ratio(
            bounded_peak,
            control_peak,
        ),
        "bounded_memory_enabled": bool(bounded.get("memory_enabled", False)),
        "bounded_avoids_all_slot_scan": bounded_avoids_all_slot_scan,
        "bounded_candidate_slots_scored_per_step": int(
            bounded.get("candidate_slots_scored", 0) or 0
        ),
        "bounded_candidate_slot_count": int(
            bounded.get("candidate_slot_count", 0) or 0
        ),
        "bounded_total_slots": int(bounded.get("total_slots", 0) or 0),
        "bounded_active_slots_per_token": int(
            bounded.get("active_slots_per_token", 0) or 0
        ),
        "memory_gate_readback": bool(bounded.get("memory_gate_readback", False)),
        "bounded_trainable_neutral_initialization": bool(
            bounded.get("memory_slot_trainable_neutral_initialization", False)
        ),
        "bounded_memory_slot_gate_gradient_nonzero": bool(
            gate_gradient.get("nonzero", False)
        ),
        "bounded_memory_slots_gradient_nonzero": bool(
            slot_gradient.get("nonzero", False)
        ),
        "bounded_sampled_vocab_loss_without_full_logits": sampled_loss,
        "bounded_memory_slot_retrieval_backend": bounded.get(
            "memory_slot_retrieval_backend"
        ),
        "triton_training_memory_slot_retrieval_backend": triton_training.get(
            "memory_slot_retrieval_backend"
        ),
        "triton_training_autograd_used": triton_training_used,
        "triton_training_autograd_faster_than_bounded": bool(
            triton_training_success
            and bounded_success
            and triton_training_tps > bounded_tps
        ),
        "evidence_status": evidence_status,
    }


def run_language_memory_slot_training_impact(
    *,
    output_path: str | Path,
    config: MemorySlotTrainingImpactConfig | None = None,
) -> dict[str, Any]:
    cfg = config or MemorySlotTrainingImpactConfig()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelLanguageTokenizer()
    if int(cfg.vocab_size) < int(tokenizer.vocab_size):
        raise ValueError("vocab_size must be at least the tokenizer vocab size")
    if int(cfg.sampled_vocab_size) <= 0:
        raise ValueError("sampled_vocab_size must be positive")
    if int(cfg.sampled_vocab_size) >= int(cfg.vocab_size):
        raise ValueError("sampled_vocab_size must be smaller than vocab_size")
    if int(cfg.memory_slot_count) <= 1:
        raise ValueError("memory_slot_count must be greater than one")
    if int(cfg.bounded_memory_slot_candidate_count) >= int(cfg.memory_slot_count):
        raise ValueError(
            "bounded_memory_slot_candidate_count must be smaller than memory_slot_count"
        )
    device = _resolve_device(str(cfg.device))
    cuda_math_policy = _apply_cuda_math_policy(device, cfg)
    try:
        batch, split_report = _build_batch(tokenizer, cfg, device=device)
        base_state = _clone_base_state(cfg, tokenizer)
        control_report = _run_arm(
            "memory_slots_disabled_control",
            memory_slot_count=0,
            memory_slot_candidate_count=0,
            base_state=base_state,
            batch=batch,
            config=cfg,
            tokenizer=tokenizer,
            device=device,
            memory_triton_training_autograd=False,
        )
        bounded_report = _run_arm(
            "bounded_memory_slots_enabled",
            memory_slot_count=int(cfg.memory_slot_count),
            memory_slot_candidate_count=int(cfg.bounded_memory_slot_candidate_count),
            base_state=base_state,
            batch=batch,
            config=cfg,
            tokenizer=tokenizer,
            device=device,
            memory_triton_training_autograd=False,
        )
        triton_training_report = _run_arm(
            "bounded_memory_slots_triton_training_autograd",
            memory_slot_count=int(cfg.memory_slot_count),
            memory_slot_candidate_count=int(cfg.bounded_memory_slot_candidate_count),
            base_state=base_state,
            batch=batch,
            config=cfg,
            tokenizer=tokenizer,
            device=device,
            memory_triton_training_autograd=True,
        )
        comparison = _comparison(control_report, bounded_report, triton_training_report)
        report = {
            "artifact_kind": ARTIFACT_KIND,
            "surface": SURFACE,
            "output_path": str(output),
            "owned_by_marulho": True,
            "external_llm_used": False,
            "loads_external_checkpoint": False,
            "active_language_path": "marulho_lm_head",
            "config": asdict(cfg),
            "cuda_math_policy": cuda_math_policy,
            "tokenizer": tokenizer.state_dict(),
            "model_vocab_size": int(cfg.vocab_size),
            "tokenizer_vocab_size": int(tokenizer.vocab_size),
            "generation_vocab_size": int(tokenizer.vocab_size),
            "padded_vocab_rows": int(cfg.vocab_size) - int(tokenizer.vocab_size),
            "batch": {
                "sequence_length": int(cfg.sequence_length),
                "batch_size": int(cfg.batch_size),
                "tokens_per_optimizer_step": int(batch.target_ids.numel()),
                "input_device": str(batch.input_ids.device),
                "target_device": str(batch.target_ids.device),
            },
            "split": split_report,
            "arms": {
                "memory_slots_disabled_control": control_report,
                "bounded_memory_slots_enabled": bounded_report,
                "bounded_memory_slots_triton_training_autograd": triton_training_report,
            },
            "comparison": comparison,
            "review": {
                "complete_training_step_impact": True,
                "includes_forward_backward_and_optimizer_step": bool(
                    bounded_report.get("success")
                ),
                "includes_memory_slot_gradient_evidence": True,
                "not_kernel_microbench_only": True,
                "sampled_loss_avoids_full_vocab_logits": bool(
                    comparison["bounded_sampled_vocab_loss_without_full_logits"]
                ),
                "uses_sparse_vocab_optimizer": (
                    bounded_report.get("optimizer_policy")
                    == "AdamW_dense_core_plus_SparseAdam_vocab_rows"
                ),
                "uses_precomputed_sampled_vocab_hot_window": bool(
                    (
                        bounded_report.get("sampled_vocab_precompute")
                        if isinstance(
                            bounded_report.get("sampled_vocab_precompute"), Mapping
                        )
                        else {}
                    ).get("hot_update_window_precomputed", False)
                ),
                "bounded_memory_slots_trainable": bool(
                    comparison["bounded_memory_slot_gate_gradient_nonzero"]
                ),
                "bounded_memory_slots_receive_gradient_after_gate_update": bool(
                    comparison["bounded_memory_slots_gradient_nonzero"]
                ),
                "compares_triton_training_autograd_backend": True,
                "triton_training_autograd_used": bool(
                    comparison["triton_training_autograd_used"]
                ),
                "triton_training_autograd_faster_than_bounded": bool(
                    comparison["triton_training_autograd_faster_than_bounded"]
                ),
                "memory_gate_readback": bool(comparison["memory_gate_readback"]),
                "promotes_hot_path": False,
                "promotes_runtime_claim": False,
                "promotes_generation_quality_claim": False,
                "next_experiment": (
                    "profile sustained generation and longer online-learning windows "
                    "with trained memory gates before promoting memory-slot growth"
                ),
            },
            "promotion_gate": {
                "training_impact_available": bool(
                    comparison["control_success"] and comparison["bounded_success"]
                ),
                "bounded_memory_slots_enabled": bool(
                    comparison["bounded_memory_enabled"]
                ),
                "bounded_avoids_all_slot_scan": bool(
                    comparison["bounded_avoids_all_slot_scan"]
                ),
                "sampled_vocab_training_without_full_logits": bool(
                    comparison["bounded_sampled_vocab_loss_without_full_logits"]
                ),
                "trainable_neutral_initialization": bool(
                    comparison["bounded_trainable_neutral_initialization"]
                ),
                "memory_gate_gradient_nonzero": bool(
                    comparison["bounded_memory_slot_gate_gradient_nonzero"]
                ),
                "memory_slots_gradient_nonzero_after_gate_update": bool(
                    comparison["bounded_memory_slots_gradient_nonzero"]
                ),
                "complete_training_step_impact_available": bool(
                    comparison["control_success"] and comparison["bounded_success"]
                ),
                "triton_training_autograd_measured": bool(
                    comparison["triton_training_success"]
                ),
                "triton_training_autograd_used": bool(
                    comparison["triton_training_autograd_used"]
                ),
                "triton_training_autograd_faster_than_bounded": bool(
                    comparison["triton_training_autograd_faster_than_bounded"]
                ),
                "promotes_hot_path": False,
                "promotes_runtime_claim": False,
            },
        }
        write_json_report_with_readme(output, report)
        return report
    finally:
        before_policy = cuda_math_policy.get("before")
        if isinstance(before_policy, Mapping):
            _restore_cuda_math_policy(before_policy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=524288)
    parser.add_argument("--sampled-vocab-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--expert-count", type=int, default=16)
    parser.add_argument("--active-expert-count", type=int, default=4)
    parser.add_argument("--route-candidate-count", type=int, default=8)
    parser.add_argument("--expert-hidden-dim", type=int, default=192)
    parser.add_argument("--adaptive-timestep-budget", type=int, default=1)
    parser.add_argument("--recurrent-gradient-horizon", type=int, default=8)
    parser.add_argument("--memory-slot-count", type=int, default=1024)
    parser.add_argument("--bounded-memory-slot-candidate-count", type=int, default=8)
    parser.add_argument("--active-memory-slot-count", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-clip-interval", type=int, default=8)
    parser.add_argument("--disable-cuda-tf32", action="store_true")
    parser.add_argument(
        "--cuda-float32-matmul-precision",
        default="high",
        choices=("highest", "high", "medium"),
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = MemorySlotTrainingImpactConfig(
        vocab_size=args.vocab_size,
        sampled_vocab_size=args.sampled_vocab_size,
        embedding_dim=args.embedding_dim,
        state_dim=args.state_dim,
        expert_count=args.expert_count,
        active_expert_count=args.active_expert_count,
        route_candidate_count=args.route_candidate_count,
        expert_hidden_dim=args.expert_hidden_dim,
        adaptive_timestep_budget=args.adaptive_timestep_budget,
        recurrent_gradient_horizon=args.recurrent_gradient_horizon,
        memory_slot_count=args.memory_slot_count,
        bounded_memory_slot_candidate_count=args.bounded_memory_slot_candidate_count,
        active_memory_slot_count=args.active_memory_slot_count,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        repeats=args.repeats,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        gradient_clip_interval=max(0, int(args.gradient_clip_interval)),
        cuda_allow_tf32=not bool(args.disable_cuda_tf32),
        cuda_float32_matmul_precision=args.cuda_float32_matmul_precision,
        device=args.device,
    )
    report = run_language_memory_slot_training_impact(
        output_path=args.output,
        config=config,
    )
    comparison = report["comparison"]
    print(
        "wrote "
        f"{args.output} bounded_tps={comparison['bounded_tokens_per_second']:.3f} "
        "triton_training_tps="
        f"{comparison['triton_training_tokens_per_second']:.3f} "
        f"control_tps={comparison['control_tokens_per_second']:.3f} "
        f"ratio={comparison['bounded_vs_control_tokens_per_second_ratio']} "
        "triton_vs_bounded="
        f"{comparison['triton_training_vs_bounded_tokens_per_second_ratio']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
