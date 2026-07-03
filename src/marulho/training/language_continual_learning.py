from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import time
from typing import Any, Mapping, Sequence

import torch

from marulho.training.language_model import (
    LanguageBatch,
    MarulhoLanguageModel,
    evaluate_language_model,
)


@dataclass(frozen=True)
class LanguageContinualLearningConfig:
    learning_rate: float = 1e-3
    max_steps: int = 1
    replay_loss_weight: float = 0.25
    max_grad_norm: float = 1.0
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
    snapshot = _clone_state_dict(model)
    snapshot_hash = _state_dict_hash(snapshot)
    old_before = evaluate_language_model(model, old_eval_batches)
    new_before = evaluate_language_model(model, new_eval_batches)
    replay_before = (
        evaluate_language_model(model, replay_batches)
        if replay_batches
        else None
    )

    was_training = model.training
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate))
    started = time.perf_counter()
    update_losses: list[float] = []
    replay_losses: list[float] = []
    grad_norms: list[float] = []
    update_token_count = 0
    step_count = max(1, int(cfg.max_steps))
    replay_count = len(replay_batches)
    for step in range(step_count):
        for index, batch in enumerate(new_batches):
            optimizer.zero_grad(set_to_none=True)
            update_result = model.next_token_loss(
                batch.input_ids.to(model.device),
                batch.target_ids.to(model.device),
            )
            loss = update_result["loss"]
            update_token_count += int(batch.target_ids.numel())
            if replay_batches:
                replay_batch = replay_batches[(step * len(new_batches) + index) % replay_count]
                replay_result = model.next_token_loss(
                    replay_batch.input_ids.to(model.device),
                    replay_batch.target_ids.to(model.device),
                )
                replay_loss = replay_result["loss"]
                replay_losses.append(float(replay_loss.detach().cpu().item()))
                loss = loss + float(cfg.replay_loss_weight) * replay_loss
                update_token_count += int(replay_batch.target_ids.numel())
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(cfg.max_grad_norm),
            )
            grad_norms.append(float(grad_norm.detach().cpu().item()))
            optimizer.step()
            update_losses.append(float(loss.detach().cpu().item()))
    elapsed_seconds = max(0.0, time.perf_counter() - started)

    candidate_state = _clone_state_dict(model)
    candidate_hash = _state_dict_hash(candidate_state)
    old_after = evaluate_language_model(model, old_eval_batches)
    new_after = evaluate_language_model(model, new_eval_batches)
    replay_after = (
        evaluate_language_model(model, replay_batches)
        if replay_batches
        else None
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
    if rollback_required:
        model.load_state_dict(snapshot)
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
    return {
        "artifact_kind": "marulho_language_continual_learning_window",
        "surface": "marulho_language_continual_learning_window.v1",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "active_language_path": model.config.active_language_path,
        "mutates_language_model_weights": not rollback_applied,
        "runtime_training": True,
        "status": status,
        "config": asdict(cfg),
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
            "update_batch_count": int(len(new_batches) * step_count),
            "replay_batch_count": int(len(replay_batches)),
            "update_token_count": int(update_token_count),
            "elapsed_seconds": float(elapsed_seconds),
            "tokens_per_second": (
                float(update_token_count) / elapsed_seconds
                if elapsed_seconds > 0.0
                else 0.0
            ),
            "mean_update_loss": (
                float(sum(update_losses) / len(update_losses)) if update_losses else 0.0
            ),
            "mean_replay_loss": (
                float(sum(replay_losses) / len(replay_losses)) if replay_losses else None
            ),
            "max_gradient_norm": max(grad_norms) if grad_norms else 0.0,
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
