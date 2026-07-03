from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.training.language_continual_learning import (
    LanguageContinualLearningConfig,
    run_language_continual_learning_window,
)
from marulho.training.language_model import (
    LanguageBatch,
    MarulhoLanguageModel,
    evaluate_language_model,
    save_language_model_checkpoint,
)
from marulho.training.language_structural_plasticity import (
    LanguageStructuralPlasticityConfig,
    apply_language_structural_plasticity_transaction,
    build_language_structural_plasticity_proposal,
)


@dataclass(frozen=True)
class LanguageCheckpointEvolutionConfig:
    max_child_loss_delta: float = 0.05
    max_old_domain_forgetting: float = 0.10
    require_child_learning: bool = True
    allow_structural_growth: bool = True
    operator_approved_child_growth: bool = True


def _state_hash(model: MarulhoLanguageModel) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _clone_model(model: MarulhoLanguageModel) -> MarulhoLanguageModel:
    clone = MarulhoLanguageModel(model.config).to(model.device)
    clone.load_state_dict(
        {
            key: value.detach().clone().to(model.device)
            for key, value in model.state_dict().items()
        }
    )
    clone.train(model.training)
    return clone


def _load_checkpoint_state_hash(path: str | Path) -> str:
    payload = torch.load(Path(path), map_location="cpu")
    digest = hashlib.sha256()
    for key, value in sorted(payload["model_state"].items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def run_language_checkpoint_evolution(
    parent_model: MarulhoLanguageModel,
    tokenizer: ByteLevelLanguageTokenizer,
    *,
    eval_batches: Sequence[LanguageBatch],
    child_train_batches: Sequence[LanguageBatch],
    child_new_eval_batches: Sequence[LanguageBatch],
    replay_batches: Sequence[LanguageBatch] = (),
    checkpoint_dir: str | Path,
    config: LanguageCheckpointEvolutionConfig | None = None,
    learning_config: LanguageContinualLearningConfig | None = None,
    structural_config: LanguageStructuralPlasticityConfig | None = None,
) -> tuple[MarulhoLanguageModel, dict[str, Any]]:
    if not eval_batches:
        raise ValueError("eval_batches must not be empty")
    if not child_train_batches:
        raise ValueError("child_train_batches must not be empty")
    if not child_new_eval_batches:
        raise ValueError("child_new_eval_batches must not be empty")
    cfg = config or LanguageCheckpointEvolutionConfig()
    output_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lineage_id = uuid4().hex
    parent_training_mode_before = bool(parent_model.training)
    parent_hash_before = _state_hash(parent_model)
    parent_checkpoint = save_language_model_checkpoint(
        output_dir / f"language-parent-{lineage_id}.pt",
        parent_model,
        tokenizer,
        metadata={
            "lineage_id": lineage_id,
            "checkpoint_role": "parent",
        },
    )
    child = _clone_model(parent_model)
    child_checkpoint = save_language_model_checkpoint(
        output_dir / f"language-child-initial-{lineage_id}.pt",
        child,
        tokenizer,
        metadata={
            "lineage_id": lineage_id,
            "checkpoint_role": "child_initial",
            "parent_checkpoint": str(parent_checkpoint),
        },
    )
    parent_eval = evaluate_language_model(parent_model, eval_batches)
    child_learning = run_language_continual_learning_window(
        child,
        new_batches=child_train_batches,
        old_eval_batches=eval_batches,
        new_eval_batches=child_new_eval_batches,
        replay_batches=replay_batches,
        config=learning_config,
    )
    structural_proposal: dict[str, Any] | None = None
    structural_transaction: dict[str, Any] | None = None
    if cfg.allow_structural_growth and child.config.expert_count > 0:
        routing = child_learning.get("new_domain_after", {}).get("spike_telemetry", {})
        routing_evidence = (
            routing.get("routing") if isinstance(routing, Mapping) else None
        )
        if not isinstance(routing_evidence, Mapping):
            routing_evidence = {
                "total_columns": int(child.config.expert_count),
                "active_columns": int(child.config.expert_count),
                "runs_all_columns": True,
            }
        structural_proposal = build_language_structural_plasticity_proposal(
            child,
            routing_evidence=routing_evidence,
            learning_evidence=child_learning.get("learning_evidence", {}),
            config=structural_config,
        )
        child, structural_transaction = apply_language_structural_plasticity_transaction(
            child,
            structural_proposal,
            eval_batches=eval_batches,
            checkpoint_path=output_dir / f"language-child-structure-baseline-{lineage_id}.pt",
            operator_approved=bool(cfg.operator_approved_child_growth),
            config=structural_config,
        )

    child_eval = evaluate_language_model(child, eval_batches)
    child_final_checkpoint = save_language_model_checkpoint(
        output_dir / f"language-child-final-{lineage_id}.pt",
        child,
        tokenizer,
        metadata={
            "lineage_id": lineage_id,
            "checkpoint_role": "child_final",
            "parent_checkpoint": str(parent_checkpoint),
            "child_initial_checkpoint": str(child_checkpoint),
        },
    )
    if parent_training_mode_before:
        parent_model.train()
    else:
        parent_model.eval()
    parent_hash_after = _state_hash(parent_model)
    parent_training_mode_after = bool(parent_model.training)
    parent_rollback_verified = (
        parent_hash_after == parent_hash_before == _load_checkpoint_state_hash(parent_checkpoint)
    )
    child_loss_delta = float(child_eval["heldout_loss"]) - float(parent_eval["heldout_loss"])
    learning_gate = (
        child_learning.get("promotion_gate")
        if isinstance(child_learning.get("promotion_gate"), Mapping)
        else {}
    )
    child_learning_ok = (
        not bool(cfg.require_child_learning)
        or bool(learning_gate.get("new_domain_improved"))
    )
    forgetting_ok = float(
        child_learning.get("learning_evidence", {}).get("old_domain_forgetting", 0.0)
        if isinstance(child_learning.get("learning_evidence"), Mapping)
        else 0.0
    ) <= float(cfg.max_old_domain_forgetting)
    child_eval_ok = child_loss_delta <= float(cfg.max_child_loss_delta)
    parent_runtime_unchanged = parent_hash_after == parent_hash_before
    eligible = (
        child_learning_ok
        and forgetting_ok
        and child_eval_ok
        and parent_rollback_verified
    )
    return child, {
        "artifact_kind": "marulho_language_checkpoint_evolution",
        "surface": "marulho_language_checkpoint_evolution.v1",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "active_language_path": parent_model.config.active_language_path,
        "mutates_parent_runtime": False,
        "lineage": {
            "lineage_id": lineage_id,
            "parent_checkpoint": str(parent_checkpoint),
            "child_initial_checkpoint": str(child_checkpoint),
            "child_final_checkpoint": str(child_final_checkpoint),
            "parent_state_hash_before": parent_hash_before,
            "parent_state_hash_after": parent_hash_after,
            "parent_training_mode_before": bool(parent_training_mode_before),
            "parent_training_mode_after": bool(parent_training_mode_after),
            "child_state_hash_final": _state_hash(child),
        },
        "parent_evaluation": parent_eval,
        "child_evaluation": child_eval,
        "child_learning": child_learning,
        "structural_proposal": structural_proposal,
        "structural_transaction": structural_transaction,
        "comparison": {
            "child_parent_heldout_loss_delta": float(child_loss_delta),
            "child_parent_perplexity_delta": (
                float(child_eval["heldout_perplexity"])
                - float(parent_eval["heldout_perplexity"])
            ),
            "max_child_loss_delta": float(cfg.max_child_loss_delta),
            "child_learning_improved": bool(child_learning_ok),
            "old_domain_forgetting_within_tolerance": bool(forgetting_ok),
            "parent_rollback_verified": bool(parent_rollback_verified),
            "parent_training_mode_unchanged": (
                bool(parent_training_mode_after) == bool(parent_training_mode_before)
            ),
        },
        "promotion_gate": {
            "status": "eligible_child_checkpoint_for_review" if eligible else "reject_child_checkpoint",
            "eligible_for_parent_promotion_review": bool(eligible),
            "requires_operator_review": True,
            "parent_runtime_unchanged": bool(parent_runtime_unchanged),
            "rollback_to_parent_verified": bool(parent_rollback_verified),
            "child_checkpoint_available": True,
        },
        "config": asdict(cfg),
    }
