"""Write standalone controlled checkpoint-evolution evidence for MARULHO LM."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Sequence

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_training_experiment import (
    _apply_cuda_math_policy,
    _resolve_device,
    _restore_cuda_math_policy,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_checkpoint_evolution import (
    LanguageCheckpointEvolutionConfig,
    run_language_checkpoint_evolution,
)
from marulho.training.language_continual_learning import (
    LanguageContinualLearningConfig,
)
from marulho.training.language_model import (
    LanguageBatch,
    LanguageModelConfig,
    MarulhoLanguageModel,
    build_language_model_splits,
    load_language_model_checkpoint,
)
from marulho.training.language_structural_plasticity import (
    LanguageStructuralPlasticityConfig,
)


SURFACE = "marulho_language_checkpoint_evolution_experiment.v1"
ARTIFACT_KIND = "marulho_language_checkpoint_evolution_experiment"

DEFAULT_OLD_CORPUS = (
    "Parent checkpoint memory protects old replay language, runtime truth, "
    "rollback evidence, routed sparse specialists, and MARULHO-owned state. "
    "The parent domain repeats audit terms so child evolution must preserve "
    "old evidence rather than hiding forgetting behind a fluent continuation. "
) * 240

DEFAULT_CHILD_CORPUS = (
    "Child checkpoint evolution trains in isolation with replay protection, "
    "checkpoint lineage, optional structural growth, and backend truth. "
    "The child domain repeats adaptation terms so the report can measure "
    "learning, rollback, throughput, and parent preservation together. "
) * 240


@dataclass(frozen=True)
class LanguageCheckpointEvolutionExperimentConfig:
    model_vocab_size: int = 0
    sampled_vocab_size: int = 0
    sparse_vocab_optimizer: bool = True
    embedding_dim: int = 32
    state_dim: int = 64
    expert_count: int = 8
    active_expert_count: int = 2
    route_candidate_count: int = 4
    expert_hidden_dim: int = 96
    recurrent_gradient_horizon: int = 0
    memory_slot_count: int = 0
    memory_slot_candidate_count: int = 0
    active_memory_slot_count: int = 1
    memory_slot_init_std: float = 0.02
    sequence_length: int = 32
    stride: int = 16
    batch_size: int = 8
    eval_fraction: float = 0.2
    max_parent_eval_batches: int = 0
    max_child_eval_batches: int = 0
    max_child_train_batches: int = 4
    max_replay_batches: int = 4
    learning_rate: float = 2e-3
    max_steps: int = 2
    replay_loss_weight: float = 0.25
    max_grad_norm: float = 1.0
    gradient_clip_interval: int = 1
    dense_adamw_backend: str = "default"
    forgetting_tolerance: float = 100.0
    replay_retention_tolerance: float = 100.0
    rollback_on_forgetting: bool = False
    collect_training_telemetry: bool = False
    sampled_vocab_ce_triton_training: bool = False
    memory_slots_triton_training: bool = False
    max_child_loss_delta: float = 100.0
    max_old_domain_forgetting: float = 100.0
    require_child_learning: bool = False
    allow_structural_growth: bool = True
    operator_approved_child_growth: bool = True
    route_saturation_threshold: float = 0.0
    max_structural_eval_loss_delta: float = 100.0
    cuda_allow_tf32: bool = True
    cuda_float32_matmul_precision: str = "high"
    seed: int = 20260705
    device: str = "auto"


def _read_text(path: str | Path | None, *, default: str) -> tuple[str, str]:
    if path is None:
        return default, "default_inline"
    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Checkpoint-evolution corpus is empty: {resolved}")
    return text, str(resolved)


def _model_config(
    tokenizer: ByteLevelLanguageTokenizer,
    config: LanguageCheckpointEvolutionExperimentConfig,
) -> LanguageModelConfig:
    model_vocab_size = (
        int(config.model_vocab_size)
        if int(config.model_vocab_size) > 0
        else int(tokenizer.vocab_size)
    )
    if model_vocab_size < int(tokenizer.vocab_size):
        raise ValueError("model_vocab_size must be at least tokenizer vocab size")
    sampled_vocab_size = max(0, int(config.sampled_vocab_size))
    if sampled_vocab_size >= model_vocab_size:
        raise ValueError("sampled_vocab_size must be smaller than model_vocab_size")
    sparse_vocab_gradients = bool(
        config.sparse_vocab_optimizer and sampled_vocab_size > 0
    )
    return LanguageModelConfig(
        vocab_size=model_vocab_size,
        embedding_dim=int(config.embedding_dim),
        state_dim=int(config.state_dim),
        expert_count=int(config.expert_count),
        active_expert_count=int(config.active_expert_count),
        route_candidate_count=int(config.route_candidate_count),
        expert_hidden_dim=int(config.expert_hidden_dim),
        sampled_vocab_size=sampled_vocab_size,
        sampled_vocab_sparse_lm_head_gradient=sparse_vocab_gradients,
        sparse_token_embedding_gradients=sparse_vocab_gradients,
        generation_vocab_size=(
            int(tokenizer.vocab_size)
            if model_vocab_size > int(tokenizer.vocab_size)
            else 0
        ),
        recurrent_gradient_horizon=max(0, int(config.recurrent_gradient_horizon)),
        memory_slot_count=max(0, int(config.memory_slot_count)),
        memory_slot_candidate_count=max(0, int(config.memory_slot_candidate_count)),
        active_memory_slot_count=max(1, int(config.active_memory_slot_count)),
        memory_slot_init_std=max(0.0, float(config.memory_slot_init_std)),
    )


def _trim(
    batches: Sequence[LanguageBatch],
    limit: int,
) -> tuple[LanguageBatch, ...]:
    if int(limit) <= 0:
        return tuple(batches)
    return tuple(batches[: int(limit)])


def _batch_token_count(batches: Sequence[LanguageBatch]) -> int:
    return int(sum(int(batch.target_ids.numel()) for batch in batches))


def _apply_training_backend_policy(
    config: LanguageCheckpointEvolutionExperimentConfig,
) -> dict[str, Any]:
    env_names = {
        "sampled_vocab_ce_triton_training": (
            "MARULHO_LANGUAGE_SAMPLED_VOCAB_CE_TRITON_TRAINING"
        ),
        "memory_slots_triton_training": (
            "MARULHO_LANGUAGE_MEMORY_SLOTS_TRITON_TRAINING"
        ),
    }
    requested = {
        "sampled_vocab_ce_triton_training": bool(
            config.sampled_vocab_ce_triton_training
        ),
        "memory_slots_triton_training": bool(config.memory_slots_triton_training),
    }
    previous = {key: os.environ.get(name) for key, name in env_names.items()}
    for key, name in env_names.items():
        os.environ[name] = "1" if requested[key] else "0"
    return {
        "surface": "marulho_language_checkpoint_evolution_backend_policy.v1",
        "scope": "checkpoint_evolution_experiment_process",
        "env_names": env_names,
        "previous_env": previous,
        "requested": requested,
        "active": {key: os.environ.get(name) for key, name in env_names.items()},
    }


def _restore_training_backend_policy(policy: dict[str, Any]) -> None:
    env_names = policy.get("env_names")
    previous = policy.get("previous_env")
    if not isinstance(env_names, dict) or not isinstance(previous, dict):
        return
    for key, name in env_names.items():
        if not isinstance(name, str):
            continue
        old_value = previous.get(key)
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(old_value)


def _build_parent_model(
    *,
    parent_checkpoint_path: str | Path | None,
    tokenizer: ByteLevelLanguageTokenizer,
    config: LanguageCheckpointEvolutionExperimentConfig,
    device: torch.device,
) -> tuple[MarulhoLanguageModel, ByteLevelLanguageTokenizer, dict[str, Any]]:
    if parent_checkpoint_path is None:
        model = MarulhoLanguageModel(_model_config(tokenizer, config)).to(device)
        return model, tokenizer, {
            "surface": "marulho_language_checkpoint_evolution_parent_source.v1",
            "source": "fresh_parent_model",
            "parent_checkpoint_path": None,
        }
    model, loaded_tokenizer, metadata = load_language_model_checkpoint(
        parent_checkpoint_path,
        map_location=device,
    )
    model = model.to(device)
    return model, loaded_tokenizer, {
        "surface": "marulho_language_checkpoint_evolution_parent_source.v1",
        "source": "loaded_parent_checkpoint",
        "parent_checkpoint_path": str(parent_checkpoint_path),
        "parent_checkpoint_metadata": dict(metadata),
    }


def run_language_checkpoint_evolution_experiment(
    *,
    output_path: str | Path,
    parent_checkpoint_path: str | Path | None = None,
    old_corpus_path: str | Path | None = None,
    child_corpus_path: str | Path | None = None,
    config: LanguageCheckpointEvolutionExperimentConfig | None = None,
) -> dict[str, Any]:
    cfg = config or LanguageCheckpointEvolutionExperimentConfig()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    training_backend_policy = _apply_training_backend_policy(cfg)
    cuda_math_policy: dict[str, Any] = {"before": None}
    try:
        cuda_math_policy = _apply_cuda_math_policy(device, cfg)
        tokenizer = ByteLevelLanguageTokenizer()
        parent_model, tokenizer, parent_source = _build_parent_model(
            parent_checkpoint_path=parent_checkpoint_path,
            tokenizer=tokenizer,
            config=cfg,
            device=device,
        )
        old_text, old_source = _read_text(old_corpus_path, default=DEFAULT_OLD_CORPUS)
        child_text, child_source = _read_text(
            child_corpus_path,
            default=DEFAULT_CHILD_CORPUS,
        )
        old_split = build_language_model_splits(
            [old_text],
            tokenizer,
            sequence_length=max(2, int(cfg.sequence_length)),
            eval_fraction=float(cfg.eval_fraction),
            stride=max(1, int(cfg.stride)),
            batch_size=max(1, int(cfg.batch_size)),
            device=device,
        )
        child_split = build_language_model_splits(
            [child_text],
            tokenizer,
            sequence_length=max(2, int(cfg.sequence_length)),
            eval_fraction=float(cfg.eval_fraction),
            stride=max(1, int(cfg.stride)),
            batch_size=max(1, int(cfg.batch_size)),
            device=device,
        )
        parent_eval_batches = _trim(
            old_split.eval,
            int(cfg.max_parent_eval_batches),
        )
        child_eval_batches = _trim(
            child_split.eval,
            int(cfg.max_child_eval_batches),
        )
        child_train_batches = _trim(
            child_split.train,
            int(cfg.max_child_train_batches),
        )
        replay_batches = _trim(old_split.train, int(cfg.max_replay_batches))
        learning_config = LanguageContinualLearningConfig(
            learning_rate=float(cfg.learning_rate),
            max_steps=int(cfg.max_steps),
            replay_loss_weight=float(cfg.replay_loss_weight),
            forgetting_tolerance=float(cfg.forgetting_tolerance),
            replay_retention_tolerance=float(cfg.replay_retention_tolerance),
            rollback_on_forgetting=bool(cfg.rollback_on_forgetting),
            sparse_vocab_optimizer=bool(cfg.sparse_vocab_optimizer),
            dense_adamw_backend=str(cfg.dense_adamw_backend),
            max_grad_norm=float(cfg.max_grad_norm),
            gradient_clip_interval=max(0, int(cfg.gradient_clip_interval)),
            collect_training_telemetry=bool(cfg.collect_training_telemetry),
        )
        evolution_config = LanguageCheckpointEvolutionConfig(
            max_child_loss_delta=float(cfg.max_child_loss_delta),
            max_old_domain_forgetting=float(cfg.max_old_domain_forgetting),
            require_child_learning=bool(cfg.require_child_learning),
            allow_structural_growth=bool(cfg.allow_structural_growth),
            operator_approved_child_growth=bool(cfg.operator_approved_child_growth),
        )
        structural_config = LanguageStructuralPlasticityConfig(
            route_saturation_threshold=float(cfg.route_saturation_threshold),
            max_eval_loss_delta=float(cfg.max_structural_eval_loss_delta),
        )
        checkpoint_dir = output.parent / f"{output.stem}-checkpoints"
        _child, evolution_report = run_language_checkpoint_evolution(
            parent_model,
            tokenizer,
            eval_batches=parent_eval_batches,
            child_train_batches=child_train_batches,
            child_new_eval_batches=child_eval_batches,
            replay_batches=replay_batches,
            checkpoint_dir=checkpoint_dir,
            config=evolution_config,
            learning_config=learning_config,
            structural_config=structural_config,
        )
        lineage = evolution_report.get("checkpoint_lineage")
        lineage_dict = lineage if isinstance(lineage, dict) else {}
        runtime_evidence = evolution_report.get("runtime_evidence")
        runtime_dict = runtime_evidence if isinstance(runtime_evidence, dict) else {}
        review = evolution_report.get("evolution_review")
        review_dict = review if isinstance(review, dict) else {}
        gate = evolution_report.get("promotion_gate")
        gate_dict = gate if isinstance(gate, dict) else {}
        report = {
            "artifact_kind": ARTIFACT_KIND,
            "surface": SURFACE,
            "output_path": str(output),
            "owned_by_marulho": True,
            "external_llm_used": False,
            "loads_external_checkpoint": False,
            "active_language_path": parent_model.config.active_language_path,
            "status": gate_dict.get("status"),
            "device": str(device),
            "cuda_math_policy": cuda_math_policy,
            "training_backend_policy": training_backend_policy,
            "experiment_config": asdict(cfg),
            "evolution_config": asdict(evolution_config),
            "continual_learning_config": asdict(learning_config),
            "structural_config": asdict(structural_config),
            "model_config": asdict(parent_model.config),
            "parent_source": parent_source,
            "corpus": {
                "old_source": old_source,
                "child_source": child_source,
                "old_character_count": len(old_text),
                "child_character_count": len(child_text),
            },
            "split": {
                "old": old_split.report,
                "child": child_split.report,
                "used_parent_eval_batches": len(parent_eval_batches),
                "used_child_eval_batches": len(child_eval_batches),
                "used_child_train_batches": len(child_train_batches),
                "used_replay_batches": len(replay_batches),
                "used_parent_eval_tokens": _batch_token_count(parent_eval_batches),
                "used_child_eval_tokens": _batch_token_count(child_eval_batches),
                "used_child_train_tokens": _batch_token_count(child_train_batches),
                "used_replay_tokens": _batch_token_count(replay_batches),
            },
            "checkpoint_dir": str(checkpoint_dir),
            "parent_checkpoint_path": lineage_dict.get("parent_checkpoint_path"),
            "child_initial_checkpoint_path": lineage_dict.get(
                "child_initial_checkpoint_path"
            ),
            "child_final_checkpoint_path": lineage_dict.get(
                "child_final_checkpoint_path"
            ),
            "child_final_checkpoint_sha256": lineage_dict.get(
                "child_final_checkpoint_sha256"
            ),
            "checkpoint_lineage": lineage_dict,
            "runtime_evidence": runtime_dict,
            "evolution_review": review_dict,
            "checkpoint_evolution": evolution_report,
            "experiment_review": {
                "surface": "marulho_language_checkpoint_evolution_experiment_review.v1",
                "records_checkpoint_lineage": bool(
                    lineage_dict.get("lineage_complete")
                ),
                "records_runtime_evidence": bool(
                    runtime_dict.get("surface")
                    == "marulho_language_checkpoint_evolution_runtime_truth.v1"
                ),
                "records_child_learning_update": int(
                    review_dict.get("child_update_token_count", 0) or 0
                )
                > 0,
                "records_training_backend_policy": bool(
                    training_backend_policy.get("surface")
                    == "marulho_language_checkpoint_evolution_backend_policy.v1"
                ),
                "records_structural_review": bool(
                    "structural_growth_attempted" in review_dict
                ),
                "parent_kept_installed": bool(
                    review_dict.get("parent_kept_installed")
                ),
                "isolated_child_training": bool(
                    review_dict.get("isolated_child_training")
                ),
                "long_run_evidence_required_for_parent_promotion": True,
                "promotes_parent_promotion": False,
                "promotes_runtime_claim": False,
            },
            "promotion_gate": {
                "checkpoint_evolution_evidence_available": bool(
                    lineage_dict.get("lineage_complete")
                    and review_dict.get("parent_kept_installed")
                    and review_dict.get("isolated_child_training")
                ),
                "eligible_for_parent_promotion_review": bool(
                    gate_dict.get("eligible_for_parent_promotion_review")
                ),
                "requires_operator_review": True,
                "parent_runtime_unchanged": bool(
                    gate_dict.get("parent_runtime_unchanged")
                ),
                "rollback_to_parent_verified": bool(
                    gate_dict.get("rollback_to_parent_verified")
                ),
                "checkpoint_lineage_complete": bool(
                    gate_dict.get("checkpoint_lineage_complete")
                ),
                "child_checkpoint_available": bool(
                    gate_dict.get("child_checkpoint_available")
                ),
                "long_run_evidence_required_for_parent_promotion": True,
                "promotes_parent_promotion": False,
                "promotes_runtime_claim": False,
            },
        }
        write_json_report_with_readme(output, report)
        return report
    finally:
        before_policy = cuda_math_policy.get("before")
        if isinstance(before_policy, dict):
            _restore_cuda_math_policy(before_policy)
        _restore_training_backend_policy(training_backend_policy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, default=None)
    parser.add_argument("--old-corpus", type=Path, default=None)
    parser.add_argument("--child-corpus", type=Path, default=None)
    parser.add_argument("--model-vocab-size", type=int, default=0)
    parser.add_argument("--sampled-vocab-size", type=int, default=0)
    parser.add_argument("--disable-sparse-vocab-optimizer", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--expert-count", type=int, default=8)
    parser.add_argument("--active-expert-count", type=int, default=2)
    parser.add_argument("--route-candidate-count", type=int, default=4)
    parser.add_argument("--expert-hidden-dim", type=int, default=96)
    parser.add_argument("--recurrent-gradient-horizon", type=int, default=0)
    parser.add_argument("--memory-slot-count", type=int, default=0)
    parser.add_argument("--memory-slot-candidate-count", type=int, default=0)
    parser.add_argument("--active-memory-slot-count", type=int, default=1)
    parser.add_argument("--memory-slot-init-std", type=float, default=0.02)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--max-parent-eval-batches", type=int, default=0)
    parser.add_argument("--max-child-eval-batches", type=int, default=0)
    parser.add_argument("--max-child-train-batches", type=int, default=4)
    parser.add_argument("--max-replay-batches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--replay-loss-weight", type=float, default=0.25)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-clip-interval", type=int, default=1)
    parser.add_argument(
        "--dense-adamw-backend",
        choices=("default", "foreach", "fused"),
        default="default",
    )
    parser.add_argument("--forgetting-tolerance", type=float, default=100.0)
    parser.add_argument("--replay-retention-tolerance", type=float, default=100.0)
    parser.add_argument("--rollback-on-forgetting", action="store_true")
    parser.add_argument("--collect-training-telemetry", action="store_true")
    parser.add_argument("--sampled-vocab-ce-triton-training", action="store_true")
    parser.add_argument("--memory-slots-triton-training", action="store_true")
    parser.add_argument("--max-child-loss-delta", type=float, default=100.0)
    parser.add_argument("--max-old-domain-forgetting", type=float, default=100.0)
    parser.add_argument("--require-child-learning", action="store_true")
    parser.add_argument("--disable-structural-growth", action="store_true")
    parser.add_argument("--disable-operator-approved-child-growth", action="store_true")
    parser.add_argument("--route-saturation-threshold", type=float, default=0.0)
    parser.add_argument("--max-structural-eval-loss-delta", type=float, default=100.0)
    parser.add_argument("--disable-cuda-tf32", action="store_true")
    parser.add_argument(
        "--cuda-float32-matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = LanguageCheckpointEvolutionExperimentConfig(
        model_vocab_size=max(0, int(args.model_vocab_size)),
        sampled_vocab_size=max(0, int(args.sampled_vocab_size)),
        sparse_vocab_optimizer=not bool(args.disable_sparse_vocab_optimizer),
        embedding_dim=int(args.embedding_dim),
        state_dim=int(args.state_dim),
        expert_count=int(args.expert_count),
        active_expert_count=int(args.active_expert_count),
        route_candidate_count=int(args.route_candidate_count),
        expert_hidden_dim=int(args.expert_hidden_dim),
        recurrent_gradient_horizon=max(0, int(args.recurrent_gradient_horizon)),
        memory_slot_count=max(0, int(args.memory_slot_count)),
        memory_slot_candidate_count=max(0, int(args.memory_slot_candidate_count)),
        active_memory_slot_count=max(1, int(args.active_memory_slot_count)),
        memory_slot_init_std=max(0.0, float(args.memory_slot_init_std)),
        sequence_length=max(2, int(args.sequence_length)),
        stride=max(1, int(args.stride)),
        batch_size=max(1, int(args.batch_size)),
        eval_fraction=float(args.eval_fraction),
        max_parent_eval_batches=max(0, int(args.max_parent_eval_batches)),
        max_child_eval_batches=max(0, int(args.max_child_eval_batches)),
        max_child_train_batches=max(1, int(args.max_child_train_batches)),
        max_replay_batches=max(0, int(args.max_replay_batches)),
        learning_rate=float(args.learning_rate),
        max_steps=int(args.max_steps),
        replay_loss_weight=float(args.replay_loss_weight),
        max_grad_norm=float(args.max_grad_norm),
        gradient_clip_interval=max(0, int(args.gradient_clip_interval)),
        dense_adamw_backend=str(args.dense_adamw_backend),
        forgetting_tolerance=float(args.forgetting_tolerance),
        replay_retention_tolerance=float(args.replay_retention_tolerance),
        rollback_on_forgetting=bool(args.rollback_on_forgetting),
        collect_training_telemetry=bool(args.collect_training_telemetry),
        sampled_vocab_ce_triton_training=bool(
            args.sampled_vocab_ce_triton_training
        ),
        memory_slots_triton_training=bool(args.memory_slots_triton_training),
        max_child_loss_delta=float(args.max_child_loss_delta),
        max_old_domain_forgetting=float(args.max_old_domain_forgetting),
        require_child_learning=bool(args.require_child_learning),
        allow_structural_growth=not bool(args.disable_structural_growth),
        operator_approved_child_growth=not bool(
            args.disable_operator_approved_child_growth
        ),
        route_saturation_threshold=float(args.route_saturation_threshold),
        max_structural_eval_loss_delta=float(args.max_structural_eval_loss_delta),
        cuda_allow_tf32=not bool(args.disable_cuda_tf32),
        cuda_float32_matmul_precision=str(args.cuda_float32_matmul_precision),
        seed=int(args.seed),
        device=str(args.device),
    )
    report = run_language_checkpoint_evolution_experiment(
        output_path=args.output,
        parent_checkpoint_path=args.parent_checkpoint,
        old_corpus_path=args.old_corpus,
        child_corpus_path=args.child_corpus,
        config=config,
    )
    return (
        0
        if report["promotion_gate"]["checkpoint_evolution_evidence_available"]
        else 1
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
