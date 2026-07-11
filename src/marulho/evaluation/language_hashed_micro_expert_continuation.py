"""General-language continuation for the qualified V11 hash checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from marulho.evaluation.language_matched_support import (
    MatchedLanguageDataConfig,
    prepare_matched_language_data,
    run_matched_training_arm,
    sha256_file,
)
from marulho.evaluation.language_relation_binding_experiment import (
    evaluate_relation_binding_cases_batched,
)
from marulho.evaluation.language_training_experiment import (
    LanguageTrainingExperimentConfig,
    _prepare_language_loss_backend,
    _resolve_device,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_hashed_micro_experts import (
    MarulhoHashedMicroExpertLanguageModel,
    load_hashed_micro_expert_checkpoint,
    save_hashed_micro_expert_checkpoint,
)
from marulho.training.language_model import LanguageBatch, evaluate_language_model


SURFACE = "marulho_hashed_micro_expert_general_continuation.v1"
ARTIFACT_KIND = "marulho_hashed_micro_expert_general_continuation"
SAVE_DECISION = "save_v11_general_continuation_for_unseen_generation"


@dataclass(frozen=True)
class HashedMicroExpertContinuationConfig:
    additional_token_budget: int = 184_550_400
    sequence_length: int = 72
    batch_size: int = 144
    eval_batches: int = 16
    relation_eval_batch_size: int = 64
    learning_rate: float = 1.5e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.02
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    seed: int = 2027
    sample_bytes_per_train_source: int = 192 * 1024 * 1024
    sample_bytes_per_eval_source: int = 32 * 1024 * 1024
    sample_range_count: int = 32
    execution_backend: str = "eager"
    compile_loss_tolerance: float = 1.0e-3
    minimum_heldout_loss_improvement: float = 0.10


def _data_config(
    config: HashedMicroExpertContinuationConfig,
) -> MatchedLanguageDataConfig:
    return MatchedLanguageDataConfig(
        token_budget=int(config.additional_token_budget),
        sequence_length=int(config.sequence_length),
        batch_size=int(config.batch_size),
        eval_batches=int(config.eval_batches),
        relation_fraction=0.0,
        seed=int(config.seed),
        sample_bytes_per_train_source=int(config.sample_bytes_per_train_source),
        sample_bytes_per_eval_source=int(config.sample_bytes_per_eval_source),
        sample_range_count=int(config.sample_range_count),
    )


def _training_config(
    config: HashedMicroExpertContinuationConfig,
) -> LanguageTrainingExperimentConfig:
    return LanguageTrainingExperimentConfig(
        learning_rate=float(config.learning_rate),
        minimum_learning_rate_fraction=float(
            config.minimum_learning_rate_fraction
        ),
        warmup_fraction=float(config.warmup_fraction),
        weight_decay=float(config.weight_decay),
        precision=str(config.precision),
        execution_backend=str(config.execution_backend),
        compile_loss_tolerance=float(config.compile_loss_tolerance),
    )


def general_continuation_decision(
    *,
    heldout_loss_before: float,
    heldout_loss_after: float,
    processed_tokens: int,
    requested_tokens: int,
    minimum_improvement: float = 0.10,
) -> str:
    if int(processed_tokens) < int(requested_tokens):
        return "incomplete_v11_general_continuation"
    improvement = float(heldout_loss_before) - float(heldout_loss_after)
    if improvement >= float(minimum_improvement):
        return SAVE_DECISION
    if improvement > 0.0:
        return "redesign_v11_general_continuation_weak_loss_gain"
    return "retire_v11_general_continuation_no_loss_gain"


def _validate_parent(
    model: MarulhoHashedMicroExpertLanguageModel,
    metadata: Mapping[str, Any],
) -> int:
    if model.hashed_config.mode != "token_hash":
        raise ValueError("V11 continuation parent must use token_hash mode")
    if metadata.get("decision") != (
        "promote_v11_hash_for_checkpoint_and_unseen_generation"
    ):
        raise ValueError("V11 continuation parent lacks durability qualification")
    processed_tokens = int(metadata.get("processed_tokens") or 0)
    if processed_tokens < 67_108_864:
        raise ValueError("V11 continuation parent token count is not qualified")
    if metadata.get("external_llm_used") is not False:
        raise ValueError("V11 continuation parent must be MARULHO-owned")
    return processed_tokens


def run_hashed_micro_expert_continuation(
    *,
    parent_checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    output_path: str | Path,
    checkpoint_output_path: str | Path | None = None,
    config: HashedMicroExpertContinuationConfig = (
        HashedMicroExpertContinuationConfig()
    ),
    device: str = "auto",
) -> dict[str, Any]:
    if config.additional_token_budget < 1:
        raise ValueError("additional_token_budget must be positive")
    if config.minimum_heldout_loss_improvement <= 0.0:
        raise ValueError("minimum_heldout_loss_improvement must be positive")
    if config.execution_backend not in {"eager", "inductor"}:
        raise ValueError("execution_backend must be 'eager' or 'inductor'")
    resolved = _resolve_device(device)
    if config.execution_backend == "inductor" and resolved.type != "cuda":
        raise ValueError("Inductor V11 continuation is admitted only on CUDA")

    parent_path = Path(parent_checkpoint_path)
    output = Path(output_path)
    checkpoint_output = (
        None if checkpoint_output_path is None else Path(checkpoint_output_path)
    )
    if checkpoint_output is not None and (
        checkpoint_output.resolve() == parent_path.resolve()
    ):
        raise ValueError("Continuation checkpoint cannot overwrite its parent")
    model, parent_tokenizer, parent_metadata = (
        load_hashed_micro_expert_checkpoint(parent_path, map_location="cpu")
    )
    parent_processed_tokens = _validate_parent(model, parent_metadata)
    parent_sha256 = sha256_file(parent_path)
    started = time.perf_counter()
    prepared = prepare_matched_language_data(
        tokenizer_checkpoint_path=parent_path,
        relation_corpus_path=relation_corpus_path,
        relation_cases_path=relation_cases_path,
        general_train_paths=general_train_paths,
        general_eval_paths=general_eval_paths,
        config=_data_config(config),
        device=resolved,
    )
    if prepared.tokenizer.vocabulary_hash() != parent_tokenizer.vocabulary_hash():
        raise RuntimeError("Continuation data tokenizer differs from parent")
    if int(model.context_length) < int(config.sequence_length):
        raise ValueError("Continuation sequence exceeds parent context length")

    model = model.to(resolved)
    model.set_hashed_micro_expert_mode("token_hash")
    initial_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    training_config = _training_config(config)
    warm_batch = LanguageBatch(
        prepared.staged.input_ids[0],
        prepared.staged.target_ids[0],
    )
    previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_matmul_precision = torch.get_float32_matmul_precision()
    if resolved.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    try:
        model.eval()
        heldout_before = evaluate_language_model(model, prepared.eval_batches)
        relation_before = evaluate_relation_binding_cases_batched(
            model,
            prepared.tokenizer,
            prepared.cases,
            batch_size=int(config.relation_eval_batch_size),
        )
        model.train()
        training_loss, execution = _prepare_language_loss_backend(
            model,
            warm_batch,
            training_config,
        )
        row = run_matched_training_arm(
            "token_hash",
            architecture="hashed_micro_experts",
            model=model,
            initial_state=initial_state,
            training_loss=training_loss,
            execution={
                **execution,
                "continuation_from_strict_v11_checkpoint": True,
                "optimizer_state_restored": False,
                "fresh_cosine_schedule_phase": True,
                "relation_updates_scheduled": False,
            },
            allocated_compile_seconds=float(execution["compile_seconds"]),
            prepared=prepared,
            training_config=training_config,
            gradient_clip=float(config.gradient_clip),
            precision=str(config.precision),
            relation_eval_batch_size=int(config.relation_eval_batch_size),
            model_seed=int(config.seed),
            device=resolved,
            progress_prefix="hashed-micro-v11-continuation",
            configure_model=lambda active, _name: (
                active.set_hashed_micro_expert_mode("token_hash")
            ),
            diagnostic_builder=None,
            extra_row={
                "parent_checkpoint_sha256": parent_sha256,
                "parent_processed_tokens": parent_processed_tokens,
                "training_mixture": "general_only_equal_source_alternation",
                "relation_updates_scheduled": False,
            },
        )
    finally:
        if resolved.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            torch.set_float32_matmul_precision(previous_matmul_precision)

    additional_processed_tokens = int(row["processed_tokens"])
    cumulative_processed_tokens = (
        parent_processed_tokens + additional_processed_tokens
    )
    decision = general_continuation_decision(
        heldout_loss_before=float(heldout_before["heldout_loss"]),
        heldout_loss_after=float(row["heldout"]["heldout_loss"]),
        processed_tokens=additional_processed_tokens,
        requested_tokens=int(config.additional_token_budget),
        minimum_improvement=float(config.minimum_heldout_loss_improvement),
    )
    checkpoint_record: dict[str, Any] | None = None
    if checkpoint_output is not None:
        if decision == SAVE_DECISION:
            saved = save_hashed_micro_expert_checkpoint(
                checkpoint_output,
                model,
                prepared.tokenizer,
                metadata={
                    "decision": decision,
                    "parent_checkpoint": str(parent_path),
                    "parent_checkpoint_sha256": parent_sha256,
                    "parent_processed_tokens": parent_processed_tokens,
                    "additional_processed_tokens": additional_processed_tokens,
                    "processed_tokens": cumulative_processed_tokens,
                    "schedule_sha256": prepared.schedule_sha256,
                    "heldout_loss_before": float(
                        heldout_before["heldout_loss"]
                    ),
                    "heldout_loss": float(row["heldout"]["heldout_loss"]),
                    "heldout_loss_improvement": float(
                        heldout_before["heldout_loss"]
                    ) - float(row["heldout"]["heldout_loss"]),
                    "relation_accuracy_before": float(
                        relation_before["accuracy"]
                    ),
                    "relation_accuracy_after": float(row["relation"]["accuracy"]),
                    "free_relation_accuracy_before": float(
                        relation_before["generation_exact_accuracy"]
                    ),
                    "free_relation_accuracy": float(
                        row["relation"]["generation_exact_accuracy"]
                    ),
                    "relation_updates_scheduled": False,
                    "optimizer_state_restored": False,
                    "optimizer_state_persisted": False,
                    "requires_unseen_generation": True,
                    "external_llm_used": False,
                },
            )
            checkpoint_record = {
                "path": str(saved),
                "sha256": sha256_file(saved),
                "saved": True,
                "quality_promoted": False,
                "requires_unseen_generation": True,
                "optimizer_state_persisted": False,
            }
        elif checkpoint_output.exists():
            checkpoint_output.unlink()

    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "configuration": asdict(config),
        "parent": {
            "path": str(parent_path),
            "sha256": parent_sha256,
            "processed_tokens": parent_processed_tokens,
            "decision": parent_metadata["decision"],
            "tokenizer_hash": parent_tokenizer.vocabulary_hash(),
        },
        "schedule": {
            "sha256": prepared.schedule_sha256,
            "step_count": int(prepared.staged.input_ids.shape[0]),
            "storage_bytes": int(prepared.staged.storage_bytes),
            "relation_updates_scheduled": False,
            "source_selections": prepared.source_selections,
        },
        "before": {
            "cumulative_processed_tokens": parent_processed_tokens,
            "heldout": heldout_before,
            "relation": relation_before,
        },
        "after": {
            "cumulative_processed_tokens": cumulative_processed_tokens,
            "arm": row,
        },
        "quality_curve": [
            {
                "cumulative_processed_tokens": parent_processed_tokens,
                "heldout_loss": float(heldout_before["heldout_loss"]),
                "heldout_perplexity": float(
                    heldout_before["heldout_perplexity"]
                ),
            },
            {
                "cumulative_processed_tokens": cumulative_processed_tokens,
                "heldout_loss": float(row["heldout"]["heldout_loss"]),
                "heldout_perplexity": float(
                    row["heldout"]["heldout_perplexity"]
                ),
            },
        ],
        "heldout_loss_improvement": float(heldout_before["heldout_loss"])
        - float(row["heldout"]["heldout_loss"]),
        "decision": decision,
        "checkpoint": checkpoint_record,
        "experiment_wall_seconds": time.perf_counter() - started,
        "promotion_boundary": {
            "base_quality_promoted": False,
            "runtime_install_allowed": False,
            "continual_memory_allowed": False,
            "requires_unseen_generation": True,
        },
    }
    write_json_report_with_readme(
        output,
        report,
        title="MARULHO Hashed Micro-Experts V11 General Continuation",
    )
    print(
        f"[hashed-micro-v11-continuation] decision {decision}; loss "
        f"{heldout_before['heldout_loss']:.4f} -> "
        f"{row['heldout']['heldout_loss']:.4f}",
        flush=True,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--relation-corpus", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--general-train", action="append", type=Path, required=True)
    parser.add_argument("--general-eval", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path)
    parser.add_argument("--additional-token-budget", type=int, default=184_550_400)
    parser.add_argument("--train-sample-mib", type=int, default=192)
    parser.add_argument("--eval-sample-mib", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument(
        "--execution-backend",
        choices=("eager", "inductor"),
        default="eager",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = HashedMicroExpertContinuationConfig(
        additional_token_budget=int(args.additional_token_budget),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        sample_bytes_per_train_source=int(args.train_sample_mib) * 1024 * 1024,
        sample_bytes_per_eval_source=int(args.eval_sample_mib) * 1024 * 1024,
        execution_backend=str(args.execution_backend),
    )
    run_hashed_micro_expert_continuation(
        parent_checkpoint_path=args.parent_checkpoint,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=tuple(args.general_train),
        general_eval_paths=tuple(args.general_eval),
        output_path=args.output,
        checkpoint_output_path=args.checkpoint_output,
        config=config,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
