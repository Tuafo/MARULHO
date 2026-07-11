"""Run a matched Transformer versus particle-field base-language falsifier."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import gc
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from marulho.evaluation.language_matched_support import (
    MatchedLanguageDataConfig,
    PreparedMatchedLanguageData,
    prepare_matched_language_data,
    run_matched_training_arm,
    sha256_file,
)
from marulho.evaluation.language_training_experiment import (
    LanguageTrainingExperimentConfig,
    _prepare_language_loss_backend,
    _resolve_device,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    evaluate_language_model,
)
from marulho.training.language_particle_field import (
    MarulhoParticleFieldLanguageModel,
    ParticleFieldConfig,
)


SURFACE = "marulho_particle_field_falsification.v1"
ARTIFACT_KIND = "marulho_particle_field_falsification"
ARM_NAMES = ("transformer", "particle_field")
MINIMUM_DECISION_TOKENS = 16_000_000


@dataclass(frozen=True)
class ParticleFieldFalsificationConfig:
    token_budget: int = 16_777_216
    sequence_length: int = 72
    batch_size: int = 32
    eval_batches: int = 16
    relation_eval_batch_size: int = 8
    relation_case_limit: int = 0
    relation_fraction: float = 0.20
    learning_rate: float = 3.0e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    data_seed: int = 15121
    model_seed: int = 15131
    sample_bytes_per_train_source: int = 64 * 1024 * 1024
    sample_bytes_per_eval_source: int = 32 * 1024 * 1024
    sample_range_count: int = 16
    schedule_mode: str = "indexed_host"
    execution_backend: str = "inductor"
    compile_loss_tolerance: float = 1.0e-3
    transformer_width: int = 512
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_mlp_ratio: float = 4.0
    particle_width: int = 256
    particle_count: int = 24_576
    particle_recurrences: int = 8
    particle_heads: int = 4
    particle_dropout: float = 0.0
    particle_state_batch_limit: int = 8
    maximum_parameter_delta_fraction: float = 0.001
    minimum_loss_gain: float = 0.005
    minimum_free_relation_gain: float = 0.02


def build_arm_model(
    arm: str,
    *,
    vocab_size: int,
    config: ParticleFieldFalsificationConfig,
):
    if arm == "transformer":
        return MarulhoLanguageModel(
            LanguageModelConfig(
                vocab_size=int(vocab_size),
                embedding_dim=int(config.transformer_width),
                state_dim=int(config.transformer_width),
                state_layers=int(config.transformer_layers),
                attention_heads=int(config.transformer_heads),
                transformer_context_length=int(config.sequence_length),
                transformer_mlp_ratio=float(config.transformer_mlp_ratio),
                transformer_dropout=0.0,
                tie_embeddings=True,
                active_language_path="marulho_transformer_v28_control",
            )
        )
    if arm == "particle_field":
        return MarulhoParticleFieldLanguageModel(
            ParticleFieldConfig(
                vocab_size=int(vocab_size),
                width=int(config.particle_width),
                particle_count=int(config.particle_count),
                recurrences=int(config.particle_recurrences),
                heads=int(config.particle_heads),
                context_length=int(config.sequence_length),
                dropout=float(config.particle_dropout),
                materialized_state_batch_limit=int(
                    config.particle_state_batch_limit
                ),
            )
        )
    raise ValueError(f"unknown particle-field falsification arm: {arm}")


def particle_field_decision(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    processed_tokens: int,
    parameter_delta_fraction: float,
    config: ParticleFieldFalsificationConfig,
) -> str:
    if set(arms) != set(ARM_NAMES):
        return "incomplete_v28_missing_matched_arm"
    if float(parameter_delta_fraction) > float(
        config.maximum_parameter_delta_fraction
    ):
        return "invalid_v28_parameter_budget_mismatch"
    if not all(
        bool(arms[name]["all_parameters_received_final_gradient"])
        for name in ARM_NAMES
    ):
        return "invalid_v28_incomplete_gradient_coverage"
    if int(processed_tokens) < MINIMUM_DECISION_TOKENS:
        return "diagnostic_v28_below_durable_token_floor"
    baseline = arms["transformer"]
    candidate = arms["particle_field"]
    loss_gain = float(baseline["heldout"]["heldout_loss"]) - float(
        candidate["heldout"]["heldout_loss"]
    )
    free_gain = float(candidate["relation"]["generation_exact_accuracy"]) - float(
        baseline["relation"]["generation_exact_accuracy"]
    )
    if (
        loss_gain >= float(config.minimum_loss_gain)
        and free_gain >= float(config.minimum_free_relation_gain)
    ):
        return "advance_v28_particle_field_to_unseen_generation"
    if loss_gain >= float(config.minimum_loss_gain) or free_gain >= float(
        config.minimum_free_relation_gain
    ):
        return "redesign_v28_disjoint_loss_and_generation_signal"
    return "retire_v28_particle_field_no_joint_language_win"


def _training_config(
    config: ParticleFieldFalsificationConfig,
) -> LanguageTrainingExperimentConfig:
    return LanguageTrainingExperimentConfig(
        sequence_length=int(config.sequence_length),
        batch_size=int(config.batch_size),
        learning_rate=float(config.learning_rate),
        minimum_learning_rate_fraction=float(
            config.minimum_learning_rate_fraction
        ),
        warmup_fraction=float(config.warmup_fraction),
        weight_decay=float(config.weight_decay),
        max_grad_norm=float(config.gradient_clip),
        precision=str(config.precision),
        execution_backend=str(config.execution_backend),
        compile_loss_tolerance=float(config.compile_loss_tolerance),
        device="cuda",
    )


def _arm_diagnostics(model, _input_ids: torch.Tensor) -> Mapping[str, Any]:
    if isinstance(model, MarulhoParticleFieldLanguageModel):
        return model.parameter_report()
    return {
        "surface": "marulho_transformer_v28_parameter_report.v1",
        "total_parameters": sum(int(value.numel()) for value in model.parameters()),
        "external_llm_used": False,
        "owned_by_marulho": True,
    }


def _assemble_report(
    *,
    config: ParticleFieldFalsificationConfig,
    prepared: PreparedMatchedLanguageData,
    arms: Mapping[str, Mapping[str, Any]],
    executed_arms: Sequence[str],
    tokenizer_checkpoint: Path,
    relation_cases: Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    counts = {
        name: int(row["parameters"])
        for name, row in arms.items()
    }
    parameter_delta = (
        abs(counts["particle_field"] - counts["transformer"])
        if set(counts) == set(ARM_NAMES)
        else None
    )
    parameter_delta_fraction: float | None = (
        float(parameter_delta) / float(counts["transformer"])
        if parameter_delta is not None
        else None
    )
    processed_tokens = (
        min(int(row["processed_tokens"]) for row in arms.values())
        if arms
        else 0
    )
    decision = particle_field_decision(
        arms,
        processed_tokens=processed_tokens,
        parameter_delta_fraction=(
            float("inf")
            if parameter_delta_fraction is None
            else parameter_delta_fraction
        ),
        config=config,
    )
    return {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "configuration": asdict(config),
        "research_basis": {
            "particle_field_inspiration": "BDH-GPU",
            "paper": "https://arxiv.org/abs/2509.26507",
            "external_weights_loaded": False,
            "paper_result_treated_as_local_evidence": False,
            "paper_evidence_boundary": (
                "raw-byte stateful Europarl language/translation, not general web coherence"
            ),
        },
        "tokenizer": {
            "checkpoint_path": str(tokenizer_checkpoint),
            "checkpoint_sha256": sha256_file(tokenizer_checkpoint),
            "vocabulary_hash": prepared.tokenizer.vocabulary_hash(),
            "vocab_size": int(prepared.tokenizer.vocab_size),
        },
        "relation_cases": {
            "path": str(relation_cases),
            "sha256": sha256_file(relation_cases),
            "case_count": len(prepared.cases),
            "labels_metrics_only": True,
        },
        "schedule": {
            "sha256": prepared.schedule_sha256,
            "step_count": int(prepared.staged.step_count),
            "tokens_per_step": int(prepared.staged.tokens_per_step),
            "processed_tokens_per_complete_arm": (
                int(prepared.staged.step_count)
                * int(prepared.staged.tokens_per_step)
            ),
            "mode": prepared.staged.mode,
            "relation_fraction": float(config.relation_fraction),
            "same_schedule_for_every_arm": True,
        },
        "sources": prepared.source_selections,
        "parameter_match": {
            "counts": counts,
            "absolute_delta": parameter_delta,
            "delta_fraction": parameter_delta_fraction,
            "maximum_delta_fraction": float(
                config.maximum_parameter_delta_fraction
            ),
            "passed": parameter_delta is not None
            and parameter_delta_fraction is not None
            and parameter_delta_fraction
            <= float(config.maximum_parameter_delta_fraction),
        },
        "arms": dict(arms),
        "arms_executed_this_run": list(executed_arms),
        "experiment_wall_seconds_this_run": float(elapsed_seconds),
        "decision": decision,
        "decision_contract": {
            "minimum_processed_tokens": MINIMUM_DECISION_TOKENS,
            "minimum_particle_loss_gain": float(config.minimum_loss_gain),
            "minimum_particle_free_relation_gain": float(
                config.minimum_free_relation_gain
            ),
            "joint_loss_and_free_generation_required": True,
            "throughput_cannot_promote_quality": True,
            "unseen_generation_required_after_statistical_pass": True,
        },
        "promotion_boundary": {
            "checkpoint_saved": False,
            "runtime_install_allowed": False,
            "base_quality_promoted": False,
            "continual_learning_claimed": False,
            "sustained_runtime_claimed": False,
        },
    }


def run_particle_field_falsification(
    *,
    tokenizer_checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    output_path: str | Path,
    config: ParticleFieldFalsificationConfig = ParticleFieldFalsificationConfig(),
    device: str = "auto",
    arm_names: Sequence[str] = ARM_NAMES,
) -> dict[str, Any]:
    resolved = _resolve_device(device)
    if resolved.type != "cuda":
        raise ValueError("V28 matched language execution requires CUDA")
    requested = tuple(dict.fromkeys(str(name) for name in arm_names))
    if not requested or any(name not in ARM_NAMES for name in requested):
        raise ValueError("arm_names must contain valid unique V28 arms")
    if str(config.execution_backend) not in {"eager", "inductor"}:
        raise ValueError("execution_backend must be eager or inductor")
    started = time.perf_counter()
    tokenizer_checkpoint = Path(tokenizer_checkpoint_path)
    relation_cases = Path(relation_cases_path)
    prepared = prepare_matched_language_data(
        tokenizer_checkpoint_path=tokenizer_checkpoint,
        relation_corpus_path=relation_corpus_path,
        relation_cases_path=relation_cases,
        general_train_paths=general_train_paths,
        general_eval_paths=general_eval_paths,
        config=MatchedLanguageDataConfig(
            token_budget=int(config.token_budget),
            sequence_length=int(config.sequence_length),
            batch_size=int(config.batch_size),
            eval_batches=int(config.eval_batches),
            relation_fraction=float(config.relation_fraction),
            seed=int(config.data_seed),
            sample_bytes_per_train_source=int(
                config.sample_bytes_per_train_source
            ),
            sample_bytes_per_eval_source=int(config.sample_bytes_per_eval_source),
            sample_range_count=int(config.sample_range_count),
            schedule_mode=str(config.schedule_mode),
        ),
        device=resolved,
    )
    if int(config.relation_case_limit) > 0:
        prepared = replace(
            prepared,
            cases=prepared.cases[: int(config.relation_case_limit)],
        )
    if int(prepared.tokenizer.vocab_size) != 8192:
        raise ValueError("V28 exact parameter match requires the 8,192-token BPE")
    output = Path(output_path)
    rows: dict[str, Mapping[str, Any]] = {}
    executed: list[str] = []
    previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    try:
        for arm in requested:
            torch.manual_seed(int(config.model_seed))
            torch.cuda.manual_seed_all(int(config.model_seed))
            model = build_arm_model(
                arm,
                vocab_size=int(prepared.tokenizer.vocab_size),
                config=config,
            )
            initial_parameters = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            model = model.to(resolved)
            model.eval()
            initial_heldout = evaluate_language_model(
                model,
                prepared.eval_batches,
            )
            model.train()
            training_config = _training_config(config)
            warm_batch = prepared.staged.batch(0, resolved)
            print(f"[particle-field-v28] compiling {arm}", flush=True)
            training_loss, execution = _prepare_language_loss_backend(
                model,
                warm_batch,
                training_config,
            )
            print(f"[particle-field-v28] training {arm}", flush=True)
            row = run_matched_training_arm(
                arm,
                architecture=(
                    "causal_transformer"
                    if arm == "transformer"
                    else "positive_particle_field"
                ),
                model=model,
                initial_state=initial_parameters,
                training_loss=training_loss,
                execution=execution,
                allocated_compile_seconds=float(execution["compile_seconds"]),
                prepared=prepared,
                training_config=training_config,
                gradient_clip=float(config.gradient_clip),
                precision=str(config.precision),
                relation_eval_batch_size=int(config.relation_eval_batch_size),
                model_seed=int(config.model_seed),
                device=resolved,
                progress_prefix="particle-field-v28",
                diagnostic_builder=_arm_diagnostics,
                extra_row={"initial_heldout": initial_heldout},
            )
            rows[arm] = row
            executed.append(arm)
            report = _assemble_report(
                config=config,
                prepared=prepared,
                arms=rows,
                executed_arms=executed,
                tokenizer_checkpoint=tokenizer_checkpoint,
                relation_cases=relation_cases,
                elapsed_seconds=time.perf_counter() - started,
            )
            write_json_report_with_readme(
                output,
                report,
                title="MARULHO V28 Particle-Field Falsification",
            )
            print(
                f"[particle-field-v28] {arm} loss="
                f"{row['heldout']['heldout_loss']:.4f} free="
                f"{row['relation']['generation_exact_accuracy']:.4f}",
                flush=True,
            )
            del training_loss, model
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.set_float32_matmul_precision(previous_precision)
    report = _assemble_report(
        config=config,
        prepared=prepared,
        arms=rows,
        executed_arms=executed,
        tokenizer_checkpoint=tokenizer_checkpoint,
        relation_cases=relation_cases,
        elapsed_seconds=time.perf_counter() - started,
    )
    write_json_report_with_readme(
        output,
        report,
        title="MARULHO V28 Particle-Field Falsification",
    )
    print(f"[particle-field-v28] decision {report['decision']}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--relation-corpus", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--general-train", action="append", type=Path, required=True)
    parser.add_argument("--general-eval", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=16_777_216)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--relation-case-limit", type=int, default=0)
    parser.add_argument("--train-sample-mib", type=int, default=64)
    parser.add_argument("--eval-sample-mib", type=int, default=32)
    parser.add_argument("--arm", action="append", choices=ARM_NAMES, default=[])
    parser.add_argument(
        "--execution-backend",
        choices=("eager", "inductor"),
        default="inductor",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_particle_field_falsification(
        tokenizer_checkpoint_path=args.tokenizer_checkpoint,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=args.general_train,
        general_eval_paths=args.general_eval,
        output_path=args.output,
        config=ParticleFieldFalsificationConfig(
            token_budget=max(1, int(args.token_budget)),
            batch_size=max(1, int(args.batch_size)),
            eval_batches=max(1, int(args.eval_batches)),
            relation_case_limit=max(0, int(args.relation_case_limit)),
            sample_bytes_per_train_source=max(1, int(args.train_sample_mib))
            * 1024
            * 1024,
            sample_bytes_per_eval_source=max(1, int(args.eval_sample_mib))
            * 1024
            * 1024,
            execution_backend=str(args.execution_backend),
        ),
        device=args.device,
        arm_names=tuple(args.arm) or ARM_NAMES,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
