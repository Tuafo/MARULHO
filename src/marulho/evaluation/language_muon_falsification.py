"""Compare AdamW and Muon learning geometry on one fixed language model."""

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
from marulho.training.language_muon import build_language_muon


SURFACE = "marulho_muon_falsification.v1"
ARTIFACT_KIND = "marulho_muon_falsification"
ARM_NAMES = ("adamw_3e4", "adamw_1e3", "muon_3e4", "muon_1e3")
MINIMUM_DECISION_TOKENS = 16_000_000


@dataclass(frozen=True)
class MuonFalsificationConfig:
    token_budget: int = 16_777_216
    sequence_length: int = 72
    batch_size: int = 32
    eval_batches: int = 16
    relation_eval_batch_size: int = 8
    relation_case_limit: int = 0
    relation_fraction: float = 0.20
    incumbent_learning_rate: float = 3.0e-4
    reference_learning_rate: float = 1.0e-3
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    data_seed: int = 16121
    model_seed: int = 16131
    sample_bytes_per_train_source: int = 64 * 1024 * 1024
    sample_bytes_per_eval_source: int = 32 * 1024 * 1024
    sample_range_count: int = 16
    schedule_mode: str = "indexed_host"
    execution_backend: str = "inductor"
    compile_loss_tolerance: float = 1.0e-3
    width: int = 512
    layers: int = 4
    heads: int = 8
    mlp_ratio: float = 4.0
    maximum_parameter_delta_fraction: float = 0.0
    minimum_loss_gain: float = 0.01
    minimum_free_relation_gain: float = 0.02


def arm_learning_rate(arm: str, config: MuonFalsificationConfig) -> float:
    if arm.endswith("3e4"):
        return float(config.incumbent_learning_rate)
    if arm.endswith("1e3"):
        return float(config.reference_learning_rate)
    raise ValueError(f"unknown V29 optimizer arm: {arm}")


def arm_optimizer_kind(arm: str) -> str:
    if arm.startswith("adamw_"):
        return "adamw"
    if arm.startswith("muon_"):
        return "muon"
    raise ValueError(f"unknown V29 optimizer arm: {arm}")


def build_model(*, vocab_size: int, config: MuonFalsificationConfig):
    return MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=int(vocab_size),
            embedding_dim=int(config.width),
            state_dim=int(config.width),
            state_layers=int(config.layers),
            attention_heads=int(config.heads),
            transformer_context_length=int(config.sequence_length),
            transformer_mlp_ratio=float(config.mlp_ratio),
            transformer_dropout=0.0,
            tie_embeddings=True,
            active_language_path="marulho_transformer_v29_optimizer_control",
        )
    )


def _training_config(
    config: MuonFalsificationConfig,
    *,
    learning_rate: float,
) -> LanguageTrainingExperimentConfig:
    return LanguageTrainingExperimentConfig(
        sequence_length=int(config.sequence_length),
        batch_size=int(config.batch_size),
        learning_rate=float(learning_rate),
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


def _best_arm(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    optimizer_kind: str,
) -> tuple[str, Mapping[str, Any]] | None:
    candidates = [
        (name, row)
        for name, row in arms.items()
        if arm_optimizer_kind(name) == optimizer_kind
    ]
    if len(candidates) != 2:
        return None
    return min(
        candidates,
        key=lambda item: float(item[1]["heldout"]["heldout_loss"]),
    )


def optimizer_comparison(
    arms: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    best_adamw = _best_arm(arms, optimizer_kind="adamw")
    best_muon = _best_arm(arms, optimizer_kind="muon")
    if best_adamw is None or best_muon is None:
        return None
    adamw_name, adamw = best_adamw
    muon_name, muon = best_muon
    return {
        "best_adamw_arm": adamw_name,
        "best_muon_arm": muon_name,
        "adamw_heldout_loss": float(adamw["heldout"]["heldout_loss"]),
        "muon_heldout_loss": float(muon["heldout"]["heldout_loss"]),
        "muon_loss_gain": float(adamw["heldout"]["heldout_loss"])
        - float(muon["heldout"]["heldout_loss"]),
        "adamw_free_relation_accuracy": float(
            adamw["relation"]["generation_exact_accuracy"]
        ),
        "muon_free_relation_accuracy": float(
            muon["relation"]["generation_exact_accuracy"]
        ),
        "muon_free_relation_gain": float(
            muon["relation"]["generation_exact_accuracy"]
        )
        - float(adamw["relation"]["generation_exact_accuracy"]),
    }


def muon_decision(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    processed_tokens: int,
    parameter_delta_fraction: float,
    config: MuonFalsificationConfig,
) -> str:
    if set(arms) != set(ARM_NAMES):
        return "incomplete_v29_missing_optimizer_arm"
    if float(parameter_delta_fraction) > float(
        config.maximum_parameter_delta_fraction
    ):
        return "invalid_v29_parameter_mismatch"
    if not all(
        bool(row["all_parameters_received_final_gradient"])
        for row in arms.values()
    ):
        return "invalid_v29_incomplete_gradient_coverage"
    if int(processed_tokens) < MINIMUM_DECISION_TOKENS:
        return "diagnostic_v29_below_durable_token_floor"
    comparison = optimizer_comparison(arms)
    if comparison is None:
        return "incomplete_v29_missing_optimizer_comparison"
    loss_pass = float(comparison["muon_loss_gain"]) >= float(
        config.minimum_loss_gain
    )
    free_pass = float(comparison["muon_free_relation_gain"]) >= float(
        config.minimum_free_relation_gain
    )
    if loss_pass and free_pass:
        return "advance_v29_muon_to_unseen_generation"
    if loss_pass or free_pass:
        return "redesign_v29_disjoint_optimizer_signal"
    return "retire_v29_muon_no_joint_language_win"


def _assemble_report(
    *,
    config: MuonFalsificationConfig,
    prepared: PreparedMatchedLanguageData,
    arms: Mapping[str, Mapping[str, Any]],
    executed_arms: Sequence[str],
    tokenizer_checkpoint: Path,
    relation_cases: Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    counts = {name: int(row["parameters"]) for name, row in arms.items()}
    parameter_delta = max(counts.values()) - min(counts.values()) if counts else None
    parameter_delta_fraction = (
        float(parameter_delta) / float(next(iter(counts.values())))
        if parameter_delta is not None and counts
        else None
    )
    processed_tokens = (
        min(int(row["processed_tokens"]) for row in arms.values()) if arms else 0
    )
    comparison = optimizer_comparison(arms)
    decision = muon_decision(
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
            "candidate": "Muon matrix-orthogonalized hidden-layer updates",
            "paper": "https://arxiv.org/abs/2502.16982",
            "official_reference": (
                "https://github.com/MoonshotAI/Moonlight/blob/master/"
                "examples/toy_train.py"
            ),
            "external_weights_loaded": False,
            "published_result_treated_as_local_evidence": False,
            "local_question": (
                "whether Muon improves MARULHO's 20.976M Transformer at its "
                "single-RTX-3060 data and batch scale"
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
        "model_match": {
            "counts": counts,
            "absolute_delta": parameter_delta,
            "delta_fraction": parameter_delta_fraction,
            "maximum_delta_fraction": float(
                config.maximum_parameter_delta_fraction
            ),
            "same_initial_weights": True,
            "same_architecture": True,
            "passed": parameter_delta == 0,
        },
        "arms": dict(arms),
        "arms_executed_this_run": list(executed_arms),
        "optimizer_comparison": comparison,
        "experiment_wall_seconds_this_run": float(elapsed_seconds),
        "decision": decision,
        "decision_contract": {
            "minimum_processed_tokens": MINIMUM_DECISION_TOKENS,
            "comparison_uses_best_learning_rate_per_optimizer": True,
            "minimum_muon_loss_gain": float(config.minimum_loss_gain),
            "minimum_muon_free_relation_gain": float(
                config.minimum_free_relation_gain
            ),
            "joint_loss_and_free_generation_required": True,
            "throughput_cannot_promote_quality": True,
            "unseen_generation_required_after_statistical_pass": True,
        },
        "promotion_boundary": {
            "checkpoint_saved": False,
            "runtime_install_allowed": False,
            "optimizer_promoted": False,
            "continual_learning_claimed": False,
            "sustained_runtime_claimed": False,
        },
    }


def run_muon_falsification(
    *,
    tokenizer_checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    output_path: str | Path,
    config: MuonFalsificationConfig = MuonFalsificationConfig(),
    device: str = "auto",
    arm_names: Sequence[str] = ARM_NAMES,
) -> dict[str, Any]:
    resolved = _resolve_device(device)
    if resolved.type != "cuda":
        raise ValueError("V29 matched optimizer execution requires CUDA")
    requested = tuple(dict.fromkeys(str(name) for name in arm_names))
    if not requested or any(name not in ARM_NAMES for name in requested):
        raise ValueError("arm_names must contain valid unique V29 arms")
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
        raise ValueError("V29 optimizer test requires the 8,192-token BPE")
    torch.manual_seed(int(config.model_seed))
    torch.cuda.manual_seed_all(int(config.model_seed))
    model = build_model(
        vocab_size=int(prepared.tokenizer.vocab_size),
        config=config,
    ).to(resolved)
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    model.eval()
    initial_heldout = evaluate_language_model(model, prepared.eval_batches)
    model.train()
    warm_batch = prepared.staged.batch(0, resolved)
    compile_config = _training_config(
        config,
        learning_rate=float(config.incumbent_learning_rate),
    )
    print("[muon-v29] compiling shared Transformer", flush=True)
    training_loss, execution = _prepare_language_loss_backend(
        model,
        warm_batch,
        compile_config,
    )
    output = Path(output_path)
    rows: dict[str, Mapping[str, Any]] = {}
    executed: list[str] = []
    previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    try:
        for arm in requested:
            learning_rate = arm_learning_rate(arm, config)
            training_config = _training_config(
                config,
                learning_rate=learning_rate,
            )
            optimizer_builder = None
            if arm_optimizer_kind(arm) == "muon":

                def optimizer_builder(model_value, config_value):
                    return build_language_muon(
                        model_value,
                        learning_rate=float(config_value.learning_rate),
                        weight_decay=float(config_value.weight_decay),
                        adamw_betas=(
                            float(config_value.adam_beta1),
                            float(config_value.adam_beta2),
                        ),
                    )

            print(f"[muon-v29] training {arm}", flush=True)
            row = run_matched_training_arm(
                arm,
                architecture="causal_transformer_optimizer_control",
                model=model,
                initial_state=initial_state,
                training_loss=training_loss,
                execution=execution,
                allocated_compile_seconds=(
                    float(execution["compile_seconds"]) / float(len(requested))
                ),
                prepared=prepared,
                training_config=training_config,
                gradient_clip=float(config.gradient_clip),
                precision=str(config.precision),
                relation_eval_batch_size=int(config.relation_eval_batch_size),
                model_seed=int(config.model_seed),
                device=resolved,
                progress_prefix="muon-v29",
                extra_row={
                    "initial_heldout": initial_heldout,
                    "optimizer_kind": arm_optimizer_kind(arm),
                    "peak_learning_rate": learning_rate,
                },
                optimizer_builder=optimizer_builder,
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
                title="MARULHO V29 Muon Falsification",
            )
            print(
                f"[muon-v29] {arm} loss={row['heldout']['heldout_loss']:.4f} "
                f"free={row['relation']['generation_exact_accuracy']:.4f}",
                flush=True,
            )
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
        title="MARULHO V29 Muon Falsification",
    )
    print(f"[muon-v29] decision {report['decision']}", flush=True)
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
    run_muon_falsification(
        tokenizer_checkpoint_path=args.tokenizer_checkpoint,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=args.general_train,
        general_eval_paths=args.general_eval,
        output_path=args.output,
        config=MuonFalsificationConfig(
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
