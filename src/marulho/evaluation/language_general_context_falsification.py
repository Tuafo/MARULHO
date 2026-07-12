"""Test general-first Muon training and longer context against the V29 base."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import gc
import hashlib
import json
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
from marulho.evaluation.language_muon_reproduction import _checkpoint_fidelity
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
    load_language_model_checkpoint,
    save_language_model_checkpoint,
)
from marulho.training.language_muon import build_language_muon


SURFACE = "marulho_general_context_falsification.v1"
ARTIFACT_KIND = "marulho_general_context_falsification"
ARM_NAMES = ("general72", "general256")
ADVANCE_DECISION = "save_v30_general_context_candidate_for_unseen_generation"
RETIRE_DECISION = "retire_v30_general_context_no_general_loss_win"
INVALID_DECISION = "invalid_v30_general_context_evidence"


@dataclass(frozen=True)
class GeneralContextFalsificationConfig:
    token_budget: int = 16_777_216
    common_sequence_length: int = 72
    common_batch_size: int = 32
    long_sequence_length: int = 256
    long_batch_size: int = 9
    eval_batches: int = 16
    relation_eval_batch_size: int = 8
    relation_case_limit: int = 0
    relation_fraction: float = 0.0
    learning_rate: float = 1.0e-3
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
    minimum_common_loss_gain: float = 0.05
    minimum_long_context_gain_over_short: float = 0.02


def arm_shape(
    arm: str,
    config: GeneralContextFalsificationConfig,
) -> tuple[int, int]:
    if arm == "general72":
        return int(config.common_sequence_length), int(config.common_batch_size)
    if arm == "general256":
        return int(config.long_sequence_length), int(config.long_batch_size)
    raise ValueError(f"unknown V30 arm: {arm}")


def build_model(
    *,
    vocab_size: int,
    context_length: int,
    config: GeneralContextFalsificationConfig,
) -> MarulhoLanguageModel:
    return MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=int(vocab_size),
            embedding_dim=int(config.width),
            state_dim=int(config.width),
            state_layers=int(config.layers),
            attention_heads=int(config.heads),
            transformer_context_length=int(context_length),
            transformer_mlp_ratio=float(config.mlp_ratio),
            transformer_dropout=0.0,
            tie_embeddings=True,
            active_language_path=f"marulho_transformer_v30_general_{context_length}",
        )
    )


def _training_config(
    config: GeneralContextFalsificationConfig,
    *,
    sequence_length: int,
    batch_size: int,
) -> LanguageTrainingExperimentConfig:
    return LanguageTrainingExperimentConfig(
        sequence_length=int(sequence_length),
        batch_size=int(batch_size),
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


def _prepare_data(
    *,
    tokenizer_checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    sequence_length: int,
    batch_size: int,
    config: GeneralContextFalsificationConfig,
    device: torch.device,
) -> PreparedMatchedLanguageData:
    return prepare_matched_language_data(
        tokenizer_checkpoint_path=tokenizer_checkpoint_path,
        relation_corpus_path=relation_corpus_path,
        relation_cases_path=relation_cases_path,
        general_train_paths=general_train_paths,
        general_eval_paths=general_eval_paths,
        config=MatchedLanguageDataConfig(
            token_budget=int(config.token_budget),
            sequence_length=int(sequence_length),
            batch_size=int(batch_size),
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
        device=device,
    )


def model_state_sha256(model: MarulhoLanguageModel) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def select_v30_candidate(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    baseline_common_loss: float,
    config: GeneralContextFalsificationConfig,
) -> str | None:
    if set(arms) != set(ARM_NAMES):
        return None
    if not all(
        bool(row["all_parameters_received_final_gradient"])
        for row in arms.values()
    ):
        return None
    passing = {
        name
        for name, row in arms.items()
        if float(baseline_common_loss)
        - float(row["common_context_heldout"]["heldout_loss"])
        >= float(config.minimum_common_loss_gain)
    }
    if not passing:
        return None
    if passing == {"general256"}:
        return "general256"
    if "general72" in passing and "general256" not in passing:
        return "general72"
    short_loss = float(
        arms["general72"]["common_context_heldout"]["heldout_loss"]
    )
    long_loss = float(
        arms["general256"]["common_context_heldout"]["heldout_loss"]
    )
    if short_loss - long_loss >= float(config.minimum_long_context_gain_over_short):
        return "general256"
    return "general72"


def v30_decision(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    baseline_common_loss: float,
    config: GeneralContextFalsificationConfig,
    checkpoint_fidelity_passed: bool,
) -> str:
    if set(arms) != set(ARM_NAMES):
        return INVALID_DECISION
    if not all(
        bool(row["all_parameters_received_final_gradient"])
        for row in arms.values()
    ):
        return INVALID_DECISION
    selected = select_v30_candidate(
        arms,
        baseline_common_loss=float(baseline_common_loss),
        config=config,
    )
    if selected is None:
        return RETIRE_DECISION
    if not bool(checkpoint_fidelity_passed):
        return INVALID_DECISION
    return ADVANCE_DECISION


def run_general_context_falsification(
    *,
    baseline_checkpoint_path: str | Path,
    baseline_reproduction_report_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    checkpoint_output_path: str | Path,
    report_output_path: str | Path,
    config: GeneralContextFalsificationConfig = GeneralContextFalsificationConfig(),
    device: str = "auto",
) -> dict[str, Any]:
    started = time.perf_counter()
    resolved = _resolve_device(device)
    if resolved.type != "cuda":
        raise ValueError("V30 general/context falsification requires CUDA")
    if int(config.common_sequence_length) * int(config.common_batch_size) != int(
        config.long_sequence_length
    ) * int(config.long_batch_size):
        raise ValueError("V30 arms must process the same tokens per optimizer step")
    checkpoint_output = Path(checkpoint_output_path)
    if checkpoint_output.exists():
        raise ValueError("V30 checkpoint output already exists")
    baseline_checkpoint = Path(baseline_checkpoint_path)
    baseline_report_path = Path(baseline_reproduction_report_path)
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    if baseline_report.get("decision") != (
        "save_v29_muon_checkpoint_for_unseen_generation"
    ):
        raise ValueError("V30 requires the qualified V29 reproduction report")
    baseline_model, baseline_tokenizer, baseline_metadata = (
        load_language_model_checkpoint(baseline_checkpoint, map_location="cpu")
    )
    if baseline_tokenizer.vocabulary_hash() != baseline_report["checkpoint"][
        "fidelity"
    ]["tokenizer_hash"]:
        raise ValueError("V30 baseline tokenizer does not match reproduction")
    common_prepared = _prepare_data(
        tokenizer_checkpoint_path=baseline_checkpoint,
        relation_corpus_path=relation_corpus_path,
        relation_cases_path=relation_cases_path,
        general_train_paths=general_train_paths,
        general_eval_paths=general_eval_paths,
        sequence_length=int(config.common_sequence_length),
        batch_size=int(config.common_batch_size),
        config=config,
        device=resolved,
    )
    long_prepared = _prepare_data(
        tokenizer_checkpoint_path=baseline_checkpoint,
        relation_corpus_path=relation_corpus_path,
        relation_cases_path=relation_cases_path,
        general_train_paths=general_train_paths,
        general_eval_paths=general_eval_paths,
        sequence_length=int(config.long_sequence_length),
        batch_size=int(config.long_batch_size),
        config=config,
        device=resolved,
    )
    if int(config.relation_case_limit) > 0:
        common_prepared = replace(
            common_prepared,
            cases=common_prepared.cases[: int(config.relation_case_limit)],
        )
        long_prepared = replace(
            long_prepared,
            cases=long_prepared.cases[: int(config.relation_case_limit)],
        )
    if common_prepared.staged.tokens_per_step != long_prepared.staged.tokens_per_step:
        raise ValueError("V30 staged schedules do not match tokens per step")
    if common_prepared.staged.step_count != long_prepared.staged.step_count:
        raise ValueError("V30 staged schedules do not match optimizer steps")
    baseline_model = baseline_model.to(resolved)
    baseline_common = evaluate_language_model(
        baseline_model,
        common_prepared.eval_batches,
    )
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()

    rows: dict[str, Mapping[str, Any]] = {}
    selected_states: dict[str, Mapping[str, torch.Tensor]] = {}
    initial_hashes: dict[str, str] = {}
    executions: dict[str, Mapping[str, Any]] = {}
    previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    try:
        for arm in ARM_NAMES:
            sequence_length, batch_size = arm_shape(arm, config)
            prepared = common_prepared if arm == "general72" else long_prepared
            torch.manual_seed(int(config.model_seed))
            torch.cuda.manual_seed_all(int(config.model_seed))
            model = build_model(
                vocab_size=int(prepared.tokenizer.vocab_size),
                context_length=sequence_length,
                config=config,
            ).to(resolved)
            initial_hashes[arm] = model_state_sha256(model)
            initial_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            model.eval()
            initial_heldout = evaluate_language_model(model, prepared.eval_batches)
            model.train()
            training_config = _training_config(
                config,
                sequence_length=sequence_length,
                batch_size=batch_size,
            )
            warm_batch = prepared.staged.batch(0, resolved)
            print(f"[general-context-v30] compiling {arm}", flush=True)
            training_loss, execution = _prepare_language_loss_backend(
                model,
                warm_batch,
                training_config,
            )
            executions[arm] = execution

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

            print(f"[general-context-v30] training {arm}", flush=True)
            row = run_matched_training_arm(
                arm,
                architecture="causal_transformer_general_first",
                model=model,
                initial_state=initial_state,
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
                progress_prefix="general-context-v30",
                extra_row={
                    "initial_heldout": initial_heldout,
                    "training_context_length": sequence_length,
                    "training_batch_size": batch_size,
                    "relation_training_fraction": 0.0,
                },
                optimizer_builder=optimizer_builder,
            )
            row["common_context_heldout"] = evaluate_language_model(
                model,
                common_prepared.eval_batches,
            )
            rows[arm] = row
            selected_states[arm] = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            print(
                f"[general-context-v30] {arm} common_loss="
                f"{row['common_context_heldout']['heldout_loss']:.4f} free="
                f"{row['relation']['generation_exact_accuracy']:.4f}",
                flush=True,
            )
            del training_loss, model
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.set_float32_matmul_precision(previous_precision)

    if len(set(initial_hashes.values())) != 1:
        raise RuntimeError("V30 context arms did not share exact initial tensors")
    selected_arm = select_v30_candidate(
        rows,
        baseline_common_loss=float(baseline_common["heldout_loss"]),
        config=config,
    )
    checkpoint_fidelity: dict[str, Any] = {"performed": False, "passed": False}
    checkpoint_sha256 = None
    if selected_arm is not None:
        selected_context, _ = arm_shape(selected_arm, config)
        selected_model = build_model(
            vocab_size=int(common_prepared.tokenizer.vocab_size),
            context_length=selected_context,
            config=config,
        ).to(resolved)
        selected_model.load_state_dict(dict(selected_states[selected_arm]), strict=True)
        selected_loss = float(
            rows[selected_arm]["common_context_heldout"]["heldout_loss"]
        )
        save_language_model_checkpoint(
            checkpoint_output,
            selected_model,
            common_prepared.tokenizer,
            metadata={
                "decision": ADVANCE_DECISION,
                "checkpoint_reproduction": True,
                "selected_arm": selected_arm,
                "processed_tokens": int(rows[selected_arm]["processed_tokens"]),
                "heldout_loss": selected_loss,
                "free_relation_accuracy": float(
                    rows[selected_arm]["relation"]["generation_exact_accuracy"]
                ),
                "baseline_checkpoint_sha256": sha256_file(baseline_checkpoint),
                "optimizer": dict(rows[selected_arm]["optimizer"]),
                "optimizer_state_saved": False,
                "external_llm_used": False,
            },
        )
        checkpoint_fidelity, _, _, _ = _checkpoint_fidelity(
            selected_model,
            checkpoint_output,
            expected_tokenizer_hash=common_prepared.tokenizer.vocabulary_hash(),
            sample_input_ids=common_prepared.eval_batches[0].input_ids,
            device=resolved,
        )
        checkpoint_fidelity["performed"] = True
        checkpoint_sha256 = sha256_file(checkpoint_output)
        if not bool(checkpoint_fidelity["passed"]):
            checkpoint_output.unlink(missing_ok=True)
            checkpoint_sha256 = None
    decision = v30_decision(
        rows,
        baseline_common_loss=float(baseline_common["heldout_loss"]),
        config=config,
        checkpoint_fidelity_passed=bool(checkpoint_fidelity["passed"]),
    )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "configuration": asdict(config),
        "baseline": {
            "checkpoint_path": str(baseline_checkpoint),
            "checkpoint_sha256": sha256_file(baseline_checkpoint),
            "reproduction_report_path": str(baseline_report_path),
            "reproduction_report_sha256": sha256_file(baseline_report_path),
            "metadata": dict(baseline_metadata),
            "common_context_heldout": baseline_common,
        },
        "tokenizer": {
            "vocab_size": int(common_prepared.tokenizer.vocab_size),
            "vocabulary_hash": common_prepared.tokenizer.vocabulary_hash(),
        },
        "matched_truth": {
            "parameter_counts": {
                name: int(row["parameters"]) for name, row in rows.items()
            },
            "initial_state_hashes": initial_hashes,
            "initial_state_tensors_equal": len(set(initial_hashes.values())) == 1,
            "tokens_per_step": int(common_prepared.staged.tokens_per_step),
            "optimizer_steps": int(common_prepared.staged.step_count),
            "same_general_source_ranges": (
                common_prepared.source_selections["general_train"]
                == long_prepared.source_selections["general_train"]
            ),
            "labels_metrics_only": True,
            "relation_updates": 0,
        },
        "schedules": {
            "general72": {
                "sha256": common_prepared.schedule_sha256,
                "source_selections": common_prepared.source_selections,
            },
            "general256": {
                "sha256": long_prepared.schedule_sha256,
                "source_selections": long_prepared.source_selections,
            },
        },
        "arms": dict(rows),
        "executions": executions,
        "selection": {
            "selected_arm": selected_arm,
            "baseline_common_loss": float(baseline_common["heldout_loss"]),
            "minimum_common_loss_gain": float(config.minimum_common_loss_gain),
            "minimum_long_context_gain_over_short": float(
                config.minimum_long_context_gain_over_short
            ),
            "relation_behavior_is_diagnostic_not_selection": True,
        },
        "checkpoint": {
            "path": str(checkpoint_output) if checkpoint_output.exists() else None,
            "sha256": checkpoint_sha256,
            "saved": checkpoint_output.exists(),
            "optimizer_state_saved": False,
            "fidelity": checkpoint_fidelity,
        },
        "decision": decision,
        "promotion_boundary": {
            "unseen_generation_admitted": decision == ADVANCE_DECISION,
            "base_quality_promoted": False,
            "runtime_install_allowed": False,
            "continual_learning_claimed": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_report_with_readme(
        report_output_path,
        report,
        title="MARULHO V30 General-Context Falsification",
    )
    print(f"[general-context-v30] decision {decision}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--relation-corpus", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--general-train", action="append", type=Path, required=True)
    parser.add_argument("--general-eval", action="append", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=16_777_216)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--relation-case-limit", type=int, default=0)
    parser.add_argument("--train-sample-mib", type=int, default=64)
    parser.add_argument("--eval-sample-mib", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = run_general_context_falsification(
        baseline_checkpoint_path=args.baseline_checkpoint,
        baseline_reproduction_report_path=args.baseline_report,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=args.general_train,
        general_eval_paths=args.general_eval,
        checkpoint_output_path=args.checkpoint_output,
        report_output_path=args.report_output,
        config=GeneralContextFalsificationConfig(
            token_budget=max(1, int(args.token_budget)),
            eval_batches=max(1, int(args.eval_batches)),
            relation_case_limit=max(0, int(args.relation_case_limit)),
            sample_bytes_per_train_source=max(1, int(args.train_sample_mib))
            * 1024
            * 1024,
            sample_bytes_per_eval_source=max(1, int(args.eval_sample_mib))
            * 1024
            * 1024,
        ),
        device=args.device,
    )
    return 0 if report["decision"] == ADVANCE_DECISION else 1


if __name__ == "__main__":
    raise SystemExit(main())
