"""Matched V13 multi-horizon future-prediction continuation for V11."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from marulho.evaluation.language_hashed_micro_expert_continuation import (
    _expand_context_with_parity,
    _validate_parent,
)
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
from marulho.training.language_future_prediction import (
    FuturePredictionConfig,
    build_future_prediction_model,
    future_prediction_objective_report,
    strip_future_prediction_heads,
)
from marulho.training.language_hashed_micro_experts import (
    load_hashed_micro_expert_checkpoint,
    save_hashed_micro_expert_checkpoint,
)
from marulho.training.language_model import LanguageBatch, evaluate_language_model


SURFACE = "marulho_future_prediction_experiment.v1"
ARTIFACT_KIND = "marulho_future_prediction_experiment"
SAVE_DECISION = "save_v13_future_prediction_candidate_for_unseen_generation"
MATCHED_CONTROL_FIELDS = (
    "additional_token_budget",
    "sequence_length",
    "batch_size",
    "eval_batches",
    "relation_eval_batch_size",
    "learning_rate",
    "minimum_learning_rate_fraction",
    "warmup_fraction",
    "weight_decay",
    "gradient_clip",
    "precision",
    "seed",
    "sample_bytes_per_train_source",
    "sample_bytes_per_eval_source",
    "sample_range_count",
    "execution_backend",
    "compile_loss_tolerance",
)


@dataclass(frozen=True)
class FuturePredictionExperimentConfig:
    additional_token_budget: int = 67_108_864
    sequence_length: int = 256
    batch_size: int = 40
    eval_batches: int = 16
    relation_eval_batch_size: int = 64
    learning_rate: float = 7.5e-5
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.02
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    seed: int = 2028
    sample_bytes_per_train_source: int = 64 * 1024 * 1024
    sample_bytes_per_eval_source: int = 32 * 1024 * 1024
    sample_range_count: int = 32
    execution_backend: str = "inductor"
    compile_loss_tolerance: float = 1.0e-3
    future_horizons: tuple[int, ...] = (2, 4, 8)
    auxiliary_weight: float = 0.25
    objective_eval_batches: int = 4
    minimum_control_loss_gain: float = 0.02


def _data_config(
    config: FuturePredictionExperimentConfig,
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
    config: FuturePredictionExperimentConfig,
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


def future_prediction_decision(
    *,
    control_heldout_loss: float,
    candidate_heldout_loss: float,
    processed_tokens: int,
    requested_tokens: int,
    minimum_gain: float = 0.02,
) -> str:
    if int(processed_tokens) < int(requested_tokens):
        return "incomplete_v13_future_prediction"
    gain = float(control_heldout_loss) - float(candidate_heldout_loss)
    if gain >= float(minimum_gain):
        return SAVE_DECISION
    if gain > 0.0:
        return "retire_v13_future_prediction_weak_control_gain"
    return "retire_v13_future_prediction_no_control_gain"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("matched control report must contain a JSON object")
    return payload


def _validate_matched_control(
    control: Mapping[str, Any],
    *,
    parent_sha256: str,
    schedule_sha256: str,
    config: FuturePredictionExperimentConfig,
) -> dict[str, Any]:
    if control.get("artifact_kind") != (
        "marulho_hashed_micro_expert_general_continuation"
    ):
        raise ValueError("V13 control report has the wrong artifact kind")
    if control.get("decision") != (
        "save_v11_general_continuation_for_unseen_generation"
    ):
        raise ValueError("V13 control report is not a saved V11 continuation")
    parent = control.get("parent")
    schedule = control.get("schedule")
    observed = control.get("configuration")
    if not isinstance(parent, Mapping) or parent.get("sha256") != parent_sha256:
        raise ValueError("V13 control uses a different parent checkpoint")
    if not isinstance(schedule, Mapping) or (
        schedule.get("sha256") != schedule_sha256
    ):
        raise ValueError("V13 control uses a different data schedule")
    if not isinstance(observed, Mapping):
        raise ValueError("V13 control lacks configuration evidence")
    expected = asdict(config)
    mismatches = {
        field: {"control": observed.get(field), "candidate": expected[field]}
        for field in MATCHED_CONTROL_FIELDS
        if observed.get(field) != expected[field]
    }
    if mismatches:
        raise ValueError(f"V13 control configuration mismatch: {mismatches}")
    after = control.get("after")
    if not isinstance(after, Mapping) or not isinstance(after.get("arm"), Mapping):
        raise ValueError("V13 control lacks its completed arm")
    arm = after["arm"]
    heldout = arm.get("heldout")
    if not isinstance(heldout, Mapping):
        raise ValueError("V13 control lacks heldout evidence")
    processed_tokens = int(arm.get("processed_tokens") or 0)
    if processed_tokens < int(config.additional_token_budget):
        raise ValueError("V13 control did not complete its token budget")
    return {
        "heldout_loss": float(heldout["heldout_loss"]),
        "heldout_perplexity": float(heldout["heldout_perplexity"]),
        "processed_tokens": processed_tokens,
        "cumulative_processed_tokens": int(after["cumulative_processed_tokens"]),
        "schedule_sha256": str(schedule["sha256"]),
    }


def run_future_prediction_experiment(
    *,
    parent_checkpoint_path: str | Path,
    matched_control_report_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    output_path: str | Path,
    checkpoint_output_path: str | Path | None = None,
    config: FuturePredictionExperimentConfig = (
        FuturePredictionExperimentConfig()
    ),
    device: str = "auto",
) -> dict[str, Any]:
    if int(config.additional_token_budget) < 1:
        raise ValueError("additional_token_budget must be positive")
    if int(config.objective_eval_batches) < 1:
        raise ValueError("objective_eval_batches must be positive")
    if float(config.minimum_control_loss_gain) <= 0.0:
        raise ValueError("minimum_control_loss_gain must be positive")
    resolved = _resolve_device(device)
    if config.execution_backend == "inductor" and resolved.type != "cuda":
        raise ValueError("Inductor V13 training is admitted only on CUDA")

    parent_path = Path(parent_checkpoint_path)
    control_path = Path(matched_control_report_path)
    output = Path(output_path)
    checkpoint_output = (
        None if checkpoint_output_path is None else Path(checkpoint_output_path)
    )
    if checkpoint_output is not None and (
        checkpoint_output.resolve() == parent_path.resolve()
    ):
        raise ValueError("V13 checkpoint cannot overwrite its parent")
    started = time.perf_counter()
    base_model, parent_tokenizer, parent_metadata = (
        load_hashed_micro_expert_checkpoint(parent_path, map_location="cpu")
    )
    parent_processed_tokens = _validate_parent(base_model, parent_metadata)
    parent_sha256 = sha256_file(parent_path)
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
        raise RuntimeError("V13 data tokenizer differs from parent")
    control_payload = _load_json(control_path)
    control_record = _validate_matched_control(
        control_payload,
        parent_sha256=parent_sha256,
        schedule_sha256=prepared.schedule_sha256,
        config=config,
    )
    parent_context_length = int(base_model.context_length)
    base_model, context_expansion = _expand_context_with_parity(
        base_model,
        sequence_length=int(config.sequence_length),
        parity_input_ids=prepared.eval_batches[0].input_ids,
    )
    future_config = FuturePredictionConfig(
        horizons=tuple(int(value) for value in config.future_horizons),
        auxiliary_weight=float(config.auxiliary_weight),
    )
    base_model.eval()
    model = build_future_prediction_model(base_model, future_config).eval()
    attachment_ids = prepared.eval_batches[0].input_ids[:2].detach().cpu()
    with torch.no_grad():
        base_logits = base_model(
            attachment_ids,
            collect_telemetry=False,
        )["logits"]
        attached_logits = model(
            attachment_ids,
            collect_telemetry=False,
        )["logits"]
    attachment_exact = bool(torch.equal(base_logits, attached_logits))
    attachment_delta = float((base_logits - attached_logits).abs().max().item())
    if not attachment_exact:
        raise RuntimeError(
            "V13 training-head attachment changed base logits: "
            f"max_abs_delta={attachment_delta}"
        )
    attachment_record = {
        "base_logits_exact": attachment_exact,
        "maximum_absolute_logit_delta": attachment_delta,
        **model.training_parameter_report(),
    }
    del base_model

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
    objective_batches = prepared.eval_batches[
        : min(int(config.objective_eval_batches), len(prepared.eval_batches))
    ]
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
        objective_before = future_prediction_objective_report(
            model,
            objective_batches,
        )
        model.train()
        training_loss, execution = _prepare_language_loss_backend(
            model,
            warm_batch,
            training_config,
        )
        row = run_matched_training_arm(
            "dyadic_future_prediction",
            architecture="hashed_micro_experts_future_prediction",
            model=model,
            initial_state=initial_state,
            training_loss=training_loss,
            execution={
                **execution,
                "training_objective": "next_token_plus_dyadic_future_tokens",
                "future_horizons": [
                    int(value) for value in config.future_horizons
                ],
                "auxiliary_weight": float(config.auxiliary_weight),
                "future_heads_inference_persistent": False,
                "matched_control_report_sha256": sha256_file(control_path),
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
            progress_prefix="future-prediction-v13",
            configure_model=lambda active, _name: (
                active.set_hashed_micro_expert_mode("token_hash")
            ),
            diagnostic_builder=None,
            extra_row={
                "parent_checkpoint_sha256": parent_sha256,
                "parent_processed_tokens": parent_processed_tokens,
                "training_mixture": "general_only_equal_source_alternation",
                "relation_updates_scheduled": False,
                "context_expansion": context_expansion,
                "training_head_attachment": attachment_record,
            },
        )
        objective_after = future_prediction_objective_report(
            model,
            objective_batches,
        )
    finally:
        if resolved.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            torch.set_float32_matmul_precision(previous_matmul_precision)

    additional_processed_tokens = int(row["processed_tokens"])
    cumulative_processed_tokens = (
        parent_processed_tokens + additional_processed_tokens
    )
    control_gain = float(control_record["heldout_loss"]) - float(
        row["heldout"]["heldout_loss"]
    )
    decision = future_prediction_decision(
        control_heldout_loss=float(control_record["heldout_loss"]),
        candidate_heldout_loss=float(row["heldout"]["heldout_loss"]),
        processed_tokens=additional_processed_tokens,
        requested_tokens=int(config.additional_token_budget),
        minimum_gain=float(config.minimum_control_loss_gain),
    )

    model.eval()
    inference_model = strip_future_prediction_heads(model).eval()
    strip_ids = prepared.eval_batches[0].input_ids[:2]
    with torch.no_grad():
        training_graph_logits = model(
            strip_ids,
            collect_telemetry=False,
        )["logits"]
        inference_logits = inference_model(
            strip_ids,
            collect_telemetry=False,
        )["logits"]
    strip_exact = bool(torch.equal(training_graph_logits, inference_logits))
    strip_delta = float(
        (training_graph_logits - inference_logits).abs().max().item()
    )
    if not strip_exact:
        raise RuntimeError(
            "Removing V13 future heads changed inference logits: "
            f"max_abs_delta={strip_delta}"
        )
    strip_record = {
        "performed": True,
        "future_heads_persisted": False,
        "inference_logits_exact": strip_exact,
        "maximum_absolute_logit_delta": strip_delta,
        "training_parameters": sum(
            int(value.numel()) for value in model.parameters()
        ),
        "inference_parameters": sum(
            int(value.numel()) for value in inference_model.parameters()
        ),
    }

    checkpoint_record: dict[str, Any] | None = None
    if checkpoint_output is not None:
        if decision == SAVE_DECISION:
            saved = save_hashed_micro_expert_checkpoint(
                checkpoint_output,
                inference_model,
                prepared.tokenizer,
                metadata={
                    "decision": decision,
                    "parent_checkpoint": str(parent_path),
                    "parent_checkpoint_sha256": parent_sha256,
                    "parent_processed_tokens": parent_processed_tokens,
                    "additional_processed_tokens": additional_processed_tokens,
                    "processed_tokens": cumulative_processed_tokens,
                    "schedule_sha256": prepared.schedule_sha256,
                    "matched_control_report": str(control_path),
                    "matched_control_report_sha256": sha256_file(control_path),
                    "control_heldout_loss": float(control_record["heldout_loss"]),
                    "heldout_loss": float(row["heldout"]["heldout_loss"]),
                    "heldout_loss_gain_over_control": control_gain,
                    "future_prediction_configuration": asdict(future_config),
                    "training_head_attachment": attachment_record,
                    "training_head_removal": strip_record,
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
                "future_heads_persisted": False,
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
            "context_length": parent_context_length,
        },
        "matched_control": {
            "path": str(control_path),
            "sha256": sha256_file(control_path),
            **control_record,
        },
        "context_expansion": context_expansion,
        "training_head_attachment": attachment_record,
        "training_head_removal": strip_record,
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
            "future_objective": objective_before,
        },
        "after": {
            "cumulative_processed_tokens": cumulative_processed_tokens,
            "arm": row,
            "future_objective": objective_after,
        },
        "heldout_loss_gain_over_control": control_gain,
        "decision": decision,
        "checkpoint": checkpoint_record,
        "experiment_wall_seconds": time.perf_counter() - started,
        "promotion_boundary": {
            "base_quality_promoted": False,
            "runtime_install_allowed": False,
            "continual_memory_allowed": False,
            "requires_unseen_generation": decision == SAVE_DECISION,
        },
    }
    write_json_report_with_readme(
        output,
        report,
        title="MARULHO V13 Multi-Horizon Future Prediction",
    )
    print(
        f"[future-prediction-v13] decision {decision}; control loss "
        f"{control_record['heldout_loss']:.4f}, candidate "
        f"{row['heldout']['heldout_loss']:.4f}, gain {control_gain:.4f}",
        flush=True,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--matched-control-report", type=Path, required=True)
    parser.add_argument("--relation-corpus", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--general-train", action="append", type=Path, required=True)
    parser.add_argument("--general-eval", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path)
    parser.add_argument("--additional-token-budget", type=int, default=67_108_864)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--train-sample-mib", type=int, default=64)
    parser.add_argument("--eval-sample-mib", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--learning-rate", type=float, default=7.5e-5)
    parser.add_argument("--future-horizon", action="append", type=int)
    parser.add_argument("--auxiliary-weight", type=float, default=0.25)
    parser.add_argument(
        "--execution-backend",
        choices=("eager", "inductor"),
        default="inductor",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = FuturePredictionExperimentConfig(
        additional_token_budget=int(args.additional_token_budget),
        sequence_length=int(args.sequence_length),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        sample_bytes_per_train_source=int(args.train_sample_mib) * 1024 * 1024,
        sample_bytes_per_eval_source=int(args.eval_sample_mib) * 1024 * 1024,
        future_horizons=(
            (2, 4, 8)
            if args.future_horizon is None
            else tuple(int(value) for value in args.future_horizon)
        ),
        auxiliary_weight=float(args.auxiliary_weight),
        execution_backend=str(args.execution_backend),
    )
    run_future_prediction_experiment(
        parent_checkpoint_path=args.parent_checkpoint,
        matched_control_report_path=args.matched_control_report,
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
