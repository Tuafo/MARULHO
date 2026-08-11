"""Run V53 frozen-base source-pointer falsification."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
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
from marulho.evaluation.language_source_grounding import (
    evaluate_source_grounding,
    load_squad_grounding_manifest,
)
from marulho.evaluation.language_source_grounding_continual import (
    _stratified_relation_cases,
)
from marulho.evaluation.language_training_experiment import (
    LanguageTrainingExperimentConfig,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    evaluate_language_model,
    language_model_state_sha256,
    load_language_model_checkpoint,
)
from marulho.training.language_source_pointer import (
    FrozenSourcePointerLanguageModel,
    load_source_pointer_checkpoint,
    save_source_pointer_checkpoint,
    source_pointer_answer_loss,
)


SURFACE = "marulho_source_pointer_falsification.v1"
ARM_NAME = "frozen_source_pointer_rank64"


@dataclass(frozen=True)
class SourcePointerFalsificationConfig:
    token_budget: int = 2_096_640
    sequence_length: int = 72
    batch_size: int = 8
    microbatches_per_optimizer_step: int = 28
    pointer_rank: int = 64
    learning_rate: float = 3.0e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    data_seed: int = 53121
    model_seed: int = 53131
    relation_case_count: int = 64
    relation_eval_batch_size: int = 8
    relation_generation_tokens: int = 16
    eval_batches: int = 16
    optimizer_warmup_steps: int = 3
    maximum_projected_seconds: float = 1_200.0
    maximum_parameter_fraction: float = 0.0025
    minimum_grounding_accuracy: float = 0.28125
    minimum_source_gain: float = 0.20
    v52_grounding_accuracy: float = 0.296875
    maximum_regression_from_v52: float = 0.015625


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def source_pointer_gate(
    row: Mapping[str, Any],
    *,
    parent: Mapping[str, Any],
    pointer_parameters: int,
    parent_parameters: int,
    config: SourcePointerFalsificationConfig,
) -> dict[str, Any]:
    source = dict(row["source_grounding"])
    intact = float(source["intact_source"]["exact_answer_accuracy"])
    source_gain = float(source["intact_gain_over_stronger_control"])
    regression_from_v52 = float(config.v52_grounding_accuracy) - intact
    parameter_fraction = int(pointer_parameters) / int(parent_parameters)
    checks = {
        "source_valid": bool(source["valid"]),
        "minimum_grounding_accuracy": intact
        >= float(config.minimum_grounding_accuracy),
        "minimum_source_gain": source_gain >= float(config.minimum_source_gain),
        "bounded_regression_from_v52": regression_from_v52
        <= float(config.maximum_regression_from_v52),
        "exact_token_budget": int(row["processed_tokens"]) == int(config.token_budget),
        "all_pointer_parameters_received_gradient": bool(
            row["all_parameters_received_final_gradient"]
        ),
        "maximum_parameter_fraction": parameter_fraction
        <= float(config.maximum_parameter_fraction),
        "parent_checkpoint_file_exact": bool(parent["checkpoint_file_exact"]),
        "parent_state_exact": bool(parent["state_exact"]),
        "parent_logits_exact": bool(parent["logits_exact"]),
        "parent_general_loss_exact": bool(parent["general_loss_exact"]),
        "parent_relation_exact": bool(parent["relation_exact"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "grounding_accuracy": intact,
            "source_gain": source_gain,
            "regression_from_v52": regression_from_v52,
            "pointer_parameters": int(pointer_parameters),
            "parent_parameters": int(parent_parameters),
            "parameter_fraction": parameter_fraction,
        },
        "thresholds": asdict(config),
    }


def run_source_pointer_falsification(
    *,
    checkpoint_path: str | Path,
    grounding_train_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    grounding_validation_manifest_path: str | Path,
    output_path: str | Path,
    arm_artifact_path: str | Path,
    candidate_checkpoint_path: str | Path,
    config: SourcePointerFalsificationConfig = SourcePointerFalsificationConfig(),
) -> dict[str, Any]:
    if len(general_train_paths) < 2 or len(general_eval_paths) != 2:
        raise ValueError("V53 requires at least two train and exactly two eval sources")
    if not torch.cuda.is_available():
        raise ValueError("V53 is a CUDA-only evidence run")
    device = torch.device("cuda")
    checkpoint = Path(checkpoint_path)
    checkpoint_sha_before = sha256_file(checkpoint)
    base, tokenizer, checkpoint_metadata = load_language_model_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    if int(base.context_length) != int(config.sequence_length):
        raise ValueError("V53 must preserve the V39 context length")
    parent_parameters = sum(parameter.numel() for parameter in base.parameters())
    parent_state_before = language_model_state_sha256(base)
    base = base.to(device)
    model = FrozenSourcePointerLanguageModel(
        base,
        context_marker_ids=torch.tensor(
            tokenizer.encode("Context:", add_bos=False, add_eos=False),
            dtype=torch.long,
        ),
        question_marker_ids=torch.tensor(
            tokenizer.encode("\nQuestion:", add_bos=False, add_eos=False),
            dtype=torch.long,
        ),
        pointer_rank=int(config.pointer_rank),
    ).to(device)
    pointer_parameters = model.pointer_parameter_count()
    if pointer_parameters / parent_parameters > float(config.maximum_parameter_fraction):
        raise ValueError("V53 pointer exceeds the preregistered parameter fraction")
    print("[source-pointer-v53] preparing aligned SQuAD schedule", flush=True)
    prepared = prepare_matched_language_data(
        tokenizer_checkpoint_path=checkpoint,
        relation_corpus_path=grounding_train_corpus_path,
        relation_cases_path=relation_cases_path,
        general_train_paths=general_train_paths,
        general_eval_paths=general_eval_paths,
        config=MatchedLanguageDataConfig(
            token_budget=int(config.token_budget),
            sequence_length=int(config.sequence_length),
            batch_size=int(config.batch_size),
            eval_batches=int(config.eval_batches),
            relation_fraction=1.0,
            seed=int(config.data_seed),
            sample_bytes_per_train_source=1 * 1024 * 1024,
            sample_bytes_per_eval_source=1 * 1024 * 1024,
            sample_range_count=8,
            schedule_mode="indexed_host",
            relation_window_mode="document_aligned",
        ),
        device=device,
    )
    prepared = replace(
        prepared,
        cases=_stratified_relation_cases(
            prepared.cases,
            case_count=int(config.relation_case_count),
        ),
    )
    if int(prepared.staged.step_count) % int(config.microbatches_per_optimizer_step):
        raise ValueError("V53 schedule does not divide into exact optimizer steps")
    validation = load_squad_grounding_manifest(
        grounding_validation_manifest_path,
        tokenizer,
    )
    validation["path"] = str(grounding_validation_manifest_path)
    print("[source-pointer-v53] auditing immutable parent", flush=True)
    baseline_general = evaluate_language_model(base, prepared.eval_batches)
    baseline_relation = evaluate_relation_binding_cases_batched(
        base,
        tokenizer,
        prepared.cases,
        batch_size=int(config.relation_eval_batch_size),
        max_new_tokens=int(config.relation_generation_tokens),
    )
    sample = prepared.eval_batches[0].to(device)
    with torch.no_grad():
        baseline_logits = base(sample.input_ids, collect_telemetry=False)[
            "logits"
        ].detach().cpu()
    answer_marker_ids = torch.tensor(
        tokenizer.encode("Answer:", add_bos=False, add_eos=False),
        dtype=torch.long,
        device=device,
    )
    initial_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    training_config = LanguageTrainingExperimentConfig(
        sequence_length=int(config.sequence_length),
        batch_size=int(config.batch_size),
        learning_rate=float(config.learning_rate),
        minimum_learning_rate_fraction=float(config.minimum_learning_rate_fraction),
        warmup_fraction=float(config.warmup_fraction),
        weight_decay=float(config.weight_decay),
        max_grad_norm=float(config.gradient_clip),
        precision=str(config.precision),
        execution_backend="eager",
        device="cuda",
    )

    def training_loss(input_ids, target_ids):
        return source_pointer_answer_loss(
            model,
            input_ids,
            target_ids,
            answer_marker_ids=answer_marker_ids,
            eos_id=int(tokenizer.eos_id),
            pad_id=int(tokenizer.pad_id),
        )

    frozen_contract = {
        "surface": SURFACE,
        "configuration": asdict(config),
        "checkpoint_sha256": checkpoint_sha_before,
        "parent_state_sha256": parent_state_before,
        "schedule_sha256": prepared.schedule_sha256,
        "train_sha256": sha256_file(grounding_train_corpus_path),
        "validation_contract": validation["contract_sha256"],
    }
    arm_contract = _canonical_sha256(frozen_contract)
    print("[source-pointer-v53] training pointer", flush=True)
    row = run_matched_training_arm(
        ARM_NAME,
        architecture="frozen_v39_plus_rank64_source_pointer",
        model=model,
        initial_state=initial_state,
        training_loss=training_loss,
        execution={
            "requested_backend": "eager",
            "effective_backend": "pytorch_eager",
            "parent_frozen": True,
            "active_models_per_route": 1,
        },
        allocated_compile_seconds=0.0,
        prepared=prepared,
        training_config=training_config,
        gradient_clip=float(config.gradient_clip),
        precision=str(config.precision),
        relation_eval_batch_size=int(config.relation_eval_batch_size),
        relation_generation_tokens=int(config.relation_generation_tokens),
        model_seed=int(config.model_seed),
        device=device,
        progress_prefix="source-pointer-v53",
        extra_row={
            "parent_frozen": True,
            "pointer_parameters": pointer_parameters,
            "parent_parameters": parent_parameters,
            "replay_used": False,
        },
        optimizer_warmup_steps=int(config.optimizer_warmup_steps),
        microbatches_per_optimizer_step=int(config.microbatches_per_optimizer_step),
        microbatch_execution="gradient_accumulation",
        maximum_projected_total_seconds=float(config.maximum_projected_seconds),
        arm_artifact_path=arm_artifact_path,
        arm_contract_sha256=arm_contract,
    )
    source_path = Path(output_path).with_name(f"{Path(output_path).stem}-source.json")
    row["source_grounding"] = evaluate_source_grounding(
        model,
        tokenizer,
        validation,
        checkpoint_path=arm_artifact_path,
        output_path=source_path,
        max_new_tokens=8,
    )
    checkpoint_sha_after = sha256_file(checkpoint)
    parent_state_after = language_model_state_sha256(model.base)
    with torch.no_grad():
        final_logits = model.base(sample.input_ids, collect_telemetry=False)[
            "logits"
        ].detach().cpu()
    parent = {
        "checkpoint_sha256_before": checkpoint_sha_before,
        "checkpoint_sha256_after": checkpoint_sha_after,
        "checkpoint_file_exact": checkpoint_sha_before == checkpoint_sha_after,
        "state_sha256_before": parent_state_before,
        "state_sha256_after": parent_state_after,
        "state_exact": parent_state_before == parent_state_after,
        "logits_exact": torch.equal(baseline_logits, final_logits),
        "general_loss_before": float(baseline_general["heldout_loss"]),
        "general_loss_after": float(row["heldout"]["heldout_loss"]),
        "general_loss_exact": float(baseline_general["heldout_loss"])
        == float(row["heldout"]["heldout_loss"]),
        "relation_before": baseline_relation,
        "relation_after": row["relation"],
        "relation_exact": baseline_relation == row["relation"],
    }
    gate = source_pointer_gate(
        row,
        parent=parent,
        pointer_parameters=pointer_parameters,
        parent_parameters=parent_parameters,
        config=config,
    )
    candidate = Path(candidate_checkpoint_path)
    candidate.unlink(missing_ok=True)
    fidelity: dict[str, Any] = {"performed": False}
    decision = (
        "advance_v53_source_pointer_to_routing"
        if bool(gate["passed"])
        else "retire_v53_frozen_source_pointer_insufficient_grounding"
    )
    if bool(gate["passed"]):
        expected_state = language_model_state_sha256(model)
        save_source_pointer_checkpoint(
            candidate,
            model,
            parent_checkpoint_sha256=checkpoint_sha_before,
            metadata={
                "source_experiment": SURFACE,
                "decision": decision,
                "processed_tokens": int(config.token_budget),
            },
        )
        restored_base, restored_tokenizer, _metadata = load_language_model_checkpoint(
            checkpoint,
            map_location="cpu",
        )
        restored, restored_metadata = load_source_pointer_checkpoint(
            candidate,
            restored_base,
            expected_parent_checkpoint_sha256=checkpoint_sha_before,
        )
        fidelity = {
            "performed": True,
            "checkpoint_path": str(candidate),
            "checkpoint_sha256": sha256_file(candidate),
            "expected_state_sha256": expected_state,
            "restored_state_sha256": language_model_state_sha256(restored),
            "tokenizer_hash_before": tokenizer.vocabulary_hash(),
            "tokenizer_hash_after": restored_tokenizer.vocabulary_hash(),
            "metadata": restored_metadata,
        }
        fidelity["passed"] = bool(
            fidelity["expected_state_sha256"] == fidelity["restored_state_sha256"]
            and fidelity["tokenizer_hash_before"]
            == fidelity["tokenizer_hash_after"]
        )
        if not bool(fidelity["passed"]):
            candidate.unlink(missing_ok=True)
            decision = "invalid_v53_checkpoint_fidelity"
    report = {
        "surface": SURFACE,
        "decision": decision,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "external_text_data": True,
        "configuration": asdict(config),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha_before,
            "metadata": checkpoint_metadata,
        },
        "data": {
            "schedule_sha256": prepared.schedule_sha256,
            "source_selections": prepared.source_selections,
            "validation_manifest_contract_sha256": validation["contract_sha256"],
        },
        "baseline": {
            "general": baseline_general,
            "relation": baseline_relation,
        },
        "arm": row,
        "parent": parent,
        "gate": gate,
        "checkpoint_fidelity": fidelity,
        "boundary": (
            "V53 tests a frozen-final-state source-copy path on aligned extractive "
            "QA. It does not prove broad retrieval, open-domain reasoning, or learned "
            "routing."
        ),
    }
    write_json_report_with_readme(
        output_path,
        report,
        title="MARULHO V53 Frozen Source Pointer",
    )
    Path(arm_artifact_path).unlink(missing_ok=True)
    if not bool(gate["passed"]):
        candidate.unlink(missing_ok=True)
    torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--grounding-train-corpus", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--general-train", type=Path, action="append", required=True)
    parser.add_argument("--general-eval", type=Path, action="append", required=True)
    parser.add_argument("--grounding-validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm-artifact", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    report = run_source_pointer_falsification(
        checkpoint_path=args.checkpoint,
        grounding_train_corpus_path=args.grounding_train_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=tuple(args.general_train),
        general_eval_paths=tuple(args.general_eval),
        grounding_validation_manifest_path=args.grounding_validation_manifest,
        output_path=args.output,
        arm_artifact_path=args.arm_artifact,
        candidate_checkpoint_path=args.candidate_checkpoint,
    )
    print(report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
