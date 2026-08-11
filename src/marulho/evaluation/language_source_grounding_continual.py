"""V48 matched continual source-grounding objective screen."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from marulho.evaluation.language_matched_support import (
    MatchedLanguageDataConfig,
    load_matched_arm_artifact,
    prepare_matched_language_data,
    run_matched_training_arm,
    sha256_file,
)
from marulho.evaluation.language_source_grounding import (
    evaluate_source_grounding,
    load_squad_grounding_manifest,
)
from marulho.evaluation.language_training_experiment import (
    LanguageTrainingExperimentConfig,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_answer_objective import (
    answer_weighted_next_token_loss,
)
from marulho.training.language_model import (
    evaluate_language_model,
    language_model_state_sha256,
    load_language_model_checkpoint,
    save_language_model_checkpoint,
)
from marulho.training.language_muon import (
    build_language_muon,
    warm_language_muon_orthogonalizer_shapes,
)


SURFACE = "marulho_source_grounding_continual.v1"
ARTIFACT_KIND = "marulho_source_grounding_continual"
ARM_NAMES = ("ordinary_causal", "answer_weight4")


@dataclass(frozen=True)
class SourceGroundingContinualConfig:
    token_budget: int = 4_193_280
    sequence_length: int = 72
    source_microbatch_size: int = 8
    microbatches_per_optimizer_step: int = 28
    eval_batches: int = 16
    relation_eval_batch_size: int = 8
    relation_case_count: int = 64
    relation_generation_tokens: int = 16
    grounding_fraction: float = 0.50
    learning_rate: float = 3.0e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    answer_weight: float = 4.0
    data_seed: int = 48121
    model_seed: int = 48131
    sample_bytes_per_train_source: int = 8 * 1024 * 1024
    sample_bytes_per_eval_source: int = 1 * 1024 * 1024
    sample_range_count: int = 16
    schedule_mode: str = "indexed_host"
    optimizer_warmup_steps: int = 3
    maximum_projected_seconds_per_arm: float = 1_200.0
    minimum_grounding_accuracy: float = 0.25
    minimum_source_gain: float = 0.10
    minimum_answer_arm_gain: float = 0.05
    maximum_relation_regression: float = 0.05
    maximum_general_loss_regression: float = 0.05


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _training_config(
    config: SourceGroundingContinualConfig,
) -> LanguageTrainingExperimentConfig:
    return LanguageTrainingExperimentConfig(
        sequence_length=int(config.sequence_length),
        batch_size=int(config.source_microbatch_size),
        learning_rate=float(config.learning_rate),
        minimum_learning_rate_fraction=float(config.minimum_learning_rate_fraction),
        warmup_fraction=float(config.warmup_fraction),
        weight_decay=float(config.weight_decay),
        max_grad_norm=float(config.gradient_clip),
        precision=str(config.precision),
        execution_backend="eager",
        device="cuda",
    )


def _stratified_relation_cases(cases, *, case_count: int):
    count = int(case_count)
    if count < 1 or count > len(cases):
        raise ValueError("relation case count is outside the available panel")
    by_kind: dict[str, list[Any]] = {}
    for case in cases:
        by_kind.setdefault(str(case.kind), []).append(case)
    selected = []
    positions = {kind: 0 for kind in by_kind}
    while len(selected) < count:
        progressed = False
        for kind in sorted(by_kind):
            position = positions[kind]
            if position >= len(by_kind[kind]):
                continue
            selected.append(by_kind[kind][position])
            positions[kind] = position + 1
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            raise ValueError("could not fill stratified relation panel")
    return tuple(selected)


def _core_gate(
    row: Mapping[str, Any],
    *,
    baseline_general_loss: float,
    baseline_relation_accuracy: float,
    config: SourceGroundingContinualConfig,
) -> dict[str, Any]:
    grounding = dict(row["source_grounding"])
    intact = float(grounding["intact_source"]["exact_answer_accuracy"])
    source_gain = float(grounding["intact_gain_over_stronger_control"])
    relation_accuracy = float(row["relation"]["generation_exact_accuracy"])
    general_loss = float(row["heldout"]["heldout_loss"])
    checks = {
        "valid_source_grounding": bool(grounding["valid"]),
        "all_parameters_received_final_gradient": bool(
            row["all_parameters_received_final_gradient"]
        ),
        "exact_token_budget": int(row["processed_tokens"])
        == int(config.token_budget),
        "grounding_accuracy": intact >= float(config.minimum_grounding_accuracy),
        "source_gain": source_gain >= float(config.minimum_source_gain),
        "relation_retention": relation_accuracy
        >= float(baseline_relation_accuracy)
        - float(config.maximum_relation_regression),
        "general_loss_retention": general_loss
        <= float(baseline_general_loss)
        + float(config.maximum_general_loss_regression),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "grounding_accuracy": intact,
            "source_gain": source_gain,
            "relation_generation_exact_accuracy": relation_accuracy,
            "general_heldout_loss": general_loss,
        },
    }


def select_v48_candidate(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    baseline_general_loss: float,
    baseline_relation_accuracy: float,
    config: SourceGroundingContinualConfig,
) -> tuple[str | None, str, dict[str, Any]]:
    if set(arms) != set(ARM_NAMES):
        return None, "invalid_v48_missing_matched_arm", {}
    gates = {
        name: _core_gate(
            row,
            baseline_general_loss=float(baseline_general_loss),
            baseline_relation_accuracy=float(baseline_relation_accuracy),
            config=config,
        )
        for name, row in arms.items()
    }
    ordinary_accuracy = float(
        arms["ordinary_causal"]["source_grounding"]["intact_source"][
            "exact_answer_accuracy"
        ]
    )
    answer_accuracy = float(
        arms["answer_weight4"]["source_grounding"]["intact_source"][
            "exact_answer_accuracy"
        ]
    )
    answer_superiority = answer_accuracy - ordinary_accuracy >= float(
        config.minimum_answer_arm_gain
    )
    gates["answer_weight4"]["checks"]["gain_over_ordinary_arm"] = (
        answer_superiority
    )
    gates["answer_weight4"]["observed"]["gain_over_ordinary_arm"] = (
        answer_accuracy - ordinary_accuracy
    )
    gates["answer_weight4"]["passed"] = bool(
        gates["answer_weight4"]["passed"] and answer_superiority
    )
    if bool(gates["answer_weight4"]["passed"]):
        return (
            "answer_weight4",
            "scale_v48_answer_objective_to_confirmation",
            gates,
        )
    if bool(gates["ordinary_causal"]["passed"]):
        return (
            "ordinary_causal",
            "scale_v48_ordinary_objective_answer_weighting_unnecessary",
            gates,
        )
    return None, "retire_v48_objective_only_grounding_repair", gates


def run_source_grounding_continual(
    *,
    checkpoint_path: str | Path,
    grounding_train_corpus_path: str | Path,
    relation_replay_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    grounding_validation_manifest_path: str | Path,
    output_path: str | Path,
    candidate_checkpoint_path: str | Path,
    arm_artifact_directory: str | Path,
    config: SourceGroundingContinualConfig = SourceGroundingContinualConfig(),
    requested_arms: Sequence[str] = ARM_NAMES,
) -> dict[str, Any]:
    requested = tuple(dict.fromkeys(str(value) for value in requested_arms))
    if not requested or any(value not in ARM_NAMES for value in requested):
        raise ValueError("requested V48 arms are invalid")
    if len(general_train_paths) != 2 or len(general_eval_paths) != 2:
        raise ValueError("V48 requires two general train and two eval corpora")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise ValueError("V48 is a CUDA-only evidence run")
    checkpoint = Path(checkpoint_path)
    model, tokenizer, checkpoint_metadata = load_language_model_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    if int(model.context_length) != int(config.sequence_length):
        raise ValueError("V48 must preserve the checkpoint context length")
    model = model.to(device)
    initial_state_hash = language_model_state_sha256(model)
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    print("[grounding-v48] preparing frozen matched data", flush=True)
    prepared = prepare_matched_language_data(
        tokenizer_checkpoint_path=checkpoint,
        relation_corpus_path=grounding_train_corpus_path,
        relation_cases_path=relation_cases_path,
        general_train_paths=(relation_replay_corpus_path, *general_train_paths),
        general_eval_paths=general_eval_paths,
        config=MatchedLanguageDataConfig(
            token_budget=int(config.token_budget),
            sequence_length=int(config.sequence_length),
            batch_size=int(config.source_microbatch_size),
            eval_batches=int(config.eval_batches),
            relation_fraction=float(config.grounding_fraction),
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
    prepared = replace(
        prepared,
        cases=_stratified_relation_cases(
            prepared.cases,
            case_count=int(config.relation_case_count),
        ),
    )
    if int(prepared.staged.step_count) % int(
        config.microbatches_per_optimizer_step
    ):
        raise ValueError("V48 schedule does not divide into exact optimizer steps")
    validation_manifest = load_squad_grounding_manifest(
        grounding_validation_manifest_path,
        tokenizer,
    )
    validation_manifest["path"] = str(grounding_validation_manifest_path)
    print("[grounding-v48] evaluating initial general holdout", flush=True)
    baseline_heldout = evaluate_language_model(model, prepared.eval_batches)
    from marulho.evaluation.language_relation_binding_experiment import (
        evaluate_relation_binding_cases_batched,
    )

    print(
        "[grounding-v48] evaluating initial stratified relation panel",
        flush=True,
    )
    baseline_relation = evaluate_relation_binding_cases_batched(
        model,
        tokenizer,
        prepared.cases,
        batch_size=int(config.relation_eval_batch_size),
        max_new_tokens=int(config.relation_generation_tokens),
    )
    shape_counts = Counter(
        tuple(int(value) for value in parameter.shape)
        for name, parameter in model.named_parameters()
        if parameter.ndim == 2
        and not name.startswith("token_embedding.")
        and not name.startswith("lm_head.")
    )
    print("[grounding-v48] warming Muon orthogonalizer", flush=True)
    optimizer_warmup = warm_language_muon_orthogonalizer_shapes(
        (
            (int(count), int(shape[0]), int(shape[1]))
            for shape, count in shape_counts.items()
        ),
        device=device,
    )
    marker_ids = torch.tensor(
        tokenizer.encode("Answer:", add_bos=False, add_eos=False),
        dtype=torch.long,
        device=device,
    )
    training_config = _training_config(config)
    execution = {
        "requested_backend": "eager",
        "effective_backend": "pytorch_eager",
        "compile_seconds": 0.0,
        "matched_execution": True,
    }

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

    arm_directory = Path(arm_artifact_directory)
    arm_directory.mkdir(parents=True, exist_ok=True)
    arms: dict[str, dict[str, Any]] = {}
    frozen_common = {
        "surface": SURFACE,
        "config": asdict(config),
        "checkpoint_sha256": sha256_file(checkpoint),
        "initial_state_sha256": initial_state_hash,
        "schedule_sha256": prepared.schedule_sha256,
        "validation_manifest_contract_sha256": validation_manifest[
            "contract_sha256"
        ],
        "grounding_train_corpus_sha256": sha256_file(
            grounding_train_corpus_path
        ),
        "relation_replay_corpus_sha256": sha256_file(relation_replay_corpus_path),
        "general_train_sha256": [sha256_file(path) for path in general_train_paths],
        "general_eval_sha256": [sha256_file(path) for path in general_eval_paths],
        "relation_cases_sha256": sha256_file(relation_cases_path),
    }
    previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    try:
        for arm in requested:
            if arm == "ordinary_causal":

                def training_loss(input_ids, target_ids):
                    return model.next_token_loss(
                        input_ids,
                        target_ids,
                        collect_telemetry=False,
                        return_evidence=False,
                    )["loss"]

            else:

                def training_loss(input_ids, target_ids):
                    return answer_weighted_next_token_loss(
                        model,
                        input_ids,
                        target_ids,
                        marker_ids=marker_ids,
                        eos_id=int(tokenizer.eos_id),
                        answer_weight=float(config.answer_weight),
                    )

            arm_contract = _canonical_sha256(
                {**frozen_common, "arm": arm, "objective": arm}
            )
            artifact_path = arm_directory / f"v48-{arm}.pt"
            print(f"[grounding-v48] training {arm}", flush=True)
            row = run_matched_training_arm(
                arm,
                architecture="marulho_v39_transformer_continual_grounding",
                model=model,
                initial_state=initial_state,
                training_loss=training_loss,
                execution=execution,
                allocated_compile_seconds=0.0,
                prepared=prepared,
                training_config=training_config,
                gradient_clip=float(config.gradient_clip),
                precision=str(config.precision),
                relation_eval_batch_size=int(config.relation_eval_batch_size),
                relation_generation_tokens=int(config.relation_generation_tokens),
                model_seed=int(config.model_seed),
                device=device,
                progress_prefix="grounding-v48",
                extra_row={
                    "objective": arm,
                    "answer_weight": (
                        1.0 if arm == "ordinary_causal" else config.answer_weight
                    ),
                    "initial_heldout": baseline_heldout,
                    "initial_relation": baseline_relation,
                },
                optimizer_builder=optimizer_builder,
                optimizer_warmup_steps=int(config.optimizer_warmup_steps),
                microbatches_per_optimizer_step=int(
                    config.microbatches_per_optimizer_step
                ),
                maximum_projected_total_seconds=float(
                    config.maximum_projected_seconds_per_arm
                ),
                arm_artifact_path=artifact_path,
                arm_contract_sha256=arm_contract,
            )
            source_report_path = Path(output_path).with_name(
                f"{Path(output_path).stem}-{arm}-source.json"
            )
            row["source_grounding"] = evaluate_source_grounding(
                model,
                tokenizer,
                validation_manifest,
                checkpoint_path=artifact_path,
                output_path=source_report_path,
                max_new_tokens=8,
            )
            row["final_state_sha256"] = language_model_state_sha256(model)
            row["arm_contract_sha256"] = arm_contract
            row["arm_artifact_path"] = str(artifact_path)
            arms[arm] = row
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.set_float32_matmul_precision(previous_precision)

    candidate, decision, gates = select_v48_candidate(
        arms,
        baseline_general_loss=float(baseline_heldout["heldout_loss"]),
        baseline_relation_accuracy=float(
            baseline_relation["generation_exact_accuracy"]
        ),
        config=config,
    ) if set(arms) == set(ARM_NAMES) else (
        None,
        "incomplete_v48_wait_for_remaining_arm",
        {},
    )
    fidelity: dict[str, Any] = {"performed": False}
    candidate_output = Path(candidate_checkpoint_path)
    if candidate is not None:
        selected_artifact = arm_directory / f"v48-{candidate}.pt"
        _selected_row, selected_state = load_matched_arm_artifact(
            selected_artifact,
            expected_arm_name=candidate,
            expected_contract_sha256=str(arms[candidate]["arm_contract_sha256"]),
        )
        if selected_state is None:
            raise ValueError("selected V48 artifact lacks exact model state")
        model.load_state_dict(selected_state, strict=True)
        expected_hash = language_model_state_sha256(model)
        save_language_model_checkpoint(
            candidate_output,
            model,
            tokenizer,
            {
                "source_experiment": SURFACE,
                "decision": decision,
                "selected_arm": candidate,
                "parent_checkpoint": str(checkpoint),
                "parent_checkpoint_sha256": sha256_file(checkpoint),
                "processed_tokens": int(config.token_budget),
            },
        )
        restored, restored_tokenizer, _metadata = load_language_model_checkpoint(
            candidate_output,
            map_location="cpu",
        )
        fidelity = {
            "performed": True,
            "checkpoint_path": str(candidate_output),
            "checkpoint_sha256": sha256_file(candidate_output),
            "expected_state_sha256": expected_hash,
            "restored_state_sha256": language_model_state_sha256(restored),
            "tokenizer_hash_before": tokenizer.vocabulary_hash(),
            "tokenizer_hash_after": restored_tokenizer.vocabulary_hash(),
        }
        fidelity["passed"] = bool(
            fidelity["expected_state_sha256"] == fidelity["restored_state_sha256"]
            and fidelity["tokenizer_hash_before"] == fidelity["tokenizer_hash_after"]
        )
        if not fidelity["passed"]:
            candidate_output.unlink(missing_ok=True)
            candidate = None
            decision = "invalid_v48_checkpoint_fidelity"
    report = {
        "surface": SURFACE,
        "artifact_kind": ARTIFACT_KIND,
        "decision": decision,
        "selected_arm": candidate,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "external_text_data": True,
        "configuration": asdict(config),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "metadata": checkpoint_metadata,
            "initial_state_sha256": initial_state_hash,
        },
        "data": {
            "schedule_sha256": prepared.schedule_sha256,
            "source_selections": prepared.source_selections,
            "grounding_train_corpus_path": str(grounding_train_corpus_path),
            "grounding_train_corpus_sha256": sha256_file(
                grounding_train_corpus_path
            ),
            "validation_manifest_path": str(grounding_validation_manifest_path),
            "validation_manifest_contract_sha256": validation_manifest[
                "contract_sha256"
            ],
            "training_fraction": {
                "squad_grounding": float(config.grounding_fraction),
                "relation_replay": (1.0 - float(config.grounding_fraction)) / 3.0,
                "fineweb_replay": (1.0 - float(config.grounding_fraction)) / 3.0,
                "cosmopedia_replay": (1.0 - float(config.grounding_fraction)) / 3.0,
            },
        },
        "optimizer_orthogonalizer_warmup": optimizer_warmup,
        "baseline": {
            "heldout": baseline_heldout,
            "relation": baseline_relation,
        },
        "arms": arms,
        "gates": gates,
        "checkpoint_fidelity": fidelity,
        "boundary": (
            "V48 tests whether objective weighting is enough to teach visible-source "
            "extractive QA while retaining V39 relation and general-language behavior. "
            "It does not test a new architecture or prove open-domain reasoning."
        ),
    }
    write_json_report_with_readme(
        output_path,
        report,
        title="MARULHO V48 Continual Source Grounding",
    )
    if set(arms) == set(ARM_NAMES):
        for artifact in arm_directory.glob("v48-*.pt"):
            artifact.unlink()
        try:
            arm_directory.rmdir()
        except OSError:
            pass
    torch.cuda.empty_cache()
    gc.collect()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--grounding-train-corpus", type=Path, required=True)
    parser.add_argument("--relation-replay-corpus", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--general-train", type=Path, action="append", required=True)
    parser.add_argument("--general-eval", type=Path, action="append", required=True)
    parser.add_argument("--grounding-validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--arm-artifact-directory", type=Path, required=True)
    parser.add_argument("--arm", action="append", choices=ARM_NAMES, default=[])
    args = parser.parse_args()
    report = run_source_grounding_continual(
        checkpoint_path=args.checkpoint,
        grounding_train_corpus_path=args.grounding_train_corpus,
        relation_replay_corpus_path=args.relation_replay_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=tuple(args.general_train),
        general_eval_paths=tuple(args.general_eval),
        grounding_validation_manifest_path=args.grounding_validation_manifest,
        output_path=args.output,
        candidate_checkpoint_path=args.candidate_checkpoint,
        arm_artifact_directory=args.arm_artifact_directory,
        requested_arms=tuple(args.arm) or ARM_NAMES,
    )
    return 0 if not str(report["decision"]).startswith("invalid") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
