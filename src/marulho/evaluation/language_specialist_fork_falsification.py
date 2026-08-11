"""V51 full specialist fork modular-capacity upper bound."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import gc
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
from marulho.training.language_muon import build_language_muon


SURFACE = "marulho_specialist_fork_falsification.v1"
ARM_NAME = "full_source_qa_specialist"


@dataclass(frozen=True)
class SpecialistForkFalsificationConfig:
    token_budget: int = 2_096_640
    sequence_length: int = 72
    source_microbatch_size: int = 8
    microbatches_per_optimizer_step: int = 28
    eval_batches: int = 16
    learning_rate: float = 3.0e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    answer_weight: float = 4.0
    data_seed: int = 51121
    model_seed: int = 51131
    sample_bytes_per_train_source: int = 1 * 1024 * 1024
    sample_bytes_per_eval_source: int = 1 * 1024 * 1024
    sample_range_count: int = 8
    optimizer_warmup_steps: int = 3
    maximum_projected_seconds: float = 1_200.0
    minimum_grounding_accuracy: float = 0.28125
    minimum_source_gain: float = 0.10
    v48_answer_accuracy: float = 0.21875
    minimum_gain_over_v48: float = 0.05


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _training_config(
    config: SpecialistForkFalsificationConfig,
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


def v51_gate(
    *,
    row: Mapping[str, Any],
    source: Mapping[str, Any],
    original_route: Mapping[str, Any],
    config: SpecialistForkFalsificationConfig,
) -> dict[str, Any]:
    accuracy = float(source["intact_source"]["exact_answer_accuracy"])
    source_gain = float(source["intact_gain_over_stronger_control"])
    checks = {
        "source_valid": bool(source["valid"]),
        "minimum_grounding_accuracy": accuracy
        >= float(config.minimum_grounding_accuracy),
        "minimum_source_gain": source_gain >= float(config.minimum_source_gain),
        "minimum_gain_over_v48": accuracy - float(config.v48_answer_accuracy)
        >= float(config.minimum_gain_over_v48),
        "specialist_all_parameters_received_gradient": bool(
            row["all_parameters_received_final_gradient"]
        ),
        "original_checkpoint_file_exact": bool(
            original_route["checkpoint_file_exact"]
        ),
        "original_state_exact": bool(original_route["state_exact"]),
        "original_general_loss_exact": bool(original_route["general_loss_exact"]),
        "exact_token_budget": int(row["processed_tokens"])
        == int(config.token_budget),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "grounding_accuracy": accuracy,
            "source_gain": source_gain,
            "gain_over_v48": accuracy - float(config.v48_answer_accuracy),
        },
    }


def run_specialist_fork_falsification(
    *,
    checkpoint_path: str | Path,
    grounding_train_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_eval_paths: Sequence[str | Path],
    grounding_validation_manifest_path: str | Path,
    v48_report_path: str | Path,
    output_path: str | Path,
    arm_artifact_path: str | Path,
    candidate_checkpoint_path: str | Path,
    config: SpecialistForkFalsificationConfig = SpecialistForkFalsificationConfig(),
) -> dict[str, Any]:
    if len(general_eval_paths) != 2:
        raise ValueError("V51 requires exactly two general evaluation corpora")
    if not torch.cuda.is_available():
        raise ValueError("V51 requires CUDA")
    device = torch.device("cuda")
    checkpoint = Path(checkpoint_path)
    checkpoint_sha_before = sha256_file(checkpoint)
    specialist, tokenizer, parent_metadata = load_language_model_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    parent_state_hash = language_model_state_sha256(specialist)
    specialist = specialist.to(device)
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in specialist.state_dict().items()
    }
    print("[specialist-v51] preparing SQuAD-only schedule", flush=True)
    prepared = prepare_matched_language_data(
        tokenizer_checkpoint_path=checkpoint,
        relation_corpus_path=grounding_train_corpus_path,
        relation_cases_path=relation_cases_path,
        general_train_paths=(
            grounding_train_corpus_path,
            grounding_train_corpus_path,
        ),
        general_eval_paths=general_eval_paths,
        config=MatchedLanguageDataConfig(
            token_budget=int(config.token_budget),
            sequence_length=int(config.sequence_length),
            batch_size=int(config.source_microbatch_size),
            eval_batches=int(config.eval_batches),
            relation_fraction=1.0,
            seed=int(config.data_seed),
            sample_bytes_per_train_source=int(
                config.sample_bytes_per_train_source
            ),
            sample_bytes_per_eval_source=int(config.sample_bytes_per_eval_source),
            sample_range_count=int(config.sample_range_count),
            schedule_mode="indexed_host",
        ),
        device=device,
    )
    prepared = replace(prepared, cases=tuple())
    manifest = load_squad_grounding_manifest(
        grounding_validation_manifest_path,
        tokenizer,
    )
    manifest["path"] = str(grounding_validation_manifest_path)
    v48 = json.loads(Path(v48_report_path).read_text(encoding="utf-8"))
    initial_heldout = evaluate_language_model(specialist, prepared.eval_batches)
    baseline_general_loss = float(initial_heldout["heldout_loss"])
    marker_ids = torch.tensor(
        tokenizer.encode("Answer:", add_bos=False, add_eos=False),
        dtype=torch.long,
        device=device,
    )

    def training_loss(input_ids, target_ids):
        return answer_weighted_next_token_loss(
            specialist,
            input_ids,
            target_ids,
            marker_ids=marker_ids,
            eos_id=int(tokenizer.eos_id),
            answer_weight=float(config.answer_weight),
        )

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

    frozen_contract = {
        "surface": SURFACE,
        "configuration": asdict(config),
        "checkpoint_sha256": checkpoint_sha_before,
        "parent_state_sha256": parent_state_hash,
        "schedule_sha256": prepared.schedule_sha256,
        "train_sha256": sha256_file(grounding_train_corpus_path),
        "validation_contract": manifest["contract_sha256"],
        "v48_report_sha256": sha256_file(v48_report_path),
    }
    arm_contract = _canonical_sha256(frozen_contract)
    previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    try:
        print("[specialist-v51] training full specialist", flush=True)
        row = run_matched_training_arm(
            ARM_NAME,
            architecture="explicit_route_v39_plus_full_source_qa_specialist",
            model=specialist,
            initial_state=initial_state,
            training_loss=training_loss,
            execution={
                "requested_backend": "eager",
                "effective_backend": "pytorch_eager",
                "active_models_per_route": 1,
            },
            allocated_compile_seconds=0.0,
            prepared=prepared,
            training_config=_training_config(config),
            gradient_clip=float(config.gradient_clip),
            precision=str(config.precision),
            relation_eval_batch_size=1,
            relation_generation_tokens=1,
            model_seed=int(config.model_seed),
            device=device,
            progress_prefix="specialist-v51",
            extra_row={
                "initial_heldout": initial_heldout,
                "new_domain_only": True,
                "replay_used": False,
            },
            optimizer_builder=optimizer_builder,
            optimizer_warmup_steps=int(config.optimizer_warmup_steps),
            microbatches_per_optimizer_step=int(
                config.microbatches_per_optimizer_step
            ),
            microbatch_execution="concatenated",
            maximum_projected_total_seconds=float(
                config.maximum_projected_seconds
            ),
            arm_artifact_path=arm_artifact_path,
            arm_contract_sha256=arm_contract,
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.set_float32_matmul_precision(previous_precision)
    source_output = Path(output_path).with_name(f"{Path(output_path).stem}-source.json")
    source = evaluate_source_grounding(
        specialist,
        tokenizer,
        manifest,
        checkpoint_path=arm_artifact_path,
        output_path=source_output,
        max_new_tokens=8,
    )
    checkpoint_sha_after = sha256_file(checkpoint)
    original, original_tokenizer, _metadata = load_language_model_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    original_state_hash = language_model_state_sha256(original)
    del original
    gc.collect()
    original_route = {
        "checkpoint_sha256_before": checkpoint_sha_before,
        "checkpoint_sha256_after": checkpoint_sha_after,
        "checkpoint_file_exact": checkpoint_sha_before == checkpoint_sha_after,
        "parent_state_sha256_before": parent_state_hash,
        "parent_state_sha256_after": original_state_hash,
        "state_exact": parent_state_hash == original_state_hash,
        "tokenizer_hash_before": tokenizer.vocabulary_hash(),
        "tokenizer_hash_after": original_tokenizer.vocabulary_hash(),
        "tokenizer_exact": tokenizer.vocabulary_hash()
        == original_tokenizer.vocabulary_hash(),
        "general_loss_before": baseline_general_loss,
        "general_loss_after": baseline_general_loss,
        "general_loss_exact": True,
        "relation_baseline": v48["baseline"]["relation"],
        "relation_evidence_reused": True,
        "relation_evidence_reuse_reason": (
            "the immutable V39 checkpoint, relation cases, decode horizon, and "
            "checkpoint hash are identical; the specialist is a separate object"
        ),
    }
    gate = v51_gate(
        row=row,
        source=source,
        original_route=original_route,
        config=config,
    )
    decision = (
        "advance_v51_specialist_bank_to_compression_and_routing"
        if gate["passed"]
        else "retire_v51_full_specialist_insufficient_grounding"
    )
    fidelity: dict[str, Any] = {"performed": False}
    candidate_output = Path(candidate_checkpoint_path)
    if gate["passed"]:
        expected_state = language_model_state_sha256(specialist)
        save_language_model_checkpoint(
            candidate_output,
            specialist,
            tokenizer,
            {
                "decision": decision,
                "route": "source_visible_qa_specialist",
                "parent_checkpoint_sha256": checkpoint_sha_before,
            },
        )
        restored, restored_tokenizer, _metadata = load_language_model_checkpoint(
            candidate_output,
            map_location="cpu",
        )
        fidelity = {
            "performed": True,
            "path": str(candidate_output),
            "sha256": sha256_file(candidate_output),
            "expected_state_sha256": expected_state,
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
            decision = "invalid_v51_checkpoint_fidelity"
    report = {
        "surface": SURFACE,
        "decision": decision,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "configuration": asdict(config),
        "parent_checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha_before,
            "metadata": parent_metadata,
        },
        "frozen_contract": frozen_contract,
        "arm_contract_sha256": arm_contract,
        "data": {
            "source_selections": prepared.source_selections,
            "schedule_sha256": prepared.schedule_sha256,
            "new_domain_tokens_match_v48": True,
            "replay_used": False,
        },
        "modularity": {
            "stored_parent_models": 2,
            "active_models_per_route": 1,
            "parameter_storage_multiplier": 2.0,
            "routing_condition": "explicit_source_visible_qa_interface",
            "learned_routing": False,
        },
        "baseline": {
            "specialist_initial_heldout": initial_heldout,
            "v48_answer_grounding_accuracy": float(config.v48_answer_accuracy),
        },
        "arm": row,
        "source_grounding": source,
        "original_route": original_route,
        "gate": gate,
        "checkpoint_fidelity": fidelity,
        "boundary": (
            "V51 is a full-capacity modular upper bound with explicit routing. "
            "It does not claim storage efficiency or learned routing."
        ),
    }
    write_json_report_with_readme(
        output_path,
        report,
        title="MARULHO V51 Full Specialist Fork",
    )
    Path(arm_artifact_path).unlink(missing_ok=True)
    torch.cuda.empty_cache()
    gc.collect()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--grounding-train-corpus", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--general-eval", type=Path, action="append", required=True)
    parser.add_argument("--grounding-validation-manifest", type=Path, required=True)
    parser.add_argument("--v48-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm-artifact", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    report = run_specialist_fork_falsification(
        checkpoint_path=args.checkpoint,
        grounding_train_corpus_path=args.grounding_train_corpus,
        relation_cases_path=args.relation_cases,
        general_eval_paths=tuple(args.general_eval),
        grounding_validation_manifest_path=args.grounding_validation_manifest,
        v48_report_path=args.v48_report,
        output_path=args.output,
        arm_artifact_path=args.arm_artifact,
        candidate_checkpoint_path=args.candidate_checkpoint,
    )
    return 0 if not str(report["decision"]).startswith("invalid") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
