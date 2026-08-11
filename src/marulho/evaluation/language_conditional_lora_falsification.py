"""V50 frozen-base hierarchical conditional low-rank falsification."""

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
    prepare_matched_language_data,
    run_matched_training_arm,
    sha256_file,
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
from marulho.training.language_answer_objective import (
    answer_weighted_next_token_loss,
)
from marulho.training.language_conditional_lora import (
    MarulhoConditionalLoRALanguageModel,
    load_conditional_lora_checkpoint,
    parent_state_sha256,
    save_conditional_lora_checkpoint,
)
from marulho.training.language_model import (
    evaluate_language_model,
    language_model_state_sha256,
    load_language_model_checkpoint,
)
from marulho.training.language_muon import (
    MarulhoMuon,
    warm_language_muon_orthogonalizer_shapes,
)


SURFACE = "marulho_conditional_lora_falsification.v1"
ARM_NAME = "frozen_base_hierarchical_lora"


@dataclass(frozen=True)
class ConditionalLoRAFalsificationConfig:
    token_budget: int = 2_096_640
    sequence_length: int = 72
    source_microbatch_size: int = 8
    microbatches_per_optimizer_step: int = 28
    eval_batches: int = 16
    relation_eval_batch_size: int = 8
    relation_case_count: int = 64
    relation_generation_tokens: int = 16
    rank: int = 16
    learning_rate: float = 3.0e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    answer_weight: float = 4.0
    data_seed: int = 50121
    model_seed: int = 50131
    sample_bytes_per_train_source: int = 1 * 1024 * 1024
    sample_bytes_per_eval_source: int = 1 * 1024 * 1024
    sample_range_count: int = 8
    optimizer_warmup_steps: int = 3
    maximum_projected_seconds: float = 1_200.0
    maximum_adapter_parameter_fraction: float = 0.05
    minimum_grounding_accuracy: float = 0.28125
    minimum_source_gain: float = 0.10
    v48_answer_accuracy: float = 0.21875
    minimum_gain_over_v48: float = 0.05


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _training_config(
    config: ConditionalLoRAFalsificationConfig,
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


def v50_gate(
    *,
    row: Mapping[str, Any],
    source: Mapping[str, Any],
    parity: Mapping[str, Any],
    gradients: Mapping[str, Any],
    adapter_fraction: float,
    baseline_relation: Mapping[str, Any],
    config: ConditionalLoRAFalsificationConfig,
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
        "adapter_fraction": float(adapter_fraction)
        <= float(config.maximum_adapter_parameter_fraction),
        "all_adapter_parameters_received_gradient": bool(
            gradients["all_received_gradient"]
        ),
        "all_adapter_parameters_nonzero_gradient": bool(
            gradients["all_nonzero_gradient"]
        ),
        "parent_state_exact": bool(parity["parent_state_exact"]),
        "inactive_logits_bit_exact": bool(parity["sample_logits_bit_exact"]),
        "inactive_general_loss_exact": bool(parity["heldout_loss_exact"]),
        "inactive_relation_generation_exact": float(
            row["relation"]["generation_exact_accuracy"]
        )
        == float(baseline_relation["generation_exact_accuracy"]),
        "inactive_relation_ranking_exact": float(row["relation"]["accuracy"])
        == float(baseline_relation["accuracy"]),
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
            "adapter_fraction": float(adapter_fraction),
        },
    }


def run_conditional_lora_falsification(
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
    config: ConditionalLoRAFalsificationConfig = ConditionalLoRAFalsificationConfig(),
) -> dict[str, Any]:
    if len(general_eval_paths) != 2:
        raise ValueError("V50 requires exactly two general evaluation corpora")
    if not torch.cuda.is_available():
        raise ValueError("V50 requires CUDA")
    device = torch.device("cuda")
    checkpoint = Path(checkpoint_path)
    parent, tokenizer, parent_metadata = load_language_model_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    candidate = MarulhoConditionalLoRALanguageModel.from_parent(
        parent,
        rank=int(config.rank),
    )
    expected_parent_hash = parent_state_sha256(parent)
    del parent
    gc.collect()
    candidate = candidate.to(device)
    candidate.set_conditional_lora_enabled(False)
    if parent_state_sha256(candidate) != expected_parent_hash:
        raise ValueError("V50 CUDA parent transfer differs")
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in candidate.state_dict().items()
    }
    print("[lora-v50] preparing SQuAD-only schedule", flush=True)
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
    prepared = replace(
        prepared,
        cases=_stratified_relation_cases(
            prepared.cases,
            case_count=int(config.relation_case_count),
        ),
    )
    manifest = load_squad_grounding_manifest(
        grounding_validation_manifest_path,
        tokenizer,
    )
    manifest["path"] = str(grounding_validation_manifest_path)
    v48 = json.loads(Path(v48_report_path).read_text(encoding="utf-8"))
    baseline_relation = dict(v48["baseline"]["relation"])
    if [row["case_id"] for row in baseline_relation["rows"]] != [
        case.case_id for case in prepared.cases
    ]:
        raise ValueError("V50 relation panel differs from V48")
    baseline_heldout = evaluate_language_model(candidate, prepared.eval_batches)
    parity_batch = prepared.eval_batches[0].to(device)
    with torch.no_grad():
        initial_logits = candidate(
            parity_batch.input_ids,
            collect_telemetry=False,
        )["logits"].detach().cpu()
    marker_ids = torch.tensor(
        tokenizer.encode("Answer:", add_bos=False, add_eos=False),
        dtype=torch.long,
        device=device,
    )
    lora_parameters = tuple(candidate.conditional_lora_parameters())
    shape_counts = Counter(
        tuple(int(value) for value in parameter.shape)
        for parameter in lora_parameters
    )
    optimizer_warmup = warm_language_muon_orthogonalizer_shapes(
        (
            (int(count), int(shape[0]), int(shape[1]))
            for shape, count in shape_counts.items()
        ),
        device=device,
    )
    training_config = _training_config(config)

    def training_loss(input_ids, target_ids):
        candidate.set_conditional_lora_enabled(True)
        try:
            return answer_weighted_next_token_loss(
                candidate,
                input_ids,
                target_ids,
                marker_ids=marker_ids,
                eos_id=int(tokenizer.eos_id),
                answer_weight=float(config.answer_weight),
            )
        finally:
            candidate.set_conditional_lora_enabled(False)

    def optimizer_builder(_model, config_value):
        optimizer = MarulhoMuon(
            muon_parameters=lora_parameters,
            adamw_parameters=(),
            learning_rate=float(config_value.learning_rate),
            weight_decay=float(config_value.weight_decay),
        )
        return optimizer, {
            "kind": "marulho_muon_conditional_lora_only",
            "fused": False,
            "learning_rate": float(config_value.learning_rate),
            "weight_decay": float(config_value.weight_decay),
            "parameter_count": sum(value.numel() for value in lora_parameters),
        }

    frozen_contract = {
        "surface": SURFACE,
        "configuration": asdict(config),
        "checkpoint_sha256": sha256_file(checkpoint),
        "parent_state_sha256": expected_parent_hash,
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
        print("[lora-v50] training hierarchical deltas", flush=True)
        row = run_matched_training_arm(
            ARM_NAME,
            architecture="v39_frozen_conditional_rank16_all_layer_deltas",
            model=candidate,
            initial_state=initial_state,
            training_loss=training_loss,
            execution={
                "requested_backend": "eager",
                "effective_backend": "pytorch_eager",
                "parent_backward": True,
                "parent_gradients": False,
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
            progress_prefix="lora-v50",
            extra_row={
                "initial_heldout": baseline_heldout,
                "initial_relation": baseline_relation,
                "rank": int(config.rank),
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
    candidate.set_conditional_lora_enabled(False)
    final_heldout = evaluate_language_model(candidate, prepared.eval_batches)
    with torch.no_grad():
        final_logits = candidate(
            parity_batch.input_ids,
            collect_telemetry=False,
        )["logits"].detach().cpu()
    parity = {
        "initial_parent_state_sha256": expected_parent_hash,
        "final_parent_state_sha256": parent_state_sha256(candidate),
        "parent_state_exact": parent_state_sha256(candidate)
        == expected_parent_hash,
        "sample_logits_bit_exact": torch.equal(initial_logits, final_logits),
        "initial_heldout_loss": float(baseline_heldout["heldout_loss"]),
        "final_heldout_loss": float(final_heldout["heldout_loss"]),
        "heldout_loss_exact": float(baseline_heldout["heldout_loss"])
        == float(final_heldout["heldout_loss"]),
    }
    named_lora = tuple(candidate.conditional_lora_named_parameters())
    gradients = {
        "parameter_count": sum(value.numel() for _name, value in named_lora),
        "parameters_with_gradient": sum(
            value.numel() for _name, value in named_lora if value.grad is not None
        ),
        "parameters_with_nonzero_gradient": sum(
            value.numel()
            for _name, value in named_lora
            if value.grad is not None and bool(torch.count_nonzero(value.grad).item())
        ),
    }
    gradients["all_received_gradient"] = bool(
        gradients["parameters_with_gradient"] == gradients["parameter_count"]
    )
    gradients["all_nonzero_gradient"] = bool(
        gradients["parameters_with_nonzero_gradient"]
        == gradients["parameter_count"]
    )
    candidate.set_conditional_lora_enabled(True)
    source_output = Path(output_path).with_name(f"{Path(output_path).stem}-source.json")
    source = evaluate_source_grounding(
        candidate,
        tokenizer,
        manifest,
        checkpoint_path=arm_artifact_path,
        output_path=source_output,
        max_new_tokens=8,
    )
    candidate.set_conditional_lora_enabled(False)
    adapter_count = candidate.conditional_lora_parameter_count()
    parent_count = candidate.parent_parameter_count()
    adapter_fraction = float(adapter_count) / float(parent_count)
    gate = v50_gate(
        row=row,
        source=source,
        parity=parity,
        gradients=gradients,
        adapter_fraction=adapter_fraction,
        baseline_relation=baseline_relation,
        config=config,
    )
    decision = (
        "advance_v50_hierarchical_lora_to_learned_routing"
        if gate["passed"]
        else "retire_v50_hierarchical_lora_insufficient_grounding"
    )
    fidelity: dict[str, Any] = {"performed": False}
    candidate_output = Path(candidate_checkpoint_path)
    if gate["passed"]:
        expected_state = language_model_state_sha256(candidate)
        save_conditional_lora_checkpoint(
            candidate_output,
            candidate,
            tokenizer,
            {
                "decision": decision,
                "parent_checkpoint_sha256": sha256_file(checkpoint),
            },
        )
        restored, restored_tokenizer, _metadata = load_conditional_lora_checkpoint(
            candidate_output
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
            decision = "invalid_v50_checkpoint_fidelity"
    report = {
        "surface": SURFACE,
        "decision": decision,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "configuration": asdict(config),
        "parent_checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
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
        "parameterization": {
            "parent_parameters": parent_count,
            "adapter_parameters": adapter_count,
            "adapter_fraction_of_parent": adapter_fraction,
        },
        "optimizer_orthogonalizer_warmup": optimizer_warmup,
        "baseline": {
            "inactive_heldout": baseline_heldout,
            "inactive_relation": baseline_relation,
            "v48_answer_grounding_accuracy": float(config.v48_answer_accuracy),
        },
        "arm": row,
        "inactive_parity": parity,
        "adapter_gradients": gradients,
        "source_grounding": source,
        "gate": gate,
        "checkpoint_fidelity": fidelity,
        "boundary": (
            "V50 tests explicit-condition low-rank plasticity throughout V39. "
            "It does not claim learned routing or runtime installation."
        ),
    }
    write_json_report_with_readme(
        output_path,
        report,
        title="MARULHO V50 Hierarchical Conditional LoRA",
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
    report = run_conditional_lora_falsification(
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
