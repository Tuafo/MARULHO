"""V49 frozen-base conditional residual sidecar falsification."""

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
from marulho.training.language_conditional_adapter import (
    ADAPTER_KEY,
    MarulhoConditionalAdapterLanguageModel,
    load_conditional_adapter_checkpoint,
    save_conditional_adapter_checkpoint,
)
from marulho.training.language_model import (
    evaluate_language_model,
    language_model_state_sha256,
    load_language_model_checkpoint,
)
from marulho.training.language_muon import (
    build_language_muon,
    warm_language_muon_orthogonalizer_shapes,
)


SURFACE = "marulho_conditional_adapter_falsification.v1"
ARTIFACT_KIND = "marulho_conditional_adapter_falsification"
ARM_NAME = "frozen_base_conditional_sidecar"


@dataclass(frozen=True)
class ConditionalAdapterFalsificationConfig:
    token_budget: int = 2_096_640
    sequence_length: int = 72
    source_microbatch_size: int = 8
    microbatches_per_optimizer_step: int = 28
    eval_batches: int = 16
    relation_eval_batch_size: int = 8
    relation_case_count: int = 64
    relation_generation_tokens: int = 16
    learning_rate: float = 3.0e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    answer_weight: float = 4.0
    data_seed: int = 49121
    model_seed: int = 49131
    sample_bytes_per_train_source: int = 1 * 1024 * 1024
    sample_bytes_per_eval_source: int = 1 * 1024 * 1024
    sample_range_count: int = 8
    schedule_mode: str = "indexed_host"
    optimizer_warmup_steps: int = 3
    maximum_projected_seconds: float = 1_200.0
    maximum_adapter_parameter_fraction: float = 0.05
    minimum_grounding_accuracy: float = 0.28125
    minimum_source_gain: float = 0.10
    v48_answer_accuracy: float = 0.21875
    minimum_gain_over_v48: float = 0.05


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
    config: ConditionalAdapterFalsificationConfig,
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


def v49_gate(
    *,
    row: Mapping[str, Any],
    source_grounding: Mapping[str, Any],
    inactive_parity: Mapping[str, Any],
    adapter_gradients: Mapping[str, Any],
    adapter_state: Mapping[str, Any],
    adapter_parameter_fraction: float,
    baseline_relation: Mapping[str, Any],
    baseline_heldout_loss: float,
    config: ConditionalAdapterFalsificationConfig,
) -> dict[str, Any]:
    accuracy = float(source_grounding["intact_source"]["exact_answer_accuracy"])
    source_gain = float(source_grounding["intact_gain_over_stronger_control"])
    relation = dict(row["relation"])
    heldout_loss = float(row["heldout"]["heldout_loss"])
    checks = {
        "source_grounding_valid": bool(source_grounding["valid"]),
        "minimum_grounding_accuracy": accuracy
        >= float(config.minimum_grounding_accuracy),
        "minimum_source_gain": source_gain >= float(config.minimum_source_gain),
        "minimum_gain_over_v48": accuracy - float(config.v48_answer_accuracy)
        >= float(config.minimum_gain_over_v48),
        "adapter_parameter_fraction": float(adapter_parameter_fraction)
        <= float(config.maximum_adapter_parameter_fraction),
        "all_adapter_parameters_received_gradient": bool(
            adapter_gradients["all_received_gradient"]
        ),
        "parent_state_exact": bool(inactive_parity["parent_state_exact"]),
        "inactive_logits_bit_exact": bool(
            inactive_parity["sample_logits_bit_exact"]
        ),
        "inactive_general_loss_exact": heldout_loss
        == float(baseline_heldout_loss),
        "inactive_relation_generation_exact": float(
            relation["generation_exact_accuracy"]
        )
        == float(baseline_relation["generation_exact_accuracy"]),
        "inactive_relation_ranking_exact": float(relation["accuracy"])
        == float(baseline_relation["accuracy"]),
        "adapter_state_bounded": bool(adapter_state["bounded"]),
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
            "adapter_parameter_fraction": float(adapter_parameter_fraction),
            "inactive_heldout_loss": heldout_loss,
            "inactive_relation_generation_accuracy": float(
                relation["generation_exact_accuracy"]
            ),
        },
    }


def run_conditional_adapter_falsification(
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
    config: ConditionalAdapterFalsificationConfig = ConditionalAdapterFalsificationConfig(),
) -> dict[str, Any]:
    if len(general_eval_paths) != 2:
        raise ValueError("V49 requires exactly two general evaluation corpora")
    if not torch.cuda.is_available():
        raise ValueError("V49 is a CUDA-only evidence run")
    device = torch.device("cuda")
    checkpoint = Path(checkpoint_path)
    parent, tokenizer, parent_metadata = load_language_model_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    candidate = MarulhoConditionalAdapterLanguageModel.from_parent(parent)
    expected_parent_hash = language_model_state_sha256(parent)
    if candidate.parent_state_sha256() != expected_parent_hash:
        raise ValueError("V49 parent transfer is not exact")
    del parent
    gc.collect()
    candidate = candidate.to(device)
    candidate.set_conditional_adapter_enabled(False)
    initial_parent_hash = candidate.parent_state_sha256()
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in candidate.state_dict().items()
    }
    print("[adapter-v49] preparing frozen SQuAD schedule", flush=True)
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
        raise ValueError("V49 schedule does not divide into exact optimizer steps")
    validation_manifest = load_squad_grounding_manifest(
        grounding_validation_manifest_path,
        tokenizer,
    )
    validation_manifest["path"] = str(grounding_validation_manifest_path)
    v48 = json.loads(Path(v48_report_path).read_text(encoding="utf-8"))
    baseline_relation = dict(v48["baseline"]["relation"])
    expected_case_ids = [str(case.case_id) for case in prepared.cases]
    baseline_case_ids = [str(value["case_id"]) for value in baseline_relation["rows"]]
    if baseline_case_ids != expected_case_ids:
        raise ValueError("V49 relation panel differs from V48 baseline")
    if int(baseline_relation["generation_max_new_tokens"]) != int(
        config.relation_generation_tokens
    ):
        raise ValueError("V49 relation generation horizon differs from V48")
    print("[adapter-v49] evaluating inactive general baseline", flush=True)
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
    shape_counts = Counter(
        tuple(int(value) for value in parameter.shape)
        for parameter in candidate.conditional_adapter.parameters()
        if parameter.ndim == 2
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
        candidate.set_conditional_adapter_enabled(True)
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
            candidate.set_conditional_adapter_enabled(False)

    def optimizer_builder(model_value, config_value):
        return build_language_muon(
            model_value.conditional_adapter,
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
        "parent_checkpoint_sha256": sha256_file(checkpoint),
        "parent_state_sha256": initial_parent_hash,
        "schedule_sha256": prepared.schedule_sha256,
        "grounding_train_sha256": sha256_file(grounding_train_corpus_path),
        "validation_contract_sha256": validation_manifest["contract_sha256"],
        "relation_cases_sha256": sha256_file(relation_cases_path),
        "general_eval_sha256": [sha256_file(path) for path in general_eval_paths],
        "v48_report_sha256": sha256_file(v48_report_path),
    }
    arm_contract = _canonical_sha256(frozen_contract)
    previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    try:
        print("[adapter-v49] training conditional sidecar", flush=True)
        row = run_matched_training_arm(
            ARM_NAME,
            architecture="v39_frozen_plus_one_causal_residual_sidecar",
            model=candidate,
            initial_state=initial_state,
            training_loss=training_loss,
            execution={
                "requested_backend": "eager",
                "effective_backend": "pytorch_eager",
                "compile_seconds": 0.0,
                "parent_backward": False,
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
            progress_prefix="adapter-v49",
            extra_row={
                "initial_heldout": baseline_heldout,
                "initial_relation": baseline_relation,
                "parent_frozen": True,
                "adapter_activation_during_training": True,
                "adapter_activation_during_retention_evaluation": False,
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
    candidate.set_conditional_adapter_enabled(False)
    with torch.no_grad():
        final_inactive_logits = candidate(
            parity_batch.input_ids,
            collect_telemetry=False,
        )["logits"].detach().cpu()
    final_parent_hash = candidate.parent_state_sha256()
    inactive_parity = {
        "initial_parent_state_sha256": initial_parent_hash,
        "final_parent_state_sha256": final_parent_hash,
        "parent_state_exact": final_parent_hash == initial_parent_hash,
        "sample_logits_bit_exact": torch.equal(
            final_inactive_logits,
            initial_logits,
        ),
    }
    adapter_parameters = tuple(candidate.conditional_adapter.named_parameters())
    adapter_gradients = {
        "parameter_count": sum(value.numel() for _name, value in adapter_parameters),
        "parameters_with_gradient": sum(
            value.numel()
            for _name, value in adapter_parameters
            if value.grad is not None
        ),
        "nonzero_gradient_elements": sum(
            int(torch.count_nonzero(value.grad).detach().cpu())
            for _name, value in adapter_parameters
            if value.grad is not None
        ),
    }
    adapter_gradients["all_received_gradient"] = bool(
        adapter_gradients["parameters_with_gradient"]
        == adapter_gradients["parameter_count"]
    )
    candidate.set_conditional_adapter_enabled(True)
    active_probe = candidate(
        parity_batch.input_ids,
        collect_telemetry=True,
    )
    adapter_cache_tokens = int(active_probe["state"][ADAPTER_KEY].shape[2])
    adapter_state = {
        "cache_tokens": adapter_cache_tokens,
        "context_bound": int(config.sequence_length),
        "bounded": adapter_cache_tokens <= int(config.sequence_length),
        "telemetry": active_probe["telemetry"],
    }
    source_output = Path(output_path).with_name(
        f"{Path(output_path).stem}-source.json"
    )
    source_grounding = evaluate_source_grounding(
        candidate,
        tokenizer,
        validation_manifest,
        checkpoint_path=arm_artifact_path,
        output_path=source_output,
        max_new_tokens=8,
    )
    candidate.set_conditional_adapter_enabled(False)
    adapter_count = candidate.conditional_adapter_parameter_count()
    parent_count = candidate.parent_parameter_count()
    adapter_fraction = float(adapter_count) / float(parent_count)
    gate = v49_gate(
        row=row,
        source_grounding=source_grounding,
        inactive_parity=inactive_parity,
        adapter_gradients=adapter_gradients,
        adapter_state=adapter_state,
        adapter_parameter_fraction=adapter_fraction,
        baseline_relation=baseline_relation,
        baseline_heldout_loss=float(baseline_heldout["heldout_loss"]),
        config=config,
    )
    decision = (
        "advance_v49_conditional_sidecar_to_learned_routing"
        if gate["passed"]
        else "retire_v49_final_sidecar_insufficient_grounding"
    )
    checkpoint_fidelity: dict[str, Any] = {"performed": False}
    candidate_output = Path(candidate_checkpoint_path)
    if gate["passed"]:
        expected_state_hash = language_model_state_sha256(candidate)
        save_conditional_adapter_checkpoint(
            candidate_output,
            candidate,
            tokenizer,
            {
                "decision": decision,
                "parent_checkpoint": str(checkpoint),
                "parent_checkpoint_sha256": sha256_file(checkpoint),
                "processed_tokens": int(config.token_budget),
            },
        )
        restored, restored_tokenizer, _metadata = (
            load_conditional_adapter_checkpoint(candidate_output)
        )
        checkpoint_fidelity = {
            "performed": True,
            "path": str(candidate_output),
            "sha256": sha256_file(candidate_output),
            "expected_state_sha256": expected_state_hash,
            "restored_state_sha256": language_model_state_sha256(restored),
            "tokenizer_hash_before": tokenizer.vocabulary_hash(),
            "tokenizer_hash_after": restored_tokenizer.vocabulary_hash(),
        }
        checkpoint_fidelity["passed"] = bool(
            checkpoint_fidelity["expected_state_sha256"]
            == checkpoint_fidelity["restored_state_sha256"]
            and checkpoint_fidelity["tokenizer_hash_before"]
            == checkpoint_fidelity["tokenizer_hash_after"]
        )
        if not checkpoint_fidelity["passed"]:
            candidate_output.unlink(missing_ok=True)
            decision = "invalid_v49_checkpoint_fidelity"
    report = {
        "surface": SURFACE,
        "artifact_kind": ARTIFACT_KIND,
        "decision": decision,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "external_text_data": True,
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
            "grounding_train_path": str(grounding_train_corpus_path),
            "validation_manifest_path": str(grounding_validation_manifest_path),
            "new_domain_tokens_match_v48": True,
            "replay_used": False,
        },
        "parameterization": {
            "parent_parameters": parent_count,
            "adapter_parameters": adapter_count,
            "adapter_fraction_of_parent": adapter_fraction,
            "maximum_adapter_fraction": float(
                config.maximum_adapter_parameter_fraction
            ),
        },
        "optimizer_orthogonalizer_warmup": optimizer_warmup,
        "baseline": {
            "inactive_heldout": baseline_heldout,
            "inactive_relation": baseline_relation,
            "v48_answer_grounding_accuracy": float(config.v48_answer_accuracy),
        },
        "arm": row,
        "inactive_parity": inactive_parity,
        "adapter_gradients": adapter_gradients,
        "adapter_state": adapter_state,
        "source_grounding": source_grounding,
        "gate": gate,
        "checkpoint_fidelity": checkpoint_fidelity,
        "boundary": (
            "V49 tests an explicitly activated frozen-base sidecar. It does not "
            "claim learned routing, general modularity, or runtime installation."
        ),
    }
    write_json_report_with_readme(
        output_path,
        report,
        title="MARULHO V49 Conditional Residual Sidecar",
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
    report = run_conditional_adapter_falsification(
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
