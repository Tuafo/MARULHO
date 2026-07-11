"""Reproduce the V29 Muon winner and save a strict unseen-review checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
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
from marulho.evaluation.language_muon_falsification import (
    ARTIFACT_KIND as QUALIFICATION_ARTIFACT_KIND,
    MuonFalsificationConfig,
    _training_config,
    build_model,
)
from marulho.evaluation.language_training_experiment import (
    _prepare_language_loss_backend,
    _resolve_device,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    evaluate_language_model,
    load_language_model_checkpoint,
    save_language_model_checkpoint,
)
from marulho.training.language_muon import build_language_muon


SURFACE = "marulho_muon_checkpoint_reproduction.v1"
ARTIFACT_KIND = "marulho_muon_checkpoint_reproduction"
REQUIRED_QUALIFICATION_DECISION = "advance_v29_muon_to_unseen_generation"
SAVE_DECISION = "save_v29_muon_checkpoint_for_unseen_generation"
REJECT_DECISION = "reject_v29_muon_checkpoint_reproduction"
WINNER_ARM = "muon_1e3"


def load_qualification_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("artifact_kind") != QUALIFICATION_ARTIFACT_KIND:
        raise ValueError("V29 reproduction requires a Muon falsification report")
    if report.get("decision") != REQUIRED_QUALIFICATION_DECISION:
        raise ValueError("V29 qualification report did not advance Muon")
    if set(report.get("arms", {})) != {
        "adamw_3e4",
        "adamw_1e3",
        "muon_3e4",
        "muon_1e3",
    }:
        raise ValueError("V29 qualification report is missing matched arms")
    return report


def reproduction_decision(
    row: Mapping[str, Any],
    qualification: Mapping[str, Any],
    *,
    config: MuonFalsificationConfig,
    checkpoint_fidelity_passed: bool,
) -> str:
    if not bool(row.get("all_parameters_received_final_gradient")):
        return REJECT_DECISION
    comparison = qualification.get("optimizer_comparison")
    if not isinstance(comparison, Mapping):
        return REJECT_DECISION
    loss_gain = float(comparison["adamw_heldout_loss"]) - float(
        row["heldout"]["heldout_loss"]
    )
    free_gain = float(row["relation"]["generation_exact_accuracy"]) - float(
        comparison["adamw_free_relation_accuracy"]
    )
    if loss_gain < float(config.minimum_loss_gain):
        return REJECT_DECISION
    if free_gain < float(config.minimum_free_relation_gain):
        return REJECT_DECISION
    if not bool(checkpoint_fidelity_passed):
        return REJECT_DECISION
    return SAVE_DECISION


def _checkpoint_fidelity(
    original_model,
    checkpoint_path: Path,
    *,
    expected_tokenizer_hash: str,
    sample_input_ids: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, Any], Any, Any, Mapping[str, Any]]:
    restored_model, restored_tokenizer, metadata = load_language_model_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    original_state = {
        name: value.detach().cpu()
        for name, value in original_model.state_dict().items()
    }
    restored_state = restored_model.state_dict()
    exact_keys = set(original_state) == set(restored_state)
    exact_tensors = exact_keys and all(
        torch.equal(original_state[name], restored_state[name])
        for name in original_state
    )
    tokenizer_hash = restored_tokenizer.vocabulary_hash()
    tokenizer_exact = tokenizer_hash == str(expected_tokenizer_hash)
    config_exact = asdict(restored_model.config) == asdict(original_model.config)
    tied_weights = (
        restored_model.token_embedding.weight.data_ptr()
        == restored_model.lm_head.weight.data_ptr()
    )
    restored_model = restored_model.to(device)
    original_model.eval()
    restored_model.eval()
    with torch.no_grad():
        original_logits = original_model(
            sample_input_ids.to(device),
            collect_telemetry=False,
        )["logits"]
        restored_logits = restored_model(
            sample_input_ids.to(device),
            collect_telemetry=False,
        )["logits"]
    maximum_logit_delta = float(
        (original_logits - restored_logits).abs().max().detach().cpu()
    )
    report = {
        "strict_state_keys_equal": bool(exact_keys),
        "strict_state_tensors_bit_equal": bool(exact_tensors),
        "tokenizer_hash": tokenizer_hash,
        "tokenizer_hash_equal": bool(tokenizer_exact),
        "model_config_equal": bool(config_exact),
        "tied_embedding_head_restored": bool(tied_weights),
        "maximum_logit_absolute_delta": maximum_logit_delta,
        "logits_bit_equal": maximum_logit_delta == 0.0,
        "metadata_marks_checkpoint_reproduction": bool(
            metadata.get("checkpoint_reproduction")
        ),
    }
    report["passed"] = all(
        (
            report["strict_state_keys_equal"],
            report["strict_state_tensors_bit_equal"],
            report["tokenizer_hash_equal"],
            report["model_config_equal"],
            report["tied_embedding_head_restored"],
            report["logits_bit_equal"],
            report["metadata_marks_checkpoint_reproduction"],
        )
    )
    return report, restored_model, restored_tokenizer, metadata


def run_muon_checkpoint_reproduction(
    *,
    qualification_report_path: str | Path,
    tokenizer_checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    checkpoint_output_path: str | Path,
    report_output_path: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    started = time.perf_counter()
    qualification_path = Path(qualification_report_path)
    qualification = load_qualification_report(qualification_path)
    config = MuonFalsificationConfig(**dict(qualification["configuration"]))
    resolved = _resolve_device(device)
    if resolved.type != "cuda":
        raise ValueError("V29 checkpoint reproduction requires CUDA")
    checkpoint_output = Path(checkpoint_output_path)
    if checkpoint_output.exists():
        raise ValueError("V29 checkpoint output already exists")
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
    if prepared.schedule_sha256 != qualification["schedule"]["sha256"]:
        raise ValueError("V29 reproduction schedule hash differs from qualification")
    if (
        prepared.tokenizer.vocabulary_hash()
        != qualification["tokenizer"]["vocabulary_hash"]
    ):
        raise ValueError("V29 reproduction tokenizer differs from qualification")
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
    training_config = _training_config(
        config,
        learning_rate=float(config.reference_learning_rate),
    )
    warm_batch = prepared.staged.batch(0, resolved)
    print("[muon-v29-reproduction] compiling shared Transformer", flush=True)
    training_loss, execution = _prepare_language_loss_backend(
        model,
        warm_batch,
        training_config,
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

    print("[muon-v29-reproduction] training muon_1e3", flush=True)
    row = run_matched_training_arm(
        WINNER_ARM,
        architecture="causal_transformer_optimizer_control",
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
        progress_prefix="muon-v29-reproduction",
        extra_row={
            "initial_heldout": initial_heldout,
            "optimizer_kind": "muon",
            "peak_learning_rate": float(config.reference_learning_rate),
        },
        optimizer_builder=optimizer_builder,
    )
    precheckpoint_decision = reproduction_decision(
        row,
        qualification,
        config=config,
        checkpoint_fidelity_passed=True,
    )
    checkpoint_fidelity: dict[str, Any] = {"passed": False, "performed": False}
    checkpoint_sha256 = None
    metadata: Mapping[str, Any] = {}
    if precheckpoint_decision == SAVE_DECISION:
        checkpoint_metadata = {
            "decision": SAVE_DECISION,
            "checkpoint_reproduction": True,
            "qualification_report_path": str(qualification_path),
            "qualification_report_sha256": sha256_file(qualification_path),
            "processed_tokens": int(row["processed_tokens"]),
            "heldout_loss": float(row["heldout"]["heldout_loss"]),
            "free_relation_accuracy": float(
                row["relation"]["generation_exact_accuracy"]
            ),
            "optimizer": dict(row["optimizer"]),
            "optimizer_state_saved": False,
            "external_llm_used": False,
        }
        save_language_model_checkpoint(
            checkpoint_output,
            model,
            prepared.tokenizer,
            metadata=checkpoint_metadata,
        )
        checkpoint_fidelity, _, _, metadata = _checkpoint_fidelity(
            model,
            checkpoint_output,
            expected_tokenizer_hash=prepared.tokenizer.vocabulary_hash(),
            sample_input_ids=prepared.eval_batches[0].input_ids,
            device=resolved,
        )
        checkpoint_fidelity["performed"] = True
        checkpoint_sha256 = sha256_file(checkpoint_output)
    decision = reproduction_decision(
        row,
        qualification,
        config=config,
        checkpoint_fidelity_passed=bool(checkpoint_fidelity["passed"]),
    )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "qualification": {
            "path": str(qualification_path),
            "sha256": sha256_file(qualification_path),
            "required_decision": REQUIRED_QUALIFICATION_DECISION,
            "observed_decision": qualification["decision"],
            "schedule_sha256_equal": True,
            "tokenizer_hash_equal": True,
            "best_adamw": dict(qualification["optimizer_comparison"]),
        },
        "configuration": asdict(config),
        "sources": prepared.source_selections,
        "schedule": {
            "sha256": prepared.schedule_sha256,
            "step_count": int(prepared.staged.step_count),
            "tokens_per_step": int(prepared.staged.tokens_per_step),
            "processed_tokens": int(row["processed_tokens"]),
        },
        "candidate": row,
        "checkpoint": {
            "path": str(checkpoint_output) if checkpoint_output.exists() else None,
            "sha256": checkpoint_sha256,
            "saved": checkpoint_output.exists(),
            "optimizer_state_saved": False,
            "metadata": dict(metadata),
            "fidelity": checkpoint_fidelity,
        },
        "decision": decision,
        "promotion_boundary": {
            "unseen_generation_admitted": decision == SAVE_DECISION,
            "runtime_install_allowed": False,
            "optimizer_installed": False,
            "continual_learning_claimed": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_report_with_readme(
        report_output_path,
        report,
        title="MARULHO V29 Muon Checkpoint Reproduction",
    )
    print(f"[muon-v29-reproduction] decision {decision}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-report", type=Path, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--relation-corpus", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--general-train", action="append", type=Path, required=True)
    parser.add_argument("--general-eval", action="append", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = run_muon_checkpoint_reproduction(
        qualification_report_path=args.qualification_report,
        tokenizer_checkpoint_path=args.tokenizer_checkpoint,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=args.general_train,
        general_eval_paths=args.general_eval,
        checkpoint_output_path=args.checkpoint_output,
        report_output_path=args.report_output,
        device=args.device,
    )
    return 0 if report["decision"] == SAVE_DECISION else 1


if __name__ == "__main__":
    raise SystemExit(main())
