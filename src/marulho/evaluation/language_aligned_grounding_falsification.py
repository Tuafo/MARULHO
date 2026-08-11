"""Falsify V52 document alignment as the missing grounding variable."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from marulho.evaluation.language_matched_support import (
    load_matched_arm_artifact,
    sha256_file,
)
from marulho.evaluation.language_source_grounding_continual import (
    SourceGroundingContinualConfig,
    run_source_grounding_continual,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    language_model_state_sha256,
    load_language_model_checkpoint,
    save_language_model_checkpoint,
)


SURFACE = "marulho_aligned_grounding_falsification.v1"
ARM_NAME = "answer_weight4"


@dataclass(frozen=True)
class AlignedGroundingGateConfig:
    minimum_grounding_accuracy: float = 0.28125
    minimum_source_gain: float = 0.10
    v48_grounding_accuracy: float = 0.21875
    minimum_gain_over_v48: float = 0.05
    maximum_relation_regression: float = 0.05
    maximum_general_loss_regression: float = 0.05


def aligned_grounding_gate(
    row: Mapping[str, Any],
    *,
    baseline_general_loss: float,
    baseline_relation_accuracy: float,
    config: AlignedGroundingGateConfig,
) -> dict[str, Any]:
    source = dict(row["source_grounding"])
    intact = float(source["intact_source"]["exact_answer_accuracy"])
    source_gain = float(source["intact_gain_over_stronger_control"])
    relation = float(row["relation"]["generation_exact_accuracy"])
    general_loss = float(row["heldout"]["heldout_loss"])
    gain_over_v48 = intact - float(config.v48_grounding_accuracy)
    relation_regression = float(baseline_relation_accuracy) - relation
    general_regression = general_loss - float(baseline_general_loss)
    capability_checks = {
        "source_valid": bool(source["valid"]),
        "minimum_grounding_accuracy": intact
        >= float(config.minimum_grounding_accuracy),
        "minimum_source_gain": source_gain >= float(config.minimum_source_gain),
        "minimum_gain_over_v48": gain_over_v48
        >= float(config.minimum_gain_over_v48),
        "exact_token_budget": int(row["processed_tokens"]) == 4_193_280,
        "all_parameters_received_final_gradient": bool(
            row["all_parameters_received_final_gradient"]
        ),
    }
    retention_checks = {
        "relation_retention": relation_regression
        <= float(config.maximum_relation_regression),
        "general_loss_retention": general_regression
        <= float(config.maximum_general_loss_regression),
    }
    capability_passed = all(capability_checks.values())
    retention_passed = all(retention_checks.values())
    return {
        "passed": capability_passed and retention_passed,
        "capability_passed": capability_passed,
        "retention_passed": retention_passed,
        "checks": {**capability_checks, **retention_checks},
        "thresholds": asdict(config),
        "observed": {
            "grounding_accuracy": intact,
            "source_gain": source_gain,
            "gain_over_v48": gain_over_v48,
            "relation_regression": relation_regression,
            "general_loss_regression": general_regression,
        },
    }


def _decision(gate: Mapping[str, Any]) -> str:
    if bool(gate["passed"]):
        return "advance_v52_aligned_grounding_confirmation"
    if bool(gate["capability_passed"]):
        return "advance_v52_aligned_signal_to_isolated_copy"
    return "retire_v52_alignment_insufficient_grounding"


def run_aligned_grounding_falsification(
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
    gate_config: AlignedGroundingGateConfig = AlignedGroundingGateConfig(),
) -> dict[str, Any]:
    training_config = SourceGroundingContinualConfig(
        grounding_window_mode="document_aligned"
    )
    raw = run_source_grounding_continual(
        checkpoint_path=checkpoint_path,
        grounding_train_corpus_path=grounding_train_corpus_path,
        relation_replay_corpus_path=relation_replay_corpus_path,
        relation_cases_path=relation_cases_path,
        general_train_paths=general_train_paths,
        general_eval_paths=general_eval_paths,
        grounding_validation_manifest_path=grounding_validation_manifest_path,
        output_path=output_path,
        candidate_checkpoint_path=candidate_checkpoint_path,
        arm_artifact_directory=arm_artifact_directory,
        config=training_config,
        requested_arms=(ARM_NAME,),
    )
    row = dict(raw["arms"][ARM_NAME])
    baseline_general_loss = float(raw["baseline"]["heldout"]["heldout_loss"])
    baseline_relation_accuracy = float(
        raw["baseline"]["relation"]["generation_exact_accuracy"]
    )
    gate = aligned_grounding_gate(
        row,
        baseline_general_loss=baseline_general_loss,
        baseline_relation_accuracy=baseline_relation_accuracy,
        config=gate_config,
    )
    decision = _decision(gate)
    candidate = Path(candidate_checkpoint_path)
    candidate.unlink(missing_ok=True)
    fidelity: dict[str, Any] = {"performed": False}
    artifact = Path(arm_artifact_directory) / "v48-answer_weight4.pt"
    if bool(gate["passed"]):
        _artifact_row, state = load_matched_arm_artifact(
            artifact,
            expected_arm_name=ARM_NAME,
            expected_contract_sha256=str(row["arm_contract_sha256"]),
        )
        if state is None:
            raise ValueError("V52 passing arm lacks exact model state")
        model, tokenizer, _metadata = load_language_model_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )
        model.load_state_dict(state, strict=True)
        expected_state = language_model_state_sha256(model)
        save_language_model_checkpoint(
            candidate,
            model,
            tokenizer,
            {
                "source_experiment": SURFACE,
                "decision": decision,
                "parent_checkpoint": str(checkpoint_path),
                "parent_checkpoint_sha256": sha256_file(checkpoint_path),
                "processed_tokens": int(training_config.token_budget),
            },
        )
        restored, restored_tokenizer, _metadata = load_language_model_checkpoint(
            candidate,
            map_location="cpu",
        )
        fidelity = {
            "performed": True,
            "checkpoint_path": str(candidate),
            "checkpoint_sha256": sha256_file(candidate),
            "expected_state_sha256": expected_state,
            "restored_state_sha256": language_model_state_sha256(restored),
            "tokenizer_hash_before": tokenizer.vocabulary_hash(),
            "tokenizer_hash_after": restored_tokenizer.vocabulary_hash(),
        }
        fidelity["passed"] = bool(
            fidelity["expected_state_sha256"]
            == fidelity["restored_state_sha256"]
            and fidelity["tokenizer_hash_before"]
            == fidelity["tokenizer_hash_after"]
        )
        if not bool(fidelity["passed"]):
            candidate.unlink(missing_ok=True)
            decision = "invalid_v52_checkpoint_fidelity"
    artifact.unlink(missing_ok=True)
    try:
        Path(arm_artifact_directory).rmdir()
    except OSError:
        pass
    report = {
        "surface": SURFACE,
        "decision": decision,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "external_text_data": True,
        "configuration": {
            "training": asdict(training_config),
            "gate": asdict(gate_config),
        },
        "checkpoint": raw["checkpoint"],
        "data": raw["data"],
        "baseline": raw["baseline"],
        "arm": row,
        "gate": gate,
        "checkpoint_fidelity": fidelity,
        "boundary": (
            "V52 tests whether document alignment repairs visible-source extractive "
            "grounding under V48-matched training. It does not prove open-domain "
            "reasoning or a new architecture."
        ),
    }
    write_json_report_with_readme(
        output_path,
        report,
        title="MARULHO V52 Document-Aligned Grounding",
    )
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
    args = parser.parse_args()
    report = run_aligned_grounding_falsification(
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
    )
    print(report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
