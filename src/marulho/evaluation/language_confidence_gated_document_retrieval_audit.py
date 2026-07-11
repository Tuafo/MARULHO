"""Calibrate retrieval abstention, then test it once on disjoint documents."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from marulho.evaluation.language_causal_document_retrieval_audit import (
    CausalDocumentRetrievalConfig,
    DocumentContinuationCase,
    EncodedDocumentContinuations,
    build_archive_groups,
    build_document_cases,
    encode_document_cases,
    evaluate_document_arm,
    paired_bootstrap_gain,
)
from marulho.evaluation.language_exact_episodic_retrieval_audit import (
    EncodedTextBank,
    lexical_tfidf_scores,
    rankings_from_scores,
)
from marulho.evaluation.language_hashed_micro_expert_continuation import (
    _validate_parent,
)
from marulho.evaluation.language_matched_support import (
    sample_corpus_ranges,
    sha256_file,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_hashed_micro_experts import (
    MarulhoHashedMicroExpertLanguageModel,
    load_hashed_micro_expert_checkpoint,
)


SURFACE = "marulho_confidence_gated_document_retrieval_audit.v1"
ARTIFACT_KIND = "marulho_confidence_gated_document_retrieval_audit"
ADVANCE_DECISION = "advance_v22b_confidence_gated_retrieval_to_joint_training"
GATED_ARM_NAMES = (
    "off",
    "lexical_always1",
    "lexical_gated1",
    "random_gated1",
    "recency_gated1",
    "oracle_gated1",
)


@dataclass(frozen=True)
class ConfidenceGatedRetrievalConfig:
    calibration_cases_per_source: int = 256
    evaluation_cases_per_source: int = 128
    facts_per_query: int = 4
    source_length: int = 48
    prefix_length: int = 48
    target_length: int = 16
    minimum_gap_tokens: int = 48
    maximum_gap_tokens: int = 192
    eval_batch_size: int = 16
    sample_bytes: int = 8 * 1024 * 1024
    sample_range_count: int = 8
    precision: str = "bfloat16"
    data_seed: int = 9801
    bootstrap_samples: int = 4096
    minimum_calibration_precision: float = 0.95
    minimum_calibration_source_precision: float = 0.90
    minimum_calibration_coverage: float = 0.25
    minimum_calibration_source_coverage: float = 0.15
    minimum_evaluation_precision: float = 0.90
    minimum_evaluation_coverage: float = 0.20
    minimum_loss_gain: float = 0.005
    minimum_control_loss_gain: float = 0.0025
    minimum_gate_gain_over_always: float = 0.0025
    maximum_source_loss_regression: float = 0.0


@dataclass(frozen=True)
class PreparedRetrievalSplit:
    name: str
    cases: tuple[DocumentContinuationCase, ...]
    bank: EncodedDocumentContinuations
    groups: torch.Tensor
    target_slots: torch.Tensor
    lexical_scores: torch.Tensor
    lexical_rankings: torch.Tensor
    lexical_margin: torch.Tensor
    source_reports: tuple[dict[str, Any], ...]


def _causal_config(
    config: ConfidenceGatedRetrievalConfig,
    *,
    case_count_per_source: int,
) -> CausalDocumentRetrievalConfig:
    return CausalDocumentRetrievalConfig(
        case_count_per_source=int(case_count_per_source),
        facts_per_query=int(config.facts_per_query),
        source_length=int(config.source_length),
        prefix_length=int(config.prefix_length),
        target_length=int(config.target_length),
        minimum_gap_tokens=int(config.minimum_gap_tokens),
        maximum_gap_tokens=int(config.maximum_gap_tokens),
        eval_batch_size=int(config.eval_batch_size),
        feature_batch_size=int(config.eval_batch_size),
        sample_bytes=int(config.sample_bytes),
        sample_range_count=int(config.sample_range_count),
        precision=str(config.precision),
        data_seed=int(config.data_seed),
        bootstrap_samples=int(config.bootstrap_samples),
    )


def _special_token_ids(tokenizer) -> tuple[int, ...]:
    return (
        int(tokenizer.pad_id),
        int(tokenizer.bos_id),
        int(tokenizer.eos_id),
        int(tokenizer.unk_id),
        int(tokenizer.checkpoint_id),
        int(tokenizer.replay_id),
    )


def _tensor_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().contiguous().cpu()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def prepare_retrieval_split(
    tokenizer,
    paths: Sequence[str | Path],
    *,
    name: str,
    case_count_per_source: int,
    config: ConfidenceGatedRetrievalConfig,
    seed: int,
) -> PreparedRetrievalSplit:
    if len(paths) != 2:
        raise ValueError(f"{name} requires exactly two corpus sources")
    causal_config = _causal_config(
        config, case_count_per_source=int(case_count_per_source)
    )
    cases: list[DocumentContinuationCase] = []
    source_reports = []
    for source_index, raw_path in enumerate(paths):
        path = Path(raw_path)
        text, sample_report = sample_corpus_ranges(
            path,
            byte_budget=int(config.sample_bytes),
            range_count=int(config.sample_range_count),
        )
        selected, selection_report = build_document_cases(
            tokenizer,
            text,
            source_index=source_index,
            source_name=path.stem,
            config=causal_config,
            seed=int(seed) + source_index,
        )
        cases.extend(selected)
        source_reports.append(
            {
                **sample_report,
                **selection_report,
                "file_sha256": sha256_file(path),
                "split": str(name),
            }
        )
    frozen_cases = tuple(cases)
    bank = encode_document_cases(frozen_cases, config=causal_config)
    groups, target_slots = build_archive_groups(
        frozen_cases,
        facts_per_query=int(config.facts_per_query),
        seed=int(seed) + 10,
    )
    source_bank = EncodedTextBank(
        ids=bank.source_ids,
        mask=torch.ones_like(bank.source_ids, dtype=torch.bool),
    )
    query_bank = EncodedTextBank(
        ids=bank.prefix_ids,
        mask=torch.ones_like(bank.prefix_ids, dtype=torch.bool),
    )
    lexical_scores = lexical_tfidf_scores(
        source_bank,
        query_bank,
        groups,
        excluded_token_ids=_special_token_ids(tokenizer),
    )
    sorted_scores, lexical_rankings = torch.sort(
        lexical_scores, dim=1, descending=True, stable=True
    )
    return PreparedRetrievalSplit(
        name=str(name),
        cases=frozen_cases,
        bank=bank,
        groups=groups,
        target_slots=target_slots,
        lexical_scores=lexical_scores,
        lexical_rankings=lexical_rankings,
        lexical_margin=sorted_scores[:, 0] - sorted_scores[:, 1],
        source_reports=tuple(source_reports),
    )


def _selection_row(
    mask: torch.Tensor,
    correct: torch.Tensor,
    cases: Sequence[DocumentContinuationCase],
    *,
    threshold: float,
) -> dict[str, Any]:
    selected_count = int(mask.sum())
    selected_correct = int((correct & mask).sum())
    per_source = {}
    for source_name in sorted({case.source_name for case in cases}):
        indices = torch.tensor(
            [
                index
                for index, case in enumerate(cases)
                if case.source_name == source_name
            ],
            dtype=torch.long,
        )
        source_mask = mask.index_select(0, indices)
        source_correct = correct.index_select(0, indices)
        count = int(source_mask.sum())
        per_source[source_name] = {
            "case_count": int(indices.numel()),
            "selected_count": count,
            "coverage": count / int(indices.numel()),
            "precision": (
                0.0 if count == 0 else float(source_correct[source_mask].float().mean())
            ),
        }
    return {
        "threshold": float(threshold),
        "case_count": len(cases),
        "selected_count": selected_count,
        "coverage": selected_count / len(cases),
        "precision": 0.0 if selected_count == 0 else selected_correct / selected_count,
        "effective_correct_retrieval_rate": selected_correct / len(cases),
        "per_source": per_source,
    }


def calibrate_margin_threshold(
    margins: torch.Tensor,
    rankings: torch.Tensor,
    target_slots: torch.Tensor,
    cases: Sequence[DocumentContinuationCase],
    *,
    config: ConfidenceGatedRetrievalConfig,
) -> tuple[float | None, dict[str, Any]]:
    if margins.ndim != 1 or rankings.ndim != 2:
        raise ValueError("calibration margin/ranking shapes are invalid")
    correct = rankings[:, 0] == target_slots
    thresholds = torch.unique(margins).sort(descending=True).values
    rows = []
    qualified = []
    for raw_threshold in thresholds.tolist():
        threshold = float(raw_threshold)
        row = _selection_row(
            margins >= threshold,
            correct,
            cases,
            threshold=threshold,
        )
        rows.append(row)
        source_rows = row["per_source"].values()
        if (
            float(row["precision"]) >= float(config.minimum_calibration_precision)
            and float(row["coverage"]) >= float(config.minimum_calibration_coverage)
            and min(float(value["precision"]) for value in source_rows)
            >= float(config.minimum_calibration_source_precision)
            and min(float(value["coverage"]) for value in source_rows)
            >= float(config.minimum_calibration_source_coverage)
        ):
            qualified.append(row)
    selected = (
        None
        if not qualified
        else max(
            qualified,
            key=lambda row: (
                float(row["coverage"]),
                float(row["precision"]),
                -float(row["threshold"]),
            ),
        )
    )
    diagnostic_coverages = []
    for requested in (0.25, 0.50, 0.75, 1.0):
        target_count = max(1, round(requested * len(cases)))
        closest = min(rows, key=lambda row: abs(int(row["selected_count"]) - target_count))
        diagnostic_coverages.append(
            {"requested_coverage": requested, **closest}
        )
    return (
        None if selected is None else float(selected["threshold"]),
        {
            "selected": selected,
            "qualified_threshold_count": len(qualified),
            "candidate_threshold_count": len(rows),
            "diagnostic_coverages": diagnostic_coverages,
            "selection_uses_same_document_identity": True,
            "selection_uses_future_target_tokens": False,
            "selection_uses_language_loss": False,
        },
    )


def gate_retrieval_metrics(
    selected_slots: torch.Tensor | None,
    gate_mask: torch.Tensor,
    target_slots: torch.Tensor,
    cases: Sequence[DocumentContinuationCase],
    *,
    source_length: int,
    oracle: bool,
) -> dict[str, Any]:
    if selected_slots is None:
        correct = torch.zeros(len(cases), dtype=torch.bool)
    else:
        correct = selected_slots[:, 0] == target_slots
    selected_count = int(gate_mask.sum())
    selected_correct = int((gate_mask & correct).sum())
    per_source = {}
    for source_name in sorted({case.source_name for case in cases}):
        indices = torch.tensor(
            [index for index, case in enumerate(cases) if case.source_name == source_name],
            dtype=torch.long,
        )
        source_gate = gate_mask.index_select(0, indices)
        source_correct = correct.index_select(0, indices)
        count = int(source_gate.sum())
        per_source[source_name] = {
            "case_count": int(indices.numel()),
            "selected_count": count,
            "coverage": count / int(indices.numel()),
            "precision": (
                0.0 if count == 0 else float(source_correct[source_gate].float().mean())
            ),
            "effective_correct_retrieval_rate": float(
                (source_gate & source_correct).float().mean()
            ),
        }
    return {
        "selected_count": selected_count,
        "coverage": selected_count / len(cases),
        "precision": 0.0 if selected_count == 0 else selected_correct / selected_count,
        "effective_correct_retrieval_rate": selected_correct / len(cases),
        "active_source_tokens_total": selected_count * int(source_length),
        "mean_active_source_tokens": (
            selected_count * int(source_length) / len(cases)
        ),
        "per_source": per_source,
        "gate_uses_target_identity": False,
        "selected_slots_use_target_identity": bool(oracle),
        "target_identity_metrics_only": True,
    }


def compose_gated_language(
    off: Mapping[str, Any],
    active: Mapping[str, Any],
    gate_mask: torch.Tensor,
    cases: Sequence[DocumentContinuationCase],
    *,
    config: ConfidenceGatedRetrievalConfig,
    seed: int,
) -> dict[str, Any]:
    off_losses = torch.tensor(off["case_losses"], dtype=torch.float64)
    active_losses = torch.tensor(active["case_losses"], dtype=torch.float64)
    off_accuracy = torch.tensor(off["case_next_token_accuracy"], dtype=torch.float64)
    active_accuracy = torch.tensor(
        active["case_next_token_accuracy"], dtype=torch.float64
    )
    losses = torch.where(gate_mask, active_losses, off_losses)
    accuracies = torch.where(gate_mask, active_accuracy, off_accuracy)
    paired = paired_bootstrap_gain(
        off_losses.tolist(),
        losses.tolist(),
        samples=int(config.bootstrap_samples),
        seed=int(seed),
    )
    per_source = {}
    for source_index, source_name in enumerate(
        sorted({case.source_name for case in cases})
    ):
        indices = torch.tensor(
            [index for index, case in enumerate(cases) if case.source_name == source_name],
            dtype=torch.long,
        )
        source_losses = losses.index_select(0, indices)
        source_accuracy = accuracies.index_select(0, indices)
        per_source[source_name] = {
            "case_count": int(indices.numel()),
            "heldout_loss": float(source_losses.mean()),
            "next_token_accuracy": float(source_accuracy.mean()),
            "paired_to_off": paired_bootstrap_gain(
                off_losses.index_select(0, indices).tolist(),
                source_losses.tolist(),
                samples=int(config.bootstrap_samples),
                seed=int(seed) + 100 + source_index,
            ),
        }
    return {
        "heldout_loss": float(losses.mean()),
        "next_token_accuracy": float(accuracies.mean()),
        "case_count": len(cases),
        "paired_to_off": paired,
        "per_source": per_source,
        "offline_composition_is_exact_per_case_path_selection": True,
        "speed_claimed": False,
    }


def confidence_gate_decision(
    *,
    calibration: Mapping[str, Any],
    arms: Mapping[str, Mapping[str, Any]],
    config: ConfidenceGatedRetrievalConfig,
) -> str:
    if calibration.get("selected") is None:
        return "retire_v22b_no_calibration_threshold_meets_precision"
    candidate = arms["lexical_gated1"]
    retrieval = candidate["retrieval"]
    language = candidate["language"]
    if (
        float(retrieval["precision"]) < float(config.minimum_evaluation_precision)
        or float(retrieval["coverage"]) < float(config.minimum_evaluation_coverage)
    ):
        return "retire_v22b_fixed_lexical_confidence_gate_nontransfer"
    oracle = arms["oracle_gated1"]["language"]["paired_to_off"]
    if float(oracle["bootstrap_95_ci"][0]) <= 0.0:
        return "redesign_v22b_calibrated_coverage_not_predictively_useful"
    off_loss = float(arms["off"]["language"]["heldout_loss"])
    candidate_loss = float(language["heldout_loss"])
    random_loss = float(arms["random_gated1"]["language"]["heldout_loss"])
    recency_loss = float(arms["recency_gated1"]["language"]["heldout_loss"])
    always_loss = float(arms["lexical_always1"]["language"]["heldout_loss"])
    source_gains = [
        float(row["paired_to_off"]["mean_loss_gain"])
        for row in language["per_source"].values()
    ]
    if (
        off_loss - candidate_loss >= float(config.minimum_loss_gain)
        and random_loss - candidate_loss >= float(config.minimum_control_loss_gain)
        and recency_loss - candidate_loss >= float(config.minimum_control_loss_gain)
        and always_loss - candidate_loss
        >= float(config.minimum_gate_gain_over_always)
        and float(language["paired_to_off"]["bootstrap_95_ci"][0]) > 0.0
        and min(source_gains) >= -float(config.maximum_source_loss_regression)
    ):
        return ADVANCE_DECISION
    return "retire_v22b_fixed_confidence_gate_insufficient_language_gain"


def run_confidence_gated_document_retrieval_audit(
    *,
    parent_checkpoint_path: str | Path,
    calibration_paths: Sequence[str | Path],
    eval_paths: Sequence[str | Path],
    output_path: str | Path,
    config: ConfidenceGatedRetrievalConfig = ConfidenceGatedRetrievalConfig(),
    device: str = "auto",
) -> dict[str, Any]:
    resolved = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for V22b but unavailable")
    started = time.perf_counter()
    parent_path = Path(parent_checkpoint_path)
    model, tokenizer, parent_metadata = load_hashed_micro_expert_checkpoint(
        parent_path, map_location="cpu"
    )
    parent_tokens = _validate_parent(model, parent_metadata)
    if parent_tokens < 1_000_000_000:
        raise ValueError("V22b requires the one-billion-token V11 parent")
    maximum_sequence = (
        int(config.source_length)
        + int(config.prefix_length)
        + int(config.target_length)
        - 1
    )
    if maximum_sequence > int(model.hashed_config.context_length):
        raise ValueError("V22b one-episode sequence exceeds the cortex window")

    print("[confidence-gate-v22b] preparing calibration documents", flush=True)
    calibration_split = prepare_retrieval_split(
        tokenizer,
        calibration_paths,
        name="calibration_replay",
        case_count_per_source=int(config.calibration_cases_per_source),
        config=config,
        seed=int(config.data_seed),
    )
    threshold, calibration = calibrate_margin_threshold(
        calibration_split.lexical_margin,
        calibration_split.lexical_rankings,
        calibration_split.target_slots,
        calibration_split.cases,
        config=config,
    )
    print(
        f"[confidence-gate-v22b] frozen threshold={threshold} "
        f"qualified={calibration['qualified_threshold_count']}",
        flush=True,
    )
    print("[confidence-gate-v22b] preparing disjoint evaluation", flush=True)
    evaluation = prepare_retrieval_split(
        tokenizer,
        eval_paths,
        name="document_disjoint_evaluation",
        case_count_per_source=int(config.evaluation_cases_per_source),
        config=config,
        seed=int(config.data_seed) + 1000,
    )
    calibration_hashes = {case.document_sha256 for case in calibration_split.cases}
    evaluation_hashes = {case.document_sha256 for case in evaluation.cases}
    overlap = calibration_hashes & evaluation_hashes
    if overlap:
        raise RuntimeError("V22b calibration/evaluation documents overlap")

    lexical_slots = evaluation.lexical_rankings[:, :1]
    generator = torch.Generator(device="cpu").manual_seed(int(config.data_seed) + 2000)
    random_slots = rankings_from_scores(
        torch.rand(evaluation.lexical_scores.shape, generator=generator)
    )[:, :1]
    recency_slots = torch.full(
        (len(evaluation.cases), 1),
        int(config.facts_per_query) - 1,
        dtype=torch.long,
    )
    oracle_slots = evaluation.target_slots.unsqueeze(1)
    gate_mask = (
        torch.zeros(len(evaluation.cases), dtype=torch.bool)
        if threshold is None
        else evaluation.lexical_margin >= float(threshold)
    )

    model = model.to(resolved).eval()
    raw_slots = {
        "off": None,
        "lexical_always1": lexical_slots,
        "random_gated1": random_slots,
        "recency_gated1": recency_slots,
        "oracle_gated1": oracle_slots,
    }
    raw = {}
    for name, slots in raw_slots.items():
        print(f"[confidence-gate-v22b] evaluating raw path {name}", flush=True)
        raw[name] = evaluate_document_arm(
            model,
            evaluation.bank,
            evaluation.groups,
            slots,
            evaluation.cases,
            batch_size=int(config.eval_batch_size),
            precision=str(config.precision),
        )

    arms = {
        "off": {
            "retrieval": gate_retrieval_metrics(
                None,
                torch.zeros_like(gate_mask),
                evaluation.target_slots,
                evaluation.cases,
                source_length=int(config.source_length),
                oracle=False,
            ),
            "language": compose_gated_language(
                raw["off"],
                raw["off"],
                torch.zeros_like(gate_mask),
                evaluation.cases,
                config=config,
                seed=int(config.data_seed) + 3000,
            ),
        },
        "lexical_always1": {
            "retrieval": gate_retrieval_metrics(
                lexical_slots,
                torch.ones_like(gate_mask),
                evaluation.target_slots,
                evaluation.cases,
                source_length=int(config.source_length),
                oracle=False,
            ),
            "language": compose_gated_language(
                raw["off"],
                raw["lexical_always1"],
                torch.ones_like(gate_mask),
                evaluation.cases,
                config=config,
                seed=int(config.data_seed) + 3010,
            ),
        },
    }
    for index, (name, raw_name, slots, oracle) in enumerate(
        (
            ("lexical_gated1", "lexical_always1", lexical_slots, False),
            ("random_gated1", "random_gated1", random_slots, False),
            ("recency_gated1", "recency_gated1", recency_slots, False),
            ("oracle_gated1", "oracle_gated1", oracle_slots, True),
        )
    ):
        arms[name] = {
            "retrieval": gate_retrieval_metrics(
                slots,
                gate_mask,
                evaluation.target_slots,
                evaluation.cases,
                source_length=int(config.source_length),
                oracle=oracle,
            ),
            "language": compose_gated_language(
                raw["off"],
                raw[raw_name],
                gate_mask,
                evaluation.cases,
                config=config,
                seed=int(config.data_seed) + 3020 + index,
            ),
        }
    decision = confidence_gate_decision(
        calibration=calibration,
        arms=arms,
        config=config,
    )

    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "configuration": asdict(config),
        "parent": {
            "path": str(parent_path),
            "sha256": sha256_file(parent_path),
            "processed_tokens": parent_tokens,
            "decision": parent_metadata.get("decision"),
            "tokenizer_hash": tokenizer.vocabulary_hash(),
            "parameters_frozen": True,
            "parameter_gradients_enabled": False,
        },
        "calibration": {
            **calibration,
            "frozen_threshold": threshold,
            "source_reports": calibration_split.source_reports,
            "case_count": len(calibration_split.cases),
            "margin_sha256": _tensor_sha256(calibration_split.lexical_margin),
            "group_sha256": _tensor_sha256(
                calibration_split.groups, calibration_split.target_slots
            ),
        },
        "evaluation": {
            "source_reports": evaluation.source_reports,
            "case_count": len(evaluation.cases),
            "margin_sha256": _tensor_sha256(evaluation.lexical_margin),
            "group_sha256": _tensor_sha256(
                evaluation.groups, evaluation.target_slots
            ),
            "calibration_document_overlap": len(overlap),
        },
        "anti_cheat": {
            "threshold_fit_split": "separate_replay_documents",
            "threshold_fit_uses_same_document_identity": True,
            "threshold_fit_uses_future_target_tokens": False,
            "threshold_fit_uses_language_loss": False,
            "evaluation_threshold_frozen_before_eval_preparation": True,
            "evaluation_gate_input": "visible_prefix_lexical_top1_minus_top2_margin",
            "evaluation_gate_uses_target_identity": False,
            "evaluation_gate_uses_future_target_tokens": False,
            "evaluation_target_identity_metrics_only": True,
            "calibration_eval_document_hash_overlap": len(overlap),
            "random_recency_controls_share_exact_gate_mask": True,
            "oracle_selected_slots_promotable": False,
        },
        "architecture": {
            "archive_content": "exact_prior_document_token_spans",
            "key": "checkpoint_bpe_tfidf",
            "read_policy": "frozen_margin_threshold_retrieve_one_or_abstain",
            "maximum_active_source_tokens": int(config.source_length),
            "learned_language_parameters": False,
            "joint_training": False,
        },
        "arms": arms,
        "decision": decision,
        "promotion_boundary": {
            "advance_to_joint_training": decision == ADVANCE_DECISION,
            "base_language_quality_promoted": False,
            "checkpoint_saved": False,
            "runtime_install_allowed": False,
            "continual_learning_claimed": False,
            "speed_claimed": False,
        },
        "hardware": {
            "device": str(resolved),
            "cuda_device_name": (
                torch.cuda.get_device_name(resolved)
                if resolved.type == "cuda"
                else None
            ),
            "torch_version": torch.__version__,
        },
        "experiment_wall_seconds": time.perf_counter() - started,
    }
    write_json_report_with_readme(
        output_path,
        report,
        title="MARULHO V22b Confidence-Gated Document Retrieval Audit",
    )
    print(f"[confidence-gate-v22b] decision {decision}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--calibration-source", type=Path, action="append", required=True
    )
    parser.add_argument("--eval-source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-cases-per-source", type=int, default=256)
    parser.add_argument("--evaluation-cases-per-source", type=int, default=128)
    parser.add_argument("--sample-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--sample-range-count", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=4096)
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--data-seed", type=int, default=9801)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = ConfidenceGatedRetrievalConfig(
        calibration_cases_per_source=int(args.calibration_cases_per_source),
        evaluation_cases_per_source=int(args.evaluation_cases_per_source),
        sample_bytes=int(args.sample_bytes),
        sample_range_count=int(args.sample_range_count),
        eval_batch_size=int(args.eval_batch_size),
        bootstrap_samples=int(args.bootstrap_samples),
        precision=str(args.precision),
        data_seed=int(args.data_seed),
    )
    run_confidence_gated_document_retrieval_audit(
        parent_checkpoint_path=args.parent_checkpoint,
        calibration_paths=args.calibration_source,
        eval_paths=args.eval_source,
        output_path=args.output,
        config=config,
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
