"""Scale the selected V30 general-first Muon recipe on unique local data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import gc
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from marulho.evaluation.language_general_context_falsification import (
    GeneralContextFalsificationConfig,
    _prepare_data,
    _training_config,
    model_state_sha256,
)
from marulho.evaluation.language_matched_support import (
    run_matched_training_arm,
    sha256_file,
)
from marulho.evaluation.language_muon_reproduction import _checkpoint_fidelity
from marulho.evaluation.language_training_experiment import (
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


SURFACE = "marulho_general_scaling.v1"
ARTIFACT_KIND = "marulho_general_scaling"
ADVANCE_DECISION = "save_v31_general_scaling_67m_for_unseen_generation"
STOP_DECISION = "stop_v31_general_scaling_no_durable_loss_gain"
INVALID_DECISION = "invalid_v31_general_scaling_evidence"


@dataclass(frozen=True)
class GeneralScalingConfig:
    token_budget: int = 67_108_864
    sequence_length: int = 72
    batch_size: int = 32
    eval_batches: int = 16
    relation_eval_batch_size: int = 8
    relation_case_limit: int = 0
    learning_rate: float = 1.0e-3
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    data_seed: int = 16121
    model_seed: int = 16131
    sample_bytes_per_train_source: int = 256 * 1024 * 1024
    sample_bytes_per_eval_source: int = 32 * 1024 * 1024
    sample_range_count: int = 16
    schedule_mode: str = "indexed_host"
    execution_backend: str = "inductor"
    compile_loss_tolerance: float = 1.0e-3
    width: int = 512
    layers: int = 4
    heads: int = 8
    mlp_ratio: float = 4.0
    minimum_loss_gain_over_v30: float = 0.15
    baseline_loss_reproduction_tolerance: float = 1.0e-5


def _v30_config(config: GeneralScalingConfig) -> GeneralContextFalsificationConfig:
    return GeneralContextFalsificationConfig(
        token_budget=int(config.token_budget),
        common_sequence_length=int(config.sequence_length),
        common_batch_size=int(config.batch_size),
        long_sequence_length=256,
        long_batch_size=9,
        eval_batches=int(config.eval_batches),
        relation_eval_batch_size=int(config.relation_eval_batch_size),
        relation_case_limit=int(config.relation_case_limit),
        relation_fraction=0.0,
        learning_rate=float(config.learning_rate),
        minimum_learning_rate_fraction=float(
            config.minimum_learning_rate_fraction
        ),
        warmup_fraction=float(config.warmup_fraction),
        weight_decay=float(config.weight_decay),
        gradient_clip=float(config.gradient_clip),
        precision=str(config.precision),
        data_seed=int(config.data_seed),
        model_seed=int(config.model_seed),
        sample_bytes_per_train_source=int(config.sample_bytes_per_train_source),
        sample_bytes_per_eval_source=int(config.sample_bytes_per_eval_source),
        sample_range_count=int(config.sample_range_count),
        schedule_mode=str(config.schedule_mode),
        execution_backend=str(config.execution_backend),
        compile_loss_tolerance=float(config.compile_loss_tolerance),
        width=int(config.width),
        layers=int(config.layers),
        heads=int(config.heads),
        mlp_ratio=float(config.mlp_ratio),
    )


def build_model(
    *,
    vocab_size: int,
    config: GeneralScalingConfig,
) -> MarulhoLanguageModel:
    return MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=int(vocab_size),
            embedding_dim=int(config.width),
            state_dim=int(config.width),
            state_layers=int(config.layers),
            attention_heads=int(config.heads),
            transformer_context_length=int(config.sequence_length),
            transformer_mlp_ratio=float(config.mlp_ratio),
            transformer_dropout=0.0,
            tie_embeddings=True,
            active_language_path="marulho_transformer_v31_general72",
        )
    )


def scaling_decision(
    row: Mapping[str, Any],
    *,
    baseline_loss: float,
    config: GeneralScalingConfig,
    unique_schedule_passed: bool,
    checkpoint_fidelity_passed: bool,
) -> str:
    if not bool(row.get("all_parameters_received_final_gradient")):
        return INVALID_DECISION
    if not bool(unique_schedule_passed):
        return INVALID_DECISION
    loss_gain = float(baseline_loss) - float(row["heldout"]["heldout_loss"])
    if loss_gain < float(config.minimum_loss_gain_over_v30):
        return STOP_DECISION
    if not bool(checkpoint_fidelity_passed):
        return INVALID_DECISION
    return ADVANCE_DECISION


def _schedule_uniqueness(schedule: Sequence[tuple[str, int]]) -> dict[str, Any]:
    counts = Counter(kind for kind, _ in schedule)
    indices: dict[str, list[int]] = defaultdict(list)
    for kind, index in schedule:
        indices[str(kind)].append(int(index))
    unique_counts = {kind: len(set(values)) for kind, values in indices.items()}
    passed = all(unique_counts[kind] == int(count) for kind, count in counts.items())
    return {
        "scheduled_counts": dict(counts),
        "unique_index_counts": unique_counts,
        "every_scheduled_source_index_unique": bool(passed),
    }


def _source_coverage_audit(
    selections: Sequence[Mapping[str, Any]],
    *,
    requested_bytes_per_source: int,
    requested_range_count: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for selection in selections:
        source_size = int(selection["source_size_bytes"])
        selected_size = int(selection["selected_size_bytes"])
        ranges = tuple(selection["ranges"])
        covered_start = min(int(row["start"]) for row in ranges)
        covered_end = max(int(row["end"]) for row in ranges)
        expected_bytes = min(int(requested_bytes_per_source), source_size)
        byte_budget_filled = selected_size >= int(expected_bytes * 0.99)
        range_count = len(ranges)
        stratified = range_count >= min(2, int(requested_range_count))
        span_fraction = (covered_end - covered_start) / max(1, source_size)
        rows.append(
            {
                "path": str(selection["path"]),
                "source_size_bytes": source_size,
                "selected_size_bytes": selected_size,
                "selected_fraction": selected_size / max(1, source_size),
                "range_count": range_count,
                "range_span_fraction": span_fraction,
                "byte_budget_filled": bool(byte_budget_filled),
                "stratified_across_source": bool(
                    stratified and span_fraction >= 0.95
                ),
            }
        )
    passed = bool(rows) and all(
        row["byte_budget_filled"] and row["stratified_across_source"]
        for row in rows
    )
    return {
        "sources": rows,
        "all_sources_stratified_and_budget_filled": passed,
    }


def _split_coverage_audit(
    split_reports: Sequence[Mapping[str, Any]],
    *,
    prepared_batch_counts: Sequence[int],
    batch_size: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, (split, prepared_count) in enumerate(
        zip(split_reports, prepared_batch_counts, strict=True)
    ):
        selection = split["train_window_selection"]
        selected_windows = int(selection["selected_window_count"])
        expected_windows = int(prepared_count) * int(batch_size)
        row = {
            "source_index": index,
            "window_selection": str(split["window_selection"]),
            "source_window_count": int(selection["source_window_count"]),
            "selected_window_count": selected_windows,
            "expected_full_batch_window_count": expected_windows,
            "spans_full_source_window": bool(
                selection["spans_full_source_window"]
            ),
            "prepared_batch_count": int(prepared_count),
        }
        row["passed"] = bool(
            row["window_selection"] == "stratified"
            and row["spans_full_source_window"]
            and selected_windows == expected_windows
        )
        rows.append(row)
    passed = bool(rows) and len(rows) == len(prepared_batch_counts) and all(
        row["passed"] for row in rows
    )
    return {
        "sources": rows,
        "all_prepared_windows_stratified_across_sources": passed,
    }


def run_general_scaling(
    *,
    baseline_checkpoint_path: str | Path,
    baseline_report_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    checkpoint_output_path: str | Path,
    report_output_path: str | Path,
    config: GeneralScalingConfig = GeneralScalingConfig(),
    device: str = "auto",
) -> dict[str, Any]:
    started = time.perf_counter()
    resolved = _resolve_device(device)
    if resolved.type != "cuda":
        raise ValueError("V31 general scaling requires CUDA")
    baseline_checkpoint = Path(baseline_checkpoint_path)
    baseline_report_file = Path(baseline_report_path)
    baseline_report = json.loads(baseline_report_file.read_text(encoding="utf-8"))
    if baseline_report.get("decision") != (
        "save_v30_general_context_candidate_for_unseen_generation"
    ):
        raise ValueError("V31 requires the selected V30 report")
    if baseline_report["selection"]["selected_arm"] != "general72":
        raise ValueError("V31 requires V30 general72 selection")
    checkpoint_output = Path(checkpoint_output_path)
    if checkpoint_output.exists():
        raise ValueError("V31 checkpoint output already exists")
    baseline_model, baseline_tokenizer, baseline_metadata = (
        load_language_model_checkpoint(baseline_checkpoint, map_location="cpu")
    )
    v30_config = _v30_config(config)
    prepared = _prepare_data(
        tokenizer_checkpoint_path=baseline_checkpoint,
        relation_corpus_path=relation_corpus_path,
        relation_cases_path=relation_cases_path,
        general_train_paths=general_train_paths,
        general_eval_paths=general_eval_paths,
        sequence_length=int(config.sequence_length),
        batch_size=int(config.batch_size),
        config=v30_config,
        device=resolved,
    )
    if int(config.relation_case_limit) > 0:
        prepared = replace(
            prepared,
            cases=prepared.cases[: int(config.relation_case_limit)],
        )
    if baseline_tokenizer.vocabulary_hash() != prepared.tokenizer.vocabulary_hash():
        raise ValueError("V31 tokenizer differs from V30")
    baseline_model = baseline_model.to(resolved)
    baseline_heldout = evaluate_language_model(
        baseline_model,
        prepared.eval_batches,
    )
    recorded_baseline_loss = float(
        baseline_report["arms"]["general72"]["common_context_heldout"][
            "heldout_loss"
        ]
    )
    baseline_loss_delta = abs(
        float(baseline_heldout["heldout_loss"]) - recorded_baseline_loss
    )
    if baseline_loss_delta > float(config.baseline_loss_reproduction_tolerance):
        raise ValueError("V31 common holdout does not reproduce V30 baseline loss")
    del baseline_model
    torch.cuda.empty_cache()
    gc.collect()

    torch.manual_seed(int(config.model_seed))
    torch.cuda.manual_seed_all(int(config.model_seed))
    model = build_model(
        vocab_size=int(prepared.tokenizer.vocab_size),
        config=config,
    ).to(resolved)
    initial_state_hash = model_state_sha256(model)
    expected_initial_hash = str(
        baseline_report["matched_truth"]["initial_state_hashes"]["general72"]
    )
    if initial_state_hash != expected_initial_hash:
        raise ValueError("V31 initial state differs from V30")
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    model.eval()
    initial_heldout = evaluate_language_model(model, prepared.eval_batches)
    model.train()
    training_config = _training_config(
        v30_config,
        sequence_length=int(config.sequence_length),
        batch_size=int(config.batch_size),
    )
    warm_batch = prepared.staged.batch(0, resolved)
    print("[general-scaling-v31] compiling general72", flush=True)
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

    print("[general-scaling-v31] training general72", flush=True)
    row = run_matched_training_arm(
        "general72_67m",
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
        progress_prefix="general-scaling-v31",
        extra_row={
            "initial_heldout": initial_heldout,
            "training_context_length": int(config.sequence_length),
            "relation_training_fraction": 0.0,
        },
        optimizer_builder=optimizer_builder,
    )
    schedule_uniqueness = _schedule_uniqueness(prepared.schedule)
    general_sources = prepared.source_selections["general_train"]
    source_coverage = _source_coverage_audit(
        general_sources,
        requested_bytes_per_source=int(config.sample_bytes_per_train_source),
        requested_range_count=int(config.sample_range_count),
    )
    prepared_counts = prepared.source_selections["training_batch_filter"][
        "general_batches_after"
    ]
    split_coverage = _split_coverage_audit(
        prepared.source_selections["general_train_splits"],
        prepared_batch_counts=prepared_counts,
        batch_size=int(config.batch_size),
    )
    scheduled_counts = schedule_uniqueness["scheduled_counts"]
    schedule_covers_prepared_batches = all(
        int(scheduled_counts.get(f"general_{index}", 0)) == int(count)
        for index, count in enumerate(prepared_counts)
    )
    unique_schedule_passed = bool(
        schedule_uniqueness["every_scheduled_source_index_unique"]
        and schedule_covers_prepared_batches
        and source_coverage["all_sources_stratified_and_budget_filled"]
        and split_coverage["all_prepared_windows_stratified_across_sources"]
    )
    precheckpoint_decision = scaling_decision(
        row,
        baseline_loss=float(baseline_heldout["heldout_loss"]),
        config=config,
        unique_schedule_passed=unique_schedule_passed,
        checkpoint_fidelity_passed=True,
    )
    checkpoint_fidelity: dict[str, Any] = {"performed": False, "passed": False}
    checkpoint_sha256 = None
    if precheckpoint_decision == ADVANCE_DECISION:
        save_language_model_checkpoint(
            checkpoint_output,
            model,
            prepared.tokenizer,
            metadata={
                "decision": ADVANCE_DECISION,
                "checkpoint_reproduction": True,
                "processed_tokens": int(row["processed_tokens"]),
                "heldout_loss": float(row["heldout"]["heldout_loss"]),
                "free_relation_accuracy": float(
                    row["relation"]["generation_exact_accuracy"]
                ),
                "baseline_checkpoint_sha256": sha256_file(baseline_checkpoint),
                "optimizer": dict(row["optimizer"]),
                "optimizer_state_saved": False,
                "external_llm_used": False,
            },
        )
        checkpoint_fidelity, _, _, _ = _checkpoint_fidelity(
            model,
            checkpoint_output,
            expected_tokenizer_hash=prepared.tokenizer.vocabulary_hash(),
            sample_input_ids=prepared.eval_batches[0].input_ids,
            device=resolved,
        )
        checkpoint_fidelity["performed"] = True
        checkpoint_sha256 = sha256_file(checkpoint_output)
        if not bool(checkpoint_fidelity["passed"]):
            checkpoint_output.unlink(missing_ok=True)
            checkpoint_sha256 = None
    decision = scaling_decision(
        row,
        baseline_loss=float(baseline_heldout["heldout_loss"]),
        config=config,
        unique_schedule_passed=unique_schedule_passed,
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
            "report_path": str(baseline_report_file),
            "report_sha256": sha256_file(baseline_report_file),
            "metadata": dict(baseline_metadata),
            "heldout": baseline_heldout,
            "recorded_heldout_loss": recorded_baseline_loss,
            "absolute_loss_delta": baseline_loss_delta,
            "loss_reproduction_tolerance": float(
                config.baseline_loss_reproduction_tolerance
            ),
            "loss_reproduced_within_tolerance": True,
        },
        "tokenizer": {
            "vocab_size": int(prepared.tokenizer.vocab_size),
            "vocabulary_hash": prepared.tokenizer.vocabulary_hash(),
        },
        "initial_state": {
            "sha256": initial_state_hash,
            "matches_v30": initial_state_hash == expected_initial_hash,
        },
        "schedule": {
            "sha256": prepared.schedule_sha256,
            "step_count": int(prepared.staged.step_count),
            "tokens_per_step": int(prepared.staged.tokens_per_step),
            "processed_tokens": int(row["processed_tokens"]),
            "source_selections": prepared.source_selections,
            "uniqueness": schedule_uniqueness,
            "schedule_covers_every_prepared_general_batch": (
                schedule_covers_prepared_batches
            ),
            "source_coverage": source_coverage,
            "split_coverage": split_coverage,
            "unique_data_gate_passed": unique_schedule_passed,
        },
        "candidate": row,
        "comparison": {
            "baseline_loss": float(baseline_heldout["heldout_loss"]),
            "candidate_loss": float(row["heldout"]["heldout_loss"]),
            "loss_gain": float(baseline_heldout["heldout_loss"])
            - float(row["heldout"]["heldout_loss"]),
            "minimum_loss_gain": float(config.minimum_loss_gain_over_v30),
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
        title="MARULHO V31 General Scaling",
    )
    print(f"[general-scaling-v31] decision {decision}", flush=True)
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
    parser.add_argument("--token-budget", type=int, default=67_108_864)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--relation-case-limit", type=int, default=0)
    parser.add_argument("--train-sample-mib", type=int, default=256)
    parser.add_argument("--eval-sample-mib", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = run_general_scaling(
        baseline_checkpoint_path=args.baseline_checkpoint,
        baseline_report_path=args.baseline_report,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=args.general_train,
        general_eval_paths=args.general_eval,
        checkpoint_output_path=args.checkpoint_output,
        report_output_path=args.report_output,
        config=GeneralScalingConfig(
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
