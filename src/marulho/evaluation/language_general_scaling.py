"""Scale the selected general-first Muon recipe on unique local data."""

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


SURFACE = "marulho_general_scaling.v2"
ARTIFACT_KIND = "marulho_general_scaling"


@dataclass(frozen=True)
class GeneralScalingStage:
    name: str
    candidate_name: str
    progress_prefix: str
    active_language_path: str
    required_baseline_artifact_kind: str
    required_baseline_decision: str
    required_selected_arm: str | None
    baseline_loss_path: tuple[str, ...]
    baseline_initial_hash_path: tuple[str, ...]
    token_budget: int
    train_sample_bytes: int
    minimum_loss_gain: float
    advance_decision: str
    stop_decision: str
    invalid_decision: str
    report_title: str


V31_STAGE = GeneralScalingStage(
    name="v31",
    candidate_name="general72_67m",
    progress_prefix="general-scaling-v31",
    active_language_path="marulho_transformer_v31_general72",
    required_baseline_artifact_kind="marulho_general_context_falsification",
    required_baseline_decision=(
        "save_v30_general_context_candidate_for_unseen_generation"
    ),
    required_selected_arm="general72",
    baseline_loss_path=(
        "arms",
        "general72",
        "common_context_heldout",
        "heldout_loss",
    ),
    baseline_initial_hash_path=(
        "matched_truth",
        "initial_state_hashes",
        "general72",
    ),
    token_budget=67_108_864,
    train_sample_bytes=256 * 1024 * 1024,
    minimum_loss_gain=0.15,
    advance_decision="save_v31_general_scaling_67m_for_unseen_generation",
    stop_decision="stop_v31_general_scaling_no_durable_loss_gain",
    invalid_decision="invalid_v31_general_scaling_evidence",
    report_title="MARULHO V31 General Scaling",
)


V32_STAGE = GeneralScalingStage(
    name="v32",
    candidate_name="general72_201m",
    progress_prefix="general-scaling-v32",
    active_language_path="marulho_transformer_v32_general72",
    required_baseline_artifact_kind=ARTIFACT_KIND,
    required_baseline_decision=V31_STAGE.advance_decision,
    required_selected_arm=None,
    baseline_loss_path=("candidate", "heldout", "heldout_loss"),
    baseline_initial_hash_path=("initial_state", "sha256"),
    token_budget=201_323_520,
    train_sample_bytes=512 * 1024 * 1024,
    minimum_loss_gain=0.20,
    advance_decision="save_v32_general_scaling_201m_for_unseen_generation",
    stop_decision="stop_v32_general_scaling_no_durable_loss_gain",
    invalid_decision="invalid_v32_general_scaling_evidence",
    report_title="MARULHO V32 General Scaling",
)


STAGES = {stage.name: stage for stage in (V31_STAGE, V32_STAGE)}
ADVANCE_DECISION = V31_STAGE.advance_decision
STOP_DECISION = V31_STAGE.stop_decision
INVALID_DECISION = V31_STAGE.invalid_decision


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
    minimum_loss_gain: float = 0.15
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
    stage: GeneralScalingStage = V31_STAGE,
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
            active_language_path=str(stage.active_language_path),
        )
    )


def scaling_decision(
    row: Mapping[str, Any],
    *,
    baseline_loss: float,
    config: GeneralScalingConfig,
    unique_schedule_passed: bool,
    checkpoint_fidelity_passed: bool,
    stage: GeneralScalingStage = V31_STAGE,
) -> str:
    if not bool(row.get("all_parameters_received_final_gradient")):
        return stage.invalid_decision
    if not bool(unique_schedule_passed):
        return stage.invalid_decision
    loss_gain = float(baseline_loss) - float(row["heldout"]["heldout_loss"])
    if loss_gain < float(config.minimum_loss_gain):
        return stage.stop_decision
    if not bool(checkpoint_fidelity_passed):
        return stage.invalid_decision
    return stage.advance_decision


def _nested_value(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"missing scaling evidence field: {'.'.join(path)}")
        value = value[key]
    return value


def _validate_baseline(
    report: Mapping[str, Any],
    *,
    stage: GeneralScalingStage,
) -> tuple[float, str]:
    if report.get("artifact_kind") != stage.required_baseline_artifact_kind:
        raise ValueError(f"{stage.name} baseline artifact kind is invalid")
    if report.get("decision") != stage.required_baseline_decision:
        raise ValueError(f"{stage.name} baseline decision is invalid")
    if stage.required_selected_arm is not None:
        selection = report.get("selection")
        if not isinstance(selection, Mapping) or selection.get(
            "selected_arm"
        ) != stage.required_selected_arm:
            raise ValueError(f"{stage.name} baseline arm is invalid")
    return (
        float(_nested_value(report, stage.baseline_loss_path)),
        str(_nested_value(report, stage.baseline_initial_hash_path)),
    )


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
        full_source_selected = selected_size == source_size
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
                "full_source_selected": full_source_selected,
                "byte_budget_filled": bool(byte_budget_filled),
                "stratified_across_source": bool(
                    (full_source_selected or stratified)
                    and span_fraction >= 0.95
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
    stage: GeneralScalingStage = V31_STAGE,
    device: str = "auto",
) -> dict[str, Any]:
    started = time.perf_counter()
    resolved = _resolve_device(device)
    if resolved.type != "cuda":
        raise ValueError(f"{stage.name} general scaling requires CUDA")
    baseline_checkpoint = Path(baseline_checkpoint_path)
    baseline_report_file = Path(baseline_report_path)
    baseline_report = json.loads(baseline_report_file.read_text(encoding="utf-8"))
    recorded_baseline_loss, expected_initial_hash = _validate_baseline(
        baseline_report,
        stage=stage,
    )
    checkpoint_output = Path(checkpoint_output_path)
    if checkpoint_output.exists():
        raise ValueError(f"{stage.name} checkpoint output already exists")
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
        raise ValueError(f"{stage.name} tokenizer differs from its baseline")
    baseline_model = baseline_model.to(resolved)
    baseline_heldout = evaluate_language_model(
        baseline_model,
        prepared.eval_batches,
    )
    baseline_loss_delta = abs(
        float(baseline_heldout["heldout_loss"]) - recorded_baseline_loss
    )
    if baseline_loss_delta > float(config.baseline_loss_reproduction_tolerance):
        raise ValueError(
            f"{stage.name} common holdout does not reproduce its baseline loss"
        )
    del baseline_model
    torch.cuda.empty_cache()
    gc.collect()

    torch.manual_seed(int(config.model_seed))
    torch.cuda.manual_seed_all(int(config.model_seed))
    model = build_model(
        vocab_size=int(prepared.tokenizer.vocab_size),
        config=config,
        stage=stage,
    ).to(resolved)
    initial_state_hash = model_state_sha256(model)
    if initial_state_hash != expected_initial_hash:
        raise ValueError(f"{stage.name} initial state differs from its baseline")
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
    print(f"[{stage.progress_prefix}] compiling general72", flush=True)
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

    print(f"[{stage.progress_prefix}] training general72", flush=True)
    row = run_matched_training_arm(
        stage.candidate_name,
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
        progress_prefix=stage.progress_prefix,
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
        stage=stage,
    )
    checkpoint_fidelity: dict[str, Any] = {"performed": False, "passed": False}
    checkpoint_sha256 = None
    if precheckpoint_decision == stage.advance_decision:
        save_language_model_checkpoint(
            checkpoint_output,
            model,
            prepared.tokenizer,
            metadata={
                "decision": stage.advance_decision,
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
        stage=stage,
    )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "stage": asdict(stage),
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
            "matches_baseline": initial_state_hash == expected_initial_hash,
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
            "minimum_loss_gain": float(config.minimum_loss_gain),
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
            "unseen_generation_admitted": decision == stage.advance_decision,
            "base_quality_promoted": False,
            "runtime_install_allowed": False,
            "continual_learning_claimed": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_report_with_readme(
        report_output_path,
        report,
        title=stage.report_title,
    )
    print(f"[{stage.progress_prefix}] decision {decision}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), default="v31")
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--relation-corpus", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--general-train", action="append", type=Path, required=True)
    parser.add_argument("--general-eval", action="append", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--relation-case-limit", type=int, default=0)
    parser.add_argument("--train-sample-mib", type=int, default=None)
    parser.add_argument("--eval-sample-mib", type=int, default=32)
    parser.add_argument("--minimum-loss-gain", type=float, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    stage = STAGES[str(args.stage)]
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
            token_budget=max(
                1,
                int(
                    stage.token_budget
                    if args.token_budget is None
                    else args.token_budget
                ),
            ),
            eval_batches=max(1, int(args.eval_batches)),
            relation_case_limit=max(0, int(args.relation_case_limit)),
            sample_bytes_per_train_source=max(
                1,
                int(
                    stage.train_sample_bytes // (1024 * 1024)
                    if args.train_sample_mib is None
                    else args.train_sample_mib
                ),
            )
            * 1024
            * 1024,
            sample_bytes_per_eval_source=max(1, int(args.eval_sample_mib))
            * 1024
            * 1024,
            minimum_loss_gain=float(
                stage.minimum_loss_gain
                if args.minimum_loss_gain is None
                else args.minimum_loss_gain
            ),
        ),
        stage=stage,
        device=args.device,
    )
    return 0 if report["decision"] == stage.advance_decision else 1


if __name__ == "__main__":
    raise SystemExit(main())
