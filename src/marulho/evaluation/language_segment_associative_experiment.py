"""Matched V14 causal segment-associative state falsifier from the 1B V11 base."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from marulho.evaluation.language_hashed_micro_expert_continuation import (
    _validate_parent,
)
from marulho.evaluation.language_matched_support import (
    MatchedLanguageDataConfig,
    prepare_matched_language_data,
    run_matched_training_arm,
    sha256_file,
)
from marulho.evaluation.language_relation_binding_experiment import (
    evaluate_relation_binding_cases_batched,
)
from marulho.evaluation.language_training_experiment import (
    LanguageTrainingExperimentConfig,
    _prepare_language_loss_backend,
    _resolve_device,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_hashed_micro_experts import (
    load_hashed_micro_expert_checkpoint,
)
from marulho.training.language_model import evaluate_language_model
from marulho.training.language_segment_associative_state import (
    SEGMENT_ASSOCIATIVE_MODES,
    SegmentAssociativeConfig,
    build_segment_associative_model,
    save_segment_associative_checkpoint,
)


SURFACE = "marulho_segment_associative_experiment.v1"
ARTIFACT_KIND = "marulho_segment_associative_experiment"
SAVE_DECISION = "save_v14_gated_segment_state_for_unseen_generation"


@dataclass(frozen=True)
class SegmentAssociativeExperimentConfig:
    token_budget: int = 67_108_864
    sequence_length: int = 256
    batch_size: int = 40
    eval_batches: int = 16
    relation_eval_batch_size: int = 64
    learning_rate: float = 5.0e-5
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.02
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    seed: int = 2041
    model_seed: int = 2042
    sample_bytes_per_train_source: int = 192 * 1024 * 1024
    sample_bytes_per_eval_source: int = 32 * 1024 * 1024
    sample_range_count: int = 32
    schedule_mode: str = "indexed_host"
    execution_backend: str = "inductor"
    compile_loss_tolerance: float = 1.0e-3
    minimum_control_loss_gain: float = 0.03
    maximum_gate_loss_regret: float = 0.005
    segment_length: int = 32
    memory_layer_index: int = 1
    memory_heads: int = 4
    key_width: int = 8
    value_width: int = 16
    retention_logit_bias: float = 4.0


def _data_config(
    config: SegmentAssociativeExperimentConfig,
) -> MatchedLanguageDataConfig:
    return MatchedLanguageDataConfig(
        token_budget=int(config.token_budget),
        sequence_length=int(config.sequence_length),
        batch_size=int(config.batch_size),
        eval_batches=int(config.eval_batches),
        relation_fraction=0.0,
        seed=int(config.seed),
        sample_bytes_per_train_source=int(config.sample_bytes_per_train_source),
        sample_bytes_per_eval_source=int(config.sample_bytes_per_eval_source),
        sample_range_count=int(config.sample_range_count),
        schedule_mode=str(config.schedule_mode),
    )


def _training_config(
    config: SegmentAssociativeExperimentConfig,
) -> LanguageTrainingExperimentConfig:
    return LanguageTrainingExperimentConfig(
        learning_rate=float(config.learning_rate),
        minimum_learning_rate_fraction=float(
            config.minimum_learning_rate_fraction
        ),
        warmup_fraction=float(config.warmup_fraction),
        weight_decay=float(config.weight_decay),
        precision=str(config.precision),
        execution_backend=str(config.execution_backend),
        compile_loss_tolerance=float(config.compile_loss_tolerance),
    )


def segment_associative_decision(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    requested_tokens: int,
    minimum_gain: float = 0.03,
    maximum_gate_regret: float = 0.005,
) -> str:
    if set(rows) != set(SEGMENT_ASSOCIATIVE_MODES):
        return "incomplete_v14_missing_control_arm"
    if any(
        int(row.get("processed_tokens") or 0) < int(requested_tokens)
        for row in rows.values()
    ):
        return "incomplete_v14_token_budget"
    losses = {
        name: float(row["heldout"]["heldout_loss"])
        for name, row in rows.items()
    }
    gated = losses["gated_delta"]
    if (
        losses["off"] - gated >= float(minimum_gain)
        and losses["local"] - gated >= float(minimum_gain)
        and gated - losses["delta"] <= float(maximum_gate_regret)
    ):
        return SAVE_DECISION
    if (
        losses["off"] - losses["delta"] >= float(minimum_gain)
        and losses["local"] - losses["delta"] >= float(minimum_gain)
    ):
        return "redesign_v14_gate_keep_ungated_delta_evidence"
    if gated < min(losses["off"], losses["local"]):
        return "retire_v14_weak_segment_state_gain"
    return "retire_v14_no_segment_state_gain"


def _validate_one_billion_parent(metadata: Mapping[str, Any]) -> int:
    processed_tokens = int(metadata.get("processed_tokens") or 0)
    if processed_tokens < 1_000_000_000:
        raise ValueError("V14 parent must be the one-billion-token V11 checkpoint")
    if metadata.get("external_llm_used") is not False:
        raise ValueError("V14 parent must be MARULHO-owned")
    return processed_tokens


def run_segment_associative_experiment(
    *,
    parent_checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    output_path: str | Path,
    checkpoint_output_path: str | Path | None = None,
    config: SegmentAssociativeExperimentConfig = (
        SegmentAssociativeExperimentConfig()
    ),
    device: str = "auto",
) -> dict[str, Any]:
    if int(config.token_budget) < 1:
        raise ValueError("token_budget must be positive")
    if config.schedule_mode not in {"expanded_device", "indexed_host"}:
        raise ValueError("invalid V14 schedule_mode")
    if config.execution_backend not in {"eager", "inductor"}:
        raise ValueError("invalid V14 execution_backend")
    resolved = _resolve_device(device)
    if config.execution_backend == "inductor" and resolved.type != "cuda":
        raise ValueError("Inductor V14 training is admitted only on CUDA")
    parent_path = Path(parent_checkpoint_path)
    output = Path(output_path)
    checkpoint_output = (
        None if checkpoint_output_path is None else Path(checkpoint_output_path)
    )
    if checkpoint_output is not None and (
        checkpoint_output.resolve() == parent_path.resolve()
    ):
        raise ValueError("V14 checkpoint cannot overwrite its parent")

    started = time.perf_counter()
    base_model, parent_tokenizer, parent_metadata = (
        load_hashed_micro_expert_checkpoint(parent_path, map_location="cpu")
    )
    _validate_parent(base_model, parent_metadata)
    parent_processed_tokens = _validate_one_billion_parent(parent_metadata)
    parent_sha256 = sha256_file(parent_path)
    prepared = prepare_matched_language_data(
        tokenizer_checkpoint_path=parent_path,
        relation_corpus_path=relation_corpus_path,
        relation_cases_path=relation_cases_path,
        general_train_paths=general_train_paths,
        general_eval_paths=general_eval_paths,
        config=_data_config(config),
        device=resolved,
    )
    if prepared.tokenizer.vocabulary_hash() != parent_tokenizer.vocabulary_hash():
        raise RuntimeError("V14 data tokenizer differs from parent")

    torch.manual_seed(int(config.model_seed))
    segment_config = SegmentAssociativeConfig(
        segment_length=int(config.segment_length),
        memory_layer_index=int(config.memory_layer_index),
        memory_heads=int(config.memory_heads),
        key_width=int(config.key_width),
        value_width=int(config.value_width),
        mode="gated_delta",
        retention_logit_bias=float(config.retention_logit_bias),
    )
    base_model.eval()
    model = build_segment_associative_model(base_model, segment_config).eval()
    parity_ids = prepared.eval_batches[0].input_ids[:2].detach().cpu()
    with torch.no_grad():
        base_logits = base_model(
            parity_ids,
            collect_telemetry=False,
        )["logits"]
    mode_parity: dict[str, Any] = {}
    for mode in SEGMENT_ASSOCIATIVE_MODES:
        model.set_segment_associative_mode(mode)
        with torch.no_grad():
            candidate_logits = model(
                parity_ids,
                collect_telemetry=False,
            )["logits"]
        exact = bool(torch.equal(candidate_logits, base_logits))
        delta = float((candidate_logits - base_logits).abs().max().item())
        if not exact:
            raise RuntimeError(
                f"V14 {mode} attachment changed parent logits: {delta}"
            )
        mode_parity[mode] = {
            "exact_parent_logits": exact,
            "maximum_absolute_logit_delta": delta,
        }
    del base_model

    model = model.to(resolved)
    model.set_hashed_micro_expert_mode("token_hash")
    initial_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    training_config = _training_config(config)
    warm_batch = prepared.staged.batch(0, resolved)
    previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_matmul_precision = torch.get_float32_matmul_precision()
    if resolved.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    rows: dict[str, dict[str, Any]] = {}
    try:
        model.set_segment_associative_mode("off")
        model.eval()
        heldout_before = evaluate_language_model(model, prepared.eval_batches)
        relation_before = evaluate_relation_binding_cases_batched(
            model,
            prepared.tokenizer,
            prepared.cases,
            batch_size=int(config.relation_eval_batch_size),
        )
        for mode in SEGMENT_ASSOCIATIVE_MODES:
            model.load_state_dict(initial_state, strict=True)
            model.set_hashed_micro_expert_mode("token_hash")
            model.set_segment_associative_mode(mode)
            model.train()
            training_loss, execution = _prepare_language_loss_backend(
                model,
                warm_batch,
                training_config,
            )
            row = run_matched_training_arm(
                mode,
                architecture="hashed_micro_experts_segment_associative",
                model=model,
                initial_state=initial_state,
                training_loss=training_loss,
                execution={
                    **execution,
                    "segment_associative_mode": mode,
                    "mode_specific_compiled_graph": True,
                    "inactive_modes_executed": False,
                    "training_objective": "ordinary_next_token_cross_entropy",
                    "write_policy_uses_labels": False,
                    "relation_updates_scheduled": False,
                },
                allocated_compile_seconds=float(execution["compile_seconds"]),
                prepared=prepared,
                training_config=training_config,
                gradient_clip=float(config.gradient_clip),
                precision=str(config.precision),
                relation_eval_batch_size=int(config.relation_eval_batch_size),
                model_seed=int(config.model_seed),
                device=resolved,
                progress_prefix="segment-associative-v14",
                configure_model=lambda active, selected: (
                    active.set_hashed_micro_expert_mode("token_hash"),
                    active.set_segment_associative_mode(selected),
                ),
                diagnostic_builder=lambda active, _input_ids: (
                    active.segment_diagnostic_report(_input_ids)
                ),
                extra_row={
                    "parent_checkpoint_sha256": parent_sha256,
                    "parent_processed_tokens": parent_processed_tokens,
                    "training_mixture": "general_only_equal_source_alternation",
                    "parameter_report": model.segment_parameter_report(),
                },
            )
            rows[mode] = row
            gc.collect()
            if resolved.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if resolved.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            torch.set_float32_matmul_precision(previous_matmul_precision)

    decision = segment_associative_decision(
        rows,
        requested_tokens=int(config.token_budget),
        minimum_gain=float(config.minimum_control_loss_gain),
        maximum_gate_regret=float(config.maximum_gate_loss_regret),
    )
    checkpoint_record: dict[str, Any] | None = None
    if checkpoint_output is not None:
        if decision == SAVE_DECISION:
            if model.state_block.associative._mode_name != "gated_delta":
                raise RuntimeError("Final V14 arm is not gated_delta")
            gated = rows["gated_delta"]
            saved = save_segment_associative_checkpoint(
                checkpoint_output,
                model,
                prepared.tokenizer,
                metadata={
                    "decision": decision,
                    "parent_checkpoint": str(parent_path),
                    "parent_checkpoint_sha256": parent_sha256,
                    "parent_processed_tokens": parent_processed_tokens,
                    "additional_processed_tokens": int(gated["processed_tokens"]),
                    "processed_tokens": parent_processed_tokens
                    + int(gated["processed_tokens"]),
                    "schedule_sha256": prepared.schedule_sha256,
                    "heldout_losses": {
                        name: float(row["heldout"]["heldout_loss"])
                        for name, row in rows.items()
                    },
                    "relation_updates_scheduled": False,
                    "optimizer_state_persisted": False,
                    "requires_unseen_generation": True,
                    "external_llm_used": False,
                },
            )
            checkpoint_record = {
                "path": str(saved),
                "sha256": sha256_file(saved),
                "saved": True,
                "quality_promoted": False,
                "requires_unseen_generation": True,
                "optimizer_state_persisted": False,
            }
        elif checkpoint_output.exists():
            checkpoint_output.unlink()

    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "configuration": asdict(config),
        "segment_configuration": asdict(segment_config),
        "parent": {
            "path": str(parent_path),
            "sha256": parent_sha256,
            "processed_tokens": parent_processed_tokens,
            "decision": parent_metadata["decision"],
            "tokenizer_hash": parent_tokenizer.vocabulary_hash(),
        },
        "mode_attachment_parity": mode_parity,
        "schedule": {
            "sha256": prepared.schedule_sha256,
            "step_count": int(prepared.staged.step_count),
            "mode": prepared.staged.mode,
            "resident_storage_bytes": prepared.staged.storage_bytes,
            "expanded_storage_bytes": prepared.staged.expanded_storage_bytes,
            "device_storage_bytes": prepared.staged.device_storage_bytes,
            "host_storage_bytes": prepared.staged.host_storage_bytes,
            "source_selections": prepared.source_selections,
        },
        "before": {
            "heldout": heldout_before,
            "relation": relation_before,
        },
        "arms": rows,
        "heldout_losses": {
            name: float(row["heldout"]["heldout_loss"])
            for name, row in rows.items()
        },
        "decision": decision,
        "checkpoint": checkpoint_record,
        "experiment_wall_seconds": time.perf_counter() - started,
        "promotion_boundary": {
            "base_quality_promoted": False,
            "runtime_install_allowed": False,
            "continual_memory_allowed": False,
            "requires_unseen_generation": decision == SAVE_DECISION,
        },
    }
    write_json_report_with_readme(
        output,
        report,
        title="MARULHO V14 Causal Segment Associative State",
    )
    print(
        f"[segment-associative-v14] decision {decision}; losses "
        + ", ".join(
            f"{name}={row['heldout']['heldout_loss']:.4f}"
            for name, row in rows.items()
        ),
        flush=True,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--relation-corpus", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--general-train", action="append", type=Path, required=True)
    parser.add_argument("--general-eval", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path)
    parser.add_argument("--token-budget", type=int, default=67_108_864)
    parser.add_argument("--train-sample-mib", type=int, default=192)
    parser.add_argument("--eval-sample-mib", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2041)
    parser.add_argument("--model-seed", type=int, default=2042)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--segment-length", type=int, default=32)
    parser.add_argument("--memory-heads", type=int, default=4)
    parser.add_argument("--key-width", type=int, default=8)
    parser.add_argument("--value-width", type=int, default=16)
    parser.add_argument(
        "--schedule-mode",
        choices=("expanded_device", "indexed_host"),
        default="indexed_host",
    )
    parser.add_argument(
        "--execution-backend",
        choices=("eager", "inductor"),
        default="inductor",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = SegmentAssociativeExperimentConfig(
        token_budget=int(args.token_budget),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        model_seed=int(args.model_seed),
        sample_bytes_per_train_source=int(args.train_sample_mib) * 1024 * 1024,
        sample_bytes_per_eval_source=int(args.eval_sample_mib) * 1024 * 1024,
        schedule_mode=str(args.schedule_mode),
        execution_backend=str(args.execution_backend),
        segment_length=int(args.segment_length),
        memory_heads=int(args.memory_heads),
        key_width=int(args.key_width),
        value_width=int(args.value_width),
    )
    run_segment_associative_experiment(
        parent_checkpoint_path=args.parent_checkpoint,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=tuple(args.general_train),
        general_eval_paths=tuple(args.general_eval),
        output_path=args.output,
        checkpoint_output_path=args.checkpoint_output,
        config=config,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
