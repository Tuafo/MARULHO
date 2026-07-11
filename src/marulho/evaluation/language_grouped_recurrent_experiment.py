"""Matched V17 grouped-recurrent language screen from the 1B V11 base."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import gc
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
import torch._dynamo

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
from marulho.training.language_grouped_recurrent_state import (
    GroupedRecurrentConfig,
    MarulhoGroupedRecurrentLanguageModel,
    build_grouped_recurrent_model,
)
from marulho.training.language_hashed_micro_experts import (
    load_hashed_micro_expert_checkpoint,
)
from marulho.training.language_model import evaluate_language_model


SURFACE = "marulho_grouped_recurrent_experiment.v1"
ARTIFACT_KIND = "marulho_grouped_recurrent_experiment"
ARM_NAMES = ("off", "local", "dense", "grouped")
ADVANCE_DECISION = "advance_v17_grouped_recurrence_to_67m_durability"
SCREEN_TOKEN_BUDGET = 33_554_432


@dataclass(frozen=True)
class GroupedRecurrentExperimentConfig:
    token_budget: int = SCREEN_TOKEN_BUDGET
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
    seed: int = 8401
    model_seed: int = 8402
    sample_bytes_per_train_source: int = 192 * 1024 * 1024
    sample_bytes_per_eval_source: int = 32 * 1024 * 1024
    sample_range_count: int = 32
    schedule_mode: str = "indexed_host"
    execution_backend: str = "inductor"
    compile_loss_tolerance: float = 1.0e-3
    minimum_grouped_loss_gain: float = 0.02
    maximum_relation_accuracy_regret: float = 0.02
    memory_layer_index: int = 1
    group_count: int = 8
    group_width: int = 32


def _data_config(
    config: GroupedRecurrentExperimentConfig,
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
    config: GroupedRecurrentExperimentConfig,
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


def grouped_recurrent_decision(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    requested_tokens: int,
    minimum_gain: float = 0.02,
    maximum_relation_regret: float = 0.02,
    minimum_screen_tokens: int = SCREEN_TOKEN_BUDGET,
) -> str:
    if set(rows) != set(ARM_NAMES):
        return "incomplete_v17_missing_control_arm"
    if any(
        int(row.get("processed_tokens") or 0) < int(requested_tokens)
        for row in rows.values()
    ):
        return "incomplete_v17_token_budget"
    if int(requested_tokens) < int(minimum_screen_tokens):
        return "diagnostic_v17_below_screen_budget"
    losses = {
        name: float(row["heldout"]["heldout_loss"])
        for name, row in rows.items()
    }
    relations = {
        name: float(row["relation"]["accuracy"])
        for name, row in rows.items()
    }
    grouped_loss_pass = all(
        losses[control] - losses["grouped"] >= float(minimum_gain)
        for control in ("off", "local", "dense")
    )
    relation_pass = relations["grouped"] >= max(
        relations[control] for control in ("off", "local", "dense")
    ) - float(maximum_relation_regret)
    if grouped_loss_pass and relation_pass:
        return ADVANCE_DECISION
    if grouped_loss_pass:
        return "redesign_v17_grouped_loss_gain_relation_regression"
    if (
        losses["off"] - losses["dense"] >= float(minimum_gain)
        and losses["local"] - losses["dense"] >= float(minimum_gain)
    ):
        return "redesign_v17_recurrence_gain_not_grouping"
    if (
        losses["off"] - losses["grouped"] >= float(minimum_gain)
        and losses["local"] - losses["grouped"] >= float(minimum_gain)
    ):
        return "redesign_v17_grouped_gain_not_dense_separation"
    if losses["off"] - losses["local"] >= float(minimum_gain):
        return "redesign_v17_local_capacity_gain_no_recurrence_gain"
    return "retire_v17_grouped_recurrence_no_language_gain"


def _validate_one_billion_parent(metadata: Mapping[str, Any]) -> int:
    processed_tokens = int(metadata.get("processed_tokens") or 0)
    if processed_tokens < 1_000_000_000:
        raise ValueError("V17 parent must be the one-billion-token V11 checkpoint")
    if metadata.get("external_llm_used") is not False:
        raise ValueError("V17 parent must be MARULHO-owned")
    return processed_tokens


def _arm_mode(name: str) -> str:
    return "recurrent" if name in {"grouped", "dense"} else name


def _run_arm(
    name: str,
    *,
    model: MarulhoGroupedRecurrentLanguageModel,
    initial_state: Mapping[str, torch.Tensor],
    prepared,
    training_config: LanguageTrainingExperimentConfig,
    config: GroupedRecurrentExperimentConfig,
    parent_sha256: str,
    parent_processed_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    model.load_state_dict(dict(initial_state), strict=True)
    model.set_hashed_micro_expert_mode("token_hash")
    model.set_grouped_recurrent_mode(_arm_mode(name))
    model.train()
    warm_batch = prepared.staged.batch(0, device)
    training_loss, execution = _prepare_language_loss_backend(
        model,
        warm_batch,
        training_config,
        compile_fullgraph=False,
    )
    return run_matched_training_arm(
        name,
        architecture=(
            "hashed_micro_experts_grouped_recurrent"
            if name != "dense"
            else "hashed_micro_experts_dense_recurrent"
        ),
        model=model,
        initial_state=initial_state,
        training_loss=training_loss,
        execution={
            **execution,
            "recurrent_architecture": model.state_block.recurrent.architecture,
            "recurrent_mode": _arm_mode(name),
            "mode_specific_compiled_graph": True,
            "inactive_modes_executed": False,
            "training_objective": "ordinary_next_token_cross_entropy",
            "write_policy_uses_labels": False,
            "relation_updates_scheduled": False,
            "torch_dynamo_allow_rnn": bool(torch._dynamo.config.allow_rnn),
            "partial_graph_boundary": "torch_cudnn_gru",
        },
        allocated_compile_seconds=float(execution["compile_seconds"]),
        prepared=prepared,
        training_config=training_config,
        gradient_clip=float(config.gradient_clip),
        precision=str(config.precision),
        relation_eval_batch_size=int(config.relation_eval_batch_size),
        model_seed=int(config.model_seed),
        device=device,
        progress_prefix="grouped-recurrent-v17",
        configure_model=lambda active, selected: (
            active.set_hashed_micro_expert_mode("token_hash"),
            active.set_grouped_recurrent_mode(_arm_mode(selected)),
        ),
        diagnostic_builder=lambda active, input_ids: (
            active.recurrent_diagnostic_report(input_ids)
        ),
        extra_row={
            "parent_checkpoint_sha256": parent_sha256,
            "parent_processed_tokens": parent_processed_tokens,
            "training_mixture": "general_only_equal_source_alternation",
            "parameter_report": model.recurrent_parameter_report(),
        },
    )


def run_grouped_recurrent_experiment(
    *,
    parent_checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    output_path: str | Path,
    config: GroupedRecurrentExperimentConfig = (
        GroupedRecurrentExperimentConfig()
    ),
    device: str = "auto",
) -> dict[str, Any]:
    if int(config.token_budget) < 1:
        raise ValueError("token_budget must be positive")
    if config.schedule_mode not in {"expanded_device", "indexed_host"}:
        raise ValueError("invalid V17 schedule_mode")
    if config.execution_backend not in {"eager", "inductor"}:
        raise ValueError("invalid V17 execution_backend")
    resolved = _resolve_device(device)
    if config.execution_backend == "inductor" and resolved.type != "cuda":
        raise ValueError("Inductor V17 training is admitted only on CUDA")
    parent_path = Path(parent_checkpoint_path)
    output = Path(output_path)
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
        raise RuntimeError("V17 data tokenizer differs from parent")

    grouped_config = GroupedRecurrentConfig(
        architecture="grouped",
        mode="recurrent",
        memory_layer_index=int(config.memory_layer_index),
        group_count=int(config.group_count),
        group_width=int(config.group_width),
    )
    dense_config = replace(grouped_config, architecture="dense")
    torch.manual_seed(int(config.model_seed))
    base_model.eval()
    grouped_model = build_grouped_recurrent_model(base_model, grouped_config).eval()
    torch.manual_seed(int(config.model_seed) + 1)
    dense_model = build_grouped_recurrent_model(base_model, dense_config).eval()
    parity_ids = prepared.eval_batches[0].input_ids[:2].detach().cpu()
    with torch.no_grad():
        base_logits = base_model(parity_ids, collect_telemetry=False)["logits"]
    attachment_parity: dict[str, Any] = {}
    for name in ("off", "local", "grouped"):
        grouped_model.set_grouped_recurrent_mode(_arm_mode(name))
        with torch.no_grad():
            logits = grouped_model(parity_ids, collect_telemetry=False)["logits"]
        exact = bool(torch.equal(logits, base_logits))
        delta = float((logits - base_logits).abs().max())
        if not exact:
            raise RuntimeError(f"V17 {name} attachment changed parent logits: {delta}")
        attachment_parity[name] = {
            "exact_parent_logits": exact,
            "maximum_absolute_logit_delta": delta,
        }
    dense_model.set_grouped_recurrent_mode("recurrent")
    with torch.no_grad():
        dense_logits = dense_model(parity_ids, collect_telemetry=False)["logits"]
    dense_exact = bool(torch.equal(dense_logits, base_logits))
    dense_delta = float((dense_logits - base_logits).abs().max())
    if not dense_exact:
        raise RuntimeError(f"V17 dense attachment changed parent logits: {dense_delta}")
    attachment_parity["dense"] = {
        "exact_parent_logits": dense_exact,
        "maximum_absolute_logit_delta": dense_delta,
    }
    del base_model

    training_config = _training_config(config)
    previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_matmul_precision = torch.get_float32_matmul_precision()
    previous_allow_rnn = bool(torch._dynamo.config.allow_rnn)
    if resolved.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if config.execution_backend == "inductor":
        torch._dynamo.config.allow_rnn = False
    rows: dict[str, dict[str, Any]] = {}
    try:
        grouped_model = grouped_model.to(resolved)
        grouped_model.set_hashed_micro_expert_mode("token_hash")
        grouped_initial = {
            name: value.detach().clone()
            for name, value in grouped_model.state_dict().items()
        }
        grouped_model.set_grouped_recurrent_mode("off")
        grouped_model.eval()
        heldout_before = evaluate_language_model(
            grouped_model, prepared.eval_batches
        )
        relation_before = evaluate_relation_binding_cases_batched(
            grouped_model,
            prepared.tokenizer,
            prepared.cases,
            batch_size=int(config.relation_eval_batch_size),
        )
        for name in ("off", "local", "grouped"):
            rows[name] = _run_arm(
                name,
                model=grouped_model,
                initial_state=grouped_initial,
                prepared=prepared,
                training_config=training_config,
                config=config,
                parent_sha256=parent_sha256,
                parent_processed_tokens=parent_processed_tokens,
                device=resolved,
            )
            gc.collect()
            if resolved.type == "cuda":
                torch.cuda.empty_cache()
        grouped_model = grouped_model.cpu()
        del grouped_model, grouped_initial
        gc.collect()
        if resolved.type == "cuda":
            torch.cuda.empty_cache()

        dense_model = dense_model.to(resolved)
        dense_model.set_hashed_micro_expert_mode("token_hash")
        dense_model.set_grouped_recurrent_mode("recurrent")
        dense_initial = {
            name: value.detach().clone()
            for name, value in dense_model.state_dict().items()
        }
        rows["dense"] = _run_arm(
            "dense",
            model=dense_model,
            initial_state=dense_initial,
            prepared=prepared,
            training_config=training_config,
            config=config,
            parent_sha256=parent_sha256,
            parent_processed_tokens=parent_processed_tokens,
            device=resolved,
        )
    finally:
        if resolved.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            torch.set_float32_matmul_precision(previous_matmul_precision)
        torch._dynamo.config.allow_rnn = previous_allow_rnn

    decision = grouped_recurrent_decision(
        rows,
        requested_tokens=int(config.token_budget),
        minimum_gain=float(config.minimum_grouped_loss_gain),
        maximum_relation_regret=float(config.maximum_relation_accuracy_regret),
        minimum_screen_tokens=SCREEN_TOKEN_BUDGET,
    )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "configuration": asdict(config),
        "grouped_configuration": asdict(grouped_config),
        "dense_configuration": asdict(dense_config),
        "parent": {
            "path": str(parent_path),
            "sha256": parent_sha256,
            "processed_tokens": parent_processed_tokens,
            "decision": parent_metadata["decision"],
            "tokenizer_hash": parent_tokenizer.vocabulary_hash(),
        },
        "attachment_parity": attachment_parity,
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
        "checkpoint": None,
        "experiment_wall_seconds": time.perf_counter() - started,
        "promotion_boundary": {
            "advance_to_durability": decision == ADVANCE_DECISION,
            "base_quality_promoted": False,
            "runtime_install_allowed": False,
            "continual_memory_allowed": False,
            "synthetic_result_used_as_language_evidence": False,
        },
        "research_basis": [
            "https://arxiv.org/abs/1909.10893",
            "https://arxiv.org/abs/2406.12272",
            "https://arxiv.org/abs/2602.12021",
            "https://arxiv.org/abs/2203.07852",
        ],
    }
    write_json_report_with_readme(
        output,
        report,
        title="MARULHO V17 Grouped Recurrent Language Screen",
    )
    print(
        f"[grouped-recurrent-v17] decision {decision}; losses "
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
    parser.add_argument("--token-budget", type=int, default=SCREEN_TOKEN_BUDGET)
    parser.add_argument("--train-sample-mib", type=int, default=192)
    parser.add_argument("--eval-sample-mib", type=int, default=32)
    parser.add_argument("--seed", type=int, default=8401)
    parser.add_argument("--model-seed", type=int, default=8402)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--group-count", type=int, default=8)
    parser.add_argument("--group-width", type=int, default=32)
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
    config = GroupedRecurrentExperimentConfig(
        token_budget=int(args.token_budget),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        model_seed=int(args.model_seed),
        sample_bytes_per_train_source=int(args.train_sample_mib) * 1024 * 1024,
        sample_bytes_per_eval_source=int(args.eval_sample_mib) * 1024 * 1024,
        schedule_mode=str(args.schedule_mode),
        execution_backend=str(args.execution_backend),
        group_count=int(args.group_count),
        group_width=int(args.group_width),
    )
    run_grouped_recurrent_experiment(
        parent_checkpoint_path=args.parent_checkpoint,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=tuple(args.general_train),
        general_eval_paths=tuple(args.general_eval),
        output_path=args.output,
        config=config,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
