"""Matched falsification for gated multiscale dynamical memory v7."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from marulho.data.language_tokenizer import (
    LANGUAGE_DOCUMENT_SEPARATOR,
    load_language_tokenizer_state,
)
from marulho.evaluation.language_relation_binding_experiment import (
    RelationCase,
    evaluate_relation_binding_cases_batched,
)
from marulho.evaluation.language_training_experiment import (
    LanguageTrainingExperimentConfig,
    _learning_rate,
    _optimizer,
    _prepare_language_loss_backend,
    _precision_context,
    _resolve_device,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_dynamical_memory import (
    DynamicalMemoryConfig,
    MarulhoDynamicalMemoryLanguageModel,
)
from marulho.training.language_model import (
    LanguageBatch,
    LanguageModelConfig,
    MarulhoLanguageModel,
    build_language_model_splits,
    evaluate_language_model,
)


SURFACE = "marulho_dynamical_memory_falsification.v1"
ARTIFACT_KIND = "marulho_dynamical_memory_falsification"
ARCHITECTURES = ("transformer", "dynamical_memory")
ARM_NAMES = (
    "transformer",
    "memory_off",
    "single_scale",
    "multiscale_always",
    "multiscale_random",
    "multiscale_learned",
)
MEMORY_CONTROL_NAMES = ARM_NAMES[1:]


@dataclass(frozen=True)
class DynamicalMemoryFalsificationConfig:
    token_budget: int = 16_777_216
    sequence_length: int = 72
    batch_size: int = 144
    eval_batches: int = 16
    relation_eval_batch_size: int = 64
    relation_fraction: float = 0.20
    learning_rate: float = 3.0e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    seed: int = 1337
    model_seed: int = 1337
    sample_bytes_per_train_source: int = 64 * 1024 * 1024
    sample_bytes_per_eval_source: int = 32 * 1024 * 1024
    sample_range_count: int = 16
    width: int = 512
    layers: int = 4
    attention_heads: int = 8
    baseline_hidden_width: int = 2048
    candidate_hidden_width: int = 1920
    memory_after_layer: int = 2
    memory_bank_count: int = 4
    memory_bank_width: int = 128
    memory_decays: tuple[float, ...] = (0.50, 0.875, 0.96875, 0.9921875)
    single_scale_decay: float = 0.875
    execution_backend: str = "eager"
    compile_loss_tolerance: float = 1.0e-3


@dataclass(frozen=True)
class StagedSchedule:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    elapsed_seconds: float
    storage_bytes: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schedule_hash(schedule: Sequence[tuple[str, int]]) -> str:
    payload = json.dumps(list(schedule), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sample_corpus_ranges(
    path: str | Path,
    *,
    byte_budget: int,
    range_count: int,
) -> tuple[str, dict[str, Any]]:
    source = Path(path)
    size = source.stat().st_size
    budget = max(1, min(int(byte_budget), size))
    if budget >= size:
        data = source.read_bytes()
        return data.decode("utf-8"), {
            "path": str(source),
            "source_size_bytes": size,
            "selected_size_bytes": len(data),
            "selected_sha256": hashlib.sha256(data).hexdigest(),
            "ranges": [{"start": 0, "end": size}],
        }
    count = max(1, int(range_count))
    chunk_size = max(1, budget // count)
    maximum_start = max(0, size - chunk_size)
    starts = [
        round(index * maximum_start / max(1, count - 1))
        for index in range(count)
    ]
    chunks: list[bytes] = []
    ranges: list[dict[str, int]] = []
    with source.open("rb") as handle:
        for nominal in starts:
            handle.seek(int(nominal))
            if nominal > 0:
                handle.readline()
            start = handle.tell()
            data = handle.read(chunk_size)
            data += handle.readline()
            if data:
                chunks.append(data)
                ranges.append({"start": int(start), "end": int(handle.tell())})
    selected = f"\n{LANGUAGE_DOCUMENT_SEPARATOR}\n".encode("utf-8").join(chunks)
    return selected.decode("utf-8"), {
        "path": str(source),
        "source_size_bytes": size,
        "selected_size_bytes": len(selected),
        "selected_sha256": hashlib.sha256(selected).hexdigest(),
        "ranges": ranges,
    }


def build_matched_schedule(
    *,
    step_count: int,
    relation_fraction: float,
    relation_batch_count: int,
    general_batch_counts: Sequence[int],
    seed: int,
) -> tuple[tuple[str, int], ...]:
    if int(step_count) < 1:
        raise ValueError("step_count must be positive")
    if int(relation_batch_count) < 1 or any(
        int(count) < 1 for count in general_batch_counts
    ):
        raise ValueError("Every scheduled source requires at least one batch")
    if not general_batch_counts:
        raise ValueError("At least one general source is required")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    orders = {
        "relation": torch.randperm(
            relation_batch_count,
            generator=generator,
        ).tolist(),
        **{
            f"general_{index}": torch.randperm(
                count,
                generator=generator,
            ).tolist()
            for index, count in enumerate(general_batch_counts)
        },
    }
    cursors = {name: 0 for name in orders}
    accumulator = 0.0
    source_cursor = 0
    schedule: list[tuple[str, int]] = []
    for _ in range(int(step_count)):
        accumulator += float(relation_fraction)
        if accumulator >= 1.0:
            accumulator -= 1.0
            kind = "relation"
        else:
            kind = f"general_{source_cursor % len(general_batch_counts)}"
            source_cursor += 1
        order = orders[kind]
        cursor = cursors[kind]
        if cursor >= len(order):
            order = torch.randperm(len(order), generator=generator).tolist()
            orders[kind] = order
            cursor = 0
        schedule.append((kind, int(order[cursor])))
        cursors[kind] = cursor + 1
    return tuple(schedule)


def _load_tokenizer(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("surface") != "marulho_transformer_language_checkpoint.v2":
        raise ValueError("Tokenizer source must be a Transformer checkpoint")
    return load_language_tokenizer_state(payload["tokenizer"])


def _load_cases(path: Path) -> tuple[RelationCase, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        RelationCase(
            case_id=str(row["case_id"]),
            kind=str(row["kind"]),
            signature=str(row["signature"]),
            prompt=str(row["prompt"]),
            candidates=tuple(str(value) for value in row["candidates"]),
            correct_index=int(row["correct_index"]),
        )
        for row in payload["cases"]
    )


def _architecture(name: str) -> str:
    return "transformer" if name == "transformer" else "dynamical_memory"


def _build_model(
    architecture: str,
    *,
    vocab_size: int,
    config: DynamicalMemoryFalsificationConfig,
) -> MarulhoLanguageModel | MarulhoDynamicalMemoryLanguageModel:
    if architecture == "transformer":
        return MarulhoLanguageModel(
            LanguageModelConfig(
                vocab_size=int(vocab_size),
                embedding_dim=config.width,
                state_dim=config.width,
                state_layers=config.layers,
                attention_heads=config.attention_heads,
                transformer_context_length=config.sequence_length,
                transformer_mlp_ratio=(
                    float(config.baseline_hidden_width) / float(config.width)
                ),
                tie_embeddings=True,
                active_language_path="marulho_transformer_v7_control",
            )
        )
    if architecture == "dynamical_memory":
        return MarulhoDynamicalMemoryLanguageModel(
            DynamicalMemoryConfig(
                vocab_size=int(vocab_size),
                width=config.width,
                layers=config.layers,
                attention_heads=config.attention_heads,
                hidden_width=config.candidate_hidden_width,
                context_length=config.sequence_length,
                memory_after_layer=config.memory_after_layer,
                memory_bank_count=config.memory_bank_count,
                memory_bank_width=config.memory_bank_width,
                memory_decays=config.memory_decays,
                single_scale_decay=config.single_scale_decay,
                mode="multiscale_learned",
            )
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def _training_config(
    config: DynamicalMemoryFalsificationConfig,
) -> LanguageTrainingExperimentConfig:
    return LanguageTrainingExperimentConfig(
        learning_rate=config.learning_rate,
        minimum_learning_rate_fraction=config.minimum_learning_rate_fraction,
        warmup_fraction=config.warmup_fraction,
        weight_decay=config.weight_decay,
        precision=config.precision,
        execution_backend=config.execution_backend,
        compile_loss_tolerance=config.compile_loss_tolerance,
    )


def _selected_batch(
    kind: str,
    index: int,
    *,
    relation_batches: Sequence[LanguageBatch],
    general_batches: Sequence[Sequence[LanguageBatch]],
) -> LanguageBatch:
    if kind == "relation":
        return relation_batches[index]
    return general_batches[int(kind.rsplit("_", 1)[1])][index]


def _stage_schedule(
    schedule: Sequence[tuple[str, int]],
    *,
    relation_batches: Sequence[LanguageBatch],
    general_batches: Sequence[Sequence[LanguageBatch]],
    device: torch.device,
) -> StagedSchedule:
    selected = [
        _selected_batch(
            kind,
            index,
            relation_batches=relation_batches,
            general_batches=general_batches,
        )
        for kind, index in schedule
    ]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    input_ids = torch.stack([batch.input_ids for batch in selected]).to(device)
    target_ids = torch.stack([batch.target_ids for batch in selected]).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return StagedSchedule(
        input_ids=input_ids,
        target_ids=target_ids,
        elapsed_seconds=elapsed,
        storage_bytes=int(
            input_ids.numel() * input_ids.element_size()
            + target_ids.numel() * target_ids.element_size()
        ),
    )


def _run_arm(
    name: str,
    *,
    model: MarulhoLanguageModel | MarulhoDynamicalMemoryLanguageModel,
    initial_state: Mapping[str, torch.Tensor],
    training_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    shared_execution: Mapping[str, Any],
    allocated_compile_seconds: float,
    tokenizer,
    staged: StagedSchedule,
    eval_batches: Sequence[LanguageBatch],
    cases: Sequence[RelationCase],
    config: DynamicalMemoryFalsificationConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.load_state_dict(initial_state, strict=True)
    if isinstance(model, MarulhoDynamicalMemoryLanguageModel):
        model.set_memory_mode(name)
    model.zero_grad(set_to_none=True)
    torch.manual_seed(config.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.model_seed)
    training_config = _training_config(config)
    optimizer, fused = _optimizer(model, training_config)
    total_steps = int(staged.input_ids.shape[0])
    warmup_steps = int(
        round(total_steps * max(0.0, float(training_config.warmup_fraction)))
    )
    trace_steps = {
        max(0, min(total_steps - 1, math.ceil(total_steps * fraction / 10) - 1))
        for fraction in range(1, 11)
    }
    trace_tensors: list[tuple[int, torch.Tensor]] = []
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    arm_started = time.perf_counter()
    training_started = arm_started
    final_loss: torch.Tensor | None = None
    for step in range(total_steps):
        learning_rate = _learning_rate(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            peak=float(training_config.learning_rate),
            minimum_fraction=float(
                training_config.minimum_learning_rate_fraction
            ),
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with _precision_context(device, config.precision):
            final_loss = training_loss(
                staged.input_ids[step],
                staged.target_ids[step],
            )
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip,
        )
        optimizer.step()
        if step in trace_steps:
            trace_tensors.append((step + 1, final_loss.detach().float()))
        if (step + 1) % max(1, total_steps // 10) == 0:
            print(
                f"[dynamical-memory-v7] {name} {step + 1}/{total_steps}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_elapsed = time.perf_counter() - training_started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    if final_loss is None:
        raise RuntimeError("Training schedule produced no steps")

    gradient_counts = [
        torch.count_nonzero(parameter.grad)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    nonzero_gradient_elements = (
        int(torch.stack(gradient_counts).sum().detach().cpu())
        if gradient_counts
        else 0
    )
    parameters_with_gradient = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    model.eval()
    evaluation_started = time.perf_counter()
    heldout = evaluate_language_model(model, eval_batches)
    relation = evaluate_relation_binding_cases_batched(
        model,
        tokenizer,
        cases,
        batch_size=config.relation_eval_batch_size,
    )
    sample = eval_batches[0].to(device)
    with torch.no_grad(), _precision_context(device, config.precision):
        sample_output = model(
            sample.input_ids,
            collect_telemetry=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_elapsed = time.perf_counter() - evaluation_started
    state_bytes = sum(
        int(value.numel() * value.element_size())
        for value in sample_output["state"].values()
        if isinstance(value, torch.Tensor)
    )
    processed = total_steps * int(staged.target_ids[0].numel())
    end_to_end = training_elapsed + float(allocated_compile_seconds)
    loss_trace = [
        {
            "step": step,
            "processed_tokens": step * int(staged.target_ids[0].numel()),
            "training_batch_loss": float(value.cpu()),
        }
        for step, value in trace_tensors
    ]
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "name": name,
        "architecture": _architecture(name),
        "parameters": total_parameters,
        "parameters_with_final_gradient": parameters_with_gradient,
        "nonzero_final_gradient_elements": nonzero_gradient_elements,
        "all_parameters_received_final_gradient": (
            parameters_with_gradient == total_parameters
        ),
        "optimizer_fused": bool(fused),
        "optimizer_state_fresh_for_arm": True,
        "initial_weights_restored_for_arm": True,
        "processed_tokens": processed,
        "training": {
            "final_batch_loss": float(final_loss.detach().float().cpu()),
            "loss_trace": loss_trace,
            "optimizer_step_count": total_steps,
            "optimizer_step_seconds": training_elapsed,
            "tokens_per_second": processed / max(training_elapsed, 1.0e-9),
            "shared_architecture_compile_seconds_allocated": float(
                allocated_compile_seconds
            ),
            "end_to_end_seconds_including_allocated_compile": end_to_end,
            "amortized_tokens_per_second_including_allocated_compile": (
                processed / max(end_to_end, 1.0e-9)
            ),
            "evaluation_seconds": evaluation_elapsed,
            "arm_wall_seconds_excluding_shared_compile": (
                time.perf_counter() - arm_started
            ),
            "peak_cuda_memory_bytes": peak_memory,
            "per_step_host_metric_readback": False,
            "scheduled_batches_pre_staged_on_device": True,
        },
        "execution": {
            **dict(shared_execution),
            "loss_graph_shared_between_memory_controls": (
                _architecture(name) == "dynamical_memory"
            ),
        },
        "heldout": heldout,
        "relation": relation,
        "runtime_state_bytes_for_training_batch": state_bytes,
        "telemetry": sample_output["telemetry"],
    }


def dynamical_memory_decision(
    arms: Sequence[Mapping[str, Any]],
    *,
    minimum_tokens: int = 16_777_216,
) -> str:
    rows = {str(row["name"]): row for row in arms}
    if any(name not in rows for name in ARM_NAMES):
        return "incomplete_v7_memory_control_comparison"
    processed = min(int(row.get("processed_tokens") or 0) for row in rows.values())
    if processed < int(minimum_tokens):
        return "incomplete_v7_mechanism_smoke"

    def loss(name: str) -> float:
        return float(rows[name]["heldout"]["heldout_loss"])

    def free(name: str) -> float:
        return float(rows[name]["relation"]["generation_exact_accuracy"])

    def qualified(candidate: str, control: str) -> bool:
        return (
            loss(candidate) <= loss(control) - 0.005
            and free(candidate) >= free(control) + 0.02
        )

    qualified_replacements = [
        name for name in MEMORY_CONTROL_NAMES if qualified(name, "transformer")
    ]
    learned_controls = (
        "memory_off",
        "single_scale",
        "multiscale_always",
        "multiscale_random",
    )
    learned_mechanism_win = (
        "multiscale_learned" in qualified_replacements
        and all(
            qualified("multiscale_learned", control)
            for control in learned_controls
        )
    )
    if learned_mechanism_win:
        return "replicate_v7_multiscale_learned_before_scale"
    if qualified_replacements:
        best = min(
            qualified_replacements,
            key=lambda name: (loss(name), -free(name)),
        )
        if best == "memory_off":
            return "replicate_v7_reduced_mlp_control_before_scale"
        return f"replicate_v7_{best}_without_learned_gate_claim"

    best_memory_free = max(free(name) for name in MEMORY_CONTROL_NAMES)
    best_memory_loss = min(loss(name) for name in MEMORY_CONTROL_NAMES)
    if (
        best_memory_free >= free("transformer") + 0.02
        and best_memory_loss > loss("transformer") - 0.005
    ):
        return "redesign_v7_behavior_signal_without_loss_gain"
    return "retire_v7_no_quality_or_control_gain"


def _assemble_report(
    *,
    config: DynamicalMemoryFalsificationConfig,
    tokenizer_checkpoint: Path,
    tokenizer,
    relation_cases_file: Path,
    cases: Sequence[RelationCase],
    source_selections: Mapping[str, Any],
    schedule: Sequence[tuple[str, int]],
    schedule_digest: str,
    staged: StagedSchedule,
    combined: Mapping[str, Mapping[str, Any]],
    architecture_runs: Mapping[str, Mapping[str, Any]],
    executed_names: Sequence[str],
    baseline_report_path: str | Path | None,
    baseline_report_sha256: str | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    arms = [dict(combined[name]) for name in ARM_NAMES if name in combined]
    decision = dynamical_memory_decision(arms)
    counts = {row["name"]: int(row["parameters"]) for row in arms}
    transformer_count = counts.get("transformer")
    candidate_count = next(
        (
            int(row["parameters"])
            for row in arms
            if row["architecture"] == "dynamical_memory"
        ),
        None,
    )
    parameter_delta = (
        (candidate_count - transformer_count) / transformer_count
        if transformer_count and candidate_count
        else None
    )
    compile_seconds = sum(
        float(row.get("compile_seconds_total") or 0.0)
        for row in architecture_runs.values()
        if row.get("executed_this_run")
    )
    requested_compile_count_without_reuse = sum(
        len(row.get("arms_sharing_graph") or [])
        for row in architecture_runs.values()
        if row.get("executed_this_run")
    )
    actual_compile_count = sum(
        bool(row.get("executed_this_run"))
        for row in architecture_runs.values()
    )
    candidate_rows = [
        row for row in arms if row["architecture"] == "dynamical_memory"
    ]
    candidate_rates = [
        float(row["training"]["tokens_per_second"]) for row in candidate_rows
    ]
    return {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "configuration": asdict(config),
        "tokenizer": {
            "checkpoint": str(tokenizer_checkpoint),
            "checkpoint_sha256": _sha256_file(tokenizer_checkpoint),
            "vocab_size": tokenizer.vocab_size,
            "hash": tokenizer.vocabulary_hash(),
        },
        "relation_cases": {
            "path": str(relation_cases_file),
            "sha256": _sha256_file(relation_cases_file),
            "count": len(cases),
            "labels_metrics_only": True,
        },
        "source_selections": dict(source_selections),
        "schedule": {
            "steps": len(schedule),
            "processed_tokens": (
                len(schedule) * config.batch_size * config.sequence_length
            ),
            "relation_steps": sum(
                kind == "relation" for kind, _index in schedule
            ),
            "sha256": schedule_digest,
            "identical_for_all_arms": True,
            "staged_once_on_device": True,
            "staging_seconds": staged.elapsed_seconds,
            "staging_storage_bytes": staged.storage_bytes,
        },
        "parameter_match": {
            "counts": counts,
            "candidate_minus_transformer_fraction": parameter_delta,
            "within_one_tenth_percent": (
                abs(parameter_delta) < 0.001
                if parameter_delta is not None
                else None
            ),
        },
        "compile_reuse": {
            "architecture_runs": dict(architecture_runs),
            "loss_graph_compile_count_actual": actual_compile_count,
            "loss_graph_compile_count_without_reuse": (
                requested_compile_count_without_reuse
            ),
            "loss_graph_compiles_avoided": max(
                0,
                requested_compile_count_without_reuse - actual_compile_count,
            ),
            "compile_seconds_this_run": compile_seconds,
            "allocated_evenly_across_same_architecture_arms": True,
        },
        "memory_control_compute": {
            "same_parameter_objects": True,
            "same_compiled_loss_graph": True,
            "mode_selected_by_mutable_tensor_buffers": True,
            "steady_tokens_per_second_min": (
                min(candidate_rates) if candidate_rates else None
            ),
            "steady_tokens_per_second_max": (
                max(candidate_rates) if candidate_rates else None
            ),
            "max_to_min_ratio": (
                max(candidate_rates) / min(candidate_rates)
                if candidate_rates
                else None
            ),
        },
        "arms": arms,
        "arms_executed_this_run": list(executed_names),
        "reused_control_report": (
            None
            if baseline_report_path is None
            else {
                "path": str(baseline_report_path),
                "sha256_before_run": baseline_report_sha256,
            }
        ),
        "experiment_wall_seconds_this_run": elapsed_seconds,
        "decision": decision,
        "decision_contract": {
            "heldout_loss_margin": 0.005,
            "free_relation_margin": 0.02,
            "learned_multiscale_must_beat": [
                "transformer",
                "memory_off",
                "single_scale",
                "multiscale_always",
                "multiscale_random",
            ],
            "positive_requires_replication_before_scale": True,
            "no_checkpoint_saved_before_survival": True,
            "behavior_only_win_routes_to_redesign_not_scale": True,
        },
        "quality_boundary": {
            "promotes_runtime_installation": False,
            "promotes_unseen_generation": False,
            "throughput_is_not_quality": True,
        },
    }


def run_dynamical_memory_falsification(
    *,
    tokenizer_checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    output_path: str | Path,
    config: DynamicalMemoryFalsificationConfig = (
        DynamicalMemoryFalsificationConfig()
    ),
    device: str = "auto",
    arm_names: Sequence[str] = ARM_NAMES,
    baseline_report_path: str | Path | None = None,
) -> dict[str, Any]:
    if len(general_train_paths) != 2 or len(general_eval_paths) != 2:
        raise ValueError("Exactly two general train and two eval sources are required")
    if config.execution_backend not in {"eager", "inductor"}:
        raise ValueError("execution_backend must be 'eager' or 'inductor'")
    if config.compile_loss_tolerance <= 0.0:
        raise ValueError("compile_loss_tolerance must be positive")
    resolved = _resolve_device(device)
    if config.execution_backend == "inductor" and resolved.type != "cuda":
        raise ValueError("Inductor v7 execution is admitted only for CUDA runs")
    requested_arms = tuple(dict.fromkeys(str(name) for name in arm_names))
    if not requested_arms or any(name not in ARM_NAMES for name in requested_arms):
        raise ValueError("arm_names must contain valid unique v7 arm names")

    run_started = time.perf_counter()
    tokenizer_checkpoint = Path(tokenizer_checkpoint_path)
    relation_cases_file = Path(relation_cases_path)
    tokenizer = _load_tokenizer(tokenizer_checkpoint)
    cases = _load_cases(relation_cases_file)
    steps = math.ceil(
        config.token_budget / (config.batch_size * config.sequence_length)
    )
    relation_steps = max(1, int(round(steps * config.relation_fraction)))
    general_steps = max(1, math.ceil((steps - relation_steps) / 2))
    relation_text, relation_selection = sample_corpus_ranges(
        relation_corpus_path,
        byte_budget=config.sample_bytes_per_train_source,
        range_count=config.sample_range_count,
    )
    train_samples = [
        sample_corpus_ranges(
            path,
            byte_budget=config.sample_bytes_per_train_source,
            range_count=config.sample_range_count,
        )
        for path in general_train_paths
    ]
    eval_samples = [
        sample_corpus_ranges(
            path,
            byte_budget=config.sample_bytes_per_eval_source,
            range_count=config.sample_range_count,
        )
        for path in general_eval_paths
    ]
    relation_split = build_language_model_splits(
        [relation_text],
        tokenizer,
        sequence_length=config.sequence_length,
        stride=config.sequence_length,
        batch_size=config.batch_size,
        max_train_batches=relation_steps,
        max_eval_batches=1,
    )
    general_splits = [
        build_language_model_splits(
            [text],
            tokenizer,
            sequence_length=config.sequence_length,
            stride=config.sequence_length,
            batch_size=config.batch_size,
            max_train_batches=general_steps,
            max_eval_batches=1,
        )
        for text, _selection in train_samples
    ]
    eval_split = build_language_model_splits(
        [text for text, _selection in train_samples],
        tokenizer,
        eval_texts=[text for text, _selection in eval_samples],
        sequence_length=config.sequence_length,
        stride=config.sequence_length,
        batch_size=config.batch_size,
        max_train_batches=1,
        max_eval_batches=config.eval_batches,
    )
    schedule = build_matched_schedule(
        step_count=steps,
        relation_fraction=config.relation_fraction,
        relation_batch_count=len(relation_split.train),
        general_batch_counts=[len(split.train) for split in general_splits],
        seed=config.seed,
    )
    schedule_digest = _schedule_hash(schedule)
    source_selections = {
        "relation": relation_selection,
        "general_train": [row for _text, row in train_samples],
        "general_eval": [row for _text, row in eval_samples],
    }
    staged = _stage_schedule(
        schedule,
        relation_batches=relation_split.train,
        general_batches=[split.train for split in general_splits],
        device=resolved,
    )

    baseline_report_sha256: str | None = None
    combined: dict[str, Mapping[str, Any]] = {}
    architecture_runs: dict[str, Mapping[str, Any]] = {}
    if baseline_report_path is not None:
        baseline_path = Path(baseline_report_path)
        baseline_report_sha256 = _sha256_file(baseline_path)
        reused = json.loads(baseline_path.read_text(encoding="utf-8"))
        if reused.get("surface") != SURFACE:
            raise ValueError("Baseline report surface does not match v7 runner")
        if dict(reused.get("configuration") or {}) != asdict(config):
            raise ValueError("Baseline report configuration does not match this run")
        if dict(reused.get("source_selections") or {}) != source_selections:
            raise ValueError("Baseline report source selections do not match this run")
        if str(reused["tokenizer"]["hash"]) != tokenizer.vocabulary_hash():
            raise ValueError("Baseline report tokenizer does not match this run")
        if str(reused["schedule"]["sha256"]) != schedule_digest:
            raise ValueError("Baseline report schedule does not match this run")
        combined = {str(row["name"]): row for row in reused["arms"]}
        architecture_runs = {
            str(name): {
                **dict(row),
                "executed_this_run": False,
            }
            for name, row in dict(
                (reused.get("compile_reuse") or {}).get("architecture_runs")
                or {}
            ).items()
        }

    executed_names: list[str] = []
    output = Path(output_path)
    previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_matmul_precision = torch.get_float32_matmul_precision()
    if resolved.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    try:
        for architecture in ARCHITECTURES:
            group_arms = [
                name
                for name in requested_arms
                if _architecture(name) == architecture and name not in combined
            ]
            if not group_arms:
                continue
            print(
                f"[dynamical-memory-v7] preparing shared {architecture} graph",
                flush=True,
            )
            torch.manual_seed(config.model_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.model_seed)
            model = _build_model(
                architecture,
                vocab_size=tokenizer.vocab_size,
                config=config,
            )
            initial_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            model = model.to(resolved)
            training_config = _training_config(config)
            warm_batch = LanguageBatch(
                staged.input_ids[0],
                staged.target_ids[0],
            )
            model.train()
            training_loss, execution = _prepare_language_loss_backend(
                model,
                warm_batch,
                training_config,
            )
            compile_seconds = float(execution["compile_seconds"])
            model.eval()
            initial_heldout = evaluate_language_model(model, eval_split.eval)
            architecture_runs[architecture] = {
                "executed_this_run": True,
                "arms_sharing_graph": list(group_arms),
                "same_model_object_reloaded_from_exact_initial_state": True,
                "fresh_optimizer_per_arm": True,
                "loss_execution": execution,
                "compile_seconds_total": compile_seconds,
                "initial_heldout": initial_heldout,
            }
            allocated_compile = compile_seconds / max(1, len(group_arms))
            for name in group_arms:
                print(f"[dynamical-memory-v7] starting {name}", flush=True)
                row = _run_arm(
                    name,
                    model=model,
                    initial_state=initial_state,
                    training_loss=training_loss,
                    shared_execution=execution,
                    allocated_compile_seconds=allocated_compile,
                    tokenizer=tokenizer,
                    staged=staged,
                    eval_batches=eval_split.eval,
                    cases=cases,
                    config=config,
                    device=resolved,
                )
                combined[name] = row
                executed_names.append(name)
                print(
                    f"[dynamical-memory-v7] completed {name}: loss "
                    f"{row['heldout']['heldout_loss']:.4f}, free "
                    f"{row['relation']['generation_exact_accuracy']:.3f}",
                    flush=True,
                )
                partial = _assemble_report(
                    config=config,
                    tokenizer_checkpoint=tokenizer_checkpoint,
                    tokenizer=tokenizer,
                    relation_cases_file=relation_cases_file,
                    cases=cases,
                    source_selections=source_selections,
                    schedule=schedule,
                    schedule_digest=schedule_digest,
                    staged=staged,
                    combined=combined,
                    architecture_runs=architecture_runs,
                    executed_names=executed_names,
                    baseline_report_path=baseline_report_path,
                    baseline_report_sha256=baseline_report_sha256,
                    elapsed_seconds=time.perf_counter() - run_started,
                )
                write_json_report_with_readme(
                    output,
                    partial,
                    title="MARULHO Dynamical Memory v7 Falsification",
                )
            del training_loss, model
            gc.collect()
            if resolved.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if resolved.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            torch.set_float32_matmul_precision(previous_matmul_precision)

    report = _assemble_report(
        config=config,
        tokenizer_checkpoint=tokenizer_checkpoint,
        tokenizer=tokenizer,
        relation_cases_file=relation_cases_file,
        cases=cases,
        source_selections=source_selections,
        schedule=schedule,
        schedule_digest=schedule_digest,
        staged=staged,
        combined=combined,
        architecture_runs=architecture_runs,
        executed_names=executed_names,
        baseline_report_path=baseline_report_path,
        baseline_report_sha256=baseline_report_sha256,
        elapsed_seconds=time.perf_counter() - run_started,
    )
    write_json_report_with_readme(
        output,
        report,
        title="MARULHO Dynamical Memory v7 Falsification",
    )
    print(f"[dynamical-memory-v7] decision {report['decision']}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--relation-corpus", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--general-train", action="append", type=Path, required=True)
    parser.add_argument("--general-eval", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=16_777_216)
    parser.add_argument("--train-sample-mib", type=int, default=64)
    parser.add_argument("--eval-sample-mib", type=int, default=32)
    parser.add_argument(
        "--arm",
        action="append",
        choices=ARM_NAMES,
        default=[],
    )
    parser.add_argument(
        "--execution-backend",
        choices=("eager", "inductor"),
        default="eager",
    )
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_dynamical_memory_falsification(
        tokenizer_checkpoint_path=args.tokenizer_checkpoint,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=args.general_train,
        general_eval_paths=args.general_eval,
        output_path=args.output,
        config=DynamicalMemoryFalsificationConfig(
            token_budget=max(1, int(args.token_budget)),
            sample_bytes_per_train_source=max(1, args.train_sample_mib)
            * 1024
            * 1024,
            sample_bytes_per_eval_source=max(1, args.eval_sample_mib)
            * 1024
            * 1024,
            execution_backend=args.execution_backend,
        ),
        device=args.device,
        arm_names=tuple(args.arm) or ARM_NAMES,
        baseline_report_path=args.baseline_report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
