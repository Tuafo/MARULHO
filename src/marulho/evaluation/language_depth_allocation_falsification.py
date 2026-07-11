"""Matched falsification for exact-budget Transformer depth allocation v8."""

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
    _precision_context,
    _prepare_language_loss_backend,
    _resolve_device,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_depth_allocation import (
    DepthAllocationConfig,
    MarulhoDepthAllocatedLanguageModel,
    matching_common_parameter_names,
    mlp_parameter_count,
    total_parameter_count,
)
from marulho.training.language_model import (
    LanguageBatch,
    build_language_model_splits,
    evaluate_language_model,
)


SURFACE = "marulho_depth_allocation_falsification.v1"
ARTIFACT_KIND = "marulho_depth_allocation_falsification"
PROFILE_WIDTHS: dict[str, tuple[int, ...]] = {
    "uniform": (2048, 2048, 2048, 2048),
    "early_heavy": (3072, 2560, 1536, 1024),
    "late_heavy": (1024, 1536, 2560, 3072),
}
ARM_NAMES = tuple(PROFILE_WIDTHS)
COMPARISON_STAGES = ("screen", "durability")
DURABILITY_ARM_NAMES = ("uniform", "early_heavy")


@dataclass(frozen=True)
class DepthAllocationFalsificationConfig:
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
    attention_heads: int = 8
    execution_backend: str = "eager"
    compile_loss_tolerance: float = 1.0e-3
    comparison_stage: str = "screen"


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


def _parameter_hash(
    model: MarulhoDepthAllocatedLanguageModel,
    names: Sequence[str],
) -> str:
    selected = set(names)
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name not in selected:
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


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


def _build_model(
    name: str,
    *,
    vocab_size: int,
    config: DepthAllocationFalsificationConfig,
) -> MarulhoDepthAllocatedLanguageModel:
    if name not in PROFILE_WIDTHS:
        raise ValueError(f"Unknown depth-allocation profile: {name}")
    return MarulhoDepthAllocatedLanguageModel(
        DepthAllocationConfig(
            vocab_size=int(vocab_size),
            width=int(config.width),
            attention_heads=int(config.attention_heads),
            context_length=int(config.sequence_length),
            mlp_hidden_widths=PROFILE_WIDTHS[name],
            initialization_seed=int(config.model_seed),
            active_language_path=f"marulho_depth_allocation_v8_{name}",
        )
    )


def _training_config(
    config: DepthAllocationFalsificationConfig,
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
    model: MarulhoDepthAllocatedLanguageModel,
    initial_state: Mapping[str, torch.Tensor],
    training_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    execution: Mapping[str, Any],
    tokenizer,
    staged: StagedSchedule,
    eval_batches: Sequence[LanguageBatch],
    cases: Sequence[RelationCase],
    config: DepthAllocationFalsificationConfig,
    device: torch.device,
    common_initialization_sha256: str,
    mlp_initialization_sha256: str,
) -> dict[str, Any]:
    model.load_state_dict(initial_state, strict=True)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        if step in trace_steps:
            trace_tensors.append((step + 1, final_loss.detach().float()))
        if (step + 1) % max(1, total_steps // 10) == 0:
            print(
                f"[depth-allocation-v8] {name} {step + 1}/{total_steps}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_elapsed = time.perf_counter() - arm_started
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
        sample_output = model(sample.input_ids, collect_telemetry=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_elapsed = time.perf_counter() - evaluation_started
    state_bytes = sum(
        int(value.numel() * value.element_size())
        for value in sample_output["state"].values()
        if isinstance(value, torch.Tensor)
    )
    processed = total_steps * int(staged.target_ids[0].numel())
    compile_seconds = float(execution["compile_seconds"])
    end_to_end = training_elapsed + compile_seconds
    total_parameters = total_parameter_count(model)
    mlp_names = [
        parameter_name
        for parameter_name, _parameter in model.named_parameters()
        if parameter_name.endswith((".gate_up.weight", ".down.weight"))
    ]
    return {
        "name": name,
        "architecture": "depth_allocated_transformer",
        "mlp_hidden_widths": list(PROFILE_WIDTHS[name]),
        "mlp_hidden_width_sum": sum(PROFILE_WIDTHS[name]),
        "mlp_parameters": mlp_parameter_count(config.width, PROFILE_WIDTHS[name]),
        "parameters": total_parameters,
        "parameters_with_final_gradient": parameters_with_gradient,
        "nonzero_final_gradient_elements": nonzero_gradient_elements,
        "all_parameters_received_final_gradient": (
            parameters_with_gradient == total_parameters
        ),
        "optimizer_fused": bool(fused),
        "optimizer_state_fresh_for_arm": True,
        "initial_weights_restored_for_arm": True,
        "common_initialization_sha256": common_initialization_sha256,
        "mlp_initialization_sha256": mlp_initialization_sha256,
        "processed_tokens": processed,
        "training": {
            "final_batch_loss": float(final_loss.detach().float().cpu()),
            "loss_trace": [
                {
                    "step": step,
                    "processed_tokens": step * int(staged.target_ids[0].numel()),
                    "training_batch_loss": float(value.cpu()),
                }
                for step, value in trace_tensors
            ],
            "optimizer_step_count": total_steps,
            "optimizer_step_seconds": training_elapsed,
            "tokens_per_second": processed / max(training_elapsed, 1.0e-9),
            "compile_seconds": compile_seconds,
            "end_to_end_seconds_including_compile": end_to_end,
            "amortized_tokens_per_second_including_compile": (
                processed / max(end_to_end, 1.0e-9)
            ),
            "evaluation_seconds": evaluation_elapsed,
            "peak_cuda_memory_bytes": peak_memory,
            "per_step_host_metric_readback": False,
            "scheduled_batches_pre_staged_on_device": True,
        },
        "execution": dict(execution),
        "heldout": heldout,
        "relation": relation,
        "runtime_state_bytes_for_training_batch": state_bytes,
        "telemetry": sample_output["telemetry"],
        "initialization_audit": {
            "common_parameter_count": total_parameters
            - sum(
                parameter.numel()
                for parameter_name, parameter in model.named_parameters()
                if parameter_name in mlp_names
            ),
            "mlp_parameter_names": mlp_names,
        },
    }


def depth_allocation_decision(
    arms: Sequence[Mapping[str, Any]],
    *,
    minimum_tokens: int | None = None,
    comparison_stage: str = "screen",
) -> str:
    if comparison_stage not in COMPARISON_STAGES:
        raise ValueError("Unknown v8 comparison stage")
    required_names = (
        ARM_NAMES if comparison_stage == "screen" else DURABILITY_ARM_NAMES
    )
    threshold = (
        int(minimum_tokens)
        if minimum_tokens is not None
        else 16_777_216
        if comparison_stage == "screen"
        else 67_108_864
    )
    rows = {str(row["name"]): row for row in arms}
    if any(name not in rows for name in required_names):
        return (
            "incomplete_v8_depth_allocation_comparison"
            if comparison_stage == "screen"
            else "incomplete_v8_durability_comparison"
        )
    processed = min(
        int(rows[name].get("processed_tokens") or 0) for name in required_names
    )
    if processed < threshold:
        return (
            "incomplete_v8_mechanism_smoke"
            if comparison_stage == "screen"
            else "incomplete_v8_durability_budget"
        )

    def loss(name: str) -> float:
        return float(rows[name]["heldout"]["heldout_loss"])

    def free(name: str) -> float:
        return float(rows[name]["relation"]["generation_exact_accuracy"])

    def qualified(name: str) -> bool:
        return (
            loss(name) <= loss("uniform") - 0.005
            and free(name) >= free("uniform") + 0.02
        )

    candidate_names = tuple(name for name in required_names if name != "uniform")
    winners = [name for name in candidate_names if qualified(name)]
    if winners:
        best = min(winners, key=lambda name: (loss(name), -free(name)))
        return (
            f"replicate_v8_{best}_before_scale"
            if comparison_stage == "screen"
            else f"promote_v8_{best}_to_quality_baseline"
        )
    loss_signal = any(
        loss(name) <= loss("uniform") - 0.005 for name in candidate_names
    )
    behavior_signal = any(
        free(name) >= free("uniform") + 0.02 for name in candidate_names
    )
    if loss_signal and behavior_signal:
        return "redesign_v8_disjoint_loss_and_behavior_signals"
    if loss_signal:
        return "redesign_v8_loss_signal_without_free_generation"
    if behavior_signal:
        return "redesign_v8_behavior_signal_without_loss_gain"
    return (
        "retire_v8_static_depth_allocation"
        if comparison_stage == "screen"
        else "retire_v8_early_heavy_not_durable"
    )


def _assemble_report(
    *,
    config: DepthAllocationFalsificationConfig,
    tokenizer_checkpoint: Path,
    tokenizer,
    relation_cases_file: Path,
    cases: Sequence[RelationCase],
    source_selections: Mapping[str, Any],
    schedule: Sequence[tuple[str, int]],
    schedule_digest: str,
    staged: StagedSchedule,
    combined: Mapping[str, Mapping[str, Any]],
    executed_names: Sequence[str],
    elapsed_seconds: float,
) -> dict[str, Any]:
    arms = [dict(combined[name]) for name in ARM_NAMES if name in combined]
    counts = {row["name"]: int(row["parameters"]) for row in arms}
    mlp_counts = {row["name"]: int(row["mlp_parameters"]) for row in arms}
    common_hashes = {
        str(row["common_initialization_sha256"]) for row in arms
    }
    rates = [float(row["training"]["tokens_per_second"]) for row in arms]
    compile_seconds = sum(
        float(row["execution"]["compile_seconds"]) for row in arms
    )
    return {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "configuration": asdict(config),
        "profiles": {name: list(widths) for name, widths in PROFILE_WIDTHS.items()},
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
            "relation_steps": sum(kind == "relation" for kind, _index in schedule),
            "sha256": schedule_digest,
            "identical_for_all_arms": True,
            "staged_once_on_device": True,
            "staging_seconds": staged.elapsed_seconds,
            "staging_storage_bytes": staged.storage_bytes,
        },
        "parameter_and_compute_match": {
            "parameter_counts": counts,
            "mlp_parameter_counts": mlp_counts,
            "all_parameter_counts_equal": len(set(counts.values())) <= 1,
            "all_mlp_parameter_counts_equal": len(set(mlp_counts.values())) <= 1,
            "all_mlp_hidden_width_sums_equal": (
                len({sum(widths) for widths in PROFILE_WIDTHS.values()}) == 1
            ),
            "same_attention_and_embedding_shapes": True,
            "same_theoretical_mlp_matmul_work": True,
            "steady_tokens_per_second_min": min(rates) if rates else None,
            "steady_tokens_per_second_max": max(rates) if rates else None,
            "max_to_min_ratio": (
                max(rates) / min(rates) if rates else None
            ),
        },
        "initialization_match": {
            "common_parameter_sha256_values": sorted(common_hashes),
            "all_non_mlp_initial_parameters_identical": len(common_hashes) <= 1,
            "mlp_initialization_is_layer_seeded": True,
        },
        "compile": {
            "distinct_shape_graph_count": len(arms),
            "compile_seconds_this_run": compile_seconds,
            "compile_cost_included_in_amortized_throughput": True,
        },
        "arms": arms,
        "arms_executed_this_run": list(executed_names),
        "experiment_wall_seconds_this_run": elapsed_seconds,
        "decision": depth_allocation_decision(
            arms,
            comparison_stage=config.comparison_stage,
        ),
        "decision_contract": {
            "comparison_stage": config.comparison_stage,
            "minimum_processed_tokens_per_arm": (
                16_777_216
                if config.comparison_stage == "screen"
                else 67_108_864
            ),
            "heldout_loss_margin": 0.005,
            "free_relation_margin": 0.02,
            "candidate_must_beat_uniform_on_both": True,
            "positive_requires_replication_before_scale": True,
            "no_checkpoint_saved_before_survival": True,
            "behavior_or_loss_only_routes_to_redesign_not_scale": True,
        },
        "quality_boundary": {
            "promotes_runtime_installation": False,
            "promotes_unseen_generation": False,
            "throughput_is_not_quality": True,
        },
    }


def run_depth_allocation_falsification(
    *,
    tokenizer_checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    output_path: str | Path,
    config: DepthAllocationFalsificationConfig = (
        DepthAllocationFalsificationConfig()
    ),
    device: str = "auto",
    arm_names: Sequence[str] = ARM_NAMES,
) -> dict[str, Any]:
    if len(general_train_paths) != 2 or len(general_eval_paths) != 2:
        raise ValueError("Exactly two general train and two eval sources are required")
    if config.execution_backend not in {"eager", "inductor"}:
        raise ValueError("execution_backend must be 'eager' or 'inductor'")
    if config.compile_loss_tolerance <= 0.0:
        raise ValueError("compile_loss_tolerance must be positive")
    if config.comparison_stage not in COMPARISON_STAGES:
        raise ValueError("comparison_stage must be 'screen' or 'durability'")
    resolved = _resolve_device(device)
    if config.execution_backend == "inductor" and resolved.type != "cuda":
        raise ValueError("Inductor v8 execution is admitted only for CUDA runs")
    requested_arms = tuple(dict.fromkeys(str(name) for name in arm_names))
    if not requested_arms or any(name not in ARM_NAMES for name in requested_arms):
        raise ValueError("arm_names must contain valid unique v8 arm names")
    if config.comparison_stage == "durability" and set(requested_arms) != set(
        DURABILITY_ARM_NAMES
    ):
        raise ValueError(
            "durability comparison requires exactly uniform and early_heavy"
        )
    profile_sums = {sum(PROFILE_WIDTHS[name]) for name in ARM_NAMES}
    if len(profile_sums) != 1:
        raise RuntimeError("V8 profiles do not have an exact shared MLP budget")

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

    combined: dict[str, Mapping[str, Any]] = {}
    executed_names: list[str] = []
    expected_common_hash: str | None = None
    output = Path(output_path)
    previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_matmul_precision = torch.get_float32_matmul_precision()
    if resolved.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    try:
        for name in requested_arms:
            print(f"[depth-allocation-v8] preparing {name} graph", flush=True)
            torch.manual_seed(config.model_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.model_seed)
            model = _build_model(name, vocab_size=tokenizer.vocab_size, config=config)
            common_names = matching_common_parameter_names(model)
            mlp_names = tuple(
                parameter_name
                for parameter_name, _parameter in model.named_parameters()
                if parameter_name not in set(common_names)
            )
            common_hash = _parameter_hash(model, common_names)
            mlp_hash = _parameter_hash(model, mlp_names)
            if expected_common_hash is None:
                expected_common_hash = common_hash
            elif common_hash != expected_common_hash:
                raise RuntimeError(
                    "Non-MLP initialization changed across V8 allocation profiles"
                )
            initial_state = {
                parameter_name: value.detach().clone()
                for parameter_name, value in model.state_dict().items()
            }
            model = model.to(resolved)
            training_config = _training_config(config)
            warm_batch = LanguageBatch(staged.input_ids[0], staged.target_ids[0])
            model.train()
            training_loss, execution = _prepare_language_loss_backend(
                model,
                warm_batch,
                training_config,
            )
            model.eval()
            initial_heldout = evaluate_language_model(model, eval_split.eval)
            print(f"[depth-allocation-v8] starting {name}", flush=True)
            row = _run_arm(
                name,
                model=model,
                initial_state=initial_state,
                training_loss=training_loss,
                execution=execution,
                tokenizer=tokenizer,
                staged=staged,
                eval_batches=eval_split.eval,
                cases=cases,
                config=config,
                device=resolved,
                common_initialization_sha256=common_hash,
                mlp_initialization_sha256=mlp_hash,
            )
            row["initial_heldout"] = initial_heldout
            combined[name] = row
            executed_names.append(name)
            print(
                f"[depth-allocation-v8] completed {name}: loss "
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
                executed_names=executed_names,
                elapsed_seconds=time.perf_counter() - run_started,
            )
            write_json_report_with_readme(
                output,
                partial,
                title="MARULHO Depth Allocation v8 Falsification",
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
        executed_names=executed_names,
        elapsed_seconds=time.perf_counter() - run_started,
    )
    write_json_report_with_readme(
        output,
        report,
        title="MARULHO Depth Allocation v8 Falsification",
    )
    print(f"[depth-allocation-v8] decision {report['decision']}", flush=True)
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
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--model-seed", type=int, default=1337)
    parser.add_argument(
        "--comparison-stage",
        choices=COMPARISON_STAGES,
        default="screen",
    )
    parser.add_argument("--arm", action="append", choices=ARM_NAMES, default=[])
    parser.add_argument(
        "--execution-backend",
        choices=("eager", "inductor"),
        default="eager",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_depth_allocation_falsification(
        tokenizer_checkpoint_path=args.tokenizer_checkpoint,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=args.general_train,
        general_eval_paths=args.general_eval,
        output_path=args.output,
        config=DepthAllocationFalsificationConfig(
            token_budget=max(1, int(args.token_budget)),
            sample_bytes_per_train_source=max(1, args.train_sample_mib)
            * 1024
            * 1024,
            sample_bytes_per_eval_source=max(1, args.eval_sample_mib)
            * 1024
            * 1024,
            seed=int(args.seed),
            model_seed=int(args.model_seed),
            execution_backend=args.execution_backend,
            comparison_stage=args.comparison_stage,
        ),
        device=args.device,
        arm_names=tuple(args.arm) or ARM_NAMES,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
