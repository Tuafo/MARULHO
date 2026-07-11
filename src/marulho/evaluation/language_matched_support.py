"""Reusable mechanics for matched MARULHO language architecture falsifiers."""

from __future__ import annotations

from dataclasses import dataclass
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
)
from marulho.training.language_model import (
    LanguageBatch,
    build_language_model_splits,
    evaluate_language_model,
)
from marulho.training.language_protocol import CausalLanguageModel


@dataclass(frozen=True)
class MatchedLanguageDataConfig:
    token_budget: int = 16_777_216
    sequence_length: int = 72
    batch_size: int = 144
    eval_batches: int = 16
    relation_fraction: float = 0.20
    seed: int = 1337
    sample_bytes_per_train_source: int = 64 * 1024 * 1024
    sample_bytes_per_eval_source: int = 32 * 1024 * 1024
    sample_range_count: int = 16
    schedule_mode: str = "expanded_device"


@dataclass(frozen=True)
class StagedSchedule:
    input_ids: torch.Tensor | None
    target_ids: torch.Tensor | None
    schedule: tuple[tuple[str, int], ...]
    relation_batches: tuple[LanguageBatch, ...]
    general_batches: tuple[tuple[LanguageBatch, ...], ...]
    mode: str
    step_count: int
    tokens_per_step: int
    elapsed_seconds: float
    storage_bytes: int
    expanded_storage_bytes: int
    device_storage_bytes: int
    host_storage_bytes: int

    def batch(
        self,
        index: int,
        device: torch.device | str,
    ) -> LanguageBatch:
        position = int(index)
        if not 0 <= position < int(self.step_count):
            raise IndexError("training schedule index is out of range")
        if self.mode == "expanded_device":
            if self.input_ids is None or self.target_ids is None:
                raise RuntimeError("expanded schedule tensors are unavailable")
            return LanguageBatch(
                self.input_ids[position].to(device),
                self.target_ids[position].to(device),
            )
        if self.mode != "indexed_host":
            raise RuntimeError(f"unknown training schedule mode: {self.mode}")
        kind, source_index = self.schedule[position]
        selected = _selected_batch(
            kind,
            source_index,
            relation_batches=self.relation_batches,
            general_batches=self.general_batches,
        )
        return selected.to(device)


@dataclass(frozen=True)
class PreparedMatchedLanguageData:
    tokenizer_checkpoint: Path
    relation_cases_file: Path
    tokenizer: Any
    cases: tuple[RelationCase, ...]
    eval_batches: tuple[LanguageBatch, ...]
    staged: StagedSchedule
    schedule: tuple[tuple[str, int], ...]
    schedule_sha256: str
    source_selections: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schedule_sha256(schedule: Sequence[tuple[str, int]]) -> str:
    payload = json.dumps(list(schedule), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parameter_sha256(
    model: CausalLanguageModel,
    *,
    excluded_names: Sequence[str] = (),
) -> str:
    excluded = set(excluded_names)
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name in excluded:
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


def full_sized_batches(
    batches: Sequence[LanguageBatch],
    *,
    batch_size: int,
) -> tuple[LanguageBatch, ...]:
    selected = tuple(
        batch
        for batch in batches
        if int(batch.input_ids.shape[0]) == int(batch_size)
        and int(batch.target_ids.shape[0]) == int(batch_size)
    )
    if not selected:
        raise ValueError("Training source contains no full-sized batches")
    return selected


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
    relation_fraction = float(relation_fraction)
    if not 0.0 <= relation_fraction <= 1.0:
        raise ValueError("relation_fraction must be in [0, 1]")
    if any(int(count) < 1 for count in general_batch_counts):
        raise ValueError("Every general source requires at least one batch")
    if relation_fraction > 0.0 and int(relation_batch_count) < 1:
        raise ValueError("Scheduled relation source requires at least one batch")
    if not general_batch_counts:
        raise ValueError("At least one general source is required")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    orders = {
            f"general_{index}": torch.randperm(
                count,
                generator=generator,
            ).tolist()
            for index, count in enumerate(general_batch_counts)
    }
    if relation_fraction > 0.0:
        orders["relation"] = torch.randperm(
            relation_batch_count,
            generator=generator,
        ).tolist()
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
    if payload.get("surface") not in {
        "marulho_transformer_language_checkpoint.v2",
        "marulho_hashed_micro_expert_language_checkpoint.v1",
    }:
        raise ValueError("Tokenizer source must be a maintained language checkpoint")
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


def stage_schedule(
    schedule: Sequence[tuple[str, int]],
    *,
    relation_batches: Sequence[LanguageBatch],
    general_batches: Sequence[Sequence[LanguageBatch]],
    device: torch.device,
    mode: str = "expanded_device",
) -> StagedSchedule:
    mode = str(mode).strip().lower()
    if mode not in {"expanded_device", "indexed_host"}:
        raise ValueError(
            "schedule_mode must be 'expanded_device' or 'indexed_host'"
        )
    schedule_tuple = tuple((str(kind), int(index)) for kind, index in schedule)
    if not schedule_tuple:
        raise ValueError("training schedule cannot be empty")
    relation_tuple = tuple(relation_batches)
    general_tuple = tuple(tuple(batches) for batches in general_batches)
    first_kind, first_index = schedule_tuple[0]
    first = _selected_batch(
        first_kind,
        first_index,
        relation_batches=relation_tuple,
        general_batches=general_tuple,
    )
    tokens_per_step = int(first.target_ids.numel())
    expanded_storage_bytes = int(
        len(schedule_tuple)
        * (
            first.input_ids.numel() * first.input_ids.element_size()
            + first.target_ids.numel() * first.target_ids.element_size()
        )
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    if mode == "expanded_device":
        selected = [
            _selected_batch(
                kind,
                index,
                relation_batches=relation_tuple,
                general_batches=general_tuple,
            )
            for kind, index in schedule_tuple
        ]
        input_ids = torch.stack([batch.input_ids for batch in selected]).to(device)
        target_ids = torch.stack([batch.target_ids for batch in selected]).to(device)
        storage_bytes = int(
            input_ids.numel() * input_ids.element_size()
            + target_ids.numel() * target_ids.element_size()
        )
        device_storage_bytes = storage_bytes if device.type == "cuda" else 0
        host_storage_bytes = storage_bytes if device.type == "cpu" else 0
    else:
        input_ids = None
        target_ids = None
        unique_batches = [*relation_tuple]
        for batches in general_tuple:
            unique_batches.extend(batches)
        storage_bytes = int(
            sum(
                batch.input_ids.numel() * batch.input_ids.element_size()
                + batch.target_ids.numel() * batch.target_ids.element_size()
                for batch in unique_batches
            )
        )
        device_storage_bytes = 0
        host_storage_bytes = storage_bytes
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return StagedSchedule(
        input_ids=input_ids,
        target_ids=target_ids,
        schedule=schedule_tuple,
        relation_batches=relation_tuple,
        general_batches=general_tuple,
        mode=mode,
        step_count=len(schedule_tuple),
        tokens_per_step=tokens_per_step,
        elapsed_seconds=time.perf_counter() - started,
        storage_bytes=storage_bytes,
        expanded_storage_bytes=expanded_storage_bytes,
        device_storage_bytes=device_storage_bytes,
        host_storage_bytes=host_storage_bytes,
    )


def prepare_matched_language_data(
    *,
    tokenizer_checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    config: MatchedLanguageDataConfig,
    device: torch.device,
) -> PreparedMatchedLanguageData:
    if len(general_train_paths) != 2 or len(general_eval_paths) != 2:
        raise ValueError("Exactly two general train and two eval sources are required")
    tokenizer_checkpoint = Path(tokenizer_checkpoint_path)
    relation_cases_file = Path(relation_cases_path)
    tokenizer = _load_tokenizer(tokenizer_checkpoint)
    cases = _load_cases(relation_cases_file)
    steps = math.ceil(
        int(config.token_budget) / (int(config.batch_size) * int(config.sequence_length))
    )
    relation_enabled = float(config.relation_fraction) > 0.0
    relation_steps = (
        max(1, int(round(steps * float(config.relation_fraction))))
        if relation_enabled
        else 0
    )
    general_steps = max(1, math.ceil((steps - relation_steps) / 2))
    if relation_enabled:
        relation_text, relation_selection = sample_corpus_ranges(
            relation_corpus_path,
            byte_budget=config.sample_bytes_per_train_source,
            range_count=config.sample_range_count,
        )
    else:
        relation_text = ""
        relation_selection = {
            "path": str(Path(relation_corpus_path)),
            "scheduled": False,
            "selected_size_bytes": 0,
            "ranges": [],
        }
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
    relation_split = (
        build_language_model_splits(
            [relation_text],
            tokenizer,
            sequence_length=config.sequence_length,
            stride=config.sequence_length,
            batch_size=config.batch_size,
            max_train_batches=relation_steps,
            max_eval_batches=1,
        )
        if relation_enabled
        else None
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
    relation_batches = (
        full_sized_batches(
            relation_split.train,
            batch_size=config.batch_size,
        )
        if relation_split is not None
        else tuple()
    )
    general_batches = [
        full_sized_batches(split.train, batch_size=config.batch_size)
        for split in general_splits
    ]
    schedule = build_matched_schedule(
        step_count=steps,
        relation_fraction=config.relation_fraction,
        relation_batch_count=len(relation_batches),
        general_batch_counts=[len(batches) for batches in general_batches],
        seed=config.seed,
    )
    staged = stage_schedule(
        schedule,
        relation_batches=relation_batches,
        general_batches=general_batches,
        device=device,
        mode=config.schedule_mode,
    )
    return PreparedMatchedLanguageData(
        tokenizer_checkpoint=tokenizer_checkpoint,
        relation_cases_file=relation_cases_file,
        tokenizer=tokenizer,
        cases=cases,
        eval_batches=tuple(eval_split.eval),
        staged=staged,
        schedule=schedule,
        schedule_sha256=schedule_sha256(schedule),
        source_selections={
            "relation": relation_selection,
            "general_train": [row for _text, row in train_samples],
            "general_eval": [row for _text, row in eval_samples],
            "training_batch_filter": {
                "required_batch_size": config.batch_size,
                "relation_batches_before": (
                    0 if relation_split is None else len(relation_split.train)
                ),
                "relation_batches_after": len(relation_batches),
                "general_batches_before": [
                    len(split.train) for split in general_splits
                ],
                "general_batches_after": [
                    len(batches) for batches in general_batches
                ],
                "partial_batches_excluded": True,
            },
            "schedule_storage": {
                "mode": staged.mode,
                "resident_storage_bytes": staged.storage_bytes,
                "expanded_storage_bytes": staged.expanded_storage_bytes,
                "device_storage_bytes": staged.device_storage_bytes,
                "host_storage_bytes": staged.host_storage_bytes,
            },
        },
    )


def run_matched_training_arm(
    name: str,
    *,
    architecture: str,
    model: CausalLanguageModel,
    initial_state: Mapping[str, torch.Tensor],
    training_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    execution: Mapping[str, Any],
    allocated_compile_seconds: float,
    prepared: PreparedMatchedLanguageData,
    training_config: LanguageTrainingExperimentConfig,
    gradient_clip: float,
    precision: str,
    relation_eval_batch_size: int,
    model_seed: int,
    device: torch.device,
    progress_prefix: str,
    configure_model: Callable[[CausalLanguageModel, str], None] | None = None,
    diagnostic_builder: (
        Callable[[CausalLanguageModel, torch.Tensor], Mapping[str, Any]] | None
    ) = None,
    extra_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model.load_state_dict(dict(initial_state), strict=True)
    if configure_model is not None:
        configure_model(model, name)
    model.zero_grad(set_to_none=True)
    torch.manual_seed(int(model_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(model_seed))
    optimizer, fused = _optimizer(model, training_config)
    total_steps = int(prepared.staged.step_count)
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
            minimum_fraction=float(training_config.minimum_learning_rate_fraction),
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with _precision_context(device, precision):
            training_batch = prepared.staged.batch(step, device)
            final_loss = training_loss(
                training_batch.input_ids,
                training_batch.target_ids,
            )
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
        optimizer.step()
        if step in trace_steps:
            trace_tensors.append((step + 1, final_loss.detach().float()))
        if (step + 1) % max(1, total_steps // 10) == 0:
            print(f"[{progress_prefix}] {name} {step + 1}/{total_steps}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_elapsed = time.perf_counter() - arm_started
    if final_loss is None:
        raise RuntimeError("Training schedule produced no steps")
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
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
    heldout = evaluate_language_model(model, prepared.eval_batches)
    relation = evaluate_relation_binding_cases_batched(
        model,
        prepared.tokenizer,
        prepared.cases,
        batch_size=int(relation_eval_batch_size),
    )
    sample = prepared.eval_batches[0].to(device)
    with torch.no_grad(), _precision_context(device, precision):
        sample_output = model(sample.input_ids, collect_telemetry=True)
    diagnostics = (
        dict(diagnostic_builder(model, sample.input_ids))
        if diagnostic_builder is not None
        else {}
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_elapsed = time.perf_counter() - evaluation_started
    state_bytes = sum(
        int(value.numel() * value.element_size())
        for value in sample_output["state"].values()
        if isinstance(value, torch.Tensor)
    )
    processed = total_steps * int(prepared.staged.tokens_per_step)
    end_to_end = training_elapsed + float(allocated_compile_seconds)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "name": name,
        "architecture": architecture,
        **dict(extra_row or {}),
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
            "loss_trace": [
                {
                    "step": trace_step,
                    "processed_tokens": trace_step
                    * int(prepared.staged.tokens_per_step),
                    "training_batch_loss": float(value.cpu()),
                }
                for trace_step, value in trace_tensors
            ],
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
            "peak_cuda_memory_bytes": peak_memory,
            "per_step_host_metric_readback": False,
            "scheduled_batches_pre_staged_on_device": (
                prepared.staged.mode == "expanded_device"
            ),
            "schedule_mode": prepared.staged.mode,
            "schedule_resident_storage_bytes": prepared.staged.storage_bytes,
            "schedule_expanded_storage_bytes": (
                prepared.staged.expanded_storage_bytes
            ),
            "schedule_device_storage_bytes": (
                prepared.staged.device_storage_bytes
            ),
            "schedule_host_storage_bytes": prepared.staged.host_storage_bytes,
        },
        "execution": dict(execution),
        "heldout": heldout,
        "relation": relation,
        "runtime_state_bytes_for_training_batch": state_bytes,
        "telemetry": sample_output["telemetry"],
        "diagnostics": diagnostics,
    }
