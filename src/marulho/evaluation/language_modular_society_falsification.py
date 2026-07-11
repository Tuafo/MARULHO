"""Matched real-language falsification for modular predictive society v3."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

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
    _resolve_device,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    LanguageBatch,
    LanguageModelConfig,
    MarulhoLanguageModel,
    build_language_model_splits,
    evaluate_language_model,
)
from marulho.training.language_modular_society import (
    MarulhoModularSocietyLanguageModel,
    ModularSocietyConfig,
)


SURFACE = "marulho_modular_society_falsification.v1"
ARTIFACT_KIND = "marulho_modular_society_falsification"
ARM_NAMES = (
    "monolith",
    "average_no_message",
    "learned_no_message",
    "learned_shuffled",
    "learned_real",
)


@dataclass(frozen=True)
class ModularSocietyFalsificationConfig:
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
    cell_count: int = 4
    cell_width: int = 296
    cell_layers: int = 2
    attention_heads: int = 4
    event_interval: int = 24
    message_dim: int = 32
    society_execution_backend: str = "eager"
    compile_loss_tolerance: float = 1.0e-3


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
    path: str | Path, *, byte_budget: int, range_count: int
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
        round(index * maximum_start / max(1, count - 1)) for index in range(count)
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
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    orders = {
        "relation": torch.randperm(
            relation_batch_count, generator=generator
        ).tolist(),
        **{
            f"general_{index}": torch.randperm(count, generator=generator).tolist()
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
    config: ModularSocietyFalsificationConfig,
):
    if name == "monolith":
        return MarulhoLanguageModel(
            LanguageModelConfig(
                vocab_size=int(vocab_size),
                embedding_dim=512,
                state_dim=512,
                state_layers=4,
                attention_heads=8,
                transformer_context_length=config.sequence_length,
                transformer_mlp_ratio=4.0,
            )
        )
    return MarulhoModularSocietyLanguageModel(
        ModularSocietyConfig(
            vocab_size=int(vocab_size),
            cell_count=config.cell_count,
            cell_width=config.cell_width,
            cell_layers=config.cell_layers,
            attention_heads=config.attention_heads,
            context_length=config.sequence_length,
            event_interval=config.event_interval,
            message_dim=config.message_dim,
            mode=name,
        )
    )


def _ordinary_loss(model):
    def loss(input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        return model.next_token_loss(
            input_ids,
            target_ids,
            collect_telemetry=False,
            return_evidence=False,
        )["loss"]

    return loss


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


def _run_arm(
    name: str,
    *,
    tokenizer,
    relation_batches: Sequence[LanguageBatch],
    general_batches: Sequence[Sequence[LanguageBatch]],
    eval_batches: Sequence[LanguageBatch],
    cases: Sequence[RelationCase],
    schedule: Sequence[tuple[str, int]],
    config: ModularSocietyFalsificationConfig,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(config.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.model_seed)
    model = _build_model(
        name,
        vocab_size=tokenizer.vocab_size,
        config=config,
    ).to(device)
    training_config = LanguageTrainingExperimentConfig(
        learning_rate=config.learning_rate,
        minimum_learning_rate_fraction=config.minimum_learning_rate_fraction,
        warmup_fraction=config.warmup_fraction,
        weight_decay=config.weight_decay,
        precision=config.precision,
    )
    optimizer, fused = _optimizer(model, training_config)
    total_steps = len(schedule)
    warmup_steps = int(round(total_steps * config.warmup_fraction))
    eager_loss = _ordinary_loss(model)
    training_loss = eager_loss
    requested_backend = (
        "eager" if name == "monolith" else config.society_execution_backend
    )
    compile_seconds = 0.0
    compile_peak_memory = 0
    compile_reference: float | None = None
    compiled_observed: float | None = None
    compile_delta: float | None = None
    model.train()

    if requested_backend == "inductor":
        warm_kind, warm_index = schedule[0]
        warm_batch = _selected_batch(
            warm_kind,
            warm_index,
            relation_batches=relation_batches,
            general_batches=general_batches,
        ).to(device)
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        with torch.no_grad(), _precision_context(device, config.precision):
            reference_tensor = eager_loss(
                warm_batch.input_ids, warm_batch.target_ids
            )
        compiled_loss = torch.compile(
            eager_loss,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        compile_started = time.perf_counter()
        with _precision_context(device, config.precision):
            observed_tensor = compiled_loss(
                warm_batch.input_ids, warm_batch.target_ids
            )
        observed_tensor.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            compile_peak_memory = int(torch.cuda.max_memory_allocated(device))
        compile_seconds = time.perf_counter() - compile_started
        compile_reference = float(reference_tensor.detach().float().cpu())
        compiled_observed = float(observed_tensor.detach().float().cpu())
        compile_delta = abs(compiled_observed - compile_reference)
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        if compile_delta > float(config.compile_loss_tolerance):
            raise RuntimeError(
                "Compiled/eager warm-up loss drift exceeds tolerance: "
                f"{compile_delta:.8f} > {config.compile_loss_tolerance:.8f}"
            )
        training_loss = compiled_loss

    processed = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    final_loss: torch.Tensor | None = None
    for step, (kind, index) in enumerate(schedule):
        batch = _selected_batch(
            kind,
            index,
            relation_batches=relation_batches,
            general_batches=general_batches,
        ).to(device)
        learning_rate = _learning_rate(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            peak=config.learning_rate,
            minimum_fraction=config.minimum_learning_rate_fraction,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with _precision_context(device, config.precision):
            final_loss = training_loss(batch.input_ids, batch.target_ids)
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        processed += int(batch.target_ids.numel())
        if (step + 1) % max(1, total_steps // 10) == 0:
            print(f"[society-v3] {name} {step + 1}/{total_steps}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    if final_loss is None:
        raise RuntimeError("Training schedule produced no steps")

    gradient_elements = sum(
        int(torch.count_nonzero(parameter.grad).detach().cpu())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    gradient_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    heldout = evaluate_language_model(model, eval_batches)
    relation = evaluate_relation_binding_cases_batched(
        model,
        tokenizer,
        cases,
        batch_size=config.relation_eval_batch_size,
    )
    sample = general_batches[0][0].to(device)
    with torch.no_grad(), _precision_context(device, config.precision):
        sample_output = model(sample.input_ids)
    telemetry = sample_output["telemetry"]
    state_bytes = sum(
        int(value.numel() * value.element_size())
        for value in sample_output["state"].values()
        if isinstance(value, torch.Tensor)
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    end_to_end = elapsed + compile_seconds
    return {
        "name": name,
        "parameters": parameters,
        "parameters_with_final_gradient": gradient_parameters,
        "nonzero_final_gradient_elements": gradient_elements,
        "optimizer_fused": bool(fused),
        "processed_tokens": processed,
        "training": {
            "final_batch_loss": float(final_loss.detach().float().cpu()),
            "optimizer_step_seconds": elapsed,
            "tokens_per_second": processed / max(elapsed, 1.0e-9),
            "compile_seconds": compile_seconds,
            "end_to_end_seconds_including_compile": end_to_end,
            "amortized_tokens_per_second_including_compile": processed
            / max(end_to_end, 1.0e-9),
            "peak_cuda_memory_bytes": peak_memory,
            "compile_peak_cuda_memory_bytes": compile_peak_memory,
        },
        "execution": {
            "requested_backend": requested_backend,
            "effective_backend": requested_backend,
            "ordinary_step_backend": (
                "torch_compile_inductor_fullgraph"
                if requested_backend == "inductor"
                else "pytorch_eager"
            ),
            "compile_fullgraph": requested_backend == "inductor",
            "compile_dynamic_shapes": False,
            "warmup_loss_parity": {
                "performed": requested_backend == "inductor",
                "eager_loss": compile_reference,
                "compiled_loss": compiled_observed,
                "absolute_delta": compile_delta,
                "tolerance": config.compile_loss_tolerance,
                "passed": (
                    compile_delta <= config.compile_loss_tolerance
                    if compile_delta is not None
                    else None
                ),
            },
        },
        "heldout": heldout,
        "relation": relation,
        "runtime_state_bytes_for_training_batch": state_bytes,
        "telemetry": telemetry,
    }


def modular_society_decision(
    arms: Sequence[Mapping[str, Any]],
    *,
    minimum_tokens: int = 16_777_216,
) -> str:
    rows = {str(row["name"]): row for row in arms}
    processed = min(int(row.get("processed_tokens") or 0) for row in rows.values())
    if processed < int(minimum_tokens):
        return "incomplete_v3_mechanism_smoke"

    monolith = rows["monolith"]
    average = rows["average_no_message"]
    no_message = rows["learned_no_message"]
    shuffled = rows["learned_shuffled"]
    real = rows["learned_real"]

    def loss(row: Mapping[str, Any]) -> float:
        return float(row["heldout"]["heldout_loss"])

    def free(row: Mapping[str, Any]) -> float:
        return float(row["relation"]["generation_exact_accuracy"])

    communication_loss_win = loss(real) <= min(loss(no_message), loss(shuffled)) - 0.005
    communication_behavior_win = free(real) >= max(free(no_message), free(shuffled)) + 0.02
    monolith_quality_guard = (
        loss(real) <= loss(monolith) + 0.02
        and free(real) >= free(monolith) - 0.02
    )
    independent_society_win = any(
        loss(row) <= loss(monolith) - 0.005
        and free(row) >= free(monolith) + 0.02
        for row in (average, no_message)
    )
    if communication_loss_win and communication_behavior_win:
        if monolith_quality_guard:
            return "scale_v3_real_communication_to_64m_and_unseen_generation"
        return "redesign_v3_cell_capacity_before_scaling_communication"
    if independent_society_win:
        return "redesign_v3_message_bus_keep_independent_cell_result"
    return "retire_v3_modular_society_no_coordination_or_quality_gain"


def run_modular_society_falsification(
    *,
    tokenizer_checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    output_path: str | Path,
    config: ModularSocietyFalsificationConfig = ModularSocietyFalsificationConfig(),
    device: str = "auto",
    arm_names: Sequence[str] = ARM_NAMES,
    baseline_report_path: str | Path | None = None,
) -> dict[str, Any]:
    if len(general_train_paths) != 2 or len(general_eval_paths) != 2:
        raise ValueError("Exactly two general train and two eval sources are required")
    if config.society_execution_backend not in {"eager", "inductor"}:
        raise ValueError("society_execution_backend must be 'eager' or 'inductor'")
    if config.compile_loss_tolerance <= 0.0:
        raise ValueError("compile_loss_tolerance must be positive")
    resolved = _resolve_device(device)
    if config.society_execution_backend == "inductor" and resolved.type != "cuda":
        raise ValueError("Inductor society execution is admitted only for CUDA runs")
    requested_arms = tuple(dict.fromkeys(str(name) for name in arm_names))
    if not requested_arms or any(name not in ARM_NAMES for name in requested_arms):
        raise ValueError("arm_names must contain valid unique v3 arm names")

    tokenizer_checkpoint = Path(tokenizer_checkpoint_path)
    relation_cases_file = Path(relation_cases_path)
    tokenizer = _load_tokenizer(tokenizer_checkpoint)
    cases = _load_cases(relation_cases_file)
    steps = math.ceil(config.token_budget / (config.batch_size * config.sequence_length))
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

    reused_report: dict[str, Any] | None = None
    combined: dict[str, dict[str, Any]] = {}
    if baseline_report_path is not None:
        baseline_path = Path(baseline_report_path)
        reused_report = json.loads(baseline_path.read_text(encoding="utf-8"))
        if reused_report.get("surface") != SURFACE:
            raise ValueError("Baseline report surface does not match v3 runner")
        if dict(reused_report.get("configuration") or {}) != asdict(config):
            raise ValueError("Baseline report configuration does not match this run")
        if dict(reused_report.get("source_selections") or {}) != source_selections:
            raise ValueError("Baseline report source selections do not match this run")
        if str(reused_report["tokenizer"]["hash"]) != tokenizer.vocabulary_hash():
            raise ValueError("Baseline report tokenizer does not match this run")
        if str(reused_report["schedule"]["sha256"]) != schedule_digest:
            raise ValueError("Baseline report schedule does not match this run")
        combined = {str(row["name"]): row for row in reused_report["arms"]}

    executed: list[dict[str, Any]] = []
    for name in requested_arms:
        print(f"[society-v3] starting {name}", flush=True)
        row = _run_arm(
            name,
            tokenizer=tokenizer,
            relation_batches=relation_split.train,
            general_batches=[split.train for split in general_splits],
            eval_batches=eval_split.eval,
            cases=cases,
            schedule=schedule,
            config=config,
            device=resolved,
        )
        executed.append(row)
        combined[name] = row
        print(
            f"[society-v3] completed {name}: loss "
            f"{row['heldout']['heldout_loss']:.4f}, free "
            f"{row['relation']['generation_exact_accuracy']:.3f}",
            flush=True,
        )
        gc.collect()
        if resolved.type == "cuda":
            torch.cuda.empty_cache()

    arms = [combined[name] for name in ARM_NAMES if name in combined]
    decision = (
        modular_society_decision(arms)
        if len(arms) == len(ARM_NAMES)
        else "incomplete_matched_v3_comparison"
    )
    parameter_counts = {row["name"]: int(row["parameters"]) for row in arms}
    monolith_parameters = parameter_counts.get("monolith")
    society_parameters = parameter_counts.get("learned_real")
    parameter_delta_fraction = (
        (society_parameters - monolith_parameters) / monolith_parameters
        if monolith_parameters and society_parameters
        else None
    )
    society_rows = [row for row in arms if row["name"] != "monolith"]
    society_steady_rates = [
        float(row["training"]["tokens_per_second"]) for row in society_rows
    ]
    report = {
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
        "source_selections": source_selections,
        "schedule": {
            "steps": steps,
            "processed_tokens": steps * config.batch_size * config.sequence_length,
            "relation_steps": sum(kind == "relation" for kind, _ in schedule),
            "sha256": schedule_digest,
            "identical_for_all_arms": True,
        },
        "parameter_match": {
            "counts": parameter_counts,
            "society_minus_monolith_fraction": parameter_delta_fraction,
            "within_two_tenths_percent": (
                abs(parameter_delta_fraction) < 0.002
                if parameter_delta_fraction is not None
                else None
            ),
        },
        "society_control_compute": {
            "same_declared_parameter_graph": True,
            "backend_may_eliminate_zero_information_paths": True,
            "steady_tokens_per_second_min": (
                min(society_steady_rates) if society_steady_rates else None
            ),
            "steady_tokens_per_second_max": (
                max(society_steady_rates) if society_steady_rates else None
            ),
            "max_to_min_ratio": (
                max(society_steady_rates) / min(society_steady_rates)
                if society_steady_rates
                else None
            ),
        },
        "arms": arms,
        "arms_executed_this_run": [row["name"] for row in executed],
        "reused_control_report": (
            None
            if baseline_report_path is None
            else {
                "path": str(baseline_report_path),
                "sha256": _sha256_file(Path(baseline_report_path)),
            }
        ),
        "decision": decision,
        "decision_contract": {
            "real_messages_must_beat": [
                "learned_no_message",
                "learned_shuffled",
            ],
            "heldout_loss_margin": 0.005,
            "free_relation_margin": 0.02,
            "no_checkpoint_saved_before_survival": True,
        },
        "quality_boundary": {
            "promotes_runtime_installation": False,
            "promotes_unseen_generation": False,
            "throughput_is_not_quality": True,
        },
    }
    write_json_report_with_readme(
        Path(output_path),
        report,
        title="MARULHO Modular Predictive Society v3 Falsification",
    )
    print(f"[society-v3] decision {decision}", flush=True)
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
        "--society-backend",
        choices=("eager", "inductor"),
        default="eager",
    )
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_modular_society_falsification(
        tokenizer_checkpoint_path=args.tokenizer_checkpoint,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=args.general_train,
        general_eval_paths=args.general_eval,
        output_path=args.output,
        config=ModularSocietyFalsificationConfig(
            token_budget=max(1, int(args.token_budget)),
            sample_bytes_per_train_source=max(1, args.train_sample_mib)
            * 1024
            * 1024,
            sample_bytes_per_eval_source=max(1, args.eval_sample_mib)
            * 1024
            * 1024,
            society_execution_backend=args.society_backend,
        ),
        device=args.device,
        arm_names=tuple(args.arm) or ARM_NAMES,
        baseline_report_path=args.baseline_report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
