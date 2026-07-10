"""Matched Transformer versus distributed predictive-organism falsification."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

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
from marulho.training.language_organism import (
    DistributedLanguageConfig,
    MarulhoDistributedLanguageModel,
    save_distributed_language_checkpoint,
)


SURFACE = "marulho_organism_falsification.v1"
ARTIFACT_KIND = "marulho_organism_falsification"
CACHE_SURFACE = "marulho_bounded_frozen_language_schedule.v1"


@dataclass(frozen=True)
class OrganismFalsificationConfig:
    token_budget: int = 4_194_304
    relation_fraction: float = 0.20
    sequence_length: int = 72
    batch_size: int = 144
    eval_batches: int = 16
    relation_eval_batch_size: int = 64
    learning_rate: float = 3.0e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    seed: int = 1337
    model_seed: int = 1337
    model_width: int = 512
    model_layers: int = 4
    attention_heads: int = 8
    transformer_mlp_ratio: float = 4.0
    organism_unit_groups: int = 8
    organism_workspace_slots: int = 2
    organism_episodic_slots: int = 16
    organism_state_update_interval: int = 24
    organism_mlp_dim: int = 1592
    organism_counterfactual_rate: float = 0.125
    organism_utility_loss_weight: float = 0.05
    train_sample_bytes_per_source: int = 64 * 1024 * 1024
    eval_sample_bytes_per_source: int = 32 * 1024 * 1024
    sample_range_count: int = 16
    save_candidate_checkpoint: bool = False
    keep_schedule_cache: bool = False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tokenizer_checkpoint(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("surface") != "marulho_transformer_language_checkpoint.v2":
        raise ValueError("Tokenizer source must be a Transformer language checkpoint")
    tokenizer = load_language_tokenizer_state(payload["tokenizer"])
    metadata = dict(payload.get("metadata") or {})
    del payload
    return tokenizer, metadata


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


def sample_corpus_ranges(
    path: str | Path,
    *,
    byte_budget: int,
    range_count: int,
) -> tuple[str, dict[str, Any]]:
    """Read deterministic newline-aligned ranges spread across a corpus."""
    source = Path(path)
    size = source.stat().st_size
    budget = max(1, min(int(byte_budget), size))
    count = max(1, int(range_count))
    if budget >= size:
        data = source.read_bytes()
        text = data.decode("utf-8")
        return text, {
            "path": str(source),
            "source_size_bytes": size,
            "selected_size_bytes": len(data),
            "selected_sha256": hashlib.sha256(data).hexdigest(),
            "ranges": [{"start": 0, "end": size}],
        }

    chunk_size = max(1, budget // count)
    count = max(1, min(count, budget // chunk_size))
    maximum_start = max(0, size - chunk_size)
    nominal_starts = [
        round(index * maximum_start / max(1, count - 1)) for index in range(count)
    ]
    chunks: list[bytes] = []
    ranges: list[dict[str, int]] = []
    with source.open("rb") as handle:
        for nominal_start in nominal_starts:
            handle.seek(int(nominal_start))
            if nominal_start > 0:
                handle.readline()
            start = handle.tell()
            data = handle.read(chunk_size)
            data += handle.readline()
            if not data:
                continue
            end = handle.tell()
            chunks.append(data)
            ranges.append({"start": int(start), "end": int(end)})
    separator = f"\n{LANGUAGE_DOCUMENT_SEPARATOR}\n".encode("utf-8")
    selected = separator.join(chunks)
    text = selected.decode("utf-8")
    return text, {
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
    if relation_batch_count < 1 or not general_batch_counts or any(
        count < 1 for count in general_batch_counts
    ):
        raise ValueError("Matched schedule requires relation and general batches")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    relation_order = torch.randperm(
        relation_batch_count, generator=generator
    ).tolist()
    general_orders = [
        torch.randperm(count, generator=generator).tolist()
        for count in general_batch_counts
    ]
    relation_cursor = 0
    general_cursors = [0] * len(general_orders)
    source_cursor = 0
    accumulator = 0.0
    fraction = min(1.0, max(0.0, float(relation_fraction)))
    rows: list[tuple[str, int]] = []
    for _ in range(max(1, int(step_count))):
        accumulator += fraction
        if accumulator >= 1.0:
            accumulator -= 1.0
            if relation_cursor >= len(relation_order):
                relation_order = torch.randperm(
                    relation_batch_count, generator=generator
                ).tolist()
                relation_cursor = 0
            rows.append(("relation", int(relation_order[relation_cursor])))
            relation_cursor += 1
            continue
        source_index = source_cursor % len(general_orders)
        source_cursor += 1
        order = general_orders[source_index]
        cursor = general_cursors[source_index]
        if cursor >= len(order):
            order = torch.randperm(
                int(general_batch_counts[source_index]), generator=generator
            ).tolist()
            general_orders[source_index] = order
            cursor = 0
        rows.append((f"general_{source_index}", int(order[cursor])))
        general_cursors[source_index] = cursor + 1
    return tuple(rows)


def _schedule_hash(schedule: Sequence[tuple[str, int]]) -> str:
    encoded = json.dumps(list(schedule), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pack_batches(batches: Sequence[LanguageBatch]) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "input_ids": batch.input_ids.detach().cpu(),
            "target_ids": batch.target_ids.detach().cpu(),
        }
        for batch in batches
    ]


def _unpack_batches(
    rows: Sequence[Mapping[str, torch.Tensor]],
) -> tuple[LanguageBatch, ...]:
    return tuple(
        LanguageBatch(
            input_ids=row["input_ids"].to(device="cpu", dtype=torch.long),
            target_ids=row["target_ids"].to(device="cpu", dtype=torch.long),
        )
        for row in rows
    )


def _contract_hash(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(contract), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_cache_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _required_source_batches(
    *,
    step_count: int,
    relation_fraction: float,
    general_source_count: int,
) -> tuple[int, tuple[int, ...]]:
    relation = max(1, math.ceil(step_count * relation_fraction) + 1)
    general_steps = max(1, step_count - math.floor(step_count * relation_fraction))
    per_general = max(1, math.ceil(general_steps / general_source_count) + 1)
    return relation, tuple(per_general for _ in range(general_source_count))


def _load_or_build_schedule(
    *,
    tokenizer,
    relation_corpus: Path,
    general_train: Sequence[Path],
    general_eval: Sequence[Path],
    config: OrganismFalsificationConfig,
    step_count: int,
    cache_path: Path,
) -> dict[str, Any]:
    source_paths = (relation_corpus, *general_train, *general_eval)
    source_metadata = {
        str(path): {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in source_paths
    }
    contract = {
        "tokenizer_hash": tokenizer.vocabulary_hash(),
        "sources": source_metadata,
        "sequence_length": int(config.sequence_length),
        "batch_size": int(config.batch_size),
        "eval_batches": int(config.eval_batches),
        "relation_fraction": float(config.relation_fraction),
        "step_count": int(step_count),
        "seed": int(config.seed),
        "train_sample_bytes_per_source": int(config.train_sample_bytes_per_source),
        "eval_sample_bytes_per_source": int(config.eval_sample_bytes_per_source),
        "sample_range_count": int(config.sample_range_count),
    }
    contract_hash = _contract_hash(contract)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            payload.get("surface") == CACHE_SURFACE
            and payload.get("contract_hash") == contract_hash
        ):
            return {**payload, "cache_hit": True}

    relation_text, relation_selection = sample_corpus_ranges(
        relation_corpus,
        byte_budget=config.train_sample_bytes_per_source,
        range_count=config.sample_range_count,
    )
    train_samples = tuple(
        sample_corpus_ranges(
            path,
            byte_budget=config.train_sample_bytes_per_source,
            range_count=config.sample_range_count,
        )
        for path in general_train
    )
    eval_samples = tuple(
        sample_corpus_ranges(
            path,
            byte_budget=config.eval_sample_bytes_per_source,
            range_count=config.sample_range_count,
        )
        for path in general_eval
    )
    required_relation, required_general = _required_source_batches(
        step_count=step_count,
        relation_fraction=config.relation_fraction,
        general_source_count=len(train_samples),
    )
    relation_split = build_language_model_splits(
        (relation_text,),
        tokenizer,
        sequence_length=config.sequence_length,
        stride=config.sequence_length,
        batch_size=config.batch_size,
        device="cpu",
        max_train_batches=required_relation,
        max_eval_batches=1,
        window_selection="stratified",
    )
    train_splits = tuple(
        build_language_model_splits(
            (text,),
            tokenizer,
            sequence_length=config.sequence_length,
            stride=config.sequence_length,
            batch_size=config.batch_size,
            device="cpu",
            max_train_batches=required_general[index],
            max_eval_batches=1,
            window_selection="stratified",
        )
        for index, (text, _selection) in enumerate(train_samples)
    )
    eval_texts = tuple(text for text, _selection in eval_samples)
    eval_split = build_language_model_splits(
        (eval_texts[0],),
        tokenizer,
        eval_texts=eval_texts,
        sequence_length=config.sequence_length,
        stride=config.sequence_length,
        batch_size=config.batch_size,
        device="cpu",
        max_train_batches=1,
        max_eval_batches=config.eval_batches,
        window_selection="stratified",
    )
    schedule = build_matched_schedule(
        step_count=step_count,
        relation_fraction=config.relation_fraction,
        relation_batch_count=len(relation_split.train),
        general_batch_counts=tuple(len(split.train) for split in train_splits),
        seed=config.seed,
    )
    payload = {
        "surface": CACHE_SURFACE,
        "contract": contract,
        "contract_hash": contract_hash,
        "source_selections": {
            "relation": relation_selection,
            "general_train": [selection for _text, selection in train_samples],
            "general_eval": [selection for _text, selection in eval_samples],
        },
        "relation_batches": _pack_batches(relation_split.train),
        "general_batches": [
            _pack_batches(split.train) for split in train_splits
        ],
        "general_eval_batches": _pack_batches(eval_split.eval),
        "schedule": list(schedule),
        "split_reports": {
            "relation": relation_split.report,
            "general_train": [split.report for split in train_splits],
            "general_eval": eval_split.report,
        },
    }
    _write_cache_atomic(cache_path, payload)
    return {**payload, "cache_hit": False}


def _build_model(
    name: str,
    *,
    vocab_size: int,
    config: OrganismFalsificationConfig,
):
    if name == "transformer":
        return MarulhoLanguageModel(
            LanguageModelConfig(
                vocab_size=vocab_size,
                embedding_dim=config.model_width,
                state_dim=config.model_width,
                state_layers=config.model_layers,
                attention_heads=config.attention_heads,
                transformer_context_length=config.sequence_length,
                transformer_mlp_ratio=config.transformer_mlp_ratio,
            )
        )
    if name != "organism":
        raise ValueError(f"Unknown organism falsification arm: {name}")
    return MarulhoDistributedLanguageModel(
        DistributedLanguageConfig(
            vocab_size=vocab_size,
            width=config.model_width,
            layers=config.model_layers,
            attention_heads=config.attention_heads,
            context_length=config.sequence_length,
            unit_groups=config.organism_unit_groups,
            workspace_slots=config.organism_workspace_slots,
            episodic_slots=config.organism_episodic_slots,
            state_update_interval=config.organism_state_update_interval,
            mlp_dim=config.organism_mlp_dim,
            counterfactual_rate=config.organism_counterfactual_rate,
            utility_loss_weight=config.organism_utility_loss_weight,
        )
    )


def _parameter_inventory(model) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    embedding = int(model.token_embedding.weight.numel())
    return {
        "total_parameters": total,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "embedding_parameters": embedding,
        "non_embedding_parameters": total - embedding,
        "tied_embedding_head": int(
            model.lm_head.weight.data_ptr()
            == model.token_embedding.weight.data_ptr()
        ),
    }


def organism_falsification_decision(
    arms: Sequence[Mapping[str, Any]],
    *,
    durable_budget: int = 16_777_216,
) -> str:
    completed = {
        str(row["name"]): row
        for row in arms
        if row.get("status") == "completed"
    }
    transformer = completed.get("transformer")
    organism = completed.get("organism")
    if transformer is None or organism is None:
        return "incomplete_matched_comparison"
    baseline_parameters = int(transformer["parameters"]["total_parameters"])
    parameter_delta = abs(
        int(organism["parameters"]["total_parameters"]) - baseline_parameters
    ) / max(1, baseline_parameters)
    if parameter_delta > 0.001:
        return "repair_organism_parameter_matching"
    throughput_ratio = float(organism["training"]["tokens_per_second"]) / max(
        float(transformer["training"]["tokens_per_second"]), 1.0e-9
    )
    if throughput_ratio < 0.20:
        return "redesign_organism_execution_before_scaling"
    loss_margin = float(
        organism["general_holdout"]["after"]["heldout_loss"]
    ) - float(transformer["general_holdout"]["after"]["heldout_loss"])
    free_margin = float(organism["relation"]["generation_exact_accuracy"]) - float(
        transformer["relation"]["generation_exact_accuracy"]
    )
    processed = min(
        int(transformer["training"]["processed_tokens"]),
        int(organism["training"]["processed_tokens"]),
    )
    if processed < int(durable_budget):
        if loss_margin <= 0.08 and free_margin >= -0.02:
            return "continue_organism_to_durable_budget_and_unseen_generation"
        return "redesign_or_retire_organism_after_screen"
    if loss_margin <= 0.0 and free_margin >= 0.0:
        return "test_organism_unseen_generation_before_any_promotion"
    return "retire_or_radically_redesign_organism_after_durable_budget"


def _state_bytes(state: Mapping[str, torch.Tensor]) -> int:
    return sum(
        int(tensor.numel()) * int(tensor.element_size())
        for tensor in state.values()
    )


def _run_arm(
    name: str,
    *,
    tokenizer,
    relation_batches: Sequence[LanguageBatch],
    general_batches: Sequence[Sequence[LanguageBatch]],
    general_eval_batches: Sequence[LanguageBatch],
    relation_cases: Sequence[RelationCase],
    schedule: Sequence[tuple[str, int]],
    output_path: Path,
    config: OrganismFalsificationConfig,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(config.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.model_seed)
    model = _build_model(
        name, vocab_size=int(tokenizer.vocab_size), config=config
    ).to(device)
    parameters = _parameter_inventory(model)
    training_config = LanguageTrainingExperimentConfig(
        learning_rate=config.learning_rate,
        minimum_learning_rate_fraction=config.minimum_learning_rate_fraction,
        warmup_fraction=config.warmup_fraction,
        weight_decay=config.weight_decay,
        precision=config.precision,
    )
    optimizer, fused_optimizer = _optimizer(model, training_config)
    general_before = evaluate_language_model(model, general_eval_batches)
    total_steps = len(schedule)
    warmup_steps = int(round(total_steps * config.warmup_fraction))
    losses: list[torch.Tensor] = []
    gradient_norms: list[torch.Tensor] = []
    utility_targets: list[float] = []
    utility_kinds: list[str] = []
    processed_tokens = 0
    relation_tokens = 0
    general_tokens = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    model.train()
    for step_index, (kind, batch_index) in enumerate(schedule):
        if kind == "relation":
            cpu_batch = relation_batches[batch_index]
        else:
            source_index = int(kind.rsplit("_", 1)[1])
            cpu_batch = general_batches[source_index][batch_index]
        batch = cpu_batch.to(device)
        learning_rate = _learning_rate(
            step_index,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            peak=config.learning_rate,
            minimum_fraction=config.minimum_learning_rate_fraction,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with _precision_context(device, config.precision):
            loss_result = model.next_token_loss(
                batch.input_ids,
                batch.target_ids,
                collect_telemetry=False,
                return_evidence=name == "organism",
            )
            loss = loss_result["loss"]
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        optimizer.step()
        token_count = int(batch.target_ids.numel())
        processed_tokens += token_count
        relation_tokens += token_count if kind == "relation" else 0
        general_tokens += token_count if kind != "relation" else 0
        losses.append(loss.detach())
        gradient_norms.append(gradient_norm.detach())
        evidence = dict(loss_result.get("loss_evidence") or {})
        counterfactual = dict(evidence.get("counterfactual") or {})
        if bool(counterfactual.get("ran")):
            utility_targets.append(float(counterfactual["mean_target"]))
            utility_kinds.append(str(counterfactual["kind"]))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    parameters_with_gradient = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    general_after = evaluate_language_model(model, general_eval_batches)
    relation = evaluate_relation_binding_cases_batched(
        model,
        tokenizer,
        relation_cases,
        batch_size=config.relation_eval_batch_size,
    )
    sample = general_batches[0][0].to(device)
    with torch.no_grad(), _precision_context(device, config.precision):
        sample_result = model.forward(sample.input_ids, collect_telemetry=True)
    telemetry = sample_result["telemetry"]
    runtime_state_bytes = _state_bytes(sample_result["state"])

    checkpoint_path: Path | None = None
    if config.save_candidate_checkpoint and name == "organism":
        checkpoint_path = output_path.with_name(
            f"{output_path.stem}-organism-checkpoint.pt"
        )
        metadata = {
            "organism_falsification_report": str(output_path),
            "cumulative_update_tokens": processed_tokens,
            "optimizer_steps": total_steps,
            "training_state": {
                "optimizer_state": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                "schedule": list(schedule),
                "schedule_hash": _schedule_hash(schedule),
                "schedule_cursor": total_steps,
            },
        }
        save_distributed_language_checkpoint(
            checkpoint_path, model, tokenizer, metadata=metadata
        )

    loss_tensor = torch.stack(losses)
    gradient_tensor = torch.stack(gradient_norms)
    return {
        "name": name,
        "status": "completed",
        "model_config": asdict(model.config),
        "parameters": {
            **parameters,
            "parameters_with_gradient_on_final_step": parameters_with_gradient,
        },
        "optimizer": {
            "kind": "AdamW",
            "fused": bool(fused_optimizer),
            "learning_rate": config.learning_rate,
            "minimum_learning_rate_fraction": config.minimum_learning_rate_fraction,
            "warmup_steps": warmup_steps,
            "weight_decay": config.weight_decay,
            "gradient_clip": config.gradient_clip,
            "precision": config.precision,
        },
        "training": {
            "optimizer_steps": total_steps,
            "processed_tokens": processed_tokens,
            "relation_tokens": relation_tokens,
            "general_tokens": general_tokens,
            "mean_loss": float(loss_tensor.mean().cpu()),
            "final_mean_loss": float(
                loss_tensor[-min(16, int(loss_tensor.numel())) :].mean().cpu()
            ),
            "maximum_gradient_norm": float(gradient_tensor.max().cpu()),
            "elapsed_seconds": elapsed,
            "tokens_per_second": processed_tokens / max(elapsed, 1.0e-9),
            "peak_cuda_memory_bytes": peak_memory,
        },
        "utility_credit": {
            "counterfactual_probe_count": len(utility_targets),
            "unit_probe_count": sum(kind == "unit" for kind in utility_kinds),
            "episode_probe_count": sum(kind == "episode" for kind in utility_kinds),
            "mean_target": (
                sum(utility_targets) / len(utility_targets)
                if utility_targets
                else 0.0
            ),
            "mean_absolute_target": (
                sum(abs(value) for value in utility_targets) / len(utility_targets)
                if utility_targets
                else 0.0
            ),
            "positive_target_fraction": (
                sum(value > 0.0 for value in utility_targets) / len(utility_targets)
                if utility_targets
                else 0.0
            ),
        },
        "general_holdout": {
            "before": general_before,
            "after": general_after,
            "loss_delta": float(general_after["heldout_loss"])
            - float(general_before["heldout_loss"]),
        },
        "relation": relation,
        "runtime": {
            "telemetry": telemetry,
            "state_bytes_for_sample_batch": runtime_state_bytes,
            "sample_batch_size": int(sample.input_ids.shape[0]),
            "sample_sequence_length": int(sample.input_ids.shape[1]),
        },
        "checkpoint": (
            None
            if checkpoint_path is None
            else {
                "path": str(checkpoint_path),
                "sha256": _sha256_file(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
                "optimizer_state_available": True,
                "rng_state_available": True,
            }
        ),
    }


def _failed_arm(name: str, exc: BaseException) -> dict[str, Any]:
    return {
        "name": name,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def run_organism_falsification(
    *,
    tokenizer_checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_corpus_paths: Sequence[str | Path],
    general_eval_corpus_paths: Sequence[str | Path],
    output_path: str | Path,
    config: OrganismFalsificationConfig = OrganismFalsificationConfig(),
    schedule_cache_path: str | Path | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    checkpoint = Path(tokenizer_checkpoint_path)
    relation_corpus = Path(relation_corpus_path)
    cases_path = Path(relation_cases_path)
    train_paths = tuple(Path(path) for path in general_train_corpus_paths)
    eval_paths = tuple(Path(path) for path in general_eval_corpus_paths)
    output = Path(output_path)
    if not train_paths or not eval_paths:
        raise ValueError("General train and evaluation corpora are required")
    resolved_device = _resolve_device(device)
    print("[organism] loading checkpoint-owned tokenizer", flush=True)
    tokenizer, checkpoint_metadata = _load_tokenizer_checkpoint(checkpoint)
    cases = _load_cases(cases_path)
    step_count = max(
        1,
        math.ceil(
            config.token_budget
            / max(1, config.batch_size * config.sequence_length)
        ),
    )
    cache_path = (
        Path(schedule_cache_path)
        if schedule_cache_path is not None
        else output.parent / "bounded-organism-language-schedule.pt"
    )
    print("[organism] loading or building bounded frozen schedule", flush=True)
    frozen = _load_or_build_schedule(
        tokenizer=tokenizer,
        relation_corpus=relation_corpus,
        general_train=train_paths,
        general_eval=eval_paths,
        config=config,
        step_count=step_count,
        cache_path=cache_path,
    )
    relation_batches = _unpack_batches(frozen["relation_batches"])
    general_batches = tuple(
        _unpack_batches(rows) for rows in frozen["general_batches"]
    )
    general_eval_batches = _unpack_batches(frozen["general_eval_batches"])
    schedule = tuple((str(kind), int(index)) for kind, index in frozen["schedule"])
    cache_sha256 = _sha256_file(cache_path)
    cache_size_bytes = cache_path.stat().st_size
    print(
        f"[organism] schedule {'cache hit' if frozen['cache_hit'] else 'cached'}: "
        f"{cache_path} ({cache_size_bytes} bytes)",
        flush=True,
    )

    arm_reports: list[dict[str, Any]] = []
    for name in ("transformer", "organism"):
        print(f"[organism] starting arm {name}", flush=True)
        try:
            row = _run_arm(
                name,
                tokenizer=tokenizer,
                relation_batches=relation_batches,
                general_batches=general_batches,
                general_eval_batches=general_eval_batches,
                relation_cases=cases,
                schedule=schedule,
                output_path=output,
                config=config,
                device=resolved_device,
            )
            arm_reports.append(row)
            print(
                f"[organism] completed {name}: "
                f"{row['training']['tokens_per_second']:.1f} tokens/s, "
                f"loss {row['general_holdout']['after']['heldout_loss']:.4f}, "
                f"free {row['relation']['generation_exact_accuracy']:.3f}",
                flush=True,
            )
        except (RuntimeError, ValueError, MemoryError) as exc:
            arm_reports.append(_failed_arm(name, exc))
            print(
                f"[organism] failed {name}: {type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            gc.collect()
            if resolved_device.type == "cuda":
                torch.cuda.empty_cache()

    completed = {
        str(row["name"]): row
        for row in arm_reports
        if row.get("status") == "completed"
    }
    comparisons: dict[str, Any] = {}
    transformer = completed.get("transformer")
    organism = completed.get("organism")
    if transformer is not None and organism is not None:
        comparisons = {
            "parameter_delta_fraction": (
                int(organism["parameters"]["total_parameters"])
                - int(transformer["parameters"]["total_parameters"])
            )
            / max(1, int(transformer["parameters"]["total_parameters"])),
            "training_throughput_ratio_vs_transformer": float(
                organism["training"]["tokens_per_second"]
            )
            / max(float(transformer["training"]["tokens_per_second"]), 1.0e-9),
            "general_loss_margin_vs_transformer": float(
                organism["general_holdout"]["after"]["heldout_loss"]
            )
            - float(transformer["general_holdout"]["after"]["heldout_loss"]),
            "free_relation_accuracy_margin_vs_transformer": float(
                organism["relation"]["generation_exact_accuracy"]
            )
            - float(transformer["relation"]["generation_exact_accuracy"]),
        }
    decision = organism_falsification_decision(arm_reports)
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "configuration": asdict(config),
        "hardware": {
            "device": str(resolved_device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": (
                torch.cuda.get_device_name(resolved_device)
                if resolved_device.type == "cuda"
                else None
            ),
            "torch_version": torch.__version__,
        },
        "tokenizer_source": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "tokenizer_hash": tokenizer.vocabulary_hash(),
            "vocab_size": int(tokenizer.vocab_size),
            "checkpoint_prior_update_tokens": int(
                checkpoint_metadata.get("cumulative_update_tokens") or 0
            ),
            "weights_reused": False,
        },
        "sources": frozen["contract"]["sources"],
        "source_selections": frozen["source_selections"],
        "split_contract": {
            **dict(frozen["split_reports"]),
            "schedule_hash": _schedule_hash(schedule),
            "schedule_step_count": len(schedule),
            "relation_step_count": sum(kind == "relation" for kind, _ in schedule),
            "general_step_count": sum(
                kind.startswith("general_") for kind, _ in schedule
            ),
            "identical_schedule_for_every_arm": True,
            "cache_path": str(cache_path),
            "cache_sha256": cache_sha256,
            "cache_size_bytes": cache_size_bytes,
            "cache_hit": bool(frozen["cache_hit"]),
            "cache_contract_hash": str(frozen["contract_hash"]),
            "cache_retained": bool(config.keep_schedule_cache),
        },
        "relation_cases": {
            "path": str(cases_path),
            "sha256": _sha256_file(cases_path),
            "case_count": len(cases),
            "correct_index_metrics_only": True,
        },
        "architecture_hypothesis": {
            "token_rate": "bounded exact attention plus parallel unit proposals",
            "event_rate": "predictive unit and latent episode update every fixed chunk",
            "event_interval_tokens": config.organism_state_update_interval,
            "communication": "two learned shared workspace slots per layer",
            "credit": "sampled delayed counterfactual future-loss target",
            "slow_learning": "ordinary AdamW weights; consolidation not yet enabled",
            "external_model_code_used": False,
        },
        "arms": arm_reports,
        "comparisons": comparisons,
        "decision": decision,
        "quality_boundary": {
            "human_unseen_generation_review_required": True,
            "promotes_generation_quality_claim": False,
            "promotes_transformer_replacement_claim": False,
            "promotes_runtime_installation": False,
        },
    }
    write_json_report_with_readme(
        output, report, title="MARULHO Distributed Predictive-Organism Falsification"
    )
    if not config.keep_schedule_cache and cache_path.exists():
        cache_path.unlink()
    print(f"[organism] decision {decision}; report {output}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--relation-corpus", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument(
        "--general-train-corpus", action="append", type=Path, required=True
    )
    parser.add_argument(
        "--general-eval-corpus", action="append", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schedule-cache", type=Path)
    parser.add_argument("--token-budget", type=int, default=4_194_304)
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=144)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--relation-eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--counterfactual-rate", type=float, default=0.125)
    parser.add_argument("--state-update-interval", type=int, default=24)
    parser.add_argument("--train-sample-mib", type=int, default=64)
    parser.add_argument("--eval-sample-mib", type=int, default=32)
    parser.add_argument("--sample-range-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--save-candidate-checkpoint", action="store_true")
    parser.add_argument("--keep-schedule-cache", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_organism_falsification(
        tokenizer_checkpoint_path=args.tokenizer_checkpoint,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_corpus_paths=tuple(args.general_train_corpus),
        general_eval_corpus_paths=tuple(args.general_eval_corpus),
        output_path=args.output,
        config=OrganismFalsificationConfig(
            token_budget=max(1, int(args.token_budget)),
            sequence_length=max(2, int(args.sequence_length)),
            batch_size=max(1, int(args.batch_size)),
            eval_batches=max(1, int(args.eval_batches)),
            relation_eval_batch_size=max(1, int(args.relation_eval_batch_size)),
            learning_rate=float(args.learning_rate),
            organism_counterfactual_rate=float(args.counterfactual_rate),
            organism_state_update_interval=max(1, int(args.state_update_interval)),
            train_sample_bytes_per_source=max(1, int(args.train_sample_mib))
            * 1024
            * 1024,
            eval_sample_bytes_per_source=max(1, int(args.eval_sample_mib))
            * 1024
            * 1024,
            sample_range_count=max(1, int(args.sample_range_count)),
            seed=int(args.seed),
            model_seed=(
                int(args.model_seed)
                if args.model_seed is not None
                else int(args.seed)
            ),
            save_candidate_checkpoint=bool(args.save_candidate_checkpoint),
            keep_schedule_cache=bool(args.keep_schedule_cache),
        ),
        schedule_cache_path=args.schedule_cache,
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
