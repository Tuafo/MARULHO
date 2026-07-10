"""Matched Transformer versus editable delta-memory language falsification."""

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

from marulho.data.language_tokenizer import load_language_tokenizer_state
from marulho.evaluation.language_relation_binding_experiment import (
    RelationCase,
    evaluate_relation_binding_cases_batched,
)
from marulho.evaluation.language_training_experiment import (
    LanguageTrainingExperimentConfig,
    _learning_rate,
    _optimizer,
    _precision_context,
    _read_corpus,
    _resolve_device,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_delta import (
    DeltaLanguageConfig,
    MarulhoDeltaLanguageModel,
)
from marulho.training.language_model import (
    LanguageBatch,
    LanguageModelConfig,
    MarulhoLanguageModel,
    build_language_model_splits,
    evaluate_language_model,
)


SURFACE = "marulho_delta_falsification.v1"
ARTIFACT_KIND = "marulho_delta_falsification"
CACHE_SURFACE = "marulho_frozen_language_schedule.v1"


@dataclass(frozen=True)
class DeltaFalsificationArm:
    name: str
    architecture: str
    local_attention_every: int = 0


@dataclass(frozen=True)
class DeltaFalsificationConfig:
    token_budget: int = 1_048_576
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
    model_width: int = 512
    model_layers: int = 4
    attention_heads: int = 8
    transformer_mlp_ratio: float = 4.0
    delta_memory_heads: int = 8
    delta_memory_head_dim: int = 32
    delta_mlp_dim: int = 2048


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


def parse_delta_arm(value: str) -> DeltaFalsificationArm:
    name = str(value).strip().lower()
    arms = {
        "transformer": DeltaFalsificationArm("transformer", "transformer"),
        "delta": DeltaFalsificationArm("delta", "delta"),
        "delta-hybrid": DeltaFalsificationArm(
            "delta-hybrid", "delta", local_attention_every=4
        ),
        "delta-hybrid-half": DeltaFalsificationArm(
            "delta-hybrid-half", "delta", local_attention_every=2
        ),
    }
    if name not in arms:
        raise ValueError(f"Unknown delta falsification arm: {value}")
    return arms[name]


def _build_model(
    arm: DeltaFalsificationArm,
    *,
    vocab_size: int,
    config: DeltaFalsificationConfig,
):
    if arm.architecture == "transformer":
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
    if arm.architecture != "delta":
        raise ValueError(f"Unknown architecture: {arm.architecture}")
    return MarulhoDeltaLanguageModel(
        DeltaLanguageConfig(
            vocab_size=vocab_size,
            width=config.model_width,
            layers=config.model_layers,
            memory_heads=config.delta_memory_heads,
            memory_head_dim=config.delta_memory_head_dim,
            attention_heads=config.attention_heads,
            local_attention_every=arm.local_attention_every,
            context_length=config.sequence_length,
            mlp_dim=config.delta_mlp_dim,
        )
    )


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


def _unpack_batches(rows: Sequence[Mapping[str, torch.Tensor]]) -> tuple[LanguageBatch, ...]:
    return tuple(
        LanguageBatch(
            input_ids=row["input_ids"].to(device="cpu", dtype=torch.long),
            target_ids=row["target_ids"].to(device="cpu", dtype=torch.long),
        )
        for row in rows
    )


def _cache_contract(
    *,
    tokenizer_hash: str,
    relation_corpus: Path,
    general_train: Sequence[Path],
    general_eval: Sequence[Path],
    config: DeltaFalsificationConfig,
    step_count: int,
) -> dict[str, Any]:
    return {
        "tokenizer_hash": tokenizer_hash,
        "relation_corpus_sha256": _sha256_file(relation_corpus),
        "general_train_sha256": [_sha256_file(path) for path in general_train],
        "general_eval_sha256": [_sha256_file(path) for path in general_eval],
        "sequence_length": int(config.sequence_length),
        "batch_size": int(config.batch_size),
        "eval_batches": int(config.eval_batches),
        "relation_fraction": float(config.relation_fraction),
        "step_count": int(step_count),
        "seed": int(config.seed),
    }


def _contract_hash(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(contract), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_cache_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_or_build_schedule(
    *,
    tokenizer,
    relation_corpus: Path,
    general_train: Sequence[Path],
    general_eval: Sequence[Path],
    config: DeltaFalsificationConfig,
    step_count: int,
    cache_path: Path,
) -> dict[str, Any]:
    contract = _cache_contract(
        tokenizer_hash=tokenizer.vocabulary_hash(),
        relation_corpus=relation_corpus,
        general_train=general_train,
        general_eval=general_eval,
        config=config,
        step_count=step_count,
    )
    contract_hash = _contract_hash(contract)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            payload.get("surface") == CACHE_SURFACE
            and payload.get("contract_hash") == contract_hash
        ):
            return {**payload, "cache_hit": True}

    relation_split = build_language_model_splits(
        (_read_corpus(relation_corpus),),
        tokenizer,
        sequence_length=config.sequence_length,
        stride=config.sequence_length,
        batch_size=config.batch_size,
        device="cpu",
        max_train_batches=step_count,
        max_eval_batches=1,
        window_selection="stratified",
    )
    train_splits = tuple(
        build_language_model_splits(
            (_read_corpus(path),),
            tokenizer,
            sequence_length=config.sequence_length,
            stride=config.sequence_length,
            batch_size=config.batch_size,
            device="cpu",
            max_train_batches=step_count,
            max_eval_batches=1,
            window_selection="stratified",
        )
        for path in general_train
    )
    eval_texts = tuple(_read_corpus(path) for path in general_eval)
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


def delta_falsification_decision(
    arms: Sequence[Mapping[str, Any]],
    *,
    minimum_finalist_tokens: int = 4_194_304,
) -> str:
    completed = {
        str(row["name"]): row
        for row in arms
        if row.get("status") == "completed"
    }
    transformer = completed.get("transformer")
    candidates = [
        row for name, row in completed.items() if name.startswith("delta")
    ]
    if transformer is None or not candidates:
        return "incomplete_matched_comparison"
    finite = [
        row
        for row in candidates
        if math.isfinite(float(row["general_holdout"]["after"]["heldout_loss"]))
    ]
    if not finite:
        return "retire_delta_numerical_failure"
    best = min(finite, key=lambda row: row["general_holdout"]["after"]["heldout_loss"])
    margin = float(best["general_holdout"]["after"]["heldout_loss"]) - float(
        transformer["general_holdout"]["after"]["heldout_loss"]
    )
    token_count = min(
        int(best["training"]["processed_tokens"]),
        int(transformer["training"]["processed_tokens"]),
    )
    parameter_delta = abs(
        int(best["parameters"]["total_parameters"])
        - int(transformer["parameters"]["total_parameters"])
    ) / max(1, int(transformer["parameters"]["total_parameters"]))
    if parameter_delta > 0.001:
        return "repair_parameter_matching"
    if token_count < int(minimum_finalist_tokens):
        return (
            f"continue_{best['name']}_to_next_budget"
            if margin <= 0.15
            else f"redesign_or_retire_{best['name']}_after_screen"
        )
    if margin <= 0.05:
        return f"scale_{best['name']}_and_test_unseen_generation"
    if margin > 0.20:
        return "retire_editable_delta_base_model"
    return f"repeat_{best['name']}_near_branch_boundary"


def _run_arm(
    arm: DeltaFalsificationArm,
    *,
    tokenizer,
    relation_batches: Sequence[LanguageBatch],
    general_batches: Sequence[Sequence[LanguageBatch]],
    general_eval_batches: Sequence[LanguageBatch],
    relation_cases: Sequence[RelationCase],
    schedule: Sequence[tuple[str, int]],
    config: DeltaFalsificationConfig,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = _build_model(
        arm, vocab_size=int(tokenizer.vocab_size), config=config
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
    processed_tokens = 0
    relation_tokens = 0
    general_tokens = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()
    for step_index, (kind, batch_index) in enumerate(schedule):
        source = (
            relation_batches
            if kind == "relation"
            else general_batches[int(kind.removeprefix("general_"))]
        )
        batch = source[int(batch_index)].to(device)
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
            loss = model.next_token_loss(
                batch.input_ids,
                batch.target_ids,
                collect_telemetry=False,
                return_evidence=False,
            )["loss"]
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
        telemetry = model.forward(
            sample.input_ids, collect_telemetry=True
        )["telemetry"]
    loss_tensor = torch.stack(losses)
    gradient_tensor = torch.stack(gradient_norms)
    return {
        "name": arm.name,
        "status": "completed",
        "arm": asdict(arm),
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
        "general_holdout": {
            "before": general_before,
            "after": general_after,
            "loss_delta": float(general_after["heldout_loss"])
            - float(general_before["heldout_loss"]),
        },
        "relation": relation,
        "runtime_telemetry": telemetry,
        "checkpoint": None,
    }


def _failed_arm(
    arm: DeltaFalsificationArm, exc: BaseException
) -> dict[str, Any]:
    return {
        "name": arm.name,
        "status": "failed",
        "arm": asdict(arm),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def run_delta_falsification(
    *,
    tokenizer_checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_corpus_paths: Sequence[str | Path],
    general_eval_corpus_paths: Sequence[str | Path],
    output_path: str | Path,
    arms: Sequence[DeltaFalsificationArm],
    config: DeltaFalsificationConfig = DeltaFalsificationConfig(),
    schedule_cache_path: str | Path | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    if not arms or len({arm.name for arm in arms}) != len(arms):
        raise ValueError("Arms must be non-empty and uniquely named")
    checkpoint = Path(tokenizer_checkpoint_path)
    relation_corpus = Path(relation_corpus_path)
    cases_path = Path(relation_cases_path)
    train_paths = tuple(Path(path) for path in general_train_corpus_paths)
    eval_paths = tuple(Path(path) for path in general_eval_corpus_paths)
    output = Path(output_path)
    if not train_paths or not eval_paths:
        raise ValueError("General train and evaluation corpora are required")
    resolved_device = _resolve_device(device)
    print("[delta] loading checkpoint-owned tokenizer", flush=True)
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
        else output.parent / "frozen-language-schedule.pt"
    )
    print("[delta] loading or building frozen schedule", flush=True)
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
    print(
        f"[delta] frozen schedule {'cache hit' if frozen['cache_hit'] else 'cached'}: "
        f"{cache_path}",
        flush=True,
    )

    arm_reports: list[dict[str, Any]] = []
    for arm in arms:
        print(f"[delta] starting arm {arm.name}", flush=True)
        try:
            row = _run_arm(
                arm,
                tokenizer=tokenizer,
                relation_batches=relation_batches,
                general_batches=general_batches,
                general_eval_batches=general_eval_batches,
                relation_cases=cases,
                schedule=schedule,
                config=config,
                device=resolved_device,
            )
            arm_reports.append(row)
            print(
                f"[delta] completed {arm.name}: "
                f"{row['training']['tokens_per_second']:.1f} tokens/s, "
                f"loss {row['general_holdout']['after']['heldout_loss']:.4f}, "
                f"free {row['relation']['generation_exact_accuracy']:.3f}",
                flush=True,
            )
        except (RuntimeError, ValueError, MemoryError) as exc:
            arm_reports.append(_failed_arm(arm, exc))
            print(
                f"[delta] failed {arm.name}: {type(exc).__name__}: {exc}",
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
    transformer = completed.get("transformer")
    comparisons: dict[str, Any] = {}
    if transformer is not None:
        for name, row in completed.items():
            if name == "transformer":
                continue
            comparisons[name] = {
                "parameter_delta_fraction": (
                    int(row["parameters"]["total_parameters"])
                    - int(transformer["parameters"]["total_parameters"])
                )
                / max(1, int(transformer["parameters"]["total_parameters"])),
                "training_throughput_ratio_vs_transformer": float(
                    row["training"]["tokens_per_second"]
                )
                / max(
                    float(transformer["training"]["tokens_per_second"]), 1.0e-9
                ),
                "general_loss_margin_vs_transformer": float(
                    row["general_holdout"]["after"]["heldout_loss"]
                )
                - float(
                    transformer["general_holdout"]["after"]["heldout_loss"]
                ),
                "free_relation_accuracy_margin_vs_transformer": float(
                    row["relation"]["generation_exact_accuracy"]
                )
                - float(transformer["relation"]["generation_exact_accuracy"]),
            }
    decision = delta_falsification_decision(arm_reports)
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
        "sources": {
            "relation_corpus": {
                "path": str(relation_corpus),
                "sha256": _sha256_file(relation_corpus),
            },
            "relation_cases": {
                "path": str(cases_path),
                "sha256": _sha256_file(cases_path),
                "case_count": len(cases),
                "correct_index_metrics_only": True,
            },
            "general_train": [
                {"path": str(path), "sha256": _sha256_file(path)}
                for path in train_paths
            ],
            "general_eval": [
                {"path": str(path), "sha256": _sha256_file(path)}
                for path in eval_paths
            ],
        },
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
            "cache_sha256": _sha256_file(cache_path),
            "cache_hit": bool(frozen["cache_hit"]),
            "cache_contract_hash": str(frozen["contract_hash"]),
        },
        "architecture_hypothesis": {
            "state": "fixed recurrent fast-weight matrix per head",
            "decay": "channel-wise",
            "erase_write": "independent channel-wise gates",
            "hybrid": "every fourth mixer is bounded local attention",
            "lcwm_transfer": "execution-coupled structured memory deferred until base quality",
            "external_model_code_used": False,
        },
        "arms": arm_reports,
        "comparisons": comparisons,
        "success_criteria": {
            "minimum_finalist_tokens": 4_194_304,
            "maximum_general_loss_margin_for_next_budget": 0.15,
            "maximum_parameter_delta_fraction": 0.001,
            "synthetic_result_alone_promotable": False,
        },
        "decision": decision,
        "quality_boundary": {
            "screening_run": config.token_budget < 4_194_304,
            "promotes_generation_quality_claim": False,
            "promotes_transformer_replacement_claim": False,
            "promotes_runtime_installation": False,
        },
    }
    write_json_report_with_readme(
        output, report, title="MARULHO Editable Delta-Memory Falsification"
    )
    print(f"[delta] decision {decision}; report {output}", flush=True)
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
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--token-budget", type=int, default=1_048_576)
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=144)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--relation-eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    arm_names = args.arm or ["transformer", "delta", "delta-hybrid"]
    run_delta_falsification(
        tokenizer_checkpoint_path=args.tokenizer_checkpoint,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_corpus_paths=tuple(args.general_train_corpus),
        general_eval_corpus_paths=tuple(args.general_eval_corpus),
        output_path=args.output,
        arms=tuple(parse_delta_arm(value) for value in arm_names),
        config=DeltaFalsificationConfig(
            token_budget=max(1, int(args.token_budget)),
            sequence_length=max(2, int(args.sequence_length)),
            batch_size=max(1, int(args.batch_size)),
            eval_batches=max(1, int(args.eval_batches)),
            relation_eval_batch_size=max(1, int(args.relation_eval_batch_size)),
            learning_rate=float(args.learning_rate),
            seed=int(args.seed),
        ),
        schedule_cache_path=args.schedule_cache,
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
