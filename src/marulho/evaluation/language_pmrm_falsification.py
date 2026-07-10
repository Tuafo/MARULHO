"""Matched Transformer versus integrated PMRM language falsification."""

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
from marulho.training.language_model import (
    LanguageBatch,
    LanguageModelConfig,
    MarulhoLanguageModel,
    build_language_model_splits,
    evaluate_language_model,
    save_language_model_checkpoint,
)
from marulho.training.language_pmrm import (
    PMRMLanguageConfig,
    MarulhoPMRMLanguageModel,
    save_pmrm_language_checkpoint,
)


SURFACE = "marulho_pmrm_falsification.v1"
ARTIFACT_KIND = "marulho_pmrm_falsification"


@dataclass(frozen=True)
class PMRMFalsificationArm:
    name: str
    architecture: str
    fusion_kind: str = "dual_parallel"
    episodic_policy: str = "surprise"
    workspace_iterations: int = 2
    relation_messages: bool = True


@dataclass(frozen=True)
class PMRMFalsificationConfig:
    token_budget: int = 1_048_576
    relation_fraction: float = 0.20
    sequence_length: int = 64
    batch_size: int = 160
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
    transformer_layers: int = 4
    attention_heads: int = 8
    transformer_mlp_ratio: float = 4.0
    column_count: int = 8
    active_columns: int = 2
    associative_dim: int = 64
    episodic_slots: int = 16
    episodic_reads: int = 2
    workspace_registers: int = 2
    workspace_layers: int = 3
    workspace_mlp_dim: int = 1712
    save_checkpoints: bool = False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tokenizer_checkpoint(path: Path):
    payload = torch.load(path, map_location="cpu")
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


def parse_pmrm_arm(value: str) -> PMRMFalsificationArm:
    name = str(value).strip().lower()
    arms = {
        "transformer": PMRMFalsificationArm("transformer", "transformer"),
        "pmrm-surprise": PMRMFalsificationArm(
            "pmrm-surprise", "pmrm", episodic_policy="surprise"
        ),
        "pmrm-none": PMRMFalsificationArm(
            "pmrm-none", "pmrm", episodic_policy="none"
        ),
        "pmrm-random": PMRMFalsificationArm(
            "pmrm-random", "pmrm", episodic_policy="random"
        ),
        "pmrm-recency": PMRMFalsificationArm(
            "pmrm-recency", "pmrm", episodic_policy="recency"
        ),
        "pmrm-temporal": PMRMFalsificationArm(
            "pmrm-temporal",
            "pmrm",
            fusion_kind="temporal_only",
            episodic_policy="surprise",
        ),
        "pmrm-associative": PMRMFalsificationArm(
            "pmrm-associative",
            "pmrm",
            fusion_kind="associative_only",
            episodic_policy="surprise",
        ),
        "pmrm-workspace1": PMRMFalsificationArm(
            "pmrm-workspace1",
            "pmrm",
            episodic_policy="surprise",
            workspace_iterations=1,
        ),
    }
    if name not in arms:
        raise ValueError(f"Unknown PMRM falsification arm: {value}")
    return arms[name]


def _build_model(
    arm: PMRMFalsificationArm,
    *,
    vocab_size: int,
    config: PMRMFalsificationConfig,
):
    width = int(config.model_width)
    if arm.architecture == "transformer":
        return MarulhoLanguageModel(
            LanguageModelConfig(
                vocab_size=int(vocab_size),
                embedding_dim=width,
                state_dim=width,
                state_layers=int(config.transformer_layers),
                attention_heads=int(config.attention_heads),
                transformer_context_length=int(config.sequence_length),
                transformer_mlp_ratio=float(config.transformer_mlp_ratio),
            )
        )
    if arm.architecture != "pmrm":
        raise ValueError(f"Unknown architecture: {arm.architecture}")
    return MarulhoPMRMLanguageModel(
        PMRMLanguageConfig(
            vocab_size=int(vocab_size),
            embedding_dim=width,
            state_dim=width,
            column_count=int(config.column_count),
            active_columns=int(config.active_columns),
            associative_dim=int(config.associative_dim),
            fusion_kind=str(arm.fusion_kind),
            relation_messages=bool(arm.relation_messages),
            episodic_policy=str(arm.episodic_policy),
            episodic_slots=int(config.episodic_slots),
            episodic_reads=int(config.episodic_reads),
            workspace_registers=int(config.workspace_registers),
            workspace_layers=int(config.workspace_layers),
            workspace_iterations=int(arm.workspace_iterations),
            workspace_mlp_dim=int(config.workspace_mlp_dim),
            context_length=int(config.sequence_length),
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
        raise ValueError("Matched schedule requires both relation and general batches")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    relation_order = torch.randperm(relation_batch_count, generator=generator).tolist()
    general_orders = [
        torch.randperm(count, generator=generator).tolist()
        for count in general_batch_counts
    ]
    relation_cursor = 0
    general_cursors = [0] * len(general_orders)
    general_source_cursor = 0
    accumulator = 0.0
    fraction = min(1.0, max(0.0, float(relation_fraction)))
    rows: list[tuple[str, int]] = []
    for _ in range(max(1, int(step_count))):
        accumulator += fraction
        use_relation = accumulator >= 1.0
        if use_relation:
            accumulator -= 1.0
            if relation_cursor >= len(relation_order):
                relation_order = torch.randperm(
                    relation_batch_count, generator=generator
                ).tolist()
                relation_cursor = 0
            rows.append(("relation", int(relation_order[relation_cursor])))
            relation_cursor += 1
        else:
            source_index = general_source_cursor % len(general_orders)
            general_source_cursor += 1
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
    payload = json.dumps(list(schedule), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parameter_inventory(model) -> dict[str, int]:
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "embedding_parameters": int(model.token_embedding.weight.numel()),
        "non_embedding_parameters": sum(parameter.numel() for parameter in model.parameters())
        - int(model.token_embedding.weight.numel()),
        "tied_embedding_head": int(
            model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
        ),
    }


def pmrm_falsification_decision(
    arms: Sequence[Mapping[str, Any]],
    *,
    minimum_finalist_tokens: int = 4_194_304,
) -> str:
    completed = {str(row["name"]): row for row in arms if row.get("status") == "completed"}
    if "transformer" not in completed or "pmrm-surprise" not in completed:
        return "incomplete_matched_pair"
    transformer = completed["transformer"]
    pmrm = completed["pmrm-surprise"]
    if not math.isfinite(float(pmrm["general_holdout"]["after"]["heldout_loss"])):
        return "repair_pmrm_numerical_instability"
    transformer_free = float(transformer["relation"]["generation_exact_accuracy"])
    free_accuracy = float(pmrm["relation"]["generation_exact_accuracy"])
    general_margin = float(pmrm["general_holdout"]["after"]["heldout_loss"]) - float(
        transformer["general_holdout"]["after"]["heldout_loss"]
    )
    parameter_delta = abs(
        int(pmrm["parameters"]["total_parameters"])
        - int(transformer["parameters"]["total_parameters"])
    ) / max(1, int(transformer["parameters"]["total_parameters"]))
    token_count = min(
        int(pmrm["training"]["processed_tokens"]),
        int(transformer["training"]["processed_tokens"]),
    )
    if (
        token_count >= int(minimum_finalist_tokens)
        and free_accuracy >= 0.60
        and free_accuracy >= transformer_free + 0.10
        and general_margin <= 0.15
        and parameter_delta <= 0.005
    ):
        return "scale_integrated_pmrm"
    if general_margin > 0.50 and free_accuracy <= transformer_free:
        return "pmrm_quality_dominated_at_screening"
    return "continue_successive_halving_or_redesign_systems_path"


def _run_arm(
    arm: PMRMFalsificationArm,
    *,
    tokenizer,
    relation_batches: Sequence[LanguageBatch],
    general_batches: Sequence[Sequence[LanguageBatch]],
    general_eval_batches: Sequence[LanguageBatch],
    relation_cases: Sequence[RelationCase],
    schedule: Sequence[tuple[str, int]],
    output_path: Path,
    config: PMRMFalsificationConfig,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(int(config.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.seed))
    model = _build_model(
        arm, vocab_size=int(tokenizer.vocab_size), config=config
    ).to(device)
    parameters = _parameter_inventory(model)
    training_config = LanguageTrainingExperimentConfig(
        learning_rate=float(config.learning_rate),
        minimum_learning_rate_fraction=float(
            config.minimum_learning_rate_fraction
        ),
        warmup_fraction=float(config.warmup_fraction),
        weight_decay=float(config.weight_decay),
        precision=str(config.precision),
    )
    optimizer, fused_optimizer = _optimizer(model, training_config)
    general_before = evaluate_language_model(model, general_eval_batches)
    total_steps = len(schedule)
    warmup_steps = int(round(total_steps * float(config.warmup_fraction)))
    losses: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    processed_tokens = 0
    relation_tokens = 0
    general_tokens = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()
    for step_index, (kind, batch_index) in enumerate(schedule):
        if kind == "relation":
            source = relation_batches
        else:
            source_index = int(kind.removeprefix("general_"))
            source = general_batches[source_index]
        batch = source[int(batch_index)].to(device)
        lr = _learning_rate(
            step_index,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            peak=float(config.learning_rate),
            minimum_fraction=float(config.minimum_learning_rate_fraction),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with _precision_context(device, str(config.precision)):
            result = model.next_token_loss(
                batch.input_ids,
                batch.target_ids,
                collect_telemetry=False,
                return_evidence=False,
            )
            loss = result["loss"]
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config.gradient_clip)
        )
        optimizer.step()
        token_count = int(batch.target_ids.numel())
        processed_tokens += token_count
        if kind == "relation":
            relation_tokens += token_count
        else:
            general_tokens += token_count
        losses.append(loss.detach())
        gradients.append(gradient_norm.detach())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_elapsed = time.perf_counter() - started
    peak_cuda_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    loss_tensor = torch.stack(losses)
    gradient_tensor = torch.stack(gradients)
    training_loss = float(loss_tensor.mean().cpu().item())
    final_training_loss = float(
        loss_tensor[-min(16, int(loss_tensor.numel())) :].mean().cpu().item()
    )
    maximum_gradient = float(gradient_tensor.max().cpu().item())
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
        batch_size=int(config.relation_eval_batch_size),
    )
    sample = general_batches[0][0].to(device)
    with torch.no_grad(), _precision_context(device, str(config.precision)):
        telemetry = model.forward(
            sample.input_ids, collect_telemetry=True
        )["telemetry"]
    checkpoint_path: Path | None = None
    if bool(config.save_checkpoints):
        checkpoint_path = output_path.with_name(
            f"{output_path.stem}-{arm.name}-checkpoint.pt"
        )
        metadata = {
            "pmrm_falsification_report": str(output_path),
            "arm": asdict(arm),
            "processed_tokens": processed_tokens,
            "optimizer_steps": total_steps,
            "training_state": {"optimizer_state": optimizer.state_dict()},
        }
        if arm.architecture == "transformer":
            save_language_model_checkpoint(
                checkpoint_path, model, tokenizer, metadata=metadata
            )
        else:
            save_pmrm_language_checkpoint(
                checkpoint_path, model, tokenizer, metadata=metadata
            )
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
            "learning_rate": float(config.learning_rate),
            "minimum_learning_rate_fraction": float(
                config.minimum_learning_rate_fraction
            ),
            "warmup_steps": warmup_steps,
            "weight_decay": float(config.weight_decay),
            "gradient_clip": float(config.gradient_clip),
            "precision": str(config.precision),
        },
        "training": {
            "optimizer_steps": total_steps,
            "processed_tokens": processed_tokens,
            "relation_tokens": relation_tokens,
            "general_tokens": general_tokens,
            "mean_loss": training_loss,
            "final_mean_loss": final_training_loss,
            "maximum_gradient_norm": maximum_gradient,
            "elapsed_seconds": training_elapsed,
            "tokens_per_second": float(processed_tokens)
            / max(training_elapsed, 1.0e-9),
            "peak_cuda_memory_bytes": peak_cuda_memory,
        },
        "general_holdout": {
            "before": general_before,
            "after": general_after,
            "loss_delta": float(general_after["heldout_loss"])
            - float(general_before["heldout_loss"]),
        },
        "relation": relation,
        "runtime_telemetry": telemetry,
        "checkpoint": (
            None
            if checkpoint_path is None
            else {
                "path": str(checkpoint_path),
                "sha256": _sha256_file(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
            }
        ),
    }


def _failed_arm(arm: PMRMFalsificationArm, exc: BaseException) -> dict[str, Any]:
    return {
        "name": arm.name,
        "status": "failed",
        "arm": asdict(arm),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def run_pmrm_falsification(
    *,
    tokenizer_checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_corpus_paths: Sequence[str | Path],
    general_eval_corpus_paths: Sequence[str | Path],
    output_path: str | Path,
    arms: Sequence[PMRMFalsificationArm],
    config: PMRMFalsificationConfig = PMRMFalsificationConfig(),
    device: str = "auto",
) -> dict[str, Any]:
    if not arms:
        raise ValueError("At least one falsification arm is required")
    if len({arm.name for arm in arms}) != len(arms):
        raise ValueError("Falsification arm names must be unique")
    checkpoint = Path(tokenizer_checkpoint_path)
    relation_corpus_file = Path(relation_corpus_path)
    cases_file = Path(relation_cases_path)
    general_train_files = tuple(Path(path) for path in general_train_corpus_paths)
    general_eval_files = tuple(Path(path) for path in general_eval_corpus_paths)
    output = Path(output_path)
    if not general_train_files or not general_eval_files:
        raise ValueError("General train and evaluation corpora are required")
    resolved_device = _resolve_device(device)
    print("[pmrm] loading checkpoint-owned tokenizer", flush=True)
    tokenizer, checkpoint_metadata = _load_tokenizer_checkpoint(checkpoint)
    cases = _load_cases(cases_file)
    step_count = max(
        1,
        math.ceil(
            int(config.token_budget)
            / max(1, int(config.batch_size) * int(config.sequence_length))
        ),
    )
    print("[pmrm] building frozen relation split", flush=True)
    relation_text = _read_corpus(relation_corpus_file)
    relation_split = build_language_model_splits(
        (relation_text,),
        tokenizer,
        sequence_length=int(config.sequence_length),
        stride=int(config.sequence_length),
        batch_size=int(config.batch_size),
        device="cpu",
        max_train_batches=step_count,
        max_eval_batches=1,
        window_selection="stratified",
    )
    print("[pmrm] building source-balanced general splits", flush=True)
    general_train_texts = tuple(_read_corpus(path) for path in general_train_files)
    general_eval_texts = tuple(_read_corpus(path) for path in general_eval_files)
    general_train_splits = tuple(
        build_language_model_splits(
            (text,),
            tokenizer,
            sequence_length=int(config.sequence_length),
            stride=int(config.sequence_length),
            batch_size=int(config.batch_size),
            device="cpu",
            max_train_batches=step_count,
            max_eval_batches=1,
            window_selection="stratified",
        )
        for text in general_train_texts
    )
    general_eval_split = build_language_model_splits(
        (general_eval_texts[0],),
        tokenizer,
        eval_texts=general_eval_texts,
        sequence_length=int(config.sequence_length),
        stride=int(config.sequence_length),
        batch_size=int(config.batch_size),
        device="cpu",
        max_train_batches=1,
        max_eval_batches=int(config.eval_batches),
        window_selection="stratified",
    )
    schedule = build_matched_schedule(
        step_count=step_count,
        relation_fraction=float(config.relation_fraction),
        relation_batch_count=len(relation_split.train),
        general_batch_counts=tuple(
            len(split.train) for split in general_train_splits
        ),
        seed=int(config.seed),
    )

    arm_reports: list[dict[str, Any]] = []
    for arm in arms:
        print(f"[pmrm] starting arm {arm.name}", flush=True)
        try:
            arm_report = _run_arm(
                    arm,
                    tokenizer=tokenizer,
                    relation_batches=relation_split.train,
                    general_batches=tuple(
                        split.train for split in general_train_splits
                    ),
                    general_eval_batches=general_eval_split.eval,
                    relation_cases=cases,
                    schedule=schedule,
                    output_path=output,
                    config=config,
                    device=resolved_device,
                )
            arm_reports.append(arm_report)
            print(
                f"[pmrm] completed {arm.name}: "
                f"{arm_report['training']['tokens_per_second']:.1f} tokens/s, "
                f"loss {arm_report['general_holdout']['after']['heldout_loss']:.4f}, "
                f"free {arm_report['relation']['generation_exact_accuracy']:.3f}",
                flush=True,
            )
        except (RuntimeError, ValueError, MemoryError) as exc:
            arm_reports.append(_failed_arm(arm, exc))
            print(f"[pmrm] failed {arm.name}: {type(exc).__name__}: {exc}", flush=True)
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
    if transformer is not None:
        transformer_parameters = int(
            transformer["parameters"]["total_parameters"]
        )
        transformer_throughput = float(
            transformer["training"]["tokens_per_second"]
        )
        for name, row in completed.items():
            if name == "transformer":
                continue
            comparisons[name] = {
                "parameter_delta_fraction": (
                    int(row["parameters"]["total_parameters"])
                    - transformer_parameters
                )
                / max(1, transformer_parameters),
                "training_throughput_ratio_vs_transformer": float(
                    row["training"]["tokens_per_second"]
                )
                / max(transformer_throughput, 1.0e-9),
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
    decision = pmrm_falsification_decision(arm_reports)
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
                "path": str(relation_corpus_file),
                "sha256": _sha256_file(relation_corpus_file),
            },
            "relation_cases": {
                "path": str(cases_file),
                "sha256": _sha256_file(cases_file),
                "case_count": len(cases),
                "correct_index_metrics_only": True,
            },
            "general_train": [
                {"path": str(path), "sha256": _sha256_file(path)}
                for path in general_train_files
            ],
            "general_eval": [
                {"path": str(path), "sha256": _sha256_file(path)}
                for path in general_eval_files
            ],
        },
        "split_contract": {
            "relation": relation_split.report,
            "general_train": [split.report for split in general_train_splits],
            "general_eval": general_eval_split.report,
            "schedule_hash": _schedule_hash(schedule),
            "schedule_step_count": len(schedule),
            "relation_step_count": sum(kind == "relation" for kind, _ in schedule),
            "general_step_count": sum(
                kind.startswith("general_") for kind, _ in schedule
            ),
            "general_source_step_counts": {
                str(index): sum(
                    kind == f"general_{index}" for kind, _ in schedule
                )
                for index in range(len(general_train_splits))
            },
            "identical_schedule_for_every_arm": True,
        },
        "arms": arm_reports,
        "comparisons": comparisons,
        "success_criteria": {
            "minimum_finalist_tokens": 4_194_304,
            "minimum_free_relation_accuracy": 0.60,
            "minimum_free_accuracy_margin_vs_transformer": 0.10,
            "maximum_general_loss_margin_vs_transformer": 0.15,
            "maximum_parameter_delta_fraction": 0.005,
            "surprise_must_beat_equal_budget_random_and_recency": True,
            "synthetic_result_alone_promotable": False,
        },
        "decision": decision,
        "quality_boundary": {
            "screening_run": int(config.token_budget) < 4_194_304,
            "promotes_generation_quality_claim": False,
            "promotes_transformer_replacement_claim": False,
            "promotes_runtime_installation": False,
        },
    }
    write_json_report_with_readme(
        output,
        report,
        title="MARULHO Integrated PMRM Falsification",
    )
    print(f"[pmrm] decision {decision}; report {output}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--relation-corpus", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--general-train-corpus", action="append", type=Path, required=True)
    parser.add_argument("--general-eval-corpus", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--token-budget", type=int, default=1_048_576)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--relation-eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    arm_names = args.arm or ["transformer", "pmrm-surprise", "pmrm-none"]
    run_pmrm_falsification(
        tokenizer_checkpoint_path=args.tokenizer_checkpoint,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_corpus_paths=tuple(args.general_train_corpus),
        general_eval_corpus_paths=tuple(args.general_eval_corpus),
        output_path=args.output,
        arms=tuple(parse_pmrm_arm(value) for value in arm_names),
        config=PMRMFalsificationConfig(
            token_budget=max(1, int(args.token_budget)),
            sequence_length=max(2, int(args.sequence_length)),
            batch_size=max(1, int(args.batch_size)),
            eval_batches=max(1, int(args.eval_batches)),
            relation_eval_batch_size=max(1, int(args.relation_eval_batch_size)),
            learning_rate=float(args.learning_rate),
            seed=int(args.seed),
            save_checkpoints=bool(args.save_checkpoints),
        ),
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
