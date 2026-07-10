"""Answer-masked MARULHO post-training with ordinary language replay."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from marulho.data.language_tokenizer import iter_language_corpus_documents
from marulho.evaluation.language_relation_binding_experiment import (
    RelationCase,
    evaluate_relation_binding_cases,
)
from marulho.evaluation.language_training_experiment import (
    LanguageTrainingExperimentConfig,
    _learning_rate,
    _optimizer,
    _parameter_inventory,
    _precision_context,
    _read_corpus,
    _resolve_device,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    LanguageBatch,
    build_language_model_splits,
    evaluate_language_model,
    load_language_model_checkpoint,
    save_language_model_checkpoint,
)


SURFACE = "marulho_answer_masked_post_training.v1"
ARTIFACT_KIND = "marulho_answer_masked_post_training"


@dataclass(frozen=True)
class MaskedAnswerBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    answer_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "MaskedAnswerBatch":
        return MaskedAnswerBatch(
            input_ids=self.input_ids.to(device),
            target_ids=self.target_ids.to(device),
            answer_mask=self.answer_mask.to(device),
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def build_masked_answer_batches(
    corpus_text: str,
    tokenizer,
    *,
    batch_size: int,
    context_length: int,
    max_documents: int,
) -> tuple[tuple[MaskedAnswerBatch, ...], dict[str, Any]]:
    examples: list[tuple[list[int], list[int], list[bool]]] = []
    skipped_without_answer = 0
    skipped_too_long = 0
    for document in iter_language_corpus_documents((corpus_text,)):
        marker_index = document.rfind("Answer:")
        if marker_index < 0:
            skipped_without_answer += 1
            continue
        prompt = document[: marker_index + len("Answer:")]
        answer = document[marker_index + len("Answer:") :].strip()
        if not answer:
            skipped_without_answer += 1
            continue
        prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        answer_ids = tokenizer.encode(
            f" {answer}",
            add_bos=False,
            add_eos=True,
        )
        combined = prompt_ids + answer_ids
        if len(combined) > int(context_length):
            skipped_too_long += 1
            continue
        input_ids = combined[:-1]
        target_ids = combined[1:]
        answer_start = len(prompt_ids) - 1
        answer_mask = [False] * answer_start + [True] * len(answer_ids)
        examples.append((input_ids, target_ids, answer_mask))
        if len(examples) >= max(1, int(max_documents)):
            break
    if not examples:
        raise ValueError("No answer-bearing relation documents were available")
    batches: list[MaskedAnswerBatch] = []
    size = max(1, int(batch_size))
    for start in range(0, len(examples), size):
        rows = examples[start : start + size]
        maximum = max(len(row[0]) for row in rows)
        inputs = torch.full(
            (len(rows), maximum),
            int(tokenizer.pad_id),
            dtype=torch.long,
        )
        targets = torch.full_like(inputs, int(tokenizer.pad_id))
        mask = torch.zeros((len(rows), maximum), dtype=torch.bool)
        for row_index, (input_ids, target_ids, answer_mask) in enumerate(rows):
            length = len(input_ids)
            inputs[row_index, :length] = torch.tensor(input_ids, dtype=torch.long)
            targets[row_index, :length] = torch.tensor(target_ids, dtype=torch.long)
            mask[row_index, :length] = torch.tensor(answer_mask, dtype=torch.bool)
        batches.append(
            MaskedAnswerBatch(
                input_ids=inputs,
                target_ids=targets,
                answer_mask=mask,
            )
        )
    return tuple(batches), {
        "document_count": len(examples),
        "batch_count": len(batches),
        "batch_size": size,
        "answer_token_count": sum(
            int(batch.answer_mask.sum().item()) for batch in batches
        ),
        "processed_target_token_count": sum(
            int(batch.input_ids.numel()) for batch in batches
        ),
        "skipped_without_answer": skipped_without_answer,
        "skipped_too_long": skipped_too_long,
        "padding_loss_masked": True,
        "prompt_loss_masked": True,
    }


def masked_post_training_branch_decision(
    *,
    free_accuracy_after: float,
    candidate_accuracy_after: float,
    general_loss_delta: float,
) -> str:
    if float(general_loss_delta) > 0.15:
        return "answer_masked_post_training_forgets_general_language"
    if float(free_accuracy_after) >= 0.60 and float(candidate_accuracy_after) >= 0.80:
        return "answer_masked_post_training_promising"
    if float(candidate_accuracy_after) >= 0.80:
        return "answer_masking_keeps_ranking_but_not_free_binding"
    return "answer_masked_objective_falsified"


def run_answer_masked_post_training(
    *,
    checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_corpus_paths: Sequence[str | Path],
    general_eval_corpus_paths: Sequence[str | Path],
    output_path: str | Path,
    optimizer_steps: int = 4096,
    relation_batch_fraction: float = 0.50,
    batch_size: int = 16,
    max_relation_documents: int = 100_000,
    learning_rate: float = 5.0e-5,
    seed: int = 20260710,
    device: str = "auto",
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    relation_corpus_file = Path(relation_corpus_path)
    cases_file = Path(relation_cases_path)
    output = Path(output_path)
    general_train_files = tuple(Path(path) for path in general_train_corpus_paths)
    general_eval_files = tuple(Path(path) for path in general_eval_corpus_paths)
    if not general_train_files or not general_eval_files:
        raise ValueError("General replay train and evaluation corpora are required")
    resolved_device = _resolve_device(device)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model, tokenizer, checkpoint_metadata = load_language_model_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    model = model.to(resolved_device)
    cases = _load_cases(cases_file)
    relation_before = evaluate_relation_binding_cases(model, tokenizer, cases)

    relation_text = _read_corpus(relation_corpus_file)
    relation_batches, relation_batch_report = build_masked_answer_batches(
        relation_text,
        tokenizer,
        batch_size=int(batch_size),
        context_length=int(model.config.transformer_context_length),
        max_documents=int(max_relation_documents),
    )
    general_train_texts = tuple(_read_corpus(path) for path in general_train_files)
    general_eval_texts = tuple(_read_corpus(path) for path in general_eval_files)
    general_split = build_language_model_splits(
        general_train_texts,
        tokenizer,
        eval_texts=general_eval_texts,
        sequence_length=256,
        stride=256,
        batch_size=int(batch_size),
        device="cpu",
        max_train_batches=max(1, int(optimizer_steps)),
        max_eval_batches=256,
        window_selection="stratified",
    )
    general_before = evaluate_language_model(model, general_split.eval)

    training_config = LanguageTrainingExperimentConfig(
        learning_rate=float(learning_rate),
        minimum_learning_rate_fraction=0.10,
        warmup_fraction=0.05,
        weight_decay=0.10,
        precision="bfloat16" if resolved_device.type == "cuda" else "float32",
    )
    optimizer, fused = _optimizer(model, training_config)
    training_state = checkpoint_metadata.get("training_state")
    if not isinstance(training_state, Mapping) or not isinstance(
        training_state.get("optimizer_state"), Mapping
    ):
        raise ValueError(
            "Answer-masked post-training requires a checkpoint with exact optimizer state"
        )
    optimizer.load_state_dict(dict(training_state["optimizer_state"]))
    total_steps = max(1, int(optimizer_steps))
    warmup_steps = int(round(total_steps * 0.05))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    relation_order = torch.randperm(
        len(relation_batches), generator=generator
    ).tolist()
    general_order = torch.randperm(
        len(general_split.train), generator=generator
    ).tolist()
    relation_cursor = 0
    general_cursor = 0
    relation_fraction = min(1.0, max(0.0, float(relation_batch_fraction)))
    relation_accumulator = 0.0
    relation_loss_tokens = 0
    relation_processed_tokens = 0
    general_loss_tokens = 0
    relation_step_count = 0
    general_step_count = 0
    loss_rows: list[dict[str, float]] = []
    curve: list[dict[str, Any]] = []
    milestones = {
        max(1, total_steps // 4),
        max(1, total_steps // 2),
        total_steps,
    }
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    started = time.perf_counter()
    model.train()
    for step in range(total_steps):
        relation_accumulator += relation_fraction
        use_relation = relation_accumulator >= 1.0
        if use_relation:
            relation_accumulator -= 1.0
        lr = _learning_rate(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            peak=float(learning_rate),
            minimum_fraction=0.10,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        if use_relation:
            if relation_cursor >= len(relation_order):
                relation_order = torch.randperm(
                    len(relation_batches), generator=generator
                ).tolist()
                relation_cursor = 0
            batch = relation_batches[relation_order[relation_cursor]].to(
                resolved_device
            )
            relation_cursor += 1
            with _precision_context(resolved_device, training_config.precision):
                output_row = model.forward(
                    batch.input_ids,
                    collect_telemetry=False,
                )
                token_losses = F.cross_entropy(
                    output_row["logits"].reshape(-1, output_row["logits"].shape[-1]),
                    batch.target_ids.reshape(-1),
                    reduction="none",
                )
                flat_mask = batch.answer_mask.reshape(-1)
                loss = token_losses[flat_mask].mean()
            supervised = int(batch.answer_mask.sum().item())
            relation_loss_tokens += supervised
            relation_processed_tokens += int(batch.target_ids.numel())
            relation_step_count += 1
            kind = "answer_masked_relation"
            loss_token_count = supervised
        else:
            if general_cursor >= len(general_order):
                general_order = torch.randperm(
                    len(general_split.train), generator=generator
                ).tolist()
                general_cursor = 0
            general_batch: LanguageBatch = general_split.train[
                general_order[general_cursor]
            ].to(resolved_device)
            general_cursor += 1
            with _precision_context(resolved_device, training_config.precision):
                result = model.next_token_loss(
                    general_batch.input_ids,
                    general_batch.target_ids,
                    collect_telemetry=False,
                    return_evidence=False,
                )
                loss = result["loss"]
            loss_token_count = int(general_batch.target_ids.numel())
            general_loss_tokens += loss_token_count
            general_step_count += 1
            kind = "general_replay"
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_rows.append(
            {
                "loss": float(loss.detach().cpu().item()),
                "gradient_norm": float(gradient_norm.detach().cpu().item()),
                "loss_token_count": float(loss_token_count),
                "relation_batch": 1.0 if use_relation else 0.0,
            }
        )
        completed_steps = step + 1
        if completed_steps in milestones:
            if resolved_device.type == "cuda":
                torch.cuda.synchronize(resolved_device)
            evaluation = evaluate_language_model(model, general_split.eval)
            recent = loss_rows[-min(128, len(loss_rows)) :]
            curve.append(
                {
                    "optimizer_steps": completed_steps,
                    "general_heldout_loss": float(evaluation["heldout_loss"]),
                    "general_heldout_perplexity": float(
                        evaluation["heldout_perplexity"]
                    ),
                    "relation_loss_tokens": relation_loss_tokens,
                    "relation_processed_tokens": relation_processed_tokens,
                    "general_loss_tokens": general_loss_tokens,
                    "relation_step_count": relation_step_count,
                    "general_step_count": general_step_count,
                    "recent_mean_loss": sum(row["loss"] for row in recent)
                    / len(recent),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            model.train()

    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    training_elapsed = time.perf_counter() - started
    peak_cuda_memory_bytes = (
        int(torch.cuda.max_memory_allocated(resolved_device))
        if resolved_device.type == "cuda"
        else 0
    )
    relation_after = evaluate_relation_binding_cases(model, tokenizer, cases)
    general_after = evaluate_language_model(model, general_split.eval)
    prior_update_tokens = int(checkpoint_metadata.get("cumulative_update_tokens") or 0)
    prior_optimizer_steps = int(
        checkpoint_metadata.get("cumulative_optimizer_steps") or 0
    )
    loss_bearing_tokens = relation_loss_tokens + general_loss_tokens
    checkpoint_output = output.with_name(f"{output.stem}-checkpoint.pt")
    save_language_model_checkpoint(
        checkpoint_output,
        model,
        tokenizer,
        metadata={
            "answer_masked_post_training_report": str(output),
            "cumulative_update_tokens": prior_update_tokens + loss_bearing_tokens,
            "cumulative_optimizer_steps": prior_optimizer_steps + total_steps,
            "training_state": {
                "optimizer_state": optimizer.state_dict(),
                "generator_state": generator.get_state(),
                "relation_batch_order": relation_order,
                "relation_batch_cursor": relation_cursor,
                "general_batch_order": general_order,
                "general_batch_cursor": general_cursor,
                "general_train_split_hash": general_split.report["train_split_hash"],
            },
        },
    )
    general_loss_delta = float(general_after["heldout_loss"]) - float(
        general_before["heldout_loss"]
    )
    decision = masked_post_training_branch_decision(
        free_accuracy_after=float(relation_after["generation_exact_accuracy"]),
        candidate_accuracy_after=float(relation_after["accuracy"]),
        general_loss_delta=general_loss_delta,
    )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "checkpoint": {
            "input_path": str(checkpoint),
            "input_sha256": _sha256_file(checkpoint),
            "output_path": str(checkpoint_output),
            "output_sha256": _sha256_file(checkpoint_output),
            "output_size_bytes": checkpoint_output.stat().st_size,
            "tokenizer_hash": tokenizer.vocabulary_hash(),
            "optimizer_state_restored": True,
            "prior_update_tokens": prior_update_tokens,
            "cumulative_update_tokens": prior_update_tokens + loss_bearing_tokens,
            "prior_optimizer_steps": prior_optimizer_steps,
            "cumulative_optimizer_steps": prior_optimizer_steps + total_steps,
        },
        "configuration": {
            "optimizer_steps": total_steps,
            "relation_batch_fraction": relation_fraction,
            "batch_size": int(batch_size),
            "max_relation_documents": int(max_relation_documents),
            "learning_rate": float(learning_rate),
            "precision": training_config.precision,
            "fused_optimizer": bool(fused),
            "warmup_steps": warmup_steps,
            "seed": int(seed),
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
        "relation_batch_report": relation_batch_report,
        "training": {
            "curve": curve,
            "relation_loss_tokens": relation_loss_tokens,
            "relation_processed_tokens": relation_processed_tokens,
            "general_loss_tokens": general_loss_tokens,
            "loss_bearing_tokens": loss_bearing_tokens,
            "processed_tokens": relation_processed_tokens + general_loss_tokens,
            "relation_step_count": relation_step_count,
            "general_step_count": general_step_count,
            "training_and_milestone_eval_elapsed_seconds": training_elapsed,
            "parameter_inventory": _parameter_inventory(model),
            "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
        },
        "general_holdout": {
            "before": general_before,
            "after": general_after,
            "loss_delta": general_loss_delta,
        },
        "relation_before": relation_before,
        "relation_after": relation_after,
        "decision": decision,
        "quality_boundary": {
            "promotes_generation_quality_claim": False,
            "promotes_runtime_claim": False,
            "candidate_promotable": False,
            "requires_open_general_prompt_review": True,
        },
    }
    write_json_report_with_readme(
        output,
        report,
        title="MARULHO Answer-Masked Post-Training",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--relation-corpus", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--general-train-corpus", action="append", type=Path, required=True)
    parser.add_argument("--general-eval-corpus", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimizer-steps", type=int, default=4096)
    parser.add_argument("--relation-batch-fraction", type=float, default=0.50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-relation-documents", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_answer_masked_post_training(
        checkpoint_path=args.checkpoint,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_corpus_paths=tuple(args.general_train_corpus),
        general_eval_corpus_paths=tuple(args.general_eval_corpus),
        output_path=args.output,
        optimizer_steps=max(1, int(args.optimizer_steps)),
        relation_batch_fraction=float(args.relation_batch_fraction),
        batch_size=max(1, int(args.batch_size)),
        max_relation_documents=max(1, int(args.max_relation_documents)),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
