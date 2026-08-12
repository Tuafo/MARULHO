"""Run V57 native long-context versus oracle-localized falsification."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from marulho.evaluation.language_matched_support import (
    build_document_aligned_batches,
    full_sized_batches,
    sample_corpus_ranges,
    sha256_file,
)
from marulho.evaluation.language_relation_binding_experiment import (
    RelationCase,
    evaluate_relation_binding_cases_batched,
)
from marulho.evaluation.language_source_grounding import (
    _contains_answer,
    load_squad_grounding_manifest,
)
from marulho.evaluation.language_source_grounding_continual import (
    _stratified_relation_cases,
)
from marulho.evaluation.language_training_experiment import _learning_rate
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    LanguageBatch,
    MarulhoLanguageModel,
    build_language_model_splits,
    evaluate_language_model,
    language_model_state_sha256,
    load_language_model_checkpoint,
    save_language_model_checkpoint,
)
from marulho.training.language_muon import (
    build_language_muon,
    warm_language_muon_orthogonalizer_shapes,
)


SURFACE = "marulho_native_context_falsification.v1"
ARCHITECTURE = "v39_transformer_native_context320_full_model_continuation"
ARM_NAMES = ("oracle_short", "native_full")


@dataclass(frozen=True)
class NativeContextFalsificationConfig:
    context_length: int = 320
    batch_size: int = 32
    grounding_epochs: int = 4
    optimizer_steps: int = 2_048
    padded_position_budget_per_arm: int = 20_971_520
    grounding_fraction: float = 0.50
    general_fraction: float = 0.25
    relation_fraction: float = 0.25
    answer_weight: float = 4.0
    learning_rate: float = 3.0e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    execution_backend: str = "pytorch_eager"
    data_seed: int = 57121
    model_seed: int = 57131
    sample_bytes_per_replay_source: int = 16 * 1024 * 1024
    sample_bytes_per_eval_source: int = 1 * 1024 * 1024
    sample_range_count: int = 16
    general_eval_batches: int = 16
    relation_case_count: int = 64
    relation_eval_batch_size: int = 8
    relation_generation_tokens: int = 16
    grounding_generation_tokens: int = 16
    maximum_training_seconds_per_arm: float = 1_800.0
    minimum_oracle_answer_count: int = 128
    minimum_native_answer_count: int = 128
    minimum_native_source_gain: float = 0.45
    maximum_native_oracle_gap: int = 16
    maximum_mismatched_answer_count: int = 16
    maximum_general_loss_regression: float = 0.10
    maximum_relation_generation_regression: float = 0.05


@dataclass(frozen=True)
class WeightedLanguageBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    answer_mask: torch.Tensor

    def to(self, device: torch.device | str) -> WeightedLanguageBatch:
        return WeightedLanguageBatch(
            input_ids=self.input_ids.to(device),
            target_ids=self.target_ids.to(device),
            answer_mask=self.answer_mask.to(device),
        )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _tensor_sha256(values: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build_native_grounding_batches(
    manifest: Mapping[str, Any],
    tokenizer,
    *,
    arm: str,
    sequence_length: int,
    batch_size: int,
) -> tuple[tuple[WeightedLanguageBatch, ...], dict[str, Any]]:
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown V57 arm: {arm}")
    cases = tuple(dict(value) for value in manifest["cases"])
    if not cases or len(cases) % int(batch_size):
        raise ValueError("V57 grounding cases must divide into complete batches")
    input_rows = []
    target_rows = []
    answer_rows = []
    encoded_lengths = []
    answer_lengths = []
    prefix_key = "causal_prompt" if arm == "native_full" else "oracle_causal_prompt"
    for case in cases:
        prefix = tokenizer.encode(
            str(case[prefix_key]), add_bos=True, add_eos=False
        )
        answer = tokenizer.encode(
            str(case["answers"][0]), add_bos=False, add_eos=True
        )
        ids = prefix + answer
        if len(ids) - 1 > int(sequence_length):
            raise ValueError("V57 causal record exceeds its context")
        row = torch.full(
            (int(sequence_length) + 1,),
            int(tokenizer.pad_id),
            dtype=torch.long,
        )
        row[: len(ids)] = torch.tensor(ids, dtype=torch.long)
        answer_mask = torch.zeros(int(sequence_length), dtype=torch.bool)
        answer_mask[len(prefix) - 1 : len(ids) - 1] = True
        input_rows.append(row[:-1])
        target_rows.append(row[1:])
        answer_rows.append(answer_mask)
        encoded_lengths.append(len(ids))
        answer_lengths.append(len(answer) - 1)
    inputs = torch.stack(input_rows)
    targets = torch.stack(target_rows)
    answers = torch.stack(answer_rows)
    batches = tuple(
        WeightedLanguageBatch(
            input_ids=inputs[index : index + int(batch_size)],
            target_ids=targets[index : index + int(batch_size)],
            answer_mask=answers[index : index + int(batch_size)],
        )
        for index in range(0, len(cases), int(batch_size))
    )
    return batches, {
        "surface": "marulho_native_context_grounding_batches.v1",
        "arm": arm,
        "case_count": len(cases),
        "batch_count": len(batches),
        "batch_size": int(batch_size),
        "sequence_length": int(sequence_length),
        "minimum_encoded_tokens": min(encoded_lengths),
        "maximum_encoded_tokens": max(encoded_lengths),
        "minimum_answer_tokens": min(answer_lengths),
        "maximum_answer_tokens": max(answer_lengths),
        "input_target_answer_sha256": _tensor_sha256((inputs, targets, answers)),
        "all_records_fit": True,
        "answer_mask_includes_eos": True,
        "right_padding_only_after_eos": True,
    }


def build_native_context_schedule(
    *,
    grounding_batch_count: int,
    relation_batch_count: int,
    general_batch_counts: Sequence[int],
    grounding_epochs: int,
    seed: int,
) -> tuple[tuple[tuple[str, int], ...], dict[str, Any]]:
    if len(general_batch_counts) != 2:
        raise ValueError("V57 requires exactly two general replay sources")
    grounding_steps = int(grounding_batch_count) * int(grounding_epochs)
    if grounding_steps % int(grounding_epochs):
        raise ValueError("V57 grounding schedule is not epoch-aligned")
    relation_steps = grounding_steps // 2
    general_steps_each = grounding_steps // 4
    if relation_steps > int(relation_batch_count) or any(
        general_steps_each > int(value) for value in general_batch_counts
    ):
        raise ValueError("V57 replay sources lack unique batches")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    relation_order = torch.randperm(
        int(relation_batch_count), generator=generator
    ).tolist()[:relation_steps]
    general_orders = [
        torch.randperm(int(count), generator=generator).tolist()[:general_steps_each]
        for count in general_batch_counts
    ]
    schedule: list[tuple[str, int]] = []
    relation_cursor = 0
    general_cursors = [0, 0]
    relation_per_epoch = relation_steps // int(grounding_epochs)
    general_per_epoch = general_steps_each // int(grounding_epochs)
    for _epoch in range(int(grounding_epochs)):
        grounding_order = torch.randperm(
            int(grounding_batch_count), generator=generator
        ).tolist()
        replay = [
            ("relation", index)
            for index in relation_order[
                relation_cursor : relation_cursor + relation_per_epoch
            ]
        ]
        relation_cursor += relation_per_epoch
        for source_index in range(2):
            replay.extend(
                (f"general_{source_index}", index)
                for index in general_orders[source_index][
                    general_cursors[source_index] : general_cursors[source_index]
                    + general_per_epoch
                ]
            )
            general_cursors[source_index] += general_per_epoch
        if len(replay) != len(grounding_order):
            raise ValueError("V57 epoch replay and grounding counts differ")
        replay_order = torch.randperm(len(replay), generator=generator).tolist()
        for grounding_index, replay_index in zip(
            grounding_order, replay_order, strict=True
        ):
            schedule.append(("grounding", int(grounding_index)))
            schedule.append(replay[replay_index])
    kind_code = {"grounding": 0, "relation": 1, "general_0": 2, "general_1": 3}
    encoded = torch.tensor(
        [(kind_code[kind] << 32) | int(index) for kind, index in schedule],
        dtype=torch.int64,
    )
    counts = Counter(kind for kind, _index in schedule)
    return tuple(schedule), {
        "surface": "marulho_native_context_schedule.v1",
        "step_count": len(schedule),
        "kind_counts": dict(sorted(counts.items())),
        "grounding_epochs": int(grounding_epochs),
        "grounding_batch_count": int(grounding_batch_count),
        "all_grounding_batches_once_per_epoch": True,
        "all_replay_batches_unique": True,
        "sha256": hashlib.sha256(encoded.numpy().tobytes()).hexdigest(),
    }


def _relation_cases(path: str | Path, *, case_count: int) -> tuple[RelationCase, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = tuple(
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
    return _stratified_relation_cases(cases, case_count=int(case_count))


def _prepare_replay(
    tokenizer,
    *,
    relation_replay_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    config: NativeContextFalsificationConfig,
) -> tuple[
    tuple[LanguageBatch, ...],
    tuple[tuple[LanguageBatch, ...], ...],
    tuple[LanguageBatch, ...],
    dict[str, Any],
]:
    relation_text, relation_selection = sample_corpus_ranges(
        relation_replay_path,
        byte_budget=int(config.sample_bytes_per_replay_source),
        range_count=int(config.sample_range_count),
    )
    relation_batches, relation_alignment = build_document_aligned_batches(
        (relation_text,),
        tokenizer,
        sequence_length=int(config.context_length),
        batch_size=int(config.batch_size),
    )
    general_batches = []
    general_selections = []
    general_reports = []
    for path in general_train_paths:
        text, selection = sample_corpus_ranges(
            path,
            byte_budget=int(config.sample_bytes_per_replay_source),
            range_count=int(config.sample_range_count),
        )
        split = build_language_model_splits(
            (text,),
            tokenizer,
            sequence_length=int(config.context_length),
            stride=int(config.context_length),
            batch_size=int(config.batch_size),
            max_train_batches=512,
            max_eval_batches=1,
        )
        batches = full_sized_batches(
            split.train, batch_size=int(config.batch_size)
        )
        general_batches.append(batches)
        general_selections.append(selection)
        general_reports.append(dict(split.report))
    eval_texts = []
    eval_selections = []
    for path in general_eval_paths:
        text, selection = sample_corpus_ranges(
            path,
            byte_budget=int(config.sample_bytes_per_eval_source),
            range_count=8,
        )
        eval_texts.append(text)
        eval_selections.append(selection)
    eval_split = build_language_model_splits(
        (),
        tokenizer,
        eval_texts=eval_texts,
        sequence_length=72,
        stride=72,
        batch_size=8,
        max_train_batches=1,
        max_eval_batches=int(config.general_eval_batches),
    )
    return (
        relation_batches,
        tuple(general_batches),
        tuple(eval_split.eval),
        {
            "relation": relation_selection,
            "relation_alignment": relation_alignment,
            "general_train": general_selections,
            "general_train_splits": general_reports,
            "general_eval": eval_selections,
            "general_eval_split": dict(eval_split.report),
        },
    )


def _selected_batch(
    kind: str,
    index: int,
    *,
    grounding_batches: Sequence[WeightedLanguageBatch],
    relation_batches: Sequence[LanguageBatch],
    general_batches: Sequence[Sequence[LanguageBatch]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if kind == "grounding":
        batch = grounding_batches[int(index)].to(device)
        return batch.input_ids, batch.target_ids, batch.answer_mask
    if kind == "relation":
        ordinary = relation_batches[int(index)].to(device)
    else:
        source = int(kind.rsplit("_", 1)[1])
        ordinary = general_batches[source][int(index)].to(device)
    return (
        ordinary.input_ids,
        ordinary.target_ids,
        torch.zeros_like(ordinary.target_ids, dtype=torch.bool),
    )


def _weighted_loss(
    model: MarulhoLanguageModel,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    answer_mask: torch.Tensor,
    *,
    pad_id: int,
    answer_weight: float,
) -> torch.Tensor:
    logits = model(input_ids, collect_telemetry=False)["logits"].float()
    losses = F.cross_entropy(
        logits.flatten(0, 1), target_ids.flatten(), reduction="none"
    ).reshape_as(target_ids)
    valid = target_ids.ne(int(pad_id))
    weights = valid.to(losses.dtype) * (
        1.0
        + answer_mask.to(losses.dtype) * (float(answer_weight) - 1.0)
    )
    return (losses * weights).sum() / weights.sum().clamp_min(1)


def _train_arm(
    model: MarulhoLanguageModel,
    *,
    grounding_batches: Sequence[WeightedLanguageBatch],
    relation_batches: Sequence[LanguageBatch],
    general_batches: Sequence[Sequence[LanguageBatch]],
    schedule: Sequence[tuple[str, int]],
    schedule_report: Mapping[str, Any],
    tokenizer,
    config: NativeContextFalsificationConfig,
) -> dict[str, Any]:
    trainable = list(model.named_parameters())
    shape_counts = Counter(
        tuple(int(value) for value in parameter.shape)
        for name, parameter in trainable
        if parameter.ndim == 2
        and not name.startswith("token_embedding.")
        and not name.startswith("lm_head.")
    )
    optimizer_warmup = warm_language_muon_orthogonalizer_shapes(
        (
            (int(count), int(shape[0]), int(shape[1]))
            for shape, count in shape_counts.items()
        ),
        device=model.device,
    )
    optimizer, optimizer_report = build_language_muon(
        model,
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        adamw_betas=(0.9, 0.95),
    )
    total_steps = len(schedule)
    if total_steps != int(config.optimizer_steps):
        raise ValueError("V57 optimizer step count differs from preregistration")
    processed_positions = (
        total_steps * int(config.batch_size) * int(config.context_length)
    )
    if processed_positions != int(config.padded_position_budget_per_arm):
        raise ValueError("V57 padded position budget differs")
    warmup_steps = max(1, math.ceil(total_steps * float(config.warmup_fraction)))
    losses_by_kind: dict[str, list[float]] = {
        "grounding": [],
        "relation": [],
        "general_0": [],
        "general_1": [],
    }
    trace = []
    nonpad_tokens = 0
    model.train()
    torch.cuda.reset_peak_memory_stats(model.device)
    torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    for step, (kind, index) in enumerate(schedule):
        input_ids, target_ids, answer_mask = _selected_batch(
            kind,
            index,
            grounding_batches=grounding_batches,
            relation_batches=relation_batches,
            general_batches=general_batches,
            device=model.device,
        )
        optimizer.zero_grad(set_to_none=True)
        learning_rate = _learning_rate(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            peak=float(config.learning_rate),
            minimum_fraction=float(config.minimum_learning_rate_fraction),
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = _weighted_loss(
                model,
                input_ids,
                target_ids,
                answer_mask,
                pad_id=int(tokenizer.pad_id),
                answer_weight=float(config.answer_weight),
            )
        if not bool(torch.isfinite(loss).item()):
            raise ValueError(f"V57 loss became non-finite at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config.gradient_clip)
        )
        optimizer.step()
        value = float(loss.detach().item())
        losses_by_kind[kind].append(value)
        nonpad_tokens += int(target_ids.ne(int(tokenizer.pad_id)).sum().item())
        if step in {
            0,
            warmup_steps - 1,
            total_steps // 4,
            total_steps // 2,
            3 * total_steps // 4,
            total_steps - 1,
        }:
            trace.append(
                {
                    "step": step + 1,
                    "kind": kind,
                    "loss": value,
                    "learning_rate": learning_rate,
                    "gradient_norm": float(gradient_norm.detach().item()),
                }
            )
        if (step + 1) % 128 == 0 or step + 1 == total_steps:
            print(
                f"[native-v57] step {step + 1}/{total_steps} "
                f"kind={kind} loss={value:.4f}",
                flush=True,
            )
    torch.cuda.synchronize(model.device)
    training_seconds = time.perf_counter() - started
    final_gradient = {
        name: parameter.grad is not None for name, parameter in trainable
    }
    final_nonzero_gradient = {
        name: parameter.grad is not None
        and bool(parameter.grad.detach().ne(0).any().item())
        for name, parameter in trainable
    }
    return {
        "architecture": ARCHITECTURE,
        "execution_backend": str(config.execution_backend),
        "optimizer": optimizer_report,
        "optimizer_warmup": optimizer_warmup,
        "optimizer_steps": total_steps,
        "processed_padded_positions": processed_positions,
        "processed_nonpad_tokens": nonpad_tokens,
        "training_seconds": training_seconds,
        "positions_per_second": processed_positions / training_seconds,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(model.device),
        "warmup_steps": warmup_steps,
        "schedule": dict(schedule_report),
        "initial_loss_by_kind": {
            kind: values[0] for kind, values in losses_by_kind.items()
        },
        "final_loss_by_kind": {
            kind: values[-1] for kind, values in losses_by_kind.items()
        },
        "mean_loss_by_kind": {
            kind: sum(values) / len(values) for kind, values in losses_by_kind.items()
        },
        "trace": trace,
        "gradient_audit_policy": (
            "final_optimizer_step_only; a final gradient proves receipt during training"
        ),
        "all_parameters_received_gradient": all(final_gradient.values()),
        "all_parameters_received_nonzero_gradient": all(
            final_nonzero_gradient.values()
        ),
        "all_parameters_received_final_gradient": all(final_gradient.values()),
        "all_parameters_received_final_nonzero_gradient": all(
            final_nonzero_gradient.values()
        ),
        "missing_gradient_parameters": [
            name for name, seen in final_gradient.items() if not seen
        ],
        "zero_gradient_parameters": [
            name for name, seen in final_nonzero_gradient.items() if not seen
        ],
        "final_missing_gradient_parameters": [
            name for name, seen in final_gradient.items() if not seen
        ],
        "final_zero_gradient_parameters": [
            name for name, seen in final_nonzero_gradient.items() if not seen
        ],
    }


def _generate_prompts(
    model: MarulhoLanguageModel,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
) -> list[str]:
    encoded = [tokenizer.encode(prompt, add_eos=False) for prompt in prompts]
    groups: dict[int, list[int]] = {}
    for index, ids in enumerate(encoded):
        groups.setdefault(len(ids), []).append(index)
    outputs = [""] * len(prompts)
    model.eval()
    for length, indices in groups.items():
        batch = torch.tensor(
            [encoded[index] for index in indices],
            device=model.device,
            dtype=torch.long,
        )
        generated = model.generate(
            batch,
            max_new_tokens=int(max_new_tokens),
            eos_id=int(tokenizer.eos_id),
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )["generated_ids"].detach().cpu()
        for row, case_index in enumerate(indices):
            outputs[case_index] = tokenizer.decode(
                [int(value) for value in generated[row, length:].tolist()]
            )
    return outputs


def _condition_report(
    cases: Sequence[Mapping[str, Any]], continuations: Sequence[str]
) -> dict[str, Any]:
    rows = []
    for case, continuation in zip(cases, continuations, strict=True):
        answers = tuple(str(value) for value in case["answers"])
        rows.append(
            {
                "case_id": str(case["case_id"]),
                "answers": list(answers),
                "continuation": str(continuation),
                "exact_answer_match": _contains_answer(continuation, answers),
            }
        )
    count = sum(bool(row["exact_answer_match"]) for row in rows)
    return {
        "case_count": len(rows),
        "exact_answer_count": count,
        "exact_answer_accuracy": count / max(1, len(rows)),
        "rows": rows,
    }


@torch.no_grad()
def _evaluate_grounding(
    model: MarulhoLanguageModel,
    tokenizer,
    manifest: Mapping[str, Any],
    *,
    primary: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    if primary not in ARM_NAMES:
        raise ValueError("unknown V57 grounding evaluation view")
    cases = tuple(dict(value) for value in manifest["cases"])
    primary_key = "causal_prompt" if primary == "native_full" else "oracle_causal_prompt"
    cross_key = "oracle_causal_prompt" if primary == "native_full" else "causal_prompt"
    state_before = language_model_state_sha256(model)
    primary_rows = _generate_prompts(
        model,
        tokenizer,
        [str(case[primary_key]) for case in cases],
        max_new_tokens=max_new_tokens,
    )
    cross_rows = _generate_prompts(
        model,
        tokenizer,
        [str(case[cross_key]) for case in cases],
        max_new_tokens=max_new_tokens,
    )
    question_rows = _generate_prompts(
        model,
        tokenizer,
        [f'{str(case["question_only_prompt"])} ' for case in cases],
        max_new_tokens=max_new_tokens,
    )
    mismatch_rows = _generate_prompts(
        model,
        tokenizer,
        [f'{str(case["mismatched_prompt"])} ' for case in cases],
        max_new_tokens=max_new_tokens,
    )
    conditions = {
        "primary": _condition_report(cases, primary_rows),
        "cross_view": _condition_report(cases, cross_rows),
        "question_only": _condition_report(cases, question_rows),
        "mismatched_source": _condition_report(cases, mismatch_rows),
    }
    primary_accuracy = float(conditions["primary"]["exact_answer_accuracy"])
    stronger_control = max(
        float(conditions["question_only"]["exact_answer_accuracy"]),
        float(conditions["mismatched_source"]["exact_answer_accuracy"]),
    )
    validity = {
        "all_answers_visible_in_full_source": all(
            _contains_answer(str(case["source_text"]), tuple(case["answers"]))
            for case in cases
        ),
        "all_answers_visible_in_oracle_source": all(
            _contains_answer(
                str(case["oracle_source_text"]), tuple(case["answers"])
            )
            for case in cases
        ),
        "all_answers_absent_from_question": all(
            not _contains_answer(str(case["question"]), tuple(case["answers"]))
            for case in cases
        ),
        "all_answers_absent_from_mismatch": all(
            not _contains_answer(
                str(case["mismatched_source_text"]), tuple(case["answers"])
            )
            for case in cases
        ),
        "model_state_immutable": language_model_state_sha256(model) == state_before,
    }
    return {
        "primary_view": primary,
        "cross_view": "oracle_short" if primary == "native_full" else "native_full",
        "valid": all(validity.values()),
        "validity": validity,
        "primary_gain_over_stronger_control": primary_accuracy - stronger_control,
        "conditions": conditions,
    }


def native_context_gate(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    baseline: Mapping[str, Any],
    parent: Mapping[str, Any],
    config: NativeContextFalsificationConfig,
    parameter_count: int,
) -> dict[str, Any]:
    native = dict(arms["native_full"])
    oracle = dict(arms["oracle_short"])
    native_source = dict(native["source_grounding"])
    oracle_source = dict(oracle["source_grounding"])
    native_count = int(native_source["conditions"]["primary"]["exact_answer_count"])
    oracle_count = int(oracle_source["conditions"]["primary"]["exact_answer_count"])
    mismatch_count = int(
        native_source["conditions"]["mismatched_source"]["exact_answer_count"]
    )
    general_regression = float(native["general"]["heldout_loss"]) - float(
        baseline["general"]["heldout_loss"]
    )
    relation_regression = float(
        baseline["relation"]["generation_exact_accuracy"]
    ) - float(native["relation"]["generation_exact_accuracy"])
    checks = {
        "source_reports_valid": bool(native_source["valid"])
        and bool(oracle_source["valid"]),
        "minimum_oracle_answer_count": oracle_count
        >= int(config.minimum_oracle_answer_count),
        "minimum_native_answer_count": native_count
        >= int(config.minimum_native_answer_count),
        "minimum_native_source_gain": float(
            native_source["primary_gain_over_stronger_control"]
        )
        >= float(config.minimum_native_source_gain),
        "maximum_native_oracle_gap": oracle_count - native_count
        <= int(config.maximum_native_oracle_gap),
        "maximum_mismatched_answer_count": mismatch_count
        <= int(config.maximum_mismatched_answer_count),
        "maximum_general_loss_regression": general_regression
        <= float(config.maximum_general_loss_regression),
        "maximum_relation_generation_regression": relation_regression
        <= float(config.maximum_relation_generation_regression),
        "unchanged_parameter_count": all(
            int(row["parameter_count"]) == int(parameter_count)
            for row in arms.values()
        ),
        "exact_optimizer_steps": all(
            int(row["training"]["optimizer_steps"]) == int(config.optimizer_steps)
            for row in arms.values()
        ),
        "exact_position_budget": all(
            int(row["training"]["processed_padded_positions"])
            == int(config.padded_position_budget_per_arm)
            for row in arms.values()
        ),
        "complete_final_gradients": all(
            bool(row["training"]["all_parameters_received_final_gradient"])
            and bool(
                row["training"]["all_parameters_received_final_nonzero_gradient"]
            )
            for row in arms.values()
        ),
        "bounded_training_time": all(
            float(row["training"]["training_seconds"])
            <= float(config.maximum_training_seconds_per_arm)
            for row in arms.values()
        ),
        "initial_short_prefix_exact": bool(parent["initial_short_prefix_exact"]),
        "parent_checkpoint_file_exact": bool(parent["checkpoint_file_exact"]),
        "parent_tokenizer_exact": bool(parent["tokenizer_exact"]),
        "all_checkpoint_fidelity": all(
            bool(row["checkpoint_fidelity"]["passed"]) for row in arms.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "native_answer_count": native_count,
            "oracle_answer_count": oracle_count,
            "native_oracle_gap": oracle_count - native_count,
            "mismatched_answer_count": mismatch_count,
            "native_source_gain": float(
                native_source["primary_gain_over_stronger_control"]
            ),
            "general_loss_regression": general_regression,
            "relation_generation_regression": relation_regression,
        },
        "thresholds": asdict(config),
    }


def _decision(gate: Mapping[str, Any]) -> str:
    if bool(gate["passed"]):
        return "advance_v57_native_context_continual_checkpoint"
    checks = dict(gate["checks"])
    if bool(checks["minimum_oracle_answer_count"]) and not bool(
        checks["minimum_native_answer_count"]
    ):
        return "pivot_v57_native_context_to_recurrent_segment_localization"
    if not bool(checks["minimum_oracle_answer_count"]):
        return "retire_v57_context_exonerated_base_or_objective_failure"
    if not bool(checks["maximum_general_loss_regression"]) or not bool(
        checks["maximum_relation_generation_regression"]
    ):
        return "redesign_v57_replay_native_capability_retention_failure"
    return "retire_v57_native_context_joint_gate_failure"


def run_native_context_falsification(
    *,
    checkpoint_path: str | Path,
    grounding_training_manifest_path: str | Path,
    grounding_validation_manifest_path: str | Path,
    relation_replay_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    output_path: str | Path,
    arm_directory: str | Path,
    candidate_checkpoint_path: str | Path,
    config: NativeContextFalsificationConfig = NativeContextFalsificationConfig(),
) -> dict[str, Any]:
    total_started = time.perf_counter()
    setup_timings: dict[str, float] = {}
    if not torch.cuda.is_available():
        raise ValueError("V57 requires CUDA")
    if len(general_train_paths) != 2 or len(general_eval_paths) != 2:
        raise ValueError("V57 requires exactly two general train/eval sources")
    device = torch.device("cuda")
    print("[native-v57] loading parent and preparing frozen data", flush=True)
    preparation_started = time.perf_counter()
    checkpoint = Path(checkpoint_path)
    checkpoint_sha_before = sha256_file(checkpoint)
    parent_model, tokenizer, parent_metadata = load_language_model_checkpoint(
        checkpoint, map_location="cpu"
    )
    tokenizer_hash_before = tokenizer.vocabulary_hash()
    parent_parameter_count = sum(p.numel() for p in parent_model.parameters())
    parent_state = {
        name: value.detach().cpu().clone()
        for name, value in parent_model.state_dict().items()
    }
    extended_config = replace(
        parent_model.config,
        transformer_context_length=int(config.context_length),
        active_language_path="marulho_transformer_v57_native_context320",
    )
    extended = MarulhoLanguageModel(extended_config)
    extended.load_state_dict(parent_state, strict=True)
    parity_ids = torch.tensor(
        [tokenizer.encode("MARULHO checks exact short prefix parity.", add_eos=False)],
        dtype=torch.long,
    )
    parent_model.eval()
    extended.eval()
    with torch.no_grad():
        parent_logits = parent_model(parity_ids, collect_telemetry=False)["logits"]
        extended_logits = extended(parity_ids, collect_telemetry=False)["logits"]
    initial_short_prefix_exact = torch.equal(parent_logits, extended_logits)
    initial_short_prefix_max_delta = float(
        (parent_logits - extended_logits).abs().max().item()
    )
    if not initial_short_prefix_exact:
        raise ValueError("V57 context extension changes short-prefix logits")
    training_manifest = load_squad_grounding_manifest(
        grounding_training_manifest_path, tokenizer
    )
    validation_manifest = load_squad_grounding_manifest(
        grounding_validation_manifest_path, tokenizer
    )
    training_ids = {str(case["case_id"]) for case in training_manifest["cases"]}
    validation_ids = {str(case["case_id"]) for case in validation_manifest["cases"]}
    if training_ids & validation_ids:
        raise ValueError("V57 train/validation IDs overlap")
    grounding_batches = {}
    grounding_reports = {}
    for arm in ARM_NAMES:
        grounding_batches[arm], grounding_reports[arm] = (
            build_native_grounding_batches(
                training_manifest,
                tokenizer,
                arm=arm,
                sequence_length=int(config.context_length),
                batch_size=int(config.batch_size),
            )
        )
    relation_batches, general_batches, general_eval_batches, replay_report = (
        _prepare_replay(
            tokenizer,
            relation_replay_path=relation_replay_path,
            general_train_paths=general_train_paths,
            general_eval_paths=general_eval_paths,
            config=config,
        )
    )
    schedule, schedule_report = build_native_context_schedule(
        grounding_batch_count=len(grounding_batches["native_full"]),
        relation_batch_count=len(relation_batches),
        general_batch_counts=[len(value) for value in general_batches],
        grounding_epochs=int(config.grounding_epochs),
        seed=int(config.data_seed),
    )
    relation_cases = _relation_cases(
        relation_cases_path, case_count=int(config.relation_case_count)
    )
    setup_timings["parent_and_data_preparation_seconds"] = (
        time.perf_counter() - preparation_started
    )
    print(
        "[native-v57] preparation complete "
        f"seconds={setup_timings['parent_and_data_preparation_seconds']:.2f}",
        flush=True,
    )
    extended = extended.to(device)
    baseline_previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    baseline_previous_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    try:
        baseline_general_started = time.perf_counter()
        baseline_general = evaluate_language_model(extended, general_eval_batches)
        setup_timings["baseline_general_seconds"] = (
            time.perf_counter() - baseline_general_started
        )
        print(
            "[native-v57] baseline general complete "
            f"seconds={setup_timings['baseline_general_seconds']:.2f} "
            f"loss={baseline_general['heldout_loss']:.4f}",
            flush=True,
        )
        baseline_relation_started = time.perf_counter()
        baseline_relation = evaluate_relation_binding_cases_batched(
            extended,
            tokenizer,
            relation_cases,
            batch_size=int(config.relation_eval_batch_size),
            max_new_tokens=int(config.relation_generation_tokens),
        )
        setup_timings["baseline_relation_seconds"] = (
            time.perf_counter() - baseline_relation_started
        )
        print(
            "[native-v57] baseline relation complete "
            f"seconds={setup_timings['baseline_relation_seconds']:.2f} "
            f"exact={baseline_relation['generation_exact_accuracy']:.4f}",
            flush=True,
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = baseline_previous_tf32
        torch.set_float32_matmul_precision(baseline_previous_precision)
    del extended
    torch.cuda.empty_cache()
    arm_root = Path(arm_directory)
    arm_root.mkdir(parents=True, exist_ok=True)
    candidate = Path(candidate_checkpoint_path)
    candidate.unlink(missing_ok=True)
    arm_rows: dict[str, dict[str, Any]] = {}
    temporary_checkpoints: dict[str, Path] = {}
    previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    try:
        for arm in ARM_NAMES:
            print(f"[native-v57] starting {arm}", flush=True)
            torch.manual_seed(int(config.model_seed))
            torch.cuda.manual_seed_all(int(config.model_seed))
            model = MarulhoLanguageModel(extended_config).to(device)
            model.load_state_dict(parent_state, strict=True)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            if parameter_count != parent_parameter_count:
                raise ValueError("V57 context extension changes parameter count")
            training = _train_arm(
                model,
                grounding_batches=grounding_batches[arm],
                relation_batches=relation_batches,
                general_batches=general_batches,
                schedule=schedule,
                schedule_report=schedule_report,
                tokenizer=tokenizer,
                config=config,
            )
            general = evaluate_language_model(model, general_eval_batches)
            relation = evaluate_relation_binding_cases_batched(
                model,
                tokenizer,
                relation_cases,
                batch_size=int(config.relation_eval_batch_size),
                max_new_tokens=int(config.relation_generation_tokens),
            )
            source = _evaluate_grounding(
                model,
                tokenizer,
                validation_manifest,
                primary=arm,
                max_new_tokens=int(config.grounding_generation_tokens),
            )
            checkpoint_output = arm_root / f"v57-{arm}.pt"
            checkpoint_output.unlink(missing_ok=True)
            model_state_hash = language_model_state_sha256(model)
            fidelity_ids = parity_ids.to(device)
            with torch.no_grad():
                fidelity_logits = model(
                    fidelity_ids, collect_telemetry=False
                )["logits"].detach().cpu()
            save_language_model_checkpoint(
                checkpoint_output,
                model,
                tokenizer,
                metadata={
                    "source_experiment": SURFACE,
                    "arm": arm,
                    "parent_checkpoint_sha256": checkpoint_sha_before,
                    "processed_padded_positions": int(
                        config.padded_position_budget_per_arm
                    ),
                    "processed_nonpad_tokens": int(training["processed_nonpad_tokens"]),
                    "cumulative_tokens": int(
                        parent_metadata.get("cumulative_tokens", 0)
                    )
                    + int(training["processed_nonpad_tokens"]),
                },
            )
            restored, restored_tokenizer, restored_metadata = (
                load_language_model_checkpoint(checkpoint_output, map_location="cpu")
            )
            restored_state_hash = language_model_state_sha256(restored)
            restored = restored.to(device).eval()
            with torch.no_grad():
                restored_logits = restored(
                    fidelity_ids, collect_telemetry=False
                )["logits"].detach().cpu()
            checkpoint_fidelity = {
                "path": str(checkpoint_output),
                "sha256": sha256_file(checkpoint_output),
                "size_bytes": checkpoint_output.stat().st_size,
                "expected_state_sha256": model_state_hash,
                "restored_state_sha256": restored_state_hash,
                "state_exact": model_state_hash == restored_state_hash,
                "logits_exact": torch.equal(fidelity_logits, restored_logits),
                "tokenizer_exact": tokenizer.vocabulary_hash()
                == restored_tokenizer.vocabulary_hash(),
                "context_exact": int(restored.context_length)
                == int(config.context_length),
                "metadata": restored_metadata,
            }
            checkpoint_fidelity["passed"] = all(
                bool(checkpoint_fidelity[key])
                for key in ("state_exact", "logits_exact", "tokenizer_exact", "context_exact")
            )
            arm_rows[arm] = {
                "parameter_count": parameter_count,
                "training": training,
                "general": general,
                "relation": relation,
                "source_grounding": source,
                "checkpoint_fidelity": checkpoint_fidelity,
            }
            temporary_checkpoints[arm] = checkpoint_output
            print(
                f"[native-v57] {arm} answers="
                f"{source['conditions']['primary']['exact_answer_count']}/256 "
                f"general={general['heldout_loss']:.4f} relation="
                f"{relation['generation_exact_accuracy']:.4f}",
                flush=True,
            )
            del model, restored
            torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.set_float32_matmul_precision(previous_precision)
    checkpoint_sha_after = sha256_file(checkpoint)
    tokenizer_hash_after = tokenizer.vocabulary_hash()
    parent = {
        "checkpoint_sha256_before": checkpoint_sha_before,
        "checkpoint_sha256_after": checkpoint_sha_after,
        "checkpoint_file_exact": checkpoint_sha_before == checkpoint_sha_after,
        "tokenizer_hash_before": tokenizer_hash_before,
        "tokenizer_hash_after": tokenizer_hash_after,
        "tokenizer_exact": tokenizer_hash_before == tokenizer_hash_after,
        "initial_short_prefix_exact": initial_short_prefix_exact,
        "initial_short_prefix_max_absolute_delta": initial_short_prefix_max_delta,
        "parameter_count": parent_parameter_count,
    }
    gate = native_context_gate(
        arm_rows,
        baseline={"general": baseline_general, "relation": baseline_relation},
        parent=parent,
        config=config,
        parameter_count=parent_parameter_count,
    )
    decision = _decision(gate)
    if bool(gate["passed"]):
        candidate.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_checkpoints["native_full"], candidate)
        arm_rows["native_full"]["checkpoint_fidelity"][
            "candidate_checkpoint_path"
        ] = str(candidate)
        arm_rows["native_full"]["checkpoint_fidelity"][
            "candidate_checkpoint_sha256"
        ] = sha256_file(candidate)
    for arm, path in temporary_checkpoints.items():
        if path.exists():
            path.unlink()
    try:
        arm_root.rmdir()
    except OSError:
        pass
    frozen_contract = {
        "surface": SURFACE,
        "config": asdict(config),
        "parent_checkpoint_sha256": checkpoint_sha_before,
        "training_manifest_contract": training_manifest["contract_sha256"],
        "validation_manifest_contract": validation_manifest["contract_sha256"],
        "grounding_batches": grounding_reports,
        "schedule": schedule_report,
        "replay": replay_report,
    }
    report = {
        "surface": SURFACE,
        "decision": decision,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "external_text_data": True,
        "configuration": asdict(config),
        "experiment_contract_sha256": _canonical_sha256(frozen_contract),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha_before,
            "metadata": parent_metadata,
        },
        "data": {
            "training_manifest_path": str(grounding_training_manifest_path),
            "training_manifest_sha256": sha256_file(
                grounding_training_manifest_path
            ),
            "training_manifest_contract_sha256": training_manifest[
                "contract_sha256"
            ],
            "validation_manifest_path": str(grounding_validation_manifest_path),
            "validation_manifest_sha256": sha256_file(
                grounding_validation_manifest_path
            ),
            "validation_manifest_contract_sha256": validation_manifest[
                "contract_sha256"
            ],
            "training_validation_case_overlap": len(training_ids & validation_ids),
            "grounding_batches": grounding_reports,
            "replay": replay_report,
            "schedule": schedule_report,
        },
        "baseline": {
            "general": baseline_general,
            "relation": baseline_relation,
            "grounding": {
                "performed": False,
                "reason": (
                    "No V57 gate or branch decision consumes parent grounding; "
                    "the disjoint trained-arm source controls own capability truth."
                ),
            },
        },
        "setup_timings": setup_timings,
        "arms": arm_rows,
        "parent": parent,
        "gate": gate,
        "total_wall_seconds": time.perf_counter() - total_started,
        "boundary": (
            "V57 tests whether V39 can learn source-visible long-context QA when "
            "evidence participates in every causal layer. It does not prove open-"
            "domain retrieval, arbitrary-length memory, or a post-Transformer "
            "architecture."
        ),
    }
    write_json_report_with_readme(
        output_path, report, title="MARULHO V57 Native Long Context"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--grounding-training-manifest", type=Path, required=True)
    parser.add_argument(
        "--grounding-validation-manifest", type=Path, required=True
    )
    parser.add_argument("--relation-replay", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--general-train", type=Path, action="append", required=True)
    parser.add_argument("--general-eval", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm-directory", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    report = run_native_context_falsification(
        checkpoint_path=args.checkpoint,
        grounding_training_manifest_path=args.grounding_training_manifest,
        grounding_validation_manifest_path=args.grounding_validation_manifest,
        relation_replay_path=args.relation_replay,
        relation_cases_path=args.relation_cases,
        general_train_paths=tuple(args.general_train),
        general_eval_paths=tuple(args.general_eval),
        output_path=args.output,
        arm_directory=args.arm_directory,
        candidate_checkpoint_path=args.candidate_checkpoint,
    )
    print(report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
