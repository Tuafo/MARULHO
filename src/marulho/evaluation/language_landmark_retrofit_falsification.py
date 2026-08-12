"""Run the preregistered V56 landmark-evidence retrofit falsification."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from marulho.evaluation.language_matched_support import (
    MatchedLanguageDataConfig,
    prepare_matched_language_data,
    sha256_file,
)
from marulho.evaluation.language_relation_binding_experiment import (
    evaluate_relation_binding_cases_batched,
)
from marulho.evaluation.language_source_grounding import (
    _contains_answer,
    load_squad_grounding_manifest,
)
from marulho.evaluation.language_source_grounding_continual import (
    _stratified_relation_cases,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_landmark_retrofit import (
    FrozenBaseLandmarkRetrofit,
    build_landmark_retrofit_batches,
    cache_landmark_retrofit_hidden,
    load_landmark_retrofit_checkpoint,
    save_landmark_retrofit_checkpoint,
)
from marulho.training.language_model import (
    evaluate_language_model,
    language_model_state_sha256,
    load_language_model_checkpoint,
)


SURFACE = "marulho_landmark_retrofit_falsification.v1"
ARCHITECTURE = "frozen_v39_landmark_retrieval_causal_cross_attention"


@dataclass(frozen=True)
class LandmarkRetrofitFalsificationConfig:
    epoch_count: int = 15
    adapter_position_budget: int = 20_643_840
    query_length: int = 72
    batch_size: int = 32
    block_tokens: int = 48
    maximum_blocks: int = 5
    selected_blocks: int = 2
    retrieval_width: int = 128
    adapter_width: int = 256
    adapter_layers: int = 2
    adapter_heads: int = 8
    maximum_answer_tokens: int = 12
    learning_rate: float = 3.0e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    data_seed: int = 56121
    model_seed: int = 56131
    relation_case_count: int = 64
    relation_eval_batch_size: int = 8
    relation_generation_tokens: int = 16
    general_eval_batches: int = 16
    maximum_cache_plus_training_seconds: float = 1_200.0
    maximum_parameter_fraction: float = 0.03
    minimum_predicted_coverage: float = 0.80
    minimum_predicted_answer_count: int = 64
    minimum_source_gain: float = 0.45
    minimum_oracle_answer_count: int = 72
    maximum_predicted_oracle_gap: int = 10
    maximum_shuffled_answer_count: int = 8


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _learning_rate_factor(
    step: int,
    *,
    step_count: int,
    warmup_steps: int,
    minimum_fraction: float,
) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, step_count - warmup_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return float(minimum_fraction) + (1.0 - float(minimum_fraction)) * cosine


def _training_schedule(
    *, batch_count: int, epoch_count: int, seed: int
) -> tuple[list[int], str]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    schedule = []
    for _epoch in range(int(epoch_count)):
        schedule.extend(torch.randperm(int(batch_count), generator=generator).tolist())
    encoded = torch.tensor(schedule, dtype=torch.int64)
    return schedule, hashlib.sha256(encoded.numpy().tobytes()).hexdigest()


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        int(value.numel() * value.element_size())
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def _train_retrofit(
    model: FrozenBaseLandmarkRetrofit,
    batches,
    *,
    cache_elapsed_seconds: float,
    config: LandmarkRetrofitFalsificationConfig,
) -> dict[str, Any]:
    schedule, schedule_sha256 = _training_schedule(
        batch_count=len(batches),
        epoch_count=int(config.epoch_count),
        seed=int(config.data_seed),
    )
    step_count = len(schedule)
    positions_per_case = int(config.query_length) + (
        int(config.selected_blocks) * int(config.block_tokens)
    )
    positions_per_step = int(config.batch_size) * positions_per_case
    processed_positions = step_count * positions_per_step
    if processed_positions != int(config.adapter_position_budget):
        raise ValueError("V56 schedule differs from the frozen position budget")
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [parameter for _name, parameter in trainable],
        lr=float(config.learning_rate),
        betas=(0.9, 0.95),
        weight_decay=float(config.weight_decay),
        fused=model.device.type == "cuda",
    )
    warmup_steps = max(1, math.ceil(step_count * float(config.warmup_fraction)))
    received_gradient = {name: False for name, _parameter in trainable}
    received_nonzero_gradient = {name: False for name, _parameter in trainable}
    trace = []
    initial_loss = None
    final_loss = None
    final_generator_loss = None
    final_retrieval_loss = None
    model.train()
    torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    for step, batch_index in enumerate(schedule):
        batch = batches[batch_index].to(model.device)
        optimizer.zero_grad(set_to_none=True)
        factor = _learning_rate_factor(
            step,
            step_count=step_count,
            warmup_steps=warmup_steps,
            minimum_fraction=float(config.minimum_learning_rate_fraction),
        )
        learning_rate = float(config.learning_rate) * factor
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=str(config.precision) == "bfloat16",
        ):
            loss, components = model.loss(batch)
        if not bool(torch.isfinite(loss).item()):
            raise ValueError(f"V56 loss became non-finite at step {step}")
        loss.backward()
        for name, parameter in trainable:
            if parameter.grad is not None:
                received_gradient[name] = True
                if bool(parameter.grad.detach().ne(0).any().item()):
                    received_nonzero_gradient[name] = True
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _name, parameter in trainable],
            max_norm=float(config.gradient_clip),
        )
        optimizer.step()
        value = float(loss.detach().item())
        generator_value = float(components["generator_loss"].item())
        retrieval_value = float(components["retrieval_loss"].item())
        initial_loss = value if initial_loss is None else initial_loss
        final_loss = value
        final_generator_loss = generator_value
        final_retrieval_loss = retrieval_value
        if step in {
            0,
            warmup_steps - 1,
            step_count // 4,
            step_count // 2,
            3 * step_count // 4,
            step_count - 1,
        }:
            trace.append(
                {
                    "step": step + 1,
                    "epoch": step // len(batches) + 1,
                    "loss": value,
                    "generator_loss": generator_value,
                    "retrieval_loss": retrieval_value,
                    "learning_rate": learning_rate,
                    "gradient_norm": float(gradient_norm.detach().item()),
                }
            )
        if (step + 1) % len(batches) == 0 or step + 1 == step_count:
            print(
                f"[landmark-v56] epoch {(step + 1) // len(batches)}/"
                f"{config.epoch_count} step {step + 1}/{step_count} "
                f"loss={value:.4f} generator={generator_value:.4f} "
                f"retrieval={retrieval_value:.4f}",
                flush=True,
            )
    torch.cuda.synchronize(model.device)
    training_seconds = time.perf_counter() - started
    final_gradient = {name: parameter.grad is not None for name, parameter in trainable}
    final_nonzero_gradient = {
        name: parameter.grad is not None
        and bool(parameter.grad.detach().ne(0).any().item())
        for name, parameter in trainable
    }
    cache_plus_training = float(cache_elapsed_seconds) + training_seconds
    return {
        "architecture": ARCHITECTURE,
        "processed_adapter_positions": processed_positions,
        "epoch_count": int(config.epoch_count),
        "optimizer_steps": step_count,
        "positions_per_step": positions_per_step,
        "training_seconds": training_seconds,
        "cache_plus_training_seconds": cache_plus_training,
        "training_positions_per_second": processed_positions / training_seconds,
        "cache_amortized_positions_per_second": processed_positions
        / cache_plus_training,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_generator_loss": final_generator_loss,
        "final_retrieval_loss": final_retrieval_loss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(model.device),
        "optimizer_state_bytes": _optimizer_state_bytes(optimizer),
        "schedule_sha256": schedule_sha256,
        "warmup_steps": warmup_steps,
        "trace": trace,
        "all_parameters_received_gradient": all(received_gradient.values()),
        "all_parameters_received_nonzero_gradient": all(
            received_nonzero_gradient.values()
        ),
        "all_parameters_received_final_gradient": all(final_gradient.values()),
        "all_parameters_received_final_nonzero_gradient": all(
            final_nonzero_gradient.values()
        ),
        "missing_gradient_parameters": [
            name for name, seen in received_gradient.items() if not seen
        ],
        "zero_gradient_parameters": [
            name for name, seen in received_nonzero_gradient.items() if not seen
        ],
        "final_missing_gradient_parameters": [
            name for name, seen in final_gradient.items() if not seen
        ],
        "final_zero_gradient_parameters": [
            name for name, seen in final_nonzero_gradient.items() if not seen
        ],
    }


def _continuation(
    model: FrozenBaseLandmarkRetrofit,
    tokenizer,
    prompt: str,
    *,
    maximum_answer_tokens: int,
    evidence_indices: torch.Tensor | None = None,
) -> str:
    prompt_ids = tokenizer.encode(prompt, add_eos=False)
    generated = model.generate(
        torch.tensor(prompt_ids, dtype=torch.long, device=model.device),
        max_new_tokens=int(maximum_answer_tokens),
        eos_id=tokenizer.eos_id,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        evidence_indices=evidence_indices,
    )["generated_ids"][0, len(prompt_ids) :]
    return tokenizer.decode([int(value) for value in generated.detach().cpu().tolist()])


def _condition_report(
    cases: Sequence[Mapping[str, Any]], rows: Sequence[str]
) -> dict[str, Any]:
    values = []
    for case, continuation in zip(cases, rows, strict=True):
        answers = tuple(str(value) for value in case["answers"])
        values.append(
            {
                "case_id": str(case["case_id"]),
                "answers": list(answers),
                "continuation": continuation,
                "exact_answer_match": _contains_answer(continuation, answers),
            }
        )
    count = sum(bool(value["exact_answer_match"]) for value in values)
    return {
        "case_count": len(values),
        "exact_answer_count": count,
        "exact_answer_accuracy": count / max(1, len(values)),
        "rows": values,
    }


@torch.no_grad()
def _evaluate_landmark_grounding(
    model: FrozenBaseLandmarkRetrofit,
    tokenizer,
    manifest: Mapping[str, Any],
    cached_batches,
    *,
    maximum_answer_tokens: int,
) -> dict[str, Any]:
    cases = tuple(dict(value) for value in manifest["cases"])
    predicted_indices = []
    top1_indices = []
    oracle_indices = []
    shuffled_indices = []
    coverage = []
    top1_coverage = []
    for batch in cached_batches:
        scores = model.retrieval_scores(
            batch.source_hidden,
            batch.source_attention_mask,
            batch.block_valid_mask,
            batch.retrieval_query_hidden,
            batch.retrieval_query_attention_mask,
        )
        predicted = scores.topk(k=2, dim=1).indices.sort(dim=1).values.cpu()
        top1 = scores.topk(k=1, dim=1).indices.cpu()
        gold_mask = batch.gold_block_mask
        valid_mask = batch.block_valid_mask
        for row in range(int(scores.shape[0])):
            predicted_row = [int(value) for value in predicted[row].tolist()]
            top1_row = [int(value) for value in top1[row].tolist()]
            gold_row = [
                int(value) for value in batch.gold_evidence_indices[row].tolist()
            ]
            positive = {
                index
                for index, value in enumerate(gold_mask[row].tolist())
                if bool(value)
            }
            nongold = [
                index
                for index, value in enumerate(valid_mask[row].tolist())
                if bool(value) and index not in positive
            ]
            if not nongold:
                raise ValueError("V56 shuffled evidence has no non-answer block")
            shuffled = nongold[:2]
            if len(shuffled) == 1:
                shuffled = shuffled * 2
            predicted_indices.append(predicted_row)
            top1_indices.append(top1_row)
            oracle_indices.append(gold_row)
            shuffled_indices.append(shuffled)
            coverage.append(positive.issubset(predicted_row))
            top1_coverage.append(positive.issubset(top1_row))
    if len(predicted_indices) != len(cases):
        raise ValueError("V56 validation cache differs from manifest")
    model_state_before = language_model_state_sha256(model)
    outputs = {
        "predicted_top2": [],
        "predicted_top1": [],
        "oracle": [],
        "shuffled": [],
        "question_only": [],
        "mismatched_source": [],
    }
    model.eval()
    for index, case in enumerate(cases):
        outputs["predicted_top2"].append(
            _continuation(
                model,
                tokenizer,
                str(case["prompt"]),
                maximum_answer_tokens=maximum_answer_tokens,
                evidence_indices=torch.tensor(predicted_indices[index]),
            )
        )
        outputs["predicted_top1"].append(
            _continuation(
                model,
                tokenizer,
                str(case["prompt"]),
                maximum_answer_tokens=maximum_answer_tokens,
                evidence_indices=torch.tensor(top1_indices[index]),
            )
        )
        outputs["oracle"].append(
            _continuation(
                model,
                tokenizer,
                str(case["prompt"]),
                maximum_answer_tokens=maximum_answer_tokens,
                evidence_indices=torch.tensor(oracle_indices[index]),
            )
        )
        outputs["shuffled"].append(
            _continuation(
                model,
                tokenizer,
                str(case["prompt"]),
                maximum_answer_tokens=maximum_answer_tokens,
                evidence_indices=torch.tensor(shuffled_indices[index]),
            )
        )
        outputs["question_only"].append(
            _continuation(
                model,
                tokenizer,
                str(case["question_only_prompt"]),
                maximum_answer_tokens=maximum_answer_tokens,
            )
        )
        outputs["mismatched_source"].append(
            _continuation(
                model,
                tokenizer,
                str(case["mismatched_prompt"]),
                maximum_answer_tokens=maximum_answer_tokens,
            )
        )
        if (index + 1) % 16 == 0 or index + 1 == len(cases):
            print(
                f"[landmark-v56] evaluated {index + 1}/{len(cases)} cases",
                flush=True,
            )
    reports = {
        name: _condition_report(cases, values) for name, values in outputs.items()
    }
    predicted = reports["predicted_top2"]
    stronger_control = max(
        float(reports["question_only"]["exact_answer_accuracy"]),
        float(reports["mismatched_source"]["exact_answer_accuracy"]),
    )
    validity = {
        "all_answers_visible_in_intact_source": all(
            _contains_answer(str(case["source_text"]), tuple(case["answers"]))
            for case in cases
        ),
        "all_answers_absent_from_question": all(
            not _contains_answer(str(case["question"]), tuple(case["answers"]))
            for case in cases
        ),
        "all_answers_absent_from_mismatched_source": all(
            not _contains_answer(
                str(case["mismatched_source_text"]), tuple(case["answers"])
            )
            for case in cases
        ),
        "model_state_immutable": language_model_state_sha256(model)
        == model_state_before,
    }
    return {
        "valid": all(validity.values()),
        "validity": validity,
        "predicted_top2_block_union_answer_count": sum(coverage),
        "predicted_top2_block_union_answer_coverage": sum(coverage)
        / max(1, len(coverage)),
        "predicted_top1_block_answer_count": sum(top1_coverage),
        "predicted_top1_block_answer_coverage": sum(top1_coverage)
        / max(1, len(top1_coverage)),
        "intact_gain_over_stronger_control": float(predicted["exact_answer_accuracy"])
        - stronger_control,
        "conditions": reports,
        "evidence_indices": {
            "predicted_top2": predicted_indices,
            "predicted_top1": top1_indices,
            "oracle": oracle_indices,
            "shuffled": shuffled_indices,
        },
    }


def landmark_retrofit_gate(
    row: Mapping[str, Any],
    *,
    parent: Mapping[str, Any],
    checkpoint_fidelity: Mapping[str, Any],
    retrofit_parameters: int,
    parent_parameters: int,
    config: LandmarkRetrofitFalsificationConfig,
) -> dict[str, Any]:
    source = dict(row["source_grounding"])
    conditions = dict(source["conditions"])
    predicted_count = int(conditions["predicted_top2"]["exact_answer_count"])
    oracle_count = int(conditions["oracle"]["exact_answer_count"])
    shuffled_count = int(conditions["shuffled"]["exact_answer_count"])
    predicted_oracle_gap = oracle_count - predicted_count
    parameter_fraction = int(retrofit_parameters) / int(parent_parameters)
    checks = {
        "source_valid": bool(source["valid"]),
        "minimum_predicted_coverage": float(
            source["predicted_top2_block_union_answer_coverage"]
        )
        >= float(config.minimum_predicted_coverage),
        "minimum_predicted_answer_count": predicted_count
        >= int(config.minimum_predicted_answer_count),
        "minimum_source_gain": float(source["intact_gain_over_stronger_control"])
        >= float(config.minimum_source_gain),
        "minimum_oracle_answer_count": oracle_count
        >= int(config.minimum_oracle_answer_count),
        "maximum_predicted_oracle_gap": predicted_oracle_gap
        <= int(config.maximum_predicted_oracle_gap),
        "maximum_shuffled_answer_count": shuffled_count
        <= int(config.maximum_shuffled_answer_count),
        "exact_adapter_position_budget": int(row["processed_adapter_positions"])
        == int(config.adapter_position_budget),
        "exact_epoch_count": int(row["epoch_count"]) == int(config.epoch_count),
        "all_retrofit_parameters_received_gradient": bool(
            row["all_parameters_received_final_gradient"]
        ),
        "all_retrofit_parameters_received_nonzero_gradient": bool(
            row["all_parameters_received_final_nonzero_gradient"]
        ),
        "maximum_parameter_fraction": parameter_fraction
        <= float(config.maximum_parameter_fraction),
        "bounded_cache_plus_training_time": float(row["cache_plus_training_seconds"])
        <= float(config.maximum_cache_plus_training_seconds),
        "parent_checkpoint_file_exact": bool(parent["checkpoint_file_exact"]),
        "parent_state_exact": bool(parent["state_exact"]),
        "parent_tokenizer_exact": bool(parent["tokenizer_exact"]),
        "parent_logits_exact": bool(parent["logits_exact"]),
        "parent_general_loss_exact": bool(parent["general_loss_exact"]),
        "parent_relation_exact": bool(parent["relation_exact"]),
        "checkpoint_fidelity": bool(checkpoint_fidelity["passed"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "predicted_coverage": float(
                source["predicted_top2_block_union_answer_coverage"]
            ),
            "predicted_answer_count": predicted_count,
            "oracle_answer_count": oracle_count,
            "predicted_oracle_gap": predicted_oracle_gap,
            "shuffled_answer_count": shuffled_count,
            "source_gain": float(source["intact_gain_over_stronger_control"]),
            "retrofit_parameters": int(retrofit_parameters),
            "parent_parameters": int(parent_parameters),
            "parameter_fraction": parameter_fraction,
            "cache_plus_training_seconds": float(row["cache_plus_training_seconds"]),
        },
        "thresholds": asdict(config),
    }


def run_landmark_retrofit_falsification(
    *,
    checkpoint_path: str | Path,
    grounding_training_manifest_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    grounding_validation_manifest_path: str | Path,
    output_path: str | Path,
    arm_artifact_path: str | Path,
    candidate_checkpoint_path: str | Path,
    config: LandmarkRetrofitFalsificationConfig = (
        LandmarkRetrofitFalsificationConfig()
    ),
) -> dict[str, Any]:
    if len(general_train_paths) < 2 or len(general_eval_paths) != 2:
        raise ValueError("V56 requires at least two train and exactly two eval sources")
    if not torch.cuda.is_available():
        raise ValueError("V56 is a CUDA-only evidence run")
    device = torch.device("cuda")
    checkpoint = Path(checkpoint_path)
    checkpoint_sha_before = sha256_file(checkpoint)
    base, tokenizer, checkpoint_metadata = load_language_model_checkpoint(
        checkpoint, map_location="cpu"
    )
    tokenizer_hash_before = tokenizer.vocabulary_hash()
    if int(base.context_length) != int(config.query_length):
        raise ValueError("V56 must preserve the V39 context length")
    parent_parameters = sum(parameter.numel() for parameter in base.parameters())
    parent_state_before = language_model_state_sha256(base)
    training_manifest = load_squad_grounding_manifest(
        grounding_training_manifest_path, tokenizer
    )
    validation = load_squad_grounding_manifest(
        grounding_validation_manifest_path, tokenizer
    )
    training_ids = {str(case["case_id"]) for case in training_manifest["cases"]}
    validation_ids = {str(case["case_id"]) for case in validation["cases"]}
    if training_ids & validation_ids:
        raise ValueError("V56 training and validation case IDs overlap")
    training_batches, training_supervision = build_landmark_retrofit_batches(
        training_manifest,
        tokenizer,
        batch_size=int(config.batch_size),
        block_tokens=int(config.block_tokens),
        maximum_blocks=int(config.maximum_blocks),
        query_length=int(config.query_length),
    )
    validation_batches, validation_supervision = build_landmark_retrofit_batches(
        validation,
        tokenizer,
        batch_size=int(config.batch_size),
        block_tokens=int(config.block_tokens),
        maximum_blocks=int(config.maximum_blocks),
        query_length=int(config.query_length),
    )
    if len(training_batches) * int(config.batch_size) != 8_192:
        raise ValueError("V56 requires the frozen 8,192-case training manifest")
    if len(validation_batches) * int(config.batch_size) != 128:
        raise ValueError("V56 requires the frozen 128-case validation manifest")
    print("[landmark-v56] preparing immutable parent audits", flush=True)
    prepared = prepare_matched_language_data(
        tokenizer_checkpoint_path=checkpoint,
        relation_corpus_path=grounding_training_manifest_path,
        relation_cases_path=relation_cases_path,
        general_train_paths=general_train_paths,
        general_eval_paths=general_eval_paths,
        config=MatchedLanguageDataConfig(
            token_budget=int(config.batch_size) * int(config.query_length),
            sequence_length=int(config.query_length),
            batch_size=8,
            eval_batches=int(config.general_eval_batches),
            relation_fraction=0.0,
            seed=int(config.data_seed),
            sample_bytes_per_train_source=1 * 1024 * 1024,
            sample_bytes_per_eval_source=1 * 1024 * 1024,
            sample_range_count=8,
            schedule_mode="indexed_host",
        ),
        device=device,
    )
    prepared = replace(
        prepared,
        cases=_stratified_relation_cases(
            prepared.cases, case_count=int(config.relation_case_count)
        ),
    )
    torch.manual_seed(int(config.model_seed))
    torch.cuda.manual_seed_all(int(config.model_seed))
    base = base.to(device)
    model = FrozenBaseLandmarkRetrofit(
        base,
        tokenizer=tokenizer,
        pad_id=int(tokenizer.pad_id),
        eos_id=int(tokenizer.eos_id),
        block_tokens=int(config.block_tokens),
        maximum_blocks=int(config.maximum_blocks),
        retrieval_width=int(config.retrieval_width),
        adapter_width=int(config.adapter_width),
        adapter_layers=int(config.adapter_layers),
        adapter_heads=int(config.adapter_heads),
    ).to(device)
    retrofit_parameters = model.retrofit_parameter_count()
    if retrofit_parameters / parent_parameters > float(
        config.maximum_parameter_fraction
    ):
        raise ValueError("V56 retrofit exceeds the parameter-fraction gate")
    previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    arm = Path(arm_artifact_path)
    candidate = Path(candidate_checkpoint_path)
    arm.unlink(missing_ok=True)
    candidate.unlink(missing_ok=True)
    try:
        baseline_general = evaluate_language_model(base, prepared.eval_batches)
        baseline_relation = evaluate_relation_binding_cases_batched(
            base,
            tokenizer,
            prepared.cases,
            batch_size=int(config.relation_eval_batch_size),
            max_new_tokens=int(config.relation_generation_tokens),
        )
        sample = prepared.eval_batches[0].to(device)
        with torch.no_grad():
            baseline_logits = (
                base(sample.input_ids, collect_telemetry=False)["logits"].detach().cpu()
            )
        torch.cuda.reset_peak_memory_stats(device)
        print("[landmark-v56] caching frozen V39 train states", flush=True)
        cached_training, training_cache = cache_landmark_retrofit_hidden(
            base, training_batches, device=device
        )
        print("[landmark-v56] caching frozen V39 validation states", flush=True)
        cached_validation, validation_cache = cache_landmark_retrofit_hidden(
            base, validation_batches, device=device
        )
        cache_elapsed = float(training_cache["elapsed_seconds"]) + float(
            validation_cache["elapsed_seconds"]
        )
        print("[landmark-v56] training landmark retrofit", flush=True)
        row = _train_retrofit(
            model,
            cached_training,
            cache_elapsed_seconds=cache_elapsed,
            config=config,
        )
        row.update(
            {
                "parent_frozen": True,
                "replay_used": False,
                "retrofit_parameters": retrofit_parameters,
                "parent_parameters": parent_parameters,
            }
        )
        frozen_contract = {
            "surface": SURFACE,
            "configuration": asdict(config),
            "checkpoint_sha256": checkpoint_sha_before,
            "parent_state_sha256": parent_state_before,
            "training_manifest_contract": training_manifest["contract_sha256"],
            "validation_manifest_contract": validation["contract_sha256"],
            "training_supervision": training_supervision,
            "validation_supervision": validation_supervision,
            "training_cache_sha256": training_cache["content_sha256"],
            "validation_cache_sha256": validation_cache["content_sha256"],
            "schedule_sha256": row["schedule_sha256"],
        }
        experiment_contract_sha256 = _canonical_sha256(frozen_contract)
        row["experiment_contract_sha256"] = experiment_contract_sha256
        save_landmark_retrofit_checkpoint(
            arm,
            model,
            parent_checkpoint_sha256=checkpoint_sha_before,
            metadata={
                "source_experiment": SURFACE,
                "processed_adapter_positions": int(config.adapter_position_budget),
                "experiment_contract_sha256": experiment_contract_sha256,
            },
        )
        print("[landmark-v56] evaluating frozen interventions", flush=True)
        source_grounding = _evaluate_landmark_grounding(
            model,
            tokenizer,
            validation,
            cached_validation,
            maximum_answer_tokens=int(config.maximum_answer_tokens),
        )
        row["source_grounding"] = source_grounding
        checkpoint_sha_after = sha256_file(checkpoint)
        parent_state_after = language_model_state_sha256(model.base)
        tokenizer_hash_after = tokenizer.vocabulary_hash()
        final_general = evaluate_language_model(base, prepared.eval_batches)
        final_relation = evaluate_relation_binding_cases_batched(
            base,
            tokenizer,
            prepared.cases,
            batch_size=int(config.relation_eval_batch_size),
            max_new_tokens=int(config.relation_generation_tokens),
        )
        with torch.no_grad():
            final_logits = (
                base(sample.input_ids, collect_telemetry=False)["logits"].detach().cpu()
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.set_float32_matmul_precision(previous_precision)
    parent = {
        "checkpoint_sha256_before": checkpoint_sha_before,
        "checkpoint_sha256_after": checkpoint_sha_after,
        "checkpoint_file_exact": checkpoint_sha_before == checkpoint_sha_after,
        "state_sha256_before": parent_state_before,
        "state_sha256_after": parent_state_after,
        "state_exact": parent_state_before == parent_state_after,
        "tokenizer_hash_before": tokenizer_hash_before,
        "tokenizer_hash_after": tokenizer_hash_after,
        "tokenizer_exact": tokenizer_hash_before == tokenizer_hash_after,
        "logits_exact": torch.equal(baseline_logits, final_logits),
        "general_loss_before": float(baseline_general["heldout_loss"]),
        "general_loss_after": float(final_general["heldout_loss"]),
        "general_loss_exact": float(baseline_general["heldout_loss"])
        == float(final_general["heldout_loss"]),
        "relation_before": baseline_relation,
        "relation_after": final_relation,
        "relation_exact": baseline_relation == final_relation,
    }
    expected_state = language_model_state_sha256(model)
    restored_base, restored_tokenizer, _metadata = load_language_model_checkpoint(
        checkpoint, map_location="cpu"
    )
    restored, restored_metadata = load_landmark_retrofit_checkpoint(
        arm,
        restored_base,
        restored_tokenizer,
        expected_parent_checkpoint_sha256=checkpoint_sha_before,
    )
    checkpoint_fidelity = {
        "performed": True,
        "arm_checkpoint_path": str(arm),
        "arm_checkpoint_sha256": sha256_file(arm),
        "arm_checkpoint_size_bytes": arm.stat().st_size,
        "expected_state_sha256": expected_state,
        "restored_state_sha256": language_model_state_sha256(restored),
        "tokenizer_hash_before": tokenizer_hash_before,
        "tokenizer_hash_after": restored_tokenizer.vocabulary_hash(),
        "metadata": restored_metadata,
    }
    checkpoint_fidelity["passed"] = bool(
        checkpoint_fidelity["expected_state_sha256"]
        == checkpoint_fidelity["restored_state_sha256"]
        and checkpoint_fidelity["tokenizer_hash_before"]
        == checkpoint_fidelity["tokenizer_hash_after"]
    )
    gate = landmark_retrofit_gate(
        row,
        parent=parent,
        checkpoint_fidelity=checkpoint_fidelity,
        retrofit_parameters=retrofit_parameters,
        parent_parameters=parent_parameters,
        config=config,
    )
    decision = (
        "advance_v56_landmark_retrofit_to_durable_evidence_memory"
        if bool(gate["passed"])
        else "retire_v56_landmark_retrofit_capability_or_retrieval_failure"
    )
    if bool(gate["passed"]):
        candidate.parent.mkdir(parents=True, exist_ok=True)
        os.replace(arm, candidate)
        checkpoint_fidelity["candidate_checkpoint_path"] = str(candidate)
        checkpoint_fidelity["candidate_checkpoint_sha256"] = sha256_file(candidate)
    else:
        arm.unlink(missing_ok=True)
        candidate.unlink(missing_ok=True)
    report = {
        "surface": SURFACE,
        "decision": decision,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "external_text_data": True,
        "configuration": asdict(config),
        "experiment_contract_sha256": experiment_contract_sha256,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha_before,
            "metadata": checkpoint_metadata,
        },
        "data": {
            "training_manifest_path": str(grounding_training_manifest_path),
            "training_manifest_sha256": sha256_file(grounding_training_manifest_path),
            "training_manifest_contract_sha256": training_manifest["contract_sha256"],
            "validation_manifest_path": str(grounding_validation_manifest_path),
            "validation_manifest_sha256": sha256_file(
                grounding_validation_manifest_path
            ),
            "validation_manifest_contract_sha256": validation["contract_sha256"],
            "training_validation_case_overlap": len(training_ids & validation_ids),
            "training_supervision": training_supervision,
            "validation_supervision": validation_supervision,
            "training_hidden_cache": training_cache,
            "validation_hidden_cache": validation_cache,
            "combined_cache_elapsed_seconds": cache_elapsed,
            "combined_cache_host_storage_bytes": int(
                training_cache["host_storage_bytes"]
            )
            + int(validation_cache["host_storage_bytes"]),
            "general_source_selections": prepared.source_selections,
        },
        "baseline": {"general": baseline_general, "relation": baseline_relation},
        "arm": row,
        "parent": parent,
        "checkpoint_fidelity": checkpoint_fidelity,
        "gate": gate,
        "boundary": (
            "V56 tests frozen-parent block retrieval and causal evidence injection "
            "on heldout extractive long-context questions. It does not prove open-"
            "domain retrieval, abstractive synthesis, continual writes, or a "
            "replacement base architecture."
        ),
    }
    write_json_report_with_readme(
        output_path, report, title="MARULHO V56 Landmark Evidence Retrofit"
    )
    del cached_training, cached_validation
    torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--grounding-training-manifest", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--general-train", type=Path, action="append", required=True)
    parser.add_argument("--general-eval", type=Path, action="append", required=True)
    parser.add_argument("--grounding-validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm-artifact", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    report = run_landmark_retrofit_falsification(
        checkpoint_path=args.checkpoint,
        grounding_training_manifest_path=args.grounding_training_manifest,
        relation_cases_path=args.relation_cases,
        general_train_paths=tuple(args.general_train),
        general_eval_paths=tuple(args.general_eval),
        grounding_validation_manifest_path=args.grounding_validation_manifest,
        output_path=args.output,
        arm_artifact_path=args.arm_artifact,
        candidate_checkpoint_path=args.candidate_checkpoint,
    )
    print(report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
