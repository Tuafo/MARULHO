"""Jointly train the V11 cortex to use selected causal document episodes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from marulho.evaluation.language_causal_document_retrieval_audit import (
    CausalDocumentRetrievalConfig,
    DocumentContinuationCase,
    EncodedDocumentContinuations,
    build_archive_groups,
    build_document_cases,
    encode_document_cases,
    evaluate_document_arm,
    gather_retrieved_episodes,
    paired_bootstrap_gain,
)
from marulho.evaluation.language_exact_episodic_retrieval_audit import (
    EncodedTextBank,
    lexical_tfidf_scores,
    rankings_from_scores,
)
from marulho.evaluation.language_exact_episodic_retrieval_screen import (
    PreparedGeneralLanguage,
    RetrievalScreenConfig,
    _precision_context,
    _scheduled_relation_steps,
    evaluate_general_language,
    prepare_general_language,
)
from marulho.evaluation.language_hashed_micro_expert_continuation import (
    _validate_parent,
)
from marulho.evaluation.language_matched_support import (
    build_matched_schedule,
    sample_corpus_ranges,
    schedule_sha256,
    sha256_file,
)
from marulho.evaluation.language_training_experiment import _learning_rate
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_hashed_micro_experts import (
    MarulhoHashedMicroExpertLanguageModel,
    load_hashed_micro_expert_checkpoint,
)


SURFACE = "marulho_balanced_joint_document_retrieval_screen.v2"
ARTIFACT_KIND = "marulho_balanced_joint_document_retrieval_screen"
ARM_NAMES = ("off", "random2", "lexical1", "lexical2", "oracle2")
ADVANCE_DECISION = "advance_v24_balanced_top_two_to_anchored_review"
MINIMUM_DECISION_STEPS = 512


@dataclass(frozen=True)
class JointDocumentRetrievalConfig:
    train_cases_per_source: int = 2048
    eval_cases_per_source: int = 128
    facts_per_query: int = 4
    source_length: int = 48
    prefix_length: int = 48
    target_length: int = 16
    minimum_gap_tokens: int = 48
    maximum_gap_tokens: int = 192
    train_steps: int = 800
    batch_size: int = 16
    eval_batch_size: int = 16
    document_fraction: float = 0.50
    general_sequence_length: int = 128
    general_eval_batches: int = 8
    document_train_sample_bytes: int = 16 * 1024 * 1024
    document_eval_sample_bytes: int = 8 * 1024 * 1024
    general_train_sample_bytes: int = 8 * 1024 * 1024
    general_eval_sample_bytes: int = 8 * 1024 * 1024
    sample_range_count: int = 8
    learning_rate: float = 5.0e-5
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.02
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    train_data_seed: int = 11101
    evaluation_seed: int = 11201
    model_seed: int = 11301
    bootstrap_samples: int = 4096
    generation_cases_per_source: int = 4
    generation_max_tokens: int = 32
    minimum_oracle_gain_over_off: float = 0.02
    minimum_lexical2_gain_over_off: float = 0.01
    minimum_lexical2_gain_over_random2: float = 0.005
    minimum_lexical2_gain_over_lexical1: float = 0.005
    minimum_true_over_wrong_gain: float = 0.02
    minimum_lexical2_target_inclusion: float = 0.85
    maximum_general_loss_regression: float = 0.10


@dataclass(frozen=True)
class PreparedDocumentSplit:
    name: str
    cases: tuple[DocumentContinuationCase, ...]
    bank: EncodedDocumentContinuations
    source_reports: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class DocumentTrainingSchedule:
    target_indices: torch.Tensor
    groups: torch.Tensor
    target_slots: torch.Tensor
    rankings: dict[str, torch.Tensor]


def _causal_config(
    config: JointDocumentRetrievalConfig,
    *,
    case_count_per_source: int,
    sample_bytes: int,
) -> CausalDocumentRetrievalConfig:
    return CausalDocumentRetrievalConfig(
        case_count_per_source=int(case_count_per_source),
        facts_per_query=int(config.facts_per_query),
        source_length=int(config.source_length),
        prefix_length=int(config.prefix_length),
        target_length=int(config.target_length),
        minimum_gap_tokens=int(config.minimum_gap_tokens),
        maximum_gap_tokens=int(config.maximum_gap_tokens),
        eval_batch_size=int(config.eval_batch_size),
        feature_batch_size=int(config.eval_batch_size),
        sample_bytes=int(sample_bytes),
        sample_range_count=int(config.sample_range_count),
        precision=str(config.precision),
        bootstrap_samples=int(config.bootstrap_samples),
    )


def _special_token_ids(tokenizer) -> tuple[int, ...]:
    return (
        int(tokenizer.pad_id),
        int(tokenizer.bos_id),
        int(tokenizer.eos_id),
        int(tokenizer.unk_id),
        int(tokenizer.checkpoint_id),
        int(tokenizer.replay_id),
    )


def _tensor_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().contiguous().cpu()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def prepare_document_split(
    tokenizer,
    paths: Sequence[str | Path],
    *,
    name: str,
    case_count_per_source: int,
    sample_bytes: int,
    config: JointDocumentRetrievalConfig,
    seed: int,
) -> PreparedDocumentSplit:
    if len(paths) != 2:
        raise ValueError(f"{name} requires exactly two corpus paths")
    causal_config = _causal_config(
        config,
        case_count_per_source=int(case_count_per_source),
        sample_bytes=int(sample_bytes),
    )
    cases: list[DocumentContinuationCase] = []
    reports = []
    for source_index, raw_path in enumerate(paths):
        path = Path(raw_path)
        text, sample_report = sample_corpus_ranges(
            path,
            byte_budget=int(sample_bytes),
            range_count=int(config.sample_range_count),
        )
        selected, selection_report = build_document_cases(
            tokenizer,
            text,
            source_index=source_index,
            source_name=path.stem,
            config=causal_config,
            seed=int(seed) + source_index,
        )
        cases.extend(selected)
        reports.append(
            {
                **sample_report,
                **selection_report,
                "file_sha256": sha256_file(path),
                "split": str(name),
            }
        )
    frozen_cases = tuple(cases)
    return PreparedDocumentSplit(
        name=str(name),
        cases=frozen_cases,
        bank=encode_document_cases(frozen_cases, config=causal_config),
        source_reports=tuple(reports),
    )


def build_document_training_schedule(
    split: PreparedDocumentSplit,
    tokenizer,
    *,
    steps: int,
    batch_size: int,
    facts_per_query: int,
    seed: int,
) -> DocumentTrainingSchedule:
    cases = split.cases
    source_buckets: dict[int, list[int]] = {}
    for index, case in enumerate(cases):
        source_buckets.setdefault(int(case.source_index), []).append(index)
    generator = random.Random(int(seed))
    target_order = []
    required = int(steps) * int(batch_size)
    while len(target_order) < required:
        epoch = list(range(len(cases)))
        generator.shuffle(epoch)
        target_order.extend(epoch)
    target_order = target_order[:required]
    groups = []
    target_slots = []
    for target in target_order:
        bucket = source_buckets[int(cases[target].source_index)]
        distractors = generator.sample(
            [index for index in bucket if index != target],
            int(facts_per_query) - 1,
        )
        row = [target, *distractors]
        generator.shuffle(row)
        groups.append(row)
        target_slots.append(row.index(target))
    targets = torch.tensor(target_order, dtype=torch.long).reshape(
        int(steps), int(batch_size)
    )
    group_tensor = torch.tensor(groups, dtype=torch.long).reshape(
        int(steps), int(batch_size), int(facts_per_query)
    )
    slot_tensor = torch.tensor(target_slots, dtype=torch.long).reshape(
        int(steps), int(batch_size)
    )
    query_bank = EncodedTextBank(
        ids=split.bank.prefix_ids.index_select(0, targets.reshape(-1)),
        mask=torch.ones(
            required,
            split.bank.prefix_ids.shape[1],
            dtype=torch.bool,
        ),
    )
    source_bank = EncodedTextBank(
        ids=split.bank.source_ids,
        mask=torch.ones_like(split.bank.source_ids, dtype=torch.bool),
    )
    lexical_scores = lexical_tfidf_scores(
        source_bank,
        query_bank,
        group_tensor.reshape(required, int(facts_per_query)),
        excluded_token_ids=_special_token_ids(tokenizer),
    )
    lexical = rankings_from_scores(lexical_scores).reshape(group_tensor.shape)
    torch_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
    random_rankings = rankings_from_scores(
        torch.rand(lexical_scores.shape, generator=torch_generator)
    ).reshape(group_tensor.shape)
    oracle_scores = torch.zeros_like(lexical_scores)
    oracle_scores.scatter_(1, slot_tensor.reshape(-1, 1), 1.0)
    oracle = rankings_from_scores(oracle_scores).reshape(group_tensor.shape)
    return DocumentTrainingSchedule(
        target_indices=targets,
        groups=group_tensor,
        target_slots=slot_tensor,
        rankings={
            "random2": random_rankings,
            "lexical1": lexical,
            "lexical2": lexical,
            "oracle2": oracle,
        },
    )


def build_document_evaluation_schedule(
    split: PreparedDocumentSplit,
    tokenizer,
    *,
    facts_per_query: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    groups, target_slots = build_archive_groups(
        split.cases,
        facts_per_query=int(facts_per_query),
        seed=int(seed),
    )
    source_bank = EncodedTextBank(
        ids=split.bank.source_ids,
        mask=torch.ones_like(split.bank.source_ids, dtype=torch.bool),
    )
    query_bank = EncodedTextBank(
        ids=split.bank.prefix_ids,
        mask=torch.ones_like(split.bank.prefix_ids, dtype=torch.bool),
    )
    lexical_scores = lexical_tfidf_scores(
        source_bank,
        query_bank,
        groups,
        excluded_token_ids=_special_token_ids(tokenizer),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
    random_scores = torch.rand(groups.shape, generator=generator)
    oracle_scores = torch.zeros(groups.shape, dtype=torch.float32)
    oracle_scores.scatter_(1, target_slots.unsqueeze(1), 1.0)
    wrong_scores = torch.zeros(groups.shape, dtype=torch.float32)
    wrong_slots_1 = (target_slots + 1) % int(facts_per_query)
    wrong_slots_2 = (target_slots + 2) % int(facts_per_query)
    wrong_scores.scatter_(1, wrong_slots_1.unsqueeze(1), 2.0)
    wrong_scores.scatter_(1, wrong_slots_2.unsqueeze(1), 1.0)
    return groups, target_slots, {
        "random2": rankings_from_scores(random_scores),
        "lexical1": rankings_from_scores(lexical_scores),
        "lexical2": rankings_from_scores(lexical_scores),
        "oracle2": rankings_from_scores(oracle_scores),
        "wrong2": rankings_from_scores(wrong_scores),
    }


def selected_slots_for_mode(
    mode: str,
    rankings: Mapping[str, torch.Tensor],
) -> torch.Tensor | None:
    if mode == "off":
        return None
    if mode == "lexical1":
        return rankings[mode][..., :1]
    if mode in {"random2", "lexical2", "oracle2", "wrong2"}:
        return rankings[mode][..., :2]
    raise ValueError(f"unknown V24 mode: {mode}")


def document_task_batch(
    split: PreparedDocumentSplit,
    schedule: DocumentTrainingSchedule,
    *,
    step_index: int,
    mode: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = schedule.target_indices[int(step_index)]
    groups = schedule.groups[int(step_index)]
    rankings = {
        name: value[int(step_index)] for name, value in schedule.rankings.items()
    }
    selected = selected_slots_for_mode(mode, rankings)
    retrieved = gather_retrieved_episodes(
        split.bank, groups, selected, device=device
    )
    return (
        retrieved,
        split.bank.query_input_ids.index_select(0, targets).to(device),
        split.bank.query_target_ids.index_select(0, targets).to(device),
        split.bank.query_loss_mask.index_select(0, targets).to(device),
    )


def document_task_loss(
    model: MarulhoHashedMicroExpertLanguageModel,
    retrieved: torch.Tensor,
    query_input: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    combined = torch.cat((retrieved, query_input), dim=1)
    if int(combined.shape[1]) > int(model.hashed_config.context_length):
        raise ValueError("V24 task context exceeds the cortex window")
    hidden = model._forward_hidden(combined, collect_telemetry=False)["hidden"]
    logits = model.lm_head(hidden[:, -int(query_input.shape[1]) :])
    return F.cross_entropy(logits[loss_mask], targets[loss_mask])


def _run_training_arm(
    mode: str,
    *,
    model: MarulhoHashedMicroExpertLanguageModel,
    initial_state: Mapping[str, torch.Tensor],
    document_split: PreparedDocumentSplit,
    document_schedule: DocumentTrainingSchedule,
    schedule: Sequence[tuple[str, int]],
    prepared_general: PreparedGeneralLanguage,
    general_baseline: Mapping[str, Any],
    config: JointDocumentRetrievalConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.load_state_dict(dict(initial_state), strict=True)
    model.train()
    torch.manual_seed(int(config.model_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.model_seed))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        fused=bool(model.device.type == "cuda"),
    )
    warmup_steps = int(round(int(config.train_steps) * float(config.warmup_fraction)))
    trace_steps = {
        max(0, min(int(config.train_steps) - 1, math.ceil(config.train_steps * x / 10) - 1))
        for x in range(1, 11)
    }
    trace = []
    document_steps = 0
    general_steps = 0
    supervised_tokens = 0
    retrieved_tokens = 0
    general_tokens = 0
    cortex_positions = 0
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    for step, (kind, source_index) in enumerate(schedule):
        learning_rate = _learning_rate(
            step,
            total_steps=int(config.train_steps),
            warmup_steps=warmup_steps,
            peak=float(config.learning_rate),
            minimum_fraction=float(config.minimum_learning_rate_fraction),
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with _precision_context(model.device, str(config.precision)):
            if kind == "relation":
                retrieved, query_input, targets, loss_mask = document_task_batch(
                    document_split,
                    document_schedule,
                    step_index=int(source_index),
                    mode=mode,
                    device=model.device,
                )
                loss = document_task_loss(
                    model, retrieved, query_input, targets, loss_mask
                )
                document_steps += 1
                supervised_tokens += int(loss_mask.sum())
                retrieved_tokens += int(retrieved.numel())
                cortex_positions += int(retrieved.numel() + query_input.numel())
            else:
                general_source = int(kind.rsplit("_", 1)[1])
                batch = prepared_general.train_batches[general_source][
                    int(source_index)
                ].to(model.device)
                loss = model.next_token_loss(
                    batch.input_ids,
                    batch.target_ids,
                    collect_telemetry=False,
                    return_evidence=False,
                )["loss"]
                general_steps += 1
                general_tokens += int(batch.target_ids.numel())
                cortex_positions += int(batch.input_ids.numel())
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite V24 loss in {mode}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config.gradient_clip)
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"non-finite V24 gradient in {mode}")
        optimizer.step()
        if step in trace_steps:
            trace.append(
                {
                    "step": step + 1,
                    "kind": kind,
                    "loss": float(loss.detach().float().cpu()),
                    "learning_rate": learning_rate,
                }
            )
        interval = max(1, int(config.train_steps) // 10)
        if (step + 1) % interval == 0 or step + 1 == int(config.train_steps):
            print(
                f"[balanced-document-v24] {mode} {step + 1}/{config.train_steps}",
                flush=True,
            )
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(model.device))
        if model.device.type == "cuda"
        else 0
    )
    model.zero_grad(set_to_none=True)
    retrieved, query_input, targets, loss_mask = document_task_batch(
        document_split,
        document_schedule,
        step_index=0,
        mode=mode,
        device=model.device,
    )
    with _precision_context(model.device, str(config.precision)):
        probe_loss = document_task_loss(
            model, retrieved, query_input, targets, loss_mask
        )
    probe_loss.backward()
    parameters = list(model.parameters())
    gradient = {
        "probe_updates_parameters": False,
        "model_parameter_tensor_count": len(parameters),
        "model_parameter_tensors_with_gradient": sum(
            parameter.grad is not None for parameter in parameters
        ),
        "model_parameter_tensors_with_nonzero_gradient": sum(
            parameter.grad is not None
            and int(torch.count_nonzero(parameter.grad).detach().cpu()) > 0
            for parameter in parameters
        ),
        "hashed_expert_rows": model.final_gradient_report(),
    }
    model.zero_grad(set_to_none=True)
    return (
        {
            "optimizer": "AdamW",
            "optimizer_state_fresh": True,
            "initial_model_state_restored": True,
            "steps": int(config.train_steps),
            "document_steps": document_steps,
            "general_replay_steps": general_steps,
            "supervised_target_tokens": supervised_tokens,
            "retrieved_source_tokens": retrieved_tokens,
            "general_training_tokens": general_tokens,
            "cortex_input_positions": cortex_positions,
            "elapsed_seconds": elapsed,
            "cortex_positions_per_second": cortex_positions / max(elapsed, 1.0e-12),
            "peak_cuda_memory_bytes": peak_memory,
            "loss_trace": trace,
            "gradient": gradient,
            "execution_backend": "eager",
        },
        evaluate_general_language(
            model, prepared_general.eval_batches, baseline=general_baseline
        ),
    )


def _retrieval_inclusion(
    selected: torch.Tensor | None,
    target_slots: torch.Tensor,
) -> float:
    if selected is None:
        return 0.0
    return float((selected == target_slots.unsqueeze(1)).any(dim=1).float().mean())


def evaluate_document_suite(
    mode: str,
    *,
    model: MarulhoHashedMicroExpertLanguageModel,
    split: PreparedDocumentSplit,
    groups: torch.Tensor,
    target_slots: torch.Tensor,
    rankings: Mapping[str, torch.Tensor],
    config: JointDocumentRetrievalConfig,
) -> dict[str, Any]:
    contexts = {}
    raw_losses = {}
    raw_accuracy = {}
    for context_mode in (
        "off",
        "random2",
        "lexical1",
        "lexical2",
        "oracle2",
        "wrong2",
    ):
        selected = selected_slots_for_mode(context_mode, rankings)
        row = evaluate_document_arm(
            model,
            split.bank,
            groups,
            selected,
            split.cases,
            batch_size=int(config.eval_batch_size),
            precision=str(config.precision),
        )
        raw_losses[context_mode] = row.pop("case_losses")
        raw_accuracy[context_mode] = row.pop("case_next_token_accuracy")
        row["target_inclusion"] = _retrieval_inclusion(selected, target_slots)
        contexts[context_mode] = row
    primary = contexts[mode]
    source_use = {
        "true_over_wrong": paired_bootstrap_gain(
            raw_losses["wrong2"],
            raw_losses["oracle2"],
            samples=int(config.bootstrap_samples),
            seed=int(config.evaluation_seed) + 200,
        ),
        "true_over_local": paired_bootstrap_gain(
            raw_losses["off"],
            raw_losses["oracle2"],
            samples=int(config.bootstrap_samples),
            seed=int(config.evaluation_seed) + 201,
        ),
        "lexical_over_local": paired_bootstrap_gain(
            raw_losses["off"],
            raw_losses["lexical2"],
            samples=int(config.bootstrap_samples),
            seed=int(config.evaluation_seed) + 202,
        ),
    }
    return {
        "primary_mode": mode,
        "primary": primary,
        "contexts": contexts,
        "source_use": source_use,
        "_primary_case_losses": raw_losses[mode],
        "_primary_case_accuracy": raw_accuracy[mode],
    }


@torch.no_grad()
def generate_document_samples(
    mode: str,
    *,
    model: MarulhoHashedMicroExpertLanguageModel,
    tokenizer,
    split: PreparedDocumentSplit,
    groups: torch.Tensor,
    rankings: Mapping[str, torch.Tensor],
    config: JointDocumentRetrievalConfig,
) -> dict[str, Any]:
    chosen = []
    for source_name in sorted({case.source_name for case in split.cases}):
        indices = [
            index for index, case in enumerate(split.cases) if case.source_name == source_name
        ]
        chosen.extend(indices[: int(config.generation_cases_per_source)])
    index_tensor = torch.tensor(chosen, dtype=torch.long)
    selected = selected_slots_for_mode(mode, rankings)
    selected_subset = None if selected is None else selected.index_select(0, index_tensor)
    retrieved = gather_retrieved_episodes(
        split.bank,
        groups.index_select(0, index_tensor),
        selected_subset,
        device=model.device,
    )
    prefixes = split.bank.prefix_ids.index_select(0, index_tensor).to(model.device)
    prompt = torch.cat((retrieved, prefixes), dim=1)
    generated = model.generate(
        prompt,
        max_new_tokens=int(config.generation_max_tokens),
        eos_id=int(tokenizer.eos_id),
        repetition_penalty=1.05,
        no_repeat_ngram_size=3,
        temperature=0.0,
    )["generated_ids"][:, int(prompt.shape[1]) :].cpu()
    special = set(_special_token_ids(tokenizer))
    rows = []
    token_position_correct = 0
    token_position_count = 0
    recall_rows = []
    for row_index, case_index in enumerate(chosen):
        case = split.cases[case_index]
        output_ids = [
            int(value)
            for value in generated[row_index].tolist()
            if int(value) != int(tokenizer.eos_id)
        ]
        expected = list(case.target_ids)
        compared = min(len(output_ids), len(expected))
        token_position_correct += sum(
            output_ids[index] == expected[index] for index in range(compared)
        )
        token_position_count += len(expected)
        expected_set = {value for value in expected if value not in special}
        generated_set = {value for value in output_ids if value not in special}
        recall_rows.append(
            len(expected_set & generated_set) / max(1, len(expected_set))
        )
        rows.append(
            {
                "case_id": case.case_id,
                "source_name": case.source_name,
                "archive_episode": tokenizer.decode(case.source_ids),
                "visible_prefix": tokenizer.decode(case.prefix_ids),
                "expected_continuation": tokenizer.decode(case.target_ids),
                "generated_continuation": tokenizer.decode(output_ids),
            }
        )
    return {
        "case_count": len(chosen),
        "generation_policy": "greedy_repetition_penalty_1.05_no_repeat_trigram",
        "maximum_new_tokens": int(config.generation_max_tokens),
        "expected_token_position_accuracy": token_position_correct
        / max(1, token_position_count),
        "mean_expected_unique_token_recall": sum(recall_rows) / len(recall_rows),
        "examples": rows,
    }


def joint_document_decision(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    train_steps: int,
    config: JointDocumentRetrievalConfig,
) -> str:
    if set(rows) != set(ARM_NAMES):
        return "incomplete_v24_missing_control_arm"
    if int(train_steps) < MINIMUM_DECISION_STEPS:
        return "diagnostic_v24_below_screen_step_floor"
    oracle = rows["oracle2"]["matched_to_off"]
    if (
        float(oracle["mean_loss_gain"])
        < float(config.minimum_oracle_gain_over_off)
        or float(oracle["bootstrap_95_ci"][0]) <= 0.0
    ):
        return "retire_v24_document_task_not_learnable_with_oracle_history"
    lexical = rows["lexical2"]
    lexical_gain = lexical["matched_to_off"]
    lexical_loss = float(lexical["evaluation"]["primary"]["heldout_loss"])
    random_loss = float(rows["random2"]["evaluation"]["primary"]["heldout_loss"])
    lexical1_loss = float(rows["lexical1"]["evaluation"]["primary"]["heldout_loss"])
    true_wrong = lexical["evaluation"]["source_use"]["true_over_wrong"]
    maximum_general_regression = max(
        float(source["heldout_loss_delta"])
        for source in lexical["general_language"]["sources"]
    )
    if maximum_general_regression > float(config.maximum_general_loss_regression):
        return "retire_v24_lexical_two_breaks_general_language"
    if (
        float(lexical["evaluation"]["primary"]["target_inclusion"])
        >= float(config.minimum_lexical2_target_inclusion)
        and float(lexical_gain["mean_loss_gain"])
        >= float(config.minimum_lexical2_gain_over_off)
        and float(lexical_gain["bootstrap_95_ci"][0]) > 0.0
        and random_loss - lexical_loss
        >= float(config.minimum_lexical2_gain_over_random2)
        and lexical1_loss - lexical_loss
        >= float(config.minimum_lexical2_gain_over_lexical1)
        and float(true_wrong["mean_loss_gain"])
        >= float(config.minimum_true_over_wrong_gain)
        and float(true_wrong["bootstrap_95_ci"][0]) > 0.0
    ):
        return ADVANCE_DECISION
    return "retire_v24_balanced_top_two_no_joint_language_win"


def run_joint_document_retrieval_screen(
    *,
    parent_checkpoint_path: str | Path,
    document_train_paths: Sequence[str | Path],
    document_eval_paths: Sequence[str | Path],
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    output_path: str | Path,
    config: JointDocumentRetrievalConfig = JointDocumentRetrievalConfig(),
    device: str = "auto",
) -> dict[str, Any]:
    if int(config.facts_per_query) != 4:
        raise ValueError("V24 currently requires four archive candidates")
    if not 0.0 < float(config.document_fraction) < 1.0:
        raise ValueError("V24 document_fraction must be strictly between zero and one")
    resolved = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for V24 but unavailable")
    started = time.perf_counter()
    parent_path = Path(parent_checkpoint_path)
    model, tokenizer, parent_metadata = load_hashed_micro_expert_checkpoint(
        parent_path, map_location="cpu"
    )
    parent_tokens = _validate_parent(model, parent_metadata)
    if parent_tokens < 1_000_000_000:
        raise ValueError("V24 requires the one-billion-token V11 parent")
    active_length = (
        2 * int(config.source_length)
        + int(config.prefix_length)
        + int(config.target_length)
        - 1
    )
    if max(active_length, int(config.general_sequence_length)) > int(
        model.hashed_config.context_length
    ):
        raise ValueError("V24 active sequence exceeds the parent context")

    print("[balanced-document-v24] preparing causal document splits", flush=True)
    train_split = prepare_document_split(
        tokenizer,
        document_train_paths,
        name="document_training",
        case_count_per_source=int(config.train_cases_per_source),
        sample_bytes=int(config.document_train_sample_bytes),
        config=config,
        seed=int(config.train_data_seed),
    )
    eval_split = prepare_document_split(
        tokenizer,
        document_eval_paths,
        name="document_disjoint_evaluation",
        case_count_per_source=int(config.eval_cases_per_source),
        sample_bytes=int(config.document_eval_sample_bytes),
        config=config,
        seed=int(config.evaluation_seed),
    )
    train_hashes = {case.document_sha256 for case in train_split.cases}
    eval_hashes = {case.document_sha256 for case in eval_split.cases}
    overlap = train_hashes & eval_hashes
    if overlap:
        raise RuntimeError("V24 train/eval document hashes overlap")
    document_steps = _scheduled_relation_steps(
        int(config.train_steps), float(config.document_fraction)
    )
    document_schedule = build_document_training_schedule(
        train_split,
        tokenizer,
        steps=document_steps,
        batch_size=int(config.batch_size),
        facts_per_query=int(config.facts_per_query),
        seed=int(config.train_data_seed) + 10,
    )
    eval_groups, eval_target_slots, eval_rankings = build_document_evaluation_schedule(
        eval_split,
        tokenizer,
        facts_per_query=int(config.facts_per_query),
        seed=int(config.evaluation_seed) + 10,
    )
    general_config = RetrievalScreenConfig(
        train_steps=int(config.train_steps),
        batch_size=int(config.batch_size),
        general_sequence_length=int(config.general_sequence_length),
        general_eval_batches=int(config.general_eval_batches),
        relation_fraction=float(config.document_fraction),
        general_train_sample_bytes=int(config.general_train_sample_bytes),
        general_eval_sample_bytes=int(config.general_eval_sample_bytes),
        sample_range_count=int(config.sample_range_count),
        precision=str(config.precision),
    )
    prepared_general = prepare_general_language(
        tokenizer,
        train_paths=general_train_paths,
        eval_paths=general_eval_paths,
        config=general_config,
    )
    schedule = build_matched_schedule(
        step_count=int(config.train_steps),
        relation_fraction=float(config.document_fraction),
        relation_batch_count=document_steps,
        general_batch_counts=[len(rows) for rows in prepared_general.train_batches],
        seed=int(config.train_data_seed) + 20,
    )
    initial_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    model = model.to(resolved)
    print("[balanced-document-v24] evaluating untouched general baseline", flush=True)
    general_baseline = evaluate_general_language(model, prepared_general.eval_batches)

    rows: dict[str, dict[str, Any]] = {}
    for mode in ARM_NAMES:
        print(f"[balanced-document-v24] training {mode}", flush=True)
        training, general_after = _run_training_arm(
            mode,
            model=model,
            initial_state=initial_state,
            document_split=train_split,
            document_schedule=document_schedule,
            schedule=schedule,
            prepared_general=prepared_general,
            general_baseline=general_baseline,
            config=config,
        )
        evaluation = evaluate_document_suite(
            mode,
            model=model,
            split=eval_split,
            groups=eval_groups,
            target_slots=eval_target_slots,
            rankings=eval_rankings,
            config=config,
        )
        generation = generate_document_samples(
            mode,
            model=model,
            tokenizer=tokenizer,
            split=eval_split,
            groups=eval_groups,
            rankings=eval_rankings,
            config=config,
        )
        rows[mode] = {
            "mode": mode,
            "promotable": mode != "oracle2",
            "training": training,
            "general_language": general_after,
            "evaluation": evaluation,
            "generation": generation,
        }
        print(
            f"[balanced-document-v24] {mode} loss="
            f"{evaluation['primary']['heldout_loss']:.4f} general_delta="
            f"{general_after['aggregate_heldout_loss_delta']:+.4f}",
            flush=True,
        )

    baseline_losses = rows["off"]["evaluation"]["_primary_case_losses"]
    for index, mode in enumerate(ARM_NAMES):
        candidate_losses = rows[mode]["evaluation"]["_primary_case_losses"]
        rows[mode]["matched_to_off"] = paired_bootstrap_gain(
            baseline_losses,
            candidate_losses,
            samples=int(config.bootstrap_samples),
            seed=int(config.evaluation_seed) + 500 + index,
        )
    decision = joint_document_decision(
        rows, train_steps=int(config.train_steps), config=config
    )
    for mode in ARM_NAMES:
        rows[mode]["evaluation"].pop("_primary_case_losses")
        rows[mode]["evaluation"].pop("_primary_case_accuracy")

    train_lexical = document_schedule.rankings["lexical2"]
    train_inclusion = float(
        (
            train_lexical[..., :2]
            == document_schedule.target_slots.unsqueeze(-1)
        )
        .any(dim=-1)
        .float()
        .mean()
    )
    eval_lexical = eval_rankings["lexical2"]
    eval_inclusion = float(
        (eval_lexical[:, :2] == eval_target_slots.unsqueeze(1)).any(dim=1).float().mean()
    )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "configuration": asdict(config),
        "parent": {
            "path": str(parent_path),
            "sha256": sha256_file(parent_path),
            "processed_tokens": parent_tokens,
            "decision": parent_metadata.get("decision"),
            "tokenizer_hash": tokenizer.vocabulary_hash(),
            "parameters_frozen": False,
            "parameter_gradients_enabled": True,
        },
        "architecture": {
            "archive_content": "exact_prior_document_token_spans",
            "archive_key": "checkpoint_bpe_tfidf",
            "candidate_selected_episode_count": 2,
            "maximum_active_source_tokens": 2 * int(config.source_length),
            "ordinary_cortex_attention_reads_episode": True,
            "separate_memory_reader": False,
            "selector_learned": False,
            "cortex_jointly_trained_with_interface": True,
        },
        "sources": {
            "document_train": train_split.source_reports,
            "document_eval": eval_split.source_reports,
            "general": prepared_general.source_reports,
        },
        "schedule": {
            "identical_optimizer_step_schedule": True,
            "schedule_sha256": schedule_sha256(schedule),
            "document_group_sha256": _tensor_sha256(
                document_schedule.target_indices,
                document_schedule.groups,
                document_schedule.target_slots,
            ),
            "document_lexical_ranking_sha256": _tensor_sha256(train_lexical),
            "eval_group_sha256": _tensor_sha256(eval_groups, eval_target_slots),
            "eval_lexical_ranking_sha256": _tensor_sha256(eval_lexical),
            "train_lexical2_target_inclusion": train_inclusion,
            "eval_lexical2_target_inclusion": eval_inclusion,
            "document_steps": document_steps,
            "general_steps": int(config.train_steps) - document_steps,
        },
        "anti_cheat": {
            "archive_episode_precedes_visible_prefix": True,
            "retrieval_input": "visible_prefix_only",
            "retrieval_uses_future_target": False,
            "retrieval_uses_document_identity": False,
            "target_slot_metrics_only": True,
            "oracle_training_uses_target_slot": True,
            "oracle_promotable": False,
            "teacher_forcing_target_visible_only_after_retrieval": True,
            "train_eval_document_hash_overlap": len(overlap),
            "random_control_same_active_source_tokens": True,
            "initial_model_state_identical": True,
        },
        "general_language_before": general_baseline,
        "arms": rows,
        "decision": decision,
        "checkpoint": None,
        "promotion_boundary": {
            "advance_to_anchored_review": decision == ADVANCE_DECISION,
            "base_quality_promoted": False,
            "checkpoint_saved": False,
            "runtime_install_allowed": False,
            "continual_learning_claimed": False,
            "speed_claimed": False,
        },
        "hardware": {
            "device": str(resolved),
            "cuda_device_name": (
                torch.cuda.get_device_name(resolved)
                if resolved.type == "cuda"
                else None
            ),
            "torch_version": torch.__version__,
        },
        "experiment_wall_seconds": time.perf_counter() - started,
    }
    write_json_report_with_readme(
        output_path,
        report,
        title="MARULHO V24 Balanced Top-Two Document Retrieval Screen",
    )
    print(f"[balanced-document-v24] decision {decision}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--document-train", type=Path, nargs=2, required=True)
    parser.add_argument("--document-eval", type=Path, nargs=2, required=True)
    parser.add_argument("--general-train", type=Path, nargs=2, required=True)
    parser.add_argument("--general-eval", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-cases-per-source", type=int, default=2048)
    parser.add_argument("--eval-cases-per-source", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--document-train-sample-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--document-eval-sample-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--bootstrap-samples", type=int, default=4096)
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = JointDocumentRetrievalConfig(
        train_cases_per_source=int(args.train_cases_per_source),
        eval_cases_per_source=int(args.eval_cases_per_source),
        train_steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        document_train_sample_bytes=int(args.document_train_sample_bytes),
        document_eval_sample_bytes=int(args.document_eval_sample_bytes),
        bootstrap_samples=int(args.bootstrap_samples),
        precision=str(args.precision),
    )
    run_joint_document_retrieval_screen(
        parent_checkpoint_path=args.parent_checkpoint,
        document_train_paths=args.document_train,
        document_eval_paths=args.document_eval,
        general_train_paths=args.general_train,
        general_eval_paths=args.general_eval,
        output_path=args.output,
        config=config,
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
