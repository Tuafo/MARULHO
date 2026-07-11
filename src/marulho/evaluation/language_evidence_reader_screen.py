"""Test an interleaved evidence reader against raw and shuffled controls."""

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

from marulho.evaluation.language_causal_document_retrieval_audit import (
    CausalDocumentRetrievalConfig,
    DocumentContinuationCase,
    EncodedDocumentContinuations,
    build_archive_groups,
    build_document_cases,
    encode_document_cases,
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
from marulho.training.language_evidence_reader import (
    EvidenceReaderConfig,
    MarulhoEvidenceReaderLanguageModel,
)
from marulho.training.language_hashed_micro_experts import (
    load_hashed_micro_expert_checkpoint,
)


SURFACE = "marulho_interleaved_evidence_reader_screen.v2"
ARTIFACT_KIND = "marulho_interleaved_evidence_reader_screen"
ARM_NAMES = (
    "gate_zero",
    "shuffled_reader",
    "raw_context",
    "lexical_reader",
    "oracle_reader",
)
ADVANCE_DECISION = "advance_v27_interleaved_evidence_reader_to_manual_review"
MINIMUM_DECISION_STEPS = 512


@dataclass(frozen=True)
class EvidenceReaderScreenConfig:
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
    train_data_seed: int = 14101
    evaluation_seed: int = 14201
    model_seed: int = 14301
    bootstrap_samples: int = 4096
    generation_cases_per_source: int = 4
    generation_max_tokens: int = 48
    reader_attention_heads: int = 8
    reader_gate_logit_initial: float = -2.0
    reader_injection_layer_indices: tuple[int, ...] = (0, 2)
    minimum_oracle_gain_over_zero: float = 0.02
    minimum_reader_gain_over_zero: float = 0.01
    minimum_reader_gain_over_shuffled: float = 0.005
    minimum_reader_gain_over_raw: float = 0.005
    minimum_true_over_wrong_gain: float = 0.02
    minimum_lexical_target_inclusion: float = 0.68
    minimum_per_source_gain: float = 0.0
    minimum_generation_source_swap_rate: float = 0.50
    maximum_general_loss_regression: float = 0.10


@dataclass(frozen=True)
class PreparedDocumentSplit:
    name: str
    cases: tuple[DocumentContinuationCase, ...]
    bank: EncodedDocumentContinuations
    source_reports: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class DocumentSchedule:
    target_indices: torch.Tensor
    groups: torch.Tensor
    target_slots: torch.Tensor
    rankings: dict[str, torch.Tensor]


def _causal_config(
    config: EvidenceReaderScreenConfig,
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
    config: EvidenceReaderScreenConfig,
    seed: int,
) -> PreparedDocumentSplit:
    if len(paths) != 2:
        raise ValueError(f"{name} requires exactly two corpus paths")
    causal = _causal_config(
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
            config=causal,
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
        bank=encode_document_cases(frozen_cases, config=causal),
        source_reports=tuple(reports),
    )


def _build_rankings(
    split: PreparedDocumentSplit,
    tokenizer,
    groups: torch.Tensor,
    target_slots: torch.Tensor,
    query_case_indices: torch.Tensor,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    facts = int(groups.shape[-1])
    flat_groups = groups.reshape(-1, facts)
    flat_queries = query_case_indices.reshape(-1)
    source_bank = EncodedTextBank(
        ids=split.bank.source_ids,
        mask=torch.ones_like(split.bank.source_ids, dtype=torch.bool),
    )
    query_ids = split.bank.prefix_ids.index_select(0, flat_queries)
    query_bank = EncodedTextBank(
        ids=query_ids,
        mask=torch.ones_like(query_ids, dtype=torch.bool),
    )
    lexical_scores = lexical_tfidf_scores(
        source_bank,
        query_bank,
        flat_groups,
        excluded_token_ids=_special_token_ids(tokenizer),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random_scores = torch.rand(lexical_scores.shape, generator=generator)
    oracle_scores = torch.zeros_like(lexical_scores)
    oracle_scores.scatter_(1, target_slots.reshape(-1, 1), 1.0)
    wrong_scores = torch.zeros_like(lexical_scores)
    wrong_slots = (target_slots.reshape(-1) + 1) % facts
    wrong_scores.scatter_(1, wrong_slots.unsqueeze(1), 1.0)
    shape = groups.shape
    return {
        "random": rankings_from_scores(random_scores).reshape(shape),
        "lexical": rankings_from_scores(lexical_scores).reshape(shape),
        "oracle": rankings_from_scores(oracle_scores).reshape(shape),
        "wrong": rankings_from_scores(wrong_scores).reshape(shape),
    }


def build_training_schedule(
    split: PreparedDocumentSplit,
    tokenizer,
    *,
    steps: int,
    batch_size: int,
    facts_per_query: int,
    seed: int,
) -> DocumentSchedule:
    source_buckets: dict[int, list[int]] = {}
    for index, case in enumerate(split.cases):
        source_buckets.setdefault(int(case.source_index), []).append(index)
    generator = random.Random(int(seed))
    required = int(steps) * int(batch_size)
    targets = []
    while len(targets) < required:
        epoch = list(range(len(split.cases)))
        generator.shuffle(epoch)
        targets.extend(epoch)
    targets = targets[:required]
    groups = []
    slots = []
    for target in targets:
        bucket = source_buckets[int(split.cases[target].source_index)]
        distractors = generator.sample(
            [value for value in bucket if value != target],
            int(facts_per_query) - 1,
        )
        row = [target, *distractors]
        generator.shuffle(row)
        groups.append(row)
        slots.append(row.index(target))
    target_tensor = torch.tensor(targets, dtype=torch.long).reshape(
        int(steps), int(batch_size)
    )
    group_tensor = torch.tensor(groups, dtype=torch.long).reshape(
        int(steps), int(batch_size), int(facts_per_query)
    )
    slot_tensor = torch.tensor(slots, dtype=torch.long).reshape(
        int(steps), int(batch_size)
    )
    return DocumentSchedule(
        target_indices=target_tensor,
        groups=group_tensor,
        target_slots=slot_tensor,
        rankings=_build_rankings(
            split,
            tokenizer,
            group_tensor,
            slot_tensor,
            target_tensor,
            seed=int(seed) + 1,
        ),
    )


def build_evaluation_schedule(
    split: PreparedDocumentSplit,
    tokenizer,
    *,
    facts_per_query: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    groups, slots = build_archive_groups(
        split.cases,
        facts_per_query=int(facts_per_query),
        seed=int(seed),
    )
    queries = torch.arange(len(split.cases), dtype=torch.long)
    return groups, slots, _build_rankings(
        split,
        tokenizer,
        groups,
        slots,
        queries,
        seed=int(seed) + 1,
    )


def arm_interface_and_policy(arm: str) -> tuple[str, str | None]:
    if arm == "gate_zero":
        return "gate_zero", None
    if arm == "shuffled_reader":
        return "separate_reader", "random"
    if arm == "raw_context":
        return "raw_context", "lexical"
    if arm == "lexical_reader":
        return "separate_reader", "lexical"
    if arm == "oracle_reader":
        return "separate_reader", "oracle"
    if arm == "wrong_reader":
        return "separate_reader", "wrong"
    raise ValueError(f"unknown V27 arm: {arm}")


def _selected_slots(
    policy: str | None,
    rankings: Mapping[str, torch.Tensor],
) -> torch.Tensor | None:
    return None if policy is None else rankings[policy][..., :1]


def document_batch(
    split: PreparedDocumentSplit,
    schedule: DocumentSchedule,
    *,
    step_index: int,
    arm: str,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    interface, policy = arm_interface_and_policy(arm)
    targets = schedule.target_indices[int(step_index)]
    groups = schedule.groups[int(step_index)]
    rankings = {
        name: values[int(step_index)] for name, values in schedule.rankings.items()
    }
    selected = _selected_slots(policy, rankings)
    evidence = (
        None
        if selected is None
        else gather_retrieved_episodes(
            split.bank, groups, selected, device=device
        )
    )
    return (
        evidence,
        split.bank.query_input_ids.index_select(0, targets).to(device),
        split.bank.query_target_ids.index_select(0, targets).to(device),
        split.bank.query_loss_mask.index_select(0, targets).to(device),
        interface,
    )


def _gradient_report(model: MarulhoEvidenceReaderLanguageModel) -> dict[str, Any]:
    cortex = []
    reader = []
    for name, parameter in model.named_parameters():
        row = {
            "name": name,
            "received_gradient": parameter.grad is not None,
            "nonzero_gradient": (
                parameter.grad is not None
                and int(torch.count_nonzero(parameter.grad).detach().cpu()) > 0
            ),
        }
        (cortex if name.startswith("cortex.") else reader).append(row)
    return {
        "probe_updates_parameters": False,
        "cortex_parameter_tensors": len(cortex),
        "cortex_tensors_with_gradient": sum(row["received_gradient"] for row in cortex),
        "cortex_tensors_with_nonzero_gradient": sum(row["nonzero_gradient"] for row in cortex),
        "reader_parameter_tensors": len(reader),
        "reader_tensors_with_gradient": sum(row["received_gradient"] for row in reader),
        "reader_tensors_with_nonzero_gradient": sum(row["nonzero_gradient"] for row in reader),
    }


def train_arm(
    arm: str,
    *,
    model: MarulhoEvidenceReaderLanguageModel,
    initial_state: Mapping[str, torch.Tensor],
    split: PreparedDocumentSplit,
    document_schedule: DocumentSchedule,
    schedule: Sequence[tuple[str, int]],
    prepared_general: PreparedGeneralLanguage,
    general_baseline: Mapping[str, Any],
    config: EvidenceReaderScreenConfig,
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
    warmup = int(round(int(config.train_steps) * float(config.warmup_fraction)))
    trace_steps = {
        max(0, min(int(config.train_steps) - 1, math.ceil(config.train_steps * x / 10) - 1))
        for x in range(1, 11)
    }
    trace = []
    document_steps = 0
    general_steps = 0
    target_tokens = 0
    evidence_tokens = 0
    general_tokens = 0
    cortex_positions = 0
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    for step, (kind, source_index) in enumerate(schedule):
        rate = _learning_rate(
            step,
            total_steps=int(config.train_steps),
            warmup_steps=warmup,
            peak=float(config.learning_rate),
            minimum_fraction=float(config.minimum_learning_rate_fraction),
        )
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        with _precision_context(model.device, str(config.precision)):
            if kind == "relation":
                evidence, query, targets, mask, interface = document_batch(
                    split,
                    document_schedule,
                    step_index=int(source_index),
                    arm=arm,
                    device=model.device,
                )
                loss = model.masked_next_token_loss(
                    query,
                    targets,
                    mask,
                    evidence,
                    interface=interface,
                )
                document_steps += 1
                target_tokens += int(mask.sum())
                evidence_tokens += 0 if evidence is None else int(evidence.numel())
                cortex_positions += int(query.numel()) + (
                    0 if evidence is None else int(evidence.numel())
                )
            else:
                source = int(kind.rsplit("_", 1)[1])
                batch = prepared_general.train_batches[source][int(source_index)].to(
                    model.device
                )
                loss = model.cortex.next_token_loss(
                    batch.input_ids,
                    batch.target_ids,
                    collect_telemetry=False,
                    return_evidence=False,
                )["loss"]
                general_steps += 1
                general_tokens += int(batch.target_ids.numel())
                cortex_positions += int(batch.input_ids.numel())
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite V27 loss in {arm}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config.gradient_clip)
        )
        if not bool(torch.isfinite(norm)):
            raise RuntimeError(f"non-finite V27 gradient in {arm}")
        optimizer.step()
        if step in trace_steps:
            trace.append(
                {
                    "step": step + 1,
                    "kind": kind,
                    "loss": float(loss.detach().float().cpu()),
                    "learning_rate": rate,
                }
            )
        interval = max(1, int(config.train_steps) // 10)
        if (step + 1) % interval == 0 or step + 1 == int(config.train_steps):
            print(
                f"[interleaved-reader-v27] {arm} {step + 1}/{config.train_steps}",
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
    evidence, query, targets, mask, interface = document_batch(
        split,
        document_schedule,
        step_index=0,
        arm=arm,
        device=model.device,
    )
    with _precision_context(model.device, str(config.precision)):
        probe = model.masked_next_token_loss(
            query, targets, mask, evidence, interface=interface
        )
    probe.backward()
    gradient = _gradient_report(model)
    model.zero_grad(set_to_none=True)
    return (
        {
            "optimizer": "AdamW",
            "optimizer_state_fresh": True,
            "initial_model_state_restored": True,
            "steps": int(config.train_steps),
            "document_steps": document_steps,
            "general_replay_steps": general_steps,
            "supervised_target_tokens": target_tokens,
            "evidence_tokens": evidence_tokens,
            "general_training_tokens": general_tokens,
            "cortex_input_positions": cortex_positions,
            "elapsed_seconds": elapsed,
            "cortex_positions_per_second": cortex_positions / max(elapsed, 1.0e-12),
            "peak_cuda_memory_bytes": peak_memory,
            "loss_trace": trace,
            "gradient": gradient,
            "reader": model.reader_parameter_report(),
            "execution_backend": "eager",
        },
        evaluate_general_language(
            model.cortex, prepared_general.eval_batches, baseline=general_baseline
        ),
    )


@torch.no_grad()
def evaluate_interface(
    arm: str,
    *,
    model: MarulhoEvidenceReaderLanguageModel,
    split: PreparedDocumentSplit,
    groups: torch.Tensor,
    target_slots: torch.Tensor,
    rankings: Mapping[str, torch.Tensor],
    config: EvidenceReaderScreenConfig,
) -> dict[str, Any]:
    interface, policy = arm_interface_and_policy(arm)
    selected = _selected_slots(policy, rankings)
    losses = []
    accuracies = []
    for start in range(0, len(split.cases), int(config.eval_batch_size)):
        end = min(len(split.cases), start + int(config.eval_batch_size))
        batch_selected = None if selected is None else selected[start:end]
        evidence = (
            None
            if batch_selected is None
            else gather_retrieved_episodes(
                split.bank,
                groups[start:end],
                batch_selected,
                device=model.device,
            )
        )
        query = split.bank.query_input_ids[start:end].to(model.device)
        targets = split.bank.query_target_ids[start:end].to(model.device)
        mask = split.bank.query_loss_mask[start:end].to(model.device)
        with _precision_context(model.device, str(config.precision)):
            logits = model.forward(
                query,
                evidence,
                interface=interface,
                collect_telemetry=False,
            )["logits"]
        token_losses = torch.nn.functional.cross_entropy(
            logits.float().transpose(1, 2), targets, reduction="none"
        )
        losses.append(((token_losses * mask).sum(1) / mask.sum(1)).cpu())
        predictions = logits.argmax(-1)
        accuracies.append(
            (((predictions == targets) & mask).sum(1) / mask.sum(1)).float().cpu()
        )
    case_losses = torch.cat(losses)
    case_accuracy = torch.cat(accuracies)
    if selected is None:
        inclusion = 0.0
    else:
        inclusion = float(
            (selected == target_slots.unsqueeze(1)).any(dim=1).float().mean()
        )
    per_source = {}
    for name in sorted({case.source_name for case in split.cases}):
        indices = torch.tensor(
            [index for index, case in enumerate(split.cases) if case.source_name == name],
            dtype=torch.long,
        )
        per_source[name] = {
            "case_count": int(indices.numel()),
            "heldout_loss": float(case_losses.index_select(0, indices).mean()),
            "next_token_accuracy": float(case_accuracy.index_select(0, indices).mean()),
        }
    return {
        "interface": interface,
        "policy": policy,
        "heldout_loss": float(case_losses.mean()),
        "next_token_accuracy": float(case_accuracy.mean()),
        "target_inclusion": inclusion,
        "per_source": per_source,
        "case_count": len(split.cases),
        "_case_losses": [float(value) for value in case_losses],
        "_case_accuracy": [float(value) for value in case_accuracy],
    }


def evaluate_suite(
    primary_arm: str,
    *,
    model: MarulhoEvidenceReaderLanguageModel,
    split: PreparedDocumentSplit,
    groups: torch.Tensor,
    target_slots: torch.Tensor,
    rankings: Mapping[str, torch.Tensor],
    config: EvidenceReaderScreenConfig,
) -> dict[str, Any]:
    context_names = (*ARM_NAMES, "wrong_reader")
    contexts = {}
    raw = {}
    for name in context_names:
        row = evaluate_interface(
            name,
            model=model,
            split=split,
            groups=groups,
            target_slots=target_slots,
            rankings=rankings,
            config=config,
        )
        raw[name] = row.pop("_case_losses")
        row.pop("_case_accuracy")
        contexts[name] = row
    return {
        "primary_arm": primary_arm,
        "primary": contexts[primary_arm],
        "contexts": contexts,
        "source_use": {
            "true_over_wrong_reader": paired_bootstrap_gain(
                raw["wrong_reader"],
                raw["oracle_reader"],
                samples=int(config.bootstrap_samples),
                seed=int(config.evaluation_seed) + 300,
            ),
            "lexical_reader_over_zero": paired_bootstrap_gain(
                raw["gate_zero"],
                raw["lexical_reader"],
                samples=int(config.bootstrap_samples),
                seed=int(config.evaluation_seed) + 301,
            ),
            "reader_over_raw": paired_bootstrap_gain(
                raw["raw_context"],
                raw["lexical_reader"],
                samples=int(config.bootstrap_samples),
                seed=int(config.evaluation_seed) + 302,
            ),
        },
        "_primary_case_losses": raw[primary_arm],
    }


def _generation_indices(
    cases: Sequence[DocumentContinuationCase],
    *,
    per_source: int,
) -> list[int]:
    selected = []
    for name in sorted({case.source_name for case in cases}):
        indices = [index for index, case in enumerate(cases) if case.source_name == name]
        selected.extend(indices[: int(per_source)])
    return selected


@torch.no_grad()
def generate_samples(
    arm: str,
    *,
    model: MarulhoEvidenceReaderLanguageModel,
    tokenizer,
    split: PreparedDocumentSplit,
    groups: torch.Tensor,
    rankings: Mapping[str, torch.Tensor],
    config: EvidenceReaderScreenConfig,
) -> dict[str, Any]:
    indices = _generation_indices(
        split.cases, per_source=int(config.generation_cases_per_source)
    )
    index_tensor = torch.tensor(indices, dtype=torch.long)
    interface, policy = arm_interface_and_policy(arm)
    selected = _selected_slots(policy, rankings)
    subset = None if selected is None else selected.index_select(0, index_tensor)
    evidence = (
        None
        if subset is None
        else gather_retrieved_episodes(
            split.bank,
            groups.index_select(0, index_tensor),
            subset,
            device=model.device,
        )
    )
    query = split.bank.prefix_ids.index_select(0, index_tensor).to(model.device)
    generated = model.generate_with_evidence(
        query,
        evidence,
        interface=interface,
        max_new_tokens=int(config.generation_max_tokens),
        eos_id=int(tokenizer.eos_id),
    )["generated_ids"][:, int(query.shape[1]) :].cpu()
    special = set(_special_token_ids(tokenizer))
    rows = []
    position_correct = 0
    position_count = 0
    recall = []
    outputs = []
    for row_index, case_index in enumerate(indices):
        case = split.cases[case_index]
        output = [
            int(value)
            for value in generated[row_index].tolist()
            if int(value) != int(tokenizer.eos_id)
        ]
        outputs.append(tuple(output))
        expected = list(case.target_ids)
        position_correct += sum(
            output[index] == expected[index]
            for index in range(min(len(output), len(expected)))
        )
        position_count += len(expected)
        expected_set = {value for value in expected if value not in special}
        output_set = {value for value in output if value not in special}
        recall.append(len(expected_set & output_set) / max(1, len(expected_set)))
        rows.append(
            {
                "case_id": case.case_id,
                "source_name": case.source_name,
                "archive_episode": tokenizer.decode(case.source_ids),
                "visible_prefix": tokenizer.decode(case.prefix_ids),
                "expected_continuation": tokenizer.decode(case.target_ids),
                "generated_continuation": tokenizer.decode(output),
            }
        )
    return {
        "case_count": len(indices),
        "interface": interface,
        "policy": policy,
        "expected_token_position_accuracy": position_correct / max(1, position_count),
        "mean_expected_unique_token_recall": sum(recall) / len(recall),
        "examples": rows,
        "_outputs": outputs,
        "_indices": indices,
    }


@torch.no_grad()
def generation_source_swap(
    *,
    model: MarulhoEvidenceReaderLanguageModel,
    tokenizer,
    split: PreparedDocumentSplit,
    groups: torch.Tensor,
    rankings: Mapping[str, torch.Tensor],
    config: EvidenceReaderScreenConfig,
) -> dict[str, Any]:
    true = generate_samples(
        "oracle_reader",
        model=model,
        tokenizer=tokenizer,
        split=split,
        groups=groups,
        rankings=rankings,
        config=config,
    )
    wrong = generate_samples(
        "wrong_reader",
        model=model,
        tokenizer=tokenizer,
        split=split,
        groups=groups,
        rankings=rankings,
        config=config,
    )
    changes = [left != right for left, right in zip(true["_outputs"], wrong["_outputs"])]
    return {
        "case_count": len(changes),
        "output_change_rate": sum(changes) / len(changes),
        "true_source_outputs": [row["generated_continuation"] for row in true["examples"]],
        "wrong_source_outputs": [row["generated_continuation"] for row in wrong["examples"]],
        "target_identity_used_only_to_build_nonpromotable_intervention": True,
    }


def evidence_reader_decision(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    train_steps: int,
    config: EvidenceReaderScreenConfig,
) -> str:
    if set(rows) != set(ARM_NAMES):
        return "incomplete_v27_missing_control_arm"
    if int(train_steps) < MINIMUM_DECISION_STEPS:
        return "diagnostic_v27_below_screen_step_floor"
    oracle = rows["oracle_reader"]["matched_to_zero"]
    if (
        float(oracle["mean_loss_gain"])
        < float(config.minimum_oracle_gain_over_zero)
        or float(oracle["bootstrap_95_ci"][0]) <= 0.0
    ):
        return "retire_v27_interleaved_task_not_learnable_with_oracle_evidence"
    candidate = rows["lexical_reader"]
    primary = candidate["evaluation"]["primary"]
    gain = candidate["matched_to_zero"]
    shuffled_loss = float(rows["shuffled_reader"]["evaluation"]["primary"]["heldout_loss"])
    raw_loss = float(rows["raw_context"]["evaluation"]["primary"]["heldout_loss"])
    candidate_loss = float(primary["heldout_loss"])
    true_wrong = candidate["evaluation"]["source_use"]["true_over_wrong_reader"]
    maximum_general = max(
        float(row["heldout_loss_delta"])
        for row in candidate["general_language"]["sources"]
    )
    if maximum_general > float(config.maximum_general_loss_regression):
        return "retire_v27_interleaved_reader_breaks_general_language"
    zero_sources = rows["gate_zero"]["evaluation"]["primary"]["per_source"]
    candidate_sources = primary["per_source"]
    source_gains = [
        float(zero_sources[name]["heldout_loss"])
        - float(candidate_sources[name]["heldout_loss"])
        for name in zero_sources
    ]
    if (
        float(primary["target_inclusion"])
        >= float(config.minimum_lexical_target_inclusion)
        and float(gain["mean_loss_gain"])
        >= float(config.minimum_reader_gain_over_zero)
        and float(gain["bootstrap_95_ci"][0]) > 0.0
        and shuffled_loss - candidate_loss
        >= float(config.minimum_reader_gain_over_shuffled)
        and raw_loss - candidate_loss >= float(config.minimum_reader_gain_over_raw)
        and min(source_gains) >= float(config.minimum_per_source_gain)
        and float(true_wrong["mean_loss_gain"])
        >= float(config.minimum_true_over_wrong_gain)
        and float(true_wrong["bootstrap_95_ci"][0]) > 0.0
        and float(candidate["generation_source_swap"]["output_change_rate"])
        >= float(config.minimum_generation_source_swap_rate)
    ):
        return ADVANCE_DECISION
    return "retire_v27_interleaved_reader_no_anchored_interface_gain"


def run_evidence_reader_screen(
    *,
    parent_checkpoint_path: str | Path,
    document_train_paths: Sequence[str | Path],
    document_eval_paths: Sequence[str | Path],
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    output_path: str | Path,
    config: EvidenceReaderScreenConfig = EvidenceReaderScreenConfig(),
    device: str = "auto",
) -> dict[str, Any]:
    if int(config.facts_per_query) != 4:
        raise ValueError("V27 requires four archive candidates")
    resolved = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for V27 but unavailable")
    started = time.perf_counter()
    parent_path = Path(parent_checkpoint_path)
    cortex, tokenizer, parent_metadata = load_hashed_micro_expert_checkpoint(
        parent_path, map_location="cpu"
    )
    parent_tokens = _validate_parent(cortex, parent_metadata)
    if parent_tokens < 1_000_000_000:
        raise ValueError("V27 requires the one-billion-token V11 parent")
    torch.manual_seed(int(config.model_seed))
    model = MarulhoEvidenceReaderLanguageModel(
        cortex,
        EvidenceReaderConfig(
            width=int(cortex.hashed_config.width),
            attention_heads=int(config.reader_attention_heads),
            gate_logit_initial=float(config.reader_gate_logit_initial),
            injection_layer_indices=tuple(
                int(value) for value in config.reader_injection_layer_indices
            ),
        ),
    )
    print("[interleaved-reader-v27] preparing document splits", flush=True)
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
    overlap = {
        case.document_sha256 for case in train_split.cases
    } & {case.document_sha256 for case in eval_split.cases}
    if overlap:
        raise RuntimeError("V27 train/eval document hashes overlap")
    document_steps = _scheduled_relation_steps(
        int(config.train_steps), float(config.document_fraction)
    )
    document_schedule = build_training_schedule(
        train_split,
        tokenizer,
        steps=document_steps,
        batch_size=int(config.batch_size),
        facts_per_query=int(config.facts_per_query),
        seed=int(config.train_data_seed) + 10,
    )
    eval_groups, eval_slots, eval_rankings = build_evaluation_schedule(
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
    general_baseline = evaluate_general_language(
        model.cortex, prepared_general.eval_batches
    )
    rows: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        print(f"[interleaved-reader-v27] training {arm}", flush=True)
        training, general_after = train_arm(
            arm,
            model=model,
            initial_state=initial_state,
            split=train_split,
            document_schedule=document_schedule,
            schedule=schedule,
            prepared_general=prepared_general,
            general_baseline=general_baseline,
            config=config,
        )
        evaluation = evaluate_suite(
            arm,
            model=model,
            split=eval_split,
            groups=eval_groups,
            target_slots=eval_slots,
            rankings=eval_rankings,
            config=config,
        )
        generation = generate_samples(
            arm,
            model=model,
            tokenizer=tokenizer,
            split=eval_split,
            groups=eval_groups,
            rankings=eval_rankings,
            config=config,
        )
        swap = generation_source_swap(
            model=model,
            tokenizer=tokenizer,
            split=eval_split,
            groups=eval_groups,
            rankings=eval_rankings,
            config=config,
        )
        rows[arm] = {
            "arm": arm,
            "promotable": arm != "oracle_reader",
            "training": training,
            "general_language": general_after,
            "evaluation": evaluation,
            "generation": generation,
            "generation_source_swap": swap,
        }
        print(
            f"[interleaved-reader-v27] {arm} loss="
            f"{evaluation['primary']['heldout_loss']:.4f} general_delta="
            f"{general_after['aggregate_heldout_loss_delta']:+.4f}",
            flush=True,
        )
    zero_losses = rows["gate_zero"]["evaluation"]["_primary_case_losses"]
    for index, arm in enumerate(ARM_NAMES):
        losses = rows[arm]["evaluation"]["_primary_case_losses"]
        rows[arm]["matched_to_zero"] = paired_bootstrap_gain(
            zero_losses,
            losses,
            samples=int(config.bootstrap_samples),
            seed=int(config.evaluation_seed) + 500 + index,
        )
    decision = evidence_reader_decision(
        rows, train_steps=int(config.train_steps), config=config
    )
    for arm in ARM_NAMES:
        rows[arm]["evaluation"].pop("_primary_case_losses")
        rows[arm]["generation"].pop("_outputs")
        rows[arm]["generation"].pop("_indices")
    train_lexical = document_schedule.rankings["lexical"]
    train_inclusion = float(
        (train_lexical[..., :1] == document_schedule.target_slots.unsqueeze(-1))
        .any(dim=-1)
        .float()
        .mean()
    )
    eval_lexical = eval_rankings["lexical"]
    eval_inclusion = float(
        (eval_lexical[:, :1] == eval_slots.unsqueeze(1)).any(dim=1).float().mean()
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
        },
        "architecture": {
            **model.reader_parameter_report(),
            "archive_content": "one_exact_prior_document_span",
            "archive_key": "checkpoint_bpe_tfidf",
            "local_query_positions_unchanged_by_reader": True,
            "separate_source_cortex_pass": True,
            "shared_cross_attention_interleaved_between_cortex_blocks": True,
            "raw_context_control": True,
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
            "train_lexical_target_inclusion": train_inclusion,
            "eval_lexical_target_inclusion": eval_inclusion,
            "document_steps": document_steps,
            "general_steps": int(config.train_steps) - document_steps,
        },
        "anti_cheat": {
            "archive_episode_precedes_visible_prefix": True,
            "retrieval_uses_visible_prefix_only": True,
            "retrieval_uses_future_target": False,
            "target_slot_metrics_only": True,
            "oracle_promotable": False,
            "teacher_forcing_target_visible_only_after_retrieval": True,
            "train_eval_document_hash_overlap": len(overlap),
            "all_arms_own_identical_reader_parameters": True,
            "initial_model_state_identical": True,
        },
        "general_language_before": general_baseline,
        "arms": rows,
        "decision": decision,
        "checkpoint": None,
        "promotion_boundary": {
            "advance_to_manual_review": decision == ADVANCE_DECISION,
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
        title="MARULHO V27 Interleaved Evidence Reader Screen",
    )
    print(f"[interleaved-reader-v27] decision {decision}", flush=True)
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
    config = EvidenceReaderScreenConfig(
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
    run_evidence_reader_screen(
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
