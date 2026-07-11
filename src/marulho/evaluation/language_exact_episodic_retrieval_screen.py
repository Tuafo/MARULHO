"""Test bounded exact-episode retrieval as language context for the V11 cortex."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from marulho.evaluation.language_exact_episodic_retrieval_audit import (
    EncodedTextBank,
    build_evaluation_groups,
    encode_text_bank,
    lexical_tfidf_scores,
    load_retrieval_cases,
    rankings_from_scores,
    split_relation_case_prompt,
)
from marulho.evaluation.language_hashed_micro_expert_continuation import (
    _validate_parent,
)
from marulho.evaluation.language_matched_support import (
    build_matched_schedule,
    full_sized_batches,
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
from marulho.training.language_model import (
    LanguageBatch,
    build_language_model_splits,
    evaluate_language_model,
)


SURFACE = "marulho_exact_episodic_retrieval_screen.v1"
ARTIFACT_KIND = "marulho_exact_episodic_retrieval_screen"
ARM_NAMES = ("off", "all4", "random2", "recency2", "lexical1", "lexical2")
MINIMUM_DECISION_STEPS = 512
ADVANCE_DECISION = "advance_v21_exact_episodic_retrieval_to_contiguous_streams"


@dataclass(frozen=True)
class RelationRecord:
    source: str
    query_prefix: str
    answer: str


@dataclass(frozen=True)
class RelationEvalCase:
    case_id: str
    kind: str
    source: str
    query_prefix: str
    candidates: tuple[str, ...]
    correct_index: int


@dataclass(frozen=True)
class EncodedRelationBank:
    source_ids: torch.Tensor
    source_mask: torch.Tensor
    read_query_ids: torch.Tensor
    read_query_mask: torch.Tensor
    query_input_ids: torch.Tensor
    query_target_ids: torch.Tensor
    query_loss_mask: torch.Tensor

    @property
    def record_count(self) -> int:
        return int(self.source_ids.shape[0])


@dataclass(frozen=True)
class RetrievalScreenConfig:
    train_record_count: int = 8192
    facts_per_query: int = 4
    source_length: int = 48
    query_length: int = 40
    train_steps: int = 800
    batch_size: int = 16
    eval_batch_size: int = 32
    general_sequence_length: int = 128
    general_eval_batches: int = 8
    relation_fraction: float = 0.75
    general_train_sample_bytes: int = 8 * 1024 * 1024
    general_eval_sample_bytes: int = 8 * 1024 * 1024
    sample_range_count: int = 8
    learning_rate: float = 5.0e-5
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.02
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    train_data_seed: int = 9301
    evaluation_seed: int = 9501
    model_seed: int = 9401
    generation_max_tokens: int = 16
    minimum_all4_candidate_accuracy: float = 0.75
    minimum_all4_free_accuracy: float = 0.30
    minimum_all4_counterfactual_gain_over_off: float = 0.10
    minimum_lexical2_candidate_accuracy: float = 0.75
    minimum_lexical2_free_accuracy: float = 0.35
    minimum_lexical2_counterfactual_accuracy: float = 0.35
    minimum_lexical2_control_gain: float = 0.05
    maximum_lexical2_regret_to_all4: float = 0.05
    maximum_general_loss_regression: float = 0.10


@dataclass(frozen=True)
class PreparedGeneralLanguage:
    train_batches: tuple[tuple[LanguageBatch, ...], ...]
    eval_batches: tuple[tuple[LanguageBatch, ...], ...]
    source_reports: dict[str, Any]


def parse_relation_training_line(line: str) -> RelationRecord | None:
    text = str(line).strip()
    if not text or text.startswith("#") or text == "<|MARULHO_DOCUMENT|>":
        return None
    if "Question:" not in text or "Answer:" not in text:
        return None
    source, remainder = text.split("Question:", 1)
    question, answer = remainder.split("Answer:", 1)
    source = source.strip()
    question = question.strip()
    answer = answer.strip()
    if not source or not question or not answer:
        raise ValueError("relation record has an empty source, question, or answer")
    return RelationRecord(
        source=source,
        query_prefix=f"Question: {question} Answer: ",
        answer=answer,
    )


def load_relation_eval_cases(path: str | Path) -> tuple[RelationEvalCase, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("relation case payload contains no cases")
    cases = []
    for row in rows:
        source, query_prefix = split_relation_case_prompt(str(row["prompt"]))
        candidates = tuple(str(value).strip() for value in row["candidates"])
        correct_index = int(row["correct_index"])
        if not 0 <= correct_index < len(candidates):
            raise ValueError("relation case correct_index is out of range")
        cases.append(
            RelationEvalCase(
                case_id=str(row["case_id"]),
                kind=str(row["kind"]),
                source=source,
                query_prefix=query_prefix,
                candidates=candidates,
                correct_index=correct_index,
            )
        )
    return tuple(cases)


def sample_relation_training_records(
    path: str | Path,
    *,
    count: int,
    seed: int,
) -> tuple[RelationRecord, ...]:
    if int(count) < 1:
        raise ValueError("training record count must be positive")
    generator = random.Random(int(seed))
    reservoir: list[RelationRecord] = []
    eligible = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = parse_relation_training_line(line)
            if record is None:
                continue
            eligible += 1
            if len(reservoir) < int(count):
                reservoir.append(record)
                continue
            replacement = generator.randrange(eligible)
            if replacement < int(count):
                reservoir[replacement] = record
    if len(reservoir) < int(count):
        raise ValueError(
            f"relation corpus has only {len(reservoir)} records; requested {count}"
        )
    generator.shuffle(reservoir)
    return tuple(reservoir)


def encode_relation_records(
    tokenizer,
    records: Sequence[RelationRecord],
    *,
    source_length: int,
    query_length: int,
) -> EncodedRelationBank:
    count = len(records)
    source_ids = torch.full(
        (count, int(source_length)), int(tokenizer.pad_id), dtype=torch.long
    )
    source_mask = torch.zeros((count, int(source_length)), dtype=torch.bool)
    read_query_ids = torch.full(
        (count, int(query_length)), int(tokenizer.pad_id), dtype=torch.long
    )
    read_query_mask = torch.zeros((count, int(query_length)), dtype=torch.bool)
    query_input_ids = torch.full(
        (count, int(query_length)), int(tokenizer.pad_id), dtype=torch.long
    )
    query_target_ids = torch.full(
        (count, int(query_length)), -100, dtype=torch.long
    )
    query_loss_mask = torch.zeros((count, int(query_length)), dtype=torch.bool)
    for index, record in enumerate(records):
        source = tokenizer.encode(record.source, add_bos=True, add_eos=True)
        prefix = tokenizer.encode(
            record.query_prefix, add_bos=True, add_eos=False
        )
        answer = tokenizer.encode(record.answer, add_bos=False, add_eos=True)
        sequence = [*prefix, *answer]
        query_input = sequence[:-1]
        targets = sequence[1:]
        answer_start = len(prefix) - 1
        loss_mask = [position >= answer_start for position in range(len(query_input))]
        if len(source) > int(source_length):
            raise ValueError(
                f"source token length {len(source)} exceeds {source_length}"
            )
        if len(query_input) > int(query_length):
            raise ValueError(
                f"query token length {len(query_input)} exceeds {query_length}"
            )
        source_ids[index, : len(source)] = torch.tensor(source)
        source_mask[index, : len(source)] = True
        read_query_ids[index, : len(prefix)] = torch.tensor(prefix)
        read_query_mask[index, : len(prefix)] = True
        query_input_ids[index, : len(query_input)] = torch.tensor(query_input)
        query_target_ids[index, : len(targets)] = torch.tensor(targets)
        query_loss_mask[index, : len(loss_mask)] = torch.tensor(loss_mask)
    return EncodedRelationBank(
        source_ids=source_ids,
        source_mask=source_mask,
        read_query_ids=read_query_ids,
        read_query_mask=read_query_mask,
        query_input_ids=query_input_ids,
        query_target_ids=query_target_ids,
        query_loss_mask=query_loss_mask,
    )


def build_group_schedule(
    *,
    record_count: int,
    steps: int,
    batch_size: int,
    facts_per_query: int,
    seed: int,
    record_labels: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(record_labels) != int(record_count):
        raise ValueError("record_labels length must equal record_count")
    buckets: dict[str, list[int]] = {}
    for index, label in enumerate(record_labels):
        buckets.setdefault(str(label), []).append(index)
    labels = sorted(buckets)
    if len(labels) < int(facts_per_query):
        raise ValueError("not enough distinct query identities for one group")
    generator = random.Random(int(seed))
    groups = []
    targets = []
    for _step in range(int(steps)):
        step_groups = []
        step_targets = []
        for _row in range(int(batch_size)):
            chosen = generator.sample(labels, int(facts_per_query))
            step_groups.append([generator.choice(buckets[label]) for label in chosen])
            step_targets.append(generator.randrange(int(facts_per_query)))
        groups.append(step_groups)
        targets.append(step_targets)
    return torch.tensor(groups, dtype=torch.long), torch.tensor(
        targets, dtype=torch.long
    )


def _tensor_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().contiguous().cpu()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _precision_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    if precision == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "float16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "float32":
        return nullcontext()
    raise ValueError("precision must be float32, float16, or bfloat16")


def _special_token_ids(tokenizer) -> tuple[int, ...]:
    return (
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.eos_id,
        tokenizer.unk_id,
        tokenizer.checkpoint_id,
        tokenizer.replay_id,
    )


def lexical_episode_rankings(
    bank: EncodedRelationBank,
    group_indices: torch.Tensor,
    query_record_indices: torch.Tensor,
    *,
    tokenizer,
) -> torch.Tensor:
    flat_queries = query_record_indices.reshape(-1)
    query_bank = EncodedTextBank(
        ids=bank.read_query_ids.index_select(0, flat_queries),
        mask=bank.read_query_mask.index_select(0, flat_queries),
    )
    source_bank = EncodedTextBank(ids=bank.source_ids, mask=bank.source_mask)
    flat_groups = group_indices.reshape(-1, int(group_indices.shape[-1]))
    scores = lexical_tfidf_scores(
        source_bank,
        query_bank,
        flat_groups,
        excluded_token_ids=_special_token_ids(tokenizer),
    )
    return rankings_from_scores(scores).reshape(group_indices.shape)


def build_policy_rankings(
    lexical_rankings: torch.Tensor,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    facts = int(lexical_rankings.shape[-1])
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random_scores = torch.rand(lexical_rankings.shape, generator=generator)
    random_rankings = rankings_from_scores(
        random_scores.reshape(-1, facts)
    ).reshape(lexical_rankings.shape)
    recency = torch.arange(facts - 1, -1, -1, dtype=torch.long)
    recency_rankings = recency.reshape(
        *((1,) * (lexical_rankings.ndim - 1)), facts
    ).expand_as(lexical_rankings)
    return {
        "random2": random_rankings,
        "recency2": recency_rankings,
        "lexical1": lexical_rankings,
        "lexical2": lexical_rankings,
    }


def selected_slots_for_mode(
    mode: str,
    rankings: Mapping[str, torch.Tensor],
) -> torch.Tensor | None:
    if mode == "off":
        return None
    if mode == "all4":
        sample = next(iter(rankings.values()))
        facts = int(sample.shape[-1])
        return torch.arange(facts, dtype=torch.long).reshape(
            *((1,) * (sample.ndim - 1)), facts
        ).expand_as(sample)
    if mode == "lexical1":
        return rankings[mode][..., :1]
    if mode in {"random2", "recency2", "lexical2"}:
        return rankings[mode][..., :2]
    raise ValueError(f"unknown V21 mode: {mode}")


def gather_retrieved_sources(
    bank: EncodedRelationBank,
    group_indices: torch.Tensor,
    selected_slots: torch.Tensor | None,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = int(group_indices.shape[0])
    if selected_slots is None:
        return (
            torch.empty(batch, 0, device=device, dtype=torch.long),
            torch.empty(batch, 0, device=device, dtype=torch.bool),
        )
    selected_records = group_indices.gather(1, selected_slots)
    flat = selected_records.reshape(-1)
    ids = bank.source_ids.index_select(0, flat).reshape(
        batch, -1
    )
    mask = bank.source_mask.index_select(0, flat).reshape(
        batch, -1
    )
    return ids.to(device), mask.to(device)


def target_query_batch(
    bank: EncodedRelationBank,
    group_indices: torch.Tensor,
    target_slots: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = group_indices.gather(1, target_slots.unsqueeze(1)).squeeze(1)
    return (
        bank.query_input_ids.index_select(0, selected).to(device),
        bank.query_target_ids.index_select(0, selected).to(device),
        bank.query_loss_mask.index_select(0, selected).to(device),
    )


def retrieved_relation_logits(
    model: MarulhoHashedMicroExpertLanguageModel,
    retrieved_source_ids: torch.Tensor,
    query_ids: torch.Tensor,
) -> torch.Tensor:
    combined = torch.cat((retrieved_source_ids, query_ids), dim=1)
    if int(combined.shape[1]) > int(model.hashed_config.context_length):
        raise ValueError("retrieved sources and query exceed the cortex context")
    hidden = model._forward_hidden(combined, collect_telemetry=False)["hidden"]
    return model.lm_head(hidden[:, -int(query_ids.shape[1]) :])


def relation_loss(
    model: MarulhoHashedMicroExpertLanguageModel,
    retrieved_source_ids: torch.Tensor,
    query_ids: torch.Tensor,
    target_ids: torch.Tensor,
    answer_mask: torch.Tensor,
) -> torch.Tensor:
    logits = retrieved_relation_logits(model, retrieved_source_ids, query_ids)
    return F.cross_entropy(logits[answer_mask], target_ids[answer_mask])


def prepare_general_language(
    tokenizer,
    *,
    train_paths: Sequence[str | Path],
    eval_paths: Sequence[str | Path],
    config: RetrievalScreenConfig,
) -> PreparedGeneralLanguage:
    if len(train_paths) != 2 or len(eval_paths) != 2:
        raise ValueError("V21 requires exactly two general train and eval sources")
    relation_steps = _scheduled_relation_steps(
        int(config.train_steps), float(config.relation_fraction)
    )
    general_steps = int(config.train_steps) - relation_steps
    per_source = max(2, math.ceil(general_steps / 2) + 1)
    train_samples = [
        sample_corpus_ranges(
            path,
            byte_budget=int(config.general_train_sample_bytes),
            range_count=int(config.sample_range_count),
        )
        for path in train_paths
    ]
    eval_samples = [
        sample_corpus_ranges(
            path,
            byte_budget=int(config.general_eval_sample_bytes),
            range_count=int(config.sample_range_count),
        )
        for path in eval_paths
    ]
    train_batches = []
    for text, _report in train_samples:
        split = build_language_model_splits(
            [text],
            tokenizer,
            sequence_length=int(config.general_sequence_length),
            stride=int(config.general_sequence_length),
            batch_size=int(config.batch_size),
            max_train_batches=per_source,
            max_eval_batches=1,
        )
        train_batches.append(
            full_sized_batches(split.train, batch_size=int(config.batch_size))
        )
    eval_batches = []
    for text, _report in eval_samples:
        split = build_language_model_splits(
            [],
            tokenizer,
            eval_texts=[text],
            sequence_length=int(config.general_sequence_length),
            stride=int(config.general_sequence_length),
            batch_size=int(config.batch_size),
            max_train_batches=1,
            max_eval_batches=int(config.general_eval_batches),
        )
        eval_batches.append(tuple(split.eval))
    return PreparedGeneralLanguage(
        train_batches=tuple(train_batches),
        eval_batches=tuple(eval_batches),
        source_reports={
            "train": [report for _text, report in train_samples],
            "eval": [report for _text, report in eval_samples],
        },
    )


def _scheduled_relation_steps(step_count: int, relation_fraction: float) -> int:
    accumulator = 0.0
    count = 0
    for _ in range(int(step_count)):
        accumulator += float(relation_fraction)
        if accumulator >= 1.0:
            accumulator -= 1.0
            count += 1
    return count


def evaluate_general_language(
    model: MarulhoHashedMicroExpertLanguageModel,
    eval_batches: Sequence[Sequence[LanguageBatch]],
    *,
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sources = []
    weighted_loss = 0.0
    token_count = 0
    baseline_sources = [] if baseline is None else list(baseline["sources"])
    for index, batches in enumerate(eval_batches):
        row = evaluate_language_model(model, batches)
        loss = float(row["heldout_loss"])
        count = int(row["token_count"])
        sources.append(
            {
                "source_index": index,
                **row,
                "heldout_loss_before": (
                    None
                    if baseline is None
                    else float(baseline_sources[index]["heldout_loss"])
                ),
                "heldout_loss_delta": (
                    0.0
                    if baseline is None
                    else loss - float(baseline_sources[index]["heldout_loss"])
                ),
            }
        )
        weighted_loss += loss * count
        token_count += count
    aggregate = weighted_loss / max(1, token_count)
    baseline_aggregate = (
        None if baseline is None else float(baseline["aggregate_heldout_loss"])
    )
    return {
        "sources": sources,
        "aggregate_heldout_loss": aggregate,
        "aggregate_heldout_loss_before": baseline_aggregate,
        "aggregate_heldout_loss_delta": (
            0.0 if baseline_aggregate is None else aggregate - baseline_aggregate
        ),
        "token_count": token_count,
    }


def _candidate_record_bank(
    tokenizer,
    cases: Sequence[RelationEvalCase],
    *,
    config: RetrievalScreenConfig,
) -> tuple[EncodedRelationBank, int]:
    candidate_count = len(cases[0].candidates)
    if any(len(case.candidates) != candidate_count for case in cases):
        raise ValueError("relation cases have inconsistent candidate counts")
    records = [
        RelationRecord(
            source=case.source,
            query_prefix=case.query_prefix,
            answer=candidate,
        )
        for case in cases
        for candidate in case.candidates
    ]
    return (
        encode_relation_records(
            tokenizer,
            records,
            source_length=int(config.source_length),
            query_length=int(config.query_length),
        ),
        candidate_count,
    )


@torch.no_grad()
def evaluate_candidate_ranking(
    mode: str,
    *,
    model: MarulhoHashedMicroExpertLanguageModel,
    case_bank: EncodedRelationBank,
    candidate_bank: EncodedRelationBank,
    candidate_count: int,
    cases: Sequence[RelationEvalCase],
    group_indices: torch.Tensor,
    policy_rankings: Mapping[str, torch.Tensor],
    config: RetrievalScreenConfig,
) -> dict[str, Any]:
    model.eval()
    predictions = []
    score_rows = []
    for start in range(0, len(cases), int(config.eval_batch_size)):
        end = min(len(cases), start + int(config.eval_batch_size))
        groups = group_indices[start:end]
        rankings = {
            name: values[start:end] for name, values in policy_rankings.items()
        }
        selected = selected_slots_for_mode(mode, rankings)
        retrieved, _mask = gather_retrieved_sources(
            case_bank, groups, selected, device=model.device
        )
        candidate_scores = []
        for candidate in range(int(candidate_count)):
            flat = (
                torch.arange(start, end, dtype=torch.long) * int(candidate_count)
                + candidate
            )
            query_ids = candidate_bank.query_input_ids.index_select(0, flat).to(
                model.device
            )
            targets = candidate_bank.query_target_ids.index_select(0, flat).to(
                model.device
            )
            answer_mask = candidate_bank.query_loss_mask.index_select(0, flat).to(
                model.device
            )
            with _precision_context(model.device, str(config.precision)):
                logits = retrieved_relation_logits(model, retrieved, query_ids)
                losses = F.cross_entropy(
                    logits.reshape(-1, int(logits.shape[-1])),
                    targets.reshape(-1),
                    reduction="none",
                    ignore_index=-100,
                ).reshape(targets.shape)
                score = (losses * answer_mask).sum(dim=1) / answer_mask.sum(
                    dim=1
                ).clamp_min(1)
            candidate_scores.append(score.float())
        scores = torch.stack(candidate_scores, dim=1)
        predictions.extend(scores.argmin(dim=1).cpu().tolist())
        score_rows.append(scores.cpu())
    correct = [int(case.correct_index) for case in cases]
    kind_rows: dict[str, list[bool]] = {}
    for case, prediction in zip(cases, predictions):
        kind_rows.setdefault(case.kind, []).append(
            prediction == int(case.correct_index)
        )
    return {
        "candidate_accuracy": sum(
            int(prediction == expected)
            for prediction, expected in zip(predictions, correct)
        )
        / len(cases),
        "candidate_kind_accuracy": {
            kind: sum(values) / len(values) for kind, values in kind_rows.items()
        },
        "mean_candidate_scores": torch.cat(score_rows).mean(dim=0).tolist(),
        "prediction_uses_correct_index": False,
        "correct_index_metrics_only": True,
    }


def _normalize_answer(value: str) -> str:
    lowered = re.sub(r"\s+", " ", str(value).strip().lower())
    return lowered.rstrip(" .!?;:")


@torch.no_grad()
def evaluate_free_generation(
    mode: str,
    *,
    model: MarulhoHashedMicroExpertLanguageModel,
    tokenizer,
    case_bank: EncodedRelationBank,
    cases: Sequence[RelationEvalCase],
    group_indices: torch.Tensor,
    policy_rankings: Mapping[str, torch.Tensor],
    config: RetrievalScreenConfig,
) -> dict[str, Any]:
    model.eval()
    exact_rows = []
    contains_rows = []
    examples = []
    counterfactual_rows = []
    for start in range(0, len(cases), int(config.eval_batch_size)):
        end = min(len(cases), start + int(config.eval_batch_size))
        selected_cases = cases[start:end]
        groups = group_indices[start:end]
        rankings = {
            name: values[start:end] for name, values in policy_rankings.items()
        }
        selected = selected_slots_for_mode(mode, rankings)
        retrieved, _mask = gather_retrieved_sources(
            case_bank, groups, selected, device=model.device
        )
        sequences = [
            tokenizer.encode(case.query_prefix, add_bos=True, add_eos=False)
            for case in selected_cases
        ]
        generated = [[] for _case in selected_cases]
        finished = [False for _case in selected_cases]
        for _step in range(int(config.generation_max_tokens)):
            maximum = max(len(sequence) for sequence in sequences)
            if maximum > int(config.query_length):
                raise RuntimeError("V21 free generation exceeded the query budget")
            query_ids = torch.full(
                (len(sequences), maximum),
                int(tokenizer.pad_id),
                device=model.device,
                dtype=torch.long,
            )
            lengths = []
            for row, sequence in enumerate(sequences):
                query_ids[row, : len(sequence)] = torch.tensor(
                    sequence, device=model.device, dtype=torch.long
                )
                lengths.append(len(sequence))
            with _precision_context(model.device, str(config.precision)):
                logits = retrieved_relation_logits(model, retrieved, query_ids)
                rows = torch.arange(len(sequences), device=model.device)
                positions = torch.tensor(lengths, device=model.device) - 1
                next_logits = logits[rows, positions]
            next_ids = next_logits.argmax(dim=-1).cpu().tolist()
            for row, next_id in enumerate(next_ids):
                if finished[row]:
                    continue
                sequences[row].append(int(next_id))
                if int(next_id) == int(tokenizer.eos_id):
                    finished[row] = True
                else:
                    generated[row].append(int(next_id))
            if all(finished):
                break
        for case, token_ids in zip(selected_cases, generated):
            observed = tokenizer.decode(token_ids).strip()
            expected = case.candidates[int(case.correct_index)]
            observed_normalized = _normalize_answer(observed)
            expected_normalized = _normalize_answer(expected)
            exact = observed_normalized == expected_normalized
            contains = expected_normalized in observed_normalized
            exact_rows.append(exact)
            contains_rows.append(contains)
            counterfactual_rows.append(
                {
                    "case_id": case.case_id,
                    "query_prefix": case.query_prefix,
                    "observed": observed_normalized,
                    "expected": expected_normalized,
                    "exact": exact,
                }
            )
            if len(examples) < 12:
                examples.append(
                    {
                        "case_id": case.case_id,
                        "kind": case.kind,
                        "observed": observed,
                        "expected": expected,
                        "exact": exact,
                    }
                )
    return {
        "free_exact_accuracy": sum(exact_rows) / len(exact_rows),
        "free_contains_accuracy": sum(contains_rows) / len(contains_rows),
        "generation_policy": "greedy_argmax",
        "maximum_new_tokens": int(config.generation_max_tokens),
        "examples": examples,
        "_counterfactual_rows": counterfactual_rows,
    }


def counterfactual_behavior_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["query_prefix"]), []).append(row)
    usable = [
        group
        for group in grouped.values()
        if len({str(row["expected"]) for row in group}) > 1
    ]
    eligible_rows = [row for group in usable for row in group]
    pairs = []
    for group in usable:
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                if str(left["expected"]) != str(right["expected"]):
                    pairs.append((left, right))
    return {
        "query_group_count": len(usable),
        "case_count": len(eligible_rows),
        "different_answer_pair_count": len(pairs),
        "source_following_exact_accuracy": (
            sum(bool(row["exact"]) for row in eligible_rows)
            / max(1, len(eligible_rows))
        ),
        "macro_query_exact_accuracy": (
            sum(
                sum(bool(row["exact"]) for row in group) / len(group)
                for group in usable
            )
            / max(1, len(usable))
        ),
        "output_change_rate_when_source_answer_changes": (
            sum(str(left["observed"]) != str(right["observed"]) for left, right in pairs)
            / max(1, len(pairs))
        ),
        "both_answers_correct_pair_rate": (
            sum(bool(left["exact"]) and bool(right["exact"]) for left, right in pairs)
            / max(1, len(pairs))
        ),
        "identical_query_and_distractors_with_target_source_swap": True,
        "promotion_metric": True,
    }


def retrieval_context_metrics(
    mode: str,
    *,
    group_indices: torch.Tensor,
    target_slots: torch.Tensor,
    policy_rankings: Mapping[str, torch.Tensor],
    cases: Sequence[RelationEvalCase],
    source_length: int,
) -> dict[str, Any]:
    selected = selected_slots_for_mode(mode, policy_rankings)
    if selected is None:
        inclusion = torch.zeros(len(cases), dtype=torch.bool)
        selected_count = 0
    else:
        inclusion = (selected == target_slots.unsqueeze(1)).any(dim=1)
        selected_count = int(selected.shape[1])
    grouped: dict[str, list[int]] = {}
    for index, case in enumerate(cases):
        grouped.setdefault(case.query_prefix, []).append(index)
    usable = [indices for indices in grouped.values() if len(indices) > 1]
    pairs = [
        (left, right)
        for indices in usable
        for left_position, left in enumerate(indices)
        for right in indices[left_position + 1 :]
    ]
    return {
        "selected_source_count": selected_count,
        "active_source_tokens": selected_count * int(source_length),
        "target_inclusion": float(inclusion.float().mean()),
        "paired_target_inclusion": (
            sum(bool(inclusion[index]) for indices in usable for index in indices)
            / max(1, sum(len(indices) for indices in usable))
        ),
        "both_targets_included_pair_rate": (
            sum(bool(inclusion[left]) and bool(inclusion[right]) for left, right in pairs)
            / max(1, len(pairs))
        ),
        "target_slot_metrics_only": True,
    }


def retrieval_screen_decision(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    train_steps: int,
    config: RetrievalScreenConfig,
) -> str:
    if set(rows) != set(ARM_NAMES):
        return "incomplete_v21_missing_control_arm"
    if int(train_steps) < MINIMUM_DECISION_STEPS:
        return "diagnostic_v21_below_screen_step_floor"
    candidate = {
        name: float(rows[name]["evaluation"]["candidate_accuracy"])
        for name in ARM_NAMES
    }
    free = {
        name: float(rows[name]["evaluation"]["free_exact_accuracy"])
        for name in ARM_NAMES
    }
    paired = {
        name: float(
            rows[name]["evaluation"]["paired_counterfactual"][
                "source_following_exact_accuracy"
            ]
        )
        for name in ARM_NAMES
    }
    if (
        candidate["all4"] < float(config.minimum_all4_candidate_accuracy)
        or free["all4"] < float(config.minimum_all4_free_accuracy)
        or paired["all4"] - paired["off"]
        < float(config.minimum_all4_counterfactual_gain_over_off)
    ):
        return "retire_v21_task_not_learnable_from_all_history"
    maximum_general_regression = max(
        float(row["heldout_loss_delta"])
        for row in rows["lexical2"]["general_language"]["sources"]
    )
    if maximum_general_regression > float(config.maximum_general_loss_regression):
        return "retire_v21_lexical_retrieval_breaks_general_language"
    controls = ("random2", "recency2", "lexical1")
    candidate_control = max(candidate[name] for name in controls)
    free_control = max(free[name] for name in controls)
    paired_control = max(paired[name] for name in controls)
    lexical_pass = (
        candidate["lexical2"] >= float(config.minimum_lexical2_candidate_accuracy)
        and free["lexical2"] >= float(config.minimum_lexical2_free_accuracy)
        and paired["lexical2"]
        >= float(config.minimum_lexical2_counterfactual_accuracy)
        and candidate["lexical2"] - candidate_control
        >= float(config.minimum_lexical2_control_gain)
        and free["lexical2"] - free_control
        >= float(config.minimum_lexical2_control_gain)
        and paired["lexical2"] - paired_control
        >= float(config.minimum_lexical2_control_gain)
        and candidate["all4"] - candidate["lexical2"]
        <= float(config.maximum_lexical2_regret_to_all4)
        and free["all4"] - free["lexical2"]
        <= float(config.maximum_lexical2_regret_to_all4)
        and paired["all4"] - paired["lexical2"]
        <= float(config.maximum_lexical2_regret_to_all4)
    )
    if lexical_pass:
        return ADVANCE_DECISION
    if max(paired[name] for name in controls) >= paired["lexical2"]:
        return "retire_v21_lexical_two_no_behavioral_retrieval_gain"
    return "retire_v21_exact_retrieval_does_not_close_generation_gap"


def _run_training_arm(
    mode: str,
    *,
    model: MarulhoHashedMicroExpertLanguageModel,
    initial_state: Mapping[str, torch.Tensor],
    relation_bank: EncodedRelationBank,
    relation_groups: torch.Tensor,
    target_slots: torch.Tensor,
    policy_rankings: Mapping[str, torch.Tensor],
    schedule: Sequence[tuple[str, int]],
    prepared_general: PreparedGeneralLanguage,
    general_baseline: Mapping[str, Any],
    config: RetrievalScreenConfig,
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
    warmup_steps = int(
        round(int(config.train_steps) * max(0.0, float(config.warmup_fraction)))
    )
    trace_steps = {
        max(0, min(int(config.train_steps) - 1, math.ceil(config.train_steps * x / 10) - 1))
        for x in range(1, 11)
    }
    trace = []
    relation_step_count = 0
    general_step_count = 0
    supervised_answer_tokens = 0
    retrieved_source_tokens = 0
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
                relation_index = int(source_index)
                groups = relation_groups[relation_index]
                slots = target_slots[relation_index]
                rankings = {
                    name: values[relation_index]
                    for name, values in policy_rankings.items()
                }
                selected = selected_slots_for_mode(mode, rankings)
                retrieved, retrieved_mask = gather_retrieved_sources(
                    relation_bank, groups, selected, device=model.device
                )
                query_ids, targets, answer_mask = target_query_batch(
                    relation_bank, groups, slots, device=model.device
                )
                loss = relation_loss(
                    model, retrieved, query_ids, targets, answer_mask
                )
                relation_step_count += 1
                supervised_answer_tokens += int(answer_mask.sum())
                retrieved_source_tokens += int(retrieved_mask.sum())
                cortex_positions += int(retrieved.numel() + query_ids.numel())
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
                general_step_count += 1
                count = int(batch.target_ids.numel())
                general_tokens += count
                cortex_positions += count
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite V21 loss in {mode}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config.gradient_clip)
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"non-finite V21 gradient in {mode}")
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
                f"[exact-episodic-v21] {mode} {step + 1}/{config.train_steps}",
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
    groups = relation_groups[0]
    slots = target_slots[0]
    rankings = {name: values[0] for name, values in policy_rankings.items()}
    selected = selected_slots_for_mode(mode, rankings)
    retrieved, _retrieved_mask = gather_retrieved_sources(
        relation_bank, groups, selected, device=model.device
    )
    query_ids, targets, answer_mask = target_query_batch(
        relation_bank, groups, slots, device=model.device
    )
    with _precision_context(model.device, str(config.precision)):
        probe_loss = relation_loss(
            model, retrieved, query_ids, targets, answer_mask
        )
    probe_loss.backward()
    model_parameter_tensors = list(model.parameters())
    gradient = {
        "probe_updates_parameters": False,
        "model_parameter_tensor_count": len(model_parameter_tensors),
        "model_parameter_tensors_with_gradient": sum(
            parameter.grad is not None for parameter in model_parameter_tensors
        ),
        "model_parameter_tensors_with_nonzero_gradient": sum(
            parameter.grad is not None
            and int(torch.count_nonzero(parameter.grad).detach().cpu()) > 0
            for parameter in model_parameter_tensors
        ),
        "hashed_expert_rows": model.final_gradient_report(),
    }
    model.zero_grad(set_to_none=True)
    general_after = evaluate_general_language(
        model, prepared_general.eval_batches, baseline=general_baseline
    )
    return (
        {
            "optimizer": "AdamW",
            "optimizer_state_fresh": True,
            "initial_model_state_restored": True,
            "steps": int(config.train_steps),
            "relation_steps": relation_step_count,
            "general_replay_steps": general_step_count,
            "supervised_answer_tokens": supervised_answer_tokens,
            "retrieved_source_tokens": retrieved_source_tokens,
            "general_training_tokens": general_tokens,
            "cortex_input_positions": cortex_positions,
            "elapsed_seconds": elapsed,
            "cortex_positions_per_second": cortex_positions / max(elapsed, 1.0e-12),
            "peak_cuda_memory_bytes": peak_memory,
            "loss_trace": trace,
            "gradient": gradient,
            "execution_backend": "eager",
        },
        general_after,
    )


def run_exact_episodic_retrieval_screen(
    *,
    parent_checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    output_path: str | Path,
    config: RetrievalScreenConfig = RetrievalScreenConfig(),
    device: str = "auto",
) -> dict[str, Any]:
    if int(config.facts_per_query) != 4:
        raise ValueError("V21 currently requires four archived facts per query")
    if int(config.train_steps) < 1 or int(config.batch_size) < 1:
        raise ValueError("V21 train_steps and batch_size must be positive")
    if not 0.0 < float(config.relation_fraction) < 1.0:
        raise ValueError("V21 relation_fraction must be strictly between zero and one")
    resolved = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for V21 but unavailable")
    started = time.perf_counter()
    parent_path = Path(parent_checkpoint_path)
    relation_path = Path(relation_corpus_path)
    cases_path = Path(relation_cases_path)
    model, tokenizer, parent_metadata = load_hashed_micro_expert_checkpoint(
        parent_path, map_location="cpu"
    )
    parent_tokens = _validate_parent(model, parent_metadata)
    if parent_tokens < 1_000_000_000:
        raise ValueError("V21 requires the one-billion-token V11 parent")
    all4_length = (
        int(config.facts_per_query) * int(config.source_length)
        + int(config.query_length)
    )
    lexical2_length = 2 * int(config.source_length) + int(config.query_length)
    if max(
        all4_length,
        lexical2_length,
        int(config.general_sequence_length),
    ) > int(model.hashed_config.context_length):
        raise ValueError("V21 active sequence exceeds the parent context")
    print("[exact-episodic-v21] sampling relation records", flush=True)
    train_records = sample_relation_training_records(
        relation_path,
        count=int(config.train_record_count),
        seed=int(config.train_data_seed),
    )
    relation_bank = encode_relation_records(
        tokenizer,
        train_records,
        source_length=int(config.source_length),
        query_length=int(config.query_length),
    )
    relation_steps = _scheduled_relation_steps(
        int(config.train_steps), float(config.relation_fraction)
    )
    relation_groups, target_slots = build_group_schedule(
        record_count=relation_bank.record_count,
        steps=relation_steps,
        batch_size=int(config.batch_size),
        facts_per_query=int(config.facts_per_query),
        seed=int(config.train_data_seed) + 1,
        record_labels=[record.query_prefix for record in train_records],
    )
    target_record_indices = relation_groups.gather(
        2, target_slots.unsqueeze(-1)
    ).squeeze(-1)
    print("[exact-episodic-v21] building label-safe lexical schedules", flush=True)
    lexical_train = lexical_episode_rankings(
        relation_bank,
        relation_groups,
        target_record_indices,
        tokenizer=tokenizer,
    )
    train_rankings = build_policy_rankings(
        lexical_train, seed=int(config.train_data_seed) + 2
    )
    prepared_general = prepare_general_language(
        tokenizer,
        train_paths=general_train_paths,
        eval_paths=general_eval_paths,
        config=config,
    )
    schedule = build_matched_schedule(
        step_count=int(config.train_steps),
        relation_fraction=float(config.relation_fraction),
        relation_batch_count=relation_steps,
        general_batch_counts=[
            len(batches) for batches in prepared_general.train_batches
        ],
        seed=int(config.train_data_seed) + 3,
    )
    if sum(kind == "relation" for kind, _index in schedule) != relation_steps:
        raise RuntimeError("V21 relation schedule count drifted")
    cases = load_relation_eval_cases(cases_path)
    retrieval_cases = load_retrieval_cases(cases_path)
    if [case.case_id for case in cases] != [case.case_id for case in retrieval_cases]:
        raise RuntimeError("V21 relation and retrieval case order diverged")
    eval_groups, eval_target_slots = build_evaluation_groups(
        case_count=len(cases),
        facts_per_query=int(config.facts_per_query),
        seed=int(config.evaluation_seed),
        case_labels=[case.query_prefix for case in cases],
    )
    if any(
        int(eval_groups[index, eval_target_slots[index]]) != index
        for index in range(len(cases))
    ):
        raise RuntimeError("V21 evaluation group lost its target source")
    case_records = [
        RelationRecord(
            source=case.source,
            query_prefix=case.query_prefix,
            answer=case.candidates[int(case.correct_index)],
        )
        for case in cases
    ]
    case_bank = encode_relation_records(
        tokenizer,
        case_records,
        source_length=int(config.source_length),
        query_length=int(config.query_length),
    )
    candidate_bank, candidate_count = _candidate_record_bank(
        tokenizer, cases, config=config
    )
    lexical_eval = lexical_episode_rankings(
        case_bank,
        eval_groups,
        torch.arange(len(cases), dtype=torch.long),
        tokenizer=tokenizer,
    )
    eval_rankings = build_policy_rankings(
        lexical_eval, seed=int(config.evaluation_seed) + 1
    )
    initial_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    model = model.to(resolved)
    print("[exact-episodic-v21] evaluating untouched general baseline", flush=True)
    general_baseline = evaluate_general_language(
        model, prepared_general.eval_batches
    )

    def evaluate_active(mode: str) -> dict[str, Any]:
        candidate = evaluate_candidate_ranking(
            mode,
            model=model,
            case_bank=case_bank,
            candidate_bank=candidate_bank,
            candidate_count=int(candidate_count),
            cases=cases,
            group_indices=eval_groups,
            policy_rankings=eval_rankings,
            config=config,
        )
        free = evaluate_free_generation(
            mode,
            model=model,
            tokenizer=tokenizer,
            case_bank=case_bank,
            cases=cases,
            group_indices=eval_groups,
            policy_rankings=eval_rankings,
            config=config,
        )
        counterfactual_rows = free.pop("_counterfactual_rows")
        return {
            **candidate,
            **free,
            "paired_counterfactual": counterfactual_behavior_metrics(
                counterfactual_rows
            ),
            "retrieval": retrieval_context_metrics(
                mode,
                group_indices=eval_groups,
                target_slots=eval_target_slots,
                policy_rankings=eval_rankings,
                cases=cases,
                source_length=int(config.source_length),
            ),
            "case_count": len(cases),
            "correct_index_metrics_only": True,
            "retrieval_uses_answer": False,
            "retrieval_uses_candidates": False,
            "retrieval_uses_target_slot": False,
        }

    rows: dict[str, dict[str, Any]] = {}
    for mode in ARM_NAMES:
        training, general_after = _run_training_arm(
            mode,
            model=model,
            initial_state=initial_state,
            relation_bank=relation_bank,
            relation_groups=relation_groups,
            target_slots=target_slots,
            policy_rankings=train_rankings,
            schedule=schedule,
            prepared_general=prepared_general,
            general_baseline=general_baseline,
            config=config,
        )
        evaluation = evaluate_active(mode)
        rows[mode] = {
            "mode": mode,
            "training": training,
            "general_language": general_after,
            "evaluation": evaluation,
        }
        print(
            f"[exact-episodic-v21] {mode} candidate="
            f"{evaluation['candidate_accuracy']:.3f} free="
            f"{evaluation['free_exact_accuracy']:.3f} paired="
            f"{evaluation['paired_counterfactual']['source_following_exact_accuracy']:.3f} "
            f"retrieval={evaluation['retrieval']['target_inclusion']:.3f} "
            f"general_delta={general_after['aggregate_heldout_loss_delta']:+.4f}",
            flush=True,
        )
    decision = retrieval_screen_decision(
        rows, train_steps=int(config.train_steps), config=config
    )
    train_lexical1 = float(
        (lexical_train[..., :1] == target_slots.unsqueeze(-1)).any(dim=-1).float().mean()
    )
    train_lexical2 = float(
        (lexical_train[..., :2] == target_slots.unsqueeze(-1)).any(dim=-1).float().mean()
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
            "archive_content": "exact_source_token_spans",
            "archive_key": "checkpoint_bpe_tfidf",
            "archive_storage_growth": "linear_in_retained_source_tokens",
            "active_source_tokens": {
                "off": 0,
                "lexical1": int(config.source_length),
                "random2": 2 * int(config.source_length),
                "recency2": 2 * int(config.source_length),
                "lexical2": 2 * int(config.source_length),
                "all4": int(config.facts_per_query) * int(config.source_length),
            },
            "lexical2_total_training_sequence_length": lexical2_length,
            "all4_total_training_sequence_length": all4_length,
            "learned_selector": False,
            "separate_memory_reader": False,
            "ordinary_local_causal_attention_reads_raw_episodes": True,
        },
        "sources": {
            "relation_corpus": {
                "path": str(relation_path),
                "sha256": sha256_file(relation_path),
                "sampled_records": len(train_records),
                "distinct_query_identities": len(
                    {record.query_prefix for record in train_records}
                ),
            },
            "relation_cases": {
                "path": str(cases_path),
                "sha256": sha256_file(cases_path),
                "case_count": len(cases),
            },
            "general": prepared_general.source_reports,
        },
        "schedule": {
            "identical_for_every_arm": True,
            "schedule_sha256": schedule_sha256(schedule),
            "relation_group_sha256": _tensor_sha256(
                relation_groups, target_slots
            ),
            "training_lexical_ranking_sha256": _tensor_sha256(lexical_train),
            "evaluation_group_sha256": _tensor_sha256(
                eval_groups, eval_target_slots
            ),
            "evaluation_lexical_ranking_sha256": _tensor_sha256(lexical_eval),
            "training_lexical_target_inclusion_at_1": train_lexical1,
            "training_lexical_target_inclusion_at_2": train_lexical2,
            "steps": int(config.train_steps),
            "relation_steps": relation_steps,
            "general_steps": int(config.train_steps) - relation_steps,
        },
        "anti_cheat": {
            "archive_write_input": "source_tokens_only",
            "retrieval_read_input": "question_prefix_without_answer",
            "retrieval_uses_answer": False,
            "retrieval_uses_candidates": False,
            "retrieval_uses_target_slot": False,
            "target_slot_metrics_only": True,
            "teacher_forcing_visible_only_after_retrieval": True,
            "teacher_forcing_visible_only_to_ordinary_prior_answer_positions": True,
            "paired_source_swap_holds_question_distractors_and_positions_fixed": True,
        },
        "general_language_before": general_baseline,
        "arms": rows,
        "decision": decision,
        "checkpoint": None,
        "promotion_boundary": {
            "advance_to_contiguous_streams": decision == ADVANCE_DECISION,
            "retrieval_recall_is_language_quality": False,
            "base_quality_promoted": False,
            "runtime_install_allowed": False,
            "continual_learning_claimed": False,
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
        title="MARULHO V21 Exact Episodic Retrieval Language Screen",
    )
    print(f"[exact-episodic-v21] decision {decision}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--relation-corpus", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--general-train", type=Path, nargs=2, required=True)
    parser.add_argument("--general-eval", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-records", type=int, default=8192)
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--generation-max-tokens", type=int, default=16)
    parser.add_argument("--general-eval-batches", type=int, default=8)
    parser.add_argument(
        "--general-train-sample-bytes", type=int, default=8 * 1024 * 1024
    )
    parser.add_argument(
        "--general-eval-sample-bytes", type=int, default=8 * 1024 * 1024
    )
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = RetrievalScreenConfig(
        train_record_count=int(args.train_records),
        train_steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        generation_max_tokens=int(args.generation_max_tokens),
        general_eval_batches=int(args.general_eval_batches),
        general_train_sample_bytes=int(args.general_train_sample_bytes),
        general_eval_sample_bytes=int(args.general_eval_sample_bytes),
        precision=str(args.precision),
    )
    run_exact_episodic_retrieval_screen(
        parent_checkpoint_path=args.parent_checkpoint,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=args.general_train,
        general_eval_paths=args.general_eval,
        output_path=args.output,
        config=config,
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
