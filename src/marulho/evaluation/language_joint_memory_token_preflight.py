"""Jointly train bounded recurrent memory tokens with the V11 language cortex."""

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
from torch import nn
import torch.nn.functional as F

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


SURFACE = "marulho_joint_memory_token_preflight.v1"
ARTIFACT_KIND = "marulho_joint_memory_token_preflight"
ARM_NAMES = ("off", "exact", "local", "recency", "mean", "recurrent")
MINIMUM_DECISION_STEPS = 512
ADVANCE_DECISION = "advance_v19_joint_memory_tokens_to_contiguous_language"


@dataclass(frozen=True)
class RelationMemoryRecord:
    source: str
    query_prefix: str
    answer: str


@dataclass(frozen=True)
class RelationMemoryCase:
    case_id: str
    kind: str
    source: str
    query_prefix: str
    candidates: tuple[str, ...]
    correct_index: int


@dataclass(frozen=True)
class EncodedRecordBank:
    source_ids: torch.Tensor
    source_mask: torch.Tensor
    query_input_ids: torch.Tensor
    query_target_ids: torch.Tensor
    query_loss_mask: torch.Tensor

    @property
    def record_count(self) -> int:
        return int(self.source_ids.shape[0])


@dataclass(frozen=True)
class JointMemoryTokenConfig:
    train_record_count: int = 8192
    facts_per_example: int = 4
    source_segments: int = 2
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
    data_seed: int = 9301
    model_seed: int = 9401
    slot_count: int = 16
    initial_memory_scale: float = 0.03
    generation_max_tokens: int = 16
    minimum_exact_candidate_accuracy: float = 0.75
    minimum_exact_free_accuracy: float = 0.30
    minimum_exact_counterfactual_gain: float = 0.10
    minimum_recurrent_candidate_accuracy: float = 0.70
    minimum_recurrent_free_accuracy: float = 0.25
    minimum_recurrent_counterfactual_accuracy: float = 0.25
    minimum_recurrent_control_gain: float = 0.05
    maximum_recurrent_counterfactual_regret_to_exact: float = 0.10
    maximum_general_loss_regression: float = 0.10


@dataclass(frozen=True)
class PreparedGeneralLanguage:
    train_batches: tuple[tuple[LanguageBatch, ...], ...]
    eval_batches: tuple[tuple[LanguageBatch, ...], ...]
    source_reports: dict[str, Any]


def parse_relation_training_line(line: str) -> RelationMemoryRecord | None:
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
    return RelationMemoryRecord(
        source=source,
        query_prefix=f"Question: {question} Answer: ",
        answer=answer,
    )


def sample_relation_training_records(
    path: str | Path,
    *,
    count: int,
    seed: int,
) -> tuple[RelationMemoryRecord, ...]:
    if int(count) < 1:
        raise ValueError("training record count must be positive")
    generator = random.Random(int(seed))
    reservoir: list[RelationMemoryRecord] = []
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


def split_relation_case_prompt(prompt: str) -> tuple[str, str]:
    text = str(prompt).strip()
    if not text.endswith("Answer:"):
        raise ValueError("relation case prompt must end with Answer:")
    without_answer = text[: -len("Answer:")].rstrip()
    if ". " not in without_answer:
        raise ValueError("relation case prompt has no source/question boundary")
    source, question = without_answer.rsplit(". ", 1)
    source = f"{source.strip()}."
    question = question.strip()
    if not question.endswith("?"):
        raise ValueError("relation case query must end with a question mark")
    return source, f"Question: {question} Answer: "


def load_relation_memory_cases(path: str | Path) -> tuple[RelationMemoryCase, ...]:
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
            RelationMemoryCase(
                case_id=str(row["case_id"]),
                kind=str(row["kind"]),
                source=source,
                query_prefix=query_prefix,
                candidates=candidates,
                correct_index=correct_index,
            )
        )
    return tuple(cases)


def _encode_record(
    tokenizer,
    record: RelationMemoryRecord,
    *,
    source_length: int,
    query_length: int,
) -> tuple[list[int], list[int], list[int], list[bool]]:
    source = tokenizer.encode(record.source, add_bos=True, add_eos=True)
    prefix = tokenizer.encode(record.query_prefix, add_bos=True, add_eos=False)
    answer = tokenizer.encode(record.answer, add_bos=False, add_eos=True)
    sequence = [*prefix, *answer]
    query_input = sequence[:-1]
    query_targets = sequence[1:]
    answer_start = len(prefix) - 1
    loss_mask = [index >= answer_start for index in range(len(query_input))]
    if len(source) > int(source_length):
        raise ValueError(f"source token length {len(source)} exceeds {source_length}")
    if len(query_input) > int(query_length):
        raise ValueError(
            f"query token length {len(query_input)} exceeds {query_length}"
        )
    return source, query_input, query_targets, loss_mask


def encode_relation_records(
    tokenizer,
    records: Sequence[RelationMemoryRecord],
    *,
    source_length: int,
    query_length: int,
) -> EncodedRecordBank:
    count = len(records)
    source_ids = torch.full(
        (count, int(source_length)), int(tokenizer.pad_id), dtype=torch.long
    )
    source_mask = torch.zeros((count, int(source_length)), dtype=torch.bool)
    query_input_ids = torch.full(
        (count, int(query_length)), int(tokenizer.pad_id), dtype=torch.long
    )
    query_target_ids = torch.full(
        (count, int(query_length)), -100, dtype=torch.long
    )
    query_loss_mask = torch.zeros((count, int(query_length)), dtype=torch.bool)
    for index, record in enumerate(records):
        source, query_input, targets, loss_mask = _encode_record(
            tokenizer,
            record,
            source_length=int(source_length),
            query_length=int(query_length),
        )
        source_ids[index, : len(source)] = torch.tensor(source)
        source_mask[index, : len(source)] = True
        query_input_ids[index, : len(query_input)] = torch.tensor(query_input)
        query_target_ids[index, : len(targets)] = torch.tensor(targets)
        query_loss_mask[index, : len(loss_mask)] = torch.tensor(loss_mask)
    return EncodedRecordBank(
        source_ids=source_ids,
        source_mask=source_mask,
        query_input_ids=query_input_ids,
        query_target_ids=query_target_ids,
        query_loss_mask=query_loss_mask,
    )


def build_group_schedule(
    *,
    record_count: int,
    steps: int,
    batch_size: int,
    facts_per_example: int,
    seed: int,
    record_labels: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(record_labels) != int(record_count):
        raise ValueError("record_labels length must equal record_count")
    buckets: dict[str, list[int]] = {}
    for index, label in enumerate(record_labels):
        buckets.setdefault(str(label), []).append(index)
    labels = sorted(buckets)
    if len(labels) < int(facts_per_example):
        raise ValueError("not enough distinct record labels for one fact group")
    generator = random.Random(int(seed))
    rows = []
    targets = []
    for _step in range(int(steps)):
        batch_rows = []
        batch_targets = []
        for _row in range(int(batch_size)):
            chosen = generator.sample(labels, int(facts_per_example))
            batch_rows.append([generator.choice(buckets[label]) for label in chosen])
            batch_targets.append(generator.randrange(int(facts_per_example)))
        rows.append(batch_rows)
        targets.append(batch_targets)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(
        targets, dtype=torch.long
    )


def build_evaluation_groups(
    *,
    case_count: int,
    facts_per_example: int,
    seed: int,
    case_labels: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(case_labels) != int(case_count):
        raise ValueError("case_labels length must equal case_count")
    buckets: dict[str, list[int]] = {}
    for index, label in enumerate(case_labels):
        buckets.setdefault(str(label), []).append(index)
    if len(buckets) < int(facts_per_example):
        raise ValueError("not enough distinct case labels for one fact group")
    generator = random.Random(int(seed))
    plans: dict[str, tuple[list[int], int]] = {}
    for target_label in sorted(buckets):
        distractor_labels = generator.sample(
            [label for label in buckets if label != target_label],
            int(facts_per_example) - 1,
        )
        distractors = [generator.choice(buckets[label]) for label in distractor_labels]
        plans[target_label] = (
            distractors,
            generator.randrange(int(facts_per_example)),
        )
    rows = []
    target_slots = []
    for target in range(int(case_count)):
        distractors, target_slot = plans[str(case_labels[target])]
        row = list(distractors)
        row.insert(int(target_slot), target)
        rows.append(row)
        target_slots.append(int(target_slot))
    return torch.tensor(rows, dtype=torch.long), torch.tensor(
        target_slots, dtype=torch.long
    )


def _tensor_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().contiguous().cpu()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _state_dict_sha256(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
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


class JointMemoryTokenOrgan(nn.Module):
    """Small learned token interface for cortex-owned recurrent state."""

    def __init__(
        self,
        *,
        width: int,
        slot_count: int,
        replay_id: int,
        initial_scale: float,
    ) -> None:
        super().__init__()
        if int(width) < 1 or int(slot_count) < 1:
            raise ValueError("memory width and slot_count must be positive")
        if not math.isfinite(float(initial_scale)) or float(initial_scale) <= 0.0:
            raise ValueError("initial memory scale must be finite and positive")
        self.width = int(width)
        self.slot_count = int(slot_count)
        self.memory_seed = nn.Parameter(torch.empty(self.slot_count, self.width))
        self.write_queries = nn.Parameter(torch.empty(self.slot_count, self.width))
        self.local_memory = nn.Parameter(torch.empty(self.slot_count, self.width))
        self.reentry_norm = nn.LayerNorm(self.width)
        self.reentry_scale = nn.Parameter(torch.tensor(float(initial_scale)))
        self.register_buffer(
            "route_ids",
            torch.arange(self.slot_count, dtype=torch.long) + int(replay_id),
        )
        for value in (self.memory_seed, self.write_queries, self.local_memory):
            nn.init.normal_(value, mean=0.0, std=0.02)

    def expand(self, value: torch.Tensor, batch_size: int) -> torch.Tensor:
        return value.unsqueeze(0).expand(int(batch_size), -1, -1)

    def routes(self, batch_size: int) -> torch.Tensor:
        return self.route_ids.unsqueeze(0).expand(int(batch_size), -1)

    def reenter(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.reentry_norm(hidden) * self.reentry_scale

    def state_bytes_per_stream(self, *, element_size: int = 4) -> int:
        return self.slot_count * self.width * int(element_size)


class JointMemoryTokenCortex(nn.Module):
    """V11 cortex with exact local attention and bounded recurrent memory tokens."""

    def __init__(
        self,
        model: MarulhoHashedMicroExpertLanguageModel,
        memory: JointMemoryTokenOrgan,
        *,
        facts_per_example: int,
        source_segments: int,
    ) -> None:
        super().__init__()
        if int(facts_per_example) % int(source_segments) != 0:
            raise ValueError("facts_per_example must divide across source_segments")
        if int(memory.width) != int(model.hashed_config.width):
            raise ValueError("memory width must match the language cortex")
        self.model = model
        self.memory = memory
        self.facts_per_example = int(facts_per_example)
        self.source_segments = int(source_segments)

    @property
    def device(self) -> torch.device:
        return self.model.device

    def _state_hidden(
        self,
        embeddings: torch.Tensor,
        route_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden, _state, _telemetry = self.model.state_block(
            embeddings,
            route_ids,
            None,
            collect_telemetry=False,
        )
        return hidden

    def _summarize_segment(
        self,
        memory: torch.Tensor,
        source_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch = int(source_ids.shape[0])
        queries = self.memory.expand(self.memory.write_queries, batch)
        memory_routes = self.memory.routes(batch).to(source_ids.device)
        embeddings = torch.cat(
            (memory, self.model.token_embedding(source_ids), queries), dim=1
        )
        routes = torch.cat((memory_routes, source_ids, memory_routes), dim=1)
        hidden = self._state_hidden(embeddings, routes)
        return self.memory.reenter(hidden[:, -self.memory.slot_count :])

    def build_source_state(
        self,
        mode: str,
        grouped_source_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        if mode not in ARM_NAMES:
            raise ValueError(f"unknown V19 mode: {mode}")
        if grouped_source_ids.ndim != 3:
            raise ValueError("grouped source ids must be [batch,facts,time]")
        batch, facts, source_length = grouped_source_ids.shape
        if int(facts) != self.facts_per_example:
            raise ValueError("grouped source fact count does not match the cortex")
        if mode == "exact":
            return grouped_source_ids.reshape(int(batch), int(facts) * int(source_length))
        if mode in {"off", "local"}:
            return None
        facts_per_segment = int(facts) // self.source_segments
        segmented = grouped_source_ids.reshape(
            int(batch),
            self.source_segments,
            facts_per_segment * int(source_length),
        )
        seed = self.memory.expand(self.memory.memory_seed, int(batch))
        if mode == "recency":
            return self._summarize_segment(seed, segmented[:, -1])
        if mode == "mean":
            states = [
                self._summarize_segment(seed, segmented[:, index])
                for index in range(self.source_segments)
            ]
            return torch.stack(states, dim=0).mean(dim=0)
        if mode == "recurrent":
            state = seed
            for index in range(self.source_segments):
                state = self._summarize_segment(state, segmented[:, index])
            return state
        raise ValueError(f"mode {mode} cannot build a source state")

    def query_logits(
        self,
        mode: str,
        source_state: torch.Tensor | None,
        query_ids: torch.Tensor,
    ) -> torch.Tensor:
        if query_ids.ndim != 2:
            raise ValueError("query ids must be [batch,time]")
        batch = int(query_ids.shape[0])
        if mode == "off":
            return self.model(query_ids, collect_telemetry=False)["logits"]
        if mode == "exact":
            if source_state is None or source_state.dtype != torch.long:
                raise ValueError("exact mode requires raw source token ids")
            combined = torch.cat((source_state, query_ids), dim=1)
            if int(combined.shape[1]) > int(self.model.hashed_config.context_length):
                raise ValueError("exact source and query exceed the cortex context")
            hidden = self.model._forward_hidden(
                combined, collect_telemetry=False
            )["hidden"]
            return self.model.lm_head(hidden[:, -int(query_ids.shape[1]) :])
        if mode == "local":
            memory = self.memory.expand(self.memory.local_memory, batch)
        else:
            if source_state is None or not source_state.is_floating_point():
                raise ValueError("bounded memory mode requires floating source state")
            memory = source_state
        memory_routes = self.memory.routes(batch).to(query_ids.device)
        embeddings = torch.cat(
            (memory, self.model.token_embedding(query_ids)), dim=1
        )
        routes = torch.cat((memory_routes, query_ids), dim=1)
        hidden = self._state_hidden(embeddings, routes)
        return self.model.lm_head(hidden[:, -int(query_ids.shape[1]) :])

    def relation_loss(
        self,
        mode: str,
        grouped_source_ids: torch.Tensor,
        query_ids: torch.Tensor,
        target_ids: torch.Tensor,
        answer_mask: torch.Tensor,
    ) -> torch.Tensor:
        source_state = self.build_source_state(mode, grouped_source_ids)
        logits = self.query_logits(mode, source_state, query_ids)
        return F.cross_entropy(logits[answer_mask], target_ids[answer_mask])


def _grouped_source_batch(
    bank: EncodedRecordBank,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, facts = indices.shape
    flat = indices.reshape(-1)
    ids = bank.source_ids.index_select(0, flat).reshape(
        int(batch), int(facts), int(bank.source_ids.shape[1])
    )
    mask = bank.source_mask.index_select(0, flat).reshape(
        int(batch), int(facts), int(bank.source_mask.shape[1])
    )
    return ids.to(device), mask.to(device)


def _target_query_batch(
    bank: EncodedRecordBank,
    indices: torch.Tensor,
    target_slots: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = indices.gather(1, target_slots.unsqueeze(1)).squeeze(1)
    return (
        bank.query_input_ids.index_select(0, selected).to(device),
        bank.query_target_ids.index_select(0, selected).to(device),
        bank.query_loss_mask.index_select(0, selected).to(device),
    )


def prepare_general_language(
    tokenizer,
    *,
    train_paths: Sequence[str | Path],
    eval_paths: Sequence[str | Path],
    config: JointMemoryTokenConfig,
) -> PreparedGeneralLanguage:
    if len(train_paths) != 2 or len(eval_paths) != 2:
        raise ValueError("V19 requires exactly two general train and eval sources")
    general_steps = max(
        1, int(config.train_steps) - round(config.train_steps * config.relation_fraction)
    )
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


def joint_memory_token_decision(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    train_steps: int,
    config: JointMemoryTokenConfig,
) -> str:
    if set(rows) != set(ARM_NAMES):
        return "incomplete_v19_missing_control_arm"
    if int(train_steps) < MINIMUM_DECISION_STEPS:
        return "diagnostic_v19_below_preflight_step_floor"
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
        candidate["exact"] < float(config.minimum_exact_candidate_accuracy)
        or free["exact"] < float(config.minimum_exact_free_accuracy)
        or paired["exact"] - paired["local"]
        < float(config.minimum_exact_counterfactual_gain)
    ):
        return "retire_v19_task_not_learnable_from_exact_history"
    maximum_general_regression = max(
        float(row["heldout_loss_delta"])
        for row in rows["recurrent"]["general_language"]["sources"]
    )
    if maximum_general_regression > float(config.maximum_general_loss_regression):
        return "retire_v19_recurrent_memory_breaks_general_language"
    controls = ("local", "recency", "mean")
    candidate_control = max(candidate[name] for name in controls)
    free_control = max(free[name] for name in controls)
    paired_control = max(paired[name] for name in controls)
    recurrent_pass = (
        candidate["recurrent"] >= float(config.minimum_recurrent_candidate_accuracy)
        and free["recurrent"] >= float(config.minimum_recurrent_free_accuracy)
        and paired["recurrent"]
        >= float(config.minimum_recurrent_counterfactual_accuracy)
        and candidate["recurrent"] - candidate_control
        >= float(config.minimum_recurrent_control_gain)
        and free["recurrent"] - free_control
        >= float(config.minimum_recurrent_control_gain)
        and paired["recurrent"] - paired_control
        >= float(config.minimum_recurrent_control_gain)
        and paired["exact"] - paired["recurrent"]
        <= float(config.maximum_recurrent_counterfactual_regret_to_exact)
    )
    if recurrent_pass:
        return ADVANCE_DECISION
    if max(paired["recency"], paired["mean"]) >= paired["recurrent"]:
        return "retire_v19_simple_summary_matches_recurrent_memory"
    return "retire_v19_joint_memory_tokens_insufficient_source_following"


def _candidate_record_bank(
    tokenizer,
    cases: Sequence[RelationMemoryCase],
    *,
    config: JointMemoryTokenConfig,
) -> tuple[EncodedRecordBank, int]:
    candidate_count = len(cases[0].candidates)
    if any(len(case.candidates) != candidate_count for case in cases):
        raise ValueError("relation cases have inconsistent candidate counts")
    records = [
        RelationMemoryRecord(
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
    cortex: JointMemoryTokenCortex,
    case_bank: EncodedRecordBank,
    candidate_bank: EncodedRecordBank,
    candidate_count: int,
    cases: Sequence[RelationMemoryCase],
    group_indices: torch.Tensor,
    config: JointMemoryTokenConfig,
) -> dict[str, Any]:
    cortex.eval()
    device = cortex.device
    predictions = []
    score_rows = []
    for start in range(0, len(cases), int(config.eval_batch_size)):
        end = min(len(cases), start + int(config.eval_batch_size))
        indices = group_indices[start:end]
        grouped_source, _grouped_mask = _grouped_source_batch(
            case_bank, indices, device=device
        )
        with _precision_context(device, str(config.precision)):
            source_state = cortex.build_source_state(mode, grouped_source)
        candidate_scores = []
        for candidate in range(int(candidate_count)):
            flat = (
                torch.arange(start, end, dtype=torch.long) * int(candidate_count)
                + candidate
            )
            query_ids = candidate_bank.query_input_ids.index_select(0, flat).to(device)
            targets = candidate_bank.query_target_ids.index_select(0, flat).to(device)
            mask = candidate_bank.query_loss_mask.index_select(0, flat).to(device)
            with _precision_context(device, str(config.precision)):
                logits = cortex.query_logits(mode, source_state, query_ids)
                losses = F.cross_entropy(
                    logits.reshape(-1, int(logits.shape[-1])),
                    targets.reshape(-1),
                    reduction="none",
                    ignore_index=-100,
                ).reshape(targets.shape)
                score = (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
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
    cortex: JointMemoryTokenCortex,
    tokenizer,
    case_bank: EncodedRecordBank,
    cases: Sequence[RelationMemoryCase],
    group_indices: torch.Tensor,
    config: JointMemoryTokenConfig,
) -> dict[str, Any]:
    cortex.eval()
    device = cortex.device
    exact_rows = []
    contains_rows = []
    examples = []
    counterfactual_rows = []
    for start in range(0, len(cases), int(config.eval_batch_size)):
        end = min(len(cases), start + int(config.eval_batch_size))
        selected_cases = cases[start:end]
        grouped_source, _grouped_mask = _grouped_source_batch(
            case_bank, group_indices[start:end], device=device
        )
        with _precision_context(device, str(config.precision)):
            source_state = cortex.build_source_state(mode, grouped_source)
        sequences = [
            tokenizer.encode(case.query_prefix, add_bos=True, add_eos=False)
            for case in selected_cases
        ]
        generated = [[] for _case in selected_cases]
        finished = [False for _case in selected_cases]
        for _step in range(int(config.generation_max_tokens)):
            maximum = max(len(sequence) for sequence in sequences)
            if maximum > int(config.query_length):
                raise RuntimeError("V19 free generation exceeded the query budget")
            input_ids = torch.full(
                (len(sequences), maximum),
                int(tokenizer.pad_id),
                device=device,
                dtype=torch.long,
            )
            lengths = []
            for row, sequence in enumerate(sequences):
                input_ids[row, : len(sequence)] = torch.tensor(
                    sequence, device=device, dtype=torch.long
                )
                lengths.append(len(sequence))
            with _precision_context(device, str(config.precision)):
                logits = cortex.query_logits(mode, source_state, input_ids)
                rows = torch.arange(len(sequences), device=device)
                positions = torch.tensor(lengths, device=device) - 1
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
        source = {
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
        sources.append(source)
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


@torch.no_grad()
def memory_diagnostic(
    mode: str,
    *,
    cortex: JointMemoryTokenCortex,
    case_bank: EncodedRecordBank,
    group_indices: torch.Tensor,
    config: JointMemoryTokenConfig,
) -> dict[str, Any]:
    indices = group_indices[: min(32, int(group_indices.shape[0]))]
    grouped_source, _mask = _grouped_source_batch(
        case_bank, indices, device=cortex.device
    )
    with _precision_context(cortex.device, str(config.precision)):
        state = cortex.build_source_state(mode, grouped_source)
    if mode == "off":
        return {
            "bounded_state": True,
            "source_dependent": False,
            "state_elements_per_stream": 0,
            "state_bytes_per_stream_float32": 0,
            "promotion_metric": False,
        }
    if mode == "exact":
        assert state is not None
        return {
            "bounded_state": False,
            "source_dependent": True,
            "exact_history_tokens_per_stream": int(state.shape[1]),
            "raw_history_bytes_per_stream_int64": int(state.shape[1]) * 8,
            "promotion_metric": False,
        }
    if mode == "local":
        state = cortex.memory.expand(
            cortex.memory.local_memory, int(grouped_source.shape[0])
        )
    if state is None:
        raise RuntimeError(f"V19 {mode} diagnostic has no state")
    matrix = state.detach().float().reshape(-1, int(state.shape[-1])).cpu()
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    variance = singular.square()
    probability = variance / variance.sum().clamp_min(1.0e-12)
    effective = torch.exp(
        -(probability * probability.clamp_min(1.0e-12).log()).sum()
    )
    return {
        "bounded_state": True,
        "source_dependent": mode not in {"off", "local"},
        "state_elements_per_stream": int(state.shape[1] * state.shape[2]),
        "state_bytes_per_stream_float32": cortex.memory.state_bytes_per_stream(),
        "matrix_rank": int(torch.linalg.matrix_rank(centered)),
        "effective_rank": float(effective),
        "mean_state_norm": float(matrix.norm(dim=-1).mean()),
        "learned_reentry_scale": float(cortex.memory.reentry_scale.detach().cpu()),
        "stable_slot_route_ids": cortex.memory.route_ids.detach().cpu().tolist(),
        "promotion_metric": False,
    }


def _gradient_report(cortex: JointMemoryTokenCortex, mode: str) -> dict[str, Any]:
    memory_rows = []
    for name, parameter in cortex.memory.named_parameters():
        gradient = parameter.grad
        memory_rows.append(
            {
                "name": name,
                "received_gradient": gradient is not None,
                "nonzero_gradient_elements": (
                    0
                    if gradient is None
                    else int(torch.count_nonzero(gradient).detach().cpu())
                ),
            }
        )
    recurrent_core = [
        row for row in memory_rows if row["name"] != "local_memory"
    ]
    model_rows = []
    for name, parameter in cortex.model.named_parameters():
        gradient = parameter.grad
        model_rows.append(
            {
                "name": name,
                "received_gradient": gradient is not None,
                "nonzero_gradient_elements": (
                    0
                    if gradient is None
                    else int(torch.count_nonzero(gradient).detach().cpu())
                ),
            }
        )
    return {
        "probe_mode": mode,
        "probe_updates_parameters": False,
        "memory_parameters": memory_rows,
        "recurrent_core_all_received_gradient": all(
            bool(row["received_gradient"]) for row in recurrent_core
        ),
        "recurrent_core_all_nonzero": all(
            int(row["nonzero_gradient_elements"]) > 0 for row in recurrent_core
        ),
        "model_parameter_tensor_count": len(model_rows),
        "model_parameter_tensors_with_gradient": sum(
            bool(row["received_gradient"]) for row in model_rows
        ),
        "model_parameter_tensors_with_nonzero_gradient": sum(
            int(row["nonzero_gradient_elements"]) > 0 for row in model_rows
        ),
        "hashed_expert_rows": cortex.model.final_gradient_report(),
    }


def _relation_positions_per_example(
    mode: str,
    config: JointMemoryTokenConfig,
) -> int:
    facts = int(config.facts_per_example)
    segments = int(config.source_segments)
    source = int(config.source_length)
    query = int(config.query_length)
    slots = int(config.slot_count)
    if mode == "off":
        return query
    if mode == "exact":
        return facts * source + query
    if mode == "local":
        return slots + query
    segment_positions = slots + (facts // segments) * source + slots
    query_positions = slots + query
    if mode == "recency":
        return segment_positions + query_positions
    if mode in {"mean", "recurrent"}:
        return segments * segment_positions + query_positions
    raise ValueError(f"unknown V19 mode: {mode}")


def _run_training_arm(
    mode: str,
    *,
    cortex: JointMemoryTokenCortex,
    initial_model_state: Mapping[str, torch.Tensor],
    initial_memory_state: Mapping[str, torch.Tensor],
    relation_bank: EncodedRecordBank,
    relation_indices: torch.Tensor,
    target_slots: torch.Tensor,
    schedule: Sequence[tuple[str, int]],
    prepared_general: PreparedGeneralLanguage,
    general_baseline: Mapping[str, Any],
    config: JointMemoryTokenConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cortex.model.load_state_dict(dict(initial_model_state), strict=True)
    cortex.memory.load_state_dict(dict(initial_memory_state), strict=True)
    cortex.train()
    torch.manual_seed(int(config.model_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.model_seed))
    optimizer = torch.optim.AdamW(
        cortex.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        fused=bool(cortex.device.type == "cuda"),
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
    observed_source_tokens = 0
    general_tokens = 0
    cortex_positions = 0
    if cortex.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(cortex.device)
        torch.cuda.synchronize(cortex.device)
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
        with _precision_context(cortex.device, str(config.precision)):
            if kind == "relation":
                indices = relation_indices[int(source_index)]
                slots = target_slots[int(source_index)]
                grouped_source, grouped_mask = _grouped_source_batch(
                    relation_bank, indices, device=cortex.device
                )
                query_ids, targets, answer_mask = _target_query_batch(
                    relation_bank, indices, slots, device=cortex.device
                )
                loss = cortex.relation_loss(
                    mode, grouped_source, query_ids, targets, answer_mask
                )
                relation_step_count += 1
                supervised_answer_tokens += int(answer_mask.sum())
                observed_source_tokens += int(grouped_mask.sum())
                cortex_positions += (
                    _relation_positions_per_example(mode, config)
                    * int(config.batch_size)
                )
            else:
                general_source = int(kind.rsplit("_", 1)[1])
                batch = prepared_general.train_batches[general_source][
                    int(source_index)
                ].to(cortex.device)
                loss = cortex.model.next_token_loss(
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
            raise RuntimeError(f"non-finite V19 loss in {mode}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            cortex.parameters(), float(config.gradient_clip)
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"non-finite V19 gradient in {mode}")
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
                f"[joint-memory-v19] {mode} {step + 1}/{config.train_steps}",
                flush=True,
            )
    if cortex.device.type == "cuda":
        torch.cuda.synchronize(cortex.device)
    elapsed = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(cortex.device))
        if cortex.device.type == "cuda"
        else 0
    )
    cortex.zero_grad(set_to_none=True)
    probe_indices = relation_indices[0]
    probe_slots = target_slots[0]
    probe_source, _probe_mask = _grouped_source_batch(
        relation_bank, probe_indices, device=cortex.device
    )
    probe_query, probe_targets, probe_answer_mask = _target_query_batch(
        relation_bank, probe_indices, probe_slots, device=cortex.device
    )
    with _precision_context(cortex.device, str(config.precision)):
        probe_loss = cortex.relation_loss(
            mode,
            probe_source,
            probe_query,
            probe_targets,
            probe_answer_mask,
        )
    probe_loss.backward()
    gradient = _gradient_report(cortex, mode)
    cortex.zero_grad(set_to_none=True)
    general_after = evaluate_general_language(
        cortex.model,
        prepared_general.eval_batches,
        baseline=general_baseline,
    )
    training = {
        "optimizer": "AdamW",
        "optimizer_state_fresh": True,
        "initial_model_state_restored": True,
        "initial_memory_state_restored": True,
        "steps": int(config.train_steps),
        "relation_steps": relation_step_count,
        "general_replay_steps": general_step_count,
        "supervised_answer_tokens": supervised_answer_tokens,
        "observed_source_tokens": observed_source_tokens,
        "general_training_tokens": general_tokens,
        "cortex_input_positions": cortex_positions,
        "elapsed_seconds": elapsed,
        "cortex_positions_per_second": cortex_positions / max(elapsed, 1.0e-12),
        "peak_cuda_memory_bytes": peak_memory,
        "loss_trace": trace,
        "gradient": gradient,
        "execution_backend": "eager",
    }
    return training, general_after


def _scheduled_relation_steps(step_count: int, relation_fraction: float) -> int:
    accumulator = 0.0
    count = 0
    for _ in range(int(step_count)):
        accumulator += float(relation_fraction)
        if accumulator >= 1.0:
            accumulator -= 1.0
            count += 1
    return count


def run_joint_memory_token_preflight(
    *,
    parent_checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    output_path: str | Path,
    config: JointMemoryTokenConfig = JointMemoryTokenConfig(),
    device: str = "auto",
) -> dict[str, Any]:
    if int(config.facts_per_example) % int(config.source_segments) != 0:
        raise ValueError("V19 facts_per_example must divide across source_segments")
    if int(config.train_steps) < 1 or int(config.batch_size) < 1:
        raise ValueError("V19 train_steps and batch_size must be positive")
    if not 0.0 < float(config.relation_fraction) < 1.0:
        raise ValueError("V19 relation_fraction must be strictly between zero and one")
    resolved = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for V19 but unavailable")
    started = time.perf_counter()
    parent_path = Path(parent_checkpoint_path)
    relation_path = Path(relation_corpus_path)
    cases_path = Path(relation_cases_path)
    general_train = tuple(Path(path) for path in general_train_paths)
    general_eval = tuple(Path(path) for path in general_eval_paths)
    model, tokenizer, parent_metadata = load_hashed_micro_expert_checkpoint(
        parent_path, map_location="cpu"
    )
    parent_tokens = _validate_parent(model, parent_metadata)
    if parent_tokens < 1_000_000_000:
        raise ValueError("V19 requires the one-billion-token V11 parent")
    context = int(model.hashed_config.context_length)
    facts_per_segment = int(config.facts_per_example) // int(config.source_segments)
    exact_length = (
        int(config.facts_per_example) * int(config.source_length)
        + int(config.query_length)
    )
    segment_length = (
        int(config.slot_count)
        + facts_per_segment * int(config.source_length)
        + int(config.slot_count)
    )
    memory_query_length = int(config.slot_count) + int(config.query_length)
    maximum_length = max(
        exact_length,
        segment_length,
        memory_query_length,
        int(config.general_sequence_length),
    )
    if maximum_length > context:
        raise ValueError(
            f"V19 maximum sequence {maximum_length} exceeds parent context {context}"
        )
    print("[joint-memory-v19] sampling relation records", flush=True)
    train_records = sample_relation_training_records(
        relation_path,
        count=int(config.train_record_count),
        seed=int(config.data_seed),
    )
    cases = load_relation_memory_cases(cases_path)
    relation_bank = encode_relation_records(
        tokenizer,
        train_records,
        source_length=int(config.source_length),
        query_length=int(config.query_length),
    )
    case_records = [
        RelationMemoryRecord(
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
    print("[joint-memory-v19] preparing general replay and holdouts", flush=True)
    prepared_general = prepare_general_language(
        tokenizer,
        train_paths=general_train,
        eval_paths=general_eval,
        config=config,
    )
    relation_steps = _scheduled_relation_steps(
        int(config.train_steps), float(config.relation_fraction)
    )
    if relation_steps < 1:
        raise ValueError("V19 schedule contains no relation steps")
    relation_indices, target_slots = build_group_schedule(
        record_count=relation_bank.record_count,
        steps=relation_steps,
        batch_size=int(config.batch_size),
        facts_per_example=int(config.facts_per_example),
        seed=int(config.data_seed) + 1,
        record_labels=[record.query_prefix for record in train_records],
    )
    schedule = build_matched_schedule(
        step_count=int(config.train_steps),
        relation_fraction=float(config.relation_fraction),
        relation_batch_count=relation_steps,
        general_batch_counts=[
            len(batches) for batches in prepared_general.train_batches
        ],
        seed=int(config.data_seed) + 2,
    )
    if sum(kind == "relation" for kind, _index in schedule) != relation_steps:
        raise RuntimeError("V19 relation schedule count drifted from its episode bank")
    eval_groups, eval_target_slots = build_evaluation_groups(
        case_count=len(cases),
        facts_per_example=int(config.facts_per_example),
        seed=int(config.data_seed) + 3,
        case_labels=[case.query_prefix for case in cases],
    )
    if any(
        int(eval_groups[index, eval_target_slots[index]]) != index
        for index in range(len(cases))
    ):
        raise RuntimeError("V19 evaluation group lost its target source")
    initial_model_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    torch.manual_seed(int(config.model_seed))
    memory = JointMemoryTokenOrgan(
        width=int(model.hashed_config.width),
        slot_count=int(config.slot_count),
        replay_id=int(tokenizer.replay_id),
        initial_scale=float(config.initial_memory_scale),
    )
    initial_memory_state = {
        name: value.detach().clone() for name, value in memory.state_dict().items()
    }
    cortex = JointMemoryTokenCortex(
        model,
        memory,
        facts_per_example=int(config.facts_per_example),
        source_segments=int(config.source_segments),
    ).to(resolved)
    parity_query = relation_bank.query_input_ids[:2].to(resolved)
    with torch.no_grad(), _precision_context(resolved, str(config.precision)):
        reference_logits = cortex.model(
            parity_query, collect_telemetry=False
        )["logits"]
        off_logits = cortex.query_logits("off", None, parity_query)
    off_delta = float((reference_logits - off_logits).abs().max().float().cpu())
    if off_delta != 0.0:
        raise RuntimeError(f"V19 off path changed parent logits by {off_delta}")
    initial_embedding_norm = float(
        cortex.model.token_embedding(parity_query).detach().float().norm(dim=-1).mean().cpu()
    )
    initial_memory_norm = float(
        cortex.memory.memory_seed.detach().float().norm(dim=-1).mean().cpu()
    )
    print("[joint-memory-v19] evaluating untouched general baseline", flush=True)
    general_baseline = evaluate_general_language(
        cortex.model, prepared_general.eval_batches
    )

    def evaluate_active(mode: str) -> dict[str, Any]:
        candidate = evaluate_candidate_ranking(
            mode,
            cortex=cortex,
            case_bank=case_bank,
            candidate_bank=candidate_bank,
            candidate_count=int(candidate_count),
            cases=cases,
            group_indices=eval_groups,
            config=config,
        )
        free = evaluate_free_generation(
            mode,
            cortex=cortex,
            tokenizer=tokenizer,
            case_bank=case_bank,
            cases=cases,
            group_indices=eval_groups,
            config=config,
        )
        counterfactual_rows = free.pop("_counterfactual_rows")
        return {
            **candidate,
            **free,
            "paired_counterfactual": counterfactual_behavior_metrics(
                counterfactual_rows
            ),
            "case_count": len(cases),
            "source_facts_per_query": int(config.facts_per_example),
            "source_segments": int(config.source_segments),
            "correct_index_metrics_only": True,
            "write_path_uses_question": False,
            "write_path_uses_answer": False,
            "write_path_uses_candidates": False,
            "write_path_uses_correct_index": False,
        }

    rows: dict[str, dict[str, Any]] = {}
    for mode in ARM_NAMES:
        training, general_after = _run_training_arm(
            mode,
            cortex=cortex,
            initial_model_state=initial_model_state,
            initial_memory_state=initial_memory_state,
            relation_bank=relation_bank,
            relation_indices=relation_indices,
            target_slots=target_slots,
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
            "memory": memory_diagnostic(
                mode,
                cortex=cortex,
                case_bank=case_bank,
                group_indices=eval_groups,
                config=config,
            ),
        }
        print(
            f"[joint-memory-v19] {mode} candidate="
            f"{evaluation['candidate_accuracy']:.3f} free="
            f"{evaluation['free_exact_accuracy']:.3f} paired="
            f"{evaluation['paired_counterfactual']['source_following_exact_accuracy']:.3f} "
            f"general_delta={general_after['aggregate_heldout_loss_delta']:+.4f}",
            flush=True,
        )
    decision = joint_memory_token_decision(
        rows, train_steps=int(config.train_steps), config=config
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
            "model_state_sha256": _state_dict_sha256(initial_model_state),
        },
        "architecture": {
            "language_cortex": cortex.model.surface,
            "memory_owner": "same_jointly_trained_v11_cortex",
            "memory_slot_count": int(config.slot_count),
            "memory_width": int(model.hashed_config.width),
            "bounded_state_bytes_per_stream_float32": (
                cortex.memory.state_bytes_per_stream()
            ),
            "source_segment_sequence_length": segment_length,
            "memory_query_sequence_length": memory_query_length,
            "exact_control_sequence_length": exact_length,
            "memory_reentry": "layer_norm_times_learned_scalar",
            "memory_slot_routes": "stable_replay_id_plus_slot_index",
            "cross_segment_gradient_path": "ordinary_autograd_through_memory_tokens",
            "separate_memory_reader": False,
            "separate_miniature_language_model": False,
            "ordinary_next_token_lm_head": True,
            "memory_parameters": sum(
                int(value.numel()) for value in cortex.memory.parameters()
            ),
            "base_parameters": sum(
                int(value.numel()) for value in cortex.model.parameters()
            ),
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
                "correct_index_metrics_only": True,
            },
            "general": prepared_general.source_reports,
        },
        "schedule": {
            "identical_for_every_arm": True,
            "schedule_sha256": schedule_sha256(schedule),
            "relation_episode_sha256": _tensor_sha256(
                relation_indices, target_slots
            ),
            "evaluation_group_sha256": _tensor_sha256(
                eval_groups, eval_target_slots
            ),
            "evaluation_pairs_hold_query_and_distractors_fixed": True,
            "steps": int(config.train_steps),
            "relation_steps": relation_steps,
            "general_steps": int(config.train_steps) - relation_steps,
            "batch_size": int(config.batch_size),
        },
        "initialization": {
            "model_state_exact_reset_every_arm": True,
            "memory_state_exact_reset_every_arm": True,
            "memory_state_sha256": _state_dict_sha256(initial_memory_state),
            "off_path_maximum_absolute_logit_delta": off_delta,
            "off_path_exact_parent_logits": off_delta == 0.0,
            "mean_token_embedding_norm": initial_embedding_norm,
            "mean_memory_seed_norm": initial_memory_norm,
            "initial_memory_reentry_scale": float(config.initial_memory_scale),
        },
        "anti_cheat": {
            "write_path_input": "source_token_segments_only",
            "question_visible_to_write": False,
            "answer_visible_to_write": False,
            "candidates_visible_to_write": False,
            "correct_index_visible_to_prediction": False,
            "correct_index_metrics_only": True,
            "teacher_forcing_visible_only_to_ordinary_prior_answer_positions": True,
            "paired_source_swap_holds_question_distractors_and_positions_fixed": True,
        },
        "general_language_before": general_baseline,
        "arms": rows,
        "decision": decision,
        "checkpoint": None,
        "promotion_boundary": {
            "advance_to_contiguous_language_screen": decision == ADVANCE_DECISION,
            "base_quality_promoted": False,
            "runtime_install_allowed": False,
            "continual_learning_claimed": False,
            "exact_history_control_promotable": False,
        },
        "research_basis": [
            "https://arxiv.org/abs/2207.06881",
            "https://arxiv.org/abs/2203.07852",
            "https://arxiv.org/abs/2501.00663",
            "https://arxiv.org/abs/2604.06169",
            "https://arxiv.org/abs/2605.22791",
        ],
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
        title="MARULHO V19 Joint Memory Token Preflight",
    )
    print(f"[joint-memory-v19] decision {decision}", flush=True)
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
    parser.add_argument("--slot-count", type=int, default=16)
    parser.add_argument("--generation-max-tokens", type=int, default=16)
    parser.add_argument("--general-eval-batches", type=int, default=8)
    parser.add_argument(
        "--general-train-sample-bytes", type=int, default=8 * 1024 * 1024
    )
    parser.add_argument(
        "--general-eval-sample-bytes", type=int, default=8 * 1024 * 1024
    )
    parser.add_argument("--data-seed", type=int, default=9301)
    parser.add_argument("--model-seed", type=int, default=9401)
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = JointMemoryTokenConfig(
        train_record_count=int(args.train_records),
        train_steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        slot_count=int(args.slot_count),
        generation_max_tokens=int(args.generation_max_tokens),
        general_eval_batches=int(args.general_eval_batches),
        general_train_sample_bytes=int(args.general_train_sample_bytes),
        general_eval_sample_bytes=int(args.general_eval_sample_bytes),
        data_seed=int(args.data_seed),
        model_seed=int(args.model_seed),
        precision=str(args.precision),
    )
    run_joint_memory_token_preflight(
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
