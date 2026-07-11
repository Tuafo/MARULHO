"""Test bounded inter-segment language state before another base-model change."""

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
from marulho.evaluation.language_matched_support import sha256_file
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_hashed_micro_experts import (
    MarulhoHashedMicroExpertLanguageModel,
    load_hashed_micro_expert_checkpoint,
)


SURFACE = "marulho_segment_memory_preflight.v2"
ARTIFACT_KIND = "marulho_segment_memory_preflight"
TRAINED_ARM_NAMES = ("exact", "local", "recency", "mean", "learned")
ARM_NAMES = ("off", *TRAINED_ARM_NAMES)
MINIMUM_DECISION_STEPS = 512
ADVANCE_DECISION = "advance_v18b_segment_memory_to_contiguous_language_screen"


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
class SegmentMemoryConfig:
    width: int = 512
    slot_count: int = 16
    attention_heads: int = 8
    mlp_ratio: float = 2.0


@dataclass(frozen=True)
class SegmentMemoryPreflightConfig:
    train_record_count: int = 8192
    facts_per_example: int = 8
    source_segments: int = 2
    source_length: int = 64
    query_length: int = 40
    train_steps: int = 800
    batch_size: int = 64
    eval_batch_size: int = 32
    feature_batch_size: int = 96
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    data_seed: int = 9101
    model_seed: int = 9201
    slot_count: int = 16
    minimum_exact_candidate_accuracy: float = 0.70
    minimum_exact_free_accuracy: float = 0.20
    minimum_learned_candidate_accuracy: float = 0.65
    minimum_learned_free_accuracy: float = 0.20
    minimum_learned_candidate_gain: float = 0.08
    minimum_learned_free_gain: float = 0.08
    minimum_exact_counterfactual_gain: float = 0.10
    minimum_learned_counterfactual_accuracy: float = 0.20
    minimum_learned_counterfactual_gain: float = 0.08
    generation_max_tokens: int = 16


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
class CachedFeatureBank:
    source_hidden: torch.Tensor
    source_mask: torch.Tensor
    answer_hidden: torch.Tensor
    answer_targets: torch.Tensor
    answer_mask: torch.Tensor

    @property
    def record_count(self) -> int:
        return int(self.source_hidden.shape[0])

    @property
    def storage_bytes(self) -> int:
        tensors = (
            self.source_hidden,
            self.source_mask,
            self.answer_hidden,
            self.answer_targets,
            self.answer_mask,
        )
        return sum(int(value.numel() * value.element_size()) for value in tensors)

    def to(self, device: torch.device) -> CachedFeatureBank:
        return CachedFeatureBank(
            source_hidden=self.source_hidden.to(device),
            source_mask=self.source_mask.to(device),
            answer_hidden=self.answer_hidden.to(device),
            answer_targets=self.answer_targets.to(device),
            answer_mask=self.answer_mask.to(device),
        )


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
            f"relation corpus has only {len(reservoir)} records; "
            f"requested {count}"
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
        raise ValueError(
            f"source token length {len(source)} exceeds {source_length}"
        )
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


@torch.no_grad()
def cache_record_features(
    model: MarulhoHashedMicroExpertLanguageModel,
    encoded: EncodedRecordBank,
    *,
    batch_size: int,
    precision: str,
) -> CachedFeatureBank:
    device = model.device
    source_rows = []
    answer_rows = []
    answer_targets = []
    answer_masks = []
    maximum_answers = int(encoded.query_loss_mask.sum(dim=1).max())
    was_training = model.training
    model.eval()
    try:
        for start in range(0, encoded.record_count, int(batch_size)):
            end = min(encoded.record_count, start + int(batch_size))
            source_ids = encoded.source_ids[start:end].to(device)
            query_ids = encoded.query_input_ids[start:end].to(device)
            with _precision_context(device, precision):
                source_hidden = model._forward_hidden(
                    source_ids, collect_telemetry=False
                )["hidden"]
                query_hidden = model._forward_hidden(
                    query_ids, collect_telemetry=False
                )["hidden"]
            source_rows.append(source_hidden.detach().to("cpu", torch.bfloat16))
            batch_count = end - start
            selected = torch.zeros(
                batch_count,
                maximum_answers,
                int(query_hidden.shape[-1]),
                dtype=torch.bfloat16,
            )
            targets = torch.full(
                (batch_count, maximum_answers), -100, dtype=torch.long
            )
            masks = torch.zeros(
                (batch_count, maximum_answers), dtype=torch.bool
            )
            for row in range(batch_count):
                mask = encoded.query_loss_mask[start + row]
                count = int(mask.sum())
                selected[row, :count] = query_hidden[row, mask.to(device)].to(
                    "cpu", torch.bfloat16
                )
                targets[row, :count] = encoded.query_target_ids[
                    start + row, mask
                ]
                masks[row, :count] = True
            answer_rows.append(selected)
            answer_targets.append(targets)
            answer_masks.append(masks)
    finally:
        model.train(was_training)
    return CachedFeatureBank(
        source_hidden=torch.cat(source_rows, dim=0),
        source_mask=encoded.source_mask.clone(),
        answer_hidden=torch.cat(answer_rows, dim=0),
        answer_targets=torch.cat(answer_targets, dim=0),
        answer_mask=torch.cat(answer_masks, dim=0),
    )


class SegmentMemoryBridge(nn.Module):
    """Bounded latent state with explicit source-only writes and query reads."""

    def __init__(self, config: SegmentMemoryConfig) -> None:
        super().__init__()
        if int(config.width) < 1 or int(config.slot_count) < 1:
            raise ValueError("segment memory width and slot_count must be positive")
        if int(config.width) % int(config.attention_heads) != 0:
            raise ValueError("segment memory heads must divide width")
        self.config = config
        width = int(config.width)
        hidden = max(width, int(round(width * float(config.mlp_ratio))))
        self.memory_seed = nn.Parameter(torch.empty(int(config.slot_count), width))
        self.local_memory = nn.Parameter(torch.empty(int(config.slot_count), width))
        self.write_query_norm = nn.LayerNorm(width)
        self.write_source_norm = nn.LayerNorm(width)
        self.write_attention = nn.MultiheadAttention(
            width, int(config.attention_heads), batch_first=True
        )
        self.write_gate = nn.Linear(2 * width, width)
        self.write_post_norm = nn.LayerNorm(width)
        self.write_mlp = nn.Sequential(
            nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, width)
        )
        self.write_state_norm = nn.LayerNorm(width)
        self.read_query_norm = nn.LayerNorm(width)
        self.read_memory_norm = nn.LayerNorm(width)
        self.read_attention = nn.MultiheadAttention(
            width, int(config.attention_heads), batch_first=True
        )
        self.read_output = nn.Linear(width, width, bias=False)
        nn.init.normal_(self.memory_seed, mean=0.0, std=0.02)
        nn.init.normal_(self.local_memory, mean=0.0, std=0.02)
        nn.init.zeros_(self.read_output.weight)

    @property
    def slot_count(self) -> int:
        return int(self.config.slot_count)

    def _learned_update(
        self,
        memory: torch.Tensor,
        source: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        proposal, _weights = self.write_attention(
            self.write_query_norm(memory),
            self.write_source_norm(source),
            self.write_source_norm(source),
            key_padding_mask=~source_mask,
            need_weights=False,
        )
        gate = torch.sigmoid(self.write_gate(torch.cat((memory, proposal), dim=-1)))
        updated = memory + gate * proposal
        return self.write_state_norm(
            updated + self.write_mlp(self.write_post_norm(updated))
        )

    def _pool_slots(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        rows = []
        for hidden, mask in zip(source, source_mask):
            valid = hidden[mask]
            pooled = F.adaptive_avg_pool1d(
                valid.transpose(0, 1).unsqueeze(0), self.slot_count
            ).squeeze(0).transpose(0, 1)
            rows.append(pooled)
        return torch.stack(rows, dim=0)

    def _recency_slots(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        rows = []
        for hidden, mask in zip(source, source_mask):
            valid = hidden[mask]
            if int(valid.shape[0]) >= self.slot_count:
                rows.append(valid[-self.slot_count :])
            else:
                pad = valid.new_zeros(self.slot_count - int(valid.shape[0]), valid.shape[1])
                rows.append(torch.cat((pad, valid), dim=0))
        return torch.stack(rows, dim=0)

    def build_memory(
        self,
        mode: str,
        grouped_source: torch.Tensor,
        grouped_mask: torch.Tensor,
        *,
        source_segments: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if mode == "off":
            return None, None
        if grouped_source.ndim != 4 or grouped_mask.shape != grouped_source.shape[:3]:
            raise ValueError("grouped source must be [batch,facts,time,width]")
        batch, facts, time_steps, width = grouped_source.shape
        if int(facts) % int(source_segments) != 0:
            raise ValueError("facts must divide evenly across source segments")
        if mode == "exact":
            return (
                grouped_source.reshape(int(batch), int(facts) * int(time_steps), int(width)),
                grouped_mask.reshape(int(batch), int(facts) * int(time_steps)),
            )
        if mode == "local":
            memory = self.local_memory.unsqueeze(0).expand(int(batch), -1, -1)
            return memory, torch.ones(
                int(batch), self.slot_count, device=memory.device, dtype=torch.bool
            )
        facts_per_segment = int(facts) // int(source_segments)
        segmented_source = grouped_source.reshape(
            int(batch), int(source_segments), facts_per_segment * int(time_steps), int(width)
        )
        segmented_mask = grouped_mask.reshape(
            int(batch), int(source_segments), facts_per_segment * int(time_steps)
        )
        if mode == "learned":
            memory = self.memory_seed.unsqueeze(0).expand(int(batch), -1, -1)
            for segment in range(int(source_segments)):
                memory = self._learned_update(
                    memory,
                    segmented_source[:, segment],
                    segmented_mask[:, segment],
                )
        elif mode == "mean":
            memory = None
            for segment in range(int(source_segments)):
                pooled = self._pool_slots(
                    segmented_source[:, segment], segmented_mask[:, segment]
                )
                memory = pooled if memory is None else 0.5 * (memory + pooled)
            assert memory is not None
        elif mode == "recency":
            memory = self._recency_slots(
                segmented_source[:, -1], segmented_mask[:, -1]
            )
        else:
            raise ValueError(f"unknown segment memory mode: {mode}")
        return memory, torch.ones(
            int(batch), self.slot_count, device=memory.device, dtype=torch.bool
        )

    def read_query(
        self,
        mode: str,
        query_hidden: torch.Tensor,
        memory: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mode == "off":
            return query_hidden
        if memory is None or memory_mask is None:
            raise ValueError("active segment memory mode requires memory tensors")
        observed, _weights = self.read_attention(
            self.read_query_norm(query_hidden),
            self.read_memory_norm(memory),
            self.read_memory_norm(memory),
            key_padding_mask=~memory_mask,
            need_weights=False,
        )
        return query_hidden + self.read_output(observed)

    def forward(
        self,
        mode: str,
        query_hidden: torch.Tensor,
        grouped_source: torch.Tensor,
        grouped_mask: torch.Tensor,
        *,
        source_segments: int,
    ) -> torch.Tensor:
        memory, memory_mask = self.build_memory(
            mode,
            grouped_source,
            grouped_mask,
            source_segments=int(source_segments),
        )
        return self.read_query(mode, query_hidden, memory, memory_mask)

    def bounded_state_bytes(self, batch_size: int, *, element_size: int = 4) -> int:
        return (
            int(batch_size)
            * self.slot_count
            * int(self.config.width)
            * int(element_size)
        )


def build_group_schedule(
    *,
    record_count: int,
    steps: int,
    batch_size: int,
    facts_per_example: int,
    seed: int,
    record_labels: Sequence[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(record_count) < int(facts_per_example):
        raise ValueError("record bank is smaller than one fact group")
    if record_labels is not None:
        if len(record_labels) != int(record_count):
            raise ValueError("record_labels length must equal record_count")
        buckets: dict[str, list[int]] = {}
        for index, label in enumerate(record_labels):
            buckets.setdefault(str(label), []).append(index)
        labels = sorted(buckets)
        if len(labels) < int(facts_per_example):
            raise ValueError("not enough distinct record labels for one fact group")
        generator_py = random.Random(int(seed))
        rows = []
        targets = []
        for _step in range(int(steps)):
            batch_rows = []
            batch_targets = []
            for _row in range(int(batch_size)):
                chosen_labels = generator_py.sample(labels, int(facts_per_example))
                batch_rows.append(
                    [generator_py.choice(buckets[label]) for label in chosen_labels]
                )
                batch_targets.append(generator_py.randrange(int(facts_per_example)))
            rows.append(batch_rows)
            targets.append(batch_targets)
        return torch.tensor(rows, dtype=torch.long), torch.tensor(
            targets, dtype=torch.long
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    starts = torch.randint(
        0, int(record_count), (int(steps), int(batch_size), 1), generator=generator
    )
    strides = 2 * torch.randint(
        0,
        max(1, int(record_count) // 2),
        (int(steps), int(batch_size), 1),
        generator=generator,
    ) + 1
    offsets = torch.arange(int(facts_per_example)).reshape(1, 1, -1)
    indices = (starts + strides * offsets) % int(record_count)
    for column in range(1, int(facts_per_example)):
        collision = (indices[..., column : column + 1] == indices[..., :column]).any(
            dim=-1, keepdim=True
        )
        while bool(collision.any()):
            indices[..., column : column + 1] = torch.where(
                collision,
                (indices[..., column : column + 1] + 1) % int(record_count),
                indices[..., column : column + 1],
            )
            collision = (
                indices[..., column : column + 1] == indices[..., :column]
            ).any(dim=-1, keepdim=True)
    target_slots = torch.randint(
        0,
        int(facts_per_example),
        (int(steps), int(batch_size)),
        generator=generator,
    )
    return indices.long(), target_slots.long()


def segment_memory_decision(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    train_steps: int,
    config: SegmentMemoryPreflightConfig,
) -> str:
    if set(rows) != set(ARM_NAMES):
        return "incomplete_v18b_missing_control_arm"
    if int(train_steps) < MINIMUM_DECISION_STEPS:
        return "diagnostic_v18b_below_preflight_step_floor"
    candidate = {
        name: float(rows[name]["evaluation"]["candidate_accuracy"])
        for name in ARM_NAMES
    }
    free = {
        name: float(rows[name]["evaluation"]["free_exact_accuracy"])
        for name in ARM_NAMES
    }
    counterfactual = {
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
        or counterfactual["exact"] - counterfactual["local"]
        < float(config.minimum_exact_counterfactual_gain)
    ):
        return "retire_v18b_exact_history_no_source_causal_gain"
    simple = ("local", "recency", "mean")
    candidate_control = max(candidate[name] for name in simple)
    free_control = max(free[name] for name in simple)
    counterfactual_control = max(counterfactual[name] for name in simple)
    learned_pass = (
        candidate["learned"] >= float(config.minimum_learned_candidate_accuracy)
        and free["learned"] >= float(config.minimum_learned_free_accuracy)
        and candidate["learned"] - candidate_control
        >= float(config.minimum_learned_candidate_gain)
        and free["learned"] - free_control
        >= float(config.minimum_learned_free_gain)
        and counterfactual["learned"]
        >= float(config.minimum_learned_counterfactual_accuracy)
        and counterfactual["learned"] - counterfactual_control
        >= float(config.minimum_learned_counterfactual_gain)
    )
    if learned_pass:
        return ADVANCE_DECISION
    if max(candidate["recency"], candidate["mean"]) >= candidate["learned"]:
        return "retire_v18b_simple_summary_matches_learned_slots"
    return "retire_v18b_compression_gap_exact_bridge_viable"


def _tensor_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().contiguous().cpu()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _grouped_source_batch(
    bank: CachedFeatureBank,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, facts = indices.shape
    flat = indices.reshape(-1).to(bank.source_hidden.device)
    hidden = bank.source_hidden.index_select(0, flat).reshape(
        int(batch),
        int(facts),
        int(bank.source_hidden.shape[1]),
        int(bank.source_hidden.shape[2]),
    )
    mask = bank.source_mask.index_select(0, flat).reshape(
        int(batch), int(facts), int(bank.source_mask.shape[1])
    )
    return hidden, mask


def _target_answer_batch(
    bank: CachedFeatureBank,
    indices: torch.Tensor,
    target_slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = indices.gather(1, target_slots.unsqueeze(1)).squeeze(1)
    selected = selected.to(bank.answer_hidden.device)
    return (
        bank.answer_hidden.index_select(0, selected),
        bank.answer_targets.index_select(0, selected),
        bank.answer_mask.index_select(0, selected),
    )


def _bridge_gradient_report(bridge: SegmentMemoryBridge) -> dict[str, Any]:
    rows = []
    for name, parameter in bridge.named_parameters():
        gradient = parameter.grad
        rows.append(
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
    learned_core = [row for row in rows if row["name"] != "local_memory"]
    return {
        "parameters": rows,
        "learned_core_all_received_gradient": all(
            bool(row["received_gradient"]) for row in learned_core
        ),
        "learned_core_all_nonzero": all(
            int(row["nonzero_gradient_elements"]) > 0 for row in learned_core
        ),
    }


def _run_training_arm(
    mode: str,
    *,
    bridge: SegmentMemoryBridge,
    initial_state: Mapping[str, torch.Tensor],
    model: MarulhoHashedMicroExpertLanguageModel,
    bank: CachedFeatureBank,
    schedule_indices: torch.Tensor,
    target_slots: torch.Tensor,
    config: SegmentMemoryPreflightConfig,
    device: torch.device,
) -> dict[str, Any]:
    bridge.load_state_dict(dict(initial_state), strict=True)
    bridge.train()
    torch.manual_seed(int(config.model_seed))
    optimizer = torch.optim.AdamW(
        bridge.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        fused=bool(device.type == "cuda"),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    supervised_tokens = 0
    source_tokens = 0
    trace = []
    for step in range(int(config.train_steps)):
        indices = schedule_indices[step]
        slots = target_slots[step]
        grouped_source, grouped_mask = _grouped_source_batch(bank, indices)
        query_hidden, targets, answer_mask = _target_answer_batch(
            bank, indices, slots
        )
        optimizer.zero_grad(set_to_none=True)
        with _precision_context(device, str(config.precision)):
            adapted = bridge(
                mode,
                query_hidden,
                grouped_source,
                grouped_mask,
                source_segments=int(config.source_segments),
            )
            logits = model.lm_head(adapted)
            loss = F.cross_entropy(logits[answer_mask], targets[answer_mask])
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            bridge.parameters(), float(config.gradient_clip)
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"non-finite V18 gradient in {mode}")
        optimizer.step()
        supervised_tokens += int(answer_mask.sum())
        source_tokens += int(grouped_mask.sum())
        interval = max(1, int(config.train_steps) // 10)
        if (step + 1) % interval == 0 or step + 1 == int(config.train_steps):
            trace.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach().float().cpu()),
                    "supervised_tokens": supervised_tokens,
                }
            )
            print(
                f"[segment-memory-v18] {mode} {step + 1}/{config.train_steps}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "optimizer": "AdamW",
        "optimizer_state_fresh": True,
        "initial_bridge_state_restored": True,
        "steps": int(config.train_steps),
        "supervised_answer_tokens": supervised_tokens,
        "observed_source_tokens": source_tokens,
        "effective_tokens_per_second": (
            (supervised_tokens + source_tokens) / max(elapsed, 1.0e-12)
        ),
        "elapsed_seconds": elapsed,
        "final_loss": trace[-1]["loss"],
        "loss_trace": trace,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "gradient": _bridge_gradient_report(bridge),
    }


def build_evaluation_groups(
    *,
    case_count: int,
    facts_per_example: int,
    seed: int,
    case_labels: Sequence[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(case_count) < int(facts_per_example):
        raise ValueError("not enough cases for an evaluation fact group")
    generator = random.Random(int(seed))
    rows = []
    target_slots = []
    population = list(range(int(case_count)))
    label_buckets: dict[str, list[int]] | None = None
    if case_labels is not None:
        if len(case_labels) != int(case_count):
            raise ValueError("case_labels length must equal case_count")
        label_buckets = {}
        for index, label in enumerate(case_labels):
            label_buckets.setdefault(str(label), []).append(index)
        if len(label_buckets) < int(facts_per_example):
            raise ValueError("not enough distinct case labels for one fact group")
    label_plans: dict[str, tuple[list[int], int]] = {}
    if label_buckets is not None:
        for target_label in sorted(label_buckets):
            distractor_labels = generator.sample(
                [label for label in label_buckets if label != target_label],
                int(facts_per_example) - 1,
            )
            distractors = [
                generator.choice(label_buckets[label])
                for label in distractor_labels
            ]
            target_slot = generator.randrange(int(facts_per_example))
            label_plans[target_label] = (distractors, target_slot)
    for target in range(int(case_count)):
        if label_buckets is None:
            eligible = [index for index in population if index != target]
            distractors = generator.sample(
                eligible, int(facts_per_example) - 1
            )
        else:
            target_label = str(case_labels[target])
            distractors, target_slot = label_plans[target_label]
            row = list(distractors)
            row.insert(int(target_slot), target)
            rows.append(row)
            target_slots.append(int(target_slot))
            continue
        row = [target, *distractors]
        generator.shuffle(row)
        rows.append(row)
        target_slots.append(row.index(target))
    return torch.tensor(rows, dtype=torch.long), torch.tensor(
        target_slots, dtype=torch.long
    )


def _candidate_feature_bank(
    model: MarulhoHashedMicroExpertLanguageModel,
    tokenizer,
    cases: Sequence[RelationMemoryCase],
    *,
    config: SegmentMemoryPreflightConfig,
) -> tuple[CachedFeatureBank, int]:
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
    encoded = encode_relation_records(
        tokenizer,
        records,
        source_length=int(config.source_length),
        query_length=int(config.query_length),
    )
    return (
        cache_record_features(
            model,
            encoded,
            batch_size=int(config.feature_batch_size),
            precision=str(config.precision),
        ),
        candidate_count,
    )


@torch.no_grad()
def evaluate_candidate_ranking(
    mode: str,
    *,
    bridge: SegmentMemoryBridge,
    model: MarulhoHashedMicroExpertLanguageModel,
    case_bank: CachedFeatureBank,
    candidate_bank: CachedFeatureBank,
    candidate_count: int,
    cases: Sequence[RelationMemoryCase],
    group_indices: torch.Tensor,
    config: SegmentMemoryPreflightConfig,
    device: torch.device,
) -> dict[str, Any]:
    bridge.eval()
    predictions = []
    all_scores = []
    for start in range(0, len(cases), int(config.eval_batch_size)):
        end = min(len(cases), start + int(config.eval_batch_size))
        indices = group_indices[start:end]
        grouped_source, grouped_mask = _grouped_source_batch(case_bank, indices)
        with _precision_context(device, str(config.precision)):
            memory, memory_mask = bridge.build_memory(
                mode,
                grouped_source,
                grouped_mask,
                source_segments=int(config.source_segments),
            )
        batch_scores = []
        for candidate in range(int(candidate_count)):
            flat_indices = torch.arange(start, end, device=device) * int(
                candidate_count
            ) + candidate
            query_hidden = candidate_bank.answer_hidden.index_select(
                0, flat_indices
            )
            targets = candidate_bank.answer_targets.index_select(0, flat_indices)
            mask = candidate_bank.answer_mask.index_select(0, flat_indices)
            with _precision_context(device, str(config.precision)):
                adapted = bridge.read_query(
                    mode, query_hidden, memory, memory_mask
                )
                logits = model.lm_head(adapted)
                token_loss = F.cross_entropy(
                    logits.reshape(-1, int(logits.shape[-1])),
                    targets.reshape(-1),
                    reduction="none",
                    ignore_index=-100,
                ).reshape(targets.shape)
                score = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            batch_scores.append(score.float())
        scores = torch.stack(batch_scores, dim=1)
        all_scores.append(scores.cpu())
        predictions.extend(scores.argmin(dim=1).cpu().tolist())
    correct = [int(case.correct_index) for case in cases]
    accuracy = sum(
        int(prediction == expected)
        for prediction, expected in zip(predictions, correct)
    ) / len(cases)
    kind_rows: dict[str, list[bool]] = {}
    for case, prediction in zip(cases, predictions):
        kind_rows.setdefault(case.kind, []).append(
            prediction == int(case.correct_index)
        )
    return {
        "candidate_accuracy": accuracy,
        "candidate_kind_accuracy": {
            kind: sum(values) / len(values) for kind, values in kind_rows.items()
        },
        "prediction_uses_correct_index": False,
        "correct_index_metrics_only": True,
        "mean_candidate_scores": torch.cat(all_scores).mean(dim=0).tolist(),
    }


def _normalize_answer(value: str) -> str:
    lowered = re.sub(r"\s+", " ", str(value).strip().lower())
    return lowered.rstrip(" .!?;:")


@torch.no_grad()
def evaluate_free_generation(
    mode: str,
    *,
    bridge: SegmentMemoryBridge,
    model: MarulhoHashedMicroExpertLanguageModel,
    tokenizer,
    case_bank: CachedFeatureBank,
    cases: Sequence[RelationMemoryCase],
    group_indices: torch.Tensor,
    config: SegmentMemoryPreflightConfig,
    device: torch.device,
) -> dict[str, Any]:
    bridge.eval()
    exact_rows = []
    contains_rows = []
    examples = []
    counterfactual_rows = []
    for start in range(0, len(cases), int(config.eval_batch_size)):
        end = min(len(cases), start + int(config.eval_batch_size))
        selected_cases = cases[start:end]
        grouped_source, grouped_mask = _grouped_source_batch(
            case_bank, group_indices[start:end]
        )
        with _precision_context(device, str(config.precision)):
            memory, memory_mask = bridge.build_memory(
                mode,
                grouped_source,
                grouped_mask,
                source_segments=int(config.source_segments),
            )
        sequences = [
            tokenizer.encode(case.query_prefix, add_bos=True, add_eos=False)
            for case in selected_cases
        ]
        generated = [[] for _case in selected_cases]
        finished = [False for _case in selected_cases]
        for _step in range(int(config.generation_max_tokens)):
            maximum = max(len(sequence) for sequence in sequences)
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
                hidden = model._forward_hidden(
                    input_ids, collect_telemetry=False
                )["hidden"]
                rows = torch.arange(len(sequences), device=device)
                positions = torch.tensor(lengths, device=device) - 1
                last = hidden[rows, positions].unsqueeze(1)
                adapted = bridge.read_query(mode, last, memory, memory_mask)
                logits = model.lm_head(adapted[:, 0])
            next_ids = logits.argmax(dim=-1).cpu().tolist()
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
                if str(left["expected"]) == str(right["expected"]):
                    continue
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


@torch.no_grad()
def _memory_diagnostic(
    mode: str,
    *,
    bridge: SegmentMemoryBridge,
    case_bank: CachedFeatureBank,
    group_indices: torch.Tensor,
    config: SegmentMemoryPreflightConfig,
) -> dict[str, Any]:
    indices = group_indices[: min(32, int(group_indices.shape[0]))]
    grouped_source, grouped_mask = _grouped_source_batch(case_bank, indices)
    device = grouped_source.device
    with _precision_context(device, str(config.precision)):
        memory, memory_mask = bridge.build_memory(
            mode,
            grouped_source,
            grouped_mask,
            source_segments=int(config.source_segments),
        )
    if memory is None or memory_mask is None:
        return {
            "state_elements_per_stream": 0,
            "state_bytes_per_stream_float32": 0,
            "matrix_rank": 0,
            "effective_rank": 0.0,
            "promotion_metric": False,
        }
    matrix = memory.detach().float().reshape(-1, int(memory.shape[-1])).cpu()
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    variance = singular.square()
    probability = variance / variance.sum().clamp_min(1.0e-12)
    effective = torch.exp(
        -(probability * probability.clamp_min(1.0e-12).log()).sum()
    )
    state_elements = int(memory.shape[1] * memory.shape[2])
    return {
        "state_elements_per_stream": state_elements,
        "state_bytes_per_stream_float32": state_elements * 4,
        "valid_memory_positions_per_stream": float(memory_mask.sum(dim=1).float().mean().cpu()),
        "matrix_rank": int(torch.linalg.matrix_rank(centered)),
        "effective_rank": float(effective),
        "mean_state_norm": float(memory.norm(dim=-1).mean().float().cpu()),
        "promotion_metric": False,
    }


def run_segment_memory_preflight(
    *,
    parent_checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    output_path: str | Path,
    config: SegmentMemoryPreflightConfig = SegmentMemoryPreflightConfig(),
    device: str = "auto",
) -> dict[str, Any]:
    if int(config.facts_per_example) % int(config.source_segments) != 0:
        raise ValueError("facts_per_example must divide across source_segments")
    if int(config.train_steps) < 1 or int(config.batch_size) < 1:
        raise ValueError("V18 train_steps and batch_size must be positive")
    if device == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for V18 but unavailable")
    started = time.perf_counter()
    parent_path = Path(parent_checkpoint_path)
    relation_path = Path(relation_corpus_path)
    cases_path = Path(relation_cases_path)
    model, tokenizer, parent_metadata = load_hashed_micro_expert_checkpoint(
        parent_path, map_location="cpu"
    )
    _validate_parent(model, parent_metadata)
    parent_tokens = int(parent_metadata.get("processed_tokens") or 0)
    if parent_tokens < 1_000_000_000:
        raise ValueError("V18 requires the one-billion-token V11 parent")
    if parent_metadata.get("external_llm_used") is not False:
        raise ValueError("V18 parent must be MARULHO-owned")
    if int(config.source_length) > int(model.hashed_config.context_length):
        raise ValueError("V18 source record exceeds parent context")
    if int(config.query_length) > int(model.hashed_config.context_length):
        raise ValueError("V18 query exceeds parent context")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = model.to(resolved).eval()
    print("[segment-memory-v18] sampling relation records", flush=True)
    train_records = sample_relation_training_records(
        relation_path,
        count=int(config.train_record_count),
        seed=int(config.data_seed),
    )
    cases = load_relation_memory_cases(cases_path)
    train_encoded = encode_relation_records(
        tokenizer,
        train_records,
        source_length=int(config.source_length),
        query_length=int(config.query_length),
    )
    feature_started = time.perf_counter()
    print("[segment-memory-v18] caching frozen V11 train features", flush=True)
    train_bank_cpu = cache_record_features(
        model,
        train_encoded,
        batch_size=int(config.feature_batch_size),
        precision=str(config.precision),
    )
    case_records = [
        RelationMemoryRecord(
            source=case.source,
            query_prefix=case.query_prefix,
            answer=case.candidates[int(case.correct_index)],
        )
        for case in cases
    ]
    case_encoded = encode_relation_records(
        tokenizer,
        case_records,
        source_length=int(config.source_length),
        query_length=int(config.query_length),
    )
    case_bank_cpu = cache_record_features(
        model,
        case_encoded,
        batch_size=int(config.feature_batch_size),
        precision=str(config.precision),
    )
    candidate_bank_cpu, candidate_count = _candidate_feature_bank(
        model, tokenizer, cases, config=config
    )
    feature_seconds = time.perf_counter() - feature_started
    train_bank = train_bank_cpu.to(resolved)
    case_bank = case_bank_cpu.to(resolved)
    candidate_bank = candidate_bank_cpu.to(resolved)
    feature_storage_bytes = (
        train_bank.storage_bytes
        + case_bank.storage_bytes
        + candidate_bank.storage_bytes
    )
    del train_bank_cpu, case_bank_cpu, candidate_bank_cpu
    schedule_indices, target_slots = build_group_schedule(
        record_count=train_bank.record_count,
        steps=int(config.train_steps),
        batch_size=int(config.batch_size),
        facts_per_example=int(config.facts_per_example),
        seed=int(config.data_seed) + 1,
        record_labels=[record.query_prefix for record in train_records],
    )
    eval_groups, eval_target_slots = build_evaluation_groups(
        case_count=len(cases),
        facts_per_example=int(config.facts_per_example),
        seed=int(config.data_seed) + 2,
        case_labels=[case.query_prefix for case in cases],
    )
    if any(
        int(eval_groups[index, eval_target_slots[index]]) != index
        for index in range(len(cases))
    ):
        raise RuntimeError("V18 evaluation group lost its target source")
    memory_config = SegmentMemoryConfig(
        width=int(model.hashed_config.width),
        slot_count=int(config.slot_count),
        attention_heads=int(model.hashed_config.attention_heads),
    )
    torch.manual_seed(int(config.model_seed))
    bridge = SegmentMemoryBridge(memory_config).to(resolved)
    initial_state = {
        name: value.detach().clone() for name, value in bridge.state_dict().items()
    }
    parity_indices = eval_groups[:2]
    parity_source, parity_mask = _grouped_source_batch(case_bank, parity_indices)
    parity_query = candidate_bank.answer_hidden[:2]
    with torch.no_grad(), _precision_context(resolved, str(config.precision)):
        base_logits = model.lm_head(parity_query)
        attachment_parity = {}
        for mode in ARM_NAMES:
            observed = bridge(
                mode,
                parity_query,
                parity_source,
                parity_mask,
                source_segments=int(config.source_segments),
            )
            logits = model.lm_head(observed)
            maximum_delta = float((logits - base_logits).abs().max().float().cpu())
            attachment_parity[mode] = {
                "maximum_absolute_logit_delta": maximum_delta,
                "exact_parent_logits": bool(torch.equal(logits, base_logits)),
            }
            if maximum_delta != 0.0:
                raise RuntimeError(
                    f"V18 {mode} attachment changed initial logits: {maximum_delta}"
                )

    def evaluate_active(mode: str) -> dict[str, Any]:
        candidate = evaluate_candidate_ranking(
            mode,
            bridge=bridge,
            model=model,
            case_bank=case_bank,
            candidate_bank=candidate_bank,
            candidate_count=int(candidate_count),
            cases=cases,
            group_indices=eval_groups,
            config=config,
            device=resolved,
        )
        free = evaluate_free_generation(
            mode,
            bridge=bridge,
            model=model,
            tokenizer=tokenizer,
            case_bank=case_bank,
            cases=cases,
            group_indices=eval_groups,
            config=config,
            device=resolved,
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
    bridge.load_state_dict(initial_state, strict=True)
    rows["off"] = {
        "mode": "off",
        "training": None,
        "evaluation": evaluate_active("off"),
        "memory": _memory_diagnostic(
            "off",
            bridge=bridge,
            case_bank=case_bank,
            group_indices=eval_groups,
            config=config,
        ),
    }
    print(
        "[segment-memory-v18] off candidate="
        f"{rows['off']['evaluation']['candidate_accuracy']:.3f} free="
        f"{rows['off']['evaluation']['free_exact_accuracy']:.3f}",
        flush=True,
    )
    for mode in TRAINED_ARM_NAMES:
        training = _run_training_arm(
            mode,
            bridge=bridge,
            initial_state=initial_state,
            model=model,
            bank=train_bank,
            schedule_indices=schedule_indices,
            target_slots=target_slots,
            config=config,
            device=resolved,
        )
        evaluation = evaluate_active(mode)
        rows[mode] = {
            "mode": mode,
            "training": training,
            "evaluation": evaluation,
            "memory": _memory_diagnostic(
                mode,
                bridge=bridge,
                case_bank=case_bank,
                group_indices=eval_groups,
                config=config,
            ),
        }
        print(
            f"[segment-memory-v18] {mode} candidate="
            f"{evaluation['candidate_accuracy']:.3f} free="
            f"{evaluation['free_exact_accuracy']:.3f}",
            flush=True,
        )
    decision = segment_memory_decision(
        rows, train_steps=int(config.train_steps), config=config
    )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "configuration": asdict(config),
        "memory_configuration": asdict(memory_config),
        "parent": {
            "path": str(parent_path),
            "sha256": sha256_file(parent_path),
            "processed_tokens": parent_tokens,
            "decision": parent_metadata.get("decision"),
            "tokenizer_hash": tokenizer.vocabulary_hash(),
            "parameters_frozen": True,
            "parameter_gradients_enabled": False,
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
        },
        "schedule": {
            "identical_for_every_trained_arm": True,
            "sha256": _tensor_sha256(schedule_indices, target_slots),
            "evaluation_group_sha256": _tensor_sha256(
                eval_groups, eval_target_slots
            ),
            "evaluation_pairs_hold_query_and_distractors_fixed": True,
            "steps": int(config.train_steps),
            "batch_size": int(config.batch_size),
            "facts_per_example": int(config.facts_per_example),
            "source_segments": int(config.source_segments),
        },
        "feature_cache": {
            "frozen_parent_features": True,
            "transient_not_checkpointed": True,
            "storage_device": str(resolved),
            "storage_bytes": feature_storage_bytes,
            "construction_seconds": feature_seconds,
            "training_throughput_includes_feature_construction": False,
            "deployment_throughput_claimed": False,
        },
        "attachment_parity": attachment_parity,
        "anti_cheat": {
            "write_path_input": "source_segments_only",
            "question_visible_to_write": False,
            "answer_visible_to_write": False,
            "candidates_visible_to_write": False,
            "correct_index_visible_to_prediction": False,
            "correct_index_metrics_only": True,
            "teacher_forcing_visible_only_to_ordinary_prior_answer_positions": True,
        },
        "bridge_parameters": sum(int(value.numel()) for value in bridge.parameters()),
        "base_parameters": sum(int(value.numel()) for value in model.parameters()),
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
                torch.cuda.get_device_name(resolved) if resolved.type == "cuda" else None
            ),
            "torch_version": torch.__version__,
        },
        "experiment_wall_seconds": time.perf_counter() - started,
    }
    write_json_report_with_readme(
        output_path,
        report,
        title="MARULHO V18 Segment Memory Preflight",
    )
    print(f"[segment-memory-v18] decision {decision}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--relation-corpus", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-records", type=int, default=8192)
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--slot-count", type=int, default=16)
    parser.add_argument("--generation-max-tokens", type=int, default=16)
    parser.add_argument("--data-seed", type=int, default=9101)
    parser.add_argument("--model-seed", type=int, default=9201)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = SegmentMemoryPreflightConfig(
        train_record_count=int(args.train_records),
        train_steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        slot_count=int(args.slot_count),
        generation_max_tokens=int(args.generation_max_tokens),
        data_seed=int(args.data_seed),
        model_seed=int(args.model_seed),
    )
    run_segment_memory_preflight(
        parent_checkpoint_path=args.parent_checkpoint,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        output_path=args.output,
        config=config,
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
