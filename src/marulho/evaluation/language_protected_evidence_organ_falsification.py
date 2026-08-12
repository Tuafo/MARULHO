"""V58 protected bidirectional evidence-organ capacity falsifier."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

import torch
from torch import nn
import torch.nn.functional as F

from marulho.data.language_tokenizer import LanguageTokenizer
from marulho.evaluation.artifact_io import sha256_json
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    language_model_state_sha256,
    load_language_model_checkpoint,
)
from marulho.training.language_transformer import _apply_rotary


SURFACE = "marulho_protected_bidirectional_evidence_organ_falsification.v1"
ARTIFACT_KIND = "marulho_protected_bidirectional_evidence_organ_falsification"
ADVANCE_DECISION = "advance_v58_protected_evidence_organ_to_routing_and_compression"
RETIRE_DECISION = "retire_v58_extractive_evidence_organ_capacity_failure"
DEFAULT_CHECKPOINT = Path(
    "reports/language_scaling/v39-answer-objective-qualified-100m-218m-20260810.pt"
)
DEFAULT_TRAIN_MANIFEST = Path(
    "reports/language_curriculum/squad-v57-native-train-8192-20260812.json"
)
DEFAULT_VALIDATION_MANIFEST = Path(
    "reports/language_curriculum/squad-v57-native-validation-256-20260812.json"
)
DEFAULT_OUTPUT = Path(
    "reports/language_scaling/protected-evidence-organ-v58-41m-20260812.json"
)
DEFAULT_CANDIDATE = Path(
    "reports/language_scaling/v58-protected-evidence-organ-qualified-100m-20260812.pt"
)


@dataclass(frozen=True)
class V58Config:
    context_length: int = 320
    maximum_source_characters: int = 1408
    maximum_answer_characters: int = 96
    maximum_token_character_offset: int = 64
    character_feature_dim: int = 16
    batch_size: int = 32
    epochs: int = 8
    optimizer_steps: int = 2048
    padded_position_budget: int = 20_971_520
    learning_rate: float = 1.0e-4
    minimum_learning_rate_fraction: float = 0.1
    warmup_fraction: float = 0.05
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    minimum_exact_answers: int = 192
    minimum_source_gain: float = 0.70
    maximum_mismatched_answers: int = 8
    minimum_initialized_advantage: int = 16
    maximum_training_seconds: float = 1800.0
    data_seed: int = 58121
    model_seed: int = 58131
    precision: str = "bfloat16"
    execution_backend: str = "pytorch_eager"


@dataclass(frozen=True)
class PreparedEvidenceCases:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    character_token_indices: torch.Tensor
    character_offsets: torch.Tensor
    character_ids: torch.Tensor
    character_mask: torch.Tensor
    start_positions: torch.Tensor | None
    end_positions: torch.Tensor | None
    cases: tuple[dict[str, Any], ...]

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Expected JSON object at {path}")
    return dict(loaded)


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


def _resolved_answer_start(source: str, answer: str, *, stored_start: int) -> int:
    exact = [match.start() for match in re.finditer(re.escape(answer), source)]
    candidates = exact or [
        match.start()
        for match in re.finditer(re.escape(answer), source, flags=re.IGNORECASE)
    ]
    if not candidates:
        raise ValueError(f"Answer {answer!r} is not an exact source substring")
    return min(candidates, key=lambda value: abs(int(value) - int(stored_start)))


def _atomic_torch_save(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _case_text(source: str, question: str) -> str:
    return f"Context: {source}\nQuestion: {question}"


def _overlap_token_indices(
    offsets: Sequence[tuple[int, int]],
    *,
    character_start: int,
    character_end: int,
) -> list[int]:
    return [
        index
        for index, (start, end) in enumerate(offsets)
        if int(end) > int(character_start) and int(start) < int(character_end)
    ]


def prepare_evidence_cases(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: LanguageTokenizer,
    *,
    context_length: int,
    maximum_source_characters: int = 1408,
    maximum_answer_characters: int = 96,
    maximum_token_character_offset: int = 64,
    source_field: str = "source_text",
    require_gold: bool,
) -> PreparedEvidenceCases:
    encoded_rows: list[list[int]] = []
    attention_rows: list[list[bool]] = []
    source_rows: list[tuple[list[int], list[int], list[int], list[bool]]] = []
    starts: list[int] = []
    ends: list[int] = []
    evidence: list[dict[str, Any]] = []
    maximum = int(context_length)
    for raw in rows:
        row = dict(raw)
        source = str(row[source_field])
        question = str(row["question"])
        text = _case_text(source, question)
        token_ids, offsets = tokenizer.encode_with_offsets(
            text,
            add_bos=True,
            add_eos=False,
        )
        if len(token_ids) > maximum:
            raise ValueError(
                f"Case {row.get('case_id')} has {len(token_ids)} tokens, above {maximum}"
            )
        source_character_start = len("Context: ")
        source_character_end = source_character_start + len(source)
        source_indices = _overlap_token_indices(
            offsets,
            character_start=source_character_start,
            character_end=source_character_end,
        )
        if not source_indices:
            raise ValueError(f"Case {row.get('case_id')} has no encoded source tokens")
        valid = len(token_ids)
        encoded_rows.append(
            token_ids + [int(tokenizer.pad_id)] * (maximum - valid)
        )
        attention_rows.append([True] * valid + [False] * (maximum - valid))
        character_token_indices = [-1] * len(source)
        character_offsets = [0] * len(source)
        for token_index, (token_start, token_end) in enumerate(offsets):
            clipped_start = max(int(token_start), source_character_start)
            clipped_end = min(int(token_end), source_character_end)
            for absolute_character in range(clipped_start, clipped_end):
                relative_character = absolute_character - source_character_start
                if character_token_indices[relative_character] >= 0:
                    continue
                character_token_indices[relative_character] = int(token_index)
                character_offsets[relative_character] = absolute_character - int(token_start)
        if any(index < 0 for index in character_token_indices):
            raise ValueError(f"Case {row.get('case_id')} has unmapped source characters")
        if len(source) > int(maximum_source_characters):
            raise ValueError(
                f"Case {row.get('case_id')} has {len(source)} source characters, "
                f"above {int(maximum_source_characters)}"
            )
        if max(character_offsets, default=0) >= int(maximum_token_character_offset):
            raise ValueError(
                f"Case {row.get('case_id')} exceeds token character offset "
                f"{int(maximum_token_character_offset) - 1}"
            )
        character_padding = int(maximum_source_characters) - len(source)
        source_rows.append(
            (
                character_token_indices + [0] * character_padding,
                character_offsets + [0] * character_padding,
                [ord(character) % 256 for character in source]
                + [0] * character_padding,
                [True] * len(source) + [False] * character_padding,
            )
        )

        gold_start: int | None = None
        gold_end: int | None = None
        oracle_text: str | None = None
        if require_gold:
            answer = str(tuple(row["answers"])[0])
            stored_start = int(row["answer_source_character_start"])
            gold_start = _resolved_answer_start(
                source,
                answer,
                stored_start=stored_start,
            )
            gold_end = gold_start + len(answer) - 1
            if len(answer) > int(maximum_answer_characters):
                raise ValueError(
                    f"Case {row.get('case_id')} has {len(answer)} answer characters, "
                    f"above {int(maximum_answer_characters)}"
                )
            if not 0 <= gold_start <= gold_end < len(source):
                raise ValueError(f"Case {row.get('case_id')} gold character span leaves source")
            oracle_text = source[gold_start : gold_end + 1]
            accepted = {_normalized(value) for value in row["answers"]}
            if _normalized(oracle_text) not in accepted:
                raise ValueError(
                    f"Case {row.get('case_id')} oracle copy {oracle_text!r} is not accepted"
                )
            starts.append(gold_start)
            ends.append(gold_end)
        evidence.append(
            {
                "case_id": str(row["case_id"]),
                "answers": [str(value) for value in row["answers"]],
                "source_text": source,
                "question": question,
                "token_count": valid,
                "source_token_count": len(source_indices),
                "source_character_count": len(source),
                "gold_start": gold_start,
                "gold_end": gold_end,
                "stored_answer_start": (
                    int(row["answer_source_character_start"])
                    if require_gold
                    else None
                ),
                "answer_start_corrected": bool(
                    require_gold
                    and gold_start != int(row["answer_source_character_start"])
                ),
                "oracle_text": oracle_text,
            }
        )
    return PreparedEvidenceCases(
        input_ids=torch.tensor(encoded_rows, dtype=torch.long),
        attention_mask=torch.tensor(attention_rows, dtype=torch.bool),
        character_token_indices=torch.tensor(
            [row[0] for row in source_rows], dtype=torch.long
        ),
        character_offsets=torch.tensor(
            [row[1] for row in source_rows], dtype=torch.long
        ),
        character_ids=torch.tensor(
            [row[2] for row in source_rows], dtype=torch.long
        ),
        character_mask=torch.tensor(
            [row[3] for row in source_rows], dtype=torch.bool
        ),
        start_positions=(torch.tensor(starts, dtype=torch.long) if require_gold else None),
        end_positions=(torch.tensor(ends, dtype=torch.long) if require_gold else None),
        cases=tuple(evidence),
    )


class ProtectedBidirectionalEvidenceOrgan(nn.Module):
    """A full-depth protected document encoder with contiguous span output."""

    surface = "marulho_protected_bidirectional_evidence_organ.v1"

    def __init__(
        self,
        source_model: MarulhoLanguageModel,
        *,
        context_length: int,
        initialized_from_parent: bool,
        model_seed: int,
        maximum_token_character_offset: int = 64,
        character_feature_dim: int = 16,
    ) -> None:
        super().__init__()
        torch.manual_seed(int(model_seed))
        if initialized_from_parent:
            donor = source_model
        else:
            donor = MarulhoLanguageModel(
                replace(
                    source_model.config,
                    transformer_context_length=int(context_length),
                )
            )
        self.token_embedding = copy.deepcopy(donor.token_embedding)
        self.state_block = copy.deepcopy(donor.state_block)
        self.context_length = int(context_length)
        width = int(source_model.config.state_dim)
        feature_dim = int(character_feature_dim)
        self.character_offset_embedding = nn.Embedding(
            int(maximum_token_character_offset), feature_dim
        )
        self.character_identity_embedding = nn.Embedding(256, feature_dim)
        head_width = width + feature_dim * 2
        self.start_head = nn.Linear(head_width, 1)
        self.end_head = nn.Linear(head_width, 1)
        nn.init.normal_(self.character_offset_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.character_identity_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.start_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.end_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.start_head.bias)
        nn.init.zeros_(self.end_head.bias)

    def _layer_forward(
        self,
        layer: nn.Module,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, time_steps, _ = value.shape
        normalized = layer.attention_norm(value)
        query, key, current_value = layer.attention.qkv(normalized).chunk(3, dim=-1)
        query = layer.attention._heads(query)
        key = layer.attention._heads(key)
        current_value = layer.attention._heads(current_value)
        positions = torch.arange(int(time_steps), device=value.device)
        query, key = _apply_rotary(query, key, positions)
        visible_keys = attention_mask[:, None, None, :]
        attention = F.scaled_dot_product_attention(
            query,
            key,
            current_value,
            attn_mask=visible_keys,
            dropout_p=0.0,
            is_causal=False,
        )
        attention = attention.transpose(1, 2).contiguous().view(
            int(batch_size),
            int(time_steps),
            int(layer.attention.width),
        )
        value = value + layer.dropout(layer.attention.output(attention))
        gate, up = layer.gate_up(layer.mlp_norm(value)).chunk(2, dim=-1)
        return value + layer.dropout(layer.down(F.silu(gate) * up))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        character_token_indices: torch.Tensor,
        character_offsets: torch.Tensor,
        character_ids: torch.Tensor,
        character_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2 or int(input_ids.shape[1]) != self.context_length:
            raise ValueError("Evidence organ expects [batch, context_length] token IDs")
        runtime_ids = input_ids.to(device=self.token_embedding.weight.device, dtype=torch.long)
        valid = attention_mask.to(device=runtime_ids.device, dtype=torch.bool)
        character_tokens = character_token_indices.to(
            device=runtime_ids.device, dtype=torch.long
        )
        offsets = character_offsets.to(device=runtime_ids.device, dtype=torch.long)
        identities = character_ids.to(device=runtime_ids.device, dtype=torch.long)
        characters = character_mask.to(device=runtime_ids.device, dtype=torch.bool)
        hidden = self.state_block.input_projection(self.token_embedding(runtime_ids))
        for layer in self.state_block.layers:
            hidden = self._layer_forward(layer, hidden, valid)
        hidden = self.state_block.output_norm(hidden)
        gathered = hidden.gather(
            1,
            character_tokens.unsqueeze(-1).expand(
                -1, -1, int(hidden.shape[-1])
            ),
        )
        character_features = torch.cat(
            (
                gathered,
                self.character_offset_embedding(offsets),
                self.character_identity_embedding(identities),
            ),
            dim=-1,
        )
        start_logits = self.start_head(character_features).squeeze(-1)
        end_logits = self.end_head(character_features).squeeze(-1)
        minimum = torch.finfo(start_logits.dtype).min
        return (
            start_logits.masked_fill(~characters, minimum),
            end_logits.masked_fill(~characters, minimum),
        )


def _learning_rate(config: V58Config, step: int) -> float:
    warmup = max(1, int(round(config.optimizer_steps * config.warmup_fraction)))
    if step < warmup:
        return config.learning_rate * float(step + 1) / float(warmup)
    progress = float(step - warmup) / float(max(1, config.optimizer_steps - warmup - 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    fraction = config.minimum_learning_rate_fraction + (
        1.0 - config.minimum_learning_rate_fraction
    ) * cosine
    return config.learning_rate * fraction


def _schedule_indices(case_count: int, config: V58Config) -> tuple[torch.Tensor, str]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.data_seed))
    epochs = [torch.randperm(int(case_count), generator=generator) for _ in range(config.epochs)]
    schedule = torch.cat(epochs)
    digest = hashlib.sha256(schedule.numpy().tobytes()).hexdigest()
    return schedule, digest


def _best_contiguous_span(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    character_mask: torch.Tensor,
    *,
    maximum_answer_characters: int,
) -> tuple[int, int]:
    positions = character_mask.nonzero(as_tuple=False).flatten().tolist()
    best_score = float("-inf")
    best = (int(positions[0]), int(positions[0]))
    source_set = set(int(value) for value in positions)
    for start in positions:
        for end in range(int(start), int(start) + int(maximum_answer_characters)):
            if end not in source_set:
                break
            score = float(start_logits[int(start)]) + float(end_logits[int(end)])
            if score > best_score:
                best_score = score
                best = (int(start), int(end))
    return best


@torch.no_grad()
def evaluate_organ(
    model: ProtectedBidirectionalEvidenceOrgan,
    prepared: PreparedEvidenceCases,
    tokenizer: LanguageTokenizer,
    config: V58Config,
) -> dict[str, Any]:
    was_training = bool(model.training)
    model.eval()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for offset in range(0, len(prepared), config.batch_size):
            stop = min(len(prepared), offset + config.batch_size)
            ids = prepared.input_ids[offset:stop].to(model.token_embedding.weight.device)
            attention = prepared.attention_mask[offset:stop].to(ids.device)
            character_tokens = prepared.character_token_indices[offset:stop].to(ids.device)
            character_offsets = prepared.character_offsets[offset:stop].to(ids.device)
            character_ids = prepared.character_ids[offset:stop].to(ids.device)
            characters = prepared.character_mask[offset:stop].to(ids.device)
            start_logits, end_logits = model(
                ids,
                attention,
                character_tokens,
                character_offsets,
                character_ids,
                characters,
            )
            start_cpu = start_logits.float().cpu()
            end_cpu = end_logits.float().cpu()
            character_cpu = characters.cpu()
            for local_index in range(stop - offset):
                start, end = _best_contiguous_span(
                    start_cpu[local_index],
                    end_cpu[local_index],
                    character_cpu[local_index],
                    maximum_answer_characters=config.maximum_answer_characters,
                )
                case = prepared.cases[offset + local_index]
                predicted = str(case["source_text"])[start : end + 1]
                accepted = {_normalized(value) for value in case["answers"]}
                exact = _normalized(predicted) in accepted
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "predicted_text": predicted,
                        "predicted_start": start,
                        "predicted_end": end,
                        "predicted_character_count": end - start + 1,
                        "exact_answer_match": exact,
                        "answers": list(case["answers"]),
                    }
                )
        if model.token_embedding.weight.device.type == "cuda":
            torch.cuda.synchronize(model.token_embedding.weight.device)
        elapsed = max(time.perf_counter() - started, 1.0e-9)
    finally:
        model.train(was_training)
    exact_count = sum(bool(row["exact_answer_match"]) for row in rows)
    return {
        "case_count": len(rows),
        "exact_answer_count": exact_count,
        "exact_answer_accuracy": exact_count / max(1, len(rows)),
        "elapsed_seconds": elapsed,
        "cases_per_second": len(rows) / elapsed,
        "rows": rows,
    }


def _sample_logits(
    model: ProtectedBidirectionalEvidenceOrgan,
    prepared: PreparedEvidenceCases,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        device = model.token_embedding.weight.device
        return tuple(
            value.detach().cpu()
            for value in model(
                prepared.input_ids[:2].to(device),
                prepared.attention_mask[:2].to(device),
                prepared.character_token_indices[:2].to(device),
                prepared.character_offsets[:2].to(device),
                prepared.character_ids[:2].to(device),
                prepared.character_mask[:2].to(device),
            )
        )  # type: ignore[return-value]


def _gradient_audit(model: nn.Module) -> dict[str, Any]:
    rows: dict[str, bool] = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        rows[name] = bool(
            gradient is not None and torch.count_nonzero(gradient.detach()).item() > 0
        )
    return {
        "tensor_count": len(rows),
        "nonzero_tensor_count": sum(rows.values()),
        "all_trainable_tensors_nonzero": bool(rows) and all(rows.values()),
        "by_parameter": rows,
    }


def train_arm(
    *,
    arm_name: str,
    parent: MarulhoLanguageModel,
    train_cases: PreparedEvidenceCases,
    validation_cases: PreparedEvidenceCases,
    mismatched_cases: PreparedEvidenceCases,
    tokenizer: LanguageTokenizer,
    config: V58Config,
    schedule: torch.Tensor,
    initialized_from_parent: bool,
    device: torch.device,
) -> tuple[dict[str, Any], ProtectedBidirectionalEvidenceOrgan]:
    model = ProtectedBidirectionalEvidenceOrgan(
        parent,
        context_length=config.context_length,
        initialized_from_parent=initialized_from_parent,
        model_seed=config.model_seed,
        maximum_token_character_offset=config.maximum_token_character_offset,
        character_feature_dim=config.character_feature_dim,
    ).to(device=device, dtype=torch.bfloat16)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    final_loss = float("nan")
    for step in range(config.optimizer_steps):
        offset = step * config.batch_size
        indices = schedule[offset : offset + config.batch_size]
        if int(indices.numel()) != config.batch_size:
            raise RuntimeError("Frozen V58 schedule does not fill the optimizer step")
        learning_rate = _learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        input_ids = train_cases.input_ids.index_select(0, indices).to(device)
        attention = train_cases.attention_mask.index_select(0, indices).to(device)
        character_tokens = train_cases.character_token_indices.index_select(0, indices).to(device)
        character_offsets = train_cases.character_offsets.index_select(0, indices).to(device)
        character_ids = train_cases.character_ids.index_select(0, indices).to(device)
        characters = train_cases.character_mask.index_select(0, indices).to(device)
        starts = train_cases.start_positions
        ends = train_cases.end_positions
        if starts is None or ends is None:
            raise RuntimeError("Training cases require gold spans")
        start_targets = starts.index_select(0, indices).to(device)
        end_targets = ends.index_select(0, indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        start_logits, end_logits = model(
            input_ids,
            attention,
            character_tokens,
            character_offsets,
            character_ids,
            characters,
        )
        loss = 0.5 * (
            F.cross_entropy(start_logits.float(), start_targets)
            + F.cross_entropy(end_logits.float(), end_targets)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        if (step + 1) % 256 == 0 or step + 1 == config.optimizer_steps:
            final_loss = float(loss.detach())
            print(
                f"[v58] {arm_name} {step + 1}/{config.optimizer_steps} "
                f"loss={final_loss:.4f} lr={learning_rate:.3g}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - started, 1.0e-9)
    gradients = _gradient_audit(model)
    peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    del optimizer
    intact = evaluate_organ(model, validation_cases, tokenizer, config)
    mismatched = evaluate_organ(model, mismatched_cases, tokenizer, config)
    return (
        {
            "arm_name": arm_name,
            "initialized_from_parent": initialized_from_parent,
            "parameter_count": parameter_count,
            "optimizer_steps": config.optimizer_steps,
            "padded_positions": config.padded_position_budget,
            "final_training_loss": final_loss,
            "training_seconds": elapsed,
            "training_positions_per_second": config.padded_position_budget / elapsed,
            "peak_cuda_bytes": peak,
            "final_gradients": gradients,
            "intact": intact,
            "mismatched_source": mismatched,
            "question_only": {
                "route": "unchanged_v39_causal_cortex",
                "evidence_organ_called": False,
                "extractive_answer_count": 0,
            },
        },
        model,
    )


def _parent_probe(
    model: MarulhoLanguageModel,
    tokenizer: LanguageTokenizer,
    manifest: Mapping[str, Any],
) -> torch.Tensor:
    row = dict(tuple(manifest["cases"])[0])
    ids = tokenizer.encode(str(row["question_only_prompt"]), add_eos=False)
    probe = torch.tensor(ids[-min(48, len(ids)) :], dtype=torch.long).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        return model(probe, collect_telemetry=False)["logits"].detach().cpu()


def run_v58(
    *,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    train_manifest_path: str | Path = DEFAULT_TRAIN_MANIFEST,
    validation_manifest_path: str | Path = DEFAULT_VALIDATION_MANIFEST,
    output_path: str | Path = DEFAULT_OUTPUT,
    candidate_path: str | Path = DEFAULT_CANDIDATE,
    device: str = "cuda",
) -> dict[str, Any]:
    config = V58Config()
    runtime_device = torch.device(device)
    if runtime_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V58 is frozen as an RTX CUDA experiment")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    checkpoint = Path(checkpoint_path)
    train_path = Path(train_manifest_path)
    validation_path = Path(validation_manifest_path)
    checkpoint_sha_before = _sha256_file(checkpoint)
    train_manifest = _load_json(train_path)
    validation_manifest = _load_json(validation_path)
    if len(tuple(train_manifest["cases"])) != 8192:
        raise ValueError("V58 requires exactly 8,192 training cases")
    if len(tuple(validation_manifest["cases"])) != 256:
        raise ValueError("V58 requires exactly 256 validation cases")

    parent, tokenizer, parent_payload = load_language_model_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    parent.eval()
    parent_parameter_count = sum(parameter.numel() for parameter in parent.parameters())
    parent_state_before = language_model_state_sha256(parent)
    tokenizer_hash_before = tokenizer.vocabulary_hash()
    parent_logits_before = _parent_probe(parent, tokenizer, validation_manifest)

    print("[v58] materializing exact span datasets", flush=True)
    setup_started = time.perf_counter()
    train_cases = prepare_evidence_cases(
        tuple(train_manifest["cases"]),
        tokenizer,
        context_length=config.context_length,
        maximum_source_characters=config.maximum_source_characters,
        maximum_answer_characters=config.maximum_answer_characters,
        maximum_token_character_offset=config.maximum_token_character_offset,
        require_gold=True,
    )
    validation_cases = prepare_evidence_cases(
        tuple(validation_manifest["cases"]),
        tokenizer,
        context_length=config.context_length,
        maximum_source_characters=config.maximum_source_characters,
        maximum_answer_characters=config.maximum_answer_characters,
        maximum_token_character_offset=config.maximum_token_character_offset,
        require_gold=True,
    )
    mismatched_cases = prepare_evidence_cases(
        tuple(validation_manifest["cases"]),
        tokenizer,
        context_length=config.context_length,
        maximum_source_characters=config.maximum_source_characters,
        maximum_token_character_offset=config.maximum_token_character_offset,
        source_field="mismatched_source_text",
        require_gold=False,
    )
    oracle_count = sum(
        _normalized(str(row["oracle_text"]))
        in {_normalized(value) for value in row["answers"]}
        for row in validation_cases.cases
    )
    schedule, schedule_sha = _schedule_indices(len(train_cases), config)
    if int(schedule.numel()) != config.optimizer_steps * config.batch_size:
        raise RuntimeError("V58 schedule violates the frozen position budget")
    setup_seconds = time.perf_counter() - setup_started

    print("[v58] training parent-initialized protected organ", flush=True)
    primary, primary_model = train_arm(
        arm_name="v39_initialized",
        parent=parent,
        train_cases=train_cases,
        validation_cases=validation_cases,
        mismatched_cases=mismatched_cases,
        tokenizer=tokenizer,
        config=config,
        schedule=schedule,
        initialized_from_parent=True,
        device=runtime_device,
    )
    primary_exact = int(primary["intact"]["exact_answer_count"])
    primary_mismatch = int(primary["mismatched_source"]["exact_answer_count"])
    primary_gain = float(primary["intact"]["exact_answer_accuracy"]) - float(
        primary["mismatched_source"]["exact_answer_accuracy"]
    )
    primary_checks = {
        "mechanical_oracle_256": oracle_count == 256,
        "minimum_exact_answers": primary_exact >= config.minimum_exact_answers,
        "minimum_source_gain": primary_gain >= config.minimum_source_gain,
        "maximum_mismatched_answers": (
            primary_mismatch <= config.maximum_mismatched_answers
        ),
        "exact_optimizer_steps": primary["optimizer_steps"] == config.optimizer_steps,
        "exact_position_budget": (
            primary["padded_positions"] == config.padded_position_budget
        ),
        "complete_final_gradients": bool(
            primary["final_gradients"]["all_trainable_tensors_nonzero"]
        ),
        "bounded_training_time": (
            float(primary["training_seconds"]) <= config.maximum_training_seconds
        ),
        "capacity_ceiling_bounded": (
            int(primary["parameter_count"]) <= parent_parameter_count + 10_000
        ),
    }
    primary_passed = all(primary_checks.values())

    primary_state: dict[str, torch.Tensor] | None = None
    primary_sample_before: tuple[torch.Tensor, torch.Tensor] | None = None
    random_arm: dict[str, Any] | None = None
    random_model: ProtectedBidirectionalEvidenceOrgan | None = None
    if primary_passed:
        primary_state = {
            name: value.detach().cpu().clone()
            for name, value in primary_model.state_dict().items()
        }
        primary_sample_before = _sample_logits(primary_model, validation_cases)
        del primary_model
        torch.cuda.empty_cache()
        print("[v58] primary passed; training mandatory random-init control", flush=True)
        random_arm, random_model = train_arm(
            arm_name="random_initialized",
            parent=parent,
            train_cases=train_cases,
            validation_cases=validation_cases,
            mismatched_cases=mismatched_cases,
            tokenizer=tokenizer,
            config=config,
            schedule=schedule,
            initialized_from_parent=False,
            device=runtime_device,
        )
        initialized_advantage = primary_exact - int(
            random_arm["intact"]["exact_answer_count"]
        )
    else:
        initialized_advantage = None

    checkpoint_sha_after = _sha256_file(checkpoint)
    parent_state_after = language_model_state_sha256(parent)
    parent_logits_after = _parent_probe(parent, tokenizer, validation_manifest)
    tokenizer_hash_after = tokenizer.vocabulary_hash()
    parent_checks = {
        "checkpoint_file_exact": checkpoint_sha_before == checkpoint_sha_after,
        "state_exact": parent_state_before == parent_state_after,
        "tokenizer_exact": tokenizer_hash_before == tokenizer_hash_after,
        "sample_logits_exact": torch.equal(parent_logits_before, parent_logits_after),
    }

    checkpoint_evidence: dict[str, Any] = {
        "saved": False,
        "path": None,
        "sha256": None,
        "strict_tensor_reload": False,
        "strict_logit_reload": False,
    }
    if primary_passed and all(parent_checks.values()):
        if primary_state is None or primary_sample_before is None:
            raise RuntimeError("Passing V58 arm did not retain its exact candidate state")
        if random_model is not None:
            del random_model
            torch.cuda.empty_cache()
        candidate = Path(candidate_path)
        payload = {
            "artifact_kind": "marulho_protected_evidence_organ_checkpoint",
            "surface": "marulho_protected_evidence_organ_checkpoint.v1",
            "owned_by_marulho": True,
            "external_llm_used": False,
            "parent_checkpoint_sha256": checkpoint_sha_before,
            "tokenizer_hash": tokenizer_hash_before,
            "configuration": asdict(config),
            "model_state": primary_state,
        }
        _atomic_torch_save(candidate, payload)
        candidate_sha = _sha256_file(candidate)
        loaded = torch.load(candidate, map_location="cpu", weights_only=False)
        reloaded = ProtectedBidirectionalEvidenceOrgan(
            parent,
            context_length=config.context_length,
            initialized_from_parent=True,
            model_seed=config.model_seed,
            maximum_token_character_offset=config.maximum_token_character_offset,
            character_feature_dim=config.character_feature_dim,
        )
        reloaded.load_state_dict(dict(loaded["model_state"]), strict=True)
        reloaded_state = reloaded.state_dict()
        tensor_exact = all(
            torch.equal(value, reloaded_state[name].detach().cpu())
            for name, value in primary_state.items()
        )
        reloaded = reloaded.to(device=runtime_device, dtype=torch.bfloat16)
        sample_after = _sample_logits(reloaded, validation_cases)
        logit_exact = all(
            torch.equal(left, right)
            for left, right in zip(primary_sample_before, sample_after)
        )
        del reloaded
        torch.cuda.empty_cache()
        checkpoint_evidence = {
            "saved": True,
            "path": str(candidate),
            "sha256": candidate_sha,
            "strict_tensor_reload": tensor_exact,
            "strict_logit_reload": logit_exact,
        }
    else:
        if not primary_passed:
            del primary_model
        if random_model is not None:
            del random_model
        torch.cuda.empty_cache()

    checkpoint_passed = (
        bool(checkpoint_evidence["saved"])
        and bool(checkpoint_evidence["strict_tensor_reload"])
        and bool(checkpoint_evidence["strict_logit_reload"])
    )
    passed = primary_passed and all(parent_checks.values()) and checkpoint_passed
    if not passed:
        Path(candidate_path).unlink(missing_ok=True)
        checkpoint_evidence["saved"] = False
        checkpoint_evidence["path"] = None
        checkpoint_evidence["sha256"] = None
    decision = ADVANCE_DECISION if passed else RETIRE_DECISION
    transfer = {
        "random_control_required": primary_passed,
        "random_control_completed": random_arm is not None,
        "initialized_advantage_cases": initialized_advantage,
        "minimum_initialized_advantage": config.minimum_initialized_advantage,
        "language_pretraining_transfer_supported": bool(
            initialized_advantage is not None
            and initialized_advantage >= config.minimum_initialized_advantage
        ),
        "interpretation": (
            "language_cortex_initialization_transfers_to_localization"
            if initialized_advantage is not None
            and initialized_advantage >= config.minimum_initialized_advantage
            else (
                "organ_capability_does_not_require_language_initialization"
                if initialized_advantage is not None
                else "not_run_after_primary_capability_failure"
            )
        ),
    }
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "decision": decision,
        "configuration": asdict(config),
        "experiment_contract_sha256": sha256_json(
            {
                "surface": SURFACE,
                "configuration": asdict(config),
                "checkpoint_sha256": checkpoint_sha_before,
                "train_manifest_contract_sha256": train_manifest["contract_sha256"],
                "validation_manifest_contract_sha256": validation_manifest[
                    "contract_sha256"
                ],
            }
        ),
        "data": {
            "train_manifest_path": str(train_path),
            "train_manifest_sha256": _sha256_file(train_path),
            "train_manifest_contract_sha256": train_manifest["contract_sha256"],
            "validation_manifest_path": str(validation_path),
            "validation_manifest_sha256": _sha256_file(validation_path),
            "validation_manifest_contract_sha256": validation_manifest[
                "contract_sha256"
            ],
            "train_case_count": len(train_cases),
            "validation_case_count": len(validation_cases),
            "mechanical_oracle_exact_answer_count": oracle_count,
            "train_answer_offsets_corrected": sum(
                bool(row["answer_start_corrected"]) for row in train_cases.cases
            ),
            "validation_answer_offsets_corrected": sum(
                bool(row["answer_start_corrected"])
                for row in validation_cases.cases
            ),
            "schedule_sha256": schedule_sha,
            "setup_seconds": setup_seconds,
        },
        "parent": {
            "path": str(checkpoint),
            "checkpoint_sha256_before": checkpoint_sha_before,
            "checkpoint_sha256_after": checkpoint_sha_after,
            "state_sha256_before": parent_state_before,
            "state_sha256_after": parent_state_after,
            "tokenizer_hash_before": tokenizer_hash_before,
            "tokenizer_hash_after": tokenizer_hash_after,
            "parameter_count": parent_parameter_count,
            "metadata": parent_payload.get("metadata", {}),
            "checks": parent_checks,
        },
        "primary": primary,
        "random_control": random_arm,
        "transfer": transfer,
        "checkpoint": checkpoint_evidence,
        "gate": {
            "passed": passed,
            "primary_checks": primary_checks,
            "parent_checks": parent_checks,
            "checkpoint_passed": checkpoint_passed,
            "observed": {
                "primary_exact_answers": primary_exact,
                "primary_mismatched_answers": primary_mismatch,
                "primary_source_gain": primary_gain,
            },
            "thresholds": asdict(config),
        },
    }
    write_json_report_with_readme(output_path, report)
    print(
        f"[v58] decision={decision} exact={primary_exact}/256 "
        f"mismatch={primary_mismatch}/256",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--train-manifest", default=str(DEFAULT_TRAIN_MANIFEST))
    parser.add_argument("--validation-manifest", default=str(DEFAULT_VALIDATION_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    run_v58(
        checkpoint_path=arguments.checkpoint,
        train_manifest_path=arguments.train_manifest,
        validation_manifest_path=arguments.validation_manifest,
        output_path=arguments.output,
        candidate_path=arguments.candidate,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
