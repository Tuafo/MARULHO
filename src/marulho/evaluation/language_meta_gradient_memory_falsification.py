"""V60 end-to-end meta-gradient episodic-memory falsifier."""

from __future__ import annotations

import argparse
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
    MarulhoLanguageModel,
    _apply_decode_controls,
    language_model_state_sha256,
    load_language_model_checkpoint,
)


SURFACE = "marulho_meta_gradient_episodic_matrix_falsification.v1"
ARTIFACT_KIND = "marulho_meta_gradient_episodic_matrix_falsification"
ADVANCE_DECISION = "advance_v60_meta_gradient_memory_to_continual_routing"
RETIRE_DECISION = "retire_v60_one_step_linear_meta_gradient_memory"
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
    "reports/language_scaling/meta-gradient-memory-v60-20m-20260812.json"
)
DEFAULT_CANDIDATE = Path(
    "reports/language_scaling/v60-meta-gradient-memory-qualified-20260812.pt"
)


@dataclass(frozen=True)
class V60Config:
    context_length: int = 96
    source_chunk_length: int = 64
    source_chunk_count: int = 5
    source_memory_positions: int = 320
    batch_size: int = 32
    epochs: int = 8
    optimizer_steps: int = 2048
    padded_source_position_budget: int = 20_971_520
    memory_heads: int = 8
    key_width_per_head: int = 16
    value_width_per_head: int = 96
    learning_rate: float = 3.0e-4
    warmup_fraction: float = 0.05
    minimum_learning_rate_fraction: float = 0.1
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    generation_tokens: int = 16
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    minimum_true_exact_answers: int = 64
    minimum_true_source_gain: float = 0.20
    maximum_shuffled_exact_answers: int = 16
    minimum_oracle_exact_answers: int = 128
    maximum_true_oracle_gap: int = 64
    maximum_parameter_fraction: float = 0.01
    maximum_training_seconds: float = 1800.0
    maximum_total_setup_training_seconds: float = 2400.0
    data_seed: int = 60121
    model_seed: int = 60131
    precision: str = "bfloat16"
    execution_backend: str = "pytorch_eager"


@dataclass(frozen=True)
class PreparedMetaCases:
    source_input_ids: torch.Tensor
    source_target_ids: torch.Tensor
    source_mask: torch.Tensor
    query_input_ids: torch.Tensor
    query_target_ids: torch.Tensor
    answer_mask: torch.Tensor
    cases: tuple[dict[str, Any], ...]

    def __len__(self) -> int:
        return int(self.source_input_ids.shape[0])


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {path}")
    return dict(payload)


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


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


def _question_prompt(row: Mapping[str, Any]) -> str:
    return f"Question: {str(row['question']).strip()}\nAnswer: "


def _prepare_source(
    source: str,
    tokenizer: LanguageTokenizer,
    config: V60Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    ids = tokenizer.encode(
        f"Context: {source}",
        add_bos=True,
        add_eos=True,
    )
    capacity = config.source_chunk_length * config.source_chunk_count
    if len(ids) - 1 > capacity:
        raise ValueError(
            f"Source requires {len(ids) - 1} next-token positions, above {capacity}"
        )
    inputs: list[int] = []
    targets: list[int] = []
    mask: list[bool] = []
    for chunk_index in range(config.source_chunk_count):
        start = chunk_index * config.source_chunk_length
        segment = ids[start : start + config.source_chunk_length + 1]
        valid = max(0, len(segment) - 1)
        chunk_inputs = segment[:-1]
        chunk_targets = segment[1:]
        padding = config.source_chunk_length - valid
        inputs.extend(chunk_inputs + [int(tokenizer.pad_id)] * padding)
        targets.extend(chunk_targets + [int(tokenizer.pad_id)] * padding)
        mask.extend([True] * valid + [False] * padding)
    if len(inputs) != config.source_memory_positions:
        raise RuntimeError("V60 source preparation violated the frozen shape")
    return (
        torch.tensor(inputs, dtype=torch.long).view(
            config.source_chunk_count, config.source_chunk_length
        ),
        torch.tensor(targets, dtype=torch.long).view(
            config.source_chunk_count, config.source_chunk_length
        ),
        torch.tensor(mask, dtype=torch.bool).view(
            config.source_chunk_count, config.source_chunk_length
        ),
        sum(mask),
    )


def _prepare_query(
    row: Mapping[str, Any],
    tokenizer: LanguageTokenizer,
    config: V60Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    prompt_ids = tokenizer.encode(_question_prompt(row), add_eos=False)
    answer_ids = tokenizer.encode(
        str(tuple(row["answers"])[0]),
        add_bos=False,
        add_eos=True,
    )
    combined = prompt_ids + answer_ids
    inputs = combined[:-1]
    targets = combined[1:]
    if len(inputs) > config.context_length:
        raise ValueError(
            f"Case {row.get('case_id')} query/answer has {len(inputs)} positions"
        )
    answer_start = len(prompt_ids) - 1
    answer_mask = [index >= answer_start for index in range(len(inputs))]
    padding = config.context_length - len(inputs)
    return (
        torch.tensor(
            inputs + [int(tokenizer.pad_id)] * padding,
            dtype=torch.long,
        ),
        torch.tensor(
            targets + [int(tokenizer.pad_id)] * padding,
            dtype=torch.long,
        ),
        torch.tensor(answer_mask + [False] * padding, dtype=torch.bool),
        sum(answer_mask),
    )


def prepare_meta_cases(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: LanguageTokenizer,
    config: V60Config,
    *,
    source_field: str = "source_text",
    include_query: bool = True,
) -> PreparedMetaCases:
    source_inputs: list[torch.Tensor] = []
    source_targets: list[torch.Tensor] = []
    source_masks: list[torch.Tensor] = []
    query_inputs: list[torch.Tensor] = []
    query_targets: list[torch.Tensor] = []
    answer_masks: list[torch.Tensor] = []
    evidence: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        source = str(row[source_field])
        source_input, source_target, source_mask, source_positions = _prepare_source(
            source,
            tokenizer,
            config,
        )
        if include_query:
            query_input, query_target, answer_mask, answer_positions = _prepare_query(
                row,
                tokenizer,
                config,
            )
        else:
            query_input = torch.full(
                (config.context_length,), int(tokenizer.pad_id), dtype=torch.long
            )
            query_target = query_input.clone()
            answer_mask = torch.zeros(config.context_length, dtype=torch.bool)
            answer_positions = 0
        source_inputs.append(source_input)
        source_targets.append(source_target)
        source_masks.append(source_mask)
        query_inputs.append(query_input)
        query_targets.append(query_target)
        answer_masks.append(answer_mask)
        evidence.append(
            {
                "case_id": str(row["case_id"]),
                "title": str(row["title"]),
                "question": str(row["question"]),
                "answers": [str(value) for value in row["answers"]],
                "source_text": source,
                "source_positions": source_positions,
                "answer_positions": answer_positions,
            }
        )
    return PreparedMetaCases(
        source_input_ids=torch.stack(source_inputs),
        source_target_ids=torch.stack(source_targets),
        source_mask=torch.stack(source_masks),
        query_input_ids=torch.stack(query_inputs),
        query_target_ids=torch.stack(query_targets),
        answer_mask=torch.stack(answer_masks),
        cases=tuple(evidence),
    )


class MetaGradientEpisodicMatrix(nn.Module):
    """One-step linear fast memory with a meta-learned write/read geometry."""

    surface = "marulho_meta_gradient_episodic_matrix.v1"

    def __init__(
        self,
        *,
        width: int,
        memory_heads: int,
        key_width_per_head: int,
        value_width_per_head: int,
        model_seed: int,
    ) -> None:
        super().__init__()
        torch.manual_seed(int(model_seed))
        self.width = int(width)
        self.memory_heads = int(memory_heads)
        self.key_width_per_head = int(key_width_per_head)
        self.value_width_per_head = int(value_width_per_head)
        if self.memory_heads * self.value_width_per_head != self.width:
            raise ValueError("Memory value heads must concatenate to model width")
        key_width = self.memory_heads * self.key_width_per_head
        self.key_projection = nn.Linear(self.width, key_width, bias=False)
        self.query_projection = nn.Linear(self.width, key_width, bias=False)
        self.read_projection = nn.Linear(self.width, self.width, bias=False)
        self.log_write_rates = nn.Parameter(torch.full((self.memory_heads,), 3.0))
        self.head_gate_logits = nn.Parameter(torch.full((self.memory_heads,), -2.0))
        self.output_gate_logit = nn.Parameter(torch.tensor(-2.0))
        nn.init.normal_(self.key_projection.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.query_projection.weight, mean=0.0, std=0.02)
        nn.init.eye_(self.read_projection.weight)

    @property
    def fast_state_values_per_document(self) -> int:
        return (
            self.memory_heads
            * self.key_width_per_head
            * self.value_width_per_head
        )

    def write(
        self,
        source_hidden: torch.Tensor,
        source_target_embeddings: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, positions, _ = source_hidden.shape
        keys = self.key_projection(source_hidden).view(
            batch,
            positions,
            self.memory_heads,
            self.key_width_per_head,
        )
        keys = F.normalize(keys.float(), dim=-1).to(source_hidden.dtype)
        values = source_target_embeddings.view(
            batch,
            positions,
            self.memory_heads,
            self.value_width_per_head,
        )
        mask = source_mask.to(device=source_hidden.device, dtype=source_hidden.dtype)
        keys = keys * mask[:, :, None, None]
        values = values * mask[:, :, None, None]
        gradient_write = torch.einsum("bshk,bshv->bhkv", keys, values)
        denominator = mask.sum(dim=1).clamp_min(1.0)[:, None, None, None]
        rates = F.softplus(self.log_write_rates).view(1, -1, 1, 1)
        return gradient_write * rates / denominator

    def read(self, query_hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch, positions, _ = query_hidden.shape
        queries = self.query_projection(query_hidden).view(
            batch,
            positions,
            self.memory_heads,
            self.key_width_per_head,
        )
        queries = F.normalize(queries.float(), dim=-1).to(query_hidden.dtype)
        read_heads = torch.einsum("bthk,bhkv->bthv", queries, memory)
        gates = torch.sigmoid(self.head_gate_logits).view(1, 1, -1, 1)
        read = (read_heads * gates).reshape(batch, positions, self.width)
        projected = self.read_projection(read)
        output_gate = torch.sigmoid(self.output_gate_logit)
        return query_hidden + output_gate * projected

    def forward(
        self,
        source_hidden: torch.Tensor,
        source_target_embeddings: torch.Tensor,
        source_mask: torch.Tensor,
        query_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.write(source_hidden, source_target_embeddings, source_mask)
        return self.read(query_hidden, memory), memory


def _schedule_indices(case_count: int, config: V60Config) -> tuple[torch.Tensor, str]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.data_seed)
    epochs = [torch.randperm(case_count, generator=generator) for _ in range(config.epochs)]
    schedule = torch.cat(epochs)
    digest = hashlib.sha256(schedule.numpy().tobytes()).hexdigest()
    return schedule, digest


def _learning_rate(config: V60Config, step: int) -> float:
    warmup = max(1, int(round(config.optimizer_steps * config.warmup_fraction)))
    if step < warmup:
        return config.learning_rate * float(step + 1) / float(warmup)
    progress = float(step - warmup) / float(max(1, config.optimizer_steps - warmup - 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    fraction = config.minimum_learning_rate_fraction + (
        1.0 - config.minimum_learning_rate_fraction
    ) * cosine
    return config.learning_rate * fraction


@torch.no_grad()
def _encode_source_batch(
    parent: MarulhoLanguageModel,
    prepared: PreparedMetaCases,
    indices: torch.Tensor,
    config: V60Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = prepared.source_input_ids.index_select(0, indices).to(parent.device)
    targets = prepared.source_target_ids.index_select(0, indices).to(parent.device)
    mask = prepared.source_mask.index_select(0, indices).to(parent.device)
    flat_inputs = inputs.reshape(-1, config.source_chunk_length)
    hidden = parent._forward_hidden(flat_inputs, collect_telemetry=False)["hidden"]
    hidden = hidden.view(len(indices), config.source_memory_positions, -1)
    target_embeddings = parent.token_embedding(
        targets.reshape(len(indices), config.source_memory_positions)
    )
    return (
        hidden,
        target_embeddings,
        mask.reshape(len(indices), config.source_memory_positions),
    )


@torch.no_grad()
def _encode_query_batch(
    parent: MarulhoLanguageModel,
    prepared: PreparedMetaCases,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = prepared.query_input_ids.index_select(0, indices).to(parent.device)
    targets = prepared.query_target_ids.index_select(0, indices).to(parent.device)
    mask = prepared.answer_mask.index_select(0, indices).to(parent.device)
    hidden = parent._forward_hidden(inputs, collect_telemetry=False)["hidden"]
    return hidden, targets, mask


def _gradient_audit(model: nn.Module) -> dict[str, Any]:
    by_parameter: dict[str, bool] = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        by_parameter[name] = bool(
            gradient is not None and torch.count_nonzero(gradient).item() > 0
        )
    return {
        "tensor_count": len(by_parameter),
        "nonzero_tensor_count": sum(by_parameter.values()),
        "all_trainable_tensors_nonzero": bool(by_parameter)
        and all(by_parameter.values()),
        "by_parameter": by_parameter,
    }


def train_controller(
    parent: MarulhoLanguageModel,
    controller: MetaGradientEpisodicMatrix,
    prepared: PreparedMetaCases,
    schedule: torch.Tensor,
    config: V60Config,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
        fused=parent.device.type == "cuda",
    )
    controller.train()
    if parent.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(parent.device)
        torch.cuda.synchronize(parent.device)
    started = time.perf_counter()
    final_loss = float("nan")
    answer_positions = 0
    for step in range(config.optimizer_steps):
        offset = step * config.batch_size
        indices = schedule[offset : offset + config.batch_size]
        source_hidden, target_embeddings, source_mask = _encode_source_batch(
            parent, prepared, indices, config
        )
        query_hidden, query_targets, answer_mask = _encode_query_batch(
            parent, prepared, indices
        )
        learning_rate = _learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        read_hidden, _memory = controller(
            source_hidden,
            target_embeddings,
            source_mask,
            query_hidden,
        )
        logits = parent.lm_head(read_hidden)
        loss = F.cross_entropy(
            logits[answer_mask].float(),
            query_targets[answer_mask],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), config.gradient_clip)
        optimizer.step()
        answer_positions += int(answer_mask.sum())
        if (step + 1) % 256 == 0 or step + 1 == config.optimizer_steps:
            final_loss = float(loss.detach())
            print(
                f"[v60] train {step + 1}/{config.optimizer_steps} "
                f"loss={final_loss:.4f} lr={learning_rate:.3g}",
                flush=True,
            )
    if parent.device.type == "cuda":
        torch.cuda.synchronize(parent.device)
    elapsed = max(time.perf_counter() - started, 1.0e-9)
    gradients = _gradient_audit(controller)
    peak = (
        int(torch.cuda.max_memory_allocated(parent.device))
        if parent.device.type == "cuda"
        else 0
    )
    del optimizer
    return {
        "optimizer_steps": config.optimizer_steps,
        "padded_source_positions": config.padded_source_position_budget,
        "answer_target_positions": answer_positions,
        "final_training_loss": final_loss,
        "training_seconds": elapsed,
        "source_positions_per_second": config.padded_source_position_budget / elapsed,
        "peak_cuda_bytes": peak,
        "final_gradients": gradients,
    }


@torch.no_grad()
def _memory_for_source(
    parent: MarulhoLanguageModel,
    controller: MetaGradientEpisodicMatrix,
    source: str,
    tokenizer: LanguageTokenizer,
    config: V60Config,
) -> torch.Tensor:
    source_input, source_target, source_mask, _ = _prepare_source(
        source,
        tokenizer,
        config,
    )
    inputs = source_input.to(parent.device)
    targets = source_target.to(parent.device)
    mask = source_mask.to(parent.device).reshape(1, config.source_memory_positions)
    hidden = parent._forward_hidden(inputs, collect_telemetry=False)["hidden"]
    hidden = hidden.reshape(1, config.source_memory_positions, -1)
    embeddings = parent.token_embedding(targets.reshape(1, config.source_memory_positions))
    return controller.write(hidden, embeddings, mask)


@torch.no_grad()
def _generate_with_memory(
    parent: MarulhoLanguageModel,
    controller: MetaGradientEpisodicMatrix,
    tokenizer: LanguageTokenizer,
    row: Mapping[str, Any],
    memory: torch.Tensor,
    config: V60Config,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(_question_prompt(row), add_eos=False)
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=parent.device).unsqueeze(0)
    result = parent._forward_hidden(prompt, collect_telemetry=False)
    state = result["state"]
    hidden = result["hidden"][:, -1:, :]
    generated: list[int] = []
    for _ in range(config.generation_tokens):
        read_hidden = controller.read(hidden, memory)
        logits = parent.lm_head(read_hidden[:, -1, :])
        generated_tensor = torch.tensor(
            generated,
            dtype=torch.long,
            device=parent.device,
        ).unsqueeze(0)
        adjusted, _ = _apply_decode_controls(
            logits,
            generated_tensor,
            repetition_penalty=config.repetition_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
        )
        next_id = int(torch.argmax(adjusted, dim=-1).item())
        generated.append(next_id)
        if next_id == int(tokenizer.eos_id):
            break
        step = parent._forward_hidden(
            torch.tensor([[next_id]], dtype=torch.long, device=parent.device),
            state,
            collect_telemetry=False,
        )
        state = step["state"]
        hidden = step["hidden"]
    continuation = tokenizer.decode(generated)
    accepted = {_normalized(value) for value in row["answers"]}
    return {
        "case_id": str(row["case_id"]),
        "title": str(row["title"]),
        "answers": [str(value) for value in row["answers"]],
        "continuation_text": continuation,
        "continuation_ids": generated,
        "exact_answer_match": _normalized(continuation) in accepted,
        "contains_accepted_answer": any(
            answer and answer in _normalized(continuation) for answer in accepted
        ),
    }


@torch.no_grad()
def evaluate_view(
    parent: MarulhoLanguageModel,
    controller: MetaGradientEpisodicMatrix,
    tokenizer: LanguageTokenizer,
    rows: Sequence[Mapping[str, Any]],
    config: V60Config,
    *,
    view: str,
) -> dict[str, Any]:
    controller.eval()
    started = time.perf_counter()
    output_rows: list[dict[str, Any]] = []
    zero_memory = torch.zeros(
        1,
        config.memory_heads,
        config.key_width_per_head,
        config.value_width_per_head,
        device=parent.device,
        dtype=parent.token_embedding.weight.dtype,
    )
    for index, row in enumerate(rows):
        if view == "zero_memory":
            memory = zero_memory
        elif view == "true_memory":
            memory = _memory_for_source(
                parent, controller, str(row["source_text"]), tokenizer, config
            )
        elif view == "oracle_short_memory":
            memory = _memory_for_source(
                parent, controller, str(row["oracle_source_text"]), tokenizer, config
            )
        elif view == "shuffled_memory":
            wrong = rows[(index + 1) % len(rows)]
            memory = _memory_for_source(
                parent, controller, str(wrong["source_text"]), tokenizer, config
            )
        else:
            raise ValueError(f"Unknown V60 view {view!r}")
        output_rows.append(
            _generate_with_memory(parent, controller, tokenizer, row, memory, config)
        )
        if (index + 1) % 64 == 0:
            exact_so_far = sum(bool(item["exact_answer_match"]) for item in output_rows)
            print(
                f"[v60] {view} {index + 1}/{len(rows)} exact={exact_so_far}",
                flush=True,
            )
    if parent.device.type == "cuda":
        torch.cuda.synchronize(parent.device)
    elapsed = max(time.perf_counter() - started, 1.0e-9)
    exact = sum(bool(row["exact_answer_match"]) for row in output_rows)
    contains = sum(bool(row["contains_accepted_answer"]) for row in output_rows)
    return {
        "view": view,
        "case_count": len(output_rows),
        "exact_answer_count": exact,
        "exact_answer_accuracy": exact / max(1, len(output_rows)),
        "contains_accepted_answer_count": contains,
        "elapsed_seconds": elapsed,
        "cases_per_second": len(output_rows) / elapsed,
        "rows": output_rows,
    }


def _parent_probe(
    parent: MarulhoLanguageModel,
    tokenizer: LanguageTokenizer,
    row: Mapping[str, Any],
) -> torch.Tensor:
    ids = tokenizer.encode(_question_prompt(row), add_eos=False)
    probe = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    parent.eval()
    with torch.no_grad():
        return parent(probe, collect_telemetry=False)["logits"].detach().cpu()


def _controller_probe(
    parent: MarulhoLanguageModel,
    controller: MetaGradientEpisodicMatrix,
    tokenizer: LanguageTokenizer,
    row: Mapping[str, Any],
    config: V60Config,
) -> torch.Tensor:
    controller.eval()
    with torch.no_grad():
        memory = _memory_for_source(
            parent, controller, str(row["source_text"]), tokenizer, config
        )
        ids = tokenizer.encode(_question_prompt(row), add_eos=False)
        hidden = parent._forward_hidden(
            torch.tensor([ids], dtype=torch.long, device=parent.device),
            collect_telemetry=False,
        )["hidden"][:, -1:, :]
        return parent.lm_head(controller.read(hidden, memory)).detach().cpu()


def run_v60(
    *,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    train_manifest_path: str | Path = DEFAULT_TRAIN_MANIFEST,
    validation_manifest_path: str | Path = DEFAULT_VALIDATION_MANIFEST,
    output_path: str | Path = DEFAULT_OUTPUT,
    candidate_path: str | Path = DEFAULT_CANDIDATE,
    device: str = "cuda",
) -> dict[str, Any]:
    config = V60Config()
    runtime_device = torch.device(device)
    if runtime_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V60 is frozen as a CUDA experiment")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    total_started = time.perf_counter()

    checkpoint = Path(checkpoint_path)
    train_path = Path(train_manifest_path)
    validation_path = Path(validation_manifest_path)
    checkpoint_sha_before = _sha256_file(checkpoint)
    train_manifest = _load_json(train_path)
    validation_manifest = _load_json(validation_path)
    parent_cpu, tokenizer, parent_payload = load_language_model_checkpoint(
        checkpoint, map_location="cpu"
    )
    parent_state_before = language_model_state_sha256(parent_cpu)
    tokenizer_hash_before = tokenizer.vocabulary_hash()
    parent_logits_before = _parent_probe(
        parent_cpu, tokenizer, validation_manifest["cases"][0]
    )
    runtime_parent, runtime_tokenizer, _ = load_language_model_checkpoint(
        checkpoint, map_location="cpu"
    )
    if runtime_tokenizer.vocabulary_hash() != tokenizer_hash_before:
        raise ValueError("V60 runtime tokenizer differs from immutable parent")
    parent = MarulhoLanguageModel(
        replace(
            runtime_parent.config,
            transformer_context_length=config.context_length,
        )
    )
    parent.load_state_dict(runtime_parent.state_dict(), strict=True)
    parity_ids = tokenizer.encode(
        _question_prompt(validation_manifest["cases"][0]), add_eos=False
    )[:72]
    parity_probe = torch.tensor([parity_ids], dtype=torch.long)
    runtime_parent.eval()
    parent.eval()
    with torch.no_grad():
        original_logits = runtime_parent(
            parity_probe, collect_telemetry=False
        )["logits"]
        extended_logits = parent(parity_probe, collect_telemetry=False)["logits"]
    initial_short_prefix_exact = torch.equal(original_logits, extended_logits)
    if not initial_short_prefix_exact:
        raise RuntimeError("V60 context-96 reconstruction changed a short prefix")
    del runtime_parent, original_logits, extended_logits
    parent = parent.to(device=runtime_device, dtype=torch.bfloat16)
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    parent_parameter_count = sum(parameter.numel() for parameter in parent.parameters())

    setup_started = time.perf_counter()
    print("[v60] preparing frozen source/query tensors", flush=True)
    train_cases = prepare_meta_cases(
        tuple(train_manifest["cases"]), tokenizer, config
    )
    validation_rows = tuple(dict(row) for row in validation_manifest["cases"])
    if len(train_cases) != 8192 or len(validation_rows) != 256:
        raise ValueError("V60 requires the exact V57 8192/256 data boundary")
    train_titles = {str(row["title"]) for row in train_manifest["cases"]}
    validation_titles = {str(row["title"]) for row in validation_rows}
    if train_titles & validation_titles:
        raise ValueError("V60 train and validation titles overlap")
    schedule, schedule_sha = _schedule_indices(len(train_cases), config)
    if int(schedule.numel()) != config.optimizer_steps * config.batch_size:
        raise RuntimeError("V60 schedule violates the frozen update count")
    prepared_sha = sha256_json(
        {
            "source_input_ids": hashlib.sha256(
                train_cases.source_input_ids.numpy().tobytes()
            ).hexdigest(),
            "source_target_ids": hashlib.sha256(
                train_cases.source_target_ids.numpy().tobytes()
            ).hexdigest(),
            "source_mask": hashlib.sha256(
                train_cases.source_mask.numpy().tobytes()
            ).hexdigest(),
            "query_input_ids": hashlib.sha256(
                train_cases.query_input_ids.numpy().tobytes()
            ).hexdigest(),
            "query_target_ids": hashlib.sha256(
                train_cases.query_target_ids.numpy().tobytes()
            ).hexdigest(),
            "answer_mask": hashlib.sha256(
                train_cases.answer_mask.numpy().tobytes()
            ).hexdigest(),
        }
    )
    setup_seconds = time.perf_counter() - setup_started

    controller = MetaGradientEpisodicMatrix(
        width=int(parent.config.state_dim),
        memory_heads=config.memory_heads,
        key_width_per_head=config.key_width_per_head,
        value_width_per_head=config.value_width_per_head,
        model_seed=config.model_seed,
    ).to(device=runtime_device, dtype=torch.bfloat16)
    controller_parameter_count = sum(
        parameter.numel() for parameter in controller.parameters()
    )
    parameter_fraction = controller_parameter_count / parent_parameter_count
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in controller.state_dict().items()
    }
    initial_true = evaluate_view(
        parent,
        controller,
        tokenizer,
        validation_rows,
        config,
        view="true_memory",
    )

    print("[v60] training meta-gradient controller", flush=True)
    training = train_controller(parent, controller, train_cases, schedule, config)
    print("[v60] evaluating learned memory controls", flush=True)
    zero = evaluate_view(
        parent, controller, tokenizer, validation_rows, config, view="zero_memory"
    )
    shuffled = evaluate_view(
        parent, controller, tokenizer, validation_rows, config, view="shuffled_memory"
    )
    true = evaluate_view(
        parent, controller, tokenizer, validation_rows, config, view="true_memory"
    )
    oracle = evaluate_view(
        parent,
        controller,
        tokenizer,
        validation_rows,
        config,
        view="oracle_short_memory",
    )

    checkpoint_sha_after = _sha256_file(checkpoint)
    tokenizer_hash_after = tokenizer.vocabulary_hash()
    parent_state_after = language_model_state_sha256(parent_cpu)
    parent_logits_after = _parent_probe(parent_cpu, tokenizer, validation_rows[0])
    parent_checks = {
        "checkpoint_file_exact": checkpoint_sha_before == checkpoint_sha_after,
        "state_exact": parent_state_before == parent_state_after,
        "tokenizer_exact": tokenizer_hash_before == tokenizer_hash_after,
        "sample_logits_exact": torch.equal(parent_logits_before, parent_logits_after),
        "initial_short_prefix_exact": initial_short_prefix_exact,
    }
    total_setup_training_seconds = setup_seconds + float(training["training_seconds"])
    true_exact = int(true["exact_answer_count"])
    zero_exact = int(zero["exact_answer_count"])
    shuffled_exact = int(shuffled["exact_answer_count"])
    oracle_exact = int(oracle["exact_answer_count"])
    source_gain = float(true["exact_answer_accuracy"]) - max(
        float(zero["exact_answer_accuracy"]),
        float(shuffled["exact_answer_accuracy"]),
    )
    checks = {
        "minimum_true_exact_answers": true_exact >= config.minimum_true_exact_answers,
        "minimum_true_source_gain": source_gain >= config.minimum_true_source_gain,
        "maximum_shuffled_exact_answers": shuffled_exact
        <= config.maximum_shuffled_exact_answers,
        "minimum_oracle_exact_answers": oracle_exact
        >= config.minimum_oracle_exact_answers,
        "maximum_true_oracle_gap": oracle_exact - true_exact
        <= config.maximum_true_oracle_gap,
        "parameter_fraction": parameter_fraction <= config.maximum_parameter_fraction,
        "exact_optimizer_steps": training["optimizer_steps"] == config.optimizer_steps,
        "exact_position_budget": training["padded_source_positions"]
        == config.padded_source_position_budget,
        "complete_final_gradients": bool(
            training["final_gradients"]["all_trainable_tensors_nonzero"]
        ),
        "maximum_training_seconds": float(training["training_seconds"])
        <= config.maximum_training_seconds,
        "maximum_total_setup_training_seconds": total_setup_training_seconds
        <= config.maximum_total_setup_training_seconds,
        "parent_fidelity": all(parent_checks.values()),
    }

    checkpoint_evidence: dict[str, Any] = {
        "saved": False,
        "path": None,
        "sha256": None,
        "strict_tensor_reload": False,
        "strict_logit_reload": False,
    }
    behavioral_pass = all(checks.values())
    if behavioral_pass:
        probe_before = _controller_probe(
            parent, controller, tokenizer, validation_rows[0], config
        )
        candidate = Path(candidate_path)
        _atomic_torch_save(
            candidate,
            {
                "artifact_kind": "marulho_meta_gradient_memory_checkpoint",
                "surface": "marulho_meta_gradient_memory_checkpoint.v1",
                "owned_by_marulho": True,
                "external_llm_used": False,
                "parent_checkpoint_sha256": checkpoint_sha_before,
                "tokenizer_hash": tokenizer_hash_before,
                "configuration": asdict(config),
                "controller_state": {
                    name: value.detach().cpu()
                    for name, value in controller.state_dict().items()
                },
            },
        )
        candidate_sha = _sha256_file(candidate)
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
        reloaded = MetaGradientEpisodicMatrix(
            width=int(parent.config.state_dim),
            memory_heads=config.memory_heads,
            key_width_per_head=config.key_width_per_head,
            value_width_per_head=config.value_width_per_head,
            model_seed=config.model_seed,
        ).to(dtype=torch.bfloat16)
        reloaded.load_state_dict(dict(payload["controller_state"]), strict=True)
        current = controller.state_dict()
        tensor_exact = all(
            torch.equal(value.detach().cpu(), reloaded.state_dict()[name])
            for name, value in current.items()
        )
        reloaded = reloaded.to(device=runtime_device, dtype=torch.bfloat16)
        probe_after = _controller_probe(
            parent, reloaded, tokenizer, validation_rows[0], config
        )
        logit_exact = torch.equal(probe_before, probe_after)
        checkpoint_evidence = {
            "saved": True,
            "path": str(candidate),
            "sha256": candidate_sha,
            "strict_tensor_reload": tensor_exact,
            "strict_logit_reload": logit_exact,
        }
        del reloaded
    checkpoint_passed = bool(checkpoint_evidence["saved"]) and bool(
        checkpoint_evidence["strict_tensor_reload"]
    ) and bool(checkpoint_evidence["strict_logit_reload"])
    passed = behavioral_pass and checkpoint_passed
    if not passed:
        Path(candidate_path).unlink(missing_ok=True)
        checkpoint_evidence = {
            **checkpoint_evidence,
            "saved": False,
            "path": None,
            "sha256": None,
        }
    decision = ADVANCE_DECISION if passed else RETIRE_DECISION
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
            "validation_case_count": len(validation_rows),
            "train_title_count": len(train_titles),
            "validation_title_count": len(validation_titles),
            "title_intersection_count": len(train_titles & validation_titles),
            "prepared_tensor_sha256": prepared_sha,
            "schedule_sha256": schedule_sha,
            "cache_policy": "online_frozen_v39_no_persistent_hidden_cache",
            "persistent_cache_bytes": 0,
            "write_inputs_exclude_question": True,
            "write_inputs_exclude_answer": True,
            "write_inputs_exclude_span": True,
            "write_inputs_exclude_labels": True,
        },
        "architecture": {
            "controller_parameter_count": controller_parameter_count,
            "parent_parameter_count": parent_parameter_count,
            "controller_parameter_fraction": parameter_fraction,
            "fast_state_values_per_document": controller.fast_state_values_per_document,
            "initial_controller_state_sha256": sha256_json(
                {
                    name: hashlib.sha256(
                        value.contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                    for name, value in initial_state.items()
                }
            ),
            "fast_write": "one_exact_linear_reconstruction_gradient_step_from_zero",
            "source_value_owner": "frozen_v39_next_token_embeddings",
        },
        "setup": {
            "seconds": setup_seconds,
            "persistent_cache_bytes": 0,
        },
        "training": training,
        "views": {
            "untrained_true_memory": initial_true,
            "zero_memory": zero,
            "shuffled_memory": shuffled,
            "true_memory": true,
            "oracle_short_memory": oracle,
        },
        "parent": {
            "path": str(checkpoint),
            "checkpoint_sha256_before": checkpoint_sha_before,
            "checkpoint_sha256_after": checkpoint_sha_after,
            "state_sha256_before": parent_state_before,
            "state_sha256_after": parent_state_after,
            "tokenizer_hash_before": tokenizer_hash_before,
            "tokenizer_hash_after": tokenizer_hash_after,
            "metadata": parent_payload.get("metadata", {}),
            "checks": parent_checks,
        },
        "checkpoint": checkpoint_evidence,
        "runtime": {
            "total_setup_training_seconds": total_setup_training_seconds,
            "total_wall_seconds": time.perf_counter() - total_started,
            "peak_cuda_bytes": training["peak_cuda_bytes"],
        },
        "gate": {
            "passed": passed,
            "behavioral_checks": checks,
            "checkpoint_passed": checkpoint_passed,
            "observed": {
                "untrained_true_exact_answers": initial_true["exact_answer_count"],
                "zero_exact_answers": zero_exact,
                "shuffled_exact_answers": shuffled_exact,
                "true_exact_answers": true_exact,
                "oracle_exact_answers": oracle_exact,
                "true_source_gain": source_gain,
                "true_oracle_gap": oracle_exact - true_exact,
            },
            "thresholds": asdict(config),
        },
    }
    write_json_report_with_readme(output_path, report)
    print(
        f"[v60] decision={decision} zero={zero_exact} shuffled={shuffled_exact} "
        f"true={true_exact} oracle={oracle_exact}",
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
    run_v60(
        checkpoint_path=arguments.checkpoint,
        train_manifest_path=arguments.train_manifest,
        validation_manifest_path=arguments.validation_manifest,
        output_path=arguments.output,
        candidate_path=arguments.candidate,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
