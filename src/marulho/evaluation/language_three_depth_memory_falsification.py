"""V62 protected three-depth shared fast-memory falsifier."""

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


SURFACE = "marulho_three_depth_shared_memory_falsification.v1"
ARTIFACT_KIND = "marulho_three_depth_shared_memory_falsification"
ADVANCE_DECISION = "advance_v62_three_depth_memory_to_routed_multidocument_state"
ORACLE_ONLY_DECISION = "pivot_v62_full_source_compression_or_localization"
RETIRE_DECISION = "retire_v62_compressed_three_depth_fast_memory"
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
    "reports/language_scaling/three-depth-memory-v62-20m-20260812.json"
)
DEFAULT_CANDIDATE = Path(
    "reports/language_scaling/v62-three-depth-memory-qualified-20260812.pt"
)


@dataclass(frozen=True)
class V62Config:
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
    injection_layers: tuple[int, ...] = (2, 5, 8)
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
    maximum_parameter_fraction: float = 0.0125
    maximum_training_seconds: float = 1800.0
    maximum_total_setup_training_seconds: float = 2400.0
    data_seed: int = 62121
    model_seed: int = 62131
    precision: str = "bfloat16"
    execution_backend: str = "pytorch_eager"


@dataclass(frozen=True)
class PreparedCases:
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


def _tensor_sha256(value: torch.Tensor) -> str:
    flat = value.detach().cpu().contiguous().reshape(-1)
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()


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
        temporary.unlink(missing_ok=True)
    return output


def _question_prompt(row: Mapping[str, Any]) -> str:
    return f"Question: {str(row['question']).strip()}\nAnswer: "


def _prepare_source(
    source: str,
    tokenizer: LanguageTokenizer,
    config: V62Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    ids = tokenizer.encode(f"Context: {source}", add_bos=True, add_eos=True)
    capacity = config.source_chunk_length * config.source_chunk_count
    if len(ids) - 1 > capacity:
        raise ValueError(f"Source requires {len(ids) - 1} positions, above {capacity}")
    inputs: list[int] = []
    targets: list[int] = []
    masks: list[bool] = []
    for chunk_index in range(config.source_chunk_count):
        start = chunk_index * config.source_chunk_length
        segment = ids[start : start + config.source_chunk_length + 1]
        valid = max(0, len(segment) - 1)
        padding = config.source_chunk_length - valid
        inputs.extend(segment[:-1] + [int(tokenizer.pad_id)] * padding)
        targets.extend(segment[1:] + [int(tokenizer.pad_id)] * padding)
        masks.extend([True] * valid + [False] * padding)
    if len(inputs) != config.source_memory_positions:
        raise RuntimeError("V62 source preparation violated the frozen shape")
    return (
        torch.tensor(inputs, dtype=torch.long).view(
            config.source_chunk_count, config.source_chunk_length
        ),
        torch.tensor(targets, dtype=torch.long).view(
            config.source_chunk_count, config.source_chunk_length
        ),
        torch.tensor(masks, dtype=torch.bool).view(
            config.source_chunk_count, config.source_chunk_length
        ),
        sum(masks),
    )


def _prepare_query(
    row: Mapping[str, Any],
    tokenizer: LanguageTokenizer,
    config: V62Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    prompt_ids = tokenizer.encode(_question_prompt(row), add_eos=False)
    answer_ids = tokenizer.encode(
        str(tuple(row["answers"])[0]), add_bos=False, add_eos=True
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
        torch.tensor(inputs + [int(tokenizer.pad_id)] * padding, dtype=torch.long),
        torch.tensor(targets + [int(tokenizer.pad_id)] * padding, dtype=torch.long),
        torch.tensor(answer_mask + [False] * padding, dtype=torch.bool),
        sum(answer_mask),
    )


def prepare_cases(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: LanguageTokenizer,
    config: V62Config,
) -> PreparedCases:
    source_inputs: list[torch.Tensor] = []
    source_targets: list[torch.Tensor] = []
    source_masks: list[torch.Tensor] = []
    query_inputs: list[torch.Tensor] = []
    query_targets: list[torch.Tensor] = []
    answer_masks: list[torch.Tensor] = []
    evidence: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        source_input, source_target, source_mask, source_positions = _prepare_source(
            str(row["source_text"]), tokenizer, config
        )
        query_input, query_target, answer_mask, answer_positions = _prepare_query(
            row, tokenizer, config
        )
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
                "source_text": str(row["source_text"]),
                "source_positions": source_positions,
                "answer_positions": answer_positions,
            }
        )
    return PreparedCases(
        source_input_ids=torch.stack(source_inputs),
        source_target_ids=torch.stack(source_targets),
        source_mask=torch.stack(source_masks),
        query_input_ids=torch.stack(query_inputs),
        query_target_ids=torch.stack(query_targets),
        answer_mask=torch.stack(answer_masks),
        cases=tuple(evidence),
    )


class ProtectedThreeDepthMemory(nn.Module):
    """One shared fast matrix read before three frozen cortex blocks."""

    surface = "marulho_protected_three_depth_memory.v1"

    def __init__(
        self,
        *,
        width: int,
        state_layers: int,
        memory_heads: int,
        key_width_per_head: int,
        value_width_per_head: int,
        injection_layers: Sequence[int],
        model_seed: int,
    ) -> None:
        super().__init__()
        torch.manual_seed(int(model_seed))
        self.width = int(width)
        self.state_layers = int(state_layers)
        self.memory_heads = int(memory_heads)
        self.key_width_per_head = int(key_width_per_head)
        self.value_width_per_head = int(value_width_per_head)
        self.injection_layers = tuple(int(value) for value in injection_layers)
        if len(set(self.injection_layers)) != len(self.injection_layers):
            raise ValueError("V62 injection layers must be unique")
        if not self.injection_layers or min(self.injection_layers) < 0:
            raise ValueError("V62 requires non-negative injection layers")
        if max(self.injection_layers) >= self.state_layers:
            raise ValueError("V62 injection layer exceeds parent depth")
        if self.memory_heads * self.value_width_per_head != self.width:
            raise ValueError("Memory value heads must concatenate to model width")
        key_width = self.memory_heads * self.key_width_per_head
        self.key_projection = nn.Linear(self.width, key_width, bias=False)
        self.query_projections = nn.ModuleList(
            nn.Linear(self.width, key_width, bias=False)
            for _ in self.injection_layers
        )
        self.token_gate_projections = nn.ModuleList(
            nn.Linear(self.width, self.memory_heads, bias=False)
            for _ in self.injection_layers
        )
        self.read_projection = nn.Linear(self.width, self.width, bias=False)
        self.log_write_rates = nn.Parameter(torch.full((self.memory_heads,), 3.0))
        self.site_gate_logits = nn.Parameter(
            torch.full((len(self.injection_layers),), -2.0)
        )
        nn.init.normal_(self.key_projection.weight, mean=0.0, std=0.02)
        for projection in self.query_projections:
            nn.init.normal_(projection.weight, mean=0.0, std=0.02)
        for projection in self.token_gate_projections:
            nn.init.zeros_(projection.weight)
        nn.init.eye_(self.read_projection.weight)
        self._site_by_layer = {
            layer_index: site_index
            for site_index, layer_index in enumerate(self.injection_layers)
        }

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
        memory = torch.einsum("bshk,bshv->bhkv", keys, values)
        denominator = mask.sum(dim=1).clamp_min(1.0)[:, None, None, None]
        rates = F.softplus(self.log_write_rates).view(1, -1, 1, 1)
        return memory * rates / denominator

    def inject(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        site_index: int,
    ) -> torch.Tensor:
        batch, positions, _ = hidden.shape
        queries = self.query_projections[site_index](hidden).view(
            batch,
            positions,
            self.memory_heads,
            self.key_width_per_head,
        )
        queries = F.normalize(queries.float(), dim=-1).to(hidden.dtype)
        read_heads = torch.einsum("bthk,bhkv->bthv", queries, memory)
        token_gates = torch.sigmoid(
            self.token_gate_projections[site_index](hidden).float()
        ).to(hidden.dtype)
        read = (read_heads * token_gates.unsqueeze(-1)).reshape(
            batch, positions, self.width
        )
        projected = self.read_projection(read)
        site_gate = torch.sigmoid(self.site_gate_logits[site_index])
        return hidden + site_gate * projected

    def forward_parent(
        self,
        parent: MarulhoLanguageModel,
        input_ids: torch.Tensor,
        memory: torch.Tensor | None = None,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        active: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if input_ids.ndim != 2:
            raise ValueError("V62 parent forward expects [batch,time] token IDs")
        runtime_ids = input_ids.to(device=parent.device, dtype=torch.long)
        inputs = parent.token_embedding(runtime_ids)
        block = parent.state_block
        batch_size, time_steps, _ = inputs.shape
        if int(time_steps) > block.context_length and state is None:
            raise ValueError("V62 query exceeds reconstructed parent context")
        if active and memory is None:
            raise ValueError("Active V62 forward requires a fast memory")
        if memory is not None and int(memory.shape[0]) != int(batch_size):
            raise ValueError("V62 memory batch does not match query batch")
        current_state = (
            block.initial_state(
                int(batch_size), device=inputs.device, dtype=inputs.dtype
            )
            if state is None
            else state
        )
        position_value = current_state.get("position")
        position_offset = (
            position_value.to(device=inputs.device, dtype=torch.long)
            if isinstance(position_value, torch.Tensor)
            else torch.zeros((), device=inputs.device, dtype=torch.long)
        )
        hidden = block.input_projection(inputs)
        next_state: dict[str, torch.Tensor] = {
            "position": position_offset + int(time_steps)
        }
        for layer_index, layer in enumerate(block.layers):
            site_index = self._site_by_layer.get(layer_index)
            if active and site_index is not None:
                assert memory is not None
                hidden = self.inject(hidden, memory, site_index)
            hidden, next_key, next_value = layer(
                hidden,
                past_key=current_state.get(f"layer_{layer_index}_key"),
                past_value=current_state.get(f"layer_{layer_index}_value"),
                position_offset=position_offset,
            )
            next_state[f"layer_{layer_index}_key"] = next_key.detach()
            next_state[f"layer_{layer_index}_value"] = next_value.detach()
        return block.output_norm(hidden), next_state


def _schedule_indices(case_count: int, config: V62Config) -> tuple[torch.Tensor, str]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.data_seed)
    schedule = torch.cat(
        [torch.randperm(case_count, generator=generator) for _ in range(config.epochs)]
    )
    return schedule, hashlib.sha256(schedule.numpy().tobytes()).hexdigest()


def _learning_rate(config: V62Config, step: int) -> float:
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
    prepared: PreparedCases,
    indices: torch.Tensor,
    config: V62Config,
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
    return hidden, target_embeddings, mask.reshape(len(indices), -1)


def _gradient_audit(model: nn.Module) -> dict[str, Any]:
    by_parameter = {
        name: bool(
            parameter.grad is not None
            and torch.count_nonzero(parameter.grad).item() > 0
        )
        for name, parameter in model.named_parameters()
    }
    return {
        "tensor_count": len(by_parameter),
        "nonzero_tensor_count": sum(by_parameter.values()),
        "all_trainable_tensors_nonzero": bool(by_parameter)
        and all(by_parameter.values()),
        "by_parameter": by_parameter,
    }


def train_controller(
    parent: MarulhoLanguageModel,
    controller: ProtectedThreeDepthMemory,
    prepared: PreparedCases,
    schedule: torch.Tensor,
    config: V62Config,
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
        query_inputs = prepared.query_input_ids.index_select(0, indices).to(parent.device)
        query_targets = prepared.query_target_ids.index_select(0, indices).to(parent.device)
        answer_mask = prepared.answer_mask.index_select(0, indices).to(parent.device)
        learning_rate = _learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        memory = controller.write(source_hidden, target_embeddings, source_mask)
        if not torch.isfinite(memory).all():
            raise FloatingPointError("V62 produced non-finite fast memory")
        read_hidden, _ = controller.forward_parent(
            parent, query_inputs, memory, active=True
        )
        logits = parent.lm_head(read_hidden)
        loss = F.cross_entropy(logits[answer_mask].float(), query_targets[answer_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), config.gradient_clip)
        optimizer.step()
        answer_positions += int(answer_mask.sum())
        if (step + 1) % 256 == 0 or step + 1 == config.optimizer_steps:
            final_loss = float(loss.detach())
            gates = torch.sigmoid(controller.site_gate_logits.detach()).cpu().tolist()
            print(
                f"[v62] train {step + 1}/{config.optimizer_steps} "
                f"loss={final_loss:.4f} gates={gates} lr={learning_rate:.3g}",
                flush=True,
            )
    if parent.device.type == "cuda":
        torch.cuda.synchronize(parent.device)
    elapsed = max(time.perf_counter() - started, 1.0e-9)
    peak = (
        int(torch.cuda.max_memory_allocated(parent.device))
        if parent.device.type == "cuda"
        else 0
    )
    gradients = _gradient_audit(controller)
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
        "final_site_gates": torch.sigmoid(
            controller.site_gate_logits.detach().float()
        ).cpu().tolist(),
    }


@torch.no_grad()
def _memory_for_source(
    parent: MarulhoLanguageModel,
    controller: ProtectedThreeDepthMemory,
    source: str,
    tokenizer: LanguageTokenizer,
    config: V62Config,
) -> torch.Tensor:
    source_input, source_target, source_mask, _ = _prepare_source(
        source, tokenizer, config
    )
    hidden = parent._forward_hidden(
        source_input.to(parent.device), collect_telemetry=False
    )["hidden"].reshape(1, config.source_memory_positions, -1)
    embeddings = parent.token_embedding(
        source_target.to(parent.device).reshape(1, config.source_memory_positions)
    )
    memory = controller.write(
        hidden, embeddings, source_mask.to(parent.device).reshape(1, -1)
    )
    if not torch.isfinite(memory).all():
        raise FloatingPointError("V62 evaluation produced non-finite fast memory")
    return memory


@torch.no_grad()
def _generate_with_memory(
    parent: MarulhoLanguageModel,
    controller: ProtectedThreeDepthMemory,
    tokenizer: LanguageTokenizer,
    row: Mapping[str, Any],
    memory: torch.Tensor | None,
    config: V62Config,
    *,
    active: bool,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(_question_prompt(row), add_eos=False)
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=parent.device).unsqueeze(0)
    hidden, state = controller.forward_parent(
        parent, prompt, memory, active=active
    )
    current_hidden = hidden[:, -1:, :]
    generated: list[int] = []
    for _ in range(config.generation_tokens):
        logits = parent.lm_head(current_hidden[:, -1, :])
        generated_tensor = torch.tensor(
            generated, dtype=torch.long, device=parent.device
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
        current_hidden, state = controller.forward_parent(
            parent,
            torch.tensor([[next_id]], dtype=torch.long, device=parent.device),
            memory,
            state,
            active=active,
        )
    continuation = tokenizer.decode(generated)
    normalized = _normalized(continuation)
    accepted = {_normalized(value) for value in row["answers"]}
    return {
        "case_id": str(row["case_id"]),
        "title": str(row["title"]),
        "answers": [str(value) for value in row["answers"]],
        "continuation_text": continuation,
        "continuation_ids": generated,
        "exact_answer_match": normalized in accepted,
        "contains_accepted_answer": any(
            answer and answer in normalized for answer in accepted
        ),
    }


@torch.no_grad()
def evaluate_view(
    parent: MarulhoLanguageModel,
    controller: ProtectedThreeDepthMemory,
    tokenizer: LanguageTokenizer,
    rows: Sequence[Mapping[str, Any]],
    config: V62Config,
    *,
    view: str,
) -> dict[str, Any]:
    controller.eval()
    started = time.perf_counter()
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if view == "inactive":
            memory = None
            active = False
        elif view == "true_memory":
            memory = _memory_for_source(
                parent, controller, str(row["source_text"]), tokenizer, config
            )
            active = True
        elif view == "oracle_short_memory":
            memory = _memory_for_source(
                parent, controller, str(row["oracle_source_text"]), tokenizer, config
            )
            active = True
        elif view == "shuffled_memory":
            wrong = rows[(index + 1) % len(rows)]
            memory = _memory_for_source(
                parent, controller, str(wrong["source_text"]), tokenizer, config
            )
            active = True
        else:
            raise ValueError(f"Unknown V62 view {view!r}")
        output_rows.append(
            _generate_with_memory(
                parent,
                controller,
                tokenizer,
                row,
                memory,
                config,
                active=active,
            )
        )
        if (index + 1) % 64 == 0:
            exact = sum(bool(item["exact_answer_match"]) for item in output_rows)
            print(f"[v62] {view} {index + 1}/{len(rows)} exact={exact}", flush=True)
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


@torch.no_grad()
def _controller_probe(
    parent: MarulhoLanguageModel,
    controller: ProtectedThreeDepthMemory,
    tokenizer: LanguageTokenizer,
    row: Mapping[str, Any],
    config: V62Config,
) -> torch.Tensor:
    memory = _memory_for_source(
        parent, controller, str(row["source_text"]), tokenizer, config
    )
    ids = tokenizer.encode(_question_prompt(row), add_eos=False)
    hidden, _ = controller.forward_parent(
        parent,
        torch.tensor([ids], dtype=torch.long, device=parent.device),
        memory,
        active=True,
    )
    return parent.lm_head(hidden[:, -1:, :]).detach().cpu()


@torch.no_grad()
def _inactive_parity(
    parent: MarulhoLanguageModel,
    controller: ProtectedThreeDepthMemory,
    tokenizer: LanguageTokenizer,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    ids = tokenizer.encode(_question_prompt(row), add_eos=False)
    probe = torch.tensor([ids], dtype=torch.long, device=parent.device)
    ordinary = parent._forward_hidden(probe, collect_telemetry=False)
    custom_hidden, custom_state = controller.forward_parent(
        parent, probe, active=False
    )
    state_exact = set(ordinary["state"]) == set(custom_state) and all(
        torch.equal(ordinary["state"][name], custom_state[name])
        for name in custom_state
    )
    return {
        "hidden_exact": torch.equal(ordinary["hidden"], custom_hidden),
        "logits_exact": torch.equal(
            parent.lm_head(ordinary["hidden"]), parent.lm_head(custom_hidden)
        ),
        "state_exact": state_exact,
        "state_key_count": len(custom_state),
    }


def run_v62(
    *,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    train_manifest_path: str | Path = DEFAULT_TRAIN_MANIFEST,
    validation_manifest_path: str | Path = DEFAULT_VALIDATION_MANIFEST,
    output_path: str | Path = DEFAULT_OUTPUT,
    candidate_path: str | Path = DEFAULT_CANDIDATE,
    device: str = "cuda",
) -> dict[str, Any]:
    config = V62Config()
    runtime_device = torch.device(device)
    if runtime_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V62 is frozen as a CUDA experiment")
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
        raise ValueError("V62 runtime tokenizer differs from immutable parent")
    parent = MarulhoLanguageModel(
        replace(runtime_parent.config, transformer_context_length=config.context_length)
    )
    parent.load_state_dict(runtime_parent.state_dict(), strict=True)
    parity_ids = tokenizer.encode(
        _question_prompt(validation_manifest["cases"][0]), add_eos=False
    )[:72]
    parity_probe = torch.tensor([parity_ids], dtype=torch.long)
    runtime_parent.eval()
    parent.eval()
    with torch.no_grad():
        original_logits = runtime_parent(parity_probe, collect_telemetry=False)["logits"]
        extended_logits = parent(parity_probe, collect_telemetry=False)["logits"]
    initial_short_prefix_exact = torch.equal(original_logits, extended_logits)
    if not initial_short_prefix_exact:
        raise RuntimeError("V62 context-96 reconstruction changed a short prefix")
    del runtime_parent, original_logits, extended_logits
    parent = parent.to(device=runtime_device, dtype=torch.bfloat16)
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    parent_parameter_count = sum(parameter.numel() for parameter in parent.parameters())

    setup_started = time.perf_counter()
    print("[v62] preparing immutable cases and schedule", flush=True)
    train_cases = prepare_cases(tuple(train_manifest["cases"]), tokenizer, config)
    validation_rows = tuple(dict(row) for row in validation_manifest["cases"])
    if len(train_cases) != 8192 or len(validation_rows) != 256:
        raise ValueError("V62 requires the exact V57 8192/256 boundary")
    train_titles = {str(row["title"]) for row in train_manifest["cases"]}
    validation_titles = {str(row["title"]) for row in validation_rows}
    if train_titles & validation_titles:
        raise ValueError("V62 train and validation titles overlap")
    schedule, schedule_sha = _schedule_indices(len(train_cases), config)
    if int(schedule.numel()) != config.optimizer_steps * config.batch_size:
        raise RuntimeError("V62 schedule violates the frozen update count")
    prepared_sha = sha256_json(
        {
            name: hashlib.sha256(value.numpy().tobytes()).hexdigest()
            for name, value in {
                "source_input_ids": train_cases.source_input_ids,
                "source_target_ids": train_cases.source_target_ids,
                "source_mask": train_cases.source_mask,
                "query_input_ids": train_cases.query_input_ids,
                "query_target_ids": train_cases.query_target_ids,
                "answer_mask": train_cases.answer_mask,
            }.items()
        }
    )
    setup_seconds = time.perf_counter() - setup_started

    controller = ProtectedThreeDepthMemory(
        width=int(parent.config.state_dim),
        state_layers=int(parent.config.state_layers),
        memory_heads=config.memory_heads,
        key_width_per_head=config.key_width_per_head,
        value_width_per_head=config.value_width_per_head,
        injection_layers=config.injection_layers,
        model_seed=config.model_seed,
    ).to(device=runtime_device, dtype=torch.bfloat16)
    controller_parameter_count = sum(
        parameter.numel() for parameter in controller.parameters()
    )
    parameter_fraction = controller_parameter_count / parent_parameter_count
    initial_state_sha = sha256_json(
        {name: _tensor_sha256(value) for name, value in controller.state_dict().items()}
    )
    inactive_parity_before = _inactive_parity(
        parent, controller, tokenizer, validation_rows[0]
    )
    if not all(
        bool(inactive_parity_before[key])
        for key in ("hidden_exact", "logits_exact", "state_exact")
    ):
        raise RuntimeError("V62 inactive forward is not exact before training")
    initial_true = evaluate_view(
        parent, controller, tokenizer, validation_rows, config, view="true_memory"
    )

    print("[v62] training three-depth shared memory", flush=True)
    training = train_controller(parent, controller, train_cases, schedule, config)
    inactive_parity_after = _inactive_parity(
        parent, controller, tokenizer, validation_rows[0]
    )
    print("[v62] evaluating learned controls", flush=True)
    inactive = evaluate_view(
        parent, controller, tokenizer, validation_rows, config, view="inactive"
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
        "inactive_hidden_exact_before": bool(inactive_parity_before["hidden_exact"]),
        "inactive_logits_exact_before": bool(inactive_parity_before["logits_exact"]),
        "inactive_state_exact_before": bool(inactive_parity_before["state_exact"]),
        "inactive_hidden_exact_after": bool(inactive_parity_after["hidden_exact"]),
        "inactive_logits_exact_after": bool(inactive_parity_after["logits_exact"]),
        "inactive_state_exact_after": bool(inactive_parity_after["state_exact"]),
    }
    total_setup_training_seconds = setup_seconds + float(training["training_seconds"])
    true_exact = int(true["exact_answer_count"])
    inactive_exact = int(inactive["exact_answer_count"])
    shuffled_exact = int(shuffled["exact_answer_count"])
    oracle_exact = int(oracle["exact_answer_count"])
    source_gain = float(true["exact_answer_accuracy"]) - max(
        float(inactive["exact_answer_accuracy"]),
        float(shuffled["exact_answer_accuracy"]),
    )
    checks = {
        "minimum_true_exact_answers": true_exact >= config.minimum_true_exact_answers,
        "minimum_true_source_gain": source_gain >= config.minimum_true_source_gain,
        "maximum_shuffled_exact_answers": shuffled_exact
        <= config.maximum_shuffled_exact_answers,
        "minimum_oracle_exact_answers": oracle_exact >= config.minimum_oracle_exact_answers,
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
        "parent_and_inactive_fidelity": all(parent_checks.values()),
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
                "artifact_kind": "marulho_three_depth_memory_checkpoint",
                "surface": "marulho_three_depth_memory_checkpoint.v1",
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
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
        reloaded = ProtectedThreeDepthMemory(
            width=int(parent.config.state_dim),
            state_layers=int(parent.config.state_layers),
            memory_heads=config.memory_heads,
            key_width_per_head=config.key_width_per_head,
            value_width_per_head=config.value_width_per_head,
            injection_layers=config.injection_layers,
            model_seed=config.model_seed,
        ).to(dtype=torch.bfloat16)
        reloaded.load_state_dict(dict(payload["controller_state"]), strict=True)
        tensor_exact = all(
            torch.equal(value.detach().cpu(), reloaded.state_dict()[name])
            for name, value in controller.state_dict().items()
        )
        reloaded = reloaded.to(device=runtime_device, dtype=torch.bfloat16)
        probe_after = _controller_probe(
            parent, reloaded, tokenizer, validation_rows[0], config
        )
        checkpoint_evidence = {
            "saved": True,
            "path": str(candidate),
            "sha256": _sha256_file(candidate),
            "strict_tensor_reload": tensor_exact,
            "strict_logit_reload": torch.equal(probe_before, probe_after),
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
    if passed:
        decision = ADVANCE_DECISION
    elif oracle_exact >= config.minimum_oracle_exact_answers:
        decision = ORACLE_ONLY_DECISION
    else:
        decision = RETIRE_DECISION

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
            "initial_controller_state_sha256": initial_state_sha,
            "fast_write": "v60_one_exact_linear_reconstruction_step",
            "injection_layers": list(config.injection_layers),
            "shared_read_projection": True,
            "input_dependent_site_gates": True,
        },
        "setup": {"seconds": setup_seconds, "persistent_cache_bytes": 0},
        "training": training,
        "views": {
            "untrained_true_memory": initial_true,
            "inactive": inactive,
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
            "inactive_parity_before": inactive_parity_before,
            "inactive_parity_after": inactive_parity_after,
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
                "inactive_exact_answers": inactive_exact,
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
        f"[v62] decision={decision} inactive={inactive_exact} "
        f"shuffled={shuffled_exact} true={true_exact} oracle={oracle_exact}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--train-manifest", default=str(DEFAULT_TRAIN_MANIFEST))
    parser.add_argument(
        "--validation-manifest", default=str(DEFAULT_VALIDATION_MANIFEST)
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    run_v62(
        checkpoint_path=arguments.checkpoint,
        train_manifest_path=arguments.train_manifest,
        validation_manifest_path=arguments.validation_manifest,
        output_path=arguments.output,
        candidate_path=arguments.candidate,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
