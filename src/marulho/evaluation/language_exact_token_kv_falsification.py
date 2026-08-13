"""Run V63 protected exact-token adaptive KV-memory falsification."""

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
from marulho.evaluation.language_source_grounding import (
    load_squad_grounding_manifest,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    MarulhoLanguageModel,
    _apply_decode_controls,
    language_model_state_sha256,
    load_language_model_checkpoint,
)
from marulho.training.language_transformer import _apply_rotary


SURFACE = "marulho_exact_token_kv_falsification.v1"
ARTIFACT_KIND = "marulho_exact_token_kv_falsification"
ADVANCE_DECISION = "advance_v63_exact_token_kv_to_selective_archive"
ORACLE_ONLY_DECISION = "pivot_v63_full_source_localization"
RETIRE_DECISION = "retire_v39_protected_memory_adaptation"
INVALID_DECISION = "invalid_v63_exact_token_kv_contract"
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
    "reports/language_scaling/exact-token-kv-v63-20m-20260812.json"
)
DEFAULT_CANDIDATE = Path(
    "reports/language_scaling/v63-exact-token-kv-qualified-20260812.pt"
)


@dataclass(frozen=True)
class V63Config:
    context_length: int = 320
    batch_size: int = 32
    epochs: int = 8
    optimizer_steps: int = 2_048
    padded_position_budget: int = 20_971_520
    correction_scale: float = 0.25
    learning_rate: float = 3.0e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
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
    maximum_training_seconds: float = 1_800.0
    maximum_total_setup_training_seconds: float = 2_400.0
    data_seed: int = 63_121
    model_seed: int = 63_131
    precision: str = "bfloat16_parent_fp32_controller"
    execution_backend: str = "pytorch_eager_sdpa"


@dataclass(frozen=True)
class PreparedCases:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    source_mask: torch.Tensor
    answer_mask: torch.Tensor
    cases: tuple[dict[str, Any], ...]
    boundary_evidence: tuple[dict[str, Any], ...]

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])


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


def _prompt_fields(view: str) -> tuple[str, str]:
    fields = {
        "true": ("causal_prompt", "source_text"),
        "oracle": ("oracle_causal_prompt", "oracle_source_text"),
        "shuffled": ("mismatched_prompt", "mismatched_source_text"),
    }
    try:
        return fields[view]
    except KeyError as error:
        raise ValueError(f"Unknown V63 active view {view!r}") from error


def _source_mask_for_prompt(
    row: Mapping[str, Any],
    tokenizer: LanguageTokenizer,
    *,
    view: str,
) -> tuple[list[int], torch.Tensor, dict[str, Any]]:
    prompt_field, source_field = _prompt_fields(view)
    prompt = str(row[prompt_field])
    source = str(row[source_field])
    source_prefix = f"Context: {source}"
    if not prompt.startswith(source_prefix):
        raise ValueError(f"Case {row.get('case_id')} has a non-exact source prefix")
    source_end = len(source_prefix)
    canonical_suffix = str(row["question_only_prompt"]).strip()
    observed_suffix = prompt[source_end:].strip()
    if observed_suffix != canonical_suffix:
        raise ValueError(f"Case {row.get('case_id')} has a mismatched QA suffix")
    ids, offsets = tokenizer.encode_with_offsets(
        prompt,
        add_bos=True,
        add_eos=False,
    )
    if len(ids) != len(offsets):
        raise ValueError("Tokenizer IDs and offsets have different lengths")
    crossing = [
        index
        for index, (start, end) in enumerate(offsets)
        if int(start) < source_end < int(end)
    ]
    if crossing:
        raise ValueError(f"Case {row.get('case_id')} has a crossing source token")
    source_mask = torch.tensor(
        [int(end) > 0 and int(end) <= source_end for _start, end in offsets],
        dtype=torch.bool,
    )
    source_indices = torch.nonzero(source_mask, as_tuple=False).flatten()
    if not int(source_indices.numel()):
        raise ValueError(f"Case {row.get('case_id')} has no source tokens")
    final_source_index = int(source_indices[-1])
    if int(offsets[final_source_index][1]) != source_end:
        raise ValueError(f"Case {row.get('case_id')} lacks an exact source boundary")
    later_offsets = [
        (int(start), int(end))
        for start, end in offsets[final_source_index + 1 :]
        if int(end) > int(start)
    ]
    if not later_offsets or later_offsets[0][0] != source_end:
        raise ValueError(f"Case {row.get('case_id')} suffix is not boundary-adjacent")
    return ids, source_mask, {
        "case_id": str(row["case_id"]),
        "view": view,
        "prompt_field": prompt_field,
        "source_field": source_field,
        "prompt_token_count": len(ids),
        "source_token_count": int(source_mask.sum()),
        "source_character_end": source_end,
        "final_source_token_end": int(offsets[final_source_index][1]),
        "first_suffix_token_start": later_offsets[0][0],
        "delimiter_normalized_suffix_exact": True,
        "token_boundary_exact": True,
        "crossing_token_count": 0,
    }


def prepare_cases(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: LanguageTokenizer,
    config: V63Config,
    *,
    view: str = "true",
) -> PreparedCases:
    input_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    source_rows: list[torch.Tensor] = []
    answer_rows: list[torch.Tensor] = []
    cases: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        prefix, source_mask, boundary = _source_mask_for_prompt(
            row,
            tokenizer,
            view=view,
        )
        answer = tokenizer.encode(
            str(tuple(row["answers"])[0]),
            add_bos=False,
            add_eos=True,
        )
        ids = prefix + answer
        input_count = len(ids) - 1
        if input_count > config.context_length:
            raise ValueError(
                f"Case {row.get('case_id')} requires {input_count} positions"
            )
        padding = config.context_length - input_count
        inputs = ids[:-1] + [int(tokenizer.pad_id)] * padding
        targets = ids[1:] + [int(tokenizer.pad_id)] * padding
        answer_start = len(prefix) - 1
        answer_mask = torch.zeros(config.context_length, dtype=torch.bool)
        answer_mask[answer_start:input_count] = True
        padded_source = torch.zeros(config.context_length, dtype=torch.bool)
        padded_source[: len(prefix)] = source_mask
        input_rows.append(torch.tensor(inputs, dtype=torch.long))
        target_rows.append(torch.tensor(targets, dtype=torch.long))
        source_rows.append(padded_source)
        answer_rows.append(answer_mask)
        cases.append(row)
        boundaries.append(
            {
                **boundary,
                "input_position_count": input_count,
                "answer_target_count": int(answer_mask.sum()),
                "right_padding_count": padding,
                "right_padding_only_after_eos": True,
                "answer_mask_includes_eos": True,
            }
        )
    if not cases:
        raise ValueError("V63 requires at least one prepared case")
    return PreparedCases(
        input_ids=torch.stack(input_rows),
        target_ids=torch.stack(target_rows),
        source_mask=torch.stack(source_rows),
        answer_mask=torch.stack(answer_rows),
        cases=tuple(cases),
        boundary_evidence=tuple(boundaries),
    )


class ExactTokenKVController(nn.Module):
    """Bounded per-layer/head corrections to exact source-token keys and values."""

    surface = "marulho_exact_token_kv_controller.v1"

    def __init__(
        self,
        *,
        state_layers: int,
        attention_heads: int,
        head_dim: int,
        correction_scale: float,
        model_seed: int,
    ) -> None:
        super().__init__()
        torch.manual_seed(int(model_seed))
        self.state_layers = int(state_layers)
        self.attention_heads = int(attention_heads)
        self.head_dim = int(head_dim)
        self.correction_scale = float(correction_scale)
        shape = (
            self.state_layers,
            self.attention_heads,
            self.head_dim,
            self.head_dim,
        )
        self.key_corrections = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        self.value_corrections = nn.Parameter(torch.zeros(shape, dtype=torch.float32))

    @property
    def bounded_scale(self) -> float:
        return self.correction_scale / math.sqrt(float(self.head_dim))

    @property
    def correction_matrix_count(self) -> int:
        return self.state_layers * self.attention_heads * 2

    def _correct(
        self,
        value: torch.Tensor,
        source_mask: torch.Tensor,
        matrices: torch.Tensor,
    ) -> torch.Tensor:
        correction = torch.einsum(
            "bhtd,hde->bhte",
            value.float(),
            torch.tanh(matrices),
        )
        correction = correction * self.bounded_scale
        correction = correction * source_mask[:, None, :, None].float()
        return value + correction.to(dtype=value.dtype)

    def forward_parent(
        self,
        parent: MarulhoLanguageModel,
        input_ids: torch.Tensor,
        source_mask: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if input_ids.ndim != 2 or source_mask.shape != input_ids.shape:
            raise ValueError("V63 expects matching [batch,time] IDs and source mask")
        runtime_ids = input_ids.to(device=parent.device, dtype=torch.long)
        runtime_mask = source_mask.to(device=parent.device, dtype=torch.bool)
        inputs = parent.token_embedding(runtime_ids)
        block = parent.state_block
        batch_size, time_steps, _ = inputs.shape
        if int(time_steps) > block.context_length and state is None:
            raise ValueError("V63 sequence exceeds reconstructed parent context")
        current_state = (
            block.initial_state(
                int(batch_size),
                device=inputs.device,
                dtype=inputs.dtype,
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
            attention_module = layer.attention
            normalized = layer.attention_norm(hidden)
            query, key, current_value = attention_module.qkv(normalized).chunk(
                3, dim=-1
            )
            query = attention_module._heads(query)
            key = attention_module._heads(key)
            current_value = attention_module._heads(current_value)
            positions = torch.arange(int(time_steps), device=inputs.device)
            positions = positions + position_offset
            query, key = _apply_rotary(query, key, positions)
            key = self._correct(
                key,
                runtime_mask,
                self.key_corrections[layer_index],
            )
            current_value = self._correct(
                current_value,
                runtime_mask,
                self.value_corrections[layer_index],
            )

            past_key = current_state.get(f"layer_{layer_index}_key")
            past_value = current_state.get(f"layer_{layer_index}_value")
            usable_past_key: torch.Tensor | None = None
            usable_past_value: torch.Tensor | None = None
            if (
                past_key is not None
                and past_value is not None
                and int(past_key.shape[2]) > 0
            ):
                keep_past = max(0, attention_module.context_length - int(time_steps))
                if keep_past > 0:
                    usable_past_key = past_key[:, :, -keep_past:].to(
                        device=inputs.device,
                        dtype=inputs.dtype,
                    )
                    usable_past_value = past_value[:, :, -keep_past:].to(
                        device=inputs.device,
                        dtype=inputs.dtype,
                    )
            if usable_past_key is None:
                full_key = key
                full_value = current_value
                past_length = 0
            else:
                full_key = torch.cat((usable_past_key, key), dim=2)
                full_value = torch.cat((usable_past_value, current_value), dim=2)
                past_length = int(usable_past_key.shape[2])
            if past_length == 0:
                attended = F.scaled_dot_product_attention(
                    query,
                    full_key,
                    full_value,
                    dropout_p=(
                        attention_module.dropout if attention_module.training else 0.0
                    ),
                    is_causal=True,
                )
            elif int(time_steps) == 1:
                attended = F.scaled_dot_product_attention(
                    query,
                    full_key,
                    full_value,
                    dropout_p=(
                        attention_module.dropout if attention_module.training else 0.0
                    ),
                    is_causal=False,
                )
            else:
                key_positions = torch.arange(
                    int(full_key.shape[2]),
                    device=inputs.device,
                ).unsqueeze(0)
                query_limits = past_length + torch.arange(
                    int(time_steps), device=inputs.device
                ).unsqueeze(1)
                causal_mask = key_positions <= query_limits
                attended = F.scaled_dot_product_attention(
                    query,
                    full_key,
                    full_value,
                    attn_mask=causal_mask,
                    dropout_p=(
                        attention_module.dropout if attention_module.training else 0.0
                    ),
                    is_causal=False,
                )
            attended = attended.transpose(1, 2).contiguous().view(
                int(batch_size),
                int(time_steps),
                attention_module.width,
            )
            attention_output = attention_module.output(attended)
            hidden = hidden + layer.dropout(attention_output)
            gate, up = layer.gate_up(layer.mlp_norm(hidden)).chunk(2, dim=-1)
            hidden = hidden + layer.dropout(layer.down(F.silu(gate) * up))
            next_state[f"layer_{layer_index}_key"] = full_key.detach()
            next_state[f"layer_{layer_index}_value"] = full_value.detach()
        return block.output_norm(hidden), next_state


def _schedule_indices(case_count: int, config: V63Config) -> tuple[torch.Tensor, str]:
    generator = torch.Generator(device="cpu").manual_seed(config.data_seed)
    schedule = torch.cat(
        [torch.randperm(case_count, generator=generator) for _ in range(config.epochs)]
    )
    return schedule, hashlib.sha256(schedule.numpy().tobytes()).hexdigest()


def _learning_rate(config: V63Config, step: int) -> float:
    warmup = max(1, int(round(config.optimizer_steps * config.warmup_fraction)))
    if step < warmup:
        return config.learning_rate * float(step + 1) / float(warmup)
    progress = float(step - warmup) / float(
        max(1, config.optimizer_steps - warmup - 1)
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    fraction = config.minimum_learning_rate_fraction + (
        1.0 - config.minimum_learning_rate_fraction
    ) * cosine
    return config.learning_rate * fraction


def _gradient_audit(controller: ExactTokenKVController) -> dict[str, Any]:
    by_parameter = {
        name: bool(
            parameter.grad is not None
            and torch.count_nonzero(parameter.grad).item() > 0
        )
        for name, parameter in controller.named_parameters()
    }
    by_matrix: dict[str, bool] = {}
    for kind, parameter in (
        ("key", controller.key_corrections),
        ("value", controller.value_corrections),
    ):
        gradient = parameter.grad
        for layer_index in range(controller.state_layers):
            for head_index in range(controller.attention_heads):
                name = f"{kind}.layer_{layer_index}.head_{head_index}"
                by_matrix[name] = bool(
                    gradient is not None
                    and torch.count_nonzero(
                        gradient[layer_index, head_index]
                    ).item()
                    > 0
                )
    return {
        "tensor_count": len(by_parameter),
        "nonzero_tensor_count": sum(by_parameter.values()),
        "all_trainable_tensors_nonzero": bool(by_parameter)
        and all(by_parameter.values()),
        "by_parameter": by_parameter,
        "matrix_count": len(by_matrix),
        "nonzero_matrix_count": sum(by_matrix.values()),
        "all_correction_matrices_nonzero": bool(by_matrix)
        and all(by_matrix.values()),
        "by_matrix": by_matrix,
    }


def train_controller(
    parent: MarulhoLanguageModel,
    controller: ExactTokenKVController,
    prepared: PreparedCases,
    schedule: torch.Tensor,
    config: V63Config,
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
        inputs = prepared.input_ids.index_select(0, indices).to(parent.device)
        targets = prepared.target_ids.index_select(0, indices).to(parent.device)
        source_mask = prepared.source_mask.index_select(0, indices).to(parent.device)
        answer_mask = prepared.answer_mask.index_select(0, indices).to(parent.device)
        learning_rate = _learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        hidden, _state = controller.forward_parent(parent, inputs, source_mask)
        logits = parent.lm_head(hidden)
        loss = F.cross_entropy(logits[answer_mask].float(), targets[answer_mask])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"V63 loss became non-finite at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), config.gradient_clip)
        optimizer.step()
        answer_positions += int(answer_mask.sum())
        if (step + 1) % 64 == 0 or step + 1 == config.optimizer_steps:
            final_loss = float(loss.detach())
            maximum = max(
                float(controller.key_corrections.detach().abs().max()),
                float(controller.value_corrections.detach().abs().max()),
            )
            print(
                f"[v63] train {step + 1}/{config.optimizer_steps} "
                f"loss={final_loss:.4f} max_raw={maximum:.4f} "
                f"lr={learning_rate:.3g}",
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
    controller_finite = all(
        bool(torch.isfinite(parameter).all()) for parameter in controller.parameters()
    )
    maximum_bounded_coefficient = max(
        float(
            (
                torch.tanh(controller.key_corrections.detach())
                * controller.bounded_scale
            )
            .abs()
            .max()
        ),
        float(
            (
                torch.tanh(controller.value_corrections.detach())
                * controller.bounded_scale
            )
            .abs()
            .max()
        ),
    )
    del optimizer
    return {
        "optimizer_steps": config.optimizer_steps,
        "padded_positions": config.padded_position_budget,
        "answer_target_positions": answer_positions,
        "final_training_loss": final_loss,
        "training_seconds": elapsed,
        "positions_per_second": config.padded_position_budget / elapsed,
        "peak_cuda_bytes": peak,
        "final_gradients": gradients,
        "controller_finite": controller_finite,
        "maximum_bounded_coefficient": maximum_bounded_coefficient,
    }


@torch.no_grad()
def active_zero_parity(
    parent: MarulhoLanguageModel,
    controller: ExactTokenKVController,
    input_ids: torch.Tensor,
    source_mask: torch.Tensor,
) -> dict[str, Any]:
    parent.eval()
    controller.eval()
    runtime_ids = input_ids.to(parent.device)
    runtime_mask = source_mask.to(parent.device)
    ordinary = parent._forward_hidden(runtime_ids, collect_telemetry=False)
    custom_hidden, custom_state = controller.forward_parent(
        parent,
        runtime_ids,
        runtime_mask,
    )
    state_keys_exact = set(ordinary["state"]) == set(custom_state)
    by_state = {
        name: bool(torch.equal(ordinary["state"][name], custom_state[name]))
        for name in custom_state
        if name in ordinary["state"]
    }
    return {
        "hidden_exact": bool(torch.equal(ordinary["hidden"], custom_hidden)),
        "logits_exact": bool(
            torch.equal(
                parent.lm_head(ordinary["hidden"]),
                parent.lm_head(custom_hidden),
            )
        ),
        "state_keys_exact": state_keys_exact,
        "state_exact": state_keys_exact and all(by_state.values()),
        "state_key_count": len(custom_state),
        "by_state": by_state,
    }


def _prompt_batch(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: LanguageTokenizer,
    *,
    view: str,
) -> tuple[list[list[int]], list[torch.Tensor], list[dict[str, Any]]]:
    ids: list[list[int]] = []
    masks: list[torch.Tensor] = []
    evidence: list[dict[str, Any]] = []
    if view == "question_only":
        for raw in rows:
            row = dict(raw)
            prompt_ids = tokenizer.encode(
                str(row["question_only_prompt"]),
                add_eos=False,
            )
            ids.append(prompt_ids)
            masks.append(torch.zeros(len(prompt_ids), dtype=torch.bool))
            evidence.append(
                {
                    "case_id": str(row["case_id"]),
                    "view": view,
                    "prompt_token_count": len(prompt_ids),
                    "source_token_count": 0,
                }
            )
    else:
        for raw in rows:
            prompt_ids, source_mask, boundary = _source_mask_for_prompt(
                raw,
                tokenizer,
                view=view,
            )
            ids.append(prompt_ids)
            masks.append(source_mask)
            evidence.append(boundary)
    return ids, masks, evidence


@torch.no_grad()
def _generate_equal_length_group(
    parent: MarulhoLanguageModel,
    controller: ExactTokenKVController,
    tokenizer: LanguageTokenizer,
    prompt_rows: Sequence[Sequence[int]],
    source_rows: Sequence[torch.Tensor],
    config: V63Config,
    *,
    active: bool,
) -> list[list[int]]:
    prompt = torch.tensor(prompt_rows, dtype=torch.long, device=parent.device)
    if active:
        source_mask = torch.stack(tuple(source_rows)).to(parent.device)
        hidden, state = controller.forward_parent(parent, prompt, source_mask)
        next_logits = parent.lm_head(hidden[:, -1])
    else:
        result = parent(prompt, collect_telemetry=False)
        state = result["state"]
        next_logits = result["logits"][:, -1]
    batch_size = int(prompt.shape[0])
    generated = torch.empty(
        batch_size,
        0,
        device=parent.device,
        dtype=torch.long,
    )
    finished = torch.zeros(batch_size, device=parent.device, dtype=torch.bool)
    for _ in range(config.generation_tokens):
        adjusted, _controls = _apply_decode_controls(
            next_logits,
            generated,
            repetition_penalty=config.repetition_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
        )
        next_ids = torch.argmax(adjusted, dim=-1)
        next_ids = torch.where(
            finished,
            torch.full_like(next_ids, int(tokenizer.eos_id)),
            next_ids,
        )
        generated = torch.cat((generated, next_ids.unsqueeze(1)), dim=1)
        finished = finished | next_ids.eq(int(tokenizer.eos_id))
        if bool(finished.all()):
            break
        step = parent.forward_step(
            next_ids,
            state,
            collect_telemetry=False,
        )
        state = step["state"]
        next_logits = step["logits"][:, -1]
    return [
        [int(value) for value in row]
        for row in generated.detach().cpu().tolist()
    ]


@torch.no_grad()
def evaluate_view(
    parent: MarulhoLanguageModel,
    controller: ExactTokenKVController,
    tokenizer: LanguageTokenizer,
    rows: Sequence[Mapping[str, Any]],
    config: V63Config,
    *,
    view: str,
) -> dict[str, Any]:
    parent.eval()
    controller.eval()
    prompt_ids, source_masks, boundary_evidence = _prompt_batch(
        rows,
        tokenizer,
        view=view,
    )
    groups: dict[int, list[int]] = {}
    for index, ids in enumerate(prompt_ids):
        groups.setdefault(len(ids), []).append(index)
    generated_by_index: list[list[int]] = [[] for _ in rows]
    started = time.perf_counter()
    completed = 0
    next_progress = 64
    for prompt_length in sorted(groups):
        indices = groups[prompt_length]
        for offset in range(0, len(indices), config.batch_size):
            batch_indices = indices[offset : offset + config.batch_size]
            generated = _generate_equal_length_group(
                parent,
                controller,
                tokenizer,
                [prompt_ids[index] for index in batch_indices],
                [source_masks[index] for index in batch_indices],
                config,
                active=view != "question_only",
            )
            for index, ids in zip(batch_indices, generated, strict=True):
                generated_by_index[index] = ids
            completed += len(batch_indices)
            if completed >= next_progress or completed == len(rows):
                print(f"[v63] {view} {completed}/{len(rows)}", flush=True)
                while next_progress <= completed:
                    next_progress += 64
    if parent.device.type == "cuda":
        torch.cuda.synchronize(parent.device)
    elapsed = max(time.perf_counter() - started, 1.0e-9)
    output_rows: list[dict[str, Any]] = []
    for raw, generated in zip(rows, generated_by_index, strict=True):
        row = dict(raw)
        if int(tokenizer.eos_id) in generated:
            generated = generated[: generated.index(int(tokenizer.eos_id))]
        continuation = tokenizer.decode(generated)
        normalized = _normalized(continuation)
        accepted = {_normalized(value) for value in row["answers"]}
        output_rows.append(
            {
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
        )
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
        "prompt_length_group_count": len(groups),
        "boundary_evidence_sha256": sha256_json(boundary_evidence),
        "rows": output_rows,
    }


@torch.no_grad()
def _controller_probe(
    parent: MarulhoLanguageModel,
    controller: ExactTokenKVController,
    tokenizer: LanguageTokenizer,
    row: Mapping[str, Any],
) -> torch.Tensor:
    ids, source_mask, _evidence = _source_mask_for_prompt(
        row,
        tokenizer,
        view="true",
    )
    hidden, _state = controller.forward_parent(
        parent,
        torch.tensor([ids], dtype=torch.long, device=parent.device),
        source_mask.unsqueeze(0).to(parent.device),
    )
    return parent.lm_head(hidden[:, -1:]).detach().cpu()


def _parent_probe(
    parent: MarulhoLanguageModel,
    tokenizer: LanguageTokenizer,
    row: Mapping[str, Any],
) -> torch.Tensor:
    ids = tokenizer.encode(str(row["question_only_prompt"]), add_eos=False)
    probe = torch.tensor([ids], dtype=torch.long)
    parent.eval()
    with torch.no_grad():
        return parent(probe, collect_telemetry=False)["logits"].detach().cpu()


def _boundary_audit(
    manifests: Mapping[str, Sequence[Mapping[str, Any]]],
    tokenizer: LanguageTokenizer,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for manifest_name, cases in manifests.items():
        for view in ("true", "oracle", "shuffled"):
            for case in cases:
                _ids, _mask, evidence = _source_mask_for_prompt(
                    case,
                    tokenizer,
                    view=view,
                )
                rows.append({"manifest": manifest_name, **evidence})
    source_counts = [int(row["source_token_count"]) for row in rows]
    return {
        "record_view_count": len(rows),
        "expected_record_view_count": 3
        * sum(len(cases) for cases in manifests.values()),
        "all_token_boundaries_exact": all(
            bool(row["token_boundary_exact"]) for row in rows
        ),
        "all_delimiter_normalized_suffixes_exact": all(
            bool(row["delimiter_normalized_suffix_exact"]) for row in rows
        ),
        "crossing_token_count": sum(
            int(row["crossing_token_count"]) for row in rows
        ),
        "minimum_source_token_count": min(source_counts),
        "maximum_source_token_count": max(source_counts),
        "sha256": sha256_json(rows),
    }


def run_v63(
    *,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    train_manifest_path: str | Path = DEFAULT_TRAIN_MANIFEST,
    validation_manifest_path: str | Path = DEFAULT_VALIDATION_MANIFEST,
    output_path: str | Path = DEFAULT_OUTPUT,
    candidate_path: str | Path = DEFAULT_CANDIDATE,
    device: str = "cuda",
) -> dict[str, Any]:
    config = V63Config()
    runtime_device = torch.device(device)
    if runtime_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V63 is frozen as a CUDA experiment")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    total_started = time.perf_counter()

    checkpoint = Path(checkpoint_path)
    train_path = Path(train_manifest_path)
    validation_path = Path(validation_manifest_path)
    checkpoint_sha_before = _sha256_file(checkpoint)
    parent_cpu, tokenizer, parent_metadata = load_language_model_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    train_manifest = load_squad_grounding_manifest(train_path, tokenizer)
    validation_manifest = load_squad_grounding_manifest(validation_path, tokenizer)
    train_rows = tuple(dict(row) for row in train_manifest["cases"])
    validation_rows = tuple(dict(row) for row in validation_manifest["cases"])
    if len(train_rows) != 8_192 or len(validation_rows) != 256:
        raise ValueError("V63 requires the exact V57 8192/256 boundary")
    train_titles = {str(row["title"]) for row in train_rows}
    validation_titles = {str(row["title"]) for row in validation_rows}
    if train_titles & validation_titles:
        raise ValueError("V63 train and validation titles overlap")

    parent_state_before = language_model_state_sha256(parent_cpu)
    tokenizer_hash_before = tokenizer.vocabulary_hash()
    parent_logits_before = _parent_probe(parent_cpu, tokenizer, validation_rows[0])
    runtime_parent, runtime_tokenizer, _runtime_metadata = (
        load_language_model_checkpoint(checkpoint, map_location="cpu")
    )
    if runtime_tokenizer.vocabulary_hash() != tokenizer_hash_before:
        raise ValueError("V63 runtime tokenizer differs from immutable parent")
    parent = MarulhoLanguageModel(
        replace(
            runtime_parent.config,
            transformer_context_length=config.context_length,
        )
    )
    parent.load_state_dict(runtime_parent.state_dict(), strict=True)
    short_ids = tokenizer.encode(
        str(validation_rows[0]["question_only_prompt"]),
        add_eos=False,
    )
    short_probe = torch.tensor([short_ids], dtype=torch.long)
    runtime_parent.eval()
    parent.eval()
    with torch.no_grad():
        original_logits = runtime_parent(
            short_probe,
            collect_telemetry=False,
        )["logits"]
        extended_logits = parent(short_probe, collect_telemetry=False)["logits"]
    initial_short_prefix_exact = bool(torch.equal(original_logits, extended_logits))
    if not initial_short_prefix_exact:
        raise RuntimeError("V63 context-320 reconstruction changed a short prefix")
    del runtime_parent, original_logits, extended_logits
    parent = parent.to(device=runtime_device, dtype=torch.bfloat16)
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    parent_parameter_count = sum(parameter.numel() for parameter in parent.parameters())

    setup_started = time.perf_counter()
    print("[v63] auditing exact tokenizer boundaries", flush=True)
    boundary_audit = _boundary_audit(
        {"train": train_rows, "validation": validation_rows},
        tokenizer,
    )
    print("[v63] preparing immutable context-320 cases", flush=True)
    train_cases = prepare_cases(train_rows, tokenizer, config, view="true")
    schedule, schedule_sha = _schedule_indices(len(train_cases), config)
    if int(schedule.numel()) != config.optimizer_steps * config.batch_size:
        raise RuntimeError("V63 schedule violates the frozen update count")
    prepared_sha = sha256_json(
        {
            name: _tensor_sha256(value)
            for name, value in {
                "input_ids": train_cases.input_ids,
                "target_ids": train_cases.target_ids,
                "source_mask": train_cases.source_mask,
                "answer_mask": train_cases.answer_mask,
            }.items()
        }
    )
    setup_seconds = time.perf_counter() - setup_started

    head_dim = int(parent.config.state_dim) // int(parent.config.attention_heads)
    controller = ExactTokenKVController(
        state_layers=int(parent.config.state_layers),
        attention_heads=int(parent.config.attention_heads),
        head_dim=head_dim,
        correction_scale=config.correction_scale,
        model_seed=config.model_seed,
    ).to(device=runtime_device)
    controller_parameter_count = sum(
        parameter.numel() for parameter in controller.parameters()
    )
    parameter_fraction = controller_parameter_count / parent_parameter_count
    controller_dtypes = {
        str(parameter.dtype) for parameter in controller.parameters()
    }
    initial_controller_state_sha = sha256_json(
        {
            name: _tensor_sha256(value)
            for name, value in controller.state_dict().items()
        }
    )
    initial_controller_zero = all(
        int(torch.count_nonzero(parameter)) == 0
        for parameter in controller.parameters()
    )
    parity_prepared = prepare_cases(
        validation_rows[:2],
        tokenizer,
        config,
        view="true",
    )
    active_zero = active_zero_parity(
        parent,
        controller,
        parity_prepared.input_ids[:, : config.context_length],
        parity_prepared.source_mask[:, : config.context_length],
    )
    if not all(
        bool(active_zero[key])
        for key in ("hidden_exact", "logits_exact", "state_exact")
    ):
        raise RuntimeError("V63 active-zero forward is not bit-exact")

    print("[v63] evaluating frozen raw true/oracle prefixes", flush=True)
    raw_true = evaluate_view(
        parent,
        controller,
        tokenizer,
        validation_rows,
        config,
        view="true",
    )
    raw_oracle = evaluate_view(
        parent,
        controller,
        tokenizer,
        validation_rows,
        config,
        view="oracle",
    )

    print("[v63] training exact-token KV controller", flush=True)
    training = train_controller(parent, controller, train_cases, schedule, config)
    print("[v63] evaluating learned controls", flush=True)
    question_only = evaluate_view(
        parent,
        controller,
        tokenizer,
        validation_rows,
        config,
        view="question_only",
    )
    shuffled = evaluate_view(
        parent,
        controller,
        tokenizer,
        validation_rows,
        config,
        view="shuffled",
    )
    true = evaluate_view(
        parent,
        controller,
        tokenizer,
        validation_rows,
        config,
        view="true",
    )
    oracle = evaluate_view(
        parent,
        controller,
        tokenizer,
        validation_rows,
        config,
        view="oracle",
    )

    checkpoint_sha_after = _sha256_file(checkpoint)
    tokenizer_hash_after = tokenizer.vocabulary_hash()
    parent_state_after = language_model_state_sha256(parent_cpu)
    parent_logits_after = _parent_probe(parent_cpu, tokenizer, validation_rows[0])
    parent_checks = {
        "checkpoint_file_exact": checkpoint_sha_before == checkpoint_sha_after,
        "state_exact": parent_state_before == parent_state_after,
        "tokenizer_exact": tokenizer_hash_before == tokenizer_hash_after,
        "sample_logits_exact": bool(
            torch.equal(parent_logits_before, parent_logits_after)
        ),
        "initial_short_prefix_exact": initial_short_prefix_exact,
        "all_parent_parameters_frozen": all(
            not parameter.requires_grad for parameter in parent.parameters()
        ),
    }
    total_setup_training_seconds = setup_seconds + float(training["training_seconds"])
    true_exact = int(true["exact_answer_count"])
    question_only_exact = int(question_only["exact_answer_count"])
    shuffled_exact = int(shuffled["exact_answer_count"])
    oracle_exact = int(oracle["exact_answer_count"])
    source_gain = float(true["exact_answer_accuracy"]) - max(
        float(question_only["exact_answer_accuracy"]),
        float(shuffled["exact_answer_accuracy"]),
    )
    mechanical_checks = {
        "exact_parameter_count": controller_parameter_count == 983_040,
        "parameter_fraction": parameter_fraction <= config.maximum_parameter_fraction,
        "controller_fp32": controller_dtypes == {"torch.float32"},
        "initial_controller_zero": initial_controller_zero,
        "active_zero_hidden_exact": bool(active_zero["hidden_exact"]),
        "active_zero_logits_exact": bool(active_zero["logits_exact"]),
        "active_zero_state_exact": bool(active_zero["state_exact"]),
        "all_boundary_records_covered": boundary_audit["record_view_count"]
        == boundary_audit["expected_record_view_count"],
        "all_token_boundaries_exact": bool(
            boundary_audit["all_token_boundaries_exact"]
        ),
        "all_normalized_suffixes_exact": bool(
            boundary_audit["all_delimiter_normalized_suffixes_exact"]
        ),
        "zero_crossing_tokens": int(boundary_audit["crossing_token_count"]) == 0,
        "exact_optimizer_steps": training["optimizer_steps"]
        == config.optimizer_steps,
        "exact_position_budget": training["padded_positions"]
        == config.padded_position_budget,
        "complete_final_gradients": bool(
            training["final_gradients"]["all_trainable_tensors_nonzero"]
        ),
        "complete_final_matrix_gradients": bool(
            training["final_gradients"]["all_correction_matrices_nonzero"]
        ),
        "finite_controller": bool(training["controller_finite"]),
        "cuda_allocation_observed": int(training["peak_cuda_bytes"]) > 0,
        "maximum_training_seconds": float(training["training_seconds"])
        <= config.maximum_training_seconds,
        "maximum_total_setup_training_seconds": total_setup_training_seconds
        <= config.maximum_total_setup_training_seconds,
        "parent_fidelity": all(parent_checks.values()),
    }
    behavioral_checks = {
        "minimum_true_exact_answers": true_exact
        >= config.minimum_true_exact_answers,
        "minimum_true_source_gain": source_gain >= config.minimum_true_source_gain,
        "maximum_shuffled_exact_answers": shuffled_exact
        <= config.maximum_shuffled_exact_answers,
        "minimum_oracle_exact_answers": oracle_exact
        >= config.minimum_oracle_exact_answers,
        "maximum_true_oracle_gap": oracle_exact - true_exact
        <= config.maximum_true_oracle_gap,
    }
    mechanical_pass = all(mechanical_checks.values())
    behavioral_pass = all(behavioral_checks.values())

    checkpoint_evidence: dict[str, Any] = {
        "saved": False,
        "path": None,
        "sha256": None,
        "strict_tensor_reload": False,
        "strict_logit_reload": False,
    }
    if mechanical_pass and behavioral_pass:
        probe_before = _controller_probe(
            parent,
            controller,
            tokenizer,
            validation_rows[0],
        )
        candidate = Path(candidate_path)
        _atomic_torch_save(
            candidate,
            {
                "artifact_kind": "marulho_exact_token_kv_checkpoint",
                "surface": "marulho_exact_token_kv_checkpoint.v1",
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
        reloaded = ExactTokenKVController(
            state_layers=int(parent.config.state_layers),
            attention_heads=int(parent.config.attention_heads),
            head_dim=head_dim,
            correction_scale=config.correction_scale,
            model_seed=config.model_seed,
        )
        reloaded.load_state_dict(dict(payload["controller_state"]), strict=True)
        tensor_exact = all(
            torch.equal(value.detach().cpu(), reloaded.state_dict()[name])
            for name, value in controller.state_dict().items()
        )
        reloaded = reloaded.to(device=runtime_device)
        probe_after = _controller_probe(
            parent,
            reloaded,
            tokenizer,
            validation_rows[0],
        )
        checkpoint_evidence = {
            "saved": True,
            "path": str(candidate),
            "sha256": _sha256_file(candidate),
            "strict_tensor_reload": tensor_exact,
            "strict_logit_reload": bool(torch.equal(probe_before, probe_after)),
        }
        del reloaded
    checkpoint_passed = all(
        bool(checkpoint_evidence[key])
        for key in ("saved", "strict_tensor_reload", "strict_logit_reload")
    )
    passed = mechanical_pass and behavioral_pass and checkpoint_passed
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
    elif not mechanical_pass:
        decision = INVALID_DECISION
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
            "boundary_audit": boundary_audit,
            "cache_policy": "online_exact_token_kv_no_persistent_hidden_cache",
            "persistent_cache_bytes": 0,
            "source_mask_excludes_answer": True,
            "source_mask_excludes_question": True,
            "source_mask_excludes_labels": True,
            "source_mask_uses_answer_span": False,
        },
        "architecture": {
            "controller_parameter_count": controller_parameter_count,
            "parent_parameter_count": parent_parameter_count,
            "controller_parameter_fraction": parameter_fraction,
            "controller_dtypes": sorted(controller_dtypes),
            "correction_matrix_count": controller.correction_matrix_count,
            "head_dim": head_dim,
            "initial_controller_state_sha256": initial_controller_state_sha,
            "initial_controller_zero": initial_controller_zero,
            "bounded_scale": controller.bounded_scale,
            "adapted_state": "exact_source_token_keys_and_values",
            "parent_update": "none_frozen",
        },
        "setup": {"seconds": setup_seconds, "persistent_cache_bytes": 0},
        "training": training,
        "views": {
            "raw_true": raw_true,
            "raw_oracle": raw_oracle,
            "question_only": question_only,
            "shuffled": shuffled,
            "true": true,
            "oracle": oracle,
        },
        "parent": {
            "path": str(checkpoint),
            "checkpoint_sha256_before": checkpoint_sha_before,
            "checkpoint_sha256_after": checkpoint_sha_after,
            "state_sha256_before": parent_state_before,
            "state_sha256_after": parent_state_after,
            "tokenizer_hash_before": tokenizer_hash_before,
            "tokenizer_hash_after": tokenizer_hash_after,
            "metadata": parent_metadata,
            "active_zero_parity": active_zero,
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
            "mechanical_passed": mechanical_pass,
            "behavioral_passed": behavioral_pass,
            "mechanical_checks": mechanical_checks,
            "behavioral_checks": behavioral_checks,
            "checkpoint_passed": checkpoint_passed,
            "observed": {
                "raw_true_exact_answers": raw_true["exact_answer_count"],
                "raw_oracle_exact_answers": raw_oracle["exact_answer_count"],
                "question_only_exact_answers": question_only_exact,
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
        f"[v63] decision={decision} question_only={question_only_exact} "
        f"shuffled={shuffled_exact} true={true_exact} oracle={oracle_exact}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--train-manifest", default=str(DEFAULT_TRAIN_MANIFEST))
    parser.add_argument(
        "--validation-manifest",
        default=str(DEFAULT_VALIDATION_MANIFEST),
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    run_v63(
        checkpoint_path=arguments.checkpoint,
        train_manifest_path=arguments.train_manifest,
        validation_manifest_path=arguments.validation_manifest,
        output_path=arguments.output,
        candidate_path=arguments.candidate,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
