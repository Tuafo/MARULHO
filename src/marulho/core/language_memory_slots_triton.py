from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional CUDA dependency
    triton = None
    tl = None


_SUPPORTED_DTYPES = {torch.float32}
_MAX_STATE_DIM = 512
_MAX_CANDIDATES = 64
_MAX_ACTIVE_SLOTS = 8
_DEFAULT_MIN_ROWS = 128


@dataclass
class _MemorySlotsTritonStats:
    triton_forward_calls: int = 0
    triton_forward_elements: int = 0
    torch_fallback_calls: int = 0
    torch_fallback_elements: int = 0
    triton_failure_count: int = 0
    last_failure: str | None = None
    last_device: str | None = None
    last_dtype: str | None = None


_STATS = _MemorySlotsTritonStats()


def ensure_language_memory_slots_triton_compiler() -> str | None:
    if os.environ.get("CC"):
        return os.environ["CC"]
    if os.name != "nt" or triton is None:
        return None
    compiler = Path(triton.__file__).parent / "runtime" / "tcc" / "tcc.exe"
    if compiler.exists():
        os.environ["CC"] = str(compiler)
        return str(compiler)
    return None


def reset_language_memory_slots_triton_stats() -> None:
    global _STATS
    _STATS = _MemorySlotsTritonStats()


def _min_rows() -> int:
    raw = os.environ.get("MARULHO_LANGUAGE_MEMORY_SLOTS_TRITON_MIN_ROWS")
    if raw is None:
        return _DEFAULT_MIN_ROWS
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_MIN_ROWS


def language_memory_slots_triton_stats() -> dict[str, Any]:
    return {
        "surface": "marulho_language_memory_slots_triton_stats.v1",
        "triton_available": bool(triton is not None),
        "default_min_rows": _min_rows(),
        "triton_forward_calls": int(_STATS.triton_forward_calls),
        "triton_forward_elements": int(_STATS.triton_forward_elements),
        "torch_fallback_calls": int(_STATS.torch_fallback_calls),
        "torch_fallback_elements": int(_STATS.torch_fallback_elements),
        "triton_failure_count": int(_STATS.triton_failure_count),
        "last_failure": _STATS.last_failure,
        "last_device": _STATS.last_device,
        "last_dtype": _STATS.last_dtype,
    }


def language_memory_slots_triton_stats_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    int_fields = (
        "triton_forward_calls",
        "triton_forward_elements",
        "torch_fallback_calls",
        "torch_fallback_elements",
        "triton_failure_count",
    )
    delta = {
        key: int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)
        for key in int_fields
    }
    return {
        "surface": "marulho_language_memory_slots_triton_stats_delta.v1",
        "triton_available": bool(after.get("triton_available")),
        **delta,
        "last_failure": after.get("last_failure"),
        "last_device": after.get("last_device"),
        "last_dtype": after.get("last_dtype"),
        "triton_kernel_used": int(delta["triton_forward_calls"]) > 0,
    }


def language_memory_slots_torch_reference(
    hidden: torch.Tensor,
    candidate_ids: torch.Tensor,
    memory_slots: torch.Tensor,
    memory_slot_gate: torch.Tensor,
    active_count: int,
) -> torch.Tensor:
    if hidden.ndim != 2 or candidate_ids.ndim != 2:
        raise ValueError("memory slots expect hidden [rows, dim] and candidate_ids [rows, candidates]")
    runtime_candidate_ids = candidate_ids.to(device=hidden.device, dtype=torch.long)
    candidate_count = int(runtime_candidate_ids.shape[1])
    active = min(max(1, int(active_count)), candidate_count)
    runtime_memory_slots = memory_slots.to(device=hidden.device, dtype=hidden.dtype)
    candidate_values = runtime_memory_slots.index_select(
        0,
        runtime_candidate_ids.reshape(-1),
    ).reshape(
        int(hidden.shape[0]),
        candidate_count,
        int(hidden.shape[1]),
    )
    scores = (
        hidden.unsqueeze(-2).to(candidate_values.dtype) * candidate_values
    ).sum(dim=-1) / math.sqrt(max(1, int(hidden.shape[1])))
    if active < candidate_count:
        top_scores, top_positions = torch.topk(scores, k=active, dim=-1)
        selected_values = candidate_values.gather(
            dim=-2,
            index=top_positions.unsqueeze(-1).expand(
                int(hidden.shape[0]),
                active,
                int(hidden.shape[1]),
            ),
        )
    else:
        top_scores = scores
        selected_values = candidate_values
    weights = F.softmax(top_scores, dim=-1)
    context = (selected_values * weights.unsqueeze(-1)).sum(dim=-2)
    gate = torch.tanh(memory_slot_gate).to(device=hidden.device, dtype=hidden.dtype)
    return hidden + context * gate


def can_use_language_memory_slots_triton(
    hidden: torch.Tensor,
    candidate_ids: torch.Tensor,
    memory_slots: torch.Tensor,
    memory_slot_gate: torch.Tensor,
    active_count: int,
) -> bool:
    if triton is None or tl is None:
        return False
    if torch.is_grad_enabled():
        return False
    tensors = (hidden, memory_slots, memory_slot_gate)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        return False
    if not isinstance(candidate_ids, torch.Tensor):
        return False
    if any(value.device.type != "cuda" for value in (*tensors, candidate_ids)):
        return False
    if hidden.dtype not in _SUPPORTED_DTYPES:
        return False
    if memory_slots.dtype != hidden.dtype:
        return False
    if candidate_ids.dtype not in {torch.int64, torch.long}:
        return False
    if hidden.ndim != 2 or candidate_ids.ndim != 2 or memory_slots.ndim != 2:
        return False
    rows, state_dim = int(hidden.shape[0]), int(hidden.shape[1])
    candidate_count = int(candidate_ids.shape[1])
    slot_count = int(memory_slots.shape[0])
    active = int(active_count)
    if rows <= 0 or state_dim <= 0 or candidate_count <= 0 or slot_count <= 0:
        return False
    if active <= 0 or active > candidate_count or active > _MAX_ACTIVE_SLOTS:
        return False
    if candidate_count > _MAX_CANDIDATES or state_dim > _MAX_STATE_DIM:
        return False
    return int(memory_slots.shape[1]) == state_dim and memory_slot_gate.numel() == 1


def should_use_language_memory_slots_triton(
    hidden: torch.Tensor,
    candidate_ids: torch.Tensor,
    memory_slots: torch.Tensor,
    memory_slot_gate: torch.Tensor,
    active_count: int,
) -> bool:
    if not can_use_language_memory_slots_triton(
        hidden,
        candidate_ids,
        memory_slots,
        memory_slot_gate,
        active_count,
    ):
        return False
    return int(hidden.shape[0]) >= _min_rows()


def _next_power_of_2(value: int) -> int:
    return 1 << (max(1, int(value)) - 1).bit_length()


if triton is not None:

    @triton.jit
    def _memory_slots_forward_kernel(
        hidden,
        candidate_ids,
        memory_slots,
        gate,
        output,
        state_dim: tl.constexpr,
        candidate_count: tl.constexpr,
        active_count: tl.constexpr,
        state_block: tl.constexpr,
        candidate_block: tl.constexpr,
        active_block: tl.constexpr,
    ):
        row = tl.program_id(0)
        candidate_offsets = tl.arange(0, candidate_block)
        active_offsets = tl.arange(0, active_block)
        state_offsets = tl.arange(0, state_block)
        candidate_mask = candidate_offsets < candidate_count
        active_mask = active_offsets < active_count
        state_mask = state_offsets < state_dim
        slot_ids = tl.load(
            candidate_ids + row * candidate_count + candidate_offsets,
            mask=candidate_mask,
            other=0,
        ).to(tl.int64)
        hidden_values = tl.load(
            hidden + row * state_dim + state_offsets,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)
        slot_values = tl.load(
            memory_slots + slot_ids[:, None] * state_dim + state_offsets[None, :],
            mask=candidate_mask[:, None] & state_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scale = 1.0 / tl.sqrt(state_dim + 0.0)
        scores = tl.sum(slot_values * hidden_values[None, :], axis=1) * scale
        neg_inf = -3.4028234663852886e38
        scores = tl.where(candidate_mask, scores, neg_inf)
        work_scores = scores
        top_scores = tl.full((active_block,), neg_inf, dtype=tl.float32)
        top_positions = tl.full((active_block,), 0, dtype=tl.int64)
        for active in tl.range(0, active_count):
            best_score = tl.max(work_scores, axis=0)
            best_positions = tl.where(
                work_scores == best_score,
                candidate_offsets,
                candidate_count + candidate_offsets,
            )
            best_position = tl.min(best_positions, axis=0)
            top_scores = tl.where(active_offsets == active, best_score, top_scores)
            top_positions = tl.where(active_offsets == active, best_position, top_positions)
            work_scores = tl.where(candidate_offsets == best_position, neg_inf, work_scores)
        max_top_score = tl.max(tl.where(active_mask, top_scores, neg_inf), axis=0)
        exp_scores = tl.where(active_mask, tl.exp(top_scores - max_top_score), 0.0)
        denom = tl.sum(exp_scores, axis=0)
        context = tl.zeros((state_block,), dtype=tl.float32)
        for active in tl.range(0, active_count):
            selected_position = tl.max(
                tl.where(active_offsets == active, top_positions, 0),
                axis=0,
            )
            selected_weight = tl.max(
                tl.where(active_offsets == active, exp_scores / denom, 0.0),
                axis=0,
            )
            selected_values = tl.sum(
                tl.where(
                    candidate_offsets[:, None] == selected_position,
                    slot_values,
                    0.0,
                ),
                axis=0,
            )
            context += selected_values * selected_weight
        gate_value = tl.load(gate).to(tl.float32)
        tl.store(
            output + row * state_dim + state_offsets,
            hidden_values + context * gate_value,
            mask=state_mask,
        )


def _language_memory_slots_triton_forward(
    hidden: torch.Tensor,
    candidate_ids: torch.Tensor,
    memory_slots: torch.Tensor,
    memory_slot_gate: torch.Tensor,
    active_count: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available")
    ensure_language_memory_slots_triton_compiler()
    runtime_hidden = hidden.contiguous()
    runtime_candidate_ids = candidate_ids.to(
        device=hidden.device,
        dtype=torch.long,
    ).contiguous()
    runtime_memory_slots = memory_slots.to(
        device=hidden.device,
        dtype=hidden.dtype,
    ).contiguous()
    runtime_gate = torch.tanh(
        memory_slot_gate.to(device=hidden.device, dtype=hidden.dtype)
    ).contiguous()
    row_count = int(runtime_hidden.shape[0])
    state_dim = int(runtime_hidden.shape[1])
    candidate_count = int(runtime_candidate_ids.shape[1])
    active = min(max(1, int(active_count)), candidate_count)
    output = torch.empty_like(runtime_hidden)
    state_block = _next_power_of_2(state_dim)
    candidate_block = _next_power_of_2(candidate_count)
    active_block = _next_power_of_2(active)
    _memory_slots_forward_kernel[(row_count,)](
        runtime_hidden,
        runtime_candidate_ids,
        runtime_memory_slots,
        runtime_gate,
        output,
        state_dim,
        candidate_count,
        active,
        state_block,
        candidate_block,
        active_block,
    )
    _STATS.triton_forward_calls += 1
    _STATS.triton_forward_elements += int(row_count * candidate_count * state_dim)
    _STATS.last_device = str(hidden.device)
    _STATS.last_dtype = str(hidden.dtype)
    return output


def language_memory_slots(
    hidden: torch.Tensor,
    candidate_ids: torch.Tensor,
    memory_slots: torch.Tensor,
    memory_slot_gate: torch.Tensor,
    active_count: int,
    *,
    prefer_triton: bool = True,
    force_triton: bool = False,
) -> torch.Tensor:
    if bool(prefer_triton) and hidden.device.type == "cuda":
        use_triton = (
            can_use_language_memory_slots_triton(
                hidden,
                candidate_ids,
                memory_slots,
                memory_slot_gate,
                active_count,
            )
            if bool(force_triton)
            else should_use_language_memory_slots_triton(
                hidden,
                candidate_ids,
                memory_slots,
                memory_slot_gate,
                active_count,
            )
        )
        if use_triton:
            try:
                return _language_memory_slots_triton_forward(
                    hidden,
                    candidate_ids,
                    memory_slots,
                    memory_slot_gate,
                    active_count,
                )
            except Exception as exc:  # pragma: no cover - hardware/runtime dependent
                _STATS.triton_failure_count += 1
                _STATS.last_failure = f"{type(exc).__name__}: {exc}"
                _STATS.last_device = str(hidden.device)
                _STATS.last_dtype = str(hidden.dtype)
        _STATS.torch_fallback_calls += 1
        _STATS.torch_fallback_elements += int(
            hidden.numel() * max(1, int(candidate_ids.shape[-1]))
        )
        _STATS.last_device = str(hidden.device)
        _STATS.last_dtype = str(hidden.dtype)
    return language_memory_slots_torch_reference(
        hidden,
        candidate_ids,
        memory_slots,
        memory_slot_gate,
        active_count,
    )
