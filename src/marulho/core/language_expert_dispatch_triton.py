from __future__ import annotations

from dataclasses import dataclass
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
_MAX_EXPERT_HIDDEN_DIM = 1024
_MAX_ACTIVE_EXPERTS = 8
_DEFAULT_OUTPUT_BLOCK = 32
_DEFAULT_HIDDEN_BLOCK = 32
_DEFAULT_MIN_TOKENS = 128


@dataclass
class _ExpertDispatchTritonStats:
    triton_forward_calls: int = 0
    triton_forward_elements: int = 0
    torch_fallback_calls: int = 0
    torch_fallback_elements: int = 0
    triton_failure_count: int = 0
    last_failure: str | None = None
    last_device: str | None = None
    last_dtype: str | None = None


_STATS = _ExpertDispatchTritonStats()


def ensure_language_expert_dispatch_triton_compiler() -> str | None:
    if os.environ.get("CC"):
        return os.environ["CC"]
    if os.name != "nt" or triton is None:
        return None
    compiler = Path(triton.__file__).parent / "runtime" / "tcc" / "tcc.exe"
    if compiler.exists():
        os.environ["CC"] = str(compiler)
        return str(compiler)
    return None


def reset_language_expert_dispatch_triton_stats() -> None:
    global _STATS
    _STATS = _ExpertDispatchTritonStats()


def language_expert_dispatch_triton_stats() -> dict[str, Any]:
    return {
        "surface": "marulho_language_expert_dispatch_triton_stats.v1",
        "triton_available": bool(triton is not None),
        "default_min_tokens": _min_tokens(),
        "triton_forward_calls": int(_STATS.triton_forward_calls),
        "triton_forward_elements": int(_STATS.triton_forward_elements),
        "torch_fallback_calls": int(_STATS.torch_fallback_calls),
        "torch_fallback_elements": int(_STATS.torch_fallback_elements),
        "triton_failure_count": int(_STATS.triton_failure_count),
        "last_failure": _STATS.last_failure,
        "last_device": _STATS.last_device,
        "last_dtype": _STATS.last_dtype,
    }


def language_expert_dispatch_triton_stats_delta(
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
        "surface": "marulho_language_expert_dispatch_triton_stats_delta.v1",
        "triton_available": bool(after.get("triton_available")),
        **delta,
        "last_failure": after.get("last_failure"),
        "last_device": after.get("last_device"),
        "last_dtype": after.get("last_dtype"),
        "triton_kernel_used": int(delta["triton_forward_calls"]) > 0,
    }


def language_expert_dispatch_torch_reference(
    hidden: torch.Tensor,
    selected_expert_ids: torch.Tensor,
    combine_weights: torch.Tensor,
    first_weights: torch.Tensor,
    first_biases: torch.Tensor,
    second_weights: torch.Tensor,
    second_biases: torch.Tensor,
) -> torch.Tensor:
    token_count, state_dim = hidden.shape
    active_count = int(selected_expert_ids.shape[1])
    selected_flat = selected_expert_ids.reshape(-1).to(torch.long)
    selected_first_weights = first_weights.index_select(0, selected_flat).reshape(
        token_count,
        active_count,
        int(first_weights.shape[1]),
        state_dim,
    )
    selected_first_biases = first_biases.index_select(0, selected_flat).reshape(
        token_count,
        active_count,
        int(first_biases.shape[1]),
    )
    selected_second_weights = second_weights.index_select(0, selected_flat).reshape(
        token_count,
        active_count,
        state_dim,
        int(first_weights.shape[1]),
    )
    selected_second_biases = second_biases.index_select(0, selected_flat).reshape(
        token_count,
        active_count,
        state_dim,
    )
    expanded_hidden = hidden.unsqueeze(1).expand(-1, active_count, -1)
    expert_hidden = F.silu(
        torch.einsum("nkd,nkhd->nkh", expanded_hidden, selected_first_weights)
        + selected_first_biases
    )
    expert_outputs = (
        torch.einsum("nkh,nkdh->nkd", expert_hidden, selected_second_weights)
        + selected_second_biases
    )
    return (expert_outputs * combine_weights.unsqueeze(-1)).sum(dim=1)


def can_use_language_expert_dispatch_triton(
    hidden: torch.Tensor,
    selected_expert_ids: torch.Tensor,
    combine_weights: torch.Tensor,
    first_weights: torch.Tensor,
    first_biases: torch.Tensor,
    second_weights: torch.Tensor,
    second_biases: torch.Tensor,
) -> bool:
    if triton is None or tl is None:
        return False
    tensors = (
        hidden,
        combine_weights,
        first_weights,
        first_biases,
        second_weights,
        second_biases,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        return False
    if not isinstance(selected_expert_ids, torch.Tensor):
        return False
    if any(value.device.type != "cuda" for value in (*tensors, selected_expert_ids)):
        return False
    if hidden.dtype not in _SUPPORTED_DTYPES:
        return False
    if any(value.dtype != hidden.dtype for value in tensors[1:]):
        return False
    if not all(value.is_floating_point() for value in tensors):
        return False
    if hidden.ndim != 2 or selected_expert_ids.ndim != 2 or combine_weights.ndim != 2:
        return False
    token_count, state_dim = int(hidden.shape[0]), int(hidden.shape[1])
    active_count = int(selected_expert_ids.shape[1])
    if token_count <= 0 or state_dim <= 0 or active_count <= 0:
        return False
    if active_count > _MAX_ACTIVE_EXPERTS or state_dim > _MAX_STATE_DIM:
        return False
    if tuple(combine_weights.shape) != tuple(selected_expert_ids.shape):
        return False
    if first_weights.ndim != 3 or second_weights.ndim != 3:
        return False
    expert_count = int(first_weights.shape[0])
    expert_hidden_dim = int(first_weights.shape[1])
    if expert_count <= 0 or expert_hidden_dim <= 0:
        return False
    if expert_hidden_dim > _MAX_EXPERT_HIDDEN_DIM:
        return False
    return (
        int(first_weights.shape[2]) == state_dim
        and tuple(first_biases.shape) == (expert_count, expert_hidden_dim)
        and tuple(second_weights.shape) == (expert_count, state_dim, expert_hidden_dim)
        and tuple(second_biases.shape) == (expert_count, state_dim)
    )


def _min_tokens() -> int:
    raw = os.environ.get("MARULHO_LANGUAGE_EXPERT_DISPATCH_TRITON_MIN_TOKENS")
    if raw is None:
        return _DEFAULT_MIN_TOKENS
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_MIN_TOKENS


def should_use_language_expert_dispatch_triton(
    hidden: torch.Tensor,
    selected_expert_ids: torch.Tensor,
    combine_weights: torch.Tensor,
    first_weights: torch.Tensor,
    first_biases: torch.Tensor,
    second_weights: torch.Tensor,
    second_biases: torch.Tensor,
) -> bool:
    if not can_use_language_expert_dispatch_triton(
        hidden,
        selected_expert_ids,
        combine_weights,
        first_weights,
        first_biases,
        second_weights,
        second_biases,
    ):
        return False
    return int(hidden.shape[0]) >= _min_tokens()


def _next_power_of_2(value: int) -> int:
    return 1 << (max(1, int(value)) - 1).bit_length()


if triton is not None:

    @triton.jit
    def _expert_first_layer_kernel(
        hidden,
        selected_expert_ids,
        first_weights,
        first_biases,
        expert_hidden_out,
        state_dim: tl.constexpr,
        active_count: tl.constexpr,
        expert_hidden_dim: tl.constexpr,
        hidden_block: tl.constexpr,
        state_block: tl.constexpr,
    ):
        selected_row = tl.program_id(0)
        hidden_block_id = tl.program_id(1)
        token = selected_row // active_count
        active = selected_row - token * active_count
        expert_id = tl.load(selected_expert_ids + selected_row)
        h_offsets = hidden_block_id * hidden_block + tl.arange(0, hidden_block)
        d_offsets = tl.arange(0, state_block)
        h_mask = h_offsets < expert_hidden_dim
        d_mask = d_offsets < state_dim
        hidden_values = tl.load(
            hidden + token * state_dim + d_offsets,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        weight_offsets = (
            expert_id * expert_hidden_dim * state_dim
            + h_offsets[:, None] * state_dim
            + d_offsets[None, :]
        )
        weights = tl.load(
            first_weights + weight_offsets,
            mask=h_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        bias = tl.load(
            first_biases + expert_id * expert_hidden_dim + h_offsets,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        projected = tl.sum(weights * hidden_values[None, :], axis=1) + bias
        activated = projected / (1.0 + tl.exp(-projected))
        tl.store(
            expert_hidden_out + selected_row * expert_hidden_dim + h_offsets,
            activated,
            mask=h_mask,
        )

    @triton.jit
    def _expert_second_layer_kernel(
        expert_hidden,
        selected_expert_ids,
        combine_weights,
        second_weights,
        second_biases,
        output,
        state_dim: tl.constexpr,
        active_count: tl.constexpr,
        expert_hidden_dim: tl.constexpr,
        output_block: tl.constexpr,
        expert_hidden_block: tl.constexpr,
    ):
        token = tl.program_id(0)
        output_block_id = tl.program_id(1)
        d_offsets = output_block_id * output_block + tl.arange(0, output_block)
        h_offsets = tl.arange(0, expert_hidden_block)
        d_mask = d_offsets < state_dim
        h_mask = h_offsets < expert_hidden_dim
        combined = tl.zeros((output_block,), dtype=tl.float32)
        for active in tl.range(0, active_count):
            selected_row = token * active_count + active
            expert_id = tl.load(selected_expert_ids + selected_row)
            expert_hidden_values = tl.load(
                expert_hidden + selected_row * expert_hidden_dim + h_offsets,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            weight_offsets = (
                expert_id * state_dim * expert_hidden_dim
                + d_offsets[:, None] * expert_hidden_dim
                + h_offsets[None, :]
            )
            weights = tl.load(
                second_weights + weight_offsets,
                mask=d_mask[:, None] & h_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            bias = tl.load(
                second_biases + expert_id * state_dim + d_offsets,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            expert_output = tl.sum(weights * expert_hidden_values[None, :], axis=1) + bias
            combine = tl.load(combine_weights + selected_row).to(tl.float32)
            combined += expert_output * combine
        tl.store(output + token * state_dim + d_offsets, combined, mask=d_mask)


def _language_expert_dispatch_triton_forward(
    hidden: torch.Tensor,
    selected_expert_ids: torch.Tensor,
    combine_weights: torch.Tensor,
    first_weights: torch.Tensor,
    first_biases: torch.Tensor,
    second_weights: torch.Tensor,
    second_biases: torch.Tensor,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available")
    ensure_language_expert_dispatch_triton_compiler()
    runtime_hidden = hidden.contiguous()
    runtime_ids = selected_expert_ids.to(device=hidden.device, dtype=torch.long).contiguous()
    runtime_combine = combine_weights.to(device=hidden.device, dtype=hidden.dtype).contiguous()
    runtime_first_weights = first_weights.to(device=hidden.device, dtype=hidden.dtype).contiguous()
    runtime_first_biases = first_biases.to(device=hidden.device, dtype=hidden.dtype).contiguous()
    runtime_second_weights = second_weights.to(device=hidden.device, dtype=hidden.dtype).contiguous()
    runtime_second_biases = second_biases.to(device=hidden.device, dtype=hidden.dtype).contiguous()
    token_count = int(runtime_hidden.shape[0])
    state_dim = int(runtime_hidden.shape[1])
    active_count = int(runtime_ids.shape[1])
    expert_hidden_dim = int(runtime_first_weights.shape[1])
    expert_hidden = torch.empty(
        (token_count, active_count, expert_hidden_dim),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    output = torch.empty_like(runtime_hidden)
    state_block = _next_power_of_2(state_dim)
    hidden_block = min(_DEFAULT_HIDDEN_BLOCK, _next_power_of_2(expert_hidden_dim))
    output_block = min(_DEFAULT_OUTPUT_BLOCK, _next_power_of_2(state_dim))
    expert_hidden_block = _next_power_of_2(expert_hidden_dim)
    _expert_first_layer_kernel[
        (token_count * active_count, triton.cdiv(expert_hidden_dim, hidden_block))
    ](
        runtime_hidden,
        runtime_ids,
        runtime_first_weights,
        runtime_first_biases,
        expert_hidden,
        state_dim,
        active_count,
        expert_hidden_dim,
        hidden_block,
        state_block,
    )
    _expert_second_layer_kernel[(token_count, triton.cdiv(state_dim, output_block))](
        expert_hidden,
        runtime_ids,
        runtime_combine,
        runtime_second_weights,
        runtime_second_biases,
        output,
        state_dim,
        active_count,
        expert_hidden_dim,
        output_block,
        expert_hidden_block,
    )
    _STATS.triton_forward_calls += 1
    _STATS.triton_forward_elements += int(token_count * active_count * state_dim)
    _STATS.last_device = str(hidden.device)
    _STATS.last_dtype = str(hidden.dtype)
    return output


def language_expert_dispatch(
    hidden: torch.Tensor,
    selected_expert_ids: torch.Tensor,
    combine_weights: torch.Tensor,
    first_weights: torch.Tensor,
    first_biases: torch.Tensor,
    second_weights: torch.Tensor,
    second_biases: torch.Tensor,
    *,
    prefer_triton: bool = True,
    force_triton: bool = False,
) -> torch.Tensor:
    if bool(prefer_triton) and hidden.device.type == "cuda":
        use_triton = (
            can_use_language_expert_dispatch_triton(
                hidden,
                selected_expert_ids,
                combine_weights,
                first_weights,
                first_biases,
                second_weights,
                second_biases,
            )
            if bool(force_triton)
            else should_use_language_expert_dispatch_triton(
                hidden,
                selected_expert_ids,
                combine_weights,
                first_weights,
                first_biases,
                second_weights,
                second_biases,
            )
        )
        if use_triton:
            try:
                return _language_expert_dispatch_triton_forward(
                    hidden,
                    selected_expert_ids,
                    combine_weights,
                    first_weights,
                    first_biases,
                    second_weights,
                    second_biases,
                )
            except Exception as exc:  # pragma: no cover - hardware/runtime dependent
                _STATS.triton_failure_count += 1
                _STATS.last_failure = f"{type(exc).__name__}: {exc}"
                _STATS.last_device = str(hidden.device)
                _STATS.last_dtype = str(hidden.dtype)
        _STATS.torch_fallback_calls += 1
        _STATS.torch_fallback_elements += int(hidden.numel())
        _STATS.last_device = str(hidden.device)
        _STATS.last_dtype = str(hidden.dtype)
    return language_expert_dispatch_torch_reference(
        hidden,
        selected_expert_ids,
        combine_weights,
        first_weights,
        first_biases,
        second_weights,
        second_biases,
    )
