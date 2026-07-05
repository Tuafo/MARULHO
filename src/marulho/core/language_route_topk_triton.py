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
_MAX_CANDIDATES = 64
_MAX_ACTIVE_EXPERTS = 8
_DEFAULT_MIN_ROWS = 1


@dataclass
class _RouteTopKTritonStats:
    triton_forward_calls: int = 0
    triton_forward_elements: int = 0
    torch_fallback_calls: int = 0
    torch_fallback_elements: int = 0
    triton_failure_count: int = 0
    last_failure: str | None = None
    last_device: str | None = None
    last_dtype: str | None = None


_STATS = _RouteTopKTritonStats()


def ensure_language_route_topk_triton_compiler() -> str | None:
    if os.environ.get("CC"):
        return os.environ["CC"]
    if os.name != "nt" or triton is None:
        return None
    compiler = Path(triton.__file__).parent / "runtime" / "tcc" / "tcc.exe"
    if compiler.exists():
        os.environ["CC"] = str(compiler)
        return str(compiler)
    return None


def reset_language_route_topk_triton_stats() -> None:
    global _STATS
    _STATS = _RouteTopKTritonStats()


def _min_rows() -> int:
    raw = os.environ.get("MARULHO_LANGUAGE_ROUTE_TOPK_TRITON_MIN_ROWS")
    if raw is None:
        return _DEFAULT_MIN_ROWS
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_MIN_ROWS


def language_route_topk_triton_stats() -> dict[str, Any]:
    return {
        "surface": "marulho_language_route_topk_triton_stats.v1",
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


def language_route_topk_triton_stats_delta(
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
        "surface": "marulho_language_route_topk_triton_stats_delta.v1",
        "triton_available": bool(after.get("triton_available")),
        **delta,
        "last_failure": after.get("last_failure"),
        "last_device": after.get("last_device"),
        "last_dtype": after.get("last_dtype"),
        "triton_kernel_used": int(delta["triton_forward_calls"]) > 0,
    }


def language_route_topk_torch_reference(
    hidden: torch.Tensor,
    candidate_ids: torch.Tensor,
    route_keys: torch.Tensor,
    route_bias: torch.Tensor,
    active_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hidden.ndim != 2 or candidate_ids.ndim != 2:
        raise ValueError("route top-k expects hidden [rows, dim] and candidate_ids [rows, candidates]")
    candidate_ids = candidate_ids.to(device=hidden.device, dtype=torch.long)
    active = min(max(1, int(active_count)), int(candidate_ids.shape[1]))
    selected_route_keys = route_keys.index_select(0, candidate_ids.reshape(-1)).reshape(
        int(hidden.shape[0]),
        int(candidate_ids.shape[1]),
        int(hidden.shape[1]),
    )
    selected_route_bias = route_bias.index_select(0, candidate_ids.reshape(-1)).reshape(
        int(hidden.shape[0]),
        int(candidate_ids.shape[1]),
    )
    route_logits = (
        hidden.unsqueeze(-2).to(selected_route_keys.dtype) * selected_route_keys
    ).sum(dim=-1) + selected_route_bias
    top_scores, top_positions = torch.topk(route_logits, k=active, dim=-1)
    selected_expert_ids = candidate_ids.gather(dim=-1, index=top_positions)
    top_weights = F.softmax(top_scores, dim=-1)
    return selected_expert_ids, top_scores, top_weights


def can_use_language_route_topk_triton(
    hidden: torch.Tensor,
    candidate_ids: torch.Tensor,
    route_keys: torch.Tensor,
    route_bias: torch.Tensor,
    active_count: int,
) -> bool:
    if triton is None or tl is None:
        return False
    if not all(isinstance(value, torch.Tensor) for value in (hidden, candidate_ids, route_keys, route_bias)):
        return False
    if any(value.device.type != "cuda" for value in (hidden, candidate_ids, route_keys, route_bias)):
        return False
    if hidden.dtype not in _SUPPORTED_DTYPES:
        return False
    if route_keys.dtype != hidden.dtype or route_bias.dtype != hidden.dtype:
        return False
    if candidate_ids.dtype not in {torch.int64, torch.long}:
        return False
    if hidden.ndim != 2 or candidate_ids.ndim != 2 or route_keys.ndim != 2 or route_bias.ndim != 1:
        return False
    rows, state_dim = int(hidden.shape[0]), int(hidden.shape[1])
    candidate_count = int(candidate_ids.shape[1])
    expert_count = int(route_keys.shape[0])
    active = int(active_count)
    if rows <= 0 or state_dim <= 0 or candidate_count <= 0 or expert_count <= 0:
        return False
    if active <= 0 or active > candidate_count or active > _MAX_ACTIVE_EXPERTS:
        return False
    if candidate_count > _MAX_CANDIDATES or state_dim > _MAX_STATE_DIM:
        return False
    return int(route_keys.shape[1]) == state_dim and int(route_bias.shape[0]) == expert_count


def should_use_language_route_topk_triton(
    hidden: torch.Tensor,
    candidate_ids: torch.Tensor,
    route_keys: torch.Tensor,
    route_bias: torch.Tensor,
    active_count: int,
) -> bool:
    if not can_use_language_route_topk_triton(
        hidden,
        candidate_ids,
        route_keys,
        route_bias,
        active_count,
    ):
        return False
    return int(hidden.shape[0]) >= _min_rows()


def _next_power_of_2(value: int) -> int:
    return 1 << (max(1, int(value)) - 1).bit_length()


if triton is not None:

    @triton.jit
    def _route_topk_kernel(
        hidden,
        candidate_ids,
        route_keys,
        route_bias,
        selected_out,
        score_out,
        state_dim: tl.constexpr,
        candidate_count: tl.constexpr,
        active_count: tl.constexpr,
        state_block: tl.constexpr,
        candidate_block: tl.constexpr,
    ):
        row = tl.program_id(0)
        candidate_offsets = tl.arange(0, candidate_block)
        state_offsets = tl.arange(0, state_block)
        candidate_mask = candidate_offsets < candidate_count
        state_mask = state_offsets < state_dim
        expert_ids = tl.load(
            candidate_ids + row * candidate_count + candidate_offsets,
            mask=candidate_mask,
            other=0,
        ).to(tl.int64)
        hidden_values = tl.load(
            hidden + row * state_dim + state_offsets,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)
        route_values = tl.load(
            route_keys + expert_ids[:, None] * state_dim + state_offsets[None, :],
            mask=candidate_mask[:, None] & state_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(route_values * hidden_values[None, :], axis=1)
        bias = tl.load(route_bias + expert_ids, mask=candidate_mask, other=0.0).to(
            tl.float32
        )
        neg_inf = -3.4028234663852886e38
        scores = tl.where(candidate_mask, scores + bias, neg_inf)
        for active in tl.range(0, active_count):
            best_score = tl.max(scores, axis=0)
            best_positions = tl.where(
                scores == best_score,
                candidate_offsets,
                candidate_count + candidate_offsets,
            )
            best_position = tl.min(best_positions, axis=0)
            best_expert_id = tl.load(
                candidate_ids + row * candidate_count + best_position
            )
            tl.store(
                selected_out + row * active_count + active,
                best_expert_id,
            )
            tl.store(score_out + row * active_count + active, best_score)
            scores = tl.where(candidate_offsets == best_position, neg_inf, scores)


def _language_route_topk_triton_forward(
    hidden: torch.Tensor,
    candidate_ids: torch.Tensor,
    route_keys: torch.Tensor,
    route_bias: torch.Tensor,
    active_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is not available")
    ensure_language_route_topk_triton_compiler()
    runtime_hidden = hidden.contiguous()
    runtime_candidate_ids = candidate_ids.to(
        device=hidden.device,
        dtype=torch.long,
    ).contiguous()
    runtime_route_keys = route_keys.to(device=hidden.device, dtype=hidden.dtype).contiguous()
    runtime_route_bias = route_bias.to(device=hidden.device, dtype=hidden.dtype).contiguous()
    row_count = int(runtime_hidden.shape[0])
    state_dim = int(runtime_hidden.shape[1])
    candidate_count = int(runtime_candidate_ids.shape[1])
    active = min(max(1, int(active_count)), candidate_count)
    selected_out = torch.empty(
        (row_count, active),
        device=hidden.device,
        dtype=torch.long,
    )
    score_out = torch.empty(
        (row_count, active),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    state_block = _next_power_of_2(state_dim)
    candidate_block = _next_power_of_2(candidate_count)
    _route_topk_kernel[(row_count,)](
        runtime_hidden,
        runtime_candidate_ids,
        runtime_route_keys,
        runtime_route_bias,
        selected_out,
        score_out,
        state_dim,
        candidate_count,
        active,
        state_block,
        candidate_block,
    )
    _STATS.triton_forward_calls += 1
    _STATS.triton_forward_elements += int(row_count * candidate_count * state_dim)
    _STATS.last_device = str(hidden.device)
    _STATS.last_dtype = str(hidden.dtype)
    top_weights = F.softmax(score_out, dim=-1)
    return selected_out, score_out, top_weights


def language_route_topk(
    hidden: torch.Tensor,
    candidate_ids: torch.Tensor,
    route_keys: torch.Tensor,
    route_bias: torch.Tensor,
    active_count: int,
    *,
    prefer_triton: bool = True,
    force_triton: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if bool(prefer_triton) and hidden.device.type == "cuda":
        use_triton = (
            can_use_language_route_topk_triton(
                hidden,
                candidate_ids,
                route_keys,
                route_bias,
                active_count,
            )
            if bool(force_triton)
            else should_use_language_route_topk_triton(
                hidden,
                candidate_ids,
                route_keys,
                route_bias,
                active_count,
            )
        )
        if use_triton:
            try:
                return _language_route_topk_triton_forward(
                    hidden,
                    candidate_ids,
                    route_keys,
                    route_bias,
                    active_count,
                )
            except Exception as exc:  # pragma: no cover - hardware/runtime dependent
                _STATS.triton_failure_count += 1
                _STATS.last_failure = f"{type(exc).__name__}: {exc}"
                _STATS.last_device = str(hidden.device)
                _STATS.last_dtype = str(hidden.dtype)
        _STATS.torch_fallback_calls += 1
        _STATS.torch_fallback_elements += int(hidden.numel() * max(1, int(candidate_ids.shape[-1])))
        _STATS.last_device = str(hidden.device)
        _STATS.last_dtype = str(hidden.dtype)
    return language_route_topk_torch_reference(
        hidden,
        candidate_ids,
        route_keys,
        route_bias,
        active_count,
    )
