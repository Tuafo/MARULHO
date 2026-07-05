from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional CUDA dependency
    triton = None
    tl = None


_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_MAX_STATE_COLUMNS = 8192
_DEFAULT_BLOCK_COLUMNS = 256
_DEFAULT_MIN_ELEMENTS = 4096


@dataclass
class _EligibilityTraceTritonStats:
    triton_forward_calls: int = 0
    triton_forward_elements: int = 0
    torch_fallback_calls: int = 0
    torch_fallback_elements: int = 0
    triton_failure_count: int = 0
    last_failure: str | None = None
    last_device: str | None = None
    last_dtype: str | None = None


_STATS = _EligibilityTraceTritonStats()


def ensure_language_eligibility_trace_triton_compiler() -> str | None:
    if os.environ.get("CC"):
        return os.environ["CC"]
    if os.name != "nt" or triton is None:
        return None
    compiler = Path(triton.__file__).parent / "runtime" / "tcc" / "tcc.exe"
    if compiler.exists():
        os.environ["CC"] = str(compiler)
        return str(compiler)
    return None


def reset_language_eligibility_trace_triton_stats() -> None:
    global _STATS
    _STATS = _EligibilityTraceTritonStats()


def _min_elements() -> int:
    raw = os.environ.get("MARULHO_LANGUAGE_ELIGIBILITY_TRACE_TRITON_MIN_ELEMENTS")
    if raw is None:
        return _DEFAULT_MIN_ELEMENTS
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_MIN_ELEMENTS


def language_eligibility_trace_triton_stats() -> dict[str, Any]:
    return {
        "surface": "marulho_language_eligibility_trace_triton_stats.v1",
        "triton_available": bool(triton is not None),
        "default_min_elements": _min_elements(),
        "triton_forward_calls": int(_STATS.triton_forward_calls),
        "triton_forward_elements": int(_STATS.triton_forward_elements),
        "torch_fallback_calls": int(_STATS.torch_fallback_calls),
        "torch_fallback_elements": int(_STATS.torch_fallback_elements),
        "triton_failure_count": int(_STATS.triton_failure_count),
        "last_failure": _STATS.last_failure,
        "last_device": _STATS.last_device,
        "last_dtype": _STATS.last_dtype,
    }


def language_eligibility_trace_triton_stats_delta(
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
        "surface": "marulho_language_eligibility_trace_triton_stats_delta.v1",
        "triton_available": bool(after.get("triton_available")),
        **delta,
        "last_failure": after.get("last_failure"),
        "last_device": after.get("last_device"),
        "last_dtype": after.get("last_dtype"),
        "triton_kernel_used": int(delta["triton_forward_calls"]) > 0,
    }


def language_eligibility_trace_torch_reference(
    initial_trace: torch.Tensor,
    spikes: torch.Tensor,
    *,
    eligibility_decay: float = 0.95,
) -> torch.Tensor:
    if initial_trace.ndim != 2 or spikes.ndim != 3:
        raise ValueError("eligibility trace expects initial [batch,state] and spikes [batch,time,state]")
    dtype = initial_trace.dtype
    state = initial_trace.float()
    for step in range(int(spikes.shape[1])):
        state = float(eligibility_decay) * state + spikes[:, step, :].float()
    return state.to(dtype)


def can_use_language_eligibility_trace_triton(
    initial_trace: torch.Tensor,
    spikes: torch.Tensor,
) -> bool:
    if triton is None or tl is None:
        return False
    if not isinstance(initial_trace, torch.Tensor) or not isinstance(spikes, torch.Tensor):
        return False
    if initial_trace.ndim != 2 or spikes.ndim != 3:
        return False
    batch_size, state_dim = int(initial_trace.shape[0]), int(initial_trace.shape[1])
    if int(spikes.shape[0]) != batch_size or int(spikes.shape[2]) != state_dim:
        return False
    if initial_trace.device.type != "cuda" or spikes.device.type != "cuda":
        return False
    if initial_trace.dtype not in _SUPPORTED_DTYPES or spikes.dtype != initial_trace.dtype:
        return False
    if not initial_trace.is_floating_point() or not spikes.is_floating_point():
        return False
    return int(spikes.shape[1]) > 0 and 0 < state_dim <= _MAX_STATE_COLUMNS


def should_use_language_eligibility_trace_triton(
    initial_trace: torch.Tensor,
    spikes: torch.Tensor,
) -> bool:
    if not can_use_language_eligibility_trace_triton(initial_trace, spikes):
        return False
    return int(spikes.numel()) >= _min_elements()


def _next_power_of_2(value: int) -> int:
    return 1 << (max(1, int(value)) - 1).bit_length()


def _block_columns(cols: int) -> int:
    return _next_power_of_2(min(max(1, int(cols)), _DEFAULT_BLOCK_COLUMNS))


if triton is not None:

    @triton.jit
    def _language_eligibility_trace_final_kernel(
        initial_trace,
        spikes,
        final_trace_out,
        time_steps: tl.constexpr,
        cols: tl.constexpr,
        eligibility_decay: tl.constexpr,
        block_cols: tl.constexpr,
    ):
        batch = tl.program_id(0)
        col_block = tl.program_id(1)
        offsets = col_block * block_cols + tl.arange(0, block_cols)
        mask = offsets < cols
        base = batch * cols + offsets
        trace = tl.load(initial_trace + base, mask=mask, other=0.0).to(tl.float32)
        for step in tl.range(0, time_steps):
            sequence_base = (batch * time_steps + step) * cols + offsets
            spike_value = tl.load(spikes + sequence_base, mask=mask, other=0.0).to(tl.float32)
            trace = eligibility_decay * trace + spike_value
        tl.store(final_trace_out + base, trace, mask=mask)


def _language_eligibility_trace_triton_forward(
    initial_trace: torch.Tensor,
    spikes: torch.Tensor,
    *,
    eligibility_decay: float,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available")
    ensure_language_eligibility_trace_triton_compiler()
    runtime_initial = initial_trace.contiguous()
    runtime_spikes = spikes.contiguous()
    batch_size = int(runtime_initial.shape[0])
    cols = int(runtime_initial.shape[1])
    time_steps = int(runtime_spikes.shape[1])
    final_trace = torch.empty_like(runtime_initial)
    block_cols = _block_columns(cols)
    grid = (batch_size, triton.cdiv(cols, block_cols))
    _language_eligibility_trace_final_kernel[grid](
        runtime_initial,
        runtime_spikes,
        final_trace,
        time_steps,
        cols,
        float(eligibility_decay),
        block_cols,
    )
    _STATS.triton_forward_calls += 1
    _STATS.triton_forward_elements += int(runtime_spikes.numel())
    _STATS.last_device = str(initial_trace.device)
    _STATS.last_dtype = str(initial_trace.dtype)
    return final_trace


def language_eligibility_trace_final(
    initial_trace: torch.Tensor,
    spikes: torch.Tensor,
    *,
    eligibility_decay: float = 0.95,
    prefer_triton: bool = True,
    force_triton: bool = False,
) -> torch.Tensor:
    if bool(prefer_triton) and initial_trace.device.type == "cuda":
        use_triton = (
            can_use_language_eligibility_trace_triton(initial_trace, spikes)
            if bool(force_triton)
            else should_use_language_eligibility_trace_triton(initial_trace, spikes)
        )
        if use_triton:
            try:
                return _language_eligibility_trace_triton_forward(
                    initial_trace,
                    spikes,
                    eligibility_decay=float(eligibility_decay),
                )
            except Exception as exc:  # pragma: no cover - hardware/runtime dependent
                _STATS.triton_failure_count += 1
                _STATS.last_failure = f"{type(exc).__name__}: {exc}"
                _STATS.last_device = str(initial_trace.device)
                _STATS.last_dtype = str(initial_trace.dtype)
        _STATS.torch_fallback_calls += 1
        _STATS.torch_fallback_elements += int(spikes.numel())
        _STATS.last_device = str(initial_trace.device)
        _STATS.last_dtype = str(initial_trace.dtype)
    return language_eligibility_trace_torch_reference(
        initial_trace,
        spikes,
        eligibility_decay=float(eligibility_decay),
    )
