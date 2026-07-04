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
_DEFAULT_MIN_SCAN_ELEMENTS = 4096


@dataclass
class _SelectiveScanTritonStats:
    triton_forward_calls: int = 0
    triton_forward_elements: int = 0
    torch_fallback_calls: int = 0
    torch_fallback_elements: int = 0
    triton_failure_count: int = 0
    last_failure: str | None = None
    last_device: str | None = None
    last_dtype: str | None = None


_STATS = _SelectiveScanTritonStats()


def ensure_language_selective_scan_triton_compiler() -> str | None:
    if os.environ.get("CC"):
        return os.environ["CC"]
    if os.name != "nt" or triton is None:
        return None
    compiler = Path(triton.__file__).parent / "runtime" / "tcc" / "tcc.exe"
    if compiler.exists():
        os.environ["CC"] = str(compiler)
        return str(compiler)
    return None


def reset_language_selective_scan_triton_stats() -> None:
    global _STATS
    _STATS = _SelectiveScanTritonStats()


def language_selective_scan_triton_stats() -> dict[str, Any]:
    return {
        "surface": "marulho_language_selective_scan_triton_stats.v1",
        "triton_available": bool(triton is not None),
        "default_min_scan_elements": _min_scan_elements(),
        "triton_forward_calls": int(_STATS.triton_forward_calls),
        "triton_forward_elements": int(_STATS.triton_forward_elements),
        "torch_fallback_calls": int(_STATS.torch_fallback_calls),
        "torch_fallback_elements": int(_STATS.torch_fallback_elements),
        "triton_failure_count": int(_STATS.triton_failure_count),
        "last_failure": _STATS.last_failure,
        "last_device": _STATS.last_device,
        "last_dtype": _STATS.last_dtype,
    }


def language_selective_scan_triton_stats_delta(
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
        "surface": "marulho_language_selective_scan_triton_stats_delta.v1",
        "triton_available": bool(after.get("triton_available")),
        **delta,
        "last_failure": after.get("last_failure"),
        "last_device": after.get("last_device"),
        "last_dtype": after.get("last_dtype"),
        "triton_kernel_used": int(delta["triton_forward_calls"]) > 0,
    }


def _same_scan_shape(
    initial_state: torch.Tensor,
    values: tuple[torch.Tensor, ...],
) -> bool:
    if initial_state.ndim != 2:
        return False
    batch_size, state_dim = int(initial_state.shape[0]), int(initial_state.shape[1])
    return all(
        value.ndim == 3
        and int(value.shape[0]) == batch_size
        and int(value.shape[2]) == state_dim
        and int(value.shape[1]) == int(values[0].shape[1])
        for value in values
    )


def language_selective_scan_torch_reference(
    initial_state: torch.Tensor,
    state_decay: torch.Tensor,
    state_input: torch.Tensor,
    spikes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = initial_state.dtype
    state = initial_state.float()
    outputs: list[torch.Tensor] = []
    time_steps = int(state_decay.shape[1])
    for step in range(time_steps):
        state = (
            state_decay[:, step, :].float() * state
            + state_input[:, step, :].float() * spikes[:, step, :].float()
        )
        outputs.append(state.to(dtype))
    states = torch.stack(outputs, dim=1)
    return states, state.to(dtype)


def can_use_language_selective_scan_triton(
    initial_state: torch.Tensor,
    state_decay: torch.Tensor,
    state_input: torch.Tensor,
    spikes: torch.Tensor,
) -> bool:
    if triton is None or tl is None:
        return False
    tensors = (initial_state, state_decay, state_input, spikes)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        return False
    if not _same_scan_shape(initial_state, tensors[1:]):
        return False
    if initial_state.device.type != "cuda":
        return False
    if any(value.device.type != "cuda" for value in tensors):
        return False
    if initial_state.dtype not in _SUPPORTED_DTYPES:
        return False
    if any(value.dtype != initial_state.dtype for value in tensors):
        return False
    if not all(value.is_floating_point() for value in tensors):
        return False
    time_steps = int(state_decay.shape[1])
    state_dim = int(initial_state.shape[1])
    return time_steps > 0 and 0 < state_dim <= _MAX_STATE_COLUMNS


def _min_scan_elements() -> int:
    raw = os.environ.get("MARULHO_LANGUAGE_SELECTIVE_SCAN_TRITON_MIN_ELEMENTS")
    if raw is None:
        return _DEFAULT_MIN_SCAN_ELEMENTS
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_MIN_SCAN_ELEMENTS


def should_use_language_selective_scan_triton(
    initial_state: torch.Tensor,
    state_decay: torch.Tensor,
    state_input: torch.Tensor,
    spikes: torch.Tensor,
) -> bool:
    if not can_use_language_selective_scan_triton(
        initial_state,
        state_decay,
        state_input,
        spikes,
    ):
        return False
    return int(state_decay.numel()) >= _min_scan_elements()


def _next_power_of_2(value: int) -> int:
    return 1 << (max(1, int(value)) - 1).bit_length()


def _block_columns(cols: int) -> int:
    return _next_power_of_2(min(max(1, int(cols)), _DEFAULT_BLOCK_COLUMNS))


if triton is not None:

    @triton.jit
    def _language_selective_scan_forward_kernel(
        initial_state,
        state_decay,
        state_input,
        spikes,
        states_out,
        final_state_out,
        time_steps: tl.constexpr,
        cols: tl.constexpr,
        block_cols: tl.constexpr,
    ):
        batch = tl.program_id(0)
        col_block = tl.program_id(1)
        offsets = col_block * block_cols + tl.arange(0, block_cols)
        mask = offsets < cols
        state_base = batch * cols + offsets
        state = tl.load(initial_state + state_base, mask=mask, other=0.0).to(tl.float32)
        for step in tl.range(0, time_steps):
            sequence_base = (batch * time_steps + step) * cols + offsets
            decay_value = tl.load(
                state_decay + sequence_base,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            input_value = tl.load(
                state_input + sequence_base,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            spike_value = tl.load(
                spikes + sequence_base,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            state = decay_value * state + input_value * spike_value
            tl.store(states_out + sequence_base, state, mask=mask)
        tl.store(final_state_out + state_base, state, mask=mask)


def _language_selective_scan_triton_forward(
    initial_state: torch.Tensor,
    state_decay: torch.Tensor,
    state_input: torch.Tensor,
    spikes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is not available")
    ensure_language_selective_scan_triton_compiler()
    runtime_initial = initial_state.contiguous()
    runtime_decay = state_decay.contiguous()
    runtime_input = state_input.contiguous()
    runtime_spikes = spikes.contiguous()
    batch_size = int(runtime_initial.shape[0])
    cols = int(runtime_initial.shape[1])
    time_steps = int(runtime_decay.shape[1])
    states = torch.empty_like(runtime_decay)
    final_state = torch.empty_like(runtime_initial)
    block_cols = _block_columns(cols)
    grid = (batch_size, triton.cdiv(cols, block_cols))
    _language_selective_scan_forward_kernel[grid](
        runtime_initial,
        runtime_decay,
        runtime_input,
        runtime_spikes,
        states,
        final_state,
        time_steps,
        cols,
        block_cols,
    )
    _STATS.triton_forward_calls += 1
    _STATS.triton_forward_elements += int(runtime_decay.numel())
    _STATS.last_device = str(initial_state.device)
    _STATS.last_dtype = str(initial_state.dtype)
    return states, final_state


def language_selective_scan(
    initial_state: torch.Tensor,
    state_decay: torch.Tensor,
    state_input: torch.Tensor,
    spikes: torch.Tensor,
    *,
    prefer_triton: bool = True,
    force_triton: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bool(prefer_triton) and initial_state.device.type == "cuda":
        use_triton = (
            can_use_language_selective_scan_triton(
                initial_state,
                state_decay,
                state_input,
                spikes,
            )
            if bool(force_triton)
            else should_use_language_selective_scan_triton(
                initial_state,
                state_decay,
                state_input,
                spikes,
            )
        )
        if use_triton:
            try:
                return _language_selective_scan_triton_forward(
                    initial_state,
                    state_decay,
                    state_input,
                    spikes,
                )
            except Exception as exc:  # pragma: no cover - hardware/runtime dependent
                _STATS.triton_failure_count += 1
                _STATS.last_failure = f"{type(exc).__name__}: {exc}"
                _STATS.last_device = str(initial_state.device)
                _STATS.last_dtype = str(initial_state.dtype)
        _STATS.torch_fallback_calls += 1
        _STATS.torch_fallback_elements += int(state_decay.numel())
        _STATS.last_device = str(initial_state.device)
        _STATS.last_dtype = str(initial_state.dtype)
    return language_selective_scan_torch_reference(
        initial_state,
        state_decay,
        state_input,
        spikes,
    )
