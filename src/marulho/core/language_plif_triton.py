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
_MAX_BLOCK_COLUMNS = 8192
_DEFAULT_MIN_TRITON_ROWS = 16


@dataclass
class _PLIFTritonStats:
    triton_forward_calls: int = 0
    triton_forward_elements: int = 0
    triton_backward_calls: int = 0
    triton_backward_elements: int = 0
    torch_fallback_calls: int = 0
    torch_fallback_elements: int = 0
    triton_failure_count: int = 0
    last_failure: str | None = None
    last_device: str | None = None
    last_dtype: str | None = None


_STATS = _PLIFTritonStats()


def ensure_language_plif_triton_compiler() -> str | None:
    if os.environ.get("CC"):
        return os.environ["CC"]
    if os.name != "nt" or triton is None:
        return None
    compiler = Path(triton.__file__).parent / "runtime" / "tcc" / "tcc.exe"
    if compiler.exists():
        os.environ["CC"] = str(compiler)
        return str(compiler)
    return None


def reset_language_plif_triton_stats() -> None:
    global _STATS
    _STATS = _PLIFTritonStats()


def language_plif_triton_stats() -> dict[str, Any]:
    return {
        "surface": "marulho_language_plif_triton_stats.v1",
        "triton_available": bool(triton is not None),
        "default_min_triton_rows": _min_triton_rows(),
        "triton_forward_calls": int(_STATS.triton_forward_calls),
        "triton_forward_elements": int(_STATS.triton_forward_elements),
        "triton_backward_calls": int(_STATS.triton_backward_calls),
        "triton_backward_elements": int(_STATS.triton_backward_elements),
        "torch_fallback_calls": int(_STATS.torch_fallback_calls),
        "torch_fallback_elements": int(_STATS.torch_fallback_elements),
        "triton_failure_count": int(_STATS.triton_failure_count),
        "last_failure": _STATS.last_failure,
        "last_device": _STATS.last_device,
        "last_dtype": _STATS.last_dtype,
    }


def language_plif_triton_stats_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    int_fields = (
        "triton_forward_calls",
        "triton_forward_elements",
        "triton_backward_calls",
        "triton_backward_elements",
        "torch_fallback_calls",
        "torch_fallback_elements",
        "triton_failure_count",
    )
    delta = {
        key: int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)
        for key in int_fields
    }
    return {
        "surface": "marulho_language_plif_triton_stats_delta.v1",
        "triton_available": bool(after.get("triton_available")),
        **delta,
        "last_failure": after.get("last_failure"),
        "last_device": after.get("last_device"),
        "last_dtype": after.get("last_dtype"),
        "triton_kernel_used": (
            int(delta["triton_forward_calls"]) > 0
            or int(delta["triton_backward_calls"]) > 0
        ),
    }


def _as_compute(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.float32)


def language_plif_torch_reference(
    *,
    membrane: torch.Tensor,
    spikes: torch.Tensor,
    selective_state: torch.Tensor,
    eligibility_trace: torch.Tensor,
    leak: torch.Tensor,
    threshold: torch.Tensor,
    drive: torch.Tensor,
    state_decay: torch.Tensor,
    state_input: torch.Tensor,
    state_output: torch.Tensor,
    eligibility_decay: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = membrane.dtype
    next_membrane = (
        _as_compute(leak) * _as_compute(membrane)
        + (1.0 - _as_compute(leak)) * _as_compute(drive)
        - _as_compute(spikes) * _as_compute(threshold)
    )
    next_spikes = (next_membrane - _as_compute(threshold) >= 0.0).to(torch.float32)
    next_selective_state = (
        _as_compute(state_decay) * _as_compute(selective_state)
        + _as_compute(state_input) * next_spikes
    )
    next_eligibility_trace = (
        float(eligibility_decay) * _as_compute(eligibility_trace) + next_spikes
    )
    mixed_state = _as_compute(state_output) * next_selective_state + next_spikes
    return (
        next_membrane.to(dtype),
        next_spikes.to(dtype),
        next_selective_state.to(dtype),
        next_eligibility_trace.to(dtype),
        mixed_state.to(dtype),
    )


def _surrogate_spike(voltage_delta: torch.Tensor, slope: float) -> torch.Tensor:
    soft = torch.sigmoid(float(slope) * voltage_delta)
    hard = (voltage_delta >= 0).to(voltage_delta.dtype)
    return hard.detach() - soft.detach() + soft


def language_plif_surrogate_torch_reference(
    *,
    membrane: torch.Tensor,
    spikes: torch.Tensor,
    selective_state: torch.Tensor,
    eligibility_trace: torch.Tensor,
    leak: torch.Tensor,
    threshold: torch.Tensor,
    drive: torch.Tensor,
    state_decay: torch.Tensor,
    state_input: torch.Tensor,
    state_output: torch.Tensor,
    eligibility_decay: float = 0.95,
    spike_slope: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    next_membrane = leak * membrane + (1.0 - leak) * drive - spikes * threshold
    next_spikes = _surrogate_spike(next_membrane - threshold, float(spike_slope))
    next_selective_state = state_decay * selective_state + state_input * next_spikes
    next_eligibility_trace = float(eligibility_decay) * eligibility_trace + next_spikes
    mixed_state = state_output * next_selective_state + next_spikes
    return (
        next_membrane,
        next_spikes,
        next_selective_state,
        next_eligibility_trace,
        mixed_state,
    )


def _same_shape(reference: torch.Tensor, values: tuple[torch.Tensor, ...]) -> bool:
    shape = tuple(reference.shape)
    return all(tuple(value.shape) == shape for value in values)


def can_use_language_plif_triton(
    *,
    membrane: torch.Tensor,
    spikes: torch.Tensor,
    selective_state: torch.Tensor,
    eligibility_trace: torch.Tensor,
    leak: torch.Tensor,
    threshold: torch.Tensor,
    drive: torch.Tensor,
    state_decay: torch.Tensor,
    state_input: torch.Tensor,
    state_output: torch.Tensor,
) -> bool:
    if triton is None or tl is None:
        return False
    tensors = (
        membrane,
        spikes,
        selective_state,
        eligibility_trace,
        leak,
        threshold,
        drive,
        state_decay,
        state_input,
        state_output,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        return False
    if not _same_shape(membrane, tensors[1:]):
        return False
    if membrane.device.type != "cuda":
        return False
    if any(value.device.type != "cuda" for value in tensors):
        return False
    if membrane.dtype not in _SUPPORTED_DTYPES:
        return False
    if any(value.dtype != membrane.dtype for value in tensors):
        return False
    if not all(value.is_floating_point() for value in tensors):
        return False
    if membrane.ndim < 1:
        return False
    cols = int(membrane.shape[-1])
    return cols > 0 and cols <= _MAX_BLOCK_COLUMNS


def _min_triton_rows() -> int:
    raw = os.environ.get("MARULHO_LANGUAGE_PLIF_TRITON_MIN_ROWS")
    if raw is None:
        return _DEFAULT_MIN_TRITON_ROWS
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_MIN_TRITON_ROWS


def should_use_language_plif_triton(
    *,
    membrane: torch.Tensor,
    spikes: torch.Tensor,
    selective_state: torch.Tensor,
    eligibility_trace: torch.Tensor,
    leak: torch.Tensor,
    threshold: torch.Tensor,
    drive: torch.Tensor,
    state_decay: torch.Tensor,
    state_input: torch.Tensor,
    state_output: torch.Tensor,
) -> bool:
    if not can_use_language_plif_triton(
        membrane=membrane,
        spikes=spikes,
        selective_state=selective_state,
        eligibility_trace=eligibility_trace,
        leak=leak,
        threshold=threshold,
        drive=drive,
        state_decay=state_decay,
        state_input=state_input,
        state_output=state_output,
    ):
        return False
    rows = int(membrane.numel() // max(1, int(membrane.shape[-1])))
    return rows >= _min_triton_rows()


def can_use_language_plif_surrogate_triton(
    *,
    membrane: torch.Tensor,
    spikes: torch.Tensor,
    selective_state: torch.Tensor,
    eligibility_trace: torch.Tensor,
    leak: torch.Tensor,
    threshold: torch.Tensor,
    drive: torch.Tensor,
    state_decay: torch.Tensor,
    state_input: torch.Tensor,
    state_output: torch.Tensor,
) -> bool:
    return (
        membrane.dtype is torch.float32
        and can_use_language_plif_triton(
            membrane=membrane,
            spikes=spikes,
            selective_state=selective_state,
            eligibility_trace=eligibility_trace,
            leak=leak,
            threshold=threshold,
            drive=drive,
            state_decay=state_decay,
            state_input=state_input,
            state_output=state_output,
        )
    )


def should_use_language_plif_surrogate_triton(
    *,
    membrane: torch.Tensor,
    spikes: torch.Tensor,
    selective_state: torch.Tensor,
    eligibility_trace: torch.Tensor,
    leak: torch.Tensor,
    threshold: torch.Tensor,
    drive: torch.Tensor,
    state_decay: torch.Tensor,
    state_input: torch.Tensor,
    state_output: torch.Tensor,
) -> bool:
    if not can_use_language_plif_surrogate_triton(
        membrane=membrane,
        spikes=spikes,
        selective_state=selective_state,
        eligibility_trace=eligibility_trace,
        leak=leak,
        threshold=threshold,
        drive=drive,
        state_decay=state_decay,
        state_input=state_input,
        state_output=state_output,
    ):
        return False
    rows = int(membrane.numel() // max(1, int(membrane.shape[-1])))
    return rows >= _min_triton_rows()


def _next_power_of_2(value: int) -> int:
    return 1 << (max(1, int(value)) - 1).bit_length()


if triton is not None:

    @triton.jit
    def _language_plif_forward_kernel(
        membrane,
        spikes,
        selective_state,
        eligibility_trace,
        leak,
        threshold,
        drive,
        state_decay,
        state_input,
        state_output,
        next_membrane_out,
        next_spikes_out,
        next_selective_state_out,
        next_eligibility_trace_out,
        mixed_state_out,
        cols: tl.constexpr,
        eligibility_decay: tl.constexpr,
        block_cols: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_cols)
        mask = offsets < cols
        base = row * cols + offsets
        old_membrane = tl.load(membrane + base, mask=mask, other=0.0).to(tl.float32)
        old_spikes = tl.load(spikes + base, mask=mask, other=0.0).to(tl.float32)
        old_selective = tl.load(selective_state + base, mask=mask, other=0.0).to(tl.float32)
        old_eligibility = tl.load(eligibility_trace + base, mask=mask, other=0.0).to(tl.float32)
        leak_value = tl.load(leak + base, mask=mask, other=0.0).to(tl.float32)
        threshold_value = tl.load(threshold + base, mask=mask, other=0.0).to(tl.float32)
        drive_value = tl.load(drive + base, mask=mask, other=0.0).to(tl.float32)
        decay_value = tl.load(state_decay + base, mask=mask, other=0.0).to(tl.float32)
        input_value = tl.load(state_input + base, mask=mask, other=0.0).to(tl.float32)
        output_value = tl.load(state_output + base, mask=mask, other=0.0).to(tl.float32)
        next_membrane = (
            leak_value * old_membrane
            + (1.0 - leak_value) * drive_value
            - old_spikes * threshold_value
        )
        next_spikes = (next_membrane - threshold_value >= 0.0).to(tl.float32)
        next_selective = decay_value * old_selective + input_value * next_spikes
        next_eligibility = eligibility_decay * old_eligibility + next_spikes
        mixed_state = output_value * next_selective + next_spikes
        tl.store(next_membrane_out + base, next_membrane, mask=mask)
        tl.store(next_spikes_out + base, next_spikes, mask=mask)
        tl.store(next_selective_state_out + base, next_selective, mask=mask)
        tl.store(next_eligibility_trace_out + base, next_eligibility, mask=mask)
        tl.store(mixed_state_out + base, mixed_state, mask=mask)

    @triton.jit
    def _language_plif_backward_kernel(
        membrane,
        spikes,
        selective_state,
        leak,
        threshold,
        drive,
        state_decay,
        state_input,
        state_output,
        next_membrane,
        next_spikes,
        next_selective_state,
        grad_next_membrane,
        grad_next_spikes,
        grad_next_selective_state,
        grad_next_eligibility_trace,
        grad_mixed_state,
        grad_membrane_out,
        grad_spikes_out,
        grad_selective_state_out,
        grad_eligibility_trace_out,
        grad_leak_out,
        grad_threshold_out,
        grad_drive_out,
        grad_state_decay_out,
        grad_state_input_out,
        grad_state_output_out,
        cols: tl.constexpr,
        eligibility_decay: tl.constexpr,
        spike_slope: tl.constexpr,
        block_cols: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_cols)
        mask = offsets < cols
        base = row * cols + offsets
        old_membrane = tl.load(membrane + base, mask=mask, other=0.0).to(tl.float32)
        old_spikes = tl.load(spikes + base, mask=mask, other=0.0).to(tl.float32)
        old_selective = tl.load(selective_state + base, mask=mask, other=0.0).to(tl.float32)
        leak_value = tl.load(leak + base, mask=mask, other=0.0).to(tl.float32)
        threshold_value = tl.load(threshold + base, mask=mask, other=0.0).to(tl.float32)
        drive_value = tl.load(drive + base, mask=mask, other=0.0).to(tl.float32)
        decay_value = tl.load(state_decay + base, mask=mask, other=0.0).to(tl.float32)
        input_value = tl.load(state_input + base, mask=mask, other=0.0).to(tl.float32)
        output_value = tl.load(state_output + base, mask=mask, other=0.0).to(tl.float32)
        next_membrane_value = tl.load(next_membrane + base, mask=mask, other=0.0).to(tl.float32)
        next_spikes_value = tl.load(next_spikes + base, mask=mask, other=0.0).to(tl.float32)
        next_selective_value = tl.load(next_selective_state + base, mask=mask, other=0.0).to(tl.float32)
        grad_membrane_direct = tl.load(grad_next_membrane + base, mask=mask, other=0.0).to(tl.float32)
        grad_spikes_direct = tl.load(grad_next_spikes + base, mask=mask, other=0.0).to(tl.float32)
        grad_selective_direct = tl.load(
            grad_next_selective_state + base,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        grad_eligibility_direct = tl.load(
            grad_next_eligibility_trace + base,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        grad_mixed = tl.load(grad_mixed_state + base, mask=mask, other=0.0).to(tl.float32)

        grad_state_output = grad_mixed * next_selective_value
        grad_selective_total = grad_selective_direct + grad_mixed * output_value
        grad_spikes_total = (
            grad_spikes_direct
            + grad_mixed
            + grad_eligibility_direct
            + grad_selective_total * input_value
        )
        grad_eligibility = grad_eligibility_direct * eligibility_decay
        grad_state_decay = grad_selective_total * old_selective
        grad_selective_state = grad_selective_total * decay_value
        grad_state_input = grad_selective_total * next_spikes_value

        voltage_delta = next_membrane_value - threshold_value
        surrogate = 1.0 / (1.0 + tl.exp(-spike_slope * voltage_delta))
        surrogate_grad = spike_slope * surrogate * (1.0 - surrogate)
        grad_voltage = grad_spikes_total * surrogate_grad
        grad_membrane_total = grad_membrane_direct + grad_voltage

        grad_leak = grad_membrane_total * (old_membrane - drive_value)
        grad_membrane = grad_membrane_total * leak_value
        grad_drive = grad_membrane_total * (1.0 - leak_value)
        grad_spikes = grad_membrane_total * (-threshold_value)
        grad_threshold = -grad_voltage + grad_membrane_total * (-old_spikes)

        tl.store(grad_membrane_out + base, grad_membrane, mask=mask)
        tl.store(grad_spikes_out + base, grad_spikes, mask=mask)
        tl.store(grad_selective_state_out + base, grad_selective_state, mask=mask)
        tl.store(grad_eligibility_trace_out + base, grad_eligibility, mask=mask)
        tl.store(grad_leak_out + base, grad_leak, mask=mask)
        tl.store(grad_threshold_out + base, grad_threshold, mask=mask)
        tl.store(grad_drive_out + base, grad_drive, mask=mask)
        tl.store(grad_state_decay_out + base, grad_state_decay, mask=mask)
        tl.store(grad_state_input_out + base, grad_state_input, mask=mask)
        tl.store(grad_state_output_out + base, grad_state_output, mask=mask)


def _language_plif_triton_forward(
    *,
    membrane: torch.Tensor,
    spikes: torch.Tensor,
    selective_state: torch.Tensor,
    eligibility_trace: torch.Tensor,
    leak: torch.Tensor,
    threshold: torch.Tensor,
    drive: torch.Tensor,
    state_decay: torch.Tensor,
    state_input: torch.Tensor,
    state_output: torch.Tensor,
    eligibility_decay: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is not available")
    ensure_language_plif_triton_compiler()
    tensors = [
        membrane.contiguous(),
        spikes.contiguous(),
        selective_state.contiguous(),
        eligibility_trace.contiguous(),
        leak.contiguous(),
        threshold.contiguous(),
        drive.contiguous(),
        state_decay.contiguous(),
        state_input.contiguous(),
        state_output.contiguous(),
    ]
    rows = int(tensors[0].numel() // int(tensors[0].shape[-1]))
    cols = int(tensors[0].shape[-1])
    output_shape = tuple(tensors[0].shape)
    next_membrane = torch.empty_like(tensors[0])
    next_spikes = torch.empty_like(tensors[0])
    next_selective_state = torch.empty_like(tensors[0])
    next_eligibility_trace = torch.empty_like(tensors[0])
    mixed_state = torch.empty_like(tensors[0])
    block_cols = _next_power_of_2(cols)
    _language_plif_forward_kernel[(rows,)](
        *[value.reshape(rows, cols) for value in tensors],
        next_membrane.reshape(rows, cols),
        next_spikes.reshape(rows, cols),
        next_selective_state.reshape(rows, cols),
        next_eligibility_trace.reshape(rows, cols),
        mixed_state.reshape(rows, cols),
        cols,
        float(eligibility_decay),
        block_cols,
    )
    _STATS.triton_forward_calls += 1
    _STATS.triton_forward_elements += int(tensors[0].numel())
    _STATS.last_device = str(membrane.device)
    _STATS.last_dtype = str(membrane.dtype)
    return (
        next_membrane.reshape(output_shape),
        next_spikes.reshape(output_shape),
        next_selective_state.reshape(output_shape),
        next_eligibility_trace.reshape(output_shape),
        mixed_state.reshape(output_shape),
    )


def _zero_grad_like(value: torch.Tensor, grad: torch.Tensor | None) -> torch.Tensor:
    if grad is None:
        return torch.zeros_like(value)
    return grad.contiguous()


def _language_plif_triton_backward(
    *,
    membrane: torch.Tensor,
    spikes: torch.Tensor,
    selective_state: torch.Tensor,
    leak: torch.Tensor,
    threshold: torch.Tensor,
    drive: torch.Tensor,
    state_decay: torch.Tensor,
    state_input: torch.Tensor,
    state_output: torch.Tensor,
    next_membrane: torch.Tensor,
    next_spikes: torch.Tensor,
    next_selective_state: torch.Tensor,
    grad_next_membrane: torch.Tensor | None,
    grad_next_spikes: torch.Tensor | None,
    grad_next_selective_state: torch.Tensor | None,
    grad_next_eligibility_trace: torch.Tensor | None,
    grad_mixed_state: torch.Tensor | None,
    eligibility_decay: float,
    spike_slope: float,
) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is not available")
    ensure_language_plif_triton_compiler()
    tensors = [
        membrane.contiguous(),
        spikes.contiguous(),
        selective_state.contiguous(),
        leak.contiguous(),
        threshold.contiguous(),
        drive.contiguous(),
        state_decay.contiguous(),
        state_input.contiguous(),
        state_output.contiguous(),
        next_membrane.contiguous(),
        next_spikes.contiguous(),
        next_selective_state.contiguous(),
    ]
    grad_tensors = [
        _zero_grad_like(next_membrane, grad_next_membrane),
        _zero_grad_like(next_spikes, grad_next_spikes),
        _zero_grad_like(next_selective_state, grad_next_selective_state),
        _zero_grad_like(next_spikes, grad_next_eligibility_trace),
        _zero_grad_like(next_spikes, grad_mixed_state),
    ]
    rows = int(tensors[0].numel() // int(tensors[0].shape[-1]))
    cols = int(tensors[0].shape[-1])
    output_shape = tuple(tensors[0].shape)
    grad_outputs = [torch.empty_like(tensors[0]) for _ in range(10)]
    block_cols = _next_power_of_2(cols)
    _language_plif_backward_kernel[(rows,)](
        *[value.reshape(rows, cols) for value in tensors],
        *[value.reshape(rows, cols) for value in grad_tensors],
        *[value.reshape(rows, cols) for value in grad_outputs],
        cols,
        float(eligibility_decay),
        float(spike_slope),
        block_cols,
    )
    _STATS.triton_backward_calls += 1
    _STATS.triton_backward_elements += int(tensors[0].numel())
    _STATS.last_device = str(membrane.device)
    _STATS.last_dtype = str(membrane.dtype)
    return tuple(value.reshape(output_shape) for value in grad_outputs)


class _LanguagePLIFSurrogateTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        membrane: torch.Tensor,
        spikes: torch.Tensor,
        selective_state: torch.Tensor,
        eligibility_trace: torch.Tensor,
        leak: torch.Tensor,
        threshold: torch.Tensor,
        drive: torch.Tensor,
        state_decay: torch.Tensor,
        state_input: torch.Tensor,
        state_output: torch.Tensor,
        eligibility_decay: float,
        spike_slope: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            next_membrane,
            next_spikes,
            next_selective_state,
            next_eligibility_trace,
            mixed_state,
        ) = _language_plif_triton_forward(
            membrane=membrane,
            spikes=spikes,
            selective_state=selective_state,
            eligibility_trace=eligibility_trace,
            leak=leak,
            threshold=threshold,
            drive=drive,
            state_decay=state_decay,
            state_input=state_input,
            state_output=state_output,
            eligibility_decay=float(eligibility_decay),
        )
        ctx.save_for_backward(
            membrane,
            spikes,
            selective_state,
            leak,
            threshold,
            drive,
            state_decay,
            state_input,
            state_output,
            next_membrane,
            next_spikes,
            next_selective_state,
        )
        ctx.eligibility_decay = float(eligibility_decay)
        ctx.spike_slope = float(spike_slope)
        return (
            next_membrane,
            next_spikes,
            next_selective_state,
            next_eligibility_trace,
            mixed_state,
        )

    @staticmethod
    def backward(
        ctx: Any,
        grad_next_membrane: torch.Tensor | None,
        grad_next_spikes: torch.Tensor | None,
        grad_next_selective_state: torch.Tensor | None,
        grad_next_eligibility_trace: torch.Tensor | None,
        grad_mixed_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        (
            membrane,
            spikes,
            selective_state,
            leak,
            threshold,
            drive,
            state_decay,
            state_input,
            state_output,
            next_membrane,
            next_spikes,
            next_selective_state,
        ) = ctx.saved_tensors
        (
            grad_membrane,
            grad_spikes,
            grad_selective_state,
            grad_eligibility_trace,
            grad_leak,
            grad_threshold,
            grad_drive,
            grad_state_decay,
            grad_state_input,
            grad_state_output,
        ) = _language_plif_triton_backward(
            membrane=membrane,
            spikes=spikes,
            selective_state=selective_state,
            leak=leak,
            threshold=threshold,
            drive=drive,
            state_decay=state_decay,
            state_input=state_input,
            state_output=state_output,
            next_membrane=next_membrane,
            next_spikes=next_spikes,
            next_selective_state=next_selective_state,
            grad_next_membrane=grad_next_membrane,
            grad_next_spikes=grad_next_spikes,
            grad_next_selective_state=grad_next_selective_state,
            grad_next_eligibility_trace=grad_next_eligibility_trace,
            grad_mixed_state=grad_mixed_state,
            eligibility_decay=float(ctx.eligibility_decay),
            spike_slope=float(ctx.spike_slope),
        )
        return (
            grad_membrane,
            grad_spikes,
            grad_selective_state,
            grad_eligibility_trace,
            grad_leak,
            grad_threshold,
            grad_drive,
            grad_state_decay,
            grad_state_input,
            grad_state_output,
            None,
            None,
        )


def language_plif_forward(
    *,
    membrane: torch.Tensor,
    spikes: torch.Tensor,
    selective_state: torch.Tensor,
    eligibility_trace: torch.Tensor,
    leak: torch.Tensor,
    threshold: torch.Tensor,
    drive: torch.Tensor,
    state_decay: torch.Tensor,
    state_input: torch.Tensor,
    state_output: torch.Tensor,
    eligibility_decay: float = 0.95,
    prefer_triton: bool = True,
    force_triton: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if bool(prefer_triton) and membrane.device.type == "cuda":
        use_triton = (
            can_use_language_plif_surrogate_triton(
                membrane=membrane,
                spikes=spikes,
                selective_state=selective_state,
                eligibility_trace=eligibility_trace,
                leak=leak,
                threshold=threshold,
                drive=drive,
                state_decay=state_decay,
                state_input=state_input,
                state_output=state_output,
            )
            if bool(force_triton)
            else should_use_language_plif_surrogate_triton(
                membrane=membrane,
                spikes=spikes,
                selective_state=selective_state,
                eligibility_trace=eligibility_trace,
                leak=leak,
                threshold=threshold,
                drive=drive,
                state_decay=state_decay,
                state_input=state_input,
                state_output=state_output,
            )
        )
        if use_triton:
            try:
                return _language_plif_triton_forward(
                    membrane=membrane,
                    spikes=spikes,
                    selective_state=selective_state,
                    eligibility_trace=eligibility_trace,
                    leak=leak,
                    threshold=threshold,
                    drive=drive,
                    state_decay=state_decay,
                    state_input=state_input,
                    state_output=state_output,
                    eligibility_decay=float(eligibility_decay),
                )
            except Exception as exc:  # pragma: no cover - hardware/runtime dependent
                _STATS.triton_failure_count += 1
                _STATS.last_failure = f"{type(exc).__name__}: {exc}"
                _STATS.last_device = str(membrane.device)
                _STATS.last_dtype = str(membrane.dtype)
        _STATS.torch_fallback_calls += 1
        _STATS.torch_fallback_elements += int(membrane.numel())
        _STATS.last_device = str(membrane.device)
        _STATS.last_dtype = str(membrane.dtype)
    return language_plif_torch_reference(
        membrane=membrane,
        spikes=spikes,
        selective_state=selective_state,
        eligibility_trace=eligibility_trace,
        leak=leak,
        threshold=threshold,
        drive=drive,
        state_decay=state_decay,
        state_input=state_input,
        state_output=state_output,
        eligibility_decay=float(eligibility_decay),
    )


def language_plif_surrogate_update(
    *,
    membrane: torch.Tensor,
    spikes: torch.Tensor,
    selective_state: torch.Tensor,
    eligibility_trace: torch.Tensor,
    leak: torch.Tensor,
    threshold: torch.Tensor,
    drive: torch.Tensor,
    state_decay: torch.Tensor,
    state_input: torch.Tensor,
    state_output: torch.Tensor,
    eligibility_decay: float = 0.95,
    spike_slope: float = 5.0,
    prefer_triton: bool = True,
    force_triton: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if bool(prefer_triton) and membrane.device.type == "cuda":
        use_triton = (
            can_use_language_plif_triton(
                membrane=membrane,
                spikes=spikes,
                selective_state=selective_state,
                eligibility_trace=eligibility_trace,
                leak=leak,
                threshold=threshold,
                drive=drive,
                state_decay=state_decay,
                state_input=state_input,
                state_output=state_output,
            )
            if bool(force_triton)
            else should_use_language_plif_triton(
                membrane=membrane,
                spikes=spikes,
                selective_state=selective_state,
                eligibility_trace=eligibility_trace,
                leak=leak,
                threshold=threshold,
                drive=drive,
                state_decay=state_decay,
                state_input=state_input,
                state_output=state_output,
            )
        )
        if use_triton:
            try:
                return _LanguagePLIFSurrogateTritonFunction.apply(
                    membrane,
                    spikes,
                    selective_state,
                    eligibility_trace,
                    leak,
                    threshold,
                    drive,
                    state_decay,
                    state_input,
                    state_output,
                    float(eligibility_decay),
                    float(spike_slope),
                )
            except Exception as exc:  # pragma: no cover - hardware/runtime dependent
                _STATS.triton_failure_count += 1
                _STATS.last_failure = f"{type(exc).__name__}: {exc}"
                _STATS.last_device = str(membrane.device)
                _STATS.last_dtype = str(membrane.dtype)
        _STATS.torch_fallback_calls += 1
        _STATS.torch_fallback_elements += int(membrane.numel())
        _STATS.last_device = str(membrane.device)
        _STATS.last_dtype = str(membrane.dtype)
    return language_plif_surrogate_torch_reference(
        membrane=membrane,
        spikes=spikes,
        selective_state=selective_state,
        eligibility_trace=eligibility_trace,
        leak=leak,
        threshold=threshold,
        drive=drive,
        state_decay=state_decay,
        state_input=state_input,
        state_output=state_output,
        eligibility_decay=float(eligibility_decay),
        spike_slope=float(spike_slope),
    )
