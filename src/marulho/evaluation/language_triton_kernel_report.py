"""Triton parity and benchmark evidence for MARULHO LM-head kernels."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch.nn import functional as F

from marulho.core.language_expert_dispatch_triton import (
    can_use_language_expert_dispatch_triton,
    language_expert_dispatch,
    language_expert_dispatch_torch_reference,
    language_expert_dispatch_triton_stats,
    language_expert_dispatch_triton_stats_delta,
)
from marulho.core.language_eligibility_trace_triton import (
    can_use_language_eligibility_trace_triton,
    language_eligibility_trace_final,
    language_eligibility_trace_torch_reference,
    language_eligibility_trace_triton_stats,
    language_eligibility_trace_triton_stats_delta,
)
from marulho.core.language_plif_triton import (
    can_use_language_plif_triton,
    can_use_language_plif_surrogate_triton,
    language_plif_forward,
    language_plif_surrogate_torch_reference,
    language_plif_surrogate_update,
    language_plif_torch_reference,
    language_plif_triton_stats,
    language_plif_triton_stats_delta,
)
from marulho.core.language_sampled_vocab_ce_triton import (
    build_sampled_vocab_ids,
    can_use_language_sampled_vocab_ce_triton,
    language_sampled_vocab_ce_triton_stats,
    language_sampled_vocab_ce_triton_stats_delta,
    language_sampled_vocab_cross_entropy,
    language_sampled_vocab_cross_entropy_torch_reference,
)
from marulho.core.language_selective_scan_triton import (
    can_use_language_selective_scan_triton,
    language_selective_scan,
    language_selective_scan_torch_reference,
    language_selective_scan_triton_stats,
    language_selective_scan_triton_stats_delta,
)
from marulho.core.language_rmsnorm_triton import (
    can_use_language_rmsnorm_triton,
    language_rmsnorm,
    language_rmsnorm_torch_reference,
    language_rmsnorm_triton_stats,
    language_rmsnorm_triton_stats_delta,
)
from marulho.core.language_route_topk_triton import (
    can_use_language_route_topk_triton,
    language_route_topk,
    language_route_topk_torch_reference,
    language_route_topk_triton_stats,
    language_route_topk_triton_stats_delta,
)
from marulho.reporting.readme_reports import write_json_report_with_readme


SURFACE = "marulho_language_triton_kernel_report.v1"
ARTIFACT_KIND = "marulho_language_triton_kernel_report"
RMSNORM_KERNEL_NAME = "language_rmsnorm_forward"
PLIF_FORWARD_KERNEL_NAME = "language_plif_forward"
PLIF_SURROGATE_KERNEL_NAME = "language_plif_surrogate_backward"
SELECTIVE_SCAN_KERNEL_NAME = "language_selective_state_scan"
ELIGIBILITY_TRACE_KERNEL_NAME = "language_local_eligibility_trace_update"
ROUTE_TOPK_KERNEL_NAME = "language_route_vote_topk"
EXPERT_DISPATCH_KERNEL_NAME = "language_block_sparse_expert_dispatch"
SAMPLED_VOCAB_CE_KERNEL_NAME = "language_sampled_vocab_cross_entropy"
KERNEL_NAME = RMSNORM_KERNEL_NAME


def _parse_shape(value: str) -> tuple[int, int]:
    text = str(value).lower().replace(",", "x")
    parts = [part.strip() for part in text.split("x") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("shape must be ROWSxCOLS")
    rows, cols = int(parts[0]), int(parts[1])
    if rows <= 0 or cols <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return rows, cols


def _dtype_from_name(value: str) -> torch.dtype:
    normalized = str(value).lower().replace("torch.", "")
    if normalized in {"float", "float32", "fp32"}:
        return torch.float32
    if normalized in {"half", "float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {value}")


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _benchmark_cuda(
    fn: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
    *,
    warmup: int,
    repeats: int,
) -> float:
    last: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    for _ in range(max(0, int(warmup))):
        last = fn()
    if last is not None:
        if isinstance(last, tuple):
            for item in last:
                item.detach()
        else:
            last.detach()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, int(repeats))):
        last = fn()
    end.record()
    torch.cuda.synchronize()
    if last is not None:
        if isinstance(last, tuple):
            for item in last:
                item.detach()
        else:
            last.detach()
    return float(start.elapsed_time(end) / float(max(1, int(repeats))))


def _rmsnorm_shape_kernel_result(
    *,
    rows: int,
    cols: int,
    dtype: torch.dtype,
    eps: float,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": RMSNORM_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "dtype": _dtype_name(dtype),
            "status": "unsupported_dtype_on_device",
            "parity_passed": False,
        }

    value = torch.randn(rows, cols, device="cuda", dtype=dtype)
    weight = torch.randn(cols, device="cuda", dtype=dtype)
    supported = can_use_language_rmsnorm_triton(value, weight)
    before_stats = language_rmsnorm_triton_stats()
    try:
        reference = language_rmsnorm_torch_reference(value, weight, eps=eps)
        triton_output = language_rmsnorm(
            value,
            weight,
            eps=eps,
            prefer_triton=True,
            force_triton=True,
        )
        torch.cuda.synchronize()
        diff = (triton_output.float() - reference.float()).abs()
        max_abs_error = float(diff.max().detach().cpu().item())
        denominator = reference.float().abs().clamp_min(1e-8)
        max_rel_error = float((diff / denominator).max().detach().cpu().item())
        tolerance = 1e-2 if dtype in {torch.float16, torch.bfloat16} else 1e-5
        parity_passed = bool(max_abs_error <= tolerance or max_rel_error <= tolerance)
        torch_ms = _benchmark_cuda(
            lambda: language_rmsnorm_torch_reference(value, weight, eps=eps),
            warmup=warmup,
            repeats=repeats,
        )
        triton_ms = _benchmark_cuda(
            lambda: language_rmsnorm(
                value,
                weight,
                eps=eps,
                prefer_triton=True,
                force_triton=True,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        after_stats = language_rmsnorm_triton_stats()
        stats_delta = language_rmsnorm_triton_stats_delta(before_stats, after_stats)
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": RMSNORM_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "dtype": _dtype_name(dtype),
            "status": "pass" if parity_passed and supported else "fail",
            "triton_supported": bool(supported),
            "parity_passed": bool(parity_passed and supported),
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "tolerance": float(tolerance),
            "torch_reference_ms": torch_ms,
            "triton_ms": triton_ms,
            "speedup_vs_torch": float(torch_ms / triton_ms) if triton_ms > 0.0 else 0.0,
            "stats_delta": stats_delta,
        }
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        after_stats = language_rmsnorm_triton_stats()
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": RMSNORM_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "dtype": _dtype_name(dtype),
            "status": "exception",
            "triton_supported": bool(supported),
            "parity_passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "stats_delta": language_rmsnorm_triton_stats_delta(before_stats, after_stats),
        }


def _plif_inputs(
    *,
    rows: int,
    cols: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    membrane = torch.randn(rows, cols, device="cuda", dtype=dtype)
    spikes = (torch.rand(rows, cols, device="cuda") > 0.82).to(dtype)
    selective_state = torch.randn(rows, cols, device="cuda", dtype=dtype)
    eligibility_trace = torch.randn(rows, cols, device="cuda", dtype=dtype).abs()
    leak = torch.sigmoid(torch.randn(rows, cols, device="cuda", dtype=dtype))
    threshold = F.softplus(torch.randn(rows, cols, device="cuda", dtype=dtype)) + 0.05
    drive = torch.randn(rows, cols, device="cuda", dtype=dtype)
    state_decay = torch.sigmoid(torch.randn(rows, cols, device="cuda", dtype=dtype))
    state_input = torch.sigmoid(torch.randn(rows, cols, device="cuda", dtype=dtype))
    state_output = torch.sigmoid(torch.randn(rows, cols, device="cuda", dtype=dtype))
    return {
        "membrane": membrane,
        "spikes": spikes,
        "selective_state": selective_state,
        "eligibility_trace": eligibility_trace,
        "leak": leak,
        "threshold": threshold,
        "drive": drive,
        "state_decay": state_decay,
        "state_input": state_input,
        "state_output": state_output,
    }


def _max_tuple_error(
    left: tuple[torch.Tensor, ...],
    right: tuple[torch.Tensor, ...],
) -> tuple[float, float]:
    max_abs_error = 0.0
    max_rel_error = 0.0
    for left_item, right_item in zip(left, right, strict=True):
        diff = (left_item.float() - right_item.float()).abs()
        denominator = right_item.float().abs().clamp_min(1e-8)
        max_abs_error = max(max_abs_error, float(diff.max().detach().cpu().item()))
        max_rel_error = max(
            max_rel_error,
            float((diff / denominator).max().detach().cpu().item()),
        )
    return max_abs_error, max_rel_error


def _plif_shape_kernel_result(
    *,
    rows: int,
    cols: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": PLIF_FORWARD_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "dtype": _dtype_name(dtype),
            "status": "unsupported_dtype_on_device",
            "parity_passed": False,
        }

    inputs = _plif_inputs(rows=rows, cols=cols, dtype=dtype)
    supported = can_use_language_plif_triton(**inputs)
    before_stats = language_plif_triton_stats()
    try:
        reference = language_plif_torch_reference(**inputs)
        triton_output = language_plif_forward(
            **inputs,
            prefer_triton=True,
            force_triton=True,
        )
        torch.cuda.synchronize()
        max_abs_error, max_rel_error = _max_tuple_error(triton_output, reference)
        tolerance = 1e-2 if dtype in {torch.float16, torch.bfloat16} else 1e-5
        parity_passed = bool(max_abs_error <= tolerance or max_rel_error <= tolerance)
        torch_ms = _benchmark_cuda(
            lambda: language_plif_torch_reference(**inputs),
            warmup=warmup,
            repeats=repeats,
        )
        triton_ms = _benchmark_cuda(
            lambda: language_plif_forward(
                **inputs,
                prefer_triton=True,
                force_triton=True,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        after_stats = language_plif_triton_stats()
        stats_delta = language_plif_triton_stats_delta(before_stats, after_stats)
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": PLIF_FORWARD_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "dtype": _dtype_name(dtype),
            "status": "pass" if parity_passed and supported else "fail",
            "triton_supported": bool(supported),
            "parity_passed": bool(parity_passed and supported),
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "tolerance": float(tolerance),
            "torch_reference_ms": torch_ms,
            "triton_ms": triton_ms,
            "speedup_vs_torch": float(torch_ms / triton_ms) if triton_ms > 0.0 else 0.0,
            "stats_delta": stats_delta,
        }
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        after_stats = language_plif_triton_stats()
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": PLIF_FORWARD_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "dtype": _dtype_name(dtype),
            "status": "exception",
            "triton_supported": bool(supported),
            "parity_passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "stats_delta": language_plif_triton_stats_delta(before_stats, after_stats),
        }


def _requires_grad_inputs(
    inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone().requires_grad_(True)
        for key, value in inputs.items()
    }


def _plif_grad_outputs(
    outputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    return tuple(torch.randn_like(output) for output in outputs)


def _plif_surrogate_backward_pass(
    inputs: dict[str, torch.Tensor],
    grad_outputs: tuple[torch.Tensor, ...],
    *,
    force_triton: bool,
) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
    runtime_inputs = _requires_grad_inputs(inputs)
    if force_triton:
        outputs = language_plif_surrogate_update(
            **runtime_inputs,
            prefer_triton=True,
            force_triton=True,
        )
    else:
        outputs = language_plif_surrogate_torch_reference(**runtime_inputs)
    torch.autograd.backward(outputs, grad_outputs)
    grads = {
        key: value.grad.detach().clone()
        for key, value in runtime_inputs.items()
        if value.grad is not None
    }
    return tuple(output.detach() for output in outputs), grads


def _max_grad_error(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> tuple[float, float]:
    max_abs_error = 0.0
    max_rel_error = 0.0
    for key, left_value in left.items():
        right_value = right[key]
        diff = (left_value.float() - right_value.float()).abs()
        denominator = right_value.float().abs().clamp_min(1e-8)
        max_abs_error = max(max_abs_error, float(diff.max().detach().cpu().item()))
        max_rel_error = max(
            max_rel_error,
            float((diff / denominator).max().detach().cpu().item()),
        )
    return max_abs_error, max_rel_error


def _plif_backward_benchmark_fn(
    inputs: dict[str, torch.Tensor],
    grad_outputs: tuple[torch.Tensor, ...],
    *,
    force_triton: bool,
) -> Callable[[], tuple[torch.Tensor, ...]]:
    def _run() -> tuple[torch.Tensor, ...]:
        runtime_inputs = _requires_grad_inputs(inputs)
        outputs = (
            language_plif_surrogate_update(
                **runtime_inputs,
                prefer_triton=True,
                force_triton=True,
            )
            if force_triton
            else language_plif_surrogate_torch_reference(**runtime_inputs)
        )
        torch.autograd.backward(outputs, grad_outputs)
        return tuple(
            value.grad.detach()
            for value in runtime_inputs.values()
            if value.grad is not None
        )

    return _run


def _plif_surrogate_shape_kernel_result(
    *,
    rows: int,
    cols: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if dtype is not torch.float32:
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": PLIF_SURROGATE_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "dtype": _dtype_name(dtype),
            "status": "unsupported_dtype_for_surrogate_backward",
            "parity_passed": False,
        }

    inputs = _plif_inputs(rows=rows, cols=cols, dtype=dtype)
    supported = can_use_language_plif_surrogate_triton(**inputs)
    before_stats = language_plif_triton_stats()
    try:
        reference_seed = language_plif_surrogate_torch_reference(
            **_requires_grad_inputs(inputs)
        )
        grad_outputs = _plif_grad_outputs(reference_seed)
        triton_outputs, triton_grads = _plif_surrogate_backward_pass(
            inputs,
            grad_outputs,
            force_triton=True,
        )
        reference_outputs, reference_grads = _plif_surrogate_backward_pass(
            inputs,
            grad_outputs,
            force_triton=False,
        )
        torch.cuda.synchronize()
        max_abs_error, max_rel_error = _max_tuple_error(
            triton_outputs,
            reference_outputs,
        )
        max_grad_abs_error, max_grad_rel_error = _max_grad_error(
            triton_grads,
            reference_grads,
        )
        tolerance = 5e-2 if dtype in {torch.float16, torch.bfloat16} else 1e-4
        parity_passed = bool(
            supported
            and (max_abs_error <= tolerance or max_rel_error <= tolerance)
            and (max_grad_abs_error <= tolerance or max_grad_rel_error <= tolerance)
        )
        torch_ms = _benchmark_cuda(
            _plif_backward_benchmark_fn(inputs, grad_outputs, force_triton=False),
            warmup=warmup,
            repeats=repeats,
        )
        triton_ms = _benchmark_cuda(
            _plif_backward_benchmark_fn(inputs, grad_outputs, force_triton=True),
            warmup=warmup,
            repeats=repeats,
        )
        after_stats = language_plif_triton_stats()
        stats_delta = language_plif_triton_stats_delta(before_stats, after_stats)
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": PLIF_SURROGATE_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "dtype": _dtype_name(dtype),
            "status": "pass" if parity_passed else "fail",
            "triton_supported": bool(supported),
            "parity_passed": bool(parity_passed),
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "max_grad_abs_error": max_grad_abs_error,
            "max_grad_rel_error": max_grad_rel_error,
            "tolerance": float(tolerance),
            "torch_reference_ms": torch_ms,
            "triton_ms": triton_ms,
            "speedup_vs_torch": float(torch_ms / triton_ms) if triton_ms > 0.0 else 0.0,
            "stats_delta": stats_delta,
        }
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        after_stats = language_plif_triton_stats()
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": PLIF_SURROGATE_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "dtype": _dtype_name(dtype),
            "status": "exception",
            "triton_supported": bool(supported),
            "parity_passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "stats_delta": language_plif_triton_stats_delta(before_stats, after_stats),
        }


def _selective_scan_inputs(
    *,
    rows: int,
    cols: int,
    time_steps: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    initial_state = torch.randn(rows, cols, device="cuda", dtype=dtype)
    state_decay = torch.sigmoid(
        torch.randn(rows, time_steps, cols, device="cuda", dtype=dtype)
    )
    state_input = torch.sigmoid(
        torch.randn(rows, time_steps, cols, device="cuda", dtype=dtype)
    )
    spikes = (torch.rand(rows, time_steps, cols, device="cuda") > 0.82).to(dtype)
    return {
        "initial_state": initial_state,
        "state_decay": state_decay,
        "state_input": state_input,
        "spikes": spikes,
    }


def _selective_scan_shape_kernel_result(
    *,
    rows: int,
    cols: int,
    time_steps: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": SELECTIVE_SCAN_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "time_steps": int(time_steps),
            "dtype": _dtype_name(dtype),
            "status": "unsupported_dtype_on_device",
            "parity_passed": False,
        }

    inputs = _selective_scan_inputs(
        rows=rows,
        cols=cols,
        time_steps=time_steps,
        dtype=dtype,
    )
    supported = can_use_language_selective_scan_triton(**inputs)
    before_stats = language_selective_scan_triton_stats()
    try:
        reference = language_selective_scan_torch_reference(**inputs)
        triton_output = language_selective_scan(
            **inputs,
            prefer_triton=True,
            force_triton=True,
        )
        torch.cuda.synchronize()
        max_abs_error, max_rel_error = _max_tuple_error(triton_output, reference)
        tolerance = 2e-2 if dtype in {torch.float16, torch.bfloat16} else 1e-5
        parity_passed = bool(max_abs_error <= tolerance or max_rel_error <= tolerance)
        torch_ms = _benchmark_cuda(
            lambda: language_selective_scan_torch_reference(**inputs),
            warmup=warmup,
            repeats=repeats,
        )
        triton_ms = _benchmark_cuda(
            lambda: language_selective_scan(
                **inputs,
                prefer_triton=True,
                force_triton=True,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        after_stats = language_selective_scan_triton_stats()
        stats_delta = language_selective_scan_triton_stats_delta(
            before_stats,
            after_stats,
        )
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": SELECTIVE_SCAN_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "time_steps": int(time_steps),
            "dtype": _dtype_name(dtype),
            "status": "pass" if parity_passed and supported else "fail",
            "triton_supported": bool(supported),
            "parity_passed": bool(parity_passed and supported),
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "tolerance": float(tolerance),
            "torch_reference_ms": torch_ms,
            "triton_ms": triton_ms,
            "speedup_vs_torch": float(torch_ms / triton_ms) if triton_ms > 0.0 else 0.0,
            "stats_delta": stats_delta,
        }
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        after_stats = language_selective_scan_triton_stats()
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": SELECTIVE_SCAN_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "time_steps": int(time_steps),
            "dtype": _dtype_name(dtype),
            "status": "exception",
            "triton_supported": bool(supported),
            "parity_passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "stats_delta": language_selective_scan_triton_stats_delta(
                before_stats,
                after_stats,
            ),
        }


def _eligibility_trace_inputs(
    *,
    rows: int,
    cols: int,
    time_steps: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    initial_trace = torch.randn(rows, cols, device="cuda", dtype=dtype).abs()
    spikes = (torch.rand(rows, time_steps, cols, device="cuda") > 0.82).to(dtype)
    return {
        "initial_trace": initial_trace,
        "spikes": spikes,
    }


def _eligibility_trace_shape_kernel_result(
    *,
    rows: int,
    cols: int,
    time_steps: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": ELIGIBILITY_TRACE_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "time_steps": int(time_steps),
            "dtype": _dtype_name(dtype),
            "status": "unsupported_dtype_on_device",
            "parity_passed": False,
        }

    inputs = _eligibility_trace_inputs(
        rows=rows,
        cols=cols,
        time_steps=time_steps,
        dtype=dtype,
    )
    supported = can_use_language_eligibility_trace_triton(**inputs)
    before_stats = language_eligibility_trace_triton_stats()
    try:
        reference = language_eligibility_trace_torch_reference(**inputs)
        triton_output = language_eligibility_trace_final(
            **inputs,
            prefer_triton=True,
            force_triton=True,
        )
        torch.cuda.synchronize()
        diff = (triton_output.float() - reference.float()).abs()
        max_abs_error = float(diff.max().detach().cpu().item())
        denominator = reference.float().abs().clamp_min(1e-8)
        max_rel_error = float((diff / denominator).max().detach().cpu().item())
        tolerance = 2e-2 if dtype in {torch.float16, torch.bfloat16} else 1e-5
        parity_passed = bool(max_abs_error <= tolerance or max_rel_error <= tolerance)
        torch_ms = _benchmark_cuda(
            lambda: language_eligibility_trace_torch_reference(**inputs),
            warmup=warmup,
            repeats=repeats,
        )
        triton_ms = _benchmark_cuda(
            lambda: language_eligibility_trace_final(
                **inputs,
                prefer_triton=True,
                force_triton=True,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        after_stats = language_eligibility_trace_triton_stats()
        stats_delta = language_eligibility_trace_triton_stats_delta(
            before_stats,
            after_stats,
        )
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": ELIGIBILITY_TRACE_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "time_steps": int(time_steps),
            "dtype": _dtype_name(dtype),
            "status": "pass" if parity_passed and supported else "fail",
            "triton_supported": bool(supported),
            "parity_passed": bool(parity_passed and supported),
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "tolerance": float(tolerance),
            "torch_reference_ms": torch_ms,
            "triton_ms": triton_ms,
            "speedup_vs_torch": float(torch_ms / triton_ms) if triton_ms > 0.0 else 0.0,
            "stats_delta": stats_delta,
        }
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        after_stats = language_eligibility_trace_triton_stats()
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": ELIGIBILITY_TRACE_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "time_steps": int(time_steps),
            "dtype": _dtype_name(dtype),
            "status": "exception",
            "triton_supported": bool(supported),
            "parity_passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "stats_delta": language_eligibility_trace_triton_stats_delta(
                before_stats,
                after_stats,
            ),
        }


def _expert_dispatch_inputs(
    *,
    rows: int,
    cols: int,
    expert_count: int,
    active_experts: int,
    expert_hidden_dim: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    hidden = torch.randn(rows, cols, device="cuda", dtype=dtype)
    selected_expert_ids = torch.randint(
        0,
        expert_count,
        (rows, active_experts),
        device="cuda",
        dtype=torch.long,
    )
    combine_logits = torch.randn(rows, active_experts, device="cuda", dtype=dtype)
    return {
        "hidden": hidden,
        "selected_expert_ids": selected_expert_ids,
        "combine_weights": torch.softmax(combine_logits, dim=-1),
        "first_weights": torch.randn(
            expert_count,
            expert_hidden_dim,
            cols,
            device="cuda",
            dtype=dtype,
        ),
        "first_biases": torch.randn(
            expert_count,
            expert_hidden_dim,
            device="cuda",
            dtype=dtype,
        ),
        "second_weights": torch.randn(
            expert_count,
            cols,
            expert_hidden_dim,
            device="cuda",
            dtype=dtype,
        ),
        "second_biases": torch.randn(
            expert_count,
            cols,
            device="cuda",
            dtype=dtype,
        ),
    }


def _route_topk_inputs(
    *,
    rows: int,
    cols: int,
    expert_count: int,
    route_candidate_count: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    hidden = torch.randn(rows, cols, device="cuda", dtype=dtype)
    candidate_offsets = torch.arange(
        route_candidate_count,
        device="cuda",
        dtype=torch.long,
    ).view(1, route_candidate_count)
    row_offsets = (
        torch.arange(rows, device="cuda", dtype=torch.long).view(rows, 1)
        * route_candidate_count
    )
    candidate_ids = (row_offsets + candidate_offsets).remainder(expert_count)
    return {
        "hidden": hidden,
        "candidate_ids": candidate_ids,
        "route_keys": torch.randn(expert_count, cols, device="cuda", dtype=dtype),
        "route_bias": torch.randn(expert_count, device="cuda", dtype=dtype),
    }


def _route_topk_shape_kernel_result(
    *,
    rows: int,
    cols: int,
    expert_count: int,
    route_candidate_count: int,
    active_experts: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if dtype is not torch.float32:
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": ROUTE_TOPK_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "expert_count": int(expert_count),
            "route_candidate_count": int(route_candidate_count),
            "active_experts": int(active_experts),
            "dtype": _dtype_name(dtype),
            "status": "unsupported_dtype_for_route_vote_topk",
            "parity_passed": False,
        }
    if route_candidate_count > expert_count:
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": ROUTE_TOPK_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "expert_count": int(expert_count),
            "route_candidate_count": int(route_candidate_count),
            "active_experts": int(active_experts),
            "dtype": _dtype_name(dtype),
            "status": "unsupported_route_candidate_count_gt_expert_count",
            "parity_passed": False,
        }

    inputs = _route_topk_inputs(
        rows=rows,
        cols=cols,
        expert_count=expert_count,
        route_candidate_count=route_candidate_count,
        dtype=dtype,
    )
    active_count = min(int(active_experts), int(route_candidate_count))
    supported = can_use_language_route_topk_triton(**inputs, active_count=active_count)
    before_stats = language_route_topk_triton_stats()
    try:
        reference = language_route_topk_torch_reference(
            **inputs,
            active_count=active_count,
        )
        triton_output = language_route_topk(
            **inputs,
            active_count=active_count,
            prefer_triton=True,
            force_triton=True,
        )
        torch.cuda.synchronize()
        selected_ids_equal = bool(torch.equal(triton_output[0], reference[0]))
        max_abs_error, max_rel_error = _max_tuple_error(
            (triton_output[1], triton_output[2]),
            (reference[1], reference[2]),
        )
        tolerance = 1e-4
        parity_passed = bool(
            selected_ids_equal
            and (max_abs_error <= tolerance or max_rel_error <= tolerance)
        )
        torch_ms = _benchmark_cuda(
            lambda: language_route_topk_torch_reference(
                **inputs,
                active_count=active_count,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        triton_ms = _benchmark_cuda(
            lambda: language_route_topk(
                **inputs,
                active_count=active_count,
                prefer_triton=True,
                force_triton=True,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        after_stats = language_route_topk_triton_stats()
        stats_delta = language_route_topk_triton_stats_delta(
            before_stats,
            after_stats,
        )
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": ROUTE_TOPK_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "expert_count": int(expert_count),
            "route_candidate_count": int(route_candidate_count),
            "active_experts": int(active_count),
            "dtype": _dtype_name(dtype),
            "status": "pass" if parity_passed and supported else "fail",
            "triton_supported": bool(supported),
            "parity_passed": bool(parity_passed and supported),
            "selected_ids_equal": selected_ids_equal,
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "tolerance": float(tolerance),
            "torch_reference_ms": torch_ms,
            "triton_ms": triton_ms,
            "speedup_vs_torch": float(torch_ms / triton_ms) if triton_ms > 0.0 else 0.0,
            "stats_delta": stats_delta,
        }
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        after_stats = language_route_topk_triton_stats()
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": ROUTE_TOPK_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "expert_count": int(expert_count),
            "route_candidate_count": int(route_candidate_count),
            "active_experts": int(active_count),
            "dtype": _dtype_name(dtype),
            "status": "exception",
            "triton_supported": bool(supported),
            "parity_passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "stats_delta": language_route_topk_triton_stats_delta(
                before_stats,
                after_stats,
            ),
        }


def _expert_dispatch_shape_kernel_result(
    *,
    rows: int,
    cols: int,
    expert_count: int,
    active_experts: int,
    expert_hidden_dim: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if dtype is not torch.float32:
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": EXPERT_DISPATCH_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "expert_count": int(expert_count),
            "active_experts": int(active_experts),
            "expert_hidden_dim": int(expert_hidden_dim),
            "dtype": _dtype_name(dtype),
            "status": "unsupported_dtype_for_expert_dispatch",
            "parity_passed": False,
        }
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": EXPERT_DISPATCH_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "expert_count": int(expert_count),
            "active_experts": int(active_experts),
            "expert_hidden_dim": int(expert_hidden_dim),
            "dtype": _dtype_name(dtype),
            "status": "unsupported_dtype_on_device",
            "parity_passed": False,
        }

    inputs = _expert_dispatch_inputs(
        rows=rows,
        cols=cols,
        expert_count=expert_count,
        active_experts=active_experts,
        expert_hidden_dim=expert_hidden_dim,
        dtype=dtype,
    )
    supported = can_use_language_expert_dispatch_triton(**inputs)
    before_stats = language_expert_dispatch_triton_stats()
    try:
        reference = language_expert_dispatch_torch_reference(**inputs)
        triton_output = language_expert_dispatch(
            **inputs,
            prefer_triton=True,
            force_triton=True,
        )
        torch.cuda.synchronize()
        diff = (triton_output.float() - reference.float()).abs()
        max_abs_error = float(diff.max().detach().cpu().item())
        denominator = reference.float().abs().clamp_min(1e-8)
        max_rel_error = float((diff / denominator).max().detach().cpu().item())
        tolerance = 5e-2 if dtype in {torch.float16, torch.bfloat16} else 1e-4
        parity_passed = bool(max_abs_error <= tolerance or max_rel_error <= tolerance)
        torch_ms = _benchmark_cuda(
            lambda: language_expert_dispatch_torch_reference(**inputs),
            warmup=warmup,
            repeats=repeats,
        )
        triton_ms = _benchmark_cuda(
            lambda: language_expert_dispatch(
                **inputs,
                prefer_triton=True,
                force_triton=True,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        after_stats = language_expert_dispatch_triton_stats()
        stats_delta = language_expert_dispatch_triton_stats_delta(
            before_stats,
            after_stats,
        )
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": EXPERT_DISPATCH_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "expert_count": int(expert_count),
            "active_experts": int(active_experts),
            "expert_hidden_dim": int(expert_hidden_dim),
            "dtype": _dtype_name(dtype),
            "status": "pass" if parity_passed and supported else "fail",
            "triton_supported": bool(supported),
            "parity_passed": bool(parity_passed and supported),
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "tolerance": float(tolerance),
            "torch_reference_ms": torch_ms,
            "triton_ms": triton_ms,
            "speedup_vs_torch": float(torch_ms / triton_ms) if triton_ms > 0.0 else 0.0,
            "stats_delta": stats_delta,
        }
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        after_stats = language_expert_dispatch_triton_stats()
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": EXPERT_DISPATCH_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "expert_count": int(expert_count),
            "active_experts": int(active_experts),
            "expert_hidden_dim": int(expert_hidden_dim),
            "dtype": _dtype_name(dtype),
            "status": "exception",
            "triton_supported": bool(supported),
            "parity_passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "stats_delta": language_expert_dispatch_triton_stats_delta(
                before_stats,
                after_stats,
            ),
        }


def _sampled_vocab_ce_inputs(
    *,
    rows: int,
    cols: int,
    vocab_size: int,
    sampled_vocab_size: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    hidden = torch.randn(rows, cols, device="cuda", dtype=dtype)
    lm_head_weight = torch.randn(vocab_size, cols, device="cuda", dtype=dtype)
    lm_head_bias = torch.randn(vocab_size, device="cuda", dtype=dtype)
    seed_targets = torch.arange(
        min(rows, sampled_vocab_size),
        device="cuda",
        dtype=torch.long,
    )
    sampled_vocab_ids = build_sampled_vocab_ids(
        seed_targets,
        vocab_size=vocab_size,
        sample_count=sampled_vocab_size,
        device="cuda",
    )
    target_positions = torch.randint(
        0,
        int(sampled_vocab_ids.numel()),
        (rows,),
        device="cuda",
        dtype=torch.long,
    )
    target_ids = sampled_vocab_ids.index_select(0, target_positions)
    return {
        "hidden": hidden,
        "target_ids": target_ids,
        "sampled_vocab_ids": sampled_vocab_ids,
        "lm_head_weight": lm_head_weight,
        "lm_head_bias": lm_head_bias,
    }


def _sampled_vocab_ce_shape_kernel_result(
    *,
    rows: int,
    cols: int,
    vocab_size: int,
    sampled_vocab_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if dtype is not torch.float32:
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": SAMPLED_VOCAB_CE_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "vocab_size": int(vocab_size),
            "sampled_vocab_size": int(sampled_vocab_size),
            "dtype": _dtype_name(dtype),
            "status": "unsupported_dtype_for_sampled_vocab_cross_entropy",
            "parity_passed": False,
        }
    if int(sampled_vocab_size) > int(vocab_size):
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": SAMPLED_VOCAB_CE_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "vocab_size": int(vocab_size),
            "sampled_vocab_size": int(sampled_vocab_size),
            "dtype": _dtype_name(dtype),
            "status": "invalid_sampled_vocab_size",
            "parity_passed": False,
        }

    inputs = _sampled_vocab_ce_inputs(
        rows=rows,
        cols=cols,
        vocab_size=vocab_size,
        sampled_vocab_size=sampled_vocab_size,
        dtype=dtype,
    )
    supported = can_use_language_sampled_vocab_ce_triton(**inputs)
    before_stats = language_sampled_vocab_ce_triton_stats()
    try:
        reference = language_sampled_vocab_cross_entropy_torch_reference(**inputs)
        triton_output = language_sampled_vocab_cross_entropy(
            **inputs,
            prefer_triton=True,
            force_triton=True,
        )
        torch.cuda.synchronize()
        max_abs_error = float(
            (triton_output.float() - reference.float()).abs().detach().cpu().item()
        )
        denominator = reference.float().abs().clamp_min(1e-8)
        max_rel_error = float(
            ((triton_output.float() - reference.float()).abs() / denominator)
            .detach()
            .cpu()
            .item()
        )
        tolerance = 1e-3
        parity_passed = bool(max_abs_error <= tolerance or max_rel_error <= tolerance)
        torch_ms = _benchmark_cuda(
            lambda: language_sampled_vocab_cross_entropy_torch_reference(**inputs),
            warmup=warmup,
            repeats=repeats,
        )
        triton_ms = _benchmark_cuda(
            lambda: language_sampled_vocab_cross_entropy(
                **inputs,
                prefer_triton=True,
                force_triton=True,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        after_stats = language_sampled_vocab_ce_triton_stats()
        stats_delta = language_sampled_vocab_ce_triton_stats_delta(
            before_stats,
            after_stats,
        )
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": SAMPLED_VOCAB_CE_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "vocab_size": int(vocab_size),
            "sampled_vocab_size": int(sampled_vocab_size),
            "target_coverage_count": int(
                torch.unique(inputs["target_ids"]).detach().numel()
            ),
            "dtype": _dtype_name(dtype),
            "status": "pass" if parity_passed and supported else "fail",
            "triton_supported": bool(supported),
            "parity_passed": bool(parity_passed and supported),
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "tolerance": float(tolerance),
            "torch_reference_ms": torch_ms,
            "triton_ms": triton_ms,
            "speedup_vs_torch": float(torch_ms / triton_ms) if triton_ms > 0.0 else 0.0,
            "stats_delta": stats_delta,
        }
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        after_stats = language_sampled_vocab_ce_triton_stats()
        return {
            "surface": "marulho_language_triton_kernel_shape_result.v1",
            "kernel_name": SAMPLED_VOCAB_CE_KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "vocab_size": int(vocab_size),
            "sampled_vocab_size": int(sampled_vocab_size),
            "dtype": _dtype_name(dtype),
            "status": "exception",
            "triton_supported": bool(supported),
            "parity_passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "stats_delta": language_sampled_vocab_ce_triton_stats_delta(
                before_stats,
                after_stats,
            ),
        }


def run_language_triton_kernel_report(
    *,
    output_path: str | Path,
    kernel: str = "rmsnorm-forward",
    shapes: Sequence[tuple[int, int]] = ((1024, 64), (2048, 128), (1024, 256)),
    dtypes: Sequence[str] = ("float32", "float16"),
    eps: float = 1e-6,
    scan_time_steps: int = 64,
    expert_count: int = 64,
    route_candidate_count: int = 8,
    active_experts: int = 4,
    expert_hidden_dim: int = 0,
    vocab_size: int = 8192,
    sampled_vocab_size: int = 1024,
    warmup: int = 20,
    repeats: int = 100,
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260703)
    normalized_kernel = str(kernel).strip().lower().replace("_", "-")
    if normalized_kernel in {
        "rmsnorm",
        "rmsnorm-forward",
        "language-rmsnorm-forward",
    }:
        kernel_name = RMSNORM_KERNEL_NAME
        kernel_backlog_item = "fused_rmsnorm_residual_membrane_centering_partial_rmsnorm_forward"
        selected_triton_available = bool(
            language_rmsnorm_triton_stats()["triton_available"]
        )
    elif normalized_kernel in {
        "plif",
        "plif-forward",
        "language-plif-forward",
    }:
        kernel_name = PLIF_FORWARD_KERNEL_NAME
        kernel_backlog_item = "plif_adaptive_lif_forward"
        selected_triton_available = bool(
            language_plif_triton_stats()["triton_available"]
        )
    elif normalized_kernel in {
        "plif-surrogate",
        "plif-backward",
        "plif-surrogate-backward",
        "language-plif-surrogate-backward",
    }:
        kernel_name = PLIF_SURROGATE_KERNEL_NAME
        kernel_backlog_item = "plif_adaptive_lif_backward_surrogate"
        selected_triton_available = bool(
            language_plif_triton_stats()["triton_available"]
        )
    elif normalized_kernel in {
        "selective-scan",
        "selective-state-scan",
        "language-selective-state-scan",
    }:
        kernel_name = SELECTIVE_SCAN_KERNEL_NAME
        kernel_backlog_item = "selective_recurrent_state_scan"
        selected_triton_available = bool(
            language_selective_scan_triton_stats()["triton_available"]
        )
    elif normalized_kernel in {
        "eligibility-trace",
        "local-eligibility-trace",
        "eligibility-trace-update",
        "language-local-eligibility-trace-update",
    }:
        kernel_name = ELIGIBILITY_TRACE_KERNEL_NAME
        kernel_backlog_item = "local_eligibility_trace_update"
        selected_triton_available = bool(
            language_eligibility_trace_triton_stats()["triton_available"]
        )
    elif normalized_kernel in {
        "route-topk",
        "route-vote-topk",
        "language-route-vote-topk",
    }:
        kernel_name = ROUTE_TOPK_KERNEL_NAME
        kernel_backlog_item = "route_vote_topk_candidate_selection"
        selected_triton_available = bool(
            language_route_topk_triton_stats()["triton_available"]
        )
    elif normalized_kernel in {
        "expert-dispatch",
        "block-sparse-expert-dispatch",
        "language-block-sparse-expert-dispatch",
    }:
        kernel_name = EXPERT_DISPATCH_KERNEL_NAME
        kernel_backlog_item = "block_sparse_expert_dispatch_combine"
        selected_triton_available = bool(
            language_expert_dispatch_triton_stats()["triton_available"]
        )
    elif normalized_kernel in {
        "sampled-vocab-ce",
        "sampled-vocab-cross-entropy",
        "adaptive-vocab-cross-entropy",
        "language-sampled-vocab-cross-entropy",
    }:
        kernel_name = SAMPLED_VOCAB_CE_KERNEL_NAME
        kernel_backlog_item = "sampled_adaptive_vocab_cross_entropy"
        selected_triton_available = bool(
            language_sampled_vocab_ce_triton_stats()["triton_available"]
        )
    else:
        raise ValueError(f"unsupported kernel: {kernel}")
    cuda_available = bool(torch.cuda.is_available())
    shape_results: list[dict[str, Any]] = []
    if cuda_available and selected_triton_available:
        for dtype_name in dtypes:
            dtype = _dtype_from_name(dtype_name)
            for rows, cols in shapes:
                if kernel_name == RMSNORM_KERNEL_NAME:
                    shape_results.append(
                        _rmsnorm_shape_kernel_result(
                            rows=rows,
                            cols=cols,
                            dtype=dtype,
                            eps=float(eps),
                            warmup=int(warmup),
                            repeats=int(repeats),
                        )
                    )
                elif kernel_name == PLIF_FORWARD_KERNEL_NAME:
                    shape_results.append(
                        _plif_shape_kernel_result(
                            rows=rows,
                            cols=cols,
                            dtype=dtype,
                            warmup=int(warmup),
                            repeats=int(repeats),
                        )
                    )
                elif kernel_name == PLIF_SURROGATE_KERNEL_NAME:
                    shape_results.append(
                        _plif_surrogate_shape_kernel_result(
                            rows=rows,
                            cols=cols,
                            dtype=dtype,
                            warmup=int(warmup),
                            repeats=int(repeats),
                        )
                    )
                elif kernel_name == SELECTIVE_SCAN_KERNEL_NAME:
                    shape_results.append(
                        _selective_scan_shape_kernel_result(
                            rows=rows,
                            cols=cols,
                            time_steps=int(scan_time_steps),
                            dtype=dtype,
                            warmup=int(warmup),
                            repeats=int(repeats),
                        )
                    )
                elif kernel_name == ELIGIBILITY_TRACE_KERNEL_NAME:
                    shape_results.append(
                        _eligibility_trace_shape_kernel_result(
                            rows=rows,
                            cols=cols,
                            time_steps=int(scan_time_steps),
                            dtype=dtype,
                            warmup=int(warmup),
                            repeats=int(repeats),
                        )
                    )
                elif kernel_name == ROUTE_TOPK_KERNEL_NAME:
                    shape_results.append(
                        _route_topk_shape_kernel_result(
                            rows=rows,
                            cols=cols,
                            expert_count=int(expert_count),
                            route_candidate_count=int(route_candidate_count),
                            active_experts=int(active_experts),
                            dtype=dtype,
                            warmup=int(warmup),
                            repeats=int(repeats),
                        )
                    )
                elif kernel_name == EXPERT_DISPATCH_KERNEL_NAME:
                    shape_results.append(
                        _expert_dispatch_shape_kernel_result(
                            rows=rows,
                            cols=cols,
                            expert_count=int(expert_count),
                            active_experts=int(active_experts),
                            expert_hidden_dim=(
                                int(expert_hidden_dim)
                                if int(expert_hidden_dim) > 0
                                else int(cols) * 2
                            ),
                            dtype=dtype,
                            warmup=int(warmup),
                            repeats=int(repeats),
                        )
                    )
                else:
                    shape_results.append(
                        _sampled_vocab_ce_shape_kernel_result(
                            rows=rows,
                            cols=cols,
                            vocab_size=int(vocab_size),
                            sampled_vocab_size=int(sampled_vocab_size),
                            dtype=dtype,
                            warmup=int(warmup),
                            repeats=int(repeats),
                        )
                    )
    valid_results = [
        item for item in shape_results if item.get("status") == "pass"
    ]
    failed_results = [
        item
        for item in shape_results
        if item.get("status")
        not in {
            "pass",
            "unsupported_dtype_on_device",
            "unsupported_dtype_for_surrogate_backward",
            "unsupported_dtype_for_route_vote_topk",
            "unsupported_route_candidate_count_gt_expert_count",
            "unsupported_dtype_for_expert_dispatch",
            "unsupported_dtype_for_sampled_vocab_cross_entropy",
        }
    ]
    parity_passed = (
        bool(shape_results)
        and not failed_results
        and len(valid_results) > 0
        and all(bool(item.get("parity_passed")) for item in valid_results)
    )
    speedups = [
        float(item.get("speedup_vs_torch", 0.0) or 0.0)
        for item in valid_results
        if float(item.get("speedup_vs_torch", 0.0) or 0.0) > 0.0
    ]
    geometric_speedup = (
        float(math.exp(sum(math.log(value) for value in speedups) / len(speedups)))
        if speedups
        else 0.0
    )
    status = (
        "pass"
        if parity_passed
        else (
            "unavailable"
            if not cuda_available or not selected_triton_available
            else "blocked_kernel_parity_failed"
        )
    )
    if kernel_name == RMSNORM_KERNEL_NAME:
        remaining_kernel_backlog = [
            "plif_triton_forward_parity",
            "plif_triton_backward_surrogate_parity",
            "selective_scan_triton_parity",
            "route_vote_topk_parity",
            "block_sparse_expert_dispatch_parity",
            "sampled_vocab_cross_entropy_parity",
            "local_eligibility_trace_update_parity",
        ]
    elif kernel_name == PLIF_FORWARD_KERNEL_NAME:
        remaining_kernel_backlog = [
            "plif_triton_backward_surrogate_parity",
            "selective_scan_triton_parity",
            "route_vote_topk_parity",
            "block_sparse_expert_dispatch_parity",
            "sampled_vocab_cross_entropy_parity",
            "local_eligibility_trace_update_parity",
        ]
    elif kernel_name == PLIF_SURROGATE_KERNEL_NAME:
        remaining_kernel_backlog = [
            "selective_scan_triton_parity",
            "route_vote_topk_parity",
            "block_sparse_expert_dispatch_parity",
            "sampled_vocab_cross_entropy_parity",
            "local_eligibility_trace_update_parity",
        ]
    elif kernel_name == SELECTIVE_SCAN_KERNEL_NAME:
        remaining_kernel_backlog = [
            "route_vote_topk_parity",
            "block_sparse_expert_dispatch_parity",
            "sampled_vocab_cross_entropy_parity",
            "local_eligibility_trace_update_parity",
        ]
    elif kernel_name == ELIGIBILITY_TRACE_KERNEL_NAME:
        remaining_kernel_backlog = [
            "route_vote_topk_parity",
            "block_sparse_expert_dispatch_parity",
            "sampled_vocab_cross_entropy_parity",
        ]
    elif kernel_name == ROUTE_TOPK_KERNEL_NAME:
        remaining_kernel_backlog = [
            "block_sparse_expert_dispatch_parity",
            "sampled_vocab_cross_entropy_parity",
            "local_eligibility_trace_update_parity",
        ]
    elif kernel_name == EXPERT_DISPATCH_KERNEL_NAME:
        remaining_kernel_backlog = [
            "sampled_vocab_cross_entropy_parity",
            "local_eligibility_trace_update_parity",
        ]
    elif kernel_name == SAMPLED_VOCAB_CE_KERNEL_NAME:
        remaining_kernel_backlog = ["local_eligibility_trace_update_parity"]
    else:
        remaining_kernel_backlog = []
    if not parity_passed and kernel_name == PLIF_FORWARD_KERNEL_NAME:
        remaining_kernel_backlog = [
            "plif_triton_forward_parity",
            *[
                item
                for item in remaining_kernel_backlog
                if item != "plif_triton_forward_parity"
            ],
        ]
    if not parity_passed and kernel_name == PLIF_SURROGATE_KERNEL_NAME:
        remaining_kernel_backlog = [
            "plif_triton_backward_surrogate_parity",
            *[
                item
                for item in remaining_kernel_backlog
                if item != "plif_triton_backward_surrogate_parity"
            ],
        ]
    if not parity_passed and kernel_name == SELECTIVE_SCAN_KERNEL_NAME:
        remaining_kernel_backlog = [
            "selective_scan_triton_parity",
            *[
                item
                for item in remaining_kernel_backlog
                if item != "selective_scan_triton_parity"
            ],
        ]
    if not parity_passed and kernel_name == ELIGIBILITY_TRACE_KERNEL_NAME:
        remaining_kernel_backlog = [
            "local_eligibility_trace_update_parity",
            *[
                item
                for item in remaining_kernel_backlog
                if item != "local_eligibility_trace_update_parity"
            ],
        ]
    if not parity_passed and kernel_name == ROUTE_TOPK_KERNEL_NAME:
        remaining_kernel_backlog = [
            "route_vote_topk_parity",
            *[
                item
                for item in remaining_kernel_backlog
                if item != "route_vote_topk_parity"
            ],
        ]
    if not parity_passed and kernel_name == EXPERT_DISPATCH_KERNEL_NAME:
        remaining_kernel_backlog = [
            "block_sparse_expert_dispatch_parity",
            *[
                item
                for item in remaining_kernel_backlog
                if item != "block_sparse_expert_dispatch_parity"
            ],
        ]
    if not parity_passed and kernel_name == SAMPLED_VOCAB_CE_KERNEL_NAME:
        remaining_kernel_backlog = [
            "sampled_vocab_cross_entropy_parity",
            *[
                item
                for item in remaining_kernel_backlog
                if item != "sampled_vocab_cross_entropy_parity"
            ],
        ]
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "output_path": str(output),
        "kernel_name": kernel_name,
        "kernel_backlog_item": kernel_backlog_item,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "cuda_available": cuda_available,
        "triton_available": selected_triton_available,
        "torch_cuda_device": (
            torch.cuda.get_device_name(0) if cuda_available else None
        ),
        "shape_result_count": len(shape_results),
        "valid_shape_result_count": len(valid_results),
        "failed_shape_result_count": len(failed_results),
        "parity_passed": bool(parity_passed),
        "dtype_coverage": sorted(
            {
                str(item.get("dtype"))
                for item in valid_results
                if item.get("dtype") is not None
            }
        ),
        "shape_results": shape_results,
        "benchmark_summary": {
            "surface": "marulho_language_triton_kernel_benchmark_summary.v1",
            "warmup": int(warmup),
            "repeats": int(repeats),
            "scan_time_steps": (
                int(scan_time_steps)
                if kernel_name in {SELECTIVE_SCAN_KERNEL_NAME, ELIGIBILITY_TRACE_KERNEL_NAME}
                else None
            ),
            "expert_count": (
                int(expert_count)
                if kernel_name in {ROUTE_TOPK_KERNEL_NAME, EXPERT_DISPATCH_KERNEL_NAME}
                else None
            ),
            "route_candidate_count": (
                int(route_candidate_count)
                if kernel_name == ROUTE_TOPK_KERNEL_NAME
                else None
            ),
            "active_experts": (
                int(active_experts)
                if kernel_name in {ROUTE_TOPK_KERNEL_NAME, EXPERT_DISPATCH_KERNEL_NAME}
                else None
            ),
            "expert_hidden_dim": (
                (
                    int(expert_hidden_dim)
                    if int(expert_hidden_dim) > 0
                    else "shape_cols_x2"
                )
                if kernel_name == EXPERT_DISPATCH_KERNEL_NAME
                else None
            ),
            "vocab_size": (
                int(vocab_size) if kernel_name == SAMPLED_VOCAB_CE_KERNEL_NAME else None
            ),
            "sampled_vocab_size": (
                int(sampled_vocab_size)
                if kernel_name == SAMPLED_VOCAB_CE_KERNEL_NAME
                else None
            ),
            "geometric_speedup_vs_torch": geometric_speedup,
            "speedup_is_microbenchmark_only": True,
        },
        "promotion_gate": {
            "status": (
                "partial_kernel_parity_evidence"
                if parity_passed
                else status
            ),
            "kernel_parity_available": bool(parity_passed),
            "complete_runtime_impact_available": False,
            "promotes_hot_path": False,
            "remaining_kernel_backlog": remaining_kernel_backlog,
        },
    }
    write_json_report_with_readme(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--kernel",
        choices=(
            "rmsnorm-forward",
            "plif-forward",
            "plif-surrogate",
            "selective-scan",
            "eligibility-trace",
            "route-topk",
            "expert-dispatch",
            "sampled-vocab-ce",
        ),
        default="rmsnorm-forward",
        help="Kernel evidence target.",
    )
    parser.add_argument(
        "--shape",
        type=_parse_shape,
        action="append",
        default=[],
        help="Rows and columns as ROWSxCOLS. May be passed multiple times.",
    )
    parser.add_argument(
        "--dtype",
        action="append",
        default=[],
        help="Dtype to cover: float32, float16, or bfloat16. May be passed multiple times.",
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--scan-time-steps",
        type=int,
        default=64,
        help="Time steps for selective-scan or eligibility-trace kernel evidence.",
    )
    parser.add_argument(
        "--expert-count",
        type=int,
        default=64,
        help="Total experts for route-topk or expert-dispatch kernel evidence.",
    )
    parser.add_argument(
        "--route-candidate-count",
        type=int,
        default=8,
        help="Candidate experts scored per token for route-topk evidence.",
    )
    parser.add_argument(
        "--active-experts",
        type=int,
        default=4,
        help="Active experts per token for route-topk or expert-dispatch evidence.",
    )
    parser.add_argument(
        "--expert-hidden-dim",
        type=int,
        default=0,
        help="Hidden size for expert-dispatch evidence; 0 uses 2x shape cols.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8192,
        help="Total vocabulary rows for sampled-vocab CE evidence.",
    )
    parser.add_argument(
        "--sampled-vocab-size",
        type=int,
        default=1024,
        help="Selected vocabulary rows for sampled-vocab CE evidence.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    run_language_triton_kernel_report(
        output_path=args.output,
        kernel=args.kernel,
        shapes=tuple(args.shape) if args.shape else ((1024, 64), (2048, 128), (1024, 256)),
        dtypes=tuple(args.dtype) if args.dtype else ("float32", "float16"),
        eps=args.eps,
        scan_time_steps=args.scan_time_steps,
        expert_count=args.expert_count,
        route_candidate_count=args.route_candidate_count,
        active_experts=args.active_experts,
        expert_hidden_dim=args.expert_hidden_dim,
        vocab_size=args.vocab_size,
        sampled_vocab_size=args.sampled_vocab_size,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
