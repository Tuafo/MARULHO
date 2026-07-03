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
class _RMSNormTritonStats:
    triton_forward_calls: int = 0
    triton_forward_elements: int = 0
    torch_fallback_calls: int = 0
    torch_fallback_elements: int = 0
    triton_failure_count: int = 0
    last_failure: str | None = None
    last_device: str | None = None
    last_dtype: str | None = None


_STATS = _RMSNormTritonStats()


def ensure_language_rmsnorm_triton_compiler() -> str | None:
    if os.environ.get("CC"):
        return os.environ["CC"]
    if os.name != "nt" or triton is None:
        return None
    compiler = Path(triton.__file__).parent / "runtime" / "tcc" / "tcc.exe"
    if compiler.exists():
        os.environ["CC"] = str(compiler)
        return str(compiler)
    return None


def reset_language_rmsnorm_triton_stats() -> None:
    global _STATS
    _STATS = _RMSNormTritonStats()


def language_rmsnorm_triton_stats() -> dict[str, Any]:
    return {
        "surface": "marulho_language_rmsnorm_triton_stats.v1",
        "triton_available": bool(triton is not None),
        "default_min_triton_rows": _min_triton_rows(),
        "triton_forward_calls": int(_STATS.triton_forward_calls),
        "triton_forward_elements": int(_STATS.triton_forward_elements),
        "torch_fallback_calls": int(_STATS.torch_fallback_calls),
        "torch_fallback_elements": int(_STATS.torch_fallback_elements),
        "triton_failure_count": int(_STATS.triton_failure_count),
        "last_failure": _STATS.last_failure,
        "last_device": _STATS.last_device,
        "last_dtype": _STATS.last_dtype,
    }


def language_rmsnorm_triton_stats_delta(
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
        "surface": "marulho_language_rmsnorm_triton_stats_delta.v1",
        "triton_available": bool(after.get("triton_available")),
        **delta,
        "last_failure": after.get("last_failure"),
        "last_device": after.get("last_device"),
        "last_dtype": after.get("last_dtype"),
        "triton_kernel_used": int(delta["triton_forward_calls"]) > 0,
    }


def language_rmsnorm_torch_reference(
    value: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    runtime_weight = weight.to(device=value.device, dtype=value.dtype)
    rms = value.pow(2).mean(dim=-1, keepdim=True).add(float(eps)).rsqrt()
    return value * rms * runtime_weight


def can_use_language_rmsnorm_triton(value: torch.Tensor, weight: torch.Tensor) -> bool:
    if triton is None or tl is None:
        return False
    if not isinstance(value, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return False
    if value.device.type != "cuda" or weight.device.type not in {"cuda", value.device.type}:
        return False
    if value.dtype not in _SUPPORTED_DTYPES:
        return False
    if not value.is_floating_point():
        return False
    if value.ndim < 1 or weight.ndim != 1:
        return False
    cols = int(value.shape[-1])
    return cols > 0 and cols == int(weight.numel()) and cols <= _MAX_BLOCK_COLUMNS


def _min_triton_rows() -> int:
    raw = os.environ.get("MARULHO_LANGUAGE_RMSNORM_TRITON_MIN_ROWS")
    if raw is None:
        return _DEFAULT_MIN_TRITON_ROWS
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_MIN_TRITON_ROWS


def should_use_language_rmsnorm_triton(value: torch.Tensor, weight: torch.Tensor) -> bool:
    if not can_use_language_rmsnorm_triton(value, weight):
        return False
    rows = int(value.numel() // max(1, int(value.shape[-1])))
    return rows >= _min_triton_rows()


def _next_power_of_2(value: int) -> int:
    return 1 << (max(1, int(value)) - 1).bit_length()


if triton is not None:

    @triton.jit
    def _language_rmsnorm_forward_kernel(
        value,
        weight,
        output,
        cols: tl.constexpr,
        eps: tl.constexpr,
        block_cols: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_cols)
        mask = offsets < cols
        base = row * cols + offsets
        x = tl.load(value + base, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(x * x, axis=0) / cols
        scale = 1.0 / tl.sqrt(mean_square + eps)
        tl.store(output + base, x * scale * w, mask=mask)


def _language_rmsnorm_triton_forward(
    value: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available")
    ensure_language_rmsnorm_triton_compiler()
    runtime_value = value.contiguous()
    runtime_weight = weight.to(device=value.device, dtype=value.dtype).contiguous()
    rows = int(runtime_value.numel() // int(runtime_value.shape[-1]))
    cols = int(runtime_value.shape[-1])
    output = torch.empty_like(runtime_value)
    block_cols = _next_power_of_2(cols)
    _language_rmsnorm_forward_kernel[(rows,)](
        runtime_value.reshape(rows, cols),
        runtime_weight,
        output.reshape(rows, cols),
        cols,
        float(eps),
        block_cols,
    )
    _STATS.triton_forward_calls += 1
    _STATS.triton_forward_elements += int(runtime_value.numel())
    _STATS.last_device = str(value.device)
    _STATS.last_dtype = str(value.dtype)
    return output.reshape_as(value)


class _LanguageRMSNormTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        value: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        output = _language_rmsnorm_triton_forward(value, weight, eps=float(eps))
        ctx.save_for_backward(value, weight)
        ctx.eps = float(eps)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        value, weight = ctx.saved_tensors
        eps = float(ctx.eps)
        x = value.float()
        w = weight.to(device=value.device).float()
        grad = grad_output.float()
        rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True).add(eps))
        weighted_grad = grad * w
        reduction = (weighted_grad * x).sum(dim=-1, keepdim=True)
        cols = float(max(1, int(value.shape[-1])))
        grad_value = (weighted_grad * rstd) - (x * rstd.pow(3) * reduction / cols)
        reduce_dims = tuple(range(max(0, grad_output.ndim - 1)))
        grad_weight = (grad * x * rstd).sum(dim=reduce_dims)
        return grad_value.to(value.dtype), grad_weight.to(weight.dtype), None


def language_rmsnorm(
    value: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
    prefer_triton: bool = True,
    force_triton: bool = False,
) -> torch.Tensor:
    if bool(prefer_triton) and value.device.type == "cuda":
        use_triton = (
            can_use_language_rmsnorm_triton(value, weight)
            if bool(force_triton)
            else should_use_language_rmsnorm_triton(value, weight)
        )
        if use_triton:
            try:
                return _LanguageRMSNormTritonFunction.apply(value, weight, float(eps))
            except Exception as exc:  # pragma: no cover - hardware/runtime dependent
                _STATS.triton_failure_count += 1
                _STATS.last_failure = f"{type(exc).__name__}: {exc}"
                _STATS.last_device = str(value.device)
                _STATS.last_dtype = str(value.dtype)
        _STATS.torch_fallback_calls += 1
        _STATS.torch_fallback_elements += int(value.numel())
        _STATS.last_device = str(value.device)
        _STATS.last_dtype = str(value.dtype)
    return language_rmsnorm_torch_reference(value, weight, eps=float(eps))
