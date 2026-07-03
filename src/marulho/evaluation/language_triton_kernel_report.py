"""Triton parity and benchmark evidence for MARULHO LM-head kernels."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from marulho.core.language_rmsnorm_triton import (
    can_use_language_rmsnorm_triton,
    language_rmsnorm,
    language_rmsnorm_torch_reference,
    language_rmsnorm_triton_stats,
    language_rmsnorm_triton_stats_delta,
)
from marulho.reporting.readme_reports import write_json_report_with_readme


SURFACE = "marulho_language_triton_kernel_report.v1"
ARTIFACT_KIND = "marulho_language_triton_kernel_report"
KERNEL_NAME = "language_rmsnorm_forward"


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
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> float:
    last: torch.Tensor | None = None
    for _ in range(max(0, int(warmup))):
        last = fn()
    if last is not None:
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
        last.detach()
    return float(start.elapsed_time(end) / float(max(1, int(repeats))))


def _shape_kernel_result(
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
            "kernel_name": KERNEL_NAME,
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
            "kernel_name": KERNEL_NAME,
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
            "kernel_name": KERNEL_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "dtype": _dtype_name(dtype),
            "status": "exception",
            "triton_supported": bool(supported),
            "parity_passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "stats_delta": language_rmsnorm_triton_stats_delta(before_stats, after_stats),
        }


def run_language_triton_kernel_report(
    *,
    output_path: str | Path,
    shapes: Sequence[tuple[int, int]] = ((1024, 64), (2048, 128), (1024, 256)),
    dtypes: Sequence[str] = ("float32", "float16"),
    eps: float = 1e-6,
    warmup: int = 20,
    repeats: int = 100,
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260703)
    triton_available = bool(language_rmsnorm_triton_stats()["triton_available"])
    cuda_available = bool(torch.cuda.is_available())
    shape_results: list[dict[str, Any]] = []
    if cuda_available and triton_available:
        for dtype_name in dtypes:
            dtype = _dtype_from_name(dtype_name)
            for rows, cols in shapes:
                shape_results.append(
                    _shape_kernel_result(
                        rows=rows,
                        cols=cols,
                        dtype=dtype,
                        eps=float(eps),
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
        if item.get("status") not in {"pass", "unsupported_dtype_on_device"}
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
            if not cuda_available or not triton_available
            else "blocked_kernel_parity_failed"
        )
    )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "output_path": str(output),
        "kernel_name": KERNEL_NAME,
        "kernel_backlog_item": "fused_rmsnorm_residual_membrane_centering_partial_rmsnorm_forward",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "cuda_available": cuda_available,
        "triton_available": triton_available,
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
            "remaining_kernel_backlog": [
                "plif_triton_parity",
                "selective_scan_triton_parity",
                "block_sparse_expert_dispatch_parity",
                "sampled_vocab_cross_entropy_parity",
            ],
        },
    }
    write_json_report_with_readme(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
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
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    run_language_triton_kernel_report(
        output_path=args.output,
        shapes=tuple(args.shape) if args.shape else ((1024, 64), (2048, 128), (1024, 256)),
        dtypes=tuple(args.dtype) if args.dtype else ("float32", "float16"),
        eps=args.eps,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
