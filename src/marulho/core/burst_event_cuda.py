from __future__ import annotations

import torch

from marulho.core.inplace_column_cuda import ensure_windows_triton_compiler

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional CUDA dependency
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _snapshot_burst_event_kernel(
        result,
        routing_key,
        assembly,
        result_ring,
        routing_ring,
        assembly_ring,
        strong_flags,
        slot,
        strong_threshold,
        result_dim: tl.constexpr,
        routing_dim: tl.constexpr,
        assembly_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.arange(0, block_size)
        row = tl.load(slot)
        reconstruction_error = tl.load(result)
        strong = reconstruction_error >= strong_threshold

        result_mask = offsets < result_dim
        result_values = tl.load(result + offsets, mask=result_mask, other=0.0)
        tl.store(
            result_ring + row * result_dim + offsets,
            result_values,
            mask=result_mask,
        )
        routing_mask = (offsets < routing_dim) & strong
        routing_values = tl.load(
            routing_key + offsets,
            mask=routing_mask,
            other=0.0,
        )
        tl.store(
            routing_ring + row * routing_dim + offsets,
            routing_values,
            mask=routing_mask,
        )
        assembly_mask = (offsets < assembly_dim) & strong
        assembly_values = tl.load(
            assembly + offsets,
            mask=assembly_mask,
            other=0.0,
        )
        tl.store(
            assembly_ring + row * assembly_dim + offsets,
            assembly_values,
            mask=assembly_mask,
        )
        tl.store(
            strong_flags + row + offsets,
            strong,
            mask=offsets == 0,
        )


def snapshot_burst_event_cuda(
    *,
    result: torch.Tensor,
    routing_key: torch.Tensor,
    assembly: torch.Tensor,
    result_ring: torch.Tensor,
    routing_ring: torch.Tensor,
    assembly_ring: torch.Tensor,
    strong_flags: torch.Tensor,
    slot: torch.Tensor,
    strong_threshold: float,
) -> None:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    tensors = (
        result,
        routing_key,
        assembly,
        result_ring,
        routing_ring,
        assembly_ring,
        strong_flags,
        slot,
    )
    if not result.is_cuda or any(tensor.device != result.device for tensor in tensors):
        raise ValueError("burst event tensors must share one CUDA device")
    capacity = int(result_ring.shape[0])
    if capacity <= 0 or result_ring.shape != (capacity, int(result.numel())):
        raise ValueError("result_ring must be [capacity, result_dim]")
    if routing_ring.shape != (capacity, int(routing_key.numel())):
        raise ValueError("routing_ring must be [capacity, routing_dim]")
    if assembly_ring.shape != (capacity, int(assembly.numel())):
        raise ValueError("assembly_ring must be [capacity, assembly_dim]")
    if strong_flags.shape != (capacity,) or strong_flags.dtype != torch.bool:
        raise ValueError("strong_flags must be bool[capacity]")
    if slot.numel() != 1 or slot.dtype != torch.long:
        raise ValueError("slot must be one int64 value")

    ensure_windows_triton_compiler()
    block_size = triton.next_power_of_2(
        max(
            int(result.numel()),
            int(routing_key.numel()),
            int(assembly.numel()),
        )
    )
    _snapshot_burst_event_kernel[(1,)](
        result,
        routing_key,
        assembly,
        result_ring,
        routing_ring,
        assembly_ring,
        strong_flags,
        slot,
        float(strong_threshold),
        result_dim=int(result.numel()),
        routing_dim=int(routing_key.numel()),
        assembly_dim=int(assembly.numel()),
        block_size=block_size,
    )
