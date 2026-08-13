"""Direct CUDA execution for MARULHO's V64 delta-state recurrence.

This module deliberately does not use ``torch.compile``.  Triton compiles one
small, explicit kernel whose program owns one value column of one batch/head
state matrix.  The PyTorch recurrence remains the numerical oracle.
"""

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
    def _delta_state_recurrent_forward_kernel(
        query,
        key,
        value,
        erase,
        write,
        log_decay,
        initial_state,
        output,
        final_state,
        heads: tl.constexpr,
        time: tl.constexpr,
        key_channels: tl.constexpr,
        value_channels: tl.constexpr,
        block_key: tl.constexpr,
    ):
        value_index = tl.program_id(0)
        batch_head = tl.program_id(1)
        key_offsets = tl.arange(0, block_key)
        key_mask = key_offsets < key_channels

        state_offsets = (
            batch_head * key_channels * value_channels
            + key_offsets * value_channels
            + value_index
        )
        state = tl.load(initial_state + state_offsets, mask=key_mask, other=0.0).to(
            tl.float32
        )

        for token_index in tl.static_range(0, time):
            vector_offsets = (
                (batch_head * time + token_index) * key_channels + key_offsets
            )
            query_vector = tl.load(
                query + vector_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            key_vector = tl.load(
                key + vector_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            erase_vector = tl.load(
                erase + vector_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            decay_vector = tl.load(
                log_decay + vector_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            value_offset = (
                (batch_head * time + token_index) * value_channels + value_index
            )
            current_value = tl.load(value + value_offset).to(tl.float32)
            current_write = tl.load(write + value_offset).to(tl.float32)

            state *= tl.exp(decay_vector)
            old_value = tl.sum(state * erase_vector * key_vector, axis=0)
            state += key_vector * (current_write * current_value - old_value)
            current_output = tl.sum(state * query_vector, axis=0)
            tl.store(output + value_offset, current_output)

        tl.store(final_state + state_offsets, state, mask=key_mask)


def delta_state_triton_available() -> bool:
    return bool(triton is not None and torch.cuda.is_available())


def gated_delta_state_recurrent_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    log_decay: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the exact V64 recurrent equation with one direct Triton kernel.

    This first operator is forward-only.  It intentionally rejects autograd so
    it cannot silently enter training before a matched backward kernel exists.
    """

    if triton is None:
        raise RuntimeError("V64 direct CUDA execution requires Triton")
    tensors = (query, key, value, erase, write, log_decay)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("delta-state tensors must be rank-four")
    if any(tensor.shape != query.shape for tensor in (key, erase, log_decay)):
        raise ValueError("query, key, erase, and log_decay shapes must match")
    if value.shape != write.shape:
        raise ValueError("value and write shapes must match")
    if value.shape[:3] != query.shape[:3]:
        raise ValueError("key-side and value-side batch/head/time shapes must match")
    if not query.is_cuda:
        raise ValueError("V64 direct execution requires CUDA tensors")
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("all V64 direct-execution tensors must share one device")
    if any(tensor.requires_grad for tensor in tensors) or (
        initial_state is not None and initial_state.requires_grad
    ):
        raise RuntimeError("V64 direct forward is not admitted for autograd yet")

    batch, heads, time, key_channels = map(int, query.shape)
    value_channels = int(value.shape[-1])
    state_shape = (batch, heads, key_channels, value_channels)
    state = (
        torch.zeros(state_shape, device=query.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.to(device=query.device, dtype=torch.float32)
    )
    if tuple(state.shape) != state_shape:
        raise ValueError("initial state has the wrong shape")

    contiguous = tuple(tensor.contiguous() for tensor in tensors)
    state = state.contiguous()
    output = torch.empty(
        (batch, heads, time, value_channels),
        device=query.device,
        dtype=torch.float32,
    )
    final_state = torch.empty_like(state)
    ensure_windows_triton_compiler()
    _delta_state_recurrent_forward_kernel[(value_channels, batch * heads)](
        *contiguous,
        state,
        output,
        final_state,
        heads=heads,
        time=time,
        key_channels=key_channels,
        value_channels=value_channels,
        block_key=triton.next_power_of_2(key_channels),
        num_warps=1,
    )
    return output, final_state
