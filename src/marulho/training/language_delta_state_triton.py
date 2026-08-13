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


TRAINING_STATE_CHECKPOINT_INTERVAL = 4


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
        state_checkpoints,
        time,
        checkpoint_count,
        key_channels: tl.constexpr,
        value_channels: tl.constexpr,
        block_key: tl.constexpr,
        checkpoint_interval: tl.constexpr,
        store_checkpoints: tl.constexpr,
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

        for token_index in tl.range(0, time):
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
            if store_checkpoints:
                boundary = (token_index + 1) % checkpoint_interval == 0
                checkpoint_index = (token_index + 1) // checkpoint_interval - 1
                checkpoint_offsets = (
                    (
                        batch_head * checkpoint_count + checkpoint_index
                    )
                    * key_channels
                    * value_channels
                    + key_offsets * value_channels
                    + value_index
                )
                tl.store(
                    state_checkpoints + checkpoint_offsets,
                    state,
                    mask=key_mask & boundary,
                )

        tl.store(final_state + state_offsets, state, mask=key_mask)

    @triton.jit
    def _delta_state_recurrent_backward_kernel(
        query,
        key,
        value,
        erase,
        write,
        log_decay,
        final_state,
        state_checkpoints,
        grad_output,
        grad_final_state,
        partial_gradients,
        grad_value,
        grad_write,
        grad_initial_state,
        time,
        checkpoint_count,
        key_channels: tl.constexpr,
        value_channels: tl.constexpr,
        value_blocks: tl.constexpr,
        block_key: tl.constexpr,
        block_value: tl.constexpr,
        checkpoint_interval: tl.constexpr,
    ):
        value_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        key_offsets = tl.arange(0, block_key)
        value_offsets = value_block * block_value + tl.arange(0, block_value)
        key_mask = key_offsets < key_channels
        value_mask = value_offsets < value_channels
        matrix_mask = key_mask[:, None] & value_mask[None, :]
        state_offsets = (
            batch_head * key_channels * value_channels
            + key_offsets[:, None] * value_channels
            + value_offsets[None, :]
        )
        state = tl.load(
            final_state + state_offsets, mask=matrix_mask, other=0.0
        ).to(tl.float32)
        grad_state = tl.load(
            grad_final_state + state_offsets, mask=matrix_mask, other=0.0
        ).to(tl.float32)
        partial_stride = (
            tl.num_programs(1) * time * value_blocks * key_channels
        )

        for reverse_index in tl.range(0, time):
            token_index = time - 1 - reverse_index
            boundary = ((token_index + 1) % checkpoint_interval == 0) & (
                token_index != time - 1
            )
            checkpoint_index = tl.maximum(
                (token_index + 1) // checkpoint_interval - 1, 0
            )
            checkpoint_offsets = (
                (
                    batch_head * checkpoint_count + checkpoint_index
                )
                * key_channels
                * value_channels
                + key_offsets[:, None] * value_channels
                + value_offsets[None, :]
            )
            checkpoint_state = tl.load(
                state_checkpoints + checkpoint_offsets,
                mask=matrix_mask & boundary,
                other=0.0,
            ).to(tl.float32)
            state = tl.where(boundary, checkpoint_state, state)
            key_side_offsets = (
                (batch_head * time + token_index) * key_channels + key_offsets
            )
            value_side_offsets = (
                (batch_head * time + token_index) * value_channels + value_offsets
            )
            query_vector = tl.load(
                query + key_side_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            key_vector = tl.load(
                key + key_side_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            erase_vector = tl.load(
                erase + key_side_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            decay_vector = tl.load(
                log_decay + key_side_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            value_vector = tl.load(
                value + value_side_offsets, mask=value_mask, other=0.0
            ).to(tl.float32)
            write_vector = tl.load(
                write + value_side_offsets, mask=value_mask, other=0.0
            ).to(tl.float32)
            output_gradient = tl.load(
                grad_output + value_side_offsets, mask=value_mask, other=0.0
            ).to(tl.float32)

            decay = tl.exp(decay_vector)
            effective_key = erase_vector * key_vector
            key_overlap = tl.sum(effective_key * key_vector, axis=0)
            written_value = write_vector * value_vector
            state_overlap = tl.sum(
                state * effective_key[:, None], axis=0
            )
            residual = (written_value - state_overlap) / tl.maximum(
                1.0 - key_overlap, 1.0e-6
            )
            decayed_state = state - key_vector[:, None] * residual[None, :]
            previous_state = decayed_state / tl.maximum(decay[:, None], 1.0e-20)

            partial_query = tl.sum(
                state * output_gradient[None, :], axis=1
            )
            grad_state += query_vector[:, None] * output_gradient[None, :]
            grad_residual = tl.sum(
                grad_state * key_vector[:, None], axis=0
            )
            partial_key = tl.sum(
                grad_state * residual[None, :], axis=1
            )
            partial_effective_key = tl.sum(
                -decayed_state * grad_residual[None, :], axis=1
            )
            grad_decayed = (
                grad_state
                - effective_key[:, None] * grad_residual[None, :]
            )
            partial_log_decay = (
                tl.sum(grad_decayed * previous_state, axis=1) * decay
            )
            partial_key += partial_effective_key * erase_vector
            partial_erase = partial_effective_key * key_vector
            value_gradient = grad_residual * write_vector
            write_gradient = grad_residual * value_vector
            grad_state = grad_decayed * decay[:, None]

            partial_offsets = (
                (
                    (batch_head * time + token_index) * value_blocks
                    + value_block
                )
                * key_channels
                + key_offsets
            )
            tl.store(
                partial_gradients + partial_offsets,
                partial_query,
                mask=key_mask,
            )
            tl.store(
                partial_gradients + partial_stride + partial_offsets,
                partial_key,
                mask=key_mask,
            )
            tl.store(
                partial_gradients + 2 * partial_stride + partial_offsets,
                partial_erase,
                mask=key_mask,
            )
            tl.store(
                partial_gradients + 3 * partial_stride + partial_offsets,
                partial_log_decay,
                mask=key_mask,
            )
            tl.store(
                grad_value + value_side_offsets,
                value_gradient,
                mask=value_mask,
            )
            tl.store(
                grad_write + value_side_offsets,
                write_gradient,
                mask=value_mask,
            )

            state = previous_state

        tl.store(
            grad_initial_state + state_offsets,
            grad_state,
            mask=matrix_mask,
        )

    @triton.jit
    def _reduce_delta_state_partial_gradients_kernel(
        partial_gradients,
        grad_query,
        grad_key,
        grad_erase,
        grad_log_decay,
        element_count,
        key_channels: tl.constexpr,
        value_blocks: tl.constexpr,
        block_elements: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_elements + tl.arange(0, block_elements)
        mask = offsets < element_count
        key_offsets = offsets % key_channels
        prefix = offsets // key_channels
        partial_stride = element_count * value_blocks
        query_sum = tl.zeros((block_elements,), dtype=tl.float32)
        key_sum = tl.zeros((block_elements,), dtype=tl.float32)
        erase_sum = tl.zeros((block_elements,), dtype=tl.float32)
        decay_sum = tl.zeros((block_elements,), dtype=tl.float32)
        for value_block in tl.static_range(0, value_blocks):
            partial_offsets = (
                (prefix * value_blocks + value_block) * key_channels + key_offsets
            )
            query_sum += tl.load(
                partial_gradients + partial_offsets, mask=mask, other=0.0
            )
            key_sum += tl.load(
                partial_gradients + partial_stride + partial_offsets,
                mask=mask,
                other=0.0,
            )
            erase_sum += tl.load(
                partial_gradients + 2 * partial_stride + partial_offsets,
                mask=mask,
                other=0.0,
            )
            decay_sum += tl.load(
                partial_gradients + 3 * partial_stride + partial_offsets,
                mask=mask,
                other=0.0,
            )
        tl.store(grad_query + offsets, query_sum, mask=mask)
        tl.store(grad_key + offsets, key_sum, mask=mask)
        tl.store(grad_erase + offsets, erase_sum, mask=mask)
        tl.store(grad_log_decay + offsets, decay_sum, mask=mask)


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

    output, final_state, _ = _launch_delta_state_recurrent_triton(
        tensors, state, checkpoint_interval=None
    )
    return output, final_state


def _launch_delta_state_recurrent_triton(
    tensors: tuple[torch.Tensor, ...],
    state: torch.Tensor,
    *,
    checkpoint_interval: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    query, _, value, _, _, _ = tensors
    batch, heads, time, key_channels = map(int, query.shape)
    value_channels = int(value.shape[-1])
    contiguous = tuple(tensor.contiguous() for tensor in tensors)
    state = state.contiguous()
    output = torch.empty(
        (batch, heads, time, value_channels),
        device=query.device,
        dtype=torch.float32,
    )
    final_state = torch.empty_like(state)
    interval = int(checkpoint_interval or 1)
    checkpoint_count = time // interval if checkpoint_interval is not None else 0
    checkpoints = (
        torch.empty(
            (batch, heads, checkpoint_count, key_channels, value_channels),
            device=query.device,
            dtype=torch.float32,
        )
        if checkpoint_interval is not None
        else None
    )
    checkpoint_storage = checkpoints if checkpoints is not None else final_state
    ensure_windows_triton_compiler()
    _delta_state_recurrent_forward_kernel[(value_channels, batch * heads)](
        *contiguous,
        state,
        output,
        final_state,
        checkpoint_storage,
        time=time,
        checkpoint_count=checkpoint_count,
        key_channels=key_channels,
        value_channels=value_channels,
        block_key=triton.next_power_of_2(key_channels),
        checkpoint_interval=interval,
        store_checkpoints=checkpoint_interval is not None,
        num_warps=1,
    )
    return output, final_state, checkpoints


def _gated_delta_state_recurrent_triton_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    log_decay: torch.Tensor,
    final_state: torch.Tensor,
    state_checkpoints: torch.Tensor,
    grad_output: torch.Tensor,
    grad_final_state: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("V64 direct CUDA execution requires Triton")
    batch, heads, time, key_channels = map(int, query.shape)
    value_channels = int(value.shape[-1])
    block_value = 16
    value_blocks = triton.cdiv(value_channels, block_value)
    batch_heads = batch * heads
    partial_shape = (4, batch_heads, time, value_blocks, key_channels)
    partial_gradients = torch.empty(
        partial_shape, device=query.device, dtype=torch.float32
    )
    grad_query = torch.empty_like(query)
    grad_key = torch.empty_like(key)
    grad_value = torch.empty_like(value)
    grad_erase = torch.empty_like(erase)
    grad_write = torch.empty_like(write)
    grad_log_decay = torch.empty_like(log_decay)
    grad_initial_state = torch.empty_like(final_state)
    ensure_windows_triton_compiler()
    _delta_state_recurrent_backward_kernel[(value_blocks, batch_heads)](
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        erase.contiguous(),
        write.contiguous(),
        log_decay.contiguous(),
        final_state.contiguous(),
        state_checkpoints.contiguous(),
        grad_output.contiguous(),
        grad_final_state.contiguous(),
        partial_gradients,
        grad_value,
        grad_write,
        grad_initial_state,
        time=time,
        checkpoint_count=int(state_checkpoints.shape[2]),
        key_channels=key_channels,
        value_channels=value_channels,
        value_blocks=value_blocks,
        block_key=triton.next_power_of_2(key_channels),
        block_value=block_value,
        checkpoint_interval=TRAINING_STATE_CHECKPOINT_INTERVAL,
        num_warps=8,
    )
    element_count = batch_heads * time * key_channels
    block_elements = 256
    _reduce_delta_state_partial_gradients_kernel[
        (triton.cdiv(element_count, block_elements),)
    ](
        partial_gradients,
        grad_query,
        grad_key,
        grad_erase,
        grad_log_decay,
        element_count=element_count,
        key_channels=key_channels,
        value_blocks=value_blocks,
        block_elements=block_elements,
        num_warps=4,
    )
    return (
        grad_query,
        grad_key,
        grad_value,
        grad_erase,
        grad_write,
        grad_log_decay,
        grad_initial_state,
    )


class _DeltaStateTritonAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        erase: torch.Tensor,
        write: torch.Tensor,
        log_decay: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = (
            query.detach(),
            key.detach(),
            value.detach(),
            erase.detach(),
            write.detach(),
            log_decay.detach(),
        )
        output, final_state, state_checkpoints = (
            _launch_delta_state_recurrent_triton(
                tensors,
                initial_state.detach(),
                checkpoint_interval=TRAINING_STATE_CHECKPOINT_INTERVAL,
            )
        )
        if state_checkpoints is None:  # pragma: no cover - fixed training contract
            raise RuntimeError("V64 training forward did not create state checkpoints")
        ctx.save_for_backward(
            query,
            key,
            value,
            erase,
            write,
            log_decay,
            final_state,
            state_checkpoints,
        )
        return output, final_state

    @staticmethod
    def backward(ctx, grad_output, grad_final_state):
        (
            query,
            key,
            value,
            erase,
            write,
            log_decay,
            final_state,
            state_checkpoints,
        ) = ctx.saved_tensors
        if grad_output is None:
            grad_output = torch.zeros(
                (*value.shape[:3], value.shape[-1]),
                device=value.device,
                dtype=torch.float32,
            )
        if grad_final_state is None:
            grad_final_state = torch.zeros_like(final_state)
        return _gated_delta_state_recurrent_triton_backward(
            query,
            key,
            value,
            erase,
            write,
            log_decay,
            final_state,
            state_checkpoints,
            grad_output,
            grad_final_state,
        )


def gated_delta_state_recurrent_triton_autograd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    log_decay: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the direct reversible V64 forward/backward CUDA operators."""

    batch, heads, _, key_channels = map(int, query.shape)
    value_channels = int(value.shape[-1])
    state = (
        torch.zeros(
            (batch, heads, key_channels, value_channels),
            device=query.device,
            dtype=torch.float32,
        )
        if initial_state is None
        else initial_state
    )
    return _DeltaStateTritonAutograd.apply(
        query, key, value, erase, write, log_decay, state
    )
