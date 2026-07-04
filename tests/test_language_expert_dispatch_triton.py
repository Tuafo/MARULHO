from __future__ import annotations

import pytest
import torch

from marulho.core.language_expert_dispatch_triton import (
    language_expert_dispatch,
    language_expert_dispatch_torch_reference,
    language_expert_dispatch_triton_stats,
    language_expert_dispatch_triton_stats_delta,
)


def _inputs(
    *,
    device: str = "cpu",
    token_count: int = 12,
    state_dim: int = 16,
    expert_count: int = 8,
    active_count: int = 2,
    expert_hidden_dim: int = 24,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(20260704)
    dtype = torch.float32
    selected_expert_ids = torch.randint(
        0,
        expert_count,
        (token_count, active_count),
        device=device,
        dtype=torch.long,
    )
    combine_logits = torch.randn(token_count, active_count, device=device, dtype=dtype)
    return {
        "hidden": torch.randn(token_count, state_dim, device=device, dtype=dtype),
        "selected_expert_ids": selected_expert_ids,
        "combine_weights": torch.softmax(combine_logits, dim=-1),
        "first_weights": torch.randn(
            expert_count,
            expert_hidden_dim,
            state_dim,
            device=device,
            dtype=dtype,
        ),
        "first_biases": torch.randn(
            expert_count,
            expert_hidden_dim,
            device=device,
            dtype=dtype,
        ),
        "second_weights": torch.randn(
            expert_count,
            state_dim,
            expert_hidden_dim,
            device=device,
            dtype=dtype,
        ),
        "second_biases": torch.randn(
            expert_count,
            state_dim,
            device=device,
            dtype=dtype,
        ),
    }


def test_language_expert_dispatch_cpu_matches_reference() -> None:
    inputs = _inputs()

    output = language_expert_dispatch(**inputs)
    reference = language_expert_dispatch_torch_reference(**inputs)

    torch.testing.assert_close(output, reference)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not bool(language_expert_dispatch_triton_stats()["triton_available"]),
    reason="CUDA and Triton are required for expert dispatch kernel parity",
)
def test_language_expert_dispatch_triton_matches_reference() -> None:
    inputs = _inputs(
        device="cuda",
        token_count=128,
        state_dim=32,
        expert_count=16,
        active_count=4,
        expert_hidden_dim=64,
    )

    before = language_expert_dispatch_triton_stats()
    output = language_expert_dispatch(**inputs, force_triton=True)
    reference = language_expert_dispatch_torch_reference(**inputs)
    torch.cuda.synchronize()
    delta = language_expert_dispatch_triton_stats_delta(
        before,
        language_expert_dispatch_triton_stats(),
    )

    torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-4)
    assert delta["triton_kernel_used"] is True
    assert delta["triton_forward_calls"] >= 1
