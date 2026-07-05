from __future__ import annotations

import pytest
import torch

from marulho.core.language_eligibility_trace_triton import (
    language_eligibility_trace_final,
    language_eligibility_trace_torch_reference,
    language_eligibility_trace_triton_stats,
    language_eligibility_trace_triton_stats_delta,
)


def _inputs(
    *,
    device: str = "cpu",
    batch_size: int = 3,
    time_steps: int = 5,
    state_dim: int = 7,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(20260705)
    dtype = torch.float32
    return {
        "initial_trace": torch.randn(batch_size, state_dim, device=device, dtype=dtype).abs(),
        "spikes": (
            torch.rand(batch_size, time_steps, state_dim, device=device) > 0.75
        ).to(dtype),
    }


def test_language_eligibility_trace_cpu_matches_reference() -> None:
    inputs = _inputs()

    final_trace = language_eligibility_trace_final(**inputs)
    reference = language_eligibility_trace_torch_reference(**inputs)

    torch.testing.assert_close(final_trace, reference)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not bool(language_eligibility_trace_triton_stats()["triton_available"]),
    reason="CUDA and Triton are required for eligibility trace kernel parity",
)
def test_language_eligibility_trace_triton_matches_reference() -> None:
    inputs = _inputs(device="cuda", batch_size=8, time_steps=16, state_dim=64)

    before = language_eligibility_trace_triton_stats()
    final_trace = language_eligibility_trace_final(**inputs, force_triton=True)
    reference = language_eligibility_trace_torch_reference(**inputs)
    torch.cuda.synchronize()
    delta = language_eligibility_trace_triton_stats_delta(
        before,
        language_eligibility_trace_triton_stats(),
    )

    torch.testing.assert_close(final_trace, reference, rtol=1e-5, atol=1e-5)
    assert delta["triton_kernel_used"] is True
    assert delta["triton_forward_calls"] >= 1
