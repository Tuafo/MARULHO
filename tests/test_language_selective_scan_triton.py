from __future__ import annotations

import pytest
import torch

from marulho.core.language_selective_scan_triton import (
    language_selective_scan,
    language_selective_scan_torch_reference,
    language_selective_scan_triton_stats,
    language_selective_scan_triton_stats_delta,
)


def _inputs(
    *,
    device: str = "cpu",
    batch_size: int = 3,
    time_steps: int = 5,
    state_dim: int = 7,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(20260704)
    dtype = torch.float32
    return {
        "initial_state": torch.randn(batch_size, state_dim, device=device, dtype=dtype),
        "state_decay": torch.sigmoid(
            torch.randn(batch_size, time_steps, state_dim, device=device, dtype=dtype)
        ),
        "state_input": torch.sigmoid(
            torch.randn(batch_size, time_steps, state_dim, device=device, dtype=dtype)
        ),
        "spikes": (
            torch.rand(batch_size, time_steps, state_dim, device=device) > 0.75
        ).to(dtype),
    }


def test_language_selective_scan_cpu_matches_reference() -> None:
    inputs = _inputs()

    states, final_state = language_selective_scan(**inputs)
    reference_states, reference_final = language_selective_scan_torch_reference(**inputs)

    torch.testing.assert_close(states, reference_states)
    torch.testing.assert_close(final_state, reference_final)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not bool(language_selective_scan_triton_stats()["triton_available"]),
    reason="CUDA and Triton are required for selective scan kernel parity",
)
def test_language_selective_scan_triton_matches_reference() -> None:
    inputs = _inputs(device="cuda", batch_size=8, time_steps=16, state_dim=64)

    before = language_selective_scan_triton_stats()
    states, final_state = language_selective_scan(**inputs, force_triton=True)
    reference_states, reference_final = language_selective_scan_torch_reference(**inputs)
    torch.cuda.synchronize()
    delta = language_selective_scan_triton_stats_delta(
        before,
        language_selective_scan_triton_stats(),
    )

    torch.testing.assert_close(states, reference_states, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(final_state, reference_final, rtol=1e-5, atol=1e-5)
    assert delta["triton_kernel_used"] is True
    assert delta["triton_forward_calls"] >= 1
