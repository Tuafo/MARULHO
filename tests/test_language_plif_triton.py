from __future__ import annotations

import pytest
import torch

from marulho.core.language_plif_triton import (
    language_plif_forward,
    language_plif_torch_reference,
    language_plif_triton_stats,
    language_plif_triton_stats_delta,
)


def _inputs(*, device: str = "cpu", rows: int = 3, cols: int = 5) -> dict[str, torch.Tensor]:
    torch.manual_seed(20260703)
    dtype = torch.float32
    membrane = torch.randn(rows, cols, device=device, dtype=dtype)
    spikes = (torch.rand(rows, cols, device=device) > 0.75).to(dtype)
    selective_state = torch.randn(rows, cols, device=device, dtype=dtype)
    eligibility_trace = torch.randn(rows, cols, device=device, dtype=dtype).abs()
    leak = torch.sigmoid(torch.randn(rows, cols, device=device, dtype=dtype))
    threshold = torch.nn.functional.softplus(
        torch.randn(rows, cols, device=device, dtype=dtype)
    ) + 0.05
    drive = torch.randn(rows, cols, device=device, dtype=dtype)
    state_decay = torch.sigmoid(torch.randn(rows, cols, device=device, dtype=dtype))
    state_input = torch.sigmoid(torch.randn(rows, cols, device=device, dtype=dtype))
    state_output = torch.sigmoid(torch.randn(rows, cols, device=device, dtype=dtype))
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


def test_language_plif_cpu_matches_reference() -> None:
    inputs = _inputs()

    output = language_plif_forward(**inputs)
    reference = language_plif_torch_reference(**inputs)

    for item, expected in zip(output, reference, strict=True):
        torch.testing.assert_close(item, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not bool(language_plif_triton_stats()["triton_available"]),
    reason="CUDA and Triton are required for PLIF forward kernel parity",
)
def test_language_plif_triton_matches_reference() -> None:
    inputs = _inputs(device="cuda", rows=16, cols=32)

    before = language_plif_triton_stats()
    output = language_plif_forward(**inputs, force_triton=True)
    reference = language_plif_torch_reference(**inputs)
    torch.cuda.synchronize()
    delta = language_plif_triton_stats_delta(
        before,
        language_plif_triton_stats(),
    )

    for item, expected in zip(output, reference, strict=True):
        torch.testing.assert_close(item, expected, rtol=1e-5, atol=1e-5)
    assert delta["triton_kernel_used"] is True
    assert delta["triton_forward_calls"] >= 1
