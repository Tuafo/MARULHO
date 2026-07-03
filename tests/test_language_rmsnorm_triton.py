from __future__ import annotations

import pytest
import torch

from marulho.core.language_rmsnorm_triton import (
    language_rmsnorm,
    language_rmsnorm_torch_reference,
    language_rmsnorm_triton_stats,
    language_rmsnorm_triton_stats_delta,
)


def test_language_rmsnorm_cpu_matches_reference_and_trains() -> None:
    torch.manual_seed(20260703)
    value = torch.randn(3, 5, requires_grad=True)
    weight = torch.randn(5, requires_grad=True)

    output = language_rmsnorm(value, weight)
    reference = language_rmsnorm_torch_reference(value, weight)
    output.sum().backward()

    torch.testing.assert_close(output, reference)
    assert value.grad is not None
    assert weight.grad is not None
    assert torch.isfinite(value.grad).all()
    assert torch.isfinite(weight.grad).all()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not bool(language_rmsnorm_triton_stats()["triton_available"]),
    reason="CUDA and Triton are required for RMSNorm kernel parity",
)
def test_language_rmsnorm_triton_matches_reference_and_backward() -> None:
    torch.manual_seed(20260703)
    value = torch.randn(8, 16, device="cuda", requires_grad=True)
    weight = torch.randn(16, device="cuda", requires_grad=True)
    reference_value = value.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)

    before = language_rmsnorm_triton_stats()
    output = language_rmsnorm(value, weight, force_triton=True)
    reference = language_rmsnorm_torch_reference(reference_value, reference_weight)
    output.sum().backward()
    reference.sum().backward()
    torch.cuda.synchronize()
    delta = language_rmsnorm_triton_stats_delta(
        before,
        language_rmsnorm_triton_stats(),
    )

    torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(value.grad, reference_value.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(weight.grad, reference_weight.grad, rtol=1e-5, atol=1e-5)
    assert delta["triton_kernel_used"] is True
    assert delta["triton_forward_calls"] >= 1
