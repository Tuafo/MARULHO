import pytest
import torch

from marulho.training.language_delta_state import gated_delta_state_recurrent
from marulho.training.language_delta_state_triton import (
    delta_state_triton_available,
    gated_delta_state_recurrent_triton,
)


@pytest.mark.skipif(not delta_state_triton_available(), reason="requires CUDA Triton")
def test_direct_triton_forward_matches_recurrent_oracle() -> None:
    torch.manual_seed(641)
    shape = (2, 3, 11, 8)
    query = torch.nn.functional.normalize(
        torch.randn(shape, device="cuda", dtype=torch.float32), dim=-1
    )
    key = torch.nn.functional.normalize(
        torch.randn(shape, device="cuda", dtype=torch.float32), dim=-1
    )
    value = torch.randn(shape, device="cuda", dtype=torch.float32)
    erase = torch.sigmoid(torch.randn(shape, device="cuda", dtype=torch.float32))
    write = torch.sigmoid(torch.randn(shape, device="cuda", dtype=torch.float32))
    log_decay = -0.03 * torch.rand(shape, device="cuda", dtype=torch.float32)
    state = 0.05 * torch.randn((2, 3, 8, 8), device="cuda", dtype=torch.float32)

    expected_output, expected_state = gated_delta_state_recurrent(
        query, key, value, erase, write, log_decay, state
    )
    actual_output, actual_state = gated_delta_state_recurrent_triton(
        query, key, value, erase, write, log_decay, state
    )
    torch.testing.assert_close(actual_output, expected_output, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-5, rtol=2e-5)


def test_direct_triton_forward_rejects_training_until_backward_exists() -> None:
    value = torch.randn((1, 1, 2, 2), requires_grad=True)
    if delta_state_triton_available():
        cuda = value.detach().cuda().requires_grad_()
        with pytest.raises(RuntimeError, match="not admitted for autograd"):
            gated_delta_state_recurrent_triton(
                cuda, cuda, cuda, cuda, cuda, cuda
            )
