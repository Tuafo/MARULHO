import copy

import pytest
import torch

from marulho.training.language_delta_state import gated_delta_state_recurrent
from marulho.training.language_delta_state import (
    DeltaStateLanguageModelConfig,
    MarulhoDeltaStateLanguageModel,
)
from marulho.training.language_delta_state_triton import (
    delta_state_triton_available,
    gated_delta_state_recurrent_triton,
    gated_delta_state_recurrent_triton_autograd,
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


@pytest.mark.skipif(not delta_state_triton_available(), reason="requires CUDA Triton")
def test_direct_triton_backward_matches_recurrent_oracle() -> None:
    torch.manual_seed(642)
    shape = (2, 3, 11, 8)
    base = [
        torch.nn.functional.normalize(
            torch.randn(shape, device="cuda", dtype=torch.float32), dim=-1
        ),
        torch.nn.functional.normalize(
            torch.randn(shape, device="cuda", dtype=torch.float32), dim=-1
        ),
        torch.randn(shape, device="cuda", dtype=torch.float32),
        torch.sigmoid(torch.randn(shape, device="cuda", dtype=torch.float32)),
        torch.sigmoid(torch.randn(shape, device="cuda", dtype=torch.float32)),
        -0.03 * torch.rand(shape, device="cuda", dtype=torch.float32),
        0.05 * torch.randn((2, 3, 8, 8), device="cuda", dtype=torch.float32),
    ]
    expected_inputs = [value.detach().clone().requires_grad_() for value in base]
    actual_inputs = [value.detach().clone().requires_grad_() for value in base]
    expected_output, expected_state = gated_delta_state_recurrent(
        *expected_inputs[:6], expected_inputs[6]
    )
    actual_output, actual_state = gated_delta_state_recurrent_triton_autograd(
        *actual_inputs[:6], actual_inputs[6]
    )
    output_weight = torch.randn_like(expected_output)
    state_weight = torch.randn_like(expected_state)
    expected_loss = (expected_output * output_weight).sum() + (
        expected_state * state_weight
    ).sum()
    actual_loss = (actual_output * output_weight).sum() + (
        actual_state * state_weight
    ).sum()
    expected_gradients = torch.autograd.grad(expected_loss, expected_inputs)
    actual_gradients = torch.autograd.grad(actual_loss, actual_inputs)
    torch.testing.assert_close(actual_output, expected_output, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-5, rtol=2e-5)
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)


@pytest.mark.skipif(not delta_state_triton_available(), reason="requires CUDA Triton")
def test_direct_triton_backward_is_stable_at_context_320() -> None:
    torch.manual_seed(643)
    shape = (1, 1, 320, 8)
    base = [
        torch.nn.functional.normalize(
            torch.randn(shape, device="cuda", dtype=torch.float32), dim=-1
        ),
        torch.nn.functional.normalize(
            torch.randn(shape, device="cuda", dtype=torch.float32), dim=-1
        ),
        torch.randn(shape, device="cuda", dtype=torch.float32),
        torch.sigmoid(torch.randn(shape, device="cuda", dtype=torch.float32)),
        torch.sigmoid(torch.randn(shape, device="cuda", dtype=torch.float32)),
        -torch.nn.functional.softplus(
            torch.randn(shape, device="cuda", dtype=torch.float32)
        ),
        torch.zeros((1, 1, 8, 8), device="cuda", dtype=torch.float32),
    ]
    expected_inputs = [value.detach().clone().requires_grad_() for value in base]
    actual_inputs = [value.detach().clone().requires_grad_() for value in base]
    expected_output, _ = gated_delta_state_recurrent(
        *expected_inputs[:6], expected_inputs[6]
    )
    actual_output, _ = gated_delta_state_recurrent_triton_autograd(
        *actual_inputs[:6], actual_inputs[6]
    )
    output_weight = torch.randn_like(expected_output)
    expected_gradients = torch.autograd.grad(
        (expected_output * output_weight).sum(), expected_inputs
    )
    actual_gradients = torch.autograd.grad(
        (actual_output * output_weight).sum(), actual_inputs
    )
    for actual, expected in zip(actual_gradients, expected_gradients):
        assert bool(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not delta_state_triton_available(), reason="requires CUDA Triton")
def test_full_tiny_model_triton_matches_eager_loss_state_and_gradients() -> None:
    common = dict(
        vocab_size=97,
        embedding_dim=32,
        state_dim=32,
        state_layers=4,
        attention_heads=4,
        transformer_context_length=24,
        transformer_mlp_ratio=2.0,
        local_attention_window=8,
        delta_chunk_size=4,
    )
    torch.manual_seed(644)
    eager = MarulhoDeltaStateLanguageModel(
        DeltaStateLanguageModelConfig(**common, delta_execution_backend="eager")
    ).cuda()
    direct = MarulhoDeltaStateLanguageModel(
        DeltaStateLanguageModelConfig(
            **common, delta_execution_backend="triton_replay"
        )
    ).cuda()
    direct.load_state_dict(eager.state_dict(), strict=True)
    ids = torch.randint(0, 97, (2, 13), device="cuda")
    eager_result = eager(ids, collect_telemetry=False)
    direct_result = direct(ids, collect_telemetry=False)
    torch.testing.assert_close(
        direct_result["logits"], eager_result["logits"], atol=2e-5, rtol=2e-5
    )
    for name, state in eager_result["state"].items():
        torch.testing.assert_close(
            direct_result["state"][name], state, atol=2e-5, rtol=2e-5
        )
    eager_loss = eager_result["logits"].float().square().mean()
    direct_loss = direct_result["logits"].float().square().mean()
    eager_loss.backward()
    direct_loss.backward()
    dot = 0.0
    eager_norm = 0.0
    direct_norm = 0.0
    maximum_delta = 0.0
    for (direct_name, direct_parameter), (eager_name, eager_parameter) in zip(
        direct.named_parameters(), eager.named_parameters()
    ):
        assert direct_name == eager_name
        assert direct_parameter.grad is not None
        assert eager_parameter.grad is not None
        assert bool(torch.isfinite(direct_parameter.grad).all())
        direct_gradient = direct_parameter.grad.float()
        eager_gradient = eager_parameter.grad.float()
        dot += float(torch.sum(direct_gradient * eager_gradient).item())
        direct_norm += float(torch.sum(direct_gradient.square()).item())
        eager_norm += float(torch.sum(eager_gradient.square()).item())
        maximum_delta = max(
            maximum_delta,
            float(torch.max(torch.abs(direct_gradient - eager_gradient)).item()),
        )
    cosine = dot / (direct_norm * eager_norm) ** 0.5
    assert cosine >= 0.999
    assert maximum_delta <= 0.01


@pytest.mark.skipif(not delta_state_triton_available(), reason="requires CUDA Triton")
def test_tiny_model_optimizer_step_replays_in_cuda_graph() -> None:
    config = DeltaStateLanguageModelConfig(
        vocab_size=97,
        embedding_dim=32,
        state_dim=32,
        state_layers=4,
        attention_heads=4,
        transformer_context_length=24,
        transformer_mlp_ratio=2.0,
        local_attention_window=8,
        delta_chunk_size=4,
        delta_execution_backend="triton_replay",
    )
    torch.manual_seed(645)
    model = MarulhoDeltaStateLanguageModel(config).cuda().train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=4.0e-4, weight_decay=0.1, fused=True, capturable=True
    )
    first_ids = torch.randint(0, config.vocab_size, (2, 13), device="cuda")
    second_ids = torch.randint(0, config.vocab_size, (2, 13), device="cuda")

    def loss(model: MarulhoDeltaStateLanguageModel, ids: torch.Tensor) -> torch.Tensor:
        return model(ids, collect_telemetry=False)["logits"].float().square().mean()

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        optimizer.zero_grad(set_to_none=True)
        warmup_loss = loss(model, first_ids)
        warmup_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()
    del warmup_loss

    optimizer.zero_grad(set_to_none=True)
    before = model.token_embedding.weight.detach().clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        first_loss = loss(model, first_ids) * 0.5
        first_loss.backward()
        second_loss = loss(model, second_ids) * 0.5
        second_loss.backward()
        captured_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    eager = MarulhoDeltaStateLanguageModel(config).cuda().train()
    eager.load_state_dict(model.state_dict(), strict=True)
    eager_optimizer = torch.optim.AdamW(
        eager.parameters(), lr=4.0e-4, weight_decay=0.1, fused=True, capturable=True
    )
    eager_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    eager_optimizer.zero_grad(set_to_none=True)
    (loss(eager, first_ids) * 0.5).backward()
    (loss(eager, second_ids) * 0.5).backward()
    eager_norm = torch.nn.utils.clip_grad_norm_(eager.parameters(), 1.0)
    eager_optimizer.step()

    graph.replay()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(captured_norm).all())
    torch.testing.assert_close(captured_norm, eager_norm, atol=1e-6, rtol=1e-6)
    assert not torch.equal(model.token_embedding.weight, before)
    for (actual_name, actual), (expected_name, expected) in zip(
        model.named_parameters(), eager.named_parameters()
    ):
        assert actual_name == expected_name
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
