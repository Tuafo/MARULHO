import torch

from marulho.training.language_delta_state import (
    DeltaStateLanguageModelConfig,
    MarulhoDeltaStateLanguageModel,
    delta_state_parameter_report,
    gated_delta_state_chunkwise,
    gated_delta_state_recurrent,
    set_delta_state_execution_backend,
)


def _inputs(dtype: torch.dtype = torch.float64):
    torch.manual_seed(91)
    shape = (2, 3, 11, 4)
    query = torch.nn.functional.normalize(torch.randn(shape, dtype=dtype), dim=-1)
    key = torch.nn.functional.normalize(torch.randn(shape, dtype=dtype), dim=-1)
    value = torch.randn(shape, dtype=dtype)
    erase = torch.sigmoid(torch.randn(shape, dtype=dtype))
    write = torch.sigmoid(torch.randn(shape, dtype=dtype))
    log_decay = -0.03 * torch.rand(shape, dtype=dtype)
    state = 0.05 * torch.randn((2, 3, 4, 4), dtype=dtype)
    return [
        tensor.requires_grad_()
        for tensor in (query, key, value, erase, write, log_decay, state)
    ]


def test_chunkwise_matches_recurrent_forward_state_and_gradients() -> None:
    recurrent_inputs = _inputs()
    chunk_inputs = [
        tensor.detach().clone().requires_grad_() for tensor in recurrent_inputs
    ]
    recurrent_output, recurrent_state = gated_delta_state_recurrent(
        *recurrent_inputs[:6], recurrent_inputs[6]
    )
    chunk_output, chunk_state = gated_delta_state_chunkwise(
        *chunk_inputs[:6], chunk_inputs[6], chunk_size=4
    )
    torch.testing.assert_close(chunk_output, recurrent_output, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(chunk_state, recurrent_state, atol=2e-6, rtol=2e-6)
    recurrent_loss = recurrent_output.square().mean() + recurrent_state.square().mean()
    chunk_loss = chunk_output.square().mean() + chunk_state.square().mean()
    recurrent_gradients = torch.autograd.grad(recurrent_loss, recurrent_inputs)
    chunk_gradients = torch.autograd.grad(chunk_loss, chunk_inputs)
    for actual, expected in zip(chunk_gradients, recurrent_gradients):
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)


def test_chunk_composition_matches_single_call() -> None:
    values = _inputs(torch.float32)
    whole_output, whole_state = gated_delta_state_chunkwise(
        *values[:6], values[6], chunk_size=4
    )
    first_output, first_state = gated_delta_state_chunkwise(
        *(value[:, :, :5] for value in values[:6]), values[6], chunk_size=4
    )
    second_output, second_state = gated_delta_state_chunkwise(
        *(value[:, :, 5:] for value in values[:6]), first_state, chunk_size=4
    )
    torch.testing.assert_close(
        torch.cat((first_output, second_output), dim=2),
        whole_output,
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(second_state, whole_state, atol=2e-5, rtol=2e-5)


def test_v64_shape_parameter_match_streaming_and_gradients() -> None:
    tiny = DeltaStateLanguageModelConfig(
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
    torch.manual_seed(92)
    model = MarulhoDeltaStateLanguageModel(tiny)
    ids = torch.randint(0, tiny.vocab_size, (2, 13))
    full = model(ids, collect_telemetry=False)
    state = None
    pieces = []
    for index in range(ids.shape[1]):
        step = model.forward_step(ids[:, index], state, collect_telemetry=False)
        pieces.append(step["logits"])
        state = step["state"]
    torch.testing.assert_close(
        torch.cat(pieces, dim=1), full["logits"], atol=2e-5, rtol=2e-5
    )
    loss = model.next_token_loss(ids[:, :-1], ids[:, 1:])["loss"]
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(
        bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()
    )

    full_config = DeltaStateLanguageModelConfig()
    full_model = MarulhoDeltaStateLanguageModel(full_config)
    report = delta_state_parameter_report(full_model)
    assert report["total_parameters"] == 100_202_970
    assert 0.99 <= report["total_parameters"] / 100_679_424 <= 1.01
    assert report["external_llm_used"] is False
    set_delta_state_execution_backend(model, "eager")
    assert all(
        module.execution_backend == "eager"
        for module in model.modules()
        if module.__class__.__name__ == "MarulhoDeltaStateMixer"
    )
