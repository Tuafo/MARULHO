from __future__ import annotations

import torch

from marulho.training.language_dynamical_memory import (
    DYNAMICAL_MEMORY_MODES,
    DynamicalMemoryConfig,
    MarulhoDynamicalMemoryLanguageModel,
)
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


def _config(**overrides) -> DynamicalMemoryConfig:
    values = {
        "vocab_size": 96,
        "width": 32,
        "layers": 2,
        "attention_heads": 4,
        "hidden_width": 48,
        "context_length": 16,
        "memory_after_layer": 1,
        "memory_bank_count": 4,
        "memory_bank_width": 8,
        "memory_decays": (0.50, 0.80, 0.90, 0.98),
    }
    values.update(overrides)
    return DynamicalMemoryConfig(**values)


def test_dynamical_memory_candidate_matches_frozen_parameter_budget() -> None:
    baseline = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=8192,
            embedding_dim=512,
            state_dim=512,
            state_layers=4,
            attention_heads=8,
            transformer_context_length=72,
            transformer_mlp_ratio=4.0,
            tie_embeddings=True,
        )
    )
    candidate = MarulhoDynamicalMemoryLanguageModel(
        DynamicalMemoryConfig(vocab_size=8192)
    )
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())

    assert baseline_parameters == 20_976_128
    assert candidate_parameters == 20_977_152
    assert abs(candidate_parameters - baseline_parameters) / baseline_parameters < 0.001
    assert candidate.lm_head.weight.data_ptr() == candidate.token_embedding.weight.data_ptr()


def test_dynamical_memory_is_causal_and_streaming_equivalent() -> None:
    torch.manual_seed(7)
    model = MarulhoDynamicalMemoryLanguageModel(_config()).eval()
    first = torch.tensor([[1, 8, 9, 10, 11, 12]], dtype=torch.long)
    second = torch.tensor([[1, 8, 9, 44, 45, 46]], dtype=torch.long)

    first_logits = model(first, collect_telemetry=False)["logits"]
    second_logits = model(second, collect_telemetry=False)["logits"]
    assert torch.allclose(first_logits[:, :3], second_logits[:, :3], atol=1.0e-6)

    with torch.no_grad():
        full = model(first, collect_telemetry=False)["logits"]
        state = None
        steps = []
        for index in range(int(first.shape[1])):
            result = model.forward_step(
                first[:, index],
                state,
                collect_telemetry=False,
            )
            state = result["state"]
            steps.append(result["logits"])
        streamed = torch.cat(steps, dim=1)

    assert torch.allclose(streamed, full, atol=2.0e-5, rtol=1.0e-5)
    assert state is not None
    assert int(state["position"].item()) == int(first.shape[1])
    assert state["memory_state"].shape == (1, 32)


def test_dynamical_memory_modes_share_one_parameter_graph() -> None:
    torch.manual_seed(11)
    model = MarulhoDynamicalMemoryLanguageModel(_config()).eval()
    parameter_names = tuple(name for name, _parameter in model.named_parameters())
    pointers = tuple(parameter.data_ptr() for parameter in model.parameters())
    token_ids = torch.tensor([[1, 3, 5, 7, 9]], dtype=torch.long)
    outputs = {}

    with torch.no_grad():
        for mode in DYNAMICAL_MEMORY_MODES:
            model.set_memory_mode(mode)
            result = model(token_ids, collect_telemetry=True)
            outputs[mode] = result["logits"]
            assert result["telemetry"]["memory"]["mode"] == mode
            assert tuple(name for name, _parameter in model.named_parameters()) == (
                parameter_names
            )
            assert tuple(parameter.data_ptr() for parameter in model.parameters()) == (
                pointers
            )

    assert not torch.equal(outputs["memory_off"], outputs["multiscale_learned"])
    assert not torch.equal(outputs["single_scale"], outputs["multiscale_learned"])


def test_parallel_memory_matches_recurrent_updates_for_every_control() -> None:
    torch.manual_seed(12)
    model = MarulhoDynamicalMemoryLanguageModel(_config()).eval()
    memory = model.state_block.memory
    hidden = torch.randn(2, 7, 32)
    initial = torch.randn(2, 32) * 0.1

    with torch.no_grad():
        for mode in DYNAMICAL_MEMORY_MODES:
            model.set_memory_mode(mode)
            parallel, parallel_state, _telemetry = memory(
                hidden,
                initial,
                position_offset=torch.tensor(5),
                collect_telemetry=False,
            )
            recurrent_state = initial
            recurrent_outputs = []
            for index in range(int(hidden.shape[1])):
                output, recurrent_state, _telemetry = memory(
                    hidden[:, index : index + 1],
                    recurrent_state,
                    position_offset=torch.tensor(5 + index),
                    collect_telemetry=False,
                )
                recurrent_outputs.append(output)
            recurrent = torch.cat(recurrent_outputs, dim=1)

            assert torch.allclose(parallel, recurrent, atol=2.0e-5, rtol=1.0e-5)
            assert torch.allclose(
                parallel_state,
                recurrent_state,
                atol=2.0e-5,
                rtol=1.0e-5,
            )


def test_dynamical_memory_backpropagates_to_every_parameter() -> None:
    torch.manual_seed(13)
    model = MarulhoDynamicalMemoryLanguageModel(_config()).train()
    token_ids = torch.tensor(
        [[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]],
        dtype=torch.long,
    )

    loss = model.next_token_loss(token_ids, torch.roll(token_ids, -1, dims=1))["loss"]
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]

    assert torch.isfinite(loss)
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert model.state_block.memory.candidate.weight.grad is not None
    assert model.state_block.memory.write_gate.weight.grad is not None
    assert model.state_block.memory.output.weight.grad is not None
    assert model.state_block.memory.residual_alpha.grad is not None


def test_dynamical_memory_state_stays_finite_beyond_attention_context() -> None:
    torch.manual_seed(17)
    model = MarulhoDynamicalMemoryLanguageModel(_config()).eval()
    state = None
    with torch.no_grad():
        for index in range(64):
            result = model.forward_step(
                torch.tensor([index % 96], dtype=torch.long),
                state,
                collect_telemetry=False,
            )
            state = result["state"]

    assert state is not None
    assert int(state["position"].item()) == 64
    assert int(state["layer_0_key"].shape[2]) == 16
    assert torch.isfinite(state["memory_state"]).all()
    assert float(state["memory_state"].float().norm()) < 100.0


def test_dynamical_memory_uses_marulho_generation_protocol() -> None:
    torch.manual_seed(19)
    model = MarulhoDynamicalMemoryLanguageModel(_config()).eval()
    prompt = torch.tensor([1, 4, 7, 9], dtype=torch.long)

    first = model.generate(
        prompt,
        max_new_tokens=6,
        temperature=0.8,
        top_p=0.9,
        seed=23,
    )
    second = model.generate(
        prompt,
        max_new_tokens=6,
        temperature=0.8,
        top_p=0.9,
        seed=23,
    )

    assert torch.equal(first["generated_ids"], second["generated_ids"])
    assert first["external_llm_used"] is False
    assert first["generation_decode"]["decode_strategy"] == "nucleus_sampling"
