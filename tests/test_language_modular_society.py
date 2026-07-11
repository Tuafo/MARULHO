from __future__ import annotations

import pytest
import torch

from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel
from marulho.training.language_modular_society import (
    MarulhoModularSocietyLanguageModel,
    ModularSocietyConfig,
)


def _config(**overrides: object) -> ModularSocietyConfig:
    values: dict[str, object] = {
        "vocab_size": 96,
        "cell_count": 4,
        "cell_width": 32,
        "cell_layers": 1,
        "attention_heads": 4,
        "context_length": 16,
        "mlp_ratio": 2.0,
        "event_interval": 4,
        "message_dim": 8,
        "mode": "learned_real",
    }
    values.update(overrides)
    return ModularSocietyConfig(**values)


def test_full_society_matches_21m_parameter_budget() -> None:
    baseline = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=8192,
            embedding_dim=512,
            state_dim=512,
            state_layers=4,
            attention_heads=8,
            transformer_context_length=72,
            transformer_mlp_ratio=4.0,
        )
    )
    society = MarulhoModularSocietyLanguageModel(
        ModularSocietyConfig(vocab_size=8192)
    )
    baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
    society_count = sum(parameter.numel() for parameter in society.parameters())
    assert baseline_count == 20_976_128
    assert abs(society_count - baseline_count) / baseline_count < 0.002


def test_cells_have_independent_weights_and_states() -> None:
    model = MarulhoModularSocietyLanguageModel(_config())
    assert (
        model.cells[0].embedding.weight.data_ptr()
        != model.cells[1].embedding.weight.data_ptr()
    )
    token_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    result = model(token_ids)
    assert "cell_0_layer_0_key" in result["state"]
    assert "cell_1_layer_0_key" in result["state"]
    assert result["telemetry"]["communication_active"] is True


def test_modular_society_is_causal_and_scan_matches_steps() -> None:
    torch.manual_seed(5)
    model = MarulhoModularSocietyLanguageModel(_config()).eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    second = first.clone()
    second[:, 5:] = torch.tensor([20, 21, 22])
    first_logits = model(first, collect_telemetry=False)["logits"]
    second_logits = model(second, collect_telemetry=False)["logits"]
    torch.testing.assert_close(first_logits[:, :5], second_logits[:, :5])
    state = None
    rows = []
    for index in range(int(first.shape[1])):
        step = model.forward_step(first[:, index], state, collect_telemetry=False)
        state = step["state"]
        rows.append(step["logits"])
    torch.testing.assert_close(
        first_logits, torch.cat(rows, dim=1), atol=2e-6, rtol=2e-5
    )


def test_real_and_shuffled_messages_have_equal_compute_but_different_outputs() -> None:
    torch.manual_seed(7)
    real = MarulhoModularSocietyLanguageModel(_config(mode="learned_real")).eval()
    shuffled = MarulhoModularSocietyLanguageModel(
        _config(mode="learned_shuffled")
    ).eval()
    shuffled.load_state_dict(real.state_dict(), strict=True)
    token_ids = torch.randint(0, real.config.vocab_size, (3, 12))
    real_result = real(token_ids)
    shuffled_result = shuffled(token_ids)
    assert real_result["telemetry"]["message_event_batch_count"] == 9
    assert shuffled_result["telemetry"]["message_event_batch_count"] == 9
    assert not torch.allclose(real_result["logits"], shuffled_result["logits"])


def test_singleton_shuffled_control_cannot_leak_its_own_example() -> None:
    torch.manual_seed(9)
    shuffled = MarulhoModularSocietyLanguageModel(
        _config(mode="learned_shuffled")
    ).eval()
    no_message = MarulhoModularSocietyLanguageModel(
        _config(mode="learned_no_message")
    ).eval()
    no_message.load_state_dict(shuffled.state_dict(), strict=True)
    token_ids = torch.randint(0, shuffled.config.vocab_size, (1, 12))
    shuffled_result = shuffled(token_ids, collect_telemetry=False)
    no_message_result = no_message(token_ids, collect_telemetry=False)
    torch.testing.assert_close(shuffled_result["logits"], no_message_result["logits"])


@pytest.mark.parametrize(
    "mode",
    (
        "average_no_message",
        "learned_no_message",
        "learned_shuffled",
        "learned_real",
    ),
)
def test_all_controls_keep_the_same_parameter_graph(mode: str) -> None:
    torch.manual_seed(11)
    model = MarulhoModularSocietyLanguageModel(_config(mode=mode)).train()
    input_ids = torch.randint(0, model.config.vocab_size, (3, 12))
    targets = torch.randint(0, model.config.vocab_size, (3, 12))
    loss = model.next_token_loss(input_ids, targets)["loss"]
    loss.backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    assert missing == []
    assert all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


@pytest.mark.parametrize("mode", ("average_no_message", "learned_no_message"))
def test_no_message_controls_do_not_leak_message_parameters(mode: str) -> None:
    torch.manual_seed(13)
    model = MarulhoModularSocietyLanguageModel(_config(mode=mode)).eval()
    token_ids = torch.randint(0, model.config.vocab_size, (3, 12))
    before = model(token_ids, collect_telemetry=False)["logits"]
    with torch.no_grad():
        for projection in (*model.message_out, *model.message_in):
            projection.weight.normal_(mean=100.0, std=20.0)
    after = model(token_ids, collect_telemetry=False)["logits"]
    torch.testing.assert_close(before, after)
