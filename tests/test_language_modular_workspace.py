from __future__ import annotations

import pytest
import torch

from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel
from marulho.training.language_modular_workspace import (
    MarulhoModularWorkspaceLanguageModel,
    ModularWorkspaceConfig,
)


def _config(**overrides: object) -> ModularWorkspaceConfig:
    values: dict[str, object] = {
        "vocab_size": 96,
        "shared_width": 32,
        "shared_layers_per_stage": 1,
        "shared_attention_heads": 4,
        "cell_count": 3,
        "cell_width": 24,
        "cell_layers_per_stage": 1,
        "cell_attention_heads": 4,
        "workspace_width": 8,
        "context_length": 16,
        "mlp_ratio": 2.0,
        "mode": "real",
    }
    values.update(overrides)
    return ModularWorkspaceConfig(**values)


def test_full_workspace_matches_21m_parameter_budget() -> None:
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
    workspace = MarulhoModularWorkspaceLanguageModel(
        ModularWorkspaceConfig(vocab_size=8192)
    )
    baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
    workspace_count = sum(parameter.numel() for parameter in workspace.parameters())
    assert baseline_count == 20_976_128
    assert workspace_count == 21_012_624
    assert abs(workspace_count - baseline_count) / baseline_count < 0.002
    assert workspace.lm_head.weight.data_ptr() == workspace.token_embedding.weight.data_ptr()


def test_workspace_is_causal_and_scan_matches_steps() -> None:
    torch.manual_seed(3)
    model = MarulhoModularWorkspaceLanguageModel(_config()).eval()
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
        first_logits,
        torch.cat(rows, dim=1),
        atol=3.0e-6,
        rtol=3.0e-5,
    )


def test_real_and_shuffled_workspace_have_different_outputs() -> None:
    torch.manual_seed(5)
    real = MarulhoModularWorkspaceLanguageModel(_config(mode="real")).eval()
    shuffled = MarulhoModularWorkspaceLanguageModel(
        _config(mode="shuffled")
    ).eval()
    shuffled.load_state_dict(real.state_dict(), strict=True)
    token_ids = torch.randint(0, real.config.vocab_size, (3, 12))
    real_result = real(token_ids)
    shuffled_result = shuffled(token_ids)
    assert real_result["telemetry"]["workspace_exchange_active"] is True
    assert shuffled_result["telemetry"]["workspace_exchange_active"] is True
    assert not torch.allclose(real_result["logits"], shuffled_result["logits"])


def test_singleton_shuffled_workspace_becomes_no_exchange() -> None:
    torch.manual_seed(7)
    shuffled = MarulhoModularWorkspaceLanguageModel(
        _config(mode="shuffled")
    ).eval()
    no_exchange = MarulhoModularWorkspaceLanguageModel(
        _config(mode="no_exchange")
    ).eval()
    no_exchange.load_state_dict(shuffled.state_dict(), strict=True)
    token_ids = torch.randint(0, shuffled.config.vocab_size, (1, 12))
    torch.testing.assert_close(
        shuffled(token_ids, collect_telemetry=False)["logits"],
        no_exchange(token_ids, collect_telemetry=False)["logits"],
    )


def test_no_exchange_cannot_leak_workspace_projection_weights() -> None:
    torch.manual_seed(9)
    model = MarulhoModularWorkspaceLanguageModel(
        _config(mode="no_exchange")
    ).eval()
    token_ids = torch.randint(0, model.config.vocab_size, (3, 12))
    before = model(token_ids, collect_telemetry=False)["logits"]
    with torch.no_grad():
        for cell in model.cells:
            cell.message_out.weight.normal_(mean=100.0, std=20.0)
            cell.message_in.weight.normal_(mean=100.0, std=20.0)
            cell.write_score.weight.normal_(mean=100.0, std=20.0)
        for parameter in model.workspace_core.parameters():
            parameter.normal_(mean=100.0, std=20.0)
    after = model(token_ids, collect_telemetry=False)["logits"]
    torch.testing.assert_close(before, after)


@pytest.mark.parametrize("mode", ("no_exchange", "shuffled", "real"))
def test_workspace_controls_keep_the_same_parameter_graph(mode: str) -> None:
    torch.manual_seed(11)
    model = MarulhoModularWorkspaceLanguageModel(_config(mode=mode)).train()
    input_ids = torch.randint(0, model.config.vocab_size, (3, 12))
    targets = torch.randint(0, model.config.vocab_size, (3, 12))
    loss = model.next_token_loss(input_ids, targets)["loss"]
    loss.backward()
    assert [name for name, parameter in model.named_parameters() if parameter.grad is None] == []
    assert all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_workspace_preserves_gradient_across_full_context() -> None:
    torch.manual_seed(13)
    model = MarulhoModularWorkspaceLanguageModel(_config(mode="real")).train()
    captured: dict[str, torch.Tensor] = {}

    def capture_embedding(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        output.retain_grad()
        captured["embedding"] = output

    handle = model.token_embedding.register_forward_hook(capture_embedding)
    token_ids = torch.randint(0, model.config.vocab_size, (2, 16))
    output = model(token_ids, collect_telemetry=False)
    output["logits"][:, -1, 0].sum().backward()
    handle.remove()
    gradient = captured["embedding"].grad
    assert gradient is not None
    assert float(gradient[:, 0].abs().sum()) > 0.0


def test_workspace_uses_owned_generation_protocol() -> None:
    torch.manual_seed(17)
    model = MarulhoModularWorkspaceLanguageModel(_config()).eval()
    generated = model.generate(
        torch.tensor([1, 2, 3]),
        max_new_tokens=2,
        temperature=0.0,
    )
    assert generated["generated_ids"].shape == (1, 5)
    assert generated["owned_by_marulho"] is True
    assert generated["external_llm_used"] is False
    assert generated["generation_decode"]["kv_cache"] == (
        "bounded_shared_cell_and_workspace_layers"
    )
