from __future__ import annotations

import torch

from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel
from marulho.training.language_sparse_event_memory import (
    MarulhoSparseEventLanguageModel,
    SparseEventMemoryConfig,
)


def _config(**overrides: object) -> SparseEventMemoryConfig:
    values: dict[str, object] = {
        "vocab_size": 96,
        "embedding_dim": 32,
        "state_dim": 32,
        "state_layers": 2,
        "attention_heads": 4,
        "transformer_context_length": 16,
        "transformer_mlp_ratio": 2.0,
        "transformer_dropout": 0.0,
        "event_interval": 4,
        "specialist_count": 4,
        "specialist_rank": 8,
        "exploration_rate": 0.0,
        "counterfactual_rate": 0.0,
    }
    values.update(overrides)
    return SparseEventMemoryConfig(**values)


def _baseline_config(config: SparseEventMemoryConfig) -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        state_dim=config.state_dim,
        state_layers=config.state_layers,
        attention_heads=config.attention_heads,
        transformer_context_length=config.transformer_context_length,
        transformer_mlp_ratio=config.transformer_mlp_ratio,
        transformer_dropout=config.transformer_dropout,
        tie_embeddings=config.tie_embeddings,
    )


def test_v2_preserves_the_complete_exact_stream_and_starts_neutral() -> None:
    torch.manual_seed(3)
    config = _config(initial_residual_scale=0.0)
    candidate = MarulhoSparseEventLanguageModel(config).eval()
    baseline = MarulhoLanguageModel(_baseline_config(config)).eval()
    baseline.load_state_dict(
        {
            name: value
            for name, value in candidate.state_dict().items()
            if not name.startswith("event_memory.")
        },
        strict=True,
    )
    token_ids = torch.randint(0, config.vocab_size, (2, 12))
    candidate_logits = candidate(token_ids, collect_telemetry=False)["logits"]
    baseline_logits = baseline(token_ids, collect_telemetry=False)["logits"]
    torch.testing.assert_close(candidate_logits, baseline_logits)
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    exact_parameters = sum(
        parameter.numel()
        for name, parameter in candidate.named_parameters()
        if not name.startswith("event_memory.")
    )
    assert exact_parameters == baseline_parameters


def test_v2_scan_matches_incremental_generation_state() -> None:
    torch.manual_seed(5)
    model = MarulhoSparseEventLanguageModel(_config()).eval()
    token_ids = torch.randint(0, model.config.vocab_size, (2, 12))
    scanned = model(token_ids, collect_telemetry=False)
    state = None
    logits = []
    for index in range(int(token_ids.shape[1])):
        step = model.forward_step(
            token_ids[:, index], state, collect_telemetry=False
        )
        state = step["state"]
        logits.append(step["logits"])
    torch.testing.assert_close(
        scanned["logits"], torch.cat(logits, dim=1), atol=2.0e-6, rtol=2.0e-5
    )
    assert state is not None
    assert int(state["position"].item()) == 12
    torch.testing.assert_close(
        scanned["state"]["event_current_residual"],
        state["event_current_residual"],
    )


def test_v2_executes_one_of_four_specialists_per_event() -> None:
    torch.manual_seed(7)
    model = MarulhoSparseEventLanguageModel(_config()).eval()
    token_ids = torch.randint(0, model.config.vocab_size, (3, 12))
    result = model(token_ids)
    telemetry = result["telemetry"]
    assert telemetry["completed_event_batch_count"] == 9
    assert telemetry["executed_specialist_count"] == 9
    assert telemetry["possible_specialist_count"] == 36
    assert telemetry["active_compute_fraction"] == 0.25
    assert telemetry["actual_sparse_execution"] is True


def test_v2_counterfactual_credit_reaches_router_and_specialists() -> None:
    torch.manual_seed(11)
    model = MarulhoSparseEventLanguageModel(
        _config(counterfactual_rate=1.0, initial_residual_scale=0.01)
    ).train()
    input_ids = torch.randint(0, model.config.vocab_size, (4, 12))
    target_ids = torch.randint(0, model.config.vocab_size, (4, 12))
    result = model.next_token_loss(
        input_ids,
        target_ids,
        collect_telemetry=False,
        counterfactual_probe=True,
    )
    assert result["training_aux"]["counterfactual"]["ran"] is True
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert model.event_memory.router.weight.grad is not None
    assert torch.isfinite(model.event_memory.router.weight.grad).all()
    assert model.event_memory.down.grad is not None
    assert model.event_memory.up.grad is not None
    assert model.event_memory.residual_scale.grad is not None


def test_dense_control_executes_every_specialist() -> None:
    torch.manual_seed(13)
    model = MarulhoSparseEventLanguageModel(
        _config(selection_mode="dense")
    ).eval()
    token_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    telemetry = model(token_ids)["telemetry"]
    assert telemetry["completed_event_batch_count"] == 4
    assert telemetry["executed_specialist_count"] == 16
    assert telemetry["active_compute_fraction"] == 1.0
    assert telemetry["actual_sparse_execution"] is False
