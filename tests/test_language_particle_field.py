from __future__ import annotations

import pytest
import torch

from marulho.training.language_particle_field import (
    MarulhoParticleFieldLanguageModel,
    ParticleFieldConfig,
)


def _model(**overrides) -> MarulhoParticleFieldLanguageModel:
    values = {
        "vocab_size": 96,
        "width": 16,
        "particle_count": 64,
        "recurrences": 3,
        "heads": 4,
        "context_length": 24,
        "dropout": 0.0,
        "materialized_state_batch_limit": 4,
    }
    values.update(overrides)
    return MarulhoParticleFieldLanguageModel(ParticleFieldConfig(**values))


def test_particle_field_is_causal_and_recurrently_exact() -> None:
    torch.manual_seed(3)
    model = _model().eval()
    first = torch.tensor([[1, 8, 9, 10, 11, 12, 13, 14]])
    second = torch.tensor([[1, 8, 9, 44, 45, 46, 47, 48]])
    with torch.no_grad():
        parallel = model(first, collect_telemetry=True)
        changed = model(second, collect_telemetry=False)
        recurrent = model.recurrent_scan(first)
    torch.testing.assert_close(parallel["logits"][:, :3], changed["logits"][:, :3])
    torch.testing.assert_close(
        recurrent["logits"],
        parallel["logits"],
        atol=2.0e-5,
        rtol=1.0e-5,
    )
    for key, value in parallel["state"].items():
        torch.testing.assert_close(recurrent["state"][key], value)
    assert parallel["telemetry"]["positive_activation_by_construction"] is True
    assert 0.0 <= parallel["telemetry"]["mean_zero_activation_fraction"] <= 1.0


def test_particle_field_loss_reaches_every_parameter() -> None:
    torch.manual_seed(5)
    model = _model().train()
    input_ids = torch.randint(0, 96, (3, 12))
    target_ids = torch.randint(0, 96, (3, 12))
    loss = model.next_token_loss(
        input_ids,
        target_ids,
        collect_telemetry=False,
    )["loss"]
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(
        int(torch.count_nonzero(parameter.grad)) > 0
        for parameter in model.parameters()
    )


def test_particle_field_default_is_nearly_exact_21m_match() -> None:
    model = MarulhoParticleFieldLanguageModel(
        ParticleFieldConfig(vocab_size=8192)
    )
    report = model.parameter_report()
    assert report["total_parameters"] == 20_971_520
    assert report["parameter_accounting_exact"] is True
    assert 20_976_128 - report["total_parameters"] == 4_608
    assert report["positive_particles"] == 24_576


def test_particle_field_owned_generation_uses_streaming_fast_weights() -> None:
    torch.manual_seed(7)
    model = _model().eval()
    row = model.generate(
        torch.tensor([1, 3, 5, 7]),
        max_new_tokens=4,
        eos_id=None,
    )
    assert row["generated_ids"].shape == (1, 8)
    assert row["external_llm_used"] is False
    assert row["owned_by_marulho"] is True
    assert row["surface"] == "marulho_particle_field_generation.v1"
    assert row["generation_decode"]["state_cache"] == (
        "per_recurrence_hebbian_fast_weight"
    )
    assert len(row["state"]) == 1 + model.particle_config.recurrences


def test_large_batch_omits_unneeded_streaming_state() -> None:
    model = _model(materialized_state_batch_limit=2).eval()
    row = model(torch.randint(0, 96, (3, 8)), collect_telemetry=True)
    assert set(row["state"]) == {"position"}
    assert row["telemetry"]["streaming_state_materialized"] is False


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"width": 15}, "width"),
        ({"particle_count": 68}, "per head"),
        ({"particle_count": 63}, "divisible"),
        ({"recurrences": 0}, "recurrences"),
        ({"dropout": 1.0}, "dropout"),
        ({"rope_theta": 1.0}, "rope_theta"),
    ],
)
def test_particle_field_rejects_invalid_configuration(overrides, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _model(**overrides)
