from __future__ import annotations

import torch

import marulho.evaluation.language_ngpt_falsification as experiment
from marulho.training.language_model import LanguageBatch
from marulho.training.language_ngpt import MarulhoNormalizedLanguageModel


def test_v6_schedule_is_exact_and_reproducible() -> None:
    first = experiment.build_matched_schedule(
        step_count=20,
        relation_fraction=0.2,
        relation_batch_count=5,
        general_batch_counts=(9, 9),
        seed=17,
    )

    assert first == experiment.build_matched_schedule(
        step_count=20,
        relation_fraction=0.2,
        relation_batch_count=5,
        general_batch_counts=(9, 9),
        seed=17,
    )
    assert sum(kind == "relation" for kind, _index in first) == 4
    assert sum(kind == "general_0" for kind, _index in first) == 8
    assert sum(kind == "general_1" for kind, _index in first) == 8


def test_v6_stages_the_exact_schedule_once() -> None:
    relation = (
        LanguageBatch(
            input_ids=torch.tensor([[1, 2]]),
            target_ids=torch.tensor([[2, 3]]),
        ),
    )
    general = (
        (
            LanguageBatch(
                input_ids=torch.tensor([[4, 5]]),
                target_ids=torch.tensor([[5, 6]]),
            ),
        ),
        (
            LanguageBatch(
                input_ids=torch.tensor([[7, 8]]),
                target_ids=torch.tensor([[8, 9]]),
            ),
        ),
    )
    schedule = (("general_1", 0), ("relation", 0), ("general_0", 0))

    staged = experiment._stage_schedule(
        schedule,
        relation_batches=relation,
        general_batches=general,
        device=torch.device("cpu"),
    )

    assert staged.input_ids.tolist() == [[[7, 8]], [[1, 2]], [[4, 5]]]
    assert staged.target_ids.tolist() == [[[8, 9]], [[2, 3]], [[5, 6]]]
    assert staged.storage_bytes == 3 * 1 * 2 * 8 * 2


def test_v6_default_models_are_within_one_tenth_percent() -> None:
    config = experiment.HypersphericalTransformerFalsificationConfig()
    baseline = experiment._build_model(
        "transformer",
        vocab_size=8192,
        config=config,
    )
    normalized = experiment._build_model(
        "normalized",
        vocab_size=8192,
        config=config,
    )
    baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
    normalized_count = sum(
        parameter.numel() for parameter in normalized.parameters()
    )

    assert baseline_count == 20_976_128
    assert normalized_count == 20_988_288
    assert abs(normalized_count - baseline_count) / baseline_count < 0.001


def test_v6_recipes_separate_architecture_from_optimizer() -> None:
    config = experiment.HypersphericalTransformerFalsificationConfig()
    standard = experiment._training_config("standard", config)
    native = experiment._training_config("native", config)

    assert standard.learning_rate == 3.0e-4
    assert standard.warmup_fraction == 0.05
    assert standard.weight_decay == 0.10
    assert native.learning_rate == 1.5e-3
    assert native.warmup_fraction == 0.0
    assert native.weight_decay == 0.0


def test_v6_eager_projection_has_no_compile_or_host_audit_in_hot_path() -> None:
    config = experiment.HypersphericalTransformerFalsificationConfig(
        sequence_length=16,
        normalized_width=32,
        normalized_layers=1,
        normalized_attention_heads=4,
        normalized_hidden_width=64,
        execution_backend="eager",
    )
    model = experiment._build_model(
        "normalized",
        vocab_size=96,
        config=config,
    )
    assert isinstance(model, MarulhoNormalizedLanguageModel)
    step, evidence = experiment._prepare_projection_backend(model, config=config)
    with torch.no_grad():
        model.token_embedding.weight.mul_(2.0)

    result = step()

    assert result is None
    assert evidence["backend"] == "pytorch_eager"
    assert evidence["compile_seconds"] == 0.0
    assert model.hyperspherical_weight_evidence()[
        "maximum_unit_norm_error"
    ] < 1.0e-5


def _arm(
    name: str,
    *,
    loss: float,
    free: float,
    tokens: int = 16_785_792,
) -> dict:
    return {
        "name": name,
        "processed_tokens": tokens,
        "heldout": {"heldout_loss": loss},
        "relation": {"generation_exact_accuracy": free},
    }


def test_v6_decision_scales_same_recipe_architecture_win() -> None:
    arms = (
        _arm("transformer_standard", loss=4.00, free=0.20),
        _arm("transformer_native", loss=3.99, free=0.21),
        _arm("normalized_standard", loss=3.99, free=0.23),
        _arm("normalized_native", loss=3.98, free=0.24),
    )

    assert experiment.hyperspherical_transformer_decision(arms) == (
        "scale_v6_hyperspherical_transformer_to_64m"
    )


def test_v6_decision_does_not_confuse_recipe_gain_with_architecture_gain() -> None:
    arms = (
        _arm("transformer_standard", loss=4.00, free=0.20),
        _arm("transformer_native", loss=3.98, free=0.24),
        _arm("normalized_standard", loss=4.01, free=0.20),
        _arm("normalized_native", loss=3.98, free=0.23),
    )

    assert experiment.hyperspherical_transformer_decision(arms) == (
        "adopt_native_recipe_on_transformer_retire_v6_architecture"
    )


def test_v6_decision_requires_native_candidate_to_beat_both_controls() -> None:
    arms = (
        _arm("transformer_standard", loss=3.98, free=0.24),
        _arm("transformer_native", loss=4.00, free=0.20),
        _arm("normalized_standard", loss=4.01, free=0.20),
        _arm("normalized_native", loss=3.99, free=0.23),
    )

    assert experiment.hyperspherical_transformer_decision(arms) == (
        "retire_v6_no_architecture_quality_gain"
    )


def test_v6_decision_preserves_behavior_only_signal_and_short_smoke() -> None:
    arms = (
        _arm("transformer_standard", loss=4.00, free=0.20),
        _arm("transformer_native", loss=4.01, free=0.20),
        _arm("normalized_standard", loss=4.01, free=0.23),
        _arm("normalized_native", loss=4.02, free=0.24),
    )

    assert experiment.hyperspherical_transformer_decision(arms) == (
        "redesign_v6_behavior_signal_without_loss_gain"
    )
    smoke = tuple({**row, "processed_tokens": 82_944} for row in arms)
    assert experiment.hyperspherical_transformer_decision(smoke) == (
        "incomplete_v6_mechanism_smoke"
    )
