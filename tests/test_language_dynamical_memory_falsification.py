from __future__ import annotations

import marulho.evaluation.language_dynamical_memory_falsification as experiment


def test_v7_schedule_is_exact_and_reproducible() -> None:
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


def test_v7_default_models_are_within_one_tenth_percent() -> None:
    config = experiment.DynamicalMemoryFalsificationConfig()
    baseline = experiment._build_model(
        "transformer",
        vocab_size=8192,
        config=config,
    )
    candidate = experiment._build_model(
        "dynamical_memory",
        vocab_size=8192,
        config=config,
    )
    baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_count = sum(parameter.numel() for parameter in candidate.parameters())

    assert baseline_count == 20_976_128
    assert candidate_count == 20_977_152
    assert abs(candidate_count - baseline_count) / baseline_count < 0.001


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


def test_v7_decision_requires_learned_memory_to_beat_every_control() -> None:
    arms = (
        _arm("transformer", loss=4.00, free=0.20),
        _arm("memory_off", loss=4.10, free=0.10),
        _arm("single_scale", loss=4.06, free=0.14),
        _arm("multiscale_always", loss=4.05, free=0.15),
        _arm("multiscale_random", loss=4.07, free=0.13),
        _arm("multiscale_learned", loss=3.98, free=0.24),
    )

    assert experiment.dynamical_memory_decision(arms) == (
        "replicate_v7_multiscale_learned_before_scale"
    )


def test_v7_decision_preserves_a_winning_control_without_gate_claim() -> None:
    arms = (
        _arm("transformer", loss=4.00, free=0.20),
        _arm("memory_off", loss=4.03, free=0.20),
        _arm("single_scale", loss=4.02, free=0.21),
        _arm("multiscale_always", loss=3.97, free=0.25),
        _arm("multiscale_random", loss=4.03, free=0.20),
        _arm("multiscale_learned", loss=3.98, free=0.24),
    )

    assert experiment.dynamical_memory_decision(arms) == (
        "replicate_v7_multiscale_always_without_learned_gate_claim"
    )


def test_v7_decision_can_select_reduced_mlp_control() -> None:
    arms = (
        _arm("transformer", loss=4.00, free=0.20),
        _arm("memory_off", loss=3.98, free=0.24),
        _arm("single_scale", loss=4.02, free=0.20),
        _arm("multiscale_always", loss=4.02, free=0.20),
        _arm("multiscale_random", loss=4.03, free=0.20),
        _arm("multiscale_learned", loss=4.01, free=0.21),
    )

    assert experiment.dynamical_memory_decision(arms) == (
        "replicate_v7_reduced_mlp_control_before_scale"
    )


def test_v7_decision_keeps_behavior_only_signal_as_redesign() -> None:
    arms = (
        _arm("transformer", loss=4.00, free=0.20),
        _arm("memory_off", loss=4.03, free=0.20),
        _arm("single_scale", loss=4.02, free=0.22),
        _arm("multiscale_always", loss=4.01, free=0.23),
        _arm("multiscale_random", loss=4.04, free=0.20),
        _arm("multiscale_learned", loss=4.01, free=0.25),
    )

    assert experiment.dynamical_memory_decision(arms) == (
        "redesign_v7_behavior_signal_without_loss_gain"
    )


def test_v7_decision_retires_no_gain_and_labels_short_smoke() -> None:
    arms = (
        _arm("transformer", loss=4.00, free=0.20),
        _arm("memory_off", loss=4.03, free=0.20),
        _arm("single_scale", loss=4.02, free=0.20),
        _arm("multiscale_always", loss=4.01, free=0.21),
        _arm("multiscale_random", loss=4.04, free=0.20),
        _arm("multiscale_learned", loss=4.01, free=0.21),
    )

    assert experiment.dynamical_memory_decision(arms) == (
        "retire_v7_no_quality_or_control_gain"
    )
    smoke = tuple({**row, "processed_tokens": 82_944} for row in arms)
    assert experiment.dynamical_memory_decision(smoke) == (
        "incomplete_v7_mechanism_smoke"
    )
