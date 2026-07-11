from marulho.evaluation.language_depth_allocation_falsification import (
    ARM_NAMES,
    PROFILE_WIDTHS,
    build_matched_schedule,
    depth_allocation_decision,
)


def _arm(name: str, *, loss: float, free: float, tokens: int = 16_777_216):
    return {
        "name": name,
        "processed_tokens": tokens,
        "heldout": {"heldout_loss": loss},
        "relation": {"generation_exact_accuracy": free},
    }


def test_v8_profiles_have_exact_shared_budget() -> None:
    assert set(PROFILE_WIDTHS) == set(ARM_NAMES)
    assert {sum(widths) for widths in PROFILE_WIDTHS.values()} == {8192}
    assert PROFILE_WIDTHS["uniform"] == (2048, 2048, 2048, 2048)
    assert PROFILE_WIDTHS["early_heavy"] == tuple(
        reversed(PROFILE_WIDTHS["late_heavy"])
    )


def test_v8_schedule_is_exact_and_reproducible() -> None:
    first = build_matched_schedule(
        step_count=1619,
        relation_fraction=0.20,
        relation_batch_count=400,
        general_batch_counts=(700, 700),
        seed=1337,
    )
    second = build_matched_schedule(
        step_count=1619,
        relation_fraction=0.20,
        relation_batch_count=400,
        general_batch_counts=(700, 700),
        seed=1337,
    )
    assert first == second
    assert len(first) == 1619
    assert sum(kind == "relation" for kind, _index in first) == 323
    assert all(index >= 0 for _kind, index in first)

    changed = build_matched_schedule(
        step_count=1619,
        relation_fraction=0.20,
        relation_batch_count=400,
        general_batch_counts=(700, 700),
        seed=7331,
    )
    assert changed != first
    assert sum(kind == "relation" for kind, _index in changed) == 323


def test_v8_decision_requires_loss_and_free_generation() -> None:
    arms = [
        _arm("uniform", loss=4.10, free=0.15),
        _arm("early_heavy", loss=4.08, free=0.16),
        _arm("late_heavy", loss=4.07, free=0.20),
    ]
    assert depth_allocation_decision(arms) == (
        "replicate_v8_late_heavy_before_scale"
    )


def test_v8_decision_does_not_promote_loss_only() -> None:
    arms = [
        _arm("uniform", loss=4.10, free=0.15),
        _arm("early_heavy", loss=4.08, free=0.15),
        _arm("late_heavy", loss=4.11, free=0.16),
    ]
    assert depth_allocation_decision(arms) == (
        "redesign_v8_loss_signal_without_free_generation"
    )


def test_v8_decision_does_not_promote_behavior_only() -> None:
    arms = [
        _arm("uniform", loss=4.10, free=0.15),
        _arm("early_heavy", loss=4.10, free=0.20),
        _arm("late_heavy", loss=4.11, free=0.19),
    ]
    assert depth_allocation_decision(arms) == (
        "redesign_v8_behavior_signal_without_loss_gain"
    )


def test_v8_decision_preserves_disjoint_quality_signals() -> None:
    arms = [
        _arm("uniform", loss=4.10, free=0.15),
        _arm("early_heavy", loss=4.08, free=0.15),
        _arm("late_heavy", loss=4.11, free=0.19),
    ]
    assert depth_allocation_decision(arms) == (
        "redesign_v8_disjoint_loss_and_behavior_signals"
    )


def test_v8_decision_labels_short_or_missing_runs() -> None:
    complete_short = [
        _arm("uniform", loss=4.10, free=0.15, tokens=10_000),
        _arm("early_heavy", loss=4.08, free=0.20, tokens=10_000),
        _arm("late_heavy", loss=4.09, free=0.20, tokens=10_000),
    ]
    assert depth_allocation_decision(complete_short) == (
        "incomplete_v8_mechanism_smoke"
    )
    assert depth_allocation_decision(complete_short[:-1]) == (
        "incomplete_v8_depth_allocation_comparison"
    )


def test_v8_decision_retires_no_gain() -> None:
    arms = [
        _arm("uniform", loss=4.10, free=0.15),
        _arm("early_heavy", loss=4.10, free=0.15),
        _arm("late_heavy", loss=4.11, free=0.14),
    ]
    assert depth_allocation_decision(arms) == (
        "retire_v8_static_depth_allocation"
    )


def test_v8_durability_promotes_only_a_large_budget_pair_win() -> None:
    arms = [
        _arm("uniform", loss=4.00, free=0.20, tokens=67_110_000),
        _arm("early_heavy", loss=3.98, free=0.25, tokens=67_110_000),
    ]
    assert depth_allocation_decision(
        arms,
        comparison_stage="durability",
    ) == "promote_v8_early_heavy_to_quality_baseline"

    short = [dict(row, processed_tokens=67_000_000) for row in arms]
    assert depth_allocation_decision(
        short,
        comparison_stage="durability",
    ) == "incomplete_v8_durability_budget"


def test_v8_durability_can_retire_a_non_durable_early_win() -> None:
    arms = [
        _arm("uniform", loss=4.00, free=0.20, tokens=67_110_000),
        _arm("early_heavy", loss=4.00, free=0.20, tokens=67_110_000),
    ]
    assert depth_allocation_decision(
        arms,
        comparison_stage="durability",
    ) == "retire_v8_early_heavy_not_durable"
