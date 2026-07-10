from __future__ import annotations

from marulho.evaluation.language_sparse_event_falsification import (
    build_matched_schedule,
    build_probe_schedule,
    sparse_event_decision,
)


def test_v2_schedules_are_exact_and_reproducible() -> None:
    first = build_matched_schedule(
        step_count=20,
        relation_fraction=0.2,
        relation_batch_count=5,
        general_batch_counts=(9, 9),
        seed=17,
    )
    assert first == build_matched_schedule(
        step_count=20,
        relation_fraction=0.2,
        relation_batch_count=5,
        general_batch_counts=(9, 9),
        seed=17,
    )
    assert sum(kind == "relation" for kind, _ in first) == 4
    probes = build_probe_schedule(steps=405, rate=0.125, seed=19)
    assert len(probes) == 405
    assert sum(probes) == round(405 * 0.125)


def _arm(name: str, *, loss: float, free: float) -> dict:
    return {
        "name": name,
        "processed_tokens": 16_785_792,
        "heldout": {"heldout_loss": loss},
        "relation": {"generation_exact_accuracy": free},
    }


def test_v2_decision_requires_utility_to_beat_random_and_exact() -> None:
    exact = _arm("exact_only", loss=4.0, free=0.10)
    dense = _arm("dense", loss=4.0, free=0.10)
    random = _arm("random", loss=4.0, free=0.10)
    utility = _arm("utility", loss=3.98, free=0.13)
    assert sparse_event_decision((exact, dense, random, utility)) == (
        "continue_v2_to_64m_and_unseen_generation"
    )
    tied = _arm("utility", loss=3.999, free=0.11)
    assert sparse_event_decision((exact, dense, random, tied)) == (
        "retire_v2_utility_selector_not_better_than_random"
    )
    harmful = _arm("utility", loss=4.03, free=0.20)
    assert sparse_event_decision((exact, dense, random, harmful)) == (
        "retire_v2_sidecar_harms_exact_stream"
    )
    smoke = tuple({**row, "processed_tokens": 82_944} for row in (
        exact, dense, random, utility
    ))
    assert sparse_event_decision(smoke) == "incomplete_mechanism_smoke"
