from __future__ import annotations

from marulho.evaluation.language_hidden_state_memory_falsification import (
    _calibration_cases,
    _config_grid,
)
from marulho.evaluation.language_relation_binding_experiment import RelationCase


def test_v41_calibration_cases_are_signature_disjoint_and_balanced() -> None:
    frozen = (
        RelationCase(
            case_id="frozen",
            kind="container",
            signature="container|frozen",
            prompt="Prompt Answer:",
            candidates=("a", "b"),
            correct_index=0,
        ),
    )
    cases = _calibration_cases(frozen, cases_per_kind=2, seed=41041)
    signatures = {case.signature for case in cases}

    assert len(cases) == 8
    assert len(signatures) == 8
    assert "container|frozen" not in signatures
    assert {kind: sum(case.kind == kind for case in cases) for kind in {
        "container", "ownership", "property", "event_order"
    }} == {
        "container": 2,
        "ownership": 2,
        "property": 2,
        "event_order": 2,
    }


def test_v41_grid_is_fixed_and_unique() -> None:
    grid = _config_grid()
    assert len(grid) == 12
    assert len({
        (row.top_k, row.similarity_threshold, row.interpolation_weight)
        for row in grid
    }) == 12
