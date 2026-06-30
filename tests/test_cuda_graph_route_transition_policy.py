from types import SimpleNamespace

import torch

from marulho.training.cuda_graph_route_transition import CudaGraphRouteTransition


def test_capture_candidate_sets_skip_dense_graph_after_homeostasis_gate() -> None:
    graph = object.__new__(CudaGraphRouteTransition)
    graph._trainer = SimpleNamespace(
        token_count=512,
        config=SimpleNamespace(
            candidate_homeostasis_start_tokens=512,
            candidate_predictive_update_start_tokens=512,
        ),
    )
    graph._runtime = SimpleNamespace(
        _all_columns=torch.arange(8),
        _route_candidates=torch.arange(4),
        candidate_predictive_transition_active=False,
    )

    candidate_sets = graph._capture_candidate_sets()

    assert [name for name, _, _ in candidate_sets] == ["candidate_subset"]
    assert candidate_sets[0][1] is graph._runtime._route_candidates
    assert candidate_sets[0][2] is False


def test_capture_candidate_sets_keep_dense_graph_before_homeostasis_gate() -> None:
    graph = object.__new__(CudaGraphRouteTransition)
    graph._trainer = SimpleNamespace(
        token_count=256,
        config=SimpleNamespace(
            candidate_homeostasis_start_tokens=512,
            candidate_predictive_update_start_tokens=512,
        ),
    )
    graph._runtime = SimpleNamespace(
        _all_columns=torch.arange(8),
        _route_candidates=torch.arange(4),
        candidate_predictive_transition_active=False,
    )

    candidate_sets = graph._capture_candidate_sets()

    assert [name for name, _, _ in candidate_sets] == [
        "all_columns",
        "candidate_subset",
    ]
    assert candidate_sets[0][1] is graph._runtime._all_columns
    assert candidate_sets[0][2] is False
    assert candidate_sets[1][1] is graph._runtime._route_candidates
    assert candidate_sets[1][2] is False
