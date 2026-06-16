from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from marulho.core.fused_route_vote_cuda import (
    fused_route_vote_cuda,
    fused_route_vote_kernel_variant,
)


def _route_state_kwargs(
    steps_since_win: torch.Tensor,
    *,
    step: int = 0,
    all_materialized_step: int = 0,
    last_update: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    return {
        "steps_since_win_last_update_step": (
            torch.zeros_like(steps_since_win)
            if last_update is None
            else last_update
        ),
        "state_transition_step_counter": torch.tensor(
            step,
            dtype=torch.long,
            device=steps_since_win.device,
        ),
        "state_transition_all_materialized_step": torch.tensor(
            all_materialized_step,
            dtype=torch.long,
            device=steps_since_win.device,
        ),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize(
    ("vector_count", "column_dim", "location_dim", "candidate_count"),
    [
        (32, 8, 4, 5),
        (1024, 64, 64, 10),
    ],
)
@pytest.mark.parametrize("silent", [False, True])
def test_fused_route_vote_matches_tensor_routing_and_vote(
    vector_count: int,
    column_dim: int,
    location_dim: int,
    candidate_count: int,
    silent: bool,
) -> None:
    torch.manual_seed(20260612)
    device = torch.device("cuda")
    routing_key = F.normalize(torch.rand(column_dim, device=device), dim=0)
    routing_vectors = F.normalize(
        torch.rand(vector_count, column_dim, device=device),
        dim=1,
    )
    routing_ids = torch.randperm(vector_count, device=device, dtype=torch.long)
    prototypes = F.normalize(
        torch.rand(vector_count, column_dim, device=device),
        dim=1,
    )
    thresholds = torch.full(
        (vector_count,),
        4.0 if silent else 0.2,
        device=device,
    )
    locations = torch.rand(vector_count, location_dim, device=device)
    previous_winner = torch.tensor([7], dtype=torch.long, device=device)
    steps_since_win = torch.zeros(vector_count, dtype=torch.long, device=device)
    memory_pressure = torch.zeros(vector_count, device=device)
    route_filter_control = torch.tensor(
        [0, 2000, 0, 950000],
        dtype=torch.long,
        device=device,
    )
    route_filter_state = torch.zeros(12, dtype=torch.long, device=device)

    search_key = F.normalize(routing_key.unsqueeze(0), dim=1)
    scores = search_key @ routing_vectors.T
    positions = torch.topk(scores, k=candidate_count, dim=1).indices[0]
    expected_candidates = routing_ids[positions]
    competition_key = F.normalize(routing_key.clamp(min=1e-6), dim=0)
    location_similarity = F.cosine_similarity(
        locations[expected_candidates],
        locations[previous_winner.item()].unsqueeze(0),
        dim=1,
    )
    combined = (
        prototypes[expected_candidates] @ competition_key
    ) * (1.0 + 0.3 * location_similarity.clamp(-1.0, 1.0))
    activation = torch.relu(combined - thresholds[expected_candidates])
    expected_local = (
        torch.argmax(activation)
        if bool((activation.max() > 0.0).item())
        else torch.argmax(combined)
    )
    expected_winner = expected_candidates[expected_local]

    scores_out = torch.empty(vector_count, device=device)
    candidates_out = torch.empty(
        candidate_count,
        dtype=torch.long,
        device=device,
    )
    winner_out = torch.empty(1, dtype=torch.long, device=device)
    strength_out = torch.empty(1, device=device)
    had_positive = torch.empty((), dtype=torch.bool, device=device)
    reconstruction_error_out = torch.empty(1, device=device)
    assert (
        fused_route_vote_kernel_variant(routing_vectors, candidates_out)
        == "two_stage_route_vote"
    )
    fused_route_vote_cuda(
        routing_key=routing_key,
        routing_vectors=routing_vectors,
        routing_ids=routing_ids,
        steps_since_win=steps_since_win,
        **_route_state_kwargs(steps_since_win),
        prototypes=prototypes,
        thresholds=thresholds,
        prediction_location=locations,
        memory_pressure=memory_pressure,
        previous_winner=previous_winner,
        route_filter_control=route_filter_control,
        route_filter_state_out=route_filter_state,
        scores_out=scores_out,
        candidates_out=candidates_out,
        winner_out=winner_out,
        strength_out=strength_out,
        competition_had_positive=had_positive,
        reconstruction_error_out=reconstruction_error_out,
    )
    torch.cuda.synchronize()

    assert torch.equal(candidates_out, expected_candidates)
    assert int(winner_out.item()) == int(expected_winner.item())
    assert int(previous_winner.item()) == int(expected_winner.item())
    assert float(strength_out.item()) == 1.0
    assert bool(had_positive.item()) is bool((activation.max() > 0.0).item())
    expected_reconstruction_error = torch.clamp(1.0 - scores.max(), min=0.0)
    assert torch.allclose(
        reconstruction_error_out.squeeze(0),
        expected_reconstruction_error.squeeze(0),
        rtol=0.0,
        atol=1e-6,
    )
    assert route_filter_state.tolist()[:3] == [0, 0, 0]
    assert route_filter_state.tolist()[8:11] == [0, 0, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_fused_route_vote_scores_indexed_route_bank_only() -> None:
    torch.manual_seed(20260616)
    device = torch.device("cuda")
    vector_count = 32
    column_dim = 8
    location_dim = 4
    candidate_count = 5
    bank_positions = torch.tensor(
        [3, 5, 8, 13, 21, 1, 2],
        dtype=torch.long,
        device=device,
    )
    routing_key = F.normalize(torch.rand(column_dim, device=device), dim=0)
    routing_vectors = F.normalize(
        torch.rand(vector_count, column_dim, device=device),
        dim=1,
    )
    routing_ids = torch.randperm(vector_count, device=device, dtype=torch.long)
    prototypes = F.normalize(
        torch.rand(vector_count, column_dim, device=device),
        dim=1,
    )
    thresholds = torch.full((vector_count,), 0.2, device=device)
    locations = torch.rand(vector_count, location_dim, device=device)
    previous_winner = torch.tensor([7], dtype=torch.long, device=device)
    steps_since_win = torch.zeros(vector_count, dtype=torch.long, device=device)
    memory_pressure = torch.zeros(vector_count, device=device)
    route_filter_control = torch.tensor(
        [0, 2000, 0, 950000],
        dtype=torch.long,
        device=device,
    )
    route_filter_state = torch.zeros(12, dtype=torch.long, device=device)
    bank_scores = routing_key.unsqueeze(0) @ routing_vectors[bank_positions].T
    expected_offsets = torch.topk(
        bank_scores,
        k=candidate_count,
        dim=1,
    ).indices[0]
    expected_candidates = routing_ids[bank_positions[expected_offsets]]
    scores_out = torch.empty(int(bank_positions.numel()), device=device)
    candidates_out = torch.empty(candidate_count, dtype=torch.long, device=device)
    winner_out = torch.empty(1, dtype=torch.long, device=device)
    strength_out = torch.empty(1, device=device)
    had_positive = torch.empty((), dtype=torch.bool, device=device)
    reconstruction_error_out = torch.empty(1, device=device)

    assert (
        fused_route_vote_kernel_variant(
            routing_vectors,
            candidates_out,
            bank_positions,
        )
        == "indexed_route_bank_vote"
    )
    fused_route_vote_cuda(
        routing_key=routing_key,
        routing_vectors=routing_vectors,
        routing_ids=routing_ids,
        route_positions=bank_positions,
        steps_since_win=steps_since_win,
        **_route_state_kwargs(steps_since_win),
        prototypes=prototypes,
        thresholds=thresholds,
        prediction_location=locations,
        memory_pressure=memory_pressure,
        previous_winner=previous_winner,
        route_filter_control=route_filter_control,
        route_filter_state_out=route_filter_state,
        scores_out=scores_out,
        candidates_out=candidates_out,
        winner_out=winner_out,
        strength_out=strength_out,
        competition_had_positive=had_positive,
        reconstruction_error_out=reconstruction_error_out,
    )
    torch.cuda.synchronize()

    assert torch.equal(candidates_out, expected_candidates)
    assert route_filter_state.tolist()[5] == int(bank_positions.numel())
    assert route_filter_state.tolist()[6] == candidate_count
    assert torch.allclose(
        reconstruction_error_out.squeeze(0),
        torch.clamp(1.0 - bank_scores.max(), min=0.0).squeeze(0),
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_fused_route_vote_de_duplicates_repeated_indexed_route_rows() -> None:
    torch.manual_seed(20260617)
    device = torch.device("cuda")
    vector_count = 32
    column_dim = 8
    location_dim = 4
    candidate_count = 5
    bank_positions = torch.tensor(
        [3, 5, 8, 13, 21, 5, 3],
        dtype=torch.long,
        device=device,
    )
    routing_key = F.normalize(torch.rand(column_dim, device=device), dim=0)
    routing_vectors = F.normalize(
        torch.rand(vector_count, column_dim, device=device),
        dim=1,
    )
    routing_ids = torch.randperm(vector_count, device=device, dtype=torch.long)
    prototypes = F.normalize(
        torch.rand(vector_count, column_dim, device=device),
        dim=1,
    )
    thresholds = torch.full((vector_count,), 0.2, device=device)
    locations = torch.rand(vector_count, location_dim, device=device)
    previous_winner = torch.tensor([7], dtype=torch.long, device=device)
    steps_since_win = torch.zeros(vector_count, dtype=torch.long, device=device)
    memory_pressure = torch.zeros(vector_count, device=device)
    route_filter_control = torch.tensor(
        [0, 2000, 0, 950000],
        dtype=torch.long,
        device=device,
    )
    route_filter_state = torch.zeros(12, dtype=torch.long, device=device)
    bank_scores = routing_key.unsqueeze(0) @ routing_vectors[bank_positions].T
    ordered_offsets = torch.argsort(bank_scores[0], descending=True)
    ordered_ids = routing_ids[bank_positions[ordered_offsets]].detach().cpu().tolist()
    expected_ids: list[int] = []
    for candidate_id in ordered_ids:
        if int(candidate_id) not in expected_ids:
            expected_ids.append(int(candidate_id))
        if len(expected_ids) == candidate_count:
            break
    scores_out = torch.empty(int(bank_positions.numel()), device=device)
    candidates_out = torch.empty(candidate_count, dtype=torch.long, device=device)
    winner_out = torch.empty(1, dtype=torch.long, device=device)
    strength_out = torch.empty(1, device=device)
    had_positive = torch.empty((), dtype=torch.bool, device=device)
    reconstruction_error_out = torch.empty(1, device=device)

    fused_route_vote_cuda(
        routing_key=routing_key,
        routing_vectors=routing_vectors,
        routing_ids=routing_ids,
        route_positions=bank_positions,
        steps_since_win=steps_since_win,
        **_route_state_kwargs(steps_since_win),
        prototypes=prototypes,
        thresholds=thresholds,
        prediction_location=locations,
        memory_pressure=memory_pressure,
        previous_winner=previous_winner,
        route_filter_control=route_filter_control,
        route_filter_state_out=route_filter_state,
        scores_out=scores_out,
        candidates_out=candidates_out,
        winner_out=winner_out,
        strength_out=strength_out,
        competition_had_positive=had_positive,
        reconstruction_error_out=reconstruction_error_out,
    )
    torch.cuda.synchronize()

    actual_ids = candidates_out.detach().cpu().tolist()
    assert actual_ids == expected_ids
    assert len(set(actual_ids)) == candidate_count
    assert route_filter_state.tolist()[5] == int(bank_positions.numel())
    assert route_filter_state.tolist()[6] == candidate_count


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_fused_route_vote_masks_deep_sleep_before_candidate_vote() -> None:
    torch.manual_seed(20260615)
    device = torch.device("cuda")
    vector_count = 16
    column_dim = 8
    location_dim = 4
    candidate_count = 4
    routing_key = F.normalize(torch.rand(column_dim, device=device), dim=0)
    routing_vectors = F.normalize(
        torch.rand(vector_count, column_dim, device=device),
        dim=1,
    )
    routing_ids = torch.arange(vector_count, device=device, dtype=torch.long)
    scores = routing_key.unsqueeze(0) @ routing_vectors.T
    unfiltered_positions = torch.topk(scores, k=candidate_count, dim=1).indices[0]
    deep_sleep_ids = routing_ids[unfiltered_positions[:2]]
    steps_since_win = torch.zeros(vector_count, dtype=torch.long, device=device)
    steps_since_win[deep_sleep_ids] = 2000
    memory_pressure = torch.zeros(vector_count, device=device)
    route_filter_control = torch.tensor(
        [1, 2000, 0, 950000],
        dtype=torch.long,
        device=device,
    )
    route_filter_state = torch.zeros(12, dtype=torch.long, device=device)
    prototypes = F.normalize(
        torch.rand(vector_count, column_dim, device=device),
        dim=1,
    )
    thresholds = torch.full((vector_count,), 0.2, device=device)
    locations = torch.rand(vector_count, location_dim, device=device)
    previous_winner = torch.tensor([3], dtype=torch.long, device=device)
    scores_out = torch.empty(vector_count, device=device)
    candidates_out = torch.empty(
        candidate_count,
        dtype=torch.long,
        device=device,
    )
    winner_out = torch.empty(1, dtype=torch.long, device=device)
    strength_out = torch.empty(1, device=device)
    had_positive = torch.empty((), dtype=torch.bool, device=device)
    reconstruction_error_out = torch.empty(1, device=device)

    fused_route_vote_cuda(
        routing_key=routing_key,
        routing_vectors=routing_vectors,
        routing_ids=routing_ids,
        steps_since_win=steps_since_win,
        **_route_state_kwargs(steps_since_win),
        prototypes=prototypes,
        thresholds=thresholds,
        prediction_location=locations,
        memory_pressure=memory_pressure,
        previous_winner=previous_winner,
        route_filter_control=route_filter_control,
        route_filter_state_out=route_filter_state,
        scores_out=scores_out,
        candidates_out=candidates_out,
        winner_out=winner_out,
        strength_out=strength_out,
        competition_had_positive=had_positive,
        reconstruction_error_out=reconstruction_error_out,
    )
    torch.cuda.synchronize()

    expected_awake_positions = torch.topk(
        scores.masked_fill(steps_since_win.unsqueeze(0) >= 2000, -float("inf")),
        k=candidate_count,
        dim=1,
    ).indices[0]
    expected_awake_candidates = routing_ids[expected_awake_positions]
    assert torch.equal(candidates_out, expected_awake_candidates)
    assert not torch.isin(candidates_out, deep_sleep_ids).any()
    state = route_filter_state.tolist()
    assert state[0] == 1
    assert state[1] == 1
    assert state[2] == 2
    assert state[3] == vector_count - 2
    assert state[4] == 0
    assert state[5] == vector_count
    assert state[6] == candidate_count
    assert state[7] == 0
    assert state[8] == 0
    assert state[9] == 0
    assert state[10] == 0
    assert state[11] == vector_count - 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_fused_route_vote_uses_logical_stale_age_for_deep_sleep() -> None:
    torch.manual_seed(20260616)
    device = torch.device("cuda")
    vector_count = 16
    column_dim = 8
    location_dim = 4
    candidate_count = 4
    routing_key = F.normalize(torch.rand(column_dim, device=device), dim=0)
    routing_vectors = F.normalize(torch.rand(vector_count, column_dim, device=device), dim=1)
    routing_ids = torch.arange(vector_count, dtype=torch.long, device=device)
    prototypes = F.normalize(torch.rand(vector_count, column_dim, device=device), dim=1)
    thresholds = torch.full((vector_count,), 0.2, device=device)
    locations = torch.rand(vector_count, location_dim, device=device)
    previous_winner = torch.tensor([1], dtype=torch.long, device=device)
    steps_since_win = torch.zeros(vector_count, dtype=torch.long, device=device)
    last_update = torch.full((vector_count,), 1999, dtype=torch.long, device=device)
    scores = F.normalize(routing_key.unsqueeze(0), dim=1) @ routing_vectors.T
    sleepy_positions = torch.topk(scores, k=2, dim=1).indices[0]
    sleepy_ids = routing_ids[sleepy_positions]
    last_update[sleepy_ids] = 0
    memory_pressure = torch.zeros(vector_count, device=device)
    route_filter_control = torch.tensor([1, 2000, 0, 950000], dtype=torch.long, device=device)
    route_filter_state = torch.zeros(12, dtype=torch.long, device=device)
    scores_out = torch.empty(vector_count, device=device)
    candidates_out = torch.empty(candidate_count, dtype=torch.long, device=device)
    winner_out = torch.empty(1, dtype=torch.long, device=device)
    strength_out = torch.empty(1, device=device)
    had_positive = torch.empty((), dtype=torch.bool, device=device)
    reconstruction_error_out = torch.empty(1, device=device)

    fused_route_vote_cuda(
        routing_key=routing_key,
        routing_vectors=routing_vectors,
        routing_ids=routing_ids,
        steps_since_win=steps_since_win,
        **_route_state_kwargs(
            steps_since_win,
            step=2000,
            all_materialized_step=0,
            last_update=last_update,
        ),
        prototypes=prototypes,
        thresholds=thresholds,
        prediction_location=locations,
        memory_pressure=memory_pressure,
        previous_winner=previous_winner,
        route_filter_control=route_filter_control,
        route_filter_state_out=route_filter_state,
        scores_out=scores_out,
        candidates_out=candidates_out,
        winner_out=winner_out,
        strength_out=strength_out,
        competition_had_positive=had_positive,
        reconstruction_error_out=reconstruction_error_out,
    )
    torch.cuda.synchronize()

    assert not torch.isin(candidates_out, sleepy_ids).any()
    state = route_filter_state.tolist()
    assert state[1] == 1
    assert state[2] == 2
    assert state[3] == vector_count - 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_fused_route_vote_backfills_when_awake_routes_are_insufficient() -> None:
    torch.manual_seed(20260616)
    device = torch.device("cuda")
    vector_count = 12
    column_dim = 8
    location_dim = 4
    candidate_count = 5
    routing_key = F.normalize(torch.rand(column_dim, device=device), dim=0)
    routing_vectors = F.normalize(
        torch.rand(vector_count, column_dim, device=device),
        dim=1,
    )
    routing_ids = torch.arange(vector_count, device=device, dtype=torch.long)
    scores = routing_key.unsqueeze(0) @ routing_vectors.T
    awake_ids = torch.tensor([1, 7], dtype=torch.long, device=device)
    steps_since_win = torch.full(
        (vector_count,),
        2000,
        dtype=torch.long,
        device=device,
    )
    steps_since_win[awake_ids] = 0
    memory_pressure = torch.zeros(vector_count, device=device)
    route_filter_control = torch.tensor(
        [1, 2000, 0, 950000],
        dtype=torch.long,
        device=device,
    )
    route_filter_state = torch.zeros(12, dtype=torch.long, device=device)
    prototypes = F.normalize(
        torch.rand(vector_count, column_dim, device=device),
        dim=1,
    )
    thresholds = torch.full((vector_count,), 0.2, device=device)
    locations = torch.rand(vector_count, location_dim, device=device)
    previous_winner = torch.tensor([3], dtype=torch.long, device=device)
    scores_out = torch.empty(vector_count, device=device)
    candidates_out = torch.empty(
        candidate_count,
        dtype=torch.long,
        device=device,
    )
    winner_out = torch.empty(1, dtype=torch.long, device=device)
    strength_out = torch.empty(1, device=device)
    had_positive = torch.empty((), dtype=torch.bool, device=device)
    reconstruction_error_out = torch.empty(1, device=device)

    fused_route_vote_cuda(
        routing_key=routing_key,
        routing_vectors=routing_vectors,
        routing_ids=routing_ids,
        steps_since_win=steps_since_win,
        **_route_state_kwargs(steps_since_win),
        prototypes=prototypes,
        thresholds=thresholds,
        prediction_location=locations,
        memory_pressure=memory_pressure,
        previous_winner=previous_winner,
        route_filter_control=route_filter_control,
        route_filter_state_out=route_filter_state,
        scores_out=scores_out,
        candidates_out=candidates_out,
        winner_out=winner_out,
        strength_out=strength_out,
        competition_had_positive=had_positive,
        reconstruction_error_out=reconstruction_error_out,
    )
    torch.cuda.synchronize()

    expected = routing_ids[
        torch.topk(scores, k=candidate_count, dim=1).indices[0]
    ]
    assert torch.equal(candidates_out, expected)
    state = route_filter_state.tolist()
    assert state[0] == 1
    assert state[1] == 0
    assert state[2] == 0
    assert state[3] == int(awake_ids.numel())
    assert state[4] == 1
    assert state[7] == 0
    assert state[8] == 0
    assert state[10] == 0
    assert state[11] == int(awake_ids.numel())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_fused_route_vote_masks_memory_pressure_before_candidate_vote() -> None:
    torch.manual_seed(20260616)
    device = torch.device("cuda")
    vector_count = 16
    column_dim = 8
    location_dim = 4
    candidate_count = 4
    routing_key = F.normalize(torch.rand(column_dim, device=device), dim=0)
    routing_vectors = F.normalize(
        torch.rand(vector_count, column_dim, device=device),
        dim=1,
    )
    routing_ids = torch.arange(vector_count, device=device, dtype=torch.long)
    scores = routing_key.unsqueeze(0) @ routing_vectors.T
    pressure_ids = routing_ids[torch.topk(scores, k=candidate_count, dim=1).indices[0][:2]]
    steps_since_win = torch.zeros(vector_count, dtype=torch.long, device=device)
    memory_pressure = torch.zeros(vector_count, device=device)
    memory_pressure[pressure_ids] = 0.99
    route_filter_control = torch.tensor(
        [0, 2000, 1, 500000],
        dtype=torch.long,
        device=device,
    )
    route_filter_state = torch.zeros(12, dtype=torch.long, device=device)
    prototypes = F.normalize(
        torch.rand(vector_count, column_dim, device=device),
        dim=1,
    )
    thresholds = torch.full((vector_count,), 0.2, device=device)
    locations = torch.rand(vector_count, location_dim, device=device)
    previous_winner = torch.tensor([3], dtype=torch.long, device=device)
    scores_out = torch.empty(vector_count, device=device)
    candidates_out = torch.empty(
        candidate_count,
        dtype=torch.long,
        device=device,
    )
    winner_out = torch.empty(1, dtype=torch.long, device=device)
    strength_out = torch.empty(1, device=device)
    had_positive = torch.empty((), dtype=torch.bool, device=device)
    reconstruction_error_out = torch.empty(1, device=device)

    fused_route_vote_cuda(
        routing_key=routing_key,
        routing_vectors=routing_vectors,
        routing_ids=routing_ids,
        steps_since_win=steps_since_win,
        **_route_state_kwargs(steps_since_win),
        prototypes=prototypes,
        thresholds=thresholds,
        prediction_location=locations,
        memory_pressure=memory_pressure,
        previous_winner=previous_winner,
        route_filter_control=route_filter_control,
        route_filter_state_out=route_filter_state,
        scores_out=scores_out,
        candidates_out=candidates_out,
        winner_out=winner_out,
        strength_out=strength_out,
        competition_had_positive=had_positive,
        reconstruction_error_out=reconstruction_error_out,
    )
    torch.cuda.synchronize()

    expected_positions = torch.topk(
        scores.masked_fill(memory_pressure.unsqueeze(0) > 0.5, -float("inf")),
        k=candidate_count,
        dim=1,
    ).indices[0]
    expected_candidates = routing_ids[expected_positions]
    assert torch.equal(candidates_out, expected_candidates)
    assert not torch.isin(candidates_out, pressure_ids).any()
    assert not torch.isin(winner_out, pressure_ids).any()
    state = route_filter_state.tolist()
    assert state[0] == 0
    assert state[1] == 1
    assert state[2] == 0
    assert state[3] == vector_count
    assert state[4] == 0
    assert state[8] == 1
    assert state[9] == 1
    assert state[10] == 2
    assert state[11] == vector_count - 2
