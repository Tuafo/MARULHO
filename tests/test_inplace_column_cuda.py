from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from marulho.core.inplace_column_cuda import (
    inplace_column_transition_cuda,
    warmup_inplace_column_transition_cuda,
)
from marulho.core.predictive_columns import dense_predictive_transition


def _expected_retained_transition(
    prototypes: torch.Tensor,
    prototype_velocity: torch.Tensor,
    thresholds: torch.Tensor,
    win_rate_ema: torch.Tensor,
    steps_since_win: torch.Tensor,
    location: torch.Tensor,
    location_velocity: torch.Tensor,
    prediction_weights: torch.Tensor,
    prediction_error: torch.Tensor,
    prediction_failure_streak: torch.Tensor,
    confidence: torch.Tensor,
    routing_key: torch.Tensor,
    previous_routing_key: torch.Tensor,
    candidates: torch.Tensor,
    context_gain: torch.Tensor,
    candidate_consolidation: torch.Tensor,
    competitive_modulator: torch.Tensor,
    dopamine: torch.Tensor,
    serotonin: torch.Tensor,
    competitive_learning_rate: torch.Tensor,
    *,
    prototype_momentum: float,
    homeostasis_beta: float,
    homeostasis_lr: float,
    target_firing_rate: float,
    threshold_min: float,
    threshold_max: float,
    candidate_scoped_homeostasis: bool,
    prediction_error_ema_alpha: float,
    prediction_failure_streak_threshold: float,
    prediction_learning_rate: float,
    steps_since_win_last_update_step: torch.Tensor | None = None,
    state_transition_step: int = 0,
    state_transition_all_materialized_step: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Test-local retained semantics for the promoted in-place CUDA kernel."""

    x = routing_key.clamp(min=1e-6)
    x = x / x.norm().clamp(min=1e-8)
    candidate_ids = candidates.to(dtype=torch.long).flatten()
    candidate_prototypes = prototypes.index_select(0, candidate_ids)
    similarity = torch.mv(candidate_prototypes, x)
    combined = similarity * torch.clamp(
        context_gain.index_select(0, candidate_ids),
        min=0.5,
        max=1.5,
    )
    activation = torch.relu(combined - thresholds.index_select(0, candidate_ids))
    top_value, top_local_index = torch.topk(activation, k=1)
    best_combined_local = torch.argmax(combined).reshape(1)
    has_positive = top_value.max() > 0
    winner_local = torch.where(has_positive, top_local_index, best_combined_local)
    winners = candidate_ids.index_select(0, winner_local)
    winner_consolidation = candidate_consolidation.index_select(0, winner_local).mean()
    strengths = torch.where(
        has_positive,
        top_value / (top_value.sum() + 1e-8),
        torch.ones_like(top_value),
    )

    competition_thresholds = torch.where(
        has_positive,
        thresholds,
        torch.clamp(
            thresholds * 0.995,
            min=float(threshold_min),
            max=float(threshold_max),
        ),
    )
    assembly = torch.zeros_like(thresholds).scatter(
        0,
        winners,
        similarity.index_select(0, winner_local),
    )

    predictive_outputs = dense_predictive_transition(
        location,
        location_velocity,
        prediction_weights,
        prediction_error,
        prediction_failure_streak,
        confidence,
        routing_key,
        previous_routing_key,
        winners,
        has_previous_routing_key=True,
        error_ema_alpha=float(prediction_error_ema_alpha),
        failure_streak_threshold=float(prediction_failure_streak_threshold),
        learning_rate=float(prediction_learning_rate),
    )
    next_prediction_error = predictive_outputs[3]
    prediction_boost = torch.clamp(
        next_prediction_error.index_select(0, winners).mean(),
        min=0.5,
        max=2.0,
    )
    wake_plasticity_scale = torch.clamp(
        1.0 - 0.8 * winner_consolidation,
        min=0.2,
    ) * torch.clamp(1.0 - 0.6 * serotonin, min=0.2)
    dopamine_ltp_gain = 0.8 + 0.4 * dopamine
    effective_modulator = (
        competitive_modulator
        * wake_plasticity_scale
        * dopamine_ltp_gain
        * prediction_boost
    )

    winner_prototypes = prototypes.index_select(0, winners)
    winner_velocity = prototype_velocity.index_select(0, winners)
    prototype_lr = competitive_learning_rate * torch.clamp(
        torch.abs(effective_modulator) + 0.1,
        min=0.0,
        max=1.0,
    )
    next_winner_velocity = (
        float(prototype_momentum) * winner_velocity
        + prototype_lr * (x.unsqueeze(0) - winner_prototypes)
    )
    next_winner_prototypes = torch.clamp(
        winner_prototypes + next_winner_velocity,
        min=1e-6,
    )
    next_winner_prototypes = next_winner_prototypes / (
        next_winner_prototypes.norm(dim=1, keepdim=True).clamp(min=1e-8)
    )
    next_prototypes = prototypes.index_copy(0, winners, next_winner_prototypes)
    next_prototype_velocity = prototype_velocity.index_copy(
        0,
        winners,
        next_winner_velocity,
    )

    if candidate_ids.numel() < steps_since_win.numel():
        if steps_since_win_last_update_step is None:
            last_update = torch.zeros_like(steps_since_win)
        else:
            last_update = steps_since_win_last_update_step
        selected_last_update = torch.maximum(
            last_update.index_select(0, candidate_ids),
            torch.full_like(
                candidate_ids,
                int(state_transition_all_materialized_step),
            ),
        )
        selected_logical_steps = steps_since_win.index_select(0, candidate_ids)
        selected_logical_steps = selected_logical_steps + torch.clamp(
            torch.full_like(candidate_ids, int(state_transition_step))
            - selected_last_update,
            min=0,
        )
        next_selected_steps = torch.where(
            candidate_ids == winners[0],
            torch.zeros_like(selected_logical_steps),
            selected_logical_steps + 1,
        )
        next_steps_since_win = steps_since_win.index_copy(
            0,
            candidate_ids,
            next_selected_steps,
        )
        next_steps_since_win = next_steps_since_win.scatter(
            0,
            winners,
            torch.zeros_like(winners, dtype=steps_since_win.dtype),
        )
    else:
        next_steps_since_win = (steps_since_win + 1).scatter(
            0,
            winners,
            torch.zeros_like(winners, dtype=steps_since_win.dtype),
        )
    activity = torch.zeros_like(win_rate_ema).scatter(
        0,
        winners,
        torch.ones_like(winners, dtype=win_rate_ema.dtype),
    )
    if candidate_scoped_homeostasis:
        homeostasis_ids = candidate_ids
        selected_win_rate = win_rate_ema.index_select(0, homeostasis_ids)
        selected_activity = activity.index_select(0, homeostasis_ids)
        next_selected_win_rate = (
            (1.0 - float(homeostasis_beta)) * selected_win_rate
            + float(homeostasis_beta) * selected_activity
        )
        next_win_rate_ema = win_rate_ema.index_copy(
            0,
            homeostasis_ids,
            next_selected_win_rate,
        )
        selected_competition_thresholds = torch.where(
            has_positive,
            thresholds.index_select(0, homeostasis_ids),
            torch.clamp(
                thresholds.index_select(0, homeostasis_ids) * 0.995,
                min=float(threshold_min),
                max=float(threshold_max),
            ),
        )
        next_selected_thresholds = torch.clamp(
            selected_competition_thresholds
            + float(homeostasis_lr)
            * (next_selected_win_rate - float(target_firing_rate)),
            min=float(threshold_min),
            max=float(threshold_max),
        )
        next_thresholds = thresholds.index_copy(
            0,
            homeostasis_ids,
            next_selected_thresholds,
        )
    else:
        next_win_rate_ema = (
            (1.0 - float(homeostasis_beta)) * win_rate_ema
            + float(homeostasis_beta) * activity
        )
        next_thresholds = torch.clamp(
            competition_thresholds
            + float(homeostasis_lr) * (next_win_rate_ema - float(target_firing_rate)),
            min=float(threshold_min),
            max=float(threshold_max),
        )

    return (
        winners,
        strengths,
        assembly,
        next_prototypes,
        next_prototype_velocity,
        next_thresholds,
        next_win_rate_ema,
        next_steps_since_win,
        activity,
        *predictive_outputs,
        prediction_boost,
        effective_modulator,
    )


def _transition_state_tensors(
    steps_since_win: torch.Tensor,
    *,
    state_step: int = 0,
    all_materialized_step: int = 0,
    last_update: torch.Tensor | None = None,
    spike_rows: int = 4,
    active_slots: int = 1,
    active_ids: torch.Tensor | None = None,
    assembly_active_winner: int = -1,
) -> dict[str, torch.Tensor]:
    if last_update is None:
        last_update = torch.zeros_like(steps_since_win)
    if active_ids is None:
        active_ids = torch.full(
            (spike_rows, active_slots),
            -1,
            dtype=torch.long,
            device=steps_since_win.device,
        )
    return {
        "steps_since_win_last_update_step": last_update,
        "state_transition_step_counter": torch.tensor(
            state_step,
            dtype=torch.long,
            device=steps_since_win.device,
        ),
        "state_transition_all_materialized_step": torch.tensor(
            all_materialized_step,
            dtype=torch.long,
            device=steps_since_win.device,
        ),
        "recent_spike_window_active_ids": active_ids,
        "assembly_active_winner": torch.tensor(
            [assembly_active_winner],
            dtype=torch.long,
            device=steps_since_win.device,
        ),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_inplace_column_transition_all_columns_warmup_uses_direct_membership() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260618)
    n_columns = 8192
    column_dim = 8
    location_dim = 8

    prototypes = F.normalize(
        torch.rand(n_columns, column_dim, generator=generator, device=device).clamp(
            min=1e-6
        ),
        dim=1,
    )
    prototype_velocity = torch.zeros_like(prototypes)
    thresholds = torch.full((n_columns,), 0.5, device=device)
    win_rate_ema = torch.zeros(n_columns, device=device)
    steps_since_win = torch.arange(n_columns, dtype=torch.long, device=device)
    location = torch.zeros(n_columns, location_dim, device=device)
    location_velocity = torch.zeros_like(location)
    prediction_weights = torch.zeros_like(location)
    prediction_error = torch.zeros(n_columns, device=device)
    prediction_failure_streak = torch.zeros(
        n_columns,
        dtype=torch.int32,
        device=device,
    )
    confidence = torch.zeros(n_columns, device=device)
    recent_spike_window = torch.zeros(4, n_columns, device=device)
    assembly = torch.empty(n_columns, device=device)
    prediction_boost_out = torch.empty((), device=device)
    effective_modulator_out = torch.empty((), device=device)
    routing_key = F.normalize(
        torch.rand(column_dim, generator=generator, device=device).clamp(min=1e-6),
        dim=0,
    )
    previous_routing_key = torch.zeros(column_dim, device=device)
    winners = torch.tensor([0], dtype=torch.long, device=device)
    candidates = torch.arange(n_columns, dtype=torch.long, device=device)
    consolidation = torch.zeros(n_columns, device=device)
    competition_had_positive = torch.ones((), dtype=torch.bool, device=device)
    recent_spike_row = torch.zeros((), dtype=torch.int32, device=device)
    transition_state = _transition_state_tensors(
        steps_since_win,
        spike_rows=recent_spike_window.shape[0],
    )

    warmup_inplace_column_transition_cuda(
        prototypes=prototypes,
        prototype_velocity=prototype_velocity,
        thresholds=thresholds,
        win_rate_ema=win_rate_ema,
        steps_since_win=steps_since_win,
        **transition_state,
        location=location,
        location_velocity=location_velocity,
        prediction_weights=prediction_weights,
        prediction_error=prediction_error,
        prediction_failure_streak=prediction_failure_streak,
        confidence=confidence,
        recent_spike_window=recent_spike_window,
        assembly=assembly,
        prediction_boost_out=prediction_boost_out,
        effective_modulator_out=effective_modulator_out,
        routing_key=routing_key,
        previous_routing_key=previous_routing_key,
        winners=winners,
        candidates=candidates,
        consolidation=consolidation,
        competition_had_positive=competition_had_positive,
        recent_spike_row=recent_spike_row,
        prototype_momentum=0.9,
        homeostasis_beta=0.01,
        homeostasis_lr=0.01,
        target_firing_rate=1.0 / n_columns,
        threshold_min=0.05,
        threshold_max=0.95,
        prediction_error_ema_alpha=0.2,
        prediction_failure_streak_threshold=0.65,
        prediction_learning_rate=0.005,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("force_fallback", [False, True])
def test_inplace_column_transition_cuda_matches_retained_semantics(
    force_fallback: bool,
) -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260612)
    n_columns = 32
    column_dim = 16
    location_dim = 8
    candidate_count = 6

    prototypes = F.normalize(
        torch.rand(
            n_columns,
            column_dim,
            generator=generator,
            device=device,
        ).clamp(min=1e-6),
        dim=1,
    )
    prototype_velocity = torch.randn(
        n_columns,
        column_dim,
        generator=generator,
        device=device,
    ) * 0.001
    thresholds = torch.rand(
        n_columns,
        generator=generator,
        device=device,
    ) * 0.1 + 0.45
    if force_fallback:
        thresholds.fill_(2.0)
    win_rate_ema = torch.rand(
        n_columns,
        generator=generator,
        device=device,
    ) * 0.05
    steps_since_win = torch.arange(
        n_columns,
        dtype=torch.int64,
        device=device,
    )
    location = torch.randn(
        n_columns,
        location_dim,
        generator=generator,
        device=device,
    ) * 0.1
    location_velocity = torch.randn(
        n_columns,
        location_dim,
        generator=generator,
        device=device,
    ) * 0.01
    prediction_weights = torch.randn(
        n_columns,
        location_dim,
        generator=generator,
        device=device,
    ) * 0.01
    prediction_error = torch.rand(
        n_columns,
        generator=generator,
        device=device,
    ) * 0.2
    prediction_failure_streak = torch.arange(
        n_columns,
        dtype=torch.int32,
        device=device,
    ) % 4
    confidence = torch.rand(
        n_columns,
        generator=generator,
        device=device,
    )
    routing_key = F.normalize(
        torch.rand(
            column_dim,
            generator=generator,
            device=device,
        ).clamp(min=1e-6),
        dim=0,
    )
    previous_routing_key = F.normalize(
        torch.rand(
            column_dim,
            generator=generator,
            device=device,
        ).clamp(min=1e-6),
        dim=0,
    )
    candidates = torch.tensor([2, 5, 9, 13, 21, 27], device=device)
    context_gain = torch.ones(n_columns, device=device)
    consolidation = torch.rand(
        n_columns,
        generator=generator,
        device=device,
    )
    base_modulator = torch.tensor(0.3, device=device)
    dopamine = torch.tensor(0.15, device=device)
    serotonin = torch.tensor(0.2, device=device)
    competitive_learning_rate = torch.tensor(0.01, device=device)

    expected = _expected_retained_transition(
        prototypes,
        prototype_velocity,
        thresholds,
        win_rate_ema,
        steps_since_win,
        location,
        location_velocity,
        prediction_weights,
        prediction_error,
        prediction_failure_streak,
        confidence,
        routing_key,
        previous_routing_key,
        candidates,
        context_gain,
        consolidation.index_select(0, candidates),
        base_modulator,
        dopamine,
        serotonin,
        competitive_learning_rate,
        prototype_momentum=0.9,
        homeostasis_beta=0.01,
        homeostasis_lr=0.01,
        target_firing_rate=1.0 / n_columns,
        threshold_min=0.05,
        threshold_max=0.95,
        candidate_scoped_homeostasis=True,
        prediction_error_ema_alpha=0.2,
        prediction_failure_streak_threshold=0.65,
        prediction_learning_rate=0.005,
    )
    winners = expected[0]
    candidate_similarity = torch.mv(
        prototypes.index_select(0, candidates),
        routing_key,
    )
    competition_had_positive = (
        torch.relu(
            candidate_similarity - thresholds.index_select(0, candidates)
        ).max()
        > 0
    )
    assert bool(competition_had_positive.item()) is not force_fallback

    actual_state = [
        prototypes.clone(),
        prototype_velocity.clone(),
        thresholds.clone(),
        win_rate_ema.clone(),
        steps_since_win.clone(),
        location.clone(),
        location_velocity.clone(),
        prediction_weights.clone(),
        prediction_error.clone(),
        prediction_failure_streak.clone(),
        confidence.clone(),
    ]
    route_ids = torch.randperm(
        n_columns,
        generator=generator,
        device=device,
    )
    routing_vectors = actual_state[0].index_select(0, route_ids).clone()
    routing_position_by_column = torch.empty(
        n_columns,
        dtype=torch.long,
        device=device,
    )
    routing_position_by_column[route_ids] = torch.arange(
        n_columns,
        dtype=torch.long,
        device=device,
    )
    recent_spike_window = torch.zeros(4, n_columns, device=device)
    recent_spike_row = torch.tensor(2, dtype=torch.int32, device=device)
    assembly = torch.zeros(n_columns, device=device)
    prediction_boost_out = torch.empty((), device=device)
    effective_modulator_out = torch.empty((), device=device)
    reconstruction_result = torch.tensor(0.25, dtype=torch.float32, device=device)
    result_out = torch.full((9,), -7.0, dtype=torch.float32, device=device)
    result_out[0] = reconstruction_result
    neuromodulator_state = torch.tensor(
        [0.41, 0.15, 0.62, 0.73, 0.2],
        dtype=torch.float32,
        device=device,
    )
    result_ring = torch.full((2, 9), -7.0, dtype=torch.float32, device=device)
    routing_ring = torch.full(
        (2, column_dim),
        -7.0,
        dtype=torch.float32,
        device=device,
    )
    assembly_ring = torch.full(
        (2, n_columns),
        -7.0,
        dtype=torch.float32,
        device=device,
    )
    strong_flags = torch.zeros(2, dtype=torch.bool, device=device)
    strong_count = torch.zeros((), dtype=torch.int32, device=device)
    slot = torch.zeros((), dtype=torch.long, device=device)
    strong_threshold = -1.0 if force_fallback else 10.0
    transition_state = _transition_state_tensors(
        actual_state[4],
        spike_rows=recent_spike_window.shape[0],
    )

    inplace_column_transition_cuda(
        prototypes=actual_state[0],
        routing_vectors=routing_vectors,
        routing_position_by_column=routing_position_by_column,
        prototype_velocity=actual_state[1],
        thresholds=actual_state[2],
        win_rate_ema=actual_state[3],
        steps_since_win=actual_state[4],
        **transition_state,
        location=actual_state[5],
        location_velocity=actual_state[6],
        prediction_weights=actual_state[7],
        prediction_error=actual_state[8],
        prediction_failure_streak=actual_state[9],
        confidence=actual_state[10],
        recent_spike_window=recent_spike_window,
        assembly=assembly,
        prediction_boost_out=prediction_boost_out,
        effective_modulator_out=effective_modulator_out,
        result_out=result_out,
        neuromodulator_state=neuromodulator_state,
        burst_result_ring=result_ring,
        burst_routing_ring=routing_ring,
        burst_assembly_ring=assembly_ring,
        burst_strong_flags=strong_flags,
        burst_strong_count=strong_count,
        burst_slot=slot,
        strong_threshold=strong_threshold,
        routing_key=routing_key,
        previous_routing_key=previous_routing_key,
        winners=winners,
        candidates=candidates,
        consolidation=consolidation,
        base_modulator=float(base_modulator.item()),
        dopamine=float(dopamine.item()),
        serotonin=float(serotonin.item()),
        competitive_learning_rate=float(competitive_learning_rate.item()),
        recent_spike_row=recent_spike_row,
        has_previous_routing_key=True,
        competition_had_positive=competition_had_positive,
        prototype_momentum=0.9,
        homeostasis_beta=0.01,
        homeostasis_lr=0.01,
        target_firing_rate=1.0 / n_columns,
        threshold_min=0.05,
        threshold_max=0.95,
        prediction_error_ema_alpha=0.2,
        prediction_failure_streak_threshold=0.65,
        prediction_learning_rate=0.005,
    )
    torch.cuda.synchronize()

    expected_state = [
        expected[3],
        expected[4],
        expected[5],
        expected[6],
        expected[7],
        expected[9],
        expected[10],
        expected[11],
        expected[12],
        expected[13],
        expected[14],
    ]
    for actual, reference in zip(actual_state, expected_state):
        if actual.dtype.is_floating_point:
            assert torch.allclose(actual, reference, atol=2e-5, rtol=2e-5)
        else:
            assert torch.equal(actual, reference)
    assert torch.allclose(assembly, expected[2], atol=2e-5, rtol=2e-5)
    assert torch.equal(recent_spike_window[2], expected[8])
    assert prediction_boost_out.item() == pytest.approx(
        expected[15].item(),
        abs=2e-5,
    )
    assert effective_modulator_out.item() == pytest.approx(
        expected[16].item(),
        abs=2e-5,
    )
    expected_competitive_surprise = torch.norm(
        actual_state[0].index_select(0, winners).squeeze(0) - routing_key
    )
    expected_result = torch.stack(
        (
            reconstruction_result,
            neuromodulator_state[0],
            neuromodulator_state[1],
            neuromodulator_state[2],
            neuromodulator_state[3],
            neuromodulator_state[4],
            winners[0].to(torch.float32),
            effective_modulator_out,
            expected_competitive_surprise,
        )
    )
    assert torch.allclose(result_out, expected_result, atol=2e-5, rtol=2e-5)
    assert torch.allclose(result_ring[0], expected_result, atol=2e-5, rtol=2e-5)
    assert int(slot.item()) == 1
    if force_fallback:
        assert bool(strong_flags[0].item()) is True
        assert int(strong_count.item()) == 1
        assert torch.allclose(routing_ring[0], routing_key, atol=2e-5, rtol=2e-5)
        assert torch.allclose(assembly_ring[0], assembly, atol=2e-5, rtol=2e-5)
    else:
        assert bool(strong_flags[0].item()) is False
        assert int(strong_count.item()) == 0
        assert torch.equal(routing_ring[0], torch.full_like(routing_ring[0], -7.0))
        assert torch.equal(assembly_ring[0], torch.full_like(assembly_ring[0], -7.0))
    assert torch.allclose(
        routing_vectors.index_select(0, routing_position_by_column),
        actual_state[0],
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_inplace_column_transition_candidate_predictive_branch_keeps_cached_rows() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260617)
    n_columns = 32
    column_dim = 16
    location_dim = 8
    prototypes = F.normalize(
        torch.rand(n_columns, column_dim, generator=generator, device=device).clamp(min=1e-6),
        dim=1,
    )
    prototype_velocity = torch.randn(n_columns, column_dim, generator=generator, device=device) * 0.001
    thresholds = torch.rand(n_columns, generator=generator, device=device) * 0.1 + 0.25
    win_rate_ema = torch.rand(n_columns, generator=generator, device=device) * 0.05
    steps_since_win = torch.arange(n_columns, dtype=torch.int64, device=device)
    location = torch.randn(n_columns, location_dim, generator=generator, device=device) * 0.1
    location_velocity = torch.randn(n_columns, location_dim, generator=generator, device=device) * 0.01
    prediction_weights = torch.randn(n_columns, location_dim, generator=generator, device=device) * 0.01
    prediction_error = torch.rand(n_columns, generator=generator, device=device) * 0.2
    prediction_failure_streak = torch.arange(n_columns, dtype=torch.int32, device=device) % 4
    confidence = torch.rand(n_columns, generator=generator, device=device)
    routing_key = F.normalize(
        torch.rand(column_dim, generator=generator, device=device).clamp(min=1e-6),
        dim=0,
    )
    previous_routing_key = F.normalize(
        torch.rand(column_dim, generator=generator, device=device).clamp(min=1e-6),
        dim=0,
    )
    candidates = torch.tensor([2, 5, 9, 13, 21, 27], dtype=torch.long, device=device)
    context_gain = torch.ones(n_columns, device=device)
    consolidation = torch.rand(n_columns, generator=generator, device=device)
    base_modulator = torch.tensor(0.3, device=device)
    dopamine = torch.tensor(0.15, device=device)
    serotonin = torch.tensor(0.2, device=device)
    competitive_learning_rate = torch.tensor(0.01, device=device)
    expected = _expected_retained_transition(
        prototypes,
        prototype_velocity,
        thresholds,
        win_rate_ema,
        steps_since_win,
        location,
        location_velocity,
        prediction_weights,
        prediction_error,
        prediction_failure_streak,
        confidence,
        routing_key,
        previous_routing_key,
        candidates,
        context_gain,
        consolidation.index_select(0, candidates),
        base_modulator,
        dopamine,
        serotonin,
        competitive_learning_rate,
        prototype_momentum=0.9,
        homeostasis_beta=0.01,
        homeostasis_lr=0.01,
        target_firing_rate=1.0 / n_columns,
        threshold_min=0.05,
        threshold_max=0.95,
        candidate_scoped_homeostasis=True,
        prediction_error_ema_alpha=0.2,
        prediction_failure_streak_threshold=0.65,
        prediction_learning_rate=0.005,
    )
    winners = expected[0]
    candidate_similarity = torch.mv(prototypes.index_select(0, candidates), routing_key)
    competition_had_positive = (
        torch.relu(candidate_similarity - thresholds.index_select(0, candidates)).max()
        > 0
    )
    actual_state = [
        prototypes.clone(),
        prototype_velocity.clone(),
        thresholds.clone(),
        win_rate_ema.clone(),
        steps_since_win.clone(),
        location.clone(),
        location_velocity.clone(),
        prediction_weights.clone(),
        prediction_error.clone(),
        prediction_failure_streak.clone(),
        confidence.clone(),
    ]
    original_predictive = [tensor.clone() for tensor in actual_state[5:]]
    recent_spike_window = torch.zeros(4, n_columns, device=device)
    recent_spike_row = torch.tensor(2, dtype=torch.int32, device=device)
    assembly = torch.zeros(n_columns, device=device)
    prediction_boost_out = torch.empty((), device=device)
    effective_modulator_out = torch.empty((), device=device)
    predictive_last_update_step = torch.zeros(n_columns, dtype=torch.long, device=device)
    predictive_step_counter = torch.tensor(7, dtype=torch.long, device=device)
    transition_state = _transition_state_tensors(
        actual_state[4],
        spike_rows=recent_spike_window.shape[0],
    )

    inplace_column_transition_cuda(
        prototypes=actual_state[0],
        prototype_velocity=actual_state[1],
        thresholds=actual_state[2],
        win_rate_ema=actual_state[3],
        steps_since_win=actual_state[4],
        **transition_state,
        location=actual_state[5],
        location_velocity=actual_state[6],
        prediction_weights=actual_state[7],
        prediction_error=actual_state[8],
        prediction_failure_streak=actual_state[9],
        confidence=actual_state[10],
        recent_spike_window=recent_spike_window,
        assembly=assembly,
        prediction_boost_out=prediction_boost_out,
        effective_modulator_out=effective_modulator_out,
        routing_key=routing_key,
        previous_routing_key=previous_routing_key,
        winners=winners,
        candidates=candidates,
        consolidation=consolidation,
        predictive_candidates=candidates,
        predictive_last_update_step=predictive_last_update_step,
        predictive_step_counter=predictive_step_counter,
        base_modulator=float(base_modulator.item()),
        dopamine=float(dopamine.item()),
        serotonin=float(serotonin.item()),
        competitive_learning_rate=float(competitive_learning_rate.item()),
        recent_spike_row=recent_spike_row,
        has_previous_routing_key=True,
        competition_had_positive=competition_had_positive,
        prototype_momentum=0.9,
        homeostasis_beta=0.01,
        homeostasis_lr=0.01,
        target_firing_rate=1.0 / n_columns,
        threshold_min=0.05,
        threshold_max=0.95,
        prediction_error_ema_alpha=0.2,
        prediction_failure_streak_threshold=0.65,
        prediction_learning_rate=0.005,
    )
    torch.cuda.synchronize()

    expected_competitive = [expected[3], expected[4], expected[5], expected[6], expected[7]]
    for actual, reference in zip(actual_state[:5], expected_competitive):
        if actual.dtype.is_floating_point:
            assert torch.allclose(actual, reference, atol=2e-5, rtol=2e-5)
        else:
            assert torch.equal(actual, reference)
    expected_predictive = [expected[9], expected[10], expected[11], expected[12], expected[13], expected[14]]
    for actual, reference in zip(actual_state[5:], expected_predictive):
        if actual.dtype.is_floating_point:
            assert torch.allclose(
                actual.index_select(0, candidates),
                reference.index_select(0, candidates),
                atol=2e-5,
                rtol=2e-5,
            )
        else:
            assert torch.equal(
                actual.index_select(0, candidates),
                reference.index_select(0, candidates),
            )
    candidate_mask = torch.zeros(n_columns, dtype=torch.bool, device=device)
    candidate_mask[candidates] = True
    non_candidates = torch.arange(n_columns, device=device)[~candidate_mask]
    for actual, original in zip(actual_state[5:], original_predictive):
        if actual.dtype.is_floating_point:
            assert torch.allclose(
                actual.index_select(0, non_candidates),
                original.index_select(0, non_candidates),
            )
        else:
            assert torch.equal(
                actual.index_select(0, non_candidates),
                original.index_select(0, non_candidates),
            )
    assert torch.equal(
        predictive_last_update_step.index_select(0, candidates),
        torch.full_like(candidates, 8),
    )
    assert int(predictive_last_update_step.index_select(0, non_candidates).sum().item()) == 0
    assert int(predictive_step_counter.item()) == 8
    assert torch.allclose(assembly, expected[2], atol=2e-5, rtol=2e-5)
    assert prediction_boost_out.item() == pytest.approx(expected[15].item(), abs=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_inplace_column_transition_sparse_state_updates_only_awake_rows() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260619)
    n_columns = 16
    column_dim = 8
    location_dim = 4
    candidates = torch.tensor([1, 4, 7, 10], dtype=torch.long, device=device)
    winner = candidates[:1]
    prototypes = F.normalize(
        torch.rand(n_columns, column_dim, generator=generator, device=device).clamp(
            min=1e-6
        ),
        dim=1,
    )
    prototype_velocity = torch.zeros_like(prototypes)
    thresholds = torch.full((n_columns,), 0.25, device=device)
    win_rate_ema = torch.zeros(n_columns, device=device)
    steps_since_win = torch.arange(n_columns, dtype=torch.long, device=device)
    original_steps_since_win = steps_since_win.clone()
    last_update = torch.arange(n_columns, dtype=torch.long, device=device) % 5
    original_last_update = last_update.clone()
    state_step = 7
    all_materialized_step = 3
    location = torch.zeros(n_columns, location_dim, device=device)
    location_velocity = torch.zeros_like(location)
    prediction_weights = torch.zeros_like(location)
    prediction_error = torch.zeros(n_columns, device=device)
    prediction_failure_streak = torch.zeros(
        n_columns,
        dtype=torch.int32,
        device=device,
    )
    confidence = torch.zeros(n_columns, device=device)
    recent_spike_window = torch.zeros(4, n_columns, device=device)
    recent_spike_row = torch.tensor(2, dtype=torch.int32, device=device)
    previous_active = torch.tensor(12, dtype=torch.long, device=device)
    recent_spike_window[2, previous_active] = 1.0
    active_ids = torch.full((4, 2), -1, dtype=torch.long, device=device)
    active_ids[2, 0] = previous_active
    assembly = torch.zeros(n_columns, device=device)
    previous_assembly_winner = 8
    assembly[previous_assembly_winner] = 0.75
    routing_key = F.normalize(
        torch.rand(column_dim, generator=generator, device=device).clamp(min=1e-6),
        dim=0,
    )
    previous_routing_key = torch.zeros(column_dim, device=device)
    prediction_boost_out = torch.empty((), device=device)
    effective_modulator_out = torch.empty((), device=device)
    competition_had_positive = torch.tensor(True, dtype=torch.bool, device=device)
    transition_state = _transition_state_tensors(
        steps_since_win,
        state_step=state_step,
        all_materialized_step=all_materialized_step,
        last_update=last_update,
        spike_rows=recent_spike_window.shape[0],
        active_slots=active_ids.shape[1],
        active_ids=active_ids,
        assembly_active_winner=previous_assembly_winner,
    )

    inplace_column_transition_cuda(
        prototypes=prototypes,
        prototype_velocity=prototype_velocity,
        thresholds=thresholds,
        win_rate_ema=win_rate_ema,
        steps_since_win=steps_since_win,
        **transition_state,
        location=location,
        location_velocity=location_velocity,
        prediction_weights=prediction_weights,
        prediction_error=prediction_error,
        prediction_failure_streak=prediction_failure_streak,
        confidence=confidence,
        recent_spike_window=recent_spike_window,
        assembly=assembly,
        prediction_boost_out=prediction_boost_out,
        effective_modulator_out=effective_modulator_out,
        routing_key=routing_key,
        previous_routing_key=previous_routing_key,
        winners=winner,
        candidates=candidates,
        consolidation=torch.zeros(n_columns, device=device),
        base_modulator=0.3,
        dopamine=0.0,
        serotonin=0.0,
        competitive_learning_rate=0.01,
        recent_spike_row=recent_spike_row,
        has_previous_routing_key=False,
        competition_had_positive=competition_had_positive,
        prototype_momentum=0.9,
        homeostasis_beta=0.01,
        homeostasis_lr=0.01,
        target_firing_rate=1.0 / n_columns,
        threshold_min=0.05,
        threshold_max=0.95,
        prediction_error_ema_alpha=0.2,
        prediction_failure_streak_threshold=0.65,
        prediction_learning_rate=0.005,
    )
    torch.cuda.synchronize()

    candidate_effective_last_update = torch.maximum(
        original_last_update.index_select(0, candidates),
        torch.full_like(candidates, all_materialized_step),
    )
    expected_candidate_steps = original_steps_since_win.index_select(0, candidates)
    expected_candidate_steps += torch.clamp(
        torch.full_like(candidates, state_step) - candidate_effective_last_update,
        min=0,
    )
    expected_candidate_steps = torch.where(
        candidates == winner[0],
        torch.zeros_like(expected_candidate_steps),
        expected_candidate_steps + 1,
    )
    assert torch.equal(
        steps_since_win.index_select(0, candidates),
        expected_candidate_steps,
    )
    candidate_mask = torch.zeros(n_columns, dtype=torch.bool, device=device)
    candidate_mask[candidates] = True
    non_candidates = torch.arange(n_columns, device=device)[~candidate_mask]
    assert torch.equal(
        steps_since_win.index_select(0, non_candidates),
        original_steps_since_win.index_select(0, non_candidates),
    )
    assert torch.equal(
        last_update.index_select(0, candidates),
        torch.full_like(candidates, state_step + 1),
    )
    assert torch.equal(
        last_update.index_select(0, non_candidates),
        original_last_update.index_select(0, non_candidates),
    )
    assert int(transition_state["state_transition_step_counter"].item()) == state_step + 1
    assert (
        int(transition_state["state_transition_all_materialized_step"].item())
        == all_materialized_step
    )
    assert float(recent_spike_window[2, previous_active].item()) == pytest.approx(0.0)
    assert float(recent_spike_window[2, winner[0]].item()) == pytest.approx(1.0)
    assert int(active_ids[2, 0].item()) == int(winner[0].item())
    assert int(active_ids[2, 1].item()) == -1
    assert float(assembly[previous_assembly_winner].item()) == pytest.approx(0.0)
    assert float(assembly[winner[0]].item()) > 0.0
    assert int(transition_state["assembly_active_winner"].item()) == int(winner[0].item())
