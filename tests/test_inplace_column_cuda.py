from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from marulho.core.column_transition import steady_state_column_transition
from marulho.core.inplace_column_cuda import inplace_column_transition_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("force_fallback", [False, True])
def test_inplace_column_transition_cuda_matches_functional_oracle(
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

    expected = steady_state_column_transition(
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
    recent_spike_window = torch.zeros(4, n_columns, device=device)
    recent_spike_row = torch.tensor(2, dtype=torch.int32, device=device)
    assembly = torch.empty(n_columns, device=device)
    prediction_boost_out = torch.empty((), device=device)
    effective_modulator_out = torch.empty((), device=device)

    inplace_column_transition_cuda(
        prototypes=actual_state[0],
        prototype_velocity=actual_state[1],
        thresholds=actual_state[2],
        win_rate_ema=actual_state[3],
        steps_since_win=actual_state[4],
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
