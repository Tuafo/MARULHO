from __future__ import annotations

import torch

from marulho.core.predictive_columns import dense_predictive_transition


def steady_state_column_transition(
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
) -> tuple[torch.Tensor, ...]:
    """One fixed-shape competition, prediction, and plasticity transition.

    This transition intentionally targets the promoted CUDA steady-state shape:
    one winner, lightweight plasticity, zero input-weight blend, and no local
    STDP. Control-plane counters and the recent-spike ring cursor remain outside.
    """

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
    activation = torch.relu(
        combined - thresholds.index_select(0, candidate_ids)
    )
    top_value, top_local_index = torch.topk(activation, k=1)
    best_combined_local = torch.argmax(combined).reshape(1)
    has_positive = top_value.max() > 0
    winner_local = torch.where(
        has_positive,
        top_local_index,
        best_combined_local,
    )
    winners = candidate_ids.index_select(0, winner_local)
    winner_consolidation = candidate_consolidation.index_select(
        0,
        winner_local,
    ).mean()
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
    next_prototypes = prototypes.index_copy(
        0,
        winners,
        next_winner_prototypes,
    )
    next_prototype_velocity = prototype_velocity.index_copy(
        0,
        winners,
        next_winner_velocity,
    )

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
        next_selected_thresholds = torch.clamp(
            competition_thresholds.index_select(0, homeostasis_ids)
            + float(homeostasis_lr)
            * (next_selected_win_rate - float(target_firing_rate)),
            min=float(threshold_min),
            max=float(threshold_max),
        )
        next_thresholds = competition_thresholds.index_copy(
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
            + float(homeostasis_lr)
            * (next_win_rate_ema - float(target_firing_rate)),
            min=float(threshold_min),
            max=float(threshold_max),
        )

    spike_sample = activity
    return (
        winners,
        strengths,
        assembly,
        next_prototypes,
        next_prototype_velocity,
        next_thresholds,
        next_win_rate_ema,
        next_steps_since_win,
        spike_sample,
        *predictive_outputs,
        prediction_boost,
        effective_modulator,
    )
