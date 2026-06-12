from __future__ import annotations

import torch

from marulho.core.column_transition import steady_state_column_transition
from marulho.core.columns import CompetitiveColumnLayer
from marulho.core.predictive_columns import PredictiveColumnState


def test_steady_state_column_transition_matches_eager_modules() -> None:
    torch.manual_seed(19)
    layer = CompetitiveColumnLayer(
        n_columns=8,
        column_dim=4,
        input_dim=16,
        device=torch.device("cpu"),
        k_routing=3,
        n_winners=1,
        lr_initial=0.01,
        lr_decay=0.0,
        input_weight_blend=0.0,
        plasticity_mode="lite",
        homeostasis_beta=0.1,
        homeostasis_lr=0.2,
        prototype_momentum=0.85,
    )
    predictive = PredictiveColumnState(
        n_columns=8,
        location_dim=4,
        device=torch.device("cpu"),
    )
    routing_key = torch.rand(4)
    previous_routing_key = torch.rand(4)
    candidates = torch.tensor([1, 3, 6])
    context_gain = torch.linspace(0.8, 1.2, 8)
    competitive_modulator = 0.37
    dopamine = 0.6
    serotonin = 0.25
    candidate_consolidation = torch.tensor([0.1, 0.4, 0.2])

    outputs = steady_state_column_transition(
        layer.prototypes.clone(),
        layer.prototype_velocity.clone(),
        layer.thresholds.clone(),
        layer.win_rate_ema.clone(),
        layer.steps_since_win.clone(),
        predictive.location.clone(),
        predictive.velocity.clone(),
        predictive._prediction_weights.clone(),
        predictive.prediction_error.clone(),
        predictive.prediction_failure_streak.clone(),
        predictive.confidence.clone(),
        routing_key,
        previous_routing_key,
        candidates,
        context_gain,
        candidate_consolidation,
        torch.tensor(competitive_modulator),
        torch.tensor(dopamine),
        torch.tensor(serotonin),
        torch.tensor(layer.get_lr()),
        prototype_momentum=layer.prototype_momentum,
        homeostasis_beta=layer.homeostasis_beta,
        homeostasis_lr=layer.homeostasis_lr,
        target_firing_rate=layer.target_firing_rate,
        threshold_min=layer.threshold_min,
        threshold_max=layer.threshold_max,
        candidate_scoped_homeostasis=True,
        prediction_error_ema_alpha=predictive._error_ema_alpha,
        prediction_failure_streak_threshold=predictive._failure_streak_threshold,
        prediction_learning_rate=0.005,
    )

    winners, strengths, _ = layer.compete(
        routing_key,
        candidates,
        context_gain=context_gain,
    )
    pred_error = predictive.apply_dense_transition(
        winners,
        routing_key,
        previous_routing_key,
        transition_mode="fused_eager",
    )
    pred_boost = torch.clamp(pred_error[winners].mean(), min=0.5, max=2.0)
    winner_local = int((candidates == winners[0]).nonzero()[0].item())
    winner_consolidation = float(candidate_consolidation[winner_local].item())
    da_ltp_gain = 0.8 + 0.4 * dopamine
    ht_patience = max(0.2, 1.0 - 0.6 * serotonin)
    wake_plasticity_scale = max(0.2, 1.0 - 0.8 * winner_consolidation) * ht_patience
    effective_modulator = (
        competitive_modulator
        * wake_plasticity_scale
        * da_ltp_gain
        * float(pred_boost.item())
    )
    assembly = layer.process(
        routing_key,
        winners,
        effective_modulator,
        winner_strengths=strengths,
        homeostasis_update_indices=candidates,
    )

    assert torch.equal(outputs[0], winners)
    assert torch.allclose(outputs[1], strengths, atol=1e-6, rtol=1e-6)
    assert torch.allclose(outputs[2], assembly, atol=1e-6, rtol=1e-6)
    assert torch.allclose(outputs[3], layer.prototypes, atol=1e-6, rtol=1e-6)
    assert torch.allclose(outputs[4], layer.prototype_velocity, atol=1e-6, rtol=1e-6)
    assert torch.allclose(outputs[5], layer.thresholds, atol=1e-6, rtol=1e-6)
    assert torch.allclose(outputs[6], layer.win_rate_ema, atol=1e-6, rtol=1e-6)
    assert torch.equal(outputs[7], layer.steps_since_win)
    assert torch.allclose(outputs[9], predictive.location, atol=1e-6, rtol=1e-6)
    assert torch.allclose(outputs[10], predictive.velocity, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        outputs[11],
        predictive._prediction_weights,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(outputs[12], predictive.prediction_error, atol=1e-6, rtol=1e-6)
    assert torch.equal(outputs[13], predictive.prediction_failure_streak)
    assert torch.allclose(outputs[14], predictive.confidence, atol=1e-6, rtol=1e-6)
    assert torch.allclose(outputs[15], pred_boost, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        outputs[16],
        torch.tensor(effective_modulator),
        atol=1e-6,
        rtol=1e-6,
    )
