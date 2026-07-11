from __future__ import annotations

import torch
from torch import nn

from marulho.evaluation.language_counterfactual_utility_gate import (
    PROMOTE_DECISION,
    _gate_metrics,
    utility_gate_decision,
)


class _FixedGate(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, :2]


def test_utility_gate_decision_requires_every_source_and_noncollapse() -> None:
    assert utility_gate_decision(
        (0.03, 0.04),
        alternative_selection_fraction=0.20,
        parameters_unchanged=True,
    ) == PROMOTE_DECISION
    assert utility_gate_decision(
        (0.03, 0.01),
        alternative_selection_fraction=0.20,
        parameters_unchanged=True,
    ) == "redesign_v12_gate_not_general_across_sources"
    assert utility_gate_decision(
        (0.03, 0.04),
        alternative_selection_fraction=0.01,
        parameters_unchanged=True,
    ) == "redesign_v12_gate_not_general_across_sources"
    assert utility_gate_decision(
        (0.03, 0.04),
        alternative_selection_fraction=0.20,
        parameters_unchanged=False,
    ) == "invalid_v12_gate_training_mutated_parent"


def test_gate_metrics_select_routes_without_reading_candidate_losses() -> None:
    features = torch.tensor(
        [
            [0.2, -0.1],
            [-0.2, 0.3],
            [-0.1, -0.2],
        ],
        dtype=torch.float32,
    )
    candidate_losses = torch.tensor(
        [
            [2.0, 1.5, 2.5],
            [2.0, 2.4, 1.6],
            [2.0, 1.0, 1.2],
        ],
        dtype=torch.float32,
    )
    report = _gate_metrics(
        _FixedGate(),
        features,
        candidate_losses,
        threshold=0.0,
        device=torch.device("cpu"),
    )
    assert report["route_selection_counts"] == [1, 1, 1]
    assert report["realized_mean_loss"] == torch.tensor(
        [1.5, 1.6, 2.0]
    ).mean().item()
    assert report["prediction_uses_targets"] is False
    assert report["targets_metrics_only"] is True
