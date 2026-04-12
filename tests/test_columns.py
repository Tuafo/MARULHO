"""Tests for CompetitiveColumnLayer, focusing on state_dict roundtrip."""

from __future__ import annotations

import torch
import pytest

from hecsn.core.columns import CompetitiveColumnLayer


def _make_layer(**overrides) -> CompetitiveColumnLayer:
    defaults = dict(
        n_columns=16,
        column_dim=8,
        input_dim=32,
        device=torch.device("cpu"),
    )
    defaults.update(overrides)
    return CompetitiveColumnLayer(**defaults)


class TestStateDict:
    """Verify state_dict / load_state_dict roundtrip fidelity."""

    def test_roundtrip_preserves_prototypes(self):
        layer = _make_layer()
        layer.prototypes += torch.randn_like(layer.prototypes) * 0.1
        snap = layer.state_dict()

        layer2 = _make_layer()
        layer2.load_state_dict(snap)
        assert torch.allclose(layer.prototypes, layer2.prototypes)

    def test_roundtrip_preserves_weights(self):
        layer = _make_layer()
        layer.W_project += 0.01
        layer.input_weights *= 1.1
        snap = layer.state_dict()

        layer2 = _make_layer()
        layer2.load_state_dict(snap)
        assert torch.allclose(layer.W_project, layer2.W_project)
        assert torch.allclose(layer.input_weights, layer2.input_weights)

    def test_roundtrip_preserves_homeostasis(self):
        layer = _make_layer()
        layer.thresholds.fill_(0.42)
        layer.win_rate_ema.fill_(0.03)
        layer.steps_since_win[0] = 999
        snap = layer.state_dict()

        layer2 = _make_layer()
        layer2.load_state_dict(snap)
        assert torch.allclose(layer.thresholds, layer2.thresholds)
        assert torch.allclose(layer.win_rate_ema, layer2.win_rate_ema)
        assert layer2.steps_since_win[0].item() == 999

    def test_roundtrip_preserves_scalars(self):
        layer = _make_layer()
        layer.update_count = 42
        layer.target_firing_rate = 0.123
        snap = layer.state_dict()

        layer2 = _make_layer()
        layer2.load_state_dict(snap)
        assert layer2.update_count == 42
        assert abs(layer2.target_firing_rate - 0.123) < 1e-6

    def test_roundtrip_preserves_velocity(self):
        layer = _make_layer()
        layer.prototype_velocity += 0.5
        snap = layer.state_dict()

        layer2 = _make_layer()
        layer2.load_state_dict(snap)
        assert torch.allclose(layer.prototype_velocity, layer2.prototype_velocity)

    def test_roundtrip_handles_none_optional_tensors(self):
        layer = _make_layer()
        assert layer.last_input_pattern is None
        snap = layer.state_dict()
        assert snap["last_input_pattern"] is None

        layer2 = _make_layer()
        layer2.load_state_dict(snap)
        assert layer2.last_input_pattern is None

    def test_roundtrip_preserves_optional_tensors(self):
        layer = _make_layer()
        layer.last_input_pattern = torch.randn(32)
        layer.last_projected_input = torch.randn(8)
        snap = layer.state_dict()

        layer2 = _make_layer()
        layer2.load_state_dict(snap)
        assert torch.allclose(layer.last_input_pattern, layer2.last_input_pattern)
        assert torch.allclose(layer.last_projected_input, layer2.last_projected_input)

    def test_roundtrip_with_local_stdp(self):
        layer = _make_layer(plasticity_mode="local_stdp")
        assert layer.local_plasticity is not None
        snap = layer.state_dict()
        assert snap["local_plasticity"] is not None

        layer2 = _make_layer(plasticity_mode="local_stdp")
        layer2.load_state_dict(snap)

    def test_state_dict_tensors_on_cpu(self):
        layer = _make_layer()
        snap = layer.state_dict()
        for key, val in snap.items():
            if isinstance(val, torch.Tensor):
                assert val.device == torch.device("cpu"), f"{key} not on CPU"

    def test_compete_after_roundtrip_matches(self):
        """Loaded layer produces identical competition results."""
        layer = _make_layer()
        routing_key = torch.rand(8)  # column_dim
        candidates = torch.arange(16)
        result_before = layer.compete(routing_key, candidate_indices=candidates)

        snap = layer.state_dict()
        layer2 = _make_layer()
        layer2.load_state_dict(snap)
        result_after = layer2.compete(routing_key, candidate_indices=candidates)

        assert torch.equal(result_before[0], result_after[0])
