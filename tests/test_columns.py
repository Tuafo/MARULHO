"""Tests for CompetitiveColumnLayer, focusing on state_dict roundtrip."""

from __future__ import annotations

import torch
import pytest

from marulho.core.columns import CompetitiveColumnLayer


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

    def test_device_report_exposes_live_tensor_devices(self):
        layer = _make_layer(plasticity_mode="local_stdp")
        assert layer.local_plasticity is not None

        report = layer.device_report()

        assert report["device"] == "cpu"
        assert report["W_project_device"] == str(layer.W_project.device)
        assert report["input_weights_device"] == str(layer.input_weights.device)
        assert report["prototypes_device"] == str(layer.prototypes.device)
        assert report["prototype_velocity_device"] == str(layer.prototype_velocity.device)
        assert report["thresholds_device"] == str(layer.thresholds.device)
        assert report["recent_spike_window_device"] == str(layer.recent_spike_window.device)
        assert report["local_plasticity"] is not None
        assert report["local_plasticity"]["pre_trace_device"] == str(layer.local_plasticity.pre_trace.device)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
    def test_cuda_device_report_exposes_live_tensor_devices(self):
        layer = _make_layer(
            device=torch.device("cuda"),
            plasticity_mode="local_stdp",
        )
        assert layer.local_plasticity is not None

        report = layer.device_report()

        assert report["device"].startswith("cuda")
        assert str(report["prototypes_device"]).startswith("cuda")
        assert str(report["W_project_device"]).startswith("cuda")
        assert str(report["local_plasticity"]["pre_trace_device"]).startswith("cuda")

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

    def test_roundtrip_preserves_spike_history_window(self):
        layer = _make_layer()
        layer.recent_spike_window.zero_()
        layer.recent_spike_window[0, 0] = 1.0
        layer.recent_spike_window[1, 1] = 1.0
        layer.recent_spike_window_cursor = 2
        layer.recent_spike_window_count = 2
        snap = layer.state_dict()

        layer2 = _make_layer()
        layer2.load_state_dict(snap)

        assert torch.allclose(layer.recent_spike_window, layer2.recent_spike_window)
        assert layer2.recent_spike_window_cursor == 2
        assert layer2.recent_spike_window_count == 2

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

    def test_process_records_bounded_recent_spike_window(self):
        layer = _make_layer(n_columns=4, column_dim=4, input_dim=8)
        layer.last_input_pattern = torch.full((8,), 1.0 / 8.0)
        routing_key = torch.tensor([1.0, 0.2, 0.1, 0.1])

        layer.process(routing_key, torch.tensor([0]), modulator=0.5)
        layer.process(routing_key, torch.tensor([1]), modulator=0.5)

        report = layer.spike_health_report()

        assert report["correlation_evidence_available"] is False
        assert report["correlation"]["status"] == "insufficient_window"
        assert report["correlation"]["sample_count"] == 2
        assert layer.recent_spike_window_count == 2

    def test_process_without_global_state_does_not_record_spike_window(self):
        layer = _make_layer(n_columns=4, column_dim=4, input_dim=8)
        layer.last_input_pattern = torch.full((8,), 1.0 / 8.0)
        routing_key = torch.tensor([1.0, 0.2, 0.1, 0.1])

        layer.process(
            routing_key,
            torch.tensor([0]),
            modulator=0.5,
            update_global_state=False,
        )

        assert layer.recent_spike_window_count == 0

    def test_spike_health_report_is_read_only_for_spike_window(self):
        layer = _make_layer(n_columns=4, column_dim=4, input_dim=8)
        layer.last_input_pattern = torch.full((8,), 1.0 / 8.0)
        routing_key = torch.tensor([1.0, 0.2, 0.1, 0.1])
        layer.process(routing_key, torch.tensor([0]), modulator=0.5)
        before_count = layer.recent_spike_window_count
        before_window = layer.recent_spike_window.clone()

        layer.spike_health_report()
        layer.spike_health_report()

        assert layer.recent_spike_window_count == before_count
        assert torch.allclose(layer.recent_spike_window, before_window)

    def test_spike_health_reports_windowed_correlation_when_enough_samples(self):
        layer = _make_layer(n_columns=4, column_dim=4, input_dim=8)
        samples = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        layer.recent_spike_window[:4] = samples
        layer.recent_spike_window_cursor = 4
        layer.recent_spike_window_count = 4

        report = layer.spike_health_report()

        assert report["correlation_evidence_available"] is True
        assert report["correlation"]["status"] == "overcorrelated_risk"
        assert report["correlation"]["active_columns"] >= 2
        assert report["correlation"]["mean_abs_offdiag_correlation"] is not None
        assert report["missing_evidence"] == []

    def test_spike_window_is_bounded(self):
        layer = _make_layer(n_columns=4, column_dim=4, input_dim=8)
        layer.last_input_pattern = torch.full((8,), 1.0 / 8.0)
        routing_key = torch.tensor([1.0, 0.2, 0.1, 0.1])

        for idx in range(layer.spike_history_window + 5):
            layer.process(
                routing_key,
                torch.tensor([idx % layer.n_columns]),
                modulator=0.5,
            )

        assert layer.recent_spike_window_count == layer.spike_history_window
        assert 0 <= layer.recent_spike_window_cursor < layer.spike_history_window
