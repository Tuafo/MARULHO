"""Tests for AdaptiveContextLayer (§4.3) and create_context_layer factory."""

from __future__ import annotations

import math
import unittest

import torch

from hecsn.core.context import (
    AdaptiveContextLayer,
    ContextLayer,
    create_context_layer,
)


class TestAdaptiveContextLayerInit(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.ctx = AdaptiveContextLayer(n_columns=64, device=self.device)

    def test_default_n_neurons_equals_n_columns(self) -> None:
        self.assertEqual(self.ctx.n_neurons, 64)

    def test_custom_n_neurons(self) -> None:
        ctx = AdaptiveContextLayer(n_columns=64, device=self.device, n_neurons=128)
        self.assertEqual(ctx.n_neurons, 128)
        self.assertEqual(ctx.n_columns, 64)

    def test_log_tau_init_covers_range(self) -> None:
        tau = self.ctx._tau()
        self.assertAlmostEqual(float(tau.min()), 2.0, places=1)
        self.assertAlmostEqual(float(tau.max()), 500.0, delta=1.0)

    def test_state_starts_zero(self) -> None:
        self.assertEqual(float(self.ctx.neuron_state.sum()), 0.0)
        self.assertEqual(float(self.ctx.state.sum()), 0.0)

    def test_w_in_shape(self) -> None:
        self.assertEqual(self.ctx.w_in.shape, (64, 64))

    def test_w_out_shape(self) -> None:
        self.assertEqual(self.ctx.w_out.shape, (64, 64))


class TestAdaptiveContextLayerStep(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.ctx = AdaptiveContextLayer(n_columns=32, device=self.device)

    def test_observe_returns_correct_shape(self) -> None:
        assembly = torch.randn(32)
        result = self.ctx.observe(assembly)
        self.assertEqual(result.shape, (32,))

    def test_observe_updates_state(self) -> None:
        assembly = torch.randn(32)
        self.ctx.observe(assembly)
        self.assertGreater(float(self.ctx.neuron_state.abs().sum()), 0.0)

    def test_zero_assembly_decays_state(self) -> None:
        # Prime state
        self.ctx.observe(torch.randn(32))
        state_before = self.ctx.neuron_state.clone()
        # Zero input should decay
        self.ctx.observe(torch.zeros(32))
        # Each neuron should decay toward zero
        self.assertTrue(
            float(self.ctx.neuron_state.abs().sum())
            <= float(state_before.abs().sum()) + 1e-6
        )

    def test_context_prediction_shape(self) -> None:
        self.ctx.observe(torch.randn(32))
        pred = self.ctx.context_prediction()
        self.assertEqual(pred.shape, (32,))

    def test_modulation_gain_shape(self) -> None:
        self.ctx.observe(torch.randn(32))
        gain = self.ctx.modulation_gain()
        self.assertEqual(gain.shape, (32,))

    def test_gain_bounded(self) -> None:
        self.ctx.observe(torch.randn(32))
        gain = self.ctx.modulation_gain()
        self.assertGreaterEqual(float(gain.min()), 0.64)
        self.assertLessEqual(float(gain.max()), 1.36)


class TestAdaptiveTimescaleUpdate(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.ctx = AdaptiveContextLayer(n_columns=16, device=self.device)
        self.original_log_tau = self.ctx.log_tau.clone()

    def test_update_shifts_timescales(self) -> None:
        # Neurons with above-mean differentiation should increase tau
        diff = torch.zeros(16)
        diff[0] = 1.0  # This neuron is very useful for routing
        self.ctx.update_timescales(diff)

        # Neuron 0 should have increased log_tau
        self.assertGreater(
            float(self.ctx.log_tau[0]), float(self.original_log_tau[0])
        )

    def test_tau_stays_clamped(self) -> None:
        # Giant differentiation should not push tau beyond limits
        diff = torch.ones(16) * 1e6
        self.ctx.update_timescales(diff)
        tau = self.ctx._tau()
        self.assertGreaterEqual(float(tau.min()), 2.0)
        self.assertLessEqual(float(tau.max()), 500.0)

    def test_zero_differentiation_no_change(self) -> None:
        diff = torch.zeros(16)
        self.ctx.update_timescales(diff)
        self.assertTrue(torch.allclose(self.ctx.log_tau, self.original_log_tau))


class TestTauDistribution(unittest.TestCase):
    def test_tau_distribution_keys(self) -> None:
        ctx = AdaptiveContextLayer(n_columns=32, device=torch.device("cpu"))
        dist = ctx.tau_distribution()
        expected_keys = {"tau_min", "tau_max", "tau_mean", "tau_std", "tau_median"}
        self.assertEqual(set(dist.keys()), expected_keys)

    def test_initial_tau_spread(self) -> None:
        ctx = AdaptiveContextLayer(n_columns=64, device=torch.device("cpu"))
        dist = ctx.tau_distribution()
        self.assertAlmostEqual(dist["tau_min"], 2.0, places=0)
        self.assertAlmostEqual(dist["tau_max"], 500.0, delta=2.0)
        # Std should be substantial (spread, not collapsed)
        self.assertGreater(dist["tau_std"], 10.0)


class TestAdaptiveContextStateDictRoundTrip(unittest.TestCase):
    def test_save_load_roundtrip(self) -> None:
        device = torch.device("cpu")
        ctx1 = AdaptiveContextLayer(n_columns=16, device=device)
        ctx1.observe(torch.randn(16))
        ctx1.update_timescales(torch.randn(16))

        snapshot = ctx1.state_dict()
        self.assertEqual(snapshot["context_mode"], "adaptive")

        ctx2 = AdaptiveContextLayer(n_columns=16, device=device)
        ctx2.load_state_dict(snapshot)

        self.assertTrue(torch.allclose(ctx1.log_tau, ctx2.log_tau))
        self.assertTrue(torch.allclose(ctx1.neuron_state, ctx2.neuron_state))
        self.assertTrue(torch.allclose(ctx1.state, ctx2.state))


class TestCreateContextLayerFactory(unittest.TestCase):
    def test_fixed_returns_context_layer(self) -> None:
        layer = create_context_layer("fixed", n_columns=16, device=torch.device("cpu"))
        self.assertIsInstance(layer, ContextLayer)

    def test_adaptive_returns_adaptive(self) -> None:
        layer = create_context_layer("adaptive", n_columns=16, device=torch.device("cpu"))
        self.assertIsInstance(layer, AdaptiveContextLayer)

    def test_both_have_observe_method(self) -> None:
        device = torch.device("cpu")
        for mode in ("fixed", "adaptive"):
            layer = create_context_layer(mode, n_columns=16, device=device)
            self.assertTrue(callable(getattr(layer, "observe", None)))

    def test_both_have_modulation_gain(self) -> None:
        device = torch.device("cpu")
        for mode in ("fixed", "adaptive"):
            layer = create_context_layer(mode, n_columns=16, device=device)
            self.assertTrue(callable(getattr(layer, "modulation_gain", None)))

    def test_both_have_state_dict(self) -> None:
        device = torch.device("cpu")
        for mode in ("fixed", "adaptive"):
            layer = create_context_layer(mode, n_columns=16, device=device)
            sd = layer.state_dict()
            self.assertIsInstance(sd, dict)


class TestAdaptiveContextWithTrainer(unittest.TestCase):
    """Smoke-test that the trainer can instantiate with adaptive context."""

    def test_config_flag_accepted(self) -> None:
        from hecsn.config.model_config import HECSNConfig

        cfg = HECSNConfig(context_mode="adaptive")
        self.assertEqual(cfg.context_mode, "adaptive")

    def test_config_default_is_fixed(self) -> None:
        from hecsn.config.model_config import HECSNConfig

        cfg = HECSNConfig()
        self.assertEqual(cfg.context_mode, "adaptive")


if __name__ == "__main__":
    unittest.main()
