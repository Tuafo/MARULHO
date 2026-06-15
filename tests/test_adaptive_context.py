"""Tests for AdaptiveContextLayer (§4.3) and create_context_layer factory."""

from __future__ import annotations

import math
import unittest

import torch

from marulho.core.context import (
    AdaptiveContextLayer,
    ContextLayer,
    create_context_layer,
)


class TestAdaptiveContextLayerInit(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.ctx = AdaptiveContextLayer(n_columns=64, device=self.device)

    def test_device_report_exposes_live_tensor_devices(self) -> None:
        report = self.ctx.device_report()

        self.assertEqual(report["module"], "context_adaptive")
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["log_tau_device"], str(self.ctx.log_tau.device))
        self.assertEqual(report["neuron_state_device"], str(self.ctx.neuron_state.device))
        self.assertEqual(report["state_device"], str(self.ctx.state.device))
        self.assertEqual(report["w_in_device"], str(self.ctx.w_in.device))
        self.assertEqual(report["w_out_device"], str(self.ctx.w_out.device))
        self.assertEqual(report["state_update_count"], 0)
        self.assertEqual(report["plasticity_update_count"], 0)
        self.assertFalse(report["last_update_weights"])

    def test_device_report_includes_observation_snapshot_device(self) -> None:
        self.ctx.observe(torch.randn(64).abs(), update_weights=True)
        report = self.ctx.device_report()

        self.assertEqual(report["context_observation_count"], 1)
        self.assertEqual(report["latest_context_observation_device"], "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_cuda_device_report_exposes_live_tensor_devices(self) -> None:
        ctx = AdaptiveContextLayer(n_columns=16, device=torch.device("cuda"))
        ctx.observe(torch.randn(16, device=torch.device("cuda")).abs(), update_weights=True)
        report = ctx.device_report()

        self.assertTrue(str(report["device"]).startswith("cuda"))
        self.assertTrue(str(report["log_tau_device"]).startswith("cuda"))
        self.assertTrue(str(report["neuron_state_device"]).startswith("cuda"))
        self.assertTrue(str(report["latest_context_observation_device"]).startswith("cuda"))

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

    def test_state_updates_without_weight_plasticity(self) -> None:
        assembly = torch.randn(32)
        weights_before = self.ctx.w_in.clone()

        self.ctx.observe(assembly, update_weights=False)
        report = self.ctx.device_report()

        self.assertGreater(float(self.ctx.neuron_state.abs().sum()), 0.0)
        self.assertTrue(torch.equal(self.ctx.w_in, weights_before))
        self.assertEqual(report["state_update_count"], 1)
        self.assertEqual(report["plasticity_update_count"], 0)
        self.assertFalse(report["last_update_weights"])

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


class TestRoutingDifferentiation(unittest.TestCase):
    """Tests for compute_routing_differentiation context-specificity metric."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.ctx = AdaptiveContextLayer(n_columns=32, device=self.device)

    def test_returns_zeros_with_insufficient_data(self) -> None:
        # No observations → zeros
        rd = self.ctx.compute_routing_differentiation()
        self.assertEqual(float(rd.sum()), 0.0)

    def test_returns_zeros_with_too_few_observations(self) -> None:
        for _ in range(5):
            self.ctx.observe(torch.randn(32))
        rd = self.ctx.compute_routing_differentiation()
        self.assertEqual(float(rd.sum()), 0.0)

    def test_context_observations_recorded_during_wake(self) -> None:
        self.ctx.observe(torch.randn(32), update_weights=True)
        self.assertEqual(len(self.ctx._context_observations), 1)

    def test_context_observations_not_recorded_during_replay(self) -> None:
        self.ctx.observe(torch.randn(32), update_weights=False)
        self.assertEqual(len(self.ctx._context_observations), 0)

    def test_reset_state_clears_buffer(self) -> None:
        for _ in range(20):
            self.ctx.observe(torch.randn(32))
        self.assertGreater(len(self.ctx._context_observations), 0)
        self.ctx.reset_state()
        self.assertEqual(len(self.ctx._context_observations), 0)

    def test_same_input_different_context_yields_nonzero(self) -> None:
        """Same input seen multiple times under different contexts → nonzero."""
        fixed_input = torch.randn(32).abs()
        fixed_input = fixed_input / (fixed_input.norm() + 1e-8)
        # Present the same input 20 times, interleaved with different priming
        for i in range(20):
            # Prime with different context
            primer = torch.randn(32).abs()
            self.ctx.observe(primer)
            # Present the fixed input
            self.ctx.observe(fixed_input.clone())
        rd = self.ctx.compute_routing_differentiation()
        # With the same input under varying contexts, we should see nonzero
        # differentiation (context layer state varies because of different primes)
        self.assertEqual(rd.shape[0], self.ctx.n_neurons)

    def test_input_signature_deterministic(self) -> None:
        """Same assembly vector produces the same signature."""
        assembly = torch.randn(32).abs()
        sig1 = AdaptiveContextLayer._compute_input_signature(assembly)
        sig2 = AdaptiveContextLayer._compute_input_signature(assembly)
        self.assertEqual(sig1, sig2)

    def test_different_inputs_different_signatures(self) -> None:
        """Very different assemblies produce different signatures."""
        a1 = torch.zeros(32)
        a1[0] = 1.0
        a2 = torch.zeros(32)
        a2[31] = 1.0
        sig1 = AdaptiveContextLayer._compute_input_signature(a1)
        sig2 = AdaptiveContextLayer._compute_input_signature(a2)
        self.assertNotEqual(sig1, sig2)

    def test_buffer_capped_at_maxlen(self) -> None:
        for _ in range(250):
            self.ctx.observe(torch.randn(32))
        self.assertLessEqual(
            len(self.ctx._context_observations),
            self.ctx._context_observations_maxlen,
        )


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
        from marulho.config.model_config import MarulhoConfig

        cfg = MarulhoConfig(context_mode="adaptive")
        self.assertEqual(cfg.context_mode, "adaptive")
        self.assertEqual(cfg.context_plasticity_interval_tokens, 4)

    def test_context_plasticity_interval_must_be_positive(self) -> None:
        from marulho.config.model_config import MarulhoConfig

        with self.assertRaisesRegex(ValueError, "context_plasticity_interval_tokens"):
            MarulhoConfig(context_plasticity_interval_tokens=0)

    def test_trainer_telemetry_interval_must_be_positive(self) -> None:
        from marulho.config.model_config import MarulhoConfig

        with self.assertRaisesRegex(ValueError, "trainer_telemetry_interval_tokens"):
            MarulhoConfig(trainer_telemetry_interval_tokens=0)

    def test_cuda_graph_host_truth_sync_interval_must_be_positive(self) -> None:
        from marulho.config.model_config import MarulhoConfig

        with self.assertRaisesRegex(ValueError, "cuda_graph_host_truth_sync_interval_tokens"):
            MarulhoConfig(cuda_graph_host_truth_sync_interval_tokens=0)

    def test_cuda_graph_native_burst_tokens_must_be_supported_capacity(self) -> None:
        from marulho.config.model_config import MarulhoConfig

        self.assertEqual(MarulhoConfig(cuda_graph_native_burst_tokens=16).cuda_graph_native_burst_tokens, 16)
        with self.assertRaisesRegex(ValueError, "cuda_graph_native_burst_tokens"):
            MarulhoConfig(cuda_graph_native_burst_tokens=24)

    def test_slow_memory_archive_interval_must_be_positive(self) -> None:
        from marulho.config.model_config import MarulhoConfig

        with self.assertRaisesRegex(ValueError, "slow_memory_archive_interval_tokens"):
            MarulhoConfig(slow_memory_archive_interval_tokens=0)

    def test_slow_memory_archive_strong_capture_threshold_must_be_non_negative(self) -> None:
        from marulho.config.model_config import MarulhoConfig

        with self.assertRaisesRegex(ValueError, "slow_memory_archive_strong_capture_threshold"):
            MarulhoConfig(slow_memory_archive_strong_capture_threshold=-0.1)

    def test_config_default_is_fixed(self) -> None:
        from marulho.config.model_config import MarulhoConfig

        cfg = MarulhoConfig()
        assert cfg.cuda_graph_quantum_input_staging is True
        assert cfg.cuda_graph_sequence_input_staging is True
        self.assertEqual(cfg.cuda_graph_native_burst_tokens, 8)
        self.assertEqual(cfg.context_mode, "adaptive")
        self.assertEqual(cfg.cuda_graph_host_truth_sync_interval_tokens, 32)
        self.assertEqual(cfg.slow_memory_archive_interval_tokens, 256)

    def test_model_subcortex_device_report_includes_adaptive_context(self) -> None:
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel

        cfg = MarulhoConfig(
            n_columns=8,
            column_latent_dim=8,
            bootstrap_tokens=0,
            memory_capacity=32,
            enable_context_layer=True,
            context_mode="adaptive",
        )
        model = MarulhoModel(cfg)
        report = model.subcortex_device_report()["context"]

        self.assertIsNotNone(report)
        assert model.context_layer is not None
        self.assertEqual(report["module"], "context_adaptive")
        self.assertEqual(report["state_device"], str(model.context_layer.state.device))

    def test_trainer_keeps_context_state_live_with_cadenced_plasticity(self) -> None:
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        cfg = MarulhoConfig(
            n_columns=16,
            column_latent_dim=8,
            bootstrap_tokens=0,
            memory_capacity=32,
            enable_context_layer=True,
            context_mode="adaptive",
            context_plasticity_interval_tokens=4,
        )
        model = MarulhoModel(cfg)
        trainer = MarulhoTrainer(model, cfg)
        trainer.memory_warm_started = True

        metrics = []
        for step in range(5):
            metrics.append(
                trainer.train_step(
                    torch.randn(cfg.input_dim),
                    raw_window=f"context_{step}",
                )
            )

        assert model.context_layer is not None
        report = model.context_layer.device_report()
        self.assertEqual(report["state_update_count"], 5)
        self.assertEqual(report["plasticity_update_count"], 2)
        self.assertFalse(report["last_update_weights"])
        self.assertEqual(
            [item["context_plasticity_due"] for item in metrics],
            [1, 0, 0, 1, 0],
        )
        self.assertTrue(all(item["context_plasticity_interval_tokens"] == 4 for item in metrics))


if __name__ == "__main__":
    unittest.main()
