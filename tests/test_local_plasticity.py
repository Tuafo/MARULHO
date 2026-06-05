from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from marulho.config.model_config import MarulhoConfig
from marulho.core.columns import CompetitiveColumnLayer
from marulho.data.rtf_encoder import RTFEncoder
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


class LocalPlasticityConfigTests(unittest.TestCase):
    def test_invalid_plasticity_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            MarulhoConfig(plasticity_mode="spike_eligibility")

    def test_local_stdp_allows_hashed_ngram_inputs(self) -> None:
        cfg = MarulhoConfig(
            input_representation="hashed_ngram",
            plasticity_mode="local_stdp",
        )
        self.assertEqual(cfg.plasticity_mode, "local_stdp")
        self.assertEqual(cfg.input_dim, cfg.hashed_ngram_dim)

    def test_local_stdp_accepts_adex_spike_backend(self) -> None:
        cfg = MarulhoConfig(
            plasticity_mode="local_stdp",
            plasticity_spike_backend="adex",
        )
        self.assertEqual(cfg.plasticity_spike_backend, "adex")


class LocalTraceTests(unittest.TestCase):
    def test_spike_trace_favors_earlier_latencies(self) -> None:
        encoder = RTFEncoder(window_size=3)

        trace = encoder.spike_trace(
            [ord("a"), ord("b"), ord("c")],
            context_confidence=1.0,
            tau=5.0,
            burst_decay=0.9,
        )

        self.assertGreater(float(trace[ord("a")].item()), float(trace[ord("b")].item()))
        self.assertGreater(float(trace[ord("b")].item()), float(trace[ord("c")].item()))


class CompetitiveLocalPlasticityTests(unittest.TestCase):
    def test_local_stdp_updates_weights_with_modulator_sign(self) -> None:
        layer = CompetitiveColumnLayer(
            n_columns=1,
            column_dim=4,
            input_dim=128,
            device=torch.device("cpu"),
            lr_initial=0.5,
            lr_decay=0.0,
            input_weight_blend=0.0,
            input_synapse_ltp=0.4,
            input_synapse_ltd=0.2,
            input_weight_row_target=1.0,
            plasticity_mode="local_stdp",
            prototype_momentum=0.0,
        )
        layer.prototypes[0] = F.normalize(torch.tensor([1.0, 0.1, 0.1, 0.1]), dim=0)
        layer.input_weights[0] = torch.full((128,), 1.0 / 128.0)
        layer.last_input_pattern = torch.full((128,), 1.0 / 128.0)
        layer.last_projected_input = F.normalize(torch.tensor([1.0, 0.4, 0.2, 0.1]), dim=0)

        local_trace = torch.zeros(128)
        local_trace[ord("a")] = 1.0
        routing_key = F.normalize(torch.tensor([1.0, 0.2, 0.0, 0.0]), dim=0)
        assembly_projection = torch.full((1, 4), 0.25)

        before_positive = layer.input_weights[0].clone()
        layer.process(
            routing_key,
            torch.tensor([0]),
            modulator=0.8,
            winner_strengths=torch.tensor([1.0]),
            eligibility_trace=local_trace,
            assembly_projection=assembly_projection,
        )
        after_positive = layer.input_weights[0].clone()
        self.assertGreater(float(after_positive[ord("a")].item()), float(before_positive[ord("a")].item()))
        self.assertLess(float(after_positive[ord("z")].item()), float(before_positive[ord("z")].item()))

        before_negative = after_positive.clone()
        layer.process(
            routing_key,
            torch.tensor([0]),
            modulator=-0.8,
            winner_strengths=torch.tensor([1.0]),
            eligibility_trace=local_trace,
            assembly_projection=assembly_projection,
        )
        after_negative = layer.input_weights[0]
        self.assertLess(float(after_negative[ord("a")].item()), float(before_negative[ord("a")].item()))

    def test_local_stdp_builds_inhibitory_tone_for_overused_column(self) -> None:
        layer = CompetitiveColumnLayer(
            n_columns=2,
            column_dim=4,
            input_dim=16,
            device=torch.device("cpu"),
            lr_initial=0.2,
            lr_decay=0.0,
            input_weight_blend=0.0,
            plasticity_mode="local_stdp",
            prototype_momentum=0.0,
        )
        layer.last_input_pattern = torch.full((16,), 1.0 / 16.0)
        layer.last_projected_input = F.normalize(torch.tensor([1.0, 0.5, 0.2, 0.1]), dim=0)
        assembly_projection = torch.full((2, 4), 0.25)
        routing_key = F.normalize(torch.tensor([1.0, 0.2, 0.0, 0.0]), dim=0)

        for _ in range(8):
            layer.process(
                routing_key,
                torch.tensor([0]),
                modulator=0.6,
                winner_strengths=torch.tensor([1.0]),
                assembly_projection=assembly_projection,
            )

        self.assertIsNotNone(layer.local_plasticity)
        assert layer.local_plasticity is not None
        self.assertGreater(
            float(layer.local_plasticity.inhibitory_tone[0].item()),
            float(layer.local_plasticity.inhibitory_tone[1].item()),
        )

    def test_local_stdp_can_use_adex_post_spike_backend(self) -> None:
        layer = CompetitiveColumnLayer(
            n_columns=2,
            column_dim=4,
            input_dim=16,
            device=torch.device("cpu"),
            lr_initial=0.2,
            lr_decay=0.0,
            input_weight_blend=0.0,
            plasticity_mode="local_stdp",
            plasticity_spike_backend="adex",
            prototype_momentum=0.0,
        )
        layer.last_input_pattern = torch.full((16,), 1.0 / 16.0)
        layer.last_projected_input = F.normalize(torch.tensor([1.0, 0.5, 0.2, 0.1]), dim=0)
        assembly_projection = torch.full((2, 4), 0.25)
        routing_key = F.normalize(torch.tensor([1.0, 0.2, 0.0, 0.0]), dim=0)

        for _ in range(4):
            layer.process(
                routing_key,
                torch.tensor([0]),
                modulator=0.6,
                winner_strengths=torch.tensor([1.0]),
                assembly_projection=assembly_projection,
            )

        self.assertIsNotNone(layer.local_plasticity)
        assert layer.local_plasticity is not None
        self.assertEqual(layer.local_plasticity.spike_backend, "adex")
        self.assertIsNotNone(layer.local_plasticity.adex_neurons)
        assert layer.local_plasticity.adex_neurons is not None
        self.assertGreaterEqual(layer.local_plasticity.last_post_spike_fraction, 0.0)
        self.assertTrue(bool(torch.isfinite(layer.local_plasticity.adex_neurons.V).all().item()))
        self.assertTrue(bool((layer.local_plasticity.adex_neurons.spike_times >= 0.0).any().item()))


class LocalPlasticityTrainerIntegrationTests(unittest.TestCase):
    def test_trainer_reports_local_trace_metrics_when_enabled(self) -> None:
        cfg = MarulhoConfig(
            n_columns=4,
            column_latent_dim=8,
            bootstrap_tokens=0,
            memory_capacity=32,
            eta_competitive=0.05,
            eta_decay=0.0,
            input_weight_blend=0.0,
            plasticity_mode="local_stdp",
        )
        model = MarulhoModel(cfg)
        trainer = MarulhoTrainer(model, cfg)
        pattern = trainer.encoder.feature_vector([ord(ch) for ch in "bank"])

        metrics = trainer.train_step(pattern, raw_window="bank")
        scope = trainer.model.runtime_scope_report()

        self.assertEqual(metrics["plasticity_mode"], "local_stdp")
        self.assertEqual(metrics["local_trace_available"], 1)
        self.assertGreater(int(metrics["local_trace_active_inputs"]), 0)
        self.assertIn("serotonin", metrics)
        self.assertTrue(bool(scope["supports_local_log_stdp"]))
        self.assertTrue(bool(scope["supports_inhibitory_balance"]))

    def test_trainer_reports_optional_adex_post_spike_backend(self) -> None:
        cfg = MarulhoConfig(
            n_columns=4,
            column_latent_dim=8,
            bootstrap_tokens=0,
            memory_capacity=32,
            eta_competitive=0.05,
            eta_decay=0.0,
            input_weight_blend=0.0,
            plasticity_mode="local_stdp",
            plasticity_spike_backend="adex",
        )
        model = MarulhoModel(cfg)
        trainer = MarulhoTrainer(model, cfg)
        pattern = trainer.encoder.feature_vector([ord(ch) for ch in "bank"])

        metrics = trainer.train_step(pattern, raw_window="bank")
        scope = trainer.model.runtime_scope_report()

        self.assertEqual(metrics["plasticity_spike_backend"], "adex")
        self.assertGreaterEqual(metrics["local_post_spike_fraction"], 0.0)
        self.assertLessEqual(metrics["local_mean_membrane_voltage"], 20.0)
        self.assertEqual(scope["plasticity_spike_backend"], "adex")
        self.assertTrue(bool(scope["uses_adex_post_spikes"]))

    def test_trainer_local_stdp_supports_hashed_ngram_without_raw_trace(self) -> None:
        cfg = MarulhoConfig(
            input_representation="hashed_ngram",
            plasticity_mode="local_stdp",
            n_columns=4,
            column_latent_dim=8,
            bootstrap_tokens=0,
            memory_capacity=32,
            eta_competitive=0.05,
            eta_decay=0.0,
            input_weight_blend=0.0,
        )
        model = MarulhoModel(cfg)
        trainer = MarulhoTrainer(model, cfg)
        pattern = trainer.encoder.feature_vector([ord(ch) for ch in "bank"])

        metrics = trainer.train_step(pattern, raw_window=None)

        self.assertEqual(metrics["plasticity_mode"], "local_stdp")
        self.assertEqual(metrics["local_trace_available"], 0)
        self.assertGreaterEqual(metrics["recon_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
