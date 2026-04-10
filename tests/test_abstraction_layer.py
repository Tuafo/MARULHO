from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F

from hecsn.config.model_config import HECSNConfig
from hecsn.core import AbstractionLayer, CompetitiveColumnLayer
from hecsn.training.checkpointing import load_trainer_checkpoint, save_trainer_checkpoint
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer


class AbstractionLayerTests(unittest.TestCase):
    def test_repeated_pattern_builds_top_down_bias(self) -> None:
        torch.manual_seed(0)
        layer = AbstractionLayer(
            n_columns=4,
            n_concepts=3,
            device=torch.device("cpu"),
            feedback_strength=0.20,
        )
        pattern = torch.tensor([1.0, 0.0, 0.0, 0.0])

        for _ in range(12):
            layer.observe(pattern, update_weights=True)

        gain = layer.routing_gain()
        self.assertGreater(float(gain[0].item()), float(gain[1].item()))
        self.assertGreater(layer.summary()["mean_stability"], 0.0)

    def test_abstraction_bias_can_direct_competition(self) -> None:
        layer = CompetitiveColumnLayer(
            n_columns=2,
            column_dim=2,
            input_dim=2,
            device=torch.device("cpu"),
            input_weight_blend=0.0,
            dead_column_steps=10**9,
        )
        layer.prototypes[0] = F.normalize(torch.tensor([1.0, 1.0]), dim=0)
        layer.prototypes[1] = F.normalize(torch.tensor([1.0, 1.0]), dim=0)
        layer.thresholds.zero_()
        layer.last_input_pattern = torch.tensor([0.5, 0.5])

        abstraction = AbstractionLayer(
            n_columns=2,
            n_concepts=1,
            device=torch.device("cpu"),
            feedback_strength=0.25,
        )
        abstraction.slow_state = torch.tensor([1.0])
        abstraction.concept_stability = torch.tensor([1.0])
        abstraction.concept_certainty = torch.tensor([1.0])
        abstraction.feedback = torch.tensor([[0.1], [1.0]], dtype=torch.float32)

        routing_key = F.normalize(torch.tensor([1.0, 1.0]), dim=0)
        winners, _, _ = layer.compete(
            routing_key,
            torch.tensor([0, 1]),
            fallback_allowed=True,
            context_gain=abstraction.routing_gain(),
        )
        self.assertEqual(int(winners[0].item()), 1)


class AbstractionTrainerIntegrationTests(unittest.TestCase):
    def test_trainer_reports_abstraction_metrics_when_enabled(self) -> None:
        cfg = HECSNConfig(
            n_columns=4,
            column_latent_dim=8,
            bootstrap_tokens=0,
            memory_capacity=32,
            eta_competitive=0.05,
            eta_decay=0.0,
            input_weight_blend=0.0,
            enable_abstraction_layer=True,
        )
        trainer = HECSNTrainer(HECSNModelLite(cfg), cfg)
        pattern = trainer.encoder.feature_vector([ord(ch) for ch in "river"])

        metrics = trainer.train_step(pattern, raw_window="river")
        scope = trainer.model.runtime_scope_report()

        self.assertIn("abstraction_stability_mean", metrics)
        self.assertIn("abstraction_gain_mean", metrics)
        self.assertTrue(bool(scope["supports_first_class_abstraction"]))
        self.assertEqual(scope["abstraction_architecture"], "slow_feature_feedback_layer")

    def test_checkpoint_roundtrip_preserves_abstraction_state(self) -> None:
        cfg = HECSNConfig(
            n_columns=4,
            column_latent_dim=8,
            bootstrap_tokens=0,
            memory_capacity=32,
            eta_competitive=0.05,
            eta_decay=0.0,
            input_weight_blend=0.0,
            enable_abstraction_layer=True,
        )
        trainer = HECSNTrainer(HECSNModelLite(cfg), cfg)
        for text in ("river", "bank", "money", "loan"):
            pattern = trainer.encoder.feature_vector([ord(ch) for ch in text])
            trainer.train_step(pattern, raw_window=text)

        assert trainer.model.abstraction_layer is not None
        before_gain = trainer.model.abstraction_layer.routing_gain().detach().clone().cpu()
        before_summary = trainer.model.abstraction_layer.summary()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = save_trainer_checkpoint(Path(tmpdir) / "abstraction.pt", trainer)
            restored, _ = load_trainer_checkpoint(checkpoint_path)

        assert restored.model.abstraction_layer is not None
        after_gain = restored.model.abstraction_layer.routing_gain().detach().clone().cpu()
        after_summary = restored.model.abstraction_layer.summary()

        self.assertTrue(torch.allclose(before_gain, after_gain, atol=1e-5))
        self.assertEqual(before_summary["updates"], after_summary["updates"])


if __name__ == "__main__":
    unittest.main()
