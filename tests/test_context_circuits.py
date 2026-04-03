from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from hecsn.config.model_config import HECSNConfig
from hecsn.core.context import BindingLayer, ContextLayer
from hecsn.training.checkpointing import load_trainer_checkpoint, save_trainer_checkpoint
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer


class ContextCircuitTests(unittest.TestCase):
    def test_multiscale_context_retains_past_while_tracking_present(self) -> None:
        torch.manual_seed(0)
        layer = ContextLayer(
            n_columns=4,
            device=torch.device("cpu"),
            fast_rate=0.8,
            medium_rate=0.25,
            slow_rate=0.05,
            recurrent_density=1.0,
            recurrent_scale=0.5,
        )
        a = torch.tensor([1.0, 0.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0, 0.0])

        layer.observe(a, update_weights=True)
        layer.observe(a, update_weights=True)
        layer.observe(b, update_weights=True)

        self.assertGreater(float(layer.fast_state[1].item()), float(layer.fast_state[0].item()))
        self.assertGreater(float(layer.slow_state[0].item()), 0.0)

    def test_context_prediction_learns_transition_bias(self) -> None:
        torch.manual_seed(1)
        layer = ContextLayer(
            n_columns=4,
            device=torch.device("cpu"),
            recurrent_density=1.0,
            recurrent_scale=0.5,
            transition_lr=0.25,
        )
        a = torch.tensor([1.0, 0.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0, 0.0])

        for _ in range(6):
            layer.observe(a, update_weights=True)
            layer.observe(b, update_weights=True)

        layer.reset_state()
        layer.observe(a, update_weights=False)
        prediction = layer.context_prediction()

        self.assertGreater(float(prediction[1].item()), float(prediction[0].item()))


class BindingCircuitTests(unittest.TestCase):
    def test_binding_facilitation_strengthens_repeated_coincidence(self) -> None:
        layer = BindingLayer(
            n_columns=4,
            device=torch.device("cpu"),
            threshold=0.0,
            association_lr=0.3,
            gain_strength=0.8,
            tau_binding=8.0,
        )
        context = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assembly = torch.tensor([1.0, 0.0, 0.0, 0.0])

        _, first_strength = layer.bind(context, assembly, update_weights=True)
        _, second_strength = layer.bind(context, assembly, update_weights=True)

        self.assertGreaterEqual(second_strength, first_strength)
        self.assertGreater(float(layer.facilitation[0].item()), 0.0)

    def test_binding_prediction_prefers_learned_conjunction(self) -> None:
        layer = BindingLayer(
            n_columns=4,
            device=torch.device("cpu"),
            threshold=0.0,
            association_lr=0.5,
            association_decay=1.0,
            gain_strength=0.6,
        )
        context = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assembly = torch.tensor([0.0, 1.0, 0.0, 0.0])

        for _ in range(4):
            layer.bind(context, assembly, update_weights=True)

        prediction = layer.binding_prediction(context)
        self.assertGreater(float(prediction[1].item()), float(prediction[0].item()))


class ContextCheckpointTests(unittest.TestCase):
    def test_checkpoint_roundtrip_preserves_context_and_binding_state(self) -> None:
        cfg = HECSNConfig(
            n_columns=4,
            column_latent_dim=8,
            bootstrap_tokens=0,
            memory_capacity=32,
            eta_competitive=0.05,
            eta_decay=0.0,
            input_weight_blend=0.0,
            enable_context_layer=True,
            enable_binding_layer=True,
        )
        model = HECSNModelLite(cfg)
        trainer = HECSNTrainer(model, cfg)
        patterns = [
            trainer.encoder.feature_vector([ord(ch) for ch in text])
            for text in ("river", "bank", "money", "loan")
        ]
        for text, pattern in zip(("river", "bank", "money", "loan"), patterns):
            trainer.train_step(pattern, raw_window=text)

        before_context = trainer.context_state().clone()
        assert trainer.model.binding_layer is not None
        before_binding = trainer.model.binding_layer.binding_state.detach().clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = save_trainer_checkpoint(Path(tmpdir) / "context.pt", trainer)
            restored, _ = load_trainer_checkpoint(checkpoint_path)

        after_context = restored.context_state()
        assert restored.model.binding_layer is not None
        after_binding = restored.model.binding_layer.binding_state.detach().cpu()

        self.assertTrue(torch.allclose(before_context, after_context, atol=1e-5))
        self.assertTrue(torch.allclose(before_binding.cpu(), after_binding, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
