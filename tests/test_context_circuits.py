from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from hecsn.config.model_config import HECSNConfig
from hecsn.core.context import BindingLayer, ContextLayer
from hecsn.training.checkpointing import load_trainer_checkpoint, save_trainer_checkpoint
from hecsn.training.trainer import HECSNModel, HECSNTrainer


class ContextCircuitTests(unittest.TestCase):
    def test_context_device_report_exposes_live_tensor_devices(self) -> None:
        layer = ContextLayer(
            n_columns=4,
            device=torch.device("cpu"),
        )
        report = layer.device_report()

        self.assertEqual(report["module"], "context_fixed")
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["fast_state_device"], str(layer.fast_state.device))
        self.assertEqual(report["medium_state_device"], str(layer.medium_state.device))
        self.assertEqual(report["slow_state_device"], str(layer.slow_state.device))
        self.assertEqual(report["recurrent_device"], str(layer.recurrent.device))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_context_cuda_device_report_exposes_live_tensor_devices(self) -> None:
        layer = ContextLayer(
            n_columns=4,
            device=torch.device("cuda"),
        )
        layer.observe(torch.randn(4, device=torch.device("cuda")).abs())
        report = layer.device_report()

        self.assertTrue(str(report["device"]).startswith("cuda"))
        self.assertTrue(str(report["fast_state_device"]).startswith("cuda"))
        self.assertTrue(str(report["recurrent_device"]).startswith("cuda"))

    def test_model_subcortex_device_report_includes_fixed_context(self) -> None:
        cfg = HECSNConfig(
            n_columns=4,
            column_latent_dim=8,
            bootstrap_tokens=0,
            memory_capacity=32,
            enable_context_layer=True,
            context_mode="fixed",
        )
        model = HECSNModel(cfg)
        report = model.subcortex_device_report()["context"]

        self.assertIsNotNone(report)
        assert model.context_layer is not None
        self.assertEqual(report["module"], "context_fixed")
        self.assertEqual(report["state_device"], str(model.context_layer.state.device))

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

    def test_low_precision_slows_context_integration(self) -> None:
        torch.manual_seed(2)
        seed_layer = ContextLayer(
            n_columns=4,
            device=torch.device("cpu"),
            recurrent_density=1.0,
            recurrent_scale=0.5,
        )
        a = torch.tensor([1.0, 0.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0, 0.0])
        seed_layer.observe(a, update_weights=True)
        seed_layer.observe(a, update_weights=True)

        high_precision = ContextLayer(
            n_columns=4,
            device=torch.device("cpu"),
            recurrent_density=1.0,
            recurrent_scale=0.5,
        )
        low_precision = ContextLayer(
            n_columns=4,
            device=torch.device("cpu"),
            recurrent_density=1.0,
            recurrent_scale=0.5,
        )
        snapshot = seed_layer.state_dict()
        high_precision.load_state_dict(snapshot)
        low_precision.load_state_dict(snapshot)

        high_precision.observe(b, update_weights=False, precision_weight=1.0)
        low_precision.observe(b, update_weights=False, precision_weight=0.0)

        self.assertGreater(float(high_precision.fast_state[1].item()), float(low_precision.fast_state[1].item()))
        self.assertGreater(float(low_precision.slow_state[0].item()), float(high_precision.slow_state[0].item()))


class BindingCircuitTests(unittest.TestCase):
    def test_binding_device_report_exposes_live_tensor_devices(self) -> None:
        layer = BindingLayer(
            n_columns=4,
            n_bindings=8,
            fan_in=2,
            device=torch.device("cpu"),
        )
        report = layer.device_report()

        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["binding_state_device"], str(layer.binding_state.device))
        self.assertEqual(report["coincidence_trace_device"], str(layer.coincidence_trace.device))
        self.assertEqual(report["connectivity_device"], str(layer.connectivity.device))
        self.assertEqual(report["output_weights_device"], str(layer.output_weights.device))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_binding_cuda_device_report_exposes_live_tensor_devices(self) -> None:
        layer = BindingLayer(
            n_columns=4,
            n_bindings=8,
            fan_in=2,
            device=torch.device("cuda"),
        )
        report = layer.device_report()

        self.assertTrue(str(report["device"]).startswith("cuda"))
        self.assertTrue(str(report["binding_state_device"]).startswith("cuda"))
        self.assertTrue(str(report["connectivity_device"]).startswith("cuda"))

    def test_binding_facilitation_strengthens_repeated_coincidence(self) -> None:
        torch.manual_seed(3)
        layer = BindingLayer(
            n_columns=4,
            n_bindings=8,
            fan_in=2,
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
        self.assertNotEqual(layer.n_bindings, layer.n_columns)
        self.assertTrue(torch.all(layer.connectivity.sum(dim=1) == 2.0))
        self.assertGreater(float(layer.facilitation.max().item()), 0.0)

    def test_binding_prediction_prefers_learned_conjunction(self) -> None:
        torch.manual_seed(4)
        layer = BindingLayer(
            n_columns=4,
            n_bindings=8,
            fan_in=2,
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

    def test_binding_growth_adds_new_sparse_subset(self) -> None:
        torch.manual_seed(5)
        layer = BindingLayer(
            n_columns=6,
            n_bindings=4,
            fan_in=2,
            device=torch.device("cpu"),
        )
        covered_pairs = {
            tuple(indices.tolist())
            for indices in (row.nonzero(as_tuple=True)[0].cpu() for row in layer.connectivity)
        }
        candidate_pair = next(pair for pair in ((0, 3), (1, 4), (2, 5), (0, 5)) if pair not in covered_pairs)

        grown = layer.grow_binding([(candidate_pair[0], candidate_pair[1], 0.9)])

        self.assertEqual(grown, 1)
        self.assertEqual(layer.n_bindings, 5)
        self.assertTrue(
            bool(
                ((layer.connectivity[:, candidate_pair[0]] > 0.0) & (layer.connectivity[:, candidate_pair[1]] > 0.0))
                .any()
                .item()
            )
        )


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
        model = HECSNModel(cfg)
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
