"""Tests for CrossModalGroundingLayer (§5.1–§5.3)."""

from __future__ import annotations

import unittest

import torch

from marulho.core.cross_modal import CrossModalGroundingLayer


class TestCrossModalInit(unittest.TestCase):
    def test_dimensions(self) -> None:
        layer = CrossModalGroundingLayer(dim_text=24, dim_visual=64, dim_audio=32)
        self.assertEqual(layer.W_tv.shape, (24, 64))
        self.assertEqual(layer.W_vt.shape, (64, 24))
        self.assertEqual(layer.W_ta.shape, (24, 32))
        self.assertEqual(layer.W_at.shape, (32, 24))

    def test_confidence_starts_zero(self) -> None:
        layer = CrossModalGroundingLayer(dim_text=10, dim_visual=20, dim_audio=8)
        self.assertEqual(layer.visual_confidence.sum().item(), 0.0)
        self.assertEqual(layer.audio_confidence.sum().item(), 0.0)

    def test_device_report_exposes_live_tensor_devices(self) -> None:
        layer = CrossModalGroundingLayer(
            dim_text=10,
            dim_visual=20,
            dim_audio=8,
            device=torch.device("cpu"),
        )
        report = layer.device_report()

        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["W_tv_device"], str(layer.W_tv.device))
        self.assertEqual(report["W_vt_device"], str(layer.W_vt.device))
        self.assertEqual(report["W_ta_device"], str(layer.W_ta.device))
        self.assertEqual(report["W_at_device"], str(layer.W_at.device))
        self.assertEqual(report["text_trace_device"], str(layer.text_trace.device))
        self.assertEqual(report["visual_confidence_device"], str(layer.visual_confidence.device))
        self.assertEqual(report["runtime_text_update_count"], 0)
        self.assertEqual(report["runtime_text_idle_skip_count"], 0)
        self.assertEqual(report["last_text_runtime_execution_mode"], "not_run")

    def test_text_idle_skip_reports_without_weight_update(self) -> None:
        layer = CrossModalGroundingLayer(dim_text=10, dim_visual=20, dim_audio=8)
        layer.W_tv.fill_(0.1)
        before = layer.W_tv.clone()

        layer.record_text_idle_skip()
        report = layer.device_report()

        self.assertTrue(torch.equal(layer.W_tv, before))
        self.assertEqual(report["runtime_text_update_count"], 0)
        self.assertEqual(report["runtime_text_idle_skip_count"], 1)
        self.assertEqual(report["last_text_runtime_execution_mode"], "text_idle_cached_state")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_cuda_device_report_exposes_live_tensor_devices_after_load(self) -> None:
        layer = CrossModalGroundingLayer(dim_text=10, dim_visual=20, dim_audio=8)
        layer.on_visual_spike(torch.ones(20))
        layer.on_text_spike(torch.ones(10))
        state = layer.state_dict()

        cuda_layer = CrossModalGroundingLayer(
            dim_text=10,
            dim_visual=20,
            dim_audio=8,
            device=torch.device("cuda"),
        )
        cuda_layer.load_state_dict(state)
        report = cuda_layer.device_report()

        self.assertTrue(str(report["device"]).startswith("cuda"))
        self.assertTrue(str(report["W_tv_device"]).startswith("cuda"))
        self.assertTrue(str(report["text_trace_device"]).startswith("cuda"))

    def test_model_subcortex_report_includes_cross_modal_devices(self) -> None:
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel

        cfg = MarulhoConfig(
            n_columns=8,
            column_latent_dim=8,
            enable_cross_modal=True,
        )
        model = MarulhoModel(cfg)
        report = model.subcortex_device_report()["cross_modal"]

        self.assertIsNotNone(report)
        assert model.cross_modal is not None
        self.assertEqual(report["W_tv_device"], str(model.cross_modal.W_tv.device))
        self.assertEqual(report["text_trace_device"], str(model.cross_modal.text_trace.device))


class TestSTDPUpdates(unittest.TestCase):
    def setUp(self) -> None:
        self.layer = CrossModalGroundingLayer(
            dim_text=10, dim_visual=16, dim_audio=8,
            A_plus=0.01, A_minus=0.012,
        )

    def test_text_visual_cooccurrence_potentiates(self) -> None:
        """When text and visual fire together, W_tv should increase."""
        w_before = self.layer.W_tv.clone()
        # Visual spike first, then text
        visual = torch.zeros(16)
        visual[0:4] = 1.0
        self.layer.on_visual_spike(visual)
        text = torch.zeros(10)
        text[0:3] = 1.0
        self.layer.on_text_spike(text)

        # Check that W_tv increased in the co-active region
        delta = self.layer.W_tv - w_before
        # Should have positive changes where text[i] * visual_trace[j] > 0
        coactive_delta = delta[0:3, 0:4]
        self.assertGreater(coactive_delta.sum().item(), 0.0)

    def test_text_alone_depresses(self) -> None:
        """Text without visual should depress W_tv (anti-Hebbian)."""
        # Set some initial weight
        self.layer.W_tv = torch.ones(10, 16) * 0.1
        w_before = self.layer.W_tv.clone()

        text = torch.zeros(10)
        text[0] = 1.0
        self.layer.on_text_spike(text)

        # Visual trace is zero → all positions should see depression
        self.assertLess(
            self.layer.W_tv[0].sum().item(),
            w_before[0].sum().item(),
        )
        self.assertEqual(self.layer.runtime_text_update_count, 1)
        self.assertEqual(self.layer.last_text_runtime_execution_mode, "text_update")

    def test_audio_text_cooccurrence(self) -> None:
        """When audio and text fire together, W_at should increase."""
        w_before = self.layer.W_at.clone()
        text = torch.zeros(10)
        text[0] = 1.0
        self.layer.on_text_spike(text)
        audio = torch.zeros(8)
        audio[0:2] = 1.0
        self.layer.on_audio_spike(audio)

        delta = self.layer.W_at - w_before
        self.assertGreater(delta[0:2, 0].sum().item(), 0.0)

    def test_trace_decays(self) -> None:
        """Traces should decay after each event."""
        visual = torch.ones(16)
        self.layer.on_visual_spike(visual)
        trace_after_spike = self.layer.visual_trace.clone()

        # Fire text (which also decays all traces)
        self.layer.on_text_spike(torch.zeros(10))
        self.assertLess(
            self.layer.visual_trace.sum().item(),
            trace_after_spike.sum().item(),
        )


class TestGroundingConfidence(unittest.TestCase):
    def setUp(self) -> None:
        self.layer = CrossModalGroundingLayer(
            dim_text=10, dim_visual=16, dim_audio=8,
            confidence_alpha=0.1,
        )

    def test_confidence_increases_with_cooccurrence(self) -> None:
        """Repeated co-occurrence should build confidence."""
        for _ in range(50):
            visual = torch.zeros(16)
            visual[0:4] = 1.0
            self.layer.on_visual_spike(visual)
            text = torch.zeros(10)
            text[0:3] = 1.0
            self.layer.on_text_spike(text)

        # After many co-occurrences, visual confidence for active dims should rise
        self.assertGreater(self.layer.visual_confidence[0:3].sum().item(), 0.0)

    def test_combined_confidence(self) -> None:
        conf = self.layer.grounding_confidence()
        self.assertEqual(conf.shape, (10,))
        self.assertEqual(conf.sum().item(), 0.0)  # initially zero


class TestPredictions(unittest.TestCase):
    def setUp(self) -> None:
        self.layer = CrossModalGroundingLayer(dim_text=10, dim_visual=16, dim_audio=8)

    def test_predict_visual(self) -> None:
        text = torch.randn(10)
        pred = self.layer.predict_visual(text)
        self.assertEqual(pred.shape, (16,))

    def test_predict_text_from_visual(self) -> None:
        visual = torch.randn(16)
        pred = self.layer.predict_text_from_visual(visual)
        self.assertEqual(pred.shape, (10,))

    def test_predict_audio(self) -> None:
        text = torch.randn(10)
        pred = self.layer.predict_audio(text)
        self.assertEqual(pred.shape, (8,))

    def test_predict_text_from_audio(self) -> None:
        audio = torch.randn(8)
        pred = self.layer.predict_text_from_audio(audio)
        self.assertEqual(pred.shape, (10,))


class TestTrainerCrossModalRuntimeWake(unittest.TestCase):
    def _trainer(self):
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        cfg = MarulhoConfig(
            n_columns=16,
            column_latent_dim=8,
            bootstrap_tokens=0,
            enable_cross_modal=True,
            cross_modal_dim_visual=8,
            cross_modal_dim_audio=4,
            cross_modal_text_idle_probe_interval_tokens=4,
            micro_sleep_interval_tokens=10**9,
            deep_sleep_interval_tokens=10**9,
        )
        return MarulhoTrainer(MarulhoModel(cfg), cfg), cfg

    def test_text_only_cross_modal_stays_asleep_without_sensory_trace(self) -> None:
        trainer, cfg = self._trainer()
        pattern = torch.rand(cfg.input_dim)

        metrics = {}
        for _ in range(4):
            metrics = trainer.train_step(
                pattern,
                raw_window="text only cross modal",
                allow_sleep_maintenance=False,
            )

        report = trainer.model.cross_modal.device_report()
        self.assertEqual(report["runtime_text_update_count"], 0)
        self.assertEqual(report["runtime_text_idle_skip_count"], 4)
        self.assertEqual(report["last_text_runtime_execution_mode"], "text_idle_cached_state")
        self.assertEqual(metrics["cross_modal_text_update_count"], 0)
        self.assertEqual(metrics["cross_modal_text_idle_skip_count"], 4)
        self.assertEqual(metrics["cross_modal_fast_idle_skip_count"], 4)
        self.assertEqual(metrics["cross_modal_idle_trace_reset_count"], 0)
        self.assertEqual(metrics["cross_modal_text_spike_prepared"], 0)
        self.assertEqual(
            metrics["cross_modal_text_execution_mode"],
            "text_idle_cached_state",
        )

    def test_cross_modal_text_idle_clears_expired_trace_once(self) -> None:
        trainer, cfg = self._trainer()
        pattern = torch.rand(cfg.input_dim)
        visual = torch.rand(cfg.cross_modal_dim_visual)

        trainer.train_step(
            pattern,
            raw_window="visual cross modal",
            visual_spikes=visual,
            allow_sleep_maintenance=False,
        )
        assert trainer.model.cross_modal is not None
        self.assertGreater(float(trainer.model.cross_modal.visual_trace.abs().sum().item()), 0.0)

        metrics = {}
        for _ in range(int(round(cfg.cross_modal_tau_trace)) + 2):
            metrics = trainer.train_step(
                pattern,
                raw_window="expired trace cross modal",
                allow_sleep_maintenance=False,
            )

        self.assertEqual(metrics["cross_modal_idle_trace_reset_count"], 1)
        self.assertEqual(metrics["cross_modal_text_spike_prepared"], 0)
        self.assertEqual(metrics["cross_modal_text_execution_mode"], "text_idle_cached_state")
        self.assertEqual(float(trainer.model.cross_modal.text_trace.abs().sum().item()), 0.0)
        self.assertEqual(float(trainer.model.cross_modal.visual_trace.abs().sum().item()), 0.0)
        self.assertEqual(float(trainer.model.cross_modal.audio_trace.abs().sum().item()), 0.0)

    def test_cross_modal_text_probe_runs_with_residual_sensory_trace(self) -> None:
        trainer, cfg = self._trainer()
        pattern = torch.rand(cfg.input_dim)
        visual = torch.rand(cfg.cross_modal_dim_visual)

        trainer.train_step(
            pattern,
            raw_window="visual cross modal",
            visual_spikes=visual,
            allow_sleep_maintenance=False,
        )
        metrics = {}
        for _ in range(3):
            metrics = trainer.train_step(
                pattern,
                raw_window="residual trace cross modal",
                allow_sleep_maintenance=False,
            )

        report = trainer.model.cross_modal.device_report()
        self.assertEqual(report["runtime_text_update_count"], 1)
        self.assertEqual(report["runtime_text_idle_skip_count"], 3)
        self.assertEqual(metrics["cross_modal_text_update_count"], 1)
        self.assertEqual(metrics["cross_modal_text_idle_skip_count"], 3)
        self.assertEqual(metrics["cross_modal_fast_idle_skip_count"], 0)
        self.assertEqual(
            metrics["cross_modal_text_execution_mode"],
            "text_idle_cached_state",
        )

    def test_text_only_idle_skip_does_not_normalize_text_spike(self) -> None:
        trainer, cfg = self._trainer()
        trainer.config.cross_modal_text_idle_probe_interval_tokens = 100
        pattern = torch.rand(cfg.input_dim)

        trainer.train_step(
            pattern,
            raw_window="first due cross modal",
            allow_sleep_maintenance=False,
        )
        metrics = trainer.train_step(
            pattern,
            raw_window="idle skipped cross modal",
            allow_sleep_maintenance=False,
        )

        self.assertEqual(metrics["cross_modal_text_update_count"], 0)
        self.assertEqual(metrics["cross_modal_text_idle_skip_count"], 2)
        self.assertEqual(metrics["cross_modal_fast_idle_skip_count"], 2)
        self.assertEqual(metrics["cross_modal_text_spike_prepared"], 0)
        self.assertEqual(
            metrics["cross_modal_text_execution_mode"],
            "text_idle_cached_state",
        )

    def test_sensory_cross_modal_text_update_runs_every_tick(self) -> None:
        trainer, cfg = self._trainer()
        pattern = torch.rand(cfg.input_dim)
        visual = torch.rand(cfg.cross_modal_dim_visual)

        metrics = {}
        for _ in range(4):
            metrics = trainer.train_step(
                pattern,
                raw_window="visual cross modal",
                visual_spikes=visual,
                allow_sleep_maintenance=False,
            )

        report = trainer.model.cross_modal.device_report()
        self.assertEqual(report["runtime_text_update_count"], 4)
        self.assertEqual(report["runtime_text_idle_skip_count"], 0)
        self.assertEqual(report["last_text_runtime_execution_mode"], "text_update")
        self.assertTrue(metrics["cross_modal_visual_accepted"])
        self.assertEqual(metrics["cross_modal_fast_idle_skip_count"], 0)
        self.assertEqual(metrics["cross_modal_text_execution_mode"], "text_update")

    def test_config_rejects_nonpositive_cross_modal_text_probe_interval(self) -> None:
        from marulho.config.model_config import MarulhoConfig

        with self.assertRaisesRegex(
            ValueError,
            "cross_modal_text_idle_probe_interval_tokens",
        ):
            MarulhoConfig(cross_modal_text_idle_probe_interval_tokens=0)


class TestAlignmentFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.layer = CrossModalGroundingLayer(dim_text=10, dim_visual=16, dim_audio=8)

    def test_no_confidence_rejects(self) -> None:
        """With no grounding confidence, alignment gate should reject."""
        text = torch.randn(10).abs()
        visual = torch.randn(16).abs()
        accept, score = self.layer.alignment_gate(text, visual)
        self.assertFalse(accept)
        self.assertAlmostEqual(score, 0.0)

    def test_with_confidence_returns_score(self) -> None:
        """With artificial confidence, gate should compute a score."""
        self.layer.visual_confidence = torch.ones(10) * 0.5
        self.layer.W_tv = torch.eye(10, 16) * 0.5
        text = torch.zeros(10)
        text[0] = 1.0
        visual = torch.zeros(16)
        visual[0] = 1.0  # Matches prediction
        accept, score = self.layer.alignment_gate(text, visual, threshold=0.0)
        self.assertGreater(score, 0.0)

    def test_audio_gate(self) -> None:
        """Audio alignment gate with no confidence should reject."""
        text = torch.randn(10).abs()
        audio = torch.randn(8).abs()
        accept, score = self.layer.alignment_gate_audio(text, audio)
        self.assertFalse(accept)


class TestSerialization(unittest.TestCase):
    def test_state_dict_roundtrip(self) -> None:
        layer = CrossModalGroundingLayer(dim_text=10, dim_visual=16, dim_audio=8)
        # Modify state
        layer.on_visual_spike(torch.ones(16))
        layer.on_text_spike(torch.ones(10))

        state = layer.state_dict()
        layer2 = CrossModalGroundingLayer(dim_text=10, dim_visual=16, dim_audio=8)
        layer2.load_state_dict(state)

        self.assertTrue(torch.allclose(layer.W_tv, layer2.W_tv))
        self.assertTrue(torch.allclose(layer.visual_confidence, layer2.visual_confidence))


class TestReset(unittest.TestCase):
    def test_reset_clears_traces(self) -> None:
        layer = CrossModalGroundingLayer(dim_text=10, dim_visual=16, dim_audio=8)
        layer.on_visual_spike(torch.ones(16))
        layer.reset()
        self.assertEqual(layer.visual_trace.sum().item(), 0.0)
        self.assertEqual(layer.text_trace.sum().item(), 0.0)


class TestSelfCriticism(unittest.TestCase):
    """Tests for §7.4 self-criticism loop."""

    def _make_layer(self) -> CrossModalGroundingLayer:
        return CrossModalGroundingLayer(dim_text=10, dim_visual=16, dim_audio=8)

    def test_no_criticism_when_low_confidence(self) -> None:
        layer = self._make_layer()
        frames = [torch.rand(16) for _ in range(10)]
        result = layer.run_self_criticism(frames)
        self.assertEqual(result["checked"], 0)
        self.assertEqual(result["penalised"], 0)

    def test_penalises_wrong_high_confidence(self) -> None:
        layer = self._make_layer()
        # Set high confidence on dim 0 and a strong W_tv association
        layer.visual_confidence[0] = 0.9
        layer.W_tv[0] = torch.randn(16)
        # Use zero frames — no alignment possible
        frames = [torch.zeros(16) for _ in range(10)]
        result = layer.run_self_criticism(frames, alignment_floor=0.2)
        self.assertEqual(result["checked"], 1)
        self.assertEqual(result["penalised"], 1)
        self.assertLess(float(layer.visual_confidence[0]), 0.9)

    def test_blacklist_after_repeated_penalties(self) -> None:
        layer = self._make_layer()
        layer.visual_confidence[0] = 0.9
        layer.W_tv[0] = torch.randn(16)
        frames = [torch.zeros(16) for _ in range(10)]
        blacklist: dict[int, int] = {}
        # First pass — should penalise but not blacklist
        layer.visual_confidence[0] = 0.9
        layer.run_self_criticism(frames, blacklist=blacklist, blacklist_strikes=2)
        self.assertEqual(blacklist.get(0, 0), 1)
        # Second pass — should blacklist and zero weights
        layer.visual_confidence[0] = 0.9
        result = layer.run_self_criticism(frames, blacklist=blacklist, blacklist_strikes=2)
        self.assertEqual(result["blacklisted"], 1)
        self.assertEqual(float(layer.visual_confidence[0]), 0.0)
        self.assertEqual(float(layer.W_tv[0].norm()), 0.0)

    def test_no_penalty_when_alignment_found(self) -> None:
        layer = self._make_layer()
        layer.visual_confidence[0] = 0.9
        direction = torch.randn(16)
        layer.W_tv[0] = direction
        # Provide a frame that closely matches the prediction
        frames = [direction.clone() for _ in range(10)]
        result = layer.run_self_criticism(frames)
        self.assertEqual(result["penalised"], 0)
        self.assertAlmostEqual(float(layer.visual_confidence[0]), 0.9, places=3)


if __name__ == "__main__":
    unittest.main()
