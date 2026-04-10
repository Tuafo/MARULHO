"""Tests for CrossModalGroundingLayer (§5.1–§5.3)."""

from __future__ import annotations

import unittest

import torch

from hecsn.core.cross_modal import CrossModalGroundingLayer


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


if __name__ == "__main__":
    unittest.main()
