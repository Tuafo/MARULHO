"""Tests for CochleagramEncoder (§5.1)."""

from __future__ import annotations

import unittest

import torch

from hecsn.data.cochleagram_encoder import CochleagramEncoder, _mel_filterbank


class TestMelFilterbank(unittest.TestCase):
    def test_shape(self) -> None:
        fb = _mel_filterbank(n_bands=64, n_fft=512, sample_rate=16000)
        self.assertEqual(fb.shape, (64, 257))  # n_fft//2 + 1 = 257

    def test_nonnegative(self) -> None:
        fb = _mel_filterbank(n_bands=32, n_fft=256, sample_rate=8000)
        self.assertTrue((fb >= 0).all())

    def test_each_band_has_nonzero(self) -> None:
        fb = _mel_filterbank(n_bands=16, n_fft=512, sample_rate=16000)
        for i in range(16):
            self.assertGreater(fb[i].sum().item(), 0.0, f"Band {i} is all zeros")


class TestCochleagramEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.enc = CochleagramEncoder(n_bands=64, n_fft=512, sample_rate=16000)

    def test_output_dim(self) -> None:
        self.assertEqual(self.enc.output_dim, 64)

    def test_device_report_exposes_encoder_state_devices(self) -> None:
        report = self.enc.device_report()
        self.assertEqual(report["encoder"], "cochleagram")
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["filterbank_device"], "cpu")
        self.assertEqual(report["baseline_device"], "cpu")
        self.assertEqual(report["trace_device"], "cpu")
        self.assertIsNone(report["last_spike_device"])
        self.assertIsNone(report["last_spike_shape"])
        self.enc.encode(torch.zeros(512))
        report = self.enc.device_report()
        self.assertEqual(report["last_spike_device"], "cpu")
        self.assertEqual(report["last_spike_shape"], (64,))

    def test_encode_pure_tone(self) -> None:
        """A 1 kHz tone should activate some bands."""
        t = torch.linspace(0, 0.032, 512)  # ~32ms at 16kHz
        waveform = torch.sin(2 * 3.14159 * 1000 * t)
        # Feed a few frames to build baseline, then check spikes
        self.enc.encode(torch.zeros(512))  # silent baseline
        self.enc.encode(torch.zeros(512))
        spikes = self.enc.encode(waveform)
        self.assertEqual(spikes.shape, (64,))
        self.assertGreater(spikes.sum().item(), 0.0, "Tone should produce spikes")

    def test_silence_after_warmup_no_spikes(self) -> None:
        """After baseline adapts to silence, silence should produce no spikes."""
        for _ in range(20):
            self.enc.encode(torch.zeros(512))
        spikes = self.enc.encode(torch.zeros(512))
        self.assertEqual(spikes.sum().item(), 0.0)

    def test_sparsity_range(self) -> None:
        """Sparsity for a tone should be in a reasonable range."""
        t = torch.linspace(0, 0.032, 512)
        tone = torch.sin(2 * 3.14159 * 440 * t) * 0.5
        self.enc.encode(torch.zeros(512))
        spikes = self.enc.encode(tone)
        s = self.enc.sparsity(spikes)
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_trace_accumulates(self) -> None:
        t = torch.linspace(0, 0.032, 512)
        tone = torch.sin(2 * 3.14159 * 1000 * t) * 0.8
        self.enc.encode(torch.zeros(512))
        self.enc.encode(tone)
        self.assertGreater(self.enc.trace.sum().item(), 0.0)

    def test_reset_clears_state(self) -> None:
        t = torch.linspace(0, 0.032, 512)
        tone = torch.sin(2 * 3.14159 * 1000 * t)
        self.enc.encode(torch.zeros(512))
        self.enc.encode(tone)
        self.enc.reset()
        self.assertEqual(self.enc.trace.sum().item(), 0.0)

    def test_short_waveform_padded(self) -> None:
        """Waveforms shorter than n_fft should be zero-padded."""
        short = torch.randn(100)
        spikes = self.enc.encode(short)
        self.assertEqual(spikes.shape, (64,))

    def test_binary_output(self) -> None:
        t = torch.linspace(0, 0.032, 512)
        tone = torch.sin(2 * 3.14159 * 800 * t)
        self.enc.encode(torch.zeros(512))
        spikes = self.enc.encode(tone)
        for v in torch.unique(spikes):
            self.assertIn(v.item(), [0.0, 1.0])

    def test_state_dict_roundtrip(self) -> None:
        t = torch.linspace(0, 0.032, 512)
        self.enc.encode(torch.sin(2 * 3.14159 * 440 * t))
        state = self.enc.state_dict()
        enc2 = CochleagramEncoder(n_bands=64, n_fft=512, sample_rate=16000)
        enc2.load_state_dict(state)
        self.assertTrue(torch.allclose(enc2._baseline, self.enc._baseline))

    def test_different_n_bands(self) -> None:
        enc = CochleagramEncoder(n_bands=32, n_fft=256, sample_rate=8000)
        self.assertEqual(enc.output_dim, 32)
        spikes = enc.encode(torch.randn(256))
        self.assertEqual(spikes.shape, (32,))


if __name__ == "__main__":
    unittest.main()
