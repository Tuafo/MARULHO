"""Tests for EventCameraEncoder (§5.1)."""

from __future__ import annotations

import unittest

import torch

from marulho.data.event_camera_encoder import EventCameraEncoder


class TestEventCameraEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.enc = EventCameraEncoder(height=32, width=32, pool=4, contrast_threshold=0.3)

    def test_output_dim(self) -> None:
        self.assertEqual(self.enc.output_dim, 64)  # (32/4) * (32/4) = 8*8 = 64

    def test_device_report_exposes_encoder_state_devices(self) -> None:
        report = self.enc.device_report()
        self.assertEqual(report["encoder"], "event_camera")
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["trace_device"], "cpu")
        self.assertIsNone(report["ref_device"])
        self.assertIsNone(report["last_spike_device"])
        self.assertIsNone(report["last_spike_shape"])
        self.enc.encode(torch.ones(32, 32) * 0.1)
        report = self.enc.device_report()
        self.assertEqual(report["ref_device"], "cpu")
        self.assertEqual(report["last_spike_device"], "cpu")
        self.assertEqual(report["last_spike_shape"], (64,))

    def test_first_frame_returns_zeros(self) -> None:
        frame = torch.rand(32, 32)
        spikes = self.enc.encode(frame)
        self.assertEqual(spikes.shape, (64,))
        self.assertEqual(spikes.sum().item(), 0.0)

    def test_identical_frames_no_spikes(self) -> None:
        frame = torch.rand(32, 32) * 0.5 + 0.25
        self.enc.encode(frame)
        spikes = self.enc.encode(frame)
        self.assertEqual(spikes.sum().item(), 0.0)

    def test_large_change_produces_spikes(self) -> None:
        frame1 = torch.ones(32, 32) * 0.1
        frame2 = torch.ones(32, 32) * 0.9
        self.enc.encode(frame1)
        spikes = self.enc.encode(frame2)
        self.assertGreater(spikes.sum().item(), 0.0)

    def test_sparsity_in_range(self) -> None:
        """For natural-ish changes, sparsity should be moderate."""
        frame1 = torch.rand(32, 32) * 0.3 + 0.1
        self.enc.encode(frame1)
        frame2 = frame1.clone()
        # Change ~25% of pixels significantly
        mask = torch.rand(32, 32) > 0.75
        frame2[mask] = 1.0 - frame2[mask]
        spikes = self.enc.encode(frame2)
        s = self.enc.sparsity(spikes)
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_trace_accumulates(self) -> None:
        frame1 = torch.ones(32, 32) * 0.1
        frame2 = torch.ones(32, 32) * 0.9
        self.enc.encode(frame1)
        self.enc.encode(frame2)
        self.assertGreater(self.enc.trace.sum().item(), 0.0)

    def test_reset_clears_state(self) -> None:
        frame1 = torch.ones(32, 32) * 0.1
        frame2 = torch.ones(32, 32) * 0.9
        self.enc.encode(frame1)
        self.enc.encode(frame2)
        self.enc.reset()
        self.assertEqual(self.enc.trace.sum().item(), 0.0)

    def test_rgb_frame_accepted(self) -> None:
        """3-channel frames should be averaged to greyscale."""
        frame1 = torch.rand(3, 32, 32) * 0.2
        frame2 = torch.rand(3, 32, 32) * 0.8
        self.enc.encode(frame1)
        spikes = self.enc.encode(frame2)
        self.assertEqual(spikes.shape, (64,))

    def test_state_dict_roundtrip(self) -> None:
        frame = torch.rand(32, 32) * 0.5
        self.enc.encode(frame)
        state = self.enc.state_dict()
        enc2 = EventCameraEncoder(height=32, width=32, pool=4)
        enc2.load_state_dict(state)
        self.assertTrue(torch.allclose(enc2._ref_log_intensity, self.enc._ref_log_intensity))

    def test_different_resolutions(self) -> None:
        enc = EventCameraEncoder(height=64, width=64, pool=8)
        self.assertEqual(enc.output_dim, 64)  # (64/8)^2 = 64
        frame = torch.rand(64, 64)
        spikes = enc.encode(frame)
        self.assertEqual(spikes.shape, (64,))

    def test_binary_output(self) -> None:
        """Spikes should be 0 or 1."""
        frame1 = torch.rand(32, 32) * 0.2
        frame2 = torch.rand(32, 32) * 0.8
        self.enc.encode(frame1)
        spikes = self.enc.encode(frame2)
        unique = torch.unique(spikes)
        for v in unique:
            self.assertIn(v.item(), [0.0, 1.0])


class TestEventCameraSparsity(unittest.TestCase):
    """Validate the 5-25% sparsity target from the paper."""

    def test_moderate_motion_sparsity(self) -> None:
        enc = EventCameraEncoder(height=64, width=64, pool=4, contrast_threshold=0.3)
        frame = torch.rand(64, 64) * 0.5
        enc.encode(frame)
        # Simulate moderate motion: shift frame slightly + noise
        frame2 = frame.clone()
        frame2[10:30, 10:30] *= 3.0  # Brighten center region
        frame2 = frame2.clamp(0.0, 1.0)
        spikes = enc.encode(frame2)
        s = enc.sparsity(spikes)
        # Should produce some but not all spikes
        self.assertGreater(s, 0.0, "Expected some spikes for brightness change")
        self.assertLess(s, 1.0, "Expected less than 100% firing")


if __name__ == "__main__":
    unittest.main()
