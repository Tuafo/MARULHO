"""Tests for MultimodalStreamLoader."""

from __future__ import annotations

import torch
import pytest

from hecsn.data.multimodal_loader import MultimodalSample, MultimodalStreamLoader
from hecsn.data.event_camera_encoder import EventCameraEncoder
from hecsn.data.cochleagram_encoder import CochleagramEncoder


def _text_iter(text: str):
    yield text


class TestMultimodalStreamLoader:
    def test_text_only_yields_samples(self):
        loader = MultimodalStreamLoader(
            text_source=_text_iter("hello world"),
            window_size=5,
        )
        samples = list(loader)
        assert len(samples) == 2
        assert samples[0].text == "hello"
        assert samples[1].text == " worl"
        assert samples[0].visual_frame is None
        assert samples[0].audio_chunk is None

    def test_synthetic_visual_produces_frames(self):
        enc = EventCameraEncoder(height=16, width=16, pool=4)
        loader = MultimodalStreamLoader(
            text_source=_text_iter("abcdefghij"),
            window_size=5,
            visual_encoder=enc,
            synthetic=True,
        )
        samples = list(loader)
        assert len(samples) == 2
        for s in samples:
            assert s.visual_frame is not None
            assert s.visual_frame.shape == (16, 16)

    def test_synthetic_audio_produces_chunks(self):
        enc = CochleagramEncoder(n_bands=32, n_fft=256)
        loader = MultimodalStreamLoader(
            text_source=_text_iter("abcdefghij"),
            window_size=5,
            audio_encoder=enc,
            synthetic=True,
        )
        samples = list(loader)
        assert len(samples) == 2
        for s in samples:
            assert s.audio_chunk is not None
            assert s.audio_chunk.shape == (256,)

    def test_synthetic_multimodal(self):
        v_enc = EventCameraEncoder(height=16, width=16, pool=4)
        a_enc = CochleagramEncoder(n_bands=32, n_fft=256)
        loader = MultimodalStreamLoader(
            text_source=_text_iter("hello world!!!!"),
            window_size=5,
            visual_encoder=v_enc,
            audio_encoder=a_enc,
            synthetic=True,
        )
        samples = list(loader)
        assert len(samples) == 3
        for s in samples:
            assert len(s.text) == 5
            assert s.visual_frame is not None
            assert s.audio_chunk is not None

    def test_explicit_visual_source(self):
        frames = [torch.rand(16, 16) for _ in range(3)]
        loader = MultimodalStreamLoader(
            text_source=_text_iter("abcdefghijklmno"),
            window_size=5,
            visual_source=iter(frames),
        )
        samples = list(loader)
        assert len(samples) == 3
        assert torch.equal(samples[0].visual_frame, frames[0])
        assert torch.equal(samples[2].visual_frame, frames[2])

    def test_visual_source_exhaustion_returns_none(self):
        frames = [torch.rand(16, 16)]
        loader = MultimodalStreamLoader(
            text_source=_text_iter("abcdefghij"),
            window_size=5,
            visual_source=iter(frames),
        )
        samples = list(loader)
        assert len(samples) == 2
        assert samples[0].visual_frame is not None
        assert samples[1].visual_frame is None

    def test_multiple_text_blocks(self):
        def multi_text():
            yield "abc"
            yield "defgh"
            yield "ij"

        loader = MultimodalStreamLoader(
            text_source=multi_text(),
            window_size=5,
        )
        samples = list(loader)
        assert len(samples) == 2
        assert samples[0].text == "abcde"
        assert samples[1].text == "fghij"

    def test_window_larger_than_text_yields_nothing(self):
        loader = MultimodalStreamLoader(
            text_source=_text_iter("hi"),
            window_size=10,
        )
        samples = list(loader)
        assert len(samples) == 0

    def test_sample_slots(self):
        s = MultimodalSample(text="abc", visual_frame=None, audio_chunk=None)
        assert s.text == "abc"
        assert s.visual_frame is None
        assert s.audio_chunk is None

    def test_encoder_integration_synthetic(self):
        """Verify synthetic frames can be encoded end-to-end."""
        v_enc = EventCameraEncoder(height=16, width=16, pool=4)
        a_enc = CochleagramEncoder(n_bands=32, n_fft=256)
        loader = MultimodalStreamLoader(
            text_source=_text_iter("abcdefghij"),
            window_size=5,
            visual_encoder=v_enc,
            audio_encoder=a_enc,
            synthetic=True,
        )
        for sample in loader:
            v_spikes = v_enc.encode(sample.visual_frame)
            assert v_spikes.shape == (v_enc.output_dim,)
            a_spikes = a_enc.encode(sample.audio_chunk)
            assert a_spikes.shape == (a_enc.output_dim,)
