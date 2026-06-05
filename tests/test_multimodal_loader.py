"""Tests for MultimodalStreamLoader."""

from __future__ import annotations

import torch
import pytest

from marulho.data import multimodal_loader as multimodal_module
from marulho.data.multimodal_loader import MultimodalSample, MultimodalStreamLoader, load_directory
from marulho.data.event_camera_encoder import EventCameraEncoder
from marulho.data.cochleagram_encoder import CochleagramEncoder


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

    def test_source_tensors_move_to_loader_device(self):
        frame = torch.rand(16, 16)
        audio = torch.randn(256)
        loader = MultimodalStreamLoader(
            text_source=_text_iter("abcde"),
            window_size=5,
            visual_source=iter([frame]),
            audio_source=iter([audio]),
            device=torch.device("cpu"),
        )

        sample = next(iter(loader))

        assert sample.visual_frame is not None
        assert sample.audio_chunk is not None
        assert sample.visual_frame.device == torch.device("cpu")
        assert sample.audio_chunk.device == torch.device("cpu")

    def test_directory_loader_maps_saved_tensors_to_runtime_device(self, tmp_path):
        (tmp_path / "text").mkdir()
        (tmp_path / "visual").mkdir()
        (tmp_path / "audio").mkdir()
        (tmp_path / "text" / "a.txt").write_text("abcde", encoding="utf-8")
        torch.save(torch.rand(1, 16, 16), tmp_path / "visual" / "a.pt")
        torch.save(torch.randn(1, 256), tmp_path / "audio" / "a.pt")

        loader = load_directory(tmp_path, window_size=5, device=torch.device("cpu"))
        sample = next(iter(loader))

        assert sample.visual_frame is not None
        assert sample.audio_chunk is not None
        assert sample.visual_frame.device == torch.device("cpu")
        assert sample.audio_chunk.device == torch.device("cpu")

    def test_default_multimodal_device_prefers_cuda_when_available(self, monkeypatch):
        monkeypatch.delenv("MARULHO_DEVICE", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        assert multimodal_module._resolve_multimodal_device(None) == torch.device("cuda")
