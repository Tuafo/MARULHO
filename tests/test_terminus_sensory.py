from __future__ import annotations

import io
from threading import Event
import time
from unittest.mock import patch
import wave

import numpy as np
from PIL import Image
import torch

from hecsn.service import terminus_sensory as sensory_module
from hecsn.service.terminus_sensory import (
    AudioCapsSensoryStream,
    S1MMAlignSensoryStream,
    bootstrap_sensory_episode_from_row,
    build_sensory_stream,
)


def _png_bytes() -> bytes:
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:24, 8:24] = 255
    buffer = io.BytesIO()
    Image.fromarray(image, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def _wav_bytes(sample_rate: int = 16000) -> bytes:
    t = np.linspace(0, 0.2, int(sample_rate * 0.2), dtype=np.float32)
    samples = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    ints = (samples * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(ints.tobytes())
    return buffer.getvalue()


def _reset_s1_recaption_runtime() -> None:
    sensory_module._reset_s1_recaption_index_runtime()


def test_s1_mmalign_stream_yields_visual_episode() -> None:
    _reset_s1_recaption_runtime()
    row = {
        "png": {
            "bytes": _png_bytes(),
            "path": "0705/0501163.tar.gz/fig004.png",
        }
    }
    recaption_index = {
        "images/0705/0501163.tar.gz/fig004.png": {
            "title": "Phase ordering and symmetries of the Potts model",
            "recaption": "A lattice image with two sharply separated regions.",
            "categories": "cond-mat.stat-mech",
        }
    }
    with patch("hecsn.service.terminus_sensory._load_hf_stream", return_value=[row]):
        with patch("hecsn.service.terminus_sensory._load_s1_recaption_index", return_value=recaption_index):
            episode = next(iter(S1MMAlignSensoryStream(year_prefixes=("07",), target_dim=64)))

    assert "Potts model" in episode.text
    assert "lattice image" in episode.text
    assert episode.audio_spikes is None
    assert episode.visual_spikes is not None
    assert episode.visual_spikes.shape == (64,)
    assert float(episode.visual_spikes.sum().item()) > 0.0
    assert episode.metadata["adapter"] == "s1_mmalign"
    assert episode.metadata["device"] == "cpu"
    assert episode.metadata["encoder"]["encoder"] == "event_camera"
    assert episode.metadata["encoder"]["device"] == "cpu"
    assert episode.metadata["spike_device"] == "cpu"
    assert episode.metadata["spike_is_cuda"] is False
    assert episode.visual_preview is not None
    assert episode.visual_preview["mime_type"] == "image/png"
    assert episode.visual_preview["width"] > 0
    assert episode.visual_preview["height"] > 0


def test_audiocaps_stream_yields_audio_episode() -> None:
    row = {
        "audiocap_id": 7,
        "youtube_id": "abc123xyz99",
        "start_time": 130,
        "caption": "Water pours while a woman talks nearby",
        "audio": {"bytes": _wav_bytes()},
    }
    with patch("hecsn.service.terminus_sensory._load_hf_stream", return_value=[row]):
        episode = next(iter(AudioCapsSensoryStream(target_dim=64, sample_rate=16000, n_fft=512)))

    assert episode.visual_spikes is None
    assert episode.audio_spikes is not None
    assert episode.audio_spikes.shape == (64,)
    assert float(episode.audio_spikes.sum().item()) > 0.0
    assert "woman talks" in episode.text
    assert episode.metadata["adapter"] == "audiocaps"
    assert episode.metadata["device"] == "cpu"
    assert episode.metadata["encoder"]["encoder"] == "cochleagram"
    assert episode.metadata["encoder"]["device"] == "cpu"
    assert episode.metadata["spike_device"] == "cpu"
    assert episode.metadata["spike_is_cuda"] is False
    assert episode.audio_preview is not None
    assert episode.audio_preview["mime_type"] == "audio/wav"
    assert episode.audio_preview["duration_s"] > 0.0
    assert len(episode.audio_preview["waveform"]) == 64


def test_s1_stream_requests_only_png_column() -> None:
    _reset_s1_recaption_runtime()
    row = {
        "png": {
            "bytes": _png_bytes(),
            "path": "0705/0501163.tar.gz/fig004.png",
        }
    }
    recaption_index = {
        "images/0705/0501163.tar.gz/fig004.png": {
            "title": "Phase ordering",
            "recaption": "A lattice image with two sharply separated regions.",
            "categories": "cond-mat.stat-mech",
        }
    }
    with patch("hecsn.service.terminus_sensory._load_hf_stream", return_value=[row]) as mocked:
        with patch("hecsn.service.terminus_sensory._load_s1_recaption_index", return_value=recaption_index):
            episode = next(iter(S1MMAlignSensoryStream(year_prefixes=("07",), target_dim=64)))

    assert episode.visual_spikes is not None
    assert mocked.call_args.kwargs["columns"] == ["png"]


def test_audiocaps_stream_requests_only_needed_columns() -> None:
    row = {
        "audiocap_id": 7,
        "youtube_id": "abc123xyz99",
        "start_time": 130,
        "caption": "Water pours while a woman talks nearby",
        "audio": {"bytes": _wav_bytes()},
    }
    with patch("hecsn.service.terminus_sensory._load_hf_stream", return_value=[row]) as mocked:
        episode = next(iter(AudioCapsSensoryStream(target_dim=64, sample_rate=16000, n_fft=512)))

    assert episode.audio_spikes is not None
    assert mocked.call_args.kwargs["columns"] == ["caption", "audio", "youtube_id", "audiocap_id", "start_time"]


def test_s1_bootstrap_episode_uses_first_rows_asset_shape() -> None:
    _reset_s1_recaption_runtime()
    row = {
        "png": {"src": "https://example.com/image.jpg"},
        "__key__": "0705/0501163.tar.gz/fig004",
    }
    recaption_index = {
        "images/0705/0501163.tar.gz/fig004.png": {
            "title": "Phase ordering and symmetries of the Potts model",
            "recaption": "A lattice image with two sharply separated regions.",
            "categories": "cond-mat.stat-mech",
        }
    }
    with patch("hecsn.service.terminus_sensory._download_binary_asset", return_value=_png_bytes()):
        with patch("hecsn.service.terminus_sensory._load_s1_recaption_index", return_value=recaption_index):
            episode = bootstrap_sensory_episode_from_row(
                {
                    "adapter": "s1_mmalign",
                    "source": "ScienceOne-AI/S1-MMAlign",
                    "year_prefixes": ["07"],
                    "max_text_chars": 480,
                },
                row,
                visual_dim=64,
                audio_dim=64,
            )

    assert episode is not None
    assert episode.visual_spikes is not None
    assert "Potts model" in episode.text


def test_s1_bootstrap_episode_falls_back_while_recaption_index_loads() -> None:
    _reset_s1_recaption_runtime()
    row = {
        "png": {"src": "https://example.com/image.jpg"},
        "__key__": "0705/0501163.tar.gz/fig004",
    }
    recaption_index = {
        "images/0705/0501163.tar.gz/fig004.png": {
            "title": "Phase ordering and symmetries of the Potts model",
            "recaption": "A lattice image with two sharply separated regions.",
            "categories": "cond-mat.stat-mech",
        }
    }
    loaded = Event()

    def _slow_index_loader(*_args, **_kwargs):
        time.sleep(0.35)
        loaded.set()
        return recaption_index

    with patch("hecsn.service.terminus_sensory._download_binary_asset", return_value=_png_bytes()):
        with patch("hecsn.service.terminus_sensory._load_s1_recaption_index", side_effect=_slow_index_loader):
            episode = bootstrap_sensory_episode_from_row(
                {
                    "adapter": "s1_mmalign",
                    "source": "ScienceOne-AI/S1-MMAlign",
                    "year_prefixes": ["07"],
                    "max_text_chars": 480,
                },
                row,
                visual_dim=64,
                audio_dim=64,
            )
            assert episode is not None
            assert episode.visual_spikes is not None
            assert episode.metadata["text_source"] == "image_path_fallback"
            assert episode.metadata["metadata_pending"] is True
            assert "Scientific figure fig004" in episode.text
            assert loaded.wait(1.0)

    _reset_s1_recaption_runtime()


def test_s1_stream_uses_recaption_metadata_after_background_index_load() -> None:
    _reset_s1_recaption_runtime()
    row = {
        "png": {
            "bytes": _png_bytes(),
            "path": "0705/0501163.tar.gz/fig004.png",
        }
    }
    recaption_index = {
        "images/0705/0501163.tar.gz/fig004.png": {
            "title": "Phase ordering and symmetries of the Potts model",
            "recaption": "A lattice image with two sharply separated regions.",
            "categories": "cond-mat.stat-mech",
        }
    }
    loaded = Event()

    def _slow_index_loader(*_args, **_kwargs):
        time.sleep(0.35)
        loaded.set()
        return recaption_index

    with patch("hecsn.service.terminus_sensory._load_hf_stream", return_value=[row, row]):
        with patch("hecsn.service.terminus_sensory._load_s1_recaption_index", side_effect=_slow_index_loader):
            stream = iter(S1MMAlignSensoryStream(year_prefixes=("07",), target_dim=64))
            first = next(stream)
            assert first.metadata["text_source"] == "image_path_fallback"
            assert loaded.wait(1.0)
            second = next(stream)

    assert second.metadata["text_source"] == "recaption_index"
    assert second.metadata["metadata_pending"] is False
    assert "Potts model" in second.text
    assert "lattice image" in second.text
    _reset_s1_recaption_runtime()


def test_s1_bootstrap_passes_timeout_to_asset_download() -> None:
    _reset_s1_recaption_runtime()
    row = {
        "png": {"src": "https://example.com/image.jpg"},
        "__key__": "0705/0501163.tar.gz/fig004",
    }
    recaption_index = {
        "images/0705/0501163.tar.gz/fig004.png": {
            "title": "Phase ordering and symmetries of the Potts model",
            "recaption": "A lattice image with two sharply separated regions.",
            "categories": "cond-mat.stat-mech",
        }
    }
    captured: dict[str, float] = {}

    def _download(url: str, *, timeout_seconds: float = 20.0) -> bytes:
        captured["timeout_seconds"] = float(timeout_seconds)
        return _png_bytes()

    with patch("hecsn.service.terminus_sensory._download_binary_asset", side_effect=_download):
        with patch("hecsn.service.terminus_sensory._load_s1_recaption_index", return_value=recaption_index):
            episode = bootstrap_sensory_episode_from_row(
                {
                    "adapter": "s1_mmalign",
                    "source": "ScienceOne-AI/S1-MMAlign",
                    "year_prefixes": ["07"],
                    "max_text_chars": 480,
                },
                row,
                visual_dim=64,
                audio_dim=64,
                timeout_seconds=0.25,
            )

    assert episode is not None
    assert captured["timeout_seconds"] == 0.25
    _reset_s1_recaption_runtime()


def test_audiocaps_bootstrap_episode_uses_first_rows_asset_shape() -> None:
    row = {
        "audiocap_id": 7,
        "youtube_id": "abc123xyz99",
        "start_time": 130,
        "caption": "Water pours while a woman talks nearby",
        "audio": [{"src": "https://example.com/audio.wav", "type": "audio/wav"}],
    }
    with patch("hecsn.service.terminus_sensory._download_binary_asset", return_value=_wav_bytes()):
        episode = bootstrap_sensory_episode_from_row(
            {
                "adapter": "audiocaps",
                "source": "OpenSound/AudioCaps",
                "sample_rate": 16000,
                "n_fft": 512,
                "max_text_chars": 240,
                "audio_candidates_per_item": 6,
            },
            row,
            visual_dim=64,
            audio_dim=64,
        )

    assert episode is not None
    assert episode.audio_spikes is not None
    assert "woman talks" in episode.text


def test_audiocaps_bootstrap_passes_timeout_to_asset_download() -> None:
    row = {
        "audiocap_id": 7,
        "youtube_id": "abc123xyz99",
        "start_time": 130,
        "caption": "Water pours while a woman talks nearby",
        "audio": [{"src": "https://example.com/audio.wav", "type": "audio/wav"}],
    }
    captured: dict[str, float] = {}

    def _download(url: str, *, timeout_seconds: float = 20.0) -> bytes:
        captured["timeout_seconds"] = float(timeout_seconds)
        return _wav_bytes()

    with patch("hecsn.service.terminus_sensory._download_binary_asset", side_effect=_download):
        episode = bootstrap_sensory_episode_from_row(
            {
                "adapter": "audiocaps",
                "source": "OpenSound/AudioCaps",
                "sample_rate": 16000,
                "n_fft": 512,
                "max_text_chars": 240,
                "audio_candidates_per_item": 6,
            },
            row,
            visual_dim=64,
            audio_dim=64,
            timeout_seconds=0.25,
        )

    assert episode is not None
    assert captured["timeout_seconds"] == 0.25


def test_build_sensory_stream_is_lazy() -> None:
    _reset_s1_recaption_runtime()
    spec = {
        "name": "science_figures",
        "adapter": "s1_mmalign",
        "source": "ScienceOne-AI/S1-MMAlign",
        "split": "train",
        "year_prefixes": ["07"],
    }
    row = {
        "png": {
            "bytes": _png_bytes(),
            "path": "0705/0501163.tar.gz/fig004.png",
        }
    }
    recaption_index = {
        "images/0705/0501163.tar.gz/fig004.png": {
            "title": "Phase ordering",
            "recaption": "A lattice image with two sharply separated regions.",
            "categories": "cond-mat.stat-mech",
        }
    }
    with patch("hecsn.service.terminus_sensory._load_hf_stream", return_value=[row]) as mocked:
        with patch("hecsn.service.terminus_sensory._load_s1_recaption_index", return_value=recaption_index):
            stream = build_sensory_stream(spec, visual_dim=64, audio_dim=64)
            assert mocked.call_count == 0
            episode = next(stream)
            assert mocked.call_count == 1
            assert episode.visual_spikes is not None


def test_default_sensory_device_prefers_cuda_when_available(monkeypatch) -> None:
    monkeypatch.delenv("HECSN_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert sensory_module._resolve_sensory_device(None) == torch.device("cuda")
