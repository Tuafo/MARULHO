from __future__ import annotations

import io
from unittest.mock import patch
import wave

import numpy as np
from PIL import Image

from hecsn.service.terminus_sensory import (
    AudioCapsSensoryStream,
    S1MMAlignSensoryStream,
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


def test_s1_mmalign_stream_yields_visual_episode() -> None:
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
    assert episode.audio_preview is not None
    assert episode.audio_preview["mime_type"] == "audio/wav"
    assert episode.audio_preview["duration_s"] > 0.0
    assert len(episode.audio_preview["waveform"]) == 64


def test_build_sensory_stream_is_lazy() -> None:
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
