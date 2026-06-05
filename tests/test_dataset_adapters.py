"""Tests for multimodal dataset adapters (§5.4).

Uses synthetic test fixtures — no external dataset downloads required.
Generates small N-MNIST-format binary files and WAV files to verify
the adapter loading, binning, chunking, and pairing logic.
"""

from __future__ import annotations

import struct
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from marulho.data.dataset_adapters import (
    DIGIT_NAMES,
    DigitEpisode,
    Event,
    FSDDAdapter,
    MultimodalStep,
    NMNISTAdapter,
    PairedDigitDataset,
    _read_nmnist_bin,
    _read_wav,
    _resample_linear,
    events_to_frames,
    iter_episode_steps,
    validate_encoder_dims,
)
from marulho.data.event_camera_encoder import EventCameraEncoder
from marulho.data.cochleagram_encoder import CochleagramEncoder


# ---------------------------------------------------------------------------
# Fixture helpers: generate synthetic dataset files
# ---------------------------------------------------------------------------

def _write_nmnist_bin(path: Path, events: list[tuple[int, int, int, int]]) -> None:
    """Write events in N-MNIST 5-byte-per-event binary format.

    Each event is (x, y, timestamp_us, polarity).
    """
    with open(path, "wb") as f:
        for x, y, t, p in events:
            ts_and_pol = (t << 1) | (p & 1)
            b2 = (ts_and_pol >> 16) & 0xFF
            b3 = (ts_and_pol >> 8) & 0xFF
            b4 = ts_and_pol & 0xFF
            f.write(bytes([x & 0xFF, y & 0xFF, b2, b3, b4]))


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 8000) -> None:
    """Write a mono WAV file from float32 samples in [-1, 1]."""
    int_samples = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())


@pytest.fixture
def nmnist_dir(tmp_path: Path) -> Path:
    """Create a minimal N-MNIST directory with 2 samples per digit."""
    for split in ["Train", "Test"]:
        for digit in range(10):
            digit_dir = tmp_path / split / str(digit)
            digit_dir.mkdir(parents=True)
            for sample_idx in range(2):
                events = []
                # Generate 50 synthetic events per sample
                for i in range(50):
                    x = (digit * 3 + i) % 34
                    y = (digit * 5 + i * 2) % 34
                    t = i * 1000 + sample_idx * 100
                    p = i % 2
                    events.append((x, y, t, p))
                _write_nmnist_bin(
                    digit_dir / f"{sample_idx:05d}.bin", events
                )
    return tmp_path


@pytest.fixture
def fsdd_dir(tmp_path: Path) -> Path:
    """Create a minimal FSDD directory with 2 samples per digit."""
    rec_dir = tmp_path / "recordings"
    rec_dir.mkdir()
    for digit in range(10):
        for speaker_idx, speaker in enumerate(["alice", "bob"]):
            # Generate a short sine wave at a frequency related to the digit
            freq = 200 + digit * 100  # 200-1100 Hz
            duration = 0.5  # 500ms
            sr = 8000
            t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
            samples = 0.8 * np.sin(2 * np.pi * freq * t).astype(np.float32)
            _write_wav(rec_dir / f"{digit}_{speaker}_{speaker_idx}.wav", samples, sr)
    return tmp_path


# ---------------------------------------------------------------------------
# N-MNIST binary format tests
# ---------------------------------------------------------------------------

class TestNMNISTBinaryFormat:
    def test_read_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.bin"
        path.write_bytes(b"")
        events = _read_nmnist_bin(path)
        assert events == []

    def test_read_single_event(self, tmp_path: Path) -> None:
        path = tmp_path / "single.bin"
        _write_nmnist_bin(path, [(10, 20, 5000, 1)])
        events = _read_nmnist_bin(path)
        assert len(events) == 1
        assert events[0].x == 10
        assert events[0].y == 20
        assert events[0].t == 5000
        assert events[0].p == 1

    def test_read_multiple_events(self, tmp_path: Path) -> None:
        path = tmp_path / "multi.bin"
        input_events = [(0, 0, 100, 0), (33, 33, 200, 1), (17, 17, 300, 0)]
        _write_nmnist_bin(path, input_events)
        events = _read_nmnist_bin(path)
        assert len(events) == 3
        for i, (x, y, t, p) in enumerate(input_events):
            assert events[i].x == x
            assert events[i].y == y
            assert events[i].t == t
            assert events[i].p == p

    def test_roundtrip_polarity(self, tmp_path: Path) -> None:
        """ON and OFF events are correctly distinguished."""
        path = tmp_path / "polarity.bin"
        _write_nmnist_bin(path, [(5, 5, 1000, 0), (5, 5, 2000, 1)])
        events = _read_nmnist_bin(path)
        assert events[0].p == 0
        assert events[1].p == 1


class TestEventsToFrames:
    def test_empty_events(self) -> None:
        frames = events_to_frames([], n_frames=5)
        assert len(frames) == 5
        assert all(f.shape == (34, 34) for f in frames)
        assert all(f.sum() == 0 for f in frames)

    def test_frame_count(self) -> None:
        events = [Event(x=10, y=10, t=i * 100, p=1) for i in range(100)]
        frames = events_to_frames(events, n_frames=10)
        assert len(frames) == 10

    def test_events_binned_correctly(self) -> None:
        # All events in first half of time → first 5 frames active
        events = [Event(x=10, y=10, t=i, p=1) for i in range(50)]
        events += [Event(x=20, y=20, t=100 + i, p=1) for i in range(50)]
        frames = events_to_frames(events, n_frames=10)
        # First frames should have activity at (10,10), last at (20,20)
        assert frames[0][10, 10] > 0
        assert frames[-1][20, 20] > 0

    def test_normalized_to_unit_range(self) -> None:
        events = [Event(x=5, y=5, t=0, p=1)] * 100
        frames = events_to_frames(events, n_frames=1)
        assert frames[0].max() <= 1.0
        assert frames[0].min() >= 0.0


# ---------------------------------------------------------------------------
# N-MNIST adapter tests
# ---------------------------------------------------------------------------

class TestNMNISTAdapter:
    def test_loads_train_split(self, nmnist_dir: Path) -> None:
        adapter = NMNISTAdapter(nmnist_dir, split="train")
        assert len(adapter) == 20  # 10 digits × 2 samples

    def test_loads_test_split(self, nmnist_dir: Path) -> None:
        adapter = NMNISTAdapter(nmnist_dir, split="test")
        assert len(adapter) == 20

    def test_labels_correct(self, nmnist_dir: Path) -> None:
        adapter = NMNISTAdapter(nmnist_dir, split="train")
        labels = [adapter.label(i) for i in range(len(adapter))]
        assert set(labels) == set(range(10))

    def test_frames_shape(self, nmnist_dir: Path) -> None:
        adapter = NMNISTAdapter(nmnist_dir, split="train", n_frames=8)
        frames = adapter.frames(0)
        assert len(frames) == 8
        assert all(f.shape == (34, 34) for f in frames)

    def test_samples_for_digit(self, nmnist_dir: Path) -> None:
        adapter = NMNISTAdapter(nmnist_dir, split="train")
        for d in range(10):
            indices = adapter.samples_for_digit(d)
            assert len(indices) == 2
            assert all(adapter.label(i) == d for i in indices)

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="N-MNIST"):
            NMNISTAdapter(tmp_path / "nonexistent")

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        (tmp_path / "Train").mkdir()
        with pytest.raises(FileNotFoundError, match="No .bin files"):
            NMNISTAdapter(tmp_path)


# ---------------------------------------------------------------------------
# WAV reading and resampling tests
# ---------------------------------------------------------------------------

class TestWAVReading:
    def test_read_16bit_wav(self, tmp_path: Path) -> None:
        sr = 8000
        t = np.linspace(0, 0.1, 800, dtype=np.float32)
        samples = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        path = tmp_path / "test.wav"
        _write_wav(path, samples, sr)
        wav, read_sr = _read_wav(path)
        assert read_sr == sr
        assert wav.shape[0] == 800
        assert wav.dtype == torch.float32
        assert wav.abs().max() <= 1.0

    def test_resample_identity(self) -> None:
        wav = torch.randn(1000)
        result = _resample_linear(wav, 8000, 8000)
        assert torch.allclose(wav, result)

    def test_resample_upsample(self) -> None:
        wav = torch.randn(1000)
        result = _resample_linear(wav, 8000, 16000)
        assert result.shape[0] == 2000

    def test_resample_downsample(self) -> None:
        wav = torch.randn(1000)
        result = _resample_linear(wav, 16000, 8000)
        assert result.shape[0] == 500


# ---------------------------------------------------------------------------
# FSDD adapter tests
# ---------------------------------------------------------------------------

class TestFSDDAdapter:
    def test_loads_recordings(self, fsdd_dir: Path) -> None:
        adapter = FSDDAdapter(fsdd_dir)
        assert len(adapter) == 20  # 10 digits × 2 speakers

    def test_labels_correct(self, fsdd_dir: Path) -> None:
        adapter = FSDDAdapter(fsdd_dir)
        labels = [adapter.label(i) for i in range(len(adapter))]
        assert set(labels) == set(range(10))

    def test_waveform_shape(self, fsdd_dir: Path) -> None:
        adapter = FSDDAdapter(fsdd_dir, target_sr=16000)
        wav = adapter.waveform(0)
        assert wav.dim() == 1
        assert wav.shape[0] > 0

    def test_resampled_to_target(self, fsdd_dir: Path) -> None:
        adapter_8k = FSDDAdapter(fsdd_dir, target_sr=8000)
        adapter_16k = FSDDAdapter(fsdd_dir, target_sr=16000)
        wav_8k = adapter_8k.waveform(0)
        wav_16k = adapter_16k.waveform(0)
        # 16kHz should have ~2× the samples
        assert abs(wav_16k.shape[0] / wav_8k.shape[0] - 2.0) < 0.1

    def test_chunks_count(self, fsdd_dir: Path) -> None:
        adapter = FSDDAdapter(fsdd_dir)
        chunks = adapter.chunks(0, n_chunks=10, chunk_size=512)
        assert len(chunks) == 10
        assert all(c.shape == (512,) for c in chunks)

    def test_samples_for_digit(self, fsdd_dir: Path) -> None:
        adapter = FSDDAdapter(fsdd_dir)
        for d in range(10):
            indices = adapter.samples_for_digit(d)
            assert len(indices) == 2
            assert all(adapter.label(i) == d for i in indices)

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="FSDD"):
            FSDDAdapter(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# Paired digit dataset tests
# ---------------------------------------------------------------------------

class TestPairedDigitDataset:
    def test_pairs_all_digits(self, nmnist_dir: Path, fsdd_dir: Path) -> None:
        vis = NMNISTAdapter(nmnist_dir, split="train")
        aud = FSDDAdapter(fsdd_dir)
        dataset = PairedDigitDataset(vis, aud, n_steps=8)
        assert len(dataset) == 20  # 2 vis × 2 aud = max(2,2) = 2 per digit × 10 digits

    def test_episode_structure(self, nmnist_dir: Path, fsdd_dir: Path) -> None:
        vis = NMNISTAdapter(nmnist_dir, split="train", n_frames=8)
        aud = FSDDAdapter(fsdd_dir)
        dataset = PairedDigitDataset(vis, aud, n_steps=8, audio_chunk_size=512)
        episode = dataset[0]
        assert isinstance(episode, DigitEpisode)
        assert 0 <= episode.digit <= 9
        assert episode.text == DIGIT_NAMES[episode.digit]
        assert len(episode.visual_frames) == 8
        assert len(episode.audio_chunks) == 8
        assert all(f.shape == (34, 34) for f in episode.visual_frames)
        assert all(c.shape == (512,) for c in episode.audio_chunks)

    def test_labels_match_across_modalities(
        self, nmnist_dir: Path, fsdd_dir: Path
    ) -> None:
        vis = NMNISTAdapter(nmnist_dir, split="train")
        aud = FSDDAdapter(fsdd_dir)
        dataset = PairedDigitDataset(vis, aud, n_steps=5)
        for i in range(len(dataset)):
            ep = dataset[i]
            assert ep.text == DIGIT_NAMES[ep.digit]

    def test_deterministic_pairing(self, nmnist_dir: Path, fsdd_dir: Path) -> None:
        vis = NMNISTAdapter(nmnist_dir, split="train")
        aud = FSDDAdapter(fsdd_dir)
        d1 = PairedDigitDataset(vis, aud, seed=42)
        d2 = PairedDigitDataset(vis, aud, seed=42)
        for i in range(len(d1)):
            assert d1[i].digit == d2[i].digit

    def test_iter_episodes(self, nmnist_dir: Path, fsdd_dir: Path) -> None:
        vis = NMNISTAdapter(nmnist_dir, split="train")
        aud = FSDDAdapter(fsdd_dir)
        dataset = PairedDigitDataset(vis, aud, n_steps=5)
        episodes = list(dataset.iter_episodes())
        assert len(episodes) == len(dataset)


# ---------------------------------------------------------------------------
# Step-wise stream with encoder integration
# ---------------------------------------------------------------------------

class TestEpisodeSteps:
    def test_yields_correct_step_count(
        self, nmnist_dir: Path, fsdd_dir: Path
    ) -> None:
        vis = NMNISTAdapter(nmnist_dir, split="train", n_frames=5)
        aud = FSDDAdapter(fsdd_dir)
        dataset = PairedDigitDataset(vis, aud, n_steps=5)
        # Take just 2 episodes
        episodes = [dataset[0], dataset[1]]
        steps = list(iter_episode_steps(iter(episodes)))
        assert len(steps) == 10  # 2 episodes × 5 steps

    def test_with_encoders(self, nmnist_dir: Path, fsdd_dir: Path) -> None:
        vis_enc = EventCameraEncoder(height=34, width=34, pool=2)
        aud_enc = CochleagramEncoder(n_bands=64, n_fft=512)
        vis = NMNISTAdapter(nmnist_dir, split="train", n_frames=5)
        aud = FSDDAdapter(fsdd_dir)
        dataset = PairedDigitDataset(vis, aud, n_steps=5, audio_chunk_size=512)
        episode = dataset[0]
        steps = list(iter_episode_steps(iter([episode]), vis_enc, aud_enc))
        assert len(steps) == 5
        for step in steps:
            assert isinstance(step, MultimodalStep)
            assert isinstance(step.text, str)
            if step.visual_spikes is not None:
                assert step.visual_spikes.shape == (vis_enc.output_dim,)
            if step.audio_spikes is not None:
                assert step.audio_spikes.shape == (aud_enc.output_dim,)

    def test_encoder_reset_between_episodes(
        self, nmnist_dir: Path, fsdd_dir: Path
    ) -> None:
        """Verify encoders are reset at episode boundaries."""
        vis_enc = EventCameraEncoder(height=34, width=34, pool=2)
        aud_enc = CochleagramEncoder(n_bands=64, n_fft=512)
        vis = NMNISTAdapter(nmnist_dir, split="train", n_frames=3)
        aud = FSDDAdapter(fsdd_dir)
        dataset = PairedDigitDataset(vis, aud, n_steps=3, audio_chunk_size=512)

        ep1, ep2 = dataset[0], dataset[1]
        steps = list(iter_episode_steps(iter([ep1, ep2]), vis_enc, aud_enc))
        assert len(steps) == 6
        # First step of each episode: visual encoder has no reference frame
        # so it returns zeros (the reset behavior)
        assert steps[0].visual_spikes is not None
        assert steps[0].visual_spikes.sum() == 0  # first frame after reset
        assert steps[3].visual_spikes is not None
        assert steps[3].visual_spikes.sum() == 0  # first frame after reset


# ---------------------------------------------------------------------------
# Dimension validation tests
# ---------------------------------------------------------------------------

class TestDimensionValidation:
    def test_matching_dims_passes(self) -> None:
        vis_enc = EventCameraEncoder(height=64, width=64, pool=4)  # 256
        aud_enc = CochleagramEncoder(n_bands=64)
        validate_encoder_dims(vis_enc, aud_enc, 256, 64)

    def test_visual_mismatch_raises(self) -> None:
        vis_enc = EventCameraEncoder(height=34, width=34, pool=2)  # 17*17=289
        with pytest.raises(ValueError, match="Visual encoder"):
            validate_encoder_dims(vis_enc, None, 256, 64)

    def test_audio_mismatch_raises(self) -> None:
        aud_enc = CochleagramEncoder(n_bands=32)
        with pytest.raises(ValueError, match="Audio encoder"):
            validate_encoder_dims(None, aud_enc, 256, 64)

    def test_none_encoders_pass(self) -> None:
        validate_encoder_dims(None, None, 256, 64)


# ---------------------------------------------------------------------------
# DIGIT_NAMES constant test
# ---------------------------------------------------------------------------

class TestDigitNames:
    def test_ten_entries(self) -> None:
        assert len(DIGIT_NAMES) == 10

    def test_correct_names(self) -> None:
        assert DIGIT_NAMES[0] == "zero"
        assert DIGIT_NAMES[7] == "seven"
        assert DIGIT_NAMES[9] == "nine"
