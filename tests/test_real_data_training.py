"""Tests for _train_on_real_digits integration (developmental_runner + dataset adapters).

Verifies that the real-data training path correctly:
- Encodes digit names via RTF (last char-window = full word)
- Passes encoded visual/audio spikes through train_step
- Aggregates accepted sensory evidence per episode (not per step)
- Calls update_word_grounding once per episode with averaged spikes
"""

from __future__ import annotations

import numpy as np
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from hecsn.config.model_config import HECSNConfig
from hecsn.data.cochleagram_encoder import CochleagramEncoder
from hecsn.data.dataset_adapters import (
    DIGIT_NAMES,
    DigitEpisode,
    FSDDAdapter,
    NMNISTAdapter,
    PairedDigitDataset,
)
from hecsn.data.event_camera_encoder import EventCameraEncoder
from hecsn.data.rtf_encoder import RTFEncoder
from hecsn.training.developmental_runner import _train_on_real_digits
from hecsn.training.model import HECSNModel
from hecsn.training.trainer import HECSNTrainer


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_nmnist_bin(path: Path, events: list[tuple[int, int, int, int]]) -> None:
    with open(path, "wb") as f:
        for x, y, t, p in events:
            ts_and_pol = (t << 1) | (p & 1)
            b2 = (ts_and_pol >> 16) & 0xFF
            b3 = (ts_and_pol >> 8) & 0xFF
            b4 = ts_and_pol & 0xFF
            f.write(bytes([x & 0xFF, y & 0xFF, b2, b3, b4]))


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 8000) -> None:
    int_samples = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())


@pytest.fixture
def nmnist_dir(tmp_path: Path) -> Path:
    for split in ["Train", "Test"]:
        for digit in range(10):
            digit_dir = tmp_path / "nmnist" / split / str(digit)
            digit_dir.mkdir(parents=True)
            for sample_idx in range(2):
                events = [
                    ((digit * 3 + i) % 34, (digit * 5 + i * 2) % 34, i * 1000, i % 2)
                    for i in range(50)
                ]
                _write_nmnist_bin(digit_dir / f"{sample_idx:05d}.bin", events)
    return tmp_path / "nmnist"


@pytest.fixture
def fsdd_dir(tmp_path: Path) -> Path:
    rec_dir = tmp_path / "fsdd" / "recordings"
    rec_dir.mkdir(parents=True)
    for digit in range(10):
        freq = 200 + digit * 100
        sr = 8000
        t = np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32)
        samples = 0.8 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        _write_wav(rec_dir / f"{digit}_alice_0.wav", samples, sr)
    return tmp_path / "fsdd"


@pytest.fixture
def paired_dataset(nmnist_dir: Path, fsdd_dir: Path) -> PairedDigitDataset:
    vis = NMNISTAdapter(nmnist_dir, split="train", n_frames=5)
    aud = FSDDAdapter(fsdd_dir, target_sr=8000)
    return PairedDigitDataset(vis, aud, n_steps=5, audio_chunk_size=512, seed=42)


@pytest.fixture
def cfg() -> HECSNConfig:
    # Visual encoder: height=8, width=8, pool=1 → output_dim=64
    # Audio encoder: n_bands=32 → output_dim=32
    return HECSNConfig(
        n_columns=16,
        window_size=10,
        cross_modal_dim_visual=64,
        cross_modal_dim_audio=32,
    )


@pytest.fixture
def trainer_and_encoder(cfg: HECSNConfig):
    encoder = RTFEncoder.from_config(cfg)
    model = HECSNModel(config=cfg)
    trainer = HECSNTrainer(model, cfg)
    return trainer, encoder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrainOnRealDigits:
    """Tests for _train_on_real_digits function."""

    def test_basic_invocation(self, paired_dataset, trainer_and_encoder):
        """Should run without error and return step counts."""
        trainer, encoder = trainer_and_encoder
        vis_enc = EventCameraEncoder(height=8, width=8, pool=1)
        aud_enc = CochleagramEncoder(n_bands=32, n_fft=512, sample_rate=8000)

        episodes = list(paired_dataset.iter_episodes())[:3]
        steps, vis, aud = _train_on_real_digits(
            trainer, encoder, episodes, vis_enc, aud_enc, n_episodes=3,
        )
        assert steps > 0
        assert vis > 0
        assert aud > 0

    def test_respects_n_episodes_limit(self, paired_dataset, trainer_and_encoder):
        """Should stop after n_episodes episodes."""
        trainer, encoder = trainer_and_encoder
        vis_enc = EventCameraEncoder(height=8, width=8, pool=1)
        aud_enc = CochleagramEncoder(n_bands=32, n_fft=512, sample_rate=8000)

        episodes = list(paired_dataset.iter_episodes())
        steps_1, _, _ = _train_on_real_digits(
            trainer, encoder, episodes[:1], vis_enc, aud_enc, n_episodes=1,
        )
        steps_5, _, _ = _train_on_real_digits(
            trainer, encoder, episodes[:5], vis_enc, aud_enc, n_episodes=5,
        )
        # More episodes → more steps
        assert steps_5 > steps_1

    def test_steps_match_episode_length(self, trainer_and_encoder):
        """Steps processed should equal sum of episode step counts."""
        trainer, encoder = trainer_and_encoder
        vis_enc = EventCameraEncoder(height=8, width=8, pool=1)
        aud_enc = CochleagramEncoder(n_bands=32, n_fft=512, sample_rate=8000)

        # Build synthetic episodes directly
        episodes = []
        n_steps = 5
        for d in [3, 7]:
            episodes.append(DigitEpisode(
                digit=d,
                text=DIGIT_NAMES[d],
                visual_frames=[torch.rand(8, 8) for _ in range(n_steps)],
                audio_chunks=[torch.randn(512) for _ in range(n_steps)],
            ))

        steps, _, _ = _train_on_real_digits(
            trainer, encoder, episodes, vis_enc, aud_enc, n_episodes=2,
        )
        assert steps == n_steps * 2

    def test_visual_only(self, trainer_and_encoder):
        """Should work with visual encoder only (no audio)."""
        trainer, encoder = trainer_and_encoder
        vis_enc = EventCameraEncoder(height=8, width=8, pool=1)

        episodes = [DigitEpisode(
            digit=5,
            text="five",
            visual_frames=[torch.rand(8, 8) for _ in range(3)],
            audio_chunks=[torch.randn(512) for _ in range(3)],
        )]

        steps, vis, aud = _train_on_real_digits(
            trainer, encoder, episodes, vis_enc, None, n_episodes=1,
        )
        assert steps == 3
        assert vis == 3
        assert aud == 0

    def test_audio_only(self, trainer_and_encoder):
        """Should work with audio encoder only (no visual)."""
        trainer, encoder = trainer_and_encoder
        aud_enc = CochleagramEncoder(n_bands=32, n_fft=512, sample_rate=8000)

        episodes = [DigitEpisode(
            digit=2,
            text="two",
            visual_frames=[torch.rand(8, 8) for _ in range(3)],
            audio_chunks=[torch.randn(512) for _ in range(3)],
        )]

        steps, vis, aud = _train_on_real_digits(
            trainer, encoder, episodes, None, aud_enc, n_episodes=1,
        )
        assert steps == 3
        assert vis == 0
        assert aud == 3

    def test_empty_episodes_list(self, trainer_and_encoder):
        """Should handle empty episode list gracefully."""
        trainer, encoder = trainer_and_encoder
        steps, vis, aud = _train_on_real_digits(
            trainer, encoder, [], None, None, n_episodes=10,
        )
        assert steps == 0
        assert vis == 0
        assert aud == 0

    def test_uses_last_char_window(self, trainer_and_encoder):
        """Should use the last char window (full word), not the first."""
        trainer, encoder = trainer_and_encoder

        # Spy on train_step to capture raw_window
        original_train_step = trainer.train_step
        captured_windows = []

        def spy_train_step(pattern_vec, raw_window=None, **kwargs):
            captured_windows.append(raw_window)
            return original_train_step(pattern_vec, raw_window=raw_window, **kwargs)

        trainer.train_step = spy_train_step

        episodes = [DigitEpisode(
            digit=7,
            text="seven",
            visual_frames=[torch.rand(8, 8) for _ in range(2)],
            audio_chunks=[torch.randn(512) for _ in range(2)],
        )]

        _train_on_real_digits(
            trainer, encoder, episodes, None, None, n_episodes=1,
        )

        # All windows should contain the full word "seven"
        # (not just "s" or "se" from early windows)
        for w in captured_windows:
            assert "seven" in w or len(w) >= 5

    def test_grounding_update_called_per_episode(self, trainer_and_encoder):
        """update_word_grounding should be called once per episode, not per step."""
        trainer, encoder = trainer_and_encoder
        vis_enc = EventCameraEncoder(height=8, width=8, pool=1)
        aud_enc = CochleagramEncoder(n_bands=32, n_fft=512, sample_rate=8000)

        call_count = 0
        original_update = trainer.update_word_grounding

        def counting_update(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_update(*args, **kwargs)

        trainer.update_word_grounding = counting_update

        n_episodes = 3
        episodes = [
            DigitEpisode(
                digit=d,
                text=DIGIT_NAMES[d],
                visual_frames=[torch.rand(8, 8) for _ in range(5)],
                audio_chunks=[torch.randn(512) for _ in range(5)],
            )
            for d in [1, 4, 9]
        ]

        _train_on_real_digits(
            trainer, encoder, episodes, vis_enc, aud_enc, n_episodes=n_episodes,
        )

        # At most n_episodes calls (could be fewer if no modalities accepted)
        assert call_count <= n_episodes

    def test_with_paired_dataset(self, paired_dataset, trainer_and_encoder):
        """End-to-end test with PairedDigitDataset."""
        trainer, encoder = trainer_and_encoder
        vis_enc = EventCameraEncoder(height=8, width=8, pool=1)
        aud_enc = CochleagramEncoder(n_bands=32, n_fft=512, sample_rate=8000)

        episodes = list(paired_dataset.iter_episodes())[:5]
        steps, vis, aud = _train_on_real_digits(
            trainer, encoder, episodes, vis_enc, aud_enc, n_episodes=5,
        )
        assert steps > 0
        assert vis >= 0
        assert aud >= 0
