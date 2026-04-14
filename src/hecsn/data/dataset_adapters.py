"""Multimodal dataset adapters for HECSN (§5.4).

Provides loaders for real neuromorphic and speech datasets to replace
synthetic concept-conditioned episodes with genuine multimodal data.

Supported datasets:
  - **N-MNIST**: Neuromorphic MNIST (34×34 DVS events) — visual modality
  - **FSDD**: Free Spoken Digit Dataset (8 kHz WAV) — audio modality
  - **TI-46**: Texas Instruments 46-word corpus (SPHERE, 12.5 kHz) — gated
    behind optional ``soundfile`` dependency
  - **PairedDigitDataset**: Combines visual + audio by digit class for
    Stage 1 training where alignment is guaranteed by construction

Episode contract:
  Each sample is an *episode* of ``n_steps`` synchronized time steps.
  Both modalities are normalized to the same step count via temporal
  resampling (repeat/subsample for visual, chunking for audio).
  Encoder reset happens at episode boundaries.
"""

from __future__ import annotations

import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


DIGIT_NAMES = [
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
]


# ---------------------------------------------------------------------------
# N-MNIST binary event reader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """Single DVS event: pixel (x, y), timestamp (µs), polarity."""
    x: int
    y: int
    t: int
    p: int


def _read_nmnist_bin(path: Path) -> list[Event]:
    """Read N-MNIST binary format (5 bytes per event).

    Format per event:
      byte 0: x address (0–33)
      byte 1: y address (0–33)
      bytes 2-4: 23-bit timestamp (µs) + 1-bit polarity (LSB of byte 4)
    """
    data = path.read_bytes()
    events: list[Event] = []
    for i in range(0, len(data) - 4, 5):
        x = data[i]
        y = data[i + 1]
        # timestamp is big-endian 23 bits, polarity is LSB of last byte
        ts_raw = (data[i + 2] << 16) | (data[i + 3] << 8) | data[i + 4]
        p = ts_raw & 1
        t = ts_raw >> 1
        events.append(Event(x=x, y=y, t=t, p=p))
    return events


def events_to_frames(
    events: list[Event],
    height: int = 34,
    width: int = 34,
    n_frames: int = 10,
) -> list[torch.Tensor]:
    """Bin DVS events into fixed-count frames.

    Splits the event stream by timestamp into ``n_frames`` equal time bins.
    Each frame is a float tensor of shape (H, W) with accumulated event
    counts normalized to [0, 1].

    For HECSN integration, these frames can be passed directly to
    ``EventCameraEncoder.encode()`` which handles resize internally.
    """
    if not events:
        return [torch.zeros(height, width) for _ in range(n_frames)]

    t_min = min(ev.t for ev in events)
    t_max = max(ev.t for ev in events)
    t_range = max(t_max - t_min, 1)

    frames = [torch.zeros(height, width) for _ in range(n_frames)]
    for ev in events:
        bin_idx = min(int((ev.t - t_min) * n_frames / t_range), n_frames - 1)
        if 0 <= ev.y < height and 0 <= ev.x < width:
            frames[bin_idx][ev.y, ev.x] += 1.0

    # Normalize each frame to [0, 1]
    for i in range(n_frames):
        mx = frames[i].max()
        if mx > 0:
            frames[i] = frames[i] / mx
    return frames


class NMNISTAdapter:
    """Loads N-MNIST dataset from disk.

    Expected directory layout::

        root/
          Train/ (or Test/)
            0/
              00001.bin
              00002.bin
              ...
            1/
            ...
            9/

    Each ``.bin`` file contains DVS events for one digit sample.

    Args:
        root: Path to dataset root (containing Train/ and/or Test/).
        split: ``"train"`` or ``"test"``.
        n_frames: Number of time-binned frames per sample.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        n_frames: int = 10,
    ) -> None:
        self.root = Path(root)
        self.n_frames = int(n_frames)

        split_dir = self.root / ("Train" if split == "train" else "Test")
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"N-MNIST {split} directory not found at {split_dir}. "
                f"Download from https://www.garrickorchard.com/datasets/n-mnist "
                f"and extract to {self.root}"
            )

        self._samples: list[tuple[Path, int]] = []
        for digit in range(10):
            digit_dir = split_dir / str(digit)
            if digit_dir.is_dir():
                for f in sorted(digit_dir.glob("*.bin")):
                    self._samples.append((f, digit))

        if not self._samples:
            raise FileNotFoundError(
                f"No .bin files found under {split_dir}. "
                f"Expected subdirectories 0/ through 9/ with .bin event files."
            )

    def __len__(self) -> int:
        return len(self._samples)

    def label(self, idx: int) -> int:
        return self._samples[idx][1]

    def frames(self, idx: int) -> list[torch.Tensor]:
        """Load sample and return time-binned frames (H=34, W=34)."""
        path, _ = self._samples[idx]
        events = _read_nmnist_bin(path)
        return events_to_frames(events, height=34, width=34, n_frames=self.n_frames)

    def samples_for_digit(self, digit: int) -> list[int]:
        """Return indices of all samples with the given digit label."""
        return [i for i, (_, d) in enumerate(self._samples) if d == digit]


# ---------------------------------------------------------------------------
# Audio digit adapters (FSDD + TI-46)
# ---------------------------------------------------------------------------

def _read_wav(path: Path) -> tuple[torch.Tensor, int]:
    """Read a WAV file using Python's built-in wave module.

    Returns (waveform_tensor, sample_rate). Waveform is float in [-1, 1].
    """
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        # 8-bit unsigned
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    elif sampwidth == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        # 24-bit — read as bytes, convert to int32
        n_samples = len(raw) // 3
        arr = np.zeros(n_samples, dtype=np.float32)
        for i in range(n_samples):
            b = raw[3 * i : 3 * i + 3]
            val = int.from_bytes(b, byteorder="little", signed=True)
            arr[i] = val / 8388608.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)

    return torch.from_numpy(arr), framerate


def _resample_linear(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Simple linear interpolation resampling (no dependency required)."""
    if orig_sr == target_sr:
        return waveform
    ratio = target_sr / orig_sr
    new_len = max(1, int(waveform.shape[0] * ratio))
    return torch.nn.functional.interpolate(
        waveform.unsqueeze(0).unsqueeze(0),
        size=new_len,
        mode="linear",
        align_corners=False,
    ).squeeze(0).squeeze(0)


class FSDDAdapter:
    """Loads Free Spoken Digit Dataset (FSDD) from disk.

    Expected directory layout::

        root/
          recordings/
            0_jackson_0.wav
            0_jackson_1.wav
            ...
            9_yweweler_49.wav

    Filename format: ``{digit}_{speaker}_{index}.wav``

    FSDD is freely available at https://github.com/Jakobovski/free-spoken-digit-dataset

    Args:
        root: Path to FSDD root (containing ``recordings/`` directory).
        target_sr: Resample all audio to this sample rate (default 16000).
    """

    def __init__(
        self,
        root: str | Path,
        target_sr: int = 16000,
    ) -> None:
        self.root = Path(root)
        self.target_sr = int(target_sr)

        rec_dir = self.root / "recordings"
        if not rec_dir.is_dir():
            raise FileNotFoundError(
                f"FSDD recordings directory not found at {rec_dir}. "
                f"Clone from https://github.com/Jakobovski/free-spoken-digit-dataset "
                f"and point root to the repository root."
            )

        self._samples: list[tuple[Path, int]] = []
        for f in sorted(rec_dir.glob("*.wav")):
            parts = f.stem.split("_")
            if parts and parts[0].isdigit():
                self._samples.append((f, int(parts[0])))

        if not self._samples:
            raise FileNotFoundError(f"No WAV files found in {rec_dir}.")

    def __len__(self) -> int:
        return len(self._samples)

    def label(self, idx: int) -> int:
        return self._samples[idx][1]

    def waveform(self, idx: int) -> torch.Tensor:
        """Load and resample audio to target sample rate."""
        path, _ = self._samples[idx]
        wav, sr = _read_wav(path)
        return _resample_linear(wav, sr, self.target_sr)

    def chunks(self, idx: int, n_chunks: int, chunk_size: int = 512) -> list[torch.Tensor]:
        """Load audio and split into fixed-count chunks for temporal alignment.

        Divides the waveform into ``n_chunks`` equal segments, each of size
        ``chunk_size`` (padded/truncated as needed). This enables temporal
        alignment with visual frames.
        """
        wav = self.waveform(idx)
        total = wav.shape[0]
        segments: list[torch.Tensor] = []
        for i in range(n_chunks):
            start = int(i * total / n_chunks)
            end = int((i + 1) * total / n_chunks)
            seg = wav[start:end]
            if seg.shape[0] < chunk_size:
                seg = torch.nn.functional.pad(seg, (0, chunk_size - seg.shape[0]))
            else:
                seg = seg[:chunk_size]
            segments.append(seg)
        return segments

    def samples_for_digit(self, digit: int) -> list[int]:
        return [i for i, (_, d) in enumerate(self._samples) if d == digit]


# ---------------------------------------------------------------------------
# Paired digit dataset (visual + audio, Stage 1)
# ---------------------------------------------------------------------------

@dataclass
class DigitEpisode:
    """One synchronized multimodal episode for a digit.

    Attributes:
        digit: Integer digit class (0-9).
        text: Digit name (e.g. "seven").
        visual_frames: List of n_steps frames, each shape (H, W).
        audio_chunks: List of n_steps waveform chunks, each shape (chunk_size,).
    """
    digit: int
    text: str
    visual_frames: list[torch.Tensor]
    audio_chunks: list[torch.Tensor]


class PairedDigitDataset:
    """Combines N-MNIST visual events with FSDD spoken audio by digit class.

    Each episode pairs one visual sample with one audio sample of the same
    digit. Both modalities are normalized to ``n_steps`` time steps.

    This implements the §5.4 "MNIST-DVS + TI-46 speech" data source where
    "digit displayed = digit spoken simultaneously" guarantees alignment.

    Args:
        visual_adapter: NMNISTAdapter instance.
        audio_adapter: FSDDAdapter instance.
        n_steps: Number of synchronized time steps per episode.
        audio_chunk_size: Samples per audio chunk (should match encoder n_fft).
        seed: Random seed for deterministic pairing.
    """

    def __init__(
        self,
        visual_adapter: NMNISTAdapter,
        audio_adapter: FSDDAdapter,
        n_steps: int = 10,
        audio_chunk_size: int = 512,
        seed: int = 42,
    ) -> None:
        self.visual = visual_adapter
        self.audio = audio_adapter
        self.n_steps = int(n_steps)
        self.audio_chunk_size = int(audio_chunk_size)

        rng = torch.Generator().manual_seed(seed)
        self._pairs: list[tuple[int, int, int]] = []  # (digit, vis_idx, aud_idx)

        for digit in range(10):
            vis_indices = self.visual.samples_for_digit(digit)
            aud_indices = self.audio.samples_for_digit(digit)
            if not vis_indices or not aud_indices:
                continue

            # Deterministic 1:1 pairing with cycling of the shorter list
            n_pairs = max(len(vis_indices), len(aud_indices))
            for i in range(n_pairs):
                vi = vis_indices[i % len(vis_indices)]
                ai = aud_indices[i % len(aud_indices)]
                self._pairs.append((digit, vi, ai))

        # Shuffle deterministically
        perm = torch.randperm(len(self._pairs), generator=rng).tolist()
        self._pairs = [self._pairs[i] for i in perm]

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> DigitEpisode:
        digit, vis_idx, aud_idx = self._pairs[idx]
        frames = self.visual.frames(vis_idx)
        chunks = self.audio.chunks(aud_idx, self.n_steps, self.audio_chunk_size)
        return DigitEpisode(
            digit=digit,
            text=DIGIT_NAMES[digit],
            visual_frames=frames,
            audio_chunks=chunks,
        )

    def iter_episodes(self) -> Iterator[DigitEpisode]:
        """Iterate over all episodes in shuffled order."""
        for i in range(len(self)):
            yield self[i]


# ---------------------------------------------------------------------------
# Step-wise multimodal stream from episodes
# ---------------------------------------------------------------------------

@dataclass
class MultimodalStep:
    """Single time step with text + encoded visual spikes + encoded audio spikes."""
    text: str
    visual_spikes: torch.Tensor | None  # shape: (visual_output_dim,)
    audio_spikes: torch.Tensor | None   # shape: (audio_output_dim,)


def iter_episode_steps(
    episodes: Iterator[DigitEpisode],
    visual_encoder: "EventCameraEncoder | None" = None,
    audio_encoder: "CochleagramEncoder | None" = None,
) -> Iterator[MultimodalStep]:
    """Flatten episodes into a step-wise stream with encoder resets at boundaries.

    This is the primary integration point with the HECSN trainer. Each step
    yields encoded spike vectors ready for ``trainer.train_step()``.

    Encoders are reset at episode boundaries to prevent cross-sample
    contamination of stateful reference frames / baselines.

    Args:
        episodes: Iterator of DigitEpisode objects.
        visual_encoder: EventCameraEncoder (reset between episodes).
        audio_encoder: CochleagramEncoder (reset between episodes).

    Yields:
        MultimodalStep with text label and encoded spikes.
    """
    for episode in episodes:
        # Reset stateful encoders at episode boundary
        if visual_encoder is not None:
            visual_encoder.reset()
        if audio_encoder is not None:
            audio_encoder.reset()

        n_steps = len(episode.visual_frames)
        for step_i in range(n_steps):
            vs = None
            aus = None

            if visual_encoder is not None and step_i < len(episode.visual_frames):
                vs = visual_encoder.encode(episode.visual_frames[step_i])

            if audio_encoder is not None and step_i < len(episode.audio_chunks):
                aus = audio_encoder.encode(episode.audio_chunks[step_i])

            yield MultimodalStep(
                text=episode.text,
                visual_spikes=vs,
                audio_spikes=aus,
            )


# ---------------------------------------------------------------------------
# Dimension validation helper
# ---------------------------------------------------------------------------

def validate_encoder_dims(
    visual_encoder: "EventCameraEncoder | None",
    audio_encoder: "CochleagramEncoder | None",
    cross_modal_dim_visual: int,
    cross_modal_dim_audio: int,
) -> None:
    """Verify encoder output dims match the cross-modal config.

    Raises ValueError with actionable guidance if dimensions mismatch.
    """
    if visual_encoder is not None:
        vd = visual_encoder.output_dim
        if vd != cross_modal_dim_visual:
            raise ValueError(
                f"Visual encoder output_dim={vd} but config "
                f"cross_modal_dim_visual={cross_modal_dim_visual}. "
                f"Adjust encoder pool/height/width or config to match. "
                f"For 34×34 N-MNIST with pool=2: output_dim={17*17}=289. "
                f"For pool=4: output_dim={(34//4)*(34//4)}."
            )
    if audio_encoder is not None:
        ad = audio_encoder.output_dim
        if ad != cross_modal_dim_audio:
            raise ValueError(
                f"Audio encoder output_dim={ad} but config "
                f"cross_modal_dim_audio={cross_modal_dim_audio}. "
                f"Adjust encoder n_bands or config to match."
            )
