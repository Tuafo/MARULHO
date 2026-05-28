"""Multimodal stream loader for HECSN (§7 Developmental Protocol).

Yields synchronized (text, visual, audio) triples for multimodal training.
Supports two modes:

1. **Synthetic mode** — generates random visual frames and audio chunks
   alongside real text for integration testing without external datasets.

2. **Directory mode** — reads aligned files from a structured directory:
   ```
   dataset_root/
     text/     *.txt  (one file per episode)
     visual/   *.pt   (saved frame tensors, shape [N, H, W])
     audio/    *.pt   (saved waveform tensors, shape [N, samples])
   ```
   Files are paired by sorted filename order within each modality.

The loader handles temporal alignment: each text window produces one
visual frame and one audio chunk.  When modality data runs out before
text, it cycles.
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Iterator, Optional

import torch

from hecsn.data.cochleagram_encoder import CochleagramEncoder
from hecsn.data.event_camera_encoder import EventCameraEncoder


class MultimodalSample:
    """A single synchronized multimodal observation."""

    __slots__ = ("text", "visual_frame", "audio_chunk")

    def __init__(
        self,
        text: str,
        visual_frame: Optional[torch.Tensor],
        audio_chunk: Optional[torch.Tensor],
    ) -> None:
        self.text = text
        self.visual_frame = visual_frame
        self.audio_chunk = audio_chunk


class MultimodalStreamLoader:
    """Yields synchronized multimodal triples for HECSN training.

    Args:
        text_source: Iterable of text strings (character stream, corpus, etc.)
        window_size: Number of characters per text window.
        visual_encoder: EventCameraEncoder instance (optional; None = text-only).
        audio_encoder: CochleagramEncoder instance (optional; None = text-only).
        visual_source: Iterable of visual frames, each shape (H, W) or (C, H, W).
        audio_source: Iterable of audio chunks, each shape (samples,).
        synthetic: If True, generate random visual/audio data when no source given.
        device: Torch device for generated data.
    """

    def __init__(
        self,
        text_source: Iterator[str],
        window_size: int = 5,
        visual_encoder: Optional[EventCameraEncoder] = None,
        audio_encoder: Optional[CochleagramEncoder] = None,
        visual_source: Optional[Iterator[torch.Tensor]] = None,
        audio_source: Optional[Iterator[torch.Tensor]] = None,
        synthetic: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        self._text_source = text_source
        self._window_size = int(window_size)
        self._visual_encoder = visual_encoder
        self._audio_encoder = audio_encoder
        self._visual_source = visual_source
        self._audio_source = audio_source
        self._synthetic = synthetic
        self._device = _resolve_multimodal_device(device)
        self._char_buffer: list[str] = []

    def __iter__(self) -> Iterator[MultimodalSample]:
        return self._generate()

    def _generate(self) -> Iterator[MultimodalSample]:
        for text_block in self._text_source:
            self._char_buffer.extend(text_block)
            while len(self._char_buffer) >= self._window_size:
                window = "".join(self._char_buffer[: self._window_size])
                self._char_buffer = self._char_buffer[self._window_size :]

                visual_frame = self._next_visual()
                audio_chunk = self._next_audio()
                yield MultimodalSample(
                    text=window,
                    visual_frame=visual_frame,
                    audio_chunk=audio_chunk,
                )

    def _next_visual(self) -> Optional[torch.Tensor]:
        if self._visual_source is not None:
            try:
                return next(self._visual_source).to(self._device)
            except StopIteration:
                self._visual_source = None
        if self._synthetic and self._visual_encoder is not None:
            h, w = self._visual_encoder.height, self._visual_encoder.width
            return torch.rand(h, w, device=self._device)
        return None

    def _next_audio(self) -> Optional[torch.Tensor]:
        if self._audio_source is not None:
            try:
                return next(self._audio_source).to(self._device)
            except StopIteration:
                self._audio_source = None
        if self._synthetic and self._audio_encoder is not None:
            n_fft = self._audio_encoder.n_fft
            return torch.randn(n_fft, device=self._device) * 0.1
        return None


def load_directory(
    root: str | Path,
    visual_encoder: Optional[EventCameraEncoder] = None,
    audio_encoder: Optional[CochleagramEncoder] = None,
    window_size: int = 5,
    device: Optional[torch.device] = None,
) -> MultimodalStreamLoader:
    """Create a MultimodalStreamLoader from a structured directory.

    Expected layout::

        root/
          text/     *.txt
          visual/   *.pt   (each: [N_frames, H, W] float tensors)
          audio/    *.pt   (each: [N_chunks, samples] float tensors)

    Files within each modality are sorted by name and concatenated.
    """
    root = Path(root)
    text_dir = root / "text"
    visual_dir = root / "visual"
    audio_dir = root / "audio"
    runtime_device = _resolve_multimodal_device(device)

    def _text_iter() -> Iterator[str]:
        if text_dir.is_dir():
            for f in sorted(text_dir.glob("*.txt")):
                yield f.read_text(encoding="utf-8")

    def _visual_iter() -> Iterator[torch.Tensor]:
        if visual_dir.is_dir():
            for f in sorted(visual_dir.glob("*.pt")):
                frames = torch.load(f, map_location=runtime_device, weights_only=True)
                for i in range(frames.shape[0]):
                    yield frames[i]

    def _audio_iter() -> Iterator[torch.Tensor]:
        if audio_dir.is_dir():
            for f in sorted(audio_dir.glob("*.pt")):
                chunks = torch.load(f, map_location=runtime_device, weights_only=True)
                for i in range(chunks.shape[0]):
                    yield chunks[i]

    has_visual = visual_dir.is_dir() and any(visual_dir.glob("*.pt"))
    has_audio = audio_dir.is_dir() and any(audio_dir.glob("*.pt"))

    return MultimodalStreamLoader(
        text_source=_text_iter(),
        window_size=window_size,
        visual_encoder=visual_encoder,
        audio_encoder=audio_encoder,
        visual_source=_visual_iter() if has_visual else None,
        audio_source=_audio_iter() if has_audio else None,
        device=runtime_device,
    )


def _resolve_multimodal_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return torch.device(device)
    env_device = os.environ.get("HECSN_DEVICE")
    if env_device:
        return torch.device(env_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
