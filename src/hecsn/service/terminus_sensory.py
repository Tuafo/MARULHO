"""Real Hugging Face multimodal streams for the live Terminus runtime.

These streams provide actual visual and audio grounding episodes so Terminus is
not limited to curriculum-derived synthetic sensory hints.

Current built-in adapters:
- `s1_mmalign`: scientific figure images + recaptions
- `audiocaps`: environmental / everyday audio + captions
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import wave
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from hecsn.data.cochleagram_encoder import CochleagramEncoder
from hecsn.data.corpus_loader import huggingface_token_from_env
from hecsn.data.event_camera_encoder import EventCameraEncoder


@dataclass
class SensoryEpisode:
    text: str
    visual_spikes: torch.Tensor | None
    audio_spikes: torch.Tensor | None
    metadata: dict[str, Any]
    visual_preview: dict[str, Any] | None = None
    audio_preview: dict[str, Any] | None = None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _load_hf_stream(
    source: str,
    *,
    hf_config: str | None = None,
    split: str = "train",
):
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "HuggingFace multimodal streaming requires the 'datasets' package. "
            "Install it with: pip install datasets"
        ) from exc

    token = huggingface_token_from_env()
    load_kwargs: dict[str, Any] = {"split": str(split or "train"), "streaming": True}
    if token:
        load_kwargs["token"] = token
    try:
        ds = load_dataset(source, hf_config, **load_kwargs) if hf_config else load_dataset(source, **load_kwargs)
    except TypeError:
        legacy_kwargs = dict(load_kwargs)
        if token:
            legacy_kwargs.pop("token", None)
            legacy_kwargs["use_auth_token"] = token
        ds = load_dataset(source, hf_config, **legacy_kwargs) if hf_config else load_dataset(source, **legacy_kwargs)
    decode = getattr(ds, "decode", None)
    if callable(decode):
        try:
            ds = decode(False)
        except Exception:
            pass
    return ds


def _reshape_spike_vector(spikes: torch.Tensor, target_dim: int) -> torch.Tensor:
    target = max(1, int(target_dim))
    vec = spikes.detach().float().reshape(-1)
    if vec.numel() == target:
        return vec
    resized = F.interpolate(
        vec.unsqueeze(0).unsqueeze(0),
        size=target,
        mode="linear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    return (resized > 0.0).float()


def _read_wave_bytes(raw_bytes: bytes) -> tuple[torch.Tensor, int]:
    with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        payload = wf.readframes(n_frames)

    if sampwidth == 1:
        arr = np.frombuffer(payload, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    elif sampwidth == 2:
        arr = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        arr = np.frombuffer(payload, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported audio sample width: {sampwidth}")

    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)
    return torch.from_numpy(arr), int(sample_rate)


def _resample_linear(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform.float()
    ratio = float(target_sr) / float(max(1, orig_sr))
    new_len = max(1, int(round(waveform.shape[0] * ratio)))
    return F.interpolate(
        waveform.float().unsqueeze(0).unsqueeze(0),
        size=new_len,
        mode="linear",
        align_corners=False,
    ).squeeze(0).squeeze(0)


def _image_bytes_to_frame(raw_bytes: bytes) -> torch.Tensor:
    with Image.open(io.BytesIO(raw_bytes)) as image:
        grey = image.convert("L")
        frame = torch.from_numpy(np.array(grey)).float() / 255.0
    return frame


def _image_preview_payload(raw_bytes: bytes, *, max_size: int = 256) -> dict[str, Any]:
    with Image.open(io.BytesIO(raw_bytes)) as image:
        preview = image.convert("RGB")
        preview.thumbnail((max_size, max_size))
        width, height = preview.size
        buffer = io.BytesIO()
        preview.save(buffer, format="PNG")
    return {
        "mime_type": "image/png",
        "bytes": buffer.getvalue(),
        "width": int(width),
        "height": int(height),
    }


def _waveform_to_wav_bytes(waveform: torch.Tensor, *, sample_rate: int) -> bytes:
    clipped = waveform.detach().float().cpu().clamp(-1.0, 1.0).numpy()
    int_samples = (clipped * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(int_samples.tobytes())
    return buffer.getvalue()


def _waveform_preview_bins(waveform: torch.Tensor, *, bins: int = 64) -> list[float]:
    wav = waveform.detach().float().cpu().abs()
    if wav.numel() <= 0:
        return [0.0] * bins
    chunk = max(1, int(np.ceil(wav.numel() / float(bins))))
    values: list[float] = []
    for start in range(0, wav.numel(), chunk):
        segment = wav[start : start + chunk]
        values.append(float(segment.mean().item()) if segment.numel() > 0 else 0.0)
        if len(values) >= bins:
            break
    if len(values) < bins:
        values.extend([0.0] * (bins - len(values)))
    return values[:bins]


def _best_waveform_segment(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    max_seconds: float = 4.0,
    candidates: int = 8,
) -> torch.Tensor:
    wav = waveform.float()
    segment_len = max(1, int(sample_rate * max_seconds))
    if wav.numel() <= segment_len:
        return wav
    step = max(1, (wav.numel() - segment_len) // max(1, candidates))
    best = wav[:segment_len]
    best_energy = float(best.pow(2).mean().item())
    for start in range(0, max(1, wav.numel() - segment_len + 1), step):
        segment = wav[start : start + segment_len]
        if segment.numel() < segment_len:
            segment = F.pad(segment, (0, segment_len - segment.numel()))
        energy = float(segment.pow(2).mean().item())
        if energy > best_energy:
            best = segment
            best_energy = energy
        if start + segment_len >= wav.numel():
            break
    return best


def _audio_preview_payload(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    max_seconds: float = 4.0,
) -> dict[str, Any]:
    segment = _best_waveform_segment(waveform, sample_rate=sample_rate, max_seconds=max_seconds)
    return {
        "mime_type": "audio/wav",
        "bytes": _waveform_to_wav_bytes(segment, sample_rate=sample_rate),
        "sample_rate": int(sample_rate),
        "duration_s": float(segment.numel() / float(max(1, sample_rate))),
        "waveform": _waveform_preview_bins(segment),
    }


def _make_visual_encoder(target_dim: int, *, device: torch.device) -> EventCameraEncoder:
    pool = 4
    side = max(8, int(round(max(1, target_dim) ** 0.5)))
    height = width = side * pool
    return EventCameraEncoder(height=height, width=width, pool=pool, device=device)


def _load_s1_recaption_index(source: str, year_prefix: str) -> dict[str, dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "ScienceOne-AI/S1-MMAlign support requires 'huggingface_hub'. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    filename = f"arxiv/jsonl/{year_prefix}_recaption.jsonl"
    token = huggingface_token_from_env()
    download_kwargs: dict[str, Any] = {"repo_id": source, "filename": filename, "repo_type": "dataset"}
    if token:
        download_kwargs["token"] = token
    try:
        path = hf_hub_download(**download_kwargs)
    except TypeError:
        if token:
            download_kwargs.pop("token", None)
            download_kwargs["use_auth_token"] = token
        path = hf_hub_download(**download_kwargs)

    index: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            image_path = _normalize_text(payload.get("image_path"))
            if image_path:
                index[image_path] = payload
    return index


class S1MMAlignSensoryStream:
    def __init__(
        self,
        *,
        source: str = "ScienceOne-AI/S1-MMAlign",
        split: str = "train",
        year_prefixes: Sequence[str] = ("07", "08", "09"),
        target_dim: int = 64,
        device: torch.device | None = None,
        max_text_chars: int = 480,
    ) -> None:
        self.source = source
        self.split = str(split or "train")
        self.year_prefixes = tuple(str(item).zfill(2)[:2] for item in year_prefixes if str(item).strip()) or ("07",)
        self.target_dim = max(1, int(target_dim))
        self.device = device or torch.device("cpu")
        self.max_text_chars = max(64, int(max_text_chars))
        self._visual_encoder = _make_visual_encoder(self.target_dim, device=self.device)
        self._stream = iter(_load_hf_stream(self.source, split=self.split))
        self._recaption_indices: dict[str, dict[str, dict[str, Any]]] = {}

    def _index_for_year(self, year_prefix: str) -> dict[str, dict[str, Any]]:
        cached = self._recaption_indices.get(year_prefix)
        if cached is None:
            cached = _load_s1_recaption_index(self.source, year_prefix)
            self._recaption_indices[year_prefix] = cached
        return cached

    def _episode_from_row(self, row: Mapping[str, Any]) -> SensoryEpisode | None:
        png = row.get("png")
        if not isinstance(png, Mapping):
            return None
        image_path = _normalize_text(png.get("path"))
        raw_bytes = png.get("bytes")
        if not image_path or not isinstance(raw_bytes, (bytes, bytearray)):
            return None
        year_prefix = image_path[:2]
        if year_prefix not in self.year_prefixes:
            return None
        entry = self._index_for_year(year_prefix).get(f"images/{image_path}")
        if not entry:
            return None

        title = _normalize_text(entry.get("title"))
        recaption = _normalize_text(entry.get("recaption"))
        categories = _normalize_text(entry.get("categories"))
        text = ". ".join(part for part in (title, recaption, f"Categories: {categories}" if categories else "") if part)
        text = text[: self.max_text_chars]
        if not text:
            return None

        frame = _image_bytes_to_frame(bytes(raw_bytes)).to(self.device)
        self._visual_encoder.reset()
        zero_frame = torch.zeros_like(frame)
        _ = self._visual_encoder.encode(zero_frame)
        visual_spikes = _reshape_spike_vector(self._visual_encoder.encode(frame), self.target_dim)
        if float(visual_spikes.sum().item()) <= 0.0:
            return None
        return SensoryEpisode(
            text=text,
            visual_spikes=visual_spikes,
            audio_spikes=None,
            metadata={
                "adapter": "s1_mmalign",
                "source": self.source,
                "image_path": image_path,
                "title": title,
                "categories": categories,
            },
            visual_preview=_image_preview_payload(bytes(raw_bytes)),
        )

    def __iter__(self) -> Iterator[SensoryEpisode]:
        while True:
            row = next(self._stream)
            episode = self._episode_from_row(row)
            if episode is not None:
                yield episode


class AudioCapsSensoryStream:
    def __init__(
        self,
        *,
        source: str = "OpenSound/AudioCaps",
        split: str = "train",
        target_dim: int = 64,
        sample_rate: int = 16000,
        n_fft: int = 512,
        device: torch.device | None = None,
        max_text_chars: int = 240,
        audio_candidates_per_item: int = 6,
    ) -> None:
        self.source = source
        self.split = str(split or "train")
        self.target_dim = max(1, int(target_dim))
        self.sample_rate = max(1000, int(sample_rate))
        self.n_fft = max(64, int(n_fft))
        self.device = device or torch.device("cpu")
        self.max_text_chars = max(32, int(max_text_chars))
        self.audio_candidates_per_item = max(1, int(audio_candidates_per_item))
        self._audio_encoder = CochleagramEncoder(
            n_bands=self.target_dim,
            n_fft=self.n_fft,
            sample_rate=self.sample_rate,
            device=self.device,
        )
        self._stream = iter(_load_hf_stream(self.source, split=self.split))

    def _select_waveform_chunk(self, waveform: torch.Tensor) -> torch.Tensor:
        wav = waveform.float().to(self.device)
        if wav.numel() <= self.n_fft:
            return F.pad(wav, (0, max(0, self.n_fft - wav.numel())))[: self.n_fft]
        step = max(1, (wav.numel() - self.n_fft) // self.audio_candidates_per_item)
        best_chunk = wav[: self.n_fft]
        best_energy = float(best_chunk.pow(2).mean().item())
        for start in range(0, max(1, wav.numel() - self.n_fft + 1), step):
            chunk = wav[start : start + self.n_fft]
            if chunk.numel() < self.n_fft:
                chunk = F.pad(chunk, (0, self.n_fft - chunk.numel()))
            energy = float(chunk.pow(2).mean().item())
            if energy > best_energy:
                best_energy = energy
                best_chunk = chunk
            if start + self.n_fft >= wav.numel():
                break
        return best_chunk

    def _episode_from_row(self, row: Mapping[str, Any]) -> SensoryEpisode | None:
        caption = _normalize_text(row.get("caption"))[: self.max_text_chars]
        audio_payload = row.get("audio")
        if not caption or not isinstance(audio_payload, Mapping):
            return None
        raw_bytes = audio_payload.get("bytes")
        if not isinstance(raw_bytes, (bytes, bytearray)):
            return None
        try:
            waveform, sample_rate = _read_wave_bytes(bytes(raw_bytes))
        except Exception:
            return None
        waveform = _resample_linear(waveform, sample_rate, self.sample_rate)
        chunk = self._select_waveform_chunk(waveform)
        self._audio_encoder.reset()
        audio_spikes = self._audio_encoder.encode(chunk)
        if float(audio_spikes.sum().item()) <= 0.0:
            return None
        return SensoryEpisode(
            text=caption,
            visual_spikes=None,
            audio_spikes=audio_spikes,
            metadata={
                "adapter": "audiocaps",
                "source": self.source,
                "youtube_id": _normalize_text(row.get("youtube_id")),
                "audiocap_id": int(row.get("audiocap_id", 0) or 0),
                "start_time": int(row.get("start_time", 0) or 0),
            },
            audio_preview=_audio_preview_payload(waveform, sample_rate=self.sample_rate),
        )

    def __iter__(self) -> Iterator[SensoryEpisode]:
        while True:
            row = next(self._stream)
            episode = self._episode_from_row(row)
            if episode is not None:
                yield episode


def build_sensory_stream(
    spec: Mapping[str, Any],
    *,
    visual_dim: int,
    audio_dim: int,
    device: torch.device | None = None,
) -> Iterator[SensoryEpisode]:
    adapter = _normalize_text(spec.get("adapter")).lower()

    def _generator() -> Iterator[SensoryEpisode]:
        if adapter == "s1_mmalign":
            yield from S1MMAlignSensoryStream(
                source=_normalize_text(spec.get("source")) or "ScienceOne-AI/S1-MMAlign",
                split=_normalize_text(spec.get("split")) or "train",
                year_prefixes=list(spec.get("year_prefixes") or ("07", "08", "09")),
                target_dim=visual_dim,
                device=device,
                max_text_chars=int(spec.get("max_text_chars", 480)),
            )
            return
        if adapter == "audiocaps":
            yield from AudioCapsSensoryStream(
                source=_normalize_text(spec.get("source")) or "OpenSound/AudioCaps",
                split=_normalize_text(spec.get("split")) or "train",
                target_dim=audio_dim,
                sample_rate=int(spec.get("sample_rate", 16000)),
                n_fft=int(spec.get("n_fft", 512)),
                device=device,
                max_text_chars=int(spec.get("max_text_chars", 240)),
                audio_candidates_per_item=int(spec.get("audio_candidates_per_item", 6)),
            )
            return
        raise ValueError(f"Unsupported sensory adapter: {adapter}")

    return _generator()
