"""Real Hugging Face multimodal streams for the live Terminus runtime.

These streams provide the maintained visual and audio grounding path for
Terminus, replacing synthetic curriculum-hint channels as the runtime’s
multimodal substrate.

Current built-in adapters:
- `s1_mmalign`: scientific figure images + recaptions
- `audiocaps`: environmental / everyday audio + captions
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import os
from threading import Event, Lock, Thread
import time
import wave
from typing import Any, Iterator, Mapping, Sequence
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from hecsn.data.cochleagram_encoder import CochleagramEncoder
from hecsn.data.corpus_loader import huggingface_token_from_env, project_dataset_columns
from hecsn.data.event_camera_encoder import EventCameraEncoder


@dataclass
class SensoryEpisode:
    text: str
    visual_spikes: torch.Tensor | None
    audio_spikes: torch.Tensor | None
    metadata: dict[str, Any]
    visual_preview: dict[str, Any] | None = None
    audio_preview: dict[str, Any] | None = None


_S1_RECAPTION_SOFT_WAIT_SECONDS = 0.1
_S1_RECAPTION_RETRY_COOLDOWN_SECONDS = 30.0
_S1_RECAPTION_INDEX_CACHE: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
_S1_RECAPTION_INDEX_READY: dict[tuple[str, str], Event] = {}
_S1_RECAPTION_INDEX_LOADING: set[tuple[str, str]] = set()
_S1_RECAPTION_LAST_ATTEMPT: dict[tuple[str, str], float] = {}
_S1_RECAPTION_CACHE_LOCK = Lock()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def sensory_bootstrap_columns(spec: Mapping[str, Any]) -> list[str]:
    adapter = _normalize_text(spec.get("adapter")).lower()
    if adapter == "s1_mmalign":
        return ["png", "__key__"]
    if adapter == "audiocaps":
        return ["caption", "audio", "youtube_id", "audiocap_id", "start_time"]
    return []


def _download_binary_asset(url: str, *, timeout_seconds: float = 20.0) -> bytes:
    request = Request(
        str(url),
        headers={
            "User-Agent": "HECSN/1.0 (+https://github.com/) hf-asset-loader",
            "Accept": "*/*",
        },
    )
    with urlopen(request, timeout=float(timeout_seconds)) as response:
        return response.read()


def _s1_image_path_from_row(row: Mapping[str, Any], png_payload: Mapping[str, Any]) -> str:
    image_path = _normalize_text(png_payload.get("path"))
    if image_path:
        return image_path
    key = _normalize_text(row.get("__key__"))
    if not key:
        return ""
    lowered = key.lower()
    if lowered.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
        return key
    return f"{key}.png"


def _s1_image_bytes_from_payload(
    row: Mapping[str, Any],
    png_payload: Mapping[str, Any],
    *,
    timeout_seconds: float | None = None,
) -> bytes | None:
    raw_bytes = png_payload.get("bytes")
    if isinstance(raw_bytes, (bytes, bytearray)):
        return bytes(raw_bytes)
    src = _normalize_text(png_payload.get("src"))
    if not src:
        return None
    try:
        return _download_binary_asset(src, timeout_seconds=20.0 if timeout_seconds is None else float(max(0.05, timeout_seconds)))
    except Exception:
        return None


def _audiocaps_audio_bytes_from_payload(audio_payload: Any, *, timeout_seconds: float | None = None) -> bytes | None:
    if isinstance(audio_payload, Mapping):
        raw_bytes = audio_payload.get("bytes")
        if isinstance(raw_bytes, (bytes, bytearray)):
            return bytes(raw_bytes)
        src = _normalize_text(audio_payload.get("src"))
        if src:
            try:
                return _download_binary_asset(src, timeout_seconds=20.0 if timeout_seconds is None else float(max(0.05, timeout_seconds)))
            except Exception:
                return None
        return None
    if isinstance(audio_payload, Sequence) and not isinstance(audio_payload, (str, bytes, bytearray)):
        for item in audio_payload:
            if not isinstance(item, Mapping):
                continue
            src = _normalize_text(item.get("src"))
            if not src:
                continue
            try:
                return _download_binary_asset(src, timeout_seconds=20.0 if timeout_seconds is None else float(max(0.05, timeout_seconds)))
            except Exception:
                return None
    return None


def _load_hf_stream(
    source: str,
    *,
    hf_config: str | None = None,
    split: str = "train",
    columns: Sequence[str] | None = None,
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
    ds = project_dataset_columns(ds, columns)
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


def _s1_recaption_cache_key(source: str, year_prefix: str) -> tuple[str, str]:
    return (_normalize_text(source), str(year_prefix).zfill(2)[:2])


def _reset_s1_recaption_index_runtime() -> None:
    with _S1_RECAPTION_CACHE_LOCK:
        _S1_RECAPTION_INDEX_CACHE.clear()
        _S1_RECAPTION_INDEX_READY.clear()
        _S1_RECAPTION_INDEX_LOADING.clear()
        _S1_RECAPTION_LAST_ATTEMPT.clear()


def _ensure_s1_recaption_index_loading(source: str, year_prefix: str) -> Event | None:
    key = _s1_recaption_cache_key(source, year_prefix)
    with _S1_RECAPTION_CACHE_LOCK:
        cached = _S1_RECAPTION_INDEX_CACHE.get(key)
        if cached is not None:
            ready = _S1_RECAPTION_INDEX_READY.get(key)
            if ready is None:
                ready = Event()
                ready.set()
                _S1_RECAPTION_INDEX_READY[key] = ready
            return ready
        ready = _S1_RECAPTION_INDEX_READY.get(key)
        if ready is None:
            ready = Event()
            _S1_RECAPTION_INDEX_READY[key] = ready
        if key in _S1_RECAPTION_INDEX_LOADING:
            return ready
        last_attempt = _S1_RECAPTION_LAST_ATTEMPT.get(key)
        if last_attempt is not None and (time.monotonic() - last_attempt) < float(_S1_RECAPTION_RETRY_COOLDOWN_SECONDS):
            return ready
        _S1_RECAPTION_INDEX_LOADING.add(key)
        _S1_RECAPTION_LAST_ATTEMPT[key] = time.monotonic()
        ready.clear()

    def _runner() -> None:
        try:
            index = _load_s1_recaption_index(source, year_prefix)
        except Exception:
            with _S1_RECAPTION_CACHE_LOCK:
                _S1_RECAPTION_INDEX_LOADING.discard(key)
                signal = _S1_RECAPTION_INDEX_READY.get(key)
                if signal is not None:
                    signal.set()
            return
        with _S1_RECAPTION_CACHE_LOCK:
            _S1_RECAPTION_INDEX_CACHE[key] = index
            _S1_RECAPTION_INDEX_LOADING.discard(key)
            signal = _S1_RECAPTION_INDEX_READY.get(key)
            if signal is not None:
                signal.set()

    thread = Thread(target=_runner, name=f"hecsn-s1-recaption-{key[1]}", daemon=True)
    thread.start()
    return ready


def _cached_s1_recaption_index(
    source: str,
    year_prefix: str,
    *,
    wait_seconds: float = 0.0,
) -> dict[str, dict[str, Any]] | None:
    key = _s1_recaption_cache_key(source, year_prefix)
    with _S1_RECAPTION_CACHE_LOCK:
        cached = _S1_RECAPTION_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    ready = _ensure_s1_recaption_index_loading(source, year_prefix)
    if ready is not None and float(wait_seconds) > 0.0:
        ready.wait(timeout=max(0.0, float(wait_seconds)))
    with _S1_RECAPTION_CACHE_LOCK:
        return _S1_RECAPTION_INDEX_CACHE.get(key)


def _s1_image_identifiers(image_path: str) -> tuple[str, str, str]:
    normalized = _normalize_text(image_path)
    if not normalized:
        return "", "", ""
    parts = [part for part in normalized.split("/") if part]
    archive_bucket = parts[0] if parts else ""
    paper_id = ""
    figure_id = ""
    if len(parts) >= 2:
        paper_id = parts[1].removesuffix(".tar.gz")
    if len(parts) >= 3:
        figure_id = parts[2].split(".", 1)[0]
    return archive_bucket, paper_id, figure_id


def _s1_episode_text(
    image_path: str,
    entry: Mapping[str, Any] | None,
    *,
    max_text_chars: int,
) -> tuple[str, str, str, str, str, str, str]:
    title = _normalize_text(entry.get("title")) if isinstance(entry, Mapping) else ""
    recaption = _normalize_text(entry.get("recaption")) if isinstance(entry, Mapping) else ""
    categories = _normalize_text(entry.get("categories")) if isinstance(entry, Mapping) else ""
    text = ". ".join(part for part in (title, recaption, f"Categories: {categories}" if categories else "") if part).strip()
    if text:
        return text[: max(64, int(max_text_chars))], "recaption_index", title, categories, *_s1_image_identifiers(image_path)
    archive_bucket, paper_id, figure_id = _s1_image_identifiers(image_path)
    subject = "Scientific figure"
    if figure_id:
        subject += f" {figure_id}"
    context: list[str] = []
    if paper_id:
        context.append(f"from paper {paper_id}")
    if archive_bucket:
        context.append(f"in archive bucket {archive_bucket}")
    fallback = subject
    if context:
        fallback += " " + " ".join(context)
    fallback = (fallback.strip() + ".")[: max(64, int(max_text_chars))]
    return fallback, "image_path_fallback", "", "", archive_bucket, paper_id, figure_id


def _build_s1_episode_from_row(
    row: Mapping[str, Any],
    *,
    source: str,
    year_prefixes: Sequence[str],
    target_dim: int,
    device: torch.device,
    max_text_chars: int,
    visual_encoder: EventCameraEncoder,
    index_for_year: Any,
    asset_timeout_seconds: float | None = None,
) -> SensoryEpisode | None:
    png = row.get("png")
    if not isinstance(png, Mapping):
        return None
    image_path = _s1_image_path_from_row(row, png)
    raw_bytes = _s1_image_bytes_from_payload(row, png, timeout_seconds=asset_timeout_seconds)
    if not image_path or raw_bytes is None:
        return None
    year_prefix = image_path[:2]
    allowed_years = tuple(str(item).zfill(2)[:2] for item in year_prefixes if str(item).strip()) or ("07",)
    if year_prefix not in allowed_years:
        return None
    entry_index = index_for_year(year_prefix)
    entry = entry_index.get(f"images/{image_path}") if isinstance(entry_index, Mapping) else None
    text, text_source, title, categories, archive_bucket, paper_id, figure_id = _s1_episode_text(
        image_path,
        entry if isinstance(entry, Mapping) else None,
        max_text_chars=max_text_chars,
    )
    if not text:
        return None

    frame = _image_bytes_to_frame(bytes(raw_bytes)).to(device)
    visual_encoder.reset()
    zero_frame = torch.zeros_like(frame)
    _ = visual_encoder.encode(zero_frame)
    visual_spikes = _reshape_spike_vector(visual_encoder.encode(frame), target_dim)
    if float(visual_spikes.sum().item()) <= 0.0:
        return None
    return SensoryEpisode(
        text=text,
        visual_spikes=visual_spikes,
        audio_spikes=None,
        metadata={
            "adapter": "s1_mmalign",
            "source": source,
            "device": str(device),
            "encoder": visual_encoder.device_report(),
            "spike_device": str(visual_spikes.device),
            "spike_is_cuda": bool(visual_spikes.is_cuda),
            "image_path": image_path,
            "title": title,
            "categories": categories,
            "archive_bucket": archive_bucket,
            "paper_id": paper_id,
            "figure_id": figure_id,
            "text_source": text_source,
            "metadata_pending": text_source != "recaption_index",
        },
        visual_preview=_image_preview_payload(bytes(raw_bytes)),
    )


def _build_audiocaps_episode_from_row(
    row: Mapping[str, Any],
    *,
    source: str,
    target_dim: int,
    sample_rate: int,
    n_fft: int,
    device: torch.device,
    max_text_chars: int,
    audio_candidates_per_item: int,
    audio_encoder: CochleagramEncoder,
    asset_timeout_seconds: float | None = None,
) -> SensoryEpisode | None:
    caption = _normalize_text(row.get("caption"))[: max(32, int(max_text_chars))]
    audio_payload = row.get("audio")
    raw_bytes = _audiocaps_audio_bytes_from_payload(audio_payload, timeout_seconds=asset_timeout_seconds)
    if not caption or raw_bytes is None:
        return None
    try:
        waveform, orig_sample_rate = _read_wave_bytes(bytes(raw_bytes))
    except Exception:
        return None
    waveform = _resample_linear(waveform, orig_sample_rate, max(1000, int(sample_rate)))

    target_n_fft = max(64, int(n_fft))
    candidates = max(1, int(audio_candidates_per_item))
    wav = waveform.float().to(device)
    if wav.numel() <= target_n_fft:
        chunk = F.pad(wav, (0, max(0, target_n_fft - wav.numel())))[: target_n_fft]
    else:
        step = max(1, (wav.numel() - target_n_fft) // candidates)
        best_chunk = wav[: target_n_fft]
        best_energy = float(best_chunk.pow(2).mean().item())
        for start in range(0, max(1, wav.numel() - target_n_fft + 1), step):
            current = wav[start : start + target_n_fft]
            if current.numel() < target_n_fft:
                current = F.pad(current, (0, target_n_fft - current.numel()))
            energy = float(current.pow(2).mean().item())
            if energy > best_energy:
                best_energy = energy
                best_chunk = current
            if start + target_n_fft >= wav.numel():
                break
        chunk = best_chunk

    audio_encoder.reset()
    audio_spikes = audio_encoder.encode(chunk)
    if float(audio_spikes.sum().item()) <= 0.0:
        return None
    normalized_sample_rate = max(1000, int(sample_rate))
    return SensoryEpisode(
        text=caption,
        visual_spikes=None,
        audio_spikes=audio_spikes,
        metadata={
            "adapter": "audiocaps",
            "source": source,
            "device": str(device),
            "encoder": audio_encoder.device_report(),
            "spike_device": str(audio_spikes.device),
            "spike_is_cuda": bool(audio_spikes.is_cuda),
            "youtube_id": _normalize_text(row.get("youtube_id")),
            "audiocap_id": int(row.get("audiocap_id", 0) or 0),
            "start_time": int(row.get("start_time", 0) or 0),
        },
        audio_preview=_audio_preview_payload(waveform, sample_rate=normalized_sample_rate),
    )


def bootstrap_sensory_episode_from_row(
    spec: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    visual_dim: int,
    audio_dim: int,
    device: torch.device | None = None,
    timeout_seconds: float | None = None,
) -> SensoryEpisode | None:
    adapter = _normalize_text(spec.get("adapter")).lower()
    runtime_device = _resolve_sensory_device(device)
    if adapter == "s1_mmalign":
        source = _normalize_text(spec.get("source")) or "ScienceOne-AI/S1-MMAlign"
        year_prefixes = list(spec.get("year_prefixes") or ("07", "08", "09"))

        def index_for_year(year_prefix: str) -> dict[str, dict[str, Any]] | None:
            wait_seconds = float(_S1_RECAPTION_SOFT_WAIT_SECONDS)
            if timeout_seconds is not None:
                wait_seconds = min(wait_seconds, float(max(0.0, timeout_seconds)))
            return _cached_s1_recaption_index(
                source,
                year_prefix,
                wait_seconds=wait_seconds,
            )

        return _build_s1_episode_from_row(
            row,
            source=source,
            year_prefixes=year_prefixes,
            target_dim=max(1, int(visual_dim)),
            device=runtime_device,
            max_text_chars=int(spec.get("max_text_chars", 480)),
            visual_encoder=_make_visual_encoder(max(1, int(visual_dim)), device=runtime_device),
            index_for_year=index_for_year,
            asset_timeout_seconds=timeout_seconds,
        )
    if adapter == "audiocaps":
        sample_rate = max(1000, int(spec.get("sample_rate", 16000)))
        n_fft = max(64, int(spec.get("n_fft", 512)))
        return _build_audiocaps_episode_from_row(
            row,
            source=_normalize_text(spec.get("source")) or "OpenSound/AudioCaps",
            target_dim=max(1, int(audio_dim)),
            sample_rate=sample_rate,
            n_fft=n_fft,
            device=runtime_device,
            max_text_chars=int(spec.get("max_text_chars", 240)),
            audio_candidates_per_item=int(spec.get("audio_candidates_per_item", 6)),
            audio_encoder=CochleagramEncoder(
                n_bands=max(1, int(audio_dim)),
                n_fft=n_fft,
                sample_rate=sample_rate,
                device=runtime_device,
            ),
            asset_timeout_seconds=timeout_seconds,
        )
    return None


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
        self.device = _resolve_sensory_device(device)
        self.max_text_chars = max(64, int(max_text_chars))
        self._visual_encoder = _make_visual_encoder(self.target_dim, device=self.device)
        self._stream = iter(_load_hf_stream(self.source, split=self.split, columns=["png"]))

    def _index_for_year(self, year_prefix: str) -> dict[str, dict[str, Any]] | None:
        return _cached_s1_recaption_index(
            self.source,
            year_prefix,
            wait_seconds=float(_S1_RECAPTION_SOFT_WAIT_SECONDS),
        )

    def _episode_from_row(self, row: Mapping[str, Any]) -> SensoryEpisode | None:
        return _build_s1_episode_from_row(
            row,
            source=self.source,
            year_prefixes=self.year_prefixes,
            target_dim=self.target_dim,
            device=self.device,
            max_text_chars=self.max_text_chars,
            visual_encoder=self._visual_encoder,
            index_for_year=self._index_for_year,
            asset_timeout_seconds=None,
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
        self.device = _resolve_sensory_device(device)
        self.max_text_chars = max(32, int(max_text_chars))
        self.audio_candidates_per_item = max(1, int(audio_candidates_per_item))
        self._audio_encoder = CochleagramEncoder(
            n_bands=self.target_dim,
            n_fft=self.n_fft,
            sample_rate=self.sample_rate,
            device=self.device,
        )
        self._stream = iter(
            _load_hf_stream(
                self.source,
                split=self.split,
                columns=["caption", "audio", "youtube_id", "audiocap_id", "start_time"],
            )
        )

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
        return _build_audiocaps_episode_from_row(
            row,
            source=self.source,
            target_dim=self.target_dim,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            device=self.device,
            max_text_chars=self.max_text_chars,
            audio_candidates_per_item=self.audio_candidates_per_item,
            audio_encoder=self._audio_encoder,
            asset_timeout_seconds=None,
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
    runtime_device = _resolve_sensory_device(device)

    def _generator() -> Iterator[SensoryEpisode]:
        if adapter == "s1_mmalign":
            yield from S1MMAlignSensoryStream(
                source=_normalize_text(spec.get("source")) or "ScienceOne-AI/S1-MMAlign",
                split=_normalize_text(spec.get("split")) or "train",
                year_prefixes=list(spec.get("year_prefixes") or ("07", "08", "09")),
                target_dim=visual_dim,
                device=runtime_device,
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
                device=runtime_device,
                max_text_chars=int(spec.get("max_text_chars", 240)),
                audio_candidates_per_item=int(spec.get("audio_candidates_per_item", 6)),
            )
            return
        raise ValueError(f"Unsupported sensory adapter: {adapter}")

    return _generator()


def _resolve_sensory_device(device: torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    env_device = os.environ.get("HECSN_DEVICE")
    if env_device:
        return torch.device(env_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
