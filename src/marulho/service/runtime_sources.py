"""Runtime source stream and cache helpers for Terminus.

This module owns source stream construction, live-remote wrapping, runtime cache
read/write helpers, and stream shutdown. It does not decide replay policy,
training policy, or memory promotion.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any, Callable, Iterator, Mapping, Sequence, cast

import torch

from marulho.data.corpus_loader import BackgroundPrefetchIterator, SourceType, StreamingCorpusLoader
from marulho.data.pattern_loader import labeled_pattern_stream
from marulho.service.terminus_sensory import SensoryEpisode, build_sensory_stream

DEFAULT_BRAIN_TICK_TOKENS = 512
DEFAULT_REMOTE_STREAM_PREFETCH_ITEMS = 4


@dataclass
class _BrainSourceRuntime:
    spec: dict[str, Any]
    stream: Iterator[tuple[str, torch.Tensor]]
    tokens_processed: int = 0
    cycles_completed: int = 0
    exhausted: bool = False
    tick_visits: int = 0
    last_tokens_trained: int = 0
    last_activity_at: str | None = None
    prefetched_tokens: int = 0
    prefetch_events: int = 0
    last_prefetch_token_count: int = 0
    last_prefetch_at: str | None = None
    last_prefetch_duration_ms: float | None = None
    last_prefetch_error: str | None = None
    queue_hits: int = 0
    last_buffer_tokens_served: int = 0
    last_semantic_match: float = 0.0
    last_selection_score: float = 0.0
    last_fairness_score: float = 0.0
    last_buffer_readiness: float = 0.0
    last_utility_score: float = 0.0
    buffered_patterns: deque[tuple[str, torch.Tensor]] = field(default_factory=deque)
    bootstrap_attempted: bool = False
    cache_material_hash: str | None = None
    cache_write_count: int = 0
    cache_schedule_count: int = 0
    cache_skip_count: int = 0
    cache_failure_count: int = 0
    cache_pending: bool = False
    last_cache_update_mode: str = "not_run"

    @property
    def name(self) -> str:
        return str(self.spec.get("name", "source"))

    @property
    def source_type(self) -> str:
        return str(self.spec.get("source_type", "auto"))


@dataclass
class _SensorySourceRuntime:
    spec: dict[str, Any]
    stream: Iterator[SensoryEpisode]
    episodes_processed: int = 0
    cycles_completed: int = 0
    exhausted: bool = False
    last_activity_at: str | None = None
    last_text: str | None = None
    last_semantic_match: float = 0.0
    last_modality_need: float = 0.0
    last_selection_score: float = 0.0
    last_window_budget: int = 0
    buffered_episodes: list[SensoryEpisode] = field(default_factory=list)
    prefetched_episodes: int = 0
    prefetch_events: int = 0
    last_prefetch_episode_count: int = 0
    last_prefetch_at: str | None = None
    last_prefetch_duration_ms: float | None = None
    last_prefetch_error: str | None = None
    queue_hits: int = 0
    last_buffer_episodes_served: int = 0
    last_item_semantic_match: float = 0.0
    last_item_candidates_considered: int = 0
    last_item_retrieval_lookahead: int = 0
    bootstrap_attempted: bool = False

    @property
    def name(self) -> str:
        return str(self.spec.get("name", "sensory_source"))

    @property
    def adapter(self) -> str:
        return str(self.spec.get("adapter", "unknown"))



@dataclass(frozen=True)
class RuntimeSourcesDependencies:
    brain_config: Callable[[], Mapping[str, Any]]
    brain_source_runtimes: Callable[[], Sequence[_BrainSourceRuntime]]
    set_brain_source_runtimes: Callable[[Sequence[_BrainSourceRuntime]], None]
    checkpoint_dir: Callable[[], Path]
    checkpoint_path: Callable[[], Path]
    encoder: Callable[[], Any]
    sensory_queue_target_items: Callable[[], int]
    sensory_source_runtimes: Callable[[], Sequence[_SensorySourceRuntime]]
    set_sensory_source_runtimes: Callable[[Sequence[_SensorySourceRuntime]], None]
    trainer: Callable[[], Any]


class RuntimeSources:
    def __init__(self, dependencies: RuntimeSourcesDependencies) -> None:
        self._dependencies = dependencies
        self._brain_cache_write_queue: Queue[
            tuple[_BrainSourceRuntime, Path, dict[str, Any], str] | None
        ] = Queue()
        self._brain_cache_worker_lock = Lock()
        self._brain_cache_worker: Thread | None = None

    @property
    def _brain_config(self) -> Mapping[str, Any]:
        return self._dependencies.brain_config()

    @property
    def _brain_source_runtimes(self) -> Sequence[_BrainSourceRuntime]:
        return self._dependencies.brain_source_runtimes()

    @_brain_source_runtimes.setter
    def _brain_source_runtimes(self, value: Sequence[_BrainSourceRuntime]) -> None:
        self._dependencies.set_brain_source_runtimes(value)

    @property
    def _checkpoint_dir(self) -> Path:
        return self._dependencies.checkpoint_dir()

    @property
    def _checkpoint_path(self) -> Path:
        return self._dependencies.checkpoint_path()

    @property
    def _encoder(self) -> Any:
        return self._dependencies.encoder()

    @property
    def _sensory_source_runtimes(self) -> Sequence[_SensorySourceRuntime]:
        return self._dependencies.sensory_source_runtimes()

    @_sensory_source_runtimes.setter
    def _sensory_source_runtimes(self, value: Sequence[_SensorySourceRuntime]) -> None:
        self._dependencies.set_sensory_source_runtimes(value)

    @property
    def _trainer(self) -> Any:
        return self._dependencies.trainer()

    def _sensory_queue_target_items_locked(self) -> int:
        return int(self._dependencies.sensory_queue_target_items())

    @staticmethod
    def _source_spec_uses_live_remote(spec: Mapping[str, Any]) -> bool:
        source_type = str(spec.get("source_type", "auto") or "auto").strip().lower()
        if source_type in {"hf", "web"}:
            return True
        if source_type == "file":
            return False
        source = str(spec.get("source", "") or "").strip()
        if source.startswith(("http://", "https://")):
            return True
        return not Path(source).exists()

    @staticmethod
    def _sensory_spec_uses_live_remote(spec: Mapping[str, Any]) -> bool:
        adapter = str(spec.get("adapter", "") or "").strip().lower()
        if adapter in {"s1_mmalign", "audiocaps"}:
            return True
        source = str(spec.get("source", "") or "").strip()
        return bool(source) and not Path(source).exists()

    @staticmethod
    def _wrap_remote_stream(spec: Mapping[str, Any], stream: Iterator[Any], *, is_sensory: bool) -> Iterator[Any]:
        uses_remote = (
            RuntimeSources._sensory_spec_uses_live_remote(spec)
            if is_sensory
            else RuntimeSources._source_spec_uses_live_remote(spec)
        )
        if not uses_remote:
            return stream
        name = str(spec.get("name", "sensory" if is_sensory else "source"))
        return BackgroundPrefetchIterator(
            stream,
            max_buffer=DEFAULT_REMOTE_STREAM_PREFETCH_ITEMS,
            name=name,
        )

    @staticmethod
    def _stream_supports_ready_reads(stream: Iterator[Any]) -> bool:
        return callable(getattr(stream, "next_ready", None))

    @staticmethod
    def _next_stream_item(stream: Iterator[Any], *, timeout: float | None = None) -> Any:
        next_ready = getattr(stream, "next_ready", None)
        if callable(next_ready):
            return next_ready(timeout=timeout)
        return next(stream)

    def _build_brain_source_stream_locked(self, spec: dict[str, Any]) -> Iterator[tuple[str, torch.Tensor]]:
        return self._build_source_stream_from_spec(
            spec,
            self._encoder,
            self._trainer.config.window_size,
        )

    @staticmethod
    def _build_source_stream_from_spec(
        spec: dict[str, Any],
        encoder: Any,
        window_size: int,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        source_type_raw = str(spec.get("source_type", "auto")).strip().lower() or "auto"
        if source_type_raw == "file":
            source_type: SourceType = "file"
        elif source_type_raw == "hf":
            source_type = "hf"
        elif source_type_raw == "web":
            source_type = "web"
        else:
            source_type = "auto"
        loader = StreamingCorpusLoader(
            source=str(spec.get("source", "")),
            source_type=source_type,
            text_field=str(spec.get("text_field", "text")),
            hf_config=spec.get("hf_config"),
        )
        stream = labeled_pattern_stream(
            loader.char_stream(),
            encoder,
            window_size,
            learn_chunking=False,
        )
        return cast(Iterator[tuple[str, torch.Tensor]], RuntimeSources._wrap_remote_stream(spec, stream, is_sensory=False))

    def _build_sensory_stream_locked(self, spec: dict[str, Any]) -> Iterator[SensoryEpisode]:
        return self._build_sensory_stream_from_spec(
            spec,
            visual_dim=int(getattr(self._trainer.config, "cross_modal_dim_visual", 64)),
            audio_dim=int(getattr(self._trainer.config, "cross_modal_dim_audio", 64)),
            device=self._trainer.model.device,
        )

    @staticmethod
    def _build_sensory_stream_from_spec(
        spec: dict[str, Any],
        *,
        visual_dim: int,
        audio_dim: int,
        device: Any,
    ) -> Iterator[SensoryEpisode]:
        stream = build_sensory_stream(
            spec,
            visual_dim=int(visual_dim),
            audio_dim=int(audio_dim),
            device=device,
        )
        return cast(Iterator[SensoryEpisode], RuntimeSources._wrap_remote_stream(spec, stream, is_sensory=True))

    def _runtime_cache_root(self) -> Path:
        root = self._checkpoint_dir / "runtime_cache"
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _source_file_fingerprint(spec: Mapping[str, Any]) -> dict[str, Any] | None:
        source = str(spec.get("source", "") or "").strip()
        if not source or source.startswith(("http://", "https://")):
            return None
        source_type = str(spec.get("source_type", "auto") or "auto").strip().lower()
        path = Path(source)
        if source_type not in {"file", "auto"} and not path.exists():
            return None
        if not path.is_file():
            return None
        try:
            stat = path.stat()
            resolved = str(path.resolve())
        except OSError:
            return None
        return {
            "resolved_path": resolved,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    @classmethod
    def _brain_runtime_cache_enabled(cls, spec: Mapping[str, Any]) -> bool:
        return cls._source_spec_uses_live_remote(spec) or cls._source_file_fingerprint(spec) is not None

    def _runtime_cache_key(self, *, kind: str, spec: Mapping[str, Any]) -> str:
        payload = {
            "kind": str(kind),
            "checkpoint": str(self._checkpoint_path.resolve()),
            "window_size": int(self._trainer.config.window_size),
            "visual_dim": int(getattr(self._trainer.config, "cross_modal_dim_visual", 64)),
            "audio_dim": int(getattr(self._trainer.config, "cross_modal_dim_audio", 64)),
            "spec": dict(spec),
            "source_file_fingerprint": self._source_file_fingerprint(spec),
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _brain_runtime_cache_path(self, spec: Mapping[str, Any]) -> Path:
        return self._runtime_cache_root() / f"brain_{self._runtime_cache_key(kind='brain', spec=spec)}.pt"

    def _sensory_runtime_cache_path(self, spec: Mapping[str, Any]) -> Path:
        return self._runtime_cache_root() / f"sensory_{self._runtime_cache_key(kind='sensory', spec=spec)}.pt"

    @staticmethod
    def _brain_runtime_cache_material_hash(raw_windows: Sequence[str]) -> str:
        payload = {
            "raw_windows": [str(item) for item in raw_windows],
            "token_count": int(len(raw_windows)),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _reconstruct_text_from_windows(raw_windows: Sequence[str]) -> str:
        windows = [str(item) for item in raw_windows if str(item)]
        if not windows:
            return ""
        reconstructed = windows[0]
        for window in windows[1:]:
            max_overlap = min(len(reconstructed), len(window))
            overlap = 0
            for size in range(max_overlap, 0, -1):
                if reconstructed.endswith(window[:size]):
                    overlap = size
                    break
            reconstructed += window[overlap:]
        return reconstructed

    def _update_brain_runtime_cache_locked(
        self,
        runtime: _BrainSourceRuntime,
        *,
        served_examples: Sequence[tuple[str, torch.Tensor]] | None = None,
    ) -> None:
        if not self._brain_runtime_cache_enabled(runtime.spec):
            return
        ingestion = self._brain_config.get("ingestion") or {}
        tick_tokens = int(self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS))
        target_tokens = max(int(tick_tokens), int(ingestion.get("queue_target_tokens", tick_tokens)))
        raw_windows: list[str] = []
        if served_examples:
            raw_windows.extend(str(raw_window) for raw_window, _pattern in served_examples if str(raw_window))
        raw_windows.extend(str(raw_window) for raw_window, _pattern in list(runtime.buffered_patterns) if str(raw_window))
        raw_windows = raw_windows[: max(1, target_tokens)]
        if not raw_windows:
            return
        material_hash = self._brain_runtime_cache_material_hash(raw_windows)
        if runtime.cache_material_hash == material_hash:
            runtime.cache_skip_count += 1
            runtime.last_cache_update_mode = "skipped_unchanged_material"
            return
        payload = {
            "raw_windows": raw_windows,
            "token_count": int(len(raw_windows)),
            "material_hash": material_hash,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        runtime.cache_material_hash = material_hash
        runtime.cache_schedule_count += 1
        runtime.cache_pending = True
        runtime.last_cache_update_mode = "scheduled"
        self._ensure_brain_cache_worker()
        self._brain_cache_write_queue.put(
            (
                runtime,
                self._brain_runtime_cache_path(runtime.spec),
                payload,
                material_hash,
            )
        )

    def _ensure_brain_cache_worker(self) -> None:
        with self._brain_cache_worker_lock:
            if self._brain_cache_worker is not None and self._brain_cache_worker.is_alive():
                return
            self._brain_cache_worker = Thread(
                target=self._brain_cache_write_loop,
                name="marulho-source-cache-writer",
                daemon=True,
            )
            self._brain_cache_worker.start()

    def _brain_cache_write_loop(self) -> None:
        while True:
            task = self._brain_cache_write_queue.get()
            try:
                if task is None:
                    return
                runtime, path, payload, material_hash = task
                temporary_path = path.with_suffix(f"{path.suffix}.tmp")
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(payload, temporary_path)
                    temporary_path.replace(path)
                    runtime.cache_write_count += 1
                    runtime.last_cache_update_mode = "written"
                except Exception:
                    runtime.cache_failure_count += 1
                    runtime.last_cache_update_mode = "write_failed"
                    if runtime.cache_material_hash == material_hash:
                        runtime.cache_material_hash = None
                    try:
                        temporary_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                finally:
                    runtime.cache_pending = runtime.cache_material_hash not in (
                        None,
                        material_hash,
                    )
            finally:
                self._brain_cache_write_queue.task_done()

    def flush_brain_runtime_cache_writes(self) -> None:
        self._brain_cache_write_queue.join()

    def close(self) -> None:
        self.flush_brain_runtime_cache_writes()
        with self._brain_cache_worker_lock:
            worker = self._brain_cache_worker
            if worker is None:
                return
            self._brain_cache_write_queue.put(None)
        self._brain_cache_write_queue.join()
        worker.join(timeout=5.0)
        with self._brain_cache_worker_lock:
            self._brain_cache_worker = None

    def _restore_brain_runtime_cache_locked(self, runtime: _BrainSourceRuntime) -> int:
        if not self._brain_runtime_cache_enabled(runtime.spec):
            return 0
        path = self._brain_runtime_cache_path(runtime.spec)
        if not path.exists():
            return 0
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            return 0
        raw_windows = [str(item) for item in list((payload or {}).get("raw_windows") or []) if str(item)]
        token_count = max(0, int((payload or {}).get("token_count", len(raw_windows)) or 0))
        if not raw_windows or token_count <= 0:
            return 0
        material_hash = str((payload or {}).get("material_hash") or "").strip()
        if not material_hash:
            material_hash = self._brain_runtime_cache_material_hash(raw_windows[:token_count])
        text = self._reconstruct_text_from_windows(raw_windows)
        if not text:
            return 0
        examples: list[tuple[str, torch.Tensor]] = []
        for raw_window, pattern in labeled_pattern_stream(
            text,
            self._encoder,
            self._trainer.config.window_size,
            learn_chunking=False,
        ):
            examples.append((raw_window, pattern))
            if len(examples) >= token_count:
                break
        if not examples:
            return 0
        runtime.buffered_patterns = deque(examples)
        runtime.cache_material_hash = material_hash
        runtime.last_cache_update_mode = "restored"
        return int(len(examples))

    def _serialize_sensory_episode(self, episode: SensoryEpisode) -> dict[str, Any]:
        return {
            "text": str(episode.text),
            "visual_spikes": None if episode.visual_spikes is None else episode.visual_spikes.detach().cpu(),
            "audio_spikes": None if episode.audio_spikes is None else episode.audio_spikes.detach().cpu(),
            "metadata": deepcopy(episode.metadata),
            "visual_preview": deepcopy(episode.visual_preview),
            "audio_preview": deepcopy(episode.audio_preview),
        }

    def _deserialize_sensory_episode(self, payload: Mapping[str, Any]) -> SensoryEpisode:
        visual_spikes = payload.get("visual_spikes")
        audio_spikes = payload.get("audio_spikes")
        device = self._trainer.model.device
        if isinstance(visual_spikes, torch.Tensor):
            visual_spikes = visual_spikes.to(device)
        else:
            visual_spikes = None
        if isinstance(audio_spikes, torch.Tensor):
            audio_spikes = audio_spikes.to(device)
        else:
            audio_spikes = None
        return SensoryEpisode(
            text=str(payload.get("text", "")),
            visual_spikes=visual_spikes,
            audio_spikes=audio_spikes,
            metadata=deepcopy(dict(payload.get("metadata") or {})),
            visual_preview=deepcopy(payload.get("visual_preview")),
            audio_preview=deepcopy(payload.get("audio_preview")),
        )

    def _update_sensory_runtime_cache_locked(
        self,
        runtime: _SensorySourceRuntime,
        *,
        served_episodes: Sequence[SensoryEpisode] | None = None,
    ) -> None:
        if not self._sensory_spec_uses_live_remote(runtime.spec):
            return
        target_items = self._sensory_queue_target_items_locked()
        episodes: list[SensoryEpisode] = []
        if served_episodes:
            episodes.extend(served_episodes)
        episodes.extend(list(runtime.buffered_episodes))
        episodes = episodes[: max(1, target_items)]
        if not episodes:
            return
        payload = {
            "episodes": [self._serialize_sensory_episode(item) for item in episodes],
            "item_count": int(len(episodes)),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            torch.save(payload, self._sensory_runtime_cache_path(runtime.spec))
        except Exception:
            return

    def _restore_sensory_runtime_cache_locked(self, runtime: _SensorySourceRuntime) -> int:
        if not self._sensory_spec_uses_live_remote(runtime.spec):
            return 0
        path = self._sensory_runtime_cache_path(runtime.spec)
        if not path.exists():
            return 0
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            return 0
        raw_episodes = list((payload or {}).get("episodes") or [])
        if not raw_episodes:
            return 0
        restored: list[SensoryEpisode] = []
        for item in raw_episodes:
            if not isinstance(item, Mapping):
                continue
            restored.append(self._deserialize_sensory_episode(item))
        if not restored:
            return 0
        runtime.buffered_episodes = list(restored)
        return int(len(restored))

    @staticmethod
    def _close_runtime_streams(runtimes: Sequence[Any]) -> None:
        for runtime in runtimes:
            close = getattr(getattr(runtime, "stream", None), "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    continue

    def _interrupt_brain_sources_locked(self) -> None:
        self._close_runtime_streams(self._brain_source_runtimes)

    def _interrupt_sensory_sources_locked(self) -> None:
        self._close_runtime_streams(self._sensory_source_runtimes)

    def _close_brain_sources_locked(self) -> None:
        self._interrupt_brain_sources_locked()
        self._brain_source_runtimes = []

    def _close_sensory_sources_locked(self) -> None:
        self._interrupt_sensory_sources_locked()
        self._sensory_source_runtimes = []

