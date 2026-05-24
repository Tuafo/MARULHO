from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast
from uuid import uuid4

from hecsn.config.runtime_env import load_runtime_env
from hecsn.reporting.io import write_json_file
from hecsn.semantics import ConceptStore, GeometricCuriosityController
from hecsn.training.checkpointing import load_trainer_checkpoint, save_trainer_checkpoint

DEFAULT_REPLAY_SAMPLE_HISTORY = 256
DEFAULT_DELAYED_CONSEQUENCE_RECORDS = 24


_FORWARDED_STATE_NAMES = frozenset({
    "_action_history",
    "_action_root",
    "_brain_config",
    "_brain_last_acquisition_summary",
    "_brain_last_acquisition_token_count",
    "_brain_last_error",
    "_brain_source_utility",
    "_checkpoint_dir",
    "_checkpoint_path",
    "_concept_store",
    "_delayed_consequence_compacted_total",
    "_delayed_consequence_cooled_total",
    "_delayed_consequence_records",
    "_delayed_consequence_remerged_total",
    "_delayed_consequence_retired_total",
    "_delayed_consequence_split_total",
    "_encoder",
    "_env_root",
    "_geometric_curiosity",
    "_interaction_pipeline",
    "_metadata",
    "_replay_sample_history",
    "_runtime_config",
    "_runtime_env",
    "_runtime_state",
    "_trace_dir",
    "_trainer",
})


@dataclass(frozen=True)
class RuntimePersistenceDependencies:
    get_state: Callable[[str], Any]
    set_state: Callable[[str, Any], None]
    brain_persisted_state: Callable[[], Mapping[str, Any]]
    brain_runtime_snapshot: Callable[..., Mapping[str, Any]]
    join_brain_thread: Callable[..., Any]
    lock: Any
    normalize_background_source_utility_state: Callable[[Any], dict[str, Any]]
    normalize_delayed_consequence_record: Callable[[Any], dict[str, Any] | None]
    rebuild_brain_sources: Callable[[], None]
    request_brain_stop: Callable[..., Any]


class RuntimePersistence:
    """Checkpoint, trace-history, and JSON-safe persistence helpers."""

    def __init__(
        self,
        dependencies: RuntimePersistenceDependencies,
        *,
        trace_history_limit: int = 200,
        trace_history: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        object.__setattr__(self, "_dependencies", dependencies)
        self._trace_history: deque[dict[str, Any]] = deque(maxlen=max(1, int(trace_history_limit)))
        self.load_persisted_traces(trace_history or [])

    def __getattr__(self, name: str) -> Any:
        if name in _FORWARDED_STATE_NAMES:
            return self._dependencies.get_state(name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_dependencies", "_trace_history"}:
            object.__setattr__(self, name, value)
            return
        if name in _FORWARDED_STATE_NAMES:
            self._dependencies.set_state(name, value)
            return
        object.__setattr__(self, name, value)

    @property
    def _lock(self) -> Any:
        return self._dependencies.lock

    def _brain_persisted_state_locked(self) -> Mapping[str, Any]:
        return self._dependencies.brain_persisted_state()

    def _brain_runtime_snapshot_locked(self, **kwargs: Any) -> Mapping[str, Any]:
        return self._dependencies.brain_runtime_snapshot(**kwargs)

    def _join_brain_thread(self, *args: Any, **kwargs: Any) -> Any:
        return self._dependencies.join_brain_thread(*args, **kwargs)

    def _normalize_background_source_utility_state(self, value: Any) -> dict[str, Any]:
        return self._dependencies.normalize_background_source_utility_state(value)

    def _normalize_delayed_consequence_record(self, value: Any) -> dict[str, Any] | None:
        return self._dependencies.normalize_delayed_consequence_record(value)

    def _rebuild_brain_sources_locked(self) -> None:
        self._dependencies.rebuild_brain_sources()

    def _request_brain_stop(self, *args: Any, **kwargs: Any) -> Any:
        return self._dependencies.request_brain_stop(*args, **kwargs)

    @property
    def trace_history(self) -> deque[dict[str, Any]]:
        return self._trace_history

    @trace_history.setter
    def trace_history(self, value: Sequence[Mapping[str, Any]]) -> None:
        self.load_persisted_traces(value)

    def checkpoint_list(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self._checkpoint_dir.exists():
                return []
            records: list[dict[str, Any]] = []
            for path in sorted(self._checkpoint_dir.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True):
                stat = path.stat()
                records.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "size_bytes": int(stat.st_size),
                        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    }
                )
            return records

    def recent_traces(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            count = max(1, int(limit))
            return [deepcopy(trace) for trace in list(self._trace_history)[:count]]

    def persist_trace(self, trace: dict[str, Any]) -> Path:
        with self._lock:
            return self._persist_trace_locked(trace)

    def load_persisted_traces(self, trace_history: Sequence[Mapping[str, Any]]) -> None:
        normalized = [
            deepcopy(dict(item))
            for item in trace_history
            if isinstance(item, Mapping)
        ]
        self._trace_history.clear()
        self._trace_history.extend(normalized)

    def save_checkpoint(self, path: str | None = None) -> dict[str, Any]:
        with self._lock:
            target = self._resolve_save_path(path)
            metadata = deepcopy(self._metadata)
            service_state = dict(metadata.get("service_state", {}))
            service_state["concept_store"] = self._concept_store.state_dict()
            service_state["terminus_runtime"] = self._brain_persisted_state_locked()
            metadata.update(
                {
                    "saved_by": "hecsn.service",
                    "state_revision": int(self._runtime_state.state_revision),
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "service_state": service_state,
                }
            )
            saved_path = save_trainer_checkpoint(target, self._trainer, metadata=metadata)
            self._checkpoint_path = saved_path
            self._checkpoint_dir = saved_path.parent
            self._metadata = metadata
            self._runtime_state.mark_clean()
            return {
                "path": str(saved_path),
                **self._runtime_state.mutation_summary(),
                "token_count": int(self._trainer.token_count),
            }

    def restore_checkpoint(self, path: str | Path) -> dict[str, Any]:
        thread = self._request_brain_stop()
        self._join_brain_thread(thread)
        with self._lock:
            checkpoint_path = Path(path)
            trainer, metadata = load_trainer_checkpoint(checkpoint_path)
            self._trainer = trainer
            self._metadata = dict(metadata)
            self._encoder = self._trainer.encoder
            self._checkpoint_path = checkpoint_path
            self._checkpoint_dir = checkpoint_path.parent if checkpoint_path.parent != Path("") else Path("checkpoints")
            self._runtime_env = load_runtime_env(anchor_paths=(self._env_root, self._checkpoint_dir))
            self._action_root = (self._env_root or self._checkpoint_dir).resolve()
            service_state = dict(self._metadata.get("service_state", {}))
            terminus_state = dict(service_state.get("terminus_runtime", service_state.get("brain_runtime")) or {})
            concept_state = service_state.get("concept_store")
            self._concept_store = ConceptStore.from_state_dict(concept_state)
            geometric_curiosity_state = cast(
                dict[str, Any] | None,
                terminus_state.get("geometric_curiosity"),
            )
            self._geometric_curiosity = GeometricCuriosityController.from_state_dict(
                self._trainer.model.abstraction_layer,
                geometric_curiosity_state,
            )
            self._brain_config = self._runtime_config._normalize_brain_config(terminus_state)
            self._brain_source_utility = self._normalize_background_source_utility_state(
                terminus_state.get("background_source_utility")
            )
            self._brain_last_error = None
            self._action_history = list(terminus_state.get("action_history") or [])
            self._replay_sample_history = list(terminus_state.get("replay_sample_history") or [])
            self._delayed_consequence_records = deque(
                (
                    item
                    for item in (
                        self._normalize_delayed_consequence_record(raw_item)
                        for raw_item in list(terminus_state.get("delayed_consequence_records") or [])
                    )
                    if item is not None
                ),
                maxlen=DEFAULT_DELAYED_CONSEQUENCE_RECORDS,
            )
            self._delayed_consequence_cooled_total = max(0, int(terminus_state.get("delayed_consequence_cooled_total", 0) or 0))
            self._delayed_consequence_retired_total = max(0, int(terminus_state.get("delayed_consequence_retired_total", 0) or 0))
            self._delayed_consequence_compacted_total = max(0, int(terminus_state.get("delayed_consequence_compacted_total", 0) or 0))
            self._delayed_consequence_split_total = max(0, int(terminus_state.get("delayed_consequence_split_total", 0) or 0))
            self._delayed_consequence_remerged_total = max(0, int(terminus_state.get("delayed_consequence_remerged_total", 0) or 0))
            self._brain_last_acquisition_summary = None
            self._brain_last_acquisition_token_count = int(self._trainer.token_count)
            self._rebuild_brain_sources_locked()
            self._interaction_pipeline.load_interaction_state(
                recent_query_gaps=list(terminus_state.get("recent_query_gaps") or []),
                runtime_episode_traces=list(terminus_state.get("runtime_episode_traces") or []),
            )
            self._runtime_state.restore_event_history(
                last_event=terminus_state.get("last_event"),
                recent_events=terminus_state.get("recent_events"),
            )
            self._runtime_state.restore_clean()
            return {
                "path": str(checkpoint_path),
                **self._runtime_state.mutation_summary(),
                "token_count": int(self._trainer.token_count),
            }

    def _service_state_snapshot(self, *, include_replay_dataset_summary: bool = True) -> dict[str, Any]:
        last_trace = self._trace_history[0] if self._trace_history else None
        return {
            "checkpoint_path": str(self._checkpoint_path),
            **self._runtime_state.mutation_summary(),
            "token_count": int(self._trainer.token_count),
            "last_trace_id": None if last_trace is None else str(last_trace.get("trace_id")),
            "concept_count": int(self._concept_store.snapshot().get("concept_count", 0)),
            "terminus_runtime": self._brain_runtime_snapshot_locked(
                include_replay_dataset_summary=include_replay_dataset_summary,
            ),
        }

    def _resolve_save_path(self, path: str | None) -> Path:
        if path:
            return Path(path)

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return self._checkpoint_dir / f"hecsn_service_{stamp}.pt"

    def _persist_trace_locked(self, trace: dict[str, Any]) -> Path:
        payload = self._json_safe(trace)
        created_at = str(payload.get("created_at", datetime.now(timezone.utc).isoformat()))
        timestamp = created_at.replace(":", "").replace("-", "")
        trace_id = str(payload.get("trace_id", uuid4()))
        trace_path = self._trace_dir / f"{timestamp}_{trace_id}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_file(trace_path, payload)
        payload["trace_path"] = str(trace_path)
        self._trace_history.appendleft(deepcopy(payload))
        return trace_path

    def _load_persisted_traces_locked(self) -> None:
        trace_files = sorted(self._trace_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        for path in trace_files[: self._trace_history.maxlen]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            payload["trace_path"] = str(path)
            self._trace_history.append(payload)

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, deque)):
            return [self._json_safe(item) for item in value]
        return str(value)

    def _record_brain_event_locked(self, event: dict[str, Any]) -> None:
        self._runtime_state.record_event(event)
