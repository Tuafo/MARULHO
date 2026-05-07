from __future__ import annotations

from collections import deque
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence


DEFAULT_BRAIN_EVENT_HISTORY = 16


class RuntimeState:
    """Private runtime truth container for dirty state, revision, and brain events."""

    def __init__(
        self,
        *,
        lock: RLock | None = None,
        history_limit: int = DEFAULT_BRAIN_EVENT_HISTORY,
    ) -> None:
        self._lock = lock
        self._dirty_state = False
        self._state_revision = 0
        self._brain_last_event: dict[str, Any] | None = None
        self._brain_event_history: deque[dict[str, Any]] = deque(maxlen=max(1, int(history_limit)))

    def _state_guard(self):
        """Return the shared runtime lock, or a no-op guard for standalone use."""
        return self._lock if self._lock is not None else nullcontext()

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): RuntimeState._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, deque)):
            return [RuntimeState._json_safe(item) for item in value]
        return str(value)

    @classmethod
    def _json_safe_event(cls, event: Mapping[str, Any]) -> dict[str, Any]:
        return {str(key): cls._json_safe(item) for key, item in event.items()}

    @property
    def dirty_state(self) -> bool:
        with self._state_guard():
            return bool(self._dirty_state)

    @dirty_state.setter
    def dirty_state(self, value: bool) -> None:
        with self._state_guard():
            self._dirty_state = bool(value)

    @property
    def state_revision(self) -> int:
        with self._state_guard():
            return int(self._state_revision)

    @state_revision.setter
    def state_revision(self, value: int) -> None:
        with self._state_guard():
            self._state_revision = max(0, int(value))

    @property
    def last_event(self) -> dict[str, Any] | None:
        with self._state_guard():
            return None if self._brain_last_event is None else deepcopy(self._brain_last_event)

    @property
    def recent_events(self) -> list[dict[str, Any]]:
        with self._state_guard():
            return [deepcopy(event) for event in list(self._brain_event_history)]

    def mark_mutated(self) -> None:
        with self._state_guard():
            self._dirty_state = True
            self._state_revision += 1

    def mark_dirty_without_revision(self) -> None:
        with self._state_guard():
            self._dirty_state = True

    def mark_clean(self) -> None:
        with self._state_guard():
            self._dirty_state = False

    def restore_clean(self) -> None:
        with self._state_guard():
            self._dirty_state = False
            self._state_revision += 1

    def record_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        with self._state_guard():
            payload = self._json_safe_event(event)
            self._brain_last_event = deepcopy(payload)
            self._brain_event_history.appendleft(deepcopy(payload))
            return deepcopy(payload)

    def restore_event_history(
        self,
        *,
        last_event: Mapping[str, Any] | None = None,
        recent_events: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        with self._state_guard():
            history_limit = max(1, int(self._brain_event_history.maxlen or DEFAULT_BRAIN_EVENT_HISTORY))
            normalized_events = [
                self._json_safe_event(event)
                for event in list(recent_events or [])
                if isinstance(event, Mapping)
            ][:history_limit]
            normalized_last_event = self._json_safe_event(last_event) if isinstance(last_event, Mapping) else None

            if normalized_last_event is not None:
                if normalized_events:
                    normalized_events[0] = deepcopy(normalized_last_event)
                else:
                    normalized_events = [deepcopy(normalized_last_event)]
                self._brain_last_event = deepcopy(normalized_last_event)
            elif normalized_events:
                self._brain_last_event = deepcopy(normalized_events[0])
            else:
                self._brain_last_event = None

            self._brain_event_history = deque(
                (deepcopy(event) for event in normalized_events),
                maxlen=history_limit,
            )

    def snapshot(self) -> dict[str, Any]:
        with self._state_guard():
            return {
                "dirty_state": bool(self._dirty_state),
                "state_revision": int(self._state_revision),
                "last_event": None if self._brain_last_event is None else deepcopy(self._brain_last_event),
                "recent_events": [deepcopy(event) for event in list(self._brain_event_history)],
            }
