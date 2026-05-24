from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import logging as _logging
from pathlib import Path
from threading import Event, Lock, Thread
import time
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

DEFAULT_CORTEX_INIT_TIMEOUT_SECONDS = 2.0

_cortex_logger = _logging.getLogger(__name__ + ".cortex")


def _build_cortex_controller_initial_state() -> dict[str, Any]:
    return {
        "_thought_loop_actual": None,
        "_lazy_thought_loop": None,
        "_cortex_available": False,
        "_cortex_init_lock": Lock(),
        "_cortex_init_event": Event(),
        "_cortex_init_thread": None,
        "_cortex_init_started": False,
        "_cortex_init_finished": False,
        "_cortex_init_timed_out": False,
        "_cortex_init_error": None,
        "_cortex_factory_refs": None,
        "_last_cortex_query_hint_text": None,
        "_last_cortex_query_hint_at": 0.0,
    }


CORTEX_CONTROLLER_STATE_FIELDS = frozenset(_build_cortex_controller_initial_state())


class _LazyThoughtLoop:
    """Compatibility proxy that initializes the real ThoughtLoop on active use."""

    def __init__(self, controller: "CortexController") -> None:
        self._controller = controller

    def _get(self) -> Any:
        loop = self._controller._ensure_cortex_initialized()
        if loop is None:
            raise RuntimeError("cortex_unavailable")
        return loop

    @property
    def is_running(self) -> bool:
        loop = self._controller._thought_loop_actual
        return bool(loop is not None and loop.is_running)

    def snapshot(self) -> dict[str, Any]:
        loop = self._controller._thought_loop_actual
        if loop is None:
            return self._controller._cortex_unavailable_snapshot()
        return loop.snapshot()

    def start(self) -> None:
        self._get().start()

    def stop(self, timeout: float = 5.0) -> None:
        loop = self._controller._thought_loop_actual
        if loop is not None:
            loop.stop(timeout=timeout)

    def request_stop(self) -> None:
        loop = self._controller._thought_loop_actual
        if loop is not None:
            loop.request_stop()

    def submit_query(self, query: str) -> None:
        self._get().submit_query(query)

    def request_sleep(self, **kwargs: Any) -> dict[str, Any]:
        return self._get().request_sleep(**kwargs)

    def inject_action_result(self, **kwargs: Any) -> None:
        self._get().inject_action_result(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get(), name)


@dataclass(frozen=True)
class CortexControllerDependencies:
    action_history: Callable[[], Sequence[Mapping[str, Any]]]
    action_history_memory_metadata: Callable[[Mapping[str, Any]], dict[str, Any]]
    action_query_terms: Callable[[str], tuple[str, ...]]
    action_focus_query_text: Callable[[str], str]
    api_request_record_matches_explicit_url: Callable[[Mapping[str, Any], str], bool]
    checkpoint_dir: Callable[[], Path]
    cortex_signal_state: Callable[[], dict[str, Any]]
    lock: Any
    query_api_url_candidate: Callable[[str], str]
    query_web_url_candidate: Callable[[str], str]
    query_workspace_path_candidate: Callable[[str], str]
    recent_relevant_action_records: Callable[..., list[dict[str, Any]]]
    record_brain_event: Callable[[Mapping[str, Any]], None]
    action_record_relevance_score: Callable[[Mapping[str, Any], str], float]
    action_record_to_response_episodes: Callable[..., list[dict[str, Any]]]
    augment_query_result_with_action_records: Callable[..., int]
    brain_running: Callable[[], bool]
    execute_digital_action: Callable[..., dict[str, Any]]


class CortexController:
    """Cortex ask/sleep/thought/action-intent control helpers."""

    def __init__(self, dependencies: CortexControllerDependencies) -> None:
        self._dependencies = dependencies
        for field_name, initial_value in _build_cortex_controller_initial_state().items():
            object.__setattr__(self, field_name, initial_value)
        object.__setattr__(self, "_lazy_thought_loop", _LazyThoughtLoop(self))

        object.__setattr__(self, "_cortex_init_finished", True)
        object.__setattr__(self, "_cortex_init_error", "Cortex runtime path retired; use Subcortex/Living Loop surfaces.")
        self._cortex_init_event.set()
        _cortex_logger.info("Cortex runtime path retired")

    @staticmethod
    def _normalize_action_text(value: Any) -> str:
        return " ".join(str(value).split()).strip()

    @staticmethod
    def _normalize_cortex_query_hint(value: Any) -> str:
        return " ".join(str(value).split()).strip()

    def _remember_cortex_query_hint_locked(self, query_text: str) -> None:
        normalized = self._normalize_cortex_query_hint(query_text)
        if not normalized:
            self._last_cortex_query_hint_text = None
            self._last_cortex_query_hint_at = 0.0
            return
        self._last_cortex_query_hint_text = normalized
        self._last_cortex_query_hint_at = time.time()

    def _consume_cortex_query_hint_locked(self, *, max_age_seconds: float = 30.0) -> str:
        hint = self._normalize_cortex_query_hint(self._last_cortex_query_hint_text or "")
        age = time.time() - float(self._last_cortex_query_hint_at or 0.0)
        self._last_cortex_query_hint_text = None
        self._last_cortex_query_hint_at = 0.0
        if not hint or age > max(0.0, float(max_age_seconds)):
            return ""
        return hint

    @property
    def _thought_loop(self) -> Any:
        if self._thought_loop_actual is not None:
            return self._thought_loop_actual
        if self._cortex_factories_are_mocked():
            return self._lazy_thought_loop
        return None

    @_thought_loop.setter
    def _thought_loop(self, value: Any) -> None:
        self._thought_loop_actual = value
        self._cortex_available = value is not None
        if value is not None:
            self._cortex_init_started = True
            self._cortex_init_finished = True
            self._cortex_init_error = None
            self._cortex_init_event.set()

    def _cortex_active(self) -> bool:
        return self._thought_loop_actual is not None and self._thought_loop_actual.is_running

    @property
    def _action_history(self) -> Sequence[Mapping[str, Any]]:
        return self._dependencies.action_history()

    @property
    def _brain_running(self) -> bool:
        return bool(self._dependencies.brain_running())

    @property
    def _checkpoint_dir(self) -> Path:
        return self._dependencies.checkpoint_dir()

    @property
    def _lock(self) -> Any:
        return self._dependencies.lock

    def _record_brain_event_locked(self, event: Mapping[str, Any]) -> None:
        return self._dependencies.record_brain_event(event)

    def _action_history_memory_metadata(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return self._dependencies.action_history_memory_metadata(record)

    def _action_query_terms(self, query_text: str) -> tuple[str, ...]:
        return self._dependencies.action_query_terms(query_text)

    def _action_focus_query_text(self, query_text: str) -> str:
        return self._dependencies.action_focus_query_text(query_text)

    def _query_workspace_path_candidate_locked(self, query_text: str) -> str:
        return self._dependencies.query_workspace_path_candidate(query_text)

    def _query_web_url_candidate(self, query_text: str) -> str:
        return self._dependencies.query_web_url_candidate(query_text)

    def _query_api_url_candidate(self, query_text: str) -> str:
        return self._dependencies.query_api_url_candidate(query_text)

    def _api_request_record_matches_explicit_url(self, record: Mapping[str, Any], explicit_url: str) -> bool:
        return self._dependencies.api_request_record_matches_explicit_url(record, explicit_url)

    def _cortex_signal_state(self) -> dict[str, Any]:
        return self._dependencies.cortex_signal_state()

    def execute_digital_action(self, inputs: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self._dependencies.execute_digital_action(inputs, **kwargs)

    def _recent_relevant_action_records_locked(
        self,
        query_text: str,
        *,
        statuses: Sequence[str] | None = None,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        return self._dependencies.recent_relevant_action_records(
            query_text,
            statuses=statuses,
            limit=limit,
        )

    def _action_record_relevance_score_locked(self, record: Mapping[str, Any], query_text: str) -> float:
        return self._dependencies.action_record_relevance_score(record, query_text)

    def _action_record_to_response_episodes_locked(
        self,
        record: Mapping[str, Any],
        *,
        query_text: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        return self._dependencies.action_record_to_response_episodes(
            record,
            query_text=query_text,
            limit=limit,
        )

    def _augment_query_result_with_action_records_locked(
        self,
        query_result: dict[str, Any],
        *,
        query_text: str,
        records: Sequence[Mapping[str, Any]],
    ) -> int:
        return self._dependencies.augment_query_result_with_action_records(
            query_result,
            query_text=query_text,
            records=records,
        )

    def _cortex_factories_are_mocked(self) -> bool:
        refs = self._cortex_factory_refs or ()
        return any(
            "unittest.mock" in type(ref).__module__ or hasattr(ref, "mock_calls")
            for ref in refs
        )

    def _cortex_unavailable_snapshot(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "retired": True,
            "reason": "cortex_runtime_retired",
            "replacement": "subcortex_living_loop",
            "initialization": {
                "started": bool(getattr(self, "_cortex_init_started", False)),
                "finished": bool(getattr(self, "_cortex_init_finished", False)),
                "timed_out": bool(getattr(self, "_cortex_init_timed_out", False)),
                "error": getattr(self, "_cortex_init_error", None),
            },
        }

    def _inject_action_record_into_loop(self, thought_loop: Any, record: Mapping[str, Any]) -> None:
        content = " ".join(str(record.get("episode_text", "")).split()).strip()
        if not content:
            return
        verification_raw = record.get("verification")
        verification: Mapping[str, Any] = verification_raw if isinstance(verification_raw, Mapping) else {}
        thought_loop.inject_action_result(
            content=content,
            topics=tuple(str(item) for item in list(record.get("topics") or []) if str(item).strip()),
            success=bool(verification.get("success", False)),
            confidence=float(verification.get("confidence", 0.0) or 0.0),
            contradicted=bool(verification.get("contradiction", False)),
            metadata=self._action_history_memory_metadata(record),
        )

    def _build_cortex_thought_loop(self, action_history: Sequence[Mapping[str, Any]]) -> Any:
        if self._cortex_factory_refs is None:
            raise RuntimeError(self._cortex_init_error or "Cortex module unavailable")
        ThoughtLoop, create_cortex_from_env, create_embedder_from_env, EpisodicMemory = self._cortex_factory_refs
        cortex = create_cortex_from_env()
        embedder = create_embedder_from_env(allow_fallback=False)
        memory = EpisodicMemory(capacity=2048, embedder=embedder)
        thought_loop = ThoughtLoop(
            cortex=cortex,
            memory=memory,
            curiosity_controller=getattr(self, "_geometric_curiosity", None),
            signal_provider=self._cortex_signal_state,
            narrative_state_path=str(Path(self._checkpoint_dir) / "cortex_narrative_self.json"),
            on_thought=self._on_cortex_thought,
            on_sleep_summary=self._on_cortex_sleep_cycle,
        )
        for record in reversed(list(action_history)):
            self._inject_action_record_into_loop(thought_loop, record)
        _cortex_logger.info("Cortex module initialised (%s, embedder=%s)", cortex.model, type(embedder).__name__)
        return thought_loop

    def _start_cortex_initialization(self) -> None:
        with self._cortex_init_lock:
            if self._thought_loop_actual is not None:
                self._cortex_init_event.set()
                return
            if self._cortex_init_thread is not None and self._cortex_init_thread.is_alive():
                return
            if self._cortex_init_finished and self._cortex_init_error:
                return
            self._cortex_init_started = True
            self._cortex_init_finished = False
            self._cortex_init_timed_out = False
            self._cortex_init_error = None
            self._cortex_init_event.clear()
            action_history = list(self._action_history)

            def _runner() -> None:
                try:
                    thought_loop = self._build_cortex_thought_loop(action_history)
                except RuntimeError as exc:
                    self._cortex_init_error = str(exc)
                    _cortex_logger.warning("Cortex disabled: %s", exc)
                except Exception as exc:  # pragma: no cover - defensive init guard
                    self._cortex_init_error = str(exc)
                    _cortex_logger.info("Cortex module unavailable: %s", exc)
                else:
                    self._thought_loop_actual = thought_loop
                    self._cortex_available = True
                    if bool(getattr(self, "_brain_running", False)) and not thought_loop.is_running:
                        try:
                            thought_loop.start()
                            _cortex_logger.info("ThoughtLoop started after delayed cortex initialization")
                        except Exception as exc:
                            _cortex_logger.warning("ThoughtLoop failed to start after delayed initialization: %s", exc)
                finally:
                    self._cortex_init_finished = True
                    self._cortex_init_event.set()

            self._cortex_init_thread = Thread(target=_runner, name="hecsn-cortex-init", daemon=True)
            self._cortex_init_thread.start()

    def _ensure_cortex_initialized(self, *, wait_seconds: float | None = DEFAULT_CORTEX_INIT_TIMEOUT_SECONDS) -> Any:
        self._cortex_init_started = False
        self._cortex_init_finished = True
        self._cortex_init_error = "Cortex runtime path retired; use Subcortex/Living Loop surfaces."
        self._cortex_init_event.set()
        return None
        if self._thought_loop_actual is not None:
            return self._thought_loop_actual
        self._start_cortex_initialization()
        if self._thought_loop_actual is not None:
            return self._thought_loop_actual
        if wait_seconds is not None:
            if not self._cortex_init_event.wait(timeout=max(0.0, float(wait_seconds))):
                self._cortex_init_timed_out = True
                _cortex_logger.warning("Cortex initialization still pending after %.2fs", float(wait_seconds))
                return None
        return self._thought_loop_actual

    def _request_cortex_sleep_locked(
        self,
        *,
        source: str,
        reason: str,
        query_text: str = "",
        thought_text: str = "",
        topics: Sequence[str] = (),
    ) -> dict[str, Any]:
        thought_loop = self._thought_loop_actual
        if thought_loop is None:
            return {"accepted": False, "reason": "cortex_unavailable"}
        normalized_source = self._normalize_action_text(source).lower() or "operator"
        normalized_reason = self._normalize_action_text(reason)
        normalized_query = self._normalize_cortex_query_hint(query_text)
        normalized_thought = self._normalize_action_text(thought_text)[:240]
        normalized_topics = [
            self._normalize_action_text(topic).lower()
            for topic in list(topics)[:4]
            if self._normalize_action_text(topic)
        ]
        control_id = str(uuid4())
        request_value = thought_loop.request_sleep(
            source=normalized_source,
            reason=normalized_reason or (
                "Operator requested cortex sleep."
                if normalized_source == "operator"
                else "Cortex requested a sleep cycle."
            ),
            metadata={
                "control_id": control_id,
                "query_text": normalized_query,
                "thought_text": normalized_thought,
                "topics": normalized_topics,
            },
        )
        request: Mapping[str, Any] = request_value if isinstance(request_value, Mapping) else {}
        request_payload_value = request.get("request")
        request_payload_mapping: Mapping[str, Any] = (
            request_payload_value if isinstance(request_payload_value, Mapping) else {}
        )
        request_payload: dict[str, Any] = deepcopy(dict(request_payload_mapping))
        metadata_value = request_payload.get("metadata")
        metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
        self._record_brain_event_locked(
            {
                "type": "cortex_sleep_requested",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "control_id": str(metadata.get("control_id", control_id)),
                "source": normalized_source,
                "action_intent": "sleep" if normalized_source == "cortex_intent" else None,
                "reason": str(request_payload.get("reason", normalized_reason)),
                "query_text": str(metadata.get("query_text", normalized_query)),
                "thought_text": str(metadata.get("thought_text", normalized_thought)),
                "topics": list(metadata.get("topics") or normalized_topics)[:4],
                "coalesced": bool(request.get("coalesced", False)),
            }
        )
        return {
            "accepted": bool(request.get("accepted", False)),
            "coalesced": bool(request.get("coalesced", False)),
            "running": bool(request.get("running", False)),
            "request": request_payload,
            "sleep_control": deepcopy(request.get("sleep_control") or {}),
        }

    def _handle_cortex_sleep_intent_locked(self, result: Any) -> dict[str, Any] | None:
        query_hint = self._consume_cortex_query_hint_locked()
        thought_text = self._normalize_action_text(getattr(result, "thought", ""))
        topics = [
            self._normalize_action_text(topic)
            for topic in list(getattr(result, "topics", ()) or ())
            if self._normalize_action_text(topic)
        ]
        return self._request_cortex_sleep_locked(
            source="cortex_intent",
            reason=thought_text or "Cortex requested a sleep cycle.",
            query_text=query_hint,
            thought_text=thought_text,
            topics=topics,
        )

    def _on_cortex_sleep_cycle(self, summary: dict[str, Any]) -> None:
        if not isinstance(summary, Mapping) or not bool(summary.get("requested", False)):
            return
        request_value = summary.get("request")
        request = request_value if isinstance(request_value, Mapping) else {}
        metadata_value = request.get("metadata")
        metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
        source = self._normalize_action_text(request.get("source", "")).lower() or "unknown"
        topics = [
            self._normalize_action_text(topic).lower()
            for topic in list(metadata.get("topics") or [])[:4]
            if self._normalize_action_text(topic)
        ]
        with self._lock:
            self._record_brain_event_locked(
                {
                    "type": "cortex_sleep_completed",
                    "timestamp": str(summary.get("completed_at") or datetime.now(timezone.utc).isoformat()),
                    "control_id": self._normalize_action_text(metadata.get("control_id", "")),
                    "source": source,
                    "action_intent": "sleep" if source == "cortex_intent" else None,
                    "reason": self._normalize_action_text(request.get("reason", "")),
                    "query_text": self._normalize_cortex_query_hint(metadata.get("query_text", "")),
                    "thought_text": self._normalize_action_text(metadata.get("thought_text", "")),
                    "topics": topics,
                    "dreams_generated": int(summary.get("dreams_generated", 0) or 0),
                    "sleep_cycles": int(summary.get("sleep_cycles", 0) or 0),
                    "trigger": self._normalize_action_text(summary.get("trigger", "")),
                }
            )

    def _cortex_action_query_locked(self, result: Any, *, query_hint: str) -> str:
        if query_hint:
            terms = self._action_query_terms(query_hint)
            if terms:
                return " ".join(terms[:4])
            return query_hint
        topics = [
            self._normalize_action_text(topic)
            for topic in list(getattr(result, "topics", ()) or ())
            if self._normalize_action_text(topic)
        ]
        if topics:
            return " ".join(topics[:4])
        thought = self._normalize_action_text(getattr(result, "thought", ""))
        if thought:
            terms = self._action_query_terms(thought)
            if terms:
                return " ".join(terms[:4])
            return thought[:160]
        return ""

    def _filter_cortex_action_records_locked(
        self,
        records: Sequence[dict[str, Any]],
        *,
        explicit_api_url: str,
        explicit_url: str,
        explicit_path: str,
    ) -> list[dict[str, Any]]:
        if explicit_api_url:
            return [
                record
                for record in records
                if self._api_request_record_matches_explicit_url(record, explicit_api_url)
            ]
        if explicit_url:
            return [
                record
                for record in records
                if str(record.get("action_type", "")) == "web_fetch"
                and self._normalize_action_text((record.get("inputs") or {}).get("url", "")) == explicit_url
            ]
        if explicit_path:
            return [
                record
                for record in records
                if str(record.get("action_type", "")) == "workspace_read"
                and self._normalize_action_text((record.get("inputs") or {}).get("path", "")) == explicit_path
            ]
        return list(records)

    @staticmethod
    def _cortex_action_type(
        *,
        explicit_api_url: str,
        explicit_url: str,
        explicit_path: str,
    ) -> str:
        if explicit_api_url:
            return "api_request"
        if explicit_url:
            return "web_fetch"
        if explicit_path:
            return "workspace_read"
        return "workspace_search"

    @classmethod
    def _cortex_action_trigger_reason(cls, action_intent: str, action_type: str) -> str:
        normalized_intent = cls._normalize_action_text(action_intent).lower()
        normalized_action = cls._normalize_action_text(action_type).lower()
        if normalized_intent == "search":
            if normalized_action == "api_request":
                return "cortex_action_api_request"
            if normalized_action == "web_fetch":
                return "cortex_action_fetch"
            if normalized_action == "workspace_read":
                return "cortex_action_read"
            return "cortex_action_search"
        if normalized_intent in {"ask", "remember", "explore"}:
            return f"cortex_action_{normalized_intent}"
        return "cortex_action_search"

    def _handle_cortex_action_intent_locked(
        self,
        result: Any,
        *,
        action_intent: str | None = None,
    ) -> dict[str, Any] | None:
        normalized_intent = self._normalize_action_text(action_intent or getattr(result, "action_intent", "")).lower()
        if normalized_intent not in {"search", "ask", "remember", "explore"}:
            return None
        query_hint = self._consume_cortex_query_hint_locked()
        search_query = self._cortex_action_query_locked(result, query_hint=query_hint)
        if not search_query:
            return None
        target_query = query_hint or search_query
        explicit_api_url = self._query_api_url_candidate(target_query)
        explicit_url = self._query_web_url_candidate(target_query) if not explicit_api_url else ""
        explicit_path = self._query_workspace_path_candidate_locked(target_query) if not (explicit_api_url or explicit_url) else ""
        recent_verified = self._filter_cortex_action_records_locked(
            self._recent_relevant_action_records_locked(target_query, statuses=("verified",), limit=2),
            explicit_api_url=explicit_api_url,
            explicit_url=explicit_url,
            explicit_path=explicit_path,
        )
        if recent_verified:
            self._record_brain_event_locked(
                {
                    "type": "cortex_action_reused",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action_intent": normalized_intent,
                    "query_text": target_query,
                    "action_id": str(recent_verified[0].get("action_id", "")),
                }
            )
            return {"reused": True, "record": deepcopy(recent_verified[0])}
        recent_contradicted = self._filter_cortex_action_records_locked(
            self._recent_relevant_action_records_locked(target_query, statuses=("contradicted",), limit=1),
            explicit_api_url=explicit_api_url,
            explicit_url=explicit_url,
            explicit_path=explicit_path,
        )
        if recent_contradicted:
            self._record_brain_event_locked(
                {
                    "type": "cortex_action_reused",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action_intent": normalized_intent,
                    "query_text": target_query,
                    "action_id": str(recent_contradicted[0].get("action_id", "")),
                }
            )
            return {"reused": True, "record": deepcopy(recent_contradicted[0])}

        action_type = self._cortex_action_type(
            explicit_api_url=explicit_api_url,
            explicit_url=explicit_url,
            explicit_path=explicit_path,
        )
        self._record_brain_event_locked(
            {
                "type": "cortex_action_requested",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action_intent": normalized_intent,
                "query_text": target_query,
                "topics": list(getattr(result, "topics", ()) or ())[:4],
                "action_type": action_type,
            }
        )
        focused_query = self._action_focus_query_text(target_query)
        intent_label = f"Cortex {normalized_intent} intent"
        if action_type == "api_request":
            return self.execute_digital_action(
                {
                    "action_type": "api_request",
                    "url": explicit_api_url,
                    "query_text": focused_query,
                    "predicted_outcome": f"{intent_label} expects requesting structured JSON from {explicit_api_url} to reveal grounded evidence relevant to: {target_query}.",
                },
                trigger_reason=self._cortex_action_trigger_reason(normalized_intent, action_type),
                trigger_query_text=target_query,
            )
        if action_type == "web_fetch":
            return self.execute_digital_action(
                {
                    "action_type": "web_fetch",
                    "url": explicit_url,
                    "query_text": focused_query,
                    "predicted_outcome": f"{intent_label} expects fetching {explicit_url} to reveal grounded evidence relevant to: {target_query}.",
                },
                trigger_reason=self._cortex_action_trigger_reason(normalized_intent, action_type),
                trigger_query_text=target_query,
            )
        if action_type == "workspace_read":
            return self.execute_digital_action(
                {
                    "action_type": "workspace_read",
                    "path": explicit_path,
                    "query_text": focused_query,
                    "predicted_outcome": f"{intent_label} expects reading {explicit_path} to reveal grounded workspace evidence relevant to: {target_query}.",
                },
                trigger_reason=self._cortex_action_trigger_reason(normalized_intent, action_type),
                trigger_query_text=target_query,
            )
        return self.execute_digital_action(
            {
                "action_type": "workspace_search",
                "query_text": search_query,
                "predicted_outcome": f"{intent_label} expects grounded workspace evidence relevant to: {target_query}.",
            },
            trigger_reason=self._cortex_action_trigger_reason(normalized_intent, action_type),
            trigger_query_text=target_query,
        )

    def _on_cortex_thought(self, result: Any) -> None:
        action_intent = self._normalize_action_text(getattr(result, "action_intent", "")).lower()
        if action_intent == "sleep":
            with self._lock:
                self._handle_cortex_sleep_intent_locked(result)
            return
        if action_intent not in {"search", "ask", "remember", "explore"}:
            return
        with self._lock:
            self._handle_cortex_action_intent_locked(result, action_intent=action_intent)

    def cortex_ask(self, query: str) -> dict[str, Any]:
        """Submit a question to the cortex and return immediately."""
        thought_loop = self._ensure_cortex_initialized()
        if thought_loop is None:
            return {"accepted": False, "reason": "cortex_runtime_retired", "replacement": "runtime.respond"}
        with self._lock:
            self._remember_cortex_query_hint_locked(query)
        thought_loop.submit_query(query)
        return {"accepted": True, "query": query}

    def cortex_sleep(self, reason: str | None = None) -> dict[str, Any]:
        """Request an explicit cortex sleep cycle on the maintained control path."""
        normalized_reason = self._normalize_action_text(reason or "")
        self._ensure_cortex_initialized()
        with self._lock:
            return self._request_cortex_sleep_locked(
                source="operator",
                reason=normalized_reason or "Operator requested cortex sleep.",
            )

    def cortex_thoughts(self, limit: int = 20) -> dict[str, Any]:
        """Return recent thoughts from the cortex thought loop."""
        thought_loop = self._thought_loop_actual
        if thought_loop is None:
            return {"enabled": False, "retired": True, "reason": "cortex_runtime_retired", "thoughts": []}
        snap = thought_loop.snapshot()
        thoughts = snap.get("recent_thoughts", [])
        return {
            "enabled": True,
            "running": snap.get("running", False),
            "thoughts_generated": snap.get("thoughts_generated", 0),
            "dreams_generated": snap.get("dreams_generated", 0),
            "current_mode": snap.get("current_mode", "idle"),
            "thoughts": thoughts[-limit:],
        }

    def cortex_snapshot(self) -> dict[str, Any]:
        """Full cortex status snapshot."""
        if self._thought_loop_actual is None:
            return self._cortex_unavailable_snapshot()
        return self._thought_loop_actual.snapshot()

