from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4


class CortexRuntimeMixin:
    """Cortex ask/sleep/thought/action-intent control helpers."""

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
        request = thought_loop.request_sleep(
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
        request_payload = deepcopy(request.get("request") or {})
        metadata = request_payload.get("metadata") if isinstance(request_payload.get("metadata"), Mapping) else {}
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
        request = summary.get("request") if isinstance(summary.get("request"), Mapping) else {}
        metadata = request.get("metadata") if isinstance(request.get("metadata"), Mapping) else {}
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

        action_type = "api_request" if explicit_api_url else ("web_fetch" if explicit_url else ("workspace_read" if explicit_path else "workspace_search"))
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
        """Submit a question to the cortex and return immediately.

        The cortex will answer asynchronously in its next deliberation cycle.
        Returns acknowledgement with queue depth.
        """
        thought_loop = self._ensure_cortex_initialized()
        if thought_loop is None:
            return {"accepted": False, "reason": "cortex_unavailable"}
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
            return {"enabled": False, "thoughts": []}
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

