from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import re
from typing import Any, Callable, Mapping, Sequence, cast
from urllib.parse import urlparse
from uuid import uuid4

from hecsn.service.action_loop import execute_digital_action
from hecsn.service.history_store import read_history_record, replace_history_record
from hecsn.semantics.grounding_text import match_terms, salient_query_terms

DEFAULT_CORTEX_ACTION_INIT_TIMEOUT_SECONDS = 0.25
DEFAULT_RUNTIME_FEEDBACK_HISTORY = 8
DEFAULT_RUNTIME_FEEDBACK_EVIDENCE_LIMIT = 8
DEFAULT_RUNTIME_FEEDBACK_TAG_LIMIT = 12
DEFAULT_RUNTIME_FEEDBACK_MAX_TEXT_CHARS = 2000
SUPPORTED_ACTION_TYPES = ("workspace_search", "workspace_read", "web_fetch", "api_request")


class ActionExecutor:
    """Digital action execution, action-history ownership, and reuse helpers."""

    def __init__(
        self,
        *,
        lock: Any,
        action_root: str | Path,
        action_history: Sequence[Mapping[str, Any]] | None = None,
        history_maxlen: int = 24,
        runtime_state_mark_mutated_fn: Callable[[], None],
        runtime_state_mutation_summary_fn: Callable[[], dict[str, Any]],
        record_brain_event_fn: Callable[[Mapping[str, Any]], None],
        brain_runtime_snapshot_fn: Callable[[], dict[str, Any]],
        runtime_trace_export_safe_value_fn: Callable[[Any], Any],
        ensure_cortex_initialized_fn: Callable[[], Any | None],
        inject_action_record_into_cortex_fn: Callable[[Any, Mapping[str, Any]], None],
        apply_provider_outcome_calibration_fn: Callable[..., bool] | None = None,
    ) -> None:
        self._lock = lock
        self._action_root = Path(action_root).resolve()
        self._history_maxlen = max(1, int(history_maxlen))
        self._runtime_state_mark_mutated_fn = runtime_state_mark_mutated_fn
        self._runtime_state_mutation_summary_fn = runtime_state_mutation_summary_fn
        self._record_brain_event_fn = record_brain_event_fn
        self._brain_runtime_snapshot_fn = brain_runtime_snapshot_fn
        self._runtime_trace_export_safe_value_fn = runtime_trace_export_safe_value_fn
        self._ensure_cortex_initialized_fn = ensure_cortex_initialized_fn
        self._inject_action_record_into_cortex_fn = inject_action_record_into_cortex_fn
        self._apply_provider_outcome_calibration_fn = apply_provider_outcome_calibration_fn
        self._action_history: deque[dict[str, Any]] = deque(maxlen=self._history_maxlen)
        self.load_action_history(action_history or [])

    @property
    def history(self) -> deque[dict[str, Any]]:
        return self._action_history

    @history.setter
    def history(self, action_history: Sequence[Mapping[str, Any]]) -> None:
        self.load_action_history(action_history)

    def load_action_history(self, action_history: Sequence[Mapping[str, Any]]) -> None:
        normalized = [
            item
            for item in (self._normalize_action_record(raw_item) for raw_item in list(action_history))
            if item is not None
        ]
        self._action_history = deque(normalized, maxlen=self._history_maxlen)

    def action_history(self, limit: int = 20) -> dict[str, Any]:
        with self._lock:
            count = max(1, int(limit))
            history = [deepcopy(item) for item in list(self._action_history)[:count]]
            return {
                "count": int(len(self._action_history)),
                "root_path": str(self._action_root),
                "supported_actions": list(SUPPORTED_ACTION_TYPES),
                "actions": history,
            }

    def _normalize_action_record(self, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, Mapping):
            return None
        action_id = " ".join(str(item.get("action_id", "")).split()).strip()
        action_type = " ".join(str(item.get("action_type", item.get("type", ""))).split()).strip().lower()
        if not action_id or not action_type:
            return None
        verification = item.get("verification") if isinstance(item.get("verification"), Mapping) else {}
        topics = [
            " ".join(str(value).split()).strip().lower()
            for value in list(item.get("topics") or [])
            if " ".join(str(value).split()).strip()
        ]
        try:
            feedback_count = max(0, int(verification.get("feedback_count", 0) or 0))
        except (TypeError, ValueError):
            feedback_count = 0
        corrected_output = item.get("corrected_output")
        return {
            "action_id": action_id,
            "action_type": action_type,
            "inputs": deepcopy(dict(item.get("inputs") or {})),
            "predicted_outcome": " ".join(str(item.get("predicted_outcome", "")).split()).strip(),
            "actual_outcome": " ".join(str(item.get("actual_outcome", "")).split()).strip(),
            "verification": {
                "status": " ".join(str(verification.get("status", "unknown")).split()).strip().lower() or "unknown",
                "success": bool(verification.get("success", False)),
                "confidence": float(verification.get("confidence", 0.0) or 0.0),
                "contradiction": bool(verification.get("contradiction", False)),
                "summary": " ".join(str(verification.get("summary", "")).split()).strip(),
                "evidence": [deepcopy(dict(raw)) for raw in list(verification.get("evidence") or []) if isinstance(raw, Mapping)],
                "provenance": self._normalize_feedback_text(verification.get("provenance", ""), max_chars=32),
                "last_feedback_id": self._normalize_feedback_text(verification.get("last_feedback_id", ""), max_chars=80),
                "last_feedback_at": self._normalize_feedback_text(verification.get("last_feedback_at", ""), max_chars=80),
                "feedback_count": feedback_count,
            },
            "feedback": self._normalize_runtime_feedback_entries(item.get("feedback", [])),
            "feedback_status": self._normalize_feedback_text(item.get("feedback_status", ""), max_chars=32),
            "feedback_provenance": self._normalize_feedback_text(item.get("feedback_provenance", ""), max_chars=32),
            "provenance": self._normalize_feedback_text(item.get("provenance", ""), max_chars=32),
            "corrected_output": self._runtime_trace_export_safe_value_fn(corrected_output)
            if corrected_output is not None
            else None,
            "topics": topics[:8],
            "recorded_at": str(item.get("recorded_at") or datetime.now(timezone.utc).isoformat()),
            "episode_text": " ".join(str(item.get("episode_text", "")).split()).strip(),
            "trigger_reason": " ".join(str(item.get("trigger_reason", "operator")).split()).strip().lower() or "operator",
            "trigger_query_text": " ".join(str(item.get("trigger_query_text", "")).split()).strip(),
        }

    def action_record(self, action_id: str) -> dict[str, Any] | None:
        with self._lock:
            return read_history_record(self._action_history, record_id=action_id, id_field="action_id")

    def replace_action_record(self, action_id: str, record: Mapping[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            self._action_history, replaced = replace_history_record(
                self._action_history,
                record_id=action_id,
                replacement=record,
                id_field="action_id",
            )
            return replaced

    def execute_digital_action(
        self,
        action: Mapping[str, Any],
        *,
        trigger_reason: str | None = None,
        trigger_query_text: str | None = None,
    ) -> dict[str, Any]:
        action_type = " ".join(str(action.get("action_type", action.get("type", ""))).split()).strip().lower()
        if action_type not in SUPPORTED_ACTION_TYPES:
            return {"accepted": False, "reason": "unsupported_action_type", "action_type": action_type or None}

        requested_root = Path(str(action.get("root_path", ".") or "."))
        candidate_root = requested_root if requested_root.is_absolute() else (self._action_root / requested_root)
        try:
            resolved_root = candidate_root.resolve()
        except Exception:
            return {"accepted": False, "reason": "invalid_root_path", "action_type": action_type}
        if resolved_root != self._action_root and self._action_root not in resolved_root.parents:
            return {
                "accepted": False,
                "reason": "root_path_outside_workspace",
                "action_type": action_type,
                "workspace_root": str(self._action_root),
            }

        try:
            result = execute_digital_action(resolved_root, action)
        except Exception as exc:
            return {"accepted": False, "reason": "execution_failed", "action_type": action_type, "message": str(exc)}

        payload = result.to_payload()
        if trigger_reason is not None:
            payload["trigger_reason"] = str(trigger_reason)
        if trigger_query_text is not None:
            payload["trigger_query_text"] = str(trigger_query_text)
        normalized = self.normalize_action_record(payload)
        if normalized is None:
            return {"accepted": False, "reason": "normalization_failed", "action_type": action_type}

        with self._lock:
            existing = [
                item
                for item in list(self._action_history)
                if str(item.get("action_id", "")) != str(normalized.get("action_id", ""))
            ]
            self._action_history = deque(existing, maxlen=self._action_history.maxlen)
            self._action_history.appendleft(normalized)
            self._inject_action_record_into_cortex_locked(normalized)
            verification = normalized.get("verification") if isinstance(normalized.get("verification"), Mapping) else {}
            normalized_trigger_query_text = self._normalize_cortex_query_hint(normalized.get("trigger_query_text", ""))
            if self._apply_provider_outcome_calibration_fn is not None and normalized_trigger_query_text:
                confidence = max(0.0, min(1.0, float(verification.get("confidence", 0.0) or 0.0)))
                if bool(verification.get("success", False)):
                    action_outcome_score = max(0.0, min(1.0, 0.55 + 0.35 * confidence))
                elif bool(verification.get("contradiction", False)):
                    action_outcome_score = 0.0
                else:
                    action_outcome_score = max(0.0, min(1.0, 0.10 + 0.20 * confidence))
                self._apply_provider_outcome_calibration_fn(
                    query_text=normalized_trigger_query_text,
                    outcome_score=action_outcome_score,
                )
            self._record_brain_event_fn(
                {
                    "type": "digital_action_executed",
                    "timestamp": str(normalized.get("recorded_at")),
                    "action_id": str(normalized.get("action_id", "")),
                    "action_type": str(normalized.get("action_type", "")),
                    "trigger_reason": str(normalized.get("trigger_reason", "operator") or "operator"),
                    "trigger_query_text": str(normalized.get("trigger_query_text", "") or ""),
                    "verification_status": str((normalized.get("verification") or {}).get("status", "unknown")),
                    "success": bool((normalized.get("verification") or {}).get("success", False)),
                    "contradiction": bool((normalized.get("verification") or {}).get("contradiction", False)),
                }
            )
            self._runtime_state_mark_mutated_fn()
            runtime = self._brain_runtime_snapshot_fn()
        return {
            "accepted": True,
            "result": deepcopy(normalized),
            "terminus_runtime": runtime,
            "state_revision": int(self._runtime_state_mutation_summary_fn()["state_revision"]),
        }

    def normalize_action_record(self, item: Any) -> dict[str, Any] | None:
        return self._normalize_action_record(item)

    def recent_relevant_action_records(
        self,
        query_text: str,
        *,
        statuses: Sequence[str] | None = None,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        with self._lock:
            return self._recent_relevant_action_records_locked(query_text, statuses=statuses, limit=limit)

    def action_record_to_response_episodes(
        self,
        record: Mapping[str, Any],
        *,
        query_text: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        with self._lock:
            return self._action_record_to_response_episodes_locked(record, query_text=query_text, limit=limit)

    def augment_query_result_with_action_records(
        self,
        query_result: dict[str, Any],
        *,
        query_text: str,
        records: Sequence[Mapping[str, Any]],
    ) -> int:
        with self._lock:
            return self._augment_query_result_with_action_records_locked(
                query_result,
                query_text=query_text,
                records=records,
            )

    def contradicted_action_note(self, record: Mapping[str, Any]) -> str:
        return self._contradicted_action_note_locked(record)

    def should_auto_execute_action(
        self,
        *,
        query_text: str,
        query_result: dict[str, Any],
        response: Mapping[str, Any],
    ) -> bool:
        return self._should_auto_execute_action_locked(query_text=query_text, query_result=query_result, response=response)

    def maybe_auto_action_assist(
        self,
        *,
        query_text: str,
        query_result: dict[str, Any],
        response: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        with self._lock:
            return self._maybe_auto_action_assist_locked(
                query_text=query_text,
                query_result=query_result,
                response=response,
            )

    def action_history_memory_metadata(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return self._action_history_memory_metadata(record)

    def replay_action_history_into_cortex(self) -> None:
        with self._lock:
            self._replay_action_history_into_cortex_locked()

    def action_loop_summary(self) -> dict[str, Any]:
        with self._lock:
            return self._action_loop_summary_locked()

    @staticmethod
    def _normalize_action_text(value: Any) -> str:
        return " ".join(str(value).split()).strip()

    @staticmethod
    def _normalize_cortex_query_hint(value: Any) -> str:
        return " ".join(str(value).split()).strip()

    @classmethod
    def _normalize_feedback_text(cls, value: Any, *, max_chars: int = 2000) -> str:
        text = cls._normalize_action_text(value)
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + "…"
        return text

    def _runtime_feedback_corrected_present(self, feedback: Mapping[str, Any]) -> bool:
        if "corrected_output" not in feedback or feedback.get("corrected_output") is None:
            return False
        corrected_output = feedback.get("corrected_output")
        if isinstance(corrected_output, str) and not self._normalize_action_text(corrected_output):
            return False
        return True

    @classmethod
    def _runtime_feedback_applied_status(cls, verdict: str, *, corrected: bool = False) -> str:
        normalized = cls._normalize_action_text(verdict).lower()
        if corrected or normalized == "contradicted":
            return "contradicted"
        if normalized == "verified":
            return "verified"
        return "unverified"

    @classmethod
    def _runtime_feedback_provenance(cls, status: str) -> str:
        normalized = cls._normalize_action_text(status).lower()
        if normalized == "verified":
            return "verified"
        if normalized == "contradicted":
            return "contradicted"
        return "unverified"

    def _sanitize_runtime_feedback_tags(self, tags: Any) -> list[str]:
        if not isinstance(tags, Sequence) or isinstance(tags, (str, bytes)):
            return []
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in list(tags)[: DEFAULT_RUNTIME_FEEDBACK_TAG_LIMIT * 2]:
            tag = self._normalize_feedback_text(raw, max_chars=64).lower()
            if not tag or tag in seen:
                continue
            seen.add(tag)
            cleaned.append(tag)
            if len(cleaned) >= DEFAULT_RUNTIME_FEEDBACK_TAG_LIMIT:
                break
        return cleaned

    def _sanitize_runtime_feedback_evidence(self, evidence: Any) -> list[Any]:
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
            return []
        sanitized: list[Any] = []
        for raw in list(evidence)[:DEFAULT_RUNTIME_FEEDBACK_EVIDENCE_LIMIT]:
            item = self._runtime_trace_export_safe_value_fn(raw)
            if item in ({}, [], None, ""):
                continue
            sanitized.append(item)
        return sanitized

    def _normalize_runtime_feedback_request(self, feedback: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(feedback, Mapping):
            raise ValueError("Runtime feedback must be an object.")
        target_type = self._normalize_action_text(feedback.get("target_type", "")).lower()
        if target_type not in {"runtime_episode", "action"}:
            raise ValueError(f"Unsupported runtime feedback target_type: {target_type or '<empty>'}")
        target_id = self._normalize_feedback_text(feedback.get("target_id", ""), max_chars=160)
        if not target_id:
            raise ValueError("Runtime feedback target_id is required.")
        verdict = self._normalize_action_text(feedback.get("verdict", "")).lower()
        if verdict not in {"verified", "contradicted", "unverified"}:
            raise ValueError(f"Unsupported runtime feedback verdict: {verdict or '<empty>'}")
        try:
            confidence = max(0.0, min(1.0, float(feedback.get("confidence", 1.0))))
        except (TypeError, ValueError) as exc:
            raise ValueError("Runtime feedback confidence must be numeric.") from exc
        corrected = self._runtime_feedback_corrected_present(feedback)
        applied_status = self._runtime_feedback_applied_status(verdict, corrected=corrected)
        corrected_output = self._runtime_trace_export_safe_value_fn(feedback.get("corrected_output")) if corrected else None
        return {
            "feedback_id": str(uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_type": target_type,
            "target_id": target_id,
            "verdict": verdict,
            "applied_status": applied_status,
            "confidence": confidence,
            "summary": self._normalize_feedback_text(feedback.get("summary", "")),
            "corrected_output": corrected_output,
            "evidence": self._sanitize_runtime_feedback_evidence(feedback.get("evidence", [])),
            "tags": self._sanitize_runtime_feedback_tags(feedback.get("tags", [])),
            "evaluator_id": self._normalize_feedback_text(feedback.get("evaluator_id", ""), max_chars=160),
        }

    def _normalize_runtime_feedback_entries(self, feedback: Any) -> list[dict[str, Any]]:
        if not isinstance(feedback, Sequence) or isinstance(feedback, (str, bytes)):
            return []
        entries: list[dict[str, Any]] = []
        for raw in list(feedback)[-DEFAULT_RUNTIME_FEEDBACK_HISTORY:]:
            if not isinstance(raw, Mapping):
                continue
            verdict = self._normalize_action_text(raw.get("verdict", "unverified")).lower()
            if verdict not in {"verified", "contradicted", "unverified"}:
                verdict = "unverified"
            corrected = self._runtime_feedback_corrected_present(raw)
            applied_status = self._normalize_action_text(raw.get("applied_status", "")).lower()
            if applied_status not in {"verified", "contradicted", "unverified"}:
                applied_status = self._runtime_feedback_applied_status(verdict, corrected=corrected)
            try:
                confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            corrected_output = self._runtime_trace_export_safe_value_fn(raw.get("corrected_output")) if corrected else None
            entries.append(
                {
                    "feedback_id": self._normalize_feedback_text(raw.get("feedback_id", ""), max_chars=80) or str(uuid4()),
                    "created_at": self._normalize_feedback_text(raw.get("created_at", ""), max_chars=80)
                    or datetime.now(timezone.utc).isoformat(),
                    "target_type": self._normalize_feedback_text(raw.get("target_type", ""), max_chars=32),
                    "target_id": self._normalize_feedback_text(raw.get("target_id", ""), max_chars=160),
                    "verdict": verdict,
                    "applied_status": applied_status,
                    "confidence": confidence,
                    "summary": self._normalize_feedback_text(raw.get("summary", "")),
                    "corrected_output": corrected_output,
                    "evidence": self._sanitize_runtime_feedback_evidence(raw.get("evidence", [])),
                    "tags": self._sanitize_runtime_feedback_tags(raw.get("tags", [])),
                    "evaluator_id": self._normalize_feedback_text(raw.get("evaluator_id", ""), max_chars=160),
                }
            )
        return entries

    @classmethod
    def _action_request_has_body(cls, inputs: Mapping[str, Any]) -> bool:
        if not isinstance(inputs, Mapping):
            return False
        if "json_body" not in inputs:
            return False
        body = inputs.get("json_body")
        if body is None:
            return False
        if isinstance(body, str):
            return bool(cls._normalize_action_text(body))
        if isinstance(body, Mapping):
            return bool(dict(body))
        if isinstance(body, Sequence) and not isinstance(body, (str, bytes, bytearray)):
            return bool(list(body))
        return True

    @classmethod
    def _api_request_record_matches_explicit_url(cls, record: Mapping[str, Any], explicit_url: str) -> bool:
        if str(record.get("action_type", "")) != "api_request":
            return False
        inputs = record.get("inputs") if isinstance(record.get("inputs"), Mapping) else {}
        if cls._normalize_action_text(inputs.get("url", "")) != explicit_url:
            return False
        method = cls._normalize_action_text(inputs.get("method", "GET")).upper() or "GET"
        if method != "GET":
            return False
        return not cls._action_request_has_body(inputs)

    @classmethod
    def _action_query_terms(cls, query_text: str) -> tuple[str, ...]:
        normalized = cls._normalize_action_text(query_text).lower()
        if not normalized:
            return ()
        terms = [term.lower() for term in salient_query_terms(normalized) if term]
        if not terms:
            terms = [
                token.lower()
                for token in re.findall(r"[a-zA-Z0-9_./:-]+", normalized)
                if len(token) >= 2
            ]
        deduped: list[str] = []
        seen: set[str] = set()
        for term in terms:
            compact = cls._normalize_action_text(term).lower()
            if not compact or compact in seen:
                continue
            deduped.append(compact)
            seen.add(compact)
        return tuple(deduped[:8])

    @classmethod
    def _action_focus_query_text(cls, query_text: str) -> str:
        normalized = cls._normalize_action_text(query_text)
        if not normalized:
            return ""
        stripped = re.sub(r"https?://[^\s'\")\]>]+", " ", normalized, flags=re.IGNORECASE)
        stripped = re.sub(
            r"(?:[A-Za-z0-9_.-]+[\\/])*[A-Za-z0-9_.-]+\.(?:py|md|txt|json|yaml|yml|toml|csv|ts|tsx|js|jsx|html|css|scss|ini|cfg|log|rst)",
            " ",
            stripped,
            flags=re.IGNORECASE,
        )
        focused_terms = cls._action_query_terms(stripped)
        if focused_terms:
            return " ".join(focused_terms[:6])
        fallback_terms = cls._action_query_terms(normalized)
        if fallback_terms:
            return " ".join(fallback_terms[:6])
        return normalized

    def _query_workspace_path_candidate_locked(self, query_text: str) -> str:
        normalized = self._normalize_action_text(query_text)
        if not normalized:
            return ""
        candidates = re.findall(
            r"(?:[A-Za-z0-9_.-]+[\\/])*[A-Za-z0-9_.-]+\.(?:py|md|txt|json|yaml|yml|toml|csv|ts|tsx|js|jsx|html|css|scss|ini|cfg|log|rst)",
            normalized,
            flags=re.IGNORECASE,
        )
        for raw in candidates:
            cleaned = raw.strip("`'\".,;:!?()[]{} ").replace("\\", "/")
            if not cleaned:
                continue
            candidate = Path(cleaned)
            resolved = candidate if candidate.is_absolute() else (self._action_root / candidate)
            try:
                resolved = resolved.resolve()
            except Exception:
                continue
            if resolved != self._action_root and self._action_root not in resolved.parents:
                continue
            if not resolved.exists() or not resolved.is_file():
                continue
            try:
                return str(resolved.relative_to(self._action_root)).replace("\\", "/")
            except Exception:
                return str(resolved)
        return ""

    @classmethod
    def _query_web_url_candidate(cls, query_text: str) -> str:
        normalized = cls._normalize_action_text(query_text)
        if not normalized:
            return ""
        matches = re.findall(r"https?://[^\s'\")\]>]+", normalized, flags=re.IGNORECASE)
        for raw in matches:
            cleaned = raw.strip("`'\".,;:!?()[]{} ")
            if cleaned:
                return cleaned
        return ""

    @classmethod
    def _query_api_url_candidate(cls, query_text: str) -> str:
        candidate = cls._query_web_url_candidate(query_text)
        if not candidate:
            return ""
        lowered = cls._normalize_action_text(query_text).lower()
        parsed = urlparse(candidate)
        path = (parsed.path or "").lower()
        if path.endswith(".json") or "/api/" in path or any(token in lowered for token in (" api ", " json ", " endpoint ")):
            return candidate
        return ""

    def _action_record_relevance_score_locked(self, record: Mapping[str, Any], query_text: str) -> float:
        normalized_query = self._normalize_action_text(query_text).lower()
        if not normalized_query:
            return 0.0
        explicit_api_url = self._query_api_url_candidate(query_text).lower()
        explicit_url = self._query_web_url_candidate(query_text).lower()
        record_url = self._normalize_action_text((record.get("inputs") or {}).get("url", "")).lower()
        if explicit_api_url and explicit_api_url == record_url:
            if str(record.get("action_type", "")) != "api_request":
                return 0.0
            if self._api_request_record_matches_explicit_url(record, explicit_api_url):
                return 1.0
        if explicit_url and explicit_url == record_url:
            return 1.0
        trigger_query = self._normalize_action_text(record.get("trigger_query_text", "")).lower()
        record_query = self._normalize_action_text((record.get("inputs") or {}).get("query_text", "")).lower()
        if normalized_query and normalized_query in {trigger_query, record_query}:
            return 1.0
        query_terms = set(self._action_query_terms(normalized_query))
        if not query_terms:
            return 0.0
        record_terms: set[str] = set(
            self._normalize_action_text(term).lower()
            for term in list(record.get("topics") or [])
            if self._normalize_action_text(term)
        )
        record_terms.update(self._action_query_terms(record_query))
        record_terms.update(self._action_query_terms(str((record.get("inputs") or {}).get("path", ""))))
        record_terms.update(self._action_query_terms(str((record.get("inputs") or {}).get("url", ""))))
        verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
        for raw_item in list(verification.get("evidence") or []):
            if not isinstance(raw_item, Mapping):
                continue
            record_terms.update(
                self._normalize_action_text(term).lower()
                for term in list(raw_item.get("matched_terms") or [])
                if self._normalize_action_text(term)
            )
            record_terms.update(self._action_query_terms(str(raw_item.get("snippet", ""))))
        if not record_terms:
            record_terms.update(self._action_query_terms(str(record.get("actual_outcome", ""))))
        overlap = len(query_terms & record_terms)
        if overlap <= 0:
            return 0.0
        return float(overlap) / float(max(1, len(query_terms)))

    def _filter_records_for_explicit_target(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        explicit_api_url: str,
        explicit_url: str,
        explicit_path: str,
    ) -> list[dict[str, Any]]:
        filtered_records = [deepcopy(dict(record)) for record in records]
        if explicit_api_url:
            return [
                record
                for record in filtered_records
                if self._api_request_record_matches_explicit_url(record, explicit_api_url)
            ]
        if explicit_url:
            return [
                record
                for record in filtered_records
                if str(record.get("action_type", "")) == "web_fetch"
                and self._normalize_action_text((record.get("inputs") or {}).get("url", "")) == explicit_url
            ]
        if explicit_path:
            return [
                record
                for record in filtered_records
                if str(record.get("action_type", "")) == "workspace_read"
                and self._normalize_action_text((record.get("inputs") or {}).get("path", "")) == explicit_path
            ]
        return filtered_records

    def _recent_relevant_action_records_locked(
        self,
        query_text: str,
        *,
        statuses: Sequence[str] | None = None,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        allowed = {
            self._normalize_action_text(status).lower()
            for status in list(statuses or [])
            if self._normalize_action_text(status)
        }
        ranked: list[tuple[float, dict[str, Any]]] = []
        for record in self._action_history:
            verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
            status = self._normalize_action_text(verification.get("status", "")).lower()
            if allowed and status not in allowed:
                continue
            score = self._action_record_relevance_score_locked(record, query_text)
            if score < 0.34:
                continue
            ranked.append((score, deepcopy(record)))
        ranked.sort(
            key=lambda item: (
                float(item[0]),
                str(item[1].get("recorded_at", "")),
            ),
            reverse=True,
        )
        return [record for _, record in ranked[: max(1, int(limit))]]

    def _action_record_to_response_episodes_locked(
        self,
        record: Mapping[str, Any],
        *,
        query_text: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
        if not bool(verification.get("success", False)):
            return []
        query_terms = list(self._action_query_terms(query_text))
        evidence_items = [
            dict(raw)
            for raw in list(verification.get("evidence") or [])
            if isinstance(raw, Mapping)
        ]
        action_seed = int(hashlib.sha256(str(record.get("action_id", "")).encode("utf-8")).hexdigest()[:8], 16)
        episodes: list[dict[str, Any]] = []
        for idx, evidence in enumerate(evidence_items[: max(1, int(limit))]):
            snippet = self._normalize_action_text(evidence.get("snippet", ""))
            if not snippet:
                continue
            matching = tuple(match_terms(query_terms, snippet))
            overlap_ratio = float(len(matching)) / float(max(1, len(query_terms)))
            exact_query = bool(evidence.get("exact_query", False))
            similarity = max(0.46, 0.56 + 0.34 * overlap_ratio + (0.10 if exact_query else 0.0))
            memory_index = -1 * int(action_seed + idx + 1)
            episodes.append(
                {
                    "text": snippet,
                    "raw_window": snippet,
                    "memory_index": memory_index,
                    "memory_indices": [memory_index],
                    "similarity": float(min(0.99, similarity)),
                    "importance": float(verification.get("confidence", 0.0) or 0.0),
                    "age_tokens": 0,
                    "match_count": 1,
                    "query_overlap": int(len(matching)),
                    "focus_overlap": 0,
                    "memory_focus_priority": 0.0,
                    "complete_sentence": int(snippet.endswith((".", "!", "?"))),
                    "clipped_overlap": 0,
                    "expansion_chars": 0,
                    "action_origin": str(record.get("action_id", "")),
                    "action_type": str(record.get("action_type", "")),
                    "source_path": self._normalize_action_text(evidence.get("path", "")),
                    "line_number": int(evidence.get("line_number", 0) or 0),
                }
            )
        if episodes:
            return episodes
        summary = self._normalize_action_text(record.get("actual_outcome", ""))
        if not summary:
            return []
        matching = tuple(match_terms(query_terms, summary))
        overlap_ratio = float(len(matching)) / float(max(1, len(query_terms)))
        memory_index = -1 * int(action_seed + 999)
        return [
            {
                "text": summary,
                "raw_window": summary,
                "memory_index": memory_index,
                "memory_indices": [memory_index],
                "similarity": float(min(0.95, 0.48 + 0.32 * overlap_ratio)),
                "importance": float(verification.get("confidence", 0.0) or 0.0),
                "age_tokens": 0,
                "match_count": 1,
                "query_overlap": int(len(matching)),
                "focus_overlap": 0,
                "memory_focus_priority": 0.0,
                "complete_sentence": int(summary.endswith((".", "!", "?"))),
                "clipped_overlap": 0,
                "expansion_chars": 0,
                "action_origin": str(record.get("action_id", "")),
                "action_type": str(record.get("action_type", "")),
            }
        ]

    def _augment_query_result_with_action_records_locked(
        self,
        query_result: dict[str, Any],
        *,
        query_text: str,
        records: Sequence[Mapping[str, Any]],
    ) -> int:
        query_summary = query_result.get("query_summary")
        if not isinstance(query_summary, dict):
            return 0
        injected: list[dict[str, Any]] = []
        seen_texts: set[str] = set()
        for record in records:
            for episode in self._action_record_to_response_episodes_locked(record, query_text=query_text):
                text_key = self._normalize_action_text(episode.get("text", "")).lower()
                if not text_key or text_key in seen_texts:
                    continue
                injected.append(episode)
                seen_texts.add(text_key)
        existing = [
            deepcopy(item)
            for item in list(query_summary.get("memory_episodes") or [])
            if isinstance(item, Mapping)
        ]
        for item in existing:
            text_key = self._normalize_action_text(item.get("text", item.get("raw_window", ""))).lower()
            if text_key:
                seen_texts.add(text_key)
        if injected:
            injected_texts = {
                self._normalize_action_text(injected_item.get("text", "")).lower()
                for injected_item in injected
            }
            query_summary["memory_episodes"] = injected + [
                item
                for item in existing
                if self._normalize_action_text(item.get("text", item.get("raw_window", ""))).lower()
                not in injected_texts
            ]
        return int(len(injected))

    def _contradicted_action_note_locked(self, record: Mapping[str, Any]) -> str:
        actual = self._normalize_action_text(record.get("actual_outcome", ""))
        if actual:
            return f" I checked the workspace and observed: {actual}"
        return " I checked the workspace and found no additional grounded evidence there."

    def _reuse_recent_action_assist(
        self,
        record: Mapping[str, Any],
        *,
        result_count: int,
        response_episode_count: int,
        used_in_response: bool,
    ) -> dict[str, Any]:
        assist = {
            "triggered": True,
            "executed": False,
            "reused_recent_action": True,
            "used_in_response": used_in_response,
            "result": deepcopy(dict(record)),
            "result_count": int(result_count),
            "response_episode_count": int(response_episode_count),
        }
        verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
        if bool(verification.get("contradiction", False)):
            assist["reason"] = "recent_contradicted_action"
            assist["response_note"] = self._contradicted_action_note_locked(record)
            return assist
        assist["reason"] = "recent_verified_action"
        return assist

    def _should_auto_execute_action_locked(
        self,
        *,
        query_text: str,
        query_result: dict[str, Any],
        response: Mapping[str, Any],
    ) -> bool:
        if not self._normalize_action_text(query_text):
            return False
        gap_plan = query_result.get("gap_plan") if isinstance(query_result.get("gap_plan"), Mapping) else {}
        meaningful_gap = bool(
            gap_plan.get("unsupported_terms")
            or gap_plan.get("gap_terms")
            or gap_plan.get("weak_concepts")
            or float(gap_plan.get("grounded_fraction", 0.0) or 0.0) < 0.999
        )
        if not meaningful_gap:
            return False
        response_mode = self._normalize_action_text(response.get("response_mode", "")).lower()
        if response_mode == "insufficient_evidence":
            return True
        unsupported_terms = list(response.get("unsupported_terms") or gap_plan.get("unsupported_terms") or [])
        evidence_coverage = float(response.get("evidence_coverage", 0.0) or 0.0)
        return bool(unsupported_terms) and evidence_coverage < 0.85

    def _maybe_auto_action_assist_locked(
        self,
        *,
        query_text: str,
        query_result: dict[str, Any],
        response: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        explicit_api_url = self._query_api_url_candidate(query_text)
        explicit_url = self._query_web_url_candidate(query_text) if not explicit_api_url else ""
        explicit_path = self._query_workspace_path_candidate_locked(query_text) if not (explicit_api_url or explicit_url) else ""
        verified_records = self._filter_records_for_explicit_target(
            self._recent_relevant_action_records_locked(query_text, statuses=("verified",), limit=2),
            explicit_api_url=explicit_api_url,
            explicit_url=explicit_url,
            explicit_path=explicit_path,
        )
        if verified_records:
            injected = self._augment_query_result_with_action_records_locked(
                query_result,
                query_text=query_text,
                records=verified_records,
            )
            return self._reuse_recent_action_assist(
                verified_records[0],
                result_count=len(verified_records),
                response_episode_count=injected,
                used_in_response=bool(injected > 0),
            )

        contradicted_records = self._filter_records_for_explicit_target(
            self._recent_relevant_action_records_locked(query_text, statuses=("contradicted",), limit=1),
            explicit_api_url=explicit_api_url,
            explicit_url=explicit_url,
            explicit_path=explicit_path,
        )
        if not self._should_auto_execute_action_locked(query_text=query_text, query_result=query_result, response=response):
            response_mode = self._normalize_action_text(response.get("response_mode", "")).lower()
            unsupported_terms = list(response.get("unsupported_terms") or [])
            if contradicted_records and (response_mode == "insufficient_evidence" or unsupported_terms):
                return self._reuse_recent_action_assist(
                    contradicted_records[0],
                    result_count=1,
                    response_episode_count=0,
                    used_in_response=False,
                )
            return None

        if contradicted_records:
            return self._reuse_recent_action_assist(
                contradicted_records[0],
                result_count=1,
                response_episode_count=0,
                used_in_response=False,
            )

        gap_plan = query_result.get("gap_plan") if isinstance(query_result.get("gap_plan"), Mapping) else {}
        retrieval_queries = [
            self._normalize_action_text(value)
            for value in list(gap_plan.get("retrieval_queries") or [])
            if self._normalize_action_text(value)
        ]
        search_query = next((value for value in retrieval_queries if value), query_text)
        action_type = "workspace_search"
        explicit_api_url = explicit_api_url or self._query_api_url_candidate(search_query)
        if explicit_api_url:
            action_type = "api_request"
        elif explicit_url:
            action_type = "web_fetch"
        elif explicit_path:
            action_type = "workspace_read"

        if action_type == "api_request":
            action_result = self.execute_digital_action(
                {
                    "action_type": "api_request",
                    "url": explicit_api_url,
                    "query_text": self._action_focus_query_text(query_text),
                    "predicted_outcome": (
                        f"Auto action assist expects requesting structured JSON from {explicit_api_url} "
                        f"to reveal grounded evidence relevant to: {query_text}."
                    ),
                },
                trigger_reason="query_gap_auto_api_request",
                trigger_query_text=query_text,
            )
            assist_reason = "query_gap_auto_api_request"
        elif action_type == "web_fetch":
            action_result = self.execute_digital_action(
                {
                    "action_type": "web_fetch",
                    "url": explicit_url,
                    "query_text": self._action_focus_query_text(query_text),
                    "predicted_outcome": (
                        f"Auto action assist expects fetching {explicit_url} to reveal grounded evidence relevant to: {query_text}."
                    ),
                },
                trigger_reason="query_gap_auto_fetch",
                trigger_query_text=query_text,
            )
            assist_reason = "query_gap_auto_fetch"
        elif action_type == "workspace_read":
            action_result = self.execute_digital_action(
                {
                    "action_type": "workspace_read",
                    "path": explicit_path,
                    "query_text": self._action_focus_query_text(query_text),
                    "predicted_outcome": (
                        f"Auto action assist expects reading {explicit_path} to reveal grounded workspace evidence relevant to: {query_text}."
                    ),
                },
                trigger_reason="query_gap_auto_read",
                trigger_query_text=query_text,
            )
            assist_reason = "query_gap_auto_read"
        else:
            action_result = self.execute_digital_action(
                {
                    "action_type": "workspace_search",
                    "query_text": search_query,
                    "predicted_outcome": (
                        f"Auto action assist expects grounded workspace evidence relevant to: {query_text}."
                    ),
                },
                trigger_reason="query_gap_auto_search",
                trigger_query_text=query_text,
            )
            assist_reason = "query_gap_auto_search"

        if not bool(action_result.get("accepted", False)):
            return {
                "triggered": True,
                "executed": False,
                "reused_recent_action": False,
                "reason": "auto_action_execution_failed",
                "used_in_response": False,
                "error": self._normalize_action_text(action_result.get("reason", "execution_failed")),
            }
        record = cast(dict[str, Any], action_result.get("result") or {})
        verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
        injected = 0
        if bool(verification.get("success", False)):
            injected = self._augment_query_result_with_action_records_locked(
                query_result,
                query_text=query_text,
                records=[record],
            )
        assist = {
            "triggered": True,
            "executed": True,
            "reused_recent_action": False,
            "reason": assist_reason,
            "used_in_response": bool(injected > 0),
            "result": deepcopy(record),
            "result_count": 1,
            "response_episode_count": int(injected),
        }
        if bool(verification.get("contradiction", False)):
            assist["response_note"] = self._contradicted_action_note_locked(record)
        return assist

    def _action_history_memory_metadata(self, record: Mapping[str, Any]) -> dict[str, Any]:
        verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
        success = bool(verification.get("success", False))
        contradiction = bool(verification.get("contradiction", False))
        evidence = [deepcopy(dict(raw)) for raw in list(verification.get("evidence") or []) if isinstance(raw, Mapping)]
        return {
            "observation_kind": "action",
            "grounded": True,
            "grounding_signal": 0.92 if success else 0.72,
            "evidence_unit_count": max(1, len(evidence)),
            "source_name": "workspace",
            "source_type": "action",
            "action_id": str(record.get("action_id", "")),
            "action_type": str(record.get("action_type", "")),
            "action_inputs": deepcopy(dict(record.get("inputs") or {})),
            "predicted_outcome": str(record.get("predicted_outcome", "")),
            "actual_outcome": str(record.get("actual_outcome", "")),
            "verification_status": str(verification.get("status", "unknown")),
            "verification_confidence": float(verification.get("confidence", 0.0) or 0.0),
            "contradiction": bool(contradiction),
            "evidence": evidence,
        }

    def _inject_action_record_into_cortex_locked(self, record: Mapping[str, Any]) -> None:
        thought_loop = self._ensure_cortex_initialized_fn()
        if thought_loop is None:
            return
        self._inject_action_record_into_cortex_fn(thought_loop, record)

    def _replay_action_history_into_cortex_locked(self) -> None:
        for record in reversed(list(self._action_history)):
            self._inject_action_record_into_cortex_locked(record)

    def _action_loop_summary_locked(self) -> dict[str, Any]:
        verified = 0
        contradicted = 0
        for record in self._action_history:
            verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
            if bool(verification.get("success", False)):
                verified += 1
            if bool(verification.get("contradiction", False)):
                contradicted += 1
        last_action = None if not self._action_history else deepcopy(self._action_history[0])
        return {
            "enabled": True,
            "root_path": str(self._action_root),
            "supported_actions": list(SUPPORTED_ACTION_TYPES),
            "actions_recorded": int(len(self._action_history)),
            "verified_actions": int(verified),
            "contradicted_actions": int(contradicted),
            "last_action": last_action,
        }
