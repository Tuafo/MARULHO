from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence, cast
from uuid import uuid4

DEFAULT_RUNTIME_FEEDBACK_HISTORY = 8
DEFAULT_RUNTIME_FEEDBACK_EVIDENCE_LIMIT = 8
DEFAULT_RUNTIME_FEEDBACK_TAG_LIMIT = 12
DEFAULT_RUNTIME_FEEDBACK_MAX_TEXT_CHARS = 2000


class _RuntimeEpisodeFeedbackStore(Protocol):
    def runtime_episode_trace(self, episode_id: str) -> dict[str, Any] | None: ...

    def replace_runtime_episode_trace(
        self,
        episode_id: str,
        episode: Mapping[str, Any],
    ) -> dict[str, Any] | None: ...


class _ActionFeedbackStore(Protocol):
    def action_record(self, action_id: str) -> dict[str, Any] | None: ...

    def replace_action_record(self, action_id: str, record: Mapping[str, Any]) -> dict[str, Any] | None: ...


class FeedbackApplier:
    """Operator feedback normalization and application helpers."""

    def __init__(
        self,
        *,
        lock: Any,
        runtime_episode_store: _RuntimeEpisodeFeedbackStore,
        action_store: _ActionFeedbackStore,
        runtime_state_mark_mutated_fn: Callable[[], None],
        runtime_state_mutation_summary_fn: Callable[[], dict[str, Any]],
        record_brain_event_fn: Callable[[Mapping[str, Any]], None],
        brain_runtime_snapshot_fn: Callable[[], dict[str, Any]],
        runtime_trace_export_safe_value_fn: Callable[[Any], Any],
    ) -> None:
        self._lock = lock
        self._runtime_episode_store = runtime_episode_store
        self._action_store = action_store
        self._runtime_state_mark_mutated_fn = runtime_state_mark_mutated_fn
        self._runtime_state_mutation_summary_fn = runtime_state_mutation_summary_fn
        self._record_brain_event_fn = record_brain_event_fn
        self._brain_runtime_snapshot_fn = brain_runtime_snapshot_fn
        self._runtime_trace_export_safe_value_fn = runtime_trace_export_safe_value_fn

    def runtime_episode_feedback_store(self) -> _RuntimeEpisodeFeedbackStore:
        return self._runtime_episode_store

    def action_feedback_store(self) -> _ActionFeedbackStore:
        return self._action_store

    def record_runtime_feedback(self, feedback: Mapping[str, Any]) -> dict[str, Any]:
        entry = self._normalize_runtime_feedback_request(feedback)
        target_type = str(entry["target_type"])
        target_id = str(entry["target_id"])
        with self._lock:
            updated_target: dict[str, Any] | None = None
            if target_type == "runtime_episode":
                episode = self._runtime_episode_store.runtime_episode_trace(target_id)
                if episode is not None:
                    updated = deepcopy(episode)
                    self._apply_runtime_feedback_to_target(updated, entry)
                    updated_target = self._runtime_episode_store.replace_runtime_episode_trace(target_id, updated)
            elif target_type == "action":
                action = self._action_store.action_record(target_id)
                if action is not None:
                    updated = deepcopy(action)
                    self._apply_runtime_feedback_to_target(updated, entry)
                    updated_target = self._action_store.replace_action_record(target_id, updated)
            else:
                raise ValueError(f"Unsupported runtime feedback target_type: {target_type}")

            if updated_target is None:
                raise ValueError(f"Runtime feedback target not found: {target_type} {target_id}")

            self._record_brain_event_fn(
                {
                    "type": "runtime_feedback_recorded",
                    "timestamp": str(entry.get("created_at", "")),
                    "target_type": target_type,
                    "target_id": target_id,
                    "verdict": str(entry.get("verdict", "")),
                    "applied_status": str(entry.get("applied_status", "")),
                    "evaluator_id": str(entry.get("evaluator_id", "")),
                }
            )
            self._runtime_state_mark_mutated_fn()
            return {
                "accepted": True,
                "target_type": target_type,
                "target_id": target_id,
                "feedback": deepcopy(entry),
                "target": updated_target,
                **self._runtime_state_mutation_summary_fn(),
                "terminus_runtime": self._brain_runtime_snapshot_fn(),
            }

    @staticmethod
    def _normalize_action_text(value: Any) -> str:
        return " ".join(str(value).split()).strip()

    @classmethod
    def _normalize_feedback_text(cls, value: Any, *, max_chars: int = DEFAULT_RUNTIME_FEEDBACK_MAX_TEXT_CHARS) -> str:
        text = cls._normalize_action_text(value)
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + "…"
        return text

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

    def _runtime_feedback_corrected_present(self, feedback: Mapping[str, Any]) -> bool:
        if "corrected_output" not in feedback or feedback.get("corrected_output") is None:
            return False
        corrected_output = feedback.get("corrected_output")
        if isinstance(corrected_output, str) and not self._normalize_action_text(corrected_output):
            return False
        return True

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

    def _apply_runtime_feedback_to_target(self, target: dict[str, Any], feedback: Mapping[str, Any]) -> None:
        feedback_entries = self._normalize_runtime_feedback_entries(target.get("feedback", []))
        feedback_entries.append(deepcopy(dict(feedback)))
        feedback_entries = feedback_entries[-DEFAULT_RUNTIME_FEEDBACK_HISTORY:]
        target["feedback"] = feedback_entries

        status = self._runtime_feedback_applied_status(
            str(feedback.get("verdict", "")),
            corrected=feedback.get("corrected_output") is not None,
        )
        provenance = self._runtime_feedback_provenance(status)
        summary = self._normalize_feedback_text(feedback.get("summary", ""))
        if not summary:
            summary = f"Runtime feedback marked target {status}."
        verification = dict(target.get("verification") or {}) if isinstance(target.get("verification"), Mapping) else {}
        verification.update(
            {
                "status": status,
                "success": status == "verified",
                "confidence": max(0.0, min(1.0, float(feedback.get("confidence", 0.0) or 0.0))),
                "contradiction": status == "contradicted",
                "summary": summary,
                "provenance": provenance,
                "last_feedback_id": str(feedback.get("feedback_id", "")),
                "last_feedback_at": str(feedback.get("created_at", "")),
                "feedback_count": int(len(feedback_entries)),
            }
        )
        target["verification"] = verification
        target["feedback_status"] = status
        target["feedback_provenance"] = provenance
        target["last_feedback_at"] = str(feedback.get("created_at", ""))
        if status in {"verified", "contradicted"} or "action_id" in target:
            target["provenance"] = provenance
        if feedback.get("corrected_output") is not None:
            target["corrected_output"] = deepcopy(feedback.get("corrected_output"))

