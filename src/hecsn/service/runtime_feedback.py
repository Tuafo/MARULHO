from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from uuid import uuid4

DEFAULT_RUNTIME_FEEDBACK_HISTORY = 8
DEFAULT_RUNTIME_FEEDBACK_EVIDENCE_LIMIT = 8
DEFAULT_RUNTIME_FEEDBACK_TAG_LIMIT = 12
DEFAULT_RUNTIME_FEEDBACK_MAX_TEXT_CHARS = 2000


class RuntimeFeedbackMixin:
    """Operator feedback normalization and application helpers."""

    def record_runtime_feedback(self, feedback: Mapping[str, Any]) -> dict[str, Any]:
        entry = self._normalize_runtime_feedback_request(feedback)
        target_type = str(entry["target_type"])
        target_id = str(entry["target_id"])
        with self._lock:
            updated_target: dict[str, Any] | None = None
            if target_type == "runtime_episode":
                episodes = list(self._runtime_episode_traces)
                for index, episode in enumerate(episodes):
                    if str(episode.get("episode_id", "")) == target_id:
                        updated = deepcopy(episode)
                        self._apply_runtime_feedback_to_target(updated, entry)
                        episodes[index] = updated
                        self._runtime_episode_traces = deque(episodes, maxlen=self._runtime_episode_traces.maxlen)
                        updated_target = deepcopy(updated)
                        break
            elif target_type == "action":
                actions = list(self._action_history)
                for index, action in enumerate(actions):
                    if str(action.get("action_id", "")) == target_id:
                        updated = deepcopy(action)
                        self._apply_runtime_feedback_to_target(updated, entry)
                        actions[index] = updated
                        self._action_history = deque(actions, maxlen=self._action_history.maxlen)
                        updated_target = deepcopy(updated)
                        break
            else:
                raise ValueError(f"Unsupported runtime feedback target_type: {target_type}")

            if updated_target is None:
                raise ValueError(f"Runtime feedback target not found: {target_type} {target_id}")

            self._record_brain_event_locked(
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
            self._mark_mutated()
            return {
                "accepted": True,
                "target_type": target_type,
                "target_id": target_id,
                "feedback": deepcopy(entry),
                "target": updated_target,
                "dirty_state": bool(self._dirty_state),
                "state_revision": int(self._state_revision),
                "terminus_runtime": self._brain_runtime_snapshot_locked(),
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
            item = self._runtime_trace_export_safe_value(raw)
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
        corrected_output = self._runtime_trace_export_safe_value(feedback.get("corrected_output")) if corrected else None
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
            corrected_output = self._runtime_trace_export_safe_value(raw.get("corrected_output")) if corrected else None
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

