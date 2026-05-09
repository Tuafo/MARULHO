from __future__ import annotations

from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, cast

from hecsn.service.action_loop import execute_digital_action

DEFAULT_CORTEX_ACTION_INIT_TIMEOUT_SECONDS = 0.25


class ActionRuntimeMixin:
    """Digital action execution, audit history, and action-loop summaries."""

    def action_history(self, limit: int = 20) -> dict[str, Any]:
        with self._lock:
            count = max(1, int(limit))
            history = [deepcopy(item) for item in list(self._action_history)[:count]]
            return {
                "count": int(len(self._action_history)),
                "root_path": str(self._action_root),
                "supported_actions": ["workspace_search", "workspace_read", "web_fetch", "api_request"],
                "actions": history,
            }

    def action_record(self, action_id: str) -> dict[str, Any] | None:
        target_id = str(action_id)
        if not target_id:
            return None
        with self._lock:
            for item in list(self._action_history):
                if str(item.get("action_id", "")) == target_id:
                    return deepcopy(item)
        return None

    def replace_action_record(self, action_id: str, record: Mapping[str, Any]) -> dict[str, Any] | None:
        target_id = str(action_id)
        if not target_id:
            return None
        replacement = deepcopy(dict(record))
        if str(replacement.get("action_id", "")) != target_id:
            return None
        with self._lock:
            existing = list(self._action_history)
            updated: list[dict[str, Any]] = []
            replaced = False
            for item in existing:
                if not replaced and str(item.get("action_id", "")) == target_id:
                    updated.append(deepcopy(replacement))
                    replaced = True
                else:
                    updated.append(item)
            if not replaced:
                return None
            self._action_history = deque(updated, maxlen=self._action_history.maxlen)
            return deepcopy(replacement)

    def execute_digital_action(
        self,
        action: Mapping[str, Any],
        *,
        trigger_reason: str | None = None,
        trigger_query_text: str | None = None,
    ) -> dict[str, Any]:
        action_type = " ".join(str(action.get("action_type", action.get("type", ""))).split()).strip().lower()
        if action_type not in {"workspace_search", "workspace_read", "web_fetch", "api_request"}:
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
        normalized = self._normalize_action_record(payload)
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
            autonomy = cast(dict[str, Any] | None, self._brain_config.get("autonomy"))
            if autonomy and normalized_trigger_query_text:
                confidence = max(0.0, min(1.0, float(verification.get("confidence", 0.0) or 0.0)))
                if bool(verification.get("success", False)):
                    action_outcome_score = max(0.0, min(1.0, 0.55 + 0.35 * confidence))
                elif bool(verification.get("contradiction", False)):
                    action_outcome_score = 0.0
                else:
                    action_outcome_score = max(0.0, min(1.0, 0.10 + 0.20 * confidence))
                self._apply_provider_outcome_calibration_locked(
                    autonomy=autonomy,
                    query_text=normalized_trigger_query_text,
                    outcome_score=action_outcome_score,
                )
            self._record_brain_event_locked(
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
            self._runtime_state.mark_mutated()
            runtime = self._brain_runtime_snapshot_locked()
        return {
            "accepted": True,
            "result": deepcopy(normalized),
            "terminus_runtime": runtime,
            "state_revision": int(self._runtime_state.state_revision),
        }

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
        thought_loop = self._thought_loop_actual or self._ensure_cortex_initialized(
            wait_seconds=DEFAULT_CORTEX_ACTION_INIT_TIMEOUT_SECONDS
        )
        if thought_loop is None:
            return
        self._inject_action_record_into_loop(thought_loop, record)

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
            "supported_actions": ["workspace_search", "workspace_read", "web_fetch", "api_request"],
            "actions_recorded": int(len(self._action_history)),
            "verified_actions": int(verified),
            "contradicted_actions": int(contradicted),
            "last_action": last_action,
        }

