from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from datetime import datetime, timezone
from itertools import islice
import time
from typing import Any, Mapping, Sequence, cast
from uuid import uuid4

from marulho.service.history_store import read_history_record, replace_history_record
from marulho.service.interaction_pipeline import (
    build_feed_runtime_actual_output,
    build_query_runtime_actual_output,
    build_query_runtime_verification,
    build_respond_runtime_actual_output,
    build_respond_runtime_verification,
)
from marulho.service.living_loop_records import RuntimeEpisodeTrace
from marulho.service.living_status import LivingStatusCore

DEFAULT_RUNTIME_TRACE_EXPORT_LIMIT = 20
MAX_RUNTIME_TRACE_EXPORT_LIMIT = 50
RUNTIME_TRACE_STATUS_SOURCE_WINDOW_LIMIT = 12
RUNTIME_TRACE_FEEDBACK_SOURCE_WINDOW_LIMIT = 64
RUNTIME_TRACE_EXPORT_SCHEMA_VERSION = 1
DEFAULT_RUNTIME_FEEDBACK_HISTORY = 8
DEFAULT_RUNTIME_FEEDBACK_TAG_LIMIT = 12
_RUNTIME_TRACE_EXPORT_MAX_STRING_CHARS = 2000
_RUNTIME_TRACE_EXPORT_MAX_LIST_ITEMS = 16
_RUNTIME_TRACE_EXPORT_MAX_MAPPING_ITEMS = 48
_RUNTIME_TRACE_EXPORT_ALLOWED_TOKEN_KEYS = {
    "runs_every_token",
    "token_count",
    "token_count_mutated",
    "tokens_processed",
    "top_k_candidates",
    "top_k_memories",
}
_RUNTIME_TRACE_EXPORT_UNSAFE_KEY_MARKERS = (
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "dotenv",
    "environment",
    "password",
    "secret",
)
_RUNTIME_TRACE_EXPORT_UNSAFE_KEYS = {
    "checkpoint_path",
    "env",
    "env_root",
    "path",
    "raw_environment",
    "root_path",
    "runtime_env",
    "trace_path",
    "workspace_root",
}


class RuntimeEvidenceReporter:
    """Runtime trace and feedback summary helpers.

    This is the evidence lane: it creates sanitized trace/export artifacts but
    does not train adapters, mutate memory, execute actions, or promote facts.
    """

    def export_runtime_trace_examples(
        self,
        *,
        limit: int = DEFAULT_RUNTIME_TRACE_EXPORT_LIMIT,
        endpoint: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            count = min(MAX_RUNTIME_TRACE_EXPORT_LIMIT, max(1, int(limit)))
            endpoint_filter = self._normalize_runtime_trace_export_filter(endpoint)
            living_loop = LivingStatusCore._living_loop_snapshot_locked(self)
            policy_decision = self._runtime_trace_export_policy_decision_summary(
                living_loop.get("policy_decision") if isinstance(living_loop, Mapping) else None
            )
            source_traces, source_window = RuntimeEvidenceReporter._runtime_episode_trace_source_window_locked(
                self,
                surface="bounded_runtime_trace_export_source_window.v1",
                policy="recent_runtime_episode_trace_export_window",
                limit=MAX_RUNTIME_TRACE_EXPORT_LIMIT,
                selection_criteria=[
                    "newest_runtime_episode_traces_first",
                    "bounded_source_window_before_trace_export",
                    "endpoint_filter_applied_inside_window",
                ],
            )
            source_trace_ids = {
                str(episode.get("trace_id", "") or "")
                for episode in source_traces
                if isinstance(episode, Mapping) and str(episode.get("trace_id", "") or "")
            }
            state_by_trace_id = self._runtime_trace_state_by_trace_id_locked(
                trace_ids=source_trace_ids,
            )
            examples: list[dict[str, Any]] = []
            scanned_trace_count = 0
            matched_trace_count = 0
            for episode in source_traces:
                scanned_trace_count += 1
                operation = str(episode.get("operation", "") or "unknown").strip().lower() or "unknown"
                endpoint_path = self._runtime_trace_export_endpoint(operation)
                if endpoint_filter is not None and endpoint_filter not in {operation, endpoint_path.lower(), endpoint_path.lower().lstrip("/")}:
                    continue
                matched_trace_count += 1
                trace_id = str(episode.get("trace_id", "") or "")
                examples.append(
                    self._runtime_trace_export_example_locked(
                        episode,
                        endpoint_path=endpoint_path,
                        state_after=state_by_trace_id.get(trace_id, {}),
                        policy_decision=policy_decision,
                    )
                )
                if len(examples) >= count:
                    break
            source_window.update(
                {
                    "endpoint_filter": endpoint_filter,
                    "source_trace_count_evaluated": int(scanned_trace_count),
                    "matched_trace_count_evaluated": int(matched_trace_count),
                    "candidate_count_returned": int(len(examples)),
                    "returned_count": int(len(examples)),
                    "return_limit_reached": bool(len(examples) >= count),
                    "source_window_exhausted": bool(scanned_trace_count >= len(source_traces)),
                    "quality_metric": "bounded_runtime_trace_export_selection",
                    "state_lookup": {
                        "requested_trace_id_count": int(len(source_trace_ids)),
                        "matched_trace_state_count": int(len(state_by_trace_id)),
                        "source": "runtime_trace_state_by_selected_trace_ids",
                        "lookup_device": "cpu",
                    },
                }
            )
            return {
                "export_kind": "terminus_runtime_trace_dataset_preview",
                "schema_version": RUNTIME_TRACE_EXPORT_SCHEMA_VERSION,
                "training_role": "adapter_distillation_dataset_preview_only_not_training",
                "description": (
                    "Bounded, sanitized Terminus runtime episode examples for future adapter "
                    "distillation dataset preparation. This endpoint does not train a model."
                ),
                "limit": count,
                "max_limit": MAX_RUNTIME_TRACE_EXPORT_LIMIT,
                "endpoint": endpoint_filter,
                "count": len(examples),
                "policy_decision": policy_decision,
                "source_window": source_window,
                "examples": examples,
                "excluded_fields": sorted(_RUNTIME_TRACE_EXPORT_UNSAFE_KEYS),
            }

    def _normalize_runtime_episode_trace(self, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, Mapping):
            return None
        try:
            return RuntimeEpisodeTrace.from_payload(cast(Mapping[str, Any], item)).to_payload()
        except Exception:
            return None

    def _append_runtime_episode_trace_locked(self, episode: Mapping[str, Any]) -> dict[str, Any]:
        return self._interaction_pipeline.append_runtime_episode_trace(episode)

    def _runtime_episode_trace_locked(self, episode_id: str) -> dict[str, Any] | None:
        return self._interaction_pipeline.runtime_episode_trace(episode_id)

    def _replace_runtime_episode_trace_locked(
        self,
        episode_id: str,
        episode: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        return self._interaction_pipeline.replace_runtime_episode_trace(episode_id, episode)

    @staticmethod
    def _normalize_runtime_trace_export_filter(endpoint: str | None) -> str | None:
        if endpoint is None:
            return None
        normalized = " ".join(str(endpoint).split()).strip().lower()
        if not normalized:
            return None
        return normalized if normalized.startswith("/") else normalized.lstrip("/")

    @staticmethod
    def _runtime_trace_export_endpoint(operation: str) -> str:
        normalized = " ".join(str(operation or "unknown").split()).strip().lower() or "unknown"
        if normalized in {"feed", "query", "respond"}:
            return f"/{normalized}"
        return f"/terminus/{normalized}"

    def _runtime_episode_trace_source_window_locked(
        self,
        *,
        surface: str,
        policy: str,
        limit: int,
        selection_criteria: Sequence[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        retained_count = int(len(self._interaction_pipeline.runtime_episode_trace_history))
        source_limit = max(0, int(limit))
        traces = self._interaction_pipeline.runtime_episode_traces(limit=source_limit)
        source_window_count = int(len(traces))
        return traces, {
            "surface": surface,
            "policy": policy,
            "window_policy": policy,
            "source": "interaction_pipeline.runtime_episode_traces",
            "selection_criteria": list(selection_criteria),
            "source_window_limit": int(source_limit),
            "source_window_count": source_window_count,
            "source_record_count": retained_count,
            "source_record_count_known": True,
            "source_payload_truncated": bool(retained_count > source_window_count),
            "source_truncated_count": max(0, retained_count - source_window_count),
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_replay_text_payload_loaded": False,
            "language_reasoning": False,
            "runs_live_tick": False,
            "runs_every_token": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "trains_adapter": False,
            "archival_storage_device": "cpu",
            "source_window_selection_device": "cpu",
            "gpu_used": False,
            "gpu_resident_archival_metadata": False,
            "memory_budget": {
                "max_runtime_episode_traces": int(source_limit),
                "archival_storage_device": "cpu",
            },
        }

    def _runtime_trace_state_by_trace_id_locked(
        self,
        *,
        trace_ids: set[str] | None = None,
    ) -> dict[str, Mapping[str, Any]]:
        states: dict[str, Mapping[str, Any]] = {}
        selection_requested = trace_ids is not None
        requested_trace_ids = {
            str(value)
            for value in (trace_ids if trace_ids is not None else set())
            if str(value)
        }
        if selection_requested and not requested_trace_ids:
            return states
        for trace in self._trace_history:
            if not isinstance(trace, Mapping):
                continue
            trace_id = str(trace.get("trace_id", "") or "")
            if selection_requested and trace_id not in requested_trace_ids:
                continue
            state_after = trace.get("state_after")
            if trace_id and isinstance(state_after, Mapping):
                states.setdefault(trace_id, state_after)
            if selection_requested and len(states) >= len(requested_trace_ids):
                break
        return states

    def _runtime_trace_export_example_locked(
        self,
        episode: Mapping[str, Any],
        *,
        endpoint_path: str,
        state_after: Mapping[str, Any],
        policy_decision: Mapping[str, Any],
    ) -> dict[str, Any]:
        operation = str(episode.get("operation", "") or "unknown").strip().lower() or "unknown"
        prediction = self._runtime_trace_export_safe_value(episode.get("prediction") or {})
        action = self._runtime_trace_export_safe_value(episode.get("action") or {})
        request = self._runtime_trace_export_safe_value(episode.get("request") or {})
        actual_output = self._runtime_trace_export_safe_value(episode.get("actual_output") or {})
        verification = self._runtime_trace_export_safe_value(episode.get("verification") or {})
        failure = self._runtime_trace_export_safe_value(episode.get("failure")) if episode.get("failure") else None
        feedback = self._normalize_runtime_feedback_entries(episode.get("feedback", []))
        feedback_summary = self._runtime_feedback_summary_from_targets(
            [("runtime_episode", str(episode.get("episode_id", "") or ""), feedback)]
        )
        prediction_map = prediction if isinstance(prediction, Mapping) else {}
        action_map = action if isinstance(action, Mapping) else {}
        failure_map = failure if isinstance(failure, Mapping) else {}
        state_revision = self._runtime_trace_export_int(
            episode.get("state_revision"),
            state_after.get("state_revision") if isinstance(state_after, Mapping) else None,
        )
        token_count = self._runtime_trace_export_int(
            episode.get("token_count"),
            state_after.get("token_count") if isinstance(state_after, Mapping) else None,
        )
        example = {
            "example_id": str(episode.get("episode_id", "") or ""),
            "trace_id": str(episode.get("trace_id", "") or ""),
            "dataset_role": "adapter_distillation_example_preview",
            "endpoint": endpoint_path,
            "type": operation,
            "operation": operation,
            "status": str(episode.get("status", "") or "unknown"),
            "timestamp": str(episode.get("created_at", "") or ""),
            "created_at": str(episode.get("created_at", "") or ""),
            "completed_at": str(episode.get("completed_at", "") or ""),
            "state_revision": state_revision,
            "token_count": token_count,
            "context": {
                "endpoint": endpoint_path,
                "operation": operation,
                "request": request,
            },
            "prediction": prediction,
            "proposed_answer": prediction_map.get("proposed_answer"),
            "proposed_action": (
                prediction_map.get("proposed_action")
                or action_map.get("proposed_action")
                or action_map.get("action_type")
            ),
            "action": action,
            "actual_output": actual_output,
            "verification": verification,
            "feedback": feedback,
            "feedback_summary": feedback_summary,
            "policy_decision": self._runtime_trace_export_policy_decision_summary(policy_decision),
            "corrected_output": self._runtime_trace_export_safe_value(episode.get("corrected_output"))
            if episode.get("corrected_output") is not None
            else None,
            "provenance": str(episode.get("provenance", "") or "observed"),
            "latency_ms": episode.get("latency_ms"),
            "failure": failure,
            "error": failure_map.get("message") if failure_map else None,
        }
        return cast(dict[str, Any], self._runtime_trace_export_safe_value(example))

    def _runtime_trace_export_policy_decision_summary(self, policy_decision: Any) -> dict[str, Any]:
        data = dict(policy_decision) if isinstance(policy_decision, Mapping) else {}
        if not data:
            return {}
        reasons = data.get("reasons")
        reason_codes: list[str] = []
        existing_reason_codes = data.get("reason_codes")
        if isinstance(existing_reason_codes, Sequence) and not isinstance(existing_reason_codes, (str, bytes)):
            reason_codes = [str(item).strip() for item in existing_reason_codes if str(item).strip()]
        elif isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes)):
            reason_codes = [
                str(item.get("code", "")).strip()
                for item in reasons
                if isinstance(item, Mapping) and str(item.get("code", "")).strip()
            ]
        summary = {
            "schema_version": data.get("schema_version"),
            "action": data.get("action"),
            "recommendation": data.get("recommendation"),
            "reason_codes": reason_codes,
            "risk": data.get("risk"),
            "expected_information_gain": data.get("expected_information_gain"),
            "expected_goal_progress": data.get("expected_goal_progress"),
            "expected_cost": data.get("expected_cost"),
            "uncertainty": data.get("uncertainty"),
            "advisory": data.get("advisory", True),
            "executable": data.get("executable", False),
            "target_episode_id": data.get("target_episode_id"),
            "target_action_id": data.get("target_action_id"),
            "suggested_endpoint": data.get("suggested_endpoint"),
            "created_at": data.get("created_at"),
        }
        return cast(dict[str, Any], self._runtime_trace_export_safe_value(summary))

    @staticmethod
    def _runtime_trace_export_key_is_safe(key: Any) -> bool:
        normalized = str(key).strip().lower()
        if not normalized:
            return False
        if normalized in _RUNTIME_TRACE_EXPORT_ALLOWED_TOKEN_KEYS:
            return True
        if normalized in _RUNTIME_TRACE_EXPORT_UNSAFE_KEYS:
            return False
        if "token" in normalized:
            return False
        return not any(marker in normalized for marker in _RUNTIME_TRACE_EXPORT_UNSAFE_KEY_MARKERS)

    def _runtime_trace_export_safe_value(
        self,
        value: Any,
        *,
        list_item_limit: int = _RUNTIME_TRACE_EXPORT_MAX_LIST_ITEMS,
    ) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            text = value
            if len(text) > _RUNTIME_TRACE_EXPORT_MAX_STRING_CHARS:
                return text[:_RUNTIME_TRACE_EXPORT_MAX_STRING_CHARS] + "…"
            return text
        if isinstance(value, Mapping):
            sanitized: dict[str, Any] = {}
            for key, item in islice(value.items(), _RUNTIME_TRACE_EXPORT_MAX_MAPPING_ITEMS):
                if not self._runtime_trace_export_key_is_safe(key):
                    continue
                sanitized[str(key)] = self._runtime_trace_export_safe_value(
                    item,
                    list_item_limit=list_item_limit,
                )
            return sanitized
        if isinstance(value, (list, tuple, deque)):
            count = max(0, int(list_item_limit))
            return [
                self._runtime_trace_export_safe_value(
                    item,
                    list_item_limit=list_item_limit,
                )
                for item in islice(value, count)
            ]
        return self._runtime_trace_export_safe_value(
            str(value),
            list_item_limit=list_item_limit,
        )

    @staticmethod
    def _runtime_trace_export_int(*values: Any) -> int | None:
        for value in values:
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    def _runtime_feedback_summary_from_targets(
        self,
        targets: Sequence[tuple[str, str, Any]],
        *,
        recent_limit: int = DEFAULT_RUNTIME_FEEDBACK_HISTORY,
    ) -> dict[str, Any]:
        status_counts: Counter[str] = Counter({"verified": 0, "contradicted": 0, "unverified": 0})
        verdict_counts: Counter[str] = Counter({"verified": 0, "contradicted": 0, "unverified": 0})
        target_counts: Counter[str] = Counter({"runtime_episode": 0, "action": 0})
        recent: list[dict[str, Any]] = []
        latest_feedback_at = ""
        total = 0
        for target_type, target_id, feedback_entries in targets:
            normalized_entries = self._normalize_runtime_feedback_entries(feedback_entries)
            if not normalized_entries:
                continue
            target_counts[target_type] += len(normalized_entries)
            for entry in normalized_entries:
                total += 1
                verdict = self._normalize_action_text(entry.get("verdict", "unverified")).lower()
                if verdict not in {"verified", "contradicted", "unverified"}:
                    verdict = "unverified"
                applied_status = self._normalize_action_text(entry.get("applied_status", "")).lower()
                if applied_status not in {"verified", "contradicted", "unverified"}:
                    applied_status = self._runtime_feedback_applied_status(
                        verdict,
                        corrected=entry.get("corrected_output") is not None,
                    )
                verdict_counts[verdict] += 1
                status_counts[applied_status] += 1
                created_at = self._normalize_feedback_text(entry.get("created_at", ""), max_chars=80)
                if created_at and created_at > latest_feedback_at:
                    latest_feedback_at = created_at
                evidence = (
                    entry.get("evidence")
                    if isinstance(entry.get("evidence"), Sequence)
                    and not isinstance(entry.get("evidence"), (str, bytes))
                    else []
                )
                tags = (
                    entry.get("tags")
                    if isinstance(entry.get("tags"), Sequence)
                    and not isinstance(entry.get("tags"), (str, bytes))
                    else []
                )
                try:
                    confidence = max(0.0, min(1.0, float(entry.get("confidence", 0.0) or 0.0)))
                except (TypeError, ValueError):
                    confidence = 0.0
                recent.append(
                    {
                        "feedback_id": self._normalize_feedback_text(entry.get("feedback_id", ""), max_chars=80),
                        "created_at": created_at,
                        "target_type": target_type,
                        "target_id": self._normalize_feedback_text(
                            entry.get("target_id") or target_id,
                            max_chars=160,
                        ),
                        "verdict": verdict,
                        "applied_status": applied_status,
                        "confidence": confidence,
                        "summary": self._normalize_feedback_text(entry.get("summary", "")),
                        "tags": [
                            self._normalize_feedback_text(tag, max_chars=64)
                            for tag in list(tags)[:DEFAULT_RUNTIME_FEEDBACK_TAG_LIMIT]
                        ],
                        "evaluator_id": self._normalize_feedback_text(entry.get("evaluator_id", ""), max_chars=160),
                        "evidence_count": int(len(evidence)),
                        "has_corrected_output": entry.get("corrected_output") is not None,
                    }
                )

        recent.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        if status_counts["contradicted"] > 0:
            grounding_impact = "contradictions_present"
        elif status_counts["unverified"] > 0:
            grounding_impact = "needs_verification"
        elif status_counts["verified"] > 0:
            grounding_impact = "operator_verified"
        else:
            grounding_impact = "none"
        return {
            "feedback_count": int(total),
            "verified_count": int(status_counts["verified"]),
            "contradicted_count": int(status_counts["contradicted"]),
            "unverified_count": int(status_counts["unverified"]),
            "status_counts": {
                "verified": int(status_counts["verified"]),
                "contradicted": int(status_counts["contradicted"]),
                "unverified": int(status_counts["unverified"]),
            },
            "verdict_counts": {
                "verified": int(verdict_counts["verified"]),
                "contradicted": int(verdict_counts["contradicted"]),
                "unverified": int(verdict_counts["unverified"]),
            },
            "target_counts": {
                "runtime_episode": int(target_counts["runtime_episode"]),
                "action": int(target_counts["action"]),
            },
            "recent_feedback": recent[: max(1, int(recent_limit))],
            "latest_feedback_at": latest_feedback_at,
            "grounding_impact": grounding_impact,
        }

    def _runtime_feedback_summary_locked(self) -> dict[str, Any]:
        targets: list[tuple[str, str, Any]] = []
        trace_window, trace_source_window = RuntimeEvidenceReporter._runtime_episode_trace_source_window_locked(
            self,
            surface="bounded_runtime_feedback_summary_trace_source_window.v1",
            policy="recent_runtime_episode_trace_feedback_window",
            limit=RUNTIME_TRACE_FEEDBACK_SOURCE_WINDOW_LIMIT,
            selection_criteria=[
                "newest_runtime_episode_traces_first",
                "bounded_feedback_summary_before_status_or_trace_export",
                "feedback_entries_read_only_from_selected_traces",
            ],
        )
        for episode in trace_window:
            if isinstance(episode, Mapping):
                targets.append(("runtime_episode", str(episode.get("episode_id", "") or ""), episode.get("feedback", [])))
        action_history = self._action_executor.history
        action_limit = max(1, int(DEFAULT_RUNTIME_FEEDBACK_HISTORY))
        action_window_count = 0
        for action in islice(action_history, action_limit):
            action_window_count += 1
            if isinstance(action, Mapping):
                targets.append(("action", str(action.get("action_id", "") or ""), action.get("feedback", [])))
        summary = self._runtime_feedback_summary_from_targets(targets)
        summary["source_window"] = {
            "surface": "bounded_runtime_feedback_summary_source_window.v1",
            "policy": "recent_runtime_trace_and_action_feedback_window",
            "window_policy": "recent_runtime_trace_and_action_feedback_window",
            "selection_criteria": [
                "newest_runtime_episode_traces_first",
                "newest_action_records_first",
                "bounded_feedback_summary_before_status_or_trace_export",
            ],
            "runtime_episode_trace_source_window": trace_source_window,
            "action_history_window_limit": int(action_limit),
            "action_history_window_count": int(action_window_count),
            "action_history_record_count": int(len(action_history)),
            "action_history_payload_truncated": bool(len(action_history) > action_window_count),
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_replay_text_payload_loaded": False,
            "language_reasoning": False,
            "runs_live_tick": False,
            "runs_every_token": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "summary_device": "cpu",
            "gpu_used": False,
            "gpu_resident_archival_metadata": False,
            "memory_budget": {
                "max_runtime_episode_traces": int(RUNTIME_TRACE_FEEDBACK_SOURCE_WINDOW_LIMIT),
                "max_action_records": int(action_limit),
                "archival_storage_device": "cpu",
            },
        }
        return summary

    def _runtime_episode_payload_locked(
        self,
        *,
        operation: str,
        request: Mapping[str, Any],
        prediction: Mapping[str, Any],
        action: Mapping[str, Any],
        actual_output: Mapping[str, Any] | None,
        verification: Mapping[str, Any] | None,
        started_perf: float,
        created_at: str,
        trace_id: str,
        trace_path: str = "",
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        completed_at = datetime.now(timezone.utc).isoformat()
        latency_ms = max(0.0, (time.perf_counter() - started_perf) * 1000.0)
        failure = None
        status = "succeeded"
        normalized_verification = dict(verification or {})
        if error is not None:
            status = "failed"
            failure = {
                "error_type": type(error).__name__,
                "message": str(error),
            }
            normalized_verification = {
                "status": "contradicted",
                "success": False,
                "confidence": 1.0,
                "contradiction": True,
                "summary": str(error),
            }
        provenance = "contradicted" if status == "failed" else str(normalized_verification.get("provenance", "observed") or "observed")
        if str(normalized_verification.get("status", "")).lower() == "verified":
            provenance = "verified"
        elif str(normalized_verification.get("status", "")).lower() == "contradicted":
            provenance = "contradicted"
        payload = {
            "episode_id": str(uuid4()),
            "trace_id": trace_id,
            "trace_path": trace_path,
            "operation": operation,
            "status": status,
            "created_at": created_at,
            "completed_at": completed_at,
            "latency_ms": latency_ms,
            "request": dict(request),
            "prediction": dict(prediction),
            "action": dict(action),
            "actual_output": dict(actual_output or {}),
            "verification": normalized_verification,
            "provenance": provenance,
            "failure": failure,
        }
        return RuntimeEpisodeTrace.from_payload(cast(Mapping[str, Any], self._json_safe(payload))).to_payload()

    def _feed_runtime_actual_output(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        return build_feed_runtime_actual_output(summary)

    def _query_runtime_actual_output(self, result: Mapping[str, Any]) -> dict[str, Any]:
        return build_query_runtime_actual_output(result)

    def _query_runtime_verification(self, result: Mapping[str, Any]) -> dict[str, Any]:
        return build_query_runtime_verification(result)

    def _respond_runtime_actual_output(
        self,
        *,
        response: Mapping[str, Any],
        action_assist: Mapping[str, Any] | None,
        outcome_score: float,
    ) -> dict[str, Any]:
        return build_respond_runtime_actual_output(
            response=response,
            action_assist=action_assist,
            outcome_score=outcome_score,
        )

    def _respond_runtime_verification(
        self,
        *,
        response: Mapping[str, Any],
        action_assist: Mapping[str, Any] | None,
        outcome_score: float,
    ) -> dict[str, Any]:
        return build_respond_runtime_verification(
            response=response,
            action_assist=action_assist,
            outcome_score=outcome_score,
        )

