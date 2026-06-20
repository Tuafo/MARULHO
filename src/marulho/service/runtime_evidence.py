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
from marulho.service.living_loop_replay import build_replay_plan

DEFAULT_RUNTIME_TRACE_EXPORT_LIMIT = 20
MAX_RUNTIME_TRACE_EXPORT_LIMIT = 50
DEFAULT_REPLAY_DATASET_EXPORT_LIMIT = DEFAULT_RUNTIME_TRACE_EXPORT_LIMIT
MAX_REPLAY_DATASET_EXPORT_LIMIT = MAX_RUNTIME_TRACE_EXPORT_LIMIT
DEFAULT_REPLAY_SAMPLE_HISTORY = 256
REPLAY_DATASET_PREVIEW_TRACE_SOURCE_WINDOW_LIMIT = MAX_REPLAY_DATASET_EXPORT_LIMIT
REPLAY_DATASET_SAMPLE_LINK_SOURCE_WINDOW_LIMIT = 64
REPLAY_DATASET_SAMPLE_LINK_CANDIDATE_WINDOW_LIMIT = 16
RUNTIME_TRACE_EXPORT_SCHEMA_VERSION = 1
REPLAY_DATASET_SCHEMA_VERSION = 1
REPLAY_DATASET_TRAINING_ROLE = "replay_dataset_preview_only_not_training_no_mutation"
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
    """Runtime trace, feedback summary, and replay dataset preview helpers.

    This is the learning-evidence lane: it creates sanitized audit/export
    artifacts but does not train adapters, mutate memory, execute actions, or
    promote facts.
    """

    @staticmethod
    def _replay_dataset_count_map(value: Any) -> dict[str, int]:
        if not isinstance(value, Mapping):
            return {}
        result: dict[str, int] = {}
        for key, count in value.items():
            try:
                result[str(key)] = int(count or 0)
            except (TypeError, ValueError):
                result[str(key)] = 0
        return result

    def _replay_dataset_latest_history_timestamp_locked(self) -> str | None:
        for record in list(self._replay_sample_history):
            if not isinstance(record, Mapping):
                continue
            created_at = self._normalize_feedback_text(record.get("created_at", ""), max_chars=80)
            if created_at:
                return created_at
        return None

    def _replay_dataset_summary_from_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        replay_sample_summary = payload.get("replay_sample_summary")
        latest_history_timestamp = payload.get("latest_history_timestamp")
        if latest_history_timestamp in ("", None) and isinstance(replay_sample_summary, Mapping):
            latest = replay_sample_summary.get("latest_history_item")
            if isinstance(latest, Mapping):
                latest_history_timestamp = latest.get("created_at")
        if latest_history_timestamp in ("", None):
            latest_history_timestamp = self._replay_dataset_latest_history_timestamp_locked()

        latest_export_timestamp = (
            payload.get("latest_export_timestamp")
            or payload.get("created_at")
        )
        summary = {
            "export_kind": str(payload.get("export_kind") or "terminus_replay_dataset_preview"),
            "schema_version": int(payload.get("schema_version", REPLAY_DATASET_SCHEMA_VERSION) or REPLAY_DATASET_SCHEMA_VERSION),
            "training_role": str(payload.get("training_role") or REPLAY_DATASET_TRAINING_ROLE),
            "endpoint": str(payload.get("endpoint") or "/terminus/replay-dataset/preview"),
            "filter_endpoint": payload.get("filter_endpoint"),
            "limit": int(payload.get("limit", 0) or 0),
            "max_limit": int(payload.get("max_limit", MAX_REPLAY_DATASET_EXPORT_LIMIT) or MAX_REPLAY_DATASET_EXPORT_LIMIT),
            "count": int(payload.get("count", 0) or 0),
            "positive_count": int(payload.get("positive_count", 0) or 0),
            "negative_count": int(payload.get("negative_count", 0) or 0),
            "provenance_counts": self._replay_dataset_count_map(payload.get("provenance_counts")),
            "example_type_counts": self._replay_dataset_count_map(payload.get("example_type_counts")),
            "safety_flags": dict(payload.get("safety_flags", {})) if isinstance(payload.get("safety_flags"), Mapping) else {},
            "empty_reason": payload.get("empty_reason"),
            "latest_export_timestamp": str(latest_export_timestamp) if latest_export_timestamp not in ("", None) else None,
            "latest_history_timestamp": str(latest_history_timestamp) if latest_history_timestamp not in ("", None) else None,
        }
        source_window = payload.get("source_window")
        if isinstance(source_window, Mapping):
            summary["source_window"] = self._runtime_trace_export_safe_value(
                dict(source_window)
            )
        return cast(dict[str, Any], self._runtime_trace_export_safe_value(summary))

    def _replay_dataset_preview_payload_locked(
        self,
        *,
        limit: int = DEFAULT_REPLAY_DATASET_EXPORT_LIMIT,
        endpoint: str | None = None,
        living_loop: Mapping[str, Any] | None = None,
        plan: Mapping[str, Any] | None = None,
        replay_sample_summary: Mapping[str, Any] | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        count = min(MAX_REPLAY_DATASET_EXPORT_LIMIT, max(1, int(limit)))
        endpoint_filter = self._normalize_runtime_trace_export_filter(endpoint)
        export_created_at = created_at or datetime.now(timezone.utc).isoformat()
        before = self._replay_sample_state_counts_locked()
        if living_loop is None:
            living_loop = self._living_loop_snapshot_locked(
                include_replay_dataset_summary=False,
            )
        policy_decision = self._runtime_trace_export_policy_decision_summary(
            living_loop.get("policy_decision") if isinstance(living_loop, Mapping) else None
        )
        replay_plan = dict(plan) if isinstance(plan, Mapping) else build_replay_plan(living_loop, limit=MAX_REPLAY_DATASET_EXPORT_LIMIT).to_payload()
        replay_plan_summary = self._replay_plan_summary(replay_plan)
        sample_summary = (
            dict(replay_sample_summary)
            if isinstance(replay_sample_summary, Mapping)
            else self._replay_sample_summary_locked()
        )
        source_traces, source_window = self._replay_dataset_preview_source_window_locked()
        source_trace_ids = {
            str(episode.get("trace_id", "") or "")
            for episode in source_traces
            if isinstance(episode, Mapping) and str(episode.get("trace_id", "") or "")
        }
        state_by_trace_id = self._runtime_trace_state_by_trace_id_locked(
            trace_ids=source_trace_ids,
        )
        candidates_by_target = self._replay_dataset_candidates_by_target(replay_plan.get("candidates", []))
        sample_links_by_target, replay_sample_link_source_window = (
            self._replay_dataset_sample_links_by_target_locked(with_report=True)
        )
        source_window["replay_sample_link_source_window"] = replay_sample_link_source_window
        source_window["state_lookup"] = {
            "requested_trace_id_count": int(len(source_trace_ids)),
            "matched_trace_state_count": int(len(state_by_trace_id)),
            "source": "runtime_trace_state_by_selected_trace_ids",
            "lookup_device": "cpu",
        }
        items: list[dict[str, Any]] = []
        matched_trace_count = 0
        scanned_trace_count = 0
        for episode in source_traces:
            scanned_trace_count += 1
            operation = str(episode.get("operation", "") or "unknown").strip().lower() or "unknown"
            endpoint_path = self._runtime_trace_export_endpoint(operation)
            if endpoint_filter is not None and endpoint_filter not in {operation, endpoint_path.lower(), endpoint_path.lower().lstrip("/")}:
                continue
            matched_trace_count += 1
            trace_id = str(episode.get("trace_id", "") or "")
            example = self._runtime_trace_export_example_locked(
                episode,
                endpoint_path=endpoint_path,
                state_after=state_by_trace_id.get(trace_id, {}),
                policy_decision=policy_decision,
                replay_plan_summary=replay_plan_summary,
                replay_sample_summary=sample_summary,
            )
            target_id = str(example.get("example_id", "") or episode.get("episode_id", "") or "")
            target_key = ("runtime_episode", target_id)
            items.append(
                self._replay_dataset_item_from_trace_example(
                    example,
                    replay_candidate=candidates_by_target.get(target_key),
                    replay_sample_linkage=sample_links_by_target.get(target_key),
                )
            )
            if len(items) >= count:
                break
        source_window.update(
            {
                "endpoint_filter": endpoint_filter,
                "source_trace_count_evaluated": int(scanned_trace_count),
                "matched_trace_count_evaluated": int(matched_trace_count),
                "candidate_count_returned": int(len(items)),
                "returned_count": int(len(items)),
                "return_limit_reached": bool(len(items) >= count),
                "source_window_exhausted": bool(scanned_trace_count >= len(source_traces)),
                "quality_metric": "bounded_replay_dataset_preview_trace_selection",
            }
        )

        positive_count = sum(1 for item in items if bool(item.get("has_positive_example")))
        negative_count = sum(1 for item in items if bool(item.get("has_negative_example")))
        provenance_counts: Counter[str] = Counter(
            str(item.get("provenance_label", "unknown") or "unknown") for item in items
        )
        example_type_counts: Counter[str] = Counter(
            str(item.get("example_type", "unknown") or "unknown") for item in items
        )
        after = self._replay_sample_state_counts_locked()
        payload = {
            "schema_version": REPLAY_DATASET_SCHEMA_VERSION,
            "export_kind": "terminus_replay_dataset_preview",
            "training_role": REPLAY_DATASET_TRAINING_ROLE,
            "description": (
                "Curated replay dataset preview assembled from sanitized runtime traces, feedback, "
                "replay-plan context, and operator-gated replay-sample linkage. This endpoint is "
                "export-only and does not train, mutate memory, post feedback, execute actions, or "
                "make external calls."
            ),
            "created_at": export_created_at,
            "latest_export_timestamp": export_created_at,
            "latest_history_timestamp": self._replay_dataset_latest_history_timestamp_locked(),
            "endpoint": "/terminus/replay-dataset/preview",
            "limit": count,
            "max_limit": MAX_REPLAY_DATASET_EXPORT_LIMIT,
            "filter_endpoint": endpoint_filter,
            "count": len(items),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "provenance_counts": dict(provenance_counts),
            "example_type_counts": dict(example_type_counts),
            "policy_decision": policy_decision,
            "replay_plan_summary": replay_plan_summary,
            "replay_sample_summary": sample_summary,
            "replay_executor_summary": sample_summary,
            "source_window": source_window,
            "safety_flags": self._replay_dataset_safety_flags(before=before, after=after),
            "before": before,
            "after": after,
            "items": items,
            "excluded_fields": sorted(_RUNTIME_TRACE_EXPORT_UNSAFE_KEYS),
        }
        if not items:
            payload["empty_reason"] = "checkpoint_contains_no_eligible_sanitized_runtime_traces"
        return cast(
            dict[str, Any],
            self._runtime_trace_export_safe_value(
                payload,
                list_item_limit=max(_RUNTIME_TRACE_EXPORT_MAX_LIST_ITEMS, count),
            ),
        )

    def _replay_dataset_preview_summary_locked(
        self,
        *,
        limit: int = DEFAULT_REPLAY_DATASET_EXPORT_LIMIT,
        endpoint: str | None = None,
        living_loop: Mapping[str, Any] | None = None,
        plan: Mapping[str, Any] | None = None,
        replay_sample_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._replay_dataset_preview_payload_locked(
            limit=limit,
            endpoint=endpoint,
            living_loop=living_loop,
            plan=plan,
            replay_sample_summary=replay_sample_summary,
        )
        return self._replay_dataset_summary_from_payload(payload)

    def export_runtime_trace_examples(
        self,
        *,
        limit: int = DEFAULT_RUNTIME_TRACE_EXPORT_LIMIT,
        endpoint: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            count = min(MAX_RUNTIME_TRACE_EXPORT_LIMIT, max(1, int(limit)))
            endpoint_filter = self._normalize_runtime_trace_export_filter(endpoint)
            living_loop = self._living_loop_snapshot_locked()
            policy_decision = self._runtime_trace_export_policy_decision_summary(
                living_loop.get("policy_decision") if isinstance(living_loop, Mapping) else None
            )
            replay_plan = (
                dict(living_loop.get("replay_plan"))
                if isinstance(living_loop, Mapping) and isinstance(living_loop.get("replay_plan"), Mapping)
                else build_replay_plan(living_loop, limit=MAX_REPLAY_DATASET_EXPORT_LIMIT).to_payload()
            )
            replay_plan_summary = self._replay_plan_summary(replay_plan)
            replay_sample_summary = self._replay_sample_summary_locked()
            replay_dataset_summary = self._replay_dataset_preview_summary_locked(
                limit=count,
                endpoint=endpoint_filter,
                living_loop=living_loop,
                plan=replay_plan,
                replay_sample_summary=replay_sample_summary,
            )
            state_by_trace_id = self._runtime_trace_state_by_trace_id_locked()
            examples: list[dict[str, Any]] = []
            for episode in list(self._interaction_pipeline.runtime_episode_traces()):
                operation = str(episode.get("operation", "") or "unknown").strip().lower() or "unknown"
                endpoint_path = self._runtime_trace_export_endpoint(operation)
                if endpoint_filter is not None and endpoint_filter not in {operation, endpoint_path.lower(), endpoint_path.lower().lstrip("/")}:
                    continue
                trace_id = str(episode.get("trace_id", "") or "")
                examples.append(
                    self._runtime_trace_export_example_locked(
                        episode,
                        endpoint_path=endpoint_path,
                        state_after=state_by_trace_id.get(trace_id, {}),
                        policy_decision=policy_decision,
                        replay_plan_summary=replay_plan_summary,
                        replay_sample_summary=replay_sample_summary,
                    )
                )
                if len(examples) >= count:
                    break
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
                "replay_plan_summary": replay_plan_summary,
                "replay_sample_summary": replay_sample_summary,
                "replay_executor_summary": replay_sample_summary,
                "replay_dataset_summary": replay_dataset_summary,
                "examples": examples,
                "excluded_fields": sorted(_RUNTIME_TRACE_EXPORT_UNSAFE_KEYS),
            }

    def replay_dataset_preview(
        self,
        *,
        limit: int = DEFAULT_REPLAY_DATASET_EXPORT_LIMIT,
        endpoint: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            return self._replay_dataset_preview_payload_locked(limit=limit, endpoint=endpoint)

    def replay_dataset_candidates(self, *, limit: int = DEFAULT_REPLAY_DATASET_EXPORT_LIMIT) -> dict[str, Any]:
        with self._lock:
            count = min(MAX_REPLAY_DATASET_EXPORT_LIMIT, max(1, int(limit)))
            before = self._replay_sample_state_counts_locked()
            living_loop = self._living_loop_snapshot_locked()
            plan = build_replay_plan(living_loop, limit=count).to_payload()
            after = self._replay_sample_state_counts_locked()
            candidates = [
                self._replay_sample_candidate_payload(candidate)
                for candidate in list(plan.get("candidates", []))
                if isinstance(candidate, Mapping)
            ]
            return cast(
                dict[str, Any],
                self._runtime_trace_export_safe_value(
                    {
                        "schema_version": REPLAY_DATASET_SCHEMA_VERSION,
                        "export_kind": "terminus_replay_dataset_candidates_preview",
                        "training_role": REPLAY_DATASET_TRAINING_ROLE,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "endpoint": "/terminus/replay-dataset/candidates",
                        "limit": count,
                        "max_limit": MAX_REPLAY_DATASET_EXPORT_LIMIT,
                        "count": len(candidates),
                        "candidates": candidates,
                        "replay_plan_summary": self._replay_plan_summary(plan),
                        "safety_flags": self._replay_dataset_safety_flags(before=before, after=after),
                        "excluded_fields": sorted(_RUNTIME_TRACE_EXPORT_UNSAFE_KEYS),
                    },
                    list_item_limit=max(_RUNTIME_TRACE_EXPORT_MAX_LIST_ITEMS, count),
                ),
            )

    def replay_dataset_history(self, *, limit: int = DEFAULT_REPLAY_SAMPLE_HISTORY) -> dict[str, Any]:
        with self._lock:
            count = max(1, min(DEFAULT_REPLAY_SAMPLE_HISTORY, int(limit)))
            before = self._replay_sample_state_counts_locked()
            history = [deepcopy(item) for item in list(self._replay_sample_history)[:count]]
            after = self._replay_sample_state_counts_locked()
            return cast(
                dict[str, Any],
                self._runtime_trace_export_safe_value(
                    {
                        "schema_version": REPLAY_DATASET_SCHEMA_VERSION,
                        "export_kind": "terminus_replay_dataset_history_preview",
                        "training_role": REPLAY_DATASET_TRAINING_ROLE,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "endpoint": "/terminus/replay-dataset/history",
                        "source_endpoint": "/terminus/replay-sample/history",
                        "limit": count,
                        "max_limit": DEFAULT_REPLAY_SAMPLE_HISTORY,
                        "count": int(len(self._replay_sample_history)),
                        "history": history,
                        "replay_sample_summary": self._replay_sample_summary_locked(),
                        "safety_flags": self._replay_dataset_safety_flags(before=before, after=after),
                        "excluded_fields": sorted(_RUNTIME_TRACE_EXPORT_UNSAFE_KEYS),
                    },
                    list_item_limit=max(_RUNTIME_TRACE_EXPORT_MAX_LIST_ITEMS, count),
                ),
            )

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

    def _replay_dataset_preview_source_window_locked(
        self,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        retained_count = int(len(self._interaction_pipeline.runtime_episode_trace_history))
        source_limit = max(1, int(REPLAY_DATASET_PREVIEW_TRACE_SOURCE_WINDOW_LIMIT))
        traces = self._interaction_pipeline.runtime_episode_traces(limit=source_limit)
        source_window_count = int(len(traces))
        return traces, {
            "surface": "bounded_replay_dataset_preview_source_window.v1",
            "policy": "recent_runtime_episode_trace_window_with_bounded_replay_links",
            "window_policy": "recent_runtime_episode_trace_window_with_bounded_replay_links",
            "source": "interaction_pipeline.runtime_episode_traces",
            "selection_criteria": [
                "newest_runtime_episode_traces_first",
                "bounded_source_window_before_dataset_preview",
                "operator_replay_sample_links_indexed_from_recent_window",
            ],
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
                "max_replay_sample_records": int(
                    REPLAY_DATASET_SAMPLE_LINK_SOURCE_WINDOW_LIMIT
                ),
                "max_selected_candidates_per_replay_sample": int(
                    REPLAY_DATASET_SAMPLE_LINK_CANDIDATE_WINDOW_LIMIT
                ),
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
        replay_plan_summary: Mapping[str, Any],
        replay_sample_summary: Mapping[str, Any],
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
            "replay_plan_summary": self._replay_plan_summary(replay_plan_summary),
            "replay_sample_summary": self._runtime_trace_export_safe_value(dict(replay_sample_summary)),
            "corrected_output": self._runtime_trace_export_safe_value(episode.get("corrected_output"))
            if episode.get("corrected_output") is not None
            else None,
            "provenance": str(episode.get("provenance", "") or "observed"),
            "latency_ms": episode.get("latency_ms"),
            "failure": failure,
            "error": failure_map.get("message") if failure_map else None,
        }
        return cast(dict[str, Any], self._runtime_trace_export_safe_value(example))

    @staticmethod
    def _replay_dataset_candidates_by_target(candidates: Any) -> dict[tuple[str, str], dict[str, Any]]:
        by_target: dict[tuple[str, str], dict[str, Any]] = {}
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            return by_target
        for raw in candidates:
            if not isinstance(raw, Mapping):
                continue
            target_type = str(raw.get("target_type", "") or "").strip()
            target_id = str(raw.get("target_id", "") or "").strip()
            if target_type and target_id:
                by_target.setdefault((target_type, target_id), dict(raw))
        return by_target

    def _replay_dataset_sample_links_by_target_locked(
        self,
        *,
        with_report: bool = False,
    ) -> Any:
        links: dict[tuple[str, str], dict[str, Any]] = {}
        retained_count = int(len(self._replay_sample_history))
        source_limit = max(0, int(REPLAY_DATASET_SAMPLE_LINK_SOURCE_WINDOW_LIMIT))
        candidate_limit = max(0, int(REPLAY_DATASET_SAMPLE_LINK_CANDIDATE_WINDOW_LIMIT))
        source_window_count = 0
        selected_candidate_source_count = 0
        selected_candidate_window_count = 0
        selected_candidate_truncated_count = 0
        for record in islice(self._replay_sample_history, source_limit):
            source_window_count += 1
            if not isinstance(record, Mapping):
                continue
            selected_candidates = record.get("selected_candidates")
            if not isinstance(selected_candidates, Sequence) or isinstance(selected_candidates, (str, bytes)):
                continue
            candidate_count = len(selected_candidates)
            selected_candidate_source_count += int(candidate_count)
            selected_candidate_truncated_count += max(0, int(candidate_count) - candidate_limit)
            for raw_candidate in islice(selected_candidates, candidate_limit):
                selected_candidate_window_count += 1
                if not isinstance(raw_candidate, Mapping):
                    continue
                target_type = self._normalize_feedback_text(raw_candidate.get("target_type", ""), max_chars=64)
                target_id = self._normalize_feedback_text(raw_candidate.get("target_id", ""), max_chars=160)
                if not target_type or not target_id:
                    continue
                key = (target_type, target_id)
                link = links.setdefault(
                    key,
                    {
                        "selected": True,
                        "target_type": target_type,
                        "target_id": target_id,
                        "replay_sample_ids": [],
                        "execution_ids": [],
                        "modes": [],
                        "candidate_ids": [],
                        "latest": None,
                    },
                )
                replay_sample_id = self._normalize_feedback_text(record.get("replay_sample_id", ""), max_chars=160)
                execution_id = self._normalize_feedback_text(record.get("execution_id", ""), max_chars=160)
                mode = self._normalize_feedback_text(record.get("mode", ""), max_chars=32)
                candidate_id = self._normalize_feedback_text(raw_candidate.get("candidate_id", ""), max_chars=160)
                if replay_sample_id and replay_sample_id not in link["replay_sample_ids"]:
                    link["replay_sample_ids"].append(replay_sample_id)
                if execution_id and execution_id not in link["execution_ids"]:
                    link["execution_ids"].append(execution_id)
                if mode and mode not in link["modes"]:
                    link["modes"].append(mode)
                if candidate_id and candidate_id not in link["candidate_ids"]:
                    link["candidate_ids"].append(candidate_id)
                if link["latest"] is None:
                    link["latest"] = {
                        "replay_sample_id": replay_sample_id,
                        "execution_id": execution_id or None,
                        "created_at": self._normalize_feedback_text(record.get("created_at", ""), max_chars=80),
                        "mode": mode,
                        "status": self._normalize_feedback_text(record.get("status", ""), max_chars=80),
                        "candidate_id": candidate_id,
                        "operator_id": self._normalize_feedback_text(record.get("operator_id", ""), max_chars=160),
                        "safety_flags": dict(record.get("safety_flags", {})) if isinstance(record.get("safety_flags"), Mapping) else {},
                    }
        if not with_report:
            return links
        report = {
            "surface": "bounded_replay_dataset_sample_link_source_window.v1",
            "policy": "recent_replay_sample_link_window",
            "window_policy": "recent_replay_sample_link_window",
            "source": "replay_controller.replay_sample_history",
            "selection_criteria": [
                "newest_replay_sample_records_first",
                "bounded_selected_candidates_per_sample",
                "link_runtime_episode_targets_only_for_preview_context",
            ],
            "source_window_limit": int(source_limit),
            "source_window_count": int(source_window_count),
            "source_record_count": int(retained_count),
            "source_record_count_known": True,
            "source_payload_truncated": bool(retained_count > source_window_count),
            "source_truncated_count": max(0, retained_count - source_window_count),
            "selected_candidate_window_limit_per_sample": int(candidate_limit),
            "selected_candidate_source_scope": "stored_sanitized_replay_sample_payload",
            "raw_selected_candidate_count_known": False,
            "stored_selected_candidate_payload_limit": int(_RUNTIME_TRACE_EXPORT_MAX_LIST_ITEMS),
            "selected_candidate_source_count": int(selected_candidate_source_count),
            "selected_candidate_window_count": int(selected_candidate_window_count),
            "selected_candidate_truncated_count": int(selected_candidate_truncated_count),
            "target_link_count": int(len(links)),
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
            "lookup_device": "cpu",
            "gpu_used": False,
            "gpu_resident_archival_metadata": False,
            "memory_budget": {
                "max_replay_sample_records": int(source_limit),
                "max_selected_candidates_per_sample": int(candidate_limit),
                "archival_storage_device": "cpu",
            },
        }
        return links, report

    def _replay_dataset_verification_label(self, example: Mapping[str, Any]) -> str:
        feedback_summary = example.get("feedback_summary") if isinstance(example.get("feedback_summary"), Mapping) else {}
        if int(feedback_summary.get("contradicted_count", 0) or 0) > 0:
            return "contradicted"
        if int(feedback_summary.get("verified_count", 0) or 0) > 0:
            return "verified"
        verification = example.get("verification") if isinstance(example.get("verification"), Mapping) else {}
        status = self._normalize_action_text(verification.get("status", example.get("status", "unverified"))).lower()
        if status in {"verified", "contradicted", "failed"}:
            return status
        if bool(verification.get("contradiction", False)):
            return "contradicted"
        if bool(verification.get("success", False)):
            return "verified"
        return "unverified"

    def _replay_dataset_output_or_none(self, value: Any) -> Any:
        safe = self._runtime_trace_export_safe_value(value)
        if safe in ({}, [], "", None):
            return None
        return safe

    def _replay_dataset_item_from_trace_example(
        self,
        example: Mapping[str, Any],
        *,
        replay_candidate: Mapping[str, Any] | None,
        replay_sample_linkage: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        verification_label = self._replay_dataset_verification_label(example)
        runtime_status = self._normalize_action_text(example.get("status", "unknown")).lower() or "unknown"
        raw_provenance = self._normalize_action_text(example.get("provenance", "observed")).lower() or "observed"
        synthetic_provenance = raw_provenance in {"dreamed", "synthetic"}
        provenance_label = (
            raw_provenance
            if synthetic_provenance
            else verification_label
            if verification_label in {"verified", "contradicted"}
            else raw_provenance
        )
        corrected_output = self._replay_dataset_output_or_none(example.get("corrected_output"))
        actual_output = self._replay_dataset_output_or_none(example.get("actual_output"))
        prediction = self._replay_dataset_output_or_none(example.get("prediction"))
        failure = self._replay_dataset_output_or_none(example.get("failure"))

        chosen_output: Any = None
        chosen_source = ""
        if corrected_output is not None:
            chosen_output = corrected_output
            chosen_source = "corrected_output"
        elif verification_label == "verified" and actual_output is not None:
            chosen_output = actual_output
            chosen_source = "verified_actual_output"

        rejected_output: Any = None
        rejected_source = ""
        if verification_label == "contradicted":
            rejected_output = actual_output if actual_output is not None else prediction
            rejected_source = "contradicted_actual_output" if actual_output is not None else "contradicted_prediction"
        elif runtime_status == "failed" or failure is not None or verification_label == "failed":
            rejected_output = failure if failure is not None else actual_output if actual_output is not None else prediction
            rejected_source = "failed_runtime_output"

        has_positive = chosen_output is not None
        has_negative = rejected_output is not None
        if has_positive and has_negative:
            example_type = "dpo_preference_pair_preview"
        elif has_positive:
            example_type = "sft_example_preview"
        elif has_negative:
            example_type = "negative_only_preview_context"
        else:
            example_type = "excluded_preview_context"

        target_id = str(example.get("example_id", "") or "")
        context = example.get("context") if isinstance(example.get("context"), Mapping) else {}
        sft_example = (
            {
                "input": self._runtime_trace_export_safe_value(context),
                "output": chosen_output,
                "output_source": chosen_source,
                "eligible_source": chosen_source in {"corrected_output", "verified_actual_output"},
                "preview_only": True,
            }
            if has_positive
            else None
        )
        preference_pair = (
            {
                "chosen": chosen_output,
                "chosen_source": chosen_source,
                "rejected": rejected_output,
                "rejected_source": rejected_source,
                "preview_only": True,
            }
            if has_positive and has_negative
            else None
        )
        is_verified_fact = (
            verification_label == "verified"
            and not synthetic_provenance
            and provenance_label not in {"contradicted", "dreamed", "synthetic"}
        )
        item = {
            "schema_version": REPLAY_DATASET_SCHEMA_VERSION,
            "item_id": f"replay-dataset-{target_id or example.get('trace_id', uuid4())}",
            "training_role": REPLAY_DATASET_TRAINING_ROLE,
            "dataset_role": "replay_dataset_item_preview",
            "example_type": example_type,
            "target_type": "runtime_episode",
            "target_id": target_id,
            "trace_id": example.get("trace_id"),
            "endpoint": example.get("endpoint"),
            "operation": example.get("operation"),
            "timestamp": example.get("timestamp") or example.get("created_at"),
            "status": runtime_status,
            "verification_label": verification_label,
            "provenance_label": provenance_label,
            "is_verified_fact": is_verified_fact,
            "has_positive_example": has_positive,
            "has_negative_example": has_negative,
            "sft_example": sft_example,
            "preference_pair": preference_pair,
            "runtime_trace": self._runtime_trace_export_safe_value(dict(example)),
            "feedback": self._runtime_trace_export_safe_value(example.get("feedback", [])),
            "feedback_summary": self._runtime_trace_export_safe_value(example.get("feedback_summary", {})),
            "policy_context": self._runtime_trace_export_policy_decision_summary(example.get("policy_decision")),
            "replay_plan_context": self._runtime_trace_export_safe_value(dict(replay_candidate or {})),
            "replay_sample_linkage": self._runtime_trace_export_safe_value(
                dict(replay_sample_linkage)
                if isinstance(replay_sample_linkage, Mapping)
                else {
                    "selected": False,
                    "target_type": "runtime_episode",
                    "target_id": target_id,
                    "replay_sample_ids": [],
                    "execution_ids": [],
                    "modes": [],
                    "candidate_ids": [],
                    "latest": None,
                }
            ),
            "safety_flags": {
                "preview_only": True,
                "training_started": False,
                "sleep_started": False,
                "memory_verification_promoted": False,
                "feedback_posted": False,
                "digital_action_executed": False,
                "external_calls_made": False,
                "memory_mutated": False,
                "not_promoted": True,
                "eligible_for_training": False,
            },
        }
        if example_type == "excluded_preview_context":
            item["excluded_reason"] = "no_verified_or_corrected_positive_output_and_no_failed_or_contradicted_rejected_output"
        elif example_type == "negative_only_preview_context":
            item["excluded_reason"] = "negative_signal_without_verified_or_corrected_chosen_output"
        return cast(dict[str, Any], self._runtime_trace_export_safe_value(item))

    def _replay_plan_summary(self, replay_plan: Any) -> dict[str, Any]:
        data = dict(replay_plan) if isinstance(replay_plan, Mapping) else {}
        if not data:
            return {}
        candidates = data.get("candidates")
        reason_codes = data.get("plan_reason_codes")
        if not isinstance(reason_codes, Sequence) or isinstance(reason_codes, (str, bytes)):
            reason_counter: Counter[str] = Counter()
            if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
                for candidate in candidates:
                    if not isinstance(candidate, Mapping):
                        continue
                    for code in candidate.get("reason_codes", []):
                        text = str(code).strip()
                        if text:
                            reason_counter[text] += 1
            reason_codes = list(reason_counter.keys())
        top_candidate: Mapping[str, Any] = {}
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)) and candidates:
            first = candidates[0]
            if isinstance(first, Mapping):
                top_candidate = first
        summary = {
            "schema_version": data.get("schema_version"),
            "generated_at": data.get("generated_at"),
            "advisory": data.get("advisory", True),
            "executable": data.get("executable", False),
            "endpoint": data.get("endpoint", "/terminus/replay-plan"),
            "limit": data.get("limit"),
            "count": data.get("count"),
            "priority_rules_version": data.get("priority_rules_version"),
            "plan_reason_codes": [str(item) for item in list(reason_codes or [])[:12]],
            "top_candidate": {
                "candidate_id": top_candidate.get("candidate_id"),
                "rank": top_candidate.get("rank"),
                "target_type": top_candidate.get("target_type"),
                "target_id": top_candidate.get("target_id"),
                "operation": top_candidate.get("operation"),
                "priority_score": top_candidate.get("priority_score"),
                "reason_codes": list(top_candidate.get("reason_codes") or [])[:8],
                "suggested_consolidation_action": top_candidate.get("suggested_consolidation_action"),
                "suggested_endpoint": top_candidate.get("suggested_endpoint"),
            }
            if top_candidate
            else None,
        }
        source_window = data.get("source_window")
        if isinstance(source_window, Mapping):
            summary["source_window"] = {
                "surface": source_window.get("surface"),
                "window_policy": source_window.get("window_policy"),
                "runs_live_tick": bool(source_window.get("runs_live_tick", False)),
                "selection_criteria": list(source_window.get("selection_criteria") or [])[:8],
                "source_limits": dict(source_window.get("source_limits") or {})
                if isinstance(source_window.get("source_limits"), Mapping)
                else {},
                "source_counts": dict(source_window.get("source_counts") or {})
                if isinstance(source_window.get("source_counts"), Mapping)
                else {},
                "window_counts": dict(source_window.get("window_counts") or {})
                if isinstance(source_window.get("window_counts"), Mapping)
                else {},
                "truncated_source_counts": dict(source_window.get("truncated_source_counts") or {})
                if isinstance(source_window.get("truncated_source_counts"), Mapping)
                else {},
                "feedback_index_entry_count": source_window.get("feedback_index_entry_count"),
                "feedback_index_target_count": source_window.get("feedback_index_target_count"),
                "candidate_count_before_rank": source_window.get("candidate_count_before_rank"),
                "candidate_count_returned": source_window.get("candidate_count_returned"),
                "device_placement": dict(source_window.get("device_placement") or {})
                if isinstance(source_window.get("device_placement"), Mapping)
                else {},
            }
        return cast(dict[str, Any], self._runtime_trace_export_safe_value(summary))

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
            for key, item in list(value.items())[:_RUNTIME_TRACE_EXPORT_MAX_MAPPING_ITEMS]:
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
                for item in list(value)[:count]
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
        for episode in list(self._interaction_pipeline.runtime_episode_traces()):
            if isinstance(episode, Mapping):
                targets.append(("runtime_episode", str(episode.get("episode_id", "") or ""), episode.get("feedback", [])))
        for action in list(self._action_history):
            if isinstance(action, Mapping):
                targets.append(("action", str(action.get("action_id", "") or ""), action.get("feedback", [])))
        return self._runtime_feedback_summary_from_targets(targets)

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

