"""Interaction Pipeline seam for query turns.

This module owns the first constructor-injected query seam extracted from the
Service Manager. It handles query turn orchestration, query-specific runtime
episode trace construction, and the query actual-output / verification payload
behavior while delegating collaborator-specific work back into injected
callbacks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
import time
from typing import Any, Callable, Mapping
from uuid import uuid4

from hecsn.semantics.grounding_text import salient_query_terms


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def build_query_runtime_actual_output(result: Mapping[str, Any]) -> dict[str, Any]:
    query_summary = _mapping_or_empty(result.get("query_summary"))
    gap_plan = _mapping_or_empty(result.get("gap_plan"))
    concept_summary = _mapping_or_empty(result.get("concept_summary"))
    memory_matches = list(query_summary.get("memory_matches") or [])
    memory_episodes = list(query_summary.get("memory_episodes") or [])
    return {
        "summary": f"Retrieved {len(memory_matches)} memory matches and {len(memory_episodes)} memory episodes.",
        "query_text": str(query_summary.get("query_text", "")),
        "winner_column": query_summary.get("winner_column"),
        "reconstruction_error": query_summary.get("reconstruction_error"),
        "top_candidate_count": int(len(list(query_summary.get("top_candidates") or []))),
        "memory_match_count": int(len(memory_matches)),
        "memory_episode_count": int(len(memory_episodes)),
        "gap_plan": {
            "planner_mode": gap_plan.get("planner_mode"),
            "grounded_fraction": float(gap_plan.get("grounded_fraction", 0.0) or 0.0),
            "unsupported_terms": list(gap_plan.get("unsupported_terms") or []),
            "retrieval_queries": list(gap_plan.get("retrieval_queries") or [])[:3],
        },
        "concept_summary": {
            "concept_count": concept_summary.get("concept_count"),
            "observations": concept_summary.get("observations"),
            "top_concepts": list(concept_summary.get("top_concepts") or [])[:3],
        },
    }


def build_query_runtime_verification(result: Mapping[str, Any]) -> dict[str, Any]:
    actual = build_query_runtime_actual_output(result)
    gap_plan = actual.get("gap_plan") if isinstance(actual.get("gap_plan"), Mapping) else {}
    confidence = max(
        0.20 if int(actual.get("memory_match_count", 0) or 0) > 0 else 0.05,
        min(1.0, float(gap_plan.get("grounded_fraction", 0.0) or 0.0)),
    )
    return {
        "status": "verified",
        "success": True,
        "confidence": confidence,
        "contradiction": False,
        "summary": str(actual.get("summary", "")),
    }


class InteractionPipeline:
    """Constructor-injected query seam for interaction turns."""

    def __init__(
        self,
        *,
        lock: RLock,
        build_query_result_fn: Callable[..., dict[str, Any]],
        observe_concepts_fn: Callable[..., dict[str, Any]],
        plan_gaps_fn: Callable[..., dict[str, Any]],
        apply_delayed_query_consequence_fn: Callable[..., dict[str, Any]],
        record_recent_query_gap_fn: Callable[..., None],
        runtime_episode_payload_fn: Callable[..., dict[str, Any]],
        persist_trace_fn: Callable[[dict[str, Any]], Path],
        append_runtime_episode_trace_fn: Callable[[Mapping[str, Any]], dict[str, Any]],
        service_state_snapshot_fn: Callable[..., dict[str, Any]],
    ) -> None:
        self._lock = lock
        self._build_query_result_fn = build_query_result_fn
        self._observe_concepts_fn = observe_concepts_fn
        self._plan_gaps_fn = plan_gaps_fn
        self._apply_delayed_query_consequence_fn = apply_delayed_query_consequence_fn
        self._record_recent_query_gap_fn = record_recent_query_gap_fn
        self._runtime_episode_payload_fn = runtime_episode_payload_fn
        self._persist_trace_fn = persist_trace_fn
        self._append_runtime_episode_trace_fn = append_runtime_episode_trace_fn
        self._service_state_snapshot_fn = service_state_snapshot_fn

    def _query_runtime_actual_output(self, result: Mapping[str, Any]) -> dict[str, Any]:
        return build_query_runtime_actual_output(result)

    def _query_runtime_verification(self, result: Mapping[str, Any]) -> dict[str, Any]:
        return build_query_runtime_verification(result)

    @staticmethod
    def _build_request(
        *,
        query_text: str,
        context_text: str | None,
        top_k_candidates: int,
        top_k_memories: int,
        top_chars: int,
    ) -> dict[str, Any]:
        return {
            "query_text": query_text,
            "context_text": context_text,
            "top_k_candidates": int(top_k_candidates),
            "top_k_memories": int(top_k_memories),
            "top_chars": int(top_chars),
        }

    @staticmethod
    def _build_prediction(query_text: str) -> dict[str, Any]:
        return {
            "kind": "retrieval_prediction",
            "predicted_output": f"Query should produce memory evidence and a semantic gap plan for: {query_text}",
            "proposed_action": "build_query_result",
            "topics": salient_query_terms(query_text)[:8],
        }

    @staticmethod
    def _build_action(
        *,
        top_k_candidates: int,
        top_k_memories: int,
        top_chars: int,
    ) -> dict[str, Any]:
        return {
            "action_type": "query",
            "top_k_candidates": int(top_k_candidates),
            "top_k_memories": int(top_k_memories),
            "top_chars": int(top_chars),
        }

    @staticmethod
    def _build_trace(
        *,
        trace_id: str,
        created_at: str,
        request: Mapping[str, Any],
        runtime_episode: Mapping[str, Any],
        state_after: Mapping[str, Any],
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        trace = {
            "trace_id": trace_id,
            "created_at": created_at,
            "operation": "query",
            "request": dict(request),
            "runtime_episode": dict(runtime_episode),
            "state_after": dict(state_after),
        }
        if error is not None:
            trace["error"] = {"type": type(error).__name__, "message": str(error)}
        return trace

    def query(
        self,
        *,
        query_text: str,
        context_text: str | None = None,
        top_k_candidates: int = 5,
        top_k_memories: int = 5,
        top_chars: int = 6,
    ) -> dict[str, Any]:
        with self._lock:
            started_perf = time.perf_counter()
            created_at = datetime.now(timezone.utc).isoformat()
            trace_id = str(uuid4())
            request = self._build_request(
                query_text=query_text,
                context_text=context_text,
                top_k_candidates=top_k_candidates,
                top_k_memories=top_k_memories,
                top_chars=top_chars,
            )
            prediction = self._build_prediction(query_text)
            action = self._build_action(
                top_k_candidates=top_k_candidates,
                top_k_memories=top_k_memories,
                top_chars=top_chars,
            )
            try:
                result = self._build_query_result_fn(
                    query_text=query_text,
                    context_text=context_text,
                    top_k_candidates=top_k_candidates,
                    top_k_memories=top_k_memories,
                    top_chars=top_chars,
                )
                result["concept_summary"] = self._observe_concepts_fn(
                    query_text=query_text,
                    query_result=result,
                )
                result["gap_plan"] = self._plan_gaps_fn(
                    query_text=query_text,
                    query_result=result,
                )
                result["delayed_consequence"] = self._apply_delayed_query_consequence_fn(
                    query_result=result,
                )
                self._record_recent_query_gap_fn(
                    query_text=query_text,
                    gap_plan=result["gap_plan"],
                    source="query",
                )
                actual_output = self._query_runtime_actual_output(result)
                verification = self._query_runtime_verification(result)
                episode = self._runtime_episode_payload_fn(
                    operation="query",
                    request=request,
                    prediction=prediction,
                    action=action,
                    actual_output=actual_output,
                    verification=verification,
                    started_perf=started_perf,
                    created_at=created_at,
                    trace_id=trace_id,
                )
                state_after = self._service_state_snapshot_fn(include_replay_dataset_summary=False)
                trace = self._build_trace(
                    trace_id=trace_id,
                    created_at=created_at,
                    request=request,
                    runtime_episode=episode,
                    state_after=state_after,
                )
                trace_path = self._persist_trace_fn(trace)
                episode["trace_path"] = str(trace_path)
                episode = self._append_runtime_episode_trace_fn(episode)
                result["service_state"] = state_after
                result["runtime_episode"] = episode
                return result
            except Exception as exc:
                episode = self._runtime_episode_payload_fn(
                    operation="query",
                    request=request,
                    prediction=prediction,
                    action=action,
                    actual_output=None,
                    verification=None,
                    started_perf=started_perf,
                    created_at=created_at,
                    trace_id=trace_id,
                    error=exc,
                )
                trace = self._build_trace(
                    trace_id=trace_id,
                    created_at=created_at,
                    request=request,
                    runtime_episode=episode,
                    state_after=self._service_state_snapshot_fn(include_replay_dataset_summary=False),
                    error=exc,
                )
                trace_path = self._persist_trace_fn(trace)
                episode["trace_path"] = str(trace_path)
                self._append_runtime_episode_trace_fn(episode)
                raise
