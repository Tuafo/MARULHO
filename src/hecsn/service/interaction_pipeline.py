"""Interaction Pipeline seam for query, feed, and respond turns.

This module owns the constructor-injected operator-turn seam extracted from
the Service Manager. It handles query, feed, and respond turn orchestration,
turn-specific runtime episode trace construction, and the query/feed/respond
actual-output / verification payload behavior while delegating collaborator-
specific work back into injected callbacks.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
import time
from typing import Any, Callable, Mapping, Sequence, TypeVar
from uuid import uuid4

from hecsn.service.history_store import read_history_record
from hecsn.service.living_loop_records import RuntimeEpisodeTrace
from hecsn.semantics.grounding_text import salient_query_terms


DEFAULT_FEED_CONCEPT_OBSERVATION_INTERVAL = 8
DEFAULT_RECENT_QUERY_GAP_HISTORY = 8
DEFAULT_RUNTIME_EPISODE_TRACE_HISTORY = 64
REQUEST_FEED_ENCODING_MODE = "lexical_rolling_segments"
_RespondCallbackT = TypeVar("_RespondCallbackT")


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
    gap_plan = _mapping_or_empty(actual.get("gap_plan"))
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


def build_feed_runtime_actual_output(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "summary": f"Processed {int(summary.get('tokens_processed', 0) or 0)} feed tokens.",
        "tokens_processed": int(summary.get("tokens_processed", 0) or 0),
        "token_count": int(summary.get("token_count", 0) or 0),
        "last_winner": summary.get("last_winner"),
        "last_recon_error": summary.get("last_recon_error"),
        "memory_buffer_size": int(summary.get("memory_buffer_size", 0) or 0),
        "feed_encoding_mode": summary.get("feed_encoding_mode"),
        "concept_observation_mode": summary.get("concept_observation_mode"),
        "concept_observations": int(summary.get("concept_observations", 0) or 0),
    }


def build_feed_runtime_verification(summary: Mapping[str, Any]) -> dict[str, Any]:
    actual = build_feed_runtime_actual_output(summary)
    tokens_processed = int(actual.get("tokens_processed", 0) or 0)
    verified = bool(tokens_processed > 0)
    return {
        "status": "verified" if verified else "unverified",
        "success": verified,
        "confidence": 1.0 if verified else 0.0,
        "contradiction": False,
        "summary": str(actual.get("summary", "")),
    }


def _normalize_action_text(value: Any) -> str:
    return " ".join(str(value).split()).strip()


def build_respond_runtime_actual_output(
    *,
    response: Mapping[str, Any],
    action_assist: Mapping[str, Any] | None,
    outcome_score: float,
) -> dict[str, Any]:
    actual = {
        "summary": _normalize_action_text(response.get("response_text", "")),
        "response_text": _normalize_action_text(response.get("response_text", "")),
        "response_mode": _normalize_action_text(response.get("response_mode", "")),
        "support_score": float(response.get("support_score", 0.0) or 0.0),
        "evidence_coverage": float(response.get("evidence_coverage", 0.0) or 0.0),
        "selected_evidence_count": int(len(list(response.get("selected_evidence") or []))),
        "unsupported_terms": list(response.get("unsupported_terms") or []),
        "outcome_score": float(outcome_score),
    }
    if isinstance(action_assist, Mapping):
        record_value = action_assist.get("result")
        record = record_value if isinstance(record_value, Mapping) else {}
        actual["action_assist"] = {
            "triggered": bool(action_assist.get("triggered", False)),
            "executed": bool(action_assist.get("executed", False)),
            "reused_recent_action": bool(action_assist.get("reused_recent_action", False)),
            "reason": _normalize_action_text(action_assist.get("reason", "")),
            "used_in_response": bool(action_assist.get("used_in_response", False)),
            "action_type": _normalize_action_text(record.get("action_type", "")),
            "action_id": _normalize_action_text(record.get("action_id", "")),
        }
    return actual


def build_respond_runtime_verification(
    *,
    response: Mapping[str, Any],
    action_assist: Mapping[str, Any] | None,
    outcome_score: float,
) -> dict[str, Any]:
    action_verification: Mapping[str, Any] = {}
    if isinstance(action_assist, Mapping):
        record_value = action_assist.get("result")
        record = record_value if isinstance(record_value, Mapping) else {}
        verification_value = record.get("verification")
        action_verification = verification_value if isinstance(verification_value, Mapping) else {}
    if bool(action_verification.get("contradiction", False)):
        return {
            "status": "contradicted",
            "success": False,
            "confidence": float(action_verification.get("confidence", 0.0) or 0.0),
            "contradiction": True,
            "summary": _normalize_action_text(
                action_verification.get("summary", "Action verification contradicted the prediction.")
            ),
        }
    if bool(action_verification.get("success", False)):
        return {
            "status": "verified",
            "success": True,
            "confidence": max(float(action_verification.get("confidence", 0.0) or 0.0), float(outcome_score)),
            "contradiction": False,
            "summary": _normalize_action_text(
                action_verification.get("summary", "Action verification supplied grounded evidence.")
            ),
        }
    selected_count = int(len(list(response.get("selected_evidence") or [])))
    response_mode = _normalize_action_text(response.get("response_mode", "")).lower()
    if response_mode == "insufficient_evidence" or selected_count <= 0:
        return {
            "status": "unverified",
            "success": False,
            "confidence": float(outcome_score),
            "contradiction": False,
            "summary": "Response did not have enough grounded evidence for verification.",
        }
    return {
        "status": "verified" if outcome_score >= 0.5 else "unverified",
        "success": bool(outcome_score >= 0.5),
        "confidence": float(outcome_score),
        "contradiction": False,
        "summary": f"Response grounded by {selected_count} selected evidence item(s).",
    }


class InteractionPipeline:
    """Constructor-injected query, feed, and respond seam for interaction turns."""

    def __init__(
        self,
        *,
        lock: RLock,
        trainer: Any,
        encoder: Any,
        build_query_result_fn: Callable[..., dict[str, Any]],
        observe_concepts_fn: Callable[..., dict[str, Any]],
        plan_gaps_fn: Callable[..., dict[str, Any]],
        apply_delayed_query_consequence_fn: Callable[..., dict[str, Any]],
        observe_runtime_concepts_fn: Callable[..., dict[str, Any] | None],
        runtime_state_mark_mutated_fn: Callable[[], None],
        runtime_state_mutation_summary_fn: Callable[[], dict[str, Any]],
        runtime_episode_payload_fn: Callable[..., dict[str, Any]],
        persist_trace_fn: Callable[[dict[str, Any]], Path],
        service_state_snapshot_fn: Callable[..., dict[str, Any]],
        build_response_fn: Callable[..., dict[str, Any]] | None = None,
        maybe_auto_action_assist_fn: Callable[..., dict[str, Any] | None] | None = None,
        response_grounded_outcome_score_fn: Callable[..., float] | None = None,
        apply_background_source_response_provenance_fn: Callable[..., bool] | None = None,
        apply_background_source_outcome_calibration_fn: Callable[..., None] | None = None,
        apply_provider_response_outcome_calibration_fn: Callable[..., bool] | None = None,
        learn_from_turn_fn: Callable[..., dict[str, Any] | None] | None = None,
        record_response_consequence_candidate_fn: Callable[..., dict[str, Any] | None] | None = None,
        recent_query_gaps: Sequence[Mapping[str, Any]] | None = None,
        runtime_episode_traces: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self._lock = lock
        self._trainer = trainer
        self._encoder = encoder
        self._build_query_result_fn = build_query_result_fn
        self._observe_concepts_fn = observe_concepts_fn
        self._plan_gaps_fn = plan_gaps_fn
        self._apply_delayed_query_consequence_fn = apply_delayed_query_consequence_fn
        self._observe_runtime_concepts_fn = observe_runtime_concepts_fn
        self._runtime_state_mark_mutated_fn = runtime_state_mark_mutated_fn
        self._runtime_state_mutation_summary_fn = runtime_state_mutation_summary_fn
        self._runtime_episode_payload_fn = runtime_episode_payload_fn
        self._persist_trace_fn = persist_trace_fn
        self._service_state_snapshot_fn = service_state_snapshot_fn
        self._build_response_fn = build_response_fn
        self._maybe_auto_action_assist_fn = maybe_auto_action_assist_fn
        self._response_grounded_outcome_score_fn = response_grounded_outcome_score_fn
        self._apply_background_source_response_provenance_fn = apply_background_source_response_provenance_fn
        self._apply_background_source_outcome_calibration_fn = apply_background_source_outcome_calibration_fn
        self._apply_provider_response_outcome_calibration_fn = apply_provider_response_outcome_calibration_fn
        self._learn_from_turn_fn = learn_from_turn_fn
        self._record_response_consequence_candidate_fn = record_response_consequence_candidate_fn
        self._skip_next_autonomy_for_grounded_query = False
        self._recent_query_gaps: deque[dict[str, Any]] = deque(maxlen=DEFAULT_RECENT_QUERY_GAP_HISTORY)
        self._runtime_episode_traces: deque[dict[str, Any]] = deque(maxlen=DEFAULT_RUNTIME_EPISODE_TRACE_HISTORY)
        self._replace_recent_query_gaps_locked(recent_query_gaps)
        self._replace_runtime_episode_traces_locked(runtime_episode_traces)

    @property
    def recent_query_gap_history(self) -> deque[dict[str, Any]]:
        return self._recent_query_gaps

    @property
    def runtime_episode_trace_history(self) -> deque[dict[str, Any]]:
        return self._runtime_episode_traces

    @staticmethod
    def _normalize_recent_query_gap(item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None
        query_text = " ".join(str(item.get("query_text", "")).split()).strip()
        if not query_text:
            return None
        unsupported_terms = [
            str(term).strip().lower()
            for term in list(item.get("unsupported_terms") or [])
            if str(term).strip()
        ]
        gap_terms: list[dict[str, Any]] = []
        for raw_gap in list(item.get("gap_terms") or []):
            if not isinstance(raw_gap, dict):
                continue
            term = str(raw_gap.get("term", "")).strip().lower()
            if not term:
                continue
            gap_terms.append(
                {
                    "term": term,
                    "weight": float(raw_gap.get("weight", 0.0)),
                }
            )
        retrieval_queries = [
            " ".join(str(value).split()).strip()
            for value in list(item.get("retrieval_queries") or [])
            if " ".join(str(value).split()).strip()
        ]
        follow_up_questions = [
            " ".join(str(value).split()).strip()
            for value in list(item.get("follow_up_questions") or [])
            if " ".join(str(value).split()).strip()
        ]
        weak_concepts: list[dict[str, Any]] = []
        for raw_concept in list(item.get("weak_concepts") or []):
            if not isinstance(raw_concept, dict):
                continue
            label = " ".join(str(raw_concept.get("label", "")).split()).strip()
            top_terms = [
                " ".join(str(value).split()).strip().lower()
                for value in list(raw_concept.get("top_terms") or [])
                if " ".join(str(value).split()).strip()
            ]
            if not label and not top_terms:
                continue
            weak_concepts.append(
                {
                    "label": label,
                    "weakness": float(raw_concept.get("weakness", 0.0)),
                    "uncertainty": float(raw_concept.get("uncertainty", 0.0)),
                    "drift": float(raw_concept.get("drift", 0.0)),
                    "top_terms": top_terms[:4],
                    "match_count": max(0, int(raw_concept.get("match_count", 0))),
                }
            )
        return {
            "recorded_at": str(item.get("recorded_at") or datetime.now(timezone.utc).isoformat()),
            "source": str(item.get("source") or "query"),
            "query_text": query_text,
            "unsupported_terms": unsupported_terms,
            "gap_terms": gap_terms,
            "retrieval_queries": retrieval_queries[:4],
            "follow_up_questions": follow_up_questions[:4],
            "weak_concepts": weak_concepts[:4],
            "grounded_fraction": float(item.get("grounded_fraction", 0.0)),
        }

    @staticmethod
    def _normalize_runtime_episode_trace(item: Any) -> dict[str, Any] | None:
        if not isinstance(item, Mapping):
            return None
        try:
            return RuntimeEpisodeTrace.from_payload(item).to_payload()
        except Exception:
            return None

    def load_interaction_state(
        self,
        *,
        recent_query_gaps: Sequence[Mapping[str, Any]] | None = None,
        runtime_episode_traces: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        with self._lock:
            self._replace_recent_query_gaps_locked(recent_query_gaps)
            self._replace_runtime_episode_traces_locked(runtime_episode_traces)

    def _replace_recent_query_gaps_locked(
        self,
        items: Sequence[Mapping[str, Any]] | None,
    ) -> None:
        self._recent_query_gaps.clear()
        for raw_item in list(items or []):
            normalized = self._normalize_recent_query_gap(raw_item)
            if normalized is not None:
                self._recent_query_gaps.append(normalized)

    def _replace_runtime_episode_traces_locked(
        self,
        items: Sequence[Mapping[str, Any]] | None,
    ) -> None:
        self._runtime_episode_traces.clear()
        for raw_item in list(items or []):
            normalized = self._normalize_runtime_episode_trace(raw_item)
            if normalized is not None:
                self._runtime_episode_traces.append(normalized)

    def recent_query_gaps(self) -> list[dict[str, Any]]:
        with self._lock:
            return [deepcopy(item) for item in list(self._recent_query_gaps)]

    def record_recent_query_gap(
        self,
        *,
        query_text: str,
        gap_plan: Mapping[str, Any],
        source: str,
    ) -> None:
        normalized_query = " ".join(str(query_text).split()).strip()
        if not normalized_query:
            return
        existing = [
            item
            for item in list(self._recent_query_gaps)
            if str(item.get("query_text", "")).lower() != normalized_query.lower()
        ]
        self._recent_query_gaps.clear()
        self._recent_query_gaps.extend(existing)
        grounded_fraction = float(gap_plan.get("grounded_fraction", 0.0))
        query_deficit = bool(gap_plan.get("unsupported_terms")) or grounded_fraction < 0.999
        self._skip_next_autonomy_for_grounded_query = not query_deficit
        meaningful = bool(query_deficit and (gap_plan.get("unsupported_terms") or gap_plan.get("gap_terms") or gap_plan.get("weak_concepts")))
        if not meaningful:
            return
        normalized = self._normalize_recent_query_gap(
            {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "query_text": normalized_query,
                "unsupported_terms": list(gap_plan.get("unsupported_terms") or []),
                "gap_terms": list(gap_plan.get("gap_terms") or []),
                "retrieval_queries": list(gap_plan.get("retrieval_queries") or []),
                "follow_up_questions": list(gap_plan.get("follow_up_questions") or []),
                "weak_concepts": list(gap_plan.get("weak_concepts") or []),
                "grounded_fraction": float(gap_plan.get("grounded_fraction", 0.0)),
            }
        )
        if normalized is not None:
            self._recent_query_gaps.appendleft(normalized)

    def consume_skip_next_autonomy_for_grounded_query(self) -> bool:
        with self._lock:
            skip = bool(self._skip_next_autonomy_for_grounded_query)
            self._skip_next_autonomy_for_grounded_query = False
            return skip

    def runtime_episode_traces(self) -> list[dict[str, Any]]:
        with self._lock:
            return [deepcopy(item) for item in list(self._runtime_episode_traces)]

    def runtime_episode_trace(self, episode_id: str) -> dict[str, Any] | None:
        with self._lock:
            return read_history_record(self._runtime_episode_traces, record_id=episode_id, id_field="episode_id")

    def append_runtime_episode_trace(self, episode: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            normalized = self._normalize_runtime_episode_trace(episode)
            if normalized is None:
                normalized = dict(episode)
            episode_id = str(normalized.get("episode_id", ""))
            if episode_id:
                existing = [
                    item for item in list(self._runtime_episode_traces) if str(item.get("episode_id", "")) != episode_id
                ]
                self._runtime_episode_traces.clear()
                self._runtime_episode_traces.extend(existing)
            self._runtime_episode_traces.appendleft(deepcopy(normalized))
            return deepcopy(normalized)

    def replace_runtime_episode_trace(self, episode_id: str, episode: Mapping[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            replaced: dict[str, Any] | None = None
            updated: deque[dict[str, Any]] = deque(maxlen=self._runtime_episode_traces.maxlen)
            for item in list(self._runtime_episode_traces):
                if str(item.get("episode_id", "")) == str(episode_id):
                    normalized = self._normalize_runtime_episode_trace(episode)
                    replaced = deepcopy(normalized if normalized is not None else dict(episode))
                    if normalized is not None:
                        updated.append(deepcopy(normalized))
                    else:
                        updated.append(deepcopy(dict(episode)))
                else:
                    updated.append(deepcopy(item))
            if replaced is None:
                return None
            self._runtime_episode_traces.clear()
            self._runtime_episode_traces.extend(updated)
            return replaced

    def interaction_state_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            return {
                "recent_query_gaps": [deepcopy(item) for item in list(self._recent_query_gaps)],
                "runtime_episode_traces": [deepcopy(item) for item in list(self._runtime_episode_traces)],
            }

    def _query_runtime_actual_output(self, result: Mapping[str, Any]) -> dict[str, Any]:
        return build_query_runtime_actual_output(result)

    def _query_runtime_verification(self, result: Mapping[str, Any]) -> dict[str, Any]:
        return build_query_runtime_verification(result)

    def _feed_runtime_actual_output(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        return build_feed_runtime_actual_output(summary)

    def _feed_runtime_verification(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        return build_feed_runtime_verification(summary)

    def _enrich_query_result(
        self,
        *,
        query_text: str,
        context_text: str | None,
        top_k_candidates: int,
        top_k_memories: int,
        top_chars: int,
        gap_source: str,
    ) -> dict[str, Any]:
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
        self.record_recent_query_gap(
            query_text=query_text,
            gap_plan=result["gap_plan"],
            source=gap_source,
        )
        return result

    def _build_runtime_episode(
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
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        return self._runtime_episode_payload_fn(
            operation=operation,
            request=request,
            prediction=prediction,
            action=action,
            actual_output=actual_output,
            verification=verification,
            started_perf=started_perf,
            created_at=created_at,
            trace_id=trace_id,
            error=error,
        )

    def _finalize_trace(
        self,
        *,
        operation: str,
        trace_id: str,
        created_at: str,
        request: Mapping[str, Any],
        episode: Mapping[str, Any],
        state_after: Mapping[str, Any],
        state_before: Mapping[str, Any] | None = None,
        error: BaseException | None = None,
        extra_trace_fields: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        trace = self._build_trace(
            trace_id=trace_id,
            created_at=created_at,
            operation=operation,
            request=request,
            runtime_episode=episode,
            state_after=state_after,
            state_before=state_before,
            error=error,
        )
        if extra_trace_fields is not None:
            trace.update(dict(extra_trace_fields))
        trace_path = self._persist_trace_fn(trace)
        finalized_episode = dict(episode)
        finalized_episode["trace_path"] = str(trace_path)
        return self.append_runtime_episode_trace(finalized_episode)

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
        operation: str,
        request: Mapping[str, Any],
        runtime_episode: Mapping[str, Any],
        state_after: Mapping[str, Any],
        state_before: Mapping[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        trace = {
            "trace_id": trace_id,
            "created_at": created_at,
            "operation": operation,
            "request": dict(request),
            "runtime_episode": dict(runtime_episode),
            "state_after": dict(state_after),
        }
        if state_before is not None:
            trace["state_before"] = dict(state_before)
        if error is not None:
            trace["error"] = {"type": type(error).__name__, "message": str(error)}
        return trace

    @staticmethod
    def _build_feed_request(*, text: str) -> dict[str, Any]:
        return {
            "text_length": int(len(text)),
            "text_preview": text[:120],
        }

    @staticmethod
    def _build_feed_prediction(text: str) -> dict[str, Any]:
        return {
            "kind": "feed_prediction",
            "predicted_output": "Feed text should be encoded into runtime memory and concept observations.",
            "proposed_action": "feed_text",
            "topics": salient_query_terms(text)[:8],
        }

    @staticmethod
    def _build_feed_action(*, text: str) -> dict[str, Any]:
        return {
            "action_type": "feed",
            "text_length": int(len(text)),
        }

    def _feed_text_for_request_locked(
        self,
        text: str,
        *,
        allow_sleep_maintenance: bool,
        concept_observation_interval: int = DEFAULT_FEED_CONCEPT_OBSERVATION_INTERVAL,
    ) -> dict[str, Any]:
        self._trainer.encoder = self._encoder
        last_metrics: dict[str, Any] | None = None
        tokens = 0
        concept_observations = 0
        pending_concept_observation: tuple[str, dict[str, Any]] | None = None
        observation_interval = max(1, int(concept_observation_interval))
        sleep_maintenance_deferred = 0
        pattern_iter = self._encoder.iter_segment_patterns(
            text,
            self._trainer.config.window_size,
            learn=False,
            use_learned_boundaries=False,
        )
        for raw_window, pattern in pattern_iter:
            raw_window_text = str(raw_window)
            last_metrics = self._trainer.train_step(
                pattern,
                raw_window=raw_window_text,
                allow_sleep_maintenance=allow_sleep_maintenance,
            )
            metrics = dict(last_metrics or {})
            sleep_maintenance_deferred += int(metrics.get("sleep_maintenance_deferred", 0) or 0)
            pending_concept_observation = (raw_window_text, metrics)
            if tokens == 0 or (tokens + 1) % observation_interval == 0:
                self._observe_runtime_concepts_fn(raw_window=raw_window_text, metrics=metrics)
                concept_observations += 1
                pending_concept_observation = None
            tokens += 1

        if pending_concept_observation is not None:
            raw_window, metrics = pending_concept_observation
            self._observe_runtime_concepts_fn(raw_window=raw_window, metrics=metrics)
            concept_observations += 1

        return {
            "tokens_processed": int(tokens),
            "token_count": int(self._trainer.token_count),
            "last_winner": None if last_metrics is None else int(last_metrics["winner"]),
            "last_recon_error": None if last_metrics is None else float(last_metrics["recon_error"]),
            "memory_buffer_size": int(len(self._trainer.model.memory_store.slow_buffer)),
            "feed_encoding_mode": REQUEST_FEED_ENCODING_MODE,
            "concept_observation_mode": "sampled",
            "concept_observation_interval": int(observation_interval),
            "concept_observations": int(concept_observations),
            "sleep_maintenance_allowed": bool(allow_sleep_maintenance),
            "sleep_maintenance_deferred": int(sleep_maintenance_deferred),
        }

    def feed(
        self,
        *,
        text: str,
    ) -> dict[str, Any]:
        with self._lock:
            started_perf = time.perf_counter()
            created_at = datetime.now(timezone.utc).isoformat()
            trace_id = str(uuid4())
            request = self._build_feed_request(text=text)
            prediction = self._build_feed_prediction(text)
            action = self._build_feed_action(text=text)
            try:
                summary = self._feed_text_for_request_locked(
                    text,
                    allow_sleep_maintenance=False,
                    concept_observation_interval=DEFAULT_FEED_CONCEPT_OBSERVATION_INTERVAL,
                )
                self._runtime_state_mark_mutated_fn()
                actual_output = self._feed_runtime_actual_output(summary)
                verification = self._feed_runtime_verification(summary)
                episode = self._build_runtime_episode(
                    operation="feed",
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
                episode = self._finalize_trace(
                    operation="feed",
                    trace_id=trace_id,
                    created_at=created_at,
                    request=request,
                    episode=episode,
                    state_after=state_after,
                )
                return {
                    "feed_summary": summary,
                    "runtime_episode": episode,
                    **self._runtime_state_mutation_summary_fn(),
                }
            except Exception as exc:
                episode = self._build_runtime_episode(
                    operation="feed",
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
                self._finalize_trace(
                    operation="feed",
                    trace_id=trace_id,
                    created_at=created_at,
                    request=request,
                    episode=episode,
                    state_after=self._service_state_snapshot_fn(include_replay_dataset_summary=False),
                    error=exc,
                )
                raise

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
                result = self._enrich_query_result(
                    query_text=query_text,
                    context_text=context_text,
                    top_k_candidates=top_k_candidates,
                    top_k_memories=top_k_memories,
                    top_chars=top_chars,
                    gap_source="query",
                )
                actual_output = self._query_runtime_actual_output(result)
                verification = self._query_runtime_verification(result)
                episode = self._build_runtime_episode(
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
                episode = self._finalize_trace(
                    operation="query",
                    trace_id=trace_id,
                    created_at=created_at,
                    request=request,
                    episode=episode,
                    state_after=state_after,
                )
                result["service_state"] = state_after
                result["runtime_episode"] = episode
                return result
            except Exception as exc:
                episode = self._build_runtime_episode(
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
                self._finalize_trace(
                    operation="query",
                    trace_id=trace_id,
                    created_at=created_at,
                    request=request,
                    episode=episode,
                    state_after=self._service_state_snapshot_fn(include_replay_dataset_summary=False),
                    error=exc,
                )
                raise

    @staticmethod
    def _build_respond_request(
        *,
        query_text: str,
        context_text: str | None,
        top_k_candidates: int,
        top_k_memories: int,
        top_chars: int,
        max_evidence_items: int,
        learn_mode: str,
    ) -> dict[str, Any]:
        return {
            "query_text": query_text,
            "context_text": context_text,
            "top_k_candidates": int(top_k_candidates),
            "top_k_memories": int(top_k_memories),
            "top_chars": int(top_chars),
            "max_evidence_items": int(max_evidence_items),
            "learn_mode": learn_mode,
        }

    @staticmethod
    def _build_respond_action(*, max_evidence_items: int, learn_mode: str) -> dict[str, Any]:
        return {
            "action_type": "respond",
            "learn_mode": learn_mode,
            "max_evidence_items": int(max_evidence_items),
        }

    @staticmethod
    def _build_respond_prediction(
        query_text: str,
        proposed_response: Mapping[str, Any],
        *,
        action: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        proposed_answer = _normalize_action_text(proposed_response.get("response_text", ""))
        prediction = {
            "kind": "response_prediction",
            "predicted_output": proposed_answer or f"Respond should produce a grounded answer for: {query_text}",
            "proposed_answer": proposed_answer,
            "confidence": float(proposed_response.get("support_score", 0.0) or 0.0),
            "topics": salient_query_terms(query_text)[:8],
        }
        if isinstance(action, Mapping) and action.get("proposed_action"):
            prediction["proposed_action"] = action["proposed_action"]
        return prediction

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

    def _require_respond_callback(self, callback: _RespondCallbackT | None) -> _RespondCallbackT:
        if callback is None:
            raise RuntimeError("InteractionPipeline respond callbacks are not configured")
        return callback

    def _require_respond_callbacks(self) -> None:
        self._require_respond_callback(self._build_response_fn)
        self._require_respond_callback(self._maybe_auto_action_assist_fn)
        self._require_respond_callback(self._response_grounded_outcome_score_fn)
        self._require_respond_callback(self._apply_background_source_response_provenance_fn)
        self._require_respond_callback(self._apply_background_source_outcome_calibration_fn)
        self._require_respond_callback(self._apply_provider_response_outcome_calibration_fn)
        self._require_respond_callback(self._learn_from_turn_fn)
        self._require_respond_callback(self._record_response_consequence_candidate_fn)

    def _build_response_payload(
        self,
        *,
        query_text: str,
        query_result: Mapping[str, Any],
        max_evidence_items: int,
    ) -> dict[str, Any]:
        build_response_fn = self._build_response_fn
        if build_response_fn is None:
            raise RuntimeError("InteractionPipeline respond callbacks are not configured")
        response = build_response_fn(
            query_text=query_text,
            query_summary=query_result.get("query_summary") or {},
            concept_summary=query_result.get("concept_summary"),
            max_evidence_items=max_evidence_items,
        )
        if not isinstance(response, dict):
            raise TypeError("build_response_fn must return a dict")
        return response

    def _apply_action_assist(
        self,
        *,
        query_text: str,
        query_result: dict[str, Any],
        response: dict[str, Any],
        max_evidence_items: int,
    ) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
        maybe_auto_action_assist_fn = self._maybe_auto_action_assist_fn
        if maybe_auto_action_assist_fn is None:
            raise RuntimeError("InteractionPipeline respond callbacks are not configured")
        action_assist = maybe_auto_action_assist_fn(
            query_text=query_text,
            query_result=query_result,
            response=response,
        )
        if action_assist is not None and not isinstance(action_assist, Mapping):
            raise TypeError("maybe_auto_action_assist_fn must return a mapping or None")
        if action_assist is None:
            return response, None

        if int(action_assist.get("response_episode_count", 0) or 0) > 0:
            response = self._build_response_payload(
                query_text=query_text,
                query_result=query_result,
                max_evidence_items=max_evidence_items,
            )
            action_assist["used_in_response"] = True

        response_note = _normalize_action_text(action_assist.get("response_note", ""))
        if response_note:
            base_text = _normalize_action_text(response.get("response_text", ""))
            if response_note.strip() not in base_text:
                response["response_text"] = (base_text + response_note).strip()
                action_assist["used_in_response"] = True

        query_result["action_assist"] = deepcopy(action_assist)
        response["action_assist"] = deepcopy(action_assist)
        return response, action_assist

    @staticmethod
    def _apply_action_assist_to_action(
        action: dict[str, Any],
        action_assist: Mapping[str, Any] | None,
    ) -> None:
        if not isinstance(action_assist, Mapping):
            return
        record_value = action_assist.get("result")
        record = record_value if isinstance(record_value, Mapping) else {}
        action["action_assist"] = {
            "triggered": bool(action_assist.get("triggered", False)),
            "executed": bool(action_assist.get("executed", False)),
            "reused_recent_action": bool(action_assist.get("reused_recent_action", False)),
            "reason": _normalize_action_text(action_assist.get("reason", "")),
            "action_type": _normalize_action_text(record.get("action_type", "")),
            "action_id": _normalize_action_text(record.get("action_id", "")),
        }
        predicted_action = _normalize_action_text(record.get("predicted_outcome", ""))
        if predicted_action:
            action["proposed_action"] = predicted_action

    def respond(
        self,
        *,
        query_text: str,
        context_text: str | None = None,
        top_k_candidates: int = 5,
        top_k_memories: int = 5,
        top_chars: int = 6,
        max_evidence_items: int = 3,
        learn_mode: str = "user_and_selected_evidence",
    ) -> dict[str, Any]:
        self._require_respond_callbacks()

        with self._lock:
            started_perf = time.perf_counter()
            created_at = datetime.now(timezone.utc).isoformat()
            trace_id = str(uuid4())
            request = self._build_respond_request(
                query_text=query_text,
                context_text=context_text,
                top_k_candidates=top_k_candidates,
                top_k_memories=top_k_memories,
                top_chars=top_chars,
                max_evidence_items=max_evidence_items,
                learn_mode=learn_mode,
            )
            state_before = self._service_state_snapshot_fn(include_replay_dataset_summary=False)
            try:
                query_result = self._enrich_query_result(
                    query_text=query_text,
                    context_text=context_text,
                    top_k_candidates=top_k_candidates,
                    top_k_memories=top_k_memories,
                    top_chars=top_chars,
                    gap_source="respond",
                )
                response = self._build_response_payload(
                    query_text=query_text,
                    query_result=query_result,
                    max_evidence_items=max_evidence_items,
                )
                proposed_response = deepcopy(response)
                response, action_assist = self._apply_action_assist(
                    query_text=query_text,
                    query_result=query_result,
                    response=response,
                    max_evidence_items=max_evidence_items,
                )
                response_grounded_outcome_score_fn = self._require_respond_callback(
                    self._response_grounded_outcome_score_fn
                )
                apply_background_source_response_provenance_fn = self._require_respond_callback(
                    self._apply_background_source_response_provenance_fn
                )
                apply_background_source_outcome_calibration_fn = self._require_respond_callback(
                    self._apply_background_source_outcome_calibration_fn
                )
                apply_provider_response_outcome_calibration_fn = self._require_respond_callback(
                    self._apply_provider_response_outcome_calibration_fn
                )
                learn_from_turn_fn = self._require_respond_callback(self._learn_from_turn_fn)
                record_response_consequence_candidate_fn = self._require_respond_callback(
                    self._record_response_consequence_candidate_fn
                )
                response_outcome_score = response_grounded_outcome_score_fn(
                    query_result=query_result,
                    response=response,
                    action_assist=action_assist,
                )
                applied_background_provenance = bool(
                    apply_background_source_response_provenance_fn(
                        response=response,
                        outcome_score=response_outcome_score,
                    )
                )
                if not applied_background_provenance:
                    apply_background_source_outcome_calibration_fn(
                        query_text=query_text,
                        outcome_score=response_outcome_score,
                    )
                apply_provider_response_outcome_calibration_fn(
                    response=response,
                    outcome_score=response_outcome_score,
                )
                learning = learn_from_turn_fn(
                    query_text=query_text,
                    response=response,
                    learn_mode=learn_mode,
                )
                delayed_candidate = record_response_consequence_candidate_fn(
                    query_result=query_result,
                    response=response,
                    outcome_score=response_outcome_score,
                )
                if delayed_candidate is not None:
                    response["delayed_consequence_candidate"] = deepcopy(delayed_candidate)
                action = self._build_respond_action(
                    max_evidence_items=max_evidence_items,
                    learn_mode=learn_mode,
                )
                self._apply_action_assist_to_action(action, action_assist)
                prediction = self._build_respond_prediction(
                    query_text,
                    proposed_response,
                    action=action,
                )
                actual_output = self._respond_runtime_actual_output(
                    response=response,
                    action_assist=action_assist,
                    outcome_score=response_outcome_score,
                )
                verification = self._respond_runtime_verification(
                    response=response,
                    action_assist=action_assist,
                    outcome_score=response_outcome_score,
                )
                state_after = self._service_state_snapshot_fn(include_replay_dataset_summary=False)
                episode = self._build_runtime_episode(
                    operation="respond",
                    request=request,
                    prediction=prediction,
                    action=action,
                    actual_output=actual_output,
                    verification=verification,
                    started_perf=started_perf,
                    created_at=created_at,
                    trace_id=trace_id,
                )
                episode = self._finalize_trace(
                    operation="respond",
                    trace_id=trace_id,
                    created_at=created_at,
                    request=request,
                    episode=episode,
                    state_after=state_after,
                    state_before=state_before,
                    extra_trace_fields={
                        "query_result": query_result,
                        "response": response,
                        "learning": learning,
                    },
                )
                return {
                    "trace_id": trace_id,
                    "trace_path": str(episode["trace_path"]),
                    "created_at": created_at,
                    "query_result": query_result,
                    "response": response,
                    "learning": learning,
                    "runtime_episode": episode,
                    **self._runtime_state_mutation_summary_fn(),
                }
            except Exception as exc:
                prediction = {
                    "kind": "response_prediction",
                    "predicted_output": f"Respond should produce a grounded answer for: {query_text}",
                    "topics": salient_query_terms(query_text)[:8],
                }
                action = self._build_respond_action(
                    max_evidence_items=max_evidence_items,
                    learn_mode=learn_mode,
                )
                episode = self._build_runtime_episode(
                    operation="respond",
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
                self._finalize_trace(
                    operation="respond",
                    trace_id=trace_id,
                    created_at=created_at,
                    request=request,
                    episode=episode,
                    state_after=self._service_state_snapshot_fn(include_replay_dataset_summary=False),
                    state_before=state_before,
                    error=exc,
                )
                raise
