"""Interaction Pipeline seam for query, feed, and respond turns.

This module owns the constructor-injected operator-turn seam extracted from
the Service Manager. It handles query, feed, and respond turn orchestration,
turn-specific runtime episode trace construction, and the query/feed/respond
actual-output / verification payload behavior while delegating collaborator-
specific work back into injected callbacks.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
import time
from typing import Any, Callable, Mapping
from uuid import uuid4

from hecsn.semantics.grounding_text import salient_query_terms


DEFAULT_FEED_CONCEPT_OBSERVATION_INTERVAL = 8
REQUEST_FEED_ENCODING_MODE = "lexical_rolling_segments"


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
        record_recent_query_gap_fn: Callable[..., None],
        observe_runtime_concepts_fn: Callable[..., dict[str, Any] | None],
        runtime_state_mark_mutated_fn: Callable[[], None],
        runtime_state_mutation_summary_fn: Callable[[], dict[str, Any]],
        runtime_episode_payload_fn: Callable[..., dict[str, Any]],
        persist_trace_fn: Callable[[dict[str, Any]], Path],
        append_runtime_episode_trace_fn: Callable[[Mapping[str, Any]], dict[str, Any]],
        service_state_snapshot_fn: Callable[..., dict[str, Any]],
        build_response_fn: Callable[..., dict[str, Any]] | None = None,
        maybe_auto_action_assist_fn: Callable[..., dict[str, Any] | None] | None = None,
        response_grounded_outcome_score_fn: Callable[..., float] | None = None,
        apply_background_source_response_provenance_fn: Callable[..., bool] | None = None,
        apply_background_source_outcome_calibration_fn: Callable[..., None] | None = None,
        apply_provider_response_outcome_calibration_fn: Callable[..., bool] | None = None,
        learn_from_turn_fn: Callable[..., dict[str, Any] | None] | None = None,
        record_response_consequence_candidate_fn: Callable[..., dict[str, Any] | None] | None = None,
    ) -> None:
        self._lock = lock
        self._trainer = trainer
        self._encoder = encoder
        self._build_query_result_fn = build_query_result_fn
        self._observe_concepts_fn = observe_concepts_fn
        self._plan_gaps_fn = plan_gaps_fn
        self._apply_delayed_query_consequence_fn = apply_delayed_query_consequence_fn
        self._record_recent_query_gap_fn = record_recent_query_gap_fn
        self._observe_runtime_concepts_fn = observe_runtime_concepts_fn
        self._runtime_state_mark_mutated_fn = runtime_state_mark_mutated_fn
        self._runtime_state_mutation_summary_fn = runtime_state_mutation_summary_fn
        self._runtime_episode_payload_fn = runtime_episode_payload_fn
        self._persist_trace_fn = persist_trace_fn
        self._append_runtime_episode_trace_fn = append_runtime_episode_trace_fn
        self._service_state_snapshot_fn = service_state_snapshot_fn
        self._build_response_fn = build_response_fn
        self._maybe_auto_action_assist_fn = maybe_auto_action_assist_fn
        self._response_grounded_outcome_score_fn = response_grounded_outcome_score_fn
        self._apply_background_source_response_provenance_fn = apply_background_source_response_provenance_fn
        self._apply_background_source_outcome_calibration_fn = apply_background_source_outcome_calibration_fn
        self._apply_provider_response_outcome_calibration_fn = apply_provider_response_outcome_calibration_fn
        self._learn_from_turn_fn = learn_from_turn_fn
        self._record_response_consequence_candidate_fn = record_response_consequence_candidate_fn

    def _query_runtime_actual_output(self, result: Mapping[str, Any]) -> dict[str, Any]:
        return build_query_runtime_actual_output(result)

    def _query_runtime_verification(self, result: Mapping[str, Any]) -> dict[str, Any]:
        return build_query_runtime_verification(result)

    def _feed_runtime_actual_output(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        return build_feed_runtime_actual_output(summary)

    def _feed_runtime_verification(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        return build_feed_runtime_verification(summary)

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
        return self._append_runtime_episode_trace_fn(finalized_episode)

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
        if self._build_response_fn is None:
            raise RuntimeError("InteractionPipeline respond callbacks are not configured")
        if self._maybe_auto_action_assist_fn is None:
            raise RuntimeError("InteractionPipeline respond callbacks are not configured")
        if self._response_grounded_outcome_score_fn is None:
            raise RuntimeError("InteractionPipeline respond callbacks are not configured")
        if self._apply_background_source_response_provenance_fn is None:
            raise RuntimeError("InteractionPipeline respond callbacks are not configured")
        if self._apply_background_source_outcome_calibration_fn is None:
            raise RuntimeError("InteractionPipeline respond callbacks are not configured")
        if self._apply_provider_response_outcome_calibration_fn is None:
            raise RuntimeError("InteractionPipeline respond callbacks are not configured")
        if self._learn_from_turn_fn is None:
            raise RuntimeError("InteractionPipeline respond callbacks are not configured")
        if self._record_response_consequence_candidate_fn is None:
            raise RuntimeError("InteractionPipeline respond callbacks are not configured")

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
                query_result = self._build_query_result_fn(
                    query_text=query_text,
                    context_text=context_text,
                    top_k_candidates=top_k_candidates,
                    top_k_memories=top_k_memories,
                    top_chars=top_chars,
                )
                query_result["concept_summary"] = self._observe_concepts_fn(
                    query_text=query_text,
                    query_result=query_result,
                )
                query_result["gap_plan"] = self._plan_gaps_fn(
                    query_text=query_text,
                    query_result=query_result,
                )
                query_result["delayed_consequence"] = self._apply_delayed_query_consequence_fn(
                    query_result=query_result,
                )
                self._record_recent_query_gap_fn(
                    query_text=query_text,
                    gap_plan=query_result["gap_plan"],
                    source="respond",
                )
                query_summary = query_result.get("query_summary") or {}
                response = self._build_response_fn(
                    query_text=query_text,
                    query_summary=query_summary,
                    concept_summary=query_result.get("concept_summary"),
                    max_evidence_items=max_evidence_items,
                )
                if not isinstance(response, dict):
                    raise TypeError("build_response_fn must return a dict")
                proposed_response = deepcopy(response)
                action_assist = self._maybe_auto_action_assist_fn(
                    query_text=query_text,
                    query_result=query_result,
                    response=response,
                )
                if action_assist is not None and not isinstance(action_assist, Mapping):
                    raise TypeError("maybe_auto_action_assist_fn must return a mapping or None")
                if action_assist is not None:
                    if int(action_assist.get("response_episode_count", 0) or 0) > 0:
                        query_summary = query_result.get("query_summary") or {}
                        response = self._build_response_fn(
                            query_text=query_text,
                            query_summary=query_summary,
                            concept_summary=query_result.get("concept_summary"),
                            max_evidence_items=max_evidence_items,
                        )
                        if not isinstance(response, dict):
                            raise TypeError("build_response_fn must return a dict")
                        action_assist["used_in_response"] = True
                    response_note = _normalize_action_text(action_assist.get("response_note", ""))
                    if response_note:
                        base_text = _normalize_action_text(response.get("response_text", ""))
                        if response_note.strip() not in base_text:
                            response["response_text"] = (base_text + response_note).strip()
                            action_assist["used_in_response"] = True
                    query_result["action_assist"] = deepcopy(action_assist)
                    response["action_assist"] = deepcopy(action_assist)
                response_outcome_score = self._response_grounded_outcome_score_fn(
                    query_result=query_result,
                    response=response,
                    action_assist=action_assist,
                )
                applied_background_provenance = bool(
                    self._apply_background_source_response_provenance_fn(
                        response=response,
                        outcome_score=response_outcome_score,
                    )
                )
                if not applied_background_provenance:
                    self._apply_background_source_outcome_calibration_fn(
                        query_text=query_text,
                        outcome_score=response_outcome_score,
                    )
                self._apply_provider_response_outcome_calibration_fn(
                    response=response,
                    outcome_score=response_outcome_score,
                )
                learning = self._learn_from_turn_fn(
                    query_text=query_text,
                    response=response,
                    learn_mode=learn_mode,
                )
                delayed_candidate = self._record_response_consequence_candidate_fn(
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
                if isinstance(action_assist, Mapping):
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
                trace = {
                    "trace_id": trace_id,
                    "created_at": created_at,
                    "operation": "respond",
                    "request": request,
                    "state_before": state_before,
                    "query_result": query_result,
                    "response": response,
                    "learning": learning,
                    "runtime_episode": episode,
                    "state_after": state_after,
                }
                trace_path = self._persist_trace_fn(trace)
                episode["trace_path"] = str(trace_path)
                episode = self._append_runtime_episode_trace_fn(episode)
                return {
                    "trace_id": trace["trace_id"],
                    "trace_path": str(trace_path),
                    "created_at": trace["created_at"],
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
                trace = {
                    "trace_id": trace_id,
                    "created_at": created_at,
                    "operation": "respond",
                    "request": request,
                    "state_before": state_before,
                    "runtime_episode": episode,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "state_after": self._service_state_snapshot_fn(include_replay_dataset_summary=False),
                }
                trace_path = self._persist_trace_fn(trace)
                episode["trace_path"] = str(trace_path)
                self._append_runtime_episode_trace_fn(episode)
                raise
