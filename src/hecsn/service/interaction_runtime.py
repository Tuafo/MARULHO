"""Compatibility mixin for operator interaction runtime.

Query and feed now delegate through the constructor-injected
InteractionPipeline seam. Respond/acquire remain here for now, along with the
shared helpers that those public methods still use.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import time
from typing import Any, Mapping, cast
from uuid import uuid4

import torch

from hecsn.config.presets import get_autonomy_acquisition_preset
from hecsn.gap_planner import plan_query_gaps
from hecsn.service.interaction_pipeline import (
    DEFAULT_FEED_CONCEPT_OBSERVATION_INTERVAL,
    REQUEST_FEED_ENCODING_MODE,
)
from hecsn.semantics.grounding_text import salient_query_terms
from hecsn.training.autonomy_acquisition_runner import run_live_acquisition
from hecsn.training.query_runner import build_query_result, feed_text

PUBLIC_ACQUISITION_PRESET = "autonomy_acquisition_hf_allocation"
PUBLIC_ACQUISITION_PRESETS: tuple[str, ...] = (PUBLIC_ACQUISITION_PRESET,)
PUBLIC_ACQUISITION_POLICIES: tuple[str, ...] = ("active", "round_robin")
DEFAULT_RECENT_QUERY_GAP_HISTORY = 8


class InteractionRuntimeMixin:
    def query(
        self,
        *,
        query_text: str,
        context_text: str | None = None,
        top_k_candidates: int = 5,
        top_k_memories: int = 5,
        top_chars: int = 6,
    ) -> dict[str, Any]:
        return self._interaction_pipeline.query(
            query_text=query_text,
            context_text=context_text,
            top_k_candidates=top_k_candidates,
            top_k_memories=top_k_memories,
            top_chars=top_chars,
        )

    def feed(
        self,
        *,
        text: str,
    ) -> dict[str, Any]:
        return self._interaction_pipeline.feed(text=text)

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
        with self._lock:
            started_perf = time.perf_counter()
            created_at = datetime.now(timezone.utc).isoformat()
            trace_id = str(uuid4())
            request = {
                "query_text": query_text,
                "context_text": context_text,
                "top_k_candidates": int(top_k_candidates),
                "top_k_memories": int(top_k_memories),
                "top_chars": int(top_chars),
                "max_evidence_items": int(max_evidence_items),
                "learn_mode": learn_mode,
            }
            state_before = self._service_state_snapshot(include_replay_dataset_summary=False)
            try:
                query_result = self._build_query_locked(
                    query_text=query_text,
                    context_text=context_text,
                    top_k_candidates=top_k_candidates,
                    top_k_memories=top_k_memories,
                    top_chars=top_chars,
                )
                query_result["concept_summary"] = self._observe_concepts_locked(
                    query_text=query_text,
                    query_result=query_result,
                )
                query_result["gap_plan"] = self._plan_gaps_locked(
                    query_text=query_text,
                    query_result=query_result,
                )
                query_result["delayed_consequence"] = self._apply_delayed_query_consequence_locked(
                    query_result=query_result,
                )
                self._record_recent_query_gap_locked(
                    query_text=query_text,
                    gap_plan=query_result["gap_plan"],
                    source="respond",
                )
                query_summary = query_result.get("query_summary") or {}
                response = self._responder.build_response(
                    query_text=query_text,
                    query_summary=query_summary,
                    concept_summary=query_result.get("concept_summary"),
                    max_evidence_items=max_evidence_items,
                )
                proposed_response = deepcopy(response)
                action_assist = self._maybe_auto_action_assist_locked(
                    query_text=query_text,
                    query_result=query_result,
                    response=response,
                )
                if action_assist is not None:
                    if int(action_assist.get("response_episode_count", 0) or 0) > 0:
                        query_summary = query_result.get("query_summary") or {}
                        response = self._responder.build_response(
                            query_text=query_text,
                            query_summary=query_summary,
                            concept_summary=query_result.get("concept_summary"),
                            max_evidence_items=max_evidence_items,
                        )
                        action_assist["used_in_response"] = True
                    response_note = self._normalize_action_text(action_assist.get("response_note", ""))
                    if response_note:
                        base_text = self._normalize_action_text(response.get("response_text", ""))
                        if response_note.strip() not in base_text:
                            response["response_text"] = (base_text + response_note).strip()
                            action_assist["used_in_response"] = True
                    query_result["action_assist"] = deepcopy(action_assist)
                    response["action_assist"] = deepcopy(action_assist)
                response_outcome_score = self._response_grounded_outcome_score_locked(
                    query_result=query_result,
                    response=response,
                    action_assist=action_assist,
                )
                applied_background_provenance = self._apply_background_source_response_provenance_locked(
                    response=response,
                    outcome_score=response_outcome_score,
                )
                if not applied_background_provenance:
                    self._apply_background_source_outcome_calibration_locked(
                        query_text=query_text,
                        outcome_score=response_outcome_score,
                    )
                autonomy = cast(dict[str, Any] | None, self._brain_config.get("autonomy"))
                if autonomy is not None:
                    self._apply_provider_response_outcome_calibration_locked(
                        autonomy=autonomy,
                        response=response,
                        outcome_score=response_outcome_score,
                    )
                learning = self._learn_from_turn_locked(query_text=query_text, response=response, learn_mode=learn_mode)
                delayed_candidate = self._record_response_consequence_candidate_locked(
                    query_result=query_result,
                    response=response,
                    outcome_score=response_outcome_score,
                )
                if delayed_candidate is not None:
                    response["delayed_consequence_candidate"] = deepcopy(delayed_candidate)

                action = {
                    "action_type": "respond",
                    "learn_mode": learn_mode,
                    "max_evidence_items": int(max_evidence_items),
                }
                if isinstance(action_assist, Mapping):
                    record = action_assist.get("result") if isinstance(action_assist.get("result"), Mapping) else {}
                    action["action_assist"] = {
                        "triggered": bool(action_assist.get("triggered", False)),
                        "executed": bool(action_assist.get("executed", False)),
                        "reused_recent_action": bool(action_assist.get("reused_recent_action", False)),
                        "reason": self._normalize_action_text(action_assist.get("reason", "")),
                        "action_type": self._normalize_action_text(record.get("action_type", "")),
                        "action_id": self._normalize_action_text(record.get("action_id", "")),
                    }
                    predicted_action = self._normalize_action_text(record.get("predicted_outcome", ""))
                    if predicted_action:
                        action["proposed_action"] = predicted_action
                prediction = {
                    "kind": "response_prediction",
                    "predicted_output": self._normalize_action_text(proposed_response.get("response_text", ""))
                    or f"Respond should produce a grounded answer for: {query_text}",
                    "proposed_answer": self._normalize_action_text(proposed_response.get("response_text", "")),
                    "confidence": float(proposed_response.get("support_score", 0.0) or 0.0),
                    "topics": salient_query_terms(query_text)[:8],
                }
                if action.get("proposed_action"):
                    prediction["proposed_action"] = action["proposed_action"]
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
                state_after = self._service_state_snapshot(include_replay_dataset_summary=False)
                episode = self._runtime_episode_payload_locked(
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
                trace_path = self._persist_trace_locked(trace)
                episode["trace_path"] = str(trace_path)
                episode = self._append_runtime_episode_trace_locked(episode)
                return {
                    "trace_id": trace["trace_id"],
                    "trace_path": str(trace_path),
                    "created_at": trace["created_at"],
                    "query_result": query_result,
                    "response": response,
                    "learning": learning,
                    "runtime_episode": episode,
                    **self._runtime_state.mutation_summary(),
                }
            except Exception as exc:
                prediction = {
                    "kind": "response_prediction",
                    "predicted_output": f"Respond should produce a grounded answer for: {query_text}",
                    "topics": salient_query_terms(query_text)[:8],
                }
                action = {
                    "action_type": "respond",
                    "learn_mode": learn_mode,
                    "max_evidence_items": int(max_evidence_items),
                }
                episode = self._runtime_episode_payload_locked(
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
                    "state_after": self._service_state_snapshot(include_replay_dataset_summary=False),
                }
                trace_path = self._persist_trace_locked(trace)
                episode["trace_path"] = str(trace_path)
                self._append_runtime_episode_trace_locked(episode)
                raise

    def acquire(
        self,
        *,
        preset: str = PUBLIC_ACQUISITION_PRESET,
        policy: str = "active",
        acquisition_slots: int | None = None,
        acquisition_tokens: int | None = None,
        save_checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if preset not in PUBLIC_ACQUISITION_PRESETS:
                raise ValueError(
                    "Unsupported acquisition preset for the maintained service surface. "
                    f"Supported presets: {', '.join(PUBLIC_ACQUISITION_PRESETS)}"
                )
            if policy not in PUBLIC_ACQUISITION_POLICIES:
                raise ValueError(
                    "Unsupported acquisition policy for the maintained service surface. "
                    f"Supported policies: {', '.join(PUBLIC_ACQUISITION_POLICIES)}"
                )
            preset_args = get_autonomy_acquisition_preset(preset)
            state_before = self._service_state_snapshot(include_replay_dataset_summary=False)
            focus_plan = self._autonomy_focus_plan_locked()
            shortlist_size, shortlist_gap_weight, shortlist_affinity_weight = self._autonomy_shortlist_settings_locked(
                candidate_bank=list(preset_args.get("candidate_bank", [])),
                config=preset_args,
                focus_plan=focus_plan,
            )
            result = run_live_acquisition(
                trainer=self._trainer,
                encoder=self._encoder,
                candidate_bank_specs=self._autonomy_candidate_specs_locked(
                    candidate_bank=list(preset_args.get("candidate_bank", [])),
                    focus_plan=focus_plan,
                ),
                candidate_train_tokens=int(preset_args.get("candidate_train_tokens", 0)),
                probe_tokens=int(preset_args.get("probe_tokens", 0)),
                acquisition_tokens=int(acquisition_tokens if acquisition_tokens is not None else preset_args.get("acquisition_tokens", 0)),
                acquisition_slots=int(acquisition_slots if acquisition_slots is not None else preset_args.get("acquisition_slots", 1)),
                gap_exploration_bonus=float(preset_args.get("gap_exploration_bonus", 0.0)),
                gap_ambiguity_weight=float(preset_args.get("gap_ambiguity_weight", 0.0)),
                gap_switch_weight=float(preset_args.get("gap_switch_weight", 0.0)),
                gap_margin_reference=float(preset_args.get("gap_margin_reference", 0.12)),
                coverage_balance_penalty=float(preset_args.get("coverage_balance_penalty", 0.0)),
                gap_focus_margin=float(preset_args.get("gap_focus_margin", 0.0)),
                policy_name=policy,
                semantic_shortlist_size=shortlist_size,
                semantic_shortlist_gap_weight=shortlist_gap_weight,
                semantic_shortlist_affinity_weight=shortlist_affinity_weight,
                semantic_plan=focus_plan,
                on_train_step=self._runtime_concept_callback_locked(),
            )
            if int(result.get("tokens_trained_total", 0)) > 0:
                self._runtime_state.mark_mutated()

            checkpoint_save = None
            if save_checkpoint_path is not None:
                checkpoint_save = self.save_checkpoint(save_checkpoint_path)

            trace = {
                "trace_id": str(uuid4()),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "operation": "acquisition",
                "request": {
                    "preset": preset,
                    "policy": policy,
                    "acquisition_slots": acquisition_slots,
                    "acquisition_tokens": acquisition_tokens,
                    "save_checkpoint_path": save_checkpoint_path,
                },
                "state_before": state_before,
                "acquisition_result": result,
                "checkpoint_save": checkpoint_save,
                "state_after": self._service_state_snapshot(include_replay_dataset_summary=False),
            }
            trace_path = self._persist_trace_locked(trace)
            return {
                "trace_id": trace["trace_id"],
                "trace_path": str(trace_path),
                "created_at": trace["created_at"],
                "preset": preset,
                "policy": policy,
                "acquisition_result": result,
                "checkpoint_save": checkpoint_save,
                **self._runtime_state.mutation_summary(),
                "token_count": int(self._trainer.token_count),
            }

    def _build_query_locked(
        self,
        *,
        query_text: str,
        context_text: str | None,
        top_k_candidates: int,
        top_k_memories: int,
        top_chars: int,
    ) -> dict[str, Any]:
        query_focus_plan = self._concept_store.focus_plan(
            query_text=query_text,
            min_observations=1,
        )
        retrieval_focus_terms = None
        memory_priority = None
        if query_focus_plan is not None:
            retrieval_focus_terms = list(
                query_focus_plan.get("focus_terms")
                or query_focus_plan.get("query_terms")
                or []
            )
            raw_memory_priority = dict(query_focus_plan.get("memory_priority") or {})
            if raw_memory_priority:
                memory_priority = raw_memory_priority
        try:
            result = build_query_result(
                trainer=self._trainer,
                checkpoint=self._checkpoint_path,
                metadata=deepcopy(self._metadata),
                encoder=self._encoder,
                query_text_resolved=query_text,
                feed_text_resolved=None,
                context_text=context_text,
                top_k_candidates=top_k_candidates,
                top_k_memories=top_k_memories,
                top_chars=top_chars,
                compare_context_a=None,
                compare_context_b=None,
                retrieval_focus_terms=retrieval_focus_terms,
                memory_priority=memory_priority,
            )
            query_summary = result.get("query_summary")
            if isinstance(query_summary, dict) and query_focus_plan is not None:
                query_summary["abstraction_focus"] = deepcopy(query_focus_plan)
            return result
        finally:
            self._trainer.reset_context_state()

    def _learn_from_turn_locked(
        self,
        *,
        query_text: str,
        response: dict[str, Any],
        learn_mode: str,
    ) -> dict[str, Any] | None:
        if learn_mode == "none":
            return None

        user_feed = feed_text(
            self._trainer,
            self._encoder,
            query_text,
            on_step=self._runtime_concept_callback_locked(),
        )
        evidence_feed = None
        selected_texts = [
            str(item.get("text", "")).strip()
            for item in response.get("selected_evidence", [])
            if str(item.get("text", "")).strip()
        ]

        if learn_mode == "user_and_selected_evidence" and selected_texts:
            evidence_feed = feed_text(
                self._trainer,
                self._encoder,
                "\n".join(selected_texts),
                on_step=self._runtime_concept_callback_locked(),
            )
        elif learn_mode != "user_only":
            raise ValueError(f"Unsupported learn_mode: {learn_mode}")

        self._runtime_state.mark_mutated()
        return {
            "learn_mode": learn_mode,
            "user_feed": user_feed,
            "evidence_feed": evidence_feed,
            "selected_evidence_count": int(len(selected_texts)),
        }

    def _observe_concepts_locked(
        self,
        *,
        query_text: str,
        query_result: dict[str, Any],
    ) -> dict[str, Any]:
        query_summary = query_result.get("query_summary") or {}
        memory_matches = query_summary.get("memory_matches") or []
        memory_episodes = query_summary.get("memory_episodes") or []
        return self._concept_store.observe(
            query_text=query_text,
            memory_matches=memory_matches,
            memory_episodes=memory_episodes,
            memory_store=self._trainer.model.memory_store,
        )

    def _runtime_concept_callback_locked(self):
        def _observe(raw_window: str, metrics: dict[str, Any]) -> None:
            self._observe_runtime_concepts_locked(raw_window=raw_window, metrics=metrics)

        return _observe

    def _observe_runtime_concepts_locked(
        self,
        *,
        raw_window: str | None,
        metrics: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(metrics, dict):
            return None
        memory_index = metrics.get("memory_index")
        try:
            idx = int(memory_index)
        except (TypeError, ValueError):
            return None

        memory_store = self._trainer.model.memory_store
        routing_keys = getattr(memory_store, "slow_routing_keys", []) or []
        if idx < 0 or idx >= len(routing_keys):
            return None
        if not isinstance(routing_keys[idx], torch.Tensor):
            return None

        stored_texts = getattr(memory_store, "slow_texts", []) or []
        stored_windows = getattr(memory_store, "slow_raw_windows", []) or []
        source_text = ""
        if idx < len(stored_texts) and stored_texts[idx] is not None:
            source_text = str(stored_texts[idx])
        elif idx < len(stored_windows) and stored_windows[idx] is not None:
            source_text = str(stored_windows[idx])
        elif raw_window is not None:
            source_text = str(raw_window)
        source_text = " ".join(source_text.split()).strip()
        if not source_text or not any(char.isalnum() for char in source_text):
            return None

        raw_match = (
            str(stored_windows[idx])
            if idx < len(stored_windows) and stored_windows[idx] is not None
            else source_text
        )
        importance = 1.0
        capture_tag = 0.0
        consolidation_level = 0.0
        slow_importance = getattr(memory_store, "slow_importance", []) or []
        slow_capture_tag = getattr(memory_store, "slow_capture_tag", []) or []
        slow_consolidation = getattr(memory_store, "slow_consolidation_level", []) or []
        if idx < len(slow_importance):
            importance = float(memory_store.slow_importance[idx])
        if idx < len(slow_capture_tag):
            capture_tag = float(memory_store.slow_capture_tag[idx])
        if idx < len(slow_consolidation):
            consolidation_level = float(memory_store.slow_consolidation_level[idx])

        observed = self._concept_store.observe(
            query_text="",
            memory_matches=[
                {
                    "memory_index": idx,
                    "text": source_text,
                    "raw_window": raw_match,
                    "similarity": 1.0,
                    "importance": importance,
                    "capture_tag": capture_tag,
                    "consolidation_level": consolidation_level,
                }
            ],
            memory_store=memory_store,
            limit=4,
        )
        abstraction_layer = self._trainer.model.abstraction_layer
        if abstraction_layer is not None:
            self._geometric_curiosity.update_lexicon(
                abstraction_layer.last_activations,
                [source_text, raw_match],
            )
        return observed

    def _plan_gaps_locked(
        self,
        *,
        query_text: str,
        query_result: dict[str, Any],
    ) -> dict[str, Any]:
        return plan_query_gaps(
            query_text=query_text,
            query_summary=query_result.get("query_summary"),
            concept_summary=query_result.get("concept_summary"),
        )

    def _normalize_recent_query_gap(self, item: Any) -> dict[str, Any] | None:
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

    def _record_recent_query_gap_locked(
        self,
        *,
        query_text: str,
        gap_plan: dict[str, Any],
        source: str,
    ) -> None:
        normalized_query = " ".join(str(query_text).split()).strip()
        if not normalized_query:
            return
        existing = [
            item
            for item in list(self._brain_recent_query_gaps)
            if str(item.get("query_text", "")).lower() != normalized_query.lower()
        ]
        self._brain_recent_query_gaps = deque(existing, maxlen=DEFAULT_RECENT_QUERY_GAP_HISTORY)
        grounded_fraction = float(gap_plan.get("grounded_fraction", 0.0))
        query_deficit = bool(gap_plan.get("unsupported_terms")) or grounded_fraction < 0.999
        self._brain_skip_next_autonomy_for_grounded_query = not query_deficit
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
            self._brain_recent_query_gaps.appendleft(normalized)
