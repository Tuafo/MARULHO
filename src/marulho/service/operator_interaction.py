"""Operator interaction runtime helpers for query, feed, respond, and acquire.

Query, feed, and respond delegate through the constructor-injected
InteractionPipeline seam. Acquisition stays here because it is an operator
interaction flow over autonomy, source selection, training, and trace capture.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import time
from typing import Any, Mapping, Sequence, cast
from uuid import uuid4

from marulho.config.presets import get_autonomy_acquisition_preset
from marulho.gap_planner import plan_query_gaps
from marulho.service.interaction_pipeline import (
    DEFAULT_FEED_CONCEPT_OBSERVATION_INTERVAL,
    InteractionPipeline,
    REQUEST_FEED_ENCODING_MODE,
)
from marulho.semantics.grounding_text import salient_query_terms
from marulho.training.autonomy_acquisition_runner import run_live_acquisition
from marulho.training.query_runner import build_query_result, feed_text

PUBLIC_ACQUISITION_PRESET = "autonomy_acquisition_hf_allocation"
PUBLIC_ACQUISITION_PRESETS: tuple[str, ...] = (PUBLIC_ACQUISITION_PRESET,)
PUBLIC_ACQUISITION_POLICIES: tuple[str, ...] = ("active", "round_robin")
RUNTIME_CONCEPT_MEMORY_LOOKUP_LIMIT = 64


class OperatorInteractionRuntime:
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
        return self._interaction_pipeline.respond(
            query_text=query_text,
            context_text=context_text,
            top_k_candidates=top_k_candidates,
            top_k_memories=top_k_memories,
            top_chars=top_chars,
            max_evidence_items=max_evidence_items,
            learn_mode=learn_mode,
        )

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
                on_train_step=OperatorInteractionRuntime._runtime_concept_callback_locked(self),
            )
            if int(result.get("tokens_trained_total", 0)) > 0:
                self._runtime_state.mark_mutated()

            checkpoint_save = None
            if save_checkpoint_path is not None:
                checkpoint_save = self._runtime_persistence.save_checkpoint(save_checkpoint_path)

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
            on_step=OperatorInteractionRuntime._runtime_concept_callback_locked(self),
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
                on_step=OperatorInteractionRuntime._runtime_concept_callback_locked(self),
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
            OperatorInteractionRuntime._observe_runtime_concepts_locked(
                self,
                raw_window=raw_window,
                metrics=metrics,
            )

        return _observe

    def _observe_runtime_concepts_locked(
        self,
        *,
        raw_window: str | None,
        metrics: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        observed = OperatorInteractionRuntime._observe_runtime_concept_batch_locked(
            self,
            observations=[(raw_window, metrics)],
        )
        return None if not observed else observed[0]

    def _observe_runtime_concept_batch_locked(
        self,
        *,
        observations: Sequence[tuple[str | None, dict[str, Any] | None]],
    ) -> list[dict[str, Any] | None]:
        memory_store = self._trainer.model.memory_store
        resolver = getattr(memory_store, "resolve_runtime_concept_memory_matches", None)
        if not callable(resolver):
            return [None] * len(observations)

        resolved = resolver(
            observations=observations,
            max_observations=RUNTIME_CONCEPT_MEMORY_LOOKUP_LIMIT,
        )
        matches = [
            dict(match)
            for match in list(resolved.get("matches", []))
            if isinstance(match, Mapping)
        ]
        source_pairs = [
            (str(left), str(right))
            for left, right in list(resolved.get("source_pairs", []))
        ]
        result_slots = [
            None if slot is None else int(slot)
            for slot in list(resolved.get("result_slots", []))
        ]
        if len(result_slots) < len(observations):
            result_slots.extend([None] * (len(observations) - len(result_slots)))
        elif len(result_slots) > len(observations):
            result_slots = result_slots[: len(observations)]

        results: list[dict[str, Any] | None] = [None] * len(observations)
        for match_index, match in enumerate(matches):
            observed = self._concept_store.observe(
                query_text="",
                memory_matches=[match],
                memory_store=memory_store,
                limit=4,
                maintain_structure=match_index == len(matches) - 1,
            )
            for result_index, slot in enumerate(result_slots):
                if slot == match_index:
                    results[result_index] = observed

        abstraction_layer = self._trainer.model.abstraction_layer
        if abstraction_layer is not None:
            for source_text, raw_match in source_pairs:
                self._geometric_curiosity.update_lexicon(
                    abstraction_layer.last_activations,
                    [source_text, raw_match],
                )
        return results

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
        return InteractionPipeline._normalize_recent_query_gap(item)

    def _record_recent_query_gap_locked(
        self,
        *,
        query_text: str,
        gap_plan: dict[str, Any],
        source: str,
    ) -> None:
        self._interaction_pipeline.record_recent_query_gap(
            query_text=query_text,
            gap_plan=gap_plan,
            source=source,
        )
