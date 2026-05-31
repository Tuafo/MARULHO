"""Living-loop status helpers for Terminus.

This component builds auditable status snapshots for the living-loop scaffold and
policy actuator. It is read-only: no replay, memory, action, or feedback
mutation is performed here.
"""

from __future__ import annotations

from copy import deepcopy
import time
from typing import Any, Mapping, cast

from hecsn.service.living_loop_records import (
    ActionExecutionRecord,
    ConsolidationRecord,
    ProvenanceState,
    RuntimeEpisodeTrace,
)
from hecsn.service.living_loop_policy import build_policy_actuator_status
from hecsn.service.living_loop_replay import build_replay_plan
from hecsn.service.living_loop_self_model import (
    OperationalSelfModel,
    build_runtime_benchmark_telemetry,
)

DEFAULT_REPLAY_DATASET_EXPORT_LIMIT = 20


class LivingStatusCore:
    def _living_loop_snapshot_locked(
        self,
        *,
        include_replay_dataset_summary: bool = False,
    ) -> dict[str, Any]:
        memory_snapshot = self._trainer.model.memory_store.summary_stats()
        provenance = ProvenanceState.from_distribution(
            cast(Mapping[str, Any], memory_snapshot).get("provenance_distribution")
            if isinstance(memory_snapshot, Mapping)
            else {}
        )
        action_records = [
            ActionExecutionRecord.from_payload(item)
            for item in list(self._action_history)[:8]
            if isinstance(item, Mapping)
        ]
        consolidation_records = [
            ConsolidationRecord.from_payload(item)
            for item in list(self._delayed_consequence_records)[:8]
            if isinstance(item, Mapping)
        ]
        runtime_episodes = [
            RuntimeEpisodeTrace.from_payload(item)
            for item in list(self._interaction_pipeline.runtime_episode_traces())[:12]
            if isinstance(item, Mapping)
        ]
        runtime_state_revision = int(self._runtime_state.state_revision)
        model = OperationalSelfModel.build(
            token_count=int(self._trainer.token_count),
            state_revision=runtime_state_revision,
            configured=bool(self._brain_config.get("source_bank")),
            running=bool(self._brain_runtime_active_locked()),
            provenance=provenance,
            predictions=[item.prediction for item in action_records],
            actions=action_records,
            consolidations=consolidation_records,
            runtime_episodes=runtime_episodes,
            action_loop=self._action_executor.action_loop_summary(),
            memory=dict(memory_snapshot) if isinstance(memory_snapshot, Mapping) else {},
            narrative={},
        )
        payload = model.to_payload()
        feedback_summary = self._runtime_feedback_summary_locked()
        payload["feedback_summary"] = feedback_summary
        payload["feedback_count"] = int(feedback_summary["feedback_count"])
        payload["verified_feedback_count"] = int(feedback_summary["verified_count"])
        payload["contradicted_feedback_count"] = int(feedback_summary["contradicted_count"])
        payload["unverified_feedback_count"] = int(feedback_summary["unverified_count"])
        payload["recent_feedback"] = [deepcopy(item) for item in feedback_summary["recent_feedback"]]
        replay_sample_summary = self._replay_sample_summary_locked()
        payload["replay_sample_summary"] = replay_sample_summary
        payload["replay_executor_summary"] = replay_sample_summary
        grounding_health = (
            dict(payload.get("grounding_health") or {})
            if isinstance(payload.get("grounding_health"), Mapping)
            else {}
        )
        grounding_health.update(
            {
                "feedback_count": int(feedback_summary["feedback_count"]),
                "feedback_verified_count": int(feedback_summary["verified_count"]),
                "feedback_contradicted_count": int(feedback_summary["contradicted_count"]),
                "feedback_unverified_count": int(feedback_summary["unverified_count"]),
                "feedback_impact": str(feedback_summary["grounding_impact"]),
            }
        )
        if feedback_summary["contradicted_count"] > 0:
            grounding_health["status"] = "contradictions_present"
        elif feedback_summary["unverified_count"] > 0 and grounding_health.get("status") == "grounded":
            grounding_health["status"] = "needs_verification"
        payload["grounding_health"] = grounding_health
        memory_stats = self._trainer.model.memory_store.summary_stats()
        memory_fill = float(memory_stats.get("fill_fraction", 0.0) or 0.0)
        subcortex_sleep_pressure = {
            "source": "subcortex_memory_and_trainer_sleep_counters",
            "pressure": max(0.0, min(1.0, memory_fill)),
            "fatigue": max(0.0, min(1.0, memory_fill)),
            "is_sleeping": False,
            "memory_fill_ratio": memory_fill,
            "sleep_events": int(self._trainer.sleep_events),
            "micro_sleep_events": int(self._trainer.micro_sleep_events),
            "deep_sleep_events": int(self._trainer.deep_sleep_events),
        }
        payload["subcortex_sleep_pressure"] = subcortex_sleep_pressure
        payload["benchmark_telemetry"] = build_runtime_benchmark_telemetry(
            runtime_episodes=runtime_episodes,
            actions=action_records,
            world_model_lite=payload.get("world_model_lite") if isinstance(payload.get("world_model_lite"), Mapping) else None,
            action_loop=payload.get("action_loop") if isinstance(payload.get("action_loop"), Mapping) else {},
            memory=payload.get("memory") if isinstance(payload.get("memory"), Mapping) else {},
            runtime_memory=memory_stats,
            runtime={
                "tokens_per_second": (
                    float(self._brain_last_tick_token_delta) / (float(self._brain_last_tick_duration_ms) / 1000.0)
                    if self._brain_last_tick_duration_ms and self._brain_last_tick_duration_ms > 0.0
                    else 0.0
                ),
                "last_tick_token_delta": int(self._brain_last_tick_token_delta),
            },
            feedback_summary=feedback_summary,
            replay_sample_summary=replay_sample_summary,
            generated_at=str(payload.get("generated_at", "")) or None,
        )
        payload["policy_decision"] = build_policy_actuator_status(
            payload,
        ).to_payload()
        replay_plan = build_replay_plan(payload).to_payload()
        payload["replay_plan"] = replay_plan
        replay_dataset_summary: dict[str, Any] | None = None
        if include_replay_dataset_summary:
            replay_dataset_summary = self._replay_dataset_preview_summary_locked(
                living_loop=payload,
                plan=replay_plan,
                replay_sample_summary=replay_sample_summary,
                limit=DEFAULT_REPLAY_DATASET_EXPORT_LIMIT,
            )
            payload["replay_dataset_summary"] = replay_dataset_summary
        if isinstance(payload.get("benchmark_telemetry"), Mapping):
            payload["benchmark_telemetry"]["replay_plan_summary"] = self._replay_plan_summary(replay_plan)
            payload["benchmark_telemetry"]["replay_sample_summary"] = replay_sample_summary
            payload["benchmark_telemetry"]["replay_executor_summary"] = replay_sample_summary
            if replay_dataset_summary is not None:
                payload["benchmark_telemetry"]["replay_dataset_summary"] = replay_dataset_summary
        return payload

    def living_loop_status(self) -> dict[str, Any]:
        with self._lock:
            runtime_mutation = self._runtime_state.mutation_summary()
            return {
                "living_loop": self._living_loop_snapshot_locked(
                    include_replay_dataset_summary=True,
                ),
                **runtime_mutation,
                "token_count": int(self._trainer.token_count),
            }

    def policy_actuator_status(self) -> dict[str, Any]:
        with self._lock:
            living_loop = self._living_loop_snapshot_locked()
            return build_policy_actuator_status(
                living_loop,
            ).to_payload()

    def _cognitive_signal_state(self) -> dict[str, Any]:
        """Expose recent Subcortex predictive/surprise signals."""
        acquired = self._lock.acquire(timeout=0.05)
        if not acquired:
            return getattr(self, "_cached_cognitive_signal_state", {})
        try:
            predictive = getattr(self._trainer.model, "predictive", None)
            surprise = getattr(self._trainer.model, "surprise", None)
            recent_concepts: list[str] = []
            concept_candidates: list[dict[str, Any]] = []
            try:
                snap = self._concept_store.snapshot(limit=6)
                for concept in snap.get("top_concepts", [])[:6]:
                    if not isinstance(concept, dict):
                        continue
                    label = str(concept.get("label", "")).strip()
                    if label:
                        recent_concepts.append(label)
                    top_terms = [
                        str(term).strip()
                        for term in list(concept.get("top_terms") or [])[:4]
                        if str(term).strip()
                    ]
                    examples = [
                        str(text).strip()
                        for text in list(concept.get("example_windows") or [])[:2]
                        if str(text).strip()
                    ]
                    if label or top_terms:
                        concept_candidates.append(
                            {
                                "label": label,
                                "top_terms": top_terms,
                                "match_count": int(concept.get("match_count", concept.get("observations", 0)) or 0),
                                "observations": int(concept.get("observations", concept.get("match_count", 0)) or 0),
                                "uncertainty": float(concept.get("uncertainty", 1.0) or 1.0),
                                "temporal_coherence": float(concept.get("temporal_coherence", 0.0) or 0.0),
                                "example_windows": examples,
                            }
                        )
            except Exception:
                pass

            payload = {
                "schema_version": "cognitive_signal.v1",
                "source": "service.living_status",
                "sampled_at": time.time(),
                "prediction_error_mean": 0.0,
                "prediction_error_max": 0.0,
                "predictive_confidence_mean": 0.5,
                "predictive_confidence_min": 0.5,
                "dopamine": float(getattr(surprise, "dopamine", 0.0)) if surprise is not None else 0.0,
                "norepinephrine": float(getattr(surprise, "norepinephrine", 0.0)) if surprise is not None else 0.0,
                "recent_concepts": recent_concepts,
                "concept_candidates": concept_candidates,
            }
            if predictive is not None:
                try:
                    prediction_error = predictive.prediction_error.detach().float().cpu()
                    confidence = predictive.confidence.detach().float().cpu()
                    if prediction_error.numel() > 0:
                        payload["prediction_error_mean"] = float(prediction_error.mean().item())
                        payload["prediction_error_max"] = float(prediction_error.max().item())
                    if confidence.numel() > 0:
                        payload["predictive_confidence_mean"] = float(confidence.mean().item())
                        payload["predictive_confidence_min"] = float(confidence.min().item())
                except Exception:
                    pass

            self._cached_cognitive_signal_state = payload
            return payload
        finally:
            self._lock.release()
