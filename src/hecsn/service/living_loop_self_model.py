"""Operational Self-Model and telemetry module for the Living Loop (Layer D).

This module contains OperationalSelfModel (build, from_payload, to_payload,
all _surface_* methods), build_runtime_benchmark_telemetry, and all
telemetry-specific private helpers.

Dependency direction: Helpers → Records → Policy → Replay → Self-Model

This module imports from Replay, Policy, Records, and Helpers only;
it never imports from any consumer module.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from hecsn.service.living_loop_helpers import (
    _as_mapping,
    _clean_text,
    _clamp01,
    _coerce_world_model_lite,
    _limited_unique_clean_text,
    _safe_float,
    _safe_ratio,
    _stable_id,
)
from hecsn.service.living_loop_records import (
    ActionExecutionRecord,
    ActionExecutionStatus,
    ConsolidationRecord,
    PredictionRecord,
    PredictionStatus,
    ProvenanceState,
    RuntimeEpisodeTrace,
    SkillMemoryRecord,
    VerificationStatus,
)
from hecsn.service.living_loop_policy import (
    WorldModelLiteSummary,
    _coerce_feedback_telemetry,
)
from hecsn.service.living_loop_replay import (
    _coerce_replay_sample_summary,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RECENT_FAILURES = 6
_MAX_UNCERTAIN_DOMAINS = 8
_MAX_SUPPORTED_ACTIONS = 64

# ---------------------------------------------------------------------------
# Telemetry helpers
# ---------------------------------------------------------------------------

def _endpoint_latency_empty() -> dict[str, Any]:
    return {
        "count": 0,
        "latency_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "min_ms": None,
        "avg_ms": None,
        "max_ms": None,
        "success_rate": 0.0,
    }


def _endpoint_bucket_name(operation: Any) -> str:
    normalized = _clean_text(operation).lower()
    if normalized in {"feed", "query", "respond"}:
        return normalized
    if normalized in {"action", "runtime_action", "digital_action", "terminus_action"}:
        return "runtime_action"
    return normalized or "unknown"


def _latency_summary(
    count: int,
    success_count: int,
    failure_count: int,
    latencies: Sequence[float],
) -> dict[str, Any]:
    latency_values = [float(value) for value in latencies if value >= 0.0]
    return {
        "count": int(max(0, count)),
        "latency_count": int(len(latency_values)),
        "success_count": int(max(0, success_count)),
        "failure_count": int(max(0, failure_count)),
        "min_ms": min(latency_values) if latency_values else None,
        "avg_ms": (sum(latency_values) / len(latency_values)) if latency_values else None,
        "max_ms": max(latency_values) if latency_values else None,
        "success_rate": _safe_ratio(success_count, count),
    }


def _extract_cache_summary(stats: Mapping[str, Any]) -> dict[str, Any]:
    hits = stats.get("cache_hits")
    misses = stats.get("cache_misses")
    if hits is None and misses is None:
        hit_rate = None
        tracked = False
        hit_count = None
        miss_count = None
    else:
        hit_count = max(0, int(hits or 0))
        miss_count = max(0, int(misses or 0))
        hit_rate = _safe_ratio(hit_count, hit_count + miss_count)
        tracked = True
    return {
        "tracked": bool(tracked),
        "hit_rate": hit_rate,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "cache_size": int(stats.get("cache_size", 0) or 0),
    }


def _extract_retired_external_adapter_summary(retired_runtime_path: Mapping[str, Any] | None) -> dict[str, Any]:
    retired_runtime_path_data = dict(retired_runtime_path or {})
    raw_episodic = retired_runtime_path_data.get("episodic_memory")
    episodic: Mapping[str, Any] = raw_episodic if isinstance(raw_episodic, Mapping) else {}
    raw_embedder = episodic.get("embedder")
    embedder: Mapping[str, Any] = raw_embedder if isinstance(raw_embedder, Mapping) else {}
    chat_generations = int(retired_runtime_path_data.get("thoughts_generated", 0) or 0) + int(
        retired_runtime_path_data.get("dreams_generated", 0) or 0
    )
    embedding_calls = int(embedder.get("external_calls", 0) or 0)
    external_throttle_hits = int(embedder.get("external_throttle_hits", 0) or 0)
    retired_runtime_path_enabled = retired_runtime_path_data.get("enabled", False) or embedder.get("available", False)
    return {
        "available": bool(retired_runtime_path_enabled),
        "chat_generations_observed": int(chat_generations),
        "embedding_external_calls": int(embedding_calls),
        "observed_call_count": int(chat_generations + embedding_calls),
        "calls_per_minute": None,
        "calls_per_minute_reason": "timestamps_unavailable",
        "external_throttle_hits": int(external_throttle_hits),
        "embedder": {
            "kind": embedder.get("kind"),
            "model": embedder.get("model"),
            "available": bool(embedder.get("available", False)),
            "degraded": bool(embedder.get("degraded", False)),
            "fallback_calls": int(embedder.get("fallback_calls", 0) or 0),
            "error_calls": int(embedder.get("error_calls", 0) or 0),
        },
    }


def _memory_counter_summary(
    memory: Mapping[str, Any] | None,
    runtime_memory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source = dict(runtime_memory or memory or {})
    size = source.get("size", source.get("memory_count"))
    capacity = source.get("capacity")
    fill_ratio = source.get("fill_ratio", source.get("fill_fraction"))
    fill_value = _safe_float(fill_ratio)
    if fill_value is None:
        status = "no_capacity_snapshot" if source else "no_memory_snapshot"
    elif fill_value >= 0.90:
        status = "capacity_pressure"
    else:
        status = "available"
    return {
        "status": status,
        "size": int(size) if isinstance(size, int) else size,
        "capacity": int(capacity) if isinstance(capacity, int) else capacity,
        "fill_ratio": fill_value,
        "total_stored": source.get("total_stored"),
        "total_evicted": source.get("total_evicted"),
        "mean_confidence": source.get("mean_confidence"),
        "source": (
            "runtime_memory_store"
            if runtime_memory
            else ("retired_runtime_path_memory" if memory else "unavailable")
        ),
    }


# ---------------------------------------------------------------------------
# Benchmark telemetry
# ---------------------------------------------------------------------------

def build_runtime_benchmark_telemetry(
    *,
    runtime_episodes: Sequence[RuntimeEpisodeTrace | Mapping[str, Any]] = (),
    actions: Sequence[ActionExecutionRecord | Mapping[str, Any]] = (),
    world_model_lite: WorldModelLiteSummary | Mapping[str, Any] | None = None,
    action_loop: Mapping[str, Any] | None = None,
    memory: Mapping[str, Any] | None = None,
    runtime_memory: Mapping[str, Any] | None = None,
    retired_runtime_path: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    feedback_summary: Mapping[str, Any] | None = None,
    replay_sample_summary: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Transparent benchmark telemetry from already-recorded runtime facts."""
    retired_runtime_path_data = dict(retired_runtime_path or {})
    episode_records = tuple(
        item if isinstance(item, RuntimeEpisodeTrace) else RuntimeEpisodeTrace.from_payload(item)
        for item in runtime_episodes
        if isinstance(item, (RuntimeEpisodeTrace, Mapping))
    )
    action_records = tuple(
        item if isinstance(item, ActionExecutionRecord) else ActionExecutionRecord.from_payload(item)
        for item in actions
        if isinstance(item, (ActionExecutionRecord, Mapping))
    )
    world = _coerce_world_model_lite(world_model_lite)
    action_loop_data = dict(action_loop or {})
    runtime_data = dict(runtime or {})

    endpoint_accumulators: dict[str, dict[str, Any]] = {
        key: {"count": 0, "success_count": 0, "failure_count": 0, "latencies": []}
        for key in ("feed", "query", "respond", "runtime_action")
    }
    tokens_for_rate = 0
    latency_seconds_for_rate = 0.0

    for episode in episode_records:
        bucket_name = _endpoint_bucket_name(episode.operation)
        bucket = endpoint_accumulators.setdefault(
            bucket_name,
            {"count": 0, "success_count": 0, "failure_count": 0, "latencies": []},
        )
        bucket["count"] += 1
        succeeded = episode.status == "succeeded" and not episode.failure
        if succeeded:
            bucket["success_count"] += 1
        else:
            bucket["failure_count"] += 1
        latency = _safe_float(episode.latency_ms)
        if latency is not None:
            bucket["latencies"].append(latency)
        token_count = 0
        for source in (episode.actual_output, episode.action, episode.request):
            for key in ("tokens_processed", "token_count", "tokens_trained", "token_delta"):
                if key in source:
                    try:
                        token_count = max(token_count, int(source.get(key, 0) or 0))
                    except (TypeError, ValueError):
                        pass
        if token_count > 0 and latency is not None and latency > 0.0:
            tokens_for_rate += token_count
            latency_seconds_for_rate += latency / 1000.0

    action_success_count = 0
    action_failure_count = 0
    for action in action_records:
        bucket = endpoint_accumulators["runtime_action"]
        bucket["count"] += 1
        success = action.verification.success or action.verification.status == VerificationStatus.VERIFIED
        failure = (
            action.execution_status == ActionExecutionStatus.FAILED
            or action.verification.contradiction
            or action.verification.status == VerificationStatus.CONTRADICTED
        )
        if success:
            action_success_count += 1
            bucket["success_count"] += 1
        if failure:
            action_failure_count += 1
            bucket["failure_count"] += 1

    endpoint_latency_ms = {
        name: _latency_summary(
            int(values["count"]),
            int(values["success_count"]),
            int(values["failure_count"]),
            values["latencies"],
        )
        for name, values in sorted(endpoint_accumulators.items())
    }

    runtime_tokens_per_second = _safe_float(runtime_data.get("tokens_per_second"))
    if tokens_for_rate > 0 and latency_seconds_for_rate > 0.0:
        tokens_per_second = {
            "value": tokens_for_rate / latency_seconds_for_rate,
            "source": "runtime_episode_traces",
            "token_count": int(tokens_for_rate),
            "seconds": float(latency_seconds_for_rate),
        }
    elif runtime_tokens_per_second is not None and runtime_tokens_per_second > 0.0:
        tokens_per_second = {
            "value": float(runtime_tokens_per_second),
            "source": "terminus_runtime",
            "token_count": int(runtime_data.get("last_tick_token_delta", 0) or 0),
            "seconds": None,
        }
    else:
        tokens_per_second = {
            "value": None,
            "source": "unavailable",
            "token_count": 0,
            "seconds": None,
        }

    cache_summary = _extract_cache_summary(
        (retired_runtime_path_data.get("episodic_memory") or {}).get("embedder", {})
        if isinstance((retired_runtime_path_data.get("episodic_memory") or {}), Mapping)
        else {}
    )

    recommendation = world.recommended_next_action
    policy_counts = Counter({recommendation: 1}) if recommendation else Counter()
    total_actions = len(action_records)
    verification_evaluated = int(world.verification_count)

    feedback_telemetry = _coerce_feedback_telemetry(feedback_summary)
    replay_summary = _coerce_replay_sample_summary(replay_sample_summary)

    return {
        "schema_version": 1,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "sample": {
            "runtime_episode_count": int(len(episode_records)),
            "action_count": int(total_actions),
            "action_history_count": int(action_loop_data.get("actions_recorded", total_actions) or 0),
        },
        "endpoint_latency_ms": endpoint_latency_ms,
        "tokens_per_second": tokens_per_second,
        "memory": _memory_counter_summary(memory, runtime_memory),
        "retired_external_adapter": _extract_retired_external_adapter_summary(retired_runtime_path_data),
        "cache": cache_summary,
        "action_success": {
            "action_count": int(total_actions),
            "success_count": int(action_success_count),
            "failure_count": int(action_failure_count),
            "success_rate": _safe_ratio(action_success_count, total_actions),
        },
        "verification_success": {
            "evaluated_count": int(verification_evaluated),
            "success_count": int(world.verified_action_count),
            "contradicted_count": int(world.contradicted_action_count),
            "unverified_count": int(world.unverified_action_count),
            "success_rate": float(world.verification_success_rate),
        },
        "feedback": feedback_telemetry,
        "replay_sample_summary": replay_summary,
        "policy_recommendations": {
            "total": int(sum(policy_counts.values())),
            "latest": recommendation,
            "counts": dict(policy_counts),
            "outcomes": {
                "information_gain": float(world.information_gain),
                "goal_progress": float(world.goal_progress),
                "risk": float(world.risk),
                "uncertainty": float(world.uncertainty),
                "evaluated_prediction_count": int(world.evaluated_prediction_count),
                "fulfilled_count": int(world.fulfilled_count),
                "contradicted_count": int(world.contradicted_count),
                "pending_count": int(world.pending_count),
            },
        },
    }


# ---------------------------------------------------------------------------
# Operational Self-Model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperationalSelfModel:
    model_id: str
    generated_at: str
    token_count: int
    state_revision: int
    configured: bool
    running: bool
    provenance: ProvenanceState
    predictions: tuple[PredictionRecord, ...] = ()
    actions: tuple[ActionExecutionRecord, ...] = ()
    consolidations: tuple[ConsolidationRecord, ...] = ()
    runtime_episodes: tuple[RuntimeEpisodeTrace, ...] = ()
    action_loop: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    narrative: dict[str, Any] = field(default_factory=dict)
    retired_runtime_path: dict[str, Any] = field(default_factory=dict)
    world_model_lite: WorldModelLiteSummary | None = None
    skill_memories: tuple[SkillMemoryRecord, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "OperationalSelfModel":
        data = dict(payload or {})
        predictions = tuple(
            PredictionRecord.from_payload(item) for item in list(data.get("predictions") or [])
        )
        actions = tuple(
            ActionExecutionRecord.from_payload(item) for item in list(data.get("actions") or [])
        )
        consolidations = tuple(
            ConsolidationRecord.from_payload(item) for item in list(data.get("consolidations") or [])
        )
        runtime_episodes = tuple(
            RuntimeEpisodeTrace.from_payload(item)
            for item in list(data.get("runtime_episodes") or [])
            if isinstance(item, Mapping)
        )
        episode_predictions = tuple(
            prediction
            for prediction in (episode.prediction_record() for episode in runtime_episodes)
            if prediction is not None
        )
        if episode_predictions:
            seen_prediction_ids = {item.prediction_id for item in predictions}
            predictions = predictions + tuple(
                item for item in episode_predictions if item.prediction_id not in seen_prediction_ids
            )
        skill_memories = tuple(
            SkillMemoryRecord.from_payload(item)
            for item in list(data.get("skill_memories") or [])
            if isinstance(item, Mapping)
        )
        provenance = ProvenanceState.from_payload(_as_mapping(data.get("provenance")))
        world_model_lite_payload = data.get("world_model_lite")
        world_model_lite = (
            WorldModelLiteSummary.from_payload(world_model_lite_payload)
            if isinstance(world_model_lite_payload, Mapping)
            else WorldModelLiteSummary.from_records(
                predictions=predictions,
                actions=actions,
                consolidations=consolidations,
                action_loop=_as_mapping(data.get("action_loop")),
            )
        )
        return cls(
            model_id=_clean_text(data.get("model_id"))
            or _stable_id(
                "osm",
                data.get("token_count"),
                data.get("state_revision"),
                [item.prediction_id for item in predictions],
                [item.consolidation_id for item in consolidations],
                [item.episode_id for item in runtime_episodes],
            ),
            generated_at=_clean_text(data.get("generated_at"))
            or datetime.now(timezone.utc).isoformat(),
            token_count=int(data.get("token_count", 0) or 0),
            state_revision=int(data.get("state_revision", 0) or 0),
            configured=bool(data.get("configured", False)),
            running=bool(data.get("running", False)),
            provenance=provenance,
            predictions=predictions,
            actions=actions,
            consolidations=consolidations,
            runtime_episodes=runtime_episodes,
            action_loop=dict(data.get("action_loop") or {}),
            memory=dict(data.get("memory") or {}),
            narrative=dict(data.get("narrative") or {}),
            retired_runtime_path=dict(data.get("retired_runtime_path") or {}),
            world_model_lite=world_model_lite,
            skill_memories=skill_memories or SkillMemoryRecord.from_action_records(actions),
        )

    @classmethod
    def build(
        cls,
        *,
        token_count: int,
        state_revision: int,
        configured: bool,
        running: bool,
        provenance: ProvenanceState,
        predictions: Sequence[PredictionRecord] = (),
        actions: Sequence[ActionExecutionRecord] = (),
        consolidations: Sequence[ConsolidationRecord] = (),
        runtime_episodes: Sequence[RuntimeEpisodeTrace] = (),
        action_loop: Mapping[str, Any] | None = None,
        memory: Mapping[str, Any] | None = None,
        narrative: Mapping[str, Any] | None = None,
        retired_runtime_path: Mapping[str, Any] | None = None,
        generated_at: str | None = None,
    ) -> "OperationalSelfModel":
        runtime_episode_records = tuple(runtime_episodes)
        episode_predictions = tuple(
            prediction
            for prediction in (episode.prediction_record() for episode in runtime_episode_records)
            if prediction is not None
        )
        explicit_prediction_records = tuple(predictions)
        seen_prediction_ids = {item.prediction_id for item in explicit_prediction_records}
        prediction_records = explicit_prediction_records + tuple(
            item for item in episode_predictions if item.prediction_id not in seen_prediction_ids
        )
        action_records = tuple(actions)
        consolidation_records = tuple(consolidations)
        world_model_lite = WorldModelLiteSummary.from_records(
            predictions=prediction_records,
            actions=action_records,
            consolidations=consolidation_records,
            action_loop=action_loop,
        )
        model_id = _stable_id(
            "osm",
            int(token_count),
            int(state_revision),
            [item.prediction_id for item in prediction_records],
            [item.action_id for item in action_records],
            [item.consolidation_id for item in consolidation_records],
            [item.episode_id for item in runtime_episode_records],
            provenance.to_payload()["distribution"],
        )
        return cls(
            model_id=model_id,
            generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
            token_count=int(token_count),
            state_revision=int(state_revision),
            configured=bool(configured),
            running=bool(running),
            provenance=provenance,
            predictions=prediction_records,
            actions=action_records,
            consolidations=consolidation_records,
            runtime_episodes=runtime_episode_records,
            action_loop=dict(action_loop or {}),
            memory=dict(memory or {}),
            narrative=dict(narrative or {}),
            retired_runtime_path=dict(retired_runtime_path or {}),
            world_model_lite=world_model_lite,
            skill_memories=SkillMemoryRecord.from_action_records(action_records),
        )

    def _surface_skill_memories(self) -> tuple[SkillMemoryRecord, ...]:
        return self.skill_memories or SkillMemoryRecord.from_action_records(self.actions)

    def _supported_action_names(
        self, skill_memories: Sequence[SkillMemoryRecord]
    ) -> tuple[str, ...]:
        raw_supported = self.action_loop.get("supported_actions")
        supported_values = (
            list(raw_supported)
            if isinstance(raw_supported, Sequence) and not isinstance(raw_supported, (str, bytes))
            else []
        )
        return tuple(
            sorted(
                set(
                    _limited_unique_clean_text(
                        [
                            *supported_values,
                            *[memory.tool for memory in skill_memories],
                            *[action.action_type for action in self.actions],
                        ],
                        limit=_MAX_SUPPORTED_ACTIONS,
                        lower=True,
                    )
                )
            )
        )

    def _surface_capabilities(
        self,
        *,
        skill_memories: Sequence[SkillMemoryRecord],
        world_model_lite: WorldModelLiteSummary,
    ) -> list[str]:
        capabilities: list[str] = []
        if self.configured:
            capabilities.append("runtime_configured")
        if self.running:
            capabilities.append("runtime_running")
        if self._supported_action_names(skill_memories):
            capabilities.append("digital_action_execution")
        if self.predictions or self.actions:
            capabilities.append("prediction_tracking")
        if self.runtime_episodes:
            capabilities.append("runtime_episode_trace")
        if world_model_lite.verification_count > 0:
            capabilities.append("verification_tracking")
        if self.consolidations:
            capabilities.append("delayed_consequence_consolidation_tracking")
        if self.memory:
            capabilities.append("episodic_memory_snapshot")
        if self.retired_runtime_path:
            capabilities.append("retired_runtime_path_snapshot")
        capabilities.append("world_model_lite_policy_scoring")
        return capabilities

    def _surface_tools(
        self, skill_memories: Sequence[SkillMemoryRecord]
    ) -> list[dict[str, Any]]:
        supported = self._supported_action_names(skill_memories)
        by_tool = {memory.tool: memory for memory in skill_memories}
        tools: list[dict[str, Any]] = []
        for name in supported:
            memory = by_tool.get(name)
            tools.append(
                {
                    "name": name,
                    "observed_action_count": int(memory.action_count) if memory else 0,
                    "success_rate": float(memory.success_rate) if memory else 0.0,
                    "last_used_at": memory.last_used_at if memory else "",
                    "status": memory.status if memory else "unobserved",
                }
            )
        return tools

    def _actions_recorded_count(self) -> int:
        """Number of actions recorded in the action loop, with safe fallback."""
        try:
            return max(
                0, int(self.action_loop.get("actions_recorded", len(self.actions)) or 0)
            )
        except (TypeError, ValueError):
            return len(self.actions)

    def _surface_limits(
        self, skill_memories: Sequence[SkillMemoryRecord]
    ) -> dict[str, Any]:
        memory_capacity = self.memory.get("capacity")
        memory_fill_ratio = self.memory.get("fill_ratio")
        actions_recorded = self._actions_recorded_count()
        return {
            "supported_actions": list(self._supported_action_names(skill_memories)),
            "snapshot_action_count": int(len(self.actions)),
            "runtime_episode_count": int(len(self.runtime_episodes)),
            "action_history_count": int(actions_recorded),
            "action_history_truncated": bool(actions_recorded > len(self.actions)),
            "memory_capacity": memory_capacity if memory_capacity is not None else None,
            "memory_fill_ratio": (
                float(memory_fill_ratio)
                if isinstance(memory_fill_ratio, (int, float))
                else None
            ),
            "state_revision": int(self.state_revision),
            "configured": bool(self.configured),
            "running": bool(self.running),
        }

    def _surface_budgets(self, world_model_lite: WorldModelLiteSummary) -> dict[str, Any]:
        actions_recorded = self._actions_recorded_count()
        memory_size = self.memory.get("size", self.memory.get("memory_count"))
        memory_capacity = self.memory.get("capacity")
        fill_ratio = self.memory.get("fill_ratio")
        fill_value = float(fill_ratio) if isinstance(fill_ratio, (int, float)) else None
        return {
            "action_history_used": int(actions_recorded),
            "action_snapshot_used": int(len(self.actions)),
            "runtime_episode_snapshot_used": int(len(self.runtime_episodes)),
            "policy_budget_use": float(world_model_lite.budget_use),
            "policy_cost": float(world_model_lite.cost),
            "policy_risk": float(world_model_lite.risk),
            "policy_uncertainty": float(world_model_lite.uncertainty),
            "memory_size": memory_size if isinstance(memory_size, (int, float)) else None,
            "memory_capacity": (
                memory_capacity if isinstance(memory_capacity, (int, float)) else None
            ),
            "memory_fill_ratio": fill_value,
        }

    def _surface_recent_failures(self) -> list[dict[str, Any]]:
        """Return up to _MAX_RECENT_FAILURES failed actions and episodes."""
        failures: list[dict[str, Any]] = []
        for action in self.actions:
            if not (
                action.execution_status == ActionExecutionStatus.FAILED
                or action.verification.contradiction
                or action.verification.status == VerificationStatus.CONTRADICTED
            ):
                continue
            failures.append(
                {
                    "action_id": action.action_id,
                    "action_type": action.action_type,
                    "verification_status": action.verification.status.value,
                    "execution_status": action.execution_status.value,
                    "summary": action.verification.summary,
                    "actual_outcome": action.actual_outcome,
                    "recorded_at": action.recorded_at,
                    "topics": list(action.topics or action.prediction.topics),
                }
            )
            if len(failures) >= _MAX_RECENT_FAILURES:
                break
        if len(failures) < _MAX_RECENT_FAILURES:
            for episode in self.runtime_episodes:
                if episode.status != "failed" and not episode.failure:
                    continue
                episode_prediction = episode.prediction_record()
                failures.append(
                    {
                        "episode_id": episode.episode_id,
                        "operation": episode.operation,
                        "verification_status": str(episode.verification.get("status", "unknown")),
                        "summary": str((episode.failure or {}).get("message", "")),
                        "actual_outcome": str(episode.actual_output.get("summary", "")),
                        "recorded_at": episode.completed_at or episode.created_at,
                        "topics": (
                            [] if episode_prediction is None else list(episode_prediction.topics)
                        ),
                    }
                )
                if len(failures) >= _MAX_RECENT_FAILURES:
                    break
        return failures

    def _surface_uncertain_domains(self) -> list[dict[str, Any]]:
        """Return up to _MAX_UNCERTAIN_DOMAINS domains ranked by uncertain-signal count."""
        domains: dict[str, dict[str, int]] = {}

        def _domain_bucket(name: str) -> dict[str, int]:
            key = _clean_text(name).lower() or "unknown"
            return domains.setdefault(
                key,
                {
                    "pending_predictions": 0,
                    "unknown_predictions": 0,
                    "unverified_actions": 0,
                    "contradictions": 0,
                },
            )

        for prediction in self.predictions:
            prediction_topics = prediction.topics or (
                (_clean_text(prediction.source_id) or "prediction"),
            )
            if prediction.status == PredictionStatus.PENDING:
                for topic in prediction_topics:
                    _domain_bucket(topic)["pending_predictions"] += 1
            elif prediction.status == PredictionStatus.UNKNOWN:
                for topic in prediction_topics:
                    _domain_bucket(topic)["unknown_predictions"] += 1
            elif prediction.status == PredictionStatus.CONTRADICTED:
                for topic in prediction_topics:
                    _domain_bucket(topic)["contradictions"] += 1

        for action in self.actions:
            action_topics = action.topics or action.prediction.topics or (action.action_type,)
            if action.verification.status in {
                VerificationStatus.UNKNOWN,
                VerificationStatus.UNVERIFIED,
            }:
                for topic in action_topics:
                    _domain_bucket(topic)["unverified_actions"] += 1
            if (
                action.verification.contradiction
                or action.verification.status == VerificationStatus.CONTRADICTED
            ):
                for topic in action_topics:
                    _domain_bucket(topic)["contradictions"] += 1

        ranked = sorted(
            domains.items(),
            key=lambda item: (-sum(item[1].values()), item[0]),
        )
        return [
            {
                "domain": domain,
                **counts,
                "total_uncertain_signals": int(sum(counts.values())),
            }
            for domain, counts in ranked[:_MAX_UNCERTAIN_DOMAINS]
            if sum(counts.values()) > 0
        ]

    def _surface_memory_health(self) -> dict[str, Any]:
        size = self.memory.get("size", self.memory.get("memory_count"))
        capacity = self.memory.get("capacity")
        fill_ratio = self.memory.get("fill_ratio")
        fill_value = float(fill_ratio) if isinstance(fill_ratio, (int, float)) else None
        if fill_value is None:
            status = "no_capacity_snapshot" if self.memory else "no_memory_snapshot"
        elif fill_value >= 0.90:
            status = "capacity_pressure"
        else:
            status = "available"
        return {
            "status": status,
            "size": int(size) if isinstance(size, int) else size,
            "capacity": int(capacity) if isinstance(capacity, int) else capacity,
            "fill_ratio": fill_value,
            "total_stored": self.memory.get("total_stored"),
            "total_evicted": self.memory.get("total_evicted"),
            "mean_confidence": self.memory.get("mean_confidence"),
            "provenance_distribution": dict(self.memory.get("provenance_distribution") or {}),
        }

    def _surface_grounding_health(
        self, world_model_lite: WorldModelLiteSummary
    ) -> dict[str, Any]:
        evidence_count = sum(
            len(action.verification.evidence) for action in self.actions
        )
        if world_model_lite.verification_count <= 0 and evidence_count <= 0:
            status = "no_grounding_observed"
        elif world_model_lite.contradicted_action_count > 0 or world_model_lite.contradicted_count > 0:
            status = "contradictions_present"
        elif world_model_lite.unverified_action_count > 0 or world_model_lite.pending_count > 0:
            status = "needs_verification"
        else:
            status = "grounded"
        return {
            "status": status,
            "verification_count": int(world_model_lite.verification_count),
            "verified_action_count": int(world_model_lite.verified_action_count),
            "contradicted_action_count": int(world_model_lite.contradicted_action_count),
            "unverified_action_count": int(world_model_lite.unverified_action_count),
            "verification_success_rate": float(world_model_lite.verification_success_rate),
            "contradiction_rate": float(world_model_lite.contradiction_rate),
            "evidence_count": int(evidence_count),
            "verified_memory_count": int(self.provenance.verified),
            "contradicted_memory_count": int(self.provenance.contradicted),
        }

    def to_payload(self) -> dict[str, Any]:
        world_model_lite = self.world_model_lite or WorldModelLiteSummary.from_records(
            predictions=self.predictions,
            actions=self.actions,
            consolidations=self.consolidations,
            action_loop=self.action_loop,
        )
        skill_memories = self._surface_skill_memories()
        return {
            "model_id": self.model_id,
            "generated_at": self.generated_at,
            "token_count": int(self.token_count),
            "state_revision": int(self.state_revision),
            "configured": bool(self.configured),
            "running": bool(self.running),
            "provenance": self.provenance.to_payload(),
            "prediction_count": int(len(self.predictions)),
            "action_count": int(len(self.actions)),
            "consolidation_count": int(len(self.consolidations)),
            "runtime_episode_count": int(len(self.runtime_episodes)),
            "predictions": [item.to_payload() for item in self.predictions],
            "actions": [item.to_payload() for item in self.actions],
            "consolidations": [item.to_payload() for item in self.consolidations],
            "runtime_episodes": [item.to_payload() for item in self.runtime_episodes],
            "world_model_lite": world_model_lite.to_payload(),
            "benchmark_telemetry": build_runtime_benchmark_telemetry(
                runtime_episodes=self.runtime_episodes,
                actions=self.actions,
                world_model_lite=world_model_lite,
                action_loop=self.action_loop,
                memory=self.memory,
                retired_runtime_path=self.retired_runtime_path,
                generated_at=self.generated_at,
            ),
            "capabilities": self._surface_capabilities(
                skill_memories=skill_memories,
                world_model_lite=world_model_lite,
            ),
            "limits": self._surface_limits(skill_memories),
            "tools": self._surface_tools(skill_memories),
            "budgets": self._surface_budgets(world_model_lite),
            "recent_failures": self._surface_recent_failures(),
            "uncertain_domains": self._surface_uncertain_domains(),
            "memory_health": self._surface_memory_health(),
            "grounding_health": self._surface_grounding_health(world_model_lite),
            "skill_memories": [item.to_payload() for item in skill_memories],
            "action_loop": dict(self.action_loop),
            "memory": dict(self.memory),
            "narrative": dict(self.narrative),
            "retired_runtime_path": dict(self.retired_runtime_path),
        }
