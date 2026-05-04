from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from hecsn.service.living_loop_helpers import (  # used throughout this module
    _as_mapping,
    _clean_text,
    _clamp01,
    _coerce_world_model_lite,
    _enum_value,
    _latest_text,
    _limited_unique_clean_text,
    _provenance_value,
    _safe_float,
    _safe_ratio,
    _stable_id,
    _verification_status_from_payload,
)
from hecsn.service.living_loop_records import (  # re-exported for backward compatibility
    ActionExecutionRecord,
    ActionExecutionStatus,
    ActionVerificationRecord,
    ConsolidationRecord,
    ConsolidationStatus,
    PredictionRecord,
    PredictionStatus,
    ProvenanceState,
    RuntimeEpisodeTrace,
    SkillMemoryRecord,
    VerificationStatus,
)


@dataclass(frozen=True)
class PolicyScore:
    information_gain: float = 0.0
    goal_progress: float = 0.0
    cost: float = 0.0
    risk: float = 0.0
    budget_use: float = 0.0
    uncertainty: float = 1.0
    recommended_next_action: str = "observe_or_execute_grounded_action"
    components: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "PolicyScore":
        data = dict(payload or {})
        return cls(
            information_gain=_clamp01(data.get("information_gain", 0.0)),
            goal_progress=_clamp01(data.get("goal_progress", 0.0)),
            cost=_clamp01(data.get("cost", 0.0)),
            risk=_clamp01(data.get("risk", 0.0)),
            budget_use=_clamp01(data.get("budget_use", 0.0)),
            uncertainty=_clamp01(data.get("uncertainty", 1.0)),
            recommended_next_action=_clean_text(data.get("recommended_next_action"))
            or "observe_or_execute_grounded_action",
            components=dict(data.get("components") or {}),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "information_gain": float(self.information_gain),
            "goal_progress": float(self.goal_progress),
            "cost": float(self.cost),
            "risk": float(self.risk),
            "budget_use": float(self.budget_use),
            "uncertainty": float(self.uncertainty),
            "recommended_next_action": self.recommended_next_action,
            "components": dict(self.components),
        }


@dataclass(frozen=True)
class WorldModelLiteSummary:
    prediction_count: int = 0
    fulfilled_count: int = 0
    contradicted_count: int = 0
    pending_count: int = 0
    unknown_count: int = 0
    evaluated_prediction_count: int = 0
    prediction_accuracy: float = 0.0
    verification_count: int = 0
    verified_action_count: int = 0
    contradicted_action_count: int = 0
    unverified_action_count: int = 0
    verification_success_rate: float = 0.0
    contradiction_rate: float = 0.0
    information_gain: float = 0.0
    goal_progress: float = 0.0
    cost: float = 0.0
    risk: float = 0.0
    budget_use: float = 0.0
    uncertainty: float = 1.0
    recommended_next_action: str = "observe_or_execute_grounded_action"
    policy_score: PolicyScore = field(default_factory=PolicyScore)
    components: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "WorldModelLiteSummary":
        data = dict(payload or {})
        policy_payload = data.get("policy_score") if isinstance(data.get("policy_score"), Mapping) else data
        policy_score = PolicyScore.from_payload(policy_payload)
        return cls(
            prediction_count=max(0, int(data.get("prediction_count", 0) or 0)),
            fulfilled_count=max(0, int(data.get("fulfilled_count", 0) or 0)),
            contradicted_count=max(0, int(data.get("contradicted_count", 0) or 0)),
            pending_count=max(0, int(data.get("pending_count", 0) or 0)),
            unknown_count=max(0, int(data.get("unknown_count", 0) or 0)),
            evaluated_prediction_count=max(0, int(data.get("evaluated_prediction_count", 0) or 0)),
            prediction_accuracy=_clamp01(data.get("prediction_accuracy", 0.0)),
            verification_count=max(0, int(data.get("verification_count", 0) or 0)),
            verified_action_count=max(0, int(data.get("verified_action_count", 0) or 0)),
            contradicted_action_count=max(0, int(data.get("contradicted_action_count", 0) or 0)),
            unverified_action_count=max(0, int(data.get("unverified_action_count", 0) or 0)),
            verification_success_rate=_clamp01(data.get("verification_success_rate", 0.0)),
            contradiction_rate=_clamp01(data.get("contradiction_rate", 0.0)),
            information_gain=policy_score.information_gain,
            goal_progress=policy_score.goal_progress,
            cost=policy_score.cost,
            risk=policy_score.risk,
            budget_use=policy_score.budget_use,
            uncertainty=policy_score.uncertainty,
            recommended_next_action=policy_score.recommended_next_action,
            policy_score=policy_score,
            components=dict(data.get("components") or {}),
        )

    @classmethod
    def from_records(
        cls,
        *,
        predictions: Sequence[PredictionRecord] = (),
        actions: Sequence[ActionExecutionRecord] = (),
        consolidations: Sequence[ConsolidationRecord] = (),
        action_loop: Mapping[str, Any] | None = None,
    ) -> "WorldModelLiteSummary":
        unique_predictions: dict[str, PredictionRecord] = {}
        for prediction in predictions:
            unique_predictions[prediction.prediction_id] = prediction
        for action in actions:
            unique_predictions.setdefault(action.prediction.prediction_id, action.prediction)
        prediction_records = tuple(unique_predictions.values())
        action_records = tuple(actions)
        consolidation_records = tuple(consolidations)

        prediction_count = len(prediction_records)
        fulfilled_count = sum(1 for item in prediction_records if item.status == PredictionStatus.FULFILLED)
        contradicted_count = sum(1 for item in prediction_records if item.status == PredictionStatus.CONTRADICTED)
        pending_count = sum(1 for item in prediction_records if item.status == PredictionStatus.PENDING)
        unknown_count = sum(1 for item in prediction_records if item.status == PredictionStatus.UNKNOWN)
        evaluated_prediction_count = fulfilled_count + contradicted_count
        prediction_accuracy = _safe_ratio(fulfilled_count, evaluated_prediction_count)
        contradiction_rate = _safe_ratio(contradicted_count, prediction_count)

        verification_records = tuple(
            action.verification for action in action_records if action.verification.status != VerificationStatus.UNKNOWN
        )
        verification_count = len(verification_records)
        verified_action_count = sum(
            1 for item in verification_records if item.success or item.status == VerificationStatus.VERIFIED
        )
        contradicted_action_count = sum(
            1 for item in verification_records if item.contradiction or item.status == VerificationStatus.CONTRADICTED
        )
        unverified_action_count = sum(
            1 for item in verification_records if item.status == VerificationStatus.UNVERIFIED
        )
        verification_success_rate = _safe_ratio(verified_action_count, verification_count)
        evidence_count = sum(len(item.evidence) for item in verification_records)

        credit_events = sum(max(0, item.credit_events) for item in consolidation_records)
        penalty_events = sum(max(0, item.penalty_events) for item in consolidation_records)
        forgiveness_events = sum(max(0, item.forgiveness_events) for item in consolidation_records)
        consolidation_events = credit_events + penalty_events + forgiveness_events
        aggregate_count = sum(max(1, item.aggregate_count) for item in consolidation_records)
        trajectory_positive = sum(max(0.0, item.trajectory_net_score) for item in consolidation_records)
        penalty_ratio = _safe_ratio(penalty_events, consolidation_events)

        information_units = (
            fulfilled_count
            + contradicted_count
            + verified_action_count
            + min(evidence_count, max(1, len(action_records)))
            + credit_events
            + forgiveness_events
            + min(trajectory_positive, float(max(1, len(consolidation_records))))
        )
        information_opportunities = (
            prediction_count
            + verification_count
            + max(1, len(consolidation_records))
            + credit_events
            + penalty_events
            + forgiveness_events
        )
        information_gain = _clamp01(_safe_ratio(information_units, max(1, information_opportunities * 2)))

        progress_components: list[float] = []
        if prediction_count > 0:
            progress_components.append(_safe_ratio(fulfilled_count, prediction_count))
        if verification_count > 0:
            progress_components.append(verification_success_rate)
        if consolidation_events > 0:
            progress_components.append(
                _clamp01(
                    0.5
                    + _safe_ratio(
                        credit_events + forgiveness_events - penalty_events,
                        consolidation_events * 2,
                    )
                )
            )
        goal_progress = _clamp01(
            _safe_ratio(sum(progress_components), len(progress_components))
            if progress_components
            else 0.0
        )

        pending_rate = _safe_ratio(pending_count, prediction_count)
        unknown_rate = _safe_ratio(unknown_count, prediction_count)
        confidence_values = [
            _clamp01(item.confidence)
            for item in prediction_records
            if item.confidence > 0.0
        ] + [
            _clamp01(item.confidence)
            for item in verification_records
            if item.confidence > 0.0
        ]
        confidence_uncertainty = (
            1.0 - _safe_ratio(sum(confidence_values), len(confidence_values))
            if confidence_values
            else (1.0 if prediction_count or verification_count else 0.0)
        )
        uncertainty = (
            1.0
            if not prediction_records and not action_records and not consolidation_records
            else _clamp01(
                (0.45 * pending_rate) + (0.25 * unknown_rate) + (0.30 * confidence_uncertainty)
            )
        )
        risk = _clamp01((0.50 * contradiction_rate) + (0.25 * pending_rate) + (0.25 * penalty_ratio))

        loop_data = dict(action_loop or {})
        actions_recorded = max(0, int(loop_data.get("actions_recorded", len(action_records)) or 0))
        recent_window_capacity = max(8, len(action_records), actions_recorded)
        budget_use = _clamp01(
            _safe_ratio(actions_recorded if actions_recorded else len(action_records), recent_window_capacity)
        )
        cost = _clamp01(_safe_ratio(float(len(action_records)) + (float(aggregate_count) / 4.0), 8.0))

        if contradicted_count > 0 or contradicted_action_count > 0:
            recommended_next_action = "investigate_contradictions"
        elif pending_count > 0 or unverified_action_count > 0:
            recommended_next_action = "verify_pending_predictions"
        elif not prediction_records and not action_records:
            recommended_next_action = "observe_or_execute_grounded_action"
        elif uncertainty >= 0.60:
            recommended_next_action = "gather_more_evidence"
        elif risk >= 0.40:
            recommended_next_action = "reduce_risk_before_next_action"
        elif goal_progress >= 0.70 and information_gain >= 0.40:
            recommended_next_action = "continue_current_policy"
        else:
            recommended_next_action = "execute_low_risk_grounded_action"

        components = {
            "evaluated_prediction_count": int(evaluated_prediction_count),
            "evidence_count": int(evidence_count),
            "credit_events": int(credit_events),
            "penalty_events": int(penalty_events),
            "forgiveness_events": int(forgiveness_events),
            "aggregate_count": int(aggregate_count),
            "trajectory_positive": float(trajectory_positive),
            "pending_rate": float(pending_rate),
            "unknown_rate": float(unknown_rate),
            "penalty_ratio": float(penalty_ratio),
            "confidence_uncertainty": float(confidence_uncertainty),
            "recent_window_capacity": int(recent_window_capacity),
            "actions_recorded": int(actions_recorded),
        }
        policy_score = PolicyScore(
            information_gain=information_gain,
            goal_progress=goal_progress,
            cost=cost,
            risk=risk,
            budget_use=budget_use,
            uncertainty=uncertainty,
            recommended_next_action=recommended_next_action,
            components=components,
        )
        return cls(
            prediction_count=prediction_count,
            fulfilled_count=fulfilled_count,
            contradicted_count=contradicted_count,
            pending_count=pending_count,
            unknown_count=unknown_count,
            evaluated_prediction_count=evaluated_prediction_count,
            prediction_accuracy=prediction_accuracy,
            verification_count=verification_count,
            verified_action_count=verified_action_count,
            contradicted_action_count=contradicted_action_count,
            unverified_action_count=unverified_action_count,
            verification_success_rate=verification_success_rate,
            contradiction_rate=contradiction_rate,
            information_gain=information_gain,
            goal_progress=goal_progress,
            cost=cost,
            risk=risk,
            budget_use=budget_use,
            uncertainty=uncertainty,
            recommended_next_action=recommended_next_action,
            policy_score=policy_score,
            components=components,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "prediction_count": int(self.prediction_count),
            "fulfilled_count": int(self.fulfilled_count),
            "contradicted_count": int(self.contradicted_count),
            "pending_count": int(self.pending_count),
            "unknown_count": int(self.unknown_count),
            "evaluated_prediction_count": int(self.evaluated_prediction_count),
            "prediction_accuracy": float(self.prediction_accuracy),
            "verification_count": int(self.verification_count),
            "verified_action_count": int(self.verified_action_count),
            "contradicted_action_count": int(self.contradicted_action_count),
            "unverified_action_count": int(self.unverified_action_count),
            "verification_success_rate": float(self.verification_success_rate),
            "contradiction_rate": float(self.contradiction_rate),
            "information_gain": float(self.information_gain),
            "goal_progress": float(self.goal_progress),
            "cost": float(self.cost),
            "risk": float(self.risk),
            "budget_use": float(self.budget_use),
            "uncertainty": float(self.uncertainty),
            "recommended_next_action": self.recommended_next_action,
            "policy_score": self.policy_score.to_payload(),
            "components": dict(self.components),
        }


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


def _latency_summary(count: int, success_count: int, failure_count: int, latencies: Sequence[float]) -> dict[str, Any]:
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


def _extract_nim_summary(cortex: Mapping[str, Any] | None) -> dict[str, Any]:
    cortex_data = dict(cortex or {})
    episodic = cortex_data.get("episodic_memory") if isinstance(cortex_data.get("episodic_memory"), Mapping) else {}
    embedder = episodic.get("embedder") if isinstance(episodic, Mapping) and isinstance(episodic.get("embedder"), Mapping) else {}
    chat_generations = int(cortex_data.get("thoughts_generated", 0) or 0) + int(cortex_data.get("dreams_generated", 0) or 0)
    embedding_calls = int(embedder.get("nim_calls", 0) or 0)
    rate_limit_hits = int(embedder.get("rate_limit_hits", 0) or 0)
    return {
        "available": bool(cortex_data.get("enabled", False) or embedder.get("available", False)),
        "chat_generations_observed": int(chat_generations),
        "embedding_nim_calls": int(embedding_calls),
        "observed_call_count": int(chat_generations + embedding_calls),
        "calls_per_minute": None,
        "calls_per_minute_reason": "timestamps_unavailable",
        "rate_limit_hits": int(rate_limit_hits),
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
        "source": "runtime_memory_store" if runtime_memory else ("cortex_episodic_memory" if memory else "unavailable"),
    }


def _coerce_feedback_telemetry(feedback_summary: Mapping[str, Any] | None) -> dict[str, Any]:
    data = dict(feedback_summary or {})
    status_counts = data.get("status_counts") if isinstance(data.get("status_counts"), Mapping) else {}
    verdict_counts = data.get("verdict_counts") if isinstance(data.get("verdict_counts"), Mapping) else {}
    target_counts = data.get("target_counts") if isinstance(data.get("target_counts"), Mapping) else {}
    recent_feedback = (
        [dict(item) for item in list(data.get("recent_feedback") or []) if isinstance(item, Mapping)]
        if isinstance(data.get("recent_feedback"), Sequence) and not isinstance(data.get("recent_feedback"), (str, bytes))
        else []
    )

    def _count(key: str) -> int:
        try:
            return max(0, int(data.get(f"{key}_count", status_counts.get(key, 0)) or 0))
        except (TypeError, ValueError):
            return 0

    def _verdict_count(key: str) -> int:
        try:
            return max(0, int(verdict_counts.get(key, 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _target_count(key: str) -> int:
        try:
            return max(0, int(target_counts.get(key, 0) or 0))
        except (TypeError, ValueError):
            return 0

    feedback_count = data.get("feedback_count", data.get("count"))
    try:
        total = max(0, int(feedback_count or 0))
    except (TypeError, ValueError):
        total = 0
    return {
        "feedback_count": int(total),
        "verified_count": _count("verified"),
        "contradicted_count": _count("contradicted"),
        "unverified_count": _count("unverified"),
        "status_counts": {
            "verified": _count("verified"),
            "contradicted": _count("contradicted"),
            "unverified": _count("unverified"),
        },
        "verdict_counts": {
            "verified": _verdict_count("verified"),
            "contradicted": _verdict_count("contradicted"),
            "unverified": _verdict_count("unverified"),
        },
        "target_counts": {
            "runtime_episode": _target_count("runtime_episode"),
            "action": _target_count("action"),
        },
        "recent_feedback": recent_feedback,
        "latest_feedback_at": _clean_text(data.get("latest_feedback_at")),
        "grounding_impact": _clean_text(data.get("grounding_impact")) or "none",
    }


POLICY_ACTUATOR_SCHEMA_VERSION = 1
POLICY_ACTUATOR_HIGH_LATENCY_AVG_MS = 1000.0
POLICY_ACTUATOR_HIGH_LATENCY_MAX_MS = 1500.0
REPLAY_PLAN_SCHEMA_VERSION = 1
REPLAY_PLAN_PRIORITY_RULES_VERSION = "deterministic-v1"
REPLAY_PLAN_DEFAULT_LIMIT = 20
REPLAY_PLAN_MAX_LIMIT = 50


@dataclass(frozen=True)
class PolicyActuatorRecommendation:
    recommendation: str
    action: str
    reasons: tuple[dict[str, str], ...]
    risk: float
    expected_information_gain: float
    expected_goal_progress: float
    expected_cost: float
    uncertainty: float
    target_episode_id: str | None = None
    target_action_id: str | None = None
    suggested_endpoint: str = "/terminus/living-loop"
    suggested_input: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    schema_version: int = POLICY_ACTUATOR_SCHEMA_VERSION
    advisory: bool = True
    executable: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "recommendation": self.recommendation,
            "action": self.action,
            "reasons": [dict(item) for item in self.reasons],
            "risk": float(self.risk),
            "expected_information_gain": float(self.expected_information_gain),
            "expected_goal_progress": float(self.expected_goal_progress),
            "expected_cost": float(self.expected_cost),
            "uncertainty": float(self.uncertainty),
            "advisory": True,
            "executable": False,
            "target_episode_id": self.target_episode_id,
            "target_action_id": self.target_action_id,
            "action_id": self.target_action_id,
            "suggested_endpoint": self.suggested_endpoint,
            "suggested_input": dict(self.suggested_input),
            "input": dict(self.suggested_input),
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(),
        }


def _policy_count(data: Mapping[str, Any], key: str) -> int:
    try:
        return max(0, int(data.get(key, 0) or 0))
    except (TypeError, ValueError):
        return 0


def _policy_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _policy_float(*values: Any) -> float:
    for value in values:
        result = _safe_float(value)
        if result is not None:
            return result
    return 0.0


def _policy_max_float(*values: Any) -> float:
    return max((_safe_float(value) or 0.0 for value in values), default=0.0)


def _policy_recent_feedback_target(
    feedback: Mapping[str, Any],
    statuses: set[str],
) -> tuple[str | None, str | None]:
    recent = feedback.get("recent_feedback")
    if not isinstance(recent, Sequence) or isinstance(recent, (str, bytes)):
        return None, None
    for item in recent:
        if not isinstance(item, Mapping):
            continue
        verdict = _clean_text(item.get("verdict")).lower()
        applied = _clean_text(item.get("applied_status")).lower()
        if verdict not in statuses and applied not in statuses:
            continue
        target_type = _clean_text(item.get("target_type")).lower()
        target_id = _clean_text(item.get("target_id"))
        if target_type == "action" and target_id:
            return None, target_id
        if target_type == "runtime_episode" and target_id:
            return target_id, None
    return None, None


def _policy_first_action_target(living_loop: Mapping[str, Any], statuses: set[str]) -> str | None:
    actions = living_loop.get("actions")
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        return None
    for item in actions:
        action = _policy_mapping(item)
        verification = _policy_mapping(action.get("verification"))
        status = _clean_text(verification.get("status")).lower()
        contradiction = bool(verification.get("contradiction", False))
        if status in statuses or ("contradicted" in statuses and contradiction):
            action_id = _clean_text(action.get("action_id"))
            if action_id:
                return action_id
    return None


def _policy_first_episode_target(living_loop: Mapping[str, Any], statuses: set[str]) -> str | None:
    episodes = living_loop.get("runtime_episodes")
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)):
        return None
    for item in episodes:
        episode = _policy_mapping(item)
        verification = _policy_mapping(episode.get("verification"))
        status = _clean_text(verification.get("status")).lower()
        failed = _clean_text(episode.get("status")).lower() == "failed" or bool(episode.get("failure"))
        prediction = _policy_mapping(episode.get("prediction"))
        prediction_status = _clean_text(prediction.get("status")).lower()
        if status in statuses or prediction_status in statuses or ("contradicted" in statuses and failed):
            episode_id = _clean_text(episode.get("episode_id"))
            if episode_id:
                return episode_id
    return None


def _policy_latency_pressure(benchmark: Mapping[str, Any]) -> tuple[float, dict[str, Any] | None]:
    endpoint_latency = benchmark.get("endpoint_latency_ms")
    if not isinstance(endpoint_latency, Mapping):
        return 0.0, None
    highest_pressure = 0.0
    highest: dict[str, Any] | None = None
    for endpoint, raw_stats in endpoint_latency.items():
        stats = _policy_mapping(raw_stats)
        avg_ms = _safe_float(stats.get("avg_ms"))
        max_ms = _safe_float(stats.get("max_ms"))
        pressure = max(
            0.0 if avg_ms is None else avg_ms / POLICY_ACTUATOR_HIGH_LATENCY_AVG_MS,
            0.0 if max_ms is None else max_ms / POLICY_ACTUATOR_HIGH_LATENCY_MAX_MS,
        )
        if pressure > highest_pressure:
            highest_pressure = pressure
            highest = {
                "endpoint": _clean_text(endpoint) or "unknown",
                "avg_ms": avg_ms,
                "max_ms": max_ms,
            }
    return _clamp01(highest_pressure), highest


def _policy_suggested_endpoint_and_input(
    action: str,
    *,
    target_episode_id: str | None,
    target_action_id: str | None,
) -> tuple[str, dict[str, Any]]:
    target_type = "action" if target_action_id else ("runtime_episode" if target_episode_id else "")
    target_id = target_action_id or target_episode_id or ""
    if action == "investigate_contradictions":
        return (
            "/terminus/living-loop",
            {"focus": "contradictions", "target_type": target_type, "target_id": target_id},
        )
    if action == "verify_pending_evidence":
        return (
            "/terminus/runtime-feedback",
            {
                "target_type": target_type,
                "target_id": target_id,
                "verdict": "verified_or_contradicted_after_review",
                "note": "operator_review_required",
            },
        )
    if action == "consolidate_or_sleep":
        return (
            "/terminus/cortex/sleep",
            {"reason": "policy_actuator_advisory_memory_or_fatigue_pressure"},
        )
    if action == "reduce_scope_or_wait":
        return (
            "/terminus",
            {"operator_note": "reduce source scope, defer new requests, or wait for pressure to clear"},
        )
    if action == "collect_more_evidence":
        return (
            "/terminus/living-loop",
            {"focus": "uncertain_domains", "operator_note": "collect evidence before changing policy"},
        )
    return "/terminus", {"operator_note": "healthy grounded state; continue observing current policy"}


def build_policy_actuator_status(
    living_loop: Mapping[str, Any],
    *,
    cortex_snapshot: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> PolicyActuatorRecommendation:
    """Build an advisory-only policy recommendation without executing any action."""
    loop = dict(living_loop or {})
    world = _policy_mapping(loop.get("world_model_lite"))
    policy_score = _policy_mapping(world.get("policy_score"))
    feedback = _coerce_feedback_telemetry(loop.get("feedback_summary") if isinstance(loop.get("feedback_summary"), Mapping) else {})
    benchmark = _policy_mapping(loop.get("benchmark_telemetry"))
    memory_health = _policy_mapping(loop.get("memory_health"))
    benchmark_memory = _policy_mapping(benchmark.get("memory"))
    budgets = _policy_mapping(loop.get("budgets"))
    grounding_health = _policy_mapping(loop.get("grounding_health"))
    cortex = dict(cortex_snapshot or _policy_mapping(loop.get("cortex")))
    drives = _policy_mapping(cortex.get("drives"))

    information_gain = _clamp01(_policy_max_float(world.get("information_gain"), policy_score.get("information_gain")))
    goal_progress = _clamp01(_policy_max_float(world.get("goal_progress"), policy_score.get("goal_progress")))
    policy_cost = _clamp01(_policy_max_float(world.get("cost"), policy_score.get("cost"), budgets.get("policy_cost")))
    budget_use = _clamp01(_policy_max_float(world.get("budget_use"), policy_score.get("budget_use"), budgets.get("policy_budget_use")))
    risk = _clamp01(_policy_max_float(world.get("risk"), policy_score.get("risk"), budgets.get("policy_risk")))
    uncertainty = _clamp01(
        max(
            _policy_max_float(world.get("uncertainty"), policy_score.get("uncertainty"), budgets.get("policy_uncertainty")),
            _policy_float(drives.get("uncertainty")),
        )
    )
    memory_fill = _clamp01(
        max(
            _policy_float(memory_health.get("fill_ratio"), memory_health.get("fill_fraction")),
            _policy_float(benchmark_memory.get("fill_ratio"), benchmark_memory.get("fill_fraction")),
            _policy_float(cortex.get("memory_fill_ratio")),
        )
    )
    fatigue = _clamp01(_policy_float(drives.get("fatigue")))
    latency_pressure, latency_detail = _policy_latency_pressure(benchmark)

    reasons: list[dict[str, str]] = []
    target_episode_id: str | None = None
    target_action_id: str | None = None

    contradicted_feedback = _policy_count(feedback, "contradicted_count")
    contradicted_predictions = _policy_count(world, "contradicted_count")
    contradicted_actions = _policy_count(world, "contradicted_action_count")
    grounding_contradicted = _clean_text(grounding_health.get("status")).lower() == "contradictions_present"
    failed_episode_id = _policy_first_episode_target(loop, {"contradicted"})
    if contradicted_feedback > 0 or contradicted_predictions > 0 or contradicted_actions > 0 or grounding_contradicted or failed_episode_id:
        target_episode_id, target_action_id = _policy_recent_feedback_target(feedback, {"contradicted"})
        target_action_id = target_action_id or _policy_first_action_target(loop, {"contradicted"})
        target_episode_id = target_episode_id or failed_episode_id
        if contradicted_feedback > 0:
            reasons.append(
                {
                    "code": "contradicted_feedback",
                    "detail": f"{contradicted_feedback} contradicted feedback item(s) require investigation.",
                }
            )
        if contradicted_actions > 0:
            reasons.append(
                {
                    "code": "contradicted_actions",
                    "detail": f"{contradicted_actions} contradicted action(s) are present.",
                }
            )
        if contradicted_predictions > 0:
            reasons.append(
                {
                    "code": "contradicted_predictions",
                    "detail": f"{contradicted_predictions} contradicted prediction(s) are present.",
                }
            )
        if failed_episode_id:
            reasons.append(
                {
                    "code": "contradicted_episodes",
                    "detail": f"Runtime episode {failed_episode_id} is failed or contradicted.",
                }
            )
        action = "investigate_contradictions"
    else:
        unverified_feedback = _policy_count(feedback, "unverified_count")
        pending_predictions = _policy_count(world, "pending_count")
        unverified_actions = _policy_count(world, "unverified_action_count")
        pending_episode_id = _policy_first_episode_target(loop, {"pending", "unverified"})
        if unverified_feedback > 0 or pending_predictions > 0 or unverified_actions > 0 or pending_episode_id:
            target_episode_id, target_action_id = _policy_recent_feedback_target(feedback, {"unverified"})
            target_action_id = target_action_id or _policy_first_action_target(loop, {"unverified", "unknown"})
            target_episode_id = target_episode_id or pending_episode_id
            if unverified_feedback > 0:
                reasons.append(
                    {
                        "code": "unverified_feedback",
                        "detail": f"{unverified_feedback} unverified feedback item(s) need evidence.",
                    }
                )
            if unverified_actions > 0:
                reasons.append(
                    {
                        "code": "unverified_actions",
                        "detail": f"{unverified_actions} unverified action(s) need evidence.",
                    }
                )
            if pending_predictions > 0:
                reasons.append(
                    {
                        "code": "pending_predictions",
                        "detail": f"{pending_predictions} pending prediction(s) need verification.",
                    }
                )
            if pending_episode_id:
                reasons.append(
                    {
                        "code": "pending_runtime_episode",
                        "detail": f"Runtime episode {pending_episode_id} has pending or unverified evidence.",
                    }
                )
            action = "verify_pending_evidence"
        elif memory_fill >= 0.90 or fatigue >= 0.70 or bool(cortex.get("is_sleeping", False)) or _clean_text(cortex.get("current_mode")).lower() == "sleeping":
            if memory_fill >= 0.90:
                reasons.append(
                    {
                        "code": "memory_capacity_pressure",
                        "detail": f"Memory fill is {memory_fill:.2f}, at or above 0.90.",
                    }
                )
            if fatigue >= 0.70 or bool(cortex.get("is_sleeping", False)) or _clean_text(cortex.get("current_mode")).lower() == "sleeping":
                reasons.append(
                    {
                        "code": "fatigue_sleep_pressure",
                        "detail": f"Cortex fatigue/sleep pressure is {fatigue:.2f}.",
                    }
                )
            action = "consolidate_or_sleep"
        elif latency_pressure >= 1.0 or policy_cost >= 0.80 or budget_use >= 0.80:
            if latency_pressure >= 1.0 and latency_detail:
                detail_parts = [
                    f"endpoint={latency_detail.get('endpoint')}",
                    f"avg_ms={latency_detail.get('avg_ms')}",
                    f"max_ms={latency_detail.get('max_ms')}",
                ]
                reasons.append({"code": "high_latency", "detail": ", ".join(detail_parts)})
            if policy_cost >= 0.80:
                reasons.append({"code": "high_cost", "detail": f"Policy cost is {policy_cost:.2f}."})
            if budget_use >= 0.80:
                reasons.append({"code": "high_budget_use", "detail": f"Policy budget use is {budget_use:.2f}."})
            action = "reduce_scope_or_wait"
        else:
            uncertain_domains = [
                item
                for item in list(loop.get("uncertain_domains") or [])
                if isinstance(item, Mapping) and int(item.get("total_uncertain_signals", 0) or 0) > 0
            ]
            unknown_predictions = _policy_count(world, "unknown_count")
            if uncertainty >= 0.60 or unknown_predictions > 0 or uncertain_domains:
                if uncertainty >= 0.60:
                    reasons.append({"code": "high_uncertainty", "detail": f"Policy uncertainty is {uncertainty:.2f}."})
                if unknown_predictions > 0:
                    reasons.append(
                        {
                            "code": "unknown_predictions",
                            "detail": f"{unknown_predictions} unknown prediction(s) need evidence.",
                        }
                    )
                if uncertain_domains:
                    domain = _clean_text(uncertain_domains[0].get("domain")) or "unknown"
                    reasons.append({"code": "uncertain_domains", "detail": f"Most uncertain domain is {domain}."})
                action = "collect_more_evidence"
            else:
                reasons.append(
                    {
                        "code": "healthy_grounded_state",
                        "detail": "No contradiction, pending evidence, memory, cost, or uncertainty pressure crossed thresholds.",
                    }
                )
                action = "continue_current_policy"

    recommendation_by_action = {
        "investigate_contradictions": "Investigate contradicted feedback, actions, or episodes before changing policy.",
        "verify_pending_evidence": "Verify pending evidence before trusting the next policy step.",
        "consolidate_or_sleep": "Consolidate memory or request sleep before collecting more evidence.",
        "reduce_scope_or_wait": "Reduce scope or wait because latency, cost, or budget pressure is high.",
        "collect_more_evidence": "Collect more evidence for uncertain domains before acting.",
        "continue_current_policy": "Continue the current grounded policy without executing a new action.",
    }
    expected_by_action = {
        "investigate_contradictions": (0.80, 0.60, 0.35, 0.75),
        "verify_pending_evidence": (0.72, 0.55, 0.25, 0.45),
        "consolidate_or_sleep": (0.45, 0.45, 0.20, 0.30),
        "reduce_scope_or_wait": (0.25, 0.40, 0.15, 0.25),
        "collect_more_evidence": (0.70, 0.50, 0.30, 0.40),
        "continue_current_policy": (0.30, max(0.50, goal_progress), 0.10, risk),
    }
    base_information, base_progress, base_cost, base_risk = expected_by_action[action]
    suggested_endpoint, suggested_input = _policy_suggested_endpoint_and_input(
        action,
        target_episode_id=target_episode_id,
        target_action_id=target_action_id,
    )
    if not reasons:
        reasons.append({"code": action, "detail": recommendation_by_action[action]})
    return PolicyActuatorRecommendation(
        recommendation=recommendation_by_action[action],
        action=action,
        reasons=tuple(reasons),
        risk=_clamp01(max(risk, base_risk, 0.45 if memory_fill >= 0.90 else 0.0, latency_pressure * 0.35)),
        expected_information_gain=_clamp01(max(information_gain, base_information)),
        expected_goal_progress=_clamp01(max(goal_progress, base_progress)),
        expected_cost=_clamp01(max(policy_cost, base_cost, latency_pressure * 0.50)),
        uncertainty=_clamp01(uncertainty),
        target_episode_id=target_episode_id,
        target_action_id=target_action_id,
        suggested_endpoint=suggested_endpoint,
        suggested_input=suggested_input,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
    )


@dataclass(frozen=True)
class ReplayCandidate:
    candidate_id: str
    rank: int
    target_type: str
    target_id: str
    target_ids: tuple[str, ...]
    operation: str
    created_at: str
    completed_at: str
    reason_codes: tuple[str, ...]
    priority_score: float
    priority_components: dict[str, float]
    suggested_consolidation_action: str
    suggested_endpoint: str
    suggested_input: dict[str, Any]
    summary: str
    provenance: dict[str, Any]
    risk: float
    uncertainty: float
    latency: dict[str, Any]
    memory_health: dict[str, Any]
    feedback: dict[str, Any]
    policy: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "rank": int(self.rank),
            "target_type": self.target_type,
            "target_id": self.target_id,
            "target_ids": list(self.target_ids),
            "operation": self.operation,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "reason_codes": list(self.reason_codes),
            "priority_score": float(self.priority_score),
            "priority_components": dict(self.priority_components),
            "suggested_consolidation_action": self.suggested_consolidation_action,
            "suggested_endpoint": self.suggested_endpoint,
            "suggested_input": dict(self.suggested_input),
            "summary": self.summary,
            "provenance": dict(self.provenance),
            "risk": float(self.risk),
            "uncertainty": float(self.uncertainty),
            "latency": dict(self.latency),
            "memory_health": dict(self.memory_health),
            "feedback": dict(self.feedback),
            "policy": dict(self.policy),
        }


REPLAY_SAMPLE_SAFETY_BOUNDARIES: tuple[str, ...] = (
    "no_training",
    "no_sleep",
    "no_memory_verification_promotion",
    "no_feedback_posting",
    "no_digital_action_execution",
    "no_external_calls",
    "audit_only",
)


def replay_candidate_safety_flags(candidate: Mapping[str, Any]) -> dict[str, Any]:
    reason_codes = {
        _clean_text(item).lower()
        for item in candidate.get("reason_codes", [])
        if _clean_text(item)
    }
    feedback = candidate.get("feedback") if isinstance(candidate.get("feedback"), Mapping) else {}
    provenance = candidate.get("provenance") if isinstance(candidate.get("provenance"), Mapping) else {}
    provenance_text = " ".join(
        _clean_text(value).lower()
        for value in (
            provenance.get("provenance"),
            provenance.get("source"),
            provenance.get("kind"),
            candidate.get("provenance"),
        )
        if _clean_text(value)
    )
    target_status = _clean_text(candidate.get("status") or candidate.get("verification_status")).lower()
    action = _clean_text(candidate.get("suggested_consolidation_action")).lower()
    contradicted = (
        bool(reason_codes & {"contradicted_feedback", "contradicted_runtime_episode", "contradicted_action"})
        or action == "review_contradiction"
        or int(feedback.get("contradicted_count", 0) or 0) > 0
        or target_status == "contradicted"
    )
    failed = bool(reason_codes & {"failed_runtime", "failed_action", "failed_runtime_episode"}) or target_status == "failed"
    dreamed_or_synthetic = any(marker in provenance_text for marker in ("dream", "synthetic", "simulated"))
    return {
        "audit_only": True,
        "not_promoted": True,
        "promoted_to_verified_fact": False,
        "non_factual": bool(contradicted or failed or dreamed_or_synthetic),
        "negative_lesson": bool(contradicted or failed),
        "dreamed_or_synthetic": bool(dreamed_or_synthetic),
        "safety_boundaries": list(REPLAY_SAMPLE_SAFETY_BOUNDARIES),
    }


def _default_replay_sample_safety_flags() -> dict[str, Any]:
    return {
        "audit_only": True,
        "operator_confirmed": False,
        "training_started": False,
        "sleep_started": False,
        "memory_verification_promoted": False,
        "feedback_posted": False,
        "digital_action_executed": False,
        "external_calls_made": False,
        "memory_mutated": False,
        "state_revision_mutated": False,
        "token_count_mutated": False,
        "action_history_mutated": False,
        "feedback_mutated": False,
        "not_promoted": True,
    }


def _coerce_replay_sample_summary(summary: Mapping[str, Any] | None) -> dict[str, Any]:
    data = dict(summary or {}) if isinstance(summary, Mapping) else {}
    mode_counts = data.get("mode_counts") if isinstance(data.get("mode_counts"), Mapping) else {}
    status_counts = data.get("status_counts") if isinstance(data.get("status_counts"), Mapping) else {}
    latest = data.get("latest_history_item") if isinstance(data.get("latest_history_item"), Mapping) else None
    safety_flags = data.get("safety_flags") if isinstance(data.get("safety_flags"), Mapping) else {}
    if not safety_flags and isinstance(latest, Mapping) and isinstance(latest.get("safety_flags"), Mapping):
        safety_flags = latest["safety_flags"]
    normalized_safety_flags = {**_default_replay_sample_safety_flags(), **dict(safety_flags)}
    return {
        "schema_version": int(data.get("schema_version", 1) or 1),
        "endpoint": str(data.get("endpoint") or "/terminus/replay-sample"),
        "execution_endpoint": str(data.get("execution_endpoint") or "/terminus/replay-execute"),
        "history_endpoint": str(data.get("history_endpoint") or "/terminus/replay-sample/history"),
        "execution_history_endpoint": str(data.get("execution_history_endpoint") or "/terminus/replay-execute/history"),
        "count": int(data.get("count", data.get("history_count", 0)) or 0),
        "history_count": int(data.get("history_count", data.get("count", 0)) or 0),
        "selected_count": int(data.get("selected_count", 0) or 0),
        "latest_selected_count": int(data.get("latest_selected_count", 0) or 0),
        "mode_counts": {str(key): int(value or 0) for key, value in dict(mode_counts).items()},
        "status_counts": {str(key): int(value or 0) for key, value in dict(status_counts).items()},
        "latest_history_item": dict(latest) if isinstance(latest, Mapping) else None,
        "safety_flags": normalized_safety_flags,
        "safety_boundaries": list(data.get("safety_boundaries") or REPLAY_SAMPLE_SAFETY_BOUNDARIES),
        "audit_only": True,
        "advisory": True,
        "executable": False,
    }


@dataclass(frozen=True)
class ReplayPlan:
    generated_at: str
    limit: int
    state_revision: int
    token_count: int
    snapshot_counts: dict[str, int]
    priority_weights: dict[str, float]
    plan_reason_codes: tuple[str, ...]
    candidates: tuple[ReplayCandidate, ...]
    schema_version: int = REPLAY_PLAN_SCHEMA_VERSION
    advisory: bool = True
    executable: bool = False
    endpoint: str = "/terminus/replay-plan"
    priority_rules_version: str = REPLAY_PLAN_PRIORITY_RULES_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "generated_at": self.generated_at,
            "advisory": True,
            "executable": False,
            "endpoint": self.endpoint,
            "limit": int(self.limit),
            "count": int(len(self.candidates)),
            "state_revision": int(self.state_revision),
            "token_count": int(self.token_count),
            "snapshot_counts": dict(self.snapshot_counts),
            "priority_rules_version": self.priority_rules_version,
            "priority_weights": dict(self.priority_weights),
            "plan_reason_codes": list(self.plan_reason_codes),
            "candidates": [candidate.to_payload() for candidate in self.candidates],
        }


REPLAY_PLAN_PRIORITY_WEIGHTS: dict[str, float] = {
    "safety": 100.0,
    "feedback": 70.0,
    "uncertainty": 55.0,
    "memory_pressure": 45.0,
    "latency_pressure": 35.0,
    "policy_pressure": 25.0,
    "provenance_gap": 15.0,
    "recency_rank": 10.0,
}

REPLAY_REASON_PRECEDENCE: dict[str, int] = {
    "contradicted_feedback": 0,
    "contradicted_runtime_episode": 1,
    "contradicted_action": 2,
    "failed_runtime_episode": 3,
    "corrected_output_available": 4,
    "unverified_feedback": 5,
    "pending_prediction": 6,
    "unverified_action": 7,
    "high_uncertainty": 8,
    "uncertain_domain": 9,
    "memory_capacity_pressure": 10,
    "fatigue_sleep_pressure": 11,
    "high_latency": 12,
    "high_cost": 13,
    "high_budget_use": 14,
    "healthy_grounded_state": 15,
}


def _replay_sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _replay_recent_feedback_for_target(feedback: Mapping[str, Any], target_type: str, target_id: str) -> list[dict[str, Any]]:
    recent = _replay_sequence(feedback.get("recent_feedback"))
    return [
        dict(item)
        for item in recent
        if isinstance(item, Mapping)
        and _clean_text(item.get("target_type")).lower() == target_type
        and _clean_text(item.get("target_id")) == target_id
    ]


def _replay_feedback_entries(target: Mapping[str, Any], feedback: Mapping[str, Any], target_type: str, target_id: str) -> list[dict[str, Any]]:
    entries = [dict(item) for item in _replay_sequence(target.get("feedback")) if isinstance(item, Mapping)]
    entries.extend(_replay_recent_feedback_for_target(feedback, target_type, target_id))
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        key = _clean_text(entry.get("feedback_id")) or _stable_id(
            "fb", target_type, target_id, entry.get("created_at"), entry.get("verdict"), entry.get("summary")
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _replay_feedback_summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter({"verified": 0, "contradicted": 0, "unverified": 0})
    has_corrected_output = False
    latest_summary = ""
    latest_at = ""
    total_confidence = 0.0
    for entry in entries:
        verdict = _clean_text(entry.get("verdict")).lower()
        applied = _clean_text(entry.get("applied_status")).lower()
        status = applied if applied in {"verified", "contradicted", "unverified"} else verdict
        if status not in {"verified", "contradicted", "unverified"}:
            status = "unverified"
        status_counts[status] += 1
        has_corrected_output = has_corrected_output or bool(entry.get("has_corrected_output")) or entry.get("corrected_output") is not None
        created_at = _clean_text(entry.get("created_at"))
        if created_at >= latest_at:
            latest_at = created_at
            latest_summary = _clean_text(entry.get("summary")) or latest_summary
        total_confidence += _clamp01(entry.get("confidence"))
    total = sum(status_counts.values())
    return {
        "feedback_count": int(total),
        "verified_count": int(status_counts["verified"]),
        "contradicted_count": int(status_counts["contradicted"]),
        "unverified_count": int(status_counts["unverified"]),
        "has_corrected_output": bool(has_corrected_output),
        "latest_feedback_at": latest_at,
        "latest_summary": latest_summary,
        "mean_confidence": _safe_ratio(total_confidence, total),
    }


def _replay_latency_pressure(latency_ms: Any, benchmark: Mapping[str, Any], operation: str) -> tuple[float, dict[str, Any]]:
    latency_value = _safe_float(latency_ms)
    pressure = 0.0 if latency_value is None else latency_value / POLICY_ACTUATOR_HIGH_LATENCY_AVG_MS
    latency = {
        "latency_ms": latency_value,
        "pressure": _clamp01(pressure),
        "source": "target",
    }
    endpoint_latency = benchmark.get("endpoint_latency_ms")
    bucket = _policy_mapping(endpoint_latency.get(operation)) if isinstance(endpoint_latency, Mapping) else {}
    avg_ms = _safe_float(bucket.get("avg_ms"))
    max_ms = _safe_float(bucket.get("max_ms"))
    if avg_ms is not None or max_ms is not None:
        bucket_pressure = max(
            0.0 if avg_ms is None else avg_ms / POLICY_ACTUATOR_HIGH_LATENCY_AVG_MS,
            0.0 if max_ms is None else max_ms / POLICY_ACTUATOR_HIGH_LATENCY_MAX_MS,
        )
        if bucket_pressure > pressure:
            pressure = bucket_pressure
            latency = {
                "latency_ms": latency_value,
                "avg_ms": avg_ms,
                "max_ms": max_ms,
                "pressure": _clamp01(bucket_pressure),
                "source": "endpoint_latency_ms",
            }
    return _clamp01(pressure), latency


def _replay_memory_pressure(memory_health: Mapping[str, Any], cortex: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    fill = _policy_float(memory_health.get("fill_ratio"), memory_health.get("fill_fraction"))
    drives = _policy_mapping(cortex.get("drives"))
    fatigue = _policy_float(drives.get("fatigue"))
    sleeping = bool(cortex.get("is_sleeping", False)) or _clean_text(cortex.get("current_mode")).lower() == "sleeping"
    pressure = max(fill, fatigue, 1.0 if sleeping else 0.0)
    return _clamp01(pressure), {
        "status": _clean_text(memory_health.get("status")) or "unknown",
        "fill_ratio": fill,
        "fatigue": fatigue,
        "is_sleeping": sleeping,
        "pressure": _clamp01(pressure),
    }


def _replay_policy_pressure(policy: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    action = _clean_text(policy.get("action"))
    reasons = _replay_sequence(policy.get("reasons"))
    reason_codes = [
        _clean_text(item.get("code")).lower()
        for item in reasons
        if isinstance(item, Mapping) and _clean_text(item.get("code"))
    ]
    if not reason_codes:
        reason_codes = [
            _clean_text(item).lower()
            for item in _replay_sequence(policy.get("reason_codes"))
            if _clean_text(item)
        ]
    pressure = 0.0 if action in {"", "continue_current_policy"} else 1.0
    pressure = max(
        pressure,
        _policy_float(policy.get("risk")),
        _policy_float(policy.get("uncertainty")),
        _policy_float(policy.get("expected_cost")) * 0.5,
    )
    return _clamp01(pressure), {
        "action": action,
        "recommendation": _clean_text(policy.get("recommendation")),
        "reason_codes": reason_codes,
        "pressure": _clamp01(pressure),
        "advisory": bool(policy.get("advisory", True)),
        "executable": bool(policy.get("executable", False)),
    }


def _replay_timestamp_sort_value(value: str) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _replay_reason_precedence(reason_codes: Sequence[str]) -> int:
    return min((REPLAY_REASON_PRECEDENCE.get(code, 99) for code in reason_codes), default=99)


def _replay_unique_reasons(reasons: Sequence[str]) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        code = _clean_text(reason).lower()
        if not code or code in seen:
            continue
        seen.add(code)
        cleaned.append(code)
    return tuple(cleaned)


def _replay_action_for_reasons(reason_codes: Sequence[str]) -> str:
    reason_set = set(reason_codes)
    if reason_set & {"contradicted_feedback", "contradicted_runtime_episode", "contradicted_action", "failed_runtime_episode", "corrected_output_available"}:
        return "review_contradiction"
    if reason_set & {"unverified_feedback", "pending_prediction", "unverified_action"}:
        return "verify_pending_evidence"
    if reason_set & {"memory_capacity_pressure", "fatigue_sleep_pressure"}:
        return "sleep_consolidation_advisory"
    if reason_set & {"high_latency", "high_cost", "high_budget_use"}:
        return "reduce_scope_or_wait"
    if reason_set & {"high_uncertainty", "uncertain_domain"}:
        return "collect_more_evidence"
    if "healthy_grounded_state" in reason_set:
        return "continue_observing"
    return "replay_episode_for_grounding"


def _replay_endpoint_for_action(action: str) -> str:
    if action in {"review_contradiction", "verify_pending_evidence"}:
        return "/terminus/runtime-feedback"
    if action == "sleep_consolidation_advisory":
        return "/terminus/cortex/sleep"
    if action == "replay_episode_for_grounding":
        return "/terminus/runtime-traces/export"
    if action == "reduce_scope_or_wait":
        return "/terminus"
    return "/terminus/living-loop"


def _replay_priority_score(components: Mapping[str, float]) -> float:
    return round(
        sum(float(REPLAY_PLAN_PRIORITY_WEIGHTS[key]) * _clamp01(components.get(key, 0.0)) for key in REPLAY_PLAN_PRIORITY_WEIGHTS),
        6,
    )


def _replay_candidate(
    *,
    target_type: str,
    target_id: str,
    target_ids: Sequence[str] = (),
    operation: str,
    created_at: str,
    completed_at: str = "",
    reason_codes: Sequence[str],
    components: Mapping[str, float],
    summary: str,
    provenance: Mapping[str, Any] | None = None,
    risk: float = 0.0,
    uncertainty: float = 0.0,
    latency: Mapping[str, Any] | None = None,
    memory_health: Mapping[str, Any] | None = None,
    feedback: Mapping[str, Any] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> ReplayCandidate:
    normalized_reasons = _replay_unique_reasons(reason_codes)
    action = _replay_action_for_reasons(normalized_reasons)
    suggested_endpoint = _replay_endpoint_for_action(action)
    target_values = _limited_unique_clean_text([target_id, *target_ids], limit=16)
    suggested_input = {
        "target_type": target_type,
        "target_id": target_id,
        "target_ids": list(target_values),
        "operation": operation,
        "reason_codes": list(normalized_reasons),
        "operator_review_required": action in {"review_contradiction", "verify_pending_evidence"},
    }
    return ReplayCandidate(
        candidate_id=_stable_id("replay", target_type, target_id, operation, normalized_reasons, created_at),
        rank=0,
        target_type=target_type,
        target_id=target_id,
        target_ids=target_values,
        operation=operation,
        created_at=created_at,
        completed_at=completed_at,
        reason_codes=normalized_reasons,
        priority_score=_replay_priority_score(components),
        priority_components={key: _clamp01(components.get(key, 0.0)) for key in REPLAY_PLAN_PRIORITY_WEIGHTS},
        suggested_consolidation_action=action,
        suggested_endpoint=suggested_endpoint,
        suggested_input=suggested_input,
        summary=summary or f"{target_type} {target_id} should be considered for replay.",
        provenance=dict(provenance or {}),
        risk=_clamp01(risk),
        uncertainty=_clamp01(uncertainty),
        latency=dict(latency or {}),
        memory_health=dict(memory_health or {}),
        feedback=dict(feedback or {}),
        policy=dict(policy or {}),
    )


def _replay_rank_candidates(candidates: Sequence[ReplayCandidate], *, limit: int) -> tuple[ReplayCandidate, ...]:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate.priority_score),
            _replay_reason_precedence(candidate.reason_codes),
            -_replay_timestamp_sort_value(candidate.created_at or candidate.completed_at),
            candidate.target_type,
            candidate.target_id,
        ),
    )
    return tuple(
        ReplayCandidate(
            **{
                **candidate.__dict__,
                "rank": index + 1,
            }
        )
        for index, candidate in enumerate(ranked[:limit])
    )


def _replay_prediction_status(data: Mapping[str, Any]) -> str:
    return _clean_text(data.get("status") or data.get("prediction_status")).lower()


def _replay_verification_status(data: Mapping[str, Any]) -> str:
    verification = _policy_mapping(data.get("verification"))
    return _clean_text(verification.get("status")).lower()


def _replay_target_summary(data: Mapping[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, Mapping):
            for nested_key in ("summary", "response_text", "predicted_outcome", "predicted_output", "proposed_answer"):
                text = _clean_text(value.get(nested_key))
                if text:
                    return text
        else:
            text = _clean_text(value)
            if text:
                return text
    return default


def build_replay_plan(
    living_loop: Mapping[str, Any],
    *,
    limit: int = REPLAY_PLAN_DEFAULT_LIMIT,
    created_at: str | None = None,
) -> ReplayPlan:
    """Build a read-only replay/consolidation plan from already-recorded runtime facts."""
    loop = dict(living_loop or {})
    count = min(REPLAY_PLAN_MAX_LIMIT, max(1, int(limit)))
    generated_at = created_at or datetime.now(timezone.utc).isoformat()
    feedback_summary = _coerce_feedback_telemetry(
        loop.get("feedback_summary") if isinstance(loop.get("feedback_summary"), Mapping) else {}
    )
    benchmark = _policy_mapping(loop.get("benchmark_telemetry"))
    memory_health = _policy_mapping(loop.get("memory_health"))
    cortex = _policy_mapping(loop.get("cortex"))
    memory_pressure, memory_context = _replay_memory_pressure(memory_health, cortex)
    policy_decision = _policy_mapping(loop.get("policy_decision") or loop.get("policy_actuator"))
    policy_pressure, policy_context = _replay_policy_pressure(policy_decision)
    world = _policy_mapping(loop.get("world_model_lite"))
    world_uncertainty = _clamp01(_policy_float(world.get("uncertainty"), _policy_mapping(world.get("policy_score")).get("uncertainty")))
    memory_reason_codes = []
    if memory_context["fill_ratio"] >= 0.90 or memory_context["status"] == "capacity_pressure":
        memory_reason_codes.append("memory_capacity_pressure")
    if memory_context["fatigue"] >= 0.70 or memory_context["is_sleeping"]:
        memory_reason_codes.append("fatigue_sleep_pressure")

    candidates: list[ReplayCandidate] = []
    episodes = [dict(item) for item in _replay_sequence(loop.get("runtime_episodes")) if isinstance(item, Mapping)]
    actions = [dict(item) for item in _replay_sequence(loop.get("actions")) if isinstance(item, Mapping)]
    predictions = [dict(item) for item in _replay_sequence(loop.get("predictions")) if isinstance(item, Mapping)]
    uncertain_domains = [dict(item) for item in _replay_sequence(loop.get("uncertain_domains")) if isinstance(item, Mapping)]

    total_targets = max(1, len(episodes) + len(actions) + len(predictions) + len(uncertain_domains))
    recency_by_id: dict[str, float] = {}
    for index, target in enumerate([*episodes, *actions, *predictions]):
        target_id = _clean_text(target.get("episode_id") or target.get("action_id") or target.get("prediction_id"))
        if target_id:
            recency_by_id[target_id] = _safe_ratio(total_targets - index, total_targets)

    for episode in episodes:
        target_id = _clean_text(episode.get("episode_id"))
        if not target_id:
            continue
        verification = _policy_mapping(episode.get("verification"))
        prediction = _policy_mapping(episode.get("prediction"))
        operation = _clean_text(episode.get("operation")).lower() or "runtime_episode"
        feedback_entries = _replay_feedback_entries(episode, feedback_summary, "runtime_episode", target_id)
        target_feedback = _replay_feedback_summary(feedback_entries)
        latency_pressure, latency_context = _replay_latency_pressure(episode.get("latency_ms"), benchmark, operation)
        reasons: list[str] = []
        status = _clean_text(episode.get("status")).lower()
        verification_status = _clean_text(verification.get("status")).lower()
        prediction_status = _replay_prediction_status(prediction)
        confidence = _clamp01(verification.get("confidence", prediction.get("confidence", 0.0)))
        if target_feedback["contradicted_count"] > 0:
            reasons.append("contradicted_feedback")
        if target_feedback["has_corrected_output"]:
            reasons.append("corrected_output_available")
        if status == "failed" or bool(episode.get("failure")):
            reasons.append("failed_runtime_episode")
        if verification_status == "contradicted" or bool(verification.get("contradiction")) or prediction_status == "contradicted":
            reasons.append("contradicted_runtime_episode")
        if target_feedback["unverified_count"] > 0:
            reasons.append("unverified_feedback")
        if verification_status in {"unverified", "unknown", ""} or prediction_status in {"pending", "unverified", "unknown"}:
            reasons.append("pending_prediction")
        if confidence and confidence < 0.40:
            reasons.append("high_uncertainty")
        if latency_pressure >= 1.0:
            reasons.append("high_latency")
        if not reasons and target_feedback["verified_count"] > 0:
            reasons.append("healthy_grounded_state")
        if not reasons:
            continue
        safety = 1.0 if set(reasons) & {"contradicted_feedback", "contradicted_runtime_episode", "failed_runtime_episode"} else 0.0
        uncertainty = _clamp01(max(world_uncertainty, 1.0 - confidence if confidence else 0.35))
        feedback_signal = _clamp01((target_feedback["contradicted_count"] + target_feedback["unverified_count"] + (1 if target_feedback["has_corrected_output"] else 0)) / 3.0)
        components = {
            "safety": safety,
            "feedback": feedback_signal,
            "uncertainty": uncertainty if set(reasons) & {"pending_prediction", "unverified_feedback", "high_uncertainty"} else uncertainty * 0.4,
            "memory_pressure": memory_pressure if memory_reason_codes else 0.0,
            "latency_pressure": latency_pressure,
            "policy_pressure": policy_pressure,
            "provenance_gap": 1.0 if verification_status in {"unverified", "unknown", ""} else 0.0,
            "recency_rank": recency_by_id.get(target_id, 0.0),
        }
        candidates.append(
            _replay_candidate(
                target_type="runtime_episode",
                target_id=target_id,
                operation=operation,
                created_at=_clean_text(episode.get("created_at")),
                completed_at=_clean_text(episode.get("completed_at")),
                reason_codes=reasons,
                components=components,
                summary=target_feedback["latest_summary"] or _replay_target_summary(episode, "actual_output", "prediction", default=f"Runtime {operation} episode."),
                provenance={"provenance": _clean_text(episode.get("provenance")) or "observed", "verification_status": verification_status},
                risk=max(safety, _policy_float(policy_decision.get("risk"))),
                uncertainty=uncertainty,
                latency=latency_context,
                memory_health=memory_context,
                feedback=target_feedback,
                policy=policy_context,
            )
        )

    for action in actions:
        target_id = _clean_text(action.get("action_id"))
        if not target_id:
            continue
        verification = _policy_mapping(action.get("verification"))
        operation = _clean_text(action.get("action_type")).lower() or "action"
        feedback_entries = _replay_feedback_entries(action, feedback_summary, "action", target_id)
        target_feedback = _replay_feedback_summary(feedback_entries)
        reasons = []
        verification_status = _clean_text(verification.get("status")).lower()
        confidence = _clamp01(verification.get("confidence", 0.0))
        if target_feedback["contradicted_count"] > 0:
            reasons.append("contradicted_feedback")
        if target_feedback["has_corrected_output"]:
            reasons.append("corrected_output_available")
        if verification_status == "contradicted" or bool(verification.get("contradiction")):
            reasons.append("contradicted_action")
        if target_feedback["unverified_count"] > 0:
            reasons.append("unverified_feedback")
        if verification_status in {"unverified", "unknown", ""}:
            reasons.append("unverified_action")
        if confidence and confidence < 0.40:
            reasons.append("high_uncertainty")
        if not reasons and target_feedback["verified_count"] > 0:
            reasons.append("healthy_grounded_state")
        if not reasons:
            continue
        safety = 1.0 if set(reasons) & {"contradicted_feedback", "contradicted_action"} else 0.0
        uncertainty = _clamp01(max(world_uncertainty, 1.0 - confidence if confidence else 0.35))
        components = {
            "safety": safety,
            "feedback": _clamp01((target_feedback["contradicted_count"] + target_feedback["unverified_count"] + (1 if target_feedback["has_corrected_output"] else 0)) / 3.0),
            "uncertainty": uncertainty if set(reasons) & {"unverified_action", "unverified_feedback", "high_uncertainty"} else uncertainty * 0.4,
            "memory_pressure": memory_pressure if memory_reason_codes else 0.0,
            "latency_pressure": 0.0,
            "policy_pressure": policy_pressure,
            "provenance_gap": 1.0 if verification_status in {"unverified", "unknown", ""} else 0.0,
            "recency_rank": recency_by_id.get(target_id, 0.0),
        }
        candidates.append(
            _replay_candidate(
                target_type="action",
                target_id=target_id,
                operation=operation,
                created_at=_clean_text(action.get("recorded_at") or action.get("created_at")),
                reason_codes=reasons,
                components=components,
                summary=target_feedback["latest_summary"] or _replay_target_summary(action, "actual_outcome", "predicted_outcome", default=f"Action {operation}."),
                provenance={"provenance": _clean_text(action.get("provenance")) or "observed", "verification_status": verification_status},
                risk=max(safety, _policy_float(policy_decision.get("risk"))),
                uncertainty=uncertainty,
                memory_health=memory_context,
                feedback=target_feedback,
                policy=policy_context,
            )
        )

    for prediction in predictions:
        target_id = _clean_text(prediction.get("prediction_id"))
        if not target_id:
            continue
        status = _replay_prediction_status(prediction)
        confidence = _clamp01(prediction.get("confidence", 0.0))
        reasons = []
        if status == "contradicted":
            reasons.append("contradicted_runtime_episode" if _clean_text(prediction.get("source_kind")).startswith("runtime_") else "contradicted_action")
        if status in {"pending", "unverified", "unknown"}:
            reasons.append("pending_prediction")
        if confidence and confidence < 0.40:
            reasons.append("high_uncertainty")
        if not reasons:
            continue
        safety = 1.0 if any(code.startswith("contradicted") for code in reasons) else 0.0
        uncertainty = _clamp01(max(world_uncertainty, 1.0 - confidence if confidence else 0.50))
        candidates.append(
            _replay_candidate(
                target_type="prediction",
                target_id=target_id,
                target_ids=[_clean_text(prediction.get("source_id"))],
                operation=_clean_text(prediction.get("source_kind")) or "prediction",
                created_at=_clean_text(prediction.get("created_at")),
                reason_codes=reasons,
                components={
                    "safety": safety,
                    "feedback": 0.0,
                    "uncertainty": uncertainty,
                    "memory_pressure": memory_pressure if memory_reason_codes else 0.0,
                    "latency_pressure": 0.0,
                    "policy_pressure": policy_pressure,
                    "provenance_gap": 1.0 if status in {"pending", "unknown"} else 0.0,
                    "recency_rank": recency_by_id.get(target_id, 0.0),
                },
                summary=_clean_text(prediction.get("predicted_outcome")) or "Prediction should be revisited.",
                provenance={"provenance": _clean_text(prediction.get("provenance")) or "inferred", "prediction_status": status},
                risk=max(safety, _policy_float(policy_decision.get("risk"))),
                uncertainty=uncertainty,
                memory_health=memory_context,
                policy=policy_context,
            )
        )

    for domain in uncertain_domains:
        target_id = _clean_text(domain.get("domain")) or "unknown"
        signal_count = _policy_count(domain, "total_uncertain_signals")
        if signal_count <= 0:
            continue
        candidates.append(
            _replay_candidate(
                target_type="uncertain_domain",
                target_id=target_id,
                operation="uncertain_domain",
                created_at=generated_at,
                reason_codes=["uncertain_domain", "high_uncertainty"],
                components={
                    "safety": 0.0,
                    "feedback": 0.0,
                    "uncertainty": _clamp01(max(world_uncertainty, min(1.0, signal_count / 4.0))),
                    "memory_pressure": memory_pressure if memory_reason_codes else 0.0,
                    "latency_pressure": 0.0,
                    "policy_pressure": policy_pressure,
                    "provenance_gap": 0.5,
                    "recency_rank": 0.25,
                },
                summary=f"Domain {target_id} has {signal_count} uncertain signal(s).",
                provenance={"provenance": "derived", "source": "living_loop.uncertain_domains"},
                risk=_policy_float(policy_decision.get("risk")),
                uncertainty=_clamp01(max(world_uncertainty, min(1.0, signal_count / 4.0))),
                memory_health=memory_context,
                policy=policy_context,
            )
        )

    if memory_reason_codes:
        candidates.append(
            _replay_candidate(
                target_type="memory_health",
                target_id=memory_context["status"],
                operation="memory_health",
                created_at=generated_at,
                reason_codes=memory_reason_codes,
                components={
                    "safety": 0.0,
                    "feedback": 0.0,
                    "uncertainty": world_uncertainty * 0.5,
                    "memory_pressure": memory_pressure,
                    "latency_pressure": 0.0,
                    "policy_pressure": policy_pressure,
                    "provenance_gap": 0.0,
                    "recency_rank": 0.5,
                },
                summary="Memory or fatigue pressure suggests advisory consolidation before adding more experiences.",
                provenance={"provenance": "derived", "source": "memory_health"},
                risk=_policy_float(policy_decision.get("risk")),
                uncertainty=world_uncertainty,
                memory_health=memory_context,
                policy=policy_context,
            )
        )

    if policy_context["action"] and policy_context["action"] != "continue_current_policy":
        policy_reasons = [
            code for code in policy_context["reason_codes"] if code in REPLAY_REASON_PRECEDENCE
        ] or ["high_uncertainty"]
        if _policy_float(policy_decision.get("expected_cost")) >= 0.80:
            policy_reasons.append("high_cost")
        candidates.append(
            _replay_candidate(
                target_type="policy_decision",
                target_id=policy_context["action"],
                operation="policy_decision",
                created_at=_clean_text(policy_decision.get("created_at")) or generated_at,
                reason_codes=policy_reasons,
                components={
                    "safety": 1.0 if "contradicted_feedback" in policy_reasons else 0.0,
                    "feedback": 0.5 if "contradicted_feedback" in policy_reasons or "unverified_feedback" in policy_reasons else 0.0,
                    "uncertainty": _policy_float(policy_decision.get("uncertainty")),
                    "memory_pressure": memory_pressure if memory_reason_codes else 0.0,
                    "latency_pressure": 1.0 if "high_latency" in policy_reasons else 0.0,
                    "policy_pressure": policy_pressure,
                    "provenance_gap": 0.25,
                    "recency_rank": 0.5,
                },
                summary=policy_context["recommendation"] or f"Policy recommends {policy_context['action']}.",
                provenance={"provenance": "derived", "source": "policy_decision"},
                risk=_policy_float(policy_decision.get("risk")),
                uncertainty=_policy_float(policy_decision.get("uncertainty")),
                memory_health=memory_context,
                policy=policy_context,
            )
        )

    if not candidates:
        candidates.append(
            _replay_candidate(
                target_type="policy_decision",
                target_id="healthy_grounded_state",
                operation="healthy_grounded_state",
                created_at=generated_at,
                reason_codes=["healthy_grounded_state"],
                components={
                    "safety": 0.0,
                    "feedback": 0.0,
                    "uncertainty": world_uncertainty * 0.25,
                    "memory_pressure": memory_pressure * 0.25,
                    "latency_pressure": 0.0,
                    "policy_pressure": 0.0,
                    "provenance_gap": 0.0,
                    "recency_rank": 0.0,
                },
                summary="No replay pressure crossed a safety, feedback, uncertainty, memory, or latency threshold.",
                provenance={"provenance": "derived", "source": "replay_plan"},
                uncertainty=world_uncertainty,
                memory_health=memory_context,
                policy=policy_context,
            )
        )

    ranked = _replay_rank_candidates(candidates, limit=count)
    plan_reasons = _replay_unique_reasons([code for candidate in ranked for code in candidate.reason_codes])
    return ReplayPlan(
        generated_at=generated_at,
        limit=count,
        state_revision=int(loop.get("state_revision", 0) or 0),
        token_count=int(loop.get("token_count", 0) or 0),
        snapshot_counts={
            "runtime_episodes": int(loop.get("runtime_episode_count", len(episodes)) or 0),
            "actions": int(loop.get("action_count", len(actions)) or 0),
            "predictions": int(loop.get("prediction_count", len(predictions)) or 0),
            "feedback": int(feedback_summary.get("feedback_count", 0) or 0),
            "uncertain_domains": int(len(uncertain_domains)),
        },
        priority_weights=dict(REPLAY_PLAN_PRIORITY_WEIGHTS),
        plan_reason_codes=plan_reasons,
        candidates=ranked,
    )


def build_runtime_benchmark_telemetry(
    *,
    runtime_episodes: Sequence[RuntimeEpisodeTrace | Mapping[str, Any]] = (),
    actions: Sequence[ActionExecutionRecord | Mapping[str, Any]] = (),
    world_model_lite: WorldModelLiteSummary | Mapping[str, Any] | None = None,
    action_loop: Mapping[str, Any] | None = None,
    memory: Mapping[str, Any] | None = None,
    runtime_memory: Mapping[str, Any] | None = None,
    cortex: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    feedback_summary: Mapping[str, Any] | None = None,
    replay_sample_summary: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Transparent benchmark telemetry from already-recorded runtime facts."""
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
            if token_count > 0 and latency > 0.0:
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
        (dict(cortex or {}).get("episodic_memory") or {}).get("embedder", {})
        if isinstance((dict(cortex or {}).get("episodic_memory") or {}), Mapping)
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
        "nim": _extract_nim_summary(cortex),
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
    cortex: dict[str, Any] = field(default_factory=dict)
    world_model_lite: WorldModelLiteSummary | None = None
    skill_memories: tuple[SkillMemoryRecord, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "OperationalSelfModel":
        data = dict(payload or {})
        predictions = tuple(PredictionRecord.from_payload(item) for item in list(data.get("predictions") or []))
        actions = tuple(ActionExecutionRecord.from_payload(item) for item in list(data.get("actions") or []))
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
        provenance = ProvenanceState.from_payload(
            data.get("provenance") if isinstance(data.get("provenance"), Mapping) else {}
        )
        world_model_lite = (
            WorldModelLiteSummary.from_payload(data.get("world_model_lite"))
            if isinstance(data.get("world_model_lite"), Mapping)
            else WorldModelLiteSummary.from_records(
                predictions=predictions,
                actions=actions,
                consolidations=consolidations,
                action_loop=data.get("action_loop") if isinstance(data.get("action_loop"), Mapping) else {},
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
            generated_at=_clean_text(data.get("generated_at")) or datetime.now(timezone.utc).isoformat(),
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
            cortex=dict(data.get("cortex") or {}),
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
        cortex: Mapping[str, Any] | None = None,
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
            cortex=dict(cortex or {}),
            world_model_lite=world_model_lite,
            skill_memories=SkillMemoryRecord.from_action_records(action_records),
        )

    def _surface_skill_memories(self) -> tuple[SkillMemoryRecord, ...]:
        return self.skill_memories or SkillMemoryRecord.from_action_records(self.actions)

    def _supported_action_names(self, skill_memories: Sequence[SkillMemoryRecord]) -> tuple[str, ...]:
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
                        limit=64,
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
        if bool(self.cortex.get("enabled", False)):
            capabilities.append("cortex_loop_snapshot")
        capabilities.append("world_model_lite_policy_scoring")
        return capabilities

    def _surface_tools(self, skill_memories: Sequence[SkillMemoryRecord]) -> list[dict[str, Any]]:
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

    def _surface_limits(self, skill_memories: Sequence[SkillMemoryRecord]) -> dict[str, Any]:
        memory_capacity = self.memory.get("capacity")
        memory_fill_ratio = self.memory.get("fill_ratio")
        try:
            actions_recorded = max(0, int(self.action_loop.get("actions_recorded", len(self.actions)) or 0))
        except (TypeError, ValueError):
            actions_recorded = len(self.actions)
        return {
            "supported_actions": list(self._supported_action_names(skill_memories)),
            "snapshot_action_count": int(len(self.actions)),
            "runtime_episode_count": int(len(self.runtime_episodes)),
            "action_history_count": int(actions_recorded),
            "action_history_truncated": bool(actions_recorded > len(self.actions)),
            "memory_capacity": memory_capacity if memory_capacity is not None else None,
            "memory_fill_ratio": float(memory_fill_ratio) if isinstance(memory_fill_ratio, (int, float)) else None,
            "state_revision": int(self.state_revision),
            "configured": bool(self.configured),
            "running": bool(self.running),
        }

    def _surface_budgets(self, world_model_lite: WorldModelLiteSummary) -> dict[str, Any]:
        try:
            actions_recorded = max(0, int(self.action_loop.get("actions_recorded", len(self.actions)) or 0))
        except (TypeError, ValueError):
            actions_recorded = len(self.actions)
        memory_size = self.memory.get("size", self.memory.get("memory_count"))
        memory_capacity = self.memory.get("capacity")
        return {
            "action_history_used": int(actions_recorded),
            "action_snapshot_used": int(len(self.actions)),
            "runtime_episode_snapshot_used": int(len(self.runtime_episodes)),
            "policy_budget_use": float(world_model_lite.budget_use),
            "policy_cost": float(world_model_lite.cost),
            "policy_risk": float(world_model_lite.risk),
            "policy_uncertainty": float(world_model_lite.uncertainty),
            "memory_size": memory_size if isinstance(memory_size, (int, float)) else None,
            "memory_capacity": memory_capacity if isinstance(memory_capacity, (int, float)) else None,
            "memory_fill_ratio": (
                float(self.memory.get("fill_ratio"))
                if isinstance(self.memory.get("fill_ratio"), (int, float))
                else None
            ),
        }

    def _surface_recent_failures(self) -> list[dict[str, Any]]:
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
            if len(failures) >= 6:
                break
        if len(failures) < 6:
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
                        "topics": [] if episode_prediction is None else list(episode_prediction.topics),
                    }
                )
                if len(failures) >= 6:
                    break
        return failures

    def _surface_uncertain_domains(self) -> list[dict[str, Any]]:
        domains: dict[str, dict[str, int]] = {}

        def _domain_bucket(name: str) -> dict[str, int]:
            key = _clean_text(name).lower() or "unknown"
            return domains.setdefault(
                key,
                {"pending_predictions": 0, "unknown_predictions": 0, "unverified_actions": 0, "contradictions": 0},
            )

        for prediction in self.predictions:
            prediction_topics = prediction.topics or ((_clean_text(prediction.source_id) or "prediction"),)
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
            if action.verification.status in {VerificationStatus.UNKNOWN, VerificationStatus.UNVERIFIED}:
                for topic in action_topics:
                    _domain_bucket(topic)["unverified_actions"] += 1
            if action.verification.contradiction or action.verification.status == VerificationStatus.CONTRADICTED:
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
            for domain, counts in ranked[:8]
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

    def _surface_grounding_health(self, world_model_lite: WorldModelLiteSummary) -> dict[str, Any]:
        evidence_count = sum(len(action.verification.evidence) for action in self.actions)
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
                cortex=self.cortex,
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
            "cortex": dict(self.cortex),
        }
