"""Policy Scoring module for the Living Loop (Layer B).

This module contains PolicyScore, WorldModelLiteSummary,
PolicyActuatorRecommendation, build_policy_actuator_status,
and the policy-specific private helpers used by the policy actuator
decision logic.

Dependency direction: Helpers → Records → Policy → Replay → Self-Model

This module imports from Records and Helpers only; it never imports from
Replay or Self-Model modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from hecsn.service.living_loop_helpers import (
    _clean_text,
    _clamp01,
    _safe_float,
    _safe_ratio,
)
from hecsn.service.living_loop_records import (
    ActionExecutionRecord,
    ConsolidationRecord,
    PredictionRecord,
    PredictionStatus,
    VerificationStatus,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLICY_ACTUATOR_SCHEMA_VERSION = 1
POLICY_ACTUATOR_HIGH_LATENCY_AVG_MS = 1000.0
POLICY_ACTUATOR_HIGH_LATENCY_MAX_MS = 1500.0

# ---------------------------------------------------------------------------
# PolicyScore
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# WorldModelLiteSummary
# ---------------------------------------------------------------------------


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
        raw_policy_score = data.get("policy_score")
        policy_payload = raw_policy_score if isinstance(raw_policy_score, Mapping) else data
        policy_score = PolicyScore.from_payload(policy_payload)
        return cls(
            prediction_count=_safe_int(data.get("prediction_count", 0)),
            fulfilled_count=_safe_int(data.get("fulfilled_count", 0)),
            contradicted_count=_safe_int(data.get("contradicted_count", 0)),
            pending_count=_safe_int(data.get("pending_count", 0)),
            unknown_count=_safe_int(data.get("unknown_count", 0)),
            evaluated_prediction_count=_safe_int(data.get("evaluated_prediction_count", 0)),
            prediction_accuracy=_clamp01(data.get("prediction_accuracy", 0.0)),
            verification_count=_safe_int(data.get("verification_count", 0)),
            verified_action_count=_safe_int(data.get("verified_action_count", 0)),
            contradicted_action_count=_safe_int(data.get("contradicted_action_count", 0)),
            unverified_action_count=_safe_int(data.get("unverified_action_count", 0)),
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
            action.verification
            for action in action_records
            if action.verification.status != VerificationStatus.UNKNOWN
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


# ---------------------------------------------------------------------------
# Shared policy helpers (also used by Replay/Self-Model; re-exported)
# ---------------------------------------------------------------------------


def _safe_int(value: Any) -> int:
    """Safely convert a value to a non-negative int, returning 0 on failure."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _policy_count(data: Mapping[str, Any], key: str) -> int:
    return _safe_int(data.get(key, 0))


def _policy_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _policy_float(*values: Any) -> float:
    for value in values:
        result = _safe_float(value)
        if result is not None:
            return result
    return 0.0


# ---------------------------------------------------------------------------
# Policy-specific private helpers (only used within this module)
# ---------------------------------------------------------------------------


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
    if target_action_id:
        target_type = "action"
    elif target_episode_id:
        target_type = "runtime_episode"
    else:
        target_type = ""
    target_id = target_action_id or target_episode_id or ""

    if action == "investigate_contradictions":
        return (
            "/terminus/living-loop",
            {"focus": "contradictions", "target_type": target_type, "target_id": target_id},
        )
    elif action == "verify_pending_evidence":
        return (
            "/terminus/runtime-feedback",
            {
                "target_type": target_type,
                "target_id": target_id,
                "verdict": "verified_or_contradicted_after_review",
                "note": "operator_review_required",
            },
        )
    elif action == "consolidate_or_sleep":
        return (
            "/terminus/living-loop",
            {"reason": "policy_actuator_advisory_memory_or_fatigue_pressure"},
        )
    elif action == "reduce_scope_or_wait":
        return (
            "/terminus",
            {"operator_note": "reduce source scope, defer new requests, or wait for pressure to clear"},
        )
    elif action == "collect_more_evidence":
        return (
            "/terminus/living-loop",
            {"focus": "uncertain_domains", "operator_note": "collect evidence before changing policy"},
        )
    else:
        return "/terminus", {"operator_note": "healthy grounded state; continue observing current policy"}


# ---------------------------------------------------------------------------
# _coerce_feedback_telemetry (also used by Self-Model; re-exported)
# ---------------------------------------------------------------------------


def _coerce_feedback_telemetry(feedback_summary: Mapping[str, Any] | None) -> dict[str, Any]:
    data = dict(feedback_summary or {})

    raw_status_counts = data.get("status_counts")
    status_counts: Mapping[str, Any] = raw_status_counts if isinstance(raw_status_counts, Mapping) else {}

    raw_verdict_counts = data.get("verdict_counts")
    verdict_counts: Mapping[str, Any] = raw_verdict_counts if isinstance(raw_verdict_counts, Mapping) else {}

    raw_target_counts = data.get("target_counts")
    target_counts: Mapping[str, Any] = raw_target_counts if isinstance(raw_target_counts, Mapping) else {}

    raw_recent_feedback = data.get("recent_feedback")
    recent_feedback = (
        [dict(item) for item in list(raw_recent_feedback or []) if isinstance(item, Mapping)]
        if isinstance(raw_recent_feedback, Sequence)
        and not isinstance(raw_recent_feedback, (str, bytes))
        else []
    )

    def _count(key: str) -> int:
        return _safe_int(data.get(f"{key}_count", status_counts.get(key, 0)))

    def _verdict_count(key: str) -> int:
        return _safe_int(verdict_counts.get(key, 0))

    def _target_count(key: str) -> int:
        return _safe_int(target_counts.get(key, 0))

    feedback_count = data.get("feedback_count", data.get("count"))
    total = _safe_int(feedback_count)

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


# ---------------------------------------------------------------------------
# PolicyActuatorRecommendation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# build_policy_actuator_status
# ---------------------------------------------------------------------------


def build_policy_actuator_status(
    living_loop: Mapping[str, Any],
    *,
    retired_runtime_path_snapshot: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> PolicyActuatorRecommendation:
    """Build an advisory-only policy recommendation without executing any action."""
    loop = dict(living_loop or {})
    world = _policy_mapping(loop.get("world_model_lite"))
    policy_score = _policy_mapping(world.get("policy_score"))
    feedback = _coerce_feedback_telemetry(
        loop.get("feedback_summary") if isinstance(loop.get("feedback_summary"), Mapping) else {}
    )
    benchmark = _policy_mapping(loop.get("benchmark_telemetry"))
    memory_health = _policy_mapping(loop.get("memory_health"))
    benchmark_memory = _policy_mapping(benchmark.get("memory"))
    budgets = _policy_mapping(loop.get("budgets"))
    grounding_health = _policy_mapping(loop.get("grounding_health"))
    retired_runtime_path = dict(retired_runtime_path_snapshot or _policy_mapping(loop.get("retired_runtime_path")))
    retired_runtime_path_state = _policy_mapping(retired_runtime_path.get("legacy_snapshot")) or retired_runtime_path
    drives = _policy_mapping(retired_runtime_path_state.get("drives"))

    information_gain = _clamp01(_policy_max_float(world.get("information_gain"), policy_score.get("information_gain")))
    goal_progress = _clamp01(_policy_max_float(world.get("goal_progress"), policy_score.get("goal_progress")))
    policy_cost = _clamp01(_policy_max_float(world.get("cost"), policy_score.get("cost"), budgets.get("policy_cost")))
    budget_use = _clamp01(
        _policy_max_float(world.get("budget_use"), policy_score.get("budget_use"), budgets.get("policy_budget_use"))
    )
    risk = _clamp01(_policy_max_float(world.get("risk"), policy_score.get("risk"), budgets.get("policy_risk")))
    uncertainty = _clamp01(
        max(
            _policy_max_float(
                world.get("uncertainty"), policy_score.get("uncertainty"), budgets.get("policy_uncertainty")
            ),
            _policy_float(drives.get("uncertainty")),
        )
    )
    memory_fill = _clamp01(
        max(
            _policy_float(memory_health.get("fill_ratio"), memory_health.get("fill_fraction")),
            _policy_float(benchmark_memory.get("fill_ratio"), benchmark_memory.get("fill_fraction")),
            _policy_float(retired_runtime_path_state.get("memory_fill_ratio")),
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
    grounding_contradicted = (
        _clean_text(grounding_health.get("status")).lower() == "contradictions_present"
    )
    failed_episode_id = _policy_first_episode_target(loop, {"contradicted"})

    retired_runtime_path_sleeping = (
        fatigue >= 0.70
        or bool(retired_runtime_path_state.get("is_sleeping", False))
        or _clean_text(retired_runtime_path_state.get("current_mode")).lower() == "sleeping"
    )
    retired_runtime_sleeping = (
        not bool(retired_runtime_path.get("active_runtime_requirement", True))
        and retired_runtime_path_sleeping
    )

    if (
        contradicted_feedback > 0
        or contradicted_predictions > 0
        or contradicted_actions > 0
        or grounding_contradicted
        or failed_episode_id
    ):
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
        elif memory_fill >= 0.90 or retired_runtime_sleeping or retired_runtime_path_sleeping:
            if memory_fill >= 0.90:
                reasons.append(
                    {
                        "code": "memory_capacity_pressure",
                        "detail": f"Memory fill is {memory_fill:.2f}, at or above 0.90.",
                    }
                )
            if retired_runtime_sleeping or retired_runtime_path_sleeping:
                reasons.append(
                    {
                        "code": "fatigue_sleep_pressure",
                        "detail": f"Retired runtime sleep pressure is {fatigue:.2f}.",
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
    # Expected outcomes per action: (information_gain, goal_progress, cost, risk)
    expected_by_action: dict[str, tuple[float, float, float, float]] = {
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
