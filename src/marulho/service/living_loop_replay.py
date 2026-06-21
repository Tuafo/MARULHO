"""Replay Planning module for the Living Loop (Layer C).

This module contains ReplayCandidate, ReplayPlan, build_replay_plan,
replay_candidate_safety_flags, and all replay-specific constants and private
helpers used by the replay planning logic.

Dependency direction: Helpers → Records → Policy → Replay → Self-Model

This module imports from Policy, Records, and Helpers only; it never imports from
the Self-Model module.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import islice
from typing import Any, Mapping, Sequence

from marulho.service.living_loop_helpers import (
    _clean_text,
    _clamp01,
    _safe_float,
    _safe_ratio,
    _stable_id,
    _limited_unique_clean_text,
)
from marulho.service.living_loop_policy import (
    POLICY_ACTUATOR_HIGH_LATENCY_AVG_MS,
    POLICY_ACTUATOR_HIGH_LATENCY_MAX_MS,
    _coerce_feedback_telemetry,
    _policy_count,
    _policy_float,
    _policy_mapping,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPLAY_PLAN_SCHEMA_VERSION = 1
REPLAY_PLAN_PRIORITY_RULES_VERSION = "deterministic-v1"
REPLAY_PLAN_DEFAULT_LIMIT = 20
REPLAY_PLAN_MAX_LIMIT = 50
REPLAY_PLAN_SOURCE_WINDOW_LIMIT = 64
REPLAY_PLAN_FEEDBACK_WINDOW_LIMIT = 128
REPLAY_PLAN_FEEDBACK_TARGET_STUB_LIMIT = 32
REPLAY_PLAN_SOURCE_WINDOW_POLICY = "bounded_recent_feedback_target_window_v1"

REPLAY_SAMPLE_SAFETY_BOUNDARIES: tuple[str, ...] = (
    "no_training",
    "no_sleep",
    "no_memory_verification_promotion",
    "no_feedback_posting",
    "no_digital_action_execution",
    "no_external_calls",
    "audit_only",
)

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

# ---------------------------------------------------------------------------
# ReplayCandidate
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# replay_candidate_safety_flags
# ---------------------------------------------------------------------------

def replay_candidate_safety_flags(candidate: Mapping[str, Any]) -> dict[str, Any]:
    reason_codes = {
        _clean_text(item).lower()
        for item in candidate.get("reason_codes", [])
        if _clean_text(item)
    }
    feedback = _policy_mapping(candidate.get("feedback"))
    provenance = _policy_mapping(candidate.get("provenance"))
    provenance_text = " ".join(
        _clean_text(value).lower()
        for value in (
            provenance.get("provenance"),
            provenance.get("source"),
            provenance.get("kind"),
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


# ---------------------------------------------------------------------------
# _default_replay_sample_safety_flags
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# _coerce_replay_sample_summary (also used by Self-Model; re-exported)
# ---------------------------------------------------------------------------

def _coerce_replay_sample_summary(summary: Mapping[str, Any] | None) -> dict[str, Any]:
    data = dict(summary or {}) if isinstance(summary, Mapping) else {}
    mode_counts = _policy_mapping(data.get("mode_counts"))
    status_counts = _policy_mapping(data.get("status_counts"))
    latest = data.get("latest_history_item") if isinstance(data.get("latest_history_item"), Mapping) else None
    latest_keys = (
        "schema_version",
        "replay_sample_id",
        "created_at",
        "mode",
        "status",
        "reason",
        "endpoint",
        "operator_id",
        "requested_candidate_id",
        "target_type",
        "target_id",
        "requested_count",
        "selected_count",
        "selected_candidate_ids",
        "safety_checks",
        "safety_flags",
        "plan_summary",
    )
    latest_item = {
        key: latest[key]
        for key in latest_keys
        if isinstance(latest, Mapping) and key in latest
    } if isinstance(latest, Mapping) else None
    if latest_item is not None:
        if str(latest_item.get("mode") or "sample") not in {"dry_run", "sample"}:
            latest_item["mode"] = "sample"
    safety_flags = _policy_mapping(data.get("safety_flags"))
    if not safety_flags and isinstance(latest, Mapping) and isinstance(latest.get("safety_flags"), Mapping):
        safety_flags = latest["safety_flags"]
    normalized_safety_flags = {**_default_replay_sample_safety_flags(), **dict(safety_flags)}
    sample_count = 0
    dry_run_count = 0
    for key, value in dict(mode_counts).items():
        mode = str(key)
        try:
            count = int(value or 0)
        except (TypeError, ValueError):
            count = 0
        if mode == "dry_run":
            dry_run_count += count
        else:
            sample_count += count
    return {
        "schema_version": int(data.get("schema_version", 1) or 1),
        "endpoint": str(data.get("endpoint") or "/terminus/replay-sample"),
        "history_endpoint": str(data.get("history_endpoint") or "/terminus/replay-sample/history"),
        "count": int(data.get("count", data.get("history_count", 0)) or 0),
        "history_count": int(data.get("history_count", data.get("count", 0)) or 0),
        "selected_count": int(data.get("selected_count", 0) or 0),
        "latest_selected_count": int(data.get("latest_selected_count", 0) or 0),
        "mode_counts": {"dry_run": dry_run_count, "sample": sample_count},
        "status_counts": {str(key): int(value or 0) for key, value in dict(status_counts).items()},
        "latest_history_item": latest_item,
        "safety_flags": normalized_safety_flags,
        "safety_boundaries": list(data.get("safety_boundaries") or REPLAY_SAMPLE_SAFETY_BOUNDARIES),
        "audit_only": True,
        "advisory": True,
        "executable": False,
    }


# ---------------------------------------------------------------------------
# ReplayPlan
# ---------------------------------------------------------------------------

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
    source_window: dict[str, Any] | None = None
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
            "source_window": dict(self.source_window or {}),
            "candidates": [candidate.to_payload() for candidate in self.candidates],
        }


# ---------------------------------------------------------------------------
# Replay-specific private helpers
# ---------------------------------------------------------------------------

def _replay_sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _replay_item_timestamp(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    for key in (
        "created_at",
        "recorded_at",
        "completed_at",
        "updated_at",
        "timestamp",
    ):
        text = _clean_text(value.get(key))
        if text:
            return text
    return ""


def _replay_sequence_is_newest_first(value: Sequence[Any]) -> bool:
    if len(value) < 2:
        return False
    try:
        first = value[0]
        last = value[len(value) - 1]
    except (IndexError, TypeError):
        return False
    first_at = _replay_item_timestamp(first)
    last_at = _replay_item_timestamp(last)
    return bool(first_at and last_at and first_at >= last_at)


def _replay_bounded_mappings(value: Any, *, max_items: int) -> tuple[list[dict[str, Any]], int, bool]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return [], 0, False
    total = len(value)
    requested = max(0, int(max_items))
    if requested <= 0:
        return [], total, total > 0
    if total <= requested:
        raw_items: Sequence[Any] | list[Any] = value
    elif _replay_sequence_is_newest_first(value):
        try:
            raw_items = value[0:requested]
        except TypeError:
            raw_items = [value[index] for index in range(0, requested)]
    elif isinstance(value, deque):
        raw_items = list(reversed(list(islice(reversed(value), requested))))
    else:
        start = max(0, total - requested)
        try:
            raw_items = value[start:total]
        except TypeError:
            raw_items = [value[index] for index in range(start, total)]
    items = [dict(item) for item in raw_items if isinstance(item, Mapping)]
    return items, total, total > requested


def _bounded_feedback_summary_for_replay(
    feedback_summary: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = dict(feedback_summary or {})
    recent, total_recent, truncated = _replay_bounded_mappings(
        source.get("recent_feedback"),
        max_items=REPLAY_PLAN_FEEDBACK_WINDOW_LIMIT,
    )
    source["recent_feedback"] = recent
    summary = _coerce_feedback_telemetry(source)
    return summary, {
        "recent_feedback_total": int(total_recent),
        "recent_feedback_window_count": int(len(recent)),
        "recent_feedback_truncated": bool(truncated),
    }


def _replay_feedback_index(feedback: Mapping[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in _replay_sequence(feedback.get("recent_feedback")):
        if not isinstance(item, Mapping):
            continue
        target_type = _clean_text(item.get("target_type")).lower()
        target_id = _clean_text(item.get("target_id"))
        if not target_type or not target_id:
            continue
        index.setdefault((target_type, target_id), []).append(dict(item))
    return index


def _replay_recent_feedback_for_target(
    feedback: Mapping[str, Any],
    target_type: str,
    target_id: str,
    feedback_index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    if feedback_index is not None:
        return [
            dict(item)
            for item in feedback_index.get((target_type, target_id), ())
            if isinstance(item, Mapping)
        ]
    recent = _replay_sequence(feedback.get("recent_feedback"))
    return [
        dict(item)
        for item in recent
        if isinstance(item, Mapping)
        and _clean_text(item.get("target_type")).lower() == target_type
        and _clean_text(item.get("target_id")) == target_id
    ]


def _replay_feedback_entries(
    target: Mapping[str, Any],
    feedback: Mapping[str, Any],
    target_type: str,
    target_id: str,
    feedback_index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    entries = [dict(item) for item in _replay_sequence(target.get("feedback")) if isinstance(item, Mapping)]
    entries.extend(_replay_recent_feedback_for_target(feedback, target_type, target_id, feedback_index))
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


def _feedback_verification_status(summary: Mapping[str, Any]) -> str:
    if int(summary.get("contradicted_count", 0) or 0) > 0:
        return "contradicted"
    if int(summary.get("unverified_count", 0) or 0) > 0:
        return "unverified"
    if int(summary.get("verified_count", 0) or 0) > 0:
        return "verified"
    return "unknown"


def _replay_feedback_target_stubs(
    feedback_index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    *,
    target_type: str,
    existing_ids: set[str],
    max_items: int,
) -> list[dict[str, Any]]:
    stubs: list[tuple[float, str, dict[str, Any]]] = []
    for (kind, target_id), entries in feedback_index.items():
        if kind != target_type or not target_id or target_id in existing_ids:
            continue
        materialized = [dict(entry) for entry in entries if isinstance(entry, Mapping)]
        if not materialized:
            continue
        summary = _replay_feedback_summary(materialized)
        status = _feedback_verification_status(summary)
        signal_score = (
            3.0 * float(int(summary.get("contradicted_count", 0) or 0) > 0)
            + 2.0 * float(bool(summary.get("has_corrected_output")))
            + 1.0 * float(int(summary.get("unverified_count", 0) or 0) > 0)
        )
        latest_at = _clean_text(summary.get("latest_feedback_at")) or _clean_text(
            materialized[-1].get("created_at")
        )
        confidence = _clamp01(summary.get("mean_confidence", 0.0))
        if target_type == "runtime_episode":
            stubs.append(
                (
                    signal_score,
                    latest_at,
                    {
                        "episode_id": target_id,
                        "operation": "feedback_target_replay",
                        "status": "unknown",
                        "created_at": latest_at,
                        "verification": {
                            "status": status,
                            "confidence": confidence,
                            "contradiction": status == "contradicted",
                        },
                        "prediction": {
                            "status": "contradicted" if status == "contradicted" else "pending",
                            "confidence": confidence,
                        },
                        "provenance": "bounded_recent_feedback_target_stub",
                    },
                )
            )
        elif target_type == "action":
            stubs.append(
                (
                    signal_score,
                    latest_at,
                    {
                        "action_id": target_id,
                        "action_type": "feedback_target_replay",
                        "recorded_at": latest_at,
                        "created_at": latest_at,
                        "verification": {
                            "status": status,
                            "confidence": confidence,
                            "contradiction": status == "contradicted",
                        },
                        "provenance": "bounded_recent_feedback_target_stub",
                    },
                )
            )
    stubs.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [payload for _, _, payload in stubs[: max(0, int(max_items))]]
    selected.reverse()
    return selected


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


def _replay_memory_pressure(
    memory_health: Mapping[str, Any],
    subcortex_sleep_pressure: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    fill = _policy_float(memory_health.get("fill_ratio"), memory_health.get("fill_fraction"))
    fatigue = max(
        _policy_float(subcortex_sleep_pressure.get("fatigue")),
        _policy_float(subcortex_sleep_pressure.get("pressure")),
    )
    sleeping = bool(subcortex_sleep_pressure.get("is_sleeping", False)) or _clean_text(
        subcortex_sleep_pressure.get("current_mode")
    ).lower() == "sleeping"
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
    elif reason_set & {"unverified_feedback", "pending_prediction", "unverified_action"}:
        return "verify_pending_evidence"
    elif reason_set & {"memory_capacity_pressure", "fatigue_sleep_pressure"}:
        return "sleep_consolidation_advisory"
    elif reason_set & {"high_latency", "high_cost", "high_budget_use"}:
        return "reduce_scope_or_wait"
    elif reason_set & {"high_uncertainty", "uncertain_domain"}:
        return "collect_more_evidence"
    elif "healthy_grounded_state" in reason_set:
        return "continue_observing"
    else:
        return "replay_episode_for_grounding"


def _replay_endpoint_for_action(action: str) -> str:
    if action in {"review_contradiction", "verify_pending_evidence"}:
        return "/terminus/runtime-feedback"
    elif action == "sleep_consolidation_advisory":
        return "/terminus/living-loop"
    elif action == "replay_episode_for_grounding":
        return "/terminus/runtime-traces/export"
    elif action == "reduce_scope_or_wait":
        return "/terminus"
    else:
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
        replace(candidate, rank=index + 1)
        for index, candidate in enumerate(ranked[:limit])
    )


def _replay_prediction_status(data: Mapping[str, Any]) -> str:
    return _clean_text(data.get("status") or data.get("prediction_status")).lower()


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


def _feedback_signal(feedback: Mapping[str, Any]) -> float:
    """Compute a 0-1 signal from feedback counts (contradicted + unverified + corrected) / 3."""
    corrected_flag = 1 if feedback["has_corrected_output"] else 0
    return _clamp01((feedback["contradicted_count"] + feedback["unverified_count"] + corrected_flag) / 3.0)


def _effective_uncertainty(world_uncertainty: float, confidence: float, default_uncertainty: float = 0.35) -> float:
    """Compute clamped uncertainty from world uncertainty and confidence."""
    base = max(world_uncertainty, 1.0 - confidence if confidence else default_uncertainty)
    return _clamp01(base)


def _memory_pressure_component(memory_pressure: float, memory_reason_codes: Sequence[str]) -> float:
    """Return memory_pressure when there are memory-related reason codes, else 0.0."""
    return memory_pressure if memory_reason_codes else 0.0


def _is_unverified_status(status: str) -> bool:
    """Check whether a verification/prediction status indicates unverified state."""
    return status in {"unverified", "unknown", ""}


def _is_pending_prediction(status: str) -> bool:
    """Check whether a prediction status is pending or unverified."""
    return status in {"pending", "unverified", "unknown"}


# ---------------------------------------------------------------------------
# build_replay_plan
# ---------------------------------------------------------------------------

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
    feedback_summary, feedback_window = _bounded_feedback_summary_for_replay(
        loop.get("feedback_summary") if isinstance(loop.get("feedback_summary"), Mapping) else {}
    )
    feedback_index = _replay_feedback_index(feedback_summary)
    benchmark = _policy_mapping(loop.get("benchmark_telemetry"))
    memory_health = _policy_mapping(loop.get("memory_health"))
    subcortex_sleep_pressure = _policy_mapping(loop.get("subcortex_sleep_pressure"))
    memory_pressure, memory_context = _replay_memory_pressure(memory_health, subcortex_sleep_pressure)
    policy_decision = _policy_mapping(loop.get("policy_decision") or loop.get("policy_actuator"))
    policy_pressure, policy_context = _replay_policy_pressure(policy_decision)
    world = _policy_mapping(loop.get("world_model_lite"))
    world_uncertainty = _clamp01(_policy_float(world.get("uncertainty"), _policy_mapping(world.get("policy_score")).get("uncertainty")))
    memory_reason_codes: list[str] = []
    if memory_context["fill_ratio"] >= 0.90 or memory_context["status"] == "capacity_pressure":
        memory_reason_codes.append("memory_capacity_pressure")
    if memory_context["fatigue"] >= 0.70 or memory_context["is_sleeping"]:
        memory_reason_codes.append("fatigue_sleep_pressure")

    candidates: list[ReplayCandidate] = []
    episodes, episode_total, episodes_truncated = _replay_bounded_mappings(
        loop.get("runtime_episodes"),
        max_items=REPLAY_PLAN_SOURCE_WINDOW_LIMIT,
    )
    actions, action_total, actions_truncated = _replay_bounded_mappings(
        loop.get("actions"),
        max_items=REPLAY_PLAN_SOURCE_WINDOW_LIMIT,
    )
    predictions, prediction_total, predictions_truncated = _replay_bounded_mappings(
        loop.get("predictions"),
        max_items=REPLAY_PLAN_SOURCE_WINDOW_LIMIT,
    )
    uncertain_domains, domain_total, domains_truncated = _replay_bounded_mappings(
        loop.get("uncertain_domains"),
        max_items=REPLAY_PLAN_SOURCE_WINDOW_LIMIT,
    )
    existing_episode_ids = {
        _clean_text(item.get("episode_id"))
        for item in episodes
        if _clean_text(item.get("episode_id"))
    }
    existing_action_ids = {
        _clean_text(item.get("action_id"))
        for item in actions
        if _clean_text(item.get("action_id"))
    }
    episode_feedback_stubs = _replay_feedback_target_stubs(
        feedback_index,
        target_type="runtime_episode",
        existing_ids=existing_episode_ids,
        max_items=REPLAY_PLAN_FEEDBACK_TARGET_STUB_LIMIT,
    )
    action_feedback_stubs = _replay_feedback_target_stubs(
        feedback_index,
        target_type="action",
        existing_ids=existing_action_ids,
        max_items=REPLAY_PLAN_FEEDBACK_TARGET_STUB_LIMIT,
    )
    episodes = [*episode_feedback_stubs, *episodes]
    actions = [*action_feedback_stubs, *actions]

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
        feedback_entries = _replay_feedback_entries(
            episode,
            feedback_summary,
            "runtime_episode",
            target_id,
            feedback_index,
        )
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
        if _is_unverified_status(verification_status) or _is_pending_prediction(prediction_status):
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
        uncertainty = _effective_uncertainty(world_uncertainty, confidence)
        feedback_signal = _feedback_signal(target_feedback)
        components = {
            "safety": safety,
            "feedback": feedback_signal,
            "uncertainty": uncertainty if set(reasons) & {"pending_prediction", "unverified_feedback", "high_uncertainty"} else uncertainty * 0.4,
            "memory_pressure": _memory_pressure_component(memory_pressure, memory_reason_codes),
            "latency_pressure": latency_pressure,
            "policy_pressure": policy_pressure,
            "provenance_gap": 1.0 if _is_unverified_status(verification_status) else 0.0,
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
        feedback_entries = _replay_feedback_entries(
            action,
            feedback_summary,
            "action",
            target_id,
            feedback_index,
        )
        target_feedback = _replay_feedback_summary(feedback_entries)
        reasons: list[str] = []
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
        if _is_unverified_status(verification_status):
            reasons.append("unverified_action")
        if confidence and confidence < 0.40:
            reasons.append("high_uncertainty")
        if not reasons and target_feedback["verified_count"] > 0:
            reasons.append("healthy_grounded_state")
        if not reasons:
            continue
        safety = 1.0 if set(reasons) & {"contradicted_feedback", "contradicted_action"} else 0.0
        uncertainty = _effective_uncertainty(world_uncertainty, confidence)
        components = {
            "safety": safety,
            "feedback": _feedback_signal(target_feedback),
            "uncertainty": uncertainty if set(reasons) & {"unverified_action", "unverified_feedback", "high_uncertainty"} else uncertainty * 0.4,
            "memory_pressure": _memory_pressure_component(memory_pressure, memory_reason_codes),
            "latency_pressure": 0.0,
            "policy_pressure": policy_pressure,
            "provenance_gap": 1.0 if _is_unverified_status(verification_status) else 0.0,
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
        reasons: list[str] = []
        if status == "contradicted":
            reasons.append("contradicted_runtime_episode" if _clean_text(prediction.get("source_kind")).startswith("runtime_") else "contradicted_action")
        if _is_pending_prediction(status):
            reasons.append("pending_prediction")
        if confidence and confidence < 0.40:
            reasons.append("high_uncertainty")
        if not reasons:
            continue
        safety = 1.0 if any(code.startswith("contradicted") for code in reasons) else 0.0
        uncertainty = _effective_uncertainty(world_uncertainty, confidence, default_uncertainty=0.50)
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
                    "memory_pressure": _memory_pressure_component(memory_pressure, memory_reason_codes),
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
                    "memory_pressure": _memory_pressure_component(memory_pressure, memory_reason_codes),
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
            code for code in policy_context["reason_codes"]
            if code in REPLAY_REASON_PRECEDENCE
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
                    "memory_pressure": _memory_pressure_component(memory_pressure, memory_reason_codes),
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

    candidate_count_before_rank = len(candidates)
    ranked = _replay_rank_candidates(candidates, limit=count)
    plan_reasons = _replay_unique_reasons([code for candidate in ranked for code in candidate.reason_codes])
    source_window = {
        "surface": "bounded_replay_plan_source_window.v1",
        "window_policy": REPLAY_PLAN_SOURCE_WINDOW_POLICY,
        "runs_live_tick": False,
        "device_placement": {
            "archival_metadata": "cpu_python_objects",
            "active_replay_computation": "cpu_service_ranking_only",
            "gpu_used": False,
        },
        "selection_criteria": [
            "timestamp_oriented_runtime_episode_window",
            "timestamp_oriented_action_window",
            "timestamp_oriented_prediction_window",
            "timestamp_oriented_uncertain_domain_window",
            "recent_feedback_target_stubs",
            "priority_ranked_with_safety_feedback_uncertainty_memory_latency_policy",
        ],
        "source_limits": {
            "runtime_episodes": REPLAY_PLAN_SOURCE_WINDOW_LIMIT,
            "actions": REPLAY_PLAN_SOURCE_WINDOW_LIMIT,
            "predictions": REPLAY_PLAN_SOURCE_WINDOW_LIMIT,
            "uncertain_domains": REPLAY_PLAN_SOURCE_WINDOW_LIMIT,
            "recent_feedback": REPLAY_PLAN_FEEDBACK_WINDOW_LIMIT,
            "feedback_target_stubs": REPLAY_PLAN_FEEDBACK_TARGET_STUB_LIMIT,
            "returned_candidates": count,
        },
        "source_counts": {
            "runtime_episodes": int(loop.get("runtime_episode_count", episode_total) or 0),
            "actions": int(loop.get("action_count", action_total) or 0),
            "predictions": int(loop.get("prediction_count", prediction_total) or 0),
            "uncertain_domains": int(domain_total),
            "recent_feedback": int(feedback_window["recent_feedback_total"]),
        },
        "window_counts": {
            "runtime_episodes": int(len(episodes)),
            "actions": int(len(actions)),
            "predictions": int(len(predictions)),
            "uncertain_domains": int(len(uncertain_domains)),
            "recent_feedback": int(feedback_window["recent_feedback_window_count"]),
            "feedback_runtime_episode_stubs": int(len(episode_feedback_stubs)),
            "feedback_action_stubs": int(len(action_feedback_stubs)),
        },
        "truncated_source_counts": {
            "runtime_episodes": bool(episodes_truncated),
            "actions": bool(actions_truncated),
            "predictions": bool(predictions_truncated),
            "uncertain_domains": bool(domains_truncated),
            "recent_feedback": bool(feedback_window["recent_feedback_truncated"]),
        },
        "feedback_index_entry_count": int(sum(len(entries) for entries in feedback_index.values())),
        "feedback_index_target_count": int(len(feedback_index)),
        "candidate_count_before_rank": int(candidate_count_before_rank),
        "candidate_count_returned": int(len(ranked)),
    }
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
        source_window=source_window,
    )
