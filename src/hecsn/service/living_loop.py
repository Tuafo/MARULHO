"""Backward-compatible re-export shim for the Living Loop modules.

This module contains only import statements and re-exports — no
implementation code.  All public symbols previously available from
``hecsn.service.living_loop`` are re-exported from their canonical
sub-modules (helpers, records, policy, replay, self-model) so that
no consumer import breaks.
"""
from __future__ import annotations

__all__ = [
    # ── Records ─────────────────────────────────────────────────────────
    "ActionExecutionRecord",
    "ActionExecutionStatus",
    "ActionVerificationRecord",
    "ConsolidationRecord",
    "ConsolidationStatus",
    "PredictionRecord",
    "PredictionStatus",
    "ProvenanceState",
    "RuntimeEpisodeTrace",
    "SkillMemoryRecord",
    "VerificationStatus",
    # ── Policy ───────────────────────────────────────────────────────────
    "POLICY_ACTUATOR_HIGH_LATENCY_AVG_MS",
    "POLICY_ACTUATOR_HIGH_LATENCY_MAX_MS",
    "POLICY_ACTUATOR_SCHEMA_VERSION",
    "PolicyActuatorRecommendation",
    "PolicyScore",
    "WorldModelLiteSummary",
    "build_policy_actuator_status",
    # ── Replay ───────────────────────────────────────────────────────────
    "REPLAY_PLAN_DEFAULT_LIMIT",
    "REPLAY_PLAN_MAX_LIMIT",
    "REPLAY_PLAN_PRIORITY_RULES_VERSION",
    "REPLAY_PLAN_PRIORITY_WEIGHTS",
    "REPLAY_PLAN_SCHEMA_VERSION",
    "REPLAY_REASON_PRECEDENCE",
    "REPLAY_SAMPLE_SAFETY_BOUNDARIES",
    "ReplayCandidate",
    "ReplayPlan",
    "build_replay_plan",
    "replay_candidate_safety_flags",
    # ── Self-Model ───────────────────────────────────────────────────────
    "OperationalSelfModel",
    "build_runtime_benchmark_telemetry",
    # ── Private helpers re-exported for cross-module use ──────────────────
    "_as_mapping",
    "_clean_text",
    "_clamp01",
    "_coerce_feedback_telemetry",
    "_coerce_replay_sample_summary",
    "_coerce_world_model_lite",
    "_default_replay_sample_safety_flags",
    "_endpoint_bucket_name",
    "_endpoint_latency_empty",
    "_enum_value",
    "_extract_cache_summary",
    "_extract_retired_external_adapter_summary",
    "_latency_summary",
    "_latest_text",
    "_limited_unique_clean_text",
    "_memory_counter_summary",
    "_policy_count",
    "_policy_float",
    "_policy_mapping",
    "_provenance_value",
    "_safe_float",
    "_safe_ratio",
    "_stable_id",
    "_verification_status_from_payload",
]

from hecsn.service.living_loop_helpers import (  # re-exported for backward compatibility
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

from hecsn.service.living_loop_policy import (  # re-exported for backward compatibility
    POLICY_ACTUATOR_HIGH_LATENCY_AVG_MS,
    POLICY_ACTUATOR_HIGH_LATENCY_MAX_MS,
    POLICY_ACTUATOR_SCHEMA_VERSION,
    PolicyActuatorRecommendation,
    PolicyScore,
    WorldModelLiteSummary,
    build_policy_actuator_status,
    _coerce_feedback_telemetry,
    _policy_count,
    _policy_float,
    _policy_mapping,
)

from hecsn.service.living_loop_replay import (  # re-exported for backward compatibility
    REPLAY_PLAN_DEFAULT_LIMIT,
    REPLAY_PLAN_MAX_LIMIT,
    REPLAY_PLAN_PRIORITY_RULES_VERSION,
    REPLAY_PLAN_PRIORITY_WEIGHTS,
    REPLAY_PLAN_SCHEMA_VERSION,
    REPLAY_REASON_PRECEDENCE,
    REPLAY_SAMPLE_SAFETY_BOUNDARIES,
    ReplayCandidate,
    ReplayPlan,
    build_replay_plan,
    replay_candidate_safety_flags,
    _coerce_replay_sample_summary,
    _default_replay_sample_safety_flags,
)

from hecsn.service.living_loop_self_model import (  # re-exported for backward compatibility
    OperationalSelfModel,
    build_runtime_benchmark_telemetry,
    _endpoint_bucket_name,
    _endpoint_latency_empty,
    _extract_cache_summary,
    _extract_retired_external_adapter_summary,
    _latency_summary,
    _memory_counter_summary,
)
