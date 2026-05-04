"""Runtime Records module for the Living Loop (Layer A).

This module contains all enums and frozen dataclasses that represent
the runtime record types used throughout the Living Loop. It is the
lowest depth layer (Layer A) and depends only on the shared helpers
module and external packages.

Dependency direction: Helpers → Records → Policy → Replay → Self-Model

This module never imports from Policy, Replay, or Self-Model modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

from hecsn.cortex.episodic_memory import Provenance
from hecsn.service.action_loop import ActionVerification, DigitalActionResult
from hecsn.service.living_loop_helpers import (
    _as_mapping,
    _clean_text,
    _clamp01,
    _enum_value,
    _latest_text,
    _limited_unique_clean_text,
    _provenance_value,
    _safe_ratio,
    _stable_id,
    _verification_status_from_payload,
)


def _safe_int(value: Any, minimum: int = 0) -> int:
    """Convert a value to a non-negative int, returning *minimum* on failure."""
    try:
        return max(minimum, int(value or 0))
    except (TypeError, ValueError):
        return minimum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PredictionStatus(str, Enum):
    PENDING = "pending"
    FULFILLED = "fulfilled"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class ActionExecutionStatus(str, Enum):
    REQUESTED = "requested"
    EXECUTED = "executed"
    FAILED = "failed"
    REUSED = "reused"


class VerificationStatus(str, Enum):
    UNKNOWN = "unknown"
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"


class ConsolidationStatus(str, Enum):
    RAW = "raw"
    COOLING = "cooling"
    CREDITED = "credited"
    PENALIZED = "penalized"
    FORGIVEN = "forgiven"
    CONSOLIDATED = "consolidated"
    RETIRED = "retired"


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    predicted_outcome: str
    status: PredictionStatus = PredictionStatus.PENDING
    provenance: Provenance = Provenance.INFERRED
    confidence: float = 0.0
    source_kind: str = "action"
    source_id: str = ""
    created_at: str = ""
    topics: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | DigitalActionResult) -> "PredictionRecord":
        data = _as_mapping(payload)
        action_id = _clean_text(data.get("action_id") or data.get("source_id"))
        verification = _as_mapping(data.get("verification"))
        verification_status = _verification_status_from_payload(verification.get("status"))
        success = bool(verification.get("success", False))
        contradiction = bool(verification.get("contradiction", False))

        if _clean_text(data.get("status")) and not verification:
            status = _enum_value(PredictionStatus, data.get("status"), PredictionStatus.UNKNOWN)
            provenance = _provenance_value(data.get("provenance"), Provenance.INFERRED)
        elif success or verification_status == VerificationStatus.VERIFIED:
            status = PredictionStatus.FULFILLED
            provenance = Provenance.VERIFIED
        elif contradiction or verification_status == VerificationStatus.CONTRADICTED:
            status = PredictionStatus.CONTRADICTED
            provenance = Provenance.CONTRADICTED
        elif verification_status == VerificationStatus.UNVERIFIED:
            status = PredictionStatus.PENDING
            provenance = Provenance.INFERRED
        else:
            status = PredictionStatus.UNKNOWN
            provenance = _provenance_value(data.get("provenance"), Provenance.INFERRED)

        predicted = _clean_text(data.get("predicted_outcome"))
        created_at = _clean_text(data.get("recorded_at") or data.get("created_at"))

        return cls(
            prediction_id=_clean_text(data.get("prediction_id")) or _stable_id("pred", action_id, predicted, created_at),
            predicted_outcome=predicted,
            status=status,  # type: ignore[arg-type]
            provenance=provenance,
            confidence=float(verification.get("confidence", data.get("confidence", 0.0)) or 0.0),
            source_kind=_clean_text(data.get("source_kind")) or "action",
            source_id=action_id,
            created_at=created_at,
            topics=tuple(_clean_text(item).lower() for item in list(data.get("topics") or []) if _clean_text(item)),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "predicted_outcome": self.predicted_outcome,
            "status": self.status.value,
            "provenance": self.provenance.value,
            "confidence": float(self.confidence),
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "created_at": self.created_at,
            "topics": list(self.topics),
        }


@dataclass(frozen=True)
class ActionVerificationRecord:
    verification_id: str
    status: VerificationStatus
    success: bool
    confidence: float
    contradiction: bool
    summary: str = ""
    evidence: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any] | ActionVerification,
        *,
        action_id: str = "",
    ) -> "ActionVerificationRecord":
        data = _as_mapping(payload)
        status = _verification_status_from_payload(data.get("status"))
        evidence = tuple(
            dict(item)
            for item in list(data.get("evidence") or [])
            if isinstance(item, Mapping)
        )
        summary = _clean_text(data.get("summary"))

        return cls(
            verification_id=_clean_text(data.get("verification_id")) or _stable_id("ver", action_id, status.value, summary, evidence),
            status=status,
            success=bool(data.get("success", status == VerificationStatus.VERIFIED)),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            contradiction=bool(data.get("contradiction", status == VerificationStatus.CONTRADICTED)),
            summary=summary,
            evidence=evidence,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "verification_id": self.verification_id,
            "status": self.status.value,
            "success": bool(self.success),
            "confidence": float(self.confidence),
            "contradiction": bool(self.contradiction),
            "summary": self.summary,
            "evidence_count": int(len(self.evidence)),
            "evidence": [dict(item) for item in self.evidence],
        }


@dataclass(frozen=True)
class ActionExecutionRecord:
    action_id: str
    action_type: str
    execution_status: ActionExecutionStatus
    prediction: PredictionRecord
    verification: ActionVerificationRecord
    inputs: dict[str, Any] = field(default_factory=dict)
    actual_outcome: str = ""
    recorded_at: str = ""
    topics: tuple[str, ...] = ()
    trigger_reason: str = "operator"
    trigger_query_text: str = ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | DigitalActionResult) -> "ActionExecutionRecord":
        data = _as_mapping(payload)
        action_id = _clean_text(data.get("action_id"))
        verification = ActionVerificationRecord.from_payload(data.get("verification") or {}, action_id=action_id)
        prediction_payload = data.get("prediction") if isinstance(data.get("prediction"), Mapping) else data
        prediction = PredictionRecord.from_payload(prediction_payload)
        action_type = _clean_text(data.get("action_type") or data.get("type")).lower()

        if _clean_text(data.get("execution_status")):
            execution_status = _enum_value(
                ActionExecutionStatus,
                data.get("execution_status"),
                ActionExecutionStatus.EXECUTED,
            )
        elif _clean_text(data.get("actual_outcome")) or verification.status != VerificationStatus.UNKNOWN:
            execution_status = ActionExecutionStatus.EXECUTED
        else:
            execution_status = ActionExecutionStatus.REQUESTED

        return cls(
            action_id=action_id,
            action_type=action_type,
            execution_status=execution_status,  # type: ignore[arg-type]
            prediction=prediction,
            verification=verification,
            inputs=dict(data.get("inputs") or {}),
            actual_outcome=_clean_text(data.get("actual_outcome")),
            recorded_at=_clean_text(data.get("recorded_at")),
            topics=tuple(_clean_text(item).lower() for item in list(data.get("topics") or []) if _clean_text(item)),
            trigger_reason=_clean_text(data.get("trigger_reason")) or "operator",
            trigger_query_text=_clean_text(data.get("trigger_query_text")),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "execution_status": self.execution_status.value,
            "prediction": self.prediction.to_payload(),
            "verification": self.verification.to_payload(),
            "inputs": dict(self.inputs),
            "actual_outcome": self.actual_outcome,
            "recorded_at": self.recorded_at,
            "topics": list(self.topics),
            "trigger_reason": self.trigger_reason,
            "trigger_query_text": self.trigger_query_text,
        }


@dataclass(frozen=True)
class RuntimeEpisodeTrace:
    episode_id: str
    operation: str
    status: str
    created_at: str
    completed_at: str = ""
    latency_ms: float | None = None
    request: dict[str, Any] = field(default_factory=dict)
    prediction: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)
    actual_output: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    feedback: list[dict[str, Any]] = field(default_factory=list)
    corrected_output: Any | None = None
    provenance: Provenance = Provenance.OBSERVED
    failure: dict[str, Any] | None = None
    trace_id: str = ""
    trace_path: str = ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RuntimeEpisodeTrace":
        data = dict(payload or {})
        verification = dict(data.get("verification") or {})
        failure = data.get("failure") if isinstance(data.get("failure"), Mapping) else None

        status = _clean_text(data.get("status")).lower()
        if not status:
            status = "failed" if failure else "succeeded"

        operation = _clean_text(data.get("operation")).lower() or "unknown"
        created_at = _clean_text(data.get("created_at")) or datetime.now(timezone.utc).isoformat()

        try:
            latency_ms = float(data.get("latency_ms")) if data.get("latency_ms") is not None else None
        except (TypeError, ValueError):
            latency_ms = None

        return cls(
            episode_id=_clean_text(data.get("episode_id")) or _stable_id("episode", operation, created_at),
            operation=operation,
            status=status,
            created_at=created_at,
            completed_at=_clean_text(data.get("completed_at")),
            latency_ms=latency_ms,
            request=dict(data.get("request") or {}),
            prediction=dict(data.get("prediction") or {}),
            action=dict(data.get("action") or {}),
            actual_output=dict(data.get("actual_output") or {}),
            verification=verification,
            feedback=[dict(item) for item in list(data.get("feedback") or []) if isinstance(item, Mapping)],
            corrected_output=data.get("corrected_output"),
            provenance=_provenance_value(data.get("provenance"), Provenance.OBSERVED),
            failure=dict(failure) if isinstance(failure, Mapping) else None,
            trace_id=_clean_text(data.get("trace_id")),
            trace_path=_clean_text(data.get("trace_path")),
        )

    def prediction_record(self) -> PredictionRecord | None:
        predicted = (
            _clean_text(self.prediction.get("predicted_output"))
            or _clean_text(self.prediction.get("proposed_answer"))
            or _clean_text(self.prediction.get("proposed_action"))
            or _clean_text(self.prediction.get("summary"))
        )
        if not predicted:
            return None

        verification_status = _verification_status_from_payload(self.verification.get("status"))

        if self.status == "failed" or self.failure:
            status = PredictionStatus.CONTRADICTED
            provenance = Provenance.CONTRADICTED
        elif self.verification.get("success") or verification_status == VerificationStatus.VERIFIED:
            status = PredictionStatus.FULFILLED
            provenance = Provenance.VERIFIED
        elif self.verification.get("contradiction") or verification_status == VerificationStatus.CONTRADICTED:
            status = PredictionStatus.CONTRADICTED
            provenance = Provenance.CONTRADICTED
        elif verification_status == VerificationStatus.UNVERIFIED:
            status = PredictionStatus.PENDING
            provenance = self.provenance
        else:
            status = PredictionStatus.UNKNOWN
            provenance = self.provenance

        raw_topics = self.prediction.get("topics") or self.action.get("topics") or []
        topics = _limited_unique_clean_text(
            list(raw_topics)
            if isinstance(raw_topics, Sequence) and not isinstance(raw_topics, (str, bytes))
            else [],
            lower=True,
        )

        try:
            confidence = float(self.verification.get("confidence", self.prediction.get("confidence", 0.0)) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        return PredictionRecord(
            prediction_id=_stable_id("pred", self.episode_id, self.operation, predicted, self.created_at),
            predicted_outcome=predicted,
            status=status,
            provenance=provenance,
            confidence=_clamp01(confidence),
            source_kind=f"runtime_{self.operation}",
            source_id=self.episode_id,
            created_at=self.created_at,
            topics=topics,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "trace_id": self.trace_id,
            "trace_path": self.trace_path,
            "operation": self.operation,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "latency_ms": self.latency_ms,
            "request": dict(self.request),
            "prediction": dict(self.prediction),
            "action": dict(self.action),
            "actual_output": dict(self.actual_output),
            "verification": dict(self.verification),
            "feedback": [dict(item) for item in self.feedback],
            "corrected_output": self.corrected_output,
            "provenance": self.provenance.value,
            "failure": None if self.failure is None else dict(self.failure),
        }


@dataclass(frozen=True)
class SkillMemoryRecord:
    skill_id: str
    action_type: str
    tool: str
    action_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.0
    preconditions: dict[str, Any] = field(default_factory=dict)
    trigger_context: dict[str, Any] = field(default_factory=dict)
    expected_outcomes: tuple[str, ...] = ()
    actual_outcomes: tuple[str, ...] = ()
    failure_modes: tuple[dict[str, Any], ...] = ()
    provenance: Provenance = Provenance.INFERRED
    status: str = "unverified"
    last_used_at: str = ""
    topics: tuple[str, ...] = ()
    source_action_ids: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SkillMemoryRecord":
        data = dict(payload or {})
        action_type = _clean_text(data.get("action_type") or data.get("tool")).lower()
        tool = _clean_text(data.get("tool") or action_type).lower()
        failures = tuple(
            dict(item)
            for item in list(data.get("failure_modes") or [])
            if isinstance(item, Mapping)
        )
        provenance = _provenance_value(data.get("provenance"), Provenance.INFERRED)

        return cls(
            skill_id=_clean_text(data.get("skill_id")) or _stable_id("skill", action_type, tool),
            action_type=action_type,
            tool=tool,
            action_count=_safe_int(data.get("action_count", 0)),
            success_count=_safe_int(data.get("success_count", 0)),
            failure_count=_safe_int(data.get("failure_count", 0)),
            success_rate=_clamp01(data.get("success_rate", 0.0)),
            preconditions=dict(data.get("preconditions") or {}),
            trigger_context=dict(data.get("trigger_context") or {}),
            expected_outcomes=_limited_unique_clean_text(list(data.get("expected_outcomes") or [])),
            actual_outcomes=_limited_unique_clean_text(list(data.get("actual_outcomes") or [])),
            failure_modes=failures,
            provenance=provenance,
            status=_clean_text(data.get("status")) or "unverified",
            last_used_at=_clean_text(data.get("last_used_at")),
            topics=_limited_unique_clean_text(list(data.get("topics") or []), lower=True),
            source_action_ids=_limited_unique_clean_text(list(data.get("source_action_ids") or [])),
        )

    @classmethod
    def from_action_records(
        cls,
        actions: Sequence[ActionExecutionRecord],
        *,
        limit_per_text_field: int = 8,
    ) -> tuple["SkillMemoryRecord", ...]:
        grouped: dict[str, list[ActionExecutionRecord]] = {}
        for action in actions:
            key = _clean_text(action.action_type).lower() or "unknown"
            grouped.setdefault(key, []).append(action)

        records: list[SkillMemoryRecord] = []
        for action_type in sorted(grouped):
            group = tuple(grouped[action_type])
            success_count = sum(
                1
                for action in group
                if action.verification.success or action.verification.status == VerificationStatus.VERIFIED
            )
            failure_actions = tuple(
                action
                for action in group
                if action.execution_status == ActionExecutionStatus.FAILED
                or action.verification.contradiction
                or action.verification.status == VerificationStatus.CONTRADICTED
            )
            failure_count = len(failure_actions)
            evaluated_count = success_count + failure_count

            if success_count > 0 and failure_count > 0:
                status = "mixed"
                provenance = Provenance.INFERRED
            elif success_count > 0:
                status = VerificationStatus.VERIFIED.value
                provenance = Provenance.VERIFIED
            elif failure_count > 0:
                status = VerificationStatus.CONTRADICTED.value
                provenance = Provenance.CONTRADICTED
            else:
                status = VerificationStatus.UNVERIFIED.value
                provenance = Provenance.INFERRED

            input_keys = sorted(
                {
                    _clean_text(key)
                    for action in group
                    for key in action.inputs.keys()
                    if _clean_text(key)
                }
            )
            trigger_reasons = sorted(
                {
                    _clean_text(action.trigger_reason).lower()
                    for action in group
                    if _clean_text(action.trigger_reason)
                }
            )
            trigger_queries = _limited_unique_clean_text(
                [action.trigger_query_text for action in group],
                limit=limit_per_text_field,
            )
            expected_outcomes = _limited_unique_clean_text(
                [action.prediction.predicted_outcome for action in group],
                limit=limit_per_text_field,
            )
            actual_outcomes = _limited_unique_clean_text(
                [action.actual_outcome for action in group],
                limit=limit_per_text_field,
            )
            topics = _limited_unique_clean_text(
                [
                    topic
                    for action in group
                    for topic in (*action.topics, *action.prediction.topics)
                ],
                limit=limit_per_text_field,
                lower=True,
            )

            failure_modes: list[dict[str, Any]] = []
            for action in failure_actions[:limit_per_text_field]:
                failure_modes.append(
                    {
                        "action_id": action.action_id,
                        "verification_status": action.verification.status.value,
                        "execution_status": action.execution_status.value,
                        "contradiction": bool(action.verification.contradiction),
                        "summary": action.verification.summary,
                        "actual_outcome": action.actual_outcome,
                        "recorded_at": action.recorded_at,
                    }
                )

            records.append(
                cls(
                    skill_id=_stable_id("skill", action_type),
                    action_type=action_type,
                    tool=action_type,
                    action_count=len(group),
                    success_count=success_count,
                    failure_count=failure_count,
                    success_rate=_safe_ratio(success_count, evaluated_count),
                    preconditions={
                        "observed_input_keys": input_keys,
                        "observed_trigger_reasons": trigger_reasons,
                        "requires_prediction_text": any(bool(action.prediction.predicted_outcome) for action in group),
                        "observed_action_count": len(group),
                    },
                    trigger_context={
                        "trigger_reasons": trigger_reasons,
                        "trigger_query_examples": list(trigger_queries),
                    },
                    expected_outcomes=expected_outcomes,
                    actual_outcomes=actual_outcomes,
                    failure_modes=tuple(failure_modes),
                    provenance=provenance,
                    status=status,
                    last_used_at=_latest_text([action.recorded_at for action in group]),
                    topics=topics,
                    source_action_ids=_limited_unique_clean_text(
                        [action.action_id for action in group],
                        limit=limit_per_text_field,
                    ),
                )
            )

        return tuple(records)

    def to_payload(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "action_type": self.action_type,
            "tool": self.tool,
            "action_count": int(self.action_count),
            "success_count": int(self.success_count),
            "failure_count": int(self.failure_count),
            "success_rate": float(self.success_rate),
            "preconditions": dict(self.preconditions),
            "trigger_context": dict(self.trigger_context),
            "expected_outcomes": list(self.expected_outcomes),
            "actual_outcomes": list(self.actual_outcomes),
            "failure_modes": [dict(item) for item in self.failure_modes],
            "provenance": self.provenance.value,
            "status": self.status,
            "last_used_at": self.last_used_at,
            "topics": list(self.topics),
            "source_action_ids": list(self.source_action_ids),
        }


@dataclass(frozen=True)
class ProvenanceState:
    observed: int = 0
    inferred: int = 0
    dreamed: int = 0
    verified: int = 0
    contradicted: int = 0
    total: int = 0

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "ProvenanceState":
        data = dict(payload or {})
        distribution = data.get("distribution")
        if isinstance(distribution, Mapping):
            data = {**dict(distribution), **data}
        values = {
            provenance.value: _safe_int(data.get(provenance.value, 0))
            for provenance in Provenance
        }
        total = _safe_int(data.get("total")) or sum(values.values())
        return cls(
            observed=values[Provenance.OBSERVED.value],
            inferred=values[Provenance.INFERRED.value],
            dreamed=values[Provenance.DREAMED.value],
            verified=values[Provenance.VERIFIED.value],
            contradicted=values[Provenance.CONTRADICTED.value],
            total=total,
        )

    @classmethod
    def from_distribution(cls, distribution: Mapping[str, Any] | None) -> "ProvenanceState":
        return cls.from_payload({"distribution": dict(distribution or {})})

    def to_payload(self) -> dict[str, Any]:
        distribution = {
            Provenance.OBSERVED.value: int(self.observed),
            Provenance.INFERRED.value: int(self.inferred),
            Provenance.DREAMED.value: int(self.dreamed),
            Provenance.VERIFIED.value: int(self.verified),
            Provenance.CONTRADICTED.value: int(self.contradicted),
        }
        total = int(self.total or sum(distribution.values()))
        return {
            **distribution,
            "total": total,
            "distribution": distribution,
        }


@dataclass(frozen=True)
class ConsolidationRecord:
    consolidation_id: str
    status: ConsolidationStatus
    origin: str
    query_text: str
    created_at: str = ""
    source_names: tuple[str, ...] = ()
    providers: tuple[str, ...] = ()
    aggregate_count: int = 1
    credit_events: int = 0
    penalty_events: int = 0
    forgiveness_events: int = 0
    trajectory_state: str = ""
    trajectory_net_score: float = 0.0
    consolidation_states: tuple[str, ...] = ()
    semantic_terms: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ConsolidationRecord":
        data = dict(payload or {})
        credit_events = _safe_int(data.get("credit_events", 0))
        penalty_events = _safe_int(data.get("penalty_events", 0))
        forgiveness_events = _safe_int(data.get("forgiveness_events", 0))
        cooling_events = _safe_int(data.get("cooling_events", 0))
        aggregate_count = _safe_int(data.get("aggregate_count", 1), minimum=1)

        if _clean_text(data.get("status")):
            status = _enum_value(ConsolidationStatus, data.get("status"), ConsolidationStatus.RAW)
        elif penalty_events > credit_events:
            status = ConsolidationStatus.PENALIZED
        elif forgiveness_events > 0:
            status = ConsolidationStatus.FORGIVEN
        elif credit_events > 0:
            status = ConsolidationStatus.CREDITED
        elif aggregate_count > 1:
            status = ConsolidationStatus.CONSOLIDATED
        elif cooling_events > 0:
            status = ConsolidationStatus.COOLING
        else:
            status = ConsolidationStatus.RAW

        source_weights = data.get("source_weights") if isinstance(data.get("source_weights"), Mapping) else {}
        provider_weights = data.get("provider_weights") if isinstance(data.get("provider_weights"), Mapping) else {}
        query_text = _clean_text(data.get("query_text"))
        created_at = _clean_text(data.get("created_at"))

        raw_query_terms = data.get("query_terms")
        query_terms = _limited_unique_clean_text(
            list(raw_query_terms)
            if isinstance(raw_query_terms, Sequence) and not isinstance(raw_query_terms, (str, bytes))
            else [],
            lower=True,
        )

        raw_states = data.get("consolidation_states") or data.get("states") or []
        explicit_states = _limited_unique_clean_text(
            list(raw_states)
            if isinstance(raw_states, Sequence) and not isinstance(raw_states, (str, bytes))
            else [],
            lower=True,
        )

        replay_count = _safe_int(data.get("replay_count", 0))
        aggregation_events = _safe_int(data.get("aggregation_events", 0))

        state_candidates: list[str] = list(explicit_states)
        if query_text or source_weights or provider_weights:
            state_candidates.append("observed")
        if _clean_text(data.get("origin")).lower().startswith("replay") or replay_count > 0:
            state_candidates.append("replayed")
        if aggregate_count > 1 or aggregation_events > 0:
            state_candidates.append("summarized")
        if credit_events > 0 or status in {ConsolidationStatus.CREDITED, ConsolidationStatus.FORGIVEN}:
            state_candidates.append("verified")
        if penalty_events > 0 or status == ConsolidationStatus.PENALIZED:
            state_candidates.append("contradicted")
        if query_terms or source_weights or provider_weights:
            state_candidates.append("semanticized")

        consolidation_states = _limited_unique_clean_text(state_candidates, lower=True)

        return cls(
            consolidation_id=_clean_text(data.get("consolidation_id") or data.get("record_id"))
                or _stable_id("con", data.get("origin"), query_text, created_at),
            status=status,  # type: ignore[arg-type]
            origin=_clean_text(data.get("origin")) or "response_selected_evidence",
            query_text=query_text,
            created_at=created_at,
            source_names=tuple(sorted(_clean_text(name) for name in source_weights.keys() if _clean_text(name))),
            providers=tuple(sorted(_clean_text(name) for name in provider_weights.keys() if _clean_text(name))),
            aggregate_count=aggregate_count,
            credit_events=credit_events,
            penalty_events=penalty_events,
            forgiveness_events=forgiveness_events,
            trajectory_state=_clean_text(data.get("trajectory_state")),
            trajectory_net_score=float(data.get("trajectory_net_score", 0.0) or 0.0),
            consolidation_states=consolidation_states,
            semantic_terms=query_terms,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "consolidation_id": self.consolidation_id,
            "status": self.status.value,
            "origin": self.origin,
            "query_text": self.query_text,
            "created_at": self.created_at,
            "source_names": list(self.source_names),
            "providers": list(self.providers),
            "aggregate_count": int(self.aggregate_count),
            "credit_events": int(self.credit_events),
            "penalty_events": int(self.penalty_events),
            "forgiveness_events": int(self.forgiveness_events),
            "trajectory_state": self.trajectory_state,
            "trajectory_net_score": float(self.trajectory_net_score),
            "consolidation_states": list(self.consolidation_states),
            "states": list(self.consolidation_states),
            "semantic_terms": list(self.semantic_terms),
        }
