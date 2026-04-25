from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, Sequence

from hecsn.cortex.episodic_memory import Provenance
from hecsn.service.action_loop import ActionVerification, DigitalActionResult


def _stable_id(prefix: str, *parts: Any) -> str:
    seed = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:12]}"


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0.0 else float(numerator) / float(denominator)


def _limited_unique_clean_text(values: Sequence[Any], *, limit: int = 8, lower: bool = False) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        if lower:
            text = text.lower()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
        if len(cleaned) >= max(1, int(limit)):
            break
    return tuple(cleaned)


def _latest_text(values: Sequence[Any]) -> str:
    candidates = tuple(_clean_text(value) for value in values if _clean_text(value))
    return max(candidates) if candidates else ""


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_payload"):
        payload = value.to_payload()
        if isinstance(payload, Mapping):
            return payload
    return {}


def _enum_value(enum_cls: type[Enum], value: Any, default: Enum) -> Enum:
    if isinstance(value, enum_cls):
        return value
    normalized = _clean_text(value).lower()
    for item in enum_cls:
        if str(item.value).lower() == normalized or item.name.lower() == normalized:
            return item
    return default


def _provenance_value(value: Any, default: Provenance = Provenance.INFERRED) -> Provenance:
    if isinstance(value, Provenance):
        return value
    normalized = _clean_text(value).lower()
    for provenance in Provenance:
        if provenance.value == normalized or provenance.name.lower() == normalized:
            return provenance
    return default


def _verification_status_from_payload(value: Any) -> "VerificationStatus":
    status = _clean_text(value).lower()
    if status == VerificationStatus.VERIFIED.value:
        return VerificationStatus.VERIFIED
    if status == VerificationStatus.CONTRADICTED.value:
        return VerificationStatus.CONTRADICTED
    if status in {"unverified", "pending"}:
        return VerificationStatus.UNVERIFIED
    return VerificationStatus.UNKNOWN


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
            prediction_id=_clean_text(data.get("prediction_id"))
            or _stable_id("pred", action_id, predicted, created_at),
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
            verification_id=_clean_text(data.get("verification_id"))
            or _stable_id("ver", action_id, status.value, summary, evidence),
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
            action_count=max(0, int(data.get("action_count", 0) or 0)),
            success_count=max(0, int(data.get("success_count", 0) or 0)),
            failure_count=max(0, int(data.get("failure_count", 0) or 0)),
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
            provenance.value: max(0, int(data.get(provenance.value, 0) or 0))
            for provenance in Provenance
        }
        total = int(data.get("total", sum(values.values())) or sum(values.values()))
        return cls(
            observed=values[Provenance.OBSERVED.value],
            inferred=values[Provenance.INFERRED.value],
            dreamed=values[Provenance.DREAMED.value],
            verified=values[Provenance.VERIFIED.value],
            contradicted=values[Provenance.CONTRADICTED.value],
            total=max(0, total),
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
        credit_events = max(0, int(data.get("credit_events", 0) or 0))
        penalty_events = max(0, int(data.get("penalty_events", 0) or 0))
        forgiveness_events = max(0, int(data.get("forgiveness_events", 0) or 0))
        cooling_events = max(0, int(data.get("cooling_events", 0) or 0))
        aggregate_count = max(1, int(data.get("aggregate_count", 1) or 1))
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
            list(raw_query_terms) if isinstance(raw_query_terms, Sequence) and not isinstance(raw_query_terms, (str, bytes)) else [],
            lower=True,
        )
        raw_states = data.get("consolidation_states") or data.get("states") or []
        explicit_states = _limited_unique_clean_text(
            list(raw_states) if isinstance(raw_states, Sequence) and not isinstance(raw_states, (str, bytes)) else [],
            lower=True,
        )
        try:
            replay_count = max(0, int(data.get("replay_count", 0) or 0))
        except (TypeError, ValueError):
            replay_count = 0
        try:
            aggregation_events = max(0, int(data.get("aggregation_events", 0) or 0))
        except (TypeError, ValueError):
            aggregation_events = 0
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
        action_loop: Mapping[str, Any] | None = None,
        memory: Mapping[str, Any] | None = None,
        narrative: Mapping[str, Any] | None = None,
        cortex: Mapping[str, Any] | None = None,
        generated_at: str | None = None,
    ) -> "OperationalSelfModel":
        prediction_records = tuple(predictions)
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
            "predictions": [item.to_payload() for item in self.predictions],
            "actions": [item.to_payload() for item in self.actions],
            "consolidations": [item.to_payload() for item in self.consolidations],
            "world_model_lite": world_model_lite.to_payload(),
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
