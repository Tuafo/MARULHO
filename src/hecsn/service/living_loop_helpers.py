"""Shared cross-layer helper functions for the Living Loop.

This module houses the 12 private helper functions that are used across
all depth layers of the Living Loop (Records, Policy, Replay, Self-Model).
They have a single source of truth here, with no upward dependency on any
other Living Loop module.

Dependency direction: Helpers → Records → Policy → Replay → Self-Model
This module depends only on existing external packages (hecsn.cortex, stdlib).
"""
from __future__ import annotations

from enum import Enum
import hashlib
import json
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from hecsn.cortex.episodic_memory import Provenance

if TYPE_CHECKING:
    from hecsn.service.living_loop import VerificationStatus, WorldModelLiteSummary


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
    candidates = tuple(text for value in values if (text := _clean_text(value)))
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


def _verification_status_from_payload(value: Any) -> VerificationStatus:
    """Convert a payload value to a VerificationStatus enum member."""
    from hecsn.service.living_loop import VerificationStatus  # lazy: avoids circular import

    status = _clean_text(value).lower()
    if status == VerificationStatus.VERIFIED.value:
        return VerificationStatus.VERIFIED
    if status == VerificationStatus.CONTRADICTED.value:
        return VerificationStatus.CONTRADICTED
    if status in {"unverified", "pending"}:
        return VerificationStatus.UNVERIFIED
    return VerificationStatus.UNKNOWN


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result < 0.0:
        return None
    return result


def _coerce_world_model_lite(value: WorldModelLiteSummary | Mapping[str, Any] | None) -> WorldModelLiteSummary:
    """Coerce a value to a WorldModelLiteSummary instance."""
    from hecsn.service.living_loop import WorldModelLiteSummary  # lazy: avoids circular import

    if isinstance(value, WorldModelLiteSummary):
        return value
    if isinstance(value, Mapping):
        return WorldModelLiteSummary.from_payload(value)
    return WorldModelLiteSummary()
