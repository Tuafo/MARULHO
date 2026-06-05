"""Structured language/readout result for Subcortex-owned surfaces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LanguageResult:
    """Parsed language/readout output with explicit control signals."""

    raw_text: str
    thought: str
    topics: tuple[str, ...] = ()
    emotional_valence: float = 0.0
    confidence: float = 0.5
    action_intent: Optional[str] = None
    latency_ms: float = 0.0
    parse_success: bool = True

    VALID_ACTIONS: frozenset[str] = frozenset(
        {"search", "ask", "remember", "sleep", "explore"}
    )

    @classmethod
    def from_json(cls, raw: str, latency_ms: float = 0.0) -> LanguageResult:
        """Parse a structured language payload, with deterministic fallbacks."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(raw[start:end])
                except (json.JSONDecodeError, TypeError):
                    return cls._fallback(raw, latency_ms)
            else:
                return cls._fallback(raw, latency_ms)

        if not isinstance(data, dict):
            return cls._fallback(raw, latency_ms)

        thought = str(data.get("thought", ""))[:500]
        if not thought:
            return cls._fallback(raw, latency_ms)

        raw_topics = data.get("topics", [])
        topics = tuple(str(topic) for topic in raw_topics[:8] if topic) if isinstance(raw_topics, list) else ()

        valence = max(-1.0, min(1.0, float(data.get("valence", 0.0))))
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))

        raw_action = data.get("action")
        action: Optional[str] = None
        if raw_action and str(raw_action).lower() not in ("none", "null", ""):
            action_str = str(raw_action).lower().strip()
            if action_str in cls.VALID_ACTIONS:
                action = action_str

        return cls(
            raw_text=raw,
            thought=thought,
            topics=topics,
            emotional_valence=valence,
            confidence=confidence,
            action_intent=action,
            latency_ms=latency_ms,
            parse_success=True,
        )

    @classmethod
    def _fallback(cls, raw: str, latency_ms: float) -> LanguageResult:
        """Fallback when structured parsing fails entirely."""
        return cls(
            raw_text=raw,
            thought=raw[:200].strip(),
            topics=(),
            emotional_valence=0.0,
            confidence=0.3,
            action_intent=None,
            latency_ms=latency_ms,
            parse_success=False,
        )


__all__ = ["LanguageResult"]
