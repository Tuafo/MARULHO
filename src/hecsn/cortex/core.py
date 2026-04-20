"""CorticalCore -- Abstract base class for LLM cortex backends.

The cortex is the language/reasoning layer of the Terminus hybrid
architecture. The SNN subcortical systems control what enters the
context and when inference fires. The cortex never initiates -- it
only responds when the SNN triggers a deliberation event.

Backends:
- NIMCortex (production): NVIDIA NIM cloud API
- MockCortex (testing): Deterministic, no network

Design decisions:
- Sync API (codebase is entirely sync, no pytest-asyncio)
- JSON-only output contract (not free-text parsing)
- Structured memory items with provenance metadata
- No local Ollama -- NIM cloud only
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)


class ThinkingMode(str, Enum):
    """Cortical operating modes -- each has a distinct system prompt."""
    THINK = "think"
    DREAM = "dream"
    REFLECT = "reflect"
    ANSWER = "answer"


@dataclass(frozen=True)
class MemoryItem:
    """A single episodic memory with SNN-computed metadata."""
    text: str
    salience: float = 0.5
    age_seconds: float = 0.0
    source: Literal["observed", "inferred", "dreamed", "verified", "external"] = "observed"
    memory_id: str = ""

    def to_prompt_str(self) -> str:
        tag = f"[{self.source}|sal={self.salience:.2f}]"
        return f"{tag} {self.text}"


@dataclass
class ContextPacket:
    """Structured input to the LLM -- budgeted slots, not free-form.

    Each slot has a clear role and token budget. The thalamic gate
    fills these slots optimally based on SNN state.
    """
    drive_summary: str = ""
    top_memories: list[MemoryItem] = field(default_factory=list)
    recent_thread: list[str] = field(default_factory=list)
    self_state: str = ""
    mode: ThinkingMode = ThinkingMode.THINK
    external_query: str = ""
    avoid_topics: list[str] = field(default_factory=list)
    forced_topic: str = ""  # SNN-injected topic to redirect thinking
    max_response_tokens: int = 160

    # Budget limits (token approximation: 1 token ~ 4 chars)
    MAX_MEMORIES: int = 8
    MAX_THREAD_ITEMS: int = 5
    MAX_DRIVE_CHARS: int = 400
    MAX_STATE_CHARS: int = 200
    MAX_QUERY_CHARS: int = 800

    def to_user_prompt(self) -> str:
        """Assemble the user-portion of the prompt from budget slots."""
        parts: list[str] = []

        if self.drive_summary:
            drive = self.drive_summary[:self.MAX_DRIVE_CHARS]
            parts.append(f"## Current Drives\n{drive}")

        if self.self_state:
            state = self.self_state[:self.MAX_STATE_CHARS]
            parts.append(f"## Internal State\n{state}")

        if self.avoid_topics:
            if self.forced_topic:
                parts.append(
                    f"## Direction\n"
                    f"Focus on: {self.forced_topic}. "
                    f"Investigate a specific fact, mechanism, or phenomenon about this."
                )
            else:
                parts.append(
                    f"## Direction\n"
                    f"Explore something fresh and concrete -- a specific fact, mechanism, "
                    f"or phenomenon you haven't considered yet."
                )

        if self.top_memories:
            mem_lines = [
                m.to_prompt_str()
                for m in self.top_memories[:self.MAX_MEMORIES]
            ]
            parts.append("## Relevant Memories\n" + "\n".join(mem_lines))

        if self.recent_thread:
            thread = self.recent_thread[-self.MAX_THREAD_ITEMS:]
            parts.append("## Recent Thoughts\n" + "\n".join(f"- {t}" for t in thread))

        if self.external_query:
            query = self.external_query[:self.MAX_QUERY_CHARS]
            parts.append(f"## External Query\n{query}")

        return "\n\n".join(parts) if parts else "No context provided. Think freely."


@dataclass(frozen=True)
class ThoughtResult:
    """Parsed LLM output -- structured signals for SNN consumption."""
    raw_text: str
    thought: str
    topics: tuple[str, ...] = ()
    emotional_valence: float = 0.0
    confidence: float = 0.5
    action_intent: Optional[str] = None
    latency_ms: float = 0.0
    parse_success: bool = True

    # Valid action intents the SNN can act on
    VALID_ACTIONS: frozenset[str] = frozenset(
        {"search", "ask", "remember", "sleep", "explore"}
    )

    @classmethod
    def from_json(cls, raw: str, latency_ms: float = 0.0) -> ThoughtResult:
        """Parse JSON response from LLM, with safe fallbacks."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # Try to extract JSON from mixed output
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

        # Parse topics — must be a list of strings
        raw_topics = data.get("topics", [])
        if isinstance(raw_topics, list):
            topics = tuple(str(t) for t in raw_topics[:8] if t)
        else:
            topics = ()

        # Parse valence (clamped -1 to 1)
        valence = float(data.get("valence", 0.0))
        valence = max(-1.0, min(1.0, valence))

        # Parse confidence (clamped 0 to 1)
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        # Parse action — only allow known actions
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
    def _fallback(cls, raw: str, latency_ms: float) -> ThoughtResult:
        """Fallback when JSON parsing fails entirely."""
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


class CorticalCore:
    """Abstract base for all cortex implementations.

    Subclasses must implement generate() and is_available().
    No local Ollama. Use NIMCortex for production, MockCortex for tests.
    """

    model: str = "abstract"
    temperature: float = 0.7

    def generate(self, context: ContextPacket) -> ThoughtResult:
        """Generate a thought from structured context. Override in subclass."""
        raise NotImplementedError("Subclasses must implement generate()")

    def is_available(self) -> bool:
        """Check if the cortex backend is reachable."""
        return False

    @property
    def generation_count(self) -> int:
        return 0

    def close(self) -> None:
        pass


class MockCortex(CorticalCore):
    """Deterministic mock cortex for testing -- no network, no LLM.

    Returns predictable responses based on the input mode and content.
    Use this in unit tests where you need the SNN control loop to work
    without any external API calls.
    """

    def __init__(
        self,
        responses: Optional[list[dict[str, Any]]] = None,
        latency_ms: float = 10.0,
        **kwargs: Any,
    ) -> None:
        self.model = "mock-cortex"
        self.temperature = 0.7
        self._generation_count = 0
        self._responses = responses or []
        self._response_idx = 0
        self._fake_latency = latency_ms

    def generate(self, context: ContextPacket) -> ThoughtResult:
        """Return a deterministic thought based on mode."""
        self._generation_count += 1

        if self._responses:
            data = self._responses[self._response_idx % len(self._responses)]
            self._response_idx += 1
            raw = json.dumps(data)
            return ThoughtResult.from_json(raw, latency_ms=self._fake_latency)

        # Default deterministic responses per mode
        mode = context.mode.value
        default_responses: dict[str, dict[str, Any]] = {
            "think": {
                "thought": f"Thinking about: {context.drive_summary[:50]}",
                "topics": ["curiosity", "exploration"],
                "valence": 0.3,
                "confidence": 0.7,
                "action": None,
            },
            "dream": {
                "thought": "Dreaming of connections between disparate memories",
                "topics": ["association", "creativity"],
                "valence": 0.1,
                "confidence": 0.4,
                "action": None,
            },
            "reflect": {
                "thought": "Reflecting on recent patterns and uncertainties",
                "topics": ["metacognition", "assessment"],
                "valence": 0.0,
                "confidence": 0.6,
                "action": None,
            },
            "answer": {
                "thought": f"Addressing query: {context.external_query[:50]}",
                "topics": ["response"],
                "valence": 0.2,
                "confidence": 0.65,
                "action": None,
            },
        }
        data = default_responses.get(mode, default_responses["think"])
        raw = json.dumps(data)
        return ThoughtResult.from_json(raw, latency_ms=self._fake_latency)

    def is_available(self) -> bool:
        return True

    @property
    def generation_count(self) -> int:
        return self._generation_count

    def close(self) -> None:
        pass


# Backwards compatibility alias -- FakeCortex is now MockCortex
FakeCortex = MockCortex
