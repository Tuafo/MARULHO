"""CorticalCore — frozen LLM neocortex for the Terminus living brain.

Wraps Ollama (Gemma 4) with a structured context-budget interface.
The SNN subcortical systems control what enters the context and when
inference fires.  The cortex never initiates — it only responds when
the SNN triggers a deliberation event.

Design decisions (from rubber-duck critique):
- Sync API (codebase is entirely sync, no pytest-asyncio)
- JSON-only output contract (not free-text parsing)
- Structured memory items with provenance metadata
- Loopback-only Ollama by default + strict timeouts
- FakeCortex for deterministic testing without LLM
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional, Sequence

import httpx

from hecsn.cortex.prompts import MODE_PROMPTS

logger = logging.getLogger(__name__)


class ThinkingMode(str, Enum):
    """Cortical operating modes — each has a distinct system prompt."""
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
    """Structured input to the LLM — budgeted slots, not free-form.

    Each slot has a clear role and token budget.  The thalamic gate
    (Phase 3) will learn to fill these slots optimally.
    """
    drive_summary: str = ""
    top_memories: list[MemoryItem] = field(default_factory=list)
    recent_thread: list[str] = field(default_factory=list)
    self_state: str = ""
    mode: ThinkingMode = ThinkingMode.THINK
    external_query: str = ""
    avoid_topics: list[str] = field(default_factory=list)
    max_response_tokens: int = 256

    # Budget limits (token approximation: 1 token ≈ 4 chars)
    MAX_MEMORIES: int = 5
    MAX_THREAD_ITEMS: int = 3
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
            avoid_str = ", ".join(self.avoid_topics[:8])
            parts.append(
                f"## Direction\n"
                f"Explore something fresh and concrete — a specific fact, mechanism, "
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
    """Parsed LLM output — structured signals for SNN consumption."""
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

        thought = str(data.get("thought", raw[:200]))
        topics_raw = data.get("topics", [])
        if isinstance(topics_raw, list):
            topics = tuple(str(t) for t in topics_raw[:8])
        else:
            topics = ()

        valence = max(-1.0, min(1.0, float(data.get("valence", 0.0))))
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))

        action = data.get("action")
        if action is not None:
            action = str(action).lower().strip()
            if action not in cls.VALID_ACTIONS or action == "null":
                action = None

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
    """Frozen LLM neocortex — generates thoughts from structured context.

    The cortex is stateless between calls.  All state (memory, drives,
    continuity) lives in the SNN subcortical systems and is passed in
    via ContextPacket.
    """

    def __init__(
        self,
        model: str = "gemma4:e4b",
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 120.0,
        temperature: float = 0.7,
    ) -> None:
        # Restrict to loopback for security
        if "127.0.0.1" not in base_url and "localhost" not in base_url:
            raise ValueError(
                f"CorticalCore only connects to loopback Ollama. Got: {base_url}"
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self.temperature = temperature
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_seconds))
        self._generation_count = 0

    def generate(self, context: ContextPacket) -> ThoughtResult:
        """Generate a thought from structured context.

        This is the ONLY entry point the SNN calls.  The SNN decides
        when to call it (event-driven, not every tick).
        """
        system_prompt = MODE_PROMPTS.get(context.mode.value, MODE_PROMPTS["think"])
        user_prompt = context.to_user_prompt()

        t0 = time.perf_counter()
        try:
            raw = self._call_ollama(system_prompt, user_prompt, context.max_response_tokens)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.warning("Cortex inference failed: %s", exc)
            return ThoughtResult(
                raw_text=str(exc),
                thought="[cortex unavailable]",
                confidence=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                parse_success=False,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        self._generation_count += 1
        result = ThoughtResult.from_json(raw, latency_ms=latency_ms)
        logger.debug(
            "Cortex gen #%d: %.0fms, parse=%s, action=%s",
            self._generation_count,
            latency_ms,
            result.parse_success,
            result.action_intent,
        )
        return result

    def _call_ollama(self, system: str, user: str, max_tokens: int) -> str:
        """HTTP POST to Ollama /api/generate endpoint."""
        payload: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "prompt": user,
            "stream": False,
            "format": "json",
            "options": {
                "num_predict": max_tokens,
                "temperature": self.temperature,
            },
        }
        resp = self._client.post(f"{self.base_url}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")

    def is_available(self) -> bool:
        """Check if Ollama is reachable and model is loaded."""
        try:
            resp = self._client.get(
                f"{self.base_url}/api/tags",
                timeout=httpx.Timeout(5.0),
            )
            if resp.status_code != 200:
                return False
            models = resp.json().get("models", [])
            return any(m.get("name", "").startswith(self.model.split(":")[0]) for m in models)
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    @property
    def generation_count(self) -> int:
        return self._generation_count

    def close(self) -> None:
        self._client.close()

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


class FakeCortex(CorticalCore):
    """Deterministic mock cortex for testing — no LLM needed.

    Returns predictable responses based on the input mode and content,
    useful for testing the SNN control loop without Ollama.
    """

    def __init__(
        self,
        responses: Optional[list[dict[str, Any]]] = None,
        latency_ms: float = 10.0,
        **kwargs: Any,
    ) -> None:
        # Don't call super().__init__ — we don't need httpx
        self.model = "fake-cortex"
        self.base_url = "http://127.0.0.1:0"
        self.timeout = 1.0
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

    def _call_ollama(self, system: str, user: str, max_tokens: int) -> str:
        raise NotImplementedError("FakeCortex does not call Ollama")

    def close(self) -> None:
        pass
