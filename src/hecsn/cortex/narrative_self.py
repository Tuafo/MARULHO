"""Narrative Self -- persistent content-level identity for Terminus.

This module implements the Phase 3 "narrative self" from
reports/deep_architecture_analysis.md. The goal is not meta-rumination
about internal drives, but a compact autobiographical summary of what the
system has been exploring, which questions remain open, and what has been
surprising.

The narrative is intentionally lightweight and deterministic:
- updated from thought results, not from a second LLM call
- persisted to disk so it survives manager restarts
- injected into prompts only when continuity is useful (queries,
  reflection, and multi-step chains)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from hecsn.cortex.core import ThoughtDepth, ThoughtResult


@dataclass
class NarrativeSelf:
    """Persistent autobiographical summary of recent cognition.

    The narrative tracks *content* continuity:
    - topics that have repeatedly mattered
    - unresolved questions
    - recent insights
    - surprising or low-confidence observations

    It deliberately avoids drive/state language ("I feel curious", etc.)
    because that led to self-referential loops. Instead it answers:
    "What have I been thinking about lately, and why does it matter?"
    """

    persistence_path: str | Path | None = None
    refresh_interval: int = 4
    thought_count: int = 0
    summary: str = ""
    interests: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    surprises: list[str] = field(default_factory=list)
    recent_insights: list[str] = field(default_factory=list)
    _topic_counts: dict[str, int] = field(default_factory=dict)
    _last_updated: float = field(default_factory=time.time)
    _last_saved: float = 0.0

    def __post_init__(self) -> None:
        self._load()
        if not self.summary:
            self._regenerate_summary()

    def observe_thought(
        self,
        result: ThoughtResult,
        depth: ThoughtDepth | str = ThoughtDepth.QUICK,
    ) -> None:
        """Integrate one stored thought into the autobiographical narrative."""
        text = (result.thought or "").strip()
        if not text:
            return

        self.thought_count += 1
        self._last_updated = time.time()
        depth_value = depth.value if isinstance(depth, ThoughtDepth) else str(depth)

        for topic in result.topics:
            key = str(topic).strip().lower()
            if key:
                self._topic_counts[key] = self._topic_counts.get(key, 0) + 1

        self._rebuild_interests()
        self._capture_open_question(text)
        self._capture_surprise(result)
        self._capture_insight(text, result.confidence, depth_value)

        if self.thought_count % max(1, self.refresh_interval) == 0 or not self.summary:
            self._regenerate_summary()

        self._save_if_due()

    def to_prompt(self) -> str:
        """Return the concise narrative text for prompt injection."""
        return self.summary.strip()

    def snapshot(self) -> dict[str, Any]:
        """Thread-safe read snapshot for UI/debugging/reporting."""
        return {
            "summary": self.summary,
            "thought_count": self.thought_count,
            "interests": list(self.interests),
            "open_questions": list(self.open_questions),
            "surprises": list(self.surprises),
            "recent_insights": list(self.recent_insights),
            "last_updated": self._last_updated,
        }

    def save(self) -> None:
        """Persist narrative state to disk if a path is configured."""
        if self.persistence_path is None:
            return
        path = Path(self.persistence_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "thought_count": self.thought_count,
            "summary": self.summary,
            "interests": self.interests,
            "open_questions": self.open_questions,
            "surprises": self.surprises,
            "recent_insights": self.recent_insights,
            "topic_counts": self._topic_counts,
            "last_updated": self._last_updated,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._last_saved = time.time()

    def _save_if_due(self, min_interval_s: float = 2.0) -> None:
        if self.persistence_path is None:
            return
        if (time.time() - self._last_saved) >= min_interval_s:
            self.save()

    def _load(self) -> None:
        if self.persistence_path is None:
            return
        path = Path(self.persistence_path)
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return

        self.thought_count = int(payload.get("thought_count", self.thought_count))
        self.summary = str(payload.get("summary", self.summary))
        self.interests = self._coerce_list(payload.get("interests"), limit=5)
        self.open_questions = self._coerce_list(payload.get("open_questions"), limit=3)
        self.surprises = self._coerce_list(payload.get("surprises"), limit=3)
        self.recent_insights = self._coerce_list(payload.get("recent_insights"), limit=3)
        topic_counts = payload.get("topic_counts", {})
        if isinstance(topic_counts, dict):
            cleaned: dict[str, int] = {}
            for key, value in topic_counts.items():
                name = str(key).strip().lower()
                if not name:
                    continue
                try:
                    cleaned[name] = int(value)
                except (TypeError, ValueError):
                    continue
            self._topic_counts = cleaned
        self._last_updated = float(payload.get("last_updated", self._last_updated))

    @staticmethod
    def _coerce_list(values: Any, *, limit: int) -> list[str]:
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
            return []
        result: list[str] = []
        for value in values:
            text = str(value).strip()
            if text and text not in result:
                result.append(text[:200])
            if len(result) >= limit:
                break
        return result

    def _rebuild_interests(self) -> None:
        ranked = sorted(
            self._topic_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        self.interests = [topic for topic, _count in ranked[:5]]

    def _capture_open_question(self, text: str) -> None:
        lower = text.lower()
        is_question = (
            "?" in text
            or lower.startswith(("how ", "why ", "what ", "when ", "where ", "could ", "might "))
            or " i wonder " in f" {lower} "
        )
        if not is_question:
            return
        question = self._best_question_excerpt(text)
        self._push_unique(self.open_questions, question, limit=3)

    def _capture_surprise(self, result: ThoughtResult) -> None:
        text = result.thought.strip()
        lower = text.lower()
        surprising = (
            result.confidence < 0.45
            or any(token in lower for token in ("however", "despite", "unexpected", "surprising", "yet "))
        )
        if surprising:
            excerpt = self._best_statement_excerpt(text)
            self._push_unique(self.surprises, excerpt, limit=3)

    def _capture_insight(self, text: str, confidence: float, depth_value: str) -> None:
        lower = text.lower()
        insight_like = (
            confidence >= 0.6
            or depth_value == ThoughtDepth.DEEP.value
            or any(token in lower for token in ("therefore", "suggests", "reveals", "this means"))
        )
        if insight_like:
            excerpt = self._best_statement_excerpt(text)
            self._push_unique(self.recent_insights, excerpt, limit=3)

    @staticmethod
    def _push_unique(target: list[str], text: str, *, limit: int) -> None:
        item = text.strip()[:200]
        if not item:
            return
        if item in target:
            target.remove(item)
        target.insert(0, item)
        del target[limit:]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text.strip()) if segment.strip()]

    def _best_question_excerpt(self, text: str) -> str:
        sentences = self._split_sentences(text)
        for sentence in sentences:
            lower = sentence.lower()
            if (
                "?" in sentence
                or lower.startswith(("how ", "why ", "what ", "when ", "where ", "could ", "might "))
                or " i wonder " in f" {lower} "
            ):
                return sentence
        return text.strip()

    def _best_statement_excerpt(self, text: str) -> str:
        sentences = self._split_sentences(text)
        for sentence in sentences:
            if "?" not in sentence:
                return sentence
        return sentences[0] if sentences else text.strip()

    def _regenerate_summary(self) -> None:
        parts: list[str] = []

        if self.interests:
            if len(self.interests) == 1:
                parts.append(f"I've recently been exploring {self.interests[0]}.")
            elif len(self.interests) == 2:
                parts.append(
                    f"I've recently been exploring {self.interests[0]} and {self.interests[1]}."
                )
            else:
                interests = ", ".join(self.interests[:3])
                parts.append(f"I've recently been exploring {interests}.")

        if self.recent_insights:
            parts.append(f"A recurring idea is: {self.recent_insights[0]}")

        if self.open_questions:
            parts.append(f"An open question is: {self.open_questions[0]}")

        if self.surprises:
            parts.append(f"What still feels unresolved: {self.surprises[0]}")

        self.summary = " ".join(parts).strip()[:600]
