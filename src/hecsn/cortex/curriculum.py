"""LLM-guided curriculum generator — cortex-driven training data selection.

Instead of blindly streaming Wikipedia, uses the NIM cortex to generate
training episodes that target HECSN's identified knowledge gaps. This
makes HECSN's training truly autonomous and curiosity-driven.

The GeometricCuriosityController identifies which concepts have the
largest gaps (high uncertainty + drift). The CurriculumGenerator asks
the LLM to produce short educational texts about those specific topics,
with explicit sensory descriptions (visual, audio) that can be used
for cross-modal grounding.

This is the cortex→SNN feedback loop applied to curriculum selection:
the cortex decides WHAT to learn, the SNN decides HOW to learn it.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Sequence

import httpx

logger = logging.getLogger(__name__)

DEFAULT_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_CURRICULUM_MODEL = "nvidia/llama-3.1-nemotron-nano-8b-v1"

SYSTEM_PROMPT = """You are a curriculum generator for a grounded learning system.
Given a list of topics the system needs to learn about, produce educational text.

RULES:
1. Each text must be 2-4 sentences about the specific topic.
2. Include concrete SENSORY descriptions: what it LOOKS like (visual), what it SOUNDS like (audio).
3. Use simple, factual language. No metaphors or abstractions.
4. Output ONLY valid JSON matching this schema:
{
  "segments": [
    {
      "text": "The educational text here with visual and audio descriptions.",
      "topic": "primary_topic",
      "visual_hint": "brief visual description",
      "audio_hint": "brief audio description"
    }
  ]
}

Each segment should be 20-50 words. Generate 3-6 segments per request."""


@dataclass
class CurriculumSegment:
    """A single curriculum segment with text and sensory hints."""
    text: str
    topic: str = ""
    visual_hint: str = ""
    audio_hint: str = ""


class CurriculumGenerator:
    """Generates training curriculum targeting HECSN's knowledge gaps.

    Uses NIM LLM to produce educational text about concepts the system
    has identified as uncertain or drifting. The generated text includes
    explicit visual and audio descriptions for cross-modal grounding.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_CURRICULUM_MODEL,
        base_url: str = DEFAULT_NIM_BASE_URL,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self._generation_count = 0
        self._cache: dict[str, list[CurriculumSegment]] = {}

        if self._api_key:
            self._client = httpx.Client(
                timeout=httpx.Timeout(timeout_seconds),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        else:
            self._client = None

    def generate(
        self,
        gap_topics: Sequence[str],
        max_segments: int = 6,
    ) -> list[CurriculumSegment]:
        """Generate curriculum segments targeting specific knowledge gaps.

        Args:
            gap_topics: List of topics the system needs to learn about.
            max_segments: Maximum number of segments to generate.

        Returns:
            List of CurriculumSegment with text and sensory hints.
        """
        if not gap_topics:
            return []

        # Check cache
        cache_key = ",".join(sorted(gap_topics))
        if cache_key in self._cache:
            return self._cache[cache_key][:max_segments]

        if self._client is None:
            return self._fallback(gap_topics, max_segments)

        topics_str = ", ".join(gap_topics[:8])
        user_prompt = (
            f"Generate educational segments about these topics the system "
            f"has knowledge gaps in: {topics_str}. "
            f"Include concrete visual and audio descriptions for each."
        )

        try:
            segments = self._call_nim(user_prompt, max_segments)
            self._generation_count += 1
            if segments:
                self._cache[cache_key] = segments
            return segments[:max_segments]
        except Exception as exc:
            logger.warning("Curriculum generation failed: %s", exc)
            return self._fallback(gap_topics, max_segments)

    def _call_nim(
        self, user_prompt: str, max_segments: int
    ) -> list[CurriculumSegment]:
        """Call NIM to generate curriculum segments."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 512,
            "temperature": 0.8,
        }
        resp = self._client.post(  # type: ignore[union-attr]
            f"{self.base_url}/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return self._parse_segments(content, max_segments)

    def _parse_segments(
        self, raw: str, max_segments: int
    ) -> list[CurriculumSegment]:
        """Parse JSON curriculum output from LLM."""
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(raw[start:end])
                except (json.JSONDecodeError, TypeError):
                    return []

        segments_raw = parsed.get("segments", [])
        if not isinstance(segments_raw, list):
            return []

        result: list[CurriculumSegment] = []
        for seg in segments_raw[:max_segments]:
            if not isinstance(seg, dict):
                continue
            text = str(seg.get("text", "")).strip()
            if not text or len(text) < 10:
                continue
            result.append(CurriculumSegment(
                text=text,
                topic=str(seg.get("topic", "")).strip(),
                visual_hint=str(seg.get("visual_hint", "")).strip(),
                audio_hint=str(seg.get("audio_hint", "")).strip(),
            ))
        return result

    def _fallback(
        self, topics: Sequence[str], max_segments: int
    ) -> list[CurriculumSegment]:
        """Generate simple template-based curriculum when NIM unavailable."""
        segments: list[CurriculumSegment] = []
        templates = [
            "The {topic} has distinct physical properties that can be observed and measured.",
            "When encountering {topic}, one can perceive it through multiple sensory channels.",
            "{topic} exhibits characteristic patterns in nature that reveal its underlying structure.",
        ]
        for i, topic in enumerate(topics[:max_segments]):
            tmpl = templates[i % len(templates)]
            text = tmpl.replace("{topic}", topic)
            segments.append(CurriculumSegment(
                text=text,
                topic=topic,
                visual_hint=f"visual appearance of {topic}",
                audio_hint=f"sound associated with {topic}",
            ))
        return segments

    @property
    def generation_count(self) -> int:
        return self._generation_count

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
