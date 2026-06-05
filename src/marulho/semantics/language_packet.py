"""Subcortex-owned language/readout packet contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class ReadoutMode(str, Enum):
    """Language/readout modes selected by Subcortex control."""

    THINK = "think"
    DREAM = "dream"
    REFLECT = "reflect"
    ANSWER = "answer"


class DeliberationDepth(str, Enum):
    """Deliberation/readout depth selected from Subcortex pressure."""

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


@dataclass(frozen=True)
class MemoryItem:
    """A single grounded or remembered item with Subcortex metadata."""

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
    """Structured language/readout context with budgeted evidence slots."""

    drive_summary: str = ""
    top_memories: list[MemoryItem] = field(default_factory=list)
    grounded_evidence: list[MemoryItem] = field(default_factory=list)
    self_state: str = ""
    mode: ReadoutMode = ReadoutMode.THINK
    external_query: str = ""
    avoid_topics: list[str] = field(default_factory=list)
    forced_topic: str = ""
    narrative_self: str = ""
    working_memory_narrative: str = ""
    deliberation_phase: str = ""
    max_response_tokens: int = 160

    MAX_MEMORIES: int = 8
    MAX_GROUNDED_EVIDENCE: int = 4
    MAX_DRIVE_CHARS: int = 400
    MAX_STATE_CHARS: int = 200
    MAX_NARRATIVE_CHARS: int = 500
    MAX_QUERY_CHARS: int = 800

    def to_user_prompt(self) -> str:
        """Assemble the text readout context from structured slots."""
        parts: list[str] = []

        if self.drive_summary:
            parts.append(f"## Current Drives\n{self.drive_summary[:self.MAX_DRIVE_CHARS]}")

        if self.self_state:
            parts.append(f"## Internal State\n{self.self_state[:self.MAX_STATE_CHARS]}")

        if self.avoid_topics:
            if self.forced_topic:
                parts.append(
                    f"## Direction\n"
                    f"Think about: {self.forced_topic}. "
                    f"Share one specific fact, mechanism, or phenomenon about this topic."
                )
            else:
                parts.append(
                    f"## Direction\n"
                    f"Switch to a completely new domain. Pick one specific fact about "
                    f"geology, music, medicine, engineering, marine biology, or space "
                    f"exploration. Be concrete and specific."
                )
        elif self.forced_topic:
            parts.append(
                f"## Direction\n"
                f"Think about: {self.forced_topic}. "
                f"Share one specific fact, mechanism, or phenomenon about this topic."
            )

        if self.narrative_self:
            parts.append(f"## Ongoing Narrative\n{self.narrative_self[:self.MAX_NARRATIVE_CHARS]}")

        if self.working_memory_narrative:
            parts.append(f"## Working Memory\n{self.working_memory_narrative}")

        if self.external_query:
            parts.append(f"## External Query\n{self.external_query[:self.MAX_QUERY_CHARS]}")

        if self.grounded_evidence:
            evidence_lines = [
                item.to_prompt_str()
                for item in self.grounded_evidence[:self.MAX_GROUNDED_EVIDENCE]
            ]
            parts.append("## Grounded Evidence\n" + "\n".join(evidence_lines))

        if self.top_memories:
            memory_lines = [
                item.to_prompt_str()
                for item in self.top_memories[:self.MAX_MEMORIES]
            ]
            parts.append("## Relevant Memories\n" + "\n".join(memory_lines))

        return "\n\n".join(parts) if parts else "No context provided. Think freely."


__all__ = ["ContextPacket", "MemoryItem", "ReadoutMode", "DeliberationDepth"]
