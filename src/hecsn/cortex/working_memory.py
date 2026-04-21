"""Working Memory — the brain's active reasoning buffer (Global Workspace).

Holds 3-7 items that the brain is actively reasoning over inside a single
multi-step deliberation chain. In the current architecture, working memory
is cleared at the start of each new wakeful chain to prevent cross-cycle
rumination, while still preserving within-chain continuity.

The key difference from episodic memory:
- Episodic memory = long-term storage (hundreds of items, slow recall)
- Working memory = active scratchpad (5 items, instant access, fast decay)

The broadcast() method generates a compressed narrative of current working
memory state. This is what the LLM sees — NOT raw thought history (which
causes repetition), but a regenerated summary of active reasoning.

Neuroscience basis:
- Baddeley's Working Memory Model (2000): central executive + buffers
- Global Workspace Theory (Baars 1988): information becomes "conscious"
  when broadcast globally from working memory
- Activity-silent WM (Nature 2024): items held in synaptic weights, not
  sustained firing — maps to strength-based decay in this implementation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class WMItemType(str, Enum):
    """What kind of thing is held in working memory."""
    OBSERVATION = "observation"    # A fact or sensory input
    QUESTION = "question"          # An open question being explored
    HYPOTHESIS = "hypothesis"      # A tentative connection or idea
    TENSION = "tension"            # A contradiction or unresolved conflict
    INSIGHT = "insight"            # A synthesized understanding


@dataclass
class WorkingMemoryItem:
    """A single item in working memory.

    Items have strength that decays over time. When strength drops below
    a threshold, the item is evicted. New thoughts can refresh existing
    items (strengthening them) or compete for slots.
    """
    content: str
    item_type: WMItemType
    strength: float = 1.0          # 1.0 = fresh, decays toward 0
    created_at: float = field(default_factory=time.time)
    last_refreshed: float = field(default_factory=time.time)
    topic: str = ""                # Primary topic for matching
    source_thought: str = ""       # Which thought created this

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def refresh(self, new_strength: float = 0.8) -> None:
        """Refresh this item — something relevant was thought about."""
        self.strength = min(1.0, self.strength + new_strength * 0.3)
        self.last_refreshed = time.time()

    def decay(self, rate: float = 0.05) -> None:
        """Apply time-based decay. Called each thought cycle."""
        self.strength = max(0.0, self.strength - rate)


class WorkingMemory:
    """Global workspace — what the brain is actively thinking about.

    Capacity is deliberately small (5 items) to mirror biological
    working memory limits (~4±1 items, Cowan 2001). When full,
    new items compete with the weakest existing item.

    In the present Terminus loop this workspace persists across the
    phases of one deliberation chain (observe → question → reason →
    synthesize) and is reset before the next wakeful chain begins.

    Thread safety: all mutations are simple attribute assignments
    on a small list. The thought loop is single-threaded for the
    deliberation path, so no lock is needed here. The snapshot()
    method can be called from any thread (reads only).
    """

    def __init__(self, capacity: int = 5, decay_rate: float = 0.05) -> None:
        self.capacity = capacity
        self.decay_rate = decay_rate
        self._items: list[WorkingMemoryItem] = []

    @property
    def items(self) -> list[WorkingMemoryItem]:
        return list(self._items)

    @property
    def size(self) -> int:
        return len(self._items)

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def has_tension(self) -> bool:
        """Is there an unresolved contradiction in working memory?"""
        return any(item.item_type == WMItemType.TENSION for item in self._items)

    def has_question(self) -> bool:
        """Is there an open question being explored?"""
        return any(item.item_type == WMItemType.QUESTION for item in self._items)

    def strongest_item(self) -> Optional[WorkingMemoryItem]:
        """Get the most active item in working memory."""
        if not self._items:
            return None
        return max(self._items, key=lambda x: x.strength)

    def add(self, item: WorkingMemoryItem) -> Optional[WorkingMemoryItem]:
        """Add an item to working memory. Returns evicted item if any.

        If working memory is full, the weakest item is evicted to make
        room — but only if the new item is stronger than the weakest.
        """
        # Check if we can refresh an existing item instead
        for existing in self._items:
            if existing.topic and item.topic and _topics_overlap(existing.topic, item.topic):
                # Same topic — refresh the existing item with new content
                existing.content = item.content
                existing.item_type = item.item_type
                existing.refresh()
                return None

        if len(self._items) < self.capacity:
            self._items.append(item)
            return None

        # Working memory full — compete for a slot
        weakest = min(self._items, key=lambda x: x.strength)
        if item.strength > weakest.strength:
            self._items.remove(weakest)
            self._items.append(item)
            return weakest

        return None  # New item not strong enough to enter

    def decay_all(self) -> None:
        """Apply decay to all items. Called once per thought cycle.

        Items below threshold (0.1) are automatically evicted.
        """
        survivors: list[WorkingMemoryItem] = []
        for item in self._items:
            item.decay(self.decay_rate)
            if item.strength >= 0.1:
                survivors.append(item)
        self._items = survivors

    def clear(self) -> None:
        """Clear all working memory (e.g., major context switch)."""
        self._items.clear()

    def update_from_thought(
        self,
        thought: str,
        topics: tuple[str, ...],
        confidence: float,
        emotional_valence: float,
        item_type: WMItemType = WMItemType.OBSERVATION,
    ) -> None:
        """Integrate a new thought result into working memory.

        Called after each deliberation step. Determines whether the thought
        should enter working memory, refresh existing items, or be ignored.
        """
        if not thought.strip():
            return

        # Determine item type from content heuristics
        resolved_type = item_type
        thought_lower = thought.lower()
        if "?" in thought or any(w in thought_lower for w in ("wonder", "how does", "why do", "what if")):
            resolved_type = WMItemType.QUESTION
        elif any(w in thought_lower for w in ("however", "but ", "contradicts", "despite", "yet ")):
            resolved_type = WMItemType.TENSION
        elif any(w in thought_lower for w in ("therefore", "this means", "insight", "reveals", "suggests that")):
            resolved_type = WMItemType.INSIGHT
        elif any(w in thought_lower for w in ("perhaps", "might", "could be", "hypothesis", "possibly")):
            resolved_type = WMItemType.HYPOTHESIS

        # Determine strength based on novelty and emotion
        strength = 0.5 + 0.3 * confidence + 0.2 * abs(emotional_valence)
        # Tensions and insights get a boost (they're more important)
        if resolved_type in (WMItemType.TENSION, WMItemType.INSIGHT):
            strength = min(1.0, strength + 0.2)

        primary_topic = topics[0] if topics else ""

        item = WorkingMemoryItem(
            content=thought[:200],  # Truncate for efficiency
            item_type=resolved_type,
            strength=strength,
            topic=primary_topic,
            source_thought=thought[:80],
        )
        self.add(item)

    def broadcast(self) -> str:
        """Generate a narrative summary of working memory for the LLM.

        This is the Global Workspace broadcast — a compressed, regenerated
        summary of what the brain is currently thinking about. NOT raw
        thought history, but a synthesized narrative.

        Returns empty string if working memory is empty (cold start).
        """
        if not self._items:
            return ""

        # Sort by strength (strongest = most active thought)
        active = sorted(self._items, key=lambda x: x.strength, reverse=True)

        parts: list[str] = []

        # Strongest item = current focus
        focus = active[0]
        parts.append(f"Currently thinking about: {focus.content}")

        # Open questions
        questions = [i for i in active[1:] if i.item_type == WMItemType.QUESTION]
        if questions:
            parts.append(f"Open question: {questions[0].content}")

        # Tensions/contradictions
        tensions = [i for i in active if i.item_type == WMItemType.TENSION]
        if tensions:
            parts.append(f"Unresolved: {tensions[0].content}")

        # Recent insights
        insights = [i for i in active if i.item_type == WMItemType.INSIGHT]
        if insights:
            parts.append(f"Recent insight: {insights[0].content}")

        # Background items (observations not covered above)
        remaining = [i for i in active[1:]
                     if i.item_type not in (WMItemType.QUESTION, WMItemType.TENSION, WMItemType.INSIGHT)]
        if remaining:
            parts.append(f"Also considering: {remaining[0].content}")

        return " | ".join(parts)

    def snapshot(self) -> dict:
        """Thread-safe snapshot for UI/debugging."""
        return {
            "size": len(self._items),
            "capacity": self.capacity,
            "items": [
                {
                    "content": item.content[:100],
                    "type": item.item_type.value,
                    "strength": round(item.strength, 3),
                    "topic": item.topic,
                    "age_s": round(item.age_seconds, 1),
                }
                for item in sorted(self._items, key=lambda x: x.strength, reverse=True)
            ],
            "has_tension": self.has_tension(),
            "has_question": self.has_question(),
            "broadcast": self.broadcast()[:300],
        }


def _topics_overlap(topic_a: str, topic_b: str) -> bool:
    """Check if two topic strings share meaningful words."""
    if not topic_a or not topic_b:
        return False
    words_a = {w.lower().strip(".,;:!?'\"()-") for w in topic_a.split() if len(w) >= 3}
    words_b = {w.lower().strip(".,;:!?'\"()-") for w in topic_b.split() if len(w) >= 3}
    return bool(words_a & words_b)
