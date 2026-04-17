"""Drive system and thalamic gate — SNN control over the cortex.

The drive system converts SNN signals (surprise, curiosity, neuromodulators)
into actionable drives that determine WHEN and WHAT the LLM thinks about.
The thalamic gate assembles budgeted context packets from drives + memories.

This is the critical integration point: the SNN doesn't understand language,
but it controls the LLM's attention through these biologically-inspired
mechanisms.

Anti-rumination: boredom circuit with exponential decay on repeated topics,
diversity penalties, and verified-progress triggers prevent degenerate loops.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

from hecsn.cortex.core import ContextPacket, MemoryItem, ThinkingMode, ThoughtResult
from hecsn.cortex.episodic_memory import EpisodicMemory, Episode, Provenance

logger = logging.getLogger(__name__)


@dataclass
class DriveState:
    """Current drive intensities — computed from SNN signals."""
    curiosity: float = 0.5       # Want to explore / learn
    anxiety: float = 0.0         # Persistent unresolved surprise
    satisfaction: float = 0.3    # Recent positive outcomes
    boredom: float = 0.0         # Repeated topics / lack of novelty
    social: float = 0.0          # Want to interact / answer questions
    fatigue: float = 0.0         # Accumulated processing → need sleep

    # Neuromodulator mirrors (from SurpriseMonitor)
    dopamine: float = 0.5
    serotonin: float = 0.5
    norepinephrine: float = 0.5
    acetylcholine: float = 0.5

    @property
    def arousal(self) -> float:
        """Overall arousal level — drives LLM temperature."""
        return min(1.0, max(0.0, (
            0.3 * self.curiosity
            + 0.3 * self.norepinephrine
            + 0.2 * self.anxiety
            - 0.2 * self.fatigue
        )))

    @property
    def valence(self) -> float:
        """Emotional valence: positive = good, negative = bad."""
        return max(-1.0, min(1.0,
            self.satisfaction - self.anxiety + 0.5 * self.dopamine - 0.5 * self.serotonin
        ))

    @property
    def dominant_drive(self) -> str:
        """Which drive is currently strongest."""
        drives = {
            "curiosity": self.curiosity,
            "anxiety": self.anxiety,
            "boredom": self.boredom,
            "social": self.social,
            "fatigue": self.fatigue,
        }
        return max(drives, key=drives.get)  # type: ignore[arg-type]

    def to_summary(self) -> str:
        """Human-readable drive summary for the LLM context."""
        dom = self.dominant_drive
        parts = [f"Primary drive: {dom} ({getattr(self, dom):.2f})"]
        parts.append(f"Arousal: {self.arousal:.2f}, Valence: {self.valence:+.2f}")
        if self.curiosity > 0.6:
            parts.append("Strong curiosity — explore something new")
        if self.anxiety > 0.5:
            parts.append("Elevated anxiety — something unresolved needs attention")
        if self.boredom > 0.6:
            parts.append("Growing bored — change topic or seek external input")
        if self.fatigue > 0.7:
            parts.append("Fatigued — consider sleep/consolidation")
        return ". ".join(parts)


class AntiRuminationCircuit:
    """Prevents degenerate thought loops via boredom and diversity tracking."""

    def __init__(
        self,
        topic_decay_rate: float = 0.9,
        boredom_threshold: int = 3,
        diversity_window: int = 10,
    ) -> None:
        self.topic_decay_rate = topic_decay_rate
        self.boredom_threshold = boredom_threshold
        self.diversity_window = diversity_window
        self._topic_counts: dict[str, float] = defaultdict(float)
        self._recent_topics: list[str] = []

    def record_topics(self, topics: Sequence[str]) -> None:
        """Record topics from a thought result."""
        # Decay all existing counts
        for k in list(self._topic_counts.keys()):
            self._topic_counts[k] *= self.topic_decay_rate
            if self._topic_counts[k] < 0.01:
                del self._topic_counts[k]

        # Increment current topics
        for t in topics:
            key = t.lower().strip()
            if key:
                self._topic_counts[key] += 1.0
                self._recent_topics.append(key)

        # Trim window
        self._recent_topics = self._recent_topics[-self.diversity_window:]

    def boredom_signal(self) -> float:
        """How bored are we? Based on topic repetition."""
        if not self._topic_counts:
            return 0.0
        max_count = max(self._topic_counts.values())
        if max_count >= self.boredom_threshold:
            return min(1.0, (max_count - self.boredom_threshold + 1) * 0.3)
        return 0.0

    def diversity_score(self) -> float:
        """How diverse are recent thoughts? 0 = all same, 1 = all different."""
        if len(self._recent_topics) < 2:
            return 1.0
        unique = len(set(self._recent_topics))
        return unique / len(self._recent_topics)

    def suggest_topic_avoidance(self) -> set[str]:
        """Topics to avoid (currently over-represented)."""
        threshold = self.boredom_threshold * 0.8
        return {t for t, c in self._topic_counts.items() if c >= threshold}


class DriveSystem:
    """Converts SNN signals into drives that control the cortex.

    Updates drives each SNN tick based on:
    - Surprise monitor neuromodulators
    - Anti-rumination boredom circuit
    - Fatigue accumulation
    - External input presence
    """

    def __init__(self) -> None:
        self.state = DriveState()
        self.anti_rumination = AntiRuminationCircuit()
        self._thought_count = 0
        self._last_external_input_time = 0.0
        self._last_thought_time = 0.0

    def update_from_surprise(
        self,
        dopamine: float,
        serotonin: float,
        norepinephrine: float,
        acetylcholine: float,
    ) -> None:
        """Update neuromodulator mirrors from SurpriseMonitor."""
        alpha = 0.15
        self.state.dopamine = alpha * dopamine + (1 - alpha) * self.state.dopamine
        self.state.serotonin = alpha * serotonin + (1 - alpha) * self.state.serotonin
        self.state.norepinephrine = alpha * norepinephrine + (1 - alpha) * self.state.norepinephrine
        self.state.acetylcholine = alpha * acetylcholine + (1 - alpha) * self.state.acetylcholine

        # Derive drives from neuromodulators
        self.state.curiosity = _clamp01(
            0.4 * self.state.acetylcholine + 0.3 * self.state.dopamine + 0.3 * (1 - self.state.serotonin)
        )
        self.state.anxiety = _clamp01(
            0.5 * self.state.norepinephrine + 0.3 * self.state.serotonin - 0.2 * self.state.dopamine
        )
        self.state.satisfaction = _clamp01(
            0.6 * self.state.dopamine - 0.3 * self.state.serotonin
        )

    def update_from_thought(self, result: ThoughtResult) -> None:
        """Update drives after a thought is generated."""
        self._thought_count += 1
        self._last_thought_time = time.time()

        # Record topics for anti-rumination
        self.anti_rumination.record_topics(result.topics)
        self.state.boredom = self.anti_rumination.boredom_signal()

        # Fatigue accumulates with thoughts, decays with time
        self.state.fatigue = min(1.0, self.state.fatigue + 0.02)

    def update_from_external_input(self) -> None:
        """External input arrived — reduce boredom, increase social drive."""
        self._last_external_input_time = time.time()
        self.state.social = min(1.0, self.state.social + 0.3)
        self.state.boredom = max(0.0, self.state.boredom - 0.3)

    def tick(self) -> None:
        """Periodic drive decay/update (call on SNN fast loop)."""
        # Fatigue decays slowly
        self.state.fatigue = max(0.0, self.state.fatigue - 0.001)
        # Social drive decays without input
        self.state.social = max(0.0, self.state.social * 0.995)
        # Boredom decays slightly each tick
        self.state.boredom = max(0.0, self.state.boredom * 0.999)

    def should_think(self) -> bool:
        """Should the cortex fire a deliberation cycle?"""
        if self.state.fatigue > 0.9:
            return False  # Too tired, need sleep
        # Think when curiosity or anxiety exceed threshold
        return (
            self.state.curiosity > 0.4
            or self.state.anxiety > 0.5
            or self.state.social > 0.3
        )

    def should_sleep(self) -> bool:
        """Should we enter sleep/consolidation mode?"""
        return self.state.fatigue > 0.7 and self.state.social < 0.2

    def choose_mode(self) -> ThinkingMode:
        """Choose thinking mode based on current drives."""
        if self.state.social > 0.3:
            return ThinkingMode.ANSWER
        if self.state.boredom > 0.5:
            return ThinkingMode.REFLECT
        if self.state.anxiety > 0.6:
            return ThinkingMode.REFLECT
        return ThinkingMode.THINK

    @property
    def thought_count(self) -> int:
        return self._thought_count


class ThalamicGate:
    """Assembles context packets from drives + memories.

    The gate is the SNN's control interface to the cortex. It selects
    which memories to include, what drives to emphasize, and how to
    budget the context window — all based on current SNN state.
    """

    def __init__(
        self,
        memory: EpisodicMemory,
        drives: DriveSystem,
        max_memories: int = 5,
        max_thread: int = 3,
        max_query_queue: int = 8,
    ) -> None:
        self.memory = memory
        self.drives = drives
        self.max_memories = max_memories
        self.max_thread = max_thread
        self._thought_thread: list[str] = []
        from collections import deque
        self._query_queue: deque[str] = deque(maxlen=max_query_queue)

    def assemble(self) -> ContextPacket:
        """Build a context packet from current SNN state."""
        drive_state = self.drives.state
        mode = self.drives.choose_mode()

        # Select memories based on drive summary (what we're curious about)
        query = drive_state.to_summary()
        memories = self.memory.recall_by_similarity(query, top_k=self.max_memories)

        # Convert to MemoryItems
        mem_items = [
            MemoryItem(
                text=ep.content,
                salience=ep.salience,
                age_seconds=ep.age_seconds,
                source=ep.provenance.value if ep.provenance.value in (
                    "observed", "inferred", "dreamed", "verified", "external"
                ) else "observed",
                memory_id=ep.episode_id,
            )
            for ep in memories
        ]

        # Self state from drives
        self_state = (
            f"Arousal: {drive_state.arousal:.2f}, "
            f"Valence: {drive_state.valence:+.2f}, "
            f"Dominant: {drive_state.dominant_drive}"
        )

        # Temperature modulation based on arousal
        max_tokens = 256
        if mode == ThinkingMode.DREAM:
            max_tokens = 384  # Dreams can be longer

        packet = ContextPacket(
            drive_summary=drive_state.to_summary(),
            top_memories=mem_items,
            recent_thread=list(self._thought_thread[-self.max_thread:]),
            self_state=self_state,
            mode=mode,
            external_query=self._query_queue.popleft() if self._query_queue else "",
            max_response_tokens=max_tokens,
        )
        return packet

    def record_thought(self, result: ThoughtResult) -> None:
        """Add a thought to the continuity thread."""
        self._thought_thread.append(result.thought)
        # Keep bounded
        if len(self._thought_thread) > self.max_thread * 2:
            self._thought_thread = self._thought_thread[-self.max_thread:]

    def submit_query(self, query: str) -> None:
        """Submit an external query for the cortex to answer.

        Uses a bounded queue so multiple queries can be pending.
        """
        self._query_queue.append(query)
        self.drives.update_from_external_input()

    def assemble_for_sleep(self) -> ContextPacket:
        """Build a dream-mode context packet for sleep consolidation."""
        episodes = self.memory.recall_for_sleep(top_k=self.max_memories)
        mem_items = [
            MemoryItem(
                text=ep.content,
                salience=ep.salience,
                age_seconds=ep.age_seconds,
                source=ep.provenance.value if ep.provenance.value in (
                    "observed", "inferred", "dreamed", "verified", "external"
                ) else "observed",
                memory_id=ep.episode_id,
            )
            for ep in episodes
        ]
        return ContextPacket(
            drive_summary="Sleep consolidation — find connections between memories",
            top_memories=mem_items,
            recent_thread=[],
            self_state="Sleeping, dreaming",
            mode=ThinkingMode.DREAM,
            max_response_tokens=384,
        )


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
