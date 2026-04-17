"""ThoughtLoop — the multi-clock autonomous brain of Terminus.

Implements the continuous thinking cycle:
  Fast SNN tick (10ms)  → drive updates, surprise, salience
  Deliberation (event)  → LLM inference triggered by spike threshold
  Sleep (periodic)      → replay, compression, dream/hypothesis generation

The loop runs in a background thread and produces a stream of thoughts
that can be observed via callbacks or polled from the UI.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from hecsn.cortex.core import CorticalCore, ContextPacket, ThoughtResult, ThinkingMode
from hecsn.cortex.episodic_memory import EpisodicMemory, Provenance
from hecsn.cortex.drives import DriveSystem, ThalamicGate

logger = logging.getLogger(__name__)


@dataclass
class BrainStats:
    """Observable statistics of the living brain."""
    thoughts_generated: int = 0
    dreams_generated: int = 0
    sleep_cycles: int = 0
    ticks: int = 0
    total_inference_ms: float = 0.0
    last_thought: str = ""
    last_thought_time: float = 0.0
    current_mode: str = "idle"
    is_sleeping: bool = False
    memory_count: int = 0
    memory_fill_ratio: float = 0.0

    @property
    def avg_inference_ms(self) -> float:
        total = self.thoughts_generated + self.dreams_generated
        if total == 0:
            return 0.0
        return self.total_inference_ms / total


class ThoughtLoop:
    """The living brain — autonomous multi-clock cognitive loop.

    Architecture:
    - Fast loop: SNN drive updates every tick_interval_ms
    - Deliberation: LLM fires when drives cross threshold (event-driven)
    - Sleep: enters when fatigue > threshold, runs dream cycles
    - Anti-rumination: boredom circuit prevents degenerate loops
    """

    def __init__(
        self,
        cortex: CorticalCore,
        memory: Optional[EpisodicMemory] = None,
        tick_interval_ms: float = 100.0,
        min_thought_interval_s: float = 2.0,
        sleep_dream_count: int = 3,
        sleep_cooldown_s: float = 30.0,
        on_thought: Optional[Callable[[ThoughtResult], None]] = None,
        on_sleep: Optional[Callable[[list[ThoughtResult]], None]] = None,
    ) -> None:
        self.cortex = cortex
        self.memory = memory or EpisodicMemory()
        self.drives = DriveSystem()
        self.gate = ThalamicGate(self.memory, self.drives)
        self.tick_interval_s = tick_interval_ms / 1000.0
        self.min_thought_interval_s = min_thought_interval_s
        self.sleep_dream_count = sleep_dream_count
        self.sleep_cooldown_s = sleep_cooldown_s

        # Callbacks
        self._on_thought = on_thought
        self._on_sleep = on_sleep

        # State
        self.stats = BrainStats()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_thought_time = 0.0
        self._last_sleep_time = 0.0
        self._lock = threading.Lock()

    # -- Lifecycle --

    def start(self) -> None:
        """Start the autonomous thought loop in a background thread."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="terminus-thought-loop",
            daemon=True,
        )
        self._thread.start()
        logger.info("ThoughtLoop started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the thought loop gracefully."""
        if not self._running:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._running = False
        logger.info("ThoughtLoop stopped (thoughts=%d, dreams=%d)",
                     self.stats.thoughts_generated, self.stats.dreams_generated)

    @property
    def is_running(self) -> bool:
        return self._running

    # -- External interface --

    def submit_query(self, query: str) -> None:
        """Submit an external query — the brain will answer it."""
        with self._lock:
            self.gate.submit_query(query)

    def inject_observation(
        self,
        content: str,
        topics: Sequence[str] = (),
        salience: float = 0.7,
    ) -> None:
        """Inject an external observation into memory."""
        with self._lock:
            self.memory.store(
                content=content,
                provenance=Provenance.OBSERVED,
                topics=topics,
                salience=salience,
            )
            self.drives.update_from_external_input()

    def inject_surprise(
        self,
        dopamine: float = 0.5,
        serotonin: float = 0.5,
        norepinephrine: float = 0.5,
        acetylcholine: float = 0.5,
    ) -> None:
        """Inject SNN surprise signals (from SurpriseMonitor)."""
        with self._lock:
            self.drives.update_from_surprise(
                dopamine=dopamine,
                serotonin=serotonin,
                norepinephrine=norepinephrine,
                acetylcholine=acetylcholine,
            )

    # -- Synchronous single-step API (for testing) --

    def step(self) -> Optional[ThoughtResult]:
        """Execute one brain cycle synchronously. Returns thought if generated."""
        self.drives.tick()
        self.stats.ticks += 1

        now = time.time()

        # Check sleep
        if self.drives.should_sleep() and (now - self._last_sleep_time) > self.sleep_cooldown_s:
            dreams = self._sleep_cycle()
            if self._on_sleep:
                self._on_sleep(dreams)
            return None

        # Check deliberation
        if (
            self.drives.should_think()
            and (now - self._last_thought_time) > self.min_thought_interval_s
        ):
            return self._deliberate()

        return None

    # -- Core loop --

    def _loop(self) -> None:
        """Main autonomous loop — runs in background thread."""
        logger.debug("ThoughtLoop entering main loop")
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    self.drives.tick()
                    self.stats.ticks += 1
                    now = time.time()

                    # Sleep check
                    if (
                        self.drives.should_sleep()
                        and (now - self._last_sleep_time) > self.sleep_cooldown_s
                    ):
                        self.stats.current_mode = "sleeping"
                        self.stats.is_sleeping = True
                        dreams = self._sleep_cycle()
                        self.stats.is_sleeping = False
                        self.stats.current_mode = "idle"
                        if self._on_sleep and dreams:
                            self._on_sleep(dreams)
                        continue

                    # Deliberation check
                    if (
                        self.drives.should_think()
                        and (now - self._last_thought_time) > self.min_thought_interval_s
                    ):
                        self.stats.current_mode = "thinking"
                        result = self._deliberate()
                        self.stats.current_mode = "idle"
                        if self._on_thought and result:
                            self._on_thought(result)

            except Exception:
                logger.exception("ThoughtLoop error")
                time.sleep(1.0)

            self._stop_event.wait(self.tick_interval_s)

    def _deliberate(self) -> ThoughtResult:
        """Fire one deliberation cycle — LLM inference."""
        packet = self.gate.assemble()
        # Modulate temperature based on arousal
        old_temp = self.cortex.temperature
        self.cortex.temperature = 0.3 + 0.7 * self.drives.state.arousal

        result = self.cortex.generate(packet)

        self.cortex.temperature = old_temp

        # Process the thought
        self._last_thought_time = time.time()
        self.stats.thoughts_generated += 1
        self.stats.total_inference_ms += result.latency_ms
        self.stats.last_thought = result.thought
        self.stats.last_thought_time = self._last_thought_time

        # Update drives from thought
        self.drives.update_from_thought(result)

        # Record in thread
        self.gate.record_thought(result)

        # Store as episodic memory (inferred)
        self.memory.store(
            content=result.thought,
            provenance=Provenance.INFERRED,
            topics=list(result.topics),
            emotional_valence=result.emotional_valence,
            confidence=result.confidence,
            salience=max(0.3, abs(result.emotional_valence) + 0.2 * result.confidence),
        )

        # Update memory stats
        self.stats.memory_count = self.memory.size
        self.stats.memory_fill_ratio = self.memory.size / max(1, self.memory.capacity)

        logger.debug(
            "Thought #%d (%.0fms): %s",
            self.stats.thoughts_generated,
            result.latency_ms,
            result.thought[:80],
        )
        return result

    def _sleep_cycle(self) -> list[ThoughtResult]:
        """Run a sleep/dream cycle — replay memories, generate hypotheses."""
        logger.info("Entering sleep cycle")
        self._last_sleep_time = time.time()
        self.stats.sleep_cycles += 1
        dreams: list[ThoughtResult] = []

        # Mark replayed episodes
        replay_episodes = self.memory.recall_for_sleep(top_k=10)
        for ep in replay_episodes:
            ep.replay_count += 1

        # Generate dream thoughts
        for i in range(self.sleep_dream_count):
            packet = self.gate.assemble_for_sleep()
            result = self.cortex.generate(packet)
            self.stats.dreams_generated += 1
            self.stats.total_inference_ms += result.latency_ms

            # Store dream as hypothesis (requires validation to graduate)
            self.memory.store(
                content=result.thought,
                provenance=Provenance.DREAMED,
                topics=list(result.topics),
                emotional_valence=result.emotional_valence,
                confidence=result.confidence * 0.5,  # Dreams are low-confidence
                salience=0.4,
            )
            dreams.append(result)

        # Reset fatigue after sleep
        self.drives.state.fatigue = max(0.0, self.drives.state.fatigue - 0.5)

        logger.info("Sleep cycle complete: %d dreams generated", len(dreams))
        return dreams
