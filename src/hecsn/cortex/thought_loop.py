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

from hecsn.cortex.core import CorticalCore, ThoughtResult
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

    # Thought quality metrics
    topic_diversity: float = 0.0  # Shannon entropy over topics (higher = more diverse)
    concreteness_ratio: float = 0.0  # Fraction of thoughts with verifiable content
    avg_novelty: float = 0.0  # Mean cosine distance between consecutive thought embeddings
    snn_alignment: float = 0.0  # Fraction of thought topics present in SNN concepts
    dream_verification_rate: float = 0.0  # Verified dreams / total dreams
    _topic_counts: dict = field(default_factory=dict)
    _total_topics: int = 0
    _concrete_count: int = 0
    _snn_aligned_count: int = 0
    _dreams_verified: int = 0
    _dreams_total: int = 0

    @property
    def avg_inference_ms(self) -> float:
        total = self.thoughts_generated + self.dreams_generated
        if total == 0:
            return 0.0
        return self.total_inference_ms / total

    def update_quality_metrics(
        self,
        result: ThoughtResult,
        snn_concepts: list[str] | None = None,
    ) -> None:
        """Update thought quality metrics from a new thought result."""
        import math as _math

        # Topic diversity (Shannon entropy)
        for topic in result.topics:
            key = topic.lower().strip()
            if key:
                self._topic_counts[key] = self._topic_counts.get(key, 0) + 1
                self._total_topics += 1

        if self._total_topics > 0 and self._topic_counts:
            total = float(self._total_topics)
            entropy = 0.0
            for count in self._topic_counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * _math.log2(p)
            max_entropy = _math.log2(max(1, len(self._topic_counts)))
            self.topic_diversity = entropy / max(1.0, max_entropy)

        # Concreteness: thoughts with confidence > 0.5 and specific topics
        if result.confidence > 0.5 and len(result.topics) > 0:
            self._concrete_count += 1
        total_thoughts = max(1, self.thoughts_generated)
        self.concreteness_ratio = self._concrete_count / total_thoughts

        # SNN alignment: fraction of topics present in SNN concept store
        if result.topics and snn_concepts:
            snn_lower = {c.lower() for c in snn_concepts}
            aligned = sum(
                1 for t in result.topics
                if any(t.lower() in c or c in t.lower() for c in snn_lower)
            )
            self._snn_aligned_count += aligned
            total_topic_instances = max(1, self._total_topics)
            self.snn_alignment = self._snn_aligned_count / total_topic_instances


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
        curiosity_controller: Any = None,
    ) -> None:
        self.cortex = cortex
        self.memory = memory or EpisodicMemory()
        self.drives = DriveSystem()
        self.gate = ThalamicGate(self.memory, self.drives)
        self.tick_interval_s = tick_interval_ms / 1000.0
        self.min_thought_interval_s = min_thought_interval_s
        self.sleep_dream_count = sleep_dream_count
        self.sleep_cooldown_s = sleep_cooldown_s

        # Cortex→SNN feedback: curiosity controller for routing boosts
        self._curiosity_controller = curiosity_controller

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

        # Thought history (bounded deque — append is CPython-atomic)
        from collections import deque
        self._thought_history: deque[dict[str, Any]] = deque(maxlen=50)

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
        """Stop the thought loop gracefully.

        Signal stop, then join the thread.  Callers should NOT hold
        external locks while calling stop() to avoid deadlock.
        """
        if not self._running:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._running = False
        logger.info("ThoughtLoop stopped (thoughts=%d, dreams=%d)",
                     self.stats.thoughts_generated, self.stats.dreams_generated)

    def request_stop(self) -> None:
        """Signal stop without joining — safe to call under external lock."""
        self._stop_event.set()
        self._running = False

    def set_curiosity_controller(self, controller: Any) -> None:
        """Set or update the curiosity controller for cortex→SNN feedback."""
        self._curiosity_controller = controller

    @property
    def is_running(self) -> bool:
        return self._running

    # -- Snapshot (thread-safe) --

    def snapshot(self) -> dict[str, Any]:
        """Thread-safe snapshot of brain stats + recent thoughts."""
        with self._lock:
            s = self.stats
            return {
                "enabled": True,
                "running": self._running,
                "thoughts_generated": s.thoughts_generated,
                "dreams_generated": s.dreams_generated,
                "sleep_cycles": s.sleep_cycles,
                "ticks": s.ticks,
                "avg_inference_ms": round(s.avg_inference_ms, 1),
                "last_thought": s.last_thought,
                "last_thought_time": s.last_thought_time,
                "current_mode": s.current_mode,
                "is_sleeping": s.is_sleeping,
                "memory_count": s.memory_count,
                "memory_fill_ratio": round(s.memory_fill_ratio, 3),
                "drives": {
                    "curiosity": round(self.drives.state.curiosity, 3),
                    "anxiety": round(self.drives.state.anxiety, 3),
                    "satisfaction": round(self.drives.state.satisfaction, 3),
                    "boredom": round(self.drives.state.boredom, 3),
                    "fatigue": round(self.drives.state.fatigue, 3),
                    "social": round(self.drives.state.social, 3),
                    "arousal": round(self.drives.state.arousal, 3),
                },
                "quality": {
                    "topic_diversity": round(s.topic_diversity, 3),
                    "concreteness_ratio": round(s.concreteness_ratio, 3),
                    "snn_alignment": round(s.snn_alignment, 3),
                    "dream_verification_rate": round(s.dream_verification_rate, 3),
                },
                "recent_thoughts": list(self._thought_history),
            }

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

    def step(self, *, force: bool = False) -> Optional[ThoughtResult]:
        """Execute one brain cycle synchronously. Returns thought if generated.

        Args:
            force: Skip the min_thought_interval_s check. Useful for testing
                where step() calls happen in rapid succession without real
                time passing between them.
        """
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
        interval_ok = force or (now - self._last_thought_time) > self._effective_thought_interval()
        if self.drives.should_think() and interval_ok:
            return self._deliberate()

        return None

    def _effective_thought_interval(self) -> float:
        """Dynamic thought interval — increases with boredom to slow rumination."""
        base = self.min_thought_interval_s
        boredom = self.drives.state.boredom
        if boredom > 0.6:
            # Scale from base to 4× base as boredom goes 0.6→1.0
            scale = 1.0 + 3.0 * ((boredom - 0.6) / 0.4)
            return base * scale
        return base

    # -- Core loop --

    def _loop(self) -> None:
        """Main autonomous loop — runs in background thread.

        Lock protocol: hold _lock only for state reads/writes, never across
        LLM inference (which can block for seconds).  Snapshot state under
        the lock, release it, run inference, then reacquire to commit.
        """
        logger.debug("ThoughtLoop entering main loop")
        while not self._stop_event.is_set():
            try:
                # --- fast tick (under lock) ---
                with self._lock:
                    self.drives.tick()
                    self.stats.ticks += 1
                    now = time.time()

                    should_sleep = (
                        self.drives.should_sleep()
                        and (now - self._last_sleep_time) > self.sleep_cooldown_s
                    )
                    should_think = (
                        not should_sleep
                        and self.drives.should_think()
                        and (now - self._last_thought_time) > self._effective_thought_interval()
                    )

                # --- slow operations (outside lock) ---
                if should_sleep:
                    with self._lock:
                        self.stats.current_mode = "sleeping"
                        self.stats.is_sleeping = True
                    dreams = self._sleep_cycle()
                    with self._lock:
                        self.stats.is_sleeping = False
                        self.stats.current_mode = "idle"
                    if self._on_sleep and dreams:
                        self._on_sleep(dreams)
                elif should_think:
                    with self._lock:
                        self.stats.current_mode = "thinking"
                    result = self._deliberate()
                    with self._lock:
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
        # Modulate temperature based on arousal (only if cortex exposes temperature)
        old_temp = getattr(self.cortex, "temperature", None)
        if old_temp is not None:
            self.cortex.temperature = 0.3 + 0.7 * self.drives.state.arousal

        result = self.cortex.generate(packet)

        if old_temp is not None:
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

        # Cortex→SNN feedback: route thought topics back into SNN systems
        feedback = self.gate.emit_cortex_feedback(result)

        # Boost curiosity routing toward cortex-generated topics
        if self._curiosity_controller is not None:
            for label, amount in feedback.get("topic_boosts", []):
                try:
                    self._curiosity_controller.boost_concept(label, amount=amount)
                except Exception:
                    pass

        # Set cortex-forced topics for next context packet assembly
        if feedback.get("forced_topics"):
            self.gate._cortex_forced_topics = feedback["forced_topics"]

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

        # Append to history (deque.append is CPython-atomic)
        self._thought_history.append({
            "thought": result.thought,
            "confidence": result.confidence,
            "emotional_valence": result.emotional_valence,
            "topics": list(result.topics),
            "latency_ms": round(result.latency_ms, 1),
            "time": self._last_thought_time,
        })

        # Update thought quality metrics
        snn_labels = self.gate._snn_concept_labels if hasattr(self.gate, '_snn_concept_labels') else None
        self.stats.update_quality_metrics(result, snn_concepts=snn_labels)

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

        # Update dream verification rate
        self.stats._dreams_total = self.stats.dreams_generated
        hypotheses = self.memory.recall_hypotheses()
        verified = sum(1 for h in hypotheses if h.provenance.value == "verified")
        self.stats._dreams_verified = verified
        if self.stats._dreams_total > 0:
            self.stats.dream_verification_rate = self.stats._dreams_verified / self.stats._dreams_total

        logger.info("Sleep cycle complete: %d dreams generated", len(dreams))
        return dreams
