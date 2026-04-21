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
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from hecsn.cortex.core import ContextPacket, MemoryItem, ThinkingMode, ThoughtResult
from hecsn.cortex.episodic_memory import EpisodicMemory

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
    prediction_error: float = 0.0  # Core predictive-processing surprise signal
    uncertainty: float = 0.5       # Low-confidence / unresolved state
    exploration_urgency: float = 0.0  # How strongly the system wants targeted exploration

    # Neuromodulator mirrors (from SurpriseMonitor)
    dopamine: float = 0.5
    serotonin: float = 0.5
    norepinephrine: float = 0.5
    acetylcholine: float = 0.5

    # Richer channelized neuromodulation (derived from the scalar monitors +
    # predictive-processing state). These are not separate biology-faithful
    # nuclei models yet, but they let the cortex-side controller react to
    # reward/novelty/salience/attention in more brain-like ways.
    da_reward: float = 0.5
    da_novelty: float = 0.5
    da_salience: float = 0.5
    ne_alerting: float = 0.5
    ne_orienting: float = 0.5
    ach_learning: float = 0.5
    ach_attention: float = 0.5
    serotonin_patience: float = 0.5

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


_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "it",
    "on", "at", "by", "with", "from", "as", "that", "this", "its", "my",
    "i", "vs", "vs.", "not", "no", "but", "so", "how", "what", "when",
    "where", "why", "which", "can", "do", "does", "did", "was", "are",
    "be", "been", "has", "had", "have", "will", "would", "could", "should",
    "about", "into", "between", "through", "over", "under", "more", "most",
    "very", "just", "also", "than", "then", "these", "those", "some",
    "all", "each", "every", "both", "few", "many", "much", "other",
    "such", "only", "own", "same", "new", "like", "after", "before",
})


class AntiRuminationCircuit:
    """Prevents degenerate thought loops via boredom and diversity tracking.

    Uses word-level counting so "pottery", "Neolithic pottery", and
    "pottery material science" all count toward the same word-stems,
    catching semantic clusters that phrase-level tracking misses.
    """

    def __init__(
        self,
        topic_decay_rate: float = 0.75,  # Faster decay so topics clear quicker
        boredom_threshold: int = 2,
        diversity_window: int = 15,  # Wider window catches more repetition
    ) -> None:
        self.topic_decay_rate = topic_decay_rate
        self.boredom_threshold = boredom_threshold
        self.diversity_window = diversity_window
        self._word_counts: dict[str, float] = defaultdict(float)
        self._phrase_counts: dict[str, float] = defaultdict(float)
        self._recent_topics: list[str] = []

    @staticmethod
    def _extract_words(topic: str) -> list[str]:
        """Extract meaningful words from a topic phrase."""
        words = []
        for w in topic.lower().split():
            w = w.strip(".,;:!?'\"()-")
            if len(w) >= 3 and w not in _STOP_WORDS:
                words.append(w)
        return words

    def record_topics(self, topics: Sequence[str]) -> None:
        """Record topics from a thought result.

        Tracks both phrase-level and word-level counts so that semantic
        clusters are detected even when the LLM uses different phrasings.
        """
        # Decay all existing counts
        for k in list(self._word_counts.keys()):
            self._word_counts[k] *= self.topic_decay_rate
            if self._word_counts[k] < 0.01:
                del self._word_counts[k]
        for k in list(self._phrase_counts.keys()):
            self._phrase_counts[k] *= self.topic_decay_rate
            if self._phrase_counts[k] < 0.01:
                del self._phrase_counts[k]

        for t in topics:
            key = t.lower().strip()
            if key:
                self._phrase_counts[key] += 1.0
                self._recent_topics.append(key)
                for word in self._extract_words(t):
                    self._word_counts[word] += 1.0

        self._recent_topics = self._recent_topics[-self.diversity_window:]

    def boredom_signal(self) -> float:
        """How bored are we? Based on topic repetition.

        Returns 0→1 where 1 means extreme rumination.
        Uses word-level counts which detect semantic clusters.
        """
        if not self._word_counts:
            return 0.0
        max_count = max(self._word_counts.values())
        if max_count >= self.boredom_threshold:
            excess = max_count - self.boredom_threshold
            return min(1.0, 0.4 + excess * 0.15)
        return 0.0

    def diversity_score(self) -> float:
        """How diverse are recent thoughts? 0 = all same, 1 = all different."""
        if len(self._recent_topics) < 2:
            return 1.0
        unique = len(set(self._recent_topics))
        return unique / len(self._recent_topics)

    def suggest_topic_avoidance(self) -> set[str]:
        """Words to avoid (currently over-represented).

        Returns individual words, not full phrases — used by the thalamic
        gate to filter memory recall and by the prompt to redirect the LLM.
        """
        threshold = self.boredom_threshold * 0.7
        return {w for w, c in self._word_counts.items() if c >= threshold}


class DriveSystem:
    """Converts SNN signals into drives that control the cortex.

    Updates drives each SNN tick based on:
    - Surprise monitor neuromodulators
    - Predictive-processing error / uncertainty
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

        # Richer channelized neuromodulation derived from the scalar monitors.
        beta = 0.22
        self.state.da_reward = _clamp01((1 - beta) * self.state.da_reward + beta * self.state.dopamine)
        self.state.da_novelty = _clamp01(
            (1 - beta) * self.state.da_novelty
            + beta * (0.6 * self.state.acetylcholine + 0.4 * self.state.dopamine)
        )
        self.state.da_salience = _clamp01(
            (1 - beta) * self.state.da_salience
            + beta * max(self.state.dopamine, self.state.norepinephrine)
        )
        self.state.ne_alerting = _clamp01(
            (1 - beta) * self.state.ne_alerting + beta * self.state.norepinephrine
        )
        self.state.ne_orienting = _clamp01(
            (1 - beta) * self.state.ne_orienting
            + beta * (0.5 * self.state.norepinephrine + 0.5 * self.state.acetylcholine)
        )
        self.state.ach_learning = _clamp01(
            (1 - beta) * self.state.ach_learning + beta * self.state.acetylcholine
        )
        self.state.ach_attention = _clamp01(
            (1 - beta) * self.state.ach_attention
            + beta * (0.6 * self.state.acetylcholine + 0.4 * self.state.norepinephrine)
        )
        self.state.serotonin_patience = _clamp01(
            (1 - beta) * self.state.serotonin_patience + beta * self.state.serotonin
        )

        # Neuromodulators are secondary modulators now; predictive error is the
        # primary signal and further shapes these drives in
        # update_from_prediction_error().
        self.state.curiosity = _clamp01(
            0.65 * self.state.curiosity
            + 0.35 * (
                0.35 * self.state.ach_learning
                + 0.35 * self.state.da_novelty
                + 0.15 * self.state.da_reward
                + 0.15 * (1 - self.state.serotonin_patience)
            )
        )
        self.state.anxiety = _clamp01(
            0.65 * self.state.anxiety
            + 0.35 * (
                0.45 * self.state.ne_alerting
                + 0.25 * self.state.da_salience
                + 0.20 * self.state.serotonin
                - 0.10 * self.state.da_reward
            )
        )
        self.state.satisfaction = _clamp01(
            0.7 * self.state.satisfaction
            + 0.3 * (0.7 * self.state.da_reward + 0.3 * self.state.serotonin_patience)
        )

    def update_from_prediction_error(
        self,
        prediction_error_mean: float,
        prediction_error_max: float,
        predictive_confidence_mean: float,
        predictive_confidence_min: float,
    ) -> None:
        """Use predictive-processing error as the core drive signal.

        High prediction error means the current internal model is failing and
        should drive curiosity/anxiety. Low error with high confidence means
        the system is overfamiliar with its current regime and boredom should
        rise. This is a lightweight free-energy-inspired control rule.
        """
        alpha = 0.22
        err_mean = _clamp01(prediction_error_mean)
        err_max = _clamp01(prediction_error_max)
        conf_mean = _clamp01(predictive_confidence_mean)
        conf_min = _clamp01(predictive_confidence_min)
        uncertainty = _clamp01(1.0 - (0.7 * conf_mean + 0.3 * conf_min))
        prediction_pressure = _clamp01(0.6 * err_mean + 0.4 * err_max)
        stability = _clamp01(conf_mean - err_mean)

        self.state.prediction_error = alpha * prediction_pressure + (1 - alpha) * self.state.prediction_error
        self.state.uncertainty = alpha * uncertainty + (1 - alpha) * self.state.uncertainty
        self.state.exploration_urgency = _clamp01(
            0.6 * self.state.prediction_error + 0.4 * self.state.uncertainty
        )

        # Channelized neuromodulation targets.
        self.state.da_novelty = _clamp01(
            (1 - alpha) * self.state.da_novelty
            + alpha * (0.65 * prediction_pressure + 0.35 * uncertainty)
        )
        self.state.da_salience = _clamp01(
            (1 - alpha) * self.state.da_salience
            + alpha * max(err_max, 0.7 * prediction_pressure + 0.3 * uncertainty)
        )
        self.state.ne_alerting = _clamp01(
            (1 - alpha) * self.state.ne_alerting
            + alpha * max(err_max, 0.6 * prediction_pressure + 0.4 * self.state.norepinephrine)
        )
        self.state.ne_orienting = _clamp01(
            (1 - alpha) * self.state.ne_orienting
            + alpha * (0.55 * self.state.exploration_urgency + 0.45 * self.state.ach_attention)
        )
        self.state.ach_learning = _clamp01(
            (1 - alpha) * self.state.ach_learning
            + alpha * (0.55 * uncertainty + 0.45 * self.state.acetylcholine)
        )
        self.state.ach_attention = _clamp01(
            (1 - alpha) * self.state.ach_attention
            + alpha * (0.55 * prediction_pressure + 0.45 * self.state.exploration_urgency)
        )
        self.state.serotonin_patience = _clamp01(
            (1 - alpha) * self.state.serotonin_patience
            + alpha * max(0.0, 0.65 * stability + 0.35 * self.state.serotonin)
        )

        curiosity_target = _clamp01(
            0.40 * self.state.da_novelty
            + 0.25 * self.state.ach_learning
            + 0.20 * prediction_pressure
            + 0.15 * uncertainty
        )
        anxiety_target = _clamp01(
            0.45 * self.state.ne_alerting
            + 0.25 * self.state.da_salience
            + 0.20 * err_max
            + 0.10 * uncertainty
        )
        boredom_target = _clamp01(max(0.0, 0.85 * stability - 0.25 * self.state.da_novelty))
        satisfaction_target = _clamp01(
            max(0.0, 0.55 * stability + 0.25 * self.state.da_reward + 0.20 * self.state.serotonin_patience)
        )

        self.state.curiosity = _clamp01((1 - alpha) * self.state.curiosity + alpha * curiosity_target)
        self.state.anxiety = _clamp01((1 - alpha) * self.state.anxiety + alpha * anxiety_target)
        self.state.boredom = _clamp01(0.7 * self.state.boredom + 0.3 * boredom_target)
        self.state.satisfaction = _clamp01((1 - alpha) * self.state.satisfaction + alpha * satisfaction_target)

    def update_from_thought(self, result: ThoughtResult) -> None:
        """Update drives after a thought is generated."""
        self._thought_count += 1
        self._last_thought_time = time.time()

        # Record topics for anti-rumination
        self.anti_rumination.record_topics(result.topics)
        self.state.boredom = self.anti_rumination.boredom_signal()

        # Fatigue accumulates with thoughts, decays with time
        # Increased from 0.02 → 0.04 so sleep triggers sooner (25 vs 50 thoughts)
        self.state.fatigue = min(1.0, self.state.fatigue + 0.04)

    def update_from_external_input(self) -> None:
        """External input arrived — reduce boredom, increase social drive."""
        self._last_external_input_time = time.time()
        self.state.social = min(1.0, self.state.social + 0.3)
        self.state.boredom = max(0.0, self.state.boredom - 0.3)

    def tick(self) -> None:
        """Periodic drive decay/update (call on SNN fast loop)."""
        # Fatigue decays very slowly — reduced from 0.001 to 0.0005
        # so sleep actually triggers during sustained operation
        self.state.fatigue = max(0.0, self.state.fatigue - 0.0005)
        # Social drive decays without input
        self.state.social = max(0.0, self.state.social * 0.995)
        # Boredom decays moderately so the system recovers from rumination
        # pauses (half-life ~70 ticks at 100ms tick = ~7s)
        self.state.boredom = max(0.0, self.state.boredom * 0.990)

    def should_think(self) -> bool:
        """Should the cortex fire a deliberation cycle?

        Boredom above 0.8 forces a cooldown — the system must wait for
        external input or drive decay before thinking again, preventing
        runaway rumination loops.
        """
        if self.state.fatigue > 0.9:
            return False  # Too tired, need sleep
        if self.state.boredom > 0.8:
            return False  # Ruminating — wait for novelty
        # Prediction error is now a first-class trigger for cognition.
        return (
            self.state.curiosity > 0.4
            or self.state.anxiety > 0.5
            or self.state.social > 0.3
            or self.state.prediction_error > 0.35
            or self.state.uncertainty > 0.55
            or self.state.exploration_urgency > 0.45
            or self.state.ne_alerting > 0.7
            or self.state.ach_attention > 0.7
        )

    def should_sleep(self) -> bool:
        """Should we enter sleep/consolidation mode?

        Lowered threshold from 0.7 → 0.5 so dream cycles actually trigger
        during sustained operation. Fatigue at 0.02/thought needs ~25 thoughts
        to reach 0.5, which is realistic for a ~4-minute active session.
        """
        return self.state.fatigue > 0.5 and self.state.social < 0.2

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
        max_memories: int = 8,
        max_query_queue: int = 8,
    ) -> None:
        self.memory = memory
        self.drives = drives
        self.max_memories = max_memories
        self._snn_concept_labels: list[str] = []
        self.working_memory: Any = None  # Set by ThoughtLoop
        self.narrative_self: Any = None  # Set by ThoughtLoop
        self.active_exploration_target: str = ""
        self.active_exploration_reason: str = ""
        self.active_exploration_source: str = ""
        self.active_exploration_score: float = 0.0
        self.active_exploration_updated_at: float = 0.0
        from collections import deque
        self._query_queue: deque[str] = deque(maxlen=max_query_queue)

    def update_snn_concepts(self, labels: list[str]) -> None:
        """Receive recent SNN concept labels for alignment/quality tracking."""
        self._snn_concept_labels = labels[-20:]  # keep recent 20

    def set_active_exploration_target(
        self,
        topic: str,
        *,
        reason: str = "",
        source: str = "prediction_error",
        score: float = 0.0,
    ) -> None:
        """Set the next wakeful exploration target chosen by the brain."""
        cleaned = " ".join(str(topic).replace("/", " ").replace("|", " ").split()).strip()
        if not cleaned:
            self.clear_active_exploration_target()
            return
        self.active_exploration_target = cleaned[:120]
        self.active_exploration_reason = " ".join(str(reason).split()).strip()[:160]
        self.active_exploration_source = str(source).strip()[:40]
        self.active_exploration_score = max(0.0, min(1.0, float(score)))
        self.active_exploration_updated_at = time.time()

    def clear_active_exploration_target(self) -> None:
        """Clear the active exploration target after it has been consumed."""
        self.active_exploration_target = ""
        self.active_exploration_reason = ""
        self.active_exploration_source = ""
        self.active_exploration_score = 0.0
        self.active_exploration_updated_at = 0.0

    def assemble(self, phase: str = "") -> ContextPacket:
        """Build a context packet from current SNN state.

        Args:
            phase: Deliberation phase ("observe", "question", "reason",
                   "synthesize"). Empty string = standard single-shot.

        Uses diverse memory recall when boredom is high — avoids the
        echo-chamber effect where the LLM only sees its own rumination.
        Wakeful thoughts also receive a rotating seed topic so the cortex
        keeps exploring new domains instead of replaying its own outputs.
        """
        import random

        drive_state = self.drives.state
        mode = self.drives.choose_mode()
        external_query = self._query_queue.popleft() if (self._query_queue and not phase) else ""

        # Determine topics to avoid
        avoid_topics = self.drives.anti_rumination.suggest_topic_avoidance()
        avoid_words = {w.lower() for w in avoid_topics}

        exploration_target = ""
        if not phase and not external_query and self.active_exploration_target:
            target_words = {w.lower() for w in self.active_exploration_target.split() if len(w) >= 3}
            if not (target_words & avoid_words):
                exploration_target = self.active_exploration_target

        # Select memories: active exploration targets get first-class retrieval,
        # otherwise use diverse recall when bored or similarity recall when stable.
        if exploration_target:
            memories = self.memory.recall_by_similarity(exploration_target, top_k=self.max_memories)
        elif drive_state.boredom > 0.4 or len(avoid_topics) > 0:
            memories = self.memory.recall_diverse(
                top_k=self.max_memories,
                avoid_topics=avoid_topics,
            )
        else:
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

        # Self state — stripped when bored or curious to prevent ANY self-reference
        # The LLM will invent drive states from even minimal hints; give it nothing
        # Self-state is always empty — prevents the LLM from fixating on
        # drive names ("curiosity", "boredom") and producing meta-thoughts
        self_state = ""

        # Temperature modulation based on arousal
        # Reduced from 256→160 tokens for faster inference while leaving room for JSON
        max_tokens = 160
        if mode == ThinkingMode.DREAM:
            max_tokens = 224

        # Choose wakeful direction: first honor an active exploration target,
        # otherwise rotate through broad seed domains to keep the system moving.
        forced_topic = ""
        if exploration_target:
            forced_topic = exploration_target
            self.clear_active_exploration_target()
        else:
            _SEED_DOMAINS = [
                "coral reef ecosystems", "volcanic eruptions", "jazz improvisation",
                "bridge engineering", "deep sea creatures", "ancient Roman roads",
                "bird migration patterns", "glacial formation", "fermentation in cooking",
                "optical illusions", "earthquake prediction", "silk production",
                "tidal forces", "cave formations", "wind turbine design",
                "animal camouflage", "constellation navigation", "paper manufacturing",
                "lightning physics", "seed dispersal mechanisms", "acoustic resonance",
                "permafrost thawing", "compass magnetism", "origami mathematics",
                "bioluminescence", "plate tectonics", "honey bee communication",
                "superconductivity", "cloud formation", "spider silk properties",
                "river delta formation", "radio telescope design", "whale migration",
                "aurora borealis", "mushroom networks", "ocean thermal vents",
            ]
            if not hasattr(self, '_seed_idx'):
                self._seed_idx = random.randint(0, len(_SEED_DOMAINS) - 1)
                self._used_seeds: set[int] = set()  # Track used indices
            # Filter out seeds that overlap with avoidance words OR already used
            for _ in range(len(_SEED_DOMAINS)):
                idx = self._seed_idx % len(_SEED_DOMAINS)
                self._seed_idx += 1
                if idx in self._used_seeds:
                    continue  # Already used this seed — skip
                candidate = _SEED_DOMAINS[idx]
                candidate_words = {w.lower() for w in candidate.split() if len(w) >= 3}
                if not (candidate_words & avoid_words):
                    forced_topic = candidate
                    self._used_seeds.add(idx)
                    # Reset used seeds when all have been used (full cycle)
                    if len(self._used_seeds) >= len(_SEED_DOMAINS):
                        self._used_seeds.clear()
                    break
            else:
                # All seeds avoided — reset and pick any
                self._used_seeds.clear()
                forced_topic = _SEED_DOMAINS[self._seed_idx % len(_SEED_DOMAINS)]
                self._seed_idx += 1

        # Strip drive details almost always — prevent self-referential loops.
        # The LLM fixates on drive names ("curiosity", "anxiety") and produces
        # meta-thoughts about those drives instead of thinking about the world.
        # Only show drive state when anxiety is high (something needs attention).
        drive_summary = drive_state.to_summary() if drive_state.anxiety > 0.5 else ""

        # Narrative self: persistent identity/context, useful for answering,
        # reflection, and dreaming. Keep it out of ordinary wakeful chains so it
        # does not dominate fresh topic exploration; working memory already carries
        # within-chain continuity.
        narrative = ""
        if self.narrative_self is not None and (
            external_query
            or mode in (ThinkingMode.ANSWER, ThinkingMode.REFLECT, ThinkingMode.DREAM)
        ):
            narrative = self.narrative_self.to_prompt()

        # Working memory narrative (global workspace broadcast)
        # Only include for chain continuation phases (question/reason/synthesize)
        # where continuity matters. For observe/quick (no phase), working memory
        # would override the new Direction seed — causing topic repetition.
        wm_narrative = ""
        if self.working_memory is not None and phase in ("question", "reason", "synthesize"):
            wm_narrative = self.working_memory.broadcast()

        # For chain phases after observe, skip memories (working memory has context)
        if phase in ("question", "reason", "synthesize"):
            mem_items = []  # Working memory IS the context
            forced_topic = ""  # Don't inject new topic mid-chain

        packet = ContextPacket(
            drive_summary=drive_summary,
            top_memories=mem_items,
            self_state=self_state,
            mode=mode,
            external_query=external_query,
            avoid_topics=sorted(avoid_topics)[:6],
            forced_topic=forced_topic,
            narrative_self=narrative,
            working_memory_narrative=wm_narrative,
            deliberation_phase=phase,
            max_response_tokens=max_tokens,
        )
        return packet

    def emit_cortex_feedback(self, result: ThoughtResult) -> dict[str, Any]:
        """Build a feedback payload from ThoughtResult for SNN consumption.

        Called by ThoughtLoop._deliberate() after each cortex inference.
        Routes thought topics back into SNN curiosity routing, cross-modal
        grounding, and context assembly — closing the cortex→SNN loop.
        """
        boosts: list[tuple[str, float]] = []
        grounding_candidates: list[str] = []

        # Uncertainty-scaled boosts: low confidence = bigger curiosity gap
        confidence_scale = float(result.confidence)
        uncertainty = 1.0 - confidence_scale

        for topic in result.topics:
            if not topic or len(topic) < 2:
                continue
            boost_amount = 0.05 + 0.10 * uncertainty  # range [0.05, 0.15]
            boosts.append((topic.lower().strip(), boost_amount))
            # Extract candidate words for cross-modal grounding
            words = [
                w.strip(".,;:!?'\"()-")
                for w in topic.split()
                if len(w) >= 3
            ]
            grounding_candidates.extend(words)

        # Emotional valence feedback into drives
        valence = float(result.emotional_valence)
        if valence > 0.2:
            self.drives.state.satisfaction = min(1.0, self.drives.state.satisfaction + 0.05)
        elif valence < -0.2:
            self.drives.state.anxiety = min(1.0, self.drives.state.anxiety + 0.05 * abs(valence))

        return {
            "topic_boosts": boosts,
            "grounding_candidates": grounding_candidates[:6],
            "emotional_valence": valence,
            "confidence": float(result.confidence),
        }

    def submit_query(self, query: str) -> None:
        """Submit an external query for the cortex to answer.

        Uses a bounded queue so multiple queries can be pending.
        """
        self._query_queue.append(query)
        self.drives.update_from_external_input()

    def assemble_for_sleep(
        self,
        episodes: Sequence[Any] | None = None,
        *,
        phase: str = "",
        hypothesis: str = "",
    ) -> ContextPacket:
        """Build a dream-mode context packet for sleep consolidation.

        Args:
            episodes: Optional subset of episodes to use for this dream step.
                If omitted, the gate selects sleep-replay episodes itself.
            phase: Dream sub-phase ("dream_compose" or "dream_test").
            hypothesis: Candidate dream hypothesis to validate during dream_test.
        """
        selected = list(episodes) if episodes is not None else self.memory.recall_for_sleep(top_k=self.max_memories)
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
            for ep in selected[:self.max_memories]
        ]

        drive_summary = "Sleep consolidation — find connections between memories"
        max_tokens = 224
        wm_narrative = ""
        external_query = ""
        self_state = "Sleeping, dreaming"
        narrative = self.narrative_self.to_prompt() if self.narrative_self is not None else ""
        mode = ThinkingMode.DREAM
        if phase == "dream_compose":
            drive_summary = "Connect these memories into one testable hypothesis"
            external_query = (
                "Connect the provided memories into one concrete, testable hypothesis. "
                "Focus on the memory content itself."
            )
            self_state = ""
            narrative = ""
            mode = ThinkingMode.THINK
            max_tokens = 192
        elif phase == "dream_test":
            drive_summary = "Evaluate whether the candidate hypothesis fits the memories"
            wm_narrative = f"Candidate hypothesis: {hypothesis}" if hypothesis else ""
            external_query = (
                "Evaluate whether the candidate hypothesis is supported by the memories. "
                "If unsupported, explain the contradiction or missing mechanism clearly."
            )
            self_state = ""
            narrative = ""
            mode = ThinkingMode.THINK
            max_tokens = 160

        return ContextPacket(
            drive_summary=drive_summary,
            top_memories=mem_items,
            self_state=self_state,
            narrative_self=narrative,
            working_memory_narrative=wm_narrative,
            mode=mode,
            external_query=external_query,
            deliberation_phase=phase,
            max_response_tokens=max_tokens,
        )


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
