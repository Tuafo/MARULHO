"""Drive system and thalamic gate for Subcortex control.

The drive system converts SNN signals (surprise, curiosity, neuromodulators)
into actionable drives that determine when language/readout surfaces wake.
The thalamic gate assembles budgeted context packets from drives + memories.

This is a control interface: the SNN does not need to reason in language, but
it can expose bounded focus targets for language-facing readouts.

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
from hecsn.semantics.exploration_state import ExplorationState

logger = logging.getLogger(__name__)


@dataclass
class DriveState:
    """Current drive intensities — computed from SNN signals."""
    curiosity: float = 0.0       # Quiet startup: no free-running curiosity by default
    anxiety: float = 0.0         # Persistent unresolved surprise
    satisfaction: float = 0.3    # Recent positive outcomes
    boredom: float = 0.0         # Repeated topics / lack of novelty
    social: float = 0.0          # Want to interact / answer questions
    fatigue: float = 0.0         # Accumulated processing → need sleep
    prediction_error: float = 0.0  # Core predictive-processing surprise signal
    uncertainty: float = 0.0       # Quiet startup: no assumed unresolved state
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
    - Grounded observations and explicit operator queries
    """

    def __init__(
        self,
        *,
        substrate_rearm_cooldown_s: float = 8.0,
        substrate_sustain_min_updates: int = 3,
        substrate_sustain_margin: float = 0.22,
        substrate_release_margin: float = 0.08,
    ) -> None:
        self.state = DriveState()
        self.anti_rumination = AntiRuminationCircuit()
        self._thought_count = 0
        self._last_external_input_time = 0.0
        self._last_thought_time = 0.0
        self._startup_quiet = True
        self._pending_grounded_observations = 0
        self._pending_substrate_wakes = 0
        self._last_gate_reason = "startup_quiet"
        self._last_prediction_pressure = 0.0
        self._last_uncertainty_signal = 0.0
        self._substrate_trigger_latched = False
        self._last_substrate_trigger_strength = 0.0
        self._last_substrate_trigger_reason = ""
        self._last_substrate_wake_reason = ""
        self._last_substrate_wake_at = 0.0
        self._substrate_rearm_cooldown_s = max(0.1, float(substrate_rearm_cooldown_s))
        self._substrate_sustain_min_updates = max(2, int(substrate_sustain_min_updates))
        self._substrate_sustain_margin = max(0.02, float(substrate_sustain_margin))
        self._substrate_release_margin = max(0.02, float(substrate_release_margin))
        self._substrate_hysteresis_reason = ""
        self._substrate_hysteresis_updates = 0
        self._substrate_hysteresis_since = 0.0

    @property
    def startup_quiet(self) -> bool:
        return self._startup_quiet

    @property
    def pending_grounded_observations(self) -> int:
        return int(self._pending_grounded_observations)

    @property
    def pending_substrate_wakes(self) -> int:
        return int(self._pending_substrate_wakes)

    @property
    def last_gate_reason(self) -> str:
        return self._last_gate_reason

    @property
    def substrate_hysteresis_active(self) -> bool:
        return bool(self._substrate_hysteresis_reason) and self._substrate_hysteresis_updates > 0

    @property
    def substrate_hysteresis_reason(self) -> str:
        return self._substrate_hysteresis_reason

    @property
    def substrate_hysteresis_updates(self) -> int:
        return int(self._substrate_hysteresis_updates)

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

        # Surprise can modulate readiness, but it is not sufficient on its own
        # to fire the cortex during startup quiet.
        self.state.curiosity = _clamp01(
            0.70 * self.state.curiosity
            + 0.30 * (
                0.35 * self.state.ach_learning
                + 0.35 * self.state.da_novelty
                + 0.15 * self.state.da_reward
                + 0.15 * (1 - self.state.serotonin_patience)
            )
        )
        self.state.anxiety = _clamp01(
            0.70 * self.state.anxiety
            + 0.30 * (
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
        """Use predictive-processing error as the core drive signal."""
        alpha = 0.22
        err_mean = _clamp01(prediction_error_mean)
        err_max = _clamp01(prediction_error_max)
        conf_mean = _clamp01(predictive_confidence_mean)
        conf_min = _clamp01(predictive_confidence_min)
        uncertainty = _clamp01(1.0 - (0.7 * conf_mean + 0.3 * conf_min))
        prediction_pressure = _clamp01(0.6 * err_mean + 0.4 * err_max)
        stability = _clamp01(conf_mean - err_mean)
        self._last_prediction_pressure = prediction_pressure
        self._last_uncertainty_signal = uncertainty

        blended_prediction = alpha * prediction_pressure + (1 - alpha) * self.state.prediction_error
        blended_uncertainty = alpha * uncertainty + (1 - alpha) * self.state.uncertainty
        if self._startup_quiet:
            self.state.prediction_error = max(blended_prediction, 0.65 * prediction_pressure)
            self.state.uncertainty = max(blended_uncertainty, 0.65 * uncertainty)
        else:
            self.state.prediction_error = blended_prediction
            self.state.uncertainty = blended_uncertainty
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

        self._update_substrate_wake_budget(
            prediction_pressure=prediction_pressure,
            uncertainty=uncertainty,
        )
        if prediction_pressure >= 0.32 or uncertainty >= 0.60 or self._substrate_trigger_active():
            self._startup_quiet = False

    def _substrate_trigger_candidates(self, *, prediction_pressure: float, uncertainty: float) -> list[tuple[str, float, float]]:
        return [
            ("prediction_error", max(self.state.prediction_error, prediction_pressure), 0.32),
            ("uncertainty", max(self.state.uncertainty, uncertainty), 0.60),
            ("exploration_urgency", self.state.exploration_urgency, 0.44),
            ("ne_alerting", self.state.ne_alerting, 0.72),
            ("ach_attention", self.state.ach_attention, 0.72),
        ]

    def _reset_substrate_wake_state(self) -> None:
        self._substrate_trigger_latched = False
        self._last_substrate_trigger_strength = 0.0
        self._last_substrate_trigger_reason = ""
        self._last_substrate_wake_reason = ""
        self._substrate_hysteresis_reason = ""
        self._substrate_hysteresis_updates = 0
        self._substrate_hysteresis_since = 0.0

    def _substrate_release_threshold(self, threshold: float) -> float:
        return max(0.0, threshold - self._substrate_release_margin)

    def _substrate_sustain_threshold(self, threshold: float) -> float:
        return min(1.0, threshold + self._substrate_sustain_margin)

    def _current_substrate_candidate(self, reason: str) -> tuple[float, float] | None:
        for candidate_reason, value, threshold in self._substrate_trigger_candidates(
            prediction_pressure=self._last_prediction_pressure,
            uncertainty=self._last_uncertainty_signal,
        ):
            if candidate_reason == reason:
                return value, threshold
        return None

    def _substrate_latch_active(self) -> bool:
        if self._last_substrate_trigger_reason:
            current = self._current_substrate_candidate(self._last_substrate_trigger_reason)
            if current is not None:
                value, threshold = current
                if value >= self._substrate_release_threshold(threshold):
                    return True
        return self._substrate_trigger_active()

    def _update_substrate_hysteresis(
        self,
        *,
        reason: str,
        strength: float,
        threshold: float,
        now: float,
    ) -> bool:
        sustain_threshold = self._substrate_sustain_threshold(threshold)
        if strength >= sustain_threshold:
            if reason == self._substrate_hysteresis_reason:
                self._substrate_hysteresis_updates += 1
            else:
                self._substrate_hysteresis_reason = reason
                self._substrate_hysteresis_updates = 1
                self._substrate_hysteresis_since = now
        else:
            self._substrate_hysteresis_reason = ""
            self._substrate_hysteresis_updates = 0
            self._substrate_hysteresis_since = 0.0
            return False

        return (
            self._substrate_hysteresis_reason == reason
            and self._substrate_hysteresis_updates >= self._substrate_sustain_min_updates
            and (now - self._last_substrate_wake_at) >= self._substrate_rearm_cooldown_s
        )

    def _update_substrate_wake_budget(
        self,
        *,
        prediction_pressure: float,
        uncertainty: float,
    ) -> None:
        now = time.time()
        active = [
            (reason, value, threshold)
            for reason, value, threshold in self._substrate_trigger_candidates(
                prediction_pressure=prediction_pressure,
                uncertainty=uncertainty,
            )
            if value >= threshold
        ]
        if not active:
            self._reset_substrate_wake_state()
            return

        reason, strength, threshold = max(active, key=lambda item: (item[1] - item[2], item[1]))
        previous_reason = self._last_substrate_trigger_reason
        previous_strength = self._last_substrate_trigger_strength
        sustained_ready = self._update_substrate_hysteresis(
            reason=reason,
            strength=strength,
            threshold=threshold,
            now=now,
        )
        wake_reason = f"{reason}_sustained" if sustained_ready else reason
        renewed = (
            not self._substrate_trigger_latched
            or strength >= (previous_strength + 0.08)
            or (
                reason != previous_reason
                and strength >= threshold + 0.03
            )
            or sustained_ready
        )
        self._substrate_trigger_latched = True
        self._last_substrate_trigger_strength = strength
        self._last_substrate_trigger_reason = reason
        if renewed:
            self._pending_substrate_wakes = min(4, self._pending_substrate_wakes + 1)
            self._last_substrate_wake_at = now
            self._last_substrate_wake_reason = wake_reason
            self._last_gate_reason = wake_reason

    def update_from_thought(self, result: ThoughtResult) -> None:
        """Update drives after a thought is generated."""
        self._thought_count += 1
        self._last_thought_time = time.time()

        # Record topics for anti-rumination
        self.anti_rumination.record_topics(result.topics)
        self.state.boredom = self.anti_rumination.boredom_signal()

        # Fatigue accumulates with thoughts, decays with time
        self.state.fatigue = min(1.0, self.state.fatigue + 0.04)

    def update_from_grounded_observation(self) -> None:
        """Grounded evidence arrived — queue a wake event for the cortex."""
        self._last_external_input_time = time.time()
        self._startup_quiet = False
        self._pending_grounded_observations = min(8, self._pending_grounded_observations + 1)
        self.state.curiosity = min(1.0, max(self.state.curiosity, 0.18))
        self.state.social = min(1.0, max(self.state.social, 0.12))
        self.state.boredom = max(0.0, self.state.boredom - 0.35)

    def has_pending_grounded_observation(self) -> bool:
        return self._pending_grounded_observations > 0

    def consume_grounded_observation(self) -> bool:
        if self._pending_grounded_observations <= 0:
            return False
        self._pending_grounded_observations -= 1
        return True

    def has_pending_substrate_wake(self) -> bool:
        return self._pending_substrate_wakes > 0

    def consume_substrate_wake(self) -> bool:
        if self._pending_substrate_wakes <= 0:
            return False
        self._pending_substrate_wakes -= 1
        return True

    def update_from_external_query(self) -> None:
        """External query arrived — wake the answer pathway decisively."""
        self._last_external_input_time = time.time()
        self._startup_quiet = False
        self.state.social = min(1.0, max(self.state.social, 0.55))
        self.state.curiosity = min(1.0, max(self.state.curiosity, 0.30))
        self.state.boredom = max(0.0, self.state.boredom - 0.4)

    def update_from_unresolved_tension(self) -> None:
        """A still-unresolved contradiction or tension requires attention."""
        self._startup_quiet = False
        self.state.anxiety = min(1.0, max(self.state.anxiety, 0.22))
        self.state.ne_alerting = min(1.0, max(self.state.ne_alerting, 0.24))

    def tick(self) -> None:
        """Periodic drive decay/update (call on SNN fast loop)."""
        self.state.fatigue = max(0.0, self.state.fatigue - 0.0005)
        self.state.social = max(0.0, self.state.social * 0.995)
        self.state.boredom = max(0.0, self.state.boredom * 0.990)
        self._last_prediction_pressure = max(0.0, self._last_prediction_pressure * 0.96)
        self._last_uncertainty_signal = max(0.0, self._last_uncertainty_signal * 0.98)
        if not self._substrate_latch_active():
            self._reset_substrate_wake_state()

    def _inhibition_reason(self) -> str:
        if self.state.fatigue > 0.9:
            return "fatigue_inhibit"
        if self.state.boredom > 0.8:
            return "boredom_inhibit"
        return ""

    def can_think(self) -> bool:
        reason = self._inhibition_reason()
        if reason:
            self._last_gate_reason = reason
            return False
        return True

    def should_answer_now(self, *, query_pending: bool) -> bool:
        if not query_pending:
            return False
        self._last_gate_reason = "query_pending"
        return True

    def _substrate_trigger_active(self) -> bool:
        return (
            max(self.state.prediction_error, self._last_prediction_pressure) >= 0.32
            or max(self.state.uncertainty, self._last_uncertainty_signal) >= 0.60
            or self.state.exploration_urgency >= 0.44
            or self.state.ne_alerting >= 0.72
            or self.state.ach_attention >= 0.72
        )

    def should_think(
        self,
        *,
        grounded_observation_pending: bool = False,
        has_tension: bool = False,
    ) -> bool:
        """Should the cortex fire a non-query deliberation cycle?

        True SNN gating means spontaneous thought requires a concrete wake
        trigger from the substrate or a freshly grounded observation. Curiosity,
        anxiety, and social tone can modulate later behavior, but they do not
        by themselves justify cortex firing anymore.
        """
        if not self.can_think():
            return False

        substrate_trigger = self.has_pending_substrate_wake()
        if self._startup_quiet and not (grounded_observation_pending or has_tension or substrate_trigger):
            self._last_gate_reason = "startup_quiet"
            return False
        if grounded_observation_pending:
            self._last_gate_reason = "grounded_observation_pending"
            return True
        if has_tension:
            self._last_gate_reason = "working_memory_tension"
            return True
        if substrate_trigger:
            self._last_gate_reason = (
                self._last_substrate_wake_reason
                or self._last_substrate_trigger_reason
                or "prediction_error"
            )
            return True
        self._last_gate_reason = "idle_no_trigger"
        return False

    def should_sleep(self) -> bool:
        """Should we enter sleep/consolidation mode?"""
        return self.state.fatigue > 0.5 and self.state.social < 0.2

    def choose_mode(self) -> ThinkingMode:
        """Choose thinking mode for non-query deliberation."""
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
        self.working_memory: Any = None  # Attached by the active deliberation surface.
        self.narrative_self: Any = None  # Attached by the active deliberation surface.
        self.active_exploration_state = ExplorationState()
        from collections import deque
        self._query_queue: deque[str] = deque(maxlen=max_query_queue)

    @property
    def active_exploration_target(self) -> str:
        return self.active_exploration_state.target

    @property
    def active_exploration_reason(self) -> str:
        return self.active_exploration_state.reason

    @property
    def active_exploration_source(self) -> str:
        return self.active_exploration_state.source

    @property
    def active_exploration_score(self) -> float:
        return self.active_exploration_state.score

    @property
    def active_exploration_updated_at(self) -> float:
        return self.active_exploration_state.updated_at

    def has_pending_query(self) -> bool:
        return bool(self._query_queue)

    def pop_query(self) -> str:
        return self._query_queue.popleft() if self._query_queue else ""

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
        state = ExplorationState.from_target(
            topic,
            reason=reason,
            source=source,
            score=score,
        )
        if not state.target:
            self.clear_active_exploration_target()
            return
        self.active_exploration_state = state

    def clear_active_exploration_target(self) -> None:
        """Clear the active exploration target after it has been consumed."""
        self.active_exploration_state = ExplorationState()

    @staticmethod
    def _memory_item_from_episode(ep: Any, *, text: str | None = None) -> MemoryItem:
        return MemoryItem(
            text=(text if text is not None else ep.content),
            salience=ep.salience,
            age_seconds=ep.age_seconds,
            source=ep.provenance.value if ep.provenance.value in (
                "observed", "inferred", "dreamed", "verified", "external"
            ) else "observed",
            memory_id=ep.episode_id,
        )

    def assemble(self, phase: str = "", *, external_query: str = "") -> ContextPacket:
        """Build a context packet from current SNN state.

        Args:
            phase: Deliberation phase ("observe", "question", "reason",
                   "synthesize"). Empty string = standard single-shot.
            external_query: Optional explicit query to inject into the packet.

        Uses a dedicated evidence-bundle retrieval path for both external
        queries and wakeful deliberation. Recent grounded evidence is
        preferred over generic prior context, while wakeful thoughts still
        receive a rotating seed topic when no grounded focus is available.
        """
        import random

        drive_state = self.drives.state
        queued_query = self.pop_query() if (self._query_queue and not phase and not external_query) else ""
        final_query = external_query or queued_query
        mode = ThinkingMode.ANSWER if final_query else self.drives.choose_mode()

        # Determine topics to avoid
        avoid_topics = self.drives.anti_rumination.suggest_topic_avoidance()
        avoid_words = {w.lower() for w in avoid_topics}

        exploration_target = ""
        if not phase and not final_query and self.active_exploration_target:
            target_words = {w.lower() for w in self.active_exploration_target.split() if len(w) >= 3}
            if not (target_words & avoid_words):
                exploration_target = self.active_exploration_target

        grounded_evidence_items: list[MemoryItem] = []
        grounded_focus = ""

        # Both query answering and spontaneous wakeful thinking now use
        # dedicated evidence-bundle retrieval instead of the older generic
        # similarity/diverse hot path. Queries retrieve against the operator
        # question; non-query deliberation retrieves against the active
        # exploration target or the freshest grounded evidence focus.
        if final_query:
            evidence_bundle = self.memory.recall_for_query(
                final_query,
                grounded_top_k=min(4, self.max_memories),
                support_top_k=self.max_memories,
            )
        else:
            evidence_bundle = self.memory.recall_for_deliberation(
                exploration_target,
                grounded_top_k=min(3, self.max_memories),
                support_top_k=self.max_memories,
                avoid_topics=avoid_topics,
            )
            if not exploration_target:
                grounded_focus = evidence_bundle.target

        mem_items = [
            self._memory_item_from_episode(match.episode, text=match.focused_text)
            for match in evidence_bundle.support
        ]
        grounded_evidence_items = [
            self._memory_item_from_episode(match.episode, text=match.focused_text)
            for match in evidence_bundle.grounded
        ]

        self_state = ""

        max_tokens = 160
        if mode == ThinkingMode.DREAM:
            max_tokens = 224
        elif final_query:
            max_tokens = 220

        forced_topic = ""
        if final_query:
            forced_topic = ""
        elif exploration_target:
            forced_topic = exploration_target
            self.clear_active_exploration_target()
        elif grounded_focus:
            forced_topic = grounded_focus
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
                self._used_seeds: set[int] = set()
            for _ in range(len(_SEED_DOMAINS)):
                idx = self._seed_idx % len(_SEED_DOMAINS)
                self._seed_idx += 1
                if idx in self._used_seeds:
                    continue
                candidate = _SEED_DOMAINS[idx]
                candidate_words = {w.lower() for w in candidate.split() if len(w) >= 3}
                if not (candidate_words & avoid_words):
                    forced_topic = candidate
                    self._used_seeds.add(idx)
                    if len(self._used_seeds) >= len(_SEED_DOMAINS):
                        self._used_seeds.clear()
                    break
            else:
                self._used_seeds.clear()
                forced_topic = _SEED_DOMAINS[self._seed_idx % len(_SEED_DOMAINS)]
                self._seed_idx += 1

        drive_summary = drive_state.to_summary() if drive_state.anxiety > 0.5 else ""

        narrative = ""
        if self.narrative_self is not None and (
            final_query
            or mode in (ThinkingMode.ANSWER, ThinkingMode.REFLECT, ThinkingMode.DREAM)
        ):
            narrative = self.narrative_self.to_prompt()

        wm_narrative = ""
        if self.working_memory is not None and phase in ("question", "reason", "synthesize"):
            wm_narrative = self.working_memory.broadcast()

        if phase in ("question", "reason", "synthesize"):
            mem_items = []
            forced_topic = ""

        packet = ContextPacket(
            drive_summary=drive_summary,
            top_memories=mem_items,
            grounded_evidence=grounded_evidence_items,
            self_state=self_state,
            mode=mode,
            external_query=final_query,
            avoid_topics=sorted(avoid_topics)[:6],
            forced_topic=forced_topic,
            narrative_self=narrative,
            working_memory_narrative=wm_narrative,
            deliberation_phase=phase,
            max_response_tokens=max_tokens,
        )
        return packet

    def emit_deliberation_feedback(self, result: ThoughtResult) -> dict[str, Any]:
        """Build feedback from a language-facing result for Subcortex control.

        Called by a deliberation surface after each supported inference.
        Routes topics back into SNN curiosity routing, cross-modal grounding,
        and context assembly so language-facing outputs stay evidence-coupled.
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
        self.drives.update_from_external_query()

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
