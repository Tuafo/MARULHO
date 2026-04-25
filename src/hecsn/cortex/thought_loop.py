"""ThoughtLoop — the multi-clock autonomous brain of Terminus.

Implements the continuous thinking cycle:
  Fast SNN tick (10ms)  → drive updates, surprise, salience
  Deliberation (event)  → LLM inference triggered by spike threshold
  Sleep (periodic)      → replay, compression, dream/hypothesis generation

Deliberation depth (System 1 / System 2):
  QUICK (1 call)    → Fast pattern matching, gut reaction
  STANDARD (2 calls) → Observe + question
  DEEP (4 calls)    → Observe → question → reason → synthesize

The SNN decides depth based on prediction error, working memory
tensions, and drive state.

The loop runs in a background thread and produces a stream of thoughts
that can be observed via callbacks or polled from the UI.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Sequence

import numpy as np

from hecsn.cortex.core import CorticalCore, ContextPacket, ThoughtDepth, ThoughtResult
from hecsn.cortex.episodic_memory import Episode, EpisodicMemory, Provenance
from hecsn.cortex.drives import DriveSystem, ThalamicGate
from hecsn.cortex.narrative_self import NarrativeSelf
from hecsn.cortex.working_memory import WorkingMemory, WMItemType
from hecsn.semantics.grounding_text import match_terms, query_focused_clauses, salient_query_terms

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
    grounded_query_alignment: float = 0.0  # Mean answer/evidence alignment for external queries
    grounded_query_recovery_rate: float = 0.0  # Fraction of query answers recovered from evidence
    grounded_query_count: int = 0
    grounded_wakeful_alignment: float = 0.0  # Mean thought/evidence alignment for non-query wakeful thoughts
    grounded_wakeful_recovery_rate: float = 0.0  # Fraction of wakeful grounded thoughts recovered from evidence
    grounded_wakeful_count: int = 0
    _topic_counts: dict = field(default_factory=dict)
    _total_topics: int = 0
    _concrete_count: int = 0
    _snn_aligned_count: int = 0
    _dreams_verified: int = 0
    _dreams_total: int = 0
    _grounded_query_alignment_total: float = 0.0
    _grounded_query_recoveries: int = 0
    _grounded_wakeful_alignment_total: float = 0.0
    _grounded_wakeful_recoveries: int = 0

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

    def update_grounding_metrics(self, diagnostics: "GroundingDiagnostics" | None) -> None:
        """Update grounding metrics from a grounded query or wakeful thought."""
        if diagnostics is None or diagnostics.grounded_evidence_count <= 0:
            return
        if diagnostics.kind == "query":
            self.grounded_query_count += 1
            self._grounded_query_alignment_total += float(diagnostics.alignment_score)
            self.grounded_query_alignment = (
                self._grounded_query_alignment_total / max(1, self.grounded_query_count)
            )
            if diagnostics.fallback_used:
                self._grounded_query_recoveries += 1
            self.grounded_query_recovery_rate = (
                self._grounded_query_recoveries / max(1, self.grounded_query_count)
            )
            return

        self.grounded_wakeful_count += 1
        self._grounded_wakeful_alignment_total += float(diagnostics.alignment_score)
        self.grounded_wakeful_alignment = (
            self._grounded_wakeful_alignment_total / max(1, self.grounded_wakeful_count)
        )
        if diagnostics.fallback_used:
            self._grounded_wakeful_recoveries += 1
        self.grounded_wakeful_recovery_rate = (
            self._grounded_wakeful_recoveries / max(1, self.grounded_wakeful_count)
        )


@dataclass(frozen=True)
class GroundingDiagnostics:
    """Grounding diagnostics for a grounded cortex output."""

    kind: str
    target: str
    target_terms: tuple[str, ...] = ()
    matched_target_terms: tuple[str, ...] = ()
    evidence_supported_terms: tuple[str, ...] = ()
    grounded_evidence_count: int = 0
    response_coverage: float = 0.0
    evidence_coverage: float = 0.0
    evidence_alignment: float = 0.0
    alignment_score: float = 0.0
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target": self.target,
            "target_terms": list(self.target_terms),
            "matched_target_terms": list(self.matched_target_terms),
            "evidence_supported_terms": list(self.evidence_supported_terms),
            "grounded_evidence_count": int(self.grounded_evidence_count),
            "response_coverage": float(self.response_coverage),
            "evidence_coverage": float(self.evidence_coverage),
            "evidence_alignment": float(self.evidence_alignment),
            "alignment_score": float(self.alignment_score),
            "fallback_used": bool(self.fallback_used),
        }


@dataclass
class CognitiveSignalState:
    """Recent SNN-side signals that influence deliberation depth."""

    prediction_error_mean: float = 0.0
    prediction_error_max: float = 0.0
    predictive_confidence_mean: float = 0.5
    predictive_confidence_min: float = 0.5
    dopamine: float = 0.0
    norepinephrine: float = 0.0
    recent_concepts: tuple[str, ...] = ()
    concept_candidates: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "CognitiveSignalState":
        payload = payload or {}
        concepts = payload.get("recent_concepts", ())
        if not isinstance(concepts, (list, tuple)):
            concepts = ()
        raw_candidates = payload.get("concept_candidates", ())
        concept_candidates: list[dict[str, Any]] = []
        if isinstance(raw_candidates, (list, tuple)):
            for item in raw_candidates[:8]:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", "")).strip()
                top_terms = [str(term).strip() for term in list(item.get("top_terms") or [])[:4] if str(term).strip()]
                example_windows = [
                    str(text).strip()
                    for text in list(item.get("example_windows") or [])[:2]
                    if str(text).strip()
                ]
                concept_candidates.append(
                    {
                        "label": label,
                        "top_terms": top_terms,
                        "match_count": int(item.get("match_count", item.get("observations", 0)) or 0),
                        "observations": int(item.get("observations", item.get("match_count", 0)) or 0),
                        "uncertainty": max(0.0, min(1.0, float(item.get("uncertainty", 1.0)))),
                        "temporal_coherence": max(0.0, min(1.0, float(item.get("temporal_coherence", 0.0)))),
                        "example_windows": example_windows,
                    }
                )
        return cls(
            prediction_error_mean=max(0.0, min(1.0, float(payload.get("prediction_error_mean", 0.0)))),
            prediction_error_max=max(0.0, min(1.0, float(payload.get("prediction_error_max", 0.0)))),
            predictive_confidence_mean=max(0.0, min(1.0, float(payload.get("predictive_confidence_mean", 0.5)))),
            predictive_confidence_min=max(0.0, min(1.0, float(payload.get("predictive_confidence_min", 0.5)))),
            dopamine=max(0.0, min(1.0, float(payload.get("dopamine", 0.0)))),
            norepinephrine=max(0.0, min(1.0, float(payload.get("norepinephrine", 0.0)))),
            recent_concepts=tuple(str(c) for c in concepts[:6] if c),
            concept_candidates=tuple(concept_candidates),
        )


@dataclass
class ExplorationState:
    """Current active-exploration target chosen to reduce uncertainty."""

    target: str = ""
    reason: str = ""
    source: str = ""
    score: float = 0.0
    updated_at: float = 0.0


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
        min_thought_interval_s: float = 6.0,  # At 20 RPM budget, need ≥3s/call spacing
        sleep_dream_count: int = 3,
        sleep_cooldown_s: float = 30.0,
        on_thought: Optional[Callable[[ThoughtResult], None]] = None,
        on_sleep: Optional[Callable[[list[ThoughtResult]], None]] = None,
        curiosity_controller: Any = None,
        signal_provider: Optional[Callable[[], dict[str, Any]]] = None,
        narrative_state_path: str | None = None,
        drives: Optional[DriveSystem] = None,
        on_sleep_summary: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        self.cortex = cortex
        self.memory = memory if memory is not None else EpisodicMemory()
        self.drives = drives if drives is not None else DriveSystem()
        self.gate = ThalamicGate(self.memory, self.drives)
        self.working_memory = WorkingMemory(capacity=5, decay_rate=0.02)  # Slow decay: ~50 cycles to evict
        self.narrative_self = NarrativeSelf(persistence_path=narrative_state_path)
        self.tick_interval_s = tick_interval_ms / 1000.0
        self.min_thought_interval_s = min_thought_interval_s
        self.sleep_dream_count = sleep_dream_count
        self.sleep_cooldown_s = sleep_cooldown_s

        # Give the gate references to active cortex-side context structures.
        self.gate.working_memory = self.working_memory
        self.gate.narrative_self = self.narrative_self

        # Cortex→SNN feedback: curiosity controller for routing boosts
        self._curiosity_controller = curiosity_controller
        self._signal_provider = signal_provider

        # Callbacks
        self._on_thought = on_thought
        self._on_sleep = on_sleep
        self._on_sleep_summary = on_sleep_summary

        # State
        self.stats = BrainStats()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_thought_time = 0.0
        self._last_sleep_time = 0.0
        self._last_nonquick_time = 0.0
        self._last_deep_time = 0.0
        self._lock = threading.Lock()
        self._cognitive_signals = CognitiveSignalState()
        self._last_signal_refresh_time = 0.0
        self._depth_counts: dict[str, int] = {d.value: 0 for d in ThoughtDepth}
        self._last_depth_reason = "startup"
        self._last_depth = ThoughtDepth.QUICK.value
        self._exploration_state = ExplorationState()
        self._pending_wake_tensions: list[dict[str, Any]] = []
        self._active_wake_tensions: list[dict[str, Any]] = []
        self._last_deliberation_grounding: GroundingDiagnostics | None = None
        self._pending_sleep_request: dict[str, Any] | None = None
        self._last_sleep_request: dict[str, Any] | None = None
        self._last_sleep_cycle_summary: dict[str, Any] | None = None
        self._sleep_requests = 0
        self._requested_sleep_cycles = 0

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
            self.narrative_self.save()
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._running = False
        self.narrative_self.save()
        logger.info("ThoughtLoop stopped (thoughts=%d, dreams=%d)",
                     self.stats.thoughts_generated, self.stats.dreams_generated)

    def request_stop(self) -> None:
        """Signal stop without joining — safe to call under external lock."""
        self._stop_event.set()
        self._running = False

    def set_curiosity_controller(self, controller: Any) -> None:
        """Set or update the curiosity controller for cortex→SNN feedback."""
        self._curiosity_controller = controller

    def set_signal_provider(self, provider: Optional[Callable[[], dict[str, Any]]]) -> None:
        """Set or update the SNN cognitive-signal provider used for depth tuning."""
        self._signal_provider = provider

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
                "model": getattr(self.cortex, "model", type(self.cortex).__name__),
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
                    "prediction_error": round(self.drives.state.prediction_error, 3),
                    "uncertainty": round(self.drives.state.uncertainty, 3),
                    "exploration_urgency": round(self.drives.state.exploration_urgency, 3),
                    "dopamine": round(self.drives.state.dopamine, 3),
                    "serotonin": round(self.drives.state.serotonin, 3),
                    "norepinephrine": round(self.drives.state.norepinephrine, 3),
                    "acetylcholine": round(self.drives.state.acetylcholine, 3),
                    "arousal": round(self.drives.state.arousal, 3),
                },
                "quality": {
                    "topic_diversity": round(s.topic_diversity, 3),
                    "concreteness_ratio": round(s.concreteness_ratio, 3),
                    "snn_alignment": round(s.snn_alignment, 3),
                    "dream_verification_rate": round(s.dream_verification_rate, 3),
                },
                "grounding": {
                    "query_answers_evaluated": int(s.grounded_query_count),
                    "mean_query_alignment": round(s.grounded_query_alignment, 3),
                    "query_recovery_rate": round(s.grounded_query_recovery_rate, 3),
                    "wakeful_thoughts_evaluated": int(s.grounded_wakeful_count),
                    "mean_wakeful_alignment": round(s.grounded_wakeful_alignment, 3),
                    "wakeful_recovery_rate": round(s.grounded_wakeful_recovery_rate, 3),
                },
                "gating": {
                    "startup_quiet": bool(self.drives.startup_quiet),
                    "pending_grounded_observations": int(self.drives.pending_grounded_observations),
                    "pending_substrate_wakes": int(self.drives.pending_substrate_wakes),
                    "active_tension_count": int(len(self._active_wake_tensions)),
                    "substrate_hysteresis_active": bool(self.drives.substrate_hysteresis_active),
                    "substrate_hysteresis_reason": self.drives.substrate_hysteresis_reason,
                    "substrate_hysteresis_updates": int(self.drives.substrate_hysteresis_updates),
                    "last_gate_reason": self.drives.last_gate_reason,
                    "inhibition_reason": self.drives._inhibition_reason(),
                },
                "cognitive_signals": {
                    "prediction_error_mean": round(self._cognitive_signals.prediction_error_mean, 3),
                    "prediction_error_max": round(self._cognitive_signals.prediction_error_max, 3),
                    "predictive_confidence_mean": round(self._cognitive_signals.predictive_confidence_mean, 3),
                    "predictive_confidence_min": round(self._cognitive_signals.predictive_confidence_min, 3),
                    "dopamine": round(self._cognitive_signals.dopamine, 3),
                    "norepinephrine": round(self._cognitive_signals.norepinephrine, 3),
                    "recent_concepts": list(self._cognitive_signals.recent_concepts),
                },
                "neuromodulation": {
                    "da_reward": round(self.drives.state.da_reward, 3),
                    "da_novelty": round(self.drives.state.da_novelty, 3),
                    "da_salience": round(self.drives.state.da_salience, 3),
                    "ne_alerting": round(self.drives.state.ne_alerting, 3),
                    "ne_orienting": round(self.drives.state.ne_orienting, 3),
                    "ach_learning": round(self.drives.state.ach_learning, 3),
                    "ach_attention": round(self.drives.state.ach_attention, 3),
                    "serotonin_patience": round(self.drives.state.serotonin_patience, 3),
                },
                "depth_policy": {
                    "last_depth": self._last_depth,
                    "last_reason": self._last_depth_reason,
                    "counts": dict(self._depth_counts),
                },
                "active_exploration": {
                    "target": self._exploration_state.target,
                    "reason": self._exploration_state.reason,
                    "source": self._exploration_state.source,
                    "score": round(self._exploration_state.score, 3),
                    "updated_at": self._exploration_state.updated_at,
                },
                "pending_wake_tensions": list(self._pending_wake_tensions),
                "active_wake_tensions": list(self._active_wake_tensions),
                "sleep_control": self._sleep_control_snapshot_locked(),
                "recent_thoughts": list(self._thought_history),
                "episodic_memory": self.memory.stats,
                "working_memory": self.working_memory.snapshot(),
                "narrative_self": self.narrative_self.snapshot(),
            }

    # -- External interface --

    @staticmethod
    def _normalize_sleep_control_text(value: Any, *, max_length: int = 240) -> str:
        return " ".join(str(value).split()).strip()[:max(1, int(max_length))]

    def _sleep_request_snapshot_locked(self, request: dict[str, Any] | None) -> dict[str, Any] | None:
        if request is None:
            return None
        snapshot = deepcopy(request)
        snapshot["coalesced_count"] = max(1, int(snapshot.get("coalesced_count", 1) or 1))
        if not isinstance(snapshot.get("metadata"), dict):
            snapshot["metadata"] = {}
        return snapshot

    def _sleep_control_snapshot_locked(self) -> dict[str, Any]:
        return {
            "requests_submitted": int(self._sleep_requests),
            "requested_cycles_completed": int(self._requested_sleep_cycles),
            "pending_request": self._sleep_request_snapshot_locked(self._pending_sleep_request),
            "last_request": self._sleep_request_snapshot_locked(self._last_sleep_request),
            "last_cycle": None if self._last_sleep_cycle_summary is None else deepcopy(self._last_sleep_cycle_summary),
        }

    def request_sleep(
        self,
        *,
        source: str = "operator",
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_source = self._normalize_sleep_control_text(source, max_length=64).lower() or "operator"
        normalized_reason = self._normalize_sleep_control_text(reason)
        normalized_metadata = deepcopy(metadata) if isinstance(metadata, dict) else {}
        requested_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._sleep_requests += 1
            existing = self._pending_sleep_request
            if existing is not None:
                existing["coalesced_count"] = max(1, int(existing.get("coalesced_count", 1) or 1)) + 1
                if normalized_reason and not self._normalize_sleep_control_text(existing.get("reason", "")):
                    existing["reason"] = normalized_reason
                if normalized_metadata:
                    merged_metadata = dict(existing.get("metadata") or {})
                    for key, value in normalized_metadata.items():
                        if str(key) == "control_id" and merged_metadata.get("control_id"):
                            continue
                        merged_metadata[str(key)] = deepcopy(value)
                    existing["metadata"] = merged_metadata
                snapshot = self._sleep_request_snapshot_locked(existing)
                self._last_sleep_request = None if snapshot is None else deepcopy(snapshot)
                return {
                    "accepted": True,
                    "coalesced": True,
                    "running": bool(self._running),
                    "request": snapshot,
                    "sleep_control": self._sleep_control_snapshot_locked(),
                }

            request = {
                "source": normalized_source,
                "reason": normalized_reason,
                "requested_at": requested_at,
                "coalesced_count": 1,
                "metadata": normalized_metadata,
            }
            self._pending_sleep_request = request
            self._last_sleep_request = deepcopy(request)
            return {
                "accepted": True,
                "coalesced": False,
                "running": bool(self._running),
                "request": self._sleep_request_snapshot_locked(request),
                "sleep_control": self._sleep_control_snapshot_locked(),
            }

    def submit_query(self, query: str) -> None:
        """Submit an external query — the brain will answer it."""
        with self._lock:
            self.gate.submit_query(query)

    def _has_pending_query_locked(self) -> bool:
        return self.gate.has_pending_query()

    def _has_pending_grounded_observation_locked(self) -> bool:
        return self.drives.has_pending_grounded_observation()

    def _consume_pending_query(self) -> str:
        with self._lock:
            return self.gate.pop_query()

    def _has_unresolved_tension(self) -> bool:
        return bool(self._pending_wake_tensions) or bool(self._active_wake_tensions)

    def inject_observation(
        self,
        content: str,
        topics: Sequence[str] = (),
        salience: float = 0.7,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Inject an external observation into memory."""
        with self._lock:
            self.memory.store(
                content=content,
                provenance=Provenance.OBSERVED,
                topics=topics,
                salience=salience,
                metadata=metadata,
            )
            self.drives.update_from_grounded_observation()

    def inject_action_result(
        self,
        content: str,
        topics: Sequence[str] = (),
        *,
        success: bool,
        confidence: float,
        contradicted: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Inject a verified or contradicted action outcome into memory."""
        with self._lock:
            provenance = Provenance.CONTRADICTED if contradicted else Provenance.VERIFIED
            clipped_confidence = max(0.0, min(1.0, float(confidence)))
            valence = 0.35 if success and not contradicted else -0.25
            salience = 0.84 if success and not contradicted else 0.74
            self.memory.store(
                content=content,
                provenance=provenance,
                topics=topics,
                salience=salience,
                confidence=clipped_confidence,
                emotional_valence=valence,
                metadata=metadata,
            )
            self.working_memory.update_from_thought(
                content,
                tuple(topics),
                clipped_confidence,
                valence,
                WMItemType.INSIGHT if success and not contradicted else WMItemType.TENSION,
            )
            self.drives.update_from_grounded_observation()

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
        self._refresh_cognitive_signals()
        self.drives.tick()
        self.stats.ticks += 1

        now = time.time()
        with self._lock:
            query_pending = self._has_pending_query_locked()
            grounded_pending = self._has_pending_grounded_observation_locked()
            substrate_pending = self.drives.has_pending_substrate_wake()
            sleep_request = None
            if not query_pending and not grounded_pending and self._pending_sleep_request is not None:
                sleep_request = self._sleep_request_snapshot_locked(self._pending_sleep_request)
                self._pending_sleep_request = None
        has_tension = self._has_unresolved_tension()
        should_answer = self.drives.should_answer_now(query_pending=query_pending)

        # External queries and fresh grounded observations are wake events and
        # take priority over passive sleep entry. Explicit sleep requests use
        # the same sleep-cycle implementation but bypass fatigue/cooldown gating.
        if sleep_request is not None:
            dreams = self._sleep_cycle(trigger=sleep_request)
            if self._on_sleep:
                self._on_sleep(dreams)
            if self._on_sleep_summary and self._last_sleep_cycle_summary is not None:
                self._on_sleep_summary(deepcopy(self._last_sleep_cycle_summary))
            return None
        if (
            not query_pending
            and not grounded_pending
            and self.drives.should_sleep()
            and (now - self._last_sleep_time) > self.sleep_cooldown_s
        ):
            dreams = self._sleep_cycle()
            if self._on_sleep:
                self._on_sleep(dreams)
            if self._on_sleep_summary and self._last_sleep_cycle_summary is not None:
                self._on_sleep_summary(deepcopy(self._last_sleep_cycle_summary))
            return None

        interval_ok = (
            force
            or should_answer
            or grounded_pending
            or substrate_pending
            or (now - self._last_thought_time) > self._effective_thought_interval()
        )
        should_think = should_answer or self.drives.should_think(
            grounded_observation_pending=grounded_pending,
            has_tension=has_tension,
        )
        if should_think and interval_ok:
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
                self._refresh_cognitive_signals()
                # --- fast tick (under lock) ---
                with self._lock:
                    self.drives.tick()
                    self.stats.ticks += 1
                    now = time.time()
                    query_pending = self._has_pending_query_locked()
                    grounded_pending = self._has_pending_grounded_observation_locked()
                    substrate_pending = self.drives.has_pending_substrate_wake()
                    should_answer = self.drives.should_answer_now(query_pending=query_pending)
                    has_tension = bool(self._pending_wake_tensions) or bool(self._active_wake_tensions)
                    sleep_request = None
                    if not query_pending and not grounded_pending and self._pending_sleep_request is not None:
                        sleep_request = self._sleep_request_snapshot_locked(self._pending_sleep_request)
                        self._pending_sleep_request = None

                    should_sleep = (
                        sleep_request is not None
                        or (
                            not query_pending
                            and not grounded_pending
                            and self.drives.should_sleep()
                            and (now - self._last_sleep_time) > self.sleep_cooldown_s
                        )
                    )
                    interval_ok = (
                        should_answer
                        or grounded_pending
                        or substrate_pending
                        or (now - self._last_thought_time) > self._effective_thought_interval()
                    )
                    should_think = (
                        not should_sleep
                        and interval_ok
                        and (
                            should_answer
                            or self.drives.should_think(
                                grounded_observation_pending=grounded_pending,
                                has_tension=has_tension,
                            )
                        )
                    )

                # --- slow operations (outside lock) ---
                if should_sleep:
                    with self._lock:
                        self.stats.current_mode = "sleeping"
                        self.stats.is_sleeping = True
                    dreams = self._sleep_cycle(trigger=sleep_request)
                    with self._lock:
                        self.stats.is_sleeping = False
                        self.stats.current_mode = "idle"
                    if self._on_sleep and dreams:
                        self._on_sleep(dreams)
                    if self._on_sleep_summary and self._last_sleep_cycle_summary is not None:
                        self._on_sleep_summary(deepcopy(self._last_sleep_cycle_summary))
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

    def _refresh_cognitive_signals(self) -> CognitiveSignalState:
        """Refresh predictive/surprise signals from the SNN side."""
        if self._signal_provider is None:
            return self._cognitive_signals
        now = time.time()
        if (now - self._last_signal_refresh_time) < 0.2:
            return self._cognitive_signals
        try:
            signals = CognitiveSignalState.from_mapping(self._signal_provider())
        except Exception:
            logger.debug("Signal provider failed", exc_info=True)
            return self._cognitive_signals
        self._last_signal_refresh_time = now
        self._cognitive_signals = signals
        self.drives.update_from_prediction_error(
            signals.prediction_error_mean,
            signals.prediction_error_max,
            signals.predictive_confidence_mean,
            signals.predictive_confidence_min,
        )
        if signals.recent_concepts:
            self.gate.update_snn_concepts(list(signals.recent_concepts))
        self._update_active_exploration_target()
        return self._cognitive_signals

    def _set_depth_decision(self, depth: ThoughtDepth, reason: str) -> ThoughtDepth:
        self._last_depth = depth.value
        self._last_depth_reason = reason
        return depth

    @staticmethod
    def _candidate_target_text(topic: str = "", fallback: str = "") -> str:
        text = " ".join(
            str(topic or fallback).replace("/", " ").replace("|", " ").split()
        ).strip(" ,;:.")
        return text[:120]

    @classmethod
    def _lexical_target_quality(cls, text: str) -> float:
        cleaned = cls._candidate_target_text(text)
        if not cleaned:
            return 0.0
        tokens = [token for token in re.findall(r"[a-zA-Z][a-zA-Z'-]+", cleaned.lower()) if token]
        if not tokens:
            return 0.0

        stopwords = {
            "the", "and", "with", "that", "this", "from", "into", "about", "than",
            "their", "there", "because", "while", "where", "which", "have", "has",
            "when", "then", "they", "them", "what", "how", "why", "does", "under",
            "through", "between", "could", "would", "should", "these", "those", "around",
        }
        generic = {
            "thing", "things", "stuff", "object", "objects", "system", "systems",
            "process", "processes", "phenomenon", "phenomena", "effect", "effects",
        }

        score = 0.20
        if 2 <= len(tokens) <= 4:
            score += 0.25
        elif len(tokens) == 1:
            score += 0.06
        elif len(tokens) <= 6:
            score += 0.10

        content_tokens = [token for token in tokens if len(token) >= 3 and token not in stopwords]
        score += 0.25 * (len(content_tokens) / max(1.0, float(len(tokens))))
        if len(set(tokens)) == len(tokens):
            score += 0.05
        if any(token in generic for token in tokens):
            score -= 0.08
        if any(len(token) > 16 for token in tokens):
            score -= 0.10
        if cleaned.endswith("?"):
            score -= 0.12
        if (
            len(tokens) >= 2
            and tokens[0].endswith("s")
            and not tokens[0].endswith(("ss", "us", "is"))
            and not tokens[1].endswith("s")
        ):
            score -= 0.18
        if len(content_tokens) == 1 and len(tokens) >= 3:
            score -= 0.08
        return max(0.0, min(1.0, score))

    def _memory_grounding_score(self, target: str) -> float:
        words = self._text_keywords(target)
        if not words:
            return 0.0

        best = 0.0
        support_hits = 0
        for ep in self.memory.recall_recent(top_k=18):
            episode_words: set[str] = set()
            for topic in ep.topics:
                episode_words.update(self._text_keywords(str(topic)))
            episode_words.update(self._text_keywords(ep.content))
            if not episode_words:
                continue
            overlap = len(words & episode_words) / max(1.0, min(float(len(words)), float(len(episode_words))))
            if overlap <= 0.0:
                continue
            provenance_weight = (
                1.0 if ep.provenance in (Provenance.OBSERVED, Provenance.VERIFIED)
                else 0.75 if ep.provenance == Provenance.INFERRED
                else 0.45
            )
            weighted = overlap * provenance_weight
            best = max(best, weighted)
            if weighted >= 0.34:
                support_hits += 1

        if support_hits:
            best = max(best, min(1.0, 0.22 * support_hits))
        return max(0.0, min(1.0, best))

    def _example_window_support(self, target: str, example_windows: Sequence[str]) -> float:
        words = self._text_keywords(target)
        if not words:
            return 0.0
        best = 0.0
        for example in list(example_windows)[:2]:
            example_words = self._text_keywords(str(example))
            if not example_words:
                continue
            overlap = len(words & example_words) / max(1.0, min(float(len(words)), float(len(example_words))))
            best = max(best, overlap)
        return max(0.0, min(1.0, best))

    def _concept_candidate_target(self, candidate: dict[str, Any]) -> tuple[str, float, float, float]:
        label = self._candidate_target_text(str(candidate.get("label", "")))
        top_terms = [
            self._candidate_target_text(str(term))
            for term in list(candidate.get("top_terms") or [])[:4]
            if self._candidate_target_text(str(term))
        ]
        options: list[str] = []
        if label:
            options.append(label)
        if top_terms:
            options.append(" ".join(top_terms[:2]).strip())
            if len(top_terms) >= 3:
                options.append(" ".join(top_terms[:3]).strip())

        examples = [str(text) for text in list(candidate.get("example_windows") or [])[:2] if str(text).strip()]
        structural = max(0.0, min(1.0, (
            0.35 * min(1.0, float(candidate.get("observations", 0)) / 3.0)
            + 0.25 * min(1.0, float(candidate.get("match_count", 0)) / 3.0)
            + 0.20 * float(candidate.get("temporal_coherence", 0.0))
            + 0.20 * (1.0 - float(candidate.get("uncertainty", 1.0)))
        )))

        best_target = ""
        best_quality = 0.0
        best_grounding = 0.0
        best_score = -1.0
        for option in dict.fromkeys(option for option in options if option):
            quality = self._lexical_target_quality(option)
            grounding = max(
                self._memory_grounding_score(option),
                0.65 * self._example_window_support(option, examples),
            )
            combined = 0.65 * quality + 0.35 * grounding
            if combined > best_score:
                best_target = option
                best_quality = quality
                best_grounding = grounding
                best_score = combined
        return best_target, best_quality, best_grounding, structural

    def _set_active_exploration_target(
        self,
        target: str,
        *,
        reason: str,
        source: str,
        score: float,
    ) -> None:
        cleaned = self._candidate_target_text(target)
        if not cleaned:
            return
        self._exploration_state = ExplorationState(
            target=cleaned,
            reason=" ".join(reason.split()).strip()[:160],
            source=source[:40],
            score=max(0.0, min(1.0, float(score))),
            updated_at=time.time(),
        )
        self.gate.set_active_exploration_target(
            cleaned,
            reason=self._exploration_state.reason,
            source=self._exploration_state.source,
            score=self._exploration_state.score,
        )

    def _clear_active_exploration_target(self) -> None:
        self._exploration_state = ExplorationState()
        self.gate.clear_active_exploration_target()

    @staticmethod
    def _topic_overlaps_avoid(topic: str, avoid_words: set[str]) -> bool:
        topic_words = {w.lower().strip(".,;:!?\"'()-") for w in topic.split() if len(w) >= 3}
        return bool(topic_words & avoid_words)

    def _update_active_exploration_target(self, result: ThoughtResult | None = None) -> None:
        """Choose what the brain should actively explore next.

        This is the lightweight active-inference loop: unresolved questions,
        tensions, low-confidence topics, and high-error recent concepts compete
        to become the next wakeful Direction target. SNN-derived targets are now
        filtered by linguistic quality and cheap lexical grounding against recent
        episodic memory so stale fragmentary concept labels do not dominate.
        """
        avoid_words = {w.lower() for w in self.drives.anti_rumination.suggest_topic_avoidance()}
        now = time.time()
        current = self._exploration_state
        candidates: list[tuple[float, str, str, str]] = []

        def add_candidate(
            target: str,
            *,
            reason: str,
            source: str,
            base_score: float,
            min_quality: float = 0.35,
            example_windows: Sequence[str] = (),
            structural: float = 0.0,
        ) -> None:
            cleaned = self._candidate_target_text(target)
            if not cleaned or self._topic_overlaps_avoid(cleaned, avoid_words):
                return
            quality = self._lexical_target_quality(cleaned)
            grounding = self._memory_grounding_score(cleaned)
            token_count = len(re.findall(r"[a-zA-Z][a-zA-Z'-]+", cleaned.lower()))
            if quality < min_quality:
                return
            if source == "snn" and grounding < 0.22 and not (quality >= 0.75 and structural >= 0.80 and token_count >= 2):
                return
            if source == "snn" and quality < 0.55 and grounding < 0.30:
                return
            if source == "snn" and token_count == 1 and (grounding < 0.45 or structural < 0.75):
                return
            if source == "snn":
                score = (
                    base_score
                    + 0.12 * quality
                    + 0.16 * grounding
                    + 0.14 * structural
                )
            else:
                score = base_score + 0.08 * quality + 0.06 * grounding
            candidates.append((max(0.0, min(1.0, score)), cleaned, reason, source))

        focus_items = [
            item for item in self.working_memory.items
            if item.item_type in (WMItemType.QUESTION, WMItemType.TENSION)
        ]
        strongest_focus = max(focus_items, key=lambda item: item.strength) if focus_items else None
        if strongest_focus is not None:
            target = self._candidate_target_text(strongest_focus.topic, strongest_focus.content)
            add_candidate(
                target,
                reason=strongest_focus.item_type.value,
                source="working_memory",
                base_score=0.82 + 0.08 * self.drives.state.ne_orienting + 0.05 * max(0.0, strongest_focus.strength - 0.5),
                min_quality=0.25,
            )

        for item in [*self._active_wake_tensions[:2], *self._pending_wake_tensions[:2]]:
            topics = item.get("topics", [])
            target = self._candidate_target_text(topics[0] if topics else "", str(item.get("content", "")))
            add_candidate(
                target,
                reason="wake_tension",
                source="sleep",
                base_score=0.72 + 0.12 * self.drives.state.ne_alerting,
                min_quality=0.25,
            )

        if result is not None:
            if result.confidence < 0.6:
                for index, topic in enumerate(result.topics[:3]):
                    add_candidate(
                        self._candidate_target_text(topic),
                        reason="low_confidence_thought",
                        source="cortex",
                        base_score=(
                            0.54
                            - 0.06 * index
                            + 0.30 * (0.6 - result.confidence)
                            + 0.08 * self.drives.state.ach_attention
                            + 0.06 * self.drives.state.da_novelty
                        ),
                    )
            if result.action_intent == "explore" and result.topics:
                add_candidate(
                    self._candidate_target_text(result.topics[0]),
                    reason="explicit_explore_action",
                    source="cortex",
                    base_score=0.64 + 0.10 * self.drives.state.da_novelty,
                )

        signals = self._cognitive_signals
        if signals.prediction_error_mean >= 0.22 or signals.predictive_confidence_min <= 0.35:
            signal_strength = max(signals.prediction_error_mean, 1.0 - signals.predictive_confidence_min)
            if signals.concept_candidates:
                for index, candidate in enumerate(signals.concept_candidates[:5]):
                    target, quality, grounding, structural = self._concept_candidate_target(candidate)
                    if not target:
                        continue
                    if quality < 0.42 or (grounding < 0.18 and structural < 0.45):
                        continue
                    add_candidate(
                        target,
                        reason="prediction_error",
                        source="snn",
                        base_score=(
                            0.18
                            + 0.16 * signal_strength
                            + 0.06 * self.drives.state.da_salience
                            + 0.06 * self.drives.state.ne_orienting
                            + 0.06 * self.drives.state.ach_attention
                            - 0.04 * index
                        ),
                        min_quality=0.42,
                        example_windows=list(candidate.get("example_windows") or []),
                        structural=structural,
                    )
            else:
                for index, concept in enumerate(signals.recent_concepts[:4]):
                    add_candidate(
                        self._candidate_target_text(concept),
                        reason="prediction_error",
                        source="snn",
                        base_score=(
                            0.20
                            + 0.16 * signal_strength
                            + 0.06 * self.drives.state.da_salience
                            + 0.06 * self.drives.state.ne_orienting
                            + 0.06 * self.drives.state.ach_attention
                            - 0.05 * index
                        ),
                        min_quality=0.45,
                    )

        current_quality = self._lexical_target_quality(current.target) if current.target else 0.0
        current_grounding = self._memory_grounding_score(current.target) if current.target else 0.0
        current_effective = current.score
        if current.target:
            current_effective -= min(0.18, (now - current.updated_at) * 0.003)
            if current.source == "snn":
                current_effective -= 0.14 * max(0.0, 0.65 - current_quality)
                current_effective -= 0.18 * max(0.0, 0.35 - current_grounding)
                if current_grounding < 0.15:
                    current_effective -= 0.18

        if not candidates:
            if current.target and (
                (now - current.updated_at) > 45.0
                or (
                    current.source == "snn"
                    and (current_quality < 0.48 or current_grounding < 0.20)
                    and (now - current.updated_at) > 10.0
                )
            ):
                self._clear_active_exploration_target()
            return

        best_score, best_target, best_reason, best_source = max(candidates, key=lambda item: item[0])
        if current.target and current.target == best_target and current_effective >= best_score:
            return
        if best_score < 0.45:
            if current.target and current_effective < 0.38 and (now - current.updated_at) > 10.0:
                self._clear_active_exploration_target()
            return
        if current.target and current.target != best_target and current_effective >= best_score:
            return
        self._set_active_exploration_target(
            best_target,
            reason=best_reason,
            source=best_source,
            score=best_score,
        )

    def _queue_wake_tension(
        self,
        text: str,
        topics: Sequence[str] = (),
        *,
        salience: float = 0.7,
        continuation_cycles: int = 2,
    ) -> None:
        """Queue an unresolved contradiction to be revisited after sleep."""
        item = {
            "content": text[:220],
            "topics": list(topics[:4]),
            "salience": max(0.1, min(1.0, float(salience))),
            "remaining_cycles": max(1, int(continuation_cycles)),
        }
        self.drives.update_from_unresolved_tension()
        self._pending_wake_tensions.append(item)
        self._pending_wake_tensions = self._pending_wake_tensions[-4:]

    def _inject_pending_wake_tensions(self) -> None:
        """Hydrate queued sleep contradictions into working memory.

        Unresolved tensions persist for a bounded number of deliberation cycles
        so continuation stays tied to explicit unresolved state rather than a
        single one-shot wake event.
        """
        if self._pending_wake_tensions:
            self._active_wake_tensions.extend(self._pending_wake_tensions)
            self._active_wake_tensions = self._active_wake_tensions[-4:]
            self._pending_wake_tensions.clear()
        if not self._active_wake_tensions:
            return

        next_active: list[dict[str, Any]] = []
        for item in self._active_wake_tensions[:2]:
            topics = tuple(str(t) for t in item.get("topics", []) if t)
            self.working_memory.update_from_thought(
                str(item.get("content", "")),
                topics,
                confidence=0.4,
                emotional_valence=-0.2,
                item_type=WMItemType.TENSION,
            )
        for item in self._active_wake_tensions:
            remaining = int(item.get("remaining_cycles", 1)) - 1
            if remaining > 0:
                next_item = dict(item)
                next_item["remaining_cycles"] = remaining
                next_active.append(next_item)
        self._active_wake_tensions = next_active[-4:]

    # -- Depth selection (System 1 / System 2) --

    def _choose_depth(self, *, query_pending: bool | None = None) -> ThoughtDepth:
        """Choose System-1 vs System-2 depth from real SNN-side signals.

        Depth is now driven by predictive-processing style triggers rather than
        only heuristics:
        - DEEP when prediction error is high, tension is active, or a query is hard
        - STANDARD when error/uncertainty is moderate or an answer needs continuity
        - QUICK by default for cheap broad scanning

        Budget guards keep this inside the 20 RPM safety budget by cooling down
        non-quick thoughts unless the signal is genuinely strong.
        """
        signals = self._refresh_cognitive_signals()
        now = time.time()
        if query_pending is None:
            with self._lock:
                query_pending = self._has_pending_query_locked()
        tension = self.working_memory.has_tension()
        question = self.working_memory.has_question()

        deep_candidate = (
            tension
            or signals.prediction_error_max >= 0.55
            or self.drives.state.ne_alerting >= 0.78
            or self.drives.state.da_salience >= 0.78
            or self.drives.state.arousal >= 0.80
            or (query_pending and signals.prediction_error_mean >= 0.30)
        )
        if deep_candidate:
            if (
                (now - self._last_deep_time) < 45.0
                and not tension
                and not query_pending
                and signals.prediction_error_max < 0.75
                and self.drives.state.ne_alerting < 0.88
            ):
                if question or signals.prediction_error_mean >= 0.35:
                    return self._set_depth_decision(ThoughtDepth.STANDARD, "deep_cooldown")
                return self._set_depth_decision(ThoughtDepth.QUICK, "deep_cooldown")
            if tension:
                return self._set_depth_decision(ThoughtDepth.DEEP, "working_memory_tension")
            if signals.prediction_error_max >= 0.55:
                return self._set_depth_decision(ThoughtDepth.DEEP, "high_prediction_error")
            if self.drives.state.ne_alerting >= 0.78:
                return self._set_depth_decision(ThoughtDepth.DEEP, "ne_alerting")
            if self.drives.state.da_salience >= 0.78:
                return self._set_depth_decision(ThoughtDepth.DEEP, "da_salience")
            if query_pending:
                return self._set_depth_decision(ThoughtDepth.DEEP, "query_with_high_uncertainty")
            return self._set_depth_decision(ThoughtDepth.DEEP, "high_arousal")

        standard_candidate = (
            query_pending
            or signals.prediction_error_mean >= 0.28
            or signals.predictive_confidence_min <= 0.30
            or signals.norepinephrine >= 0.65
            or self.drives.state.ach_attention >= 0.68
            or self.drives.state.da_novelty >= 0.65
            or (question and signals.prediction_error_mean >= 0.18)
            or (self.stats.thoughts_generated > 0 and self.stats.thoughts_generated % 10 == 0)
        )
        if standard_candidate:
            if (
                (now - self._last_nonquick_time) < 20.0
                and not query_pending
                and signals.prediction_error_mean < 0.35
                and signals.norepinephrine < 0.75
                and self.drives.state.ach_attention < 0.78
            ):
                return self._set_depth_decision(ThoughtDepth.QUICK, "nonquick_cooldown")
            if query_pending:
                return self._set_depth_decision(ThoughtDepth.STANDARD, "external_query")
            if signals.predictive_confidence_min <= 0.30:
                return self._set_depth_decision(ThoughtDepth.STANDARD, "low_predictive_confidence")
            if signals.prediction_error_mean >= 0.28:
                return self._set_depth_decision(ThoughtDepth.STANDARD, "moderate_prediction_error")
            if signals.norepinephrine >= 0.65:
                return self._set_depth_decision(ThoughtDepth.STANDARD, "high_norepinephrine")
            if self.drives.state.ach_attention >= 0.68:
                return self._set_depth_decision(ThoughtDepth.STANDARD, "ach_attention")
            if self.drives.state.da_novelty >= 0.65:
                return self._set_depth_decision(ThoughtDepth.STANDARD, "da_novelty")
            if question:
                return self._set_depth_decision(ThoughtDepth.STANDARD, "open_question")
            return self._set_depth_decision(ThoughtDepth.STANDARD, "periodic_standard")

        return self._set_depth_decision(ThoughtDepth.QUICK, "default_quick_scan")

    # -- Deliberation (the core thinking mechanism) --

    def _deliberate(self) -> ThoughtResult:
        """Fire one deliberation cycle — depth-adapted.

        QUICK: single LLM call (original behavior).
        STANDARD: observe → question (2 calls).
        DEEP: observe → question → reason → synthesize (4 calls).

        Each phase feeds its output into working memory, which is then
        broadcast as context for the next phase. This creates genuine
        chain-of-thought reasoning.
        """
        # Clear working memory from previous chain — each deliberation cycle
        # explores a fresh topic. Working memory accumulates WITHIN a chain
        # (observe→question→reason→synthesize) but resets between chains.
        self.working_memory.clear()
        self._inject_pending_wake_tensions()
        query_text = self._consume_pending_query()
        if not query_text:
            grounded_consumed = self.drives.consume_grounded_observation()
            if not grounded_consumed:
                self.drives.consume_substrate_wake()
        depth = self._choose_depth(query_pending=bool(query_text))

        grounding: GroundingDiagnostics | None = None
        self._last_deliberation_grounding = None
        if query_text:
            result, grounding = self._deliberate_external_query(query_text, depth)
        elif depth == ThoughtDepth.QUICK:
            result = self._deliberate_quick()
            grounding = self._last_deliberation_grounding
        elif depth == ThoughtDepth.STANDARD:
            result = self._deliberate_standard()
            grounding = self._last_deliberation_grounding
        else:
            result = self._deliberate_deep()
            grounding = self._last_deliberation_grounding

        # Post-process the final result (same for all depths)
        self._post_process_thought(result, depth, grounding=grounding)
        return result

    def _set_temperature(self) -> float | None:
        """Set cortex temperature from arousal. Returns old value."""
        old_temp = getattr(self.cortex, "temperature", None)
        if old_temp is not None:
            self.cortex.temperature = 0.6 + 0.4 * self.drives.state.arousal
        return old_temp

    def _restore_temperature(self, old_temp: float | None) -> None:
        """Restore cortex temperature after deliberation."""
        if old_temp is not None:
            self.cortex.temperature = old_temp

    @staticmethod
    def _coverage_ratio(expected_terms: Sequence[str], matched_terms: Sequence[str]) -> float:
        expected = {str(term).strip().lower() for term in expected_terms if str(term).strip()}
        if not expected:
            return 0.0
        matched = {str(term).strip().lower() for term in matched_terms if str(term).strip()}
        return min(1.0, float(len(expected & matched)) / float(len(expected)))

    def _grounding_diagnostics(
        self,
        *,
        kind: str,
        target_text: str,
        response_text: str,
        packet: ContextPacket,
        fallback_used: bool = False,
    ) -> GroundingDiagnostics:
        target = " ".join(str(target_text).split()).strip()
        target_terms = tuple(salient_query_terms(target)[:8]) if target else ()
        evidence_texts = [str(item.text).strip() for item in packet.grounded_evidence if str(item.text).strip()]
        evidence_text = " ".join(evidence_texts).strip()
        matched_target_terms = tuple(match_terms(target_terms, response_text)) if target_terms else ()
        evidence_supported_terms = tuple(match_terms(target_terms, evidence_text)) if target_terms and evidence_text else ()
        response_coverage = self._coverage_ratio(target_terms, matched_target_terms)
        evidence_coverage = self._coverage_ratio(target_terms, evidence_supported_terms)
        response_terms = tuple(salient_query_terms(response_text)[:12])
        supported_response_terms = tuple(match_terms(response_terms, evidence_text)) if response_terms and evidence_text else ()
        evidence_alignment = max(
            self._coverage_ratio(response_terms, supported_response_terms),
            self._text_overlap(response_text, evidence_text),
        ) if evidence_text else 0.0
        alignment_score = (
            0.55 * response_coverage + 0.45 * evidence_alignment
            if target_terms else evidence_alignment
        ) if evidence_texts else 0.0
        return GroundingDiagnostics(
            kind=kind,
            target=target,
            target_terms=target_terms,
            matched_target_terms=matched_target_terms,
            evidence_supported_terms=evidence_supported_terms,
            grounded_evidence_count=len(evidence_texts),
            response_coverage=max(0.0, min(1.0, float(response_coverage))),
            evidence_coverage=max(0.0, min(1.0, float(evidence_coverage))),
            evidence_alignment=max(0.0, min(1.0, float(evidence_alignment))),
            alignment_score=max(0.0, min(1.0, float(alignment_score))),
            fallback_used=fallback_used,
        )

    def _build_grounded_fallback(self, *, target_text: str, packet: ContextPacket) -> str:
        target_terms = tuple(salient_query_terms(target_text)[:8]) if str(target_text).strip() else ()
        selected: list[str] = []
        covered_terms: set[str] = set()

        for item in packet.grounded_evidence:
            evidence_text = str(item.text).strip()
            if not evidence_text:
                continue
            clauses = query_focused_clauses(evidence_text, target_terms) if target_terms else [evidence_text]
            if not clauses:
                clauses = [evidence_text]
            for clause in clauses:
                normalized = " ".join(str(clause).split()).strip()
                if not normalized:
                    continue
                matched = {
                    str(term).strip().lower()
                    for term in match_terms(target_terms, normalized)
                    if str(term).strip()
                } if target_terms else set()
                if target_terms and not matched:
                    continue
                adds_new_terms = bool(matched - covered_terms)
                if selected and target_terms and not adds_new_terms:
                    continue
                if any(self._text_overlap(normalized, existing) >= 0.78 for existing in selected):
                    continue
                selected.append(normalized)
                covered_terms.update(matched)
                if len(selected) >= 2 or (target_terms and len(covered_terms) >= len(target_terms)):
                    break
            if len(selected) >= 2 or (target_terms and len(covered_terms) >= len(target_terms)):
                break

        if not selected:
            return ""

        fallback = self._dedupe_sentences(" ".join(selected).strip())
        if fallback and fallback[-1] not in ".!?":
            fallback = f"{fallback}."
        if len(fallback) > 320:
            clipped = fallback[:320].rstrip()
            fallback = clipped.rsplit(" ", 1)[0].rstrip(" ,;:") + "..."
        return fallback

    def _stabilize_grounded_result(
        self,
        *,
        kind: str,
        target_text: str,
        packet: ContextPacket,
        result: ThoughtResult,
        threshold: float,
        allow_recovery: bool = True,
    ) -> tuple[ThoughtResult, GroundingDiagnostics | None]:
        diagnostics = self._grounding_diagnostics(
            kind=kind,
            target_text=target_text,
            response_text=result.thought,
            packet=packet,
        )
        if diagnostics.grounded_evidence_count <= 0:
            return result, diagnostics
        if diagnostics.alignment_score >= threshold or not allow_recovery:
            return result, diagnostics

        fallback_text = self._build_grounded_fallback(target_text=target_text, packet=packet)
        if not fallback_text:
            return result, diagnostics

        recovered = ThoughtResult(
            raw_text=result.raw_text,
            thought=fallback_text,
            topics=result.topics,
            emotional_valence=result.emotional_valence,
            confidence=max(result.confidence, 0.72 if kind == "query" else 0.68),
            action_intent=result.action_intent,
            latency_ms=result.latency_ms,
            parse_success=result.parse_success,
        )
        recovered_diagnostics = self._grounding_diagnostics(
            kind=kind,
            target_text=target_text,
            response_text=recovered.thought,
            packet=packet,
            fallback_used=True,
        )
        return recovered, recovered_diagnostics

    def _deliberate_external_query(
        self,
        query_text: str,
        depth: ThoughtDepth,
    ) -> tuple[ThoughtResult, GroundingDiagnostics | None]:
        """Answer an operator query directly."""
        packet = self.gate.assemble(external_query=query_text)
        if depth == ThoughtDepth.STANDARD:
            packet.max_response_tokens = max(packet.max_response_tokens, 220)
        elif depth == ThoughtDepth.DEEP:
            packet.max_response_tokens = max(packet.max_response_tokens, 280)
        old_temp = self._set_temperature()
        result = self.cortex.generate(packet)
        self._restore_temperature(old_temp)
        result, diagnostics = self._stabilize_grounded_result(
            kind="query",
            target_text=query_text,
            packet=packet,
            result=result,
            threshold=0.40,
            allow_recovery=True,
        )
        self.working_memory.update_from_thought(
            result.thought, result.topics, result.confidence,
            result.emotional_valence, WMItemType.INSIGHT,
        )
        return result, diagnostics

    def _deliberate_quick(self) -> ThoughtResult:
        """System 1: single LLM call — fast pattern matching."""
        packet = self.gate.assemble()
        consumed_exploration = bool(self._exploration_state.target and packet.forced_topic == self._exploration_state.target)
        old_temp = self._set_temperature()
        result = self.cortex.generate(packet)
        self._restore_temperature(old_temp)
        if consumed_exploration:
            self._clear_active_exploration_target()
        result, diagnostics = self._stabilize_grounded_result(
            kind="wakeful",
            target_text=packet.forced_topic,
            packet=packet,
            result=result,
            threshold=0.40,
            allow_recovery=True,
        )
        self._last_deliberation_grounding = (
            diagnostics if diagnostics is not None and diagnostics.grounded_evidence_count > 0 else None
        )

        # Update working memory with the observation
        self.working_memory.update_from_thought(
            result.thought, result.topics, result.confidence,
            result.emotional_valence, WMItemType.OBSERVATION,
        )
        return result

    def _deliberate_standard(self) -> ThoughtResult:
        """System 2 lite: observe → question (2 calls)."""
        old_temp = self._set_temperature()

        # Phase 1: OBSERVE
        observe_packet = self.gate.assemble(phase="observe")
        consumed_exploration = bool(self._exploration_state.target and observe_packet.forced_topic == self._exploration_state.target)
        observation = self.cortex.generate(observe_packet)
        if consumed_exploration:
            self._clear_active_exploration_target()
        observation, observation_grounding = self._stabilize_grounded_result(
            kind="wakeful",
            target_text=observe_packet.forced_topic,
            packet=observe_packet,
            result=observation,
            threshold=0.40,
            allow_recovery=True,
        )
        self.working_memory.update_from_thought(
            observation.thought, observation.topics, observation.confidence,
            observation.emotional_valence, WMItemType.OBSERVATION,
        )

        # Phase 2: QUESTION (sees observation in working memory)
        question_packet = self.gate.assemble(phase="question")
        question = self.cortex.generate(question_packet)
        self.working_memory.update_from_thought(
            question.thought, question.topics, question.confidence,
            question.emotional_valence, WMItemType.QUESTION,
        )

        self._restore_temperature(old_temp)

        final_result = self._merge_chain_results([observation, question])
        _, final_grounding = self._stabilize_grounded_result(
            kind="wakeful",
            target_text=observe_packet.forced_topic,
            packet=observe_packet,
            result=final_result,
            threshold=0.25,
            allow_recovery=False,
        )
        chosen_grounding = final_grounding or observation_grounding
        self._last_deliberation_grounding = (
            chosen_grounding if chosen_grounding is not None and chosen_grounding.grounded_evidence_count > 0 else None
        )
        return final_result

    def _deliberate_deep(self) -> ThoughtResult:
        """System 2 full: observe → question → reason → synthesize (4 calls)."""
        old_temp = self._set_temperature()
        chain: list[ThoughtResult] = []
        observe_packet: ContextPacket | None = None
        observe_grounding: GroundingDiagnostics | None = None

        phases = [
            ("observe", WMItemType.OBSERVATION),
            ("question", WMItemType.QUESTION),
            ("reason", WMItemType.HYPOTHESIS),
            ("synthesize", WMItemType.INSIGHT),
        ]

        for phase_name, wm_type in phases:
            try:
                packet = self.gate.assemble(phase=phase_name)
                consumed_exploration = bool(
                    phase_name == "observe"
                    and self._exploration_state.target
                    and packet.forced_topic == self._exploration_state.target
                )
                result = self.cortex.generate(packet)
                if phase_name == "observe":
                    observe_packet = packet
                    result, observe_grounding = self._stabilize_grounded_result(
                        kind="wakeful",
                        target_text=packet.forced_topic,
                        packet=packet,
                        result=result,
                        threshold=0.40,
                        allow_recovery=True,
                    )
                if consumed_exploration:
                    self._clear_active_exploration_target()
                chain.append(result)
                self.working_memory.update_from_thought(
                    result.thought, result.topics, result.confidence,
                    result.emotional_valence, wm_type,
                )
            except Exception:
                logger.warning("Deliberation chain interrupted at phase '%s'", phase_name)
                break

        self._restore_temperature(old_temp)

        if not chain:
            # All phases failed — fall back to quick
            return self._deliberate_quick()

        final_result = self._merge_chain_results(chain)
        if observe_packet is not None:
            _, final_grounding = self._stabilize_grounded_result(
                kind="wakeful",
                target_text=observe_packet.forced_topic,
                packet=observe_packet,
                result=final_result,
                threshold=0.25,
                allow_recovery=False,
            )
        else:
            final_grounding = None
        chosen_grounding = final_grounding or observe_grounding
        self._last_deliberation_grounding = (
            chosen_grounding if chosen_grounding is not None and chosen_grounding.grounded_evidence_count > 0 else None
        )
        return final_result

    @staticmethod
    def _text_keywords(text: str) -> set[str]:
        stopwords = {
            "the", "and", "with", "that", "this", "from", "into", "about", "than",
            "their", "there", "because", "while", "where", "which", "have", "has",
            "when", "then", "they", "them", "what", "how", "why", "does", "under",
            "through", "between", "could", "would", "should", "these", "those",
        }
        words: set[str] = set()
        for raw in re.findall(r"[a-zA-Z][a-zA-Z'-]+", text.lower()):
            word = raw.strip("'")
            if len(word) >= 4 and word not in stopwords:
                words.add(word)
        return words

    @classmethod
    def _text_overlap(cls, left: str, right: str) -> float:
        left_words = cls._text_keywords(left)
        right_words = cls._text_keywords(right)
        if not left_words or not right_words:
            return 0.0
        return len(left_words & right_words) / max(1.0, min(len(left_words), len(right_words)))

    @staticmethod
    def _statementize_question(text: str) -> str:
        cleaned = " ".join(text.split()).strip()
        if not cleaned:
            return ""
        lowered = cleaned.lower()
        if lowered.startswith("open question:") or lowered.startswith("question:"):
            _, _, suffix = cleaned.partition(":")
            cleaned = suffix.strip()
            lowered = cleaned.lower()
        if not cleaned.endswith("?"):
            return cleaned

        stem = cleaned[:-1].strip()
        lower_stem = stem.lower()
        replacements = (
            ("how do ", "how "),
            ("how does ", "how "),
            ("how did ", "how "),
            ("why do ", "why "),
            ("why does ", "why "),
            ("why did ", "why "),
            ("what is ", "what "),
            ("what are ", "what "),
        )
        for prefix, replacement in replacements:
            if lower_stem.startswith(prefix):
                tail = stem[len(prefix):].strip()
                if prefix == "what is ":
                    stem = f"what {tail} is"
                elif prefix == "what are ":
                    stem = f"what {tail} are"
                else:
                    stem = f"{replacement}{tail}"
                lower_stem = stem.lower()
                break

        yes_no_prefixes = ("can ", "could ", "should ", "would ", "is ", "are ", "do ", "does ", "did ", "was ", "were ")
        if any(lower_stem.startswith(prefix) for prefix in yes_no_prefixes):
            _, _, remainder = stem.partition(" ")
            stem = f"whether {remainder.strip()}"
        elif stem and stem.split()[0] in {"What", "How", "Why", "When", "Where", "Which"}:
            stem = stem[0].lower() + stem[1:]

        return f"A key open question is {stem}."

    @classmethod
    def _dedupe_sentences(cls, text: str) -> str:
        cleaned = " ".join(text.split()).strip()
        if not cleaned:
            return ""
        parts = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", cleaned) if segment.strip()]
        if len(parts) <= 1:
            return cleaned

        kept: list[str] = []
        for part in parts:
            matched = False
            for idx, existing in enumerate(kept):
                if cls._text_overlap(part, existing) >= 0.78:
                    if len(part) > len(existing):
                        kept[idx] = part
                    matched = True
                    break
            if not matched:
                kept.append(part)
        return " ".join(kept).strip()

    @classmethod
    def _merge_chain_results(cls, chain: list[ThoughtResult]) -> ThoughtResult:
        """Merge a chain of thoughts into a single result.

        The last step remains primary, but short chains are merged more carefully
        so observation+question outputs stay readable and do not echo themselves.
        """
        if len(chain) == 1:
            return chain[0]

        final = chain[-1]
        all_topics: list[str] = []
        seen: set[str] = set()
        for r in chain:
            for t in r.topics:
                key = t.lower().strip()
                if key and key not in seen:
                    all_topics.append(t)
                    seen.add(key)

        total_latency = sum(r.latency_ms for r in chain)

        if len(chain) >= 3:
            thought = cls._dedupe_sentences(cls._statementize_question(final.thought))
        else:
            observation = cls._dedupe_sentences(chain[0].thought)
            follow_up = cls._dedupe_sentences(cls._statementize_question(final.thought))
            if not follow_up:
                thought = observation
            elif cls._text_overlap(observation, follow_up) >= 0.78:
                thought = follow_up if len(follow_up) >= len(observation) else observation
            else:
                thought = cls._dedupe_sentences(f"{observation} {follow_up}")
            if len(thought) > 300:
                clipped = thought[:300].rstrip()
                thought = clipped.rsplit(" ", 1)[0].rstrip(" ,;:") + "..."

        return ThoughtResult(
            raw_text=final.raw_text,
            thought=thought,
            topics=tuple(all_topics[:8]),
            emotional_valence=final.emotional_valence,
            confidence=final.confidence,
            action_intent=final.action_intent,
            latency_ms=total_latency,
            parse_success=final.parse_success,
        )

    def _post_process_thought(
        self,
        result: ThoughtResult,
        depth: ThoughtDepth,
        *,
        grounding: GroundingDiagnostics | None = None,
    ) -> None:
        """Common post-processing after any deliberation depth."""
        now = time.time()

        # Deduplication: if this exact thought was recently generated, skip storing
        # it again. Still advance the pacing clock because the API call already
        # happened and we do not want duplicate generations to create burst loops.
        recent_texts = {h["thought"] for h in self._thought_history}
        if grounding is None and result.thought in recent_texts:
            self._last_thought_time = now
            logger.debug("Duplicate thought suppressed: %s", result.thought[:60])
            return

        self._last_thought_time = now
        self.stats.thoughts_generated += 1
        self.stats.total_inference_ms += result.latency_ms
        self.stats.last_thought = result.thought
        self.stats.last_thought_time = self._last_thought_time
        self._depth_counts[depth.value] = self._depth_counts.get(depth.value, 0) + 1
        if depth is not ThoughtDepth.QUICK:
            self._last_nonquick_time = now
        if depth is ThoughtDepth.DEEP:
            self._last_deep_time = now

        # Update persistent narrative identity
        self.narrative_self.observe_thought(result, depth=depth)

        # Update drives from thought
        self.drives.update_from_thought(result)

        # Cortex→SNN feedback
        feedback = self.gate.emit_cortex_feedback(result)

        if self._curiosity_controller is not None:
            for label, amount in feedback.get("topic_boosts", []):
                try:
                    self._curiosity_controller.boost_concept(label, amount=amount)
                except Exception:
                    pass

        # Pick the next thing to actively explore from uncertainty/tension.
        self._update_active_exploration_target(result)

        inferred_metadata = None
        if grounding is not None:
            inferred_metadata = {
                "external_query": grounding.kind == "query",
                "grounding": grounding.to_dict(),
            }
            self.stats.update_grounding_metrics(grounding)

        # Store as episodic memory (inferred)
        self.memory.store(
            content=result.thought,
            provenance=Provenance.INFERRED,
            topics=list(result.topics),
            emotional_valence=result.emotional_valence,
            confidence=result.confidence,
            salience=max(0.3, abs(result.emotional_valence) + 0.2 * result.confidence),
            metadata=inferred_metadata,
        )

        # Update memory stats
        self.stats.memory_count = self.memory.size
        self.stats.memory_fill_ratio = self.memory.size / max(1, self.memory.capacity)

        logger.debug(
            "Thought #%d [%s] (%.0fms): %s",
            self.stats.thoughts_generated,
            depth.value,
            result.latency_ms,
            result.thought[:80],
        )

        history_item = {
            "thought": result.thought,
            "confidence": result.confidence,
            "emotional_valence": result.emotional_valence,
            "topics": list(result.topics),
            "action_intent": result.action_intent,
            "latency_ms": round(result.latency_ms, 1),
            "time": self._last_thought_time,
            "depth": depth.value,
        }
        if grounding is not None:
            history_item["grounding"] = grounding.to_dict()

        # Append to history (deque.append is CPython-atomic)
        self._thought_history.append(history_item)

        # Update thought quality metrics
        snn_labels = self.gate._snn_concept_labels if hasattr(self.gate, '_snn_concept_labels') else None
        self.stats.update_quality_metrics(result, snn_concepts=snn_labels)

    @staticmethod
    def _episode_topics(ep: Episode) -> set[str]:
        topics = {str(t).lower().strip() for t in ep.topics if str(t).strip()}
        if topics:
            return topics
        words = [w.lower().strip(".,;:!?\"'()[]{}") for w in ep.content.split() if len(w) >= 4]
        return set(words[:4])

    def _dream_group_score(self, group: Sequence[Episode]) -> float:
        if not group:
            return 0.0
        salience = sum(ep.salience for ep in group) / len(group)
        trust = sum(ep.provenance.trust_weight for ep in group) / len(group)
        topic_union: set[str] = set()
        for ep in group:
            topic_union.update(self._episode_topics(ep))
        topic_diversity = min(1.0, len(topic_union) / max(1.0, len(group) * 3.0))
        provenance_diversity = len({ep.provenance for ep in group}) / max(1.0, len(group))

        embedding_diversity = 0.0
        if len(group) >= 2 and all(ep.embedding is not None for ep in group):
            sims: list[float] = []
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    sims.append(float(np.dot(group[i].embedding, group[j].embedding)))
            if sims:
                embedding_diversity = max(0.0, 1.0 - (sum(sims) / len(sims)))

        replay_penalty = sum(ep.replay_count for ep in group) / max(1.0, len(group))
        return (
            0.35 * salience
            + 0.20 * trust
            + 0.20 * topic_diversity
            + 0.20 * embedding_diversity
            + 0.05 * provenance_diversity
            - 0.03 * replay_penalty
        )

    def _select_dream_groups(self, episodes: Sequence[Episode]) -> list[list[Episode]]:
        """Select diverse memory pairs/triples for compositional dreaming."""
        from collections import defaultdict as _defaultdict
        from itertools import combinations

        candidates = list(episodes[: min(len(episodes), 7)])
        if not candidates:
            return []

        scored: list[tuple[float, tuple[Episode, ...]]] = []
        if len(candidates) >= 3:
            for group in combinations(candidates[:6], 3):
                scored.append((self._dream_group_score(group) + 0.05, group))
        if len(candidates) >= 2:
            for group in combinations(candidates, 2):
                scored.append((self._dream_group_score(group), group))
        if not scored:
            return [[candidates[0]]]

        scored.sort(key=lambda item: item[0], reverse=True)
        usage: dict[str, int] = _defaultdict(int)
        selected: list[list[Episode]] = []
        seen: set[tuple[str, ...]] = set()

        for _base_score, group in scored:
            signature = tuple(sorted(ep.episode_id for ep in group))
            if signature in seen:
                continue
            if any(usage[ep.episode_id] >= 2 for ep in group):
                continue
            selected.append(list(group))
            seen.add(signature)
            for ep in group:
                usage[ep.episode_id] += 1
            if len(selected) >= self.sleep_dream_count:
                break

        return selected or [list(scored[0][1])]

    def _dream_memory_support_score(
        self,
        hypothesis: ThoughtResult | None,
        memories: Sequence[Episode],
    ) -> float:
        if hypothesis is None or not memories:
            return 0.0

        hypothesis_words: set[str] = set()
        for topic in hypothesis.topics:
            hypothesis_words.update(self._text_keywords(str(topic)))
        hypothesis_words.update(self._text_keywords(hypothesis.thought))
        if not hypothesis_words:
            return 0.0

        memory_words: set[str] = set()
        for ep in memories:
            for topic in ep.topics:
                memory_words.update(self._text_keywords(str(topic)))
            memory_words.update(self._text_keywords(ep.content))

        overlap = len(hypothesis_words & memory_words)
        return overlap / max(1.0, min(6.0, float(len(hypothesis_words))))

    def _dream_validation_verdict(
        self,
        result: ThoughtResult,
        *,
        hypothesis: ThoughtResult | None = None,
        memories: Sequence[Episode] = (),
    ) -> str:
        """Interpret dream validation output as supported/unresolved/contradicted.

        The model is asked to prefix the validation thought with SUPPORTED / 
        UNRESOLVED / CONTRADICTED, but the parser also uses confidence and a
        lightweight lexical support score against the evidence memories to avoid
        overly brittle verdicts.
        """
        text = result.thought.lower().strip()
        support_score = self._dream_memory_support_score(hypothesis, memories)

        if text.startswith("supported:"):
            return "supported" if (result.confidence >= 0.45 or support_score >= 0.35) else "unresolved"
        if text.startswith("contradicted:"):
            return "contradicted" if result.confidence <= 0.6 else "unresolved"
        if text.startswith("unresolved:"):
            return "unresolved"

        contradiction_markers = (
            "contradict",
            "unsupported",
            "unlikely",
            "inconsistent",
            "no clear connection",
            "does not fit",
            "doesn't fit",
            "weak",
            "not supported",
            "missing mechanism",
        )
        support_markers = (
            "support",
            "supports",
            "supported",
            "consistent",
            "fits",
            "aligns",
            "shared mechanism",
            "same mechanism",
            "strongest evidence",
            "evidence",
        )
        unresolved_markers = (
            "unclear",
            "partial",
            "plausible",
            "possible",
            "not enough evidence",
            "insufficient",
        )

        has_contradiction = any(marker in text for marker in contradiction_markers)
        has_support = any(marker in text for marker in support_markers)
        has_uncertainty = any(marker in text for marker in unresolved_markers)

        if has_contradiction and (result.confidence <= 0.55 or support_score < 0.35):
            return "contradicted"
        if result.confidence <= 0.30 and support_score < 0.40:
            return "contradicted"
        if result.confidence >= 0.65 and not has_contradiction:
            return "supported"
        if result.confidence >= 0.55 and has_support and support_score >= 0.35:
            return "supported"
        if has_uncertainty or result.confidence < 0.55:
            return "unresolved"
        return "unresolved"

    def _sleep_cycle(self, trigger: dict[str, Any] | None = None) -> list[ThoughtResult]:
        """Run a sleep/dream cycle — compositional replay and hypothesis testing."""
        logger.info("Entering sleep cycle")
        self._last_sleep_time = time.time()
        self.stats.sleep_cycles += 1
        dreams: list[ThoughtResult] = []
        trigger_snapshot = None if trigger is None else deepcopy(trigger)
        fatigue_before = float(self.drives.state.fatigue)

        old_temp = getattr(self.cortex, "temperature", None)
        if old_temp is not None:
            self.cortex.temperature = 0.0

        try:
            # Mark replayed episodes and build diverse memory groups.
            replay_episodes = self.memory.recall_for_sleep(top_k=max(10, self.sleep_dream_count * 3))
            for ep in replay_episodes:
                ep.replay_count += 1
            dream_groups = self._select_dream_groups(replay_episodes)

            if not dream_groups:
                # No memory material yet — fall back to generic dream generation.
                for _ in range(self.sleep_dream_count):
                    result = self.cortex.generate(self.gate.assemble_for_sleep())
                    self.stats.dreams_generated += 1
                    self.stats.total_inference_ms += result.latency_ms
                    self.memory.store(
                        content=result.thought,
                        provenance=Provenance.DREAMED,
                        topics=list(result.topics),
                        emotional_valence=result.emotional_valence,
                        confidence=result.confidence * 0.5,
                        salience=0.35,
                    )
                    dreams.append(result)
            else:
                for group in dream_groups[: self.sleep_dream_count]:
                    compose_packet = self.gate.assemble_for_sleep(group, phase="dream_compose")
                    hypothesis = self.cortex.generate(compose_packet)
                    self.stats.dreams_generated += 1
                    self.stats.total_inference_ms += hypothesis.latency_ms

                    source_ids = ",".join(ep.episode_id for ep in group)
                    dream_episode = self.memory.store(
                        content=hypothesis.thought,
                        provenance=Provenance.DREAMED,
                        topics=list(hypothesis.topics),
                        emotional_valence=hypothesis.emotional_valence,
                        confidence=max(0.2, hypothesis.confidence * 0.6),
                        salience=min(0.9, 0.35 + 0.2 * len(group)),
                        source_thought_id=source_ids,
                    )
                    dreams.append(hypothesis)

                    # Validate the hypothesis against the broader replay pool.
                    validation_memories = list(group)
                    group_ids = {g.episode_id for g in group}
                    for ep in replay_episodes:
                        if ep.episode_id not in group_ids:
                            validation_memories.append(ep)
                        if len(validation_memories) >= self.gate.max_memories:
                            break

                    validation = self.cortex.generate(
                        self.gate.assemble_for_sleep(
                            validation_memories,
                            phase="dream_test",
                            hypothesis=hypothesis.thought,
                        )
                    )
                    self.stats.total_inference_ms += validation.latency_ms
                    verdict = self._dream_validation_verdict(
                        validation,
                        hypothesis=hypothesis,
                        memories=validation_memories,
                    )

                    if verdict == "supported":
                        self.memory.graduate_hypothesis(dream_episode.episode_id)
                    elif verdict == "contradicted":
                        self.memory.contradict_episode(dream_episode.episode_id)
                        self._queue_wake_tension(
                            f"Dream contradiction: {validation.thought}",
                            topics=validation.topics or hypothesis.topics,
                            salience=0.8,
                        )
                    else:
                        self._queue_wake_tension(
                            f"Unresolved dream hypothesis: {hypothesis.thought}",
                            topics=hypothesis.topics,
                            salience=0.6,
                        )

            # Reset fatigue after sleep
            self.drives.state.fatigue = max(0.0, self.drives.state.fatigue - 0.5)

            # Update dream verification rate using all dream-origin episodes.
            dream_lineage = self.memory.recall_dream_lineage()
            self.stats._dreams_total = len(dream_lineage)
            self.stats._dreams_verified = sum(1 for ep in dream_lineage if ep.provenance == Provenance.VERIFIED)
            if self.stats._dreams_total > 0:
                self.stats.dream_verification_rate = self.stats._dreams_verified / self.stats._dreams_total
            else:
                self.stats.dream_verification_rate = 0.0
        finally:
            if old_temp is not None:
                self.cortex.temperature = old_temp

        with self._lock:
            if trigger_snapshot is not None:
                self._requested_sleep_cycles += 1
            self._last_sleep_cycle_summary = {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "requested": trigger_snapshot is not None,
                "trigger": "requested" if trigger_snapshot is not None else "passive",
                "request": self._sleep_request_snapshot_locked(trigger_snapshot),
                "dreams_generated": int(len(dreams)),
                "sleep_cycles": int(self.stats.sleep_cycles),
                "fatigue_before": float(fatigue_before),
                "fatigue_after": float(self.drives.state.fatigue),
            }

        logger.info("Sleep cycle complete: %d dreams generated", len(dreams))
        return dreams
