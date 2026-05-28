"""Runtime-quality metrics for language-facing brain surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hecsn.semantics.language_result import LanguageResult


@dataclass
class BrainStats:
    """Observable quality statistics for a language-facing runtime surface."""

    readouts_generated: int = 0
    replay_samples_generated: int = 0
    sleep_cycles: int = 0
    ticks: int = 0
    total_inference_ms: float = 0.0
    last_readout: str = ""
    last_readout_time: float = 0.0
    current_mode: str = "idle"
    is_sleeping: bool = False
    memory_count: int = 0
    memory_fill_ratio: float = 0.0

    topic_diversity: float = 0.0
    concreteness_ratio: float = 0.0
    avg_novelty: float = 0.0
    snn_alignment: float = 0.0
    replay_verification_rate: float = 0.0
    grounded_query_alignment: float = 0.0
    grounded_query_recovery_rate: float = 0.0
    grounded_query_count: int = 0
    grounded_wakeful_alignment: float = 0.0
    grounded_wakeful_recovery_rate: float = 0.0
    grounded_wakeful_count: int = 0
    _topic_counts: dict = field(default_factory=dict)
    _total_topics: int = 0
    _concrete_count: int = 0
    _snn_aligned_count: int = 0
    _replay_samples_verified: int = 0
    _replay_samples_total: int = 0
    _grounded_query_alignment_total: float = 0.0
    _grounded_query_recoveries: int = 0
    _grounded_wakeful_alignment_total: float = 0.0
    _grounded_wakeful_recoveries: int = 0

    @property
    def avg_inference_ms(self) -> float:
        total = self.readouts_generated + self.replay_samples_generated
        if total == 0:
            return 0.0
        return self.total_inference_ms / total

    def update_quality_metrics(
        self,
        result: LanguageResult,
        snn_concepts: list[str] | None = None,
    ) -> None:
        """Update quality metrics from a language-facing result."""
        import math as _math

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

        if result.confidence > 0.5 and len(result.topics) > 0:
            self._concrete_count += 1
        total_readouts = max(1, self.readouts_generated)
        self.concreteness_ratio = self._concrete_count / total_readouts

        if result.topics and snn_concepts:
            snn_lower = {concept.lower() for concept in snn_concepts}
            aligned = sum(
                1
                for topic in result.topics
                if any(topic.lower() in concept or concept in topic.lower() for concept in snn_lower)
            )
            self._snn_aligned_count += aligned
            total_topic_instances = max(1, self._total_topics)
            self.snn_alignment = self._snn_aligned_count / total_topic_instances

    def update_grounding_metrics(self, diagnostics: Any | None) -> None:
        """Update grounding metrics from a grounded query or wakeful result."""
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
