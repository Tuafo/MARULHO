"""Tests for canonical runtime-quality metrics."""

from __future__ import annotations

from types import SimpleNamespace

from hecsn.cortex.core import ThoughtResult
from hecsn.semantics.brain_stats import BrainStats


def test_brain_stats_average_inference_counts_thoughts_and_dreams() -> None:
    stats = BrainStats(thoughts_generated=2, dreams_generated=1, total_inference_ms=120.0)
    assert stats.avg_inference_ms == 40.0


def test_brain_stats_updates_quality_alignment_from_topics() -> None:
    stats = BrainStats(thoughts_generated=2)
    result = ThoughtResult(
        raw_text="",
        thought="Reef chemistry changes under thermal stress.",
        topics=("reef chemistry", "thermal stress"),
        confidence=0.8,
    )

    stats.update_quality_metrics(result, snn_concepts=["reef chemistry", "carbonate balance"])

    assert stats.topic_diversity > 0.0
    assert stats.concreteness_ratio == 0.5
    assert stats.snn_alignment == 0.5


def test_brain_stats_updates_grounding_metrics_without_thought_loop_types() -> None:
    stats = BrainStats()

    stats.update_grounding_metrics(
        SimpleNamespace(
            kind="query",
            grounded_evidence_count=2,
            alignment_score=0.75,
            fallback_used=True,
        )
    )
    stats.update_grounding_metrics(
        SimpleNamespace(
            kind="wakeful",
            grounded_evidence_count=1,
            alignment_score=0.5,
            fallback_used=False,
        )
    )

    assert stats.grounded_query_count == 1
    assert stats.grounded_query_alignment == 0.75
    assert stats.grounded_query_recovery_rate == 1.0
    assert stats.grounded_wakeful_count == 1
    assert stats.grounded_wakeful_alignment == 0.5
    assert stats.grounded_wakeful_recovery_rate == 0.0
