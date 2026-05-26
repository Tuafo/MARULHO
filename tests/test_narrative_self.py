"""Tests for Narrative Self persistence and Phase-2 depth tuning.

Covers:
  1. NarrativeSelf autobiographical summary generation
  2. NarrativeSelf persistence across sessions
  3. ThalamicGate narrative injection
  4. Retired ThoughtLoop depth tuning behavior (skipped)
"""

from __future__ import annotations

import time

import pytest

from hecsn.cortex.core import MockCortex, ThoughtDepth, ThoughtResult
from hecsn.cortex.drives import DriveSystem, ThalamicGate
from hecsn.cortex.episodic_memory import EpisodicMemory
from hecsn.cortex.narrative_self import NarrativeSelf
from hecsn.cortex.thought_loop import ThoughtLoop
from hecsn.cortex.working_memory import WorkingMemory, WorkingMemoryItem, WMItemType


class TestNarrativeSelf:
    def test_builds_summary_from_topics_questions_and_surprises(self):
        narrative = NarrativeSelf(refresh_interval=1)
        narrative.observe_thought(
            ThoughtResult(
                raw_text="",
                thought="How do coral reefs survive thermal stress?",
                topics=("coral reefs", "thermal stress"),
                confidence=0.7,
            ),
            depth=ThoughtDepth.STANDARD,
        )
        narrative.observe_thought(
            ThoughtResult(
                raw_text="",
                thought="However, some reef systems recover faster than expected.",
                topics=("coral reefs", "recovery"),
                confidence=0.4,
            ),
            depth=ThoughtDepth.DEEP,
        )
        summary = narrative.to_prompt()
        assert "coral reefs" in summary
        assert "open question" in summary.lower() or "question" in summary.lower()
        assert "unresolved" in summary.lower() or "however" in summary.lower()

    def test_persistence_round_trip(self, tmp_path):
        path = tmp_path / "narrative.json"
        narrative = NarrativeSelf(persistence_path=path, refresh_interval=1)
        narrative.observe_thought(
            ThoughtResult(
                raw_text="",
                thought="Bird migration depends on magnetic field cues.",
                topics=("bird migration", "magnetic fields"),
                confidence=0.8,
            ),
            depth=ThoughtDepth.DEEP,
        )
        narrative.save()

        restored = NarrativeSelf(persistence_path=path)
        snap = restored.snapshot()
        assert snap["thought_count"] >= 1
        assert any("bird migration" in interest for interest in snap["interests"])
        assert "Bird migration depends" in " ".join(snap["recent_insights"])

    def test_duplicate_questions_stay_deduplicated(self):
        narrative = NarrativeSelf(refresh_interval=1)
        question = "Why do whales migrate across entire ocean basins?"
        result = ThoughtResult(raw_text="", thought=question, topics=("whale migration",), confidence=0.6)
        narrative.observe_thought(result, depth=ThoughtDepth.STANDARD)
        narrative.observe_thought(result, depth=ThoughtDepth.STANDARD)
        assert narrative.open_questions.count(question) == 1

    def test_question_and_statement_are_extracted_separately(self):
        narrative = NarrativeSelf(refresh_interval=1)
        text = (
            "Bioluminescence is produced by luciferin reacting with oxygen. "
            "What is the connection between this chemistry and whale migration?"
        )
        narrative.observe_thought(
            ThoughtResult(raw_text="", thought=text, topics=("bioluminescence", "whales"), confidence=0.8),
            depth=ThoughtDepth.STANDARD,
        )
        assert narrative.recent_insights[0].startswith("Bioluminescence is produced")
        assert narrative.open_questions[0].startswith("What is the connection")


class TestNarrativeGateIntegration:
    def test_gate_includes_narrative_for_external_query(self):
        mem = EpisodicMemory(capacity=100)
        drives = DriveSystem()
        gate = ThalamicGate(mem, drives)
        narrative = NarrativeSelf(refresh_interval=1)
        narrative.summary = "I've recently been exploring coral reefs and bleaching recovery."
        gate.narrative_self = narrative

        gate.submit_query("What matters about coral bleaching?")
        packet = gate.assemble()
        assert "coral reefs" in packet.narrative_self
        assert packet.external_query == "What matters about coral bleaching?"

    def test_gate_omits_narrative_for_chain_phase(self):
        mem = EpisodicMemory(capacity=100)
        drives = DriveSystem()
        gate = ThalamicGate(mem, drives)
        gate.working_memory = WorkingMemory()
        gate.working_memory.add(WorkingMemoryItem(
            content="observing volcanic ash and lightning",
            item_type=WMItemType.OBSERVATION,
            topic="volcanic eruptions",
        ))
        narrative = NarrativeSelf(refresh_interval=1)
        narrative.summary = "I've recently been exploring volcanism and atmospheric electricity."
        gate.narrative_self = narrative

        packet = gate.assemble(phase="question")
        assert packet.narrative_self == ""
        assert "volcanic" in packet.working_memory_narrative


@pytest.mark.skip(reason="ThoughtLoop runtime path is retired; narrative primitives stay tested without the old loop")
class TestPhase2DepthTuning:
    def _make_loop(self, signal_payload=None, *, narrative_state_path=None):
        signal_payload = signal_payload or {}
        return ThoughtLoop(
            cortex=MockCortex(latency_ms=5.0),
            min_thought_interval_s=0.0,
            signal_provider=lambda: signal_payload,
            narrative_state_path=narrative_state_path,
        )

    def test_high_prediction_error_triggers_deep(self):
        loop = self._make_loop({
            "prediction_error_mean": 0.40,
            "prediction_error_max": 0.70,
            "predictive_confidence_mean": 0.45,
            "predictive_confidence_min": 0.35,
        })
        assert loop._choose_depth() == ThoughtDepth.DEEP
        assert loop.snapshot()["depth_policy"]["last_reason"] == "high_prediction_error"

    def test_low_predictive_confidence_triggers_standard(self):
        loop = self._make_loop({
            "prediction_error_mean": 0.12,
            "prediction_error_max": 0.20,
            "predictive_confidence_mean": 0.42,
            "predictive_confidence_min": 0.22,
        })
        assert loop._choose_depth() == ThoughtDepth.STANDARD
        assert loop.snapshot()["depth_policy"]["last_reason"] == "low_predictive_confidence"

    def test_external_query_triggers_standard(self):
        loop = self._make_loop({
            "prediction_error_mean": 0.05,
            "prediction_error_max": 0.10,
            "predictive_confidence_mean": 0.8,
            "predictive_confidence_min": 0.7,
        })
        loop.submit_query("How do volcanoes trigger lightning?")
        assert loop._choose_depth() == ThoughtDepth.STANDARD
        assert loop.snapshot()["depth_policy"]["last_reason"] == "external_query"

    def test_deep_cooldown_prevents_back_to_back_deep_chains(self):
        loop = self._make_loop({
            "prediction_error_mean": 0.25,
            "prediction_error_max": 0.60,
            "predictive_confidence_mean": 0.5,
            "predictive_confidence_min": 0.45,
        })
        loop._last_deep_time = time.time()
        assert loop._choose_depth() == ThoughtDepth.QUICK
        assert loop.snapshot()["depth_policy"]["last_reason"] == "deep_cooldown"

    def test_snapshot_includes_cognitive_signals_and_narrative(self, tmp_path):
        path = tmp_path / "narrative_state.json"
        loop = self._make_loop({
            "prediction_error_mean": 0.33,
            "prediction_error_max": 0.44,
            "predictive_confidence_mean": 0.55,
            "predictive_confidence_min": 0.41,
            "recent_concepts": ["reef ecology", "ocean acidification"],
        }, narrative_state_path=str(path))
        loop.narrative_self.summary = "I've recently been exploring reef ecology."
        loop._choose_depth()
        snap = loop.snapshot()
        assert snap["cognitive_signals"]["prediction_error_mean"] == 0.33
        assert "reef ecology" in snap["cognitive_signals"]["recent_concepts"]
        assert "narrative_self" in snap
        assert "reef ecology" in snap["narrative_self"]["summary"]

    def test_narrative_persists_across_thought_loop_sessions(self, tmp_path):
        path = tmp_path / "cortex_narrative_self.json"
        loop = ThoughtLoop(
            cortex=MockCortex(responses=[{
                "thought": "Honey bees communicate through the waggle dance.",
                "topics": ["honey bees", "waggle dance"],
                "valence": 0.1,
                "confidence": 0.8,
                "action": None,
            }], latency_ms=5.0),
            min_thought_interval_s=0.0,
            narrative_state_path=str(path),
        )
        loop._post_process_thought(
            ThoughtResult(
                raw_text="",
                thought="Honey bees communicate through the waggle dance.",
                topics=("honey bees", "waggle dance"),
                confidence=0.8,
                latency_ms=5.0,
            ),
            ThoughtDepth.STANDARD,
        )
        loop.narrative_self.save()

        restored = ThoughtLoop(
            cortex=MockCortex(latency_ms=5.0),
            min_thought_interval_s=0.0,
            narrative_state_path=str(path),
        )
        assert "honey bees" in restored.narrative_self.to_prompt().lower()
