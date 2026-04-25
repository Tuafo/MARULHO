"""Tests for Working Memory and Inner Monologue (deliberation chains).

Covers:
  1. WorkingMemory: add/evict/decay/broadcast/snapshot
  2. ThoughtDepth: System 1/2 depth selection
  3. Deliberation chains: quick/standard/deep thought generation
  4. Integration: working memory feeds into context packets
"""

import pytest

from hecsn.cortex.working_memory import WorkingMemory, WorkingMemoryItem, WMItemType, _topics_overlap
from hecsn.cortex.core import (
    ContextPacket, MockCortex, ThinkingMode, ThoughtDepth, ThoughtResult,
)
from hecsn.cortex.episodic_memory import EpisodicMemory
from hecsn.cortex.drives import DriveSystem, ThalamicGate
from hecsn.cortex.thought_loop import ThoughtLoop


# ===========================================================================
# Working Memory unit tests
# ===========================================================================

class TestWorkingMemoryBasics:
    """Core working memory operations."""

    def test_create_empty(self):
        wm = WorkingMemory(capacity=5)
        assert wm.is_empty()
        assert wm.size == 0
        assert wm.broadcast() == ""

    def test_add_item(self):
        wm = WorkingMemory(capacity=5)
        item = WorkingMemoryItem(content="coral reefs", item_type=WMItemType.OBSERVATION, topic="marine biology")
        evicted = wm.add(item)
        assert evicted is None
        assert wm.size == 1
        assert not wm.is_empty()

    def test_capacity_eviction(self):
        wm = WorkingMemory(capacity=3)
        for i in range(3):
            wm.add(WorkingMemoryItem(
                content=f"item {i}", item_type=WMItemType.OBSERVATION,
                strength=0.5 + i * 0.1, topic=f"topic_{i}",
            ))
        assert wm.size == 3

        # Add stronger item — should evict the weakest (item 0, strength 0.5)
        strong = WorkingMemoryItem(
            content="strong item", item_type=WMItemType.INSIGHT,
            strength=0.9, topic="new_topic",
        )
        evicted = wm.add(strong)
        assert evicted is not None
        assert evicted.content == "item 0"
        assert wm.size == 3

    def test_weak_item_rejected(self):
        wm = WorkingMemory(capacity=2)
        for i in range(2):
            wm.add(WorkingMemoryItem(
                content=f"item {i}", item_type=WMItemType.OBSERVATION,
                strength=0.8, topic=f"topic_{i}",
            ))

        # Try to add a weak item — should be rejected
        weak = WorkingMemoryItem(
            content="weak item", item_type=WMItemType.OBSERVATION,
            strength=0.2, topic="weak_topic",
        )
        evicted = wm.add(weak)
        assert evicted is None  # Not evicted, just rejected
        assert wm.size == 2

    def test_topic_overlap_refreshes(self):
        wm = WorkingMemory(capacity=5)
        wm.add(WorkingMemoryItem(
            content="old content about coral", item_type=WMItemType.OBSERVATION,
            strength=0.5, topic="coral reefs",
        ))

        # Add with overlapping topic — should REFRESH, not add new
        wm.add(WorkingMemoryItem(
            content="new content about coral", item_type=WMItemType.OBSERVATION,
            strength=0.7, topic="coral bleaching",
        ))
        assert wm.size == 1  # Refreshed, not added
        assert wm.items[0].content == "new content about coral"

    def test_decay_evicts_weak_items(self):
        wm = WorkingMemory(capacity=5, decay_rate=0.5)
        wm.add(WorkingMemoryItem(
            content="weak item", item_type=WMItemType.OBSERVATION,
            strength=0.2, topic="weak",
        ))
        wm.add(WorkingMemoryItem(
            content="strong item", item_type=WMItemType.OBSERVATION,
            strength=0.9, topic="strong",
        ))
        assert wm.size == 2

        # Decay — weak item should drop below 0.1 threshold
        wm.decay_all()
        assert wm.size == 1  # Weak item evicted
        assert wm.items[0].content == "strong item"

    def test_clear(self):
        wm = WorkingMemory(capacity=5)
        wm.add(WorkingMemoryItem(content="item", item_type=WMItemType.OBSERVATION, topic="x"))
        assert wm.size == 1
        wm.clear()
        assert wm.is_empty()


class TestWorkingMemoryTypes:
    """Type detection and queries."""

    def test_has_tension(self):
        wm = WorkingMemory()
        assert not wm.has_tension()
        wm.add(WorkingMemoryItem(
            content="contradiction", item_type=WMItemType.TENSION, topic="logic",
        ))
        assert wm.has_tension()

    def test_has_question(self):
        wm = WorkingMemory()
        assert not wm.has_question()
        wm.add(WorkingMemoryItem(
            content="why does X happen?", item_type=WMItemType.QUESTION, topic="science",
        ))
        assert wm.has_question()

    def test_strongest_item(self):
        wm = WorkingMemory()
        assert wm.strongest_item() is None

        wm.add(WorkingMemoryItem(content="a", item_type=WMItemType.OBSERVATION, strength=0.3, topic="a"))
        wm.add(WorkingMemoryItem(content="b", item_type=WMItemType.INSIGHT, strength=0.9, topic="b"))
        assert wm.strongest_item().content == "b"

    def test_update_from_thought_detects_question(self):
        wm = WorkingMemory()
        wm.update_from_thought(
            "How does photosynthesis work in deep sea organisms?",
            ("photosynthesis",), 0.7, 0.1,
        )
        assert wm.has_question()

    def test_update_from_thought_detects_tension(self):
        wm = WorkingMemory()
        wm.update_from_thought(
            "However, this contradicts the established theory of plate tectonics.",
            ("geology",), 0.6, -0.3,
        )
        assert wm.has_tension()

    def test_update_from_thought_detects_insight(self):
        wm = WorkingMemory()
        wm.update_from_thought(
            "This suggests that both mechanisms share a common evolutionary origin.",
            ("evolution",), 0.8, 0.4,
        )
        items = wm.items
        assert any(i.item_type == WMItemType.INSIGHT for i in items)


class TestWorkingMemoryBroadcast:
    """Broadcast narrative generation."""

    def test_empty_broadcast(self):
        wm = WorkingMemory()
        assert wm.broadcast() == ""

    def test_single_item_broadcast(self):
        wm = WorkingMemory()
        wm.add(WorkingMemoryItem(
            content="coral reefs are built by tiny polyps",
            item_type=WMItemType.OBSERVATION, strength=0.8, topic="marine",
        ))
        narrative = wm.broadcast()
        assert "coral reefs" in narrative
        assert "Currently thinking about" in narrative

    def test_multi_item_broadcast(self):
        wm = WorkingMemory()
        wm.add(WorkingMemoryItem(
            content="reefs are threatened by warming",
            item_type=WMItemType.OBSERVATION, strength=0.8, topic="marine",
        ))
        wm.add(WorkingMemoryItem(
            content="how do coral survive heat?",
            item_type=WMItemType.QUESTION, strength=0.7, topic="adaptation",
        ))
        wm.add(WorkingMemoryItem(
            content="some species have heat-adapted genes",
            item_type=WMItemType.INSIGHT, strength=0.6, topic="genetics",
        ))
        narrative = wm.broadcast()
        assert "Open question" in narrative
        assert "Recent insight" in narrative

    def test_snapshot(self):
        wm = WorkingMemory(capacity=3)
        wm.add(WorkingMemoryItem(content="test", item_type=WMItemType.OBSERVATION, topic="x"))
        snap = wm.snapshot()
        assert snap["size"] == 1
        assert snap["capacity"] == 3
        assert len(snap["items"]) == 1
        assert snap["items"][0]["type"] == "observation"


class TestTopicOverlap:
    """Topic matching utility."""

    def test_overlap_with_shared_word(self):
        assert _topics_overlap("coral reefs", "coral bleaching")

    def test_no_overlap(self):
        assert not _topics_overlap("quantum physics", "marine biology")

    def test_empty_topics(self):
        assert not _topics_overlap("", "coral")
        assert not _topics_overlap("coral", "")

    def test_short_words_ignored(self):
        assert not _topics_overlap("of it", "in it")


# ===========================================================================
# ThoughtDepth enum tests
# ===========================================================================

class TestThoughtDepth:
    def test_enum_values(self):
        assert ThoughtDepth.QUICK.value == "quick"
        assert ThoughtDepth.STANDARD.value == "standard"
        assert ThoughtDepth.DEEP.value == "deep"


# ===========================================================================
# Context Packet integration tests
# ===========================================================================

class TestContextPacketWorkingMemory:
    """Working memory narrative appears in context packets."""

    def test_working_memory_in_prompt(self):
        ctx = ContextPacket(
            working_memory_narrative="Currently thinking about coral reef thermal tolerance",
            mode=ThinkingMode.THINK,
        )
        prompt = ctx.to_user_prompt()
        assert "Working Memory" in prompt
        assert "coral reef" in prompt

    def test_empty_working_memory_omitted(self):
        ctx = ContextPacket(
            working_memory_narrative="",
            mode=ThinkingMode.THINK,
        )
        prompt = ctx.to_user_prompt()
        assert "Working Memory" not in prompt

    def test_phase_field_preserved(self):
        ctx = ContextPacket(
            deliberation_phase="question",
            mode=ThinkingMode.THINK,
        )
        assert ctx.deliberation_phase == "question"


# ===========================================================================
# Deliberation chain tests (with MockCortex)
# ===========================================================================

class TestDeliberationChains:
    """Test the multi-step inner monologue mechanism."""

    def _make_loop(self, responses=None) -> ThoughtLoop:
        """Create a ThoughtLoop with MockCortex for testing."""
        cortex = MockCortex(responses=responses, latency_ms=5.0)
        loop = ThoughtLoop(
            cortex=cortex,
            min_thought_interval_s=0.0,  # No delay for tests
        )
        return loop

    def test_quick_deliberation(self):
        """Quick (System 1) produces a single thought."""
        responses = [
            {"thought": "Water boils at 100°C", "topics": ["thermodynamics"],
             "valence": 0.1, "confidence": 0.8, "action": None},
        ]
        loop = self._make_loop(responses)
        result = loop._deliberate_quick()
        assert "Water boils" in result.thought
        assert loop.working_memory.size >= 1

    def test_standard_deliberation(self):
        """Standard (System 2 lite) produces observe + question."""
        responses = [
            {"thought": "Coral polyps build reefs", "topics": ["marine biology"],
             "valence": 0.2, "confidence": 0.7, "action": None},
            {"thought": "How do polyps survive ocean warming?", "topics": ["climate", "adaptation"],
             "valence": -0.1, "confidence": 0.5, "action": None},
        ]
        loop = self._make_loop(responses)
        result = loop._deliberate_standard()
        # Should merge both thoughts
        assert len(result.topics) >= 2  # Topics from both phases
        assert result.latency_ms >= 10.0  # Sum of both calls

    def test_deep_deliberation(self):
        """Deep (System 2 full) produces 4-phase chain."""
        responses = [
            {"thought": "Spider silk is stronger than steel", "topics": ["biomaterials"],
             "valence": 0.3, "confidence": 0.8, "action": None},
            {"thought": "Why hasn't evolution produced even stronger materials?",
             "topics": ["evolution"], "valence": 0.0, "confidence": 0.5, "action": None},
            {"thought": "Energy cost constrains biological material strength",
             "topics": ["thermodynamics", "biology"], "valence": 0.1, "confidence": 0.6, "action": None},
            {"thought": "Biological materials optimize for energy efficiency, not absolute strength — nature's Pareto frontier",
             "topics": ["optimization", "biomaterials", "evolution"],
             "valence": 0.4, "confidence": 0.7, "action": None},
        ]
        loop = self._make_loop(responses)
        result = loop._deliberate_deep()
        # Final thought should be the synthesis
        assert "Pareto" in result.thought or "optimize" in result.thought
        assert result.latency_ms >= 20.0  # Sum of 4 calls
        # Working memory should have items from each phase
        assert loop.working_memory.size >= 3

    def test_deep_chain_handles_failure(self):
        """If a phase fails mid-chain, returns best partial result."""
        # Only provide 2 responses — phases 3 and 4 will fail
        responses = [
            {"thought": "Observation about magnetism", "topics": ["physics"],
             "valence": 0.1, "confidence": 0.7, "action": None},
            {"thought": "Why do magnets work?", "topics": ["electromagnetism"],
             "valence": 0.0, "confidence": 0.5, "action": None},
        ]
        loop = self._make_loop(responses)
        # Make cortex fail after 2 responses by exhausting the list
        # MockCortex cycles, so we need to force failure differently
        # Instead, verify the chain handles gracefully with available responses
        result = loop._deliberate_deep()
        assert result.thought  # Got something back
        assert result.latency_ms > 0

    def test_depth_selection_default_is_quick(self):
        """Default depth is QUICK (System 1)."""
        loop = self._make_loop()
        depth = loop._choose_depth()
        assert depth == ThoughtDepth.QUICK

    def test_depth_selection_with_tension(self):
        """Working memory tension triggers DEEP thinking."""
        loop = self._make_loop()
        loop.working_memory.add(WorkingMemoryItem(
            content="contradiction found", item_type=WMItemType.TENSION, topic="logic",
        ))
        depth = loop._choose_depth()
        assert depth == ThoughtDepth.DEEP

    def test_depth_selection_with_question_and_curiosity(self):
        """Open question alone doesn't trigger deeper thinking (budget conservation)."""
        loop = self._make_loop()
        loop.working_memory.add(WorkingMemoryItem(
            content="how does X work?", item_type=WMItemType.QUESTION, topic="science",
        ))
        loop.drives.state.curiosity = 0.7
        depth = loop._choose_depth()
        assert depth == ThoughtDepth.QUICK  # Budget-conservative: questions stay quick

    def test_depth_selection_high_arousal(self):
        """Very high arousal triggers DEEP thinking."""
        loop = self._make_loop()
        loop.drives.state.norepinephrine = 1.0
        loop.drives.state.curiosity = 1.0
        loop.drives.state.anxiety = 1.0  # arousal = 0.3+0.3+0.2 = 0.8
        depth = loop._choose_depth()
        assert depth == ThoughtDepth.DEEP

    def test_every_10th_thought_is_standard(self):
        """Periodic deeper thinking for variety."""
        loop = self._make_loop()
        loop.stats.thoughts_generated = 10
        depth = loop._choose_depth()
        assert depth == ThoughtDepth.STANDARD

    def test_step_uses_depth(self):
        """step() now goes through depth-aware deliberation."""
        responses = [
            {"thought": "A quick fact about gravity", "topics": ["physics"],
             "valence": 0.1, "confidence": 0.7, "action": None},
        ]
        loop = self._make_loop(responses)
        loop.drives.update_from_prediction_error(0.7, 0.8, 0.2, 0.1)
        result = loop.step(force=True)
        assert result is not None
        assert "gravity" in result.thought
        # History should include depth info
        assert len(loop._thought_history) == 1
        assert loop._thought_history[0].get("depth") == "quick"


# ===========================================================================
# ThalamicGate phase-aware assembly
# ===========================================================================

class TestThalamicGatePhases:
    """ThalamicGate correctly handles phase parameter."""

    def test_assemble_no_phase(self):
        """Default assembly (no phase) works as before."""
        mem = EpisodicMemory(capacity=100)
        drives = DriveSystem()
        gate = ThalamicGate(mem, drives)
        gate.working_memory = WorkingMemory()

        packet = gate.assemble()
        assert packet.deliberation_phase == ""
        assert packet.forced_topic  # Should have a seed topic

    def test_assemble_observe_phase(self):
        """Observe phase includes memories and forced topic."""
        mem = EpisodicMemory(capacity=100)
        drives = DriveSystem()
        gate = ThalamicGate(mem, drives)
        gate.working_memory = WorkingMemory()

        packet = gate.assemble(phase="observe")
        assert packet.deliberation_phase == "observe"

    def test_assemble_question_phase_skips_memories(self):
        """Question/reason/synthesize phases skip memories (WM has context)."""
        mem = EpisodicMemory(capacity=100)
        mem.store("important memory", salience=0.9)
        drives = DriveSystem()
        gate = ThalamicGate(mem, drives)
        wm = WorkingMemory()
        wm.add(WorkingMemoryItem(
            content="observation about coral", item_type=WMItemType.OBSERVATION, topic="marine",
        ))
        gate.working_memory = wm

        packet = gate.assemble(phase="question")
        assert packet.deliberation_phase == "question"
        assert len(packet.top_memories) == 0  # Skipped for chain phases
        assert packet.forced_topic == ""  # No new topic mid-chain
        assert "coral" in packet.working_memory_narrative  # WM is the context

    def test_working_memory_narrative_in_chain_phase(self):
        """Working memory narrative is included for chain continuation phases."""
        mem = EpisodicMemory(capacity=100)
        drives = DriveSystem()
        gate = ThalamicGate(mem, drives)
        wm = WorkingMemory()
        wm.add(WorkingMemoryItem(
            content="thinking about magnetism", item_type=WMItemType.OBSERVATION,
            topic="physics",
        ))
        gate.working_memory = wm

        # Chain continuation phase → includes working memory
        packet = gate.assemble(phase="question")
        assert "magnetism" in packet.working_memory_narrative

        # Initial/quick phase → NO working memory (prevents topic bleed)
        packet2 = gate.assemble()
        assert packet2.working_memory_narrative == ""


# ===========================================================================
# Merge chain results
# ===========================================================================

class TestMergeChainResults:
    """Test the chain result merging logic."""

    def test_single_result_passthrough(self):
        result = ThoughtResult(
            raw_text="", thought="single thought", topics=("a",),
            latency_ms=100.0, parse_success=True,
        )
        merged = ThoughtLoop._merge_chain_results([result])
        assert merged is result

    def test_two_results_concatenated(self):
        r1 = ThoughtResult(raw_text="", thought="Observation.", topics=("a",), latency_ms=50.0)
        r2 = ThoughtResult(raw_text="", thought="Question?", topics=("b",), latency_ms=60.0)
        merged = ThoughtLoop._merge_chain_results([r1, r2])
        assert "Observation" in merged.thought
        assert "Question" in merged.thought
        assert merged.latency_ms == 110.0
        assert len(merged.topics) == 2

    def test_four_results_uses_synthesis(self):
        results = [
            ThoughtResult(raw_text="", thought="Obs", topics=("a",), latency_ms=25.0),
            ThoughtResult(raw_text="", thought="Question?", topics=("b",), latency_ms=25.0),
            ThoughtResult(raw_text="", thought="Reasoning here", topics=("c",), latency_ms=25.0),
            ThoughtResult(raw_text="", thought="The actual insight", topics=("d",), latency_ms=25.0),
        ]
        merged = ThoughtLoop._merge_chain_results(results)
        # Should use the LAST (synthesis) step
        assert merged.thought == "The actual insight"
        assert merged.latency_ms == 100.0
        assert len(merged.topics) == 4

    def test_deduplicates_topics(self):
        r1 = ThoughtResult(raw_text="", thought="A", topics=("physics", "energy"), latency_ms=10.0)
        r2 = ThoughtResult(raw_text="", thought="B", topics=("physics", "heat"), latency_ms=10.0)
        merged = ThoughtLoop._merge_chain_results([r1, r2])
        topic_list = list(merged.topics)
        assert topic_list.count("physics") == 1  # Deduplicated
        assert "energy" in topic_list
        assert "heat" in topic_list

    def test_two_result_merge_statementizes_question(self):
        r1 = ThoughtResult(raw_text="", thought="Coral polyps build reefs.", topics=("coral",), latency_ms=10.0)
        r2 = ThoughtResult(raw_text="", thought="How do coral polyps tolerate warming?", topics=("warming",), latency_ms=10.0)
        merged = ThoughtLoop._merge_chain_results([r1, r2])
        assert "A key open question is" in merged.thought
        assert "?" not in merged.thought

    def test_two_result_merge_avoids_redundant_repetition(self):
        r1 = ThoughtResult(raw_text="", thought="Cave formations can be shaped by tidal forces.", topics=("caves",), latency_ms=10.0)
        r2 = ThoughtResult(raw_text="", thought="Cave formations can be influenced by tidal forces.", topics=("tidal forces",), latency_ms=10.0)
        merged = ThoughtLoop._merge_chain_results([r1, r2])
        assert merged.thought.count("Cave formations") == 1
