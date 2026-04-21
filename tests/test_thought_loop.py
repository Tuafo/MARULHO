"""Tests for episodic memory, drives, thalamic gate, and thought loop."""

from __future__ import annotations

import time
import json
import pytest

from hecsn.cortex.core import MockCortex, ContextPacket, ThinkingMode, ThoughtDepth, ThoughtResult
from hecsn.cortex.episodic_memory import (
    EpisodicMemory,
    Episode,
    Provenance,
    SimpleEmbedder,
    _make_episode_id,
)
from hecsn.cortex.drives import (
    DriveSystem,
    DriveState,
    ThalamicGate,
    AntiRuminationCircuit,
)
from hecsn.cortex.thought_loop import ThoughtLoop, BrainStats
from hecsn.cortex.working_memory import WMItemType


# ---------------------------------------------------------------------------
# SimpleEmbedder
# ---------------------------------------------------------------------------

class TestSimpleEmbedder:
    def test_embedding_shape(self):
        emb = SimpleEmbedder(dim=64)
        vec = emb.embed("hello world")
        assert vec.shape == (64,)

    def test_normalized(self):
        import numpy as np
        emb = SimpleEmbedder(dim=128)
        vec = emb.embed("test sentence")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_similar_texts_high_similarity(self):
        emb = SimpleEmbedder(dim=128)
        a = emb.embed("the cat sat on the mat")
        b = emb.embed("the cat sat on the hat")
        sim = emb.similarity(a, b)
        assert sim > 0.8

    def test_different_texts_lower_similarity(self):
        emb = SimpleEmbedder(dim=128)
        a = emb.embed("quantum physics equations")
        b = emb.embed("chocolate cake recipe")
        sim = emb.similarity(a, b)
        assert sim < 0.5

    def test_empty_string(self):
        import numpy as np
        emb = SimpleEmbedder(dim=64)
        vec = emb.embed("")
        assert vec.shape == (64,)
        # Short text has zero trigrams → zero vector
        assert np.allclose(vec, 0.0)


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

class TestEpisode:
    def test_defaults(self):
        ep = Episode(episode_id="ep-1", content="hello")
        assert ep.provenance == Provenance.OBSERVED
        assert ep.confidence == 0.5
        assert ep.access_count == 0
        assert ep.created_at > 0

    def test_touch_increments(self):
        ep = Episode(episode_id="ep-1", content="hello")
        old_time = ep.last_accessed
        ep.touch()
        assert ep.access_count == 1
        assert ep.last_accessed >= old_time

    def test_graduate(self):
        ep = Episode(episode_id="ep-1", content="hyp", provenance=Provenance.DREAMED, confidence=0.3)
        ep.graduate()
        assert ep.provenance == Provenance.VERIFIED
        assert ep.confidence >= 0.7

    def test_graduate_only_dreamed(self):
        ep = Episode(episode_id="ep-1", content="obs", provenance=Provenance.OBSERVED, confidence=0.3)
        ep.graduate()
        assert ep.provenance == Provenance.OBSERVED  # unchanged

    def test_contradict(self):
        ep = Episode(episode_id="ep-1", content="wrong", provenance=Provenance.INFERRED, confidence=0.8)
        ep.contradict()
        assert ep.provenance == Provenance.CONTRADICTED
        assert ep.confidence <= 0.2

    def test_composite_importance(self):
        ep = Episode(
            episode_id="ep-1", content="important",
            provenance=Provenance.VERIFIED, salience=0.9, confidence=0.9,
        )
        imp = ep.composite_importance
        assert 0.0 < imp <= 1.0

    def test_trust_weights(self):
        assert Provenance.VERIFIED.trust_weight > Provenance.DREAMED.trust_weight
        assert Provenance.OBSERVED.trust_weight > Provenance.CONTRADICTED.trust_weight


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class TestEpisodicMemory:
    def test_store_and_retrieve(self):
        mem = EpisodicMemory(capacity=100)
        ep = mem.store("The sky is blue", topics=["nature", "sky"])
        assert mem.size == 1
        assert ep.episode_id in mem

    def test_deduplicate(self):
        mem = EpisodicMemory(capacity=100)
        ep1 = mem.store("The sky is blue")
        ep2 = mem.store("The sky is blue")
        assert mem.size == 1
        assert ep2.access_count >= 1

    def test_recall_by_similarity(self):
        mem = EpisodicMemory(capacity=100)
        mem.store("cats love to sleep in warm places", topics=["cats"])
        mem.store("quantum mechanics describes particle behavior", topics=["physics"])
        mem.store("kittens are young cats", topics=["cats"])

        results = mem.recall_by_similarity("cats and kittens", top_k=2)
        assert len(results) == 2
        contents = [r.content for r in results]
        assert any("cats" in c.lower() or "kitten" in c.lower() for c in contents)

    def test_recall_by_topic(self):
        mem = EpisodicMemory(capacity=100)
        mem.store("apples are fruits", topics=["food"])
        mem.store("bananas are yellow", topics=["food", "color"])
        mem.store("the sky is blue", topics=["nature"])

        results = mem.recall_by_topic("food")
        assert len(results) == 2

    def test_recall_recent(self):
        mem = EpisodicMemory(capacity=100)
        ep1 = mem.store("first thought", episode_id="ep-1")
        ep1.created_at = 1000.0
        ep2 = mem.store("second thought", episode_id="ep-2")
        ep2.created_at = 2000.0
        ep3 = mem.store("third thought", episode_id="ep-3")
        ep3.created_at = 3000.0

        results = mem.recall_recent(top_k=2)
        assert len(results) == 2
        assert results[0].content == "third thought"

    def test_eviction_at_capacity(self):
        mem = EpisodicMemory(capacity=3)
        mem.store("a", salience=0.1, episode_id="ep-a")
        mem.store("b", salience=0.9, episode_id="ep-b")
        mem.store("c", salience=0.5, episode_id="ep-c")
        # This should evict the least important
        mem.store("d", salience=0.8, episode_id="ep-d")
        assert mem.size == 3
        # 'a' had lowest salience and should be evicted
        assert "ep-a" not in mem

    def test_verified_protected_from_eviction(self):
        mem = EpisodicMemory(capacity=2)
        ep = mem.store("verified fact", salience=0.1, provenance=Provenance.VERIFIED, episode_id="ep-v")
        mem.store("normal", salience=0.2, episode_id="ep-n")
        mem.store("new", salience=0.9, episode_id="ep-new")
        # Verified should survive even with low salience
        assert "ep-v" in mem

    def test_remove(self):
        mem = EpisodicMemory(capacity=100)
        mem.store("to remove", topics=["test"], episode_id="ep-r")
        assert mem.size == 1
        removed = mem.remove("ep-r")
        assert removed is not None
        assert mem.size == 0

    def test_graduate_hypothesis(self):
        mem = EpisodicMemory(capacity=100)
        mem.store("dream idea", provenance=Provenance.DREAMED, episode_id="ep-d")
        assert mem.graduate_hypothesis("ep-d")
        ep = mem.get("ep-d")
        assert ep is not None
        assert ep.provenance == Provenance.VERIFIED

    def test_contradict_episode(self):
        mem = EpisodicMemory(capacity=100)
        mem.store("wrong claim", episode_id="ep-w")
        assert mem.contradict_episode("ep-w")
        ep = mem.get("ep-w")
        assert ep is not None
        assert ep.provenance == Provenance.CONTRADICTED

    def test_recall_hypotheses(self):
        mem = EpisodicMemory(capacity=100)
        mem.store("fact", provenance=Provenance.OBSERVED, episode_id="ep-f")
        mem.store("dream 1", provenance=Provenance.DREAMED, episode_id="ep-d1")
        mem.store("dream 2", provenance=Provenance.DREAMED, episode_id="ep-d2")
        hyps = mem.recall_hypotheses()
        assert len(hyps) == 2

    def test_recall_dream_lineage_tracks_graduated_dreams(self):
        mem = EpisodicMemory(capacity=100)
        mem.store("dream 1", provenance=Provenance.DREAMED, episode_id="ep-d1")
        mem.store("dream 2", provenance=Provenance.DREAMED, episode_id="ep-d2")
        mem.graduate_hypothesis("ep-d1")
        lineage = mem.recall_dream_lineage()
        assert len(lineage) == 2
        assert any(ep.provenance == Provenance.VERIFIED for ep in lineage)
        assert all(ep.dream_origin for ep in lineage)

    def test_recall_for_sleep(self):
        mem = EpisodicMemory(capacity=100)
        mem.store("high salience", salience=0.9, episode_id="ep-h")
        mem.store("low salience", salience=0.1, episode_id="ep-l")
        results = mem.recall_for_sleep(top_k=1)
        assert len(results) == 1
        assert results[0].salience == 0.9

    def test_stats(self):
        mem = EpisodicMemory(capacity=100)
        mem.store("one", provenance=Provenance.OBSERVED, episode_id="ep-1")
        mem.store("two", provenance=Provenance.DREAMED, episode_id="ep-2")
        stats = mem.stats
        assert stats["size"] == 2
        assert stats["provenance_distribution"]["observed"] == 1
        assert stats["provenance_distribution"]["dreamed"] == 1
        assert stats["embedder"]["kind"] == "SimpleEmbedder"

    def test_min_trust_filter(self):
        mem = EpisodicMemory(capacity=100)
        mem.store("dream", provenance=Provenance.DREAMED, episode_id="ep-d")
        mem.store("fact", provenance=Provenance.VERIFIED, episode_id="ep-v")
        results = mem.recall_by_similarity("anything", top_k=10, min_trust=0.5)
        assert all(r.provenance.trust_weight >= 0.5 for r in results)


# ---------------------------------------------------------------------------
# AntiRuminationCircuit
# ---------------------------------------------------------------------------

class TestAntiRumination:
    def test_no_boredom_initially(self):
        ar = AntiRuminationCircuit()
        assert ar.boredom_signal() == 0.0

    def test_repeated_topics_increase_boredom(self):
        ar = AntiRuminationCircuit(boredom_threshold=3)
        for _ in range(5):
            ar.record_topics(["same_topic"])
        assert ar.boredom_signal() > 0.0

    def test_diverse_topics_no_boredom(self):
        ar = AntiRuminationCircuit(boredom_threshold=3)
        for i in range(5):
            ar.record_topics([f"topic_{i}"])
        assert ar.boredom_signal() == 0.0

    def test_diversity_score(self):
        ar = AntiRuminationCircuit()
        ar.record_topics(["a", "b", "c"])
        assert ar.diversity_score() == 1.0
        ar.record_topics(["a", "a", "a"])
        assert ar.diversity_score() < 1.0

    def test_topic_avoidance(self):
        ar = AntiRuminationCircuit(boredom_threshold=3)
        for _ in range(5):
            ar.record_topics(["boring_topic"])
        avoid = ar.suggest_topic_avoidance()
        assert "boring_topic" in avoid


# ---------------------------------------------------------------------------
# DriveSystem
# ---------------------------------------------------------------------------

class TestDriveSystem:
    def test_initial_state(self):
        ds = DriveSystem()
        assert ds.state.curiosity == 0.5
        assert ds.state.fatigue == 0.0
        assert ds.thought_count == 0

    def test_surprise_update(self):
        ds = DriveSystem()
        # Apply multiple updates to overcome EMA smoothing
        for _ in range(10):
            ds.update_from_surprise(
                dopamine=0.9, serotonin=0.1,
                norepinephrine=0.3, acetylcholine=0.8,
            )
        assert ds.state.curiosity > 0.5  # High ACh + DA
        assert ds.state.satisfaction > 0.3  # High DA after convergence

    def test_thought_increases_fatigue(self):
        ds = DriveSystem()
        initial_fatigue = ds.state.fatigue
        result = ThoughtResult(
            raw_text="test", thought="test", topics=("topic",),
            parse_success=True,
        )
        ds.update_from_thought(result)
        assert ds.state.fatigue > initial_fatigue

    def test_external_input_reduces_boredom(self):
        ds = DriveSystem()
        ds.state.boredom = 0.8
        ds.update_from_external_input()
        assert ds.state.boredom < 0.8

    def test_should_sleep_when_fatigued(self):
        ds = DriveSystem()
        ds.state.fatigue = 0.9
        ds.state.social = 0.0
        assert ds.should_sleep()

    def test_should_not_sleep_when_social(self):
        ds = DriveSystem()
        ds.state.fatigue = 0.9
        ds.state.social = 0.5
        assert not ds.should_sleep()

    def test_choose_mode_answer_when_social(self):
        ds = DriveSystem()
        ds.state.social = 0.5
        assert ds.choose_mode() == ThinkingMode.ANSWER

    def test_choose_mode_reflect_when_bored(self):
        ds = DriveSystem()
        ds.state.boredom = 0.6
        assert ds.choose_mode() == ThinkingMode.REFLECT

    def test_tick_decays_fatigue(self):
        ds = DriveSystem()
        ds.state.fatigue = 0.5
        for _ in range(100):
            ds.tick()
        assert ds.state.fatigue < 0.5


# ---------------------------------------------------------------------------
# DriveState
# ---------------------------------------------------------------------------

class TestDriveState:
    def test_arousal(self):
        s = DriveState(curiosity=1.0, norepinephrine=1.0)
        assert s.arousal > 0.5

    def test_valence(self):
        s = DriveState(satisfaction=0.9, anxiety=0.1, dopamine=0.9, serotonin=0.1)
        assert s.valence > 0.0

    def test_dominant_drive(self):
        s = DriveState(curiosity=0.9, anxiety=0.1, boredom=0.1)
        assert s.dominant_drive == "curiosity"

    def test_to_summary(self):
        s = DriveState(curiosity=0.8)
        summary = s.to_summary()
        assert "curiosity" in summary.lower()


# ---------------------------------------------------------------------------
# ThalamicGate
# ---------------------------------------------------------------------------

class TestThalamicGate:
    def test_assemble_basic(self):
        mem = EpisodicMemory(capacity=100)
        mem.store("important memory", salience=0.9)
        drives = DriveSystem()
        gate = ThalamicGate(mem, drives)

        packet = gate.assemble()
        assert isinstance(packet, ContextPacket)
        # Drive summary is suppressed by default to prevent self-referential loops.
        # Instead, a seed topic direction is always provided.
        assert packet.mode == ThinkingMode.THINK

    def test_query_sets_answer_mode(self):
        mem = EpisodicMemory(capacity=100)
        drives = DriveSystem()
        gate = ThalamicGate(mem, drives)

        gate.submit_query("What is consciousness?")
        # submit_query calls update_from_external_input which boosts social to 0.3
        # Make sure social is above the 0.3 threshold for ANSWER mode
        drives.state.social = 0.5
        packet = gate.assemble()
        assert packet.external_query == "What is consciousness?"
        assert packet.mode == ThinkingMode.ANSWER

    def test_query_consumed_after_assembly(self):
        mem = EpisodicMemory(capacity=100)
        drives = DriveSystem()
        gate = ThalamicGate(mem, drives)

        gate.submit_query("test")
        gate.assemble()
        # Second assembly should not have the query
        packet2 = gate.assemble()
        assert packet2.external_query == ""

    def test_working_memory_replaces_thread_replay(self):
        """Wakeful assembly should not include raw thought-thread replay.

        Raw thread replay caused repetition loops, so the gate now relies on
        fresh seed rotation plus working-memory broadcast only for later chain
        phases.
        """
        mem = EpisodicMemory(capacity=100)
        drives = DriveSystem()
        gate = ThalamicGate(mem, drives)

        packet = gate.assemble()
        assert packet.working_memory_narrative == ""
        assert packet.forced_topic

    def test_sleep_assembly(self):
        mem = EpisodicMemory(capacity=100)
        mem.store("memory for dreams", salience=0.9)
        drives = DriveSystem()
        gate = ThalamicGate(mem, drives)

        packet = gate.assemble_for_sleep()
        assert packet.mode == ThinkingMode.DREAM
        assert len(packet.top_memories) > 0

    def test_sleep_assembly_compose_phase(self):
        mem = EpisodicMemory(capacity=100)
        ep = mem.store("reef chemistry", salience=0.9)
        drives = DriveSystem()
        gate = ThalamicGate(mem, drives)

        packet = gate.assemble_for_sleep([ep], phase="dream_compose")
        assert packet.deliberation_phase == "dream_compose"
        assert "testable hypothesis" in packet.drive_summary.lower()
        assert packet.max_response_tokens == 192

    def test_sleep_assembly_test_phase(self):
        mem = EpisodicMemory(capacity=100)
        ep = mem.store("reef chemistry", salience=0.9)
        drives = DriveSystem()
        gate = ThalamicGate(mem, drives)

        packet = gate.assemble_for_sleep([ep], phase="dream_test", hypothesis="Reef chemistry mirrors cave mineral deposition")
        assert packet.deliberation_phase == "dream_test"
        assert "Candidate hypothesis" in packet.working_memory_narrative
        assert packet.max_response_tokens == 160


# ---------------------------------------------------------------------------
# ThoughtLoop
# ---------------------------------------------------------------------------

class TestThoughtLoop:
    def test_step_with_fake_cortex(self):
        cortex = MockCortex()
        loop = ThoughtLoop(cortex=cortex, min_thought_interval_s=0.0)
        # Boost curiosity to trigger thinking
        loop.drives.state.curiosity = 0.8
        result = loop.step()
        assert result is not None
        assert result.parse_success
        assert loop.stats.thoughts_generated == 1

    def test_step_no_think_when_low_drives(self):
        cortex = MockCortex()
        loop = ThoughtLoop(cortex=cortex, min_thought_interval_s=0.0)
        loop.drives.state.curiosity = 0.1
        loop.drives.state.anxiety = 0.1
        loop.drives.state.social = 0.0
        result = loop.step()
        assert result is None

    def test_uses_provided_memory_instance_even_when_empty(self):
        cortex = MockCortex()
        mem = EpisodicMemory(capacity=100)
        loop = ThoughtLoop(cortex=cortex, memory=mem, min_thought_interval_s=0.0)
        assert loop.memory is mem

    def test_memory_stored_after_thought(self):
        cortex = MockCortex()
        loop = ThoughtLoop(cortex=cortex, min_thought_interval_s=0.0)
        loop.drives.state.curiosity = 0.8
        loop.step()
        assert loop.memory.size >= 1

    def test_sleep_triggered_by_fatigue(self):
        cortex = MockCortex()
        loop = ThoughtLoop(
            cortex=cortex,
            min_thought_interval_s=0.0,
            sleep_cooldown_s=0.0,
            sleep_dream_count=2,
        )
        loop.drives.state.fatigue = 0.9
        loop.drives.state.social = 0.0
        loop.drives.state.curiosity = 0.1
        loop.step()  # Should trigger sleep
        assert loop.stats.sleep_cycles == 1
        assert loop.stats.dreams_generated == 2

    def test_fatigue_reduces_after_sleep(self):
        cortex = MockCortex()
        loop = ThoughtLoop(
            cortex=cortex,
            min_thought_interval_s=0.0,
            sleep_cooldown_s=0.0,
        )
        loop.drives.state.fatigue = 0.9
        loop.drives.state.social = 0.0
        initial = loop.drives.state.fatigue
        loop.step()
        assert loop.drives.state.fatigue < initial

    def test_sleep_cycle_graduates_supported_dream_hypothesis(self):
        cortex = MockCortex(responses=[
            {
                "thought": "Coral skeletons and cave formations both rely on calcium carbonate precipitation.",
                "topics": ["coral reefs", "caves", "calcium carbonate"],
                "valence": 0.1,
                "confidence": 0.7,
                "action": None,
            },
            {
                "thought": "The memories support a shared calcium-carbonate precipitation mechanism.",
                "topics": ["calcium carbonate"],
                "valence": 0.1,
                "confidence": 0.85,
                "action": None,
            },
        ])
        loop = ThoughtLoop(cortex=cortex, min_thought_interval_s=0.0, sleep_dream_count=1)
        loop.memory.store("Coral reefs build skeletons from calcium carbonate.", provenance=Provenance.OBSERVED, topics=["coral reefs"], salience=0.9)
        loop.memory.store("Cave stalactites form through calcium carbonate precipitation.", provenance=Provenance.OBSERVED, topics=["caves"], salience=0.8)

        dreams = loop._sleep_cycle()
        lineage = loop.memory.recall_dream_lineage()
        assert len(dreams) == 1
        assert len(lineage) == 1
        assert lineage[0].provenance == Provenance.VERIFIED
        assert loop.stats.dream_verification_rate == 1.0

    def test_sleep_cycle_contradiction_queues_wake_tension(self):
        cortex = MockCortex(responses=[
            {
                "thought": "Whale migration and optical illusions may share long-distance signaling principles.",
                "topics": ["whales", "optics"],
                "valence": 0.0,
                "confidence": 0.55,
                "action": None,
            },
            {
                "thought": "This hypothesis is unsupported because the memories do not share a real signaling mechanism.",
                "topics": ["whales", "optics"],
                "valence": -0.2,
                "confidence": 0.2,
                "action": None,
            },
        ])
        loop = ThoughtLoop(cortex=cortex, min_thought_interval_s=0.0, sleep_dream_count=1)
        loop.memory.store("Humpback whales migrate long distances using ocean cues.", provenance=Provenance.OBSERVED, topics=["whales"], salience=0.9)
        loop.memory.store("Optical illusions distort perceived line length.", provenance=Provenance.OBSERVED, topics=["optics"], salience=0.8)

        loop._sleep_cycle()
        lineage = loop.memory.recall_dream_lineage()
        assert len(lineage) == 1
        assert lineage[0].provenance == Provenance.CONTRADICTED
        assert loop._pending_wake_tensions

    def test_pending_wake_tension_hydrates_working_memory(self):
        loop = ThoughtLoop(cortex=MockCortex(), min_thought_interval_s=0.0)
        loop._queue_wake_tension("Dream contradiction: coral chemistry does not fit the memory evidence.", topics=["coral", "chemistry"])
        loop.working_memory.clear()
        loop._inject_pending_wake_tensions()
        assert loop.working_memory.has_tension()
        assert loop._choose_depth() == ThoughtDepth.DEEP
        assert loop.working_memory.strongest_item().item_type == WMItemType.TENSION

    def test_dream_validation_prefix_supported_is_honored(self):
        loop = ThoughtLoop(cortex=MockCortex(), min_thought_interval_s=0.0)
        hypothesis = ThoughtResult(
            raw_text="",
            thought="Coral reefs and caves share a calcium-carbonate precipitation mechanism.",
            topics=("coral reefs", "caves", "calcium carbonate"),
            confidence=0.6,
        )
        memories = [
            Episode(episode_id="ep-1", content="Coral reefs build skeletons from calcium carbonate.", topics=("coral reefs", "calcium carbonate")),
            Episode(episode_id="ep-2", content="Cave stalactites form by calcium carbonate precipitation.", topics=("caves", "calcium carbonate")),
        ]
        validation = ThoughtResult(
            raw_text="",
            thought="SUPPORTED: Both memories point to the same calcium-carbonate precipitation mechanism.",
            topics=("calcium carbonate",),
            confidence=0.55,
        )
        assert loop._dream_validation_verdict(validation, hypothesis=hypothesis, memories=memories) == "supported"

    def test_dream_validation_moderate_confidence_can_still_be_supported_with_evidence(self):
        loop = ThoughtLoop(cortex=MockCortex(), min_thought_interval_s=0.0)
        hypothesis = ThoughtResult(
            raw_text="",
            thought="Coral reefs and caves share a calcium-carbonate precipitation mechanism.",
            topics=("coral reefs", "caves", "calcium carbonate"),
            confidence=0.6,
        )
        memories = [
            Episode(episode_id="ep-1", content="Coral reefs build skeletons from calcium carbonate.", topics=("coral reefs", "calcium carbonate")),
            Episode(episode_id="ep-2", content="Cave stalactites form by calcium carbonate precipitation.", topics=("caves", "calcium carbonate")),
        ]
        validation = ThoughtResult(
            raw_text="",
            thought="The memories support a shared calcium-carbonate precipitation mechanism.",
            topics=("calcium carbonate",),
            confidence=0.58,
        )
        assert loop._dream_validation_verdict(validation, hypothesis=hypothesis, memories=memories) == "supported"

    def test_inject_observation(self):
        cortex = MockCortex()
        loop = ThoughtLoop(cortex=cortex)
        loop.inject_observation("The sun is a star", topics=["astronomy"])
        assert loop.memory.size == 1

    def test_inject_surprise(self):
        cortex = MockCortex()
        loop = ThoughtLoop(cortex=cortex)
        loop.inject_surprise(dopamine=0.9, acetylcholine=0.9)
        assert loop.drives.state.curiosity > 0.5

    def test_submit_query(self):
        cortex = MockCortex()
        loop = ThoughtLoop(cortex=cortex, min_thought_interval_s=0.0)
        loop.submit_query("What is life?")
        loop.drives.state.social = 0.5  # Ensure thinking is triggered
        result = loop.step()
        assert result is not None

    def test_background_loop_start_stop(self):
        cortex = MockCortex()
        loop = ThoughtLoop(
            cortex=cortex,
            tick_interval_ms=50.0,
            min_thought_interval_s=0.1,
        )
        loop.drives.state.curiosity = 0.8

        loop.start()
        assert loop.is_running
        time.sleep(0.5)  # Let it run a few cycles
        loop.stop()
        assert not loop.is_running
        assert loop.stats.thoughts_generated > 0

    def test_callback_on_thought(self):
        thoughts: list[ThoughtResult] = []
        cortex = MockCortex()
        loop = ThoughtLoop(
            cortex=cortex,
            min_thought_interval_s=0.0,
            on_thought=lambda t: thoughts.append(t),
        )
        loop.drives.state.curiosity = 0.8
        loop.start()
        time.sleep(0.5)
        loop.stop()
        assert len(thoughts) > 0

    def test_stats_tracking(self):
        cortex = MockCortex()
        loop = ThoughtLoop(cortex=cortex, min_thought_interval_s=0.0)
        loop.drives.state.curiosity = 0.8
        loop.step()
        assert loop.stats.thoughts_generated == 1
        assert loop.stats.last_thought != ""
        assert loop.stats.total_inference_ms > 0
        assert loop.stats.memory_count > 0

    def test_snapshot_includes_episodic_memory_embedder_stats(self):
        loop = ThoughtLoop(cortex=MockCortex(), min_thought_interval_s=0.0)
        snap = loop.snapshot()
        assert snap["episodic_memory"]["embedder"]["kind"] == "SimpleEmbedder"
        assert snap["episodic_memory"]["embedder"]["available"] is False


# ---------------------------------------------------------------------------
# BrainStats
# ---------------------------------------------------------------------------

class TestBrainStats:
    def test_avg_inference_empty(self):
        s = BrainStats()
        assert s.avg_inference_ms == 0.0

    def test_avg_inference(self):
        s = BrainStats(thoughts_generated=2, total_inference_ms=100.0)
        assert s.avg_inference_ms == 50.0
