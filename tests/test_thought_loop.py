"""Tests for episodic memory, drives, thalamic gate, and thought loop."""

from __future__ import annotations

import time
import json
import pytest

from hecsn.cortex.core import CorticalCore, MockCortex, ContextPacket, ThinkingMode, ThoughtDepth, ThoughtResult
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
from hecsn.cortex.thought_loop import (
    THOUGHT_LOOP_RETIRED_REASON,
    ThoughtLoop,
)
from hecsn.cortex.working_memory import WMItemType
from hecsn.semantics.brain_stats import BrainStats
from hecsn.semantics.cognitive_signal import CognitiveSignalState


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

    def test_recall_for_deliberation_prefers_recent_relevant_sensory_evidence(self):
        mem = EpisodicMemory(capacity=100)
        mem.store(
            "Cats rest indoors and chase mice at night.",
            provenance=Provenance.OBSERVED,
            topics=["cats", "mice"],
            salience=0.8,
            metadata={
                "grounded": True,
                "observation_kind": "source",
                "grounding_signal": 0.58,
                "evidence_unit_count": 16,
                "focus_terms": ["cats", "indoors", "mice"],
            },
            episode_id="ep-source",
        )
        sensory = mem.store(
            "Water splashes while wind and footsteps move through an outdoor path.",
            provenance=Provenance.OBSERVED,
            topics=["water", "wind", "footsteps"],
            salience=0.9,
            metadata={
                "grounded": True,
                "observation_kind": "sensory",
                "modality": "audio",
                "semantic_match": 0.92,
                "grounding_signal": 0.90,
                "evidence_unit_count": 2,
                "focus_terms": ["water", "wind", "footsteps"],
            },
            episode_id="ep-sensory",
        )

        bundle = mem.recall_for_deliberation(
            "environmental sound of water and footsteps",
            grounded_top_k=2,
            support_top_k=2,
        )
        assert len(bundle.grounded) > 0
        assert bundle.grounded[0].episode_id == sensory.episode_id
        assert "water" in bundle.grounded[0].focused_text.lower()
        assert "footsteps" in bundle.grounded[0].focused_text.lower()
        assert "water" in bundle.target.lower()

    def test_recall_for_query_prefers_recent_grounded_evidence(self):
        mem = EpisodicMemory(capacity=100)
        older = mem.store(
            "Cats rest on rooftops and chase birds at dawn.",
            provenance=Provenance.OBSERVED,
            topics=["cats", "birds"],
            salience=0.9,
            episode_id="ep-older",
        )
        older.created_at = time.time() - (8 * 3600)
        grounded = mem.store(
            "Cats rest indoors and chase mice at night.",
            provenance=Provenance.OBSERVED,
            topics=["cats", "mice"],
            salience=0.8,
            metadata={"grounded": True, "observation_kind": "source"},
            episode_id="ep-grounded",
        )
        bundle = mem.recall_for_query(
            "Where do cats rest and what do they chase at night?",
            grounded_top_k=2,
            support_top_k=2,
        )

        assert len(bundle.grounded) > 0
        assert bundle.grounded[0].episode_id == grounded.episode_id
        assert "indoors" in bundle.grounded[0].focused_text.lower()
        assert "mice" in bundle.grounded[0].focused_text.lower()
        assert bundle.grounded_coverage > 0.3
        assert all(match.episode_id != grounded.episode_id for match in bundle.support)

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

    def test_recent_grounded_focus_prefers_metadata_focus_terms(self):
        mem = EpisodicMemory(capacity=100)
        mem.store(
            "Water splashes while wind and footsteps move through an outdoor path.",
            provenance=Provenance.OBSERVED,
            topics=["audio"],
            salience=0.9,
            metadata={
                "grounded": True,
                "observation_kind": "sensory",
                "modality": "audio",
                "grounding_signal": 0.9,
                "focus_terms": ["water", "wind", "footsteps"],
            },
            episode_id="ep-sensory-focus",
        )
        focus = mem.recent_grounded_focus()
        assert "water" in focus.lower()
        assert "footsteps" in focus.lower()


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
        assert ds.state.curiosity == 0.0
        assert ds.state.uncertainty == 0.0
        assert ds.state.fatigue == 0.0
        assert ds.startup_quiet is True
        assert ds.pending_grounded_observations == 0
        assert ds.substrate_hysteresis_active is False
        assert ds.substrate_hysteresis_updates == 0
        assert ds.thought_count == 0

    def test_surprise_update(self):
        ds = DriveSystem()
        initial_curiosity = ds.state.curiosity
        # Apply multiple updates to overcome EMA smoothing
        for _ in range(10):
            ds.update_from_surprise(
                dopamine=0.9, serotonin=0.1,
                norepinephrine=0.3, acetylcholine=0.8,
            )
        assert ds.state.curiosity > initial_curiosity
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

    def test_grounded_observation_reduces_boredom_and_queues_wake(self):
        ds = DriveSystem()
        ds.state.boredom = 0.8
        ds.update_from_grounded_observation()
        assert ds.state.boredom < 0.8
        assert ds.pending_grounded_observations == 1
        assert ds.startup_quiet is False

    def test_should_think_requires_concrete_wake_trigger(self):
        ds = DriveSystem()
        ds.state.curiosity = 0.9
        ds.state.anxiety = 0.8
        assert ds.should_think() is False
        assert ds.last_gate_reason == "startup_quiet"

    def test_prediction_error_breaks_startup_quiet(self):
        ds = DriveSystem()
        ds.update_from_prediction_error(0.6, 0.8, 0.2, 0.1)
        assert ds.startup_quiet is False
        assert ds.pending_substrate_wakes == 1
        assert ds.should_think() is True
        assert ds.last_gate_reason in {"prediction_error", "uncertainty", "exploration_urgency", "ne_alerting", "ach_attention"}

    def test_substrate_wake_is_one_shot_without_renewal(self):
        ds = DriveSystem()
        ds.update_from_prediction_error(0.6, 0.8, 0.2, 0.1)
        assert ds.pending_substrate_wakes == 1
        assert ds.should_think() is True
        assert ds.consume_substrate_wake() is True
        assert ds.pending_substrate_wakes == 0
        assert ds.should_think() is False
        assert ds.last_gate_reason == "idle_no_trigger"

    def test_prediction_error_renewal_rearms_substrate_wake(self):
        ds = DriveSystem()
        ds.update_from_prediction_error(0.6, 0.8, 0.2, 0.1)
        assert ds.consume_substrate_wake() is True
        ds.update_from_prediction_error(0.62, 0.82, 0.22, 0.12)
        assert ds.pending_substrate_wakes == 0
        ds.update_from_prediction_error(0.8, 0.95, 0.15, 0.05)
        assert ds.pending_substrate_wakes == 1
        assert ds.should_think() is True

    def test_moderate_pressure_does_not_rearm_without_strong_sustain(self):
        ds = DriveSystem(
            substrate_rearm_cooldown_s=0.12,
            substrate_sustain_min_updates=2,
            substrate_sustain_margin=0.22,
        )
        ds.update_from_prediction_error(0.35, 0.40, 0.86, 0.82)
        assert ds.consume_substrate_wake() is True
        time.sleep(0.13)
        ds.update_from_prediction_error(0.35, 0.40, 0.86, 0.82)
        assert ds.pending_substrate_wakes == 0
        assert ds.substrate_hysteresis_active is False
        assert ds.last_gate_reason in {"prediction_error", "startup_quiet"}

    def test_strong_sustained_pressure_rearms_after_hysteresis_window(self):
        ds = DriveSystem(
            substrate_rearm_cooldown_s=0.12,
            substrate_sustain_min_updates=2,
            substrate_sustain_margin=0.22,
        )
        ds.update_from_prediction_error(0.7, 0.85, 0.2, 0.1)
        assert ds.consume_substrate_wake() is True
        time.sleep(0.13)
        ds.update_from_prediction_error(0.7, 0.85, 0.2, 0.1)
        assert ds.pending_substrate_wakes == 1
        assert ds.substrate_hysteresis_active is True
        assert ds.substrate_hysteresis_reason == "prediction_error"
        assert ds.substrate_hysteresis_updates >= 2
        assert ds.should_think() is True
        assert ds.last_gate_reason == "prediction_error_sustained"

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

    def test_choose_mode_think_when_social_without_query(self):
        ds = DriveSystem()
        ds.state.social = 0.5
        assert ds.choose_mode() == ThinkingMode.THINK

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

    @pytest.mark.skip(reason="ThoughtLoop runtime is retired; Cognitive Signal is tested through Subcortex status surfaces")
    def test_cognitive_signal_state_accepts_typed_provider_payload(self):
        signal = CognitiveSignalState(
            source="test.subcortex",
            sampled_at=123.0,
            prediction_error_mean=0.4,
            prediction_error_max=0.6,
            predictive_confidence_mean=0.3,
            predictive_confidence_min=0.2,
            recent_concepts=("reef chemistry",),
        )
        loop = ThoughtLoop(
            cortex=MockCortex(),
            min_thought_interval_s=0.0,
            signal_provider=lambda: signal,
        )

        refreshed = loop._refresh_cognitive_signals()
        snap = loop.snapshot()

        assert refreshed is signal
        assert loop.drives.pending_substrate_wakes == 1
        assert snap["cognitive_signals"]["schema_version"] == "cognitive_signal.v1"
        assert snap["cognitive_signals"]["source"] == "test.subcortex"
        assert snap["cognitive_signals"]["sampled_at"] == 123.0
        assert "reef chemistry" in snap["cognitive_signals"]["recent_concepts"]


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

    def test_query_assembles_grounded_evidence(self):
        mem = EpisodicMemory(capacity=100)
        mem.store(
            "Cats rest indoors and chase mice at night.",
            provenance=Provenance.OBSERVED,
            topics=["cats", "mice"],
            salience=0.9,
            metadata={"grounded": True, "observation_kind": "source"},
        )
        drives = DriveSystem()
        gate = ThalamicGate(mem, drives)

        gate.submit_query("Where do cats rest and what do they chase at night?")
        packet = gate.assemble()
        assert packet.mode == ThinkingMode.ANSWER
        assert packet.external_query
        assert len(packet.grounded_evidence) > 0
        assert any("indoors" in item.text.lower() for item in packet.grounded_evidence)

    def test_non_query_assembly_reuses_recent_grounded_evidence(self):
        mem = EpisodicMemory(capacity=100)
        mem.store(
            "Cats rest indoors and chase mice at night.",
            provenance=Provenance.OBSERVED,
            topics=["cats", "rest", "indoors", "mice"],
            salience=0.9,
            metadata={"grounded": True, "observation_kind": "source"},
        )
        drives = DriveSystem()
        gate = ThalamicGate(mem, drives)

        packet = gate.assemble()
        assert packet.mode == ThinkingMode.THINK
        assert len(packet.grounded_evidence) > 0
        assert any("indoors" in item.text.lower() for item in packet.grounded_evidence)
        assert "cats" in packet.forced_topic.lower()

    def test_emit_deliberation_feedback_routes_topics_and_valence(self):
        mem = EpisodicMemory(capacity=100)
        drives = DriveSystem()
        gate = ThalamicGate(mem, drives)

        result = ThoughtResult(
            raw_text="",
            thought="Explore reef chemistry uncertainty.",
            topics=("Reef Chemistry", "AI"),
            confidence=0.25,
            emotional_valence=-0.6,
        )

        feedback = gate.emit_deliberation_feedback(result)

        assert feedback["topic_boosts"] == [
            ("reef chemistry", pytest.approx(0.125)),
            ("ai", pytest.approx(0.125)),
        ]
        assert feedback["grounding_candidates"] == ["Reef", "Chemistry"]
        assert feedback["emotional_valence"] == pytest.approx(-0.6)
        assert feedback["confidence"] == pytest.approx(0.25)
        assert drives.state.anxiety == pytest.approx(0.03)

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

def test_thought_loop_constructor_is_retired():
    with pytest.raises(RuntimeError, match=THOUGHT_LOOP_RETIRED_REASON):
        ThoughtLoop(cortex=MockCortex())


@pytest.mark.skip(reason="ThoughtLoop runtime path is retired; keep only primitive/static helper tests active")
class TestThoughtLoop:
    def test_step_with_fake_cortex(self):
        cortex = MockCortex()
        loop = ThoughtLoop(cortex=cortex, min_thought_interval_s=0.0)
        loop.drives.update_from_prediction_error(0.7, 0.8, 0.2, 0.1)
        result = loop.step()
        assert result is not None
        assert result.parse_success
        assert loop.stats.thoughts_generated == 1

    def test_step_idle_startup_stays_quiet_without_input(self):
        cortex = MockCortex()
        loop = ThoughtLoop(cortex=cortex, min_thought_interval_s=0.0)
        result = loop.step()
        assert result is None
        snap = loop.snapshot()
        assert snap["thoughts_generated"] == 0
        assert snap["gating"]["startup_quiet"] is True
        assert snap["gating"]["last_gate_reason"] == "startup_quiet"

    def test_step_no_think_when_only_affective_drives_are_high(self):
        cortex = MockCortex()
        loop = ThoughtLoop(cortex=cortex, min_thought_interval_s=0.0)
        loop.drives.state.curiosity = 0.9
        loop.drives.state.anxiety = 0.8
        loop.drives.state.social = 0.6
        result = loop.step()
        assert result is None

    def test_prediction_error_wake_does_not_free_run_without_renewal(self):
        cortex = MockCortex()
        loop = ThoughtLoop(cortex=cortex, min_thought_interval_s=0.0)
        loop.drives.update_from_prediction_error(0.7, 0.8, 0.2, 0.1)
        first = loop.step(force=True)
        second = loop.step(force=True)
        assert first is not None
        assert second is None
        assert loop.snapshot()["gating"]["pending_substrate_wakes"] == 0
        assert loop.snapshot()["gating"]["last_gate_reason"] == "idle_no_trigger"

    def test_prediction_error_renewal_can_rearm_thinking(self):
        cortex = MockCortex(responses=[
            {"thought": "A quick fact about gravity", "topics": ["physics"], "valence": 0.1, "confidence": 0.7, "action": None},
            {"thought": "A second fact about magnetism", "topics": ["physics"], "valence": 0.1, "confidence": 0.7, "action": None},
        ])
        loop = ThoughtLoop(cortex=cortex, min_thought_interval_s=0.0)
        loop.drives.update_from_prediction_error(0.7, 0.8, 0.2, 0.1)
        first = loop.step(force=True)
        loop.drives.update_from_prediction_error(0.8, 0.95, 0.15, 0.05)
        second = loop.step(force=True)
        assert first is not None
        assert second is not None
        assert first.thought != second.thought
        assert loop.stats.thoughts_generated == 2

    def test_unresolved_tension_can_continue_for_bounded_cycles(self):
        loop = ThoughtLoop(cortex=MockCortex(responses=[
            {"thought": "A contradiction remains in coral growth evidence.", "topics": ["coral"], "valence": -0.1, "confidence": 0.4, "action": None},
            {"thought": "The contradiction still needs explanation.", "topics": ["coral"], "valence": -0.1, "confidence": 0.4, "action": None},
        ]), min_thought_interval_s=0.0)
        loop._queue_wake_tension(
            "Dream contradiction: coral growth does not match the inferred mechanism.",
            topics=["coral growth", "mechanism"],
            salience=0.9,
            continuation_cycles=2,
        )
        first = loop.step(force=True)
        second = loop.step(force=True)
        third = loop.step(force=True)
        assert first is not None
        assert second is not None
        assert third is None
        snap = loop.snapshot()
        assert snap["gating"]["active_tension_count"] == 0
        assert snap["gating"]["last_gate_reason"] == "idle_no_trigger"

    def test_uses_provided_memory_instance_even_when_empty(self):
        cortex = MockCortex()
        mem = EpisodicMemory(capacity=100)
        loop = ThoughtLoop(cortex=cortex, memory=mem, min_thought_interval_s=0.0)
        assert loop.memory is mem

    def test_memory_stored_after_thought(self):
        cortex = MockCortex()
        loop = ThoughtLoop(cortex=cortex, min_thought_interval_s=0.0)
        loop.drives.update_from_prediction_error(0.7, 0.8, 0.2, 0.1)
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

    def test_explicit_sleep_request_runs_without_fatigue_gate(self):
        cortex = MockCortex()
        loop = ThoughtLoop(
            cortex=cortex,
            min_thought_interval_s=0.0,
            sleep_cooldown_s=30.0,
            sleep_dream_count=1,
        )

        request = loop.request_sleep(source="operator", reason="Need a consolidation pass.")
        assert request["accepted"]
        assert request["request"]["source"] == "operator"
        assert loop.stats.sleep_cycles == 0

        loop.step(force=True)
        snap = loop.snapshot()

        assert loop.stats.sleep_cycles == 1
        assert loop.stats.dreams_generated == 1
        assert snap["sleep_control"]["requests_submitted"] == 1
        assert snap["sleep_control"]["requested_cycles_completed"] == 1
        assert snap["sleep_control"]["pending_request"] is None
        assert snap["sleep_control"]["last_request"]["source"] == "operator"
        assert snap["sleep_control"]["last_cycle"]["trigger"] == "requested"
        assert snap["sleep_control"]["last_cycle"]["request"]["source"] == "operator"

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
        assert loop.drives.state.curiosity > 0.0

    def test_submit_query(self):
        cortex = MockCortex()
        loop = ThoughtLoop(cortex=cortex, min_thought_interval_s=0.0)
        loop.submit_query("What is life?")
        result = loop.step()
        assert result is not None

    def test_submit_query_bypasses_thought_interval(self):
        cortex = MockCortex()
        loop = ThoughtLoop(cortex=cortex, min_thought_interval_s=60.0)
        loop._last_thought_time = time.time()
        loop.submit_query("What is life?")
        result = loop.step()
        assert result is not None

    def test_external_query_uses_grounded_evidence_context(self):
        class RecordingCortex(CorticalCore):
            model = "recording-cortex"
            temperature = 0.7

            def __init__(self) -> None:
                self.last_context = None

            def generate(self, context: ContextPacket) -> ThoughtResult:
                self.last_context = context
                evidence_text = " ".join(item.text for item in context.grounded_evidence)
                return ThoughtResult(
                    raw_text="",
                    thought=evidence_text or context.external_query,
                    topics=("cats", "mice"),
                    confidence=0.9,
                    latency_ms=1.0,
                    parse_success=True,
                )

            def is_available(self) -> bool:
                return True

        cortex = RecordingCortex()
        loop = ThoughtLoop(cortex=cortex, min_thought_interval_s=60.0)
        loop.inject_observation(
            "Cats rest indoors and chase mice at night.",
            topics=["cats", "mice"],
            salience=0.9,
            metadata={"grounded": True, "observation_kind": "source"},
        )
        loop.submit_query("Where do cats rest and what do they chase at night?")
        result = loop.step()
        assert result is not None
        assert cortex.last_context is not None
        assert cortex.last_context.external_query
        assert len(cortex.last_context.grounded_evidence) > 0
        assert any("indoors" in item.text.lower() for item in cortex.last_context.grounded_evidence)

    def test_external_query_recovers_from_generic_answer_drift(self):
        class DriftCortex(CorticalCore):
            model = "drift-cortex"
            temperature = 0.7

            def generate(self, context: ContextPacket) -> ThoughtResult:
                return ThoughtResult(
                    raw_text="",
                    thought="Cats are curious animals with distinct personalities.",
                    topics=("cats",),
                    confidence=0.5,
                    latency_ms=1.0,
                    parse_success=True,
                )

            def is_available(self) -> bool:
                return True

        loop = ThoughtLoop(cortex=DriftCortex(), min_thought_interval_s=60.0)
        loop.inject_observation(
            "Cats rest indoors and chase mice at night.",
            topics=["cats", "mice"],
            salience=0.9,
            metadata={"grounded": True, "observation_kind": "source"},
        )
        loop.submit_query("Where do cats rest and what do they chase at night?")
        result = loop.step()
        assert result is not None
        assert "indoors" in result.thought.lower()
        assert "mice" in result.thought.lower()

        snap = loop.snapshot()
        assert snap["grounding"]["query_answers_evaluated"] == 1
        assert snap["grounding"]["query_recovery_rate"] == 1.0
        assert snap["recent_thoughts"][-1]["grounding"]["fallback_used"] is True
        assert snap["recent_thoughts"][-1]["grounding"]["alignment_score"] >= 0.4

    def test_wakeful_thought_uses_grounded_evidence_context(self):
        class RecordingCortex(CorticalCore):
            model = "recording-cortex"
            temperature = 0.7

            def __init__(self) -> None:
                self.last_context = None

            def generate(self, context: ContextPacket) -> ThoughtResult:
                self.last_context = context
                evidence_text = " ".join(item.text for item in context.grounded_evidence)
                return ThoughtResult(
                    raw_text="",
                    thought=evidence_text or context.forced_topic,
                    topics=("cats", "mice"),
                    confidence=0.9,
                    latency_ms=1.0,
                    parse_success=True,
                )

            def is_available(self) -> bool:
                return True

        cortex = RecordingCortex()
        loop = ThoughtLoop(cortex=cortex, min_thought_interval_s=0.0)
        loop.inject_observation(
            "Cats rest indoors and chase mice at night.",
            topics=["cats", "mice"],
            salience=0.9,
            metadata={"grounded": True, "observation_kind": "source"},
        )
        result = loop.step()
        assert result is not None
        assert cortex.last_context is not None
        assert len(cortex.last_context.grounded_evidence) > 0
        assert "cats" in cortex.last_context.forced_topic.lower()
        assert "indoors" in result.thought.lower()
        assert "mice" in result.thought.lower()

        snap = loop.snapshot()
        assert snap["grounding"]["wakeful_thoughts_evaluated"] == 1
        assert snap["grounding"]["mean_wakeful_alignment"] >= 0.4

    def test_wakeful_thought_recovers_from_generic_drift(self):
        class DriftCortex(CorticalCore):
            model = "drift-cortex"
            temperature = 0.7

            def generate(self, context: ContextPacket) -> ThoughtResult:
                return ThoughtResult(
                    raw_text="",
                    thought="Cats are familiar domestic animals with individual temperaments.",
                    topics=("cats",),
                    confidence=0.5,
                    latency_ms=1.0,
                    parse_success=True,
                )

            def is_available(self) -> bool:
                return True

        loop = ThoughtLoop(cortex=DriftCortex(), min_thought_interval_s=0.0)
        loop.inject_observation(
            "Cats rest indoors and chase mice at night.",
            topics=["cats", "mice"],
            salience=0.9,
            metadata={"grounded": True, "observation_kind": "source"},
        )
        result = loop.step()
        assert result is not None
        assert "indoors" in result.thought.lower()
        assert "mice" in result.thought.lower()

        snap = loop.snapshot()
        assert snap["grounding"]["wakeful_thoughts_evaluated"] == 1
        assert snap["grounding"]["wakeful_recovery_rate"] == 1.0
        assert snap["recent_thoughts"][-1]["grounding"]["kind"] == "wakeful"
        assert snap["recent_thoughts"][-1]["grounding"]["fallback_used"] is True

    def test_background_loop_stays_quiet_without_input(self):
        cortex = MockCortex()
        loop = ThoughtLoop(
            cortex=cortex,
            tick_interval_ms=50.0,
            min_thought_interval_s=0.0,
        )
        loop.start()
        time.sleep(0.25)
        loop.stop()
        assert loop.stats.thoughts_generated == 0
        assert loop.snapshot()["gating"]["last_gate_reason"] in {"startup_quiet", "idle_no_trigger"}

    def test_background_loop_constant_moderate_pressure_fires_once_then_quiets(self):
        thoughts_at: list[float] = []
        signal = {
            "prediction_error_mean": 0.35,
            "prediction_error_max": 0.40,
            "predictive_confidence_mean": 0.86,
            "predictive_confidence_min": 0.82,
            "recent_concepts": ["reef current"],
        }
        loop = ThoughtLoop(
            cortex=MockCortex(),
            tick_interval_ms=50.0,
            min_thought_interval_s=0.0,
            signal_provider=lambda: signal,
            drives=DriveSystem(
                substrate_rearm_cooldown_s=0.18,
                substrate_sustain_min_updates=2,
                substrate_sustain_margin=0.22,
            ),
            on_thought=lambda _: thoughts_at.append(time.monotonic()),
        )
        loop.start()
        time.sleep(0.6)
        loop.stop()

        snap = loop.snapshot()
        assert len(thoughts_at) == 1
        assert snap["thoughts_generated"] == 1
        assert snap["gating"]["pending_substrate_wakes"] == 0
        assert snap["gating"]["substrate_hysteresis_active"] is False

    def test_background_loop_strong_sustained_pressure_rearms_with_hysteresis(self):
        thoughts_at: list[float] = []
        signal = {
            "prediction_error_mean": 0.72,
            "prediction_error_max": 0.90,
            "predictive_confidence_mean": 0.18,
            "predictive_confidence_min": 0.08,
            "recent_concepts": ["aurora borealis"],
        }
        loop = ThoughtLoop(
            cortex=MockCortex(),
            tick_interval_ms=50.0,
            min_thought_interval_s=0.0,
            signal_provider=lambda: signal,
            drives=DriveSystem(
                substrate_rearm_cooldown_s=0.18,
                substrate_sustain_min_updates=2,
                substrate_sustain_margin=0.22,
            ),
            on_thought=lambda _: thoughts_at.append(time.monotonic()),
        )
        loop.start()
        time.sleep(0.7)
        loop.stop()

        snap = loop.snapshot()
        assert len(thoughts_at) >= 2
        assert all((b - a) >= 0.14 for a, b in zip(thoughts_at, thoughts_at[1:]))
        assert snap["gating"]["substrate_hysteresis_active"] is True
        assert snap["gating"]["substrate_hysteresis_reason"] == "prediction_error"
        assert snap["gating"]["substrate_hysteresis_updates"] >= 2

    def test_background_loop_answers_pending_query(self):
        thoughts: list[ThoughtResult] = []
        cortex = MockCortex()
        loop = ThoughtLoop(
            cortex=cortex,
            tick_interval_ms=50.0,
            min_thought_interval_s=60.0,
            on_thought=lambda t: thoughts.append(t),
        )
        loop.drives.state.curiosity = 0.0
        loop.drives.state.social = 0.0

        loop.start()
        loop.submit_query("How do volcanoes trigger lightning?")
        time.sleep(0.4)
        loop.stop()

        assert len(thoughts) > 0

    def test_background_loop_start_stop(self):
        cortex = MockCortex()
        loop = ThoughtLoop(
            cortex=cortex,
            tick_interval_ms=50.0,
            min_thought_interval_s=0.1,
        )
        loop.drives.update_from_prediction_error(0.7, 0.8, 0.2, 0.1)

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
        loop.drives.update_from_prediction_error(0.7, 0.8, 0.2, 0.1)
        loop.start()
        time.sleep(0.5)
        loop.stop()
        assert len(thoughts) > 0

    def test_stats_tracking(self):
        cortex = MockCortex()
        loop = ThoughtLoop(cortex=cortex, min_thought_interval_s=0.0)
        loop.drives.update_from_prediction_error(0.7, 0.8, 0.2, 0.1)
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

    def test_snapshot_reports_replaceable_non_llm_backend(self):
        loop = ThoughtLoop(cortex=MockCortex(), min_thought_interval_s=0.0)
        backend = loop.snapshot()["cortex_backend"]
        assert backend["backend_kind"] == "deterministic_mock"
        assert backend["llm_backed"] is False
        assert backend["replaceable"] is True
        assert backend["retention_gate"] == "runtime_evidence"


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
