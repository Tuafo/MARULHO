"""Phase 5 tests: prediction error as core signal + active exploration.

Covers:
  1. DriveSystem predictive-processing updates
  2. Active exploration target selection and consumption
  3. ThoughtLoop prediction-error-triggered thinking
  4. Snapshot telemetry for exploration state and neuromodulation
"""

from __future__ import annotations

from hecsn.cortex.core import MockCortex, ThoughtDepth, ThoughtResult
from hecsn.cortex.drives import DriveSystem, ThalamicGate
from hecsn.cortex.episodic_memory import EpisodicMemory
from hecsn.cortex.thought_loop import ThoughtLoop


class TestPredictiveDriveUpdates:
    def test_surprise_update_derives_channelized_neuromodulation(self):
        drives = DriveSystem()
        drives.update_from_surprise(
            dopamine=0.9,
            serotonin=0.2,
            norepinephrine=0.8,
            acetylcholine=0.7,
        )
        assert drives.state.da_reward > 0.5
        assert drives.state.da_novelty > 0.5
        assert drives.state.da_salience > 0.5
        assert drives.state.ne_alerting > 0.5
        assert drives.state.ne_orienting > 0.5
        assert drives.state.ach_learning > 0.5
        assert drives.state.ach_attention > 0.5
        assert drives.state.serotonin_patience >= 0.0

    def test_high_prediction_error_increases_curiosity_and_anxiety(self):
        drives = DriveSystem()
        drives.state.curiosity = 0.1
        drives.state.anxiety = 0.0
        drives.update_from_prediction_error(
            prediction_error_mean=0.7,
            prediction_error_max=0.9,
            predictive_confidence_mean=0.2,
            predictive_confidence_min=0.1,
        )
        assert drives.state.prediction_error > 0.0
        assert drives.state.uncertainty > 0.5
        assert drives.state.curiosity > 0.1
        assert drives.state.anxiety > 0.1
        assert drives.state.exploration_urgency > 0.3
        assert drives.state.da_novelty > 0.5
        assert drives.state.da_salience > 0.5
        assert drives.state.ne_alerting > 0.5
        assert drives.state.ach_learning > 0.5

    def test_low_error_high_confidence_raises_boredom(self):
        drives = DriveSystem()
        drives.update_from_prediction_error(
            prediction_error_mean=0.02,
            prediction_error_max=0.05,
            predictive_confidence_mean=0.95,
            predictive_confidence_min=0.9,
        )
        assert drives.state.boredom > 0.1
        assert drives.state.satisfaction >= 0.3
        assert drives.state.serotonin_patience > 0.5

    def test_prediction_error_can_trigger_thinking(self):
        drives = DriveSystem()
        drives.state.curiosity = 0.0
        drives.state.anxiety = 0.0
        drives.state.social = 0.0
        drives.update_from_prediction_error(
            prediction_error_mean=0.6,
            prediction_error_max=0.7,
            predictive_confidence_mean=0.3,
            predictive_confidence_min=0.2,
        )
        assert drives.should_think()


class TestActiveExplorationGate:
    def test_gate_consumes_active_exploration_target(self):
        mem = EpisodicMemory(capacity=100)
        drives = DriveSystem()
        gate = ThalamicGate(mem, drives)
        gate.set_active_exploration_target(
            "reef chemistry",
            reason="prediction_error",
            score=0.8,
        )

        packet = gate.assemble()
        assert packet.forced_topic == "reef chemistry"
        assert gate.active_exploration_target == ""

    def test_gate_normalizes_exploration_target_text(self):
        mem = EpisodicMemory(capacity=100)
        drives = DriveSystem()
        gate = ThalamicGate(mem, drives)
        gate.set_active_exploration_target(
            "bears / claw | adaptation",
            reason="prediction_error",
            score=0.7,
        )
        assert gate.active_exploration_target == "bears claw adaptation"


class TestThoughtLoopExploration:
    def test_signal_refresh_sets_exploration_target(self):
        memory = EpisodicMemory(capacity=100)
        memory.store(
            "Reef chemistry changes as ocean acidification lowers carbonate availability.",
            topics=["reef chemistry", "ocean acidification"],
            salience=0.9,
        )
        loop = ThoughtLoop(
            cortex=MockCortex(),
            memory=memory,
            signal_provider=lambda: {
                "prediction_error_mean": 0.45,
                "prediction_error_max": 0.6,
                "predictive_confidence_mean": 0.35,
                "predictive_confidence_min": 0.2,
                "recent_concepts": ["reef chemistry", "ocean acidification"],
                "concept_candidates": [
                    {
                        "label": "reef chemistry",
                        "top_terms": ["reef", "chemistry"],
                        "match_count": 3,
                        "observations": 3,
                        "uncertainty": 0.25,
                        "temporal_coherence": 0.7,
                        "example_windows": ["Reef chemistry changes under ocean acidification."],
                    },
                    {
                        "label": "ocean acidification",
                        "top_terms": ["ocean", "acidification"],
                        "match_count": 3,
                        "observations": 3,
                        "uncertainty": 0.3,
                        "temporal_coherence": 0.7,
                        "example_windows": ["Ocean acidification lowers carbonate availability."],
                    },
                ],
            },
            min_thought_interval_s=0.0,
        )
        loop._refresh_cognitive_signals()
        snap = loop.snapshot()
        assert snap["active_exploration"]["target"] in {"reef chemistry", "ocean acidification"}
        assert snap["active_exploration"]["source"] == "snn"
        assert snap["drives"]["prediction_error"] > 0.0
        assert snap["drives"]["exploration_urgency"] > 0.0

    def test_grounded_concept_candidate_beats_fragmented_snn_label(self):
        memory = EpisodicMemory(capacity=100)
        memory.store(
            "Plate tectonics is driven by convection currents in the mantle.",
            topics=["plate tectonics"],
            salience=0.9,
        )
        loop = ThoughtLoop(
            cortex=MockCortex(),
            memory=memory,
            signal_provider=lambda: {
                "prediction_error_mean": 0.48,
                "prediction_error_max": 0.62,
                "predictive_confidence_mean": 0.32,
                "predictive_confidence_min": 0.22,
                "recent_concepts": ["bears / claw", "plate tectonics"],
                "concept_candidates": [
                    {
                        "label": "bears / claw",
                        "top_terms": ["bears", "claw"],
                        "match_count": 2,
                        "observations": 2,
                        "uncertainty": 0.9,
                        "temporal_coherence": 0.1,
                        "example_windows": ["Bears use claws for climbing and defense."],
                    },
                    {
                        "label": "plate tectonics",
                        "top_terms": ["plate", "tectonics"],
                        "match_count": 5,
                        "observations": 5,
                        "uncertainty": 0.2,
                        "temporal_coherence": 0.8,
                        "example_windows": ["Plate tectonics is driven by convection currents in the mantle."],
                    },
                ],
            },
            min_thought_interval_s=0.0,
        )
        loop._refresh_cognitive_signals()
        snap = loop.snapshot()
        assert snap["active_exploration"]["target"] == "plate tectonics"
        assert snap["active_exploration"]["source"] == "snn"

    def test_low_confidence_thought_sets_next_exploration_target(self):
        loop = ThoughtLoop(cortex=MockCortex(), min_thought_interval_s=0.0)
        result = ThoughtResult(
            raw_text="",
            thought="Maybe coral chemistry changes under heat stress.",
            topics=("coral chemistry", "heat stress"),
            confidence=0.3,
            parse_success=True,
        )
        loop._post_process_thought(result, ThoughtDepth.QUICK)
        snap = loop.snapshot()
        assert snap["active_exploration"]["target"] in {"coral chemistry", "heat stress"}
        assert snap["active_exploration"]["reason"] in {"low_confidence_thought", "prediction_error"}

    def test_step_thinks_from_prediction_error_even_with_low_curiosity(self):
        loop = ThoughtLoop(
            cortex=MockCortex(),
            signal_provider=lambda: {
                "prediction_error_mean": 0.55,
                "prediction_error_max": 0.7,
                "predictive_confidence_mean": 0.25,
                "predictive_confidence_min": 0.2,
                "recent_concepts": ["aurora borealis"],
            },
            min_thought_interval_s=0.0,
        )
        loop.drives.state.curiosity = 0.0
        loop.drives.state.anxiety = 0.0
        loop.drives.state.social = 0.0
        result = loop.step(force=True)
        assert result is not None
        assert loop.stats.thoughts_generated == 1

    def test_ne_alerting_can_trigger_deep_depth(self):
        loop = ThoughtLoop(cortex=MockCortex(), min_thought_interval_s=0.0)
        loop.drives.state.ne_alerting = 0.9
        loop.drives.state.da_salience = 0.7
        assert loop._choose_depth() == ThoughtDepth.DEEP
        assert loop.snapshot()["depth_policy"]["last_reason"] == "ne_alerting"

    def test_ach_attention_can_trigger_standard_depth(self):
        loop = ThoughtLoop(cortex=MockCortex(), min_thought_interval_s=0.0)
        loop.drives.state.ach_attention = 0.8
        loop.drives.state.da_novelty = 0.6
        assert loop._choose_depth() == ThoughtDepth.STANDARD
        assert loop.snapshot()["depth_policy"]["last_reason"] == "ach_attention"

    def test_pending_wake_tension_becomes_exploration_target(self):
        loop = ThoughtLoop(cortex=MockCortex(), min_thought_interval_s=0.0)
        loop._queue_wake_tension(
            "Dream contradiction: reef growth does not match the hypothesized mechanism.",
            topics=["reef growth", "mechanism"],
            salience=0.9,
        )
        loop._update_active_exploration_target()
        snap = loop.snapshot()
        assert snap["active_exploration"]["target"] == "reef growth"
        assert snap["active_exploration"]["reason"] == "wake_tension"

    def test_stale_snn_target_can_be_replaced_by_recent_cortex_topic(self):
        loop = ThoughtLoop(cortex=MockCortex(), min_thought_interval_s=0.0)
        loop._set_active_exploration_target(
            "bears claw",
            reason="prediction_error",
            source="snn",
            score=0.72,
        )
        loop._exploration_state.updated_at -= 20.0
        loop._update_active_exploration_target(
            ThoughtResult(
                raw_text="",
                thought="Origami mathematics uses symmetry and folding to create complex structures.",
                topics=("origami mathematics", "symmetry"),
                confidence=0.3,
                parse_success=True,
            )
        )
        snap = loop.snapshot()
        assert snap["active_exploration"]["target"] in {"origami mathematics", "symmetry"}
        assert snap["active_exploration"]["source"] == "cortex"

    def test_snapshot_exposes_active_exploration(self):
        loop = ThoughtLoop(cortex=MockCortex(), min_thought_interval_s=0.0)
        loop.gate.set_active_exploration_target(
            "plate tectonics",
            reason="prediction_error",
            source="snn",
            score=0.72,
        )
        loop._exploration_state.target = "plate tectonics"
        loop._exploration_state.reason = "prediction_error"
        loop._exploration_state.source = "snn"
        loop._exploration_state.score = 0.72
        snap = loop.snapshot()
        assert snap["active_exploration"]["target"] == "plate tectonics"
        assert snap["active_exploration"]["score"] == 0.72

    def test_snapshot_exposes_channelized_neuromodulation(self):
        loop = ThoughtLoop(cortex=MockCortex(), min_thought_interval_s=0.0)
        loop.drives.update_from_surprise(
            dopamine=0.8,
            serotonin=0.3,
            norepinephrine=0.9,
            acetylcholine=0.7,
        )
        loop.drives.update_from_prediction_error(
            prediction_error_mean=0.6,
            prediction_error_max=0.8,
            predictive_confidence_mean=0.3,
            predictive_confidence_min=0.2,
        )
        snap = loop.snapshot()
        neuromod = snap["neuromodulation"]
        assert neuromod["da_salience"] > 0.5
        assert neuromod["ne_alerting"] > 0.5
        assert neuromod["ach_attention"] > 0.5
        assert neuromod["serotonin_patience"] >= 0.0
