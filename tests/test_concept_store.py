from __future__ import annotations

from copy import deepcopy
import unittest

import torch

from hecsn.semantics import ConceptStore


class _FakeMemoryStore:
    def __init__(self) -> None:
        self.slow_routing_keys = [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.95, 0.05, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
        ]


class _FlatMemoryStore:
    def __init__(self) -> None:
        self.slow_routing_keys = [torch.tensor([1.0, 0.0, 0.0]) for _ in range(4)]


class ConceptStoreTests(unittest.TestCase):
    def test_observe_learns_slow_feature_concepts_and_restores(self) -> None:
        store = ConceptStore()
        memory_store = _FakeMemoryStore()
        matches = [
            {"memory_index": 0, "raw_window": "river bank current water", "similarity": 0.82, "importance": 1.0},
            {"memory_index": 1, "raw_window": "river bank shore mud", "similarity": 0.80, "importance": 1.0},
            {"memory_index": 2, "raw_window": "bank account credit deposit", "similarity": 0.78, "importance": 1.0},
        ]

        first = store.observe(query_text="bank river account", memory_matches=matches, memory_store=memory_store)
        second = store.observe(query_text="bank river account", memory_matches=matches, memory_store=memory_store)
        snapshot = store.snapshot()

        self.assertEqual(first["concept_mode"], "slow_feature_concept_memory")
        self.assertEqual(first["abstraction"]["mode"], "online_sfa_proxy")
        self.assertEqual(second["concept_count"], 2)
        self.assertEqual(snapshot["concept_count"], 2)
        self.assertGreaterEqual(snapshot["top_concepts"][0]["observations"], 2)
        self.assertLessEqual(snapshot["top_concepts"][0]["uncertainty"], 1.0)
        self.assertIn("temporal_coherence", snapshot["top_concepts"][0])
        self.assertIn("abstraction_gain", snapshot["top_concepts"][0])

        restored = ConceptStore.from_state_dict(store.state_dict())
        restored_snapshot = restored.snapshot()

        self.assertEqual(restored_snapshot["concept_count"], snapshot["concept_count"])
        self.assertEqual(restored_snapshot["top_concepts"][0]["concept_id"], snapshot["top_concepts"][0]["concept_id"])
        self.assertEqual(restored_snapshot["concept_mode"], "slow_feature_concept_memory")

    def test_observe_prefers_query_overlapping_episodes_and_keeps_cat_dog_separate(self) -> None:
        store = ConceptStore()
        memory_store = _FlatMemoryStore()
        episodes = [
            {
                "memory_indices": [0, 1],
                "text": "a cat purrs when it feels safe.",
                "raw_window": "cat purrs when it feels",
                "similarity": 0.99,
                "importance": 1.0,
                "complete_sentence": 1,
                "clipped_overlap": 0,
            },
            {
                "memory_indices": [2, 3],
                "text": "a dog guards the house and barks at strangers.",
                "raw_window": "dog guards the house and bark",
                "similarity": 0.99,
                "importance": 1.0,
                "complete_sentence": 1,
                "clipped_overlap": 0,
            },
        ]

        cat_summary = store.observe(
            query_text="What purrs when it feels safe?",
            memory_matches=[],
            memory_episodes=episodes,
            memory_store=memory_store,
        )
        dog_summary = store.observe(
            query_text="What guards the house and barks at strangers?",
            memory_matches=[],
            memory_episodes=episodes,
            memory_store=memory_store,
        )
        snapshot = store.snapshot()

        self.assertEqual(cat_summary["concept_count"], 1)
        self.assertIn("cat", cat_summary["concepts"][0]["top_terms"])
        self.assertNotIn("dog", cat_summary["concepts"][0]["top_terms"])
        self.assertEqual(dog_summary["concept_count"], 1)
        self.assertIn("dog", dog_summary["concepts"][0]["top_terms"])
        self.assertNotIn("cat", dog_summary["concepts"][0]["top_terms"])
        self.assertGreaterEqual(snapshot["concept_count"], 2)

    def test_focus_plan_surfaces_persistent_weak_concepts_for_autonomy(self) -> None:
        store = ConceptStore()
        memory_store = _FakeMemoryStore()
        matches = [
            {"memory_index": 0, "raw_window": "submarine ballast tanks control buoyancy", "similarity": 0.82, "importance": 1.0},
            {"memory_index": 1, "raw_window": "submarine ballast shifts buoyancy underwater", "similarity": 0.80, "importance": 1.0},
        ]

        store.observe(query_text="submarine buoyancy ballast", memory_matches=matches, memory_store=memory_store)
        store.observe(query_text="submarine buoyancy ballast", memory_matches=matches, memory_store=memory_store)

        focus_plan = store.focus_plan()

        self.assertIsNotNone(focus_plan)
        assert focus_plan is not None
        self.assertEqual(focus_plan["planner_mode"], "concept_store_abstraction_focus")
        self.assertTrue(focus_plan["weak_concepts"])
        self.assertIn("submarine", " ".join(focus_plan["query_terms"]).lower())
        self.assertIn("submarine", " ".join(focus_plan["retrieval_queries"]).lower())
        self.assertIn("submarine", focus_plan["weak_concepts"][0]["top_terms"])
        self.assertTrue(focus_plan["memory_priority"])
        self.assertIn("structural_growth", focus_plan)

    def test_focus_plan_can_condition_runtime_abstraction_on_query_terms(self) -> None:
        store = ConceptStore()
        memory_store = _FakeMemoryStore()
        matches = [
            {"memory_index": 0, "raw_window": "submarine ballast tanks control buoyancy", "similarity": 0.82, "importance": 1.0},
            {"memory_index": 1, "raw_window": "submarine ballast shifts buoyancy underwater", "similarity": 0.80, "importance": 1.0},
        ]

        store.observe(query_text="submarine buoyancy ballast", memory_matches=matches, memory_store=memory_store)
        store.observe(query_text="submarine buoyancy ballast", memory_matches=matches, memory_store=memory_store)

        focus_plan = store.focus_plan(query_text="submarine control depth", min_observations=1)

        self.assertIsNotNone(focus_plan)
        assert focus_plan is not None
        self.assertIn("submarine", " ".join(focus_plan["focus_terms"]).lower())
        self.assertIn("ballast", " ".join(focus_plan["retrieval_queries"]).lower())
        self.assertIn("0", focus_plan["memory_priority"])
        self.assertIn("1", focus_plan["memory_priority"])

    def test_persistent_weak_concepts_expand_structural_capacity(self) -> None:
        store = ConceptStore(slow_feature_dim=1, max_slow_feature_dim=3)
        memory_store = _FakeMemoryStore()
        matches = [
            {"memory_index": 0, "raw_window": "submarine ballast tanks control buoyancy", "similarity": 0.82, "importance": 1.0},
            {"memory_index": 1, "raw_window": "submarine ballast shifts buoyancy underwater", "similarity": 0.80, "importance": 1.0},
        ]

        for _ in range(4):
            store.observe(query_text="submarine buoyancy ballast", memory_matches=matches, memory_store=memory_store)

        snapshot = store.snapshot()
        restored = ConceptStore.from_state_dict(store.state_dict()).snapshot()

        self.assertGreaterEqual(snapshot["abstraction"]["requested_output_dim"], 2)
        self.assertGreater(snapshot["growth"]["expansion_events"], 0)
        self.assertGreater(snapshot["top_concepts"][0]["split_bias"], 0.0)
        self.assertEqual(restored["growth"]["expansion_events"], snapshot["growth"]["expansion_events"])
        self.assertEqual(restored["abstraction"]["requested_output_dim"], snapshot["abstraction"]["requested_output_dim"])

    def test_refresh_structural_capacity_prunes_redundant_stable_concepts(self) -> None:
        store = ConceptStore()
        memory_store = _FlatMemoryStore()
        episodes = [
            {
                "memory_indices": [0, 1],
                "text": "a cat purrs when it feels safe.",
                "raw_window": "cat purrs when it feels",
                "similarity": 0.99,
                "importance": 1.0,
                "complete_sentence": 1,
                "clipped_overlap": 0,
            },
            {
                "memory_indices": [2, 3],
                "text": "a dog guards the house and barks at strangers.",
                "raw_window": "dog guards the house and bark",
                "similarity": 0.99,
                "importance": 1.0,
                "complete_sentence": 1,
                "clipped_overlap": 0,
            },
        ]

        store.observe(query_text="What purrs when it feels safe?", memory_matches=[], memory_episodes=episodes, memory_store=memory_store)
        store.observe(query_text="What guards the house and barks at strangers?", memory_matches=[], memory_episodes=episodes, memory_store=memory_store)

        duplicate = deepcopy(store._entries["c1"])
        duplicate["concept_id"] = "c999"
        duplicate["observations"] = 1
        duplicate["match_count_total"] = 1
        duplicate["score_total"] = 0.5
        duplicate["growth_pressure_ema"] = 0.1
        duplicate["growth_streak"] = 0
        duplicate["split_bias"] = 0.0
        store._entries["c999"] = duplicate
        store._episode_index = 8

        report = store.refresh_structural_capacity()

        self.assertEqual(store.snapshot()["concept_count"], 2)
        self.assertGreater(report["prune_events"], 0)
        self.assertNotIn("c999", store._entries)


if __name__ == "__main__":
    unittest.main()
