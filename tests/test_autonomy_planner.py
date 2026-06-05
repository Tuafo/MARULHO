from __future__ import annotations

from pathlib import Path
import unittest

from marulho.service.autonomy_planner import AutonomyPlanner


class _FakeInteractionPipeline:
    def __init__(self, gaps: list[dict[str, object]]) -> None:
        self._gaps = gaps

    def recent_query_gaps(self) -> list[dict[str, object]]:
        return list(self._gaps)


class _FakeConceptStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def focus_plan(self, *, query_text: str = "", min_observations: int = 0) -> dict[str, object]:
        self.calls.append({"query_text": query_text, "min_observations": min_observations})
        return {
            "planner_mode": "concept_store_abstraction_focus",
            "query_terms": ["domestication", "cats"],
            "unsupported_terms": ["birds"],
            "gap_terms": [{"term": "domestication", "weight": 0.3}],
            "retrieval_queries": ["cats domestication"],
            "follow_up_questions": ["Why did cats domesticate?"],
            "weak_concepts": [
                {
                    "label": "cats",
                    "weakness": 0.5,
                    "uncertainty": 0.4,
                    "drift": 0.1,
                    "top_terms": ["felines"],
                }
            ],
            "structural_growth": {"expansion_events": 1},
        }


class _FakeGeometricCuriosity:
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    def focus_plan(self, *, query_text: str | None = None) -> dict[str, object]:
        self.calls.append(query_text)
        return {
            "planner_mode": "geometric_abstraction_gap_focus",
            "query_terms": ["phase transitions"],
            "geometric_gaps": [{"concept": "phase transitions"}],
        }


class _PlannerManager:
    def __init__(self) -> None:
        self._interaction_pipeline = _FakeInteractionPipeline(
            [
                {
                    "query_text": "cats chase mice",
                    "unsupported_terms": ["mice"],
                    "gap_terms": [{"term": "cats", "weight": 0.8}],
                    "retrieval_queries": ["cats chase mice"],
                    "weak_concepts": [
                        {
                            "label": "cats",
                            "weakness": 0.7,
                            "uncertainty": 0.6,
                            "drift": 0.2,
                            "top_terms": ["felines"],
                        }
                    ],
                }
            ]
        )
        self._concept_store = _FakeConceptStore()
        self._geometric_curiosity = _FakeGeometricCuriosity()
        self._brain_config = {
            "autonomy": {
                "provider_curriculum": {
                    "wikipedia": {
                        "attempts": 3,
                        "commits": 2,
                        "successes": 2,
                        "gap_gain_ema": 0.2,
                        "diagnostic_gain_ema": 0.5,
                        "semantic_relevance_ema": 0.7,
                        "answerability_gain_ema": 0.8,
                        "uncertainty_reduction_ema": 0.6,
                        "weak_concept_stabilization_ema": 0.5,
                        "utility_ema": 0.6,
                        "focus_alignment_ema": 0.5,
                        "grounded_outcome_ema": 0.6,
                        "grounded_family_summary_ema": 0.55,
                        "delayed_consequence_ema": 0.1,
                        "contradiction_decay_ema": 0.0,
                        "topic_terms": {"cats": 1.0, "mice": 0.5},
                        "topic_families": {
                            "cats": {
                                "commits": 2,
                                "successes": 2,
                                "semantic_relevance_ema": 0.8,
                                "answerability_gain_ema": 0.7,
                                "uncertainty_reduction_ema": 0.6,
                                "weak_concept_stabilization_ema": 0.4,
                                "last_selected_at": "2026-05-10T00:00:00+00:00",
                            }
                        },
                        "query_families": {
                            "cats chase mice": {
                                "commits": 2,
                                "successes": 2,
                                "semantic_relevance_ema": 0.9,
                                "answerability_gain_ema": 0.8,
                                "uncertainty_reduction_ema": 0.7,
                                "weak_concept_stabilization_ema": 0.6,
                                "last_selected_at": "2026-05-10T00:00:00+00:00",
                            }
                        },
                    },
                    "arxiv": {
                        "attempts": 1,
                        "commits": 0,
                        "successes": 0,
                        "semantic_relevance_ema": 0.05,
                        "answerability_gain_ema": 0.0,
                        "uncertainty_reduction_ema": 0.0,
                        "weak_concept_stabilization_ema": 0.0,
                        "utility_ema": 0.0,
                        "focus_alignment_ema": 0.0,
                        "grounded_outcome_ema": 0.0,
                        "grounded_family_summary_ema": 0.0,
                        "topic_terms": {"astronomy": 1.0},
                        "topic_families": {},
                        "query_families": {},
                    },
                }
            }
        }

    def _normalize_provider_curriculum(self, value):
        return dict(value or {})


class AutonomyPlannerTests(unittest.TestCase):
    def test_planner_no_longer_uses_manager_bound_transition_base(self) -> None:
        source = Path("src/marulho/service/autonomy_planner.py").read_text(encoding="utf-8")

        self.assertNotIn("ExplicitOwnerModule", source)
        self.assertNotIn("install_owner_forwarders", source)
        self.assertNotIn('"_manager"', source)
        self.assertNotIn("'_manager'", source)

    def test_planner_has_no_mixin_base(self) -> None:
        mro_names = {base.__name__ for base in AutonomyPlanner.__mro__}
        self.assertFalse(
            {name for name in mro_names if name.endswith("Mixin")},
            "AutonomyPlanner must not inherit active mixin-named bases",
        )

    def test_terminus_autonomy_core_is_not_mixin_named(self) -> None:
        source = Path("src/marulho/service/terminus_autonomy.py").read_text(encoding="utf-8")
        planner_source = Path("src/marulho/service/autonomy_planner.py").read_text(encoding="utf-8")
        self.assertNotIn("class TerminusAutonomyMixin", source)
        self.assertNotIn("TerminusAutonomyMixin", planner_source)
        self.assertIn("class TerminusAutonomyCore", source)

    def test_focus_plan_merges_interaction_concept_and_geometric_signals(self) -> None:
        manager = _PlannerManager()
        planner = AutonomyPlanner(manager)

        plan = planner._autonomy_focus_plan_locked()

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan["planner_mode"], "merged_runtime_abstraction_focus")
        self.assertIn("cats", plan["query_terms"])
        self.assertIn("mice", plan["unsupported_terms"])
        self.assertIn("cats domestication", plan["retrieval_queries"])
        self.assertIn("phase transitions", plan["query_terms"])
        self.assertIn({"concept": "phase transitions"}, plan["geometric_gaps"])
        self.assertEqual(manager._concept_store.calls[0]["min_observations"], 1)
        self.assertIn("cats chase mice", str(manager._concept_store.calls[0]["query_text"]))
        self.assertEqual(manager._geometric_curiosity.calls[0], manager._concept_store.calls[0]["query_text"])

    def test_candidate_specs_and_shortlist_use_focus_signals(self) -> None:
        manager = _PlannerManager()
        planner = AutonomyPlanner(manager)
        focus_plan = {
            "query_terms": ["cats", "mice"],
            "unsupported_terms": ["submarine"],
            "gap_terms": [{"term": "cats", "weight": 0.8}],
            "retrieval_queries": ["cats chase mice"],
            "follow_up_questions": ["Why do cats chase mice?"],
            "weak_concepts": [
                {
                    "label": "cats",
                    "weakness": 0.7,
                    "uncertainty": 0.6,
                    "drift": 0.2,
                    "top_terms": ["felines"],
                }
            ],
        }
        candidate_bank = [
            {
                "catalog_mode": "live_remote_search",
                "catalog_providers": ["wikipedia"],
                "catalog_queries_per_provider": 1,
                "catalog_provider_result_limit": 1,
                "catalog_limit": 1,
                "catalog_focus_text": "",
            },
            {"metadata": {"query_text": "none"}},
        ]

        specs = planner._autonomy_candidate_specs_locked(candidate_bank=candidate_bank, focus_plan=focus_plan)
        shortlist_size, shortlist_gap_weight, shortlist_affinity_weight = planner._autonomy_shortlist_settings_locked(
            candidate_bank=specs,
            config={},
            focus_plan=focus_plan,
        )

        self.assertIn("cats", specs[0]["catalog_focus_text"])
        self.assertIn("submarine", specs[0]["catalog_focus_text"])
        self.assertIn("cats chase mice", specs[0]["catalog_focus_text"])
        self.assertEqual(specs[0]["catalog_queries_per_provider"], 4)
        self.assertIn("cats", specs[1]["metadata"]["query_text"])
        self.assertIn("mice", specs[1]["metadata"]["query_text"])
        self.assertIn("submarine", specs[1]["metadata"]["query_text"])
        self.assertEqual(shortlist_size, 3)
        self.assertAlmostEqual(shortlist_gap_weight, 0.2)
        self.assertAlmostEqual(shortlist_affinity_weight, 0.8)

    def test_provider_curriculum_snapshot_scores_query_families(self) -> None:
        manager = _PlannerManager()
        planner = AutonomyPlanner(manager)
        focus_plan = {
            "query_terms": ["cats", "mice"],
            "unsupported_terms": ["submarine"],
            "gap_terms": [{"term": "cats", "weight": 0.9}],
            "retrieval_queries": ["cats chase mice"],
            "follow_up_questions": ["Why do cats chase mice?"],
            "weak_concepts": [
                {
                    "label": "cats",
                    "weakness": 0.7,
                    "uncertainty": 0.6,
                    "drift": 0.2,
                    "top_terms": ["felines"],
                }
            ],
            "geometric_gaps": [{"concept": "cats"}],
        }
        autonomy = manager._brain_config["autonomy"]

        snapshot = planner._provider_curriculum_snapshot_locked(autonomy, focus_plan)

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["ranked_providers"][0]["provider"], "wikipedia")
        self.assertGreater(snapshot["ranked_providers"][0]["query_family_focus_score"], 0.0)
        self.assertGreater(snapshot["ranked_providers"][0]["query_family_scores"]["cats chase mice"], 0.0)
        self.assertIn("cat", snapshot["focus_terms"])


if __name__ == "__main__":
    unittest.main()
