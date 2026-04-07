from __future__ import annotations

import unittest
from types import SimpleNamespace

from hecsn.gap_planner import (
    bank_semantic_relevance_score,
    frontier_gap_plan,
    frontier_gap_terms,
    plan_query_gaps,
)
from hecsn.semantics.frontier import build_bank_query_text


class _FakeMemoryStore:
    def __init__(self) -> None:
        self.slow_raw_windows = [
            "river bank current water",
            "credit loan deposit account",
            "shore reeds current mud",
        ]
        self.slow_importance = [0.5, 1.0, 0.4]
        self.slow_capture_tag = [0.2, 0.9, 0.1]
        self.slow_consolidation_level = [0.8, 0.1, 0.9]

    def _effective_capture_strength(self, idx: int, current_token: int) -> float:
        _ = current_token
        return float(self.slow_capture_tag[idx])


class _FragmentMemoryStore:
    def __init__(self) -> None:
        self.slow_raw_windows = [
            "t-sellers,",
            "rs - Short",
            "rt-sellers",
            "ort-seller",
        ]
        self.slow_importance = [0.8, 0.9, 0.7, 0.6]
        self.slow_capture_tag = [0.8, 0.9, 0.7, 0.6]
        self.slow_consolidation_level = [0.1, 0.2, 0.3, 0.2]

    def _effective_capture_strength(self, idx: int, current_token: int) -> float:
        _ = current_token
        return float(self.slow_capture_tag[idx])


class _PrefixMemoryStore:
    def __init__(self) -> None:
        self.slow_raw_windows = [
            "neut",
            "neutr",
            "neutra",
            "neutral signal",
        ]
        self.slow_importance = [0.8, 0.9, 0.7, 1.0]
        self.slow_capture_tag = [0.8, 0.9, 0.7, 1.0]
        self.slow_consolidation_level = [0.1, 0.2, 0.3, 0.2]

    def _effective_capture_strength(self, idx: int, current_token: int) -> float:
        _ = current_token
        return float(self.slow_capture_tag[idx])


class GapPlannerTests(unittest.TestCase):
    def test_plan_query_gaps_surfaces_unsupported_terms_and_follow_ups(self) -> None:
        plan = plan_query_gaps(
            query_text="river bank loan",
            query_summary={
                "memory_matches": [
                    {"raw_window": "river bank current water", "similarity": 0.84},
                    {"raw_window": "river bank reeds shore", "similarity": 0.78},
                ]
            },
            concept_summary={
                "concepts": [
                    {
                        "label": "river / bank",
                        "top_terms": ["river", "current", "shore"],
                        "uncertainty": 0.82,
                        "drift": 0.18,
                        "match_count": 1,
                        "observations": 1,
                    }
                ]
            },
        )

        self.assertEqual(plan["planner_mode"], "semantic_gap_planner")
        self.assertIn("loan", plan["unsupported_terms"])
        self.assertTrue(plan["follow_up_questions"])
        self.assertTrue(plan["retrieval_queries"])

    def test_frontier_gap_terms_prioritize_unstable_memory(self) -> None:
        terms = frontier_gap_terms(
            memory_store=_FakeMemoryStore(),
            current_token=100,
            limit=4,
        )

        ranked_terms = [item["term"] for item in terms]
        self.assertIn("credit", ranked_terms[:2])
        self.assertIn("loan", ranked_terms[:3])

    def test_frontier_gap_plan_generates_queries_and_followups(self) -> None:
        plan = frontier_gap_plan(
            memory_store=_FakeMemoryStore(),
            current_token=100,
            max_terms=4,
            max_queries=3,
            max_questions=3,
        )

        self.assertEqual(plan["planner_mode"], "frontier_gap_planner")
        self.assertTrue(plan["retrieval_queries"])
        self.assertTrue(plan["follow_up_questions"])

    def test_frontier_gap_plan_reconstructs_phrase_queries_from_fragment_windows(self) -> None:
        plan = frontier_gap_plan(
            memory_store=_FragmentMemoryStore(),
            current_token=100,
            max_terms=4,
            max_queries=3,
            max_questions=3,
        )

        self.assertTrue(plan["frontier_phrases"])
        self.assertTrue(any("short" in query for query in plan["retrieval_queries"]))
        self.assertFalse(any(query in {"rs", "rt", "ort"} for query in plan["retrieval_queries"]))

    def test_frontier_gap_terms_filter_short_prefix_fragments(self) -> None:
        terms = frontier_gap_terms(
            memory_store=_FragmentMemoryStore(),
            current_token=100,
            limit=6,
        )

        ranked_terms = [item["term"] for item in terms]
        self.assertTrue(ranked_terms)
        self.assertIn("short", ranked_terms)
        self.assertFalse(any(term in {"rs", "rt", "ort"} for term in ranked_terms))

    def test_frontier_gap_terms_filter_prefix_chains_when_full_term_exists(self) -> None:
        terms = frontier_gap_terms(
            memory_store=_PrefixMemoryStore(),
            current_token=100,
            limit=6,
        )

        ranked_terms = [item["term"] for item in terms]
        self.assertIn("neutral", ranked_terms)
        self.assertNotIn("neut", ranked_terms)
        self.assertNotIn("neutr", ranked_terms)
        self.assertNotIn("neutra", ranked_terms)

    def test_bank_semantic_relevance_prefers_related_candidate(self) -> None:
        plan = {
            "gap_terms": [
                {"term": "credit", "weight": 2.0},
                {"term": "loan", "weight": 1.5},
                {"term": "deposit", "weight": 1.0},
            ],
            "unsupported_terms": ["credit", "loan"],
            "retrieval_queries": ["credit loan deposit"],
            "follow_up_questions": ["What grounded evidence links credit and loan in current frontier memory?"],
        }
        finance_bank = SimpleNamespace(
            name="finance",
            source="https://example.com/finance",
            train_raw_windows=["credit loan interest bank", "deposit savings account"],
        )
        river_bank = SimpleNamespace(
            name="river",
            source="https://example.com/river",
            train_raw_windows=["river current shore water", "mud reeds stream"],
        )

        self.assertGreater(
            bank_semantic_relevance_score(finance_bank, plan),
            bank_semantic_relevance_score(river_bank, plan),
        )

    def test_bank_semantic_relevance_matches_fragmented_frontier_terms(self) -> None:
        plan = {
            "gap_terms": [
                {"term": "anoth", "weight": 2.0},
                {"term": "offic", "weight": 1.0},
            ],
            "unsupported_terms": ["anoth", "offic"],
            "retrieval_queries": ["anoth offic"],
            "follow_up_questions": ["What grounded evidence is still missing for anoth and offic?"],
        }
        related_bank = SimpleNamespace(
            name="related",
            source="https://example.com/related",
            train_raw_windows=["another office visit", "pipeline office memo"],
        )
        unrelated_bank = SimpleNamespace(
            name="unrelated",
            source="https://example.com/unrelated",
            train_raw_windows=["restaurant dinner menu", "movie review actor"],
        )

        related_score = bank_semantic_relevance_score(related_bank, plan)
        unrelated_score = bank_semantic_relevance_score(unrelated_bank, plan)

        self.assertGreater(related_score, 0.0)
        self.assertGreater(related_score, unrelated_score)

    def test_build_bank_query_text_filters_short_prefix_fragments(self) -> None:
        bank = SimpleNamespace(
            name="frontier",
            probe_raw_windows=[
                "wa wal wall street update",
                "va val value signal",
            ],
        )

        query_text = build_bank_query_text(bank)

        self.assertIn("wall", query_text)
        self.assertIn("street", query_text)
        self.assertIn("value", query_text)
        self.assertNotIn(" wa ", f" {query_text} ")
        self.assertNotIn(" wal ", f" {query_text} ")
        self.assertNotIn(" va ", f" {query_text} ")
        self.assertNotIn(" val ", f" {query_text} ")


if __name__ == "__main__":
    unittest.main()
