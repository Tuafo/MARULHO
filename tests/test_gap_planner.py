from __future__ import annotations

import unittest
from types import SimpleNamespace

from marulho import gap_planner
from marulho.gap_planner import (
    bank_semantic_relevance_score,
    frontier_gap_plan,
    plan_query_gaps,
)
from marulho.semantics.frontier import build_bank_query_text


class _FrontierCollectorMixin:
    _row_texts: list[str]

    def collect_frontier_gap_indices(self, *, current_token: int, max_candidates: int, scope: str) -> dict[str, object]:
        _ = current_token
        limit = max(0, int(max_candidates))
        indices = list(range(min(len(self._row_texts), limit)))
        return {
            "surface": "bounded_frontier_gap_candidates.v1",
            "status": "collected" if indices else "empty",
            "scope": str(scope),
            "memory_size": len(self._row_texts),
            "current_token": int(current_token),
            "requested_count": int(max_candidates),
            "candidate_window_limit": int(max_candidates),
            "candidate_window_policy": "test_bounded_fixture_window",
            "candidate_scope": "test_bounded_fixture_window",
            "candidate_bucket_ids": [],
            "candidate_bucket_count": 0,
            "candidate_index_available_count": len(self._row_texts),
            "candidate_index_available_count_is_lower_bound": len(self._row_texts) > len(indices),
            "candidate_index_count": len(indices),
            "candidate_indices": indices,
            "global_score_scan": False,
            "global_candidate_scan": False,
            "runs_live_tick": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "fallback_reason": None,
        }

    def query_match_row(
        self,
        index: int,
        current_token: int | None = None,
        *,
        include_text_payload: bool = False,
    ) -> dict[str, object]:
        _ = current_token
        idx = int(index)
        raw_window = self._row_texts[idx]
        capture = float(self._row_capture_tag[idx])
        row: dict[str, object] = {
            "surface": "bounded_query_memory_match_row.v1",
            "memory_index": idx,
            "read_only": True,
            "importance": float(self._row_importance[idx]),
            "capture_tag": capture,
            "capture_strength": capture,
            "consolidation_level": float(self._row_consolidation_level[idx]),
            "raw_window": None,
            "text": None,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
        }
        if include_text_payload:
            row.update(
                {
                    "raw_window": raw_window,
                    "text": raw_window,
                    "raw_text_payload_loaded": bool(raw_window),
                }
            )
        return row


class _FakeMemoryStore(_FrontierCollectorMixin):
    def __init__(self) -> None:
        self._row_texts = [
            "river bank current water",
            "credit loan deposit account",
            "shore reeds current mud",
        ]
        self._row_importance = [0.5, 1.0, 0.4]
        self._row_capture_tag = [0.2, 0.9, 0.1]
        self._row_consolidation_level = [0.8, 0.1, 0.9]


class _FragmentMemoryStore(_FrontierCollectorMixin):
    def __init__(self) -> None:
        self._row_texts = [
            "t-sellers,",
            "rs - Short",
            "rt-sellers",
            "ort-seller",
        ]
        self._row_importance = [0.8, 0.9, 0.7, 0.6]
        self._row_capture_tag = [0.8, 0.9, 0.7, 0.6]
        self._row_consolidation_level = [0.1, 0.2, 0.3, 0.2]


class _PrefixMemoryStore(_FrontierCollectorMixin):
    def __init__(self) -> None:
        self._row_texts = [
            "neut",
            "neutr",
            "neutra",
            "neutral signal",
        ]
        self._row_importance = [0.8, 0.9, 0.7, 1.0]
        self._row_capture_tag = [0.8, 0.9, 0.7, 1.0]
        self._row_consolidation_level = [0.1, 0.2, 0.3, 0.2]


class _MissingCollectorMemoryStore:
    def __init__(self) -> None:
        self.size = 65_536

    def live_summary_stats(self, current_token: int | None = None) -> dict[str, object]:
        _ = current_token
        return {"size": self.size}


class _BoundedFrontierMemoryStore:
    def __init__(self) -> None:
        self.size = 65_536
        self.selected_indices = [65_520, 12, 65_534]
        self.collect_calls = 0
        self._row_text_by_index = {
            65_520: "credit loan deposit account",
            12: "river bank current water",
            65_534: "shore reeds current mud",
        }
        self._row_importance_by_index = {65_520: 1.0, 12: 0.4, 65_534: 0.3}
        self._row_capture_by_index = {65_520: 0.9, 12: 0.2, 65_534: 0.1}
        self._row_consolidation_by_index = {65_520: 0.1, 12: 0.8, 65_534: 0.9}

    def collect_frontier_gap_indices(self, *, current_token: int, max_candidates: int, scope: str) -> dict[str, object]:
        self.collect_calls += 1
        self.requested_count = int(max_candidates)
        return {
            "surface": "bounded_frontier_gap_candidates.v1",
            "status": "collected",
            "scope": str(scope),
            "memory_size": self.size,
            "current_token": int(current_token),
            "requested_count": int(max_candidates),
            "candidate_window_limit": int(max_candidates),
            "candidate_window_policy": "test_bounded_collector",
            "candidate_scope": "test_bounded_collector",
            "candidate_bucket_ids": [],
            "candidate_bucket_count": 0,
            "candidate_index_available_count": len(self.selected_indices),
            "candidate_index_available_count_is_lower_bound": False,
            "candidate_index_count": len(self.selected_indices),
            "candidate_indices": list(self.selected_indices),
            "global_score_scan": False,
            "global_candidate_scan": False,
            "runs_live_tick": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "fallback_reason": None,
        }

    def query_match_row(
        self,
        index: int,
        current_token: int | None = None,
        *,
        include_text_payload: bool = False,
    ) -> dict[str, object]:
        _ = current_token
        idx = int(index)
        raw_window = self._row_text_by_index.get(idx, "")
        capture = float(self._row_capture_by_index.get(idx, 0.0))
        row: dict[str, object] = {
            "surface": "bounded_query_memory_match_row.v1",
            "memory_index": idx,
            "read_only": True,
            "importance": float(self._row_importance_by_index.get(idx, 0.0)),
            "capture_tag": capture,
            "capture_strength": capture,
            "consolidation_level": float(self._row_consolidation_by_index.get(idx, 0.0)),
            "raw_window": None,
            "text": None,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
        }
        if include_text_payload:
            row.update(
                {
                    "raw_window": raw_window,
                    "text": raw_window,
                    "raw_text_payload_loaded": bool(raw_window),
                }
            )
        return row


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

    def test_plan_query_gaps_matches_boundary_free_query_against_spaced_evidence(self) -> None:
        plan = plan_query_gaps(
            query_text="submarineballastcontrol",
            query_summary={
                "memory_matches": [
                    {
                        "raw_window": "submarine ballast control regulates buoyancy underwater",
                        "similarity": 0.91,
                    }
                ]
            },
            concept_summary={"concepts": []},
        )

        self.assertEqual(plan["query_terms"], ["submarineballastcontrol"])
        self.assertEqual(plan["unsupported_terms"], [])
        self.assertAlmostEqual(float(plan["grounded_fraction"]), 1.0, places=6)

    def test_plan_query_gaps_supports_unsegmented_character_stream_terms(self) -> None:
        plan = plan_query_gaps(
            query_text="submarineballast",
            query_summary={
                "memory_matches": [
                    {
                        "raw_window": "submarines regulate buoyancy with ballast tanks",
                        "similarity": 0.92,
                    }
                ]
            },
            concept_summary={"concepts": []},
        )

        self.assertEqual(plan["planner_mode"], "semantic_gap_planner")
        self.assertEqual(plan["query_terms"], ["submarineballast"])
        self.assertEqual(plan["unsupported_terms"], [])
        self.assertGreater(plan["grounded_fraction"], 0.0)

    def test_plan_query_gaps_derives_chunked_retrieval_query_from_boundary_free_match(self) -> None:
        plan = plan_query_gaps(
            query_text="submarineballastcontrol",
            query_summary={
                "memory_matches": [
                    {
                        "raw_window": "submarine ballast control regulates buoyancy underwater",
                        "similarity": 0.91,
                    }
                ]
            },
            concept_summary={"concepts": []},
        )

        self.assertTrue(plan["retrieval_queries"])
        self.assertTrue(
            any(query.startswith("submarine ballast control") for query in plan["retrieval_queries"])
        )

    def test_frontier_gap_plan_prioritizes_unstable_memory_with_report(self) -> None:
        plan = frontier_gap_plan(
            memory_store=_FakeMemoryStore(),
            current_token=100,
            max_terms=4,
        )
        terms = plan["gap_terms"]

        ranked_terms = [item["term"] for item in terms]
        self.assertIn("credit", ranked_terms[:2])
        self.assertIn("loan", ranked_terms[:3])
        self.assertEqual(
            plan["frontier_selection_report"]["surface"],
            "bounded_frontier_gap_selection.v1",
        )

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

    def test_frontier_gap_plan_uses_bounded_collector_without_archive_materialization(self) -> None:
        store = _BoundedFrontierMemoryStore()

        plan = frontier_gap_plan(
            memory_store=store,
            current_token=65_536,
            max_terms=4,
            max_queries=3,
            max_questions=3,
            top_entries=4,
        )

        ranked_terms = [item["term"] for item in plan["gap_terms"]]
        report = plan["frontier_selection_report"]
        self.assertEqual(store.collect_calls, 1)
        self.assertLessEqual(store.requested_count, 32)
        self.assertIn("credit", ranked_terms[:2])
        self.assertIn("loan", ranked_terms[:3])
        self.assertEqual(report["candidate_index_count"], len(store.selected_indices))
        self.assertEqual(report["selected_indices"][0], 65_520)
        self.assertFalse(report["global_candidate_scan"])
        self.assertFalse(report["global_score_scan"])
        self.assertTrue(report["raw_text_payload_loaded"])
        self.assertEqual(report["frontier_row_surface"], "bounded_query_memory_match_row.v1")
        self.assertTrue(report["frontier_row_reader_owned_by_store"])
        self.assertEqual(report["frontier_row_read_count"], len(store.selected_indices))
        self.assertTrue(report["direct_slow_memory_array_reads_retired"])
        self.assertFalse(report["effective_capture_reader_used"])
        self.assertFalse(report["stc_state_advance"])
        self.assertFalse(report["language_reasoning"])

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

    def test_frontier_gap_plan_filters_short_prefix_fragments(self) -> None:
        plan = frontier_gap_plan(
            memory_store=_FragmentMemoryStore(),
            current_token=100,
            max_terms=6,
        )
        terms = plan["gap_terms"]

        ranked_terms = [item["term"] for item in terms]
        self.assertTrue(ranked_terms)
        self.assertIn("short", ranked_terms)
        self.assertFalse(any(term in {"rs", "rt", "ort"} for term in ranked_terms))

    def test_frontier_gap_plan_filters_prefix_chains_when_full_term_exists(self) -> None:
        plan = frontier_gap_plan(
            memory_store=_PrefixMemoryStore(),
            current_token=100,
            max_terms=6,
        )
        terms = plan["gap_terms"]

        ranked_terms = [item["term"] for item in terms]
        self.assertIn("neutral", ranked_terms)
        self.assertNotIn("neut", ranked_terms)
        self.assertNotIn("neutr", ranked_terms)
        self.assertNotIn("neutra", ranked_terms)

    def test_frontier_gap_plan_requires_bounded_collector(self) -> None:
        plan = frontier_gap_plan(
            memory_store=_MissingCollectorMemoryStore(),
            current_token=65_536,
            max_terms=4,
            top_entries=4,
        )
        report = plan["frontier_selection_report"]

        self.assertEqual(plan["gap_terms"], [])
        self.assertEqual(report["surface"], "bounded_frontier_gap_selection.v1")
        self.assertEqual(
            report["fallback_reason"],
            "memory_store_missing_bounded_frontier_collector",
        )
        self.assertEqual(report["candidate_index_count"], 0)
        self.assertFalse(report["raw_text_payload_loaded"])
        self.assertFalse(report["global_candidate_scan"])
        self.assertFalse(report["global_score_scan"])
        self.assertFalse(hasattr(gap_planner, "frontier_gap_terms"))

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

    def test_bank_semantic_relevance_matches_unsegmented_character_stream_query(self) -> None:
        plan = {
            "gap_terms": [
                {"term": "submarineballast", "weight": 2.0},
            ],
            "unsupported_terms": ["submarineballast"],
            "retrieval_queries": ["submarineballast"],
            "follow_up_questions": [],
        }
        related_bank = SimpleNamespace(
            name="related",
            source="https://example.com/related",
            train_raw_windows=[
                "submarines regulate buoyancy with ballast tanks",
                "ballast water shifts pressure inside a submarine",
            ],
        )
        unrelated_bank = SimpleNamespace(
            name="unrelated",
            source="https://example.com/unrelated",
            train_raw_windows=["garden tomato soil sunlight", "library reading room quiet books"],
        )

        related_score = bank_semantic_relevance_score(related_bank, plan)
        unrelated_score = bank_semantic_relevance_score(unrelated_bank, plan)

        self.assertGreater(related_score, 0.0)
        self.assertGreater(related_score, unrelated_score)

    def test_bank_semantic_relevance_matches_boundary_free_compound_terms(self) -> None:
        plan = {
            "gap_terms": [
                {"term": "submarineballastcontrol", "weight": 2.0},
            ],
            "unsupported_terms": ["submarineballastcontrol"],
            "retrieval_queries": ["submarineballastcontrol"],
            "follow_up_questions": [
                "What grounded evidence is still missing for submarineballastcontrol?"
            ],
        }
        related_bank = SimpleNamespace(
            name="related",
            source="https://example.com/related",
            train_raw_windows=["submarine ballast control regulates buoyancy underwater"],
        )
        unrelated_bank = SimpleNamespace(
            name="unrelated",
            source="https://example.com/unrelated",
            train_raw_windows=["garden tomato soil moisture and sunlight"],
        )

        related_score = bank_semantic_relevance_score(related_bank, plan)
        unrelated_score = bank_semantic_relevance_score(unrelated_bank, plan)

        self.assertGreater(related_score, 0.0)
        self.assertGreater(related_score, unrelated_score)

    def test_bank_semantic_relevance_uses_catalog_summary_metadata_for_candidate_focus(self) -> None:
        plan = {
            "gap_terms": [
                {"term": "ballast", "weight": 4.0},
                {"term": "buoyancy", "weight": 4.0},
                {"term": "submarine", "weight": 4.0},
            ],
            "unsupported_terms": ["ballast", "buoyancy", "submarine"],
            "retrieval_queries": ["submarine buoyancy ballast"],
            "follow_up_questions": [
                "What grounded evidence is still missing for submarine?",
                "What grounded evidence is still missing for buoyancy?",
                "What grounded evidence is still missing for ballast?",
            ],
        }
        ballast_bank = SimpleNamespace(
            name="ballast_tank",
            source="https://en.wikipedia.org/wiki/Ballast_tank",
            probe_raw_windows=["B", "Ba", "Bal", "Ball", "Balla"],
            train_raw_windows=["B", "Ba", "Bal", "Ball", "Balla"],
            metadata={
                "catalog_title": "Ballast tank",
                "catalog_summary": (
                    "A ballast tank controls buoyancy in a submarine by moving ballast water."
                ),
                "catalog_terms": ["marine engineering", "ballast tank"],
            },
        )
        submarine_bank = SimpleNamespace(
            name="submarine",
            source="https://en.wikipedia.org/wiki/Submarine",
            probe_raw_windows=["S", "Su", "Sub", "Subm", "Subma"],
            train_raw_windows=["S", "Su", "Sub", "Subm", "Subma"],
            metadata={
                "catalog_title": "Submarine",
                "catalog_summary": (
                    "A submarine is a watercraft capable of independent underwater operation."
                ),
                "catalog_terms": [],
            },
        )

        ballast_score = bank_semantic_relevance_score(ballast_bank, plan)
        submarine_score = bank_semantic_relevance_score(submarine_bank, plan)

        self.assertGreater(ballast_score, submarine_score)

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
