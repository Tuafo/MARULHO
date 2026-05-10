from __future__ import annotations

from collections import deque
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch

from hecsn.service.runtime_sources import _BrainSourceRuntime
from hecsn.service.source_focus import SourceFocusMixin, SourceFocusScorer


class _FakeRuntimeState:
    def __init__(self) -> None:
        self.mutated = 0

    def mark_mutated(self) -> None:
        self.mutated += 1


class _FakeGeometricCuriosity:
    def focus_plan(self, *, query_text: str | None = None):
        return {
            "geometric_gaps": [
                {"concept": "phase transitions"},
            ]
        }


class _FakeConceptStore:
    def snapshot(self, limit: int = 4):
        return {
            "top_concepts": [
                {
                    "label": "cats",
                    "top_terms": ["felines", "mice"],
                }
            ]
        }


class _FakeThoughtLoop:
    def __init__(self) -> None:
        self.gate = SimpleNamespace(active_exploration_target="quantum cats")


class _FakeManager:
    def __init__(self) -> None:
        self._thought_loop_actual = _FakeThoughtLoop()
        self._geometric_curiosity = _FakeGeometricCuriosity()
        self._concept_store = _FakeConceptStore()
        self._brain_recent_query_gaps = deque(
            [
                {
                    "query_text": "cats chase mice",
                    "unsupported_terms": ["mice"],
                }
            ],
            maxlen=8,
        )
        self._brain_config = {"tick_tokens": 8}
        self._runtime_state = _FakeRuntimeState()
        self._brain_source_runtimes = []
        self._brain_source_utility = {
            "science_source": {
                "attempts": 1,
                "selections": 1,
                "tokens_trained_total": 4,
                "utility_ema": 0.4,
                "semantic_alignment_ema": 0.3,
                "grounding_signal_ema": 0.2,
                "focus_overlap_ema": 0.2,
                "grounded_family_summary_ema": 0.5,
                "contradiction_decay_ema": 0.1,
            }
        }

    def _autonomy_focus_plan_locked(self):
        return {
            "query_terms": ["cats", "mice"],
            "unsupported_terms": ["gaps"],
            "gap_terms": [{"term": "cats", "weight": 0.8}],
            "retrieval_queries": ["cats chase mice"],
            "weak_concepts": [{"label": "cats", "top_terms": ["felines"]}],
        }

    def _background_focus_terms_locked(self, *, focus_plan=None):
        return ["cats", "mice"]

    def _background_source_utility_entry_locked(self, runtime):
        return self._brain_source_utility.setdefault(
            runtime.name,
            {
                "attempts": 0,
                "selections": 0,
                "tokens_trained_total": 0,
                "utility_ema": 0.0,
                "semantic_alignment_ema": 0.0,
                "grounding_signal_ema": 0.0,
                "focus_overlap_ema": 0.0,
                "grounded_family_summary_ema": 0.0,
                "contradiction_decay_ema": 0.0,
            },
        )

    @staticmethod
    def _source_text_overlap(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        left_terms = set(left.lower().split())
        right_terms = set(right.lower().split())
        return float(len(left_terms & right_terms)) / float(max(1, min(len(left_terms), len(right_terms))))


class SourceFocusSeamTests(unittest.TestCase):
    def test_alias_points_to_constructed_module(self) -> None:
        self.assertIs(SourceFocusMixin, SourceFocusScorer)

    def test_focus_terms_and_selection_score_are_semantic(self) -> None:
        module = SourceFocusScorer(_FakeManager())
        runtime = _BrainSourceRuntime(
            spec={
                "name": "science_source",
                "source": "science.txt",
                "topic_terms": ["cats", "mice"],
                "metadata": {"label": "cats in science"},
            },
            stream=iter([]),
            buffered_patterns=deque([("cats chase mice", torch.tensor([1.0]))]),
        )
        focus_terms = module._background_focus_terms_locked(limit=6)
        score, semantic_match, fairness, readiness, utility = module._brain_source_selection_score_locked(
            runtime,
            focus_terms=focus_terms,
            focus_pressure=0.8,
            tick_tokens=8,
        )

        self.assertTrue(focus_terms)
        self.assertIn("mice", focus_terms)
        self.assertGreater(semantic_match, 0.0)
        self.assertGreater(score, 0.0)
        self.assertGreaterEqual(fairness, 0.0)
        self.assertGreater(readiness, 0.0)
        self.assertGreater(utility, 0.0)
        self.assertAlmostEqual(runtime.last_selection_score, score)
        self.assertAlmostEqual(runtime.last_semantic_match, semantic_match)

    def test_update_background_source_utility_marks_mutation(self) -> None:
        manager = _FakeManager()
        module = SourceFocusScorer(manager)
        runtime = _BrainSourceRuntime(
            spec={
                "name": "science_source",
                "source": "science.txt",
                "topic_terms": ["cats", "mice"],
            },
            stream=iter([]),
        )

        module._update_background_source_utility_locked(
            runtime=runtime,
            grounded_observation={"content": "cats and mice", "grounding_signal": 0.75},
            total_trained=6,
        )

        entry = manager._brain_source_utility["science_source"]
        self.assertEqual(manager._runtime_state.mutated, 1)
        self.assertEqual(entry["attempts"], 2)
        self.assertEqual(entry["selections"], 2)
        self.assertGreater(entry["utility_ema"], 0.0)
        self.assertTrue(entry["last_selected_at"])

    def test_selected_evidence_weight_map_handles_plural_and_singular_names(self) -> None:
        provider_weighted = SourceFocusScorer._selected_evidence_weight_map(
            {
                "selected_evidence": [
                    {
                        "provider": "Web",
                        "providers": ["web", "arxiv"],
                        "score": 0.9,
                        "term_coverage": 0.8,
                    },
                    {
                        "source_name": "notes.md",
                        "source_names": ["notes.md", "archive.md"],
                        "score": 0.7,
                        "term_coverage": 0.6,
                    },
                ]
            },
            singular_field="provider",
            plural_field="providers",
        )
        source_weighted = SourceFocusScorer._selected_evidence_weight_map(
            {
                "selected_evidence": [
                    {
                        "source_name": "notes.md",
                        "source_names": ["notes.md", "archive.md"],
                        "score": 0.7,
                        "term_coverage": 0.6,
                    }
                ]
            },
            singular_field="source_name",
            plural_field="source_names",
        )

        self.assertGreater(provider_weighted["web"], 0.0)
        self.assertGreater(provider_weighted["arxiv"], 0.0)
        self.assertIn("notes.md", source_weighted)
        self.assertIn("archive.md", source_weighted)
