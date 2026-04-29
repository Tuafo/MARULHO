from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hecsn.training import query_runner


class _FakeMemoryStore:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.slow_buffer = [torch.tensor([1.0, 0.0], dtype=torch.float32) for _ in texts]
        self.slow_input_patterns = [item.clone() for item in self.slow_buffer]
        self.slow_routing_keys = [None for _ in texts]
        self.slow_raw_windows = list(texts)
        self.slow_bucket_ids = [None for _ in texts]
        self.slow_importance = [1.0 for _ in texts]
        self.slow_entry_timestamps = [0 for _ in texts]
        self.slow_replay_count = [0 for _ in texts]

    def replay_scores(self, token_count: int) -> torch.Tensor:
        return torch.zeros(len(self._texts), dtype=torch.float32)

    def replay_entry(self, idx: int, current_token: int | None = None) -> dict[str, object]:
        return {
            "text": self._texts[idx],
            "raw_window": self.slow_raw_windows[idx],
            "metadata": {},
        }


class _FakeTrainer:
    def __init__(self, texts: list[str]) -> None:
        self.token_count = 10
        self.config = SimpleNamespace(input_representation="hashed_ngram")
        self.model = SimpleNamespace(memory_store=_FakeMemoryStore(texts))


class QueryRunnerTermMatchingTests(unittest.TestCase):
    def test_memory_matches_caches_repeated_semantic_term_pairs(self) -> None:
        trainer = _FakeTrainer(["alphabeto gamma delta."] * 16)
        pattern = torch.tensor([1.0, 0.0], dtype=torch.float32)
        calls: list[tuple[str, str]] = []

        def fake_similarity(left: str, right: str) -> float:
            calls.append((left, right))
            return 1.0 if (left, right) == ("alphabeta", "alphabeto") else 0.0

        with patch.object(query_runner, "semantic_unit_similarity", side_effect=fake_similarity):
            matches = query_runner.memory_matches(
                trainer,
                pattern,
                pattern,
                top_k=4,
                top_chars=1,
                query_terms=["alphabeta"],
            )

        self.assertEqual(calls.count(("alphabeta", "alphabeto")), 1)
        self.assertTrue(matches)
        self.assertTrue(all(match["matched_query_terms"] == ["alphabeta"] for match in matches))
        self.assertTrue(all(match["query_overlap"] == 1 for match in matches))

    def test_memory_matches_keeps_health_smoke_scale_semantic_work_bounded(self) -> None:
        trainer = _FakeTrainer(["alphabeto gamma delta."] * 512)
        pattern = torch.tensor([1.0, 0.0], dtype=torch.float32)
        calls: list[tuple[str, str]] = []

        def fake_similarity(left: str, right: str) -> float:
            calls.append((left, right))
            return 1.0 if (left, right) == ("alphabeta", "alphabeto") else 0.0

        with patch.object(query_runner, "semantic_unit_similarity", side_effect=fake_similarity):
            matches = query_runner.memory_matches(
                trainer,
                pattern,
                pattern,
                top_k=4,
                top_chars=1,
                query_terms=["alphabeta", "needleword", "longtermzz"],
            )

        self.assertTrue(matches)
        self.assertEqual(calls.count(("alphabeta", "alphabeto")), 1)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertLessEqual(len(calls), 9)
        self.assertLess(len(calls), len(trainer.model.memory_store.slow_buffer))

    def test_memory_matches_preserves_literal_and_inflection_matches(self) -> None:
        trainer = _FakeTrainer(
            [
                "submarine ballast control keeps buoyancy stable.",
                "cats rest indoors.",
            ]
        )
        pattern = torch.tensor([1.0, 0.0], dtype=torch.float32)

        compound_matches = query_runner.memory_matches(
            trainer,
            pattern,
            pattern,
            top_k=2,
            top_chars=1,
            query_terms=["submarineballastcontrol"],
        )
        cat_matches = query_runner.memory_matches(
            trainer,
            pattern,
            pattern,
            top_k=2,
            top_chars=1,
            query_terms=["cat"],
        )

        self.assertEqual(compound_matches[0]["memory_index"], 0)
        self.assertEqual(compound_matches[0]["matched_query_terms"], ["submarineballastcontrol"])
        self.assertEqual(cat_matches[0]["memory_index"], 1)
        self.assertEqual(cat_matches[0]["matched_query_terms"], ["cat"])


if __name__ == "__main__":
    unittest.main()
