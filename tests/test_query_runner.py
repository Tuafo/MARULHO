from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from marulho.training import query_runner


class _FakeMemoryStore:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.slow_buffer = [torch.tensor([1.0, 0.0], dtype=torch.float32) for _ in texts]
        self.slow_input_patterns = [item.clone() for item in self.slow_buffer]
        self.slow_routing_keys = [None for _ in texts]
        self.slow_raw_windows = list(texts)
        self.slow_bucket_ids = [index % 8 for index, _text in enumerate(texts)]
        self.slow_importance = [1.0 for _ in texts]
        self.slow_entry_timestamps = [0 for _ in texts]
        self.slow_replay_count = [0 for _ in texts]
        self.replay_entry_calls: list[int] = []
        self.last_query_memory_match_report: dict[str, object] = {}

    def replay_scores(self, token_count: int) -> torch.Tensor:
        return torch.zeros(len(self._texts), dtype=torch.float32)

    def replay_scores_for_indices(
        self,
        indices: list[int],
        token_count: int,
    ) -> dict[int, float]:
        return {int(index): 0.0 for index in indices}

    def collect_query_memory_match_indices(
        self,
        *,
        candidate_bucket_ids: list[int] | torch.Tensor | None,
        max_candidates: int,
        scope: str = "query_memory_match_slow_path",
    ) -> dict[str, object]:
        buckets = (
            []
            if candidate_bucket_ids is None
            else [int(value) for value in candidate_bucket_ids]
        )
        bucket_set = set(buckets)
        candidates = [
            index
            for index, bucket_id in enumerate(self.slow_bucket_ids)
            if int(bucket_id) in bucket_set
        ][: max(0, int(max_candidates))]
        report: dict[str, object] = {
            "surface": "bounded_query_memory_match_candidates.v1",
            "status": "collected" if candidates else "empty",
            "scope": scope,
            "memory_size": len(self.slow_buffer),
            "requested_count": int(max_candidates),
            "candidate_window_limit": int(max_candidates),
            "candidate_window_policy": "recent_bucket_round_robin_candidate_pool",
            "candidate_scope": "bucket_indexed_candidate_window",
            "candidate_bucket_ids": buckets,
            "candidate_bucket_count": len(buckets),
            "candidate_index_available_count": sum(
                1 for bucket_id in self.slow_bucket_ids if int(bucket_id) in bucket_set
            ),
            "candidate_index_count": len(candidates),
            "match_indices": candidates,
            "score_count": 0,
            "global_score_scan": False,
            "global_candidate_scan": False,
            "runs_live_tick": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "fallback_reason": None if candidates else "empty_query_candidate_window",
        }
        self.last_query_memory_match_report = report
        return report

    def replay_entry(self, idx: int, current_token: int | None = None) -> dict[str, object]:
        self.replay_entry_calls.append(int(idx))
        return {
            "text": self._texts[idx],
            "raw_window": self.slow_raw_windows[idx],
            "metadata": {},
        }


class _FakeRoutingIndex:
    def search_tensors(self, queries: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        count = max(1, int(k))
        ids = torch.arange(count, dtype=torch.long).unsqueeze(0)
        scores = torch.ones((1, count), dtype=torch.float32)
        return ids, scores


class _FakeTrainer:
    def __init__(self, texts: list[str]) -> None:
        self.token_count = 10
        self.config = SimpleNamespace(input_representation="hashed_ngram", k_routing=8)
        self.model = SimpleNamespace(
            memory_store=_FakeMemoryStore(texts),
            routing_index=_FakeRoutingIndex(),
        )


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

    def test_memory_matches_uses_bounded_query_candidate_window(self) -> None:
        trainer = _FakeTrainer(["alphabeto gamma delta."] * 128)
        pattern = torch.tensor([1.0, 0.0], dtype=torch.float32)

        matches, report = query_runner.memory_matches_with_report(
            trainer,
            pattern,
            pattern,
            top_k=4,
            top_chars=1,
            query_terms=["alphabeta"],
            memory_candidate_limit=5,
        )

        self.assertTrue(matches)
        self.assertEqual(report["surface"], "bounded_query_memory_match.v1")
        self.assertEqual(
            report["candidate_surface"],
            "bounded_query_memory_match_candidates.v1",
        )
        self.assertEqual(report["candidate_window_limit"], 5)
        self.assertEqual(report["candidate_index_count"], 5)
        self.assertEqual(report["similarity_score_count"], 5)
        self.assertEqual(report["replay_priority_score_count"], 5)
        self.assertFalse(report["global_score_scan"])
        self.assertFalse(report["global_candidate_scan"])
        self.assertFalse(report["runs_live_tick"])
        self.assertLessEqual(max(trainer.model.memory_store.replay_entry_calls), 4)

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
