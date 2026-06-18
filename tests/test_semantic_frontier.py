from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

import marulho.semantics.frontier as frontier
from marulho.semantics.frontier import bank_gap_plan, bank_memory_matches_with_report


class _FakeMemoryStore:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.slow_buffer = [torch.tensor([1.0, 0.0], dtype=torch.float32) for _ in texts]
        self.slow_input_patterns = [item.clone() for item in self.slow_buffer]
        self.slow_routing_keys = [torch.tensor([1.0, 0.0], dtype=torch.float32) for _ in texts]
        self.slow_raw_windows = list(texts)
        self.slow_bucket_ids = [index % 8 for index, _text in enumerate(texts)]
        self.slow_importance = [1.0 for _ in texts]
        self.slow_entry_timestamps = [0 for _ in texts]
        self.slow_replay_count = [0 for _ in texts]
        self.slow_consolidation_level = [0.5 for _ in texts]
        self.replay_entry_calls: list[int] = []
        self.last_bank_memory_match_report: dict[str, object] = {}

    def replay_scores_for_indices(self, indices: list[int], current_token: int) -> dict[int, float]:
        return {int(index): 0.0 for index in indices}

    def collect_query_memory_match_indices(
        self,
        *,
        candidate_bucket_ids,
        max_candidates: int,
        scope: str = "query_memory_match_slow_path",
    ) -> dict[str, object]:
        buckets = [int(value) for value in (candidate_bucket_ids or [])]
        bucket_set = set(buckets)
        candidates = [
            index
            for index, bucket_id in enumerate(self.slow_bucket_ids)
            if int(bucket_id) in bucket_set
        ][: max(0, int(max_candidates))]
        return {
            "surface": "bounded_query_memory_match_candidates.v1",
            "status": "collected",
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
            "global_score_scan": False,
            "global_candidate_scan": False,
            "runs_live_tick": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "fallback_reason": None,
        }

    def replay_entry(self, idx: int, current_token: int | None = None) -> dict[str, object]:
        self.replay_entry_calls.append(int(idx))
        return {
            "text": self._texts[idx],
            "raw_window": self.slow_raw_windows[idx],
            "metadata": {},
        }

    def record_bank_memory_match_report(self, report) -> dict[str, object]:
        self.last_bank_memory_match_report = dict(report)
        return self.last_bank_memory_match_report


class _FakeRoutingIndex:
    def search_tensors(self, queries: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        count = max(1, int(k))
        return torch.arange(count, dtype=torch.long).unsqueeze(0), torch.ones((1, count), dtype=torch.float32)


class _FakeTrainer:
    def __init__(self, texts: list[str]) -> None:
        self.token_count = 10
        self.config = SimpleNamespace(input_representation="hashed_ngram", k_routing=8)
        self.model = SimpleNamespace(
            memory_store=_FakeMemoryStore(texts),
            routing_index=_FakeRoutingIndex(),
        )

    def routing_key_for_pattern(self, pattern: torch.Tensor) -> torch.Tensor:
        return pattern


class SemanticFrontierTests(unittest.TestCase):
    def test_source_bank_memory_matches_requires_reported_api(self) -> None:
        self.assertFalse(hasattr(frontier, "bank_memory_matches"))
        self.assertNotIn("bank_memory_matches", frontier.__all__)

    def test_bank_memory_matches_aggregates_bounded_probe_reports_and_payload_cache(self) -> None:
        trainer = _FakeTrainer([f"memory episode {index}" for index in range(64)])
        bank = SimpleNamespace(
            name="bank",
            probe_patterns=[
                torch.tensor([1.0, 0.0], dtype=torch.float32),
                torch.tensor([1.0, 0.0], dtype=torch.float32),
            ],
            probe_raw_windows=["memory probe", "memory probe"],
            train_raw_windows=[],
        )

        matches, report = bank_memory_matches_with_report(
            trainer,
            bank,
            probe_samples=2,
            memories_per_probe=2,
            max_matches=8,
        )

        self.assertEqual([match["memory_index"] for match in matches], [0, 1])
        self.assertEqual(report["surface"], "bounded_source_bank_memory_match.v1")
        self.assertEqual(report["probe_count"], 2)
        self.assertEqual(report["candidate_index_count"], 64)
        self.assertEqual(report["unique_candidate_index_count"], 32)
        self.assertEqual(report["similarity_score_count"], 64)
        self.assertEqual(report["raw_text_payload_count"], 2)
        self.assertEqual(report["raw_text_payload_cache_hits"], 2)
        self.assertEqual(report["raw_text_payload_policy"], "shared_returned_similarity_matches_only")
        self.assertFalse(report["global_candidate_scan"])
        self.assertFalse(report["global_score_scan"])
        self.assertFalse(report["runs_live_tick"])
        self.assertFalse(report["language_reasoning"])
        self.assertEqual(trainer.model.memory_store.replay_entry_calls, [0, 1])
        self.assertEqual(
            trainer.model.memory_store.last_bank_memory_match_report["match_indices"],
            [0, 1],
        )

    def test_bank_gap_plan_exposes_bank_memory_match_report(self) -> None:
        trainer = _FakeTrainer(["submarine ballast control.", "garden soil."])
        bank = SimpleNamespace(
            name="submarine",
            probe_patterns=[torch.tensor([1.0, 0.0], dtype=torch.float32)],
            probe_raw_windows=["submarine ballast"],
            train_raw_windows=["submarine ballast"],
        )

        plan = bank_gap_plan(trainer, bank, probe_samples=1, memories_per_probe=1)

        report = plan["bank_memory_match_report"]
        self.assertEqual(report["surface"], "bounded_source_bank_memory_match.v1")
        self.assertEqual(plan["query_summary"]["memory_match_report"]["surface"], report["surface"])
        self.assertFalse(report["runs_live_tick"])
        self.assertFalse(report["language_reasoning"])


if __name__ == "__main__":
    unittest.main()
