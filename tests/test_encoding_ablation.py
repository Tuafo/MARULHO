"""Tests for encoding ablation study (§4.2)."""

from __future__ import annotations

import unittest

from marulho.evaluation.encoding_ablation import run_encoding_ablation


class TestEncodingAblation(unittest.TestCase):
    def test_returns_all_schemes(self) -> None:
        result = run_encoding_ablation(input_dim=64, window_size=8, min_tokens=10)
        self.assertIn("schemes", result)
        for scheme in ("unigram_ascii", "order_weighted_ascii", "hashed_ngram", "rtf_burst"):
            self.assertIn(scheme, result["schemes"])

    def test_coherence_values_are_floats(self) -> None:
        result = run_encoding_ablation(input_dim=64, window_size=8, min_tokens=10)
        for name, data in result["schemes"].items():
            self.assertIsInstance(data["coherence"], float)
            self.assertGreaterEqual(data["coherence"], -1.0)
            self.assertLessEqual(data["coherence"], 1.0)

    def test_ranking_sorted_descending(self) -> None:
        result = run_encoding_ablation(input_dim=64, window_size=8, min_tokens=10)
        ranking = result["ranking"]
        self.assertGreater(len(ranking), 0)
        coherences = [r["coherence"] for r in ranking]
        self.assertEqual(coherences, sorted(coherences, reverse=True))

    def test_best_encoding_in_ranking(self) -> None:
        result = run_encoding_ablation(input_dim=64, window_size=8, min_tokens=10)
        best = result["best_encoding"]
        self.assertIn(best, [r["scheme"] for r in result["ranking"]])

    def test_custom_corpus(self) -> None:
        corpus = "hello world " * 100
        result = run_encoding_ablation(corpus=corpus, input_dim=64, window_size=6, min_tokens=10)
        self.assertIn("best_encoding", result)
        self.assertGreater(len(result["ranking"]), 0)

    def test_metadata_present(self) -> None:
        result = run_encoding_ablation(input_dim=64, window_size=8, min_tokens=10)
        self.assertIn("metadata", result)
        self.assertIn("input_dim", result["metadata"])
        self.assertIn("note", result["metadata"])


if __name__ == "__main__":
    unittest.main()
