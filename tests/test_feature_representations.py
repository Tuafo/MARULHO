from __future__ import annotations

import unittest

import torch

from hecsn.config.model_config import HECSNConfig
from hecsn.data.rtf_encoder import RTFEncoder
from hecsn.training.baselines import CharNGramMemory, OnlineKMeans
from hecsn.training.behavioral_metrics import (
    completion_coherence,
    compositionality_score,
    grounding_probe,
    novelty_coverage_curve,
    representation_retention,
    temporal_coherence,
)


class FeatureRepresentationTests(unittest.TestCase):
    def test_order_weighted_representation_distinguishes_anagrams(self) -> None:
        encoder = RTFEncoder(window_size=3, representation="order_weighted_ascii")

        left = encoder.feature_vector([ord(ch) for ch in "tea"])
        right = encoder.feature_vector([ord(ch) for ch in "eat"])
        similarity = float(torch.nn.functional.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0), dim=1).item())

        self.assertFalse(torch.allclose(left, right))
        self.assertLess(similarity, 0.999)

    def test_unigram_representation_collapses_anagrams(self) -> None:
        encoder = RTFEncoder(window_size=3, representation="unigram_ascii")

        left = encoder.feature_vector([ord(ch) for ch in "tea"])
        right = encoder.feature_vector([ord(ch) for ch in "eat"])

        self.assertTrue(torch.allclose(left, right))

    def test_hashed_ngram_representation_uses_configured_dim(self) -> None:
        config = HECSNConfig(
            window_size=4,
            input_representation="hashed_ngram",
            hashed_ngram_dim=64,
            hashed_ngram_min_n=2,
            hashed_ngram_max_n=3,
        )
        encoder = RTFEncoder.from_config(config)

        left = encoder.feature_vector([ord(ch) for ch in "bank"])
        right = encoder.feature_vector([ord(ch) for ch in "knab"])

        self.assertEqual(left.numel(), 64)
        self.assertFalse(torch.allclose(left, right))


class OnlineKMeansTests(unittest.TestCase):
    def test_online_kmeans_separates_simple_clusters(self) -> None:
        baseline = OnlineKMeans(n_clusters=2, feature_dim=2)
        features = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.9, 0.1]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.1, 0.9]),
        ]

        baseline.fit(features)
        labels = baseline.predict_many(features)

        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])


class BehavioralBaselineTests(unittest.TestCase):
    def test_char_ngram_memory_prefers_true_completion(self) -> None:
        baseline = CharNGramMemory(max_context=3)
        baseline.fit(["hell", "help", "helm", "hero"])

        positive = baseline.completion_score("hel", "hell")
        negative = baseline.completion_score("hel", "zero")

        self.assertGreater(positive, negative)

    def test_completion_coherence_reports_positive_margin(self) -> None:
        baseline = CharNGramMemory(max_context=3)
        baseline.fit(["hell", "help", "helm", "hero", "good", "gold"])

        result = completion_coherence(
            baseline.completion_score,
            ["hell", "help", "gold", "good"],
            max_samples=8,
        )

        self.assertGreater(result["sample_count"], 0)
        self.assertIsNotNone(result["mean_margin"])
        self.assertGreater(float(result["mean_margin"]), 0.0)

    def test_temporal_coherence_reports_stable_winners(self) -> None:
        result = temporal_coherence(
            [
                ("cat", 1),
                ("dog", 2),
                ("cat", 1),
                ("cat", 1),
                ("dog", 2),
                ("dog", 2),
            ]
        )

        self.assertEqual(result["supported_pattern_count"], 2)
        self.assertIsNotNone(result["mean_coherence"])
        self.assertGreaterEqual(float(result["mean_coherence"]), 1.0)

    def test_compositionality_score_prefers_additive_composite(self) -> None:
        left = torch.tensor([1.0, 0.0, 0.0])
        right = torch.tensor([0.0, 1.0, 0.0])
        combined = torch.tensor([1.0, 1.0, 0.0])

        result = compositionality_score([(left, right, combined)])

        self.assertEqual(result["sample_count"], 1)
        self.assertIsNotNone(result["mean_score"])
        self.assertGreater(float(result["mean_score"]), 0.99)

    def test_grounding_probe_prefers_positive_pair(self) -> None:
        anchor = torch.tensor([1.0, 0.0, 0.0])
        positive = torch.tensor([0.8, 0.2, 0.0])
        negative = torch.tensor([0.0, 1.0, 0.0])

        result = grounding_probe([(anchor, positive, negative)])

        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(float(result["accuracy"]), 1.0)
        self.assertGreater(float(result["mean_margin"]), 0.0)

    def test_novelty_coverage_curve_reports_final_rate(self) -> None:
        result = novelty_coverage_curve(
            [True, True, False, False, True, False],
            [2, 4, 6],
            healthy_range=(0.1, 0.9),
        )

        self.assertEqual(len(result["novelty_rate_by_checkpoint"]), 3)
        self.assertAlmostEqual(float(result["final_novelty_rate"]), 0.5)
        self.assertTrue(result["healthy_final_range"])

    def test_representation_retention_reports_similarity(self) -> None:
        before = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
        after = [torch.tensor([0.9, 0.1]), torch.tensor([0.1, 0.9])]

        result = representation_retention(before, after)

        self.assertEqual(result["sample_count"], 2)
        self.assertIsNotNone(result["mean_retention"])
        self.assertGreater(float(result["mean_retention"]), 0.85)


if __name__ == "__main__":
    unittest.main()
