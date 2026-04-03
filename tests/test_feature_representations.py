from __future__ import annotations

import unittest

import torch

from hecsn.config.model_config import HECSNConfig
from hecsn.data.rtf_encoder import RTFEncoder
from hecsn.training.baselines import CharNGramMemory, OnlineKMeans
from hecsn.training.behavioral_metrics import completion_coherence


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


if __name__ == "__main__":
    unittest.main()