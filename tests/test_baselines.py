"""Tests for baseline models (§8.1).

Covers: OnlineSOM, 4-gram character model, fastText character n-grams,
and the combined baseline runner.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from hecsn.evaluation.baselines import (
    BaselineResults,
    CharNGramEmbedder,
    FourGramModel,
    OnlineSOM,
    _text_to_char_ngram_vector,
    evaluate_fasttext_grounding_probe,
    evaluate_som_grounding_probe,
    run_all_baselines,
    train_4gram_on_corpus,
    train_fasttext_baseline,
    train_som_on_corpus,
)
from hecsn.evaluation.grounding_probe import GroundingProbeResult


# Minimal corpus for fast tests
MINI_CORPUS = (
    "the ocean is deep and blue. water covers most of the earth. "
    "submarines dive deep into the ocean. fire is hot and dangerous. "
    "ice is cold and slippery. the sun provides light and heat. "
    "dogs guard houses and bark at strangers. cats hunt mice silently. "
    "democracy requires voting by citizens. justice demands fairness for all. "
) * 10


class OnlineSOMTests(unittest.TestCase):
    """Tests for the Online SOM baseline."""

    def test_init_shape(self) -> None:
        som = OnlineSOM(input_dim=32, n_prototypes=8, seed=0)
        self.assertEqual(som.weights.shape, (8, 32))

    def test_weights_normalized_after_init(self) -> None:
        som = OnlineSOM(input_dim=64, n_prototypes=16, seed=0)
        norms = np.linalg.norm(som.weights, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_train_vector_returns_bmu(self) -> None:
        som = OnlineSOM(input_dim=32, n_prototypes=8, seed=0)
        x = np.random.default_rng(1).standard_normal(32).astype(np.float32)
        bmu = som.train_vector(x)
        self.assertIsInstance(bmu, int)
        self.assertGreaterEqual(bmu, 0)
        self.assertLess(bmu, 8)

    def test_weights_remain_normalized_after_training(self) -> None:
        som = OnlineSOM(input_dim=32, n_prototypes=8, seed=0)
        rng = np.random.default_rng(1)
        for _ in range(100):
            x = rng.standard_normal(32).astype(np.float32)
            som.train_vector(x)
        norms = np.linalg.norm(som.weights, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_get_vector_returns_unit_vector(self) -> None:
        som = train_som_on_corpus("hello world test " * 20, input_dim=32, n_prototypes=8)
        vec = som.get_vector("hello")
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_get_vector_deterministic(self) -> None:
        som = train_som_on_corpus("hello world test " * 20, input_dim=32, n_prototypes=8)
        v1 = som.get_vector("hello")
        v2 = som.get_vector("hello")
        np.testing.assert_array_equal(v1, v2)

    def test_train_som_on_corpus(self) -> None:
        som = train_som_on_corpus(MINI_CORPUS, input_dim=64, n_prototypes=16)
        self.assertEqual(som.weights.shape, (16, 64))
        self.assertGreater(som._step, 0)

    def test_evaluate_som_grounding_probe(self) -> None:
        som = train_som_on_corpus(MINI_CORPUS, input_dim=64, n_prototypes=16)
        result = evaluate_som_grounding_probe(som)
        self.assertIsInstance(result, GroundingProbeResult)
        self.assertEqual(result.total_count, 50)
        # SOM accuracy should be between 0 and 1
        self.assertGreaterEqual(result.total_accuracy, 0.0)
        self.assertLessEqual(result.total_accuracy, 1.0)


class CharNGramVectorTests(unittest.TestCase):
    """Tests for the character n-gram vector encoding."""

    def test_output_shape(self) -> None:
        vec = _text_to_char_ngram_vector("hello", dim=64)
        self.assertEqual(vec.shape, (64,))

    def test_normalized(self) -> None:
        vec = _text_to_char_ngram_vector("hello world", dim=128)
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_empty_string_returns_zero(self) -> None:
        vec = _text_to_char_ngram_vector("", dim=32)
        self.assertAlmostEqual(np.linalg.norm(vec), 0.0)

    def test_different_texts_different_vectors(self) -> None:
        v1 = _text_to_char_ngram_vector("ocean", dim=128)
        v2 = _text_to_char_ngram_vector("mountain", dim=128)
        cos = np.dot(v1, v2)
        self.assertLess(cos, 0.99)  # not identical


class FourGramModelTests(unittest.TestCase):
    """Tests for the 4-gram character model."""

    def test_train_creates_contexts(self) -> None:
        model = FourGramModel()
        model.train("abcdefgh")
        self.assertIn("abcd", model.counts)

    def test_predict_known_context(self) -> None:
        model = FourGramModel()
        model.train("aaaabbbbb")
        pred = model.predict("aabb")
        self.assertIsNotNone(pred)

    def test_predict_unknown_context_returns_none(self) -> None:
        model = FourGramModel()
        model.train("hello")
        pred = model.predict("zzzz")
        self.assertIsNone(pred)

    def test_predict_short_context_returns_none(self) -> None:
        model = FourGramModel()
        model.train("hello")
        pred = model.predict("hi")
        self.assertIsNone(pred)

    def test_evaluate_prediction_accuracy(self) -> None:
        model = train_4gram_on_corpus(MINI_CORPUS)
        result = model.evaluate_prediction_accuracy(MINI_CORPUS)
        self.assertIn("accuracy", result)
        self.assertIn("correct", result)
        self.assertIn("total", result)
        self.assertGreater(result["total"], 0)
        # On training data, accuracy should be above chance for repeated text
        self.assertGreater(result["accuracy"], 0.0)

    def test_corpus_accuracy_on_repetitive_text(self) -> None:
        """Highly repetitive text should yield high prediction accuracy."""
        repetitive = "abcabcabcabc" * 100
        model = train_4gram_on_corpus(repetitive)
        result = model.evaluate_prediction_accuracy(repetitive)
        self.assertGreater(result["accuracy"], 0.5)


class CharNGramEmbedderTests(unittest.TestCase):
    """Tests for the fastText-style character n-gram embedder."""

    def test_get_word_vector_shape(self) -> None:
        model = CharNGramEmbedder(dim=64, seed=0)
        vec = model.get_word_vector("ocean")
        self.assertEqual(vec.shape, (64,))

    def test_get_word_vector_normalized(self) -> None:
        model = CharNGramEmbedder(dim=64, seed=0)
        vec = model.get_word_vector("ocean")
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_get_word_vector_deterministic(self) -> None:
        model = CharNGramEmbedder(dim=64, seed=0)
        v1 = model.get_word_vector("ocean")
        v2 = model.get_word_vector("ocean")
        np.testing.assert_array_equal(v1, v2)

    def test_train_changes_vectors(self) -> None:
        model = CharNGramEmbedder(dim=64, seed=0)
        before = model.get_word_vector("ocean").copy()
        model.train_on_corpus(MINI_CORPUS)
        after = model.get_word_vector("ocean")
        # Training should change the vectors
        self.assertFalse(np.allclose(before, after, atol=1e-6))

    def test_trained_vectors_still_normalized(self) -> None:
        model = train_fasttext_baseline(MINI_CORPUS, dim=64, seed=0)
        for word in ["ocean", "fire", "dog", "justice"]:
            vec = model.get_word_vector(word)
            norm = np.linalg.norm(vec)
            self.assertAlmostEqual(norm, 1.0, places=3)

    def test_evaluate_fasttext_grounding_probe(self) -> None:
        model = train_fasttext_baseline(MINI_CORPUS, dim=64, seed=0)
        result = evaluate_fasttext_grounding_probe(model)
        self.assertIsInstance(result, GroundingProbeResult)
        self.assertEqual(result.total_count, 50)
        self.assertGreaterEqual(result.total_accuracy, 0.0)
        self.assertLessEqual(result.total_accuracy, 1.0)


class CombinedBaselineTests(unittest.TestCase):
    """Tests for the combined baseline runner."""

    def test_run_all_baselines_returns_results(self) -> None:
        results = run_all_baselines(MINI_CORPUS, input_dim=64, n_prototypes=16, seed=42)
        self.assertIsInstance(results, BaselineResults)

    def test_summary_structure(self) -> None:
        results = run_all_baselines(MINI_CORPUS, input_dim=64, n_prototypes=16, seed=42)
        summary = results.summary()

        # Check all expected keys
        self.assertIn("online_som", summary)
        self.assertIn("four_gram", summary)
        self.assertIn("fasttext", summary)
        self.assertIn("calibrated_target", summary)

        # Check SOM fields
        self.assertIn("grounding_probe_accuracy", summary["online_som"])
        self.assertIn("concreteness_gap", summary["online_som"])

        # Check 4-gram fields
        self.assertIn("accuracy", summary["four_gram"])
        self.assertIn("total", summary["four_gram"])

        # Check fastText fields
        self.assertIn("grounding_probe_accuracy", summary["fasttext"])
        self.assertIn("concreteness_gap", summary["fasttext"])

        # Check calibrated target
        self.assertIn("fasttext_score", summary["calibrated_target"])
        self.assertIn("hecsn_multimodal_target", summary["calibrated_target"])

    def test_calibrated_target_uses_fasttext_score(self) -> None:
        results = run_all_baselines(MINI_CORPUS, input_dim=64, n_prototypes=16, seed=42)
        summary = results.summary()
        ft_score = summary["fasttext"]["grounding_probe_accuracy"]
        cal = summary["calibrated_target"]
        self.assertAlmostEqual(cal["fasttext_score"], ft_score)
        self.assertAlmostEqual(cal["hecsn_multimodal_target"], ft_score + 0.05)

    def test_som_accuracy_is_float(self) -> None:
        results = run_all_baselines(MINI_CORPUS, input_dim=64, n_prototypes=16, seed=42)
        self.assertIsInstance(results.som_probe.total_accuracy, float)

    def test_four_gram_accuracy_positive(self) -> None:
        results = run_all_baselines(MINI_CORPUS, input_dim=64, n_prototypes=16, seed=42)
        self.assertGreater(results.four_gram["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
