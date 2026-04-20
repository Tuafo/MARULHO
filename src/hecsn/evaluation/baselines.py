"""Baseline models for calibrating HECSN evaluation metrics (§8.1).

Three baselines:
  1. Online SOM — same dim, same prototypes, no SNN dynamics
  2. 4-gram character model — next-character prediction accuracy
  3. fastText character n-grams — grounding probe calibration

All baselines are trained on the same text corpus that HECSN uses,
so comparisons are apples-to-apples.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from hecsn.evaluation.grounding_probe import (
    GroundingProbeResult,
    evaluate_grounding_probe,
)


# ---------------------------------------------------------------------------
# Baseline 1: Online SOM
# ---------------------------------------------------------------------------

class OnlineSOM:
    """Minimal online Self-Organizing Map for baseline comparison.

    Same dimension and prototype count as HECSN, but no SNN dynamics,
    no multimodal grounding, no sleep consolidation.
    """

    def __init__(
        self,
        input_dim: int,
        n_prototypes: int,
        lr_init: float = 0.3,
        lr_final: float = 0.01,
        sigma_init: float | None = None,
        sigma_final: float = 0.5,
        total_steps: int = 10_000,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.weights = rng.standard_normal((n_prototypes, input_dim)).astype(np.float32)
        norms = np.linalg.norm(self.weights, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.weights /= norms

        self.n_prototypes = n_prototypes
        self.input_dim = input_dim
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.sigma_init = sigma_init or max(1.0, n_prototypes / 4.0)
        self.sigma_final = sigma_final
        self.total_steps = total_steps
        self._step = 0

        # 1-D topology for neighbourhood
        self._coords = np.arange(n_prototypes, dtype=np.float32)

    def _lr(self) -> float:
        t = min(self._step / max(self.total_steps, 1), 1.0)
        return self.lr_init * (self.lr_final / self.lr_init) ** t

    def _sigma(self) -> float:
        t = min(self._step / max(self.total_steps, 1), 1.0)
        return self.sigma_init * (self.sigma_final / self.sigma_init) ** t

    def train_vector(self, x: np.ndarray) -> int:
        """Present a single vector; return BMU index."""
        x = x.astype(np.float32)
        x_norm = np.linalg.norm(x)
        if x_norm > 1e-8:
            x = x / x_norm

        dists = np.linalg.norm(self.weights - x[None, :], axis=1)
        bmu = int(np.argmin(dists))

        sigma = self._sigma()
        lr = self._lr()
        neighborhood = np.exp(-0.5 * ((self._coords - bmu) / max(sigma, 1e-8)) ** 2)
        delta = lr * neighborhood[:, None] * (x[None, :] - self.weights)
        self.weights += delta

        # re-normalize
        norms = np.linalg.norm(self.weights, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.weights /= norms

        self._step += 1
        return bmu

    def get_vector(self, text: str) -> np.ndarray:
        """Get representation for a text string by encoding and routing."""
        x = _text_to_char_ngram_vector(text, self.input_dim)
        dists = np.linalg.norm(self.weights - x[None, :], axis=1)
        bmu = int(np.argmin(dists))
        return self.weights[bmu].copy()


def _text_to_char_ngram_vector(text: str, dim: int, ngram_range: tuple[int, int] = (2, 5)) -> np.ndarray:
    """Hash-based character n-gram encoding into a fixed-dim vector."""
    vec = np.zeros(dim, dtype=np.float32)
    text_lower = text.lower().strip()
    for n in range(ngram_range[0], ngram_range[1] + 1):
        for i in range(len(text_lower) - n + 1):
            gram = text_lower[i : i + n]
            idx = hash(gram) % dim
            vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec /= norm
    return vec


def train_som_on_corpus(
    corpus: str,
    input_dim: int = 128,
    n_prototypes: int = 64,
    window_size: int = 8,
    seed: int = 42,
) -> OnlineSOM:
    """Train an OnlineSOM on a text corpus using sliding character windows."""
    som = OnlineSOM(
        input_dim=input_dim,
        n_prototypes=n_prototypes,
        seed=seed,
        total_steps=max(len(corpus) - window_size, 1),
    )
    for i in range(0, len(corpus) - window_size):
        chunk = corpus[i : i + window_size]
        x = _text_to_char_ngram_vector(chunk, input_dim)
        som.train_vector(x)
    return som


def evaluate_som_grounding_probe(som: OnlineSOM) -> GroundingProbeResult:
    """Evaluate the 50-triple grounding probe on an OnlineSOM."""

    def vector_fn(text: str) -> torch.Tensor:
        vec = som.get_vector(text)
        return torch.from_numpy(vec)

    return evaluate_grounding_probe(vector_fn)


# ---------------------------------------------------------------------------
# Baseline 2: 4-gram character model
# ---------------------------------------------------------------------------

@dataclass
class FourGramModel:
    """Online 4-gram character model for next-character prediction."""

    n: int = 4
    counts: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    _total_predictions: int = 0
    _correct_predictions: int = 0

    def train(self, text: str) -> None:
        """Train on a text string, updating counts."""
        for i in range(len(text) - self.n):
            context = text[i : i + self.n]
            target = text[i + self.n]
            self.counts[context][target] += 1

    def predict(self, context: str) -> str | None:
        """Predict next character given context."""
        if len(context) < self.n:
            return None
        ctx = context[-self.n :]
        if ctx not in self.counts:
            return None
        return self.counts[ctx].most_common(1)[0][0]

    def evaluate_prediction_accuracy(self, text: str) -> dict[str, Any]:
        """Evaluate next-character prediction accuracy on text."""
        correct = 0
        total = 0
        for i in range(len(text) - self.n):
            context = text[i : i + self.n]
            target = text[i + self.n]
            predicted = self.predict(context)
            if predicted is not None:
                total += 1
                if predicted == target:
                    correct += 1
        accuracy = correct / total if total > 0 else 0.0
        return {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "vocabulary_size": sum(len(v) for v in self.counts.values()),
            "context_count": len(self.counts),
        }


def train_4gram_on_corpus(corpus: str) -> FourGramModel:
    """Train a 4-gram character model on a text corpus."""
    model = FourGramModel()
    model.train(corpus)
    return model


# ---------------------------------------------------------------------------
# Baseline 3: fastText character n-grams (lightweight reimplementation)
# ---------------------------------------------------------------------------

class CharNGramEmbedder:
    """Lightweight character n-gram embedder for grounding probe calibration.

    This is a simplified fastText-style model that uses hashed character
    n-grams to produce word vectors. No neural network training — just
    accumulates co-occurrence statistics via online SVD-like projection.

    For the grounding probe, we only need word-level vectors, so this
    suffices as a calibration baseline.
    """

    def __init__(
        self,
        dim: int = 100,
        min_n: int = 1,
        max_n: int = 6,
        n_buckets: int = 10_000,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.min_n = min_n
        self.max_n = max_n
        self.n_buckets = n_buckets

        rng = np.random.default_rng(seed)
        self.bucket_vectors = rng.standard_normal((n_buckets, dim)).astype(np.float32) * 0.01
        self._word_cache: dict[str, np.ndarray] = {}

    def _char_ngrams(self, word: str) -> list[str]:
        """Extract character n-grams with boundary markers."""
        padded = f"<{word}>"
        ngrams = []
        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(padded) - n + 1):
                ngrams.append(padded[i : i + n])
        return ngrams

    @staticmethod
    def _stable_hash(s: str) -> int:
        """FNV-1a 32-bit hash for cross-run reproducibility (Python hash() is randomized)."""
        h = 2166136261
        for ch in s.encode("utf-8"):
            h ^= ch
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    def _ngram_hashes(self, word: str) -> list[int]:
        """Hash character n-grams to bucket indices."""
        return [self._stable_hash(ng) % self.n_buckets for ng in self._char_ngrams(word)]

    def get_word_vector(self, word: str) -> np.ndarray:
        """Get the vector representation of a word."""
        word_lower = word.lower().strip()
        if word_lower in self._word_cache:
            return self._word_cache[word_lower].copy()
        hashes = self._ngram_hashes(word_lower)
        if not hashes:
            return np.zeros(self.dim, dtype=np.float32)
        vec = self.bucket_vectors[hashes].mean(axis=0)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        self._word_cache[word_lower] = vec
        return vec.copy()

    def train_on_corpus(self, corpus: str, window: int = 5, lr: float = 0.05, epochs: int = 3) -> None:
        """Train word vectors via skip-gram-like co-occurrence updates."""
        words = corpus.lower().split()
        if len(words) < 2:
            return

        self._word_cache.clear()

        for epoch in range(epochs):
            current_lr = lr * (1.0 - epoch / epochs)
            for i, target_word in enumerate(words):
                target_hashes = self._ngram_hashes(target_word)
                if not target_hashes:
                    continue
                target_vec = self.bucket_vectors[target_hashes].mean(axis=0)

                # context window
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                for j in range(start, end):
                    if j == i:
                        continue
                    ctx_hashes = self._ngram_hashes(words[j])
                    if not ctx_hashes:
                        continue
                    ctx_vec = self.bucket_vectors[ctx_hashes].mean(axis=0)

                    # simple co-occurrence gradient: push target toward context
                    diff = ctx_vec - target_vec
                    update = current_lr * diff / len(target_hashes)
                    self.bucket_vectors[target_hashes] += update

        self._word_cache.clear()


def train_fasttext_baseline(
    corpus: str,
    dim: int = 100,
    min_n: int = 1,
    max_n: int = 6,
    seed: int = 42,
) -> CharNGramEmbedder:
    """Train a fastText-style character n-gram model on corpus."""
    model = CharNGramEmbedder(dim=dim, min_n=min_n, max_n=max_n, seed=seed)
    model.train_on_corpus(corpus)
    return model


def evaluate_fasttext_grounding_probe(model: CharNGramEmbedder) -> GroundingProbeResult:
    """Evaluate the 50-triple grounding probe on a CharNGramEmbedder."""

    def vector_fn(text: str) -> torch.Tensor:
        words = text.lower().split()
        if not words:
            return torch.zeros(model.dim)
        vecs = [model.get_word_vector(w) for w in words]
        avg = np.mean(vecs, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 1e-8:
            avg = avg / norm
        return torch.from_numpy(avg)

    return evaluate_grounding_probe(vector_fn)


# ---------------------------------------------------------------------------
# Combined baseline runner
# ---------------------------------------------------------------------------

@dataclass
class BaselineResults:
    """Aggregated results from all three baselines."""

    som_probe: GroundingProbeResult
    som_details: dict[str, Any]
    four_gram: dict[str, Any]
    fasttext_probe: GroundingProbeResult
    fasttext_details: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "online_som": {
                "grounding_probe_accuracy": self.som_probe.total_accuracy,
                "concrete_accuracy": self.som_probe.concrete_accuracy,
                "abstract_accuracy": self.som_probe.abstract_accuracy,
                "concreteness_gap": self.som_probe.concreteness_gap,
                "n_prototypes": self.som_details.get("n_prototypes"),
                "input_dim": self.som_details.get("input_dim"),
            },
            "four_gram": self.four_gram,
            "fasttext": {
                "grounding_probe_accuracy": self.fasttext_probe.total_accuracy,
                "concrete_accuracy": self.fasttext_probe.concrete_accuracy,
                "abstract_accuracy": self.fasttext_probe.abstract_accuracy,
                "concreteness_gap": self.fasttext_probe.concreteness_gap,
                "dim": self.fasttext_details.get("dim"),
            },
            "calibrated_target": self._calibrated_target(),
        }

    def _calibrated_target(self) -> dict[str, Any]:
        """Calibrate HECSN target based on fastText score (§8.1)."""
        ft_score = self.fasttext_probe.total_accuracy
        return {
            "fasttext_score": ft_score,
            "hecsn_text_only_target": ft_score,
            "hecsn_multimodal_target": ft_score + 0.05,
            "note": "HECSN multimodal should exceed fastText + 0.05; "
            "text-only should match fastText (§8.1).",
        }


def run_all_baselines(
    corpus: str,
    input_dim: int = 128,
    n_prototypes: int = 64,
    seed: int = 42,
) -> BaselineResults:
    """Run all three baselines on the given corpus.

    Args:
        corpus: Text corpus to train on (same as HECSN training corpus).
        input_dim: Dimensionality for SOM vectors.
        n_prototypes: Number of SOM prototypes (matches HECSN column count).
        seed: Random seed for reproducibility.

    Returns:
        BaselineResults with all three baseline evaluations.
    """
    # Baseline 1: Online SOM
    som = train_som_on_corpus(
        corpus, input_dim=input_dim, n_prototypes=n_prototypes, seed=seed
    )
    som_probe = evaluate_som_grounding_probe(som)

    # Baseline 2: 4-gram model
    four_gram_model = train_4gram_on_corpus(corpus)
    four_gram_eval = four_gram_model.evaluate_prediction_accuracy(corpus)

    # Baseline 3: fastText character n-grams
    ft_model = train_fasttext_baseline(corpus, dim=input_dim, seed=seed)
    ft_probe = evaluate_fasttext_grounding_probe(ft_model)

    return BaselineResults(
        som_probe=som_probe,
        som_details={"n_prototypes": n_prototypes, "input_dim": input_dim},
        four_gram=four_gram_eval,
        fasttext_probe=ft_probe,
        fasttext_details={"dim": input_dim},
    )
