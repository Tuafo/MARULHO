"""Encoding ablation study (§4.2 — RTF as design choice).

Compares four character-to-vector encoding schemes on temporal coherence:
  1. uniform_ascii — bag-of-characters, uniform weight
  2. order_weighted_ascii — position-decayed ASCII weights (current RTF default)
  3. hashed_ngram — FNV-hashed character bigrams/trigrams
  4. rtf_burst — full RTF with chunk projection + burst layer

Temporal coherence is measured as the mean cosine similarity between
adjacent sliding windows (lag-1 autocorrelation) over a synthetic corpus.
Higher coherence → the encoding preserves temporal locality, which is
essential for STDP-based learning.

This module is a documentation/validation tool, NOT a runtime component.
The paper states RTF is "a design choice, not a claim" (§4.2); the
ablation verifies the choice is reasonable.
"""

from __future__ import annotations

from typing import Any

import torch

from marulho.config.model_config import MarulhoConfig
from marulho.data.rtf_encoder import RTFEncoder


_DEFAULT_CORPUS = (
    "the cat sat on the mat and the dog chased the cat across the yard "
    "then the bird flew over the tree and landed near the river where "
    "fish swim in the cold water under the bright warm sunlight that "
    "shines through the green leaves of the tall old oak tree growing "
    "beside the old stone wall near the garden gate where flowers bloom "
) * 20  # ~1000 tokens when repeated


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D vectors."""
    dot = float((a * b).sum().item())
    na = float(torch.norm(a, p=2).item())
    nb = float(torch.norm(b, p=2).item())
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


def _lag1_coherence(vectors: list[torch.Tensor]) -> float:
    """Mean cosine similarity between consecutive vector pairs."""
    if len(vectors) < 2:
        return 0.0
    sims = [_cosine_sim(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    return sum(sims) / len(sims)


def _encode_corpus(
    representation: str,
    corpus: str,
    input_dim: int,
    window_size: int,
) -> list[torch.Tensor]:
    """Encode corpus into vectors using a given representation mode."""
    cfg = MarulhoConfig()
    cfg.input_dim = input_dim
    cfg.window_size = window_size
    cfg.input_representation = representation
    encoder = RTFEncoder.from_config(cfg)
    vectors: list[torch.Tensor] = []
    for _, vec in encoder.iter_char_patterns(corpus, window_size, learn=False):
        vectors.append(vec.detach().clone())
    return vectors


def run_encoding_ablation(
    corpus: str | None = None,
    input_dim: int = 128,
    window_size: int = 12,
    min_tokens: int = 200,
) -> dict[str, Any]:
    """Run the 4-way encoding ablation and return results.

    Args:
        corpus: Text corpus. If None, uses default synthetic corpus.
        input_dim: Vector dimensionality.
        window_size: Character window size.
        min_tokens: Minimum number of vectors required for valid measurement.

    Returns:
        Dict with per-scheme coherence scores, ranking, and metadata.
    """
    text = corpus if corpus else _DEFAULT_CORPUS
    schemes = ["unigram_ascii", "order_weighted_ascii", "hashed_ngram"]
    results: dict[str, dict[str, Any]] = {}

    for scheme in schemes:
        try:
            vectors = _encode_corpus(scheme, text, input_dim, window_size)
        except Exception as exc:
            results[scheme] = {"coherence": 0.0, "n_vectors": 0, "error": str(exc)}
            continue

        coherence = _lag1_coherence(vectors)
        results[scheme] = {
            "coherence": round(coherence, 6),
            "n_vectors": len(vectors),
            "valid": len(vectors) >= min_tokens,
        }

    # RTF burst uses order_weighted_ascii with chunk projection enabled
    # The RTFEncoder always produces the same vectors via iter_char_patterns
    # since chunk projection is applied on top of the base representation.
    # We measure the full pipeline as "rtf_burst".
    cfg = MarulhoConfig()
    cfg.input_dim = input_dim
    cfg.window_size = window_size
    cfg.input_representation = "order_weighted_ascii"
    encoder = RTFEncoder.from_config(cfg)
    rtf_vectors: list[torch.Tensor] = []
    for _, vec in encoder.iter_char_patterns(text, window_size, learn=True):
        rtf_vectors.append(vec.detach().clone())
    rtf_coherence = _lag1_coherence(rtf_vectors)
    results["rtf_burst"] = {
        "coherence": round(rtf_coherence, 6),
        "n_vectors": len(rtf_vectors),
        "valid": len(rtf_vectors) >= min_tokens,
    }

    # Rank by coherence
    ranking = sorted(
        [(name, data["coherence"]) for name, data in results.items() if data.get("valid")],
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "schemes": results,
        "ranking": [{"scheme": name, "coherence": score} for name, score in ranking],
        "best_encoding": ranking[0][0] if ranking else "unknown",
        "metadata": {
            "input_dim": input_dim,
            "window_size": window_size,
            "corpus_length": len(text),
            "note": "RTF encoding is a design choice, not a claim (§4.2). "
            "This ablation documents temporal coherence across schemes.",
        },
    }
