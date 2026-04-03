from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sklearn.metrics import davies_bouldin_score, silhouette_score  # type: ignore
except Exception:  # pragma: no cover
    davies_bouldin_score = None
    silhouette_score = None


TextScoreFn = Callable[[str, str], float]
TextVectorFn = Callable[[str], torch.Tensor]


def ascii_codes(text: str) -> list[int]:
    return [ord(ch) if ord(ch) < 128 else 0 for ch in text]


def mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 or right.numel() == 0:
        return float("nan")
    return float(F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0), dim=1).item())


def order_sensitivity(vector_for_text: TextVectorFn, eval_windows: Sequence[str], max_samples: int) -> dict[str, Any]:
    reversed_similarities: list[float] = []
    negative_similarities: list[float] = []
    total = len(eval_windows)
    if total == 0 or max_samples <= 0:
        return {
            "sample_count": 0,
            "mean_reversed_similarity": None,
            "mean_negative_similarity": None,
            "mean_margin": None,
        }

    offset = max(1, total // 3)
    for idx, window in enumerate(eval_windows):
        if len(window) < 4 or window == window[::-1] or len(set(window)) < 3:
            continue

        anchor = vector_for_text(window)
        reversed_window = vector_for_text(window[::-1])
        negative_window = eval_windows[(idx + offset) % total]
        if negative_window == window:
            continue

        reversed_similarity = cosine_similarity(anchor, reversed_window)
        negative_similarity = cosine_similarity(anchor, vector_for_text(negative_window))
        reversed_similarities.append(reversed_similarity)
        negative_similarities.append(negative_similarity)
        if len(reversed_similarities) >= max_samples:
            break

    if not reversed_similarities:
        return {
            "sample_count": 0,
            "mean_reversed_similarity": None,
            "mean_negative_similarity": None,
            "mean_margin": None,
        }

    margins = [negative - reversed_ for reversed_, negative in zip(reversed_similarities, negative_similarities)]
    return {
        "sample_count": int(len(reversed_similarities)),
        "mean_reversed_similarity": mean(reversed_similarities),
        "mean_negative_similarity": mean(negative_similarities),
        "mean_margin": mean(margins),
    }


def completion_coherence(score_fn: TextScoreFn, eval_windows: Sequence[str], max_samples: int) -> dict[str, Any]:
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    margins: list[float] = []
    successes = 0
    total = len(eval_windows)
    if total == 0 or max_samples <= 0:
        return {
            "sample_count": 0,
            "success_rate": None,
            "mean_margin": None,
            "mean_positive_score": None,
            "mean_negative_score": None,
        }

    offset = max(1, total // 2)
    for idx, window in enumerate(eval_windows):
        if len(window) < 4:
            continue

        prefix = window[:-1]
        negative_window = eval_windows[(idx + offset) % total]
        if negative_window == window or negative_window.startswith(prefix):
            continue

        positive_score = float(score_fn(prefix, window))
        negative_score = float(score_fn(prefix, negative_window))
        margin = positive_score - negative_score
        positive_scores.append(positive_score)
        negative_scores.append(negative_score)
        margins.append(margin)
        if margin > 0.0:
            successes += 1
        if len(margins) >= max_samples:
            break

    if not margins:
        return {
            "sample_count": 0,
            "success_rate": None,
            "mean_margin": None,
            "mean_positive_score": None,
            "mean_negative_score": None,
        }

    return {
        "sample_count": int(len(margins)),
        "success_rate": float(successes / len(margins)),
        "mean_margin": mean(margins),
        "mean_positive_score": mean(positive_scores),
        "mean_negative_score": mean(negative_scores),
    }


def vector_completion_coherence(vector_for_text: TextVectorFn, eval_windows: Sequence[str], max_samples: int) -> dict[str, Any]:
    return completion_coherence(
        lambda prefix, candidate: cosine_similarity(vector_for_text(prefix), vector_for_text(candidate)),
        eval_windows,
        max_samples,
    )


def clustering_metrics(
    embeddings: Sequence[torch.Tensor],
    labels: Sequence[int],
) -> tuple[Optional[float], Optional[float], str]:
    if silhouette_score is None or davies_bouldin_score is None:
        return None, None, "skipped: scikit-learn unavailable"
    if not embeddings or len(embeddings) != len(labels):
        return None, None, "skipped: no embeddings"
    if len(embeddings) < 20:
        return None, None, "skipped: too few examples"
    if len(set(labels)) < 2:
        return None, None, "skipped: single cluster"

    array = np.stack([embedding.detach().cpu().float().numpy() for embedding in embeddings], axis=0)
    return float(silhouette_score(array, labels)), float(davies_bouldin_score(array, labels)), "ok"