from __future__ import annotations

from collections import Counter, defaultdict
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


def _normalize_vector(value: torch.Tensor) -> torch.Tensor | None:
    vector = value.detach().clone().float().reshape(-1)
    if int(vector.numel()) <= 0:
        return None
    norm = float(vector.norm().item())
    if norm <= 1e-8:
        return None
    return F.normalize(vector, dim=0)


def _align_vectors(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target_dim = max(int(left.numel()), int(right.numel()))
    if int(left.numel()) < target_dim:
        left = F.pad(left, (0, target_dim - int(left.numel())))
    if int(right.numel()) < target_dim:
        right = F.pad(right, (0, target_dim - int(right.numel())))
    return left, right


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def ascii_codes(text: str) -> list[int]:
    return [ord(ch) if ord(ch) < 128 else 0 for ch in text]


def mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = _normalize_vector(left)
    right_norm = _normalize_vector(right)
    if left_norm is None or right_norm is None:
        return float("nan")
    left_aligned, right_aligned = _align_vectors(left_norm, right_norm)
    return float(torch.dot(left_aligned, right_aligned).item())


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


def temporal_coherence(
    routing_history: Sequence[tuple[str, int]],
    *,
    window: int | None = None,
    min_pattern_occurrences: int = 2,
) -> dict[str, Any]:
    if window is not None and window > 0:
        history = list(routing_history[-int(window) :])
    else:
        history = list(routing_history)
    if not history:
        return {
            "history_length": 0,
            "pattern_count": 0,
            "supported_pattern_count": 0,
            "sample_count": 0,
            "mean_coherence": None,
        }

    grouped: dict[str, list[int]] = defaultdict(list)
    for pattern_hash, winner_idx in history:
        grouped[str(pattern_hash)].append(int(winner_idx))

    supported_pattern_count = 0
    sample_count = 0
    coherences: list[float] = []
    minimum = max(2, int(min_pattern_occurrences))
    for winners in grouped.values():
        if len(winners) < minimum:
            continue
        supported_pattern_count += 1
        sample_count += len(winners)
        mode_count = Counter(winners).most_common(1)[0][1]
        coherences.append(float(mode_count) / float(len(winners)))

    return {
        "history_length": int(len(history)),
        "pattern_count": int(len(grouped)),
        "supported_pattern_count": int(supported_pattern_count),
        "sample_count": int(sample_count),
        "mean_coherence": _mean_or_none(coherences),
    }


def compositionality_score(
    vector_triples: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    scores: list[float] = []
    successes = 0
    for left, right, combined in vector_triples:
        left_norm = _normalize_vector(left)
        right_norm = _normalize_vector(right)
        combined_norm = _normalize_vector(combined)
        if left_norm is None or right_norm is None or combined_norm is None:
            continue
        left_aligned, right_aligned = _align_vectors(left_norm, right_norm)
        combined_aligned, _ = _align_vectors(combined_norm, left_aligned)
        left_aligned, right_aligned = _align_vectors(left_aligned, right_aligned)
        expected = left_aligned + right_aligned
        expected_norm = _normalize_vector(expected)
        if expected_norm is None:
            continue
        expected_norm, combined_aligned = _align_vectors(expected_norm, combined_aligned)
        score = float(torch.dot(expected_norm, combined_aligned).item())
        scores.append(score)
        if score > 0.0:
            successes += 1

    return {
        "sample_count": int(len(scores)),
        "mean_score": _mean_or_none(scores),
        "success_rate": None if not scores else float(successes / len(scores)),
    }


def grounding_probe(
    vector_triples: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    margins: list[float] = []
    correct = 0
    for anchor, positive, negative in vector_triples:
        positive_similarity = cosine_similarity(anchor, positive)
        negative_similarity = cosine_similarity(anchor, negative)
        if np.isnan(positive_similarity) or np.isnan(negative_similarity):
            continue
        positive_scores.append(float(positive_similarity))
        negative_scores.append(float(negative_similarity))
        margin = float(positive_similarity - negative_similarity)
        margins.append(margin)
        if margin > 0.0:
            correct += 1

    return {
        "sample_count": int(len(margins)),
        "accuracy": None if not margins else float(correct / len(margins)),
        "mean_margin": _mean_or_none(margins),
        "mean_positive_similarity": _mean_or_none(positive_scores),
        "mean_negative_similarity": _mean_or_none(negative_scores),
    }


def novelty_coverage_curve(
    novelty_events: Sequence[bool],
    token_checkpoints: Sequence[int],
    *,
    healthy_range: tuple[float, float] = (0.05, 0.20),
    saturation_threshold: float = 0.02,
    instability_threshold: float = 0.90,
) -> dict[str, Any]:
    if not novelty_events:
        return {
            "novelty_rate_by_checkpoint": [],
            "final_novelty_rate": None,
            "saturation_detected": False,
            "instability_detected": False,
            "healthy_range": {
                "min": float(healthy_range[0]),
                "max": float(healthy_range[1]),
            },
            "healthy_final_range": False,
        }

    max_token = int(len(novelty_events))
    checkpoints = sorted(
        {
            int(max(1, min(max_token, int(value))))
            for value in token_checkpoints
            if int(value) > 0
        }
    )
    if not checkpoints or checkpoints[-1] != max_token:
        checkpoints.append(max_token)

    rows: list[dict[str, Any]] = []
    start = 0
    for end in checkpoints:
        if end <= start:
            continue
        segment = novelty_events[start:end]
        rate = float(sum(1 for event in segment if bool(event))) / float(len(segment))
        rows.append(
            {
                "token_start": int(start + 1),
                "token_end": int(end),
                "window_size": int(len(segment)),
                "novelty_rate": float(rate),
            }
        )
        start = end

    final_rate = None if not rows else float(rows[-1]["novelty_rate"])
    healthy_min = float(healthy_range[0])
    healthy_max = float(healthy_range[1])
    return {
        "novelty_rate_by_checkpoint": rows,
        "final_novelty_rate": final_rate,
        "saturation_detected": bool(final_rate is not None and final_rate < saturation_threshold),
        "instability_detected": bool(final_rate is not None and final_rate > instability_threshold),
        "healthy_range": {"min": healthy_min, "max": healthy_max},
        "healthy_final_range": bool(final_rate is not None and healthy_min < final_rate < healthy_max),
    }


def representation_retention(
    before_vectors: Sequence[torch.Tensor],
    after_vectors: Sequence[torch.Tensor],
) -> dict[str, Any]:
    scores: list[float] = []
    for before, after in zip(before_vectors, after_vectors):
        score = cosine_similarity(before, after)
        if np.isnan(score):
            continue
        scores.append(float(score))

    return {
        "sample_count": int(len(scores)),
        "mean_retention": _mean_or_none(scores),
        "min_retention": None if not scores else float(min(scores)),
    }
