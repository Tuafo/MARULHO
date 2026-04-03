from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Sequence

import torch
import torch.nn.functional as F


class OnlineKMeans:
    """Simple cosine-space online k-means baseline for streaming comparisons."""

    def __init__(self, n_clusters: int, feature_dim: int, device: torch.device | str = "cpu") -> None:
        self.n_clusters = max(1, int(n_clusters))
        self.feature_dim = max(1, int(feature_dim))
        self.device = torch.device(device)
        self.centroids = torch.zeros(self.n_clusters, self.feature_dim, device=self.device)
        self.counts = torch.zeros(self.n_clusters, dtype=torch.long, device=self.device)
        self.n_initialized = 0

    def _normalize(self, feature: torch.Tensor) -> torch.Tensor:
        return F.normalize(feature.to(self.device).float(), dim=0)

    def partial_fit(self, feature: torch.Tensor) -> int:
        normalized = self._normalize(feature)
        if self.n_initialized < self.n_clusters:
            index = self.n_initialized
            self.centroids[index] = normalized
            self.counts[index] = 1
            self.n_initialized += 1
            return int(index)

        similarities = torch.mv(self.centroids[: self.n_initialized], normalized)
        index = int(torch.argmax(similarities).item())
        self.counts[index] += 1
        eta = 1.0 / float(self.counts[index].item())
        updated = (1.0 - eta) * self.centroids[index] + eta * normalized
        self.centroids[index] = F.normalize(updated, dim=0)
        return index

    def fit(self, features: Sequence[torch.Tensor]) -> None:
        for feature in features:
            self.partial_fit(feature)

    def predict(self, feature: torch.Tensor) -> int:
        if self.n_initialized == 0:
            raise RuntimeError("OnlineKMeans must be fit before predict")
        normalized = self._normalize(feature)
        similarities = torch.mv(self.centroids[: self.n_initialized], normalized)
        return int(torch.argmax(similarities).item())

    def predict_many(self, features: Sequence[torch.Tensor]) -> list[int]:
        return [self.predict(feature) for feature in features]

    def mean_assignment_distance(self, features: Sequence[torch.Tensor]) -> float:
        if self.n_initialized == 0 or not features:
            return float("nan")

        distances: list[float] = []
        for feature in features:
            normalized = self._normalize(feature)
            similarities = torch.mv(self.centroids[: self.n_initialized], normalized)
            distances.append(float(1.0 - similarities.max().item()))
        return float(sum(distances) / len(distances))


class CharNGramMemory:
    """Count-based character n-gram completion baseline kept for Stage-0 behavior checks."""

    def __init__(self, max_context: int = 4, backoff: float = 0.5) -> None:
        self.max_context = max(1, int(max_context))
        self.backoff = float(max(0.0, backoff))
        self.next_char_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.alphabet: set[str] = set()

    def partial_fit_window(self, window: str) -> None:
        text = str(window)
        if not text:
            return

        self.alphabet.update(text)
        for end in range(1, len(text)):
            next_char = text[end]
            history = text[:end]
            self.next_char_counts[""][next_char] += 1
            max_ctx = min(self.max_context, len(history))
            for ctx_len in range(1, max_ctx + 1):
                context = history[-ctx_len:]
                self.next_char_counts[context][next_char] += 1

    def fit(self, windows: Sequence[str]) -> None:
        for window in windows:
            self.partial_fit_window(window)

    def _next_char_log_prob(self, prefix: str, next_char: str) -> float:
        alphabet_size = max(1, len(self.alphabet))
        max_ctx = min(self.max_context, len(prefix))
        weighted_log_prob = 0.0
        total_weight = 0.0
        weight = 1.0

        for ctx_len in range(max_ctx, -1, -1):
            context = prefix[-ctx_len:] if ctx_len > 0 else ""
            counts = self.next_char_counts.get(context)
            if counts is None or not counts:
                weight *= self.backoff
                continue

            total = sum(counts.values())
            prob = float(counts[next_char] + 1.0) / float(total + alphabet_size)
            weighted_log_prob += weight * math.log(prob)
            total_weight += weight
            weight *= self.backoff

        if total_weight <= 0.0:
            return -math.log(float(alphabet_size))
        return float(weighted_log_prob / total_weight)

    def completion_score(self, prefix: str, candidate: str) -> float:
        if not candidate or len(candidate) <= len(prefix):
            return float("-inf")

        score = 0.0
        current = prefix
        mismatch_penalty = 0.0
        tail = candidate
        if candidate.startswith(prefix):
            tail = candidate[len(prefix):]
        else:
            mismatch_penalty = float(len(prefix))

        for ch in tail:
            score += self._next_char_log_prob(current, ch)
            current = current + ch
        return float(score - mismatch_penalty)