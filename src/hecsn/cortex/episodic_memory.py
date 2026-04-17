"""Episodic memory — the hippocampus of the Terminus living brain.

Stores typed episodes (observations, inferences, hypotheses, dreams) with
provenance tracking, emotional valence, confidence, and lightweight
embedding-based similarity search.

Design principles:
- Episodes are text-based (not tensor-based like DualMemoryStore)
- Each episode has provenance: observed / inferred / dreamed / verified / contradicted
- Dream outputs are HYPOTHESES — they require external validation to graduate
- SNN surprise signals drive salience scoring
- Lightweight cosine-similarity search via normalized embeddings
- Capacity-bounded with importance-weighted eviction
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class Provenance(str, Enum):
    """How an episode was acquired — determines trust level."""
    OBSERVED = "observed"       # Direct external input
    INFERRED = "inferred"       # Model reasoning / thought
    DREAMED = "dreamed"         # Sleep recombination hypothesis
    VERIFIED = "verified"       # Externally confirmed
    CONTRADICTED = "contradicted"  # Proven wrong (kept for learning)

    @property
    def trust_weight(self) -> float:
        """Default trust multiplier for retrieval ranking."""
        return {
            Provenance.OBSERVED: 0.8,
            Provenance.INFERRED: 0.6,
            Provenance.DREAMED: 0.3,
            Provenance.VERIFIED: 1.0,
            Provenance.CONTRADICTED: 0.1,
        }[self]


@dataclass
class Episode:
    """A single episodic memory with full metadata."""
    episode_id: str
    content: str
    provenance: Provenance = Provenance.OBSERVED
    topics: tuple[str, ...] = ()
    emotional_valence: float = 0.0    # -1 to +1
    confidence: float = 0.5           # 0 to 1
    salience: float = 0.5             # SNN-computed importance
    created_at: float = 0.0           # time.time()
    last_accessed: float = 0.0
    access_count: int = 0
    replay_count: int = 0             # Times replayed during sleep
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    source_thought_id: str = ""       # Link to generating thought

    def __post_init__(self) -> None:
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.last_accessed == 0.0:
            self.last_accessed = self.created_at

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.created_at)

    @property
    def recency_score(self) -> float:
        """Exponential recency decay (half-life ~1 hour)."""
        hours = self.age_seconds / 3600.0
        return math.exp(-0.693 * hours)

    @property
    def composite_importance(self) -> float:
        """Combined importance for eviction decisions."""
        trust = self.provenance.trust_weight
        recency = self.recency_score
        access_boost = min(1.0, self.access_count * 0.1)
        replay_boost = min(0.5, self.replay_count * 0.05)
        return (
            0.3 * self.salience
            + 0.2 * trust
            + 0.2 * recency
            + 0.15 * self.confidence
            + 0.1 * access_boost
            + 0.05 * replay_boost
        )

    def touch(self) -> None:
        """Mark as recently accessed."""
        self.last_accessed = time.time()
        self.access_count += 1

    def graduate(self) -> None:
        """Promote a dreamed hypothesis to verified status."""
        if self.provenance == Provenance.DREAMED:
            self.provenance = Provenance.VERIFIED
            self.confidence = max(self.confidence, 0.7)

    def contradict(self) -> None:
        """Mark as contradicted by evidence."""
        self.provenance = Provenance.CONTRADICTED
        self.confidence = min(self.confidence, 0.2)


def _make_episode_id(content: str) -> str:
    """Deterministic short ID from content hash."""
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"ep-{h}"


class SimpleEmbedder:
    """Lightweight bag-of-characters embedder for memory indexing.

    This is a placeholder — will be replaced with a proper sentence
    embedder (e.g. via Ollama embed endpoint) once the basic system
    works. Uses character n-gram hashing into a fixed-size vector,
    similar to the existing RTF encoder approach.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        """Produce a normalized embedding vector for text."""
        vec = np.zeros(self.dim, dtype=np.float32)
        text_lower = text.lower()
        for i in range(len(text_lower) - 2):
            trigram = text_lower[i:i + 3]
            h = hash(trigram) % self.dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm
        return vec

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        return float(np.dot(a, b))


class EpisodicMemory:
    """The hippocampus — stores and retrieves episodic memories.

    Separate stores by provenance type, with unified retrieval.
    Capacity-bounded with importance-weighted eviction.
    """

    def __init__(
        self,
        capacity: int = 2048,
        embedder: Optional[SimpleEmbedder] = None,
        embedding_dim: int = 128,
    ) -> None:
        self.capacity = capacity
        self.embedder = embedder or SimpleEmbedder(dim=embedding_dim)
        self._episodes: dict[str, Episode] = {}
        self._topic_index: dict[str, set[str]] = defaultdict(set)
        self._total_stored = 0
        self._total_evicted = 0

    # -- Storage --

    def store(
        self,
        content: str,
        provenance: Provenance = Provenance.OBSERVED,
        topics: Sequence[str] = (),
        emotional_valence: float = 0.0,
        confidence: float = 0.5,
        salience: float = 0.5,
        source_thought_id: str = "",
        episode_id: str = "",
    ) -> Episode:
        """Store a new episode in memory."""
        if not episode_id:
            episode_id = _make_episode_id(content)

        # Deduplicate: if exact same ID exists, update instead
        if episode_id in self._episodes:
            existing = self._episodes[episode_id]
            existing.salience = max(existing.salience, salience)
            existing.access_count += 1
            existing.touch()
            return existing

        embedding = self.embedder.embed(content)

        episode = Episode(
            episode_id=episode_id,
            content=content,
            provenance=provenance,
            topics=tuple(topics),
            emotional_valence=emotional_valence,
            confidence=confidence,
            salience=salience,
            embedding=embedding,
            source_thought_id=source_thought_id,
        )

        # Evict if at capacity
        if len(self._episodes) >= self.capacity:
            self._evict_least_important()

        self._episodes[episode_id] = episode
        for topic in episode.topics:
            self._topic_index[topic.lower()].add(episode_id)
        self._total_stored += 1

        return episode

    def _evict_least_important(self) -> None:
        """Remove the least important episode to make room."""
        if not self._episodes:
            return
        # Never evict verified episodes if alternatives exist
        candidates = [
            ep for ep in self._episodes.values()
            if ep.provenance != Provenance.VERIFIED
        ]
        if not candidates:
            candidates = list(self._episodes.values())

        victim = min(candidates, key=lambda ep: ep.composite_importance)
        self.remove(victim.episode_id)
        self._total_evicted += 1

    def remove(self, episode_id: str) -> Optional[Episode]:
        """Remove an episode by ID."""
        ep = self._episodes.pop(episode_id, None)
        if ep:
            for topic in ep.topics:
                topic_set = self._topic_index.get(topic.lower())
                if topic_set:
                    topic_set.discard(episode_id)
        return ep

    # -- Retrieval --

    def recall_by_similarity(
        self,
        query: str,
        top_k: int = 5,
        min_trust: float = 0.0,
    ) -> list[Episode]:
        """Retrieve most similar episodes to a query string."""
        if not self._episodes:
            return []

        query_emb = self.embedder.embed(query)
        scored: list[tuple[float, Episode]] = []

        for ep in self._episodes.values():
            if ep.provenance.trust_weight < min_trust:
                continue
            if ep.embedding is None:
                continue
            sim = self.embedder.similarity(query_emb, ep.embedding)
            # Weight by trust and salience
            score = sim * ep.provenance.trust_weight * (0.5 + 0.5 * ep.salience)
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [ep for _, ep in scored[:top_k]]
        for ep in results:
            ep.touch()
        return results

    def recall_diverse(
        self,
        top_k: int = 5,
        avoid_topics: set[str] | None = None,
    ) -> list[Episode]:
        """Retrieve a diverse set of memories — prioritise OBSERVED over INFERRED.

        Specifically avoids episodes whose topics overlap with *avoid_topics*,
        breaking the echo-chamber effect where the LLM only sees its own
        ruminating thoughts as memories.
        """
        if not self._episodes:
            return []

        avoid = {t.lower() for t in (avoid_topics or set())}

        # Separate observed/external from self-generated
        observed: list[Episode] = []
        other: list[Episode] = []
        for ep in self._episodes.values():
            ep_topics = {t.lower() for t in ep.topics}
            if avoid and ep_topics & avoid:
                continue  # skip over-represented topics
            if ep.provenance in (Provenance.OBSERVED, Provenance.VERIFIED):
                observed.append(ep)
            else:
                other.append(ep)

        # Prefer observed (external) content, supplement with diverse inferred
        observed.sort(key=lambda ep: ep.created_at, reverse=True)
        other.sort(key=lambda ep: ep.composite_importance, reverse=True)

        results = observed[:top_k]
        remaining = top_k - len(results)
        if remaining > 0:
            results.extend(other[:remaining])

        for ep in results:
            ep.touch()
        return results

    def recall_by_topic(self, topic: str, top_k: int = 10) -> list[Episode]:
        """Retrieve episodes tagged with a specific topic."""
        ids = self._topic_index.get(topic.lower(), set())
        episodes = [self._episodes[eid] for eid in ids if eid in self._episodes]
        episodes.sort(key=lambda ep: ep.composite_importance, reverse=True)
        return episodes[:top_k]

    def recall_recent(self, top_k: int = 5) -> list[Episode]:
        """Retrieve most recent episodes."""
        episodes = sorted(
            self._episodes.values(),
            key=lambda ep: ep.created_at,
            reverse=True,
        )
        return episodes[:top_k]

    def recall_for_sleep(self, top_k: int = 20) -> list[Episode]:
        """Select episodes for sleep replay — high salience, not yet replayed much."""
        candidates = [
            ep for ep in self._episodes.values()
            if ep.provenance not in (Provenance.CONTRADICTED,)
        ]
        candidates.sort(
            key=lambda ep: ep.salience * (1.0 / (1.0 + ep.replay_count)),
            reverse=True,
        )
        return candidates[:top_k]

    def recall_hypotheses(self) -> list[Episode]:
        """Get all dreamed hypotheses awaiting validation."""
        return [
            ep for ep in self._episodes.values()
            if ep.provenance == Provenance.DREAMED
        ]

    # -- Lifecycle --

    def graduate_hypothesis(self, episode_id: str) -> bool:
        """Promote a dreamed episode to verified."""
        ep = self._episodes.get(episode_id)
        if ep and ep.provenance == Provenance.DREAMED:
            ep.graduate()
            return True
        return False

    def contradict_episode(self, episode_id: str) -> bool:
        """Mark an episode as contradicted."""
        ep = self._episodes.get(episode_id)
        if ep:
            ep.contradict()
            return True
        return False

    # -- Stats --

    @property
    def size(self) -> int:
        return len(self._episodes)

    @property
    def stats(self) -> dict[str, Any]:
        prov_counts: dict[str, int] = defaultdict(int)
        for ep in self._episodes.values():
            prov_counts[ep.provenance.value] += 1
        return {
            "size": self.size,
            "capacity": self.capacity,
            "fill_ratio": self.size / max(1, self.capacity),
            "total_stored": self._total_stored,
            "total_evicted": self._total_evicted,
            "provenance_distribution": dict(prov_counts),
            "mean_salience": (
                sum(ep.salience for ep in self._episodes.values()) / max(1, self.size)
            ),
            "mean_confidence": (
                sum(ep.confidence for ep in self._episodes.values()) / max(1, self.size)
            ),
        }

    def __len__(self) -> int:
        return self.size

    def __contains__(self, episode_id: str) -> bool:
        return episode_id in self._episodes

    def __iter__(self) -> Iterator[Episode]:
        return iter(self._episodes.values())

    def get(self, episode_id: str) -> Optional[Episode]:
        return self._episodes.get(episode_id)
