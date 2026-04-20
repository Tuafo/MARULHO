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

    This is a simple fallback embedder -- use NIMEmbedder for production.
    Uses character n-gram hashing into a fixed-size vector,
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


class NIMEmbedder:
    """NVIDIA NIM embedding service — replaces SimpleEmbedder with frontier quality.

    Uses the OpenAI-compatible /v1/embeddings endpoint on NVIDIA NIM.
    Falls back to SimpleEmbedder when NIM is unavailable (no API key,
    network error, rate limit exceeded).

    Supports caching to reduce API calls for repeated queries.
    """

    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
    DEFAULT_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
    DEFAULT_DIM = 2048

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        dim: int = DEFAULT_DIM,
        fallback_dim: int = 128,
        cache_size: int = 256,
    ) -> None:
        import os

        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self._fallback = SimpleEmbedder(dim=fallback_dim)
        self._cache: dict[str, np.ndarray] = {}
        self._cache_size = cache_size
        self._call_count = 0
        self._fallback_count = 0

        if self._api_key:
            import httpx
            self._client = httpx.Client(
                timeout=httpx.Timeout(30.0),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            self._available = True
        else:
            self._client = None
            self._available = False

    def embed(self, text: str) -> np.ndarray:
        """Produce a normalized embedding vector for text.

        Tries NIM API first, falls back to SimpleEmbedder on failure.
        """
        if not text or not text.strip():
            return np.zeros(self.dim, dtype=np.float32)

        # Check cache
        cache_key = text[:200].lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try NIM
        if self._available and self._client is not None:
            try:
                result = self._call_nim_embedding(text)
                if result is not None:
                    self._call_count += 1
                    # Cache management
                    if len(self._cache) >= self._cache_size:
                        self._cache.pop(next(iter(self._cache)))
                    self._cache[cache_key] = result
                    return result
            except Exception:
                self._fallback_count += 1

        # Fallback to SimpleEmbedder, then pad/resize to target dim
        fb = self._fallback.embed(text)
        if fb.shape[0] == self.dim:
            return fb
        result = np.zeros(self.dim, dtype=np.float32)
        result[:fb.shape[0]] = fb
        norm = np.linalg.norm(result)
        if norm > 1e-8:
            result /= norm
        return result

    def _call_nim_embedding(self, text: str) -> np.ndarray | None:
        """Call NVIDIA NIM embedding endpoint."""
        payload = {
            "model": self.model,
            "input": [text],
            "input_type": "query",
            "truncate": "END",
        }
        resp = self._client.post(  # type: ignore[union-attr]
            f"{self.base_url}/embeddings",
            json=payload,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        embeddings = data.get("data", [])
        if not embeddings:
            return None
        vec = embeddings[0].get("embedding", [])
        if not vec:
            return None
        result = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(result)
        if norm > 1e-8:
            result /= norm
        return result

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        return float(np.dot(a, b))

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "nim_calls": self._call_count,
            "fallback_calls": self._fallback_count,
            "cache_size": len(self._cache),
            "available": self._available,
            "model": self.model,
        }

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class EpisodicMemory:
    """The hippocampus — stores and retrieves episodic memories.

    Separate stores by provenance type, with unified retrieval.
    Capacity-bounded with importance-weighted eviction.
    """

    def __init__(
        self,
        capacity: int = 2048,
        embedder: Optional[SimpleEmbedder | NIMEmbedder] = None,
        embedding_dim: int = 128,
        use_nim_embeddings: bool = False,
    ) -> None:
        self.capacity = capacity
        if embedder is not None:
            self.embedder = embedder
        elif use_nim_embeddings:
            self.embedder = NIMEmbedder()
        else:
            self.embedder = SimpleEmbedder(dim=embedding_dim)
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

        Uses word-level avoidance: if *avoid_topics* contains the word
        "pottery", then an episode tagged "Neolithic pottery" is also
        filtered out.  This catches semantic clusters that phrase-level
        matching misses.
        """
        if not self._episodes:
            return []

        avoid_words = {t.lower() for t in (avoid_topics or set())}

        def _episode_overlaps_avoidance(ep: Episode) -> bool:
            if not avoid_words:
                return False
            for topic in ep.topics:
                topic_words = {w.strip(".,;:!?'\"()-").lower()
                               for w in topic.split() if len(w) >= 3}
                if topic_words & avoid_words:
                    return True
            return False

        # Separate observed/external from self-generated
        observed: list[Episode] = []
        other: list[Episode] = []
        skipped: list[Episode] = []
        for ep in self._episodes.values():
            if _episode_overlaps_avoidance(ep):
                skipped.append(ep)
                continue
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

        # If avoidance is too aggressive and we found nothing,
        # fall back to recent observations (even if they match avoidance words)
        if not results and skipped:
            skipped.sort(key=lambda ep: ep.created_at, reverse=True)
            obs_skipped = [ep for ep in skipped
                          if ep.provenance in (Provenance.OBSERVED, Provenance.VERIFIED)]
            results = (obs_skipped or skipped)[:top_k]

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
