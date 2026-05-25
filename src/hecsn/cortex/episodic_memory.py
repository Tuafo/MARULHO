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
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np

from hecsn.cortex.rate_limit import DEFAULT_MAX_RPM, SharedRateLimiter
from hecsn.semantics.grounding_text import match_terms, query_focused_text, salient_query_terms
from hecsn.semantics.provenance import Provenance

logger = logging.getLogger(__name__)


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
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)
    source_thought_id: str = ""       # Link to generating thought
    dream_origin: bool = False        # True if this episode began as a dream hypothesis

    def __post_init__(self) -> None:
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.last_accessed == 0.0:
            self.last_accessed = self.created_at
        if self.provenance == Provenance.DREAMED:
            self.dream_origin = True

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


@dataclass(frozen=True)
class EvidenceEpisodeMatch:
    """A ranked episode prepared for grounded cortex deliberation."""
    episode: Episode = field(compare=False, repr=False)
    focused_text: str
    matched_terms: tuple[str, ...] = ()
    score: float = 0.0
    grounded: bool = False
    lexical_coverage: float = 0.0
    semantic_similarity: float = 0.0

    @property
    def episode_id(self) -> str:
        return self.episode.episode_id


@dataclass(frozen=True)
class EvidenceRecallBundle:
    """Structured evidence bundle for query and wakeful deliberation."""
    target: str
    target_terms: tuple[str, ...] = ()
    grounded: tuple[EvidenceEpisodeMatch, ...] = ()
    support: tuple[EvidenceEpisodeMatch, ...] = ()
    grounded_coverage: float = 0.0
    combined_coverage: float = 0.0


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

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "kind": type(self).__name__,
            "model": "simple-hash-trigram",
            "dim": self.dim,
            "available": False,
            "degraded": False,
            "allow_fallback": True,
            "nim_calls": 0,
            "fallback_calls": 0,
            "error_calls": 0,
            "rate_limit_hits": 0,
            "cache_size": 0,
            "last_error": "",
        }


class NIMEmbedder:
    """NVIDIA NIM embedding service with shared budget awareness.

    Uses the OpenAI-compatible /v1/embeddings endpoint on NVIDIA NIM.
    Embedding calls share the same API-key rate limiter as cortex chat
    completions so Terminus stays under the global NIM request budget.

    When ``allow_fallback`` is enabled, failures degrade to SimpleEmbedder.
    In strict mode failures are surfaced through stats/telemetry and return a
    zero vector rather than silently changing embedding models.
    """

    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
    DEFAULT_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"  # Multimodal: text + vision
    DEFAULT_DIM = 2048

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        dim: int = DEFAULT_DIM,
        fallback_dim: int = 128,
        cache_size: int = 256,
        max_rpm: int = DEFAULT_MAX_RPM,
        allow_fallback: bool = True,
    ) -> None:
        import os
        import httpx

        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self._allow_fallback = allow_fallback
        self._fallback = SimpleEmbedder(dim=fallback_dim)
        self._cache: dict[str, np.ndarray] = {}
        self._cache_size = cache_size
        self._call_count = 0
        self._fallback_count = 0
        self._error_count = 0
        self._rate_limit_hits = 0
        self._last_error = ""
        self._degraded = False
        self._max_retries = 1

        if self._api_key:
            self._client = httpx.Client(
                timeout=httpx.Timeout(30.0),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            self._available = True
            self._rate_limiter = SharedRateLimiter.for_key(self._api_key, max_rpm=max_rpm)
        else:
            if not allow_fallback:
                raise RuntimeError(
                    "NVIDIA_API_KEY not set. NIMEmbedder requires a key unless allow_fallback=True."
                )
            self._client = None
            self._available = False
            self._rate_limiter = None
            self._degraded = True
            self._last_error = "NVIDIA_API_KEY not set"

    def embed(self, text: str) -> np.ndarray:
        """Produce a normalized embedding vector for text."""
        if not text or not text.strip():
            return np.zeros(self.dim, dtype=np.float32)

        cache_key = text[:200].lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._available and self._client is not None:
            result = self._call_nim_embedding(text)
            if result is not None:
                self._call_count += 1
                if len(self._cache) >= self._cache_size:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[cache_key] = result
                self._degraded = False
                self._last_error = ""
                return result

        return self._fallback_or_zero(text)

    def _fallback_or_zero(self, text: str) -> np.ndarray:
        self._degraded = True
        if self._allow_fallback:
            self._fallback_count += 1
            fb = self._fallback.embed(text)
            if fb.shape[0] == self.dim:
                return fb
            result = np.zeros(self.dim, dtype=np.float32)
            result[:fb.shape[0]] = fb
            norm = np.linalg.norm(result)
            if norm > 1e-8:
                result /= norm
            return result
        return np.zeros(self.dim, dtype=np.float32)

    @staticmethod
    def _retry_after_seconds(response: Any) -> float | None:
        value = response.headers.get("retry-after") if getattr(response, "headers", None) else None
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _post_embedding_payload(self, payload: dict[str, Any]) -> np.ndarray | None:
        if not self._available or self._client is None or self._rate_limiter is None:
            return None

        for attempt in range(1 + self._max_retries):
            try:
                self._rate_limiter.wait()
                resp = self._client.post(
                    f"{self.base_url}/embeddings",
                    json=payload,
                )
                if resp.status_code == 429:
                    self._rate_limit_hits += 1
                    wait = self._retry_after_seconds(resp) or (6.0 * (attempt + 1))
                    self._rate_limiter.backoff(wait)
                    self._last_error = f"HTTP 429: rate limited ({wait:.1f}s cooldown)"
                    self._degraded = True
                    if attempt < self._max_retries:
                        logger.info("NIM embedder 429, backing off %.1fs (attempt %d)", wait, attempt + 1)
                        continue
                    self._error_count += 1
                    return None
                if resp.status_code != 200:
                    self._error_count += 1
                    self._last_error = f"HTTP {resp.status_code}"
                    self._degraded = True
                    logger.warning("NIM embedder failed: HTTP %s", resp.status_code)
                    return None
                data = resp.json()
                embeddings = data.get("data", [])
                if not embeddings:
                    self._error_count += 1
                    self._last_error = "empty embedding payload"
                    self._degraded = True
                    return None
                vec = embeddings[0].get("embedding", [])
                if not vec:
                    self._error_count += 1
                    self._last_error = "missing embedding vector"
                    self._degraded = True
                    return None
                result = np.array(vec, dtype=np.float32)
                norm = np.linalg.norm(result)
                if norm > 1e-8:
                    result /= norm
                return result
            except Exception as exc:
                self._error_count += 1
                self._last_error = str(exc)
                self._degraded = True
                logger.warning("NIM embedder failed: %s", exc)
                return None
        return None

    def _call_nim_embedding(self, text: str) -> np.ndarray | None:
        """Call NVIDIA NIM embedding endpoint for text."""
        payload = {
            "model": self.model,
            "input": [text],
            "input_type": "query",
            "truncate": "END",
        }
        return self._post_embedding_payload(payload)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        return float(np.dot(a, b))

    def embed_image(self, image_base64: str, mime_type: str = "image/png") -> np.ndarray:
        """Embed an image using the multimodal VL model."""
        if not self._available or not self._client:
            self._degraded = True
            return np.zeros(self.dim, dtype=np.float32)

        data_uri = f"data:{mime_type};base64,{image_base64}"
        cache_key = f"img:{image_base64[:100]}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        payload = {
            "model": self.model,
            "input": [data_uri],
            "input_type": "passage",
            "truncate": "END",
        }
        result = self._post_embedding_payload(payload)
        if result is None:
            return np.zeros(self.dim, dtype=np.float32)
        self._call_count += 1
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = result
        self._degraded = False
        self._last_error = ""
        return result

    def embed_multimodal(self, text: str, image_base64: str | None = None, mime_type: str = "image/png") -> np.ndarray:
        """Embed text + optional image as a combined multimodal embedding.

        If image is provided, averages text and image embeddings (both
        from the same VL model space, so averaging is meaningful).
        """
        text_emb = self.embed(text)
        if image_base64 is None:
            return text_emb
        img_emb = self.embed_image(image_base64, mime_type)
        if np.linalg.norm(img_emb) < 1e-8:
            return text_emb
        # Weighted average (text gets more weight for Q&A retrieval)
        combined = 0.6 * text_emb + 0.4 * img_emb
        norm = np.linalg.norm(combined)
        if norm > 1e-8:
            combined /= norm
        return combined

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "kind": type(self).__name__,
            "model": self.model,
            "dim": self.dim,
            "available": self._available,
            "degraded": self._degraded,
            "allow_fallback": self._allow_fallback,
            "nim_calls": self._call_count,
            "fallback_calls": self._fallback_count,
            "error_calls": self._error_count,
            "rate_limit_hits": self._rate_limit_hits,
            "cache_size": len(self._cache),
            "last_error": self._last_error,
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
        metadata: Mapping[str, Any] | None = None,
    ) -> Episode:
        """Store a new episode in memory."""
        if not episode_id:
            episode_id = _make_episode_id(content)

        # Deduplicate: if exact same ID exists, update instead
        if episode_id in self._episodes:
            existing = self._episodes[episode_id]
            existing.salience = max(existing.salience, salience)
            if metadata:
                existing.metadata.update({str(key): value for key, value in dict(metadata).items()})
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
            metadata={str(key): value for key, value in dict(metadata or {}).items()},
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

    def recent_action_episodes(
        self,
        *,
        limit: int = 8,
        statuses: Sequence[str] | None = None,
    ) -> list[Episode]:
        allowed = {
            str(status).strip().lower()
            for status in list(statuses or ())
            if str(status).strip()
        }
        episodes = [
            ep
            for ep in self._episodes.values()
            if str(ep.metadata.get("observation_kind", "")).strip().lower() == "action"
        ]
        if allowed:
            episodes = [
                ep
                for ep in episodes
                if str(ep.metadata.get("verification_status", "")).strip().lower() in allowed
            ]
        episodes.sort(key=lambda ep: ep.created_at, reverse=True)
        return episodes[: max(1, int(limit))]

    # -- Retrieval --

    @staticmethod
    def _is_grounded_episode(ep: Episode) -> bool:
        observation_kind = str(ep.metadata.get("observation_kind", "")).strip().lower()
        return bool(ep.metadata.get("grounded", False)) or observation_kind in {"source", "sensory"} or ep.provenance in (
            Provenance.OBSERVED,
            Provenance.VERIFIED,
        )

    @staticmethod
    def _grounding_signal(ep: Episode) -> float:
        explicit = ep.metadata.get("grounding_signal")
        if explicit is not None:
            try:
                return max(0.0, min(1.0, float(explicit)))
            except (TypeError, ValueError):
                pass
        semantic_match = ep.metadata.get("semantic_match")
        try:
            semantic = max(0.0, min(1.0, float(semantic_match))) if semantic_match is not None else 0.0
        except (TypeError, ValueError):
            semantic = 0.0
        evidence_units = ep.metadata.get("evidence_unit_count", ep.metadata.get("evidence_window_count", 1))
        try:
            evidence_bonus = min(1.0, max(0.0, float(evidence_units)) / 8.0)
        except (TypeError, ValueError):
            evidence_bonus = 0.0
        return max(
            0.0,
            min(
                1.0,
                0.45 * ep.salience
                + 0.25 * ep.provenance.trust_weight
                + 0.20 * semantic
                + 0.10 * evidence_bonus,
            ),
        )

    @staticmethod
    def _metadata_focus_terms(ep: Episode) -> tuple[str, ...]:
        raw = ep.metadata.get("focus_terms")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return ()
        terms: list[str] = []
        seen: set[str] = set()
        for item in list(raw)[:6]:
            cleaned = " ".join(str(item).split()).strip()
            lowered = cleaned.lower()
            if not cleaned or lowered in seen:
                continue
            seen.add(lowered)
            terms.append(cleaned)
        return tuple(terms)

    @staticmethod
    def _episode_overlaps_avoidance(ep: Episode, avoid_topics: set[str] | None = None) -> bool:
        avoid_words = {str(topic).strip().lower() for topic in (avoid_topics or set()) if str(topic).strip()}
        if not avoid_words:
            return False
        for topic in ep.topics:
            topic_words = {
                word.strip(".,;:!?\"'()-").lower()
                for word in str(topic).split()
                if len(word) >= 3
            }
            if topic_words & avoid_words:
                return True
        return False

    @staticmethod
    def _target_term_coverage(target_terms: Sequence[str], matched_terms: Sequence[str]) -> float:
        if not target_terms:
            return 0.0
        target_set = {str(term).strip().lower() for term in target_terms if str(term).strip()}
        if not target_set:
            return 0.0
        matched_set = {str(term).strip().lower() for term in matched_terms if str(term).strip()}
        return min(1.0, float(len(target_set & matched_set)) / float(len(target_set)))

    @classmethod
    def _bundle_coverage(cls, target_terms: Sequence[str], matches: Sequence[EvidenceEpisodeMatch]) -> float:
        covered: set[str] = set()
        for match in matches:
            covered.update(str(term).strip().lower() for term in match.matched_terms if str(term).strip())
        return cls._target_term_coverage(target_terms, tuple(covered))

    def _score_target_episode(
        self,
        ep: Episode,
        *,
        target_emb: np.ndarray | None,
        target_terms: Sequence[str],
        grounded_priority: float = 0.0,
        skip_recent_inferred: bool = False,
    ) -> EvidenceEpisodeMatch | None:
        if ep.embedding is None or ep.provenance == Provenance.CONTRADICTED:
            return None
        if skip_recent_inferred and ep.provenance == Provenance.INFERRED and ep.age_seconds < 120.0:
            return None

        semantic_similarity = self.embedder.similarity(target_emb, ep.embedding) if target_emb is not None else 0.0
        focused_text = query_focused_text(ep.content, target_terms) if target_terms else ep.content
        focus_terms = self._metadata_focus_terms(ep)
        match_text = " ".join([*(str(topic) for topic in ep.topics), *focus_terms, focused_text]).strip()
        matched_terms = tuple(match_terms(target_terms, match_text)) if target_terms else ()
        lexical_coverage = self._target_term_coverage(target_terms, matched_terms)

        if target_terms and semantic_similarity <= 0.0 and lexical_coverage <= 0.0:
            return None

        grounded = self._is_grounded_episode(ep)
        grounding_signal = self._grounding_signal(ep)
        if target_terms:
            score = (
                0.44 * lexical_coverage
                + 0.22 * max(0.0, semantic_similarity)
                + 0.13 * max(0.0, min(1.0, ep.salience))
                + 0.09 * max(0.0, min(1.0, ep.provenance.trust_weight))
                + 0.04 * max(0.0, min(1.0, ep.confidence))
            )
        else:
            score = (
                0.34 * max(0.0, min(1.0, ep.recency_score))
                + 0.20 * max(0.0, min(1.0, ep.salience))
                + 0.16 * max(0.0, min(1.0, ep.provenance.trust_weight))
                + 0.08 * max(0.0, min(1.0, ep.confidence))
            )

        if grounded:
            score += grounded_priority + 0.10 * ep.recency_score + 0.14 * grounding_signal
            if ep.metadata.get("grounded", False):
                score += 0.08
        else:
            score += 0.02 * ep.recency_score
            if ep.provenance == Provenance.INFERRED:
                score -= 0.06
            elif ep.provenance == Provenance.DREAMED:
                score -= 0.10

        return EvidenceEpisodeMatch(
            episode=ep,
            focused_text=focused_text,
            matched_terms=matched_terms,
            score=max(0.0, float(score)),
            grounded=grounded,
            lexical_coverage=max(0.0, min(1.0, float(lexical_coverage))),
            semantic_similarity=max(0.0, float(semantic_similarity)),
        )

    @staticmethod
    def _select_evidence_matches(
        candidates: Sequence[EvidenceEpisodeMatch],
        *,
        top_k: int,
        covered_terms: Sequence[str] = (),
        prefer_new_terms: bool = True,
        allow_redundant: bool = True,
    ) -> list[EvidenceEpisodeMatch]:
        limit = max(0, int(top_k))
        if limit <= 0:
            return []

        covered = {str(term).strip().lower() for term in covered_terms if str(term).strip()}
        selected: list[EvidenceEpisodeMatch] = []
        deferred: list[EvidenceEpisodeMatch] = []
        seen_ids: set[str] = set()

        for match in candidates:
            if match.episode_id in seen_ids:
                continue
            match_terms_set = {str(term).strip().lower() for term in match.matched_terms if str(term).strip()}
            adds_new_terms = bool(match_terms_set - covered)
            if prefer_new_terms and covered and not adds_new_terms:
                deferred.append(match)
                continue
            selected.append(match)
            seen_ids.add(match.episode_id)
            covered.update(match_terms_set)
            if len(selected) >= limit:
                return selected

        if allow_redundant and len(selected) < limit:
            for match in deferred:
                if match.episode_id in seen_ids:
                    continue
                selected.append(match)
                seen_ids.add(match.episode_id)
                if len(selected) >= limit:
                    break
        return selected

    def _build_evidence_bundle(
        self,
        *,
        target: str,
        target_terms: Sequence[str],
        ranked_matches: Sequence[EvidenceEpisodeMatch],
        grounded_top_k: int,
        support_top_k: int,
    ) -> EvidenceRecallBundle:
        grounded_candidates = [match for match in ranked_matches if match.grounded]
        grounded = self._select_evidence_matches(
            grounded_candidates,
            top_k=max(0, int(grounded_top_k)),
            prefer_new_terms=bool(target_terms),
            allow_redundant=True,
        )
        grounded_ids = {match.episode_id for match in grounded}
        grounded_terms = [term for match in grounded for term in match.matched_terms]
        grounded_coverage = self._bundle_coverage(target_terms, grounded)

        support_candidates = [match for match in ranked_matches if match.episode_id not in grounded_ids]
        support = self._select_evidence_matches(
            support_candidates,
            top_k=max(0, int(support_top_k)),
            covered_terms=grounded_terms,
            prefer_new_terms=bool(target_terms),
            allow_redundant=(grounded_coverage < 0.6) if target_terms else True,
        )
        combined_coverage = self._bundle_coverage(target_terms, [*grounded, *support])

        for match in [*grounded, *support]:
            match.episode.touch()

        return EvidenceRecallBundle(
            target=target,
            target_terms=tuple(str(term) for term in target_terms if str(term).strip()),
            grounded=tuple(grounded),
            support=tuple(support),
            grounded_coverage=float(grounded_coverage),
            combined_coverage=float(combined_coverage),
        )

    def recent_grounded_episodes(
        self,
        *,
        top_k: int = 4,
        max_age_s: float = 1800.0,
        avoid_topics: set[str] | None = None,
    ) -> list[Episode]:
        candidates = [
            ep
            for ep in self._episodes.values()
            if self._is_grounded_episode(ep)
            and ep.provenance != Provenance.CONTRADICTED
            and not self._episode_overlaps_avoidance(ep, avoid_topics)
        ]
        if max_age_s > 0.0:
            recent = [ep for ep in candidates if ep.age_seconds <= float(max_age_s)]
            if recent:
                candidates = recent
        candidates.sort(
            key=lambda ep: (
                bool(ep.metadata.get("grounded", False)),
                self._grounding_signal(ep),
                ep.recency_score,
                ep.salience,
                ep.created_at,
            ),
            reverse=True,
        )
        return candidates[: max(0, int(top_k))]

    def recent_grounded_focus(
        self,
        *,
        top_k: int = 3,
        max_age_s: float = 1800.0,
        avoid_topics: set[str] | None = None,
    ) -> str:
        episodes = self.recent_grounded_episodes(
            top_k=max(1, int(top_k)),
            max_age_s=max_age_s,
            avoid_topics=avoid_topics,
        )
        if not episodes:
            return ""
        primary = episodes[0]
        focus_terms = list(self._metadata_focus_terms(primary))
        if focus_terms:
            return " ".join(focus_terms[:3])[:120]
        topic_terms = [
            " ".join(str(topic).split()).strip()
            for topic in primary.topics
            if " ".join(str(topic).split()).strip()
        ]
        if topic_terms:
            ordered = list(dict.fromkeys(topic_terms))[:3]
            return " ".join(ordered)[:120]
        tokens = salient_query_terms(primary.content)[:4]
        return " ".join(tokens)[:120]

    def recall_for_query(
        self,
        query: str,
        *,
        grounded_top_k: int = 4,
        support_top_k: int = 4,
    ) -> EvidenceRecallBundle:
        """Retrieve a query bundle with grounded evidence first."""
        if not self._episodes:
            return EvidenceRecallBundle(target=query)

        target = " ".join(str(query).split()).strip()
        target_terms = tuple(salient_query_terms(target)[:8])
        target_emb = self.embedder.embed(target) if target else None
        ranked_matches = [
            match
            for match in (
                self._score_target_episode(
                    ep,
                    target_emb=target_emb,
                    target_terms=target_terms,
                    grounded_priority=0.20,
                )
                for ep in self._episodes.values()
            )
            if match is not None
        ]
        ranked_matches.sort(
            key=lambda match: (
                match.score,
                match.grounded,
                match.episode.recency_score,
                match.episode.created_at,
            ),
            reverse=True,
        )
        return self._build_evidence_bundle(
            target=target,
            target_terms=target_terms,
            ranked_matches=ranked_matches,
            grounded_top_k=grounded_top_k,
            support_top_k=support_top_k,
        )

    def recall_for_deliberation(
        self,
        target: str = "",
        *,
        grounded_top_k: int = 3,
        support_top_k: int = 5,
        avoid_topics: set[str] | None = None,
        max_recent_grounded_age_s: float = 1800.0,
    ) -> EvidenceRecallBundle:
        """Retrieve a wakeful-thought bundle with recent grounded evidence first."""
        if not self._episodes:
            return EvidenceRecallBundle(target=" ".join(str(target).split()).strip())

        normalized_target = " ".join(str(target).split()).strip()
        derived_target = normalized_target or self.recent_grounded_focus(
            top_k=max(1, int(grounded_top_k)),
            max_age_s=max_recent_grounded_age_s,
            avoid_topics=avoid_topics,
        )
        target_terms = tuple(salient_query_terms(derived_target)[:8]) if derived_target else ()
        target_emb = self.embedder.embed(derived_target) if derived_target else None
        ranked_matches = [
            match
            for match in (
                self._score_target_episode(
                    ep,
                    target_emb=target_emb,
                    target_terms=target_terms,
                    grounded_priority=0.24 if not normalized_target else 0.18,
                    skip_recent_inferred=True,
                )
                for ep in self._episodes.values()
                if not self._episode_overlaps_avoidance(ep, avoid_topics)
            )
            if match is not None
        ]
        ranked_matches.sort(
            key=lambda match: (
                match.score,
                match.grounded,
                match.episode.recency_score,
                match.episode.created_at,
            ),
            reverse=True,
        )
        return self._build_evidence_bundle(
            target=derived_target,
            target_terms=target_terms,
            ranked_matches=ranked_matches,
            grounded_top_k=grounded_top_k,
            support_top_k=support_top_k,
        )

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

    def recall_dream_lineage(self) -> list[Episode]:
        """Get all episodes that originated as dream hypotheses.

        Includes current dreamed hypotheses plus those later graduated to
        VERIFIED or marked CONTRADICTED.
        """
        return [ep for ep in self._episodes.values() if ep.dream_origin]

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
            "embedder": dict(getattr(self.embedder, "stats", {"kind": type(self.embedder).__name__})),
        }

    def __len__(self) -> int:
        return self.size

    def __contains__(self, episode_id: str) -> bool:
        return episode_id in self._episodes

    def __iter__(self) -> Iterator[Episode]:
        return iter(self._episodes.values())

    def get(self, episode_id: str) -> Optional[Episode]:
        return self._episodes.get(episode_id)
