"""Semantic n-gram encoder with GloVe-initialized bucket embeddings.

Replaces ASCII-position encoding with character n-gram composition through
pre-trained embedding space. Architecture:

    chars → token-boundary n-grams → FNV-1a hash → bucket lookup
    → average → adapter → split-sign → [2*embed_dim] nonneg L2-norm

Split-sign encoding [ReLU(x), ReLU(-x)] preserves sign information while
maintaining the system-wide nonneg invariant.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Iterable, Iterator, List, Sequence

import torch

if TYPE_CHECKING:
    from hecsn.config.model_config import HECSNConfig

from hecsn.data.rtf_encoder import LearnedChunkingLayer

logger = logging.getLogger(__name__)


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    total = float(torch.norm(vector.float(), p=2).item())
    if total <= 0.0:
        return torch.zeros_like(vector, dtype=torch.float32)
    return vector.float() / (torch.norm(vector.float(), p=2) + 1e-8)


def _ascii_code(ch: str) -> int:
    code = ord(ch)
    return int(code if code < 128 else 0)


class SemanticEncoder:
    """Semantic n-gram encoder implementing BaseEncoder protocol.

    Produces nonneg L2-normalized feature vectors from character n-gram
    compositions through pre-trained embedding buckets.
    """

    def __init__(
        self,
        *,
        n_buckets: int = 10_000,
        embed_dim: int = 64,
        window_size: int = 10,
        top_k_sparse: int = 0,
        ngram_min_n: int = 2,
        ngram_max_n: int = 4,
        enable_learned_chunking: bool = False,
        learned_chunk_detector_count: int = 128,
        learned_chunk_min_len: int = 2,
        learned_chunk_max_len: int = 12,
        learned_chunk_feature_mode: str = "blend",
        learned_chunk_concat_dim: int = 128,
        learned_chunk_blend: float = 0.5,
        learned_chunk_similarity_floor: float = 0.30,
        learned_chunk_boundary_threshold: float = 0.08,
        learned_chunk_update_lr: float = 0.25,
        learned_chunk_association_blend: float = 0.35,
        learned_chunk_association_lr: float = 0.15,
        learned_chunk_association_decay: float = 0.995,
    ) -> None:
        self.n_buckets = int(n_buckets)
        self.embed_dim = int(embed_dim)
        self.window_size = int(window_size)
        self.top_k_sparse = int(top_k_sparse)
        self.ngram_min_n = int(ngram_min_n)
        self.ngram_max_n = int(ngram_max_n)
        self.learned_chunk_feature_mode = str(learned_chunk_feature_mode)
        self.learned_chunk_concat_dim = int(learned_chunk_concat_dim)
        self.learned_chunk_blend = float(learned_chunk_blend)

        # Bucket embeddings [n_buckets, embed_dim]
        self.bucket_embeddings = torch.randn(n_buckets, embed_dim, dtype=torch.float32) * 0.01
        # Trainable diagonal adapter
        self.adapter = torch.ones(embed_dim, dtype=torch.float32)
        self._glove_initialized = False

        self.learned_chunking = (
            LearnedChunkingLayer(
                n_detectors=learned_chunk_detector_count,
                min_chunk_len=learned_chunk_min_len,
                max_chunk_len=learned_chunk_max_len,
                similarity_floor=learned_chunk_similarity_floor,
                boundary_threshold=learned_chunk_boundary_threshold,
                update_lr=learned_chunk_update_lr,
                association_blend=learned_chunk_association_blend,
                association_lr=learned_chunk_association_lr,
                association_decay=learned_chunk_association_decay,
            )
            if enable_learned_chunking
            else None
        )

    @classmethod
    def from_config(cls, config: "HECSNConfig") -> "SemanticEncoder":
        return cls(
            n_buckets=config.semantic_n_buckets,
            embed_dim=config.semantic_embed_dim,
            window_size=config.window_size,
            top_k_sparse=config.semantic_top_k_sparse,
            enable_learned_chunking=config.enable_learned_chunking,
            learned_chunk_detector_count=config.learned_chunk_detector_count,
            learned_chunk_min_len=config.learned_chunk_min_len,
            learned_chunk_max_len=config.learned_chunk_max_len,
            learned_chunk_feature_mode=config.learned_chunk_feature_mode,
            learned_chunk_concat_dim=config.learned_chunk_concat_dim,
            learned_chunk_blend=config.learned_chunk_blend,
            learned_chunk_similarity_floor=config.learned_chunk_similarity_floor,
            learned_chunk_boundary_threshold=config.learned_chunk_boundary_threshold,
            learned_chunk_update_lr=config.learned_chunk_update_lr,
            learned_chunk_association_blend=config.learned_chunk_association_blend,
            learned_chunk_association_lr=config.learned_chunk_association_lr,
            learned_chunk_association_decay=config.learned_chunk_association_decay,
        )

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def base_output_dim(self) -> int:
        return 2 * self.embed_dim

    @property
    def uses_learned_chunking(self) -> bool:
        return self.learned_chunking is not None

    @property
    def uses_concat_chunk_channel(self) -> bool:
        return self.learned_chunking is not None and self.learned_chunk_feature_mode == "concat"

    @property
    def chunk_output_dim(self) -> int:
        if self.uses_concat_chunk_channel:
            return self.learned_chunk_concat_dim
        return self.base_output_dim

    @property
    def chunk_projection_work_dim(self) -> int:
        if self.uses_concat_chunk_channel:
            return self.output_dim
        return self.base_output_dim

    @property
    def output_dim(self) -> int:
        if self.uses_concat_chunk_channel:
            return self.base_output_dim + self.learned_chunk_concat_dim
        return self.base_output_dim

    # ── N-gram hashing ──────────────────────────────────────────────────

    def _hash_ngram_to_bucket(self, codes: Sequence[int]) -> int:
        """FNV-1a hash of character codes to bucket index."""
        h = 2166136261
        for c in codes:
            h ^= int(c) + 1
            h = (h * 16777619) & 0xFFFFFFFF
        return int(h % max(1, self.n_buckets))

    def _collect_ngram_buckets(self, codes: Sequence[int]) -> List[int]:
        """Collect all n-gram bucket indices for a sequence of character codes."""
        buckets: List[int] = []
        n_codes = len(codes)
        if n_codes == 0:
            return buckets

        min_n = min(self.ngram_min_n, n_codes)
        max_n = min(self.ngram_max_n, n_codes)

        for n in range(min_n, max_n + 1):
            for start in range(n_codes - n + 1):
                bucket = self._hash_ngram_to_bucket(codes[start : start + n])
                buckets.append(bucket)

        if not buckets:
            for c in codes:
                h = (2166136261 ^ (int(c) + 1)) * 16777619 & 0xFFFFFFFF
                buckets.append(int(h % max(1, self.n_buckets)))

        return buckets

    # ── Core semantic computation ───────────────────────────────────────

    def _raw_semantic_vector(self, codes: Sequence[int]) -> torch.Tensor:
        """Compute raw semantic embedding from character codes."""
        if not codes:
            return torch.zeros(self.embed_dim, dtype=torch.float32)

        buckets = self._collect_ngram_buckets(codes)
        if not buckets:
            return torch.zeros(self.embed_dim, dtype=torch.float32)

        bucket_indices = torch.tensor(buckets, dtype=torch.long)
        raw = self.bucket_embeddings[bucket_indices].mean(dim=0)
        return raw * self.adapter

    def _split_sign_encode(self, raw: torch.Tensor) -> torch.Tensor:
        """Split-sign encoding: [ReLU(x), ReLU(-x)] → nonneg L2-normalized."""
        pos = torch.relu(raw)
        neg = torch.relu(-raw)
        combined = torch.cat([pos, neg])

        if self.top_k_sparse > 0:
            k = min(self.top_k_sparse, combined.numel())
            topk_vals, topk_idx = combined.topk(k)
            sparse = torch.zeros_like(combined)
            sparse[topk_idx] = topk_vals
            combined = sparse

        return _normalize(combined)

    def _base_feature_vector(self, codes: Sequence[int]) -> torch.Tensor:
        raw = self._raw_semantic_vector(codes)
        return self._split_sign_encode(raw)

    # ── Token boundary handling ─────────────────────────────────────────

    @staticmethod
    def _is_token_boundary(ch: str) -> bool:
        return ch in " \t\n\r" or (not ch.isalnum() and ch not in ("'", "-"))

    @staticmethod
    def _token_aware_codes(
        window_codes: List[int],
        token_codes: List[int],
    ) -> Sequence[int]:
        if token_codes:
            return token_codes
        return window_codes

    # ── Chunk handling (same logic as RTFEncoder) ───────────────────────

    def _project_chunk_vector(self, detector_vector: torch.Tensor) -> torch.Tensor:
        projected = torch.zeros(self.chunk_projection_work_dim, dtype=torch.float32)
        if detector_vector.numel() == 0 or float(detector_vector.sum().item()) <= 0.0:
            return projected
        if int(detector_vector.numel()) == int(projected.numel()):
            return _normalize(detector_vector.float())
        for index, value in enumerate(detector_vector.tolist()):
            if float(value) <= 0.0:
                continue
            projected[index % int(projected.numel())] += float(value)
        return _normalize(projected)

    def _chunk_signature_vector(self, chunk_codes: Sequence[int]) -> torch.Tensor:
        signature = torch.zeros(self.chunk_projection_work_dim, dtype=torch.float32)
        if not chunk_codes:
            return signature
        rolling = 2166136261
        for code in chunk_codes:
            rolling ^= int(code) + 1
            rolling = (rolling * 16777619) & 0xFFFFFFFF
            signature[int(rolling % max(1, self.chunk_projection_work_dim))] += 1.0
        for left, right in zip(chunk_codes, chunk_codes[1:]):
            pair_hash = ((int(left) + 17) * 1315423911) ^ ((int(right) + 31) * 2654435761)
            signature[int(pair_hash % max(1, self.chunk_projection_work_dim))] += 0.75
        signature[int(len(chunk_codes) % max(1, self.chunk_projection_work_dim))] += 0.5
        return _normalize(signature)

    def _chunk_projection(
        self,
        *,
        chunk_state: torch.Tensor | None = None,
        chunk_codes: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if self.learned_chunking is None or chunk_state is None:
            return torch.zeros(self.chunk_output_dim, dtype=torch.float32)
        projected = self._project_chunk_vector(chunk_state)
        signature = self._chunk_signature_vector(chunk_codes or [])
        chunk_projection = _normalize(projected + signature)
        if self.uses_concat_chunk_channel and int(chunk_projection.numel()) != self.chunk_output_dim:
            chunk_projection = _normalize(chunk_projection[: self.chunk_output_dim])
        return chunk_projection

    def _combine_features(
        self,
        base: torch.Tensor,
        *,
        chunk_state: torch.Tensor | None = None,
        chunk_codes: Sequence[int] | None = None,
    ) -> torch.Tensor:
        base_features = _normalize(base.float())
        if self.learned_chunking is None:
            return base_features
        chunk_projection = self._chunk_projection(chunk_state=chunk_state, chunk_codes=chunk_codes)
        if self.uses_concat_chunk_channel:
            return _normalize(torch.cat([base_features, chunk_projection], dim=0))
        if chunk_state is None or self.learned_chunk_blend <= 0.0:
            return base_features
        if float(chunk_projection.sum().item()) <= 0.0:
            return base_features
        blended = ((1.0 - self.learned_chunk_blend) * base_features) + (
            self.learned_chunk_blend * chunk_projection
        )
        return _normalize(blended)

    def _update_chunk_context(self, context: torch.Tensor, chunk_codes: Sequence[int]) -> torch.Tensor:
        if self.learned_chunking is None:
            return context
        chunk_vector = self.learned_chunking.detector_activations(chunk_codes)
        if float(chunk_vector.sum().item()) <= 0.0:
            return context
        if float(context.sum().item()) <= 0.0:
            return chunk_vector
        return _normalize((0.7 * context) + (0.3 * chunk_vector))

    def _current_chunk_state(self, context: torch.Tensor, chunk_codes: Sequence[int]) -> torch.Tensor:
        if self.learned_chunking is None:
            return torch.zeros(0, dtype=torch.float32)
        current = self.learned_chunking.detector_activations(chunk_codes)
        if float(current.sum().item()) <= 0.0:
            return context
        if float(context.sum().item()) <= 0.0:
            return current
        return _normalize(context + current)

    # ── BaseEncoder interface ───────────────────────────────────────────

    def feature_vector(self, chars: Iterable[int]) -> torch.Tensor:
        codes = list(chars)[-self.window_size :]
        base = self._base_feature_vector(codes)
        return self._combine_features(base)

    def iter_char_patterns(
        self,
        chars: Iterable[str],
        window_size: int,
        *,
        learn: bool = False,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        maxlen = max(1, int(window_size))
        window_codes: List[int] = []
        window_chars: List[str] = []
        token_codes: List[int] = []
        chunk_codes: List[int] = []
        chunk_context = (
            torch.zeros(self.learned_chunking.n_detectors, dtype=torch.float32)
            if self.learned_chunking is not None
            else torch.zeros(0, dtype=torch.float32)
        )

        for ch in chars:
            code = _ascii_code(ch)
            display = ch if ord(ch) < 128 else "?"
            window_codes.append(code)
            window_chars.append(display)
            if len(window_codes) > maxlen:
                window_codes.pop(0)
                window_chars.pop(0)

            # Token boundary handling for semantic features
            if self._is_token_boundary(ch):
                token_codes = []
            else:
                token_codes.append(code)

            # Learned chunking boundary handling
            if self.learned_chunking is not None:
                if self.learned_chunking.is_separator(code):
                    if chunk_codes:
                        if learn:
                            self.learned_chunking.learn_chunk(chunk_codes, context=chunk_context)
                        chunk_context = self._update_chunk_context(chunk_context, chunk_codes)
                        chunk_codes = []
                    if self.learned_chunking.is_hard_boundary(code):
                        chunk_context = torch.zeros_like(chunk_context)
                    chunk_state = chunk_context
                else:
                    if chunk_codes and self.learned_chunking.should_boundary(chunk_codes, code):
                        if learn:
                            self.learned_chunking.learn_chunk(chunk_codes, context=chunk_context)
                        chunk_context = self._update_chunk_context(chunk_context, chunk_codes)
                        chunk_codes = [code]
                    else:
                        chunk_codes.append(code)
                    chunk_state = self._current_chunk_state(chunk_context, chunk_codes)
            else:
                chunk_state = None

            active_codes = self._token_aware_codes(window_codes, token_codes)
            base = self._base_feature_vector(active_codes)

            yield "".join(window_chars), self._combine_features(
                base,
                chunk_state=chunk_state,
                chunk_codes=chunk_codes,
            )

        if learn and self.learned_chunking is not None and chunk_codes:
            self.learned_chunking.learn_chunk(chunk_codes, context=chunk_context)

    def segment_text(self, text: str, *, learn: bool = False) -> list[str]:
        if not text:
            return []
        if self.learned_chunking is None:
            return [w for w in text.split() if w.strip()]

        segments: list[str] = []
        chunk_codes: list[int] = []
        chunk_chars: list[str] = []
        chunk_context = torch.zeros(self.learned_chunking.n_detectors, dtype=torch.float32)
        sentence_punct = {".", "!", "?"}

        for ch in text:
            code = _ascii_code(ch)
            if self.learned_chunking.is_separator(code):
                if chunk_chars:
                    segment = "".join(chunk_chars).strip()
                    if segment:
                        segments.append(segment)
                    if learn:
                        self.learned_chunking.learn_chunk(chunk_codes, context=chunk_context)
                    chunk_context = self._update_chunk_context(chunk_context, chunk_codes)
                    chunk_codes = []
                    chunk_chars = []
                if self.learned_chunking.is_hard_boundary(code):
                    chunk_context = torch.zeros_like(chunk_context)
                if ch in sentence_punct:
                    segments.append(ch)
            elif chunk_codes and self.learned_chunking.should_boundary(chunk_codes, code):
                if chunk_chars:
                    segment = "".join(chunk_chars).strip()
                    if segment:
                        segments.append(segment)
                    if learn:
                        self.learned_chunking.learn_chunk(chunk_codes, context=chunk_context)
                    chunk_context = self._update_chunk_context(chunk_context, chunk_codes)
                chunk_codes = [code]
                chunk_chars = [ch]
            else:
                chunk_codes.append(code)
                chunk_chars.append(ch)

        if chunk_chars:
            segment = "".join(chunk_chars).strip()
            if segment:
                segments.append(segment)
            if learn:
                self.learned_chunking.learn_chunk(chunk_codes, context=chunk_context)
        return segments

    def spike_trace(
        self,
        chars: Iterable[int],
        context_confidence: float,
        *,
        tau: float | None = None,
        burst_decay: float = 0.85,
    ) -> torch.Tensor:
        """Spike trace for semantic encoding.

        Returns the feature vector modulated by confidence — semantic encoding
        has no temporal spike times to simulate.
        """
        vec = self.feature_vector(chars)
        confidence = max(0.1, min(1.0, float(context_confidence)))
        return vec * confidence

    def state_dict(self) -> dict[str, Any]:
        return {
            "bucket_embeddings": self.bucket_embeddings.clone(),
            "adapter": self.adapter.clone(),
            "glove_initialized": self._glove_initialized,
            "learned_chunking": (
                None if self.learned_chunking is None else self.learned_chunking.state_dict()
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "bucket_embeddings" in state:
            self.bucket_embeddings = state["bucket_embeddings"]
        if "adapter" in state:
            self.adapter = state["adapter"]
        if "glove_initialized" in state:
            self._glove_initialized = state["glove_initialized"]
        if self.learned_chunking is not None:
            lc = state.get("learned_chunking")
            if isinstance(lc, dict):
                self.learned_chunking.load_state_dict(lc)

    # ── GloVe initialization ───────────────────────────────────────────

    def initialize_from_glove(
        self,
        source: str = "glove-wiki-gigaword-300",
        vocab_limit: int = 50_000,
        cache_dir: str | None = None,
        ridge_alpha: float = 1.0,
    ) -> dict[str, Any]:
        """Initialize bucket embeddings from GloVe via ridge regression.

        Solves: bucket_embeddings = (A^T A + αI)^{-1} A^T Y
        where A is the n-gram-to-word design matrix and Y is PCA-projected GloVe.
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "hecsn")
        os.makedirs(cache_dir, exist_ok=True)

        cache_file = os.path.join(
            cache_dir,
            f"semantic_buckets_{self.n_buckets}_{self.embed_dim}_{vocab_limit}.pt",
        )

        if os.path.exists(cache_file):
            try:
                cached = torch.load(cache_file, weights_only=True)
                if cached["n_buckets"] == self.n_buckets and cached["embed_dim"] == self.embed_dim:
                    self.bucket_embeddings = cached["bucket_embeddings"]
                    self._glove_initialized = True
                    logger.info("Loaded cached semantic bucket embeddings from %s", cache_file)
                    return {"source": "cache", "file": cache_file}
            except Exception as e:
                logger.warning("Cache load failed: %s", e)

        try:
            import gensim.downloader as gensim_api
        except ImportError:
            logger.warning("gensim not available; using random bucket embeddings")
            return {"source": "random", "reason": "gensim not installed"}

        logger.info("Loading GloVe embeddings from '%s'...", source)
        try:
            model = gensim_api.load(source)
        except Exception as e:
            logger.warning("GloVe load failed: %s", e)
            return {"source": "random", "reason": str(e)}

        import numpy as np
        from sklearn.decomposition import PCA

        words = list(model.key_to_index.keys())[:vocab_limit]
        vectors = np.array([model[w] for w in words], dtype=np.float32)

        pca = PCA(n_components=self.embed_dim)
        projected = pca.fit_transform(vectors)
        explained_var = float(pca.explained_variance_ratio_.sum())
        logger.info(
            "PCA: %d → %d dimensions, %.1f%% variance explained",
            vectors.shape[1],
            self.embed_dim,
            explained_var * 100,
        )

        # Build AtA and AtY incrementally (avoids materializing full design matrix)
        AtA = np.zeros((self.n_buckets, self.n_buckets), dtype=np.float64)
        AtY = np.zeros((self.n_buckets, self.embed_dim), dtype=np.float64)

        for w_idx, word in enumerate(words):
            codes = [ord(c) if ord(c) < 128 else 0 for c in word]
            buckets = self._collect_ngram_buckets(codes)
            a = np.zeros(self.n_buckets, dtype=np.float64)
            for b in buckets:
                a[b] += 1.0
            nz = np.nonzero(a)[0]
            if len(nz) == 0:
                continue
            a_nz = a[nz]
            AtA[np.ix_(nz, nz)] += np.outer(a_nz, a_nz)
            AtY[nz] += a_nz[:, None] * projected[w_idx].astype(np.float64)[None, :]

        E = np.linalg.solve(
            AtA + ridge_alpha * np.eye(self.n_buckets, dtype=np.float64),
            AtY,
        )

        self.bucket_embeddings = torch.from_numpy(E.astype(np.float32))
        self._glove_initialized = True

        # Evaluate reconstruction quality on a sample
        cos_sims: list[float] = []
        sample_size = min(1000, len(words))
        for w_idx in range(sample_size):
            codes = [ord(c) if ord(c) < 128 else 0 for c in words[w_idx]]
            buckets = self._collect_ngram_buckets(codes)
            a = np.zeros(self.n_buckets, dtype=np.float64)
            for b in buckets:
                a[b] += 1.0
            recon = a @ E
            r_norm = recon / (np.linalg.norm(recon) + 1e-8)
            o_norm = projected[w_idx] / (np.linalg.norm(projected[w_idx]) + 1e-8)
            cos_sims.append(float(np.dot(r_norm, o_norm)))
        mean_cos = float(np.mean(cos_sims))

        torch.save(
            {
                "n_buckets": self.n_buckets,
                "embed_dim": self.embed_dim,
                "bucket_embeddings": self.bucket_embeddings,
                "explained_var": explained_var,
                "mean_cos_sim": mean_cos,
            },
            cache_file,
        )

        logger.info(
            "Semantic bucket init: %d words → %d buckets, mean cos sim: %.3f, cached to %s",
            len(words),
            self.n_buckets,
            mean_cos,
            cache_file,
        )

        return {
            "source": source,
            "vocab_size": len(words),
            "n_buckets": self.n_buckets,
            "embed_dim": self.embed_dim,
            "explained_variance": explained_var,
            "mean_cosine_similarity": mean_cos,
            "cache_file": cache_file,
        }
