"""Warm Companion bootstrap: distill teacher embeddings into HECSN's positive manifold.

Implements the constrained bootstrap pipeline:
  1. Load word-level embeddings (GloVe via gensim)
  2. PCA to column_dim
  3. ReLU + L2-normalize (nonneg cone projection)
  4. k-means centroids → prototype candidates
  5. Optionally learn a nonneg W_project alignment

The key insight from the alignment study: HECSN routing keys live on a
nonneg L2-normalized cone.  Raw PCA embeddings are ~47% negative.
PCA + ReLU preserves 89% of neighborhood structure while satisfying
the nonneg constraint.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch

logger = logging.getLogger(__name__)

BootstrapMode = Literal["random", "teacher"]

# Minimum vocab size for meaningful bootstrap
_MIN_VOCAB = 50


def load_teacher_embeddings(
    vocab: list[str] | None = None,
    source: str = "glove-wiki-gigaword-300",
    limit: int = 50_000,
) -> tuple[np.ndarray, list[str]]:
    """Load word-level embeddings from gensim-data.

    Returns (embeddings [n_words, embed_dim], words).
    """
    import gensim.downloader as api

    logger.info("Loading teacher embeddings: %s (limit=%d)", source, limit)
    model = api.load(source)

    if vocab is not None:
        words = [w for w in vocab if w in model]
        embeds = np.array([model[w] for w in words])
    else:
        words = list(model.key_to_index.keys())[:limit]
        embeds = np.array([model[w] for w in words])

    logger.info("Loaded %d embeddings of dim %d", len(words), embeds.shape[1])
    return embeds, words


def project_to_nonneg_cone(
    embeddings: np.ndarray,
    target_dim: int,
) -> np.ndarray:
    """Project embeddings into HECSN's nonneg L2-normalized cone.

    Strategy: PCA → ReLU → L2-normalize
    (Best strategy from alignment study: 0.889 correlation with original structure)
    """
    from sklearn.decomposition import PCA

    n_comp = min(target_dim, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    projected = pca.fit_transform(embeddings)

    if n_comp < target_dim:
        pad = np.zeros((projected.shape[0], target_dim - n_comp))
        projected = np.hstack([projected, pad])

    # ReLU: clamp negatives to zero
    projected = np.maximum(projected, 0)

    # L2 normalize (matching HECSN's _normalize_positive_vector)
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    projected = projected / np.maximum(norms, 1e-8)

    explained = pca.explained_variance_ratio_.sum()
    logger.info(
        "PCA+ReLU projection: %dd → %dd (%.1f%% variance, %d components)",
        embeddings.shape[1], target_dim, explained * 100, n_comp,
    )
    return projected


def select_prototype_centroids(
    projected_embeddings: np.ndarray,
    n_centroids: int,
) -> np.ndarray:
    """Select n_centroids via k-means from projected embeddings.

    Returns L2-normalized nonneg centroid vectors [n_centroids, dim].
    """
    from sklearn.cluster import MiniBatchKMeans

    n_centroids = min(n_centroids, projected_embeddings.shape[0])
    kmeans = MiniBatchKMeans(n_clusters=n_centroids, random_state=42, n_init=3)
    kmeans.fit(projected_embeddings)
    centroids = kmeans.cluster_centers_

    # Ensure nonneg + normalized
    centroids = np.maximum(centroids, 0)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norms, 1e-8)

    logger.info("Selected %d prototype centroids via k-means", n_centroids)
    return centroids


def compute_bootstrap_alignment(
    prototypes: torch.Tensor,
    routing_keys: list[torch.Tensor],
) -> float:
    """Compute bootstrap_alignment_score (§7.3 of Warm Companion).

    Mean cosine similarity between each routing key and its nearest prototype.
    """
    if not routing_keys or prototypes.numel() == 0:
        return 0.0

    proto = prototypes.detach().float()
    scores = []
    for rk in routing_keys:
        rk_f = rk.detach().float().to(proto.device)
        sim = torch.mv(proto, rk_f)
        scores.append(float(sim.max().item()))

    return float(np.mean(scores))


def generate_bootstrap_prototypes(
    n_columns: int,
    column_dim: int,
    source: str = "glove-wiki-gigaword-300",
    vocab_limit: int = 50_000,
) -> torch.Tensor:
    """Full bootstrap pipeline: load → project → select centroids.

    Returns tensor [n_columns, column_dim] ready for injection into
    CompetitiveColumnLayer.prototypes.
    """
    embeddings, words = load_teacher_embeddings(source=source, limit=vocab_limit)

    if len(words) < _MIN_VOCAB:
        logger.warning(
            "Only %d words loaded (need >= %d). Falling back to random init.",
            len(words), _MIN_VOCAB,
        )
        proto = torch.rand(n_columns, column_dim)
        proto = proto.clamp(min=1e-6)
        return proto / proto.norm(dim=1, keepdim=True).clamp(min=1e-8)

    projected = project_to_nonneg_cone(embeddings, target_dim=column_dim)
    centroids = select_prototype_centroids(projected, n_centroids=n_columns)

    proto = torch.from_numpy(centroids).float()
    proto = proto.clamp(min=1e-6)
    proto = proto / proto.norm(dim=1, keepdim=True).clamp(min=1e-8)

    logger.info(
        "Bootstrap prototypes: shape=%s, min=%.4f, max=%.4f",
        list(proto.shape), proto.min().item(), proto.max().item(),
    )
    return proto


def save_bootstrap(prototypes: torch.Tensor, path: str | Path) -> None:
    """Save bootstrap prototypes for later loading."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prototypes, path)
    logger.info("Saved bootstrap prototypes to %s", path)


def load_bootstrap(path: str | Path) -> torch.Tensor:
    """Load previously saved bootstrap prototypes."""
    return torch.load(path, weights_only=True)
