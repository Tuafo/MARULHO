"""Warm Companion Phase 1: Alignment Study

Quantify the geometric relationship between:
1. HECSN's actual routing key space (nonneg L2-normalized cone)
2. fastText word-level embedding space (signed Euclidean)

Goal: determine if teacher structure from fastText is usable
for prototype bootstrap, and what transformation is needed.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from hecsn.config.model_config import HECSNConfig
from hecsn.core.columns import CompetitiveColumnLayer, _normalize_positive_vector
from hecsn.data.rtf_encoder import RTFEncoder
from hecsn.training.trainer import HECSNModel

# ── Test vocabulary: common English words spanning semantic categories ──
VOCAB = [
    # Animals
    "dog", "cat", "bird", "fish", "horse", "elephant", "lion", "tiger", "bear", "wolf",
    "rabbit", "snake", "monkey", "whale", "shark", "eagle", "deer", "fox", "mouse", "frog",
    # Food
    "apple", "bread", "cheese", "rice", "meat", "milk", "water", "juice", "cake", "soup",
    "butter", "sugar", "salt", "pepper", "potato", "tomato", "banana", "grape", "lemon", "honey",
    # Nature
    "tree", "flower", "mountain", "river", "ocean", "sun", "moon", "star", "rain", "snow",
    "cloud", "wind", "fire", "ice", "earth", "lake", "stone", "sand", "grass", "leaf",
    # Body
    "hand", "foot", "head", "eye", "heart", "brain", "blood", "bone", "skin", "hair",
    "finger", "tooth", "mouth", "nose", "ear", "neck", "shoulder", "knee", "arm", "leg",
    # Objects
    "book", "table", "chair", "door", "window", "car", "phone", "computer", "clock", "key",
    "glass", "bottle", "knife", "plate", "paper", "pen", "bag", "hat", "shoe", "ring",
    # Abstract
    "love", "time", "music", "dream", "fear", "hope", "truth", "peace", "freedom", "power",
    "thought", "idea", "memory", "beauty", "anger", "joy", "pain", "death", "life", "soul",
    # Actions
    "run", "walk", "jump", "swim", "fly", "sleep", "eat", "drink", "read", "write",
    "fight", "sing", "dance", "build", "break", "grow", "fall", "burn", "open", "close",
    # Colors
    "red", "blue", "green", "black", "white", "yellow", "orange", "purple", "pink", "brown",
    "silver", "gold", "dark", "light", "bright", "gray", "deep", "pale", "warm", "cold",
    # Numbers (as words)
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "hundred", "thousand", "million", "half", "double", "first", "second", "third", "last",
    # Places
    "house", "school", "city", "forest", "garden", "beach", "island", "bridge", "tower", "road",
    "church", "market", "castle", "prison", "hospital", "harbor", "valley", "desert", "cave", "field",
    # Weather/time
    "morning", "night", "summer", "winter", "spring", "autumn", "storm", "thunder", "lightning", "fog",
    # Clothing
    "shirt", "dress", "coat", "boot", "belt", "scarf", "glove", "crown", "mask", "armor",
    # Tools
    "hammer", "sword", "wheel", "rope", "chain", "shield", "arrow", "spear", "axe", "ladder",
    # Materials
    "wood", "iron", "copper", "steel", "cotton", "leather", "rubber", "plastic", "diamond", "crystal",
]


def get_hecsn_routing_keys(vocab: list[str], cfg: HECSNConfig) -> tuple[np.ndarray, list[str]]:
    """Encode vocabulary through HECSN pipeline, return routing keys."""
    encoder = RTFEncoder(
        window_size=cfg.window_size,
        representation=cfg.input_representation,
        enable_learned_chunking=cfg.enable_learned_chunking,
        learned_chunk_detector_count=cfg.learned_chunk_detector_count,
        learned_chunk_min_len=cfg.learned_chunk_min_len,
        learned_chunk_max_len=cfg.learned_chunk_max_len,
        learned_chunk_feature_mode=cfg.learned_chunk_feature_mode,
        learned_chunk_concat_dim=cfg.learned_chunk_concat_dim,
        learned_chunk_blend=cfg.learned_chunk_blend,
        learned_chunk_similarity_floor=cfg.learned_chunk_similarity_floor,
        learned_chunk_boundary_threshold=cfg.learned_chunk_boundary_threshold,
        learned_chunk_update_lr=cfg.learned_chunk_update_lr,
        learned_chunk_association_blend=cfg.learned_chunk_association_blend,
        learned_chunk_association_lr=cfg.learned_chunk_association_lr,
        learned_chunk_association_decay=cfg.learned_chunk_association_decay,
    )
    model = HECSNModel(config=cfg)

    keys = []
    valid_words = []
    for word in vocab:
        patterns = list(encoder.iter_char_patterns(word, cfg.window_size, learn=False))
        if not patterns:
            continue
        vecs = [p for _, p in patterns]
        raw_pattern = torch.stack(vecs).mean(dim=0)
        routing_key = model.routing_key_from_pattern(raw_pattern)
        keys.append(routing_key.detach().cpu().numpy())
        valid_words.append(word)

    return np.array(keys), valid_words


def get_fasttext_embeddings(vocab: list[str]) -> tuple[np.ndarray, list[str]]:
    """Load fastText embeddings via gensim downloader (lightweight glove-wiki-gigaword-300)."""
    import gensim.downloader as api
    print("Loading word embeddings (glove-wiki-gigaword-300)...")
    print("  (First run downloads ~376MB, subsequent runs use cache)")
    model = api.load("glove-wiki-gigaword-300")

    embeds = []
    valid_words = []
    for word in vocab:
        if word in model:
            embeds.append(model[word])
            valid_words.append(word)

    return np.array(embeds), valid_words


def analyze_geometry(name: str, vectors: np.ndarray) -> dict:
    """Analyze the geometric properties of a set of vectors."""
    stats = {}
    stats["shape"] = vectors.shape
    stats["min"] = float(vectors.min())
    stats["max"] = float(vectors.max())
    stats["mean"] = float(vectors.mean())
    stats["std"] = float(vectors.std())
    stats["fraction_negative"] = float((vectors < 0).mean())
    stats["fraction_near_zero"] = float((np.abs(vectors) < 1e-4).mean())

    norms = np.linalg.norm(vectors, axis=1)
    stats["norm_mean"] = float(norms.mean())
    stats["norm_std"] = float(norms.std())

    # Pairwise cosine similarity distribution
    cos_sim = sklearn_cosine(vectors)
    np.fill_diagonal(cos_sim, np.nan)
    stats["cosine_sim_mean"] = float(np.nanmean(cos_sim))
    stats["cosine_sim_std"] = float(np.nanstd(cos_sim))
    stats["cosine_sim_min"] = float(np.nanmin(cos_sim))
    stats["cosine_sim_max"] = float(np.nanmax(cos_sim))

    print(f"\n{'='*60}")
    print(f"  Geometry Analysis: {name}")
    print(f"{'='*60}")
    print(f"  Shape:               {stats['shape']}")
    print(f"  Value range:         [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  Mean / Std:          {stats['mean']:.4f} / {stats['std']:.4f}")
    print(f"  Fraction negative:   {stats['fraction_negative']:.4f}")
    print(f"  Fraction ~zero:      {stats['fraction_near_zero']:.4f}")
    print(f"  L2 norm mean/std:    {stats['norm_mean']:.4f} / {stats['norm_std']:.4f}")
    print(f"  Cosine sim mean/std: {stats['cosine_sim_mean']:.4f} / {stats['cosine_sim_std']:.4f}")
    print(f"  Cosine sim range:    [{stats['cosine_sim_min']:.4f}, {stats['cosine_sim_max']:.4f}]")

    return stats


def analyze_neighborhood_preservation(
    hecsn_keys: np.ndarray, ft_embeds: np.ndarray,
    hecsn_words: list[str], ft_words: list[str], k: int = 5
) -> dict:
    """Check if semantic neighbors in fastText space are also neighbors in HECSN space."""
    # Find common words
    common = [w for w in hecsn_words if w in ft_words]
    if len(common) < k + 1:
        print("Too few common words for neighborhood analysis")
        return {}

    h_idx = {w: i for i, w in enumerate(hecsn_words)}
    f_idx = {w: i for i, w in enumerate(ft_words)}

    h_vecs = np.array([hecsn_keys[h_idx[w]] for w in common])
    f_vecs = np.array([ft_embeds[f_idx[w]] for w in common])

    h_sim = sklearn_cosine(h_vecs)
    f_sim = sklearn_cosine(f_vecs)

    overlaps = []
    for i in range(len(common)):
        h_neighbors = set(np.argsort(-h_sim[i])[:k+1]) - {i}
        f_neighbors = set(np.argsort(-f_sim[i])[:k+1]) - {i}
        overlap = len(h_neighbors & f_neighbors) / k
        overlaps.append(overlap)

    result = {
        "mean_neighborhood_overlap": float(np.mean(overlaps)),
        "std_neighborhood_overlap": float(np.std(overlaps)),
        "n_common_words": len(common),
    }

    print(f"\n{'='*60}")
    print(f"  Neighborhood Preservation (k={k})")
    print(f"{'='*60}")
    print(f"  Common words:           {len(common)}")
    print(f"  Mean overlap:           {result['mean_neighborhood_overlap']:.4f}")
    print(f"  Std overlap:            {result['std_neighborhood_overlap']:.4f}")
    print(f"  (1.0 = perfect preservation, ~{k/len(common):.4f} = random chance)")

    # Show some example neighborhoods
    print(f"\n  Example neighborhoods (top-{k}):")
    for word in ["dog", "apple", "love", "red", "run"]:
        if word not in common:
            continue
        i = common.index(word)
        h_top = [common[j] for j in np.argsort(-h_sim[i]) if j != i][:k]
        f_top = [common[j] for j in np.argsort(-f_sim[i]) if j != i][:k]
        print(f"    '{word}':")
        print(f"      HECSN neighbors:   {h_top}")
        print(f"      fastText neighbors: {f_top}")

    return result


def analyze_nonneg_projection(ft_embeds: np.ndarray, target_dim: int) -> dict:
    """Test different strategies for projecting fastText into HECSN's nonneg cone."""
    from sklearn.decomposition import NMF

    print(f"\n{'='*60}")
    print(f"  Projection Strategy Analysis")
    print(f"{'='*60}")

    # Strategy A: Raw PCA (as proposed in warm companion doc)
    n_comp = min(target_dim, ft_embeds.shape[0] - 1, ft_embeds.shape[1])
    pca = PCA(n_components=n_comp)
    pca_proj = pca.fit_transform(ft_embeds)
    if n_comp < target_dim:
        # Pad with zeros to reach target_dim
        pad = np.zeros((pca_proj.shape[0], target_dim - n_comp))
        pca_proj = np.hstack([pca_proj, pad])
        print(f"    (PCA limited to {n_comp} components; padded to {target_dim})")
    pca_var = pca.explained_variance_ratio_.sum()
    pca_neg_frac = float((pca_proj < 0).mean())
    print(f"\n  Strategy A: Raw PCA → {target_dim}d")
    print(f"    Explained variance: {pca_var:.4f}")
    print(f"    Fraction negative:  {pca_neg_frac:.4f}")
    print(f"    ⚠ INCOMPATIBLE — {pca_neg_frac*100:.1f}% of values are negative")

    # Strategy B: PCA + ReLU + renormalize
    pca_relu = np.maximum(pca_proj, 0)
    norms = np.linalg.norm(pca_relu, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    pca_relu_norm = pca_relu / norms
    # Check neighborhood preservation vs original
    orig_sim = sklearn_cosine(ft_embeds[:50])
    relu_sim = sklearn_cosine(pca_relu_norm[:50])
    corr_b = float(np.corrcoef(orig_sim.flatten(), relu_sim.flatten())[0, 1])
    print(f"\n  Strategy B: PCA + ReLU + L2-normalize")
    print(f"    Sim correlation with original: {corr_b:.4f}")
    print(f"    All nonnegative: {bool((pca_relu_norm >= 0).all())}")

    # Strategy C: NMF (nonneg matrix factorization)
    # Shift to nonneg first
    ft_shifted = ft_embeds - ft_embeds.min(axis=0, keepdims=True)
    n_comp_nmf = min(target_dim, ft_shifted.shape[0] - 1, ft_shifted.shape[1])
    try:
        nmf = NMF(n_components=n_comp_nmf, max_iter=300, random_state=42)
        nmf_proj = nmf.fit_transform(ft_shifted)
        if n_comp_nmf < target_dim:
            nmf_proj = np.hstack([nmf_proj, np.zeros((nmf_proj.shape[0], target_dim - n_comp_nmf))])
        norms = np.linalg.norm(nmf_proj, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        nmf_norm = nmf_proj / norms
        nmf_sim = sklearn_cosine(nmf_norm[:50])
        corr_c = float(np.corrcoef(orig_sim.flatten(), nmf_sim.flatten())[0, 1])
        print(f"\n  Strategy C: NMF → {target_dim}d")
        print(f"    Sim correlation with original: {corr_c:.4f}")
        print(f"    All nonnegative: {bool((nmf_norm >= 0).all())}")
        print(f"    Reconstruction error: {nmf.reconstruction_err_:.2f}")
    except Exception as e:
        corr_c = 0.0
        print(f"\n  Strategy C: NMF failed — {e}")

    # Strategy D: Absolute value of PCA + renormalize
    pca_abs = np.abs(pca_proj)
    norms = np.linalg.norm(pca_abs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    pca_abs_norm = pca_abs / norms
    abs_sim = sklearn_cosine(pca_abs_norm[:50])
    corr_d = float(np.corrcoef(orig_sim.flatten(), abs_sim.flatten())[0, 1])
    print(f"\n  Strategy D: |PCA| + L2-normalize")
    print(f"    Sim correlation with original: {corr_d:.4f}")
    print(f"    All nonnegative: {bool((pca_abs_norm >= 0).all())}")

    best = max([("B_pca_relu", corr_b), ("C_nmf", corr_c), ("D_abs_pca", corr_d)], key=lambda x: x[1])
    print(f"\n  ★ Best nonneg strategy: {best[0]} (corr={best[1]:.4f})")

    return {
        "pca_explained_variance": pca_var,
        "pca_negative_fraction": pca_neg_frac,
        "corr_pca_relu": corr_b,
        "corr_nmf": corr_c,
        "corr_abs_pca": corr_d,
        "best_strategy": best[0],
        "best_corr": best[1],
    }


def main():
    print("=" * 60)
    print("  HECSN Warm Companion — Phase 1: Alignment Study")
    print("=" * 60)

    cfg = HECSNConfig(n_columns=256, column_latent_dim=256)

    # Step 1: Get HECSN routing keys for vocabulary
    print("\n[1/5] Encoding vocabulary through HECSN pipeline...")
    hecsn_keys, hecsn_words = get_hecsn_routing_keys(VOCAB, cfg)
    hecsn_stats = analyze_geometry("HECSN Routing Keys", hecsn_keys)

    # Step 2: Get fastText/GloVe embeddings
    print("\n[2/5] Loading fastText/GloVe embeddings...")
    ft_embeds, ft_words = get_fasttext_embeddings(VOCAB)
    ft_stats = analyze_geometry("GloVe-300d Embeddings", ft_embeds)

    # Step 3: Compare neighborhood preservation (before any projection)
    print("\n[3/5] Comparing neighborhood structure...")
    neighborhood = analyze_neighborhood_preservation(hecsn_keys, ft_embeds, hecsn_words, ft_words)

    # Step 4: Test nonneg projection strategies
    print("\n[4/5] Testing projection strategies for nonneg compatibility...")
    projection = analyze_nonneg_projection(ft_embeds, target_dim=cfg.column_latent_dim)

    # Step 5: Test projected bootstrap quality
    print("\n[5/5] Testing bootstrap with best nonneg strategy...")
    # Apply best strategy and check neighborhood vs HECSN
    n_comp = min(cfg.column_latent_dim, ft_embeds.shape[0] - 1, ft_embeds.shape[1])
    pca = PCA(n_components=n_comp)
    pca_proj = pca.fit_transform(ft_embeds)
    if n_comp < cfg.column_latent_dim:
        pad = np.zeros((pca_proj.shape[0], cfg.column_latent_dim - n_comp))
        pca_proj = np.hstack([pca_proj, pad])

    # Apply the best nonneg transform
    best = projection["best_strategy"]
    if best == "B_pca_relu":
        projected = np.maximum(pca_proj, 0)
    elif best == "D_abs_pca":
        projected = np.abs(pca_proj)
    else:
        projected = np.maximum(pca_proj, 0)  # fallback

    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    projected = projected / np.maximum(norms, 1e-8)

    # Check k-means centroids as prototype candidates
    n_centroids = min(cfg.n_columns, len(projected))
    kmeans = MiniBatchKMeans(n_clusters=n_centroids, random_state=42)
    kmeans.fit(projected)
    centroids = kmeans.cluster_centers_
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(centroid_norms, 1e-8)

    print(f"\n  K-means centroids: {centroids.shape}")
    print(f"  All nonneg: {bool((centroids >= 0).all())}")
    print(f"  Min value: {centroids.min():.6f}")

    # Compare centroid neighborhood quality
    c_sim = sklearn_cosine(centroids)
    print(f"  Centroid cosine sim mean: {np.nanmean(c_sim[np.triu_indices_from(c_sim, k=1)]):.4f}")

    # Final verdict
    print(f"\n{'='*60}")
    print(f"  ALIGNMENT STUDY VERDICT")
    print(f"{'='*60}")

    # Key decision metrics
    baseline_neighborhood_overlap = neighborhood.get("mean_neighborhood_overlap", 0)
    nonneg_corr = projection["best_corr"]
    hecsn_all_nonneg = hecsn_stats["fraction_negative"] < 0.01
    pca_neg = projection["pca_negative_fraction"]

    print(f"\n  HECSN routing keys all nonneg: {hecsn_all_nonneg}")
    print(f"  GloVe PCA negative fraction:   {pca_neg:.2%}")
    print(f"  Best nonneg strategy:          {best} (corr={nonneg_corr:.4f})")
    print(f"  Neighborhood overlap (raw):    {baseline_neighborhood_overlap:.4f}")

    if nonneg_corr > 0.7 and baseline_neighborhood_overlap > 0.1:
        print(f"\n  ✅ GO — Teacher structure is usable with {best} projection")
        print(f"     Recommend proceeding to Phase 2: Constrained Bootstrap")
    elif nonneg_corr > 0.5:
        print(f"\n  ⚠️  CAUTIOUS GO — Moderate alignment, proceed with careful A/B testing")
        print(f"     Teacher structure partially preserved after nonneg projection")
    else:
        print(f"\n  ❌ NO-GO — Teacher structure destroyed by nonneg constraint")
        print(f"     Bootstrap unlikely to help; recommend staying with random init")

    return {
        "hecsn_stats": hecsn_stats,
        "ft_stats": ft_stats,
        "neighborhood": neighborhood,
        "projection": projection,
    }


if __name__ == "__main__":
    results = main()
