"""Quick ablation: top-k sparsification and SimHash routing transform.

Tests whether the semantic encoder's dead-column problem can be fixed by:
  1. top-k sparsification (k=16, k=32)
  2. SimHash random projection (semantic → random hyperplanes → binary → nonneg routing)
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from hecsn.config.model_config import HECSNConfig
from hecsn.data.encoder_factory import build_encoder
from hecsn.data.semantic_encoder import SemanticEncoder
from hecsn.training.trainer import HECSNModel, HECSNTrainer

import functools
print = functools.partial(print, flush=True)

N_COLUMNS = 256
COLUMN_DIM = 128
MAX_TOKENS = 5_001

CORPUS = """
The ocean waves crashed against the rocky shore as the sun set behind the mountains.
Birds flew overhead calling across the water. A fisherman cast his line into the deep
blue sea hoping for a good catch. The wind picked up carrying the scent of salt and
seaweed. In the distance a lighthouse beam swept across the darkening sky.

A professor of mathematics stood before the blackboard chalk in hand. She explained
the theorem carefully drawing diagrams. The students leaned forward pencils scratching
notes. Abstract algebra was not easy but the proofs had a beauty of their own.

The fire crackled in the stone hearth casting flickering shadows on the cabin walls.
Outside snow fell silently blanketing the forest in white. A dog lay by the fire ears
twitching at distant sounds. The kettle whistled on the stove steam curling upward.

Thunder rumbled in the distance as dark clouds gathered over the valley. Lightning
flashed illuminating the landscape in brief stark white. The rain began first as
scattered drops then as a steady downpour. Rivers swelled and streams overflowed.
""".strip()


def measure_input_geometry(encoder, config, n_words=50):
    """Measure pairwise cosine distribution of encoder outputs."""
    words = list(set(CORPUS.lower().split()))[:n_words]
    vecs = []
    for w in words:
        patterns = list(encoder.iter_char_patterns(w + " ", config.window_size, learn=False))
        if patterns:
            # Use final vector (full word)
            vecs.append(patterns[-1][1])
    if len(vecs) < 2:
        return {"mean_pairwise_cos": 0.0, "std_pairwise_cos": 0.0, "min_cos": 0.0, "max_cos": 0.0}
    
    mat = torch.stack(vecs)
    norms = mat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = mat / norms
    cos_sim = normed @ normed.T
    mask = ~torch.eye(len(vecs), dtype=torch.bool)
    pairwise = cos_sim[mask]
    
    return {
        "mean_pairwise_cos": round(float(pairwise.mean()), 4),
        "std_pairwise_cos": round(float(pairwise.std()), 4),
        "min_cos": round(float(pairwise.min()), 4),
        "max_cos": round(float(pairwise.max()), 4),
        "n_words": len(vecs),
    }


def run_condition(label, config):
    """Train and measure dead columns + diversity."""
    model = HECSNModel(config)
    trainer = HECSNTrainer(model, config)
    encoder = build_encoder(config)
    
    # Measure input geometry BEFORE training
    geom = measure_input_geometry(encoder, config)
    
    winner_hist = {}
    token_count = 0
    full_text = (CORPUS + "\n") * 10
    t0 = time.perf_counter()
    
    for raw_window, pattern in encoder.iter_char_patterns(full_text, config.window_size, learn=True):
        metrics = trainer.train_step(pattern, raw_window=raw_window)
        w = int(metrics["winner"])
        winner_hist[w] = winner_hist.get(w, 0) + 1
        token_count += 1
        if token_count >= MAX_TOKENS:
            break
    
    elapsed = time.perf_counter() - t0
    
    # Compute diversity
    counts = np.array(list(winner_hist.values()), dtype=np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))
    max_ent = np.log2(max(1, len(counts)))
    diversity = entropy / max_ent if max_ent > 0 else 0.0
    
    n_active = len([c for c in winner_hist.values() if c > 0])
    dead_ratio = 1.0 - n_active / config.n_columns
    
    print(f"\n  {label}")
    print(f"    input_dim={config.input_dim}  tokens={token_count}")
    print(f"    dead_columns={dead_ratio:.3f}  diversity={diversity:.3f}  throughput={token_count/elapsed:.1f} tok/s")
    print(f"    input geometry: mean_cos={geom['mean_pairwise_cos']:.4f}  "
          f"std={geom['std_pairwise_cos']:.4f}  "
          f"range=[{geom['min_cos']:.3f}, {geom['max_cos']:.3f}]")
    
    return {"label": label, "dead_ratio": dead_ratio, "diversity": diversity, 
            "throughput": token_count/elapsed, **geom}


def main():
    print("=" * 60)
    print("  Semantic Encoder Ablation Study")
    print("=" * 60)
    
    results = []
    
    # 1. RTF baseline
    cfg_rtf = HECSNConfig(n_columns=N_COLUMNS, column_latent_dim=COLUMN_DIM, 
                           binding_mode="hypercube", device="cpu")
    results.append(run_condition("RTF (baseline)", cfg_rtf))
    
    # 2. Semantic top_k=0 (current)
    cfg_sem0 = replace(cfg_rtf, input_representation="semantic", 
                        semantic_n_buckets=10_000, semantic_embed_dim=64, semantic_top_k_sparse=0)
    results.append(run_condition("Semantic top_k=0", cfg_sem0))
    
    # 3. Semantic top_k=32
    cfg_sem32 = replace(cfg_sem0, semantic_top_k_sparse=32)
    results.append(run_condition("Semantic top_k=32", cfg_sem32))
    
    # 4. Semantic top_k=16
    cfg_sem16 = replace(cfg_sem0, semantic_top_k_sparse=16)
    results.append(run_condition("Semantic top_k=16", cfg_sem16))
    
    # 5. Semantic top_k=8
    cfg_sem8 = replace(cfg_sem0, semantic_top_k_sparse=8)
    results.append(run_condition("Semantic top_k=8", cfg_sem8))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Label':<25} {'Dead%':>6} {'Div':>6} {'MeanCos':>8} {'Tok/s':>6}")
    print(f"  {'─'*53}")
    for r in results:
        print(f"  {r['label']:<25} {r['dead_ratio']:>5.1%} {r['diversity']:>6.3f} "
              f"{r['mean_pairwise_cos']:>8.4f} {r['throughput']:>6.1f}")


if __name__ == "__main__":
    main()
