"""Semantic vs RTF Encoding — A/B Evaluation.

Compares two encoding modes on identical training data:
  A. RTF (order_weighted_ascii) — current default character-based encoding
  B. Semantic — n-gram composition with GloVe-initialized bucket embeddings

Metrics collected at checkpoints:
  - grounding_probe_accuracy (50-triple evaluation)
  - winner_diversity (entropy of winner histogram)
  - dead_column_ratio (columns that never won)
  - prototype_spread (mean pairwise cosine distance)
  - throughput (tokens/sec)
  - semantic_coherence (cosine similarity of semantically related words)
"""

from __future__ import annotations

import json
import logging
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
from hecsn.evaluation.grounding_probe import evaluate_grounding_probe
from hecsn.training.trainer import HECSNModel, HECSNTrainer

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)
import functools
print = functools.partial(print, flush=True)

# ── Parameters ──────────────────────────────────────────────────────────
N_COLUMNS = 256
COLUMN_DIM = 128
MAX_TOKENS = 10_001
CHECKPOINT_TOKENS = [1_000, 3_000, 5_000, 10_000]

CORPUS = """
The ocean waves crashed against the rocky shore as the sun set behind the mountains.
Birds flew overhead, their calls echoing across the water. A fisherman cast his line
into the deep blue sea, hoping for a good catch. The wind picked up, carrying the
scent of salt and seaweed. In the distance, a lighthouse beam swept across the
darkening sky. The old harbor was quiet now, boats rocking gently at their moorings.

A professor of mathematics stood before the blackboard, chalk in hand. She explained
the theorem carefully, drawing diagrams to illustrate each step. The students leaned
forward, pencils scratching notes. Abstract algebra was not easy, but the elegant
proofs had a beauty of their own. Numbers and symbols danced across the board in
patterns that only the trained eye could appreciate.

The fire crackled in the stone hearth, casting flickering shadows on the cabin walls.
Outside, snow fell silently, blanketing the forest in white. A dog lay by the fire,
ears twitching at distant sounds. The kettle whistled on the stove, steam curling
toward the ceiling. Someone poured hot tea into ceramic mugs, the liquid golden
and fragrant.

In the laboratory, the scientist examined the specimen under the microscope. Cells
divided and multiplied in real time, their structures shifting and reforming.
The data showed promising results: the new compound inhibited growth of the target
pathogen without damaging healthy tissue. She recorded her observations carefully,
knowing that reproducibility was the cornerstone of good science.

The marketplace bustled with activity. Vendors called out prices for fresh fruit,
vegetables, and fish. Children ran between the stalls, laughing and pointing at
colorful displays. The aroma of baking bread mingled with spices from the far east.
An old woman counted coins at her stall, her weathered hands moving with practiced
efficiency. A musician played guitar near the fountain, his melody floating above
the crowd noise.

Thunder rumbled in the distance as dark clouds gathered over the valley. Lightning
flashed, illuminating the landscape in brief, stark white. The rain began, first
as scattered drops, then as a steady downpour. Rivers swelled and streams overflowed
their banks. The earth drank deeply, grateful after weeks of drought. Farmers watched
from their porches, hopeful that the harvest would be saved.

The philosopher contemplated the nature of consciousness. What is awareness? How
does subjective experience arise from physical matter? These questions had puzzled
thinkers for millennia, and no definitive answer had emerged. Perhaps the mystery
itself was the point — a reminder that some truths lie beyond the reach of logic
and language, accessible only through direct experience.

Metal clanged against metal in the blacksmith's workshop. Sparks flew as the hammer
struck the glowing iron, shaping it into a blade. The forge burned white-hot, its
heat radiating outward in waves. The smith wiped sweat from his brow and examined
his work, turning the metal this way and that, checking for flaws. Craftsmanship
required patience, skill, and an intimate knowledge of the material.
""".strip()

# Word pairs for semantic coherence test
SEMANTIC_PAIRS = [
    ("ocean", "water"),
    ("fire", "heat"),
    ("snow", "cold"),
    ("bird", "flight"),
    ("dog", "bark"),
    ("sun", "light"),
    ("rain", "wet"),
    ("mountain", "rock"),
]
RANDOM_PAIRS = [
    ("ocean", "algebra"),
    ("fire", "philosophy"),
    ("snow", "market"),
    ("bird", "theorem"),
    ("dog", "microscope"),
    ("sun", "compound"),
    ("rain", "chalk"),
    ("mountain", "guitar"),
]


def _build_config(mode: str) -> HECSNConfig:
    """Build config for either 'rtf' or 'semantic' mode."""
    base = HECSNConfig(
        n_columns=N_COLUMNS,
        column_latent_dim=COLUMN_DIM,
        binding_mode="hypercube",
        device="cpu",
    )
    if mode == "semantic":
        base = replace(
            base,
            input_representation="semantic",
            semantic_n_buckets=10_000,
            semantic_embed_dim=64,
            semantic_top_k_sparse=8,
        )
    return base


def _build_model(config: HECSNConfig):
    model = HECSNModel(config)
    trainer = HECSNTrainer(model, config)
    return model, trainer


def _get_word_vector(trainer, encoder, config, word: str) -> torch.Tensor:
    """Get the routing key vector for a word."""
    patterns = list(encoder.iter_char_patterns(word, config.window_size, learn=False))
    if not patterns:
        return torch.zeros(config.column_latent_dim)
    vecs = [p for _, p in patterns]
    raw_pattern = torch.stack(vecs).mean(dim=0)
    return trainer.model.routing_key_from_pattern(raw_pattern).detach().cpu()


def _semantic_coherence(trainer, encoder, config, pairs, label=""):
    """Compute mean cosine similarity for word pairs."""
    sims = []
    for w1, w2 in pairs:
        v1 = _get_word_vector(trainer, encoder, config, w1)
        v2 = _get_word_vector(trainer, encoder, config, w2)
        n1, n2 = v1.norm(), v2.norm()
        if n1 > 1e-8 and n2 > 1e-8:
            sim = float(torch.dot(v1, v2) / (n1 * n2))
        else:
            sim = 0.0
        sims.append(sim)
    return float(np.mean(sims))


def _compute_metrics(trainer, encoder, config, winner_hist, elapsed, tokens):
    """Compute all evaluation metrics."""
    # Winner diversity (entropy)
    counts = np.array(list(winner_hist.values()), dtype=np.float64)
    if counts.sum() > 0:
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs)))
        max_entropy = np.log2(max(1, len(counts)))
        diversity = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        diversity = 0.0

    # Dead columns
    n_active = len([c for c in winner_hist.values() if c > 0])
    dead_ratio = 1.0 - n_active / config.n_columns

    # Prototype spread
    protos = trainer.model.competitive.prototypes.detach().cpu()
    n = protos.shape[0]
    if n > 1:
        norms = protos.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normed = protos / norms
        cos_sim = normed @ normed.T
        mask = ~torch.eye(n, dtype=torch.bool)
        spread = float(1.0 - cos_sim[mask].mean())
    else:
        spread = 0.0

    # Semantic coherence
    related_sim = _semantic_coherence(trainer, encoder, config, SEMANTIC_PAIRS, "related")
    random_sim = _semantic_coherence(trainer, encoder, config, RANDOM_PAIRS, "random")

    # Throughput
    throughput = tokens / elapsed if elapsed > 0 else 0.0

    # Grounding probe
    def vector_fn(text: str) -> torch.Tensor:
        return _get_word_vector(trainer, encoder, config, text)

    probe = evaluate_grounding_probe(vector_fn)

    return {
        "tokens": tokens,
        "winner_diversity": round(diversity, 4),
        "dead_column_ratio": round(dead_ratio, 4),
        "prototype_spread": round(spread, 4),
        "related_sim": round(related_sim, 4),
        "random_sim": round(random_sim, 4),
        "coherence_gap": round(related_sim - random_sim, 4),
        "throughput_tok_s": round(throughput, 1),
        "grounding_accuracy": round(probe.total_accuracy, 4),
        "concrete_accuracy": round(probe.concrete_accuracy, 4),
        "abstract_accuracy": round(probe.abstract_accuracy, 4),
        "concreteness_gap": round(probe.concreteness_gap, 4),
    }


def run_condition(mode: str) -> list[dict]:
    """Run a single condition (rtf or semantic) and return metrics at checkpoints."""
    print(f"\n{'='*60}")
    print(f"  Condition: {mode.upper()}")
    print(f"{'='*60}")

    config = _build_config(mode)
    model, trainer = _build_model(config)
    encoder = build_encoder(config)

    print(f"  input_representation: {config.input_representation}")
    print(f"  input_dim: {config.input_dim}")
    print(f"  n_columns: {config.n_columns}")
    print(f"  column_latent_dim: {config.column_latent_dim}")

    winner_hist: dict[int, int] = {}
    results = []
    token_count = 0
    next_cp_idx = 0
    start_time = time.perf_counter()

    # Repeat corpus to reach MAX_TOKENS
    full_text = (CORPUS + "\n") * ((MAX_TOKENS // len(CORPUS)) + 2)

    for raw_window, pattern in encoder.iter_char_patterns(full_text, config.window_size, learn=True):
        metrics = trainer.train_step(pattern, raw_window=raw_window)
        w = int(metrics["winner"])
        winner_hist[w] = winner_hist.get(w, 0) + 1
        token_count += 1

        if next_cp_idx < len(CHECKPOINT_TOKENS) and token_count >= CHECKPOINT_TOKENS[next_cp_idx]:
            elapsed = time.perf_counter() - start_time
            print(f"\n  [{mode}] Checkpoint @ {token_count} tokens ({elapsed:.1f}s)...")
            m = _compute_metrics(trainer, encoder, config, winner_hist, elapsed, token_count)
            m["mode"] = mode
            results.append(m)
            print(f"    diversity={m['winner_diversity']:.3f}  dead={m['dead_column_ratio']:.3f}  "
                  f"spread={m['prototype_spread']:.3f}")
            print(f"    coherence_gap={m['coherence_gap']:.4f}  "
                  f"(related={m['related_sim']:.3f} random={m['random_sim']:.3f})")
            print(f"    grounding={m['grounding_accuracy']:.3f}  "
                  f"concrete={m['concrete_accuracy']:.3f}  abstract={m['abstract_accuracy']:.3f}")
            print(f"    throughput={m['throughput_tok_s']:.1f} tok/s")
            next_cp_idx += 1

        if token_count >= MAX_TOKENS:
            break

    return results


def main():
    print("=" * 60)
    print("  HECSN Semantic vs RTF — A/B Evaluation")
    print(f"  Columns={N_COLUMNS}  dim={COLUMN_DIM}  max_tokens={MAX_TOKENS}")
    print("=" * 60)

    rtf_results = run_condition("rtf")
    semantic_results = run_condition("semantic")

    # Summary comparison
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)

    headers = ["Metric", "RTF @10K", "Semantic @10K", "Δ"]
    print(f"\n  {'─'*56}")
    print(f"  {headers[0]:<24} {headers[1]:>10} {headers[2]:>12} {headers[3]:>8}")
    print(f"  {'─'*56}")

    if rtf_results and semantic_results:
        r = rtf_results[-1]
        s = semantic_results[-1]
        for key in ["winner_diversity", "dead_column_ratio", "prototype_spread",
                     "coherence_gap", "grounding_accuracy", "concrete_accuracy",
                     "abstract_accuracy", "concreteness_gap", "throughput_tok_s"]:
            rv, sv = r[key], s[key]
            delta = sv - rv
            sign = "+" if delta > 0 else ""
            print(f"  {key:<24} {rv:>10.4f} {sv:>12.4f} {sign}{delta:>7.4f}")

    print(f"  {'─'*56}")

    # Save results
    out_dir = Path(__file__).parent.parent / "reports"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "semantic_ab_eval.json"
    all_results = {"rtf": rtf_results, "semantic": semantic_results}
    out_file.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Results saved to {out_file}")


if __name__ == "__main__":
    main()
