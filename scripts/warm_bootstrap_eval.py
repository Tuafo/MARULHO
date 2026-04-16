"""Warm Companion Bootstrap — A/B/C/D Evaluation Runner.

Compares four conditions on identical training data:
  A. random  — standard random prototype init (pure HECSN)
  B. teacher — constrained bootstrap (PCA+ReLU+k-means)
  C. shuffled — shuffled bootstrap (placebo control: same geometry, wrong assignment)
  D. uniform — uniform prototypes (stress test: all prototypes identical)

Metrics collected at checkpoints {1K, 5K, 10K, 50K} tokens:
  - winner_diversity (entropy of winner histogram)
  - dead_column_ratio (columns that never won)
  - prototype_spread (mean pairwise cosine distance between prototypes)
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

# --- ensure src/ is importable ---
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from hecsn.config.model_config import HECSNConfig
from hecsn.core.columns import _normalize_positive_vector
from hecsn.data.rtf_encoder import RTFEncoder
from hecsn.training.query_runner import feed_text
from hecsn.training.trainer import HECSNModel, HECSNTrainer

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)
# Ensure stdout flushes immediately
import functools
print = functools.partial(print, flush=True)

CHECKPOINT_TOKENS = [500, 2_000, 5_000]
N_COLUMNS = 64
COLUMN_DIM = 64
MAX_TOKENS = 5_001


def _build_model(config: HECSNConfig) -> tuple[HECSNModel, HECSNTrainer]:
    model = HECSNModel(config=config)
    trainer = HECSNTrainer(model=model, config=config)
    return model, trainer


def _make_shuffled_prototypes(teacher_proto: torch.Tensor) -> torch.Tensor:
    """Shuffle rows of teacher prototypes — same geometry, random assignment."""
    perm = torch.randperm(teacher_proto.shape[0])
    return teacher_proto[perm].clone()


def _make_uniform_prototypes(n: int, dim: int) -> torch.Tensor:
    """All prototypes are the same vector — worst-case baseline."""
    v = torch.ones(1, dim)
    v = v / v.norm()
    return v.expand(n, -1).clone()


def _winner_diversity(model: HECSNModel) -> float:
    """Shannon entropy of win_rate_ema, normalized to [0, 1]."""
    rates = model.competitive.win_rate_ema.float()
    total = rates.sum()
    if total < 1e-8:
        return 0.0
    p = rates / total
    p = p[p > 0]
    entropy = -(p * p.log()).sum().item()
    max_entropy = np.log(model.competitive.n_columns)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _dead_column_ratio(model: HECSNModel) -> float:
    """Fraction of columns that haven't won recently (steps_since_win >= dead threshold)."""
    ssw = model.competitive.steps_since_win
    threshold = model.competitive.dead_column_steps
    return float((ssw >= threshold).sum().item() / max(1, ssw.numel()))


def _prototype_spread(model: HECSNModel) -> float:
    """Mean pairwise cosine distance between prototypes."""
    proto = model.competitive.prototypes.detach().float()
    sim = proto @ proto.T
    n = proto.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=proto.device)
    return float(1.0 - sim[mask].mean().item())


def _collect_metrics(model: HECSNModel, tokens: int) -> dict:
    return {
        "tokens": tokens,
        "winner_diversity": round(_winner_diversity(model), 4),
        "dead_column_ratio": round(_dead_column_ratio(model), 4),
        "prototype_spread": round(_prototype_spread(model), 4),
    }


def _generate_training_corpus(n_tokens: int) -> list[str]:
    """Simple corpus from wikitext-like sentences for evaluation."""
    sentences = [
        "the cat sat on the mat and looked at the bird through the window",
        "neural networks learn representations from data through gradient descent",
        "the weather was warm and sunny with a gentle breeze from the west",
        "quantum computing uses qubits that can exist in superposition states",
        "she walked through the garden admiring the colorful flowers in bloom",
        "the stock market experienced significant volatility during the trading session",
        "astronomers discovered a new exoplanet orbiting a distant red dwarf star",
        "the recipe calls for fresh herbs olive oil and a pinch of salt",
        "machine learning models require large datasets for effective training",
        "the ancient temple stood on a hill overlooking the peaceful valley below",
        "deep learning has revolutionized natural language processing and computer vision",
        "the children played in the park while their parents watched from the bench",
        "protein folding prediction has been advanced by artificial intelligence methods",
        "the symphony orchestra performed beethoven fifth under the stars tonight",
        "climate change affects ecosystems across every continent and ocean on earth",
        "the library contained thousands of books on history science and literature",
        "spiking neural networks process information through precise spike timing patterns",
        "the river flowed gently through the forest creating a soothing sound",
        "competitive learning algorithms cluster data by adjusting prototype vectors",
        "the mountain peak was covered with snow even during the summer months",
    ]
    words: list[str] = []
    while len(words) < n_tokens:
        for s in sentences:
            words.extend(s.split())
            if len(words) >= n_tokens:
                break
    return words[:n_tokens]


def run_condition(
    name: str,
    config: HECSNConfig,
    corpus: list[str],
    bootstrap_prototypes: torch.Tensor | None = None,
) -> list[dict]:
    """Run one experimental condition and collect metrics at checkpoints."""
    cfg = replace(config, prototype_init_mode="random")
    model, trainer = _build_model(cfg)

    if bootstrap_prototypes is not None:
        bp = bootstrap_prototypes.to(model.device).float().clamp(min=1e-6)
        bp = bp / bp.norm(dim=1, keepdim=True).clamp(min=1e-8)
        model.competitive.prototypes = bp

    encoder = RTFEncoder.from_config(config)

    results: list[dict] = []
    next_cp = 0
    t0 = time.time()
    total_tokens = 0

    def on_step(raw_window: str, metrics: dict) -> None:
        nonlocal total_tokens, next_cp
        total_tokens += 1
        if next_cp < len(CHECKPOINT_TOKENS) and total_tokens >= CHECKPOINT_TOKENS[next_cp]:
            m = _collect_metrics(model, CHECKPOINT_TOKENS[next_cp])
            m["condition"] = name
            m["elapsed_s"] = round(time.time() - t0, 1)
            results.append(m)
            print(f"  [{name}] {m['tokens']} tokens: diversity={m['winner_diversity']:.3f} "
                  f"dead={m['dead_column_ratio']:.3f} spread={m['prototype_spread']:.3f}", flush=True)
            next_cp += 1

    full_text = " ".join(corpus)
    while total_tokens < MAX_TOKENS and next_cp < len(CHECKPOINT_TOKENS):
        feed_text(trainer, encoder, full_text, on_step=on_step)

    return results


def main():
    P = lambda msg: print(msg, flush=True)

    P("=" * 60)
    P("Warm Companion Bootstrap — A/B/C/D Evaluation")
    P("=" * 60)

    base_config = HECSNConfig(
        n_columns=N_COLUMNS,
        column_latent_dim=COLUMN_DIM,
        prototype_init_mode="random",
        enable_context_layer=False,
        enable_abstraction_layer=False,
        enable_binding_layer=False,
    )

    P(f"\nGenerating training corpus ({MAX_TOKENS} tokens)...")
    corpus = _generate_training_corpus(MAX_TOKENS)

    P("\nGenerating teacher bootstrap prototypes...")
    try:
        from hecsn.training.warm_bootstrap import generate_bootstrap_prototypes
        teacher_proto = generate_bootstrap_prototypes(
            n_columns=N_COLUMNS,
            column_dim=COLUMN_DIM,
        )
    except Exception as e:
        P(f"Failed to generate bootstrap prototypes: {e}")
        P("Install gensim: pip install gensim")
        sys.exit(1)

    shuffled_proto = _make_shuffled_prototypes(teacher_proto)
    uniform_proto = _make_uniform_prototypes(N_COLUMNS, COLUMN_DIM)

    all_results: list[dict] = []

    P("\n--- Condition A: Random Init (Pure HECSN) ---")
    all_results.extend(run_condition("A_random", base_config, corpus))

    P("\n--- Condition B: Teacher Bootstrap (PCA+ReLU+k-means) ---")
    all_results.extend(run_condition("B_teacher", base_config, corpus,
                                     bootstrap_prototypes=teacher_proto))

    P("\n--- Condition C: Shuffled Bootstrap (Placebo) ---")
    all_results.extend(run_condition("C_shuffled", base_config, corpus,
                                     bootstrap_prototypes=shuffled_proto))

    P("\n--- Condition D: Uniform Prototypes (Stress Test) ---")
    all_results.extend(run_condition("D_uniform", base_config, corpus,
                                     bootstrap_prototypes=uniform_proto))

    P("\n" + "=" * 80)
    P("RESULTS SUMMARY")
    P("=" * 80)
    P(f"{'Condition':<15} {'Tokens':>8} {'Diversity':>10} {'Dead%':>8} {'Spread':>8} {'Time':>8}")
    P("-" * 80)
    for r in all_results:
        P(f"{r['condition']:<15} {r['tokens']:>8} {r['winner_diversity']:>10.4f} "
          f"{r['dead_column_ratio']:>8.4f} {r['prototype_spread']:>8.4f} {r['elapsed_s']:>7.1f}s")

    out_path = Path("reports") / "warm_bootstrap_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    P(f"\nResults saved to {out_path}")

    P("\n" + "=" * 60)
    P("VERDICT")
    P("=" * 60)

    final = {r["condition"]: r for r in all_results if r["tokens"] == CHECKPOINT_TOKENS[-1]}
    if "B_teacher" in final and "A_random" in final:
        teacher = final["B_teacher"]
        random = final["A_random"]

        div_delta = teacher["winner_diversity"] - random["winner_diversity"]
        dead_delta = teacher["dead_column_ratio"] - random["dead_column_ratio"]
        spread_delta = teacher["prototype_spread"] - random["prototype_spread"]

        P(f"Teacher vs Random at {CHECKPOINT_TOKENS[-1]} tokens:")
        P(f"  Winner diversity: {div_delta:+.4f} (higher = better)")
        P(f"  Dead columns:     {dead_delta:+.4f} (lower = better)")
        P(f"  Prototype spread: {spread_delta:+.4f} (higher = better)")

        if div_delta > 0.02 and dead_delta < -0.02:
            P("\n✅ BOOTSTRAP HELPS: Teacher init produces better diversity with fewer dead columns")
        elif abs(div_delta) < 0.02 and abs(dead_delta) < 0.02:
            P("\n⚖️ NEUTRAL: Bootstrap shows no significant difference from random")
        else:
            P("\n❌ BOOTSTRAP HURTS or INCONCLUSIVE: No clear benefit over random init")


if __name__ == "__main__":
    main()
