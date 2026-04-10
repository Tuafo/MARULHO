from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from hecsn.config.model_config import HECSNConfig
from hecsn.core.columns import CompetitiveColumnLayer
from hecsn.reporting.io import write_json_file
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer


TOKENS = (
    "river",
    "stream",
    "water",
    "river",
    "stream",
    "water",
    "loan",
    "money",
)


def _probe_equal_score_routing(dominant_gain: float, weak_gain: float) -> dict[str, Any]:
    layer = CompetitiveColumnLayer(
        n_columns=2,
        column_dim=2,
        input_dim=2,
        device=torch.device("cpu"),
        input_weight_blend=0.0,
        dead_column_steps=10**9,
    )
    prototype = F.normalize(torch.tensor([1.0, 1.0]), dim=0)
    layer.prototypes[0] = prototype
    layer.prototypes[1] = prototype
    layer.thresholds.zero_()
    layer.last_input_pattern = torch.tensor([0.5, 0.5])

    routing_key = prototype.clone()
    neutral_winners, _, _ = layer.compete(
        routing_key,
        torch.tensor([0, 1]),
        fallback_allowed=True,
        context_gain=torch.ones(2),
    )
    biased_winners, _, _ = layer.compete(
        routing_key,
        torch.tensor([0, 1]),
        fallback_allowed=True,
        context_gain=torch.tensor([weak_gain, dominant_gain]),
    )
    return {
        "neutral_winner": int(neutral_winners[0].item()),
        "biased_winner": int(biased_winners[0].item()),
        "winner_changed": int(neutral_winners[0].item()) != int(biased_winners[0].item()),
    }


def run_abstraction_routing_benchmark(*, output_dir: Path, seed: int = 17) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(seed))

    cfg = HECSNConfig(
        n_columns=8,
        column_latent_dim=16,
        bootstrap_tokens=0,
        memory_capacity=64,
        eta_competitive=0.05,
        eta_decay=0.0,
        input_weight_blend=0.0,
        enable_abstraction_layer=True,
    )
    model = HECSNModelLite(cfg)
    trainer = HECSNTrainer(model, cfg)

    winner_counts: dict[int, int] = {}
    recon_errors: list[float] = []
    for _ in range(3):
        for token in TOKENS:
            pattern = trainer.encoder.feature_vector([ord(ch) for ch in token])
            metrics = trainer.train_step(pattern, raw_window=token)
            winner = int(metrics["winner"])
            winner_counts[winner] = winner_counts.get(winner, 0) + 1
            recon_errors.append(float(metrics["recon_error"]))

    assert model.abstraction_layer is not None
    scope = model.runtime_scope_report()
    gain = model.abstraction_layer.routing_gain().detach().cpu()
    dominant_winner = max(winner_counts.items(), key=lambda item: item[1])[0]
    weak_winner = int(torch.argmin(gain).item())
    probe = _probe_equal_score_routing(
        dominant_gain=float(gain[dominant_winner].item()),
        weak_gain=float(gain[weak_winner].item()),
    )

    summary = {
        "benchmark": "abstraction_routing_smoke",
        "seed": int(seed),
        "runtime_scope": {
            "mode": "first_class_abstraction_feedback",
            "supports_first_class_abstraction": bool(scope["supports_first_class_abstraction"]),
            "abstraction_architecture": scope["abstraction_architecture"],
        },
        "winner_histogram": {str(key): int(value) for key, value in sorted(winner_counts.items())},
        "abstraction": {
            "summary": model.abstraction_layer.summary(),
            "gain_mean": float(gain.mean().item()),
            "gain_std": float(gain.std(unbiased=False).item()),
            "gain_max": float(gain.max().item()),
            "gain_min": float(gain.min().item()),
            "gap_score_max": max(
                (float(item["gap_score"]) for item in model.abstraction_layer.curiosity_gaps(top_n=4)),
                default=0.0,
            ),
        },
        "routing_bias": {
            "dominant_winner": int(dominant_winner),
            "dominant_winner_gain": float(gain[dominant_winner].item()),
            "weak_winner": int(weak_winner),
            "weak_winner_gain": float(gain[weak_winner].item()),
            "gain_margin": float(gain[dominant_winner].item() - gain[weak_winner].item()),
            "equal_score_probe": probe,
        },
        "metrics": {
            "mean_recon_error": float(sum(recon_errors) / max(1, len(recon_errors))),
        },
    }
    summary["abstraction_routing_gate"] = {
        "pass": bool(
            scope["supports_first_class_abstraction"]
            and math.isfinite(summary["metrics"]["mean_recon_error"])
            and summary["abstraction"]["summary"]["mean_stability"] > 0.0
            and summary["routing_bias"]["gain_margin"] > 0.0
            and probe["winner_changed"]
            and probe["biased_winner"] == 1
        ),
        "thresholds": {
            "mean_stability_gt": 0.0,
            "gain_margin_gt": 0.0,
            "equal_score_biased_winner": 1,
        },
    }
    write_json_file(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the abstraction routing smoke.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "phase7_abstraction_routing_smoke",
        help="Directory where the benchmark summary will be written.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Deterministic seed for the benchmark.")
    args = parser.parse_args()

    summary = run_abstraction_routing_benchmark(output_dir=args.output_dir, seed=args.seed)
    print(f"[abstraction_routing_smoke] pass={summary['abstraction_routing_gate']['pass']}")


if __name__ == "__main__":
    main()
