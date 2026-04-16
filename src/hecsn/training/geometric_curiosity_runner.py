from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from hecsn.config.model_config import HECSNConfig
from hecsn.reporting.io import write_json_file
from hecsn.semantics import GeometricCuriosityController
from hecsn.training.trainer import HECSNModel, HECSNTrainer


TOKENS = (
    "river",
    "stream",
    "water",
    "current",
    "loan",
    "credit",
    "bank",
)


def run_geometric_curiosity_benchmark(*, output_dir: Path, seed: int = 23) -> dict[str, Any]:
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
    model = HECSNModel(cfg)
    trainer = HECSNTrainer(model, cfg)
    controller = GeometricCuriosityController(model.abstraction_layer)

    recon_errors: list[float] = []
    for _ in range(3):
        for token in TOKENS:
            pattern = trainer.encoder.feature_vector([ord(ch) for ch in token])
            metrics = trainer.train_step(pattern, raw_window=token)
            recon_errors.append(float(metrics["recon_error"]))
            if model.abstraction_layer is not None:
                controller.update_lexicon(model.abstraction_layer.last_activations, [token])

    focus_plan = controller.focus_plan()
    summary = {
        "benchmark": "geometric_curiosity_smoke",
        "seed": int(seed),
        "runtime_scope": model.runtime_scope_report(),
        "controller": controller.summary(),
        "focus_plan": focus_plan,
        "metrics": {
            "mean_recon_error": float(sum(recon_errors) / max(1, len(recon_errors))),
        },
    }
    summary["geometric_curiosity_gate"] = {
        "pass": bool(
            focus_plan is not None
            and bool(focus_plan.get("retrieval_queries"))
            and bool(focus_plan.get("geometric_gaps"))
            and int(controller.summary()["lexicon_concept_count"]) > 0
        ),
        "thresholds": {
            "retrieval_queries": True,
            "geometric_gaps": True,
            "lexicon_concept_count_gt": 0,
        },
    }
    write_json_file(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the geometric curiosity smoke.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "phase7_geometric_curiosity_smoke",
        help="Directory where the benchmark summary will be written.",
    )
    parser.add_argument("--seed", type=int, default=23, help="Deterministic seed for the benchmark.")
    args = parser.parse_args()

    summary = run_geometric_curiosity_benchmark(output_dir=args.output_dir, seed=args.seed)
    print(f"[geometric_curiosity_smoke] pass={summary['geometric_curiosity_gate']['pass']}")


if __name__ == "__main__":
    main()
