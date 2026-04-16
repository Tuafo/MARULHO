from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from hecsn.config.model_config import HECSNConfig
from hecsn.reporting.io import write_json_file
from hecsn.training.trainer import HECSNModel, HECSNTrainer


TOKENS = (
    "bank",
    "bond",
    "river",
    "money",
    "cat",
    "dog",
    "bird",
    "fish",
)


def _run_backend(backend: str, *, seed: int) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    cfg = HECSNConfig(
        n_columns=8,
        column_latent_dim=16,
        bootstrap_tokens=0,
        memory_capacity=64,
        eta_competitive=0.05,
        eta_decay=0.0,
        input_weight_blend=0.0,
        plasticity_mode="local_stdp",
        plasticity_spike_backend=backend,
        enable_learned_chunking=False,
    )
    model = HECSNModel(cfg)
    trainer = HECSNTrainer(model, cfg)

    recon_errors: list[float] = []
    active_columns: list[int] = []
    post_spike_fractions: list[float] = []
    mean_membrane_voltages: list[float] = []

    for _ in range(3):
        for token in TOKENS:
            pattern = trainer.encoder.feature_vector([ord(ch) for ch in token])
            metrics = trainer.train_step(pattern, raw_window=token)
            recon_errors.append(float(metrics["recon_error"]))
            active_columns.append(int(metrics["active_columns"]))
            post_spike_fractions.append(float(metrics["local_post_spike_fraction"]))
            mean_membrane_voltages.append(float(metrics["local_mean_membrane_voltage"]))

    scope = model.runtime_scope_report()
    finite_parameters = bool(
        torch.isfinite(model.competitive.prototypes).all().item()
        and torch.isfinite(model.competitive.input_weights).all().item()
        and torch.isfinite(model.competitive.W_project).all().item()
    )
    mean_recon_error = float(sum(recon_errors) / max(1, len(recon_errors)))
    mean_active_columns = float(sum(active_columns) / max(1, len(active_columns)))
    mean_post_spike_fraction = float(sum(post_spike_fractions) / max(1, len(post_spike_fractions)))
    mean_membrane_voltage = float(sum(mean_membrane_voltages) / max(1, len(mean_membrane_voltages)))

    gate_pass = bool(
        finite_parameters
        and math.isfinite(mean_recon_error)
        and mean_active_columns > 0.0
        and (
            backend != "adex"
            or (bool(scope["uses_adex_post_spikes"]) and mean_post_spike_fraction > 0.0)
        )
    )
    return {
        "backend": backend,
        "uses_adex_post_spikes": bool(scope["uses_adex_post_spikes"]),
        "finite_parameters": finite_parameters,
        "mean_recon_error": mean_recon_error,
        "mean_active_columns": mean_active_columns,
        "mean_post_spike_fraction": mean_post_spike_fraction,
        "mean_membrane_voltage": mean_membrane_voltage,
        "gate_pass": gate_pass,
    }


def run_adex_backend_benchmark(*, output_dir: Path, seed: int = 7) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    proxy = _run_backend("proxy", seed=seed)
    adex = _run_backend("adex", seed=seed)
    summary = {
        "benchmark": "adex_backend_smoke",
        "seed": int(seed),
        "runtime_scope": {
            "mode": "local_plasticity_backend_comparison",
            "note": "This smoke compares proxy versus optional AdEx postsynaptic spike backends inside the maintained local plasticity path.",
        },
        "backends": {
            "proxy": proxy,
            "adex": adex,
        },
        "comparison": {
            "recon_error_delta": float(adex["mean_recon_error"] - proxy["mean_recon_error"]),
            "active_columns_delta": float(adex["mean_active_columns"] - proxy["mean_active_columns"]),
            "post_spike_fraction_delta": float(adex["mean_post_spike_fraction"] - proxy["mean_post_spike_fraction"]),
        },
        "adex_backend_gate": {
            "pass": bool(proxy["gate_pass"] and adex["gate_pass"]),
            "thresholds": {
                "finite_parameters": True,
                "adex_post_spike_fraction_gt": 0.0,
            },
        },
    }
    write_json_file(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AdEx backend comparison smoke.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "phase7_adex_backend_smoke",
        help="Directory where the benchmark summary will be written.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Deterministic seed for the benchmark.")
    args = parser.parse_args()

    summary = run_adex_backend_benchmark(output_dir=args.output_dir, seed=args.seed)
    print(f"[adex_backend_smoke] pass={summary['adex_backend_gate']['pass']}")


if __name__ == "__main__":
    main()
