"""Reversed complete-tick A/B for graph-owned competitive surprise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

from marulho.evaluation.fused_route_vote_hot_window_benchmark import (
    install_cuda_graph_route_transition_for_benchmark,
)
from marulho.evaluation.hot_window_benchmark import run_hot_window_benchmark
from marulho.training.trainer import MarulhoTrainer


def _install_control(trainer_object: object) -> None:
    trainer = trainer_object
    if not isinstance(trainer, MarulhoTrainer):
        raise TypeError("control setup requires MarulhoTrainer")
    trainer._disable_graph_competitive_surprise_for_evaluation = True
    install_cuda_graph_route_transition_for_benchmark(trainer)
    trainer._benchmark_transition_executor = (
        "persistent_tick_post_transition_competitive_surprise"
    )


def _install_variant(trainer_object: object) -> None:
    trainer = trainer_object
    if not isinstance(trainer, MarulhoTrainer):
        raise TypeError("variant setup requires MarulhoTrainer")
    install_cuda_graph_route_transition_for_benchmark(trainer)
    trainer._benchmark_transition_executor = (
        "persistent_tick_graph_competitive_surprise"
    )


def run_graph_competitive_surprise_ab(
    checkpoint: Path,
    *,
    samples: int = 256,
    warmup_steps: int = 64,
    seed: int = 20260613,
) -> dict[str, Any]:
    arms: list[dict[str, Any]] = []
    for name, setup in (
        ("control_a", _install_control),
        ("variant_a", _install_variant),
        ("variant_b", _install_variant),
        ("control_b", _install_control),
    ):
        report = run_hot_window_benchmark(
            checkpoint,
            samples=samples,
            warmup_steps=warmup_steps,
            seed=seed,
            _trainer_setup=setup,
        )
        graph_runtime = report["runtime_counters"]["column_transition_runtime"][
            "cuda_graph_route_transition"
        ]
        arms.append(
            {
                "name": name,
                "ticks_per_second": report["tokens_per_second"],
                "latency_ms": report["step_latency_ms"],
                "transition_executor": report["transition_executor"],
                "graph_runtime": graph_runtime,
                "cuda_memory": report["cuda_memory"],
            }
        )

    control_mean = statistics.fmean(
        float(arm["ticks_per_second"])
        for arm in arms
        if str(arm["name"]).startswith("control")
    )
    variant_mean = statistics.fmean(
        float(arm["ticks_per_second"])
        for arm in arms
        if str(arm["name"]).startswith("variant")
    )
    return {
        "surface": "graph_competitive_surprise_ab.v1",
        "checkpoint": str(checkpoint),
        "scope": "complete_encoded_text_train_step_no_service_no_source_no_sleep",
        "claim_boundary": (
            "variant records post-transition competitive error from the existing "
            "persistent graph result readback; control launches torch.norm and "
            "performs a second host-visible scalar extraction after transition"
        ),
        "samples_per_arm": samples,
        "warmup_steps_per_arm": warmup_steps,
        "seed": seed,
        "arms": arms,
        "control_mean_ticks_per_second": control_mean,
        "variant_mean_ticks_per_second": variant_mean,
        "speedup": variant_mean / max(control_mean, 1e-9),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260613)
    args = parser.parse_args()
    report = run_graph_competitive_surprise_ab(
        args.checkpoint,
        samples=args.samples,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
    )
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
