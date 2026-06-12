"""Full hot-window A/B runner for the evaluation-only in-place CUDA transition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marulho.evaluation.hot_window_benchmark import run_hot_window_benchmark
from marulho.training.column_transition_runtime import ColumnTransitionRuntime
from marulho.training.trainer import MarulhoTrainer


def install_inplace_transition_for_benchmark(trainer_object: object) -> None:
    trainer = trainer_object
    if not isinstance(trainer, MarulhoTrainer):
        raise TypeError("in-place transition setup requires MarulhoTrainer")
    trainer.config.predictive_dense_transition_mode = "inplace_triton"
    trainer._column_transition_runtime = ColumnTransitionRuntime(trainer)
    if not trainer._column_transition_runtime.active:
        raise RuntimeError(
            "in-place transition unavailable: "
            f"{trainer._column_transition_runtime.fallback_reason}"
        )
    trainer._benchmark_transition_executor = "inplace_triton"


def install_retained_selection_for_benchmark(trainer_object: object) -> None:
    trainer = trainer_object
    if not isinstance(trainer, MarulhoTrainer):
        raise TypeError("in-place transition setup requires MarulhoTrainer")
    trainer.config.predictive_dense_transition_mode = "inplace_triton"
    trainer._column_transition_runtime = ColumnTransitionRuntime(
        trainer,
        device_selection=False,
    )
    if not trainer._column_transition_runtime.active:
        raise RuntimeError(
            "in-place transition unavailable: "
            f"{trainer._column_transition_runtime.fallback_reason}"
        )
    trainer._benchmark_transition_executor = "inplace_triton_retained_selection"


def install_unfused_device_selection_for_benchmark(trainer_object: object) -> None:
    trainer = trainer_object
    if not isinstance(trainer, MarulhoTrainer):
        raise TypeError("in-place transition setup requires MarulhoTrainer")
    trainer.config.predictive_dense_transition_mode = "inplace_triton"
    trainer._column_transition_runtime = ColumnTransitionRuntime(
        trainer,
        fused_vote_competition=False,
    )
    if not trainer._column_transition_runtime.active:
        raise RuntimeError(
            "in-place transition unavailable: "
            f"{trainer._column_transition_runtime.fallback_reason}"
        )
    trainer._benchmark_transition_executor = "inplace_triton_unfused_device_selection"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument(
        "--winner-selection",
        choices=("fused", "device", "retained"),
        default="fused",
    )
    args = parser.parse_args()
    setup = {
        "fused": install_inplace_transition_for_benchmark,
        "device": install_unfused_device_selection_for_benchmark,
        "retained": install_retained_selection_for_benchmark,
    }[args.winner_selection]
    report = run_hot_window_benchmark(
        args.checkpoint,
        samples=args.samples,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        _trainer_setup=setup,
    )
    report["surface"] = "inplace_hot_window_benchmark.v1"
    report["promotion_status"] = "production_executor_benchmark"
    report["winner_selection"] = args.winner_selection
    report["claim_boundary"] = (
        "complete configured train_step without service/source/sleep; "
        "production executor lifecycle is active only when checkpoint config "
        "requests it and startup warmup succeeds"
    )
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
