"""Complete train-step A/B for checkpoint-opt-in fused text routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from marulho.evaluation.hot_window_benchmark import run_hot_window_benchmark
from marulho.training.column_transition_runtime import ColumnTransitionRuntime
from marulho.training.trainer import MarulhoTrainer


def _set_route_vote_mode_for_evaluation(
    trainer: MarulhoTrainer,
    mode: str,
) -> None:
    trainer._route_vote_mode_override_for_evaluation = mode
    trainer._column_transition_runtime = ColumnTransitionRuntime(trainer)


def _install_production(trainer_object: object) -> None:
    trainer = trainer_object
    if not isinstance(trainer, MarulhoTrainer):
        raise TypeError("production setup requires MarulhoTrainer")
    trainer.config.predictive_dense_transition_mode = "inplace_triton"
    _set_route_vote_mode_for_evaluation(trainer, "tensor")
    if not trainer._column_transition_runtime.active:
        raise RuntimeError(
            "in-place transition unavailable: "
            f"{trainer._column_transition_runtime.fallback_reason}"
        )
    trainer._benchmark_transition_executor = "production_tensor_route_fused_vote"


def install_fused_route_vote_for_benchmark(trainer_object: object) -> None:
    trainer = trainer_object
    if not isinstance(trainer, MarulhoTrainer):
        raise TypeError("fused route/vote setup requires MarulhoTrainer")
    trainer.config.predictive_dense_transition_mode = "inplace_triton"
    _set_route_vote_mode_for_evaluation(trainer, "fused_triton_text")
    if not trainer._column_transition_runtime.handles_route_vote:
        raise RuntimeError(
            "fused route/vote unavailable: "
            f"{trainer._column_transition_runtime.route_vote_fallback_reason}"
        )
    trainer._benchmark_transition_executor = "production_fused_triton_text_route_vote"


def install_cuda_graph_route_transition_for_benchmark(
    trainer_object: object,
) -> None:
    trainer = trainer_object
    if not isinstance(trainer, MarulhoTrainer):
        raise TypeError("CUDA graph setup requires MarulhoTrainer")
    trainer.config.predictive_dense_transition_mode = "inplace_triton"
    _set_route_vote_mode_for_evaluation(trainer, "cuda_graph_text")
    graph_report = trainer._column_transition_runtime.report().get(
        "cuda_graph_route_transition"
    )
    if not isinstance(graph_report, dict) or not graph_report.get("active"):
        raise RuntimeError(
            "CUDA graph route/transition unavailable: "
            f"{trainer._column_transition_runtime.route_vote_fallback_reason}"
        )
    trainer._benchmark_transition_executor = (
        "production_cuda_graph_text_route_transition"
    )


def run_fused_route_vote_hot_window_ab(
    checkpoint: Path,
    *,
    samples: int = 256,
    warmup_steps: int = 32,
    seed: int = 20260612,
) -> dict[str, object]:
    arms: list[dict[str, object]] = []
    for name, setup in (
        ("production_a", _install_production),
        ("fused_a", install_fused_route_vote_for_benchmark),
        ("fused_b", install_fused_route_vote_for_benchmark),
        ("production_b", _install_production),
    ):
        report = run_hot_window_benchmark(
            checkpoint,
            samples=samples,
            warmup_steps=warmup_steps,
            seed=seed,
            _trainer_setup=setup,
        )
        arms.append(
            {
                "name": name,
                "tokens_per_second": report["tokens_per_second"],
                "step_latency_ms": report["step_latency_ms"],
                "transition_executor": report["transition_executor"],
                "runtime_counters": report["runtime_counters"],
                "cuda_memory": report["cuda_memory"],
            }
        )
    production = [
        float(arm["tokens_per_second"])
        for arm in arms
        if str(arm["name"]).startswith("production")
    ]
    fused = [
        float(arm["tokens_per_second"])
        for arm in arms
        if str(arm["name"]).startswith("fused")
    ]
    production_mean = sum(production) / len(production)
    fused_mean = sum(fused) / len(fused)
    return {
        "surface": "fused_route_vote_hot_window_ab.v1",
        "checkpoint": str(checkpoint),
        "scope": "complete_encoded_tensor_train_step_no_service_no_source_no_sleep",
        "claim_boundary": (
            "checkpoint-opt-in production lifecycle fuses routing plus winner "
            "selection on text/idle ticks; sensory ticks retain tensor routing"
        ),
        "samples_per_arm": samples,
        "warmup_steps_per_arm": warmup_steps,
        "seed": seed,
        "arms": arms,
        "production_mean_tokens_per_second": production_mean,
        "fused_mean_tokens_per_second": fused_mean,
        "speedup": fused_mean / max(production_mean, 1e-9),
        "promotion_status": (
            "requires_grounded_trajectory_gate"
            if fused_mean > production_mean
            else "rejected_no_complete_tick_gain"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260612)
    args = parser.parse_args()
    report = run_fused_route_vote_hot_window_ab(
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
