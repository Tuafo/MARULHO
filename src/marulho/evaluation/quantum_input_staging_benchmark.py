"""Reversed A/B for persistent CUDA quantum input/control staging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

from marulho.evaluation.hot_window_benchmark import run_hot_window_benchmark
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.column_transition_runtime import ColumnTransitionRuntime
from marulho.training.trainer import MarulhoTrainer


def _install(trainer_object: object, *, enabled: bool) -> None:
    if not isinstance(trainer_object, MarulhoTrainer):
        raise TypeError("quantum input staging benchmark requires MarulhoTrainer")
    trainer = trainer_object
    trainer.config.predictive_dense_transition_mode = "inplace_triton"
    trainer.config.predictive_route_vote_mode = "cuda_graph_text"
    trainer.config.cuda_graph_quantum_input_staging = bool(enabled)
    trainer._column_transition_runtime = ColumnTransitionRuntime(trainer)
    graph = trainer.column_transition_runtime_report().get(
        "cuda_graph_route_transition"
    )
    if not isinstance(graph, dict) or not graph.get("active"):
        raise RuntimeError(f"persistent CUDA graph unavailable: {graph}")
    trainer._benchmark_transition_executor = (
        "quantum_input_staging"
        if enabled
        else "per_token_input_staging"
    )


def _disabled(trainer: object) -> None:
    _install(trainer, enabled=False)


def _enabled(trainer: object) -> None:
    _install(trainer, enabled=True)


def run_quantum_input_staging_ab(
    checkpoint: Path,
    *,
    output_path: Path,
    samples: int = 256,
    warmup_steps: int = 32,
    quantum_tokens: int = 8,
    seed: int = 20260614,
    profile_trainer_stages: bool = False,
    sync_mode: str = "window",
) -> dict[str, Any]:
    arm_specs = (
        ("per_token_a", _disabled),
        ("quantum_a", _enabled),
        ("quantum_b", _enabled),
        ("per_token_b", _disabled),
    )
    arms: list[dict[str, Any]] = []
    for name, setup in arm_specs:
        result = run_hot_window_benchmark(
            checkpoint,
            samples=samples,
            warmup_steps=warmup_steps,
            seed=seed,
            _trainer_setup=setup,
            profile_trainer_stages=profile_trainer_stages,
            sync_mode=sync_mode,
            input_quantum_tokens=quantum_tokens,
        )
        graph = result["runtime_counters"]["column_transition_runtime"][
            "cuda_graph_route_transition"
        ]
        arm = {
            "name": name,
            "tokens_per_second": result["tokens_per_second"],
            "step_latency_ms": result["step_latency_ms"],
            "quantum_input_stage_elapsed_ms": result[
                "quantum_input_stage_elapsed_ms"
            ],
            "cuda_memory": result["cuda_memory"],
            "graph_runtime": graph,
        }
        if profile_trainer_stages:
            arm["trainer_stage_profile"] = result["trainer_stage_profile"]
        arms.append(arm)

    per_token = [
        float(arm["tokens_per_second"])
        for arm in arms
        if str(arm["name"]).startswith("per_token")
    ]
    quantum = [
        float(arm["tokens_per_second"])
        for arm in arms
        if str(arm["name"]).startswith("quantum")
    ]
    per_token_mean = float(statistics.fmean(per_token))
    quantum_mean = float(statistics.fmean(quantum))
    report = {
        "surface": "quantum_input_staging_ab.v1",
        "checkpoint": str(checkpoint),
        "scope": (
            "complete_encoded_sequential_train_step_with_persistent_cuda_graph"
        ),
        "claim_boundary": (
            "compares one fixed-address CUDA input-ring stage per bounded quantum "
            "against per-token static-buffer copies; neural token order and graph "
            "transition math remain unchanged"
        ),
        "samples_per_arm": int(samples),
        "warmup_steps_per_arm": int(warmup_steps),
        "quantum_tokens": int(quantum_tokens),
        "sync_mode": str(sync_mode),
        "profile_trainer_stages": bool(profile_trainer_stages),
        "arms": arms,
        "per_token_mean_tokens_per_second": per_token_mean,
        "quantum_mean_tokens_per_second": quantum_mean,
        "speedup": quantum_mean / max(per_token_mean, 1e-9),
        "success": bool(quantum_mean > per_token_mean),
    }
    write_json_report_with_readme(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--quantum-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--profile-trainer-stages", action="store_true")
    parser.add_argument(
        "--sync-mode",
        choices=("step", "window"),
        default="window",
    )
    args = parser.parse_args()
    report = run_quantum_input_staging_ab(
        args.checkpoint,
        output_path=args.output,
        samples=args.samples,
        warmup_steps=args.warmup_steps,
        quantum_tokens=args.quantum_tokens,
        seed=args.seed,
        profile_trainer_stages=args.profile_trainer_stages,
        sync_mode=args.sync_mode,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
