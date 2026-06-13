"""Complete train-step A/B for the persistent CUDA text-tick executor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any, Callable

from marulho.evaluation.fused_route_vote_hot_window_benchmark import (
    install_cuda_graph_route_transition_for_benchmark,
    install_fused_route_vote_for_benchmark,
)
from marulho.evaluation.hot_window_benchmark import run_hot_window_benchmark


def _profile_per_tick_ms(arm: dict[str, Any]) -> dict[str, float]:
    profile = arm.get("trainer_stage_profile")
    if not isinstance(profile, dict):
        return {}
    per_tick = profile.get("per_tick_ms")
    if not isinstance(per_tick, dict):
        return {}
    return {
        str(name): float(value)
        for name, value in per_tick.items()
        if isinstance(value, (int, float))
    }


def _mean_stage_profile(
    arms: list[dict[str, Any]],
    prefix: str,
) -> dict[str, float]:
    stage_values: dict[str, list[float]] = {}
    for arm in arms:
        if not str(arm.get("name", "")).startswith(prefix):
            continue
        for name, value in _profile_per_tick_ms(arm).items():
            stage_values.setdefault(name, []).append(float(value))
    return {
        name: round(float(statistics.fmean(values)), 6)
        for name, values in sorted(stage_values.items())
        if values
    }


def _stage_deltas(
    baseline: dict[str, float],
    variant: dict[str, float],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name in sorted(set(baseline) | set(variant)):
        baseline_ms = float(baseline.get(name, 0.0))
        variant_ms = float(variant.get(name, 0.0))
        rows.append(
            {
                "stage": name,
                "baseline_ms": round(baseline_ms, 6),
                "variant_ms": round(variant_ms, 6),
                "delta_ms_variant_minus_baseline": round(
                    variant_ms - baseline_ms,
                    6,
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: abs(float(row["delta_ms_variant_minus_baseline"])),
        reverse=True,
    )


def run_persistent_tick_hot_window_ab(
    checkpoint: Path,
    *,
    samples: int = 512,
    warmup_steps: int = 64,
    seed: int = 20260612,
    profile_trainer_stages: bool = False,
    sync_mode: str = "step",
    _arm_setups: tuple[tuple[str, Callable[[object], None]], ...] | None = None,
) -> dict[str, object]:
    arms: list[dict[str, Any]] = []
    arm_setups = _arm_setups or (
        ("fused_a", install_fused_route_vote_for_benchmark),
        ("persistent_a", install_cuda_graph_route_transition_for_benchmark),
        ("persistent_b", install_cuda_graph_route_transition_for_benchmark),
        ("fused_b", install_fused_route_vote_for_benchmark),
    )
    for name, setup in arm_setups:
        report = run_hot_window_benchmark(
            checkpoint,
            samples=samples,
            warmup_steps=warmup_steps,
            seed=seed,
            _trainer_setup=setup,
            profile_trainer_stages=profile_trainer_stages,
            sync_mode=sync_mode,
        )
        arm = {
            "name": name,
            "tokens_per_second": report["tokens_per_second"],
            "step_latency_ms": report["step_latency_ms"],
            "transition_executor": report["transition_executor"],
            "runtime_counters": report["runtime_counters"],
            "cuda_memory": report["cuda_memory"],
        }
        if profile_trainer_stages:
            arm["trainer_stage_profile"] = report["trainer_stage_profile"]
        arms.append(arm)

    fused = [
        float(arm["tokens_per_second"])
        for arm in arms
        if str(arm["name"]).startswith("fused")
    ]
    persistent = [
        float(arm["tokens_per_second"])
        for arm in arms
        if str(arm["name"]).startswith("persistent")
    ]
    fused_mean = sum(fused) / len(fused)
    persistent_mean = sum(persistent) / len(persistent)
    fused_stage_mean = _mean_stage_profile(arms, "fused")
    persistent_stage_mean = _mean_stage_profile(arms, "persistent")
    return {
        "surface": "persistent_tick_hot_window_ab.v1",
        "checkpoint": str(checkpoint),
        "scope": "complete_encoded_tensor_train_step_no_service_no_source_no_sleep",
        "claim_boundary": (
            "eligible text ticks capture reconstruction, neuromodulation, "
            "route/vote, and in-place transition in one fixed-address replay; "
            "sensory ticks and graph-ineligible configurations retain fallback; "
            "optional trainer-stage profiling is measurement-only and excludes warmup"
        ),
        "samples_per_arm": samples,
        "warmup_steps_per_arm": warmup_steps,
        "seed": seed,
        "profile_trainer_stages": bool(profile_trainer_stages),
        "sync_mode": sync_mode,
        "sync_mode_semantics": (
            "step mode synchronizes around every measured token; "
            "window mode synchronizes once around each measured arm to expose "
            "continuous sequential throughput without artificial per-token host barriers"
        ),
        "arms": arms,
        "fused_mean_tokens_per_second": fused_mean,
        "persistent_mean_tokens_per_second": persistent_mean,
        "speedup": persistent_mean / max(fused_mean, 1e-9),
        "fused_mean_stage_per_tick_ms": fused_stage_mean,
        "persistent_mean_stage_per_tick_ms": persistent_stage_mean,
        "largest_stage_deltas": _stage_deltas(
            fused_stage_mean,
            persistent_stage_mean,
        )[:12],
        "promotion_status": (
            "requires_grounded_trajectory_gate"
            if persistent_mean > fused_mean
            else "rejected_no_complete_tick_gain"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--profile-trainer-stages", action="store_true")
    parser.add_argument(
        "--sync-mode",
        choices=("step", "window"),
        default="step",
    )
    args = parser.parse_args()
    report = run_persistent_tick_hot_window_ab(
        args.checkpoint,
        samples=args.samples,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        profile_trainer_stages=args.profile_trainer_stages,
        sync_mode=args.sync_mode,
    )
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
