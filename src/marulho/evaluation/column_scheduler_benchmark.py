"""A/B benchmark for bounded retained column-scheduler work."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import time
from typing import Sequence

import torch

from marulho.config.model_config import MarulhoConfig
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


@dataclass(frozen=True)
class SchedulerBenchmarkArm:
    mode: str
    samples: int
    median_ms: float
    p95_ms: float
    mean_ms: float
    tokens_per_second: float
    winner_ids: list[int]
    predictive_vote_mode: str
    predictive_vote_updated_columns: int
    predictive_vote_cached_columns: int
    predictive_vote_runs_all_columns: bool
    predictive_vote_fallback_reason: str | None
    predictive_update_mode: str
    predictive_update_updated_columns: int
    predictive_update_cached_columns: int
    predictive_update_runs_all_columns: bool
    predictive_update_fallback_reason: str | None
    predictive_location_update_columns: int
    predictive_location_cached_columns: int
    predictive_location_runs_all_columns: bool
    candidate_sleep_filter_mode: str
    candidate_sleep_filter_input_candidates: int
    candidate_sleep_filter_output_candidates: int
    candidate_sleep_filter_deep_sleep_filtered: int
    candidate_sleep_filter_runs_all_columns: bool
    candidate_sleep_filter_fallback_reason: str | None
    column_wake_plan_mode: str
    column_wake_plan_awake_count: int
    column_wake_plan_bounded: bool
    column_wake_plan_runs_all_columns: bool
    column_wake_plan_wake_reason: str | None
    column_wake_plan_fallback_reason: str | None
    competitive_candidate_count: int
    competitive_scored_count: int
    awake_budget: int
    awake_count: int
    cached_vote_count: int
    idle_count: int
    runs_all_columns: bool


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def _make_config(
    *,
    n_columns: int,
    column_latent_dim: int,
    k_routing: int,
    device: str,
    candidate_homeostasis_start_tokens: int,
    candidate_predictive_update_start_tokens: int,
    candidate_deep_sleep_filter_start_tokens: int,
) -> MarulhoConfig:
    return MarulhoConfig(
        n_columns=int(n_columns),
        column_latent_dim=int(column_latent_dim),
        k_routing=int(k_routing),
        bootstrap_tokens=0,
        memory_capacity=max(64, int(n_columns) * 2),
        routing_index_mode="torch_topk",
        candidate_homeostasis_start_tokens=int(candidate_homeostasis_start_tokens),
        candidate_predictive_update_start_tokens=int(candidate_predictive_update_start_tokens),
        candidate_deep_sleep_filter_start_tokens=int(candidate_deep_sleep_filter_start_tokens),
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_cross_modal=False,
        micro_sleep_interval_tokens=10**9,
        deep_sleep_interval_tokens=10**9,
        slow_memory_archive_interval_tokens=10**9,
        slow_memory_archive_strong_capture_threshold=999.0,
        trainer_telemetry_interval_tokens=10**9,
        device=device,
    )


def _run_arm(
    *,
    cfg: MarulhoConfig,
    patterns: list[torch.Tensor],
    raw_prefix: str,
    force_all_column_vote: bool,
    seed: int,
    warmup_steps: int,
) -> SchedulerBenchmarkArm:
    torch.manual_seed(int(seed))
    trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
    trainer.memory_warm_started = False

    if force_all_column_vote:
        original_vote = trainer.model.predictive.vote

        def all_column_vote(
            winners: list[int],
            top_k_activations: torch.Tensor,
            candidate_indices: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return original_vote(winners, top_k_activations, candidate_indices=None)

        trainer.model.predictive.vote = all_column_vote  # type: ignore[method-assign]

    if cfg.resolve_device().type == "cuda":
        torch.cuda.synchronize()

    for index, pattern in enumerate(patterns[:warmup_steps]):
        trainer.train_step(
            pattern,
            raw_window=f"{raw_prefix} warmup {index}",
            allow_sleep_maintenance=False,
            return_metrics=False,
        )

    timings: list[float] = []
    winner_ids: list[int] = []
    measure = patterns[warmup_steps:]
    for offset, pattern in enumerate(measure):
        if cfg.resolve_device().type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        metrics = trainer.train_step(
            pattern,
            raw_window=f"{raw_prefix} measure {offset}",
            allow_sleep_maintenance=False,
        )
        if cfg.resolve_device().type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter_ns() - started) / 1e6)
        winner = metrics.get("winner")
        if winner is not None:
            winner_ids.append(int(winner))

    vote = trainer.model.predictive.vote_execution_report()
    update = trainer.model.predictive.prediction_update_execution_report()
    column_runtime = trainer.model.column_runtime_report(
        token_count=trainer.token_count,
        last_winner=trainer.last_winner,
    )
    execution = column_runtime.get("execution", {})
    sleep_filter = column_runtime.get("candidate_sleep_filter_execution", {})
    wake_plan = column_runtime.get("column_wake_plan", {})
    elapsed_ms = sum(timings)
    samples = len(timings)
    return SchedulerBenchmarkArm(
        mode="legacy_all_column_vote" if force_all_column_vote else "scoped_cached_vote",
        samples=samples,
        median_ms=statistics.median(timings),
        p95_ms=_percentile(timings, 0.95),
        mean_ms=statistics.mean(timings),
        tokens_per_second=(samples / max(elapsed_ms / 1000.0, 1e-9)),
        winner_ids=winner_ids,
        predictive_vote_mode=str(vote.get("mode")),
        predictive_vote_updated_columns=int(vote.get("updated_column_count", 0) or 0),
        predictive_vote_cached_columns=int(vote.get("cached_vote_use_count", 0) or 0),
        predictive_vote_runs_all_columns=bool(vote.get("runs_all_columns", False)),
        predictive_vote_fallback_reason=(
            None
            if vote.get("fallback_reason") is None
            else str(vote.get("fallback_reason"))
        ),
        predictive_update_mode=str(update.get("mode")),
        predictive_update_updated_columns=int(update.get("updated_column_count", 0) or 0),
        predictive_update_cached_columns=int(update.get("cached_state_count", 0) or 0),
        predictive_update_runs_all_columns=bool(update.get("runs_all_columns", False)),
        predictive_update_fallback_reason=(
            None
            if update.get("fallback_reason") is None
            else str(update.get("fallback_reason"))
        ),
        predictive_location_update_columns=int(
            update.get("location_update_count", 0) or 0
        ),
        predictive_location_cached_columns=int(
            update.get("location_cached_count", 0) or 0
        ),
        predictive_location_runs_all_columns=bool(
            update.get("location_update_runs_all_columns", False)
        ),
        candidate_sleep_filter_mode=str(sleep_filter.get("mode")),
        candidate_sleep_filter_input_candidates=int(
            sleep_filter.get("input_candidate_count", 0) or 0
        ),
        candidate_sleep_filter_output_candidates=int(
            sleep_filter.get("output_candidate_count", 0) or 0
        ),
        candidate_sleep_filter_deep_sleep_filtered=int(
            sleep_filter.get("filtered_deep_sleep_count", 0) or 0
        ),
        candidate_sleep_filter_runs_all_columns=bool(
            sleep_filter.get("runs_all_columns", False)
        ),
        candidate_sleep_filter_fallback_reason=(
            None
            if sleep_filter.get("fallback_reason") is None
            else str(sleep_filter.get("fallback_reason"))
        ),
        column_wake_plan_mode=str(wake_plan.get("mode")),
        column_wake_plan_awake_count=int(wake_plan.get("awake_count", 0) or 0),
        column_wake_plan_bounded=bool(wake_plan.get("bounded", False)),
        column_wake_plan_runs_all_columns=bool(
            wake_plan.get("runs_all_columns", False)
        ),
        column_wake_plan_wake_reason=(
            None
            if wake_plan.get("wake_reason") is None
            else str(wake_plan.get("wake_reason"))
        ),
        column_wake_plan_fallback_reason=(
            None
            if wake_plan.get("fallback_reason") is None
            else str(wake_plan.get("fallback_reason"))
        ),
        competitive_candidate_count=int(execution.get("candidate_count", 0) or 0),
        competitive_scored_count=int(execution.get("scored_column_count", 0) or 0),
        awake_budget=int(column_runtime.get("awake_budget", 0) or 0),
        awake_count=int(column_runtime.get("awake_count", 0) or 0),
        cached_vote_count=int(column_runtime.get("cached_vote_count", 0) or 0),
        idle_count=int(column_runtime.get("idle_count", 0) or 0),
        runs_all_columns=bool(column_runtime.get("runs_all_columns", False)),
    )


def run_benchmark(
    *,
    n_columns: int = 2048,
    column_latent_dim: int = 64,
    k_routing: int = 10,
    samples: int = 80,
    warmup_steps: int = 10,
    seed: int = 20260615,
    device: str = "cpu",
) -> dict[str, object]:
    if samples <= 0 or warmup_steps < 0:
        raise ValueError("samples must be positive and warmup_steps non-negative")
    if k_routing <= 0 or k_routing > n_columns:
        raise ValueError("k_routing must be in [1, n_columns]")

    all_column_cfg = _make_config(
        n_columns=n_columns,
        column_latent_dim=column_latent_dim,
        k_routing=k_routing,
        device=device,
        candidate_homeostasis_start_tokens=max(samples + warmup_steps + 1, 10**9),
        candidate_predictive_update_start_tokens=max(samples + warmup_steps + 1, 10**9),
        candidate_deep_sleep_filter_start_tokens=max(samples + warmup_steps + 1, 10**9),
    )
    scoped_cfg = _make_config(
        n_columns=n_columns,
        column_latent_dim=column_latent_dim,
        k_routing=k_routing,
        device=device,
        candidate_homeostasis_start_tokens=0,
        candidate_predictive_update_start_tokens=0,
        candidate_deep_sleep_filter_start_tokens=0,
    )
    resolved_device = scoped_cfg.resolve_device()
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    generator = torch.Generator(device=resolved_device).manual_seed(int(seed))
    patterns = [
        torch.rand(scoped_cfg.input_dim, generator=generator, device=resolved_device)
        for _ in range(samples + warmup_steps)
    ]

    all_vote = _run_arm(
        cfg=all_column_cfg,
        patterns=patterns,
        raw_prefix="all-column vote and prediction update",
        force_all_column_vote=True,
        seed=seed,
        warmup_steps=warmup_steps,
    )
    scoped = _run_arm(
        cfg=scoped_cfg,
        patterns=patterns,
        raw_prefix="scoped cached vote and prediction update",
        force_all_column_vote=False,
        seed=seed,
        warmup_steps=warmup_steps,
    )
    median_delta_percent = (
        (all_vote.median_ms - scoped.median_ms) / max(all_vote.median_ms, 1e-9)
    ) * 100.0
    mean_delta_percent = (
        (all_vote.mean_ms - scoped.mean_ms) / max(all_vote.mean_ms, 1e-9)
    ) * 100.0
    return {
        "surface": "column_scheduler_benchmark.v1",
        "scope": "complete_train_step_deep_sleep_filter_predictive_update_and_vote_awake_mask_ab",
        "torch": torch.__version__,
        "device": str(resolved_device),
        "cuda_device_name": (
            torch.cuda.get_device_name(resolved_device)
            if resolved_device.type == "cuda"
            else None
        ),
        "seed": int(seed),
        "n_columns": int(n_columns),
        "column_latent_dim": int(column_latent_dim),
        "k_routing": int(k_routing),
        "samples": int(samples),
        "warmup_steps": int(warmup_steps),
        "all_column_vote": asdict(all_vote),
        "scoped_cached_vote": asdict(scoped),
        "winner_sequence_equal": all_vote.winner_ids == scoped.winner_ids,
        "awake_count_bounded": scoped.awake_count <= int(k_routing),
        "column_wake_plan_bounded": bool(
            scoped.column_wake_plan_bounded
            and scoped.column_wake_plan_awake_count <= int(k_routing)
            and not scoped.column_wake_plan_runs_all_columns
        ),
        "predictive_vote_bounded": scoped.predictive_vote_updated_columns <= int(k_routing),
        "predictive_update_bounded": scoped.predictive_update_updated_columns <= int(k_routing),
        "predictive_location_update_bounded": (
            scoped.predictive_location_update_columns <= int(k_routing)
            and not scoped.predictive_location_runs_all_columns
        ),
        "candidate_sleep_filter_bounded": (
            scoped.candidate_sleep_filter_output_candidates <= int(k_routing)
            and not scoped.candidate_sleep_filter_runs_all_columns
        ),
        "bounded_specialist_work": bool(
            scoped.predictive_vote_updated_columns <= int(k_routing)
            and scoped.predictive_update_updated_columns <= int(k_routing)
            and scoped.predictive_location_update_columns <= int(k_routing)
            and scoped.column_wake_plan_bounded
            and scoped.column_wake_plan_awake_count <= int(k_routing)
            and not scoped.column_wake_plan_runs_all_columns
            and not scoped.predictive_location_runs_all_columns
            and scoped.candidate_sleep_filter_output_candidates <= int(k_routing)
            and not scoped.candidate_sleep_filter_runs_all_columns
            and not scoped.runs_all_columns
        ),
        "scoped_runs_all_columns": bool(scoped.runs_all_columns),
        "median_delta_percent": median_delta_percent,
        "mean_delta_percent": mean_delta_percent,
        "neutral_or_better_complete_tick": scoped.mean_ms <= all_vote.mean_ms * 1.02,
        "claim_boundary": (
            "complete_train_step_ab_for_candidate_deep_sleep_filter_predictive_update_and_vote_scheduler_not_service_runtime_or_growth_pruning_claim"
        ),
    }


def run_scaling_benchmark(
    *,
    column_counts: Sequence[int] = (512, 2048, 8192),
    column_latent_dim: int = 64,
    k_routing: int = 10,
    samples: int = 40,
    warmup_steps: int = 5,
    seed: int = 20260615,
    device: str = "cpu",
) -> dict[str, object]:
    reports = [
        run_benchmark(
            n_columns=int(n_columns),
            column_latent_dim=column_latent_dim,
            k_routing=k_routing,
            samples=samples,
            warmup_steps=warmup_steps,
            seed=seed + offset,
            device=device,
        )
        for offset, n_columns in enumerate(column_counts)
    ]
    scoped_rows = [
        {
            "n_columns": int(report["n_columns"]),
            "winner_sequence_equal": bool(report["winner_sequence_equal"]),
            "awake_budget": int(report["k_routing"]),
            "scoped_runs_all_columns": bool(report["scoped_runs_all_columns"]),
            "predictive_update_updated_columns": int(
                report["scoped_cached_vote"]["predictive_update_updated_columns"]
            ),
            "predictive_vote_updated_columns": int(
                report["scoped_cached_vote"]["predictive_vote_updated_columns"]
            ),
            "predictive_location_update_columns": int(
                report["scoped_cached_vote"]["predictive_location_update_columns"]
            ),
            "candidate_sleep_filter_output_candidates": int(
                report["scoped_cached_vote"]["candidate_sleep_filter_output_candidates"]
            ),
            "column_wake_plan_awake_count": int(
                report["scoped_cached_vote"]["column_wake_plan_awake_count"]
            ),
            "column_wake_plan_bounded": bool(
                report["scoped_cached_vote"]["column_wake_plan_bounded"]
            ),
            "mean_ms": float(report["scoped_cached_vote"]["mean_ms"]),
            "tokens_per_second": float(
                report["scoped_cached_vote"]["tokens_per_second"]
            ),
            "neutral_or_better_complete_tick": bool(
                report["neutral_or_better_complete_tick"]
            ),
        }
        for report in reports
    ]
    return {
        "surface": "column_scheduler_scaling_benchmark.v1",
        "scope": "constant_k_candidate_deep_sleep_filter_predictive_update_and_vote_scaling",
        "torch": torch.__version__,
        "device": str(reports[0]["device"]) if reports else str(device),
        "column_counts": [int(value) for value in column_counts],
        "column_latent_dim": int(column_latent_dim),
        "k_routing": int(k_routing),
        "samples": int(samples),
        "warmup_steps": int(warmup_steps),
        "seed": int(seed),
        "runs": scoped_rows,
        "all_winner_sequences_equal": all(
            bool(report["winner_sequence_equal"]) for report in reports
        ),
        "awake_count_remains_bounded": all(
            int(row["predictive_update_updated_columns"]) <= int(k_routing)
            and int(row["predictive_vote_updated_columns"]) <= int(k_routing)
            and int(row["predictive_location_update_columns"]) <= int(k_routing)
            and int(row["candidate_sleep_filter_output_candidates"]) <= int(k_routing)
            and int(row["column_wake_plan_awake_count"]) <= int(k_routing)
            and bool(row["column_wake_plan_bounded"])
            for row in scoped_rows
        ),
        "scoped_never_runs_all_columns": all(
            not bool(row["scoped_runs_all_columns"]) for row in scoped_rows
        ),
        "neutral_or_better_all_sizes": all(
            bool(row["neutral_or_better_complete_tick"]) for row in scoped_rows
        ),
        "claim_boundary": (
            "scaling_sweep_for_constant_k_scheduler_evidence_not_growth_pruning_or_cuda_claim"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-columns", type=int, default=2048)
    parser.add_argument("--column-latent-dim", type=int, default=64)
    parser.add_argument("--k-routing", type=int, default=10)
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--sweep-columns", nargs="*", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.sweep_columns:
        report = run_scaling_benchmark(
            column_counts=args.sweep_columns,
            column_latent_dim=args.column_latent_dim,
            k_routing=args.k_routing,
            samples=args.samples,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
            device=args.device,
        )
    else:
        report = run_benchmark(
            n_columns=args.n_columns,
            column_latent_dim=args.column_latent_dim,
            k_routing=args.k_routing,
            samples=args.samples,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
            device=args.device,
        )
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
