"""A/B benchmark for predictive-vote cached column scheduling."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import time

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
) -> MarulhoConfig:
    return MarulhoConfig(
        n_columns=int(n_columns),
        column_latent_dim=int(column_latent_dim),
        k_routing=int(k_routing),
        bootstrap_tokens=0,
        memory_capacity=max(64, int(n_columns) * 2),
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="legacy",
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
    column_runtime = trainer.model.column_runtime_report(
        token_count=trainer.token_count,
        last_winner=trainer.last_winner,
    )
    execution = column_runtime.get("execution", {})
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

    cfg = _make_config(
        n_columns=n_columns,
        column_latent_dim=column_latent_dim,
        k_routing=k_routing,
        device=device,
    )
    resolved_device = cfg.resolve_device()
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    generator = torch.Generator(device=resolved_device).manual_seed(int(seed))
    patterns = [
        torch.rand(cfg.input_dim, generator=generator, device=resolved_device)
        for _ in range(samples + warmup_steps)
    ]

    all_vote = _run_arm(
        cfg=cfg,
        patterns=patterns,
        raw_prefix="all-column vote",
        force_all_column_vote=True,
        seed=seed,
        warmup_steps=warmup_steps,
    )
    scoped = _run_arm(
        cfg=cfg,
        patterns=patterns,
        raw_prefix="scoped cached vote",
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
        "scope": "complete_train_step_predictive_vote_awake_mask_ab",
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
        "predictive_vote_bounded": scoped.predictive_vote_updated_columns <= int(k_routing),
        "scoped_runs_all_columns": bool(scoped.predictive_vote_runs_all_columns),
        "median_delta_percent": median_delta_percent,
        "mean_delta_percent": mean_delta_percent,
        "neutral_or_better_complete_tick": scoped.mean_ms <= all_vote.mean_ms * 1.02,
        "claim_boundary": (
            "complete_train_step_ab_for_predictive_vote_scheduler_not_service_runtime_or_growth_pruning_claim"
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

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
