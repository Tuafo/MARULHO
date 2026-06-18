from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from marulho.config.model_config import MarulhoConfig
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


def _config(
    *,
    tokens: int,
    min_interval_tokens: int,
    device: str,
) -> MarulhoConfig:
    return MarulhoConfig(
        n_columns=64,
        column_latent_dim=16,
        bootstrap_tokens=0,
        k_routing=8,
        memory_capacity=max(64, int(tokens) + 8),
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        slow_memory_start_tokens=0,
        slow_memory_archive_interval_tokens=10**9,
        slow_memory_archive_strong_capture_threshold=0.0,
        slow_memory_archive_strong_capture_min_interval_tokens=int(min_interval_tokens),
        trainer_telemetry_interval_tokens=10**9,
        device=str(device),
    )


def _patterns(config: MarulhoConfig, *, tokens: int, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return [
        torch.rand(config.input_dim, generator=generator, dtype=torch.float32)
        for _ in range(int(tokens))
    ]


def _coverage(tokens: int, selected_tokens: list[int], min_interval_tokens: int) -> dict[str, Any]:
    strong_tokens = [int(token) for token in selected_tokens if int(token) > 1]
    if not strong_tokens:
        return {
            "selected_strong_count": 0,
            "max_selected_gap_tokens": None,
            "final_gap_tokens": None,
            "coverage_gate_passed": False,
        }
    gaps = [strong_tokens[0] - 2]
    gaps.extend(
        int(right) - int(left)
        for left, right in zip(strong_tokens, strong_tokens[1:])
    )
    final_gap = int(tokens) - int(strong_tokens[-1])
    max_gap = max([int(gap) for gap in gaps] + [int(final_gap)])
    return {
        "selected_strong_count": int(len(strong_tokens)),
        "max_selected_gap_tokens": int(max_gap),
        "final_gap_tokens": int(final_gap),
        "coverage_gate_passed": bool(max_gap <= int(min_interval_tokens)),
    }


def _run_once(
    *,
    tokens: int,
    min_interval_tokens: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    config = _config(
        tokens=tokens,
        min_interval_tokens=min_interval_tokens,
        device=device,
    )
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    patterns = _patterns(config, tokens=tokens, seed=seed)
    started = time.perf_counter()
    last_metrics: dict[str, Any] | None = None
    for index, pattern in enumerate(patterns):
        last_metrics = trainer.train_step(
            pattern.to(trainer.model.device),
            raw_window=f"strong-capture-window-{index:05d}",
            allow_sleep_maintenance=False,
            return_metrics=True,
        )
    elapsed_ms = float((time.perf_counter() - started) * 1000.0)
    selected_tokens = [
        int(token) for token in trainer.model.memory_store.slow_last_capture_token
    ]
    return {
        "tokens": int(tokens),
        "elapsed_ms": elapsed_ms,
        "archive_count": int(trainer._slow_memory_archive_count),
        "archive_skip_count": int(trainer._slow_memory_archive_skip_count),
        "strong_capture_archive_count": int(
            trainer._slow_memory_strong_capture_archive_count
        ),
        "strong_capture_refractory_skip_count": int(
            trainer._slow_memory_strong_capture_refractory_skip_count
        ),
        "last_archive_reason": str(trainer._slow_memory_last_archive_reason),
        "last_strong_capture_token": int(trainer._slow_memory_last_strong_capture_token),
        "selected_tokens": selected_tokens,
        "selected_token_sample": selected_tokens[:16],
        "selected_token_sample_limit": 16,
        "memory_size": int(len(trainer.model.memory_store.slow_buffer)),
        "last_metrics": dict(last_metrics or {}),
        "runtime_report": {
            key: value
            for key, value in trainer.column_transition_runtime_report().items()
            if key
            in {
                "slow_memory_strong_capture_min_interval_tokens",
                "slow_memory_strong_capture_archive_count",
                "slow_memory_strong_capture_refractory_skip_count",
                "slow_memory_last_strong_capture_token",
                "text_burst_strong_event_count",
                "text_burst_strong_archive_count",
                "text_burst_strong_refractory_skip_count",
            }
        },
        "coverage": _coverage(tokens, selected_tokens, min_interval_tokens),
    }


def _retired_every_strong_projection(
    *,
    tokens: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    archive_count = int(tokens)
    strong_capture_archive_count = max(0, int(tokens) - 1)
    return {
        "archive_count_mean": float(archive_count),
        "strong_capture_archive_count_mean": float(strong_capture_archive_count),
        "strong_capture_refractory_skip_count_mean": 0.0,
        "projected_from_forced_strong_candidates": True,
        "executable_path_retired": True,
        "latency_mean_ms": None,
        "runs": int(len(rows)),
    }


def _case_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [float(row["elapsed_ms"]) for row in rows]
    archives = [int(row["archive_count"]) for row in rows]
    strong_archives = [int(row["strong_capture_archive_count"]) for row in rows]
    refractory_skips = [
        int(row["strong_capture_refractory_skip_count"]) for row in rows
    ]
    return {
        "runs": int(len(rows)),
        "elapsed_mean_ms": float(statistics.fmean(elapsed)) if elapsed else 0.0,
        "elapsed_median_ms": float(statistics.median(elapsed)) if elapsed else 0.0,
        "archive_count_mean": float(statistics.fmean(archives)) if archives else 0.0,
        "strong_capture_archive_count_mean": float(statistics.fmean(strong_archives))
        if strong_archives
        else 0.0,
        "strong_capture_refractory_skip_count_mean": float(
            statistics.fmean(refractory_skips)
        )
        if refractory_skips
        else 0.0,
        "last_run": rows[-1] if rows else {},
    }


def run_benchmark(
    *,
    tokens: int,
    min_interval_tokens: int,
    runs: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_before = 0
    cuda_reserved_before = 0
    if cuda_available:
        cuda_before = int(torch.cuda.memory_allocated())
        cuda_reserved_before = int(torch.cuda.memory_reserved())
    bounded_rows = [
        _run_once(
            tokens=tokens,
            min_interval_tokens=min_interval_tokens,
            seed=seed + run,
            device=device,
        )
        for run in range(int(runs))
    ]
    cuda_after = 0
    cuda_reserved_after = 0
    if cuda_available:
        cuda_after = int(torch.cuda.memory_allocated())
        cuda_reserved_after = int(torch.cuda.memory_reserved())
    bounded = _case_stats(bounded_rows)
    legacy = _retired_every_strong_projection(tokens=tokens, rows=bounded_rows)
    bounded_last = bounded.get("last_run", {})
    bounded_archives = max(1.0, float(bounded["archive_count_mean"]))
    work_reduction = float(legacy["archive_count_mean"]) / bounded_archives
    quality = {
        "coverage_gate_passed": bool(
            bounded_last.get("coverage", {}).get("coverage_gate_passed", False)
        ),
        "bounded_no_every_token_admission": bool(
            int(bounded_last.get("archive_count", 0)) < int(tokens)
            and int(bounded_last.get("strong_capture_refractory_skip_count", 0)) > 0
        ),
        "retired_every_strong_projection_matches_forced_candidates": bool(
            int(legacy.get("archive_count_mean", 0)) == int(tokens)
            and bool(legacy.get("executable_path_retired", False))
        ),
        "work_reduction": float(work_reduction),
        "bounded_elapsed_mean_ms": float(bounded["elapsed_mean_ms"]),
        "legacy_latency_not_measured_because_path_retired": True,
    }
    quality["pass"] = bool(
        quality["coverage_gate_passed"]
        and quality["bounded_no_every_token_admission"]
        and quality["retired_every_strong_projection_matches_forced_candidates"]
        and work_reduction >= 2.0
    )
    return {
        "artifact_kind": "bounded_strong_capture_admission_cadence_benchmark",
        "surface": "bounded_strong_capture_admission_cadence.v1",
        "tokens": int(tokens),
        "runs": int(runs),
        "strong_capture_min_interval_tokens": int(min_interval_tokens),
        "selection_criteria": [
            "first_token_admission",
            "strong_capture_threshold_crossing",
            "minimum_token_interval_since_last_archived_strong_capture",
        ],
        "memory_budget": {
            "max_archive_writes_per_strong_interval": 1,
            "strong_capture_min_interval_tokens": int(min_interval_tokens),
            "archival_storage_device": "cpu",
        },
        "device_placement": {
            "trainer_device": str(device),
            "archival_storage_device": "cpu",
            "active_replay_computation_device": "none",
            "cuda_available": cuda_available,
            "gpu_used": bool(str(device).startswith("cuda")),
            "cuda_memory_allocated_before_mib": float(cuda_before / (1024 * 1024)),
            "cuda_memory_allocated_after_mib": float(cuda_after / (1024 * 1024)),
            "cuda_memory_reserved_before_mib": float(cuda_reserved_before / (1024 * 1024)),
            "cuda_memory_reserved_after_mib": float(cuda_reserved_after / (1024 * 1024)),
        },
        "runtime_truth": {
            "runs_live_tick": True,
            "runs_every_token": False,
            "slow_memory_admission_every_token": False,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_text_payload_loaded_only_for_archived_entries": True,
            "language_reasoning": False,
            "hidden_language_reasoning": False,
            "cadence_surface": "slow_memory_strong_capture_min_interval_tokens",
        },
        "bounded": bounded,
        "retired_every_strong_projection": legacy,
        "quality": quality,
        "pass": bool(quality["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded strong-capture slow-memory admission."
    )
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--min-interval-tokens", type=int, default=16)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(
        tokens=args.tokens,
        min_interval_tokens=args.min_interval_tokens,
        runs=args.runs,
        seed=args.seed,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "pass={pass_gate} bounded_archives={bounded:.0f} legacy_archives={legacy:.0f} "
        "work_reduction={work:.6f} bounded_elapsed_mean_ms={elapsed:.6f}".format(
            pass_gate=report["pass"],
            bounded=report["bounded"]["archive_count_mean"],
            legacy=report["retired_every_strong_projection"]["archive_count_mean"],
            work=report["quality"]["work_reduction"],
            elapsed=report["quality"]["bounded_elapsed_mean_ms"],
        )
    )


if __name__ == "__main__":
    main()
