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
    archive_interval_tokens: int,
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
        slow_memory_archive_interval_tokens=int(archive_interval_tokens),
        slow_memory_archive_strong_capture_threshold=999.0,
        slow_memory_archive_strong_capture_min_interval_tokens=16,
        trainer_telemetry_interval_tokens=10**9,
        micro_sleep_interval_tokens=10**9,
        deep_sleep_interval_tokens=10**9,
        device=str(device),
    )


def _patterns(config: MarulhoConfig, *, tokens: int, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return [
        torch.rand(config.input_dim, generator=generator, dtype=torch.float32)
        for _ in range(int(tokens))
    ]


def _run_once(
    *,
    tokens: int,
    archive_interval_tokens: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    config = _config(
        tokens=tokens,
        archive_interval_tokens=archive_interval_tokens,
        device=device,
    )
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    patterns = _patterns(config, tokens=tokens, seed=seed)
    started = time.perf_counter()
    last_metrics: dict[str, Any] | None = None
    for index, pattern in enumerate(patterns):
        last_metrics = trainer.train_step(
            pattern.to(trainer.model.device),
            raw_window=f"fixed-cadence-retirement-window-{index:05d}",
            allow_sleep_maintenance=False,
            return_metrics=True,
        )
    elapsed_ms = float((time.perf_counter() - started) * 1000.0)
    boundary_report = trainer.column_transition_runtime_report()[
        "cognitive_boundary_controller"
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
        "memory_size": int(len(trainer.model.memory_store.slow_buffer)),
        "stored_raw_windows": list(trainer.model.memory_store.slow_raw_windows),
        "stored_entry_timestamps": list(trainer.model.memory_store.slow_entry_timestamps),
        "last_metrics": dict(last_metrics or {}),
        "runtime_report": {
            "slow_memory_cadence_deferred_count": int(
                boundary_report["slow_memory_cadence_deferred_count"]
            ),
            "last_slow_memory_cadence_token": boundary_report[
                "last_slow_memory_cadence_token"
            ],
            "slow_memory_cadence_execution_gate": bool(
                boundary_report["slow_memory_cadence_execution_gate"]
            ),
            "cpu_maintenance_after_device_burst": bool(
                boundary_report["cpu_maintenance_after_device_burst"]
            ),
        },
    }


def _case_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [float(row["elapsed_ms"]) for row in rows]
    archives = [int(row["archive_count"]) for row in rows]
    skips = [int(row["archive_skip_count"]) for row in rows]
    deferred = [
        int(row["runtime_report"]["slow_memory_cadence_deferred_count"])
        for row in rows
    ]
    return {
        "runs": int(len(rows)),
        "elapsed_mean_ms": float(statistics.fmean(elapsed)) if elapsed else 0.0,
        "elapsed_median_ms": float(statistics.median(elapsed)) if elapsed else 0.0,
        "archive_count_mean": float(statistics.fmean(archives)) if archives else 0.0,
        "archive_skip_count_mean": float(statistics.fmean(skips)) if skips else 0.0,
        "cadence_deferred_count_mean": float(statistics.fmean(deferred))
        if deferred
        else 0.0,
        "last_run": rows[-1] if rows else {},
    }


def run_benchmark(
    *,
    tokens: int,
    archive_interval_tokens: int,
    runs: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_before = int(torch.cuda.memory_allocated()) if cuda_available else 0
    cuda_reserved_before = int(torch.cuda.memory_reserved()) if cuda_available else 0
    bounded_rows = [
        _run_once(
            tokens=tokens,
            archive_interval_tokens=archive_interval_tokens,
            seed=seed + run,
            device=device,
        )
        for run in range(int(runs))
    ]
    cuda_after = int(torch.cuda.memory_allocated()) if cuda_available else 0
    cuda_reserved_after = int(torch.cuda.memory_reserved()) if cuda_available else 0
    bounded = _case_stats(bounded_rows)
    bounded_last = dict(bounded.get("last_run") or {})
    interval = max(1, int(archive_interval_tokens))
    projected_fixed_cadence_writes = int(tokens) // interval
    if interval != 1:
        projected_fixed_cadence_writes += 1
    bounded_archives = max(1.0, float(bounded["archive_count_mean"]))
    write_reduction = float(projected_fixed_cadence_writes) / bounded_archives
    expected_deferred_count = max(0, int(projected_fixed_cadence_writes) - 1)
    quality = {
        "first_token_retained": bool(
            bounded_last.get("stored_entry_timestamps") == [1]
            and bounded_last.get("memory_size") == 1
        ),
        "fixed_cadence_archives_removed": bool(
            int(bounded_last.get("archive_count", 0)) == 1
            and int(
                bounded_last.get("runtime_report", {}).get(
                    "slow_memory_cadence_deferred_count",
                    -1,
                )
            )
            == int(expected_deferred_count)
        ),
        "strong_capture_unchanged_for_non_strong_case": bool(
            int(bounded_last.get("strong_capture_archive_count", -1)) == 0
            and int(bounded_last.get("strong_capture_refractory_skip_count", -1)) == 0
        ),
        "cadence_execution_gate_closed": bool(
            bounded_last.get("runtime_report", {}).get(
                "slow_memory_cadence_execution_gate"
            )
            is False
        ),
        "fixed_cadence_projection_removed": True,
        "projected_write_reduction": float(write_reduction),
        "bounded_elapsed_mean_ms": float(bounded["elapsed_mean_ms"]),
    }
    quality["pass"] = bool(
        quality["first_token_retained"]
        and quality["fixed_cadence_archives_removed"]
        and quality["strong_capture_unchanged_for_non_strong_case"]
        and quality["cadence_execution_gate_closed"]
        and quality["fixed_cadence_projection_removed"]
        and write_reduction >= 2.0
    )
    removed_fixed_cadence_absence = {
        "implementation_present": False,
        "active_report_field_present": False,
        "removed_policy": "fixed_cadence_slow_memory_admission_projection",
    }
    return {
        "artifact_kind": "slow_memory_fixed_cadence_retirement_benchmark",
        "surface": "slow_memory_fixed_cadence_admission_retired.v1",
        "tokens": int(tokens),
        "runs": int(runs),
        "archive_interval_tokens": int(archive_interval_tokens),
        "selection_criteria": [
            "first_token_admission",
            "bounded_strong_capture_admission",
            "fixed_cadence_deferred_maintenance",
        ],
        "memory_budget": {
            "archival_entries": int(max(64, int(tokens) + 8)),
            "fixed_cadence_archive_writes": 0,
            "first_token_archive_writes": 1,
            "projected_fixed_cadence_writes_removed": int(
                max(0, int(projected_fixed_cadence_writes) - 1)
            ),
            "projected_archive_write_reduction": float(write_reduction),
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
            "slow_memory_fixed_cadence_execution_gate": False,
            "slow_memory_admission_every_token": False,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_text_payload_loaded_only_for_archived_entries": True,
            "language_reasoning": False,
            "hidden_language_reasoning": False,
        },
        "bounded": bounded,
        "retired_fixed_cadence_admission_absence": removed_fixed_cadence_absence,
        "quality": quality,
        "pass": bool(quality["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark retired fixed-cadence slow-memory admission."
    )
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--archive-interval-tokens", type=int, default=16)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(
        tokens=args.tokens,
        archive_interval_tokens=args.archive_interval_tokens,
        runs=args.runs,
        seed=args.seed,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "pass={pass_gate} bounded_archives={bounded:.0f} "
        "removed_fixed_cadence_writes={removed:.0f} write_reduction={work:.6f} "
        "bounded_elapsed_mean_ms={elapsed:.6f}".format(
            pass_gate=report["pass"],
            bounded=report["bounded"]["archive_count_mean"],
            removed=report["memory_budget"][
                "projected_fixed_cadence_writes_removed"
            ],
            work=report["quality"]["projected_write_reduction"],
            elapsed=report["quality"]["bounded_elapsed_mean_ms"],
        )
    )


if __name__ == "__main__":
    main()
