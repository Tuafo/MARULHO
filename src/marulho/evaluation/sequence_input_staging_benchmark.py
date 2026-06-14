"""A/B gate for source-sequence CUDA graph input staging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.checkpointing import load_trainer_checkpoint
from marulho.training.column_transition_runtime import ColumnTransitionRuntime
from marulho.training.trainer import MarulhoTrainer


def _prepare_trainer(
    checkpoint: Path,
    *,
    sequence_staging_enabled: bool,
    host_truth_sync_interval_tokens: int,
) -> MarulhoTrainer:
    trainer, _metadata = load_trainer_checkpoint(checkpoint)
    trainer.config.predictive_dense_transition_mode = "inplace_triton"
    trainer.config.predictive_route_vote_mode = "cuda_graph_text"
    trainer.config.cuda_graph_quantum_input_staging = True
    trainer.config.cuda_graph_sequence_input_staging = bool(sequence_staging_enabled)
    trainer.config.cuda_graph_host_truth_sync_interval_tokens = max(
        1,
        int(host_truth_sync_interval_tokens),
    )
    trainer.config.micro_sleep_interval_tokens = 10**9
    trainer.config.deep_sleep_interval_tokens = 10**9
    trainer.pending_emergency_deep_sleep = False
    trainer._column_transition_runtime = ColumnTransitionRuntime(trainer)
    report = trainer.column_transition_runtime_report()
    graph = report.get("cuda_graph_route_transition")
    if not isinstance(graph, dict) or not graph.get("active"):
        raise RuntimeError(f"persistent CUDA graph unavailable: {graph}")
    trainer._benchmark_transition_executor = (
        "sequence_input_staging"
        if sequence_staging_enabled
        else "per_quantum_input_staging"
    )
    return trainer


def _pattern_batches(
    trainer: MarulhoTrainer,
    *,
    seed: int,
    sequence_count: int,
    sequence_tokens: int,
    warmup_sequences: int,
) -> list[list[torch.Tensor]]:
    torch.manual_seed(int(seed))
    total_sequences = int(warmup_sequences) + int(sequence_count)
    return [
        [
            torch.rand(
                trainer.config.input_dim,
                device=trainer.model.device,
            )
            for _ in range(int(sequence_tokens))
        ]
        for _ in range(total_sequences)
    ]


def _metric_subset(report: dict[str, Any]) -> dict[str, int]:
    graph = dict(report.get("cuda_graph_route_transition") or {})
    return {
        "text_sequence_input_stage_count": int(
            report.get("text_sequence_input_stage_count", 0) or 0
        ),
        "text_sequence_input_staged_token_count": int(
            report.get("text_sequence_input_staged_token_count", 0) or 0
        ),
        "text_sequence_input_stage_skip_count": int(
            report.get("text_sequence_input_stage_skip_count", 0) or 0
        ),
        "text_burst_execution_count": int(
            report.get("text_burst_execution_count", 0) or 0
        ),
        "text_burst_fallback_count": int(
            report.get("text_burst_fallback_count", 0) or 0
        ),
        "graph_quantum_input_stage_count": int(
            graph.get("quantum_input_stage_count", 0) or 0
        ),
        "graph_quantum_input_staged_token_count": int(
            graph.get("quantum_input_staged_token_count", 0) or 0
        ),
        "graph_quantum_input_reuse_count": int(
            graph.get("quantum_input_reuse_count", 0) or 0
        ),
        "graph_replay_count": int(graph.get("replay_count", 0) or 0),
        "graph_failure_count": int(graph.get("failure_count", 0) or 0),
        "graph_burst_failure_count": int(
            graph.get("burst_replay_failure_count", 0) or 0
        ),
    }


def _metric_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        key: int(after.get(key, 0) - before.get(key, 0))
        for key in sorted(set(before) | set(after))
    }


def _run_arm(
    checkpoint: Path,
    *,
    name: str,
    sequence_staging_enabled: bool,
    sequence_count: int,
    sequence_tokens: int,
    warmup_sequences: int,
    quantum_tokens: int,
    host_truth_sync_interval_tokens: int,
    seed: int,
    profile_trainer_stages: bool,
    sync_mode: str,
) -> dict[str, Any]:
    trainer = _prepare_trainer(
        checkpoint,
        sequence_staging_enabled=sequence_staging_enabled,
        host_truth_sync_interval_tokens=host_truth_sync_interval_tokens,
    )
    batches = _pattern_batches(
        trainer,
        seed=seed,
        sequence_count=sequence_count,
        sequence_tokens=sequence_tokens,
        warmup_sequences=warmup_sequences,
    )
    warm_pattern = torch.rand(trainer.config.input_dim, device=trainer.model.device)
    trainer.train_step(
        warm_pattern,
        raw_window=f"{name} sequence staging warmup",
        allow_sleep_maintenance=False,
        return_metrics=False,
    )
    for warmup_index in range(int(warmup_sequences)):
        trainer.train_text_sequence(
            batches[warmup_index],
            raw_windows=[
                f"{name} warmup {warmup_index}:{token_index}"
                for token_index in range(int(sequence_tokens))
            ],
            quantum_tokens=int(quantum_tokens),
            metric_indices=set(),
        )
    if trainer.model.device.type == "cuda":
        torch.cuda.synchronize()
    before = _metric_subset(trainer.column_transition_runtime_report())
    if profile_trainer_stages:
        trainer.enable_train_step_profile(reset=True)
    started_ns = time.perf_counter_ns()
    for sequence_index in range(int(sequence_count)):
        batch = batches[int(warmup_sequences) + sequence_index]
        trainer.train_text_sequence(
            batch,
            raw_windows=[
                f"{name} measured {sequence_index}:{token_index}"
                for token_index in range(int(sequence_tokens))
            ],
            quantum_tokens=int(quantum_tokens),
            metric_indices=set(),
        )
        if trainer.model.device.type == "cuda" and sync_mode == "sequence":
            torch.cuda.synchronize()
    if trainer.model.device.type == "cuda" and sync_mode == "window":
        torch.cuda.synchronize()
    elapsed_s = (time.perf_counter_ns() - started_ns) / 1e9
    trainer_stage_profile: dict[str, Any] | None = None
    if profile_trainer_stages:
        trainer_stage_profile = dict(trainer.train_step_profile_report())
        trainer_stage_profile["scope"] = "measured_train_text_sequence_only"
        trainer.disable_train_step_profile()
    after_report = trainer.column_transition_runtime_report()
    after = _metric_subset(after_report)
    token_count = int(sequence_count) * int(sequence_tokens)
    arm = {
        "name": str(name),
        "sequence_input_staging_enabled": bool(sequence_staging_enabled),
        "sequence_count": int(sequence_count),
        "sequence_tokens": int(sequence_tokens),
        "tokens": int(token_count),
        "elapsed_seconds": float(elapsed_s),
        "tokens_per_second": float(token_count / max(elapsed_s, 1e-9)),
        "metric_delta": _metric_delta(before, after),
        "column_transition_runtime": after_report,
        "cuda_memory": (
            {
                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            }
            if trainer.model.device.type == "cuda"
            else {}
        ),
    }
    if trainer_stage_profile is not None:
        arm["trainer_stage_profile"] = trainer_stage_profile
    return arm


def run_sequence_input_staging_ab(
    checkpoint: Path,
    *,
    output_path: Path,
    sequence_count: int = 64,
    sequence_tokens: int = 128,
    warmup_sequences: int = 4,
    quantum_tokens: int = 16,
    host_truth_sync_interval_tokens: int = 32,
    seed: int = 20260614,
    profile_trainer_stages: bool = False,
    sync_mode: str = "window",
) -> dict[str, Any]:
    arm_specs = (
        ("per_quantum_a", False),
        ("sequence_a", True),
        ("sequence_b", True),
        ("per_quantum_b", False),
    )
    arms = [
        _run_arm(
            checkpoint,
            name=name,
            sequence_staging_enabled=enabled,
            sequence_count=sequence_count,
            sequence_tokens=sequence_tokens,
            warmup_sequences=warmup_sequences,
            quantum_tokens=quantum_tokens,
            host_truth_sync_interval_tokens=host_truth_sync_interval_tokens,
            seed=seed,
            profile_trainer_stages=profile_trainer_stages,
            sync_mode=sync_mode,
        )
        for name, enabled in arm_specs
    ]
    per_quantum = [
        float(arm["tokens_per_second"])
        for arm in arms
        if not bool(arm["sequence_input_staging_enabled"])
    ]
    sequence = [
        float(arm["tokens_per_second"])
        for arm in arms
        if bool(arm["sequence_input_staging_enabled"])
    ]
    per_quantum_mean = float(statistics.fmean(per_quantum))
    sequence_mean = float(statistics.fmean(sequence))
    report = {
        "surface": "sequence_input_staging_ab.v1",
        "checkpoint": str(checkpoint),
        "scope": "training_owned_text_sequence_with_persistent_cuda_graph",
        "claim_boundary": (
            "compares one fixed-address CUDA input-ring stage per source sequence "
            "against the retained per-quantum input staging path; token order, "
            "8-token burst execution, host-truth cadence, and SNN transition math "
            "remain unchanged"
        ),
        "sequence_count_per_arm": int(sequence_count),
        "sequence_tokens": int(sequence_tokens),
        "warmup_sequences_per_arm": int(warmup_sequences),
        "quantum_tokens": int(quantum_tokens),
        "host_truth_sync_interval_tokens": int(host_truth_sync_interval_tokens),
        "sync_mode": str(sync_mode),
        "profile_trainer_stages": bool(profile_trainer_stages),
        "arms": arms,
        "per_quantum_mean_tokens_per_second": per_quantum_mean,
        "sequence_mean_tokens_per_second": sequence_mean,
        "speedup": sequence_mean / max(per_quantum_mean, 1e-9),
        "success": bool(sequence_mean > per_quantum_mean),
    }
    write_json_report_with_readme(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-count", type=int, default=64)
    parser.add_argument("--sequence-tokens", type=int, default=128)
    parser.add_argument("--warmup-sequences", type=int, default=4)
    parser.add_argument("--quantum-tokens", type=int, default=16)
    parser.add_argument("--host-truth-sync-interval-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--profile-trainer-stages", action="store_true")
    parser.add_argument(
        "--sync-mode",
        choices=("sequence", "window"),
        default="window",
    )
    args = parser.parse_args()
    report = run_sequence_input_staging_ab(
        args.checkpoint,
        output_path=args.output,
        sequence_count=args.sequence_count,
        sequence_tokens=args.sequence_tokens,
        warmup_sequences=args.warmup_sequences,
        quantum_tokens=args.quantum_tokens,
        host_truth_sync_interval_tokens=args.host_truth_sync_interval_tokens,
        seed=args.seed,
        profile_trainer_stages=args.profile_trainer_stages,
        sync_mode=args.sync_mode,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
