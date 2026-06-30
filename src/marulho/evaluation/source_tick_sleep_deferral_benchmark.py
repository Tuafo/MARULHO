from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from marulho.brain import MarulhoBrain
from marulho.config.model_config import MarulhoConfig
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


def _config() -> MarulhoConfig:
    return MarulhoConfig(
        n_columns=16,
        column_latent_dim=8,
        bootstrap_tokens=0,
        memory_capacity=32,
        slow_memory_start_tokens=0,
        slow_memory_archive_interval_tokens=10**9,
        micro_sleep_interval_tokens=10**9,
        deep_sleep_interval_tokens=1,
        deep_sleep_replay_steps=1,
        deep_sleep_candidate_pool=4,
        trainer_telemetry_interval_tokens=10**9,
        device="cpu",
    )


def _install_sleep_probe(trainer: MarulhoTrainer, *, sleep_cost_ms: float) -> dict[str, Any]:
    calls: list[str] = []

    def _sleep_replay(self: MarulhoTrainer, mode: str = "deep") -> int:
        calls.append(str(mode))
        if sleep_cost_ms > 0.0:
            time.sleep(float(sleep_cost_ms) / 1000.0)
        return 1

    trainer._sleep_replay = MethodType(_sleep_replay, trainer)  # type: ignore[method-assign]
    return {"calls": calls}


def _force_sequence_fallback(trainer: MarulhoTrainer) -> dict[str, Any]:
    burst_calls: list[int] = []

    def _train_text_burst(
        self: MarulhoTrainer,
        patterns: list[torch.Tensor],
        **_kwargs: Any,
    ) -> bool:
        burst_calls.append(len(patterns))
        return False

    trainer.train_text_burst = MethodType(_train_text_burst, trainer)  # type: ignore[method-assign]
    return {"burst_calls": burst_calls}


def _build_brain(*, seed: int, sleep_cost_ms: float) -> tuple[MarulhoBrain, dict[str, Any]]:
    torch.manual_seed(int(seed))
    config = _config()
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    trainer.token_count = int(config.deep_sleep_interval_tokens)
    probe = _install_sleep_probe(trainer, sleep_cost_ms=sleep_cost_ms)
    return MarulhoBrain.from_trainer(trainer), probe


def _feed_probe_text(brain: MarulhoBrain, text: str, *, source: str) -> int:
    accepted = int(brain.feed(text, source=source)["accepted_tokens"])
    if accepted <= 0:
        raise RuntimeError("probe text produced no brain source patterns")
    return accepted


def _run_brain_deferred_once(*, seed: int, sleep_cost_ms: float) -> dict[str, Any]:
    brain, probe = _build_brain(seed=seed, sleep_cost_ms=sleep_cost_ms)
    fallback_probe = _force_sequence_fallback(brain.trainer)
    accepted = _feed_probe_text(
        brain,
        "sleep deferral source tick probe",
        source="sleep-deferral-benchmark",
    )
    started = time.perf_counter()
    tick = brain.tick(
        tokens=1,
        quantum_tokens=1,
        source="sleep-deferral-benchmark",
        allow_sleep_maintenance=False,
    )
    elapsed_ms = float((time.perf_counter() - started) * 1000.0)
    trainer_report = dict(tick.get("trainer") or {})
    return {
        "elapsed_ms": elapsed_ms,
        "accepted_tokens": int(accepted),
        "trained": int(tick.get("trained_tokens", 0)),
        "burst_attempts": list(fallback_probe["burst_calls"]),
        "sleep_probe_calls": list(probe["calls"]),
        "sleep_maintenance_allowed": bool(
            trainer_report.get("sleep_maintenance_allowed", True)
        ),
        "fallback_train_step_count": int(
            trainer_report.get("fallback_train_step_count", 0) or 0
        ),
        "sleep_maintenance_deferred": int(
            trainer_report.get("fallback_sleep_maintenance_deferred_count", 0)
            or 0
        ),
        "trace_event": str((tick.get("trace") or {}).get("event")),
        "runtime_owner": "MarulhoBrain",
    }


def _run_brain_sequence_deferred_once(*, seed: int, sleep_cost_ms: float) -> dict[str, Any]:
    brain, probe = _build_brain(seed=seed, sleep_cost_ms=sleep_cost_ms)
    fallback_probe = _force_sequence_fallback(brain.trainer)
    accepted = _feed_probe_text(
        brain,
        "sleep deferral source sequence probe " * 3,
        source="sleep-deferral-sequence-benchmark",
    )
    tick_tokens = min(16, accepted)
    started = time.perf_counter()
    tick = brain.tick(
        tokens=tick_tokens,
        quantum_tokens=16,
        source="sleep-deferral-sequence-benchmark",
        allow_sleep_maintenance=False,
    )
    elapsed_ms = float((time.perf_counter() - started) * 1000.0)
    trainer_report = dict(tick.get("trainer") or {})
    return {
        "elapsed_ms": elapsed_ms,
        "accepted_tokens": int(accepted),
        "trained": int(tick.get("trained_tokens", 0)),
        "sleep_probe_calls": list(probe["calls"]),
        "sequence_burst_attempts": list(fallback_probe["burst_calls"]),
        "fallback_train_step_count": int(
            trainer_report.get("fallback_train_step_count", 0) or 0
        ),
        "sleep_maintenance_deferred": int(
            trainer_report.get("fallback_sleep_maintenance_deferred_count", 0)
            or 0
        ),
        "sleep_maintenance_allowed": bool(
            trainer_report.get("sleep_maintenance_allowed", True)
        ),
        "trace_event": str((tick.get("trace") or {}).get("event")),
        "runtime_owner": "MarulhoBrain",
    }


def _run_allowed_projection_once(*, seed: int, sleep_cost_ms: float) -> dict[str, Any]:
    brain, probe = _build_brain(seed=seed, sleep_cost_ms=sleep_cost_ms)
    pattern = torch.rand(brain.trainer.config.input_dim, dtype=torch.float32)
    started = time.perf_counter()
    metrics = brain.trainer.train_step(
        pattern,
        raw_window="sleep-allowed-projection-window",
        allow_sleep_maintenance=True,
        return_metrics=True,
    )
    elapsed_ms = float((time.perf_counter() - started) * 1000.0)
    return {
        "elapsed_ms": elapsed_ms,
        "sleep_probe_calls": list(probe["calls"]),
        "sleep_maintenance_deferred": int(metrics.get("sleep_maintenance_deferred", 0)),
        "sleep_triggered": int(metrics.get("sleep_triggered", 0)),
        "sleep_type": str(metrics.get("sleep_type", "none")),
    }


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [float(row["elapsed_ms"]) for row in rows]
    return {
        "runs": int(len(rows)),
        "elapsed_mean_ms": float(statistics.fmean(elapsed)) if elapsed else 0.0,
        "elapsed_median_ms": float(statistics.median(elapsed)) if elapsed else 0.0,
        "last_run": rows[-1] if rows else {},
    }


def run_benchmark(*, runs: int, seed: int, sleep_cost_ms: float) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_before = int(torch.cuda.memory_allocated()) if cuda_available else 0
    brain_rows = [
        _run_brain_deferred_once(seed=seed + run, sleep_cost_ms=sleep_cost_ms)
        for run in range(int(runs))
    ]
    brain_sequence_rows = [
        _run_brain_sequence_deferred_once(
            seed=seed + 5_000 + run,
            sleep_cost_ms=sleep_cost_ms,
        )
        for run in range(int(runs))
    ]
    allowed_rows = [
        _run_allowed_projection_once(seed=seed + 10_000 + run, sleep_cost_ms=sleep_cost_ms)
        for run in range(int(runs))
    ]
    cuda_after = int(torch.cuda.memory_allocated()) if cuda_available else 0
    brain = _stats(brain_rows)
    brain_sequence = _stats(brain_sequence_rows)
    allowed = _stats(allowed_rows)
    brain_last = dict(brain.get("last_run") or {})
    brain_sequence_last = dict(brain_sequence.get("last_run") or {})
    allowed_last = dict(allowed.get("last_run") or {})
    quality = {
        "brain_tick_replay_deferred": bool(
            brain_last.get("sleep_probe_calls") == []
            and int(brain_last.get("sleep_maintenance_deferred", 0)) > 0
            and int(brain_last.get("fallback_train_step_count", 0)) > 0
            and bool(brain_last.get("sleep_maintenance_allowed", True)) is False
            and str(brain_last.get("runtime_owner")) == "MarulhoBrain"
        ),
        "brain_sequence_tick_replay_deferred": bool(
            brain_sequence_last.get("sleep_probe_calls") == []
            and str(brain_sequence_last.get("runtime_owner")) == "MarulhoBrain"
            and not bool(
                brain_sequence_last.get("sleep_maintenance_allowed", True)
            )
            and int(
                brain_sequence_last.get(
                    "fallback_train_step_count",
                    0,
                )
            )
            > 0
            and int(
                brain_sequence_last.get(
                    "sleep_maintenance_deferred",
                    0,
                )
            )
            > 0
        ),
        "explicit_slow_path_remains_available": bool(
            allowed_last.get("sleep_probe_calls") == ["deep"]
            and int(allowed_last.get("sleep_triggered", 0)) == 1
            and str(allowed_last.get("sleep_type")) == "deep"
        ),
        "latency_reduction_mean_ms": float(
            max(0.0, float(allowed["elapsed_mean_ms"]) - float(brain["elapsed_mean_ms"]))
        ),
    }
    quality["pass"] = bool(
        quality["brain_tick_replay_deferred"]
        and quality["brain_sequence_tick_replay_deferred"]
        and quality["explicit_slow_path_remains_available"]
    )
    return {
        "artifact_kind": "source_tick_sleep_deferral_benchmark",
        "surface": "marulho_brain_source_tick_sleep_replay_deferred.v1",
        "runs": int(runs),
        "selection_criteria": [
            "marulho_brain_source_tick",
            "per_token_fallback_due_to_metrics_or_burst_unavailable",
            "brain_tick_delegates_to_training_text_sequence",
            "deep_sleep_due",
        ],
        "memory_budget": {
            "live_tick_sleep_replay_executions": 0,
            "brain_sequence_live_tick_sleep_replay_executions": 0,
            "explicit_slow_path_projection_replay_executions": 1,
            "archival_storage_device": "cpu",
        },
        "device_placement": {
            "trainer_device": "cpu",
            "archival_storage_device": "cpu",
            "active_replay_computation_device": "deferred_slow_path",
            "cuda_available": cuda_available,
            "cuda_memory_allocated_before_mib": float(cuda_before / (1024 * 1024)),
            "cuda_memory_allocated_after_mib": float(cuda_after / (1024 * 1024)),
        },
        "runtime_truth": {
            "runs_live_tick": True,
            "runs_every_token": False,
            "runtime_owner": "MarulhoBrain",
            "sleep_replay_execution_gate": False,
            "sleep_replay_deferred_count_visible": True,
            "sequence_fallback_sleep_replay_deferred_count_visible": True,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "hidden_language_reasoning": False,
        },
        "brain_deferred": brain,
        "brain_sequence_deferred": brain_sequence,
        "allowed_slow_path_projection": allowed,
        "quality": quality,
        "pass": bool(quality["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MarulhoBrain source-tick sleep replay deferral."
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--sleep-cost-ms", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(
        runs=args.runs,
        seed=args.seed,
        sleep_cost_ms=args.sleep_cost_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "pass={pass_gate} brain_sleep_calls={brain_calls} "
        "sequence_sleep_calls={sequence_calls} "
        "allowed_sleep_calls={allowed_calls} brain_mean_ms={brain_ms:.6f} "
        "sequence_mean_ms={sequence_ms:.6f} "
        "allowed_mean_ms={allowed_ms:.6f}".format(
            pass_gate=report["pass"],
            brain_calls=len(
                report["brain_deferred"]["last_run"]["sleep_probe_calls"]
            ),
            sequence_calls=len(
                report["brain_sequence_deferred"]["last_run"][
                    "sleep_probe_calls"
                ]
            ),
            allowed_calls=len(
                report["allowed_slow_path_projection"]["last_run"][
                    "sleep_probe_calls"
                ]
            ),
            brain_ms=report["brain_deferred"]["elapsed_mean_ms"],
            sequence_ms=report["brain_sequence_deferred"]["elapsed_mean_ms"],
            allowed_ms=report["allowed_slow_path_projection"]["elapsed_mean_ms"],
        )
    )


if __name__ == "__main__":
    main()
