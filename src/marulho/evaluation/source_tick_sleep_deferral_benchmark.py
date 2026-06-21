from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from threading import RLock
from types import MethodType, SimpleNamespace
from typing import Any

import torch

from marulho.config.model_config import MarulhoConfig
from marulho.service.brain_runtime import BrainRuntime, BrainRuntimeDependencies
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


class _RuntimeState:
    def __init__(self) -> None:
        self.mutation_count = 0

    def mark_mutated(self) -> None:
        self.mutation_count += 1


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


def _build_trainer(*, seed: int, sleep_cost_ms: float) -> tuple[MarulhoTrainer, dict[str, Any], torch.Tensor]:
    torch.manual_seed(int(seed))
    config = _config()
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    trainer.token_count = int(config.deep_sleep_interval_tokens)
    pattern = torch.rand(config.input_dim, dtype=torch.float32)
    probe = _install_sleep_probe(trainer, sleep_cost_ms=sleep_cost_ms)
    return trainer, probe, pattern


def _brain_runtime(trainer: MarulhoTrainer, runtime_state: _RuntimeState) -> BrainRuntime:
    deps = BrainRuntimeDependencies(
        lock=RLock(),
        trainer=trainer,
        encoder=None,
        runtime_state=runtime_state,
        brain_config=lambda: {"tick_tokens": 1},
        runtime_control=lambda: SimpleNamespace(),
        runtime_sources=lambda: SimpleNamespace(),
        delayed_consequence=lambda: SimpleNamespace(),
        autonomy_planner=lambda: SimpleNamespace(),
        source_focus=lambda: SimpleNamespace(),
        interaction_pipeline=lambda: SimpleNamespace(
            recent_query_gaps=lambda: [],
            runtime_episode_traces=lambda: [],
        ),
        action_executor=lambda: SimpleNamespace(),
        replay_controller=lambda: SimpleNamespace(),
        concept_store=lambda: SimpleNamespace(snapshot=lambda limit=5: {}),
        geometric_curiosity=lambda: SimpleNamespace(summary=lambda: {}),
        runtime_environment_summary=lambda: {},
        huggingface_runtime_summary_locked=lambda: {},
        ingestion_runtime_summary_locked=lambda: {},
        multimodal_runtime_summary_locked=lambda: {},
        sensory_runtime_summary_locked=lambda _sensory: {},
        living_loop_snapshot_locked=lambda **_kwargs: {},
        maybe_mark_ingestion_warm_locked=lambda **_kwargs: None,
        maybe_mark_sensory_warm_locked=lambda **_kwargs: None,
        observe_runtime_concepts_locked=lambda **_kwargs: None,
        observe_runtime_concept_batch_locked=lambda **_kwargs: [],
        runtime_concept_callback_locked=lambda **_kwargs: None,
        run_real_sensory_episode_locked=lambda: None,
        record_brain_event_locked=lambda _event: None,
        build_brain_source_stream_locked=lambda _spec: iter(()),
        build_sensory_stream_locked=lambda _spec: iter(()),
    )
    return BrainRuntime(deps)


def _run_service_deferred_once(*, seed: int, sleep_cost_ms: float) -> dict[str, Any]:
    trainer, probe, pattern = _build_trainer(seed=seed, sleep_cost_ms=sleep_cost_ms)
    runtime_state = _RuntimeState()
    runtime = _brain_runtime(trainer, runtime_state)
    started = time.perf_counter()
    trained, metrics, _windows, observation = runtime._train_chunk_in_sub_batches(
        [("sleep-deferral-window", pattern)],
        stop_event=None,
        sub_batch_size=1,
        yield_seconds=0.0,
        concept_observation_due=False,
    )
    elapsed_ms = float((time.perf_counter() - started) * 1000.0)
    return {
        "elapsed_ms": elapsed_ms,
        "trained": int(trained),
        "sleep_probe_calls": list(probe["calls"]),
        "sleep_maintenance_deferred": int(metrics.get("sleep_maintenance_deferred", 0)),
        "sleep_triggered": int(metrics.get("sleep_triggered", 0)),
        "sleep_type": str(metrics.get("sleep_type", "none")),
        "mutation_count": int(runtime_state.mutation_count),
        "concept_observation_mode": str(observation.get("mode")),
    }


def _run_allowed_projection_once(*, seed: int, sleep_cost_ms: float) -> dict[str, Any]:
    trainer, probe, pattern = _build_trainer(seed=seed, sleep_cost_ms=sleep_cost_ms)
    started = time.perf_counter()
    metrics = trainer.train_step(
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
    service_rows = [
        _run_service_deferred_once(seed=seed + run, sleep_cost_ms=sleep_cost_ms)
        for run in range(int(runs))
    ]
    allowed_rows = [
        _run_allowed_projection_once(seed=seed + 10_000 + run, sleep_cost_ms=sleep_cost_ms)
        for run in range(int(runs))
    ]
    cuda_after = int(torch.cuda.memory_allocated()) if cuda_available else 0
    service = _stats(service_rows)
    allowed = _stats(allowed_rows)
    service_last = dict(service.get("last_run") or {})
    allowed_last = dict(allowed.get("last_run") or {})
    quality = {
        "service_tick_replay_deferred": bool(
            service_last.get("sleep_probe_calls") == []
            and int(service_last.get("sleep_maintenance_deferred", 0)) == 1
            and int(service_last.get("sleep_triggered", -1)) == 0
        ),
        "explicit_slow_path_remains_available": bool(
            allowed_last.get("sleep_probe_calls") == ["deep"]
            and int(allowed_last.get("sleep_triggered", 0)) == 1
            and str(allowed_last.get("sleep_type")) == "deep"
        ),
        "latency_reduction_mean_ms": float(
            max(0.0, float(allowed["elapsed_mean_ms"]) - float(service["elapsed_mean_ms"]))
        ),
    }
    quality["pass"] = bool(
        quality["service_tick_replay_deferred"]
        and quality["explicit_slow_path_remains_available"]
    )
    return {
        "artifact_kind": "source_tick_sleep_deferral_benchmark",
        "surface": "source_tick_sleep_replay_deferred.v1",
        "runs": int(runs),
        "selection_criteria": [
            "background_source_tick",
            "per_token_fallback_due_to_metrics_or_burst_unavailable",
            "deep_sleep_due",
        ],
        "memory_budget": {
            "live_tick_sleep_replay_executions": 0,
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
            "sleep_replay_execution_gate": False,
            "sleep_replay_deferred_count_visible": True,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "hidden_language_reasoning": False,
        },
        "service_deferred": service,
        "allowed_slow_path_projection": allowed,
        "quality": quality,
        "pass": bool(quality["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark service source-tick sleep replay deferral."
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
        "pass={pass_gate} service_sleep_calls={service_calls} "
        "allowed_sleep_calls={allowed_calls} service_mean_ms={service_ms:.6f} "
        "allowed_mean_ms={allowed_ms:.6f}".format(
            pass_gate=report["pass"],
            service_calls=len(
                report["service_deferred"]["last_run"]["sleep_probe_calls"]
            ),
            allowed_calls=len(
                report["allowed_slow_path_projection"]["last_run"][
                    "sleep_probe_calls"
                ]
            ),
            service_ms=report["service_deferred"]["elapsed_mean_ms"],
            allowed_ms=report["allowed_slow_path_projection"]["elapsed_mean_ms"],
        )
    )


if __name__ == "__main__":
    main()
