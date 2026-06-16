"""Long full-warm stress gate for the continuous Terminus CUDA runtime."""

from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Iterable, Mapping

from marulho.evaluation.continuous_runtime_quantum_benchmark import (
    DEFAULT_SOURCE_TEXT,
    _wait_for_full_warm,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.service.brain_runtime import DEFAULT_EXECUTION_QUANTUM_TOKENS
from marulho.service.manager import MarulhoServiceManager


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    rank = min(
        len(ordered) - 1,
        max(0, int(round((float(percentile) / 100.0) * (len(ordered) - 1)))),
    )
    return float(ordered[rank])


def _stats(values: Iterable[float]) -> dict[str, float | None]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(materialized),
        "min": float(min(materialized)),
        "mean": float(statistics.fmean(materialized)),
        "median": float(statistics.median(materialized)),
        "p95": _percentile(materialized, 95.0),
        "max": float(max(materialized)),
    }


def _tick_event_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        event.get("timestamp"),
        event.get("token_delta"),
        event.get("tick_duration_ms"),
        tuple(sorted(dict(event.get("stage_timings_ms") or {}).items())),
    )


def _collect_tick_events(
    snapshot: Mapping[str, Any],
    *,
    seen_keys: set[tuple[Any, ...]],
    events: list[dict[str, Any]],
) -> None:
    for event in snapshot.get("recent_events") or ():
        if not isinstance(event, Mapping):
            continue
        if event.get("type") != "tick":
            continue
        key = _tick_event_key(event)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        events.append(dict(event))


def _summarize_tick_events(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    tick_events = [dict(event) for event in events]
    token_deltas = [
        int(event.get("token_delta") or 0)
        for event in tick_events
        if int(event.get("token_delta") or 0) > 0
    ]
    durations = [
        float(event.get("tick_duration_ms") or 0.0)
        for event in tick_events
        if event.get("tick_duration_ms") is not None
    ]
    stage_values: dict[str, list[float]] = {}
    stage_token_totals: dict[str, float] = {}
    concept_modes: dict[str, int] = {}
    concept_due_count = 0
    concept_observation_attempts = 0
    concept_observation_count = 0
    concept_skipped_attempts = 0
    observed_tokens = sum(token_deltas)
    for event in tick_events:
        token_delta = max(1, int(event.get("token_delta") or 0))
        source = event.get("source")
        if isinstance(source, Mapping):
            concept = source.get("concept_observation")
            if isinstance(concept, Mapping):
                mode = str(concept.get("mode") or "unknown")
                concept_modes[mode] = concept_modes.get(mode, 0) + 1
                if bool(concept.get("tick_due")):
                    concept_due_count += 1
                concept_observation_attempts += int(
                    concept.get("attempts") or 0
                )
                concept_observation_count += int(
                    concept.get("observations") or 0
                )
                concept_skipped_attempts += int(
                    concept.get("skipped_attempts") or 0
                )
        timings = dict(event.get("stage_timings_ms") or {})
        for name, value in timings.items():
            measured = float(value)
            stage_values.setdefault(str(name), []).append(measured)
            stage_token_totals[str(name)] = (
                stage_token_totals.get(str(name), 0.0) + measured / token_delta
            )

    stages = {
        name: {
            **_stats(values),
            "mean_ms_per_token": float(
                stage_token_totals.get(name, 0.0) / max(len(values), 1)
            ),
        }
        for name, values in sorted(stage_values.items())
    }
    top_stage_mean_ms_per_token = sorted(
        (
            {
                "stage": name,
                "mean_ms_per_token": float(summary["mean_ms_per_token"]),
                "mean_ms_per_tick": summary["mean"],
                "p95_ms_per_tick": summary["p95"],
            }
            for name, summary in stages.items()
            if summary["mean"] is not None
        ),
        key=lambda item: float(item["mean_ms_per_token"]),
        reverse=True,
    )[:8]
    return {
        "tick_event_count": len(tick_events),
        "observed_tick_tokens": int(observed_tokens),
        "tick_token_delta": _stats(token_deltas),
        "tick_duration_ms": _stats(durations),
        "concept_observation": {
            "modes": dict(sorted(concept_modes.items())),
            "tick_due_count": int(concept_due_count),
            "tick_skip_count": int(
                concept_modes.get("cadenced_tick_skip", 0)
            ),
            "attempts": int(concept_observation_attempts),
            "observations": int(concept_observation_count),
            "skipped_attempts": int(concept_skipped_attempts),
        },
        "stages": stages,
        "top_stage_mean_ms_per_token": top_stage_mean_ms_per_token,
    }


def _source_text_for_target(target_tokens: int) -> str:
    repeat_count = max(8, int(target_tokens // 8))
    return DEFAULT_SOURCE_TEXT * repeat_count


def _flush_source_cache_writes(manager: MarulhoServiceManager) -> dict[str, Any]:
    flush = getattr(manager._runtime_sources, "flush_brain_runtime_cache_writes")
    started = time.perf_counter()
    flush()
    latency_ms = (time.perf_counter() - started) * 1000.0
    with manager._lock:
        snapshot = manager._brain_runtime_snapshot_locked()
    return {
        "source_cache_flush_latency_ms": float(latency_ms),
        "ingestion_after_flush": dict(snapshot.get("ingestion") or {}),
    }


def _runtime_event_history_limit(manager: MarulhoServiceManager) -> int:
    history = getattr(manager._runtime_state, "_brain_event_history", None)
    maxlen = getattr(history, "maxlen", None)
    return max(1, int(maxlen or 1))


def _ensure_runtime_event_history_capacity(
    manager: MarulhoServiceManager,
    required_events: int,
) -> dict[str, Any]:
    history = getattr(manager._runtime_state, "_brain_event_history", None)
    before_limit = max(1, int(getattr(history, "maxlen", None) or 1))
    required_limit = max(1, int(required_events))
    if before_limit >= required_limit or not isinstance(history, deque):
        return {
            "extended": False,
            "before_limit": int(before_limit),
            "after_limit": int(before_limit),
            "required_events": int(required_limit),
        }
    manager._runtime_state._brain_event_history = deque(
        list(history),
        maxlen=required_limit,
    )
    return {
        "extended": True,
        "before_limit": int(before_limit),
        "after_limit": int(required_limit),
        "required_events": int(required_limit),
    }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text or text.lower() in {"n/a", "nan", "[not supported]"}:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    number = _float_or_none(value)
    if number is None:
        return None
    return int(number)


def _parse_nvidia_smi_gpu_row(row: str) -> dict[str, Any]:
    try:
        values = [cell.strip() for cell in next(csv.reader([row]))]
    except (csv.Error, StopIteration):
        return {
            "available": False,
            "sample_source": "nvidia-smi",
            "error": "failed_to_parse_nvidia_smi_csv",
        }
    if len(values) < 9:
        return {
            "available": False,
            "sample_source": "nvidia-smi",
            "error": "missing_nvidia_smi_fields",
            "field_count": len(values),
        }
    return {
        "available": True,
        "sample_source": "nvidia-smi",
        "name": values[0],
        "pstate": values[1],
        "graphics_clock_mhz": _int_or_none(values[2]),
        "memory_clock_mhz": _int_or_none(values[3]),
        "power_draw_w": _float_or_none(values[4]),
        "gpu_utilization_percent": _float_or_none(values[5]),
        "memory_utilization_percent": _float_or_none(values[6]),
        "temperature_c": _float_or_none(values[7]),
        "memory_used_mib": _int_or_none(values[8]),
    }


def _collect_nvidia_smi_gpu_snapshot() -> dict[str, Any]:
    query = (
        "name,pstate,clocks.current.graphics,clocks.current.memory,"
        "power.draw,utilization.gpu,utilization.memory,temperature.gpu,"
        "memory.used"
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "sample_source": "nvidia-smi",
            "error": str(exc),
        }
    if result.returncode != 0:
        return {
            "available": False,
            "sample_source": "nvidia-smi",
            "error": (result.stderr or result.stdout).strip()[:500],
            "returncode": int(result.returncode),
        }
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        return {
            "available": False,
            "sample_source": "nvidia-smi",
            "error": "empty_nvidia_smi_output",
        }
    snapshot = _parse_nvidia_smi_gpu_row(rows[0])
    if len(rows) > 1:
        snapshot["ignored_gpu_rows"] = len(rows) - 1
    return snapshot


def _collect_windows_cpu_snapshot() -> dict[str, Any]:
    script = (
        "$cpu = Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor | "
        "Where-Object { $_.Name -eq '_Total' } | Select-Object -First 1 "
        "Name,PercentProcessorTime; "
        "$sys = Get-CimInstance Win32_PerfFormattedData_PerfOS_System | "
        "Select-Object -First 1 ProcessorQueueLength,Threads,Processes; "
        "[pscustomobject]@{"
        "available=$true;"
        "percent_processor_time=[double]$cpu.PercentProcessorTime;"
        "processor_queue_length=[double]$sys.ProcessorQueueLength;"
        "threads=[int]$sys.Threads;"
        "processes=[int]$sys.Processes"
        "} | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "sample_source": "windows_perf_counters",
            "error": str(exc),
        }
    if result.returncode != 0:
        return {
            "available": False,
            "sample_source": "windows_perf_counters",
            "error": (result.stderr or result.stdout).strip()[:500],
            "returncode": int(result.returncode),
        }
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "available": False,
            "sample_source": "windows_perf_counters",
            "error": f"failed_to_parse_cpu_json: {exc}",
        }
    return {
        "available": bool(data.get("available", True)),
        "sample_source": "windows_perf_counters",
        "percent_processor_time": _float_or_none(
            data.get("percent_processor_time")
        ),
        "processor_queue_length": _float_or_none(
            data.get("processor_queue_length")
        ),
        "threads": _int_or_none(data.get("threads")),
        "processes": _int_or_none(data.get("processes")),
    }


def _collect_cpu_snapshot() -> dict[str, Any]:
    if platform.system().lower() == "windows":
        return _collect_windows_cpu_snapshot()
    load_average: tuple[float, float, float] | None = None
    try:
        load_average = tuple(float(value) for value in os.getloadavg())
    except (AttributeError, OSError):
        pass
    return {
        "available": load_average is not None,
        "sample_source": "os.getloadavg",
        "cpu_count": os.cpu_count(),
        "load_average_1m": None if load_average is None else load_average[0],
        "load_average_5m": None if load_average is None else load_average[1],
        "load_average_15m": None if load_average is None else load_average[2],
    }


def _collect_velocity_environment_snapshot() -> dict[str, Any]:
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "cpu": _collect_cpu_snapshot(),
        "gpu": _collect_nvidia_smi_gpu_snapshot(),
    }


def _available_metric(
    snapshot: Mapping[str, Any] | None,
    section: str,
    key: str,
) -> float | None:
    if not isinstance(snapshot, Mapping):
        return None
    payload = snapshot.get(section)
    if not isinstance(payload, Mapping):
        return None
    if not bool(payload.get("available")):
        return None
    return _float_or_none(payload.get(key))


def _summarize_velocity_environment(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> dict[str, Any]:
    cpu_values = [
        value
        for value in (
            _available_metric(before, "cpu", "percent_processor_time"),
            _available_metric(after, "cpu", "percent_processor_time"),
        )
        if value is not None
    ]
    gpu_values = [
        value
        for value in (
            _available_metric(before, "gpu", "gpu_utilization_percent"),
            _available_metric(after, "gpu", "gpu_utilization_percent"),
        )
        if value is not None
    ]
    memory_values = [
        value
        for value in (
            _available_metric(before, "gpu", "memory_utilization_percent"),
            _available_metric(after, "gpu", "memory_utilization_percent"),
        )
        if value is not None
    ]
    cpu_threshold = 90.0
    gpu_threshold = 20.0
    cpu_busy = any(value >= cpu_threshold for value in cpu_values)
    gpu_busy = any(value >= gpu_threshold for value in gpu_values)
    unknown = not cpu_values and not gpu_values
    verdict = (
        "unknown"
        if unknown
        else "contention_observed"
        if cpu_busy or gpu_busy
        else "not_observed"
    )
    return {
        "surface": "velocity_environment.v1",
        "not_hot_path": True,
        "claim_boundary": (
            "slow-path benchmark run-condition evidence only; it does not "
            "measure or change cognitive execution"
        ),
        "contention": {
            "verdict": verdict,
            "cpu_busy": bool(cpu_busy),
            "gpu_busy": bool(gpu_busy),
            "thresholds": {
                "cpu_percent_busy": cpu_threshold,
                "gpu_graphics_utilization_percent_busy": gpu_threshold,
            },
            "max_cpu_percent": max(cpu_values) if cpu_values else None,
            "max_gpu_utilization_percent": max(gpu_values) if gpu_values else None,
            "max_gpu_memory_utilization_percent": (
                max(memory_values) if memory_values else None
            ),
        },
        "before": dict(before or {}),
        "after": dict(after or {}),
    }


def run_continuous_runtime_stress(
    checkpoint: Path,
    *,
    output_path: Path,
    target_tokens: int = 1024,
    tick_tokens: int = 128,
    quantum_tokens: int = DEFAULT_EXECUTION_QUANTUM_TOKENS,
    source_concept_observation_tick_interval: int = 4,
    timeout_seconds: float = 60.0,
    sample_interval_seconds: float = 0.02,
    profile_trainer_stages: bool = False,
    host_truth_sync_interval_tokens: int | None = None,
    native_burst_replay: bool | None = None,
    sequence_executor: str | None = None,
) -> dict[str, Any]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if tick_tokens <= 0:
        raise ValueError("tick_tokens must be positive")
    if quantum_tokens <= 0:
        raise ValueError("quantum_tokens must be positive")
    if source_concept_observation_tick_interval <= 0:
        raise ValueError("source_concept_observation_tick_interval must be positive")
    if (
        host_truth_sync_interval_tokens is not None
        and int(host_truth_sync_interval_tokens) <= 0
    ):
        raise ValueError("host_truth_sync_interval_tokens must be positive")
    if sequence_executor is not None:
        normalized_sequence_executor = (
            str(sequence_executor).strip().lower().replace("-", "_")
        )
        if normalized_sequence_executor not in {
            "native_repeated_child_graph",
            "repeated_child",
            "native8",
            "default",
            "conditional_while",
            "cuda_graph_conditional_while",
        }:
            raise ValueError(
                "sequence_executor must be native_repeated_child_graph "
                "or conditional_while"
            )
    else:
        normalized_sequence_executor = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_root = output_path.parent / output_path.stem
    run_root.mkdir(parents=True, exist_ok=True)
    source_path = run_root / "continuous-runtime-stress-source.txt"
    source_path.write_text(_source_text_for_target(target_tokens), encoding="utf-8")
    previous_native_replay_env = os.environ.get(
        "MARULHO_CUDA_GRAPH_NATIVE_BURST_REPLAY"
    )
    previous_sequence_executor_env = os.environ.get(
        "MARULHO_CUDA_GRAPH_SEQUENCE_EXECUTOR"
    )
    if native_burst_replay is not None:
        os.environ["MARULHO_CUDA_GRAPH_NATIVE_BURST_REPLAY"] = (
            "1" if bool(native_burst_replay) else "0"
        )
    if normalized_sequence_executor is not None:
        os.environ["MARULHO_CUDA_GRAPH_SEQUENCE_EXECUTOR"] = (
            normalized_sequence_executor
        )
    manager = MarulhoServiceManager(
        checkpoint,
        trace_dir=run_root / "traces",
        env_root=run_root,
    )
    runtime = manager.runtime_facade
    seen_events: set[tuple[Any, ...]] = set()
    tick_events: list[dict[str, Any]] = []
    config_overrides: dict[str, Any] = {}
    try:
        if host_truth_sync_interval_tokens is not None:
            with manager._lock:
                previous_interval = int(
                    manager._trainer.config.cuda_graph_host_truth_sync_interval_tokens
                )
                manager._trainer.config.cuda_graph_host_truth_sync_interval_tokens = int(
                    host_truth_sync_interval_tokens
                )
            config_overrides["cuda_graph_host_truth_sync_interval_tokens"] = {
                "from": previous_interval,
                "to": int(host_truth_sync_interval_tokens),
            }
        if native_burst_replay is not None:
            with manager._lock:
                previous_native_replay = bool(
                    manager._trainer.config.cuda_graph_native_burst_replay
                )
                manager._trainer.config.cuda_graph_native_burst_replay = bool(
                    native_burst_replay
                )
            config_overrides["cuda_graph_native_burst_replay"] = {
                "from": previous_native_replay,
                "to": bool(native_burst_replay),
            }
        if normalized_sequence_executor is not None:
            with manager._lock:
                previous_sequence_executor = str(
                    manager._trainer.config.cuda_graph_sequence_executor
                )
                manager._trainer.config.cuda_graph_sequence_executor = (
                    normalized_sequence_executor
                )
            config_overrides["cuda_graph_sequence_executor"] = {
                "from": previous_sequence_executor,
                "to": normalized_sequence_executor,
            }
        runtime.configure_terminus(
            source_bank=[
                {
                    "name": "continuous_runtime_stress_source",
                    "source": str(source_path),
                    "source_type": "file",
                }
            ],
            tick_tokens=int(tick_tokens),
            source_concept_observation_tick_interval=int(
                source_concept_observation_tick_interval
            ),
            sleep_interval_seconds=0.01,
            execution_quantum_tokens=int(quantum_tokens),
            execution_yield_seconds=0.0,
            repeat_sources=True,
            ingestion={
                "enabled": True,
                "queue_target_tokens": max(int(tick_tokens), int(target_tokens)),
                "prewarm_on_startup": True,
                "prewarm_max_seconds": max(1.0, float(timeout_seconds)),
            },
        )
        warm_snapshot = _wait_for_full_warm(
            manager,
            timeout_seconds=timeout_seconds,
        )
        warm_ingestion = dict(warm_snapshot.get("ingestion") or {})
        if not bool(warm_ingestion.get("full_warm_ready")):
            velocity_environment = _summarize_velocity_environment(
                _collect_velocity_environment_snapshot(),
                None,
            )
            report = {
                "surface": "continuous_runtime_stress.v1",
                "success": False,
                "failure_reason": "source_queue_not_fully_warm",
                "checkpoint": str(checkpoint),
                "run_root": str(run_root),
                "source_path": str(source_path),
                "target_tokens": int(target_tokens),
                "tick_tokens": int(tick_tokens),
                "quantum_tokens": int(quantum_tokens),
                "source_concept_observation_tick_interval": int(
                    source_concept_observation_tick_interval
                ),
                "config_overrides": dict(config_overrides),
                "trainer_config": {
                    "slow_memory_archive_interval_tokens": int(
                        manager._trainer.config.slow_memory_archive_interval_tokens
                    ),
                    "trainer_telemetry_interval_tokens": int(
                        manager._trainer.config.trainer_telemetry_interval_tokens
                    ),
                    "cuda_graph_host_truth_sync_interval_tokens": int(
                        manager._trainer.config.cuda_graph_host_truth_sync_interval_tokens
                    ),
                    "cuda_graph_native_burst_replay": bool(
                        manager._trainer.config.cuda_graph_native_burst_replay
                    ),
                    "cuda_graph_native_burst_tokens": int(
                        manager._trainer.config.cuda_graph_native_burst_tokens
                    ),
                    "cuda_graph_sequence_executor": str(
                        manager._trainer.config.cuda_graph_sequence_executor
                    ),
                    "cuda_graph_sequence_loop_tokens": int(
                        manager._trainer.config.cuda_graph_sequence_loop_tokens
                    ),
                },
                "checkpoint_metadata": {
                    "config_migrations": list(
                        manager._metadata.get("config_migrations") or []
                    ),
                    "hot_path_config_defaults_revision": manager._metadata.get(
                        "hot_path_config_defaults_revision"
                    ),
                },
                "timeout_seconds": float(timeout_seconds),
                "warm_ingestion": warm_ingestion,
                "velocity_environment": velocity_environment,
            }
            write_json_report_with_readme(output_path, report)
            return report

        velocity_environment_before = _collect_velocity_environment_snapshot()
        with manager._lock:
            start_token = int(manager._trainer.token_count)
            if bool(profile_trainer_stages):
                manager._trainer.enable_train_step_profile(reset=True)
        runtime.start_terminus()
        started = time.perf_counter()
        deadline = started + max(0.1, float(timeout_seconds))
        current_token = start_token
        expected_tick_count = max(
            1,
            (int(target_tokens) + int(tick_tokens) - 1) // int(tick_tokens),
        )
        event_history_capacity = _ensure_runtime_event_history_capacity(
            manager,
            expected_tick_count + 16,
        )
        event_history_limit = _runtime_event_history_limit(manager)
        poll_snapshots = expected_tick_count > event_history_limit
        poll_snapshot_count = 0
        while time.perf_counter() < deadline:
            with manager._lock:
                current_token = int(manager._trainer.token_count)
                if poll_snapshots:
                    snapshot = manager._brain_runtime_snapshot_locked()
                else:
                    snapshot = None
            if snapshot is not None:
                poll_snapshot_count += 1
                _collect_tick_events(
                    snapshot,
                    seen_keys=seen_events,
                    events=tick_events,
                )
            if current_token - start_token >= int(target_tokens):
                break
            time.sleep(max(0.001, float(sample_interval_seconds)))
        elapsed_seconds = time.perf_counter() - started
        stop_started = time.perf_counter()
        runtime.stop_terminus()
        stop_latency_ms = (time.perf_counter() - stop_started) * 1000.0
        velocity_environment_after = _collect_velocity_environment_snapshot()
        with manager._lock:
            final_token = int(manager._trainer.token_count)
            final_snapshot = manager._brain_runtime_snapshot_locked()
            _collect_tick_events(
                final_snapshot,
                seen_keys=seen_events,
                events=tick_events,
            )
            transition_report = manager._trainer.column_transition_runtime_report()
            device_report = manager._trainer.config.device_report()
            trainer_stage_profile = (
                manager._trainer.train_step_profile_report()
                if bool(profile_trainer_stages)
                else None
            )
            if bool(profile_trainer_stages):
                manager._trainer.disable_train_step_profile()
        cache_flush = _flush_source_cache_writes(manager)
        token_delta = max(0, final_token - start_token)
        event_summary = _summarize_tick_events(tick_events)
        velocity_environment = _summarize_velocity_environment(
            velocity_environment_before,
            velocity_environment_after,
        )
        report = {
            "surface": "continuous_runtime_stress.v1",
            "checkpoint": str(checkpoint),
            "run_root": str(run_root),
            "source_path": str(source_path),
            "scope": (
                "background_terminus_loop_with_full_prewarmed_source_queue_"
                "and_training_owned_sequential_cuda_text_sequence"
            ),
            "claim_boundary": (
                "measures complete warm continuous runtime cost; does not "
                "promote a new algorithm or skip per-token SNN updates"
            ),
            "success": bool(token_delta >= int(target_tokens)),
            "failure_reason": (
                None
                if token_delta >= int(target_tokens)
                else "target_tokens_not_reached_before_timeout"
            ),
            "target_tokens": int(target_tokens),
            "tick_tokens": int(tick_tokens),
            "quantum_tokens": int(quantum_tokens),
            "source_concept_observation_tick_interval": int(
                source_concept_observation_tick_interval
            ),
            "config_overrides": dict(config_overrides),
            "trainer_config": {
                "slow_memory_archive_interval_tokens": int(
                    manager._trainer.config.slow_memory_archive_interval_tokens
                ),
                "trainer_telemetry_interval_tokens": int(
                    manager._trainer.config.trainer_telemetry_interval_tokens
                ),
                "cuda_graph_host_truth_sync_interval_tokens": int(
                    manager._trainer.config.cuda_graph_host_truth_sync_interval_tokens
                ),
                "cuda_graph_native_burst_replay": bool(
                    manager._trainer.config.cuda_graph_native_burst_replay
                ),
                "cuda_graph_native_burst_tokens": int(
                    manager._trainer.config.cuda_graph_native_burst_tokens
                ),
                "cuda_graph_sequence_executor": str(
                    manager._trainer.config.cuda_graph_sequence_executor
                ),
                "cuda_graph_sequence_loop_tokens": int(
                    manager._trainer.config.cuda_graph_sequence_loop_tokens
                ),
            },
            "checkpoint_metadata": {
                "config_migrations": list(
                    manager._metadata.get("config_migrations") or []
                ),
                "hot_path_config_defaults_revision": manager._metadata.get(
                    "hot_path_config_defaults_revision"
                ),
            },
            "timeout_seconds": float(timeout_seconds),
            "sample_interval_seconds": float(sample_interval_seconds),
            "observer": {
                "expected_tick_count": int(expected_tick_count),
                "runtime_event_history_limit": int(event_history_limit),
                "event_history_capacity": dict(event_history_capacity),
                "poll_snapshots_during_measurement": bool(poll_snapshots),
                "poll_snapshot_count": int(poll_snapshot_count),
            },
            "trainer_stage_profile": trainer_stage_profile,
            "token_delta": int(token_delta),
            "elapsed_seconds": float(elapsed_seconds),
            "tokens_per_second": float(token_delta / max(elapsed_seconds, 1e-9)),
            "stop_latency_ms": float(stop_latency_ms),
            "warm_ingestion": warm_ingestion,
            "event_summary": event_summary,
            "last_tick_duration_ms": final_snapshot.get("last_tick_duration_ms"),
            "last_tick_token_delta": final_snapshot.get("last_tick_token_delta"),
            "last_tick_stage_timings_ms": dict(
                final_snapshot.get("last_tick_stage_timings_ms") or {}
            ),
            "execution_schedule": dict(final_snapshot.get("execution_schedule") or {}),
            "shutdown": dict(final_snapshot.get("shutdown") or {}),
            "cache_flush": cache_flush,
            "velocity_environment": velocity_environment,
            "runtime_device": device_report,
            "column_transition_runtime": transition_report,
        }
        write_json_report_with_readme(output_path, report)
        return report
    finally:
        manager.close()
        if native_burst_replay is not None:
            if previous_native_replay_env is None:
                os.environ.pop("MARULHO_CUDA_GRAPH_NATIVE_BURST_REPLAY", None)
            else:
                os.environ["MARULHO_CUDA_GRAPH_NATIVE_BURST_REPLAY"] = (
                    previous_native_replay_env
                )
        if normalized_sequence_executor is not None:
            if previous_sequence_executor_env is None:
                os.environ.pop("MARULHO_CUDA_GRAPH_SEQUENCE_EXECUTOR", None)
            else:
                os.environ["MARULHO_CUDA_GRAPH_SEQUENCE_EXECUTOR"] = (
                    previous_sequence_executor_env
                )

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=1024)
    parser.add_argument("--tick-tokens", type=int, default=128)
    parser.add_argument(
        "--quantum-tokens",
        type=int,
        default=DEFAULT_EXECUTION_QUANTUM_TOKENS,
    )
    parser.add_argument("--source-concept-observation-tick-interval", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.02)
    parser.add_argument(
        "--host-truth-sync-interval-tokens",
        type=int,
        default=None,
        help=(
            "Evaluation-only override for "
            "cuda_graph_host_truth_sync_interval_tokens. Use to remeasure "
            "Runtime Truth freshness/speed tradeoffs without editing a "
            "checkpoint or production preset."
        ),
    )
    parser.add_argument(
        "--profile-trainer-stages",
        action="store_true",
        help=(
            "Enable MarulhoTrainer train_step stage profiling during the "
            "measured stress window. This is profiling evidence and may "
            "perturb throughput."
        ),
    )
    parser.add_argument(
        "--disable-native-burst-replay",
        action="store_true",
        help=(
            "Evaluation-only override that keeps the current CUDA graph path "
            "but replays each token through the older Python CUDAGraph.replay "
            "loop for A/B comparison."
        ),
    )
    parser.add_argument(
        "--sequence-executor",
        choices=(
            "native_repeated_child_graph",
            "repeated_child",
            "native8",
            "default",
            "conditional_while",
            "cuda_graph_conditional_while",
        ),
        default=None,
        help=(
            "Evaluation-only override for the native text sequence executor. "
            "Default keeps the checkpoint or production config, currently "
            "the promoted CUDA conditional-WHILE sequence executor. Use "
            "native_repeated_child_graph to opt out to the retained native8 "
            "fallback path."
        ),
    )
    args = parser.parse_args()
    report = run_continuous_runtime_stress(
        args.checkpoint,
        output_path=args.output,
        target_tokens=args.target_tokens,
        tick_tokens=args.tick_tokens,
        quantum_tokens=args.quantum_tokens,
        source_concept_observation_tick_interval=args.source_concept_observation_tick_interval,
        timeout_seconds=args.timeout_seconds,
        sample_interval_seconds=args.sample_interval_seconds,
        profile_trainer_stages=args.profile_trainer_stages,
        host_truth_sync_interval_tokens=args.host_truth_sync_interval_tokens,
        native_burst_replay=False if args.disable_native_burst_replay else None,
        sequence_executor=args.sequence_executor,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
