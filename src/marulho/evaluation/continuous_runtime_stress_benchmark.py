"""Long warm stress gate for the continuous MarulhoBrain CUDA runtime."""

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
from typing import Any, Callable, Iterable, Mapping

from marulho.brain import MarulhoBrain
from marulho.brain.runtime import DEFAULT_BRAIN_QUANTUM_TOKENS
from marulho.evaluation.continuous_runtime_quantum_benchmark import (
    source_text_for_target as _quantum_source_text_for_target,
)
from marulho.reporting.readme_reports import write_json_report_with_readme


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
    return _quantum_source_text_for_target(
        target_tokens,
        slack_tokens=max(32, int(target_tokens) // 8),
    )


def _flush_source_cache_writes(runtime: Any) -> dict[str, Any]:
    flush = getattr(getattr(runtime, "_runtime_sources", None), "flush_brain_runtime_cache_writes", None)
    started = time.perf_counter()
    if callable(flush):
        flush()
    latency_ms = (time.perf_counter() - started) * 1000.0
    snapshot: Mapping[str, Any] = {}
    snapshot_fn = getattr(runtime, "_brain_runtime_snapshot_locked", None)
    lock = getattr(runtime, "_lock", None)
    if callable(snapshot_fn) and lock is not None:
        with lock:
            snapshot = snapshot_fn()
    return {
        "source_cache_flush_latency_ms": float(latency_ms),
        "mode": "legacy_runtime_sources_flush" if callable(flush) else "not_applicable_brain_source_buffer",
        "ingestion_after_flush": dict(snapshot.get("ingestion") or {}),
    }


def _runtime_event_history_limit(runtime: Any) -> int:
    history = getattr(getattr(runtime, "_runtime_state", None), "_brain_event_history", None)
    if history is None:
        history = getattr(runtime, "_trace_history", None)
    maxlen = getattr(history, "maxlen", None)
    return max(1, int(maxlen or 1))


def _ensure_runtime_event_history_capacity(
    runtime: Any,
    required_events: int,
) -> dict[str, Any]:
    runtime_state = getattr(runtime, "_runtime_state", None)
    history = getattr(runtime_state, "_brain_event_history", None)
    if history is None:
        history = getattr(runtime, "_trace_history", None)
    before_limit = max(1, int(getattr(history, "maxlen", None) or 1))
    required_limit = max(1, int(required_events))
    if before_limit >= required_limit or not isinstance(history, deque):
        return {
            "extended": False,
            "before_limit": int(before_limit),
            "after_limit": int(before_limit),
            "required_events": int(required_limit),
        }
    replacement = deque(list(history), maxlen=required_limit)
    if runtime_state is not None and hasattr(runtime_state, "_brain_event_history"):
        runtime_state._brain_event_history = replacement
    elif hasattr(runtime, "_trace_history"):
        runtime._trace_history = replacement
    return {
        "extended": True,
        "before_limit": int(before_limit),
        "after_limit": int(required_limit),
        "required_events": int(required_limit),
    }


def _brain_trace_snapshot(brain: MarulhoBrain, *, limit: int) -> dict[str, Any]:
    recent_events: list[dict[str, Any]] = []
    for trace in brain.trace_history(limit=limit):
        if trace.get("event") != "tick":
            continue
        token_delta = int(trace.get("trained_tokens", 0) or 0)
        elapsed_ms = float(trace.get("elapsed_ms", 0.0) or 0.0)
        recent_events.append(
            {
                "type": "tick",
                "timestamp": trace.get("created_at"),
                "token_delta": token_delta,
                "tick_duration_ms": elapsed_ms,
                "stage_timings_ms": {"trainer_step": elapsed_ms},
                "source": {
                    "concept_observation": {
                        "mode": "brain_trace",
                        "tick_due": token_delta > 0,
                        "attempts": 0,
                        "observations": 0,
                        "skipped_attempts": 0,
                    }
                },
            }
        )
    return {"recent_events": recent_events}


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


def _safe_report_section(
    producer: Callable[[], Mapping[str, Any]],
    *,
    unavailable_reason: str,
) -> dict[str, Any]:
    try:
        return dict(producer())
    except Exception as exc:  # pragma: no cover - defensive evidence guard
        return {
            "available": False,
            "unavailable_reason": unavailable_reason,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _safe_trainer_config_report(brain: MarulhoBrain | None) -> dict[str, Any]:
    if brain is None:
        return {
            "available": False,
            "unavailable_reason": "brain_not_loaded",
        }
    config = brain.trainer.config
    return {
        "slow_memory_archive_interval_tokens": int(
            config.slow_memory_archive_interval_tokens
        ),
        "trainer_telemetry_interval_tokens": int(
            config.trainer_telemetry_interval_tokens
        ),
        "cuda_graph_host_truth_sync_interval_tokens": int(
            config.cuda_graph_host_truth_sync_interval_tokens
        ),
        "cuda_graph_native_burst_replay": bool(
            config.cuda_graph_native_burst_replay
        ),
        "cuda_graph_native_burst_tokens": int(
            config.cuda_graph_native_burst_tokens
        ),
        "cuda_graph_sequence_executor": str(
            config.cuda_graph_sequence_executor
        ),
        "cuda_graph_sequence_loop_tokens": int(
            config.cuda_graph_sequence_loop_tokens
        ),
    }


def _safe_brain_trace(brain: MarulhoBrain | None) -> dict[str, Any]:
    if brain is None:
        return {
            "surface": "marulho_brain_trace.v1",
            "available": False,
            "unavailable_reason": "brain_not_loaded",
        }
    return _safe_report_section(
        lambda: brain.trace(),
        unavailable_reason="brain_trace_unavailable",
    )


def _safe_brain_status(brain: MarulhoBrain | None) -> dict[str, Any]:
    if brain is None:
        return {
            "surface": "marulho_brain_runtime.v1",
            "available": False,
            "unavailable_reason": "brain_not_loaded",
        }
    return _safe_report_section(
        lambda: brain.status(),
        unavailable_reason="brain_status_unavailable",
    )


def _source_buffer_snapshot(brain: MarulhoBrain | None) -> dict[str, int]:
    status = _safe_brain_status(brain)
    source_buffer = status.get("source_buffer")
    if not isinstance(source_buffer, Mapping):
        source_buffer = {}
    queued = status.get("queued_tokens", source_buffer.get("queued_tokens", 0))
    return {
        "queued_tokens": int(queued or 0),
        "dropped_total": int(source_buffer.get("dropped_total", 0) or 0),
        "max_items": int(source_buffer.get("max_items", 0) or 0),
    }


def _safe_column_transition_runtime_report(
    brain: MarulhoBrain | None,
) -> dict[str, Any]:
    if brain is None:
        return {
            "available": False,
            "unavailable_reason": "brain_not_loaded",
        }
    return _safe_report_section(
        lambda: brain.trainer.column_transition_runtime_report(),
        unavailable_reason="column_transition_runtime_unavailable",
    )


def _safe_runtime_device_report(brain: MarulhoBrain | None) -> dict[str, Any]:
    if brain is None:
        return {
            "available": False,
            "unavailable_reason": "brain_not_loaded",
        }
    return _safe_report_section(
        lambda: brain.trainer.config.device_report(),
        unavailable_reason="device_report_unavailable",
    )


def _safe_token_count(brain: MarulhoBrain | None) -> int | None:
    if brain is None:
        return None
    try:
        return int(brain.trainer.token_count)
    except Exception:  # pragma: no cover - defensive evidence guard
        return None


def _extract_failure_fallback_counters(
    transition_report: Mapping[str, Any],
) -> dict[str, Any]:
    graph_report = transition_report.get("cuda_graph_route_transition")
    if not isinstance(graph_report, Mapping):
        graph_report = {}
    return {
        "surface": "continuous_runtime_failure_fallback_counters.v1",
        "route_vote_fallback_reason": transition_report.get(
            "route_vote_fallback_reason"
        ),
        "runtime_failure_count": transition_report.get("failure_count"),
        "runtime_selection_failure_count": transition_report.get(
            "selection_failure_count"
        ),
        "text_burst_execution_count": transition_report.get(
            "text_burst_execution_count"
        ),
        "text_burst_token_count": transition_report.get("text_burst_token_count"),
        "text_burst_fallback_count": transition_report.get(
            "text_burst_fallback_count"
        ),
        "text_burst_fallback_reasons": dict(
            transition_report.get("text_burst_fallback_reasons") or {}
        ),
        "text_burst_last_fallback_reason": transition_report.get(
            "text_burst_last_fallback_reason"
        ),
        "text_sequence_execution_count": transition_report.get(
            "text_sequence_execution_count"
        ),
        "text_sequence_token_count": transition_report.get(
            "text_sequence_token_count"
        ),
        "cuda_graph_replay_count": graph_report.get("replay_count"),
        "cuda_graph_failure_count": graph_report.get("failure_count"),
        "native_burst_replay_fallback_count": graph_report.get(
            "native_burst_replay_fallback_count"
        ),
        "native_burst_replay_failure_count": graph_report.get(
            "native_burst_replay_failure_count"
        ),
        "native_burst_replay_last_error": graph_report.get(
            "native_burst_replay_last_error"
        ),
        "native_sequence_loop_fallback_count": graph_report.get(
            "native_sequence_loop_fallback_count"
        ),
        "native_sequence_loop_failure_count": graph_report.get(
            "native_sequence_loop_failure_count"
        ),
        "native_sequence_loop_last_error": graph_report.get(
            "native_sequence_loop_last_error"
        ),
        "burst_replay_failure_count": graph_report.get(
            "burst_replay_failure_count"
        ),
    }


def _executor_evidence(
    brain: MarulhoBrain | None,
    *,
    transition_report: Mapping[str, Any],
    device_report: Mapping[str, Any],
) -> dict[str, Any]:
    graph_report = transition_report.get("cuda_graph_route_transition")
    if not isinstance(graph_report, Mapping):
        graph_report = {}
    config = _safe_trainer_config_report(brain)
    return {
        "surface": "continuous_runtime_executor_evidence.v1",
        "runtime_owner": "MarulhoBrain",
        "trainer_owner": "MarulhoTrainer",
        "route_vote_mode": (
            str(getattr(brain.trainer.config, "predictive_route_vote_mode", ""))
            if brain is not None
            else None
        ),
        "sequence_executor": config.get("cuda_graph_sequence_executor"),
        "sequence_loop_tokens": config.get("cuda_graph_sequence_loop_tokens"),
        "native_burst_tokens": config.get("cuda_graph_native_burst_tokens"),
        "device": dict(device_report),
        "cuda_graph": {
            "active": graph_report.get("active"),
            "fallback_reason": graph_report.get("fallback_reason"),
            "capture_succeeded": graph_report.get("capture_succeeded"),
            "replay_count": graph_report.get("replay_count"),
            "failure_count": graph_report.get("failure_count"),
            "native_burst_replay_backend": graph_report.get(
                "native_burst_replay_backend"
            ),
            "native_sequence_executor_requested": graph_report.get(
                "native_sequence_executor_requested"
            ),
            "native_sequence_loop_backend": graph_report.get(
                "native_sequence_loop_backend"
            ),
        },
    }


def _evidence_status_for_run(
    *,
    success: bool,
    failure_reason: str | None,
    interrupted: bool,
    exception: BaseException | None,
    deadline_reached: bool,
) -> str:
    if interrupted:
        return "interrupt"
    if exception is not None:
        return "exception"
    if success:
        return "final"
    if deadline_reached or failure_reason == "target_tokens_not_reached_before_timeout":
        return "timeout"
    return "partial"


def _continuous_runtime_report(
    *,
    checkpoint: Path,
    output_path: Path,
    run_root: Path,
    source_path: Path,
    target_tokens: int,
    tick_tokens: int,
    quantum_tokens: int,
    source_concept_observation_tick_interval: int,
    timeout_seconds: float,
    sample_interval_seconds: float,
    environment_sample_interval_seconds: float,
    config_overrides: Mapping[str, Any],
    warm_ingestion: Mapping[str, Any] | None,
    start_token: int | None,
    final_token: int | None,
    started: float | None,
    deadline: float | None,
    tick_events: Iterable[Mapping[str, Any]],
    event_history_capacity: Mapping[str, Any] | None,
    event_history_limit: int | None,
    expected_tick_count: int | None,
    poll_snapshots: bool,
    poll_snapshot_count: int,
    velocity_environment_before: Mapping[str, Any] | None,
    velocity_environment_after: Mapping[str, Any] | None,
    measurement_environment_samples: Iterable[Mapping[str, Any]] | None,
    trainer_stage_profile: Mapping[str, Any] | None,
    brain: MarulhoBrain | None,
    success: bool,
    failure_reason: str | None,
    interrupted: bool = False,
    exception: BaseException | None = None,
    cache_flush: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    now = time.perf_counter()
    elapsed_seconds = 0.0 if started is None else max(0.0, now - started)
    resolved_final_token = final_token
    if resolved_final_token is None:
        resolved_final_token = _safe_token_count(brain)
    token_delta = max(
        0,
        int((resolved_final_token or 0) - (start_token or 0)),
    )
    deadline_reached = bool(
        deadline is not None
        and now >= float(deadline)
        and token_delta < int(target_tokens)
    )
    evidence_status = _evidence_status_for_run(
        success=bool(success),
        failure_reason=failure_reason,
        interrupted=bool(interrupted),
        exception=exception,
        deadline_reached=deadline_reached,
    )
    transition_report = _safe_column_transition_runtime_report(brain)
    device_report = _safe_runtime_device_report(brain)
    last_trace = _safe_brain_trace(brain)
    event_summary = _summarize_tick_events(tick_events)
    velocity_environment = _summarize_velocity_environment(
        velocity_environment_before,
        velocity_environment_after,
        measurement_environment_samples,
    )
    report = {
        "surface": "continuous_runtime_stress.v1",
        "checkpoint": str(checkpoint),
        "checkpoint_path": str(checkpoint),
        "report_path": str(output_path),
        "run_root": str(run_root),
        "source_path": str(source_path),
        "runtime_owner": "MarulhoBrain",
        "scope": (
            "manual_marulho_brain_tick_loop_with_streaming_source_buffer_refill_"
            "and_training_owned_sequential_cuda_text_sequence"
        ),
        "claim_boundary": (
            "measures complete warm continuous runtime cost; does not "
            "promote a new algorithm or skip per-token SNN updates"
        ),
        "success": bool(success),
        "failure_reason": failure_reason,
        "target_tokens": int(target_tokens),
        "token_delta": int(token_delta),
        "elapsed_seconds": float(elapsed_seconds),
        "tokens_per_second": (
            float(token_delta / elapsed_seconds)
            if elapsed_seconds > 0.0 and token_delta > 0
            else None
        ),
        "tick_tokens": int(tick_tokens),
        "quantum_tokens": int(quantum_tokens),
        "source_concept_observation_tick_interval": int(
            source_concept_observation_tick_interval
        ),
        "config_overrides": dict(config_overrides),
        "trainer_config": _safe_trainer_config_report(brain),
        "timeout_seconds": float(timeout_seconds),
        "sample_interval_seconds": float(sample_interval_seconds),
        "environment_sample_interval_seconds": float(
            environment_sample_interval_seconds
        ),
        "evidence_status": evidence_status,
        "evidence_kind": evidence_status,
        "evidence_state": {
            "final": evidence_status == "final",
            "partial": evidence_status in {"partial", "timeout", "interrupt", "exception"},
            "timeout": evidence_status == "timeout",
            "interrupt": evidence_status == "interrupt",
            "exception": evidence_status == "exception",
        },
        "observer": {
            "expected_tick_count": expected_tick_count,
            "runtime_event_history_limit": event_history_limit,
            "event_history_capacity": dict(event_history_capacity or {}),
            "poll_snapshots_during_measurement": bool(poll_snapshots),
            "poll_snapshot_count": int(poll_snapshot_count),
            "loop_owner": "MarulhoBrain.manual_tick_loop",
        },
        "trainer_stage_profile": (
            dict(trainer_stage_profile) if trainer_stage_profile is not None else None
        ),
        "stop_latency_ms": 0.0,
        "warm_ingestion": dict(warm_ingestion or {}),
        "event_summary": event_summary,
        "last_tick_duration_ms": last_trace.get("elapsed_ms"),
        "last_tick_token_delta": last_trace.get("trained_tokens"),
        "last_tick_stage_timings_ms": {
            "trainer_step": last_trace.get("elapsed_ms")
        },
        "last_brain_trace": last_trace,
        "final_brain_trace": last_trace,
        "brain_status": _safe_brain_status(brain),
        "execution_schedule": {
            "surface": "marulho_brain_manual_tick_schedule.v1",
            "owner": "MarulhoBrain",
            "tick_tokens": int(tick_tokens),
            "quantum_tokens": int(quantum_tokens),
            "stop_boundary": "between_ticks",
        },
        "shutdown": {
            "surface": "marulho_brain_manual_tick_shutdown.v1",
            "running": False,
        },
        "cache_flush": dict(cache_flush or {}),
        "velocity_environment": velocity_environment,
        "environment_contention_summary": dict(
            velocity_environment.get("contention") or {}
        ),
        "runtime_device": device_report,
        "executor_evidence": _executor_evidence(
            brain,
            transition_report=transition_report,
            device_report=device_report,
        ),
        "failure_fallback_counters": _extract_failure_fallback_counters(
            transition_report
        ),
        "column_transition_runtime": transition_report,
    }
    if exception is not None:
        report["exception"] = {
            "type": type(exception).__name__,
            "message": str(exception),
        }
    return report


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


def _available_metric_values(
    snapshots: Iterable[Mapping[str, Any] | None],
    section: str,
    key: str,
) -> list[float]:
    return [
        value
        for value in (
            _available_metric(snapshot, section, key) for snapshot in snapshots
        )
        if value is not None
    ]


def _summarize_velocity_environment(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
    measurement_samples: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    samples = [
        dict(sample)
        for sample in (measurement_samples or ())
        if isinstance(sample, Mapping)
    ]
    all_snapshots: list[Mapping[str, Any] | None] = [before, *samples, after]
    cpu_values = _available_metric_values(
        all_snapshots,
        "cpu",
        "percent_processor_time",
    )
    gpu_values = _available_metric_values(
        all_snapshots,
        "gpu",
        "gpu_utilization_percent",
    )
    memory_values = _available_metric_values(
        all_snapshots,
        "gpu",
        "memory_utilization_percent",
    )
    sample_cpu_values = _available_metric_values(
        samples,
        "cpu",
        "percent_processor_time",
    )
    sample_gpu_values = _available_metric_values(
        samples,
        "gpu",
        "gpu_utilization_percent",
    )
    sample_memory_values = _available_metric_values(
        samples,
        "gpu",
        "memory_utilization_percent",
    )
    sample_memory_used_values = _available_metric_values(
        samples,
        "gpu",
        "memory_used_mib",
    )
    sample_graphics_clock_values = _available_metric_values(
        samples,
        "gpu",
        "graphics_clock_mhz",
    )
    sample_memory_clock_values = _available_metric_values(
        samples,
        "gpu",
        "memory_clock_mhz",
    )
    sample_power_values = _available_metric_values(
        samples,
        "gpu",
        "power_draw_w",
    )
    sample_temperature_values = _available_metric_values(
        samples,
        "gpu",
        "temperature_c",
    )
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
        "measurement": {
            "sample_count": len(samples),
            "samples": samples,
            "max_cpu_percent": max(sample_cpu_values)
            if sample_cpu_values
            else None,
            "max_gpu_utilization_percent": max(sample_gpu_values)
            if sample_gpu_values
            else None,
            "max_gpu_memory_utilization_percent": max(sample_memory_values)
            if sample_memory_values
            else None,
            "max_gpu_memory_used_mib": max(sample_memory_used_values)
            if sample_memory_used_values
            else None,
            "max_graphics_clock_mhz": max(sample_graphics_clock_values)
            if sample_graphics_clock_values
            else None,
            "max_memory_clock_mhz": max(sample_memory_clock_values)
            if sample_memory_clock_values
            else None,
            "max_power_draw_w": max(sample_power_values)
            if sample_power_values
            else None,
            "max_temperature_c": max(sample_temperature_values)
            if sample_temperature_values
            else None,
        },
    }


def run_continuous_runtime_stress(
    checkpoint: Path,
    *,
    output_path: Path,
    target_tokens: int = 1024,
    tick_tokens: int = 128,
    quantum_tokens: int = DEFAULT_BRAIN_QUANTUM_TOKENS,
    source_concept_observation_tick_interval: int = 4,
    timeout_seconds: float = 60.0,
    sample_interval_seconds: float = 0.001,
    environment_sample_interval_seconds: float = 0.0,
    profile_trainer_stages: bool = False,
    host_truth_sync_interval_tokens: int | None = None,
    native_burst_replay: bool | None = None,
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
    if float(sample_interval_seconds) < 0.0:
        raise ValueError("sample_interval_seconds must be non-negative")
    if float(environment_sample_interval_seconds) < 0.0:
        raise ValueError("environment_sample_interval_seconds must be non-negative")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_root = output_path.parent / output_path.stem
    run_root.mkdir(parents=True, exist_ok=True)
    source_path = run_root / "continuous-runtime-stress-source.txt"
    source_text = _source_text_for_target(target_tokens)
    source_path.write_text(source_text, encoding="utf-8")
    brain: MarulhoBrain | None = None
    seen_events: set[tuple[Any, ...]] = set()
    tick_events: list[dict[str, Any]] = []
    config_overrides: dict[str, Any] = {}
    source_feed_state: dict[str, Any] = {
        "surface": "marulho_brain_source_buffer.v1",
        "accepted_tokens": 0,
        "queued_tokens": 0,
        "full_warm_ready": False,
        "mode": "brain_feed_streaming_refill",
        "source_text_chars": len(source_text),
        "source_text_consumed_chars": 0,
        "source_buffer_max_items": None,
        "source_buffer_dropped_total": 0,
        "feed_call_count": 0,
        "initial_feed_call_count": 0,
        "refill_count": 0,
        "refill_accepted_tokens": 0,
        "source_exhausted": False,
    }
    warm_ingestion: dict[str, Any] | None = source_feed_state
    source_offset = 0
    queued_tokens_estimate = 0
    start_token: int | None = None
    final_token: int | None = None
    started: float | None = None
    deadline: float | None = None
    velocity_environment_before: dict[str, Any] | None = None
    velocity_environment_after: dict[str, Any] | None = None
    measurement_environment_samples: list[dict[str, Any]] = []
    expected_tick_count: int | None = None
    event_history_capacity: dict[str, Any] | None = None
    event_history_limit: int | None = None
    poll_snapshots = False
    poll_snapshot_count = 0
    trainer_stage_profile: dict[str, Any] | None = None
    cache_flush: dict[str, Any] | None = None

    def feed_source_buffer(target_queue_tokens: int, *, phase: str) -> dict[str, Any]:
        nonlocal source_offset, queued_tokens_estimate
        if brain is None:
            return {
                "accepted_tokens": 0,
                "queued_tokens": int(queued_tokens_estimate),
                "source_exhausted": True,
            }
        target_queue_tokens = max(0, int(target_queue_tokens))
        accepted_total = 0
        feed_calls = 0
        while (
            queued_tokens_estimate < target_queue_tokens
            and source_offset < len(source_text)
        ):
            chunk_size = min(
                target_queue_tokens - queued_tokens_estimate,
                len(source_text) - source_offset,
            )
            if chunk_size <= 0:
                break
            chunk = source_text[source_offset : source_offset + chunk_size]
            source_offset += len(chunk)
            feed = brain.feed(
                chunk,
                source="continuous_runtime_stress_source",
                learn=False,
            )
            feed_calls += 1
            accepted = int(feed.get("accepted_tokens", 0) or 0)
            accepted_total += accepted
            queued_tokens_estimate = int(
                feed.get("queued_tokens", queued_tokens_estimate + accepted)
                or 0
            )
            if accepted <= 0:
                break
        source_snapshot = _source_buffer_snapshot(brain)
        queued_tokens_estimate = int(
            source_snapshot.get("queued_tokens", queued_tokens_estimate)
            or 0
        )
        source_feed_state["accepted_tokens"] = int(
            source_feed_state.get("accepted_tokens", 0) or 0
        ) + int(accepted_total)
        source_feed_state["queued_tokens"] = int(queued_tokens_estimate)
        source_feed_state["full_warm_ready"] = int(queued_tokens_estimate) > 0
        source_feed_state["source_text_consumed_chars"] = int(source_offset)
        source_feed_state["source_buffer_max_items"] = int(
            source_snapshot.get("max_items", 0) or 0
        )
        source_feed_state["source_buffer_dropped_total"] = int(
            source_snapshot.get("dropped_total", 0) or 0
        )
        source_feed_state["feed_call_count"] = int(
            source_feed_state.get("feed_call_count", 0) or 0
        ) + int(feed_calls)
        if phase == "initial":
            source_feed_state["initial_feed_call_count"] = int(
                source_feed_state.get("initial_feed_call_count", 0) or 0
            ) + int(feed_calls)
        else:
            source_feed_state["refill_count"] = int(
                source_feed_state.get("refill_count", 0) or 0
            ) + int(feed_calls)
            source_feed_state["refill_accepted_tokens"] = int(
                source_feed_state.get("refill_accepted_tokens", 0) or 0
            ) + int(accepted_total)
        source_feed_state["source_exhausted"] = bool(
            source_offset >= len(source_text)
        )
        return {
            "accepted_tokens": int(accepted_total),
            "queued_tokens": int(queued_tokens_estimate),
            "feed_call_count": int(feed_calls),
            "source_exhausted": bool(source_offset >= len(source_text)),
        }

    def build_report(
        *,
        success: bool,
        failure_reason: str | None,
        interrupted: bool = False,
        exception: BaseException | None = None,
    ) -> dict[str, Any]:
        return _continuous_runtime_report(
            checkpoint=checkpoint,
            output_path=output_path,
            run_root=run_root,
            source_path=source_path,
            target_tokens=target_tokens,
            tick_tokens=tick_tokens,
            quantum_tokens=quantum_tokens,
            source_concept_observation_tick_interval=(
                source_concept_observation_tick_interval
            ),
            timeout_seconds=timeout_seconds,
            sample_interval_seconds=sample_interval_seconds,
            environment_sample_interval_seconds=(
                environment_sample_interval_seconds
            ),
            config_overrides=config_overrides,
            warm_ingestion=warm_ingestion,
            start_token=start_token,
            final_token=final_token,
            started=started,
            deadline=deadline,
            tick_events=tick_events,
            event_history_capacity=event_history_capacity,
            event_history_limit=event_history_limit,
            expected_tick_count=expected_tick_count,
            poll_snapshots=poll_snapshots,
            poll_snapshot_count=poll_snapshot_count,
            velocity_environment_before=velocity_environment_before,
            velocity_environment_after=velocity_environment_after,
            measurement_environment_samples=measurement_environment_samples,
            trainer_stage_profile=trainer_stage_profile,
            brain=brain,
            success=success,
            failure_reason=failure_reason,
            interrupted=interrupted,
            exception=exception,
            cache_flush=cache_flush,
        )

    previous_native_replay_env = os.environ.get(
        "MARULHO_CUDA_GRAPH_NATIVE_BURST_REPLAY"
    )
    if native_burst_replay is not None:
        os.environ["MARULHO_CUDA_GRAPH_NATIVE_BURST_REPLAY"] = (
            "1" if bool(native_burst_replay) else "0"
        )
    try:
        brain = MarulhoBrain.load(
            checkpoint,
            trace_limit=max(64, int(target_tokens // max(1, tick_tokens)) + 32),
        )
        if host_truth_sync_interval_tokens is not None:
            previous_interval = int(
                brain.trainer.config.cuda_graph_host_truth_sync_interval_tokens
            )
            brain.trainer.config.cuda_graph_host_truth_sync_interval_tokens = int(
                host_truth_sync_interval_tokens
            )
            config_overrides["cuda_graph_host_truth_sync_interval_tokens"] = {
                "from": previous_interval,
                "to": int(host_truth_sync_interval_tokens),
            }
        if native_burst_replay is not None:
            previous_native_replay = bool(
                brain.trainer.config.cuda_graph_native_burst_replay
            )
            brain.trainer.config.cuda_graph_native_burst_replay = bool(
                native_burst_replay
            )
            config_overrides["cuda_graph_native_burst_replay"] = {
                "from": previous_native_replay,
                "to": bool(native_burst_replay),
            }
        source_buffer_capacity = _source_buffer_snapshot(brain).get("max_items", 0)
        if source_buffer_capacity <= 0:
            source_buffer_capacity = min(
                max(int(tick_tokens) * 2, int(quantum_tokens) * 2),
                max(int(target_tokens), int(tick_tokens)),
            )
        source_feed_state["source_buffer_max_items"] = int(source_buffer_capacity)
        initial_queue_target = min(
            int(source_buffer_capacity),
            max(int(tick_tokens), min(int(target_tokens), int(source_buffer_capacity))),
        )
        feed_source_buffer(initial_queue_target, phase="initial")
        if not bool(warm_ingestion.get("full_warm_ready")):
            velocity_environment_before = _collect_velocity_environment_snapshot()
            report = build_report(
                success=False,
                failure_reason="source_queue_not_fully_warm",
            )
            write_json_report_with_readme(output_path, report)
            return report

        velocity_environment_before = _collect_velocity_environment_snapshot()
        start_token = int(brain.trainer.token_count)
        if bool(profile_trainer_stages):
            brain.trainer.enable_train_step_profile(reset=True)
        started = time.perf_counter()
        deadline = started + max(0.1, float(timeout_seconds))
        current_token = start_token
        next_environment_sample_at = (
            started + max(0.001, float(environment_sample_interval_seconds))
            if float(environment_sample_interval_seconds) > 0.0
            else None
        )
        expected_tick_count = max(
            1,
            (int(target_tokens) + int(tick_tokens) - 1) // int(tick_tokens),
        )
        event_history_capacity = _ensure_runtime_event_history_capacity(
            brain,
            expected_tick_count + 16,
        )
        event_history_limit = _runtime_event_history_limit(brain)
        poll_snapshots = expected_tick_count > event_history_limit
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if (
                next_environment_sample_at is not None
                and now >= next_environment_sample_at
            ):
                sample = _collect_velocity_environment_snapshot()
                sample["measurement_elapsed_seconds"] = float(now - started)
                measurement_environment_samples.append(sample)
                while next_environment_sample_at <= now:
                    next_environment_sample_at += max(
                        0.001,
                        float(environment_sample_interval_seconds),
                    )
            current_token = int(brain.trainer.token_count)
            if current_token - start_token >= int(target_tokens):
                break
            remaining_tokens = max(1, int(target_tokens) - (current_token - start_token))
            desired_queue = min(
                int(source_buffer_capacity),
                max(int(tick_tokens), int(remaining_tokens)),
            )
            if queued_tokens_estimate < min(int(tick_tokens), int(remaining_tokens)):
                feed_source_buffer(desired_queue, phase="refill")
            tick = brain.tick(
                tokens=min(int(tick_tokens), remaining_tokens),
                quantum_tokens=int(quantum_tokens),
                source="continuous_runtime_stress_source",
            )
            queued_tokens_estimate = int(
                tick.get(
                    "queued_tokens",
                    max(
                        0,
                        int(queued_tokens_estimate)
                        - int(tick.get("trained_tokens", 0) or 0),
                    ),
                )
                or 0
            )
            snapshot = (
                _brain_trace_snapshot(brain, limit=expected_tick_count + 16)
                if poll_snapshots
                else None
            )
            if snapshot is not None:
                poll_snapshot_count += 1
                _collect_tick_events(
                    snapshot,
                    seen_keys=seen_events,
                    events=tick_events,
                )
            current_token = int(brain.trainer.token_count)
            if int(tick.get("trained_tokens", 0) or 0) <= 0 and int(queued_tokens_estimate) <= 0:
                feed_source_buffer(desired_queue, phase="refill")
            if int(tick.get("trained_tokens", 0) or 0) <= 0 and int(queued_tokens_estimate) <= 0:
                break
            time.sleep(max(0.001, float(sample_interval_seconds)))
        velocity_environment_after = _collect_velocity_environment_snapshot()
        final_token = int(brain.trainer.token_count)
        final_source_snapshot = _source_buffer_snapshot(brain)
        source_feed_state["queued_tokens"] = int(
            final_source_snapshot.get("queued_tokens", queued_tokens_estimate) or 0
        )
        source_feed_state["source_buffer_dropped_total"] = int(
            final_source_snapshot.get("dropped_total", 0) or 0
        )
        final_snapshot = _brain_trace_snapshot(brain, limit=expected_tick_count + 16)
        _collect_tick_events(
            final_snapshot,
            seen_keys=seen_events,
            events=tick_events,
        )
        trainer_stage_profile = (
            brain.trainer.train_step_profile_report()
            if bool(profile_trainer_stages)
            else None
        )
        if bool(profile_trainer_stages):
            brain.trainer.disable_train_step_profile()
        cache_flush = _flush_source_cache_writes(brain)
        token_delta = max(0, final_token - start_token)
        success = token_delta >= int(target_tokens)
        deadline_reached = time.perf_counter() >= float(deadline)
        failure_reason = (
            None
            if success
            else "target_tokens_not_reached_before_timeout"
            if deadline_reached
            else "target_tokens_not_reached_before_source_exhausted"
        )
        report = build_report(success=success, failure_reason=failure_reason)
        write_json_report_with_readme(output_path, report)
        return report
    except KeyboardInterrupt as exc:
        final_token = _safe_token_count(brain)
        velocity_environment_after = _collect_velocity_environment_snapshot()
        if brain is not None and expected_tick_count is not None:
            try:
                _collect_tick_events(
                    _brain_trace_snapshot(brain, limit=expected_tick_count + 16),
                    seen_keys=seen_events,
                    events=tick_events,
                )
            except Exception:
                pass
        if brain is not None and bool(profile_trainer_stages):
            trainer_stage_profile = _safe_report_section(
                lambda: brain.trainer.train_step_profile_report(),
                unavailable_reason="trainer_stage_profile_unavailable",
            )
        if brain is not None:
            cache_flush = _safe_report_section(
                lambda: _flush_source_cache_writes(brain),
                unavailable_reason="source_cache_flush_unavailable",
            )
        report = build_report(
            success=False,
            failure_reason="keyboard_interrupt_manual_stop",
            interrupted=True,
            exception=exc,
        )
        write_json_report_with_readme(output_path, report)
        return report
    except Exception as exc:
        final_token = _safe_token_count(brain)
        velocity_environment_after = _collect_velocity_environment_snapshot()
        if brain is not None and expected_tick_count is not None:
            try:
                _collect_tick_events(
                    _brain_trace_snapshot(brain, limit=expected_tick_count + 16),
                    seen_keys=seen_events,
                    events=tick_events,
                )
            except Exception:
                pass
        if brain is not None and bool(profile_trainer_stages):
            trainer_stage_profile = _safe_report_section(
                lambda: brain.trainer.train_step_profile_report(),
                unavailable_reason="trainer_stage_profile_unavailable",
            )
        if brain is not None:
            cache_flush = _safe_report_section(
                lambda: _flush_source_cache_writes(brain),
                unavailable_reason="source_cache_flush_unavailable",
            )
        report = build_report(
            success=False,
            failure_reason=f"exception:{type(exc).__name__}",
            exception=exc,
        )
        write_json_report_with_readme(output_path, report)
        return report
    finally:
        if brain is not None and bool(profile_trainer_stages):
            try:
                brain.trainer.disable_train_step_profile()
            except Exception:
                pass
        if brain is not None:
            brain.stop(timeout_seconds=1.0)
        if native_burst_replay is not None:
            if previous_native_replay_env is None:
                os.environ.pop("MARULHO_CUDA_GRAPH_NATIVE_BURST_REPLAY", None)
            else:
                os.environ["MARULHO_CUDA_GRAPH_NATIVE_BURST_REPLAY"] = (
                    previous_native_replay_env
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
        default=DEFAULT_BRAIN_QUANTUM_TOKENS,
    )
    parser.add_argument("--source-concept-observation-tick-interval", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.001)
    parser.add_argument(
        "--environment-sample-interval-seconds",
        type=float,
        default=0.0,
        help=(
            "Slow-path device/CPU sampling cadence during the measured window. "
            "Use 0 to disable measurement samples."
        ),
    )
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
        environment_sample_interval_seconds=args.environment_sample_interval_seconds,
        profile_trainer_stages=args.profile_trainer_stages,
        host_truth_sync_interval_tokens=args.host_truth_sync_interval_tokens,
        native_burst_replay=False if args.disable_native_burst_replay else None,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
