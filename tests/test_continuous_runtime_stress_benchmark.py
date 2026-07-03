from collections import deque
import json
import time

import marulho.evaluation.continuous_runtime_stress_benchmark as stress_benchmark
from marulho.evaluation.continuous_runtime_stress_benchmark import (
    _collect_tick_events,
    _ensure_runtime_event_history_capacity,
    _parse_nvidia_smi_gpu_row,
    _runtime_event_history_limit,
    _source_text_for_target,
    _summarize_tick_events,
    _summarize_velocity_environment,
    main,
    run_continuous_runtime_stress,
)


class _FakeStressConfig:
    slow_memory_archive_interval_tokens = 256
    trainer_telemetry_interval_tokens = 32
    cuda_graph_host_truth_sync_interval_tokens = 32
    cuda_graph_native_burst_replay = True
    cuda_graph_native_burst_tokens = 8
    cuda_graph_sequence_executor = "conditional_while"
    cuda_graph_sequence_loop_tokens = 16
    predictive_route_vote_mode = "cuda_graph_text"

    def device_report(self) -> dict:
        return {
            "requested_device": "auto",
            "resolved_device": "cuda:0",
            "cuda_available": True,
            "cuda_selected": True,
        }


class _FakeStressTrainer:
    def __init__(self) -> None:
        self.config = _FakeStressConfig()
        self.token_count = 0
        self.profile_disabled = False

    def column_transition_runtime_report(self) -> dict:
        return {
            "surface": "column_transition_runtime.v1",
            "failure_count": 0,
            "selection_failure_count": 0,
            "route_vote_fallback_reason": None,
            "text_burst_execution_count": 1,
            "text_burst_token_count": int(self.token_count),
            "text_burst_fallback_count": 0,
            "text_burst_fallback_reasons": {},
            "text_burst_last_fallback_reason": None,
            "text_sequence_execution_count": 1,
            "text_sequence_token_count": int(self.token_count),
            "cuda_graph_route_transition": {
                "active": True,
                "capture_succeeded": True,
                "replay_count": 1,
                "failure_count": 0,
                "native_burst_replay_fallback_count": 0,
                "native_burst_replay_failure_count": 0,
                "native_sequence_loop_fallback_count": 0,
                "native_sequence_loop_failure_count": 0,
                "burst_replay_failure_count": 0,
                "native_sequence_executor_requested": "cuda_graph_conditional_while",
                "native_sequence_loop_backend": "cuda_graph_conditional_while",
            },
        }

    def enable_train_step_profile(self, *, reset: bool = True) -> None:
        del reset

    def disable_train_step_profile(self) -> None:
        self.profile_disabled = True

    def train_step_profile_report(self) -> dict:
        return {"enabled": True, "count": int(self.token_count)}


class _FakeStressBrain:
    def __init__(self, exc: BaseException) -> None:
        self.trainer = _FakeStressTrainer()
        self._trace_history = deque([], maxlen=2)
        self.exc = exc
        self.stopped = False

    def feed(self, text: str, *, source: str, learn: bool) -> dict:
        del text, source, learn
        return {"accepted_tokens": 64, "queued_tokens": 64}

    def tick(self, *, tokens: int, quantum_tokens: int, source: str) -> dict:
        del tokens, quantum_tokens, source
        if isinstance(self.exc, KeyboardInterrupt):
            self.trainer.token_count += 4
        raise self.exc

    def status(self) -> dict:
        return {
            "surface": "marulho_brain_runtime.v1",
            "queued_tokens": 64,
            "token_count": int(self.trainer.token_count),
        }

    def trace(self) -> dict:
        return {
            "surface": "marulho_brain_trace.v1",
            "event": "tick",
            "trained_tokens": int(self.trainer.token_count),
            "elapsed_ms": 1.5,
            "executor": "conditional_while",
        }

    def trace_history(self, *, limit: int) -> list[dict]:
        del limit
        return [self.trace()]

    def stop(self, *, timeout_seconds: float) -> dict:
        del timeout_seconds
        self.stopped = True
        return {"stopped": True}


class _BoundedSourceStressBrain:
    def __init__(self, *, max_queue: int = 16) -> None:
        self.trainer = _FakeStressTrainer()
        self.max_queue = int(max_queue)
        self.queued = 0
        self.dropped_total = 0
        self.feed_calls: list[int] = []
        self._trace_history = deque([], maxlen=8)
        self.stopped = False

    def feed(self, text: str, *, source: str, learn: bool) -> dict:
        del source, learn
        accepted = len(text)
        overflow = max(0, self.queued + accepted - self.max_queue)
        self.dropped_total += overflow
        self.queued = min(self.max_queue, self.queued + accepted)
        self.feed_calls.append(accepted)
        return {"accepted_tokens": accepted, "queued_tokens": int(self.queued)}

    def tick(self, *, tokens: int, quantum_tokens: int, source: str) -> dict:
        del quantum_tokens, source
        trained = min(int(tokens), int(self.queued))
        self.queued -= trained
        self.trainer.token_count += trained
        trace = {
            "surface": "marulho_brain_trace.v1",
            "step": len(self._trace_history) + 1,
            "event": "tick",
            "trained_tokens": trained,
            "elapsed_ms": 1.0,
            "executor": "conditional_while",
        }
        self._trace_history.append(trace)
        return {"trained_tokens": trained, "queued_tokens": int(self.queued)}

    def status(self) -> dict:
        return {
            "surface": "marulho_brain_runtime.v1",
            "queued_tokens": int(self.queued),
            "token_count": int(self.trainer.token_count),
            "source_buffer": {
                "queued_tokens": int(self.queued),
                "dropped_total": int(self.dropped_total),
                "max_items": int(self.max_queue),
            },
        }

    def trace(self) -> dict:
        if self._trace_history:
            return dict(self._trace_history[-1])
        return {
            "surface": "marulho_brain_trace.v1",
            "event": "tick",
            "trained_tokens": 0,
            "elapsed_ms": 0.0,
            "executor": "conditional_while",
        }

    def trace_history(self, *, limit: int) -> list[dict]:
        return [dict(item) for item in list(self._trace_history)[-int(limit) :]]

    def stop(self, *, timeout_seconds: float) -> dict:
        del timeout_seconds
        self.stopped = True
        return {"stopped": True}


class _SlowTimeoutStressBrain(_BoundedSourceStressBrain):
    def tick(self, *, tokens: int, quantum_tokens: int, source: str) -> dict:
        time.sleep(0.12)
        return super().tick(
            tokens=tokens,
            quantum_tokens=quantum_tokens,
            source=source,
        )


def _patch_stress_runtime(monkeypatch, brain: _FakeStressBrain) -> None:
    monkeypatch.setattr(
        stress_benchmark.MarulhoBrain,
        "load",
        staticmethod(lambda checkpoint, trace_limit=64: brain),
    )
    monkeypatch.setattr(
        stress_benchmark,
        "_collect_velocity_environment_snapshot",
        lambda: {
            "cpu": {"available": False},
            "gpu": {"available": False},
        },
    )


def test_collect_tick_events_deduplicates_recent_history() -> None:
    events: list[dict] = []
    seen: set[tuple] = set()
    snapshot = {
        "recent_events": [
            {
                "type": "tick",
                "timestamp": 1.0,
                "token_delta": 8,
                "tick_duration_ms": 16.0,
                "stage_timings_ms": {"trainer_step": 12.0},
            },
            {"type": "status", "timestamp": 2.0},
        ]
    }

    _collect_tick_events(snapshot, seen_keys=seen, events=events)
    _collect_tick_events(snapshot, seen_keys=seen, events=events)

    assert len(events) == 1
    assert events[0]["token_delta"] == 8


def test_summarize_tick_events_reports_stage_cost_per_token() -> None:
    report = _summarize_tick_events(
        [
            {
                "type": "tick",
                "timestamp": 1.0,
                "token_delta": 8,
                "tick_duration_ms": 16.0,
                "source": {
                    "concept_observation": {
                        "mode": "sampled_batched",
                        "tick_due": True,
                        "attempts": 2,
                        "observations": 2,
                        "skipped_attempts": 0,
                    }
                },
                "stage_timings_ms": {
                    "collect_source_queue": 0.8,
                    "trainer_step": 12.0,
                },
            },
            {
                "type": "tick",
                "timestamp": 2.0,
                "token_delta": 8,
                "tick_duration_ms": 24.0,
                "source": {
                    "concept_observation": {
                        "mode": "cadenced_tick_skip",
                        "tick_due": False,
                        "attempts": 0,
                        "observations": 0,
                        "skipped_attempts": 0,
                    }
                },
                "stage_timings_ms": {
                    "collect_source_queue": 1.6,
                    "trainer_step": 16.0,
                },
            },
        ]
    )

    assert report["tick_event_count"] == 2
    assert report["observed_tick_tokens"] == 16
    assert report["tick_duration_ms"]["mean"] == 20.0
    assert report["concept_observation"]["modes"] == {
        "cadenced_tick_skip": 1,
        "sampled_batched": 1,
    }
    assert report["concept_observation"]["tick_due_count"] == 1
    assert report["concept_observation"]["tick_skip_count"] == 1
    assert report["concept_observation"]["observations"] == 2
    assert report["stages"]["trainer_step"]["mean_ms_per_token"] == 1.75
    assert report["top_stage_mean_ms_per_token"][0]["stage"] == "trainer_step"


def test_source_text_scales_for_long_full_warm_runs() -> None:
    short = _source_text_for_target(128)
    long = _source_text_for_target(1024)

    assert len(long) > len(short)
    assert "Adaptive memory plasticity" in long


def test_stress_runner_refills_bounded_source_buffer_for_long_targets(
    monkeypatch,
    tmp_path,
) -> None:
    brain = _BoundedSourceStressBrain(max_queue=16)
    _patch_stress_runtime(monkeypatch, brain)  # type: ignore[arg-type]
    output = tmp_path / "long-source-report.json"

    report = run_continuous_runtime_stress(
        tmp_path / "runtime.pt",
        output_path=output,
        target_tokens=40,
        tick_tokens=8,
        timeout_seconds=5.0,
        sample_interval_seconds=0.001,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["success"] is True
    assert report["token_delta"] == 40
    assert brain.dropped_total == 0
    assert len(brain.feed_calls) > 1
    assert written["warm_ingestion"]["mode"] == "brain_feed_streaming_refill"
    assert written["warm_ingestion"]["refill_count"] >= 1
    assert written["warm_ingestion"]["source_buffer_max_items"] == 16
    assert written["warm_ingestion"]["source_buffer_dropped_total"] == 0


def test_stress_runner_writes_exception_report(monkeypatch, tmp_path) -> None:
    brain = _FakeStressBrain(RuntimeError("simulated tick failure"))
    _patch_stress_runtime(monkeypatch, brain)
    output = tmp_path / "exception-report.json"

    report = run_continuous_runtime_stress(
        tmp_path / "runtime.pt",
        output_path=output,
        target_tokens=16,
        tick_tokens=8,
        timeout_seconds=5.0,
        profile_trainer_stages=True,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert brain.stopped is True
    assert report["success"] is False
    assert report["evidence_status"] == "exception"
    assert report["evidence_state"]["exception"] is True
    assert report["failure_reason"] == "exception:RuntimeError"
    assert written["exception"]["type"] == "RuntimeError"
    assert written["runtime_owner"] == "MarulhoBrain"
    assert written["runtime_device"]["resolved_device"] == "cuda:0"
    assert written["final_brain_trace"]["surface"] == "marulho_brain_trace.v1"
    assert written["failure_fallback_counters"]["cuda_graph_failure_count"] == 0
    assert written["executor_evidence"]["trainer_owner"] == "MarulhoTrainer"


def test_stress_runner_writes_keyboard_interrupt_report(monkeypatch, tmp_path) -> None:
    brain = _FakeStressBrain(KeyboardInterrupt())
    _patch_stress_runtime(monkeypatch, brain)
    output = tmp_path / "interrupt-report.json"

    report = run_continuous_runtime_stress(
        tmp_path / "runtime.pt",
        output_path=output,
        target_tokens=16,
        tick_tokens=8,
        timeout_seconds=5.0,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert brain.stopped is True
    assert report["success"] is False
    assert report["evidence_status"] == "interrupt"
    assert report["evidence_state"]["interrupt"] is True
    assert report["failure_reason"] == "keyboard_interrupt_manual_stop"
    assert report["token_delta"] == 4
    assert written["exception"]["type"] == "KeyboardInterrupt"
    assert written["token_delta"] == 4
    assert written["event_summary"]["tick_event_count"] >= 1


def test_stress_runner_writes_timeout_report(monkeypatch, tmp_path) -> None:
    brain = _SlowTimeoutStressBrain(max_queue=16)
    _patch_stress_runtime(monkeypatch, brain)  # type: ignore[arg-type]
    output = tmp_path / "timeout-report.json"

    report = run_continuous_runtime_stress(
        tmp_path / "runtime.pt",
        output_path=output,
        target_tokens=64,
        tick_tokens=8,
        timeout_seconds=0.1,
        sample_interval_seconds=0.001,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert brain.stopped is True
    assert report["success"] is False
    assert report["evidence_status"] == "timeout"
    assert report["evidence_state"]["timeout"] is True
    assert report["failure_reason"] == "target_tokens_not_reached_before_timeout"
    assert written["token_delta"] < 64
    assert written["warm_ingestion"]["mode"] == "brain_feed_streaming_refill"


def test_parse_nvidia_smi_gpu_row_reports_numeric_run_conditions() -> None:
    report = _parse_nvidia_smi_gpu_row(
        "NVIDIA GeForce RTX 3060, P3, 1140, 5001, 29.50, 25, 9, 44, 1849"
    )

    assert report["available"] is True
    assert report["name"] == "NVIDIA GeForce RTX 3060"
    assert report["pstate"] == "P3"
    assert report["graphics_clock_mhz"] == 1140
    assert report["memory_clock_mhz"] == 5001
    assert report["power_draw_w"] == 29.5
    assert report["gpu_utilization_percent"] == 25.0
    assert report["memory_used_mib"] == 1849


def test_velocity_environment_summary_marks_contention_as_slow_path_evidence() -> None:
    report = _summarize_velocity_environment(
        {
            "cpu": {
                "available": True,
                "percent_processor_time": 96.0,
            },
            "gpu": {
                "available": True,
                "gpu_utilization_percent": 8.0,
                "memory_utilization_percent": 11.0,
            },
        },
        {
            "cpu": {
                "available": True,
                "percent_processor_time": 38.0,
            },
            "gpu": {
                "available": True,
                "gpu_utilization_percent": 22.0,
                "memory_utilization_percent": 13.0,
            },
        },
    )

    assert report["surface"] == "velocity_environment.v1"
    assert report["not_hot_path"] is True
    assert report["contention"]["verdict"] == "contention_observed"
    assert report["contention"]["cpu_busy"] is True
    assert report["contention"]["gpu_busy"] is True
    assert report["contention"]["max_cpu_percent"] == 96.0
    assert report["contention"]["max_gpu_utilization_percent"] == 22.0


def test_velocity_environment_summary_reports_clean_or_unknown_comparability() -> None:
    clean = _summarize_velocity_environment(
        {
            "cpu": {"available": True, "percent_processor_time": 18.0},
            "gpu": {"available": True, "gpu_utilization_percent": 3.0},
        },
        {
            "cpu": {"available": True, "percent_processor_time": 32.0},
            "gpu": {"available": True, "gpu_utilization_percent": 14.0},
        },
    )
    unknown = _summarize_velocity_environment(
        {
            "cpu": {"available": False},
            "gpu": {"available": False},
        },
        None,
    )

    assert clean["contention"]["verdict"] == "not_observed"
    assert unknown["contention"]["verdict"] == "unknown"


def test_velocity_environment_summary_includes_measurement_samples() -> None:
    report = _summarize_velocity_environment(
        {
            "cpu": {"available": True, "percent_processor_time": 18.0},
            "gpu": {"available": True, "gpu_utilization_percent": 3.0},
        },
        {
            "cpu": {"available": True, "percent_processor_time": 31.0},
            "gpu": {"available": True, "gpu_utilization_percent": 6.0},
        },
        [
            {
                "measurement_elapsed_seconds": 10.0,
                "cpu": {"available": True, "percent_processor_time": 44.0},
                "gpu": {
                    "available": True,
                    "gpu_utilization_percent": 85.0,
                    "memory_utilization_percent": 19.0,
                    "memory_used_mib": 2048,
                    "graphics_clock_mhz": 1815,
                    "memory_clock_mhz": 7501,
                    "power_draw_w": 122.5,
                    "temperature_c": 61.0,
                },
            }
        ],
    )

    assert report["contention"]["verdict"] == "contention_observed"
    assert report["contention"]["max_gpu_utilization_percent"] == 85.0
    assert report["measurement"]["sample_count"] == 1
    assert report["measurement"]["max_cpu_percent"] == 44.0
    assert report["measurement"]["max_gpu_memory_used_mib"] == 2048
    assert report["measurement"]["max_graphics_clock_mhz"] == 1815
    assert report["measurement"]["max_power_draw_w"] == 122.5


def test_stress_runner_rejects_invalid_concept_tick_interval(tmp_path) -> None:
    try:
        run_continuous_runtime_stress(
            tmp_path / "missing.pt",
            output_path=tmp_path / "report.json",
            source_concept_observation_tick_interval=0,
        )
    except ValueError as exc:
        assert "source_concept_observation_tick_interval" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected invalid concept interval rejection")


def test_runtime_event_history_limit_reads_manager_capacity() -> None:
    class _State:
        _brain_event_history = [None]

    class _Manager:
        _runtime_state = _State()

    assert _runtime_event_history_limit(_Manager()) == 1


def test_stress_runner_extends_runtime_event_history_for_long_runs() -> None:
    from collections import deque

    class _State:
        _brain_event_history = deque([{"type": "existing"}], maxlen=2)

    class _Manager:
        _runtime_state = _State()

    report = _ensure_runtime_event_history_capacity(_Manager(), 5)

    assert report == {
        "extended": True,
        "before_limit": 2,
        "after_limit": 5,
        "required_events": 5,
    }
    history = _Manager._runtime_state._brain_event_history
    assert history.maxlen == 5
    assert list(history) == [{"type": "existing"}]


def test_main_forwards_trainer_stage_profile_and_host_truth_override(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, object] = {}

    def _run(checkpoint, **kwargs):
        captured["checkpoint"] = checkpoint
        captured.update(kwargs)
        return {"success": True}

    monkeypatch.setattr(
        "marulho.evaluation.continuous_runtime_stress_benchmark."
        "run_continuous_runtime_stress",
        _run,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "continuous-runtime-stress",
            "--checkpoint",
            str(tmp_path / "runtime.pt"),
            "--output",
            str(tmp_path / "report.json"),
            "--profile-trainer-stages",
            "--host-truth-sync-interval-tokens",
            "32",
            "--environment-sample-interval-seconds",
            "12.5",
        ],
    )

    assert main() == 0
    assert captured["profile_trainer_stages"] is True
    assert captured["host_truth_sync_interval_tokens"] == 32
    assert captured["environment_sample_interval_seconds"] == 12.5


def test_main_defaults_environment_sampling_outside_measured_window(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, object] = {}

    def _run(checkpoint, **kwargs):
        captured["checkpoint"] = checkpoint
        captured.update(kwargs)
        return {"success": True}

    monkeypatch.setattr(
        "marulho.evaluation.continuous_runtime_stress_benchmark."
        "run_continuous_runtime_stress",
        _run,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "continuous-runtime-stress",
            "--checkpoint",
            str(tmp_path / "runtime.pt"),
            "--output",
            str(tmp_path / "report.json"),
        ],
    )

    assert main() == 0
    assert captured["environment_sample_interval_seconds"] == 0.0


def test_stress_runner_rejects_invalid_host_truth_interval(tmp_path) -> None:
    try:
        run_continuous_runtime_stress(
            tmp_path / "missing.pt",
            output_path=tmp_path / "report.json",
            host_truth_sync_interval_tokens=0,
        )
    except ValueError as exc:
        assert "host_truth_sync_interval_tokens" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected invalid host truth interval rejection")


def test_stress_runner_rejects_negative_environment_sample_interval(tmp_path) -> None:
    try:
        run_continuous_runtime_stress(
            tmp_path / "missing.pt",
            output_path=tmp_path / "report.json",
            environment_sample_interval_seconds=-1.0,
        )
    except ValueError as exc:
        assert "environment_sample_interval_seconds" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected invalid environment sample interval rejection")
