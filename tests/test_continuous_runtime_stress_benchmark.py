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
            "--native-burst-tokens",
            "16",
            "--sequence-executor",
            "conditional_while",
        ],
    )

    assert main() == 0
    assert captured["profile_trainer_stages"] is True
    assert captured["host_truth_sync_interval_tokens"] == 32
    assert captured["native_burst_tokens"] == 16
    assert captured["sequence_executor"] == "conditional_while"


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


def test_stress_runner_rejects_native_burst_tokens_above_quantum(tmp_path) -> None:
    try:
        run_continuous_runtime_stress(
            tmp_path / "missing.pt",
            output_path=tmp_path / "report.json",
            quantum_tokens=16,
            native_burst_tokens=32,
        )
    except ValueError as exc:
        assert "native_burst_tokens" in str(exc)
        assert "quantum_tokens" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected native burst quantum alignment rejection")


def test_stress_runner_rejects_native_burst_tokens_that_do_not_divide_quantum(
    tmp_path,
) -> None:
    try:
        run_continuous_runtime_stress(
            tmp_path / "missing.pt",
            output_path=tmp_path / "report.json",
            quantum_tokens=24,
            native_burst_tokens=16,
        )
    except ValueError as exc:
        assert "native_burst_tokens" in str(exc)
        assert "divide quantum_tokens" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected native burst divisibility rejection")


def test_stress_runner_rejects_unknown_sequence_executor(tmp_path) -> None:
    try:
        run_continuous_runtime_stress(
            tmp_path / "missing.pt",
            output_path=tmp_path / "report.json",
            sequence_executor="python-wrapper",
        )
    except ValueError as exc:
        assert "sequence_executor" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected unknown sequence executor rejection")
