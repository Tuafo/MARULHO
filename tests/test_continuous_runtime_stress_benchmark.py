from marulho.evaluation.continuous_runtime_stress_benchmark import (
    _collect_tick_events,
    _runtime_event_history_limit,
    _source_text_for_target,
    _summarize_tick_events,
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
