from marulho.evaluation.continuous_runtime_stress_benchmark import (
    _collect_tick_events,
    _source_text_for_target,
    _summarize_tick_events,
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
    assert report["stages"]["trainer_step"]["mean_ms_per_token"] == 1.75
    assert report["top_stage_mean_ms_per_token"][0]["stage"] == "trainer_step"


def test_source_text_scales_for_long_full_warm_runs() -> None:
    short = _source_text_for_target(128)
    long = _source_text_for_target(1024)

    assert len(long) > len(short)
    assert "Adaptive memory plasticity" in long
