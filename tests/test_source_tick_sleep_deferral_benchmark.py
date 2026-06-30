from __future__ import annotations

from marulho.evaluation.source_tick_sleep_deferral_benchmark import run_benchmark


def test_source_tick_sleep_deferral_benchmark_uses_marulho_brain() -> None:
    report = run_benchmark(runs=1, seed=20260630, sleep_cost_ms=0.0)

    assert report["pass"] is True
    assert report["surface"] == "marulho_brain_source_tick_sleep_replay_deferred.v1"
    assert report["runtime_truth"]["runtime_owner"] == "MarulhoBrain"
    assert report["quality"]["brain_tick_replay_deferred"] is True
    assert report["quality"]["brain_sequence_tick_replay_deferred"] is True
    assert report["quality"]["explicit_slow_path_remains_available"] is True
    assert (
        report["brain_deferred"]["last_run"]["sleep_maintenance_allowed"]
        is False
    )
    assert report["brain_deferred"]["last_run"]["sleep_probe_calls"] == []
    assert (
        report["brain_sequence_deferred"]["last_run"]["runtime_owner"]
        == "MarulhoBrain"
    )
    assert (
        report["brain_sequence_deferred"]["last_run"]["sleep_probe_calls"]
        == []
    )
    assert (
        report["allowed_slow_path_projection"]["last_run"]["sleep_probe_calls"]
        == ["deep"]
    )
