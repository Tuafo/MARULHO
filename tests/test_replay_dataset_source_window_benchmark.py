from __future__ import annotations

import argparse

from marulho.evaluation.replay_dataset_source_window_benchmark import run_benchmark


def test_replay_dataset_source_window_benchmark_passes() -> None:
    report = run_benchmark(
        argparse.Namespace(
            trace_count=64,
            sample_count=128,
            selected_candidates_per_sample=16,
            limit=50,
            endpoint="respond",
            runs=2,
        )
    )

    assert report["surface"] == "bounded_replay_dataset_source_window_benchmark.v1"
    assert report["pass"] is True
    assert report["quality"]["selected_target_ids_match"] is True
    assert report["quality"]["link_coverage_matches"] is True
    assert report["pass_checks"]["runs_live_tick_false"] is True
    assert report["pass_checks"]["runs_every_token_false"] is True
    assert report["device_placement"]["archival_storage_device"] == "cpu"
    assert report["device_placement"]["gpu_resident_archival_metadata"] is False
    assert report["source_window"]["source_window_count"] == 50
    assert report["source_window"]["replay_sample_link_source_window"]["source_window_count"] == 64
    assert report["work_reduction"]["replay_sample_record_reduction"] == 2.0
    assert report["work_reduction"]["selected_candidate_record_reduction"] == 2.0
