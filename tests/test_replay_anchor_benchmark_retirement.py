from __future__ import annotations

from pathlib import Path

from marulho.evaluation.replay_query_anchor_source_window_benchmark import (
    run_benchmark as run_replay_query_anchor_benchmark,
)
from marulho.evaluation.sleep_replay_anchor_source_window_benchmark import (
    run_benchmark as run_sleep_replay_anchor_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]


def test_anchor_source_benchmarks_do_not_keep_all_anchor_baseline_code() -> None:
    benchmark_paths = [
        ROOT / "src/marulho/evaluation/replay_query_anchor_source_window_benchmark.py",
        ROOT / "src/marulho/evaluation/sleep_replay_anchor_source_window_benchmark.py",
    ]
    forbidden_fragments = [
        "_time_legacy",
        "legacy_all_anchor",
        "sorted_all_column_anchors",
        "candidate_bucket_ids=all_anchor_buckets",
        "speedup_vs_legacy",
        "source_speedup_vs_retired",
    ]
    for path in benchmark_paths:
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source


def test_replay_query_anchor_benchmark_reports_maintained_only_path() -> None:
    report = run_replay_query_anchor_benchmark(
        anchor_count=64,
        column_latent_dim=8,
        max_queries=8,
        max_candidates=8,
        iterations=2,
        seed=20260623,
    )

    assert report["status"] == "passed"
    assert report["retired_all_anchor_source_absence"]["implementation_present"] is False
    assert report["retired_all_anchor_source_absence"]["all_anchor_collection_called"] is False
    assert "legacy_all_anchor_source" not in report
    assert report["bounded_anchor_source"]["anchor_source_full_scan"] is False
    assert report["quality"]["bounded_recent_anchor_query_hit_rate"] == 1.0
    assert report["runtime_truth"]["runs_live_tick"] is False
    assert report["runtime_truth"]["global_candidate_scan"] is False


def test_sleep_replay_anchor_benchmark_reports_maintained_only_path() -> None:
    report = run_sleep_replay_anchor_benchmark(
        anchor_count=64,
        column_latent_dim=8,
        replay_steps=8,
        candidate_pool=8,
        iterations=2,
        seed=20260623,
    )

    assert report["status"] == "passed"
    assert report["retired_all_anchor_source_absence"]["implementation_present"] is False
    assert report["retired_all_anchor_source_absence"]["all_anchor_source_called"] is False
    assert "retired_all_anchor_source" not in report
    assert report["bounded_sleep_anchor_source"]["anchor_source_full_scan"] is False
    assert report["quality"]["bounded_source_hit_rate"] == 1.0
    assert report["runtime_truth"]["runs_live_tick"] is False
    assert report["runtime_truth"]["global_candidate_scan"] is False
