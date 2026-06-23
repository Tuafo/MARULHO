from __future__ import annotations

import argparse
from pathlib import Path

from marulho.evaluation.snn_readout_replay_priority_source_window_benchmark import (
    run_benchmark as run_readout_replay_priority_benchmark,
)
from marulho.evaluation.snn_emission_review_replay_policy_source_window_benchmark import (
    run_benchmark as run_emission_review_replay_policy_benchmark,
)
from marulho.evaluation.sleep_plasticity_ticket_queue_source_window_benchmark import (
    run_benchmark as run_sleep_plasticity_ticket_queue_benchmark,
)
from marulho.evaluation.snn_rollout_rehearsal_source_window_benchmark import (
    run_benchmark as run_rollout_rehearsal_benchmark,
)
from marulho.evaluation.source_bank_memory_match_benchmark import (
    run_benchmark as run_source_bank_memory_match_benchmark,
)
from marulho.evaluation.status_replay_path_source_window_benchmark import (
    run_benchmark as run_status_replay_path_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]


def test_bounded_replay_window_benchmarks_do_not_keep_retired_baselines() -> None:
    benchmark_paths = [
        ROOT / "src/marulho/evaluation/source_bank_memory_match_benchmark.py",
        ROOT / "src/marulho/evaluation/snn_readout_replay_priority_source_window_benchmark.py",
        ROOT / "src/marulho/evaluation/snn_emission_review_replay_policy_source_window_benchmark.py",
        ROOT / "src/marulho/evaluation/snn_rollout_rehearsal_source_window_benchmark.py",
        ROOT / "src/marulho/evaluation/status_replay_path_source_window_benchmark.py",
        ROOT / "src/marulho/evaluation/sleep_plasticity_ticket_queue_source_window_benchmark.py",
        ROOT / "src/marulho/semantics/frontier.py",
    ]
    forbidden_fragments = [
        "from marulho.training.query_runner import memory_matches_with_report",
        "_diagnostic_legacy_bank_memory_matches_no_cache",
        "_legacy_full_retained_priority",
        "diagnostic_legacy_snn_readout_full_retained_priority",
        "_legacy_full_retained_policy",
        "diagnostic_legacy_snn_rollout_full_retained_policy",
        "diagnostic_legacy_snn_emission_review_full_retained_policy_and_design",
        "_legacy_emission_projection",
        "_legacy_rollout_projection",
        "_legacy_emission_review_history",
        "latest_emission_matches_legacy",
        "latest_history_matches_legacy",
        "latest_rollout_matches_legacy",
        "legacy_emission",
        "legacy_rollout",
        "legacy_history",
        "_diagnostic_sleep_ticket_queue",
        "_diagnostic_scheduler_design_ticket_queue",
        "diagnostic_full_retained_sleep_plasticity_review_ticket_queue",
        "diagnostic_full_retained_sleep_plasticity_scheduler_design",
        "sleep_latest_verified_matches_diagnostic",
        "design_latest_verified_matches_diagnostic",
        "legacy_samples",
        "legacy_mean",
        "bounded_speedup_vs_legacy",
        "retired_per_probe_query_match_call_count",
        "retired_path_comparison",
    ]
    for path in benchmark_paths:
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source


def test_source_bank_memory_match_benchmark_reports_maintained_only_path() -> None:
    report = run_source_bank_memory_match_benchmark(
        argparse.Namespace(
            capacity=128,
            bucket_count=8,
            probe_samples=4,
            memories_per_probe=4,
            max_matches=8,
            payload_repeats=2,
            iterations=2,
        )
    )

    assert report["passed"] is True
    assert report["quality"]["selected_indices_match_expected"] is True
    assert report["retired_per_probe_query_match_absence"] == {
        "implementation_present": False,
        "diagnostic_callable": False,
        "active_report_field_present": False,
        "removed_policy": "per_probe_query_memory_match_aggregation",
    }
    assert "legacy_report" not in report
    assert "bounded_speedup_vs_legacy" not in report
    assert "retired_per_probe_query_match_call_count" not in report["bounded_report"]
    assert report["bounded_report"]["runs_live_tick"] is False
    assert report["bounded_report"]["global_candidate_scan"] is False
    assert report["bounded_report"]["global_score_scan"] is False
    assert report["bounded_report"]["language_reasoning"] is False


def test_readout_replay_priority_benchmark_reports_maintained_only_path() -> None:
    report = run_readout_replay_priority_benchmark(
        argparse.Namespace(retention_count=64, limit=8, runs=2, output=None)
    )

    assert report["pass"] is True
    assert report["pass_checks"]["top_hash_present"] is True
    assert report["pass_checks"]["recent_high_signal_selected"] is True
    assert report["retired_full_retained_priority_absence"][
        "implementation_present"
    ] is False
    assert report["retired_full_retained_priority_absence"][
        "diagnostic_callable"
    ] is False
    assert "legacy_selected_hashes" not in report["quality"]
    assert "legacy_mean_ms" not in report["latency"]
    assert "bounded_speedup_vs_legacy" not in report["latency"]
    assert "retired_path_comparison" not in report
    assert report["source_window"]["runs_live_tick"] is False
    assert report["source_window"]["global_candidate_scan"] is False
    assert report["source_window"]["global_score_scan"] is False
    assert report["source_window"]["language_reasoning"] is False


def test_emission_review_replay_policy_benchmark_reports_maintained_only_path() -> None:
    report = run_emission_review_replay_policy_benchmark(
        argparse.Namespace(retention_count=64, limit=8, runs=2, output=None)
    )

    assert report["pass"] is True
    assert report["pass_checks"]["top_quality_matches_expected_seed"] is True
    assert report["pass_checks"]["recent_high_signal_selected"] is True
    assert report["retired_full_retained_emission_review_policy_absence"][
        "implementation_present"
    ] is False
    assert report["retired_full_retained_emission_review_policy_absence"][
        "diagnostic_callable"
    ] is False
    assert "legacy_selected_prediction_hashes" not in report["quality"]
    assert "legacy_mean_ms" not in report["latency"]
    assert "bounded_speedup_vs_legacy" not in report["latency"]
    assert "retired_path_comparison" not in report
    assert report["policy_source_window"]["runs_live_tick"] is False
    assert report["policy_source_window"]["global_candidate_scan"] is False
    assert report["policy_source_window"]["global_score_scan"] is False
    assert report["policy_source_window"]["language_reasoning"] is False
    assert report["design_source_window"]["runs_live_tick"] is False
    assert report["design_source_window"]["global_candidate_scan"] is False
    assert report["design_source_window"]["global_score_scan"] is False
    assert report["design_source_window"]["language_reasoning"] is False


def test_rollout_rehearsal_benchmark_reports_maintained_only_path() -> None:
    report = run_rollout_rehearsal_benchmark(
        argparse.Namespace(retention_count=64, limit=8, runs=2, output=None)
    )

    assert report["pass"] is True
    assert report["pass_checks"]["top_hash_present"] is True
    assert report["pass_checks"]["recent_high_signal_selected"] is True
    assert report["retired_full_retained_rollout_policy_absence"][
        "implementation_present"
    ] is False
    assert report["retired_full_retained_rollout_policy_absence"][
        "diagnostic_callable"
    ] is False
    assert "legacy_selected_hashes" not in report["quality"]
    assert "legacy_mean_ms" not in report["latency"]
    assert "bounded_speedup_vs_legacy" not in report["latency"]
    assert "retired_path_comparison" not in report
    assert report["source_window"]["runs_live_tick"] is False
    assert report["source_window"]["global_candidate_scan"] is False
    assert report["source_window"]["global_score_scan"] is False
    assert report["source_window"]["language_reasoning"] is False


def test_status_replay_path_benchmark_reports_maintained_only_path() -> None:
    report = run_status_replay_path_benchmark(
        argparse.Namespace(retention_count=64, runs=2, output=None)
    )

    assert report["pass"] is True
    assert report["pass_checks"]["latest_emission_matches_expected_seed"] is True
    assert report["pass_checks"]["latest_history_matches_expected_seed"] is True
    assert report["pass_checks"]["latest_rollout_matches_expected_seed"] is True
    assert report["retired_full_retained_status_projection_absence"][
        "implementation_present"
    ] is False
    assert report["retired_full_retained_status_projection_absence"][
        "diagnostic_callable"
    ] is False
    assert "legacy_emission_seed_count" not in report["quality"]
    assert "legacy_history_unique_count" not in report["quality"]
    assert "legacy_rollout_unique_count" not in report["quality"]
    assert "legacy_combined_mean_ms" not in report["latency"]
    assert "bounded_speedup_vs_legacy" not in report["latency"]
    assert "retired_path_comparison" not in report
    assert report["emission_source_window"]["runs_live_tick"] is False
    assert report["emission_source_window"]["global_candidate_scan"] is False
    assert report["emission_source_window"]["global_score_scan"] is False
    assert report["emission_source_window"]["language_reasoning"] is False
    assert report["emission_review_history_source_window"]["runs_live_tick"] is False
    assert (
        report["emission_review_history_source_window"]["global_candidate_scan"]
        is False
    )
    assert report["rollout_source_window"]["runs_live_tick"] is False
    assert report["rollout_source_window"]["global_candidate_scan"] is False


def test_sleep_plasticity_ticket_queue_benchmark_reports_maintained_only_path() -> None:
    report = run_sleep_plasticity_ticket_queue_benchmark(retained_count=64, runs=2)

    assert report["pass"] is True
    assert report["quality"]["sleep_latest_verified_matches_expected_seed"] is True
    assert report["quality"]["design_latest_verified_matches_expected_seed"] is True
    assert report["retired_full_retained_ticket_queue_absence"][
        "implementation_present"
    ] is False
    assert report["retired_full_retained_ticket_queue_absence"][
        "diagnostic_callable"
    ] is False
    assert "diagnostic" not in report
    assert "diagnostic_full_sleep_review_ticket_queue" not in report["latency_ms"]
    assert (
        "diagnostic_full_scheduler_design_review_ticket_queue"
        not in report["latency_ms"]
    )
    sleep_window = report["source_window"]["sleep_review_ticket_queue"]
    design_window = report["source_window"]["scheduler_design_review_ticket_queue"]
    assert sleep_window["runs_live_tick"] is False
    assert sleep_window["runs_every_token"] is False
    assert sleep_window["global_candidate_scan"] is False
    assert sleep_window["global_score_scan"] is False
    assert design_window["runs_live_tick"] is False
    assert design_window["runs_every_token"] is False
    assert design_window["global_candidate_scan"] is False
    assert design_window["global_score_scan"] is False
