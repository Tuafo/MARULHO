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
from marulho.evaluation.sleep_replay_routing_index_refresh_benchmark import (
    run_benchmark as run_sleep_replay_routing_index_refresh_benchmark,
)
from marulho.evaluation.status_replay_path_source_window_benchmark import (
    run_benchmark as run_status_replay_path_benchmark,
)
from marulho.evaluation.applied_replay_lineage_checkpoint_summary_benchmark import (
    run_benchmark as run_applied_replay_lineage_checkpoint_summary_benchmark,
)
from marulho.evaluation.bucket_consolidation_cache_lookup_benchmark import (
    run_benchmark as run_bucket_consolidation_cache_lookup_benchmark,
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
        ROOT
        / "src/marulho/evaluation/applied_replay_lineage_checkpoint_summary_benchmark.py",
        ROOT / "src/marulho/evaluation/sleep_replay_routing_index_refresh_benchmark.py",
        ROOT / "src/marulho/evaluation/bucket_consolidation_cache_lookup_benchmark.py",
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
        "_retired_full_scan_summary",
        "retired_applied_replay_lineage_full_scan_summary",
        "counts_match_retired_diagnostic",
        "active_speedup_vs_retired",
        "diagnostic_full_retained_sleep_plasticity_review_ticket_queue",
        "diagnostic_full_retained_sleep_plasticity_scheduler_design",
        "sleep_latest_verified_matches_diagnostic",
        "design_latest_verified_matches_diagnostic",
        "legacy_samples",
        "legacy_mean",
        "bounded_speedup_vs_legacy",
        "retired_per_probe_query_match_call_count",
        "retired_path_comparison",
        "def _full()",
        "\"retired_full_rebuild\":",
        "last_full_rebuild",
        "full_top1_matches",
        "full_path_rebuilt",
        "latency_reduction_mean_ms",
        "full_mean_ms",
        "_retired_scan_level",
        "\"retired_full_bucket_scan\":",
        "cached_value_matches_retired_scan",
        "retired_mean_ms",
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


def test_applied_replay_lineage_benchmark_reports_maintained_only_path() -> None:
    report = run_applied_replay_lineage_checkpoint_summary_benchmark(
        argparse.Namespace(entry_count=64, runs=2, output=None)
    )

    assert report["pass"] is True
    assert report["quality"]["lineage_incremental_summary_parity"] is True
    assert report["quality"]["source_reads_eliminated"] is True
    assert report["work"]["active_source_reads"] == 0
    assert report["work"]["source_reads_removed"] == "all_provenance_reads_eliminated"
    assert report["retired_full_scan_absence"] == {
        "implementation_present": False,
        "diagnostic_callable": False,
        "active_report_field_present": False,
        "removed_policy": (
            "runtime_persistence_checkpoint_summary_full_synapse_provenance_scan"
        ),
    }
    assert "retired_path_comparison" not in report
    assert "retired_full_scan" not in report["latency"]
    assert "retired_python_peak_mib" not in report["resource_behavior"]
    assert report["active_evidence"]["full_provenance_scan"] is False
    assert report["active_evidence"]["source_record_scan_count"] == 0
    assert report["active_evidence"]["language_reasoning"] is False


def test_sleep_replay_routing_index_refresh_benchmark_reports_maintained_only_path() -> None:
    report = run_sleep_replay_routing_index_refresh_benchmark(
        entries=256,
        dim=16,
        update_count=8,
        missing_update_count=1,
        runs=2,
        shards=1,
        seed=20260621,
        device_name="cpu",
    )

    assert report["pass"] is True
    assert report["quality"]["bounded_top1_matches"] is True
    assert report["quality"]["bounded_no_full_rebuild"] is True
    assert report["quality"]["bounded_deferred_missing_recovery"] is True
    assert report["retired_full_rebuild_absence"] == {
        "implementation_present": False,
        "diagnostic_callable": False,
        "active_report_field_present": False,
        "removed_policy": "selected_sleep_replay_full_routing_index_rebuild",
    }
    assert "retired_full_rebuild" not in report["latency"]
    assert "last_full_rebuild" not in report
    assert report["resource_behavior"]["python_tracemalloc_peak_mib"] >= 0.0
    assert (
        report["resource_behavior"]["cuda_memory_allocated_after_mib"]
        == report["device_placement"]["cuda_memory_allocated_after_mib"]
    )
    assert report["runtime_truth"]["routing_index_full_rebuild"] is False
    assert report["runtime_truth"]["runs_live_tick"] is False
    assert report["runtime_truth"]["runs_every_token"] is False


def test_bucket_consolidation_cache_lookup_benchmark_reports_maintained_only_path() -> None:
    report = run_bucket_consolidation_cache_lookup_benchmark(
        entries=128,
        buckets=128,
        bucket_id=7,
        runs=2,
    )

    assert report["pass"] is True
    assert report["quality"]["cached_value_matches_seeded_bucket"] is True
    assert report["quality"]["cached_lookup_no_scan"] is True
    assert report["retired_full_bucket_scan_absence"] == {
        "implementation_present": False,
        "diagnostic_callable": False,
        "active_report_field_present": False,
        "removed_policy": "scalar_bucket_consolidation_full_slow_memory_scan",
    }
    assert "retired_full_bucket_scan" not in report["latency"]
    assert report["memory_budget"]["seeded_expected_bucket_rows"] == 1
    assert report["runtime_truth"]["full_memory_scan"] is False
    assert report["runtime_truth"]["scan_entry_count"] == 0
    assert report["resource_behavior"]["python_tracemalloc_peak_mib"] >= 0.0
