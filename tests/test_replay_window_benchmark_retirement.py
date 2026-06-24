from __future__ import annotations

import argparse
from pathlib import Path

from marulho.evaluation.snn_readout_replay_priority_source_window_benchmark import (
    run_benchmark as run_readout_replay_priority_benchmark,
)
from marulho.evaluation.snn_replay_priority_source_window_benchmark import (
    run_benchmark as run_snn_replay_priority_benchmark,
)
from marulho.evaluation.snn_replay_artifact_provenance_source_window_benchmark import (
    run_benchmark as run_snn_replay_artifact_provenance_benchmark,
)
from marulho.evaluation.status_transition_memory_source_window_benchmark import (
    run_benchmark as run_status_transition_memory_benchmark,
)
from marulho.evaluation.synapse_provenance_audit_source_window_benchmark import (
    run_benchmark as run_synapse_provenance_audit_benchmark,
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
from marulho.evaluation.live_memory_summary_projection_benchmark import (
    run_benchmark as run_live_memory_summary_projection_benchmark,
)
from marulho.evaluation.snn_readout_ledger_normalization_source_window_benchmark import (
    run_benchmark as run_snn_readout_ledger_normalization_benchmark,
)
from marulho.evaluation.concept_signature_lookup_benchmark import (
    run_benchmark as run_concept_signature_lookup_benchmark,
)
from marulho.evaluation.concept_frontier_scope_benchmark import (
    run_benchmark as run_concept_frontier_scope_benchmark,
)
from marulho.evaluation.frontier_gap_bounded_benchmark import (
    run as run_frontier_gap_bounded_benchmark,
)
from marulho.evaluation.bucket_candidate_source_window_benchmark import (
    run_benchmark as run_bucket_candidate_source_window_benchmark,
)
from marulho.evaluation.sfa_sample_scope_benchmark import (
    run_benchmark as run_sfa_sample_scope_benchmark,
)
from marulho.evaluation.awake_ripple_scope_benchmark import (
    run_awake_ripple_scope_benchmark,
)
from marulho.evaluation.context_memory_match_benchmark import (
    run_benchmark as run_context_memory_match_benchmark,
)
from marulho.evaluation.query_memory_payload_benchmark import (
    run_benchmark as run_query_memory_payload_benchmark,
)
from marulho.evaluation.runtime_concept_memory_lookup_benchmark import (
    run_benchmark as run_runtime_concept_memory_lookup_benchmark,
)
from marulho.evaluation.query_episode_readout_benchmark import (
    run_benchmark as run_query_episode_readout_benchmark,
)
from marulho.evaluation.slow_memory_fixed_cadence_retirement_benchmark import (
    run_benchmark as run_slow_memory_fixed_cadence_benchmark,
)
from marulho.evaluation.strong_capture_admission_cadence_benchmark import (
    run_benchmark as run_strong_capture_admission_cadence_benchmark,
)
from marulho.evaluation.selected_replay_consolidation_cache_benchmark import (
    run_benchmark as run_selected_replay_consolidation_cache_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]


def test_bounded_replay_window_benchmarks_do_not_keep_retired_baselines() -> None:
    benchmark_paths = [
        ROOT / "src/marulho/evaluation/source_bank_memory_match_benchmark.py",
        ROOT / "src/marulho/evaluation/snn_readout_replay_priority_source_window_benchmark.py",
        ROOT / "src/marulho/evaluation/snn_replay_priority_source_window_benchmark.py",
        ROOT
        / "src/marulho/evaluation/snn_replay_artifact_provenance_source_window_benchmark.py",
        ROOT / "src/marulho/evaluation/status_transition_memory_source_window_benchmark.py",
        ROOT
        / "src/marulho/evaluation/synapse_provenance_audit_source_window_benchmark.py",
        ROOT / "src/marulho/evaluation/snn_emission_review_replay_policy_source_window_benchmark.py",
        ROOT / "src/marulho/evaluation/snn_rollout_rehearsal_source_window_benchmark.py",
        ROOT / "src/marulho/evaluation/status_replay_path_source_window_benchmark.py",
        ROOT / "src/marulho/evaluation/sleep_plasticity_ticket_queue_source_window_benchmark.py",
        ROOT
        / "src/marulho/evaluation/applied_replay_lineage_checkpoint_summary_benchmark.py",
        ROOT / "src/marulho/evaluation/sleep_replay_routing_index_refresh_benchmark.py",
        ROOT / "src/marulho/evaluation/bucket_consolidation_cache_lookup_benchmark.py",
        ROOT / "src/marulho/evaluation/live_memory_summary_projection_benchmark.py",
        ROOT
        / "src/marulho/evaluation/snn_readout_ledger_normalization_source_window_benchmark.py",
        ROOT / "src/marulho/evaluation/concept_signature_lookup_benchmark.py",
        ROOT / "src/marulho/evaluation/concept_frontier_scope_benchmark.py",
        ROOT / "src/marulho/evaluation/frontier_gap_bounded_benchmark.py",
        ROOT / "src/marulho/evaluation/bucket_candidate_source_window_benchmark.py",
        ROOT / "src/marulho/evaluation/sfa_sample_scope_benchmark.py",
        ROOT / "src/marulho/evaluation/awake_ripple_scope_benchmark.py",
        ROOT / "src/marulho/evaluation/context_memory_match_benchmark.py",
        ROOT / "src/marulho/evaluation/query_memory_payload_benchmark.py",
        ROOT / "src/marulho/evaluation/runtime_concept_memory_lookup_benchmark.py",
        ROOT / "src/marulho/evaluation/query_episode_readout_benchmark.py",
        ROOT
        / "src/marulho/evaluation/slow_memory_fixed_cadence_retirement_benchmark.py",
        ROOT
        / "src/marulho/evaluation/strong_capture_admission_cadence_benchmark.py",
        ROOT
        / "src/marulho/evaluation/selected_replay_consolidation_cache_benchmark.py",
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
        "\"full_summary_stats\":",
        "full_path_scans_entries",
        "full_summary_scan_entry_count",
        "_legacy_signature",
        "legacy_archive_materializing_lookup",
        "_legacy_full_scan_metrics",
        "legacy_full_scan",
        "diagnostic_full_slow_memory_scan",
        "_diagnostic_legacy_frontier_terms",
        "term_recall_against_legacy",
        "archive_window_materialization_count",
        "side_list_materialization_count",
        "_legacy_materialized_candidates",
        "legacy_hot_bucket_candidate_materialization",
        "_legacy_full_buffer_sfa_sample",
        "retired_global_full_buffer_sfa_sample",
        "_legacy_global_awake_ripple_scan",
        "legacy_global_baseline",
        "global_unscoped",
        "retired_scalar_full_memory_scan",
        "retired_vector_full_memory_scan",
        "_run_report_dropping_context",
        "legacy_selected_by_context",
        "legacy_raw_text_payload",
        "_diagnostic_eager_payload",
        "diagnostic_eager_query_payload",
        "_legacy_direct_runtime_concept_lookup",
        "retired_service_direct_runtime_concept_lookup",
        "legacy_report",
        "legacy_target_hit",
        "legacy_top_text",
        "_legacy_full_materialized_normalized_state",
        "_legacy_full_materialized_store_state",
        "def _broad_normalized",
        "bounded_speedup_vs_broad_normalized",
        "bounded_speedup_vs_legacy",
        "\"broad_normalized\"",
        "def _retired_broad_projection",
        "retired_broad_scan",
        "bounded_speedup_vs_retired",
        "retired_rows_read_total",
        "retired_python_peak_mib",
        "_diagnostic_full_applied_synapse_audit",
        "diagnostic_full_applied_synapse_scan",
        "diagnostic_full_applied_synapse_provenance_audit_scan",
        "_retired_fixed_cadence_archive_tokens",
        "retired_fixed_cadence_projection",
        "_retired_every_strong_projection",
        "retired_every_strong_projection",
        "retired_every_strong_projection_matches_forced_candidates",
        "_run_retired_diagnostic",
        "selected_state_matches_retired_diagnostic",
        "retired_diagnostic_full_cache_rebuild_then_replay",
        "retired_diagnostic_report",
    ]
    for path in benchmark_paths:
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source


def test_snn_replay_priority_benchmark_reports_maintained_only_path() -> None:
    report = run_snn_replay_priority_benchmark(
        argparse.Namespace(retention_count=64, limit=4, runs=2, output=None)
    )

    assert report["pass"] is True
    assert report["quality"]["old_readout_target_selected"] is True
    assert "retired_path_comparison" not in report
    memory_budget = report["memory_budget"]
    assert memory_budget["retained_context_count"] == 64
    assert 0 < memory_budget["bounded_verified_context_count"] < 64
    assert memory_budget["archival_storage_device"] == "cpu"
    assert memory_budget["runs_live_tick"] is False
    assert memory_budget["runs_every_token"] is False
    assert memory_budget["global_candidate_scan"] is False
    assert memory_budget["global_score_scan"] is False
    assert memory_budget["language_reasoning"] is False


def test_snn_replay_artifact_provenance_benchmark_reports_maintained_only_path() -> None:
    report = run_snn_replay_artifact_provenance_benchmark(
        argparse.Namespace(retention_count=64, runs=2, output=None)
    )

    assert report["pass"] is True
    assert report["quality"]["old_permit_verified"] is True
    assert "retired_path_comparison" not in report
    memory_budget = report["memory_budget"]
    assert memory_budget["retention_count_per_artifact_family"] == 64
    assert memory_budget["source_record_count"] == 4
    assert memory_budget["index_lookup_count"] == 4
    assert memory_budget["index_hit_count"] == 4
    assert memory_budget["archival_storage_device"] == "cpu"
    assert memory_budget["runs_live_tick"] is False
    assert memory_budget["runs_every_token"] is False
    assert memory_budget["global_candidate_scan"] is False
    assert memory_budget["global_score_scan"] is False
    assert memory_budget["language_reasoning"] is False
    assert (
        memory_budget["raw_caller_window_artifact_recording"][
            "public_raw_recorder_callable"
        ]
        is False
    )


def test_status_transition_memory_benchmark_reports_maintained_only_path() -> None:
    report = run_status_transition_memory_benchmark(
        argparse.Namespace(entry_count=128, runs=2, output=None)
    )

    assert report["pass"] is True
    assert report["quality"]["quality_gate_passed"] is True
    assert "retired_path_comparison" not in report
    assert "retired_broad_scan" not in report["latency"]
    assert "bounded_speedup_vs_retired" not in report["latency"]
    assert "retired_rows_read_total" not in report["work"]
    assert "retired_row_reads" not in report["work"]
    assert report["removed_broad_projection_absence"] == {
        "implementation_present": False,
        "active_report_field_present": False,
        "removed_policy": (
            "status_read_model_materialized_transition_memory_maps_per_projection"
        ),
    }
    work = report["work"]
    assert work["retained_sparse_transition_weight_count"] == 128
    assert work["retained_synapse_provenance_count"] == 128
    assert work["bounded_rows_read_total"] > 0
    assert work["archival_storage_device"] == "cpu"
    assert work["runs_live_tick"] is False
    assert work["runs_every_token"] is False
    assert work["global_candidate_scan"] is False
    assert work["global_score_scan"] is False
    assert work["language_reasoning"] is False


def test_synapse_provenance_audit_benchmark_reports_maintained_only_path() -> None:
    report = run_synapse_provenance_audit_benchmark(
        argparse.Namespace(entry_count=128, runs=2, output=None)
    )

    assert report["pass"] is True
    assert report["quality"]["source_window_keys_match"] is True
    assert "diagnostic" not in report
    assert "work_reduction" not in report
    assert "diagnostic_full_applied_synapse_scan" not in report["latency_ms"]
    assert report["retired_full_applied_synapse_audit_scan_absence"] == {
        "implementation_present": False,
        "active_report_field_present": False,
        "removed_policy": "full_applied_synapse_audit_scan",
    }
    budget = report["memory_budget"]
    assert budget["bounded_sparse_transition_weight_rows"] == 64
    assert budget["bounded_synapse_provenance_rows"] == 64
    assert budget["bounded_source_rows_total"] == 128
    assert budget["projected_full_scan_rows_removed"] == 128
    assert budget["archival_storage_device"] == "cpu"
    assert budget["runs_live_tick"] is False
    assert budget["runs_every_token"] is False
    assert budget["global_candidate_scan"] is False
    assert budget["global_score_scan"] is False
    assert budget["language_reasoning"] is False
    assert report["source_window"]["source_payload_truncated"] is True
    assert report["source_window"]["gpu_resident_archival_metadata"] is False


def test_slow_memory_fixed_cadence_benchmark_reports_maintained_only_path() -> None:
    report = run_slow_memory_fixed_cadence_benchmark(
        tokens=32,
        archive_interval_tokens=8,
        runs=1,
        seed=20260624,
        device="cpu",
    )

    assert report["pass"] is True
    assert "retired_fixed_cadence_projection" not in report
    assert report["retired_fixed_cadence_admission_absence"] == {
        "implementation_present": False,
        "active_report_field_present": False,
        "removed_policy": "fixed_cadence_slow_memory_admission_projection",
    }
    assert report["quality"]["fixed_cadence_projection_removed"] is True
    assert report["memory_budget"]["fixed_cadence_archive_writes"] == 0
    assert report["memory_budget"]["first_token_archive_writes"] == 1
    assert report["memory_budget"]["projected_fixed_cadence_writes_removed"] == 4
    assert report["memory_budget"]["archival_storage_device"] == "cpu"
    assert report["runtime_truth"]["runs_every_token"] is False
    assert report["runtime_truth"]["slow_memory_admission_every_token"] is False
    assert report["runtime_truth"]["hidden_language_reasoning"] is False


def test_strong_capture_admission_benchmark_reports_maintained_only_path() -> None:
    report = run_strong_capture_admission_cadence_benchmark(
        tokens=32,
        min_interval_tokens=8,
        runs=1,
        seed=20260624,
        device="cpu",
    )

    assert report["pass"] is True
    assert "retired_every_strong_projection" not in report
    assert report["retired_every_strong_admission_absence"] == {
        "implementation_present": False,
        "active_report_field_present": False,
        "removed_policy": "every_strong_slow_memory_admission_projection",
    }
    assert report["quality"]["every_strong_projection_removed"] is True
    assert report["memory_budget"]["max_archive_writes_per_strong_interval"] == 1
    assert report["memory_budget"]["strong_capture_min_interval_tokens"] == 8
    assert report["memory_budget"]["projected_every_strong_archive_writes_removed"] > 0
    assert report["memory_budget"]["archival_storage_device"] == "cpu"
    assert report["runtime_truth"]["runs_every_token"] is False
    assert report["runtime_truth"]["slow_memory_admission_every_token"] is False
    assert report["runtime_truth"]["hidden_language_reasoning"] is False


def test_selected_replay_consolidation_cache_benchmark_reports_maintained_only_path() -> None:
    report = run_selected_replay_consolidation_cache_benchmark(
        entries=128,
        buckets=32,
        selected_count=8,
        runs=2,
    )

    assert report["pass"] is True
    assert "retired_diagnostic_report" not in report
    assert "retired_diagnostic_full_cache_rebuild_then_replay" not in report["latency"]
    assert report["retired_full_cache_rebuild_diagnostic_absence"] == {
        "implementation_present": False,
        "active_report_field_present": False,
        "removed_policy": "selected_replay_full_bucket_cache_rebuild_diagnostic",
    }
    quality = report["quality"]
    assert quality["metric"] == "seeded_selected_replay_consolidation_update_integrity"
    assert quality["selected_replay_counts_incremented"] is True
    assert quality["selected_consolidation_events_incremented"] is True
    assert quality["selected_consolidation_levels_above_seeded_initial"] is True
    assert quality["selected_fast_ema_matches_expected_centroid"] is True
    assert quality["retired_full_cache_rebuild_diagnostic_removed"] is True
    budget = report["memory_budget"]
    assert budget["selected_window_indices"] == 8
    assert budget["retained_entries"] == 128
    assert budget["bounded_rebuild_scan_entries"] == 0
    assert budget["projected_full_cache_rebuild_entries_removed"] == 128
    assert budget["archival_storage_device"] == "cpu"
    assert report["runtime_truth"]["full_memory_scan"] is False
    assert report["runtime_truth"]["runs_live_tick"] is False
    assert report["runtime_truth"]["runs_every_token"] is False
    assert report["runtime_truth"]["hidden_language_reasoning"] is False


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


def test_live_memory_summary_projection_benchmark_reports_maintained_only_path() -> None:
    report = run_live_memory_summary_projection_benchmark(
        entries=128,
        dim=8,
        runs=2,
        seed=20260621,
    )

    assert report["pass"] is True
    assert report["quality"]["scalar_fill_exact"] is True
    assert report["quality"]["live_projection_read_only"] is True
    assert report["quality"]["live_scan_count_zero"] is True
    assert report["quality"]["live_projection_has_expected_marker"] is True
    assert report["retired_live_full_summary_scan_absence"] == {
        "implementation_present": False,
        "diagnostic_callable": False,
        "active_report_field_present": False,
        "removed_policy": "live_projection_full_summary_stats_scan_comparator",
    }
    assert "full_summary_stats" not in report["latency"]
    assert report["runtime_truth"]["live_summary_full_memory_scan"] is False
    assert report["runtime_truth"]["live_summary_scan_entry_count"] == 0
    assert report["memory_budget"]["retired_live_full_summary_scan_rows_removed"] == 128
    assert report["resource_behavior"]["python_tracemalloc_peak_mib"] >= 0.0


def test_snn_readout_ledger_normalization_benchmark_reports_maintained_only_path() -> None:
    report = run_snn_readout_ledger_normalization_benchmark(
        argparse.Namespace(retention_count=64, ledger_limit=8, runs=2, output=None)
    )

    assert report["pass"] is True
    assert report["retired_full_materialized_ledger_normalization_absence"] == {
        "implementation_present": False,
        "active_report_field_present": False,
        "removed_policy": (
            "snn_readout_ledger_full_materialized_normalization_comparator"
        ),
    }
    assert report["retired_broad_normalized_ledger_comparator_absence"] == {
        "implementation_present": False,
        "active_report_field_present": False,
        "removed_policy": "snn_readout_ledger_broad_normalized_boundary_comparators",
    }
    assert "retired_path_comparison" not in report
    assert "legacy" not in report["latency"]
    assert "broad_normalized" not in report["latency"]
    assert "bounded_speedup_vs_legacy" not in report["latency"]
    assert "bounded_speedup_vs_broad_normalized" not in report["latency"]
    expected_rows = (
        report["input"]["event_field_count"]
        * report["input"]["retention_count_per_field"]
    )
    bounded_rows = report["memory_budget"]["bounded_window_rows"]
    assert report["memory_budget"]["source_rows_known"] == expected_rows
    assert report["memory_budget"]["retired_full_materialized_rows_removed"] == (
        expected_rows - bounded_rows
    )
    assert report["memory_budget"]["runs_live_tick"] is False
    assert report["memory_budget"]["runs_every_token"] is False
    assert report["memory_budget"]["global_candidate_scan"] is False
    assert report["memory_budget"]["global_score_scan"] is False
    assert report["memory_budget"]["language_reasoning"] is False
    assert report["normalization_source_window"]["archival_storage_device"] == "cpu"
    assert report["normalization_source_window"]["gpu_used"] is False


def test_concept_signature_lookup_benchmark_reports_maintained_only_path() -> None:
    report = run_concept_signature_lookup_benchmark(
        capacity=128,
        dim=8,
        iterations=4,
        group_size=4,
        seed=20260623,
    )

    assert report["passed"] is True
    assert report["quality"]["seeded_expected_signature_matches"] is True
    assert report["quality"]["min"] >= 0.9999
    assert report["retired_archive_materializing_signature_lookup_absence"] == {
        "implementation_present": False,
        "diagnostic_callable": False,
        "active_report_field_present": False,
        "removed_policy": "concept_signature_lookup_archive_materializing_comparator",
    }
    assert "legacy_archive_materializing_lookup" not in report
    assert "legacy_mean_latency_ms" not in report["latency"]
    bounded = report["bounded_direct_index_lookup"]
    assert bounded["archive_list_materialization_count"] == 0
    assert bounded["global_candidate_scan"] is False
    assert bounded["global_score_scan"] is False
    assert report["memory_budget"]["retired_archive_materializing_lookup_rows_removed"] == 128
    assert report["device_placement"]["archival_storage_device"] == "cpu"
    assert report["resource_behavior"]["python_tracemalloc_peak_mib"] >= 0.0


def test_concept_frontier_scope_benchmark_reports_maintained_only_path() -> None:
    report = run_concept_frontier_scope_benchmark(
        capacity=256,
        bucket_count=64,
        candidate_bucket_count=8,
        probe_count=16,
        dim=8,
        iterations=2,
        seed=20260623,
    )

    assert report["passed"] is True
    assert all(bool(value) for value in report["gates"].values())
    assert report["quality"]["top1_target_match"] is True
    assert report["quality"]["target_hit_rate"] == 1.0
    assert report["retired_concept_frontier_full_scan_absence"] == {
        "implementation_present": False,
        "diagnostic_callable": False,
        "active_report_field_present": False,
        "removed_policy": "concept_frontier_metrics_full_slow_memory_scan_comparator",
    }
    assert "legacy_full_scan" not in report
    assert "speedup" not in report["latency"]
    bounded = report["bounded_candidate_window"]
    assert bounded["score_count"] <= report["memory_budget"]["candidate_window_limit"]
    assert bounded["global_candidate_scan"] is False
    assert bounded["global_score_scan"] is False
    assert bounded["runs_live_tick"] is False
    assert bounded["frontier_row_reader_owned_by_store"] is True
    assert bounded["direct_slow_memory_array_reads_retired"] is True
    assert report["device_placement"]["archival_storage_device"] == "cpu"
    assert report["resource_behavior"]["python_tracemalloc_peak_mib"] >= 0.0


def test_frontier_gap_benchmark_reports_maintained_only_path() -> None:
    report = run_frontier_gap_bounded_benchmark(
        argparse.Namespace(
            output=None,
            capacity=512,
            iterations=2,
            top_entries=8,
            max_terms=4,
            min_quality=1.0,
        )
    )

    assert report["passed"] is True
    assert all(bool(value) for value in report["gates"].values())
    assert report["quality"]["expected_term_recall"] == 1.0
    assert report["retired_frontier_full_archive_scan_absence"] == {
        "implementation_present": False,
        "diagnostic_callable": False,
        "active_report_field_present": False,
        "removed_policy": "frontier_gap_full_archive_raw_window_scan_comparator",
    }
    assert "legacy" not in report
    assert "legacy_mean" not in report["latency_ms"]
    bounded = report["bounded_selection_report"]
    assert bounded["candidate_index_count"] <= report["memory_budget"]["candidate_window_limit"]
    assert bounded["global_candidate_scan"] is False
    assert bounded["global_score_scan"] is False
    assert bounded["runs_live_tick"] is False
    assert bounded["language_reasoning"] is False
    assert bounded["frontier_row_reader_owned_by_store"] is True
    assert report["missing_collector_gate"]["passed"] is True
    assert report["device"]["archival_storage_device"] == "cpu"
    assert report["resource_behavior"]["python_tracemalloc_peak_mib"] >= 0.0


def test_bucket_candidate_source_window_benchmark_reports_maintained_only_path(
    tmp_path: Path,
) -> None:
    report = run_bucket_candidate_source_window_benchmark(
        output=tmp_path / "bucket-candidate.json",
        archive_size=128,
        candidate_limit=8,
        iterations=2,
        bucket_id=7,
    )

    assert report["status"] == "passed"
    assert report["expected_recent_candidate_indices"] == list(range(127, 119, -1))
    assert report["retired_hot_bucket_materialization_absence"] == {
        "implementation_present": False,
        "diagnostic_callable": False,
        "active_report_field_present": False,
        "removed_policy": "hot_bucket_candidate_source_full_materialization_comparator",
    }
    assert "legacy" not in report
    assert report["memory_budget"]["retired_full_bucket_materialization_rows_removed"] == 128
    bounded = report["bounded"]["result"]
    assert bounded["match_indices"] == report["expected_recent_candidate_indices"]
    assert bounded["candidate_source_entry_read_count"] <= 8
    assert bounded["candidate_source_full_bucket_scan"] is False
    assert bounded["candidate_source_full_bucket_materialization"] is False
    assert report["device"]["archival_storage_device"] == "cpu"
    assert report["device"]["cuda_memory_delta_mib"] == 0.0


def test_sfa_sample_scope_benchmark_reports_maintained_only_path() -> None:
    report = run_sfa_sample_scope_benchmark(
        argparse.Namespace(
            capacity=128,
            vector_dim=16,
            candidate_count=16,
            sample_count=8,
            iterations=2,
            seed=17,
            min_selected_window_purity=1.0,
            max_bounded_mean_ms=10.0,
        )
    )

    assert report["passed"] is True
    assert report["retired_full_buffer_sfa_sample_absence"] == {
        "implementation_present": False,
        "diagnostic_callable": False,
        "active_report_field_present": False,
        "removed_policy": "sfa_full_buffer_sample_comparator",
    }
    assert "legacy_report" not in report
    assert report["quality"]["bounded_mean"] == 1.0
    assert set(report["quality"]["selected_sample_indices"]).issubset(set(range(16)))
    bounded = report["bounded_report"]
    assert bounded["surface"] == "bounded_sfa_sample.v1"
    assert bounded["candidate_scope"] == "selected_replay_window"
    assert bounded["global_candidate_scan"] is False
    assert bounded["runs_live_tick"] is False
    assert bounded["language_reasoning"] is False
    assert report["device_placement"]["archival_storage_device"] == "cpu"
    assert report["device_placement"]["cuda_memory_delta_mib"] == 0.0


def test_awake_ripple_scope_benchmark_reports_maintained_only_path(
    tmp_path: Path,
) -> None:
    report = run_awake_ripple_scope_benchmark(
        output_path=tmp_path / "awake-ripple.json",
        capacity=256,
        bucket_count=256,
        awake_bucket_count=10,
        iterations=4,
        dim=8,
    )

    assert report["passed"] is True
    assert report["retired_global_awake_ripple_scan_absence"] == {
        "implementation_present": False,
        "diagnostic_callable": False,
        "active_report_field_present": False,
        "removed_policy": "awake_ripple_scalar_vector_full_memory_scan_comparator",
    }
    assert "global_unscoped" not in report
    assert report["memory_budget"]["retired_full_scan_rows_removed"] == 256
    scoped = report["wake_bucket_scoped"]
    assert scoped["ripple_scalar_scan_count"] == 0
    assert scoped["ripple_vector_scan_count"] == 0
    assert scoped["ripple_awake_bucket_scan_count"] == 4
    assert scoped["last_ripple_awake_candidate_count"] <= 10
    assert all(bool(value) for value in report["gates"].values())


def test_context_memory_match_benchmark_reports_maintained_only_path() -> None:
    report = run_context_memory_match_benchmark(
        argparse.Namespace(
            capacity=128,
            bucket_count=8,
            candidate_limit=16,
            top_k=4,
            context_count=2,
            text_repeats=2,
            iterations=2,
            max_bounded_mean_ms=500.0,
        )
    )

    assert report["passed"] is True
    assert report["retired_report_dropping_context_absence"] == {
        "implementation_present": False,
        "diagnostic_callable": False,
        "active_report_field_present": False,
        "removed_policy": "context_memory_match_report_dropping_comparator",
    }
    assert "legacy_report" not in report
    assert report["quality"]["selected_indices_consistent"] is True
    assert report["quality"]["selected_indices_inside_candidate_window"] is True
    aggregate = report["aggregate_report"]
    assert aggregate["surface"] == "bounded_context_comparison_memory_match.v1"
    assert aggregate["global_candidate_scan"] is False
    assert aggregate["global_score_scan"] is False
    assert aggregate["runs_live_tick"] is False
    assert aggregate["language_reasoning"] is False
    assert report["payload"]["aggregate_raw_text_payload_count"] <= 4


def test_query_memory_payload_benchmark_reports_maintained_only_path() -> None:
    report = run_query_memory_payload_benchmark(
        capacity=128,
        bucket_count=8,
        candidate_limit=16,
        top_k=4,
        iterations=2,
    )

    assert report["passed"] is True
    assert report["retired_eager_query_payload_absence"] == {
        "implementation_present": False,
        "diagnostic_callable": False,
        "active_report_field_present": False,
        "removed_policy": "query_memory_eager_candidate_text_payload_comparator",
    }
    assert "legacy_report" not in report
    assert report["quality"]["selected_indices_inside_candidate_window"] is True
    assert report["payload"]["bounded_report_raw_text_payload_count"] <= 4
    bounded = report["bounded_report"]
    assert bounded["raw_text_payload_policy"] == "returned_similarity_matches_only"
    assert bounded["query_row_surface"] == "bounded_query_memory_match_row.v1"
    assert bounded["global_candidate_scan"] is False
    assert bounded["global_score_scan"] is False
    assert bounded["language_reasoning"] is False


def test_runtime_concept_memory_lookup_benchmark_reports_maintained_only_path() -> None:
    report = run_runtime_concept_memory_lookup_benchmark(
        argparse.Namespace(
            capacity=128,
            dim=8,
            observation_count=16,
            unique_indices=4,
            max_observations=16,
            text_repeats=2,
            iterations=2,
            max_bounded_mean_ms=500.0,
        )
    )

    assert report["passed"] is True
    assert report["retired_direct_runtime_concept_lookup_absence"] == {
        "implementation_present": False,
        "diagnostic_callable": False,
        "active_report_field_present": False,
        "removed_policy": "runtime_concept_direct_archive_lookup_comparator",
    }
    assert "legacy_report" not in report
    assert report["quality"]["selected_indices_match_expected"] is True
    assert report["quality"]["expected_indices"] == [0, 32, 64, 96]
    bounded = report["bounded_report"]
    assert bounded["candidate_scope"] == "train_step_memory_index_evidence"
    assert bounded["global_candidate_scan"] is False
    assert bounded["global_score_scan"] is False
    assert bounded["language_reasoning"] is False
    assert bounded["archival_storage_device"] == "cpu"


def test_query_episode_readout_benchmark_reports_maintained_only_path() -> None:
    report = run_query_episode_readout_benchmark(
        capacity=128,
        iterations=4,
        neighbor_radius=3,
        max_bounded_mean_ms=10.0,
    )

    assert report["passed"] is True
    assert report["retired_fragment_only_episode_readout_absence"] == {
        "implementation_present": False,
        "diagnostic_callable": False,
        "active_report_field_present": False,
        "removed_policy": "query_episode_fragment_only_readout_comparator",
    }
    assert "legacy_top_text" not in report["quality"]
    assert report["quality"]["bounded_target_hit"] is True
    assert "cat purrs when it feels safe" in report["quality"]["bounded_top_text"]
    bounded = report["bounded_report"]
    assert bounded["raw_text_payload_policy"] == "selected_match_neighbor_windows_only"
    assert bounded["global_candidate_scan"] is False
    assert bounded["global_score_scan"] is False
    assert bounded["runs_live_tick"] is False
    assert bounded["language_reasoning"] is False
