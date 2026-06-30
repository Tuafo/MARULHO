from __future__ import annotations

from marulho.evaluation.column_scheduler_benchmark import run_benchmark, run_scaling_benchmark


def test_column_scheduler_benchmark_reports_bounded_predictive_vote() -> None:
    report = run_benchmark(
        n_columns=16,
        column_latent_dim=8,
        k_routing=4,
        samples=3,
        warmup_steps=1,
        seed=123,
        device="cpu",
    )

    scoped = report["scoped_cached_vote"]

    assert report["surface"] == "column_scheduler_benchmark.v2"
    assert report["scope"] == "complete_train_step_deep_sleep_pressure_usefulness_filter_predictive_update_vote_and_structural_review_queue_awake_mask_maintained_only"
    assert "all_column_vote" not in report
    assert "winner_sequence_equal" not in report
    assert "median_delta_percent" not in report
    assert "mean_delta_percent" not in report
    assert "neutral_or_better_complete_tick" not in report
    assert report["awake_count_bounded"] is True
    assert report["column_wake_plan_bounded"] is True
    assert report["predictive_vote_bounded"] is True
    assert report["predictive_update_bounded"] is True
    assert report["predictive_location_update_bounded"] is True
    assert report["column_metabolism_bounded"] is True
    assert report["structural_review_bounded"] is True
    assert report["structural_review_continuation_reviewed"] is True
    assert report["candidate_sleep_filter_bounded"] is True
    assert report["bounded_specialist_work"] is True
    assert report["scoped_runs_all_columns"] is False
    assert scoped["predictive_vote_updated_columns"] == 4
    assert scoped["predictive_vote_cached_columns"] == 12
    assert scoped["predictive_vote_runs_all_columns"] is False
    assert scoped["predictive_update_updated_columns"] == 4
    assert scoped["predictive_update_cached_columns"] == 12
    assert scoped["predictive_update_runs_all_columns"] is False
    assert scoped["predictive_location_update_columns"] == 4
    assert scoped["predictive_location_cached_columns"] == 12
    assert scoped["predictive_location_runs_all_columns"] is False
    assert scoped["candidate_sleep_filter_output_candidates"] == 4
    assert scoped["candidate_sleep_filter_memory_pressure_filtered"] == 0
    assert scoped["candidate_sleep_filter_low_usefulness_filtered"] == 0
    assert scoped["candidate_sleep_filter_runs_all_columns"] is False
    assert scoped["column_wake_plan_awake_count"] == 4
    assert scoped["column_wake_plan_memory_pressure_filtered"] == 0
    assert scoped["column_wake_plan_low_usefulness_filtered"] == 0
    assert scoped["column_wake_plan_bounded"] is True
    assert scoped["column_wake_plan_runs_all_columns"] is False
    assert scoped["column_metabolism_updated_columns"] == 4
    assert scoped["column_metabolism_cached_columns"] == 12
    assert scoped["column_metabolism_runs_all_columns"] is False
    assert scoped["structural_review_last_evaluated_columns"] == 4
    assert scoped["structural_review_last_cached_columns"] == 12
    assert scoped["structural_review_last_growth_candidate_count"] == 0
    assert scoped["structural_review_last_prune_or_sleep_candidate_count"] == 0
    assert scoped["structural_review_runs_all_columns"] is False
    assert scoped["structural_review_checkpoint_backed"] is True
    assert scoped["structural_review_requires_operator_review"] is True
    assert scoped["structural_review_mutates_runtime_state"] is False
    assert scoped["column_wake_plan_wake_reason"] == (
        "retrieved_candidate_before_deep_sleep_age_gate"
    )
    assert scoped["runs_all_columns"] is False


def test_column_scheduler_benchmark_can_queue_reviewed_structural_tickets() -> None:
    report = run_benchmark(
        n_columns=16,
        column_latent_dim=8,
        k_routing=4,
        samples=3,
        warmup_steps=1,
        seed=124,
        device="cpu",
        force_structural_review_evidence=True,
    )

    scoped = report["scoped_cached_vote"]

    assert report["forced_structural_review_evidence"] is True
    assert (
        report["forced_structural_review_capture_scope"]
        == "post_measurement_bounded_wake_plan_ticket_audit_not_timed"
    )
    assert report["structural_review_bounded"] is True
    assert report["structural_review_continuation_reviewed"] is True
    assert report["bounded_specialist_work"] is True
    assert scoped["structural_review_pending_count"] >= 8
    assert scoped["structural_review_growth_ticket_count"] >= 4
    assert scoped["structural_review_prune_or_sleep_ticket_count"] >= 1
    assert (
        scoped["structural_review_last_update_mode"]
        == "benchmark_forced_bounded_structural_review_evidence"
    )
    assert scoped["structural_review_last_evaluated_columns"] == 4
    assert scoped["structural_review_last_cached_columns"] == 12
    assert scoped["structural_review_last_growth_candidate_count"] == 4
    assert scoped["structural_review_last_prune_or_sleep_candidate_count"] == 4
    assert scoped["structural_review_next_gate"] == "operator_review_column_structural_ticket"
    assert scoped["structural_review_checkpoint_backed"] is True
    assert scoped["structural_review_requires_operator_review"] is True
    assert scoped["structural_review_mutates_runtime_state"] is False
    assert scoped["structural_review_runs_all_columns"] is False


def test_column_scheduler_scaling_benchmark_keeps_awake_work_constant() -> None:
    report = run_scaling_benchmark(
        column_counts=(16, 32),
        column_latent_dim=8,
        k_routing=4,
        samples=2,
        warmup_steps=1,
        seed=456,
        device="cpu",
    )

    assert report["surface"] == "column_scheduler_scaling_benchmark.v1"
    assert report["scope"] == "constant_k_candidate_deep_sleep_pressure_usefulness_filter_predictive_update_vote_and_structural_review_queue_scaling"
    assert report["column_counts"] == [16, 32]
    assert report["awake_count_remains_bounded"] is True
    assert report["scoped_never_runs_all_columns"] is True
    assert "all_winner_sequences_equal" not in report
    assert "neutral_or_better_all_sizes" not in report
    assert len(report["runs"]) == 2
    assert not any("winner_sequence_equal" in row for row in report["runs"])
    assert not any("neutral_or_better_complete_tick" in row for row in report["runs"])
    assert {row["candidate_sleep_filter_output_candidates"] for row in report["runs"]} == {4}
    assert {row["candidate_sleep_filter_memory_pressure_filtered"] for row in report["runs"]} == {0}
    assert {row["candidate_sleep_filter_low_usefulness_filtered"] for row in report["runs"]} == {0}
    assert {row["predictive_location_update_columns"] for row in report["runs"]} == {4}
    assert {row["column_metabolism_updated_columns"] for row in report["runs"]} == {4}
    assert {row["structural_review_last_evaluated_columns"] for row in report["runs"]} == {4}
    assert {row["structural_review_last_growth_candidate_count"] for row in report["runs"]} == {0}
    assert {
        row["structural_review_last_prune_or_sleep_candidate_count"]
        for row in report["runs"]
    } == {0}
    assert {row["column_wake_plan_awake_count"] for row in report["runs"]} == {4}
    assert all(row["structural_review_checkpoint_backed"] for row in report["runs"])
    assert all(row["structural_review_requires_operator_review"] for row in report["runs"])
    assert not any(row["structural_review_mutates_runtime_state"] for row in report["runs"])
    assert not any(row["structural_review_runs_all_columns"] for row in report["runs"])
    assert all(row["column_wake_plan_bounded"] for row in report["runs"])
