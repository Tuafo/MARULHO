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
    all_vote = report["all_column_vote"]

    assert report["surface"] == "column_scheduler_benchmark.v1"
    assert report["scope"] == "complete_train_step_deep_sleep_filter_predictive_update_and_vote_awake_mask_ab"
    assert report["winner_sequence_equal"] is True
    assert report["awake_count_bounded"] is True
    assert report["column_wake_plan_bounded"] is True
    assert report["predictive_vote_bounded"] is True
    assert report["predictive_update_bounded"] is True
    assert report["predictive_location_update_bounded"] is True
    assert report["column_metabolism_bounded"] is True
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
    assert scoped["candidate_sleep_filter_runs_all_columns"] is False
    assert scoped["column_wake_plan_awake_count"] == 4
    assert scoped["column_wake_plan_memory_pressure_filtered"] == 0
    assert scoped["column_wake_plan_bounded"] is True
    assert scoped["column_wake_plan_runs_all_columns"] is False
    assert scoped["column_metabolism_updated_columns"] == 4
    assert scoped["column_metabolism_cached_columns"] == 12
    assert scoped["column_metabolism_runs_all_columns"] is False
    assert scoped["column_wake_plan_wake_reason"] == (
        "retrieved_candidate_before_deep_sleep_age_gate"
    )
    assert scoped["runs_all_columns"] is False
    assert all_vote["predictive_vote_updated_columns"] == 16
    assert all_vote["predictive_vote_runs_all_columns"] is True
    assert all_vote["predictive_update_updated_columns"] == 16
    assert all_vote["predictive_update_runs_all_columns"] is True
    assert all_vote["predictive_location_runs_all_columns"] is True
    assert all_vote["runs_all_columns"] is True


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
    assert report["scope"] == "constant_k_candidate_deep_sleep_filter_predictive_update_and_vote_scaling"
    assert report["column_counts"] == [16, 32]
    assert report["awake_count_remains_bounded"] is True
    assert report["scoped_never_runs_all_columns"] is True
    assert len(report["runs"]) == 2
    assert all("winner_sequence_equal" in row for row in report["runs"])
    assert {row["candidate_sleep_filter_output_candidates"] for row in report["runs"]} == {4}
    assert {row["candidate_sleep_filter_memory_pressure_filtered"] for row in report["runs"]} == {0}
    assert {row["predictive_location_update_columns"] for row in report["runs"]} == {4}
    assert {row["column_metabolism_updated_columns"] for row in report["runs"]} == {4}
    assert {row["column_wake_plan_awake_count"] for row in report["runs"]} == {4}
    assert all(row["column_wake_plan_bounded"] for row in report["runs"])
