from __future__ import annotations

from marulho.evaluation.column_scheduler_benchmark import run_benchmark


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
    assert report["winner_sequence_equal"] is True
    assert report["awake_count_bounded"] is True
    assert report["predictive_vote_bounded"] is True
    assert scoped["predictive_vote_updated_columns"] == 4
    assert scoped["predictive_vote_cached_columns"] == 12
    assert scoped["predictive_vote_runs_all_columns"] is False
    assert all_vote["predictive_vote_updated_columns"] == 16
    assert all_vote["predictive_vote_runs_all_columns"] is True
