from __future__ import annotations

import pytest
import torch

from marulho.core.column_runtime import bounded_column_associative_recall, build_column_runtime_report


def test_column_runtime_report_keeps_awake_columns_bounded_and_votes_cached() -> None:
    report = build_column_runtime_report(
        n_columns=12,
        prediction_error=torch.tensor([0.9, 0.1, 0.8, 0.0, 0.05, 0.7, 0.1, 0.2, 0.0, 0.6, 0.1, 0.05]),
        confidence=torch.tensor([0.2, 0.9, 0.3, 0.8, 0.95, 0.4, 0.7, 0.6, 0.9, 0.3, 0.8, 0.85]),
        steps_since_win=torch.tensor([1, 2, 4, 80, 0, 3, 70, 5, 900, 10, 1, 2]),
        win_rate_ema=torch.ones(12) / 12.0,
        last_winner_ids=[5],
        awake_limit=3,
        sleep_after_steps=64,
        deep_sleep_after_steps=512,
        token_count=42,
        device="cpu",
    )

    assert report["surface"] == "column_runtime_metabolism.v1"
    assert report["summary_role"] == "training_owned_scheduler_evidence_with_cached_vote_execution"
    assert report["total_columns"] == 12
    assert report["awake_budget"] == 3
    assert report["awake_count"] <= 3
    assert report["active_count"] == report["awake_count"]
    assert report["candidate_count"] == report["awake_count"]
    assert report["idle_count"] >= 0
    assert report["retired_count"] == 0
    assert report["runs_all_columns"] is False
    assert report["scheduler"]["runs_all_columns"] is False
    assert report["scheduler"]["promoted_to_execution"] is True
    assert report["scheduler"]["execution_scope"] == "candidate_scoring_homeostasis_and_predictive_vote_cache"
    assert 5 in report["scheduler"]["awake_column_ids"]
    assert report["registry"]["surface"] == "column_registry.v1"
    assert report["registry"]["mutates_runtime_state"] is False
    assert report["registry"]["columns_sample"]
    assert report["local_associative_recall"]["available"] is True
    assert report["local_associative_recall"]["enabled_in_runtime_tick"] is False
    assert report["sleeping_count"] >= 1
    assert report["deep_sleeping_count"] == 1
    assert report["growth_gate"]["requires_operator_review"] is True
    assert report["growth_gate"]["mutates_runtime_state"] is False
    assert report["pruning_homeostasis"]["mutates_runtime_state"] is False
    assert all(vote["mutates_column"] is False for vote in report["votes"])
    assert report["metabolism"]["source_tensor_device"] == "cpu"
    assert report["metabolism"]["report_compute_device"] == "cpu"
    assert report["metabolism"]["source_tensor_count"] == 4
    assert report["metabolism"]["snapshot_tensor_count"] == 4
    assert report["metabolism"]["materialized_column_state_count"] < 12
    assert report["metabolism"]["snapshot_bytes"] == (
        report["metabolism"]["materialized_column_state_count"] * 4 * 4
    )
    assert report["metabolism"]["device_transfer_count"] == 0
    assert report["metabolism"]["hot_path_effect"] == "none_latency_first_runtime_truth_snapshot"
    assert report["metabolism"]["claim_boundary"] == "latency_first_column_status_snapshot_not_hot_path_execution"


def test_column_runtime_growth_gate_needs_repeated_surprise() -> None:
    calm = build_column_runtime_report(
        n_columns=6,
        prediction_error=torch.zeros(6),
        confidence=torch.ones(6) * 0.8,
        steps_since_win=torch.zeros(6),
        win_rate_ema=torch.ones(6) / 6.0,
        awake_limit=2,
    )
    one_shot_surprised = build_column_runtime_report(
        n_columns=6,
        prediction_error=torch.tensor([0.9, 0.8, 0.7, 0.1, 0.0, 0.0]),
        confidence=torch.ones(6) * 0.4,
        steps_since_win=torch.zeros(6),
        win_rate_ema=torch.ones(6) / 6.0,
        awake_limit=2,
    )
    repeated_surprised = build_column_runtime_report(
        n_columns=6,
        prediction_error=torch.tensor([0.9, 0.8, 0.7, 0.1, 0.0, 0.0]),
        confidence=torch.ones(6) * 0.4,
        steps_since_win=torch.zeros(6),
        win_rate_ema=torch.ones(6) / 6.0,
        prediction_failure_streak=torch.tensor([3, 4, 2, 9, 0, 0]),
        awake_limit=2,
        growth_streak_threshold=3,
    )

    assert calm["growth_gate"]["ready"] is False
    assert calm["growth_gate"]["candidate_column_count"] == 0
    assert one_shot_surprised["growth_gate"]["ready"] is False
    assert one_shot_surprised["growth_gate"]["one_shot_surprise_count"] == 3
    assert one_shot_surprised["growth_gate"]["candidate_column_count"] == 0
    assert one_shot_surprised["growth_gate"]["evidence"] == "missing_prediction_failure_streak"
    assert repeated_surprised["growth_gate"]["ready"] is True
    assert repeated_surprised["growth_gate"]["candidate_column_count"] == 2
    assert repeated_surprised["growth_gate"]["reversible_path"] == (
        "isolated_binding_trial_then_operator_checkpoint_transaction"
    )
    assert repeated_surprised["growth_gate"]["binding_growth_trial_available"] is True
    assert repeated_surprised["growth_gate"]["next_gate"] == "explicit_binding_growth_trial_design"


def test_bounded_column_associative_recall_is_local_topk_and_non_mutating() -> None:
    query = torch.tensor([1.0, 0.0, 0.0, 0.0])
    memory = torch.eye(4)
    result = bounded_column_associative_recall(
        query=query,
        memory=memory,
        top_k=2,
        beta=16.0,
        max_memory=3,
    )

    assert result["surface"] == "bounded_column_associative_recall.v1"
    assert result["scope"] == "single_column_bounded_recall"
    assert result["used_memory_count"] == 3
    assert result["top_k"] == 2
    assert int(result["indices"][0].item()) == 0
    assert torch.isclose(result["weights"].sum(), torch.tensor(1.0), atol=1e-6)
    assert result["recalled"].shape == query.shape
    assert result["mutates_runtime_state"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_column_runtime_uses_one_bounded_cuda_snapshot_for_report_compute() -> None:
    report = build_column_runtime_report(
        n_columns=1024,
        prediction_error=torch.rand(1024, device="cuda"),
        confidence=torch.rand(1024, device="cuda"),
        steps_since_win=torch.zeros(1024, device="cuda"),
        win_rate_ema=torch.ones(1024, device="cuda") / 1024.0,
        awake_limit=10,
        device="cuda",
    )

    assert report["device"] == "cuda"
    assert report["metabolism"]["source_tensor_device"] == "cuda:0"
    assert report["metabolism"]["report_compute_device"] == "cpu"
    assert report["metabolism"]["source_tensor_count"] == 4
    assert report["metabolism"]["snapshot_tensor_count"] == 4
    assert report["metabolism"]["materialized_column_state_count"] == 1024
    assert report["metabolism"]["snapshot_bytes"] == 1024 * 4 * 4
    assert report["metabolism"]["device_transfer_count"] == 1
    assert report["metabolism"]["claim_boundary"] == "latency_first_column_status_snapshot_not_hot_path_execution"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_bounded_column_associative_recall_stays_on_cuda_when_given_cuda_memory() -> None:
    query = torch.randn(8, device="cuda")
    memory = torch.randn(16, 8, device="cuda")

    result = bounded_column_associative_recall(
        query=query,
        memory=memory,
        top_k=4,
        max_memory=8,
    )

    assert result["recalled"].is_cuda
    assert result["indices"].is_cuda
    assert result["weights"].is_cuda
    assert result["used_memory_count"] == 8
