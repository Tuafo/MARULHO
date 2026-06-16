from __future__ import annotations

import torch

from marulho.training.column_structural_review import ColumnStructuralReviewQueue


def test_structural_review_queue_records_only_bounded_awake_candidates() -> None:
    queue = ColumnStructuralReviewQueue(n_columns=1024, device="cpu")
    prediction_error = torch.zeros(1024)
    confidence = torch.ones(1024)
    streak = torch.zeros(1024, dtype=torch.long)
    cost = torch.zeros(1024)
    pressure = torch.zeros(1024)
    prediction_error[7] = 0.9
    confidence[7] = 0.2
    streak[7] = 4
    pressure[9] = 0.99

    queue.record_candidates(
        torch.tensor([7, 9]),
        token_count=12,
        mode="awake_mask_tick",
        prediction_error=prediction_error,
        confidence=confidence,
        prediction_failure_streak=streak,
        estimated_cost=cost,
        memory_pressure=pressure,
        wake_reason="unit_awake",
        sleep_reason=None,
    )

    report = queue.report()
    assert report["surface"] == "column_structural_review_queue.v1"
    assert report["pending_count"] == 2
    assert report["growth_ticket_count"] == 1
    assert report["prune_or_sleep_ticket_count"] == 1
    assert report["last_evaluated_column_count"] == 2
    assert report["last_cached_column_count"] == 1022
    assert report["runs_all_columns"] is False
    assert report["checkpoint_backed"] is True
    assert report["requires_operator_review"] is True
    assert report["mutates_runtime_state"] is False
    assert report["next_gate"] == "operator_review_column_structural_ticket"
    assert {ticket["column_id"] for ticket in report["tickets_sample"]} == {7, 9}
    assert all(ticket["mutates_runtime_state"] is False for ticket in report["tickets_sample"])


def test_structural_review_queue_deferred_burst_records_truth_not_fake_scan() -> None:
    queue = ColumnStructuralReviewQueue(n_columns=8192, device="cpu")

    queue.record_deferred(
        token_count=32,
        mode="cuda_graph_text_burst_deferred",
        reason="host_truth_not_synced_for_structural_review_queue",
    )

    report = queue.report()
    assert report["pending_count"] == 0
    assert report["last_evaluated_column_count"] == 0
    assert report["last_cached_column_count"] == 8192
    assert report["deferred_update_count"] == 1
    assert report["last_deferred_reason"] == "host_truth_not_synced_for_structural_review_queue"
    assert report["runs_all_columns"] is False


def test_structural_review_queue_state_roundtrip_preserves_tickets() -> None:
    source = ColumnStructuralReviewQueue(n_columns=8, device="cpu")
    source.record_candidates(
        torch.tensor([2]),
        token_count=4,
        mode="awake_mask_tick",
        prediction_error=torch.tensor([0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]),
        confidence=torch.tensor([1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0]),
        prediction_failure_streak=torch.tensor([0, 0, 5, 0, 0, 0, 0, 0]),
        estimated_cost=torch.zeros(8),
        memory_pressure=torch.zeros(8),
        wake_reason="unit_awake",
        sleep_reason=None,
    )

    restored = ColumnStructuralReviewQueue(n_columns=8, device="cpu")
    restored.load_state_dict(source.state_dict())

    report = restored.report()
    assert report["pending_count"] == 1
    assert report["growth_ticket_count"] == 1
    assert report["tickets_sample"][0]["column_id"] == 2
    assert report["tickets_sample"][0]["requires_checkpoint_transaction"] is True
