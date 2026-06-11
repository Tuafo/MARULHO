from __future__ import annotations

import torch

from marulho.core.column_runtime import build_column_runtime_report


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
    assert report["summary_role"] == "report_only_scheduler_evidence_not_execution_scheduler"
    assert report["total_columns"] == 12
    assert report["awake_budget"] == 3
    assert report["awake_count"] <= 3
    assert report["runs_all_columns"] is False
    assert report["scheduler"]["runs_all_columns"] is False
    assert 5 in report["scheduler"]["awake_column_ids"]
    assert report["sleeping_count"] >= 1
    assert report["deep_sleeping_count"] == 1
    assert report["growth_gate"]["requires_operator_review"] is True
    assert report["growth_gate"]["mutates_runtime_state"] is False
    assert report["pruning_homeostasis"]["mutates_runtime_state"] is False
    assert all(vote["mutates_column"] is False for vote in report["votes"])


def test_column_runtime_growth_gate_needs_repeated_surprise() -> None:
    calm = build_column_runtime_report(
        n_columns=6,
        prediction_error=torch.zeros(6),
        confidence=torch.ones(6) * 0.8,
        steps_since_win=torch.zeros(6),
        win_rate_ema=torch.ones(6) / 6.0,
        awake_limit=2,
    )
    surprised = build_column_runtime_report(
        n_columns=6,
        prediction_error=torch.tensor([0.9, 0.8, 0.7, 0.1, 0.0, 0.0]),
        confidence=torch.ones(6) * 0.4,
        steps_since_win=torch.zeros(6),
        win_rate_ema=torch.ones(6) / 6.0,
        awake_limit=2,
    )

    assert calm["growth_gate"]["ready"] is False
    assert calm["growth_gate"]["candidate_column_count"] == 0
    assert surprised["growth_gate"]["ready"] is True
    assert surprised["growth_gate"]["candidate_column_count"] == 3
