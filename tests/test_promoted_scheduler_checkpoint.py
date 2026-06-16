from __future__ import annotations

from pathlib import Path

import torch

from marulho.evaluation.promoted_scheduler_checkpoint import (
    build_promoted_scheduler_checkpoint,
    promoted_scheduler_config,
)


def test_promoted_scheduler_config_uses_single_route_bank_path() -> None:
    cfg = promoted_scheduler_config(n_columns=128, device="cuda")

    assert cfg.n_columns == 128
    assert cfg.k_routing == 10
    assert cfg.predictive_dense_transition_mode == "inplace_triton"
    assert cfg.predictive_route_vote_mode == "cuda_graph_text"
    assert cfg.candidate_homeostasis_start_tokens == 0
    assert cfg.candidate_predictive_update_start_tokens == 0
    assert cfg.candidate_deep_sleep_filter_start_tokens == 0
    assert cfg.micro_sleep_interval_tokens == 10**9
    assert cfg.deep_sleep_interval_tokens == 10**9
    assert not hasattr(cfg, "route_candidate_bank_size")


@torch.no_grad()
def test_promoted_scheduler_checkpoint_restores_bounded_route_bank(
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        return

    report = build_promoted_scheduler_checkpoint(
        checkpoint_path=tmp_path / "promoted-scheduler.pt",
        report_path=tmp_path / "promoted-scheduler.json",
        n_columns=32,
        column_latent_dim=8,
        k_routing=5,
        seed=20260616,
        device="cuda",
    )

    assert Path(report["checkpoint"]).exists()
    seed_scoring = report["seed_tick"]["route_vote_scoring"]
    assert seed_scoring["route_input_rows_scored"] == 32
    assert seed_scoring["route_rows_run_all_columns"] is True
    assert seed_scoring["bounded_route_scoring"] is False

    before_bank = report["restore_before_tick"]["route_candidate_bank"]
    assert before_bank["ready"] is True
    assert before_bank["checkpoint_restore_count"] == 1
    assert before_bank["restore_reason"] == "route_candidate_bank_restored_from_checkpoint"

    after = report["restore_after_tick"]
    after_scoring = after["route_vote_scoring"]
    assert after_scoring["route_input_rows_scored"] == 7
    assert after_scoring["route_output_candidate_count"] == 5
    assert after_scoring["route_rows_run_all_columns"] is False
    assert after_scoring["bounded_route_scoring"] is True
    assert after_scoring["route_scoring_unbounded_reason"] is None
    assert after["state_transition_cached_count"] == 27
    assert after["state_transition_runs_all_columns"] is False
