"""Build promoted route-bank scheduler checkpoints for scaling gates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from marulho.config.model_config import MarulhoConfig
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.checkpointing import (
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


def promoted_scheduler_config(
    *,
    n_columns: int,
    column_latent_dim: int = 64,
    k_routing: int = 10,
    device: str = "cuda",
) -> MarulhoConfig:
    return MarulhoConfig(
        n_columns=int(n_columns),
        column_latent_dim=int(column_latent_dim),
        bootstrap_tokens=0,
        k_routing=int(k_routing),
        memory_capacity=64,
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="cuda_graph_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        candidate_homeostasis_start_tokens=0,
        candidate_predictive_update_start_tokens=0,
        candidate_deep_sleep_filter_start_tokens=0,
        candidate_memory_pressure_filter_start_tokens=512,
        micro_sleep_interval_tokens=10**9,
        deep_sleep_interval_tokens=10**9,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        enable_cross_modal=False,
        device=str(device),
    )


def _sync_if_cuda(cfg: MarulhoConfig) -> None:
    if cfg.resolve_device().type == "cuda":
        torch.cuda.synchronize()


def build_promoted_scheduler_checkpoint(
    *,
    checkpoint_path: Path,
    report_path: Path,
    n_columns: int,
    column_latent_dim: int = 64,
    k_routing: int = 10,
    seed: int = 20260616,
    device: str = "cuda",
    verify_restore: bool = True,
) -> dict[str, Any]:
    cfg = promoted_scheduler_config(
        n_columns=int(n_columns),
        column_latent_dim=int(column_latent_dim),
        k_routing=int(k_routing),
        device=str(device),
    )
    resolved_device = cfg.resolve_device()
    if resolved_device.type != "cuda":
        raise RuntimeError(
            "promoted route-bank scaling checkpoints require CUDA; use "
            "column_scheduler_benchmark for CPU cached-vote evidence"
        )

    torch.manual_seed(int(seed))
    trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
    seed_pattern = torch.rand(cfg.input_dim, device=resolved_device)
    trainer.train_step(
        seed_pattern,
        raw_window="promoted scheduler checkpoint route-bank seed",
        allow_sleep_maintenance=False,
    )
    _sync_if_cuda(cfg)
    seed_report = trainer.column_transition_runtime_report()
    route_bank = dict(seed_report.get("route_candidate_bank") or {})
    route_scoring = dict(seed_report.get("route_vote_scoring") or {})
    if not bool(route_bank.get("ready", False)):
        raise RuntimeError("route candidate bank did not become ready before save")

    checkpoint_path = save_trainer_checkpoint(
        checkpoint_path,
        trainer,
        metadata={
            "purpose": "promoted_route_bank_scheduler_scaling_gate",
            "n_columns": int(n_columns),
            "column_latent_dim": int(column_latent_dim),
            "k_routing": int(k_routing),
            "seed": int(seed),
            "synthetic_fresh_checkpoint": True,
            "candidate_scheduler_promoted_from_token": 0,
            "claim_boundary": (
                "builds a restored route-bank checkpoint for long complete-runtime "
                "scaling gates; seed cost is explicit and not measured as steady "
                "bounded scheduler work"
            ),
        },
    )

    restore_before: dict[str, Any] | None = None
    restore_after: dict[str, Any] | None = None
    if bool(verify_restore):
        restored, metadata = load_trainer_checkpoint(checkpoint_path)
        restore_before = restored.column_transition_runtime_report()
        restore_pattern = torch.rand(cfg.input_dim, device=resolved_device)
        restored.train_step(
            restore_pattern,
            raw_window="promoted scheduler checkpoint bounded restore tick",
            allow_sleep_maintenance=False,
        )
        _sync_if_cuda(cfg)
        restore_after = restored.column_transition_runtime_report()
    else:
        metadata = {}

    report = {
        "surface": "promoted_scheduler_checkpoint.v1",
        "checkpoint": str(checkpoint_path),
        "success": True,
        "n_columns": int(n_columns),
        "column_latent_dim": int(column_latent_dim),
        "k_routing": int(k_routing),
        "seed": int(seed),
        "runtime_device": cfg.device_report(),
        "checkpoint_metadata": dict(metadata),
        "seed_tick": {
            "route_candidate_bank": route_bank,
            "route_vote_scoring": route_scoring,
            "state_transition_runs_all_columns": bool(
                seed_report.get("state_transition_runs_all_columns", False)
            ),
        },
        "restore_before_tick": (
            {
                "route_candidate_bank": dict(
                    restore_before.get("route_candidate_bank") or {}
                ),
                "route_vote_scoring": dict(
                    restore_before.get("route_vote_scoring") or {}
                ),
            }
            if restore_before is not None
            else None
        ),
        "restore_after_tick": (
            {
                "route_candidate_bank": dict(
                    restore_after.get("route_candidate_bank") or {}
                ),
                "route_vote_scoring": dict(
                    restore_after.get("route_vote_scoring") or {}
                ),
                "state_transition_cached_count": int(
                    restore_after.get("state_transition_cached_count", 0) or 0
                ),
                "state_transition_runs_all_columns": bool(
                    restore_after.get("state_transition_runs_all_columns", False)
                ),
            }
            if restore_after is not None
            else None
        ),
        "claim_boundary": (
            "checkpoint builder only prepares the promoted route-bank/probe-lane "
            "path for complete-runtime stress gates; it does not prove quality "
            "or hide the explicit seed"
        ),
    }
    write_json_report_with_readme(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--n-columns", type=int, required=True)
    parser.add_argument("--column-latent-dim", type=int, default=64)
    parser.add_argument("--k-routing", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-restore-verify", action="store_true")
    args = parser.parse_args()
    build_promoted_scheduler_checkpoint(
        checkpoint_path=args.checkpoint,
        report_path=args.report,
        n_columns=args.n_columns,
        column_latent_dim=args.column_latent_dim,
        k_routing=args.k_routing,
        seed=args.seed,
        device=args.device,
        verify_restore=not args.skip_restore_verify,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
