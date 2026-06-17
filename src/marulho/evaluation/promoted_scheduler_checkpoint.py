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
    candidate_memory_pressure_filter_start_tokens: int = 512,
    candidate_usefulness_filter_start_tokens: int = 512,
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
        candidate_memory_pressure_filter_start_tokens=int(
            candidate_memory_pressure_filter_start_tokens
        ),
        candidate_usefulness_filter_start_tokens=int(
            candidate_usefulness_filter_start_tokens
        ),
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


def _install_active_scheduler_filter_fixture(
    trainer: MarulhoTrainer,
    *,
    pressure_count: int,
    low_usefulness_count: int,
) -> dict[str, Any]:
    runtime = trainer._column_transition_runtime
    snapshot = runtime.route_candidate_bank_checkpoint()
    bank_ids = snapshot.get("ids")
    if not isinstance(bank_ids, torch.Tensor) or not bool(snapshot.get("valid")):
        raise RuntimeError("active scheduler filter fixture requires a ready route bank")
    probe_rows = int(snapshot.get("probe_rows", 0) or 0)
    pressure_total = max(0, int(pressure_count))
    usefulness_total = max(0, int(low_usefulness_count))
    filtered_total = pressure_total + usefulness_total
    if filtered_total > int(probe_rows):
        raise ValueError(
            "active scheduler filter fixture would force route-vote fallback; "
            f"requested {filtered_total} filtered rows but only {probe_rows} "
            "probe rows can be masked while preserving k awake outputs"
        )
    if filtered_total <= 0:
        return {
            "enabled": False,
            "pressure_count": 0,
            "low_usefulness_count": 0,
            "max_filtered_without_fallback": int(probe_rows),
            "claim_boundary": "no active scheduler filter fixture installed",
        }

    metabolism = trainer.model.column_metabolism
    ids = bank_ids.to(device=trainer.model.device, dtype=torch.long).flatten()
    pressure_ids = ids[:pressure_total]
    usefulness_ids = ids[pressure_total:filtered_total]
    if pressure_total > 0:
        metabolism.memory_pressure[pressure_ids] = 1.0
        metabolism.last_memory_pressure_source = (
            "promoted_scheduler_checkpoint_active_pressure_fixture"
        )
    if usefulness_total > 0:
        metabolism.usefulness[usefulness_ids] = 0.0
        metabolism.last_usefulness_source = (
            "promoted_scheduler_checkpoint_active_usefulness_fixture"
        )
    metabolism.last_filter_report = {
        **metabolism.last_filter_report,
        "mode": "active_scheduler_filter_checkpoint_fixture",
        "input_candidate_count": int(snapshot.get("bank_size", int(ids.numel())) or 0),
        "filtered_memory_pressure_count": int(pressure_total),
        "filtered_low_usefulness_count": int(usefulness_total),
        "memory_pressure_source": str(metabolism.last_memory_pressure_source),
        "usefulness_source": str(metabolism.last_usefulness_source),
        "runs_all_columns": False,
    }
    return {
        "enabled": True,
        "pressure_count": int(pressure_total),
        "low_usefulness_count": int(usefulness_total),
        "pressure_ids": [int(value) for value in pressure_ids.detach().cpu().tolist()],
        "low_usefulness_ids": [
            int(value) for value in usefulness_ids.detach().cpu().tolist()
        ],
        "max_filtered_without_fallback": int(probe_rows),
        "memory_pressure_source": str(metabolism.last_memory_pressure_source),
        "usefulness_source": str(metabolism.last_usefulness_source),
        "claim_boundary": (
            "checkpoint fixture marks cached metabolism rows so restored "
            "route-vote can prove active scheduler masking without adding an "
            "all-column hot-path scan"
        ),
    }


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
    active_pressure_filter_count: int = 0,
    active_low_usefulness_filter_count: int = 0,
    candidate_memory_pressure_filter_start_tokens: int = 512,
    candidate_usefulness_filter_start_tokens: int = 512,
) -> dict[str, Any]:
    cfg = promoted_scheduler_config(
        n_columns=int(n_columns),
        column_latent_dim=int(column_latent_dim),
        k_routing=int(k_routing),
        device=str(device),
        candidate_memory_pressure_filter_start_tokens=int(
            candidate_memory_pressure_filter_start_tokens
        ),
        candidate_usefulness_filter_start_tokens=int(
            candidate_usefulness_filter_start_tokens
        ),
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
    scheduler_filter_fixture = _install_active_scheduler_filter_fixture(
        trainer,
        pressure_count=int(active_pressure_filter_count),
        low_usefulness_count=int(active_low_usefulness_filter_count),
    )

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
            "scheduler_filter_fixture": dict(scheduler_filter_fixture),
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
        "scheduler_filter_fixture": dict(scheduler_filter_fixture),
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
                "route_vote_scheduler_filter": dict(
                    restore_after.get("route_vote_deep_sleep_filter") or {}
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
    parser.add_argument("--active-pressure-filter-count", type=int, default=0)
    parser.add_argument("--active-low-usefulness-filter-count", type=int, default=0)
    parser.add_argument(
        "--candidate-memory-pressure-filter-start-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--candidate-usefulness-filter-start-tokens",
        type=int,
        default=512,
    )
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
        active_pressure_filter_count=args.active_pressure_filter_count,
        active_low_usefulness_filter_count=args.active_low_usefulness_filter_count,
        candidate_memory_pressure_filter_start_tokens=(
            args.candidate_memory_pressure_filter_start_tokens
        ),
        candidate_usefulness_filter_start_tokens=(
            args.candidate_usefulness_filter_start_tokens
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
