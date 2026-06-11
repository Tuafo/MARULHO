from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from marulho.config.model_config import MarulhoConfig
from marulho.core.column_runtime import build_column_runtime_report
from marulho.evaluation.binding_growth_trial import run_binding_growth_trial
from marulho.semantics import build_binding_growth_trial_design
from marulho.training.checkpointing import save_trainer_checkpoint
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_binding_growth_trial_runs_on_checkpoint_clones_without_touching_source(
    tmp_path: Path,
) -> None:
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        memory_capacity=16,
        enable_context_layer=True,
        enable_binding_layer=True,
        binding_mode="hypercube",
    )
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    torch.manual_seed(4)
    for index in range(6):
        trainer.model.memory_store.update(
            torch.rand(config.n_columns),
            input_pattern=torch.rand(config.input_dim),
            raw_window=f"isolated trial pattern {index}",
            token_count=index,
        )
    column_runtime = build_column_runtime_report(
        n_columns=config.n_columns,
        prediction_error=torch.full((config.n_columns,), 0.9),
        confidence=torch.full((config.n_columns,), 0.1),
        steps_since_win=torch.zeros(config.n_columns),
        win_rate_ema=torch.ones(config.n_columns) / config.n_columns,
        prediction_failure_streak=torch.full((config.n_columns,), 4),
        awake_limit=4,
    )
    binding_plan = trainer.model.binding_layer.plan_candidate_hub_topology(
        column_runtime["growth_gate"]["candidate_column_ids_sample"],
        max_total_edge_delta=8,
    )
    design = build_binding_growth_trial_design(
        column_runtime,
        binding_plan,
        state_revision=0,
    )
    checkpoint = save_trainer_checkpoint(
        tmp_path / "source.pt",
        trainer,
        metadata={"state_revision": 0},
    )
    before_hash = _sha256(checkpoint)

    report = run_binding_growth_trial(
        checkpoint_path=checkpoint,
        trial_design=design,
        max_samples=4,
    )

    assert report["status"] in {
        "evidence_supported_for_operator_review",
        "evaluated_without_cognitive_improvement",
    }
    assert report["baseline"]["available"] is True
    assert report["variant"]["available"] is True
    assert report["baseline"]["sample_count"] == 4
    assert report["variant"]["application"]["applied_total_edge_delta"] == 8
    assert report["mutates_live_runtime"] is False
    assert report["writes_live_checkpoint"] is False
    assert _sha256(checkpoint) == before_hash


def test_binding_growth_trial_blocks_unready_design(tmp_path: Path) -> None:
    report = run_binding_growth_trial(
        checkpoint_path=tmp_path / "missing.pt",
        trial_design={},
    )

    assert report["status"] == "blocked_missing_binding_growth_trial_evidence"
    assert report["passed"] is False
    assert report["mutates_live_runtime"] is False
