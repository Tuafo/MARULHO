from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from marulho.config.model_config import MarulhoConfig
from marulho.evaluation.predictive_transition_benchmark import (
    run_predictive_transition_benchmark,
)
from marulho.training.checkpointing import save_trainer_checkpoint
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


def test_predictive_transition_benchmark_reports_non_mutating_scope() -> None:
    with TemporaryDirectory() as tmpdir:
        cfg = MarulhoConfig(
            n_columns=8,
            column_latent_dim=4,
            bootstrap_tokens=0,
            memory_capacity=16,
            enable_context_layer=False,
            enable_binding_layer=False,
            enable_cross_modal=False,
        )
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
        checkpoint = save_trainer_checkpoint(Path(tmpdir) / "predictive-transition.pt", trainer)

        with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
            report = run_predictive_transition_benchmark(
                checkpoint,
                iterations=2,
                warmup_iterations=1,
                seed=123,
            )

    assert report["surface"] == "predictive_transition_benchmark.v1"
    assert report["scope"] == "isolated_dense_predictive_state_transition_no_runtime_writeback"
    assert report["device"] == "cpu"
    assert report["arms"][0]["name"] == "eager"
    assert report["best_arm"]["tokens_per_second"] > 0.0
    assert len(report["output_shapes"]) == 6
    assert "writeback_experiment" not in report
