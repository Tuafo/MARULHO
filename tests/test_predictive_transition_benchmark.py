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
                candidate_count=3,
            )

    assert report["surface"] == "predictive_transition_benchmark.v1"
    assert report["scope"] == "isolated_dense_predictive_state_transition_no_runtime_writeback"
    assert report["device"] == "cpu"
    assert report["arms"][0]["name"] == "eager"
    assert report["best_arm"]["tokens_per_second"] > 0.0
    assert len(report["output_shapes"]) == 6
    assert report["candidate_count"] == 3
    writeback = report["writeback_experiment"]
    assert writeback["scope"] == "isolated_predictive_state_writeback_candidate_scope_experiment"
    assert writeback["candidate_count"] == 3
    assert writeback["total_columns"] == 8
    assert writeback["candidate_rows_match_dense"] is True
    assert writeback["arms"][0]["name"] == "dense_all_columns_writeback"
    assert writeback["arms"][0]["runs_all_columns"] is True
    assert writeback["arms"][1]["name"] == "candidate_scoped_eager_writeback"
    assert writeback["arms"][1]["runs_all_columns"] is False
    assert writeback["arms"][1]["updated_column_count"] == 3
    assert writeback["arms"][1]["cached_state_count"] == 5
    assert writeback["promotion_decision"] in {
        "promote_candidate_scoped_writeback_for_further_runtime_testing",
        "retain_dense_cuda_predictive_update",
    }
