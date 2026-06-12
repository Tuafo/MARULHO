from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from marulho.config.model_config import MarulhoConfig
from marulho.evaluation.steady_state_column_transition_benchmark import (
    run_steady_state_column_transition_benchmark,
)
from marulho.training.checkpointing import save_trainer_checkpoint
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


def test_steady_state_column_transition_benchmark_stays_evaluation_only() -> None:
    with TemporaryDirectory() as tmpdir:
        config = MarulhoConfig(
            n_columns=8,
            column_latent_dim=4,
            k_routing=4,
            bootstrap_tokens=0,
            memory_capacity=16,
            routing_index_mode="torch_topk",
            enable_context_layer=False,
            enable_binding_layer=False,
            enable_cross_modal=False,
        )
        trainer = MarulhoTrainer(MarulhoModel(config), config)
        checkpoint = save_trainer_checkpoint(
            Path(tmpdir) / "steady-transition.pt",
            trainer,
        )
        with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
            report = run_steady_state_column_transition_benchmark(
                checkpoint,
                iterations=2,
                warmup_iterations=1,
                seed=123,
            )

    assert report["surface"] == "steady_state_column_transition_benchmark.v1"
    assert report["promotion_status"] == "rejected_for_always_on_runtime"
    assert report["device"] == "cpu"
    assert report["arms"][0]["name"] == "functional_eager"
    assert report["best_arm"]["transitions_per_second"] > 0.0
    assert report["state_tensor_count"] == 11
