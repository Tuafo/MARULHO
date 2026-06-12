from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from marulho.config.model_config import MarulhoConfig
from marulho.evaluation.compiled_column_kernel_benchmark import (
    run_compiled_column_kernel_benchmark,
)
from marulho.training.checkpointing import save_trainer_checkpoint
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


def test_compiled_column_kernel_benchmark_reports_isolated_scope() -> None:
    with TemporaryDirectory() as tmpdir:
        cfg = MarulhoConfig(
            n_columns=16,
            column_latent_dim=4,
            n_ascii=8,
            k_routing=3,
            bootstrap_tokens=0,
            memory_capacity=16,
            enable_context_layer=False,
            enable_binding_layer=False,
            enable_cross_modal=False,
        )
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
        checkpoint = save_trainer_checkpoint(Path(tmpdir) / "kernel.pt", trainer)

        with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
            report = run_compiled_column_kernel_benchmark(
                checkpoint,
                batch_size=4,
                iterations=2,
                warmup_iterations=1,
                seed=123,
            )

    assert report["surface"] == "compiled_column_kernel_benchmark.v1"
    assert report["scope"] == "isolated_fixed_shape_candidate_competition_no_runtime_mutation"
    assert report["batch_size"] == 4
    assert report["iterations"] == 2
    assert report["device"] == "cpu"
    assert report["arms"]
    assert report["best_arm"]["tokens_per_second"] > 0.0
    assert report["throughput_reference"]["reference_floor_tokens_per_second"] == 1000.0
    assert report["throughput_reference"]["goal"] == "maximize_sustainable_local_throughput_not_stop_at_reference_floor"
