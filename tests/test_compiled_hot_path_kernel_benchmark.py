from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from marulho.config.model_config import MarulhoConfig
from marulho.evaluation.compiled_hot_path_kernel_benchmark import (
    run_compiled_hot_path_kernel_benchmark,
)
from marulho.training.checkpointing import save_trainer_checkpoint
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


def test_compiled_hot_path_kernel_benchmark_reports_non_mutating_scope() -> None:
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
        checkpoint = save_trainer_checkpoint(Path(tmpdir) / "hot-path-kernel.pt", trainer)

        with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
            report = run_compiled_hot_path_kernel_benchmark(
                checkpoint,
                batch_size=4,
                iterations=2,
                warmup_iterations=1,
                matmul_precision="default",
                candidate_source="random",
                merge_torch_shards=True,
                seed=123,
            )

    assert report["surface"] == "compiled_hot_path_kernel_benchmark.v1"
    assert report["scope"] == "isolated_fixed_shape_projection_competition_predictive_no_runtime_mutation"
    assert report["batch_size"] == 4
    assert report["iterations"] == 2
    assert report["matmul_precision"] == "default"
    assert report["candidate_source"] == "random"
    assert report["merge_torch_shards"] is True
    assert report["candidate_prep"]["candidate_source"] == "random"
    assert report["candidate_prep"]["fallback_rows"] == 0
    assert report["device"] == "cpu"
    assert report["arms"]
    assert report["best_arm"]["tokens_per_second"] > 0.0
    assert report["throughput_reference"]["goal"] == "maximize_sustainable_local_throughput_not_stop_at_reference_floor"


def test_compiled_hot_path_kernel_benchmark_can_use_routing_index_candidates() -> None:
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
        checkpoint = save_trainer_checkpoint(Path(tmpdir) / "hot-path-kernel.pt", trainer)

        with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
            report = run_compiled_hot_path_kernel_benchmark(
                checkpoint,
                batch_size=4,
                iterations=2,
                warmup_iterations=0,
                matmul_precision="default",
                candidate_source="routing_index",
                seed=123,
            )

    assert report["candidate_source"] == "routing_index"
    assert report["candidate_prep"]["candidate_source"] == "routing_index"
    assert report["candidate_prep"]["fallback_rows"] == 0
    assert report["candidate_prep"]["candidate_prep_latency_ms"] >= 0.0
    assert report["candidate_prep"]["routing_index_stats"]["unique_vectors"] == 16
    assert report["arms"]
    assert report["best_arm"]["tokens_per_second"] > 0.0


def test_compiled_hot_path_kernel_benchmark_can_use_tensor_routing_candidates() -> None:
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
        checkpoint = save_trainer_checkpoint(Path(tmpdir) / "hot-path-kernel.pt", trainer)

        with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
            report = run_compiled_hot_path_kernel_benchmark(
                checkpoint,
                batch_size=4,
                iterations=2,
                warmup_iterations=0,
                matmul_precision="default",
                candidate_source="routing_index_tensor",
                seed=123,
            )

    assert report["candidate_source"] == "routing_index_tensor"
    assert report["candidate_prep"]["candidate_source"] == "routing_index_tensor"
    assert report["candidate_prep"]["fallback_rows"] == 0
    assert report["candidate_prep"]["candidate_prep_latency_ms"] >= 0.0
    assert report["candidate_prep"]["routing_index_stats"]["unique_vectors"] == 16
    assert report["arms"]
    assert report["best_arm"]["tokens_per_second"] > 0.0


def test_compiled_hot_path_kernel_benchmark_exact_route_compete_requires_promoted_shape() -> None:
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
            input_weight_blend=0.02,
        )
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
        checkpoint = save_trainer_checkpoint(Path(tmpdir) / "hot-path-kernel.pt", trainer)

        with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
            report = run_compiled_hot_path_kernel_benchmark(
                checkpoint,
                batch_size=4,
                iterations=2,
                warmup_iterations=0,
                matmul_precision="default",
                candidate_source="routing_index_tensor",
                exact_route_compete=True,
                seed=123,
            )

    exact = report["exact_route_compete"]
    assert exact["enabled"] is True
    assert exact["supported"] is False
    assert "requires_zero_input_weight_blend" in exact["unsupported_reasons"]
    assert exact["promotion_status"] == "blocked_until_exact_candidate_and_winner_parity"


def test_compiled_hot_path_kernel_benchmark_exact_route_compete_reports_parity() -> None:
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
            input_weight_blend=0.0,
            routing_index_mode="torch_topk",
        )
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
        checkpoint = save_trainer_checkpoint(Path(tmpdir) / "hot-path-kernel.pt", trainer)

        with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
            report = run_compiled_hot_path_kernel_benchmark(
                checkpoint,
                batch_size=4,
                iterations=2,
                warmup_iterations=0,
                matmul_precision="default",
                candidate_source="routing_index_tensor",
                exact_route_compete=True,
                route_compete_last_winner=0,
                seed=123,
            )

    exact = report["exact_route_compete"]
    assert exact["enabled"] is True
    assert exact["supported"] is True
    assert exact["parity"]["candidate_match"] is True
    assert exact["parity"]["winner_match"] is True
    assert exact["parity"]["promotion_safe"] is True
    assert exact["arms"]
    assert exact["best_arm"]["tokens_per_second"] > 0.0
