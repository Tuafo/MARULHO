from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from marulho.config.model_config import MarulhoConfig
from marulho.evaluation.hot_window_benchmark import run_hot_window_benchmark
from marulho.evaluation.persistent_tick_hot_window_benchmark import (
    run_persistent_tick_hot_window_ab,
)
from marulho.evaluation.quantum_input_staging_benchmark import (
    run_quantum_input_staging_ab,
)
from marulho.training.checkpointing import save_trainer_checkpoint
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


def test_hot_window_benchmark_reports_encoded_tensor_scope() -> None:
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
        checkpoint = save_trainer_checkpoint(Path(tmpdir) / "hot-window.pt", trainer)

        with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
            report = run_hot_window_benchmark(
                checkpoint,
                samples=4,
                warmup_steps=1,
                predictive_transition_mode="fused_eager",
                seed=123,
            )

    assert report["surface"] == "hot_window_benchmark.v1"
    assert report["scope"] == "already_encoded_tensor_train_step_no_service_no_source_no_sleep"
    assert "full train_step is not fused" in report["claim_boundary"]
    assert report["samples"] == 4
    assert report["warmup_steps"] == 1
    assert report["routing_candidate_mode"] == "tensor"
    assert report["routing_cache_boundary"] == "merged_torch_route_cache_required"
    assert report["predictive_transition_mode"] == "fused_eager"
    assert report["warmup_elapsed_s"] >= 0.0
    assert report["runtime_counters"]["routing_index"]["last_search_mode"] == "tensor"
    assert (
        report["runtime_counters"][
            "cross_modal_text_idle_probe_interval_tokens"
        ]
        == cfg.cross_modal_text_idle_probe_interval_tokens
    )
    assert (
        report["runtime_counters"]["candidate_homeostasis_start_tokens"]
        == cfg.candidate_homeostasis_start_tokens
    )
    assert (
        report["runtime_counters"]["candidate_predictive_update_start_tokens"]
        == cfg.candidate_predictive_update_start_tokens
    )
    assert (
        report["runtime_counters"]["candidate_deep_sleep_filter_start_tokens"]
        == cfg.candidate_deep_sleep_filter_start_tokens
    )
    assert (
        report["runtime_counters"]["candidate_sleep_filter_execution"]["surface"]
        == "column_candidate_sleep_scheduler.v1"
    )
    assert (
        report["runtime_counters"]["predictive_update_execution"]["surface"]
        == "predictive_column_update_scheduler.v1"
    )
    assert (
        report["runtime_counters"]["predictive_vote_execution"]["surface"]
        == "predictive_column_vote_scheduler.v1"
    )
    assert report["runtime_counters"]["competitive"]["input_plasticity_mode"] in {
        "lite_active",
        "skipped_zero_blend",
    }
    assert report["device"] == "cpu"
    assert report["tokens_per_second"] > 0.0
    assert report["step_latency_ms"]["median"] > 0.0
    assert report["target_gap"]["target_tokens_per_second"] == 1000.0
    assert report["target_gap"]["target_met"] is False


def test_hot_window_benchmark_supports_evaluation_only_trainer_setup() -> None:
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
        checkpoint = save_trainer_checkpoint(
            Path(tmpdir) / "hot-window-setup.pt",
            trainer,
        )

        def setup(loaded_trainer: object) -> None:
            setattr(loaded_trainer, "_benchmark_transition_executor", "test")

        with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
            report = run_hot_window_benchmark(
                checkpoint,
                samples=1,
                warmup_steps=0,
                _trainer_setup=setup,
            )

    assert report["transition_executor"] == "test"


def test_hot_window_benchmark_can_profile_measured_steps_only() -> None:
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
        checkpoint = save_trainer_checkpoint(
            Path(tmpdir) / "hot-window-profile.pt",
            trainer,
        )

        with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
            report = run_hot_window_benchmark(
                checkpoint,
                samples=3,
                warmup_steps=2,
                profile_trainer_stages=True,
                seed=321,
            )

    profile = report["trainer_stage_profile"]
    assert isinstance(profile, dict)
    assert profile["enabled"] is True
    assert profile["count"] == 3
    assert profile["scope"] == "measured_hot_window_steps_only_no_service_no_source_no_sleep"
    assert profile["tokens_per_second_observed"] > 0.0
    assert profile["per_tick_ms"]["total"] > 0.0


def test_hot_window_benchmark_supports_window_sync_throughput_mode() -> None:
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
        checkpoint = save_trainer_checkpoint(
            Path(tmpdir) / "hot-window-window-sync.pt",
            trainer,
        )

        with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
            report = run_hot_window_benchmark(
                checkpoint,
                samples=2,
                warmup_steps=1,
                sync_mode="window",
                seed=456,
            )

    assert report["sync_mode"] == "window"
    assert (
        report["latency_sample_scope"]
        == "host_dispatch_latency_with_single_window_cuda_sync"
    )
    assert report["tokens_per_second"] > 0.0


def test_persistent_tick_ab_reports_stage_deltas_without_cuda() -> None:
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
        checkpoint = save_trainer_checkpoint(
            Path(tmpdir) / "persistent-ab-profile.pt",
            trainer,
        )

        def fused_setup(loaded_trainer: object) -> None:
            setattr(loaded_trainer, "_benchmark_transition_executor", "fused-test")

        def persistent_setup(loaded_trainer: object) -> None:
            setattr(
                loaded_trainer,
                "_benchmark_transition_executor",
                "persistent-test",
            )

        with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
            report = run_persistent_tick_hot_window_ab(
                checkpoint,
                samples=2,
                warmup_steps=1,
                profile_trainer_stages=True,
                sync_mode="window",
                _arm_setups=(
                    ("fused_a", fused_setup),
                    ("persistent_a", persistent_setup),
                    ("persistent_b", persistent_setup),
                    ("fused_b", fused_setup),
                ),
            )

    assert report["surface"] == "persistent_tick_hot_window_ab.v1"
    assert report["profile_trainer_stages"] is True
    assert report["sync_mode"] == "window"
    assert "continuous sequential throughput" in report["sync_mode_semantics"]
    assert len(report["arms"]) == 4
    assert report["fused_mean_stage_per_tick_ms"]["total"] > 0.0
    assert report["persistent_mean_stage_per_tick_ms"]["total"] > 0.0
    assert report["largest_stage_deltas"]
    assert report["largest_stage_deltas"][0]["stage"]


def test_quantum_input_staging_ab_uses_reversed_arm_order() -> None:
    calls: list[str] = []

    def fake_hot_window(
        _checkpoint: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        setup = kwargs["_trainer_setup"]
        setup_name = getattr(setup, "__name__")
        calls.append(setup_name)
        enabled = setup_name == "_enabled"
        return {
            "tokens_per_second": 900.0 if enabled else 600.0,
            "step_latency_ms": {"median": 1.0},
            "quantum_input_stage_elapsed_ms": 0.25 if enabled else 0.0,
            "cuda_memory": {},
            "runtime_counters": {
                "column_transition_runtime": {
                    "cuda_graph_route_transition": {
                        "active": True,
                        "quantum_input_staging_enabled": enabled,
                    }
                }
            },
        }

    with TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "quantum-input-ab.json"
        with patch(
            "marulho.evaluation.quantum_input_staging_benchmark."
            "run_hot_window_benchmark",
            side_effect=fake_hot_window,
        ):
            report = run_quantum_input_staging_ab(
                Path(tmpdir) / "runtime.pt",
                output_path=output,
                samples=8,
                warmup_steps=2,
                quantum_tokens=4,
            )

        assert output.exists()

    assert calls == ["_disabled", "_enabled", "_enabled", "_disabled"]
    assert report["per_token_mean_tokens_per_second"] == 600.0
    assert report["quantum_mean_tokens_per_second"] == 900.0
    assert report["speedup"] == 1.5
    assert report["success"] is True
