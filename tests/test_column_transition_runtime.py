from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
import pytest

from marulho.config.model_config import MarulhoConfig
from marulho.core.inplace_column_cuda import (
    select_fused_vote_competition_cuda,
    select_single_winner_cuda,
)
from marulho.training.column_transition_runtime import ColumnTransitionRuntime
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


def test_inplace_transition_falls_back_before_mutation_on_cpu() -> None:
    config = MarulhoConfig(
        n_columns=16,
        column_latent_dim=8,
        memory_capacity=16,
        predictive_dense_transition_mode="inplace_triton",
        input_weight_blend=0.0,
        device="cpu",
    )
    trainer = MarulhoTrainer(MarulhoModel(config), config)

    report = trainer.column_transition_runtime_report()

    assert report["requested_mode"] == "inplace_triton"
    assert report["resolved_mode"] == "retained_runtime"
    assert report["active"] is False
    assert report["fallback_reason"] == "inplace_triton_requires_cuda"
    assert report["warmup_attempted"] is False
    assert report["execution_count"] == 0
    assert report["fallback_happens_before_mutation"] is True


def test_inplace_transition_rejects_unsupported_plasticity_before_warmup() -> None:
    config = MarulhoConfig(
        n_columns=16,
        column_latent_dim=8,
        memory_capacity=16,
        predictive_dense_transition_mode="inplace_triton",
        input_weight_blend=0.25,
        device="cpu",
    )
    trainer = MarulhoTrainer(MarulhoModel(config), config)

    report = trainer.column_transition_runtime_report()

    assert report["active"] is False
    assert report["warmup_attempted"] is False
    assert report["execution_count"] == 0


def test_fused_route_vote_reports_pre_mutation_fallback_on_cpu() -> None:
    config = MarulhoConfig(
        n_columns=16,
        column_latent_dim=8,
        memory_capacity=16,
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="fused_triton_text",
        input_weight_blend=0.0,
        device="cpu",
    )
    trainer = MarulhoTrainer(MarulhoModel(config), config)

    report = trainer.column_transition_runtime_report()

    assert report["route_vote_requested_mode"] == "fused_triton_text"
    assert report["route_vote_resolved_mode"] == "tensor"
    assert report["route_vote_active"] is False
    assert report["route_vote_fallback_reason"] == (
        "fused_route_vote_requires_inplace_runtime"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize(
    ("combined_values", "inhibition_values", "expected_winner", "expected_positive"),
    [
        ([0.2, 0.9, 0.4], [0.5, 0.3, 0.5], 9, True),
        ([0.4, 0.3, 0.2], [1.0, 1.0, 1.0], 4, False),
    ],
)
def test_single_winner_cuda_matches_retained_selection_semantics(
    combined_values: list[float],
    inhibition_values: list[float],
    expected_winner: int,
    expected_positive: bool,
) -> None:
    device = torch.device("cuda")
    winner = torch.empty(1, dtype=torch.long, device=device)
    strength = torch.empty(1, device=device)
    had_positive = torch.empty((), dtype=torch.bool, device=device)

    select_single_winner_cuda(
        combined=torch.tensor(combined_values, device=device),
        inhibition=torch.tensor(inhibition_values, device=device),
        candidates=torch.tensor([4, 9, 2], dtype=torch.long, device=device),
        winner_out=winner,
        strength_out=strength,
        competition_had_positive=had_positive,
    )
    torch.cuda.synchronize()

    assert int(winner.item()) == expected_winner
    assert float(strength.item()) == 1.0
    assert bool(had_positive.item()) is expected_positive


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("silent", [False, True])
def test_fused_vote_competition_matches_retained_candidate_math(
    silent: bool,
) -> None:
    device = torch.device("cuda")
    prototypes = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
    )
    routing_key = torch.tensor([0.8, 0.5, 0.2, 0.1], device=device)
    routing_key = routing_key.clamp(min=1e-6)
    routing_key = routing_key / routing_key.norm()
    locations = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [-1.0, 0.0],
            [0.6, 0.4],
        ],
        device=device,
    )
    candidates = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
    previous_winner = torch.tensor([3], dtype=torch.long, device=device)
    thresholds = torch.full(
        (4,),
        2.0 if silent else 0.2,
        device=device,
    )
    winner = torch.empty(1, dtype=torch.long, device=device)
    strength = torch.empty(1, device=device)
    had_positive = torch.empty((), dtype=torch.bool, device=device)

    previous_location = locations[3]
    location_similarity = torch.nn.functional.cosine_similarity(
        locations[candidates],
        previous_location.unsqueeze(0),
        dim=1,
    )
    consensus_gain = 1.0 + 0.3 * location_similarity.clamp(-1.0, 1.0)
    combined = (prototypes[candidates] @ routing_key) * consensus_gain
    activation = torch.relu(combined - thresholds[candidates])
    expected_local = (
        int(torch.argmax(activation).item())
        if bool((activation.max() > 0).item())
        else int(torch.argmax(combined).item())
    )
    expected_winner = int(candidates[expected_local].item())
    expected_positive = bool((activation.max() > 0).item())

    select_fused_vote_competition_cuda(
        routing_key=routing_key,
        prototypes=prototypes,
        thresholds=thresholds,
        prediction_location=locations,
        candidates=candidates,
        previous_winner=previous_winner,
        winner_out=winner,
        strength_out=strength,
        competition_had_positive=had_positive,
    )
    torch.cuda.synchronize()

    assert int(winner.item()) == expected_winner
    assert int(previous_winner.item()) == expected_winner
    assert float(strength.item()) == 1.0
    assert bool(had_positive.item()) is expected_positive


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_checkpoint_opt_in_fused_route_vote_matches_tensor_candidates() -> None:
    torch.manual_seed(20260612)
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=16,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="fused_triton_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        cuda_graph_host_truth_sync_interval_tokens=1,
        device="cuda",
    )
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    runtime = trainer._column_transition_runtime
    routing_key = trainer.model.routing_key_from_pattern(
        torch.rand(config.input_dim, device=trainer.model.device)
    )
    expected, _ = trainer.model.hnsw_index.search_tensors(
        routing_key.unsqueeze(0),
        k=config.k_routing,
    )

    candidates = runtime.route_candidates(routing_key, sensory_tick=False)
    assert candidates is not None
    torch.cuda.synchronize()

    assert torch.equal(candidates, expected[0])
    assert runtime.handles_route_vote is True
    assert runtime.route_vote_execution_count == 1
    assert runtime.route_vote_clean_cache_reuse_count == 1
    assert runtime.route_candidates(routing_key, sensory_tick=True) is None
    assert runtime.route_vote_sensory_fallback_count == 1


def test_cuda_graph_route_transition_reports_pre_mutation_fallback_on_cpu() -> None:
    config = MarulhoConfig(
        n_columns=8,
        column_latent_dim=4,
        bootstrap_tokens=0,
        k_routing=2,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="cuda_graph_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        device="cpu",
    )
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    report = trainer.column_transition_runtime_report()

    assert report["active"] is False
    assert report["route_vote_requested_mode"] == "cuda_graph_text"
    assert report["route_vote_resolved_mode"] == "tensor"
    assert report["route_vote_active"] is False
    assert report["route_vote_fallback_reason"] == (
        "fused_route_vote_requires_inplace_runtime"
    )
    assert report["cuda_graph_route_transition"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_graph_route_transition_matches_fused_sequential_state() -> None:
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=16,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="fused_triton_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        device="cuda",
    )
    torch.manual_seed(20260612)
    retained = MarulhoTrainer(MarulhoModel(config), config)
    graph_config = replace(
        config,
        predictive_route_vote_mode="cuda_graph_text",
        cuda_graph_host_truth_sync_interval_tokens=1,
    )
    torch.manual_seed(20260612)
    graph = MarulhoTrainer(MarulhoModel(graph_config), graph_config)
    graph_report = graph.column_transition_runtime_report()
    assert graph_report["cuda_graph_route_transition"]["active"] is True
    assert graph_report["execution_count"] == 0

    generator = torch.Generator(device="cuda").manual_seed(20260613)
    patterns = [
        torch.rand(config.input_dim, generator=generator, device="cuda")
        for _ in range(16)
    ]
    for index, pattern in enumerate(patterns):
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state()
        retained_metrics = retained.train_step(
            pattern,
            raw_window=f"graph parity {index}",
            allow_sleep_maintenance=False,
        )
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        graph_metrics = graph.train_step(
            pattern,
            raw_window=f"graph parity {index}",
            allow_sleep_maintenance=False,
        )
        assert retained.last_winner == graph.last_winner
        assert graph_metrics["recon_error"] == retained_metrics["recon_error"]

    for retained_tensor, graph_tensor in (
        (retained.model.competitive.prototypes, graph.model.competitive.prototypes),
        (
            retained.model.competitive.prototype_velocity,
            graph.model.competitive.prototype_velocity,
        ),
        (retained.model.competitive.thresholds, graph.model.competitive.thresholds),
        (retained.model.competitive.win_rate_ema, graph.model.competitive.win_rate_ema),
        (
            retained.model.competitive.steps_since_win,
            graph.model.competitive.steps_since_win,
        ),
        (retained.model.predictive.location, graph.model.predictive.location),
        (retained.model.predictive.velocity, graph.model.predictive.velocity),
        (
            retained.model.predictive._prediction_weights,
            graph.model.predictive._prediction_weights,
        ),
        (
            retained.model.predictive.prediction_error,
            graph.model.predictive.prediction_error,
        ),
        (retained.model.predictive.confidence, graph.model.predictive.confidence),
        (
            retained.model.competitive.recent_spike_window,
            graph.model.competitive.recent_spike_window,
        ),
    ):
        assert torch.allclose(
            retained_tensor,
            graph_tensor,
            rtol=0.0,
            atol=1e-7,
        )
    retained_errors = list(
        retained.model.surprise.layers["competitive"]["errors"]
    )
    graph_errors = list(graph.model.surprise.layers["competitive"]["errors"])
    assert graph_errors == pytest.approx(retained_errors, abs=1e-7)
    assert (
        graph.model.surprise.layers["competitive"]["precision"]
        == pytest.approx(
            retained.model.surprise.layers["competitive"]["precision"],
            rel=1e-6,
        )
    )
    final_report = graph.column_transition_runtime_report()
    graph_runtime = final_report["cuda_graph_route_transition"]
    route_vectors, route_ids = graph.model.hnsw_index.routing_tensor_cache()
    route_positions = torch.empty(
        config.n_columns,
        dtype=torch.long,
        device="cuda",
    )
    route_positions[route_ids] = torch.arange(
        config.n_columns,
        dtype=torch.long,
        device="cuda",
    )
    assert torch.allclose(
        route_vectors.index_select(0, route_positions),
        graph.model.competitive.prototypes,
        rtol=0.0,
        atol=1e-7,
    )
    assert final_report["last_execution_mode"] == "cuda_graph_route_transition"
    assert graph_runtime["pre_route_replay_count"] == 16
    assert graph_runtime["tick_replay_count"] == 16
    assert graph_runtime["replay_count"] == 16
    assert graph_runtime["host_truth_sync_count"] == 16
    assert graph_runtime["host_truth_skip_count"] == 0
    assert graph_runtime["surprise_update_count"] == 16
    assert graph_runtime["host_truth_mirror_update_count"] == 16
    assert graph_runtime["competitive_surprise_update_count"] == 16
    assert graph_runtime["route_cache_generation_fastpath_count"] == 16
    assert graph_runtime["route_cache_clean_fastpath_count"] == 0
    assert graph_runtime["route_cache_generation_mismatch_count"] == 0
    assert graph_runtime["route_cache_device_owned"] is True
    assert graph_runtime["route_cache_device_update_count"] == 16
    assert graph_runtime["persistent_tick_graph"] is True
    assert graph_runtime["owns_competitive_surprise"] is True
    assert graph_runtime["owns_neuromodulator_update"] is True
    assert graph_runtime["failure_count"] == 0
    assert graph_runtime["previous_flag_device_owned_count"] == 16
    assert graph_runtime["learning_rate_device_owned_count"] == 16
    assert graph_runtime["learning_rate_host_resync_count"] == 0
    assert graph_runtime["modulator_stage_copy_count"] == 16
    assert graph_runtime["modulator_stage_skip_count"] == 0
    assert final_report["route_vote_prepared_graph_reuse_count"] == 16
    assert graph_metrics["routing_index_device_update_count"] == 16
    assert graph_metrics["routing_index_buffer_skip_count"] == 16
    assert graph_metrics["routing_index_host_mirror_sync_count"] == 0
    assert graph_metrics["routing_index_cpu_mirror_stale"] == 1
    assert graph._hnsw_buffer_ids == []
    assert graph._hnsw_buffer_vecs == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_graph_host_truth_mirror_is_cadenced() -> None:
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=16,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="cuda_graph_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        cuda_graph_host_truth_sync_interval_tokens=4,
        trainer_telemetry_interval_tokens=2,
        device="cuda",
    )
    torch.manual_seed(20260614)
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    trainer.enable_train_step_profile()
    for index in range(8):
        trainer.train_step(
            torch.rand(config.input_dim, device="cuda"),
            raw_window=f"cadenced graph truth {index}",
            allow_sleep_maintenance=False,
        )

    report = trainer.column_transition_runtime_report()["cuda_graph_route_transition"]
    assert report["tick_replay_count"] == 8
    assert report["surprise_update_count"] == 8
    assert report["host_truth_sync_interval_tokens"] == 4
    assert report["host_truth_sync_count"] == 3
    assert report["host_truth_skip_count"] == 5
    assert report["host_truth_mirror_update_count"] == 3
    assert report["last_result_from_host_sync"] is True
    assert report["competitive_surprise_update_count"] == 3
    assert report["failure_count"] == 0
    runtime_report = trainer.column_transition_runtime_report()
    assert runtime_report["graph_host_winner_reuse_count"] == 3
    assert runtime_report["winner_consolidation_cpu_metric_count"] == 1
    assert runtime_report["winner_consolidation_cached_metric_count"] == 3
    assert runtime_report["winner_host_mirror_sync_count"] == 3
    assert runtime_report["winner_host_mirror_skip_count"] == 5
    assert runtime_report["winner_host_mirror_fresh"] is True
    assert len(trainer.model.surprise.layers["competitive"]["errors"]) == 3
    profile = trainer.train_step_profile_report()["per_tick_ms"]
    assert profile["cuda_graph_prepare_parameter_stage"] >= 0.0
    assert profile["cuda_graph_prepare_recent_row_fill"] >= 0.0
    assert profile["cuda_graph_prepare_input_stage"] >= 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_graph_fails_closed_when_consolidation_cache_generation_changes() -> None:
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=16,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="cuda_graph_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        device="cuda",
    )
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    trainer.memory_warm_started = True
    trainer._column_transition_runtime = ColumnTransitionRuntime(trainer)
    graph = trainer._column_transition_runtime._cuda_graph_runtime
    assert graph is not None
    assert graph.active is True

    assert graph.eligible(assume_route_cache_current=True) is True
    trainer.model.memory_store._invalidate_bucket_consolidation_cache()

    assert graph.eligible(assume_route_cache_current=True) is False
    report = graph.report()
    assert report["active"] is False
    assert report["fallback_reason"] == (
        "cuda_graph_consolidation_cache_generation_changed"
    )
    assert report["consolidation_cache_generation_fastpath_count"] == 1
    assert report["consolidation_cache_generation_mismatch_count"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_graph_pre_route_bypasses_sensory_ticks() -> None:
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=16,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="cuda_graph_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        device="cuda",
    )
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    runtime = trainer._column_transition_runtime

    prepared = runtime.prepare_routing(
        torch.rand(config.input_dim, device="cuda"),
        sensory_tick=True,
    )
    report = runtime.report()["cuda_graph_route_transition"]

    assert prepared is None
    assert report["pre_route_sensory_bypass_count"] == 1
    assert report["pre_route_replay_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_graph_learning_rate_counter_resyncs_after_sensory_fallback() -> None:
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=16,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="cuda_graph_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        cuda_graph_host_truth_sync_interval_tokens=1,
        device="cuda",
    )
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    for index, sensory_tick in enumerate((False, True, False)):
        trainer.train_step(
            torch.rand(config.input_dim, device="cuda"),
            raw_window=f"lr counter resync {index}",
            visual_spikes=(
                torch.rand(config.input_dim, device="cuda")
                if sensory_tick
                else None
            ),
            allow_sleep_maintenance=False,
        )

    report = trainer.column_transition_runtime_report()[
        "cuda_graph_route_transition"
    ]
    assert report["tick_replay_count"] == 2
    assert report["pre_route_sensory_bypass_count"] == 1
    assert report["learning_rate_device_owned_count"] == 2
    assert report["learning_rate_host_resync_count"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_device_owned_route_cache_syncs_host_mirror_before_retained_flush() -> None:
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=16,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="cuda_graph_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        device="cuda",
    )
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    trainer.train_step(
        torch.rand(config.input_dim, device="cuda"),
        raw_window="device route cache owner",
        allow_sleep_maintenance=False,
    )
    assert trainer._routing_index_cpu_mirror_stale is True
    assert trainer._routing_index_host_mirror_sync_count == 0

    trainer._hnsw_flush_interval = 1
    retained_id = torch.tensor([0], dtype=torch.long, device="cuda")
    trainer._buffer_hnsw_update(
        retained_id,
        trainer.model.competitive.prototypes.index_select(
            0,
            retained_id,
        ),
        known_ids=[0],
    )

    assert trainer._routing_index_cpu_mirror_stale is False
    assert trainer._routing_index_host_mirror_sync_count == 1
    assert trainer._hnsw_buffer_ids == []
    assert trainer._hnsw_buffer_vecs == []
    normalized = torch.nn.functional.normalize(
        trainer.model.competitive.prototypes.detach(),
        dim=1,
    ).cpu()
    for column_id in range(config.n_columns):
        index = trainer.model.hnsw_index
        store_owner = (
            index.shards[index.shard_for_id(column_id)]
            if hasattr(index, "shards")
            else index
        )
        assert np.allclose(
            store_owner._vector_store[column_id],
            normalized[column_id].numpy(),
            atol=1e-6,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_graph_modulator_stage_is_revision_cached_between_host_syncs() -> None:
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=16,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="cuda_graph_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        cuda_graph_host_truth_sync_interval_tokens=4,
        device="cuda",
    )
    torch.manual_seed(20260615)
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    for index in range(8):
        trainer.train_step(
            torch.rand(config.input_dim, device="cuda"),
            raw_window=f"modulator cache {index}",
            allow_sleep_maintenance=False,
        )

    report = trainer.column_transition_runtime_report()[
        "cuda_graph_route_transition"
    ]
    assert report["tick_replay_count"] == 8
    assert report["host_truth_sync_count"] == 3
    assert report["modulator_stage_copy_count"] == 3
    assert report["modulator_stage_skip_count"] == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_graph_pre_route_bypasses_bootstrap_ticks() -> None:
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=4,
        k_routing=5,
        memory_capacity=16,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="cuda_graph_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        device="cuda",
    )
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    runtime = trainer._column_transition_runtime

    prepared = runtime.prepare_routing(
        torch.rand(config.input_dim, device="cuda"),
        sensory_tick=False,
    )
    report = runtime.report()["cuda_graph_route_transition"]

    assert prepared is None
    assert report["pre_route_bootstrap_bypass_count"] == 1
    assert report["tick_replay_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_inplace_transition_compile_only_warmup_preserves_brain_state() -> None:
    torch.manual_seed(20260612)
    config = MarulhoConfig(
        n_columns=16,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=4,
        memory_capacity=16,
        predictive_dense_transition_mode="inplace_triton",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        device="cuda",
    )
    model = MarulhoModel(config)
    before = {
        "prototypes": model.competitive.prototypes.clone(),
        "velocity": model.competitive.prototype_velocity.clone(),
        "thresholds": model.competitive.thresholds.clone(),
        "win_rate": model.competitive.win_rate_ema.clone(),
        "steps": model.competitive.steps_since_win.clone(),
        "location": model.predictive.location.clone(),
        "prediction_weights": model.predictive._prediction_weights.clone(),
        "prediction_error": model.predictive.prediction_error.clone(),
        "confidence": model.predictive.confidence.clone(),
        "spikes": model.competitive.recent_spike_window.clone(),
    }

    trainer = MarulhoTrainer(model, config)
    torch.cuda.synchronize()
    report = trainer.column_transition_runtime_report()

    assert report["active"] is True
    assert report["resolved_mode"] == "inplace_triton"
    assert report["warmup_attempted"] is True
    assert report["warmup_succeeded"] is True
    assert report["fallback_reason"] is None
    assert report["precompiled_candidate_counts"] == [4, 16]
    assert report["execution_count"] == 0
    assert all(
        torch.equal(getattr_tensor, before[name])
        for name, getattr_tensor in {
            "prototypes": model.competitive.prototypes,
            "velocity": model.competitive.prototype_velocity,
            "thresholds": model.competitive.thresholds,
            "win_rate": model.competitive.win_rate_ema,
            "steps": model.competitive.steps_since_win,
            "location": model.predictive.location,
            "prediction_weights": model.predictive._prediction_weights,
            "prediction_error": model.predictive.prediction_error,
            "confidence": model.predictive.confidence,
            "spikes": model.competitive.recent_spike_window,
        }.items()
    )

    trainer.train_step(
        torch.randn(config.input_dim, device=model.device),
        raw_window="production in-place transition",
        allow_sleep_maintenance=False,
    )
    torch.cuda.synchronize()
    after = trainer.column_transition_runtime_report()

    assert after["execution_count"] == 1
    assert after["failure_count"] == 0
    assert after["last_execution_mode"] == "inplace_triton"
    assert after["selection_execution_count"] == 1
    assert after["selection_failure_count"] == 0
    assert after["last_selection_mode"] == "fused_vote_competition_triton"
    assert after["selection_host_sync_required"] is False
    assert after["fused_vote_competition_active"] is True
    assert after["fused_vote_competition_execution_count"] == 1
    assert after["fused_vote_competition_fallback_count"] == 0
    assert model.predictive.last_dense_transition_mode == "inplace_triton"

    routing_key = model.routing_key_from_pattern(
        torch.randn(config.input_dim, device=model.device)
    )
    candidates = trainer._routing_candidates(routing_key)
    fallback_winner, _, _ = trainer._column_transition_runtime.select_winner(
        routing_key=routing_key,
        candidates=candidates,
        context_gain=torch.ones(config.n_columns, device=model.device),
        fallback_allowed=False,
    )
    torch.cuda.synchronize()
    fallback_report = trainer.column_transition_runtime_report()

    assert fallback_report["fused_vote_competition_fallback_count"] == 1
    assert int(trainer._column_transition_runtime._previous_winner.item()) == int(
        fallback_winner.item()
    )
