from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

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


def test_route_filter_snapshot_separates_pressure_fallback_from_applied_filter() -> None:
    runtime = object.__new__(ColumnTransitionRuntime)
    runtime._route_sleep_filter_state_host = [
        0,  # deep-sleep enabled
        0,  # combined filter applied
        0,  # deep-sleep filtered count
        12,  # deep-sleep eligible count
        3,  # insufficient pressure-eligible rows
        12,  # route input rows
        5,  # output candidate count
        0,  # sleep backfill
        1,  # memory-pressure enabled
        0,  # memory-pressure applied
        10,  # over-threshold rows observed
        2,  # pressure-eligible rows
    ]
    runtime._route_sleep_filter_control_mirror = (0, 2000, 1, 500000)
    runtime._route_ids = None
    runtime._route_candidates = None
    runtime._route_sleep_filter_state_dirty = False
    runtime._route_sleep_filter_state = torch.zeros(12, dtype=torch.long)
    runtime.route_vote_deep_sleep_filter_control_update_count = 1
    runtime.route_vote_deep_sleep_filter_state_sync_count = 1
    runtime._route_memory_pressure_source_mirror = "unit_test_cached_pressure"
    runtime._trainer = SimpleNamespace(
        model=SimpleNamespace(
            column_metabolism=SimpleNamespace(
                last_memory_pressure_source="unit_test_cached_pressure"
            )
        )
    )

    snapshot = runtime.route_sleep_filter_snapshot()

    assert snapshot["memory_pressure_enabled"] is True
    assert snapshot["memory_pressure_applied"] is False
    assert snapshot["filtered_memory_pressure_count"] == 0
    assert snapshot["memory_pressure_over_threshold_count"] == 10
    assert snapshot["fallback_reason"] == (
        "insufficient_awake_route_scores_after_memory_pressure_filter"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_fused_inplace_candidate_predictive_transition_reports_bounded_scope() -> None:
    config = MarulhoConfig(
        n_columns=64,
        column_latent_dim=8,
        k_routing=4,
        memory_capacity=128,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        input_weight_blend=0.0,
        bootstrap_tokens=0,
        candidate_homeostasis_start_tokens=0,
        candidate_predictive_update_start_tokens=0,
        candidate_deep_sleep_filter_start_tokens=10**9,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_cross_modal=False,
        device="cuda",
    )
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    assert trainer.column_transition_runtime_report()[
        "candidate_predictive_transition_active"
    ] is True

    pattern = torch.rand(config.input_dim, device=trainer.model.device)
    trainer.train_step(
        pattern,
        raw_window="candidate predictive fused runtime test",
        allow_sleep_maintenance=False,
    )

    update = trainer.model.predictive.prediction_update_execution_report()
    runtime = trainer.column_transition_runtime_report()
    assert update["mode"] == "candidate_subset"
    assert update["updated_column_count"] == config.k_routing
    assert update["cached_state_count"] == config.n_columns - config.k_routing
    assert update["runs_all_columns"] is False
    assert update["fallback_reason"] is None
    assert runtime["candidate_predictive_transition_execution_count"] == 1
    assert (
        runtime["candidate_predictive_transition_cached_count"]
        == config.n_columns - config.k_routing
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
    expected, _ = trainer.model.routing_index.search_tensors(
        routing_key.unsqueeze(0),
        k=config.k_routing,
    )

    candidates = runtime.route_candidates(routing_key, sensory_tick=False)
    assert candidates is not None
    torch.cuda.synchronize()

    assert torch.equal(candidates, expected[0])
    assert runtime.handles_route_vote is True
    assert runtime.route_vote_execution_count == 1
    assert runtime.route_vote_kernel_variant == "two_stage_route_vote"
    assert (
        trainer.column_transition_runtime_report()["route_vote_kernel_variant"]
        == "two_stage_route_vote"
    )
    assert runtime.route_vote_clean_cache_reuse_count == 1
    assert runtime.route_candidates(routing_key, sensory_tick=True) is None
    assert runtime.route_vote_sensory_fallback_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("route_vote_mode", ["fused_triton_text", "cuda_graph_text"])
def test_route_vote_sleep_filter_updates_trainer_wake_plan(
    route_vote_mode: str,
) -> None:
    torch.manual_seed(20260615)
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=16,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode=route_vote_mode,
        plasticity_mode="lite",
        input_weight_blend=0.0,
        dead_column_steps=1,
        candidate_deep_sleep_filter_start_tokens=0,
        candidate_homeostasis_start_tokens=0,
        candidate_predictive_update_start_tokens=0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        cuda_graph_host_truth_sync_interval_tokens=1,
        device="cuda",
    )
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    trainer.token_count = 1
    pattern = torch.rand(config.input_dim, device=trainer.model.device)
    routing_key = trainer.model.routing_key_from_pattern(pattern)
    expected, _ = trainer.model.routing_index.search_tensors(
        routing_key.unsqueeze(0),
        k=config.k_routing,
    )
    deep_sleep_ids = expected[0, :2].detach().clone()
    trainer.model.competitive.steps_since_win.zero_()
    trainer.model.competitive.steps_since_win[deep_sleep_ids] = int(
        config.dead_column_steps
    )

    trainer.train_step(
        pattern,
        allow_sleep_maintenance=False,
    )
    torch.cuda.synchronize()

    wake_plan = trainer.model.last_column_wake_plan
    assert wake_plan.mode == "candidate_deep_sleep_filter_route_vote"
    assert wake_plan.runs_all_columns is False
    assert wake_plan.awake_count == config.k_routing
    assert wake_plan.input_candidate_count == config.n_columns
    assert wake_plan.filtered_deep_sleep_count == 2
    assert wake_plan.sleep_reason == "deep_sleep_route_score_masked_before_topk_vote"
    assert not torch.isin(wake_plan.candidates(), deep_sleep_ids).any()
    route_filter = trainer.column_transition_runtime_report()[
        "route_vote_deep_sleep_filter"
    ]
    assert route_filter["enabled"] is True
    assert route_filter["state_current_for_control"] is True
    assert route_filter["filtered_deep_sleep_count"] == 2
    assert route_filter["state_sync_count"] >= 1
    transition_report = trainer.column_transition_runtime_report()
    assert transition_report["state_transition_runs_all_columns"] is False
    assert transition_report["state_transition_mode"].startswith(
        "candidate_subset_sparse_"
    )
    assert transition_report["state_transition_column_count"] == config.k_routing
    assert (
        transition_report["state_transition_cached_count"]
        == config.n_columns - config.k_routing
    )
    assert transition_report["state_transition_fallback_reason"] is None
    runtime_truth = trainer.model.column_runtime_report(
        token_count=trainer.token_count,
        last_winner=trainer.last_winner,
    )
    assert runtime_truth["runs_all_columns"] is False
    assert runtime_truth["execution"]["state_transition_runs_all_columns"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("route_vote_mode", ["fused_triton_text", "cuda_graph_text"])
def test_route_vote_memory_pressure_filter_updates_trainer_wake_plan(
    route_vote_mode: str,
) -> None:
    torch.manual_seed(20260616)
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=16,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode=route_vote_mode,
        plasticity_mode="lite",
        input_weight_blend=0.0,
        dead_column_steps=1000,
        candidate_deep_sleep_filter_start_tokens=10**9,
        candidate_memory_pressure_filter_start_tokens=0,
        candidate_memory_pressure_threshold=0.5,
        candidate_homeostasis_start_tokens=0,
        candidate_predictive_update_start_tokens=0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        cuda_graph_host_truth_sync_interval_tokens=1,
        device="cuda",
    )
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    trainer.token_count = 1
    pattern = torch.rand(config.input_dim, device=trainer.model.device)
    routing_key = trainer.model.routing_key_from_pattern(pattern)
    expected, _ = trainer.model.routing_index.search_tensors(
        routing_key.unsqueeze(0),
        k=config.k_routing,
    )
    pressure_ids = expected[0, :2].detach().clone()
    trainer.model.column_metabolism.memory_pressure.zero_()
    trainer.model.column_metabolism.memory_pressure[pressure_ids] = 0.99
    trainer.model.column_metabolism.last_memory_pressure_source = (
        "unit_test_cached_pressure"
    )

    trainer.train_step(
        pattern,
        allow_sleep_maintenance=False,
    )
    torch.cuda.synchronize()

    wake_plan = trainer.model.last_column_wake_plan
    assert wake_plan.mode == "candidate_memory_pressure_filter_route_vote"
    assert wake_plan.runs_all_columns is False
    assert wake_plan.awake_count == config.k_routing
    assert wake_plan.input_candidate_count == config.n_columns
    assert wake_plan.filtered_deep_sleep_count == 0
    assert wake_plan.filtered_memory_pressure_count == 2
    assert wake_plan.memory_pressure_threshold == 0.5
    assert wake_plan.memory_pressure_source == "unit_test_cached_pressure"
    assert (
        wake_plan.sleep_reason
        == "memory_pressure_route_score_masked_before_topk_vote"
    )
    assert not torch.isin(wake_plan.candidates(), pressure_ids).any()
    route_filter = trainer.column_transition_runtime_report()[
        "route_vote_deep_sleep_filter"
    ]
    assert route_filter["enabled"] is False
    assert route_filter["memory_pressure_enabled"] is True
    assert route_filter["memory_pressure_state_enabled"] is True
    assert route_filter["memory_pressure_applied"] is True
    assert route_filter["filtered_memory_pressure_count"] == 2
    assert route_filter["memory_pressure_over_threshold_count"] == 2
    assert route_filter["memory_pressure_threshold"] == 0.5
    assert route_filter["memory_pressure_source"] == "unit_test_cached_pressure"
    assert route_filter["state_sync_count"] >= 1
    runtime_truth = trainer.model.column_runtime_report(
        token_count=trainer.token_count,
        last_winner=trainer.last_winner,
    )
    assert runtime_truth["column_wake_plan"][
        "filtered_memory_pressure_count"
    ] == 2
    assert runtime_truth["candidate_sleep_filter_execution"][
        "filtered_memory_pressure_count"
    ] == 2


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
    assert graph_report["route_vote_kernel_variant"] == "two_stage_route_vote"
    assert (
        graph_report["cuda_graph_route_transition"]["route_vote_kernel_variant"]
        == "two_stage_route_vote"
    )
    assert graph_report["execution_count"] == 0
    consolidation_lookup_count = 0
    original_consolidation_lookup = (
        graph.model.memory_store.bucket_consolidation_tensor
    )

    def _tracked_consolidation_lookup(*args, **kwargs):
        nonlocal consolidation_lookup_count
        consolidation_lookup_count += 1
        return original_consolidation_lookup(*args, **kwargs)

    graph.model.memory_store.bucket_consolidation_tensor = (
        _tracked_consolidation_lookup
    )
    empty_revival_tensor = (
        graph._column_transition_runtime._empty_revived_indices
    )

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
        assert graph_metrics["recon_error"] == pytest.approx(
            retained_metrics["recon_error"],
            rel=0.0,
            abs=1e-6,
        )

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
    route_vectors, route_ids = graph.model.routing_index.routing_tensor_cache()
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
    assert graph_runtime["reconstruction_error_source"] == "fused_route_score_max"
    assert graph_runtime["route_vote_kernel_variant"] == "two_stage_route_vote"
    assert graph_runtime["fused_reconstruction_error_active"] is True
    assert graph_runtime["fused_reconstruction_error_update_count"] == 16
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
    assert final_report["graph_consolidation_lookup_skip_count"] == 16
    assert final_report["graph_empty_revival_tensor_reuse_count"] == 16
    assert consolidation_lookup_count == 0
    assert graph.model.competitive.last_revived_indices is empty_revival_tensor
    assert graph_metrics["routing_index_device_update_count"] == 16
    assert graph_metrics["routing_index_buffer_skip_count"] == 16
    assert graph_metrics["routing_index_host_mirror_sync_count"] == 0
    assert graph_metrics["routing_index_cpu_mirror_stale"] == 1
    assert graph._routing_index_buffer_ids == []
    assert graph._routing_index_buffer_vecs == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_graph_candidate_predictive_transition_matches_non_graph_path() -> None:
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=16,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="fused_triton_text",
        candidate_homeostasis_start_tokens=0,
        candidate_predictive_update_start_tokens=0,
        candidate_deep_sleep_filter_start_tokens=10**9,
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        device="cuda",
    )
    torch.manual_seed(20260618)
    retained = MarulhoTrainer(MarulhoModel(config), config)
    graph_config = replace(
        config,
        predictive_route_vote_mode="cuda_graph_text",
        cuda_graph_host_truth_sync_interval_tokens=1,
    )
    torch.manual_seed(20260618)
    graph = MarulhoTrainer(MarulhoModel(graph_config), graph_config)
    assert graph.column_transition_runtime_report()["cuda_graph_route_transition"]["active"]

    generator = torch.Generator(device="cuda").manual_seed(20260619)
    patterns = [
        torch.rand(config.input_dim, generator=generator, device="cuda")
        for _ in range(6)
    ]
    for index, pattern in enumerate(patterns):
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state()
        retained.train_step(
            pattern,
            raw_window=f"candidate graph parity {index}",
            allow_sleep_maintenance=False,
            return_metrics=False,
        )
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        graph.train_step(
            pattern,
            raw_window=f"candidate graph parity {index}",
            allow_sleep_maintenance=False,
            return_metrics=False,
        )
        assert retained.last_winner == graph.last_winner

    for retained_tensor, graph_tensor in (
        (retained.model.competitive.prototypes, graph.model.competitive.prototypes),
        (retained.model.competitive.thresholds, graph.model.competitive.thresholds),
        (retained.model.competitive.win_rate_ema, graph.model.competitive.win_rate_ema),
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
    ):
        assert torch.allclose(retained_tensor, graph_tensor, rtol=0.0, atol=1e-7)
    assert torch.equal(
        retained.model.predictive.prediction_failure_streak,
        graph.model.predictive.prediction_failure_streak,
    )
    assert torch.equal(
        retained.model.predictive.predictive_last_update_step,
        graph.model.predictive.predictive_last_update_step,
    )
    assert (
        retained.model.predictive.predictive_step_count
        == graph.model.predictive.predictive_step_count
    )
    update = graph.model.predictive.prediction_update_execution_report()
    runtime = graph.column_transition_runtime_report()
    assert update["mode"] == "candidate_subset"
    assert update["updated_column_count"] == config.k_routing
    assert update["runs_all_columns"] is False
    assert runtime["candidate_predictive_transition_execution_count"] == len(patterns)


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
def test_cuda_graph_quantum_input_staging_preserves_sequential_trajectory() -> None:
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
    unstaged_config = replace(
        config,
        cuda_graph_quantum_input_staging=False,
    )
    torch.manual_seed(20260614)
    retained = MarulhoTrainer(
        MarulhoModel(unstaged_config),
        unstaged_config,
    )
    torch.manual_seed(20260614)
    staged = MarulhoTrainer(MarulhoModel(config), config)
    patterns = [
        torch.rand(config.input_dim, device="cuda")
        for _ in range(12)
    ]

    for start in range(0, len(patterns), 8):
        quantum = patterns[start : start + 8]
        assert staged.stage_text_input_quantum(quantum) is True
        for offset, pattern in enumerate(quantum, start=start):
            retained_metrics = retained.train_step(
                pattern,
                raw_window=f"retained {offset}",
                allow_sleep_maintenance=False,
            )
            staged_metrics = staged.train_step(
                pattern,
                raw_window=f"staged {offset}",
                allow_sleep_maintenance=False,
            )
            assert staged_metrics["winner"] == retained_metrics["winner"]

    for retained_tensor, staged_tensor in (
        (retained.model.competitive.prototypes, staged.model.competitive.prototypes),
        (
            retained.model.competitive.prototype_velocity,
            staged.model.competitive.prototype_velocity,
        ),
        (retained.model.competitive.thresholds, staged.model.competitive.thresholds),
        (retained.model.competitive.win_rate_ema, staged.model.competitive.win_rate_ema),
        (
            retained.model.competitive.steps_since_win,
            staged.model.competitive.steps_since_win,
        ),
        (retained.model.predictive.location, staged.model.predictive.location),
        (retained.model.predictive.velocity, staged.model.predictive.velocity),
        (
            retained.model.predictive._prediction_weights,
            staged.model.predictive._prediction_weights,
        ),
        (
            retained.model.predictive.prediction_error,
            staged.model.predictive.prediction_error,
        ),
        (retained.model.predictive.confidence, staged.model.predictive.confidence),
        (
            retained.model.competitive.recent_spike_window,
            staged.model.competitive.recent_spike_window,
        ),
    ):
        assert torch.allclose(
            retained_tensor,
            staged_tensor,
            rtol=0.0,
            atol=1e-7,
        )
    report = staged.column_transition_runtime_report()[
        "cuda_graph_route_transition"
    ]
    assert report["quantum_input_stage_count"] == 2
    assert report["quantum_input_staged_token_count"] == 12
    assert report["quantum_input_reuse_count"] == 12
    assert report["quantum_input_fallback_copy_count"] == 0
    assert report["quantum_input_mismatch_count"] == 0
    assert report["recent_spike_row_device_owned_count"] == 12
    assert int(staged._column_transition_runtime._recent_spike_row.item()) == (
        staged.model.competitive.recent_spike_window_cursor
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_text_burst_matches_eight_sequential_graph_ticks() -> None:
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
        cuda_graph_host_truth_sync_interval_tokens=9,
        slow_memory_start_tokens=0,
        slow_memory_archive_interval_tokens=256,
        slow_memory_archive_strong_capture_threshold=10.0,
        trainer_telemetry_interval_tokens=64,
        device="cuda",
    )
    torch.manual_seed(20260614)
    sequential = MarulhoTrainer(MarulhoModel(config), config)
    torch.manual_seed(20260614)
    quantum = MarulhoTrainer(MarulhoModel(config), config)
    warm_pattern = torch.rand(config.input_dim, device="cuda")
    patterns = [
        torch.rand(config.input_dim, device="cuda")
        for _ in range(8)
    ]

    for trainer in (sequential, quantum):
        trainer.train_step(
            warm_pattern,
            raw_window="persistent executor warmup",
            allow_sleep_maintenance=False,
            return_metrics=False,
        )
    for index, pattern in enumerate(patterns):
        sequential.train_step(
            pattern,
            raw_window=f"sequential persistent executor {index}",
            allow_sleep_maintenance=False,
            return_metrics=False,
        )
    assert quantum.train_text_burst(
        patterns,
        raw_windows=[
            f"quantum persistent executor {index}" for index in range(8)
        ],
    ) is True
    torch.cuda.synchronize()

    for sequential_tensor, quantum_tensor in (
        (
            sequential.model.competitive.prototypes,
            quantum.model.competitive.prototypes,
        ),
        (
            sequential.model.competitive.prototype_velocity,
            quantum.model.competitive.prototype_velocity,
        ),
        (
            sequential.model.competitive.thresholds,
            quantum.model.competitive.thresholds,
        ),
        (
            sequential.model.competitive.win_rate_ema,
            quantum.model.competitive.win_rate_ema,
        ),
        (
            sequential.model.competitive.steps_since_win,
            quantum.model.competitive.steps_since_win,
        ),
        (
            sequential.model.competitive.recent_spike_window,
            quantum.model.competitive.recent_spike_window,
        ),
        (
            sequential.model.predictive.location,
            quantum.model.predictive.location,
        ),
        (
            sequential.model.predictive.velocity,
            quantum.model.predictive.velocity,
        ),
        (
            sequential.model.predictive._prediction_weights,
            quantum.model.predictive._prediction_weights,
        ),
        (
            sequential.model.predictive.prediction_error,
            quantum.model.predictive.prediction_error,
        ),
        (
            sequential.model.predictive.prediction_failure_streak,
            quantum.model.predictive.prediction_failure_streak,
        ),
        (
            sequential.model.predictive.confidence,
            quantum.model.predictive.confidence,
        ),
    ):
        assert torch.equal(sequential_tensor, quantum_tensor)
    assert quantum.token_count == sequential.token_count == 9
    assert (
        quantum.model.competitive.recent_spike_window_cursor
        == sequential.model.competitive.recent_spike_window_cursor
    )
    assert quantum.last_winner == sequential.last_winner
    assert (
        quantum._winner_host_mirror_sync_count
        == sequential._winner_host_mirror_sync_count
    )
    assert (
        quantum._winner_host_mirror_skip_count
        == sequential._winner_host_mirror_skip_count
    )
    runtime_report = quantum.column_transition_runtime_report()
    report = runtime_report["cuda_graph_route_transition"]
    assert report["burst_replay_count"] == 1
    assert report["burst_replayed_token_count"] == 8
    assert report["burst_replay_failure_count"] == 0
    assert runtime_report["text_burst_execution_count"] == 1
    assert runtime_report["text_burst_token_count"] == 8
    assert runtime_report["text_burst_fallback_count"] == 0
    assert runtime_report["text_burst_fallback_reasons"] == {}
    assert runtime_report["text_burst_strong_event_count"] == 0
    assert runtime_report["text_burst_event_deferred_apply_skip_count"] == 0
    assert report["burst_event_ring_device_owned"] is True
    assert report["burst_event_strong_count_device_owned"] is True
    assert report["quantum_input_stage_count"] == 1
    assert report["quantum_input_staged_token_count"] == 8
    assert report["quantum_input_reuse_count"] == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_text_burst_defers_slow_memory_cadence_without_fallback() -> None:
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
        cuda_graph_host_truth_sync_interval_tokens=9,
        slow_memory_start_tokens=0,
        slow_memory_archive_interval_tokens=8,
        slow_memory_archive_strong_capture_threshold=10.0,
        trainer_telemetry_interval_tokens=64,
        device="cuda",
    )
    torch.manual_seed(20260614)
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    trainer.train_step(
        torch.rand(config.input_dim, device="cuda"),
        raw_window="cadence deferred warmup",
        allow_sleep_maintenance=False,
        return_metrics=False,
    )
    patterns = [
        torch.rand(config.input_dim, device="cuda")
        for _ in range(8)
    ]
    raw_windows = [f"cadence deferred burst {index}" for index in range(8)]

    assert trainer.train_text_burst(patterns, raw_windows=raw_windows) is True

    runtime_report = trainer.column_transition_runtime_report()
    controller = runtime_report["cognitive_boundary_controller"]
    assert runtime_report["text_burst_fallback_count"] == 0
    assert runtime_report["text_burst_fallback_reasons"] == {}
    assert runtime_report["text_burst_strong_event_count"] == 0
    assert trainer._slow_memory_archive_count == 1
    assert trainer._slow_memory_archive_skip_count == 9
    assert trainer._slow_memory_last_archive_reason == "cadence_deferred"
    assert trainer.model.memory_store.slow_raw_windows == ["cadence deferred warmup"]
    assert controller["slow_memory_cadence_deferred_count"] == 1
    assert controller["last_slow_memory_cadence_token"] == 8
    assert controller["slow_memory_cadence_execution_gate"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_text_burst_drift_refresh_does_not_force_event_drain() -> None:
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=64,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="cuda_graph_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        cuda_graph_host_truth_sync_interval_tokens=101,
        slow_memory_start_tokens=0,
        slow_memory_archive_interval_tokens=256,
        slow_memory_archive_strong_capture_threshold=10.0,
        trainer_telemetry_interval_tokens=64,
        device="cuda",
    )
    torch.manual_seed(20260614)
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    for index in range(49):
        trainer.train_step(
            torch.rand(config.input_dim, device="cuda"),
            raw_window=f"sync-free drift warmup {index}",
            allow_sleep_maintenance=False,
            return_metrics=False,
        )
    patterns = [
        torch.rand(config.input_dim, device="cuda")
        for _ in range(8)
    ]

    assert trainer.train_text_burst(
        patterns,
        raw_windows=[f"sync-free drift burst {index}" for index in range(8)],
    ) is True

    runtime_report = trainer.column_transition_runtime_report()
    graph_report = runtime_report["cuda_graph_route_transition"]
    controller = runtime_report["cognitive_boundary_controller"]
    assert runtime_report["text_burst_fallback_count"] == 0
    assert runtime_report["text_burst_event_pending_tokens"] == 8
    assert runtime_report["text_burst_event_forced_flush_count"] == 0
    assert runtime_report["text_burst_event_deferred_apply_skip_count"] == 1
    assert graph_report["burst_event_forced_drain_count"] == 0
    assert controller["drift_refresh_count"] == 1
    assert controller["drift_refresh_sync_free_count"] == 1
    assert controller["drift_refresh_global_count"] == 1
    assert controller["drift_refresh_requires_host_truth"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_text_burst_preserves_all_strong_memory_capture_events() -> None:
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
        cuda_graph_host_truth_sync_interval_tokens=9,
        slow_memory_start_tokens=0,
        slow_memory_archive_interval_tokens=256,
        slow_memory_archive_strong_capture_threshold=0.0,
        trainer_telemetry_interval_tokens=64,
        device="cuda",
    )
    torch.manual_seed(20260614)
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    trainer.train_step(
        torch.rand(config.input_dim, device="cuda"),
        raw_window="strong capture warmup",
        allow_sleep_maintenance=False,
        return_metrics=False,
    )
    patterns = [
        torch.rand(config.input_dim, device="cuda")
        for _ in range(8)
    ]
    raw_windows = [f"strong capture burst {index}" for index in range(8)]

    assert trainer.train_text_burst(
        patterns,
        raw_windows=raw_windows,
    ) is True

    assert trainer._slow_memory_archive_count == 9
    assert trainer._slow_memory_archive_skip_count == 0
    assert trainer._slow_memory_last_archive_reason == "strong_capture"
    runtime_report = trainer.column_transition_runtime_report()
    assert runtime_report["text_burst_strong_event_count"] == 8
    assert trainer.model.memory_store.slow_raw_windows[-8:] == raw_windows
    assert trainer.model.memory_store.slow_last_capture_token[-8:] == list(
        range(2, 10)
    )
    assert all(
        pattern is not None and pattern.device.type == "cpu"
        for pattern in trainer.model.memory_store.slow_input_patterns[-8:]
    )
    assert all(
        routing is not None and routing.device.type == "cpu"
        for routing in trainer.model.memory_store.slow_routing_keys[-8:]
    )
    assert all(
        value >= 0.0
        for value in trainer.model.memory_store.slow_capture_tag[-8:]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_text_burst_defers_host_truth_across_two_quanta() -> None:
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=32,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="cuda_graph_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        cuda_graph_host_truth_sync_interval_tokens=17,
        slow_memory_start_tokens=0,
        slow_memory_archive_interval_tokens=256,
        slow_memory_archive_strong_capture_threshold=10.0,
        trainer_telemetry_interval_tokens=64,
        device="cuda",
    )
    torch.manual_seed(20260614)
    sequential = MarulhoTrainer(MarulhoModel(config), config)
    torch.manual_seed(20260614)
    quantum = MarulhoTrainer(MarulhoModel(config), config)
    warm_pattern = torch.rand(config.input_dim, device="cuda")
    patterns = [
        torch.rand(config.input_dim, device="cuda")
        for _ in range(16)
    ]
    for trainer in (sequential, quantum):
        trainer.train_step(
            warm_pattern,
            raw_window="two quantum warmup",
            allow_sleep_maintenance=False,
            return_metrics=False,
        )
    for index, pattern in enumerate(patterns):
        sequential.train_step(
            pattern,
            raw_window=f"sequential two quantum {index}",
            allow_sleep_maintenance=False,
            return_metrics=False,
        )

    assert quantum.train_text_burst(
        patterns[:8],
        raw_windows=[f"deferred quantum {index}" for index in range(8)],
    ) is True
    deferred = quantum.column_transition_runtime_report()
    assert deferred["text_burst_event_pending_tokens"] == 8
    assert deferred["cuda_graph_route_transition"]["burst_event_pending_tokens"] == 8
    assert deferred["cuda_graph_route_transition"]["burst_event_drain_count"] == 0
    assert quantum.train_text_burst(
        patterns[8:],
        raw_windows=[f"drained quantum {index}" for index in range(8, 16)],
    ) is True
    torch.cuda.synchronize()

    for sequential_tensor, quantum_tensor in (
        (
            sequential.model.competitive.prototypes,
            quantum.model.competitive.prototypes,
        ),
        (
            sequential.model.competitive.prototype_velocity,
            quantum.model.competitive.prototype_velocity,
        ),
        (
            sequential.model.competitive.thresholds,
            quantum.model.competitive.thresholds,
        ),
        (
            sequential.model.competitive.recent_spike_window,
            quantum.model.competitive.recent_spike_window,
        ),
        (
            sequential.model.predictive.location,
            quantum.model.predictive.location,
        ),
        (
            sequential.model.predictive.velocity,
            quantum.model.predictive.velocity,
        ),
        (
            sequential.model.predictive._prediction_weights,
            quantum.model.predictive._prediction_weights,
        ),
        (
            sequential.model.predictive.prediction_error,
            quantum.model.predictive.prediction_error,
        ),
    ):
        assert torch.equal(sequential_tensor, quantum_tensor)
    report = quantum.column_transition_runtime_report()
    graph_report = report["cuda_graph_route_transition"]
    assert quantum.token_count == sequential.token_count == 17
    assert quantum.last_winner == sequential.last_winner
    assert report["text_burst_event_pending_tokens"] == 0
    assert report["text_burst_event_flush_count"] == 1
    assert graph_report["burst_event_capacity_tokens"] == 32
    assert graph_report["burst_event_deferred_count"] == 1
    assert graph_report["burst_event_drain_count"] == 1
    assert graph_report["burst_event_drained_token_count"] == 16
    assert graph_report["burst_event_forced_drain_count"] == 0
    assert graph_report["burst_event_slim_result_packet_count"] == 1
    assert graph_report["burst_event_strong_result_row_count"] == 0
    assert graph_report["burst_event_strong_flag_scan_count"] == 0
    assert graph_report["burst_event_no_strong_flag_scan_skip_count"] == 1
    assert graph_report["burst_event_strong_count_total"] == 0
    assert graph_report["burst_event_slot_reset_count"] == 1
    assert graph_report["burst_event_slot_reset_skip_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_text_burst_forced_flush_preserves_pending_strong_events() -> None:
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
        cuda_graph_host_truth_sync_interval_tokens=17,
        slow_memory_start_tokens=0,
        slow_memory_archive_interval_tokens=256,
        slow_memory_archive_strong_capture_threshold=0.0,
        trainer_telemetry_interval_tokens=64,
        device="cuda",
    )
    torch.manual_seed(20260614)
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    trainer.train_step(
        torch.rand(config.input_dim, device="cuda"),
        raw_window="forced flush warmup",
        allow_sleep_maintenance=False,
        return_metrics=False,
    )
    patterns = [
        torch.rand(config.input_dim, device="cuda")
        for _ in range(8)
    ]
    raw_windows = [f"forced flush burst {index}" for index in range(8)]

    assert trainer.train_text_burst(patterns, raw_windows=raw_windows) is True
    assert trainer.column_transition_runtime_report()[
        "text_burst_event_pending_tokens"
    ] == 8
    assert trainer.flush_text_burst_events(reason="test_boundary") is True

    report = trainer.column_transition_runtime_report()
    graph_report = report["cuda_graph_route_transition"]
    assert report["text_burst_event_pending_tokens"] == 0
    assert report["text_burst_event_forced_flush_count"] == 1
    assert report["text_burst_event_last_flush_reason"] == "test_boundary"
    assert report["text_burst_strong_event_count"] == 8
    assert graph_report["burst_event_forced_drain_count"] == 1
    assert graph_report["burst_event_slim_result_packet_count"] == 1
    assert graph_report["burst_event_strong_result_row_count"] == 8
    assert graph_report["burst_event_strong_flag_scan_count"] == 1
    assert graph_report["burst_event_no_strong_flag_scan_skip_count"] == 0
    assert graph_report["burst_event_strong_count_total"] == 8
    assert graph_report["burst_event_slot_reset_count"] == 1
    assert graph_report["burst_event_slot_reset_skip_count"] == 0
    assert trainer.model.memory_store.slow_raw_windows[-8:] == raw_windows
    assert trainer.model.memory_store.slow_last_capture_token[-8:] == list(
        range(2, 10)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_text_burst_crosses_telemetry_observation_without_fallback() -> None:
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
        cuda_graph_host_truth_sync_interval_tokens=9,
        slow_memory_start_tokens=0,
        slow_memory_archive_interval_tokens=256,
        trainer_telemetry_interval_tokens=4,
        device="cuda",
    )
    torch.manual_seed(20260614)
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    trainer.train_step(
        torch.rand(config.input_dim, device="cuda"),
        raw_window="boundary controller warmup",
        allow_sleep_maintenance=False,
        return_metrics=False,
    )
    patterns = [
        torch.rand(config.input_dim, device="cuda")
        for _ in range(8)
    ]

    assert trainer.train_text_burst(
        patterns,
        raw_windows=[f"boundary controller {index}" for index in range(8)],
    ) is True

    report = trainer.column_transition_runtime_report()
    controller = report["cognitive_boundary_controller"]
    assert report["text_burst_fallback_count"] == 0
    assert controller["device_continuous_count"] == 1
    assert controller["telemetry_observation_deferred_count"] == 1
    assert controller["telemetry_execution_gate"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_text_burst_profile_keeps_burst_executor_active() -> None:
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
        cuda_graph_host_truth_sync_interval_tokens=9,
        slow_memory_start_tokens=0,
        slow_memory_archive_interval_tokens=256,
        trainer_telemetry_interval_tokens=64,
        device="cuda",
    )
    torch.manual_seed(20260614)
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    trainer.train_step(
        torch.rand(config.input_dim, device="cuda"),
        raw_window="profile burst warmup",
        allow_sleep_maintenance=False,
        return_metrics=False,
    )
    trainer.enable_train_step_profile(reset=True)
    patterns = [
        torch.rand(config.input_dim, device="cuda")
        for _ in range(8)
    ]

    assert trainer.train_text_burst(
        patterns,
        raw_windows=[f"profile burst {index}" for index in range(8)],
    ) is True

    runtime_report = trainer.column_transition_runtime_report()
    profile = trainer.train_step_profile_report()
    assert runtime_report["text_burst_execution_count"] == 1
    assert runtime_report["text_burst_fallback_reasons"] == {}
    assert profile["count"] == 8
    assert profile["totals_ms"]["text_burst_graph_replay"] > 0.0
    assert profile["totals_ms"]["text_burst_runtime_replay_loop"] > 0.0
    assert profile["totals_ms"]["text_burst_runtime_python_mirrors"] > 0.0
    assert profile["totals_ms"]["text_burst_total"] > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_training_owned_text_sequence_matches_sequential_cuda_ticks() -> None:
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=128,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="cuda_graph_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        cuda_graph_host_truth_sync_interval_tokens=9,
        slow_memory_start_tokens=0,
        slow_memory_archive_interval_tokens=256,
        slow_memory_archive_strong_capture_threshold=10.0,
        trainer_telemetry_interval_tokens=64,
        device="cuda",
    )
    torch.manual_seed(20260614)
    sequential = MarulhoTrainer(MarulhoModel(config), config)
    torch.manual_seed(20260614)
    sequence = MarulhoTrainer(MarulhoModel(config), config)
    warm_pattern = torch.rand(config.input_dim, device="cuda")
    patterns = [
        torch.rand(config.input_dim, device="cuda")
        for _ in range(64)
    ]
    raw_windows = [
        f"training owned sequence {index}"
        for index in range(64)
    ]
    for trainer in (sequential, sequence):
        trainer.train_step(
            warm_pattern,
            raw_window="training owned sequence warmup",
            allow_sleep_maintenance=False,
            return_metrics=False,
        )
    for pattern, raw_window in zip(patterns, raw_windows):
        sequential.train_step(
            pattern,
            raw_window=raw_window,
            allow_sleep_maintenance=False,
            return_metrics=False,
        )

    result = sequence.train_text_sequence(
        patterns,
        raw_windows=raw_windows,
        quantum_tokens=8,
        metric_indices={63},
    )
    torch.cuda.synchronize()

    assert result["trained"] == 64
    assert result["quantum_count"] == 8
    assert set(result["metrics_by_index"]) == {63}
    for sequential_tensor, sequence_tensor in (
        (
            sequential.model.competitive.prototypes,
            sequence.model.competitive.prototypes,
        ),
        (
            sequential.model.competitive.prototype_velocity,
            sequence.model.competitive.prototype_velocity,
        ),
        (
            sequential.model.competitive.thresholds,
            sequence.model.competitive.thresholds,
        ),
        (
            sequential.model.competitive.recent_spike_window,
            sequence.model.competitive.recent_spike_window,
        ),
        (
            sequential.model.predictive.location,
            sequence.model.predictive.location,
        ),
        (
            sequential.model.predictive.velocity,
            sequence.model.predictive.velocity,
        ),
        (
            sequential.model.predictive._prediction_weights,
            sequence.model.predictive._prediction_weights,
        ),
        (
            sequential.model.predictive.prediction_error,
            sequence.model.predictive.prediction_error,
        ),
    ):
        assert torch.equal(sequential_tensor, sequence_tensor)
    assert sequence.token_count == sequential.token_count == 65
    report = sequence.column_transition_runtime_report()
    assert report["text_sequence_execution_count"] == 1
    assert report["text_sequence_token_count"] == 64
    assert report["text_sequence_quantum_count"] == 8
    assert report["text_sequence_stop_count"] == 0
    assert report["text_sequence_owner"] == "training"
    assert report["text_sequence_stop_boundary"] == "between_quanta"
    assert report["text_burst_execution_count"] >= 1
    assert report["cuda_graph_route_transition"]["burst_replay_failure_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_training_owned_wide_quantum_uses_exact_device_bursts() -> None:
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=128,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="cuda_graph_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        cuda_graph_host_truth_sync_interval_tokens=33,
        cuda_graph_sequence_executor="native_repeated_child_graph",
        slow_memory_start_tokens=0,
        slow_memory_archive_interval_tokens=256,
        slow_memory_archive_strong_capture_threshold=10.0,
        trainer_telemetry_interval_tokens=64,
        device="cuda",
    )
    torch.manual_seed(20260614)
    sequential = MarulhoTrainer(MarulhoModel(config), config)
    torch.manual_seed(20260614)
    sequence = MarulhoTrainer(MarulhoModel(config), config)
    warm_pattern = torch.rand(config.input_dim, device="cuda")
    patterns = [
        torch.rand(config.input_dim, device="cuda")
        for _ in range(32)
    ]
    raw_windows = [
        f"wide training owned sequence {index}"
        for index in range(32)
    ]
    for trainer in (sequential, sequence):
        trainer.train_step(
            warm_pattern,
            raw_window="wide training owned sequence warmup",
            allow_sleep_maintenance=False,
            return_metrics=False,
        )
    for pattern, raw_window in zip(patterns, raw_windows):
        sequential.train_step(
            pattern,
            raw_window=raw_window,
            allow_sleep_maintenance=False,
            return_metrics=False,
        )

    result = sequence.train_text_sequence(
        patterns,
        raw_windows=raw_windows,
        quantum_tokens=16,
        metric_indices=set(),
    )
    torch.cuda.synchronize()

    assert result["trained"] == 32
    assert result["quantum_count"] == 2
    assert result["metrics_by_index"] == {}
    for sequential_tensor, sequence_tensor in (
        (
            sequential.model.competitive.prototypes,
            sequence.model.competitive.prototypes,
        ),
        (
            sequential.model.competitive.prototype_velocity,
            sequence.model.competitive.prototype_velocity,
        ),
        (
            sequential.model.competitive.thresholds,
            sequence.model.competitive.thresholds,
        ),
        (
            sequential.model.competitive.recent_spike_window,
            sequence.model.competitive.recent_spike_window,
        ),
        (
            sequential.model.predictive.location,
            sequence.model.predictive.location,
        ),
        (
            sequential.model.predictive.velocity,
            sequence.model.predictive.velocity,
        ),
        (
            sequential.model.predictive._prediction_weights,
            sequence.model.predictive._prediction_weights,
        ),
        (
            sequential.model.predictive.prediction_error,
            sequence.model.predictive.prediction_error,
        ),
    ):
        assert torch.equal(sequential_tensor, sequence_tensor)
    assert sequence.token_count == sequential.token_count == 33
    report = sequence.column_transition_runtime_report()
    graph_report = report["cuda_graph_route_transition"]
    assert report["text_sequence_execution_count"] == 1
    assert report["text_sequence_token_count"] == 32
    assert report["text_sequence_quantum_count"] == 2
    assert report["text_sequence_input_staging_enabled"] is True
    assert report["text_sequence_input_stage_count"] == 1
    assert report["text_sequence_input_staged_token_count"] == 32
    assert report["text_sequence_input_stage_skip_count"] == 0
    assert report["text_burst_execution_count"] == 4
    assert report["text_burst_token_count"] == 32
    assert report["text_burst_fallback_count"] == 0
    assert graph_report["persistent_executor_burst_tokens"] == 8
    assert graph_report["native_burst_replay_configured"] is True
    assert graph_report["native_burst_replay_enabled"] is True
    assert graph_report["native_partial_burst_replay_enabled"] is False
    assert graph_report["native_burst_replay_backend"] == "native_repeated_child_graph"
    assert graph_report["native_burst_replay_parent_graph_count"] == 2
    assert graph_report["native_burst_replay_parent_graph_token_counts"] == [8]
    assert graph_report["native_burst_replay_success_count"] == 4
    assert graph_report["native_burst_replay_token_count"] == 32
    assert graph_report["native_burst_replay_lazy_compile_count"] == 0
    assert graph_report["native_burst_replay_python_loop_token_count"] == 0
    assert graph_report["native_burst_replay_fallback_count"] == 0
    assert graph_report["native_burst_replay_failure_count"] == 0
    assert graph_report["burst_replay_count"] == 4
    assert graph_report["burst_event_drain_count"] == 1
    assert graph_report["burst_event_drained_token_count"] == 32
    assert graph_report["burst_event_slot_reset_count"] == 0
    assert graph_report["burst_event_slot_reset_skip_count"] == 1
    assert graph_report["burst_event_strong_flag_scan_count"] == 0
    assert graph_report["burst_event_no_strong_flag_scan_skip_count"] == 1
    assert graph_report["burst_event_strong_count_total"] == 0
    assert graph_report["burst_replay_failure_count"] == 0
    assert graph_report["quantum_input_stage_count"] == 1
    assert graph_report["quantum_input_staged_token_count"] == 32
    assert graph_report["quantum_input_reuse_count"] == 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_training_owned_sequence_can_use_conditional_while_executor() -> None:
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=128,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="cuda_graph_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        cuda_graph_host_truth_sync_interval_tokens=33,
        cuda_graph_sequence_executor="conditional_while",
        cuda_graph_sequence_loop_tokens=16,
        slow_memory_start_tokens=0,
        slow_memory_archive_interval_tokens=256,
        slow_memory_archive_strong_capture_threshold=10.0,
        trainer_telemetry_interval_tokens=64,
        device="cuda",
    )
    torch.manual_seed(20260615)
    sequential = MarulhoTrainer(MarulhoModel(config), config)
    torch.manual_seed(20260615)
    sequence = MarulhoTrainer(MarulhoModel(config), config)
    warm_pattern = torch.rand(config.input_dim, device="cuda")
    patterns = [
        torch.rand(config.input_dim, device="cuda")
        for _ in range(32)
    ]
    raw_windows = [
        f"conditional while graph {index}"
        for index in range(32)
    ]
    for trainer in (sequential, sequence):
        trainer.train_step(
            warm_pattern,
            raw_window="conditional while graph warmup",
            allow_sleep_maintenance=False,
            return_metrics=False,
        )
    for pattern, raw_window in zip(patterns, raw_windows):
        sequential.train_step(
            pattern,
            raw_window=raw_window,
            allow_sleep_maintenance=False,
            return_metrics=False,
        )

    result = sequence.train_text_sequence(
        patterns,
        raw_windows=raw_windows,
        quantum_tokens=16,
        metric_indices=set(),
    )
    torch.cuda.synchronize()

    assert result["trained"] == 32
    assert result["quantum_count"] == 2
    for sequential_tensor, sequence_tensor in (
        (
            sequential.model.competitive.prototypes,
            sequence.model.competitive.prototypes,
        ),
        (
            sequential.model.competitive.prototype_velocity,
            sequence.model.competitive.prototype_velocity,
        ),
        (
            sequential.model.competitive.thresholds,
            sequence.model.competitive.thresholds,
        ),
        (
            sequential.model.competitive.recent_spike_window,
            sequence.model.competitive.recent_spike_window,
        ),
        (
            sequential.model.predictive.location,
            sequence.model.predictive.location,
        ),
        (
            sequential.model.predictive.velocity,
            sequence.model.predictive.velocity,
        ),
        (
            sequential.model.predictive._prediction_weights,
            sequence.model.predictive._prediction_weights,
        ),
        (
            sequential.model.predictive.prediction_error,
            sequence.model.predictive.prediction_error,
        ),
    ):
        assert torch.equal(sequential_tensor, sequence_tensor)
    report = sequence.column_transition_runtime_report()
    graph_report = report["cuda_graph_route_transition"]
    assert report["text_burst_execution_count"] == 2
    assert report["text_burst_token_count"] == 32
    assert report["text_burst_fallback_count"] == 0
    assert graph_report["native_burst_replay_backend"] == (
        "cuda_graph_conditional_while"
    )
    assert graph_report["persistent_executor_burst_tokens"] == 16
    assert graph_report["persistent_executor_repeated_child_burst_tokens"] == 8
    assert graph_report["persistent_executor_sequence_loop_tokens"] == 16
    assert graph_report["native_burst_replay_parent_graph_count"] == 0
    assert graph_report["native_sequence_executor_requested"] == (
        "cuda_graph_conditional_while"
    )
    assert (
        graph_report[
            "native_sequence_loop_sequential_state_parity_gate_status"
        ]
        == "passed_focused_cuda_state_parity"
    )
    assert (
        graph_report["native_sequence_loop_sequential_state_parity_gate_passed"]
        is True
    )
    assert (
        graph_report["native_sequence_loop_bounded_quality_gate_status"]
        == "passed_retained_one_tick_graph_body_quality_boundary"
    )
    assert graph_report["native_sequence_loop_bounded_quality_gate_passed"] is True
    assert graph_report["native_sequence_loop_loaded"] is True
    assert graph_report["native_sequence_loop_backend"] == (
        "cuda_graph_conditional_while"
    )
    assert graph_report["native_sequence_loop_parent_graph_count"] == 2
    assert graph_report["native_sequence_loop_parent_graph_token_counts"] == [16]
    assert graph_report["native_sequence_loop_success_count"] == 2
    assert graph_report["native_sequence_loop_token_count"] == 32
    assert graph_report["native_sequence_loop_fallback_count"] == 0
    assert graph_report["native_sequence_loop_failure_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_conditional_while_unavailable_falls_back_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marulho.training import cuda_graph_route_transition as graph_module

    monkeypatch.setattr(
        graph_module,
        "native_cuda_graph_sequence_error",
        lambda: "forced conditional sequence unavailable",
    )
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=128,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="cuda_graph_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        cuda_graph_host_truth_sync_interval_tokens=33,
        cuda_graph_sequence_executor="conditional_while",
        slow_memory_start_tokens=0,
        slow_memory_archive_interval_tokens=256,
        slow_memory_archive_strong_capture_threshold=10.0,
        trainer_telemetry_interval_tokens=64,
        device="cuda",
    )
    torch.manual_seed(20260615)
    sequential = MarulhoTrainer(MarulhoModel(config), config)
    torch.manual_seed(20260615)
    sequence = MarulhoTrainer(MarulhoModel(config), config)
    warm_pattern = torch.rand(config.input_dim, device="cuda")
    patterns = [
        torch.rand(config.input_dim, device="cuda")
        for _ in range(16)
    ]
    raw_windows = [
        f"conditional unavailable fallback {index}"
        for index in range(16)
    ]
    for trainer in (sequential, sequence):
        trainer.train_step(
            warm_pattern,
            raw_window="conditional unavailable fallback warmup",
            allow_sleep_maintenance=False,
            return_metrics=False,
        )
    for pattern, raw_window in zip(patterns, raw_windows):
        sequential.train_step(
            pattern,
            raw_window=raw_window,
            allow_sleep_maintenance=False,
            return_metrics=False,
        )

    result = sequence.train_text_sequence(
        patterns,
        raw_windows=raw_windows,
        quantum_tokens=16,
        metric_indices=set(),
    )
    torch.cuda.synchronize()

    assert result["trained"] == 16
    for sequential_tensor, sequence_tensor in (
        (
            sequential.model.competitive.prototypes,
            sequence.model.competitive.prototypes,
        ),
        (
            sequential.model.competitive.prototype_velocity,
            sequence.model.competitive.prototype_velocity,
        ),
        (
            sequential.model.competitive.thresholds,
            sequence.model.competitive.thresholds,
        ),
        (
            sequential.model.competitive.recent_spike_window,
            sequence.model.competitive.recent_spike_window,
        ),
        (
            sequential.model.predictive.location,
            sequence.model.predictive.location,
        ),
        (
            sequential.model.predictive.velocity,
            sequence.model.predictive.velocity,
        ),
        (
            sequential.model.predictive._prediction_weights,
            sequence.model.predictive._prediction_weights,
        ),
        (
            sequential.model.predictive.prediction_error,
            sequence.model.predictive.prediction_error,
        ),
    ):
        assert torch.equal(sequential_tensor, sequence_tensor)
    report = sequence.column_transition_runtime_report()
    graph_report = report["cuda_graph_route_transition"]
    assert report["text_burst_execution_count"] == 2
    assert report["text_burst_token_count"] == 16
    assert report["text_burst_fallback_count"] == 0
    assert graph_report["native_sequence_executor_requested"] == (
        "cuda_graph_conditional_while"
    )
    assert graph_report["native_sequence_loop_loaded"] is False
    assert (
        graph_report[
            "native_sequence_loop_sequential_state_parity_gate_status"
        ]
        == "not_exercised_fallback_before_mutation"
    )
    assert (
        graph_report["native_sequence_loop_sequential_state_parity_gate_passed"]
        is False
    )
    assert (
        graph_report["native_sequence_loop_bounded_quality_gate_status"]
        == "not_exercised_fallback_before_mutation"
    )
    assert graph_report["native_sequence_loop_bounded_quality_gate_passed"] is False
    assert graph_report["native_sequence_loop_success_count"] == 0
    assert graph_report["native_sequence_loop_token_count"] == 0
    assert graph_report["native_sequence_loop_fallback_count"] == 1
    assert graph_report["native_sequence_loop_failure_count"] == 0
    assert graph_report["native_sequence_loop_parent_graph_count"] == 0
    assert graph_report["persistent_executor_burst_tokens"] == 8
    assert graph_report["persistent_executor_repeated_child_burst_tokens"] == 8
    assert graph_report["persistent_executor_sequence_loop_tokens"] == 16
    assert graph_report["native_burst_replay_backend"] == (
        "native_repeated_child_graph"
    )
    assert graph_report["native_burst_replay_success_count"] == 2
    assert graph_report["native_burst_replay_token_count"] == 16
    assert graph_report["native_burst_replay_fallback_count"] == 0
    assert graph_report["native_burst_replay_failure_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_training_owned_sequence_can_use_startup_warmed_sixteen_token_parent_graph() -> None:
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=128,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="cuda_graph_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        cuda_graph_host_truth_sync_interval_tokens=33,
        cuda_graph_native_burst_tokens=16,
        cuda_graph_sequence_executor="native_repeated_child_graph",
        slow_memory_start_tokens=0,
        slow_memory_archive_interval_tokens=256,
        slow_memory_archive_strong_capture_threshold=10.0,
        trainer_telemetry_interval_tokens=64,
        device="cuda",
    )
    torch.manual_seed(20260615)
    sequential = MarulhoTrainer(MarulhoModel(config), config)
    torch.manual_seed(20260615)
    sequence = MarulhoTrainer(MarulhoModel(config), config)
    warm_pattern = torch.rand(config.input_dim, device="cuda")
    patterns = [
        torch.rand(config.input_dim, device="cuda")
        for _ in range(32)
    ]
    raw_windows = [
        f"wide native parent graph {index}"
        for index in range(32)
    ]
    for trainer in (sequential, sequence):
        trainer.train_step(
            warm_pattern,
            raw_window="wide native parent graph warmup",
            allow_sleep_maintenance=False,
            return_metrics=False,
        )
    for pattern, raw_window in zip(patterns, raw_windows):
        sequential.train_step(
            pattern,
            raw_window=raw_window,
            allow_sleep_maintenance=False,
            return_metrics=False,
        )

    result = sequence.train_text_sequence(
        patterns,
        raw_windows=raw_windows,
        quantum_tokens=16,
        metric_indices=set(),
    )
    torch.cuda.synchronize()

    assert result["trained"] == 32
    assert result["quantum_count"] == 2
    for sequential_tensor, sequence_tensor in (
        (
            sequential.model.competitive.prototypes,
            sequence.model.competitive.prototypes,
        ),
        (
            sequential.model.competitive.prototype_velocity,
            sequence.model.competitive.prototype_velocity,
        ),
        (
            sequential.model.competitive.thresholds,
            sequence.model.competitive.thresholds,
        ),
        (
            sequential.model.competitive.recent_spike_window,
            sequence.model.competitive.recent_spike_window,
        ),
        (
            sequential.model.predictive.location,
            sequence.model.predictive.location,
        ),
        (
            sequential.model.predictive.velocity,
            sequence.model.predictive.velocity,
        ),
        (
            sequential.model.predictive._prediction_weights,
            sequence.model.predictive._prediction_weights,
        ),
        (
            sequential.model.predictive.prediction_error,
            sequence.model.predictive.prediction_error,
        ),
    ):
        assert torch.equal(sequential_tensor, sequence_tensor)
    report = sequence.column_transition_runtime_report()
    graph_report = report["cuda_graph_route_transition"]
    assert report["text_burst_execution_count"] == 2
    assert report["text_burst_token_count"] == 32
    assert report["text_burst_fallback_count"] == 0
    assert graph_report["persistent_executor_burst_tokens"] == 16
    assert graph_report["persistent_executor_default_burst_tokens"] == 16
    assert graph_report["persistent_executor_default_repeated_child_burst_tokens"] == 8
    assert graph_report["persistent_executor_repeated_child_burst_tokens"] == 16
    assert graph_report["persistent_executor_sequence_loop_tokens"] == 16
    assert graph_report["native_burst_replay_parent_graph_count"] == 2
    assert graph_report["native_burst_replay_parent_graph_token_counts"] == [16]
    assert graph_report["native_burst_replay_success_count"] == 2
    assert graph_report["native_burst_replay_token_count"] == 32
    assert graph_report["native_burst_replay_fallback_count"] == 0
    assert graph_report["native_burst_replay_failure_count"] == 0
    assert graph_report["native_burst_replay_python_loop_token_count"] == 0
    assert graph_report["burst_replay_count"] == 2
    assert graph_report["burst_event_drain_count"] == 1
    assert graph_report["burst_event_drained_token_count"] == 32
    assert graph_report["quantum_input_reuse_count"] == 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_training_owned_partial_sequence_uses_lazy_native_parent_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MARULHO_CUDA_GRAPH_NATIVE_PARTIAL_BURST_REPLAY", "1")
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=128,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="cuda_graph_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        cuda_graph_host_truth_sync_interval_tokens=33,
        cuda_graph_sequence_executor="native_repeated_child_graph",
        slow_memory_start_tokens=0,
        slow_memory_archive_interval_tokens=256,
        slow_memory_archive_strong_capture_threshold=10.0,
        trainer_telemetry_interval_tokens=64,
        device="cuda",
    )
    torch.manual_seed(20260615)
    sequential = MarulhoTrainer(MarulhoModel(config), config)
    torch.manual_seed(20260615)
    sequence = MarulhoTrainer(MarulhoModel(config), config)
    warm_pattern = torch.rand(config.input_dim, device="cuda")
    patterns = [
        torch.rand(config.input_dim, device="cuda")
        for _ in range(18)
    ]
    raw_windows = [
        f"partial native burst replay {index}"
        for index in range(18)
    ]
    for trainer in (sequential, sequence):
        trainer.train_step(
            warm_pattern,
            raw_window="partial native burst replay warmup",
            allow_sleep_maintenance=False,
            return_metrics=False,
        )
    for pattern, raw_window in zip(patterns, raw_windows):
        sequential.train_step(
            pattern,
            raw_window=raw_window,
            allow_sleep_maintenance=False,
            return_metrics=False,
        )

    result = sequence.train_text_sequence(
        patterns,
        raw_windows=raw_windows,
        quantum_tokens=16,
        metric_indices=set(),
    )
    torch.cuda.synchronize()

    assert result["trained"] == 18
    assert result["quantum_count"] == 2
    for sequential_tensor, sequence_tensor in (
        (
            sequential.model.competitive.prototypes,
            sequence.model.competitive.prototypes,
        ),
        (
            sequential.model.competitive.prototype_velocity,
            sequence.model.competitive.prototype_velocity,
        ),
        (
            sequential.model.competitive.thresholds,
            sequence.model.competitive.thresholds,
        ),
        (
            sequential.model.competitive.recent_spike_window,
            sequence.model.competitive.recent_spike_window,
        ),
        (
            sequential.model.predictive.location,
            sequence.model.predictive.location,
        ),
        (
            sequential.model.predictive.velocity,
            sequence.model.predictive.velocity,
        ),
        (
            sequential.model.predictive._prediction_weights,
            sequence.model.predictive._prediction_weights,
        ),
        (
            sequential.model.predictive.prediction_error,
            sequence.model.predictive.prediction_error,
        ),
    ):
        assert torch.equal(sequential_tensor, sequence_tensor)
    assert sequence.token_count == sequential.token_count == 19
    report = sequence.column_transition_runtime_report()
    graph_report = report["cuda_graph_route_transition"]
    assert report["text_burst_execution_count"] == 3
    assert report["text_burst_token_count"] == 18
    assert report["text_burst_fallback_count"] == 0
    assert graph_report["native_partial_burst_replay_enabled"] is True
    assert graph_report["native_burst_replay_parent_graph_count"] == 3
    assert graph_report["native_burst_replay_parent_graph_token_counts"] == [2, 8]
    assert graph_report["native_burst_replay_success_count"] == 3
    assert graph_report["native_burst_replay_token_count"] == 18
    assert graph_report["native_burst_replay_lazy_compile_count"] == 1
    assert graph_report["native_burst_replay_lazy_compile_failure_count"] == 0
    assert graph_report["native_burst_replay_fallback_count"] == 0
    assert graph_report["native_burst_replay_python_loop_token_count"] == 0
    assert graph_report["native_burst_replay_failure_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_training_owned_sequence_input_staging_segments_around_host_truth_phase() -> None:
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=256,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="cuda_graph_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        cuda_graph_host_truth_sync_interval_tokens=32,
        cuda_graph_sequence_executor="native_repeated_child_graph",
        slow_memory_start_tokens=0,
        slow_memory_archive_interval_tokens=256,
        slow_memory_archive_strong_capture_threshold=10.0,
        trainer_telemetry_interval_tokens=256,
        device="cuda",
    )
    torch.manual_seed(20260614)
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    trainer.train_step(
        torch.rand(config.input_dim, device="cuda"),
        raw_window="sequence staging off-phase warmup",
        allow_sleep_maintenance=False,
        return_metrics=False,
    )
    patterns = [
        torch.rand(config.input_dim, device="cuda")
        for _ in range(128)
    ]

    result = trainer.train_text_sequence(
        patterns,
        raw_windows=[f"sequence staging off-phase {index}" for index in range(128)],
        quantum_tokens=16,
        metric_indices=set(),
    )
    torch.cuda.synchronize()

    report = trainer.column_transition_runtime_report()
    graph_report = report["cuda_graph_route_transition"]
    assert result["trained"] == 128
    assert report["text_sequence_input_stage_count"] == 4
    assert report["text_sequence_input_staged_token_count"] == 96
    assert report["text_sequence_input_stage_skip_count"] == 0
    assert report["text_burst_execution_count"] == 12
    assert report["text_burst_token_count"] == 96
    assert report["text_burst_fallback_reasons"] == {"host_truth_boundary": 4}
    assert graph_report["quantum_input_stage_count"] == 8
    assert graph_report["quantum_input_staged_token_count"] == 128
    assert graph_report["quantum_input_reuse_count"] == 128


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_training_owned_sequence_input_staging_can_be_disabled() -> None:
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=128,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="cuda_graph_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        cuda_graph_host_truth_sync_interval_tokens=33,
        cuda_graph_sequence_input_staging=False,
        slow_memory_start_tokens=0,
        slow_memory_archive_interval_tokens=256,
        slow_memory_archive_strong_capture_threshold=10.0,
        trainer_telemetry_interval_tokens=64,
        device="cuda",
    )
    torch.manual_seed(20260614)
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    trainer.train_step(
        torch.rand(config.input_dim, device="cuda"),
        raw_window="sequence staging disabled warmup",
        allow_sleep_maintenance=False,
        return_metrics=False,
    )
    patterns = [
        torch.rand(config.input_dim, device="cuda")
        for _ in range(32)
    ]

    result = trainer.train_text_sequence(
        patterns,
        raw_windows=[f"sequence staging disabled {index}" for index in range(32)],
        quantum_tokens=16,
        metric_indices=set(),
    )
    torch.cuda.synchronize()

    report = trainer.column_transition_runtime_report()
    graph_report = report["cuda_graph_route_transition"]
    assert result["trained"] == 32
    assert report["text_sequence_input_staging_enabled"] is False
    assert report["text_sequence_input_stage_count"] == 0
    assert report["text_sequence_input_staged_token_count"] == 0
    assert graph_report["quantum_input_stage_count"] == 2
    assert graph_report["quantum_input_staged_token_count"] == 32
    assert graph_report["quantum_input_reuse_count"] == 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_training_owned_quantum_does_not_prestage_across_sleep_boundary() -> None:
    config = MarulhoConfig(
        n_columns=32,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=5,
        memory_capacity=128,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="cuda_graph_text",
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        cuda_graph_host_truth_sync_interval_tokens=17,
        cuda_graph_sequence_executor="native_repeated_child_graph",
        slow_memory_start_tokens=0,
        slow_memory_archive_interval_tokens=256,
        slow_memory_archive_strong_capture_threshold=10.0,
        trainer_telemetry_interval_tokens=64,
        micro_sleep_interval_tokens=9,
        deep_sleep_interval_tokens=10**9,
        device="cuda",
    )
    torch.manual_seed(20260614)
    trainer = MarulhoTrainer(MarulhoModel(config), config)
    trainer.train_step(
        torch.rand(config.input_dim, device="cuda"),
        raw_window="boundary preflight warmup",
        allow_sleep_maintenance=False,
        return_metrics=False,
    )
    trainer.last_micro_sleep_token = 0
    patterns = [
        torch.rand(config.input_dim, device="cuda")
        for _ in range(16)
    ]
    raw_windows = [
        f"boundary preflight sequence {index}"
        for index in range(16)
    ]

    result = trainer.train_text_sequence(
        patterns,
        raw_windows=raw_windows,
        quantum_tokens=16,
        metric_indices=set(),
    )
    torch.cuda.synchronize()

    report = trainer.column_transition_runtime_report()
    graph_report = report["cuda_graph_route_transition"]
    assert result["trained"] == 16
    assert result["quantum_count"] == 1
    assert report["text_burst_execution_count"] == 1
    assert report["text_burst_token_count"] == 8
    assert report["text_burst_fallback_count"] == 1
    assert report["text_burst_fallback_reasons"] == {"sleep_boundary": 1}
    assert report["text_sequence_input_stage_count"] == 1
    assert report["text_sequence_input_staged_token_count"] == 8
    assert report["text_sequence_input_stage_skip_count"] == 0
    assert graph_report["quantum_input_staged_token_count"] == 16
    assert graph_report["quantum_input_reuse_count"] == 16
    assert graph_report["quantum_input_discard_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_graph_quantum_input_staging_discards_mismatched_order() -> None:
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
    patterns = [
        torch.rand(config.input_dim, device="cuda")
        for _ in range(3)
    ]

    assert trainer.stage_text_input_quantum(patterns[:2]) is True
    trainer.train_step(
        patterns[2],
        raw_window="mismatched staged order",
        allow_sleep_maintenance=False,
    )

    report = trainer.column_transition_runtime_report()[
        "cuda_graph_route_transition"
    ]
    assert report["quantum_input_reuse_count"] == 0
    assert report["quantum_input_fallback_copy_count"] == 1
    assert report["quantum_input_mismatch_count"] == 1
    assert report["quantum_input_discard_count"] == 2
    assert report["failure_count"] == 0


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

    trainer._routing_index_flush_interval = 1
    retained_id = torch.tensor([0], dtype=torch.long, device="cuda")
    trainer._buffer_routing_index_update(
        retained_id,
        trainer.model.competitive.prototypes.index_select(
            0,
            retained_id,
        ),
        known_ids=[0],
    )

    assert trainer._routing_index_cpu_mirror_stale is False
    assert trainer._routing_index_host_mirror_sync_count == 1
    assert trainer._routing_index_buffer_ids == []
    assert trainer._routing_index_buffer_vecs == []
    normalized = torch.nn.functional.normalize(
        trainer.model.competitive.prototypes.detach(),
        dim=1,
    ).cpu()
    for column_id in range(config.n_columns):
        index = trainer.model.routing_index
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
        predictive_route_vote_mode="tensor",
        candidate_predictive_update_start_tokens=0,
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
    assert after["candidate_predictive_transition_execution_count"] == 1
    assert after["candidate_predictive_transition_cached_count"] == (
        config.n_columns - config.k_routing
    )
    assert after["candidate_predictive_transition_fallback_reason"] is None
    assert model.predictive.last_dense_transition_mode == "inplace_triton"
    predictive_report = model.predictive.prediction_update_execution_report()
    assert predictive_report["mode"] == "candidate_subset"
    assert predictive_report["updated_column_count"] == config.k_routing
    assert predictive_report["cached_state_count"] == (
        config.n_columns - config.k_routing
    )
    assert predictive_report["location_update_mode"] == "candidate_subset"
    assert predictive_report["location_update_count"] == config.k_routing
    assert predictive_report["location_update_runs_all_columns"] is False
    assert predictive_report["runs_all_columns"] is False
    assert predictive_report["fallback_reason"] is None

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
