from __future__ import annotations

from dataclasses import replace

import torch
import pytest

from marulho.config.model_config import MarulhoConfig
from marulho.core.inplace_column_cuda import (
    select_fused_vote_competition_cuda,
    select_single_winner_cuda,
)
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
        assert torch.equal(retained_tensor, graph_tensor)
    final_report = graph.column_transition_runtime_report()
    graph_runtime = final_report["cuda_graph_route_transition"]
    assert final_report["last_execution_mode"] == "cuda_graph_route_transition"
    assert graph_runtime["pre_route_replay_count"] == 16
    assert graph_runtime["replay_count"] == 16
    assert graph_runtime["failure_count"] == 0


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
