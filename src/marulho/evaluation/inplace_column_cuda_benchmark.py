"""Benchmark the evaluation-only in-place CUDA column transition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

from marulho.core.column_transition import steady_state_column_transition
from marulho.core.inplace_column_cuda import inplace_column_transition_cuda
from marulho.training.checkpointing import load_trainer_checkpoint


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def _competition(
    prototypes: torch.Tensor,
    thresholds: torch.Tensor,
    routing_key: torch.Tensor,
    candidates: torch.Tensor,
    context_gain: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = routing_key.clamp(min=1e-6)
    x = x / x.norm().clamp(min=1e-8)
    candidate_prototypes = prototypes.index_select(0, candidates)
    similarity = torch.mv(candidate_prototypes, x)
    combined = similarity * torch.clamp(
        context_gain.index_select(0, candidates),
        min=0.5,
        max=1.5,
    )
    activation = torch.relu(
        combined - thresholds.index_select(0, candidates)
    )
    top_value, top_local_index = torch.topk(activation, k=1)
    has_positive = top_value.max() > 0
    winner_local = torch.where(
        has_positive,
        top_local_index,
        torch.argmax(combined).reshape(1),
    )
    return candidates.index_select(0, winner_local), has_positive


def run_inplace_column_cuda_benchmark(
    checkpoint: Path,
    *,
    iterations: int = 512,
    warmup_iterations: int = 32,
    seed: int = 20260612,
) -> dict[str, object]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")

    trainer, _ = load_trainer_checkpoint(checkpoint)
    model = trainer.model
    if model.device.type != "cuda":
        raise RuntimeError("in-place CUDA benchmark requires a CUDA checkpoint load")
    comp = model.competitive
    predictive = model.predictive
    generator = torch.Generator(device=model.device).manual_seed(seed)
    routing_key = F.normalize(
        torch.rand(
            trainer.config.column_latent_dim,
            generator=generator,
            device=model.device,
        ),
        dim=0,
    )
    previous_routing_key = F.normalize(
        torch.rand(
            trainer.config.column_latent_dim,
            generator=generator,
            device=model.device,
        ),
        dim=0,
    )
    candidates = trainer._routing_candidates(routing_key)
    if candidates is None:
        raise RuntimeError("checkpoint routing index returned no candidates")
    context_gain = torch.ones(comp.n_columns, device=model.device)
    consolidation = model.memory_store.bucket_consolidation_tensor(
        comp.n_columns,
        device=model.device,
    )
    base_modulator = 0.25
    dopamine = float(model.surprise.dopamine)
    serotonin = float(model.surprise.serotonin)
    competitive_learning_rate = float(comp.get_lr())
    initial_state = (
        comp.prototypes.detach().clone(),
        comp.prototype_velocity.detach().clone(),
        comp.thresholds.detach().clone(),
        comp.win_rate_ema.detach().clone(),
        comp.steps_since_win.detach().clone(),
        predictive.location.detach().clone(),
        predictive.velocity.detach().clone(),
        predictive._prediction_weights.detach().clone(),
        predictive.prediction_error.detach().clone(),
        predictive.prediction_failure_streak.detach().clone(),
        predictive.confidence.detach().clone(),
    )

    def measure_functional() -> dict[str, object]:
        state = tuple(value.clone() for value in initial_state)

        def step() -> None:
            outputs = steady_state_column_transition(
                *state,
                routing_key,
                previous_routing_key,
                candidates,
                context_gain,
                consolidation.index_select(0, candidates),
                torch.tensor(base_modulator, device=model.device),
                torch.tensor(dopamine, device=model.device),
                torch.tensor(serotonin, device=model.device),
                torch.tensor(competitive_learning_rate, device=model.device),
                prototype_momentum=comp.prototype_momentum,
                homeostasis_beta=comp.homeostasis_beta,
                homeostasis_lr=comp.homeostasis_lr,
                target_firing_rate=comp.target_firing_rate,
                threshold_min=comp.threshold_min,
                threshold_max=comp.threshold_max,
                candidate_scoped_homeostasis=True,
                prediction_error_ema_alpha=predictive._error_ema_alpha,
                prediction_failure_streak_threshold=(
                    predictive._failure_streak_threshold
                ),
                prediction_learning_rate=0.005,
            )
            for target, source in zip(state, outputs[3:8] + outputs[9:15]):
                target.copy_(source)

        return _measure(
            "functional_eager_stable_writeback",
            step,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            device=model.device,
        )

    def measure_inplace() -> dict[str, object]:
        state = tuple(value.clone() for value in initial_state)
        state_addresses = [value.data_ptr() for value in state]
        recent_spike_window = torch.zeros_like(comp.recent_spike_window)
        recent_spike_row = torch.zeros(
            (),
            dtype=torch.int32,
            device=model.device,
        )
        assembly = torch.empty(comp.n_columns, device=model.device)
        prediction_boost = torch.empty((), device=model.device)
        effective_modulator = torch.empty((), device=model.device)
        row = 0

        def step() -> None:
            nonlocal row
            winners, has_positive = _competition(
                state[0],
                state[2],
                routing_key,
                candidates,
                context_gain,
            )
            inplace_column_transition_cuda(
                prototypes=state[0],
                prototype_velocity=state[1],
                thresholds=state[2],
                win_rate_ema=state[3],
                steps_since_win=state[4],
                location=state[5],
                location_velocity=state[6],
                prediction_weights=state[7],
                prediction_error=state[8],
                prediction_failure_streak=state[9],
                confidence=state[10],
                recent_spike_window=recent_spike_window,
                assembly=assembly,
                prediction_boost_out=prediction_boost,
                effective_modulator_out=effective_modulator,
                routing_key=routing_key,
                previous_routing_key=previous_routing_key,
                winners=winners,
                candidates=candidates,
                consolidation=consolidation,
                base_modulator=base_modulator,
                dopamine=dopamine,
                serotonin=serotonin,
                competitive_learning_rate=competitive_learning_rate,
                recent_spike_row=recent_spike_row,
                has_previous_routing_key=True,
                competition_had_positive=has_positive,
                prototype_momentum=comp.prototype_momentum,
                homeostasis_beta=comp.homeostasis_beta,
                homeostasis_lr=comp.homeostasis_lr,
                target_firing_rate=comp.target_firing_rate,
                threshold_min=comp.threshold_min,
                threshold_max=comp.threshold_max,
                prediction_error_ema_alpha=predictive._error_ema_alpha,
                prediction_failure_streak_threshold=(
                    predictive._failure_streak_threshold
                ),
                prediction_learning_rate=0.005,
            )
            row = (row + 1) % int(recent_spike_window.shape[0])
            recent_spike_row.fill_(row)

        report = _measure(
            "eager_competition_plus_inplace_triton",
            step,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            device=model.device,
        )
        report["finite_state"] = bool(
            all(
                torch.isfinite(value).all().item()
                for value in state
                if value.dtype.is_floating_point
            )
        )
        report["stable_state_addresses"] = bool(
            all(
                value.data_ptr() == initial_ptr
                for value, initial_ptr in zip(
                    state,
                    state_addresses,
                )
            )
        )
        return report

    arms = [measure_functional(), measure_inplace()]
    functional_rate = float(arms[0]["transitions_per_second"])
    inplace_rate = float(arms[1]["transitions_per_second"])
    return {
        "surface": "inplace_column_cuda_benchmark.v1",
        "checkpoint": str(checkpoint),
        "scope": (
            "evaluation_only_competition_plus_one_inplace_triton_launch_for_"
            "predictive_state_prototype_plasticity_homeostasis_spike_history"
        ),
        "promotion_status": "evaluation_only_pending_full_train_step_ab",
        "claim_boundary": (
            "does not include encoder, routing search, context, binding, "
            "cross-modal grounding, memory writes, replay, service, or checkpointing"
        ),
        "device": str(model.device),
        "cuda_device_name": torch.cuda.get_device_name(model.device),
        "n_columns": int(comp.n_columns),
        "candidate_count": int(candidates.numel()),
        "iterations": int(iterations),
        "warmup_iterations": int(warmup_iterations),
        "arms": arms,
        "inplace_speedup_over_functional": (
            inplace_rate / max(functional_rate, 1e-9)
        ),
        "cuda_memory": {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        },
    }


def _measure(
    name: str,
    step,
    *,
    iterations: int,
    warmup_iterations: int,
    device: torch.device,
) -> dict[str, object]:
    warmup_started = time.perf_counter()
    for _ in range(warmup_iterations):
        step()
    torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started
    timings: list[float] = []
    for _ in range(iterations):
        torch.cuda.synchronize(device)
        started = time.perf_counter_ns()
        step()
        torch.cuda.synchronize(device)
        timings.append((time.perf_counter_ns() - started) / 1e6)
    elapsed_s = sum(timings) / 1000.0
    return {
        "name": name,
        "transitions_per_second": iterations / max(elapsed_s, 1e-9),
        "warmup_seconds": warmup_seconds,
        "latency_ms": {
            "median": statistics.median(timings),
            "p95": _percentile(timings, 0.95),
            "mean": statistics.mean(timings),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=512)
    parser.add_argument("--warmup-iterations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260612)
    args = parser.parse_args()
    report = run_inplace_column_cuda_benchmark(
        args.checkpoint,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        seed=args.seed,
    )
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
