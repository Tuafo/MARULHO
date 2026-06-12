"""Full hot-window A/B runner for the evaluation-only in-place CUDA transition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MethodType

import torch

from marulho.core.inplace_column_cuda import inplace_column_transition_cuda
from marulho.evaluation.hot_window_benchmark import run_hot_window_benchmark
from marulho.training.trainer import MarulhoTrainer


def install_inplace_transition_for_evaluation(trainer_object: object) -> None:
    trainer = trainer_object
    if not isinstance(trainer, MarulhoTrainer):
        raise TypeError("in-place transition setup requires MarulhoTrainer")
    if trainer.model.device.type != "cuda":
        raise RuntimeError("in-place transition evaluation requires CUDA")
    comp = trainer.model.competitive
    if comp.plasticity_mode != "lite" or comp.input_weight_blend != 0.0:
        raise RuntimeError(
            "in-place transition currently supports lite zero-blend plasticity"
        )

    assembly = torch.empty(comp.n_columns, device=trainer.model.device)
    prediction_boost = torch.empty((), device=trainer.model.device)
    effective_modulator = torch.empty((), device=trainer.model.device)
    zero_consolidation = torch.zeros(comp.n_columns, device=trainer.model.device)
    all_columns = torch.arange(comp.n_columns, device=trainer.model.device)
    competition_already_applied_fallback = torch.ones(
        (),
        dtype=torch.bool,
        device=trainer.model.device,
    )
    recent_spike_row = torch.zeros(
        (),
        dtype=torch.int32,
        device=trainer.model.device,
    )

    def _evaluation_transition(
        self: MarulhoTrainer,
        *,
        routing_key: torch.Tensor,
        candidates: torch.Tensor | None,
        winners: torch.Tensor,
        strengths: torch.Tensor,
        modulator: float,
        local_trace: torch.Tensor | None,
        compute_metrics: bool,
    ) -> tuple[
        torch.Tensor,
        list[int],
        int,
        float,
        float,
        float,
        float,
    ]:
        del strengths, local_trace, compute_metrics
        if candidates is None:
            candidates = all_columns
        predictive_scope_ready = self.token_count >= int(
            self.config.dead_column_steps
        )
        homeostasis_candidates = (
            candidates if predictive_scope_ready else all_columns
        )
        consolidation = (
            self.model.memory_store.bucket_consolidation_tensor(
                comp.n_columns,
                device=self.model.device,
            )
            if self.memory_warm_started
            else zero_consolidation
        )
        previous = (
            routing_key
            if self._prev_routing_key is None
            else self._prev_routing_key
        )
        inplace_column_transition_cuda(
            prototypes=comp.prototypes,
            prototype_velocity=comp.prototype_velocity,
            thresholds=comp.thresholds,
            win_rate_ema=comp.win_rate_ema,
            steps_since_win=comp.steps_since_win,
            location=self.model.predictive.location,
            location_velocity=self.model.predictive.velocity,
            prediction_weights=self.model.predictive._prediction_weights,
            prediction_error=self.model.predictive.prediction_error,
            prediction_failure_streak=(
                self.model.predictive.prediction_failure_streak
            ),
            confidence=self.model.predictive.confidence,
            recent_spike_window=comp.recent_spike_window,
            assembly=assembly,
            prediction_boost_out=prediction_boost,
            effective_modulator_out=effective_modulator,
            routing_key=routing_key,
            previous_routing_key=previous,
            winners=winners,
            candidates=homeostasis_candidates,
            consolidation=consolidation,
            base_modulator=float(modulator),
            dopamine=float(self.model.surprise.dopamine),
            serotonin=float(self.model.surprise.serotonin),
            competitive_learning_rate=float(comp.get_lr()),
            recent_spike_row=recent_spike_row,
            has_previous_routing_key=self._prev_routing_key is not None,
            competition_had_positive=competition_already_applied_fallback,
            prototype_momentum=comp.prototype_momentum,
            homeostasis_beta=comp.homeostasis_beta,
            homeostasis_lr=comp.homeostasis_lr,
            target_firing_rate=comp.target_firing_rate,
            threshold_min=comp.threshold_min,
            threshold_max=comp.threshold_max,
            prediction_error_ema_alpha=(
                self.model.predictive._error_ema_alpha
            ),
            prediction_failure_streak_threshold=(
                self.model.predictive._failure_streak_threshold
            ),
            prediction_learning_rate=0.005,
        )

        winner_id_list = winners.tolist()
        winner_id = int(winner_id_list[0])
        winner_consolidation = (
            float(consolidation.index_select(0, winners).mean().item())
            if self.memory_warm_started
            else 0.0
        )
        dopamine_ltp_gain = 0.8 + 0.4 * self.model.surprise.dopamine
        serotonin_patience = max(
            0.2,
            1.0 - 0.6 * self.model.surprise.serotonin,
        )

        self._prev_routing_key = routing_key.detach().clone()
        self.model.predictive.last_dense_transition_mode = (
            "inplace_triton_evaluation"
        )
        self.model.predictive.last_dense_transition_fallback_reason = None
        self.model.predictive._record_prediction_update_scope(None)
        comp.last_input_plasticity_mode = "skipped_zero_blend"
        comp.input_plasticity_skip_count += 1
        comp.last_revived_indices = torch.empty(
            0,
            device=self.model.device,
            dtype=torch.long,
        )
        comp.last_homeostasis_update_count = int(
            homeostasis_candidates.numel()
        )
        comp.last_homeostasis_update_mode = (
            "candidate_subset"
            if int(homeostasis_candidates.numel()) < comp.n_columns
            else "all_columns"
        )
        comp.recent_spike_window_cursor = (
            comp.recent_spike_window_cursor + 1
        ) % comp.spike_history_window
        recent_spike_row.fill_(comp.recent_spike_window_cursor)
        comp.recent_spike_window_count = min(
            comp.spike_history_window,
            comp.recent_spike_window_count + 1,
        )
        comp.update_count += int(winners.numel())
        comp._cached_proto_sim = None
        comp._cached_raw_drive = None
        return (
            assembly,
            winner_id_list,
            winner_id,
            winner_consolidation,
            float(effective_modulator.item()),
            float(dopamine_ltp_gain),
            float(serotonin_patience),
        )

    trainer._apply_awake_column_transition = MethodType(  # type: ignore[method-assign]
        _evaluation_transition,
        trainer,
    )
    trainer._benchmark_transition_executor = "inplace_triton_evaluation"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260612)
    args = parser.parse_args()
    report = run_hot_window_benchmark(
        args.checkpoint,
        samples=args.samples,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        _trainer_setup=install_inplace_transition_for_evaluation,
    )
    report["surface"] = "inplace_hot_window_benchmark.v1"
    report["promotion_status"] = "evaluation_only_pending_repeated_ab"
    report["claim_boundary"] = (
        "complete configured train_step without service/source/sleep; "
        "runtime remains unchanged until repeated reversed-order A/B wins"
    )
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
