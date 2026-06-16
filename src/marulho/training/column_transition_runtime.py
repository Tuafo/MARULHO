from __future__ import annotations

import time
from typing import Any

import torch

from marulho.core.fused_route_vote_cuda import (
    fused_route_vote_cuda,
    fused_route_vote_kernel_variant,
    warmup_fused_route_vote_cuda,
)
from marulho.core.inplace_column_cuda import (
    inplace_column_transition_cuda,
    select_fused_vote_competition_cuda,
    select_single_winner_cuda,
    warmup_fused_vote_competition_cuda,
    warmup_inplace_column_transition_cuda,
    warmup_single_winner_cuda,
)
from marulho.training.cuda_graph_route_transition import (
    CudaGraphRouteTransition,
)


class ColumnTransitionRuntime:
    """Own the production lifecycle for the in-place CUDA column transition."""

    def __init__(
        self,
        trainer: Any,
        *,
        device_selection: bool = True,
        fused_vote_competition: bool = True,
    ) -> None:
        self._trainer = trainer
        self.device_selection = bool(device_selection)
        self.fused_vote_competition_requested = bool(fused_vote_competition)
        self.requested_mode = str(trainer.config.predictive_dense_transition_mode)
        self.resolved_mode = "retained_runtime"
        self.fallback_reason: str | None = None
        self.warmup_attempted = False
        self.warmup_succeeded = False
        self.warmup_latency_ms = 0.0
        self.precompiled_candidate_counts: list[int] = []
        self.execution_count = 0
        self.failure_count = 0
        self.last_execution_mode = "not_run"
        self.selection_execution_count = 0
        self.selection_failure_count = 0
        self.last_selection_mode = "not_run"
        self.fused_vote_competition_execution_count = 0
        self.fused_vote_competition_fallback_count = 0
        self.route_vote_requested_mode = str(
            getattr(trainer.config, "predictive_route_vote_mode", "tensor")
        )
        self.route_vote_resolved_mode = "tensor"
        self.route_vote_fallback_reason: str | None = None
        self.route_vote_warmup_attempted = False
        self.route_vote_warmup_succeeded = False
        self.route_vote_warmup_latency_ms = 0.0
        self.route_vote_execution_count = 0
        self.route_vote_sensory_fallback_count = 0
        self.route_vote_cache_refresh_count = 0
        self.route_vote_clean_cache_reuse_count = 0
        self.route_vote_prepared_graph_reuse_count = 0
        self.route_vote_kernel_variant = "unavailable"
        self.graph_host_winner_reuse_count = 0
        self.graph_consolidation_lookup_skip_count = 0
        self.graph_empty_revival_tensor_reuse_count = 0
        self.candidate_predictive_transition_mode = "fused_inplace"
        self.candidate_predictive_transition_active = False
        self.candidate_predictive_transition_fallback_reason: str | None = None
        self.candidate_predictive_transition_execution_count = 0
        self.candidate_predictive_transition_cached_count = 0
        self._route_vote_ready = False
        self._route_transition_graph_ready = False
        self._prepared_graph_token: int | None = None
        self._cuda_graph_runtime: CudaGraphRouteTransition | None = None
        self._route_vectors: torch.Tensor | None = None
        self._route_ids: torch.Tensor | None = None
        self._route_scores: torch.Tensor | None = None
        self._full_route_scores: torch.Tensor | None = None
        self._route_position_by_column: torch.Tensor | None = None
        self._route_bank_positions: torch.Tensor | None = None
        self._route_candidates: torch.Tensor | None = None
        comp = trainer.model.competitive
        device = trainer.model.device
        requested_route_bank = int(
            getattr(trainer.config, "route_candidate_bank_size", 0) or 0
        )
        self.route_candidate_bank_size = min(
            int(comp.n_columns),
            max(1, requested_route_bank or int(trainer.config.k_routing)),
        )
        self.route_candidate_bank_enabled = bool(
            self.route_candidate_bank_size < int(comp.n_columns)
        )
        self.route_candidate_bank_ready = False
        self.route_candidate_bank_seed_count = 0
        self.route_candidate_bank_refresh_count = 0
        self.route_candidate_bank_graph_bypass_count = 0
        self.route_candidate_bank_fallback_count = 0
        self.route_candidate_bank_last_reason: str | None = None
        self._last_route_input_rows_scored = 0
        self._last_route_output_candidate_count = 0
        self._last_route_candidate_boundary = "not_run"
        self._last_route_input_source = "not_run"
        self._last_route_scoring_unbounded_reason: str | None = None
        self._last_tick_device_owned_routing_cache = False
        self._route_reconstruction_error = torch.empty(1, device=device)
        pressure_threshold = int(
            round(
                float(trainer.config.candidate_memory_pressure_threshold)
                * 1_000_000.0
            )
        )
        self._route_sleep_filter_control = torch.tensor(
            [0, int(trainer.config.dead_column_steps), 0, pressure_threshold],
            dtype=torch.long,
            device=device,
        )
        self._route_sleep_filter_control_mirror = (
            0,
            int(trainer.config.dead_column_steps),
            0,
            pressure_threshold,
        )
        self._route_memory_pressure_source_mirror = str(
            getattr(
                trainer.model.column_metabolism,
                "last_memory_pressure_source",
                "not_run",
            )
        )
        self._route_sleep_filter_state = torch.zeros(
            12,
            dtype=torch.long,
            device=device,
        )
        self._route_sleep_filter_state_host = [0] * 12
        self._route_sleep_filter_state_dirty = False
        self.route_vote_deep_sleep_filter_control_update_count = 0
        self.route_vote_deep_sleep_filter_state_sync_count = 0
        self._predictive_step_counter = torch.tensor(
            int(trainer.model.predictive.predictive_step_count),
            dtype=torch.long,
            device=device,
        )
        self._state_transition_step_counter = torch.tensor(
            int(comp.state_transition_step_count),
            dtype=torch.long,
            device=device,
        )
        self._state_transition_all_materialized_step = torch.tensor(
            int(comp.state_transition_all_materialized_step),
            dtype=torch.long,
            device=device,
        )
        self._assembly = torch.zeros(comp.n_columns, device=device)
        self._assembly_active_winner = torch.tensor(
            [-1],
            dtype=torch.long,
            device=device,
        )
        self._winner = torch.empty(1, dtype=torch.long, device=device)
        self._strength = torch.ones(1, device=device)
        self._prediction_boost = torch.empty((), device=device)
        self._effective_modulator = torch.empty((), device=device)
        self._zero_consolidation = torch.zeros(comp.n_columns, device=device)
        self._empty_revived_indices = torch.empty(
            0,
            device=device,
            dtype=torch.long,
        )
        self._all_columns = torch.arange(comp.n_columns, device=device)
        self._competition_had_positive = torch.ones(
            (),
            dtype=torch.bool,
            device=device,
        )
        self._previous_winner = torch.tensor(
            [-1 if trainer.last_winner is None else int(trainer.last_winner)],
            dtype=torch.long,
            device=device,
        )
        self._last_winner_consolidation = 0.0
        self._last_effective_modulator = 0.0
        self.winner_consolidation_cpu_metric_count = 0
        self.winner_consolidation_cached_metric_count = 0
        self._recent_spike_row = torch.tensor(
            int(comp.recent_spike_window_cursor),
            dtype=torch.int32,
            device=device,
        )

        if self.requested_mode != "inplace_triton":
            self.fallback_reason = "inplace_triton_not_requested"
            if self.route_vote_requested_mode in {
                "fused_triton_text",
                "cuda_graph_text",
            }:
                self.route_vote_fallback_reason = (
                    "fused_route_vote_requires_inplace_runtime"
                )
            return
        if device.type != "cuda":
            self.fallback_reason = "inplace_triton_requires_cuda"
            if self.route_vote_requested_mode in {
                "fused_triton_text",
                "cuda_graph_text",
            }:
                self.route_vote_fallback_reason = (
                    "fused_route_vote_requires_inplace_runtime"
                )
            return
        if comp.plasticity_mode != "lite":
            self.fallback_reason = "inplace_triton_requires_lite_plasticity"
            if self.route_vote_requested_mode in {
                "fused_triton_text",
                "cuda_graph_text",
            }:
                self.route_vote_fallback_reason = (
                    "fused_route_vote_requires_inplace_runtime"
                )
            return
        if float(comp.input_weight_blend) != 0.0:
            self.fallback_reason = "inplace_triton_requires_zero_input_weight_blend"
            if self.route_vote_requested_mode in {
                "fused_triton_text",
                "cuda_graph_text",
            }:
                self.route_vote_fallback_reason = (
                    "fused_route_vote_requires_inplace_runtime"
                )
            return
        self.fused_vote_competition_active = bool(
            self.fused_vote_competition_requested
            and self.device_selection
            and bool(trainer.config.enable_learned_chunking)
            and int(comp.n_winners) == 1
            and trainer.model.context_layer is None
            and trainer.model.abstraction_layer is None
            and trainer.model.binding_layer is None
        )
        if int(trainer.config.candidate_predictive_update_start_tokens) > int(
            trainer.config.candidate_homeostasis_start_tokens
        ):
            self.candidate_predictive_transition_fallback_reason = (
                "fused_inplace_requires_predictive_gate_no_later_than_graph_candidate_gate"
            )
        else:
            self.candidate_predictive_transition_active = True

        self.warmup_attempted = True
        started = time.perf_counter_ns()
        try:
            routed_candidate_count = min(
                int(comp.n_columns),
                max(1, int(trainer.config.k_routing)),
            )
            candidate_counts = {routed_candidate_count}
            if int(trainer.token_count) < int(
                trainer.config.candidate_homeostasis_start_tokens
            ):
                candidate_counts.add(int(comp.n_columns))
            for candidate_count in sorted(candidate_counts):
                if (
                    self.fused_vote_competition_active
                    and candidate_count
                    == min(int(comp.n_columns), max(1, int(trainer.config.k_routing)))
                ):
                    warmup_fused_vote_competition_cuda(
                        routing_key=torch.empty(comp.column_dim, device=device),
                        prototypes=comp.prototypes,
                        thresholds=comp.thresholds,
                        prediction_location=trainer.model.predictive.location,
                        candidates=torch.empty(
                            candidate_count,
                            dtype=torch.long,
                            device=device,
                        ),
                        previous_winner=self._previous_winner,
                        winner_out=self._winner,
                        strength_out=self._strength,
                        competition_had_positive=self._competition_had_positive,
                    )
                elif self.device_selection:
                    warmup_single_winner_cuda(
                        combined=torch.empty(candidate_count, device=device),
                        inhibition=torch.empty(candidate_count, device=device),
                        candidates=torch.empty(
                            candidate_count,
                            dtype=torch.long,
                            device=device,
                        ),
                        winner_out=self._winner,
                        strength_out=self._strength,
                        competition_had_positive=self._competition_had_positive,
                    )
                warmup_inplace_column_transition_cuda(
                    prototypes=comp.prototypes,
                    prototype_velocity=comp.prototype_velocity,
                    thresholds=comp.thresholds,
                    win_rate_ema=comp.win_rate_ema,
                    steps_since_win=comp.steps_since_win,
                    steps_since_win_last_update_step=(
                        comp.steps_since_win_last_update_step
                    ),
                    state_transition_step_counter=(
                        self._state_transition_step_counter
                    ),
                    state_transition_all_materialized_step=(
                        self._state_transition_all_materialized_step
                    ),
                    location=trainer.model.predictive.location,
                    location_velocity=trainer.model.predictive.velocity,
                    prediction_weights=trainer.model.predictive._prediction_weights,
                    prediction_error=trainer.model.predictive.prediction_error,
                    prediction_failure_streak=(
                        trainer.model.predictive.prediction_failure_streak
                    ),
                    confidence=trainer.model.predictive.confidence,
                    recent_spike_window=comp.recent_spike_window,
                    recent_spike_window_active_ids=(
                        comp.recent_spike_window_active_ids
                    ),
                    assembly=self._assembly,
                    assembly_active_winner=self._assembly_active_winner,
                    prediction_boost_out=self._prediction_boost,
                    effective_modulator_out=self._effective_modulator,
                    routing_key=torch.empty(comp.column_dim, device=device),
                    previous_routing_key=torch.empty(comp.column_dim, device=device),
                    winners=torch.empty(1, dtype=torch.long, device=device),
                    candidates=torch.empty(
                        candidate_count,
                        dtype=torch.long,
                        device=device,
                    ),
                    consolidation=self._zero_consolidation,
                    predictive_candidates=(
                        torch.empty(
                            candidate_count,
                            dtype=torch.long,
                            device=device,
                        )
                        if (
                            self.candidate_predictive_transition_active
                            and candidate_count < int(comp.n_columns)
                        )
                        else None
                    ),
                    predictive_last_update_step=(
                        trainer.model.predictive.predictive_last_update_step
                        if (
                            self.candidate_predictive_transition_active
                            and candidate_count < int(comp.n_columns)
                        )
                        else None
                    ),
                    predictive_step_counter=(
                        self._predictive_step_counter
                        if (
                            self.candidate_predictive_transition_active
                            and candidate_count < int(comp.n_columns)
                        )
                        else None
                    ),
                    competition_had_positive=self._competition_had_positive,
                    recent_spike_row=self._recent_spike_row,
                    prototype_momentum=comp.prototype_momentum,
                    homeostasis_beta=comp.homeostasis_beta,
                    homeostasis_lr=comp.homeostasis_lr,
                    target_firing_rate=comp.target_firing_rate,
                    threshold_min=comp.threshold_min,
                    threshold_max=comp.threshold_max,
                    prediction_error_ema_alpha=(
                        trainer.model.predictive._error_ema_alpha
                    ),
                    prediction_failure_streak_threshold=(
                        trainer.model.predictive._failure_streak_threshold
                    ),
                    prediction_learning_rate=0.005,
                )
                self.precompiled_candidate_counts.append(candidate_count)
            self.warmup_succeeded = True
            self.resolved_mode = "inplace_triton"
        except Exception as exc:
            self.fallback_reason = (
                f"inplace_triton_warmup_failed:{type(exc).__name__}:{exc}"
            )
        finally:
            self.warmup_latency_ms = (
                time.perf_counter_ns() - started
            ) / 1e6
        self._warmup_route_vote()
        if (
            self.route_vote_requested_mode == "cuda_graph_text"
            and self.route_vote_resolved_mode == "cuda_graph_text"
        ):
            self._cuda_graph_runtime = CudaGraphRouteTransition(
                trainer,
                self,
            )
            if not self._cuda_graph_runtime.active:
                self.route_vote_resolved_mode = "fused_triton_text"
                self.route_vote_fallback_reason = (
                    self._cuda_graph_runtime.fallback_reason
                )

    def _rebuild_route_position_map(self, ids: torch.Tensor) -> None:
        comp = self._trainer.model.competitive
        self._route_position_by_column = torch.full(
            (int(comp.n_columns),),
            -1,
            dtype=torch.long,
            device=ids.device,
        )
        if int(ids.numel()) <= 0:
            return
        self._route_position_by_column[ids.long()] = torch.arange(
            int(ids.numel()),
            dtype=torch.long,
            device=ids.device,
        )

    def _record_route_scoring(
        self,
        *,
        input_rows: int,
        output_candidates: int,
        candidate_boundary: str,
        route_input_source: str,
        unbounded_reason: str | None,
    ) -> None:
        self._last_route_input_rows_scored = max(0, int(input_rows))
        self._last_route_output_candidate_count = max(0, int(output_candidates))
        self._last_route_candidate_boundary = str(candidate_boundary)
        self._last_route_input_source = str(route_input_source)
        self._last_route_scoring_unbounded_reason = unbounded_reason

    def _refresh_route_bank_from_candidates(
        self,
        candidates: torch.Tensor,
        *,
        reason: str,
        validate: bool = True,
    ) -> bool:
        if not self.route_candidate_bank_enabled:
            return False
        if self._route_bank_positions is None or self._route_position_by_column is None:
            self.route_candidate_bank_last_reason = "route_bank_workspace_unavailable"
            return False
        ids = candidates.to(
            device=self._route_position_by_column.device,
            dtype=torch.long,
        ).flatten()
        if int(ids.numel()) < int(self.route_candidate_bank_size):
            self.route_candidate_bank_last_reason = "insufficient_candidates_for_route_bank"
            return False
        selected = ids[: int(self.route_candidate_bank_size)]
        positions = self._route_position_by_column.index_select(0, selected)
        if validate and bool((positions < 0).any().item()):
            self.route_candidate_bank_last_reason = "candidate_missing_from_route_cache"
            return False
        self._route_bank_positions.copy_(positions)
        self.route_candidate_bank_ready = True
        self.route_candidate_bank_refresh_count += 1
        self.route_candidate_bank_last_reason = str(reason)
        return True

    def _warmup_route_vote(self) -> None:
        if self.route_vote_requested_mode not in {
            "fused_triton_text",
            "cuda_graph_text",
        }:
            self.route_vote_fallback_reason = "fused_route_vote_not_requested"
            return
        if not self.active:
            self.route_vote_fallback_reason = "fused_route_vote_requires_inplace_runtime"
            return
        if not self.fused_vote_competition_active:
            self.route_vote_fallback_reason = "fused_route_vote_requires_fused_vote_shape"
            return
        index = self._trainer.model.routing_index
        if not hasattr(index, "routing_tensor_cache"):
            self.route_vote_fallback_reason = "routing_tensor_cache_unavailable"
            return

        self.route_vote_warmup_attempted = True
        started = time.perf_counter_ns()
        try:
            vectors, ids = index.routing_tensor_cache()
            if int(ids.numel()) <= 0:
                raise RuntimeError("routing tensor cache is empty")
            candidate_count = min(
                int(self._trainer.config.k_routing),
                int(ids.numel()),
            )
            self._route_vectors = vectors
            self._route_ids = ids
            self._rebuild_route_position_map(ids)
            self._full_route_scores = torch.empty(
                int(ids.numel()),
                device=vectors.device,
            )
            if self.route_candidate_bank_enabled:
                self._route_bank_positions = torch.arange(
                    self.route_candidate_bank_size,
                    dtype=torch.long,
                    device=vectors.device,
                )
                self._route_scores = torch.empty(
                    self.route_candidate_bank_size,
                    device=vectors.device,
                )
            else:
                self._route_bank_positions = None
                self._route_scores = self._full_route_scores
            self._route_candidates = torch.empty(
                candidate_count,
                dtype=torch.long,
                device=vectors.device,
            )
            if self.route_candidate_bank_enabled:
                warmup_fused_route_vote_cuda(
                    routing_key=torch.empty(
                        int(vectors.shape[1]),
                        device=vectors.device,
                    ),
                    routing_vectors=vectors,
                    routing_ids=ids,
                    route_positions=None,
                    steps_since_win=self._trainer.model.competitive.steps_since_win,
                    steps_since_win_last_update_step=(
                        self._trainer.model.competitive.steps_since_win_last_update_step
                    ),
                    state_transition_step_counter=self._state_transition_step_counter,
                    state_transition_all_materialized_step=(
                        self._state_transition_all_materialized_step
                    ),
                    prototypes=self._trainer.model.competitive.prototypes,
                    thresholds=self._trainer.model.competitive.thresholds,
                    prediction_location=self._trainer.model.predictive.location,
                    memory_pressure=self._trainer.model.column_metabolism.memory_pressure,
                    previous_winner=self._previous_winner,
                    route_filter_control=self._route_sleep_filter_control,
                    route_filter_state_out=self._route_sleep_filter_state,
                    scores_out=self._full_route_scores,
                    candidates_out=self._route_candidates,
                    winner_out=self._winner,
                    strength_out=self._strength,
                    competition_had_positive=self._competition_had_positive,
                    reconstruction_error_out=self._route_reconstruction_error,
                )
            warmup_fused_route_vote_cuda(
                routing_key=torch.empty(
                    int(vectors.shape[1]),
                    device=vectors.device,
                ),
                routing_vectors=vectors,
                routing_ids=ids,
                route_positions=(
                    self._route_bank_positions
                    if self.route_candidate_bank_enabled
                    else None
                ),
                steps_since_win=self._trainer.model.competitive.steps_since_win,
                steps_since_win_last_update_step=(
                    self._trainer.model.competitive.steps_since_win_last_update_step
                ),
                state_transition_step_counter=self._state_transition_step_counter,
                state_transition_all_materialized_step=(
                    self._state_transition_all_materialized_step
                ),
                prototypes=self._trainer.model.competitive.prototypes,
                thresholds=self._trainer.model.competitive.thresholds,
                prediction_location=self._trainer.model.predictive.location,
                memory_pressure=self._trainer.model.column_metabolism.memory_pressure,
                previous_winner=self._previous_winner,
                route_filter_control=self._route_sleep_filter_control,
                route_filter_state_out=self._route_sleep_filter_state,
                scores_out=self._route_scores,
                candidates_out=self._route_candidates,
                winner_out=self._winner,
                strength_out=self._strength,
                competition_had_positive=self._competition_had_positive,
                reconstruction_error_out=self._route_reconstruction_error,
            )
            self.route_vote_kernel_variant = fused_route_vote_kernel_variant(
                vectors,
                self._route_candidates,
                self._route_bank_positions
                if self.route_candidate_bank_enabled
                else None,
            )
            self.route_vote_warmup_succeeded = True
            self.route_vote_resolved_mode = self.route_vote_requested_mode
        except Exception as exc:
            self.route_vote_fallback_reason = (
                f"fused_route_vote_warmup_failed:{type(exc).__name__}:{exc}"
            )
        finally:
            self.route_vote_warmup_latency_ms = (
                time.perf_counter_ns() - started
            ) / 1e6

    @property
    def active(self) -> bool:
        return self.resolved_mode == "inplace_triton"

    @property
    def handles_predictive_vote(self) -> bool:
        return bool(self.active and self.fused_vote_competition_active)

    @property
    def handles_route_vote(self) -> bool:
        return bool(
            self.active
            and self.route_vote_resolved_mode
            in {"fused_triton_text", "cuda_graph_text"}
        )

    def _sync_state_transition_step_tensors_from_core(self) -> None:
        comp = self._trainer.model.competitive
        self._state_transition_step_counter.fill_(
            int(comp.state_transition_step_count)
        )
        self._state_transition_all_materialized_step.fill_(
            int(comp.state_transition_all_materialized_step)
        )

    def route_sleep_filter_due(self) -> bool:
        return bool(
            self._trainer.token_count
            >= int(self._trainer.config.candidate_deep_sleep_filter_start_tokens)
            and self._trainer.token_count
            >= int(self._trainer.config.dead_column_steps)
        )

    def route_memory_pressure_filter_due(self) -> bool:
        source = str(
            getattr(
                self._trainer.model.column_metabolism,
                "last_memory_pressure_source",
                "not_run",
            )
        )
        has_pressure_evidence = source not in {
            "not_run",
            "no_awake_candidates",
            "no_memory_store_bucket_evidence",
        }
        return bool(
            self._trainer.token_count
            >= int(self._trainer.config.candidate_memory_pressure_filter_start_tokens)
            and has_pressure_evidence
        )

    def prepare_route_sleep_filter_control(self) -> None:
        enabled = 1 if self.route_sleep_filter_due() else 0
        threshold = int(self._trainer.config.dead_column_steps)
        pressure_enabled = 1 if self.route_memory_pressure_filter_due() else 0
        self._route_memory_pressure_source_mirror = str(
            getattr(
                self._trainer.model.column_metabolism,
                "last_memory_pressure_source",
                "not_run",
            )
        )
        pressure_threshold = int(
            round(
                float(self._trainer.config.candidate_memory_pressure_threshold)
                * 1_000_000.0
            )
        )
        desired = (enabled, threshold, pressure_enabled, pressure_threshold)
        if desired == self._route_sleep_filter_control_mirror:
            return
        self._route_sleep_filter_control[0].fill_(enabled)
        self._route_sleep_filter_control[1].fill_(threshold)
        self._route_sleep_filter_control[2].fill_(pressure_enabled)
        self._route_sleep_filter_control[3].fill_(pressure_threshold)
        self._route_sleep_filter_control_mirror = desired
        self.route_vote_deep_sleep_filter_control_update_count += 1

    def mark_route_sleep_filter_state_dirty(self) -> None:
        self._route_sleep_filter_state_dirty = True

    def sync_route_sleep_filter_state_from_device(self) -> dict[str, Any]:
        self._route_sleep_filter_state_host = [
            int(value)
            for value in self._route_sleep_filter_state.detach()
            .to(device="cpu", dtype=torch.long)
            .tolist()
        ]
        self._route_sleep_filter_state_dirty = False
        self.route_vote_deep_sleep_filter_state_sync_count += 1
        return self.route_sleep_filter_snapshot()

    def _route_score_count_snapshot(
        self,
        state: list[int],
    ) -> tuple[int, int, int]:
        route_ids = getattr(self, "_route_ids", None)
        route_candidates = getattr(self, "_route_candidates", None)
        route_input_count = (
            int(self._last_route_input_rows_scored)
            if int(getattr(self, "_last_route_input_rows_scored", 0)) > 0
            else
            int(route_ids.numel())
            if isinstance(route_ids, torch.Tensor)
            else int(state[5])
            if len(state) > 5
            else 0
        )
        route_output_count = (
            int(self._last_route_output_candidate_count)
            if int(getattr(self, "_last_route_output_candidate_count", 0)) > 0
            else
            int(route_candidates.numel())
            if isinstance(route_candidates, torch.Tensor)
            else int(state[6])
            if len(state) > 6
            else 0
        )
        comp = getattr(
            getattr(getattr(self, "_trainer", None), "model", None),
            "competitive",
            None,
        )
        total_columns = int(getattr(comp, "n_columns", 0) or 0)
        if total_columns <= 0:
            total_columns = route_input_count
        return route_input_count, route_output_count, total_columns

    def route_scoring_snapshot(self) -> dict[str, Any]:
        state = list(getattr(self, "_route_sleep_filter_state_host", []))
        route_input_count, route_output_count, total_columns = (
            self._route_score_count_snapshot(state)
        )
        route_vote_active = bool(
            getattr(self, "resolved_mode", None) == "inplace_triton"
            and str(getattr(self, "route_vote_resolved_mode", ""))
            in {"fused_triton_text", "cuda_graph_text"}
        )
        route_rows_run_all_columns = bool(
            route_vote_active
            and total_columns > 0
            and route_input_count >= total_columns
        )
        route_output_fraction = (
            float(route_output_count) / float(max(1, total_columns))
        )
        route_input_fraction = (
            float(route_input_count) / float(max(1, total_columns))
        )
        bounded_route_scoring = bool(
            route_vote_active
            and route_output_count > 0
            and not route_rows_run_all_columns
            and route_input_count > 0
        )
        route_scoring_unbounded_reason = None
        if self._last_route_scoring_unbounded_reason is not None:
            route_scoring_unbounded_reason = self._last_route_scoring_unbounded_reason
        elif route_rows_run_all_columns:
            route_scoring_unbounded_reason = (
                "exact_full_cache_route_scoring_before_bounded_candidate_selection"
            )
        elif not route_vote_active:
            route_scoring_unbounded_reason = getattr(
                self,
                "route_vote_fallback_reason",
                None,
            )
        return {
            "surface": "route_vote_scoring_scope.v1",
            "mode": str(getattr(self, "route_vote_resolved_mode", "tensor")),
            "kernel_variant": str(
                getattr(self, "route_vote_kernel_variant", "unavailable")
            ),
            "total_columns": int(total_columns),
            "route_input_rows_scored": int(route_input_count),
            "route_output_candidate_count": int(route_output_count),
            "route_input_fraction": route_input_fraction,
            "route_output_fraction": route_output_fraction,
            "route_rows_run_all_columns": route_rows_run_all_columns,
            "bounded_route_scoring": bounded_route_scoring,
            "candidate_boundary": (
                self._last_route_candidate_boundary
                if self._last_route_candidate_boundary != "not_run"
                else "exact_full_cache_score_then_filter_select"
                if route_vote_active
                else "retained_tensor_or_inactive_route_vote"
            ),
            "route_input_source": (
                self._last_route_input_source
                if self._last_route_input_source != "not_run"
                else "complete_routing_tensor_cache"
                if route_vote_active
                else "not_owned_by_fused_route_vote"
            ),
            "route_scoring_unbounded_reason": route_scoring_unbounded_reason,
            "claim_boundary": (
                "route_cost_truth_separate_from_bounded_awake_specialist_execution"
            ),
        }

    def route_sleep_filter_snapshot(self) -> dict[str, Any]:
        state = list(self._route_sleep_filter_state_host)
        enabled = bool(self._route_sleep_filter_control_mirror[0])
        pressure_enabled = bool(self._route_sleep_filter_control_mirror[2])
        pressure_threshold = (
            float(self._route_sleep_filter_control_mirror[3]) / 1_000_000.0
        )
        state_enabled = bool(state[0])
        pressure_state_enabled = bool(state[8]) if len(state) > 8 else False
        pressure_applied = bool(state[9]) if len(state) > 9 else False
        pressure_over_threshold_count = (
            int(state[10]) if pressure_state_enabled and len(state) > 10 else 0
        )
        fallback_code = int(state[4]) if state_enabled or pressure_state_enabled else 0
        fallback_reason = {
            1: "insufficient_awake_route_scores_after_deep_sleep_filter",
            2: "all_route_scores_deep_sleep",
            3: "insufficient_awake_route_scores_after_memory_pressure_filter",
            4: "all_route_scores_over_memory_pressure_threshold",
        }.get(fallback_code)
        route_input_count, route_output_count, total_columns = (
            self._route_score_count_snapshot(state)
        )
        route_vote_active = bool(
            getattr(self, "resolved_mode", None) == "inplace_triton"
            and str(getattr(self, "route_vote_resolved_mode", ""))
            in {"fused_triton_text", "cuda_graph_text"}
        )
        route_rows_run_all_columns = bool(
            route_vote_active
            and total_columns > 0
            and route_input_count >= total_columns
        )
        state_current_for_control = bool(
            state_enabled == enabled
            and pressure_state_enabled == pressure_enabled
            and int(state[5]) == int(route_input_count)
            and int(state[6]) == int(route_output_count)
        )
        return {
            "surface": "route_vote_scheduler_filter.v1",
            "enabled": enabled,
            "state_enabled": state_enabled,
            "memory_pressure_enabled": pressure_enabled,
            "memory_pressure_state_enabled": pressure_state_enabled,
            "memory_pressure_applied": pressure_applied,
            "state_current_for_control": state_current_for_control,
            "input_candidate_count": route_input_count,
            "output_candidate_count": route_output_count,
            "route_input_rows_scored": route_input_count,
            "route_output_candidate_count": route_output_count,
            "route_rows_run_all_columns": route_rows_run_all_columns,
            "bounded_route_scoring": bool(
                route_vote_active
                and route_output_count > 0
                and not route_rows_run_all_columns
                and route_input_count > 0
            ),
            "filtered_deep_sleep_count": int(state[2]) if state_enabled else 0,
            "filtered_memory_pressure_count": (
                pressure_over_threshold_count if pressure_applied else 0
            ),
            "memory_pressure_over_threshold_count": pressure_over_threshold_count,
            "eligible_route_count": (
                int(state[3])
                if state_enabled
                else int(route_input_count)
            ),
            "memory_pressure_eligible_route_count": (
                int(state[11])
                if pressure_state_enabled and len(state) > 11
                else int(route_input_count)
            ),
            "memory_pressure_threshold": (
                pressure_threshold if pressure_enabled else None
            ),
            "memory_pressure_source": str(
                self._route_memory_pressure_source_mirror
            ),
            "sleep_backfill_count": int(state[7]) if state_enabled else 0,
            "fallback_reason": fallback_reason,
            "control_update_count": int(
                self.route_vote_deep_sleep_filter_control_update_count
            ),
            "state_sync_count": int(
                self.route_vote_deep_sleep_filter_state_sync_count
            ),
            "state_dirty": bool(self._route_sleep_filter_state_dirty),
            "tensor_device": str(self._route_sleep_filter_state.device),
            "claim_boundary": (
                "training_owned_route_vote_masks_sleep_and_memory_pressure_inside_existing_route_selection"
            ),
            "route_cost_claim_boundary": (
                "sleep_and_pressure_are_masked_before_selection_but_route_score_rows_remain_explicit"
            ),
        }

    def route_candidates(
        self,
        routing_key: torch.Tensor,
        *,
        sensory_tick: bool,
    ) -> torch.Tensor | None:
        """Route and select on text/idle ticks, or request retained fallback."""

        self._route_vote_ready = False
        self._route_transition_graph_ready = False
        if not self.handles_route_vote:
            return None
        if sensory_tick:
            self.route_vote_sensory_fallback_count += 1
            return None
        self._sync_state_transition_step_tensors_from_core()
        self.prepare_route_sleep_filter_control()
        if (
            self._prepared_graph_token == self._trainer.token_count
            and self._cuda_graph_runtime is not None
            and self._cuda_graph_runtime.active
            and self._route_candidates is not None
        ):
            comp = self._trainer.model.competitive
            comp.last_candidate_count = int(self._route_candidates.numel())
            comp.last_scored_column_count = int(self._route_candidates.numel())
            comp.last_execution_mode = (
                "candidate_subset_cuda_graph_route_transition"
            )
            self.route_vote_prepared_graph_reuse_count += 1
            self._route_vote_ready = True
            self._route_transition_graph_ready = True
            self._record_route_scoring(
                input_rows=(
                    int(self._route_bank_positions.numel())
                    if self.route_candidate_bank_enabled
                    and self._route_bank_positions is not None
                    else int(self._route_ids.numel())
                    if self._route_ids is not None
                    else int(self._route_candidates.numel())
                ),
                output_candidates=int(self._route_candidates.numel()),
                candidate_boundary=(
                    "bounded_route_bank_score_then_filter_select"
                    if self.route_candidate_bank_enabled
                    else "exact_full_cache_score_then_filter_select"
                ),
                route_input_source=(
                    "training_owned_route_candidate_bank"
                    if self.route_candidate_bank_enabled
                    else "complete_routing_tensor_cache"
                ),
                unbounded_reason=None,
            )
            if self.route_candidate_bank_enabled:
                self._refresh_route_bank_from_candidates(
                    self._route_candidates,
                    reason="bounded_route_bank_graph_refresh",
                    validate=False,
                )
            return self._route_candidates
        index = self._trainer.model.routing_index
        assert self._route_vectors is not None
        assert self._route_ids is not None
        assert self._route_scores is not None
        assert self._full_route_scores is not None
        assert self._route_candidates is not None
        cache_dirty = True
        cache_dirty_fn = getattr(index, "routing_tensor_cache_is_dirty", None)
        if callable(cache_dirty_fn):
            cache_dirty = bool(cache_dirty_fn())
        if cache_dirty:
            vectors, ids = index.routing_tensor_cache()
        else:
            vectors = self._route_vectors
            ids = self._route_ids
            self.route_vote_clean_cache_reuse_count += 1
        if (
            tuple(vectors.shape) != tuple(self._route_vectors.shape)
            or tuple(ids.shape) != tuple(self._route_ids.shape)
        ):
            self.route_vote_fallback_reason = "routing_tensor_cache_shape_changed"
            self.route_vote_resolved_mode = "tensor"
            return None
        if (
            vectors.data_ptr() != self._route_vectors.data_ptr()
            or ids.data_ptr() != self._route_ids.data_ptr()
        ):
            self._route_vectors = vectors
            self._route_ids = ids
            self._rebuild_route_position_map(ids)
            self.route_candidate_bank_ready = False
            self.route_vote_cache_refresh_count += 1
        route_vectors = self._route_vectors
        route_ids = self._route_ids
        route_candidates = self._route_candidates
        assert route_vectors is not None
        assert route_ids is not None
        assert route_candidates is not None
        if (
            (
                not self.route_candidate_bank_enabled
                or self.route_candidate_bank_ready
            )
            and
            self._cuda_graph_runtime is not None
            and self._cuda_graph_runtime.active
            and self._cuda_graph_runtime.eligible(
                assume_route_cache_current=True
            )
        ):
            comp = self._trainer.model.competitive
            comp.last_candidate_count = int(route_candidates.numel())
            comp.last_scored_column_count = int(route_candidates.numel())
            comp.last_execution_mode = (
                "candidate_subset_cuda_graph_route_transition"
            )
            self._route_vote_ready = True
            self._route_transition_graph_ready = True
            self._record_route_scoring(
                input_rows=(
                    int(self._route_bank_positions.numel())
                    if self.route_candidate_bank_enabled
                    and self._route_bank_positions is not None
                    else int(route_ids.numel())
                ),
                output_candidates=int(route_candidates.numel()),
                candidate_boundary=(
                    "bounded_route_bank_score_then_filter_select"
                    if self.route_candidate_bank_enabled
                    else "exact_full_cache_score_then_filter_select"
                ),
                route_input_source=(
                    "training_owned_route_candidate_bank"
                    if self.route_candidate_bank_enabled
                    else "complete_routing_tensor_cache"
                ),
                unbounded_reason=None,
            )
            return route_candidates
        route_positions = (
            self._route_bank_positions
            if self.route_candidate_bank_enabled
            and self.route_candidate_bank_ready
            else None
        )
        route_scores = (
            self._route_scores
            if route_positions is not None
            else self._full_route_scores
        )
        route_input_rows = (
            int(route_positions.numel())
            if route_positions is not None
            else int(route_ids.numel())
        )
        if self.route_candidate_bank_enabled and route_positions is None:
            self.route_candidate_bank_seed_count += 1
            self.route_candidate_bank_fallback_count += 1
            self.route_candidate_bank_last_reason = (
                "route_candidate_bank_not_ready_exact_seed"
            )
        fused_route_vote_cuda(
            routing_key=routing_key,
            routing_vectors=route_vectors,
            routing_ids=route_ids,
            route_positions=route_positions,
            prototypes=self._trainer.model.competitive.prototypes,
            thresholds=self._trainer.model.competitive.thresholds,
            prediction_location=self._trainer.model.predictive.location,
            memory_pressure=self._trainer.model.column_metabolism.memory_pressure,
            previous_winner=self._previous_winner,
            steps_since_win=self._trainer.model.competitive.steps_since_win,
            steps_since_win_last_update_step=(
                self._trainer.model.competitive.steps_since_win_last_update_step
            ),
            state_transition_step_counter=self._state_transition_step_counter,
            state_transition_all_materialized_step=(
                self._state_transition_all_materialized_step
            ),
            route_filter_control=self._route_sleep_filter_control,
            route_filter_state_out=self._route_sleep_filter_state,
            scores_out=route_scores,
            candidates_out=route_candidates,
            winner_out=self._winner,
            strength_out=self._strength,
            competition_had_positive=self._competition_had_positive,
            reconstruction_error_out=self._route_reconstruction_error,
        )
        comp = self._trainer.model.competitive
        comp.last_candidate_count = int(route_candidates.numel())
        comp.last_scored_column_count = int(route_candidates.numel())
        comp.last_execution_mode = "candidate_subset_fused_route_vote"
        self.route_vote_execution_count += 1
        self.mark_route_sleep_filter_state_dirty()
        self._route_vote_ready = True
        self._record_route_scoring(
            input_rows=route_input_rows,
            output_candidates=int(route_candidates.numel()),
            candidate_boundary=(
                "bounded_route_bank_score_then_filter_select"
                if route_positions is not None
                else "exact_full_cache_score_seed_route_bank"
                if self.route_candidate_bank_enabled
                else "exact_full_cache_score_then_filter_select"
            ),
            route_input_source=(
                "training_owned_route_candidate_bank"
                if route_positions is not None
                else "complete_routing_tensor_cache_seed"
                if self.route_candidate_bank_enabled
                else "complete_routing_tensor_cache"
            ),
            unbounded_reason=(
                None
                if route_positions is not None
                else "route_candidate_bank_not_ready_exact_seed"
                if self.route_candidate_bank_enabled
                else "exact_full_cache_route_scoring_before_bounded_candidate_selection"
            ),
        )
        if self.route_candidate_bank_enabled:
            self._refresh_route_bank_from_candidates(
                route_candidates,
                reason=(
                    "route_candidate_bank_seeded_from_exact_route"
                    if route_positions is None
                    else "bounded_route_bank_refreshed_from_candidates"
                ),
                validate=False,
            )
        return route_candidates

    def prepare_routing(
        self,
        pattern: torch.Tensor,
        *,
        sensory_tick: bool,
    ) -> tuple[torch.Tensor, float] | None:
        if (
            self._cuda_graph_runtime is None
            or not self._cuda_graph_runtime.active
        ):
            self._prepared_graph_token = None
            return None
        if (
            self.route_candidate_bank_enabled
            and not self.route_candidate_bank_ready
            and not sensory_tick
            and self._trainer.token_count >= self._trainer.config.bootstrap_tokens
        ):
            self._prepared_graph_token = None
            self.route_candidate_bank_graph_bypass_count += 1
            return None
        prepared = self._cuda_graph_runtime.prepare_routing(
            pattern,
            sensory_tick=sensory_tick,
            assume_eligible=bool(self._route_transition_graph_ready),
        )
        self._prepared_graph_token = (
            int(self._trainer.token_count) if prepared is not None else None
        )
        return prepared

    def stage_text_input_quantum(
        self,
        patterns: list[torch.Tensor],
    ) -> bool:
        if self._cuda_graph_runtime is None:
            return False
        return self._cuda_graph_runtime.stage_input_quantum(patterns)

    def can_prestage_text_input_quantum(self) -> bool:
        if self._cuda_graph_runtime is None:
            return False
        return self._cuda_graph_runtime.can_prestage_input_quantum()

    def replay_text_burst(
        self,
        patterns: list[torch.Tensor],
    ) -> dict[str, Any]:
        if self._cuda_graph_runtime is None:
            raise RuntimeError("cuda_graph_runtime_unavailable")
        return self._cuda_graph_runtime.replay_staged_text_burst(
            patterns
        )

    def text_burst_token_capacity(self) -> int:
        if self._cuda_graph_runtime is None:
            return 0
        return int(self._cuda_graph_runtime.text_burst_token_capacity())

    def drain_text_burst_events(self) -> dict[str, Any]:
        if self._cuda_graph_runtime is None:
            return {
                "truth_synced": False,
                "result_rows": [],
                "strong_indices": [],
                "strong_assemblies": [],
                "strong_routing_keys": [],
            }
        return self._cuda_graph_runtime.drain_burst_events()

    def _retained_consensus_gain(
        self,
        context_gain: torch.Tensor | None,
        routing_key: torch.Tensor,
    ) -> torch.Tensor | None:
        trainer = self._trainer
        if trainer.last_winner is None:
            return context_gain
        consensus_gain = trainer.model.predictive.vote(
            [trainer.last_winner],
            routing_key,
        )
        if context_gain is None:
            return consensus_gain
        return torch.clamp(
            context_gain * consensus_gain,
            min=0.5,
            max=1.5,
        )

    def select_winner(
        self,
        *,
        routing_key: torch.Tensor,
        candidates: torch.Tensor | None,
        context_gain: torch.Tensor | None,
        fallback_allowed: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run candidate competition without a host-visible positivity branch."""

        if not self.active:
            raise RuntimeError("in-place Triton column transition is not active")
        if self._route_vote_ready:
            if candidates is not self._route_candidates:
                raise RuntimeError("fused route/vote candidate workspace changed")
            if context_gain is not None:
                raise RuntimeError("fused route/vote does not support context gain")
            self._route_vote_ready = False
            self.selection_execution_count += 1
            self.fused_vote_competition_execution_count += 1
            self.last_selection_mode = "fused_route_vote_cuda"
            assert self._route_candidates is not None
            return self._winner, self._strength, self._route_candidates
        comp = self._trainer.model.competitive
        if not self.device_selection:
            self._competition_had_positive.fill_(True)
            self.selection_execution_count += 1
            self.last_selection_mode = "retained_host_branch"
            return comp.compete(
                routing_key,
                candidates,
                fallback_allowed=fallback_allowed,
                context_gain=context_gain,
            )
        x = comp._cached_normalize_key(routing_key)
        if candidates is not None and int(candidates.numel()) > 0:
            selected = candidates.to(comp.device).long()
        elif fallback_allowed:
            selected = self._all_columns
        else:
            raise RuntimeError(
                "No candidates available and fallback disabled; initialize routing index first."
            )

        candidate_count = int(selected.numel())
        expected_candidate_count = min(
            int(comp.n_columns),
            max(1, int(self._trainer.config.k_routing)),
        )
        if (
            self.fused_vote_competition_active
            and candidate_count == expected_candidate_count
            and comp._cached_proto_sim is None
            and context_gain is None
        ):
            comp.materialize_homeostasis(selected)
            try:
                select_fused_vote_competition_cuda(
                    routing_key=x,
                    prototypes=comp.prototypes,
                    thresholds=comp.thresholds,
                    prediction_location=self._trainer.model.predictive.location,
                    candidates=selected,
                    previous_winner=self._previous_winner,
                    winner_out=self._winner,
                    strength_out=self._strength,
                    competition_had_positive=self._competition_had_positive,
                )
            except Exception:
                self.selection_failure_count += 1
                self.last_selection_mode = "fused_vote_competition_failed_closed"
                raise
            comp.last_candidate_count = candidate_count
            comp.last_scored_column_count = candidate_count
            comp.last_execution_mode = "candidate_subset_fused_vote_competition"
            self.selection_execution_count += 1
            self.fused_vote_competition_execution_count += 1
            self.last_selection_mode = "fused_vote_competition_triton"
            return self._winner, self._strength, selected

        if self.fused_vote_competition_active:
            self.fused_vote_competition_fallback_count += 1
            context_gain = self._retained_consensus_gain(
                context_gain,
                routing_key,
            )
        comp.last_candidate_count = candidate_count
        comp.last_scored_column_count = (
            comp.n_columns if comp._cached_proto_sim is not None else candidate_count
        )
        comp.last_execution_mode = (
            "candidate_subset_device_select"
            if comp._cached_proto_sim is None and candidate_count < comp.n_columns
            else "dense_cached_device_select"
            if comp._cached_proto_sim is not None
            else "all_columns_device_select"
        )

        if comp._cached_proto_sim is not None:
            similarity = comp._cached_proto_sim[selected]
        else:
            similarity = torch.mv(comp.prototypes[selected], x)
        combined = comp._combine_similarity_and_input_drive(
            similarity,
            candidates=selected,
        )
        if context_gain is not None:
            gain = context_gain.to(comp.device)
            if gain.dim() != 1 or int(gain.numel()) != comp.n_columns:
                raise ValueError(
                    "context_gain must be a 1D tensor with n_columns entries"
                )
            combined = combined * torch.clamp(
                gain[selected],
                min=0.5,
                max=1.5,
            )
        comp.materialize_homeostasis(selected)
        inhibition = comp._inhibition(selected)
        try:
            select_single_winner_cuda(
                combined=combined,
                inhibition=inhibition,
                candidates=selected,
                winner_out=self._winner,
                strength_out=self._strength,
                competition_had_positive=self._competition_had_positive,
            )
        except Exception:
            self.selection_failure_count += 1
            self.last_selection_mode = "inplace_triton_failed_closed"
            raise
        self.selection_execution_count += 1
        self.last_selection_mode = "inplace_triton"
        if self.fused_vote_competition_active:
            self._previous_winner.copy_(self._winner)
        return self._winner, self._strength, selected

    def apply(
        self,
        *,
        routing_key: torch.Tensor,
        candidates: torch.Tensor | None,
        winners: torch.Tensor,
        modulator: float,
        compute_metrics: bool = False,
    ) -> tuple[
        torch.Tensor,
        list[int] | None,
        int | None,
        float,
        float,
        float,
        float,
    ]:
        if not self.active:
            raise RuntimeError("in-place Triton column transition is not active")
        self._last_tick_device_owned_routing_cache = False
        trainer = self._trainer
        comp = trainer.model.competitive
        if candidates is None:
            candidates = self._all_columns
        predictive_scope_ready = trainer.token_count >= int(
            trainer.config.candidate_predictive_update_start_tokens
        )
        use_candidate_predictive_transition = bool(
            self.candidate_predictive_transition_active
            and predictive_scope_ready
            and int(candidates.numel()) < comp.n_columns
        )
        homeostasis_scope_ready = trainer.token_count >= int(
            trainer.config.candidate_homeostasis_start_tokens
        )
        homeostasis_candidates = (
            candidates if homeostasis_scope_ready else self._all_columns
        )
        used_cuda_graph = bool(
            self._route_transition_graph_ready
            and self._cuda_graph_runtime is not None
        )
        if used_cuda_graph:
            consolidation = None
            previous = None
            self.graph_consolidation_lookup_skip_count += 1
        else:
            consolidation = (
                trainer.model.memory_store.bucket_consolidation_tensor(
                    comp.n_columns,
                    device=trainer.model.device,
                )
                if trainer.memory_warm_started
                else self._zero_consolidation
            )
            previous = (
                routing_key
                if trainer._prev_routing_key is None
                else trainer._prev_routing_key
            )
        profile_enabled = bool(
            getattr(trainer, "_train_step_profile_enabled", False)
        )
        profile_totals = (
            getattr(trainer, "_train_step_profile_totals_ms", {})
            if profile_enabled
            else {}
        )
        profile_last = time.perf_counter() if profile_enabled else 0.0
        if use_candidate_predictive_transition and not used_cuda_graph:
            self._predictive_step_counter.fill_(
                int(trainer.model.predictive.predictive_step_count)
            )

        def _profile_mark(name: str) -> None:
            nonlocal profile_last
            if not profile_enabled:
                return
            now = time.perf_counter()
            profile_totals[name] = profile_totals.get(name, 0.0) + (
                now - profile_last
            ) * 1000.0
            profile_last = now

        try:
            if used_cuda_graph:
                assert self._cuda_graph_runtime is not None
                graph_result = self._cuda_graph_runtime.consume_result()
                self.route_vote_execution_count += 1
                self._route_transition_graph_ready = False
            else:
                assert consolidation is not None
                assert previous is not None
                inplace_column_transition_cuda(
                    prototypes=comp.prototypes,
                    routing_vectors=self._route_vectors,
                    routing_position_by_column=self._route_position_by_column,
                    prototype_velocity=comp.prototype_velocity,
                    thresholds=comp.thresholds,
                    win_rate_ema=comp.win_rate_ema,
                    steps_since_win=comp.steps_since_win,
                    steps_since_win_last_update_step=(
                        comp.steps_since_win_last_update_step
                    ),
                    state_transition_step_counter=(
                        self._state_transition_step_counter
                    ),
                    state_transition_all_materialized_step=(
                        self._state_transition_all_materialized_step
                    ),
                    location=trainer.model.predictive.location,
                    location_velocity=trainer.model.predictive.velocity,
                    prediction_weights=trainer.model.predictive._prediction_weights,
                    prediction_error=trainer.model.predictive.prediction_error,
                    prediction_failure_streak=(
                        trainer.model.predictive.prediction_failure_streak
                    ),
                    confidence=trainer.model.predictive.confidence,
                    recent_spike_window=comp.recent_spike_window,
                    recent_spike_window_active_ids=(
                        comp.recent_spike_window_active_ids
                    ),
                    assembly=self._assembly,
                    assembly_active_winner=self._assembly_active_winner,
                    prediction_boost_out=self._prediction_boost,
                    effective_modulator_out=self._effective_modulator,
                    routing_key=routing_key,
                    previous_routing_key=previous,
                    winners=winners,
                    candidates=homeostasis_candidates,
                    consolidation=consolidation,
                    predictive_candidates=(
                        candidates if use_candidate_predictive_transition else None
                    ),
                    predictive_last_update_step=(
                        trainer.model.predictive.predictive_last_update_step
                        if use_candidate_predictive_transition
                        else None
                    ),
                    predictive_step_counter=(
                        self._predictive_step_counter
                        if use_candidate_predictive_transition
                        else None
                    ),
                    base_modulator=float(modulator),
                    dopamine=float(trainer.model.surprise.dopamine),
                    serotonin=float(trainer.model.surprise.serotonin),
                    competitive_learning_rate=float(comp.get_lr()),
                    recent_spike_row=self._recent_spike_row,
                    has_previous_routing_key=trainer._prev_routing_key is not None,
                    competition_had_positive=self._competition_had_positive,
                    prototype_momentum=comp.prototype_momentum,
                    homeostasis_beta=comp.homeostasis_beta,
                    homeostasis_lr=comp.homeostasis_lr,
                    target_firing_rate=comp.target_firing_rate,
                    threshold_min=comp.threshold_min,
                    threshold_max=comp.threshold_max,
                    prediction_error_ema_alpha=(
                        trainer.model.predictive._error_ema_alpha
                    ),
                    prediction_failure_streak_threshold=(
                        trainer.model.predictive._failure_streak_threshold
                    ),
                    prediction_learning_rate=0.005,
                )
            _profile_mark("column_transition_kernel_or_graph")
        except Exception:
            self.failure_count += 1
            self.last_execution_mode = "inplace_triton_failed_closed"
            raise

        if used_cuda_graph:
            if (
                self._cuda_graph_runtime is not None
                and self._cuda_graph_runtime.last_result_from_host_sync
                and len(graph_result) > 6
            ):
                winner_id = int(graph_result[6])
                winner_id_list = [winner_id]
                self.graph_host_winner_reuse_count += 1
            else:
                winner_id_list = None
                winner_id = None
        else:
            winner_id_list = winners.tolist()
            winner_id = int(winner_id_list[0])
            if self._route_sleep_filter_state_dirty:
                self.sync_route_sleep_filter_state_from_device()
        _profile_mark("column_transition_winner_readback")
        if (
            used_cuda_graph
            and trainer.memory_warm_started
            and compute_metrics
            and winner_id_list is not None
        ):
            levels = [
                float(trainer.model.memory_store.bucket_consolidation_level(wid))
                for wid in winner_id_list
            ]
            winner_consolidation = (
                float(sum(levels) / len(levels)) if levels else 0.0
            )
            self._last_winner_consolidation = winner_consolidation
            self.winner_consolidation_cpu_metric_count += 1
        elif used_cuda_graph and trainer.memory_warm_started:
            winner_consolidation = float(self._last_winner_consolidation)
            self.winner_consolidation_cached_metric_count += int(compute_metrics)
        elif trainer.memory_warm_started and compute_metrics:
            assert consolidation is not None
            winner_consolidation = float(
                consolidation.index_select(0, winners).mean().item()
            )
            self._last_winner_consolidation = winner_consolidation
        elif trainer.memory_warm_started:
            winner_consolidation = float(self._last_winner_consolidation)
        else:
            winner_consolidation = 0.0
            self._last_winner_consolidation = 0.0
        _profile_mark("column_transition_consolidation_readback")
        dopamine_ltp_gain = 0.8 + 0.4 * trainer.model.surprise.dopamine
        serotonin_patience = max(
            0.2,
            1.0 - 0.6 * trainer.model.surprise.serotonin,
        )

        if not used_cuda_graph:
            trainer._prev_routing_key = routing_key.detach().clone()
        self._last_tick_device_owned_routing_cache = bool(
            used_cuda_graph
            or (
                self._route_vectors is not None
                and self._route_position_by_column is not None
            )
        )
        if (
            not used_cuda_graph
            and self._cuda_graph_runtime is not None
            and self._cuda_graph_runtime.active
        ):
            self._cuda_graph_runtime.sync_after_external_transition(
                reconstruction_error=float(self._route_reconstruction_error.item()),
                winner_id=winner_id,
                effective_modulator=float(self._effective_modulator.item()),
            )
        trainer.model.predictive.last_dense_transition_mode = "inplace_triton"
        trainer.model.predictive.last_dense_transition_fallback_reason = None
        predictive_update_fallback_reason = None
        if not predictive_scope_ready:
            predictive_update_fallback_reason = "candidate_predictive_update_not_due"
        elif use_candidate_predictive_transition:
            predictive_update_fallback_reason = None
        elif (
            trainer.model.device.type == "cuda"
            and int(candidates.numel()) < comp.n_columns
        ):
            predictive_update_fallback_reason = (
                "cuda_sparse_prediction_update_launch_bound_dense_retained"
            )
        if use_candidate_predictive_transition:
            # The kernel has already advanced candidate row stamps on device.
            next_step = int(trainer.model.predictive.predictive_step_count) + 1
            trainer.model.predictive._record_prediction_update_scope(
                candidates,
                fallback_reason=None,
            )
            trainer.model.predictive._record_location_update_scope(candidates)
            trainer.model.predictive.predictive_step_count = next_step
            trainer.model.predictive._predictive_has_cached_columns = True
            trainer.model.predictive._last_predictive_completed_candidates = (
                candidates.detach().clone()
            )
            trainer.model.predictive._last_predictive_completed_step = next_step
            self.candidate_predictive_transition_execution_count += 1
            self.candidate_predictive_transition_cached_count += max(
                0,
                int(comp.n_columns) - int(candidates.numel()),
            )
        else:
            trainer.model.predictive._record_prediction_update_scope(
                None,
                fallback_reason=predictive_update_fallback_reason,
            )
            trainer.model.predictive._record_location_update_scope(None)
            trainer.model.predictive._mark_predictive_update_complete(None)
        comp.last_input_plasticity_mode = "skipped_zero_blend"
        comp.input_plasticity_skip_count += 1
        comp.last_revived_indices = self._empty_revived_indices
        self.graph_empty_revival_tensor_reuse_count += int(used_cuda_graph)
        comp.last_homeostasis_update_count = int(
            homeostasis_candidates.numel()
        )
        comp.last_homeostasis_update_mode = (
            "candidate_subset"
            if int(homeostasis_candidates.numel()) < comp.n_columns
            else "all_columns"
        )
        state_transition_count = int(candidates.numel())
        sparse_state_transition = state_transition_count < int(comp.n_columns)
        comp.last_state_transition_mode = (
            (
                "candidate_subset_sparse_cuda_graph_route_transition"
                if used_cuda_graph
                else "candidate_subset_sparse_inplace_triton"
            )
            if sparse_state_transition
            else (
                "dense_all_columns_cuda_graph_route_transition"
                if used_cuda_graph
                else "dense_all_columns_inplace_triton"
            )
        )
        comp.last_state_transition_column_count = (
            state_transition_count
            if sparse_state_transition
            else int(comp.n_columns)
        )
        comp.last_state_transition_cached_count = (
            max(0, int(comp.n_columns) - state_transition_count)
            if sparse_state_transition
            else 0
        )
        comp.last_state_transition_materialize_mode = (
            "candidate_subset_sparse_cuda_transition"
            if sparse_state_transition
            else "dense_cuda_transition"
        )
        comp.last_state_transition_materialize_count = 0
        comp.last_state_transition_materialize_max_age = 0
        comp.state_transition_step_count += 1
        if not sparse_state_transition:
            mark_state_materialized = getattr(
                comp,
                "_mark_all_state_transition_materialized",
                None,
            )
            if callable(mark_state_materialized):
                mark_state_materialized(
                    int(comp.state_transition_step_count),
                    sync_last_update_tensor=False,
                )
            else:
                comp.steps_since_win_last_update_step.fill_(
                    int(comp.state_transition_step_count)
                )
        comp.homeostasis_last_update_step[
            homeostasis_candidates.to(comp.device).long().flatten()
        ] = int(comp.homeostasis_step_count) + 1
        comp.homeostasis_step_count += 1
        comp.recent_spike_window_cursor = (
            comp.recent_spike_window_cursor + 1
        ) % comp.spike_history_window
        if not used_cuda_graph:
            self._recent_spike_row.fill_(comp.recent_spike_window_cursor)
        comp.recent_spike_window_count = min(
            comp.spike_history_window,
            comp.recent_spike_window_count + 1,
        )
        comp.update_count += int(winners.numel())
        comp._cached_proto_sim = None
        comp._cached_raw_drive = None
        self.execution_count += 1
        self.last_execution_mode = (
            "cuda_graph_route_transition"
            if used_cuda_graph
            else "inplace_triton"
        )
        _profile_mark("column_transition_python_bookkeeping")
        return (
            self._assembly,
            winner_id_list,
            winner_id,
            winner_consolidation,
            (
                float(graph_result[7])
                if used_cuda_graph
                else float(self._effective_modulator.item())
            ),
            float(dopamine_ltp_gain),
            float(serotonin_patience),
        )

    def consume_graph_surprise_update(self) -> bool:
        if self._cuda_graph_runtime is None:
            return False
        return self._cuda_graph_runtime.consume_surprise_update()

    def consume_graph_competitive_surprise(self) -> float | None:
        if self._cuda_graph_runtime is None:
            return None
        return self._cuda_graph_runtime.consume_competitive_surprise()

    def graph_owns_competitive_surprise(self) -> bool:
        return bool(
            self._cuda_graph_runtime is not None
            and self._cuda_graph_runtime.active
            and self._cuda_graph_runtime.owns_competitive_surprise
        )

    def last_tick_used_device_owned_routing_cache(self) -> bool:
        return bool(self._last_tick_device_owned_routing_cache)

    def report(self) -> dict[str, Any]:
        comp = self._trainer.model.competitive
        state_transition_mode = str(
            getattr(comp, "last_state_transition_mode", "not_run")
        )
        state_transition_count = int(
            max(
                0,
                min(
                    int(getattr(comp, "last_state_transition_column_count", 0)),
                    int(comp.n_columns),
                ),
            )
        )
        state_transition_runs_all_columns = bool(
            state_transition_mode != "not_run"
            and state_transition_count >= int(comp.n_columns)
        )
        state_transition_cached_count = int(
            max(
                0,
                min(
                    int(getattr(comp, "last_state_transition_cached_count", 0)),
                    int(comp.n_columns),
                ),
            )
        )
        state_transition_materialize_count = int(
            max(
                0,
                min(
                    int(getattr(comp, "last_state_transition_materialize_count", 0)),
                    int(comp.n_columns),
                ),
            )
        )
        route_scoring = self.route_scoring_snapshot()
        return {
            "surface": "column_transition_runtime.v1",
            "requested_mode": self.requested_mode,
            "resolved_mode": self.resolved_mode,
            "active": self.active,
            "fallback_reason": self.fallback_reason,
            "warmup_attempted": self.warmup_attempted,
            "warmup_succeeded": self.warmup_succeeded,
            "warmup_latency_ms": float(self.warmup_latency_ms),
            "precompiled_candidate_counts": list(
                self.precompiled_candidate_counts
            ),
            "execution_count": int(self.execution_count),
            "failure_count": int(self.failure_count),
            "last_execution_mode": self.last_execution_mode,
            "state_transition_mode": state_transition_mode,
            "state_transition_column_count": state_transition_count,
            "state_transition_cached_count": state_transition_cached_count,
            "state_transition_cached_fraction": (
                float(state_transition_cached_count)
                / float(max(1, int(comp.n_columns)))
            ),
            "state_transition_runs_all_columns": state_transition_runs_all_columns,
            "state_transition_materialize_mode": str(
                getattr(comp, "last_state_transition_materialize_mode", "not_run")
            ),
            "state_transition_materialize_count": state_transition_materialize_count,
            "state_transition_materialize_max_age": int(
                max(0, int(getattr(comp, "last_state_transition_materialize_max_age", 0)))
            ),
            "state_transition_fallback_reason": (
                "dense_state_transition_retained_until_lazy_column_state"
                if state_transition_runs_all_columns
                else None
            ),
            "selection_execution_count": int(self.selection_execution_count),
            "selection_failure_count": int(self.selection_failure_count),
            "last_selection_mode": self.last_selection_mode,
            "device_selection": bool(self.device_selection),
            "selection_host_sync_required": not self.device_selection,
            "fused_vote_competition_requested": bool(
                self.fused_vote_competition_requested
            ),
            "fused_vote_competition_active": bool(
                getattr(self, "fused_vote_competition_active", False)
            ),
            "fused_vote_competition_execution_count": int(
                self.fused_vote_competition_execution_count
            ),
            "fused_vote_competition_fallback_count": int(
                self.fused_vote_competition_fallback_count
            ),
            "route_vote_requested_mode": self.route_vote_requested_mode,
            "route_vote_resolved_mode": self.route_vote_resolved_mode,
            "route_vote_active": self.handles_route_vote,
            "route_vote_fallback_reason": self.route_vote_fallback_reason,
            "route_vote_warmup_attempted": self.route_vote_warmup_attempted,
            "route_vote_warmup_succeeded": self.route_vote_warmup_succeeded,
            "route_vote_warmup_latency_ms": self.route_vote_warmup_latency_ms,
            "route_vote_execution_count": self.route_vote_execution_count,
            "route_vote_sensory_fallback_count": (
                self.route_vote_sensory_fallback_count
            ),
            "route_vote_cache_refresh_count": (
                self.route_vote_cache_refresh_count
            ),
            "route_vote_clean_cache_reuse_count": (
                self.route_vote_clean_cache_reuse_count
            ),
            "route_vote_prepared_graph_reuse_count": (
                self.route_vote_prepared_graph_reuse_count
            ),
            "route_vote_kernel_variant": self.route_vote_kernel_variant,
            "route_candidate_bank": {
                "enabled": bool(self.route_candidate_bank_enabled),
                "ready": bool(self.route_candidate_bank_ready),
                "bank_size": int(self.route_candidate_bank_size),
                "seed_count": int(self.route_candidate_bank_seed_count),
                "refresh_count": int(self.route_candidate_bank_refresh_count),
                "graph_bypass_count": int(
                    self.route_candidate_bank_graph_bypass_count
                ),
                "fallback_count": int(self.route_candidate_bank_fallback_count),
                "last_reason": self.route_candidate_bank_last_reason,
                "claim_boundary": (
                    "training_owned_bounded_route_rows_seeded_from_exact_route_without_hot_path_all_column_rescore"
                ),
            },
            "route_vote_scoring": route_scoring,
            "route_vote_input_rows_scored": route_scoring[
                "route_input_rows_scored"
            ],
            "route_vote_output_candidate_count": route_scoring[
                "route_output_candidate_count"
            ],
            "route_vote_rows_run_all_columns": route_scoring[
                "route_rows_run_all_columns"
            ],
            "route_vote_bounded_route_scoring": route_scoring[
                "bounded_route_scoring"
            ],
            "route_vote_deep_sleep_filter": self.route_sleep_filter_snapshot(),
            "graph_host_winner_reuse_count": int(
                self.graph_host_winner_reuse_count
            ),
            "graph_consolidation_lookup_skip_count": int(
                self.graph_consolidation_lookup_skip_count
            ),
            "graph_empty_revival_tensor_reuse_count": int(
                self.graph_empty_revival_tensor_reuse_count
            ),
            "candidate_predictive_transition_mode": (
                self.candidate_predictive_transition_mode
            ),
            "candidate_predictive_transition_active": bool(
                self.candidate_predictive_transition_active
            ),
            "candidate_predictive_transition_fallback_reason": (
                self.candidate_predictive_transition_fallback_reason
            ),
            "candidate_predictive_transition_execution_count": int(
                self.candidate_predictive_transition_execution_count
            ),
            "candidate_predictive_transition_cached_count": int(
                self.candidate_predictive_transition_cached_count
            ),
            "winner_consolidation_cpu_metric_count": int(
                self.winner_consolidation_cpu_metric_count
            ),
            "winner_consolidation_cached_metric_count": int(
                self.winner_consolidation_cached_metric_count
            ),
            "cuda_graph_route_transition": (
                None
                if self._cuda_graph_runtime is None
                else self._cuda_graph_runtime.report()
            ),
            "tensor_device": str(self._assembly.device),
            "mutates_runtime_state": bool(self.active),
            "fallback_happens_before_mutation": True,
            "runtime_failure_policy": "fail_closed_no_post_mutation_fallback",
        }
