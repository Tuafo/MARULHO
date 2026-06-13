from __future__ import annotations

import time
from typing import Any

import torch

from marulho.core.fused_route_vote_cuda import (
    fused_route_vote_cuda,
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
        self.graph_host_winner_reuse_count = 0
        self._route_vote_ready = False
        self._route_transition_graph_ready = False
        self._prepared_graph_token: int | None = None
        self._cuda_graph_runtime: CudaGraphRouteTransition | None = None
        self._route_vectors: torch.Tensor | None = None
        self._route_ids: torch.Tensor | None = None
        self._route_scores: torch.Tensor | None = None
        self._route_candidates: torch.Tensor | None = None
        comp = trainer.model.competitive
        device = trainer.model.device
        self._assembly = torch.empty(comp.n_columns, device=device)
        self._winner = torch.empty(1, dtype=torch.long, device=device)
        self._strength = torch.ones(1, device=device)
        self._prediction_boost = torch.empty((), device=device)
        self._effective_modulator = torch.empty((), device=device)
        self._zero_consolidation = torch.zeros(comp.n_columns, device=device)
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

        self.warmup_attempted = True
        started = time.perf_counter_ns()
        try:
            candidate_counts = sorted(
                {
                    int(comp.n_columns),
                    min(int(comp.n_columns), max(1, int(trainer.config.k_routing))),
                }
            )
            for candidate_count in candidate_counts:
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
                    location=trainer.model.predictive.location,
                    location_velocity=trainer.model.predictive.velocity,
                    prediction_weights=trainer.model.predictive._prediction_weights,
                    prediction_error=trainer.model.predictive.prediction_error,
                    prediction_failure_streak=(
                        trainer.model.predictive.prediction_failure_streak
                    ),
                    confidence=trainer.model.predictive.confidence,
                    recent_spike_window=comp.recent_spike_window,
                    assembly=self._assembly,
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
        index = self._trainer.model.hnsw_index
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
            self._route_scores = torch.empty(
                int(ids.numel()),
                device=vectors.device,
            )
            self._route_candidates = torch.empty(
                candidate_count,
                dtype=torch.long,
                device=vectors.device,
            )
            warmup_fused_route_vote_cuda(
                routing_key=torch.empty(
                    int(vectors.shape[1]),
                    device=vectors.device,
                ),
                routing_vectors=vectors,
                routing_ids=ids,
                prototypes=self._trainer.model.competitive.prototypes,
                thresholds=self._trainer.model.competitive.thresholds,
                prediction_location=self._trainer.model.predictive.location,
                previous_winner=self._previous_winner,
                scores_out=self._route_scores,
                candidates_out=self._route_candidates,
                winner_out=self._winner,
                strength_out=self._strength,
                competition_had_positive=self._competition_had_positive,
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
            return self._route_candidates
        index = self._trainer.model.hnsw_index
        assert self._route_vectors is not None
        assert self._route_ids is not None
        assert self._route_scores is not None
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
            self.route_vote_cache_refresh_count += 1
        route_vectors = self._route_vectors
        route_ids = self._route_ids
        route_scores = self._route_scores
        route_candidates = self._route_candidates
        assert route_vectors is not None
        assert route_ids is not None
        assert route_scores is not None
        assert route_candidates is not None
        if (
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
            return route_candidates
        fused_route_vote_cuda(
            routing_key=routing_key,
            routing_vectors=route_vectors,
            routing_ids=route_ids,
            prototypes=self._trainer.model.competitive.prototypes,
            thresholds=self._trainer.model.competitive.thresholds,
            prediction_location=self._trainer.model.predictive.location,
            previous_winner=self._previous_winner,
            scores_out=route_scores,
            candidates_out=route_candidates,
            winner_out=self._winner,
            strength_out=self._strength,
            competition_had_positive=self._competition_had_positive,
        )
        comp = self._trainer.model.competitive
        comp.last_candidate_count = int(route_candidates.numel())
        comp.last_scored_column_count = int(route_candidates.numel())
        comp.last_execution_mode = "candidate_subset_fused_route_vote"
        self.route_vote_execution_count += 1
        self._route_vote_ready = True
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
            self.last_selection_mode = "fused_route_vote_triton"
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
        trainer = self._trainer
        comp = trainer.model.competitive
        if candidates is None:
            candidates = self._all_columns
        homeostasis_scope_ready = trainer.token_count >= int(
            trainer.config.candidate_homeostasis_start_tokens
        )
        homeostasis_candidates = (
            candidates if homeostasis_scope_ready else self._all_columns
        )
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
        used_cuda_graph = bool(
            self._route_transition_graph_ready
            and self._cuda_graph_runtime is not None
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
                inplace_column_transition_cuda(
                    prototypes=comp.prototypes,
                    prototype_velocity=comp.prototype_velocity,
                    thresholds=comp.thresholds,
                    win_rate_ema=comp.win_rate_ema,
                    steps_since_win=comp.steps_since_win,
                    location=trainer.model.predictive.location,
                    location_velocity=trainer.model.predictive.velocity,
                    prediction_weights=trainer.model.predictive._prediction_weights,
                    prediction_error=trainer.model.predictive.prediction_error,
                    prediction_failure_streak=(
                        trainer.model.predictive.prediction_failure_streak
                    ),
                    confidence=trainer.model.predictive.confidence,
                    recent_spike_window=comp.recent_spike_window,
                    assembly=self._assembly,
                    prediction_boost_out=self._prediction_boost,
                    effective_modulator_out=self._effective_modulator,
                    routing_key=routing_key,
                    previous_routing_key=previous,
                    winners=winners,
                    candidates=homeostasis_candidates,
                    consolidation=consolidation,
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
        trainer.model.predictive.last_dense_transition_mode = "inplace_triton"
        trainer.model.predictive.last_dense_transition_fallback_reason = None
        trainer.model.predictive._record_prediction_update_scope(None)
        comp.last_input_plasticity_mode = "skipped_zero_blend"
        comp.input_plasticity_skip_count += 1
        comp.last_revived_indices = torch.empty(
            0,
            device=trainer.model.device,
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
        return bool(
            self.last_execution_mode == "cuda_graph_route_transition"
            and self._cuda_graph_runtime is not None
            and self._cuda_graph_runtime.owns_routing_cache_updates
        )

    def report(self) -> dict[str, Any]:
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
            "graph_host_winner_reuse_count": int(
                self.graph_host_winner_reuse_count
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
