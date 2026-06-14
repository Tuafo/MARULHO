from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F

from marulho.core.columns import (
    _normalize_routing_key,
)
from marulho.core.fused_route_vote_cuda import fused_route_vote_cuda
from marulho.core.inplace_column_cuda import inplace_column_transition_cuda


MAX_QUANTUM_INPUT_TOKENS = 128


class CudaGraphRouteTransition:
    """Capture the fixed-shape text tick as one persistent CUDA replay."""

    def __init__(self, trainer: Any, runtime: Any) -> None:
        self._trainer = trainer
        self._runtime = runtime
        self.active = False
        self.fallback_reason: str | None = None
        self.capture_attempted = False
        self.capture_succeeded = False
        self.capture_latency_ms = 0.0
        self.capture_count = 0
        self.tick_replay_count = 0
        self.pre_route_sensory_bypass_count = 0
        self.pre_route_bootstrap_bypass_count = 0
        self.replay_count = 0
        self.failure_count = 0
        self.host_truth_sync_count = 0
        self.host_truth_skip_count = 0
        self.surprise_update_count = 0
        self.host_truth_mirror_update_count = 0
        self.competitive_surprise_update_count = 0
        self.route_cache_clean_fastpath_count = 0
        self.route_cache_rebuild_check_count = 0
        self.route_cache_generation_fastpath_count = 0
        self.route_cache_generation_mismatch_count = 0
        self.route_cache_device_update_count = 0
        self.consolidation_cache_generation_fastpath_count = 0
        self.consolidation_cache_generation_mismatch_count = 0
        self.consolidation_memory_warm_state_mismatch_count = 0
        self.previous_flag_device_owned_count = 0
        self.learning_rate_device_owned_count = 0
        self.learning_rate_host_resync_count = 0
        self.modulator_stage_copy_count = 0
        self.modulator_stage_skip_count = 0
        self.quantum_input_stage_count = 0
        self.quantum_input_staged_token_count = 0
        self.quantum_input_reuse_count = 0
        self.quantum_input_fallback_copy_count = 0
        self.quantum_input_mismatch_count = 0
        self.quantum_input_discard_count = 0
        self.recent_spike_row_device_owned_count = 0
        self._graphs: dict[str, torch.cuda.CUDAGraph] = {}
        self._graph_outputs: dict[str, dict[str, torch.Tensor]] = {}
        self._route_vectors: torch.Tensor | None = None
        self._route_ids: torch.Tensor | None = None
        self._route_position_by_column: torch.Tensor | None = None
        self._route_cache_generation: int | None = None
        self._consolidation: torch.Tensor | None = None
        self._consolidation_cache_generation: int | None = None
        self._captured_memory_warm_started: bool | None = None
        self._previous_routing_key: torch.Tensor | None = None
        self._parameters: torch.Tensor | None = None
        self._parameter_host_prefix: torch.Tensor | None = None
        self._parameter_device_prefix: torch.Tensor | None = None
        self._host_parameters: torch.Tensor | None = None
        self._cached_modulator_revision: int | None = None
        self._cached_modulator_value = 0.0
        self._learning_rate_update_count: torch.Tensor | None = None
        self._learning_rate_update_count_mirror: int | None = None
        self._input_patterns: torch.Tensor | None = None
        self._input_slot: torch.Tensor | None = None
        self._input_slot_mirror = 0
        self._staged_pattern_pointers: list[int] = []
        self._staged_pattern_offset = 0
        self._neuromodulator_state: torch.Tensor | None = None
        self._result: torch.Tensor | None = None
        self._last_graph_name: str | None = None
        self._last_result: tuple[float, ...] | None = None
        self._last_result_from_host_sync = False
        self._surprise_update_pending = False
        self._competitive_surprise_pending: float | None = None
        self._owns_competitive_surprise = not bool(
            getattr(
                trainer,
                "_disable_graph_competitive_surprise_for_evaluation",
                False,
            )
        )
        self._capture()

    def _pre_route_ops(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trainer = self._trainer
        comp = trainer.model.competitive
        assert self._input_patterns is not None
        assert self._input_slot is not None
        x = torch.index_select(
            self._input_patterns,
            0,
            self._input_slot.reshape(1),
        ).squeeze(0)
        normalized_input = x.float().clamp(min=0.0)
        normalized_input = normalized_input / torch.clamp(
            normalized_input.sum(),
            min=1e-8,
        )
        projected = torch.mv(
            comp.W_project.t(),
            x.clamp(min=0.0),
        )
        projected = _normalize_routing_key(
            projected,
            trainer.model.device,
        )
        routing_key = F.normalize(projected, dim=0)
        return (
            normalized_input,
            projected,
            routing_key,
        )

    def _update_neuromodulators(
        self,
        reconstruction_error: torch.Tensor,
    ) -> None:
        assert self._neuromodulator_state is not None
        assert self._parameters is not None
        predicted_error = self._neuromodulator_state[0]
        dopamine = self._neuromodulator_state[1]
        acetylcholine = self._neuromodulator_state[2]
        norepinephrine = self._neuromodulator_state[3]
        serotonin = self._neuromodulator_state[4]
        baseline = predicted_error + 1e-6
        fraction = (predicted_error - reconstruction_error) / baseline
        rpe = torch.tanh(fraction * 3.0)
        serotonin_drive = torch.tanh(
            torch.clamp(-fraction, min=0.0) * 3.0,
        )
        unexpected_uncertainty = torch.clamp(
            torch.tanh(torch.abs(fraction) * 2.0),
            min=0.0,
            max=1.0,
        )
        novelty = torch.clamp(reconstruction_error, min=0.0, max=1.0)
        dopamine = torch.clamp(
            0.85 * dopamine + 0.15 * torch.clamp(rpe, min=0.0),
            min=0.0,
            max=1.0,
        )
        serotonin = torch.clamp(
            0.85 * serotonin + 0.15 * serotonin_drive,
            min=0.0,
            max=1.0,
        )
        acetylcholine = torch.clamp(
            0.90 * acetylcholine + 0.10 * novelty,
            min=0.0,
            max=1.0,
        )
        norepinephrine = torch.clamp(
            0.85 * norepinephrine + 0.15 * unexpected_uncertainty,
            min=0.0,
            max=1.0,
        )
        norepinephrine = torch.clamp(
            norepinephrine
            + torch.where(
                serotonin_drive > 0.4,
                0.10 * serotonin_drive,
                torch.zeros_like(serotonin_drive),
            ),
            min=0.0,
            max=1.0,
        )
        predicted_error = (
            0.01 * reconstruction_error + 0.99 * predicted_error
        )
        self._neuromodulator_state.copy_(
            torch.stack(
                (
                    predicted_error,
                    dopamine,
                    acetylcholine,
                    norepinephrine,
                    serotonin,
                )
            )
        )
        self._parameters[1].copy_(dopamine)
        self._parameters[2].copy_(serotonin)

    def _tick_ops(
        self,
        candidates: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        trainer = self._trainer
        normalized_input, projected, routing_key = self._pre_route_ops()
        assert self._result is not None
        self._launch_route_vote(
            candidates,
            routing_key=routing_key,
            reconstruction_error_out=self._result.narrow(0, 0, 1),
        )
        reconstruction_error = self._result[0]
        self._update_neuromodulators(reconstruction_error)
        assert self._parameters is not None
        assert self._learning_rate_update_count is not None
        comp = trainer.model.competitive
        self._parameters[3].copy_(
            comp.lr_initial
            / (1.0 + comp.lr_decay * self._learning_rate_update_count)
        )
        self._launch_transition(candidates, routing_key=routing_key)
        self._learning_rate_update_count.add_(1.0)
        runtime = self._runtime
        assert self._neuromodulator_state is not None
        assert self._result is not None
        self._result[1:6].copy_(self._neuromodulator_state)
        self._result[6].copy_(runtime._winner[0])
        self._result[7].copy_(runtime._effective_modulator)
        if self._owns_competitive_surprise:
            self._result[8].copy_(
                torch.norm(
                    trainer.model.competitive.prototypes.index_select(
                        0,
                        runtime._winner,
                    ).squeeze(0)
                    - routing_key,
                )
            )
        assert self._input_slot is not None
        self._input_slot.add_(1)
        self._input_slot.remainder_(MAX_QUANTUM_INPUT_TOKENS)
        return {
            "normalized_input": normalized_input,
            "projected_input": projected,
            "routing_key": routing_key,
            "reconstruction_error": reconstruction_error,
            "result": self._result,
        }

    def _capture(self) -> None:
        trainer = self._trainer
        runtime = self._runtime
        comp = trainer.model.competitive
        pred = trainer.model.predictive
        device = trainer.model.device
        self.capture_attempted = True
        started = time.perf_counter_ns()
        try:
            if device.type != "cuda":
                raise RuntimeError("cuda_graph_requires_cuda")
            vectors, ids = trainer.model.hnsw_index.routing_tensor_cache()
            route_generation_fn = getattr(
                trainer.model.hnsw_index,
                "routing_tensor_cache_generation",
                None,
            )
            if runtime._route_scores is None or runtime._route_candidates is None:
                raise RuntimeError("cuda_graph_requires_route_workspaces")
            self._route_vectors = vectors
            self._route_ids = ids
            route_ids_cpu = ids.detach().to(device="cpu", dtype=torch.long)
            if int(route_ids_cpu.numel()) != int(comp.n_columns):
                raise RuntimeError(
                    "cuda_graph_requires_complete_routing_cache"
                )
            route_ids_list = route_ids_cpu.tolist()
            if (
                len(set(int(value) for value in route_ids_list))
                != int(comp.n_columns)
                or min(route_ids_list, default=-1) < 0
                or max(route_ids_list, default=-1) >= int(comp.n_columns)
            ):
                raise RuntimeError(
                    "cuda_graph_requires_unique_column_routing_ids"
                )
            self._route_position_by_column = torch.empty(
                comp.n_columns,
                dtype=torch.long,
                device=device,
            )
            self._route_position_by_column[ids.long()] = torch.arange(
                int(ids.numel()),
                dtype=torch.long,
                device=device,
            )
            self._route_cache_generation = (
                int(route_generation_fn())
                if callable(route_generation_fn)
                else None
            )
            self._input_patterns = torch.empty(
                (MAX_QUANTUM_INPUT_TOKENS, trainer.config.input_dim),
                device=device,
            )
            self._input_patterns.zero_()
            self._input_slot = torch.zeros(
                (),
                dtype=torch.long,
                device=device,
            )
            self._input_slot_mirror = 0
            self._previous_routing_key = torch.zeros(
                comp.column_dim,
                device=device,
            )
            has_previous = trainer._prev_routing_key is not None
            if has_previous:
                self._previous_routing_key.copy_(trainer._prev_routing_key)
            self._parameters = torch.empty(5, device=device)
            self._parameters.zero_()
            self._parameters[4].fill_(float(has_previous))
            self._parameter_device_prefix = self._parameters.narrow(0, 0, 1)
            self._host_parameters = torch.empty(
                1,
                dtype=torch.float32,
                pin_memory=True,
            )
            self._parameter_host_prefix = self._host_parameters
            self._host_parameters.zero_()
            assert self._parameter_device_prefix is not None
            self._parameter_device_prefix.copy_(
                self._parameter_host_prefix,
                non_blocking=True,
            )
            self._learning_rate_update_count = torch.tensor(
                float(comp.update_count),
                dtype=torch.float32,
                device=device,
            )
            self._learning_rate_update_count_mirror = int(comp.update_count)
            surprise = trainer.model.surprise
            self._neuromodulator_state = torch.tensor(
                [
                    surprise.predicted_error,
                    surprise.dopamine,
                    surprise.acetylcholine,
                    surprise.norepinephrine,
                    surprise.serotonin,
                ],
                dtype=torch.float32,
                device=device,
            )
            self._result = torch.empty(
                9 if self._owns_competitive_surprise else 8,
                dtype=torch.float32,
                device=device,
            )
            self._consolidation = (
                trainer.model.memory_store.bucket_consolidation_tensor(
                    comp.n_columns,
                    device=device,
                )
                if trainer.memory_warm_started
                else runtime._zero_consolidation
            )
            self._captured_memory_warm_started = bool(trainer.memory_warm_started)
            self._consolidation_cache_generation = (
                int(
                    trainer.model.memory_store.bucket_consolidation_cache_generation
                )
                if trainer.memory_warm_started
                else None
            )
            mutable = (
                comp.prototypes,
                comp.prototype_velocity,
                comp.thresholds,
                comp.win_rate_ema,
                comp.steps_since_win,
                comp.recent_spike_window,
                pred.location,
                pred.velocity,
                pred._prediction_weights,
                pred.prediction_error,
                pred.prediction_failure_streak,
                pred.confidence,
                runtime._assembly,
                runtime._winner,
                runtime._strength,
                runtime._prediction_boost,
                runtime._effective_modulator,
                runtime._competition_had_positive,
                runtime._previous_winner,
                runtime._recent_spike_row,
                self._previous_routing_key,
                self._parameters,
                self._learning_rate_update_count,
                self._neuromodulator_state,
                self._result,
                self._route_vectors,
                self._input_slot,
            )
            snapshots = tuple(tensor.clone() for tensor in mutable)
            stream = torch.cuda.Stream(device=device)
            for name, candidates in (
                ("all_columns", runtime._all_columns),
                ("candidate_subset", runtime._route_candidates),
            ):
                for tensor, snapshot in zip(mutable, snapshots):
                    tensor.copy_(snapshot)
                torch.cuda.synchronize(device)
                self._tick_ops(candidates)
                torch.cuda.synchronize(device)
                for tensor, snapshot in zip(mutable, snapshots):
                    tensor.copy_(snapshot)
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    outputs = self._tick_ops(candidates)
                torch.cuda.synchronize(device)
                self._graphs[name] = graph
                self._graph_outputs[name] = outputs
                self.capture_count += 1
            for tensor, snapshot in zip(mutable, snapshots):
                tensor.copy_(snapshot)
            torch.cuda.synchronize(device)
            self.capture_succeeded = True
            self.active = True
        except Exception as exc:
            self.fallback_reason = (
                f"cuda_graph_capture_failed:{type(exc).__name__}:{exc}"
            )
            self._graphs.clear()
            self._graph_outputs.clear()
        finally:
            self.capture_latency_ms = (
                time.perf_counter_ns() - started
            ) / 1e6

    def _launch_route_vote(
        self,
        candidates: torch.Tensor,
        *,
        routing_key: torch.Tensor,
        reconstruction_error_out: torch.Tensor,
    ) -> None:
        trainer = self._trainer
        runtime = self._runtime
        comp = trainer.model.competitive
        pred = trainer.model.predictive
        assert self._previous_routing_key is not None
        assert self._parameters is not None
        assert self._route_vectors is not None
        assert self._route_ids is not None
        assert self._route_position_by_column is not None
        assert self._consolidation is not None
        assert runtime._route_scores is not None
        assert runtime._route_candidates is not None
        fused_route_vote_cuda(
            routing_key=routing_key,
            routing_vectors=self._route_vectors,
            routing_ids=self._route_ids,
            prototypes=comp.prototypes,
            thresholds=comp.thresholds,
            prediction_location=pred.location,
            previous_winner=runtime._previous_winner,
            scores_out=runtime._route_scores,
            candidates_out=runtime._route_candidates,
            winner_out=runtime._winner,
            strength_out=runtime._strength,
            competition_had_positive=runtime._competition_had_positive,
            reconstruction_error_out=reconstruction_error_out,
        )

    def _launch_transition(
        self,
        candidates: torch.Tensor,
        *,
        routing_key: torch.Tensor,
    ) -> None:
        trainer = self._trainer
        runtime = self._runtime
        comp = trainer.model.competitive
        pred = trainer.model.predictive
        assert self._previous_routing_key is not None
        assert self._parameters is not None
        assert self._route_vectors is not None
        assert self._route_position_by_column is not None
        assert self._consolidation is not None
        inplace_column_transition_cuda(
            prototypes=comp.prototypes,
            routing_vectors=self._route_vectors,
            routing_position_by_column=self._route_position_by_column,
            prototype_velocity=comp.prototype_velocity,
            thresholds=comp.thresholds,
            win_rate_ema=comp.win_rate_ema,
            steps_since_win=comp.steps_since_win,
            location=pred.location,
            location_velocity=pred.velocity,
            prediction_weights=pred._prediction_weights,
            prediction_error=pred.prediction_error,
            prediction_failure_streak=pred.prediction_failure_streak,
            confidence=pred.confidence,
            recent_spike_window=comp.recent_spike_window,
            assembly=runtime._assembly,
            prediction_boost_out=runtime._prediction_boost,
            effective_modulator_out=runtime._effective_modulator,
            routing_key=routing_key,
            previous_routing_key=self._previous_routing_key,
            winners=runtime._winner,
            candidates=candidates,
            consolidation=self._consolidation,
            transition_parameters=self._parameters,
            base_modulator=0.0,
            dopamine=0.0,
            serotonin=0.0,
            competitive_learning_rate=0.0,
            recent_spike_row=runtime._recent_spike_row,
            has_previous_routing_key=False,
            persist_previous_routing_key=True,
            advance_recent_spike_row=True,
            spike_history_window=comp.spike_history_window,
            competition_had_positive=runtime._competition_had_positive,
            prototype_momentum=comp.prototype_momentum,
            homeostasis_beta=comp.homeostasis_beta,
            homeostasis_lr=comp.homeostasis_lr,
            target_firing_rate=comp.target_firing_rate,
            threshold_min=comp.threshold_min,
            threshold_max=comp.threshold_max,
            prediction_error_ema_alpha=pred._error_ema_alpha,
            prediction_failure_streak_threshold=pred._failure_streak_threshold,
            prediction_learning_rate=0.005,
        )

    def eligible(self, *, assume_route_cache_current: bool = False) -> bool:
        if not self.active or not self._graphs:
            return False
        vectors = self._route_vectors
        ids = self._route_ids
        route_pointers_known_current = bool(assume_route_cache_current)
        if not assume_route_cache_current:
            index = self._trainer.model.hnsw_index
            generation_fn = getattr(
                index,
                "routing_tensor_cache_generation",
                None,
            )
            generation = (
                int(generation_fn())
                if callable(generation_fn)
                else None
            )
            if (
                generation is not None
                and self._route_cache_generation == generation
            ):
                self.route_cache_generation_fastpath_count += 1
                route_pointers_known_current = True
            else:
                if generation is not None:
                    self.route_cache_generation_mismatch_count += 1
                cache_dirty = True
                cache_dirty_fn = getattr(index, "routing_tensor_cache_is_dirty", None)
                if callable(cache_dirty_fn):
                    cache_dirty = bool(cache_dirty_fn())
                if cache_dirty:
                    vectors, ids = index.routing_tensor_cache()
                    self.route_cache_rebuild_check_count += 1
                    if callable(generation_fn):
                        self._route_cache_generation = int(generation_fn())
                else:
                    self.route_cache_clean_fastpath_count += 1
                    self._route_cache_generation = generation
        if not route_pointers_known_current:
            if vectors is None or ids is None:
                self.active = False
                self.fallback_reason = "cuda_graph_static_pointer_changed"
                return False
            if (
                self._route_vectors is None
                or self._route_ids is None
                or vectors.data_ptr() != self._route_vectors.data_ptr()
                or ids.data_ptr() != self._route_ids.data_ptr()
            ):
                self.active = False
                self.fallback_reason = "cuda_graph_static_pointer_changed"
                return False
        memory_warm_started = bool(self._trainer.memory_warm_started)
        if memory_warm_started != bool(self._captured_memory_warm_started):
            self.consolidation_memory_warm_state_mismatch_count += 1
            self.active = False
            self.fallback_reason = "cuda_graph_memory_warm_state_changed"
            return False
        if memory_warm_started:
            generation = int(
                self._trainer.model.memory_store.bucket_consolidation_cache_generation
            )
            if generation != self._consolidation_cache_generation:
                self.consolidation_cache_generation_mismatch_count += 1
                self.active = False
                self.fallback_reason = (
                    "cuda_graph_consolidation_cache_generation_changed"
                )
                return False
            self.consolidation_cache_generation_fastpath_count += 1
        consolidation = (
            self._consolidation
            if memory_warm_started
            else self._runtime._zero_consolidation
        )
        if (
            self._route_vectors is None
            or self._route_ids is None
            or self._consolidation is None
            or consolidation is None
            or consolidation.data_ptr() != self._consolidation.data_ptr()
        ):
            self.active = False
            self.fallback_reason = "cuda_graph_static_pointer_changed"
            return False
        return True

    def prepare_routing(
        self,
        pattern: torch.Tensor,
        *,
        sensory_tick: bool,
        assume_eligible: bool = False,
    ) -> tuple[torch.Tensor, float] | None:
        profile_enabled = bool(
            getattr(self._trainer, "_train_step_profile_enabled", False)
        )
        profile_totals = (
            getattr(self._trainer, "_train_step_profile_totals_ms", {})
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

        if sensory_tick:
            self._discard_staged_inputs()
            self.pre_route_sensory_bypass_count += 1
            return None
        if self._trainer.token_count < self._trainer.config.bootstrap_tokens:
            self.pre_route_bootstrap_bypass_count += 1
            return None
        if not assume_eligible and not self.eligible():
            raise RuntimeError(self.fallback_reason or "cuda_graph_not_active")
        _profile_mark("cuda_graph_prepare_eligible")
        assert self._input_patterns is not None
        assert self._input_slot is not None
        assert self._parameters is not None
        assert self._parameter_device_prefix is not None
        assert self._parameter_host_prefix is not None
        assert self._host_parameters is not None
        assert self._previous_routing_key is not None
        assert self._learning_rate_update_count is not None
        assert self._learning_rate_update_count_mirror is not None
        trainer = self._trainer
        comp = trainer.model.competitive
        has_previous = trainer._prev_routing_key is not None
        if (
            has_previous
            and trainer._prev_routing_key is not None
            and trainer._prev_routing_key.data_ptr()
            != self._previous_routing_key.data_ptr()
        ):
            self._previous_routing_key.copy_(trainer._prev_routing_key)
        _profile_mark("cuda_graph_prepare_previous_key")
        if int(comp.update_count) != int(self._learning_rate_update_count_mirror):
            self._learning_rate_update_count.fill_(float(comp.update_count))
            self._learning_rate_update_count_mirror = int(comp.update_count)
            self.learning_rate_host_resync_count += 1
        surprise = trainer.model.surprise
        modulator_revision = int(
            getattr(surprise, "modulator_revision", 0)
        )
        if self._cached_modulator_revision != modulator_revision:
            self._cached_modulator_value = float(
                surprise.get_modulator("competitive")
            )
            self._cached_modulator_revision = modulator_revision
            self._host_parameters[0] = self._cached_modulator_value
            self._parameter_device_prefix.copy_(
                self._parameter_host_prefix,
                non_blocking=True,
            )
            self.modulator_stage_copy_count += 1
        else:
            self.modulator_stage_skip_count += 1
        self.previous_flag_device_owned_count += 1
        self.learning_rate_device_owned_count += 1
        _profile_mark("cuda_graph_prepare_parameter_stage")
        self.recent_spike_row_device_owned_count += 1
        _profile_mark("cuda_graph_prepare_recent_row_fill")
        pattern_pointer = int(pattern.data_ptr())
        staged_pointer = self._next_staged_pattern_pointer()
        if staged_pointer is not None and staged_pointer == pattern_pointer:
            self._staged_pattern_offset += 1
            self.quantum_input_reuse_count += 1
            if self._staged_pattern_offset >= len(self._staged_pattern_pointers):
                self._discard_staged_inputs(count_discard=False)
        else:
            if staged_pointer is not None:
                self.quantum_input_mismatch_count += 1
                self._discard_staged_inputs()
            self._input_patterns[self._input_slot_mirror].copy_(
                pattern,
                non_blocking=True,
            )
            self.quantum_input_fallback_copy_count += 1
        _profile_mark("cuda_graph_prepare_input_stage")
        _profile_mark("cuda_graph_prepare_input_copy")
        graph_name = (
            "candidate_subset"
            if trainer.token_count
            >= int(trainer.config.candidate_homeostasis_start_tokens)
            else "all_columns"
        )
        try:
            self._graphs[graph_name].replay()
        except Exception:
            self.failure_count += 1
            raise
        _profile_mark("cuda_graph_prepare_replay")
        self._learning_rate_update_count_mirror += 1
        self._input_slot_mirror = (
            self._input_slot_mirror + 1
        ) % MAX_QUANTUM_INPUT_TOKENS
        self.tick_replay_count += 1
        self.replay_count += 1
        self.route_cache_device_update_count += 1
        self._last_graph_name = graph_name
        outputs = self._graph_outputs[graph_name]
        self.surprise_update_count += 1
        self._surprise_update_pending = True
        self._competitive_surprise_pending = None
        sync_interval = max(
            1,
            int(
                getattr(
                    trainer.config,
                    "cuda_graph_host_truth_sync_interval_tokens",
                    1,
                )
            ),
        )
        sync_due = (
            self._last_result is None
            or sync_interval <= 1
            or self.replay_count % sync_interval == 0
        )
        if sync_due:
            result = tuple(float(value) for value in outputs["result"].tolist())
            _profile_mark("cuda_graph_prepare_host_truth_sync")
            self.host_truth_sync_count += 1
            self.host_truth_mirror_update_count += 1
            self._last_result = result
            self._last_result_from_host_sync = True
            (
                reconstruction_error,
                predicted_error,
                dopamine,
                acetylcholine,
                norepinephrine,
                serotonin,
                _winner,
                _effective_modulator,
                *optional_competitive_surprise,
            ) = result
            surprise = trainer.model.surprise
            surprise.predicted_error = predicted_error
            surprise.dopamine = dopamine
            surprise.acetylcholine = acetylcholine
            surprise.norepinephrine = norepinephrine
            surprise.serotonin = serotonin
            mark_changed = getattr(
                surprise,
                "mark_modulator_state_changed",
                None,
            )
            if callable(mark_changed):
                mark_changed()
            if optional_competitive_surprise:
                self._competitive_surprise_pending = float(
                    optional_competitive_surprise[0]
                )
                self.competitive_surprise_update_count += 1
        else:
            self.host_truth_skip_count += 1
            _profile_mark("cuda_graph_prepare_host_truth_skip")
            assert self._last_result is not None
            self._last_result_from_host_sync = False
            reconstruction_error = float(self._last_result[0])
        trainer._prev_routing_key = self._previous_routing_key
        comp.last_input_pattern = outputs["normalized_input"]
        comp.last_projected_input = outputs["projected_input"]
        comp._cached_proto_sim = None
        comp._cached_raw_drive = None
        comp.last_execution_mode = "candidate_subset_persistent_tick_graph"
        comp.last_scored_column_count = int(
            self._runtime._route_candidates.numel()
        )
        comp.last_candidate_count = int(
            self._runtime._route_candidates.numel()
        )
        _profile_mark("cuda_graph_prepare_bookkeeping")
        return (
            outputs["routing_key"],
            reconstruction_error,
        )

    @torch.no_grad()
    def stage_input_quantum(self, patterns: list[torch.Tensor]) -> bool:
        if (
            not self.active
            or not bool(
                getattr(
                    self._trainer.config,
                    "cuda_graph_quantum_input_staging",
                    True,
                )
            )
            or not patterns
            or len(patterns) > MAX_QUANTUM_INPUT_TOKENS
        ):
            return False
        assert self._input_patterns is not None
        expected_shape = (int(self._trainer.config.input_dim),)
        device = self._input_patterns.device
        for pattern in patterns:
            if tuple(pattern.shape) != expected_shape or pattern.device != device:
                self.quantum_input_mismatch_count += 1
                self._discard_staged_inputs()
                return False

        self._discard_staged_inputs(count_discard=False)
        start = int(self._input_slot_mirror)
        first_count = min(
            len(patterns),
            MAX_QUANTUM_INPUT_TOKENS - start,
        )
        torch.stack(
            patterns[:first_count],
            out=self._input_patterns[start : start + first_count],
        )
        remaining = len(patterns) - first_count
        if remaining > 0:
            torch.stack(
                patterns[first_count:],
                out=self._input_patterns[:remaining],
            )
        self._staged_pattern_pointers = [
            int(pattern.data_ptr()) for pattern in patterns
        ]
        self._staged_pattern_offset = 0
        self.quantum_input_stage_count += 1
        self.quantum_input_staged_token_count += len(patterns)
        return True

    def _next_staged_pattern_pointer(self) -> int | None:
        if self._staged_pattern_offset >= len(self._staged_pattern_pointers):
            return None
        return int(self._staged_pattern_pointers[self._staged_pattern_offset])

    def _discard_staged_inputs(self, *, count_discard: bool = True) -> None:
        if self._staged_pattern_pointers and count_discard:
            self.quantum_input_discard_count += (
                len(self._staged_pattern_pointers) - self._staged_pattern_offset
            )
        self._staged_pattern_pointers = []
        self._staged_pattern_offset = 0

    def consume_result(self) -> tuple[float, ...]:
        if self._last_result is None:
            raise RuntimeError("persistent tick result is unavailable")
        return self._last_result

    @property
    def last_result_from_host_sync(self) -> bool:
        return bool(self._last_result_from_host_sync)

    def consume_surprise_update(self) -> bool:
        pending = self._surprise_update_pending
        self._surprise_update_pending = False
        return pending

    def consume_competitive_surprise(self) -> float | None:
        pending = self._competitive_surprise_pending
        self._competitive_surprise_pending = None
        return pending

    @property
    def owns_competitive_surprise(self) -> bool:
        return bool(self._owns_competitive_surprise)

    @property
    def owns_routing_cache_updates(self) -> bool:
        return bool(
            self.active
            and self._route_vectors is not None
            and self._route_position_by_column is not None
        )

    def report(self) -> dict[str, Any]:
        return {
            "surface": "cuda_graph_persistent_text_tick.v1",
            "active": bool(self.active),
            "fallback_reason": self.fallback_reason,
            "capture_attempted": bool(self.capture_attempted),
            "capture_succeeded": bool(self.capture_succeeded),
            "capture_latency_ms": float(self.capture_latency_ms),
            "capture_count": int(self.capture_count),
            "pre_route_replay_count": int(self.tick_replay_count),
            "tick_replay_count": int(self.tick_replay_count),
            "pre_route_sensory_bypass_count": int(
                self.pre_route_sensory_bypass_count
            ),
            "pre_route_bootstrap_bypass_count": int(
                self.pre_route_bootstrap_bypass_count
            ),
            "replay_count": int(self.replay_count),
            "failure_count": int(self.failure_count),
            "host_truth_sync_count": int(self.host_truth_sync_count),
            "host_truth_skip_count": int(self.host_truth_skip_count),
            "host_truth_sync_interval_tokens": int(
                getattr(
                    self._trainer.config,
                    "cuda_graph_host_truth_sync_interval_tokens",
                    1,
                )
            ),
            "surprise_update_count": int(self.surprise_update_count),
            "host_truth_mirror_update_count": int(
                self.host_truth_mirror_update_count
            ),
            "last_result_from_host_sync": bool(self._last_result_from_host_sync),
            "competitive_surprise_update_count": int(
                self.competitive_surprise_update_count
            ),
            "route_cache_clean_fastpath_count": int(
                self.route_cache_clean_fastpath_count
            ),
            "route_cache_rebuild_check_count": int(
                self.route_cache_rebuild_check_count
            ),
            "route_cache_generation_fastpath_count": int(
                self.route_cache_generation_fastpath_count
            ),
            "route_cache_generation_mismatch_count": int(
                self.route_cache_generation_mismatch_count
            ),
            "route_cache_device_owned": bool(
                self.owns_routing_cache_updates
            ),
            "route_cache_device_update_count": int(
                self.route_cache_device_update_count
            ),
            "consolidation_cache_generation_fastpath_count": int(
                self.consolidation_cache_generation_fastpath_count
            ),
            "consolidation_cache_generation_mismatch_count": int(
                self.consolidation_cache_generation_mismatch_count
            ),
            "consolidation_memory_warm_state_mismatch_count": int(
                self.consolidation_memory_warm_state_mismatch_count
            ),
            "previous_flag_device_owned_count": int(
                self.previous_flag_device_owned_count
            ),
            "learning_rate_device_owned_count": int(
                self.learning_rate_device_owned_count
            ),
            "learning_rate_host_resync_count": int(
                self.learning_rate_host_resync_count
            ),
            "modulator_stage_copy_count": int(self.modulator_stage_copy_count),
            "modulator_stage_skip_count": int(self.modulator_stage_skip_count),
            "quantum_input_staging_enabled": bool(
                getattr(
                    self._trainer.config,
                    "cuda_graph_quantum_input_staging",
                    True,
                )
            ),
            "quantum_input_capacity_tokens": MAX_QUANTUM_INPUT_TOKENS,
            "quantum_input_stage_count": int(self.quantum_input_stage_count),
            "quantum_input_staged_token_count": int(
                self.quantum_input_staged_token_count
            ),
            "quantum_input_reuse_count": int(self.quantum_input_reuse_count),
            "quantum_input_fallback_copy_count": int(
                self.quantum_input_fallback_copy_count
            ),
            "quantum_input_mismatch_count": int(
                self.quantum_input_mismatch_count
            ),
            "quantum_input_discard_count": int(
                self.quantum_input_discard_count
            ),
            "recent_spike_row_device_owned_count": int(
                self.recent_spike_row_device_owned_count
            ),
            "reconstruction_error_source": (
                "fused_route_score_max" if self.active else "retained_dense_scan"
            ),
            "fused_reconstruction_error_active": bool(self.active),
            "fused_reconstruction_error_update_count": int(self.tick_replay_count),
            "graph_names": sorted(self._graphs),
            "pre_route_graph": False,
            "persistent_tick_graph": bool(self._graphs),
            "tensor_device": (
                None
                if self._input_patterns is None
                else str(self._input_patterns.device)
            ),
            "fixed_address_inputs": True,
            "mutates_runtime_state": bool(self.active),
            "owns_neuromodulator_update": bool(self.active),
            "owns_competitive_surprise": bool(
                self.active and self._owns_competitive_surprise
            ),
            "last_graph_name": self._last_graph_name,
        }
