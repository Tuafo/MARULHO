from __future__ import annotations

import gc
import os
import time
from typing import Any

import torch
import torch.nn.functional as F
from marulho.core.columns import _normalize_routing_key
from marulho.core.fused_route_vote_cuda import (
    fused_route_vote_cuda,
    fused_route_vote_kernel_variant,
)
from marulho.core.inplace_column_cuda import inplace_column_transition_cuda
from marulho.core.native_cuda_graph_replay import (
    make_repeated_cuda_graph_exec,
    native_cuda_graph_replay_error,
    replay_repeated_cuda_graph_exec,
)
from marulho.core.native_cuda_graph_sequence import (
    make_conditional_loop_cuda_graph_exec,
    native_cuda_graph_sequence_error,
    replay_conditional_loop_cuda_graph_exec,
)


MAX_QUANTUM_INPUT_TOKENS = 128
PERSISTENT_EXECUTOR_BURST_TOKENS = 8
PERSISTENT_EXECUTOR_SEQUENCE_LOOP_TOKENS = 16
PERSISTENT_EXECUTOR_EVENT_CAPACITY_TOKENS = 32
PERSISTENT_EXECUTOR_ALLOWED_SEQUENCE_LOOP_TOKENS = (8, 16, 32)


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
        self._host_truth_cadence_tick_count = 0
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
        self.burst_replay_count = 0
        self.burst_replayed_token_count = 0
        self.burst_replay_failure_count = 0
        self.burst_event_deferred_count = 0
        self.burst_event_drain_count = 0
        self.burst_event_drained_token_count = 0
        self.burst_event_forced_drain_count = 0
        self.burst_event_slim_result_packet_count = 0
        self.burst_event_strong_result_row_count = 0
        self.burst_event_strong_flag_scan_count = 0
        self.burst_event_no_strong_flag_scan_skip_count = 0
        self.burst_event_slot_reset_count = 0
        self.burst_event_slot_reset_skip_count = 0
        self.route_vote_kernel_variant = "unavailable"
        self.recent_spike_row_device_owned_count = 0
        self.native_burst_replay_enabled = False
        self.native_burst_replay_attempt_count = 0
        self.native_burst_replay_success_count = 0
        self.native_burst_replay_token_count = 0
        self.native_burst_replay_fallback_count = 0
        self.native_burst_replay_failure_count = 0
        self.native_burst_replay_lazy_compile_count = 0
        self.native_burst_replay_lazy_compile_failure_count = 0
        self.native_burst_replay_python_loop_token_count = 0
        self.native_burst_replay_compile_latency_ms = 0.0
        self.native_burst_replay_backend = "python_loop"
        self.native_burst_replay_last_error: str | None = None
        self.native_sequence_loop_enabled = False
        self.native_sequence_loop_attempt_count = 0
        self.native_sequence_loop_success_count = 0
        self.native_sequence_loop_token_count = 0
        self.native_sequence_loop_fallback_count = 0
        self.native_sequence_loop_failure_count = 0
        self.native_sequence_loop_lazy_compile_count = 0
        self.native_sequence_loop_lazy_compile_failure_count = 0
        self.native_sequence_loop_compile_latency_ms = 0.0
        self.native_sequence_loop_backend = "disabled"
        self.native_sequence_loop_last_error: str | None = None
        self._native_burst_token_capacity = self._resolve_native_burst_token_capacity()
        self._sequence_loop_token_capacity = self._resolve_sequence_loop_token_capacity()
        self._burst_token_capacity = (
            self._sequence_loop_token_capacity
            if self._native_sequence_loop_requested()
            else self._native_burst_token_capacity
        )
        self._graphs: dict[str, torch.cuda.CUDAGraph] = {}
        self._graph_outputs: dict[str, dict[str, torch.Tensor]] = {}
        self._burst_graphs: dict[str, torch.cuda.CUDAGraph] = {}
        self._burst_graph_outputs: dict[str, dict[str, torch.Tensor]] = {}
        self._native_burst_graph_execs: dict[tuple[str, int], Any] = {}
        self._native_sequence_graph_execs: dict[tuple[str, int], Any] = {}
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
        self._predictive_step_counter: torch.Tensor | None = None
        self._predictive_step_count_mirror: int | None = None
        self._input_patterns: torch.Tensor | None = None
        self._input_slot: torch.Tensor | None = None
        self._input_slot_mirror = 0
        self._staged_pattern_pointers: list[int] = []
        self._staged_pattern_offset = 0
        self._neuromodulator_state: torch.Tensor | None = None
        self._result: torch.Tensor | None = None
        self._burst_result_ring: torch.Tensor | None = None
        self._burst_routing_ring: torch.Tensor | None = None
        self._burst_assembly_ring: torch.Tensor | None = None
        self._burst_strong_flags: torch.Tensor | None = None
        self._burst_strong_count: torch.Tensor | None = None
        self._burst_strong_count_mirror = 0
        self._burst_slot: torch.Tensor | None = None
        self._burst_pending_event_count = 0
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
        *,
        write_burst_event: bool = False,
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
        self._launch_transition(
            candidates,
            routing_key=routing_key,
            write_burst_event=write_burst_event,
        )
        self._learning_rate_update_count.add_(1.0)
        assert self._neuromodulator_state is not None
        assert self._result is not None
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

    def _burst_tick_ops(
        self,
        candidates: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self._tick_ops(candidates, write_burst_event=True)

    def _capture_candidate_sets(self) -> tuple[tuple[str, torch.Tensor], ...]:
        threshold = int(self._trainer.config.candidate_homeostasis_start_tokens)
        if int(self._trainer.token_count) >= threshold:
            return (("candidate_subset", self._runtime._route_candidates),)
        return (
            ("all_columns", self._runtime._all_columns),
            ("candidate_subset", self._runtime._route_candidates),
        )

    def _capture(self) -> None:
        trainer = self._trainer
        runtime = self._runtime
        comp = trainer.model.competitive
        pred = trainer.model.predictive
        device = trainer.model.device
        self.capture_attempted = True
        started = time.perf_counter_ns()
        try:
            # CUDAGraph finalization during a new capture poisons CUDA's
            # process-wide capture state. Retire older trainer graphs first.
            gc.collect()
            if device.type != "cuda":
                raise RuntimeError("cuda_graph_requires_cuda")
            vectors, ids = trainer.model.routing_index.routing_tensor_cache()
            route_generation_fn = getattr(
                trainer.model.routing_index,
                "routing_tensor_cache_generation",
                None,
            )
            if runtime._route_scores is None or runtime._route_candidates is None:
                raise RuntimeError("cuda_graph_requires_route_workspaces")
            self._route_vectors = vectors
            self._route_ids = ids
            self.route_vote_kernel_variant = fused_route_vote_kernel_variant(
                vectors,
                runtime._route_candidates,
                runtime._route_bank_positions
                if runtime.route_candidate_bank_enabled
                else None,
            )
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
            runtime._sync_state_transition_step_tensors_from_core()
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
            self._predictive_step_counter = torch.tensor(
                int(pred.predictive_step_count),
                dtype=torch.long,
                device=device,
            )
            self._predictive_step_count_mirror = int(pred.predictive_step_count)
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
            self._burst_result_ring = torch.empty(
                (
                    PERSISTENT_EXECUTOR_EVENT_CAPACITY_TOKENS,
                    int(self._result.numel()),
                ),
                dtype=torch.float32,
                device=device,
            )
            self._burst_routing_ring = torch.empty(
                (PERSISTENT_EXECUTOR_EVENT_CAPACITY_TOKENS, comp.column_dim),
                dtype=torch.float32,
                device=device,
            )
            self._burst_assembly_ring = torch.empty(
                (PERSISTENT_EXECUTOR_EVENT_CAPACITY_TOKENS, comp.n_columns),
                dtype=torch.float32,
                device=device,
            )
            self._burst_strong_flags = torch.zeros(
                PERSISTENT_EXECUTOR_EVENT_CAPACITY_TOKENS,
                dtype=torch.bool,
                device=device,
            )
            self._burst_strong_count = torch.zeros(
                (),
                dtype=torch.int32,
                device=device,
            )
            self._burst_strong_count_mirror = 0
            self._burst_slot = torch.zeros(
                (),
                dtype=torch.long,
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
                comp.steps_since_win_last_update_step,
                runtime._state_transition_step_counter,
                runtime._state_transition_all_materialized_step,
                comp.recent_spike_window,
                comp.recent_spike_window_active_ids,
                pred.location,
                pred.velocity,
                pred._prediction_weights,
                pred.prediction_error,
                pred.prediction_failure_streak,
                pred.confidence,
                pred.predictive_last_update_step,
                runtime._assembly,
                runtime._assembly_active_winner,
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
                self._predictive_step_counter,
                self._neuromodulator_state,
                self._result,
                self._burst_result_ring,
                self._burst_routing_ring,
                self._burst_assembly_ring,
                self._burst_strong_flags,
                self._burst_strong_count,
                self._burst_slot,
                self._route_vectors,
                self._input_slot,
                runtime._route_sleep_filter_control,
                runtime._route_sleep_filter_state,
            )
            snapshots = tuple(tensor.clone() for tensor in mutable)
            stream = torch.cuda.Stream(device=device)
            for name, candidates in self._capture_candidate_sets():
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
                self._burst_tick_ops(candidates)
                torch.cuda.synchronize(device)
                for tensor, snapshot in zip(mutable, snapshots):
                    tensor.copy_(snapshot)
                torch.cuda.synchronize(device)
                burst_graph = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(burst_graph, stream=stream):
                    burst_outputs = self._burst_tick_ops(candidates)
                torch.cuda.synchronize(device)
                burst_graph.instantiate()
                self._burst_graphs[name] = burst_graph
                self._burst_graph_outputs[name] = burst_outputs
                self.capture_count += 1
            for tensor, snapshot in zip(mutable, snapshots):
                tensor.copy_(snapshot)
            torch.cuda.synchronize(device)
            self._warm_native_burst_replay()
            self.capture_succeeded = True
            self.active = True
        except Exception as exc:
            self.fallback_reason = (
                f"cuda_graph_capture_failed:{type(exc).__name__}:{exc}"
            )
            self._graphs.clear()
            self._graph_outputs.clear()
            self._burst_graphs.clear()
            self._burst_graph_outputs.clear()
            self._native_burst_graph_execs.clear()
            self._native_sequence_graph_execs.clear()
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
            route_positions=(
                runtime._route_bank_positions
                if runtime.route_candidate_bank_enabled
                else None
            ),
            steps_since_win=comp.steps_since_win,
            steps_since_win_last_update_step=comp.steps_since_win_last_update_step,
            state_transition_step_counter=runtime._state_transition_step_counter,
            state_transition_all_materialized_step=(
                runtime._state_transition_all_materialized_step
            ),
            prototypes=comp.prototypes,
            thresholds=comp.thresholds,
            prediction_location=pred.location,
            memory_pressure=trainer.model.column_metabolism.memory_pressure,
            previous_winner=runtime._previous_winner,
            route_filter_control=runtime._route_sleep_filter_control,
            route_filter_state_out=runtime._route_sleep_filter_state,
            scores_out=runtime._route_scores,
            candidates_out=runtime._route_candidates,
            winner_out=runtime._winner,
            strength_out=runtime._strength,
            competition_had_positive=runtime._competition_had_positive,
            reconstruction_error_out=reconstruction_error_out,
        )

    def _replay_burst_graph(self, graph_name: str, token_count: int) -> None:
        graph = self._burst_graphs[graph_name]
        if not self._native_burst_replay_requested():
            self.native_burst_replay_backend = "python_loop_disabled"
            for _ in range(token_count):
                graph.replay()
            return

        self.native_burst_replay_attempt_count += 1
        if (
            self._native_sequence_loop_requested()
            and self.native_sequence_loop_enabled
            and self._replay_native_sequence_loop(graph_name, token_count, graph)
        ):
            return

        if not self.native_burst_replay_enabled:
            started = time.perf_counter_ns()
            load_error = native_cuda_graph_replay_error()
            self.native_burst_replay_compile_latency_ms += (
                time.perf_counter_ns() - started
            ) / 1e6
            if load_error is not None:
                self.native_burst_replay_fallback_count += 1
                self.native_burst_replay_last_error = load_error
                self.native_burst_replay_backend = (
                    "python_loop_after_native_unavailable"
                )
                for _ in range(token_count):
                    graph.replay()
                return

        repeated_graph_exec = self._native_burst_graph_execs.get(
            (graph_name, token_count)
        )
        if repeated_graph_exec is None:
            repeated_graph_exec = self._ensure_native_burst_graph_exec(
                graph_name,
                token_count,
                graph,
            )
        if repeated_graph_exec is None:
            self.native_burst_replay_fallback_count += 1
            self.native_burst_replay_python_loop_token_count += token_count
            if token_count != self._burst_token_capacity:
                self.native_burst_replay_backend = "python_loop_partial_disabled"
            else:
                self.native_burst_replay_backend = "python_loop_no_parent_graph"
            for _ in range(token_count):
                graph.replay()
            return
        try:
            replay_repeated_cuda_graph_exec(repeated_graph_exec)
            self.native_burst_replay_backend = "native_repeated_child_graph"
        except Exception as exc:
            self.native_burst_replay_failure_count += 1
            self.native_burst_replay_last_error = f"{type(exc).__name__}: {exc}"
            self.native_burst_replay_backend = "native_repeated_child_graph_failed"
            raise
        self.native_burst_replay_enabled = True
        self.native_burst_replay_success_count += 1
        self.native_burst_replay_token_count += token_count
        self.native_burst_replay_last_error = None

    def _replay_native_sequence_loop(
        self,
        graph_name: str,
        token_count: int,
        graph: torch.cuda.CUDAGraph,
    ) -> bool:
        self.native_sequence_loop_attempt_count += 1
        sequence_graph_exec = self._native_sequence_graph_execs.get(
            (graph_name, token_count)
        )
        if sequence_graph_exec is None:
            sequence_graph_exec = self._ensure_native_sequence_loop_graph_exec(
                graph_name,
                token_count,
                graph,
            )
        if sequence_graph_exec is None:
            self.native_sequence_loop_fallback_count += 1
            return False
        try:
            replay_conditional_loop_cuda_graph_exec(sequence_graph_exec)
            self.native_burst_replay_backend = "cuda_graph_conditional_while"
            self.native_sequence_loop_backend = "cuda_graph_conditional_while"
        except Exception as exc:
            self.native_burst_replay_failure_count += 1
            self.native_sequence_loop_failure_count += 1
            self.native_burst_replay_last_error = f"{type(exc).__name__}: {exc}"
            self.native_sequence_loop_last_error = (
                f"{type(exc).__name__}: {exc}"
            )
            self.native_burst_replay_backend = (
                "cuda_graph_conditional_while_failed"
            )
            self.native_sequence_loop_backend = (
                "cuda_graph_conditional_while_failed"
            )
            raise
        self.native_burst_replay_enabled = True
        self.native_burst_replay_success_count += 1
        self.native_burst_replay_token_count += token_count
        self.native_burst_replay_last_error = None
        self.native_sequence_loop_success_count += 1
        self.native_sequence_loop_token_count += token_count
        self.native_sequence_loop_last_error = None
        return True

    def _ensure_native_burst_graph_exec(
        self,
        graph_name: str,
        token_count: int,
        graph: torch.cuda.CUDAGraph,
    ) -> Any | None:
        key = (graph_name, token_count)
        repeated_graph_exec = self._native_burst_graph_execs.get(key)
        if repeated_graph_exec is not None:
            return repeated_graph_exec
        if token_count != self._native_burst_token_capacity:
            return None
        build_started = time.perf_counter_ns()
        try:
            repeated_graph_exec = make_repeated_cuda_graph_exec(
                graph,
                token_count,
            )
        except Exception as exc:
            self.native_burst_replay_compile_latency_ms += (
                time.perf_counter_ns() - build_started
            ) / 1e6
            self.native_burst_replay_lazy_compile_failure_count += 1
            self.native_burst_replay_last_error = f"{type(exc).__name__}: {exc}"
            self.native_burst_replay_backend = (
                "python_loop_after_native_partial_parent_graph_unavailable"
            )
            return None
        self.native_burst_replay_compile_latency_ms += (
            time.perf_counter_ns() - build_started
        ) / 1e6
        self._native_burst_graph_execs[key] = repeated_graph_exec
        if token_count != self._native_burst_token_capacity:
            self.native_burst_replay_lazy_compile_count += 1
            self.native_burst_replay_backend = (
                "native_repeated_child_graph_partial_ready"
            )
        return repeated_graph_exec

    def _ensure_native_sequence_loop_graph_exec(
        self,
        graph_name: str,
        token_count: int,
        graph: torch.cuda.CUDAGraph,
    ) -> Any | None:
        key = (graph_name, token_count)
        sequence_graph_exec = self._native_sequence_graph_execs.get(key)
        if sequence_graph_exec is not None:
            return sequence_graph_exec
        if token_count != self._sequence_loop_token_capacity:
            return None
        build_started = time.perf_counter_ns()
        try:
            sequence_graph_exec = make_conditional_loop_cuda_graph_exec(
                graph,
                token_count,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter_ns() - build_started) / 1e6
            self.native_burst_replay_compile_latency_ms += latency_ms
            self.native_sequence_loop_compile_latency_ms += latency_ms
            self.native_sequence_loop_lazy_compile_failure_count += 1
            self.native_sequence_loop_last_error = (
                f"{type(exc).__name__}: {exc}"
            )
            self.native_sequence_loop_backend = (
                "native_repeated_child_graph_after_conditional_partial_unavailable"
            )
            return None
        latency_ms = (time.perf_counter_ns() - build_started) / 1e6
        self.native_burst_replay_compile_latency_ms += latency_ms
        self.native_sequence_loop_compile_latency_ms += latency_ms
        self._native_sequence_graph_execs[key] = sequence_graph_exec
        if token_count != self._sequence_loop_token_capacity:
            self.native_sequence_loop_lazy_compile_count += 1
            self.native_sequence_loop_backend = (
                "cuda_graph_conditional_while_partial_ready"
            )
        return sequence_graph_exec

    def _resolve_native_burst_token_capacity(self) -> int:
        return PERSISTENT_EXECUTOR_BURST_TOKENS

    def _resolve_sequence_loop_token_capacity(self) -> int:
        raw = os.environ.get("MARULHO_CUDA_GRAPH_SEQUENCE_LOOP_TOKENS")
        if raw is None:
            raw = str(
                getattr(
                    self._trainer.config,
                    "cuda_graph_sequence_loop_tokens",
                    PERSISTENT_EXECUTOR_SEQUENCE_LOOP_TOKENS,
                )
            )
        try:
            value = int(str(raw).strip())
        except ValueError as exc:
            raise ValueError(
                "cuda_graph_sequence_loop_tokens must be one of "
                f"{PERSISTENT_EXECUTOR_ALLOWED_SEQUENCE_LOOP_TOKENS}"
            ) from exc
        if value not in PERSISTENT_EXECUTOR_ALLOWED_SEQUENCE_LOOP_TOKENS:
            raise ValueError(
                "cuda_graph_sequence_loop_tokens must be one of "
                f"{PERSISTENT_EXECUTOR_ALLOWED_SEQUENCE_LOOP_TOKENS}"
            )
        if value > PERSISTENT_EXECUTOR_EVENT_CAPACITY_TOKENS:
            raise ValueError(
                "cuda_graph_sequence_loop_tokens must not exceed burst event capacity"
            )
        return value

    def _native_burst_replay_requested(self) -> bool:
        env = os.environ.get("MARULHO_CUDA_GRAPH_NATIVE_BURST_REPLAY")
        if env is not None:
            return env.strip().lower() not in {"0", "false", "no", "off"}
        return bool(
            getattr(
                self._trainer.config,
                "cuda_graph_native_burst_replay",
                True,
            )
        )

    def _native_sequence_executor_mode(self) -> str:
        env = os.environ.get("MARULHO_CUDA_GRAPH_SEQUENCE_EXECUTOR")
        if env is not None:
            raw = env
        else:
            raw = str(
                getattr(
                    self._trainer.config,
                    "cuda_graph_sequence_executor",
                    "native_repeated_child_graph",
                )
            )
        value = raw.strip().lower().replace("-", "_")
        if value in {
            "conditional_while",
            "cuda_graph_conditional_while",
            "conditional_loop",
        }:
            return "cuda_graph_conditional_while"
        if value in {
            "",
            "0",
            "false",
            "no",
            "off",
            "default",
            "native8",
            "repeated_child",
            "native_repeated_child_graph",
        }:
            return "native_repeated_child_graph"
        return f"unknown:{value}"

    def _native_sequence_loop_requested(self) -> bool:
        return (
            self._native_sequence_executor_mode()
            == "cuda_graph_conditional_while"
        )

    def _warm_native_burst_replay(self) -> None:
        self._native_burst_graph_execs.clear()
        self._native_sequence_graph_execs.clear()
        self.native_sequence_loop_enabled = False
        if not self._native_burst_replay_requested():
            self.native_burst_replay_backend = "python_loop_disabled"
            self.native_sequence_loop_backend = "disabled"
            return
        if (
            self._native_sequence_loop_requested()
            and self._warm_native_sequence_loop()
        ):
            return
        self._burst_token_capacity = self._native_burst_token_capacity
        self._warm_repeated_child_burst_replay()

    def _warm_repeated_child_burst_replay(self) -> None:
        started = time.perf_counter_ns()
        load_error = native_cuda_graph_replay_error()
        self.native_burst_replay_compile_latency_ms += (
            time.perf_counter_ns() - started
        ) / 1e6
        if load_error is not None:
            self.native_burst_replay_enabled = False
            self.native_burst_replay_last_error = load_error
            self.native_burst_replay_backend = "python_loop_after_native_unavailable"
            return
        build_started = time.perf_counter_ns()
        try:
            for graph_name, graph in self._burst_graphs.items():
                self._native_burst_graph_execs[
                    (graph_name, self._native_burst_token_capacity)
                ] = make_repeated_cuda_graph_exec(
                    graph,
                    self._native_burst_token_capacity,
                )
        except Exception as exc:
            self.native_burst_replay_compile_latency_ms += (
                time.perf_counter_ns() - build_started
            ) / 1e6
            self._native_burst_graph_execs.clear()
            self.native_burst_replay_enabled = False
            self.native_burst_replay_last_error = f"{type(exc).__name__}: {exc}"
            self.native_burst_replay_backend = (
                "python_loop_after_native_parent_graph_unavailable"
            )
            return
        self.native_burst_replay_compile_latency_ms += (
            time.perf_counter_ns() - build_started
        ) / 1e6
        self.native_burst_replay_enabled = True
        self.native_burst_replay_backend = "native_repeated_child_graph_ready"
        self.native_burst_replay_last_error = None

    def _warm_native_sequence_loop(self) -> bool:
        started = time.perf_counter_ns()
        load_error = native_cuda_graph_sequence_error()
        latency_ms = (time.perf_counter_ns() - started) / 1e6
        self.native_burst_replay_compile_latency_ms += latency_ms
        self.native_sequence_loop_compile_latency_ms += latency_ms
        if load_error is not None:
            self.native_sequence_loop_enabled = False
            self.native_sequence_loop_last_error = load_error
            self.native_sequence_loop_backend = (
                "native_repeated_child_graph_after_conditional_unavailable"
            )
            self.native_sequence_loop_fallback_count += 1
            return False
        build_started = time.perf_counter_ns()
        try:
            for graph_name, graph in self._burst_graphs.items():
                self._native_sequence_graph_execs[
                    (graph_name, self._sequence_loop_token_capacity)
                ] = make_conditional_loop_cuda_graph_exec(
                    graph,
                    self._sequence_loop_token_capacity,
                )
        except Exception as exc:
            latency_ms = (time.perf_counter_ns() - build_started) / 1e6
            self.native_burst_replay_compile_latency_ms += latency_ms
            self.native_sequence_loop_compile_latency_ms += latency_ms
            self._native_sequence_graph_execs.clear()
            self.native_sequence_loop_enabled = False
            self.native_sequence_loop_last_error = (
                f"{type(exc).__name__}: {exc}"
            )
            self.native_sequence_loop_backend = (
                "native_repeated_child_graph_after_conditional_parent_unavailable"
            )
            self.native_sequence_loop_fallback_count += 1
            return False
        latency_ms = (time.perf_counter_ns() - build_started) / 1e6
        self.native_burst_replay_compile_latency_ms += latency_ms
        self.native_sequence_loop_compile_latency_ms += latency_ms
        self.native_burst_replay_enabled = True
        self.native_burst_replay_backend = "cuda_graph_conditional_while_ready"
        self.native_burst_replay_last_error = None
        self.native_sequence_loop_enabled = True
        self.native_sequence_loop_backend = "cuda_graph_conditional_while_ready"
        self.native_sequence_loop_last_error = None
        self._burst_token_capacity = self._sequence_loop_token_capacity
        return True

    def _launch_transition(
        self,
        candidates: torch.Tensor,
        *,
        routing_key: torch.Tensor,
        write_burst_event: bool = False,
    ) -> None:
        trainer = self._trainer
        runtime = self._runtime
        comp = trainer.model.competitive
        pred = trainer.model.predictive
        assert self._previous_routing_key is not None
        assert self._parameters is not None
        assert self._neuromodulator_state is not None
        assert self._result is not None
        assert self._route_vectors is not None
        assert self._route_position_by_column is not None
        assert self._consolidation is not None
        assert self._predictive_step_counter is not None
        if write_burst_event:
            assert self._burst_result_ring is not None
            assert self._burst_routing_ring is not None
            assert self._burst_assembly_ring is not None
            assert self._burst_strong_flags is not None
            assert self._burst_strong_count is not None
            assert self._burst_slot is not None
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
                runtime._state_transition_step_counter
            ),
            state_transition_all_materialized_step=(
                runtime._state_transition_all_materialized_step
            ),
            location=pred.location,
            location_velocity=pred.velocity,
            prediction_weights=pred._prediction_weights,
            prediction_error=pred.prediction_error,
            prediction_failure_streak=pred.prediction_failure_streak,
            confidence=pred.confidence,
            recent_spike_window=comp.recent_spike_window,
            recent_spike_window_active_ids=comp.recent_spike_window_active_ids,
            assembly=runtime._assembly,
            assembly_active_winner=runtime._assembly_active_winner,
            prediction_boost_out=runtime._prediction_boost,
            effective_modulator_out=runtime._effective_modulator,
            result_out=self._result,
            neuromodulator_state=self._neuromodulator_state,
            burst_result_ring=(
                self._burst_result_ring if write_burst_event else None
            ),
            burst_routing_ring=(
                self._burst_routing_ring if write_burst_event else None
            ),
            burst_assembly_ring=(
                self._burst_assembly_ring if write_burst_event else None
            ),
            burst_strong_flags=(
                self._burst_strong_flags if write_burst_event else None
            ),
            burst_strong_count=(
                self._burst_strong_count if write_burst_event else None
            ),
            burst_slot=self._burst_slot if write_burst_event else None,
            strong_threshold=float(
                self._trainer.config.slow_memory_archive_strong_capture_threshold
            ),
            routing_key=routing_key,
            previous_routing_key=self._previous_routing_key,
            winners=runtime._winner,
            candidates=candidates,
            consolidation=self._consolidation,
            predictive_candidates=(
                candidates
                if (
                    runtime.candidate_predictive_transition_active
                    and int(candidates.numel()) < int(comp.n_columns)
                )
                else None
            ),
            predictive_last_update_step=(
                pred.predictive_last_update_step
                if (
                    runtime.candidate_predictive_transition_active
                    and int(candidates.numel()) < int(comp.n_columns)
                )
                else None
            ),
            predictive_step_counter=(
                self._predictive_step_counter
                if (
                    runtime.candidate_predictive_transition_active
                    and int(candidates.numel()) < int(comp.n_columns)
                )
                else None
            ),
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
            index = self._trainer.model.routing_index
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
        assert self._predictive_step_counter is not None
        assert self._predictive_step_count_mirror is not None
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
        self._parameters[4].fill_(1.0 if has_previous else 0.0)
        _profile_mark("cuda_graph_prepare_previous_key")
        if int(comp.update_count) != int(self._learning_rate_update_count_mirror):
            self._learning_rate_update_count.fill_(float(comp.update_count))
            self._learning_rate_update_count_mirror = int(comp.update_count)
            self.learning_rate_host_resync_count += 1
        pred = trainer.model.predictive
        if int(pred.predictive_step_count) != int(self._predictive_step_count_mirror):
            self._predictive_step_counter.fill_(int(pred.predictive_step_count))
            self._predictive_step_count_mirror = int(pred.predictive_step_count)
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
        self._runtime.prepare_route_sleep_filter_control()
        try:
            self._graphs[graph_name].replay()
        except Exception:
            self.failure_count += 1
            raise
        _profile_mark("cuda_graph_prepare_replay")
        self._learning_rate_update_count_mirror += 1
        self._predictive_step_count_mirror += 1
        self._input_slot_mirror = (
            self._input_slot_mirror + 1
        ) % MAX_QUANTUM_INPUT_TOKENS
        self.tick_replay_count += 1
        self.replay_count += 1
        self._host_truth_cadence_tick_count += 1
        self.route_cache_device_update_count += 1
        self._runtime.mark_route_sleep_filter_state_dirty()
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
            or self._host_truth_cadence_tick_count % sync_interval == 0
        )
        if sync_due:
            result = tuple(float(value) for value in outputs["result"].tolist())
            _profile_mark("cuda_graph_prepare_host_truth_sync")
            self.host_truth_sync_count += 1
            self.host_truth_mirror_update_count += 1
            self._last_result = result
            self._last_result_from_host_sync = True
            if self._route_filter_state_sync_due(sync_interval):
                self._runtime.sync_route_sleep_filter_state_from_device()
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
    def can_prestage_input_quantum(self) -> bool:
        return (
            bool(self.active)
            and self._last_result is not None
            and bool(
                getattr(
                    self._trainer.config,
                    "cuda_graph_quantum_input_staging",
                    True,
                )
            )
        )

    def _route_filter_state_sync_due(self, sync_interval: int) -> bool:
        if not self._runtime._route_sleep_filter_state_dirty:
            return False
        if int(sync_interval) <= 1:
            return True
        if self._runtime.route_vote_deep_sleep_filter_state_sync_count <= 0:
            return True
        cadence = max(1, int(sync_interval) * 32)
        return self.replay_count % cadence == 0

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

    @torch.no_grad()
    def replay_staged_text_burst(
        self,
        patterns: list[torch.Tensor],
    ) -> dict[str, Any]:
        """Burst-replay a bounded token run without interleaved host bookkeeping."""

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

        if (
            len(patterns) <= 0
            or len(patterns) > self._burst_token_capacity
        ):
            raise ValueError(
                "persistent executor burst requires between 1 and "
                f"{self._burst_token_capacity} patterns"
            )
        if not self.eligible():
            raise RuntimeError(self.fallback_reason or "cuda_graph_not_active")
        trainer = self._trainer
        start_token = int(trainer.token_count)
        threshold = int(trainer.config.candidate_homeostasis_start_tokens)
        graph_name = "candidate_subset" if start_token >= threshold else "all_columns"
        if start_token < threshold < start_token + len(patterns):
            raise RuntimeError("persistent executor burst crosses routing-mode boundary")
        _profile_mark("text_burst_runtime_eligibility")
        sync_interval = max(
            1,
            int(trainer.config.cuda_graph_host_truth_sync_interval_tokens),
        )
        sync_offsets = [
            offset
            for offset in range(1, len(patterns) + 1)
            if (
                self._host_truth_cadence_tick_count + offset
            )
            % sync_interval
            == 0
        ]
        if sync_offsets not in ([], [len(patterns)]):
            raise RuntimeError(
                "persistent executor burst crosses host-truth boundary"
            )
        if (
            self._burst_pending_event_count + len(patterns)
            > PERSISTENT_EXECUTOR_EVENT_CAPACITY_TOKENS
        ):
            raise RuntimeError("persistent executor event queue capacity exceeded")
        _profile_mark("text_burst_runtime_host_truth_gate")
        if not self._staged_inputs_cover(patterns) and not self.stage_input_quantum(
            patterns
        ):
            raise RuntimeError("persistent executor burst input staging failed")
        _profile_mark("text_burst_runtime_input_stage")

        assert self._parameters is not None
        assert self._parameter_device_prefix is not None
        assert self._parameter_host_prefix is not None
        assert self._host_parameters is not None
        assert self._previous_routing_key is not None
        assert self._learning_rate_update_count is not None
        assert self._learning_rate_update_count_mirror is not None
        assert self._predictive_step_counter is not None
        assert self._predictive_step_count_mirror is not None
        comp = trainer.model.competitive
        has_previous = trainer._prev_routing_key is not None
        if (
            has_previous
            and trainer._prev_routing_key is not None
            and trainer._prev_routing_key.data_ptr()
            != self._previous_routing_key.data_ptr()
        ):
            self._previous_routing_key.copy_(trainer._prev_routing_key)
        self._parameters[4].fill_(1.0 if has_previous else 0.0)
        if int(comp.update_count) != int(self._learning_rate_update_count_mirror):
            self._learning_rate_update_count.fill_(float(comp.update_count))
            self._learning_rate_update_count_mirror = int(comp.update_count)
            self.learning_rate_host_resync_count += 1
        pred = trainer.model.predictive
        if int(pred.predictive_step_count) != int(self._predictive_step_count_mirror):
            self._predictive_step_counter.fill_(int(pred.predictive_step_count))
            self._predictive_step_count_mirror = int(pred.predictive_step_count)
        surprise = trainer.model.surprise
        modulator_revision = int(getattr(surprise, "modulator_revision", 0))
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
        self._runtime.prepare_route_sleep_filter_control()
        _profile_mark("text_burst_runtime_control_state")

        token_count = len(patterns)
        try:
            self._replay_burst_graph(graph_name, token_count)
        except Exception:
            self.burst_replay_failure_count += 1
            raise
        _profile_mark("text_burst_runtime_replay_loop")

        self._consume_staged_inputs(token_count)
        self._learning_rate_update_count_mirror += token_count
        self._predictive_step_count_mirror += token_count
        self._input_slot_mirror = (
            self._input_slot_mirror + token_count
        ) % MAX_QUANTUM_INPUT_TOKENS
        self.quantum_input_reuse_count += token_count
        self.tick_replay_count += token_count
        self.replay_count += token_count
        self._host_truth_cadence_tick_count += token_count
        self.burst_replay_count += 1
        self.burst_replayed_token_count += token_count
        self.route_cache_device_update_count += token_count
        self._runtime.mark_route_sleep_filter_state_dirty()
        self.surprise_update_count += token_count
        self.previous_flag_device_owned_count += token_count
        self.learning_rate_device_owned_count += token_count
        self.recent_spike_row_device_owned_count += token_count
        self._burst_pending_event_count += token_count
        self._last_graph_name = graph_name
        outputs = self._burst_graph_outputs[graph_name]
        trainer._prev_routing_key = self._previous_routing_key
        comp.last_input_pattern = outputs["normalized_input"]
        comp.last_projected_input = outputs["projected_input"]
        comp._cached_proto_sim = None
        comp._cached_raw_drive = None
        comp.last_execution_mode = "candidate_subset_persistent_burst_graph"
        comp.last_scored_column_count = int(self._runtime._route_candidates.numel())
        comp.last_candidate_count = int(self._runtime._route_candidates.numel())
        if self._runtime.route_candidate_bank_enabled:
            route_input_rows = (
                int(self._runtime._route_bank_positions.numel())
                if self._runtime._route_bank_positions is not None
                else int(self._runtime._route_candidates.numel())
            )
            self._runtime._record_route_scoring(
                input_rows=route_input_rows,
                output_candidates=int(self._runtime._route_candidates.numel()),
                candidate_boundary="bounded_route_bank_burst_score_then_filter_select",
                route_input_source="training_owned_route_candidate_bank",
                unbounded_reason=None,
            )
            self._runtime._refresh_route_bank_from_candidates(
                self._runtime._route_candidates,
                reason="bounded_route_bank_burst_refresh",
                validate=False,
            )
        elif self._route_ids is not None:
            self._runtime._record_route_scoring(
                input_rows=int(self._route_ids.numel()),
                output_candidates=int(self._runtime._route_candidates.numel()),
                candidate_boundary="exact_full_cache_burst_score_then_filter_select",
                route_input_source="complete_routing_tensor_cache",
                unbounded_reason=(
                    "exact_full_cache_route_scoring_before_bounded_candidate_selection"
                ),
            )
        comp.recent_spike_window_cursor = (
            comp.recent_spike_window_cursor + token_count
        ) % comp.spike_history_window
        comp.recent_spike_window_count = min(
            comp.spike_history_window,
            comp.recent_spike_window_count + token_count,
        )
        comp.update_count += token_count
        trainer.token_count += token_count
        _profile_mark("text_burst_runtime_python_mirrors")
        if sync_offsets or (
            self._burst_pending_event_count
            >= PERSISTENT_EXECUTOR_EVENT_CAPACITY_TOKENS
        ):
            result = {
                **outputs,
                **self._drain_burst_events(forced=False),
            }
            _profile_mark("text_burst_runtime_event_drain")
            return result
        self.burst_event_deferred_count += 1
        self._last_result_from_host_sync = False
        result = {
            **outputs,
            "truth_synced": False,
            "result_rows": [],
            "strong_indices": [],
            "strong_assemblies": [],
            "strong_routing_keys": [],
        }
        _profile_mark("text_burst_runtime_event_defer")
        return result

    def text_burst_token_capacity(self) -> int:
        return int(self._burst_token_capacity)

    @torch.no_grad()
    def _drain_burst_events(self, *, forced: bool) -> dict[str, Any]:
        token_count = int(self._burst_pending_event_count)
        if token_count <= 0:
            return {
                "truth_synced": False,
                "result_rows": [],
                "strong_indices": [],
                "strong_assemblies": [],
                "strong_routing_keys": [],
            }
        assert self._burst_result_ring is not None
        assert self._burst_strong_flags is not None
        assert self._burst_strong_count is not None
        result = tuple(
            float(value)
            for value in self._burst_result_ring[token_count - 1].tolist()
        )
        cumulative_strong_count = int(self._burst_strong_count.item())
        pending_strong_count = max(
            0,
            cumulative_strong_count - int(self._burst_strong_count_mirror),
        )
        self._burst_strong_count_mirror = cumulative_strong_count
        self.host_truth_sync_count += 1
        self.host_truth_skip_count += token_count - 1
        self.host_truth_mirror_update_count += 1
        self.burst_event_drain_count += 1
        self.burst_event_drained_token_count += token_count
        self.burst_event_forced_drain_count += int(forced)
        self.burst_event_slim_result_packet_count += 1
        self._last_result = result
        self._last_result_from_host_sync = True
        sync_interval = max(
            1,
            int(
                getattr(
                    self._trainer.config,
                    "cuda_graph_host_truth_sync_interval_tokens",
                    1,
                )
            ),
        )
        if self._route_filter_state_sync_due(sync_interval):
            self._runtime.sync_route_sleep_filter_state_from_device()
        surprise = self._trainer.model.surprise
        (
            _reconstruction_error,
            predicted_error,
            dopamine,
            acetylcholine,
            norepinephrine,
            serotonin,
            _winner,
            _effective_modulator,
            *optional_competitive_surprise,
        ) = result
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
        self._competitive_surprise_pending = (
            float(optional_competitive_surprise[0])
            if optional_competitive_surprise
            else None
        )
        self.competitive_surprise_update_count += int(
            bool(optional_competitive_surprise)
        )
        if pending_strong_count > 0:
            strong_flags = self._burst_strong_flags[:token_count].tolist()
            self.burst_event_strong_flag_scan_count += 1
            strong_indices = [
                index for index, strong in enumerate(strong_flags) if bool(strong)
            ]
            if len(strong_indices) != pending_strong_count:
                raise RuntimeError(
                    "burst strong-count mirror disagrees with strong flags"
                )
        else:
            self.burst_event_no_strong_flag_scan_skip_count += 1
            strong_indices = []
        strong_result_rows: list[list[float]] = []
        strong_assemblies: list[torch.Tensor] = []
        strong_routing_keys: list[torch.Tensor] = []
        if strong_indices:
            assert self._burst_assembly_ring is not None
            assert self._burst_routing_ring is not None
            index_tensor = torch.tensor(
                strong_indices,
                dtype=torch.long,
                device=self._burst_assembly_ring.device,
            )
            strong_result_rows = [
                [float(value) for value in row]
                for row in self._burst_result_ring.index_select(
                    0,
                    index_tensor,
                ).cpu().tolist()
            ]
            strong_assemblies = list(
                self._burst_assembly_ring.index_select(0, index_tensor).cpu()
            )
            strong_routing_keys = list(
                self._burst_routing_ring.index_select(0, index_tensor).cpu()
            )
            self.burst_event_strong_result_row_count += len(strong_result_rows)
        assert self._burst_slot is not None
        if (
            not forced
            and token_count % PERSISTENT_EXECUTOR_EVENT_CAPACITY_TOKENS == 0
        ):
            self.burst_event_slot_reset_skip_count += 1
        else:
            self._burst_slot.zero_()
            self.burst_event_slot_reset_count += 1
        self._burst_pending_event_count = 0
        return {
            "truth_synced": True,
            "final_result": result,
            "strong_result_rows": strong_result_rows,
            "strong_indices": strong_indices,
            "strong_assemblies": strong_assemblies,
            "strong_routing_keys": strong_routing_keys,
        }

    @torch.no_grad()
    def drain_burst_events(self) -> dict[str, Any]:
        """Materialize bounded pending burst evidence at an explicit slow boundary."""

        return self._drain_burst_events(forced=True)

    def _next_staged_pattern_pointer(self) -> int | None:
        if self._staged_pattern_offset >= len(self._staged_pattern_pointers):
            return None
        return int(self._staged_pattern_pointers[self._staged_pattern_offset])

    def _staged_inputs_cover(self, patterns: list[torch.Tensor]) -> bool:
        if not self._staged_pattern_pointers:
            return False
        end = self._staged_pattern_offset + len(patterns)
        if end > len(self._staged_pattern_pointers):
            return False
        for offset, pattern in enumerate(patterns):
            if (
                self._staged_pattern_pointers[self._staged_pattern_offset + offset]
                != int(pattern.data_ptr())
            ):
                return False
        return True

    def _consume_staged_inputs(self, token_count: int) -> None:
        if not self._staged_pattern_pointers:
            return
        self._staged_pattern_offset += int(token_count)
        if self._staged_pattern_offset >= len(self._staged_pattern_pointers):
            self._discard_staged_inputs(count_discard=False)

    def _discard_staged_inputs(self, *, count_discard: bool = True) -> None:
        if self._staged_pattern_pointers and count_discard:
            self.quantum_input_discard_count += (
                len(self._staged_pattern_pointers) - self._staged_pattern_offset
            )
        self._staged_pattern_pointers = []
        self._staged_pattern_offset = 0

    @torch.no_grad()
    def sync_after_external_transition(
        self,
        *,
        reconstruction_error: float,
        winner_id: int | None,
        effective_modulator: float,
    ) -> None:
        if not self.active:
            return
        trainer = self._trainer
        comp = trainer.model.competitive
        pred = trainer.model.predictive
        if self._learning_rate_update_count is not None:
            self._learning_rate_update_count.fill_(float(comp.update_count))
            self._learning_rate_update_count_mirror = int(comp.update_count)
        if self._predictive_step_counter is not None:
            self._predictive_step_counter.fill_(int(pred.predictive_step_count))
            self._predictive_step_count_mirror = int(pred.predictive_step_count)
        if (
            self._previous_routing_key is not None
            and trainer._prev_routing_key is not None
        ):
            self._previous_routing_key.copy_(trainer._prev_routing_key)
        if self._parameters is not None:
            self._parameters[4].fill_(
                1.0 if trainer._prev_routing_key is not None else 0.0
            )
        surprise = trainer.model.surprise
        if self._neuromodulator_state is not None:
            self._neuromodulator_state.copy_(
                torch.tensor(
                    [
                        float(surprise.predicted_error),
                        float(surprise.dopamine),
                        float(surprise.acetylcholine),
                        float(surprise.norepinephrine),
                        float(surprise.serotonin),
                    ],
                    dtype=self._neuromodulator_state.dtype,
                    device=self._neuromodulator_state.device,
                )
            )
        self._cached_modulator_revision = int(
            getattr(surprise, "modulator_revision", 0)
        )
        self._cached_modulator_value = float(surprise.get_modulator("competitive"))
        if (
            self._host_parameters is not None
            and self._parameter_device_prefix is not None
            and self._parameter_host_prefix is not None
        ):
            self._host_parameters[0] = self._cached_modulator_value
            self._parameter_device_prefix.copy_(
                self._parameter_host_prefix,
                non_blocking=True,
            )
        result = (
            float(reconstruction_error),
            float(surprise.predicted_error),
            float(surprise.dopamine),
            float(surprise.acetylcholine),
            float(surprise.norepinephrine),
            float(surprise.serotonin),
            float(-1 if winner_id is None else int(winner_id)),
            float(effective_modulator),
        )
        self._last_result = result
        self._last_result_from_host_sync = True
        self.host_truth_mirror_update_count += 1
        self._host_truth_cadence_tick_count += 1

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
        sequence_executor = self._native_sequence_executor_mode()
        sequence_loop_requested = sequence_executor == "cuda_graph_conditional_while"
        if sequence_loop_requested and self.native_sequence_loop_enabled:
            sequence_parity_status = "passed_focused_cuda_state_parity"
            sequence_quality_status = (
                "passed_retained_one_tick_graph_body_quality_boundary"
            )
            sequence_parity_passed = True
            sequence_quality_passed = True
        elif sequence_loop_requested:
            sequence_parity_status = "not_exercised_fallback_before_mutation"
            sequence_quality_status = "not_exercised_fallback_before_mutation"
            sequence_parity_passed = False
            sequence_quality_passed = False
        else:
            sequence_parity_status = "not_applicable_repeated_child_executor"
            sequence_quality_status = "not_applicable_repeated_child_executor"
            sequence_parity_passed = False
            sequence_quality_passed = False
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
            "host_truth_cadence_tick_count": int(
                self._host_truth_cadence_tick_count
            ),
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
            "persistent_executor_burst_tokens": (
                int(self._burst_token_capacity)
            ),
            "persistent_executor_default_burst_tokens": (
                PERSISTENT_EXECUTOR_SEQUENCE_LOOP_TOKENS
            ),
            "persistent_executor_repeated_child_burst_tokens": (
                int(self._native_burst_token_capacity)
            ),
            "persistent_executor_default_repeated_child_burst_tokens": (
                PERSISTENT_EXECUTOR_BURST_TOKENS
            ),
            "persistent_executor_sequence_loop_tokens": (
                int(self._sequence_loop_token_capacity)
            ),
            "persistent_executor_default_sequence_loop_tokens": (
                PERSISTENT_EXECUTOR_SEQUENCE_LOOP_TOKENS
            ),
            "persistent_executor_allowed_sequence_loop_tokens": list(
                PERSISTENT_EXECUTOR_ALLOWED_SEQUENCE_LOOP_TOKENS
            ),
            "native_burst_replay_configured": bool(
                self._native_burst_replay_requested()
            ),
            "native_burst_replay_loaded": bool(
                self.native_burst_replay_enabled
            ),
            "native_burst_replay_enabled": bool(
                self._native_burst_replay_requested()
                and self.native_burst_replay_enabled
            ),
            "native_partial_burst_replay_enabled": False,
            "native_burst_replay_backend": self.native_burst_replay_backend,
            "native_burst_replay_parent_graph_count": int(
                len(self._native_burst_graph_execs)
            ),
            "native_burst_replay_parent_graph_token_counts": [
                int(token_count)
                for token_count in sorted(
                    {token_count for _, token_count in self._native_burst_graph_execs}
                )
            ],
            "native_burst_replay_attempt_count": int(
                self.native_burst_replay_attempt_count
            ),
            "native_burst_replay_success_count": int(
                self.native_burst_replay_success_count
            ),
            "native_burst_replay_token_count": int(
                self.native_burst_replay_token_count
            ),
            "native_burst_replay_fallback_count": int(
                self.native_burst_replay_fallback_count
            ),
            "native_burst_replay_failure_count": int(
                self.native_burst_replay_failure_count
            ),
            "native_burst_replay_lazy_compile_count": int(
                self.native_burst_replay_lazy_compile_count
            ),
            "native_burst_replay_lazy_compile_failure_count": int(
                self.native_burst_replay_lazy_compile_failure_count
            ),
            "native_burst_replay_python_loop_token_count": int(
                self.native_burst_replay_python_loop_token_count
            ),
            "native_burst_replay_compile_latency_ms": float(
                self.native_burst_replay_compile_latency_ms
            ),
            "native_burst_replay_last_error": self.native_burst_replay_last_error,
            "native_sequence_executor_requested": sequence_executor,
            "native_sequence_loop_sequential_state_parity_gate_status": (
                sequence_parity_status
            ),
            "native_sequence_loop_sequential_state_parity_gate_passed": (
                sequence_parity_passed
            ),
            "native_sequence_loop_bounded_quality_gate_status": (
                sequence_quality_status
            ),
            "native_sequence_loop_bounded_quality_gate_passed": (
                sequence_quality_passed
            ),
            "native_sequence_loop_loaded": bool(
                self.native_sequence_loop_enabled
            ),
            "native_sequence_loop_backend": self.native_sequence_loop_backend,
            "native_sequence_loop_parent_graph_count": int(
                len(self._native_sequence_graph_execs)
            ),
            "native_sequence_loop_parent_graph_token_counts": [
                int(token_count)
                for token_count in sorted(
                    {
                        token_count
                        for _, token_count in self._native_sequence_graph_execs
                    }
                )
            ],
            "native_sequence_loop_attempt_count": int(
                self.native_sequence_loop_attempt_count
            ),
            "native_sequence_loop_success_count": int(
                self.native_sequence_loop_success_count
            ),
            "native_sequence_loop_token_count": int(
                self.native_sequence_loop_token_count
            ),
            "native_sequence_loop_fallback_count": int(
                self.native_sequence_loop_fallback_count
            ),
            "native_sequence_loop_failure_count": int(
                self.native_sequence_loop_failure_count
            ),
            "native_sequence_loop_lazy_compile_count": int(
                self.native_sequence_loop_lazy_compile_count
            ),
            "native_sequence_loop_lazy_compile_failure_count": int(
                self.native_sequence_loop_lazy_compile_failure_count
            ),
            "native_sequence_loop_compile_latency_ms": float(
                self.native_sequence_loop_compile_latency_ms
            ),
            "native_sequence_loop_last_error": self.native_sequence_loop_last_error,
            "burst_replay_count": int(self.burst_replay_count),
            "burst_replayed_token_count": int(
                self.burst_replayed_token_count
            ),
            "burst_replay_failure_count": int(
                self.burst_replay_failure_count
            ),
            "burst_event_capacity_tokens": (
                PERSISTENT_EXECUTOR_EVENT_CAPACITY_TOKENS
            ),
            "burst_event_pending_tokens": int(self._burst_pending_event_count),
            "burst_event_deferred_count": int(self.burst_event_deferred_count),
            "burst_event_drain_count": int(self.burst_event_drain_count),
            "burst_event_drained_token_count": int(
                self.burst_event_drained_token_count
            ),
            "burst_event_forced_drain_count": int(
                self.burst_event_forced_drain_count
            ),
            "burst_event_slim_result_packet_count": int(
                self.burst_event_slim_result_packet_count
            ),
            "burst_event_strong_result_row_count": int(
                self.burst_event_strong_result_row_count
            ),
            "burst_event_strong_flag_scan_count": int(
                self.burst_event_strong_flag_scan_count
            ),
            "burst_event_no_strong_flag_scan_skip_count": int(
                self.burst_event_no_strong_flag_scan_skip_count
            ),
            "burst_event_strong_count_device_owned": bool(
                self._burst_strong_count is not None
            ),
            "burst_event_strong_count_total": int(
                self._burst_strong_count_mirror
            ),
            "burst_event_slot_reset_count": int(
                self.burst_event_slot_reset_count
            ),
            "burst_event_slot_reset_skip_count": int(
                self.burst_event_slot_reset_skip_count
            ),
            "burst_event_ring_device_owned": bool(
                self._burst_result_ring is not None
                and self._burst_routing_ring is not None
                and self._burst_assembly_ring is not None
                and self._burst_strong_flags is not None
                and self._burst_strong_count is not None
            ),
            "burst_graph_names": sorted(self._burst_graphs),
            "recent_spike_row_device_owned_count": int(
                self.recent_spike_row_device_owned_count
            ),
            "reconstruction_error_source": (
                "fused_route_score_max" if self.active else "retained_dense_scan"
            ),
            "route_vote_kernel_variant": self.route_vote_kernel_variant,
            "route_vote_deep_sleep_filter": (
                self._runtime.route_sleep_filter_snapshot()
            ),
            "fused_reconstruction_error_active": bool(self.active),
            "fused_reconstruction_error_update_count": int(self.tick_replay_count),
            "graph_names": sorted(self._graphs),
            "capture_graph_policy": (
                "candidate_subset_only_after_homeostasis_gate"
                if "all_columns" not in self._graphs
                else "includes_all_columns_until_homeostasis_gate"
            ),
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
