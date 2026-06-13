from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F

from marulho.core.columns import (
    _normalize_positive_vector,
    _normalize_routing_key,
)
from marulho.core.fused_route_vote_cuda import fused_route_vote_cuda
from marulho.core.inplace_column_cuda import inplace_column_transition_cuda


class CudaGraphRouteTransition:
    """Capture the exact text route/vote and in-place transition as one replay."""

    def __init__(self, trainer: Any, runtime: Any) -> None:
        self._trainer = trainer
        self._runtime = runtime
        self.active = False
        self.fallback_reason: str | None = None
        self.capture_attempted = False
        self.capture_succeeded = False
        self.capture_latency_ms = 0.0
        self.capture_count = 0
        self.pre_route_replay_count = 0
        self.pre_route_sensory_bypass_count = 0
        self.replay_count = 0
        self.failure_count = 0
        self._graphs: dict[str, torch.cuda.CUDAGraph] = {}
        self._pre_route_graph: torch.cuda.CUDAGraph | None = None
        self._route_vectors: torch.Tensor | None = None
        self._route_ids: torch.Tensor | None = None
        self._consolidation: torch.Tensor | None = None
        self._routing_key: torch.Tensor | None = None
        self._previous_routing_key: torch.Tensor | None = None
        self._parameters: torch.Tensor | None = None
        self._host_parameters: torch.Tensor | None = None
        self._input_pattern: torch.Tensor | None = None
        self._normalized_input: torch.Tensor | None = None
        self._projected_input: torch.Tensor | None = None
        self._prepared_routing_key: torch.Tensor | None = None
        self._reconstruction_error: torch.Tensor | None = None
        self._capture()

    def _pre_route_ops(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        trainer = self._trainer
        comp = trainer.model.competitive
        assert self._input_pattern is not None
        x = self._input_pattern
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
        reconstruction_key = _normalize_positive_vector(
            routing_key.unsqueeze(0),
            dim=1,
        ).squeeze(0)
        reconstruction_error = torch.clamp(
            1.0 - torch.mv(
                comp.prototypes,
                reconstruction_key,
            ).max(),
            min=0.0,
        )
        return (
            normalized_input,
            projected,
            routing_key,
            reconstruction_error,
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
            if device.type != "cuda":
                raise RuntimeError("cuda_graph_requires_cuda")
            vectors, ids = trainer.model.hnsw_index.routing_tensor_cache()
            if runtime._route_scores is None or runtime._route_candidates is None:
                raise RuntimeError("cuda_graph_requires_route_workspaces")
            self._route_vectors = vectors
            self._route_ids = ids
            self._input_pattern = torch.empty(
                trainer.config.input_dim,
                device=device,
            )
            (
                self._normalized_input,
                self._projected_input,
                self._prepared_routing_key,
                self._reconstruction_error,
            ) = self._pre_route_ops()
            torch.cuda.synchronize(device)
            pre_route_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(pre_route_graph):
                (
                    self._normalized_input,
                    self._projected_input,
                    self._prepared_routing_key,
                    self._reconstruction_error,
                ) = self._pre_route_ops()
            torch.cuda.synchronize(device)
            self._pre_route_graph = pre_route_graph
            self._routing_key = torch.empty(comp.column_dim, device=device)
            self._previous_routing_key = torch.zeros(
                comp.column_dim,
                device=device,
            )
            has_previous = trainer._prev_routing_key is not None
            if has_previous:
                self._previous_routing_key.copy_(trainer._prev_routing_key)
            self._parameters = torch.empty(5, device=device)
            self._host_parameters = torch.empty(
                5,
                dtype=torch.float32,
                pin_memory=True,
            )
            self._host_parameters.zero_()
            self._host_parameters[4] = float(has_previous)
            self._parameters.copy_(self._host_parameters, non_blocking=True)
            self._consolidation = (
                trainer.model.memory_store.bucket_consolidation_tensor(
                    comp.n_columns,
                    device=device,
                )
                if trainer.memory_warm_started
                else runtime._zero_consolidation
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
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    self._launch(candidates)
                torch.cuda.synchronize(device)
                self._graphs[name] = graph
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
            self._pre_route_graph = None
        finally:
            self.capture_latency_ms = (
                time.perf_counter_ns() - started
            ) / 1e6

    def _launch(self, candidates: torch.Tensor) -> None:
        trainer = self._trainer
        runtime = self._runtime
        comp = trainer.model.competitive
        pred = trainer.model.predictive
        assert self._routing_key is not None
        assert self._previous_routing_key is not None
        assert self._parameters is not None
        assert self._route_vectors is not None
        assert self._route_ids is not None
        assert self._consolidation is not None
        assert runtime._route_scores is not None
        assert runtime._route_candidates is not None
        fused_route_vote_cuda(
            routing_key=self._routing_key,
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
        )
        inplace_column_transition_cuda(
            prototypes=comp.prototypes,
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
            routing_key=self._routing_key,
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

    def eligible(self) -> bool:
        if not self.active or self._pre_route_graph is None:
            return False
        vectors, ids = self._trainer.model.hnsw_index.routing_tensor_cache()
        consolidation = (
            self._trainer.model.memory_store.bucket_consolidation_tensor(
                self._trainer.model.competitive.n_columns,
                device=self._trainer.model.device,
            )
            if self._trainer.memory_warm_started
            else self._runtime._zero_consolidation
        )
        if (
            self._route_vectors is None
            or self._route_ids is None
            or self._consolidation is None
            or vectors.data_ptr() != self._route_vectors.data_ptr()
            or ids.data_ptr() != self._route_ids.data_ptr()
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
    ) -> tuple[torch.Tensor, float] | None:
        if sensory_tick:
            self.pre_route_sensory_bypass_count += 1
            return None
        if not self.eligible():
            raise RuntimeError(self.fallback_reason or "cuda_graph_not_active")
        assert self._input_pattern is not None
        assert self._normalized_input is not None
        assert self._projected_input is not None
        assert self._prepared_routing_key is not None
        assert self._reconstruction_error is not None
        assert self._pre_route_graph is not None
        self._input_pattern.copy_(pattern)
        self._pre_route_graph.replay()
        self.pre_route_replay_count += 1
        comp = self._trainer.model.competitive
        comp.last_input_pattern = self._normalized_input
        comp.last_projected_input = self._projected_input
        comp._cached_proto_sim = None
        comp._cached_raw_drive = None
        comp.last_execution_mode = "candidate_routing_pending_cuda_graph"
        comp.last_scored_column_count = 0
        comp.last_candidate_count = 0
        return (
            self._prepared_routing_key,
            float(self._reconstruction_error.item()),
        )

    def replay(
        self,
        routing_key: torch.Tensor,
        *,
        base_modulator: float,
        dopamine: float,
        serotonin: float,
        learning_rate: float,
        candidate_homeostasis: bool,
        recent_spike_row: int,
    ) -> None:
        if not self.eligible():
            raise RuntimeError(self.fallback_reason or "cuda_graph_not_active")
        assert self._routing_key is not None
        assert self._previous_routing_key is not None
        assert self._parameters is not None
        assert self._host_parameters is not None
        if (
            self._trainer._prev_routing_key is not None
            and self._trainer._prev_routing_key.data_ptr()
            != self._previous_routing_key.data_ptr()
        ):
            self._previous_routing_key.copy_(
                self._trainer._prev_routing_key,
            )
            self._host_parameters[4] = 1.0
        self._routing_key.copy_(routing_key)
        self._host_parameters[0] = float(base_modulator)
        self._host_parameters[1] = float(dopamine)
        self._host_parameters[2] = float(serotonin)
        self._host_parameters[3] = float(learning_rate)
        self._parameters.copy_(self._host_parameters, non_blocking=True)
        self._runtime._recent_spike_row.fill_(int(recent_spike_row))
        graph_name = (
            "candidate_subset"
            if candidate_homeostasis
            else "all_columns"
        )
        try:
            self._graphs[graph_name].replay()
        except Exception:
            self.failure_count += 1
            raise
        self._host_parameters[4] = 1.0
        self.replay_count += 1
        self._trainer._prev_routing_key = self._previous_routing_key

    def report(self) -> dict[str, Any]:
        return {
            "surface": "cuda_graph_route_transition.v1",
            "active": bool(self.active),
            "fallback_reason": self.fallback_reason,
            "capture_attempted": bool(self.capture_attempted),
            "capture_succeeded": bool(self.capture_succeeded),
            "capture_latency_ms": float(self.capture_latency_ms),
            "capture_count": int(self.capture_count),
            "pre_route_replay_count": int(self.pre_route_replay_count),
            "pre_route_sensory_bypass_count": int(
                self.pre_route_sensory_bypass_count
            ),
            "replay_count": int(self.replay_count),
            "failure_count": int(self.failure_count),
            "graph_names": sorted(self._graphs),
            "pre_route_graph": self._pre_route_graph is not None,
            "tensor_device": (
                None
                if self._routing_key is None
                else str(self._routing_key.device)
            ),
            "fixed_address_inputs": True,
            "mutates_runtime_state": bool(self.active),
        }
