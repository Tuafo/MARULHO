"""MarulhoModel -- the stage-0 executable model holding all SNN layers.

Contains the full representation contract: competitive routing, context,
binding, abstraction, cross-modal grounding, surprise monitor, and memory.
This is the "brain" data structure; MarulhoTrainer (in trainer.py) drives
the learning loop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, Optional
import numpy as np
import torch
import torch.nn.functional as F

from marulho.config.model_config import MarulhoConfig
from marulho.core.abstraction import AbstractionLayer
from marulho.core.columns import CompetitiveColumnLayer
from marulho.core.binding import BindingLayer
from marulho.core.column_metabolism import ColumnMetabolismState
from marulho.core.column_runtime import build_column_runtime_report
from marulho.core.context import AdaptiveContextLayer, create_context_layer
from marulho.core.cross_modal import CrossModalGroundingLayer
from marulho.core.surprise import SurpriseMonitor
from marulho.consolidation.memory_store import DualMemoryStore
from marulho.data.base_encoder import BaseEncoder
from marulho.data.encoder_factory import build_encoder
from marulho.retrieval.routing_index import HierarchicalAssemblyIndex, ShardedHierarchicalAssemblyIndex
from marulho.training.bootstrap import PredictiveBootstrap
from marulho.training.column_scheduler import ColumnWakePlan, WAKE_PLAN_EXECUTION_CONSUMERS



@dataclass
class MarulhoModel:
    """Stage-0 executable subset of MARULHO.

    This model implements the representation contract, competitive routing,
    local prototype updates, surprise modulation, and memory drift tracking.
    """

    config: MarulhoConfig

    def __post_init__(self) -> None:
        self.device = self.config.resolve_device()

        # Resolve bootstrap prototypes if teacher mode requested
        bootstrap_proto = None
        if self.config.prototype_init_mode == "teacher":
            try:
                from marulho.training.warm_bootstrap import generate_bootstrap_prototypes
                bootstrap_proto = generate_bootstrap_prototypes(
                    n_columns=self.config.n_columns,
                    column_dim=self.config.column_latent_dim,
                    source=self.config.teacher_embedding_source,
                    vocab_limit=self.config.teacher_vocab_limit,
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Warm bootstrap failed (%s), falling back to random init", e
                )

        self.competitive = CompetitiveColumnLayer(
            n_columns=self.config.n_columns,
            column_dim=self.config.column_latent_dim,
            input_dim=self.config.input_dim,
            device=self.device,
            k_routing=self.config.k_routing,
            lr_initial=self.config.eta_competitive,
            lr_decay=self.config.eta_decay,
            input_weight_blend=self.config.input_weight_blend,
            input_synapse_ltp=self.config.input_synapse_ltp,
            input_synapse_ltd=self.config.input_synapse_ltd,
            input_weight_row_target=self.config.input_weight_row_target,
            plasticity_mode=self.config.plasticity_mode,
            plasticity_spike_backend=self.config.plasticity_spike_backend,
            homeostasis_beta=self.config.homeostasis_beta,
            homeostasis_lr=self.config.homeostasis_lr,
            threshold_min=self.config.threshold_min,
            threshold_max=self.config.threshold_max,
            dead_column_steps=self.config.dead_column_steps,
            dead_column_noise=self.config.dead_column_noise,
            prototype_momentum=self.config.prototype_momentum,
            stdp_trace_tau=self.config.stdp_trace_tau,
            stdp_eligibility_tau=self.config.stdp_eligibility_tau,
            stdp_mu_plus=self.config.stdp_mu_plus,
            stdp_mu_minus=self.config.stdp_mu_minus,
            synaptic_scaling_alpha=self.config.synaptic_scaling_alpha,
            inhibitory_plasticity_lr=self.config.inhibitory_plasticity_lr,
            inhibitory_decay=self.config.inhibitory_decay,
            projection_plasticity_scale=self.config.projection_plasticity_scale,
            assembly_projection_plasticity_scale=self.config.assembly_projection_plasticity_scale,
            projection_norm_target=self.config.projection_norm_target,
            plasticity_rule=self.config.plasticity_rule,
            triplet_tau_plus=self.config.triplet_tau_plus,
            triplet_tau_minus=self.config.triplet_tau_minus,
            triplet_tau_x=self.config.triplet_tau_x,
            triplet_tau_y=self.config.triplet_tau_y,
            triplet_A2_plus=self.config.triplet_A2_plus,
            triplet_A2_minus=self.config.triplet_A2_minus,
            triplet_A3_plus=self.config.triplet_A3_plus,
            triplet_A3_minus=self.config.triplet_A3_minus,
            prototype_init_mode=self.config.prototype_init_mode,
            bootstrap_prototypes=bootstrap_proto,
        )
        self.surprise = SurpriseMonitor(layer_names=["competitive"])
        if not self.config.enable_context_layer:
            self.context_layer = None
        elif self.config.context_mode == "adaptive":
            self.context_layer = create_context_layer(
                mode="adaptive",
                n_columns=self.config.n_columns,
                device=self.device,
                inhibition_strength=self.config.context_inhibition_strength,
                modulation_strength=self.config.context_modulation_strength,
                transition_lr=self.config.context_transition_lr,
            )
        else:
            self.context_layer = create_context_layer(
                mode="fixed",
                n_columns=self.config.n_columns,
                device=self.device,
                decay=self.config.context_decay,
                transition_lr=self.config.context_transition_lr,
                modulation_strength=self.config.context_modulation_strength,
                fast_rate=self.config.context_fast_rate,
                medium_rate=self.config.context_medium_rate,
                slow_rate=self.config.context_slow_rate,
                recurrent_density=self.config.context_recurrent_density,
                recurrent_scale=self.config.context_recurrent_scale,
                inhibition_strength=self.config.context_inhibition_strength,
            )
        self.abstraction_layer = AbstractionLayer(
            n_columns=self.config.n_columns,
            n_concepts=self.config.abstraction_n_concepts,
            device=self.device,
            slow_rate=self.config.abstraction_slow_rate,
            fast_rate=self.config.abstraction_fast_rate,
            learning_rate=self.config.abstraction_learning_rate,
            feedback_lr=self.config.abstraction_feedback_lr,
            feedback_strength=self.config.abstraction_feedback_strength,
        ) if self.config.enable_abstraction_layer else None
        if not self.config.enable_binding_layer:
            self.binding_layer = None
        elif self.config.binding_mode == "spatial":
            from marulho.core.topographic import SpatialBindingLayer
            self.binding_layer = SpatialBindingLayer(
                n_columns=self.config.n_columns,
                device=self.device,
                threshold=self.config.binding_threshold,
                gain_strength=self.config.binding_gain_strength,
                tau_binding=self.config.binding_tau,
                stp_u_inc=self.config.binding_stp_u_inc,
                stp_tau_f=self.config.binding_stp_tau_f,
                stp_tau_d=self.config.binding_stp_tau_d,
                pv_threshold=self.config.binding_pv_threshold,
                pv_gain=self.config.binding_pv_gain,
                association_lr=self.config.binding_association_lr,
                association_decay=self.config.binding_association_decay,
            )
        elif self.config.binding_mode == "hypercube":
            from marulho.core.hypercube import HypercubeBindingLayer
            self.binding_layer = HypercubeBindingLayer(
                n_columns=self.config.n_columns,
                device=self.device,
                threshold=self.config.binding_threshold,
                gain_strength=self.config.binding_gain_strength,
                tau_binding=self.config.binding_tau,
                stp_u_inc=self.config.binding_stp_u_inc,
                stp_tau_f=self.config.binding_stp_tau_f,
                stp_tau_d=self.config.binding_stp_tau_d,
                pv_threshold=self.config.binding_pv_threshold,
                pv_gain=self.config.binding_pv_gain,
                association_lr=self.config.binding_association_lr,
                association_decay=self.config.binding_association_decay,
            )
        else:
            self.binding_layer = BindingLayer(
                n_columns=self.config.n_columns,
                device=self.device,
                threshold=self.config.binding_threshold,
                association_lr=self.config.binding_association_lr,
                association_decay=self.config.binding_association_decay,
                gain_strength=self.config.binding_gain_strength,
                n_bindings=self.config.binding_n_bindings,
                fan_in=self.config.binding_fan_in,
                tau_binding=self.config.binding_tau,
                stp_u_inc=self.config.binding_stp_u_inc,
                stp_tau_f=self.config.binding_stp_tau_f,
                stp_tau_d=self.config.binding_stp_tau_d,
                pv_threshold=self.config.binding_pv_threshold,
                pv_gain=self.config.binding_pv_gain,
            )
        self.cross_modal = CrossModalGroundingLayer(
            dim_text=self.config.input_dim,
            dim_visual=self.config.cross_modal_dim_visual,
            dim_audio=self.config.cross_modal_dim_audio,
            A_plus=self.config.cross_modal_A_plus,
            A_minus=self.config.cross_modal_A_minus,
            tau_trace=self.config.cross_modal_tau_trace,
            confidence_alpha=self.config.cross_modal_confidence_alpha,
            device=self.device,
        ) if self.config.enable_cross_modal else None
        self.memory_store = DualMemoryStore(
            capacity=self.config.memory_capacity,
            ema_alpha=self.config.ema_alpha,
            slow_mean_decay=self.config.slow_mean_decay,
            capture_tag_decay=self.config.stc_tag_decay,
            capture_release=self.config.stc_capture_release,
            consolidation_rate=self.config.stc_consolidation_rate,
            functional_minute=self.config.functional_minute,
            tag_duration_weak=self.config.stc_tag_duration_weak,
            tag_duration_strong=self.config.stc_tag_duration_strong,
            prp_tau_weak=self.config.stc_prp_tau_weak,
            prp_tau_strong=self.config.stc_prp_tau_strong,
            prp_synthesis_rate=self.config.stc_prp_synthesis_rate,
            prp_capture_threshold=self.config.stc_prp_capture_threshold,
            prp_consumption=self.config.stc_prp_consumption,
            strong_event_threshold=self.config.stc_strong_event_threshold,
        )
        if self.config.routing_shards > 1:
            self.routing_index = ShardedHierarchicalAssemblyIndex(
                dim=self.config.column_latent_dim,
                n_shards=self.config.routing_shards,
                rebuild_threshold=self.config.index_rebuild_threshold,
                shard_candidate_factor=self.config.shard_candidate_factor,
                device=self.device,
            )
        else:
            self.routing_index = HierarchicalAssemblyIndex(
                dim=self.config.column_latent_dim,
                rebuild_threshold=self.config.index_rebuild_threshold,
                device=self.device,
            )
        self.last_candidate_sleep_filter_execution = {
            "surface": "column_candidate_sleep_scheduler.v1",
            "mode": "not_run",
            "total_columns": int(self.config.n_columns),
            "awake_budget": int(min(self.config.k_routing, self.config.n_columns)),
            "input_candidate_count": 0,
            "output_candidate_count": 0,
            "filtered_deep_sleep_count": 0,
            "filtered_memory_pressure_count": 0,
            "backfill_candidate_count": 0,
            "deep_sleep_threshold_steps": int(self.config.dead_column_steps),
            "start_token": int(self.config.candidate_deep_sleep_filter_start_tokens),
            "backfill_factor": int(self.config.candidate_deep_sleep_backfill_factor),
            "memory_pressure_threshold": None,
            "memory_pressure_source": None,
            "runs_all_columns": False,
            "fallback_reason": None,
            "tensor_device": str(self.device),
            "claim_boundary": (
                "training_owned_candidate_deep_sleep_filter_skips_deep_sleep_candidates_without_all_column_scan"
            ),
        }
        self.last_column_wake_plan = {
            "surface": "column_wake_plan.v1",
            "mode": "not_run",
            "total_columns": int(self.config.n_columns),
            "awake_budget": int(min(self.config.k_routing, self.config.n_columns)),
            "awake_count": 0,
            "input_candidate_count": 0,
            "filtered_deep_sleep_count": 0,
            "filtered_memory_pressure_count": 0,
            "backfill_candidate_count": 0,
            "bounded": True,
            "runs_all_columns": False,
            "wake_reason": "not_run",
            "sleep_reason": None,
            "fallback_reason": None,
            "memory_pressure_threshold": None,
            "memory_pressure_source": None,
            "tensor_device": str(self.device),
            "awake_column_ids_sample": [],
            "execution_consumers": [],
            "claim_boundary": (
                "training_owned_column_wake_plan_bounds_specialist_execution_without_all_column_sleep_scan"
            ),
        }
        self.column_metabolism = ColumnMetabolismState(
            n_columns=self.config.n_columns,
            device=self.device,
        )

        self.W_assembly_project = torch.empty(
            self.config.n_columns,
            self.config.column_latent_dim,
            device=self.device,
        )
        self.W_assembly_project.log_normal_(mean=-2.3, std=0.5)
        self.W_assembly_project = F.normalize(self.W_assembly_project, dim=0) * self.config.projection_norm_target
        self._W_assembly_project_t = self.W_assembly_project.t().contiguous()

        init_ids = np.arange(self.config.n_columns, dtype=np.int64)
        self.routing_index.add(self.competitive.prototypes.detach(), init_ids)

        # Predictive columns (Thousand Brains Theory)
        # Each column maintains location state, makes predictions, and votes
        from marulho.core.predictive_columns import PredictiveColumnState
        self.predictive = PredictiveColumnState(
            n_columns=self.config.n_columns,
            location_dim=min(8, self.config.column_latent_dim),
            device=self.device,
        )

    def _invalidate_projection_cache(self) -> None:
        """Call after modifying W_assembly_project to refresh the transpose cache."""
        self._W_assembly_project_t = self.W_assembly_project.t().contiguous()

    def subcortex_device_report(self) -> dict[str, Any]:
        """Return live tensor placement evidence for tensor-heavy subcortex modules."""
        context_report = None
        if self.context_layer is not None and hasattr(self.context_layer, "device_report"):
            context_report = self.context_layer.device_report()
        abstraction_report = None
        if self.abstraction_layer is not None:
            abstraction_report = self.abstraction_layer.device_report()
        binding_report = None
        if self.binding_layer is not None and hasattr(self.binding_layer, "device_report"):
            binding_report = self.binding_layer.device_report()
        cross_modal_report = None
        if self.cross_modal is not None:
            cross_modal_report = self.cross_modal.device_report()
        return {
            "competitive": self.competitive.device_report(),
            "predictive": self.predictive.device_report(),
            "context": context_report,
            "abstraction": abstraction_report,
            "binding": binding_report,
            "cross_modal": cross_modal_report,
            "memory_store": self.memory_store.device_report(),
            "assembly_projection_device": str(self.W_assembly_project.device),
            "assembly_projection_transpose_device": str(self._W_assembly_project_t.device),
        }

    def routing_key_from_pattern(self, pattern_vec: torch.Tensor) -> torch.Tensor:
        """Route using spike-proxy assembly activations projected to latent space."""
        x = pattern_vec.to(self.device)
        if self.config.enable_learned_chunking:
            projected = self.competitive.prepare_input_for_candidate_routing(x)
            return F.normalize(projected, dim=0)
        assembly = self.competitive.assembly_from_input(x)
        routing_key = torch.mv(self._W_assembly_project_t, assembly)
        return F.normalize(routing_key, dim=0)

    def column_runtime_report(
        self,
        *,
        token_count: int | None = None,
        last_winner: int | None = None,
    ) -> dict[str, Any]:
        """Return many-column scheduler, execution-scope, and voting evidence."""
        awake_limit = min(int(self.config.k_routing), int(self.config.n_columns))
        stored_wake_plan = getattr(self, "last_column_wake_plan", None)
        execution_awake_indices: torch.Tensor | list[int] | None = None
        execution_wake_reason: str | None = None
        execution_sleep_reason: str | None = None
        execution_scheduler_mode: str | None = None
        execution_fallback_reason: str | None = None
        execution_consumers: list[str] = []
        if isinstance(stored_wake_plan, ColumnWakePlan):
            execution_awake_indices = stored_wake_plan.candidates()
            execution_wake_reason = stored_wake_plan.wake_reason
            execution_sleep_reason = stored_wake_plan.sleep_reason
            execution_scheduler_mode = stored_wake_plan.mode
            execution_fallback_reason = stored_wake_plan.fallback_reason
            execution_consumers = list(WAKE_PLAN_EXECUTION_CONSUMERS)
        elif isinstance(stored_wake_plan, dict):
            execution_scheduler_mode = str(stored_wake_plan.get("mode", "not_run"))
            raw_awake_sample = stored_wake_plan.get("awake_column_ids_sample", [])
            if execution_scheduler_mode == "not_run":
                execution_awake_indices = []
            elif isinstance(raw_awake_sample, list):
                execution_awake_indices = [int(value) for value in raw_awake_sample]
            execution_wake_reason = (
                None
                if stored_wake_plan.get("wake_reason") is None
                else str(stored_wake_plan.get("wake_reason"))
            )
            execution_sleep_reason = (
                None
                if stored_wake_plan.get("sleep_reason") is None
                else str(stored_wake_plan.get("sleep_reason"))
            )
            execution_fallback_reason = (
                None
                if stored_wake_plan.get("fallback_reason") is None
                else str(stored_wake_plan.get("fallback_reason"))
            )
            execution_consumers = [
                str(value)
                for value in stored_wake_plan.get("execution_consumers", [])
                if isinstance(value, str)
            ]
        report = build_column_runtime_report(
            n_columns=int(self.config.n_columns),
            prediction_error=getattr(self.predictive, "prediction_error", None),
            confidence=getattr(self.predictive, "confidence", None),
            steps_since_win=(
                self.competitive.state_transition_steps_snapshot()
                if hasattr(self.competitive, "state_transition_steps_snapshot")
                else getattr(self.competitive, "steps_since_win", None)
            ),
            win_rate_ema=getattr(self.competitive, "win_rate_ema", None),
            prediction_failure_streak=getattr(self.predictive, "prediction_failure_streak", None),
            estimated_cost=getattr(self.column_metabolism, "estimated_cost", None),
            memory_pressure=getattr(self.column_metabolism, "memory_pressure", None),
            memory_pressure_source=getattr(
                self.column_metabolism,
                "last_memory_pressure_source",
                None,
            ),
            last_winner_ids=[] if last_winner is None else [int(last_winner)],
            awake_limit=awake_limit,
            sleep_after_steps=max(1, min(64, int(self.config.dead_column_steps // 4))),
            deep_sleep_after_steps=max(awake_limit + 1, int(self.config.dead_column_steps)),
            token_count=token_count,
            device=str(self.device),
            execution_awake_indices=execution_awake_indices,
            execution_wake_reason=execution_wake_reason,
            execution_sleep_reason=execution_sleep_reason,
            execution_scheduler_mode=execution_scheduler_mode,
            execution_fallback_reason=execution_fallback_reason,
            execution_consumers=execution_consumers,
        )
        execution = self.competitive.execution_report()
        predictive_update_execution = self.predictive.prediction_update_execution_report()
        predictive_vote_execution = self.predictive.vote_execution_report()
        column_metabolism_execution = self.column_metabolism.report()
        if isinstance(stored_wake_plan, ColumnWakePlan):
            column_wake_plan = stored_wake_plan.to_report()
            candidate_sleep_filter_execution = stored_wake_plan.to_execution_report()
        else:
            column_wake_plan = dict(
                stored_wake_plan
                if isinstance(stored_wake_plan, dict)
                else {
                    "surface": "column_wake_plan.v1",
                    "mode": "not_run",
                    "total_columns": int(self.config.n_columns),
                    "awake_budget": awake_limit,
                    "awake_count": 0,
                    "input_candidate_count": 0,
                    "filtered_deep_sleep_count": 0,
                    "filtered_memory_pressure_count": 0,
                    "backfill_candidate_count": 0,
                    "bounded": True,
                    "runs_all_columns": False,
                    "wake_reason": "not_run",
                    "sleep_reason": None,
                    "fallback_reason": None,
                    "memory_pressure_threshold": None,
                    "memory_pressure_source": None,
                    "tensor_device": str(self.device),
                    "awake_column_ids_sample": [],
                    "execution_consumers": [],
                    "claim_boundary": (
                        "training_owned_column_wake_plan_bounds_specialist_execution_without_all_column_sleep_scan"
                    ),
                }
            )
            candidate_sleep_filter_execution = dict(
                getattr(
                    self,
                    "last_candidate_sleep_filter_execution",
                    {
                        "surface": "column_candidate_sleep_scheduler.v1",
                        "mode": "not_run",
                        "total_columns": int(self.config.n_columns),
                        "awake_budget": awake_limit,
                        "input_candidate_count": 0,
                        "output_candidate_count": 0,
                        "filtered_deep_sleep_count": 0,
                        "filtered_memory_pressure_count": 0,
                        "backfill_candidate_count": 0,
                        "deep_sleep_threshold_steps": int(self.config.dead_column_steps),
                        "start_token": int(
                            self.config.candidate_deep_sleep_filter_start_tokens
                        ),
                        "backfill_factor": int(
                            self.config.candidate_deep_sleep_backfill_factor
                        ),
                        "memory_pressure_threshold": None,
                        "memory_pressure_source": None,
                        "runs_all_columns": False,
                        "fallback_reason": None,
                        "tensor_device": str(self.device),
                        "claim_boundary": (
                            "training_owned_candidate_deep_sleep_filter_skips_deep_sleep_candidates_without_all_column_scan"
                        ),
                    },
                )
            )
        total_columns = int(report.get("total_columns", 0) or 0)
        competitive_runs_all = bool(execution.get("runs_all_columns", False))
        if total_columns > 0:
            competitive_runs_all = competitive_runs_all or (
                int(execution.get("scored_column_count", 0) or 0) >= total_columns
                and execution.get("mode") not in {None, "not_run"}
            )
            competitive_runs_all = competitive_runs_all or (
                int(execution.get("homeostasis_update_count", 0) or 0) >= total_columns
                and execution.get("homeostasis_update_mode") not in {None, "not_run"}
            )
        runs_all_columns = bool(
            competitive_runs_all
            or predictive_update_execution.get("runs_all_columns", False)
            or predictive_vote_execution.get("runs_all_columns", False)
            or column_metabolism_execution.get("runs_all_columns", False)
            or bool(report.get("runs_all_columns", False))
        )
        report["execution"] = execution
        report["candidate_sleep_filter_execution"] = candidate_sleep_filter_execution
        report["column_wake_plan"] = column_wake_plan
        report["predictive_update_execution"] = predictive_update_execution
        report["predictive_vote_execution"] = predictive_vote_execution
        report["column_metabolism_execution"] = column_metabolism_execution
        report["runs_all_columns"] = runs_all_columns
        if isinstance(report.get("scheduler"), dict):
            report["scheduler"]["runs_all_columns"] = runs_all_columns
            report["scheduler"]["execution_scope"] = (
                "candidate_deep_sleep_and_memory_pressure_filter_scoring_homeostasis_"
                "predictive_update_and_vote_cache"
            )
            report["scheduler"]["fallback_reason"] = (
                "one_or_more_specialists_ran_all_columns"
                if runs_all_columns
                else report["scheduler"].get("fallback_reason")
            )
        return report

    def runtime_scope_report(
        self,
        *,
        token_count: int | None = None,
        last_winner: int | None = None,
    ) -> dict[str, Any]:
        routing_index_stats = self.routing_index.stats()
        device_report = self.config.device_report()
        subcortex_device_report = self.subcortex_device_report()
        column_runtime = self.column_runtime_report(
            token_count=token_count,
            last_winner=last_winner,
        )
        local_stdp_active = self.config.plasticity_mode == "local_stdp"
        adex_post_spikes = local_stdp_active and self.config.plasticity_spike_backend == "adex"
        sharding_active = self.config.routing_shards > 1
        reason = (
            "The runnable scaffold exposes a maintained local plasticity circuit with log-STDP-style eligibility traces, "
            "iSTDP-style inhibitory balancing, synaptic scaling, competitive prototypes, plastic latent projections, "
            "AdEx-backed postsynaptic spikes for biologically faithful STDP timing, "
            "validated log-normal synaptic weight targets, "
            + ("column sharding for scalable routing, " if sharding_active else "")
            + "and an explicit tag/PRP replay-consolidation stack."
            if adex_post_spikes
            else (
                "The runnable scaffold exposes a maintained local plasticity circuit with log-STDP-style eligibility traces, "
                "iSTDP-style inhibitory balancing, synaptic scaling, competitive prototypes, plastic latent projections, "
                "validated log-normal synaptic weight targets, "
                + ("column sharding for scalable routing, " if sharding_active else "")
                + "and an explicit tag/PRP replay-consolidation stack."
                if local_stdp_active
                else (
                    "The runnable scaffold uses "
                    + (
                        "active blended column input weights, "
                        if self.competitive.input_weight_blend > 0.0
                        else "dormant checkpointed column input weights excluded from live scoring and lightweight plasticity, "
                    )
                    + "competitive prototypes, a latent projection matrix, "
                    + ("column sharding for scalable routing, " if sharding_active else "")
                    + "and an explicit tag/PRP replay-consolidation stack."
                )
            )
        )
        return {
            "model_type": "MarulhoModel",
            "benchmark_family": (
                "contextual_routing_multiscale"
                if self.context_layer is not None
                else "mechanism_validation_competitive"
            ),
            "input_representation": str(self.config.input_representation),
            "plasticity_mode": str(self.config.plasticity_mode),
            "plasticity_spike_backend": str(self.config.plasticity_spike_backend) if local_stdp_active else None,
            "input_dim": int(self.config.input_dim),
            "validates_full_log_stdp_weight_target": bool(
                self.competitive.validate_synaptic_health()["validates"]
            ),
            "supports_local_log_stdp": bool(local_stdp_active),
            "supports_inhibitory_balance": bool(local_stdp_active),
            "uses_precise_spike_trace_when_available": bool(local_stdp_active),
            "uses_adex_post_spikes": bool(adex_post_spikes),
            "supports_stc_like_memory_consolidation": True,
            "supports_explicit_prp_state_stack": True,
            "memory_consolidation_mode": "tag_prp_replay_consolidation",
            "reason": reason,
            "supports_contextual_routing": self.context_layer is not None,
            "supports_first_class_abstraction": self.abstraction_layer is not None,
            "supports_binding_conjunction_memory": self.binding_layer is not None,
            "supports_approximate_attractor_context": self.context_layer is not None,
            "supports_binding_coincidence": self.binding_layer is not None,
            "context_architecture": "multiscale_recurrent_attractor" if self.context_layer is not None else None,
            "abstraction_architecture": (
                "slow_feature_feedback_layer"
                if self.abstraction_layer is not None
                else None
            ),
            "binding_architecture": (
                "sparse_subset_stp_coincidence_with_pv_inhibition"
                if self.binding_layer is not None
                else None
            ),
            "supports_column_sharding_proxy": bool(self.config.routing_shards > 1),
            "warm_bootstrap": self.config.prototype_init_mode == "teacher",
            "estimated_neurons": int(self.config.n_columns * self.config.neurons_per_column_assumption),
            "neurons_per_column_assumption": int(self.config.neurons_per_column_assumption),
            "routing_candidate_fraction": float(self.config.k_routing / max(1, self.config.n_columns)),
            "routing_backend_mode": "torch_topk",
            "routing_index": routing_index_stats,
            "spike_health": self.competitive.spike_health_report(),
            "column_runtime": column_runtime,
            "device": device_report,
            "cuda_first_runtime": {
                "enabled_when_available": self.config.device == "auto" and device_report["env_device"] is None,
                "tensor_device": str(self.device),
                "routing_search_device": routing_index_stats.get("search_device"),
                "subcortex_tensor_devices": subcortex_device_report,
                "routing_backend_cuda_capable": routing_index_stats.get("index_type") in {
                    "torch_topk",
                    "sharded_torch_topk",
                },
                "unit_tests_default_cpu": True,
            },
            "weight_distribution": self.competitive.distribution_proxy_stats(),
        }


