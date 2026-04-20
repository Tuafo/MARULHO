from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, Optional
import numpy as np
import torch
import torch.nn.functional as F

from hecsn.config.model_config import HECSNConfig
from hecsn.core.abstraction import AbstractionLayer
from hecsn.core.columns import CompetitiveColumnLayer
from hecsn.core.context import AdaptiveContextLayer, BindingLayer, create_context_layer
from hecsn.core.cross_modal import CrossModalGroundingLayer
from hecsn.core.surprise import SurpriseMonitor
from hecsn.consolidation.memory_store import DualMemoryStore
from hecsn.data.base_encoder import BaseEncoder
from hecsn.data.encoder_factory import build_encoder
from hecsn.retrieval.hnsw_index import HierarchicalAssemblyIndex, ShardedHierarchicalAssemblyIndex
from hecsn.training.bootstrap import PredictiveBootstrap


@dataclass
class HECSNModel:
    """Stage-0 executable subset of HECSN.

    This model implements the representation contract, competitive routing,
    local prototype updates, surprise modulation, and memory drift tracking.
    """

    config: HECSNConfig

    def __post_init__(self) -> None:
        self.device = self.config.resolve_device()

        # Resolve bootstrap prototypes if teacher mode requested
        bootstrap_proto = None
        if self.config.prototype_init_mode == "teacher":
            try:
                from hecsn.training.warm_bootstrap import generate_bootstrap_prototypes
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
            from hecsn.core.topographic import SpatialBindingLayer
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
            from hecsn.core.hypercube import HypercubeBindingLayer
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
            self.hnsw_index = ShardedHierarchicalAssemblyIndex(
                dim=self.config.column_latent_dim,
                n_shards=self.config.routing_shards,
                rebuild_threshold=self.config.index_rebuild_threshold,
                shard_candidate_factor=self.config.shard_candidate_factor,
                device=self.device,
                backend=self.config.routing_index_mode,
            )
        else:
            self.hnsw_index = HierarchicalAssemblyIndex(
                dim=self.config.column_latent_dim,
                rebuild_threshold=self.config.index_rebuild_threshold,
                device=self.device,
                backend=self.config.routing_index_mode,
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
        self.hnsw_index.add(self.competitive.prototypes.detach(), init_ids)

    def _invalidate_projection_cache(self) -> None:
        """Call after modifying W_assembly_project to refresh the transpose cache."""
        self._W_assembly_project_t = self.W_assembly_project.t().contiguous()

    def routing_key_from_pattern(self, pattern_vec: torch.Tensor) -> torch.Tensor:
        """Route using spike-proxy assembly activations projected to latent space."""
        x = pattern_vec.to(self.device)
        assembly = self.competitive.assembly_from_input(x)
        if self.config.enable_learned_chunking:
            projected = self.competitive.last_projected_input
            if projected is None:
                projected = self.competitive.project_input(x)
            return F.normalize(projected, dim=0)
        routing_key = torch.mv(self._W_assembly_project_t, assembly)
        return F.normalize(routing_key, dim=0)

    def runtime_scope_report(self) -> dict[str, Any]:
        routing_index_stats = self.hnsw_index.stats()
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
                    "The runnable scaffold uses active column input weights, competitive prototypes, a latent projection matrix, "
                    + ("column sharding for scalable routing, " if sharding_active else "")
                    + "and an explicit tag/PRP replay-consolidation stack."
                )
            )
        )
        return {
            "model_type": "HECSNModel",
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
            "routing_backend_mode": str(self.config.routing_index_mode),
            "routing_index": routing_index_stats,
            "weight_distribution": self.competitive.distribution_proxy_stats(),
        }


class HECSNTrainer:
    """Main stage-0 trainer."""

    def __init__(self, model: HECSNModel, config: HECSNConfig):
        self.model = model
        self.config = config
        self.token_count = 0
        self.is_bootstrap = True
        self.sleep_events = 0
        self.micro_sleep_events = 0
        self.deep_sleep_events = 0
        self.last_micro_sleep_token = -10**9
        self.last_deep_sleep_token = -10**9
        self.current_window_min_drift = float("inf")
        self.previous_window_min_drift: float | None = None
        self.recent_drifts = deque(maxlen=self.config.drift_floor_history_tokens)
        self.current_rolling_drift_floor: float | None = None
        self.previous_rolling_drift_floor: float | None = None
        self.last_floor_check_token = -10**9
        self.memory_warm_started = self.config.slow_memory_start_tokens <= 0
        self.last_winner: int | None = None
        self.pending_emergency_deep_sleep = False
        self.last_network_reset_token: int = -10**9
        self._exploration_noise_scale: float = 1.0
        self._cached_drift: float | None = None
        self._dead_column_census: dict[str, int | float] = {}
        self.column_anchors: dict[int, dict[str, torch.Tensor | float]] = {}
        self.bootstrap = PredictiveBootstrap(device=self.model.device, input_dim=self.config.input_dim)
        self.encoder: BaseEncoder = build_encoder(self.config)
        self._recent_stream_text = ""
        self._last_raw_window_text: str | None = None
        self._cached_episode_text: str | None = None
        self._last_episode_refresh_length = 0
        self._segment_cache_key: str | None = None
        self._segment_cache_result: list[str] | None = None
        # §7.4 self-criticism state
        self._recent_visual_frames: list[torch.Tensor] = []
        self._recent_audio_frames: list[torch.Tensor] = []
        self._visual_frame_limit = 100
        self._audio_frame_limit = 100
        self._self_criticism_interval = 1000
        self._last_self_criticism_token = 0
        self._self_criticism_blacklist: dict[int, int] = {}
        self._self_criticism_audio_blacklist: dict[int, int] = {}
        # Self-criticism history for find-rate computation (§7.3 criterion 2)
        self._self_criticism_history: list[dict[str, int]] = []

        # Per-word grounding confidence (§5.3): tracks cross-modal prediction
        # quality per concept word.  Updated by the developmental runner when
        # multimodal pairs are processed — words with consistent visual/audio
        # associations develop high confidence, words never paired stay at 0.
        self.word_grounding_confidence: dict[str, float] = {}
        self._word_grounding_alpha: float = 0.05  # EMA rate

        # Per-word accumulated visual/audio signatures (cell assembly
        # encoding).  Each grounded word develops its own sensory prototype
        # via EMA of observed visual/audio patterns during training.  Used
        # by the grounding probe to construct word representations.
        self.word_visual_signature: dict[str, torch.Tensor] = {}
        self.word_audio_signature: dict[str, torch.Tensor] = {}
        self._word_sig_alpha: float = 0.1  # EMA rate for signatures

        # Developmental stage (§7): controls cross-modal gating behaviour.
        # Stage 1: accept all visual/audio (no filter)
        # Stage 2+: apply alignment_gate() before cross-modal updates
        self.developmental_stage: int = 1
        # Bootstrap budget for Stage 2: accept first N multimodal pairs
        # without gating so confidence can bootstrap from zero.
        # Tracked separately for visual and audio modalities.
        self._stage2_bootstrap_budget: int = 50
        self._stage2_bootstrap_used_visual: int = 0
        self._stage2_bootstrap_used_audio: int = 0
        # Legacy alias kept for checkpoint backward compat
        self._stage2_bootstrap_used: int = 0

        # HNSW update buffer — flush every N steps to amortize add() overhead
        self._hnsw_buffer_ids: list[int] = []
        self._hnsw_buffer_vecs: list[torch.Tensor] = []
        self._hnsw_flush_interval = 16

    def _buffer_hnsw_update(self, indices: torch.Tensor, vectors: torch.Tensor) -> None:
        """Buffer HNSW updates; flush when buffer reaches interval size."""
        ids = indices.detach().tolist() if not indices.is_cuda else indices.detach().cpu().tolist()
        vecs = vectors.detach()
        for i, vid in enumerate(ids):
            self._hnsw_buffer_ids.append(int(vid))
            self._hnsw_buffer_vecs.append(vecs[i])
        if len(self._hnsw_buffer_ids) >= self._hnsw_flush_interval:
            self._flush_hnsw_buffer()

    def _flush_hnsw_buffer(self) -> None:
        """Flush buffered HNSW updates in a single batch."""
        if not self._hnsw_buffer_ids:
            return
        # Deduplicate: keep latest vector per id
        seen: dict[int, torch.Tensor] = {}
        for vid, vec in zip(self._hnsw_buffer_ids, self._hnsw_buffer_vecs):
            seen[vid] = vec
        ids_arr = np.array(list(seen.keys()), dtype=np.int64)
        vecs_batch = torch.stack(list(seen.values()))
        self.model.hnsw_index.add(vecs_batch, ids_arr)
        self._hnsw_buffer_ids.clear()
        self._hnsw_buffer_vecs.clear()

    def update_word_grounding(
        self,
        word: str,
        text_spike: torch.Tensor,
        actual_visual: torch.Tensor | None = None,
        actual_audio: torch.Tensor | None = None,
    ) -> float:
        """Update per-word grounding confidence and sensory signatures.

        Computes cosine similarity between the predicted and actual sensory
        patterns, then updates the word's EMA confidence.  Also accumulates
        per-word visual/audio signature prototypes via EMA.

        Returns the current confidence for this word.
        """
        if self.model.cross_modal is None:
            return 0.0

        qualities: list[float] = []
        cm = self.model.cross_modal
        dev = cm.device
        if actual_visual is not None and actual_visual.norm() > 1e-6:
            pred = cm.predict_visual(text_spike)
            av = actual_visual.to(dev)
            if pred.norm() > 1e-6:
                cos = F.cosine_similarity(
                    pred.unsqueeze(0), av.unsqueeze(0),
                ).item()
                qualities.append(max(0.0, cos))
            # Accumulate per-word visual signature (EMA)
            v = av.detach()
            if word in self.word_visual_signature:
                a = self._word_sig_alpha
                self.word_visual_signature[word] = (
                    (1 - a) * self.word_visual_signature[word] + a * v
                )
            else:
                self.word_visual_signature[word] = v.clone()

        if actual_audio is not None and actual_audio.norm() > 1e-6:
            pred = cm.predict_audio(text_spike)
            aa = actual_audio.to(dev)
            if pred.norm() > 1e-6:
                cos = F.cosine_similarity(
                    pred.unsqueeze(0), aa.unsqueeze(0),
                ).item()
                qualities.append(max(0.0, cos))
            # Accumulate per-word audio signature (EMA)
            a_sig = aa.detach()
            if word in self.word_audio_signature:
                a = self._word_sig_alpha
                self.word_audio_signature[word] = (
                    (1 - a) * self.word_audio_signature[word] + a * a_sig
                )
            else:
                self.word_audio_signature[word] = a_sig.clone()

        if not qualities:
            return self.word_grounding_confidence.get(word, 0.0)

        quality = sum(qualities) / len(qualities)
        prev = self.word_grounding_confidence.get(word, quality)
        alpha = self._word_grounding_alpha
        updated = (1.0 - alpha) * prev + alpha * quality
        self.word_grounding_confidence[word] = updated
        return updated

    def _update_stream_text(self, raw_window: Optional[str]) -> Optional[str]:
        if raw_window is None:
            self._last_raw_window_text = None
            self._recent_stream_text = ""
            self._cached_episode_text = None
            self._last_episode_refresh_length = 0
            return None

        current = str(raw_window)
        if not current:
            return None

        previous = self._last_raw_window_text
        appended = current
        if previous is None:
            self._recent_stream_text = current
            self._last_raw_window_text = current
            episode_text = self._current_episode_text(current)
            self._cached_episode_text = episode_text
            self._last_episode_refresh_length = len(self._recent_stream_text)
            return episode_text

        best_overlap = 0
        max_overlap = min(len(previous), len(current))
        # Fast path: try the most common overlap (len-1, len-2) first
        for overlap in (max_overlap, max_overlap - 1, max_overlap - 2):
            if overlap > 0 and previous[-overlap:] == current[:overlap]:
                best_overlap = overlap
                break
        else:
            # Fall back to scanning from max_overlap downward
            for overlap in range(max_overlap - 3, 0, -1):
                if previous[-overlap:] == current[:overlap]:
                    best_overlap = overlap
                    break

        required_overlap = max(1, min(len(previous), len(current)) - 2)
        if best_overlap >= required_overlap:
            appended = current[best_overlap:]
            if appended:
                self._recent_stream_text += appended
        else:
            self._recent_stream_text = current
            appended = current

        self._last_raw_window_text = current
        if len(self._recent_stream_text) > 512:
            self._recent_stream_text = self._recent_stream_text[-512:]
            self._last_episode_refresh_length = min(self._last_episode_refresh_length, len(self._recent_stream_text))

        if getattr(self.encoder, "uses_learned_chunking", False):
            cached_terms = {
                token.lower()
                for token in re.findall(r"[A-Za-z0-9']+", str(self._cached_episode_text or ""))
                if len(token) > 2
            }
            current_terms = {
                token.lower()
                for token in re.findall(r"[A-Za-z0-9']+", current)
                if len(token) > 2
            }
            refresh_due = (
                self._cached_episode_text is None
                or len(self._recent_stream_text) - self._last_episode_refresh_length >= 24
                or any(ch in ".!?\n" for ch in appended)
                or bool(current_terms - cached_terms)
            )
            if not refresh_due:
                return self._cached_episode_text

        episode_text = self._current_episode_text(current)
        self._cached_episode_text = episode_text
        self._last_episode_refresh_length = len(self._recent_stream_text)
        return episode_text

    def _current_episode_text(self, raw_window: str) -> Optional[str]:
        text = self._recent_stream_text.strip()
        if not text:
            return None

        sentence_like_segments = [
            segment.strip()
            for segment in re.split(r"(?<=[.!?])\s+|\n+", text)
            if segment and segment.strip()
        ]
        if getattr(self.encoder, "uses_learned_chunking", False) and len(sentence_like_segments) <= 1:
            window_terms = {token.lower() for token in re.findall(r"[A-Za-z0-9']+", raw_window)}
            tail_chars = max(96, min(192, len(raw_window) * 8))
            seg_key = text[-tail_chars:]
            # Re-segment only every ~8 new characters — boundaries are stable
            cache_valid = (
                self._segment_cache_result is not None
                and self._segment_cache_key is not None
                and abs(len(seg_key) - len(self._segment_cache_key)) < 8
            )
            if cache_valid:
                learned_segments = self._segment_cache_result
            else:
                learned_segments = self.encoder.segment_text(seg_key)
                self._segment_cache_key = seg_key
                self._segment_cache_result = learned_segments
            if learned_segments:
                best_candidate: str | None = None
                best_score: tuple[int, int, int] | None = None
                max_span = min(12, len(learned_segments))
                for span in range(1, max_span + 1):
                    candidate = " ".join(segment for segment in learned_segments[-span:] if segment).strip()
                    if not candidate:
                        continue
                    candidate_terms = {token.lower() for token in re.findall(r"[A-Za-z0-9']+", candidate)}
                    overlap = len(window_terms & candidate_terms)
                    if raw_window.lower() in candidate.lower():
                        overlap += len(raw_window)
                    completeness = int(candidate.endswith((".", "!", "?")))
                    score = (overlap, completeness, len(candidate))
                    if best_score is None or score > best_score:
                        best_score = score
                        best_candidate = candidate
                if best_candidate is not None and best_score is not None and (best_score[0] > 0 or len(learned_segments) == 1):
                    return best_candidate[-240:]

        segments = sentence_like_segments
        if segments:
            window_terms = {token.lower() for token in re.findall(r"[A-Za-z0-9']+", raw_window)}
            candidates = segments[-2:] if len(segments) > 1 else segments

            def _segment_score(segment: str) -> tuple[int, int]:
                segment_terms = {token.lower() for token in re.findall(r"[A-Za-z0-9']+", segment)}
                overlap = len(window_terms & segment_terms)
                if raw_window.lower() in segment.lower():
                    overlap += len(raw_window)
                completeness = int(segment.endswith((".", "!", "?")))
                return (overlap, completeness)

            current = max(candidates, key=_segment_score)
            current_tokens = re.findall(r"[A-Za-z0-9']+", current)
            if len(current_tokens) < 4 and not current.endswith((".", "!", "?")) and len(segments) > 1:
                previous = segments[-2]
                if _segment_score(previous) >= _segment_score(current):
                    current = previous
                else:
                    current = f"{previous} {current}".strip()
            return current[-240:]

        return text[-240:]

    def _reset_drift_tracking(self) -> None:
        self.recent_drifts.clear()
        self.current_rolling_drift_floor = None
        self.previous_rolling_drift_floor = None
        self.last_floor_check_token = -10**9
        self.current_window_min_drift = float("inf")
        self.previous_window_min_drift = None

    def _maybe_warm_start_memory(self, next_token: int) -> bool:
        if self.memory_warm_started:
            return False
        if next_token < self.config.slow_memory_start_tokens:
            return False

        self.model.memory_store.reset()
        self._reset_drift_tracking()
        self.last_micro_sleep_token = next_token
        self.last_deep_sleep_token = next_token
        self.memory_warm_started = True
        return True

    def _close_drift_floor_window(self) -> None:
        if self.current_window_min_drift == float("inf"):
            return

        rising_floor = (
            self.previous_window_min_drift is not None
            and self.current_window_min_drift
            > self.previous_window_min_drift + self.config.drift_floor_rise_tolerance
        )
        self.previous_window_min_drift = self.current_window_min_drift
        self.current_window_min_drift = float("inf")
        self.pending_emergency_deep_sleep = bool(rising_floor)

    def _update_rolling_drift_floor(self, drift: float) -> bool:
        self.recent_drifts.append(float(drift))
        self.current_rolling_drift_floor = float(min(self.recent_drifts)) if self.recent_drifts else None

        if len(self.recent_drifts) < self.config.drift_floor_history_tokens:
            return False
        if (self.token_count - self.last_floor_check_token) < self.config.drift_floor_check_interval_tokens:
            return False

        self.last_floor_check_token = self.token_count
        rolling_floor = float(min(self.recent_drifts))
        rising_floor = (
            self.token_count >= self.config.drift_floor_trigger_min_tokens
            and self.previous_rolling_drift_floor is not None
            and rolling_floor > self.previous_rolling_drift_floor + self.config.drift_floor_rise_tolerance
            and rolling_floor > self.config.drift_threshold
        )
        self.previous_rolling_drift_floor = rolling_floor
        self.current_rolling_drift_floor = rolling_floor
        return rising_floor

    def _refresh_latest_drift(self, drift: float) -> None:
        if not self.recent_drifts:
            return
        self.recent_drifts[-1] = float(drift)
        self.current_rolling_drift_floor = float(min(self.recent_drifts))

    @staticmethod
    def _normalize_signal(signal: torch.Tensor) -> torch.Tensor:
        signal = torch.clamp(signal.float(), min=0.0)
        total = float(signal.sum().item())
        if total <= 0.0:
            return torch.zeros_like(signal)
        return signal / (signal.sum() + 1e-8)

    def _local_trace_from_raw_window(
        self,
        raw_window: Optional[str],
        *,
        context_confidence: float,
    ) -> torch.Tensor | None:
        if self.config.plasticity_mode != "local_stdp":
            return None
        if raw_window is None or self.config.input_representation not in ("order_weighted_ascii", "unigram_ascii"):
            return None

        ascii_codes = [ord(ch) if ord(ch) < self.config.n_ascii else 0 for ch in str(raw_window)]
        if not ascii_codes:
            return None

        confidence = max(0.0, min(1.0, float(context_confidence)))
        trace = self.encoder.spike_trace(
            ascii_codes,
            context_confidence=confidence,
            tau=self.config.spike_trace_tau,
            burst_decay=self.config.spike_burst_decay,
        )
        if float(trace.sum().item()) <= 0.0:
            return None
        return trace.to(self.model.device)

    def _context_prediction_and_gain(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.model.context_layer is None:
            context_prediction = None
            routing_gain = None
        else:
            context_prediction = self.model.context_layer.context_prediction()
            routing_gain = self.model.context_layer.modulation_gain_for_signal(
                context_prediction,
                norepinephrine=self.model.surprise.norepinephrine,
                acetylcholine=self.model.surprise.acetylcholine,
            )
        if self.model.abstraction_layer is not None:
            abstraction_gain = self.model.abstraction_layer.routing_gain()
            routing_gain = abstraction_gain if routing_gain is None else torch.clamp(
                routing_gain * abstraction_gain,
                min=0.5,
                max=1.5,
            )
        if self.model.binding_layer is not None and context_prediction is not None:
            if routing_gain is None:
                routing_gain = torch.ones(self.config.n_columns, device=self.model.device)
            routing_gain = torch.clamp(
                routing_gain * self.model.binding_layer.modulation_gain(context_prediction),
                min=0.5,
                max=1.5,
            )
        return context_prediction, routing_gain

    def _context_precision_weight(self) -> float:
        return float(self.model.surprise.precision_weight("competitive"))

    def _offline_context_source_and_gain(
        self,
        *,
        blend_state: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.model.context_layer is None:
            context_source = None
            routing_gain = None
        else:
            context_source = self.model.context_layer.context_prediction()
            if blend_state:
                context_source = self._normalize_signal(self.model.context_layer.state + context_source)

            routing_gain = self.model.context_layer.modulation_gain_for_signal(
                context_source,
                norepinephrine=self.model.surprise.norepinephrine,
                acetylcholine=self.model.surprise.acetylcholine,
            )
        if self.model.abstraction_layer is not None:
            abstraction_gain = self.model.abstraction_layer.routing_gain()
            routing_gain = abstraction_gain if routing_gain is None else torch.clamp(
                routing_gain * abstraction_gain,
                min=0.5,
                max=1.5,
            )
        if self.model.binding_layer is not None and context_source is not None:
            if routing_gain is None:
                routing_gain = torch.ones(self.config.n_columns, device=self.model.device)
            routing_gain = torch.clamp(
                routing_gain * self.model.binding_layer.modulation_gain_for_context(context_source),
                min=0.5,
                max=1.5,
            )
        return context_source, routing_gain

    def _apply_binding(
        self,
        assembly: torch.Tensor,
        context_prediction: torch.Tensor | None,
        *,
        update_weights: bool,
    ) -> tuple[torch.Tensor, float]:
        if self.model.binding_layer is None or context_prediction is None:
            return assembly, 0.0

        bound_assembly, binding_strength = self.model.binding_layer.bind(
            context_prediction,
            assembly,
            update_weights=update_weights,
        )
        if float(bound_assembly.sum().item()) <= 0.0:
            return assembly, float(binding_strength)
        return bound_assembly, float(binding_strength)

    def _apply_column_anchors(
        self,
        column_ids: Iterable[int],
        *,
        blend_scale: float = 0.05,
        blend_cap: float = 0.35,
    ) -> None:
        seen: set[int] = set()
        for column_id in column_ids:
            idx = int(column_id)
            if idx in seen or idx not in self.column_anchors:
                continue
            seen.add(idx)
            anchor = self.column_anchors[idx]
            strength = float(anchor["strength"])
            blend = max(0.0, min(float(blend_cap), float(blend_scale) * strength))
            if blend <= 0.0:
                continue

            prototype_anchor = anchor["prototype"]
            input_weight_anchor = anchor["input_weights"]
            if not isinstance(prototype_anchor, torch.Tensor) or not isinstance(input_weight_anchor, torch.Tensor):
                continue

            prototype_anchor = prototype_anchor.to(self.model.device)
            input_weight_anchor = input_weight_anchor.to(self.model.device)

            current_proto = self.model.competitive.prototypes[idx]
            current_weights = self.model.competitive.input_weights[idx]
            self.model.competitive.prototypes[idx] = F.normalize(
                (1.0 - blend) * current_proto + blend * prototype_anchor,
                dim=0,
            )
            anchored_weights = torch.clamp(
                (1.0 - blend) * current_weights + blend * input_weight_anchor,
                min=1e-6,
            )
            anchored_weights = anchored_weights * (
                self.model.competitive.input_weight_row_target
                / (anchored_weights.sum() + 1e-8)
            )
            self.model.competitive.input_weights[idx] = anchored_weights

    def _repair_column_from_replay(
        self,
        column_id: int,
        routing_key: torch.Tensor,
        *,
        strength: float = 0.30,
    ) -> None:
        idx = int(column_id)
        if idx < 0 or idx >= self.config.n_columns:
            return
        if float(routing_key.abs().sum().item()) <= 0.0:
            return

        target = F.normalize(routing_key.to(self.model.device), dim=0)
        current = self.model.competitive.prototypes[idx]
        repaired = F.normalize((1.0 - strength) * current + strength * target, dim=0)
        self.model.competitive.prototypes[idx] = repaired
        self.model.competitive.prototype_velocity[idx] = 0.0

    def _sleep_replay(self, mode: str) -> int:
        """Replay slow-buffer assemblies with spaced priority and mode-specific depth."""
        replay_use_stored_bucket = True
        anchor_blend_scale = 0.05
        anchor_blend_cap = 0.35
        repair_anchor_strength = 0.30

        if mode == "micro":
            steps = self.config.micro_sleep_replay_steps
            candidate_pool = self.config.micro_sleep_candidate_pool
            sampling_strategy = "maintenance"
            memory_blend = self.config.micro_sleep_memory_blend
            modulator = 0.0
            protein_synthesis_level = 0.0
            prototype_lr_scale = 0.0
            input_lr_scale = 0.0
        elif mode == "deep":
            steps = self.config.deep_sleep_replay_steps
            candidate_pool = self.config.deep_sleep_candidate_pool
            sampling_strategy = "consolidation"
            memory_blend = self.config.deep_sleep_memory_blend
            modulator = 0.08
            protein_synthesis_level = 1.35
            prototype_lr_scale = 0.10
            input_lr_scale = 0.05
        elif mode == "repair":
            steps = self.config.deep_sleep_replay_steps
            candidate_pool = self.config.deep_sleep_candidate_pool
            sampling_strategy = "repair"
            memory_blend = 0.0
            modulator = 0.0
            protein_synthesis_level = 0.0
            prototype_lr_scale = 0.0
            input_lr_scale = 0.0
        else:
            raise ValueError(f"Unknown sleep mode: {mode}")

        replay_idx = self.model.memory_store.sample_replay_indices(
            n=steps,
            current_token=self.token_count,
            candidate_pool=candidate_pool,
            strategy=sampling_strategy,
        )
        if not replay_idx:
            return 0

        applied = 0
        updated_ids = []
        processed_indices: list[int] = []

        for idx in replay_idx:
            replay_entry = self.model.memory_store.replay_entry(idx, current_token=self.token_count)
            assembly = replay_entry["assembly"]
            input_pattern = replay_entry["input_pattern"]
            stored_routing_key = replay_entry["routing_key"]
            stored_bucket_id = replay_entry["bucket_id"]
            importance_value = replay_entry["importance"]
            tag_strength_value = replay_entry.get("capture_tag", replay_entry.get("tag_strength", 0.0))
            prp_level_value = replay_entry.get("prp_level", 0.0)
            capture_strength_value = replay_entry.get("capture_strength", 0.0)
            consolidation_value = replay_entry.get("consolidation_level", 0.0)
            replay_importance = float(importance_value) if isinstance(importance_value, (int, float)) else 0.0
            replay_tag_strength = float(tag_strength_value) if isinstance(tag_strength_value, (int, float)) else 0.0
            replay_prp_level = float(prp_level_value) if isinstance(prp_level_value, (int, float)) else 0.0
            replay_capture_strength = float(capture_strength_value) if isinstance(capture_strength_value, (int, float)) else 0.0
            replay_consolidation = float(consolidation_value) if isinstance(consolidation_value, (int, float)) else 0.0

            if not isinstance(assembly, torch.Tensor):
                continue
            assembly = assembly.to(self.model.device)
            if assembly.dim() != 1:
                continue
            if float(assembly.abs().sum().item()) <= 0.0:
                continue

            if isinstance(input_pattern, torch.Tensor):
                replay_input = input_pattern.to(self.model.device)
                self.model.competitive.assembly_from_input(replay_input)
            else:
                replay_input = None
                self.model.competitive.last_input_pattern = None

            if isinstance(stored_routing_key, torch.Tensor):
                routing_key = F.normalize(stored_routing_key.to(self.model.device), dim=0)
            elif replay_input is not None:
                routing_key = self.model.routing_key_from_pattern(replay_input)
            else:
                routing_key = torch.mv(self.model._W_assembly_project_t, assembly)
                routing_key = F.normalize(routing_key, dim=0)

            context_prediction, context_gain = self._context_prediction_and_gain()

            if replay_use_stored_bucket and stored_bucket_id is not None:
                winner = torch.tensor([int(stored_bucket_id)], device=self.model.device)
            else:
                candidate_ids, _ = self.model.hnsw_index.search(
                    routing_key.unsqueeze(0),
                    k=self.config.k_routing,
                )
                candidates = (
                    torch.tensor(candidate_ids[0], device=self.model.device)
                    if candidate_ids and candidate_ids[0]
                    else None
                )
                winner, _, _ = self.model.competitive.compete(
                    routing_key,
                    candidates,
                    fallback_allowed=True,
                    context_gain=context_gain,
                )

            if mode == "repair":
                self._repair_column_from_replay(
                    int(winner.item()),
                    routing_key,
                    strength=repair_anchor_strength,
                )
                updated_ids.append(int(winner.item()))
                processed_indices.append(int(idx))
                applied += 1
                continue

            replay_priority = (
                replay_importance
                + replay_capture_strength
                + 0.40 * replay_prp_level
                + 0.30 * replay_tag_strength
                + max(0.0, 1.0 - replay_consolidation)
            )
            replay_modulator = modulator * (1.0 + min(2.0, replay_priority))
            replay_local_trace = self._local_trace_from_raw_window(
                replay_entry.get("raw_window"),
                context_confidence=max(
                    0.0,
                    min(
                        1.0,
                        0.25 * replay_importance
                        + 0.25 * replay_tag_strength
                        + 0.25 * replay_prp_level
                        + 0.25 * replay_consolidation,
                    ),
                ),
            )

            self.model.competitive.process(
                routing_key,
                winner,
                modulator=replay_modulator,
                eligibility_trace=replay_local_trace,
                assembly_projection=self.model.W_assembly_project,
                prototype_lr_scale=prototype_lr_scale,
                input_lr_scale=input_lr_scale,
                update_global_state=False,
            )
            self.model._invalidate_projection_cache()
            if mode == "deep":
                self._apply_column_anchors(
                    [int(winner.item())],
                    blend_scale=anchor_blend_scale,
                    blend_cap=anchor_blend_cap,
                )
            if self.model.context_layer is not None and mode == "deep":
                replay_assembly = self.model.competitive.winner_assembly(routing_key, winner)
                replay_assembly, _ = self._apply_binding(
                    replay_assembly,
                    context_prediction,
                    update_weights=False,
                )
                self.model.context_layer.observe(
                    replay_assembly,
                    update_weights=False,
                    precision_weight=self._context_precision_weight(),
                )
            if mode == "deep":
                updated_ids.append(int(winner.item()))
                revived_ids = self.model.competitive.last_revived_indices.detach().cpu().tolist()
                updated_ids.extend(int(idx) for idx in revived_ids)
            processed_indices.append(int(idx))
            applied += 1

        if applied > 0:
            if mode == "deep":
                # HNSW approximate index rebuild AFTER consolidation loop (§4.8).
                # Avoids stale-cell issue: prototype positions shift during
                # anchor_lr updates above; rebuilding now uses final positions.
                uniq = sorted(set(updated_ids))
                id_arr = np.asarray(uniq, dtype=np.int64)
                vecs = self.model.competitive.prototypes[id_arr].detach()
                self.model.hnsw_index.add(vecs, id_arr)
                self.model.hnsw_index.rebuild()
                self.model.memory_store.consolidate_replay(
                    processed_indices,
                    current_token=self.token_count,
                    blend=memory_blend,
                    protein_synthesis_level=protein_synthesis_level,
                )
            elif mode == "micro":
                self.model.memory_store.refresh_maintenance(
                    processed_indices,
                    current_token=self.token_count,
                )
            else:
                uniq = sorted(set(updated_ids))
                id_arr = np.asarray(uniq, dtype=np.int64)
                vecs = self.model.competitive.prototypes[id_arr].detach()
                self.model.hnsw_index.add(vecs, id_arr)
                self.model.hnsw_index.rebuild()
                self.model.memory_store.mark_repair_replay(
                    processed_indices,
                    current_token=self.token_count,
                )

            self.sleep_events += 1
            if mode == "micro":
                self.micro_sleep_events += 1
                self.last_micro_sleep_token = self.token_count
            else:
                self.deep_sleep_events += 1
                self.last_deep_sleep_token = self.token_count

        # SFA correction during deep sleep (§4.8)
        if mode == "deep" and applied > 0 and self.model.abstraction_layer is not None:
            sfa_samples = self.model.memory_store.sample_for_sfa(
                n=min(100, max(10, applied)),
            )
            if len(sfa_samples) >= 2:
                self.model.abstraction_layer.sfa_correction_step(
                    sfa_samples,
                    lr=0.01,
                )

        # Dead column census during deep sleep (§4.9)
        if mode == "deep":
            comp = self.model.competitive
            dead_mask = comp.steps_since_win >= comp.dead_column_steps
            n_dead = int(dead_mask.sum().item())
            n_total = comp.n_columns
            dead_pct = 100.0 * n_dead / max(1, n_total)
            self._dead_column_census = {
                "n_dead": n_dead,
                "n_total": n_total,
                "dead_pct": round(dead_pct, 1),
            }
            # Only revive when ≥5% of columns are dead (avoids noise in small nets)
            if n_dead > 0 and dead_pct >= 5.0:
                revived = comp.force_revive_dead_columns()
                self._dead_column_census["revived"] = revived

        # Adaptive timescale update during deep sleep (§4.3)
        if mode == "deep" and isinstance(self.model.context_layer, AdaptiveContextLayer):
            rd = self.model.context_layer.compute_routing_differentiation()
            if rd.sum().item() > 0:
                self.model.context_layer.update_timescales(rd)

        return applied

    @torch.no_grad()
    def train_step(
        self,
        pattern_vec: torch.Tensor,
        raw_window: Optional[str] = None,
        visual_spikes: Optional[torch.Tensor] = None,
        audio_spikes: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        x = pattern_vec.to(self.model.device)
        context_gain = None
        context_prediction = None
        binding_strength = 0.0
        _telemetry_tick = (self.token_count % 10 == 0)

        # Drift computation is expensive; only recompute every N steps
        if self.token_count % 50 == 0 or self._cached_drift is None:
            drift_bucket = self.last_winner if self.config.use_winner_local_drift else None
            drift = self.model.memory_store.compute_drift(drift_bucket)
            self._cached_drift = drift
            floor_rising = self._update_rolling_drift_floor(drift)
        else:
            drift = self._cached_drift
            floor_rising = False
        sleep_type = "none"
        replay_updates = 0
        deep_sleep_emergency = False

        deep_due_interval = (
            self.token_count >= self.config.deep_sleep_interval_tokens
            and (self.token_count - self.last_deep_sleep_token) >= self.config.deep_sleep_interval_tokens
        )
        deep_due_emergency = (
            self.pending_emergency_deep_sleep
            and (self.token_count - self.last_deep_sleep_token) >= self.config.emergency_deep_sleep_cooldown_tokens
        )
        if deep_due_interval or deep_due_emergency:
            self._flush_hnsw_buffer()
            replay_updates = self._sleep_replay("repair" if deep_due_emergency else "deep")
            if replay_updates > 0:
                sleep_type = "deep"
                deep_sleep_emergency = bool(deep_due_emergency)
                if deep_due_emergency:
                    self.pending_emergency_deep_sleep = False
        elif (
            self.token_count >= self.config.micro_sleep_interval_tokens
            and (self.token_count - self.last_micro_sleep_token) >= self.config.micro_sleep_interval_tokens
        ):
            self._flush_hnsw_buffer()
            replay_updates = self._sleep_replay("micro")
            if replay_updates > 0:
                sleep_type = "micro"
        elif self.token_count % self._hnsw_flush_interval == 0:
            self._flush_hnsw_buffer()

        sleep_triggered = sleep_type != "none"
        if sleep_triggered:
            drift_bucket = self.last_winner if self.config.use_winner_local_drift else None
            drift = self.model.memory_store.compute_drift(drift_bucket)
            self._refresh_latest_drift(drift)

        metrics["drift"] = drift
        metrics["sleep_triggered"] = int(sleep_triggered)
        metrics["sleep_type"] = sleep_type
        metrics["sleep_replay_updates"] = int(replay_updates)
        metrics["sleep_events_total"] = int(self.sleep_events)
        metrics["micro_sleep_events_total"] = int(self.micro_sleep_events)
        metrics["deep_sleep_events_total"] = int(self.deep_sleep_events)
        metrics["deep_sleep_emergency"] = int(deep_sleep_emergency)
        metrics["drift_floor"] = float(self.current_rolling_drift_floor if self.current_rolling_drift_floor is not None else drift)
        metrics["drift_floor_rising"] = int(floor_rising)
        if self._dead_column_census:
            metrics["dead_column_census"] = self._dead_column_census

        if self.token_count < self.config.bootstrap_tokens:
            pred_error = self.bootstrap.update(x)
            self.model.surprise.update_neuromodulators(current_error=pred_error, novelty=min(1.0, pred_error))
            modulator = self.model.surprise.get_modulator("competitive")
            metrics["pred_error"] = pred_error
        else:
            self.is_bootstrap = False
            modulator = self.model.surprise.get_modulator("competitive")

        routing_key = self.model.routing_key_from_pattern(x)
        recon_error = self.model.competitive.nearest_prototype_distance(routing_key)
        metrics["recon_error"] = float(recon_error)

        # Post-bootstrap neuromodulator update using reconstruction error as the error signal.
        # During bootstrap the update is driven by the linear predictor; post-bootstrap we use
        # the prototype reconstruction error (same biological role: discrepancy → norepinephrine).
        if self.token_count >= self.config.bootstrap_tokens:
            self.model.surprise.update_neuromodulators(current_error=recon_error, novelty=min(1.0, recon_error))

        # If sustained norepinephrine exceeds the exploration threshold and the cooldown
        # has elapsed, boost exploration noise and curiosity urgency.  Do NOT reset learned
        # state — destroying consolidated knowledge at the moment of highest uncertainty
        # is counterproductive.  Dead column revival is handled during deep sleep census.
        exploration_boosted = False
        reset_cooldown = self.config.emergency_deep_sleep_cooldown_tokens
        if (
            self.model.surprise.should_boost_exploration()
            and (self.token_count - self.last_network_reset_token) >= reset_cooldown
        ):
            self._exploration_noise_scale = min(2.0, self._exploration_noise_scale * 1.5)
            self.last_network_reset_token = self.token_count
            exploration_boosted = True
        else:
            # Gradually decay exploration noise back to baseline
            self._exploration_noise_scale = max(1.0, self._exploration_noise_scale * 0.99)

        local_trace = self._local_trace_from_raw_window(
            raw_window,
            context_confidence=max(0.0, min(1.0, 1.0 - recon_error)),
        )

        context_prediction, context_gain = self._context_prediction_and_gain()

        # Curiosity-driven routing bias (§3.1): boost columns for uncertain concepts
        if self.model.abstraction_layer is not None:
            curiosity_gain = self.model.abstraction_layer.curiosity_routing_gain()
            if curiosity_gain is not None:
                if context_gain is not None:
                    context_gain = torch.clamp(context_gain * curiosity_gain, min=0.5, max=1.5)
                else:
                    context_gain = curiosity_gain

        candidate_ids, _ = self.model.hnsw_index.search(
            routing_key.unsqueeze(0),
            k=self.config.k_routing,
        )
        candidates = (
            torch.tensor(candidate_ids[0], device=self.model.device)
            if candidate_ids and candidate_ids[0]
            else None
        )

        winners, strengths, _ = self.model.competitive.compete(
            routing_key,
            candidates,
            fallback_allowed=self.is_bootstrap,
            context_gain=context_gain,
        )

        # Extract winner IDs once — avoid repeated .item() sync
        winner_id_list = winners.tolist()
        winner_id = int(winner_id_list[0])

        winner_consolidation = 0.0
        if self.memory_warm_started:
            winner_levels = [
                self.model.memory_store.bucket_consolidation_level(wid)
                for wid in winner_id_list
            ]
            if winner_levels:
                winner_consolidation = float(sum(winner_levels) / len(winner_levels))

        # Neuromodulator-specific plasticity gates (§4.5, §4.7):
        #   DA (dopamine): scales LTP gain — positive RPE amplifies learning
        #   5-HT (serotonin): modulates consolidation patience — high 5-HT
        #     suppresses plasticity on consolidated winners (patience gate)
        #   NE: wired to exploration boost via should_boost_exploration()
        #   ACh: already wired to context precision via _context_precision_weight()
        da = self.model.surprise.dopamine
        ht = self.model.surprise.serotonin
        da_ltp_gain = 0.8 + 0.4 * da  # range [0.8, 1.2]: DA amplifies learning
        ht_patience = max(0.2, 1.0 - 0.6 * ht)  # high 5-HT → reduced wake plasticity
        wake_plasticity_scale = max(0.2, 1.0 - 0.8 * winner_consolidation) * ht_patience
        effective_modulator = float(modulator) * wake_plasticity_scale * da_ltp_gain

        assembly = self.model.competitive.process(
            routing_key,
            winners,
            effective_modulator,
            winner_strengths=strengths,
            eligibility_trace=local_trace,
            assembly_projection=self.model.W_assembly_project,
            compute_metrics=_telemetry_tick,
        )
        # Refresh transpose cache since process() modifies W_assembly_project in-place
        self.model._invalidate_projection_cache()
        abstraction_input = assembly.clone()
        if self.model.abstraction_layer is not None:
            self.model.abstraction_layer.observe(
                abstraction_input,
                update_weights=True,
                precision_weight=self._context_precision_weight(),
            )
            # Top-down boundary bias: Abstraction Layer → Chunking Layer (§3.1)
            if getattr(self.encoder, "uses_learned_chunking", False):
                max_gap = self.model.abstraction_layer.max_curiosity_gap_score()
                mean_cert = float(self.model.abstraction_layer.concept_certainty.mean().item())
                self.encoder.learned_chunking.set_abstraction_bias(mean_cert, max_gap)
        assembly, binding_strength = self._apply_binding(
            assembly,
            context_prediction,
            update_weights=True,
        )
        self._apply_column_anchors(wid for wid in winner_id_list)
        if self.model.context_layer is not None:
            self.model.context_layer.observe(
                assembly,
                update_weights=True,
                precision_weight=self._context_precision_weight(),
            )

        # Cross-modal grounding updates (§5): sensory spikes fire BEFORE
        # text so that on_text_spike() sees the current visual/audio traces
        # (not stale traces from a previous concept).  Stage-aware gating
        # (§7): Stage 1 accepts all pairs; Stage 2+ applies alignment_gate()
        # with a bootstrap budget so confidence can build from zero.
        cross_modal_visual_conf = 0.0
        cross_modal_audio_conf = 0.0
        cross_modal_visual_accepted = None
        cross_modal_audio_accepted = None
        if self.model.cross_modal is not None:
            # Use L2-normalized raw pattern as text representation for
            # cross-modal Hebbian learning.  With hashed_ngram encoding
            # different words are nearly orthogonal, giving each word its
            # own cross-modal association without column contamination.
            text_spike = F.normalize(x.detach().unsqueeze(0), dim=1).squeeze(0)

            # Visual path — fire BEFORE text
            if visual_spikes is not None:
                vs = visual_spikes.to(self.model.device)
                accept_visual = True
                if self.developmental_stage >= 2:
                    if self._stage2_bootstrap_used_visual < self._stage2_bootstrap_budget:
                        self._stage2_bootstrap_used_visual += 1
                    else:
                        accept_visual, _vscore = self.model.cross_modal.alignment_gate(
                            text_spike, vs,
                        )
                cross_modal_visual_accepted = accept_visual
                if accept_visual:
                    self.model.cross_modal.on_visual_spike(vs)

            # Audio path — fire BEFORE text
            if audio_spikes is not None:
                aus = audio_spikes.to(self.model.device)
                accept_audio = True
                if self.developmental_stage >= 2:
                    if self._stage2_bootstrap_used_audio < self._stage2_bootstrap_budget:
                        self._stage2_bootstrap_used_audio += 1
                    else:
                        accept_audio, _ascore = self.model.cross_modal.alignment_gate_audio(
                            text_spike, aus,
                        )
                cross_modal_audio_accepted = accept_audio
                if accept_audio:
                    self.model.cross_modal.on_audio_spike(aus)

            # Text spike LAST — now visual/audio traces contain current data
            self.model.cross_modal.on_text_spike(text_spike)

            # Cross-modal confidence — periodic (every 10 steps) for metrics only
            if _telemetry_tick:
                cross_modal_visual_conf = float(self.model.cross_modal.visual_confidence.mean().item())
                cross_modal_audio_conf = float(self.model.cross_modal.audio_confidence.mean().item())
                self._cached_cross_modal_conf = (cross_modal_visual_conf, cross_modal_audio_conf)
            elif hasattr(self, "_cached_cross_modal_conf"):
                cross_modal_visual_conf, cross_modal_audio_conf = self._cached_cross_modal_conf

            # Buffer visual frames for self-criticism (§7.4)
            if visual_spikes is not None and cross_modal_visual_accepted:
                self._recent_visual_frames.append(vs.detach().clone())
                if len(self._recent_visual_frames) > self._visual_frame_limit:
                    self._recent_visual_frames = self._recent_visual_frames[-self._visual_frame_limit:]

            # Buffer audio frames for audio self-criticism (§7.4)
            if audio_spikes is not None and cross_modal_audio_accepted:
                self._recent_audio_frames.append(aus.detach().clone())
                if len(self._recent_audio_frames) > self._audio_frame_limit:
                    self._recent_audio_frames = self._recent_audio_frames[-self._audio_frame_limit:]

            # Periodic self-criticism loop (§7.4) — visual AND audio
            n_visual = len(self._recent_visual_frames)
            n_audio = len(self._recent_audio_frames)
            if (self.token_count - self._last_self_criticism_token >= self._self_criticism_interval
                    and (n_visual >= 3 or n_audio >= 3)):
                early_stage = n_visual < 10
                sc_checked = 0
                sc_penalised = 0
                if n_visual >= 3:
                    sc_vis = self.model.cross_modal.run_self_criticism(
                        recent_visual_frames=self._recent_visual_frames,
                        blacklist=self._self_criticism_blacklist,
                        penalty=0.05 if early_stage else 0.10,
                        blacklist_strikes=3 if early_stage else 2,
                    )
                    sc_checked += sc_vis["checked"]
                    sc_penalised += sc_vis["penalised"]
                if n_audio >= 3:
                    sc_aud = self.model.cross_modal.run_self_criticism_audio(
                        recent_audio_frames=self._recent_audio_frames,
                        blacklist=self._self_criticism_audio_blacklist,
                        penalty=0.05 if early_stage else 0.10,
                        blacklist_strikes=3 if early_stage else 2,
                    )
                    sc_checked += sc_aud["checked"]
                    sc_penalised += sc_aud["penalised"]
                if sc_checked > 0:
                    self._self_criticism_history.append({
                        "checked": sc_checked,
                        "penalised": sc_penalised,
                        "token": self.token_count,
                    })
                self._last_self_criticism_token = self.token_count

        updated_indices = winners
        if int(self.model.competitive.last_revived_indices.numel()) > 0:
            updated_indices = torch.unique(
                torch.cat([winners, self.model.competitive.last_revived_indices.to(self.model.device)]),
                sorted=True,
            )
        winner_vectors = self.model.competitive.prototypes[updated_indices].detach()
        self._buffer_hnsw_update(updated_indices, winner_vectors)

        next_token = self.token_count + 1
        warm_started = self._maybe_warm_start_memory(next_token)
        capture_tag = max(0.0, float(recon_error))
        text_context = self._update_stream_text(raw_window)
        memory_index = None
        if self.memory_warm_started:
            memory_index = self.model.memory_store.update(
                assembly,
                importance=max(1e-3, abs(effective_modulator)),
                token_count=next_token,
                bucket_id=winner_id,
                input_pattern=x,
                routing_key=routing_key,
                raw_window=raw_window,
                text=text_context,
                capture_tag=capture_tag,
            )
        self.last_winner = winner_id

        # Surprise update from reconstruction mismatch proxy.
        winner_proto = self.model.competitive.prototypes[winners[0]]
        self.model.surprise.update("competitive", winner_proto, routing_key)

        self.token_count = next_token
        if warm_started:
            drift_bucket = self.last_winner if self.config.use_winner_local_drift else None
            drift = self.model.memory_store.compute_drift(drift_bucket)
            metrics["drift"] = drift
            metrics["drift_floor"] = drift
        self.current_window_min_drift = min(self.current_window_min_drift, float(drift))
        if self.token_count % self.config.drift_floor_window_tokens == 0:
            self._close_drift_floor_window()

        memory_stats = (
            self.model.memory_store.summary_stats()
            if self.memory_warm_started and _telemetry_tick
            else self._cached_memory_stats if hasattr(self, "_cached_memory_stats") else {}
        )
        if _telemetry_tick:
            self._cached_memory_stats = memory_stats

        metrics["token"] = self.token_count
        metrics["surprise"] = float(modulator)
        metrics["dopamine"] = float(self.model.surprise.dopamine)
        metrics["serotonin"] = float(self.model.surprise.serotonin)
        metrics["acetylcholine"] = float(self.model.surprise.acetylcholine)
        metrics["norepinephrine"] = float(self.model.surprise.norepinephrine)
        metrics["exploration_boosted"] = int(exploration_boosted)
        metrics["exploration_noise_scale"] = self._exploration_noise_scale
        metrics["plasticity_mode"] = str(self.config.plasticity_mode)
        metrics["plasticity_spike_backend"] = (
            str(self.model.competitive.local_plasticity.spike_backend)
            if self.model.competitive.local_plasticity is not None
            else "proxy"
        )
        metrics["local_trace_available"] = int(local_trace is not None)
        if _telemetry_tick:
            _lt_active = int((local_trace > 0).sum().item()) if local_trace is not None else 0
            self._cached_lt_active = _lt_active
        metrics["local_trace_active_inputs"] = getattr(self, "_cached_lt_active", 0)
        metrics["local_post_spike_fraction"] = (
            float(self.model.competitive.local_plasticity.last_post_spike_fraction)
            if self.model.competitive.local_plasticity is not None
            else 0.0
        )
        metrics["local_mean_membrane_voltage"] = (
            float(self.model.competitive.local_plasticity.last_mean_membrane_voltage)
            if self.model.competitive.local_plasticity is not None
            else 0.0
        )
        metrics["capture_tag"] = float(capture_tag)
        metrics["mean_memory_capture_tag"] = float(memory_stats.get("mean_capture_tag", 0.0))
        metrics["mean_memory_prp_level"] = float(memory_stats.get("mean_prp_level", 0.0))
        metrics["mean_memory_capture_strength"] = float(memory_stats.get("mean_capture_strength", 0.0))
        metrics["mean_memory_consolidation_level"] = float(memory_stats.get("mean_consolidation_level", 0.0))
        metrics["mean_memory_fragility"] = float(memory_stats.get("mean_fragility", 0.0))
        metrics["winner_consolidation_level"] = float(winner_consolidation)
        metrics["effective_modulator"] = float(effective_modulator)
        metrics["da_ltp_gain"] = float(da_ltp_gain)
        metrics["ht_patience_gate"] = float(ht_patience)
        if _telemetry_tick:
            _ctx_str = float(context_prediction.sum().item()) if isinstance(context_prediction, torch.Tensor) else 0.0
            _ctx_gain = float(context_gain.mean().item()) if isinstance(context_gain, torch.Tensor) else 1.0
            self._cached_ctx_metrics = (_ctx_str, _ctx_gain)
        elif not hasattr(self, "_cached_ctx_metrics"):
            self._cached_ctx_metrics = (0.0, 1.0)
        metrics["context_strength"], metrics["context_gain_mean"] = self._cached_ctx_metrics
        metrics["context_precision_weight"] = (
            float(self.model.context_layer.last_precision_weight)
            if self.model.context_layer is not None
            else 1.0
        )
        if self.model.abstraction_layer is not None and _telemetry_tick:
            _abs_stab = float(self.model.abstraction_layer.concept_stability.mean().item())
            _abs_cert = float(self.model.abstraction_layer.concept_certainty.mean().item())
            _abs_gain = float(self.model.abstraction_layer.routing_gain().mean().item())
            _abs_gap = self.model.abstraction_layer.max_curiosity_gap_score()
            self._cached_abs_metrics = (_abs_stab, _abs_cert, _abs_gain, _abs_gap)
        elif not hasattr(self, "_cached_abs_metrics"):
            self._cached_abs_metrics = (0.0, 0.0, 1.0, 0.0)
        _abs_stab, _abs_cert, _abs_gain, _abs_gap = self._cached_abs_metrics
        metrics["abstraction_stability_mean"] = _abs_stab
        metrics["abstraction_certainty_mean"] = _abs_cert
        metrics["abstraction_gain_mean"] = _abs_gain
        metrics["abstraction_gap_score_max"] = _abs_gap
        metrics["binding_strength"] = float(binding_strength)
        metrics["cross_modal_visual_confidence"] = cross_modal_visual_conf
        metrics["cross_modal_audio_confidence"] = cross_modal_audio_conf
        metrics["cross_modal_visual_accepted"] = cross_modal_visual_accepted
        metrics["cross_modal_audio_accepted"] = cross_modal_audio_accepted
        metrics["developmental_stage"] = self.developmental_stage
        metrics["winner"] = winner_id
        if _telemetry_tick:
            _active = int((assembly > 0).sum().item())
            _sparsity = float((assembly > 0).float().mean().item())
            self._cached_active_sparsity = (_active, _sparsity)
        elif not hasattr(self, "_cached_active_sparsity"):
            self._cached_active_sparsity = (0, 0.0)
        metrics["active_columns"], metrics["sparsity"] = self._cached_active_sparsity
        metrics["memory_index"] = None if memory_index is None else int(memory_index)
        return metrics

    def reconstruction_error(self, pattern_vec: torch.Tensor) -> float:
        """Distance to nearest prototype for a single pattern."""
        routing_key = self.model.routing_key_from_pattern(pattern_vec)
        return self.model.competitive.nearest_prototype_distance(routing_key)

    def assembly_for_pattern(self, pattern_vec: torch.Tensor) -> torch.Tensor:
        x = pattern_vec.to(self.model.device)
        return self.model.competitive.assembly_from_input(x).detach().cpu()

    def reset_context_state(self) -> None:
        if self.model.context_layer is not None:
            self.model.context_layer.reset_state()
        if self.model.abstraction_layer is not None:
            self.model.abstraction_layer.reset_state()
        if self.model.binding_layer is not None:
            self.model.binding_layer.reset_state()

    def context_state(self) -> torch.Tensor:
        if self.model.context_layer is None:
            return torch.zeros(self.config.n_columns)
        return self.model.context_layer.state.detach().cpu()

    def _offline_context_signature(
        self,
        pattern_vec: torch.Tensor,
        *,
        blend_context_state: bool = False,
        readout_mode: str = "softmax",
    ) -> torch.Tensor:
        x = pattern_vec.to(self.model.device)
        routing_key = self.model.routing_key_from_pattern(x)
        candidate_ids, _ = self.model.hnsw_index.search(
            routing_key.unsqueeze(0),
            k=self.config.k_routing,
        )
        candidates = (
            torch.tensor(candidate_ids[0], device=self.model.device)
            if candidate_ids and candidate_ids[0]
            else torch.arange(self.config.n_columns, device=self.model.device)
        )

        proto = self.model.competitive.prototypes[candidates]
        sim = torch.mv(proto, F.normalize(routing_key, dim=0))
        drive = self.model.competitive._input_drive(self.model.competitive.last_input_pattern, candidates)
        input_blend = self._offline_input_blend()
        combined = (1.0 - input_blend) * sim + input_blend * drive

        context_source, context_gain = self._offline_context_source_and_gain(
            blend_state=blend_context_state,
        )
        if context_gain is not None:
            combined = combined * torch.clamp(context_gain[candidates], min=0.5, max=1.5)

        signature = torch.zeros(self.config.n_columns, device=self.model.device)
        if int(candidates.numel()) <= 0:
            return signature

        if readout_mode == "relu":
            values = torch.relu(combined)
            if float(values.sum().item()) <= 0.0:
                winner_idx = int(candidates[torch.argmax(combined)].item())
                signature[winner_idx] = 1.0
            else:
                signature[candidates] = values / (values.sum() + 1e-8)
        else:
            signature[candidates] = torch.softmax(combined, dim=0)
        signature, _ = self._apply_binding(signature, context_source, update_weights=False)
        total = float(signature.sum().item())
        if total <= 0.0:
            winner_idx = int(candidates[torch.argmax(combined)].item())
            signature = torch.zeros(self.config.n_columns, device=self.model.device)
            signature[winner_idx] = 1.0
            return signature
        return signature / (signature.sum() + 1e-8)

    def _offline_input_blend(self) -> float:
        blend = float(self.model.competitive.input_weight_blend)
        if getattr(self.encoder, "uses_learned_chunking", False):
            return max(blend, float(self.config.learned_chunk_query_blend_floor))
        return blend

    def _offline_competition(self, pattern_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = pattern_vec.to(self.model.device)
        routing_key = self.model.routing_key_from_pattern(x)
        candidate_ids, _ = self.model.hnsw_index.search(
            routing_key.unsqueeze(0),
            k=self.config.k_routing,
        )
        candidates = (
            torch.tensor(candidate_ids[0], device=self.model.device)
            if candidate_ids and candidate_ids[0]
            else None
        )
        context_gain = None
        context_prediction, context_gain = self._context_prediction_and_gain()
        original_input_blend = float(self.model.competitive.input_weight_blend)
        self.model.competitive.input_weight_blend = self._offline_input_blend()
        try:
            winners, _, _ = self.model.competitive.compete(
                routing_key,
                candidates,
                fallback_allowed=True,
                context_gain=context_gain,
            )
        finally:
            self.model.competitive.input_weight_blend = original_input_blend
        assembly = self.model.competitive.winner_assembly(routing_key, winners)
        assembly, _ = self._apply_binding(assembly, context_prediction, update_weights=False)
        return winners, assembly

    def prime_context(self, patterns: Iterable[torch.Tensor], update_weights: bool = False) -> None:
        self.reset_context_state()
        if self.model.context_layer is None:
            return
        for pattern in patterns:
            _, assembly = self._offline_competition(pattern)
            self.model.context_layer.observe(assembly, update_weights=update_weights)

    def prime_context_with_signatures(
        self,
        patterns: Iterable[torch.Tensor],
        update_weights: bool = False,
        *,
        blend_context_state: bool = False,
        readout_mode: str = "softmax",
    ) -> None:
        self.reset_context_state()
        if self.model.context_layer is None:
            return
        for pattern in patterns:
            signature = self._offline_context_signature(
                pattern,
                blend_context_state=blend_context_state,
                readout_mode=readout_mode,
            )
            self.model.context_layer.observe(signature, update_weights=update_weights)

    def contextual_winner_for_pattern(self, pattern_vec: torch.Tensor) -> int:
        winners, _ = self._offline_competition(pattern_vec)
        return int(winners[0].item())

    def contextual_assembly_for_pattern(self, pattern_vec: torch.Tensor) -> torch.Tensor:
        _, assembly = self._offline_competition(pattern_vec)
        return assembly.detach().cpu()

    def contextual_signature_for_pattern(
        self,
        pattern_vec: torch.Tensor,
        *,
        blend_context_state: bool = False,
        readout_mode: str = "softmax",
    ) -> torch.Tensor:
        signature = self._offline_context_signature(
            pattern_vec,
            blend_context_state=blend_context_state,
            readout_mode=readout_mode,
        )
        return signature.detach().cpu()

    def run_sleep_maintenance(self, mode: str = "deep", cycles: int = 1) -> int:
        total_updates = 0
        for _ in range(max(0, int(cycles))):
            total_updates += self._sleep_replay(mode)
        return total_updates

    def tag_recent_memories(self, window_tokens: int, strength: float) -> int:
        return self.model.memory_store.tag_recent_entries(
            current_token=self.token_count,
            window_tokens=window_tokens,
            strength=strength,
        )

    def capture_recent_memory_anchors(self, window_tokens: int, strength: float) -> int:
        if window_tokens <= 0:
            return 0

        floor_token = max(0, self.token_count - int(window_tokens))
        captured = 0
        for idx, token_marker in enumerate(self.model.memory_store.slow_entry_timestamps):
            if int(token_marker) < floor_token:
                continue
            bucket_id = self.model.memory_store.slow_bucket_ids[idx]
            if bucket_id is None:
                continue
            bucket = int(bucket_id)
            self.column_anchors[bucket] = {
                "prototype": self.model.competitive.prototypes[bucket].detach().clone(),
                "input_weights": self.model.competitive.input_weights[bucket].detach().clone(),
                "strength": float(max(0.0, strength)),
            }
            captured += 1
        return len(self.column_anchors) if captured > 0 else 0

    def winner_for_pattern(self, pattern_vec: torch.Tensor) -> int:
        """Deterministic offline winner used for evaluation and query readout."""
        winners, _ = self._offline_competition(pattern_vec)
        return int(winners[0].item())

    def routing_key_for_pattern(self, pattern_vec: torch.Tensor) -> torch.Tensor:
        """Expose routing key generation for clustering diagnostics."""
        return self.model.routing_key_from_pattern(pattern_vec)

    def self_criticism_find_rate(self, last_n: int = 5) -> float:
        """Compute self-criticism find-rate over the last N cycles (§7.3 criterion 2).

        Returns fraction of checked groundings that were penalised.
        If no cycles have run or none had checked > 0, returns 0.0.
        """
        recent = self._self_criticism_history[-last_n:]
        if not recent:
            return 0.0
        total_checked = sum(e["checked"] for e in recent)
        total_penalised = sum(e["penalised"] for e in recent)
        if total_checked == 0:
            return 0.0
        return total_penalised / total_checked

    def train_stream(self, stream: Iterable[torch.Tensor]) -> Iterable[Dict[str, Any]]:
        for vec in stream:
            yield self.train_step(vec)
