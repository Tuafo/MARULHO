from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import torch


@dataclass
class HECSNConfig:
    """Configuration with explicit Stage-0 defaults and derived constraints."""

    n_ascii: int = 128
    window_size: int = 10
    input_representation: Literal["order_weighted_ascii", "unigram_ascii", "hashed_ngram"] = "order_weighted_ascii"
    context_mode: Literal["fixed", "adaptive"] = "adaptive"
    plasticity_mode: Literal["lite", "local_stdp"] = "lite"
    plasticity_rule: Literal["pair", "triplet"] = "triplet"
    plasticity_spike_backend: Literal["proxy", "adex"] = "proxy"
    spike_trace_tau: float = 6.0
    spike_burst_decay: float = 0.85
    stdp_trace_tau: float = 20.0
    stdp_eligibility_tau: float = 200.0
    stdp_mu_plus: float = 0.0
    stdp_mu_minus: float = 1.0
    # Triplet STDP parameters (Pfister & Gerstner 2006, hippocampal fit)
    triplet_tau_plus: float = 16.8
    triplet_tau_minus: float = 33.7
    triplet_tau_x: float = 101.0
    triplet_tau_y: float = 114.0
    triplet_A2_plus: float = 5e-10
    triplet_A2_minus: float = 7e-3
    triplet_A3_plus: float = 6.2e-3
    triplet_A3_minus: float = 2.3e-4
    synaptic_scaling_alpha: float = 0.1
    inhibitory_plasticity_lr: float = 0.05
    inhibitory_decay: float = 0.95
    projection_plasticity_scale: float = 0.35
    assembly_projection_plasticity_scale: float = 0.25
    projection_norm_target: float = 0.1
    hashed_ngram_dim: int = 2048
    hashed_ngram_min_n: int = 2
    hashed_ngram_max_n: int = 3
    enable_learned_chunking: bool = True
    learned_chunk_detector_count: int = 128
    learned_chunk_min_len: int = 2
    learned_chunk_max_len: int = 12
    learned_chunk_feature_mode: Literal["blend", "concat"] = "blend"
    learned_chunk_concat_dim: int = 128
    learned_chunk_blend: float = 0.5
    learned_chunk_similarity_floor: float = 0.30
    learned_chunk_boundary_threshold: float = 0.08
    learned_chunk_update_lr: float = 0.25
    learned_chunk_query_blend_floor: float = 0.15
    learned_chunk_association_blend: float = 0.35
    learned_chunk_association_lr: float = 0.15
    learned_chunk_association_decay: float = 0.995

    n_columns: int = 10
    column_latent_dim: int = 256

    bootstrap_tokens: int = 5000
    k_routing: int = 10
    index_rebuild_threshold: int = 256
    routing_index_mode: Literal["auto", "faiss_hnsw", "torch_topk", "exact_cosine"] = "auto"
    routing_shards: int = 1
    shard_candidate_factor: int = 2

    eta_competitive: float = 0.01
    eta_decay: float = 1e-6
    input_weight_blend: float = 0.02
    input_synapse_ltp: float = 0.02
    input_synapse_ltd: float = 0.01
    input_weight_row_target: float = 1.0

    homeostasis_beta: float = 0.01
    homeostasis_lr: float = 0.2
    threshold_min: float = 0.05
    threshold_max: float = 1.5
    dead_column_steps: int = 2000
    dead_column_noise: float = 0.05

    silhouette_min: float = 0.20
    davies_bouldin_max: float = 1.5
    recon_slope_max: float = 0.0
    winner_entropy_min_bits: float = 1.5

    memory_capacity: int = 1000
    ema_alpha: float = 0.01
    slow_mean_decay: float = 1.0
    stc_tag_decay: float = 0.985
    stc_capture_release: float = 0.70
    stc_consolidation_rate: float = 1.0
    functional_minute: int = 500
    stc_tag_duration_weak: float = 30.0
    stc_tag_duration_strong: float = 120.0
    stc_prp_tau_weak: float = 60.0
    stc_prp_tau_strong: float = 240.0
    stc_prp_synthesis_rate: float = 0.18
    stc_prp_capture_threshold: float = 0.15
    stc_prp_consumption: float = 0.50
    stc_strong_event_threshold: float = 0.60
    slow_memory_start_tokens: int = 0
    use_winner_local_drift: bool = True

    drift_threshold: float = 0.02
    micro_sleep_interval_tokens: int = 200
    micro_sleep_replay_steps: int = 10
    micro_sleep_candidate_pool: int = 5
    micro_sleep_memory_blend: float = 0.05
    deep_sleep_interval_tokens: int = 2500  # Aligned with all presets (was 5000, causing silent divergence)
    deep_sleep_replay_steps: int = 100
    deep_sleep_candidate_pool: int = 100
    deep_sleep_memory_blend: float = 0.20
    deep_sleep_cooldown_tokens: int = 1000
    emergency_deep_sleep_cooldown_tokens: int = 1000
    drift_floor_history_tokens: int = 1000
    drift_floor_check_interval_tokens: int = 200
    drift_floor_window_tokens: int = 10000
    drift_floor_trigger_min_tokens: int = 1000
    drift_floor_rise_tolerance: float = 0.0
    prototype_momentum: float = 0.85

    enable_context_layer: bool = False
    context_decay: float = 0.92
    context_transition_lr: float = 0.05
    context_modulation_strength: float = 0.60
    context_fast_rate: float = 0.55
    context_medium_rate: float = 0.25
    context_slow_rate: float = 0.08
    context_recurrent_density: float = 0.35
    context_recurrent_scale: float = 0.85
    context_inhibition_strength: float = 0.25

    enable_abstraction_layer: bool = False
    abstraction_n_concepts: int = 8
    abstraction_slow_rate: float = 0.05
    abstraction_fast_rate: float = 0.30
    abstraction_learning_rate: float = 0.02
    abstraction_feedback_lr: float = 0.05
    abstraction_feedback_strength: float = 0.15

    enable_binding_layer: bool = False
    binding_threshold: float = 0.02
    binding_association_lr: float = 0.20
    binding_association_decay: float = 0.995
    binding_gain_strength: float = 0.80
    binding_n_bindings: int = 0
    binding_fan_in: int = 4
    binding_tau: float = 6.0
    binding_stp_u_inc: float = 0.15
    binding_stp_tau_f: float = 12.0
    binding_stp_tau_d: float = 4.0
    binding_pv_threshold: float = 0.12
    binding_pv_gain: float = 0.60

    enable_cross_modal: bool = False
    cross_modal_dim_visual: int = 256
    cross_modal_dim_audio: int = 64
    cross_modal_A_plus: float = 0.010
    cross_modal_A_minus: float = 0.012
    cross_modal_tau_trace: float = 10.0
    cross_modal_confidence_alpha: float = 0.01

    acquisition_concept_novelty_weight: float = 0.08  # Reduced from 0.20 to prevent cold-start uncertainty inflation
    acquisition_concept_uncertainty_weight: float = 0.10  # Reduced from 0.25 to prevent cold-start uncertainty inflation

    # Reporting-only assumption used to map column scale to neuron scale in summaries.
    neurons_per_column_assumption: int = 100

    device: str = "auto"

    input_dim: int = field(init=False)

    def __post_init__(self) -> None:
        valid_representations = {"order_weighted_ascii", "unigram_ascii", "hashed_ngram"}
        valid_plasticity_modes = {"lite", "local_stdp"}
        valid_spike_backends = {"proxy", "adex"}
        if self.input_representation not in valid_representations:
            raise ValueError(f"input_representation must be one of {sorted(valid_representations)}")
        if self.plasticity_mode not in valid_plasticity_modes:
            raise ValueError(f"plasticity_mode must be one of {sorted(valid_plasticity_modes)}")
        if self.plasticity_spike_backend not in valid_spike_backends:
            raise ValueError(f"plasticity_spike_backend must be one of {sorted(valid_spike_backends)}")
        if self.spike_trace_tau <= 0.0:
            raise ValueError("spike_trace_tau must be positive")
        if not 0.0 < self.spike_burst_decay <= 1.0:
            raise ValueError("spike_burst_decay must be in (0, 1]")
        if self.stdp_trace_tau <= 0.0:
            raise ValueError("stdp_trace_tau must be positive")
        if self.stdp_eligibility_tau <= 0.0:
            raise ValueError("stdp_eligibility_tau must be positive")
        if self.synaptic_scaling_alpha < 0.0:
            raise ValueError("synaptic_scaling_alpha must be non-negative")
        if self.inhibitory_plasticity_lr < 0.0:
            raise ValueError("inhibitory_plasticity_lr must be non-negative")
        if not 0.0 <= self.inhibitory_decay <= 1.0:
            raise ValueError("inhibitory_decay must be in [0, 1]")
        if self.projection_plasticity_scale < 0.0:
            raise ValueError("projection_plasticity_scale must be non-negative")
        if self.assembly_projection_plasticity_scale < 0.0:
            raise ValueError("assembly_projection_plasticity_scale must be non-negative")
        if self.projection_norm_target <= 0.0:
            raise ValueError("projection_norm_target must be positive")
        if self.hashed_ngram_dim <= 0:
            raise ValueError("hashed_ngram_dim must be positive")
        if self.hashed_ngram_min_n <= 0:
            raise ValueError("hashed_ngram_min_n must be positive")
        if self.hashed_ngram_max_n < self.hashed_ngram_min_n:
            raise ValueError("hashed_ngram_max_n must be greater than or equal to hashed_ngram_min_n")
        if self.learned_chunk_detector_count <= 0:
            raise ValueError("learned_chunk_detector_count must be positive")
        if self.learned_chunk_min_len <= 0:
            raise ValueError("learned_chunk_min_len must be positive")
        if self.learned_chunk_max_len < self.learned_chunk_min_len:
            raise ValueError("learned_chunk_max_len must be greater than or equal to learned_chunk_min_len")
        if self.learned_chunk_feature_mode not in {"blend", "concat"}:
            raise ValueError("learned_chunk_feature_mode must be one of blend or concat")
        if self.learned_chunk_concat_dim <= 0:
            raise ValueError("learned_chunk_concat_dim must be positive")
        if not 0.0 <= self.learned_chunk_blend <= 1.0:
            raise ValueError("learned_chunk_blend must be in [0, 1]")
        if not 0.0 <= self.learned_chunk_similarity_floor <= 1.0:
            raise ValueError("learned_chunk_similarity_floor must be in [0, 1]")
        if self.learned_chunk_boundary_threshold < 0.0:
            raise ValueError("learned_chunk_boundary_threshold must be non-negative")
        if not 0.0 < self.learned_chunk_update_lr <= 1.0:
            raise ValueError("learned_chunk_update_lr must be in (0, 1]")
        if not 0.0 <= self.learned_chunk_query_blend_floor <= 1.0:
            raise ValueError("learned_chunk_query_blend_floor must be in [0, 1]")
        if not 0.0 <= self.learned_chunk_association_blend <= 1.0:
            raise ValueError("learned_chunk_association_blend must be in [0, 1]")
        if not 0.0 <= self.learned_chunk_association_lr <= 1.0:
            raise ValueError("learned_chunk_association_lr must be in [0, 1]")
        if not 0.0 <= self.learned_chunk_association_decay <= 1.0:
            raise ValueError("learned_chunk_association_decay must be in [0, 1]")
        if self.binding_association_lr < 0.0:
            raise ValueError("binding_association_lr must be non-negative")
        if not 0.0 <= self.binding_association_decay <= 1.0:
            raise ValueError("binding_association_decay must be in [0, 1]")
        if self.binding_gain_strength < 0.0:
            raise ValueError("binding_gain_strength must be non-negative")
        if self.binding_n_bindings < 0:
            raise ValueError("binding_n_bindings must be non-negative")
        if self.binding_fan_in < 2:
            raise ValueError("binding_fan_in must be at least 2")
        if not 0.0 < self.context_fast_rate <= 1.0:
            raise ValueError("context_fast_rate must be in (0, 1]")
        if not 0.0 < self.context_medium_rate <= 1.0:
            raise ValueError("context_medium_rate must be in (0, 1]")
        if not 0.0 < self.context_slow_rate <= 1.0:
            raise ValueError("context_slow_rate must be in (0, 1]")
        if not 0.0 <= self.context_recurrent_density <= 1.0:
            raise ValueError("context_recurrent_density must be in [0, 1]")
        if self.context_recurrent_scale < 0.0:
            raise ValueError("context_recurrent_scale must be non-negative")
        if self.context_inhibition_strength < 0.0:
            raise ValueError("context_inhibition_strength must be non-negative")
        if self.abstraction_n_concepts <= 0:
            raise ValueError("abstraction_n_concepts must be positive")
        if not 0.0 < self.abstraction_slow_rate <= 1.0:
            raise ValueError("abstraction_slow_rate must be in (0, 1]")
        if not 0.0 < self.abstraction_fast_rate <= 1.0:
            raise ValueError("abstraction_fast_rate must be in (0, 1]")
        if self.abstraction_learning_rate < 0.0:
            raise ValueError("abstraction_learning_rate must be non-negative")
        if self.abstraction_feedback_lr < 0.0:
            raise ValueError("abstraction_feedback_lr must be non-negative")
        if self.abstraction_feedback_strength < 0.0:
            raise ValueError("abstraction_feedback_strength must be non-negative")
        if self.binding_tau <= 0.0:
            raise ValueError("binding_tau must be positive")
        if self.binding_stp_u_inc < 0.0:
            raise ValueError("binding_stp_u_inc must be non-negative")
        if self.binding_stp_tau_f <= 0.0:
            raise ValueError("binding_stp_tau_f must be positive")
        if self.binding_stp_tau_d <= 0.0:
            raise ValueError("binding_stp_tau_d must be positive")
        if self.binding_pv_threshold < 0.0:
            raise ValueError("binding_pv_threshold must be non-negative")
        if self.binding_pv_gain < 0.0:
            raise ValueError("binding_pv_gain must be non-negative")
        if self.enable_binding_layer and self.n_columns < 2:
            raise ValueError("enable_binding_layer requires at least 2 columns")
        if not 0.0 < self.stc_tag_decay <= 1.0:
            raise ValueError("stc_tag_decay must be in (0, 1]")
        if not 0.0 <= self.stc_capture_release <= 1.0:
            raise ValueError("stc_capture_release must be in [0, 1]")
        if self.stc_consolidation_rate <= 0.0:
            raise ValueError("stc_consolidation_rate must be positive")
        if self.functional_minute <= 0:
            raise ValueError("functional_minute must be positive")
        if self.stc_tag_duration_weak <= 0.0:
            raise ValueError("stc_tag_duration_weak must be positive")
        if self.stc_tag_duration_strong <= 0.0:
            raise ValueError("stc_tag_duration_strong must be positive")
        if self.stc_prp_tau_weak <= 0.0:
            raise ValueError("stc_prp_tau_weak must be positive")
        if self.stc_prp_tau_strong <= 0.0:
            raise ValueError("stc_prp_tau_strong must be positive")
        if self.stc_prp_synthesis_rate < 0.0:
            raise ValueError("stc_prp_synthesis_rate must be non-negative")
        if self.stc_prp_capture_threshold < 0.0:
            raise ValueError("stc_prp_capture_threshold must be non-negative")
        if self.stc_prp_consumption < 0.0:
            raise ValueError("stc_prp_consumption must be non-negative")
        if self.stc_strong_event_threshold < 0.0:
            raise ValueError("stc_strong_event_threshold must be non-negative")
        if self.acquisition_concept_novelty_weight < 0.0:
            raise ValueError("acquisition_concept_novelty_weight must be non-negative")
        if self.acquisition_concept_uncertainty_weight < 0.0:
            raise ValueError("acquisition_concept_uncertainty_weight must be non-negative")
        base_input_dim = self.hashed_ngram_dim if self.input_representation == "hashed_ngram" else self.n_ascii
        if self.enable_learned_chunking and self.learned_chunk_feature_mode == "concat":
            self.input_dim = int(base_input_dim + self.learned_chunk_concat_dim)
        else:
            self.input_dim = int(base_input_dim)
        if self.column_latent_dim <= 0:
            raise ValueError("column_latent_dim must be positive")
        if self.index_rebuild_threshold <= 0:
            raise ValueError("index_rebuild_threshold must be positive")
        if self.routing_shards <= 0:
            raise ValueError("routing_shards must be positive")
        if self.routing_shards > self.n_columns:
            raise ValueError("routing_shards must be less than or equal to n_columns")
        if self.routing_index_mode not in {"auto", "faiss_hnsw", "torch_topk", "exact_cosine"}:
            raise ValueError("routing_index_mode must be one of auto, faiss_hnsw, torch_topk, exact_cosine")
        if self.shard_candidate_factor <= 0:
            raise ValueError("shard_candidate_factor must be positive")

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
