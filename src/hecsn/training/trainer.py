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
from hecsn.core.context import BindingLayer, ContextLayer
from hecsn.core.surprise import SurpriseMonitor
from hecsn.consolidation.memory_store import DualMemoryStore
from hecsn.data.rtf_encoder import RTFEncoder
from hecsn.retrieval.hnsw_index import HierarchicalAssemblyIndex, ShardedHierarchicalAssemblyIndex
from hecsn.training.bootstrap import PredictiveBootstrap


@dataclass
class HECSNModelLite:
    """Stage-0 executable subset of HECSN.

    This model implements the representation contract, competitive routing,
    local prototype updates, surprise modulation, and memory drift tracking.
    """

    config: HECSNConfig

    def __post_init__(self) -> None:
        self.device = self.config.resolve_device()

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
        )
        self.surprise = SurpriseMonitor(layer_names=["competitive"])
        self.context_layer = ContextLayer(
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
        ) if self.config.enable_context_layer else None
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
        ) if self.config.enable_binding_layer else None
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

        init_ids = np.arange(self.config.n_columns, dtype=np.int64)
        self.hnsw_index.add(self.competitive.prototypes.detach(), init_ids)

    def routing_key_from_pattern(self, pattern_vec: torch.Tensor) -> torch.Tensor:
        """Route using spike-proxy assembly activations projected to latent space."""
        x = pattern_vec.to(self.device)
        assembly = self.competitive.assembly_from_input(x)
        if self.config.enable_learned_chunking:
            projected = self.competitive.last_projected_input
            if projected is None:
                projected = self.competitive.project_input(x)
            return F.normalize(projected, dim=0)
        routing_key = torch.mv(self.W_assembly_project.t(), assembly)
        return F.normalize(routing_key, dim=0)

    def runtime_scope_report(self) -> dict[str, Any]:
        routing_index_stats = self.hnsw_index.stats()
        local_stdp_active = self.config.plasticity_mode == "local_stdp"
        adex_post_spikes = local_stdp_active and self.config.plasticity_spike_backend == "adex"
        reason = (
            "The runnable scaffold now exposes a maintained local plasticity circuit with log-STDP-style eligibility traces, "
            "iSTDP-style inhibitory balancing, synaptic scaling, competitive prototypes, plastic latent projections, "
            "an optional AdEx-backed local postsynaptic spike backend, and an explicit tag/PRP replay-consolidation stack. "
            "It still does not expose the paper's full recurrent AdEx / molecular-STC circuit."
            if adex_post_spikes
            else (
                "The runnable scaffold now exposes a maintained local plasticity circuit with log-STDP-style eligibility traces, "
                "iSTDP-style inhibitory balancing, synaptic scaling, competitive prototypes, plastic latent projections, "
                "and an explicit tag/PRP replay-consolidation stack. "
                "It still does not expose the paper's full recurrent AdEx / molecular-STC circuit."
                if local_stdp_active
                else (
                    "The runnable scaffold now uses active column input weights, competitive prototypes, a latent projection matrix, "
                    "and an explicit tag/PRP replay-consolidation stack. "
                    "It still does not expose the paper's full recurrent AdEx / molecular-STC circuit."
                )
            )
        )
        return {
            "model_type": "HECSNModelLite",
            "benchmark_family": (
                "contextual_routing_multiscale"
                if self.context_layer is not None
                else "mechanism_validation_competitive"
            ),
            "input_representation": str(self.config.input_representation),
            "plasticity_mode": str(self.config.plasticity_mode),
            "plasticity_spike_backend": str(self.config.plasticity_spike_backend) if local_stdp_active else None,
            "input_dim": int(self.config.input_dim),
            "validates_full_log_stdp_weight_target": False,
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
            "estimated_neurons": int(self.config.n_columns * self.config.neurons_per_column_assumption),
            "neurons_per_column_assumption": int(self.config.neurons_per_column_assumption),
            "routing_candidate_fraction": float(self.config.k_routing / max(1, self.config.n_columns)),
            "routing_backend_mode": str(self.config.routing_index_mode),
            "routing_index": routing_index_stats,
            "weight_distribution": self.competitive.distribution_proxy_stats(),
        }


class HECSNTrainer:
    """Main stage-0 trainer."""

    def __init__(self, model: HECSNModelLite, config: HECSNConfig):
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
        self.column_anchors: dict[int, dict[str, torch.Tensor | float]] = {}
        self.bootstrap = PredictiveBootstrap(device=self.model.device, input_dim=self.config.input_dim)
        self.encoder = RTFEncoder.from_config(self.config)
        self._recent_stream_text = ""
        self._last_raw_window_text: str | None = None
        self._cached_episode_text: str | None = None
        self._last_episode_refresh_length = 0

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
        for overlap in range(max_overlap, 0, -1):
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

        if self.encoder.uses_learned_chunking:
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
        if self.encoder.uses_learned_chunking and len(sentence_like_segments) <= 1:
            window_terms = {token.lower() for token in re.findall(r"[A-Za-z0-9']+", raw_window)}
            tail_chars = max(96, min(192, len(raw_window) * 8))
            learned_segments = self.encoder.segment_text(text[-tail_chars:])
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
        if raw_window is None or self.config.input_dim != self.config.n_ascii:
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
            routing_gain = self.model.context_layer.modulation_gain(
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
                routing_key = torch.mv(self.model.W_assembly_project.t(), assembly)
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

        return applied

    def train_step(self, pattern_vec: torch.Tensor, raw_window: Optional[str] = None) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        x = pattern_vec.to(self.model.device)
        context_gain = None
        context_prediction = None
        binding_strength = 0.0

        drift_bucket = self.last_winner if self.config.use_winner_local_drift else None
        drift = self.model.memory_store.compute_drift(drift_bucket)
        floor_rising = self._update_rolling_drift_floor(drift)
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
            replay_updates = self._sleep_replay("micro")
            if replay_updates > 0:
                sleep_type = "micro"

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

        # If sustained norepinephrine exceeds the reset threshold and the cooldown has elapsed,
        # revive dead columns to re-enable plasticity.  This is the network's self-repair reflex:
        # persistent prediction failure → column revival → new routing capacity.
        network_reset_triggered = False
        reset_cooldown = self.config.emergency_deep_sleep_cooldown_tokens
        if (
            self.model.surprise.should_reset_network()
            and (self.token_count - self.last_network_reset_token) >= reset_cooldown
        ):
            self.model.competitive.force_revive_dead_columns(routing_key=routing_key)
            self.last_network_reset_token = self.token_count
            network_reset_triggered = True

        local_trace = self._local_trace_from_raw_window(
            raw_window,
            context_confidence=max(0.0, min(1.0, 1.0 - recon_error)),
        )

        context_prediction, context_gain = self._context_prediction_and_gain()

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

        winner_consolidation = 0.0
        if self.memory_warm_started:
            winner_levels = [
                self.model.memory_store.bucket_consolidation_level(int(winner.item()))
                for winner in winners
            ]
            if winner_levels:
                winner_consolidation = float(sum(winner_levels) / len(winner_levels))
        wake_plasticity_scale = max(0.2, 1.0 - 0.8 * winner_consolidation)
        effective_modulator = float(modulator) * wake_plasticity_scale

        assembly = self.model.competitive.process(
            routing_key,
            winners,
            effective_modulator,
            winner_strengths=strengths,
            eligibility_trace=local_trace,
            assembly_projection=self.model.W_assembly_project,
        )
        abstraction_input = assembly.clone()
        if self.model.abstraction_layer is not None:
            self.model.abstraction_layer.observe(
                abstraction_input,
                update_weights=True,
                precision_weight=self._context_precision_weight(),
            )
        assembly, binding_strength = self._apply_binding(
            assembly,
            context_prediction,
            update_weights=True,
        )
        self._apply_column_anchors(int(winner.item()) for winner in winners)
        if self.model.context_layer is not None:
            self.model.context_layer.observe(
                assembly,
                update_weights=True,
                precision_weight=self._context_precision_weight(),
            )

        updated_indices = winners
        if int(self.model.competitive.last_revived_indices.numel()) > 0:
            updated_indices = torch.unique(
                torch.cat([winners, self.model.competitive.last_revived_indices.to(self.model.device)]),
                sorted=True,
            )
        winner_ids = updated_indices.detach().cpu().numpy().astype(np.int64)
        winner_vectors = self.model.competitive.prototypes[updated_indices].detach()
        self.model.hnsw_index.add(winner_vectors, winner_ids)

        next_token = self.token_count + 1
        warm_started = self._maybe_warm_start_memory(next_token)
        winner_id = int(winners[0].item())
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

        memory_stats = self.model.memory_store.summary_stats() if self.memory_warm_started else {}

        metrics["token"] = self.token_count
        metrics["surprise"] = float(modulator)
        metrics["dopamine"] = float(self.model.surprise.dopamine)
        metrics["serotonin"] = float(self.model.surprise.serotonin)
        metrics["acetylcholine"] = float(self.model.surprise.acetylcholine)
        metrics["norepinephrine"] = float(self.model.surprise.norepinephrine)
        metrics["network_reset_triggered"] = int(network_reset_triggered)
        metrics["plasticity_mode"] = str(self.config.plasticity_mode)
        metrics["plasticity_spike_backend"] = (
            str(self.model.competitive.local_plasticity.spike_backend)
            if self.model.competitive.local_plasticity is not None
            else "proxy"
        )
        metrics["local_trace_available"] = int(local_trace is not None)
        metrics["local_trace_active_inputs"] = int((local_trace > 0).sum().item()) if local_trace is not None else 0
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
        metrics["context_strength"] = float(context_prediction.sum().item()) if isinstance(context_prediction, torch.Tensor) else 0.0
        metrics["context_gain_mean"] = float(context_gain.mean().item()) if isinstance(context_gain, torch.Tensor) else 1.0
        metrics["context_precision_weight"] = (
            float(self.model.context_layer.last_precision_weight)
            if self.model.context_layer is not None
            else 1.0
        )
        metrics["abstraction_stability_mean"] = (
            float(self.model.abstraction_layer.concept_stability.mean().item())
            if self.model.abstraction_layer is not None
            else 0.0
        )
        metrics["abstraction_certainty_mean"] = (
            float(self.model.abstraction_layer.concept_certainty.mean().item())
            if self.model.abstraction_layer is not None
            else 0.0
        )
        metrics["abstraction_gain_mean"] = (
            float(self.model.abstraction_layer.routing_gain().mean().item())
            if self.model.abstraction_layer is not None
            else 1.0
        )
        metrics["abstraction_gap_score_max"] = (
            max((float(item["gap_score"]) for item in self.model.abstraction_layer.curiosity_gaps(top_n=4)), default=0.0)
            if self.model.abstraction_layer is not None
            else 0.0
        )
        metrics["binding_strength"] = float(binding_strength)
        metrics["winner"] = int(winners[0].item())
        metrics["active_columns"] = int((assembly > 0).sum().item())
        metrics["sparsity"] = float((assembly > 0).float().mean().item())
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
        if self.encoder.uses_learned_chunking:
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

    def train_stream(self, stream: Iterable[torch.Tensor]) -> Iterable[Dict[str, Any]]:
        for vec in stream:
            yield self.train_step(vec)
