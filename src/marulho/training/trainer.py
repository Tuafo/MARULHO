"""MarulhoTrainer -- training loop with STDP, sleep consolidation, and drift tracking.

Drives the MarulhoModel through streaming text/multimodal data using
local Hebbian plasticity rules. Handles micro-sleep, deep-sleep,
emergency sleep, and the bootstrap phase.
"""

from __future__ import annotations

from collections import deque
import re
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence
import numpy as np
import torch
import torch.nn.functional as F

from marulho.config.model_config import MarulhoConfig
from marulho.core.context import AdaptiveContextLayer
from marulho.data.base_encoder import BaseEncoder
from marulho.data.encoder_factory import build_encoder
from marulho.training.model import MarulhoModel
from marulho.training.bootstrap import PredictiveBootstrap
from marulho.training.cognitive_boundary_controller import (
    CognitiveBoundaryPlan,
    CognitiveBoundaryController,
)
from marulho.training.column_scheduler import ColumnWakePlan
from marulho.training.column_transition_runtime import ColumnTransitionRuntime
from marulho.training.cuda_graph_route_transition import MAX_QUANTUM_INPUT_TOKENS


class MarulhoTrainer:
    """Main stage-0 trainer."""

    def __init__(self, model: MarulhoModel, config: MarulhoConfig):
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
        self._prev_routing_key: torch.Tensor | None = None  # For predictive columns
        self.pending_emergency_deep_sleep = False
        self._cached_drift: float | None = None
        self._dead_column_census: dict[str, int | float] = {}
        self.column_anchors: dict[int, dict[str, torch.Tensor | float]] = {}
        self.bootstrap = PredictiveBootstrap(device=self.model.device, input_dim=self.config.input_dim)
        self.encoder: BaseEncoder = build_encoder(self.config, device=self.model.device)
        self._recent_stream_text = ""
        self._last_raw_window_text: str | None = None
        self._cached_episode_text: str | None = None
        self._cached_episode_terms: set[str] = set()
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
        self._cross_modal_sensory_trace_until_token: int = -1
        self._cross_modal_fast_idle_skip_count: int = 0
        self._cross_modal_idle_trace_reset_count: int = 0
        self._cross_modal_traces_cleared_for_idle: bool = True

        # HNSW update buffer — flush every N steps to amortize add() overhead
        self._hnsw_buffer_ids: list[int | torch.Tensor] = []
        self._hnsw_buffer_vecs: list[torch.Tensor] = []
        self._hnsw_flush_interval = 16
        self._routing_index_device_update_count = 0
        self._routing_index_buffer_skip_count = 0
        self._routing_index_host_mirror_sync_count = 0
        self._routing_index_cpu_mirror_stale = False
        self._candidate_sleep_filter_execution = (
            self._build_candidate_sleep_filter_execution(
                mode="not_run",
                input_candidate_count=0,
                output_candidate_count=0,
                filtered_deep_sleep_count=0,
                backfill_candidate_count=0,
                fallback_reason=None,
            )
        )
        self.model.last_candidate_sleep_filter_execution = dict(
            self._candidate_sleep_filter_execution
        )
        self._column_wake_plan = self._build_column_wake_plan(
            mode="not_run",
            awake_indices=torch.empty(0, dtype=torch.long, device=self.model.device),
            input_candidate_count=0,
            filtered_deep_sleep_count=0,
            backfill_candidate_count=0,
            fallback_reason=None,
            wake_reason="not_run",
            sleep_reason=None,
        )
        self.model.last_column_wake_plan = self._column_wake_plan
        self._column_transition_runtime = ColumnTransitionRuntime(self)
        self._cognitive_boundary_controller = CognitiveBoundaryController()
        self._slow_memory_archive_count = 0
        self._slow_memory_archive_skip_count = 0
        self._slow_memory_last_archive_reason = "not_run"
        self._awake_ripple_tag_count = 0
        self._awake_ripple_tag_skip_count = 0
        self._awake_ripple_last_reason = "not_run"
        self._awake_ripple_last_tagged = 0
        self._winner_host_mirror_sync_count = 0
        self._winner_host_mirror_skip_count = 0
        self._winner_host_mirror_fresh = False
        self._train_step_metrics_full_count = 0
        self._train_step_metrics_skip_count = 0
        self._train_step_profile_enabled = False
        self._train_step_profile_totals_ms: dict[str, float] = {}
        self._train_step_profile_count = 0
        self._text_burst_execution_count = 0
        self._text_burst_token_count = 0
        self._text_burst_fallback_count = 0
        self._text_burst_fallback_reasons: dict[str, int] = {}
        self._text_burst_strong_event_count = 0
        self._text_burst_last_fallback_reason: str | None = None
        self._pending_text_burst_events: list[
            tuple[int, torch.Tensor, str, dict[str, Any] | None]
        ] = []
        self._text_burst_event_flush_count = 0
        self._text_burst_event_forced_flush_count = 0
        self._text_burst_event_deferred_apply_skip_count = 0
        self._text_burst_event_last_flush_reason: str | None = None
        self._text_sequence_execution_count = 0
        self._text_sequence_token_count = 0
        self._text_sequence_quantum_count = 0
        self._text_sequence_stop_count = 0
        self._text_sequence_input_stage_count = 0
        self._text_sequence_input_staged_token_count = 0
        self._text_sequence_input_stage_skip_count = 0
        self._last_text_burst_metrics: dict[str, Any] | None = None

    def enable_train_step_profile(self, *, reset: bool = True) -> None:
        """Enable opt-in trainer stage timing for benchmarks and diagnosis."""
        if reset:
            self.reset_train_step_profile()
        self._train_step_profile_enabled = True

    def disable_train_step_profile(self) -> None:
        self._train_step_profile_enabled = False

    def reset_train_step_profile(self) -> None:
        self._train_step_profile_totals_ms = {}
        self._train_step_profile_count = 0

    def _record_train_step_profile_stage(self, name: str, elapsed_ms: float) -> None:
        self._train_step_profile_totals_ms[name] = (
            self._train_step_profile_totals_ms.get(name, 0.0)
            + float(elapsed_ms)
        )

    def train_step_profile_report(self) -> dict[str, Any]:
        totals = {
            str(name): round(float(value), 6)
            for name, value in sorted(self._train_step_profile_totals_ms.items())
        }
        count = max(0, int(self._train_step_profile_count))
        per_tick = {
            name: round(float(value) / float(count), 6)
            for name, value in totals.items()
            if count > 0
        }
        total_ms = float(totals.get("total", 0.0))
        return {
            "enabled": bool(self._train_step_profile_enabled),
            "count": count,
            "totals_ms": totals,
            "per_tick_ms": per_tick,
            "tokens_per_second_observed": (
                round(float(count) * 1000.0 / total_ms, 6)
                if count > 0 and total_ms > 0.0
                else 0.0
            ),
        }

    def column_transition_runtime_report(self) -> dict[str, Any]:
        report = self._column_transition_runtime.report()
        report["winner_host_mirror_sync_count"] = int(
            self._winner_host_mirror_sync_count
        )
        report["winner_host_mirror_skip_count"] = int(
            self._winner_host_mirror_skip_count
        )
        report["winner_host_mirror_fresh"] = bool(
            self._winner_host_mirror_fresh
        )
        report["text_burst_execution_count"] = int(
            self._text_burst_execution_count
        )
        report["text_burst_token_count"] = int(self._text_burst_token_count)
        report["text_burst_fallback_count"] = int(
            self._text_burst_fallback_count
        )
        report["text_burst_fallback_reasons"] = dict(
            sorted(self._text_burst_fallback_reasons.items())
        )
        report["text_burst_strong_event_count"] = int(
            self._text_burst_strong_event_count
        )
        report["text_burst_last_fallback_reason"] = (
            self._text_burst_last_fallback_reason
        )
        report["text_burst_event_pending_tokens"] = len(
            self._pending_text_burst_events
        )
        report["text_burst_event_flush_count"] = int(
            self._text_burst_event_flush_count
        )
        report["text_burst_event_forced_flush_count"] = int(
            self._text_burst_event_forced_flush_count
        )
        report["text_burst_event_deferred_apply_skip_count"] = int(
            self._text_burst_event_deferred_apply_skip_count
        )
        report["text_burst_event_last_flush_reason"] = (
            self._text_burst_event_last_flush_reason
        )
        report["text_sequence_execution_count"] = int(
            self._text_sequence_execution_count
        )
        report["text_sequence_token_count"] = int(
            self._text_sequence_token_count
        )
        report["text_sequence_quantum_count"] = int(
            self._text_sequence_quantum_count
        )
        report["text_sequence_stop_count"] = int(
            self._text_sequence_stop_count
        )
        report["text_sequence_input_staging_enabled"] = bool(
            getattr(self.config, "cuda_graph_sequence_input_staging", True)
        )
        report["text_sequence_input_stage_count"] = int(
            self._text_sequence_input_stage_count
        )
        report["text_sequence_input_staged_token_count"] = int(
            self._text_sequence_input_staged_token_count
        )
        report["text_sequence_input_stage_skip_count"] = int(
            self._text_sequence_input_stage_skip_count
        )
        report["text_sequence_owner"] = "training"
        report["text_sequence_stop_boundary"] = "between_quanta"
        report["cognitive_boundary_controller"] = (
            self._cognitive_boundary_controller.report()
        )
        return report

    def stage_text_input_quantum(self, patterns: list[torch.Tensor]) -> bool:
        """Stage one bounded text quantum into persistent CUDA graph input state."""

        return self._column_transition_runtime.stage_text_input_quantum(patterns)

    def _text_burst_fallback(self, reason: str) -> bool:
        normalized_reason = str(reason)
        if self._pending_text_burst_events:
            self.flush_text_burst_events(reason=normalized_reason)
        self._text_burst_fallback_count += 1
        self._text_burst_fallback_reasons[normalized_reason] = (
            self._text_burst_fallback_reasons.get(normalized_reason, 0) + 1
        )
        self._text_burst_last_fallback_reason = normalized_reason
        return False

    def _apply_text_burst_events(
        self,
        burst_outputs: Mapping[str, Any],
        *,
        reason: str,
        forced: bool,
    ) -> bool:
        if not bool(burst_outputs.get("truth_synced", False)):
            return False
        pending = self._pending_text_burst_events
        strong_indices = [
            int(index) for index in burst_outputs["strong_indices"]
        ]
        result_rows = list(burst_outputs.get("result_rows", []))
        strong_result_rows = list(burst_outputs.get("strong_result_rows", []))
        if result_rows:
            if len(result_rows) != len(pending):
                raise RuntimeError(
                    "persistent executor event metadata does not match device evidence"
                )
            strong_result_rows = [result_rows[index] for index in strong_indices]
        elif len(strong_result_rows) != len(strong_indices):
            raise RuntimeError(
                "persistent executor strong-event metadata does not match device evidence"
            )
        strong_assemblies = list(burst_outputs["strong_assemblies"])
        strong_routing_keys = list(burst_outputs["strong_routing_keys"])
        for position, index in enumerate(strong_indices):
            token_marker, pattern, raw_window, metadata = pending[index]
            row = strong_result_rows[position]
            self.model.memory_store.update(
                strong_assemblies[position],
                importance=max(1e-3, abs(float(row[7]))),
                token_count=token_marker,
                bucket_id=int(row[6]),
                input_pattern=pattern,
                routing_key=strong_routing_keys[position],
                raw_window=raw_window,
                text=raw_window,
                metadata=metadata,
                capture_tag=max(0.0, float(row[0])),
            )
        event_count = len(pending)
        strong_count = len(strong_indices)
        self._text_burst_strong_event_count += strong_count
        self._slow_memory_archive_count += strong_count
        self._slow_memory_archive_skip_count += event_count - strong_count
        self._slow_memory_last_archive_reason = (
            "strong_capture" if strong_count else "cadence_skip"
        )
        graph = self._column_transition_runtime._cuda_graph_runtime
        assert graph is not None
        final_result = graph.consume_result()
        final_token = int(pending[-1][0]) if pending else int(self.token_count)
        self._last_text_burst_metrics = {
            "token": final_token,
            "winner": int(final_result[6]),
            "recon_error": float(final_result[0]),
            "pred_error": max(0.0, min(1.0, float(final_result[1]))),
            "surprise": max(0.0, min(1.0, float(final_result[0]))),
            "dopamine": float(final_result[2]),
            "acetylcholine": float(final_result[3]),
            "norepinephrine": float(final_result[4]),
            "serotonin": float(final_result[5]),
            "effective_modulator": float(final_result[7]),
            "train_step_metrics_mode": "device_burst_lightweight",
            "memory_index": None,
        }
        self.last_winner = int(final_result[6])
        self._winner_host_mirror_sync_count += 1
        self._winner_host_mirror_skip_count += event_count - 1
        self._winner_host_mirror_fresh = True
        self._column_transition_runtime.graph_host_winner_reuse_count += 1
        if self.model.surprise.dopamine > 0.7 and strong_count:
            tagged = self.model.memory_store.ripple_tag_awake(
                current_token=int(self.token_count),
                window_tokens=max(1, self.config.functional_minute // 2),
                da_level=self.model.surprise.dopamine,
            )
            self._awake_ripple_tag_count += 1
            self._awake_ripple_last_tagged = int(tagged)
            self._awake_ripple_last_reason = "strong_capture"
            self._awake_ripple_tag_skip_count += event_count - strong_count
        elif self.model.surprise.dopamine > 0.7:
            self._awake_ripple_tag_skip_count += event_count
            self._awake_ripple_last_tagged = 0
            self._awake_ripple_last_reason = "cadence_skip"
        else:
            self._awake_ripple_last_reason = "dopamine_below_threshold"
        graph_competitive_surprise = (
            self._column_transition_runtime.consume_graph_competitive_surprise()
        )
        if graph_competitive_surprise is not None:
            self.model.surprise.record_error(
                "competitive",
                graph_competitive_surprise,
            )
        self._pending_text_burst_events = []
        self._text_burst_event_flush_count += 1
        self._text_burst_event_forced_flush_count += int(forced)
        self._text_burst_event_last_flush_reason = str(reason)
        return True

    @torch.no_grad()
    def flush_text_burst_events(self, *, reason: str = "explicit") -> bool:
        """Drain bounded device evidence before a CPU-owned boundary."""

        if not self._pending_text_burst_events:
            return False
        burst_outputs = self._column_transition_runtime.drain_text_burst_events()
        return self._apply_text_burst_events(
            burst_outputs,
            reason=reason,
            forced=True,
        )

    def _text_burst_precheck_fallback_reason(
        self,
        *,
        token_count: int,
        raw_window_count: int | None,
        start_token: int | None = None,
    ) -> str | None:
        graph = self._column_transition_runtime._cuda_graph_runtime
        if graph is None or not graph.active:
            return "persistent_cuda_graph_inactive"
        burst_capacity = max(
            1,
            int(self._column_transition_runtime.text_burst_token_capacity()),
        )
        if token_count <= 0 or token_count > burst_capacity:
            return "burst_exceeds_device_capacity"
        if raw_window_count is None or raw_window_count != token_count:
            return "burst_requires_raw_windows"
        token_start = self.token_count if start_token is None else int(start_token)
        if token_start < self.config.bootstrap_tokens:
            return "bootstrap_active"
        if (
            self.model.context_layer is not None
            or self.model.binding_layer is not None
            or self.model.abstraction_layer is not None
            or self.column_anchors
        ):
            return "higher_layer_tick_active"
        if not self.memory_warm_started or self._cached_drift is None:
            return "runtime_not_fully_warm"
        if graph._last_result is None:
            return "host_truth_not_initialized"
        cross_modal = self.model.cross_modal
        if cross_modal is not None and (
            token_start <= self._cross_modal_sensory_trace_until_token
            or not self._cross_modal_traces_cleared_for_idle
            or len(self._recent_visual_frames) >= 3
            or len(self._recent_audio_frames) >= 3
        ):
            return "cross_modal_wake_boundary"
        return None

    def _classify_text_burst_boundaries(
        self,
        *,
        start_token: int,
        token_count: int,
        record: bool,
    ) -> CognitiveBoundaryPlan:
        telemetry_interval = max(
            1,
            int(self.config.trainer_telemetry_interval_tokens),
        )
        archive_interval = max(
            1,
            int(self.config.slow_memory_archive_interval_tokens),
        )
        planner = (
            self._cognitive_boundary_controller.plan
            if bool(record)
            else self._cognitive_boundary_controller.classify
        )
        return planner(
            start_token=start_token,
            token_count=token_count,
            telemetry_interval=telemetry_interval,
            slow_memory_archive_interval=archive_interval,
            drift_floor_window_tokens=self.config.drift_floor_window_tokens,
            hnsw_flush_interval=self._hnsw_flush_interval,
            hnsw_buffer_pending=bool(
                self._hnsw_buffer_ids or self._hnsw_buffer_vecs
            ),
            deep_sleep_interval_tokens=self.config.deep_sleep_interval_tokens,
            last_deep_sleep_token=self.last_deep_sleep_token,
            pending_emergency_deep_sleep=self.pending_emergency_deep_sleep,
            emergency_deep_sleep_cooldown_tokens=(
                self.config.emergency_deep_sleep_cooldown_tokens
            ),
            micro_sleep_interval_tokens=self.config.micro_sleep_interval_tokens,
            last_micro_sleep_token=self.last_micro_sleep_token,
        )

    def _can_prestage_text_quantum(
        self,
        *,
        start: int,
        end: int,
        burst_capacity: int,
        requested_metrics: set[int],
    ) -> bool:
        if requested_metrics.intersection(range(start, end)):
            return False
        graph = self._column_transition_runtime._cuda_graph_runtime
        if graph is None:
            return False
        simulated_token = int(self.token_count)
        simulated_replay = int(graph.replay_count)
        for chunk_start in range(start, end, burst_capacity):
            chunk_end = min(end, chunk_start + burst_capacity)
            chunk_len = int(chunk_end - chunk_start)
            reason = self._text_burst_precheck_fallback_reason(
                token_count=chunk_len,
                raw_window_count=chunk_len,
                start_token=simulated_token,
            )
            if reason is not None:
                return False
            boundary_plan = self._classify_text_burst_boundaries(
                start_token=simulated_token,
                token_count=chunk_len,
                record=False,
            )
            if boundary_plan.fallback_reason is not None:
                return False
            threshold = int(self.config.candidate_homeostasis_start_tokens)
            if simulated_token < threshold < simulated_token + chunk_len:
                return False
            sync_interval = max(
                1,
                int(self.config.cuda_graph_host_truth_sync_interval_tokens),
            )
            sync_offsets = [
                offset
                for offset in range(1, chunk_len + 1)
                if (simulated_replay + offset) % sync_interval == 0
            ]
            if sync_offsets not in ([], [chunk_len]):
                return False
            simulated_token += chunk_len
            simulated_replay += chunk_len
        return True

    def _stageable_text_sequence_end(
        self,
        *,
        start: int,
        token_count: int,
        burst_capacity: int,
        requested_metrics: set[int],
    ) -> int:
        start = int(start)
        token_count = int(token_count)
        if start < 0 or start >= token_count:
            return start
        if self._column_transition_runtime._cuda_graph_runtime is None:
            return start
        burst_capacity = max(1, int(burst_capacity))
        max_end = min(token_count, start + int(MAX_QUANTUM_INPUT_TOKENS))
        best_end = start
        graph = self._column_transition_runtime._cuda_graph_runtime
        assert graph is not None
        simulated_token = int(self.token_count)
        simulated_replay = int(graph.replay_count)
        for chunk_start in range(start, max_end, burst_capacity):
            chunk_end = min(max_end, chunk_start + burst_capacity)
            chunk_len = int(chunk_end - chunk_start)
            if requested_metrics.intersection(range(chunk_start, chunk_end)):
                break
            reason = self._text_burst_precheck_fallback_reason(
                token_count=chunk_len,
                raw_window_count=chunk_len,
                start_token=simulated_token,
            )
            if reason is not None:
                break
            boundary_plan = self._classify_text_burst_boundaries(
                start_token=simulated_token,
                token_count=chunk_len,
                record=False,
            )
            if boundary_plan.fallback_reason is not None:
                break
            threshold = int(self.config.candidate_homeostasis_start_tokens)
            if simulated_token < threshold < simulated_token + chunk_len:
                break
            sync_interval = max(
                1,
                int(self.config.cuda_graph_host_truth_sync_interval_tokens),
            )
            sync_offsets = [
                offset
                for offset in range(1, chunk_len + 1)
                if (simulated_replay + offset) % sync_interval == 0
            ]
            if sync_offsets not in ([], [chunk_len]):
                break
            best_end = chunk_end
            simulated_token += chunk_len
            simulated_replay += chunk_len
        return best_end

    @torch.no_grad()
    def train_text_burst(
        self,
        patterns: list[torch.Tensor],
        *,
        raw_windows: list[str] | None = None,
        memory_metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """Collapse ordinary text ticks between explicit cognitive boundaries."""

        token_count = len(patterns)
        graph = self._column_transition_runtime._cuda_graph_runtime
        raw_window_count = None if raw_windows is None else len(raw_windows)
        precheck_reason = self._text_burst_precheck_fallback_reason(
            token_count=token_count,
            raw_window_count=raw_window_count,
        )
        if precheck_reason is not None:
            return self._text_burst_fallback(precheck_reason)
        assert graph is not None
        profile_enabled = bool(self._train_step_profile_enabled)
        profile_started = time.perf_counter() if profile_enabled else 0.0
        profile_last = profile_started
        cross_modal = self.model.cross_modal
        if profile_enabled:
            profile_now = time.perf_counter()
            self._record_train_step_profile_stage(
                "text_burst_precheck",
                (profile_now - profile_last) * 1000.0,
            )
            profile_last = profile_now

        start = int(self.token_count)
        end = start + token_count
        archive_interval = max(
            1,
            int(self.config.slow_memory_archive_interval_tokens),
        )
        boundary_plan = self._classify_text_burst_boundaries(
            start_token=start,
            token_count=token_count,
            record=True,
        )
        if boundary_plan.fallback_reason is not None:
            return self._text_burst_fallback(boundary_plan.fallback_reason)
        if profile_enabled:
            profile_now = time.perf_counter()
            self._record_train_step_profile_stage(
                "text_burst_boundary_plan",
                (profile_now - profile_last) * 1000.0,
            )
            profile_last = profile_now
        sync_interval = max(
            1,
            int(self.config.cuda_graph_host_truth_sync_interval_tokens),
        )
        sync_offsets = [
            offset
            for offset in range(1, token_count + 1)
            if (graph.replay_count + offset) % sync_interval == 0
        ]
        if sync_offsets not in ([], [token_count]):
            return self._text_burst_fallback("host_truth_boundary")
        if profile_enabled:
            profile_now = time.perf_counter()
            self._record_train_step_profile_stage(
                "text_burst_host_truth_gate",
                (profile_now - profile_last) * 1000.0,
            )
            profile_last = profile_now

        try:
            burst_outputs = self._column_transition_runtime.replay_text_burst(
                patterns
            )
        except RuntimeError as exc:
            return self._text_burst_fallback(str(exc))
        if profile_enabled:
            profile_now = time.perf_counter()
            self._record_train_step_profile_stage(
                "text_burst_graph_replay",
                (profile_now - profile_last) * 1000.0,
            )
            profile_last = profile_now
        stable_metadata = (
            None
            if memory_metadata is None
            else {str(key): value for key, value in dict(memory_metadata).items()}
        )
        self._pending_text_burst_events.extend(
            (
                start + index + 1,
                pattern,
                str(raw_windows[index]),
                None if stable_metadata is None else dict(stable_metadata),
            )
            for index, pattern in enumerate(patterns)
        )
        if profile_enabled:
            profile_now = time.perf_counter()
            self._record_train_step_profile_stage(
                "text_burst_pending_metadata",
                (profile_now - profile_last) * 1000.0,
            )
            profile_last = profile_now

        runtime = self._column_transition_runtime
        comp = self.model.competitive
        pred = self.model.predictive
        runtime.route_vote_execution_count += token_count
        runtime.selection_execution_count += token_count
        runtime.fused_vote_competition_execution_count += token_count
        runtime.graph_consolidation_lookup_skip_count += token_count
        runtime.graph_empty_revival_tensor_reuse_count += token_count
        runtime.execution_count += token_count
        runtime.last_execution_mode = "cuda_graph_route_transition_burst"
        runtime.last_selection_mode = "fused_route_vote_cuda"
        comp.last_input_plasticity_mode = "skipped_zero_blend"
        comp.input_plasticity_skip_count += token_count
        comp.last_revived_indices = runtime._empty_revived_indices
        comp.last_homeostasis_update_count = int(
            runtime._route_candidates.numel()
        )
        comp.last_homeostasis_update_mode = "candidate_subset"
        comp.last_state_transition_mode = (
            "dense_all_columns_cuda_graph_route_transition_burst"
        )
        comp.last_state_transition_column_count = int(comp.n_columns)
        comp.last_state_transition_cached_count = 0
        comp.last_state_transition_materialize_mode = "dense_cuda_graph_burst"
        comp.last_state_transition_materialize_count = 0
        comp.last_state_transition_materialize_max_age = 0
        comp.state_transition_step_count += int(token_count)
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
        pred.last_dense_transition_mode = "inplace_triton"
        pred.last_dense_transition_fallback_reason = None
        candidate_predictive_graph = bool(
            runtime.candidate_predictive_transition_active
            and start
            >= int(self.config.candidate_homeostasis_start_tokens)
            and runtime._route_candidates is not None
        )
        if candidate_predictive_graph:
            candidates = runtime._route_candidates
            pred._record_prediction_update_scope(candidates)
            pred._record_location_update_scope(candidates)
            pred.predictive_step_count += int(token_count)
            pred._predictive_has_cached_columns = True
            pred._last_predictive_completed_candidates = None
            pred._last_predictive_completed_step = int(pred.predictive_step_count)
            runtime.candidate_predictive_transition_execution_count += int(token_count)
            runtime.candidate_predictive_transition_cached_count += int(token_count) * max(
                0,
                int(comp.n_columns) - int(candidates.numel()),
            )
        else:
            pred._record_prediction_update_scope(None)
            pred._mark_predictive_update_complete(None, step_count=token_count)

        updated_count = token_count * int(runtime._winner.numel())
        self._routing_index_device_update_count += updated_count
        self._routing_index_buffer_skip_count += updated_count
        self._routing_index_cpu_mirror_stale = True
        if bool(burst_outputs.get("truth_synced", False)):
            self._apply_text_burst_events(
                burst_outputs,
                reason="host_truth_boundary",
                forced=False,
            )
        else:
            self._text_burst_event_deferred_apply_skip_count += 1
            self._winner_host_mirror_fresh = False
        if cross_modal is not None:
            cross_modal.record_text_idle_skip(
                decay_traces=False,
                count=token_count,
            )
            self._cross_modal_fast_idle_skip_count += token_count
        if profile_enabled:
            profile_now = time.perf_counter()
            self._record_train_step_profile_stage(
                "text_burst_event_and_idle",
                (profile_now - profile_last) * 1000.0,
            )
            profile_last = profile_now
        if boundary_plan.drift_refresh_due:
            drift_bucket = None
            global_drift = True
            if (
                self.config.use_winner_local_drift
                and self._winner_host_mirror_fresh
                and self.last_winner is not None
            ):
                drift_bucket = int(self.last_winner)
                global_drift = False
            refreshed_drift = self.model.memory_store.compute_drift(drift_bucket)
            self._cached_drift = refreshed_drift
            self._update_rolling_drift_floor(refreshed_drift)
            self.current_window_min_drift = min(
                self.current_window_min_drift,
                float(refreshed_drift),
            )
            self._cognitive_boundary_controller.record_drift_refresh(
                token=self.token_count,
                sync_free=True,
                global_drift=global_drift,
            )
        if boundary_plan.drift_floor_close_due:
            self._close_drift_floor_window()
            self._cognitive_boundary_controller.record_drift_floor_close(
                token=self.token_count
            )
        if boundary_plan.slow_memory_cadence_due:
            deferred_token = next(
                (
                    start + index + 1
                    for index in range(token_count)
                    if (start + index + 1) % archive_interval == 0
                ),
                end,
            )
            self._slow_memory_archive_skip_count += 1
            self._slow_memory_last_archive_reason = "cadence_deferred"
            self._cognitive_boundary_controller.record_slow_memory_cadence_deferred(
                token=deferred_token,
            )
        if boundary_plan.telemetry_observation_due:
            self._cognitive_boundary_controller.record_telemetry_deferred(
                token=self.token_count
            )
        self.current_window_min_drift = min(
            self.current_window_min_drift,
            float(self._cached_drift),
        )
        if profile_enabled:
            profile_now = time.perf_counter()
            self._record_train_step_profile_stage(
                "text_burst_cpu_maintenance",
                (profile_now - profile_last) * 1000.0,
            )
            profile_last = profile_now
        self._train_step_metrics_skip_count += token_count
        self._text_burst_execution_count += 1
        self._text_burst_token_count += token_count
        self._text_burst_last_fallback_reason = None
        if profile_enabled:
            profile_now = time.perf_counter()
            self._record_train_step_profile_stage(
                "text_burst_total",
                (profile_now - profile_started) * 1000.0,
            )
            self._record_train_step_profile_stage(
                "total",
                (profile_now - profile_started) * 1000.0,
            )
            self._train_step_profile_count += token_count
        return True

    @torch.no_grad()
    def train_text_sequence(
        self,
        patterns: Sequence[torch.Tensor],
        *,
        raw_windows: Sequence[str],
        memory_metadata: Mapping[str, Any] | None = None,
        quantum_tokens: int = 8,
        metric_indices: Iterable[int] = (),
        should_continue: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Execute one service-owned text sequence through training-owned quanta."""

        token_count = len(patterns)
        if len(raw_windows) != token_count:
            raise ValueError("raw_windows must align with patterns")
        quantum = max(1, int(quantum_tokens))
        requested_metrics = {
            int(index)
            for index in metric_indices
            if 0 <= int(index) < token_count
        }
        burst_capacity = max(
            1,
            int(self._column_transition_runtime.text_burst_token_capacity()),
        )
        metrics_by_index: dict[int, dict[str, Any]] = {}
        last_metrics: dict[str, Any] | None = None
        trained = 0
        quantum_count = 0
        stopped = False
        profile_enabled = bool(self._train_step_profile_enabled)
        can_stage_sequence = (
            bool(getattr(self.config, "cuda_graph_sequence_input_staging", True))
            and self._column_transition_runtime.can_prestage_text_input_quantum()
            and self.memory_warm_started
            and self._cached_drift is not None
            and self.model.context_layer is None
            and self.model.binding_layer is None
            and self.model.abstraction_layer is None
            and not self.column_anchors
        )
        sequence_staged_until = 0

        def stage_sequence_segment(stage_start: int) -> None:
            nonlocal sequence_staged_until
            stage_start = int(stage_start)
            if not can_stage_sequence or stage_start < sequence_staged_until:
                return
            stage_end = self._stageable_text_sequence_end(
                start=stage_start,
                token_count=token_count,
                burst_capacity=burst_capacity,
                requested_metrics=requested_metrics,
            )
            if stage_end <= stage_start:
                return
            profile_started = time.perf_counter() if profile_enabled else 0.0
            sequence_staged = self.stage_text_input_quantum(
                list(patterns[stage_start:stage_end])
            )
            if sequence_staged:
                self._text_sequence_input_stage_count += 1
                self._text_sequence_input_staged_token_count += (
                    stage_end - stage_start
                )
                sequence_staged_until = stage_end
            else:
                self._text_sequence_input_stage_skip_count += 1
            if profile_enabled:
                elapsed_ms = (time.perf_counter() - profile_started) * 1000.0
                self._record_train_step_profile_stage(
                    "text_sequence_input_stage",
                    elapsed_ms,
                )
                self._record_train_step_profile_stage("total", elapsed_ms)

        for start in range(0, token_count, quantum):
            if should_continue is not None and not bool(should_continue()):
                stopped = True
                break
            end = min(token_count, start + quantum)
            stage_sequence_segment(start)
            can_prestage_quantum = (
                start >= sequence_staged_until
                and self._column_transition_runtime.can_prestage_text_input_quantum()
                and self.memory_warm_started
                and self._cached_drift is not None
                and self.model.context_layer is None
                and self.model.binding_layer is None
                and self.model.abstraction_layer is None
                and not self.column_anchors
            )
            if (
                can_prestage_quantum
                and end - start > burst_capacity
                and self._can_prestage_text_quantum(
                    start=start,
                    end=end,
                    burst_capacity=burst_capacity,
                    requested_metrics=requested_metrics,
                )
            ):
                profile_started = time.perf_counter() if profile_enabled else 0.0
                self.stage_text_input_quantum(list(patterns[start:end]))
                if profile_enabled:
                    elapsed_ms = (time.perf_counter() - profile_started) * 1000.0
                    self._record_train_step_profile_stage(
                        "text_sequence_quantum_input_stage",
                        elapsed_ms,
                    )
                    self._record_train_step_profile_stage("total", elapsed_ms)
            for chunk_start in range(start, end, burst_capacity):
                stage_sequence_segment(chunk_start)
                chunk_end = min(end, chunk_start + burst_capacity)
                chunk_patterns = list(patterns[chunk_start:chunk_end])
                chunk_windows = [
                    str(value) for value in raw_windows[chunk_start:chunk_end]
                ]
                chunk_metric_indices = requested_metrics.intersection(
                    range(chunk_start, chunk_end)
                )
                burst_executed = bool(
                    len(chunk_patterns) <= burst_capacity
                    and not chunk_metric_indices
                    and self.train_text_burst(
                        chunk_patterns,
                        raw_windows=chunk_windows,
                        memory_metadata=memory_metadata,
                    )
                )
                if not burst_executed:
                    self.flush_text_burst_events(
                        reason="text_sequence_per_token_boundary"
                    )
                    self.stage_text_input_quantum(chunk_patterns)
                    for offset, (pattern, raw_window) in enumerate(
                        zip(chunk_patterns, chunk_windows)
                    ):
                        index = chunk_start + offset
                        return_metrics = index in requested_metrics
                        metrics = self.train_step(
                            pattern,
                            raw_window=raw_window,
                            memory_metadata=memory_metadata,
                            return_metrics=return_metrics,
                        )
                        if return_metrics:
                            metrics_by_index[index] = dict(metrics or {})
                            if metrics:
                                last_metrics = dict(metrics)
                trained += len(chunk_patterns)
            quantum_count += 1

        self.flush_text_burst_events(reason="text_sequence_complete")
        if last_metrics is None and self._last_text_burst_metrics is not None:
            last_metrics = dict(self._last_text_burst_metrics)
        self._text_sequence_execution_count += 1
        self._text_sequence_token_count += trained
        self._text_sequence_quantum_count += quantum_count
        if stopped:
            self._text_sequence_stop_count += 1
        return {
            "trained": int(trained),
            "metrics_by_index": metrics_by_index,
            "last_metrics": last_metrics or {},
            "quantum_count": int(quantum_count),
            "stopped": bool(stopped),
        }

    def _buffer_hnsw_update(
        self,
        indices: torch.Tensor,
        vectors: torch.Tensor,
        *,
        known_ids: list[int] | None = None,
    ) -> None:
        """Buffer HNSW updates; flush when buffer reaches interval size."""
        if known_ids is not None:
            ids: list[int | torch.Tensor] = [int(value) for value in known_ids]
        else:
            id_tensor = indices.detach().reshape(-1)
            if id_tensor.device.type == "cpu":
                ids = [int(value) for value in id_tensor.tolist()]
            else:
                ids = [id_tensor[i] for i in range(int(id_tensor.numel()))]
        if len(ids) != int(vectors.shape[0]):
            raise ValueError("known_ids must align with buffered HNSW vectors")
        vecs = vectors.detach()
        for i, vid in enumerate(ids):
            self._hnsw_buffer_ids.append(vid)
            self._hnsw_buffer_vecs.append(vecs[i])
        if len(self._hnsw_buffer_ids) >= self._hnsw_flush_interval:
            self._flush_hnsw_buffer()

    def _flush_hnsw_buffer(self) -> None:
        """Flush buffered HNSW updates in a single batch."""
        if not self._hnsw_buffer_ids:
            return
        if self._routing_index_cpu_mirror_stale:
            sync_host_store = getattr(
                self.model.hnsw_index,
                "synchronize_host_store",
                None,
            )
            if not callable(sync_host_store):
                raise RuntimeError(
                    "routing index cannot synchronize a stale host mirror"
                )
            all_ids = np.arange(
                self.model.competitive.n_columns,
                dtype=np.int64,
            )
            sync_host_store(
                self.model.competitive.prototypes.detach(),
                all_ids,
            )
            self._routing_index_cpu_mirror_stale = False
            self._routing_index_host_mirror_sync_count += 1
        # Deduplicate: keep latest vector per id. CUDA ids are materialized in
        # one transfer to avoid one scalar sync per buffered entry.
        materialized_ids: list[int] = [0] * len(self._hnsw_buffer_ids)
        tensor_positions: list[int] = []
        tensor_ids: list[torch.Tensor] = []
        for position, vid in enumerate(self._hnsw_buffer_ids):
            if isinstance(vid, torch.Tensor):
                tensor_positions.append(position)
                tensor_ids.append(vid.reshape(()))
            else:
                materialized_ids[position] = int(vid)
        if tensor_ids:
            tensor_values = (
                torch.stack(tensor_ids)
                .detach()
                .to(device="cpu", dtype=torch.long)
                .numpy()
                .astype(np.int64, copy=False)
            )
            for position, value in zip(tensor_positions, tensor_values):
                materialized_ids[position] = int(value)

        seen: dict[int, torch.Tensor] = {}
        for vid_int, vec in zip(materialized_ids, self._hnsw_buffer_vecs):
            seen[int(vid_int)] = vec
        ids_arr = np.array(list(seen.keys()), dtype=np.int64)
        vecs_batch = torch.stack(list(seen.values()))
        self.model.hnsw_index.add(vecs_batch, ids_arr)
        self._hnsw_buffer_ids.clear()
        self._hnsw_buffer_vecs.clear()

    def _build_candidate_sleep_filter_execution(
        self,
        *,
        mode: str,
        input_candidate_count: int,
        output_candidate_count: int,
        filtered_deep_sleep_count: int,
        backfill_candidate_count: int,
        fallback_reason: str | None,
    ) -> dict[str, Any]:
        total_columns = int(self.config.n_columns)
        return {
            "surface": "column_candidate_sleep_scheduler.v1",
            "mode": str(mode),
            "total_columns": total_columns,
            "awake_budget": int(min(self.config.k_routing, total_columns)),
            "input_candidate_count": int(max(0, input_candidate_count)),
            "output_candidate_count": int(max(0, output_candidate_count)),
            "filtered_deep_sleep_count": int(max(0, filtered_deep_sleep_count)),
            "backfill_candidate_count": int(max(0, backfill_candidate_count)),
            "deep_sleep_threshold_steps": int(self.config.dead_column_steps),
            "start_token": int(self.config.candidate_deep_sleep_filter_start_tokens),
            "backfill_factor": int(self.config.candidate_deep_sleep_backfill_factor),
            "runs_all_columns": False,
            "fallback_reason": fallback_reason,
            "tensor_device": str(self.model.device),
            "claim_boundary": (
                "training_owned_candidate_deep_sleep_filter_skips_deep_sleep_candidates_without_all_column_scan"
            ),
        }

    def _build_column_wake_plan(
        self,
        *,
        mode: str,
        awake_indices: torch.Tensor,
        input_candidate_count: int,
        filtered_deep_sleep_count: int,
        backfill_candidate_count: int,
        fallback_reason: str | None,
        wake_reason: str,
        sleep_reason: str | None,
    ) -> ColumnWakePlan:
        total_columns = int(self.config.n_columns)
        return ColumnWakePlan(
            mode=str(mode),
            total_columns=total_columns,
            awake_budget=int(min(self.config.k_routing, total_columns)),
            awake_indices=awake_indices.to(device=self.model.device, dtype=torch.long),
            input_candidate_count=int(max(0, input_candidate_count)),
            filtered_deep_sleep_count=int(max(0, filtered_deep_sleep_count)),
            backfill_candidate_count=int(max(0, backfill_candidate_count)),
            deep_sleep_threshold_steps=int(self.config.dead_column_steps),
            start_token=int(self.config.candidate_deep_sleep_filter_start_tokens),
            backfill_factor=int(self.config.candidate_deep_sleep_backfill_factor),
            wake_reason=str(wake_reason),
            sleep_reason=sleep_reason,
            fallback_reason=fallback_reason,
            tensor_device=str(self.model.device),
            runs_all_columns=False,
        )

    def _record_column_wake_plan(self, plan: ColumnWakePlan) -> None:
        self._column_wake_plan = plan
        self.model.last_column_wake_plan = plan

    def _record_candidate_sleep_filter_execution(
        self,
        report: dict[str, Any],
    ) -> None:
        self._candidate_sleep_filter_execution = dict(report)
        self.model.last_candidate_sleep_filter_execution = dict(report)

    def _candidate_sleep_filter_not_due_report(
        self,
        candidates: torch.Tensor | None,
        *,
        fallback_reason: str,
    ) -> dict[str, Any]:
        count = 0 if candidates is None else int(candidates.numel())
        return self._build_candidate_sleep_filter_execution(
            mode="not_due",
            input_candidate_count=count,
            output_candidate_count=count,
            filtered_deep_sleep_count=0,
            backfill_candidate_count=0,
            fallback_reason=fallback_reason,
        )

    def _filter_candidate_deep_sleep_plan(
        self,
        candidates: torch.Tensor,
        *,
        target_count: int,
    ) -> ColumnWakePlan:
        candidate_count = int(candidates.numel())
        target = max(0, min(int(target_count), int(self.config.n_columns)))
        if candidate_count <= 0 or target <= 0:
            plan = self._build_column_wake_plan(
                mode="candidate_deep_sleep_filter_empty",
                awake_indices=candidates[:0],
                input_candidate_count=candidate_count,
                filtered_deep_sleep_count=0,
                backfill_candidate_count=0,
                fallback_reason="no_candidates",
                wake_reason="no_awake_candidates",
                sleep_reason="empty_candidate_set",
            )
            self._record_column_wake_plan(plan)
            return plan

        if self.model.device.type == "cuda":
            bounded = candidates[:target]
            plan = self._build_column_wake_plan(
                mode="candidate_deep_sleep_filter_cuda_fallback",
                awake_indices=bounded,
                input_candidate_count=candidate_count,
                filtered_deep_sleep_count=0,
                backfill_candidate_count=max(0, candidate_count - int(bounded.numel())),
                fallback_reason="cuda_candidate_deep_sleep_filter_unmeasured_retained_candidate_set",
                wake_reason="cuda_retained_candidate_set",
                sleep_reason="cuda_deep_sleep_filter_unmeasured",
            )
            self._record_column_wake_plan(plan)
            return plan

        materialize_state = getattr(
            self.model.competitive,
            "materialize_state_transition",
            None,
        )
        if callable(materialize_state):
            materialize_state(candidates, record_noop=False)
        steps = self.model.competitive.steps_since_win[
            candidates.to(self.model.device).long()
        ]
        awake_mask = steps < int(self.config.dead_column_steps)
        awake_candidates = candidates[awake_mask]
        filtered_count = candidate_count - int(awake_candidates.numel())
        if int(awake_candidates.numel()) <= 0:
            bounded = candidates[:target]
            plan = self._build_column_wake_plan(
                mode="candidate_deep_sleep_filter_fallback",
                awake_indices=bounded,
                input_candidate_count=candidate_count,
                filtered_deep_sleep_count=filtered_count,
                backfill_candidate_count=max(0, candidate_count - int(bounded.numel())),
                fallback_reason="all_retrieved_candidates_deep_sleep",
                wake_reason="fallback_all_candidates_deep_sleep",
                sleep_reason="all_retrieved_candidates_deep_sleep",
            )
            self._record_column_wake_plan(plan)
            return plan

        filtered = awake_candidates[:target]
        plan = self._build_column_wake_plan(
            mode="candidate_deep_sleep_filter",
            awake_indices=filtered,
            input_candidate_count=candidate_count,
            filtered_deep_sleep_count=filtered_count,
            backfill_candidate_count=max(0, candidate_count - int(filtered.numel())),
            fallback_reason=(
                None
                if int(filtered.numel()) >= min(target, candidate_count)
                else "insufficient_awake_candidates_after_deep_sleep_filter"
            ),
            wake_reason="retrieved_candidate_not_in_deep_sleep",
            sleep_reason="deep_sleep_candidate_filtered_from_awake_mask",
        )
        self._record_column_wake_plan(plan)
        return plan

    def _filter_candidate_deep_sleep(
        self,
        candidates: torch.Tensor,
        *,
        target_count: int,
    ) -> torch.Tensor:
        return self._filter_candidate_deep_sleep_plan(
            candidates,
            target_count=target_count,
        ).candidates()

    def _route_vote_owner_wake_plan(
        self,
        candidates: torch.Tensor,
    ) -> ColumnWakePlan:
        snapshot = self._column_transition_runtime.route_sleep_filter_snapshot()
        filter_enabled = bool(snapshot.get("enabled", False))
        fallback_reason = snapshot.get("fallback_reason")
        if filter_enabled:
            mode = (
                "candidate_deep_sleep_filter_route_vote_fallback"
                if fallback_reason
                else "candidate_deep_sleep_filter_route_vote"
            )
            wake_reason = "route_vote_primary_score_not_in_deep_sleep"
            sleep_reason = "deep_sleep_route_score_masked_before_topk_vote"
            filtered_count = int(snapshot.get("filtered_deep_sleep_count", 0) or 0)
            backfill_count = int(snapshot.get("sleep_backfill_count", 0) or 0)
        else:
            mode = "candidate_deep_sleep_filter_route_vote_not_due"
            fallback_reason = (
                "candidate_deep_sleep_filter_no_column_can_be_deep_sleep_yet"
                if self.token_count < int(self.config.dead_column_steps)
                else "candidate_deep_sleep_filter_not_due"
            )
            wake_reason = "route_vote_selected_candidate_before_sleep_gate"
            sleep_reason = None
            filtered_count = 0
            backfill_count = 0
        plan = self._build_column_wake_plan(
            mode=mode,
            awake_indices=candidates,
            input_candidate_count=int(
                snapshot.get("input_candidate_count", int(candidates.numel())) or 0
            ),
            filtered_deep_sleep_count=filtered_count,
            backfill_candidate_count=backfill_count,
            fallback_reason=fallback_reason,
            wake_reason=wake_reason,
            sleep_reason=sleep_reason,
        )
        self._record_column_wake_plan(plan)
        return plan

    def _routing_wake_plan(
        self,
        routing_key: torch.Tensor,
        *,
        apply_sleep_filter: bool = False,
    ) -> ColumnWakePlan | None:
        target_k = max(1, int(self.config.k_routing))
        filter_start_due = bool(
            apply_sleep_filter
            and self.token_count
            >= int(self.config.candidate_deep_sleep_filter_start_tokens)
        )
        filter_age_ready = bool(
            int(self.token_count) >= int(self.config.dead_column_steps)
        )
        filter_due = bool(filter_start_due and filter_age_ready)
        search_k = target_k
        if filter_due and self.model.device.type != "cuda":
            search_k = min(
                int(self.config.n_columns),
                target_k * max(1, int(self.config.candidate_deep_sleep_backfill_factor)),
            )
        candidate_ids, candidate_distances = self.model.hnsw_index.search_tensors(
            routing_key.unsqueeze(0),
            k=search_k,
        )
        if candidate_ids.dim() != 2 or int(candidate_ids.shape[1]) <= 0:
            plan = self._build_column_wake_plan(
                mode="candidate_deep_sleep_filter_empty"
                if apply_sleep_filter
                else "candidate_routing_empty",
                awake_indices=torch.empty(
                    0,
                    dtype=torch.long,
                    device=self.model.device,
                ),
                input_candidate_count=0,
                filtered_deep_sleep_count=0,
                backfill_candidate_count=0,
                fallback_reason="routing_index_returned_no_candidates",
                wake_reason="no_awake_candidates",
                sleep_reason="routing_index_returned_no_candidates",
            )
            self._record_column_wake_plan(plan)
            return None
        candidates = candidate_ids[0].to(device=self.model.device, dtype=torch.long)
        if (
            candidate_distances.dim() == 2
            and int(candidate_distances.shape[1]) == int(candidates.numel())
        ):
            distance_row = candidate_distances[0].to(device=self.model.device)
            candidates = candidates[torch.argsort(distance_row, descending=False)]
        if not apply_sleep_filter:
            bounded = candidates[:target_k]
            plan = self._build_column_wake_plan(
                mode="candidate_routing",
                awake_indices=bounded,
                input_candidate_count=int(candidates.numel()),
                filtered_deep_sleep_count=0,
                backfill_candidate_count=max(0, int(candidates.numel()) - int(bounded.numel())),
                fallback_reason=None,
                wake_reason="retrieved_candidate",
                sleep_reason=None,
            )
            self._record_column_wake_plan(plan)
            return plan
        if not filter_start_due:
            bounded = candidates[:target_k]
            plan = self._build_column_wake_plan(
                mode="not_due",
                awake_indices=bounded,
                input_candidate_count=int(bounded.numel()),
                filtered_deep_sleep_count=0,
                backfill_candidate_count=0,
                fallback_reason="candidate_deep_sleep_filter_not_due",
                wake_reason="retrieved_candidate_before_sleep_gate",
                sleep_reason=None,
            )
            self._record_column_wake_plan(plan)
            return plan
        if not filter_age_ready:
            bounded = candidates[:target_k]
            plan = self._build_column_wake_plan(
                mode="not_due",
                awake_indices=bounded,
                input_candidate_count=int(bounded.numel()),
                filtered_deep_sleep_count=0,
                backfill_candidate_count=0,
                fallback_reason="candidate_deep_sleep_filter_no_column_can_be_deep_sleep_yet",
                wake_reason="retrieved_candidate_before_deep_sleep_age_gate",
                sleep_reason=None,
            )
            self._record_column_wake_plan(plan)
            return plan
        return self._filter_candidate_deep_sleep_plan(candidates, target_count=target_k)

    def _routing_candidates(
        self,
        routing_key: torch.Tensor,
        *,
        apply_sleep_filter: bool = False,
    ) -> torch.Tensor | None:
        plan = self._routing_wake_plan(
            routing_key,
            apply_sleep_filter=apply_sleep_filter,
        )
        return None if plan is None else plan.candidates()


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
            self._cached_episode_terms = set()
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
            self._cached_episode_terms = self._episode_terms(episode_text)
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
            current_terms = {
                token.lower()
                for token in re.findall(r"[A-Za-z0-9']+", current)
                if len(token) > 2
            }
            refresh_due = (
                self._cached_episode_text is None
                or len(self._recent_stream_text) - self._last_episode_refresh_length >= 24
                or any(ch in ".!?\n" for ch in appended)
                or bool(current_terms - self._cached_episode_terms)
            )
            if not refresh_due:
                return self._cached_episode_text

        episode_text = self._current_episode_text(current)
        self._cached_episode_text = episode_text
        self._cached_episode_terms = self._episode_terms(episode_text)
        self._last_episode_refresh_length = len(self._recent_stream_text)
        return episode_text

    @staticmethod
    def _episode_terms(text: str | None) -> set[str]:
        return {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9']+", str(text or ""))
            if len(token) > 2
        }

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
        binding_runtime_active = bool(
            self.model.binding_layer is not None
            and getattr(self.model.binding_layer, "runtime_active", True)
        )
        if (
            self.model.binding_layer is not None
            and context_prediction is not None
            and binding_runtime_active
        ):
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
    ) -> tuple[torch.Tensor, float | torch.Tensor]:
        if self.model.binding_layer is None or context_prediction is None:
            return assembly, 0.0

        runtime_bind = getattr(self.model.binding_layer, "bind_runtime", None)
        if callable(runtime_bind):
            interval = max(
                1,
                int(self.config.binding_idle_probe_interval_tokens),
            )
            runtime_active = bool(
                getattr(self.model.binding_layer, "runtime_active", False)
            )
            probe_due = runtime_active or self.token_count % interval == 0
            if not probe_due:
                record_skip = getattr(
                    self.model.binding_layer,
                    "record_runtime_idle_skip",
                    None,
                )
                if callable(record_skip):
                    record_skip()
                return assembly, getattr(self, "_cached_binding_strength", 0.0)

            context_active = bool(
                self.model.context_layer is not None
                and getattr(self.model.context_layer, "state_update_count", 0) > 0
            )
            bound_assembly, binding_strength = runtime_bind(
                context_prediction,
                assembly,
                update_weights=update_weights,
                inputs_active=context_active,
            )
            if not runtime_active:
                observed_strength = float(binding_strength.item())
                self._cached_binding_strength = observed_strength
                self.model.binding_layer.runtime_active = observed_strength > 0.0
                binding_strength = observed_strength
            has_binding = bound_assembly.sum() > 0.0
            return torch.where(has_binding, bound_assembly, assembly), binding_strength

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
                candidates = self._routing_candidates(routing_key)
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
            materialize_state = getattr(comp, "materialize_state_transition", None)
            if callable(materialize_state):
                materialize_state(None, record_noop=False)
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

    def _apply_awake_column_transition(
        self,
        *,
        routing_key: torch.Tensor,
        candidates: torch.Tensor | None,
        winners: torch.Tensor,
        strengths: torch.Tensor,
        modulator: float,
        local_trace: torch.Tensor | None,
        compute_metrics: bool,
        predictive_candidates_already_materialized: bool = False,
    ) -> tuple[
        torch.Tensor,
        list[int] | None,
        int | None,
        float,
        float,
        float,
        float,
    ]:
        if self._column_transition_runtime.active:
            return self._column_transition_runtime.apply(
            routing_key=routing_key,
            candidates=candidates,
            winners=winners,
            modulator=modulator,
            compute_metrics=compute_metrics,
        )

        winner_id_list = winners.tolist()
        winner_id = int(winner_id_list[0])

        winner_consolidation = 0.0
        if self.memory_warm_started:
            winner_levels = [
                self.model.memory_store.bucket_consolidation_level(wid)
                for wid in winner_id_list
            ]
            if winner_levels:
                winner_consolidation = float(
                    sum(winner_levels) / len(winner_levels)
                )

        dopamine = self.model.surprise.dopamine
        serotonin = self.model.surprise.serotonin
        dopamine_ltp_gain = 0.8 + 0.4 * dopamine
        serotonin_patience = max(0.2, 1.0 - 0.6 * serotonin)
        wake_plasticity_scale = (
            max(0.2, 1.0 - 0.8 * winner_consolidation)
            * serotonin_patience
        )

        predictive_scope_ready = self.token_count >= int(
            self.config.candidate_predictive_update_start_tokens
        )
        homeostasis_scope_ready = self.token_count >= int(
            self.config.candidate_homeostasis_start_tokens
        )
        predictive_scope_cuda_fallback = (
            predictive_scope_ready and self.model.device.type == "cuda"
        )
        predictive_update_indices = (
            candidates
            if predictive_scope_ready and not predictive_scope_cuda_fallback
            else None
        )

        use_dense_transition = (
            self.model.device.type == "cuda"
            and self.config.predictive_dense_transition_mode != "legacy"
        )
        used_candidate_transition = False
        if use_dense_transition:
            dense_transition_mode = self.config.predictive_dense_transition_mode
            transition_runtime_fallback = None
            if dense_transition_mode == "inplace_triton":
                dense_transition_mode = "fused_eager"
                transition_runtime_fallback = (
                    self._column_transition_runtime.fallback_reason
                )
            pred_error_mod = self.model.predictive.apply_dense_transition(
                winners,
                routing_key,
                self._prev_routing_key,
                learning_rate=0.005,
                transition_mode=dense_transition_mode,
            )
            if predictive_scope_cuda_fallback:
                self.model.predictive.last_prediction_update_fallback_reason = (
                    "cuda_sparse_prediction_update_launch_bound_dense_retained"
                )
            elif not predictive_scope_ready:
                self.model.predictive.last_prediction_update_fallback_reason = (
                    "candidate_predictive_update_not_due"
                )
            if transition_runtime_fallback is not None:
                self.model.predictive.last_dense_transition_fallback_reason = transition_runtime_fallback
        else:
            if predictive_update_indices is not None:
                pred_error_mod = self.model.predictive.update_candidate_prediction_transition(
                    winner_id_list,
                    routing_key,
                    self._prev_routing_key,
                    learning_rate=0.005,
                    candidate_indices=predictive_update_indices,
                    assume_materialized=predictive_candidates_already_materialized,
                )
                used_candidate_transition = True
            else:
                pred_error_mod = self.model.predictive.compute_prediction_error(
                    winner_id_list,
                    routing_key,
                    candidate_indices=None,
                )
        pred_boost = (
            float(pred_error_mod[winner_id_list].mean().item())
            if winner_id_list
            else 1.0
        )
        pred_boost = min(2.0, max(0.5, pred_boost))

        if not use_dense_transition and not used_candidate_transition:
            self.model.predictive.update_location(
                winner_id_list,
                routing_key,
                self._prev_routing_key,
                candidate_indices=predictive_update_indices,
            )
            self.model.predictive.update_predictions(
                winner_id_list,
                learning_rate=0.005,
                candidate_indices=predictive_update_indices,
            )
            if predictive_scope_cuda_fallback:
                self.model.predictive.last_prediction_update_fallback_reason = (
                    "cuda_sparse_prediction_update_launch_bound_dense_retained"
                )
            elif not predictive_scope_ready:
                self.model.predictive.last_prediction_update_fallback_reason = (
                    "candidate_predictive_update_not_due"
                )
        self._prev_routing_key = routing_key.detach().clone()

        effective_modulator = (
            float(modulator)
            * wake_plasticity_scale
            * dopamine_ltp_gain
            * pred_boost
        )
        homeostasis_update_indices = (
            candidates if homeostasis_scope_ready else None
        )
        assembly = self.model.competitive.process(
            routing_key,
            winners,
            effective_modulator,
            winner_strengths=strengths,
            eligibility_trace=local_trace,
            assembly_projection=self.model.W_assembly_project,
            compute_metrics=compute_metrics,
            homeostasis_update_indices=homeostasis_update_indices,
        )
        if self.model.competitive.plasticity_mode == "local_stdp":
            self.model._invalidate_projection_cache()
        return (
            assembly,
            winner_id_list,
            winner_id,
            winner_consolidation,
            effective_modulator,
            dopamine_ltp_gain,
            serotonin_patience,
        )

    @torch.no_grad()
    def train_step(
        self,
        pattern_vec: torch.Tensor,
        raw_window: Optional[str] = None,
        visual_spikes: Optional[torch.Tensor] = None,
        audio_spikes: Optional[torch.Tensor] = None,
        memory_metadata: Mapping[str, Any] | None = None,
        allow_sleep_maintenance: bool = True,
        return_metrics: bool = True,
    ) -> Dict[str, Any]:
        profile_enabled = bool(self._train_step_profile_enabled)
        profile_last = time.perf_counter() if profile_enabled else 0.0
        profile_started = profile_last
        profile_totals = self._train_step_profile_totals_ms
        metrics: Dict[str, Any] = {}
        x = pattern_vec.to(self.model.device)
        context_gain = None
        context_prediction = None
        binding_strength = 0.0
        telemetry_interval = max(
            1,
            int(self.config.trainer_telemetry_interval_tokens),
        )
        _telemetry_tick = (self.token_count % telemetry_interval == 0)
        if profile_enabled:
            profile_now = time.perf_counter()
            profile_totals["input_prepare"] = profile_totals.get("input_prepare", 0.0) + (
                profile_now - profile_last
            ) * 1000.0
            profile_last = profile_now

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
        micro_due = (
            self.token_count >= self.config.micro_sleep_interval_tokens
            and (self.token_count - self.last_micro_sleep_token) >= self.config.micro_sleep_interval_tokens
        )
        sleep_maintenance_deferred = bool(
            not allow_sleep_maintenance
            and (deep_due_interval or deep_due_emergency or micro_due)
        )
        if allow_sleep_maintenance and (deep_due_interval or deep_due_emergency):
            self._flush_hnsw_buffer()
            replay_updates = self._sleep_replay("repair" if deep_due_emergency else "deep")
            if replay_updates > 0:
                sleep_type = "deep"
                deep_sleep_emergency = bool(deep_due_emergency)
                if deep_due_emergency:
                    self.pending_emergency_deep_sleep = False
        elif allow_sleep_maintenance and micro_due:
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
        metrics["sleep_maintenance_deferred"] = int(sleep_maintenance_deferred)
        metrics["drift_floor"] = float(self.current_rolling_drift_floor if self.current_rolling_drift_floor is not None else drift)
        metrics["drift_floor_rising"] = int(floor_rising)
        if self._dead_column_census:
            metrics["dead_column_census"] = self._dead_column_census
        if profile_enabled:
            profile_now = time.perf_counter()
            profile_totals["drift_sleep"] = profile_totals.get("drift_sleep", 0.0) + (
                profile_now - profile_last
            ) * 1000.0
            profile_last = profile_now

        if self.token_count < self.config.bootstrap_tokens:
            pred_error = self.bootstrap.update(x)
            self.model.surprise.update_neuromodulators(current_error=pred_error, novelty=min(1.0, pred_error))
            modulator = self.model.surprise.get_modulator("competitive")
            metrics["pred_error"] = pred_error
        else:
            self.is_bootstrap = False
            modulator = self.model.surprise.get_modulator("competitive")

        sensory_tick = visual_spikes is not None or audio_spikes is not None
        prepared_routing = self._column_transition_runtime.prepare_routing(
            x,
            sensory_tick=sensory_tick,
        )
        if prepared_routing is None:
            routing_key = self.model.routing_key_from_pattern(x)
            recon_error = self.model.competitive.nearest_prototype_distance(
                routing_key
            )
        else:
            routing_key, recon_error = prepared_routing
        metrics["recon_error"] = float(recon_error)
        if profile_enabled:
            profile_now = time.perf_counter()
            profile_totals["routing_prepare"] = profile_totals.get("routing_prepare", 0.0) + (
                profile_now - profile_last
            ) * 1000.0
            profile_last = profile_now

        graph_owns_surprise = bool(
            prepared_routing is not None
            and self._column_transition_runtime.consume_graph_surprise_update()
        )

        # Post-bootstrap neuromodulator update using reconstruction error as the error signal.
        # The persistent CUDA tick performs this before its transition and mirrors the result.
        if (
            self.token_count >= self.config.bootstrap_tokens
            and not graph_owns_surprise
        ):
            self.model.surprise.update_neuromodulators(current_error=recon_error, novelty=min(1.0, recon_error))

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

        candidates = self._column_transition_runtime.route_candidates(
            routing_key,
            sensory_tick=sensory_tick,
        )
        if candidates is None:
            wake_plan = self._routing_wake_plan(
                routing_key,
                apply_sleep_filter=True,
            )
            candidates = None if wake_plan is None else wake_plan.candidates()
        else:
            wake_plan = self._route_vote_owner_wake_plan(candidates)

        # Predictive column consensus voting: columns that agree with
        # recent winners get a routing boost (Thousand Brains voting).
        # The retained path updates only the routed awake mask; non-awake
        # columns keep cached vote state and are not scanned for this tick.
        predictive_vote_materialized_candidates = False
        if (
            self.last_winner is not None
            and not self._column_transition_runtime.handles_predictive_vote
        ):
            consensus_gain = self.model.predictive.vote(
                [self.last_winner],
                routing_key,
                candidate_indices=candidates,
            )
            predictive_vote_materialized_candidates = bool(
                candidates is not None and int(candidates.numel()) > 0
            )
            if context_gain is not None:
                context_gain = torch.clamp(context_gain * consensus_gain, min=0.5, max=1.5)
            else:
                context_gain = consensus_gain
        if profile_enabled:
            profile_now = time.perf_counter()
            profile_totals["control_gain"] = profile_totals.get("control_gain", 0.0) + (
                profile_now - profile_last
            ) * 1000.0
            profile_last = profile_now

        if self._column_transition_runtime.active:
            winners, strengths, _ = self._column_transition_runtime.select_winner(
                routing_key=routing_key,
                candidates=candidates,
                fallback_allowed=self.is_bootstrap,
                context_gain=context_gain,
            )
        else:
            winners, strengths, _ = self.model.competitive.compete(
                routing_key,
                candidates,
                fallback_allowed=self.is_bootstrap,
                context_gain=context_gain,
            )
        if profile_enabled:
            profile_now = time.perf_counter()
            profile_totals["candidate_winner"] = profile_totals.get("candidate_winner", 0.0) + (
                profile_now - profile_last
            ) * 1000.0
            profile_last = profile_now

        (
            assembly,
            winner_id_list,
            winner_id,
            winner_consolidation,
            effective_modulator,
            da_ltp_gain,
            ht_patience,
        ) = self._apply_awake_column_transition(
            routing_key=routing_key,
            candidates=candidates,
            winners=winners,
            strengths=strengths,
            modulator=modulator,
            local_trace=local_trace,
            compute_metrics=_telemetry_tick,
            predictive_candidates_already_materialized=(
                predictive_vote_materialized_candidates
            ),
        )
        if (
            candidates is not None
            and wake_plan is not None
            and str(wake_plan.mode).startswith(
                "candidate_deep_sleep_filter_route_vote"
            )
        ):
            wake_plan = self._route_vote_owner_wake_plan(candidates)
        def _materialize_winner_ids() -> tuple[list[int], int]:
            nonlocal winner_id_list, winner_id, profile_last
            if winner_id_list is not None and winner_id is not None:
                return winner_id_list, int(winner_id)
            ids = winners.tolist()
            wid = int(ids[0])
            winner_id_list = ids
            winner_id = wid
            if profile_enabled:
                profile_now = time.perf_counter()
                profile_totals["winner_host_materialize"] = (
                    profile_totals.get("winner_host_materialize", 0.0)
                    + (profile_now - profile_last) * 1000.0
                )
                profile_last = profile_now
            return ids, wid

        graph_winner_host_mirror_can_skip = bool(
            winner_id_list is None
            and winner_id is None
            and self._column_transition_runtime.last_execution_mode
            == "cuda_graph_route_transition"
        )

        if profile_enabled:
            profile_now = time.perf_counter()
            profile_totals["column_transition"] = profile_totals.get("column_transition", 0.0) + (
                profile_now - profile_last
            ) * 1000.0
            profile_last = profile_now
        if self.model.abstraction_layer is not None:
            self.model.abstraction_layer.observe(
                assembly.clone(),
                update_weights=True,
                precision_weight=self._context_precision_weight(),
            )
            # Top-down boundary bias: Abstraction Layer → Chunking Layer (§3.1)
            if getattr(self.encoder, "uses_learned_chunking", False):
                max_gap = self.model.abstraction_layer.max_curiosity_gap_score()
                mean_cert = float(self.model.abstraction_layer.concept_certainty.mean().item())
                self.encoder.learned_chunking.set_abstraction_bias(mean_cert, max_gap)
        if profile_enabled:
            profile_now = time.perf_counter()
            profile_totals["abstraction"] = profile_totals.get("abstraction", 0.0) + (
                profile_now - profile_last
            ) * 1000.0
            profile_last = profile_now
        assembly, binding_strength = self._apply_binding(
            assembly,
            context_prediction,
            update_weights=True,
        )
        if self.column_anchors:
            anchor_winner_ids, _ = _materialize_winner_ids()
            self._apply_column_anchors(wid for wid in anchor_winner_ids)
        if self.model.context_layer is not None:
            context_plasticity_interval = max(
                1,
                int(self.config.context_plasticity_interval_tokens),
            )
            context_plasticity_due = (
                self.token_count == 0
                or (self.token_count + 1) % context_plasticity_interval == 0
            )
            self.model.context_layer.observe(
                assembly,
                update_weights=context_plasticity_due,
                precision_weight=self._context_precision_weight(),
            )
        else:
            context_plasticity_interval = 0
            context_plasticity_due = False
        if profile_enabled:
            profile_now = time.perf_counter()
            profile_totals["binding_context"] = profile_totals.get("binding_context", 0.0) + (
                profile_now - profile_last
            ) * 1000.0
            profile_last = profile_now

        # Cross-modal grounding updates (§5): sensory spikes fire BEFORE
        # text so that on_text_spike() sees the current visual/audio traces
        # (not stale traces from a previous concept).  Stage-aware gating
        # (§7): Stage 1 accepts all pairs; Stage 2+ applies alignment_gate()
        # with a bootstrap budget so confidence can build from zero.
        cross_modal_visual_conf = 0.0
        cross_modal_audio_conf = 0.0
        cross_modal_visual_accepted = None
        cross_modal_audio_accepted = None
        cross_modal_text_spike_prepared = False
        if self.model.cross_modal is not None:
            text_spike = None
            fast_text_idle_skip = bool(
                visual_spikes is None
                and audio_spikes is None
                and self.token_count > self._cross_modal_sensory_trace_until_token
                and not (
                    self.token_count - self._last_self_criticism_token
                    >= self._self_criticism_interval
                    and (
                        len(self._recent_visual_frames) >= 3
                        or len(self._recent_audio_frames) >= 3
                    )
                )
            )

            if fast_text_idle_skip:
                if not self._cross_modal_traces_cleared_for_idle:
                    self.model.cross_modal.reset()
                    self._cross_modal_idle_trace_reset_count += 1
                    self._cross_modal_traces_cleared_for_idle = True
                record_text_skip = getattr(
                    self.model.cross_modal,
                    "record_text_idle_skip",
                    None,
                )
                if callable(record_text_skip):
                    record_text_skip(decay_traces=False)
                self._cross_modal_fast_idle_skip_count += 1
                if hasattr(self, "_cached_cross_modal_conf"):
                    cross_modal_visual_conf, cross_modal_audio_conf = (
                        self._cached_cross_modal_conf
                    )

            # Visual path — fire BEFORE text
            elif visual_spikes is not None:
                # Use L2-normalized raw pattern as text representation for
                # cross-modal Hebbian learning and alignment gates.
                text_spike = F.normalize(
                    x.detach().unsqueeze(0),
                    dim=1,
                ).squeeze(0)
                cross_modal_text_spike_prepared = True
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
            if not fast_text_idle_skip and audio_spikes is not None:
                if text_spike is None:
                    text_spike = F.normalize(
                        x.detach().unsqueeze(0),
                        dim=1,
                    ).squeeze(0)
                    cross_modal_text_spike_prepared = True
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

            text_has_sensory_evidence = bool(
                (visual_spikes is not None and cross_modal_visual_accepted)
                or (audio_spikes is not None and cross_modal_audio_accepted)
            )
            text_probe_interval = max(
                1,
                int(self.config.cross_modal_text_idle_probe_interval_tokens),
            )
            if not fast_text_idle_skip and text_has_sensory_evidence:
                self._cross_modal_traces_cleared_for_idle = False
                trace_window = max(
                    1,
                    int(round(float(self.config.cross_modal_tau_trace))),
                )
                self._cross_modal_sensory_trace_until_token = max(
                    self._cross_modal_sensory_trace_until_token,
                    self.token_count + trace_window,
                )
            has_cross_modal_trace = (
                self.token_count <= self._cross_modal_sensory_trace_until_token
            )
            text_probe_due = (
                not fast_text_idle_skip
                and (
                    text_has_sensory_evidence
                    or (
                        has_cross_modal_trace
                        and self.token_count % text_probe_interval == 0
                    )
                )
            )
            if text_probe_due:
                if text_spike is None:
                    text_spike = F.normalize(
                        x.detach().unsqueeze(0),
                        dim=1,
                    ).squeeze(0)
                    cross_modal_text_spike_prepared = True
                # Text spike LAST — now visual/audio traces contain current data
                self.model.cross_modal.on_text_spike(text_spike)
            elif not fast_text_idle_skip:
                record_text_skip = getattr(
                    self.model.cross_modal,
                    "record_text_idle_skip",
                    None,
                )
                if callable(record_text_skip):
                    record_text_skip()

            # Cross-modal confidence — periodic (every 10 steps) for metrics only
            if not fast_text_idle_skip and _telemetry_tick:
                cross_modal_visual_conf = float(self.model.cross_modal.visual_confidence.mean().item())
                cross_modal_audio_conf = float(self.model.cross_modal.audio_confidence.mean().item())
                self._cached_cross_modal_conf = (cross_modal_visual_conf, cross_modal_audio_conf)
            elif hasattr(self, "_cached_cross_modal_conf"):
                cross_modal_visual_conf, cross_modal_audio_conf = self._cached_cross_modal_conf

            # Buffer visual frames for self-criticism (§7.4)
            if not fast_text_idle_skip and visual_spikes is not None and cross_modal_visual_accepted:
                self._recent_visual_frames.append(vs.detach().clone())
                if len(self._recent_visual_frames) > self._visual_frame_limit:
                    self._recent_visual_frames = self._recent_visual_frames[-self._visual_frame_limit:]

            # Buffer audio frames for audio self-criticism (§7.4)
            if not fast_text_idle_skip and audio_spikes is not None and cross_modal_audio_accepted:
                self._recent_audio_frames.append(aus.detach().clone())
                if len(self._recent_audio_frames) > self._audio_frame_limit:
                    self._recent_audio_frames = self._recent_audio_frames[-self._audio_frame_limit:]

            # Periodic self-criticism loop (§7.4) — visual AND audio
            n_visual = len(self._recent_visual_frames)
            n_audio = len(self._recent_audio_frames)
            if (not fast_text_idle_skip
                    and self.token_count - self._last_self_criticism_token >= self._self_criticism_interval
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
        if profile_enabled:
            profile_now = time.perf_counter()
            profile_totals["cross_modal"] = profile_totals.get("cross_modal", 0.0) + (
                profile_now - profile_last
            ) * 1000.0
            profile_last = profile_now

        updated_indices = winners
        updated_id_list: list[int] | None = winner_id_list
        if int(self.model.competitive.last_revived_indices.numel()) > 0:
            updated_indices = torch.unique(
                torch.cat([winners, self.model.competitive.last_revived_indices.to(self.model.device)]),
                sorted=True,
            )
            updated_id_list = updated_indices.detach().cpu().tolist()
        if (
            int(self.model.competitive.last_revived_indices.numel()) == 0
            and self._column_transition_runtime.last_tick_used_device_owned_routing_cache()
        ):
            self._routing_index_device_update_count += int(
                updated_indices.numel()
            )
            self._routing_index_buffer_skip_count += int(
                updated_indices.numel()
            )
            self._routing_index_cpu_mirror_stale = True
        else:
            winner_vectors = self.model.competitive.prototypes[
                updated_indices
            ].detach()
            self._buffer_hnsw_update(
                updated_indices,
                winner_vectors,
                known_ids=updated_id_list,
            )
        if profile_enabled:
            profile_now = time.perf_counter()
            profile_totals["routing_index_buffer"] = profile_totals.get("routing_index_buffer", 0.0) + (
                profile_now - profile_last
            ) * 1000.0
            profile_last = profile_now

        next_token = self.token_count + 1
        warm_started = self._maybe_warm_start_memory(next_token)
        capture_tag = max(0.0, float(metrics["recon_error"]))
        memory_index = None
        slow_memory_archive_interval = max(
            1,
            int(self.config.slow_memory_archive_interval_tokens),
        )
        strong_capture_threshold = float(
            self.config.slow_memory_archive_strong_capture_threshold
        )
        slow_memory_archive_reason = "memory_not_warm"
        slow_memory_archive_due = False
        if self.memory_warm_started:
            if next_token == 1:
                slow_memory_archive_due = True
                slow_memory_archive_reason = "first_token"
            elif next_token % slow_memory_archive_interval == 0:
                slow_memory_archive_due = True
                slow_memory_archive_reason = "cadence"
            elif capture_tag >= strong_capture_threshold:
                slow_memory_archive_due = True
                slow_memory_archive_reason = "strong_capture"
            else:
                slow_memory_archive_reason = "cadence_skip"
        if profile_enabled:
            profile_now = time.perf_counter()
            profile_totals["memory_gate"] = profile_totals.get("memory_gate", 0.0) + (
                profile_now - profile_last
            ) * 1000.0
            profile_last = profile_now
        text_context = str(raw_window) if slow_memory_archive_due and raw_window is not None else None
        if profile_enabled:
            profile_now = time.perf_counter()
            profile_totals["stream_text_context"] = profile_totals.get(
                "stream_text_context", 0.0
            ) + (profile_now - profile_last) * 1000.0
            profile_last = profile_now
        if self.memory_warm_started and slow_memory_archive_due:
            _, archive_winner_id = _materialize_winner_ids()
            memory_index = self.model.memory_store.update(
                assembly,
                importance=max(1e-3, abs(effective_modulator)),
                token_count=next_token,
                bucket_id=archive_winner_id,
                input_pattern=x,
                routing_key=routing_key,
                raw_window=raw_window,
                text=text_context,
                metadata=memory_metadata,
                capture_tag=capture_tag,
            )
            self._slow_memory_archive_count += 1
        elif self.memory_warm_started:
            self._slow_memory_archive_skip_count += 1
        self._slow_memory_last_archive_reason = slow_memory_archive_reason
        if winner_id is not None:
            self.last_winner = int(winner_id)
            self._winner_host_mirror_sync_count += 1
            self._winner_host_mirror_fresh = True
        elif graph_winner_host_mirror_can_skip:
            self._winner_host_mirror_skip_count += 1
            self._winner_host_mirror_fresh = False
        else:
            _, current_winner_id = _materialize_winner_ids()
            self.last_winner = current_winner_id
            self._winner_host_mirror_sync_count += 1
            self._winner_host_mirror_fresh = True
        if profile_enabled:
            profile_now = time.perf_counter()
            profile_totals["memory_archive"] = profile_totals.get("memory_archive", 0.0) + (
                profile_now - profile_last
            ) * 1000.0
            profile_last = profile_now

        graph_competitive_surprise = (
            self._column_transition_runtime.consume_graph_competitive_surprise()
        )
        if (
            graph_competitive_surprise is None
            and not (
                prepared_routing is not None
                and self._column_transition_runtime.graph_owns_competitive_surprise()
            )
        ):
            winner_proto = self.model.competitive.prototypes[winners[0]]
            self.model.surprise.update("competitive", winner_proto, routing_key)
        elif graph_competitive_surprise is not None:
            self.model.surprise.record_error(
                "competitive",
                graph_competitive_surprise,
            )

        # Awake ripple tagging supports replay priority, so it follows the
        # slow-memory archive cadence instead of scanning memory every hot tick.
        da_level = self.model.surprise.dopamine
        awake_ripple_due = bool(
            self.memory_warm_started
            and slow_memory_archive_due
            and da_level > 0.7
        )
        if awake_ripple_due:
            tagged = self.model.memory_store.ripple_tag_awake(
                current_token=next_token,
                window_tokens=max(1, self.config.functional_minute // 2),
                da_level=da_level,
            )
            self._awake_ripple_tag_count += 1
            self._awake_ripple_last_tagged = int(tagged)
            self._awake_ripple_last_reason = str(slow_memory_archive_reason)
        elif da_level > 0.7 and self.memory_warm_started:
            self._awake_ripple_tag_skip_count += 1
            self._awake_ripple_last_tagged = 0
            self._awake_ripple_last_reason = "cadence_skip"
        elif not self.memory_warm_started:
            self._awake_ripple_last_reason = "memory_not_warm"
        else:
            self._awake_ripple_last_reason = "dopamine_below_threshold"
        if profile_enabled:
            profile_now = time.perf_counter()
            profile_totals["post_surprise_replay_tag"] = profile_totals.get(
                "post_surprise_replay_tag", 0.0
            ) + (profile_now - profile_last) * 1000.0
            profile_last = profile_now

        self.token_count = next_token
        if warm_started:
            drift_bucket = self.last_winner if self.config.use_winner_local_drift else None
            drift = self.model.memory_store.compute_drift(drift_bucket)
            metrics["drift"] = drift
            metrics["drift_floor"] = drift
        self.current_window_min_drift = min(self.current_window_min_drift, float(drift))
        if self.token_count % self.config.drift_floor_window_tokens == 0:
            self._close_drift_floor_window()

        if not bool(return_metrics):
            self._train_step_metrics_skip_count += 1
            if profile_enabled:
                profile_now = time.perf_counter()
                profile_totals["metrics_build_skipped"] = profile_totals.get(
                    "metrics_build_skipped",
                    0.0,
                ) + (profile_now - profile_last) * 1000.0
                profile_totals["total"] = profile_totals.get("total", 0.0) + (
                    profile_now - profile_started
                ) * 1000.0
                self._train_step_profile_count += 1
            return {}

        memory_stats = (
            self.model.memory_store.summary_stats()
            if self.memory_warm_started and _telemetry_tick
            else self._cached_memory_stats if hasattr(self, "_cached_memory_stats") else {}
        )
        if _telemetry_tick:
            self._cached_memory_stats = memory_stats
        self._train_step_metrics_full_count += 1

        metrics["token"] = self.token_count
        metrics["train_step_metrics_mode"] = "full"
        metrics["train_step_metrics_full_count"] = int(
            self._train_step_metrics_full_count
        )
        metrics["train_step_metrics_skip_count"] = int(
            self._train_step_metrics_skip_count
        )
        metrics["surprise"] = float(modulator)
        metrics["dopamine"] = float(self.model.surprise.dopamine)
        metrics["serotonin"] = float(self.model.surprise.serotonin)
        metrics["acetylcholine"] = float(self.model.surprise.acetylcholine)
        metrics["norepinephrine"] = float(self.model.surprise.norepinephrine)
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
        metrics["context_plasticity_interval_tokens"] = int(context_plasticity_interval)
        metrics["context_plasticity_due"] = int(context_plasticity_due)
        metrics["trainer_telemetry_interval_tokens"] = int(telemetry_interval)
        metrics["trainer_telemetry_due"] = int(_telemetry_tick)
        metrics["candidate_homeostasis_start_tokens"] = int(
            self.config.candidate_homeostasis_start_tokens
        )
        metrics["candidate_homeostasis_due"] = int(
            self.token_count
            >= int(self.config.candidate_homeostasis_start_tokens)
        )
        metrics["candidate_predictive_update_start_tokens"] = int(
            self.config.candidate_predictive_update_start_tokens
        )
        metrics["candidate_predictive_update_due"] = int(
            self.token_count
            >= int(self.config.candidate_predictive_update_start_tokens)
        )
        metrics["candidate_deep_sleep_filter_start_tokens"] = int(
            self.config.candidate_deep_sleep_filter_start_tokens
        )
        metrics["candidate_deep_sleep_filter_due"] = int(
            self.token_count
            >= int(self.config.candidate_deep_sleep_filter_start_tokens)
        )
        metrics["candidate_deep_sleep_filtered_count"] = int(
            self._column_wake_plan.filtered_deep_sleep_count
        )
        metrics["candidate_deep_sleep_filter_mode"] = str(
            self._column_wake_plan.mode
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
        if isinstance(binding_strength, torch.Tensor):
            if _telemetry_tick:
                self._cached_binding_strength = float(binding_strength.item())
            elif not hasattr(self, "_cached_binding_strength"):
                self._cached_binding_strength = 0.0
            metrics["binding_strength"] = self._cached_binding_strength
        else:
            metrics["binding_strength"] = float(binding_strength)
        binding_layer = self.model.binding_layer
        metrics["binding_runtime_active"] = bool(
            binding_layer is not None
            and getattr(binding_layer, "runtime_active", True)
        )
        metrics["binding_execution_mode"] = (
            str(getattr(binding_layer, "last_runtime_execution_mode", "not_run"))
            if binding_layer is not None
            else "disabled"
        )
        metrics["binding_runtime_bind_count"] = int(
            getattr(binding_layer, "runtime_bind_count", 0)
        )
        metrics["binding_runtime_idle_skip_count"] = int(
            getattr(binding_layer, "runtime_idle_skip_count", 0)
        )
        metrics["binding_idle_probe_interval_tokens"] = int(
            self.config.binding_idle_probe_interval_tokens
        )
        metrics["cross_modal_visual_confidence"] = cross_modal_visual_conf
        metrics["cross_modal_audio_confidence"] = cross_modal_audio_conf
        metrics["cross_modal_visual_accepted"] = cross_modal_visual_accepted
        metrics["cross_modal_audio_accepted"] = cross_modal_audio_accepted
        cross_modal_layer = self.model.cross_modal
        metrics["cross_modal_text_execution_mode"] = (
            str(getattr(cross_modal_layer, "last_text_runtime_execution_mode", "disabled"))
            if cross_modal_layer is not None
            else "disabled"
        )
        metrics["cross_modal_text_update_count"] = int(
            getattr(cross_modal_layer, "runtime_text_update_count", 0)
        )
        metrics["cross_modal_text_idle_skip_count"] = int(
            getattr(cross_modal_layer, "runtime_text_idle_skip_count", 0)
        )
        metrics["cross_modal_fast_idle_skip_count"] = int(
            self._cross_modal_fast_idle_skip_count
        )
        metrics["cross_modal_idle_trace_reset_count"] = int(
            self._cross_modal_idle_trace_reset_count
        )
        metrics["cross_modal_text_idle_probe_interval_tokens"] = int(
            self.config.cross_modal_text_idle_probe_interval_tokens
        )
        metrics["cross_modal_text_spike_prepared"] = int(
            cross_modal_text_spike_prepared
        )
        metrics["developmental_stage"] = self.developmental_stage
        metrics["winner"] = (
            int(winner_id)
            if winner_id is not None
            else None
            if self.last_winner is None
            else int(self.last_winner)
        )
        metrics["winner_host_mirror_fresh"] = int(self._winner_host_mirror_fresh)
        metrics["winner_host_mirror_sync_count"] = int(
            self._winner_host_mirror_sync_count
        )
        metrics["winner_host_mirror_skip_count"] = int(
            self._winner_host_mirror_skip_count
        )
        metrics["routing_index_device_update_count"] = int(
            self._routing_index_device_update_count
        )
        metrics["routing_index_buffer_skip_count"] = int(
            self._routing_index_buffer_skip_count
        )
        metrics["routing_index_host_mirror_sync_count"] = int(
            self._routing_index_host_mirror_sync_count
        )
        metrics["routing_index_cpu_mirror_stale"] = int(
            self._routing_index_cpu_mirror_stale
        )
        if _telemetry_tick:
            _active = int((assembly > 0).sum().item())
            _sparsity = float((assembly > 0).float().mean().item())
            self._cached_active_sparsity = (_active, _sparsity)
        elif not hasattr(self, "_cached_active_sparsity"):
            self._cached_active_sparsity = (0, 0.0)
        metrics["active_columns"], metrics["sparsity"] = self._cached_active_sparsity
        metrics["memory_index"] = None if memory_index is None else int(memory_index)
        metrics["slow_memory_archive_interval_tokens"] = int(
            slow_memory_archive_interval
        )
        metrics["slow_memory_archive_due"] = int(slow_memory_archive_due)
        metrics["slow_memory_archive_reason"] = str(slow_memory_archive_reason)
        metrics["slow_memory_archive_count"] = int(self._slow_memory_archive_count)
        metrics["slow_memory_archive_skip_count"] = int(
            self._slow_memory_archive_skip_count
        )
        metrics["awake_ripple_tag_count"] = int(self._awake_ripple_tag_count)
        metrics["awake_ripple_tag_skip_count"] = int(
            self._awake_ripple_tag_skip_count
        )
        metrics["awake_ripple_last_reason"] = str(self._awake_ripple_last_reason)
        metrics["awake_ripple_last_tagged"] = int(self._awake_ripple_last_tagged)
        if profile_enabled:
            profile_now = time.perf_counter()
            profile_totals["metrics_build"] = profile_totals.get("metrics_build", 0.0) + (
                profile_now - profile_last
            ) * 1000.0
            profile_totals["total"] = profile_totals.get("total", 0.0) + (
                profile_now - profile_started
            ) * 1000.0
            self._train_step_profile_count += 1
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
        candidates = self._routing_candidates(routing_key)
        if candidates is None:
            candidates = torch.arange(self.config.n_columns, device=self.model.device)

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
        candidates = self._routing_candidates(routing_key)
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
