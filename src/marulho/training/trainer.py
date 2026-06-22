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
from marulho.training.replay_anchor_window import (
    SLEEP_REPLAY_ANCHOR_BUCKET_WINDOW_LIMIT,
    sleep_replay_anchor_bucket_source_window,
)

SLEEP_REPLAY_ASSOCIATIVE_RECALL_QUERY_LIMIT = 4


class MarulhoTrainer:
    """Main stage-0 trainer."""

    def __init__(
        self,
        model: MarulhoModel,
        config: MarulhoConfig,
        *,
        defer_cuda_graph_route_transition: bool = False,
    ):
        self.model = model
        self.config = config
        self._defer_cuda_graph_route_transition = bool(
            defer_cuda_graph_route_transition
        )
        self._route_vote_mode_override_for_evaluation: str | None = None
        self._restored_route_candidate_bank_snapshot: dict[str, Any] | None = None
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
        self.column_anchors: dict[int, dict[str, torch.Tensor | float | int]] = {}
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

        # Routing-index update buffer - flush every N steps to amortize add() overhead.
        self._routing_index_buffer_ids: list[int | torch.Tensor] = []
        self._routing_index_buffer_vecs: list[torch.Tensor] = []
        self._routing_index_flush_interval = 16
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
        self._slow_memory_strong_capture_archive_count = 0
        self._slow_memory_strong_capture_refractory_skip_count = 0
        self._slow_memory_last_strong_capture_token = -1
        self._slow_memory_last_archive_reason = "not_run"
        self._last_sleep_replay_selection_report: dict[str, Any] = {
            "surface": "bounded_replay_window_selection.v1",
            "status": "not_run",
            "scope": "sleep_slow_path",
            "runs_live_tick": False,
            "sleep_replay_applied_count": 0,
            "sleep_replay_mutates_runtime_state": False,
            "sleep_replay_applies_plasticity": False,
        }
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
        self._text_burst_strong_archive_count = 0
        self._text_burst_strong_refractory_skip_count = 0
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
        self._column_structural_review_cuda_cadence_tokens = 4096

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
        report["text_burst_strong_archive_count"] = int(
            self._text_burst_strong_archive_count
        )
        report["text_burst_strong_refractory_skip_count"] = int(
            self._text_burst_strong_refractory_skip_count
        )
        report["slow_memory_strong_capture_min_interval_tokens"] = int(
            self._slow_memory_strong_capture_min_interval_tokens()
        )
        report["slow_memory_strong_capture_archive_count"] = int(
            self._slow_memory_strong_capture_archive_count
        )
        report["slow_memory_strong_capture_refractory_skip_count"] = int(
            self._slow_memory_strong_capture_refractory_skip_count
        )
        report["slow_memory_last_strong_capture_token"] = int(
            self._slow_memory_last_strong_capture_token
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

    def _slow_memory_strong_capture_min_interval_tokens(self) -> int:
        return max(
            2,
            int(
                getattr(
                    self.config,
                    "slow_memory_archive_strong_capture_min_interval_tokens",
                    16,
                )
            ),
        )

    def _slow_memory_strong_capture_allowed(self, token_marker: int) -> bool:
        if int(self._slow_memory_last_strong_capture_token) < 0:
            return True
        return (
            int(token_marker) - int(self._slow_memory_last_strong_capture_token)
        ) >= self._slow_memory_strong_capture_min_interval_tokens()

    def _record_slow_memory_strong_capture_archive(self, token_marker: int) -> None:
        self._slow_memory_last_strong_capture_token = int(token_marker)
        self._slow_memory_strong_capture_archive_count += 1

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
        archived_strong_count = 0
        for position, index in enumerate(strong_indices):
            token_marker, pattern, raw_window, metadata = pending[index]
            if not self._slow_memory_strong_capture_allowed(token_marker):
                continue
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
            self._record_slow_memory_strong_capture_archive(token_marker)
            archived_strong_count += 1
        event_count = len(pending)
        strong_count = len(strong_indices)
        refractory_skip_count = max(0, strong_count - archived_strong_count)
        self._text_burst_strong_event_count += strong_count
        self._text_burst_strong_archive_count += archived_strong_count
        self._text_burst_strong_refractory_skip_count += refractory_skip_count
        self._slow_memory_strong_capture_refractory_skip_count += refractory_skip_count
        self._slow_memory_archive_count += archived_strong_count
        self._slow_memory_archive_skip_count += event_count - archived_strong_count
        self._slow_memory_last_archive_reason = (
            "strong_capture"
            if archived_strong_count
            else ("strong_capture_refractory_skip" if strong_count else "cadence_skip")
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
        if self.model.surprise.dopamine > 0.7 and archived_strong_count:
            graph_runtime = getattr(graph, "_runtime", None)
            route_candidates = getattr(graph_runtime, "_route_candidates", None)
            awake_bucket_ids: torch.Tensor | list[int]
            if (
                isinstance(route_candidates, torch.Tensor)
                and int(route_candidates.numel()) > 0
            ):
                awake_bucket_ids = route_candidates
            else:
                awake_bucket_ids = [int(final_result[6])]
            tagged = self.model.memory_store.ripple_tag_awake(
                current_token=int(self.token_count),
                window_tokens=max(1, self.config.functional_minute // 2),
                da_level=self.model.surprise.dopamine,
                awake_bucket_ids=awake_bucket_ids,
                max_candidate_entries=self._recent_replay_setup_limit(),
            )
            self._awake_ripple_tag_count += 1
            self._awake_ripple_last_tagged = int(tagged)
            self._awake_ripple_last_reason = "strong_capture"
            self._awake_ripple_tag_skip_count += event_count - archived_strong_count
        elif self.model.surprise.dopamine > 0.7 and strong_count:
            self._awake_ripple_tag_skip_count += event_count
            self._awake_ripple_last_tagged = 0
            self._awake_ripple_last_reason = "strong_capture_refractory_skip"
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
            routing_index_flush_interval=self._routing_index_flush_interval,
            routing_index_buffer_pending=bool(
                self._routing_index_buffer_ids or self._routing_index_buffer_vecs
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

    def _host_truth_limited_burst_len(
        self,
        *,
        requested_len: int,
        cadence_tick_count: int,
    ) -> int:
        sync_interval = max(
            1,
            int(self.config.cuda_graph_host_truth_sync_interval_tokens),
        )
        requested = max(1, int(requested_len))
        cadence = max(0, int(cadence_tick_count))
        next_sync = sync_interval - (cadence % sync_interval)
        if next_sync <= 0:
            next_sync = sync_interval
        return max(1, min(requested, int(next_sync)))

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
        simulated_cadence = int(
            getattr(graph, "_host_truth_cadence_tick_count", graph.replay_count)
        )
        chunk_start = int(start)
        while chunk_start < int(end):
            requested_len = min(int(burst_capacity), int(end) - chunk_start)
            chunk_len = self._host_truth_limited_burst_len(
                requested_len=requested_len,
                cadence_tick_count=simulated_cadence,
            )
            chunk_end = chunk_start + chunk_len
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
            graph_start = graph._candidate_graph_plan_for_token(simulated_token)[0]
            graph_end = graph._candidate_graph_plan_for_token(
                simulated_token + chunk_len - 1
            )[0]
            if graph_start != graph_end:
                return False
            sync_interval = max(
                1,
                int(self.config.cuda_graph_host_truth_sync_interval_tokens),
            )
            sync_offsets = [
                offset
                for offset in range(1, chunk_len + 1)
                if (simulated_cadence + offset) % sync_interval == 0
            ]
            if sync_offsets not in ([], [chunk_len]):
                return False
            simulated_token += chunk_len
            simulated_cadence += chunk_len
            chunk_start = chunk_end
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
        simulated_cadence = int(
            getattr(graph, "_host_truth_cadence_tick_count", graph.replay_count)
        )
        chunk_start = int(start)
        while chunk_start < int(max_end):
            requested_len = min(int(burst_capacity), int(max_end) - chunk_start)
            chunk_len = self._host_truth_limited_burst_len(
                requested_len=requested_len,
                cadence_tick_count=simulated_cadence,
            )
            chunk_end = chunk_start + chunk_len
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
            graph_start = graph._candidate_graph_plan_for_token(simulated_token)[0]
            graph_end = graph._candidate_graph_plan_for_token(
                simulated_token + chunk_len - 1
            )[0]
            if graph_start != graph_end:
                break
            sync_interval = max(
                1,
                int(self.config.cuda_graph_host_truth_sync_interval_tokens),
            )
            sync_offsets = [
                offset
                for offset in range(1, chunk_len + 1)
                if (simulated_cadence + offset) % sync_interval == 0
            ]
            if sync_offsets not in ([], [chunk_len]):
                break
            best_end = chunk_end
            simulated_token += chunk_len
            simulated_cadence += chunk_len
            chunk_start = chunk_end
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
            if (
                int(
                    getattr(
                        graph,
                        "_host_truth_cadence_tick_count",
                        graph.replay_count,
                    )
                )
                + offset
            )
            % sync_interval
            == 0
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
        state_transition_count = int(runtime._route_candidates.numel())
        sparse_state_transition = state_transition_count < int(comp.n_columns)
        comp.last_state_transition_mode = (
            "candidate_subset_sparse_cuda_graph_route_transition_burst"
            if sparse_state_transition
            else "dense_all_columns_cuda_graph_route_transition_burst"
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
            "candidate_subset_sparse_cuda_graph_burst"
            if sparse_state_transition
            else "dense_cuda_graph_burst"
        )
        comp.last_state_transition_materialize_count = 0
        comp.last_state_transition_materialize_max_age = 0
        comp.state_transition_step_count += int(token_count)
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
        pred.last_dense_transition_mode = "inplace_triton"
        pred.last_dense_transition_fallback_reason = None
        candidate_predictive_graph = bool(
            runtime.candidate_predictive_transition_active
            and start
            >= int(self.config.candidate_predictive_update_start_tokens)
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
            runtime.record_candidate_predictive_transition_scope(
                n_columns=int(comp.n_columns),
                candidate_count=int(candidates.numel()),
                token_count=int(token_count),
            )
        else:
            pred._record_prediction_update_scope(None)
            pred._mark_predictive_update_complete(None, step_count=token_count)

        structural_review_due = (
            end
            % max(1, int(self._column_structural_review_cuda_cadence_tokens))
            == 0
        )
        if bool(burst_outputs.get("truth_synced", False)) and structural_review_due:
            self._record_column_structural_review(
                None,
                token_count=end,
                mode="cuda_graph_text_burst_host_truth",
                candidates=runtime._route_candidates,
            )
        else:
            self._record_column_structural_review(
                None,
                token_count=end,
                mode="cuda_graph_text_burst_deferred",
                deferred_reason=(
                    "structural_review_cuda_cadence_not_due"
                    if bool(burst_outputs.get("truth_synced", False))
                    else "host_truth_not_synced_for_structural_review_queue"
                ),
            )

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
        allow_sleep_maintenance: bool = True,
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
        fallback_train_step_count = 0
        fallback_sleep_maintenance_deferred_count = 0
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
            chunk_start = start
            while chunk_start < end:
                stage_sequence_segment(chunk_start)
                chunk_len = min(burst_capacity, end - chunk_start)
                graph = self._column_transition_runtime._cuda_graph_runtime
                if graph is not None:
                    chunk_len = self._host_truth_limited_burst_len(
                        requested_len=chunk_len,
                        cadence_tick_count=int(
                            getattr(
                                graph,
                                "_host_truth_cadence_tick_count",
                                graph.replay_count,
                            )
                        ),
                    )
                chunk_end = min(end, chunk_start + chunk_len)
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
                        sleep_replay_due = bool(
                            not allow_sleep_maintenance
                            and (
                                (
                                    self.token_count
                                    >= self.config.deep_sleep_interval_tokens
                                    and (
                                        self.token_count
                                        - self.last_deep_sleep_token
                                    )
                                    >= self.config.deep_sleep_interval_tokens
                                )
                                or (
                                    self.pending_emergency_deep_sleep
                                    and (
                                        self.token_count
                                        - self.last_deep_sleep_token
                                    )
                                    >= self.config.emergency_deep_sleep_cooldown_tokens
                                )
                                or (
                                    self.token_count
                                    >= self.config.micro_sleep_interval_tokens
                                    and (
                                        self.token_count
                                        - self.last_micro_sleep_token
                                    )
                                    >= self.config.micro_sleep_interval_tokens
                                )
                            )
                        )
                        fallback_train_step_count += 1
                        if sleep_replay_due:
                            fallback_sleep_maintenance_deferred_count += 1
                        metrics = self.train_step(
                            pattern,
                            raw_window=raw_window,
                            memory_metadata=memory_metadata,
                            return_metrics=return_metrics,
                            allow_sleep_maintenance=allow_sleep_maintenance,
                        )
                        if return_metrics:
                            metrics_by_index[index] = dict(metrics or {})
                            if metrics:
                                last_metrics = dict(metrics)
                trained += len(chunk_patterns)
                chunk_start = chunk_end
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
            "sleep_maintenance_allowed": bool(allow_sleep_maintenance),
            "fallback_train_step_count": int(fallback_train_step_count),
            "fallback_sleep_maintenance_deferred_count": int(
                fallback_sleep_maintenance_deferred_count
            ),
        }

    def _buffer_routing_index_update(
        self,
        indices: torch.Tensor,
        vectors: torch.Tensor,
        *,
        known_ids: list[int] | None = None,
    ) -> None:
        """Buffer routing-index updates; flush when buffer reaches interval size."""
        if known_ids is not None:
            ids: list[int | torch.Tensor] = [int(value) for value in known_ids]
        else:
            id_tensor = indices.detach().reshape(-1)
            if id_tensor.device.type == "cpu":
                ids = [int(value) for value in id_tensor.tolist()]
            else:
                ids = [id_tensor[i] for i in range(int(id_tensor.numel()))]
        if len(ids) != int(vectors.shape[0]):
            raise ValueError("known_ids must align with buffered routing-index vectors")
        vecs = vectors.detach()
        for i, vid in enumerate(ids):
            self._routing_index_buffer_ids.append(vid)
            self._routing_index_buffer_vecs.append(vecs[i])
        if len(self._routing_index_buffer_ids) >= self._routing_index_flush_interval:
            self._flush_routing_index_buffer()

    def _flush_routing_index_buffer(self) -> None:
        """Flush buffered routing-index updates in a single batch."""
        if not self._routing_index_buffer_ids:
            return
        if self._routing_index_cpu_mirror_stale:
            sync_host_store = getattr(
                self.model.routing_index,
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
        materialized_ids: list[int] = [0] * len(self._routing_index_buffer_ids)
        tensor_positions: list[int] = []
        tensor_ids: list[torch.Tensor] = []
        for position, vid in enumerate(self._routing_index_buffer_ids):
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
        for vid_int, vec in zip(materialized_ids, self._routing_index_buffer_vecs):
            seen[int(vid_int)] = vec
        ids_arr = np.array(list(seen.keys()), dtype=np.int64)
        vecs_batch = torch.stack(list(seen.values()))
        self.model.routing_index.add(vecs_batch, ids_arr)
        self._routing_index_buffer_ids.clear()
        self._routing_index_buffer_vecs.clear()

    def _build_candidate_sleep_filter_execution(
        self,
        *,
        mode: str,
        input_candidate_count: int,
        output_candidate_count: int,
        filtered_deep_sleep_count: int,
        backfill_candidate_count: int,
        fallback_reason: str | None,
        filtered_memory_pressure_count: int = 0,
        memory_pressure_threshold: float | None = None,
        memory_pressure_source: str | None = None,
        filtered_low_usefulness_count: int = 0,
        usefulness_threshold: float | None = None,
        usefulness_source: str | None = None,
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
            "filtered_memory_pressure_count": int(
                max(0, filtered_memory_pressure_count)
            ),
            "filtered_low_usefulness_count": int(
                max(0, filtered_low_usefulness_count)
            ),
            "backfill_candidate_count": int(max(0, backfill_candidate_count)),
            "deep_sleep_threshold_steps": int(self.config.dead_column_steps),
            "start_token": int(self.config.candidate_deep_sleep_filter_start_tokens),
            "backfill_factor": int(self.config.candidate_deep_sleep_backfill_factor),
            "memory_pressure_threshold": memory_pressure_threshold,
            "memory_pressure_source": memory_pressure_source,
            "usefulness_threshold": usefulness_threshold,
            "usefulness_source": usefulness_source,
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
        filtered_memory_pressure_count: int = 0,
        memory_pressure_threshold: float | None = None,
        memory_pressure_source: str | None = None,
        filtered_low_usefulness_count: int = 0,
        usefulness_threshold: float | None = None,
        usefulness_source: str | None = None,
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
            filtered_memory_pressure_count=int(
                max(0, filtered_memory_pressure_count)
            ),
            memory_pressure_threshold=memory_pressure_threshold,
            memory_pressure_source=memory_pressure_source,
            filtered_low_usefulness_count=int(
                max(0, filtered_low_usefulness_count)
            ),
            usefulness_threshold=usefulness_threshold,
            usefulness_source=usefulness_source,
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

    def _candidate_memory_pressure_filter_due(self, *, apply_sleep_filter: bool) -> bool:
        source = str(
            getattr(
                getattr(self.model, "column_metabolism", None),
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
            apply_sleep_filter
            and int(self.token_count)
            >= int(self.config.candidate_memory_pressure_filter_start_tokens)
            and has_pressure_evidence
        )

    def _candidate_usefulness_filter_due(self, *, apply_sleep_filter: bool) -> bool:
        source = str(
            getattr(
                getattr(self.model, "column_metabolism", None),
                "last_usefulness_source",
                "not_run",
            )
        )
        has_usefulness_evidence = source not in {
            "not_run",
            "no_awake_candidates",
        }
        return bool(
            apply_sleep_filter
            and int(self.token_count)
            >= int(self.config.candidate_usefulness_filter_start_tokens)
            and has_usefulness_evidence
        )

    def _filter_candidate_memory_pressure_plan(
        self,
        candidates: torch.Tensor,
        *,
        target_count: int,
        mode: str = "candidate_memory_pressure_filter",
        wake_reason: str = "retrieved_candidate_below_memory_pressure_threshold",
        deep_sleep_filtered_count: int = 0,
        fallback_reason: str | None = None,
        record: bool = True,
    ) -> ColumnWakePlan:
        metabolism = getattr(self.model, "column_metabolism", None)
        candidate_count = int(candidates.numel())
        if metabolism is None:
            bounded = candidates[: max(0, int(target_count))]
            plan = self._build_column_wake_plan(
                mode=f"{mode}_unavailable",
                awake_indices=bounded,
                input_candidate_count=candidate_count,
                filtered_deep_sleep_count=deep_sleep_filtered_count,
                filtered_memory_pressure_count=0,
                backfill_candidate_count=max(
                    0,
                    candidate_count - int(bounded.numel()),
                ),
                fallback_reason=(
                    fallback_reason
                    or "column_metabolism_state_unavailable"
                ),
                wake_reason=wake_reason,
                sleep_reason=None,
            )
            if record:
                self._record_column_wake_plan(plan)
            return plan

        filtered, report = metabolism.filter_candidates(
            candidates,
            target_count=target_count,
            threshold=float(self.config.candidate_memory_pressure_threshold),
            usefulness_threshold=(
                float(self.config.candidate_usefulness_threshold)
                if self._candidate_usefulness_filter_due(apply_sleep_filter=True)
                else None
            ),
        )
        pressure_filtered_count = int(
            report.get("filtered_memory_pressure_count", 0) or 0
        )
        usefulness_filtered_count = int(
            report.get("filtered_low_usefulness_count", 0) or 0
        )
        pressure_fallback = report.get("fallback_reason")
        plan = self._build_column_wake_plan(
            mode=str(report.get("mode") or mode),
            awake_indices=filtered,
            input_candidate_count=candidate_count,
            filtered_deep_sleep_count=deep_sleep_filtered_count,
            filtered_memory_pressure_count=pressure_filtered_count,
            filtered_low_usefulness_count=usefulness_filtered_count,
            backfill_candidate_count=max(0, candidate_count - int(filtered.numel())),
            fallback_reason=(
                fallback_reason
                if fallback_reason is not None
                else None
                if pressure_fallback is None
                else str(pressure_fallback)
            ),
            wake_reason=wake_reason,
            sleep_reason=(
                "memory_pressure_or_low_usefulness_candidate_filtered_from_awake_mask"
                if pressure_filtered_count > 0 and usefulness_filtered_count > 0
                else "memory_pressure_candidate_filtered_from_awake_mask"
                if pressure_filtered_count > 0
                else "low_usefulness_candidate_filtered_from_awake_mask"
                if usefulness_filtered_count > 0
                else None
            ),
            memory_pressure_threshold=float(report.get("threshold", 0.0) or 0.0),
            memory_pressure_source=str(
                report.get(
                    "memory_pressure_source",
                    getattr(metabolism, "last_memory_pressure_source", "unknown"),
                )
            ),
            usefulness_threshold=(
                None
                if report.get("usefulness_threshold") is None
                else float(report.get("usefulness_threshold"))
            ),
            usefulness_source=str(
                report.get(
                    "usefulness_source",
                    getattr(metabolism, "last_usefulness_source", "unknown"),
                )
            ),
        )
        if record:
            self._record_column_wake_plan(plan)
        return plan

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

        pressure_plan: ColumnWakePlan | None = None
        if self._candidate_memory_pressure_filter_due(apply_sleep_filter=True):
            pressure_plan = self._filter_candidate_memory_pressure_plan(
                awake_candidates,
                target_count=target,
                mode="candidate_deep_sleep_memory_pressure_filter",
                wake_reason="retrieved_candidate_not_in_deep_sleep_or_memory_pressure",
                deep_sleep_filtered_count=filtered_count,
                record=False,
            )
            filtered = pressure_plan.candidates()
        else:
            filtered = awake_candidates[:target]

        pressure_filtered_count = (
            0
            if pressure_plan is None
            else int(pressure_plan.filtered_memory_pressure_count)
        )
        usefulness_filtered_count = (
            0
            if pressure_plan is None
            else int(pressure_plan.filtered_low_usefulness_count)
        )
        pressure_threshold = (
            None
            if pressure_plan is None
            else pressure_plan.memory_pressure_threshold
        )
        pressure_source = (
            None
            if pressure_plan is None
            else pressure_plan.memory_pressure_source
        )
        usefulness_threshold = (
            None
            if pressure_plan is None
            else pressure_plan.usefulness_threshold
        )
        usefulness_source = (
            None
            if pressure_plan is None
            else pressure_plan.usefulness_source
        )
        pressure_fallback = (
            None
            if pressure_plan is None
            else pressure_plan.fallback_reason
        )
        plan = self._build_column_wake_plan(
            mode=(
                "candidate_deep_sleep_memory_pressure_filter"
                if pressure_plan is not None
                else "candidate_deep_sleep_filter"
            ),
            awake_indices=filtered,
            input_candidate_count=candidate_count,
            filtered_deep_sleep_count=filtered_count,
            filtered_memory_pressure_count=pressure_filtered_count,
            filtered_low_usefulness_count=usefulness_filtered_count,
            backfill_candidate_count=max(0, candidate_count - int(filtered.numel())),
            fallback_reason=(
                pressure_fallback
                if pressure_fallback is not None
                else None
                if int(filtered.numel()) >= min(target, candidate_count)
                else "insufficient_awake_candidates_after_deep_sleep_filter"
            ),
            wake_reason=(
                "retrieved_candidate_not_in_deep_sleep_or_memory_pressure"
                if pressure_plan is not None
                else "retrieved_candidate_not_in_deep_sleep"
            ),
            sleep_reason=(
                "deep_sleep_memory_pressure_or_low_usefulness_candidate_filtered_from_awake_mask"
                if pressure_filtered_count > 0 and usefulness_filtered_count > 0
                else "deep_sleep_or_memory_pressure_candidate_filtered_from_awake_mask"
                if pressure_filtered_count > 0
                else "deep_sleep_or_low_usefulness_candidate_filtered_from_awake_mask"
                if usefulness_filtered_count > 0
                else "deep_sleep_candidate_filtered_from_awake_mask"
            ),
            memory_pressure_threshold=pressure_threshold,
            memory_pressure_source=pressure_source,
            usefulness_threshold=usefulness_threshold,
            usefulness_source=usefulness_source,
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
        pressure_enabled = bool(snapshot.get("memory_pressure_enabled", False))
        fallback_reason = snapshot.get("fallback_reason")
        pressure_filtered_count = int(
            snapshot.get("filtered_memory_pressure_count", 0) or 0
        )
        usefulness_enabled = bool(snapshot.get("usefulness_enabled", False))
        usefulness_filtered_count = int(
            snapshot.get("filtered_low_usefulness_count", 0) or 0
        )
        pressure_threshold = snapshot.get("memory_pressure_threshold")
        pressure_source = snapshot.get("memory_pressure_source")
        usefulness_threshold = snapshot.get("usefulness_threshold")
        usefulness_source = snapshot.get("usefulness_source")
        if filter_enabled or pressure_enabled or usefulness_enabled:
            if usefulness_enabled:
                filter_parts: list[str] = []
                if filter_enabled:
                    filter_parts.append("deep_sleep")
                if pressure_enabled:
                    filter_parts.append("memory_pressure")
                filter_parts.append("usefulness")
                mode = "candidate_" + "_".join(filter_parts) + "_filter_route_vote"
                if fallback_reason:
                    mode = f"{mode}_fallback"
                wake_reason = (
                    "route_vote_primary_score_passed_"
                    + "_".join(filter_parts)
                    + "_gates"
                )
                sleep_parts: list[str] = []
                if filter_enabled:
                    sleep_parts.append("deep_sleep")
                if pressure_filtered_count > 0:
                    sleep_parts.append("memory_pressure")
                if usefulness_filtered_count > 0:
                    sleep_parts.append("low_usefulness")
                sleep_reason = (
                    "_or_".join(sleep_parts) + "_route_score_masked_before_topk_vote"
                    if sleep_parts
                    else "usefulness_route_score_gate_evaluated_before_topk_vote"
                )
            else:
                mode = (
                    "candidate_deep_sleep_memory_pressure_filter_route_vote"
                    if filter_enabled and pressure_enabled and not fallback_reason
                    else "candidate_deep_sleep_memory_pressure_filter_route_vote_fallback"
                    if filter_enabled and pressure_enabled
                    else "candidate_deep_sleep_filter_route_vote_fallback"
                    if filter_enabled and fallback_reason
                    else "candidate_deep_sleep_filter_route_vote"
                    if filter_enabled
                    else "candidate_memory_pressure_filter_route_vote_fallback"
                    if fallback_reason
                    else "candidate_memory_pressure_filter_route_vote"
                )
                wake_reason = (
                    "route_vote_primary_score_not_in_deep_sleep_or_memory_pressure"
                    if filter_enabled and pressure_enabled
                    else "route_vote_primary_score_not_in_deep_sleep"
                    if filter_enabled
                    else "route_vote_primary_score_below_memory_pressure_threshold"
                )
                sleep_reason = (
                    "deep_sleep_or_memory_pressure_route_score_masked_before_topk_vote"
                    if filter_enabled and pressure_enabled and pressure_filtered_count > 0
                    else "deep_sleep_route_score_masked_before_topk_vote"
                    if filter_enabled and pressure_filtered_count <= 0
                    else "memory_pressure_route_score_masked_before_topk_vote"
                )
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
            filtered_memory_pressure_count=pressure_filtered_count,
            filtered_low_usefulness_count=usefulness_filtered_count,
            backfill_candidate_count=backfill_count,
            fallback_reason=fallback_reason,
            wake_reason=wake_reason,
            sleep_reason=sleep_reason,
            memory_pressure_threshold=(
                float(pressure_threshold)
                if pressure_threshold is not None
                else None
            ),
            memory_pressure_source=(
                str(pressure_source)
                if pressure_source is not None
                else None
            ),
            usefulness_threshold=(
                float(usefulness_threshold)
                if usefulness_threshold is not None
                else None
            ),
            usefulness_source=(
                str(usefulness_source)
                if usefulness_source is not None
                else None
            ),
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
        memory_pressure_due = self._candidate_memory_pressure_filter_due(
            apply_sleep_filter=apply_sleep_filter,
        )
        usefulness_due = self._candidate_usefulness_filter_due(
            apply_sleep_filter=apply_sleep_filter,
        )
        search_k = target_k
        backfill_factor = 1
        if filter_due and self.model.device.type != "cuda":
            backfill_factor = max(
                backfill_factor,
                int(self.config.candidate_deep_sleep_backfill_factor),
            )
        if memory_pressure_due:
            backfill_factor = max(
                backfill_factor,
                int(self.config.candidate_memory_pressure_backfill_factor),
            )
        if usefulness_due:
            backfill_factor = max(
                backfill_factor,
                int(self.config.candidate_usefulness_backfill_factor),
            )
        if backfill_factor > 1:
            search_k = min(
                int(self.config.n_columns),
                target_k * max(1, int(backfill_factor)),
            )
        candidate_ids, candidate_distances = self.model.routing_index.search_tensors(
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
            if memory_pressure_due or usefulness_due:
                plan = self._filter_candidate_memory_pressure_plan(
                    candidates,
                    target_count=target_k,
                )
                return plan
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
            if memory_pressure_due or usefulness_due:
                plan = self._filter_candidate_memory_pressure_plan(
                    candidates,
                    target_count=target_k,
                )
                return plan
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

    def _cached_bucket_consolidation_for_column_metabolism(self) -> torch.Tensor | None:
        store = getattr(self.model, "memory_store", None)
        cache = getattr(store, "_bucket_consolidation_devices", None)
        if not isinstance(cache, dict):
            return None
        cached = cache.get(str(self.model.device))
        if cached is None and self.model.device.type == "cpu":
            cached = cache.get("cpu")
        if (
            isinstance(cached, torch.Tensor)
            and int(cached.numel()) == int(self.config.n_columns)
        ):
            return cached
        return None

    def _record_column_metabolism(self, wake_plan: ColumnWakePlan | None) -> None:
        metabolism = getattr(self.model, "column_metabolism", None)
        if metabolism is None:
            return
        if isinstance(wake_plan, ColumnWakePlan):
            candidates = wake_plan.candidates()
            input_candidate_count = int(wake_plan.input_candidate_count)
            awake_budget = int(wake_plan.awake_budget)
        else:
            candidates = None
            input_candidate_count = 0
            awake_budget = int(min(self.config.k_routing, self.config.n_columns))
        metabolism.record_awake(
            candidates,
            token_count=int(self.token_count),
            awake_budget=awake_budget,
            input_candidate_count=input_candidate_count,
            memory_consolidation=self._cached_bucket_consolidation_for_column_metabolism(),
            confidence=getattr(self.model.predictive, "confidence", None),
            prediction_error=getattr(self.model.predictive, "prediction_error", None),
            win_rate_ema=getattr(self.model.competitive, "win_rate_ema", None),
        )

    def _record_column_structural_review(
        self,
        wake_plan: ColumnWakePlan | None,
        *,
        token_count: int,
        mode: str,
        candidates: torch.Tensor | None = None,
        deferred_reason: str | None = None,
    ) -> None:
        queue = getattr(self.model, "column_structural_review_queue", None)
        if queue is None:
            return
        if deferred_reason is not None:
            queue.record_deferred(
                token_count=int(token_count),
                mode=str(mode),
                reason=str(deferred_reason),
            )
            return
        review_candidates = candidates
        wake_reason = None
        sleep_reason = None
        if isinstance(wake_plan, ColumnWakePlan):
            if review_candidates is None:
                review_candidates = wake_plan.candidates()
            wake_reason = wake_plan.wake_reason
            sleep_reason = wake_plan.sleep_reason
        queue.record_candidates(
            review_candidates,
            token_count=int(token_count),
            mode=str(mode),
            prediction_error=getattr(self.model.predictive, "prediction_error", None),
            confidence=getattr(self.model.predictive, "confidence", None),
            prediction_failure_streak=getattr(
                self.model.predictive,
                "prediction_failure_streak",
                None,
            ),
            estimated_cost=getattr(self.model.column_metabolism, "estimated_cost", None),
            memory_pressure=getattr(self.model.column_metabolism, "memory_pressure", None),
            usefulness=getattr(self.model.column_metabolism, "usefulness", None),
            wake_reason=wake_reason,
            sleep_reason=sleep_reason,
        )


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

    @staticmethod
    def _sleep_replay_trace_signature(
        input_pattern: torch.Tensor | None,
        routing_key: torch.Tensor,
    ) -> tuple[tuple[int, int], ...]:
        source = input_pattern if isinstance(input_pattern, torch.Tensor) else routing_key
        flat = source.detach().cpu().float().flatten()
        if int(flat.numel()) <= 0:
            return ()
        active = torch.nonzero(flat.abs() > 1e-8, as_tuple=False).flatten()
        if 0 < int(active.numel()) <= 32:
            return tuple(
                (int(index), int(round(float(flat[int(index)].item()) * 1_000_000)))
                for index in active.tolist()
            )
        k = min(32, int(flat.numel()))
        values, indices = torch.topk(flat.abs(), k=k)
        return tuple(
            (
                int(index),
                int(round(float(flat[int(index)].item()) * 1_000_000)),
            )
            for value, index in zip(values.tolist(), indices.tolist())
            if float(value) > 0.0
        )

    def _bounded_replay_reconstruction_error(
        self,
        routing_keys: Sequence[torch.Tensor],
        candidate_ids: Sequence[int],
    ) -> float:
        if not routing_keys or not candidate_ids:
            return float("inf")
        unique_candidates = sorted(
            {
                int(candidate_id)
                for candidate_id in candidate_ids
                if 0 <= int(candidate_id) < int(self.config.n_columns)
            }
        )
        if not unique_candidates:
            return float("inf")
        candidates = torch.tensor(
            unique_candidates,
            device=self.model.device,
            dtype=torch.long,
        )
        prototypes = self.model.competitive.prototypes[candidates]
        total = 0.0
        for routing_key in routing_keys:
            target = F.normalize(routing_key.to(self.model.device), dim=0)
            similarities = torch.mv(prototypes, target)
            best = float(similarities.max().item()) if int(similarities.numel()) > 0 else -1.0
            total += max(0.0, 1.0 - best)
        return float(total / max(1, len(routing_keys)))

    def _sleep_replay_bounded_candidate_repair(
        self,
        replay_idx: Sequence[int],
        *,
        repair_strength: float = 1.0,
    ) -> tuple[int, list[int], list[int], dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen_signatures: set[tuple[tuple[int, int], ...]] = set()
        duplicate_trace_skips = 0
        invalid_trace_skips = 0
        missing_routing_key_skips = 0
        stored_routing_key_count = 0
        no_candidate_skips = 0
        stored_bucket_candidate_injections = 0
        candidate_union: list[int] = []

        for raw_idx in replay_idx:
            idx = int(raw_idx)
            replay_entry = self.model.memory_store.replay_entry(
                idx,
                current_token=self.token_count,
                include_text_payload=False,
            )
            assembly = replay_entry["assembly"]
            input_pattern = replay_entry["input_pattern"]
            stored_routing_key = replay_entry["routing_key"]
            stored_bucket_id = replay_entry["bucket_id"]

            if not isinstance(assembly, torch.Tensor):
                invalid_trace_skips += 1
                continue
            assembly = assembly.to(self.model.device)
            if assembly.dim() != 1 or float(assembly.abs().sum().item()) <= 0.0:
                invalid_trace_skips += 1
                continue

            replay_input = input_pattern.to(self.model.device) if isinstance(input_pattern, torch.Tensor) else None
            if isinstance(stored_routing_key, torch.Tensor):
                stored_routing_key_count += 1
                routing_key = F.normalize(stored_routing_key.to(self.model.device), dim=0)
            else:
                missing_routing_key_skips += 1
                continue
            if float(routing_key.abs().sum().item()) <= 0.0:
                invalid_trace_skips += 1
                continue

            signature = self._sleep_replay_trace_signature(replay_input, routing_key)
            if signature in seen_signatures:
                duplicate_trace_skips += 1
                continue
            seen_signatures.add(signature)

            candidates_tensor = self._routing_candidates(routing_key)
            candidate_ids: list[int] = []
            if isinstance(candidates_tensor, torch.Tensor) and int(candidates_tensor.numel()) > 0:
                for candidate in candidates_tensor.detach().cpu().tolist():
                    candidate_id = int(candidate)
                    if 0 <= candidate_id < int(self.config.n_columns) and candidate_id not in candidate_ids:
                        candidate_ids.append(candidate_id)

            if stored_bucket_id is not None:
                stored_candidate = int(stored_bucket_id)
                if (
                    0 <= stored_candidate < int(self.config.n_columns)
                    and stored_candidate not in candidate_ids
                ):
                    candidate_ids.append(stored_candidate)
                    stored_bucket_candidate_injections += 1

            if not candidate_ids:
                no_candidate_skips += 1
                continue

            for candidate_id in candidate_ids:
                if candidate_id not in candidate_union:
                    candidate_union.append(candidate_id)
            records.append(
                {
                    "index": idx,
                    "routing_key": routing_key,
                    "candidate_ids": candidate_ids,
                }
            )

        routing_keys = [record["routing_key"] for record in records]
        quality_before = (
            self._bounded_replay_reconstruction_error(routing_keys, candidate_union)
            if records
            else None
        )
        quality_current = float(quality_before) if quality_before is not None else None
        used_columns: set[int] = set()
        updated_ids: list[int] = []
        processed_indices: list[int] = []
        rejected_commits = 0
        candidate_trial_count = 0
        epsilon = 1e-10

        for record in records:
            if quality_current is None:
                rejected_commits += 1
                continue
            before_score = float(quality_current)
            best_column: int | None = None
            best_score = before_score
            routing_key = record["routing_key"]
            for candidate_id in record["candidate_ids"]:
                candidate = int(candidate_id)
                if candidate in used_columns:
                    continue
                original_proto = self.model.competitive.prototypes[candidate].detach().clone()
                original_velocity = self.model.competitive.prototype_velocity[candidate].detach().clone()
                self._repair_column_from_replay(
                    candidate,
                    routing_key,
                    strength=repair_strength,
                )
                candidate_trial_count += 1
                score = self._bounded_replay_reconstruction_error(
                    routing_keys,
                    candidate_union,
                )
                self.model.competitive.prototypes[candidate] = original_proto
                self.model.competitive.prototype_velocity[candidate] = original_velocity
                if score < best_score - epsilon:
                    best_score = score
                    best_column = candidate

            if best_column is None:
                rejected_commits += 1
                continue

            self._repair_column_from_replay(
                best_column,
                routing_key,
                strength=repair_strength,
            )
            used_columns.add(best_column)
            updated_ids.append(best_column)
            processed_indices.append(int(record["index"]))
            quality_current = float(best_score)

        applied = len(processed_indices)
        commit_report = {
            "sleep_replay_commit_strategy": "bounded_reconstruction_gated_candidate_repair",
            "sleep_replay_winner_source": "bounded_route_candidates",
            "sleep_replay_forced_stored_bucket_winner": False,
            "sleep_replay_selected_trace_count": int(len(replay_idx)),
            "sleep_replay_unique_trace_count": int(len(records)),
            "sleep_replay_duplicate_trace_skip_count": int(duplicate_trace_skips),
            "sleep_replay_invalid_trace_skip_count": int(invalid_trace_skips),
            "sleep_replay_stored_routing_key_count": int(stored_routing_key_count),
            "sleep_replay_missing_routing_key_count": int(missing_routing_key_skips),
            "sleep_replay_missing_routing_key_deferred_count": int(
                missing_routing_key_skips
            ),
            "sleep_replay_local_trace_prepare_policy": (
                "stored_routing_key_required_missing_keys_deferred"
            ),
            "sleep_replay_no_candidate_skip_count": int(no_candidate_skips),
            "sleep_replay_rejected_commit_count": int(rejected_commits),
            "sleep_replay_candidate_column_union_count": int(len(candidate_union)),
            "sleep_replay_candidate_column_trial_count": int(candidate_trial_count),
            "sleep_replay_stored_bucket_candidate_injection_count": int(
                stored_bucket_candidate_injections
            ),
            "sleep_replay_updated_column_count": int(len(set(updated_ids))),
            "sleep_replay_quality_metric": (
                "mean_one_minus_best_similarity_over_selected_replay_routing_keys"
            ),
            "sleep_replay_quality_scope": "selected_replay_window_candidate_columns",
            "sleep_replay_quality_before": quality_before,
            "sleep_replay_quality_after": quality_current,
            "sleep_replay_quality_delta": (
                None
                if quality_before is None or quality_current is None
                else float(quality_before - quality_current)
            ),
            "sleep_replay_repair_strength": float(repair_strength),
        }
        return applied, updated_ids, processed_indices, commit_report

    def _sleep_replay_candidate_bucket_ids(
        self,
        mode: str,
    ) -> tuple[list[int] | None, dict[str, Any]]:
        return sleep_replay_anchor_bucket_source_window(
            self,
            mode=mode,
            max_buckets=SLEEP_REPLAY_ANCHOR_BUCKET_WINDOW_LIMIT,
        )

    def _empty_sleep_replay_associative_recall_report(
        self,
        *,
        mode: str,
        candidate_bucket_ids: Sequence[int] | None,
        fallback_reason: str,
    ) -> dict[str, Any]:
        return {
            "surface": "bounded_sleep_replay_associative_recall.v1",
            "status": "empty",
            "sleep_mode": str(mode),
            "scope": f"{mode}_sleep_replay_window_associative_recall",
            "selection_criteria": [
                "reuse_selected_sleep_replay_indices_as_queries",
                "recall_only_inside_anchor_bucket_window",
                "tensor_payloads_only_no_raw_text",
            ],
            "candidate_bucket_ids": [int(value) for value in (candidate_bucket_ids or [])],
            "candidate_bucket_count": int(len(candidate_bucket_ids or [])),
            "selected_replay_index_count": 0,
            "query_budget": int(SLEEP_REPLAY_ASSOCIATIVE_RECALL_QUERY_LIMIT),
            "query_count": 0,
            "query_indices": [],
            "report_count": 0,
            "candidate_scope": "bucket_indexed_candidate_window"
            if candidate_bucket_ids is not None
            else "bucket_index_scope_required",
            "candidate_window_limit": 0,
            "candidate_index_count": 0,
            "selected_count_total": 0,
            "routing_key_count_total": 0,
            "input_pattern_count_total": 0,
            "mean_best_distance": None,
            "mean_best_input_distance": None,
            "mean_recalled_distance": None,
            "exact_input_recall_count": 0,
            "quality_metric": "mean_best_input_distance_over_selected_sleep_replay_queries",
            "quality_pass": False,
            "query_prepare_missing_routing_key_count": 0,
            "query_prepare_missing_input_pattern_count": 0,
            "score_device": "cpu",
            "archival_storage_device": "cpu",
            "device_placement": {
                "archival_storage_device": "cpu",
                "source_window_device": "cpu",
                "score_device": "cpu",
                "gpu_used": False,
                "gpu_resident_archival_metadata": False,
            },
            "runs_live_tick": False,
            "runs_every_token": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "fallback_reason": fallback_reason,
            "reports": [],
        }

    def _sleep_replay_associative_recall(
        self,
        replay_idx: Sequence[int],
        *,
        mode: str,
        candidate_bucket_ids: Sequence[int] | None,
        max_candidates: int,
    ) -> dict[str, Any]:
        if mode != "deep":
            return self._empty_sleep_replay_associative_recall_report(
                mode=mode,
                candidate_bucket_ids=candidate_bucket_ids,
                fallback_reason="associative_recall_enabled_for_deep_sleep_only",
            )
        if not replay_idx:
            return self._empty_sleep_replay_associative_recall_report(
                mode=mode,
                candidate_bucket_ids=candidate_bucket_ids,
                fallback_reason="empty_selected_sleep_replay_window",
            )
        if candidate_bucket_ids is None:
            return self._empty_sleep_replay_associative_recall_report(
                mode=mode,
                candidate_bucket_ids=candidate_bucket_ids,
                fallback_reason="anchor_bucket_scope_required_for_sleep_replay_recall",
            )

        query_limit = min(
            int(SLEEP_REPLAY_ASSOCIATIVE_RECALL_QUERY_LIMIT),
            max(0, len(replay_idx)),
        )
        queries: list[tuple[int, torch.Tensor, torch.Tensor | None]] = []
        missing_routing_key_count = 0
        missing_input_pattern_count = 0
        for raw_idx in replay_idx[:query_limit]:
            idx = int(raw_idx)
            try:
                replay_entry = self.model.memory_store.replay_entry(
                    idx,
                    current_token=self.token_count,
                    include_text_payload=False,
                )
            except IndexError:
                missing_routing_key_count += 1
                continue
            routing_key = replay_entry.get("routing_key")
            input_pattern = replay_entry.get("input_pattern")
            if not isinstance(routing_key, torch.Tensor):
                if isinstance(input_pattern, torch.Tensor):
                    routing_key = (
                        self.model.routing_key_from_pattern(
                            input_pattern.to(self.model.device)
                        )
                        .detach()
                        .cpu()
                    )
                else:
                    missing_routing_key_count += 1
                    continue
            if not isinstance(input_pattern, torch.Tensor):
                missing_input_pattern_count += 1
                input_pattern = None
            queries.append(
                (
                    idx,
                    routing_key.detach().clone().cpu(),
                    input_pattern.detach().clone().cpu()
                    if isinstance(input_pattern, torch.Tensor)
                    else None,
                )
            )

        if not queries:
            report = self._empty_sleep_replay_associative_recall_report(
                mode=mode,
                candidate_bucket_ids=candidate_bucket_ids,
                fallback_reason="no_tensor_queries_in_selected_sleep_replay_window",
            )
            report.update(
                {
                    "selected_replay_index_count": int(len(replay_idx)),
                    "query_prepare_missing_routing_key_count": int(
                        missing_routing_key_count
                    ),
                    "query_prepare_missing_input_pattern_count": int(
                        missing_input_pattern_count
                    ),
                }
            )
            self.model.memory_store.last_replay_recall_report = dict(
                self.model.memory_store._empty_replay_recall_report()
            )
            self.model.memory_store.last_replay_recall_report.update(
                {
                    "scope": f"{mode}_sleep_replay_window_associative_recall",
                    "fallback_reason": report["fallback_reason"],
                }
            )
            return report

        reports: list[dict[str, Any]] = []
        best_distances: list[float] = []
        best_input_distances: list[float] = []
        recalled_distances: list[float] = []
        for _idx, routing_key, input_pattern in queries:
            recall_report = self.model.memory_store.recall_replay_window(
                query_routing_key=routing_key,
                query_input_pattern=input_pattern,
                current_token=self.token_count,
                candidate_bucket_ids=candidate_bucket_ids,
                max_candidates=max(1, int(max_candidates)),
                strategy="consolidation",
                scope=f"{mode}_sleep_replay_window_associative_recall",
            )
            reports.append(recall_report)
            for value, target in (
                (recall_report.get("best_distance"), best_distances),
                (recall_report.get("best_input_distance"), best_input_distances),
                (recall_report.get("recalled_distance"), recalled_distances),
            ):
                if isinstance(value, (float, int)):
                    target.append(float(value))

        def _mean(values: Sequence[float]) -> float | None:
            return float(sum(values) / len(values)) if values else None

        mean_best_input_distance = _mean(best_input_distances)
        exact_input_recall_count = sum(
            1 for value in best_input_distances if float(value) <= 1e-5
        )
        all_bucket_scoped = all(
            report.get("candidate_scope") == "bucket_indexed_candidate_window"
            and bool(report.get("selection_report", {}).get("bounded_by_bucket_index"))
            for report in reports
        )
        any_live_tick = any(bool(report.get("runs_live_tick")) for report in reports)
        any_every_token = any(bool(report.get("runs_every_token")) for report in reports)
        any_raw_text = any(bool(report.get("raw_text_payload_loaded")) for report in reports)
        any_language_reasoning = any(
            bool(report.get("language_reasoning")) for report in reports
        )
        any_mutation = any(bool(report.get("mutates_runtime_state")) for report in reports)
        any_plasticity = any(bool(report.get("applies_plasticity")) for report in reports)
        aggregate = {
            "surface": "bounded_sleep_replay_associative_recall.v1",
            "status": "recalled" if reports else "empty",
            "sleep_mode": str(mode),
            "scope": f"{mode}_sleep_replay_window_associative_recall",
            "selection_criteria": [
                "reuse_selected_sleep_replay_indices_as_queries",
                "recall_only_inside_anchor_bucket_window",
                "tensor_payloads_only_no_raw_text",
            ],
            "candidate_bucket_ids": [int(value) for value in candidate_bucket_ids],
            "candidate_bucket_count": int(len(candidate_bucket_ids)),
            "selected_replay_index_count": int(len(replay_idx)),
            "query_budget": int(query_limit),
            "query_count": int(len(queries)),
            "query_indices": [int(index) for index, _key, _pattern in queries],
            "report_count": int(len(reports)),
            "candidate_scope": (
                "bucket_indexed_candidate_window"
                if all_bucket_scoped
                else str(reports[0].get("candidate_scope") or "unknown")
            ),
            "candidate_window_limit": int(
                max(
                    int(report.get("selection_report", {}).get("candidate_window_limit", 0) or 0)
                    for report in reports
                )
            ),
            "candidate_index_count": int(
                sum(int(report.get("candidate_index_count", 0) or 0) for report in reports)
            ),
            "selected_count_total": int(
                sum(int(report.get("selected_count", 0) or 0) for report in reports)
            ),
            "routing_key_count_total": int(
                sum(int(report.get("routing_key_count", 0) or 0) for report in reports)
            ),
            "input_pattern_count_total": int(
                sum(int(report.get("input_pattern_count", 0) or 0) for report in reports)
            ),
            "mean_best_distance": _mean(best_distances),
            "mean_best_input_distance": mean_best_input_distance,
            "mean_recalled_distance": _mean(recalled_distances),
            "exact_input_recall_count": int(exact_input_recall_count),
            "quality_metric": "mean_best_input_distance_over_selected_sleep_replay_queries",
            "quality_pass": bool(
                len(queries) > 0
                and len(best_input_distances) == len(queries)
                and mean_best_input_distance is not None
                and mean_best_input_distance <= 1e-5
            ),
            "query_prepare_missing_routing_key_count": int(missing_routing_key_count),
            "query_prepare_missing_input_pattern_count": int(missing_input_pattern_count),
            "score_device": "cpu",
            "archival_storage_device": "cpu",
            "device_placement": {
                "archival_storage_device": "cpu",
                "source_window_device": "cpu",
                "score_device": "cpu",
                "gpu_used": False,
                "gpu_resident_archival_metadata": False,
            },
            "runs_live_tick": bool(any_live_tick),
            "runs_every_token": bool(any_every_token),
            "raw_text_payload_loaded": bool(any_raw_text),
            "language_reasoning": bool(any_language_reasoning),
            "mutates_runtime_state": bool(any_mutation),
            "applies_plasticity": bool(any_plasticity),
            "global_candidate_scan": any(
                bool(report.get("global_candidate_scan")) for report in reports
            ),
            "global_score_scan": any(
                bool(report.get("global_score_scan")) for report in reports
            ),
            "fallback_reason": None,
            "reports": reports,
        }
        return aggregate

    def _refresh_sleep_replay_routing_index(
        self,
        updated_ids: list[int],
    ) -> dict[str, Any]:
        uniq = sorted(set(int(value) for value in updated_ids))
        if not uniq:
            return {
                "sleep_replay_routing_index_refresh_surface": (
                    "routing_index_existing_row_refresh.v1"
                ),
                "sleep_replay_routing_index_refresh_mode": "not_run",
                "sleep_replay_routing_index_updated_count": 0,
                "sleep_replay_routing_index_full_rebuild": False,
                "sleep_replay_routing_index_full_rebuild_executed": False,
                "sleep_replay_routing_index_deferred_rebuild_required": False,
            }
        id_arr = np.asarray(uniq, dtype=np.int64)
        vecs = self.model.competitive.prototypes[id_arr].detach()
        updater = getattr(self.model.routing_index, "update_existing", None)
        if not callable(updater):
            return {
                "sleep_replay_routing_index_refresh_surface": (
                    "routing_index_existing_row_refresh.v1"
                ),
                "sleep_replay_routing_index_refresh_mode": (
                    "deferred_missing_existing_row_api"
                ),
                "sleep_replay_routing_index_updated_count": int(len(uniq)),
                "sleep_replay_routing_index_direct_update_count": 0,
                "sleep_replay_routing_index_merged_direct_update_count": 0,
                "sleep_replay_routing_index_missing_id_count": 0,
                "sleep_replay_routing_index_row_lookup_miss_count": 0,
                "sleep_replay_routing_index_skipped_update_count": int(len(uniq)),
                "sleep_replay_routing_index_row_lookup_mode": "missing_update_api",
                "sleep_replay_routing_index_full_rebuild": False,
                "sleep_replay_routing_index_full_rebuild_executed": False,
                "sleep_replay_routing_index_deferred_rebuild_required": True,
                "sleep_replay_routing_index_recovery_reason": (
                    "missing_existing_row_update_api"
                ),
            }
        report = dict(updater(vecs, id_arr))
        recovery_required = bool(report.get("recovery_required"))
        return {
            "sleep_replay_routing_index_refresh_surface": str(
                report.get("surface", "routing_index_existing_row_refresh.v1")
            ),
            "sleep_replay_routing_index_refresh_mode": (
                "existing_row_in_place"
                if not recovery_required
                else "deferred_rebuild_required"
            ),
            "sleep_replay_routing_index_updated_count": int(len(uniq)),
            "sleep_replay_routing_index_direct_update_count": int(
                report.get("direct_update_count", 0) or 0
            ),
            "sleep_replay_routing_index_merged_direct_update_count": int(
                report.get(
                    "merged_direct_update_count",
                    report.get("direct_update_count", 0),
                )
                or 0
            ),
            "sleep_replay_routing_index_missing_id_count": int(
                report.get("missing_id_count", 0) or 0
            ),
            "sleep_replay_routing_index_row_lookup_miss_count": int(
                report.get("row_lookup_miss_count", 0) or 0
            ),
            "sleep_replay_routing_index_skipped_update_count": int(
                report.get("skipped_update_count", 0) or 0
            ),
            "sleep_replay_routing_index_row_lookup_mode": str(
                report.get("row_lookup_mode", "unknown")
            ),
            "sleep_replay_routing_index_full_rebuild": False,
            "sleep_replay_routing_index_full_rebuild_executed": False,
            "sleep_replay_routing_index_deferred_rebuild_required": recovery_required,
            "sleep_replay_routing_index_recovery_reason": report.get(
                "recovery_reason"
            ),
            "sleep_replay_routing_index_cache_dirty_after": bool(
                report.get("cache_dirty_after")
            ),
            "sleep_replay_routing_index_cache_generation": int(
                report.get("cache_generation", -1) or -1
            ),
        }

    def _sleep_replay(
        self,
        mode: str,
        *,
        deep_replay_repair_strength: float | None = None,
    ) -> int:
        """Replay slow-buffer assemblies with spaced priority and mode-specific depth."""
        repair_anchor_strength = 0.30
        deep_replay_repair_strength_value = (
            1.0
            if deep_replay_repair_strength is None
            else float(deep_replay_repair_strength)
        )

        if mode == "micro":
            steps = self.config.micro_sleep_replay_steps
            candidate_pool = self.config.micro_sleep_candidate_pool
            sampling_strategy = "maintenance"
        elif mode == "deep":
            steps = self.config.deep_sleep_replay_steps
            candidate_pool = self.config.deep_sleep_candidate_pool
            sampling_strategy = "consolidation"
            memory_blend = self.config.deep_sleep_memory_blend
            protein_synthesis_level = 1.35
        elif mode == "repair":
            steps = self.config.deep_sleep_replay_steps
            candidate_pool = self.config.deep_sleep_candidate_pool
            sampling_strategy = "repair"
        else:
            raise ValueError(f"Unknown sleep mode: {mode}")

        candidate_bucket_ids, anchor_bucket_source_window = (
            self._sleep_replay_candidate_bucket_ids(mode)
        )
        selection_report = self.model.memory_store.select_replay_window(
            n=steps,
            current_token=self.token_count,
            candidate_pool=candidate_pool,
            strategy=sampling_strategy,
            candidate_bucket_ids=candidate_bucket_ids,
            scope=f"{mode}_sleep_slow_path",
        )
        global_fallback_blocked_reason: str | None = None
        if (
            candidate_bucket_ids is not None
            and sampling_strategy != "random"
            and float(selection_report.get("selected_score_max", 0.0) or 0.0) <= 0.0
        ):
            selection_report = {
                **selection_report,
                "selected_indices": [],
                "selected_count": 0,
                "status": "empty",
            }
            global_fallback_blocked_reason = (
                f"no_anchor_bucket_scope_for_{mode}_replay"
                if mode in {"micro", "deep", "repair"} and not candidate_bucket_ids
                else "bucket_window_zero_positive_replay_pressure"
            )
        replay_idx = [
            int(index)
            for index in selection_report.get("selected_indices", [])
        ]
        associative_recall_report = (
            self._sleep_replay_associative_recall(
                replay_idx,
                mode=mode,
                candidate_bucket_ids=candidate_bucket_ids,
                max_candidates=max(steps, candidate_pool),
            )
            if mode == "deep"
            else self._empty_sleep_replay_associative_recall_report(
                mode=mode,
                candidate_bucket_ids=candidate_bucket_ids,
                fallback_reason="associative_recall_enabled_for_deep_sleep_only",
            )
        )
        self._last_sleep_replay_selection_report = {
            **selection_report,
            "sleep_mode": str(mode),
            "candidate_bucket_source": (
                "column_anchor_bucket_index"
                if candidate_bucket_ids is not None
                else "unanchored_slow_path"
            ),
            "bounded_bucket_source": (
                "column_anchor_bucket_index"
                if candidate_bucket_ids is not None
                else None
            ),
            "bounded_bucket_fallback": False,
            "unscoped_global_fallback_retired": bool(
                global_fallback_blocked_reason is not None
            ),
            "global_fallback_blocked_reason": global_fallback_blocked_reason,
            "anchor_bucket_source_window": dict(anchor_bucket_source_window),
            "candidate_bucket_source_window": dict(anchor_bucket_source_window),
            "anchor_bucket_source_window_surface": anchor_bucket_source_window.get(
                "surface"
            ),
            "anchor_bucket_source_total_count": int(
                anchor_bucket_source_window.get("anchor_bucket_source_total_count", 0)
                or 0
            ),
            "anchor_bucket_window_limit": int(
                anchor_bucket_source_window.get("anchor_bucket_window_limit", 0) or 0
            ),
            "anchor_bucket_window_count": int(
                anchor_bucket_source_window.get("anchor_bucket_window_count", 0) or 0
            ),
            "anchor_bucket_source_truncated_count": int(
                anchor_bucket_source_window.get(
                    "anchor_bucket_source_truncated_count",
                    0,
                )
                or 0
            ),
            "anchor_bucket_source_read_count": int(
                anchor_bucket_source_window.get("anchor_bucket_source_read_count", 0)
                or 0
            ),
            "anchor_bucket_source_window_policy": anchor_bucket_source_window.get(
                "window_policy"
            ),
            "anchor_source_full_scan": bool(
                anchor_bucket_source_window.get("anchor_source_full_scan", False)
            ),
            "sleep_replay_applied_count": 0,
            "sleep_replay_mutates_runtime_state": False,
            "sleep_replay_applies_plasticity": False,
            "sleep_replay_commit_strategy": "not_run",
            "sleep_replay_text_payload_loaded": False,
            "sleep_replay_language_reasoning": False,
            "sleep_replay_text_payload_policy": "sleep_replay_uses_tensor_payloads_only",
            "sleep_replay_local_trace_source": "stored_input_pattern_or_routing_key",
            "sleep_replay_unconditional_dense_input_assembly_retired": True,
            "sleep_replay_dense_input_assembly_fallback_count": 0,
            "sleep_replay_bounded_input_prepare_count": 0,
            "sleep_replay_stored_routing_key_count": 0,
            "sleep_replay_missing_routing_key_count": 0,
            "sleep_replay_missing_routing_key_deferred_count": 0,
            "sleep_replay_associative_recall": dict(associative_recall_report),
            "sleep_replay_associative_recall_surface": associative_recall_report.get(
                "surface"
            ),
            "sleep_replay_associative_recall_status": associative_recall_report.get(
                "status"
            ),
            "sleep_replay_associative_recall_query_count": int(
                associative_recall_report.get("query_count", 0) or 0
            ),
            "sleep_replay_associative_recall_quality_metric": associative_recall_report.get(
                "quality_metric"
            ),
            "sleep_replay_associative_recall_quality_pass": bool(
                associative_recall_report.get("quality_pass", False)
            ),
            "sleep_replay_associative_recall_mean_best_input_distance": (
                associative_recall_report.get("mean_best_input_distance")
            ),
            "sleep_replay_associative_recall_runs_live_tick": bool(
                associative_recall_report.get("runs_live_tick", False)
            ),
            "sleep_replay_associative_recall_runs_every_token": bool(
                associative_recall_report.get("runs_every_token", False)
            ),
            "sleep_replay_associative_recall_raw_text_payload_loaded": bool(
                associative_recall_report.get("raw_text_payload_loaded", False)
            ),
            "sleep_replay_associative_recall_language_reasoning": bool(
                associative_recall_report.get("language_reasoning", False)
            ),
            "sleep_replay_associative_recall_mutates_runtime_state": bool(
                associative_recall_report.get("mutates_runtime_state", False)
            ),
            "sleep_replay_associative_recall_applies_plasticity": bool(
                associative_recall_report.get("applies_plasticity", False)
            ),
            "sleep_replay_associative_recall_device_placement": dict(
                associative_recall_report.get("device_placement", {})
            ),
        }
        if not replay_idx:
            self.model.memory_store.last_replay_selection_report = dict(
                self._last_sleep_replay_selection_report
            )
            self.model.memory_store._invalidate_summary_cache()
            return 0

        applied = 0
        updated_ids = []
        processed_indices: list[int] = []
        commit_report: dict[str, Any] = {}

        if mode == "deep":
            (
                applied,
                updated_ids,
                processed_indices,
                commit_report,
            ) = self._sleep_replay_bounded_candidate_repair(
                replay_idx,
                repair_strength=deep_replay_repair_strength_value,
            )
        elif mode == "micro":
            processed_indices = [int(index) for index in replay_idx]
            applied = len(processed_indices)
            commit_report = {
                "sleep_replay_commit_strategy": "bounded_micro_maintenance_refresh",
                "sleep_replay_winner_source": "bucket_indexed_replay_window",
                "sleep_replay_bypasses_competitive_process": True,
                "sleep_replay_selected_trace_count": int(len(processed_indices)),
                "sleep_replay_quality_metric": "maintenance_score_tag_refresh_only",
                "sleep_replay_quality_scope": "anchored_replay_window_cpu_metadata",
            }
        elif mode == "repair":
            dense_input_assembly_fallback_count = 0
            bounded_input_prepare_count = 0
            stored_routing_key_count = 0
            missing_routing_key_count = 0
            missing_routing_key_deferred_count = 0
            commit_report = {
                "sleep_replay_commit_strategy": "bounded_repair_reanchor",
                "sleep_replay_winner_source": "stored_replay_bucket_with_anchor_scope",
                "sleep_replay_local_trace_prepare_policy": (
                    "stored_routing_key_required_missing_keys_deferred"
                ),
                "sleep_replay_unconditional_dense_input_assembly_retired": True,
            }

            for idx in replay_idx:
                replay_entry = self.model.memory_store.replay_entry(
                    idx,
                    current_token=self.token_count,
                    include_text_payload=False,
                )
                assembly = replay_entry["assembly"]
                input_pattern = replay_entry["input_pattern"]
                stored_routing_key = replay_entry["routing_key"]
                stored_bucket_id = replay_entry["bucket_id"]

                if not isinstance(assembly, torch.Tensor):
                    continue
                assembly = assembly.to(self.model.device)
                if assembly.dim() != 1:
                    continue
                if float(assembly.abs().sum().item()) <= 0.0:
                    continue

                if isinstance(input_pattern, torch.Tensor):
                    replay_input = input_pattern.to(self.model.device)
                else:
                    replay_input = None
                    self.model.competitive.last_input_pattern = None
                    self.model.competitive._cached_proto_sim = None
                    self.model.competitive._cached_raw_drive = None

                if isinstance(stored_routing_key, torch.Tensor):
                    stored_routing_key_count += 1
                    if replay_input is not None:
                        self.model.competitive.prepare_input_for_candidate_routing(
                            replay_input
                        )
                        bounded_input_prepare_count += 1
                    routing_key = F.normalize(stored_routing_key.to(self.model.device), dim=0)
                else:
                    missing_routing_key_count += 1
                    missing_routing_key_deferred_count += 1
                    continue

                context_prediction, context_gain = self._context_prediction_and_gain()

                if stored_bucket_id is not None:
                    winner = torch.tensor([int(stored_bucket_id)], device=self.model.device)
                else:
                    candidates = self._routing_candidates(routing_key)
                    winner, _, _ = self.model.competitive.compete(
                        routing_key,
                        candidates,
                        fallback_allowed=True,
                        context_gain=context_gain,
                    )

                self._repair_column_from_replay(
                    int(winner.item()),
                    routing_key,
                    strength=repair_anchor_strength,
                )
                updated_ids.append(int(winner.item()))
                processed_indices.append(int(idx))
                applied += 1
            commit_report.update(
                {
                    "sleep_replay_dense_input_assembly_fallback_count": int(
                        dense_input_assembly_fallback_count
                    ),
                    "sleep_replay_bounded_input_prepare_count": int(
                        bounded_input_prepare_count
                    ),
                    "sleep_replay_stored_routing_key_count": int(
                        stored_routing_key_count
                    ),
                    "sleep_replay_missing_routing_key_count": int(
                        missing_routing_key_count
                    ),
                    "sleep_replay_missing_routing_key_deferred_count": int(
                        missing_routing_key_deferred_count
                    ),
                }
            )

        if applied > 0:
            routing_index_report: dict[str, Any] = {}
            if mode == "deep":
                routing_index_report = self._refresh_sleep_replay_routing_index(
                    updated_ids
                )
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
                routing_index_report = self._refresh_sleep_replay_routing_index(
                    updated_ids
                )
                self.model.memory_store.mark_repair_replay(
                    processed_indices,
                    current_token=self.token_count,
                )
            if routing_index_report:
                commit_report.update(routing_index_report)

            self.sleep_events += 1
            if mode == "micro":
                self.micro_sleep_events += 1
                self.last_micro_sleep_token = self.token_count
            else:
                self.deep_sleep_events += 1
                self.last_deep_sleep_token = self.token_count
        sfa_report = {
            "sleep_replay_sfa_correction_scope": "not_run",
            "sleep_replay_sfa_full_memory_sample_retired": True,
            "sleep_replay_sfa_candidate_index_count": 0,
            "sleep_replay_sfa_sample_count": 0,
            "sleep_replay_sfa_applied": False,
            "sleep_replay_sfa_sample_report": dict(
                self.model.memory_store._empty_sfa_sample_report()
            ),
        }

        # SFA correction during deep sleep (§4.8)
        if mode == "deep":
            sfa_report.update(
                {
                    "sleep_replay_sfa_correction_scope": (
                        "selected_replay_window"
                        if applied > 0 and self.model.abstraction_layer is not None
                        else "not_run"
                    ),
                    "sleep_replay_sfa_candidate_index_count": int(
                        len(set(int(index) for index in processed_indices))
                    ),
                }
            )
        if mode == "deep" and applied > 0 and self.model.abstraction_layer is not None:
            sfa_samples, sfa_sample_report = (
                self.model.memory_store.sample_for_sfa_with_report(
                    n=min(100, max(10, applied)),
                    candidate_indices=processed_indices,
                    scope="deep_sleep_sfa_correction",
                )
            )
            sfa_report["sleep_replay_sfa_sample_report"] = dict(sfa_sample_report)
            sfa_report["sleep_replay_sfa_sample_count"] = int(len(sfa_samples))
            if len(sfa_samples) >= 2:
                self.model.abstraction_layer.sfa_correction_step(
                    sfa_samples,
                    lr=0.01,
                )
                sfa_report["sleep_replay_sfa_applied"] = True

        self._last_sleep_replay_selection_report = {
            **self._last_sleep_replay_selection_report,
            **commit_report,
            **sfa_report,
            "sleep_replay_applied_count": int(applied),
            "sleep_replay_mutates_runtime_state": bool(applied > 0),
            "sleep_replay_applies_plasticity": bool(
                mode == "deep" and applied > 0
            ),
        }
        self.model.memory_store.last_replay_selection_report = dict(
            self._last_sleep_replay_selection_report
        )
        self.model.memory_store._invalidate_summary_cache()

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

        use_dense_transition = self.model.device.type == "cuda"
        used_candidate_transition = False
        if use_dense_transition:
            transition_runtime_fallback = None
            if self.config.predictive_dense_transition_mode == "inplace_triton":
                transition_runtime_fallback = (
                    self._column_transition_runtime.fallback_reason
                )
            pred_error_mod = self.model.predictive.apply_dense_transition(
                winners,
                routing_key,
                self._prev_routing_key,
                learning_rate=0.005,
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
            self._flush_routing_index_buffer()
            replay_updates = self._sleep_replay("repair" if deep_due_emergency else "deep")
            if replay_updates > 0:
                sleep_type = "deep"
                deep_sleep_emergency = bool(deep_due_emergency)
                if deep_due_emergency:
                    self.pending_emergency_deep_sleep = False
        elif allow_sleep_maintenance and micro_due:
            self._flush_routing_index_buffer()
            replay_updates = self._sleep_replay("micro")
            if replay_updates > 0:
                sleep_type = "micro"
        elif self.token_count % self._routing_index_flush_interval == 0:
            self._flush_routing_index_buffer()

        sleep_triggered = sleep_type != "none"
        if sleep_triggered:
            drift_bucket = self.last_winner if self.config.use_winner_local_drift else None
            drift = self.model.memory_store.compute_drift(drift_bucket)
            self._refresh_latest_drift(drift)

        metrics["drift"] = drift
        metrics["sleep_triggered"] = int(sleep_triggered)
        metrics["sleep_type"] = sleep_type
        metrics["sleep_replay_updates"] = int(replay_updates)
        metrics["sleep_replay_selection"] = dict(
            self._last_sleep_replay_selection_report
        )
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
            and "filter_route_vote" in str(wake_plan.mode)
        ):
            wake_plan = self._route_vote_owner_wake_plan(candidates)
        self._record_column_metabolism(wake_plan)
        self._record_column_structural_review(
            wake_plan,
            token_count=int(self.token_count) + 1,
            mode="awake_mask_tick",
        )
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
            self._buffer_routing_index_update(
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
        strong_capture_candidate = bool(capture_tag >= strong_capture_threshold)
        slow_memory_archive_reason = "memory_not_warm"
        slow_memory_archive_due = False
        if self.memory_warm_started:
            if next_token == 1:
                slow_memory_archive_due = True
                slow_memory_archive_reason = "first_token"
            elif strong_capture_candidate:
                if self._slow_memory_strong_capture_allowed(next_token):
                    slow_memory_archive_due = True
                    slow_memory_archive_reason = "strong_capture"
                else:
                    self._slow_memory_strong_capture_refractory_skip_count += 1
                    slow_memory_archive_reason = "strong_capture_refractory_skip"
            elif next_token % slow_memory_archive_interval == 0:
                slow_memory_archive_reason = "cadence_deferred"
                self._cognitive_boundary_controller.record_slow_memory_cadence_deferred(
                    token=next_token,
                )
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
            if slow_memory_archive_reason == "strong_capture":
                self._record_slow_memory_strong_capture_archive(next_token)
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
            awake_bucket_ids = candidates
            if (
                not isinstance(awake_bucket_ids, torch.Tensor)
                or int(awake_bucket_ids.numel()) <= 0
            ):
                awake_bucket_ids = winners
            tagged = self.model.memory_store.ripple_tag_awake(
                current_token=next_token,
                window_tokens=max(1, self.config.functional_minute // 2),
                da_level=da_level,
                awake_bucket_ids=awake_bucket_ids,
                max_candidate_entries=self._recent_replay_setup_limit(),
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
            self.model.memory_store.live_summary_stats()
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
        metrics["candidate_memory_pressure_filter_start_tokens"] = int(
            self.config.candidate_memory_pressure_filter_start_tokens
        )
        metrics["candidate_memory_pressure_filter_due"] = int(
            self._candidate_memory_pressure_filter_due(apply_sleep_filter=True)
        )
        metrics["candidate_memory_pressure_filtered_count"] = int(
            self._column_wake_plan.filtered_memory_pressure_count
        )
        metrics["candidate_memory_pressure_filter_mode"] = str(
            self._column_wake_plan.mode
        )
        metrics["candidate_usefulness_filter_start_tokens"] = int(
            self.config.candidate_usefulness_filter_start_tokens
        )
        metrics["candidate_usefulness_filter_due"] = int(
            self._candidate_usefulness_filter_due(apply_sleep_filter=True)
        )
        metrics["candidate_low_usefulness_filtered_count"] = int(
            self._column_wake_plan.filtered_low_usefulness_count
        )
        metrics["candidate_usefulness_filter_mode"] = str(
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
        metrics["slow_memory_cadence_deferred_count"] = int(
            self._cognitive_boundary_controller.slow_memory_cadence_deferred_count
        )
        last_deferred_cadence_token = (
            self._cognitive_boundary_controller.last_slow_memory_cadence_token
        )
        metrics["slow_memory_last_deferred_cadence_token"] = (
            None
            if last_deferred_cadence_token is None
            else int(last_deferred_cadence_token)
        )
        metrics["slow_memory_strong_capture_min_interval_tokens"] = int(
            self._slow_memory_strong_capture_min_interval_tokens()
        )
        metrics["slow_memory_strong_capture_archive_count"] = int(
            self._slow_memory_strong_capture_archive_count
        )
        metrics["slow_memory_strong_capture_refractory_skip_count"] = int(
            self._slow_memory_strong_capture_refractory_skip_count
        )
        metrics["slow_memory_last_strong_capture_token"] = int(
            self._slow_memory_last_strong_capture_token
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

    def _offline_competition(
        self,
        pattern_vec: torch.Tensor,
        *,
        return_routing_key: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        if return_routing_key:
            return winners, assembly, routing_key
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

    def run_sleep_maintenance(
        self,
        mode: str = "deep",
        cycles: int = 1,
        *,
        deep_replay_repair_strength: float | None = None,
    ) -> int:
        total_updates = 0
        for _ in range(max(0, int(cycles))):
            total_updates += self._sleep_replay(
                mode,
                deep_replay_repair_strength=deep_replay_repair_strength,
            )
        return total_updates

    def _recent_replay_setup_limit(self) -> int:
        candidate_pool = int(getattr(self.config, "deep_sleep_candidate_pool", 32))
        return max(32, candidate_pool * 8)

    def tag_recent_memories(self, window_tokens: int, strength: float) -> int:
        return self.model.memory_store.tag_recent_entries(
            current_token=self.token_count,
            window_tokens=window_tokens,
            strength=strength,
            max_recent_entries=self._recent_replay_setup_limit(),
        )

    def capture_recent_memory_anchors(self, window_tokens: int, strength: float) -> int:
        if window_tokens <= 0:
            self.model.memory_store.last_anchor_capture_report = {
                **self.model.memory_store._empty_anchor_capture_report(),
                "status": "empty",
                "current_token": int(self.token_count),
                "window_tokens": int(window_tokens),
                "strength": float(strength),
                "fallback_reason": "empty_recent_window",
            }
            return 0

        window_report = self.model.memory_store.collect_recent_entry_indices(
            current_token=int(self.token_count),
            window_tokens=int(window_tokens),
            max_entries=self._recent_replay_setup_limit(),
            require_bucket=True,
            scope="recent_anchor_capture_slow_path",
        )
        captured = 0
        candidate_buckets: set[int] = set()
        for idx in window_report.get("candidate_indices", []):
            idx = int(idx)
            bucket_id = self.model.memory_store.slow_bucket_ids[idx]
            if bucket_id is None:
                continue
            bucket = int(bucket_id)
            candidate_buckets.add(bucket)
            self.column_anchors.pop(bucket, None)
            self.column_anchors[bucket] = {
                "prototype": self.model.competitive.prototypes[bucket].detach().clone(),
                "input_weights": self.model.competitive.input_weights[bucket].detach().clone(),
                "strength": float(max(0.0, strength)),
                "captured_at_token": int(self.token_count),
                "captured_source_index": int(idx),
                "capture_sequence": int(captured),
            }
            captured += 1
        anchor_count = len(self.column_anchors) if captured > 0 else 0
        self.model.memory_store.last_anchor_capture_report = {
            **dict(window_report),
            "surface": "bounded_recent_anchor_capture.v1",
            "status": "captured" if captured else "empty",
            "scope": "recent_anchor_capture_slow_path",
            "captured_entry_count": int(captured),
            "captured_anchor_count": int(anchor_count),
            "candidate_bucket_ids": sorted(candidate_buckets),
            "strength": float(strength),
            "mutates_runtime_state": bool(captured),
            "applies_plasticity": False,
            "fallback_reason": None
            if captured
            else window_report.get("fallback_reason", "no_recent_anchor_entries"),
        }
        return anchor_count

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
