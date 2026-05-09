from __future__ import annotations

import base64
from collections import Counter, deque
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import random
import re
from threading import Event, Lock, RLock, Thread
import time
from typing import Any, Iterator, Mapping, Sequence, cast
from urllib.parse import urlparse
from uuid import uuid4

import torch

from hecsn.config.presets import get_autonomy_acquisition_preset
from hecsn.config.model_config import HECSNConfig
from hecsn.config.runtime_env import load_runtime_env
from hecsn.data.corpus_loader import BackgroundPrefetchIterator, StreamingCorpusLoader, huggingface_token_from_env, load_hf_first_rows
from hecsn.data.pattern_loader import labeled_pattern_stream
from hecsn.gap_planner import plan_query_gaps
from hecsn.interaction import EvidenceResponder
from hecsn.semantics import ConceptStore, GeometricCuriosityController
from hecsn.semantics.grounding_text import match_terms, salient_query_terms
from hecsn.training.autonomy_acquisition_runner import run_live_acquisition
from hecsn.training.checkpointing import load_trainer_checkpoint
from hecsn.training.trainer import HECSNModel, HECSNTrainer
from hecsn.training.query_runner import build_query_result, feed_text

import logging as _logging

_cortex_logger = _logging.getLogger(__name__ + ".cortex")


PUBLIC_ACQUISITION_PRESET = "autonomy_acquisition_hf_allocation"
PUBLIC_ACQUISITION_PRESETS: tuple[str, ...] = (PUBLIC_ACQUISITION_PRESET,)
PUBLIC_ACQUISITION_POLICIES: tuple[str, ...] = ("active", "round_robin")
DEFAULT_BRAIN_TICK_TOKENS = 512
DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS = 0.01
DEFAULT_AUTONOMY_TRIGGER_INTERVAL_TOKENS = 4096
DEFAULT_RECENT_QUERY_GAP_HISTORY = 8
DEFAULT_BRAIN_STOP_TIMEOUT_SECONDS = 15.0
DEFAULT_INGESTION_QUEUE_MULTIPLIER = 2
DEFAULT_REMOTE_PREWARM_GRACE_SECONDS = 0.25
DEFAULT_REMOTE_PREWARM_POLL_SECONDS = 0.05
DEFAULT_REMOTE_PROMOTION_BOOTSTRAP_GRACE_SECONDS = 0.3
DEFAULT_REMOTE_ACTIVE_FETCH_WAIT_SECONDS = 0.25
DEFAULT_REMOTE_STREAM_PREFETCH_ITEMS = 4
DEFAULT_REMOTE_BOOTSTRAP_ROWS = 2
DEFAULT_REMOTE_BOOTSTRAP_BUDGET_SECONDS = 3.0
DEFAULT_DELAYED_CONSEQUENCE_RECORDS = 24
DEFAULT_DELAYED_CONSEQUENCE_MATCH_THRESHOLD = 0.34
DEFAULT_DELAYED_CONSEQUENCE_DELTA_THRESHOLD = 0.08
DEFAULT_DELAYED_CONTRADICTION_DECAY_THRESHOLD = 0.18
DEFAULT_DELAYED_CONTRADICTION_UNSUPPORTED_THRESHOLD = 0.34
DEFAULT_DELAYED_CONSEQUENCE_COOLING_START_TOKENS = 512
DEFAULT_DELAYED_CONSEQUENCE_COOLING_WINDOW_TOKENS = 1024
DEFAULT_DELAYED_CONSEQUENCE_RETIREMENT_TOKENS = 4096
DEFAULT_DELAYED_CONSEQUENCE_RETIREMENT_BALANCE_THRESHOLD = 0.05
DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_MATCH_THRESHOLD = 0.52
DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_PROVENANCE_THRESHOLD = 0.60
DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT = 6
DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_TERM_LIMIT = 12
DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_SUPPORT_SCALE = 0.18
DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_SUPPORT_MAX = 1.35
DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_STATE_THRESHOLD = 0.12
DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SUPPORT_MAX = 1.25
DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_RECENT_ALPHA = 0.50
DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT = 4.0
DEFAULT_DELAYED_CONSEQUENCE_SPLIT_MAX_BRANCH_OVERLAP = 0.70
DEFAULT_DELAYED_CONSEQUENCE_SPLIT_MIN_BRANCH_OCCURRENCES = 1
DEFAULT_DELAYED_CONSEQUENCE_REMERGE_MIN_CROSS_OCCURRENCES = 1
DEFAULT_FORGIVENESS_RECOVERY_RATIO = 0.80
DEFAULT_UTILITY_PENALTY_WEIGHT = 0.65
DEFAULT_RUNTIME_TRACE_EXPORT_LIMIT = 20
MAX_RUNTIME_TRACE_EXPORT_LIMIT = 50
DEFAULT_REPLAY_DATASET_EXPORT_LIMIT = DEFAULT_RUNTIME_TRACE_EXPORT_LIMIT
MAX_REPLAY_DATASET_EXPORT_LIMIT = MAX_RUNTIME_TRACE_EXPORT_LIMIT
DEFAULT_REPLAY_SAMPLE_HISTORY = 256
MAX_REPLAY_SAMPLE_LIMIT = 20
DEFAULT_CORTEX_INIT_TIMEOUT_SECONDS = 2.0
DEFAULT_CORTEX_ACTION_INIT_TIMEOUT_SECONDS = 0.25
RUNTIME_TRACE_EXPORT_SCHEMA_VERSION = 1
REPLAY_DATASET_SCHEMA_VERSION = 1
REPLAY_DATASET_TRAINING_ROLE = "replay_dataset_preview_only_not_training_no_mutation"
DEFAULT_RUNTIME_FEEDBACK_HISTORY = 8
DEFAULT_RUNTIME_FEEDBACK_EVIDENCE_LIMIT = 8
DEFAULT_RUNTIME_FEEDBACK_TAG_LIMIT = 12
DEFAULT_RUNTIME_FEEDBACK_MAX_TEXT_CHARS = 2000
_RUNTIME_TRACE_EXPORT_MAX_STRING_CHARS = 2000
_RUNTIME_TRACE_EXPORT_MAX_LIST_ITEMS = 16
_RUNTIME_TRACE_EXPORT_MAX_MAPPING_ITEMS = 48
_RUNTIME_TRACE_EXPORT_ALLOWED_TOKEN_KEYS = {
    "token_count",
    "token_count_mutated",
    "tokens_processed",
    "top_k_candidates",
    "top_k_memories",
}
_RUNTIME_TRACE_EXPORT_UNSAFE_KEY_MARKERS = (
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "dotenv",
    "environment",
    "password",
    "secret",
)
_RUNTIME_TRACE_EXPORT_UNSAFE_KEYS = {
    "checkpoint_path",
    "env",
    "env_root",
    "path",
    "raw_environment",
    "root_path",
    "runtime_env",
    "trace_path",
    "workspace_root",
}

# Re-export autonomy constants for backwards compatibility
from hecsn.service.terminus_autonomy import (  # noqa: E402
    DEFAULT_AUTONOMY_REMOTE_PROVIDERS,
    DEFAULT_AUTONOMY_REMOTE_CATALOG_LIMIT,
    DEFAULT_AUTONOMY_REMOTE_PROBE_POOL_LIMIT,
    DEFAULT_AUTONOMY_REMOTE_QUERIES_PER_PROVIDER,
    DEFAULT_AUTONOMY_REMOTE_PROVIDER_RESULT_LIMIT,
    AUTO_REMOTE_QUERY_BUDGET_MAX,
    AUTO_REMOTE_PROVIDER_PRIORITY_WEIGHT,
    AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT,
    AUTO_REMOTE_PROVIDER_QUERY_FAMILY_LIMIT,
    AUTO_FOCUS_SHORTLIST_MAX_SIZE,
    AUTO_FOCUS_SHORTLIST_GAP_WEIGHT,
    AUTO_FOCUS_SHORTLIST_AFFINITY_WEIGHT,
)

from hecsn.service.replay_dataset_bundle import ReplayDatasetBundleMixin
from hecsn.service.interaction_pipeline import InteractionPipeline
from hecsn.service.runtime_evidence import RuntimeEvidenceMixin
from hecsn.service.runtime_feedback import RuntimeFeedbackMixin
from hecsn.service.action_assist import ActionAssistMixin
from hecsn.service.action_runtime import ActionRuntimeMixin
from hecsn.service.brain_runtime import BrainRuntimeMixin
from hecsn.service.delayed_consequence import DelayedConsequenceMixin
from hecsn.service.persistence import ServicePersistenceMixin
from hecsn.service.cortex_runtime import CortexRuntimeMixin
from hecsn.service.reporting import ServiceReportingMixin
from hecsn.service.replay_runtime import ReplayRuntimeMixin
from hecsn.service.interaction_runtime import InteractionRuntimeMixin
from hecsn.service.living_status import LivingStatusMixin
from hecsn.service.runtime_config import RuntimeConfigMixin
from hecsn.service.runtime_control import RuntimeControlMixin
from hecsn.service.runtime_state import RuntimeState
from hecsn.service.runtime_prewarm import RuntimePrewarmMixin
from hecsn.service.runtime_sources import RuntimeSourcesMixin, _BrainSourceRuntime, _SensorySourceRuntime
from hecsn.service.sensory_runtime import SensoryRuntimeMixin
from hecsn.service.status_runtime import StatusRuntimeMixin
from hecsn.service.status_read_model import StatusReadModel
from hecsn.service.sensory_preview import SensoryPreviewMixin
from hecsn.service.source_focus import SourceFocusMixin
from hecsn.service.living_loop import (
    ActionExecutionRecord,
    ConsolidationRecord,
    OperationalSelfModel,
    ProvenanceState,
    RuntimeEpisodeTrace,
    REPLAY_SAMPLE_SAFETY_BOUNDARIES,
    build_policy_actuator_status,
    build_replay_plan,
    build_runtime_benchmark_telemetry,
    replay_candidate_safety_flags,
)
from hecsn.service.terminus_presets import TERMINUS_QUICK_START_PRESETS
from hecsn.service.terminus_sensory import SensoryEpisode, bootstrap_sensory_episode_from_row, build_sensory_stream, sensory_bootstrap_columns


from hecsn.service.terminus_autonomy import _canonical_provider_term  # noqa: E402


class _TimedCallFailure:
    def __init__(self, error: BaseException) -> None:
        self.error = error


class _LazyThoughtLoop:
    """Compatibility proxy that initializes the real ThoughtLoop on active use."""

    def __init__(self, manager: "HECSNServiceManager") -> None:
        self._manager = manager

    def _get(self) -> Any:
        loop = self._manager._ensure_cortex_initialized()
        if loop is None:
            raise RuntimeError("cortex_unavailable")
        return loop

    @property
    def is_running(self) -> bool:
        loop = self._manager._thought_loop_actual
        return bool(loop is not None and loop.is_running)

    def snapshot(self) -> dict[str, Any]:
        loop = self._manager._thought_loop_actual
        if loop is None:
            return self._manager._cortex_unavailable_snapshot()
        return loop.snapshot()

    def start(self) -> None:
        loop = self._get()
        loop.start()

    def stop(self, timeout: float = 5.0) -> None:
        loop = self._manager._thought_loop_actual
        if loop is not None:
            loop.stop(timeout=timeout)

    def request_stop(self) -> None:
        loop = self._manager._thought_loop_actual
        if loop is not None:
            loop.request_stop()

    def submit_query(self, query: str) -> None:
        self._get().submit_query(query)

    def request_sleep(self, **kwargs: Any) -> dict[str, Any]:
        return self._get().request_sleep(**kwargs)

    def inject_action_result(self, **kwargs: Any) -> None:
        self._get().inject_action_result(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get(), name)


from hecsn.service.terminus_autonomy import TerminusAutonomyMixin


class HECSNServiceManager(
    ReplayDatasetBundleMixin,
    RuntimeEvidenceMixin,
    RuntimeFeedbackMixin,
    ActionAssistMixin,
    ActionRuntimeMixin,
    BrainRuntimeMixin,
    DelayedConsequenceMixin,
    ServicePersistenceMixin,
    CortexRuntimeMixin,
    ServiceReportingMixin,
    ReplayRuntimeMixin,
    InteractionRuntimeMixin,
    LivingStatusMixin,
    RuntimeConfigMixin,
    RuntimeControlMixin,
    RuntimePrewarmMixin,
    RuntimeSourcesMixin,
    SensoryRuntimeMixin,
    SourceFocusMixin,
    StatusRuntimeMixin,
    SensoryPreviewMixin,
    TerminusAutonomyMixin,
):
    """Main service orchestrator for HECSN/Terminus.

    Manages the SNN model, cortex integration, brain loop,
    multimodal training, checkpointing, and the REST API state.
    Autonomy / targeted acquisition logic is in TerminusAutonomyMixin.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        trace_history_limit: int = 200,
        trace_dir: str | Path | None = None,
        env_root: str | Path | None = None,
    ) -> None:
        self._lock = RLock()
        self._runtime_state: RuntimeState = RuntimeState(lock=self._lock)
        self._brain_execution_lock = Lock()
        self._checkpoint_path = Path(checkpoint_path)
        self._checkpoint_dir = self._checkpoint_path.parent if self._checkpoint_path.parent != Path("") else Path("checkpoints")
        self._env_root = None if env_root is None else Path(env_root)
        self._runtime_env = load_runtime_env(anchor_paths=(self._env_root, self._checkpoint_dir))
        self._action_root = (self._env_root or self._checkpoint_dir).resolve()
        self._trace_dir = Path(trace_dir) if trace_dir is not None else (Path("reports") / "service" / "traces")
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        self._trainer, self._metadata = load_trainer_checkpoint(self._checkpoint_path)
        self._encoder = self._trainer.encoder
        self._responder = EvidenceResponder()
        self._trace_history: deque[dict[str, Any]] = deque(maxlen=max(1, int(trace_history_limit)))
        service_state = dict(self._metadata.get("service_state", {}))
        terminus_state = dict(service_state.get("terminus_runtime", service_state.get("brain_runtime")) or {})
        concept_state = service_state.get("concept_store")
        self._concept_store = ConceptStore.from_state_dict(concept_state)
        self._geometric_curiosity = GeometricCuriosityController.from_state_dict(
            self._trainer.model.abstraction_layer,
            cast(dict[str, Any] | None, terminus_state.get("geometric_curiosity")),
        )
        self._brain_config = self._normalize_brain_config(
            terminus_state
        )
        self._brain_source_runtimes: list[_BrainSourceRuntime] = []
        self._sensory_source_runtimes: list[_SensorySourceRuntime] = []
        self._brain_source_index = 0
        self._sensory_source_index = 0
        self._brain_tick_count = 0
        self._brain_background_tokens = 0
        self._brain_autonomy_tokens = 0
        self._brain_source_utility = self._normalize_background_source_utility_state(
            terminus_state.get("background_source_utility")
        )
        self._brain_last_error: str | None = None
        self._brain_recent_query_gaps: deque[dict[str, Any]] = deque(
            (
                item
                for item in (
                    self._normalize_recent_query_gap(raw_item)
                    for raw_item in list(terminus_state.get("recent_query_gaps") or [])
                )
                if item is not None
            ),
            maxlen=DEFAULT_RECENT_QUERY_GAP_HISTORY,
        )
        self._last_cortex_query_hint_text: str | None = None
        self._last_cortex_query_hint_at = 0.0
        self._action_history: deque[dict[str, Any]] = deque(
            (
                item
                for item in (
                    self._normalize_action_record(raw_item)
                    for raw_item in list(terminus_state.get("action_history") or [])
                )
                if item is not None
            ),
            maxlen=24,
        )
        self._runtime_episode_traces: deque[dict[str, Any]] = deque(
            (
                item
                for item in (
                    self._normalize_runtime_episode_trace(raw_item)
                    for raw_item in list(terminus_state.get("runtime_episode_traces") or [])
                )
                if item is not None
            ),
            maxlen=64,
        )
        self._replay_sample_history: deque[dict[str, Any]] = deque(
            (
                item
                for item in (
                    self._normalize_replay_sample_record(raw_item)
                    for raw_item in list(terminus_state.get("replay_sample_history") or [])
                )
                if item is not None
            ),
            maxlen=DEFAULT_REPLAY_SAMPLE_HISTORY,
        )
        self._delayed_consequence_records: deque[dict[str, Any]] = deque(
            (
                item
                for item in (
                    self._normalize_delayed_consequence_record(raw_item)
                    for raw_item in list(terminus_state.get("delayed_consequence_records") or [])
                )
                if item is not None
            ),
            maxlen=DEFAULT_DELAYED_CONSEQUENCE_RECORDS,
        )
        self._delayed_consequence_cooled_total = max(0, int(terminus_state.get("delayed_consequence_cooled_total", 0) or 0))
        self._delayed_consequence_retired_total = max(0, int(terminus_state.get("delayed_consequence_retired_total", 0) or 0))
        self._delayed_consequence_compacted_total = max(0, int(terminus_state.get("delayed_consequence_compacted_total", 0) or 0))
        self._delayed_consequence_split_total = max(0, int(terminus_state.get("delayed_consequence_split_total", 0) or 0))
        self._delayed_consequence_remerged_total = max(0, int(terminus_state.get("delayed_consequence_remerged_total", 0) or 0))
        self._brain_skip_next_autonomy_for_grounded_query = False
        self._brain_last_acquisition_summary: dict[str, Any] | None = None
        self._brain_last_acquisition_token_count = int(self._trainer.token_count)
        self._brain_running_since: str | None = None
        self._brain_last_tick_completed_at: str | None = None
        self._brain_last_tick_duration_ms: float | None = None
        self._brain_last_tick_token_delta = 0
        self._brain_last_work_at: str | None = None
        self._last_real_sensory_episode_time = 0.0
        self._last_real_sensory_episode_token_count = int(self._trainer.token_count)
        self._real_sensory_last_error: str | None = None
        self._last_sensory_focus_terms: tuple[str, ...] = ()
        self._sensory_preview_history: deque[dict[str, Any]] = deque(maxlen=8)
        self._brain_thread: Thread | None = None
        self._brain_stop_event: Event | None = None
        self._brain_running = False
        self._brain_stop_requested_at: str | None = None
        self._brain_stop_requested_reason: str | None = None
        self._brain_stop_requested_perf: float | None = None
        self._brain_stop_timed_out = False
        self._brain_last_stop_duration_ms: float | None = None
        self._real_sensory_episodes_completed = 0
        self._real_visual_accepted = 0
        self._real_audio_accepted = 0
        self._ingestion_prewarm_thread: Thread | None = None
        self._ingestion_prewarm_stop_event: Event | None = None
        self._ingestion_prewarm_running = False
        self._ingestion_configured_at: str | None = None
        self._ingestion_configured_perf: float | None = None
        self._ingestion_prewarm_started_at: str | None = None
        self._ingestion_prewarm_started_perf: float | None = None
        self._ingestion_prewarm_completed_at: str | None = None
        self._ingestion_prewarm_last_duration_ms: float | None = None
        self._ingestion_prewarm_last_error: str | None = None
        self._ingestion_prewarm_run_count = 0
        self._ingestion_prewarm_last_trigger: str | None = None
        self._ingestion_prewarm_budget_exhausted = False
        self._ingestion_warm_ready_at: str | None = None
        self._ingestion_startup_warm_latency_ms: float | None = None
        self._remote_warm_promotion_thread: Thread | None = None
        self._remote_warm_promotion_stop_event: Event | None = None
        self._remote_warm_promotion_running = False
        self._remote_warm_promotion_last_trigger: str | None = None
        self._sensory_configured_at: str | None = None
        self._sensory_configured_perf: float | None = None
        self._sensory_prewarm_budget_exhausted = False
        self._sensory_warm_ready_at: str | None = None
        self._sensory_startup_warm_latency_ms: float | None = None
        self._active_execution_requests = 0
        self._active_execution_idle_event = Event()
        self._active_execution_idle_event.set()
        self._brain_stream_epoch = 0
        self._sensory_stream_epoch = 0
        self._rebuild_brain_sources_locked()
        self._runtime_state.restore_event_history(
            last_event=terminus_state.get("last_event"),
            recent_events=terminus_state.get("recent_events"),
        )
        self._runtime_state.restore_clean()
        self._load_persisted_traces_locked()

        # --- Cortex / ThoughtLoop (requires NVIDIA_API_KEY) ---
        # Cortex/NIM clients are intentionally not constructed during service
        # startup. Real NVIDIA environments can block for minutes during NIM
        # health checks; initialization is triggered only by active cortex use.
        self._thought_loop_actual: Any = None  # type: ThoughtLoop | None
        self._lazy_thought_loop = _LazyThoughtLoop(self)
        self._cortex_available = False
        self._cortex_init_lock = Lock()
        self._cortex_init_event = Event()
        self._cortex_init_thread: Thread | None = None
        self._cortex_init_started = False
        self._cortex_init_finished = False
        self._cortex_init_timed_out = False
        self._cortex_init_error: str | None = None
        self._cortex_factory_refs: tuple[Any, Any, Any, Any] | None = None
        try:
            from hecsn.cortex.thought_loop import ThoughtLoop
            from hecsn.cortex.multi_cortex import create_cortex_from_env, create_embedder_from_env
            from hecsn.cortex.episodic_memory import EpisodicMemory

            self._cortex_factory_refs = (
                ThoughtLoop,
                create_cortex_from_env,
                create_embedder_from_env,
                EpisodicMemory,
            )
            _cortex_logger.info("Cortex module available for lazy initialization")
        except Exception as exc:
            self._cortex_init_finished = True
            self._cortex_init_error = str(exc)
            self._cortex_init_event.set()
            _cortex_logger.info("Cortex module unavailable: %s", exc)

        # --- Status Read Model (ADR 0003 deep module extraction) ---
        self._status_read_model = self._build_status_read_model()
        # --- Interaction Pipeline (ADR 0003 query/feed-turn extraction) ---
        self._interaction_pipeline = self._build_interaction_pipeline()

    @property
    def _thought_loop(self) -> Any:
        if self._thought_loop_actual is not None:
            return self._thought_loop_actual
        if self._cortex_factories_are_mocked():
            return self._lazy_thought_loop
        return None

    @_thought_loop.setter
    def _thought_loop(self, value: Any) -> None:
        self._thought_loop_actual = value
        self._cortex_available = value is not None
        if value is not None:
            self._cortex_init_started = True
            self._cortex_init_finished = True
            self._cortex_init_error = None
            self._cortex_init_event.set()

    def _build_status_read_model(self) -> StatusReadModel:
        return StatusReadModel(
            lock=self._lock,
            runtime_state=self._runtime_state,
            trainer=self._trainer,
            trace_history=self._trace_history,
            metadata=self._metadata,
            checkpoint_path_str=str(self._checkpoint_path),
            trace_dir_str=str(self._trace_dir),
            concept_store_snapshot_fn=lambda: self._concept_store.snapshot(),
            brain_runtime_snapshot_fn=self._brain_runtime_snapshot_locked,
            multimodal_runtime_summary_fn=self._multimodal_runtime_summary_locked,
            sensory_preview_history=self._sensory_preview_history,
            architecture_snapshot_fn=self._architecture_summary_impl,
            cortex_active_fn=self._cortex_active,
            animation_snapshot_fn=self._animation_snapshot_locked,
            living_loop_status_fn=self._living_loop_status_impl,
            policy_actuator_status_fn=self._policy_actuator_status_impl,
            cortex_signal_state_fn=self._cortex_signal_state_impl,
        )

    def _build_interaction_pipeline(self) -> InteractionPipeline:
        return InteractionPipeline(
            lock=self._lock,
            trainer=self._trainer,
            encoder=self._encoder,
            build_query_result_fn=lambda **kwargs: self._build_query_locked(**kwargs),
            observe_concepts_fn=lambda **kwargs: self._observe_concepts_locked(**kwargs),
            plan_gaps_fn=lambda **kwargs: self._plan_gaps_locked(**kwargs),
            apply_delayed_query_consequence_fn=lambda **kwargs: self._apply_delayed_query_consequence_locked(**kwargs),
            record_recent_query_gap_fn=lambda **kwargs: self._record_recent_query_gap_locked(**kwargs),
            observe_runtime_concepts_fn=lambda **kwargs: self._observe_runtime_concepts_locked(**kwargs),
            runtime_state_mark_mutated_fn=lambda: self._runtime_state.mark_mutated(),
            runtime_state_mutation_summary_fn=lambda: self._runtime_state.mutation_summary(),
            runtime_episode_payload_fn=lambda **kwargs: self._runtime_episode_payload_locked(**kwargs),
            persist_trace_fn=lambda trace: self._persist_trace_locked(trace),
            append_runtime_episode_trace_fn=lambda episode: self._append_runtime_episode_trace_locked(episode),
            runtime_episode_trace_fn=lambda episode_id: self._runtime_episode_trace_locked(episode_id),
            replace_runtime_episode_trace_fn=lambda episode_id, episode: self._replace_runtime_episode_trace_locked(
                episode_id,
                episode,
            ),
            service_state_snapshot_fn=lambda **kwargs: self._service_state_snapshot(**kwargs),
        )

    def _cortex_active(self) -> bool:
        return self._thought_loop_actual is not None and self._thought_loop_actual.is_running

    # --- Status Read Model delegation (ADR 0003) ---
    def status(self, *, fresh_wait_seconds: float | None = None) -> dict[str, Any]:
        """Delegate to StatusReadModel for status snapshots."""
        return self._status_read_model.status(fresh_wait_seconds=fresh_wait_seconds)

    def terminus_status(self, *, fresh_wait_seconds: float | None = None) -> dict[str, Any]:
        """Delegate to StatusReadModel for terminus status snapshots."""
        return self._status_read_model.terminus_status(fresh_wait_seconds=fresh_wait_seconds)

    def sensory_previews(self, limit: int = 6) -> dict[str, Any]:
        """Delegate to StatusReadModel for sensory preview payloads."""
        return self._status_read_model.sensory_previews(limit=limit)

    def architecture_summary(self) -> dict[str, Any]:
        """Delegate to StatusReadModel for architecture summary."""
        return self._status_read_model.architecture_summary()

    def _architecture_summary_impl(self) -> dict[str, Any]:
        """Build the architecture summary under lock (called by read model callback)."""
        return ServiceReportingMixin.architecture_summary(self)

    def telemetry_snapshot(self) -> dict[str, Any]:
        """Delegate to StatusReadModel for telemetry snapshots."""
        return self._status_read_model.telemetry_snapshot()

    def living_loop_status(self) -> dict[str, Any]:
        """Delegate to StatusReadModel for living loop status snapshots."""
        return self._status_read_model.living_loop_status()

    def _living_loop_status_impl(self) -> dict[str, Any]:
        """Build living loop status under lock (called by read model callback)."""
        return LivingStatusMixin.living_loop_status(self)

    def policy_actuator_status(self) -> dict[str, Any]:
        """Delegate to StatusReadModel for policy actuator status snapshots."""
        return self._status_read_model.policy_actuator_status()

    def _policy_actuator_status_impl(self) -> dict[str, Any]:
        """Build policy actuator status under lock (called by read model callback)."""
        return LivingStatusMixin.policy_actuator_status(self)

    def _cortex_signal_state(self) -> dict[str, Any]:
        """Delegate to StatusReadModel for cortex signal state."""
        return self._status_read_model.cortex_signal_state()

    def _cortex_signal_state_impl(self) -> dict[str, Any]:
        """Build cortex signal state under lock (called by read model callback)."""
        return LivingStatusMixin._cortex_signal_state(self)

    def _cortex_factories_are_mocked(self) -> bool:
        refs = self._cortex_factory_refs or ()
        return any(
            "unittest.mock" in type(ref).__module__ or hasattr(ref, "mock_calls")
            for ref in refs
        )

    def _cortex_unavailable_snapshot(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "initialization": {
                "started": bool(getattr(self, "_cortex_init_started", False)),
                "finished": bool(getattr(self, "_cortex_init_finished", False)),
                "timed_out": bool(getattr(self, "_cortex_init_timed_out", False)),
                "error": getattr(self, "_cortex_init_error", None),
            },
        }

    def _inject_action_record_into_loop(self, thought_loop: Any, record: Mapping[str, Any]) -> None:
        content = " ".join(str(record.get("episode_text", "")).split()).strip()
        if not content:
            return
        verification_raw = record.get("verification")
        verification: Mapping[str, Any] = verification_raw if isinstance(verification_raw, Mapping) else {}
        thought_loop.inject_action_result(
            content=content,
            topics=tuple(str(item) for item in list(record.get("topics") or []) if str(item).strip()),
            success=bool(verification.get("success", False)),
            confidence=float(verification.get("confidence", 0.0) or 0.0),
            contradicted=bool(verification.get("contradiction", False)),
            metadata=self._action_history_memory_metadata(record),
        )

    def _build_cortex_thought_loop(self, action_history: Sequence[Mapping[str, Any]]) -> Any:
        if self._cortex_factory_refs is None:
            raise RuntimeError(self._cortex_init_error or "Cortex module unavailable")
        ThoughtLoop, create_cortex_from_env, create_embedder_from_env, EpisodicMemory = self._cortex_factory_refs
        cortex = create_cortex_from_env()
        embedder = create_embedder_from_env(allow_fallback=False)
        memory = EpisodicMemory(capacity=2048, embedder=embedder)
        thought_loop = ThoughtLoop(
            cortex=cortex,
            memory=memory,
            curiosity_controller=getattr(self, "_geometric_curiosity", None),
            signal_provider=self._cortex_signal_state,
            narrative_state_path=str(self._checkpoint_dir / "cortex_narrative_self.json"),
            on_thought=self._on_cortex_thought,
            on_sleep_summary=self._on_cortex_sleep_cycle,
        )
        for record in reversed(list(action_history)):
            self._inject_action_record_into_loop(thought_loop, record)
        _cortex_logger.info("Cortex module initialised (%s, embedder=%s)", cortex.model, type(embedder).__name__)
        return thought_loop

    def _start_cortex_initialization(self) -> None:
        with self._cortex_init_lock:
            if self._thought_loop_actual is not None:
                self._cortex_init_event.set()
                return
            if self._cortex_init_thread is not None and self._cortex_init_thread.is_alive():
                return
            if self._cortex_init_finished and self._cortex_init_error:
                return
            self._cortex_init_started = True
            self._cortex_init_finished = False
            self._cortex_init_timed_out = False
            self._cortex_init_error = None
            self._cortex_init_event.clear()
            action_history = list(self._action_history)

            def _runner() -> None:
                try:
                    thought_loop = self._build_cortex_thought_loop(action_history)
                except RuntimeError as exc:
                    self._cortex_init_error = str(exc)
                    _cortex_logger.warning("Cortex disabled: %s", exc)
                except Exception as exc:  # pragma: no cover - defensive init guard
                    self._cortex_init_error = str(exc)
                    _cortex_logger.info("Cortex module unavailable: %s", exc)
                else:
                    self._thought_loop_actual = thought_loop
                    self._cortex_available = True
                    if bool(getattr(self, "_brain_running", False)) and not thought_loop.is_running:
                        try:
                            thought_loop.start()
                            _cortex_logger.info("ThoughtLoop started after delayed cortex initialization")
                        except Exception as exc:
                            _cortex_logger.warning("ThoughtLoop failed to start after delayed initialization: %s", exc)
                finally:
                    self._cortex_init_finished = True
                    self._cortex_init_event.set()

            self._cortex_init_thread = Thread(target=_runner, name="hecsn-cortex-init", daemon=True)
            self._cortex_init_thread.start()

    def _ensure_cortex_initialized(self, *, wait_seconds: float | None = DEFAULT_CORTEX_INIT_TIMEOUT_SECONDS) -> Any:
        if self._thought_loop_actual is not None:
            return self._thought_loop_actual
        self._start_cortex_initialization()
        if self._thought_loop_actual is not None:
            return self._thought_loop_actual
        if wait_seconds is not None:
            if not self._cortex_init_event.wait(timeout=max(0.0, float(wait_seconds))):
                self._cortex_init_timed_out = True
                _cortex_logger.warning("Cortex initialization still pending after %.2fs", float(wait_seconds))
                return None
        return self._thought_loop_actual

    def close(self) -> None:
        # Stop cortex first (signal, no join yet)
        if self._thought_loop is not None and self._thought_loop.is_running:
            self._thought_loop.request_stop()

        thread = self._request_brain_stop(reason="shutdown")
        self._join_brain_thread(thread, raise_on_timeout=False)
        prewarm_thread = self._request_ingestion_prewarm_stop()
        self._join_ingestion_prewarm_thread(prewarm_thread)
        promotion_thread = self._request_remote_warm_promotion_stop()
        self._join_remote_warm_promotion_thread(promotion_thread)

        # Join cortex thread outside locks
        if self._thought_loop is not None:
            try:
                self._thought_loop.stop(timeout=3.0)
            except Exception:
                pass

        with self._lock:
            self._close_brain_sources_locked()
            self._close_sensory_sources_locked()
