from __future__ import annotations

import base64
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
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
from hecsn.reporting.io import write_json_file
from hecsn.semantics import ConceptStore, GeometricCuriosityController
from hecsn.semantics.grounding_text import match_terms, salient_query_terms
from hecsn.training.autonomy_acquisition_runner import run_live_acquisition
from hecsn.training.checkpointing import load_trainer_checkpoint, save_trainer_checkpoint
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

from hecsn.service.action_loop import DigitalActionResult, execute_digital_action
from hecsn.service.living_loop import (
    ActionExecutionRecord,
    ConsolidationRecord,
    OperationalSelfModel,
    ProvenanceState,
)
from hecsn.service.terminus_presets import TERMINUS_QUICK_START_PRESETS
from hecsn.service.terminus_sensory import SensoryEpisode, bootstrap_sensory_episode_from_row, build_sensory_stream, sensory_bootstrap_columns


from hecsn.service.terminus_autonomy import _canonical_provider_term  # noqa: E402


class _TimedCallFailure:
    def __init__(self, error: BaseException) -> None:
        self.error = error


@dataclass
class _BrainSourceRuntime:
    spec: dict[str, Any]
    stream: Iterator[tuple[str, torch.Tensor]]
    tokens_processed: int = 0
    cycles_completed: int = 0
    exhausted: bool = False
    tick_visits: int = 0
    last_tokens_trained: int = 0
    last_activity_at: str | None = None
    prefetched_tokens: int = 0
    prefetch_events: int = 0
    last_prefetch_token_count: int = 0
    last_prefetch_at: str | None = None
    last_prefetch_duration_ms: float | None = None
    last_prefetch_error: str | None = None
    queue_hits: int = 0
    last_buffer_tokens_served: int = 0
    last_semantic_match: float = 0.0
    last_selection_score: float = 0.0
    last_fairness_score: float = 0.0
    last_buffer_readiness: float = 0.0
    last_utility_score: float = 0.0
    buffered_patterns: deque[tuple[str, torch.Tensor]] = field(default_factory=deque)
    bootstrap_attempted: bool = False

    @property
    def name(self) -> str:
        return str(self.spec.get("name", "source"))

    @property
    def source_type(self) -> str:
        return str(self.spec.get("source_type", "auto"))


@dataclass
class _SensorySourceRuntime:
    spec: dict[str, Any]
    stream: Iterator[SensoryEpisode]
    episodes_processed: int = 0
    cycles_completed: int = 0
    exhausted: bool = False
    last_activity_at: str | None = None
    last_text: str | None = None
    last_semantic_match: float = 0.0
    last_modality_need: float = 0.0
    last_selection_score: float = 0.0
    last_window_budget: int = 0
    buffered_episodes: list[SensoryEpisode] = field(default_factory=list)
    prefetched_episodes: int = 0
    prefetch_events: int = 0
    last_prefetch_episode_count: int = 0
    last_prefetch_at: str | None = None
    last_prefetch_duration_ms: float | None = None
    last_prefetch_error: str | None = None
    queue_hits: int = 0
    last_buffer_episodes_served: int = 0
    last_item_semantic_match: float = 0.0
    last_item_candidates_considered: int = 0
    last_item_retrieval_lookahead: int = 0
    bootstrap_attempted: bool = False

    @property
    def name(self) -> str:
        return str(self.spec.get("name", "sensory_source"))

    @property
    def adapter(self) -> str:
        return str(self.spec.get("adapter", "unknown"))


from hecsn.service.terminus_autonomy import TerminusAutonomyMixin


class HECSNServiceManager(TerminusAutonomyMixin):
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
        self._brain_last_event: dict[str, Any] | None = None
        self._brain_event_history: deque[dict[str, Any]] = deque(maxlen=16)
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
        self._dirty_state = False
        self._state_revision = 0
        self._load_persisted_traces_locked()

        # --- Cortex / ThoughtLoop (requires NVIDIA_API_KEY) ---
        self._thought_loop: Any = None  # type: ThoughtLoop | None
        self._cortex_available = False
        try:
            from hecsn.cortex.thought_loop import ThoughtLoop
            from hecsn.cortex.multi_cortex import create_cortex_from_env, create_embedder_from_env
            from hecsn.cortex.episodic_memory import EpisodicMemory

            cortex = create_cortex_from_env()
            embedder = create_embedder_from_env(allow_fallback=False)
            memory = EpisodicMemory(capacity=2048, embedder=embedder)
            curiosity_ctrl = getattr(self, "_geometric_curiosity", None)
            self._thought_loop = ThoughtLoop(
                cortex=cortex,
                memory=memory,
                curiosity_controller=curiosity_ctrl,
                signal_provider=self._cortex_signal_state,
                narrative_state_path=str(self._checkpoint_dir / "cortex_narrative_self.json"),
                on_thought=self._on_cortex_thought,
                on_sleep_summary=self._on_cortex_sleep_cycle,
            )
            self._replay_action_history_into_cortex_locked()
            self._cortex_available = True
            _cortex_logger.info("Cortex module initialised (%s, embedder=%s)", cortex.model, type(embedder).__name__)

        except RuntimeError as exc:
            # API key missing or NIM unreachable — cortex disabled
            _cortex_logger.warning("Cortex disabled: %s", exc)
        except Exception as exc:
            _cortex_logger.info("Cortex module unavailable: %s", exc)

    def _runtime_environment_summary(self) -> dict[str, Any]:
        return {
            "env_root": None if self._env_root is None else str(self._env_root),
            "dotenv_available": bool(self._runtime_env.get("dotenv_available", False)),
            "dotenv_loaded": bool(self._runtime_env.get("dotenv_loaded", False)),
            "dotenv_path": self._runtime_env.get("dotenv_path"),
            "reason": str(self._runtime_env.get("reason", "unknown")),
            "nvidia_api_key_present": bool(os.environ.get("NVIDIA_API_KEY", "").strip()),
            "hf_token_present": bool(huggingface_token_from_env()),
        }

    def _status_snapshot_locked(self) -> dict[str, Any]:
        last_trace = self._trace_history[0] if self._trace_history else None
        return {
            "checkpoint_path": str(self._checkpoint_path),
            "dirty_state": bool(self._dirty_state),
            "state_revision": int(self._state_revision),
            "token_count": int(self._trainer.token_count),
            "last_winner": None if self._trainer.last_winner is None else int(self._trainer.last_winner),
            "context_supported": bool(self._trainer.model.context_layer is not None),
            "context_state_norm": float(torch.norm(self._trainer.context_state().float()).item()),
            "trace_history_size": int(len(self._trace_history)),
            "trace_storage_dir": str(self._trace_dir),
            "last_trace_id": None if last_trace is None else str(last_trace.get("trace_id")),
            "last_trace_created_at": None if last_trace is None else str(last_trace.get("created_at")),
            "checkpoint_metadata": deepcopy(self._metadata),
            "dopamine": float(self._trainer.model.surprise.dopamine),
            "serotonin": float(self._trainer.model.surprise.serotonin),
            "acetylcholine": float(self._trainer.model.surprise.acetylcholine),
            "norepinephrine": float(self._trainer.model.surprise.norepinephrine),
            "runtime_scope": self._trainer.model.runtime_scope_report(),
            "memory_store": self._trainer.model.memory_store.summary_stats(),
            "concept_store": self._concept_store.snapshot(),
            "terminus_runtime": self._brain_runtime_snapshot_locked(),
        }

    def status(self, *, fresh_wait_seconds: float | None = None) -> dict[str, Any]:
        # Default behavior stays non-blocking for the service surface. When a
        # caller explicitly requests a fresh snapshot, keep retrying briefly and
        # then block rather than silently serving a stale cached snapshot.
        if fresh_wait_seconds is None:
            acquired = self._lock.acquire(timeout=0.15)
            if not acquired:
                cached = getattr(self, "_cached_status", None)
                if cached is not None:
                    return cached
                self._lock.acquire()
        else:
            deadline = time.perf_counter() + max(0.0, float(fresh_wait_seconds))
            acquired = False
            while time.perf_counter() < deadline:
                remaining = max(0.0, deadline - time.perf_counter())
                if self._lock.acquire(timeout=min(0.15, remaining)):
                    acquired = True
                    break
            if not acquired:
                self._lock.acquire()
        try:
            result = self._status_snapshot_locked()
            self._cached_status = result
            return result
        finally:
            self._lock.release()

    def telemetry_snapshot(self) -> dict[str, Any]:
        # Non-blocking: return cached data when brain loop holds the lock
        # (prevents SSE/API starvation during training or HF network I/O).
        acquired = self._lock.acquire(timeout=0.15)
        if not acquired:
            cached = getattr(self, "_cached_telemetry", None)
            if cached is not None:
                return cached
            # No cache yet (first call) — must block
            self._lock.acquire()
        try:
            return self._telemetry_snapshot_locked()
        finally:
            self._lock.release()

    def _telemetry_snapshot_locked(self) -> dict[str, Any]:
        """Build the telemetry dict. Caller MUST hold self._lock."""
        current_rev = int(self._state_revision)
        cortex_active = self._thought_loop is not None and self._thought_loop.is_running
        cached = getattr(self, "_cached_telemetry", None)
        cached_rev = getattr(self, "_cached_telemetry_rev", -1)
        if not cortex_active and cached is not None and cached_rev == current_rev:
            return cached

        memory_store = self._trainer.model.memory_store.summary_stats()
        last_trace = self._trace_history[0] if self._trace_history else None
        drift_value = (
            self._trainer._cached_drift
            if self._trainer._cached_drift is not None
            else self._trainer.model.memory_store.compute_drift(
                self._trainer.last_winner if self._trainer.config.use_winner_local_drift else None
            )
        )
        snapshot = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_path": str(self._checkpoint_path),
            "dirty_state": bool(self._dirty_state),
            "state_revision": current_rev,
            "token_count": int(self._trainer.token_count),
            "last_winner": None if self._trainer.last_winner is None else int(self._trainer.last_winner),
            "context_state_norm": float(torch.norm(self._trainer.context_state().float()).item()),
            "trace_history_size": int(len(self._trace_history)),
            "last_trace_id": None if last_trace is None else str(last_trace.get("trace_id")),
            "last_trace_created_at": None if last_trace is None else str(last_trace.get("created_at")),
            "memory_fill_fraction": float(memory_store.get("fill_fraction", 0.0)),
            "memory_buffer_size": int(memory_store.get("size", 0)),
            "sleep_events": int(self._trainer.sleep_events),
            "micro_sleep_events": int(self._trainer.micro_sleep_events),
            "deep_sleep_events": int(self._trainer.deep_sleep_events),
            "dopamine": float(self._trainer.model.surprise.dopamine),
            "serotonin": float(self._trainer.model.surprise.serotonin),
            "acetylcholine": float(self._trainer.model.surprise.acetylcholine),
            "norepinephrine": float(self._trainer.model.surprise.norepinephrine),
            "drift": float(drift_value),
            "drift_floor": float(self._trainer.current_rolling_drift_floor if self._trainer.current_rolling_drift_floor is not None else drift_value),
            "grounding_confidence": {
                w: round(c, 4)
                for w, c in self._trainer.word_grounding_confidence.items()
            },
            "n_visual_signatures": len(self._trainer.word_visual_signature),
            "n_audio_signatures": len(self._trainer.word_audio_signature),
            "cross_modal_visual_confidence": (
                float(self._trainer.model.cross_modal.visual_confidence.mean().item())
                if self._trainer.model.cross_modal is not None else None
            ),
            "cross_modal_audio_confidence": (
                float(self._trainer.model.cross_modal.audio_confidence.mean().item())
                if self._trainer.model.cross_modal is not None else None
            ),
            "animation": self._animation_snapshot_locked(),
            "terminus_runtime": self._brain_runtime_snapshot_locked(),
        }
        self._cached_telemetry = snapshot
        self._cached_telemetry_rev = current_rev
        return snapshot

    def checkpoint_list(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self._checkpoint_dir.exists():
                return []
            records: list[dict[str, Any]] = []
            for path in sorted(self._checkpoint_dir.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True):
                stat = path.stat()
                records.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "size_bytes": int(stat.st_size),
                        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    }
                )
            return records

    def query(
        self,
        *,
        query_text: str,
        context_text: str | None = None,
        top_k_candidates: int = 5,
        top_k_memories: int = 5,
        top_chars: int = 6,
    ) -> dict[str, Any]:
        with self._lock:
            result = self._build_query_locked(
                query_text=query_text,
                context_text=context_text,
                top_k_candidates=top_k_candidates,
                top_k_memories=top_k_memories,
                top_chars=top_chars,
            )
            result["concept_summary"] = self._observe_concepts_locked(
                query_text=query_text,
                query_result=result,
            )
            result["gap_plan"] = self._plan_gaps_locked(
                query_text=query_text,
                query_result=result,
            )
            result["delayed_consequence"] = self._apply_delayed_query_consequence_locked(
                query_result=result,
            )
            self._record_recent_query_gap_locked(
                query_text=query_text,
                gap_plan=result["gap_plan"],
                source="query",
            )
            result["service_state"] = self._service_state_snapshot()
            return result

    def feed(self, *, text: str) -> dict[str, Any]:
        with self._lock:
            summary = feed_text(
                self._trainer,
                self._encoder,
                text,
                on_step=self._runtime_concept_callback_locked(),
            )
            self._mark_mutated()
            return {
                "feed_summary": summary,
                "dirty_state": bool(self._dirty_state),
                "state_revision": int(self._state_revision),
            }

    def respond(
        self,
        *,
        query_text: str,
        context_text: str | None = None,
        top_k_candidates: int = 5,
        top_k_memories: int = 5,
        top_chars: int = 6,
        max_evidence_items: int = 3,
        learn_mode: str = "user_and_selected_evidence",
    ) -> dict[str, Any]:
        with self._lock:
            state_before = self._service_state_snapshot()
            query_result = self._build_query_locked(
                query_text=query_text,
                context_text=context_text,
                top_k_candidates=top_k_candidates,
                top_k_memories=top_k_memories,
                top_chars=top_chars,
            )
            query_result["concept_summary"] = self._observe_concepts_locked(
                query_text=query_text,
                query_result=query_result,
            )
            query_result["gap_plan"] = self._plan_gaps_locked(
                query_text=query_text,
                query_result=query_result,
            )
            query_result["delayed_consequence"] = self._apply_delayed_query_consequence_locked(
                query_result=query_result,
            )
            self._record_recent_query_gap_locked(
                query_text=query_text,
                gap_plan=query_result["gap_plan"],
                source="respond",
            )
            query_summary = query_result.get("query_summary") or {}
            response = self._responder.build_response(
                query_text=query_text,
                query_summary=query_summary,
                concept_summary=query_result.get("concept_summary"),
                max_evidence_items=max_evidence_items,
            )
            action_assist = self._maybe_auto_action_assist_locked(
                query_text=query_text,
                query_result=query_result,
                response=response,
            )
            if action_assist is not None:
                if int(action_assist.get("response_episode_count", 0) or 0) > 0:
                    query_summary = query_result.get("query_summary") or {}
                    response = self._responder.build_response(
                        query_text=query_text,
                        query_summary=query_summary,
                        concept_summary=query_result.get("concept_summary"),
                        max_evidence_items=max_evidence_items,
                    )
                    action_assist["used_in_response"] = True
                response_note = self._normalize_action_text(action_assist.get("response_note", ""))
                if response_note:
                    base_text = self._normalize_action_text(response.get("response_text", ""))
                    if response_note.strip() not in base_text:
                        response["response_text"] = (base_text + response_note).strip()
                        action_assist["used_in_response"] = True
                query_result["action_assist"] = deepcopy(action_assist)
                response["action_assist"] = deepcopy(action_assist)
            response_outcome_score = self._response_grounded_outcome_score_locked(
                query_result=query_result,
                response=response,
                action_assist=action_assist,
            )
            applied_background_provenance = self._apply_background_source_response_provenance_locked(
                response=response,
                outcome_score=response_outcome_score,
            )
            if not applied_background_provenance:
                self._apply_background_source_outcome_calibration_locked(
                    query_text=query_text,
                    outcome_score=response_outcome_score,
                )
            autonomy = cast(dict[str, Any] | None, self._brain_config.get("autonomy"))
            if autonomy is not None:
                self._apply_provider_response_outcome_calibration_locked(
                    autonomy=autonomy,
                    response=response,
                    outcome_score=response_outcome_score,
                )
            learning = self._learn_from_turn_locked(query_text=query_text, response=response, learn_mode=learn_mode)
            delayed_candidate = self._record_response_consequence_candidate_locked(
                query_result=query_result,
                response=response,
                outcome_score=response_outcome_score,
            )
            if delayed_candidate is not None:
                response["delayed_consequence_candidate"] = deepcopy(delayed_candidate)
            trace = {
                "trace_id": str(uuid4()),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "operation": "respond",
                "request": {
                    "query_text": query_text,
                    "context_text": context_text,
                    "top_k_candidates": int(top_k_candidates),
                    "top_k_memories": int(top_k_memories),
                    "top_chars": int(top_chars),
                    "max_evidence_items": int(max_evidence_items),
                    "learn_mode": learn_mode,
                },
                "state_before": state_before,
                "query_result": query_result,
                "response": response,
                "learning": learning,
                "state_after": self._service_state_snapshot(),
            }
            trace_path = self._persist_trace_locked(trace)
            return {
                "trace_id": trace["trace_id"],
                "trace_path": str(trace_path),
                "created_at": trace["created_at"],
                "query_result": query_result,
                "response": response,
                "learning": learning,
                "dirty_state": bool(self._dirty_state),
                "state_revision": int(self._state_revision),
            }

    def acquire(
        self,
        *,
        preset: str = PUBLIC_ACQUISITION_PRESET,
        policy: str = "active",
        acquisition_slots: int | None = None,
        acquisition_tokens: int | None = None,
        save_checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if preset not in PUBLIC_ACQUISITION_PRESETS:
                raise ValueError(
                    "Unsupported acquisition preset for the maintained service surface. "
                    f"Supported presets: {', '.join(PUBLIC_ACQUISITION_PRESETS)}"
                )
            if policy not in PUBLIC_ACQUISITION_POLICIES:
                raise ValueError(
                    "Unsupported acquisition policy for the maintained service surface. "
                    f"Supported policies: {', '.join(PUBLIC_ACQUISITION_POLICIES)}"
                )
            preset_args = get_autonomy_acquisition_preset(preset)
            state_before = self._service_state_snapshot()
            focus_plan = self._autonomy_focus_plan_locked()
            shortlist_size, shortlist_gap_weight, shortlist_affinity_weight = self._autonomy_shortlist_settings_locked(
                candidate_bank=list(preset_args.get("candidate_bank", [])),
                config=preset_args,
                focus_plan=focus_plan,
            )
            result = run_live_acquisition(
                trainer=self._trainer,
                encoder=self._encoder,
                candidate_bank_specs=self._autonomy_candidate_specs_locked(
                    candidate_bank=list(preset_args.get("candidate_bank", [])),
                    focus_plan=focus_plan,
                ),
                candidate_train_tokens=int(preset_args.get("candidate_train_tokens", 0)),
                probe_tokens=int(preset_args.get("probe_tokens", 0)),
                acquisition_tokens=int(acquisition_tokens if acquisition_tokens is not None else preset_args.get("acquisition_tokens", 0)),
                acquisition_slots=int(acquisition_slots if acquisition_slots is not None else preset_args.get("acquisition_slots", 1)),
                gap_exploration_bonus=float(preset_args.get("gap_exploration_bonus", 0.0)),
                gap_ambiguity_weight=float(preset_args.get("gap_ambiguity_weight", 0.0)),
                gap_switch_weight=float(preset_args.get("gap_switch_weight", 0.0)),
                gap_margin_reference=float(preset_args.get("gap_margin_reference", 0.12)),
                coverage_balance_penalty=float(preset_args.get("coverage_balance_penalty", 0.0)),
                gap_focus_margin=float(preset_args.get("gap_focus_margin", 0.0)),
                policy_name=policy,
                semantic_shortlist_size=shortlist_size,
                semantic_shortlist_gap_weight=shortlist_gap_weight,
                semantic_shortlist_affinity_weight=shortlist_affinity_weight,
                semantic_plan=focus_plan,
                on_train_step=self._runtime_concept_callback_locked(),
            )
            if int(result.get("tokens_trained_total", 0)) > 0:
                self._mark_mutated()

            checkpoint_save = None
            if save_checkpoint_path is not None:
                checkpoint_save = self.save_checkpoint(save_checkpoint_path)

            trace = {
                "trace_id": str(uuid4()),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "operation": "acquisition",
                "request": {
                    "preset": preset,
                    "policy": policy,
                    "acquisition_slots": acquisition_slots,
                    "acquisition_tokens": acquisition_tokens,
                    "save_checkpoint_path": save_checkpoint_path,
                },
                "state_before": state_before,
                "acquisition_result": result,
                "checkpoint_save": checkpoint_save,
                "state_after": self._service_state_snapshot(),
            }
            trace_path = self._persist_trace_locked(trace)
            return {
                "trace_id": trace["trace_id"],
                "trace_path": str(trace_path),
                "created_at": trace["created_at"],
                "preset": preset,
                "policy": policy,
                "acquisition_result": result,
                "checkpoint_save": checkpoint_save,
                "dirty_state": bool(self._dirty_state),
                "state_revision": int(self._state_revision),
                "token_count": int(self._trainer.token_count),
            }

    def _multimodal_runtime_summary_locked(self) -> dict[str, Any]:
        sensory = self._brain_config.get("sensory") or {}
        cross_modal_enabled = bool(getattr(self._trainer.config, "enable_cross_modal", False))
        real_enabled = bool(sensory.get("enabled", False)) and cross_modal_enabled
        visual_confidence, audio_confidence = self._cross_modal_confidence_means_locked()
        next_source_name = None
        if self._sensory_source_runtimes:
            next_source_name = self._sensory_source_runtimes[
                self._sensory_source_index % len(self._sensory_source_runtimes)
            ].name
        return {
            "enabled": bool(real_enabled),
            "mode": "real_hf_sensory" if real_enabled else "disabled",
            "episodes_completed": int(self._real_sensory_episodes_completed),
            "real_episodes_completed": int(self._real_sensory_episodes_completed),
            "tokens_since_real_episode": int(
                max(0, int(self._trainer.token_count) - int(self._last_real_sensory_episode_token_count))
            ),
            "real_episode_interval": int(sensory.get("episode_interval_tokens", 2048)) if sensory else 0,
            "items_per_real_episode": int(sensory.get("items_per_episode", 1)) if sensory else 0,
            "base_windows_per_item": int(sensory.get("base_windows_per_item", 0)) if sensory else 0,
            "max_windows_per_item": int(sensory.get("max_windows_per_item", 0)) if sensory else 0,
            "confidence_window_gain": float(sensory.get("confidence_window_gain", 0.0)) if sensory else 0.0,
            "semantic_window_gain": float(sensory.get("semantic_window_gain", 0.0)) if sensory else 0.0,
            "item_retrieval_lookahead": int(sensory.get("item_retrieval_lookahead", 1)) if sensory else 0,
            "item_retrieval_semantic_weight": float(sensory.get("item_retrieval_semantic_weight", 0.0)) if sensory else 0.0,
            "observation_salience": float(sensory.get("observation_salience", 0.0)) if sensory else 0.0,
            "cross_modal_visual_accepted": int(self._real_visual_accepted),
            "cross_modal_audio_accepted": int(self._real_audio_accepted),
            "real_cross_modal_visual_accepted": int(self._real_visual_accepted),
            "real_cross_modal_audio_accepted": int(self._real_audio_accepted),
            "visual_confidence_mean": visual_confidence,
            "audio_confidence_mean": audio_confidence,
            "focus_terms": list(self._last_sensory_focus_terms),
            "recent_preview_count": int(len(self._sensory_preview_history)),
            "latest_preview_id": (
                None if not self._sensory_preview_history else str(self._sensory_preview_history[0].get("preview_id", ""))
            ),
            "source_names": [runtime.name for runtime in self._sensory_source_runtimes],
            "next_source_name": next_source_name,
            "last_real_error": self._real_sensory_last_error,
        }

    def _ingestion_ready_source_count_locked(self) -> int:
        return int(sum(1 for runtime in self._brain_source_runtimes if len(runtime.buffered_patterns) > 0))

    def _ingestion_full_queue_source_count_locked(self) -> int:
        ingestion = self._brain_config.get("ingestion") or {}
        tick_tokens = int(self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS))
        queue_target_tokens = int(ingestion.get("queue_target_tokens", tick_tokens))
        return int(
            sum(
                1
                for runtime in self._brain_source_runtimes
                if len(runtime.buffered_patterns) >= queue_target_tokens
            )
        )

    def _ingestion_startup_state_locked(self) -> str:
        if not self._brain_config.get("source_bank"):
            return "unconfigured"
        ingestion = self._brain_config.get("ingestion") or {}
        if not bool(ingestion.get("enabled", True)):
            return "disabled"
        if self._ingestion_warm_ready_at is not None:
            return "warm"
        if self._ingestion_prewarm_running:
            return "warming"
        return "cold"

    def _maybe_mark_ingestion_warm_locked(self, *, trigger: str) -> None:
        if self._ingestion_warm_ready_at is not None:
            return
        ready_source_count = self._ingestion_ready_source_count_locked()
        if ready_source_count <= 0:
            return
        self._ingestion_warm_ready_at = datetime.now(timezone.utc).isoformat()
        if self._ingestion_configured_perf is not None:
            self._ingestion_startup_warm_latency_ms = float(
                (time.perf_counter() - self._ingestion_configured_perf) * 1000.0
            )
        self._record_brain_event_locked(
            {
                "type": "ingestion_warm_ready",
                "timestamp": self._ingestion_warm_ready_at,
                "trigger": trigger,
                "ready_source_count": int(ready_source_count),
                "startup_warm_latency_ms": self._ingestion_startup_warm_latency_ms,
            }
        )

    def _ingestion_runtime_summary_locked(self) -> dict[str, Any]:
        ingestion = self._brain_config.get("ingestion") or {}
        tick_tokens = int(self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS))
        queue_target_tokens = int(ingestion.get("queue_target_tokens", tick_tokens))
        total_buffered_tokens = sum(len(runtime.buffered_patterns) for runtime in self._brain_source_runtimes)
        ready_source_count = self._ingestion_ready_source_count_locked()
        full_queue_source_count = self._ingestion_full_queue_source_count_locked()
        latest_prefetch_at = max(
            (str(runtime.last_prefetch_at) for runtime in self._brain_source_runtimes if runtime.last_prefetch_at),
            default=None,
        )
        return {
            "enabled": bool(ingestion.get("enabled", True)),
            "queue_target_tokens": int(queue_target_tokens),
            "prewarm_on_startup": bool(ingestion.get("prewarm_on_startup", False)),
            "prewarm_max_seconds": float(ingestion.get("prewarm_max_seconds", 5.0)),
            "prewarm_budget_exhausted": bool(self._ingestion_prewarm_budget_exhausted),
            "source_count": int(len(self._brain_source_runtimes)),
            "startup_state": self._ingestion_startup_state_locked(),
            "configured_at": self._ingestion_configured_at,
            "prewarm_running": bool(self._ingestion_prewarm_running),
            "prewarm_started_at": self._ingestion_prewarm_started_at,
            "prewarm_completed_at": self._ingestion_prewarm_completed_at,
            "prewarm_last_duration_ms": self._ingestion_prewarm_last_duration_ms,
            "prewarm_last_error": self._ingestion_prewarm_last_error,
            "prewarm_runs": int(self._ingestion_prewarm_run_count),
            "prewarm_last_trigger": self._ingestion_prewarm_last_trigger,
            "warm_ready_at": self._ingestion_warm_ready_at,
            "startup_warm_latency_ms": self._ingestion_startup_warm_latency_ms,
            "total_buffered_tokens": int(total_buffered_tokens),
            "buffered_source_count": int(sum(1 for runtime in self._brain_source_runtimes if runtime.buffered_patterns)),
            "ready_source_count": int(ready_source_count),
            "warm_ready": bool(ready_source_count > 0),
            "full_queue_source_count": int(full_queue_source_count),
            "full_warm_ready": bool(full_queue_source_count > 0),
            "prefetch_events": int(sum(runtime.prefetch_events for runtime in self._brain_source_runtimes)),
            "prefetched_tokens": int(sum(runtime.prefetched_tokens for runtime in self._brain_source_runtimes)),
            "queue_hits": int(sum(runtime.queue_hits for runtime in self._brain_source_runtimes)),
            "last_prefetch_at": latest_prefetch_at,
        }

    def _sensory_queue_target_items_locked(self) -> int:
        sensory = self._brain_config.get("sensory") or {}
        items_per_episode = max(1, int(sensory.get("items_per_episode", 1)))
        lookahead = max(1, int(sensory.get("item_retrieval_lookahead", 1)))
        return max(1, int(sensory.get("queue_target_items", max(items_per_episode, lookahead))))

    def _sensory_ready_source_count_locked(self) -> int:
        return int(sum(1 for runtime in self._sensory_source_runtimes if len(runtime.buffered_episodes) > 0))

    def _sensory_full_queue_source_count_locked(self) -> int:
        target_items = self._sensory_queue_target_items_locked()
        return int(sum(1 for runtime in self._sensory_source_runtimes if len(runtime.buffered_episodes) >= target_items))

    def _sensory_startup_state_locked(self) -> str:
        sensory = self._brain_config.get("sensory")
        if sensory is None or not bool(sensory.get("enabled", False)):
            return "disabled"
        if self._sensory_warm_ready_at is not None:
            return "warm"
        if self._ingestion_prewarm_running and bool(sensory.get("prewarm_on_startup", False)):
            return "warming"
        return "cold"

    def _maybe_mark_sensory_warm_locked(self, *, trigger: str) -> None:
        if self._sensory_warm_ready_at is not None:
            return
        ready_source_count = self._sensory_ready_source_count_locked()
        if ready_source_count <= 0:
            return
        self._sensory_warm_ready_at = datetime.now(timezone.utc).isoformat()
        if self._sensory_configured_perf is not None:
            self._sensory_startup_warm_latency_ms = float(
                (time.perf_counter() - self._sensory_configured_perf) * 1000.0
            )
        self._record_brain_event_locked(
            {
                "type": "sensory_warm_ready",
                "timestamp": self._sensory_warm_ready_at,
                "trigger": trigger,
                "ready_source_count": int(ready_source_count),
                "startup_warm_latency_ms": self._sensory_startup_warm_latency_ms,
            }
        )

    def _sensory_runtime_summary_locked(self, sensory: dict[str, Any]) -> dict[str, Any]:
        queue_target_items = self._sensory_queue_target_items_locked()
        latest_prefetch_at = max(
            (str(runtime.last_prefetch_at) for runtime in self._sensory_source_runtimes if runtime.last_prefetch_at),
            default=None,
        )
        ready_source_count = self._sensory_ready_source_count_locked()
        full_queue_source_count = self._sensory_full_queue_source_count_locked()
        return {
            "enabled": bool(sensory.get("enabled", False)),
            "episode_interval_tokens": int(sensory.get("episode_interval_tokens", 2048)),
            "items_per_episode": int(sensory.get("items_per_episode", 1)),
            "base_windows_per_item": int(sensory.get("base_windows_per_item", 4)),
            "max_windows_per_item": int(sensory.get("max_windows_per_item", 10)),
            "confidence_window_gain": float(sensory.get("confidence_window_gain", 3.0)),
            "semantic_window_gain": float(sensory.get("semantic_window_gain", 3.0)),
            "item_retrieval_lookahead": int(sensory.get("item_retrieval_lookahead", 1)),
            "item_retrieval_semantic_weight": float(sensory.get("item_retrieval_semantic_weight", 0.72)),
            "modality_target_confidence": float(sensory.get("modality_target_confidence", 0.70)),
            "observation_salience": float(sensory.get("observation_salience", 0.82)),
            "cooldown_seconds": float(sensory.get("cooldown_seconds", 10.0)),
            "repeat_sources": bool(sensory.get("repeat_sources", True)),
            "queue_target_items": int(queue_target_items),
            "prewarm_on_startup": bool(sensory.get("prewarm_on_startup", False)),
            "prewarm_max_seconds": float(sensory.get("prewarm_max_seconds", 5.0)),
            "prewarm_budget_exhausted": bool(self._sensory_prewarm_budget_exhausted),
            "startup_state": self._sensory_startup_state_locked(),
            "configured_at": self._sensory_configured_at,
            "prewarm_running": bool(self._ingestion_prewarm_running and bool(sensory.get("prewarm_on_startup", False))),
            "prewarm_started_at": self._ingestion_prewarm_started_at,
            "prewarm_completed_at": self._ingestion_prewarm_completed_at,
            "prewarm_last_duration_ms": self._ingestion_prewarm_last_duration_ms,
            "prewarm_last_error": self._ingestion_prewarm_last_error,
            "prewarm_runs": int(self._ingestion_prewarm_run_count),
            "prewarm_last_trigger": self._ingestion_prewarm_last_trigger,
            "warm_ready_at": self._sensory_warm_ready_at,
            "startup_warm_latency_ms": self._sensory_startup_warm_latency_ms,
            "total_buffered_items": int(sum(len(runtime.buffered_episodes) for runtime in self._sensory_source_runtimes)),
            "buffered_source_count": int(sum(1 for runtime in self._sensory_source_runtimes if runtime.buffered_episodes)),
            "ready_source_count": int(ready_source_count),
            "warm_ready": bool(ready_source_count > 0),
            "full_queue_source_count": int(full_queue_source_count),
            "full_warm_ready": bool(full_queue_source_count > 0),
            "prefetch_events": int(sum(runtime.prefetch_events for runtime in self._sensory_source_runtimes)),
            "prefetched_items": int(sum(runtime.prefetched_episodes for runtime in self._sensory_source_runtimes)),
            "queue_hits": int(sum(runtime.queue_hits for runtime in self._sensory_source_runtimes)),
            "last_prefetch_at": latest_prefetch_at,
            "tokens_until_trigger": None,
            "trigger_ready": None,
            "last_episode_at": None if self._last_real_sensory_episode_time <= 0 else self._last_real_sensory_episode_time,
            "last_episode_token_count": int(self._last_real_sensory_episode_token_count),
            "source_bank": deepcopy(list(sensory.get("source_bank", []))),
            "focus_terms": list(self._last_sensory_focus_terms),
            "source_progress": [
                {
                    "name": runtime.name,
                    "adapter": runtime.adapter,
                    "episodes_processed": int(runtime.episodes_processed),
                    "cycles_completed": int(runtime.cycles_completed),
                    "exhausted": bool(runtime.exhausted),
                    "last_activity_at": runtime.last_activity_at,
                    "last_text": runtime.last_text,
                    "buffered_items": int(len(runtime.buffered_episodes)),
                    "buffer_fill_fraction": float(
                        0.0 if queue_target_items <= 0 else float(len(runtime.buffered_episodes)) / float(queue_target_items)
                    ),
                    "prefetch_events": int(runtime.prefetch_events),
                    "prefetched_items": int(runtime.prefetched_episodes),
                    "last_prefetch_item_count": int(runtime.last_prefetch_episode_count),
                    "last_prefetch_at": runtime.last_prefetch_at,
                    "last_prefetch_duration_ms": runtime.last_prefetch_duration_ms,
                    "last_prefetch_error": runtime.last_prefetch_error,
                    "queue_hits": int(runtime.queue_hits),
                    "last_buffer_items_served": int(runtime.last_buffer_episodes_served),
                    "last_semantic_match": float(runtime.last_semantic_match),
                    "last_item_semantic_match": float(runtime.last_item_semantic_match),
                    "last_item_candidates_considered": int(runtime.last_item_candidates_considered),
                    "last_item_retrieval_lookahead": int(runtime.last_item_retrieval_lookahead),
                    "last_modality_need": float(runtime.last_modality_need),
                    "last_selection_score": float(runtime.last_selection_score),
                    "last_window_budget": int(runtime.last_window_budget),
                }
                for runtime in self._sensory_source_runtimes
            ],
        }

    def _huggingface_runtime_summary_locked(self) -> dict[str, Any]:
        hf_runtimes = [
            runtime
            for runtime in self._brain_source_runtimes
            if runtime.source_type == "hf"
        ]
        return {
            "token_configured": bool(huggingface_token_from_env()),
            "background_source_count": sum(
                1
                for spec in self._brain_config.get("source_bank", [])
                if str(spec.get("source_type", "auto")) == "hf"
            ),
            "sensory_source_count": len(self._sensory_source_runtimes),
            "source_count": sum(
                1
                for spec in self._brain_config.get("source_bank", [])
                if str(spec.get("source_type", "auto")) == "hf"
            ) + len(self._sensory_source_runtimes),
            "buffered_tokens": int(sum(len(runtime.buffered_patterns) for runtime in hf_runtimes)),
            "prefetch_events": int(sum(runtime.prefetch_events for runtime in hf_runtimes)),
            "prefetched_tokens": int(sum(runtime.prefetched_tokens for runtime in hf_runtimes)),
        }

    def _terminus_status_snapshot_locked(self) -> dict[str, Any]:
        return {
            "terminus_runtime": self._brain_runtime_snapshot_locked(),
            "dirty_state": bool(self._dirty_state),
            "state_revision": int(self._state_revision),
            "token_count": int(self._trainer.token_count),
            "multimodal": self._multimodal_runtime_summary_locked(),
        }

    def terminus_status(self, *, fresh_wait_seconds: float | None = None) -> dict[str, Any]:
        # Non-blocking by default for operator/UI polling, but long-running
        # diagnostics can ask for a fresh snapshot instead of stale cached data.
        if fresh_wait_seconds is None:
            acquired = self._lock.acquire(timeout=0.15)
            if not acquired:
                cached = getattr(self, "_cached_terminus_status", None)
                if cached is not None:
                    return cached
                self._lock.acquire()
        else:
            deadline = time.perf_counter() + max(0.0, float(fresh_wait_seconds))
            acquired = False
            while time.perf_counter() < deadline:
                remaining = max(0.0, deadline - time.perf_counter())
                if self._lock.acquire(timeout=min(0.15, remaining)):
                    acquired = True
                    break
            if not acquired:
                self._lock.acquire()
        try:
            result = self._terminus_status_snapshot_locked()
            self._cached_terminus_status = result
            return result
        finally:
            self._lock.release()

    @staticmethod
    def _source_spec_uses_live_remote(spec: Mapping[str, Any]) -> bool:
        source_type = str(spec.get("source_type", "auto") or "auto").strip().lower()
        if source_type in {"hf", "web"}:
            return True
        if source_type == "file":
            return False
        source = str(spec.get("source", "") or "").strip()
        if source.startswith(("http://", "https://")):
            return True
        return not Path(source).exists()

    @staticmethod
    def _sensory_spec_uses_live_remote(spec: Mapping[str, Any]) -> bool:
        adapter = str(spec.get("adapter", "") or "").strip().lower()
        if adapter in {"s1_mmalign", "audiocaps"}:
            return True
        source = str(spec.get("source", "") or "").strip()
        return bool(source) and not Path(source).exists()

    def _request_active_execution_locked(self) -> None:
        self._active_execution_requests += 1
        if self._active_execution_requests > 0:
            self._active_execution_idle_event.clear()

    def _release_active_execution_locked(self) -> None:
        self._active_execution_requests = max(0, int(self._active_execution_requests) - 1)
        if self._active_execution_requests <= 0:
            self._active_execution_idle_event.set()

    def _request_active_execution(self) -> None:
        with self._lock:
            self._request_active_execution_locked()

    def _release_active_execution(self) -> None:
        with self._lock:
            self._release_active_execution_locked()

    def _wait_for_remote_prewarm_clearance(
        self,
        stop_event: Event | None,
        *,
        remote_text_target: bool,
        remote_sensory_target: bool,
    ) -> bool:
        if not (remote_text_target or remote_sensory_target):
            return True

        grace_seconds = max(0.0, float(DEFAULT_REMOTE_PREWARM_GRACE_SECONDS))
        if grace_seconds > 0.0:
            deadline = time.perf_counter() + grace_seconds
            while True:
                if stop_event is not None and stop_event.is_set():
                    return False
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                time.sleep(min(float(DEFAULT_REMOTE_PREWARM_POLL_SECONDS), remaining))

        wait_started_perf: float | None = None
        wait_started_at: str | None = None
        while True:
            if stop_event is not None and stop_event.is_set():
                return False
            with self._lock:
                active_requested = bool(self._active_execution_requests > 0)
                trigger = self._ingestion_prewarm_last_trigger
            if not active_requested:
                if wait_started_perf is not None:
                    waited_ms = float((time.perf_counter() - wait_started_perf) * 1000.0)
                    with self._lock:
                        self._record_brain_event_locked(
                            {
                                "type": "ingestion_prewarm_active_execution_cleared",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "trigger": trigger,
                                "wait_started_at": wait_started_at,
                                "wait_duration_ms": waited_ms,
                            }
                        )
                return True
            if wait_started_perf is None:
                wait_started_perf = time.perf_counter()
                wait_started_at = datetime.now(timezone.utc).isoformat()
                with self._lock:
                    self._record_brain_event_locked(
                        {
                            "type": "ingestion_prewarm_waiting_for_active_execution",
                            "timestamp": wait_started_at,
                            "trigger": trigger,
                            "remote_text_prewarm": bool(remote_text_target),
                            "remote_sensory_prewarm": bool(remote_sensory_target),
                            "grace_seconds": float(grace_seconds),
                        }
                    )
            self._active_execution_idle_event.wait(timeout=float(DEFAULT_REMOTE_PREWARM_POLL_SECONDS))

    def _remote_warm_promotion_text_needed_locked(self) -> bool:
        ingestion = self._brain_config.get("ingestion") or {}
        if not bool(ingestion.get("enabled", True)) or bool(ingestion.get("prewarm_on_startup", False)):
            return False
        target_tokens = max(1, int(ingestion.get("queue_target_tokens", self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS))))
        return any(
            self._source_spec_uses_live_remote(runtime.spec)
            and self._stream_supports_ready_reads(runtime.stream)
            and len(runtime.buffered_patterns) < target_tokens
            and not runtime.exhausted
            for runtime in self._brain_source_runtimes
        )

    def _remote_warm_promotion_sensory_needed_locked(self) -> bool:
        sensory = self._brain_config.get("sensory") or {}
        if not bool(sensory.get("enabled", False)) or bool(sensory.get("prewarm_on_startup", False)):
            return False
        target_items = self._sensory_queue_target_items_locked()
        return any(
            self._sensory_spec_uses_live_remote(runtime.spec)
            and self._stream_supports_ready_reads(runtime.stream)
            and len(runtime.buffered_episodes) < target_items
            and not runtime.exhausted
            for runtime in self._sensory_source_runtimes
        )

    def _request_remote_warm_promotion_stop(self) -> Thread | None:
        with self._lock:
            thread = (
                self._remote_warm_promotion_thread
                if self._remote_warm_promotion_thread is not None and self._remote_warm_promotion_thread.is_alive()
                else None
            )
            stop_event = self._remote_warm_promotion_stop_event
            if stop_event is not None:
                stop_event.set()
            self._remote_warm_promotion_running = False
            return thread

    def _join_remote_warm_promotion_thread(self, thread: Thread | None, *, timeout: float = 5.0) -> bool:
        if thread is None:
            with self._lock:
                if self._remote_warm_promotion_thread is not None and not self._remote_warm_promotion_thread.is_alive():
                    self._remote_warm_promotion_thread = None
                    self._remote_warm_promotion_stop_event = None
            return True
        thread.join(timeout=timeout)
        with self._lock:
            if self._remote_warm_promotion_thread is thread and not thread.is_alive():
                self._remote_warm_promotion_thread = None
                self._remote_warm_promotion_stop_event = None
        return not thread.is_alive()

    def _record_remote_warm_promotion_completed_locked(self) -> None:
        last_event = self._brain_last_event if isinstance(self._brain_last_event, Mapping) else {}
        if str(last_event.get("type", "")) == "remote_warm_promotion_completed":
            return
        self._record_brain_event_locked(
            {
                "type": "remote_warm_promotion_completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trigger": self._remote_warm_promotion_last_trigger,
                "ready_source_count": int(self._ingestion_ready_source_count_locked()),
                "sensory_ready_source_count": int(self._sensory_ready_source_count_locked()),
            }
        )

    def _start_remote_warm_promotion_locked(self, *, trigger: str) -> bool:
        text_needed = self._remote_warm_promotion_text_needed_locked()
        sensory_needed = self._remote_warm_promotion_sensory_needed_locked()
        if not (text_needed or sensory_needed):
            return False
        thread = self._remote_warm_promotion_thread
        if thread is not None and thread.is_alive():
            return False
        self._remote_warm_promotion_stop_event = Event()
        self._remote_warm_promotion_running = True
        self._remote_warm_promotion_last_trigger = trigger
        self._record_brain_event_locked(
            {
                "type": "remote_warm_promotion_started",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trigger": trigger,
                "text_sources": int(
                    sum(
                        1
                        for runtime in self._brain_source_runtimes
                        if self._source_spec_uses_live_remote(runtime.spec) and not runtime.exhausted
                    )
                ),
                "sensory_sources": int(
                    sum(
                        1
                        for runtime in self._sensory_source_runtimes
                        if self._sensory_spec_uses_live_remote(runtime.spec) and not runtime.exhausted
                    )
                ),
            }
        )
        thread = Thread(target=self._remote_warm_promotion_loop, name="hecsn-remote-warm-promotion", daemon=True)
        self._remote_warm_promotion_thread = thread
        thread.start()
        return True

    @staticmethod
    def _remaining_budget_seconds(deadline_perf: float | None) -> float | None:
        if deadline_perf is None:
            return None
        return max(0.0, float(deadline_perf - time.perf_counter()))

    @staticmethod
    def _run_budgeted_call(func: Any, /, *args: Any, wait_seconds: float | None = None, **kwargs: Any) -> tuple[bool, Any]:
        if wait_seconds is None:
            return True, func(*args, **kwargs)
        budget = max(0.0, float(wait_seconds))
        if budget <= 0.0:
            return False, None
        payloads: Queue[object] = Queue(maxsize=1)

        def _runner() -> None:
            try:
                payload: object = func(*args, **kwargs)
            except BaseException as exc:  # pragma: no cover - background guard
                payload = _TimedCallFailure(exc)
            try:
                payloads.put_nowait(payload)
            except Exception:
                pass

        Thread(target=_runner, name="hecsn-budgeted-call", daemon=True).start()
        try:
            payload = payloads.get(timeout=budget)
        except Empty:
            return False, None
        if isinstance(payload, _TimedCallFailure):
            raise payload.error
        return True, payload

    def _remote_text_bootstrap_candidates_locked(self) -> list[tuple[_BrainSourceRuntime, dict[str, Any], int]]:
        ingestion = self._brain_config.get("ingestion") or {}
        target_tokens = max(1, int(ingestion.get("queue_target_tokens", self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS))))
        candidates: list[tuple[_BrainSourceRuntime, dict[str, Any], int]] = []
        for runtime in self._brain_source_runtimes:
            if (
                runtime.bootstrap_attempted
                or not self._source_spec_uses_live_remote(runtime.spec)
                or str(runtime.spec.get("source_type", "auto")).strip().lower() != "hf"
                or len(runtime.buffered_patterns) > 0
                or runtime.exhausted
            ):
                continue
            runtime.bootstrap_attempted = True
            candidates.append((runtime, deepcopy(runtime.spec), int(target_tokens)))
        return candidates

    def _fetch_remote_text_bootstrap_rows(
        self,
        spec: Mapping[str, Any],
        *,
        deadline_perf: float | None = None,
    ) -> tuple[list[str], bool]:
        remaining = self._remaining_budget_seconds(deadline_perf)
        if remaining is not None and remaining <= 0.0:
            return [], True
        call_budget = float(DEFAULT_REMOTE_BOOTSTRAP_BUDGET_SECONDS if remaining is None else min(DEFAULT_REMOTE_BOOTSTRAP_BUDGET_SECONDS, remaining))
        try:
            completed, rows = self._run_budgeted_call(
                load_hf_first_rows,
                str(spec.get("source", "")),
                wait_seconds=call_budget,
                hf_config=cast(str | None, spec.get("hf_config")),
                split="train",
                columns=[str(spec.get("text_field", "text") or "text")],
                max_rows=DEFAULT_REMOTE_BOOTSTRAP_ROWS,
                timeout_seconds=call_budget,
            )
        except Exception:
            return [], False
        if not completed:
            return [], True
        text_field = str(spec.get("text_field", "text") or "text")
        texts: list[str] = []
        for row in list(rows or []):
            if not isinstance(row, Mapping):
                continue
            text = str(row.get(text_field, "")).strip()
            if text:
                texts.append(text)
        return texts, False

    def _apply_remote_text_bootstrap_locked(
        self,
        runtime: _BrainSourceRuntime,
        texts: Sequence[str],
        *,
        target_tokens: int,
    ) -> int:
        if len(runtime.buffered_patterns) > 0 or runtime.exhausted:
            return 0
        examples: list[tuple[str, torch.Tensor]] = []
        for text in texts:
            for raw_window, pattern in labeled_pattern_stream(
                text,
                self._encoder,
                self._trainer.config.window_size,
                learn_chunking=True,
            ):
                examples.append((raw_window, pattern))
                if len(examples) >= target_tokens:
                    break
            if len(examples) >= target_tokens:
                break
        if not examples:
            return 0
        runtime.buffered_patterns.extend(examples)
        now = datetime.now(timezone.utc).isoformat()
        self._commit_collected_runtime_locked(
            {
                "runtime": runtime,
                "cycles": runtime.cycles_completed,
                "exhausted": runtime.exhausted,
                "new_stream": None,
                "served_tokens": 0,
                "queue_hit": False,
                "prefetch_tokens": int(len(examples)),
                "prefetch_duration_ms": 0.0,
                "prefetch_at": now,
                "prefetch_error": None,
                "warm_trigger": "remote_bootstrap",
            }
        )
        self._update_brain_runtime_cache_locked(runtime)
        self._record_brain_event_locked(
            {
                "type": "remote_text_bootstrap_applied",
                "timestamp": now,
                "source_name": runtime.name,
                "token_count": int(len(examples)),
            }
        )
        self._mark_mutated()
        return int(len(examples))

    def _remote_sensory_bootstrap_candidates_locked(self) -> list[tuple[_SensorySourceRuntime, dict[str, Any], int, int, torch.device]]:
        target_items = self._sensory_queue_target_items_locked()
        visual_dim = int(getattr(self._trainer.config, "cross_modal_dim_visual", 64))
        audio_dim = int(getattr(self._trainer.config, "cross_modal_dim_audio", 64))
        device = self._trainer.model.device
        candidates: list[tuple[_SensorySourceRuntime, dict[str, Any], int, int, torch.device]] = []
        for runtime in self._sensory_source_runtimes:
            if (
                runtime.bootstrap_attempted
                or not self._sensory_spec_uses_live_remote(runtime.spec)
                or len(runtime.buffered_episodes) > 0
                or runtime.exhausted
            ):
                continue
            runtime.bootstrap_attempted = True
            candidates.append((runtime, deepcopy(runtime.spec), int(visual_dim), int(audio_dim), device))
        return candidates

    def _fetch_remote_sensory_bootstrap_episodes(
        self,
        spec: Mapping[str, Any],
        *,
        visual_dim: int,
        audio_dim: int,
        device: torch.device,
        deadline_perf: float | None = None,
    ) -> tuple[list[SensoryEpisode], bool]:
        remaining = self._remaining_budget_seconds(deadline_perf)
        if remaining is not None and remaining <= 0.0:
            return [], True
        call_budget = float(DEFAULT_REMOTE_BOOTSTRAP_BUDGET_SECONDS if remaining is None else min(DEFAULT_REMOTE_BOOTSTRAP_BUDGET_SECONDS, remaining))
        try:
            completed, rows = self._run_budgeted_call(
                load_hf_first_rows,
                str(spec.get("source", "")),
                wait_seconds=call_budget,
                hf_config=cast(str | None, spec.get("hf_config")) or "default",
                split=str(spec.get("split", "train") or "train"),
                columns=sensory_bootstrap_columns(spec),
                max_rows=DEFAULT_REMOTE_BOOTSTRAP_ROWS,
                timeout_seconds=call_budget,
            )
        except Exception:
            return [], False
        if not completed:
            return [], True
        episodes: list[SensoryEpisode] = []
        for row in list(rows or []):
            if not isinstance(row, Mapping):
                continue
            remaining = self._remaining_budget_seconds(deadline_perf)
            if remaining is not None and remaining <= 0.0:
                return episodes, True
            build_budget = float(DEFAULT_REMOTE_BOOTSTRAP_BUDGET_SECONDS if remaining is None else min(DEFAULT_REMOTE_BOOTSTRAP_BUDGET_SECONDS, remaining))
            try:
                completed, episode = self._run_budgeted_call(
                    bootstrap_sensory_episode_from_row,
                    spec,
                    row,
                    wait_seconds=build_budget,
                    visual_dim=visual_dim,
                    audio_dim=audio_dim,
                    device=device,
                    timeout_seconds=build_budget,
                )
            except Exception:
                continue
            if not completed:
                return episodes, True
            if episode is not None:
                episodes.append(episode)
        return episodes, False

    def _apply_remote_sensory_bootstrap_locked(
        self,
        runtime: _SensorySourceRuntime,
        episodes: Sequence[SensoryEpisode],
        *,
        target_items: int,
    ) -> int:
        if len(runtime.buffered_episodes) > 0 or runtime.exhausted:
            return 0
        applied = list(episodes[: max(1, int(target_items))])
        if not applied:
            return 0
        runtime.buffered_episodes.extend(applied)
        now = datetime.now(timezone.utc).isoformat()
        self._commit_prefetched_sensory_runtime_locked(
            {
                "runtime": runtime,
                "cycles": runtime.cycles_completed,
                "exhausted": runtime.exhausted,
                "new_stream": None,
                "served_items": 0,
                "queue_hit": False,
                "prefetch_items": int(len(applied)),
                "prefetch_duration_ms": 0.0,
                "prefetch_at": now,
                "prefetch_error": None,
                "warm_trigger": "remote_bootstrap",
            }
        )
        self._update_sensory_runtime_cache_locked(runtime)
        self._record_brain_event_locked(
            {
                "type": "remote_sensory_bootstrap_applied",
                "timestamp": now,
                "source_name": runtime.name,
                "item_count": int(len(applied)),
            }
        )
        self._mark_mutated()
        return int(len(applied))

    def _promote_ready_remote_brain_items_locked(self) -> int:
        ingestion = self._brain_config.get("ingestion") or {}
        target_tokens = max(1, int(ingestion.get("queue_target_tokens", self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS))))
        repeat = bool(self._brain_config.get("repeat_sources", True))
        promoted_total = 0
        for runtime in self._brain_source_runtimes:
            if (
                not self._source_spec_uses_live_remote(runtime.spec)
                or not self._stream_supports_ready_reads(runtime.stream)
                or len(runtime.buffered_patterns) >= target_tokens
                or runtime.exhausted
            ):
                continue
            cycles = runtime.cycles_completed
            exhausted = runtime.exhausted
            new_stream = None
            prefetch_error: str | None = None
            promoted = 0
            started = time.perf_counter()
            while len(runtime.buffered_patterns) < target_tokens and not exhausted:
                try:
                    runtime.buffered_patterns.append(self._next_stream_item(runtime.stream, timeout=0.0))
                    promoted += 1
                except TimeoutError:
                    break
                except StopIteration:
                    if repeat:
                        cycles += 1
                        rebuilt = self._build_source_stream_from_spec(runtime.spec, self._encoder, self._trainer.config.window_size)
                        runtime.stream = rebuilt
                        new_stream = rebuilt
                        exhausted = False
                        try:
                            runtime.buffered_patterns.append(self._next_stream_item(runtime.stream, timeout=0.0))
                            promoted += 1
                        except TimeoutError:
                            break
                        except StopIteration:
                            exhausted = True
                            break
                    else:
                        exhausted = True
                        break
                except Exception as exc:
                    prefetch_error = str(exc)
                    break
            if promoted > 0 or new_stream is not None or exhausted != runtime.exhausted or prefetch_error is not None:
                duration_ms = float((time.perf_counter() - started) * 1000.0)
                self._commit_collected_runtime_locked(
                    {
                        "runtime": runtime,
                        "cycles": cycles,
                        "exhausted": exhausted,
                        "new_stream": new_stream,
                        "served_tokens": 0,
                        "queue_hit": False,
                        "prefetch_tokens": int(promoted),
                        "prefetch_duration_ms": duration_ms if promoted > 0 or prefetch_error is not None else None,
                        "prefetch_at": datetime.now(timezone.utc).isoformat() if promoted > 0 or prefetch_error is not None else None,
                        "prefetch_error": prefetch_error,
                        "warm_trigger": "remote_promotion",
                    }
                )
                self._mark_mutated()
            promoted_total += promoted
        return promoted_total

    def _promote_ready_remote_sensory_items_locked(self) -> int:
        target_items = self._sensory_queue_target_items_locked()
        repeat_sources = bool((self._brain_config.get("sensory") or {}).get("repeat_sources", True))
        promoted_total = 0
        for runtime in self._sensory_source_runtimes:
            if (
                not self._sensory_spec_uses_live_remote(runtime.spec)
                or not self._stream_supports_ready_reads(runtime.stream)
                or len(runtime.buffered_episodes) >= target_items
                or runtime.exhausted
            ):
                continue
            cycles = runtime.cycles_completed
            exhausted = runtime.exhausted
            new_stream = None
            prefetch_error: str | None = None
            promoted = 0
            started = time.perf_counter()
            while len(runtime.buffered_episodes) < target_items and not exhausted:
                try:
                    runtime.buffered_episodes.append(self._next_stream_item(runtime.stream, timeout=0.0))
                    promoted += 1
                except TimeoutError:
                    break
                except StopIteration:
                    if repeat_sources:
                        cycles += 1
                        rebuilt = self._build_sensory_stream_from_spec(
                            runtime.spec,
                            visual_dim=int(getattr(self._trainer.config, "cross_modal_dim_visual", 64)),
                            audio_dim=int(getattr(self._trainer.config, "cross_modal_dim_audio", 64)),
                            device=self._trainer.model.device,
                        )
                        runtime.stream = rebuilt
                        new_stream = rebuilt
                        exhausted = False
                        try:
                            runtime.buffered_episodes.append(self._next_stream_item(runtime.stream, timeout=0.0))
                            promoted += 1
                        except TimeoutError:
                            break
                        except StopIteration:
                            exhausted = True
                            break
                    else:
                        exhausted = True
                        break
                except Exception as exc:
                    prefetch_error = str(exc)
                    break
            if promoted > 0 or new_stream is not None or exhausted != runtime.exhausted or prefetch_error is not None:
                duration_ms = float((time.perf_counter() - started) * 1000.0)
                self._commit_prefetched_sensory_runtime_locked(
                    {
                        "runtime": runtime,
                        "cycles": cycles,
                        "exhausted": exhausted,
                        "new_stream": new_stream,
                        "served_items": 0,
                        "queue_hit": False,
                        "prefetch_items": int(promoted),
                        "prefetch_duration_ms": duration_ms if promoted > 0 or prefetch_error is not None else None,
                        "prefetch_at": datetime.now(timezone.utc).isoformat() if promoted > 0 or prefetch_error is not None else None,
                        "prefetch_error": prefetch_error,
                        "warm_trigger": "remote_promotion",
                    }
                )
                self._mark_mutated()
            promoted_total += promoted
        return promoted_total

    def _remote_warm_promotion_loop(self) -> None:
        while True:
            stop_requested = False
            completed = False
            promoted_text = 0
            promoted_sensory = 0
            initial_ready_text = 0
            initial_ready_sensory = 0
            text_bootstrap_candidates: list[tuple[_BrainSourceRuntime, dict[str, Any], int]] = []
            sensory_bootstrap_candidates: list[tuple[_SensorySourceRuntime, dict[str, Any], int, int, torch.device]] = []
            with self._lock:
                stop_event = self._remote_warm_promotion_stop_event
                stop_requested = bool(stop_event is not None and stop_event.is_set())
                if not stop_requested:
                    initial_ready_text = self._promote_ready_remote_brain_items_locked()
                    initial_ready_sensory = self._promote_ready_remote_sensory_items_locked()
                    if not self._remote_warm_promotion_text_needed_locked() and not self._remote_warm_promotion_sensory_needed_locked():
                        self._record_remote_warm_promotion_completed_locked()
                        self._remote_warm_promotion_running = False
                        self._remote_warm_promotion_thread = None
                        self._remote_warm_promotion_stop_event = None
                        return

            wait_deadline = None
            if not stop_requested and initial_ready_text <= 0 and initial_ready_sensory <= 0:
                wait_deadline = time.perf_counter() + float(
                    min(DEFAULT_REMOTE_PROMOTION_BOOTSTRAP_GRACE_SECONDS, DEFAULT_REMOTE_BOOTSTRAP_BUDGET_SECONDS)
                )
            while not stop_requested and wait_deadline is not None and time.perf_counter() < wait_deadline:
                time.sleep(float(DEFAULT_REMOTE_PREWARM_POLL_SECONDS))
                with self._lock:
                    stop_event = self._remote_warm_promotion_stop_event
                    stop_requested = bool(stop_event is not None and stop_event.is_set())
                    if stop_requested:
                        break
                    initial_ready_text += self._promote_ready_remote_brain_items_locked()
                    initial_ready_sensory += self._promote_ready_remote_sensory_items_locked()
                    if initial_ready_text > 0 or initial_ready_sensory > 0:
                        break
                    if not self._remote_warm_promotion_text_needed_locked() and not self._remote_warm_promotion_sensory_needed_locked():
                        self._record_remote_warm_promotion_completed_locked()
                        self._remote_warm_promotion_running = False
                        self._remote_warm_promotion_thread = None
                        self._remote_warm_promotion_stop_event = None
                        return

            with self._lock:
                stop_event = self._remote_warm_promotion_stop_event
                stop_requested = stop_requested or bool(stop_event is not None and stop_event.is_set())
                if not stop_requested:
                    initial_ready_text += self._promote_ready_remote_brain_items_locked()
                    initial_ready_sensory += self._promote_ready_remote_sensory_items_locked()
                    if not self._remote_warm_promotion_text_needed_locked() and not self._remote_warm_promotion_sensory_needed_locked():
                        self._record_remote_warm_promotion_completed_locked()
                        self._remote_warm_promotion_running = False
                        self._remote_warm_promotion_thread = None
                        self._remote_warm_promotion_stop_event = None
                        return
                    text_bootstrap_candidates = self._remote_text_bootstrap_candidates_locked()
                    sensory_bootstrap_candidates = self._remote_sensory_bootstrap_candidates_locked()

            text_bootstrap_promoted = 0
            for runtime, spec, target_tokens in text_bootstrap_candidates:
                deadline_perf = time.perf_counter() + float(DEFAULT_REMOTE_BOOTSTRAP_BUDGET_SECONDS)
                texts, bootstrap_timed_out = self._fetch_remote_text_bootstrap_rows(spec, deadline_perf=deadline_perf)
                if stop_requested:
                    break
                with self._lock:
                    current_stop_event = self._remote_warm_promotion_stop_event
                    if current_stop_event is not None and current_stop_event.is_set():
                        stop_requested = True
                        break
                    if any(current is runtime for current in self._brain_source_runtimes):
                        if bootstrap_timed_out:
                            self._record_brain_event_locked(
                                {
                                    "type": "remote_text_bootstrap_timed_out",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "source_name": runtime.name,
                                    "budget_seconds": float(DEFAULT_REMOTE_BOOTSTRAP_BUDGET_SECONDS),
                                }
                            )
                        text_bootstrap_promoted += self._apply_remote_text_bootstrap_locked(
                            runtime,
                            texts,
                            target_tokens=target_tokens,
                        )

            sensory_bootstrap_promoted = 0
            for runtime, spec, visual_dim, audio_dim, device in sensory_bootstrap_candidates:
                deadline_perf = time.perf_counter() + float(DEFAULT_REMOTE_BOOTSTRAP_BUDGET_SECONDS)
                episodes, bootstrap_timed_out = self._fetch_remote_sensory_bootstrap_episodes(
                    spec,
                    visual_dim=visual_dim,
                    audio_dim=audio_dim,
                    device=device,
                    deadline_perf=deadline_perf,
                )
                if stop_requested:
                    break
                with self._lock:
                    current_stop_event = self._remote_warm_promotion_stop_event
                    if current_stop_event is not None and current_stop_event.is_set():
                        stop_requested = True
                        break
                    if any(current is runtime for current in self._sensory_source_runtimes):
                        if bootstrap_timed_out:
                            self._record_brain_event_locked(
                                {
                                    "type": "remote_sensory_bootstrap_timed_out",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "source_name": runtime.name,
                                    "budget_seconds": float(DEFAULT_REMOTE_BOOTSTRAP_BUDGET_SECONDS),
                                    "item_count": int(len(episodes)),
                                }
                            )
                        sensory_bootstrap_promoted += self._apply_remote_sensory_bootstrap_locked(
                            runtime,
                            episodes,
                            target_items=self._sensory_queue_target_items_locked(),
                        )

            with self._lock:
                stop_event = self._remote_warm_promotion_stop_event
                stop_requested = stop_requested or bool(stop_event is not None and stop_event.is_set())
                if not stop_requested:
                    promoted_text = initial_ready_text + self._promote_ready_remote_brain_items_locked() + text_bootstrap_promoted
                    promoted_sensory = initial_ready_sensory + self._promote_ready_remote_sensory_items_locked() + sensory_bootstrap_promoted
                    completed = not self._remote_warm_promotion_text_needed_locked() and not self._remote_warm_promotion_sensory_needed_locked()
                    if completed:
                        self._record_remote_warm_promotion_completed_locked()
                if stop_requested or completed:
                    self._remote_warm_promotion_running = False
                    self._remote_warm_promotion_thread = None
                    self._remote_warm_promotion_stop_event = None
                    return
            if promoted_text <= 0 and promoted_sensory <= 0:
                time.sleep(float(DEFAULT_REMOTE_PREWARM_POLL_SECONDS))

    def _request_ingestion_prewarm_stop(self) -> Thread | None:
        with self._lock:
            thread = self._ingestion_prewarm_thread if self._ingestion_prewarm_thread is not None and self._ingestion_prewarm_thread.is_alive() else None
            stop_event = self._ingestion_prewarm_stop_event
            if stop_event is not None:
                stop_event.set()
            self._ingestion_prewarm_running = False
            return thread

    def _join_ingestion_prewarm_thread(self, thread: Thread | None, *, timeout: float = 5.0) -> bool:
        if thread is None:
            with self._lock:
                if self._ingestion_prewarm_thread is not None and not self._ingestion_prewarm_thread.is_alive():
                    self._ingestion_prewarm_thread = None
                    self._ingestion_prewarm_stop_event = None
            return True
        thread.join(timeout=timeout)
        with self._lock:
            if self._ingestion_prewarm_thread is thread and not thread.is_alive():
                self._ingestion_prewarm_thread = None
                self._ingestion_prewarm_stop_event = None
        return not thread.is_alive()

    def _start_ingestion_prewarm_locked(self, *, trigger: str) -> bool:
        ingestion = self._brain_config.get("ingestion") or {}
        sensory = self._brain_config.get("sensory") or {}
        text_target = (
            bool(self._brain_config.get("source_bank"))
            and bool(ingestion.get("enabled", True))
            and bool(ingestion.get("prewarm_on_startup", False))
            and self._ingestion_warm_ready_at is None
        )
        sensory_target = (
            bool(sensory)
            and bool(sensory.get("enabled", False))
            and bool(sensory.get("source_bank"))
            and bool(sensory.get("prewarm_on_startup", False))
            and self._sensory_warm_ready_at is None
        )
        if not (text_target or sensory_target):
            return False
        thread = self._ingestion_prewarm_thread
        if thread is not None and thread.is_alive():
            return False
        self._ingestion_prewarm_stop_event = Event()
        self._ingestion_prewarm_running = True
        self._ingestion_prewarm_started_at = datetime.now(timezone.utc).isoformat()
        self._ingestion_prewarm_started_perf = time.perf_counter()
        self._ingestion_prewarm_completed_at = None
        self._ingestion_prewarm_last_duration_ms = None
        self._ingestion_prewarm_last_error = None
        self._ingestion_prewarm_last_trigger = trigger
        self._ingestion_prewarm_budget_exhausted = False
        self._sensory_prewarm_budget_exhausted = False
        self._ingestion_prewarm_run_count += 1
        self._record_brain_event_locked(
            {
                "type": "ingestion_prewarm_started",
                "timestamp": self._ingestion_prewarm_started_at,
                "trigger": trigger,
                "text_prewarm": bool(text_target),
                "sensory_prewarm": bool(sensory_target),
                "queue_target_tokens": int(ingestion.get("queue_target_tokens", DEFAULT_BRAIN_TICK_TOKENS)),
                "prewarm_max_seconds": float(ingestion.get("prewarm_max_seconds", 5.0)),
                "sensory_queue_target_items": int(self._sensory_queue_target_items_locked()) if sensory_target else 0,
                "sensory_prewarm_max_seconds": float(sensory.get("prewarm_max_seconds", 5.0)) if sensory_target else 0.0,
            }
        )
        thread = Thread(target=self._ingestion_prewarm_loop, name="hecsn-ingestion-prewarm", daemon=True)
        self._ingestion_prewarm_thread = thread
        thread.start()
        return True

    def _apply_detached_brain_prewarm_locked(
        self,
        detached_runtimes: Sequence[_BrainSourceRuntime],
        prefetched: Sequence[dict[str, Any]],
        *,
        expected_epoch: int,
    ) -> bool:
        if expected_epoch != self._brain_stream_epoch:
            return False
        if len(detached_runtimes) > len(self._brain_source_runtimes):
            return False
        for idx, detached in enumerate(detached_runtimes[: len(prefetched)]):
            active = self._brain_source_runtimes[idx]
            active.buffered_patterns = deque(detached.buffered_patterns)
            self._commit_collected_runtime_locked(
                {
                    **dict(prefetched[idx]),
                    "runtime": active,
                    "cycles": detached.cycles_completed,
                    "exhausted": detached.exhausted,
                    "new_stream": detached.stream,
                    "served_tokens": 0,
                    "queue_hit": False,
                }
            )
        return True

    def _apply_detached_sensory_prewarm_locked(
        self,
        detached_runtimes: Sequence[_SensorySourceRuntime],
        prefetched: Sequence[dict[str, Any]],
        *,
        expected_epoch: int,
    ) -> bool:
        if expected_epoch != self._sensory_stream_epoch:
            return False
        if len(detached_runtimes) > len(self._sensory_source_runtimes):
            return False
        for idx, detached in enumerate(detached_runtimes[: len(prefetched)]):
            active = self._sensory_source_runtimes[idx]
            active.buffered_episodes = list(detached.buffered_episodes)
            self._commit_prefetched_sensory_runtime_locked(
                {
                    **dict(prefetched[idx]),
                    "runtime": active,
                    "cycles": detached.cycles_completed,
                    "exhausted": detached.exhausted,
                    "new_stream": detached.stream,
                    "served_items": 0,
                    "queue_hit": False,
                }
            )
        return True

    def _ingestion_prewarm_loop(self) -> None:
        with self._lock:
            stop_event = self._ingestion_prewarm_stop_event
            brain_epoch = self._brain_stream_epoch
            sensory_epoch = self._sensory_stream_epoch
            brain_specs = [deepcopy(runtime.spec) for runtime in self._brain_source_runtimes]
            repeat = bool(self._brain_config.get("repeat_sources", True))
            tick_tokens = int(self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS))
            ingestion = self._brain_config.get("ingestion") or {}
            queue_target_tokens = int(ingestion.get("queue_target_tokens", tick_tokens))
            ingestion_budget_seconds = float(ingestion.get("prewarm_max_seconds", 5.0))
            encoder_ref = self._encoder
            window_size = self._trainer.config.window_size
            sensory = self._brain_config.get("sensory") or {}
            sensory_specs = [deepcopy(runtime.spec) for runtime in self._sensory_source_runtimes]
            sensory_repeat = bool(sensory.get("repeat_sources", True))
            sensory_queue_target_items = self._sensory_queue_target_items_locked()
            sensory_budget_seconds = float(sensory.get("prewarm_max_seconds", 5.0))
            visual_dim = int(getattr(self._trainer.config, "cross_modal_dim_visual", 64))
            audio_dim = int(getattr(self._trainer.config, "cross_modal_dim_audio", 64))
            device = self._trainer.model.device
            text_target = bool(ingestion.get("enabled", True)) and bool(ingestion.get("prewarm_on_startup", False))
            sensory_target = bool(sensory.get("enabled", False)) and bool(sensory.get("prewarm_on_startup", False))
            remote_text_target = bool(text_target and any(self._source_spec_uses_live_remote(spec) for spec in brain_specs))
            remote_sensory_target = bool(sensory_target and any(self._sensory_spec_uses_live_remote(spec) for spec in sensory_specs))
            text_processed_at_start = int(sum(int(runtime.tokens_processed) for runtime in self._brain_source_runtimes))
            sensory_processed_at_start = int(sum(int(runtime.episodes_processed) for runtime in self._sensory_source_runtimes))
            text_ready_at_start = int(self._ingestion_ready_source_count_locked())
            text_full_at_start = int(self._ingestion_full_queue_source_count_locked())
            sensory_ready_at_start = int(self._sensory_ready_source_count_locked())
            sensory_full_at_start = int(self._sensory_full_queue_source_count_locked())
            if remote_text_target and (text_processed_at_start > 0 or text_ready_at_start > 0):
                self._record_brain_event_locked(
                    {
                        "type": "ingestion_prewarm_skipped_after_active_execution",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "trigger": self._ingestion_prewarm_last_trigger,
                        "reason": "active_runtime_already_progressed_before_prewarm_start",
                        "ready_source_count": text_ready_at_start,
                        "full_queue_source_count": text_full_at_start,
                    }
                )
                text_target = False
                remote_text_target = False
            if remote_sensory_target and (sensory_processed_at_start > 0 or sensory_ready_at_start > 0):
                self._record_brain_event_locked(
                    {
                        "type": "sensory_prewarm_skipped_after_active_execution",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "trigger": self._ingestion_prewarm_last_trigger,
                        "reason": "active_runtime_already_progressed_before_prewarm_start",
                        "ready_source_count": sensory_ready_at_start,
                        "full_queue_source_count": sensory_full_at_start,
                    }
                )
                sensory_target = False
                remote_sensory_target = False

        if not self._wait_for_remote_prewarm_clearance(
            stop_event,
            remote_text_target=remote_text_target,
            remote_sensory_target=remote_sensory_target,
        ):
            with self._lock:
                self._record_brain_event_locked(
                    {
                        "type": "ingestion_prewarm_discarded",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "trigger": self._ingestion_prewarm_last_trigger,
                        "reason": "stop_requested",
                    }
                )
            text_target = False
            sensory_target = False

        with self._lock:
            text_processed_now = int(sum(int(runtime.tokens_processed) for runtime in self._brain_source_runtimes))
            sensory_processed_now = int(sum(int(runtime.episodes_processed) for runtime in self._sensory_source_runtimes))
            text_ready_now = int(self._ingestion_ready_source_count_locked())
            text_full_now = int(self._ingestion_full_queue_source_count_locked())
            sensory_ready_now = int(self._sensory_ready_source_count_locked())
            sensory_full_now = int(self._sensory_full_queue_source_count_locked())
            text_progressed_after_start = bool(
                text_processed_now > text_processed_at_start
                or text_ready_now > text_ready_at_start
                or text_full_now > text_full_at_start
            )
            sensory_progressed_after_start = bool(
                sensory_processed_now > sensory_processed_at_start
                or sensory_ready_now > sensory_ready_at_start
                or sensory_full_now > sensory_full_at_start
            )
            if text_target and text_progressed_after_start:
                self._record_brain_event_locked(
                    {
                        "type": "ingestion_prewarm_skipped_after_active_execution",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "trigger": self._ingestion_prewarm_last_trigger,
                        "reason": "active_runtime_progressed_while_prewarm_waited",
                        "ready_source_count": text_ready_now,
                        "full_queue_source_count": text_full_now,
                    }
                )
                text_target = False
            if sensory_target and sensory_progressed_after_start:
                self._record_brain_event_locked(
                    {
                        "type": "sensory_prewarm_skipped_after_active_execution",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "trigger": self._ingestion_prewarm_last_trigger,
                        "reason": "active_runtime_progressed_while_prewarm_waited",
                        "ready_source_count": sensory_ready_now,
                        "full_queue_source_count": sensory_full_now,
                    }
                )
                sensory_target = False

        detached_brain_runtimes = [
            _BrainSourceRuntime(
                spec=spec,
                stream=self._build_source_stream_from_spec(spec, encoder_ref, window_size),
            )
            for spec in brain_specs
        ] if text_target else []
        detached_sensory_runtimes = [
            _SensorySourceRuntime(
                spec=spec,
                stream=self._build_sensory_stream_from_spec(
                    spec,
                    visual_dim=visual_dim,
                    audio_dim=audio_dim,
                    device=device,
                ),
            )
            for spec in sensory_specs
        ] if sensory_target else []

        prefetched: list[dict[str, Any]] = []
        sensory_prefetched: list[dict[str, Any]] = []
        error: str | None = None
        applied_brain = False
        applied_sensory = False
        try:
            if text_target:
                prefetched = self._prefetch_source_queues_unlocked(
                    detached_brain_runtimes,
                    queue_target_tokens,
                    repeat,
                    encoder_ref,
                    window_size,
                    stop_event,
                    warm_trigger="prewarm",
                    deadline_perf=(None if ingestion_budget_seconds <= 0.0 else time.perf_counter() + ingestion_budget_seconds),
                )
            if sensory_target:
                sensory_prefetched = self._prefetch_sensory_queues_unlocked(
                    detached_sensory_runtimes,
                    sensory_queue_target_items,
                    sensory_repeat,
                    visual_dim,
                    audio_dim,
                    device,
                    stop_event,
                    warm_trigger="prewarm",
                    deadline_perf=(None if sensory_budget_seconds <= 0.0 else time.perf_counter() + sensory_budget_seconds),
                )
            with self._lock:
                self._ingestion_prewarm_budget_exhausted = any(bool(meta.get("budget_exhausted", False)) for meta in prefetched)
                self._sensory_prewarm_budget_exhausted = any(bool(meta.get("budget_exhausted", False)) for meta in sensory_prefetched)
                if text_target and prefetched:
                    applied_brain = self._apply_detached_brain_prewarm_locked(
                        detached_brain_runtimes,
                        prefetched,
                        expected_epoch=brain_epoch,
                    )
                if sensory_target and sensory_prefetched:
                    applied_sensory = self._apply_detached_sensory_prewarm_locked(
                        detached_sensory_runtimes,
                        sensory_prefetched,
                        expected_epoch=sensory_epoch,
                    )
                if text_target and prefetched and not applied_brain:
                    self._record_brain_event_locked(
                        {
                            "type": "ingestion_prewarm_discarded",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "trigger": self._ingestion_prewarm_last_trigger,
                            "reason": "runtime_progressed",
                        }
                    )
                if sensory_target and sensory_prefetched and not applied_sensory:
                    self._record_brain_event_locked(
                        {
                            "type": "sensory_prewarm_discarded",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "trigger": self._ingestion_prewarm_last_trigger,
                            "reason": "runtime_progressed",
                        }
                    )
                if self._ingestion_prewarm_budget_exhausted:
                    self._record_brain_event_locked(
                        {
                            "type": "ingestion_prewarm_budget_exhausted",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "trigger": self._ingestion_prewarm_last_trigger,
                            "prewarm_max_seconds": ingestion_budget_seconds,
                            "ready_source_count": int(self._ingestion_ready_source_count_locked()),
                            "full_queue_source_count": int(self._ingestion_full_queue_source_count_locked()),
                        }
                    )
                if self._sensory_prewarm_budget_exhausted:
                    self._record_brain_event_locked(
                        {
                            "type": "sensory_prewarm_budget_exhausted",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "trigger": self._ingestion_prewarm_last_trigger,
                            "prewarm_max_seconds": sensory_budget_seconds,
                            "ready_source_count": int(self._sensory_ready_source_count_locked()),
                            "full_queue_source_count": int(self._sensory_full_queue_source_count_locked()),
                        }
                    )
        except Exception as exc:
            error = str(exc)

        with self._lock:
            completed_at = datetime.now(timezone.utc).isoformat()
            self._ingestion_prewarm_completed_at = completed_at
            if self._ingestion_prewarm_started_perf is not None:
                self._ingestion_prewarm_last_duration_ms = float(
                    (time.perf_counter() - self._ingestion_prewarm_started_perf) * 1000.0
                )
            if error is not None:
                self._ingestion_prewarm_last_error = error
                self._record_brain_event_locked(
                    {
                        "type": "ingestion_prewarm_error",
                        "timestamp": completed_at,
                        "trigger": self._ingestion_prewarm_last_trigger,
                        "message": error,
                    }
                )
            else:
                self._record_brain_event_locked(
                    {
                        "type": "ingestion_prewarm_completed",
                        "timestamp": completed_at,
                        "trigger": self._ingestion_prewarm_last_trigger,
                        "applied_text_results": bool(applied_brain),
                        "applied_sensory_results": bool(applied_sensory),
                        "prefetch_events": int(sum(runtime.prefetch_events for runtime in self._brain_source_runtimes)),
                        "ready_source_count": int(self._ingestion_ready_source_count_locked()),
                        "sensory_prefetch_events": int(sum(runtime.prefetch_events for runtime in self._sensory_source_runtimes)),
                        "sensory_ready_source_count": int(self._sensory_ready_source_count_locked()),
                        "startup_warm_latency_ms": self._ingestion_startup_warm_latency_ms,
                        "sensory_startup_warm_latency_ms": self._sensory_startup_warm_latency_ms,
                    }
                )
            self._ingestion_prewarm_running = False
            self._ingestion_prewarm_thread = None
            self._ingestion_prewarm_stop_event = None

    def configure_terminus(
        self,
        *,
        source_bank: list[dict[str, Any]],
        tick_tokens: int = DEFAULT_BRAIN_TICK_TOKENS,
        sleep_interval_seconds: float = DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS,
        repeat_sources: bool = True,
        autonomy: dict[str, Any] | None = None,
        sensory: dict[str, Any] | None = None,
        ingestion: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        thread = self._request_brain_stop()
        self._join_brain_thread(thread)
        prewarm_thread = self._request_ingestion_prewarm_stop()
        self._join_ingestion_prewarm_thread(prewarm_thread)
        promotion_thread = self._request_remote_warm_promotion_stop()
        self._join_remote_warm_promotion_thread(promotion_thread)
        with self._lock:
            self._brain_config = self._normalize_brain_config(
                {
                    "source_bank": source_bank,
                    "tick_tokens": tick_tokens,
                    "sleep_interval_seconds": sleep_interval_seconds,
                    "repeat_sources": repeat_sources,
                    "autonomy": autonomy,
                    "sensory": sensory,
                    "ingestion": ingestion,
                }
            )
            self._brain_source_utility = {}
            self._brain_last_error = None
            self._last_real_sensory_episode_time = 0.0
            self._last_real_sensory_episode_token_count = int(self._trainer.token_count)
            self._real_sensory_last_error = None
            self._record_brain_event_locked({
                "type": "configured",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_names": [str(item.get("name", "")) for item in self._brain_config.get("source_bank", [])],
            })
            self._brain_last_acquisition_summary = None
            self._brain_last_acquisition_token_count = int(self._trainer.token_count)
            self._rebuild_brain_sources_locked()
            self._start_ingestion_prewarm_locked(trigger="configure")
            self._start_remote_warm_promotion_locked(trigger="configure")
            self._mark_mutated()
            return {
                "terminus_runtime": self._brain_runtime_snapshot_locked(),
                "dirty_state": bool(self._dirty_state),
                "state_revision": int(self._state_revision),
                "token_count": int(self._trainer.token_count),
            }

    def _brain_runtime_active_locked(self) -> bool:
        thread = self._brain_thread
        if thread is None:
            return False
        if thread.is_alive():
            return True
        self._finalize_brain_stop_locked(thread)
        return False

    def _assert_manual_tick_allowed_locked(self) -> None:
        if self._brain_runtime_active_locked():
            raise ValueError(
                "Cannot tick Terminus manually while the background runtime is active. Stop the runtime first."
            )

    def start_terminus(self) -> dict[str, Any]:
        with self._lock:
            if not self._brain_config.get("source_bank"):
                raise ValueError("Terminus runtime has no configured source_bank")
            if self._brain_runtime_active_locked():
                return {
                    "terminus_runtime": self._brain_runtime_snapshot_locked(),
                    "dirty_state": bool(self._dirty_state),
                    "state_revision": int(self._state_revision),
                    "token_count": int(self._trainer.token_count),
                }
            self._brain_stop_event = Event()
            self._start_ingestion_prewarm_locked(trigger="start")
            self._start_remote_warm_promotion_locked(trigger="start")
            self._brain_thread = Thread(target=self._brain_loop, name="hecsn-brain-loop", daemon=True)
            self._brain_running = True
            self._brain_running_since = datetime.now(timezone.utc).isoformat()
            self._brain_last_error = None
            self._record_brain_event_locked({
                "type": "started",
                "timestamp": self._brain_running_since,
            })
            self._brain_thread.start()

            # Start cortex thought loop alongside brain
            if self._thought_loop is not None and not self._thought_loop.is_running:
                try:
                    self._thought_loop.start()
                    _cortex_logger.info("ThoughtLoop started alongside Terminus brain")
                except Exception as exc:
                    _cortex_logger.warning("ThoughtLoop failed to start: %s", exc)

            return {
                "terminus_runtime": self._brain_runtime_snapshot_locked(),
                "dirty_state": bool(self._dirty_state),
                "state_revision": int(self._state_revision),
                "token_count": int(self._trainer.token_count),
            }

    def stop_terminus(self) -> dict[str, Any]:
        # Signal ThoughtLoop stop under lock (safe), join outside (avoids deadlock)
        thought_loop = self._thought_loop
        if thought_loop is not None and thought_loop.is_running:
            thought_loop.request_stop()

        thread = self._request_brain_stop(reason="manual")
        self._join_brain_thread(thread)
        prewarm_thread = self._request_ingestion_prewarm_stop()
        self._join_ingestion_prewarm_thread(prewarm_thread)

        # Join ThoughtLoop thread outside all locks
        if thought_loop is not None:
            try:
                thought_loop.stop(timeout=3.0)
            except Exception:
                pass

        with self._lock:
            return {
                "terminus_runtime": self._brain_runtime_snapshot_locked(),
                "dirty_state": bool(self._dirty_state),
                "state_revision": int(self._state_revision),
                "token_count": int(self._trainer.token_count),
            }

    def quick_start_terminus(self, *, preset: str = "curriculum") -> dict[str, Any]:
        """Configure and start Terminus in one atomic call using a named preset.

        If the preset includes ``model_overrides`` that differ from the current
        model (e.g. different n_columns or binding_mode), the model is rebuilt
        from scratch with the new config before starting.
        """
        if preset not in TERMINUS_QUICK_START_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {', '.join(sorted(TERMINUS_QUICK_START_PRESETS))}")
        with self._lock:
            if self._brain_runtime_active_locked():
                return {
                    "terminus_runtime": self._brain_runtime_snapshot_locked(),
                    "dirty_state": bool(self._dirty_state),
                    "state_revision": int(self._state_revision),
                    "token_count": int(self._trainer.token_count),
                    "already_running": True,
                }
        config = TERMINUS_QUICK_START_PRESETS[preset]
        overrides = config.get("model_overrides")
        if overrides:
            current_cfg = self._trainer.config
            needs_rebuild = any(
                getattr(current_cfg, k, None) != v for k, v in overrides.items()
            )
            if needs_rebuild:
                cfg_dict = {
                    field_name: getattr(current_cfg, field_name)
                    for field_name, field_obj in current_cfg.__dataclass_fields__.items()
                    if field_obj.init
                }
                cfg_dict.update(overrides)
                new_cfg = HECSNConfig(**cfg_dict)
                new_model = HECSNModel(new_cfg)
                self._trainer = HECSNTrainer(new_model, new_cfg)
                self._encoder = self._trainer.encoder
        self.configure_terminus(
            source_bank=config["source_bank"],
            tick_tokens=config["tick_tokens"],
            sleep_interval_seconds=config["sleep_interval_seconds"],
            repeat_sources=config["repeat_sources"],
            autonomy=config.get("autonomy"),
            sensory=config.get("sensory"),
        )
        result = self.start_terminus()
        result["already_running"] = False
        result["preset_applied"] = preset
        return result

    @staticmethod
    def quick_start_presets() -> list[dict[str, Any]]:
        """Return available quick-start presets for the UI/API.

        The preset surface is intentionally narrow: only the current supported
        Terminus runtime path is exposed.
        """
        presets = [
            {
                "id": key,
                "label": val["label"],
                "description": val["description"],
                "source_count": len(val["source_bank"]),
                "default": bool(val.get("default", False)),
                "legacy": bool(val.get("legacy", False)),
            }
            for key, val in TERMINUS_QUICK_START_PRESETS.items()
        ]
        presets.sort(key=lambda item: (not item["default"], item["legacy"], item["label"]))
        return presets

    # --- Cortex / ThoughtLoop public interface ---

    @staticmethod
    def _normalize_cortex_query_hint(value: Any) -> str:
        return " ".join(str(value).split()).strip()

    def _remember_cortex_query_hint_locked(self, query_text: str) -> None:
        normalized = self._normalize_cortex_query_hint(query_text)
        if not normalized:
            self._last_cortex_query_hint_text = None
            self._last_cortex_query_hint_at = 0.0
            return
        self._last_cortex_query_hint_text = normalized
        self._last_cortex_query_hint_at = time.time()

    def _consume_cortex_query_hint_locked(self, *, max_age_seconds: float = 30.0) -> str:
        hint = self._normalize_cortex_query_hint(self._last_cortex_query_hint_text or "")
        age = time.time() - float(self._last_cortex_query_hint_at or 0.0)
        self._last_cortex_query_hint_text = None
        self._last_cortex_query_hint_at = 0.0
        if not hint or age > max(0.0, float(max_age_seconds)):
            return ""
        return hint

    def _request_cortex_sleep_locked(
        self,
        *,
        source: str,
        reason: str,
        query_text: str = "",
        thought_text: str = "",
        topics: Sequence[str] = (),
    ) -> dict[str, Any]:
        if self._thought_loop is None:
            return {"accepted": False, "reason": "cortex_unavailable"}
        normalized_source = self._normalize_action_text(source).lower() or "operator"
        normalized_reason = self._normalize_action_text(reason)
        normalized_query = self._normalize_cortex_query_hint(query_text)
        normalized_thought = self._normalize_action_text(thought_text)[:240]
        normalized_topics = [
            self._normalize_action_text(topic).lower()
            for topic in list(topics)[:4]
            if self._normalize_action_text(topic)
        ]
        control_id = str(uuid4())
        request = self._thought_loop.request_sleep(
            source=normalized_source,
            reason=normalized_reason or (
                "Operator requested cortex sleep."
                if normalized_source == "operator"
                else "Cortex requested a sleep cycle."
            ),
            metadata={
                "control_id": control_id,
                "query_text": normalized_query,
                "thought_text": normalized_thought,
                "topics": normalized_topics,
            },
        )
        request_payload = deepcopy(request.get("request") or {})
        metadata = request_payload.get("metadata") if isinstance(request_payload.get("metadata"), Mapping) else {}
        self._record_brain_event_locked(
            {
                "type": "cortex_sleep_requested",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "control_id": str(metadata.get("control_id", control_id)),
                "source": normalized_source,
                "action_intent": "sleep" if normalized_source == "cortex_intent" else None,
                "reason": str(request_payload.get("reason", normalized_reason)),
                "query_text": str(metadata.get("query_text", normalized_query)),
                "thought_text": str(metadata.get("thought_text", normalized_thought)),
                "topics": list(metadata.get("topics") or normalized_topics)[:4],
                "coalesced": bool(request.get("coalesced", False)),
            }
        )
        return {
            "accepted": bool(request.get("accepted", False)),
            "coalesced": bool(request.get("coalesced", False)),
            "running": bool(request.get("running", False)),
            "request": request_payload,
            "sleep_control": deepcopy(request.get("sleep_control") or {}),
        }

    def _handle_cortex_sleep_intent_locked(self, result: Any) -> dict[str, Any] | None:
        query_hint = self._consume_cortex_query_hint_locked()
        thought_text = self._normalize_action_text(getattr(result, "thought", ""))
        topics = [
            self._normalize_action_text(topic)
            for topic in list(getattr(result, "topics", ()) or ())
            if self._normalize_action_text(topic)
        ]
        return self._request_cortex_sleep_locked(
            source="cortex_intent",
            reason=thought_text or "Cortex requested a sleep cycle.",
            query_text=query_hint,
            thought_text=thought_text,
            topics=topics,
        )

    def _on_cortex_sleep_cycle(self, summary: dict[str, Any]) -> None:
        if not isinstance(summary, Mapping) or not bool(summary.get("requested", False)):
            return
        request = summary.get("request") if isinstance(summary.get("request"), Mapping) else {}
        metadata = request.get("metadata") if isinstance(request.get("metadata"), Mapping) else {}
        source = self._normalize_action_text(request.get("source", "")).lower() or "unknown"
        topics = [
            self._normalize_action_text(topic).lower()
            for topic in list(metadata.get("topics") or [])[:4]
            if self._normalize_action_text(topic)
        ]
        with self._lock:
            self._record_brain_event_locked(
                {
                    "type": "cortex_sleep_completed",
                    "timestamp": str(summary.get("completed_at") or datetime.now(timezone.utc).isoformat()),
                    "control_id": self._normalize_action_text(metadata.get("control_id", "")),
                    "source": source,
                    "action_intent": "sleep" if source == "cortex_intent" else None,
                    "reason": self._normalize_action_text(request.get("reason", "")),
                    "query_text": self._normalize_cortex_query_hint(metadata.get("query_text", "")),
                    "thought_text": self._normalize_action_text(metadata.get("thought_text", "")),
                    "topics": topics,
                    "dreams_generated": int(summary.get("dreams_generated", 0) or 0),
                    "sleep_cycles": int(summary.get("sleep_cycles", 0) or 0),
                    "trigger": self._normalize_action_text(summary.get("trigger", "")),
                }
            )

    def _cortex_action_query_locked(self, result: Any, *, query_hint: str) -> str:
        if query_hint:
            terms = self._action_query_terms(query_hint)
            if terms:
                return " ".join(terms[:4])
            return query_hint
        topics = [
            self._normalize_action_text(topic)
            for topic in list(getattr(result, "topics", ()) or ())
            if self._normalize_action_text(topic)
        ]
        if topics:
            return " ".join(topics[:4])
        thought = self._normalize_action_text(getattr(result, "thought", ""))
        if thought:
            terms = self._action_query_terms(thought)
            if terms:
                return " ".join(terms[:4])
            return thought[:160]
        return ""

    def _filter_cortex_action_records_locked(
        self,
        records: Sequence[dict[str, Any]],
        *,
        explicit_api_url: str,
        explicit_url: str,
        explicit_path: str,
    ) -> list[dict[str, Any]]:
        if explicit_api_url:
            return [
                record
                for record in records
                if self._api_request_record_matches_explicit_url(record, explicit_api_url)
            ]
        if explicit_url:
            return [
                record
                for record in records
                if str(record.get("action_type", "")) == "web_fetch"
                and self._normalize_action_text((record.get("inputs") or {}).get("url", "")) == explicit_url
            ]
        if explicit_path:
            return [
                record
                for record in records
                if str(record.get("action_type", "")) == "workspace_read"
                and self._normalize_action_text((record.get("inputs") or {}).get("path", "")) == explicit_path
            ]
        return list(records)

    @classmethod
    def _cortex_action_trigger_reason(cls, action_intent: str, action_type: str) -> str:
        normalized_intent = cls._normalize_action_text(action_intent).lower()
        normalized_action = cls._normalize_action_text(action_type).lower()
        if normalized_intent == "search":
            if normalized_action == "api_request":
                return "cortex_action_api_request"
            if normalized_action == "web_fetch":
                return "cortex_action_fetch"
            if normalized_action == "workspace_read":
                return "cortex_action_read"
            return "cortex_action_search"
        if normalized_intent in {"ask", "remember", "explore"}:
            return f"cortex_action_{normalized_intent}"
        return "cortex_action_search"

    def _handle_cortex_action_intent_locked(
        self,
        result: Any,
        *,
        action_intent: str | None = None,
    ) -> dict[str, Any] | None:
        normalized_intent = self._normalize_action_text(action_intent or getattr(result, "action_intent", "")).lower()
        if normalized_intent not in {"search", "ask", "remember", "explore"}:
            return None
        query_hint = self._consume_cortex_query_hint_locked()
        search_query = self._cortex_action_query_locked(result, query_hint=query_hint)
        if not search_query:
            return None
        target_query = query_hint or search_query
        explicit_api_url = self._query_api_url_candidate(target_query)
        explicit_url = self._query_web_url_candidate(target_query) if not explicit_api_url else ""
        explicit_path = self._query_workspace_path_candidate_locked(target_query) if not (explicit_api_url or explicit_url) else ""
        recent_verified = self._filter_cortex_action_records_locked(
            self._recent_relevant_action_records_locked(target_query, statuses=("verified",), limit=2),
            explicit_api_url=explicit_api_url,
            explicit_url=explicit_url,
            explicit_path=explicit_path,
        )
        if recent_verified:
            self._record_brain_event_locked(
                {
                    "type": "cortex_action_reused",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action_intent": normalized_intent,
                    "query_text": target_query,
                    "action_id": str(recent_verified[0].get("action_id", "")),
                }
            )
            return {"reused": True, "record": deepcopy(recent_verified[0])}
        recent_contradicted = self._filter_cortex_action_records_locked(
            self._recent_relevant_action_records_locked(target_query, statuses=("contradicted",), limit=1),
            explicit_api_url=explicit_api_url,
            explicit_url=explicit_url,
            explicit_path=explicit_path,
        )
        if recent_contradicted:
            self._record_brain_event_locked(
                {
                    "type": "cortex_action_reused",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action_intent": normalized_intent,
                    "query_text": target_query,
                    "action_id": str(recent_contradicted[0].get("action_id", "")),
                }
            )
            return {"reused": True, "record": deepcopy(recent_contradicted[0])}

        action_type = "api_request" if explicit_api_url else ("web_fetch" if explicit_url else ("workspace_read" if explicit_path else "workspace_search"))
        self._record_brain_event_locked(
            {
                "type": "cortex_action_requested",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action_intent": normalized_intent,
                "query_text": target_query,
                "topics": list(getattr(result, "topics", ()) or ())[:4],
                "action_type": action_type,
            }
        )
        focused_query = self._action_focus_query_text(target_query)
        intent_label = f"Cortex {normalized_intent} intent"
        if action_type == "api_request":
            return self.execute_digital_action(
                {
                    "action_type": "api_request",
                    "url": explicit_api_url,
                    "query_text": focused_query,
                    "predicted_outcome": f"{intent_label} expects requesting structured JSON from {explicit_api_url} to reveal grounded evidence relevant to: {target_query}.",
                },
                trigger_reason=self._cortex_action_trigger_reason(normalized_intent, action_type),
                trigger_query_text=target_query,
            )
        if action_type == "web_fetch":
            return self.execute_digital_action(
                {
                    "action_type": "web_fetch",
                    "url": explicit_url,
                    "query_text": focused_query,
                    "predicted_outcome": f"{intent_label} expects fetching {explicit_url} to reveal grounded evidence relevant to: {target_query}.",
                },
                trigger_reason=self._cortex_action_trigger_reason(normalized_intent, action_type),
                trigger_query_text=target_query,
            )
        if action_type == "workspace_read":
            return self.execute_digital_action(
                {
                    "action_type": "workspace_read",
                    "path": explicit_path,
                    "query_text": focused_query,
                    "predicted_outcome": f"{intent_label} expects reading {explicit_path} to reveal grounded workspace evidence relevant to: {target_query}.",
                },
                trigger_reason=self._cortex_action_trigger_reason(normalized_intent, action_type),
                trigger_query_text=target_query,
            )
        return self.execute_digital_action(
            {
                "action_type": "workspace_search",
                "query_text": search_query,
                "predicted_outcome": f"{intent_label} expects grounded workspace evidence relevant to: {target_query}.",
            },
            trigger_reason=self._cortex_action_trigger_reason(normalized_intent, action_type),
            trigger_query_text=target_query,
        )

    def _on_cortex_thought(self, result: Any) -> None:
        action_intent = self._normalize_action_text(getattr(result, "action_intent", "")).lower()
        if action_intent == "sleep":
            with self._lock:
                self._handle_cortex_sleep_intent_locked(result)
            return
        if action_intent not in {"search", "ask", "remember", "explore"}:
            return
        with self._lock:
            self._handle_cortex_action_intent_locked(result, action_intent=action_intent)

    def cortex_ask(self, query: str) -> dict[str, Any]:
        """Submit a question to the cortex and return immediately.

        The cortex will answer asynchronously in its next deliberation cycle.
        Returns acknowledgement with queue depth.
        """
        if self._thought_loop is None:
            return {"accepted": False, "reason": "cortex_unavailable"}
        with self._lock:
            self._remember_cortex_query_hint_locked(query)
        self._thought_loop.submit_query(query)
        return {"accepted": True, "query": query}

    def cortex_sleep(self, reason: str | None = None) -> dict[str, Any]:
        """Request an explicit cortex sleep cycle on the maintained control path."""
        normalized_reason = self._normalize_action_text(reason or "")
        with self._lock:
            return self._request_cortex_sleep_locked(
                source="operator",
                reason=normalized_reason or "Operator requested cortex sleep.",
            )

    def cortex_thoughts(self, limit: int = 20) -> dict[str, Any]:
        """Return recent thoughts from the cortex thought loop."""
        if self._thought_loop is None:
            return {"enabled": False, "thoughts": []}
        snap = self._thought_loop.snapshot()
        thoughts = snap.get("recent_thoughts", [])
        return {
            "enabled": True,
            "running": snap.get("running", False),
            "thoughts_generated": snap.get("thoughts_generated", 0),
            "dreams_generated": snap.get("dreams_generated", 0),
            "current_mode": snap.get("current_mode", "idle"),
            "thoughts": thoughts[-limit:],
        }

    def cortex_snapshot(self) -> dict[str, Any]:
        """Full cortex status snapshot."""
        if self._thought_loop is None:
            return {"enabled": False}
        return self._thought_loop.snapshot()

    def _living_loop_snapshot_locked(self, *, cortex_snapshot: Mapping[str, Any] | None = None) -> dict[str, Any]:
        cortex_data = dict(cortex_snapshot or (self._thought_loop.snapshot() if self._thought_loop is not None else {"enabled": False}))
        episodic_memory = cortex_data.get("episodic_memory") if isinstance(cortex_data.get("episodic_memory"), Mapping) else {}
        provenance = ProvenanceState.from_distribution(
            cast(Mapping[str, Any], episodic_memory).get("provenance_distribution")
            if isinstance(episodic_memory, Mapping)
            else {}
        )
        action_records = [
            ActionExecutionRecord.from_payload(item)
            for item in list(self._action_history)[:8]
            if isinstance(item, Mapping)
        ]
        consolidation_records = [
            ConsolidationRecord.from_payload(item)
            for item in list(self._delayed_consequence_records)[:8]
            if isinstance(item, Mapping)
        ]
        narrative = cortex_data.get("narrative_self") if isinstance(cortex_data.get("narrative_self"), Mapping) else {}
        cortex_summary = {
            "enabled": bool(cortex_data.get("enabled", False)),
            "running": bool(cortex_data.get("running", False)),
            "current_mode": str(cortex_data.get("current_mode", "idle")),
            "thoughts_generated": int(cortex_data.get("thoughts_generated", 0) or 0),
            "dreams_generated": int(cortex_data.get("dreams_generated", 0) or 0),
            "sleep_cycles": int(cortex_data.get("sleep_cycles", 0) or 0),
            "memory_count": int(cortex_data.get("memory_count", 0) or 0),
        }
        model = OperationalSelfModel.build(
            token_count=int(self._trainer.token_count),
            state_revision=int(self._state_revision),
            configured=bool(self._brain_config.get("source_bank")),
            running=bool(self._brain_runtime_active_locked()),
            provenance=provenance,
            predictions=[item.prediction for item in action_records],
            actions=action_records,
            consolidations=consolidation_records,
            action_loop=self._action_loop_summary_locked(),
            memory=dict(episodic_memory) if isinstance(episodic_memory, Mapping) else {},
            narrative=dict(narrative) if isinstance(narrative, Mapping) else {},
            cortex=cortex_summary,
        )
        return model.to_payload()

    def living_loop_status(self) -> dict[str, Any]:
        with self._lock:
            cortex_snapshot = self._thought_loop.snapshot() if self._thought_loop is not None else {"enabled": False}
            return {
                "living_loop": self._living_loop_snapshot_locked(cortex_snapshot=cortex_snapshot),
                "dirty_state": bool(self._dirty_state),
                "state_revision": int(self._state_revision),
                "token_count": int(self._trainer.token_count),
            }

    def action_history(self, limit: int = 20) -> dict[str, Any]:
        with self._lock:
            count = max(1, int(limit))
            history = [deepcopy(item) for item in list(self._action_history)[:count]]
            return {
                "count": int(len(self._action_history)),
                "root_path": str(self._action_root),
                "supported_actions": ["workspace_search", "workspace_read", "web_fetch", "api_request"],
                "actions": history,
            }

    def execute_digital_action(
        self,
        action: Mapping[str, Any],
        *,
        trigger_reason: str | None = None,
        trigger_query_text: str | None = None,
    ) -> dict[str, Any]:
        action_type = " ".join(str(action.get("action_type", action.get("type", ""))).split()).strip().lower()
        if action_type not in {"workspace_search", "workspace_read", "web_fetch", "api_request"}:
            return {"accepted": False, "reason": "unsupported_action_type", "action_type": action_type or None}

        requested_root = Path(str(action.get("root_path", ".") or "."))
        candidate_root = requested_root if requested_root.is_absolute() else (self._action_root / requested_root)
        try:
            resolved_root = candidate_root.resolve()
        except Exception:
            return {"accepted": False, "reason": "invalid_root_path", "action_type": action_type}
        if resolved_root != self._action_root and self._action_root not in resolved_root.parents:
            return {
                "accepted": False,
                "reason": "root_path_outside_workspace",
                "action_type": action_type,
                "workspace_root": str(self._action_root),
            }

        try:
            result = execute_digital_action(resolved_root, action)
        except Exception as exc:
            return {"accepted": False, "reason": "execution_failed", "action_type": action_type, "message": str(exc)}

        payload = result.to_payload()
        if trigger_reason is not None:
            payload["trigger_reason"] = str(trigger_reason)
        if trigger_query_text is not None:
            payload["trigger_query_text"] = str(trigger_query_text)
        normalized = self._normalize_action_record(payload)
        if normalized is None:
            return {"accepted": False, "reason": "normalization_failed", "action_type": action_type}

        with self._lock:
            existing = [
                item
                for item in list(self._action_history)
                if str(item.get("action_id", "")) != str(normalized.get("action_id", ""))
            ]
            self._action_history = deque(existing, maxlen=self._action_history.maxlen)
            self._action_history.appendleft(normalized)
            self._inject_action_record_into_cortex_locked(normalized)
            verification = normalized.get("verification") if isinstance(normalized.get("verification"), Mapping) else {}
            normalized_trigger_query_text = self._normalize_cortex_query_hint(normalized.get("trigger_query_text", ""))
            autonomy = cast(dict[str, Any] | None, self._brain_config.get("autonomy"))
            if autonomy and normalized_trigger_query_text:
                confidence = max(0.0, min(1.0, float(verification.get("confidence", 0.0) or 0.0)))
                if bool(verification.get("success", False)):
                    action_outcome_score = max(0.0, min(1.0, 0.55 + 0.35 * confidence))
                elif bool(verification.get("contradiction", False)):
                    action_outcome_score = 0.0
                else:
                    action_outcome_score = max(0.0, min(1.0, 0.10 + 0.20 * confidence))
                self._apply_provider_outcome_calibration_locked(
                    autonomy=autonomy,
                    query_text=normalized_trigger_query_text,
                    outcome_score=action_outcome_score,
                )
            self._record_brain_event_locked(
                {
                    "type": "digital_action_executed",
                    "timestamp": str(normalized.get("recorded_at")),
                    "action_id": str(normalized.get("action_id", "")),
                    "action_type": str(normalized.get("action_type", "")),
                    "trigger_reason": str(normalized.get("trigger_reason", "operator") or "operator"),
                    "trigger_query_text": str(normalized.get("trigger_query_text", "") or ""),
                    "verification_status": str((normalized.get("verification") or {}).get("status", "unknown")),
                    "success": bool((normalized.get("verification") or {}).get("success", False)),
                    "contradiction": bool((normalized.get("verification") or {}).get("contradiction", False)),
                }
            )
            self._mark_mutated()
            runtime = self._brain_runtime_snapshot_locked()
        return {
            "accepted": True,
            "result": deepcopy(normalized),
            "terminus_runtime": runtime,
            "state_revision": int(self._state_revision),
        }

    @staticmethod
    def _sensory_media_payload(media: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(media, dict):
            return None
        raw_bytes = media.get("bytes")
        if not isinstance(raw_bytes, (bytes, bytearray)):
            return None
        mime_type = str(media.get("mime_type", "application/octet-stream"))
        data_url = f"data:{mime_type};base64,{base64.b64encode(bytes(raw_bytes)).decode('ascii')}"
        payload = {
            key: deepcopy(value)
            for key, value in media.items()
            if key != "bytes"
        }
        payload["byte_size"] = len(raw_bytes)
        payload["data_url"] = data_url
        return payload

    def sensory_previews(self, limit: int = 6) -> dict[str, Any]:
        acquired = self._lock.acquire(timeout=0.15)
        if not acquired:
            self._lock.acquire()
        try:
            previews = []
            for item in list(self._sensory_preview_history)[: max(1, int(limit))]:
                previews.append(
                    {
                        "preview_id": str(item.get("preview_id", "")),
                        "captured_at": str(item.get("captured_at", "")),
                        "source_name": str(item.get("source_name", "")),
                        "adapter": str(item.get("adapter", "")),
                        "text": str(item.get("text", "")),
                        "semantic_match": float(item.get("semantic_match", 0.0) or 0.0),
                        "modality_need": float(item.get("modality_need", 0.0) or 0.0),
                        "item_semantic_match": float(item.get("item_semantic_match", 0.0) or 0.0),
                        "item_candidates_considered": int(item.get("item_candidates_considered", 0) or 0),
                        "item_retrieval_lookahead": int(item.get("item_retrieval_lookahead", 1) or 1),
                        "selection_score": float(item.get("selection_score", 0.0) or 0.0),
                        "window_budget": int(item.get("window_budget", 0) or 0),
                        "topics": [str(topic) for topic in list(item.get("topics") or [])],
                        "focus_terms": [str(term) for term in list(item.get("focus_terms") or [])],
                        "metadata": deepcopy(item.get("metadata") or {}),
                        "visual": self._sensory_media_payload(cast(dict[str, Any] | None, item.get("visual"))),
                        "audio": self._sensory_media_payload(cast(dict[str, Any] | None, item.get("audio"))),
                    }
                )
            return {
                "count": int(len(self._sensory_preview_history)),
                "latest_preview_id": None if not self._sensory_preview_history else str(self._sensory_preview_history[0].get("preview_id", "")),
                "previews": previews,
            }
        finally:
            self._lock.release()

    def _cortex_signal_state(self) -> dict[str, Any]:
        """Expose recent SNN predictive/surprise signals to the ThoughtLoop."""
        acquired = self._lock.acquire(timeout=0.05)
        if not acquired:
            return getattr(self, "_cached_cortex_signal_state", {})
        try:
            predictive = getattr(self._trainer.model, "predictive", None)
            surprise = getattr(self._trainer.model, "surprise", None)
            recent_concepts: list[str] = []
            concept_candidates: list[dict[str, Any]] = []
            try:
                snap = self._concept_store.snapshot(limit=6)
                for concept in snap.get("top_concepts", [])[:6]:
                    if not isinstance(concept, dict):
                        continue
                    label = str(concept.get("label", "")).strip()
                    if label:
                        recent_concepts.append(label)
                    top_terms = [
                        str(term).strip()
                        for term in list(concept.get("top_terms") or [])[:4]
                        if str(term).strip()
                    ]
                    examples = [
                        str(text).strip()
                        for text in list(concept.get("example_windows") or [])[:2]
                        if str(text).strip()
                    ]
                    if label or top_terms:
                        concept_candidates.append(
                            {
                                "label": label,
                                "top_terms": top_terms,
                                "match_count": int(concept.get("match_count", concept.get("observations", 0)) or 0),
                                "observations": int(concept.get("observations", concept.get("match_count", 0)) or 0),
                                "uncertainty": float(concept.get("uncertainty", 1.0) or 1.0),
                                "temporal_coherence": float(concept.get("temporal_coherence", 0.0) or 0.0),
                                "example_windows": examples,
                            }
                        )
            except Exception:
                pass

            payload = {
                "prediction_error_mean": 0.0,
                "prediction_error_max": 0.0,
                "predictive_confidence_mean": 0.5,
                "predictive_confidence_min": 0.5,
                "dopamine": float(getattr(surprise, "dopamine", 0.0)) if surprise is not None else 0.0,
                "norepinephrine": float(getattr(surprise, "norepinephrine", 0.0)) if surprise is not None else 0.0,
                "recent_concepts": recent_concepts,
                "concept_candidates": concept_candidates,
            }
            if predictive is not None:
                try:
                    prediction_error = predictive.prediction_error.detach().float().cpu()
                    confidence = predictive.confidence.detach().float().cpu()
                    if prediction_error.numel() > 0:
                        payload["prediction_error_mean"] = float(prediction_error.mean().item())
                        payload["prediction_error_max"] = float(prediction_error.max().item())
                    if confidence.numel() > 0:
                        payload["predictive_confidence_mean"] = float(confidence.mean().item())
                        payload["predictive_confidence_min"] = float(confidence.min().item())
                except Exception:
                    pass

            self._cached_cortex_signal_state = payload
            return payload
        finally:
            self._lock.release()

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

    def architecture_summary(self) -> dict[str, Any]:
        """Return a current runtime-driven description of the active Terminus architecture."""
        with self._lock:
            model = self._trainer.model
            config = self._trainer.config
            sensory = self._brain_config.get("sensory") or {}
            autonomy = self._brain_config.get("autonomy") or {}
            predictive_enabled = bool(getattr(model, "predictive", None) is not None)
            cortex_snapshot = self.cortex_snapshot() if self._thought_loop is not None else {"enabled": False}
            layers: list[dict[str, Any]] = []
            layers.append({
                "id": "input_encoding",
                "name": "Input + Stream Ingestion",
                "enabled": True,
                "type": "input",
                "params": {
                    "input_dim": int(config.input_dim),
                    "representation": config.input_representation,
                    "background_sources": int(len(self._brain_source_runtimes)),
                    "background_routing": "focus_aware_allocation",
                    "sensory_sources": int(len(self._sensory_source_runtimes)),
                    "learned_chunking": bool(config.enable_learned_chunking),
                },
            })
            layers.append({
                "id": "competitive_routing",
                "name": "GPCSN Column Field",
                "enabled": True,
                "type": "core",
                "params": {
                    "n_columns": int(config.n_columns),
                    "k_routing": int(config.k_routing),
                    "plasticity_mode": config.plasticity_mode,
                    "plasticity_rule": config.plasticity_rule,
                },
            })
            layers.append({
                "id": "predictive_columns",
                "name": "Predictive Columns",
                "enabled": predictive_enabled,
                "type": "prediction",
                "params": {
                    "enabled": predictive_enabled,
                    "prediction_error_driven": predictive_enabled,
                } if predictive_enabled else {},
            })
            layers.append({
                "id": "context_prediction",
                "name": f"Context Attractor ({config.context_mode})",
                "enabled": model.context_layer is not None,
                "type": "context",
                "params": {
                    "context_mode": config.context_mode,
                },
            })
            layers.append({
                "id": "binding",
                "name": "Hypercube Binding + Hubs",
                "enabled": model.binding_layer is not None,
                "type": "binding",
                "params": {
                    "n_bindings": int(config.binding_n_bindings),
                    "fan_in": int(config.binding_fan_in),
                    "topology": type(model.binding_layer).__name__ if model.binding_layer is not None else "disabled",
                } if model.binding_layer is not None else {},
            })
            layers.append({
                "id": "abstraction",
                "name": "Abstraction Layer",
                "enabled": model.abstraction_layer is not None,
                "type": "abstraction",
                "params": {
                    "n_concepts": int(config.abstraction_n_concepts),
                } if model.abstraction_layer is not None else {},
            })
            layers.append({
                "id": "cross_modal_grounding",
                "name": "Real Cross-Modal Grounding",
                "enabled": model.cross_modal is not None,
                "type": "grounding",
                "params": {
                    "dim_visual": int(config.cross_modal_dim_visual),
                    "dim_audio": int(config.cross_modal_dim_audio),
                    "visual_confidence": float(model.cross_modal.visual_confidence.mean().item()) if model.cross_modal else 0.0,
                    "audio_confidence": float(model.cross_modal.audio_confidence.mean().item()) if model.cross_modal else 0.0,
                    "sensory_active": bool(sensory.get("enabled", False)),
                },
            })
            layers.append({
                "id": "memory_consolidation",
                "name": "Dual Memory + Consolidation",
                "enabled": True,
                "type": "memory",
                "params": {
                    "memory_capacity": int(config.memory_capacity),
                    "stc_tag_duration_strong": float(config.stc_tag_duration_strong),
                },
            })
            layers.append({
                "id": "nim_cortex",
                "name": "NIM Mind Layer",
                "enabled": bool(cortex_snapshot.get("enabled", False)),
                "type": "cortex",
                "params": {
                    "thoughts_generated": int(cortex_snapshot.get("thoughts_generated", 0) or 0),
                    "working_memory": bool(cortex_snapshot.get("working_memory") is not None),
                    "narrative_self": bool(cortex_snapshot.get("narrative_self") is not None),
                } if bool(cortex_snapshot.get("enabled", False)) else {},
            })
            layers.append({
                "id": "autonomy_guidance",
                "name": "Active Exploration + Grounded-Family-Summary Lineage-Reconvergent Divergence-Split Trajectory-Sensitive Compacted Age-Sensitive Consequence-Calibrated Real-Source Guidance",
                "enabled": bool(autonomy.get("enabled", False)) or bool(sensory.get("enabled", False)),
                "type": "autonomy",
                "params": {
                    "autonomy_enabled": bool(autonomy.get("enabled", False)),
                    "candidate_count": int(len(autonomy.get("candidate_bank", []))) if autonomy else 0,
                    "adaptive_focus_budgeting": bool(autonomy.get("enabled", False)),
                    "grounded_outcome_calibration": bool(autonomy.get("enabled", False)),
                    "evidence_provenance_credit": True,
                    "delayed_multi_turn_consequence_tracking": True,
                    "contradiction_decay_penalties": True,
                    "mixed_evidence_forgiveness_scheduling": True,
                    "age_sensitive_consequence_cooling": True,
                    "consequence_state_retirement": True,
                    "consequence_record_compaction": True,
                    "trajectory_sensitive_consequence_families": True,
                    "divergence_sensitive_consequence_splitting": True,
                    "lineage_aware_consequence_remerge": True,
                    "grounded_family_summary_calibration": True,
                    "sensory_enabled": bool(sensory.get("enabled", False)),
                    "items_per_episode": int(sensory.get("items_per_episode", 0)) if sensory else 0,
                },
            })
            return {
                "model_name": "Terminus",
                "core_name": "GPCSN",
                "version": "current",
                "family": "hybrid_snn_llm",
                "layers": layers,
                "config": {
                    "context_mode": config.context_mode,
                    "plasticity_rule": config.plasticity_rule,
                    "n_columns": int(config.n_columns),
                    "cross_modal": bool(model.cross_modal is not None),
                },
            }

    def run_grounding_probe(self) -> dict[str, Any]:
        """Run the 50-triple grounding probe and return results.

        When cross-modal grounding is enabled, the probe vector blends
        the routing key with the visual prediction from W_tv, so that
        concrete concepts with strong visual grounding produce distinct
        representations from abstract concepts (§8.7).
        """
        from hecsn.evaluation.grounding_probe import evaluate_grounding_probe
        with self._lock:
            trainer = self._trainer
            encoder = self._encoder
            cross_modal = trainer.model.cross_modal

            def _vector_fn(text: str) -> torch.Tensor:
                patterns = list(encoder.iter_char_patterns(text, window_size=8, learn=False))
                if not patterns:
                    return torch.zeros(trainer.config.n_columns, device=trainer.model.device)
                vecs = [trainer.model.routing_key_from_pattern(p[1]) for p in patterns]
                routing_key = torch.stack(vecs).mean(dim=0)

                if cross_modal is not None and routing_key.shape[0] == cross_modal.W_tv.shape[0]:
                    # Predict visual representation and blend with routing key
                    pred_visual = torch.mv(cross_modal.W_tv.T, routing_key)
                    visual_conf = float(cross_modal.visual_confidence.mean().item())
                    if pred_visual.norm() > 1e-6 and visual_conf > 0.01:
                        # Project visual prediction back to text space
                        visual_feedback = torch.mv(cross_modal.W_vt.T, pred_visual)
                        if visual_feedback.shape == routing_key.shape:
                            blend = min(0.3, visual_conf)
                            routing_key = (1.0 - blend) * routing_key + blend * visual_feedback
                return routing_key

            result = evaluate_grounding_probe(_vector_fn)
            return {
                "total_accuracy": float(result.total_accuracy),
                "concrete_accuracy": float(result.concrete_accuracy),
                "abstract_accuracy": float(result.abstract_accuracy),
                "concreteness_gap": float(result.concreteness_gap),
                "visual_text_accuracy": float(result.visual_text_accuracy),
                "audio_text_accuracy": float(result.audio_text_accuracy),
                "visual_text_count": result.visual_text_count,
                "audio_text_count": result.audio_text_count,
                "sample_count": result.total_count,
            }

    def terminus_tick(self, *, steps: int = 1) -> dict[str, Any]:
        tick_summaries: list[dict[str, Any]] = []
        step_count = max(1, int(steps))

        with self._lock:
            self._assert_manual_tick_allowed_locked()

        self._request_active_execution()
        try:
            with self._brain_execution_lock:
                with self._lock:
                    self._assert_manual_tick_allowed_locked()

                for _ in range(step_count):
                    summary = self._run_brain_tick_once(
                        stop_event=None,
                        sub_batch_size=1,
                        yield_seconds=0.0,
                    )
                    if summary is None:
                        break
                    tick_summaries.append(summary)
                    if not bool(summary.get("did_work", False)):
                        break
        finally:
            self._release_active_execution()

        with self._lock:
            return {
                "terminus_runtime": self._brain_runtime_snapshot_locked(),
                "tick_summaries": tick_summaries,
                "dirty_state": bool(self._dirty_state),
                "state_revision": int(self._state_revision),
                "token_count": int(self._trainer.token_count),
            }

    def recent_traces(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            count = max(1, int(limit))
            return [deepcopy(trace) for trace in list(self._trace_history)[:count]]

    def save_checkpoint(self, path: str | None = None) -> dict[str, Any]:
        with self._lock:
            target = self._resolve_save_path(path)
            metadata = deepcopy(self._metadata)
            service_state = dict(metadata.get("service_state", {}))
            service_state["concept_store"] = self._concept_store.state_dict()
            service_state["terminus_runtime"] = self._brain_persisted_state_locked()
            metadata.update(
                {
                    "saved_by": "hecsn.service",
                    "state_revision": int(self._state_revision),
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "service_state": service_state,
                }
            )
            saved_path = save_trainer_checkpoint(target, self._trainer, metadata=metadata)
            self._checkpoint_path = saved_path
            self._checkpoint_dir = saved_path.parent
            self._metadata = metadata
            self._dirty_state = False
            return {
                "path": str(saved_path),
                "dirty_state": bool(self._dirty_state),
                "state_revision": int(self._state_revision),
                "token_count": int(self._trainer.token_count),
            }

    def restore_checkpoint(self, path: str | Path) -> dict[str, Any]:
        thread = self._request_brain_stop()
        self._join_brain_thread(thread)
        with self._lock:
            checkpoint_path = Path(path)
            trainer, metadata = load_trainer_checkpoint(checkpoint_path)
            self._trainer = trainer
            self._metadata = dict(metadata)
            self._encoder = self._trainer.encoder
            self._checkpoint_path = checkpoint_path
            self._checkpoint_dir = checkpoint_path.parent if checkpoint_path.parent != Path("") else Path("checkpoints")
            self._runtime_env = load_runtime_env(anchor_paths=(self._env_root, self._checkpoint_dir))
            self._action_root = (self._env_root or self._checkpoint_dir).resolve()
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
            self._brain_source_utility = self._normalize_background_source_utility_state(
                terminus_state.get("background_source_utility")
            )
            self._brain_last_error = None
            self._brain_recent_query_gaps = deque(
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
            self._action_history = deque(
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
            self._delayed_consequence_records = deque(
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
            self._replay_action_history_into_cortex_locked()
            self._brain_last_acquisition_summary = None
            self._brain_last_acquisition_token_count = int(self._trainer.token_count)
            self._rebuild_brain_sources_locked()
            self._dirty_state = False
            self._state_revision += 1
            return {
                "path": str(checkpoint_path),
                "dirty_state": bool(self._dirty_state),
                "state_revision": int(self._state_revision),
                "token_count": int(self._trainer.token_count),
            }

    def _build_query_locked(
        self,
        *,
        query_text: str,
        context_text: str | None,
        top_k_candidates: int,
        top_k_memories: int,
        top_chars: int,
    ) -> dict[str, Any]:
        query_focus_plan = self._concept_store.focus_plan(
            query_text=query_text,
            min_observations=1,
        )
        retrieval_focus_terms = None
        memory_priority = None
        if query_focus_plan is not None:
            retrieval_focus_terms = list(
                query_focus_plan.get("focus_terms")
                or query_focus_plan.get("query_terms")
                or []
            )
            raw_memory_priority = dict(query_focus_plan.get("memory_priority") or {})
            if raw_memory_priority:
                memory_priority = raw_memory_priority
        try:
            result = build_query_result(
                trainer=self._trainer,
                checkpoint=self._checkpoint_path,
                metadata=deepcopy(self._metadata),
                encoder=self._encoder,
                query_text_resolved=query_text,
                feed_text_resolved=None,
                context_text=context_text,
                top_k_candidates=top_k_candidates,
                top_k_memories=top_k_memories,
                top_chars=top_chars,
                compare_context_a=None,
                compare_context_b=None,
                retrieval_focus_terms=retrieval_focus_terms,
                memory_priority=memory_priority,
            )
            query_summary = result.get("query_summary")
            if isinstance(query_summary, dict) and query_focus_plan is not None:
                query_summary["abstraction_focus"] = deepcopy(query_focus_plan)
            return result
        finally:
            self._trainer.reset_context_state()

    def _learn_from_turn_locked(
        self,
        *,
        query_text: str,
        response: dict[str, Any],
        learn_mode: str,
    ) -> dict[str, Any] | None:
        if learn_mode == "none":
            return None

        user_feed = feed_text(
            self._trainer,
            self._encoder,
            query_text,
            on_step=self._runtime_concept_callback_locked(),
        )
        evidence_feed = None
        selected_texts = [
            str(item.get("text", "")).strip()
            for item in response.get("selected_evidence", [])
            if str(item.get("text", "")).strip()
        ]

        if learn_mode == "user_and_selected_evidence" and selected_texts:
            evidence_feed = feed_text(
                self._trainer,
                self._encoder,
                "\n".join(selected_texts),
                on_step=self._runtime_concept_callback_locked(),
            )
        elif learn_mode != "user_only":
            raise ValueError(f"Unsupported learn_mode: {learn_mode}")

        self._mark_mutated()
        return {
            "learn_mode": learn_mode,
            "user_feed": user_feed,
            "evidence_feed": evidence_feed,
            "selected_evidence_count": int(len(selected_texts)),
        }

    def _mark_mutated(self) -> None:
        self._dirty_state = True
        self._state_revision += 1

    def _observe_concepts_locked(
        self,
        *,
        query_text: str,
        query_result: dict[str, Any],
    ) -> dict[str, Any]:
        query_summary = query_result.get("query_summary") or {}
        memory_matches = query_summary.get("memory_matches") or []
        memory_episodes = query_summary.get("memory_episodes") or []
        return self._concept_store.observe(
            query_text=query_text,
            memory_matches=memory_matches,
            memory_episodes=memory_episodes,
            memory_store=self._trainer.model.memory_store,
        )

    def _runtime_concept_callback_locked(self):
        def _observe(raw_window: str, metrics: dict[str, Any]) -> None:
            self._observe_runtime_concepts_locked(raw_window=raw_window, metrics=metrics)

        return _observe

    def _observe_runtime_concepts_locked(
        self,
        *,
        raw_window: str | None,
        metrics: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(metrics, dict):
            return None
        memory_index = metrics.get("memory_index")
        try:
            idx = int(memory_index)
        except (TypeError, ValueError):
            return None

        memory_store = self._trainer.model.memory_store
        routing_keys = getattr(memory_store, "slow_routing_keys", []) or []
        if idx < 0 or idx >= len(routing_keys):
            return None
        if not isinstance(routing_keys[idx], torch.Tensor):
            return None

        stored_texts = getattr(memory_store, "slow_texts", []) or []
        stored_windows = getattr(memory_store, "slow_raw_windows", []) or []
        source_text = ""
        if idx < len(stored_texts) and stored_texts[idx] is not None:
            source_text = str(stored_texts[idx])
        elif idx < len(stored_windows) and stored_windows[idx] is not None:
            source_text = str(stored_windows[idx])
        elif raw_window is not None:
            source_text = str(raw_window)
        source_text = " ".join(source_text.split()).strip()
        if not source_text or not any(char.isalnum() for char in source_text):
            return None

        raw_match = (
            str(stored_windows[idx])
            if idx < len(stored_windows) and stored_windows[idx] is not None
            else source_text
        )
        importance = 1.0
        capture_tag = 0.0
        consolidation_level = 0.0
        slow_importance = getattr(memory_store, "slow_importance", []) or []
        slow_capture_tag = getattr(memory_store, "slow_capture_tag", []) or []
        slow_consolidation = getattr(memory_store, "slow_consolidation_level", []) or []
        if idx < len(slow_importance):
            importance = float(memory_store.slow_importance[idx])
        if idx < len(slow_capture_tag):
            capture_tag = float(memory_store.slow_capture_tag[idx])
        if idx < len(slow_consolidation):
            consolidation_level = float(memory_store.slow_consolidation_level[idx])

        observed = self._concept_store.observe(
            query_text="",
            memory_matches=[
                {
                    "memory_index": idx,
                    "text": source_text,
                    "raw_window": raw_match,
                    "similarity": 1.0,
                    "importance": importance,
                    "capture_tag": capture_tag,
                    "consolidation_level": consolidation_level,
                }
            ],
            memory_store=memory_store,
            limit=4,
        )
        abstraction_layer = self._trainer.model.abstraction_layer
        if abstraction_layer is not None:
            self._geometric_curiosity.update_lexicon(
                abstraction_layer.last_activations,
                [source_text, raw_match],
            )
        return observed

    def _service_state_snapshot(self) -> dict[str, Any]:
        last_trace = self._trace_history[0] if self._trace_history else None
        return {
            "checkpoint_path": str(self._checkpoint_path),
            "dirty_state": bool(self._dirty_state),
            "state_revision": int(self._state_revision),
            "token_count": int(self._trainer.token_count),
            "last_trace_id": None if last_trace is None else str(last_trace.get("trace_id")),
            "concept_count": int(self._concept_store.snapshot().get("concept_count", 0)),
            "terminus_runtime": self._brain_runtime_snapshot_locked(),
        }

    def _normalize_brain_source_spec(self, spec: Any, index: int) -> dict[str, Any]:
        if not isinstance(spec, dict):
            raise ValueError("Each Terminus source must be an object")
        source = str(spec.get("source", "")).strip()
        if not source:
            raise ValueError("Each Terminus source requires a non-empty source")
        source_type = str(spec.get("source_type", "auto")).strip() or "auto"
        if source_type not in {"auto", "file", "hf", "web"}:
            raise ValueError("Terminus sources only support source_type auto/file/hf/web")
        name = str(spec.get("name", f"source_{index + 1}")).strip() or f"source_{index + 1}"
        text_field = str(spec.get("text_field", "text")).strip() or "text"
        hf_config_raw = spec.get("hf_config")
        hf_config = None if hf_config_raw in (None, "", "None") else str(hf_config_raw)
        normalized = {
            "name": name,
            "source": source,
            "source_type": source_type,
            "text_field": text_field,
            "hf_config": hf_config,
        }
        topic_terms = spec.get("topic_terms")
        if isinstance(topic_terms, Sequence) and not isinstance(topic_terms, (str, bytes)):
            normalized["topic_terms"] = [
                _canonical_provider_term(term)
                for term in list(topic_terms)
                if _canonical_provider_term(term)
            ]
        metadata = spec.get("metadata")
        if isinstance(metadata, dict) and metadata:
            normalized["metadata"] = deepcopy(metadata)
        return normalized

    def _normalize_sensory_source_spec(self, spec: Any, index: int) -> dict[str, Any]:
        if not isinstance(spec, dict):
            raise ValueError("Each Terminus sensory source must be an object")
        adapter = str(spec.get("adapter", "")).strip().lower()
        if adapter not in {"s1_mmalign", "audiocaps"}:
            raise ValueError("Terminus sensory sources require adapter 's1_mmalign' or 'audiocaps'")
        source = str(spec.get("source", "")).strip()
        if not source:
            source = "ScienceOne-AI/S1-MMAlign" if adapter == "s1_mmalign" else "OpenSound/AudioCaps"
        name = str(spec.get("name", f"sensory_{index + 1}")).strip() or f"sensory_{index + 1}"
        split = str(spec.get("split", "train")).strip() or "train"
        normalized: dict[str, Any] = {
            "name": name,
            "adapter": adapter,
            "source": source,
            "split": split,
        }
        if adapter == "s1_mmalign":
            year_prefixes = spec.get("year_prefixes")
            if isinstance(year_prefixes, Sequence) and not isinstance(year_prefixes, (str, bytes)):
                normalized["year_prefixes"] = [
                    str(item).zfill(2)[:2]
                    for item in list(year_prefixes)
                    if str(item).strip()
                ] or ["07", "08", "09"]
            else:
                normalized["year_prefixes"] = ["07", "08", "09"]
            normalized["max_text_chars"] = max(64, int(spec.get("max_text_chars", 480)))
        else:
            normalized["sample_rate"] = max(1000, int(spec.get("sample_rate", 16000)))
            normalized["n_fft"] = max(64, int(spec.get("n_fft", 512)))
            normalized["max_text_chars"] = max(32, int(spec.get("max_text_chars", 240)))
            normalized["audio_candidates_per_item"] = max(1, int(spec.get("audio_candidates_per_item", 6)))
        topic_terms = spec.get("topic_terms")
        if isinstance(topic_terms, Sequence) and not isinstance(topic_terms, (str, bytes)):
            normalized["topic_terms"] = [
                " ".join(str(term).split()).strip().lower()
                for term in list(topic_terms)
                if " ".join(str(term).split()).strip()
            ]
        metadata = spec.get("metadata")
        if isinstance(metadata, dict) and metadata:
            normalized["metadata"] = deepcopy(metadata)
        return normalized

    def _normalize_catalog_candidate_spec(self, spec: Any, index: int) -> dict[str, Any]:
        if not isinstance(spec, dict):
            raise ValueError("Each Terminus autonomy candidate must be an object")
        catalog_mode = str(spec.get("catalog_mode", "")).strip().lower()
        if catalog_mode not in {"semantic_registry", "live_remote_search"}:
            raise ValueError(
                "Catalog-backed candidate specs require catalog_mode "
                "'semantic_registry' or 'live_remote_search'"
            )
        name = str(spec.get("name", f"{catalog_mode}_{index + 1}")).strip() or f"{catalog_mode}_{index + 1}"
        normalized: dict[str, Any] = {
            "name": name,
            "catalog_mode": catalog_mode,
            "catalog_limit": max(1, int(spec.get("catalog_limit", 8))),
            "catalog_diversity_weight": float(spec.get("catalog_diversity_weight", 0.20)),
            "catalog_semantic_weight": float(spec.get("catalog_semantic_weight", 1.0)),
            "catalog_prior_weight": float(spec.get("catalog_prior_weight", 1.0)),
            "catalog_provider_timeout_seconds": max(
                1.0,
                float(spec.get("catalog_provider_timeout_seconds", 15.0)),
            ),
        }
        if "catalog_probe_pool_limit" in spec and spec.get("catalog_probe_pool_limit") is not None:
            normalized["catalog_probe_pool_limit"] = max(1, int(spec.get("catalog_probe_pool_limit", 1)))
        focus_text = " ".join(str(spec.get("catalog_focus_text", "")).split()).strip()
        if focus_text:
            normalized["catalog_focus_text"] = focus_text
        focus_terms = spec.get("catalog_focus_terms")
        if isinstance(focus_terms, Sequence) and not isinstance(focus_terms, (str, bytes)):
            normalized["catalog_focus_terms"] = [
                str(term).strip()
                for term in list(focus_terms)
                if str(term).strip()
            ]
        exclude_sources = spec.get("catalog_exclude_sources")
        if isinstance(exclude_sources, Sequence) and not isinstance(exclude_sources, (str, bytes)):
            normalized["catalog_exclude_sources"] = [
                str(item).strip()
                for item in list(exclude_sources)
                if str(item).strip()
            ]
        exclude_names = spec.get("catalog_exclude_names")
        if isinstance(exclude_names, Sequence) and not isinstance(exclude_names, (str, bytes)):
            normalized["catalog_exclude_names"] = [
                str(item).strip()
                for item in list(exclude_names)
                if str(item).strip()
            ]
        if catalog_mode == "semantic_registry":
            entries = list(spec.get("catalog_entries") or [])
            if not entries:
                raise ValueError("semantic_registry candidate specs require catalog_entries")
            normalized_entries: list[dict[str, Any]] = []
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise ValueError("catalog_entries items must be objects")
                normalized_entry = {
                    "name": str(entry.get("name", "")).strip(),
                    "source": str(entry.get("source", "")).strip(),
                    "source_type": str(entry.get("source_type", "auto")).strip() or "auto",
                    "text_field": str(entry.get("text_field", "text")).strip() or "text",
                }
                if normalized_entry["source_type"] not in {"auto", "hf", "web"}:
                    raise ValueError("catalog_entries source_type must be auto/hf/web")
                if not normalized_entry["name"] or not normalized_entry["source"]:
                    raise ValueError("catalog_entries items require non-empty name and source")
                hf_config_raw = entry.get("hf_config")
                if hf_config_raw not in (None, "", "None"):
                    normalized_entry["hf_config"] = str(hf_config_raw)
                for key in (
                    "summary",
                    "title",
                    "description",
                    "query_text",
                    "provider",
                ):
                    value = " ".join(str(entry.get(key, "")).split()).strip()
                    if value:
                        normalized_entry[key] = value
                for key in ("tags", "terms"):
                    values = entry.get(key)
                    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                        normalized_entry[key] = [
                            str(item).strip()
                            for item in list(values)
                            if str(item).strip()
                        ]
                for key in ("catalog_priority", "prior_weight"):
                    if key in entry and entry.get(key) is not None:
                        normalized_entry[key] = float(entry.get(key))
                normalized_entries.append(normalized_entry)
            normalized["catalog_entries"] = normalized_entries
        else:
            providers = spec.get("catalog_providers")
            if isinstance(providers, Sequence) and not isinstance(providers, (str, bytes)):
                normalized["catalog_providers"] = [
                    str(provider).strip()
                    for provider in list(providers)
                    if str(provider).strip()
                ]
            normalized["catalog_queries_per_provider"] = max(
                1,
                int(spec.get("catalog_queries_per_provider", 2)),
            )
            normalized["catalog_provider_result_limit"] = max(
                1,
                int(spec.get("catalog_provider_result_limit", 4)),
            )
        return normalized

    def _normalize_autonomy_candidate_spec(self, spec: Any, index: int) -> dict[str, Any]:
        if isinstance(spec, dict) and str(spec.get("catalog_mode", "")).strip():
            return self._normalize_catalog_candidate_spec(spec, index)
        return self._normalize_brain_source_spec(spec, index)

    def _normalize_provider_curriculum(self, value: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(value, Mapping):
            return {}

        def _safe_int(raw_value: Any) -> int:
            try:
                return max(0, int(raw_value))
            except (TypeError, ValueError):
                return 0

        def _safe_float(raw_value: Any) -> float:
            try:
                return max(0.0, float(raw_value))
            except (TypeError, ValueError):
                return 0.0

        normalized: dict[str, dict[str, Any]] = {}
        for raw_provider, raw_entry in value.items():
            provider = " ".join(str(raw_provider).split()).strip().lower()
            if not provider or not isinstance(raw_entry, Mapping):
                continue
            topic_terms: dict[str, float] = {}
            raw_topic_terms = raw_entry.get("topic_terms")
            if isinstance(raw_topic_terms, Mapping):
                for raw_term, raw_weight in raw_topic_terms.items():
                    term = _canonical_provider_term(raw_term)
                    if not term:
                        continue
                    weight = _safe_float(raw_weight)
                    if weight > 0.0:
                        topic_terms[term] = float(weight)
            topic_families: dict[str, dict[str, Any]] = {}
            raw_topic_families = raw_entry.get("topic_families")
            if isinstance(raw_topic_families, Mapping):
                for raw_family, raw_family_entry in raw_topic_families.items():
                    family = _canonical_provider_term(raw_family)
                    if not family or not isinstance(raw_family_entry, Mapping):
                        continue
                    topic_families[family] = {
                        "commits": _safe_int(raw_family_entry.get("commits", 0)),
                        "successes": _safe_int(raw_family_entry.get("successes", 0)),
                        "semantic_relevance_ema": _safe_float(raw_family_entry.get("semantic_relevance_ema", 0.0)),
                        "answerability_gain_ema": _safe_float(raw_family_entry.get("answerability_gain_ema", 0.0)),
                        "uncertainty_reduction_ema": _safe_float(
                            raw_family_entry.get("uncertainty_reduction_ema", 0.0)
                        ),
                        "weak_concept_stabilization_ema": _safe_float(
                            raw_family_entry.get("weak_concept_stabilization_ema", 0.0)
                        ),
                        "last_selected_at": " ".join(
                            str(raw_family_entry.get("last_selected_at", "")).split()
                        ).strip(),
                    }
            query_families: dict[str, dict[str, Any]] = {}
            raw_query_families = raw_entry.get("query_families")
            if isinstance(raw_query_families, Mapping):
                for raw_family, raw_family_entry in raw_query_families.items():
                    family = _canonical_provider_term(raw_family)
                    if not family or not isinstance(raw_family_entry, Mapping):
                        continue
                    query_families[family] = {
                        "commits": _safe_int(raw_family_entry.get("commits", 0)),
                        "successes": _safe_int(raw_family_entry.get("successes", 0)),
                        "semantic_relevance_ema": _safe_float(raw_family_entry.get("semantic_relevance_ema", 0.0)),
                        "answerability_gain_ema": _safe_float(raw_family_entry.get("answerability_gain_ema", 0.0)),
                        "uncertainty_reduction_ema": _safe_float(
                            raw_family_entry.get("uncertainty_reduction_ema", 0.0)
                        ),
                        "weak_concept_stabilization_ema": _safe_float(
                            raw_family_entry.get("weak_concept_stabilization_ema", 0.0)
                        ),
                        "last_selected_at": " ".join(
                            str(raw_family_entry.get("last_selected_at", "")).split()
                        ).strip(),
                    }
            normalized[provider] = {
                "attempts": _safe_int(raw_entry.get("attempts", 0)),
                "commits": _safe_int(raw_entry.get("commits", 0)),
                "successes": _safe_int(raw_entry.get("successes", 0)),
                "gap_gain_ema": _safe_float(raw_entry.get("gap_gain_ema", 0.0)),
                "diagnostic_gain_ema": _safe_float(raw_entry.get("diagnostic_gain_ema", 0.0)),
                "semantic_relevance_ema": _safe_float(raw_entry.get("semantic_relevance_ema", 0.0)),
                "answerability_gain_ema": _safe_float(raw_entry.get("answerability_gain_ema", 0.0)),
                "uncertainty_reduction_ema": _safe_float(raw_entry.get("uncertainty_reduction_ema", 0.0)),
                "weak_concept_stabilization_ema": _safe_float(
                    raw_entry.get("weak_concept_stabilization_ema", 0.0)
                ),
                "utility_ema": _safe_float(raw_entry.get("utility_ema", 0.0)),
                "focus_alignment_ema": _safe_float(raw_entry.get("focus_alignment_ema", 0.0)),
                "grounded_outcome_ema": _safe_float(raw_entry.get("grounded_outcome_ema", 0.0)),
                "grounded_family_summary_ema": _safe_float(raw_entry.get("grounded_family_summary_ema", 0.0)),
                "delayed_consequence_ema": _safe_float(raw_entry.get("delayed_consequence_ema", 0.0)),
                "contradiction_decay_ema": _safe_float(raw_entry.get("contradiction_decay_ema", 0.0)),
                "last_query_text": " ".join(str(raw_entry.get("last_query_text", "")).split()).strip(),
                "last_selected_at": " ".join(str(raw_entry.get("last_selected_at", "")).split()).strip(),
                "topic_terms": dict(
                    sorted(
                        topic_terms.items(),
                        key=lambda item: (-float(item[1]), item[0]),
                    )[:AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT]
                ),
                "topic_families": dict(
                    sorted(
                        topic_families.items(),
                        key=lambda item: (-self._provider_topic_family_priority_locked(item[1]), item[0]),
                    )[:AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT]
                ),
                "query_families": dict(
                    sorted(
                        query_families.items(),
                        key=lambda item: (-self._provider_query_family_priority_locked(item[1]), item[0]),
                    )[:AUTO_REMOTE_PROVIDER_QUERY_FAMILY_LIMIT]
                ),
            }
        return normalized

    def _default_autonomy_candidate_bank(self) -> list[dict[str, Any]]:
        return [
            self._normalize_catalog_candidate_spec(
                {
                    "name": "autonomy_live_remote_search",
                    "catalog_mode": "live_remote_search",
                    "catalog_providers": list(DEFAULT_AUTONOMY_REMOTE_PROVIDERS),
                    "catalog_queries_per_provider": DEFAULT_AUTONOMY_REMOTE_QUERIES_PER_PROVIDER,
                    "catalog_provider_result_limit": DEFAULT_AUTONOMY_REMOTE_PROVIDER_RESULT_LIMIT,
                    "catalog_limit": DEFAULT_AUTONOMY_REMOTE_CATALOG_LIMIT,
                    "catalog_probe_pool_limit": DEFAULT_AUTONOMY_REMOTE_PROBE_POOL_LIMIT,
                },
                0,
            )
        ]

    def _normalize_autonomy_config(self, autonomy: Any) -> dict[str, Any] | None:
        if autonomy is None:
            return None
        if not isinstance(autonomy, dict):
            raise ValueError("Terminus autonomy configuration must be an object")
        candidate_specs = [
            self._normalize_autonomy_candidate_spec(item, index)
            for index, item in enumerate(list(autonomy.get("candidate_bank") or []))
        ]
        enabled = bool(autonomy.get("enabled", bool(candidate_specs)))
        using_default_remote_search = False
        if enabled and not candidate_specs:
            candidate_specs = self._default_autonomy_candidate_bank()
            using_default_remote_search = True
        policy = str(autonomy.get("policy", "active")).strip() or "active"
        if policy not in PUBLIC_ACQUISITION_POLICIES:
            raise ValueError(
                "Unsupported Terminus autonomy policy. "
                f"Supported policies: {', '.join(PUBLIC_ACQUISITION_POLICIES)}"
            )
        shortlist_size_raw = autonomy.get("semantic_shortlist_size")
        shortlist_gap_weight_raw = autonomy.get("semantic_shortlist_gap_weight")
        shortlist_affinity_weight_raw = autonomy.get("semantic_shortlist_affinity_weight")
        if using_default_remote_search:
            shortlist_size = max(
                1,
                int(1 if shortlist_size_raw in (None, 0, "0") else shortlist_size_raw),
            )
            if shortlist_gap_weight_raw in (None, 0.5, "0.5") and shortlist_affinity_weight_raw in (None, 0.5, "0.5"):
                shortlist_gap_weight = 0.0
                shortlist_affinity_weight = 1.0
            else:
                shortlist_gap_weight = float(
                    0.0 if shortlist_gap_weight_raw in (None, "", "None") else shortlist_gap_weight_raw
                )
                shortlist_affinity_weight = float(
                    1.0 if shortlist_affinity_weight_raw in (None, "", "None") else shortlist_affinity_weight_raw
                )
        else:
            shortlist_size = max(0, int(autonomy.get("semantic_shortlist_size", 0)))
            shortlist_gap_weight = float(autonomy.get("semantic_shortlist_gap_weight", 0.5))
            shortlist_affinity_weight = float(autonomy.get("semantic_shortlist_affinity_weight", 0.5))
        return {
            "enabled": enabled,
            "policy": policy,
            "candidate_bank": candidate_specs,
            "provider_curriculum": self._normalize_provider_curriculum(autonomy.get("provider_curriculum")),
            "trigger_interval_tokens": max(
                1,
                int(autonomy.get("trigger_interval_tokens", DEFAULT_AUTONOMY_TRIGGER_INTERVAL_TOKENS)),
            ),
            "candidate_train_tokens": max(1, int(autonomy.get("candidate_train_tokens", 768))),
            "probe_tokens": max(1, int(autonomy.get("probe_tokens", 96))),
            "acquisition_tokens": max(1, int(autonomy.get("acquisition_tokens", 512))),
            "acquisition_slots": max(1, int(autonomy.get("acquisition_slots", 1))),
            "gap_exploration_bonus": float(autonomy.get("gap_exploration_bonus", 0.03)),
            "gap_ambiguity_weight": float(autonomy.get("gap_ambiguity_weight", 0.4)),
            "gap_switch_weight": float(autonomy.get("gap_switch_weight", 0.2)),
            "gap_margin_reference": float(autonomy.get("gap_margin_reference", 0.12)),
            "coverage_balance_penalty": float(autonomy.get("coverage_balance_penalty", 0.2)),
            "gap_focus_margin": float(autonomy.get("gap_focus_margin", 0.05)),
            "scout_commit_tokens": max(0, int(autonomy.get("scout_commit_tokens", 0))),
            "scout_top_k": max(1, int(autonomy.get("scout_top_k", 1))),
            "semantic_shortlist_size": shortlist_size,
            "semantic_shortlist_gap_weight": shortlist_gap_weight,
            "semantic_shortlist_affinity_weight": shortlist_affinity_weight,
        }

    def _normalize_brain_config(self, config: Any) -> dict[str, Any]:
        if config is None:
            tick_tokens = DEFAULT_BRAIN_TICK_TOKENS
            return {
                "source_bank": [],
                "tick_tokens": tick_tokens,
                "sleep_interval_seconds": DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS,
                "repeat_sources": True,
                "autonomy": None,
                "sensory": None,
                "ingestion": self._normalize_ingestion_config(None, tick_tokens=tick_tokens),
            }
        if not isinstance(config, dict):
            raise ValueError("Terminus runtime configuration must be an object")
        source_bank = [
            self._normalize_brain_source_spec(item, index)
            for index, item in enumerate(list(config.get("source_bank") or []))
        ]
        tick_tokens = max(1, int(config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS)))
        normalized = {
            "source_bank": source_bank,
            "tick_tokens": tick_tokens,
            "sleep_interval_seconds": max(
                0.01,
                float(config.get("sleep_interval_seconds", DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS)),
            ),
            "repeat_sources": bool(config.get("repeat_sources", True)),
            "autonomy": self._normalize_autonomy_config(config.get("autonomy")),
            "sensory": self._normalize_sensory_config(config.get("sensory")),
            "ingestion": self._normalize_ingestion_config(config.get("ingestion"), tick_tokens=tick_tokens),
        }
        return normalized

    @staticmethod
    def _normalize_ingestion_config(config: Any, *, tick_tokens: int) -> dict[str, Any]:
        raw = config if isinstance(config, dict) else {}
        enabled = bool(raw.get("enabled", True))
        default_queue_target = max(
            int(tick_tokens),
            int(tick_tokens) * DEFAULT_INGESTION_QUEUE_MULTIPLIER,
        )
        queue_target_tokens = max(
            int(tick_tokens),
            int(raw.get("queue_target_tokens", default_queue_target)),
        )
        return {
            "enabled": enabled,
            "queue_target_tokens": queue_target_tokens,
            "prewarm_on_startup": bool(raw.get("prewarm_on_startup", False)),
            "prewarm_max_seconds": max(0.05, float(raw.get("prewarm_max_seconds", 5.0))),
        }

    def _normalize_sensory_config(self, config: Any) -> dict[str, Any] | None:
        if config is None or not isinstance(config, dict):
            return None
        if not config.get("enabled"):
            return None
        source_bank = [
            self._normalize_sensory_source_spec(item, index)
            for index, item in enumerate(list(config.get("source_bank") or []))
        ]
        if not source_bank:
            return None
        base_windows = max(1, int(config.get("base_windows_per_item", 4)))
        max_windows = max(base_windows, int(config.get("max_windows_per_item", 10)))
        items_per_episode = max(1, int(config.get("items_per_episode", 2)))
        lookahead = max(1, int(config.get("item_retrieval_lookahead", 6)))
        queue_target_items = max(
            1,
            int(config.get("queue_target_items", max(items_per_episode, lookahead))),
        )
        return {
            "enabled": True,
            "source_bank": source_bank,
            "episode_interval_tokens": max(256, int(config.get("episode_interval_tokens", 1536))),
            "items_per_episode": items_per_episode,
            "base_windows_per_item": base_windows,
            "max_windows_per_item": max_windows,
            "confidence_window_gain": max(0.0, float(config.get("confidence_window_gain", 3.0))),
            "semantic_window_gain": max(0.0, float(config.get("semantic_window_gain", 3.0))),
            "item_retrieval_lookahead": lookahead,
            "item_retrieval_semantic_weight": max(0.0, min(1.0, float(config.get("item_retrieval_semantic_weight", 0.72)))),
            "modality_target_confidence": max(0.1, min(1.0, float(config.get("modality_target_confidence", 0.70)))),
            "observation_salience": max(0.1, min(1.0, float(config.get("observation_salience", 0.82)))),
            "cooldown_seconds": max(1.0, float(config.get("cooldown_seconds", 8.0))),
            "repeat_sources": bool(config.get("repeat_sources", True)),
            "queue_target_items": queue_target_items,
            "prewarm_on_startup": bool(config.get("prewarm_on_startup", False)),
            "prewarm_max_seconds": max(0.05, float(config.get("prewarm_max_seconds", 5.0))),
        }

    @staticmethod
    def _wrap_remote_stream(spec: Mapping[str, Any], stream: Iterator[Any], *, is_sensory: bool) -> Iterator[Any]:
        uses_remote = (
            HECSNServiceManager._sensory_spec_uses_live_remote(spec)
            if is_sensory
            else HECSNServiceManager._source_spec_uses_live_remote(spec)
        )
        if not uses_remote:
            return stream
        name = str(spec.get("name", "sensory" if is_sensory else "source"))
        return BackgroundPrefetchIterator(
            stream,
            max_buffer=DEFAULT_REMOTE_STREAM_PREFETCH_ITEMS,
            name=name,
        )

    @staticmethod
    def _stream_supports_ready_reads(stream: Iterator[Any]) -> bool:
        return callable(getattr(stream, "next_ready", None))

    @staticmethod
    def _next_stream_item(stream: Iterator[Any], *, timeout: float | None = None) -> Any:
        next_ready = getattr(stream, "next_ready", None)
        if callable(next_ready):
            return next_ready(timeout=timeout)
        return next(stream)

    def _build_brain_source_stream_locked(self, spec: dict[str, Any]) -> Iterator[tuple[str, torch.Tensor]]:
        source_type = str(spec.get("source_type", "auto"))
        loader = StreamingCorpusLoader(
            source=str(spec.get("source", "")),
            source_type=source_type,
            text_field=str(spec.get("text_field", "text")),
            hf_config=spec.get("hf_config"),
        )
        stream = labeled_pattern_stream(
            loader.char_stream(),
            self._encoder,
            self._trainer.config.window_size,
            learn_chunking=True,
        )
        return cast(Iterator[tuple[str, torch.Tensor]], self._wrap_remote_stream(spec, stream, is_sensory=False))

    def _build_sensory_stream_locked(self, spec: dict[str, Any]) -> Iterator[SensoryEpisode]:
        return self._build_sensory_stream_from_spec(
            spec,
            visual_dim=int(getattr(self._trainer.config, "cross_modal_dim_visual", 64)),
            audio_dim=int(getattr(self._trainer.config, "cross_modal_dim_audio", 64)),
            device=self._trainer.model.device,
        )

    @staticmethod
    def _build_sensory_stream_from_spec(
        spec: dict[str, Any],
        *,
        visual_dim: int,
        audio_dim: int,
        device: torch.device,
    ) -> Iterator[SensoryEpisode]:
        stream = build_sensory_stream(
            spec,
            visual_dim=int(visual_dim),
            audio_dim=int(audio_dim),
            device=device,
        )
        return cast(Iterator[SensoryEpisode], HECSNServiceManager._wrap_remote_stream(spec, stream, is_sensory=True))

    def _runtime_cache_root(self) -> Path:
        root = self._checkpoint_dir / "runtime_cache"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _runtime_cache_key(self, *, kind: str, spec: Mapping[str, Any]) -> str:
        payload = {
            "kind": str(kind),
            "checkpoint": str(self._checkpoint_path.resolve()),
            "window_size": int(self._trainer.config.window_size),
            "visual_dim": int(getattr(self._trainer.config, "cross_modal_dim_visual", 64)),
            "audio_dim": int(getattr(self._trainer.config, "cross_modal_dim_audio", 64)),
            "spec": dict(spec),
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _brain_runtime_cache_path(self, spec: Mapping[str, Any]) -> Path:
        return self._runtime_cache_root() / f"brain_{self._runtime_cache_key(kind='brain', spec=spec)}.pt"

    def _sensory_runtime_cache_path(self, spec: Mapping[str, Any]) -> Path:
        return self._runtime_cache_root() / f"sensory_{self._runtime_cache_key(kind='sensory', spec=spec)}.pt"

    @staticmethod
    def _reconstruct_text_from_windows(raw_windows: Sequence[str]) -> str:
        windows = [str(item) for item in raw_windows if str(item)]
        if not windows:
            return ""
        reconstructed = windows[0]
        for window in windows[1:]:
            max_overlap = min(len(reconstructed), len(window))
            overlap = 0
            for size in range(max_overlap, 0, -1):
                if reconstructed.endswith(window[:size]):
                    overlap = size
                    break
            reconstructed += window[overlap:]
        return reconstructed

    def _update_brain_runtime_cache_locked(
        self,
        runtime: _BrainSourceRuntime,
        *,
        served_examples: Sequence[tuple[str, torch.Tensor]] | None = None,
    ) -> None:
        if not self._source_spec_uses_live_remote(runtime.spec):
            return
        ingestion = self._brain_config.get("ingestion") or {}
        tick_tokens = int(self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS))
        target_tokens = max(int(tick_tokens), int(ingestion.get("queue_target_tokens", tick_tokens)))
        raw_windows: list[str] = []
        if served_examples:
            raw_windows.extend(str(raw_window) for raw_window, _pattern in served_examples if str(raw_window))
        raw_windows.extend(str(raw_window) for raw_window, _pattern in list(runtime.buffered_patterns) if str(raw_window))
        raw_windows = raw_windows[: max(1, target_tokens)]
        if not raw_windows:
            return
        payload = {
            "raw_windows": raw_windows,
            "token_count": int(len(raw_windows)),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            torch.save(payload, self._brain_runtime_cache_path(runtime.spec))
        except Exception:
            return

    def _restore_brain_runtime_cache_locked(self, runtime: _BrainSourceRuntime) -> int:
        if not self._source_spec_uses_live_remote(runtime.spec):
            return 0
        path = self._brain_runtime_cache_path(runtime.spec)
        if not path.exists():
            return 0
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            return 0
        raw_windows = [str(item) for item in list((payload or {}).get("raw_windows") or []) if str(item)]
        token_count = max(0, int((payload or {}).get("token_count", len(raw_windows)) or 0))
        if not raw_windows or token_count <= 0:
            return 0
        text = self._reconstruct_text_from_windows(raw_windows)
        if not text:
            return 0
        examples: list[tuple[str, torch.Tensor]] = []
        for raw_window, pattern in labeled_pattern_stream(
            text,
            self._encoder,
            self._trainer.config.window_size,
            learn_chunking=False,
        ):
            examples.append((raw_window, pattern))
            if len(examples) >= token_count:
                break
        if not examples:
            return 0
        runtime.buffered_patterns = deque(examples)
        return int(len(examples))

    def _serialize_sensory_episode(self, episode: SensoryEpisode) -> dict[str, Any]:
        return {
            "text": str(episode.text),
            "visual_spikes": None if episode.visual_spikes is None else episode.visual_spikes.detach().cpu(),
            "audio_spikes": None if episode.audio_spikes is None else episode.audio_spikes.detach().cpu(),
            "metadata": deepcopy(episode.metadata),
            "visual_preview": deepcopy(episode.visual_preview),
            "audio_preview": deepcopy(episode.audio_preview),
        }

    def _deserialize_sensory_episode(self, payload: Mapping[str, Any]) -> SensoryEpisode:
        visual_spikes = payload.get("visual_spikes")
        audio_spikes = payload.get("audio_spikes")
        device = self._trainer.model.device
        if isinstance(visual_spikes, torch.Tensor):
            visual_spikes = visual_spikes.to(device)
        else:
            visual_spikes = None
        if isinstance(audio_spikes, torch.Tensor):
            audio_spikes = audio_spikes.to(device)
        else:
            audio_spikes = None
        return SensoryEpisode(
            text=str(payload.get("text", "")),
            visual_spikes=visual_spikes,
            audio_spikes=audio_spikes,
            metadata=deepcopy(dict(payload.get("metadata") or {})),
            visual_preview=deepcopy(payload.get("visual_preview")),
            audio_preview=deepcopy(payload.get("audio_preview")),
        )

    def _update_sensory_runtime_cache_locked(
        self,
        runtime: _SensorySourceRuntime,
        *,
        served_episodes: Sequence[SensoryEpisode] | None = None,
    ) -> None:
        if not self._sensory_spec_uses_live_remote(runtime.spec):
            return
        target_items = self._sensory_queue_target_items_locked()
        episodes: list[SensoryEpisode] = []
        if served_episodes:
            episodes.extend(served_episodes)
        episodes.extend(list(runtime.buffered_episodes))
        episodes = episodes[: max(1, target_items)]
        if not episodes:
            return
        payload = {
            "episodes": [self._serialize_sensory_episode(item) for item in episodes],
            "item_count": int(len(episodes)),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            torch.save(payload, self._sensory_runtime_cache_path(runtime.spec))
        except Exception:
            return

    def _restore_sensory_runtime_cache_locked(self, runtime: _SensorySourceRuntime) -> int:
        if not self._sensory_spec_uses_live_remote(runtime.spec):
            return 0
        path = self._sensory_runtime_cache_path(runtime.spec)
        if not path.exists():
            return 0
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            return 0
        raw_episodes = list((payload or {}).get("episodes") or [])
        if not raw_episodes:
            return 0
        restored: list[SensoryEpisode] = []
        for item in raw_episodes:
            if not isinstance(item, Mapping):
                continue
            restored.append(self._deserialize_sensory_episode(item))
        if not restored:
            return 0
        runtime.buffered_episodes = list(restored)
        return int(len(restored))

    @staticmethod
    def _close_runtime_streams(runtimes: Sequence[Any]) -> None:
        for runtime in runtimes:
            close = getattr(getattr(runtime, "stream", None), "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    continue

    def _interrupt_brain_sources_locked(self) -> None:
        self._close_runtime_streams(self._brain_source_runtimes)

    def _interrupt_sensory_sources_locked(self) -> None:
        self._close_runtime_streams(self._sensory_source_runtimes)

    def _close_brain_sources_locked(self) -> None:
        self._interrupt_brain_sources_locked()
        self._brain_source_runtimes = []

    def _close_sensory_sources_locked(self) -> None:
        self._interrupt_sensory_sources_locked()
        self._sensory_source_runtimes = []

    def _focus_gap_terms_locked(self, limit: int = 4) -> list[str]:
        terms: list[str] = []

        exploration_target = ""
        if self._thought_loop is not None and hasattr(self._thought_loop, "gate"):
            exploration_target = str(getattr(self._thought_loop.gate, "active_exploration_target", "")).strip()
        if exploration_target:
            terms.append(exploration_target)

        try:
            plan = self._geometric_curiosity.focus_plan(top_n=max(1, limit))
            for item in list((plan or {}).get("geometric_gaps", []))[:limit]:
                concept = " ".join(str(item.get("concept", "")).split()).strip()
                if concept:
                    terms.append(concept)
        except Exception:
            pass

        try:
            snap = self._concept_store.snapshot(limit=max(1, limit))
            for concept in list(snap.get("top_concepts", []))[:limit]:
                if not isinstance(concept, dict):
                    continue
                label = " ".join(str(concept.get("label", "")).split()).strip()
                if label:
                    terms.append(label)
                for term in list(concept.get("top_terms", []))[:2]:
                    cleaned = " ".join(str(term).split()).strip()
                    if cleaned:
                        terms.append(cleaned)
        except Exception:
            pass

        normalized: list[str] = []
        seen: set[str] = set()
        for term in terms:
            cleaned = " ".join(term.replace("/", " ").replace("|", " ").split()).strip().lower()
            if cleaned and cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)
            if len(normalized) >= max(1, limit):
                break
        return normalized

    def _background_focus_terms_locked(
        self,
        limit: int = 12,
        *,
        focus_plan: Mapping[str, Any] | None = None,
    ) -> list[str]:
        plan = focus_plan if focus_plan is not None else self._autonomy_focus_plan_locked()
        phrases: list[str] = []
        if isinstance(plan, Mapping):
            phrases.extend(str(item) for item in list(plan.get("query_terms") or []) if str(item).strip())
            phrases.extend(str(item) for item in list(plan.get("unsupported_terms") or []) if str(item).strip())
            phrases.extend(
                str(item.get("term", ""))
                for item in list(plan.get("gap_terms") or [])
                if isinstance(item, Mapping) and str(item.get("term", "")).strip()
            )
            phrases.extend(str(item) for item in list(plan.get("retrieval_queries") or []) if str(item).strip())
            for raw_concept in list(plan.get("weak_concepts") or []):
                if not isinstance(raw_concept, Mapping):
                    continue
                phrases.append(str(raw_concept.get("label", "")))
                phrases.extend(
                    str(item)
                    for item in list(raw_concept.get("top_terms") or [])
                    if str(item).strip()
                )
        if not phrases and self._brain_recent_query_gaps:
            recent_gap = self._brain_recent_query_gaps[0]
            phrases.append(str(recent_gap.get("query_text", "")))
            phrases.extend(str(term) for term in list(recent_gap.get("unsupported_terms") or [])[:4])
        if not phrases:
            phrases.extend(self._focus_gap_terms_locked(limit=max(4, limit // 2)))

        ordered: list[str] = []
        seen: set[str] = set()
        for phrase in phrases:
            for term in salient_query_terms(str(phrase)):
                cleaned = _canonical_provider_term(term)
                if len(cleaned) < 4 or cleaned in seen:
                    continue
                seen.add(cleaned)
                ordered.append(cleaned)
                if len(ordered) >= max(1, limit):
                    return ordered
        return ordered

    @staticmethod
    def _brain_source_memory_metadata(runtime: _BrainSourceRuntime) -> dict[str, Any]:
        metadata = runtime.spec.get("metadata") if isinstance(runtime.spec.get("metadata"), Mapping) else {}
        topic_terms = [
            _canonical_provider_term(term)
            for term in list(runtime.spec.get("topic_terms") or [])
            if _canonical_provider_term(term)
        ]
        catalog_terms = [str(term) for term in topic_terms]
        raw_catalog_terms = metadata.get("catalog_terms") if isinstance(metadata, Mapping) else None
        if isinstance(raw_catalog_terms, Sequence) and not isinstance(raw_catalog_terms, (str, bytes)):
            catalog_terms = list(
                dict.fromkeys(
                    [
                        *catalog_terms,
                        *[
                            _canonical_provider_term(term)
                            for term in list(raw_catalog_terms)
                            if _canonical_provider_term(term)
                        ],
                    ]
                )
            )
        memory_metadata: dict[str, Any] = {
            "observation_kind": "source",
            "source_name": runtime.name,
            "source_type": runtime.source_type,
            "source": str(runtime.spec.get("source", "")),
            "catalog_terms": catalog_terms[:8],
        }
        for key in ("provider", "query_text", "catalog_title", "catalog_summary"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                memory_metadata[str(key)] = value.strip()
        return memory_metadata

    @staticmethod
    def _brain_source_topic_terms(runtime: _BrainSourceRuntime) -> set[str]:
        terms: set[str] = set()
        for raw in list(runtime.spec.get("topic_terms") or []):
            for term in salient_query_terms(str(raw)):
                cleaned = _canonical_provider_term(term)
                if len(cleaned) >= 4:
                    terms.add(cleaned)
        for raw in (
            runtime.spec.get("name", ""),
            runtime.spec.get("source", ""),
            runtime.spec.get("hf_config", ""),
        ):
            for term in salient_query_terms(str(raw)):
                cleaned = _canonical_provider_term(term)
                if len(cleaned) >= 4:
                    terms.add(cleaned)
        metadata = runtime.spec.get("metadata")
        if isinstance(metadata, Mapping):
            for key in ("role", "label", "why", "summary", "description", "title"):
                for term in salient_query_terms(str(metadata.get(key, ""))):
                    cleaned = _canonical_provider_term(term)
                    if len(cleaned) >= 4:
                        terms.add(cleaned)
        return terms

    def _brain_source_semantic_match_locked(
        self,
        runtime: _BrainSourceRuntime,
        focus_terms: Sequence[str] | None = None,
    ) -> float:
        normalized_focus = [
            _canonical_provider_term(term)
            for term in list(focus_terms or self._background_focus_terms_locked())
            if _canonical_provider_term(term)
        ]
        source_terms = self._brain_source_topic_terms(runtime)
        if not normalized_focus or not source_terms:
            return 0.0
        focus_set = set(normalized_focus)
        overlap = len(focus_set & source_terms) / max(1.0, min(float(len(focus_set)), float(len(source_terms))))
        head_hits = sum(1 for term in normalized_focus[:3] if term in source_terms)
        head_bonus = min(1.0, 0.5 * head_hits)
        metadata = runtime.spec.get("metadata") if isinstance(runtime.spec.get("metadata"), Mapping) else {}
        combined_text = " ".join(
            part
            for part in [
                str(runtime.spec.get("name", "")),
                str(runtime.spec.get("source", "")),
                str(runtime.spec.get("hf_config", "")),
                *(str(metadata.get(key, "")) for key in ("role", "label", "why", "summary", "description", "title")),
            ]
            if part
        ).lower()
        phrase_hits = sum(1 for term in normalized_focus[:4] if term and term in combined_text)
        phrase_bonus = min(1.0, 0.34 * phrase_hits)
        return max(0.0, min(1.0, 0.55 * overlap + 0.30 * head_bonus + 0.15 * phrase_bonus))

    def _brain_source_selection_score_locked(
        self,
        runtime: _BrainSourceRuntime,
        *,
        focus_terms: Sequence[str],
        focus_pressure: float,
        tick_tokens: int,
    ) -> tuple[float, float, float, float, float]:
        semantic_match = self._brain_source_semantic_match_locked(runtime, focus_terms)
        source_count = max(1, len(self._brain_source_runtimes))
        min_visits = min((rt.tick_visits for rt in self._brain_source_runtimes), default=0)
        fairness = max(
            0.0,
            min(
                1.0,
                1.0 - max(0, runtime.tick_visits - min_visits) / float(source_count + 1),
            ),
        )
        readiness = max(
            0.0,
            min(1.0, float(len(runtime.buffered_patterns)) / float(max(1, int(tick_tokens)))),
        )
        freshness = 1.0 if runtime.last_activity_at is None else 0.0
        focus_factor = max(0.0, min(1.0, float(focus_pressure)))
        source_utility = self._background_source_utility_entry_locked(runtime)
        utility_ema = max(0.0, min(1.0, float(source_utility.get("utility_ema", 0.0))))
        grounded_family_summary = max(0.0, min(1.0, float(source_utility.get("grounded_family_summary_ema", 0.0))))
        contradiction_decay = max(0.0, min(1.0, float(source_utility.get("contradiction_decay_ema", 0.0))))
        net_utility = max(
            0.0,
            max(utility_ema, grounded_family_summary) - DEFAULT_UTILITY_PENALTY_WEIGHT * contradiction_decay,
        )
        effective_utility = float(
            max(net_utility, grounded_family_summary)
            * max(0.0, min(1.0, max(float(semantic_match), 0.35 * focus_factor)))
        )
        semantic_weight = 0.10 + 0.24 * focus_factor
        utility_weight = 0.04 + 0.34 * focus_factor
        fairness_weight = max(0.16, 0.54 - 0.43 * focus_factor)
        readiness_weight = 0.10
        freshness_weight = 0.10
        score = (
            semantic_weight * semantic_match
            + utility_weight * effective_utility
            + fairness_weight * fairness
            + readiness_weight * readiness
            + freshness_weight * freshness
        )
        runtime.last_semantic_match = float(semantic_match)
        runtime.last_selection_score = float(score)
        runtime.last_fairness_score = float(fairness)
        runtime.last_buffer_readiness = float(readiness)
        runtime.last_utility_score = float(effective_utility)
        return score, semantic_match, fairness, readiness, effective_utility

    def _background_focus_overlap_locked(
        self,
        focus_terms: Sequence[str],
        grounded_observation: Mapping[str, Any] | None,
    ) -> float:
        normalized_focus = [
            _canonical_provider_term(term)
            for term in list(focus_terms or [])
            if _canonical_provider_term(term)
        ]
        if not normalized_focus:
            return 0.0
        overlap_sources: list[str] = []
        if isinstance(grounded_observation, Mapping):
            overlap_sources.append(str(grounded_observation.get("content", "")))
            overlap_sources.extend(str(item) for item in list(grounded_observation.get("topics") or []) if str(item).strip())
        combined = " ".join(part for part in overlap_sources if part).strip()
        if not combined:
            return 0.0
        overlap = self._source_text_overlap(" ".join(normalized_focus), combined)
        phrase_hits = sum(1 for term in normalized_focus[:4] if term and term in combined.lower())
        phrase_bonus = min(1.0, 0.34 * phrase_hits)
        return max(0.0, min(1.0, 0.70 * overlap + 0.30 * phrase_bonus))

    def _update_background_source_utility_locked(
        self,
        *,
        runtime: _BrainSourceRuntime,
        grounded_observation: Mapping[str, Any] | None,
        total_trained: int,
    ) -> None:
        entry = self._background_source_utility_entry_locked(runtime)
        focus_plan = self._autonomy_focus_plan_locked()
        focus_terms = self._background_focus_terms_locked(focus_plan=focus_plan)
        semantic_alignment = max(0.0, min(1.0, float(runtime.last_semantic_match)))
        grounding_signal = 0.0
        if isinstance(grounded_observation, Mapping):
            grounding_signal = max(0.0, min(1.0, float(grounded_observation.get("grounding_signal", 0.0) or 0.0)))
        focus_overlap = self._background_focus_overlap_locked(focus_terms, grounded_observation)
        token_fraction = min(
            1.0,
            float(max(0, int(total_trained))) / float(max(1, int(self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS)))),
        )
        utility_sample = max(
            0.0,
            min(
                1.0,
                0.50 * semantic_alignment
                + 0.20 * grounding_signal
                + 0.20 * focus_overlap
                + 0.10 * token_fraction,
            ),
        )

        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        entry["selections"] = int(entry.get("selections", 0)) + 1
        entry["tokens_trained_total"] = int(entry.get("tokens_trained_total", 0)) + max(0, int(total_trained))
        alpha = 0.30
        for key, sample in (
            ("utility_ema", utility_sample),
            ("semantic_alignment_ema", semantic_alignment),
            ("grounding_signal_ema", grounding_signal),
            ("focus_overlap_ema", focus_overlap),
        ):
            previous = max(0.0, min(1.0, float(entry.get(key, 0.0) or 0.0)))
            entry[key] = float(sample if int(entry["selections"]) <= 1 else (1.0 - alpha) * previous + alpha * float(sample))
        entry["last_selected_at"] = datetime.now(timezone.utc).isoformat()
        self._mark_mutated()

    @staticmethod
    def _selected_evidence_weight_map(
        response: Mapping[str, Any],
        *,
        singular_field: str,
        plural_field: str,
    ) -> dict[str, float]:
        weighted: dict[str, float] = {}
        for rank, raw_item in enumerate(list(response.get("selected_evidence") or [])):
            if not isinstance(raw_item, Mapping):
                continue
            rank_weight = 1.0 / float(rank + 1)
            term_coverage = max(0.0, min(1.0, float(raw_item.get("term_coverage", 0.0) or 0.0)))
            score_weight = max(0.0, min(1.0, float(raw_item.get("score", 0.0) or 0.0)))
            item_weight = max(0.20, min(1.0, 0.40 * rank_weight + 0.30 * term_coverage + 0.30 * score_weight))
            names: list[str] = []
            single_value = " ".join(str(raw_item.get(singular_field, "")).split()).strip()
            if single_value:
                names.append(single_value)
            raw_values = raw_item.get(plural_field)
            if isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes)):
                names.extend(" ".join(str(item).split()).strip() for item in list(raw_values) if " ".join(str(item).split()).strip())
            for name in names:
                key = name.lower() if singular_field == "provider" else name
                weighted[key] = max(float(weighted.get(key, 0.0)), float(item_weight))
        return weighted

    def _response_grounded_outcome_score_locked(
        self,
        *,
        query_result: Mapping[str, Any],
        response: Mapping[str, Any],
        action_assist: Mapping[str, Any] | None,
    ) -> float:
        gap_plan = query_result.get("gap_plan") if isinstance(query_result.get("gap_plan"), Mapping) else {}
        grounded_fraction = max(0.0, min(1.0, float(gap_plan.get("grounded_fraction", 0.0) or 0.0)))
        evidence_coverage = max(0.0, min(1.0, float(response.get("evidence_coverage", 0.0) or 0.0)))
        selected_evidence_count = int(len(list(response.get("selected_evidence") or [])))
        selected_evidence_bonus = min(1.0, float(selected_evidence_count) / 2.0)
        response_mode = self._normalize_action_text(response.get("response_mode", "")).lower()
        unsupported_terms = [
            str(item).strip().lower()
            for item in list(response.get("unsupported_terms") or gap_plan.get("unsupported_terms") or [])
            if str(item).strip()
        ]
        unsupported_penalty = min(1.0, float(len(unsupported_terms)) / 4.0)
        score = max(
            0.0,
            min(
                1.0,
                0.36 * grounded_fraction
                + 0.34 * evidence_coverage
                + 0.15 * selected_evidence_bonus
                + 0.15 * (0.0 if response_mode == "insufficient_evidence" else 1.0)
                - 0.25 * unsupported_penalty,
            ),
        )
        if isinstance(action_assist, Mapping):
            record = action_assist.get("result") if isinstance(action_assist.get("result"), Mapping) else {}
            verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
            if bool(verification.get("success", False)):
                confidence = max(0.0, min(1.0, float(verification.get("confidence", 0.0) or 0.0)))
                score = max(score, max(0.0, min(1.0, 0.55 + 0.35 * confidence)))
            elif bool(verification.get("contradiction", False)):
                score *= 0.25
        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def _consequence_query_terms(value: Any) -> list[str]:
        normalized_text = " ".join(str(value).split()).strip()
        if not normalized_text:
            return []
        ordered: list[str] = []
        seen: set[str] = set()
        for raw_term in salient_query_terms(normalized_text):
            term = _canonical_provider_term(raw_term)
            if not term or term in seen:
                continue
            seen.add(term)
            ordered.append(term)
            if len(ordered) >= 8:
                break
        return ordered

    def _query_progress_snapshot_locked(
        self,
        query_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        query_summary = query_result.get("query_summary") if isinstance(query_result.get("query_summary"), Mapping) else {}
        gap_plan = query_result.get("gap_plan") if isinstance(query_result.get("gap_plan"), Mapping) else {}
        query_text = self._normalize_action_text(query_summary.get("query_text", ""))
        query_terms = [
            _canonical_provider_term(term)
            for term in list(gap_plan.get("query_terms") or [])
            if _canonical_provider_term(term)
        ]
        if not query_terms:
            query_terms = self._consequence_query_terms(query_text)
        candidate_items = list(query_summary.get("memory_episodes") or query_summary.get("memory_matches") or [])
        query_term_count = max(1, len(query_terms))
        top_similarity = 0.0
        top_query_overlap_ratio = 0.0
        supported_episode_hits = 0
        for raw_item in candidate_items[:3]:
            if not isinstance(raw_item, Mapping):
                continue
            similarity = max(0.0, min(1.0, float(raw_item.get("similarity", 0.0) or 0.0)))
            query_overlap = max(0, int(raw_item.get("query_overlap", 0) or 0))
            top_similarity = max(top_similarity, similarity)
            top_query_overlap_ratio = max(
                top_query_overlap_ratio,
                min(1.0, float(query_overlap) / float(query_term_count)),
            )
            if query_overlap > 0:
                supported_episode_hits += 1
        grounded_fraction = max(0.0, min(1.0, float(gap_plan.get("grounded_fraction", 0.0) or 0.0)))
        support_episode_bonus = min(1.0, float(supported_episode_hits) / 2.0)
        query_score = max(
            0.0,
            min(
                1.0,
                0.60 * grounded_fraction
                + 0.20 * top_query_overlap_ratio
                + 0.10 * top_similarity
                + 0.10 * support_episode_bonus,
            ),
        )
        return {
            "query_text": query_text,
            "query_terms": list(query_terms),
            "grounded_fraction": float(grounded_fraction),
            "query_score": float(query_score),
            "top_similarity": float(top_similarity),
            "top_query_overlap_ratio": float(top_query_overlap_ratio),
            "supported_episode_hits": int(supported_episode_hits),
            "memory_episode_count": int(len(candidate_items)),
            "unsupported_terms": [
                _canonical_provider_term(term)
                for term in list(gap_plan.get("unsupported_terms") or [])
                if _canonical_provider_term(term)
            ],
        }

    @staticmethod
    def _delayed_consequence_query_examples(record: Mapping[str, Any]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()

        def _append(raw_value: Any) -> None:
            text = " ".join(str(raw_value).split()).strip()
            if not text:
                return
            key = text.lower()
            if key in seen:
                return
            seen.add(key)
            ordered.append(text)

        _append(record.get("query_text", ""))
        raw_examples = record.get("query_examples")
        if isinstance(raw_examples, Sequence) and not isinstance(raw_examples, (str, bytes)):
            for raw_value in list(raw_examples):
                _append(raw_value)
                if len(ordered) >= DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT:
                    break
        return ordered[:DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT]

    def _delayed_consequence_match_score_locked(
        self,
        record: Mapping[str, Any],
        query_snapshot: Mapping[str, Any],
    ) -> float:
        record_queries = [text.lower() for text in self._delayed_consequence_query_examples(record) if text]
        current_query = self._normalize_action_text(query_snapshot.get("query_text", "")).lower()
        if not record_queries or not current_query:
            return 0.0
        record_terms = {
            _canonical_provider_term(term)
            for term in list(record.get("query_terms") or [])
            if _canonical_provider_term(term)
        }
        if not record_terms:
            record_terms = {
                _canonical_provider_term(term)
                for query_text in record_queries
                for term in self._consequence_query_terms(query_text)
                if _canonical_provider_term(term)
            }
        current_terms = {
            _canonical_provider_term(term)
            for term in list(query_snapshot.get("query_terms") or self._consequence_query_terms(current_query))
            if _canonical_provider_term(term)
        }
        term_overlap = 0.0
        if record_terms and current_terms:
            term_overlap = float(len(record_terms & current_terms)) / float(max(1, min(len(record_terms), len(current_terms))))
        text_overlap = max(self._source_text_overlap(record_query, current_query) for record_query in record_queries)
        return max(0.0, min(1.0, max(term_overlap, 0.65 * term_overlap + 0.35 * text_overlap, text_overlap)))

    def _recent_action_contradiction_signal_locked(self, query_text: str) -> tuple[float, int]:
        contradicted_records = self._recent_relevant_action_records_locked(
            query_text,
            statuses=["contradicted"],
            limit=3,
        )
        if not contradicted_records:
            return 0.0, 0
        best_signal = 0.0
        for record in contradicted_records:
            verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
            confidence = max(0.0, min(1.0, float(verification.get("confidence", 0.0) or 0.0)))
            relevance = max(0.0, min(1.0, self._action_record_relevance_score_locked(record, query_text)))
            signal = max(0.0, min(1.0, relevance * max(0.35, 0.55 + 0.45 * confidence)))
            best_signal = max(best_signal, signal)
        return float(best_signal), int(len(contradicted_records))

    @staticmethod
    def _delayed_consequence_support_multiplier(record: Mapping[str, Any]) -> float:
        aggregate_count = max(1, int(record.get("aggregate_count", 1) or 1))
        if aggregate_count <= 1:
            return 1.0
        return float(
            min(
                DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_SUPPORT_MAX,
                1.0 + DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_SUPPORT_SCALE * math.log1p(float(aggregate_count - 1)),
            )
        )

    @staticmethod
    def _delayed_consequence_trajectory_totals(record: Mapping[str, Any]) -> tuple[float, float, float, float]:
        credit_total = max(0.0, float(record.get("trajectory_credit_total", 0.0) or 0.0))
        penalty_total = max(0.0, float(record.get("trajectory_penalty_total", 0.0) or 0.0))
        forgiveness_total = max(0.0, float(record.get("trajectory_forgiveness_total", 0.0) or 0.0))
        raw_net = float(record.get("trajectory_net_score", credit_total + forgiveness_total - penalty_total) or 0.0)
        net_score = max(
            -DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT,
            min(DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT, raw_net),
        )
        return float(credit_total), float(penalty_total), float(forgiveness_total), float(net_score)

    @classmethod
    def _delayed_consequence_trajectory_balance(cls, record: Mapping[str, Any]) -> float:
        credit_total, penalty_total, forgiveness_total, _net_score = cls._delayed_consequence_trajectory_totals(record)
        positive_total = max(0.0, credit_total + forgiveness_total)
        negative_total = max(0.0, penalty_total)
        total = positive_total + negative_total
        if total <= 1e-6:
            return 0.0
        return float(max(-1.0, min(1.0, (positive_total - negative_total) / total)))

    @staticmethod
    def _delayed_consequence_trajectory_recent_signal(record: Mapping[str, Any]) -> float:
        return float(max(-1.0, min(1.0, float(record.get("trajectory_recent_delta_ema", 0.0) or 0.0))))

    @classmethod
    def _delayed_consequence_trajectory_state(cls, record: Mapping[str, Any]) -> str:
        credit_total, penalty_total, forgiveness_total, net_score = cls._delayed_consequence_trajectory_totals(record)
        balance = cls._delayed_consequence_trajectory_balance(record)
        recent_signal = cls._delayed_consequence_trajectory_recent_signal(record)
        unresolved_penalty_balance = max(0.0, min(1.0, float(record.get("unresolved_penalty_balance", 0.0) or 0.0)))
        last_event_type = " ".join(str(record.get("last_trajectory_event_type", "")).split()).strip().lower()
        trajectory_floor = max(
            -DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT,
            min(
                DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT,
                float(record.get("trajectory_floor_score", net_score) or net_score),
            ),
        )
        if (credit_total + penalty_total + forgiveness_total) <= 1e-6:
            return "neutral"
        if (
            last_event_type in {"credit", "forgiveness"}
            and penalty_total > 0.0
            and unresolved_penalty_balance > 0.0
            and float(net_score) > float(trajectory_floor) + 0.05
        ):
            return "recovering"
        if (
            recent_signal >= DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_STATE_THRESHOLD
            and balance < 0.0
            and unresolved_penalty_balance > 0.0
        ):
            return "recovering"
        if balance >= DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_STATE_THRESHOLD:
            return "positive"
        if balance <= -DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_STATE_THRESHOLD:
            return "negative"
        return "mixed"

    @classmethod
    def _delayed_consequence_trajectory_support_multiplier(
        cls,
        record: Mapping[str, Any],
        *,
        mode: str,
    ) -> float:
        balance = cls._delayed_consequence_trajectory_balance(record)
        recent_signal = cls._delayed_consequence_trajectory_recent_signal(record)
        normalized_mode = " ".join(str(mode).split()).strip().lower()
        if normalized_mode == "penalty":
            aligned_signal = 0.70 * max(0.0, -balance) + 0.30 * max(0.0, -recent_signal)
            opposing_signal = 0.70 * max(0.0, balance) + 0.30 * max(0.0, recent_signal)
        else:
            aligned_signal = 0.70 * max(0.0, balance) + 0.30 * max(0.0, recent_signal)
            opposing_signal = 0.70 * max(0.0, -balance) + 0.30 * max(0.0, -recent_signal)
        factor = 1.0 + 0.18 * float(aligned_signal) - 0.10 * float(opposing_signal)
        return float(max(0.85, min(DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SUPPORT_MAX, factor)))

    @classmethod
    def _delayed_consequence_family_support_multiplier(
        cls,
        record: Mapping[str, Any],
        *,
        mode: str,
    ) -> float:
        aggregate_support = cls._delayed_consequence_support_multiplier(record)
        trajectory_support = cls._delayed_consequence_trajectory_support_multiplier(record, mode=mode)
        return float(
            max(
                0.80,
                min(
                    DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_SUPPORT_MAX * DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SUPPORT_MAX,
                    float(aggregate_support) * float(trajectory_support),
                ),
            )
        )

    @classmethod
    def _grounded_family_summary_score(cls, record: Mapping[str, Any]) -> float:
        best_grounded_fraction = max(0.0, min(1.0, float(record.get("best_grounded_fraction", 0.0) or 0.0)))
        baseline_grounded_fraction = max(
            0.0,
            min(1.0, float(record.get("baseline_grounded_fraction", 0.0) or 0.0)),
        )
        best_query_score = max(0.0, min(1.0, float(record.get("best_query_score", 0.0) or 0.0)))
        baseline_query_score = max(0.0, min(1.0, float(record.get("baseline_query_score", 0.0) or 0.0)))
        grounded_gain = max(0.0, best_grounded_fraction - baseline_grounded_fraction)
        query_gain = max(0.0, best_query_score - baseline_query_score)
        aggregate_count = max(1, int(record.get("aggregate_count", 1) or 1))
        aggregate_support = min(1.0, math.log1p(float(aggregate_count)) / math.log1p(4.0))
        trajectory_balance = cls._delayed_consequence_trajectory_balance(record)
        recent_signal = cls._delayed_consequence_trajectory_recent_signal(record)
        unresolved_penalty_balance = max(
            0.0,
            min(1.0, float(record.get("unresolved_penalty_balance", 0.0) or 0.0)),
        )
        trajectory_state = cls._delayed_consequence_trajectory_state(record)
        state_bonus = {
            "positive": 0.12,
            "recovering": 0.08,
            "mixed": 0.02,
            "negative": -0.12,
            "neutral": 0.0,
        }.get(trajectory_state, 0.0)
        split_branch = " ".join(str(record.get("split_branch", "")).split()).strip().lower()
        branch_bonus = 0.0
        if split_branch == "supportive":
            branch_bonus = 0.08
        elif split_branch == "adverse":
            branch_bonus = -0.12
        remerge_bonus = 0.08 if int(record.get("remerge_events", 0) or 0) > 0 else 0.0
        score = (
            0.26 * best_grounded_fraction
            + 0.18 * best_query_score
            + 0.16 * grounded_gain
            + 0.10 * query_gain
            + 0.10 * aggregate_support
            + 0.12 * max(0.0, trajectory_balance)
            + 0.08 * max(0.0, recent_signal)
            + float(state_bonus)
            + float(branch_bonus)
            + float(remerge_bonus)
            - 0.22 * unresolved_penalty_balance
            - 0.10 * max(0.0, -trajectory_balance)
        )
        return float(max(0.0, min(1.0, score)))

    def _update_delayed_consequence_trajectory_locked(
        self,
        record: dict[str, Any],
        *,
        event_type: str,
        event_score: float,
        timestamp: str,
        current_token: int,
    ) -> None:
        score = max(0.0, min(1.0, float(event_score)))
        if score <= 0.0:
            return
        credit_total, penalty_total, forgiveness_total, _net_score = self._delayed_consequence_trajectory_totals(record)
        normalized_event_type = " ".join(str(event_type).split()).strip().lower()
        signed_delta = score
        if normalized_event_type == "penalty":
            penalty_total += score
            signed_delta = -score
        elif normalized_event_type == "forgiveness":
            forgiveness_total += score
        else:
            credit_total += score
            normalized_event_type = "credit"
        raw_net = credit_total + forgiveness_total - penalty_total
        net_score = max(
            -DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT,
            min(DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT, raw_net),
        )
        previous_recent = self._delayed_consequence_trajectory_recent_signal(record)
        event_count = max(0, int(record.get("trajectory_event_count", 0) or 0)) + 1
        record["trajectory_credit_total"] = float(credit_total)
        record["trajectory_penalty_total"] = float(penalty_total)
        record["trajectory_forgiveness_total"] = float(forgiveness_total)
        record["trajectory_event_count"] = int(event_count)
        record["trajectory_net_score"] = float(net_score)
        alpha = float(DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_RECENT_ALPHA)
        record["trajectory_recent_delta_ema"] = float(
            signed_delta if event_count <= 1 else (1.0 - alpha) * float(previous_recent) + alpha * float(signed_delta)
        )
        record["trajectory_peak_score"] = float(
            max(float(record.get("trajectory_peak_score", net_score) or net_score), float(net_score))
        )
        record["trajectory_floor_score"] = float(
            min(float(record.get("trajectory_floor_score", net_score) or net_score), float(net_score))
        )
        record["last_trajectory_event_type"] = str(normalized_event_type)
        record["last_trajectory_event_score"] = float(score)
        record["last_trajectory_event_at"] = str(timestamp)
        record["last_trajectory_event_token_count"] = int(current_token)

    @staticmethod
    def _delayed_consequence_branch_examples(
        record: Mapping[str, Any],
        *,
        field: str,
    ) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        raw_values = record.get(field)
        if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
            return ordered
        for raw_value in list(raw_values):
            text = " ".join(str(raw_value).split()).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(text)
            if len(ordered) >= DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT:
                break
        return ordered

    def _update_delayed_consequence_branch_partition_locked(
        self,
        record: dict[str, Any],
        *,
        event_type: str,
        query_text: str,
    ) -> None:
        normalized_query = self._normalize_action_text(query_text)
        if not normalized_query:
            return
        normalized_event_type = " ".join(str(event_type).split()).strip().lower()
        if normalized_event_type == "penalty":
            field = "adverse_query_examples"
            count_field = "adverse_occurrence_count"
        else:
            field = "supportive_query_examples"
            count_field = "supportive_occurrence_count"
        examples = self._delayed_consequence_branch_examples(record, field=field)
        lowered = {item.lower() for item in examples}
        if normalized_query.lower() not in lowered:
            examples.append(normalized_query)
            examples = examples[-DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT :]
            record[field] = list(examples)
        elif field not in record:
            record[field] = list(examples)
        record[count_field] = max(int(record.get(count_field, 0) or 0), len(examples))

    def _delayed_consequence_query_text_overlap_locked(self, left_text: str, right_text: str) -> float:
        left = self._normalize_action_text(left_text).lower()
        right = self._normalize_action_text(right_text).lower()
        if not left or not right:
            return 0.0
        left_terms = {
            _canonical_provider_term(term)
            for term in self._consequence_query_terms(left)
            if _canonical_provider_term(term)
        }
        right_terms = {
            _canonical_provider_term(term)
            for term in self._consequence_query_terms(right)
            if _canonical_provider_term(term)
        }
        term_overlap = 0.0
        if left_terms and right_terms:
            term_overlap = float(len(left_terms & right_terms)) / float(max(1, min(len(left_terms), len(right_terms))))
        text_overlap = self._source_text_overlap(left, right)
        return float(max(0.0, min(1.0, max(term_overlap, 0.65 * term_overlap + 0.35 * text_overlap, text_overlap))))

    def _delayed_consequence_branch_overlap_locked(self, record: Mapping[str, Any]) -> float:
        supportive_examples = self._delayed_consequence_branch_examples(record, field="supportive_query_examples")
        adverse_examples = self._delayed_consequence_branch_examples(record, field="adverse_query_examples")
        if not supportive_examples or not adverse_examples:
            return 1.0
        return float(
            max(
                self._delayed_consequence_query_text_overlap_locked(left_text, right_text)
                for left_text in supportive_examples
                for right_text in adverse_examples
            )
        )

    def _build_delayed_consequence_split_child_locked(
        self,
        parent: Mapping[str, Any],
        *,
        branch: str,
        split_group_id: str,
        timestamp: str,
    ) -> dict[str, Any] | None:
        normalized_branch = " ".join(str(branch).split()).strip().lower()
        if normalized_branch not in {"supportive", "adverse"}:
            return None

        def _safe_int(raw_value: Any) -> int:
            try:
                return max(0, int(raw_value))
            except (TypeError, ValueError):
                return 0

        def _safe_float(raw_value: Any) -> float:
            try:
                return max(0.0, min(1.0, float(raw_value)))
            except (TypeError, ValueError):
                return 0.0

        supportive_examples = self._delayed_consequence_branch_examples(parent, field="supportive_query_examples")
        adverse_examples = self._delayed_consequence_branch_examples(parent, field="adverse_query_examples")
        supportive_count = max(1, int(parent.get("supportive_occurrence_count", 0) or 0), len(supportive_examples))
        adverse_count = max(1, int(parent.get("adverse_occurrence_count", 0) or 0), len(adverse_examples))
        trajectory_credit_total, trajectory_penalty_total, trajectory_forgiveness_total, _trajectory_net = (
            self._delayed_consequence_trajectory_totals(parent)
        )
        current_token = int(self._trainer.token_count)
        split_generation = max(1, int(parent.get("split_generation", 0) or 0) + 1)
        split_parent_record_id = str(parent.get("record_id", "")) or str(uuid4())
        baseline_query_score = float(parent.get("baseline_query_score", 0.0) or 0.0)
        baseline_grounded_fraction = float(parent.get("baseline_grounded_fraction", 0.0) or 0.0)
        if normalized_branch == "supportive":
            query_examples = supportive_examples
            aggregate_count = int(supportive_count)
            best_query_score = max(baseline_query_score, float(parent.get("best_query_score", baseline_query_score) or baseline_query_score))
            best_grounded_fraction = max(
                baseline_grounded_fraction,
                float(parent.get("best_grounded_fraction", baseline_grounded_fraction) or baseline_grounded_fraction),
            )
            credit_events = int(parent.get("credit_events", 0) or 0)
            penalty_events = 0
            forgiveness_events = int(parent.get("forgiveness_events", 0) or 0)
            unresolved_penalty_balance = 0.0
            resolved_improvement = float(parent.get("resolved_improvement", 0.0) or 0.0)
            max_regression = 0.0
            max_contradiction_signal = 0.0
            last_credit_score = float(parent.get("last_credit_score", 0.0) or 0.0)
            last_forgiveness_score = float(parent.get("last_forgiveness_score", 0.0) or 0.0)
            last_penalty_score = 0.0
            last_penalty_reason = ""
            branch_recent = max(
                0.20,
                float(parent.get("last_forgiveness_score", 0.0) or 0.0),
                float(parent.get("last_credit_score", 0.0) or 0.0),
                max(0.0, self._delayed_consequence_trajectory_recent_signal(parent)),
            )
            last_event_type = (
                "forgiveness"
                if float(parent.get("last_forgiveness_score", 0.0) or 0.0) > 0.0
                else "credit"
            )
            last_event_score = max(
                float(parent.get("last_forgiveness_score", 0.0) or 0.0),
                float(parent.get("last_credit_score", 0.0) or 0.0),
            )
            last_event_token = max(
                int(parent.get("last_forgiveness_token_count", 0) or 0),
                int(parent.get("last_credit_token_count", 0) or 0),
            )
            branch_credit_total = float(trajectory_credit_total)
            branch_penalty_total = 0.0
            branch_forgiveness_total = float(trajectory_forgiveness_total)
            branch_event_count = max(1, credit_events + forgiveness_events)
            branch_net_score = max(
                -DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT,
                min(
                    DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT,
                    branch_credit_total + branch_forgiveness_total - branch_penalty_total,
                ),
            )
            branch_peak_score = max(0.0, branch_net_score)
            branch_floor_score = min(0.0, branch_net_score)
            supportive_branch_examples = list(query_examples)
            adverse_branch_examples: list[str] = []
            supportive_occurrence_count = int(aggregate_count)
            adverse_occurrence_count = 0
            cumulative_cooling_delta = 0.0
            cooling_events = 0
        else:
            query_examples = adverse_examples
            aggregate_count = int(adverse_count)
            best_query_score = float(parent.get("baseline_query_score", 0.0) or 0.0)
            best_grounded_fraction = float(parent.get("baseline_grounded_fraction", 0.0) or 0.0)
            credit_events = 0
            penalty_events = int(parent.get("penalty_events", 0) or 0)
            forgiveness_events = 0
            unresolved_penalty_balance = float(parent.get("unresolved_penalty_balance", 0.0) or 0.0)
            resolved_improvement = 0.0
            max_regression = float(parent.get("max_regression", 0.0) or 0.0)
            max_contradiction_signal = float(parent.get("max_contradiction_signal", 0.0) or 0.0)
            last_credit_score = 0.0
            last_forgiveness_score = 0.0
            last_penalty_score = float(parent.get("last_penalty_score", 0.0) or 0.0)
            last_penalty_reason = str(parent.get("last_penalty_reason", "") or "")
            branch_recent = -max(
                0.20,
                float(parent.get("last_penalty_score", 0.0) or 0.0),
                abs(self._delayed_consequence_trajectory_recent_signal(parent)),
            )
            last_event_type = "penalty"
            last_event_score = float(parent.get("last_penalty_score", 0.0) or 0.0)
            last_event_token = int(parent.get("last_penalty_token_count", 0) or 0)
            branch_credit_total = 0.0
            branch_penalty_total = float(trajectory_penalty_total)
            branch_forgiveness_total = 0.0
            branch_event_count = max(1, penalty_events)
            branch_net_score = max(
                -DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT,
                min(DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT, -branch_penalty_total),
            )
            branch_peak_score = max(0.0, branch_net_score)
            branch_floor_score = min(0.0, branch_net_score)
            supportive_branch_examples = []
            adverse_branch_examples = list(query_examples)
            supportive_occurrence_count = 0
            adverse_occurrence_count = int(aggregate_count)
            cumulative_cooling_delta = float(parent.get("cumulative_cooling_delta", 0.0) or 0.0)
            cooling_events = int(parent.get("cooling_events", 0) or 0)
        if not query_examples:
            return None
        child = self._normalize_delayed_consequence_record(
            {
                "record_id": str(uuid4()),
                "created_at": str(parent.get("created_at") or timestamp),
                "created_token_count": int(parent.get("created_token_count", current_token) or current_token),
                "origin": str(parent.get("origin", "response_selected_evidence") or "response_selected_evidence"),
                "query_text": query_examples[0],
                "query_examples": list(query_examples),
                "baseline_query_score": float(baseline_query_score),
                "best_query_score": float(best_query_score),
                "baseline_grounded_fraction": float(baseline_grounded_fraction),
                "best_grounded_fraction": float(best_grounded_fraction),
                "outcome_score": float(parent.get("outcome_score", 0.0) or 0.0),
                "source_weights": deepcopy(dict(parent.get("source_weights") or {})),
                "provider_weights": deepcopy(dict(parent.get("provider_weights") or {})),
                "credit_events": int(credit_events),
                "penalty_events": int(penalty_events),
                "forgiveness_events": int(forgiveness_events),
                "cooling_events": int(cooling_events),
                "aggregate_count": int(aggregate_count),
                "aggregation_events": max(0, int(aggregate_count) - 1),
                "supportive_query_examples": list(supportive_branch_examples),
                "adverse_query_examples": list(adverse_branch_examples),
                "supportive_occurrence_count": int(supportive_occurrence_count),
                "adverse_occurrence_count": int(adverse_occurrence_count),
                "trajectory_credit_total": float(branch_credit_total),
                "trajectory_penalty_total": float(branch_penalty_total),
                "trajectory_forgiveness_total": float(branch_forgiveness_total),
                "trajectory_event_count": int(branch_event_count),
                "trajectory_net_score": float(branch_net_score),
                "trajectory_recent_delta_ema": float(max(-1.0, min(1.0, branch_recent))),
                "trajectory_peak_score": float(branch_peak_score),
                "trajectory_floor_score": float(branch_floor_score),
                "unresolved_penalty_balance": float(_safe_float(unresolved_penalty_balance)),
                "resolved_improvement": float(_safe_float(resolved_improvement)),
                "max_regression": float(_safe_float(max_regression)),
                "max_contradiction_signal": float(_safe_float(max_contradiction_signal)),
                "cumulative_cooling_delta": float(_safe_float(cumulative_cooling_delta)),
                "last_match_score": float(_safe_float(parent.get("last_match_score", 0.0))),
                "last_credit_score": float(_safe_float(last_credit_score)),
                "last_penalty_score": float(_safe_float(last_penalty_score)),
                "last_forgiveness_score": float(_safe_float(last_forgiveness_score)),
                "last_penalty_reason": str(last_penalty_reason),
                "last_activity_token_count": int(parent.get("last_activity_token_count", current_token) or current_token),
                "last_evaluated_token_count": int(parent.get("last_evaluated_token_count", current_token) or current_token),
                "last_cooling_token_count": int(parent.get("last_cooling_token_count", current_token) or current_token),
                "last_credit_token_count": int(parent.get("last_credit_token_count", 0) or 0) if normalized_branch == "supportive" else 0,
                "last_penalty_token_count": int(parent.get("last_penalty_token_count", 0) or 0) if normalized_branch == "adverse" else 0,
                "last_forgiveness_token_count": int(parent.get("last_forgiveness_token_count", 0) or 0) if normalized_branch == "supportive" else 0,
                "last_trajectory_event_type": str(last_event_type),
                "last_trajectory_event_score": float(_safe_float(last_event_score)),
                "last_trajectory_event_at": str(parent.get("last_evaluated_at", timestamp) or timestamp),
                "last_trajectory_event_token_count": int(last_event_token),
                "split_generation": int(split_generation),
                "split_parent_record_id": str(split_parent_record_id),
                "split_group_id": str(split_group_id),
                "split_branch": str(normalized_branch),
                "last_split_at": str(timestamp),
                "last_aggregated_at": str(parent.get("last_aggregated_at", "") or ""),
                "last_cooled_at": str(parent.get("last_cooled_at", "") or ""),
                "last_evaluated_at": str(parent.get("last_evaluated_at", timestamp) or timestamp),
                "last_evaluated_query_text": str(parent.get("last_evaluated_query_text", query_examples[0]) or query_examples[0]),
            }
        )
        return cast(dict[str, Any] | None, child)

    def _split_divergent_delayed_consequence_families_locked(self) -> dict[str, Any]:
        records = list(self._delayed_consequence_records)
        if not records:
            return {
                "split_records": 0,
                "max_branch_overlap": 1.0,
                "record_ids": [],
            }
        updated_records: list[dict[str, Any]] = []
        split_records = 0
        max_branch_overlap = 1.0
        split_record_ids: list[str] = []
        timestamp = datetime.now(timezone.utc).isoformat()
        for record in records:
            if int(record.get("aggregate_count", 1) or 1) < 2:
                updated_records.append(record)
                continue
            if str(record.get("split_branch", "") or ""):
                updated_records.append(record)
                continue
            supportive_examples = self._delayed_consequence_branch_examples(record, field="supportive_query_examples")
            adverse_examples = self._delayed_consequence_branch_examples(record, field="adverse_query_examples")
            supportive_occurrence_count = max(int(record.get("supportive_occurrence_count", 0) or 0), len(supportive_examples))
            adverse_occurrence_count = max(int(record.get("adverse_occurrence_count", 0) or 0), len(adverse_examples))
            if supportive_occurrence_count < DEFAULT_DELAYED_CONSEQUENCE_SPLIT_MIN_BRANCH_OCCURRENCES:
                updated_records.append(record)
                continue
            if adverse_occurrence_count < DEFAULT_DELAYED_CONSEQUENCE_SPLIT_MIN_BRANCH_OCCURRENCES:
                updated_records.append(record)
                continue
            trajectory_credit_total, trajectory_penalty_total, trajectory_forgiveness_total, _trajectory_net = (
                self._delayed_consequence_trajectory_totals(record)
            )
            if (trajectory_credit_total + trajectory_forgiveness_total) <= 0.0 or trajectory_penalty_total <= 0.0:
                updated_records.append(record)
                continue
            branch_overlap = self._delayed_consequence_branch_overlap_locked(record)
            max_branch_overlap = min(max_branch_overlap, float(branch_overlap))
            if branch_overlap > DEFAULT_DELAYED_CONSEQUENCE_SPLIT_MAX_BRANCH_OVERLAP:
                updated_records.append(record)
                continue
            split_group_id = str(record.get("split_group_id", "") or record.get("record_id", "") or uuid4())
            supportive_child = self._build_delayed_consequence_split_child_locked(
                record,
                branch="supportive",
                split_group_id=split_group_id,
                timestamp=timestamp,
            )
            adverse_child = self._build_delayed_consequence_split_child_locked(
                record,
                branch="adverse",
                split_group_id=split_group_id,
                timestamp=timestamp,
            )
            if supportive_child is None or adverse_child is None:
                updated_records.append(record)
                continue
            updated_records.extend([supportive_child, adverse_child])
            split_records += 1
            split_record_ids.append(str(record.get("record_id", "")))
        if split_records <= 0:
            return {
                "split_records": 0,
                "max_branch_overlap": 1.0 if max_branch_overlap >= 1.0 else float(max_branch_overlap),
                "record_ids": [],
            }
        self._delayed_consequence_records = deque(updated_records[:DEFAULT_DELAYED_CONSEQUENCE_RECORDS], maxlen=DEFAULT_DELAYED_CONSEQUENCE_RECORDS)
        self._delayed_consequence_split_total += int(split_records)
        self._record_brain_event_locked(
            {
                "type": "delayed_consequence_state_split",
                "timestamp": timestamp,
                "split_records": int(split_records),
                "record_ids": split_record_ids[:8],
                "max_branch_overlap": float(max_branch_overlap),
            }
        )
        self._mark_mutated()
        return {
            "split_records": int(split_records),
            "max_branch_overlap": float(max_branch_overlap),
            "record_ids": split_record_ids[:8],
        }

    def _should_remerge_delayed_consequence_split_group_locked(
        self,
        group_records: Sequence[Mapping[str, Any]],
    ) -> bool:
        branches = {
            self._normalize_action_text(record.get("split_branch", "")).lower()
            for record in list(group_records)
            if self._normalize_action_text(record.get("split_branch", "")).lower() in {"supportive", "adverse"}
        }
        if branches != {"supportive", "adverse"}:
            return False
        for record in list(group_records):
            branch = self._normalize_action_text(record.get("split_branch", "")).lower()
            if branch != "adverse":
                continue
            supportive_cross = max(
                int(record.get("supportive_occurrence_count", 0) or 0),
                len(self._delayed_consequence_branch_examples(record, field="supportive_query_examples")),
            )
            if supportive_cross < DEFAULT_DELAYED_CONSEQUENCE_REMERGE_MIN_CROSS_OCCURRENCES:
                continue
            recent_signal = self._delayed_consequence_trajectory_recent_signal(record)
            trajectory_state = self._delayed_consequence_trajectory_state(record)
            net_score = float(record.get("trajectory_net_score", 0.0) or 0.0)
            floor_score = float(record.get("trajectory_floor_score", net_score) or net_score)
            if recent_signal > 0.0 or trajectory_state in {"recovering", "positive", "mixed"} or net_score > floor_score + 0.05:
                return True
        return False

    def _build_remerged_delayed_consequence_family_locked(
        self,
        group_records: Sequence[Mapping[str, Any]],
        *,
        split_group_id: str,
        timestamp: str,
    ) -> dict[str, Any] | None:
        ordered_records = [cast(dict[str, Any], deepcopy(record)) for record in list(group_records) if isinstance(record, Mapping)]
        if not ordered_records:
            return None
        merged = cast(dict[str, Any], ordered_records[0])
        for record in ordered_records[1:]:
            merged = self._merge_delayed_consequence_records_locked(merged, record)
        merged["supportive_query_examples"] = []
        merged["adverse_query_examples"] = []
        merged["supportive_occurrence_count"] = 0
        merged["adverse_occurrence_count"] = 0
        merged["split_group_id"] = str(split_group_id)
        merged["split_branch"] = ""
        merged["split_generation"] = max(
            int(record.get("split_generation", 0) or 0)
            for record in ordered_records
        )
        merged["split_parent_record_id"] = self._normalize_action_text(merged.get("split_parent_record_id", "")) or str(
            ordered_records[0].get("split_parent_record_id", "") or ordered_records[0].get("record_id", "")
        )
        merged["remerge_events"] = (
            sum(int(record.get("remerge_events", 0) or 0) for record in ordered_records) + 1
        )
        merged["last_remerged_at"] = str(timestamp)
        normalized = self._normalize_delayed_consequence_record(merged)
        return cast(dict[str, Any] | None, normalized)

    def _remerge_converged_delayed_consequence_families_locked(self) -> dict[str, Any]:
        records = list(self._delayed_consequence_records)
        if len(records) < 2:
            return {
                "remerged_records": 0,
                "record_ids": [],
            }
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            split_group_id = self._normalize_action_text(record.get("split_group_id", ""))
            split_branch = self._normalize_action_text(record.get("split_branch", "")).lower()
            if not split_group_id or split_branch not in {"supportive", "adverse"}:
                continue
            groups.setdefault(split_group_id, []).append(record)
        if not groups:
            return {
                "remerged_records": 0,
                "record_ids": [],
            }
        remerge_map: dict[str, dict[str, Any]] = {}
        remerged_record_ids: list[str] = []
        timestamp = datetime.now(timezone.utc).isoformat()
        for split_group_id, group_records in groups.items():
            if not self._should_remerge_delayed_consequence_split_group_locked(group_records):
                continue
            merged = self._build_remerged_delayed_consequence_family_locked(
                group_records,
                split_group_id=split_group_id,
                timestamp=timestamp,
            )
            if merged is None:
                continue
            remerge_map[split_group_id] = merged
            remerged_record_ids.extend(str(record.get("record_id", "")) for record in group_records)
        if not remerge_map:
            return {
                "remerged_records": 0,
                "record_ids": [],
            }
        updated_records: list[dict[str, Any]] = []
        inserted_groups: set[str] = set()
        for record in records:
            split_group_id = self._normalize_action_text(record.get("split_group_id", ""))
            split_branch = self._normalize_action_text(record.get("split_branch", "")).lower()
            if split_group_id in remerge_map and split_branch in {"supportive", "adverse"}:
                if split_group_id in inserted_groups:
                    continue
                updated_records.append(remerge_map[split_group_id])
                inserted_groups.add(split_group_id)
                continue
            updated_records.append(record)
        self._delayed_consequence_records = deque(updated_records[:DEFAULT_DELAYED_CONSEQUENCE_RECORDS], maxlen=DEFAULT_DELAYED_CONSEQUENCE_RECORDS)
        self._delayed_consequence_remerged_total += int(len(remerge_map))
        self._record_brain_event_locked(
            {
                "type": "delayed_consequence_state_remerged",
                "timestamp": timestamp,
                "remerged_records": int(len(remerge_map)),
                "record_ids": remerged_record_ids[:8],
            }
        )
        self._mark_mutated()
        return {
            "remerged_records": int(len(remerge_map)),
            "record_ids": remerged_record_ids[:8],
        }

    @staticmethod
    def _delayed_consequence_weight_overlap(
        left: Mapping[str, Any],
        right: Mapping[str, Any],
    ) -> float:
        def _normalized(value: Mapping[str, Any]) -> dict[str, float]:
            result: dict[str, float] = {}
            for raw_key, raw_weight in value.items():
                key = " ".join(str(raw_key).split()).strip().lower()
                if not key:
                    continue
                try:
                    weight = max(0.0, min(1.0, float(raw_weight)))
                except (TypeError, ValueError):
                    continue
                if weight <= 0.0:
                    continue
                result[key] = weight
            return result

        normalized_left = _normalized(left)
        normalized_right = _normalized(right)
        if not normalized_left or not normalized_right:
            return 0.0
        all_keys = set(normalized_left) | set(normalized_right)
        shared_keys = set(normalized_left) & set(normalized_right)
        if not all_keys or not shared_keys:
            return 0.0
        weighted_overlap = sum(min(normalized_left.get(key, 0.0), normalized_right.get(key, 0.0)) for key in all_keys) / max(
            1e-6,
            sum(max(normalized_left.get(key, 0.0), normalized_right.get(key, 0.0)) for key in all_keys),
        )
        key_overlap = float(len(shared_keys)) / float(max(1, len(all_keys)))
        return float(max(0.0, min(1.0, max(weighted_overlap, key_overlap))))

    def _delayed_consequence_provenance_overlap_locked(
        self,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
    ) -> float:
        overlap_scores: list[float] = []
        left_sources = cast(Mapping[str, Any], left.get("source_weights") or {})
        right_sources = cast(Mapping[str, Any], right.get("source_weights") or {})
        if left_sources and right_sources:
            overlap_scores.append(self._delayed_consequence_weight_overlap(left_sources, right_sources))
        left_providers = cast(Mapping[str, Any], left.get("provider_weights") or {})
        right_providers = cast(Mapping[str, Any], right.get("provider_weights") or {})
        if left_providers and right_providers:
            overlap_scores.append(self._delayed_consequence_weight_overlap(left_providers, right_providers))
        if not overlap_scores:
            return 0.0
        if len(overlap_scores) == 1:
            return float(overlap_scores[0])
        return float(min(overlap_scores))

    def _delayed_consequence_aggregation_score_locked(
        self,
        existing: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> float:
        existing_split_group = self._normalize_action_text(existing.get("split_group_id", ""))
        candidate_split_group = self._normalize_action_text(candidate.get("split_group_id", ""))
        existing_split_branch = self._normalize_action_text(existing.get("split_branch", "")).lower()
        candidate_split_branch = self._normalize_action_text(candidate.get("split_branch", "")).lower()
        if (
            existing_split_group
            and candidate_split_group
            and existing_split_group == candidate_split_group
            and existing_split_branch
            and candidate_split_branch
            and existing_split_branch != candidate_split_branch
        ):
            return 0.0
        provenance_overlap = self._delayed_consequence_provenance_overlap_locked(existing, candidate)
        if provenance_overlap < DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_PROVENANCE_THRESHOLD:
            return 0.0
        query_score = self._delayed_consequence_match_score_locked(
            existing,
            {
                "query_text": str(candidate.get("query_text", "")),
                "query_terms": list(candidate.get("query_terms") or []),
            },
        )
        if query_score < DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_MATCH_THRESHOLD:
            return 0.0
        return float(max(0.0, min(1.0, 0.55 * float(query_score) + 0.45 * float(provenance_overlap))))

    def _merge_delayed_consequence_records_locked(
        self,
        primary: Mapping[str, Any],
        secondary: Mapping[str, Any],
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()

        def _safe_int(raw_value: Any) -> int:
            try:
                return max(0, int(raw_value))
            except (TypeError, ValueError):
                return 0

        def _safe_float(raw_value: Any) -> float:
            try:
                return max(0.0, min(1.0, float(raw_value)))
            except (TypeError, ValueError):
                return 0.0

        def _safe_total(raw_value: Any) -> float:
            try:
                return max(0.0, float(raw_value))
            except (TypeError, ValueError):
                return 0.0

        def _safe_signed(raw_value: Any, *, limit: float = DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT) -> float:
            try:
                return max(-float(limit), min(float(limit), float(raw_value)))
            except (TypeError, ValueError):
                return 0.0

        def _merged_weights(*values: Mapping[str, Any]) -> dict[str, float]:
            merged: dict[str, float] = {}
            for raw_value in values:
                for raw_name, raw_weight in dict(raw_value).items():
                    name = " ".join(str(raw_name).split()).strip()
                    if not name:
                        continue
                    merged[name] = max(float(merged.get(name, 0.0)), _safe_float(raw_weight))
            return {name: weight for name, weight in merged.items() if weight > 0.0}

        merged_examples: list[str] = []
        seen_examples: set[str] = set()
        for record in (primary, secondary):
            for example in self._delayed_consequence_query_examples(record):
                key = example.lower()
                if key in seen_examples:
                    continue
                seen_examples.add(key)
                merged_examples.append(example)
                if len(merged_examples) >= DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT:
                    break
            if len(merged_examples) >= DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT:
                break

        term_counter: Counter[str] = Counter()
        first_seen: dict[str, int] = {}
        for record in (primary, secondary):
            for query_text in self._delayed_consequence_query_examples(record):
                for raw_term in self._consequence_query_terms(query_text):
                    term = _canonical_provider_term(raw_term)
                    if not term:
                        continue
                    term_counter[term] += 1
                    first_seen.setdefault(term, len(first_seen))
            for raw_term in list(record.get("query_terms") or []):
                term = _canonical_provider_term(raw_term)
                if not term:
                    continue
                term_counter[term] += 1
                first_seen.setdefault(term, len(first_seen))
        merged_terms = [
            term
            for term, _count in sorted(
                term_counter.items(),
                key=lambda item: (-int(item[1]), int(first_seen.get(item[0], 0)), item[0]),
            )
        ][:DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_TERM_LIMIT]

        merged_supportive_examples: list[str] = []
        merged_adverse_examples: list[str] = []
        for field, target in (
            ("supportive_query_examples", merged_supportive_examples),
            ("adverse_query_examples", merged_adverse_examples),
        ):
            seen_branch: set[str] = set()
            for record in (primary, secondary):
                for example in self._delayed_consequence_branch_examples(record, field=field):
                    key = example.lower()
                    if key in seen_branch:
                        continue
                    seen_branch.add(key)
                    target.append(example)
                    if len(target) >= DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT:
                        break
                if len(target) >= DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT:
                    break
        merged_supportive_occurrence_count = max(
            len(merged_supportive_examples),
            _safe_int(primary.get("supportive_occurrence_count", 0)),
            _safe_int(secondary.get("supportive_occurrence_count", 0)),
        )
        merged_adverse_occurrence_count = max(
            len(merged_adverse_examples),
            _safe_int(primary.get("adverse_occurrence_count", 0)),
            _safe_int(secondary.get("adverse_occurrence_count", 0)),
        )
        merged_split_generation = max(
            _safe_int(primary.get("split_generation", 0)),
            _safe_int(secondary.get("split_generation", 0)),
        )
        merged_split_parent_record_id = self._normalize_action_text(primary.get("split_parent_record_id", "")) or self._normalize_action_text(
            secondary.get("split_parent_record_id", "")
        )
        merged_split_group_id = self._normalize_action_text(primary.get("split_group_id", "")) or self._normalize_action_text(
            secondary.get("split_group_id", "")
        )
        primary_split_branch = self._normalize_action_text(primary.get("split_branch", ""))
        secondary_split_branch = self._normalize_action_text(secondary.get("split_branch", ""))
        merged_split_branch = (
            ""
            if primary_split_branch and secondary_split_branch and primary_split_branch != secondary_split_branch
            else primary_split_branch or secondary_split_branch
        )
        merged_last_split_at = self._normalize_action_text(primary.get("last_split_at", "")) or self._normalize_action_text(
            secondary.get("last_split_at", "")
        )
        merged_remerge_events = _safe_int(primary.get("remerge_events", 0)) + _safe_int(secondary.get("remerge_events", 0))
        merged_last_remerged_at = self._normalize_action_text(primary.get("last_remerged_at", "")) or self._normalize_action_text(
            secondary.get("last_remerged_at", "")
        )

        primary_created_token = _safe_int(primary.get("created_token_count", 0))
        secondary_created_token = _safe_int(secondary.get("created_token_count", 0))
        if primary_created_token <= 0 and secondary_created_token > 0:
            family_created_token = secondary_created_token
            family_created_at = str(secondary.get("created_at") or now)
        elif secondary_created_token <= 0 and primary_created_token > 0:
            family_created_token = primary_created_token
            family_created_at = str(primary.get("created_at") or now)
        elif 0 < secondary_created_token < primary_created_token:
            family_created_token = secondary_created_token
            family_created_at = str(secondary.get("created_at") or now)
        else:
            family_created_token = max(0, primary_created_token or secondary_created_token)
            family_created_at = str(primary.get("created_at") or secondary.get("created_at") or now)

        primary_credit_total, primary_penalty_total, primary_forgiveness_total, _primary_net = self._delayed_consequence_trajectory_totals(primary)
        secondary_credit_total, secondary_penalty_total, secondary_forgiveness_total, _secondary_net = self._delayed_consequence_trajectory_totals(secondary)
        merged_credit_total = float(primary_credit_total + secondary_credit_total)
        merged_penalty_total = float(primary_penalty_total + secondary_penalty_total)
        merged_forgiveness_total = float(primary_forgiveness_total + secondary_forgiveness_total)
        merged_net_score = float(
            max(
                -DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT,
                min(
                    DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT,
                    merged_credit_total + merged_forgiveness_total - merged_penalty_total,
                ),
            )
        )
        primary_trajectory_events = max(0, int(primary.get("trajectory_event_count", 0) or 0))
        secondary_trajectory_events = max(0, int(secondary.get("trajectory_event_count", 0) or 0))
        merged_trajectory_events = int(primary_trajectory_events + secondary_trajectory_events)
        if merged_trajectory_events > 0:
            merged_recent_delta = (
                float(primary_trajectory_events)
                * _safe_signed(primary.get("trajectory_recent_delta_ema", 0.0), limit=1.0)
                + float(secondary_trajectory_events)
                * _safe_signed(secondary.get("trajectory_recent_delta_ema", 0.0), limit=1.0)
            ) / float(max(1, merged_trajectory_events))
        else:
            merged_recent_delta = 0.0
        primary_peak = _safe_signed(primary.get("trajectory_peak_score", primary.get("trajectory_net_score", 0.0)))
        secondary_peak = _safe_signed(secondary.get("trajectory_peak_score", secondary.get("trajectory_net_score", 0.0)))
        primary_floor = _safe_signed(primary.get("trajectory_floor_score", primary.get("trajectory_net_score", 0.0)))
        secondary_floor = _safe_signed(secondary.get("trajectory_floor_score", secondary.get("trajectory_net_score", 0.0)))
        primary_event_token = max(
            _safe_int(primary.get("last_trajectory_event_token_count", 0)),
            _safe_int(primary.get("last_activity_token_count", 0)),
        )
        secondary_event_token = max(
            _safe_int(secondary.get("last_trajectory_event_token_count", 0)),
            _safe_int(secondary.get("last_activity_token_count", 0)),
        )
        latest_trajectory_record = primary if primary_event_token >= secondary_event_token else secondary

        normalized = self._normalize_delayed_consequence_record(
            {
                "record_id": str(primary.get("record_id", "")) or str(uuid4()),
                "created_at": str(family_created_at),
                "created_token_count": int(family_created_token),
                "origin": self._normalize_action_text(primary.get("origin", "response_selected_evidence"))
                or "response_selected_evidence",
                "query_text": merged_examples[0] if merged_examples else str(primary.get("query_text", "")),
                "query_examples": merged_examples,
                "query_terms": merged_terms,
                "baseline_query_score": max(
                    _safe_float(primary.get("baseline_query_score", 0.0)),
                    _safe_float(secondary.get("baseline_query_score", 0.0)),
                ),
                "best_query_score": max(
                    _safe_float(primary.get("best_query_score", 0.0)),
                    _safe_float(secondary.get("best_query_score", 0.0)),
                ),
                "baseline_grounded_fraction": max(
                    _safe_float(primary.get("baseline_grounded_fraction", 0.0)),
                    _safe_float(secondary.get("baseline_grounded_fraction", 0.0)),
                ),
                "best_grounded_fraction": max(
                    _safe_float(primary.get("best_grounded_fraction", 0.0)),
                    _safe_float(secondary.get("best_grounded_fraction", 0.0)),
                ),
                "outcome_score": max(
                    _safe_float(primary.get("outcome_score", 0.0)),
                    _safe_float(secondary.get("outcome_score", 0.0)),
                ),
                "source_weights": _merged_weights(
                    cast(Mapping[str, Any], primary.get("source_weights") or {}),
                    cast(Mapping[str, Any], secondary.get("source_weights") or {}),
                ),
                "provider_weights": _merged_weights(
                    cast(Mapping[str, Any], primary.get("provider_weights") or {}),
                    cast(Mapping[str, Any], secondary.get("provider_weights") or {}),
                ),
                "credit_events": _safe_int(primary.get("credit_events", 0)) + _safe_int(secondary.get("credit_events", 0)),
                "penalty_events": _safe_int(primary.get("penalty_events", 0)) + _safe_int(secondary.get("penalty_events", 0)),
                "forgiveness_events": _safe_int(primary.get("forgiveness_events", 0)) + _safe_int(secondary.get("forgiveness_events", 0)),
                "cooling_events": _safe_int(primary.get("cooling_events", 0)) + _safe_int(secondary.get("cooling_events", 0)),
                "aggregate_count": max(1, _safe_int(primary.get("aggregate_count", 1)))
                + max(1, _safe_int(secondary.get("aggregate_count", 1))),
                "aggregation_events": _safe_int(primary.get("aggregation_events", 0))
                + _safe_int(secondary.get("aggregation_events", 0))
                + 1,
                "supportive_query_examples": list(merged_supportive_examples),
                "adverse_query_examples": list(merged_adverse_examples),
                "supportive_occurrence_count": int(merged_supportive_occurrence_count),
                "adverse_occurrence_count": int(merged_adverse_occurrence_count),
                "trajectory_credit_total": float(merged_credit_total),
                "trajectory_penalty_total": float(merged_penalty_total),
                "trajectory_forgiveness_total": float(merged_forgiveness_total),
                "trajectory_event_count": int(merged_trajectory_events),
                "trajectory_net_score": float(merged_net_score),
                "trajectory_recent_delta_ema": float(_safe_signed(merged_recent_delta, limit=1.0)),
                "trajectory_peak_score": float(max(primary_peak, secondary_peak, merged_net_score)),
                "trajectory_floor_score": float(min(primary_floor, secondary_floor, merged_net_score)),
                "unresolved_penalty_balance": min(
                    1.0,
                    _safe_float(primary.get("unresolved_penalty_balance", 0.0))
                    + _safe_float(secondary.get("unresolved_penalty_balance", 0.0)),
                ),
                "resolved_improvement": max(
                    _safe_float(primary.get("resolved_improvement", 0.0)),
                    _safe_float(secondary.get("resolved_improvement", 0.0)),
                ),
                "max_regression": max(
                    _safe_float(primary.get("max_regression", 0.0)),
                    _safe_float(secondary.get("max_regression", 0.0)),
                ),
                "max_contradiction_signal": max(
                    _safe_float(primary.get("max_contradiction_signal", 0.0)),
                    _safe_float(secondary.get("max_contradiction_signal", 0.0)),
                ),
                "cumulative_cooling_delta": min(
                    1.0,
                    _safe_float(primary.get("cumulative_cooling_delta", 0.0))
                    + _safe_float(secondary.get("cumulative_cooling_delta", 0.0)),
                ),
                "last_match_score": max(
                    _safe_float(primary.get("last_match_score", 0.0)),
                    _safe_float(secondary.get("last_match_score", 0.0)),
                ),
                "last_credit_score": max(
                    _safe_float(primary.get("last_credit_score", 0.0)),
                    _safe_float(secondary.get("last_credit_score", 0.0)),
                ),
                "last_penalty_score": max(
                    _safe_float(primary.get("last_penalty_score", 0.0)),
                    _safe_float(secondary.get("last_penalty_score", 0.0)),
                ),
                "last_forgiveness_score": max(
                    _safe_float(primary.get("last_forgiveness_score", 0.0)),
                    _safe_float(secondary.get("last_forgiveness_score", 0.0)),
                ),
                "last_penalty_reason": self._normalize_action_text(primary.get("last_penalty_reason", ""))
                or self._normalize_action_text(secondary.get("last_penalty_reason", "")),
                "last_activity_token_count": max(
                    _safe_int(primary.get("last_activity_token_count", 0)),
                    _safe_int(secondary.get("last_activity_token_count", 0)),
                ),
                "last_evaluated_token_count": max(
                    _safe_int(primary.get("last_evaluated_token_count", 0)),
                    _safe_int(secondary.get("last_evaluated_token_count", 0)),
                ),
                "last_cooling_token_count": max(
                    _safe_int(primary.get("last_cooling_token_count", 0)),
                    _safe_int(secondary.get("last_cooling_token_count", 0)),
                ),
                "last_credit_token_count": max(
                    _safe_int(primary.get("last_credit_token_count", 0)),
                    _safe_int(secondary.get("last_credit_token_count", 0)),
                ),
                "last_penalty_token_count": max(
                    _safe_int(primary.get("last_penalty_token_count", 0)),
                    _safe_int(secondary.get("last_penalty_token_count", 0)),
                ),
                "last_forgiveness_token_count": max(
                    _safe_int(primary.get("last_forgiveness_token_count", 0)),
                    _safe_int(secondary.get("last_forgiveness_token_count", 0)),
                ),
                "last_trajectory_event_type": self._normalize_action_text(
                    latest_trajectory_record.get("last_trajectory_event_type", "")
                ),
                "last_trajectory_event_score": float(
                    _safe_total(latest_trajectory_record.get("last_trajectory_event_score", 0.0))
                ),
                "last_trajectory_event_at": self._normalize_action_text(
                    latest_trajectory_record.get("last_trajectory_event_at", "")
                ),
                "last_trajectory_event_token_count": int(
                    _safe_int(latest_trajectory_record.get("last_trajectory_event_token_count", 0))
                ),
                "split_generation": int(merged_split_generation),
                "split_parent_record_id": str(merged_split_parent_record_id),
                "split_group_id": str(merged_split_group_id),
                "split_branch": str(merged_split_branch),
                "remerge_events": int(merged_remerge_events),
                "last_split_at": str(merged_last_split_at),
                "last_remerged_at": str(merged_last_remerged_at),
                "last_cooled_at": self._normalize_action_text(primary.get("last_cooled_at", ""))
                or self._normalize_action_text(secondary.get("last_cooled_at", "")),
                "last_evaluated_at": self._normalize_action_text(primary.get("last_evaluated_at", ""))
                or self._normalize_action_text(secondary.get("last_evaluated_at", "")),
                "last_evaluated_query_text": self._normalize_action_text(primary.get("last_evaluated_query_text", ""))
                or self._normalize_action_text(secondary.get("last_evaluated_query_text", "")),
                "last_aggregated_at": now,
            }
        )
        return cast(dict[str, Any], normalized if normalized is not None else dict(primary))

    def _upsert_delayed_consequence_record_locked(
        self,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        existing_records = list(self._delayed_consequence_records)
        best_index: int | None = None
        best_score = 0.0
        for index, record in enumerate(existing_records):
            score = self._delayed_consequence_aggregation_score_locked(record, candidate)
            if score > best_score:
                best_score = float(score)
                best_index = index
        if best_index is None:
            self._delayed_consequence_records = deque(
                [cast(dict[str, Any], candidate), *existing_records][:DEFAULT_DELAYED_CONSEQUENCE_RECORDS],
                maxlen=DEFAULT_DELAYED_CONSEQUENCE_RECORDS,
            )
            self._mark_mutated()
            return cast(dict[str, Any], candidate)
        merged = self._merge_delayed_consequence_records_locked(candidate, existing_records[best_index])
        self._delayed_consequence_records = deque(
            [merged, *(record for index, record in enumerate(existing_records) if index != best_index)],
            maxlen=DEFAULT_DELAYED_CONSEQUENCE_RECORDS,
        )
        self._delayed_consequence_compacted_total += 1
        self._record_brain_event_locked(
            {
                "type": "delayed_consequence_state_compacted",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "compacted_records": 1,
                "max_aggregate_count": int(merged.get("aggregate_count", 1) or 1),
                "record_id": str(merged.get("record_id", "")),
            }
        )
        self._mark_mutated()
        return merged

    def _compact_delayed_consequence_records_locked(self) -> dict[str, Any]:
        records = list(self._delayed_consequence_records)
        if len(records) < 2:
            return {
                "compacted_records": 0,
                "max_aggregate_count": max(
                    1,
                    max((int(record.get("aggregate_count", 1) or 1) for record in records), default=1),
                ),
            }
        compacted: list[dict[str, Any]] = []
        compacted_records = 0
        max_aggregate_count = 1
        for record in records:
            best_index: int | None = None
            best_score = 0.0
            for index, existing in enumerate(compacted):
                score = self._delayed_consequence_aggregation_score_locked(existing, record)
                if score > best_score:
                    best_score = float(score)
                    best_index = index
            if best_index is None:
                compacted.append(record)
                max_aggregate_count = max(max_aggregate_count, int(record.get("aggregate_count", 1) or 1))
                continue
            compacted[best_index] = self._merge_delayed_consequence_records_locked(compacted[best_index], record)
            compacted_records += 1
            max_aggregate_count = max(max_aggregate_count, int(compacted[best_index].get("aggregate_count", 1) or 1))
        if compacted_records <= 0:
            return {
                "compacted_records": 0,
                "max_aggregate_count": int(max_aggregate_count),
            }
        self._delayed_consequence_records = deque(compacted, maxlen=DEFAULT_DELAYED_CONSEQUENCE_RECORDS)
        self._delayed_consequence_compacted_total += int(compacted_records)
        self._record_brain_event_locked(
            {
                "type": "delayed_consequence_state_compacted",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "compacted_records": int(compacted_records),
                "max_aggregate_count": int(max_aggregate_count),
            }
        )
        self._mark_mutated()
        return {
            "compacted_records": int(compacted_records),
            "max_aggregate_count": int(max_aggregate_count),
        }

    def _cool_delayed_consequence_records_locked(
        self,
        *,
        current_token: int | None = None,
    ) -> dict[str, Any]:
        token = int(self._trainer.token_count if current_token is None else current_token)
        if not self._delayed_consequence_records:
            return {
                "cooled_records": 0,
                "retired_records": 0,
                "max_cooling_delta": 0.0,
                "retired_record_ids": [],
            }
        now = datetime.now(timezone.utc).isoformat()
        remaining: list[dict[str, Any]] = []
        cooled_records = 0
        retired_records = 0
        max_cooling_delta = 0.0
        retired_record_ids: list[str] = []
        for record in list(self._delayed_consequence_records):
            created_token_count = max(0, int(record.get("created_token_count", token)))
            last_activity_token_count = max(
                created_token_count,
                int(record.get("last_activity_token_count", created_token_count)),
            )
            last_cooling_token_count = max(
                created_token_count,
                int(record.get("last_cooling_token_count", last_activity_token_count)),
            )
            unresolved_penalty_balance = max(
                0.0,
                min(1.0, float(record.get("unresolved_penalty_balance", 0.0) or 0.0)),
            )
            inactivity_tokens = max(0, token - last_activity_token_count)
            if (
                unresolved_penalty_balance > 0.0
                and inactivity_tokens >= DEFAULT_DELAYED_CONSEQUENCE_COOLING_START_TOKENS
            ):
                cooling_anchor = max(
                    last_cooling_token_count,
                    last_activity_token_count + DEFAULT_DELAYED_CONSEQUENCE_COOLING_START_TOKENS,
                )
                cooling_delta_tokens = max(0, token - cooling_anchor)
                if cooling_delta_tokens > 0:
                    decay = math.exp(
                        -float(cooling_delta_tokens)
                        / float(max(1, DEFAULT_DELAYED_CONSEQUENCE_COOLING_WINDOW_TOKENS))
                    )
                    cooled_balance = max(0.0, unresolved_penalty_balance * float(decay))
                    cooling_delta = max(0.0, unresolved_penalty_balance - cooled_balance)
                    if cooling_delta > 1e-6:
                        record["unresolved_penalty_balance"] = float(cooled_balance)
                        record["last_cooling_token_count"] = int(token)
                        record["cooling_events"] = int(record.get("cooling_events", 0) or 0) + 1
                        record["cumulative_cooling_delta"] = float(
                            max(0.0, float(record.get("cumulative_cooling_delta", 0.0) or 0.0)) + cooling_delta
                        )
                        record["last_cooled_at"] = now
                        cooled_records += 1
                        max_cooling_delta = max(max_cooling_delta, cooling_delta)
                        self._delayed_consequence_cooled_total += 1
                        unresolved_penalty_balance = cooled_balance
            retirement_age_tokens = max(
                max(0, token - created_token_count),
                max(0, token - last_activity_token_count),
            )
            if (
                retirement_age_tokens >= DEFAULT_DELAYED_CONSEQUENCE_RETIREMENT_TOKENS
                and unresolved_penalty_balance <= DEFAULT_DELAYED_CONSEQUENCE_RETIREMENT_BALANCE_THRESHOLD
            ):
                retired_records += 1
                self._delayed_consequence_retired_total += 1
                retired_record_ids.append(str(record.get("record_id", "")))
                continue
            remaining.append(record)
        mutated = cooled_records > 0 or retired_records > 0
        if mutated:
            self._delayed_consequence_records = deque(remaining, maxlen=DEFAULT_DELAYED_CONSEQUENCE_RECORDS)
            if cooled_records > 0:
                self._record_brain_event_locked(
                    {
                        "type": "delayed_consequence_state_cooled",
                        "timestamp": now,
                        "cooled_records": int(cooled_records),
                        "max_cooling_delta": float(max_cooling_delta),
                    }
                )
            if retired_records > 0:
                self._record_brain_event_locked(
                    {
                        "type": "delayed_consequence_state_retired",
                        "timestamp": now,
                        "retired_records": int(retired_records),
                        "record_ids": retired_record_ids[:8],
                    }
                )
            self._mark_mutated()
        return {
            "cooled_records": int(cooled_records),
            "retired_records": int(retired_records),
            "max_cooling_delta": float(max_cooling_delta),
            "retired_record_ids": retired_record_ids[:8],
        }

    def _apply_background_source_delayed_penalty_locked(
        self,
        *,
        source_weights: Mapping[str, Any],
        penalty_score: float,
    ) -> list[str]:
        applied: list[str] = []
        calibrated_score = max(0.0, min(1.0, float(penalty_score)))
        if calibrated_score <= 0.0:
            return applied
        for runtime in self._brain_source_runtimes:
            weight = max(0.0, min(1.0, float(source_weights.get(runtime.name, 0.0) or 0.0)))
            if weight <= 0.0:
                continue
            entry = self._background_source_utility_entry_locked(runtime)
            sample = max(0.0, min(1.0, calibrated_score * weight))
            previous_penalty = max(0.0, min(1.0, float(entry.get("contradiction_decay_ema", 0.0) or 0.0)))
            entry["contradiction_decay_ema"] = float(
                sample if previous_penalty <= 0.0 else 0.75 * previous_penalty + 0.25 * sample
            )
            applied.append(runtime.name)
        return applied

    def _apply_background_source_forgiveness_locked(
        self,
        *,
        source_weights: Mapping[str, Any],
        forgiveness_score: float,
    ) -> list[str]:
        applied: list[str] = []
        calibrated_score = max(0.0, min(1.0, float(forgiveness_score)))
        if calibrated_score <= 0.0:
            return applied
        for runtime in self._brain_source_runtimes:
            weight = max(0.0, min(1.0, float(source_weights.get(runtime.name, 0.0) or 0.0)))
            if weight <= 0.0:
                continue
            entry = self._background_source_utility_entry_locked(runtime)
            previous_penalty = max(0.0, min(1.0, float(entry.get("contradiction_decay_ema", 0.0) or 0.0)))
            if previous_penalty <= 0.0:
                continue
            reduction = min(previous_penalty, float(calibrated_score) * float(weight))
            if reduction <= 0.0:
                continue
            entry["contradiction_decay_ema"] = float(max(0.0, previous_penalty - reduction))
            applied.append(runtime.name)
        return applied

    def _apply_background_source_delayed_consequence_locked(
        self,
        *,
        source_weights: Mapping[str, Any],
        consequence_score: float,
    ) -> list[str]:
        applied: list[str] = []
        calibrated_score = max(0.0, min(1.0, float(consequence_score)))
        if calibrated_score <= 0.0:
            return applied
        for runtime in self._brain_source_runtimes:
            weight = max(0.0, min(1.0, float(source_weights.get(runtime.name, 0.0) or 0.0)))
            if weight <= 0.0:
                continue
            entry = self._background_source_utility_entry_locked(runtime)
            sample = max(0.0, min(1.0, calibrated_score * weight))
            previous_delayed = max(0.0, min(1.0, float(entry.get("delayed_consequence_ema", 0.0) or 0.0)))
            entry["delayed_consequence_ema"] = float(
                sample if previous_delayed <= 0.0 else 0.75 * previous_delayed + 0.25 * sample
            )
            previous_utility = max(0.0, min(1.0, float(entry.get("utility_ema", 0.0) or 0.0)))
            reinforced_utility = max(
                previous_utility,
                float(entry.get("grounded_outcome_ema", 0.0) or 0.0),
                float(entry.get("delayed_consequence_ema", 0.0) or 0.0),
            )
            entry["utility_ema"] = float(
                reinforced_utility if previous_utility <= 0.0 else 0.80 * previous_utility + 0.20 * reinforced_utility
            )
            applied.append(runtime.name)
        return applied

    def _apply_background_source_family_summary_locked(
        self,
        *,
        source_weights: Mapping[str, Any],
        family_summary_score: float,
    ) -> list[str]:
        applied: list[str] = []
        calibrated_score = max(0.0, min(1.0, float(family_summary_score)))
        for runtime in self._brain_source_runtimes:
            weight = max(0.0, min(1.0, float(source_weights.get(runtime.name, 0.0) or 0.0)))
            if weight <= 0.0:
                continue
            entry = self._background_source_utility_entry_locked(runtime)
            sample = max(0.0, min(1.0, calibrated_score * weight))
            previous_summary = max(0.0, min(1.0, float(entry.get("grounded_family_summary_ema", 0.0) or 0.0)))
            entry["grounded_family_summary_ema"] = float(
                sample if previous_summary <= 0.0 else 0.70 * previous_summary + 0.30 * sample
            )
            previous_utility = max(0.0, min(1.0, float(entry.get("utility_ema", 0.0) or 0.0)))
            reinforced_utility = max(
                previous_utility,
                float(entry.get("grounded_outcome_ema", 0.0) or 0.0),
                float(entry.get("delayed_consequence_ema", 0.0) or 0.0),
                float(entry.get("grounded_family_summary_ema", 0.0) or 0.0),
            )
            entry["utility_ema"] = float(
                reinforced_utility if previous_utility <= 0.0 else 0.80 * previous_utility + 0.20 * reinforced_utility
            )
            applied.append(runtime.name)
        return applied

    def _apply_provider_delayed_penalty_locked(
        self,
        *,
        autonomy: dict[str, Any],
        provider_weights: Mapping[str, Any],
        penalty_score: float,
    ) -> list[str]:
        curriculum = self._normalize_provider_curriculum(autonomy.get("provider_curriculum"))
        calibrated_score = max(0.0, min(1.0, float(penalty_score)))
        if not curriculum or calibrated_score <= 0.0:
            return []
        applied: list[str] = []
        for raw_provider, raw_weight in dict(provider_weights).items():
            provider = " ".join(str(raw_provider).split()).strip().lower()
            weight = max(0.0, min(1.0, float(raw_weight or 0.0)))
            if not provider or weight <= 0.0:
                continue
            entry = curriculum.get(provider)
            if not isinstance(entry, Mapping):
                continue
            sample = max(0.0, min(1.0, calibrated_score * weight))
            previous_penalty = max(0.0, min(1.0, float(entry.get("contradiction_decay_ema", 0.0) or 0.0)))
            entry["contradiction_decay_ema"] = float(
                sample if previous_penalty <= 0.0 else 0.75 * previous_penalty + 0.25 * sample
            )
            applied.append(provider)
        if applied:
            autonomy["provider_curriculum"] = curriculum
        return applied

    def _apply_provider_forgiveness_locked(
        self,
        *,
        autonomy: dict[str, Any],
        provider_weights: Mapping[str, Any],
        forgiveness_score: float,
    ) -> list[str]:
        curriculum = self._normalize_provider_curriculum(autonomy.get("provider_curriculum"))
        calibrated_score = max(0.0, min(1.0, float(forgiveness_score)))
        if not curriculum or calibrated_score <= 0.0:
            return []
        applied: list[str] = []
        for raw_provider, raw_weight in dict(provider_weights).items():
            provider = " ".join(str(raw_provider).split()).strip().lower()
            weight = max(0.0, min(1.0, float(raw_weight or 0.0)))
            if not provider or weight <= 0.0:
                continue
            entry = curriculum.get(provider)
            if not isinstance(entry, Mapping):
                continue
            previous_penalty = max(0.0, min(1.0, float(entry.get("contradiction_decay_ema", 0.0) or 0.0)))
            if previous_penalty <= 0.0:
                continue
            reduction = min(previous_penalty, float(calibrated_score) * float(weight))
            if reduction <= 0.0:
                continue
            entry["contradiction_decay_ema"] = float(max(0.0, previous_penalty - reduction))
            applied.append(provider)
        if applied:
            autonomy["provider_curriculum"] = curriculum
        return applied

    def _apply_provider_delayed_consequence_locked(
        self,
        *,
        autonomy: dict[str, Any],
        provider_weights: Mapping[str, Any],
        consequence_score: float,
    ) -> list[str]:
        curriculum = self._normalize_provider_curriculum(autonomy.get("provider_curriculum"))
        calibrated_score = max(0.0, min(1.0, float(consequence_score)))
        if not curriculum or calibrated_score <= 0.0:
            return []
        applied: list[str] = []
        for raw_provider, raw_weight in dict(provider_weights).items():
            provider = " ".join(str(raw_provider).split()).strip().lower()
            weight = max(0.0, min(1.0, float(raw_weight or 0.0)))
            if not provider or weight <= 0.0:
                continue
            entry = curriculum.get(provider)
            if not isinstance(entry, Mapping):
                continue
            sample = max(0.0, min(1.0, calibrated_score * weight))
            previous_delayed = max(0.0, min(1.0, float(entry.get("delayed_consequence_ema", 0.0) or 0.0)))
            entry["delayed_consequence_ema"] = float(
                sample if previous_delayed <= 0.0 else 0.75 * previous_delayed + 0.25 * sample
            )
            previous_utility = max(0.0, min(1.0, float(entry.get("utility_ema", 0.0) or 0.0)))
            reinforced_utility = max(
                previous_utility,
                float(entry.get("grounded_outcome_ema", 0.0) or 0.0),
                float(entry.get("delayed_consequence_ema", 0.0) or 0.0),
            )
            entry["utility_ema"] = float(
                reinforced_utility if previous_utility <= 0.0 else 0.80 * previous_utility + 0.20 * reinforced_utility
            )
            applied.append(provider)
        if applied:
            autonomy["provider_curriculum"] = curriculum
        return applied

    def _apply_provider_family_summary_locked(
        self,
        *,
        autonomy: dict[str, Any],
        provider_weights: Mapping[str, Any],
        family_summary_score: float,
    ) -> list[str]:
        curriculum = self._normalize_provider_curriculum(autonomy.get("provider_curriculum"))
        calibrated_score = max(0.0, min(1.0, float(family_summary_score)))
        if not curriculum:
            return []
        applied: list[str] = []
        for raw_provider, raw_weight in dict(provider_weights).items():
            provider = " ".join(str(raw_provider).split()).strip().lower()
            weight = max(0.0, min(1.0, float(raw_weight or 0.0)))
            if not provider or weight <= 0.0:
                continue
            entry = curriculum.get(provider)
            if not isinstance(entry, Mapping):
                continue
            sample = max(0.0, min(1.0, calibrated_score * weight))
            previous_summary = max(0.0, min(1.0, float(entry.get("grounded_family_summary_ema", 0.0) or 0.0)))
            entry["grounded_family_summary_ema"] = float(
                sample if previous_summary <= 0.0 else 0.70 * previous_summary + 0.30 * sample
            )
            previous_utility = max(0.0, min(1.0, float(entry.get("utility_ema", 0.0) or 0.0)))
            reinforced_utility = max(
                previous_utility,
                float(entry.get("grounded_outcome_ema", 0.0) or 0.0),
                float(entry.get("delayed_consequence_ema", 0.0) or 0.0),
                float(entry.get("grounded_family_summary_ema", 0.0) or 0.0),
            )
            entry["utility_ema"] = float(
                reinforced_utility if previous_utility <= 0.0 else 0.80 * previous_utility + 0.20 * reinforced_utility
            )
            applied.append(provider)
        if applied:
            autonomy["provider_curriculum"] = curriculum
        return applied

    def _apply_delayed_query_consequence_locked(
        self,
        *,
        query_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        remerge = self._remerge_converged_delayed_consequence_families_locked()
        split = self._split_divergent_delayed_consequence_families_locked()
        compaction = self._compact_delayed_consequence_records_locked()
        maintenance = self._cool_delayed_consequence_records_locked()
        summary = {
            "enabled": True,
            "record_count": int(len(self._delayed_consequence_records)),
            "matched_records": 0,
            "credited_records": 0,
            "penalized_records": 0,
            "forgiven_records": 0,
            "remerged_records": int(remerge.get("remerged_records", 0) or 0),
            "split_records": int(split.get("split_records", 0) or 0),
            "max_split_branch_overlap": float(split.get("max_branch_overlap", 1.0) or 1.0),
            "compacted_records": int(compaction.get("compacted_records", 0) or 0),
            "max_aggregate_count": int(compaction.get("max_aggregate_count", 1) or 1),
            "cooled_records": int(maintenance.get("cooled_records", 0) or 0),
            "retired_records": int(maintenance.get("retired_records", 0) or 0),
            "credited_source_names": [],
            "credited_providers": [],
            "penalized_source_names": [],
            "penalized_providers": [],
            "forgiven_source_names": [],
            "forgiven_providers": [],
            "max_improvement": 0.0,
            "max_penalty": 0.0,
            "max_regression": 0.0,
            "max_forgiveness": 0.0,
            "max_family_summary_score": 0.0,
            "max_cooling_delta": float(maintenance.get("max_cooling_delta", 0.0) or 0.0),
            "contradicted_action_count": 0,
            "contradiction_signal": 0.0,
        }
        if not self._delayed_consequence_records:
            return summary
        query_snapshot = self._query_progress_snapshot_locked(query_result)
        if not query_snapshot.get("query_text") or not list(query_snapshot.get("query_terms") or []):
            return summary
        autonomy = cast(dict[str, Any] | None, self._brain_config.get("autonomy"))
        contradiction_signal, contradicted_action_count = self._recent_action_contradiction_signal_locked(
            str(query_snapshot.get("query_text", "")),
        )
        summary["contradicted_action_count"] = int(contradicted_action_count)
        summary["contradiction_signal"] = float(contradiction_signal)
        credited_sources: set[str] = set()
        credited_providers: set[str] = set()
        penalized_sources: set[str] = set()
        penalized_providers: set[str] = set()
        forgiven_sources: set[str] = set()
        forgiven_providers: set[str] = set()
        mutated = False
        timestamp = datetime.now(timezone.utc).isoformat()
        current_token = int(self._trainer.token_count)
        current_query_score = max(0.0, min(1.0, float(query_snapshot.get("query_score", 0.0) or 0.0)))
        current_grounded_fraction = max(0.0, min(1.0, float(query_snapshot.get("grounded_fraction", 0.0) or 0.0)))
        query_terms = list(query_snapshot.get("query_terms") or [])
        unsupported_terms = list(query_snapshot.get("unsupported_terms") or [])
        unsupported_ratio = min(1.0, float(len(unsupported_terms)) / float(max(1, len(query_terms))))
        for record in list(self._delayed_consequence_records):
            match_score = self._delayed_consequence_match_score_locked(record, query_snapshot)
            if match_score < DEFAULT_DELAYED_CONSEQUENCE_MATCH_THRESHOLD:
                continue
            summary["matched_records"] = int(summary["matched_records"]) + 1
            best_query_score = max(
                0.0,
                min(
                    1.0,
                    max(
                        float(record.get("baseline_query_score", 0.0) or 0.0),
                        float(record.get("best_query_score", 0.0) or 0.0),
                    ),
                ),
            )
            best_grounded_fraction = max(
                0.0,
                min(
                    1.0,
                    max(
                        float(record.get("baseline_grounded_fraction", 0.0) or 0.0),
                        float(record.get("best_grounded_fraction", 0.0) or 0.0),
                    ),
                ),
            )
            unresolved_penalty_balance = max(
                0.0,
                min(1.0, float(record.get("unresolved_penalty_balance", 0.0) or 0.0)),
            )
            score_improvement = max(0.0, current_query_score - best_query_score)
            grounded_improvement = max(0.0, current_grounded_fraction - best_grounded_fraction)
            improvement = max(score_improvement, 0.85 * grounded_improvement)
            if improvement >= DEFAULT_DELAYED_CONSEQUENCE_DELTA_THRESHOLD:
                support_multiplier = self._delayed_consequence_family_support_multiplier(record, mode="credit")
                delayed_sample = max(
                    0.0,
                    min(
                        1.0,
                        1.5
                        * float(record.get("outcome_score", 0.0) or 0.0)
                        * float(match_score)
                        * float(improvement)
                        * float(support_multiplier),
                    ),
                )
                if delayed_sample > 0.0:
                    applied_sources = self._apply_background_source_delayed_consequence_locked(
                        source_weights=cast(Mapping[str, Any], record.get("source_weights") or {}),
                        consequence_score=delayed_sample,
                    )
                    applied_providers = []
                    if autonomy is not None:
                        applied_providers = self._apply_provider_delayed_consequence_locked(
                            autonomy=autonomy,
                            provider_weights=cast(Mapping[str, Any], record.get("provider_weights") or {}),
                            consequence_score=delayed_sample,
                        )
                    if applied_sources or applied_providers:
                        mutated = True
                        credited_sources.update(applied_sources)
                        credited_providers.update(applied_providers)
                        summary["credited_records"] = int(summary["credited_records"]) + 1
                        summary["max_improvement"] = max(float(summary["max_improvement"]), float(improvement))
                        record["best_query_score"] = max(best_query_score, current_query_score)
                        record["best_grounded_fraction"] = max(best_grounded_fraction, current_grounded_fraction)
                        record["credit_events"] = int(record.get("credit_events", 0)) + 1
                        record["resolved_improvement"] = max(
                            float(record.get("resolved_improvement", 0.0) or 0.0),
                            float(improvement),
                        )
                        record["last_match_score"] = float(match_score)
                        record["last_evaluated_at"] = timestamp
                        record["last_evaluated_query_text"] = str(query_snapshot.get("query_text", ""))
                        record["last_evaluated_token_count"] = int(current_token)
                        record["last_activity_token_count"] = int(current_token)
                        record["last_credit_token_count"] = int(current_token)
                        record["last_cooling_token_count"] = int(current_token)
                        record["last_credit_score"] = float(delayed_sample)
                        record["unresolved_penalty_balance"] = float(unresolved_penalty_balance)
                        self._update_delayed_consequence_trajectory_locked(
                            record,
                            event_type="credit",
                            event_score=delayed_sample,
                            timestamp=timestamp,
                            current_token=current_token,
                        )
                        self._update_delayed_consequence_branch_partition_locked(
                            record,
                            event_type="credit",
                            query_text=str(query_snapshot.get("query_text", "")),
                        )
                        forgiveness_budget = min(
                            unresolved_penalty_balance,
                            float(DEFAULT_FORGIVENESS_RECOVERY_RATIO) * float(delayed_sample),
                        )
                        if forgiveness_budget > 0.0:
                            forgiven_sources_now = self._apply_background_source_forgiveness_locked(
                                source_weights=cast(Mapping[str, Any], record.get("source_weights") or {}),
                                forgiveness_score=forgiveness_budget,
                            )
                            forgiven_providers_now = []
                            if autonomy is not None:
                                forgiven_providers_now = self._apply_provider_forgiveness_locked(
                                    autonomy=autonomy,
                                    provider_weights=cast(Mapping[str, Any], record.get("provider_weights") or {}),
                                    forgiveness_score=forgiveness_budget,
                                )
                            if forgiven_sources_now or forgiven_providers_now:
                                forgiven_sources.update(forgiven_sources_now)
                                forgiven_providers.update(forgiven_providers_now)
                                summary["forgiven_records"] = int(summary["forgiven_records"]) + 1
                                summary["max_forgiveness"] = max(
                                    float(summary["max_forgiveness"]),
                                    float(forgiveness_budget),
                                )
                                record["forgiveness_events"] = int(record.get("forgiveness_events", 0)) + 1
                                record["last_forgiveness_score"] = float(forgiveness_budget)
                                record["last_forgiveness_token_count"] = int(current_token)
                                record["last_activity_token_count"] = int(current_token)
                                record["last_cooling_token_count"] = int(current_token)
                                record["unresolved_penalty_balance"] = float(
                                    max(0.0, unresolved_penalty_balance - forgiveness_budget)
                                )
                                self._update_delayed_consequence_trajectory_locked(
                                    record,
                                    event_type="forgiveness",
                                    event_score=forgiveness_budget,
                                    timestamp=timestamp,
                                    current_token=current_token,
                                )
                                self._update_delayed_consequence_branch_partition_locked(
                                    record,
                                    event_type="forgiveness",
                                    query_text=str(query_snapshot.get("query_text", "")),
                                )
                        family_summary_score = self._grounded_family_summary_score(record)
                        summary["max_family_summary_score"] = max(
                            float(summary["max_family_summary_score"]),
                            float(family_summary_score),
                        )
                        self._apply_background_source_family_summary_locked(
                            source_weights=cast(Mapping[str, Any], record.get("source_weights") or {}),
                            family_summary_score=family_summary_score,
                        )
                        if autonomy is not None:
                            self._apply_provider_family_summary_locked(
                                autonomy=autonomy,
                                provider_weights=cast(Mapping[str, Any], record.get("provider_weights") or {}),
                                family_summary_score=family_summary_score,
                            )
                        continue
            expectation_score = max(
                float(record.get("outcome_score", 0.0) or 0.0),
                float(best_query_score),
                float(best_grounded_fraction),
            )
            regression = max(
                0.0,
                expectation_score - max(current_query_score, current_grounded_fraction),
                best_query_score - current_query_score,
                best_grounded_fraction - current_grounded_fraction,
            )
            contradiction_decay = max(
                float(regression),
                0.75 * unsupported_ratio
                if expectation_score >= 0.60 and unsupported_ratio >= DEFAULT_DELAYED_CONTRADICTION_UNSUPPORTED_THRESHOLD
                else 0.0,
                0.85 * float(contradiction_signal),
            )
            if contradiction_decay < DEFAULT_DELAYED_CONTRADICTION_DECAY_THRESHOLD:
                continue
            support_multiplier = self._delayed_consequence_family_support_multiplier(record, mode="penalty")
            penalty_sample = max(
                0.0,
                min(
                    1.0,
                    1.35
                    * max(0.35, float(record.get("outcome_score", 0.0) or 0.0))
                    * float(match_score)
                    * float(contradiction_decay)
                    * float(support_multiplier),
                ),
            )
            if penalty_sample <= 0.0:
                continue
            penalized_sources_now = self._apply_background_source_delayed_penalty_locked(
                source_weights=cast(Mapping[str, Any], record.get("source_weights") or {}),
                penalty_score=penalty_sample,
            )
            penalized_providers_now = []
            if autonomy is not None:
                penalized_providers_now = self._apply_provider_delayed_penalty_locked(
                    autonomy=autonomy,
                    provider_weights=cast(Mapping[str, Any], record.get("provider_weights") or {}),
                    penalty_score=penalty_sample,
                )
            if not penalized_sources_now and not penalized_providers_now:
                continue
            mutated = True
            penalized_sources.update(penalized_sources_now)
            penalized_providers.update(penalized_providers_now)
            summary["penalized_records"] = int(summary["penalized_records"]) + 1
            summary["max_penalty"] = max(float(summary["max_penalty"]), float(penalty_sample))
            summary["max_regression"] = max(float(summary["max_regression"]), float(regression))
            record["penalty_events"] = int(record.get("penalty_events", 0)) + 1
            record["max_regression"] = max(float(record.get("max_regression", 0.0) or 0.0), float(regression))
            record["max_contradiction_signal"] = max(
                float(record.get("max_contradiction_signal", 0.0) or 0.0),
                float(contradiction_signal),
            )
            record["unresolved_penalty_balance"] = float(min(1.0, unresolved_penalty_balance + penalty_sample))
            record["last_match_score"] = float(match_score)
            record["last_evaluated_at"] = timestamp
            record["last_evaluated_query_text"] = str(query_snapshot.get("query_text", ""))
            record["last_evaluated_token_count"] = int(current_token)
            record["last_activity_token_count"] = int(current_token)
            record["last_penalty_token_count"] = int(current_token)
            record["last_cooling_token_count"] = int(current_token)
            record["last_penalty_score"] = float(penalty_sample)
            self._update_delayed_consequence_trajectory_locked(
                record,
                event_type="penalty",
                event_score=penalty_sample,
                timestamp=timestamp,
                current_token=current_token,
            )
            self._update_delayed_consequence_branch_partition_locked(
                record,
                event_type="penalty",
                query_text=str(query_snapshot.get("query_text", "")),
            )
            record["last_penalty_reason"] = (
                "contradicted_action"
                if contradiction_signal >= max(float(regression), 0.75 * unsupported_ratio)
                else "unsupported_decay"
                if unsupported_ratio >= DEFAULT_DELAYED_CONTRADICTION_UNSUPPORTED_THRESHOLD
                else "regression_decay"
            )
            family_summary_score = self._grounded_family_summary_score(record)
            summary["max_family_summary_score"] = max(
                float(summary["max_family_summary_score"]),
                float(family_summary_score),
            )
            self._apply_background_source_family_summary_locked(
                source_weights=cast(Mapping[str, Any], record.get("source_weights") or {}),
                family_summary_score=family_summary_score,
            )
            if autonomy is not None:
                self._apply_provider_family_summary_locked(
                    autonomy=autonomy,
                    provider_weights=cast(Mapping[str, Any], record.get("provider_weights") or {}),
                    family_summary_score=family_summary_score,
                )
        if credited_sources:
            summary["credited_source_names"] = sorted(credited_sources)
        if credited_providers:
            summary["credited_providers"] = sorted(credited_providers)
        if penalized_sources:
            summary["penalized_source_names"] = sorted(penalized_sources)
        if penalized_providers:
            summary["penalized_providers"] = sorted(penalized_providers)
        if forgiven_sources:
            summary["forgiven_source_names"] = sorted(forgiven_sources)
        if forgiven_providers:
            summary["forgiven_providers"] = sorted(forgiven_providers)
        if mutated:
            if int(summary["credited_records"]) > 0:
                self._record_brain_event_locked(
                    {
                        "type": "delayed_consequence_applied",
                        "timestamp": timestamp,
                        "query_text": str(query_snapshot.get("query_text", "")),
                        "credited_records": int(summary["credited_records"]),
                        "credited_source_names": sorted(credited_sources),
                        "credited_providers": sorted(credited_providers),
                        "max_improvement": float(summary["max_improvement"]),
                    }
                )
            if int(summary["penalized_records"]) > 0:
                self._record_brain_event_locked(
                    {
                        "type": "delayed_consequence_penalized",
                        "timestamp": timestamp,
                        "query_text": str(query_snapshot.get("query_text", "")),
                        "penalized_records": int(summary["penalized_records"]),
                        "penalized_source_names": sorted(penalized_sources),
                        "penalized_providers": sorted(penalized_providers),
                        "max_penalty": float(summary["max_penalty"]),
                        "max_regression": float(summary["max_regression"]),
                        "contradiction_signal": float(contradiction_signal),
                    }
                )
            if int(summary["forgiven_records"]) > 0:
                self._record_brain_event_locked(
                    {
                        "type": "delayed_consequence_forgiven",
                        "timestamp": timestamp,
                        "query_text": str(query_snapshot.get("query_text", "")),
                        "forgiven_records": int(summary["forgiven_records"]),
                        "forgiven_source_names": sorted(forgiven_sources),
                        "forgiven_providers": sorted(forgiven_providers),
                        "max_forgiveness": float(summary["max_forgiveness"]),
                    }
                )
            remerge_post = self._remerge_converged_delayed_consequence_families_locked()
            summary["remerged_records"] = int(summary["remerged_records"]) + int(remerge_post.get("remerged_records", 0) or 0)
            split_post = self._split_divergent_delayed_consequence_families_locked()
            summary["split_records"] = int(summary["split_records"]) + int(split_post.get("split_records", 0) or 0)
            summary["max_split_branch_overlap"] = min(
                float(summary["max_split_branch_overlap"]),
                float(split_post.get("max_branch_overlap", 1.0) or 1.0),
            )
            summary["record_count"] = int(len(self._delayed_consequence_records))
            self._mark_mutated()
        return summary

    def _record_response_consequence_candidate_locked(
        self,
        *,
        query_result: Mapping[str, Any],
        response: Mapping[str, Any],
        outcome_score: float,
    ) -> dict[str, Any] | None:
        source_weights = self._selected_evidence_weight_map(
            response,
            singular_field="source_name",
            plural_field="source_names",
        )
        provider_weights = self._selected_evidence_weight_map(
            response,
            singular_field="provider",
            plural_field="providers",
        )
        if not source_weights and not provider_weights:
            return None
        query_snapshot = self._query_progress_snapshot_locked(query_result)
        current_token = int(self._trainer.token_count)
        normalized = self._normalize_delayed_consequence_record(
            {
                "record_id": str(uuid4()),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "created_token_count": int(current_token),
                "origin": "response_selected_evidence",
                "query_text": str(query_snapshot.get("query_text", "")),
                "query_terms": list(query_snapshot.get("query_terms") or []),
                "baseline_grounded_fraction": float(query_snapshot.get("grounded_fraction", 0.0) or 0.0),
                "best_grounded_fraction": float(query_snapshot.get("grounded_fraction", 0.0) or 0.0),
                "baseline_query_score": float(query_snapshot.get("query_score", 0.0) or 0.0),
                "best_query_score": float(query_snapshot.get("query_score", 0.0) or 0.0),
                "outcome_score": float(outcome_score),
                "source_weights": dict(source_weights),
                "provider_weights": dict(provider_weights),
                "last_activity_token_count": int(current_token),
                "last_evaluated_token_count": int(current_token),
                "last_cooling_token_count": int(current_token),
            }
        )
        if normalized is None:
            return None
        merged_record = self._upsert_delayed_consequence_record_locked(normalized)
        return {
            "record_id": str(merged_record.get("record_id", "")),
            "query_text": str(merged_record.get("query_text", "")),
            "query_examples": self._delayed_consequence_query_examples(merged_record),
            "aggregate_count": int(merged_record.get("aggregate_count", 1) or 1),
            "source_names": sorted(dict(merged_record.get("source_weights") or {}).keys()),
            "providers": sorted(dict(merged_record.get("provider_weights") or {}).keys()),
            "baseline_query_score": float(merged_record.get("baseline_query_score", 0.0) or 0.0),
            "baseline_grounded_fraction": float(merged_record.get("baseline_grounded_fraction", 0.0) or 0.0),
        }

    def _apply_background_source_response_provenance_locked(
        self,
        *,
        response: Mapping[str, Any],
        outcome_score: float,
    ) -> bool:
        weighted_sources = self._selected_evidence_weight_map(
            response,
            singular_field="source_name",
            plural_field="source_names",
        )
        if not weighted_sources:
            return False
        applied = False
        for runtime in self._brain_source_runtimes:
            if runtime.name not in weighted_sources:
                continue
            entry = self._background_source_utility_entry_locked(runtime)
            sample = max(0.0, min(1.0, float(outcome_score) * float(weighted_sources[runtime.name])))
            previous_outcome = max(0.0, min(1.0, float(entry.get("grounded_outcome_ema", 0.0) or 0.0)))
            entry["grounded_outcome_ema"] = float(sample if previous_outcome <= 0.0 else 0.70 * previous_outcome + 0.30 * sample)
            previous_utility = max(0.0, min(1.0, float(entry.get("utility_ema", 0.0) or 0.0)))
            reinforced_utility = max(previous_utility, float(entry["grounded_outcome_ema"]))
            entry["utility_ema"] = float(
                reinforced_utility if previous_utility <= 0.0 else 0.75 * previous_utility + 0.25 * reinforced_utility
            )
            applied = True
        if applied:
            self._mark_mutated()
        return applied

    def _apply_background_source_outcome_calibration_locked(
        self,
        *,
        query_text: str,
        outcome_score: float,
    ) -> None:
        calibrated_score = max(0.0, min(1.0, float(outcome_score)))
        if calibrated_score <= 0.0 or not self._brain_source_runtimes:
            return
        focus_terms = self._background_focus_terms_locked(limit=12)
        query_terms = [
            _canonical_provider_term(term)
            for term in salient_query_terms(query_text)
            if _canonical_provider_term(term)
        ]
        combined_focus_terms = list(dict.fromkeys([*query_terms, *focus_terms]))[:12]
        ranked: list[tuple[float, float, float, _BrainSourceRuntime]] = []
        for runtime in self._brain_source_runtimes:
            entry = self._background_source_utility_entry_locked(runtime)
            if int(entry.get("selections", 0)) <= 0 and float(entry.get("utility_ema", 0.0) or 0.0) <= 0.0:
                continue
            semantic_alignment = self._brain_source_semantic_match_locked(runtime, combined_focus_terms)
            historical_alignment = max(
                float(entry.get("semantic_alignment_ema", 0.0) or 0.0),
                float(entry.get("focus_overlap_ema", 0.0) or 0.0),
            )
            utility_signal = max(
                0.0,
                float(entry.get("utility_ema", 0.0) or 0.0)
                - DEFAULT_UTILITY_PENALTY_WEIGHT * float(entry.get("contradiction_decay_ema", 0.0) or 0.0),
            )
            priority = max(semantic_alignment, historical_alignment) * max(0.35, utility_signal if utility_signal > 0.0 else 0.35)
            ranked.append((float(priority), float(semantic_alignment), float(utility_signal), runtime))
        if not ranked:
            return
        ranked.sort(key=lambda item: (-float(item[0]), -float(item[1]), -float(item[2]), item[3].name))
        priority, semantic_alignment, _utility_signal, runtime = ranked[0]
        if float(priority) <= 0.0:
            return
        entry = self._background_source_utility_entry_locked(runtime)
        sample = max(0.0, min(1.0, calibrated_score * max(float(priority), float(semantic_alignment), 0.35)))
        previous_outcome = max(0.0, min(1.0, float(entry.get("grounded_outcome_ema", 0.0) or 0.0)))
        entry["grounded_outcome_ema"] = float(sample if previous_outcome <= 0.0 else 0.70 * previous_outcome + 0.30 * sample)
        previous_utility = max(0.0, min(1.0, float(entry.get("utility_ema", 0.0) or 0.0)))
        reinforced_utility = max(previous_utility, float(entry["grounded_outcome_ema"]))
        entry["utility_ema"] = float(
            reinforced_utility
            if previous_utility <= 0.0
            else 0.75 * previous_utility + 0.25 * reinforced_utility
        )
        self._mark_mutated()

    def _ordered_brain_runtime_indices_locked(
        self,
        *,
        start_index: int,
        excluded_indices: set[int] | None = None,
    ) -> tuple[list[int], list[str], float]:
        excluded = excluded_indices or set()
        focus_plan = self._autonomy_focus_plan_locked()
        focus_terms = self._background_focus_terms_locked(focus_plan=focus_plan)
        focus_pressure, _focus_pressure_details = self._autonomy_focus_pressure_locked(focus_plan)
        tick_tokens = int(self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS))
        source_count = max(1, len(self._brain_source_runtimes))
        ranked: list[tuple[int, float, float, float, float, int, str]] = []
        for idx, runtime in enumerate(self._brain_source_runtimes):
            if idx in excluded or runtime.exhausted:
                continue
            score, semantic_match, fairness, readiness, effective_utility = self._brain_source_selection_score_locked(
                runtime,
                focus_terms=focus_terms,
                focus_pressure=focus_pressure,
                tick_tokens=tick_tokens,
            )
            cyclic_distance = (idx - start_index) % source_count
            ranked.append(
                (
                    idx,
                    float(score),
                    float(semantic_match),
                    float(effective_utility),
                    float(fairness),
                    float(readiness),
                    int(cyclic_distance),
                    str(runtime.name),
                )
            )
        ranked.sort(
            key=lambda item: (
                -float(item[1]),
                -float(item[2]),
                -float(item[3]),
                -float(item[4]),
                -float(item[5]),
                int(item[6]),
                item[7],
            )
        )
        return [int(item[0]) for item in ranked], focus_terms, float(focus_pressure)

    def _cross_modal_confidence_means_locked(self) -> tuple[float, float]:
        cross_modal = getattr(self._trainer.model, "cross_modal", None)
        if cross_modal is None:
            return 0.0, 0.0
        try:
            visual_conf = float(cross_modal.visual_confidence.mean().item())
        except Exception:
            visual_conf = 0.0
        try:
            audio_conf = float(cross_modal.audio_confidence.mean().item())
        except Exception:
            audio_conf = 0.0
        return max(0.0, min(1.0, visual_conf)), max(0.0, min(1.0, audio_conf))

    @staticmethod
    def _sensory_runtime_modalities(adapter: str) -> tuple[bool, bool]:
        cleaned = str(adapter).strip().lower()
        if cleaned == "s1_mmalign":
            return True, False
        if cleaned == "audiocaps":
            return False, True
        return False, False

    def _sensory_focus_terms_locked(self, limit: int = 12) -> list[str]:
        phrases: list[str] = []
        if self._thought_loop is not None and hasattr(self._thought_loop, "gate"):
            target = str(getattr(self._thought_loop.gate, "active_exploration_target", "")).strip()
            if target:
                phrases.append(target)
        if self._brain_recent_query_gaps:
            recent_gap = self._brain_recent_query_gaps[0]
            phrases.append(str(recent_gap.get("query_text", "")))
            phrases.extend(str(term) for term in list(recent_gap.get("unsupported_terms") or [])[:4])
            phrases.extend(
                str(item.get("term", ""))
                for item in list(recent_gap.get("gap_terms") or [])[:4]
                if isinstance(item, dict)
            )
        if not phrases:
            phrases.extend(self._focus_gap_terms_locked(limit=max(4, limit // 2)))

        ordered: list[str] = []
        seen: set[str] = set()
        for phrase in phrases:
            for term in salient_query_terms(str(phrase)):
                cleaned = " ".join(str(term).split()).strip().lower()
                if len(cleaned) < 4 or cleaned in seen:
                    continue
                seen.add(cleaned)
                ordered.append(cleaned)
                if len(ordered) >= max(1, limit):
                    return ordered
        return ordered

    @staticmethod
    def _sensory_source_topic_terms(runtime: _SensorySourceRuntime) -> set[str]:
        terms: set[str] = set()
        for raw in list(runtime.spec.get("topic_terms") or []):
            for term in salient_query_terms(str(raw)):
                cleaned = " ".join(str(term).split()).strip().lower()
                if len(cleaned) >= 4:
                    terms.add(cleaned)
        metadata = runtime.spec.get("metadata")
        if isinstance(metadata, dict):
            for key in ("role", "label"):
                for term in salient_query_terms(str(metadata.get(key, ""))):
                    cleaned = " ".join(str(term).split()).strip().lower()
                    if len(cleaned) >= 4:
                        terms.add(cleaned)
        return terms

    @staticmethod
    def _sensory_episode_terms(episode: SensoryEpisode) -> set[str]:
        terms: set[str] = set()
        text_parts = [str(episode.text)]
        metadata = episode.metadata if isinstance(episode.metadata, Mapping) else {}
        for key in ("title", "caption", "categories", "summary", "label", "observation"):
            value = metadata.get(key)
            if isinstance(value, str):
                text_parts.append(value)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                text_parts.extend(str(item) for item in list(value) if str(item).strip())
        for chunk in text_parts:
            for term in salient_query_terms(str(chunk)):
                cleaned = " ".join(str(term).split()).strip().lower()
                if len(cleaned) >= 4:
                    terms.add(cleaned)
        return terms

    def _sensory_episode_semantic_match_locked(
        self,
        episode: SensoryEpisode,
        focus_terms: Sequence[str] | None = None,
    ) -> float:
        normalized_focus = [
            " ".join(str(term).split()).strip().lower()
            for term in list(focus_terms or self._sensory_focus_terms_locked())
            if " ".join(str(term).split()).strip()
        ]
        episode_terms = self._sensory_episode_terms(episode)
        if not normalized_focus or not episode_terms:
            return 0.0
        focus_set = set(normalized_focus)
        overlap = len(focus_set & episode_terms) / max(1.0, min(float(len(focus_set)), float(len(episode_terms))))
        head_hits = sum(1 for term in normalized_focus[:3] if term in episode_terms)
        head_bonus = min(1.0, 0.5 * head_hits)
        combined_text = " ".join(
            part
            for part in [
                str(episode.text),
                *(str(value) for value in list((episode.metadata or {}).values()) if isinstance(value, str)),
            ]
            if part
        ).lower()
        phrase_hits = sum(1 for term in normalized_focus[:4] if term and term in combined_text)
        phrase_bonus = min(1.0, 0.34 * phrase_hits)
        return max(0.0, min(1.0, 0.55 * overlap + 0.30 * head_bonus + 0.15 * phrase_bonus))

    def _sensory_semantic_match_locked(
        self,
        runtime: _SensorySourceRuntime,
        focus_terms: Sequence[str] | None = None,
    ) -> float:
        normalized_focus = [
            " ".join(str(term).split()).strip().lower()
            for term in list(focus_terms or self._sensory_focus_terms_locked())
            if " ".join(str(term).split()).strip()
        ]
        source_terms = self._sensory_source_topic_terms(runtime)
        if not normalized_focus or not source_terms:
            return 0.0
        focus_set = set(normalized_focus)
        overlap = len(focus_set & source_terms) / max(1.0, min(float(len(focus_set)), float(len(source_terms))))
        head_hits = sum(1 for term in normalized_focus[:3] if term in source_terms)
        head_bonus = min(1.0, 0.5 * head_hits)
        return max(0.0, min(1.0, 0.65 * overlap + 0.35 * head_bonus))

    def _sensory_selection_score_locked(
        self,
        runtime: _SensorySourceRuntime,
        *,
        focus_terms: Sequence[str],
    ) -> tuple[float, float, float]:
        semantic_match = self._sensory_semantic_match_locked(runtime, focus_terms)
        modality_need = self._sensory_modality_need_locked(runtime.adapter)
        source_count = max(1, len(self._sensory_source_runtimes))
        min_episodes = min((rt.episodes_processed for rt in self._sensory_source_runtimes), default=0)
        fairness = max(
            0.0,
            min(
                1.0,
                1.0 - max(0, runtime.episodes_processed - min_episodes) / float(source_count + 1),
            ),
        )
        freshness = 1.0 if runtime.last_activity_at is None else 0.0
        score = 0.46 * semantic_match + 0.34 * modality_need + 0.12 * fairness + 0.08 * freshness
        runtime.last_semantic_match = semantic_match
        runtime.last_modality_need = modality_need
        runtime.last_selection_score = score
        return score, semantic_match, modality_need

    def _select_sensory_runtime_locked(
        self,
        excluded_indices: set[int] | None = None,
    ) -> tuple[int, _SensorySourceRuntime, float, float, float] | None:
        excluded = excluded_indices or set()
        focus_terms = self._sensory_focus_terms_locked()
        self._last_sensory_focus_terms = tuple(focus_terms)
        best: tuple[int, _SensorySourceRuntime, float, float, float] | None = None
        for idx, runtime in enumerate(self._sensory_source_runtimes):
            if idx in excluded or runtime.exhausted:
                continue
            score, semantic_match, modality_need = self._sensory_selection_score_locked(
                runtime,
                focus_terms=focus_terms,
            )
            if best is None or score > best[4] + 1e-6:
                best = (idx, runtime, semantic_match, modality_need, score)
                continue
            if best is not None and abs(score - best[4]) <= 1e-6 and runtime.episodes_processed < best[1].episodes_processed:
                best = (idx, runtime, semantic_match, modality_need, score)
        return best

    def _sensory_item_retrieval_config_locked(self) -> tuple[int, float]:
        sensory = self._brain_config.get("sensory") or {}
        lookahead = max(1, int(sensory.get("item_retrieval_lookahead", 6)))
        semantic_weight = max(0.0, min(1.0, float(sensory.get("item_retrieval_semantic_weight", 0.72))))
        return lookahead, semantic_weight

    def _prefetch_sensory_runtime_unlocked(
        self,
        runtime: _SensorySourceRuntime,
        target_items: int,
        repeat_sources: bool,
        visual_dim: int,
        audio_dim: int,
        device: torch.device,
        stop_event: Event | None,
        *,
        warm_trigger: str,
        deadline_perf: float | None = None,
    ) -> dict[str, Any] | None:
        cycles = runtime.cycles_completed
        exhausted = runtime.exhausted
        new_stream = None
        prefetched_items = 0
        prefetch_duration_ms: float | None = None
        prefetch_at: str | None = None
        prefetch_error: str | None = None
        budget_exhausted = False
        if len(runtime.buffered_episodes) < target_items and not exhausted:
            started = time.perf_counter()
            try:
                while len(runtime.buffered_episodes) < target_items:
                    if stop_event is not None and stop_event.is_set():
                        return None
                    wait_timeout = None
                    if deadline_perf is not None:
                        remaining = deadline_perf - time.perf_counter()
                        if remaining <= 0.0:
                            budget_exhausted = True
                            break
                        wait_timeout = remaining
                    try:
                        runtime.exhausted = False
                        runtime.buffered_episodes.append(self._next_stream_item(runtime.stream, timeout=wait_timeout))
                        prefetched_items += 1
                    except TimeoutError:
                        budget_exhausted = True
                        break
                    except StopIteration:
                        if not repeat_sources:
                            exhausted = True
                            runtime.exhausted = True
                            break
                        cycles += 1
                        rebuilt = self._build_sensory_stream_from_spec(
                            runtime.spec,
                            visual_dim=visual_dim,
                            audio_dim=audio_dim,
                            device=device,
                        )
                        runtime.stream = rebuilt
                        new_stream = rebuilt
                        runtime.exhausted = False
                        try:
                            runtime.buffered_episodes.append(self._next_stream_item(runtime.stream, timeout=wait_timeout))
                            prefetched_items += 1
                        except TimeoutError:
                            budget_exhausted = True
                            break
                        except StopIteration:
                            exhausted = True
                            runtime.exhausted = True
                            break
                    if deadline_perf is not None and time.perf_counter() >= deadline_perf:
                        budget_exhausted = True
                        break
            except Exception as exc:
                if stop_event is not None and stop_event.is_set():
                    return None
                exhausted = True
                runtime.exhausted = True
                prefetch_error = str(exc)
            if prefetched_items > 0 or prefetch_error is not None:
                prefetch_duration_ms = float((time.perf_counter() - started) * 1000.0)
                prefetch_at = datetime.now(timezone.utc).isoformat()
        return {
            "runtime": runtime,
            "cycles": cycles,
            "exhausted": exhausted,
            "new_stream": new_stream,
            "prefetch_items": int(prefetched_items),
            "prefetch_duration_ms": prefetch_duration_ms,
            "prefetch_at": prefetch_at,
            "prefetch_error": prefetch_error,
            "budget_exhausted": bool(budget_exhausted),
            "warm_trigger": warm_trigger,
        }

    def _prefetch_sensory_queues_unlocked(
        self,
        runtimes: Sequence[_SensorySourceRuntime],
        target_items: int,
        repeat_sources: bool,
        visual_dim: int,
        audio_dim: int,
        device: torch.device,
        stop_event: Event | None,
        *,
        warm_trigger: str,
        deadline_perf: float | None = None,
    ) -> list[dict[str, Any]]:
        prefetched: list[dict[str, Any]] = []
        for runtime in runtimes:
            if stop_event is not None and stop_event.is_set():
                break
            meta = self._prefetch_sensory_runtime_unlocked(
                runtime,
                target_items,
                repeat_sources,
                visual_dim,
                audio_dim,
                device,
                stop_event,
                warm_trigger=warm_trigger,
                deadline_perf=deadline_perf,
            )
            if meta is not None:
                prefetched.append(meta)
        return prefetched

    def _commit_prefetched_sensory_runtime_locked(self, meta: dict[str, Any] | None) -> None:
        if meta is None:
            return
        runtime = meta["runtime"]
        runtime.cycles_completed = int(meta.get("cycles", runtime.cycles_completed))
        runtime.exhausted = bool(meta.get("exhausted", runtime.exhausted))
        if meta.get("new_stream") is not None:
            runtime.stream = meta["new_stream"]
        runtime.last_buffer_episodes_served = int(meta.get("served_items", 0) or 0)
        if bool(meta.get("queue_hit", False)):
            runtime.queue_hits += 1
        prefetched_items = int(meta.get("prefetch_items", 0) or 0)
        if prefetched_items > 0:
            runtime.prefetch_events += 1
            runtime.prefetched_episodes += prefetched_items
            runtime.last_prefetch_episode_count = prefetched_items
            runtime.last_prefetch_at = meta.get("prefetch_at")
            runtime.last_prefetch_duration_ms = meta.get("prefetch_duration_ms")
        prefetch_error = meta.get("prefetch_error")
        runtime.last_prefetch_error = None if prefetch_error in (None, "") else str(prefetch_error)
        if runtime.last_prefetch_error:
            self._real_sensory_last_error = runtime.last_prefetch_error
        self._update_sensory_runtime_cache_locked(runtime)
        self._maybe_mark_sensory_warm_locked(trigger=str(meta.get("warm_trigger", "sensory") or "sensory"))

    def _next_sensory_episode_locked(
        self,
        runtime: _SensorySourceRuntime,
        *,
        repeat_sources: bool,
        focus_terms: Sequence[str],
    ) -> SensoryEpisode | None:
        lookahead, semantic_weight = self._sensory_item_retrieval_config_locked()
        queue_target_items = self._sensory_queue_target_items_locked()
        visual_dim = int(getattr(self._trainer.config, "cross_modal_dim_visual", 64))
        audio_dim = int(getattr(self._trainer.config, "cross_modal_dim_audio", 64))
        buffer_before = len(runtime.buffered_episodes)
        fill_target = max(lookahead, queue_target_items) if buffer_before <= 0 else buffer_before
        deadline_perf = None
        if buffer_before <= 0 and self._sensory_spec_uses_live_remote(runtime.spec):
            deadline_perf = time.perf_counter() + float(DEFAULT_REMOTE_ACTIVE_FETCH_WAIT_SECONDS)
        meta = self._prefetch_sensory_runtime_unlocked(
            runtime,
            fill_target,
            repeat_sources,
            visual_dim,
            audio_dim,
            self._trainer.model.device,
            None,
            warm_trigger="sensory_tick",
            deadline_perf=deadline_perf,
        )
        self._commit_prefetched_sensory_runtime_locked(meta)
        if not runtime.buffered_episodes:
            if meta is not None and bool(meta.get("budget_exhausted", False)):
                self._start_remote_warm_promotion_locked(trigger="sensory_tick")
            runtime.last_item_semantic_match = 0.0
            runtime.last_item_candidates_considered = 0
            runtime.last_item_retrieval_lookahead = lookahead
            return None

        considered = min(len(runtime.buffered_episodes), lookahead)
        best_index = 0
        best_match = self._sensory_episode_semantic_match_locked(runtime.buffered_episodes[0], focus_terms)
        best_score = semantic_weight * best_match + (1.0 - semantic_weight)
        if considered > 1:
            denom = max(1, considered - 1)
            for idx, episode in enumerate(runtime.buffered_episodes[:considered]):
                item_match = self._sensory_episode_semantic_match_locked(episode, focus_terms)
                recency = 1.0 - (idx / float(denom))
                score = semantic_weight * item_match + (1.0 - semantic_weight) * recency
                if score > best_score + 1e-6:
                    best_index = idx
                    best_match = item_match
                    best_score = score
                    continue
                if abs(score - best_score) <= 1e-6 and item_match > best_match + 1e-6:
                    best_index = idx
                    best_match = item_match
                    best_score = score

        runtime.last_item_semantic_match = float(max(0.0, min(1.0, best_match)))
        runtime.last_item_candidates_considered = int(considered)
        runtime.last_item_retrieval_lookahead = int(lookahead)
        runtime.last_buffer_episodes_served = 1
        if buffer_before > 0 and int(meta.get("prefetch_items", 0) or 0) == 0:
            runtime.queue_hits += 1
        return runtime.buffered_episodes.pop(best_index)

    def _sensory_modality_need_locked(self, adapter: str) -> float:
        sensory = self._brain_config.get("sensory") or {}
        target_confidence = float(sensory.get("modality_target_confidence", 0.70))
        visual_conf, audio_conf = self._cross_modal_confidence_means_locked()
        use_visual, use_audio = self._sensory_runtime_modalities(adapter)
        confs: list[float] = []
        if use_visual:
            confs.append(visual_conf)
        if use_audio:
            confs.append(audio_conf)
        if not confs:
            return 0.0
        mean_conf = sum(confs) / float(len(confs))
        if mean_conf >= target_confidence:
            return 0.0
        return max(0.0, min(1.0, (target_confidence - mean_conf) / max(0.1, target_confidence)))

    def _sensory_window_budget_locked(
        self,
        runtime: _SensorySourceRuntime,
        *,
        semantic_match: float | None = None,
        modality_need: float | None = None,
    ) -> int:
        sensory = self._brain_config.get("sensory") or {}
        base_windows = max(1, int(sensory.get("base_windows_per_item", 4)))
        max_windows = max(base_windows, int(sensory.get("max_windows_per_item", 10)))
        confidence_gain = max(0.0, float(sensory.get("confidence_window_gain", 3.0)))
        semantic_gain = max(0.0, float(sensory.get("semantic_window_gain", 3.0)))
        need = runtime.last_modality_need if modality_need is None else max(0.0, min(1.0, float(modality_need)))
        semantic = runtime.last_semantic_match if semantic_match is None else max(0.0, min(1.0, float(semantic_match)))
        bonus = int(round(confidence_gain * need + semantic_gain * semantic))
        return max(base_windows, min(max_windows, base_windows + bonus))

    def _inject_sensory_observation_locked(
        self,
        *,
        runtime: _SensorySourceRuntime,
        episode: SensoryEpisode,
        last_metrics: dict[str, Any] | None,
        semantic_match: float | None = None,
        evidence_unit_count: int | None = None,
    ) -> dict[str, Any]:
        if self._thought_loop is None:
            return {"topics": [], "salience": 0.0}
        text = " ".join(str(episode.text).split()).strip()
        if not text:
            return {"topics": [], "salience": 0.0}
        sensory = self._brain_config.get("sensory") or {}
        base_salience = float(sensory.get("observation_salience", 0.82))
        modality_need = self._sensory_modality_need_locked(runtime.adapter)
        semantic_score = runtime.last_semantic_match if semantic_match is None else max(0.0, min(1.0, float(semantic_match)))
        accepted_bonus = 0.0
        if isinstance(last_metrics, dict):
            if last_metrics.get("cross_modal_visual_accepted"):
                accepted_bonus += 0.04
            if last_metrics.get("cross_modal_audio_accepted"):
                accepted_bonus += 0.04
        salience = max(
            0.25,
            min(
                0.98,
                base_salience
                + 0.10 * modality_need
                + 0.08 * semantic_score
                + accepted_bonus,
            ),
        )
        topics: list[str] = []
        if runtime.adapter == "s1_mmalign":
            title = " ".join(str(episode.metadata.get("title", "")).split()).strip()
            categories = " ".join(str(episode.metadata.get("categories", "")).split()).strip()
            if title:
                topics.extend(salient_query_terms(title)[:2])
            if categories:
                topics.extend(salient_query_terms(categories)[:2])
        focus_terms = list(self._last_sensory_focus_terms)[:4]
        topics.extend(focus_terms[:2])
        topics.extend(salient_query_terms(text)[:4])
        deduped_topics = self._dedupe_grounded_topics(topics, limit=6)
        modality = "image" if episode.visual_spikes is not None and episode.audio_spikes is None else (
            "audio" if episode.audio_spikes is not None and episode.visual_spikes is None else "multisensory"
        )
        normalized_units = max(1, int(evidence_unit_count or 1))
        grounding_signal = max(
            0.35,
            min(
                1.0,
                0.48 * semantic_score
                + 0.22 * modality_need
                + 0.20 * salience
                + 0.10 * min(1.0, float(normalized_units) / 4.0),
            ),
        )
        metadata = self._grounded_observation_metadata(
            observation_kind="sensory",
            source_name=runtime.name,
            source_type="sensory",
            salience=salience,
            grounding_signal=grounding_signal,
            evidence_unit_count=normalized_units,
            modality=modality,
            focus_terms=deduped_topics[:4],
            extra={
                "adapter": runtime.adapter,
                "semantic_match": float(semantic_score),
                "item_semantic_match": float(runtime.last_item_semantic_match),
            },
        )
        self._thought_loop.inject_observation(
            content=text,
            topics=deduped_topics,
            salience=salience,
            metadata=metadata,
        )
        return {
            "observation_kind": "sensory",
            "source_name": runtime.name,
            "source_type": "sensory",
            "adapter": runtime.adapter,
            "modality": modality,
            "content": text,
            "topics": deduped_topics,
            "salience": salience,
            "grounding_signal": grounding_signal,
            "evidence_unit_count": normalized_units,
            "semantic_match": float(semantic_score),
            "item_semantic_match": float(runtime.last_item_semantic_match),
            "focus_terms": focus_terms,
        }

    def _record_sensory_preview_locked(
        self,
        *,
        runtime: _SensorySourceRuntime,
        episode: SensoryEpisode,
        text: str,
        topics: Sequence[str],
        semantic_match: float,
        item_semantic_match: float,
        modality_need: float,
        selection_score: float,
        window_budget: int,
    ) -> None:
        if episode.visual_preview is None and episode.audio_preview is None:
            return
        metadata = {
            key: deepcopy(value)
            for key, value in (episode.metadata or {}).items()
            if key not in {"bytes", "raw_bytes"}
        }
        entry = {
            "preview_id": str(uuid4()),
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source_name": runtime.name,
            "adapter": runtime.adapter,
            "text": text[:480],
            "semantic_match": float(max(0.0, min(1.0, semantic_match))),
            "modality_need": float(max(0.0, min(1.0, modality_need))),
            "item_semantic_match": float(max(0.0, min(1.0, item_semantic_match))),
            "item_candidates_considered": int(max(0, runtime.last_item_candidates_considered)),
            "item_retrieval_lookahead": int(max(1, runtime.last_item_retrieval_lookahead or 1)),
            "selection_score": float(max(0.0, min(1.0, selection_score))),
            "window_budget": int(max(0, window_budget)),
            "topics": list(topics)[:8],
            "focus_terms": list(self._last_sensory_focus_terms)[:8],
            "metadata": metadata,
            "visual": deepcopy(episode.visual_preview),
            "audio": deepcopy(episode.audio_preview),
        }
        self._sensory_preview_history.appendleft(entry)

    def _run_real_sensory_episode_locked(self) -> dict[str, Any] | None:
        sensory = self._brain_config.get("sensory")
        if (
            not sensory
            or not sensory.get("enabled")
            or not getattr(self._trainer.config, "enable_cross_modal", False)
            or not self._sensory_source_runtimes
        ):
            return None

        self._request_active_execution_locked()
        try:
            current_tokens = int(self._trainer.token_count)
            trigger_interval = int(sensory.get("episode_interval_tokens", 2048))
            cooldown = float(sensory.get("cooldown_seconds", 10.0))
            now = time.time()
            if current_tokens - self._last_real_sensory_episode_token_count < trigger_interval:
                return None
            if (now - self._last_real_sensory_episode_time) < cooldown:
                return None

            items_per_episode = int(sensory.get("items_per_episode", 2))
            repeat_sources = bool(sensory.get("repeat_sources", True))
            source_count = len(self._sensory_source_runtimes)
            if source_count <= 0:
                return None
            self._sensory_stream_epoch += 1

            episodes_run = 0
            steps_trained = 0
            last_metrics: dict[str, Any] | None = None
            used_sources: list[dict[str, Any]] = []
            self._real_sensory_last_error = None

            selected_indices: set[int] = set()
            max_items = min(items_per_episode, source_count)
            for _ in range(max_items):
                selection = self._select_sensory_runtime_locked(selected_indices)
                if selection is None:
                    break
                idx, runtime, semantic_match, modality_need, selection_score = selection
                selected_indices.add(idx)
                focus_terms = list(self._last_sensory_focus_terms)
                episode = self._next_sensory_episode_locked(
                    runtime,
                    repeat_sources=repeat_sources,
                    focus_terms=focus_terms,
                )
                if episode is None:
                    continue
                runtime.exhausted = False
                self._sensory_source_index = (idx + 1) % source_count
                text = " ".join(str(episode.text).split()).strip()
                if not text:
                    continue
                if episode.visual_spikes is None and episode.audio_spikes is None:
                    continue

                effective_semantic_match = max(float(semantic_match), float(runtime.last_item_semantic_match))
                window_budget = self._sensory_window_budget_locked(
                    runtime,
                    semantic_match=effective_semantic_match,
                    modality_need=modality_need,
                )
                item_steps = 0
                last_raw_window = text
                for raw_window, pattern in self._encoder.iter_char_patterns(text, self._trainer.config.window_size):
                    last_raw_window = raw_window
                    last_metrics = self._trainer.train_step(
                        pattern,
                        raw_window=raw_window,
                        visual_spikes=episode.visual_spikes,
                        audio_spikes=episode.audio_spikes,
                    )
                    item_steps += 1
                    steps_trained += 1
                    if last_metrics:
                        if last_metrics.get("cross_modal_visual_accepted"):
                            self._real_visual_accepted += 1
                        if last_metrics.get("cross_modal_audio_accepted"):
                            self._real_audio_accepted += 1
                    if item_steps >= window_budget:
                        break

                if item_steps <= 0:
                    continue

                runtime.episodes_processed += 1
                runtime.last_activity_at = datetime.now(timezone.utc).isoformat()
                runtime.last_text = text[:160]
                self._observe_runtime_concepts_locked(raw_window=last_raw_window, metrics=last_metrics)
                runtime.last_window_budget = int(window_budget)
                observation = self._inject_sensory_observation_locked(
                    runtime=runtime,
                    episode=episode,
                    last_metrics=last_metrics,
                    semantic_match=semantic_match,
                    evidence_unit_count=window_budget,
                )
                self._record_sensory_preview_locked(
                    runtime=runtime,
                    episode=episode,
                    text=text,
                    topics=list(observation.get("topics") or []),
                    semantic_match=semantic_match,
                    item_semantic_match=runtime.last_item_semantic_match,
                    modality_need=modality_need,
                    selection_score=selection_score,
                    window_budget=window_budget,
                )
                used_sources.append(
                    {
                        "name": runtime.name,
                        "adapter": runtime.adapter,
                        "steps_trained": int(item_steps),
                        "window_budget": int(window_budget),
                        "semantic_match": float(semantic_match),
                        "item_semantic_match": float(runtime.last_item_semantic_match),
                        "item_candidates_considered": int(runtime.last_item_candidates_considered),
                        "modality_need": float(modality_need),
                        "selection_score": float(selection_score),
                        "has_visual": bool(episode.visual_spikes is not None),
                        "has_audio": bool(episode.audio_spikes is not None),
                        "grounded_observation": observation,
                    }
                )
                self._update_sensory_runtime_cache_locked(runtime, served_episodes=[episode])
                episodes_run += 1

            if episodes_run <= 0:
                return None

            self._real_sensory_episodes_completed += episodes_run
            self._last_real_sensory_episode_time = now
            self._last_real_sensory_episode_token_count = int(self._trainer.token_count)
            self._mark_mutated()
            return {
                "type": "real_sensory_episode",
                "episodes_completed": int(self._real_sensory_episodes_completed),
                "episode_count": int(episodes_run),
                "steps_trained": int(steps_trained),
                "sources": used_sources,
                "last_metrics": last_metrics,
            }
        finally:
            self._release_active_execution_locked()

    def _rebuild_brain_sources_locked(self) -> None:
        self._close_brain_sources_locked()
        self._close_sensory_sources_locked()
        self._brain_source_runtimes = [
            _BrainSourceRuntime(spec=deepcopy(spec), stream=self._build_brain_source_stream_locked(spec))
            for spec in self._brain_config.get("source_bank", [])
        ]
        for runtime in self._brain_source_runtimes:
            self._background_source_utility_entry_locked(runtime)
        sensory_config = self._brain_config.get("sensory") or {}
        self._sensory_source_runtimes = [
            _SensorySourceRuntime(spec=deepcopy(spec), stream=self._build_sensory_stream_locked(spec))
            for spec in sensory_config.get("source_bank", [])
        ]
        self._brain_source_index = 0
        self._sensory_source_index = 0
        self._brain_tick_count = 0
        self._brain_background_tokens = 0
        self._brain_autonomy_tokens = 0
        self._brain_source_utility = {
            name: value
            for name, value in self._brain_source_utility.items()
            if any(str(spec.get("name", "")).strip() == name for spec in self._brain_config.get("source_bank", []))
        }
        self._brain_last_tick_completed_at = None
        self._brain_last_tick_duration_ms = None
        self._brain_last_tick_token_delta = 0
        self._brain_last_work_at = None
        self._brain_stop_requested_at = None
        self._brain_stop_requested_reason = None
        self._brain_stop_requested_perf = None
        self._brain_stop_timed_out = False
        self._brain_last_stop_duration_ms = None
        self._real_sensory_episodes_completed = 0
        self._real_visual_accepted = 0
        self._real_audio_accepted = 0
        self._last_real_sensory_episode_time = 0.0
        self._last_real_sensory_episode_token_count = int(self._trainer.token_count)
        self._real_sensory_last_error = None
        self._last_sensory_focus_terms = ()
        self._sensory_preview_history.clear()
        self._ingestion_configured_at = (
            datetime.now(timezone.utc).isoformat() if self._brain_config.get("source_bank") else None
        )
        self._ingestion_configured_perf = time.perf_counter() if self._brain_config.get("source_bank") else None
        self._ingestion_prewarm_started_at = None
        self._ingestion_prewarm_started_perf = None
        self._ingestion_prewarm_completed_at = None
        self._ingestion_prewarm_last_duration_ms = None
        self._ingestion_prewarm_last_error = None
        self._ingestion_prewarm_run_count = 0
        self._ingestion_prewarm_last_trigger = None
        self._ingestion_prewarm_budget_exhausted = False
        self._ingestion_prewarm_running = False
        self._ingestion_prewarm_thread = None
        self._ingestion_prewarm_stop_event = None
        self._ingestion_warm_ready_at = None
        self._ingestion_startup_warm_latency_ms = None
        self._remote_warm_promotion_thread = None
        self._remote_warm_promotion_stop_event = None
        self._remote_warm_promotion_running = False
        self._remote_warm_promotion_last_trigger = None
        self._active_execution_requests = 0
        self._active_execution_idle_event.set()
        self._brain_stream_epoch += 1
        self._sensory_configured_at = (
            datetime.now(timezone.utc).isoformat() if self._sensory_source_runtimes else None
        )
        self._sensory_configured_perf = time.perf_counter() if self._sensory_source_runtimes else None
        self._sensory_prewarm_budget_exhausted = False
        self._sensory_warm_ready_at = None
        self._sensory_startup_warm_latency_ms = None
        self._sensory_stream_epoch += 1

        restored_text_sources = 0
        restored_text_tokens = 0
        for runtime in self._brain_source_runtimes:
            restored = self._restore_brain_runtime_cache_locked(runtime)
            if restored > 0:
                restored_text_sources += 1
                restored_text_tokens += restored
        if restored_text_sources > 0:
            self._maybe_mark_ingestion_warm_locked(trigger="cache_restore")
            self._record_brain_event_locked(
                {
                    "type": "ingestion_cache_restored",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_count": int(restored_text_sources),
                    "token_count": int(restored_text_tokens),
                }
            )

        restored_sensory_sources = 0
        restored_sensory_items = 0
        for runtime in self._sensory_source_runtimes:
            restored = self._restore_sensory_runtime_cache_locked(runtime)
            if restored > 0:
                restored_sensory_sources += 1
                restored_sensory_items += restored
        if restored_sensory_sources > 0:
            self._maybe_mark_sensory_warm_locked(trigger="cache_restore")
            self._record_brain_event_locked(
                {
                    "type": "sensory_cache_restored",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_count": int(restored_sensory_sources),
                    "item_count": int(restored_sensory_items),
                }
            )

    def _request_brain_stop(self, *, reason: str | None = None) -> Thread | None:
        with self._lock:
            return self._request_brain_stop_locked(reason=reason)

    def _finalize_brain_stop_locked(self, thread: Thread | None) -> None:
        active_thread = self._brain_thread
        if thread is not None and active_thread is not None and active_thread is not thread and active_thread.is_alive():
            return
        elapsed_ms = None
        if self._brain_stop_requested_perf is not None:
            elapsed_ms = (time.perf_counter() - self._brain_stop_requested_perf) * 1000.0
        self._brain_running = False
        self._brain_running_since = None
        self._brain_thread = None
        self._brain_stop_event = None
        event_reason = self._brain_stop_requested_reason
        timed_out = bool(self._brain_stop_timed_out)
        self._brain_stop_requested_perf = None
        self._brain_stop_requested_at = None
        self._brain_stop_requested_reason = None
        self._brain_stop_timed_out = False
        self._active_execution_requests = 0
        self._active_execution_idle_event.set()
        self._brain_last_stop_duration_ms = elapsed_ms
        if event_reason is not None:
            self._record_brain_event_locked(
                {
                    "type": "stopped_after_timeout" if timed_out else "stopped",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": event_reason,
                    "stop_duration_ms": None if elapsed_ms is None else float(elapsed_ms),
                }
            )

    def _join_brain_thread(
        self,
        thread: Thread | None,
        *,
        timeout: float = DEFAULT_BRAIN_STOP_TIMEOUT_SECONDS,
        raise_on_timeout: bool = True,
    ) -> bool:
        if thread is None:
            with self._lock:
                self._finalize_brain_stop_locked(thread)
            return True
        thread.join(timeout=timeout)
        if not thread.is_alive():
            with self._lock:
                self._finalize_brain_stop_locked(thread)
            return True

        message = (
            f"Terminus runtime did not stop within {timeout:.1f}s. "
            f"Reason={self._brain_stop_requested_reason or 'unknown'}"
        )
        with self._lock:
            self._brain_stop_timed_out = True
            self._brain_last_stop_duration_ms = float(timeout * 1000.0)
            self._brain_last_error = message
            self._record_brain_event_locked(
                {
                    "type": "stop_timeout",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": self._brain_stop_requested_reason,
                    "timeout_seconds": float(timeout),
                    "thread_alive": True,
                }
            )
        if raise_on_timeout:
            raise RuntimeError(message)
        _cortex_logger.warning(message)
        return False

    def _request_brain_stop_locked(self, *, reason: str | None = None) -> Thread | None:
        thread = self._brain_thread if self._brain_thread is not None and self._brain_thread.is_alive() else None
        stop_event = self._brain_stop_event
        if stop_event is not None:
            stop_event.set()
        self._brain_running = False
        if thread is not None:
            self._brain_stop_requested_at = datetime.now(timezone.utc).isoformat()
            self._brain_stop_requested_reason = reason
            self._brain_stop_requested_perf = time.perf_counter()
            self._brain_stop_timed_out = False
            self._interrupt_brain_sources_locked()
            self._interrupt_sensory_sources_locked()
            if reason is not None:
                self._record_brain_event_locked(
                    {
                        "type": "stop_requested",
                        "timestamp": self._brain_stop_requested_at,
                        "reason": reason,
                    }
                )
        else:
            self._finalize_brain_stop_locked(thread)
        return thread

    def _brain_loop(self) -> None:
        _SUB_BATCH = 8  # max tokens trained per lock acquisition
        _YIELD_SECONDS = 0.05  # 50ms yield between sub-batches for SSE/API
        while True:
            with self._lock:
                stop_event = self._brain_stop_event
                sleep_interval = float(self._brain_config.get("sleep_interval_seconds", DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS))
            if stop_event is None or stop_event.is_set():
                break
            try:
                self._request_active_execution()
                try:
                    with self._brain_execution_lock:
                        result = self._run_brain_tick_once(
                            stop_event=stop_event,
                            sub_batch_size=_SUB_BATCH,
                            yield_seconds=_YIELD_SECONDS,
                        )
                finally:
                    self._release_active_execution()
                if result is None:
                    break
                did_work = result.get("did_work", False) if isinstance(result, dict) else False
                actual_sleep = max(0.001, sleep_interval * 0.1) if did_work else max(0.05, sleep_interval)
            except Exception as exc:
                with self._lock:
                    self._brain_last_error = str(exc)
                    self._record_brain_event_locked({
                        "type": "error",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "message": str(exc),
                    })
                    self._request_brain_stop_locked(reason="error")
                break
            time.sleep(actual_sleep)

    def _run_brain_tick_once(
        self,
        *,
        stop_event: Event | None,
        sub_batch_size: int,
        yield_seconds: float,
    ) -> dict[str, Any] | None:
        tick_started = time.perf_counter()
        with self._lock:
            if stop_event is not None and stop_event.is_set():
                return None
            if not self._brain_source_runtimes:
                return self._brain_tick_idle_locked(tick_started)
            self._brain_stream_epoch += 1
            runtimes = list(self._brain_source_runtimes)
            src_index = self._brain_source_index
            tick_tokens = int(self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS))
            repeat = bool(self._brain_config.get("repeat_sources", True))
            ingestion = self._brain_config.get("ingestion") or {}
            queue_target_tokens = int(
                tick_tokens
                if not bool(ingestion.get("enabled", True))
                else ingestion.get("queue_target_tokens", tick_tokens)
            )
            encoder_ref = self._encoder
            window_size = self._trainer.config.window_size

        if len(runtimes) <= 1:
            ordered_indices = [0]
        else:
            ordered_indices: list[int]
            with self._lock:
                ordered_indices, _background_focus_terms, _background_focus_pressure = self._ordered_brain_runtime_indices_locked(
                    start_index=src_index,
                )

        chunk, collect_meta = self._collect_chunk_unlocked(
            runtimes,
            ordered_indices,
            tick_tokens,
            queue_target_tokens,
            repeat,
            encoder_ref,
            window_size,
            stop_event,
        )

        if stop_event is not None and stop_event.is_set():
            return None

        if chunk is None:
            with self._lock:
                return self._brain_tick_idle_locked(tick_started, source_meta=collect_meta)

        with self._lock:
            self._commit_collected_runtime_locked(collect_meta)
            if collect_meta is not None:
                self._update_brain_runtime_cache_locked(collect_meta["runtime"], served_examples=chunk)

        background_memory_metadata = None if collect_meta is None else self._brain_source_memory_metadata(cast(_BrainSourceRuntime, collect_meta["runtime"]))
        total_trained, last_metrics, evidence_windows = self._train_chunk_in_sub_batches(
            chunk,
            stop_event=stop_event,
            sub_batch_size=sub_batch_size,
            yield_seconds=yield_seconds,
            memory_metadata=background_memory_metadata,
        )
        source_info = {
            "runtime": collect_meta["runtime"],
            "idx": collect_meta["idx"],
            "source_count": collect_meta["source_count"],
        } if collect_meta else None
        with self._lock:
            return self._finalize_tick_locked(
                tick_started,
                source_info,
                total_trained,
                last_metrics,
                evidence_windows,
            )

    def _commit_collected_runtime_locked(self, collect_meta: dict[str, Any] | None) -> None:
        if collect_meta is None:
            return
        runtime = collect_meta["runtime"]
        runtime.cycles_completed = collect_meta["cycles"]
        runtime.exhausted = collect_meta["exhausted"]
        if collect_meta.get("new_stream") is not None:
            runtime.stream = collect_meta["new_stream"]
        runtime.last_buffer_tokens_served = int(collect_meta.get("served_tokens", 0) or 0)
        if bool(collect_meta.get("queue_hit", False)):
            runtime.queue_hits += 1
        prefetch_tokens = int(collect_meta.get("prefetch_tokens", 0) or 0)
        if prefetch_tokens > 0:
            runtime.prefetch_events += 1
            runtime.prefetched_tokens += prefetch_tokens
            runtime.last_prefetch_token_count = prefetch_tokens
            runtime.last_prefetch_at = collect_meta.get("prefetch_at")
            runtime.last_prefetch_duration_ms = collect_meta.get("prefetch_duration_ms")
        prefetch_error = collect_meta.get("prefetch_error")
        runtime.last_prefetch_error = None if prefetch_error in (None, "") else str(prefetch_error)
        self._update_brain_runtime_cache_locked(runtime)
        self._maybe_mark_ingestion_warm_locked(trigger=str(collect_meta.get("warm_trigger", "tick") or "tick"))

    def _train_chunk_in_sub_batches(
        self,
        chunk: list[tuple[str, "torch.Tensor"]],
        *,
        stop_event: Event | None,
        sub_batch_size: int,
        yield_seconds: float,
        memory_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[int, Any, list[str]]:
        total_trained = 0
        last_metrics = None
        batch_size = max(1, int(sub_batch_size))
        pause_seconds = max(0.0, float(yield_seconds))
        evidence_windows: deque[str] = deque(maxlen=128)
        for i in range(0, len(chunk), batch_size):
            if stop_event is not None and stop_event.is_set():
                break
            sub = chunk[i : i + batch_size]
            with self._lock:
                for raw_window, pattern in sub:
                    last_metrics = self._trainer.train_step(
                        pattern,
                        raw_window=raw_window,
                        memory_metadata=memory_metadata,
                    )
                    raw_text = str(raw_window)
                    if raw_text:
                        evidence_windows.append(raw_text)
                if sub:
                    self._observe_runtime_concepts_locked(raw_window=sub[-1][0], metrics=last_metrics)
                total_trained += len(sub)
                self._mark_mutated()
            if pause_seconds > 0.0:
                time.sleep(pause_seconds)
        return total_trained, last_metrics, list(evidence_windows)

    def _prefetch_runtime_queue_unlocked(
        self,
        runtime: _BrainSourceRuntime,
        target_tokens: int,
        repeat: bool,
        encoder_ref: Any,
        window_size: int,
        stop_event: Event | None,
        *,
        warm_trigger: str,
        deadline_perf: float | None = None,
    ) -> dict[str, Any] | None:
        cycles = runtime.cycles_completed
        exhausted = runtime.exhausted
        new_stream = None
        prefetch_tokens = 0
        prefetch_duration_ms: float | None = None
        prefetch_at: str | None = None
        prefetch_error: str | None = None
        budget_exhausted = False
        if len(runtime.buffered_patterns) < target_tokens and not exhausted:
            started = time.perf_counter()
            try:
                while len(runtime.buffered_patterns) < target_tokens:
                    if stop_event is not None and stop_event.is_set():
                        return None
                    wait_timeout = None
                    if deadline_perf is not None:
                        remaining = deadline_perf - time.perf_counter()
                        if remaining <= 0.0:
                            budget_exhausted = True
                            break
                        wait_timeout = remaining
                    try:
                        runtime.buffered_patterns.append(self._next_stream_item(runtime.stream, timeout=wait_timeout))
                        prefetch_tokens += 1
                    except TimeoutError:
                        budget_exhausted = True
                        break
                    except StopIteration:
                        if repeat:
                            cycles += 1
                            rebuilt = self._build_source_stream_from_spec(
                                runtime.spec, encoder_ref, window_size,
                            )
                            runtime.stream = rebuilt
                            new_stream = rebuilt
                            exhausted = False
                            try:
                                runtime.buffered_patterns.append(self._next_stream_item(runtime.stream, timeout=wait_timeout))
                                prefetch_tokens += 1
                            except TimeoutError:
                                budget_exhausted = True
                                break
                            except StopIteration:
                                exhausted = True
                                break
                        else:
                            exhausted = True
                            break
                    if deadline_perf is not None and time.perf_counter() >= deadline_perf:
                        budget_exhausted = True
                        break
            except Exception as exc:
                if stop_event is not None and stop_event.is_set():
                    return None
                prefetch_error = str(exc)
            if prefetch_tokens > 0 or prefetch_error is not None:
                prefetch_duration_ms = float((time.perf_counter() - started) * 1000.0)
                prefetch_at = datetime.now(timezone.utc).isoformat()
        return {
            "runtime": runtime,
            "cycles": cycles,
            "exhausted": exhausted,
            "new_stream": new_stream,
            "prefetch_tokens": int(prefetch_tokens),
            "prefetch_duration_ms": prefetch_duration_ms,
            "prefetch_at": prefetch_at,
            "prefetch_error": prefetch_error,
            "budget_exhausted": bool(budget_exhausted),
            "warm_trigger": warm_trigger,
        }

    def _prefetch_source_queues_unlocked(
        self,
        runtimes: Sequence[_BrainSourceRuntime],
        target_tokens: int,
        repeat: bool,
        encoder_ref: Any,
        window_size: int,
        stop_event: Event | None,
        *,
        warm_trigger: str,
        deadline_perf: float | None = None,
    ) -> list[dict[str, Any]]:
        prefetched: list[dict[str, Any]] = []
        for runtime in runtimes:
            if stop_event is not None and stop_event.is_set():
                break
            meta = self._prefetch_runtime_queue_unlocked(
                runtime,
                target_tokens,
                repeat,
                encoder_ref,
                window_size,
                stop_event,
                warm_trigger=warm_trigger,
                deadline_perf=deadline_perf,
            )
            if meta is not None:
                prefetched.append(meta)
        return prefetched

    def _collect_chunk_unlocked(
        self,
        runtimes: list,
        ordered_indices: Sequence[int],
        tick_tokens: int,
        queue_target_tokens: int,
        repeat: bool,
        encoder_ref: Any,
        window_size: int,
        stop_event: Event | None,
    ) -> tuple[list[tuple[str, "torch.Tensor"]] | None, dict[str, Any] | None]:
        """Collect tokens from source queues WITHOUT holding self._lock.

        Remote I/O happens while filling the per-source warm queue. Deliberation
        then consumes from the in-memory queue so later ticks are less exposed to
        remote startup or transient stalls.
        """
        source_count = len(runtimes)
        target_tokens = max(int(tick_tokens), int(queue_target_tokens))
        last_meta: dict[str, Any] | None = None
        if not ordered_indices:
            ordered_indices = list(range(source_count))
        for rank, idx in enumerate(list(ordered_indices)[:source_count]):
            if stop_event is not None and stop_event.is_set():
                return None, None
            runtime = runtimes[idx]
            buffer_before = len(runtime.buffered_patterns)
            fill_target = target_tokens if buffer_before < tick_tokens else buffer_before
            deadline_perf = None
            if buffer_before < tick_tokens and self._source_spec_uses_live_remote(runtime.spec):
                deadline_perf = time.perf_counter() + float(DEFAULT_REMOTE_ACTIVE_FETCH_WAIT_SECONDS)
            meta = self._prefetch_runtime_queue_unlocked(
                runtime,
                fill_target,
                repeat,
                encoder_ref,
                window_size,
                stop_event,
                warm_trigger="tick",
                deadline_perf=deadline_perf,
            )
            if meta is None:
                return None, None
            meta.update({"idx": idx, "source_count": source_count, "selection_rank": int(rank)})
            last_meta = meta
            if not runtime.buffered_patterns:
                continue

            served_tokens = min(int(tick_tokens), len(runtime.buffered_patterns))
            queue_hit = buffer_before >= tick_tokens and int(meta.get("prefetch_tokens", 0) or 0) == 0
            chunk = [runtime.buffered_patterns.popleft() for _ in range(served_tokens)]
            if not chunk:
                continue
            meta.update(
                {
                    "served_tokens": int(served_tokens),
                    "queue_hit": bool(queue_hit),
                }
            )
            return chunk, meta
        return None, last_meta

    @staticmethod
    def _build_source_stream_from_spec(
        spec: dict[str, Any],
        encoder: Any,
        window_size: int,
    ) -> Iterator[tuple[str, "torch.Tensor"]]:
        """Build a pattern stream without needing self._lock."""
        source_type = str(spec.get("source_type", "auto"))
        loader = StreamingCorpusLoader(
            source=str(spec.get("source", "")),
            source_type=source_type,
            text_field=str(spec.get("text_field", "text")),
            hf_config=spec.get("hf_config"),
        )
        stream = labeled_pattern_stream(
            loader.char_stream(),
            encoder,
            window_size,
            learn_chunking=True,
        )
        return cast(Iterator[tuple[str, torch.Tensor]], HECSNServiceManager._wrap_remote_stream(spec, stream, is_sensory=False))

    @staticmethod
    def _source_text_overlap(left: str, right: str) -> float:
        left_words = {word for word in re.findall(r"[a-zA-Z][a-zA-Z'-]+", left.lower()) if len(word) >= 4}
        right_words = {word for word in re.findall(r"[a-zA-Z][a-zA-Z'-]+", right.lower()) if len(word) >= 4}
        if not left_words or not right_words:
            return 0.0
        return len(left_words & right_words) / max(1.0, min(float(len(left_words)), float(len(right_words))))

    @classmethod
    def _grounded_source_sentences(
        cls,
        raw_windows: Sequence[str],
        *,
        max_sentences: int = 3,
    ) -> list[str]:
        raw_text_windows = [str(raw) for raw in raw_windows if str(raw)]
        if not raw_text_windows:
            return []

        reconstructed_raw = raw_text_windows[0]
        for window in raw_text_windows[1:]:
            max_overlap = min(len(reconstructed_raw), len(window))
            overlap = 0
            for size in range(max_overlap, 0, -1):
                if reconstructed_raw.endswith(window[:size]):
                    overlap = size
                    break
            reconstructed_raw += window[overlap:]
        reconstructed = " ".join(reconstructed_raw.split()).strip()

        normalized_windows = [" ".join(window.split()).strip() for window in raw_text_windows if " ".join(window.split()).strip()]
        selected: list[str] = []
        candidate_windows = [reconstructed, *list(reversed(normalized_windows[-24:]))]
        for window in candidate_windows:
            fragments = [fragment.strip(" ,;:") for fragment in re.split(r"(?<=[.!?])\s+", window) if fragment.strip()]
            if not fragments:
                fragments = [window]
            for fragment in fragments:
                cleaned = " ".join(fragment.split()).strip()
                words = re.findall(r"[A-Za-z][A-Za-z'-]+", cleaned)
                if len(words) < 3 or len(cleaned) < 24:
                    continue
                if any(cls._source_text_overlap(cleaned, existing) >= 0.82 for existing in selected):
                    continue
                selected.append(cleaned)
                if len(selected) >= max_sentences:
                    return selected[:max_sentences]
        if selected:
            return selected[:max_sentences]
        return [reconstructed[:320]]

    @staticmethod
    def _dedupe_grounded_topics(
        topics: Sequence[str],
        *,
        limit: int = 6,
    ) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for topic in topics:
            cleaned = " ".join(str(topic).split()).strip()
            lowered = cleaned.lower()
            if not cleaned or lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(cleaned)
            if len(deduped) >= max(1, int(limit)):
                break
        return deduped

    @staticmethod
    def _grounded_observation_metadata(
        *,
        observation_kind: str,
        source_name: str,
        source_type: str,
        salience: float,
        grounding_signal: float,
        evidence_unit_count: int,
        modality: str,
        focus_terms: Sequence[str] = (),
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "grounded": True,
            "observation_kind": str(observation_kind).strip().lower(),
            "source_name": str(source_name).strip(),
            "source_type": str(source_type).strip(),
            "salience": float(max(0.0, min(1.0, salience))),
            "grounding_signal": float(max(0.0, min(1.0, grounding_signal))),
            "evidence_unit_count": int(max(1, evidence_unit_count)),
            "modality": str(modality).strip().lower() or "text",
            "focus_terms": [
                " ".join(str(term).split()).strip()
                for term in list(focus_terms)[:6]
                if " ".join(str(term).split()).strip()
            ],
        }
        if extra:
            metadata.update({str(key): deepcopy(value) for key, value in dict(extra).items()})
        return metadata

    def _inject_source_observation_locked(
        self,
        *,
        runtime: _BrainSourceRuntime,
        evidence_windows: Sequence[str],
        total_trained: int,
        last_metrics: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if self._thought_loop is None:
            return {"content": "", "topics": [], "salience": 0.0}
        sentences = self._grounded_source_sentences(evidence_windows)
        excerpt = " ".join(sentences).strip()
        if not excerpt:
            return {"content": "", "topics": [], "salience": 0.0}

        concept_snapshot = self._concept_store.snapshot(limit=5)
        recent_concepts = [
            str(concept.get("label", "")).strip()
            for concept in concept_snapshot.get("top_concepts", [])[:5]
            if isinstance(concept, dict) and str(concept.get("label", "")).strip()
        ]
        topics: list[str] = []
        topics.extend(salient_query_terms(excerpt)[:6])
        topics.extend(recent_concepts[:3])
        deduped_topics = self._dedupe_grounded_topics(topics, limit=6)

        pred_error = 0.0
        surprise = 0.0
        if isinstance(last_metrics, dict):
            pred_error = max(0.0, min(1.0, float(last_metrics.get("pred_error", 0.0) or 0.0)))
            surprise = max(0.0, min(1.0, float(last_metrics.get("surprise", 0.0) or 0.0)))
        salience = max(
            0.35,
            min(
                0.95,
                0.55
                + 0.20 * pred_error
                + 0.10 * surprise
                + 0.04 * min(1.0, float(len(sentences)) / 2.0),
            ),
        )
        grounding_signal = max(
            0.35,
            min(
                1.0,
                0.52
                + 0.28 * pred_error
                + 0.14 * surprise
                + 0.06 * min(1.0, float(len(sentences)) / 2.0),
            ),
        )
        metadata = self._grounded_observation_metadata(
            observation_kind="source",
            source_name=runtime.name,
            source_type=runtime.source_type,
            salience=salience,
            grounding_signal=grounding_signal,
            evidence_unit_count=int(len(evidence_windows)),
            modality="text",
            focus_terms=deduped_topics[:4],
            extra={
                "evidence_window_count": int(len(evidence_windows)),
            },
        )
        self._thought_loop.inject_observation(
            content=excerpt,
            topics=deduped_topics,
            salience=salience,
            metadata=metadata,
        )
        return {
            "observation_kind": "source",
            "source_name": runtime.name,
            "source_type": runtime.source_type,
            "content": excerpt,
            "topics": deduped_topics,
            "salience": salience,
            "grounding_signal": grounding_signal,
            "evidence_unit_count": int(len(evidence_windows)),
            "evidence_window_count": int(len(evidence_windows)),
            "token_count": int(total_trained),
            "recent_concepts": recent_concepts[:3],
        }

    def _finalize_tick_locked(
        self,
        tick_started: float,
        source_info: dict[str, Any] | None,
        total_trained: int,
        last_metrics: Any,
        evidence_windows: Sequence[str],
    ) -> dict[str, Any]:
        """Update counters after training, run multimodal + autonomy. Under lock."""
        token_count_before = int(self._trainer.token_count) - total_trained
        token_count_after = int(self._trainer.token_count)

        # Update source runtime counters
        source_summary: dict[str, Any]
        if source_info is not None and total_trained > 0:
            runtime = source_info["runtime"]
            idx = source_info["idx"]
            source_count = source_info["source_count"]
            runtime.tokens_processed += total_trained
            runtime.tick_visits += 1
            runtime.last_tokens_trained = int(total_trained)
            runtime.last_activity_at = datetime.now(timezone.utc).isoformat()
            self._brain_background_tokens += total_trained
            self._brain_tick_count += 1
            self._brain_source_index = (idx + 1) % source_count
            self._mark_mutated()
            source_summary = {
                "did_work": True,
                "source_name": runtime.name,
                "source_type": runtime.source_type,
                "source_index": int(idx),
                "tokens_trained": int(total_trained),
                "cycles_completed": int(runtime.cycles_completed),
                "exhausted": bool(runtime.exhausted),
                "buffered_tokens_remaining": int(len(runtime.buffered_patterns)),
                "prefetch_events": int(runtime.prefetch_events),
                "queue_hits": int(runtime.queue_hits),
                "last_metrics": last_metrics,
            }
        else:
            source_summary = {"did_work": False, "reason": "no_tokens"}

        autonomy_summary = self._run_brain_autonomy_locked()
        cortex_work = bool(source_summary.get("did_work")) or autonomy_summary is not None

        if cortex_work and self._thought_loop is not None:
            try:
                surprise = self._trainer.model.surprise
                self._thought_loop.inject_surprise(
                    dopamine=float(surprise.dopamine),
                    serotonin=float(surprise.serotonin),
                    norepinephrine=float(surprise.norepinephrine),
                    acetylcholine=float(surprise.acetylcholine),
                )
            except Exception:
                pass

        if source_info is not None and total_trained > 0:
            try:
                source_runtime = cast(_BrainSourceRuntime, source_info["runtime"])
                grounded_observation = self._inject_source_observation_locked(
                    runtime=source_runtime,
                    evidence_windows=evidence_windows,
                    total_trained=total_trained,
                    last_metrics=cast(dict[str, Any] | None, last_metrics),
                )
                if grounded_observation.get("content"):
                    source_summary["grounded_observation"] = grounded_observation
                self._update_background_source_utility_locked(
                    runtime=source_runtime,
                    grounded_observation=cast(Mapping[str, Any] | None, grounded_observation),
                    total_trained=total_trained,
                )
            except Exception:
                pass

        sensory_summary = self._run_real_sensory_episode_locked()
        token_count_after = int(self._trainer.token_count)
        multimodal_summary = self._multimodal_runtime_summary_locked() if sensory_summary is not None else None
        did_work = bool(source_summary.get("did_work")) or autonomy_summary is not None or sensory_summary is not None

        completed_at = datetime.now(timezone.utc).isoformat()
        token_delta = int(token_count_after - token_count_before)
        summary = {
            "type": "tick",
            "did_work": did_work,
            "timestamp": completed_at,
            "source": source_summary,
            "multimodal": multimodal_summary,
            "autonomy": autonomy_summary,
            "tick_duration_ms": float((time.perf_counter() - tick_started) * 1000.0),
            "token_delta": int(token_delta),
        }
        self._brain_last_tick_completed_at = completed_at
        self._brain_last_tick_duration_ms = float(summary["tick_duration_ms"])
        self._brain_last_tick_token_delta = int(token_delta)
        if did_work:
            self._brain_last_work_at = completed_at
        self._record_brain_event_locked(summary)
        return summary

    def _brain_tick_idle_locked(self, tick_started: float, source_meta: dict[str, Any] | None = None) -> dict[str, Any]:
        """Handle a tick where no source tokens were available."""
        if not self._brain_config.get("source_bank"):
            summary = {
                "type": "tick",
                "did_work": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": "unconfigured",
            }
            self._brain_last_tick_completed_at = str(summary["timestamp"])
            self._brain_last_tick_duration_ms = float((time.perf_counter() - tick_started) * 1000.0)
            self._brain_last_tick_token_delta = 0
            self._record_brain_event_locked(summary)
            return summary

        source_summary: dict[str, Any] = {"did_work": False, "reason": "sources_exhausted"}
        if source_meta is not None:
            runtime = source_meta.get("runtime")
            if runtime is not None:
                source_summary.update(
                    {
                        "source_name": getattr(runtime, "name", "source"),
                        "source_type": getattr(runtime, "source_type", "auto"),
                        "source_index": int(source_meta.get("idx", 0) or 0),
                    }
                )
            if bool(source_meta.get("budget_exhausted", False)):
                source_summary["reason"] = "warming_remote_source"
                self._start_remote_warm_promotion_locked(trigger="tick")

        autonomy_summary = self._run_brain_autonomy_locked()
        sensory_summary = self._run_real_sensory_episode_locked()
        multimodal_summary = self._multimodal_runtime_summary_locked() if sensory_summary is not None else None
        did_work = autonomy_summary is not None or sensory_summary is not None
        completed_at = datetime.now(timezone.utc).isoformat()
        summary = {
            "type": "tick",
            "did_work": did_work,
            "timestamp": completed_at,
            "source": source_summary,
            "multimodal": multimodal_summary,
            "autonomy": autonomy_summary,
            "tick_duration_ms": float((time.perf_counter() - tick_started) * 1000.0),
            "token_delta": int(
                (0 if autonomy_summary is None else int(autonomy_summary.get("tokens_trained_total", 0) or 0))
                + (0 if sensory_summary is None else int(sensory_summary.get("steps_trained", 0) or 0))
            ),
        }
        self._brain_last_tick_completed_at = completed_at
        self._brain_last_tick_duration_ms = float(summary["tick_duration_ms"])
        self._brain_last_tick_token_delta = int(summary["token_delta"])
        if did_work:
            self._brain_last_work_at = completed_at
        self._record_brain_event_locked(summary)
        return summary

    def _run_brain_autonomy_locked(self) -> dict[str, Any] | None:
        autonomy = self._brain_config.get("autonomy")
        if not autonomy or not bool(autonomy.get("enabled", False)):
            return None
        token_delta = int(self._trainer.token_count) - int(self._brain_last_acquisition_token_count)
        focus_plan = self._autonomy_focus_plan_locked()
        adaptive_learning = self._adaptive_autonomy_settings_locked(autonomy, focus_plan)
        trigger_interval = int(
            adaptive_learning.get("effective_trigger_interval_tokens", autonomy.get("trigger_interval_tokens", DEFAULT_AUTONOMY_TRIGGER_INTERVAL_TOKENS))
        )

        # Curiosity-based trigger: allow early acquisition when gap score exceeds threshold
        curiosity_gap_threshold = float(autonomy.get("curiosity_gap_threshold", 0.0))
        curiosity_cooldown = int(autonomy.get("curiosity_cooldown_tokens", max(1, trigger_interval // 2)))
        curiosity_triggered = False
        trigger_reason = "interval"

        if curiosity_gap_threshold > 0.0 and token_delta >= curiosity_cooldown:
            abstraction = getattr(self._trainer.model, "abstraction_layer", None)
            if abstraction is not None:
                gaps = abstraction.curiosity_gaps(top_n=1)
                max_gap = float(gaps[0]["gap_score"]) if gaps else 0.0
                if max_gap >= curiosity_gap_threshold:
                    curiosity_triggered = True
                    trigger_reason = "curiosity_gap"

        if not curiosity_triggered and token_delta < trigger_interval:
            return None
        if self._brain_skip_next_autonomy_for_grounded_query:
            self._brain_skip_next_autonomy_for_grounded_query = False
            self._brain_last_acquisition_summary = None
            return None
        candidate_specs = self._autonomy_candidate_specs_locked(
            candidate_bank=list(autonomy.get("candidate_bank", [])),
            focus_plan=focus_plan,
        )
        shortlist_size, shortlist_gap_weight, shortlist_affinity_weight = self._autonomy_shortlist_settings_locked(
            candidate_bank=candidate_specs,
            config=autonomy,
            focus_plan=focus_plan,
        )
        curriculum_before = deepcopy(autonomy.get("provider_curriculum"))
        result = run_live_acquisition(
            trainer=self._trainer,
            encoder=self._encoder,
            candidate_bank_specs=candidate_specs,
            candidate_train_tokens=int(autonomy.get("candidate_train_tokens", 768)),
            probe_tokens=int(autonomy.get("probe_tokens", 96)),
            acquisition_tokens=int(adaptive_learning.get("effective_acquisition_tokens", autonomy.get("acquisition_tokens", 512))),
            acquisition_slots=int(adaptive_learning.get("effective_acquisition_slots", autonomy.get("acquisition_slots", 1))),
            gap_exploration_bonus=float(autonomy.get("gap_exploration_bonus", 0.03)),
            gap_ambiguity_weight=float(autonomy.get("gap_ambiguity_weight", 0.4)),
            gap_switch_weight=float(autonomy.get("gap_switch_weight", 0.2)),
            gap_margin_reference=float(autonomy.get("gap_margin_reference", 0.12)),
            coverage_balance_penalty=float(autonomy.get("coverage_balance_penalty", 0.2)),
            gap_focus_margin=float(autonomy.get("gap_focus_margin", 0.05)),
            policy_name=str(autonomy.get("policy", "active")),
            scout_commit_tokens=int(autonomy.get("scout_commit_tokens", 0)),
            scout_top_k=int(autonomy.get("scout_top_k", 1)),
            semantic_shortlist_size=shortlist_size,
            semantic_shortlist_gap_weight=shortlist_gap_weight,
            semantic_shortlist_affinity_weight=shortlist_affinity_weight,
            semantic_plan=focus_plan,
            on_train_step=self._runtime_concept_callback_locked(),
        )
        self._update_provider_curriculum_locked(
            autonomy=autonomy,
            result=result,
            candidate_specs=candidate_specs,
            focus_plan=focus_plan,
        )
        tokens_trained_total = int(result.get("tokens_trained_total", 0) or 0)
        self._brain_last_acquisition_token_count = int(self._trainer.token_count)
        self._brain_autonomy_tokens += tokens_trained_total
        if curriculum_before != autonomy.get("provider_curriculum"):
            self._mark_mutated()
        if tokens_trained_total > 0:
            self._mark_mutated()
        summary = {
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "trigger_reason": trigger_reason,
            "policy": str(result.get("policy", autonomy.get("policy", "active"))),
            "tokens_trained_total": int(tokens_trained_total),
            "acquired_sources": list(result.get("acquired_sources", [])),
            "stopped_early": bool(result.get("stopped_early", False)),
            "final_mean_candidate_gap": result.get("final_mean_candidate_gap"),
            "final_max_candidate_gap": result.get("final_max_candidate_gap"),
            "stop_reason": result.get("stop_reason"),
            "focus_plan": deepcopy(result.get("semantic_plan")),
            "recent_query_gap_count": int(len(self._brain_recent_query_gaps)),
            "adaptive_learning": deepcopy(adaptive_learning),
            "provider_curriculum": deepcopy(self._provider_curriculum_snapshot_locked(autonomy, focus_plan)),
        }
        self._brain_last_acquisition_summary = summary
        return summary

    def _animation_snapshot_locked(self) -> dict[str, Any]:
        """Lightweight snapshot for UI animation: active column, spike counts, layer state."""
        model = self._trainer.model
        competitive = model.competitive
        n_columns = int(competitive.n_columns)
        winner = self._trainer.last_winner
        activations = competitive.thresholds.detach().cpu().tolist()
        spike_counts = competitive.spike_counts.detach().cpu().tolist() if hasattr(competitive, "spike_counts") else [0] * n_columns
        cross_modal_state = None
        if model.cross_modal is not None:
            cross_modal_state = {
                "visual_confidence": float(model.cross_modal.visual_confidence.mean().item()),
                "audio_confidence": float(model.cross_modal.audio_confidence.mean().item()),
            }
        context_tau = None
        if model.context_layer is not None and hasattr(model.context_layer, "log_tau"):
            context_tau = torch.exp(model.context_layer.log_tau).detach().cpu().tolist()

        # Binding layer summary
        binding_state = None
        binding = getattr(model, "binding", None)
        if binding is not None:
            binding_state = {
                "n_binding_neurons": int(binding.n_binding),
                "mean_weight": float(binding.W.detach().abs().mean().item()),
            }

        # Abstraction layer summary
        abstraction_state = None
        abstraction = getattr(model, "abstraction", None)
        if abstraction is not None:
            abstraction_state = {
                "curiosity": float(abstraction.curiosity.item()) if hasattr(abstraction, "curiosity") else 0.0,
                "n_abstract": int(abstraction.n_abstract) if hasattr(abstraction, "n_abstract") else 0,
            }

        # STDP layer summary
        stdp_state = None
        stdp = getattr(model, "stdp", None)
        if stdp is not None:
            stdp_state = {
                "mean_weight": float(stdp.weights.detach().abs().mean().item()) if hasattr(stdp, "weights") else 0.0,
            }

        return {
            "n_columns": n_columns,
            "winner_id": None if winner is None else int(winner),
            "activations": activations,
            "spike_counts": spike_counts,
            "cross_modal": cross_modal_state,
            "context_tau": context_tau,
            "binding": binding_state,
            "abstraction": abstraction_state,
            "stdp": stdp_state,
            "memory_fill": float(model.memory_store.summary_stats().get("fill_fraction", 0.0)),
        }

    def _brain_runtime_snapshot_locked(self) -> dict[str, Any]:
        self._remerge_converged_delayed_consequence_families_locked()
        self._split_divergent_delayed_consequence_families_locked()
        self._compact_delayed_consequence_records_locked()
        self._cool_delayed_consequence_records_locked()
        autonomy = self._brain_config.get("autonomy")
        exhausted_source_count = sum(1 for runtime in self._brain_source_runtimes if runtime.exhausted)
        background_focus_plan: Mapping[str, Any] | None = None
        background_focus_terms: list[str] = []
        background_focus_pressure = 0.0
        next_source_name = None
        background_selection_order: list[str] = []
        if len(self._brain_source_runtimes) == 1:
            next_source_name = self._brain_source_runtimes[0].name
            background_selection_order = [self._brain_source_runtimes[0].name]
        elif len(self._brain_source_runtimes) > 1:
            background_focus_plan = self._autonomy_focus_plan_locked()
            background_focus_terms = self._background_focus_terms_locked(focus_plan=background_focus_plan)
            background_focus_pressure, _background_focus_pressure_details = self._autonomy_focus_pressure_locked(background_focus_plan)
            ordered_indices, _focus_terms, _focus_pressure = self._ordered_brain_runtime_indices_locked(
                start_index=self._brain_source_index,
            )
            if ordered_indices:
                next_source_name = self._brain_source_runtimes[ordered_indices[0]].name
                background_selection_order = [
                    self._brain_source_runtimes[idx].name
                    for idx in ordered_indices
                    if 0 <= idx < len(self._brain_source_runtimes)
                ]
        autonomy_tokens_until_trigger = None
        autonomy_trigger_ready = None
        autonomy_candidate_names = None
        autonomy_focus_plan = background_focus_plan
        autonomy_provider_curriculum = None
        autonomy_adaptive_learning = None
        sensory = self._brain_config.get("sensory")
        sensory_tokens_until_trigger = None
        sensory_trigger_ready = None
        if autonomy is not None:
            if autonomy_focus_plan is None:
                autonomy_focus_plan = self._autonomy_focus_plan_locked()
            autonomy_provider_curriculum = self._provider_curriculum_snapshot_locked(autonomy, autonomy_focus_plan)
            autonomy_adaptive_learning = self._adaptive_autonomy_settings_locked(autonomy, autonomy_focus_plan)
            trigger_interval = int(
                autonomy_adaptive_learning.get(
                    "effective_trigger_interval_tokens",
                    autonomy.get("trigger_interval_tokens", DEFAULT_AUTONOMY_TRIGGER_INTERVAL_TOKENS),
                )
            )
            token_delta = int(self._trainer.token_count) - int(self._brain_last_acquisition_token_count)
            autonomy_tokens_until_trigger = int(max(0, trigger_interval - token_delta))
            autonomy_trigger_ready = bool(token_delta >= trigger_interval)
            autonomy_candidate_names = [
                str(item.get("name", "candidate"))
                for item in list(autonomy.get("candidate_bank", []))
            ]
        if sensory is not None:
            sensory_trigger_interval = int(sensory.get("episode_interval_tokens", 2048))
            sensory_token_delta = int(self._trainer.token_count) - int(self._last_real_sensory_episode_token_count)
            sensory_tokens_until_trigger = int(max(0, sensory_trigger_interval - sensory_token_delta))
            sensory_trigger_ready = bool(sensory_token_delta >= sensory_trigger_interval)
        if self._remote_warm_promotion_running and not self._remote_warm_promotion_text_needed_locked() and not self._remote_warm_promotion_sensory_needed_locked():
            self._record_remote_warm_promotion_completed_locked()
        thread_alive = self._brain_runtime_active_locked()
        total_text_learning_tokens = int(self._brain_background_tokens + self._brain_autonomy_tokens)
        autonomy_share_of_text_learning = float(
            0.0
            if total_text_learning_tokens <= 0
            else float(self._brain_autonomy_tokens) / float(total_text_learning_tokens)
        )
        background_share_of_text_learning = float(
            0.0
            if total_text_learning_tokens <= 0
            else max(0.0, 1.0 - autonomy_share_of_text_learning)
        )
        cortex_snapshot = self._thought_loop.snapshot() if self._thought_loop is not None else {"enabled": False}
        living_loop_snapshot = self._living_loop_snapshot_locked(cortex_snapshot=cortex_snapshot)
        return {
            "configured": bool(self._brain_config.get("source_bank")),
            "running": bool(thread_alive),
            "running_since": self._brain_running_since,
            "shutdown": {
                "stop_requested": self._brain_stop_requested_at is not None,
                "stop_requested_at": self._brain_stop_requested_at,
                "stop_reason": self._brain_stop_requested_reason,
                "stop_timed_out": bool(self._brain_stop_timed_out),
                "last_stop_duration_ms": self._brain_last_stop_duration_ms,
                "join_timeout_seconds": float(DEFAULT_BRAIN_STOP_TIMEOUT_SECONDS),
                "thread_alive": bool(thread_alive),
            },
            "environment": self._runtime_environment_summary(),
            "action_loop": self._action_loop_summary_locked(),
            "tick_tokens": int(self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS)),
            "sleep_interval_seconds": float(
                self._brain_config.get("sleep_interval_seconds", DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS)
            ),
            "repeat_sources": bool(self._brain_config.get("repeat_sources", True)),
            "source_count": int(len(self._brain_source_runtimes)),
            "exhausted_source_count": int(exhausted_source_count),
            "next_source_name": next_source_name,
            "background_tokens_processed": int(self._brain_background_tokens),
            "tick_count": int(self._brain_tick_count),
            "last_tick_completed_at": self._brain_last_tick_completed_at,
            "last_tick_duration_ms": self._brain_last_tick_duration_ms,
            "last_tick_token_delta": int(self._brain_last_tick_token_delta),
            "tokens_per_second": float(
                (self._brain_last_tick_token_delta / (self._brain_last_tick_duration_ms / 1000.0))
                if self._brain_last_tick_duration_ms and self._brain_last_tick_duration_ms > 0
                else 0.0
            ),
            "last_work_at": self._brain_last_work_at,
            "last_error": self._brain_last_error,
            "last_event": deepcopy(self._brain_last_event),
            "recent_events": [deepcopy(event) for event in list(self._brain_event_history)],
            "source_bank": deepcopy(self._brain_config.get("source_bank", [])),
            "ingestion": self._ingestion_runtime_summary_locked(),
            "background_source_routing": {
                "mode": "focus_aware_allocation",
                "utility_mode": "provenance_grounded_family_summary_lineage_reconvergent_divergence_split_trajectory_sensitive_compacted_age_sensitive_consequence_calibration",
                "evidence_provenance_credit": True,
                "delayed_consequence_tracking": self._delayed_consequence_summary_locked(limit=4),
                "focus_terms": list(background_focus_terms),
                "focus_pressure": float(background_focus_pressure),
                "selection_order": list(background_selection_order),
            },
            "text_learning_balance": {
                "background_tokens_processed": int(self._brain_background_tokens),
                "autonomy_tokens_processed": int(self._brain_autonomy_tokens),
                "total_text_learning_tokens": int(total_text_learning_tokens),
                "autonomy_share_of_text_learning": float(autonomy_share_of_text_learning),
                "background_share_of_text_learning": float(background_share_of_text_learning),
            },
            "source_progress": [
                {
                    "name": runtime.name,
                    "source_type": runtime.source_type,
                    "tokens_processed": int(runtime.tokens_processed),
                    "tick_visits": int(runtime.tick_visits),
                    "last_tokens_trained": int(runtime.last_tokens_trained),
                    "last_activity_at": runtime.last_activity_at,
                    "cycles_completed": int(runtime.cycles_completed),
                    "exhausted": bool(runtime.exhausted),
                    "buffered_tokens": int(len(runtime.buffered_patterns)),
                    "buffer_fill_fraction": float(
                        0.0
                        if int((self._brain_config.get("ingestion") or {}).get("queue_target_tokens", 0) or 0) <= 0
                        else float(len(runtime.buffered_patterns))
                        / float(int((self._brain_config.get("ingestion") or {}).get("queue_target_tokens", 0) or 1))
                    ),
                    "prefetch_events": int(runtime.prefetch_events),
                    "prefetched_tokens": int(runtime.prefetched_tokens),
                    "last_prefetch_token_count": int(runtime.last_prefetch_token_count),
                    "last_prefetch_at": runtime.last_prefetch_at,
                    "last_prefetch_duration_ms": runtime.last_prefetch_duration_ms,
                    "last_prefetch_error": runtime.last_prefetch_error,
                    "queue_hits": int(runtime.queue_hits),
                    "last_buffer_tokens_served": int(runtime.last_buffer_tokens_served),
                    "last_semantic_match": float(runtime.last_semantic_match),
                    "last_selection_score": float(runtime.last_selection_score),
                    "last_fairness_score": float(runtime.last_fairness_score),
                    "last_buffer_readiness": float(runtime.last_buffer_readiness),
                    "last_utility_score": float(runtime.last_utility_score),
                    "utility_ema": float(self._background_source_utility_entry_locked(runtime).get("utility_ema", 0.0)),
                    "semantic_alignment_ema": float(self._background_source_utility_entry_locked(runtime).get("semantic_alignment_ema", 0.0)),
                    "grounding_signal_ema": float(self._background_source_utility_entry_locked(runtime).get("grounding_signal_ema", 0.0)),
                    "focus_overlap_ema": float(self._background_source_utility_entry_locked(runtime).get("focus_overlap_ema", 0.0)),
                    "grounded_outcome_ema": float(self._background_source_utility_entry_locked(runtime).get("grounded_outcome_ema", 0.0)),
                    "grounded_family_summary_ema": float(self._background_source_utility_entry_locked(runtime).get("grounded_family_summary_ema", 0.0)),
                    "delayed_consequence_ema": float(self._background_source_utility_entry_locked(runtime).get("delayed_consequence_ema", 0.0)),
                    "contradiction_decay_ema": float(self._background_source_utility_entry_locked(runtime).get("contradiction_decay_ema", 0.0)),
                    "share_of_background_tokens": float(
                        0.0
                        if self._brain_background_tokens <= 0
                        else float(runtime.tokens_processed) / float(self._brain_background_tokens)
                    ),
                }
                for runtime in self._brain_source_runtimes
            ],
            "huggingface": self._huggingface_runtime_summary_locked(),
            "sensory": None
            if sensory is None
            else (
                lambda snapshot: {
                    **snapshot,
                    "tokens_until_trigger": sensory_tokens_until_trigger,
                    "trigger_ready": sensory_trigger_ready,
                }
            )(self._sensory_runtime_summary_locked(sensory)),
            "autonomy": None
            if autonomy is None
            else {
                "enabled": bool(autonomy.get("enabled", False)),
                "policy": str(autonomy.get("policy", "active")),
                "candidate_count": int(len(autonomy.get("candidate_bank", []))),
                "candidate_bank": deepcopy(list(autonomy.get("candidate_bank", []))),
                "candidate_names": autonomy_candidate_names,
                "trigger_interval_tokens": int(
                    autonomy.get("trigger_interval_tokens", DEFAULT_AUTONOMY_TRIGGER_INTERVAL_TOKENS)
                ),
                "tokens_processed": int(self._brain_autonomy_tokens),
                "share_of_text_learning_tokens": float(autonomy_share_of_text_learning),
                "tokens_until_trigger": autonomy_tokens_until_trigger,
                "trigger_ready": autonomy_trigger_ready,
                "recent_query_gaps": [deepcopy(item) for item in list(self._brain_recent_query_gaps)],
                "focus_plan": deepcopy(autonomy_focus_plan),
                "adaptive_learning": deepcopy(autonomy_adaptive_learning),
                "provider_curriculum": deepcopy(autonomy_provider_curriculum),
                "delayed_consequence_tracking": self._delayed_consequence_summary_locked(limit=4),
                "last_acquisition_token_count": int(self._brain_last_acquisition_token_count),
                "last_acquisition_summary": deepcopy(self._brain_last_acquisition_summary),
                "geometric_curiosity": deepcopy(self._geometric_curiosity.summary()),
            },
            "multimodal": self._multimodal_runtime_summary_locked(),
            "living_loop": living_loop_snapshot,
            "cortex": cortex_snapshot,
        }

    def _delayed_consequence_summary_locked(self, limit: int = 4) -> dict[str, Any]:
        records = list(self._delayed_consequence_records)
        current_token = int(self._trainer.token_count)
        credited_count = sum(1 for record in records if int(record.get("credit_events", 0) or 0) > 0)
        penalized_count = sum(1 for record in records if int(record.get("penalty_events", 0) or 0) > 0)
        forgiven_count = sum(1 for record in records if int(record.get("forgiveness_events", 0) or 0) > 0)
        aggregated_count = sum(1 for record in records if int(record.get("aggregate_count", 1) or 1) > 1)
        aggregate_occurrence_count = sum(max(1, int(record.get("aggregate_count", 1) or 1)) for record in records)
        trajectory_state_counts = Counter(self._delayed_consequence_trajectory_state(record) for record in records)
        recent_records: list[dict[str, Any]] = []
        for record in records[: max(1, int(limit))]:
            recent_records.append(
                {
                    "record_id": str(record.get("record_id", "")),
                    "origin": str(record.get("origin", "response_selected_evidence")),
                    "created_at": str(record.get("created_at", "")),
                    "query_text": str(record.get("query_text", "")),
                    "query_examples": self._delayed_consequence_query_examples(record),
                    "aggregate_count": int(record.get("aggregate_count", 1) or 1),
                    "aggregation_events": int(record.get("aggregation_events", 0) or 0),
                    "supportive_query_examples": self._delayed_consequence_branch_examples(record, field="supportive_query_examples"),
                    "adverse_query_examples": self._delayed_consequence_branch_examples(record, field="adverse_query_examples"),
                    "supportive_occurrence_count": int(record.get("supportive_occurrence_count", 0) or 0),
                    "adverse_occurrence_count": int(record.get("adverse_occurrence_count", 0) or 0),
                    "aggregate_support_multiplier": float(self._delayed_consequence_support_multiplier(record)),
                    "family_support_multiplier": float(self._delayed_consequence_family_support_multiplier(record, mode="credit")),
                    "trajectory_support_multiplier": float(
                        self._delayed_consequence_trajectory_support_multiplier(record, mode="credit")
                    ),
                    "trajectory_penalty_multiplier": float(
                        self._delayed_consequence_trajectory_support_multiplier(record, mode="penalty")
                    ),
                    "grounded_family_summary_score": float(self._grounded_family_summary_score(record)),
                    "source_names": sorted(dict(record.get("source_weights") or {}).keys()),
                    "providers": sorted(dict(record.get("provider_weights") or {}).keys()),
                    "baseline_query_score": float(record.get("baseline_query_score", 0.0) or 0.0),
                    "best_query_score": float(record.get("best_query_score", 0.0) or 0.0),
                    "baseline_grounded_fraction": float(record.get("baseline_grounded_fraction", 0.0) or 0.0),
                    "best_grounded_fraction": float(record.get("best_grounded_fraction", 0.0) or 0.0),
                    "credit_events": int(record.get("credit_events", 0) or 0),
                    "penalty_events": int(record.get("penalty_events", 0) or 0),
                    "forgiveness_events": int(record.get("forgiveness_events", 0) or 0),
                    "cooling_events": int(record.get("cooling_events", 0) or 0),
                    "trajectory_state": self._delayed_consequence_trajectory_state(record),
                    "trajectory_event_count": int(record.get("trajectory_event_count", 0) or 0),
                    "trajectory_credit_total": float(record.get("trajectory_credit_total", 0.0) or 0.0),
                    "trajectory_penalty_total": float(record.get("trajectory_penalty_total", 0.0) or 0.0),
                    "trajectory_forgiveness_total": float(record.get("trajectory_forgiveness_total", 0.0) or 0.0),
                    "trajectory_net_score": float(record.get("trajectory_net_score", 0.0) or 0.0),
                    "trajectory_signal_balance": float(self._delayed_consequence_trajectory_balance(record)),
                    "trajectory_recent_delta_ema": float(self._delayed_consequence_trajectory_recent_signal(record)),
                    "trajectory_peak_score": float(record.get("trajectory_peak_score", 0.0) or 0.0),
                    "trajectory_floor_score": float(record.get("trajectory_floor_score", 0.0) or 0.0),
                    "unresolved_penalty_balance": float(record.get("unresolved_penalty_balance", 0.0) or 0.0),
                    "resolved_improvement": float(record.get("resolved_improvement", 0.0) or 0.0),
                    "max_regression": float(record.get("max_regression", 0.0) or 0.0),
                    "max_contradiction_signal": float(record.get("max_contradiction_signal", 0.0) or 0.0),
                    "cumulative_cooling_delta": float(record.get("cumulative_cooling_delta", 0.0) or 0.0),
                    "created_token_count": int(record.get("created_token_count", 0) or 0),
                    "last_activity_token_count": int(record.get("last_activity_token_count", 0) or 0),
                    "last_cooling_token_count": int(record.get("last_cooling_token_count", 0) or 0),
                    "age_tokens": int(max(0, current_token - int(record.get("created_token_count", current_token)))),
                    "activity_age_tokens": int(max(0, current_token - int(record.get("last_activity_token_count", current_token)))),
                    "last_credit_score": float(record.get("last_credit_score", 0.0) or 0.0),
                    "last_penalty_score": float(record.get("last_penalty_score", 0.0) or 0.0),
                    "last_forgiveness_score": float(record.get("last_forgiveness_score", 0.0) or 0.0),
                    "last_penalty_reason": str(record.get("last_penalty_reason", "")),
                    "last_trajectory_event_type": str(record.get("last_trajectory_event_type", "")),
                    "last_trajectory_event_score": float(record.get("last_trajectory_event_score", 0.0) or 0.0),
                    "last_trajectory_event_at": str(record.get("last_trajectory_event_at", "")),
                    "split_generation": int(record.get("split_generation", 0) or 0),
                    "split_parent_record_id": str(record.get("split_parent_record_id", "")),
                    "split_group_id": str(record.get("split_group_id", "")),
                    "split_branch": str(record.get("split_branch", "")),
                    "remerge_events": int(record.get("remerge_events", 0) or 0),
                    "last_split_at": str(record.get("last_split_at", "")),
                    "last_remerged_at": str(record.get("last_remerged_at", "")),
                    "last_aggregated_at": str(record.get("last_aggregated_at", "")),
                    "last_cooled_at": str(record.get("last_cooled_at", "")),
                    "last_evaluated_at": str(record.get("last_evaluated_at", "")),
                    "last_evaluated_query_text": str(record.get("last_evaluated_query_text", "")),
                }
            )
        return {
            "enabled": True,
            "record_count": int(len(records)),
            "credited_record_count": int(credited_count),
            "penalized_record_count": int(penalized_count),
            "forgiven_record_count": int(forgiven_count),
            "aggregated_record_count": int(aggregated_count),
            "aggregate_occurrence_count": int(aggregate_occurrence_count),
            "trajectory_state_counts": {str(state): int(count) for state, count in dict(trajectory_state_counts).items()},
            "max_grounded_family_summary_score": float(
                max((self._grounded_family_summary_score(record) for record in records), default=0.0)
            ),
            "cooled_record_count_total": int(self._delayed_consequence_cooled_total),
            "retired_record_count_total": int(self._delayed_consequence_retired_total),
            "compacted_record_count_total": int(self._delayed_consequence_compacted_total),
            "split_record_count_total": int(self._delayed_consequence_split_total),
            "remerged_record_count_total": int(self._delayed_consequence_remerged_total),
            "pending_record_count": int(
                sum(
                    1
                    for record in records
                    if int(record.get("credit_events", 0) or 0) <= 0
                    and int(record.get("penalty_events", 0) or 0) <= 0
                    and int(record.get("forgiveness_events", 0) or 0) <= 0
                )
            ),
            "recent_records": recent_records,
        }

    def _brain_persisted_state_locked(self) -> dict[str, Any]:
        return {
            "source_bank": deepcopy(self._brain_config.get("source_bank", [])),
            "tick_tokens": int(self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS)),
            "sleep_interval_seconds": float(
                self._brain_config.get("sleep_interval_seconds", DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS)
            ),
            "repeat_sources": bool(self._brain_config.get("repeat_sources", True)),
            "autonomy": deepcopy(self._brain_config.get("autonomy")),
            "sensory": deepcopy(self._brain_config.get("sensory")),
            "ingestion": deepcopy(self._brain_config.get("ingestion")),
            "background_source_utility": deepcopy(self._brain_source_utility),
            "delayed_consequence_records": [deepcopy(item) for item in list(self._delayed_consequence_records)],
            "delayed_consequence_cooled_total": int(self._delayed_consequence_cooled_total),
            "delayed_consequence_retired_total": int(self._delayed_consequence_retired_total),
            "delayed_consequence_compacted_total": int(self._delayed_consequence_compacted_total),
            "delayed_consequence_split_total": int(self._delayed_consequence_split_total),
            "delayed_consequence_remerged_total": int(self._delayed_consequence_remerged_total),
            "recent_query_gaps": [deepcopy(item) for item in list(self._brain_recent_query_gaps)],
            "action_history": [deepcopy(item) for item in list(self._action_history)],
            "geometric_curiosity": self._geometric_curiosity.state_dict(),
        }

    @staticmethod
    def _normalize_background_source_utility_state(value: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(value, Mapping):
            return {}

        def _safe_int(raw_value: Any) -> int:
            try:
                return max(0, int(raw_value))
            except (TypeError, ValueError):
                return 0

        def _safe_float(raw_value: Any) -> float:
            try:
                return max(0.0, min(1.0, float(raw_value)))
            except (TypeError, ValueError):
                return 0.0

        normalized: dict[str, dict[str, Any]] = {}
        for raw_name, raw_entry in value.items():
            name = " ".join(str(raw_name).split()).strip()
            if not name or not isinstance(raw_entry, Mapping):
                continue
            normalized[name] = {
                "attempts": _safe_int(raw_entry.get("attempts", 0)),
                "selections": _safe_int(raw_entry.get("selections", 0)),
                "tokens_trained_total": _safe_int(raw_entry.get("tokens_trained_total", 0)),
                "utility_ema": _safe_float(raw_entry.get("utility_ema", 0.0)),
                "semantic_alignment_ema": _safe_float(raw_entry.get("semantic_alignment_ema", 0.0)),
                "grounding_signal_ema": _safe_float(raw_entry.get("grounding_signal_ema", 0.0)),
                "focus_overlap_ema": _safe_float(raw_entry.get("focus_overlap_ema", 0.0)),
                "grounded_outcome_ema": _safe_float(raw_entry.get("grounded_outcome_ema", 0.0)),
                "grounded_family_summary_ema": _safe_float(raw_entry.get("grounded_family_summary_ema", 0.0)),
                "delayed_consequence_ema": _safe_float(raw_entry.get("delayed_consequence_ema", 0.0)),
                "contradiction_decay_ema": _safe_float(raw_entry.get("contradiction_decay_ema", 0.0)),
                "last_selected_at": " ".join(str(raw_entry.get("last_selected_at", "")).split()).strip(),
            }
        return normalized

    def _background_source_utility_entry_locked(self, runtime: _BrainSourceRuntime) -> dict[str, Any]:
        name = str(runtime.name).strip()
        entry = self._brain_source_utility.setdefault(
            name,
            {
                "attempts": 0,
                "selections": 0,
                "tokens_trained_total": 0,
                "utility_ema": 0.0,
                "semantic_alignment_ema": 0.0,
                "grounding_signal_ema": 0.0,
                "focus_overlap_ema": 0.0,
                "grounded_outcome_ema": 0.0,
                "grounded_family_summary_ema": 0.0,
                "delayed_consequence_ema": 0.0,
                "contradiction_decay_ema": 0.0,
                "last_selected_at": "",
            },
        )
        entry.setdefault("grounded_family_summary_ema", 0.0)
        entry.setdefault("delayed_consequence_ema", 0.0)
        entry.setdefault("contradiction_decay_ema", 0.0)
        return entry

    def _normalize_delayed_consequence_record(self, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, Mapping):
            return None

        def _safe_float(raw_value: Any) -> float:
            try:
                return max(0.0, min(1.0, float(raw_value)))
            except (TypeError, ValueError):
                return 0.0

        def _safe_int(raw_value: Any) -> int:
            try:
                return max(0, int(raw_value))
            except (TypeError, ValueError):
                return 0

        def _safe_total(raw_value: Any) -> float:
            try:
                return max(0.0, float(raw_value))
            except (TypeError, ValueError):
                return 0.0

        def _safe_signed(raw_value: Any, *, limit: float = DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT) -> float:
            try:
                return max(-float(limit), min(float(limit), float(raw_value)))
            except (TypeError, ValueError):
                return 0.0

        current_token = int(self._trainer.token_count)
        query_text = self._normalize_action_text(item.get("query_text", ""))
        if not query_text:
            return None
        query_examples: list[str] = []
        seen_query_examples: set[str] = set()
        raw_query_examples = item.get("query_examples")
        query_example_values = [query_text]
        if isinstance(raw_query_examples, Sequence) and not isinstance(raw_query_examples, (str, bytes)):
            query_example_values.extend(list(raw_query_examples))
        for raw_value in query_example_values:
            text = self._normalize_action_text(raw_value)
            if not text:
                continue
            key = text.lower()
            if key in seen_query_examples:
                continue
            seen_query_examples.add(key)
            query_examples.append(text)
            if len(query_examples) >= DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT:
                break
        query_terms = [
            _canonical_provider_term(term)
            for term in list(
                item.get("query_terms")
                or [term for text in query_examples for term in self._consequence_query_terms(text)]
                or self._consequence_query_terms(query_text)
            )
            if _canonical_provider_term(term)
        ]

        def _weight_map(raw_value: Any, *, lowercase_keys: bool) -> dict[str, float]:
            weighted: dict[str, float] = {}
            if not isinstance(raw_value, Mapping):
                return weighted
            for raw_name, raw_weight in raw_value.items():
                name = " ".join(str(raw_name).split()).strip()
                if not name:
                    continue
                key = name.lower() if lowercase_keys else name
                weight = _safe_float(raw_weight)
                if weight <= 0.0:
                    continue
                weighted[key] = weight
            return weighted

        source_weights = _weight_map(item.get("source_weights"), lowercase_keys=False)
        provider_weights = _weight_map(item.get("provider_weights"), lowercase_keys=True)
        if not source_weights and not provider_weights:
            return None
        baseline_query_score = _safe_float(item.get("baseline_query_score", 0.0))
        best_query_score = max(baseline_query_score, _safe_float(item.get("best_query_score", baseline_query_score)))
        baseline_grounded_fraction = _safe_float(item.get("baseline_grounded_fraction", 0.0))
        best_grounded_fraction = max(
            baseline_grounded_fraction,
            _safe_float(item.get("best_grounded_fraction", baseline_grounded_fraction)),
        )
        credit_events = _safe_int(item.get("credit_events", 0))
        penalty_events = _safe_int(item.get("penalty_events", 0))
        forgiveness_events = _safe_int(item.get("forgiveness_events", 0))
        trajectory_credit_total = max(
            _safe_total(item.get("trajectory_credit_total", 0.0)),
            _safe_total(item.get("resolved_improvement", 0.0)) if credit_events > 0 else 0.0,
        )
        trajectory_penalty_total = max(
            _safe_total(item.get("trajectory_penalty_total", 0.0)),
            _safe_total(item.get("max_regression", 0.0)) if penalty_events > 0 else 0.0,
            _safe_total(item.get("unresolved_penalty_balance", 0.0)) if penalty_events > 0 else 0.0,
        )
        trajectory_forgiveness_total = max(
            _safe_total(item.get("trajectory_forgiveness_total", 0.0)),
            _safe_total(item.get("last_forgiveness_score", 0.0)) if forgiveness_events > 0 else 0.0,
        )
        trajectory_event_count = max(
            _safe_int(item.get("trajectory_event_count", 0)),
            credit_events + penalty_events + forgiveness_events,
        )
        raw_trajectory_net = item.get(
            "trajectory_net_score",
            trajectory_credit_total + trajectory_forgiveness_total - trajectory_penalty_total,
        )
        trajectory_net_score = _safe_signed(raw_trajectory_net)
        trajectory_peak_score = max(
            trajectory_net_score,
            _safe_signed(item.get("trajectory_peak_score", trajectory_net_score)),
        )
        trajectory_floor_score = min(
            trajectory_net_score,
            _safe_signed(item.get("trajectory_floor_score", trajectory_net_score)),
        )
        split_branch = self._normalize_action_text(item.get("split_branch", "")).lower()
        supportive_query_examples: list[str] = []
        seen_supportive_examples: set[str] = set()
        raw_supportive_examples = item.get("supportive_query_examples")
        if isinstance(raw_supportive_examples, Sequence) and not isinstance(raw_supportive_examples, (str, bytes)):
            for raw_value in list(raw_supportive_examples):
                text = self._normalize_action_text(raw_value)
                if not text:
                    continue
                key = text.lower()
                if key in seen_supportive_examples:
                    continue
                seen_supportive_examples.add(key)
                supportive_query_examples.append(text)
                if len(supportive_query_examples) >= DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT:
                    break
        if split_branch == "supportive" and not supportive_query_examples:
            supportive_query_examples = list(query_examples)[:DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT]
        adverse_query_examples: list[str] = []
        seen_adverse_examples: set[str] = set()
        raw_adverse_examples = item.get("adverse_query_examples")
        if isinstance(raw_adverse_examples, Sequence) and not isinstance(raw_adverse_examples, (str, bytes)):
            for raw_value in list(raw_adverse_examples):
                text = self._normalize_action_text(raw_value)
                if not text:
                    continue
                key = text.lower()
                if key in seen_adverse_examples:
                    continue
                seen_adverse_examples.add(key)
                adverse_query_examples.append(text)
                if len(adverse_query_examples) >= DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT:
                    break
        if split_branch == "adverse" and not adverse_query_examples:
            adverse_query_examples = list(query_examples)[:DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT]
        supportive_occurrence_count = max(
            len(supportive_query_examples),
            _safe_int(item.get("supportive_occurrence_count", 0)),
        )
        adverse_occurrence_count = max(
            len(adverse_query_examples),
            _safe_int(item.get("adverse_occurrence_count", 0)),
        )
        remerge_events = _safe_int(item.get("remerge_events", 0))
        return {
            "record_id": self._normalize_action_text(item.get("record_id", "")) or str(uuid4()),
            "created_at": str(item.get("created_at") or datetime.now(timezone.utc).isoformat()),
            "created_token_count": _safe_int(item.get("created_token_count", current_token)),
            "origin": self._normalize_action_text(item.get("origin", "response_selected_evidence")) or "response_selected_evidence",
            "query_text": query_text,
            "query_examples": list(query_examples)[:DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT],
            "query_terms": list(dict.fromkeys(query_terms))[:DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_TERM_LIMIT],
            "baseline_query_score": float(baseline_query_score),
            "best_query_score": float(best_query_score),
            "baseline_grounded_fraction": float(baseline_grounded_fraction),
            "best_grounded_fraction": float(best_grounded_fraction),
            "outcome_score": float(_safe_float(item.get("outcome_score", 0.0))),
            "source_weights": dict(source_weights),
            "provider_weights": dict(provider_weights),
            "credit_events": int(credit_events),
            "penalty_events": int(penalty_events),
            "forgiveness_events": int(forgiveness_events),
            "cooling_events": _safe_int(item.get("cooling_events", 0)),
            "aggregate_count": max(1, _safe_int(item.get("aggregate_count", 1))),
            "aggregation_events": _safe_int(item.get("aggregation_events", 0)),
            "supportive_query_examples": list(supportive_query_examples),
            "adverse_query_examples": list(adverse_query_examples),
            "supportive_occurrence_count": int(supportive_occurrence_count),
            "adverse_occurrence_count": int(adverse_occurrence_count),
            "trajectory_credit_total": float(trajectory_credit_total),
            "trajectory_penalty_total": float(trajectory_penalty_total),
            "trajectory_forgiveness_total": float(trajectory_forgiveness_total),
            "trajectory_event_count": int(trajectory_event_count),
            "trajectory_net_score": float(trajectory_net_score),
            "trajectory_recent_delta_ema": float(_safe_signed(item.get("trajectory_recent_delta_ema", 0.0), limit=1.0)),
            "trajectory_peak_score": float(trajectory_peak_score),
            "trajectory_floor_score": float(trajectory_floor_score),
            "unresolved_penalty_balance": float(_safe_float(item.get("unresolved_penalty_balance", 0.0))),
            "resolved_improvement": float(_safe_float(item.get("resolved_improvement", 0.0))),
            "max_regression": float(_safe_float(item.get("max_regression", 0.0))),
            "max_contradiction_signal": float(_safe_float(item.get("max_contradiction_signal", 0.0))),
            "cumulative_cooling_delta": float(_safe_float(item.get("cumulative_cooling_delta", 0.0))),
            "last_match_score": float(_safe_float(item.get("last_match_score", 0.0))),
            "last_credit_score": float(_safe_float(item.get("last_credit_score", 0.0))),
            "last_penalty_score": float(_safe_float(item.get("last_penalty_score", 0.0))),
            "last_forgiveness_score": float(_safe_float(item.get("last_forgiveness_score", 0.0))),
            "last_penalty_reason": self._normalize_action_text(item.get("last_penalty_reason", "")),
            "last_activity_token_count": _safe_int(item.get("last_activity_token_count", current_token)),
            "last_evaluated_token_count": _safe_int(item.get("last_evaluated_token_count", current_token)),
            "last_cooling_token_count": _safe_int(item.get("last_cooling_token_count", current_token)),
            "last_credit_token_count": _safe_int(item.get("last_credit_token_count", 0)),
            "last_penalty_token_count": _safe_int(item.get("last_penalty_token_count", 0)),
            "last_forgiveness_token_count": _safe_int(item.get("last_forgiveness_token_count", 0)),
            "last_trajectory_event_type": self._normalize_action_text(item.get("last_trajectory_event_type", "")),
            "last_trajectory_event_score": float(_safe_total(item.get("last_trajectory_event_score", 0.0))),
            "last_trajectory_event_at": self._normalize_action_text(item.get("last_trajectory_event_at", "")),
            "last_trajectory_event_token_count": _safe_int(item.get("last_trajectory_event_token_count", 0)),
            "split_generation": _safe_int(item.get("split_generation", 0)),
            "split_parent_record_id": self._normalize_action_text(item.get("split_parent_record_id", "")),
            "split_group_id": self._normalize_action_text(item.get("split_group_id", "")),
            "split_branch": split_branch,
            "remerge_events": int(remerge_events),
            "last_split_at": self._normalize_action_text(item.get("last_split_at", "")),
            "last_remerged_at": self._normalize_action_text(item.get("last_remerged_at", "")),
            "last_aggregated_at": self._normalize_action_text(item.get("last_aggregated_at", "")),
            "last_cooled_at": self._normalize_action_text(item.get("last_cooled_at", "")),
            "last_evaluated_at": self._normalize_action_text(item.get("last_evaluated_at", "")),
            "last_evaluated_query_text": self._normalize_action_text(item.get("last_evaluated_query_text", "")),
        }

    def _normalize_action_record(self, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, Mapping):
            return None
        action_id = " ".join(str(item.get("action_id", "")).split()).strip()
        action_type = " ".join(str(item.get("action_type", item.get("type", ""))).split()).strip().lower()
        if not action_id or not action_type:
            return None
        verification = item.get("verification") if isinstance(item.get("verification"), Mapping) else {}
        topics = [
            " ".join(str(value).split()).strip().lower()
            for value in list(item.get("topics") or [])
            if " ".join(str(value).split()).strip()
        ]
        return {
            "action_id": action_id,
            "action_type": action_type,
            "inputs": deepcopy(dict(item.get("inputs") or {})),
            "predicted_outcome": " ".join(str(item.get("predicted_outcome", "")).split()).strip(),
            "actual_outcome": " ".join(str(item.get("actual_outcome", "")).split()).strip(),
            "verification": {
                "status": " ".join(str(verification.get("status", "unknown")).split()).strip().lower() or "unknown",
                "success": bool(verification.get("success", False)),
                "confidence": float(verification.get("confidence", 0.0) or 0.0),
                "contradiction": bool(verification.get("contradiction", False)),
                "summary": " ".join(str(verification.get("summary", "")).split()).strip(),
                "evidence": [deepcopy(dict(raw)) for raw in list(verification.get("evidence") or []) if isinstance(raw, Mapping)],
            },
            "topics": topics[:8],
            "recorded_at": str(item.get("recorded_at") or datetime.now(timezone.utc).isoformat()),
            "episode_text": " ".join(str(item.get("episode_text", "")).split()).strip(),
            "trigger_reason": " ".join(str(item.get("trigger_reason", "operator")).split()).strip().lower() or "operator",
            "trigger_query_text": " ".join(str(item.get("trigger_query_text", "")).split()).strip(),
        }

    def _action_history_memory_metadata(self, record: Mapping[str, Any]) -> dict[str, Any]:
        verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
        success = bool(verification.get("success", False))
        contradiction = bool(verification.get("contradiction", False))
        evidence = [deepcopy(dict(raw)) for raw in list(verification.get("evidence") or []) if isinstance(raw, Mapping)]
        return {
            "observation_kind": "action",
            "grounded": True,
            "grounding_signal": 0.92 if success else 0.72,
            "evidence_unit_count": max(1, len(evidence)),
            "source_name": "workspace",
            "source_type": "action",
            "action_id": str(record.get("action_id", "")),
            "action_type": str(record.get("action_type", "")),
            "action_inputs": deepcopy(dict(record.get("inputs") or {})),
            "predicted_outcome": str(record.get("predicted_outcome", "")),
            "actual_outcome": str(record.get("actual_outcome", "")),
            "verification_status": str(verification.get("status", "unknown")),
            "verification_confidence": float(verification.get("confidence", 0.0) or 0.0),
            "contradiction": bool(contradiction),
            "evidence": evidence,
        }

    def _inject_action_record_into_cortex_locked(self, record: Mapping[str, Any]) -> None:
        if self._thought_loop is None:
            return
        content = " ".join(str(record.get("episode_text", "")).split()).strip()
        if not content:
            return
        verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
        self._thought_loop.inject_action_result(
            content=content,
            topics=tuple(str(item) for item in list(record.get("topics") or []) if str(item).strip()),
            success=bool(verification.get("success", False)),
            confidence=float(verification.get("confidence", 0.0) or 0.0),
            contradicted=bool(verification.get("contradiction", False)),
            metadata=self._action_history_memory_metadata(record),
        )

    def _replay_action_history_into_cortex_locked(self) -> None:
        for record in reversed(list(self._action_history)):
            self._inject_action_record_into_cortex_locked(record)

    def _action_loop_summary_locked(self) -> dict[str, Any]:
        verified = 0
        contradicted = 0
        for record in self._action_history:
            verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
            if bool(verification.get("success", False)):
                verified += 1
            if bool(verification.get("contradiction", False)):
                contradicted += 1
        last_action = None if not self._action_history else deepcopy(self._action_history[0])
        return {
            "enabled": True,
            "root_path": str(self._action_root),
            "supported_actions": ["workspace_search", "workspace_read", "web_fetch", "api_request"],
            "actions_recorded": int(len(self._action_history)),
            "verified_actions": int(verified),
            "contradicted_actions": int(contradicted),
            "last_action": last_action,
        }

    @staticmethod
    def _normalize_action_text(value: Any) -> str:
        return " ".join(str(value).split()).strip()

    @classmethod
    def _action_request_has_body(cls, inputs: Mapping[str, Any]) -> bool:
        if not isinstance(inputs, Mapping):
            return False
        if "json_body" not in inputs:
            return False
        body = inputs.get("json_body")
        if body is None:
            return False
        if isinstance(body, str):
            return bool(cls._normalize_action_text(body))
        if isinstance(body, Mapping):
            return bool(dict(body))
        if isinstance(body, Sequence) and not isinstance(body, (str, bytes, bytearray)):
            return bool(list(body))
        return True

    @classmethod
    def _api_request_record_matches_explicit_url(cls, record: Mapping[str, Any], explicit_url: str) -> bool:
        if str(record.get("action_type", "")) != "api_request":
            return False
        inputs = record.get("inputs") if isinstance(record.get("inputs"), Mapping) else {}
        if cls._normalize_action_text(inputs.get("url", "")) != explicit_url:
            return False
        method = cls._normalize_action_text(inputs.get("method", "GET")).upper() or "GET"
        if method != "GET":
            return False
        return not cls._action_request_has_body(inputs)

    @classmethod
    def _action_query_terms(cls, query_text: str) -> tuple[str, ...]:
        normalized = cls._normalize_action_text(query_text).lower()
        if not normalized:
            return ()
        terms = [term.lower() for term in salient_query_terms(normalized) if term]
        if not terms:
            terms = [
                token.lower()
                for token in re.findall(r"[a-zA-Z0-9_./:-]+", normalized)
                if len(token) >= 2
            ]
        deduped: list[str] = []
        seen: set[str] = set()
        for term in terms:
            compact = cls._normalize_action_text(term).lower()
            if not compact or compact in seen:
                continue
            deduped.append(compact)
            seen.add(compact)
        return tuple(deduped[:8])

    @classmethod
    def _action_focus_query_text(cls, query_text: str) -> str:
        normalized = cls._normalize_action_text(query_text)
        if not normalized:
            return ""
        stripped = re.sub(r"https?://[^\s'\")\]>]+", " ", normalized, flags=re.IGNORECASE)
        stripped = re.sub(
            r"(?:[A-Za-z0-9_.-]+[\\/])*[A-Za-z0-9_.-]+\.(?:py|md|txt|json|yaml|yml|toml|csv|ts|tsx|js|jsx|html|css|scss|ini|cfg|log|rst)",
            " ",
            stripped,
            flags=re.IGNORECASE,
        )
        focused_terms = cls._action_query_terms(stripped)
        if focused_terms:
            return " ".join(focused_terms[:6])
        fallback_terms = cls._action_query_terms(normalized)
        if fallback_terms:
            return " ".join(fallback_terms[:6])
        return normalized

    def _query_workspace_path_candidate_locked(self, query_text: str) -> str:
        normalized = self._normalize_action_text(query_text)
        if not normalized:
            return ""
        candidates = re.findall(
            r"(?:[A-Za-z0-9_.-]+[\\/])*[A-Za-z0-9_.-]+\.(?:py|md|txt|json|yaml|yml|toml|csv|ts|tsx|js|jsx|html|css|scss|ini|cfg|log|rst)",
            normalized,
            flags=re.IGNORECASE,
        )
        for raw in candidates:
            cleaned = raw.strip("`'\".,;:!?()[]{} ").replace("\\", "/")
            if not cleaned:
                continue
            candidate = Path(cleaned)
            resolved = candidate if candidate.is_absolute() else (self._action_root / candidate)
            try:
                resolved = resolved.resolve()
            except Exception:
                continue
            if resolved != self._action_root and self._action_root not in resolved.parents:
                continue
            if not resolved.exists() or not resolved.is_file():
                continue
            try:
                return str(resolved.relative_to(self._action_root)).replace("\\", "/")
            except Exception:
                return str(resolved)
        return ""

    @classmethod
    def _query_web_url_candidate(cls, query_text: str) -> str:
        normalized = cls._normalize_action_text(query_text)
        if not normalized:
            return ""
        matches = re.findall(r"https?://[^\s'\")\]>]+", normalized, flags=re.IGNORECASE)
        for raw in matches:
            cleaned = raw.strip("`'\".,;:!?()[]{} ")
            if cleaned:
                return cleaned
        return ""

    @classmethod
    def _query_api_url_candidate(cls, query_text: str) -> str:
        candidate = cls._query_web_url_candidate(query_text)
        if not candidate:
            return ""
        lowered = cls._normalize_action_text(query_text).lower()
        parsed = urlparse(candidate)
        path = (parsed.path or "").lower()
        if path.endswith(".json") or "/api/" in path or any(token in lowered for token in (" api ", " json ", " endpoint ")):
            return candidate
        return ""

    def _action_record_relevance_score_locked(self, record: Mapping[str, Any], query_text: str) -> float:
        normalized_query = self._normalize_action_text(query_text).lower()
        if not normalized_query:
            return 0.0
        explicit_api_url = self._query_api_url_candidate(query_text).lower()
        explicit_url = self._query_web_url_candidate(query_text).lower()
        record_url = self._normalize_action_text((record.get("inputs") or {}).get("url", "")).lower()
        if explicit_api_url and explicit_api_url == record_url:
            if str(record.get("action_type", "")) != "api_request":
                return 0.0
            if self._api_request_record_matches_explicit_url(record, explicit_api_url):
                return 1.0
        if explicit_url and explicit_url == record_url:
            return 1.0
        trigger_query = self._normalize_action_text(record.get("trigger_query_text", "")).lower()
        record_query = self._normalize_action_text((record.get("inputs") or {}).get("query_text", "")).lower()
        if normalized_query and normalized_query in {trigger_query, record_query}:
            return 1.0
        query_terms = set(self._action_query_terms(normalized_query))
        if not query_terms:
            return 0.0
        record_terms: set[str] = set(
            self._normalize_action_text(term).lower()
            for term in list(record.get("topics") or [])
            if self._normalize_action_text(term)
        )
        record_terms.update(self._action_query_terms(record_query))
        record_terms.update(self._action_query_terms(str((record.get("inputs") or {}).get("path", ""))))
        record_terms.update(self._action_query_terms(str((record.get("inputs") or {}).get("url", ""))))
        verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
        for raw_item in list(verification.get("evidence") or []):
            if not isinstance(raw_item, Mapping):
                continue
            record_terms.update(
                self._normalize_action_text(term).lower()
                for term in list(raw_item.get("matched_terms") or [])
                if self._normalize_action_text(term)
            )
            record_terms.update(self._action_query_terms(str(raw_item.get("snippet", ""))))
        if not record_terms:
            record_terms.update(self._action_query_terms(str(record.get("actual_outcome", ""))))
        overlap = len(query_terms & record_terms)
        if overlap <= 0:
            return 0.0
        return float(overlap) / float(max(1, len(query_terms)))

    def _recent_relevant_action_records_locked(
        self,
        query_text: str,
        *,
        statuses: Sequence[str] | None = None,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        allowed = {
            self._normalize_action_text(status).lower()
            for status in list(statuses or [])
            if self._normalize_action_text(status)
        }
        ranked: list[tuple[float, dict[str, Any]]] = []
        for record in self._action_history:
            verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
            status = self._normalize_action_text(verification.get("status", "")).lower()
            if allowed and status not in allowed:
                continue
            score = self._action_record_relevance_score_locked(record, query_text)
            if score < 0.34:
                continue
            ranked.append((score, deepcopy(record)))
        ranked.sort(
            key=lambda item: (
                float(item[0]),
                str(item[1].get("recorded_at", "")),
            ),
            reverse=True,
        )
        return [record for _, record in ranked[: max(1, int(limit))]]

    def _action_record_to_response_episodes_locked(
        self,
        record: Mapping[str, Any],
        *,
        query_text: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
        if not bool(verification.get("success", False)):
            return []
        query_terms = list(self._action_query_terms(query_text))
        evidence_items = [
            dict(raw)
            for raw in list(verification.get("evidence") or [])
            if isinstance(raw, Mapping)
        ]
        action_seed = int(hashlib.sha256(str(record.get("action_id", "")).encode("utf-8")).hexdigest()[:8], 16)
        episodes: list[dict[str, Any]] = []
        for idx, evidence in enumerate(evidence_items[: max(1, int(limit))]):
            snippet = self._normalize_action_text(evidence.get("snippet", ""))
            if not snippet:
                continue
            matching = tuple(match_terms(query_terms, snippet))
            overlap_ratio = float(len(matching)) / float(max(1, len(query_terms)))
            exact_query = bool(evidence.get("exact_query", False))
            similarity = max(0.46, 0.56 + 0.34 * overlap_ratio + (0.10 if exact_query else 0.0))
            memory_index = -1 * int(action_seed + idx + 1)
            episodes.append(
                {
                    "text": snippet,
                    "raw_window": snippet,
                    "memory_index": memory_index,
                    "memory_indices": [memory_index],
                    "similarity": float(min(0.99, similarity)),
                    "importance": float(verification.get("confidence", 0.0) or 0.0),
                    "age_tokens": 0,
                    "match_count": 1,
                    "query_overlap": int(len(matching)),
                    "focus_overlap": 0,
                    "memory_focus_priority": 0.0,
                    "complete_sentence": int(snippet.endswith((".", "!", "?"))),
                    "clipped_overlap": 0,
                    "expansion_chars": 0,
                    "action_origin": str(record.get("action_id", "")),
                    "action_type": str(record.get("action_type", "")),
                    "source_path": self._normalize_action_text(evidence.get("path", "")),
                    "line_number": int(evidence.get("line_number", 0) or 0),
                }
            )
        if episodes:
            return episodes
        summary = self._normalize_action_text(record.get("actual_outcome", ""))
        if not summary:
            return []
        matching = tuple(match_terms(query_terms, summary))
        overlap_ratio = float(len(matching)) / float(max(1, len(query_terms)))
        memory_index = -1 * int(action_seed + 999)
        return [
            {
                "text": summary,
                "raw_window": summary,
                "memory_index": memory_index,
                "memory_indices": [memory_index],
                "similarity": float(min(0.95, 0.48 + 0.32 * overlap_ratio)),
                "importance": float(verification.get("confidence", 0.0) or 0.0),
                "age_tokens": 0,
                "match_count": 1,
                "query_overlap": int(len(matching)),
                "focus_overlap": 0,
                "memory_focus_priority": 0.0,
                "complete_sentence": int(summary.endswith((".", "!", "?"))),
                "clipped_overlap": 0,
                "expansion_chars": 0,
                "action_origin": str(record.get("action_id", "")),
                "action_type": str(record.get("action_type", "")),
            }
        ]

    def _augment_query_result_with_action_records_locked(
        self,
        query_result: dict[str, Any],
        *,
        query_text: str,
        records: Sequence[Mapping[str, Any]],
    ) -> int:
        query_summary = query_result.get("query_summary")
        if not isinstance(query_summary, dict):
            return 0
        injected: list[dict[str, Any]] = []
        seen_texts: set[str] = set()
        for record in records:
            for episode in self._action_record_to_response_episodes_locked(record, query_text=query_text):
                text_key = self._normalize_action_text(episode.get("text", "")).lower()
                if not text_key or text_key in seen_texts:
                    continue
                injected.append(episode)
                seen_texts.add(text_key)
        existing = [
            deepcopy(item)
            for item in list(query_summary.get("memory_episodes") or [])
            if isinstance(item, Mapping)
        ]
        for item in existing:
            text_key = self._normalize_action_text(item.get("text", item.get("raw_window", ""))).lower()
            if text_key:
                seen_texts.add(text_key)
        if injected:
            query_summary["memory_episodes"] = injected + [
                item
                for item in existing
                if self._normalize_action_text(item.get("text", item.get("raw_window", ""))).lower() not in {
                    self._normalize_action_text(injected_item.get("text", "")).lower()
                    for injected_item in injected
                }
            ]
        return int(len(injected))

    def _contradicted_action_note_locked(self, record: Mapping[str, Any]) -> str:
        actual = self._normalize_action_text(record.get("actual_outcome", ""))
        if actual:
            return f" I checked the workspace and observed: {actual}"
        return " I checked the workspace and found no additional grounded evidence there."

    def _should_auto_execute_action_locked(
        self,
        *,
        query_text: str,
        query_result: dict[str, Any],
        response: Mapping[str, Any],
    ) -> bool:
        if not self._normalize_action_text(query_text):
            return False
        gap_plan = query_result.get("gap_plan") if isinstance(query_result.get("gap_plan"), Mapping) else {}
        meaningful_gap = bool(
            gap_plan.get("unsupported_terms")
            or gap_plan.get("gap_terms")
            or gap_plan.get("weak_concepts")
            or float(gap_plan.get("grounded_fraction", 0.0) or 0.0) < 0.999
        )
        if not meaningful_gap:
            return False
        response_mode = self._normalize_action_text(response.get("response_mode", "")).lower()
        if response_mode == "insufficient_evidence":
            return True
        unsupported_terms = list(response.get("unsupported_terms") or gap_plan.get("unsupported_terms") or [])
        evidence_coverage = float(response.get("evidence_coverage", 0.0) or 0.0)
        return bool(unsupported_terms) and evidence_coverage < 0.85

    def _maybe_auto_action_assist_locked(
        self,
        *,
        query_text: str,
        query_result: dict[str, Any],
        response: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        explicit_api_url = self._query_api_url_candidate(query_text)
        explicit_url = self._query_web_url_candidate(query_text) if not explicit_api_url else ""
        explicit_path = self._query_workspace_path_candidate_locked(query_text) if not (explicit_api_url or explicit_url) else ""
        verified_records = self._recent_relevant_action_records_locked(query_text, statuses=("verified",), limit=2)
        if explicit_api_url:
            verified_records = [
                record
                for record in verified_records
                if self._api_request_record_matches_explicit_url(record, explicit_api_url)
            ]
        elif explicit_url:
            verified_records = [
                record
                for record in verified_records
                if str(record.get("action_type", "")) == "web_fetch"
                and self._normalize_action_text((record.get("inputs") or {}).get("url", "")) == explicit_url
            ]
        elif explicit_path:
            verified_records = [
                record
                for record in verified_records
                if str(record.get("action_type", "")) == "workspace_read"
                and self._normalize_action_text((record.get("inputs") or {}).get("path", "")) == explicit_path
            ]
        if verified_records:
            injected = self._augment_query_result_with_action_records_locked(
                query_result,
                query_text=query_text,
                records=verified_records,
            )
            return {
                "triggered": True,
                "executed": False,
                "reused_recent_action": True,
                "reason": "recent_verified_action",
                "used_in_response": bool(injected > 0),
                "result": deepcopy(verified_records[0]),
                "result_count": int(len(verified_records)),
                "response_episode_count": int(injected),
            }

        contradicted_records = self._recent_relevant_action_records_locked(query_text, statuses=("contradicted",), limit=1)
        if explicit_api_url:
            contradicted_records = [
                record
                for record in contradicted_records
                if self._api_request_record_matches_explicit_url(record, explicit_api_url)
            ]
        elif explicit_url:
            contradicted_records = [
                record
                for record in contradicted_records
                if str(record.get("action_type", "")) == "web_fetch"
                and self._normalize_action_text((record.get("inputs") or {}).get("url", "")) == explicit_url
            ]
        elif explicit_path:
            contradicted_records = [
                record
                for record in contradicted_records
                if str(record.get("action_type", "")) == "workspace_read"
                and self._normalize_action_text((record.get("inputs") or {}).get("path", "")) == explicit_path
            ]
        if not self._should_auto_execute_action_locked(query_text=query_text, query_result=query_result, response=response):
            response_mode = self._normalize_action_text(response.get("response_mode", "")).lower()
            unsupported_terms = list(response.get("unsupported_terms") or [])
            if contradicted_records and (response_mode == "insufficient_evidence" or unsupported_terms):
                return {
                    "triggered": True,
                    "executed": False,
                    "reused_recent_action": True,
                    "reason": "recent_contradicted_action",
                    "used_in_response": False,
                    "result": deepcopy(contradicted_records[0]),
                    "result_count": 1,
                    "response_episode_count": 0,
                    "response_note": self._contradicted_action_note_locked(contradicted_records[0]),
                }
            return None

        if contradicted_records:
            return {
                "triggered": True,
                "executed": False,
                "reused_recent_action": True,
                "reason": "recent_contradicted_action",
                "used_in_response": False,
                "result": deepcopy(contradicted_records[0]),
                "result_count": 1,
                "response_episode_count": 0,
                "response_note": self._contradicted_action_note_locked(contradicted_records[0]),
            }

        gap_plan = query_result.get("gap_plan") if isinstance(query_result.get("gap_plan"), Mapping) else {}
        retrieval_queries = [
            self._normalize_action_text(value)
            for value in list(gap_plan.get("retrieval_queries") or [])
            if self._normalize_action_text(value)
        ]
        search_query = retrieval_queries[0] if retrieval_queries else self._normalize_action_text(query_text)
        if explicit_api_url:
            focused_query = self._action_focus_query_text(query_text)
            action_result = self.execute_digital_action(
                {
                    "action_type": "api_request",
                    "url": explicit_api_url,
                    "query_text": focused_query,
                    "predicted_outcome": f"I expect requesting structured JSON from {explicit_api_url} to provide grounded evidence relevant to: {self._normalize_action_text(query_text)}.",
                },
                trigger_reason="query_gap_auto_api_request",
                trigger_query_text=query_text,
            )
            assist_reason = "query_gap_auto_api_request"
        elif explicit_url:
            focused_query = self._action_focus_query_text(query_text)
            action_result = self.execute_digital_action(
                {
                    "action_type": "web_fetch",
                    "url": explicit_url,
                    "query_text": focused_query,
                    "predicted_outcome": f"I expect fetching {explicit_url} to provide grounded evidence relevant to: {self._normalize_action_text(query_text)}.",
                },
                trigger_reason="query_gap_auto_fetch",
                trigger_query_text=query_text,
            )
            assist_reason = "query_gap_auto_fetch"
        elif explicit_path:
            focused_query = self._action_focus_query_text(query_text)
            action_result = self.execute_digital_action(
                {
                    "action_type": "workspace_read",
                    "path": explicit_path,
                    "query_text": focused_query,
                    "predicted_outcome": f"I expect reading {explicit_path} to provide grounded evidence relevant to: {self._normalize_action_text(query_text)}.",
                },
                trigger_reason="query_gap_auto_read",
                trigger_query_text=query_text,
            )
            assist_reason = "query_gap_auto_read"
        else:
            action_result = self.execute_digital_action(
                {
                    "action_type": "workspace_search",
                    "query_text": search_query,
                    "predicted_outcome": f"I expect workspace search to find grounded evidence relevant to: {self._normalize_action_text(query_text)}.",
                },
                trigger_reason="query_gap_auto_search",
                trigger_query_text=query_text,
            )
            assist_reason = "query_gap_auto_search"
        if not bool(action_result.get("accepted", False)):
            return {
                "triggered": True,
                "executed": False,
                "reused_recent_action": False,
                "reason": "auto_action_execution_failed",
                "used_in_response": False,
                "error": self._normalize_action_text(action_result.get("reason", "execution_failed")),
            }
        record = cast(dict[str, Any], action_result.get("result") or {})
        verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
        injected = 0
        if bool(verification.get("success", False)):
            injected = self._augment_query_result_with_action_records_locked(
                query_result,
                query_text=query_text,
                records=[record],
            )
        assist = {
            "triggered": True,
            "executed": True,
            "reused_recent_action": False,
            "reason": assist_reason,
            "used_in_response": bool(injected > 0),
            "result": deepcopy(record),
            "result_count": 1,
            "response_episode_count": int(injected),
        }
        if bool(verification.get("contradiction", False)):
            assist["response_note"] = self._contradicted_action_note_locked(record)
        return assist

    def _plan_gaps_locked(
        self,
        *,
        query_text: str,
        query_result: dict[str, Any],
    ) -> dict[str, Any]:
        return plan_query_gaps(
            query_text=query_text,
            query_summary=query_result.get("query_summary"),
            concept_summary=query_result.get("concept_summary"),
        )

    def _normalize_recent_query_gap(self, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None
        query_text = " ".join(str(item.get("query_text", "")).split()).strip()
        if not query_text:
            return None
        unsupported_terms = [
            str(term).strip().lower()
            for term in list(item.get("unsupported_terms") or [])
            if str(term).strip()
        ]
        gap_terms: list[dict[str, Any]] = []
        for raw_gap in list(item.get("gap_terms") or []):
            if not isinstance(raw_gap, dict):
                continue
            term = str(raw_gap.get("term", "")).strip().lower()
            if not term:
                continue
            gap_terms.append(
                {
                    "term": term,
                    "weight": float(raw_gap.get("weight", 0.0)),
                }
            )
        retrieval_queries = [
            " ".join(str(value).split()).strip()
            for value in list(item.get("retrieval_queries") or [])
            if " ".join(str(value).split()).strip()
        ]
        follow_up_questions = [
            " ".join(str(value).split()).strip()
            for value in list(item.get("follow_up_questions") or [])
            if " ".join(str(value).split()).strip()
        ]
        weak_concepts: list[dict[str, Any]] = []
        for raw_concept in list(item.get("weak_concepts") or []):
            if not isinstance(raw_concept, dict):
                continue
            label = " ".join(str(raw_concept.get("label", "")).split()).strip()
            top_terms = [
                " ".join(str(value).split()).strip().lower()
                for value in list(raw_concept.get("top_terms") or [])
                if " ".join(str(value).split()).strip()
            ]
            if not label and not top_terms:
                continue
            weak_concepts.append(
                {
                    "label": label,
                    "weakness": float(raw_concept.get("weakness", 0.0)),
                    "uncertainty": float(raw_concept.get("uncertainty", 0.0)),
                    "drift": float(raw_concept.get("drift", 0.0)),
                    "top_terms": top_terms[:4],
                    "match_count": max(0, int(raw_concept.get("match_count", 0))),
                }
            )
        return {
            "recorded_at": str(item.get("recorded_at") or datetime.now(timezone.utc).isoformat()),
            "source": str(item.get("source") or "query"),
            "query_text": query_text,
            "unsupported_terms": unsupported_terms,
            "gap_terms": gap_terms,
            "retrieval_queries": retrieval_queries[:4],
            "follow_up_questions": follow_up_questions[:4],
            "weak_concepts": weak_concepts[:4],
            "grounded_fraction": float(item.get("grounded_fraction", 0.0)),
        }

    def _record_recent_query_gap_locked(
        self,
        *,
        query_text: str,
        gap_plan: dict[str, Any],
        source: str,
    ) -> None:
        normalized_query = " ".join(str(query_text).split()).strip()
        if not normalized_query:
            return
        existing = [
            item
            for item in list(self._brain_recent_query_gaps)
            if str(item.get("query_text", "")).lower() != normalized_query.lower()
        ]
        self._brain_recent_query_gaps = deque(existing, maxlen=DEFAULT_RECENT_QUERY_GAP_HISTORY)
        grounded_fraction = float(gap_plan.get("grounded_fraction", 0.0))
        query_deficit = bool(gap_plan.get("unsupported_terms")) or grounded_fraction < 0.999
        self._brain_skip_next_autonomy_for_grounded_query = not query_deficit
        meaningful = bool(query_deficit and (gap_plan.get("unsupported_terms") or gap_plan.get("gap_terms") or gap_plan.get("weak_concepts")))
        if not meaningful:
            return
        normalized = self._normalize_recent_query_gap(
            {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "query_text": normalized_query,
                "unsupported_terms": list(gap_plan.get("unsupported_terms") or []),
                "gap_terms": list(gap_plan.get("gap_terms") or []),
                "retrieval_queries": list(gap_plan.get("retrieval_queries") or []),
                "follow_up_questions": list(gap_plan.get("follow_up_questions") or []),
                "weak_concepts": list(gap_plan.get("weak_concepts") or []),
                "grounded_fraction": float(gap_plan.get("grounded_fraction", 0.0)),
            }
        )
        if normalized is not None:
            self._brain_recent_query_gaps.appendleft(normalized)

    def _resolve_save_path(self, path: str | None) -> Path:
        if path:
            return Path(path)

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return self._checkpoint_dir / f"hecsn_service_{stamp}.pt"

    def _persist_trace_locked(self, trace: dict[str, Any]) -> Path:
        payload = self._json_safe(trace)
        created_at = str(payload.get("created_at", datetime.now(timezone.utc).isoformat()))
        timestamp = created_at.replace(":", "").replace("-", "")
        trace_id = str(payload.get("trace_id", uuid4()))
        trace_path = self._trace_dir / f"{timestamp}_{trace_id}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_file(trace_path, payload)
        payload["trace_path"] = str(trace_path)
        self._trace_history.appendleft(deepcopy(payload))
        return trace_path

    def _load_persisted_traces_locked(self) -> None:
        trace_files = sorted(self._trace_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        for path in trace_files[: self._trace_history.maxlen]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            payload["trace_path"] = str(path)
            self._trace_history.append(payload)

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, deque)):
            return [self._json_safe(item) for item in value]
        return str(value)

    def _record_brain_event_locked(self, event: dict[str, Any]) -> None:
        payload = cast(dict[str, Any], self._json_safe(event))
        self._brain_last_event = deepcopy(payload)
        self._brain_event_history.appendleft(deepcopy(payload))
