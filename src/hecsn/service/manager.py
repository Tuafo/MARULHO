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
from hecsn.service.action_executor import ActionExecutor
from hecsn.service.feedback_applier import FeedbackApplier
from hecsn.service.brain_runtime import BRAIN_RUNTIME_STATE_FIELDS, BrainRuntime, BrainRuntimeDependencies
from hecsn.service.delayed_consequence import DELAYED_CONSEQUENCE_STATE_FIELDS, DelayedConsequenceDependencies, DelayedConsequenceTracker
from hecsn.service.persistence import RuntimePersistence, RuntimePersistenceDependencies
from hecsn.service.cortex_controller import CORTEX_CONTROLLER_STATE_FIELDS, CortexController, CortexControllerDependencies
from hecsn.service.reporting import ServiceReportingMixin
from hecsn.service.replay_runtime import ReplayController, ReplayControllerDependencies
from hecsn.service.interaction_runtime import InteractionRuntimeMixin
from hecsn.service.living_status import LivingStatusMixin
from hecsn.service.runtime_config import RuntimeConfig
from hecsn.service.runtime_control import RUNTIME_CONTROL_STATE_FIELDS, RuntimeControl
from hecsn.service.runtime_state import RuntimeState
from hecsn.service.runtime_prewarm import RuntimePrewarmMixin
from hecsn.service.runtime_sources import RuntimeSources, RuntimeSourcesDependencies, _BrainSourceRuntime, _SensorySourceRuntime
from hecsn.service.sensory_runtime import SensoryRuntimeMixin
from hecsn.service.autonomy_planner import AutonomyPlanner
from hecsn.service.status_runtime import StatusRuntimeMixin
from hecsn.service.status_read_model import StatusReadModel
from hecsn.service.sensory_preview import SensoryPreviewMixin
from hecsn.service.source_focus import SourceFocusDependencies, SourceFocusScorer
from hecsn.service.terminus_autonomy import TerminusAutonomyMixin
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

def _public_names(cls: type) -> frozenset[str]:
    """Return the set of non-dunder attribute names defined on *cls*."""
    return frozenset(name for name in cls.__dict__ if not name.startswith("__"))


_PUBLIC_DUNDER = frozenset({"__dict__", "__doc__", "__init__", "__module__", "__qualname__"})

_BRAIN_RUNTIME_DELEGATE_NAMES = frozenset(BrainRuntime.__dict__)
_BRAIN_RUNTIME_DELEGATE_ATTRS = BRAIN_RUNTIME_STATE_FIELDS | _BRAIN_RUNTIME_DELEGATE_NAMES
_RUNTIME_CONTROL_DELEGATE_NAMES = frozenset(RuntimeControl.__dict__)
_RUNTIME_CONTROL_DELEGATE_ATTRS = RUNTIME_CONTROL_STATE_FIELDS | _RUNTIME_CONTROL_DELEGATE_NAMES
# CortexController methods that just pass through to self._manager should
# NOT be routed from the manager back to the controller (recursion risk).
_CORTEX_CONTROLLER_PASSTHROUGH_NAMES = frozenset({
    "_action_focus_query_text",
    "_action_history_memory_metadata",
    "_action_query_terms",
    "_action_record_relevance_score_locked",
    "_action_record_to_response_episodes_locked",
    "_augment_query_result_with_action_records_locked",
    "_query_api_url_candidate",
    "_query_web_url_candidate",
    "_query_workspace_path_candidate_locked",
    "_recent_relevant_action_records_locked",
    "_record_brain_event_locked",
})
_CORTEX_CONTROLLER_DELEGATE_NAMES = frozenset(
    name for name in CortexController.__dict__
    if not name.startswith("__") and name not in _CORTEX_CONTROLLER_PASSTHROUGH_NAMES
)
_DELAYED_CONSEQUENCE_DELEGATE_NAMES = _public_names(DelayedConsequenceTracker)
_DELAYED_CONSEQUENCE_DELEGATE_ATTRS = DELAYED_CONSEQUENCE_STATE_FIELDS | _DELAYED_CONSEQUENCE_DELEGATE_NAMES
_RUNTIME_EVIDENCE_DELEGATE_NAMES = _public_names(RuntimeEvidenceMixin)
_INTERACTION_RUNTIME_DELEGATE_NAMES = _public_names(InteractionRuntimeMixin)
_LIVING_STATUS_DELEGATE_NAMES = _public_names(LivingStatusMixin)
_STATUS_RUNTIME_DELEGATE_NAMES = _public_names(StatusRuntimeMixin)
_SENSORY_RUNTIME_DELEGATE_NAMES = _public_names(SensoryRuntimeMixin)
_RUNTIME_PREWARM_DELEGATE_NAMES = _public_names(RuntimePrewarmMixin)
_RUNTIME_SOURCES_DELEGATE_NAMES = _public_names(RuntimeSources)
_RUNTIME_CONFIG_DELEGATE_NAMES = _public_names(RuntimeConfig)
_REPLAY_DATASET_BUNDLE_DELEGATE_NAMES = _public_names(ReplayDatasetBundleMixin)
_SENSORY_PREVIEW_DELEGATE_NAMES = _public_names(SensoryPreviewMixin)
_SERVICE_REPORTING_DELEGATE_NAMES = _public_names(ServiceReportingMixin)
_TERMINUS_AUTONOMY_DELEGATE_NAMES = _public_names(TerminusAutonomyMixin)
_REPLAY_CONTROLLER_DELEGATE_NAMES = _public_names(ReplayController)
_SOURCE_FOCUS_DELEGATE_NAMES = _public_names(SourceFocusScorer)
_RUNTIME_PERSISTENCE_INTERNAL_DELEGATE_NAMES = frozenset({
    "_brain_persisted_state_locked",
    "_brain_runtime_snapshot_locked",
    "_join_brain_thread",
    "_normalize_background_source_utility_state",
    "_normalize_delayed_consequence_record",
    "_rebuild_brain_sources_locked",
    "_replay_action_history_into_cortex_locked",
    "_request_brain_stop",
})


class _TimedCallFailure:
    def __init__(self, error: BaseException) -> None:
        self.error = error




class HECSNServiceManager:
    """Main service orchestrator for HECSN/Terminus (ADR 0003 composition root).

    Thin composition root that wires deep modules and exposes the public
    HECSNServiceManager / FastAPI contract. All runtime behaviour is owned by
    explicit constructor-injected modules; no mixin inheritance remains.

    Deep modules:
      _runtime_state        � mutation truth and brain event history
      _brain_runtime        � source rebuild, tick, source utility, autonomy, snapshots
      _runtime_control      � configure/start/stop/tick lifecycle and prewarm
      _cortex_controller    � cortex ask/sleep/thought/action-intent control
      _delayed_consequence   � long-horizon consequence record state machines
      _interaction_pipeline � query/feed/respond turn seam
      _action_executor      � digital action execution and history
      _feedback_applier     � verdict normalization and feedback application
      _status_read_model    � read-only status/telemetry projections
      _runtime_persistence  � checkpoint save/restore and trace history
      _replay_controller    � replay planning and sampling
      _autonomy_planner     � focus planning and provider curriculum
      _runtime_config       � config validation and normalization
      _runtime_sources      � source stream construction and caches
      _source_focus         � source selection scoring and utility EMA
    """

    def start_terminus(self) -> dict[str, Any]:
        return self._runtime_control.start_terminus()

    @staticmethod
    def quick_start_presets() -> list[dict[str, Any]]:
        return RuntimeControl.quick_start_presets()

    @staticmethod
    def _build_source_stream_from_spec(
        spec: dict[str, Any],
        encoder: Any,
        window_size: int,
    ) -> Iterator[tuple[str, "torch.Tensor"]]:
        return BrainRuntime._build_source_stream_from_spec(spec, encoder, window_size)

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
        self._runtime_config = RuntimeConfig(
            provider_query_family_priority=self._provider_query_family_priority_locked,
            provider_topic_family_priority=self._provider_topic_family_priority_locked,
        )
        self._runtime_sources = RuntimeSources(
            RuntimeSourcesDependencies(
                brain_config=lambda: self._brain_config,
                brain_source_runtimes=lambda: self._brain_source_runtimes,
                set_brain_source_runtimes=lambda value: setattr(self, "_brain_source_runtimes", list(value)),
                checkpoint_dir=lambda: self._checkpoint_dir,
                checkpoint_path=lambda: self._checkpoint_path,
                encoder=lambda: self._encoder,
                sensory_queue_target_items=self._sensory_queue_target_items_locked,
                sensory_source_runtimes=lambda: self._sensory_source_runtimes,
                set_sensory_source_runtimes=lambda value: setattr(self, "_sensory_source_runtimes", list(value)),
                trainer=lambda: self._trainer,
            )
        )
        self._source_focus = SourceFocusScorer(
            SourceFocusDependencies(
                autonomy_focus_plan=self._autonomy_focus_plan_locked,
                background_source_utility_entry=self._background_source_utility_entry_locked,
                brain_source_runtimes=lambda: self._brain_source_runtimes,
                concept_store_snapshot=lambda limit: self._concept_store.snapshot(limit=limit),
                geometric_curiosity_focus_plan=lambda top_n: self._geometric_curiosity.focus_plan(top_n=top_n),
                interaction_recent_query_gaps=lambda: self._interaction_pipeline.recent_query_gaps(),
                normalize_action_text=self._normalize_action_text,
                source_text_overlap=self._source_text_overlap,
                thought_loop=lambda: self._thought_loop_actual,
            )
        )
        self._autonomy_planner = AutonomyPlanner(self)
        self._runtime_control = RuntimeControl(self)
        self._cortex_controller = CortexController(
            CortexControllerDependencies(
                action_history=lambda: self._action_history,
                action_history_memory_metadata=self._action_history_memory_metadata,
                action_query_terms=self._action_query_terms,
                action_focus_query_text=self._action_focus_query_text,
                api_request_record_matches_explicit_url=self._api_request_record_matches_explicit_url,
                checkpoint_dir=lambda: self._checkpoint_dir,
                cortex_signal_state=self._cortex_signal_state,
                lock=self._lock,
                query_api_url_candidate=self._query_api_url_candidate,
                query_web_url_candidate=self._query_web_url_candidate,
                query_workspace_path_candidate=self._query_workspace_path_candidate_locked,
                recent_relevant_action_records=self._recent_relevant_action_records_locked,
                record_brain_event=self._record_brain_event_locked,
                action_record_relevance_score=self._action_record_relevance_score_locked,
                action_record_to_response_episodes=self._action_record_to_response_episodes_locked,
                augment_query_result_with_action_records=self._augment_query_result_with_action_records_locked,
                brain_running=lambda: self._brain_running,
                execute_digital_action=self.execute_digital_action,
            )
        )
        service_state = dict(self._metadata.get("service_state", {}))
        terminus_state = dict(service_state.get("terminus_runtime", service_state.get("brain_runtime")) or {})
        concept_state = service_state.get("concept_store")
        self._concept_store = ConceptStore.from_state_dict(concept_state)
        self._runtime_persistence = RuntimePersistence(
            RuntimePersistenceDependencies(
                get_state=lambda name: object.__getattribute__(self, name),
                set_state=lambda name, value: setattr(self, name, value),
                brain_persisted_state=lambda: self._brain_runtime._brain_persisted_state_locked(),
                brain_runtime_snapshot=self._brain_runtime_snapshot_locked,
                join_brain_thread=lambda *args, **kwargs: self._runtime_control._join_brain_thread(*args, **kwargs),
                lock=self._lock,
                normalize_background_source_utility_state=self._normalize_background_source_utility_state,
                normalize_delayed_consequence_record=self._normalize_delayed_consequence_record,
                rebuild_brain_sources=lambda: self._brain_runtime._rebuild_brain_sources_locked(),
                replay_action_history_into_cortex=self._replay_action_history_into_cortex_locked,
                request_brain_stop=lambda *args, **kwargs: self._runtime_control._request_brain_stop(*args, **kwargs),
            ),
            trace_history_limit=trace_history_limit,
        )
        self._geometric_curiosity = GeometricCuriosityController.from_state_dict(
            self._trainer.model.abstraction_layer,
            cast(dict[str, Any] | None, terminus_state.get("geometric_curiosity")),
        )
        self._brain_config = self._runtime_config._normalize_brain_config(
            terminus_state
        )
        self._brain_runtime = BrainRuntime(self._build_brain_runtime_dependencies())
        self._brain_runtime.restore_runtime_state(terminus_state)
        self._action_executor = self._build_action_executor(
            action_history=list(terminus_state.get("action_history") or [])
        )
        self._replay_controller = ReplayController(
            ReplayControllerDependencies(
                action_history=lambda: self._action_history,
                cortex_unavailable_snapshot=self._cortex_unavailable_snapshot,
                living_loop_snapshot=lambda **kwargs: LivingStatusMixin._living_loop_snapshot_locked(self, **kwargs),
                lock=self._lock,
                normalize_action_text=self._normalize_action_text,
                normalize_feedback_text=self._normalize_feedback_text,
                replay_plan_summary=lambda replay_plan: RuntimeEvidenceMixin._replay_plan_summary(self, replay_plan),
                runtime_feedback_summary=lambda: RuntimeEvidenceMixin._runtime_feedback_summary_locked(self),
                runtime_state=self._runtime_state,
                runtime_trace_export_safe_value=lambda value: RuntimeEvidenceMixin._runtime_trace_export_safe_value(self, value),
                thought_loop=lambda: self._thought_loop_actual,
                trainer=lambda: self._trainer,
            ),
            replay_sample_history=list(terminus_state.get("replay_sample_history") or []),
        )
        self._delayed_consequence = DelayedConsequenceTracker(
            DelayedConsequenceDependencies(
                action_record_relevance_score=self._action_record_relevance_score_locked,
                background_focus_terms=self._background_focus_terms_locked,
                background_source_utility_entry=self._background_source_utility_entry_locked,
                brain_config=lambda: self._brain_config,
                brain_source_runtimes=lambda: self._brain_source_runtimes,
                brain_source_semantic_match=self._brain_source_semantic_match_locked,
                normalize_action_text=self._normalize_action_text,
                normalize_provider_curriculum=self._normalize_provider_curriculum,
                recent_relevant_action_records=self._recent_relevant_action_records_locked,
                record_brain_event=self._record_brain_event_locked,
                runtime_state=self._runtime_state,
                selected_evidence_weight_map=self._selected_evidence_weight_map,
                source_text_overlap=self._source_text_overlap,
                trainer=lambda: self._trainer,
            )
        )
        self._delayed_consequence.restore_state(terminus_state)
        self._rebuild_brain_sources_locked()
        self._runtime_state.restore_event_history(
            last_event=terminus_state.get("last_event"),
            recent_events=terminus_state.get("recent_events"),
        )
        self._runtime_state.restore_clean()
        self._load_persisted_traces_locked()

        # --- Status Read Model (ADR 0003 deep module extraction) ---
        self._status_read_model = self._build_status_read_model()
        # --- Interaction Pipeline (ADR 0003 query/feed/respond-turn extraction) ---
        self._interaction_pipeline = self._build_interaction_pipeline(
            recent_query_gaps=list(terminus_state.get("recent_query_gaps") or []),
            runtime_episode_traces=list(terminus_state.get("runtime_episode_traces") or []),
        )
        self._feedback_applier = self._build_feedback_applier()

    def _build_status_read_model(self) -> StatusReadModel:
        return StatusReadModel(
            lock=self._lock,
            runtime_state=self._runtime_state,
            trainer=self._trainer,
            trace_history=self._runtime_persistence.trace_history,
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

    def _build_brain_runtime_dependencies(self) -> BrainRuntimeDependencies:
        return BrainRuntimeDependencies(
            lock=self._lock,
            trainer=self._trainer,
            encoder=self._encoder,
            runtime_state=self._runtime_state,
            brain_config=lambda: self._brain_config,
            runtime_control=lambda: self._runtime_control,
            runtime_sources=lambda: self._runtime_sources,
            delayed_consequence=lambda: self._delayed_consequence,
            autonomy_planner=lambda: self._autonomy_planner,
            source_focus=lambda: self._source_focus,
            interaction_pipeline=lambda: self._interaction_pipeline,
            action_executor=lambda: self._action_executor,
            replay_controller=lambda: self._replay_controller,
            cortex_controller=lambda: self._cortex_controller,
            concept_store=lambda: self._concept_store,
            geometric_curiosity=lambda: self._geometric_curiosity,
            runtime_environment_summary=self._runtime_environment_summary,
            huggingface_runtime_summary_locked=self._huggingface_runtime_summary_locked,
            ingestion_runtime_summary_locked=self._ingestion_runtime_summary_locked,
            multimodal_runtime_summary_locked=self._multimodal_runtime_summary_locked,
            sensory_runtime_summary_locked=self._sensory_runtime_summary_locked,
            living_loop_snapshot_locked=self._living_loop_snapshot_locked,
            maybe_mark_ingestion_warm_locked=self._maybe_mark_ingestion_warm_locked,
            maybe_mark_sensory_warm_locked=self._maybe_mark_sensory_warm_locked,
            observe_runtime_concepts_locked=self._observe_runtime_concepts_locked,
            runtime_concept_callback_locked=self._runtime_concept_callback_locked,
            run_real_sensory_episode_locked=self._run_real_sensory_episode_locked,
            record_brain_event_locked=self._record_brain_event_locked,
            build_brain_source_stream_locked=lambda spec: self._build_brain_source_stream_locked(spec),
            build_sensory_stream_locked=lambda spec: self._build_sensory_stream_locked(spec),
        )

    def _autonomy_planner_or_self(self) -> Any:
        planner = getattr(self, "_autonomy_planner", None)
        return planner if planner is not None else self

    def _build_interaction_pipeline(
        self,
        *,
        recent_query_gaps: Sequence[Mapping[str, Any]] | None = None,
        runtime_episode_traces: Sequence[Mapping[str, Any]] | None = None,
    ) -> InteractionPipeline:
        def apply_provider_response_outcome_calibration_fn(**kwargs: Any) -> bool:
            autonomy = cast(dict[str, Any] | None, self._brain_config.get("autonomy"))
            if autonomy is None:
                return False
            self._autonomy_planner_or_self()._apply_provider_response_outcome_calibration_locked(
                autonomy=autonomy,
                response=kwargs["response"],
                outcome_score=kwargs["outcome_score"],
            )
            return True

        return InteractionPipeline(
            lock=self._lock,
            trainer=self._trainer,
            encoder=self._encoder,
            build_query_result_fn=lambda **kwargs: self._build_query_locked(**kwargs),
            observe_concepts_fn=lambda **kwargs: self._observe_concepts_locked(**kwargs),
            plan_gaps_fn=lambda **kwargs: self._plan_gaps_locked(**kwargs),
            apply_delayed_query_consequence_fn=lambda **kwargs: self._apply_delayed_query_consequence_locked(**kwargs),
            observe_runtime_concepts_fn=lambda **kwargs: self._observe_runtime_concepts_locked(**kwargs),
            runtime_state_mark_mutated_fn=lambda: self._runtime_state.mark_mutated(),
            runtime_state_mutation_summary_fn=lambda: self._runtime_state.mutation_summary(),
            runtime_episode_payload_fn=lambda **kwargs: self._runtime_episode_payload_locked(**kwargs),
            persist_trace_fn=lambda trace: self._runtime_persistence.persist_trace(trace),
            service_state_snapshot_fn=lambda **kwargs: self._service_state_snapshot(**kwargs),
            build_response_fn=lambda **kwargs: self._responder.build_response(**kwargs),
            maybe_auto_action_assist_fn=lambda **kwargs: self._maybe_auto_action_assist_locked(**kwargs),
            response_grounded_outcome_score_fn=lambda **kwargs: self._response_grounded_outcome_score_locked(**kwargs),
            apply_background_source_response_provenance_fn=lambda **kwargs: self._apply_background_source_response_provenance_locked(**kwargs),
            apply_background_source_outcome_calibration_fn=lambda **kwargs: self._apply_background_source_outcome_calibration_locked(**kwargs),
            apply_provider_response_outcome_calibration_fn=apply_provider_response_outcome_calibration_fn,
            learn_from_turn_fn=lambda **kwargs: self._learn_from_turn_locked(**kwargs),
            record_response_consequence_candidate_fn=lambda **kwargs: self._record_response_consequence_candidate_locked(**kwargs),
            recent_query_gaps=recent_query_gaps,
            runtime_episode_traces=runtime_episode_traces,
        )

    def _build_action_executor(self, *, action_history: Sequence[Mapping[str, Any]] | None = None) -> ActionExecutor:
        def apply_provider_outcome_calibration_fn(**kwargs: Any) -> bool:
            autonomy = cast(dict[str, Any] | None, self._brain_config.get("autonomy"))
            if autonomy is None:
                return False
            self._autonomy_planner_or_self()._apply_provider_outcome_calibration_locked(
                autonomy=autonomy,
                query_text=kwargs["query_text"],
                outcome_score=kwargs["outcome_score"],
            )
            return True

        return ActionExecutor(
            lock=self._lock,
            action_root=self._action_root,
            action_history=action_history,
            history_maxlen=24,
            runtime_state_mark_mutated_fn=lambda: self._runtime_state.mark_mutated(),
            runtime_state_mutation_summary_fn=lambda: self._runtime_state.mutation_summary(),
            record_brain_event_fn=lambda event: self._record_brain_event_locked(dict(event)),
            brain_runtime_snapshot_fn=self._brain_runtime_snapshot_locked,
            runtime_trace_export_safe_value_fn=lambda value: self._runtime_trace_export_safe_value(value),
            ensure_cortex_initialized_fn=lambda: self._ensure_cortex_initialized(),
            inject_action_record_into_cortex_fn=lambda thought_loop, record: self._inject_action_record_into_loop(
                thought_loop, record
            ),
            apply_provider_outcome_calibration_fn=apply_provider_outcome_calibration_fn,
        )

    def _build_feedback_applier(self) -> FeedbackApplier:
        return FeedbackApplier(
            lock=self._lock,
            runtime_episode_store=self._interaction_pipeline,
            action_store=self._action_executor,
            runtime_state_mark_mutated_fn=lambda: self._runtime_state.mark_mutated(),
            runtime_state_mutation_summary_fn=lambda: self._runtime_state.mutation_summary(),
            record_brain_event_fn=lambda event: self._record_brain_event_locked(dict(event)),
            brain_runtime_snapshot_fn=self._brain_runtime_snapshot_locked,
            runtime_trace_export_safe_value_fn=lambda value: self._runtime_trace_export_safe_value(value),
        )

    @property
    def _action_history(self) -> deque[dict[str, Any]]:
        return self._action_executor.history

    @_action_history.setter
    def _action_history(self, value: Sequence[Mapping[str, Any]]) -> None:
        self._action_executor.history = value

    @property
    def _trace_history(self) -> deque[dict[str, Any]]:
        return self._runtime_persistence.trace_history

    @_trace_history.setter
    def _trace_history(self, value: Sequence[Mapping[str, Any]]) -> None:
        self._runtime_persistence.trace_history = value

    @property
    def _replay_sample_history(self) -> deque[dict[str, Any]]:
        return self._replay_controller.history

    @_replay_sample_history.setter
    def _replay_sample_history(self, value: Sequence[Mapping[str, Any]]) -> None:
        self._replay_controller.history = value

    @property
    def _brain_recent_query_gaps(self) -> deque[dict[str, Any]]:
        return self._interaction_pipeline.recent_query_gap_history

    @property
    def _runtime_episode_traces(self) -> deque[dict[str, Any]]:
        return self._interaction_pipeline.runtime_episode_trace_history

    # --- Action Executor / Feedback Applier delegation ---
    def action_history(self, limit: int = 20) -> dict[str, Any]:
        return self._action_executor.action_history(limit=limit)

    def action_record(self, action_id: str) -> dict[str, Any] | None:
        return self._action_executor.action_record(action_id)

    def replace_action_record(self, action_id: str, record: Mapping[str, Any]) -> dict[str, Any] | None:
        return self._action_executor.replace_action_record(action_id, record)

    def execute_digital_action(
        self,
        action: Mapping[str, Any],
        *,
        trigger_reason: str | None = None,
        trigger_query_text: str | None = None,
    ) -> dict[str, Any]:
        return self._action_executor.execute_digital_action(
            action,
            trigger_reason=trigger_reason,
            trigger_query_text=trigger_query_text,
        )

    def record_runtime_feedback(self, feedback: Mapping[str, Any]) -> dict[str, Any]:
        return self._feedback_applier.record_runtime_feedback(feedback)

    @staticmethod
    def _normalize_action_text(value: Any) -> str:
        return ActionExecutor._normalize_action_text(value)

    def _normalize_feedback_text(self, value: Any, *, max_chars: int = DEFAULT_RUNTIME_FEEDBACK_MAX_TEXT_CHARS) -> str:
        text = self._normalize_action_text(value)
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + "…"
        return text

    def _runtime_feedback_applied_status(self, verdict: str, *, corrected: bool = False) -> str:
        normalized = self._normalize_action_text(verdict).lower()
        if corrected or normalized == "contradicted":
            return "contradicted"
        if normalized == "verified":
            return "verified"
        return "unverified"

    def _runtime_feedback_provenance(self, status: str) -> str:
        normalized = self._normalize_action_text(status).lower()
        if normalized == "verified":
            return "verified"
        if normalized == "contradicted":
            return "contradicted"
        return "unverified"

    def _sanitize_runtime_feedback_tags(self, tags: Any) -> list[str]:
        if not isinstance(tags, Sequence) or isinstance(tags, (str, bytes)):
            return []
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in list(tags)[: DEFAULT_RUNTIME_FEEDBACK_TAG_LIMIT * 2]:
            tag = self._normalize_feedback_text(raw, max_chars=64).lower()
            if not tag or tag in seen:
                continue
            seen.add(tag)
            cleaned.append(tag)
            if len(cleaned) >= DEFAULT_RUNTIME_FEEDBACK_TAG_LIMIT:
                break
        return cleaned

    def _sanitize_runtime_feedback_evidence(self, evidence: Any) -> list[Any]:
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
            return []
        sanitized: list[Any] = []
        for raw in list(evidence)[:DEFAULT_RUNTIME_FEEDBACK_EVIDENCE_LIMIT]:
            item = self._runtime_trace_export_safe_value(raw)
            if item in ({}, [], None, ""):
                continue
            sanitized.append(item)
        return sanitized

    def _runtime_feedback_corrected_present(self, feedback: Mapping[str, Any]) -> bool:
        if "corrected_output" not in feedback or feedback.get("corrected_output") is None:
            return False
        corrected_output = feedback.get("corrected_output")
        if isinstance(corrected_output, str) and not self._normalize_action_text(corrected_output):
            return False
        return True

    def _normalize_runtime_feedback_request(self, feedback: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(feedback, Mapping):
            raise ValueError("Runtime feedback must be an object.")
        target_type = self._normalize_action_text(feedback.get("target_type", "")).lower()
        if target_type not in {"runtime_episode", "action"}:
            raise ValueError(f"Unsupported runtime feedback target_type: {target_type or '<empty>'}")
        target_id = self._normalize_feedback_text(feedback.get("target_id", ""), max_chars=160)
        if not target_id:
            raise ValueError("Runtime feedback target_id is required.")
        verdict = self._normalize_action_text(feedback.get("verdict", "")).lower()
        if verdict not in {"verified", "contradicted", "unverified"}:
            raise ValueError(f"Unsupported runtime feedback verdict: {verdict or '<empty>'}")
        try:
            confidence = max(0.0, min(1.0, float(feedback.get("confidence", 1.0))))
        except (TypeError, ValueError) as exc:
            raise ValueError("Runtime feedback confidence must be numeric.") from exc
        corrected = self._runtime_feedback_corrected_present(feedback)
        applied_status = self._runtime_feedback_applied_status(verdict, corrected=corrected)
        corrected_output = self._runtime_trace_export_safe_value(feedback.get("corrected_output")) if corrected else None
        return {
            "feedback_id": str(uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_type": target_type,
            "target_id": target_id,
            "verdict": verdict,
            "applied_status": applied_status,
            "confidence": confidence,
            "summary": self._normalize_feedback_text(feedback.get("summary", "")),
            "corrected_output": corrected_output,
            "evidence": self._sanitize_runtime_feedback_evidence(feedback.get("evidence", [])),
            "tags": self._sanitize_runtime_feedback_tags(feedback.get("tags", [])),
            "evaluator_id": self._normalize_feedback_text(feedback.get("evaluator_id", ""), max_chars=160),
        }

    def _normalize_runtime_feedback_entries(self, feedback: Any) -> list[dict[str, Any]]:
        if not isinstance(feedback, Sequence) or isinstance(feedback, (str, bytes)):
            return []
        entries: list[dict[str, Any]] = []
        for raw in list(feedback)[-DEFAULT_RUNTIME_FEEDBACK_HISTORY:]:
            if not isinstance(raw, Mapping):
                continue
            verdict = self._normalize_action_text(raw.get("verdict", "unverified")).lower()
            if verdict not in {"verified", "contradicted", "unverified"}:
                verdict = "unverified"
            corrected = self._runtime_feedback_corrected_present(raw)
            applied_status = self._normalize_action_text(raw.get("applied_status", "")).lower()
            if applied_status not in {"verified", "contradicted", "unverified"}:
                applied_status = self._runtime_feedback_applied_status(verdict, corrected=corrected)
            try:
                confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            corrected_output = self._runtime_trace_export_safe_value(raw.get("corrected_output")) if corrected else None
            entries.append(
                {
                    "feedback_id": self._normalize_feedback_text(raw.get("feedback_id", ""), max_chars=80) or str(uuid4()),
                    "created_at": self._normalize_feedback_text(raw.get("created_at", ""), max_chars=80)
                    or datetime.now(timezone.utc).isoformat(),
                    "target_type": self._normalize_feedback_text(raw.get("target_type", ""), max_chars=32),
                    "target_id": self._normalize_feedback_text(raw.get("target_id", ""), max_chars=160),
                    "verdict": verdict,
                    "applied_status": applied_status,
                    "confidence": confidence,
                    "summary": self._normalize_feedback_text(raw.get("summary", "")),
                    "corrected_output": corrected_output,
                    "evidence": self._sanitize_runtime_feedback_evidence(raw.get("evidence", [])),
                    "tags": self._sanitize_runtime_feedback_tags(raw.get("tags", [])),
                    "evaluator_id": self._normalize_feedback_text(raw.get("evaluator_id", ""), max_chars=160),
                }
            )
        return entries

    def _apply_runtime_feedback_to_target(self, target: dict[str, Any], feedback: Mapping[str, Any]) -> None:
        normalized_feedback = self._normalize_runtime_feedback_request(feedback)
        feedback_entries = list(target.get("feedback") or [])
        feedback_entries.append(normalized_feedback)
        target["feedback"] = self._normalize_runtime_feedback_entries(feedback_entries)
        target["feedback_status"] = normalized_feedback["applied_status"]
        target["feedback_provenance"] = self._runtime_feedback_provenance(normalized_feedback["applied_status"])
        target["provenance"] = target["feedback_provenance"]
        if normalized_feedback["target_type"] == "action":
            verification = target.get("verification") if isinstance(target.get("verification"), Mapping) else {}
            verification = dict(verification)
            if normalized_feedback["applied_status"] == "contradicted":
                verification["status"] = "contradicted"
                verification["success"] = False
                verification["contradiction"] = True
                verification["confidence"] = normalized_feedback["confidence"]
            elif normalized_feedback["applied_status"] == "verified":
                verification["status"] = "verified"
                verification["success"] = True
                verification["contradiction"] = False
                verification["confidence"] = max(float(verification.get("confidence", 0.0) or 0.0), normalized_feedback["confidence"])
            else:
                verification["status"] = "unverified"
                verification["success"] = False
                verification["contradiction"] = False
                verification["confidence"] = normalized_feedback["confidence"]
            verification["summary"] = normalized_feedback["summary"] or str(verification.get("summary", ""))
            verification["provenance"] = normalized_feedback["applied_status"]
            verification["feedback_count"] = int(verification.get("feedback_count", 0) or 0) + 1
            verification["last_feedback_id"] = normalized_feedback["feedback_id"]
            verification["last_feedback_at"] = normalized_feedback["created_at"]
            target["verification"] = verification

    def _normalize_action_record(self, item: Any) -> dict[str, Any] | None:
        return self._action_executor.normalize_action_record(item)

    def _action_request_has_body(self, inputs: Mapping[str, Any]) -> bool:
        return self._action_executor._action_request_has_body(inputs)

    def _api_request_record_matches_explicit_url(self, record: Mapping[str, Any], explicit_url: str) -> bool:
        return self._action_executor._api_request_record_matches_explicit_url(record, explicit_url)

    def _action_query_terms(self, query_text: str) -> tuple[str, ...]:
        return self._action_executor._action_query_terms(query_text)

    def _action_focus_query_text(self, query_text: str) -> str:
        return self._action_executor._action_focus_query_text(query_text)

    def _query_workspace_path_candidate_locked(self, query_text: str) -> str:
        return self._action_executor._query_workspace_path_candidate_locked(query_text)

    def _query_web_url_candidate(self, query_text: str) -> str:
        return self._action_executor._query_web_url_candidate(query_text)

    def _query_api_url_candidate(self, query_text: str) -> str:
        return self._action_executor._query_api_url_candidate(query_text)

    def _action_record_relevance_score_locked(self, record: Mapping[str, Any], query_text: str) -> float:
        return self._action_executor._action_record_relevance_score_locked(record, query_text)

    def _recent_relevant_action_records_locked(
        self,
        query_text: str,
        *,
        statuses: Sequence[str] | None = None,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        return self._action_executor.recent_relevant_action_records(
            query_text,
            statuses=statuses,
            limit=limit,
        )

    def _action_record_to_response_episodes_locked(
        self,
        record: Mapping[str, Any],
        *,
        query_text: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        return self._action_executor.action_record_to_response_episodes(
            record,
            query_text=query_text,
            limit=limit,
        )

    def _augment_query_result_with_action_records_locked(
        self,
        query_result: dict[str, Any],
        *,
        query_text: str,
        records: Sequence[Mapping[str, Any]],
    ) -> int:
        return self._action_executor.augment_query_result_with_action_records(
            query_result,
            query_text=query_text,
            records=records,
        )

    def _contradicted_action_note_locked(self, record: Mapping[str, Any]) -> str:
        return self._action_executor.contradicted_action_note(record)

    def _should_auto_execute_action_locked(
        self,
        *,
        query_text: str,
        query_result: dict[str, Any],
        response: Mapping[str, Any],
    ) -> bool:
        return self._action_executor.should_auto_execute_action(
            query_text=query_text,
            query_result=query_result,
            response=response,
        )

    def _maybe_auto_action_assist_locked(
        self,
        *,
        query_text: str,
        query_result: dict[str, Any],
        response: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        return self._action_executor.maybe_auto_action_assist(
            query_text=query_text,
            query_result=query_result,
            response=response,
        )

    def _action_history_memory_metadata(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return self._action_executor.action_history_memory_metadata(record)

    def _replay_action_history_into_cortex_locked(self) -> None:
        self._action_executor.replay_action_history_into_cortex()

    def _action_loop_summary_locked(self) -> dict[str, Any]:
        return self._action_executor.action_loop_summary()

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

    # --- ReplayDatasetBundle delegation (class-level access compatibility) ---
    # ReplayDatasetBundleMixin._replay_dataset_bundle_hash uses cls._replay_dataset_bundle_canonical_json
    # which requires the method to exist on the class itself, not just on instances.
    @staticmethod
    def _replay_dataset_bundle_canonical_json(value: Any) -> str:
        return ReplayDatasetBundleMixin._replay_dataset_bundle_canonical_json(value)

    @classmethod
    def _replay_dataset_bundle_hash(cls, value: Any) -> str:
        return ReplayDatasetBundleMixin._replay_dataset_bundle_hash(value)

    @staticmethod
    def _replay_dataset_bundle_timestamp(value: Any) -> Any:
        return ReplayDatasetBundleMixin._replay_dataset_bundle_timestamp(value)

    def _replay_dataset_safety_flags(self, *, before: Any, after: Any) -> dict[str, Any]:
        return ReplayDatasetBundleMixin._replay_dataset_safety_flags(self, before=before, after=after)

    # --- RuntimeSources delegation (patch.object compatibility) ---
    @staticmethod
    def _build_source_stream_from_spec(
        spec: dict[str, Any], encoder: Any, window_size: int,
    ) -> Iterator[tuple[str, "torch.Tensor"]]:
        return BrainRuntime._build_source_stream_from_spec(spec, encoder, window_size)

    def _build_brain_source_stream_locked(self, spec: dict[str, Any]) -> Iterator[tuple[str, "torch.Tensor"]]:
        return self._runtime_sources._build_brain_source_stream_locked(spec)

    def _build_sensory_stream_locked(self, spec: dict[str, Any]) -> Iterator[tuple[str, "torch.Tensor"]]:
        return self._runtime_sources._build_sensory_stream_locked(spec)

    @staticmethod
    def _build_sensory_stream_from_spec(
        spec: dict[str, Any],
        *,
        visual_dim: int = 0,
        audio_dim: int = 0,
        device: Any = None,
    ) -> Any:
        return RuntimeSources._build_sensory_stream_from_spec(
            spec, visual_dim=visual_dim, audio_dim=audio_dim, device=device,
        )

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


def _install_module_delegate(
    name: str,
    module_attr: str,
) -> None:
    if name in HECSNServiceManager.__dict__:
        return

    def _delegated(self: HECSNServiceManager, *args: Any, **kwargs: Any) -> Any:
        module = object.__getattribute__(self, module_attr)
        return getattr(module, name)(*args, **kwargs)

    _delegated.__name__ = name
    setattr(HECSNServiceManager, name, _delegated)


def _install_mixin_delegate(
    name: str,
    mixin_cls: type,
) -> None:
    if name in HECSNServiceManager.__dict__:
        return
    descriptor = mixin_cls.__dict__[name]

    def _delegated(self: HECSNServiceManager, *args: Any, **kwargs: Any) -> Any:
        return descriptor.__get__(self, type(self))(*args, **kwargs)

    _delegated.__name__ = name
    setattr(HECSNServiceManager, name, _delegated)


def _install_state_property(
    name: str,
    module_attr: str,
) -> None:
    if name in HECSNServiceManager.__dict__:
        return

    def _get(self: HECSNServiceManager) -> Any:
        module = object.__getattribute__(self, module_attr)
        return getattr(module, name)

    def _set(self: HECSNServiceManager, value: Any) -> None:
        module = object.__getattribute__(self, module_attr)
        setattr(module, name, value)

    setattr(HECSNServiceManager, name, property(_get, _set))


for _name in BRAIN_RUNTIME_STATE_FIELDS:
    _install_state_property(_name, "_brain_runtime")
for _name in RUNTIME_CONTROL_STATE_FIELDS:
    _install_state_property(_name, "_runtime_control")
for _name in CORTEX_CONTROLLER_STATE_FIELDS:
    _install_state_property(_name, "_cortex_controller")
_install_state_property("_thought_loop", "_cortex_controller")
for _name in DELAYED_CONSEQUENCE_STATE_FIELDS:
    _install_state_property(_name, "_delayed_consequence")

for _module_attr, _module_cls in (
    ("_runtime_persistence", RuntimePersistence),
    ("_replay_controller", ReplayController),
    ("_interaction_pipeline", InteractionPipeline),
    ("_status_read_model", StatusReadModel),
    ("_runtime_sources", RuntimeSources),
    ("_brain_runtime", BrainRuntime),
    ("_runtime_control", RuntimeControl),
    ("_runtime_config", RuntimeConfig),
    ("_delayed_consequence", DelayedConsequenceTracker),
    ("_autonomy_planner", AutonomyPlanner),
    ("_source_focus", SourceFocusScorer),
    ("_cortex_controller", CortexController),
):
    for _name in _module_cls.__dict__:
        if _name.startswith("__") or _name in _PUBLIC_DUNDER:
            continue
        if _module_cls is RuntimePersistence and _name in _RUNTIME_PERSISTENCE_INTERNAL_DELEGATE_NAMES:
            continue
        if callable(getattr(_module_cls, _name, None)):
            _install_module_delegate(_name, _module_attr)

for _mixin_cls in (
    StatusRuntimeMixin,
    LivingStatusMixin,
    SensoryPreviewMixin,
    ServiceReportingMixin,
    SensoryRuntimeMixin,
    RuntimePrewarmMixin,
    RuntimeEvidenceMixin,
    InteractionRuntimeMixin,
    ReplayDatasetBundleMixin,
    TerminusAutonomyMixin,
):
    for _name, _value in _mixin_cls.__dict__.items():
        if _name.startswith("__") or _name in HECSNServiceManager.__dict__:
            continue
        if callable(_value) or isinstance(_value, (staticmethod, classmethod)):
            _install_mixin_delegate(_name, _mixin_cls)
