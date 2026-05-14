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
from hecsn.service.runtime_facade import RuntimeFacade
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
            provider_query_family_priority=lambda family_entry: TerminusAutonomyMixin._provider_query_family_priority_locked(self, family_entry),
            provider_topic_family_priority=lambda family_entry: TerminusAutonomyMixin._provider_topic_family_priority_locked(self, family_entry),
        )
        self._runtime_sources = RuntimeSources(
            RuntimeSourcesDependencies(
                brain_config=lambda: self._brain_config,
                brain_source_runtimes=lambda: self._brain_source_runtimes,
                set_brain_source_runtimes=lambda value: setattr(self, "_brain_source_runtimes", list(value)),
                checkpoint_dir=lambda: self._checkpoint_dir,
                checkpoint_path=lambda: self._checkpoint_path,
                encoder=lambda: self._encoder,
                sensory_queue_target_items=lambda: StatusRuntimeMixin._sensory_queue_target_items_locked(self),
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
                execute_digital_action=lambda *args, **kwargs: self._action_executor.execute_digital_action(*args, **kwargs),
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
                cortex_unavailable_snapshot=lambda: self._cortex_controller._cortex_unavailable_snapshot(),
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
                background_focus_terms=lambda **kwargs: self._source_focus._background_focus_terms_locked(**kwargs),
                background_source_utility_entry=self._background_source_utility_entry_locked,
                brain_config=lambda: self._brain_config,
                brain_source_runtimes=lambda: self._brain_source_runtimes,
                brain_source_semantic_match=self._source_focus._brain_source_semantic_match_locked,
                normalize_action_text=self._normalize_action_text,
                normalize_provider_curriculum=self._normalize_provider_curriculum,
                recent_relevant_action_records=self._recent_relevant_action_records_locked,
                record_brain_event=self._record_brain_event_locked,
                runtime_state=self._runtime_state,
                selected_evidence_weight_map=SourceFocusScorer._selected_evidence_weight_map,
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
        self._runtime_facade = RuntimeFacade(self)

    @property
    def runtime_facade(self) -> RuntimeFacade:
        return self._runtime_facade

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
    def action_record(self, action_id: str) -> dict[str, Any] | None:
        return self._action_executor.action_record(action_id)

    def replace_action_record(self, action_id: str, record: Mapping[str, Any]) -> dict[str, Any] | None:
        return self._action_executor.replace_action_record(action_id, record)

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

    def _architecture_summary_impl(self) -> dict[str, Any]:
        """Build the architecture summary under lock (called by read model callback)."""
        return ServiceReportingMixin.architecture_summary(self)

    def _living_loop_status_impl(self) -> dict[str, Any]:
        """Build living loop status under lock (called by read model callback)."""
        return LivingStatusMixin.living_loop_status(self)

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


    # --- Explicit compatibility facade methods (ADR 0003) ---
    # Generated once from the former broad delegate surface so manager.py has
    # named methods instead of import-time dynamic delegate installation.
    def persist_trace(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_persistence.persist_trace(*args, **kwargs)

    def load_persisted_traces(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_persistence.load_persisted_traces(*args, **kwargs)

    def _service_state_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_persistence._service_state_snapshot(*args, **kwargs)

    def _resolve_save_path(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_persistence._resolve_save_path(*args, **kwargs)

    def _persist_trace_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_persistence._persist_trace_locked(*args, **kwargs)

    def _load_persisted_traces_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_persistence._load_persisted_traces_locked(*args, **kwargs)

    def _json_safe(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_persistence._json_safe(*args, **kwargs)

    def _record_brain_event_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_persistence._record_brain_event_locked(*args, **kwargs)

    def _cortex_unavailable_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return self._replay_controller._cortex_unavailable_snapshot(*args, **kwargs)

    def _living_loop_snapshot_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._replay_controller._living_loop_snapshot_locked(*args, **kwargs)

    def _replay_plan_summary(self, *args: Any, **kwargs: Any) -> Any:
        return self._replay_controller._replay_plan_summary(*args, **kwargs)

    def _runtime_feedback_summary_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._replay_controller._runtime_feedback_summary_locked(*args, **kwargs)

    def _runtime_trace_export_safe_value(self, *args: Any, **kwargs: Any) -> Any:
        return RuntimeEvidenceMixin._runtime_trace_export_safe_value(self, *args, **kwargs)

    def _replay_sample_summary_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._replay_controller._replay_sample_summary_locked(*args, **kwargs)

    def _replay_sample_state_counts_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._replay_controller._replay_sample_state_counts_locked(*args, **kwargs)

    def _sample_replay_candidates(self, *args: Any, **kwargs: Any) -> Any:
        return self._replay_controller._sample_replay_candidates(*args, **kwargs)

    def _replay_sample_candidate_payload(self, *args: Any, **kwargs: Any) -> Any:
        return self._replay_controller._replay_sample_candidate_payload(*args, **kwargs)

    def _normalize_replay_sample_record(self, *args: Any, **kwargs: Any) -> Any:
        return self._replay_controller._normalize_replay_sample_record(*args, **kwargs)

    def _normalize_recent_query_gap(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._normalize_recent_query_gap(*args, **kwargs)

    def _normalize_runtime_episode_trace(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._normalize_runtime_episode_trace(*args, **kwargs)

    def load_interaction_state(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline.load_interaction_state(*args, **kwargs)

    def _replace_recent_query_gaps_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._replace_recent_query_gaps_locked(*args, **kwargs)

    def _replace_runtime_episode_traces_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._replace_runtime_episode_traces_locked(*args, **kwargs)

    def recent_query_gaps(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline.recent_query_gaps(*args, **kwargs)

    def record_recent_query_gap(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline.record_recent_query_gap(*args, **kwargs)

    def consume_skip_next_autonomy_for_grounded_query(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline.consume_skip_next_autonomy_for_grounded_query(*args, **kwargs)

    def runtime_episode_traces(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline.runtime_episode_traces(*args, **kwargs)

    def runtime_episode_trace(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline.runtime_episode_trace(*args, **kwargs)

    def append_runtime_episode_trace(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline.append_runtime_episode_trace(*args, **kwargs)

    def replace_runtime_episode_trace(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline.replace_runtime_episode_trace(*args, **kwargs)

    def interaction_state_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline.interaction_state_snapshot(*args, **kwargs)

    def _query_runtime_actual_output(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._query_runtime_actual_output(*args, **kwargs)

    def _query_runtime_verification(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._query_runtime_verification(*args, **kwargs)

    def _feed_runtime_actual_output(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._feed_runtime_actual_output(*args, **kwargs)

    def _feed_runtime_verification(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._feed_runtime_verification(*args, **kwargs)

    def _enrich_query_result(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._enrich_query_result(*args, **kwargs)

    def _build_runtime_episode(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._build_runtime_episode(*args, **kwargs)

    def _finalize_trace(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._finalize_trace(*args, **kwargs)

    def _build_request(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._build_request(*args, **kwargs)

    def _build_prediction(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._build_prediction(*args, **kwargs)

    def _build_action(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._build_action(*args, **kwargs)

    def _build_trace(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._build_trace(*args, **kwargs)

    def _build_feed_request(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._build_feed_request(*args, **kwargs)

    def _build_feed_prediction(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._build_feed_prediction(*args, **kwargs)

    def _build_feed_action(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._build_feed_action(*args, **kwargs)

    def _feed_text_for_request_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._feed_text_for_request_locked(*args, **kwargs)

    def _build_respond_request(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._build_respond_request(*args, **kwargs)

    def _build_respond_action(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._build_respond_action(*args, **kwargs)

    def _build_respond_prediction(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._build_respond_prediction(*args, **kwargs)

    def _respond_runtime_actual_output(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._respond_runtime_actual_output(*args, **kwargs)

    def _respond_runtime_verification(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._respond_runtime_verification(*args, **kwargs)

    def _require_respond_callback(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._require_respond_callback(*args, **kwargs)

    def _require_respond_callbacks(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._require_respond_callbacks(*args, **kwargs)

    def _build_response_payload(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._build_response_payload(*args, **kwargs)

    def _apply_action_assist(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._apply_action_assist(*args, **kwargs)

    def _apply_action_assist_to_action(self, *args: Any, **kwargs: Any) -> Any:
        return self._interaction_pipeline._apply_action_assist_to_action(*args, **kwargs)

    def _replay_dataset_summary_from_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return self._status_read_model._replay_dataset_summary_from_runtime(*args, **kwargs)

    def _runtime_source_configuration_evidence(self, *args: Any, **kwargs: Any) -> Any:
        return self._status_read_model._runtime_source_configuration_evidence(*args, **kwargs)

    def _runtime_truth_contract_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._status_read_model._runtime_truth_contract_locked(*args, **kwargs)

    def _working_set_decision(self, *args: Any, **kwargs: Any) -> Any:
        return self._status_read_model._working_set_decision(*args, **kwargs)

    def _state_norm(self, *args: Any, **kwargs: Any) -> Any:
        return self._status_read_model._state_norm(*args, **kwargs)

    def _last_trace_fields(self, *args: Any, **kwargs: Any) -> Any:
        return self._status_read_model._last_trace_fields(*args, **kwargs)

    def _runtime_mutation_payload(self, *args: Any, **kwargs: Any) -> Any:
        return self._status_read_model._runtime_mutation_payload(*args, **kwargs)

    def _status_snapshot_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._status_read_model._status_snapshot_locked(*args, **kwargs)

    def _terminus_status_snapshot_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._status_read_model._terminus_status_snapshot_locked(*args, **kwargs)

    def _telemetry_snapshot_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._status_read_model._telemetry_snapshot_locked(*args, **kwargs)

    def _read_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return self._status_read_model._read_snapshot(*args, **kwargs)

    def _sensory_media_payload(self, *args: Any, **kwargs: Any) -> Any:
        return self._status_read_model._sensory_media_payload(*args, **kwargs)

    def _read_sensory_previews(self, *args: Any, **kwargs: Any) -> Any:
        return self._status_read_model._read_sensory_previews(*args, **kwargs)

    def cortex_signal_state(self, *args: Any, **kwargs: Any) -> Any:
        return self._status_read_model.cortex_signal_state(*args, **kwargs)

    def _sensory_queue_target_items_locked(self, *args: Any, **kwargs: Any) -> Any:
        return StatusRuntimeMixin._sensory_queue_target_items_locked(self, *args, **kwargs)

    def _source_spec_uses_live_remote(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._source_spec_uses_live_remote(*args, **kwargs)

    def _sensory_spec_uses_live_remote(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._sensory_spec_uses_live_remote(*args, **kwargs)

    def _wrap_remote_stream(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._wrap_remote_stream(*args, **kwargs)

    def _stream_supports_ready_reads(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._stream_supports_ready_reads(*args, **kwargs)

    def _next_stream_item(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._next_stream_item(*args, **kwargs)

    def _runtime_cache_root(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._runtime_cache_root(*args, **kwargs)

    def _runtime_cache_key(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._runtime_cache_key(*args, **kwargs)

    def _brain_runtime_cache_path(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._brain_runtime_cache_path(*args, **kwargs)

    def _sensory_runtime_cache_path(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._sensory_runtime_cache_path(*args, **kwargs)

    def _reconstruct_text_from_windows(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._reconstruct_text_from_windows(*args, **kwargs)

    def _update_brain_runtime_cache_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._update_brain_runtime_cache_locked(*args, **kwargs)

    def _restore_brain_runtime_cache_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._restore_brain_runtime_cache_locked(*args, **kwargs)

    def _serialize_sensory_episode(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._serialize_sensory_episode(*args, **kwargs)

    def _deserialize_sensory_episode(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._deserialize_sensory_episode(*args, **kwargs)

    def _update_sensory_runtime_cache_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._update_sensory_runtime_cache_locked(*args, **kwargs)

    def _restore_sensory_runtime_cache_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._restore_sensory_runtime_cache_locked(*args, **kwargs)

    def _close_runtime_streams(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._close_runtime_streams(*args, **kwargs)

    def _interrupt_brain_sources_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._interrupt_brain_sources_locked(*args, **kwargs)

    def _interrupt_sensory_sources_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._interrupt_sensory_sources_locked(*args, **kwargs)

    def _close_brain_sources_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._close_brain_sources_locked(*args, **kwargs)

    def _close_sensory_sources_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_sources._close_sensory_sources_locked(*args, **kwargs)

    def restore_runtime_state(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime.restore_runtime_state(*args, **kwargs)

    def _ordered_brain_runtime_indices_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._ordered_brain_runtime_indices_locked(*args, **kwargs)

    def _rebuild_brain_sources_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._rebuild_brain_sources_locked(*args, **kwargs)

    def _commit_collected_runtime_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._commit_collected_runtime_locked(*args, **kwargs)

    def _train_chunk_in_sub_batches(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._train_chunk_in_sub_batches(*args, **kwargs)

    def _prefetch_runtime_queue_unlocked(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._prefetch_runtime_queue_unlocked(*args, **kwargs)

    def _prefetch_source_queues_unlocked(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._prefetch_source_queues_unlocked(*args, **kwargs)

    def _collect_chunk_unlocked(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._collect_chunk_unlocked(*args, **kwargs)

    def _source_text_overlap(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._source_text_overlap(*args, **kwargs)

    def _grounded_source_sentences(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._grounded_source_sentences(*args, **kwargs)

    def _dedupe_grounded_topics(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._dedupe_grounded_topics(*args, **kwargs)

    def _grounded_observation_metadata(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._grounded_observation_metadata(*args, **kwargs)

    def _inject_source_observation_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._inject_source_observation_locked(*args, **kwargs)

    def _normalize_background_source_utility_state(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._normalize_background_source_utility_state(*args, **kwargs)

    def _background_source_utility_entry_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._background_source_utility_entry_locked(*args, **kwargs)

    def _background_source_utility_metrics_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._background_source_utility_metrics_locked(*args, **kwargs)

    def _update_background_source_utility_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._update_background_source_utility_locked(*args, **kwargs)

    def _finalize_tick_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._finalize_tick_locked(*args, **kwargs)

    def _brain_tick_idle_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._brain_tick_idle_locked(*args, **kwargs)

    def _run_brain_autonomy_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._run_brain_autonomy_locked(*args, **kwargs)

    def _animation_snapshot_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._animation_snapshot_locked(*args, **kwargs)

    def _brain_runtime_snapshot_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._brain_runtime_snapshot_locked(*args, **kwargs)

    def _brain_persisted_state_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._brain_runtime._brain_persisted_state_locked(*args, **kwargs)

    def _brain_runtime_active_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_control._brain_runtime_active_locked(*args, **kwargs)

    def _assert_manual_tick_allowed_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_control._assert_manual_tick_allowed_locked(*args, **kwargs)

    def _request_brain_stop(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_control._request_brain_stop(*args, **kwargs)

    def _finalize_brain_stop_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_control._finalize_brain_stop_locked(*args, **kwargs)

    def _join_brain_thread(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_control._join_brain_thread(*args, **kwargs)

    def _request_brain_stop_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_control._request_brain_stop_locked(*args, **kwargs)

    def _brain_loop(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_control._brain_loop(*args, **kwargs)

    def _run_brain_tick_once(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_control._run_brain_tick_once(*args, **kwargs)

    def _provider_query_family_priority_locked(self, *args: Any, **kwargs: Any) -> Any:
        return TerminusAutonomyMixin._provider_query_family_priority_locked(self, *args, **kwargs)

    def _provider_topic_family_priority_locked(self, *args: Any, **kwargs: Any) -> Any:
        return TerminusAutonomyMixin._provider_topic_family_priority_locked(self, *args, **kwargs)

    def _normalize_brain_source_spec(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_config._normalize_brain_source_spec(*args, **kwargs)

    def _normalize_sensory_source_spec(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_config._normalize_sensory_source_spec(*args, **kwargs)

    def _normalize_catalog_candidate_spec(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_config._normalize_catalog_candidate_spec(*args, **kwargs)

    def _normalize_autonomy_candidate_spec(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_config._normalize_autonomy_candidate_spec(*args, **kwargs)

    def _normalize_provider_curriculum(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_config._normalize_provider_curriculum(*args, **kwargs)

    def _default_autonomy_candidate_bank(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_config._default_autonomy_candidate_bank(*args, **kwargs)

    def _normalize_autonomy_config(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_config._normalize_autonomy_config(*args, **kwargs)

    def _normalize_brain_config(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_config._normalize_brain_config(*args, **kwargs)

    def _normalize_ingestion_config(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_config._normalize_ingestion_config(*args, **kwargs)

    def _normalize_sensory_config(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime_config._normalize_sensory_config(*args, **kwargs)

    def _background_focus_terms_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._source_focus._background_focus_terms_locked(*args, **kwargs)

    def _brain_source_semantic_match_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._source_focus._brain_source_semantic_match_locked(*args, **kwargs)

    def _selected_evidence_weight_map(self, *args: Any, **kwargs: Any) -> Any:
        return SourceFocusScorer._selected_evidence_weight_map(*args, **kwargs)

    def _consequence_query_terms(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._consequence_query_terms(*args, **kwargs)

    def _query_progress_snapshot_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._query_progress_snapshot_locked(*args, **kwargs)

    def _delayed_consequence_query_examples(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._delayed_consequence_query_examples(*args, **kwargs)

    def _delayed_consequence_match_score_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._delayed_consequence_match_score_locked(*args, **kwargs)

    def _recent_action_contradiction_signal_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._recent_action_contradiction_signal_locked(*args, **kwargs)

    def _delayed_consequence_support_multiplier(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._delayed_consequence_support_multiplier(*args, **kwargs)

    def _delayed_consequence_trajectory_totals(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._delayed_consequence_trajectory_totals(*args, **kwargs)

    def _delayed_consequence_trajectory_balance(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._delayed_consequence_trajectory_balance(*args, **kwargs)

    def _delayed_consequence_trajectory_recent_signal(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._delayed_consequence_trajectory_recent_signal(*args, **kwargs)

    def _delayed_consequence_trajectory_state(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._delayed_consequence_trajectory_state(*args, **kwargs)

    def _delayed_consequence_trajectory_support_multiplier(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._delayed_consequence_trajectory_support_multiplier(*args, **kwargs)

    def _delayed_consequence_family_support_multiplier(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._delayed_consequence_family_support_multiplier(*args, **kwargs)

    def _grounded_family_summary_score(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._grounded_family_summary_score(*args, **kwargs)

    def _update_delayed_consequence_trajectory_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._update_delayed_consequence_trajectory_locked(*args, **kwargs)

    def _delayed_consequence_branch_examples(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._delayed_consequence_branch_examples(*args, **kwargs)

    def _update_delayed_consequence_branch_partition_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._update_delayed_consequence_branch_partition_locked(*args, **kwargs)

    def _delayed_consequence_query_text_overlap_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._delayed_consequence_query_text_overlap_locked(*args, **kwargs)

    def _delayed_consequence_branch_overlap_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._delayed_consequence_branch_overlap_locked(*args, **kwargs)

    def _build_delayed_consequence_split_child_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._build_delayed_consequence_split_child_locked(*args, **kwargs)

    def _split_divergent_delayed_consequence_families_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._split_divergent_delayed_consequence_families_locked(*args, **kwargs)

    def _should_remerge_delayed_consequence_split_group_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._should_remerge_delayed_consequence_split_group_locked(*args, **kwargs)

    def _build_remerged_delayed_consequence_family_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._build_remerged_delayed_consequence_family_locked(*args, **kwargs)

    def _remerge_converged_delayed_consequence_families_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._remerge_converged_delayed_consequence_families_locked(*args, **kwargs)

    def _delayed_consequence_weight_overlap(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._delayed_consequence_weight_overlap(*args, **kwargs)

    def _delayed_consequence_provenance_overlap_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._delayed_consequence_provenance_overlap_locked(*args, **kwargs)

    def _delayed_consequence_aggregation_score_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._delayed_consequence_aggregation_score_locked(*args, **kwargs)

    def _merge_delayed_consequence_records_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._merge_delayed_consequence_records_locked(*args, **kwargs)

    def _upsert_delayed_consequence_record_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._upsert_delayed_consequence_record_locked(*args, **kwargs)

    def _compact_delayed_consequence_records_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._compact_delayed_consequence_records_locked(*args, **kwargs)

    def _cool_delayed_consequence_records_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._cool_delayed_consequence_records_locked(*args, **kwargs)

    def _apply_background_source_delayed_penalty_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._apply_background_source_delayed_penalty_locked(*args, **kwargs)

    def _apply_background_source_forgiveness_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._apply_background_source_forgiveness_locked(*args, **kwargs)

    def _apply_background_source_delayed_consequence_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._apply_background_source_delayed_consequence_locked(*args, **kwargs)

    def _apply_background_source_family_summary_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._apply_background_source_family_summary_locked(*args, **kwargs)

    def _apply_provider_delayed_penalty_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._apply_provider_delayed_penalty_locked(*args, **kwargs)

    def _apply_provider_forgiveness_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._apply_provider_forgiveness_locked(*args, **kwargs)

    def _apply_provider_delayed_consequence_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._apply_provider_delayed_consequence_locked(*args, **kwargs)

    def _apply_provider_family_summary_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._apply_provider_family_summary_locked(*args, **kwargs)

    def _apply_delayed_query_consequence_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._apply_delayed_query_consequence_locked(*args, **kwargs)

    def _record_response_consequence_candidate_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._record_response_consequence_candidate_locked(*args, **kwargs)

    def _apply_background_source_response_provenance_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._apply_background_source_response_provenance_locked(*args, **kwargs)

    def _apply_background_source_outcome_calibration_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._apply_background_source_outcome_calibration_locked(*args, **kwargs)

    def _delayed_consequence_summary_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._delayed_consequence_summary_locked(*args, **kwargs)

    def _normalize_delayed_consequence_record(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence._normalize_delayed_consequence_record(*args, **kwargs)

    def restore_state(self, *args: Any, **kwargs: Any) -> Any:
        return self._delayed_consequence.restore_state(*args, **kwargs)

    def _focus_gap_terms_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._source_focus._focus_gap_terms_locked(*args, **kwargs)

    def _brain_source_memory_metadata(self, *args: Any, **kwargs: Any) -> Any:
        return self._source_focus._brain_source_memory_metadata(*args, **kwargs)

    def _brain_source_topic_terms(self, *args: Any, **kwargs: Any) -> Any:
        return self._source_focus._brain_source_topic_terms(*args, **kwargs)

    def _brain_source_selection_score_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._source_focus._brain_source_selection_score_locked(*args, **kwargs)

    def _background_focus_overlap_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._source_focus._background_focus_overlap_locked(*args, **kwargs)

    def _response_grounded_outcome_score_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._source_focus._response_grounded_outcome_score_locked(*args, **kwargs)

    def _normalize_cortex_query_hint(self, *args: Any, **kwargs: Any) -> Any:
        return self._cortex_controller._normalize_cortex_query_hint(*args, **kwargs)

    def _remember_cortex_query_hint_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._cortex_controller._remember_cortex_query_hint_locked(*args, **kwargs)

    def _consume_cortex_query_hint_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._cortex_controller._consume_cortex_query_hint_locked(*args, **kwargs)

    def _cortex_active(self, *args: Any, **kwargs: Any) -> Any:
        return self._cortex_controller._cortex_active(*args, **kwargs)

    def _cortex_factories_are_mocked(self, *args: Any, **kwargs: Any) -> Any:
        return self._cortex_controller._cortex_factories_are_mocked(*args, **kwargs)

    def _inject_action_record_into_loop(self, *args: Any, **kwargs: Any) -> Any:
        return self._cortex_controller._inject_action_record_into_loop(*args, **kwargs)

    def _build_cortex_thought_loop(self, *args: Any, **kwargs: Any) -> Any:
        return self._cortex_controller._build_cortex_thought_loop(*args, **kwargs)

    def _start_cortex_initialization(self, *args: Any, **kwargs: Any) -> Any:
        return self._cortex_controller._start_cortex_initialization(*args, **kwargs)

    def _ensure_cortex_initialized(self, *args: Any, **kwargs: Any) -> Any:
        return self._cortex_controller._ensure_cortex_initialized(*args, **kwargs)

    def _request_cortex_sleep_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._cortex_controller._request_cortex_sleep_locked(*args, **kwargs)

    def _handle_cortex_sleep_intent_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._cortex_controller._handle_cortex_sleep_intent_locked(*args, **kwargs)

    def _on_cortex_sleep_cycle(self, *args: Any, **kwargs: Any) -> Any:
        return self._cortex_controller._on_cortex_sleep_cycle(*args, **kwargs)

    def _cortex_action_query_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._cortex_controller._cortex_action_query_locked(*args, **kwargs)

    def _filter_cortex_action_records_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._cortex_controller._filter_cortex_action_records_locked(*args, **kwargs)

    def _cortex_action_type(self, *args: Any, **kwargs: Any) -> Any:
        return self._cortex_controller._cortex_action_type(*args, **kwargs)

    def _cortex_action_trigger_reason(self, *args: Any, **kwargs: Any) -> Any:
        return self._cortex_controller._cortex_action_trigger_reason(*args, **kwargs)

    def _handle_cortex_action_intent_locked(self, *args: Any, **kwargs: Any) -> Any:
        return self._cortex_controller._handle_cortex_action_intent_locked(*args, **kwargs)

    def _on_cortex_thought(self, *args: Any, **kwargs: Any) -> Any:
        return self._cortex_controller._on_cortex_thought(*args, **kwargs)

    def _runtime_environment_summary(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(StatusRuntimeMixin, '_runtime_environment_summary', self, *args, **kwargs)

    def _multimodal_runtime_summary_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(StatusRuntimeMixin, '_multimodal_runtime_summary_locked', self, *args, **kwargs)

    def _ingestion_ready_source_count_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(StatusRuntimeMixin, '_ingestion_ready_source_count_locked', self, *args, **kwargs)

    def _ingestion_full_queue_source_count_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(StatusRuntimeMixin, '_ingestion_full_queue_source_count_locked', self, *args, **kwargs)

    def _ingestion_startup_state_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(StatusRuntimeMixin, '_ingestion_startup_state_locked', self, *args, **kwargs)

    def _maybe_mark_ingestion_warm_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(StatusRuntimeMixin, '_maybe_mark_ingestion_warm_locked', self, *args, **kwargs)

    def _ingestion_runtime_summary_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(StatusRuntimeMixin, '_ingestion_runtime_summary_locked', self, *args, **kwargs)

    def _sensory_ready_source_count_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(StatusRuntimeMixin, '_sensory_ready_source_count_locked', self, *args, **kwargs)

    def _sensory_full_queue_source_count_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(StatusRuntimeMixin, '_sensory_full_queue_source_count_locked', self, *args, **kwargs)

    def _sensory_startup_state_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(StatusRuntimeMixin, '_sensory_startup_state_locked', self, *args, **kwargs)

    def _maybe_mark_sensory_warm_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(StatusRuntimeMixin, '_maybe_mark_sensory_warm_locked', self, *args, **kwargs)

    def _sensory_runtime_summary_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(StatusRuntimeMixin, '_sensory_runtime_summary_locked', self, *args, **kwargs)

    def _huggingface_runtime_summary_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(StatusRuntimeMixin, '_huggingface_runtime_summary_locked', self, *args, **kwargs)

    def _cross_modal_confidence_means_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(SensoryRuntimeMixin, '_cross_modal_confidence_means_locked', self, *args, **kwargs)

    def _sensory_runtime_modalities(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(SensoryRuntimeMixin, '_sensory_runtime_modalities', self, *args, **kwargs)

    def _sensory_focus_terms_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(SensoryRuntimeMixin, '_sensory_focus_terms_locked', self, *args, **kwargs)

    def _sensory_source_topic_terms(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(SensoryRuntimeMixin, '_sensory_source_topic_terms', self, *args, **kwargs)

    def _sensory_episode_terms(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(SensoryRuntimeMixin, '_sensory_episode_terms', self, *args, **kwargs)

    def _sensory_episode_semantic_match_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(SensoryRuntimeMixin, '_sensory_episode_semantic_match_locked', self, *args, **kwargs)

    def _sensory_semantic_match_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(SensoryRuntimeMixin, '_sensory_semantic_match_locked', self, *args, **kwargs)

    def _sensory_selection_score_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(SensoryRuntimeMixin, '_sensory_selection_score_locked', self, *args, **kwargs)

    def _select_sensory_runtime_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(SensoryRuntimeMixin, '_select_sensory_runtime_locked', self, *args, **kwargs)

    def _sensory_item_retrieval_config_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(SensoryRuntimeMixin, '_sensory_item_retrieval_config_locked', self, *args, **kwargs)

    def _prefetch_sensory_runtime_unlocked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(SensoryRuntimeMixin, '_prefetch_sensory_runtime_unlocked', self, *args, **kwargs)

    def _prefetch_sensory_queues_unlocked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(SensoryRuntimeMixin, '_prefetch_sensory_queues_unlocked', self, *args, **kwargs)

    def _commit_prefetched_sensory_runtime_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(SensoryRuntimeMixin, '_commit_prefetched_sensory_runtime_locked', self, *args, **kwargs)

    def _next_sensory_episode_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(SensoryRuntimeMixin, '_next_sensory_episode_locked', self, *args, **kwargs)

    def _sensory_modality_need_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(SensoryRuntimeMixin, '_sensory_modality_need_locked', self, *args, **kwargs)

    def _sensory_window_budget_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(SensoryRuntimeMixin, '_sensory_window_budget_locked', self, *args, **kwargs)

    def _inject_sensory_observation_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(SensoryRuntimeMixin, '_inject_sensory_observation_locked', self, *args, **kwargs)

    def _record_sensory_preview_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(SensoryRuntimeMixin, '_record_sensory_preview_locked', self, *args, **kwargs)

    def _run_real_sensory_episode_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(SensoryRuntimeMixin, '_run_real_sensory_episode_locked', self, *args, **kwargs)

    def _request_active_execution_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_request_active_execution_locked', self, *args, **kwargs)

    def _release_active_execution_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_release_active_execution_locked', self, *args, **kwargs)

    def _request_active_execution(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_request_active_execution', self, *args, **kwargs)

    def _release_active_execution(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_release_active_execution', self, *args, **kwargs)

    def _wait_for_remote_prewarm_clearance(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_wait_for_remote_prewarm_clearance', self, *args, **kwargs)

    def _remote_warm_promotion_text_needed_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_remote_warm_promotion_text_needed_locked', self, *args, **kwargs)

    def _remote_warm_promotion_sensory_needed_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_remote_warm_promotion_sensory_needed_locked', self, *args, **kwargs)

    def _request_remote_warm_promotion_stop(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_request_remote_warm_promotion_stop', self, *args, **kwargs)

    def _join_remote_warm_promotion_thread(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_join_remote_warm_promotion_thread', self, *args, **kwargs)

    def _record_remote_warm_promotion_completed_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_record_remote_warm_promotion_completed_locked', self, *args, **kwargs)

    def _start_remote_warm_promotion_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_start_remote_warm_promotion_locked', self, *args, **kwargs)

    def _remaining_budget_seconds(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_remaining_budget_seconds', self, *args, **kwargs)

    def _run_budgeted_call(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_run_budgeted_call', self, *args, **kwargs)

    def _remote_text_bootstrap_candidates_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_remote_text_bootstrap_candidates_locked', self, *args, **kwargs)

    def _fetch_remote_text_bootstrap_rows(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_fetch_remote_text_bootstrap_rows', self, *args, **kwargs)

    def _apply_remote_text_bootstrap_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_apply_remote_text_bootstrap_locked', self, *args, **kwargs)

    def _remote_sensory_bootstrap_candidates_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_remote_sensory_bootstrap_candidates_locked', self, *args, **kwargs)

    def _fetch_remote_sensory_bootstrap_episodes(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_fetch_remote_sensory_bootstrap_episodes', self, *args, **kwargs)

    def _apply_remote_sensory_bootstrap_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_apply_remote_sensory_bootstrap_locked', self, *args, **kwargs)

    def _promote_ready_remote_brain_items_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_promote_ready_remote_brain_items_locked', self, *args, **kwargs)

    def _promote_ready_remote_sensory_items_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_promote_ready_remote_sensory_items_locked', self, *args, **kwargs)

    def _remote_warm_promotion_loop(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_remote_warm_promotion_loop', self, *args, **kwargs)

    def _request_ingestion_prewarm_stop(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_request_ingestion_prewarm_stop', self, *args, **kwargs)

    def _join_ingestion_prewarm_thread(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_join_ingestion_prewarm_thread', self, *args, **kwargs)

    def _start_ingestion_prewarm_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_start_ingestion_prewarm_locked', self, *args, **kwargs)

    def _apply_detached_brain_prewarm_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_apply_detached_brain_prewarm_locked', self, *args, **kwargs)

    def _apply_detached_sensory_prewarm_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_apply_detached_sensory_prewarm_locked', self, *args, **kwargs)

    def _ingestion_prewarm_loop(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimePrewarmMixin, '_ingestion_prewarm_loop', self, *args, **kwargs)

    def _replay_dataset_count_map(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_replay_dataset_count_map', self, *args, **kwargs)

    def _replay_dataset_latest_history_timestamp_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_replay_dataset_latest_history_timestamp_locked', self, *args, **kwargs)

    def _replay_dataset_summary_from_payload(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_replay_dataset_summary_from_payload', self, *args, **kwargs)

    def _replay_dataset_preview_payload_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_replay_dataset_preview_payload_locked', self, *args, **kwargs)

    def _replay_dataset_preview_summary_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_replay_dataset_preview_summary_locked', self, *args, **kwargs)

    def _append_runtime_episode_trace_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_append_runtime_episode_trace_locked', self, *args, **kwargs)

    def _runtime_episode_trace_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_runtime_episode_trace_locked', self, *args, **kwargs)

    def _replace_runtime_episode_trace_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_replace_runtime_episode_trace_locked', self, *args, **kwargs)

    def _normalize_runtime_trace_export_filter(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_normalize_runtime_trace_export_filter', self, *args, **kwargs)

    def _runtime_trace_export_endpoint(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_runtime_trace_export_endpoint', self, *args, **kwargs)

    def _runtime_trace_state_by_trace_id_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_runtime_trace_state_by_trace_id_locked', self, *args, **kwargs)

    def _runtime_trace_export_example_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_runtime_trace_export_example_locked', self, *args, **kwargs)

    def _replay_dataset_candidates_by_target(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_replay_dataset_candidates_by_target', self, *args, **kwargs)

    def _replay_dataset_sample_links_by_target_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_replay_dataset_sample_links_by_target_locked', self, *args, **kwargs)

    def _replay_dataset_verification_label(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_replay_dataset_verification_label', self, *args, **kwargs)

    def _replay_dataset_output_or_none(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_replay_dataset_output_or_none', self, *args, **kwargs)

    def _replay_dataset_item_from_trace_example(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_replay_dataset_item_from_trace_example', self, *args, **kwargs)

    def _runtime_trace_export_policy_decision_summary(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_runtime_trace_export_policy_decision_summary', self, *args, **kwargs)

    def _runtime_trace_export_key_is_safe(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_runtime_trace_export_key_is_safe', self, *args, **kwargs)

    def _runtime_trace_export_int(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_runtime_trace_export_int', self, *args, **kwargs)

    def _runtime_feedback_summary_from_targets(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_runtime_feedback_summary_from_targets', self, *args, **kwargs)

    def _runtime_episode_payload_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(RuntimeEvidenceMixin, '_runtime_episode_payload_locked', self, *args, **kwargs)

    def _build_query_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(InteractionRuntimeMixin, '_build_query_locked', self, *args, **kwargs)

    def _learn_from_turn_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(InteractionRuntimeMixin, '_learn_from_turn_locked', self, *args, **kwargs)

    def _observe_concepts_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(InteractionRuntimeMixin, '_observe_concepts_locked', self, *args, **kwargs)

    def _runtime_concept_callback_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(InteractionRuntimeMixin, '_runtime_concept_callback_locked', self, *args, **kwargs)

    def _observe_runtime_concepts_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(InteractionRuntimeMixin, '_observe_runtime_concepts_locked', self, *args, **kwargs)

    def _plan_gaps_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(InteractionRuntimeMixin, '_plan_gaps_locked', self, *args, **kwargs)

    def _record_recent_query_gap_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(InteractionRuntimeMixin, '_record_recent_query_gap_locked', self, *args, **kwargs)

    def _replay_dataset_bundle_fraction(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(ReplayDatasetBundleMixin, '_replay_dataset_bundle_fraction', self, *args, **kwargs)

    def _replay_dataset_bundle_terms(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(ReplayDatasetBundleMixin, '_replay_dataset_bundle_terms', self, *args, **kwargs)

    def _replay_dataset_bundle_item_fingerprint(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(ReplayDatasetBundleMixin, '_replay_dataset_bundle_item_fingerprint', self, *args, **kwargs)

    def _replay_dataset_bundle_exclusion_reasons(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(ReplayDatasetBundleMixin, '_replay_dataset_bundle_exclusion_reasons', self, *args, **kwargs)

    def _replay_dataset_bundle_filter_items(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(ReplayDatasetBundleMixin, '_replay_dataset_bundle_filter_items', self, *args, **kwargs)

    def _replay_dataset_bundle_split_items(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(ReplayDatasetBundleMixin, '_replay_dataset_bundle_split_items', self, *args, **kwargs)

    def _replay_dataset_bundle_split_summary(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(ReplayDatasetBundleMixin, '_replay_dataset_bundle_split_summary', self, *args, **kwargs)

    def _replay_dataset_bundle_payload_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(ReplayDatasetBundleMixin, '_replay_dataset_bundle_payload_locked', self, *args, **kwargs)

    def _autonomy_focus_plan_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_autonomy_focus_plan_locked', self, *args, **kwargs)

    def _merge_focus_plans_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_merge_focus_plans_locked', self, *args, **kwargs)

    def _recent_query_focus_plan_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_recent_query_focus_plan_locked', self, *args, **kwargs)

    def _autonomy_candidate_specs_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_autonomy_candidate_specs_locked', self, *args, **kwargs)

    def _curiosity_ready_weak_concept_count_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_curiosity_ready_weak_concept_count_locked', self, *args, **kwargs)

    def _provider_curriculum_focus_terms_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_provider_curriculum_focus_terms_locked', self, *args, **kwargs)

    def _autonomy_focus_pressure_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_autonomy_focus_pressure_locked', self, *args, **kwargs)

    def _provider_topic_family_match_score_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_provider_topic_family_match_score_locked', self, *args, **kwargs)

    def _provider_topic_family_details_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_provider_topic_family_details_locked', self, *args, **kwargs)

    def _provider_query_family_match_score_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_provider_query_family_match_score_locked', self, *args, **kwargs)

    def _provider_query_family_details_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_provider_query_family_details_locked', self, *args, **kwargs)

    def _provider_curriculum_priority_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_provider_curriculum_priority_locked', self, *args, **kwargs)

    def _provider_curriculum_snapshot_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_provider_curriculum_snapshot_locked', self, *args, **kwargs)

    def _provider_curriculum_signal_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_provider_curriculum_signal_locked', self, *args, **kwargs)

    def _adaptive_autonomy_settings_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_adaptive_autonomy_settings_locked', self, *args, **kwargs)

    def _apply_provider_response_outcome_calibration_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_apply_provider_response_outcome_calibration_locked', self, *args, **kwargs)

    def _apply_provider_outcome_calibration_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_apply_provider_outcome_calibration_locked', self, *args, **kwargs)

    def _apply_provider_curriculum_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_apply_provider_curriculum_locked', self, *args, **kwargs)

    def _update_provider_curriculum_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_update_provider_curriculum_locked', self, *args, **kwargs)

    def _candidate_pool_size_hint(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_candidate_pool_size_hint', self, *args, **kwargs)

    def _autonomy_shortlist_settings_locked(self, *args: Any, **kwargs: Any) -> Any:
        return _call_mixin_delegate(TerminusAutonomyMixin, '_autonomy_shortlist_settings_locked', self, *args, **kwargs)


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

def _call_mixin_delegate(mixin_cls: type, method_name: str, instance: HECSNServiceManager, *args: Any, **kwargs: Any) -> Any:
    descriptor = mixin_cls.__dict__[method_name]
    return descriptor.__get__(instance, type(instance))(*args, **kwargs)

