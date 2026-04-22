from __future__ import annotations

import base64
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from threading import Event, RLock, Thread
import time
from typing import Any, Iterator, Mapping, Sequence, cast
from uuid import uuid4

import torch

from hecsn.config.presets import get_autonomy_acquisition_preset
from hecsn.config.model_config import HECSNConfig
from hecsn.data.corpus_loader import StreamingCorpusLoader, huggingface_token_from_env
from hecsn.data.pattern_loader import labeled_pattern_stream
from hecsn.gap_planner import plan_query_gaps
from hecsn.interaction import EvidenceResponder
from hecsn.reporting.io import write_json_file
from hecsn.semantics import ConceptStore, GeometricCuriosityController
from hecsn.semantics.grounding_text import salient_query_terms
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

from hecsn.service.terminus_presets import TERMINUS_QUICK_START_PRESETS
from hecsn.service.terminus_sensory import SensoryEpisode, build_sensory_stream


from hecsn.service.terminus_autonomy import _canonical_provider_term  # noqa: E402


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
    last_item_semantic_match: float = 0.0
    last_item_candidates_considered: int = 0
    last_item_retrieval_lookahead: int = 0

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
    Autonomy/curriculum logic is in TerminusAutonomyMixin.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        trace_history_limit: int = 200,
        trace_dir: str | Path | None = None,
    ) -> None:
        self._lock = RLock()
        self._checkpoint_path = Path(checkpoint_path)
        self._checkpoint_dir = self._checkpoint_path.parent if self._checkpoint_path.parent != Path("") else Path("checkpoints")
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
        self._brain_skip_next_autonomy_for_grounded_query = False
        self._brain_last_acquisition_summary: dict[str, Any] | None = None
        self._brain_last_acquisition_token_count = int(self._trainer.token_count)
        self._brain_running_since: str | None = None
        self._brain_last_tick_completed_at: str | None = None
        self._brain_last_tick_duration_ms: float | None = None
        self._brain_last_tick_token_delta = 0
        self._brain_last_work_at: str | None = None
        self._last_curriculum_injection_time = 0.0
        self._last_curriculum_injection_token_count = int(self._trainer.token_count)
        self._last_real_sensory_episode_time = 0.0
        self._last_real_sensory_episode_token_count = int(self._trainer.token_count)
        self._real_sensory_last_error: str | None = None
        self._last_sensory_focus_terms: tuple[str, ...] = ()
        self._sensory_preview_history: deque[dict[str, Any]] = deque(maxlen=8)
        self._brain_thread: Thread | None = None
        self._brain_stop_event: Event | None = None
        self._brain_running = False
        self._hint_sensory_episodes_completed = 0
        self._hint_visual_accepted = 0
        self._hint_audio_accepted = 0
        self._real_sensory_episodes_completed = 0
        self._real_visual_accepted = 0
        self._real_audio_accepted = 0
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
            )
            self._cortex_available = True
            _cortex_logger.info("Cortex module initialised (%s, embedder=%s)", cortex.model, type(embedder).__name__)

        except RuntimeError as exc:
            # API key missing or NIM unreachable — cortex disabled
            _cortex_logger.warning("Cortex disabled: %s", exc)
        except Exception as exc:
            _cortex_logger.info("Cortex module unavailable: %s", exc)

    def status(self) -> dict[str, Any]:
        # Non-blocking: return cached data when brain loop holds the lock
        acquired = self._lock.acquire(timeout=0.15)
        if not acquired:
            cached = getattr(self, "_cached_status", None)
            if cached is not None:
                return cached
            self._lock.acquire()
        try:
            last_trace = self._trace_history[0] if self._trace_history else None
            result = {
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
            learning = self._learn_from_turn_locked(query_text=query_text, response=response, learn_mode=learn_mode)
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
        curriculum = self._brain_config.get("curriculum") or {}
        sensory = self._brain_config.get("sensory") or {}
        cross_modal_enabled = bool(getattr(self._trainer.config, "enable_cross_modal", False))
        hint_enabled = bool(curriculum.get("enabled", False)) and cross_modal_enabled
        real_enabled = bool(sensory.get("enabled", False)) and cross_modal_enabled
        visual_confidence, audio_confidence = self._cross_modal_confidence_means_locked()
        mode_parts: list[str] = []
        if real_enabled:
            mode_parts.append("real_hf_sensory")
        if hint_enabled:
            mode_parts.append("curriculum_hints")
        total_visual = int(self._hint_visual_accepted + self._real_visual_accepted)
        total_audio = int(self._hint_audio_accepted + self._real_audio_accepted)
        next_source_name = None
        if self._sensory_source_runtimes:
            next_source_name = self._sensory_source_runtimes[
                self._sensory_source_index % len(self._sensory_source_runtimes)
            ].name
        return {
            "enabled": bool(mode_parts),
            "mode": "+".join(mode_parts) if mode_parts else "disabled",
            "episodes_completed": int(self._hint_sensory_episodes_completed + self._real_sensory_episodes_completed),
            "hint_episodes_completed": int(self._hint_sensory_episodes_completed),
            "real_episodes_completed": int(self._real_sensory_episodes_completed),
            "tokens_since_hint_episode": int(
                max(0, int(self._trainer.token_count) - int(self._last_curriculum_injection_token_count))
            ),
            "tokens_since_real_episode": int(
                max(0, int(self._trainer.token_count) - int(self._last_real_sensory_episode_token_count))
            ),
            "hint_episode_interval": int(curriculum.get("trigger_interval_tokens", 1024)) if curriculum else 0,
            "real_episode_interval": int(sensory.get("episode_interval_tokens", 2048)) if sensory else 0,
            "items_per_real_episode": int(sensory.get("items_per_episode", 1)) if sensory else 0,
            "base_windows_per_item": int(sensory.get("base_windows_per_item", 0)) if sensory else 0,
            "max_windows_per_item": int(sensory.get("max_windows_per_item", 0)) if sensory else 0,
            "confidence_window_gain": float(sensory.get("confidence_window_gain", 0.0)) if sensory else 0.0,
            "semantic_window_gain": float(sensory.get("semantic_window_gain", 0.0)) if sensory else 0.0,
            "item_retrieval_lookahead": int(sensory.get("item_retrieval_lookahead", 1)) if sensory else 0,
            "item_retrieval_semantic_weight": float(sensory.get("item_retrieval_semantic_weight", 0.0)) if sensory else 0.0,
            "observation_salience": float(sensory.get("observation_salience", 0.0)) if sensory else 0.0,
            "cross_modal_visual_accepted": total_visual,
            "cross_modal_audio_accepted": total_audio,
            "hint_cross_modal_visual_accepted": int(self._hint_visual_accepted),
            "hint_cross_modal_audio_accepted": int(self._hint_audio_accepted),
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

    def _huggingface_runtime_summary_locked(self) -> dict[str, Any]:
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
        }

    def terminus_status(self) -> dict[str, Any]:
        # Non-blocking: return cached data when brain loop holds the lock
        acquired = self._lock.acquire(timeout=0.15)
        if not acquired:
            cached = getattr(self, "_cached_terminus_status", None)
            if cached is not None:
                return cached
            self._lock.acquire()
        try:
            result = {
                "terminus_runtime": self._brain_runtime_snapshot_locked(),
                "dirty_state": bool(self._dirty_state),
                "state_revision": int(self._state_revision),
                "token_count": int(self._trainer.token_count),
                "multimodal": self._multimodal_runtime_summary_locked(),
            }
            self._cached_terminus_status = result
            return result
        finally:
            self._lock.release()

    def configure_terminus(
        self,
        *,
        source_bank: list[dict[str, Any]],
        tick_tokens: int = DEFAULT_BRAIN_TICK_TOKENS,
        sleep_interval_seconds: float = DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS,
        repeat_sources: bool = True,
        autonomy: dict[str, Any] | None = None,
        curriculum: dict[str, Any] | None = None,
        sensory: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        thread = self._request_brain_stop()
        self._join_brain_thread(thread)
        with self._lock:
            self._brain_config = self._normalize_brain_config(
                {
                    "source_bank": source_bank,
                    "tick_tokens": tick_tokens,
                    "sleep_interval_seconds": sleep_interval_seconds,
                    "repeat_sources": repeat_sources,
                    "autonomy": autonomy,
                    "curriculum": curriculum,
                    "sensory": sensory,
                }
            )
            self._brain_last_error = None
            self._last_curriculum_injection_time = 0.0
            self._last_curriculum_injection_token_count = int(self._trainer.token_count)
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
            self._mark_mutated()
            return {
                "terminus_runtime": self._brain_runtime_snapshot_locked(),
                "dirty_state": bool(self._dirty_state),
                "state_revision": int(self._state_revision),
                "token_count": int(self._trainer.token_count),
            }

    def start_terminus(self) -> dict[str, Any]:
        with self._lock:
            if not self._brain_config.get("source_bank"):
                raise ValueError("Terminus runtime has no configured source_bank")
            if self._brain_running and self._brain_thread is not None and self._brain_thread.is_alive():
                return {
                    "terminus_runtime": self._brain_runtime_snapshot_locked(),
                    "dirty_state": bool(self._dirty_state),
                    "state_revision": int(self._state_revision),
                    "token_count": int(self._trainer.token_count),
                }
            self._brain_stop_event = Event()
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

        # Join ThoughtLoop thread outside all locks
        if thought_loop is not None:
            try:
                thought_loop.stop(timeout=3.0)
            except Exception:
                pass

        with self._lock:
            self._record_brain_event_locked(
                {
                    "type": "stopped",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": "manual",
                }
            )
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
            if self._brain_running and self._brain_thread is not None and self._brain_thread.is_alive():
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
            autonomy=None,
            curriculum=config.get("curriculum"),
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

    def cortex_ask(self, query: str) -> dict[str, Any]:
        """Submit a question to the cortex and return immediately.

        The cortex will answer asynchronously in its next deliberation cycle.
        Returns acknowledgement with queue depth.
        """
        if self._thought_loop is None:
            return {"accepted": False, "reason": "cortex_unavailable"}
        self._thought_loop.submit_query(query)
        return {"accepted": True, "query": query}

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
        self._join_brain_thread(thread)

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
            curriculum = self._brain_config.get("curriculum") or {}
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
                "id": "autonomy_curriculum",
                "name": "Active Exploration + Curriculum",
                "enabled": bool(curriculum.get("enabled", False)) or bool(sensory.get("enabled", False)),
                "type": "autonomy",
                "params": {
                    "curriculum_enabled": bool(curriculum.get("enabled", False)),
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
        with self._lock:
            for _ in range(max(1, int(steps))):
                summary = self._brain_tick_locked()
                tick_summaries.append(summary)
                if not bool(summary.get("did_work", False)):
                    break
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
            return {
                "source_bank": [],
                "tick_tokens": DEFAULT_BRAIN_TICK_TOKENS,
                "sleep_interval_seconds": DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS,
                "repeat_sources": True,
                "autonomy": None,
                "curriculum": None,
                "sensory": None,
            }
        if not isinstance(config, dict):
            raise ValueError("Terminus runtime configuration must be an object")
        source_bank = [
            self._normalize_brain_source_spec(item, index)
            for index, item in enumerate(list(config.get("source_bank") or []))
        ]
        normalized = {
            "source_bank": source_bank,
            "tick_tokens": max(1, int(config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS))),
            "sleep_interval_seconds": max(
                0.01,
                float(config.get("sleep_interval_seconds", DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS)),
            ),
            "repeat_sources": bool(config.get("repeat_sources", True)),
            "autonomy": self._normalize_autonomy_config(config.get("autonomy")),
            "curriculum": self._normalize_curriculum_config(config.get("curriculum")),
            "sensory": self._normalize_sensory_config(config.get("sensory")),
        }
        return normalized

    @staticmethod
    def _normalize_curriculum_config(config: Any) -> dict[str, Any] | None:
        if config is None or not isinstance(config, dict):
            return None
        if not config.get("enabled"):
            return None
        return {
            "enabled": True,
            "topics_per_cycle": max(1, int(config.get("topics_per_cycle", 3))),
            "episode_length_tokens": max(64, int(config.get("episode_length_tokens", 256))),
            "diversity_threshold": max(0.0, min(1.0, float(config.get("diversity_threshold", 0.7)))),
            "trigger_interval_tokens": max(256, int(config.get("trigger_interval_tokens", 1024))),
            "cooldown_seconds": max(5.0, float(config.get("cooldown_seconds", 30.0))),
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
        return {
            "enabled": True,
            "source_bank": source_bank,
            "episode_interval_tokens": max(256, int(config.get("episode_interval_tokens", 1536))),
            "items_per_episode": max(1, int(config.get("items_per_episode", 2))),
            "base_windows_per_item": base_windows,
            "max_windows_per_item": max_windows,
            "confidence_window_gain": max(0.0, float(config.get("confidence_window_gain", 3.0))),
            "semantic_window_gain": max(0.0, float(config.get("semantic_window_gain", 3.0))),
            "item_retrieval_lookahead": max(1, int(config.get("item_retrieval_lookahead", 6))),
            "item_retrieval_semantic_weight": max(0.0, min(1.0, float(config.get("item_retrieval_semantic_weight", 0.72)))),
            "modality_target_confidence": max(0.1, min(1.0, float(config.get("modality_target_confidence", 0.70)))),
            "observation_salience": max(0.1, min(1.0, float(config.get("observation_salience", 0.82)))),
            "cooldown_seconds": max(1.0, float(config.get("cooldown_seconds", 8.0))),
            "repeat_sources": bool(config.get("repeat_sources", True)),
        }

    def _build_brain_source_stream_locked(self, spec: dict[str, Any]) -> Iterator[tuple[str, torch.Tensor]]:
        source_type = str(spec.get("source_type", "auto"))
        loader = StreamingCorpusLoader(
            source=str(spec.get("source", "")),
            source_type=source_type,
            text_field=str(spec.get("text_field", "text")),
            hf_config=spec.get("hf_config"),
        )
        return labeled_pattern_stream(
            loader.char_stream(),
            self._encoder,
            self._trainer.config.window_size,
            learn_chunking=True,
        )

    def _build_sensory_stream_locked(self, spec: dict[str, Any]) -> Iterator[SensoryEpisode]:
        return build_sensory_stream(
            spec,
            visual_dim=int(getattr(self._trainer.config, "cross_modal_dim_visual", 64)),
            audio_dim=int(getattr(self._trainer.config, "cross_modal_dim_audio", 64)),
            device=self._trainer.model.device,
        )

    def _close_brain_sources_locked(self) -> None:
        for runtime in self._brain_source_runtimes:
            close = getattr(runtime.stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    continue
        self._brain_source_runtimes = []

    def _close_sensory_sources_locked(self) -> None:
        for runtime in self._sensory_source_runtimes:
            close = getattr(runtime.stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    continue
        self._sensory_source_runtimes = []

    def _curriculum_gap_terms_locked(self, limit: int = 4) -> list[str]:
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

    def _curriculum_hint_spikes(self, hint: str, dim: int, *, salt: str) -> torch.Tensor | None:
        cleaned = " ".join(str(hint).split()).strip().lower()
        if not cleaned or dim <= 0:
            return None
        tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]+", cleaned)
        if not tokens:
            return None
        vec = torch.zeros(dim, device=self._trainer.model.device)
        for token in tokens:
            grams = [token]
            if len(token) >= 3:
                grams.extend(token[i:i + 3] for i in range(len(token) - 2))
            for gram in grams:
                digest = hashlib.sha256(f"{salt}:{gram}".encode("utf-8")).digest()
                idx_a = int.from_bytes(digest[:4], byteorder="little") % dim
                idx_b = int.from_bytes(digest[4:8], byteorder="little") % dim
                vec[idx_a] += 1.0
                vec[idx_b] += 0.5
        total = vec.sum()
        if float(total.item()) <= 0.0:
            return None
        return vec / total

    def _run_curriculum_injection_locked(self) -> int:
        """Inject LLM-guided curriculum and synthetic sensory hints.

        The current Terminus runtime uses a cheap local text stream for steady
        background learning and relies on curriculum generation for targeted,
        higher-value episodes. Curriculum can now also provide synthetic visual
        and audio hint channels so the live runtime retains multimodal support
        without depending on narrow digit datasets.
        """
        curriculum = self._brain_config.get("curriculum")
        if (
            self._thought_loop is None
            or not hasattr(self, "_geometric_curiosity")
            or self._trainer is None
            or not curriculum
            or not curriculum.get("enabled")
        ):
            return 0

        current_tokens = int(self._trainer.token_count)
        trigger_interval = int(curriculum.get("trigger_interval_tokens", 2048))
        cooldown = float(curriculum.get("cooldown_seconds", 30.0))
        now = time.time()
        if current_tokens - self._last_curriculum_injection_token_count < trigger_interval:
            return 0
        if (now - self._last_curriculum_injection_time) < cooldown:
            return 0

        gap_terms = self._curriculum_gap_terms_locked(limit=int(curriculum.get("topics_per_cycle", 3)))
        if not gap_terms:
            return 0

        try:
            from hecsn.cortex.curriculum import CurriculumGenerator

            curriculum_gen = CurriculumGenerator()
            segments = curriculum_gen.generate(
                gap_terms,
                max_segments=int(curriculum.get("topics_per_cycle", 3)),
            )
            curriculum_gen.close()
            if not segments:
                return 0

            extra = 0
            dim_visual = int(getattr(self._trainer.config, "cross_modal_dim_visual", 64))
            dim_audio = int(getattr(self._trainer.config, "cross_modal_dim_audio", 64))
            for seg in segments:
                text = " ".join(str(getattr(seg, "text", "")).split()).strip()
                if not text:
                    continue
                visual_spikes = self._curriculum_hint_spikes(
                    getattr(seg, "visual_hint", "") or text,
                    dim_visual,
                    salt="visual",
                )
                audio_spikes = self._curriculum_hint_spikes(
                    getattr(seg, "audio_hint", "") or text,
                    dim_audio,
                    salt="audio",
                )
                for raw_window, pattern in self._encoder.iter_char_patterns(text, self._trainer.config.window_size):
                    metrics = self._trainer.train_step(
                        pattern,
                        raw_window=raw_window,
                        visual_spikes=visual_spikes,
                        audio_spikes=audio_spikes,
                    )
                    if metrics:
                        if metrics.get("cross_modal_visual_accepted"):
                            self._hint_visual_accepted += 1
                        if metrics.get("cross_modal_audio_accepted"):
                            self._hint_audio_accepted += 1
                    extra += 1

            if extra > 0:
                self._hint_sensory_episodes_completed += len(segments)
                self._mark_mutated()
                self._last_curriculum_injection_time = now
                self._last_curriculum_injection_token_count = int(self._trainer.token_count)
            return extra
        except Exception:
            return 0

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
            phrases.extend(self._curriculum_gap_terms_locked(limit=max(4, limit // 2)))

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

    def _pull_next_sensory_episode_locked(
        self,
        runtime: _SensorySourceRuntime,
        *,
        repeat_sources: bool,
    ) -> SensoryEpisode | None:
        restarted = False
        while True:
            try:
                runtime.exhausted = False
                return next(runtime.stream)
            except StopIteration:
                if not repeat_sources or restarted:
                    runtime.exhausted = True
                    return None
                runtime.cycles_completed += 1
                runtime.stream = self._build_sensory_stream_locked(runtime.spec)
                runtime.exhausted = False
                restarted = True
            except Exception as exc:
                runtime.exhausted = True
                self._real_sensory_last_error = str(exc)
                return None

    def _next_sensory_episode_locked(
        self,
        runtime: _SensorySourceRuntime,
        *,
        repeat_sources: bool,
        focus_terms: Sequence[str],
    ) -> SensoryEpisode | None:
        lookahead, semantic_weight = self._sensory_item_retrieval_config_locked()
        while len(runtime.buffered_episodes) < lookahead:
            episode = self._pull_next_sensory_episode_locked(runtime, repeat_sources=repeat_sources)
            if episode is None:
                break
            runtime.buffered_episodes.append(episode)
        if not runtime.buffered_episodes:
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
        topics.extend(list(self._last_sensory_focus_terms)[:2])
        topics.extend(salient_query_terms(text)[:4])
        seen: set[str] = set()
        deduped_topics: list[str] = []
        for topic in topics:
            cleaned = " ".join(str(topic).split()).strip()
            lowered = cleaned.lower()
            if not cleaned or lowered in seen:
                continue
            seen.add(lowered)
            deduped_topics.append(cleaned)
            if len(deduped_topics) >= 6:
                break
        modality_label = "image-grounded" if episode.visual_spikes is not None and episode.audio_spikes is None else (
            "audio-grounded" if episode.audio_spikes is not None and episode.visual_spikes is None else "multisensory"
        )
        content = f"{modality_label} episode from {runtime.name}: {text}"
        self._thought_loop.inject_observation(
            content=content,
            topics=deduped_topics,
            salience=salience,
        )
        return {"topics": deduped_topics, "salience": salience, "content": content}

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
                }
            )
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

    def _rebuild_brain_sources_locked(self) -> None:
        self._close_brain_sources_locked()
        self._close_sensory_sources_locked()
        self._brain_source_runtimes = [
            _BrainSourceRuntime(spec=deepcopy(spec), stream=self._build_brain_source_stream_locked(spec))
            for spec in self._brain_config.get("source_bank", [])
        ]
        sensory_config = self._brain_config.get("sensory") or {}
        self._sensory_source_runtimes = [
            _SensorySourceRuntime(spec=deepcopy(spec), stream=self._build_sensory_stream_locked(spec))
            for spec in sensory_config.get("source_bank", [])
        ]
        self._brain_source_index = 0
        self._sensory_source_index = 0
        self._brain_tick_count = 0
        self._brain_background_tokens = 0
        self._brain_last_tick_completed_at = None
        self._brain_last_tick_duration_ms = None
        self._brain_last_tick_token_delta = 0
        self._brain_last_work_at = None
        self._hint_sensory_episodes_completed = 0
        self._hint_visual_accepted = 0
        self._hint_audio_accepted = 0
        self._real_sensory_episodes_completed = 0
        self._real_visual_accepted = 0
        self._real_audio_accepted = 0
        self._last_real_sensory_episode_time = 0.0
        self._last_real_sensory_episode_token_count = int(self._trainer.token_count)
        self._real_sensory_last_error = None
        self._last_sensory_focus_terms = ()
        self._sensory_preview_history.clear()

    def _request_brain_stop(self, *, reason: str | None = None) -> Thread | None:
        with self._lock:
            return self._request_brain_stop_locked(reason=reason)

    def _join_brain_thread(self, thread: Thread | None, *, timeout: float = 15.0) -> None:
        if thread is None:
            return
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise RuntimeError("Terminus runtime did not stop cleanly")

    def _request_brain_stop_locked(self, *, reason: str | None = None) -> Thread | None:
        thread = self._brain_thread if self._brain_thread is not None and self._brain_thread.is_alive() else None
        stop_event = self._brain_stop_event
        if stop_event is not None:
            stop_event.set()
        self._brain_running = False
        self._brain_running_since = None
        if reason is not None:
            self._record_brain_event_locked({
                "type": "stopped",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
            })
        self._brain_thread = None
        self._brain_stop_event = None
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
                tick_start = time.perf_counter()

                # Phase 0: snapshot config under lock (fast)
                with self._lock:
                    if not self._brain_source_runtimes:
                        runtimes = None
                    else:
                        runtimes = list(self._brain_source_runtimes)
                        src_index = self._brain_source_index
                        tick_tokens = int(self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS))
                        repeat = bool(self._brain_config.get("repeat_sources", True))
                        encoder_ref = self._encoder
                        window_size = self._trainer.config.window_size

                if runtimes is None:
                    with self._lock:
                        result = self._brain_tick_idle_locked(tick_start)
                    did_work = result.get("did_work", False) if isinstance(result, dict) else False
                    actual_sleep = max(0.001, sleep_interval * 0.1) if did_work else max(0.05, sleep_interval)
                    time.sleep(actual_sleep)
                    continue

                # Phase 1: collect tokens OUTSIDE the lock (network I/O safe)
                chunk, collect_meta = self._collect_chunk_unlocked(
                    runtimes, src_index, tick_tokens, repeat,
                    encoder_ref, window_size, stop_event,
                )

                if stop_event is not None and stop_event.is_set():
                    break

                if chunk is None:
                    with self._lock:
                        result = self._brain_tick_idle_locked(tick_start)
                    did_work = result.get("did_work", False) if isinstance(result, dict) else False
                    actual_sleep = max(0.001, sleep_interval * 0.1) if did_work else max(0.05, sleep_interval)
                    time.sleep(actual_sleep)
                    continue

                # Commit collection metadata under lock (fast)
                with self._lock:
                    if collect_meta is not None:
                        rt = collect_meta["runtime"]
                        rt.cycles_completed = collect_meta["cycles"]
                        rt.exhausted = collect_meta["exhausted"]
                        if collect_meta.get("new_stream") is not None:
                            rt.stream = collect_meta["new_stream"]

                # Phase 2: train in sub-batches, releasing lock between each
                total_trained = 0
                last_metrics = None
                for i in range(0, len(chunk), _SUB_BATCH):
                    if stop_event is not None and stop_event.is_set():
                        break
                    sub = chunk[i : i + _SUB_BATCH]
                    with self._lock:
                        for raw_window, pattern in sub:
                            last_metrics = self._trainer.train_step(pattern, raw_window=raw_window)
                        # Observe concepts once per sub-batch (last token only)
                        # to avoid O(n_concepts) linear scan per token.
                        if sub:
                            self._observe_runtime_concepts_locked(raw_window=sub[-1][0], metrics=last_metrics)
                        total_trained += len(sub)
                        self._mark_mutated()
                    time.sleep(_YIELD_SECONDS)

                # Phase 3: finalize tick counters under lock
                source_info = {
                    "runtime": collect_meta["runtime"],
                    "idx": collect_meta["idx"],
                    "source_count": collect_meta["source_count"],
                } if collect_meta else None
                with self._lock:
                    result = self._finalize_tick_locked(
                        tick_start, source_info, total_trained, last_metrics,
                    )

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

    def _collect_chunk_unlocked(
        self,
        runtimes: list,
        src_index: int,
        tick_tokens: int,
        repeat: bool,
        encoder_ref: Any,
        window_size: int,
        stop_event: Event | None,
    ) -> tuple[list[tuple[str, "torch.Tensor"]] | None, dict[str, Any] | None]:
        """Collect tokens from sources WITHOUT holding self._lock.

        Network I/O (HuggingFace streaming) happens here, so this must not
        hold the lock — otherwise SSE/API starve for seconds during retries.
        All runtime field mutations are deferred: the caller commits them
        under lock using the returned ``collect_meta`` dict.
        """
        source_count = len(runtimes)
        for offset in range(source_count):
            if stop_event is not None and stop_event.is_set():
                return None, None
            idx = (src_index + offset) % source_count
            runtime = runtimes[idx]
            chunk: list[tuple[str, torch.Tensor]] = []
            cycles = runtime.cycles_completed
            exhausted = runtime.exhausted
            new_stream = None
            while len(chunk) < tick_tokens:
                if stop_event is not None and stop_event.is_set():
                    return None, None
                try:
                    chunk.append(next(runtime.stream))
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
                            chunk.append(next(runtime.stream))
                        except StopIteration:
                            exhausted = True
                            break
                    else:
                        exhausted = True
                        break
                except Exception:
                    # Stream closed by concurrent stop — bail out
                    return None, None
            if not chunk:
                continue
            meta = {
                "runtime": runtime,
                "idx": idx,
                "source_count": source_count,
                "cycles": cycles,
                "exhausted": exhausted,
                "new_stream": new_stream,
            }
            return chunk, meta
        return None, None

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
        return labeled_pattern_stream(
            loader.char_stream(),
            encoder,
            window_size,
            learn_chunking=True,
        )

    def _finalize_tick_locked(
        self,
        tick_started: float,
        source_info: dict[str, Any] | None,
        total_trained: int,
        last_metrics: Any,
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
                "last_metrics": last_metrics,
            }
        else:
            source_summary = {"did_work": False, "reason": "no_tokens"}

        autonomy_summary = self._run_brain_autonomy_locked()
        cortex_work = bool(source_summary.get("did_work")) or autonomy_summary is not None

        # Inject SNN neuromodulator state into cortex drives
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

            # Inject a content observation so Cortex has real data to think about
            if source_info is not None and total_trained > 0:
                try:
                    src_rt = source_info.get("runtime")
                    src_label = getattr(src_rt, "name", "unknown") if src_rt is not None else "unknown"
                    # Get recent concept labels from concept store
                    recent_concepts: list[str] = []
                    snap = self._concept_store.snapshot(limit=5)
                    for c in snap.get("top_concepts", [])[:5]:
                        label = c.get("label", "")
                        if label:
                            recent_concepts.append(label)
                    obs_text = f"SNN processed {total_trained} tokens from {src_label}."
                    if recent_concepts:
                        obs_text += f" Recent concepts: {', '.join(recent_concepts)}."
                    self._thought_loop.inject_observation(
                        content=obs_text,
                        topics=[src_label] + recent_concepts[:3],
                        salience=0.6,
                    )
                except Exception:
                    pass

        # LLM-guided curriculum + real sensory grounding
        curriculum_extra = self._run_curriculum_injection_locked()
        sensory_summary = self._run_real_sensory_episode_locked()
        total_trained += curriculum_extra
        token_count_after = int(self._trainer.token_count)
        multimodal_summary = self._multimodal_runtime_summary_locked() if (curriculum_extra > 0 or sensory_summary is not None) else None
        did_work = bool(source_summary.get("did_work")) or autonomy_summary is not None or curriculum_extra > 0 or sensory_summary is not None

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

    def _brain_tick_idle_locked(self, tick_started: float) -> dict[str, Any]:
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

        autonomy_summary = self._run_brain_autonomy_locked()
        curriculum_extra = self._run_curriculum_injection_locked()
        sensory_summary = self._run_real_sensory_episode_locked()
        multimodal_summary = self._multimodal_runtime_summary_locked() if (curriculum_extra > 0 or sensory_summary is not None) else None
        did_work = autonomy_summary is not None or curriculum_extra > 0 or sensory_summary is not None
        completed_at = datetime.now(timezone.utc).isoformat()
        summary = {
            "type": "tick",
            "did_work": did_work,
            "timestamp": completed_at,
            "source": {"did_work": False, "reason": "sources_exhausted"},
            "multimodal": multimodal_summary,
            "autonomy": autonomy_summary,
            "tick_duration_ms": float((time.perf_counter() - tick_started) * 1000.0),
            "token_delta": int(curriculum_extra + (0 if sensory_summary is None else sensory_summary.get("steps_trained", 0))),
        }
        self._brain_last_tick_completed_at = completed_at
        self._brain_last_tick_duration_ms = float(summary["tick_duration_ms"])
        self._brain_last_tick_token_delta = int(summary["token_delta"])
        if did_work:
            self._brain_last_work_at = completed_at
        self._record_brain_event_locked(summary)
        return summary

    def _brain_tick_locked(self) -> dict[str, Any]:
        tick_started = time.perf_counter()
        token_count_before = int(self._trainer.token_count)
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

        source_summary = self._consume_next_source_locked()
        autonomy_summary = self._run_brain_autonomy_locked()
        curriculum_extra = self._run_curriculum_injection_locked()
        sensory_summary = self._run_real_sensory_episode_locked()
        multimodal_summary = self._multimodal_runtime_summary_locked() if (curriculum_extra > 0 or sensory_summary is not None) else None
        did_work = bool(source_summary.get("did_work")) or autonomy_summary is not None or curriculum_extra > 0 or sensory_summary is not None
        token_count_after = int(self._trainer.token_count)

        # --- Inject SNN neuromodulator state into cortex drives ---
        if did_work and self._thought_loop is not None:
            try:
                surprise = self._trainer.model.surprise
                self._thought_loop.inject_surprise(
                    dopamine=float(surprise.dopamine),
                    serotonin=float(surprise.serotonin),
                    norepinephrine=float(surprise.norepinephrine),
                    acetylcholine=float(surprise.acetylcholine),
                )
            except Exception:
                pass  # cortex is non-critical

            # Inject content observations so Cortex has real data
            if bool(source_summary.get("did_work")):
                try:
                    src_label = str(source_summary.get("source_name", "unknown"))
                    tok_count = int(source_summary.get("tokens_trained", 0))
                    recent_concepts: list[str] = []
                    snap = self._concept_store.snapshot(limit=5)
                    for c in snap.get("top_concepts", [])[:5]:
                        label = c.get("label", "")
                        if label:
                            recent_concepts.append(label)
                    obs_text = f"SNN processed {tok_count} tokens from {src_label}."
                    if recent_concepts:
                        obs_text += f" Recent concepts: {', '.join(recent_concepts)}."
                    self._thought_loop.inject_observation(
                        content=obs_text,
                        topics=[src_label] + recent_concepts[:3],
                        salience=0.6,
                    )
                except Exception:
                    pass

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

    def _consume_next_source_locked(self) -> dict[str, Any]:
        if not self._brain_source_runtimes:
            return {"did_work": False, "reason": "no_sources"}
        tick_tokens = int(self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS))
        source_count = len(self._brain_source_runtimes)
        for offset in range(source_count):
            idx = (self._brain_source_index + offset) % source_count
            runtime = self._brain_source_runtimes[idx]
            chunk: list[tuple[str, torch.Tensor]] = []
            while len(chunk) < tick_tokens:
                try:
                    chunk.append(next(runtime.stream))
                except StopIteration:
                    if bool(self._brain_config.get("repeat_sources", True)):
                        runtime.cycles_completed += 1
                        runtime.stream = self._build_brain_source_stream_locked(runtime.spec)
                        runtime.exhausted = False
                        try:
                            chunk.append(next(runtime.stream))
                        except StopIteration:
                            runtime.exhausted = True
                            break
                    else:
                        runtime.exhausted = True
                        break
            if not chunk:
                continue
            last_metrics = None
            for raw_window, pattern in chunk:
                last_metrics = self._trainer.train_step(pattern, raw_window=raw_window)
                self._observe_runtime_concepts_locked(raw_window=raw_window, metrics=last_metrics)
            token_count = len(chunk)
            runtime.tokens_processed += token_count
            runtime.tick_visits += 1
            runtime.last_tokens_trained = int(token_count)
            runtime.last_activity_at = datetime.now(timezone.utc).isoformat()
            self._brain_background_tokens += token_count
            self._brain_tick_count += 1
            self._brain_source_index = (idx + 1) % source_count
            self._mark_mutated()
            return {
                "did_work": True,
                "source_name": runtime.name,
                "source_type": runtime.source_type,
                "source_index": int(idx),
                "tokens_trained": int(token_count),
                "cycles_completed": int(runtime.cycles_completed),
                "exhausted": bool(runtime.exhausted),
                "last_metrics": last_metrics,
            }
        return {"did_work": False, "reason": "sources_exhausted"}

    def _run_brain_autonomy_locked(self) -> dict[str, Any] | None:
        autonomy = self._brain_config.get("autonomy")
        if not autonomy or not bool(autonomy.get("enabled", False)):
            return None
        token_delta = int(self._trainer.token_count) - int(self._brain_last_acquisition_token_count)
        trigger_interval = int(autonomy.get("trigger_interval_tokens", DEFAULT_AUTONOMY_TRIGGER_INTERVAL_TOKENS))

        # Curiosity-based trigger: allow early acquisition when gap score exceeds threshold
        curiosity_gap_threshold = float(autonomy.get("curiosity_gap_threshold", 0.0))
        curiosity_cooldown = int(autonomy.get("curiosity_cooldown_tokens", trigger_interval // 2))
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
        focus_plan = self._autonomy_focus_plan_locked()
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
            acquisition_tokens=int(autonomy.get("acquisition_tokens", 512)),
            acquisition_slots=int(autonomy.get("acquisition_slots", 1)),
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
        self._brain_last_acquisition_token_count = int(self._trainer.token_count)
        if curriculum_before != autonomy.get("provider_curriculum"):
            self._mark_mutated()
        if int(result.get("tokens_trained_total", 0)) > 0:
            self._mark_mutated()
        summary = {
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "trigger_reason": trigger_reason,
            "policy": str(result.get("policy", autonomy.get("policy", "active"))),
            "tokens_trained_total": int(result.get("tokens_trained_total", 0)),
            "acquired_sources": list(result.get("acquired_sources", [])),
            "stopped_early": bool(result.get("stopped_early", False)),
            "final_mean_candidate_gap": result.get("final_mean_candidate_gap"),
            "final_max_candidate_gap": result.get("final_max_candidate_gap"),
            "stop_reason": result.get("stop_reason"),
            "focus_plan": deepcopy(result.get("semantic_plan")),
            "recent_query_gap_count": int(len(self._brain_recent_query_gaps)),
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
        autonomy = self._brain_config.get("autonomy")
        next_source_name = None
        if self._brain_source_runtimes:
            next_source_name = self._brain_source_runtimes[self._brain_source_index % len(self._brain_source_runtimes)].name
        exhausted_source_count = sum(1 for runtime in self._brain_source_runtimes if runtime.exhausted)
        autonomy_tokens_until_trigger = None
        autonomy_trigger_ready = None
        autonomy_candidate_names = None
        autonomy_focus_plan = None
        autonomy_provider_curriculum = None
        curriculum = self._brain_config.get("curriculum")
        sensory = self._brain_config.get("sensory")
        curriculum_tokens_until_trigger = None
        curriculum_trigger_ready = None
        sensory_tokens_until_trigger = None
        sensory_trigger_ready = None
        if autonomy is not None:
            trigger_interval = int(autonomy.get("trigger_interval_tokens", DEFAULT_AUTONOMY_TRIGGER_INTERVAL_TOKENS))
            token_delta = int(self._trainer.token_count) - int(self._brain_last_acquisition_token_count)
            autonomy_tokens_until_trigger = int(max(0, trigger_interval - token_delta))
            autonomy_trigger_ready = bool(token_delta >= trigger_interval)
            autonomy_candidate_names = [
                str(item.get("name", "candidate"))
                for item in list(autonomy.get("candidate_bank", []))
            ]
            autonomy_focus_plan = self._autonomy_focus_plan_locked()
            autonomy_provider_curriculum = self._provider_curriculum_snapshot_locked(autonomy, autonomy_focus_plan)
        if curriculum is not None:
            curriculum_trigger_interval = int(curriculum.get("trigger_interval_tokens", 1024))
            curriculum_token_delta = int(self._trainer.token_count) - int(self._last_curriculum_injection_token_count)
            curriculum_tokens_until_trigger = int(max(0, curriculum_trigger_interval - curriculum_token_delta))
            curriculum_trigger_ready = bool(curriculum_token_delta >= curriculum_trigger_interval)
        if sensory is not None:
            sensory_trigger_interval = int(sensory.get("episode_interval_tokens", 2048))
            sensory_token_delta = int(self._trainer.token_count) - int(self._last_real_sensory_episode_token_count)
            sensory_tokens_until_trigger = int(max(0, sensory_trigger_interval - sensory_token_delta))
            sensory_trigger_ready = bool(sensory_token_delta >= sensory_trigger_interval)
        return {
            "configured": bool(self._brain_config.get("source_bank")),
            "running": bool(
                self._brain_running
                and self._brain_thread is not None
                and self._brain_thread.is_alive()
            ),
            "running_since": self._brain_running_since,
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
                    "share_of_background_tokens": float(
                        0.0
                        if self._brain_background_tokens <= 0
                        else float(runtime.tokens_processed) / float(self._brain_background_tokens)
                    ),
                }
                for runtime in self._brain_source_runtimes
            ],
            "huggingface": self._huggingface_runtime_summary_locked(),
            "curriculum": None
            if curriculum is None
            else {
                "enabled": bool(curriculum.get("enabled", False)),
                "topics_per_cycle": int(curriculum.get("topics_per_cycle", 3)),
                "trigger_interval_tokens": int(curriculum.get("trigger_interval_tokens", 1024)),
                "cooldown_seconds": float(curriculum.get("cooldown_seconds", 30.0)),
                "tokens_until_trigger": curriculum_tokens_until_trigger,
                "trigger_ready": curriculum_trigger_ready,
                "last_injection_at": None if self._last_curriculum_injection_time <= 0 else self._last_curriculum_injection_time,
                "last_injection_token_count": int(self._last_curriculum_injection_token_count),
            },
            "sensory": None
            if sensory is None
            else {
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
                "tokens_until_trigger": sensory_tokens_until_trigger,
                "trigger_ready": sensory_trigger_ready,
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
            },
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
                "tokens_until_trigger": autonomy_tokens_until_trigger,
                "trigger_ready": autonomy_trigger_ready,
                "recent_query_gaps": [deepcopy(item) for item in list(self._brain_recent_query_gaps)],
                "focus_plan": deepcopy(autonomy_focus_plan),
                "provider_curriculum": deepcopy(autonomy_provider_curriculum),
                "last_acquisition_token_count": int(self._brain_last_acquisition_token_count),
                "last_acquisition_summary": deepcopy(self._brain_last_acquisition_summary),
                "geometric_curiosity": deepcopy(self._geometric_curiosity.summary()),
            },
            "multimodal": self._multimodal_runtime_summary_locked(),
            "cortex": self._thought_loop.snapshot() if self._thought_loop is not None else {"enabled": False},
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
            "curriculum": deepcopy(self._brain_config.get("curriculum")),
            "sensory": deepcopy(self._brain_config.get("sensory")),
            "recent_query_gaps": [deepcopy(item) for item in list(self._brain_recent_query_gaps)],
            "geometric_curiosity": self._geometric_curiosity.state_dict(),
        }

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
